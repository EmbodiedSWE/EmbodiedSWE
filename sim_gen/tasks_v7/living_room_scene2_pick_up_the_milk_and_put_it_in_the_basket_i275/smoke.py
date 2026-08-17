"""Smoke / rubric-REJECTION battery for DropChuteScene (sim_gen task
`living_room_scene2_pick_up_the_milk_and_put_it_in_the_basket_i275`) — NullRobot,
teleported probe states, RECORDED.

This is NOT a solution (solve.py — lay the carton on the shelf, push it through the
one-way flap — is the acceptance evidence). Every teleport here is instrumentation
that CONSTRUCTS a wrong (or partial) outcome as a judged state and asserts the rubric
REJECTS it, plus physics probes that prove the mechanism is real (the roof genuinely
bars the seed's top-drop, the flap genuinely opens inward / self-closes / locks
outward, the below-the-aperture gate genuinely refuses a carton in the doorway, and a
delivered carton genuinely cannot be shoved back out). Two probes DO construct the
genuine end state on purpose — the acceptance construct (12) and the decoy-removed
flip (15) — and the retention probe (13) intentionally KEEPS it; every other judged
point must stay success()=False and a final audit asserts exactly that.

  1-2.  settle/no-NaN     — reset settles finite; flap SHUT (readback |angle| < 3
                            deg), milk outside on the floor; score ~0, no success;
  3-4.  randomization     — READBACK over 8 seeded resets: bin xy + free yaw vary;
                            milk and juice poses vary;
  5.   null policy        — 300 idle steps -> score ~0, no success (both cartons
                            spawn standing well beyond the shelf);
  6.   roof (seed means)  — the seed's strategy, physically denied: the carton
                            dropped from ABOVE the bin lands ON the roof (readback
                            z ~ roof height), never enters — no credit, no success;
  7.   staged only        — the carton lying on the shelf: staging credit only
                            (0.20), no entry, no success;
  8.   doorstep           — the carton nosed up against the CLOSED flap from
                            outside, settled: the flap holds shut (< 5 deg), still
                            staging credit only, no success;
  9.   flap self-closes   — +0.08 N*m inward torque swings the flap wide open
                            (>= 40 deg, non-vacuous), released -> it falls back
                            shut (< 10 deg) ON ITS OWN;
  10.  flap is one-way    — the SAME 0.08 N*m applied outward: the flap stays at
                            its -1 deg stop (>= -3 deg) — nothing swings out;
  11.  in the doorway     — the carton posed mid-transit in the slot (entered box,
                            above the sill), judged immediately: entered credit at
                            most (<= 0.55), NOT inside (the inside z window sits
                            BELOW the aperture), no success;
  12.  acceptance         — the carton lying settled on the bin floor, flap shut,
                            decoy away -> success TRUE (the rubric accepts exactly
                            the delivered end state);
  13.  retention          — 4 N of outward shove on the delivered carton for 2 s:
                            it MOVES (>= 8 mm, non-vacuous) but the sealed front
                            wall + one-way flap keep it in; it re-settles inside ->
                            still success TRUE;
  14.  decoy in doorway   — the RED juice carton posed in the bin's doorway volume:
                            success flips FALSE (decoy clause), milk untouched;
  15.  decoy removed      — juice back on the floor: success returns TRUE (14's
                            rejection was the decoy clause and nothing else);
  16.  settle gate        — the delivered carton kicked and judged immediately: NOT
                            success (must be at rest);
  17.  wrong object       — fresh reset, the JUICE carton placed inside while the
                            milk stands on the floor: rejected twice over (milk
                            not inside AND the decoy clause), score ~0;
  18.  rejection audit    — success() was never True at any judged point EXCEPT
                            the constructed acceptance probes (12, 13, 15);
  19.  final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.<task>.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version and silently rejects RTX -> the
# annotator returns EMPTY frames. Disable the driver check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math
import os
import threading

import numpy as np
import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from .scene import (  # noqa: F401
        BIN_H, BIN_HX, FLOOR_T, MILK_H, MILK_W, SILL, _qapply, _qinv, _qmul, _qy, _qz,
    )
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import (  # noqa: F401
        BIN_H, BIN_HX, FLOOR_T, MILK_H, MILK_W, SILL, _qapply, _qinv, _qmul, _qy, _qz,
    )
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

LIE_Z = SILL + MILK_W / 2 + 0.004  # lying carton CoM at shelf/sill height
FLOOR_LIE_Z = FLOOR_T + MILK_W / 2 + 0.004  # lying carton CoM on the bin floor


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.drop_chute")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.35, -1.15, 0.80)) + o),
                                tuple(np.array((0.10, 0.00, 0.18)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action)
            if (annot is not None and step_i % args.record_every == 0
                    and len(frames) < args.max_frames):
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    ever_bad_success = [False]

    def judge() -> tuple[float, bool]:
        """Judge a REJECTION probe: success here is a rubric failure."""
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_bad_success[0] = ever_bad_success[0] or ok
        return s, ok

    def judge_accept() -> tuple[float, bool]:
        """Judge an ACCEPTANCE construct: success here is expected and allowed."""
        return float(scene.score()[0]), bool(scene.success()[0])

    def flap() -> float:
        return float(scene.flap_deg()[0])

    def report(tag: str, s: float, ok: bool) -> None:
        ml = scene.bin_local(scene.milk.data.root_pos_w)[0]
        ju = scene.bin_local(scene.juice.data.root_pos_w)[0]
        print(f"[smoke] {tag:18s} | flap={flap():+6.2f}deg "
              f"milk_bin=({float(ml[0]):+.3f},{float(ml[1]):+.3f},{float(ml[2]):+.3f}) "
              f"juice_bin=({float(ju[0]):+.3f},{float(ju[1]):+.3f}) "
              f"shelf={bool(scene.milk_on_shelf()[0])} "
              f"ent={bool(scene.milk_entered()[0])} "
              f"in={bool(scene.milk_inside()[0])} "
              f"closed={bool(scene.flap_closed()[0])} "
              f"decoy_out={bool(scene.decoy_out()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def bin_pose(local, extra_quat=None):
        q_bin = scene.bin.data.root_quat_w
        pos = scene.bin.data.root_pos_w + _qapply(
            q_bin, torch.tensor(local, device=device).expand(n, 3))
        q = q_bin if extra_quat is None else _qmul(q_bin, extra_quat)
        return pos, q

    def teleport(body, pos, quat, vel=None, ang=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3:7] = quat
        if vel is not None:
            st[:, 7:10] = torch.tensor(vel, device=device)
        if ang is not None:
            st[:, 10:13] = torch.tensor(ang, device=device)
        body.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def settle_all(max_steps: int = 600) -> None:
        for _ in range(max_steps // 30):
            step(30)
            if bool(scene.settled(scene.milk)[0]) and bool(scene.settled(scene.juice)[0]) \
                    and bool(scene.settled(scene.flap)[0]):
                break

    def clear_forces() -> None:
        scene.milk.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
        scene.flap.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def all_finite() -> bool:
        bodies = [scene.bin, scene.flap, scene.milk, scene.juice]
        return all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)

    up_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4)
    half_pi = torch.full((n,), math.pi / 2, device=device)
    lie_quat = _qy(half_pi)  # carton long axis along bin x (composed with bin quat)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    settle_all(480)
    s, ok = judge()
    report("reset", s, ok)
    check("settle: all states finite, flap SHUT (readback "
          f"angle={flap():+.2f}deg, |angle| < 3), milk outside on the floor",
          all_finite() and abs(flap()) < 3.0 and not bool(scene.milk_entered()[0])
          and not bool(scene.milk_inside()[0]))
    check(f"settle: score ~0 and no success on a fresh reset (score={s:.3f})",
          s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(20)
        bp = (scene.bin.data.root_pos_w - scene.env_origins)[0]
        ex = _qapply(scene.bin.data.root_quat_w,
                     torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))[0]
        yaw = math.atan2(float(ex[1]), float(ex[0]))
        ml = (scene.milk.data.root_pos_w - scene.env_origins)[0]
        ju = (scene.juice.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(bp[0]), float(bp[1]), yaw, float(ml[0]), float(ml[1]),
                      float(ju[0]), float(ju[1])))
    arr = np.array(reads)
    print("[smoke] randomization readback (bin_x, bin_y, bin_yaw, milk_x, milk_y, "
          f"ju_x, ju_y):\n{arr.round(3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: bin pose varies across seeded resets (readback: "
          f"dx={spread[0]:.3f} dy={spread[1]:.3f} dyaw={spread[2]:.2f} rad)",
          spread[0] > 0.02 and spread[1] > 0.02 and spread[2] > 0.8)
    check("randomization: milk and juice poses vary (readback: "
          f"dmilk={max(spread[3], spread[4]):.3f} djuice={max(spread[5], spread[6]):.3f})",
          max(spread[3], spread[4]) > 0.10 and max(spread[5], spread[6]) > 0.10)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(300)
    s, ok = judge()
    report("null-policy", s, ok)
    check(f"null policy: score ~0 and no success after 300 idle steps (score={s:.3f})",
          s <= 0.02 and not ok)

    # =========================== 6. the roof denies the seed's top-drop =====================
    # The seed's whole delivery — carry the item OVER the container and release it
    # from above — attempted as physics: the carton dropped over the bin's center.
    settle_all(300)
    pos, q = bin_pose((0.0, 0.0, BIN_H + MILK_W / 2 + 0.05), lie_quat)
    teleport(scene.milk, pos, q, settle_steps=0)
    settle_all(420)
    s, ok = judge()
    report("roof-drop", s, ok)
    ml = scene.bin_local(scene.milk.data.root_pos_w)[0]
    check("roof (the seed's means, denied): the carton dropped from ABOVE the bin "
          f"lands ON the roof (readback z={float(ml[2]) * 1000:.0f}mm >= "
          f"{(BIN_H - 0.02) * 1000:.0f}mm) — never enters, no credit, no success "
          f"(score={s:.3f})",
          float(ml[2]) >= BIN_H - 0.02 and not bool(scene.milk_entered()[0])
          and not bool(scene.milk_inside()[0]) and s <= 0.02 and not ok)

    # =========================== 7. staged only =============================================
    env.reset(seed=41)
    settle_all(300)
    pos, q = bin_pose((0.290, 0.0, LIE_Z), lie_quat)
    teleport(scene.milk, pos, q, settle_steps=150)
    s, ok = judge()
    report("staged-only", s, ok)
    check("staged only: the carton lying on the shelf earns staging credit only "
          f"(score={s:.3f} in [0.19,0.21]), no entry, no success",
          0.19 <= s <= 0.21 and not ok and not bool(scene.milk_entered()[0]))

    # =========================== 8. doorstep: the closed flap holds =========================
    pos, q = bin_pose((BIN_HX + MILK_H / 2 - 0.015, 0.0, LIE_Z), lie_quat)
    teleport(scene.milk, pos, q, settle_steps=180)
    s, ok = judge()
    report("doorstep", s, ok)
    check("doorstep: the carton nosed into the slot against the CLOSED flap, "
          f"settled — the flap holds shut (readback {flap():+.2f}deg < 5), staging "
          f"credit at most (score={s:.3f} <= 0.21), no success",
          abs(flap()) < 5.0 and s <= 0.21 and not ok
          and not bool(scene.milk_inside()[0]))

    # =========================== 9-10. the flap: self-closing and one-way ===================
    # Same-magnitude pair: +0.08 N*m INWARD swings it wide open (proves the probe
    # torque dominates the gravity bias), then released it falls shut on its own;
    # the SAME 0.08 N*m OUTWARD cannot move it past the -1 deg stop.
    env.reset(seed=51)
    settle_all(300)
    peak = flap()
    tq = torch.tensor([0.0, 0.08, 0.0], device=device).expand(n, 3)  # body y = hinge
    for _ in range(180):
        scene.flap.set_external_force_and_torque(zero_wrench, tq.view(n, 1, 3),
                                                 env_ids=all_ids)
        step(1)
        peak = max(peak, flap())
    clear_forces()
    settle_all(360)
    closed_back = flap()
    s, ok = judge()
    report("flap-cycle", s, ok)
    check("flap self-closes: +0.08 N*m inward opened it to "
          f"{peak:.1f}deg (>= 40, non-vacuous), released it fell back shut on its "
          f"own (readback {closed_back:+.2f}deg, |angle| < 10)",
          peak >= 40.0 and abs(closed_back) < 10.0)
    low = 0.0
    for _ in range(120):
        scene.flap.set_external_force_and_torque(zero_wrench, -tq.view(n, 1, 3),
                                                 env_ids=all_ids)
        step(1)
        low = min(low, flap())
    clear_forces()
    settle_all(240)
    s, ok = judge()
    report("flap-oneway", s, ok)
    check("flap is one-way: the SAME 0.08 N*m applied OUTWARD leaves it at the "
          f"stop (min angle {low:+.2f}deg >= -3) — nothing swings out",
          low >= -3.0)

    # =========================== 11. in the doorway is NOT inside ===========================
    # Mid-transit pose: entered box satisfied, but the carton sits AT slot height —
    # the inside z window lives BELOW the aperture, so transit never reads as inside.
    pos, q = bin_pose((0.150, 0.0, LIE_Z), lie_quat)
    teleport(scene.milk, pos, q, settle_steps=2)  # judge mid-transit
    s, ok = judge()
    ml = scene.bin_local(scene.milk.data.root_pos_w)[0]
    report("doorway", s, ok)
    check("in the doorway: the carton judged mid-transit in the slot (readback "
          f"z={float(ml[2]) * 1000:.0f}mm, above the sill) earns entered credit at "
          f"most (score={s:.3f} <= 0.551) and is NOT inside — no success",
          float(ml[2]) > SILL - 0.010 and s <= 0.55 + 1e-3 and not ok
          and not bool(scene.milk_inside()[0]))

    # =========================== 12. acceptance construct ===================================
    env.reset(seed=61)
    settle_all(300)
    pos, q = bin_pose((-0.030, 0.0, FLOOR_LIE_Z), lie_quat)
    teleport(scene.milk, pos, q, settle_steps=0)
    settle_all(480)
    s, ok = judge_accept()
    report("accept", s, ok)
    check("acceptance: the carton lying settled on the bin floor, flap shut "
          f"(readback {flap():+.2f}deg), decoy away -> success TRUE "
          f"(score={s:.3f})", ok and s >= 0.99 and abs(flap()) < 3.0)

    # =========================== 13. retention: it cannot be shoved out =====================
    x0 = float(scene.bin_local(scene.milk.data.root_pos_w)[0, 0])
    peak_x = x0
    for _ in range(240):
        f_bin = torch.tensor([4.0, 0.0, 0.0], device=device).expand(n, 3)
        f_world = _qapply(scene.bin.data.root_quat_w, f_bin)
        f_body = _qapply(_qinv(scene.milk.data.root_quat_w), f_world)
        scene.milk.set_external_force_and_torque(f_body.view(n, 1, 3), zero_wrench,
                                                 env_ids=all_ids)
        step(1)
        peak_x = max(peak_x, float(scene.bin_local(scene.milk.data.root_pos_w)[0, 0]))
    clear_forces()
    settle_all(360)
    s, ok = judge_accept()
    report("retention", s, ok)
    check("retention: 4 N of outward shove moved the delivered carton "
          f"{(peak_x - x0) * 1000:.0f}mm (>= 8, non-vacuous) but the sealed front "
          f"wall kept it in (peak x={peak_x * 1000:.0f}mm < "
          f"{(BIN_HX + 0.005) * 1000:.0f}mm); it re-settled inside -> success still "
          f"TRUE (score={s:.3f})",
          peak_x - x0 >= 0.008 and peak_x < BIN_HX + 0.005
          and bool(scene.milk_inside()[0]) and ok)

    # =========================== 14-15. decoy clause flips success ==========================
    pos, q = bin_pose((BIN_HX - 0.010, 0.0, LIE_Z), lie_quat)
    teleport(scene.juice, pos, q, settle_steps=2)  # judge posed in the doorway
    s, ok = judge()
    report("decoy-doorway", s, ok)
    check("decoy in the doorway: the RED juice carton posed in the bin's doorway "
          "volume flips success FALSE (decoy clause), milk untouched",
          not ok and not bool(scene.decoy_out()[0]) and bool(scene.milk_inside()[0]))
    pos, q = bin_pose((0.60, -0.45, MILK_H / 2 + 0.003), None)
    teleport(scene.juice, pos, up_quat, settle_steps=120)
    s, ok = judge_accept()
    report("decoy-removed", s, ok)
    check("decoy removed: juice back on the floor -> success returns TRUE (14's "
          "rejection was the decoy clause and nothing else)", ok)

    # =========================== 16. settle gate ============================================
    pos = scene.milk.data.root_pos_w
    q = scene.milk.data.root_quat_w
    teleport(scene.milk, pos, q, vel=[0.0, 0.0, 0.35], ang=[0.0, 0.0, 6.0],
             settle_steps=0)
    step(1)  # refresh buffers only — judge while still moving
    lv = float(scene.milk.data.root_lin_vel_w[0].norm())
    av = float(scene.milk.data.root_ang_vel_w[0].norm())
    s, ok = judge()
    report("settle-gate", s, ok)
    check("settle gate: the delivered carton kicked (lin={:.2f} m/s, ang={:.1f} "
          "rad/s) and judged immediately is NOT success (must be at rest)"
          .format(lv, av), (lv > c.settle_lin or av > c.settle_ang) and not ok)

    # =========================== 17. wrong object ===========================================
    env.reset(seed=81)
    settle_all(300)
    pos, q = bin_pose((-0.030, 0.0, FLOOR_LIE_Z), lie_quat)
    teleport(scene.juice, pos, q, settle_steps=0)
    settle_all(420)
    s, ok = judge()
    report("wrong-object", s, ok)
    check("wrong object: the JUICE carton inside the bin while the milk stands on "
          f"the floor is rejected twice over — milk not inside AND the decoy "
          f"clause (score={s:.3f} <= 0.02)",
          not ok and s <= 0.02 and not bool(scene.milk_inside()[0])
          and not bool(scene.decoy_out()[0]))

    # =========================== 18-19. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point except the "
          "constructed acceptance probes", not ever_bad_success[0])
    check("final: all task-object states finite (no NaN)", all_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.drop_chute")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(okc for _nm, okc in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, okc in checks:
            if not okc:
                print(f"[smoke]   FAILED: {nm}", flush=True)
    code = 0 if all_ok else 1
    # Hard exit: Kit teardown hangs — watchdog then die.
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)

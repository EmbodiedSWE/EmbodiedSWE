"""Smoke / rubric-REJECTION battery for RollerFreightScene (sim_gen task
`living_room_scene4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray_i73`) —
NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — stage the rollers, push the slab off the plinth,
ride the migrating bed to the dock — is the acceptance evidence). Every teleport here
is instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled state and
asserts the rubric REJECTS it, plus one physics probe that proves the force economics
are real (the SAME capped push that launches the slab off the slick plinth JAMS it on
the gritty floor when no rollers are staged — and the probe asserts the slab actually
moved first, so the jam is not vacuous). One probe DOES construct the genuine end state
on purpose — the acceptance construct — and one flip pair re-enters it; every other
judged point must stay success()=False and a final audit asserts exactly that.

  1-2.  settle/no-NaN     — reset settles finite; slab parked on the plinth, bowl
                            aboard, rollers chocked in the rack; score ~0, no success;
  3-5.  randomization     — READBACK over 8 seeded resets: fixture xy + yaw vary; slab
                            start depth + roller slot jitter vary; bowl deck spot varies;
  6.    null policy       — 300 idle steps -> score ~0, no success;
  7-8.  force economics   — the full-budget (40 N) fingertip shove with NO rollers:
        are real            (7) the slab MOVES on the slick plinth (launch readback —
                            the actuator is proven live, the vacuous-probe trap), then
                            (8) JAMS on the gritty floor short of mid-channel with the
                            force pinned at the cap and the slab at rest — direct
                            sliding CANNOT deliver the freight; score ~0, no success;
  9.    seed strategy     — the seed's move: the BOWL alone carried to the dock floor
                            (freight abandoned on the plinth) -> score ~0, no success;
  10.   dock by fiat      — the slab + bowl stood at the dock ON THE FLOOR (a state
                            the jam probe proves physics cannot reach; no rollers
                            under, not at riding height) -> riding/under gates reject;
                            staging never happened so score stays ~0;
  11.   near miss +       — the freight built riding a genuine 3-roller bed (straddling
        partial credit      the CoM) but 9 cm SHORT of the dock: no success, and the
                            latched score reads exactly staged+riding+half = 0.70;
  12.   acceptance        — the same riding assembly advanced to the dock window:
                            success TRUE, score 1.0;
  13-14. cargo clause     — the bowl lifted off the deck and stood on the open floor:
                            success flips FALSE; bowl restored upright on the deck:
                            success returns TRUE (13's rejection was the cargo clause
                            and nothing else);
  15.   settle gate       — the docked bowl kicked and judged immediately: NOT success
                            (must be at rest);
  16.   rejection audit   — success() was never True at any judged point EXCEPT the
                            acceptance construct and the flip-back (12, 14);
  17.   final no-NaN      — all task-object states finite at the end.

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
        BOWL_H, FLOOR_T, RACK_SLOTS, ROLL_R, SLAB_H, SLAB_L, X_MID, X_PE, X_STOP,
        _qapply, _qz, encode_force,
    )
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import (  # noqa: F401
        BOWL_H, FLOOR_T, RACK_SLOTS, ROLL_R, SLAB_H, SLAB_L, X_MID, X_PE, X_STOP,
        _qapply, _qz, encode_force,
    )
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

V_DES = 0.080  # shove waypoint speed (matches the solve)
KP_V = 800.0


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.roller_freight")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    n = env.num_envs
    cap = float(scene.cfg.push_cap)  # the FULL declared budget — the honesty line
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
        env.sim.set_camera_view(tuple(np.array((1.30, -1.10, 0.90)) + o),
                                tuple(np.array((0.00, 0.05, 0.06)) + o),
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

    def report(tag: str, s: float, ok: bool) -> None:
        fx = float(scene.slab_front_x()[0])
        sl = scene.slab_local()[0]
        print(f"[smoke] {tag:16s} | front_x={fx:+.3f} slab_z={float(sl[2]):.3f} "
              f"staged={bool(scene.rollers_staged()[0])} "
              f"riding={bool(scene.slab_riding()[0])} "
              f"under={bool(scene.roller_under()[0])} "
              f"aboard={bool(scene.bowl_aboard()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

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

    def fix_pose(local) -> torch.Tensor:
        return scene.fixture.data.root_pos_w + _qapply(
            scene.fixture.data.root_quat_w,
            torch.tensor(local, device=device).expand(n, 3))

    def q_fix() -> torch.Tensor:
        return scene.fixture.data.root_quat_w

    def chan_dir() -> torch.Tensor:
        ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
        d = _qapply(q_fix(), ex)
        return d / d.norm(dim=-1, keepdim=True)

    def clear_forces() -> None:
        scene.slab.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                 env_ids=all_ids)

    def settle_all(max_steps: int = 600) -> None:
        for _ in range(max_steps // 30):
            step(30)
            if all(bool(scene.settled(b)[0]) for b in
                   (scene.slab, scene.bowl, *scene.rollers)):
                break

    def all_finite() -> bool:
        bodies = [scene.fixture, scene.slab, scene.bowl, *scene.rollers]
        return all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)

    def place_bed_and_slab(cx: float, settle_steps: int = 180) -> None:
        """Construct the riding assembly by fiat (instrumentation): three rollers at
        slab-frame -0.10 / 0 / +0.10 (straddling the CoM at +0.04), the slab bottom at
        one roller diameter, the bowl upright on the deck."""
        for i, roller in enumerate(scene.rollers):
            rx = cx + (-0.10, 0.0, 0.10)[i]
            teleport(roller, fix_pose((rx, 0.0, FLOOR_T + ROLL_R + 0.001)), q_fix(),
                     settle_steps=0)
        teleport(scene.slab, fix_pose((cx, 0.0, FLOOR_T + 2 * ROLL_R + 0.0015)),
                 q_fix(), settle_steps=0)
        teleport(scene.bowl,
                 fix_pose((cx - 0.05, 0.0,
                           FLOOR_T + 2 * ROLL_R + SLAB_H + BOWL_H / 2 + 0.003)),
                 q_fix(), settle_steps=settle_steps)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    settle_all(480)
    s, ok = judge()
    report("reset", s, ok)
    check("settle: all states finite, slab parked on the plinth, bowl upright aboard, "
          "no roller in the channel",
          all_finite() and bool(scene.bowl_aboard()[0])
          and not bool(scene.roller_under()[0])
          and not bool(scene.rollers_staged()[0])
          and float(scene.slab_local()[0, 2]) > FLOOR_T + 0.02)
    check(f"settle: score ~0 and no success on a fresh reset (score={s:.3f})",
          s <= 0.02 and not ok)

    # =========================== 3-5. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(20)
        fp = (scene.fixture.data.root_pos_w - scene.env_origins)[0]
        ex = _qapply(q_fix(), torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))[0]
        yaw = math.atan2(float(ex[1]), float(ex[0]))
        ra = scene._local(scene.fixture, scene.rollers[0].data.root_pos_w)[0]
        bl = scene._local(scene.slab, scene.bowl.data.root_pos_w)[0]
        reads.append((float(fp[0]), float(fp[1]), yaw,
                      float(scene.slab_front_x()[0]),
                      float(ra[0]) - RACK_SLOTS[0], float(bl[0]), float(bl[1])))
    arr = np.array(reads)
    print("[smoke] randomization readback (fix_x, fix_y, fix_yaw, slab_front_x, "
          f"rollerA_slot_dx, bowl_x, bowl_y):\n{arr.round(3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: fixture pose varies across seeded resets (readback: "
          f"dx={spread[0]:.3f} dy={spread[1]:.3f} dyaw={spread[2]:.2f} rad)",
          spread[0] > 0.015 and spread[1] > 0.015 and spread[2] > 0.15)
    check("randomization: slab start depth and roller slot jitter vary (readback: "
          f"dfront={spread[3]:.3f} dslot={spread[4]:.3f})",
          spread[3] > 0.010 and spread[4] > 0.008)
    check("randomization: bowl deck spot varies (readback: "
          f"dx={spread[5]:.3f} dy={spread[6]:.3f})",
          spread[5] > 0.015 and spread[6] > 0.015)

    # =========================== 6. null policy fails =======================================
    env.reset(seed=31)
    step(300)
    s, ok = judge()
    report("null-policy", s, ok)
    check(f"null policy: score ~0 and no success after 300 idle steps (score={s:.3f})",
          s <= 0.02 and not ok)

    # =========================== 7-8. the force economics are real ==========================
    # The FULL declared budget (40 N) shoved along the channel with NO rollers staged:
    # the slab must launch off the slick plinth (the actuator is proven live) and then
    # JAM on the gritty floor — sliding friction demands ~2x the budget.
    env.reset(seed=35)
    settle_all(360)
    snap = scene.get_state(all_ids)
    q_ref = scene.slab.data.root_quat_w.clone()

    def shove(mode: int) -> tuple[bool, float, float, bool]:
        """Full-budget velocity-servo shove. Returns (frame_ok, travel, v_end, jammed)."""
        x0 = float(scene.slab_front_x()[0])
        jam, jammed = 0, False
        v_end = 0.0
        for i in range(1500):
            d = chan_dir()
            v_along = (scene.slab.data.root_lin_vel_w * d).sum(-1)
            f_mag = (KP_V * (V_DES - v_along)).clamp(0.0, cap)
            f = encode_force(mode, q_ref, scene.slab.data.root_quat_w,
                             d * f_mag.unsqueeze(-1))
            scene.slab.set_external_force_and_torque(
                f.view(n, 1, 3), zero_wrench, env_ids=all_ids, is_global=True)
            step(1)
            fx = float(scene.slab_front_x()[0])
            v_end = float(v_along[0])
            if i == 239 and fx < x0 + 0.010:
                clear_forces()
                return False, fx - x0, v_end, False  # no launch: wrong force frame
            if i % 240 == 239:
                print(f"[smoke] shove[m{mode}] @{i + 1}: front_x={fx:+.3f} "
                      f"v={v_end:+.3f} F={float(f_mag[0]):.1f}", flush=True)
            jam = jam + 1 if (abs(v_end) < 0.005 and float(f_mag[0]) > cap - 0.5
                              and i > 300) else 0
            if jam > 240:  # 2 s at rest with the force pinned at the cap
                jammed = True
                break
        clear_forces()
        return True, float(scene.slab_front_x()[0]) - x0, v_end, jammed

    frame_ok, travel, v_end, jammed = shove(1)
    if not frame_ok:
        scene.set_state(snap, all_ids)
        step(2)
        frame_ok, travel, v_end, jammed = shove(0)
    fx_jam = float(scene.slab_front_x()[0])
    s, ok = judge()
    report("shove-no-rollers", s, ok)
    check("force economics (the actuator is live): the capped shove launches the slab "
          f"off the SLICK plinth — it genuinely moved (travel={travel:+.3f} m)",
          frame_ok and travel > 0.03)
    check("force economics (sliding is unaffordable): with NO rollers the same "
          f"full-budget push JAMS on the gritty floor short of mid-channel "
          f"(front_x={fx_jam:+.3f} < {X_MID:.3f}, at rest with F pinned at "
          f"{cap:.0f} N), score ~0, no success",
          jammed and fx_jam < X_MID and not bool(scene.roller_under()[0])
          and s <= 0.02 and not ok)

    # =========================== 9. the seed's strategy =====================================
    # Pick-and-place the BOWL alone to the dock (the freight abandoned on the plinth).
    env.reset(seed=41)
    settle_all(360)
    teleport(scene.bowl, fix_pose((X_STOP - 0.06, 0.0, FLOOR_T + BOWL_H / 2 + 0.003)),
             q_fix(), settle_steps=150)
    s, ok = judge()
    report("seed-strategy", s, ok)
    check("seed strategy (bowl alone carried to the dock, freight abandoned): the bowl "
          f"is cargo, not the freight — score ~0 ({s:.3f}), no success",
          s <= 0.02 and not ok and not bool(scene.bowl_aboard()[0]))

    # =========================== 10. dock by fiat (no rollers) ==============================
    # Slab + bowl stood AT the dock directly on the gritty floor — a state probe 7-8
    # proves physics cannot reach — built here by fiat: riding/under gates reject it.
    env.reset(seed=51)
    settle_all(360)
    cx_dock = X_STOP - 0.012 - SLAB_L / 2
    teleport(scene.slab, fix_pose((cx_dock, 0.0, FLOOR_T + 0.001)), q_fix(),
             settle_steps=0)
    teleport(scene.bowl,
             fix_pose((cx_dock - 0.05, 0.0, FLOOR_T + SLAB_H + BOWL_H / 2 + 0.003)),
             q_fix(), settle_steps=180)
    s, ok = judge()
    report("dock-by-fiat", s, ok)
    check("dock by fiat: the slab (bowl aboard) stood at the dock ON THE FLOOR — no "
          "rollers under it, not at riding height — is rejected by the riding/under "
          f"gates; staging never happened so score stays ~0 ({s:.3f})",
          not ok and s <= 0.02 and bool(scene.bowl_aboard()[0])
          and not bool(scene.slab_riding()[0]) and not bool(scene.roller_under()[0]))

    # =========================== 11. near miss + partial credit =============================
    env.reset(seed=61)
    settle_all(360)
    place_bed_and_slab(cx=X_STOP - 0.09 - SLAB_L / 2, settle_steps=200)
    s, ok = judge()
    report("near-miss", s, ok)
    check("near miss + partial credit: the freight riding a genuine 3-roller bed but "
          f"9 cm SHORT of the dock is not success, and the latched score reads exactly "
          f"staged+riding+half = 0.70 (score={s:.3f})",
          not ok and abs(s - 0.70) < 0.02 and bool(scene.slab_riding()[0])
          and bool(scene.roller_under()[0]))

    # =========================== 12. acceptance construct ===================================
    # A free-standing bed assembly wanders along the channel (GPU contact creep) until
    # it meets a hard face — probe 11 parks with its rear against the plinth step. The
    # genuine end state is the OTHER hard face: nose pressed INTO the end stop (exactly
    # how the solve docks, where it holds through 3.3 s of persistence).
    place_bed_and_slab(cx=X_STOP - SLAB_L / 2, settle_steps=150)
    settle_all(240)
    s, ok = judge_accept()
    report("accept-construct", s, ok)
    check("acceptance construct: the same riding assembly pressed against the end "
          f"stop -> success TRUE (score={s:.3f})", ok and s >= 0.99)
    snap_acc = scene.get_state(all_ids)  # settled accepted state (near-zero velocities)

    # =========================== 13-14. cargo clause flips success ==========================
    # The flip pair must hold the freight state fixed while only the bowl clause changes.
    # A free assembly cannot be trusted to hold station across two long settles (the
    # roller creep walks the slab at ~2x roller speed once perturbed), so 14 restores the
    # EXACT accepted snapshot — bitwise-identical slab/bed state, bowl back aboard.
    teleport(scene.bowl, fix_pose((0.0, -0.45, BOWL_H / 2 + 0.012)),
             q_fix(), settle_steps=120)
    s, ok = judge()
    report("bowl-off", s, ok)
    check("cargo clause: the bowl lifted off the deck and stood on the open floor "
          "flips success FALSE", not ok and not bool(scene.bowl_aboard()[0]))
    scene.set_state(snap_acc, all_ids)
    step(10)
    s, ok = judge_accept()
    report("bowl-restored", s, ok)
    check("bowl restored aboard (exact accepted state restored) -> success returns "
          "TRUE (13's rejection was the cargo clause and nothing else)", ok)

    # =========================== 15. settle gate ============================================
    teleport(scene.bowl, scene.bowl.data.root_pos_w, scene.bowl.data.root_quat_w,
             vel=[0.0, 0.0, 0.35], ang=[0.0, 0.0, 6.0], settle_steps=0)
    step(1)  # refresh buffers only — judge while still moving
    lv = float(scene.bowl.data.root_lin_vel_w[0].norm())
    av = float(scene.bowl.data.root_ang_vel_w[0].norm())
    s, ok = judge()
    report("settle-gate", s, ok)
    check(f"settle gate: the docked bowl kicked (lin={lv:.2f} m/s, ang={av:.1f} rad/s) "
          "and judged immediately is NOT success (must be at rest)",
          (lv > scene.cfg.settle_lin or av > scene.cfg.settle_ang) and not ok)

    # =========================== 16-17. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point except the "
          "acceptance construct and the flip-back", not ever_bad_success[0])
    check("final: all task-object states finite (no NaN)", all_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.roller_freight")
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

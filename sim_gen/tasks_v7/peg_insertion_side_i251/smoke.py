"""Smoke / rubric-REJECTION battery for QuarterLatchScene (sim_gen task
`peg_insertion_side_i251`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — insert the key under a floating-hand force
controller, twist the dial a quarter turn through the key, grind the key back out —
is the acceptance evidence that the rubric ACCEPTS a correct outcome). Every teleport
here is instrumentation that CONSTRUCTS a wrong (or partial) outcome and asserts the
rubric REJECTS it; no probe in this battery ever reaches success(), and a final audit
check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: key and decoy flat on the
                            floor, dial resting on its 0-stop (angle readback);
                            score ~0 at rest, no success;
  3-4. randomization      — READBACK over 6 seeded resets: housing xy + yaw, key xy +
                            yaw, decoy xy, and the approach baseline d0 all move;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  SEED strategy       — the seed's whole plan ("get the peg deep into the hole,
                            done") = key blade seated 12 mm deep in the slot, never
                            twisted: verified deep by readback, dial still on its
                            0-stop -> NOT success, score <= 0.45 (approach + engage
                            credit only);
  7.  under-rotation      — dial released at ~40 deg (below the 45-deg over-center):
                            the counterweight snaps it BACK to the 0-stop (readback)
                            -> NOT at_stop, NOT success (a half-turned dial is
                            worth only partial angle credit);
  8.  key left inserted   — dial released at ~60 deg completes OVER-CENTER to its far
                            stop (mechanism honesty, readback ~90 deg) while the key
                            still hangs in the porthole -> at_stop TRUE but NOT
                            success, score capped at 0.95 (withdrawal is required);
  9.  wrong object        — the RED decoy pressed at the porthole by a real force
                            servo, same approach pose as the blue key: its 56 mm
                            blade never passes the collar mouth (readback: it
                            reached the mouth AND never entered) -> no credit, NOT
                            success (identity control);
  10. latched credit      — inserting the key deep then removing it to the floor
                            leaves the latched approach + engage score unchanged
                            (and the removed key alone is still not success);
  11. monotonicity        — moving the key closer to the collar mouth latches
                            strictly more approach credit than a farther placement;
  12. rejection audit     — success() was never True at ANY judged point;
  13. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.peg_insertion_side_i251.smoke --headless
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
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.quarter_latch_drum")().build(num_envs=args.num_envs,
                                                       device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.72, -0.62, 0.52)) + o),
                                tuple(np.array((0.0, 0.08, 0.12)) + o),
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

    ever_success = [False]

    def judge() -> tuple[float, bool]:
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return s, ok

    def h_frame() -> tuple[torch.Tensor, torch.Tensor, float]:
        """(world housing origin of env0, world bore axis, housing yaw)."""
        from isaaclab.utils.math import quat_apply

        hp = scene.housing.data.root_pos_w[0]
        q = scene.housing.data.root_quat_w[0]
        ax = quat_apply(q.unsqueeze(0),
                        torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
        hyaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return hp, ax, hyaw

    def theta_deg() -> float:
        return math.degrees(float(scene.dial_theta()[0]))

    def report(tag: str) -> None:
        _, tip, _ = scene._key_pts(scene.key)
        th = scene._h_local(tip)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | tip_h=({float(th[0]):+.3f},{float(th[1]):+.3f},"
              f"{float(th[2]):+.3f}) theta={theta_deg():+6.1f}deg "
              f"clear={bool(scene.key_clear()[0])} at_stop={bool(scene.dial_at_stop()[0])} "
              f"appr={float(scene.approach_latch[0]):.3f} "
              f"eng={float(scene.engage_latch[0]):.3f} "
              f"turn={float(scene.turn_latch[0]):.3f} "
              f"stop={float(scene.stop_latch[0]):.3f} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_body(body, pos_w: torch.Tensor, quat, settle_steps: int = 30) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap). pos_w is a WORLD position."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def insert_quat(hyaw: float) -> tuple[float, float, float, float]:
        """q = qz(hyaw) * qy(-90 deg): key local +z (tip) points INTO the bore,
        blade width vertical (the slot orientation at the 0-stop)."""
        c45 = math.cos(math.pi / 4)
        cy2, sy2 = math.cos(hyaw / 2), math.sin(hyaw / 2)
        return (cy2 * c45, sy2 * c45, -cy2 * c45, sy2 * c45)

    def key_at_depth(body, tip_x: float, settle_steps: int = 30) -> None:
        """Place a key body along the bore with its tip at housing-local x = tip_x."""
        hp, ax, hyaw = h_frame()
        place_body(body, hp + ax * (tip_x + c.key_half), insert_quat(hyaw),
                   settle_steps=settle_steps)

    def rotate_dial_to(theta: float, settle_steps: int) -> None:
        """Consistent linkage write: dial recentred in its pocket at angle theta
        (radians about the bore axis), zero velocity, then REAL settle steps."""
        from isaaclab.utils.math import quat_apply, quat_mul

        hp = scene.housing.data.root_pos_w
        q_h = scene.housing.data.root_quat_w
        off = torch.tensor([c.disc_x, 0.0, 0.0], device=device).expand(n, 3)
        qx = torch.tensor([math.cos(theta / 2), math.sin(theta / 2), 0.0, 0.0],
                          device=device).expand(n, 4)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = hp + quat_apply(q_h, off)
        st[:, 3:7] = quat_mul(q_h, qx)
        scene.dial.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def floor_flat_quat(yaw: float) -> tuple[float, float, float, float]:
        """q = qz(yaw) * qx(-90 deg): the reset lying-flat pose."""
        c45 = math.cos(math.pi / 4)
        cy2, sy2 = math.cos(yaw / 2), math.sin(yaw / 2)
        return (cy2 * c45, -cy2 * c45, -sy2 * c45, sy2 * c45)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    kz = float((scene.key.data.root_pos_w - scene.env_origins)[0, 2])
    dz = float((scene.decoy.data.root_pos_w - scene.env_origins)[0, 2])
    fin0 = bool(torch.isfinite(scene.key.data.root_state_w).all()
                and torch.isfinite(scene.dial.data.root_state_w).all()
                and torch.isfinite(scene.decoy.data.root_state_w).all())
    th0 = theta_deg()
    check("settle: states finite; key and decoy flat on the floor, dial resting on its "
          "0-stop (readback)",
          fin0 and abs(kz - c.shaft_r) < 0.006 and abs(dz - c.shaft_r) < 0.006
          and -12.0 < th0 < 6.0 and bool(scene.key_settled()[0])
          and bool(scene.dial_settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(5)
        hp, _ax, hyaw = h_frame()
        hpo = hp - scene.env_origins[0]
        k = (scene.key.data.root_pos_w - scene.env_origins)[0]
        qk = scene.key.data.root_quat_w[0]
        # yaw of the lying key: twist of its long axis in the ground plane
        _, tipw, _ = scene._key_pts(scene.key)
        d = (tipw - scene.key.data.root_pos_w)[0]
        kyaw = math.atan2(float(d[1]), float(d[0]))
        de = (scene.decoy.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(hpo[0]), float(hpo[1]), hyaw, float(k[0]), float(k[1]),
                      kyaw, float(de[0]), float(de[1]), float(scene.d0[0])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (h_x, h_y, h_yaw, key_x, key_y, key_yaw, "
          f"decoy_x, decoy_y, d0):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: housing xy + yaw vary across seeded resets (readback)",
          spread[0] > 0.005 and spread[1] > 0.005 and spread[2] > 0.2)
    check("randomization: key xy + yaw, decoy xy, and the approach baseline d0 vary "
          "across seeded resets (readback)",
          spread[3] > 0.005 and spread[4] > 0.005 and spread[5] > 0.2
          and spread[6] > 0.005 and spread[7] > 0.005 and spread[8] > 0.005)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. SEED strategy ===========================================
    # The seed's whole plan is "get the peg deep into the hole — done". Here that is a
    # key blade seated 12 mm deep in the dial's slot and NEVER twisted: the judged
    # body (the dial) still rests on its 0-stop. Deepest possible insertion must top
    # out at the approach + engage credit.
    env.reset(seed=41)
    step(10)
    key_at_depth(scene.key, c.insert_tip_x, settle_steps=3)
    _, tip, _ = scene._key_pts(scene.key)
    th = scene._h_local(tip)[0]
    deep = float(th[0]) <= c.insert_tip_x + 0.006 and float(th[1:3].norm()) < 0.012
    step(57)  # release: the unsupported key may cam partway back out of the slick slot
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy (key seated 12 mm deep in the slot, never twisted): verified "
          "deep by readback (engage latched 1.0), still in the porthole when settled, "
          "dial still on its 0-stop, NOT success, score <= 0.45",
          deep and float(scene.engage_latch[0]) > 0.99
          and not bool(scene.key_clear()[0]) and abs(theta_deg()) < 12.0
          and not ok and s <= 0.45)

    # =========================== 7. under-rotation snaps BACK ===============================
    # Bistability, refusal side: a dial released at ~40 deg (below the 45-deg
    # over-center) must fall back to its 0-stop — a half-turn earns only the partial
    # angle credit and never the stop latch. Key stays on the floor (clear).
    env.reset(seed=51)
    step(10)
    rotate_dial_to(math.radians(40.0), settle_steps=2)
    th_set = theta_deg()
    rotate_dial_to(math.radians(40.0), settle_steps=300)
    report("under-rotation")
    s, ok = judge()
    check("under-rotation: dial released at ~40 deg (readback) snaps BACK to the "
          "0-stop — NOT at_stop, NOT success",
          abs(th_set - 40.0) < 4.0 and theta_deg() < 20.0
          and not bool(scene.dial_at_stop()[0]) and not ok)

    # =========================== 8. key left inserted (cap 0.95) ============================
    # Over-center completion side + the task's signature refusal: the dial released at
    # ~60 deg completes to its far stop on its own (mechanism honesty), but the key
    # still hangs in the porthole -> at_stop TRUE, key NOT clear, success FALSE, and
    # the score is capped at 0.95: withdrawal is a required part of the outcome.
    env.reset(seed=61)
    step(10)
    key_at_depth(scene.key, -0.005, settle_steps=30)  # in the aperture, NOT in the slot
    rotate_dial_to(math.radians(60.0), settle_steps=300)
    report("key-inserted")
    s, ok = judge()
    check("key left inserted: dial released at ~60 deg completes OVER-CENTER to the "
          "far stop (readback ~90 deg) but the key still hangs in the porthole — "
          "at_stop TRUE yet NOT success, score <= 0.95",
          bool(scene.dial_at_stop()[0]) and not bool(scene.key_clear()[0])
          and not ok and s <= 0.95 + 1e-6)

    # =========================== 9. wrong object under a REAL push ==========================
    # Identity control with a non-vacuous physical probe: the RED decoy is pressed at
    # the porthole by a force servo in the SAME approach pose the blue key uses. Its
    # 56 mm blade is wider than the octagon's maximal chord: it must reach the collar
    # mouth (the push is real) and never pass it.
    env.reset(seed=71)
    step(10)
    hp, ax, hyaw = h_frame()
    place_body(scene.decoy, hp + ax * (c.mouth_x + 0.012 + c.key_half),
               insert_quat(hyaw), settle_steps=1)
    m_d = c.decoy_mass
    min_tip = 1.0
    for _ in range(240):
        axis, tip_w, _ = scene._key_pts(scene.decoy)
        tx = float(scene._h_local(tip_w)[0, 0])
        min_tip = min(min_tip, tx)
        v = scene.decoy.data.root_lin_vel_w[0]
        w = scene.decoy.data.root_ang_vel_w[0]
        v_ax = float(torch.dot(v, ax))
        f_ax = m_d * 25.0 * (-0.05 - v_ax)  # press INTO the bore at 5 cm/s
        f_ax = max(-2.0, min(2.0, f_ax))
        rel = tip_w[0] - hp
        e_perp = -(rel - torch.dot(rel, ax) * ax)
        v_perp = v - torch.dot(v, ax) * ax
        f_lat = (m_d * (600.0 * e_perp - 40.0 * v_perp)).clamp(-3.0, 3.0)
        f = f_ax * ax + f_lat + m_d * 9.81 * ez
        tq = (0.04 * torch.linalg.cross(axis[0], -ax)
              - 0.004 * (w - torch.dot(w, axis[0]) * axis[0])).clamp(-0.08, 0.08)
        scene.decoy.set_external_force_and_torque(
            f.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            tq.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            env_ids=all_ids, is_global=True)
        env.step(no_action)
    scene.decoy.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    step(30)
    report("wrong-object")
    s, ok = judge()
    print(f"[smoke] decoy push readback: min_tip_x={min_tip:+.4f} "
          f"(mouth at {c.mouth_x:+.3f})", flush=True)
    check("wrong object: the RED decoy pressed at the porthole reached the collar "
          "mouth (the push was real) yet never passed it — no credit, NOT success",
          c.mouth_x - 0.008 < min_tip < c.mouth_x + 0.008 and s <= 0.05 and not ok)

    # =========================== 10. latched credit survives regression =====================
    env.reset(seed=81)
    step(10)
    key_at_depth(scene.key, c.insert_tip_x, settle_steps=60)
    report("inserted")
    s_in, _ = judge()
    hp, _ax, _ = h_frame()
    far = hp.clone()
    far[0] += 0.35
    far[1] -= 0.30
    far[2] = float(scene.env_origins[0, 2]) + c.shaft_r + 0.0005
    place_body(scene.key, far, floor_flat_quat(0.3), settle_steps=60)
    report("key-removed")
    s_out, ok = judge()
    check("latched credit: removing the deeply inserted key to the floor leaves the "
          "latched approach + engage score unchanged (and still not success)",
          s_in >= 0.30 and abs(s_out - s_in) < 0.02 and not ok)

    # =========================== 11. approach monotonicity ==================================
    env.reset(seed=91)
    step(5)
    hp, ax, hyaw = h_frame()
    mouth = hp + ax * c.mouth_x
    _, tip0, _ = scene._key_pts(scene.key)
    v0 = mouth - tip0[0]
    place_body(scene.key, scene.key.data.root_pos_w[0] + 0.5 * v0,
               insert_quat(hyaw), settle_steps=3)
    a_half = float(scene.approach_latch[0])
    place_body(scene.key, scene.key.data.root_pos_w[0] + 0.35 * v0,
               insert_quat(hyaw), settle_steps=3)
    a_near = float(scene.approach_latch[0])
    check("monotonicity: moving the key closer to the collar mouth latches strictly "
          f"more approach credit ({a_half:.3f} < {a_near:.3f})", a_half + 0.10 < a_near)

    # =========================== 12-13. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.housing.data.root_state_w).all()
           and torch.isfinite(scene.dial.data.root_state_w).all()
           and torch.isfinite(scene.key.data.root_state_w).all()
           and torch.isfinite(scene.decoy.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.quarter_latch_drum")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, ok in checks:
            if not ok:
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
    main()

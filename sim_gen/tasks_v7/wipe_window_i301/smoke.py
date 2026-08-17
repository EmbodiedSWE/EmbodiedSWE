"""Smoke battery for GlazeWindowScene — REJECTION tests for the rubric, NullRobot,
RECORDED.

This is NOT a solution (the solution is solve.py — lift the cap out of its tray, thread
the pane down the open top into the sill groove under contact, recap; the Franka
strategy is TASK.md's embodiment argument). Probe drives here are the same
fingertip-scale servos as solve.py (shared numbers = shared honesty). Wrong OUTCOMES
are constructed as settled states (teleport, settle, evaluate) — teleports here build
the states the rubric must REJECT, never credit.

One linear run, 12 named checks:
  1. settle    — clean reset: finite state, pane racked upright in the stand and cap
                 seated level in its tray (world-pose READBACK vs the cfg formulas),
                 no latch pre-fired, score ~0, no success;
  2. random    — frame x/y/yaw AND stand x/y/yaw vary across 8 seeds; pane and cap
                 spawn at the sampled poses on every draw (READBACK);
  3. null      — 2 s of nothing: score < 0.05, no success — and the cap sitting in its
                 tray the whole time still latches NOTHING (order-aware gate);
  4. negative  — the SEED's plan (wipe the glass): pressing the racked pane flat-on
                 with an up/down rubbing force for 1.5 s seats nothing, ~0;
  5. negative  — cap ON, honest insertion attempted: the pane, lowered by the same
                 descent servo as solve.py, STALLS on the cap lid far above the groove
                 — the corridor really is closed; no latch, ~0, cap undisturbed;
  6. negative  — near miss: pane standing on the table right in FRONT of the frame
                 (x off by 10 cm, bottom on the table) — not seated, ~0;
  7. negative  — wrong ORDER: uncap, then recap the still-EMPTY frame — the cap sits
                 geometrically perfect in its tray, yet the capped latch stays cold;
                 only the cap_off share (0.20) is earned, no success;
  8. partial   — honest uncap + honest pane insertion, cap left parked: exactly the
                 cap_off + seated share (~0.60), no success (frame not roofed);
  9. exactness — continue: honest recap -> success() and score == 1.0, still true 1 s
                 later hands-off;
 10. latch     — plucking the cap back off the finished window revokes success (live
                 state); every latched share (0.85) remains;
 11. negative  — the cap dropped back on the tray ROTATED 90 deg cannot find the
                 tray: its narrow axis slips between the posts and it jams tilted
                 ~30 mm below the rest height: cap_seated_now stays False — a
                 misplaced lid is not a seated lid;
 12. frames    — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.wipe_window_i301.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=10)
parser.add_argument("--max_frames", type=int, default=280)
parser.add_argument("--out", type=str, default="frames.npz")  # CWD — the pipeline fetches it
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe (proven on this render stack): kit mis-decodes the driver version and
# silently rejects RTX -> the annotator returns EMPTY frames.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.wipe_window_i301 import scene as scene_mod
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod

CAP = scene_mod.CAP
PANE = scene_mod.PANE

# Watchdog: never leave a GPU zombie.
threading.Timer(1500.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                 os._exit(3))).start()

# Same fingertip-scale servos as solve.py (shared numbers = shared honesty).
G = 9.81
F_MAX = 4.0
KZ = 2.0
V_UP = 0.15
V_DOWN = 0.15
V_DOWN_SLOW = 0.06
KP_XY = 20.0
KD_XY = 1.0
F_XY = 0.3
WIPE_F = 1.5  # N — the seed-style flat-on wiping press against the pane

PROBE_SEED = 5


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.glaze_window")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    ids0 = torch.tensor([0], device=device)

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.55, -0.85, 1.20)) + o),
                                tuple(np.array((0.40, 0.0, 0.55)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (800, 500))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames — check the RTX recipe",
                  flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action, render=annot is not None)
            if annot is not None and step_i % args.record_every == 0 \
                    and len(frames) < args.max_frames:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    def fyaw() -> float:
        return float(scene.frame_yaw[0])

    def report(tag: str) -> None:
        b = scene.to_frame(scene.pane_bottom_env())[0]
        p = scene.to_frame(scene.cap_pos())[0]
        print(f"[smoke] {tag:14s} | pane_bot=({float(b[0]):+.3f},{float(b[1]):+.3f},"
              f"{float(b[2]):.3f}) cap=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):.3f}) latch=[off={int(bool(scene.cap_off_latch[0]))} "
              f"seat={int(bool(scene.seated_latch[0]))} cap={int(bool(scene.capped_latch[0]))}] "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def finite_all() -> bool:
        return bool(torch.isfinite(scene.pane.data.root_state_w).all()) \
            and bool(torch.isfinite(scene.cap.data.root_state_w).all()) \
            and bool(torch.isfinite(scene.score()).all())

    def spawn_readback_err() -> float:
        """Max |body pos - the cfg spawn formula| for pane and cap (readback, m)."""
        pane_want = scene._place_env(
            scene.stand_xy[:1], scene.stand_yaw[:1],
            (0.0, 0.0, c.pane_stand_z))[0] + scene.env_origins[0]
        cap_want = scene._place_env(
            scene.frame_xy[:1], scene.frame_yaw[:1],
            (0.0, 0.0, c.cap_rest_z))[0] + scene.env_origins[0]
        e1 = float((scene.pane.data.root_pos_w[0] - pane_want).norm())
        e2 = float((scene.cap.data.root_pos_w[0] - cap_want).norm())
        return max(e1, e2)

    def teleport(body, row: int, pos_env, yaw: float) -> None:
        scene.drive_f[0, row, :] = 0.0
        st = torch.zeros(1, 13, device=device)
        st[0, 0] = scene.env_origins[0, 0] + pos_env[0]
        st[0, 1] = scene.env_origins[0, 1] + pos_env[1]
        st[0, 2] = pos_env[2]
        st[0, 3] = math.cos(yaw / 2)
        st[0, 6] = math.sin(yaw / 2)
        body.write_root_state_to_sim(st, ids0)
        step(1)

    def lift_to(body, row: int, mass: float, z_env: float, timeout: int = 600) -> bool:
        """solve.py's vertical velocity-cascade lift."""
        for _ in range(timeout):
            if float(body.data.root_pos_w[0, 2]) >= z_env:
                scene.drive_f[0, row, :] = 0.0
                return True
            vz = float(body.data.root_lin_vel_w[0, 2])
            scene.drive_f[0, row, 2] = max(0.0, min(F_MAX, mass * G + KZ * (V_UP - vz)))
            step(1)
        scene.drive_f[0, row, :] = 0.0
        return False

    def guided_descent(body, row: int, mass: float, bottom_fn, z_goal: float,
                       z_slow: float, timeout: int = 900) -> bool:
        """solve.py's contact descent servo. True iff the tracked point reached z_goal."""
        z_hist: list[float] = []
        for _k in range(timeout):
            b = scene.to_frame(bottom_fn())[0]
            bz = float(b[2])
            if bz <= z_goal:
                scene.drive_f[0, row, :] = 0.0
                return True
            z_hist.append(bz)
            if len(z_hist) > 150 and abs(z_hist[-1] - z_hist[-150]) < 0.002 \
                    and bz > z_goal + 0.015:
                scene.drive_f[0, row, :] = 0.0
                print(f"[smoke] descent stalled at z={bz:.3f} (goal {z_goal:.3f})",
                      flush=True)
                return False
            v = body.data.root_lin_vel_w[0]
            v_des = -V_DOWN if bz > z_slow else -V_DOWN_SLOW
            fz = mass * G + KZ * (v_des - float(v[2]))
            cy, sy = math.cos(fyaw()), math.sin(fyaw())
            vx_f = cy * float(v[0]) + sy * float(v[1])
            vy_f = -sy * float(v[0]) + cy * float(v[1])
            fx_f = max(-F_XY, min(F_XY, -KP_XY * float(b[0]) - KD_XY * vx_f))
            fy_f = max(-F_XY, min(F_XY, -KP_XY * float(b[1]) - KD_XY * vy_f))
            scene.drive_f[0, row, 0] = cy * fx_f - sy * fy_f
            scene.drive_f[0, row, 1] = sy * fx_f + cy * fy_f
            scene.drive_f[0, row, 2] = max(0.0, min(F_MAX, fz))
            step(1)
        scene.drive_f[0, row, :] = 0.0
        return False

    def uncap() -> bool:
        """solve.py's phase 1: lift the cap out of the tray, park it on the table."""
        if not lift_to(scene.cap, CAP, c.cap_mass, c.table_top + c.post_top + 0.08):
            return False
        park_y = -0.38 if float(scene.stand_xy[0, 1]) >= 0.0 else 0.38
        teleport(scene.cap, CAP, (-0.02, park_y, c.table_top + 0.03), 0.0)
        step(90)
        return bool(scene.cap_off_latch[0])

    def seat_pane() -> bool:
        """solve.py's phase 2: lift the pane from the stand, thread it into the groove."""
        if not lift_to(scene.pane, PANE, c.pane_mass, c.table_top + 0.24):
            return False
        for _attempt in range(3):
            hover = scene.frame_point_env((0.0, 0.0, 0.41))[0]
            teleport(scene.pane, PANE,
                     (float(hover[0]), float(hover[1]), float(hover[2])), fyaw())
            if guided_descent(scene.pane, PANE, c.pane_mass, scene.pane_bottom_env,
                              z_goal=c.groove_floor + 0.005, z_slow=0.13):
                scene.drive_f[0, PANE, 2] = -0.6
                step(30)
                scene.drive_f[0, PANE, :] = 0.0
                step(60)
                if bool(scene.pane_seated_now()[0]) and bool(scene.seated_latch[0]):
                    return True
            if not lift_to(scene.pane, PANE, c.pane_mass, c.table_top + 0.41):
                return False
        return False

    def recap() -> bool:
        """solve.py's phase 3: lift the parked cap, seat it back in the tray."""
        if not lift_to(scene.cap, CAP, c.cap_mass, c.table_top + 0.12):
            return False
        hover = scene.frame_point_env((0.0, 0.0, 0.36))[0]
        teleport(scene.cap, CAP, (float(hover[0]), float(hover[1]), float(hover[2])),
                 fyaw())
        ok = guided_descent(scene.cap, CAP, c.cap_mass, scene.cap_pos,
                            z_goal=c.cap_rest_z + 0.004, z_slow=0.34)
        scene.drive_f[0, CAP, :] = 0.0
        step(90)
        return ok and bool(scene.cap_seated_now()[0])

    # ========================= 1. settle / clean-slate + spawn readback ========================
    torch.manual_seed(PROBE_SEED)
    env.reset()
    step(60)
    report("reset")
    rerr = spawn_readback_err()
    pane_yaw = 2.0 * math.atan2(float(scene.pane.data.root_quat_w[0, 3]),
                                float(scene.pane.data.root_quat_w[0, 0]))
    want_yaw = float(scene.stand_yaw[0]) - math.pi / 2
    yaw_err = abs(math.atan2(math.sin(pane_yaw - want_yaw), math.cos(pane_yaw - want_yaw)))
    print(f"[smoke] spawn readback err={rerr * 1000:.1f}mm pane_yaw_err={yaw_err:.3f}rad",
          flush=True)
    check("settle: clean reset (finite, pane racked in the stand + cap seated in the "
          "tray by readback, no latch pre-fired, score ~0)",
          finite_all() and rerr < 0.010 and yaw_err < 0.10
          and bool(scene.cap_seated_now()[0]) and not bool(scene.pane_seated_now()[0])
          and not bool(scene.cap_off_latch[0]) and not bool(scene.seated_latch[0])
          and not bool(scene.capped_latch[0])
          and float(scene.score()[0]) < 0.02 and not bool(scene.success()[0]))

    # ========================= 2. randomization across seeds ==================================
    draws = []
    ok_rb = True
    for seed in (11, 12, 13, 14, 15, 16, 17, 18):
        torch.manual_seed(seed)
        env.reset()
        step(2)
        draws.append((round(float(scene.frame_xy[0, 0]), 3),
                      round(float(scene.frame_xy[0, 1]), 3),
                      round(fyaw(), 3),
                      round(float(scene.stand_xy[0, 0]), 3),
                      round(float(scene.stand_xy[0, 1]), 3),
                      round(float(scene.stand_yaw[0]), 3)))
        ok_rb &= spawn_readback_err() < 0.012
    print(f"[smoke] draws (fx, fy, fyaw, sx, sy, syaw): {draws}", flush=True)
    check("randomization is real (frame pose + yaw and stand pose + yaw vary; spawn "
          "poses match by readback)",
          len({d[2] for d in draws}) >= 5 and len({d[5] for d in draws}) >= 5
          and len({(d[0], d[1]) for d in draws}) >= 5
          and len({(d[3], d[4]) for d in draws}) >= 5 and ok_rb)

    # ========================= 3. null policy =================================================
    torch.manual_seed(PROBE_SEED)
    env.reset()
    step(240)  # 2 s of nothing
    report("null")
    check("null policy: score < 0.05, no success — and the cap resting in its tray all "
          "along latches NOTHING (order-aware gate)",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0])
          and bool(scene.cap_seated_now()[0]) and not bool(scene.capped_latch[0]))

    # ========================= 4. negative: the SEED's plan (wipe the pane) ===================
    torch.manual_seed(PROBE_SEED)
    env.reset()
    step(30)
    syaw = float(scene.stand_yaw[0])
    press = (-math.sin(syaw) * WIPE_F, math.cos(syaw) * WIPE_F)  # flat-on, into the pane
    for k in range(180):  # 1.5 s of pressing + up/down rubbing
        scene.drive_f[0, PANE, 0] = press[0]
        scene.drive_f[0, PANE, 1] = press[1]
        scene.drive_f[0, PANE, 2] = 0.3 if (k // 30) % 2 == 0 else -0.3
        step(1)
    scene.drive_f[0, PANE, :] = 0.0
    step(120)
    report("wipe-press")
    check("negative (seed strategy): a flat-on wiping press against the pane seats "
          "nothing — no latch, ~0, no success",
          finite_all() and not bool(scene.pane_seated_now()[0])
          and not bool(scene.seated_latch[0])
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 5. negative: cap ON, insertion attempted =======================
    torch.manual_seed(PROBE_SEED)
    env.reset()
    step(30)
    ok_lift = lift_to(scene.pane, PANE, c.pane_mass, c.table_top + 0.24)
    hover = scene.frame_point_env((0.0, 0.0, 0.41))[0]
    teleport(scene.pane, PANE, (float(hover[0]), float(hover[1]), float(hover[2])), fyaw())
    reached = guided_descent(scene.pane, PANE, c.pane_mass, scene.pane_bottom_env,
                             z_goal=c.groove_floor + 0.005, z_slow=0.13)
    step(60)
    report("capped-block")
    bz = float(scene.to_frame(scene.pane_bottom_env())[0, 2])
    check("negative (corridor closed): with the cap ON, solve.py's own descent servo "
          "stalls on the lid far above the groove — no seat, ~0, cap undisturbed",
          ok_lift and not reached and 0.20 < bz < 0.37
          and not bool(scene.pane_seated_now()[0]) and not bool(scene.seated_latch[0])
          and bool(scene.cap_seated_now()[0])
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 6. negative: near miss (in front of the frame) =================
    torch.manual_seed(PROBE_SEED)
    env.reset()
    step(30)
    spot = scene.frame_point_env((-0.10, 0.0, c.pane_size[2] / 2 + 0.003))[0]
    teleport(scene.pane, PANE, (float(spot[0]), float(spot[1]), float(spot[2])), fyaw())
    step(150)
    report("near-miss")
    check("negative (near miss): pane standing on the table right in FRONT of the "
          "frame, 10 cm from the groove — not seated, ~0",
          finite_all() and not bool(scene.pane_seated_now()[0])
          and not bool(scene.seated_latch[0])
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 7. negative: wrong order (recap the EMPTY frame) ===============
    torch.manual_seed(PROBE_SEED)
    env.reset()
    step(30)
    ok_off = uncap()
    ok_back = recap()  # geometrically perfect recap — but the frame is still empty
    step(60)
    report("empty-recap")
    s7 = float(scene.score()[0])
    check("negative (wrong order): recapping the still-empty frame parks the cap "
          "perfectly in its tray yet latches NOTHING beyond cap_off (0.20), no success",
          ok_off and ok_back and bool(scene.cap_seated_now()[0])
          and not bool(scene.capped_latch[0]) and not bool(scene.seated_latch[0])
          and abs(s7 - c.w_cap_off) < 0.02 and not bool(scene.success()[0]))

    # ========================= 8. partial: uncap + seat, frame left open ======================
    torch.manual_seed(PROBE_SEED)
    env.reset()
    step(30)
    ok_off = uncap()
    ok_seat = seat_pane()
    step(60)
    report("open-glazed")
    s8 = float(scene.score()[0])
    check("partial credit: honest uncap + honest pane seating, cap left parked -> "
          "exactly the cap_off + seated share, no success",
          ok_off and ok_seat and abs(s8 - (c.w_cap_off + c.w_seated)) < 0.02
          and not bool(scene.capped_latch[0]) and not bool(scene.success()[0]))

    # ========================= 9. exactness: finish the assembly ==============================
    ok_recap = recap()
    good = False
    for _ in range(48):
        if bool(scene.success()[0]):
            good = True
            break
        step(10)
    report("goal-state")
    s_goal = float(scene.score()[0])
    step(120)  # ...and it persists hands-off
    check("exactness: pane seated + cap reseated -> success() and score == 1.0, stable",
          ok_recap and good and abs(s_goal - 1.0) < 1e-3
          and bool(scene.success()[0]) and abs(float(scene.score()[0]) - 1.0) < 1e-3)

    # ========================= 10. achievement latch ==========================================
    park_y = -0.38 if float(scene.stand_xy[0, 1]) >= 0.0 else 0.38
    teleport(scene.cap, CAP, (-0.02, park_y, c.table_top + 0.03), 0.0)
    step(120)
    report("revoked")
    sr = float(scene.score()[0])
    check("achievement latch: plucking the cap back off revokes success (live state); "
          "the latched 0.85 remains",
          not bool(scene.success()[0])
          and abs(sr - (c.w_cap_off + c.w_seated + c.w_capped)) < 0.02)

    # ========================= 11. negative: misoriented cap ==================================
    hover = scene.frame_point_env((0.0, 0.0, 0.36))[0]
    teleport(scene.cap, CAP, (float(hover[0]), float(hover[1]), float(hover[2])),
             fyaw() + math.pi / 2)  # 90 deg off: the long axis bridges the corner nubs
    step(180)
    report("cap-90deg")
    p = scene.to_frame(scene.cap_pos())[0]
    check("negative (misoriented cap): dropped back 90 deg rotated it never finds the "
          "tray (slips between the posts, jams off-height) — cap_seated stays False, "
          "success stays revoked",
          finite_all() and not bool(scene.cap_seated_now()[0])
          and not bool(scene.success()[0]) and float(p[2]) > 0.05)

    # ========================= 12. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.glaze_window")
        print(f"[smoke] saved {arr.shape} -> {os.path.abspath(args.out)}", flush=True)
    check("video frames recorded", len(frames) >= 20)

    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        bad = [nm for nm, ok in checks if not ok]
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)} — failing: {bad}", flush=True)
    # Kit teardown hangs are routine: watchdog then hard exit.
    threading.Timer(10.0, lambda: os._exit(0 if all_ok else 1)).start()
    try:
        env.close()
        app.close()
    finally:
        os._exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

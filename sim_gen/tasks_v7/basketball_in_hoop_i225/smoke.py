"""Smoke / rubric-REJECTION battery for TiltDispenserScene (sim_gen task
`basketball_in_hoop_i225`) — NullRobot, teleported probe states + torque probes, RECORDED.

This is NOT a solution (solve.py — read the open side, press the paddle, hold at the
tilt stop, release — is the acceptance evidence that the rubric ACCEPTS a correct
outcome). Every teleport here is instrumentation that CONSTRUCTS a wrong (or partial)
outcome as a settled state and asserts the rubric REJECTS it — plus applied-torque
probes that prove the mechanism physics is real. No probe in this battery ever
reaches success(), and a final audit asserts exactly that.

  1-2.  settle/no-NaN     — reset layout settles finite: cage level on its hinge,
                            shutter covering the blocked port, ball self-centred at
                            the dish valley; score ~0 at rest, no success;
  3-4.  randomization     — READBACK over 8 seeded resets: BOTH open sides occur and
                            the shutter TRACKS the blocked port; stand yaw/xy jitter
                            is physically posed and varies; the ball start x varies
                            (instant readback) and then SELF-CENTRES (the transient
                            is the point: a level cage re-centres the ball);
  5.   null policy        — 240 idle steps -> score ~0, no success;
  6.   seed strategy N/A  — rlbench/basketball_in_hoop's verb is grasp-carry-drop.
                            That family does not exist here BY CONSTRUCTED PREMISE:
                            the ball (88 mm) exceeds the Franka jaw span (80 mm) AND
                            is sealed behind sub-ball slots — readback from the live
                            cfg; documented N/A;
  7-8.  wrong paddle      — a real press torque on the BLOCKED side swings the cage
                            to its -15 deg stop (non-vacuous: tilt readback), but
                            the red shutter retains the ball OVER the dish (no exit,
                            no latch, score ~0); on release the cage swings level
                            and the ball re-centres — recoverable, still score ~0;
  9.   nudge self-return  — ball shoved 0.35 m/s toward the OPEN port climbs the
                            dish, turns around and re-centres (the dish is a real
                            adversary: only a held tilt dispenses), score ~0;
  10.  wrong-side bin     — ball settled in the BLOCKED side's bin (canonical -x)
                            -> rejected by the folded window, score ~0;
  11.  ground rest        — ball on the ground beside the stand -> below the bin
                            height window, score ~0;
  12.  side-wall perch    — ball balanced on a bin side wall top reads above/outside
                            the bin window -> rejected, removed before it topples;
  13.  settle gate        — ball INSIDE the open bin geometrically but still moving
                            (shoved + spun) is NOT success; removed before settling;
  14-15. latches + cap    — all three stage latches constructed honestly with no
                            prefix ever satisfying the goal (ball parked OUTSIDE
                            first, then a real press to the stop, then moving
                            fly-through probes): score == 0.60 cap (float32 + eps),
                            NOT success; parking the ball on open ground leaves the
                            latched credit unchanged;
  16.  rejection audit    — success() was never True at ANY judged point;
  17.  final no-NaN       — all task-object states finite at the end;
  18.  camera             — >= 20 rgb frames captured -> frames.npz.

Run (forge): python -u -m simgen_tasks.basketball_in_hoop_i225.smoke --headless
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
    env = ENVS.get("simgen.tilt_dispenser")().build(num_envs=args.num_envs,
                                                    device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_rows = torch.zeros(n, 1, 3, device=device)
    r = c.ball_r
    a = math.radians(c.dish_deg)
    valley_rest_z = c.hinge_h + c.dish_valley_z + r / math.cos(a)  # two-plate rest
    bin_rest_z = c.bin_floor_top + r

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.25, -1.25, 0.95)) + o),
                                tuple(np.array((0.0, 0.0, 0.22)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video",
              flush=True)

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

    def report(tag: str) -> None:
        s, ok = judge()
        p = scene.ball_canon()[0]
        print(f"[smoke] {tag:16s} | side={float(scene.side[0]):+.0f}"
              f" tilt_c={float(scene.tilt_canon_deg()[0]):+.2f}"
              f" canon=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f})"
              f" latches=({float(scene.tilt_latch[0]):.0f},"
              f"{float(scene.exit_latch[0]):.0f},{float(scene.bin_latch[0]):.0f})"
              f" in_bin={bool(scene.in_open_bin()[0])}"
              f" score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_canon(canon_xyz, vel=None, ang=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement at a CANONICAL-frame point (instrumentation,
        not a solution) + REAL physics steps before judging (the zero-step trap).
        The canonical->world transform reads the live stand pose, so probes track
        side/yaw/jitter automatically."""
        canon = torch.tensor(canon_xyz, device=device).view(1, 3).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.canon_to_world(canon.clone())
        st[:, 3] = 1.0
        if vel is not None:
            st[:, 7:10] = torch.tensor(vel, device=device)
        if ang is not None:
            st[:, 10:13] = torch.tensor(ang, device=device)
        scene.ball.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def press(tau: float) -> None:
        """Paddle-press stand-in: torque about the cage's body-frame y (hinge axis)."""
        t_rows = zero_rows.clone()
        t_rows[:, 0, 1] = tau
        scene.cage.set_external_force_and_torque(zero_rows, t_rows, env_ids=all_ids)

    def clear_press() -> None:
        scene.cage.set_external_force_and_torque(zero_rows, zero_rows, env_ids=all_ids)

    def panel_canon_x() -> float:
        """Canonical x of the shutter panel centre (must sit at the BLOCKED port,
        canonical -x, every episode)."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        off = torch.tensor([[-c.panel_x, 0.0,
                             (c.panel_z0 + c.panel_z1) / 2]], device=device)
        w = scene.shutter.data.root_pos_w + quat_apply(
            scene.shutter.data.root_quat_w, off.expand(n, 3))
        loc = quat_apply_inverse(scene.stand.data.root_quat_w,
                                 w - scene.stand.data.root_pos_w)
        return float((scene.side * loc[:, 0])[0])

    def layout_sane(tag: str, ball_x_tol: float = 0.06) -> bool:
        """Reset honesty: stand posed at the workspace (within jitter), cage level
        on the hinge, shutter covering the blocked port, ball in the dish."""
        sp = (scene.stand.data.root_pos_w - scene.env_origins)[0]
        p = scene.ball_canon()[0]
        tilt = float(scene.tilt_deg()[0])
        px = panel_canon_x()
        ok = (float(sp[0]) ** 2 + float(sp[1]) ** 2 <= (c.pos_jitter + 0.012) ** 2
              and abs(tilt) < 2.5
              and px < -(c.panel_x - 0.02)
              and abs(float(p[0])) <= ball_x_tol
              and abs(float(p[1])) <= 0.03
              and abs(float(p[2]) - valley_rest_z) < 0.02
              and not bool(scene.in_open_bin()[0]))
        if not ok:
            print(f"[smoke] layout sanity VIOLATION at {tag}: stand={sp} ball={p} "
                  f"tilt={tilt:.2f} panel_cx={px:.3f}", flush=True)
        return ok

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(150)
    report("reset")
    bodies = (scene.stand, scene.cage, scene.shutter, scene.ball)
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("settle: all states finite; cage level on its hinge, shutter covering the "
          "blocked port, ball self-centred in the dish valley",
          fin and layout_sane("reset", ball_x_tol=0.03))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    sane = True
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(2)
        bx_instant = float(scene.ball_canon()[0][0])  # spawn jitter, pre-roll
        step(88)
        sane = sane and layout_sane(f"seed {sd}")
        q = scene.stand.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        sp = (scene.stand.data.root_pos_w - scene.env_origins)[0]
        p = scene.ball_canon()[0]
        reads.append((float(scene.side[0]), yaw, float(sp[0]), float(sp[1]),
                      bx_instant, float(p[0])))
    arr = np.array(reads)
    print("[smoke] randomization readback (side, yaw, x, y, ball_x0, ball_x_end):\n"
          f"{np.round(arr, 4)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    sides = {float(x) for x in arr[:, 0]}
    check("randomization: BOTH open sides occur, the shutter tracks the blocked "
          f"port every episode, and the stand yaw/xy jitter is physically posed and "
          f"varies (readback: {len(sides)} sides, yaw spread "
          f"{math.degrees(spread[1]):.1f} deg, xy spread ({spread[2]:.3f},"
          f"{spread[3]:.3f}))",
          len(sides) == 2 and spread[1] > 0.05
          and (spread[2] > 0.008 or spread[3] > 0.008) and sane)
    check("randomization: ball start x varies in the dish (instant readback spread "
          f"{spread[4]:.3f} m) and then SELF-CENTRES (all |x_end| <= 0.06, the "
          "level-cage re-centring premise)",
          spread[4] > 0.02 and float(np.abs(arr[:, 5]).max()) <= 0.06)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 6. seed strategy N/A (by constructed premise) ==============
    # rlbench/basketball_in_hoop's verb is grasp-carry-drop. No grasp end-state can
    # be constructed here: the ball exceeds the Franka jaw span AND is sealed behind
    # sub-ball slots. Readback from the LIVE cfg — documented N/A.
    slot = c.roof_zbot - c.wall_top
    check("seed strategy (grasp-carry-drop) N/A by premise: ball diameter "
          f"{2 * r * 1000:.0f} mm > jaw span {c.jaw_span * 1000:.0f} mm + 5 mm AND "
          f"the cage slots ({slot * 1000:.0f} mm) cannot pass the ball — it can "
          "never be grasped or even touched",
          2 * r > c.jaw_span + 0.005 and slot < 2 * r - 0.015)

    # =========================== 7-8. wrong paddle: shutter retains, recoverable ============
    env.reset(seed=41)
    step(120)
    tau_wrong = -float(scene.side[0]) * c.press_torque
    tilt_min, ball_x_min = 0.0, 0.0
    for _ in range(480):  # press the BLOCKED side's paddle for 4 s
        press(tau_wrong)
        env.step(no_action)
        tilt_min = min(tilt_min, float(scene.tilt_canon_deg()[0]))
        ball_x_min = min(ball_x_min, float(scene.ball_canon()[0][0]))
    report("wrong-press")
    s, ok = judge()
    panel_stop = -(c.panel_x - c.panel_t / 2 - r)  # shutter-limited ball centre
    check("wrong paddle: the press is REAL (canonical tilt reached "
          f"{tilt_min:.1f} deg <= -{c.tilt_stop_deg - 1:.0f}) but the red shutter "
          f"retains the ball over the dish (ball canonical x min {ball_x_min:+.3f} "
          f">= {panel_stop - 0.012:+.3f}) — no exit, no latch, score ~0",
          tilt_min <= -(c.tilt_stop_deg - 1.0)
          and ball_x_min >= panel_stop - 0.012
          and s <= 0.02 and not ok)
    clear_press()
    step(480)  # release: keel swings the cage level, ball re-centres
    report("wrong-release")
    p = scene.ball_canon()[0]
    s, ok = judge()
    check("wrong paddle release: recoverable — the cage swings back level (tilt "
          f"{float(scene.tilt_deg()[0]):+.2f} deg) and the dish re-centres the ball "
          f"(|x|={abs(float(p[0])):.3f} <= 0.05), still score ~0",
          abs(float(scene.tilt_deg()[0])) < 2.5 and abs(float(p[0])) <= 0.05
          and abs(float(p[2]) - valley_rest_z) < 0.02 and s <= 0.02 and not ok)

    # =========================== 9. nudge toward the open port self-returns =================
    env.reset(seed=51)
    step(120)
    st = scene.ball.data.root_state_w.clone()
    from isaaclab.utils.math import quat_apply

    dir_open = torch.zeros(n, 3, device=device)
    dir_open[:, 0] = scene.side  # canonical +x -> local
    st[:, 7:10] = quat_apply(scene.stand.data.root_quat_w, dir_open) * 0.35
    st[:, 10:13] = 0.0
    scene.ball.write_root_state_to_sim(st, all_ids)
    x_max = 0.0
    for _ in range(300):
        step(1)
        x_max = max(x_max, float(scene.ball_canon()[0][0]))
    step(120)
    report("nudge-return")
    p = scene.ball_canon()[0]
    s, ok = judge()
    check("nudge self-return: ball shoved 0.35 m/s toward the OPEN port climbs the "
          f"dish (x max {x_max:+.3f} >= 0.03, non-vacuous) but turns around and "
          f"re-centres (|x_end|={abs(float(p[0])):.3f} <= 0.05) — only a HELD tilt "
          "dispenses; score ~0",
          x_max >= 0.03 and abs(float(p[0])) <= 0.05 and s <= 0.02 and not ok)

    # =========================== 10. wrong-side bin =========================================
    env.reset(seed=61)
    step(30)
    place_canon((-0.29, 0.0, bin_rest_z + 0.002), settle_steps=90)
    report("wrong-bin")
    p = scene.ball_canon()[0]
    s, ok = judge()
    check("wrong-side bin: ball settled in the BLOCKED side's bin reads canonical "
          f"x={float(p[0]):+.3f} < 0 -> rejected by the folded window (side must be "
          "READ, not memorized), score ~0",
          float(p[0]) < -c.bin_x_lo and not bool(scene.in_open_bin()[0])
          and s <= 0.02 and not ok)

    # =========================== 11. ground rest ============================================
    env.reset(seed=71)
    step(30)
    place_canon((0.0, 0.35, r + 0.002), settle_steps=60)
    report("ground-ball")
    p = scene.ball_canon()[0]
    s, ok = judge()
    check("ground rest: ball on the floor beside the stand reads canonical "
          f"z={float(p[2]) * 1000:.0f} mm < {c.bin_z_lo * 1000:.0f} mm -> below the "
          "bin height window, score ~0",
          float(p[2]) < c.bin_z_lo and not bool(scene.in_open_bin()[0])
          and s <= 0.02 and not ok)

    # =========================== 12. side-wall perch ========================================
    env.reset(seed=81)
    step(30)
    place_canon((0.29, c.bin_half_y + c.bin_wall_t / 2,
                 c.bin_side_top + r + 0.001), settle_steps=4)
    p = scene.ball_canon()[0]
    report("wall-perch")
    _s, ok = judge()
    perch_ok = (float(p[2]) > c.bin_z_hi and abs(float(p[1])) > c.bin_y
                and not bool(scene.in_open_bin()[0]) and not ok)
    place_canon((0.0, 0.35, r + 0.002), settle_steps=20)  # remove before it topples
    check("side-wall perch: ball balanced on the bin side wall top reads canonical "
          f"z={float(p[2]) * 1000:.0f} mm > {c.bin_z_hi * 1000:.0f} mm and "
          f"|y|={abs(float(p[1])):.3f} > {c.bin_y:.3f} -> rejected by the bin "
          "window", perch_ok)

    # =========================== 13. settle gate ============================================
    env.reset(seed=91)
    step(30)
    place_canon((0.29, 0.0, bin_rest_z + 0.002),
                vel=(0.30, 0.0, 0.0), ang=(0.0, 0.0, 9.0), settle_steps=2)
    v_now = float(scene.ball.data.root_lin_vel_w[0].norm())
    w_now = float(scene.ball.data.root_ang_vel_w[0].norm())
    report("settle-gate")
    _s, ok = judge()
    gate_ok = (bool(scene.in_open_bin()[0]) and (v_now > c.settle_lin
                                                 or w_now > c.settle_ang) and not ok)
    # remove it BEFORE it can settle in the bin (this battery must never succeed)
    place_canon((0.0, 0.35, r + 0.002), settle_steps=30)
    check("settle gate: ball INSIDE the open bin geometrically but moving "
          f"(|v|={v_now:.2f} m/s, |w|={w_now:.1f} rad/s) is NOT success (velocity "
          "gates are real); removed before it can settle", gate_ok)

    # =========================== 14-15. latch construction + the 0.60 cap ===================
    # Order so no prefix ever satisfies the goal: the ball is parked OUTSIDE the
    # mechanism BEFORE the press (an empty cage tilting cannot dispense anything),
    # and the exit/bin probes are moving fly-throughs, removed before settling.
    env.reset(seed=101)
    step(60)
    place_canon((0.0, 0.35, r + 0.002), settle_steps=30)  # park the ball outside
    tau_right = float(scene.side[0]) * c.press_torque
    tilt_max = 0.0
    for _ in range(300):  # real press on the OPEN side's paddle, empty cage
        press(tau_right)
        env.step(no_action)
        tilt_max = max(tilt_max, float(scene.tilt_canon_deg()[0]))
    clear_press()
    step(240)  # cage swings back level
    place_canon((0.30, 0.0, 0.20), settle_steps=2)  # exit-window fly-through
    place_canon((0.29, 0.0, bin_rest_z + 0.002),
                vel=(0.30, 0.0, 0.0), ang=(0.0, 0.0, 9.0),
                settle_steps=2)  # bin-window fly-through, moving
    place_canon((0.0, 0.35, r + 0.002), settle_steps=60)  # park before it settles
    report("part-way")
    s_cap, ok = judge()
    lat = (float(scene.tilt_latch[0]), float(scene.exit_latch[0]),
           float(scene.bin_latch[0]))
    check("part-way + cap: all three stage latches constructed honestly (press "
          f"readback tilt max {tilt_max:.1f} deg; latches={lat}) with the ball "
          f"parked on open ground -> score == 0.60 cap ({s_cap:.4f}, float32 + "
          "eps), NOT success — full credit short of success is impossible",
          tilt_max >= c.tilt_latch_deg + 1.0 and lat == (1.0, 1.0, 1.0)
          and 0.595 <= s_cap <= 0.60 + 1e-5 and not ok)
    step(90)
    s_after, ok = judge()
    p = scene.ball_canon()[0]
    check("latched credit: 90 further idle steps with the ball parked on the "
          f"ground leave the latched score unchanged ({s_cap:.4f} -> "
          f"{s_after:.4f}) while nothing re-satisfies any window",
          abs(s_after - s_cap) < 1e-3 and not bool(scene.in_open_bin()[0])
          and not ok)

    # =========================== 16-17. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== 18. camera + save + verdict ================================
    check(f"camera: >= 20 rgb frames captured ({len(frames)})", len(frames) >= 20)
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.tilt_dispenser")
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

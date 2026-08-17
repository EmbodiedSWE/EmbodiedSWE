"""Smoke / rubric-REJECTION battery for BayonetDrawerScene (sim_gen task
`close_drawer_i386`) — NullRobot, force-driven probes + teleported constructs,
RECORDED.

This is NOT a solution (solve.py — square the T-handle, press the drawer home,
twist a quarter turn, release — is the acceptance evidence; it passes on forge
seeds 0/1/2). Every probe here CONSTRUCTS a wrong (or partial) outcome and asserts
the rubric REJECTS it, or proves a mechanism is real, not a prop. Force probes use
the solve's own regulated servos (constant-force probes walk latches) and assert
the actuator actually moved (no vacuous rejections). No probe in this battery ever
reaches success(), and a final audit check asserts exactly that.

   1. settle/no-NaN     — reset settles finite; the DEAD-MAN is real: the drawer
                          spawned part-open GLIDES FULLY OPEN on its own (readback
                          q -> the open stop), everything settled;
   2. fresh reset       — score exactly ~0 (spawns start beyond the closure-credit
                          ramp and glide AWAY from it), no success;
   3. randomization A   — READBACK over 6 seeded resets: station yaw and xy jitter
                          are real (spreads asserted);
   4. randomization B   — immediate post-reset readback: drawer opening q0 varies,
                          rotor misalignment theta0 varies in magnitude AND sign;
   5. null policy       — 240 idle steps: drawer back on the open stop, score ~0;
   6. SEED strategY     — the seed family's whole strategy (push the drawer shut,
                          let go), done for real: rotor squared, drawer pressed to
                          the closed band with the solve's own force servo, held
                          still, then released -> it GLIDES BACK OPEN; NOT success,
                          score <= 0.31 (closure+pressed latches only, and the
                          latched credit survives the re-opening);
   7. pre-twist trap    — rotor twisted to ~90 deg while OPEN, then a real press:
                          the vertical bar JAMS on the bezel plate ~25 mm short
                          (q readback ~ q_trap, far outside closed_tol); the locked
                          latch never fires (gate needs q inside the band), press
                          moved the drawer >8 cm so the jam is non-vacuous;
   8. under-twist snag  — squared, pressed home, twisted only ~50 deg (> pass
                          angle, < lock_min), released: the bar SNAGS the plate and
                          the drawer IS mechanically retained near q_rest — but the
                          rubric REFUSES it (twist < lock_min): NOT success,
                          score <= 0.31;
   9. wrong-direction   — same press, twisted the WRONG way to ~-24 deg: also
                          mechanically retained, also REFUSED;
  10. locked-open decoy — rotor twisted past lock_min while the drawer sits OPEN on
                          its stop: theta reads locked but q is at the stop, so the
                          locked latch (q-gated) never fires; score ~0, NOT success;
  11. settle gate       — the linkage placed INTO the locked goal pose WHILE STILL
                          MOVING (a real opening velocity written with the pose — a
                          zero-velocity teleport would leave the stillness counter
                          running, the teleport-vacuous trap): q and theta both pass
                          but the motion resets the stillness streak -> NOT success
                          (latches may fire; score <= 0.60 cap, never 1.0); the
                          construct is removed before it can settle (the battery
                          never succeeds);
  12. rejection audit   — success() was never True at ANY judged point;
  13. final no-NaN      — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.close_drawer_i386.smoke --headless
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

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:  # older isaaclab name
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

robobench.discover()
try:
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# The solve's own regulated servo gains (constant-force probes walk latches).
R_KP, R_KD, R_TMAX = 0.020, 0.004, 0.05
D_KP, D_VCAP, D_KV = 3.0, 0.12, 30.0
D_FMIN, D_FMAX = -12.0, 5.0
G = 9.81


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.twistlock_drawer_i386")().build(
        num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)
    lock_min = math.radians(c.lock_min_deg)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.85, -0.70, 0.55)) + o),
                                tuple(np.array((-0.05, 0.00, 0.14)) + o),
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

    def q0() -> float:
        return float(scene.drawer_q()[0])

    def th_deg() -> float:
        return math.degrees(float(scene.rotor_theta()[0]))

    def station_yaw() -> float:
        ex = torch.zeros(n, 3, device=device)
        ex[:, 0] = 1.0
        x_w = scene_mod._qapply(scene.station.data.root_quat_w, ex)[0]
        return float(torch.atan2(x_w[1], x_w[0]))

    def report(tag: str) -> None:
        s, ok = judge()
        print(f"[smoke] {tag:16s} | q={q0():+.4f} theta={th_deg():+.1f}deg "
              f"| latches c/p/l={float(scene.closure_latch[0]):.2f}/"
              f"{float(scene.pressed_latch[0]):.0f}/{float(scene.locked_latch[0]):.0f} "
              f"| settled={bool(scene.settled()[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    # ----- probe actuators: the solve's own regulated servos -------------------------------
    def rotor_hold(theta_ref: float) -> None:
        th = scene.rotor_theta()
        w_body = quat_apply_inverse(scene.rotor.data.root_quat_w,
                                    scene.rotor.data.root_ang_vel_w)
        tau = (R_KP * (theta_ref - th) - R_KD * w_body[:, 0]).clamp(-R_TMAX, R_TMAX)
        t = torch.zeros(n, 1, 3, device=device)
        t[:, 0, 0] = tau
        scene.rotor.set_external_force_and_torque(zero_w, t, env_ids=all_ids)

    def drawer_press(q_ref: float) -> None:
        q = scene.drawer_q()
        v_body = quat_apply_inverse(scene.drawer.data.root_quat_w,
                                    scene.drawer.data.root_lin_vel_w)
        v_des = (D_KP * (q_ref - q)).clamp(-D_VCAP, D_VCAP)
        fx = (-c.drawer_mass * G * math.sin(c.pitch)
              + D_KV * (v_des - v_body[:, 0])).clamp(D_FMIN, D_FMAX)
        f = torch.zeros(n, 1, 3, device=device)
        f[:, 0, 0] = fx
        scene.drawer.set_external_force_and_torque(f, zero_w, env_ids=all_ids)

    def wrenches_off() -> None:
        scene.rotor.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
        scene.drawer.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)

    def twist_to(theta_ref_deg: float, max_steps: int = 400,
                 press: bool = False) -> None:
        tr = math.radians(theta_ref_deg)
        streak = 0
        for _ in range(max_steps):
            rotor_hold(tr)
            if press:
                drawer_press(0.0005)
            env.step(no_action)
            streak = streak + 1 if abs(float(scene.rotor_theta()[0]) - tr) \
                < math.radians(2.0) else 0
            if streak >= 20:
                break

    def press_home(max_steps: int = 600, hold_deg: float = 0.0) -> int:
        """Square-held press toward the closed stop; returns servo steps used."""
        streak, i = 0, 0
        for i in range(max_steps):
            rotor_hold(math.radians(hold_deg))
            drawer_press(0.0005)
            env.step(no_action)
            q = float(scene.drawer_q()[0])
            v = float(scene.drawer.data.root_lin_vel_w[0].norm())
            streak = streak + 1 if (q <= 0.004 and v < 0.03) else 0
            if streak >= 20:
                break
        return i + 1

    def place_linkage(q: float, theta_deg: float, settle_steps: int = 0,
                      vx: float = 0.0) -> None:
        """Teleport the WHOLE drawer+rotor linkage consistently along the rails of
        the station's CURRENT pose (instrumentation — teleporting one body of a
        linkage gets depenetrated back by the joint). `vx` writes a real rail-axis
        velocity onto both bodies (a zero-velocity teleport leaves the stillness
        counter running — the teleport-vacuous trap)."""
        p_st = scene.station.data.root_pos_w.clone()
        q_st = scene.station.data.root_quat_w.clone()
        th = math.radians(theta_deg)
        qx = torch.tensor([math.cos(th / 2), math.sin(th / 2), 0.0, 0.0],
                          device=device).expand(n, 4)
        ex = torch.zeros(n, 3, device=device)
        ex[:, 0] = 1.0
        v_w = scene_mod._qapply(q_st, ex) * float(vx)
        off = torch.zeros(n, 3, device=device)
        off[:, 0] = c.closed_x + q
        off[:, 2] = c.axis_z
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = p_st + scene_mod._qapply(q_st, off)
        st[:, 3:7] = q_st
        st[:, 7:10] = v_w
        scene.drawer.write_root_state_to_sim(st, all_ids)
        off[:, 0] = q
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = p_st + scene_mod._qapply(q_st, off)
        st[:, 3:7] = scene_mod._qmul(q_st, qx)
        st[:, 7:10] = v_w
        scene.rotor.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    bodies = (scene.station, scene.drawer, scene.rotor)

    # =========================== 1-2. settle / dead-man / fresh =============================
    env.reset(seed=11)
    step(150)
    report("reset")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("settle: all states finite; the DEAD-MAN is real — the drawer spawned "
          f"part-open glided FULLY OPEN on its own (readback q={q0():+.4f} ~ open "
          f"stop {c.stroke:.3f}), everything settled",
          fin and q0() >= c.stroke - 0.010 and bool(scene.settled()[0]))
    s, ok = judge()
    check("fresh reset: score ~0 (spawns start beyond the closure-credit ramp and "
          "glide AWAY), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(2)  # refresh buffers; the drawer moves < 1 mm in 2 steps from rest
        px, py = (float(v) for v in
                  (scene.station.data.root_pos_w[0, :2] - scene.env_origins[0, :2]))
        reads.append((station_yaw(), px, py, q0(), th_deg()))
        print(f"[smoke] seed {sd}: yaw={reads[-1][0]:+.2f} station=({px:+.3f},"
              f"{py:+.3f}) q0={reads[-1][3]:+.4f} theta0={reads[-1][4]:+.1f}deg",
              flush=True)
    yaws = [r[0] for r in reads]
    xs = [r[1] for r in reads]
    ys = [r[2] for r in reads]
    check("randomization A: station yaw varies (readback spread "
          f"{max(yaws) - min(yaws):.2f} rad) and xy jitter is real (x spread "
          f"{(max(xs) - min(xs)) * 1000:.0f} mm, y spread "
          f"{(max(ys) - min(ys)) * 1000:.0f} mm)",
          max(yaws) - min(yaws) > 1.5 and max(xs) - min(xs) > 0.01
          and max(ys) - min(ys) > 0.01)
    q0s = [r[3] for r in reads]
    ths = [r[4] for r in reads]
    check("randomization B: drawer opening q0 varies (readback spread "
          f"{(max(q0s) - min(q0s)) * 1000:.0f} mm) and rotor misalignment theta0 "
          f"varies in magnitude AND sign (readback {[f'{t:+.0f}' for t in ths]})",
          max(q0s) - min(q0s) > 0.015 and max(ths) > 4.0 and min(ths) < -4.0
          and max(abs(t) for t in ths) - min(abs(t) for t in ths) > 3.0)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: 240 idle steps — the drawer is back on the open stop "
          f"(q={q0():+.4f}), score ~0, no success",
          q0() >= c.stroke - 0.010 and s <= 0.02 and not ok)

    # =========================== 6. SEED strategy scores ~0.30, not success ================
    env.reset(seed=41)
    step(60)
    used = press_home()
    q_pressed = q0()
    wrenches_off()
    step(240)  # let go — the dead-man glides it back open
    report("seed-strategy")
    s, ok = judge()
    check("SEED strategy (push shut, let go), done with the solve's own servos: the "
          f"drawer WAS genuinely pressed home ({used} servo steps, q reached "
          f"{q_pressed:+.4f} <= closed_tol) but released it GLIDES BACK OPEN "
          f"(q={q0():+.4f} >= 0.05): NOT success, score <= 0.31, and the latched "
          "closure+pressed credit survives the re-opening",
          q_pressed <= c.closed_tol and q0() >= 0.05
          and float(scene.pressed_latch[0]) > 0.5
          and float(scene.locked_latch[0]) < 0.5
          and 0.28 <= s <= 0.31 and not ok)

    # =========================== 7. pre-twist trap arrests the press ========================
    env.reset(seed=51)
    step(60)
    q_open = q0()
    twist_to(90.0)  # twist while OPEN — the decoy order
    th_open = th_deg()
    # now press for real, holding the twist (fixed budget; it must jam, not close)
    q_min = 1.0
    for _ in range(450):
        rotor_hold(math.radians(90.0))
        drawer_press(0.0005)
        env.step(no_action)
        q_min = min(q_min, float(scene.drawer_q()[0]))
    wrenches_off()
    step(90)
    report("pre-twist-trap")
    s, ok = judge()
    check("pre-twist trap: rotor twisted to "
          f"{th_open:+.1f} deg while open, then a REAL press — the vertical bar "
          f"jams on the bezel plate (q never below {q_min:+.4f} ~ q_trap "
          f"{c.q_trap:+.4f}, far outside closed_tol {c.closed_tol:.3f}); the press "
          f"moved the drawer {(q_open - q_min) * 1000:.0f} mm so the jam is "
          "non-vacuous; the locked latch (q-gated) never fires; NOT success",
          th_open >= 80.0 and q_open - q_min > 0.08
          and c.q_trap - 0.006 <= q_min <= c.q_trap + 0.007
          and q_min > c.closed_tol + c.lock_q_slack
          and float(scene.locked_latch[0]) < 0.5
          and s <= 0.16 and not ok)

    # =========================== 8. under-twist snag is REFUSED =============================
    env.reset(seed=61)
    step(60)
    press_home()
    twist_to(50.0, press=True)  # > pass angle, < lock_min
    wrenches_off()
    th_max = -180.0
    for _ in range(240):
        env.step(no_action)
        th_max = max(th_max, th_deg())
    report("under-twist")
    s, ok = judge()
    check("under-twist snag: pressed home then twisted only ~50 deg (> pass angle "
          f"{c.pass_deg:.0f}, < lock_min {c.lock_min_deg:.0f}) and released — the "
          f"bar snags the plate and the drawer IS mechanically retained "
          f"(q={q0():+.4f} < 0.05) but the rubric REFUSES it (theta never passed "
          f"{th_max:+.1f} deg): NOT success, score <= 0.31",
          q0() < 0.05 and th_max < c.lock_min_deg - 5.0
          and float(scene.locked_latch[0]) < 0.5 and s <= 0.31 and not ok)

    # =========================== 9. wrong-direction twist is REFUSED ========================
    env.reset(seed=71)
    step(60)
    press_home()
    twist_to(-24.0, press=True)  # wrong way, against the -25 deg stop
    th_wrong = th_deg()
    wrenches_off()
    step(240)
    report("wrong-direction")
    s, ok = judge()
    check("wrong-direction twist: pressed home then twisted the WRONG way "
          f"(theta={th_wrong:+.1f} deg) and released — mechanically retained "
          f"(q={q0():+.4f} < 0.05) but REFUSED (twist sign): NOT success, "
          "score <= 0.31",
          th_wrong <= -15.0 and q0() < 0.05
          and float(scene.locked_latch[0]) < 0.5 and s <= 0.31 and not ok)

    # =========================== 10. locked-open decoy ======================================
    env.reset(seed=81)
    step(60)
    twist_to(90.0)
    wrenches_off()
    step(120)
    report("locked-open")
    s, ok = judge()
    check("locked-open decoy: rotor twisted past lock_min "
          f"(theta={th_deg():+.1f} deg) while the drawer sits OPEN on its stop "
          f"(q={q0():+.4f}) — the q-gated locked latch never fires, score ~0, "
          "NOT success",
          th_deg() >= c.lock_min_deg and q0() >= c.stroke - 0.010
          and float(scene.locked_latch[0]) < 0.5 and s <= 0.02 and not ok)

    # =========================== 11. settle gate ============================================
    env.reset(seed=91)
    step(60)
    # place the locked pose WITH a real closing velocity: the pose passes the q and
    # theta gates but the motion resets the stillness streak (a zero-velocity
    # teleport would leave the counter running — proven vacuous on the forge).
    place_linkage(c.q_rest - 0.0005, 90.0, vx=-0.12)
    step(2)
    q_now, th_now = q0(), th_deg()
    v_now = float(scene.drawer.data.root_lin_vel_w[0].norm())
    settled_now = bool(scene.settled()[0])
    s, ok = judge()
    report("settle-gate")
    # remove the construct BEFORE it can settle (the battery must never succeed)
    place_linkage(0.11, 0.0)
    step(60)
    check("settle gate: the linkage placed INTO the locked goal pose while STILL "
          f"MOVING (q={q_now:+.4f} <= closed_tol, theta={th_now:+.1f} >= lock_min, "
          f"drawer speed readback {v_now:.3f} > settle_lin {c.settle_lin}) — pose "
          f"passes but stillness has not persisted (settled={settled_now}) -> NOT "
          "success, score <= 0.60 latch cap (never 1.0); construct removed before "
          "ring-down",
          q_now <= c.closed_tol and th_now >= c.lock_min_deg
          and v_now > c.settle_lin and not settled_now
          and not ok and s <= 0.601 and not bool(scene.success()[0]))

    # =========================== 12-13. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.twistlock_drawer_i386")
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

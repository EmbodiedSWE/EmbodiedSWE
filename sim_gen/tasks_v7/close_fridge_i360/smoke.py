"""Smoke battery for SpringLatchFridgeScene — REJECTION tests for the rubric, NullRobot,
RECORDED.

This is NOT a solution (the solution is solve.py — retract the bolt, press the door
flush against its return spring with a one-finger servo, throw the bolt into the keeper,
let go; the Franka strategy is TASK.md's embodiment argument). Teleported states here
are rubric INSTRUMENTATION: construct an outcome as a settled state under the scene's
live spring plant, then assert the rubric's verdict on it.

One linear run, 12 named checks:
  1. settle    — clean reset: finite state, door resting AT its sampled spring
                 equilibrium, bolt at its sampled extension (readback), score ~0;
  2. random    — theta_eq / bolt ext0 / spring k draws differ across 3 seeds
                 (max-pairwise, READBACK where physical);
  3. null      — 2 s of nothing: door stays put, score < 0.05, no success;
  4. spring    — the plant is live: a door pressed to ~20 deg and RELEASED springs back
                 to its equilibrium (an unsecured door does not stay pushed);
  5. negative A— the SEED's plan on the as-spawned scene (one uncontrolled shove until
                 the door is shut): the protruding bolt tip slams into the proud keeper
                 housing and ARRESTS the door >= 3.5 deg short of flush (outside both
                 the close latch and closed_tol); released, the spring reopens it —
                 score ~0;
  6. negative B— the seed's plan done "properly" (bolt retracted first, door pressed
                 flush, then released WITHOUT throwing the bolt): the spring reopens
                 the door during the hands-off window — no success, score caps at the
                 latched 0.60;
  7. negative C— a FULL bolt throw at a door held merely 6 deg ajar misses the keeper
                 bore entirely (swing misalignment ~0.49 m/rad vs 4 mm play) — not
                 engaged: the throw only works on a flush door;
  8. negative D— near miss: door flush, bolt thrown only ~13 mm past the mouth (below
                 `engage_min` = 18 mm) and released — under-engaged is rejected, no
                 success, score stays at the latched 0.60;
  9. exactness — constructed full goal state (door flush, bolt seated in the bore) ->
                 success() and score == 1.0;
 10. latch     — pulling the bolt back OUT of the keeper revokes success: the spring
                 swings the door open; score falls to the latched 0.60, not 1.0;
 11. negative E— fully extending the bolt at the OPEN door (throw without ever closing)
                 earns ~0: the bolt is only worth anything inside the keeper;
 12. frames    — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.close_fridge_i360.smoke --headless
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
# RTX recipe (proven on this render stack): kit mis-decodes the L20 driver version and
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
    from simgen_tasks.close_fridge_i360 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie.
threading.Timer(1200.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                 os._exit(3))).start()

# Same fingertip-scale drive as solve.py (shared numbers = shared honesty).
F_MAX = 8.0
F_RETRACT = 2.5
F_HOLD = 0.8
KV = 3.0
TAU_BIAS = 0.4
OMEGA_CAP = 0.5
K_APPROACH = 1.5
LEVER = 0.39
SLAM_TAU = 4.0  # the seed-style uncontrolled shove (~10 N at the door edge)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.spring_latch_fridge")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.60, -0.55, 0.95)) + o),
                                tuple(np.array((0.33, 0.08, 0.42)) + o),
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

    def th_deg() -> float:
        return math.degrees(float(scene.door_angle()[0]))

    def ext_mm() -> float:
        return float(scene.bolt_ext()[0]) * 1000.0

    def report(tag: str) -> None:
        print(f"[smoke] {tag:12s} | door={th_deg():6.1f}deg ext={ext_mm():5.1f}mm "
              f"engaged={bool(scene.engaged()[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def settle_until(pred, max_steps: int = 480, poll: int = 10) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    # ----- shared probe motions (solve.py's one-finger drives) -----------------------------
    def retract_bolt(budget: int = 600) -> bool:
        for _ in range(budget):
            if float(scene.bolt_ext()[0]) <= 0.005:
                break
            scene.bolt_force[0, 1] = -F_RETRACT
            step(1)
        scene.bolt_force[0] = 0.0
        step(30)
        return float(scene.bolt_ext()[0]) <= 0.008

    def press_force() -> float:
        theta = float(scene.door_angle()[0])
        w = float(scene.door_rate()[0])
        w_des = -min(OMEGA_CAP, K_APPROACH * theta)
        tau_des = float(scene.k_spring[0]) * (theta - float(scene.theta_eq[0])) \
            - TAU_BIAS + KV * (w_des - w)
        return max(0.0, min(F_MAX, -tau_des / LEVER))

    def press_flush(budget: int = 2400) -> bool:
        """Solve-style press: +x servo with a -y hold bias pinning the bolt retracted."""
        for _ in range(budget):
            if float(scene.door_angle()[0]) <= math.radians(0.3):
                return True
            scene.bolt_force[0, 0] = press_force()
            scene.bolt_force[0, 1] = -F_HOLD
            step(1)
        return False

    def bolt_vel_y() -> float:
        from isaaclab.utils.math import quat_apply_inverse

        return float(quat_apply_inverse(scene.door.data.root_quat_w,
                                        scene.bolt.data.root_lin_vel_w)[0, 1])

    def drive_hold(theta_t: float, steps: int, extra=None) -> None:
        """Hold the door at `theta_t` with the door_drive probe torque (PD + spring
        feedforward), running `extra()` each step (e.g. a bolt push)."""
        for _ in range(steps):
            theta = float(scene.door_angle()[0])
            w = float(scene.door_rate()[0])
            tq = float(scene.k_spring[0]) * (theta - float(scene.theta_eq[0])) \
                + 8.0 * (theta_t - theta) - 1.0 * w
            scene.door_drive[0] = max(-3.0, min(3.0, tq))
            if extra is not None:
                extra()
            step(1)
        scene.door_drive[0] = 0.0

    def teleport_linkage(theta: float, ext: float) -> None:
        """Rubric instrumentation: write door AND bolt together (one consistent frame,
        zero velocity) — never one body of the linkage alone."""
        st = torch.zeros(n, 13, device=device)
        hx, hy = c.hinge_xy
        arm = c.door_center_closed[1] - c.hinge_y
        st[:, 0] = hx - arm * math.sin(theta)
        st[:, 1] = hy + arm * math.cos(theta)
        st[:, 2] = c.door_center_closed[2]
        st[:, 3] = math.cos(theta / 2)
        st[:, 6] = math.sin(theta / 2)
        st[:, 0:3] += scene.env_origins
        scene.door.write_root_state_to_sim(st, all_ids)
        bs = torch.zeros(n, 13, device=device)
        lx, ly = c.bolt_local_x, c.bolt_retract_y + ext
        bs[:, 0] = st[:, 0] + lx * math.cos(theta) - ly * math.sin(theta)
        bs[:, 1] = st[:, 1] + lx * math.sin(theta) + ly * math.cos(theta)
        bs[:, 2] = st[:, 2] + c.bolt_local_z
        bs[:, 3] = math.cos(theta / 2)
        bs[:, 6] = math.sin(theta / 2)
        scene.bolt.write_root_state_to_sim(bs, all_ids)

    # ========================= 1. settle / clean-slate ========================================
    torch.manual_seed(3)
    env.reset()
    step(60)
    report("reset")
    finite = all(bool(torch.isfinite(b.data.root_state_w).all())
                 for b in (scene.door, scene.bolt)) \
        and bool(torch.isfinite(scene.score()).all())
    eq_err = abs(th_deg() - math.degrees(float(scene.theta_eq[0])))
    ext_err = abs(float(scene.bolt_ext()[0]) - float(scene.ext0[0]))
    check("clean reset: finite, door AT its spring equilibrium, bolt at ext0, score ~0",
          finite and eq_err < 2.0 and ext_err < 0.004
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 2. randomization by readback ===================================
    draws = []
    for s in (10, 11, 12):
        torch.manual_seed(s)
        env.reset()
        step(30)
        draws.append((th_deg(), float(scene.bolt_ext()[0]), float(scene.k_spring[0])))
        print(f"[smoke] seed {s}: theta={draws[-1][0]:.1f}deg "
              f"ext0={draws[-1][1] * 1000:.1f}mm k={draws[-1][2]:.3f}", flush=True)
    d_th = max(abs(a[0] - b[0]) for a in draws for b in draws)
    d_ex = max(abs(a[1] - b[1]) for a in draws for b in draws)
    d_k = max(abs(a[2] - b[2]) for a in draws for b in draws)
    check("randomization: theta_eq (door readback), bolt ext0 (readback), spring k differ",
          d_th > 2.0 and d_ex > 0.0015 and d_k > 0.05)

    # ========================= 3. null policy =================================================
    torch.manual_seed(4)
    env.reset()
    th_start = None
    step(30)
    th_start = th_deg()
    step(240)  # 2 s of nothing
    report("null")
    check("null policy: door stays at equilibrium, score < 0.05, no success",
          abs(th_deg() - th_start) < 2.0 and float(scene.score()[0]) < 0.05
          and not bool(scene.success()[0]))

    # ========================= 4. the spring is live ==========================================
    torch.manual_seed(5)
    env.reset()
    step(30)
    drive_hold(math.radians(20.0), 360)  # press the door to ~20 deg ...
    pressed_to = th_deg()
    step(480)  # ... release: the spring must take it back
    report("spring-back")
    check("spring return: a pressed-and-released door swings back to its equilibrium",
          pressed_to < 25.0
          and abs(th_deg() - math.degrees(float(scene.theta_eq[0]))) < 6.0
          and float(scene.score()[0]) < 0.05)

    # ========================= 5. negative A: the seed's slam (bolt out) ======================
    torch.manual_seed(6)
    env.reset()
    step(30)
    min_th = 999.0
    for _ in range(360):  # 3 s of uncontrolled shove
        scene.door_drive[0] = -SLAM_TAU
        step(1)
        min_th = min(min_th, th_deg())
    scene.door_drive[0] = 0.0
    report("slam-jammed")
    jam_ok = 3.5 <= min_th <= 12.0  # arrested by the tip, outside BOTH rubric bands
    step(480)  # release: spring reopens
    report("slam-after")
    check("seed slam with the bolt out: tip jams on the keeper housing >= 3.5 deg short, "
          "spring reopens, score ~0",
          jam_ok and th_deg() > 25.0 and float(scene.score()[0]) < 0.05
          and not bool(scene.success()[0]))

    # ========================= 6. negative B: push-only (retract, press, release) =============
    torch.manual_seed(7)
    env.reset()
    step(30)
    ok_r = retract_bolt()
    ok_p = press_flush()
    for _ in range(120):  # hold it flush a moment (close latch fires)
        scene.bolt_force[0, 0] = press_force()
        scene.bolt_force[0, 1] = -F_HOLD
        step(1)
    scene.bolt_force[0] = 0.0  # ... and let go WITHOUT throwing the bolt
    step(480)
    report("push-only")
    s6 = float(scene.score()[0])
    check("push-only (the seed's plan, bolt never thrown): spring reopens the door — "
          "no success, score caps at the latched 0.60",
          ok_r and ok_p and th_deg() > 25.0 and not bool(scene.success()[0])
          and 0.55 <= s6 <= 0.625)

    # ========================= 7. negative C: full throw at an ajar door ======================
    torch.manual_seed(8)
    env.reset()
    step(30)
    ok_r = retract_bolt()
    drive_hold(math.radians(6.0), 360)  # bring the door to 6 deg ajar and hold it there
    held_at = th_deg()

    def push_bolt() -> None:
        scene.bolt_force[0, 1] = 3.0

    drive_hold(math.radians(6.0), 300, extra=push_bolt)  # throw the bolt, hard, while ajar
    scene.bolt_force[0] = 0.0
    thrown_ext = float(scene.bolt_ext()[0])
    eng7 = bool(scene.engaged()[0])
    report("ajar-throw")
    step(480)  # release everything
    report("ajar-after")
    check("full throw at a 6-deg-ajar door MISSES the keeper bore (not engaged) and the "
          "door still springs open",
          ok_r and 4.0 <= held_at <= 9.0 and thrown_ext > 0.035 and not eng7
          and th_deg() > 25.0 and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.25) < 0.02)  # ONLY the retract latch
          # (earned legitimately in this check's own retract step); the 6-deg-ajar hold
          # never enters the 2-deg close band and the throw never engages -> no other credit

    # ========================= 8. negative D: under-engaged throw (near miss) =================
    torch.manual_seed(9)
    env.reset()
    step(30)
    ok_r = retract_bolt()
    ok_p = press_flush()
    target = 0.028  # tip ~13 mm past the mouth — inside the bore but < engage_min (18 mm)
    for _ in range(600):
        e = float(scene.bolt_ext()[0])
        if abs(e - target) < 0.001 and abs(bolt_vel_y()) < 0.01:
            break
        scene.bolt_force[0, 0] = press_force()
        scene.bolt_force[0, 1] = max(-2.0, min(2.0, 60.0 * (target - e) - 8.0 * bolt_vel_y()))
        step(1)
    part_ext = float(scene.bolt_ext()[0])
    scene.bolt_force[0] = 0.0  # release with the bolt only part-way home
    step(480)
    report("under-eng")
    s8 = float(scene.score()[0])
    check("under-engaged bolt (~13 mm < engage_min): no success, score stays at the "
          "latched 0.60",
          ok_r and ok_p and 0.020 <= part_ext <= 0.032 and not bool(scene.engaged()[0])
          and not bool(scene.success()[0]) and s8 <= 0.625)

    # ========================= 9. exactness: constructed goal state ===========================
    torch.manual_seed(13)
    env.reset()
    step(30)
    teleport_linkage(math.radians(0.4), 0.044)  # door flush, bolt seated in the bore
    step(90)  # settle onto the spring-vs-bolt equilibrium
    report("goal-state")
    ok9 = settle_until(lambda: bool(scene.success()[0]), max_steps=240)
    check("exactness: constructed goal state -> success() and score == 1.0",
          ok9 and abs(float(scene.score()[0]) - 1.0) < 1e-3)

    # ========================= 10. achievement latch ==========================================
    pulled = False
    for _ in range(900):
        scene.bolt_force[0, 1] = -6.0  # drag the bolt back out of the keeper
        step(1)
        if float(scene.bolt_ext()[0]) <= 0.005:
            pulled = True
            break
    scene.bolt_force[0] = 0.0
    step(480)  # the spring takes the unbolted door
    report("unbolted")
    check("achievement latch: pulling the bolt out revokes success — the spring reopens "
          "the door; the latched 0.60 remains",
          pulled and th_deg() > 25.0 and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.60) < 0.02)

    # ========================= 11. negative E: throw at the open door =========================
    torch.manual_seed(14)
    env.reset()
    step(30)
    for _ in range(300):
        if float(scene.bolt_ext()[0]) >= 0.048:
            break
        scene.bolt_force[0, 1] = 2.0
        step(1)
    scene.bolt_force[0] = 0.0
    step(240)
    report("open-throw")
    check("fully extending the bolt at the OPEN door earns ~0 (no latch, no success)",
          float(scene.bolt_ext()[0]) > 0.045 and float(scene.score()[0]) < 0.05
          and not bool(scene.success()[0]))

    # ========================= 12. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.spring_latch_fridge")
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
    try:
        main()
    except Exception as e:  # noqa: BLE001 — never leave a GPU zombie
        print(f"[smoke] CRASH: {type(e).__name__}: {e}", flush=True)
        os._exit(1)

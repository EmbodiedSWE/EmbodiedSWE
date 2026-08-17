"""solve — solution for ChannelRunScene (hand_trajectory_i200): the legitimacy certificate.

This solution uses ZERO teleports. The puck spawns in the start bay and every millimetre
of its journey is contact dynamics: a velocity-servoed horizontal force at the puck's CoM
(the applied-wrench emulation of a fingertip push) drives it along the channel FLOOR
through both 90-degree corners, then presses it under the amber roof until its centre of
mass passes the pocket lip — at which point the force is CUT and gravity alone tips and
drops the puck into the recessed pocket. The final containment, the settling, and the
success() judgement are entirely hands-off.

Force discipline (tip safety): the quasi-static tipping threshold of the standing puck
under a CoM-height horizontal force is m*g*r / (h/2) = 0.15 * 9.81 * 0.028 / 0.030
= 1.37 N; the servo cap starts at 1.1 N and stall escalation never exceeds 1.30 N, so
the puck cannot be flipped mid-channel — it can only SLIDE, exactly like a pushed puck.
The servo gain K = 6 N/(m/s) keeps K*dt/m = 0.33 < 1 (one-substep wrench delay stable);
forces are commanded in the WORLD frame (`is_global=True`) so the pod-dependent
body-frame wrench drag cannot rotate the push.

Waypoints are NOT memorized world coordinates: the route is read from the scene's own
randomized track frame (position + yaw + mirror readback via `world_to_local` /
`t_mir`), so the same code solves left- and right-handed S-channels at any jitter.

Phases (SIM_GEN_SCORE printed at each boundary, asserted non-decreasing):
  P0 reset + settle                                     -> 0.000
  P1 push the south leg (checkpoints 1, 2)              -> 0.180
  P2 push the east leg (checkpoints 3, 4)               -> 0.360
  P3 push the north leg run-up (checkpoint 5)           -> 0.450
  P4 press under the roof to local x <= -0.142, CUT the
     force; gravity tips the puck into the pocket;
     delivered latch + live success                     -> 1.000
  P5 hands-off persistence >= 3.5 simulated seconds, success() must still hold
     -> exactly `SIM_GEN_SOLVE: SUCCESS`.

Run (forge): python -u -m simgen_tasks.hand_trajectory_i200.solve --headless [--seed N]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_sec", type=float, default=1350.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.hand_trajectory_i200 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below wedges.
_wd = threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                             os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.channel_run")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    ids = torch.zeros(1, dtype=torch.long, device=device)
    zero = torch.zeros(1, 1, 3, device=device)

    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def clear_force() -> None:
        scene.puck.set_external_force_and_torque(zero, zero, env_ids=ids)
        # one step so the zero write reaches the sim before any hands-off claim
        env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        p = scene.puck_local()[0]
        v = float(scene.puck.data.root_lin_vel_w[0].norm())
        print(f"[solve] {tag:14s} local=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):.3f}) |v|={v:.3f} "
              f"cp={''.join('1' if bool(b) else '0' for b in scene.cp_latch[0])} "
              f"delivered={bool(scene.delivered[0])} in_well={bool(scene.in_well()[0])} "
              f"settled={bool(scene.settled()[0])} score={sc():.3f} "
              f"success={bool(scene.success()[0])}", flush=True)

    last_score = [-1.0]

    def phase_score(tag: str) -> float:
        s = sc()
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.3f}", flush=True)
        assert s >= last_score[0] - 1e-6, f"score decreased at {tag}: {last_score[0]} -> {s}"
        last_score[0] = s
        return s

    def verdict(ok: bool) -> None:
        print("SIM_GEN_SOLVE: SUCCESS" if ok else "SIM_GEN_SOLVE: FAIL", flush=True)
        code = 0 if ok else 1
        threading.Timer(10.0, lambda: os._exit(code)).start()
        try:
            env.close()
            app.close()
        except Exception:  # noqa: BLE001
            pass
        os._exit(code)

    def local_dir_to_world(dx: float, dy: float) -> torch.Tensor:
        """Canonical track-local direction -> world xy, applying mirror then yaw."""
        mir = float(scene.t_mir[0])
        cy, sy = float(torch.cos(scene.t_yaw[0])), float(torch.sin(scene.t_yaw[0]))
        ly = mir * dy
        return torch.tensor([cy * dx - sy * ly, sy * dx + cy * ly], device=device)

    def drive_to(tx: float, ty: float, tag: str, cut_x: float | None = None,
                 max_steps: int = 4000, v_des: float = 0.12, tol: float = 0.020) -> bool:
        """Velocity-servoed WORLD-frame planar force at the puck CoM toward the canonical
        track-local target (tx, ty). If `cut_x` is given, the force is cut the instant the
        puck's local x drops to it (the gravity-drop handoff); otherwise the drive ends
        inside `tol` of the target. Gain escalates on stall; the force cap stays below the
        1.37 N quasi-static tip threshold at all times."""
        gain, cap = 6.0, 1.10
        best, best_i = float("inf"), 0
        for i in range(max_steps):
            p = scene.puck_local()[0]
            if cut_x is not None and float(p[0]) <= cut_x:
                clear_force()
                print(f"[solve] {tag}: force CUT at local x={float(p[0]):+.3f} "
                      f"(step {i})", flush=True)
                return True
            ex, ey = tx - float(p[0]), ty - float(p[1])
            d = math.hypot(ex, ey)
            if cut_x is None and d < tol:
                clear_force()
                print(f"[solve] {tag}: reached (d={d * 1000:.0f} mm, step {i})", flush=True)
                return True
            dw = local_dir_to_world(ex / d, ey / d)
            v_want = v_des * dw
            v_now = scene.puck.data.root_lin_vel_w[0, :2]
            f_xy = gain * (v_want - v_now)
            f_mag = float(f_xy.norm())
            if f_mag > cap:
                f_xy = f_xy * (cap / f_mag)
            f = torch.zeros(1, 1, 3, device=device)
            f[0, 0, 0], f[0, 0, 1] = f_xy[0], f_xy[1]
            scene.puck.set_external_force_and_torque(f, zero, env_ids=ids, is_global=True)
            env.step(no_action)
            metric = float(p[0]) if cut_x is not None else d
            if metric < best - 0.004:
                best, best_i = metric, i
            elif i - best_i > 360:
                gain = min(gain * 1.6, 24.0)
                cap = min(cap + 0.07, 1.30)
                best_i = i
                print(f"[solve] {tag}: stall at d={d * 1000:.0f} mm -> gain={gain:.1f} "
                      f"cap={cap:.2f} N", flush=True)
        clear_force()
        print(f"[solve] {tag}: NOT reached in {max_steps} steps", flush=True)
        return False

    def settle_until(pred, max_steps: int = 900, poll: int = 10) -> bool:
        waited = 0
        while waited <= max_steps:
            if pred():
                return True
            step(poll)
            waited += poll
        return pred()

    # ================= P0: reset + settle ======================================================
    step(90)
    print(f"[solve] track readback (seed {args.seed}): "
          f"pos=({float(scene.t_pos[0, 0]):+.3f},{float(scene.t_pos[0, 1]):+.3f}) "
          f"yaw={math.degrees(float(scene.t_yaw[0])):+.1f}deg "
          f"mirror={float(scene.t_mir[0]):+.0f}", flush=True)
    report("reset")
    assert sc() <= 1e-6, f"reset score must be 0, got {sc()}"
    assert not bool(scene.success()[0]), "success at reset"
    phase_score("P0 reset")  # 0.000

    cps = c.checkpoints  # canonical local xy, route order

    # ================= P1: south leg (checkpoints 1, 2) ========================================
    ok = drive_to(*cps[0], "cp1") and drive_to(*cps[1], "cp2")
    report("south leg")
    if not (ok and bool(scene.cp_latch[0, 0]) and bool(scene.cp_latch[0, 1])):
        print("[solve] P1 FAILED: south-leg checkpoints not latched", flush=True)
        verdict(False)
    phase_score("P1 south leg")  # 0.180

    # ================= P2: east leg (checkpoints 3, 4) =========================================
    ok = drive_to(*cps[2], "cp3") and drive_to(*cps[3], "cp4")
    report("east leg")
    if not (ok and bool(scene.cp_latch[0, 2]) and bool(scene.cp_latch[0, 3])):
        print("[solve] P2 FAILED: east-leg checkpoints not latched", flush=True)
        verdict(False)
    phase_score("P2 east leg")  # 0.360

    # ================= P3: north leg run-up (checkpoint 5) =====================================
    ok = drive_to(*cps[4], "cp5")
    report("north leg")
    if not (ok and bool(scene.cp_latch.all())):
        print("[solve] P3 FAILED: full route not latched", flush=True)
        verdict(False)
    assert not bool(scene.delivered[0]), "delivered latched before the pocket press"
    phase_score("P3 north leg")  # 0.450

    # ================= P4: press under the roof, cut, gravity drop =============================
    # Push toward deep inside the pocket line; the cut at local x = -0.142 leaves the CoM
    # 7 mm past the lip (-0.135) with everything after that pure gravity + contact.
    ok = drive_to(-0.170, cps[4][1], "press", cut_x=-0.142, v_des=0.08)
    if not ok:
        print("[solve] P4 FAILED: press never crossed the lip", flush=True)
        verdict(False)
    ok = settle_until(lambda: bool(scene.success()[0]), max_steps=900)
    report("dropped")
    if not ok:
        print("[solve] P4 FAILED: puck did not come to rest inside the pocket", flush=True)
        verdict(False)
    phase_score("P4 delivered")  # 1.000

    # ================= P5: hands-off persistence >= 3.5 simulated seconds ======================
    persist_steps = int(round(3.5 / env.dt))
    step(persist_steps)
    report("final")
    phase_score("P5 final")
    still_ok = bool(scene.success()[0]) and sc() == 1.0
    print(f"[solve] persistence: {persist_steps} steps ({persist_steps * env.dt:.2f} s) "
          f"hands-off, success={still_ok}", flush=True)
    verdict(still_ok)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(2)

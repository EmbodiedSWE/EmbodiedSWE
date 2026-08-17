"""solve — solution for KeyturnSpreaderScene (approach_grasp_knife_i292).

Teleport = TRANSPORT ONLY: the lying key is teleported once, through free air, to a
hover pose above the roof slot (upright, paddle aligned with the slot's long axis).
Everything load-bearing after that is contact dynamics driven by applied wrenches on
the key:

  - DESCENT: a gravity-feedforward vertical velocity servo lowers the paddle through
    the slot (the only admitting orientation), pauses just under the roof to re-centre,
    then seats the paddle on the tunnel floor in the gap between the anchors. An
    upright PD torque (world frame) and a yaw-hold torque keep the key vertical and
    slot-aligned; a world-frame xy PD holds it over the slot centre. All gains respect
    the one-substep wrench delay (K*dt/m = 0.05..0.15 << 1; sqrt(Kp/I)*dt <= 0.11).
  - TWIST: a feedforward +z torque with a bang-bang speed governor (tau applied only
    while wz < 0.9 rad/s, escalating 0.18 -> 1.2 N*m on stall) turns the key a quarter
    turn under load; the 88 mm paddle cams both anchor blocks apart through contact to
    a ~90 mm gap (> split_gap = 80 mm). Overshoot past 90 deg is harmless by design:
    the anchors slide, never tip, and friction holds them spread.
  - RELEASE: all wrenches are cut; the anchors settle; success() = split latched during
    the armed covered run AND both anchors covered AND live gap >= 75 mm AND settled.

The judged objects (the anchors) are never touched by anything but the cam contact;
the key is never teleported after the single transport hop.

Phases (SIM_GEN_SCORE printed at each boundary, asserted non-decreasing):
  P0 reset + settle + rig readback              -> 0.000
  P1 transport teleport to the hover pose       -> 0.000
  P2 wrench-servo descent, paddle seated        -> 0.250  (key_in + keyed latches)
  P3 quarter-turn cam spread                    -> 0.850  (armed-run progress + split)
  P4 wrench cut, anchors settle, success        -> 1.000
  P5 hands-off persistence >= 3.5 sim seconds, success() must still hold
     -> exactly `SIM_GEN_SOLVE: SUCCESS`.

Run (forge): python -u -m simgen_tasks.approach_grasp_knife_i292.solve --headless [--seed N]
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
    from simgen_tasks.approach_grasp_knife_i292 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below wedges.
_wd = threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                             os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    from isaaclab.utils.math import quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.keyturn_spreader")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    ids = torch.zeros(1, dtype=torch.long, device=device)
    zero = torch.zeros(1, 1, 3, device=device)
    MG = c.key_mass * 9.81
    ez = torch.tensor([0.0, 0.0, 1.0], device=device)
    ey = torch.tensor([0.0, 1.0, 0.0], device=device)

    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def apply_wrench(fx: float, fy: float, fz: float,
                     tx: float, ty: float, tz: float) -> None:
        f = torch.tensor([[[fx, fy, fz]]], device=device)
        t = torch.tensor([[[tx, ty, tz]]], device=device)
        scene.key.set_external_force_and_torque(f, t, env_ids=ids, is_global=True)

    def clear_wrench() -> None:
        scene.key.set_external_force_and_torque(zero, zero, env_ids=ids)
        env.step(no_action)  # let the zero write reach the sim before hands-off claims

    def sc() -> float:
        return float(scene.score()[0])

    def key_root_local() -> torch.Tensor:
        return scene.world_to_local(scene.key.data.root_pos_w)[0]

    def paddle_axis() -> tuple[float, float]:
        """The paddle's long axis (key local +y) expressed in the rig-local xy plane."""
        a = quat_apply(scene.key.data.root_quat_w, ey.expand(1, 3))[0]
        cy, sy = math.cos(float(scene.r_yaw[0])), math.sin(float(scene.r_yaw[0]))
        return (cy * float(a[0]) + sy * float(a[1]),
                -sy * float(a[0]) + cy * float(a[1]))

    def twist_deg() -> float:
        ax, ay = paddle_axis()
        return math.degrees(math.atan2(abs(ax), ay))  # 0 = slot-aligned, 90 = broadside

    def yaw_dev() -> float:
        ax, ay = paddle_axis()
        return math.atan2(-ax, ay)  # signed CCW deviation from slot alignment

    def report(tag: str) -> None:
        p = key_root_local()
        pl, pr = scene.anchors_local()
        print(f"[solve] {tag:14s} key=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):+.3f}) twist={twist_deg():5.1f}deg "
              f"gap={float(scene.gap()[0]) * 1000:5.1f}mm "
              f"ax=({float(pl[0, 0]):+.3f},{float(pr[0, 0]):+.3f}) "
              f"in={bool(scene.key_in[0])} keyed={bool(scene.keyed[0])} "
              f"split={bool(scene.split_latch[0])} run={bool(scene.run_ok[0])} "
              f"cov={bool(scene.covered_both()[0])} settled={bool(scene.settled()[0])} "
              f"score={sc():.3f} success={bool(scene.success()[0])}", flush=True)

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

    # ----- key wrench controller pieces --------------------------------------------------------
    def stab_torques(yaw_hold: bool) -> tuple[float, float, float]:
        """Upright PD about world x/y (+ optional slot-alignment yaw hold about z)."""
        u = quat_apply(scene.key.data.root_quat_w, ez.expand(1, 3))[0]
        w = scene.key.data.root_ang_vel_w[0]
        tx = 0.40 * float(u[1]) - 0.03 * float(w[0])
        ty = -0.40 * float(u[0]) - 0.03 * float(w[1])
        tx = max(-0.6, min(0.6, tx))
        ty = max(-0.6, min(0.6, ty))
        tz = 0.0
        if yaw_hold:
            tz = -0.04 * yaw_dev() - 0.004 * float(w[2])
            tz = max(-0.06, min(0.06, tz))
        return tx, ty, tz

    def xy_force(kp: float = 30.0, kd: float = 8.0, cap: float = 3.0) -> tuple[float, float]:
        """World-frame xy PD pulling the key root toward the slot centre."""
        p = scene.key.data.root_pos_w[0, :2]
        v = scene.key.data.root_lin_vel_w[0, :2]
        fx = kp * float(slot_w[0] - p[0]) - kd * float(v[0])
        fy = kp * float(slot_w[1] - p[1]) - kd * float(v[1])
        m = math.hypot(fx, fy)
        if m > cap:
            fx, fy = fx * cap / m, fy * cap / m
        return fx, fy

    def hold_at(z_t: float, steps: int) -> None:
        """Hover / re-centre: z position PD (gravity feedforward) + xy PD + yaw hold."""
        for _ in range(steps):
            z = float(key_root_local()[2])
            vz = float(scene.key.data.root_lin_vel_w[0, 2])
            fz = MG + 18.0 * (z_t - z) - 6.0 * vz
            fz = max(0.0, min(MG + 4.0, fz))
            fx, fy = xy_force()
            tx, ty, tz = stab_torques(yaw_hold=True)
            apply_wrench(fx, fy, fz, tx, ty, tz)
            env.step(no_action)

    def descend_to(z_t: float, v_des: float, tag: str, max_steps: int = 1500,
                   seat: bool = False) -> bool:
        """Vertical velocity servo downward until root z <= z_t (seat: also nearly still).
        Returns False on a stall (z not dropping for 300 steps) so the caller can lift
        and retry — a stalled seat descent means the paddle tip caught a block-top edge."""
        best_z, best_i = float("inf"), 0
        for i in range(max_steps):
            z = float(key_root_local()[2])
            vz = float(scene.key.data.root_lin_vel_w[0, 2])
            if z <= z_t and (not seat or abs(vz) < 0.06):
                clear_wrench()
                print(f"[solve] {tag}: reached z={z:+.4f} (step {i})", flush=True)
                return True
            fz = MG + 6.0 * (-v_des - vz)
            fz = max(-1.0, min(MG + 3.0, fz))
            fx, fy = xy_force()
            tx, ty, tz = stab_torques(yaw_hold=True)
            apply_wrench(fx, fy, fz, tx, ty, tz)
            env.step(no_action)
            if z < best_z - 0.003:
                best_z, best_i = z, i
            elif i - best_i > 300:
                clear_wrench()
                print(f"[solve] {tag}: STALL at z={z:+.4f} (step {i})", flush=True)
                return False
        clear_wrench()
        print(f"[solve] {tag}: NOT reached in {max_steps} steps (z={z:+.4f})", flush=True)
        return False

    def twist_run(max_steps: int = 4200) -> bool:
        """Feedforward +z torque with a speed governor; escalates on stall. Ends when the
        live gap clears split_gap with margin or the paddle passes ~90 deg broadside."""
        tau, om_cap = 0.18, 0.9
        best_th, best_i = -1.0, 0
        for i in range(max_steps):
            g = float(scene.gap()[0])
            th = twist_deg()
            if g >= 0.085 or th >= 92.0:
                clear_wrench()
                print(f"[solve] twist: done at {th:.1f}deg gap={g * 1000:.1f}mm "
                      f"(step {i}, tau={tau:.2f})", flush=True)
                return True
            wz = float(scene.key.data.root_ang_vel_w[0, 2])
            tz = tau if wz < om_cap else 0.0
            fx, fy = xy_force(kp=20.0, kd=6.0, cap=2.0)
            tx, ty, _ = stab_torques(yaw_hold=False)
            apply_wrench(fx, fy, -1.0, tx, ty, tz)  # slight down-force keeps the cam seated
            env.step(no_action)
            if th > best_th + 0.5:
                best_th, best_i = th, i
            elif i - best_i > 300:
                tau = min(tau * 1.7, 1.2)
                best_i = i
                print(f"[solve] twist: stall at {th:.1f}deg gap={g * 1000:.1f}mm "
                      f"-> tau={tau:.2f} N*m", flush=True)
        clear_wrench()
        print(f"[solve] twist: NOT finished in {max_steps} steps", flush=True)
        return False

    def settle_until(pred, max_steps: int = 900, poll: int = 10) -> bool:
        waited = 0
        while waited <= max_steps:
            if pred():
                return True
            step(poll)
            waited += poll
        return pred()

    # ================= P0: reset + settle + rig readback =======================================
    step(60)
    print(f"[solve] rig readback (seed {args.seed}): "
          f"pos=({float(scene.r_pos[0, 0]):+.3f},{float(scene.r_pos[0, 1]):+.3f}) "
          f"yaw={math.degrees(float(scene.r_yaw[0])):+.1f}deg "
          f"g0={float(scene.g0[0]) * 1000:.1f}mm", flush=True)
    report("reset")
    assert sc() <= 1e-6, f"reset score must be 0, got {sc()}"
    assert not bool(scene.success()[0]), "success at reset"
    slot_w = scene.local_to_world(torch.tensor([[0.0, 0.0, 0.0]], device=device),
                                  ids)[0, :2].clone()
    phase_score("P0 reset")  # 0.000

    # ================= P1: transport teleport to the hover pose ================================
    # Upright above the roof slot: root (paddle bottom) 20 mm above the roof top, paddle
    # long axis along the slot's long axis (= rig-local y). Free-air pose, zero velocity;
    # this is the single transport hop — the key is never teleported again.
    hover_z = c.interior_h + c.roof_t + 0.020
    st = torch.zeros(1, 13, device=device)
    st[:, 0:3] = scene.local_to_world(
        torch.tensor([[0.0, 0.0, hover_z]], device=device), ids)
    half = float(scene.r_yaw[0]) / 2
    st[:, 3] = math.cos(half)
    st[:, 6] = math.sin(half)
    scene.key.write_root_state_to_sim(st, ids)
    hold_at(hover_z, 40)
    report("hover")
    assert not bool(scene.key_in[0]), "key_in latched from a hover above the roof"
    assert sc() <= 1e-6, "score moved on transport alone"
    phase_score("P1 hover")  # 0.000

    # ================= P2: wrench-servo descent through the slot, seat the paddle ==============
    ok = descend_to(0.070, 0.10, "descend-slot")  # paddle just below the roof underside
    if not ok:
        print("[solve] P2 FAILED: could not pass the slot", flush=True)
        verdict(False)
    hold_at(0.070, 60)  # re-centre with the shank captive in the slot
    seated = False
    for attempt in range(4):
        if descend_to(0.004, 0.05, f"seat[{attempt}]", max_steps=900, seat=True):
            seated = True
            break
        hold_at(0.075, 90)  # lift off the block-top edge, re-centre, retry
    report("seated")
    if not (seated and bool(scene.keyed[0])):
        print("[solve] P2 FAILED: paddle not seated in the gap", flush=True)
        verdict(False)
    phase_score("P2 seated")  # 0.250

    # ================= P3: quarter-turn cam spread =============================================
    ok = twist_run()
    report("twisted")
    if not (ok and bool(scene.split_latch[0])):
        print("[solve] P3 FAILED: split not latched", flush=True)
        verdict(False)
    phase_score("P3 split")  # 0.850

    # ================= P4: hands-off settle to success =========================================
    ok = settle_until(lambda: bool(scene.success()[0]), max_steps=900)
    report("settled")
    if not ok:
        print("[solve] P4 FAILED: success did not hold after release", flush=True)
        verdict(False)
    phase_score("P4 success")  # 1.000

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

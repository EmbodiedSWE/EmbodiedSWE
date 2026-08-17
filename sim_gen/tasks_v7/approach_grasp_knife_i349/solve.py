"""solve — solution for TimerFlipDockScene (approach_grasp_knife_i349).

Teleport = TRANSPORT ONLY: the standing capsule is teleported once, through free air, to
an INVERTED hover pose above the well (foot end down, faces aligned to the well mod 90
deg); the captive marble is re-written at its unchanged BODY-FRAME offset in the same
write, so the flip does not move it relative to the capsule. Everything load-bearing
after that is contact dynamics driven by applied wrenches on the capsule:

  - HOVER: a gravity-feedforward z position servo + world xy PD + flipped-upright PD
    torque + yaw hold (mod 90 deg, square symmetry) holds the capsule inverted over the
    well while GRAVITY makes the marble fall through the internal waist into the foot
    chamber — the physical proof of the flip (marble_across latches here).
  - DESCENT: a gravity-feedforward vertical velocity servo lowers the foot into the well
    (only admitted inverted + square-on; the collar cannot enter and the tube cannot
    enter turned 45 deg). On a stall (foot caught on the rim edge) the capsule lifts
    back to the hover, re-centres, and retries. The descent ends when the steel collar
    lands on the amber rim — the geometric stop; a brief gentle press snugs it.
  - RELEASE: all wrenches are cut; the capsule rests collar-on-rim, tube captive in the
    well; success() = live seated inverted pose AND marble in the foot chamber AND
    everything settled.

All gains respect the one-substep wrench delay (K*dt/m <= 0.35; sqrt(Kp/I)*dt <= 0.13).
The capsule is never teleported after the single transport hop; the marble is never
touched after that hop (gravity and contact do everything).

Phases (SIM_GEN_SCORE printed at each boundary, asserted non-decreasing):
  P0 reset + settle + dock readback                     -> 0.000
  P1 transport teleport to the inverted hover; marble
     falls through the waist under gravity              -> 0.400  (lifted+inverted+marble)
  P2 wrench-servo descent into the well, collar seated  -> 0.850  (tip_in + full depth)
  P3 wrench cut, capsule settles seated, success        -> 1.000
  P4 hands-off persistence >= 3.5 sim seconds, success() must still hold
     -> exactly `SIM_GEN_SOLVE: SUCCESS`.

Run (forge): python -u -m simgen_tasks.approach_grasp_knife_i349.solve --headless [--seed N]
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
    from simgen_tasks.approach_grasp_knife_i349 import scene as scene_mod  # noqa: F401
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
    env = ENVS.get("simgen.timer_flip_dock")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    ids = torch.zeros(1, dtype=torch.long, device=device)
    zero = torch.zeros(1, 1, 3, device=device)
    MG = (c.timer_mass + c.marble_mass) * 9.81
    ez = torch.tensor([0.0, 0.0, 1.0], device=device)
    ex = torch.tensor([1.0, 0.0, 0.0], device=device)
    HOVER_Z = c.rim_z + c.half_len + 0.015  # centre height: foot tip 15 mm above the rim

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
        scene.timer.set_external_force_and_torque(f, t, env_ids=ids, is_global=True)

    def clear_wrench() -> None:
        scene.timer.set_external_force_and_torque(zero, zero, env_ids=ids)
        env.step(no_action)  # let the zero write reach the sim before hands-off claims

    def sc() -> float:
        return float(scene.score()[0])

    def timer_local() -> torch.Tensor:
        return scene.world_to_local(scene.timer.data.root_pos_w)[0]

    def up_z() -> float:
        return float(scene.timer_up()[0, 2])

    def yaw_dev() -> float:
        """Signed deviation of the body x-axis from the nearest well face (mod 90 deg)."""
        a = quat_apply(scene.timer.data.root_quat_w, ex.expand(1, 3))[0]
        cy, sy = math.cos(float(scene.d_yaw[0])), math.sin(float(scene.d_yaw[0]))
        ang = math.atan2(-sy * float(a[0]) + cy * float(a[1]),
                         cy * float(a[0]) + sy * float(a[1]))
        return (ang + math.pi / 4) % (math.pi / 2) - math.pi / 4

    def report(tag: str) -> None:
        p = timer_local()
        ml = scene.marble_local()[0]
        foot, _top = scene.end_tips()
        fz = float(scene.world_to_local(foot)[0, 2])
        print(f"[solve] {tag:14s} timer=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):+.3f}) up_z={up_z():+.2f} foot_z={fz:+.3f} "
              f"marble_l=({float(ml[0]):+.3f},{float(ml[1]):+.3f},{float(ml[2]):+.3f}) "
              f"lift={bool(scene.lifted[0])} inv={bool(scene.inverted[0])} "
              f"tip={bool(scene.tip_in[0])} depth={float(scene.depth_max[0]):.2f} "
              f"mb={bool(scene.marble_across[0])} settled={bool(scene.settled()[0])} "
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

    # ----- wrench controller pieces ------------------------------------------------------------
    def stab_torques() -> tuple[float, float, float]:
        """Flipped-upright PD (drive body +z toward world DOWN) + yaw hold mod 90 deg."""
        u = quat_apply(scene.timer.data.root_quat_w, ez.expand(1, 3))[0]
        w = scene.timer.data.root_ang_vel_w[0]
        # torque = kp * cross(u, -ez) - kd * w_xy
        tx = -0.15 * float(u[1]) - 0.012 * float(w[0])
        ty = 0.15 * float(u[0]) - 0.012 * float(w[1])
        tx = max(-0.3, min(0.3, tx))
        ty = max(-0.3, min(0.3, ty))
        tz = -0.02 * yaw_dev() - 0.003 * float(w[2])
        tz = max(-0.04, min(0.04, tz))
        return tx, ty, tz

    def xy_force(kp: float = 12.0, kd: float = 4.0, cap: float = 2.0) -> tuple[float, float]:
        """World-frame xy PD pulling the capsule centre onto the well axis."""
        p = scene.timer.data.root_pos_w[0, :2]
        v = scene.timer.data.root_lin_vel_w[0, :2]
        fx = kp * float(well_w[0] - p[0]) - kd * float(v[0])
        fy = kp * float(well_w[1] - p[1]) - kd * float(v[1])
        m = math.hypot(fx, fy)
        if m > cap:
            fx, fy = fx * cap / m, fy * cap / m
        return fx, fy

    def hold_at(z_t: float, steps: int) -> None:
        """Hover / re-centre: z position PD (gravity feedforward) + xy PD + attitude."""
        for _ in range(steps):
            z = float(timer_local()[2])
            vz = float(scene.timer.data.root_lin_vel_w[0, 2])
            fz = MG + 10.0 * (z_t - z) - 4.0 * vz
            fz = max(0.0, min(MG + 3.0, fz))
            fx, fy = xy_force()
            tx, ty, tz = stab_torques()
            apply_wrench(fx, fy, fz, tx, ty, tz)
            env.step(no_action)

    def descend_to(z_t: float, v_des: float, tag: str, max_steps: int = 1200,
                   seat: bool = False) -> bool:
        """Vertical velocity servo downward until centre z <= z_t (seat: also nearly
        still). Returns False on a stall (z not dropping for 240 steps) so the caller can
        lift and retry — a stalled descent means the foot caught the rim edge."""
        z = float(timer_local()[2])
        best_z, best_i = float("inf"), 0
        for i in range(max_steps):
            z = float(timer_local()[2])
            vz = float(scene.timer.data.root_lin_vel_w[0, 2])
            if z <= z_t and (not seat or abs(vz) < 0.05):
                clear_wrench()
                print(f"[solve] {tag}: reached z={z:+.4f} (step {i})", flush=True)
                return True
            fz = MG + 4.0 * (-v_des - vz)
            fz = max(-0.5, min(MG + 2.0, fz))
            fx, fy = xy_force()
            tx, ty, tz = stab_torques()
            apply_wrench(fx, fy, fz, tx, ty, tz)
            env.step(no_action)
            if z < best_z - 0.002:
                best_z, best_i = z, i
            elif i - best_i > 240:
                clear_wrench()
                print(f"[solve] {tag}: STALL at z={z:+.4f} (step {i})", flush=True)
                return False
        clear_wrench()
        print(f"[solve] {tag}: NOT reached in {max_steps} steps (z={z:+.4f})", flush=True)
        return False

    def press(steps: int) -> None:
        """Snug the collar onto the rim: small net down-force, keep centred + inverted."""
        for _ in range(steps):
            fx, fy = xy_force(kp=8.0, kd=3.0, cap=1.0)
            tx, ty, tz = stab_torques()
            apply_wrench(fx, fy, MG * 0.5, tx, ty, tz)  # net downward ~ 0.5 * weight
            env.step(no_action)

    def settle_until(pred, max_steps: int = 900, poll: int = 10) -> bool:
        waited = 0
        while waited <= max_steps:
            if pred():
                return True
            step(poll)
            waited += poll
        return pred()

    # ================= P0: reset + settle + dock readback ======================================
    step(60)
    print(f"[solve] dock readback (seed {args.seed}): "
          f"pos=({float(scene.d_pos[0, 0]):+.3f},{float(scene.d_pos[0, 1]):+.3f}) "
          f"yaw={math.degrees(float(scene.d_yaw[0])):+.1f}deg "
          f"spawn r={float(scene.sp_r[0]):.3f} az={math.degrees(float(scene.sp_az[0])):.0f}deg "
          f"tyaw={math.degrees(float(scene.t_yaw0[0])):.0f}deg", flush=True)
    report("reset")
    assert sc() <= 1e-6, f"reset score must be 0, got {sc()}"
    assert not bool(scene.success()[0]), "success at reset"
    ml0 = scene.marble_local()[0]
    assert abs(float(ml0[2]) - c.marble_home_z) < 0.006, "marble not at its chamber home"
    well_w = scene.local_to_world(torch.tensor([[0.0, 0.0, 0.0]], device=device),
                                  ids)[0, :2].clone()
    phase_score("P0 reset")  # 0.000

    # ================= P1: transport teleport to the INVERTED hover ============================
    # Single transport hop: capsule flipped upside-down above the well (foot down, faces
    # aligned mod 90 deg), marble re-written at its unchanged body-frame offset so the
    # hop moves the ASSEMBLY coherently. Gravity then drops the marble through the waist
    # into the foot chamber while the hover servo holds the capsule — the flip's physics.
    ml = scene.marble_local()[0].clone()  # body-frame offset, preserved across the hop
    psi = float(scene.d_yaw[0])
    st = torch.zeros(1, 13, device=device)
    st[:, 0:3] = scene.local_to_world(
        torch.tensor([[0.0, 0.0, HOVER_Z]], device=device), ids)
    st[:, 3] = 0.0  # q = qz(psi) * qx(pi): 180 deg flip + dock-yaw alignment
    st[:, 4] = math.cos(psi / 2)
    st[:, 5] = math.sin(psi / 2)
    st[:, 6] = 0.0
    scene.timer.write_root_state_to_sim(st, ids)
    sm = torch.zeros(1, 13, device=device)
    sm[:, 0:3] = st[:, 0:3] + quat_apply(st[:, 3:7], ml.unsqueeze(0))
    sm[:, 3] = 1.0
    scene.marble.write_root_state_to_sim(sm, ids)
    hold_at(HOVER_Z, 240)  # ~2 s: marble falls ~125 mm through the waist and settles
    report("hover")
    assert bool(scene.marble_across[0]), "marble did not cross the waist at the hover"
    mlh = scene.marble_local()[0]
    assert c.mb_z[0] < float(mlh[2]) < c.mb_z[1], "marble not resting in the foot chamber"
    assert not bool(scene.tip_in[0]), "tip_in latched from a hover above the rim"
    phase_score("P1 hover")  # 0.400 (lifted + inverted + marble_across)

    # ================= P2: wrench-servo descent into the well, collar onto the rim =============
    seated = False
    for attempt in range(4):
        if descend_to(c.seat_z + 0.004, 0.08, f"seat[{attempt}]", seat=True):
            seated = True
            break
        hold_at(HOVER_Z, 120)  # lift off the rim edge, re-centre, retry
    if not seated:
        print("[solve] P2 FAILED: could not insert the foot into the well", flush=True)
        verdict(False)
    press(60)
    clear_wrench()
    report("seated")
    if not (bool(scene.tip_in[0]) and float(scene.depth_max[0]) > 0.95):
        print("[solve] P2 FAILED: not fully seated", flush=True)
        verdict(False)
    phase_score("P2 seated")  # 0.850

    # ================= P3: hands-off settle to success =========================================
    ok = settle_until(lambda: bool(scene.success()[0]), max_steps=900)
    report("settled")
    if not ok:
        print("[solve] P3 FAILED: success did not hold after release", flush=True)
        verdict(False)
    phase_score("P3 success")  # 1.000

    # ================= P4: hands-off persistence >= 3.5 simulated seconds ======================
    persist_steps = int(round(3.5 / env.dt))
    step(persist_steps)
    report("final")
    phase_score("P4 final")
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

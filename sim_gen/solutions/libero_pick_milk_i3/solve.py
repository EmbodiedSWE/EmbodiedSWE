"""solve — crate_turnover (libero_pick_milk_i3) with a REAL Franka arm, OSC servo loop.

Plan (base at BASE, chosen once; everything within 0.35-0.71 m reach):

  MILK OUT   top-down grasp of the carton's upper half (it pokes 70 mm above the rim;
             fingertips never go below the rim plane), jaw aligned to the carton yaw
             (mod 90), lift clear, closed-loop carry on the CARTON xy to the pad,
             lower until the base is ~4 mm off the slab, slow release -> it stands.
  ITEMS IN   for each present item (red cube / yellow can): top-down pinch at its
             center height, lift, carry over a crate-frame drop slot, release ~12 mm
             above rest height -> it drops into the open crate.
  LID ON     pinch the gray knob post, lift the board, carry it over the crate while
             rotating the wrist so the (square) lid squares up with the crate yaw
             (mod 90), lower to ~4 mm above the rim, slow release -> it seats flush.

The lid phase is executed LAST because the geometry demands it (the 140 mm carton
blocks the 70 mm seat). No task-object state is ever written; only arm + gripper
joints are commanded through the robobench OSC controller.

Prints the scene readouts and `SIM_GEN_SCORE <score>` at each phase boundary (latched
credit is monotone along this trajectory) and exactly `SIM_GEN_SOLVE: SUCCESS` once
scene.success() holds and persists.

Run:  python -m simgen_tasks.libero_pick_milk_i3.solve --headless [--seed N]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_sec", type=float, default=420.0, help="sim-time budget (s)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import torch  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    axis_angle_from_quat,
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_from_matrix,
    quat_mul,
)

import robobench  # noqa: E402
from robobench.core import EnvCfg  # noqa: E402
from robobench.robots.franka import FrankaRobotCfg  # noqa: E402

try:
    from . import scene as task_scene  # noqa: F401  (registers the scene)
except ImportError:  # direct-script fallback
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import scene as task_scene  # noqa: F401

# watchdog: never hang the forge — verdict or death
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(1))).start()

OPEN = 0.04          # per-finger target, fully open (80 mm aperture)
FINGER_LEN = 0.112   # panda_hand frame -> fingertip midpoint
BASE = (-0.42, 0.0, 0.0)  # arm base on the ground, west of the work area


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    robobench.discover()
    rcfg = FrankaRobotCfg(
        base_pos=BASE,
        nullspace_dof_pos=(),      # default posture target winds the arm on long servos
        gripper_effort_limit=120.0,
        gripper_stiffness=4000.0,
    )
    env = EnvCfg(scene="crate_turnover", robot="franka", control_mode="osc",
                 env_spacing=3.0, robot_cfg=rcfg).build(num_envs=1, device=device)
    scene, robot = env.scene, env.robot
    c = scene.cfg
    art = robot.articulation
    ee_idx = art.body_names.index("panda_hand")
    fj1 = art.find_joints(["panda_finger_joint1"])[0]
    osc = robot.controller.controllers[0]
    osc._kp = torch.tensor([220.0, 220.0, 220.0, 600.0, 600.0, 600.0], device=device)
    osc._kd = 2.0 * osc._kp.sqrt()
    osc.cfg.rot_scale = 0.15
    osc.cfg.kp_null = 3.0
    osc.cfg.kd_null = 2.0 * math.sqrt(3.0)
    n_act = robot.action_dim
    ctrl_hz = 1.0 / (env.dt * robot.control_period)
    SEC = lambda s: max(1, round(s * ctrl_hz))  # noqa: E731

    env.reset(seed=args.seed)  # seed at reset time: build consumes/reset RNG state
    origin = env.iscene.env_origins[0]
    print("[solve] describe():\n" + scene.describe(), flush=True)
    cp0 = scene.crate.data.root_pos_w[0]
    mp0 = scene.milk.data.root_pos_w[0]
    print(f"[solve] layout: crate=({cp0[0]:.3f},{cp0[1]:.3f}) "
          f"milk=({mp0[0]:.3f},{mp0[1]:.3f}) "
          f"pad=({scene.pad.data.root_pos_w[0][0]:.3f},"
          f"{scene.pad.data.root_pos_w[0][1]:.3f}) "
          f"present={scene.present[0].int().tolist()}", flush=True)

    V3 = lambda x, y, z: torch.tensor([float(x), float(y), float(z)], device=device)  # noqa: E731

    # ----- kernel (franka_session lineage) --------------------------------------------------
    def ee_pose():
        return art.data.body_pos_w[0, ee_idx], art.data.body_quat_w[0, ee_idx]

    def width() -> float:
        return 2.0 * art.data.joint_pos[0, fj1].item()

    def servo(goal_pos, goal_quat, grip, a, xy_boost: float = 1.0):
        p, q = ee_pose()
        err = goal_pos - p
        err = torch.cat([err[:2] * xy_boost, err[2:3]])
        a[0, 0:3] = (err / osc.cfg.pos_scale).clamp(-1.0, 1.0)
        qe = quat_mul(goal_quat.unsqueeze(0), quat_conjugate(q.unsqueeze(0)))
        a[0, 3:6] = (axis_angle_from_quat(qe)[0] / osc.cfg.rot_scale).clamp(-1.0, 1.0)
        a[0, 6:8] = grip

    sim_t = {"t": 0.0}

    def tick(a):
        env.step(a, render=False)
        sim_t["t"] += env.dt * robot.control_period

    def hold(pos, quat, grip, secs: float, xy_boost: float = 1.0):
        for _ in range(SEC(secs)):
            a = torch.zeros(1, n_act, device=device)
            servo(pos, quat, grip, a, xy_boost)
            tick(a)

    def run_phase(goal_fn, gate_fn, grip, timeout_s: float, tag: str = "",
                  xy_boost: float = 1.0) -> bool:
        deadline = sim_t["t"] + timeout_s
        while sim_t["t"] < deadline:
            a = torch.zeros(1, n_act, device=device)
            gp, gq = goal_fn()
            servo(gp, gq, grip, a, xy_boost)
            tick(a)
            if gate_fn():
                return True
        if tag:
            p, _ = ee_pose()
            gp, _ = goal_fn()
            print(f"[phase:{tag}] timeout: ee=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
                  f"goal=({gp[0]:.3f},{gp[1]:.3f},{gp[2]:.3f}) "
                  f"err={float((gp - p).norm()) * 1000:.0f}mm w={width() * 1000:.1f}mm",
                  flush=True)
        return False

    def close_ramp(pos, quat, grip_target: float, secs: float = 1.2):
        n = SEC(secs)
        for k in range(n):
            a = torch.zeros(1, n_act, device=device)
            f = min(1.0, (k + 1) / (n * 0.7))
            servo(pos, quat, OPEN + (grip_target - OPEN) * f, a)
            tick(a)

    def jaw_quat(azimuth: float) -> torch.Tensor:
        yh = V3(math.cos(azimuth), math.sin(azimuth), 0.0)
        zh = V3(0.0, 0.0, -1.0)
        xh = torch.cross(yh, zh, dim=0)
        return quat_from_matrix(torch.stack([xh, yh, zh], dim=1).unsqueeze(0))[0]

    def tilt_quat(azimuth: float, target_xy: torch.Tensor,
                  near: float = 0.37, max_tilt: float = 0.35) -> torch.Tensor:
        gq = jaw_quat(azimuth)
        base_xy = art.data.root_pos_w[0, :2]
        d = float((target_xy - base_xy).norm())
        if d < near:
            tilt = min(max_tilt, (near - d) * 5.0)
            u = (target_xy - base_xy) / max(d, 1e-6)
            axis = V3(-u[1], u[0], 0.0)
            gq = quat_mul(quat_from_angle_axis(
                torch.tensor([tilt], device=device), axis.unsqueeze(0))[0].unsqueeze(0),
                gq.unsqueeze(0))[0]
        return gq

    def dewind(grip: float = OPEN) -> None:
        q = art.data.joint_pos[0]
        if (abs(q[0].item()) > 2.6 or q[3].item() < -2.95 or q[3].item() > -0.15
                or abs(q[6].item()) > 2.6):
            print(f"[solve] wound arm (q1={q[0]:.2f} q4={q[3]:.2f} q7={q[6]:.2f}) — reset",
                  flush=True)
            robot.reset(torch.tensor([0], device=device, dtype=torch.long))
            p, qq = ee_pose()
            hold(p, qq, grip, 0.5)

    def yaw_of(quat_row: torch.Tensor) -> float:
        w, x, y, z = (float(v) for v in quat_row)
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def fold90(az: float) -> float:
        """Fold an angle into [-45, 45) deg (square objects: 4 equivalent jaw azimuths)."""
        return (az + math.pi / 4) % (math.pi / 2) - math.pi / 4

    # ----- scene readouts ---------------------------------------------------------------------
    score_trace: list[float] = []

    def boundary(tag: str) -> None:
        s = float(scene.score()[0])
        score_trace.append(s)
        print(f"[readout] {tag}: milk_in_crate={bool(scene.milk_in_crate()[0])} "
              f"delivered={bool(scene.delivered()[0])} "
              f"stowed={scene.stowed()[0].int().tolist()} "
              f"present={scene.present[0].int().tolist()} "
              f"lid_seated={bool(scene.lid_seated()[0])} "
              f"success={bool(scene.success()[0])}", flush=True)
        print(f"SIM_GEN_SCORE {s:.3f}", flush=True)

    # ----- generic pick / carry / place ------------------------------------------------------
    def pick_carry_place(*, name: str, obj_pos_fn, azimuth_fn, grasp_tip_z_fn, grip: float,
                         w_lo: float, w_hi: float, lift_obj_z: float, carry_hand_z: float,
                         target_xy_fn, place_obj_z: float, place_z_tol: float,
                         place_xy_tol: float, settle_pred, carry_quat_fn=None,
                         attempts: int = 3) -> bool:
        """Hover -> descend -> ramped close -> lift verdict -> closed-loop carry on the
        OBJECT -> closed-loop lower on the OBJECT -> slow release -> settle_pred()."""
        for attempt in range(attempts):
            dewind()
            az = azimuth_fn(attempt)
            gq = tilt_quat(az, obj_pos_fn()[:2])

            def tip_goal(z, gq=gq):
                o = obj_pos_fn()
                return V3(o[0], o[1], z + FINGER_LEN), gq

            hover_z = carry_hand_z - FINGER_LEN
            if not run_phase(lambda: tip_goal(hover_z),
                             lambda: (ee_pose()[0][:2] - obj_pos_fn()[:2]).norm() < 0.006,
                             OPEN, 8.0, tag=f"{name}-hover{attempt}"):
                continue
            if not run_phase(lambda: tip_goal(grasp_tip_z_fn()),
                             lambda: abs(float(ee_pose()[0][2])
                                         - (grasp_tip_z_fn() + FINGER_LEN)) < 0.005
                             and (ee_pose()[0][:2] - obj_pos_fn()[:2]).norm() < 0.005,
                             OPEN, 8.0, tag=f"{name}-descend{attempt}"):
                p, _ = ee_pose()
                hold(V3(p[0], p[1], hover_z + FINGER_LEN), gq, OPEN, 0.8)
                continue

            gp, _ = tip_goal(grasp_tip_z_fn())
            close_ramp(gp, gq, grip)

            z0 = float(obj_pos_fn()[2])
            p, _ = ee_pose()
            run_phase(lambda p=p, gq=gq: (V3(p[0], p[1], carry_hand_z), gq),
                      lambda: float(obj_pos_fn()[2]) > lift_obj_z,
                      grip, 6.0, tag=f"{name}-lift{attempt}")
            w = width()
            if float(obj_pos_fn()[2]) - z0 < 0.04 or not (w_lo < w < w_hi):
                print(f"[solve] {name}: lift verdict failed (dz="
                      f"{float(obj_pos_fn()[2]) - z0:.3f} w={w * 1000:.0f}mm) — retry",
                      flush=True)
                p, _ = ee_pose()
                hold(p, gq, OPEN, 0.6)
                hold(V3(p[0], p[1], hover_z + FINGER_LEN + 0.05), gq, OPEN, 1.0)
                continue

            cq = carry_quat_fn(az) if carry_quat_fn else gq

            def carry_goal(cq=cq):
                o, h = obj_pos_fn(), ee_pose()[0]
                t = target_xy_fn()
                corr = (1.3 * (t - o[:2])).clamp(-0.015, 0.015)
                return V3(h[0] + corr[0], h[1] + corr[1], carry_hand_z), cq

            run_phase(carry_goal,
                      lambda: (obj_pos_fn()[:2] - target_xy_fn()).norm() < 0.006,
                      grip, 14.0, tag=f"{name}-carry{attempt}", xy_boost=1.6)
            if not (w_lo < width() < w_hi):
                print(f"[solve] {name}: slipped in transit — retry", flush=True)
                continue

            def lower_goal(cq=cq):
                o, h = obj_pos_fn(), ee_pose()[0]
                t = target_xy_fn()
                corr = (1.2 * (t - o[:2])).clamp(-0.01, 0.01)
                return V3(h[0] + corr[0], h[1] + corr[1],
                          h[2] - (float(o[2]) - place_obj_z)), cq

            ok = run_phase(lower_goal,
                           lambda: abs(float(obj_pos_fn()[2]) - place_obj_z) < place_z_tol
                           and (obj_pos_fn()[:2] - target_xy_fn()).norm() < place_xy_tol,
                           grip, 10.0, tag=f"{name}-lower{attempt}", xy_boost=1.6)
            if not ok:
                p, _ = ee_pose()
                hold(V3(p[0], p[1], carry_hand_z), cq, grip, 1.5)
                continue

            p, q = ee_pose()
            hold(p, cq, OPEN, 0.8)                                  # slow release
            hold(V3(p[0], p[1], p[2] + 0.12), cq, OPEN, 1.0)        # retreat up

            deadline = sim_t["t"] + 5.0
            while sim_t["t"] < deadline:
                pp, _ = ee_pose()
                hold(pp, cq, OPEN, env.dt * robot.control_period)
                if settle_pred():
                    return True
            print(f"[solve] {name}: placed but settle_pred false — retry", flush=True)
        return False

    # ----- geometry handles (ALL WORLD-frame: root_pos_w already includes the env origin,
    # and servo goals are world-frame too, so nothing needs origin arithmetic) ---------------
    def crate_yaw() -> float:
        return yaw_of(scene.crate.data.root_quat_w[0])

    def crate_slot_xy(i: int) -> torch.Tensor:
        cp = scene.crate.data.root_pos_w[0]
        cq = scene.crate.data.root_quat_w[0]
        sx, sy = c.stow_slots[i]
        off = quat_apply(cq.unsqueeze(0), V3(sx, sy, 0.0).unsqueeze(0))[0]
        return (cp + off)[:2]

    # =========================== run ===========================================================
    p0, q0 = ee_pose()
    hold(p0, q0, OPEN, 1.0)
    boundary("start")

    wall_top_w = float(origin[2]) + 0.0005 + c.wall_top_local  # crate root sits at z=0.0005

    # ---- phase 1: MILK OUT --------------------------------------------------------------------
    def milk_az(attempt: int) -> float:
        return fold90(yaw_of(scene.milk.data.root_quat_w[0])) + (attempt % 2) * math.pi / 2

    ok = pick_carry_place(
        name="milk",
        obj_pos_fn=lambda: scene.milk.data.root_pos_w[0],
        azimuth_fn=milk_az,
        grasp_tip_z_fn=lambda: float(scene.milk.data.root_pos_w[0][2]) + c.milk_h / 2 - 0.032,
        grip=0.026, w_lo=0.050, w_hi=0.074,
        lift_obj_z=0.26, carry_hand_z=0.26 + 0.040 + FINGER_LEN,
        target_xy_fn=lambda: scene.pad.data.root_pos_w[0][:2],
        place_obj_z=float(scene.pad.data.root_pos_w[0][2]) + c.pad_t / 2 + c.milk_h / 2 + 0.004,
        place_z_tol=0.006, place_xy_tol=0.010,
        settle_pred=lambda: bool(scene.delivered()[0]),
    )
    if not ok:
        boundary("milk-FAILED")
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)
    boundary("milk-delivered")

    # ---- phase 2: ITEMS IN ----------------------------------------------------------------------
    for i, (nm, kind, r, height, _mass, _rgb) in enumerate(c.items):
        if not bool(scene.present[0, i]):
            continue
        half = height / 2 if kind == "cyl" else r
        if kind == "cyl":
            # standing can: any azimuth; tipped-over can: pinch ACROSS the lying axis
            def az_fn(a, nm_=nm):
                q = scene.items[nm_].data.root_quat_w[0]
                axis = quat_apply(q.unsqueeze(0), V3(0.0, 0.0, 1.0).unsqueeze(0))[0]
                if float(axis[2]) > 0.8:
                    return (a % 2) * math.pi / 4
                return math.atan2(float(axis[1]), float(axis[0])) + math.pi / 2

            grip_v, wl, wh = 0.022, 0.044, 0.065
        else:
            az_fn = (lambda nm_=nm: lambda a: fold90(
                yaw_of(scene.items[nm_].data.root_quat_w[0])) + (a % 2) * math.pi / 2)()
            grip_v, wl, wh = 0.019, 0.040, 0.060
        drop_z = wall_top_w + half + 0.012

        ok = pick_carry_place(
            name=nm,
            obj_pos_fn=(lambda nm_=nm: lambda: scene.items[nm_].data.root_pos_w[0])(),
            azimuth_fn=az_fn,
            grasp_tip_z_fn=(lambda nm_=nm: lambda: float(
                scene.items[nm_].data.root_pos_w[0][2]))(),
            grip=grip_v, w_lo=wl, w_hi=wh,
            lift_obj_z=0.20, carry_hand_z=0.20 + FINGER_LEN + 0.02,
            target_xy_fn=(lambda i_=i: lambda: crate_slot_xy(i_))(),
            place_obj_z=drop_z, place_z_tol=0.008, place_xy_tol=0.015,
            settle_pred=(lambda i_=i: lambda: bool(scene.stowed()[0, i_]))(),
        )
        if not ok:
            boundary(f"{nm}-FAILED")
            print("SIM_GEN_SOLVE: FAIL", flush=True)
            os._exit(1)
        boundary(f"{nm}-stowed")

    # ---- phase 3: LID ON --------------------------------------------------------------------------
    knob_mid_dz = c.lid_t / 2 + c.knob_h / 2  # board center -> knob center
    seat_w = float(origin[2]) + 0.0005 + c.seat_z_local  # world z, seated board CENTER

    def lid_az(attempt: int) -> float:
        return fold90(yaw_of(scene.lid.data.root_quat_w[0])) + (attempt % 2) * math.pi / 2

    def lid_carry_quat(az0: float) -> torch.Tensor:
        # rotate the wrist so the (square) lid squares up with the crate, mod 90
        delta = fold90(crate_yaw() - az0)
        return jaw_quat(az0 + delta)

    ok = pick_carry_place(
        name="lid",
        obj_pos_fn=lambda: scene.lid.data.root_pos_w[0],
        azimuth_fn=lid_az,
        grasp_tip_z_fn=lambda: float(scene.lid.data.root_pos_w[0][2]) + knob_mid_dz,
        grip=0.007, w_lo=0.015, w_hi=0.034,
        lift_obj_z=0.18, carry_hand_z=0.18 + knob_mid_dz + FINGER_LEN + 0.01,
        target_xy_fn=lambda: scene.crate.data.root_pos_w[0][:2],
        place_obj_z=seat_w + 0.006, place_z_tol=0.004, place_xy_tol=0.008,
        settle_pred=lambda: bool(scene.lid_seated()[0]) and bool(scene.lid_still()[0]),
        carry_quat_fn=lid_carry_quat,
    )
    if not ok:
        boundary("lid-FAILED")
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)
    boundary("lid-seated")

    # ---- persistence: park the arm, keep simulating, success must hold ------------------------
    dewind()
    p, q = ee_pose()
    hold(V3(float(p[0]), float(p[1]), 0.45), jaw_quat(0.0), OPEN, 1.0)
    okp = True
    for _ in range(6):
        pp, _ = ee_pose()
        hold(pp, jaw_quat(0.0), OPEN, 0.5)
        okp = okp and bool(scene.success()[0])
    boundary("final")

    mono = all(score_trace[j] <= score_trace[j + 1] + 1e-6
               for j in range(len(score_trace) - 1))
    print(f"[solve] score trace: {[f'{v:.3f}' for v in score_trace]} monotone={mono}",
          flush=True)
    if okp and bool(scene.success()[0]) and float(scene.score()[0]) == 1.0 and mono:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
        os._exit(0)
    print("SIM_GEN_SOLVE: FAIL", flush=True)
    os._exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)

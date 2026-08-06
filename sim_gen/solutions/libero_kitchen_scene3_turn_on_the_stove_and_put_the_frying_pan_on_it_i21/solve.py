"""solve — primer_stove (libero_kitchen_scene3 ... i21) with a REAL Franka arm, OSC servo.

Plan (base at BASE = the scene's stage_anchor; every contact 0.42-0.57 m out):

  PUMP    with the gripper CLOSED, press the red pump cap straight down with the
          fingertips until the scene's own pump_depth() readback crosses full depth,
          then lift off and let the spring return the cap fully up — one full stroke.
          Repeat, reading scene.strokes()/scene.primed(), until the burner is primed
          (2-4 strokes, sampled per episode; the stroke counter is the scene's own
          full-hysteresis edge detector, so the deadline is met by construction).
  PLACE   top-down pinch of the griddle's steel center knob (24 mm square post — the
          proven knob-post pinch; CoM hangs under the pinch), lift-verdict, closed-loop
          carry on the GRIDDLE xy to the live burner axis, lower to rest on the burner
          disc, slow release, retreat; wait for the sustained success hold.

No task-object state is ever written; only arm + gripper joints are commanded through
the robobench OSC controller. Prints the scene readouts and `SIM_GEN_SCORE <score>` at
each phase boundary (latched credit is monotone along this trajectory) and exactly
`SIM_GEN_SOLVE: SUCCESS` once scene.success() holds and persists.

Run:  python -m simgen_tasks.libero_kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it_i21.solve --headless [--seed N]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import torch  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    axis_angle_from_quat,
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
SHUT = 0.0           # per-finger target, fully closed (the pump-press "tool tip")
FINGER_LEN = 0.112   # panda_hand frame -> fingertip midpoint
BASE = (-0.50, 0.0, 0.0)  # arm base on the ground = scene cfg stage_anchor
CARRY_Z = 0.22       # griddle-center transit height (clears the 159 mm cap top)

KNOB_GRIP = 0.007    # proven 24 mm knob-post pinch: grip command + width gates
KNOB_W_LO = 0.015
KNOB_W_HI = 0.034


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
    env = EnvCfg(scene="primer_stove", robot="franka", control_mode="osc",
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

    env.reset(seed=args.seed)  # seed at reset time: build consumes RNG state
    print("[solve] describe():\n" + scene.describe(), flush=True)
    pan0 = scene.pan.data.root_pos_w[0]
    moka0 = scene.moka.data.root_pos_w[0]
    print(f"[solve] layout: pan=({pan0[0]:.3f},{pan0[1]:.3f}) "
          f"moka=({moka0[0]:.3f},{moka0[1]:.3f}) "
          f"n_required={int(scene.n_required()[0])}", flush=True)

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
                  f"err={float((gp - p).norm()) * 1000:.0f}mm w={width() * 1000:.1f}mm "
                  f"depth={float(scene.pump_depth()[0]) * 1000:.1f}mm", flush=True)
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

    # ----- scene readouts ---------------------------------------------------------------------
    score_trace: list[float] = []

    def boundary(tag: str) -> None:
        s = float(scene.score()[0])
        score_trace.append(s)
        print(f"[readout] {tag}: strokes={int(scene.strokes()[0])}/"
              f"{int(scene.n_required()[0])} primed={bool(scene.primed()[0])} "
              f"depth={float(scene.pump_depth()[0]) * 1000:.1f}mm "
              f"pan_on_burner={bool(scene.pan_on_burner_now()[0])} "
              f"hold={int(scene._hold[0])} success={bool(scene.success()[0])}", flush=True)
        print(f"SIM_GEN_SCORE {s:.3f}", flush=True)

    # ----- geometry handles (ALL WORLD-frame) --------------------------------------------------
    def cap_xy() -> torch.Tensor:
        return scene.cap.data.root_pos_w[0][:2]

    def cap_top_z() -> float:
        return float(scene.cap.data.root_pos_w[0][2]) + c.cap_size[2] / 2

    cap_home_top = float(scene.env_origins[0, 2]) + c.cap_home_lz + c.cap_size[2] / 2

    def burner_xy() -> torch.Tensor:
        return scene.burner.data.root_pos_w[0][:2]

    def pan_pos() -> torch.Tensor:
        return scene.pan.data.root_pos_w[0]

    def bearing_from_base(xy: torch.Tensor) -> float:
        return math.atan2(float(xy[1]) - BASE[1], float(xy[0]) - BASE[0])

    pan_rest_z = float(scene.env_origins[0, 2]) + c.pan_rest_lz
    # gripped knob: fingertip midpoint sits at knob mid-height above the pan center
    knob_tip_dz = c.pan_disc_t / 2 + c.pan_knob_h / 2

    # ----- PUMP phase ---------------------------------------------------------------------------
    def pump_until_primed() -> bool:
        az = bearing_from_base(cap_xy())
        gq = jaw_quat(az)
        hover_z = cap_home_top + 0.07

        def above(z: float):
            xy = cap_xy()
            return V3(xy[0], xy[1], z + FINGER_LEN), gq

        # close the jaw first, away from the cap
        p, q0 = ee_pose()
        hold(p, q0, SHUT, 0.6)

        stroke_budget = int(scene.n_required()[0]) * 2 + 4
        for k in range(stroke_budget):
            if bool(scene.primed()[0]):
                return True
            dewind(SHUT)
            # hover over the live cap center
            if not run_phase(lambda: above(hover_z),
                             lambda: (ee_pose()[0][:2] - cap_xy()).norm() < 0.006
                             and abs(float(ee_pose()[0][2]) - (hover_z + FINGER_LEN)) < 0.010,
                             SHUT, 8.0, tag=f"pump-hover{k}"):
                continue

            # press: chase a target just below the live cap top until full depth
            def press_goal():
                xy = cap_xy()
                return V3(xy[0], xy[1], cap_top_z() - 0.012 + FINGER_LEN), gq

            if not run_phase(press_goal,
                             lambda: float(scene.pump_depth()[0]) >= 0.030,
                             SHUT, 6.0, tag=f"pump-press{k}"):
                p, _ = ee_pose()
                hold(V3(p[0], p[1], hover_z + FINGER_LEN), gq, SHUT, 0.8)
                continue

            # release: lift clear, spring returns the cap fully up
            if not run_phase(lambda: above(hover_z),
                             lambda: float(scene.pump_depth()[0]) <= 0.004,
                             SHUT, 5.0, tag=f"pump-release{k}"):
                continue
            boundary(f"stroke-{int(scene.strokes()[0])}")
        return bool(scene.primed()[0])

    # ----- PLACE phase ----------------------------------------------------------------------------
    def place_griddle() -> bool:
        for attempt in range(5):
            dewind()
            az = bearing_from_base(pan_pos()[:2]) + (0.0, math.pi / 4, -math.pi / 4,
                                                     math.pi / 2, math.pi / 3)[attempt]
            gq = tilt_quat(az, pan_pos()[:2])

            def knob_tip(z_extra: float, gq=gq):
                o = pan_pos()
                return V3(o[0], o[1],
                          float(o[2]) + knob_tip_dz + z_extra + FINGER_LEN), gq

            # hover, then descend to the knob mid-height
            if not run_phase(lambda: knob_tip(0.12),
                             lambda: (ee_pose()[0][:2] - pan_pos()[:2]).norm() < 0.006,
                             OPEN, 8.0, tag=f"pan-hover{attempt}"):
                continue
            if not run_phase(lambda: knob_tip(0.0),
                             lambda: abs(float(ee_pose()[0][2])
                                         - (float(pan_pos()[2]) + knob_tip_dz + FINGER_LEN))
                             < 0.005
                             and (ee_pose()[0][:2] - pan_pos()[:2]).norm() < 0.005,
                             OPEN, 8.0, tag=f"pan-descend{attempt}"):
                p, _ = ee_pose()
                hold(V3(p[0], p[1], CARRY_Z + FINGER_LEN), gq, OPEN, 0.8)
                continue

            gp, _ = knob_tip(0.0)
            close_ramp(gp, gq, KNOB_GRIP)

            z0 = float(pan_pos()[2])
            p, _ = ee_pose()
            run_phase(lambda p=p, gq=gq: (V3(p[0], p[1], CARRY_Z + knob_tip_dz + FINGER_LEN), gq),
                      lambda: float(pan_pos()[2]) > z0 + 0.05,
                      KNOB_GRIP, 6.0, tag=f"pan-lift{attempt}")
            w = width()
            if float(pan_pos()[2]) - z0 < 0.04 or not (KNOB_W_LO < w < KNOB_W_HI):
                print(f"[solve] pan lift verdict failed (dz={float(pan_pos()[2]) - z0:.3f} "
                      f"w={w * 1000:.0f}mm) — retry", flush=True)
                p, _ = ee_pose()
                hold(p, gq, OPEN, 0.6)
                hold(V3(p[0], p[1], p[2] + 0.10), gq, OPEN, 1.0)
                continue

            # closed-loop carry on the GRIDDLE xy onto the live burner axis
            def carry_goal(gq=gq):
                o, hd = pan_pos(), ee_pose()[0]
                corr = (1.3 * (burner_xy() - o[:2])).clamp(-0.015, 0.015)
                return V3(hd[0] + corr[0], hd[1] + corr[1],
                          CARRY_Z + knob_tip_dz + FINGER_LEN), gq

            run_phase(carry_goal,
                      lambda: (pan_pos()[:2] - burner_xy()).norm() < 0.006,
                      KNOB_GRIP, 14.0, tag=f"pan-carry{attempt}", xy_boost=1.6)
            if not (KNOB_W_LO < width() < KNOB_W_HI):
                print("[solve] pan slipped in transit — retry", flush=True)
                continue

            # lower onto the burner disc
            def lower_goal(gq=gq):
                o, hd = pan_pos(), ee_pose()[0]
                corr = (1.2 * (burner_xy() - o[:2])).clamp(-0.008, 0.008)
                return V3(hd[0] + corr[0], hd[1] + corr[1],
                          pan_rest_z + 0.001 + knob_tip_dz + FINGER_LEN), gq

            if not run_phase(lower_goal,
                             lambda: bool(scene.pan_on_burner_geom()[0]),
                             KNOB_GRIP, 10.0, tag=f"pan-lower{attempt}", xy_boost=1.6):
                p, _ = ee_pose()
                hold(V3(p[0], p[1], CARRY_Z + knob_tip_dz + FINGER_LEN), gq, KNOB_GRIP, 1.5)
                continue

            p, _ = ee_pose()
            hold(p, gq, OPEN, 0.8)                                  # slow release
            hold(V3(p[0], p[1], p[2] + 0.14), gq, OPEN, 1.2)        # retreat up

            deadline = sim_t["t"] + 6.0
            while sim_t["t"] < deadline:
                pp, _ = ee_pose()
                hold(pp, gq, OPEN, env.dt * robot.control_period)
                if bool(scene.success()[0]):
                    return True
            print("[solve] placed but success() false — retry", flush=True)
        return False

    # =========================== run ===========================================================
    p0, q0 = ee_pose()
    hold(p0, q0, OPEN, 1.0)
    boundary("start")

    if not pump_until_primed():
        boundary("pump-FAILED")
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)
    boundary("primed")

    if not place_griddle():
        boundary("place-FAILED")
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)
    boundary("placed")

    # ---- persistence: park the arm, keep simulating, success must hold ------------------------
    dewind()
    p, _q = ee_pose()
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

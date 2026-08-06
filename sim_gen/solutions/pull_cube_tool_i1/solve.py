"""solve — chute_dispatch (pull_cube_tool_i1) with a REAL Franka arm, OSC servo loop.

Strategy: read the sampled green-target side from the scene (mat pose / target_side
readback), then execute ONE closed-loop pick and a committed release into the correct
chute mouth — gravity does the delivery into the unreachable pen:

  HOVER    top-down hand above the live cube position (jaw snapped to the cube's yaw)
  DESCEND  fingertips to cube-center height
  CLOSE    slow-ramp pinch of the 45 mm cube; width gate
  LIFT     straight up to carry height (clears the chute mouth backstop); lift verdict
  CARRY    closed-loop on the CUBE xy to the drop point over the correct chute mouth
  LOWER    cube to just above the channel (fingers stay above the wall tops), xy gated
  RELEASE  open, retreat straight up
  WATCH    poll the scene's own transit latch, then success() while the cube slides
           down the chute and settles inside the green pen

The pens and the lower chutes are beyond reach by design; the arm only ever touches
the cube, near the base. No task-object state is ever written.

Phase boundaries print `SIM_GEN_SCORE <scene.score()>` (latched rubric credit — never
decreases along this trajectory); the final verdict prints `SIM_GEN_SOLVE: SUCCESS`.

Run:  python -m simgen_tasks.pull_cube_tool_i1.solve --headless [--seed N]
Base pose (recorded in TASK.md): Franka base at (-0.32, 0.0, 0.0), facing +x.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_sec", type=float, default=150.0, help="sim-time budget (s)")
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
    quat_from_matrix,
    quat_mul,
)

import robobench  # noqa: E402
from robobench.core import EnvCfg  # noqa: E402
from robobench.robots.franka import FrankaRobotCfg  # noqa: E402

try:
    from .scene import ChuteDispatchScene  # noqa: F401  (registers the scene)
except ImportError:  # pragma: no cover - forge fallback
    from scene import ChuteDispatchScene  # noqa: F401

OPEN = 0.04           # per-finger position target, fully open (80 mm aperture)
GRIP = 0.019          # gentle pinch target for the 45 mm / 40 g cube
FINGER_LEN = 0.112    # panda_hand frame -> fingertip midpoint (corpus-measured)
BASE = (-0.32, 0.0, 0.0)  # arm base on the ground, west of the cube spawn zone
CARRY_Z = 0.26        # cube-center carry height: cube bottom 0.2375 clears the
#                       0.195-high mouth backstop by > 40 mm
DROP_X = 0.36         # drop point down-slope of the backstop (channel surface 0.081)
RELEASE_Z = 0.165     # cube-center at release: cube falls ~60 mm into the channel,
#                       finger undersides stay above the ~0.139 wall-top there
WIDTH_LO, WIDTH_HI = 0.040, 0.052  # jaw-width band that certifies a held 45 mm cube


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    robobench.discover()
    rcfg = FrankaRobotCfg(
        base_pos=BASE,
        nullspace_dof_pos=(),   # pen_holder lesson: default posture winds the arm
        gripper_effort_limit=30.0,
        gripper_stiffness=2000.0,
    )
    cfg = EnvCfg(scene="chute_dispatch", robot="franka", control_mode="osc",
                 env_spacing=4.0, robot_cfg=rcfg, seed=args.seed)
    env = cfg.build(num_envs=1, device=device)
    torch.manual_seed(args.seed)  # env construction reseeds RNGs; re-assert ours
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
    osc.cfg.kd_null = 3.46
    n_act = robot.action_dim
    ctrl_hz = 1.0 / (env.dt * robot.control_period)
    SEC = lambda s: max(1, round(s * ctrl_hz))  # noqa: E731

    env.reset()
    origin = env.iscene.env_origins[0]
    V3 = lambda x, y, z: torch.tensor([float(x), float(y), float(z)], device=device)  # noqa: E731

    print("[solve] scene description:\n" + scene.describe(), flush=True)

    # ----- low-level servo (franka_session lineage) -----------------------------------
    def ee_pose():
        return art.data.body_pos_w[0, ee_idx], art.data.body_quat_w[0, ee_idx]

    def width() -> float:
        return 2.0 * art.data.joint_pos[0, fj1].item()

    def cube_pos() -> torch.Tensor:
        return scene.cube.data.root_pos_w[0] - origin  # env-local

    def servo(goal_pos, goal_quat, grip, a, xy_boost: float = 1.0):
        p, q = ee_pose()
        err = (goal_pos + origin) - p
        err = torch.cat([err[:2] * xy_boost, err[2:3]])
        a[0, 0:3] = (err / osc.cfg.pos_scale).clamp(-1.0, 1.0)
        qe = quat_mul(goal_quat.unsqueeze(0), quat_conjugate(q.unsqueeze(0)))
        a[0, 3:6] = (axis_angle_from_quat(qe)[0] / osc.cfg.rot_scale).clamp(-1.0, 1.0)
        a[0, 6:8] = grip

    def jaw_quat(azimuth: float) -> torch.Tensor:
        yh = V3(math.cos(azimuth), math.sin(azimuth), 0.0)
        zh = V3(0.0, 0.0, -1.0)
        xh = torch.cross(yh, zh, dim=0)
        return quat_from_matrix(torch.stack([xh, yh, zh], dim=1).unsqueeze(0))[0]

    sim_t = {"t": 0.0}

    def tick(a):
        env.step(a)
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
                  f"goal=({gp[0]:.3f},{gp[1]:.3f},{gp[2]:.3f}) w={width()*1000:.1f}mm",
                  flush=True)
        return False

    def dewind() -> None:
        q = art.data.joint_pos[0]
        if (abs(q[0].item()) > 2.6 or q[3].item() < -2.95 or q[3].item() > -0.15
                or abs(q[6].item()) > 2.6):
            print("[solve] wound arm — joint reset in place", flush=True)
            robot.reset(torch.tensor([0], device=device, dtype=torch.long))
            for _ in range(SEC(0.5)):
                a = torch.zeros(1, n_act, device=device)
                p, qq = ee_pose()
                a[0, 6:8] = OPEN
                servo(p - origin, qq, OPEN, a)
                tick(a)

    def close_ramp(goal_fn, gq, secs: float = 1.2):
        n = SEC(secs)
        for k in range(n):
            a = torch.zeros(1, n_act, device=device)
            f = min(1.0, (k + 1) / (n * 0.7))
            gp, _ = goal_fn()
            servo(gp, gq, OPEN + (GRIP - OPEN) * f, a)
            tick(a)

    def report(tag: str) -> None:
        p = cube_pos()
        print(f"[readout] {tag:10s} | cube=({p[0]:+.3f},{p[1]:+.3f},{p[2]:.3f}) "
              f"side={float(scene.target_side[0]):+.0f} lifted={bool(scene.lifted[0])} "
              f"transit={scene.transit[0].tolist()} in_pen={bool(scene.in_target_pen()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def score_print() -> None:
        print(f"SIM_GEN_SCORE {float(scene.score()[0]):.3f}", flush=True)

    def cube_yaw_azimuth() -> float:
        """Jaw azimuth snapped to the cube's yaw (mod 90 deg) for a flat-face grip."""
        q = scene.cube.data.root_quat_w[0]
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]).item(),
                         1 - 2 * (q[2] * q[2] + q[3] * q[3]).item())
        yaw = math.remainder(yaw, math.pi / 2)
        return yaw

    # ----- the solve ------------------------------------------------------------------
    t_start = sim_t["t"]
    p0, q0 = ee_pose()
    hold(p0 - origin, q0, OPEN, 1.0)
    side = float(scene.target_side[0])
    lane_y = side * c.lane_dy
    report("start")
    score_print()  # phase boundary: settled start (expected 0)

    grasped = False
    for attempt in range(4):
        if sim_t["t"] - t_start > args.max_sec:
            break
        dewind()
        gq = jaw_quat(cube_yaw_azimuth())

        def tip_goal(z, gq=gq):
            cp = cube_pos()
            return V3(cp[0], cp[1], z + FINGER_LEN), gq

        # HOVER above the cube
        run_phase(lambda: tip_goal(0.18),
                  lambda: (ee_pose()[0][:2] - (cube_pos()[:2] + origin[:2])).norm() < 0.006,
                  OPEN, 8.0, tag=f"hover{attempt}")
        # DESCEND to grasp height (fingertips at cube center)
        ok = run_phase(lambda: tip_goal(float(cube_pos()[2])),
                       lambda: abs(ee_pose()[0][2] - (cube_pos()[2] + origin[2]
                                                      + FINGER_LEN)) < 0.006,
                       OPEN, 8.0, tag=f"descend{attempt}")
        if not ok:
            p, _ = ee_pose()
            hold(V3(p[0] - origin[0], p[1] - origin[1], 0.30), gq, OPEN, 1.0)
            continue
        # CLOSE (slow ramp; goal tracks the live cube)
        close_ramp(lambda: tip_goal(float(cube_pos()[2])), gq)
        # LIFT straight up + verdict
        z0 = float(cube_pos()[2])
        px, py = float(ee_pose()[0][0] - origin[0]), float(ee_pose()[0][1] - origin[1])
        run_phase(lambda: (V3(px, py, CARRY_Z + FINGER_LEN), gq),
                  lambda: float(cube_pos()[2]) > CARRY_Z - 0.02,
                  GRIP, 6.0, tag=f"lift{attempt}")
        w = width()
        if float(cube_pos()[2]) - z0 > 0.05 and WIDTH_LO < w < WIDTH_HI:
            grasped = True
            break
        print(f"[solve] lift verdict failed (rise={float(cube_pos()[2]) - z0:.3f} "
              f"w={w * 1000:.1f}mm) — retry", flush=True)
        p, _ = ee_pose()
        hold(p - origin, gq, OPEN, 0.6)
        hold(V3(p[0] - origin[0], p[1] - origin[1], 0.30), gq, OPEN, 1.0)

    if not grasped:
        report("no-grasp")
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    report("lifted")
    score_print()  # phase boundary: lift latched (expected 0.15)

    # CARRY: closed-loop on the CUBE xy onto the drop point over the correct mouth
    gq = jaw_quat(cube_yaw_azimuth())

    def carry_goal():
        cp = cube_pos()
        h = ee_pose()[0] - origin
        corr = torch.stack([DROP_X - cp[0], lane_y - cp[1]]).clamp(-0.015, 0.015)
        return V3(h[0] + corr[0] * 1.3, h[1] + corr[1] * 1.3, CARRY_Z + FINGER_LEN), gq

    run_phase(carry_goal,
              lambda: (cube_pos()[:2] - V3(DROP_X, lane_y, 0)[:2]).norm() < 0.008,
              GRIP, 15.0, tag="carry", xy_boost=1.6)
    if not (WIDTH_LO < width() < WIDTH_HI):
        report("slipped")
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # LOWER: cube to just above the channel, xy stays gated on the live cube
    def lower_goal():
        cp = cube_pos()
        h = ee_pose()[0] - origin
        corr = torch.stack([DROP_X - cp[0], lane_y - cp[1]]).clamp(-0.010, 0.010)
        return V3(h[0] + corr[0] * 1.2, h[1] + corr[1] * 1.2,
                  h[2] - (cp[2] - RELEASE_Z)), gq

    run_phase(lower_goal,
              lambda: abs(float(cube_pos()[2]) - RELEASE_Z) < 0.006
              and (cube_pos()[:2] - V3(DROP_X, lane_y, 0)[:2]).norm() < 0.008,
              GRIP, 10.0, tag="lower", xy_boost=1.6)
    report("pre-release")

    # RELEASE + retreat straight up
    p, _ = ee_pose()
    hold(p - origin, gq, OPEN, 0.6)
    hold(V3(p[0] - origin[0], p[1] - origin[1], 0.42), gq, OPEN, 1.2)

    # WATCH: the chute delivers; poll the scene's own latches / predicates
    deadline = sim_t["t"] + 8.0
    transit_seen = False
    while sim_t["t"] < deadline:
        pp, qq = ee_pose()
        hold(pp - origin, qq, OPEN, 1.0 / ctrl_hz)
        if not transit_seen and bool(scene.transit_correct()[0]):
            transit_seen = True
            report("transit")
            score_print()  # phase boundary: correct-chute transit latched (0.5)
        if bool(scene.success()[0]):
            break
    report("settled")

    # persistence: keep simulating with the arm parked; success must not flicker
    stable = True
    for _ in range(SEC(2.0)):
        pp, qq = ee_pose()
        hold(pp - origin, qq, OPEN, 1.0 / ctrl_hz)
        if not bool(scene.success()[0]):
            stable = False
    report("persist")
    score_print()  # phase boundary: final (expected 1.0)

    ok = bool(scene.success()[0]) and stable
    print(f"[solve] sim time used: {sim_t['t'] - t_start:.1f} s", flush=True)
    print("SIM_GEN_SOLVE: SUCCESS" if ok else "SIM_GEN_SOLVE: FAIL", flush=True)
    code = 0 if ok else 1
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
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)

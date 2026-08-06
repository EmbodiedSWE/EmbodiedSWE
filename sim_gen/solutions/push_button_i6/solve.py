"""solve — real single-Franka solution for WeighbridgeScene (the feasibility certificate).

Env: scene "weighbridge" + robot "franka" (OSC), base at (-0.48, 0, 0) facing +x — the
scene layout was designed around this pose: pedestal ~0.63 m ahead, blocks on a lateral
arc 0.34-0.45 m from the base.

Plan (no execution-order constraint beyond pick-then-place):
  SETTLE     let the scatter come to rest; read the scene
  PICK       top-down pinch of the dark-red LOAD block (55 mm, 0.5 kg): hover over its
             live xy, descend fingertips to pad-height, quasi-static close ramp,
             lift + verdict (block rose AND jaw width in the 55 mm band)
  CARRY      closed-loop on the BLOCK xy onto the plate axis at transit height
  LOWER      closed-loop press-down: track the block onto the plate and keep lowering
             until the plate is measured >= 30 mm deep (the spring bottoms under the
             gripped block)
  RELEASE    slow open, retreat straight up, park clear
  VERIFY     wait for the scene's own sustained-press counter: success() must latch on
             the resting block alone and PERSIST (extra 3 s of simulation)

Arm-only manipulation: no task-object state writes, no external forces. The gripper and
arm joints are the only actuators commanded. Prints SIM_GEN_SCORE at each phase
boundary (latched credit never decreases) and SIM_GEN_SOLVE: SUCCESS at the end.

Run: python -u -m simgen_tasks.push_button_i6.solve --headless [--seed N]
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_wall_s", type=float, default=1200.0)
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

from simgen_tasks.push_button_i6 import scene as scene_mod  # noqa: F401, E402

# Emergency watchdog: whatever happens, this process must die (Kit teardown hangs).
threading.Timer(args.max_wall_s, lambda: (print("[solve] WALL TIMEOUT", flush=True),
                                          os._exit(3))).start()

OPEN = 0.04          # per-finger position target, fully open (80 mm aperture)
FINGER_LEN = 0.112   # panda_hand frame -> fingertip along the approach axis
BASE_POS = (-0.48, 0.0, 0.0)
TRAVEL_Z = 0.35      # hand transit height (block hangs ~0.11 m below the tips' level)
GRIP = 0.022         # per-finger close target for the 55 mm block (quasi-static ramp)
W_LO, W_HI = 0.046, 0.062  # jaw-width verdict band for a held 55 mm block


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    robobench.discover()

    rcfg = FrankaRobotCfg(base_pos=BASE_POS, nullspace_dof_pos=(),
                          gripper_effort_limit=120.0, gripper_stiffness=4000.0)
    env = EnvCfg(scene="weighbridge", robot="franka", control_mode="osc",
                 env_spacing=3, robot_cfg=rcfg).build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    robot = env.robot
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

    # EnvCfg.seed=0 reseeds the RNGs at build time — pass the CLI seed explicitly so
    # different --seed values really give different episodes.
    env.reset(seed=args.seed)
    origin = env.iscene.env_origins[0]

    V3 = lambda x, y, z: torch.tensor([float(x), float(y), float(z)], device=device)
    SEC = lambda s: max(1, round(s * ctrl_hz))

    # ----- kernel (franka OSC session, corpus constants) -------------------------------------
    def ee_pose():
        return art.data.body_pos_w[0, ee_idx], art.data.body_quat_w[0, ee_idx]

    def width() -> float:
        return 2.0 * art.data.joint_pos[0, fj1].item()

    def tip_pos():
        p, q = ee_pose()
        zh = quat_apply(q.unsqueeze(0), V3(0, 0, 1).unsqueeze(0))[0]
        return p + FINGER_LEN * zh

    def servo(goal_pos, goal_quat, grip, a, xy_boost=1.0):
        p, q = ee_pose()
        err = goal_pos - p
        err = torch.cat([err[:2] * xy_boost, err[2:3]])
        a[0, 0:3] = (err / osc.cfg.pos_scale).clamp(-1.0, 1.0)
        qe = quat_mul(goal_quat.unsqueeze(0), quat_conjugate(q.unsqueeze(0)))
        a[0, 3:6] = (axis_angle_from_quat(qe)[0] / osc.cfg.rot_scale).clamp(-1.0, 1.0)
        a[0, 6:8] = grip

    sim_t = {"t": 0.0}

    def tick(a):
        env.step(a)
        sim_t["t"] += env.dt * robot.control_period

    def hold(pos, quat, grip, secs, xy_boost=1.0):
        for _ in range(SEC(secs)):
            a = torch.zeros(1, n_act, device=device)
            servo(pos, quat, grip, a, xy_boost)
            tick(a)

    def run_phase(goal_fn, gate_fn, grip, timeout_s, tag="", xy_boost=1.0):
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
                  f"err={float((gp - p).norm()) * 1000:.0f}mm w={width()*1000:.1f}mm",
                  flush=True)
        return False

    def jaw_quat(azimuth: float):
        yh = V3(math.cos(azimuth), math.sin(azimuth), 0.0)
        zh = V3(0.0, 0.0, -1.0)
        xh = torch.cross(yh, zh, dim=0)
        return quat_from_matrix(torch.stack([xh, yh, zh], dim=1).unsqueeze(0))[0]

    def tilt_quat(azimuth: float, target_xy, near=0.37, max_tilt=0.35):
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

    def dewind() -> bool:
        q = art.data.joint_pos[0]
        if (abs(q[0].item()) > 2.6 or q[3].item() < -2.95 or q[3].item() > -0.15
                or abs(q[6].item()) > 2.6):
            print(f"[dewind] wound arm (q1={q[0]:.2f} q4={q[3]:.2f} q7={q[6]:.2f}) — reset",
                  flush=True)
            robot.reset(torch.tensor([0], device=device, dtype=torch.long))
            p, qq = ee_pose()
            hold(p, qq, OPEN, 0.5)
            return True
        return False

    def close_ramp(pos_fn, quat, grip_target, secs=1.4):
        n = SEC(secs)
        for k in range(n):
            a = torch.zeros(1, n_act, device=device)
            f = min(1.0, (k + 1) / (n * 0.7))
            servo(pos_fn(), quat, OPEN + (grip_target - OPEN) * f, a)
            tick(a)

    def wrap_pi(x: float) -> float:
        return (x + math.pi) % (2 * math.pi) - math.pi

    def jaw_az_of(q):
        ey = quat_apply(q.unsqueeze(0), torch.tensor([[0.0, 1.0, 0.0]], device=device))[0]
        return math.atan2(ey[1].item(), ey[0].item())

    # ----- scene readers ----------------------------------------------------------------------
    load = scene.blocks["load"]
    half = 0.055 / 2

    def load_pos():
        return load.data.root_pos_w[0]

    def plate_xy():
        return scene.plate.data.root_pos_w[0, :2]

    def plate_top_z() -> float:
        return float(scene.plate.data.root_pos_w[0, 2]) + c.plate_size[2] / 2

    def readout(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] {tag:14s} depth={float(scene.plate_depth()[0])*1000:5.1f}mm "
              f"m_on={float(scene.loaded_mass()[0]):.3f}kg score={s:.2f} "
              f"success={bool(scene.success()[0])} sim_t={sim_t['t']:.1f}s", flush=True)
        print(f"SIM_GEN_SCORE {s:.3f}", flush=True)
        return s

    # ----- mission ------------------------------------------------------------------------------
    print(f"[solve] seed={args.seed} ctrl={ctrl_hz:.0f}Hz base={BASE_POS}", flush=True)
    print(env.describe(), flush=True)

    hold(*ee_pose(), OPEN, 1.5)  # settle the scatter
    base_xy = art.data.root_pos_w[0, :2]
    for nm, b in scene.blocks.items():
        p = (b.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] settled {nm}: ({p[0]:+.3f},{p[1]:+.3f},{p[2]:.3f}) "
              f"d_base={float((b.data.root_pos_w[0, :2] - base_xy).norm()):.3f}", flush=True)
    readout("settled")

    home_az = jaw_az_of(ee_pose()[1])
    park_q = jaw_quat(home_az)

    def park(grip=OPEN):
        dewind()
        p, q = ee_pose()
        hold(V3(p[0], p[1], max(float(p[2]), TRAVEL_Z)), q, grip, 0.8)
        hold(origin + V3(-0.08, 0.0, TRAVEL_Z), park_q, grip, 1.2)

    picked = False
    for attempt in range(4):
        dewind()
        lp = load_pos()
        # jaw azimuth: across the cube faces, the 90-deg branch nearest the home azimuth
        qw, qx, qy, qz = (float(v) for v in load.data.root_quat_w[0])
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        best = min((yaw + k * math.pi / 2 for k in range(-2, 3)),
                   key=lambda az: abs(wrap_pi(az - home_az)))
        gq = tilt_quat(best, lp[:2])

        def tip_goal(z):
            o = load_pos()
            zh = quat_apply(gq.unsqueeze(0), V3(0, 0, 1).unsqueeze(0))[0]
            return V3(o[0], o[1], z) - FINGER_LEN * zh, gq

        # HOVER over the live block
        if not run_phase(lambda: tip_goal(0.24),
                         lambda: float((ee_pose()[0][:2] - load_pos()[:2]).norm()) < 0.006,
                         OPEN, 8.0, tag=f"hover{attempt}"):
            park()
            continue

        # DESCEND: fingertips to 14 mm above the block bottom (pads centred on the cube)
        def grasp_tip_z() -> float:
            return float(load_pos()[2]) - half + 0.014

        if not run_phase(lambda: tip_goal(grasp_tip_z()),
                         lambda: abs(float(tip_pos()[2]) - grasp_tip_z()) < 0.005
                         and float((ee_pose()[0][:2] - load_pos()[:2]).norm()) < 0.006,
                         OPEN, 8.0, tag=f"descend{attempt}"):
            park()
            continue

        # CLOSE (quasi-static) + LIFT + verdict
        close_ramp(lambda: tip_goal(grasp_tip_z())[0], gq, GRIP)
        z0 = float(load_pos()[2])
        p, _ = ee_pose()
        hold(V3(p[0], p[1], TRAVEL_Z), gq, GRIP, 1.2)
        w = width()
        if (float(load_pos()[2]) - z0) > 0.05 and W_LO < w < W_HI:
            print(f"[solve] HELD (w={w*1000:.1f}mm) after attempt {attempt}", flush=True)
            picked = True
            break
        print(f"[solve] lift verdict failed (w={w*1000:.1f}mm rise="
              f"{(float(load_pos()[2]) - z0)*1000:.0f}mm) — retry", flush=True)
        p, _ = ee_pose()
        hold(V3(p[0], p[1], 0.24), gq, OPEN, 0.8)
        park()

    if not picked:
        readout("pick FAILED")
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(2)

    s_lift = readout("lifted")

    # CARRY the block over the plate axis (closed-loop on the BLOCK)
    def carry_goal():
        o, h = load_pos(), ee_pose()[0]
        corr = (1.3 * (plate_xy() - o[:2])).clamp(-0.015, 0.015)
        return V3(h[0] + corr[0], h[1] + corr[1], TRAVEL_Z), gq

    run_phase(carry_goal,
              lambda: float((load_pos()[:2] - plate_xy()).norm()) < 0.008,
              GRIP, 14.0, tag="carry", xy_boost=1.6)
    readout("carried")

    # LOWER: press the gripped block down until the plate is measured near bottom
    def lower_goal():
        o, h = load_pos(), ee_pose()[0]
        corr = (1.2 * (plate_xy() - o[:2])).clamp(-0.010, 0.010)
        target_block_z = origin[2] + c.surface_z + c.ped_size[2] + c.plate_size[2] + half - 0.002
        dz = float(o[2]) - float(target_block_z)
        return V3(h[0] + corr[0], h[1] + corr[1], h[2] - dz), gq

    ok_lower = run_phase(
        lower_goal,
        lambda: float(scene.plate_depth()[0]) >= 0.030
        and float((load_pos()[:2] - plate_xy()).norm()) < 0.025,
        GRIP, 12.0, tag="lower", xy_boost=1.6)
    if not ok_lower:
        print(f"[solve] lower did not bottom the plate "
              f"(depth={float(scene.plate_depth()[0])*1000:.1f}mm)", flush=True)
    p, _ = ee_pose()
    hold(p, gq, GRIP, 1.0)  # let the press settle under grip
    readout("pressed")

    # RELEASE slowly, retreat straight up, park clear
    p, _ = ee_pose()
    hold(p, gq, OPEN, 0.9)
    p, _ = ee_pose()
    hold(V3(p[0], p[1], TRAVEL_Z), gq, OPEN, 1.0)
    park()
    s_rel = readout("released")

    # VERIFY: the scene's own sustained-press gate on the RESTING block
    deadline = sim_t["t"] + 12.0
    while sim_t["t"] < deadline and not bool(scene.success()[0]):
        hold(*ee_pose(), OPEN, 0.5)
    s_final = readout("verified")

    # persistence: keep simulating; success must not flicker off
    hold(*ee_pose(), OPEN, 3.0)
    ok = bool(scene.success()[0])
    s_last = readout("persisted")
    lp = (load.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] final load block: ({lp[0]:+.3f},{lp[1]:+.3f},{lp[2]:.3f}) "
          f"plate_top_z={plate_top_z() - float(origin[2]):.3f} "
          f"off_axis={float((load_pos()[:2] - plate_xy()).norm())*1000:.0f}mm", flush=True)
    print(f"SIM_GEN_SOLVE: {'SUCCESS' if ok else 'FAIL'}", flush=True)

    threading.Timer(10.0, lambda: os._exit(0 if ok else 1)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()

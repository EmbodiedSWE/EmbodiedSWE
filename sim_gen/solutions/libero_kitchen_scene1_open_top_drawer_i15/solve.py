"""solve — real Franka-arm solution for LatchDrawerScene
(libero_kitchen_scene1_open_top_drawer_i15).

Single Franka + parallel jaw, OSC, base at (-0.28, 0, 0) facing +x. Strategy (the
task's own intended plan — the seed's grasp-handle-and-pull is impossible here, the
flush face offers nothing to grasp):

  PHASE 1  PRESS: with the jaw nearly closed (12 mm gap) and the hand pitched 30 deg
           forward (a vertical finger cannot reach the recessed plate — probed on the
           forge), push the drawer's flush front plate straight inward (~16 mm,
           quasi-static ramp) until the latch arms, then withdraw back-and-up; the
           latch releases and the ejection spring slides the drawer out toward the
           robot on its own. The robot never moves the drawer in the opening direction.
  PHASE 2  RETRIEVE: top-down cage-then-squeeze pick of the red cube out of the open
           drawer cavity (live pose — the cube rode the ejecting drawer).
  PHASE 3  PLACE: closed-loop carry to the green pad, lower, slow release, retreat;
           settle until scene.success().

Arm-only manipulation: the robot commands ONLY its own joints/gripper; task-object
state is never written and no external forces are applied to task objects.

Prints the scene readouts, `SIM_GEN_SCORE <score>` at each phase boundary (latched
credit — never decreases), and `SIM_GEN_SOLVE: SUCCESS` at the end.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene1_open_top_drawer_i15.solve --headless
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

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
    from simgen_tasks.libero_kitchen_scene1_open_top_drawer_i15 import (  # noqa: F401
        scene as scene_mod,
    )
except ImportError:
    import scene as scene_mod  # noqa: F401

OPEN = 0.04
PRESS_GRIP = 0.012
FINGER_LEN = 0.112
BASE_POS = (-0.28, 0.0, 0.0)


# ----- Franka OSC session (franka_session.py substrate, vendored trimmed) --------------------------
class Session:
    def __init__(self, env):
        self.env = env
        self.robot = env.robot
        self.scene = env.scene
        self.device = env.device
        art = self.robot.articulation
        self.art = art
        self.ee_idx = art.body_names.index("panda_hand")
        self.fj1 = art.find_joints(["panda_finger_joint1"])[0]
        osc = self.robot.controller.controllers[0]
        osc._kp = torch.tensor([220.0, 220.0, 220.0, 600.0, 600.0, 600.0], device=self.device)
        osc._kd = 2.0 * osc._kp.sqrt()
        osc.cfg.rot_scale = 0.15
        osc.cfg.kp_null = 3.0
        osc.cfg.kd_null = 2.0 * math.sqrt(3.0)
        self.osc = osc
        self.n_act = self.robot.action_dim
        self.ctrl_hz = 1.0 / (env.dt * self.robot.control_period)
        self.origin = env.iscene.env_origins[0]
        self.sim_t = 0.0

    def V3(self, x, y, z):
        return torch.tensor([float(x), float(y), float(z)], device=self.device)

    def ee_pose(self):
        return self.art.data.body_pos_w[0, self.ee_idx], self.art.data.body_quat_w[0, self.ee_idx]

    def width(self) -> float:
        return 2.0 * self.art.data.joint_pos[0, self.fj1].item()

    def tip_pos(self):
        p, q = self.ee_pose()
        zh = quat_apply(q.unsqueeze(0), self.V3(0, 0, 1).unsqueeze(0))[0]
        return p + FINGER_LEN * zh

    def hand_for_tip(self, tip, quat):
        zh = quat_apply(quat.unsqueeze(0), self.V3(0, 0, 1).unsqueeze(0))[0]
        return tip - FINGER_LEN * zh

    def jaw_quat(self, azimuth: float):
        yh = self.V3(math.cos(azimuth), math.sin(azimuth), 0.0)
        zh = self.V3(0.0, 0.0, -1.0)
        xh = torch.cross(yh, zh, dim=0)
        return quat_from_matrix(torch.stack([xh, yh, zh], dim=1).unsqueeze(0))[0]

    def tilt_quat(self, azimuth: float, target_xy, near: float = 0.37, max_tilt: float = 0.35):
        gq = self.jaw_quat(azimuth)
        base_xy = self.art.data.root_pos_w[0, :2]
        d = float((target_xy - base_xy).norm())
        if d < near:
            tilt = min(max_tilt, (near - d) * 5.0)
            u = (target_xy - base_xy) / max(d, 1e-6)
            axis = self.V3(-u[1], u[0], 0.0)
            gq = quat_mul(quat_from_angle_axis(
                torch.tensor([tilt], device=self.device), axis.unsqueeze(0))[0].unsqueeze(0),
                gq.unsqueeze(0))[0]
        return gq

    def SEC(self, s: float) -> int:
        return max(1, round(s * self.ctrl_hz))

    def servo(self, goal_pos, goal_quat, grip, a, xy_boost: float = 1.0):
        p, q = self.ee_pose()
        err = goal_pos - p
        err = torch.cat([err[:2] * xy_boost, err[2:3]])
        a[0, 0:3] = (err / self.osc.cfg.pos_scale).clamp(-1.0, 1.0)
        qe = quat_mul(goal_quat.unsqueeze(0), quat_conjugate(q.unsqueeze(0)))
        a[0, 3:6] = (axis_angle_from_quat(qe)[0] / self.osc.cfg.rot_scale).clamp(-1.0, 1.0)
        a[0, 6:8] = grip

    def tick(self, a):
        self.env.step(a)
        self.sim_t += self.env.dt * self.robot.control_period

    def hold(self, pos, quat, grip, secs: float, xy_boost: float = 1.0):
        for _ in range(self.SEC(secs)):
            a = torch.zeros(1, self.n_act, device=self.device)
            self.servo(pos, quat, grip, a, xy_boost)
            self.tick(a)

    def run_phase(self, goal_fn, gate_fn, grip, timeout_s: float, tag: str = "",
                  xy_boost: float = 1.0, abort_fn=None) -> bool:
        deadline = self.sim_t + timeout_s
        while self.sim_t < deadline:
            a = torch.zeros(1, self.n_act, device=self.device)
            gp, gq = goal_fn()
            self.servo(gp, gq, grip, a, xy_boost)
            self.tick(a)
            if abort_fn is not None and abort_fn():
                return False
            if gate_fn():
                return True
        if tag:
            p, _ = self.ee_pose()
            gp, _ = goal_fn()
            print(f"[phase:{tag}] timeout: ee=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
                  f"goal=({gp[0]:.3f},{gp[1]:.3f},{gp[2]:.3f}) "
                  f"err={float((gp - p).norm()) * 1000:.0f}mm w={self.width() * 1000:.1f}mm",
                  flush=True)
        return False

    def close_ramp(self, pos, quat, grip_target: float, secs: float = 1.2,
                   grip_from: float = OPEN):
        n = self.SEC(secs)
        for k in range(n):
            a = torch.zeros(1, self.n_act, device=self.device)
            f = min(1.0, (k + 1) / (n * 0.7))
            self.servo(pos, quat, grip_from + (grip_target - grip_from) * f, a)
            self.tick(a)

    def dewind(self, grip: float = OPEN) -> bool:
        q = self.art.data.joint_pos[0]
        if (abs(q[0].item()) > 2.6 or q[3].item() < -2.95 or q[3].item() > -0.15
                or abs(q[6].item()) > 2.6):
            print(f"[dewind] wound arm (q1={q[0]:.2f} q4={q[3]:.2f} q7={q[6]:.2f}) — reset",
                  flush=True)
            self.robot.reset(torch.tensor([0], device=self.device, dtype=torch.long))
            p, qq = self.ee_pose()
            self.hold(p, qq, grip, 0.5)
            return True
        return False


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()

    rcfg = FrankaRobotCfg(base_pos=BASE_POS, nullspace_dof_pos=(),
                          gripper_effort_limit=120.0, gripper_stiffness=4000.0)
    cfg = EnvCfg(scene="latch_drawer", robot="franka", control_mode="osc",
                 env_spacing=3, robot_cfg=rcfg)
    cfg.seed = args.seed  # EnvCfg.seed defaults to 0 and build() re-seeds all RNGs
    env = cfg.build(num_envs=1, device=device)
    torch.manual_seed(args.seed)
    env.reset(seed=args.seed)
    s = Session(env)
    scene = env.scene
    c = scene.cfg
    V3 = s.V3

    PLATE_Z = c.drw_floor_z + c.plate_cz  # press height (plate center)

    def sc() -> float:
        return float(scene.score()[0])

    def disp() -> float:
        return float(scene.drawer_disp()[0])

    def cube_pos():
        return scene.cube.data.root_pos_w[0] - s.origin

    def pad_xy():
        return scene._pad_xy[0]

    def cube_yaw() -> float:
        ex = quat_apply(scene.cube.data.root_quat_w[0:1],
                        torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
        return math.atan2(float(ex[1]), float(ex[0]))

    def wrap_pi(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    def report(tag: str) -> None:
        cp = cube_pos()
        px = pad_xy()
        print(f"[solve] {tag:12s} disp={disp() * 1000:+7.1f}mm armed={bool(scene._armed[0])} "
              f"released={bool(scene._released[0])} "
              f"cube=({cp[0]:+.3f},{cp[1]:+.3f},{cp[2]:+.3f}) pad=({px[0]:+.3f},{px[1]:+.3f}) "
              f"in_drw={bool(scene.cube_in_drawer()[0])} on_pad={bool(scene.cube_on_pad()[0])} "
              f"score={sc():.2f} success={bool(scene.success()[0])}", flush=True)

    home_az = math.atan2(
        float(quat_apply(s.ee_pose()[1].unsqueeze(0),
                         torch.tensor([[0.0, 1.0, 0.0]], device=device))[0][1]),
        float(quat_apply(s.ee_pose()[1].unsqueeze(0),
                         torch.tensor([[0.0, 1.0, 0.0]], device=device))[0][0]))

    def best_az(candidates) -> float:
        best, err = candidates[0], 1e9
        for az in candidates:
            for br in (az, az + math.pi):
                e = abs(wrap_pi(br - home_az))
                if e < err:
                    err, best = e, br
        return best

    PARK = V3(0.02, 0.0, 0.42)
    park_q = s.jaw_quat(home_az)

    def park(grip=OPEN):
        s.dewind()
        p, q = s.ee_pose()
        s.hold(V3(p[0], p[1], max(float(p[2]), 0.40)), q, grip, 0.8)
        s.hold(PARK, park_q, grip, 1.2)

    TRAVEL_Z = 0.34

    # ----- mission --------------------------------------------------------------------------------
    print(f"[solve] ctrl={s.ctrl_hz:.0f}Hz seed={args.seed} base={BASE_POS}", flush=True)
    print(env.describe(), flush=True)
    s.hold(*s.ee_pose(), OPEN, 1.2)
    dp = scene.decoy.data.root_pos_w[0] - s.origin
    print(f"[solve] layout readback: pad=({float(pad_xy()[0]):+.3f},{float(pad_xy()[1]):+.3f}) "
          f"cube=({float(cube_pos()[0]):+.3f},{float(cube_pos()[1]):+.3f}) "
          f"decoy=({float(dp[0]):+.3f},{float(dp[1]):+.3f})", flush=True)
    report("start")
    print(f"SIM_GEN_SCORE {sc():.3f}", flush=True)

    # press pose: 30-deg forward pitch (probed on the forge: 30 deg aligns to 5 mm and
    # presses cleanly; 45 deg rides the q5 limit; 60 deg hits the q6 limit; a vertical
    # finger cannot reach the recessed plate at all — its column would cross the
    # carcass top-panel front edge). Jaw opens along y (home branch): both fingertips
    # press the plate symmetrically.
    press_pitch = quat_from_angle_axis(
        torch.tensor([math.radians(-30.0)], device=device),
        torch.tensor([[0.0, 1.0, 0.0]], device=device))[0]  # tip leads forward-down (+x)
    press_q = quat_mul(press_pitch.unsqueeze(0),
                       s.jaw_quat(best_az([math.pi / 2])).unsqueeze(0))[0]

    def press_tip(x, z=PLATE_Z):
        return s.hand_for_tip(V3(x, 0.0, z), press_q), press_q

    # =================== PHASE 1: press the latch, let the drawer eject ==========================
    ejected = False
    for attempt in range(3):
        # fresh arm configuration, then servo DIRECTLY to the align pose (probed: an
        # intermediate hover can bend the arm into a bad configuration branch)
        s.robot.reset(torch.tensor([0], device=device, dtype=torch.long))
        s.hold(*s.ee_pose(), PRESS_GRIP, 0.5)
        gp0, _ = press_tip(0.26)
        s.hold(gp0, press_q, PRESS_GRIP, 4.0)
        t0 = s.tip_pos()
        perr = float((t0 - V3(0.26, 0.0, PLATE_Z)).norm())
        print(f"[solve] press{attempt}-align: tip=({t0[0]:.3f},{t0[1]:.3f},{t0[2]:.3f}) "
              f"err={perr * 1000:.0f}mm", flush=True)
        if perr > 0.02:
            continue
        # quasi-static press ramp: tip target from just before the face to face + lead
        n = s.SEC(4.0 + attempt * 1.5)
        lead = 0.075 + attempt * 0.015
        armed = False
        for k in range(n):
            a = torch.zeros(1, s.n_act, device=device)
            tx = 0.27 + lead * min(1.0, (k + 1) / (n * 0.85))
            gp, gq = press_tip(tx)
            s.servo(gp, gq, PRESS_GRIP, a)
            s.tick(a)
            if bool(scene._armed[0]) and disp() > c.press_arm + 0.0015:
                armed = True
                break
        tp_dbg = s.tip_pos()
        print(f"[solve] press{attempt}: disp={disp() * 1000:+.1f}mm armed={armed} "
              f"tip=({tp_dbg[0]:.3f},{tp_dbg[1]:.3f},{tp_dbg[2]:.3f})", flush=True)
        if not armed:
            continue
        print(f"SIM_GEN_SCORE {sc():.3f}", flush=True)
        # withdraw back-and-up: the latch releases behind us and the spring ejects
        s.run_phase(lambda: press_tip(0.10, 0.30),
                    lambda: float(s.tip_pos()[0]) < 0.17 and float(s.tip_pos()[2]) > 0.20,
                    PRESS_GRIP, 3.0, tag=f"press{attempt}-withdraw")
        park()
        deadline = s.sim_t + 10.0
        while s.sim_t < deadline:
            s.hold(PARK, park_q, OPEN, 0.25)
            if (bool(scene._released[0]) and float(scene.drawer_open()[0]) > 0.10
                    and bool(scene.drawer_still()[0]) and bool(scene.cube_settled()[0])):
                ejected = True
                break
        report(f"eject{attempt}")
        if ejected:
            break

    if not ejected:
        print("[solve] FAILED to eject the drawer", flush=True)
        report("verdict")
        print(f"SIM_GEN_SCORE {sc():.3f}", flush=True)
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    print(f"SIM_GEN_SCORE {sc():.3f}", flush=True)

    # =================== PHASE 2+3: pick the cube out, place it on the pad =======================
    placed = False
    CAGE = 0.032  # pre-closed descent: fingers at 64 mm, clear of the cavity walls
    for attempt in range(4):
        if bool(scene.success()[0]):
            placed = True
            break
        # fresh arm configuration: the press leaves the arm in a pitched/wound branch
        # that fights the swing to the vertical pick pose (measured: 1-3 wasted hovers)
        s.robot.reset(torch.tensor([0], device=device, dtype=torch.long))
        s.hold(*s.ee_pose(), OPEN, 0.5)
        park()
        yaw = cube_yaw()
        az = best_az([yaw + math.pi / 2])  # jaw opens across the drawer width (y-ish)
        gq = s.tilt_quat(az, cube_pos()[:2])

        def tip_goal(z):
            o = cube_pos()
            return s.hand_for_tip(V3(float(o[0]), float(o[1]), z), gq), gq

        def grasp_z() -> float:
            return float(cube_pos()[2]) + 0.002

        if not s.run_phase(lambda: tip_goal(0.30),
                           lambda: float((s.ee_pose()[0][:2] - cube_pos()[:2]).norm()) < 0.006,
                           OPEN, 8.0, tag=f"cube{attempt}-hover"):
            continue
        if not s.run_phase(lambda: tip_goal(grasp_z()),
                           lambda: abs(float(s.tip_pos()[2]) - grasp_z()) < 0.005
                           and float((s.ee_pose()[0][:2] - cube_pos()[:2]).norm()) < 0.006,
                           CAGE, 8.0, tag=f"cube{attempt}-descend"):
            continue
        gp, _ = tip_goal(grasp_z())
        s.hold(gp, gq, CAGE, 0.4)
        s.close_ramp(gp, gq, 0.019, secs=1.3, grip_from=CAGE)
        z0 = float(cube_pos()[2])
        p, _ = s.ee_pose()
        s.hold(V3(float(p[0]), float(p[1]), 0.22), gq, 0.019, 0.9)
        s.hold(V3(float(p[0]), float(p[1]), TRAVEL_Z), gq, 0.019, 0.9)
        w = s.width()
        rise = float(cube_pos()[2]) - z0
        held = rise > 0.035 and 0.040 < w < 0.050
        print(f"[solve] cube{attempt}: rise={rise * 1000:.0f}mm w={w * 1000:.1f}mm held={held}",
              flush=True)
        if not held:
            p, _ = s.ee_pose()
            s.hold(V3(float(p[0]), float(p[1]), 0.30), gq, OPEN, 0.8)
            continue
        print(f"SIM_GEN_SCORE {sc():.3f}", flush=True)

        def dropped() -> bool:
            return float(cube_pos()[2]) < 0.05 and float(s.ee_pose()[0][2]) > 0.25

        def carry_goal():
            b = cube_pos()
            h, _ = s.ee_pose()
            t = pad_xy()
            corr = (1.3 * (t - b[:2])).clamp(-0.010, 0.010)
            return V3(float(h[0]) + float(corr[0]), float(h[1]) + float(corr[1]),
                      TRAVEL_Z), gq

        if not s.run_phase(carry_goal,
                           lambda: float((cube_pos()[:2] - pad_xy()).norm()) < 0.008,
                           0.019, 16.0, tag=f"cube{attempt}-carry", xy_boost=1.3,
                           abort_fn=dropped):
            if dropped() or float(cube_pos()[2]) < 0.05:
                print("[solve] cube lost mid-carry — re-pick", flush=True)
                continue

        rest_z = c.pad_t + c.cube_size / 2  # cube center resting on the pad

        def lower_goal():
            b = cube_pos()
            h, _ = s.ee_pose()
            t = pad_xy()
            corr = (1.2 * (t - b[:2])).clamp(-0.008, 0.008)
            dz = float(h[2]) - float(b[2])
            return V3(float(h[0]) + float(corr[0]), float(h[1]) + float(corr[1]),
                      rest_z + 0.004 + dz), gq

        if not s.run_phase(lower_goal,
                           lambda: abs(float(cube_pos()[2]) - rest_z) < 0.006
                           and float((cube_pos()[:2] - pad_xy()).norm()) < 0.012,
                           0.019, 10.0, tag=f"cube{attempt}-lower", xy_boost=1.2,
                           abort_fn=dropped):
            if dropped() or float(cube_pos()[2]) < 0.05:
                print("[solve] cube lost during lower — re-pick", flush=True)
                continue

        p, q = s.ee_pose()
        s.hold(p, q, 0.019, 0.5)
        s.hold(p, q, OPEN, 0.8)
        p, _ = s.ee_pose()
        s.hold(V3(float(p[0]), float(p[1]), TRAVEL_Z + 0.05), park_q, OPEN, 1.0)
        park()
        deadline = s.sim_t + 8.0
        while s.sim_t < deadline:
            s.hold(PARK, park_q, OPEN, 0.25)
            if bool(scene.success()[0]):
                placed = True
                break
        report(f"place{attempt}")
        if placed:
            break

    # =================== verdict ===================================================================
    park()
    s.hold(PARK, park_q, OPEN, 2.0)
    report("verdict")
    ok = bool(scene.success()[0])
    print(f"SIM_GEN_SCORE {sc():.3f}", flush=True)
    if ok:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL", flush=True)
    code = 0 if ok else 1
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()

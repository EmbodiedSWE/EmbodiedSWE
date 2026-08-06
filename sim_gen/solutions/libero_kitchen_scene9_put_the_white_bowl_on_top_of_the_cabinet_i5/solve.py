"""solve — real Franka-arm solution for CounterweightShelfScene
(libero_kitchen_scene9_put_the_white_bowl_on_top_of_the_cabinet_i5).

Single Franka + parallel jaw, OSC, base at (-0.25, 0, 0) facing +x. Strategy (the task's
own intended plan — the seed's bare pick-and-place fails on the tipped shelf):

  PHASE 1  pick the dark counterweight cube (top-down pinch across two faces) and drop
           it into the shelf's rear socket (computed LIVE from the tray pose — the
           socket rides on the tipped tray); the rear torque swings the shelf level
           against its stop.
  PHASE 2  pick the white bowl by its 12 mm rim (cage-then-squeeze: one finger inside,
           one outside the wall), carry it over the now-level front platform, lower
           until the bowl rests, release slowly, retreat; settle until scene.success().

Arm-only manipulation: the robot commands ONLY its own joints/gripper; task-object
state is never written and no external forces are applied.

Prints the scene readouts, `SIM_GEN_SCORE <score>` at each phase boundary (latched
credit — never decreases), and `SIM_GEN_SOLVE: SUCCESS` at the end.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene9_put_the_white_bowl_on_top_of_the_cabinet_i5.solve --headless
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_sec", type=float, default=300.0)
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
    from simgen_tasks.libero_kitchen_scene9_put_the_white_bowl_on_top_of_the_cabinet_i5 import (  # noqa: F401,E501
        scene as scene_mod,
    )
except ImportError:
    import scene as scene_mod  # noqa: F401

OPEN = 0.04
FINGER_LEN = 0.112
BASE_POS = (-0.25, 0.0, 0.0)


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

    def close_ramp(self, pos, quat, grip_target: float, secs: float = 1.2):
        n = self.SEC(secs)
        for k in range(n):
            a = torch.zeros(1, self.n_act, device=self.device)
            f = min(1.0, (k + 1) / (n * 0.7))
            self.servo(pos, quat, OPEN + (grip_target - OPEN) * f, a)
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
    torch.manual_seed(args.seed)
    robobench.discover()

    rcfg = FrankaRobotCfg(base_pos=BASE_POS, nullspace_dof_pos=(),
                          gripper_effort_limit=120.0, gripper_stiffness=4000.0)
    cfg = EnvCfg(scene="counterweight_shelf", robot="franka", control_mode="osc",
                 env_spacing=3, robot_cfg=rcfg)
    cfg.seed = args.seed  # EnvCfg.seed defaults to 0 and build() re-seeds all RNGs
    env = cfg.build(num_envs=1, device=device)
    env.reset(seed=args.seed)
    s = Session(env)
    scene = env.scene
    c = scene.cfg
    V3 = s.V3

    def sc() -> float:
        return float(scene.score()[0])

    def ang_deg() -> float:
        return math.degrees(float(scene.tray_angle()[0]))

    def report(tag: str) -> None:
        bl = scene._to_tray_frame(scene.block.data.root_pos_w)[0]
        wl = scene._to_tray_frame(scene.bowl.data.root_pos_w)[0]
        print(f"[solve] {tag:12s} ang={ang_deg():+7.2f} "
              f"block_loc=({bl[0]:+.3f},{bl[1]:+.3f},{bl[2]:+.3f}) "
              f"bowl_loc=({wl[0]:+.3f},{wl[1]:+.3f},{wl[2]:+.3f}) "
              f"seated={bool(scene.block_seated()[0])} level={bool(scene.tray_level()[0])} "
              f"on_plat={bool(scene.bowl_on_platform()[0])} settled={bool(scene.settled()[0])} "
              f"score={sc():.2f} success={bool(scene.success()[0])}", flush=True)

    def block_pos():
        return scene.block.data.root_pos_w[0] - s.origin

    def bowl_pos():
        return scene.bowl.data.root_pos_w[0] - s.origin

    def block_yaw() -> float:
        ex = quat_apply(scene.block.data.root_quat_w[0:1],
                        torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
        return math.atan2(float(ex[1]), float(ex[0]))

    def bowl_yaw() -> float:
        ex = quat_apply(scene.bowl.data.root_quat_w[0:1],
                        torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
        return math.atan2(float(ex[1]), float(ex[0]))

    def tray_local_to_world(loc):
        p = quat_apply(scene.tray.data.root_quat_w[0:1],
                       torch.tensor([loc], device=device))[0]
        return p + scene.tray.data.root_pos_w[0] - s.origin

    def wrap_pi(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

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

    PARK = V3(0.05, 0.0, 0.42)
    park_q = s.jaw_quat(home_az)

    def park(grip=OPEN):
        s.dewind()
        p, q = s.ee_pose()
        s.hold(V3(p[0], p[1], max(float(p[2]), 0.42)), q, grip, 0.8)
        s.hold(PARK, park_q, grip, 1.4)

    TRAVEL_Z = 0.44  # hand height for loaded transit (clears the tipped tray rear arm)

    # ----- generic top-down pick ----------------------------------------------------------------
    def pick(obj_pos_fn, jaw_az: float, tip_z_fn, grip_target: float,
             width_lo: float, width_hi: float, min_rise: float, tag: str):
        gq = s.tilt_quat(jaw_az, obj_pos_fn()[:2])

        def tip_goal(z):
            o = obj_pos_fn()
            return s.hand_for_tip(V3(o[0], o[1], z), gq), gq

        if not s.run_phase(lambda: tip_goal(0.30),
                           lambda: (s.ee_pose()[0][:2] - obj_pos_fn()[:2]).norm() < 0.006,
                           OPEN, 8.0, tag=f"{tag}-hover"):
            return False, gq
        if not s.run_phase(lambda: tip_goal(tip_z_fn()),
                           lambda: abs(float(s.tip_pos()[2]) - tip_z_fn()) < 0.005
                           and (s.ee_pose()[0][:2] - obj_pos_fn()[:2]).norm() < 0.006,
                           OPEN, 8.0, tag=f"{tag}-descend"):
            return False, gq
        # micro-stabilise then ramped close
        gp, _ = tip_goal(tip_z_fn())
        s.hold(gp, gq, OPEN, 0.4)
        s.close_ramp(gp, gq, grip_target, secs=1.3)
        z0 = float(obj_pos_fn()[2])
        p, _ = s.ee_pose()
        s.hold(V3(p[0], p[1], TRAVEL_Z), gq, grip_target, 1.8)
        w = s.width()
        rise = float(obj_pos_fn()[2]) - z0
        held = rise > min_rise and width_lo < w < width_hi
        print(f"[solve] {tag}: rise={rise * 1000:.0f}mm w={w * 1000:.1f}mm held={held}",
              flush=True)
        if not held:
            p, _ = s.ee_pose()
            s.hold(V3(p[0], p[1], 0.30), gq, OPEN, 0.8)
        return held, gq

    # ----- mission --------------------------------------------------------------------------------
    print(f"[solve] ctrl={s.ctrl_hz:.0f}Hz seed={args.seed} base={BASE_POS}", flush=True)
    print(env.describe(), flush=True)
    s.hold(*s.ee_pose(), OPEN, 1.5)
    report("start")
    print(f"SIM_GEN_SCORE {sc():.3f}", flush=True)

    # =================== PHASE 1: counterweight into the socket =================================
    seated = False
    for attempt in range(4):
        if bool(scene.block_seated()[0]) and bool(scene.tray_level()[0]):
            seated = True
            break
        park()
        bp = block_pos()
        if float(bp[2]) > 0.15:
            # block ended up somewhere elevated (e.g. on the tray) — nudge attempts only
            print(f"[solve] block elevated at z={float(bp[2]):.3f}, retrying pick anyway",
                  flush=True)
        yaw = block_yaw()
        az = best_az([yaw, yaw + math.pi / 2])
        ok, gq_pick = pick(block_pos, az,
                           lambda: float(block_pos()[2]) + 0.002,
                           grip_target=0.018, width_lo=0.040, width_hi=0.050,
                           min_rise=0.035, tag=f"block{attempt}")
        if not ok:
            continue

        # carry over the socket (live: the socket rides on the tipped tray). Aim at the
        # midpoint of the projected inner-rim opening; drop from low above the rim plane.
        def sock_drop_goal():
            th = float(scene.tray_angle()[0])
            # inner-rim top corners (tray frame): downhill (x_lo) and uphill (x_hi)
            x_lo = c.sock_cx - c.sock_in_x / 2
            x_hi = c.sock_cx + c.sock_in_x / 2
            p_lo = tray_local_to_world([x_lo, 0.0, c.sock_wall_h])
            p_hi = tray_local_to_world([x_hi, 0.0, c.sock_wall_h])
            mid = (p_lo + p_hi) / 2
            # block center target: above the rim midpoint by half block + clearance
            tgt = V3(float(mid[0]), float(mid[1]), float(max(p_lo[2], p_hi[2])) + 0.028)
            bp2 = block_pos()
            h, _ = s.ee_pose()
            corr = (1.3 * (tgt[:2] - bp2[:2])).clamp(-0.010, 0.010)
            gz = min(float(tgt[2]) + (float(h[2]) - float(bp2[2])), 0.60)
            return V3(float(h[0]) + float(corr[0]), float(h[1]) + float(corr[1]),
                      gz), gq_pick

        def over_socket():
            th = float(scene.tray_angle()[0])
            x_mid_l = [c.sock_cx, 0.0, c.sock_wall_h]
            mid = tray_local_to_world(x_mid_l)
            bp2 = block_pos()
            return ((bp2[:2] - mid[:2]).norm() < 0.010
                    and float(bp2[2]) - float(mid[2]) < 0.045)

        # first go high to travel, then descend onto the drop point (SAME wrist
        # orientation as the grasp — re-orienting a loaded jaw flings the block)
        p, _ = s.ee_pose()
        s.hold(V3(p[0], p[1], TRAVEL_Z), gq_pick, 0.018, 0.9)

        def block_dropped() -> bool:
            return float(block_pos()[2]) < 0.08 and float(s.ee_pose()[0][2]) > 0.30

        if not s.run_phase(sock_drop_goal, over_socket, 0.018, 16.0,
                           tag=f"carry-block{attempt}", xy_boost=1.2,
                           abort_fn=block_dropped):
            if block_dropped() or float(block_pos()[2]) < 0.06:
                bp3 = block_pos()
                h3, _ = s.ee_pose()
                print(f"[solve] block lost during carry (block_z={float(bp3[2]):.3f} "
                      f"hand_z={float(h3[2]):.3f} w={s.width() * 1000:.1f}mm) — re-pick",
                      flush=True)
                continue
        # stillness, then slow release
        gp2, gq2 = sock_drop_goal()
        s.hold(gp2, gq2, 0.018, 0.6)
        p, q = s.ee_pose()
        s.hold(p, q, OPEN, 0.8)
        s.hold(V3(float(p[0]), float(p[1]), TRAVEL_Z + 0.04), q, OPEN, 1.0)
        park()
        # wait for the swing: seated + level + still
        deadline = s.sim_t + 8.0
        while s.sim_t < deadline:
            s.hold(*s.ee_pose(), OPEN, 0.25)
            if bool(scene.block_seated()[0]) and bool(scene.tray_level()[0]) \
                    and bool(scene.tray_still()[0]):
                seated = True
                break
        report(f"seat{attempt}")
        if seated:
            break
        if float(block_pos()[2]) < 0.06:
            print("[solve] block missed the socket, on the floor — retry", flush=True)
            continue

    if not seated:
        print("[solve] FAILED to seat the counterweight", flush=True)
        report("verdict")
        print(f"SIM_GEN_SCORE {sc():.3f}", flush=True)
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    print(f"SIM_GEN_SCORE {sc():.3f}", flush=True)

    # =================== PHASE 2: bowl onto the level platform ==================================
    placed = False
    for attempt in range(4):
        if bool(scene.success()[0]):
            placed = True
            break
        park()
        # rim grasp: pinch one octagon wall segment. Wall mid-plane radius:
        r_mid = c.bowl_inner_r + c.bowl_wall_t / 2
        byaw0 = bowl_yaw()
        # choose the wall segment (index) whose outward normal is most wrist-friendly,
        # then TRACK that segment live (the bowl can rotate under finger contact)
        k0, az0, err0 = 0, byaw0, 1e9
        for k in range(8):
            azc = byaw0 + k * math.pi / 4
            for br in (azc, azc + math.pi):
                e = abs(wrap_pi(br - home_az))
                if e < err0:
                    err0, k0, az0 = e, k, br
        flip = abs(wrap_pi(az0 - (byaw0 + k0 * math.pi / 4))) > 1.0

        def wall_az() -> float:
            azc = bowl_yaw() + k0 * math.pi / 4
            return azc + math.pi if flip else azc

        def u_live():
            azc = bowl_yaw() + k0 * math.pi / 4  # outward normal of segment k0
            return V3(math.cos(azc), math.sin(azc), 0.0)

        def rim_point():
            b = bowl_pos()
            return b[:2] + r_mid * u_live()[:2]

        gq = s.tilt_quat(az0, bowl_pos()[:2])

        def rim_tip_goal(z):
            xy = rim_point()
            return s.hand_for_tip(V3(float(xy[0]), float(xy[1]), z), gq), gq

        rim_top = float(bowl_pos()[2]) + c.bowl_h / 2
        grasp_z = rim_top - 0.022  # fingers straddle the top 22 mm of the wall
        # CAGE: pre-closed to wall + 14 mm so the light bowl cannot squirt away when a
        # pad makes first contact during the close (the pen-corpus cage lesson)
        CAGE = (c.bowl_wall_t + 0.014) / 2

        if not s.run_phase(lambda: rim_tip_goal(0.30),
                           lambda: (s.ee_pose()[0][:2] - rim_point()).norm() < 0.0035,
                           OPEN, 8.0, tag=f"bowl{attempt}-hover"):
            continue
        if not s.run_phase(lambda: rim_tip_goal(grasp_z),
                           lambda: abs(float(s.tip_pos()[2]) - grasp_z) < 0.005
                           and (s.ee_pose()[0][:2] - rim_point()).norm() < 0.0035,
                           CAGE, 8.0, tag=f"bowl{attempt}-descend"):
            continue
        # micro-stabilise on the LIVE rim point, then slow two-stage squeeze
        for _ in range(s.SEC(0.6)):
            a = torch.zeros(1, s.n_act, device=device)
            gp, _ = rim_tip_goal(grasp_z)
            s.servo(gp, gq, CAGE, a)
            s.tick(a)
        gp, _ = rim_tip_goal(grasp_z)
        n_close = s.SEC(1.5)
        for k2 in range(n_close):
            a = torch.zeros(1, s.n_act, device=device)
            gp, _ = rim_tip_goal(grasp_z)
            g2 = CAGE + (0.004 - CAGE) * min(1.0, (k2 + 1) / (n_close * 0.7))
            s.servo(gp, gq, g2, a)
            s.tick(a)
        w = s.width()
        if w < 0.008:  # empty close: the wall is not between the pads
            print(f"[solve] bowl close missed the wall (w={w * 1000:.1f}mm), retry",
                  flush=True)
            p, _ = s.ee_pose()
            s.hold(V3(float(p[0]), float(p[1]), 0.30), gq, OPEN, 0.8)
            continue
        z0 = float(bowl_pos()[2])
        p, _ = s.ee_pose()
        # slow straight lift (the rim pinch is torque-loaded — no jerk)
        for zt in (0.24, 0.34, TRAVEL_Z):
            s.hold(V3(float(p[0]), float(p[1]), zt), gq, 0.004, 0.7)
        w = s.width()
        rise = float(bowl_pos()[2]) - z0
        if not (rise > 0.05 and 0.008 < w < 0.022):
            print(f"[solve] bowl grasp failed (rise={rise * 1000:.0f}mm w={w * 1000:.1f}mm)",
                  flush=True)
            p, _ = s.ee_pose()
            s.hold(V3(float(p[0]), float(p[1]), 0.30), gq, OPEN, 0.8)
            continue
        print(f"[solve] bowl HELD (w={w * 1000:.1f}mm)", flush=True)
        print(f"SIM_GEN_SCORE {sc():.3f}", flush=True)

        def dropped() -> bool:
            return float(bowl_pos()[2]) < 0.10 and float(s.ee_pose()[0][2]) > 0.30

        # carry: closed-loop on the BOWL center onto the platform target (tray frame)
        tgt_loc = [(c.place_x_lo + c.place_x_hi) / 2, 0.0, c.bowl_h / 2]

        def place_tgt():
            return tray_local_to_world(tgt_loc)

        def carry_goal():
            b = bowl_pos()
            h, _ = s.ee_pose()
            t = place_tgt()
            corr = (1.2 * (t[:2] - b[:2])).clamp(-0.008, 0.008)
            return V3(float(h[0]) + float(corr[0]), float(h[1]) + float(corr[1]),
                      TRAVEL_Z), gq

        if not s.run_phase(carry_goal,
                           lambda: (bowl_pos()[:2] - place_tgt()[:2]).norm() < 0.008,
                           0.004, 20.0, tag=f"bowl{attempt}-carry", xy_boost=1.2,
                           abort_fn=dropped):
            if dropped() or float(bowl_pos()[2]) < 0.06:
                print("[solve] bowl lost mid-carry — re-pick", flush=True)
                continue

        def lower_goal():
            b = bowl_pos()
            h, _ = s.ee_pose()
            t = place_tgt()
            corr = (1.2 * (t[:2] - b[:2])).clamp(-0.008, 0.008)
            dz = float(h[2]) - float(b[2])
            return V3(float(h[0]) + float(corr[0]), float(h[1]) + float(corr[1]),
                      float(t[2]) + 0.004 + dz), gq

        if not s.run_phase(lower_goal,
                           lambda: abs(float(bowl_pos()[2]) - float(place_tgt()[2])) < 0.006
                           and (bowl_pos()[:2] - place_tgt()[:2]).norm() < 0.010,
                           0.004, 10.0, tag=f"bowl{attempt}-lower", xy_boost=1.2,
                           abort_fn=dropped):
            if dropped() or float(bowl_pos()[2]) < 0.06:
                print("[solve] bowl lost during lower — re-pick", flush=True)
                continue

        # pre-release stillness, slow release, retreat
        p, q = s.ee_pose()
        s.hold(p, q, 0.004, 0.6)
        s.hold(p, q, OPEN, 0.8)
        p, _ = s.ee_pose()
        s.hold(V3(float(p[0]), float(p[1]), TRAVEL_Z + 0.05), park_q, OPEN, 1.0)
        park()
        # settle until success
        deadline = s.sim_t + 8.0
        while s.sim_t < deadline:
            s.hold(*s.ee_pose(), OPEN, 0.25)
            if bool(scene.success()[0]):
                placed = True
                break
        report(f"place{attempt}")
        if placed:
            break
        if float(bowl_pos()[2]) < 0.06:
            print("[solve] bowl slid off / missed — retry", flush=True)

    # =================== verdict ===================================================================
    park()
    s.hold(*s.ee_pose(), OPEN, 2.0)
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

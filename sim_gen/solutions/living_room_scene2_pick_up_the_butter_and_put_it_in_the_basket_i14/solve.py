"""solve — real single-Franka solution for ButterHopperScene (the feasibility certificate).

Env: scene "butter_hopper" + robot "franka" (OSC), base at (-0.50, 0, 0) facing +x — the
scene is laid out around this pose: shaft axis 0.60 m ahead, handle-fin pull sweep at
0.60-0.66 m horizontal / 0.45 m hand height, basket spawn arc 0.44-0.54 m from the base.

Plan (execution order REQUIRED: basket first — the drop is irreversible):
  SETTLE      let the layout come to rest; read the scene
  GRASP       rim-pinch the basket's most robot-facing wall (12 mm wall, cage 26 mm,
              quasi-static squeeze to 8 mm), lift + verdict (basket rose, width band)
  CARRY       closed-loop on the BASKET center to a staging spot 8 cm in front of the
              drop zone, with a gentle continuous yaw servo squaring the basket mod 90
  PLACE       set down, slow release, retreat
  PUSH        fingertips low against the basket's -x wall, push it the last 8 cm onto
              the green drop-zone square (closed-loop with lateral steering) — the hand
              stays far below the shaft mouth the whole time
  PULL        pinch the yellow handle fin (12 mm, positive engagement) and pull the
              tray out along -y until the scene reads the opening past 0.165 m; the
              butter is scraped off the receding tray and falls into the basket
  VERIFY      wait for the scene's own sustained-success counter, then keep simulating
              3 s more: latched credit and success must persist

Arm-only manipulation: no task-object state writes, no external forces. Prints
SIM_GEN_SCORE at each phase boundary (latched credit never decreases) and
SIM_GEN_SOLVE: SUCCESS at the end.

Run: python -u -m simgen_tasks.living_room_scene2_pick_up_the_butter_and_put_it_in_the_basket_i14.solve --headless [--seed N]
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

from simgen_tasks.living_room_scene2_pick_up_the_butter_and_put_it_in_the_basket_i14 import (  # noqa: E402
    scene as scene_mod,  # noqa: F401
)

# Emergency watchdog: whatever happens, this process must die (Kit teardown hangs).
threading.Timer(args.max_wall_s, lambda: (print("[solve] WALL TIMEOUT", flush=True),
                                          os._exit(3))).start()

OPEN = 0.04          # per-finger position target, fully open (80 mm aperture)
FINGER_LEN = 0.112   # panda_hand frame -> fingertip along the approach axis
BASE_POS = (-0.50, 0.0, 0.0)
TRAVEL_Z = 0.30      # hand transit height while carrying the basket
CAGE_B = 0.013       # pre-close cage for the 12 mm basket wall (26 mm aperture)
GRIP_B = 0.003       # per-finger close target on the wall (6 mm — firm squeeze)
WB_LO, WB_HI = 0.008, 0.018  # jaw-width verdict band for a held 12 mm wall
GRIP_F = 0.019       # per-finger close target across the fin's 48 mm width (jaw along
                     # x: the hand's WIDE axis stays parallel to the dispenser face —
                     # a y-axis jaw physically cannot descend next to the wall/lid)
WF_LO, WF_HI = 0.040, 0.054  # jaw-width band for a held 48 mm fin
PRESTAGE_R = 0.065   # staging spot: this far out from the shaft axis ALONG the live
                     # face normal of the wall to be pushed (the push then lands on the
                     # stage point regardless of residual basket yaw)
PUSH_TIP_Z = 0.045   # fingertip height while pushing (low: keeps the wrist well under
                     # the tray plate, which now rides at z 0.336+)
FIN_TIP_Z = 0.385    # fingertip height on the fin (fin spans z 0.350-0.425)
PULL_OPEN = 0.165    # tray opening the pull drives to (success needs >= 0.14)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    robobench.discover()

    rcfg = FrankaRobotCfg(base_pos=BASE_POS, nullspace_dof_pos=(),
                          gripper_effort_limit=120.0, gripper_stiffness=4000.0)
    env = EnvCfg(scene="butter_hopper", robot="franka", control_mode="osc",
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

    def close_ramp(pos_fn, quat_fn, grip_from, grip_target, secs=1.4):
        n = SEC(secs)
        for k in range(n):
            a = torch.zeros(1, n_act, device=device)
            f = min(1.0, (k + 1) / (n * 0.7))
            servo(pos_fn(), quat_fn(), grip_from + (grip_target - grip_from) * f, a)
            tick(a)

    def wrap_pi(x: float) -> float:
        return (x + math.pi) % (2 * math.pi) - math.pi

    def jaw_az_of(q):
        ey = quat_apply(q.unsqueeze(0), torch.tensor([[0.0, 1.0, 0.0]], device=device))[0]
        return math.atan2(ey[1].item(), ey[0].item())

    # ----- scene readers ----------------------------------------------------------------------
    woff = c.basket_inner_half + c.basket_wall_t / 2  # wall centerline offset (0.091)

    def basket_pos():
        return scene.basket.data.root_pos_w[0]

    def basket_yaw() -> float:
        qw, qx, qy, qz = (float(v) for v in scene.basket.data.root_quat_w[0])
        return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

    def wall_center(k: int):
        """World center of wall k (normal at yaw + k*90deg), on the wall centerline."""
        az = basket_yaw() + k * math.pi / 2
        bp = basket_pos()
        return V3(bp[0] + woff * math.cos(az), bp[1] + woff * math.sin(az), bp[2]), az

    def stage_xy():
        return scene.shaft_axis_w()[0]

    def fin_xy():
        tp = scene.tray.data.root_pos_w[0]
        return V3(tp[0], tp[1] + c.fin_y_local, 0.0)[:2]

    def yaw_err90() -> float:
        """Basket yaw folded to the nearest multiple of 90 deg, in [-45, 45)."""
        y = basket_yaw()
        return (y + math.pi / 4) % (math.pi / 2) - math.pi / 4

    def face_frame(toward=None):
        """(normal az, outward normal n, tangent t) of a basket wall. Default: the
        most robot-facing wall (for grasp/staging). With `toward` (a world xy
        direction), the wall whose OUTWARD normal is most anti-parallel to it — the
        right wall to PUSH to move the basket that way (choosing the robot-facing
        wall froze the push when a dropped basket lay sideways of the target)."""
        if toward is None:
            k = min(range(4), key=lambda kk: math.cos(basket_yaw() + kk * math.pi / 2))
        else:
            k = min(range(4),
                    key=lambda kk: math.cos(basket_yaw() + kk * math.pi / 2)
                    * float(toward[0])
                    + math.sin(basket_yaw() + kk * math.pi / 2) * float(toward[1]))
        az = basket_yaw() + k * math.pi / 2
        n = V3(math.cos(az), math.sin(az), 0.0)
        t = V3(-math.sin(az), math.cos(az), 0.0)
        return az, n, t

    def push_dir():
        d = stage_xy() - basket_pos()[:2]
        nrm = float(d.norm())
        return d / max(nrm, 1e-6)

    def prestage_xy():
        """Staging spot: PRESTAGE_R out from the shaft axis along the live face normal
        — pushing along -n then lands on the stage point whatever the basket yaw."""
        _az, n, _t = face_frame()
        return stage_xy() + PRESTAGE_R * n[:2]

    def readout(tag: str) -> float:
        s = float(scene.score()[0])
        bp = (basket_pos() - origin)
        off = float((basket_pos()[:2] - stage_xy()).norm())
        print(f"[solve] {tag:12s} open={float(scene.opening()[0])*1000:5.1f}mm "
              f"basket=({bp[0]:+.3f},{bp[1]:+.3f}) stage_off={off*1000:4.0f}mm "
              f"contained={bool(scene.contained()[0])} score={s:.2f} "
              f"success={bool(scene.success()[0])} sim_t={sim_t['t']:.1f}s", flush=True)
        print(f"SIM_GEN_SCORE {s:.3f}", flush=True)
        return s

    # ----- mission ------------------------------------------------------------------------------
    print(f"[solve] seed={args.seed} ctrl={ctrl_hz:.0f}Hz base={BASE_POS}", flush=True)
    print(env.describe(), flush=True)

    hold(*ee_pose(), OPEN, 1.5)  # settle the layout
    bp = (basket_pos() - origin)
    bu = (scene.butter.data.root_pos_w[0] - origin)
    print(f"[solve] settled basket=({bp[0]:+.3f},{bp[1]:+.3f},{bp[2]:.3f}) "
          f"yaw={math.degrees(basket_yaw()):+.1f}deg "
          f"butter=({bu[0]:+.3f},{bu[1]:+.3f},{bu[2]:.3f})", flush=True)
    readout("settled")

    home_az = jaw_az_of(ee_pose()[1])
    park_q = jaw_quat(home_az)

    def park(grip=OPEN):
        dewind()
        p, q = ee_pose()
        hold(V3(p[0], p[1], max(float(p[2]), 0.35)), q, grip, 0.8)
        hold(origin + V3(-0.10, 0.0, 0.35), park_q, grip, 1.2)

    # --- GRASP + CARRY + PLACE, with mid-carry drop recovery ---------------------------------
    def clamp_goal(g):
        # never command the hand into the dispenser volume while handling the basket
        return V3(min(float(g[0]), 0.03), max(-0.45, min(0.45, float(g[1]))),
                  max(0.14, min(0.40, float(g[2]))))

    def basket_lost() -> bool:
        # jaw closed to its empty target AND the basket is not under the hand
        return (width() < 0.010
                and float((ee_pose()[0][:2] - basket_pos()[:2]).norm()) > 0.13)

    placed = False
    az_hold = 0.0
    for cycle in range(3):
        # GRASP the rim of the most robot-facing wall (up to 2 approaches per cycle)
        picked = False
        for attempt in range(2):
            dewind()
            k_best = min(range(4),
                         key=lambda k: math.cos(basket_yaw() + k * math.pi / 2))
            _, az_raw = wall_center(k_best)
            az_hold = min((az_raw, az_raw + math.pi, az_raw - math.pi),
                          key=lambda a: abs(wrap_pi(a - home_az)))

            def gq_now():
                return tilt_quat(az_hold, basket_pos()[:2])

            def rim_tip_z() -> float:
                return float(basket_pos()[2]) + c.basket_rim_local - 0.038

            def tip_goal(z):
                wc, _ = wall_center(k_best)
                gq = gq_now()
                zh = quat_apply(gq.unsqueeze(0), V3(0, 0, 1).unsqueeze(0))[0]
                return V3(wc[0], wc[1], z) - FINGER_LEN * zh, gq

            # HOVER above the wall centerline (cage aperture clears walls on descent)
            if not run_phase(
                    lambda: tip_goal(0.22),
                    lambda: float((tip_pos()[:2] - wall_center(k_best)[0][:2]).norm())
                    < 0.006,
                    CAGE_B, 8.0, tag=f"hoverB{cycle}.{attempt}"):
                park()
                continue
            # DESCEND: fingertips straddle the wall's top 38 mm (deep bite)
            if not run_phase(
                    lambda: tip_goal(rim_tip_z()),
                    lambda: abs(float(tip_pos()[2]) - rim_tip_z()) < 0.005
                    and float((tip_pos()[:2] - wall_center(k_best)[0][:2]).norm())
                    < 0.006,
                    CAGE_B, 8.0, tag=f"descendB{cycle}.{attempt}"):
                park()
                continue
            # CLOSE (quasi-static) + grip verdict (width band; no lift — see below)
            close_ramp(lambda: tip_goal(rim_tip_z())[0], gq_now, CAGE_B, GRIP_B)
            w = width()
            if WB_LO < w < WB_HI:
                print(f"[solve] BASKET GRIPPED (w={w*1000:.1f}mm) cycle {cycle}",
                      flush=True)
                picked = True
                break
            print(f"[solve] basket grip verdict failed (w={w*1000:.1f}mm) — retry",
                  flush=True)
            p, _ = ee_pose()
            hold(V3(p[0], p[1], 0.25), gq_now(), OPEN, 0.8)
            park()
        if not picked:
            continue

        # DRAG the basket along the floor to the staging spot: the ground carries the
        # weight the whole way (a lifted rim-pinch carry pendulum-slipped and DROPPED
        # the basket on 2 of 3 spawns — measured), one fixed jaw azimuth throughout
        gq_c = jaw_quat(az_hold)
        h_drag = float(ee_pose()[0][2])  # keep the grasp-height hand plane

        def drag_goal():
            bpv, h = basket_pos(), ee_pose()[0]
            corr = (1.2 * (prestage_xy() - bpv[:2])).clamp(-0.010, 0.010)
            return clamp_goal(V3(h[0] + corr[0], h[1] + corr[1], h_drag)), gq_c

        run_phase(drag_goal,
                  lambda: float((basket_pos()[:2] - prestage_xy()).norm()) < 0.010
                  or basket_lost(),
                  GRIP_B, 24.0, tag=f"drag{cycle}", xy_boost=1.4)
        if basket_lost():
            print(f"[solve] basket LOST mid-drag (cycle {cycle}, basket at "
                  f"{[round(float(v), 3) for v in (basket_pos() - origin)[:2]]}) "
                  f"— regrasp from its resting pose", flush=True)
            park()
            continue
        print(f"[solve] dragged: prestage_off="
              f"{float((basket_pos()[:2] - prestage_xy()).norm())*1000:.0f}mm "
              f"yaw_err={math.degrees(yaw_err90()):+.1f}deg "
              f"open={float(scene.opening()[0])*1000:.1f}mm", flush=True)

        # RELEASE: slow open, retreat up
        p, _ = ee_pose()
        hold(p, gq_c, CAGE_B, 0.6)   # slow open to the cage width first
        hold(p, gq_c, OPEN, 0.5)
        p, _ = ee_pose()
        hold(V3(p[0], p[1], 0.30), gq_c, OPEN, 0.9)
        placed = True
        break

    if not placed:
        readout("place FAILED")
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(2)

    # --- PUSH the basket the last stretch under the shaft (hand stays low + behind) -----------
    # Push PERPENDICULAR to the robot-facing wall (its live face normal): with residual
    # yaw the basket then translates along the push direction instead of slipping
    # sideways (a +x push on a yawed face drifted the basket 35-40 mm — measured).
    tgt = stage_xy()

    def push_errs():
        _az, n, t = face_frame(toward=push_dir())
        bpv = basket_pos()
        d = torch.cat([tgt - bpv[:2], torch.zeros(1, device=device)])
        s = -float(torch.dot(d, n))   # remaining travel along the push direction (-n)
        lat = float(torch.dot(d, t))  # lateral offset still to fix (along t)
        return s, lat

    def push_done() -> bool:
        s, lat = push_errs()
        return s <= 0.005 and abs(lat) <= 0.020 and bool(scene.basket_upright()[0])

    ok_push = False
    for rnd in range(3):
        dewind()
        if push_done():
            ok_push = True
            break
        # freeze the round's push frame and a WORLD-anchored tip rail: the tip creeps
        # along the rail at ~22 mm/s, so the basket can never run away from the goal
        # (a face-tracking goal maintained whatever speed developed — measured 30+ mm
        # of coast past the target)
        az_f, n, t = face_frame(toward=push_dir())
        n2, t2 = n[:2].clone(), t[:2].clone()
        s0, _lat0 = push_errs()
        rail0 = basket_pos()[:2].clone() + (c.basket_inner_half + c.basket_wall_t
                                            + 0.010) * n2
        pq = jaw_quat(min((az_f + math.pi / 2, az_f - math.pi / 2),
                          key=lambda a: abs(wrap_pi(a - home_az))))
        prog = {"p": -0.030}  # tip starts 30 mm behind the face

        def rail_tip_xy():
            s, lat = push_errs()
            if s > 0.005:
                prog["p"] = min(prog["p"] + 0.0015, s0 + 0.006)
            steer = max(-0.020, min(0.020, -2.0 * lat))
            return rail0 - prog["p"] * n2 + steer * t2

        def rail_goal():
            txy = rail_tip_xy()
            zh = quat_apply(pq.unsqueeze(0), V3(0, 0, 1).unsqueeze(0))[0]
            tp = V3(min(float(txy[0]), 0.020), float(txy[1]), PUSH_TIP_Z)
            return tp - FINGER_LEN * zh, pq

        # hover above the rail start (never drag the hand across the basket rim),
        # then descend to push height on a frozen goal (prog must not creep yet)
        b0 = rail0 + 0.030 * n2

        def descend_goal():
            zh = quat_apply(pq.unsqueeze(0), V3(0, 0, 1).unsqueeze(0))[0]
            return V3(float(b0[0]), float(b0[1]), PUSH_TIP_Z) - FINGER_LEN * zh, pq

        hold(V3(float(b0[0]), float(b0[1]), 0.30), pq, 0.0, 1.2)
        run_phase(descend_goal,
                  lambda: abs(float(tip_pos()[2]) - PUSH_TIP_Z) < 0.006,
                  0.0, 6.0, tag=f"push-descend{rnd}")
        ok_push = run_phase(rail_goal, push_done, 0.0, 25.0,
                            tag=f"push{rnd}", xy_boost=1.4)
        # retreat straight back along the face normal, then up
        p, _ = ee_pose()
        hold(V3(float(p[0] + 0.10 * n2[0]), float(p[1] + 0.10 * n2[1]), p[2]),
             pq, 0.0, 0.8)
        p, _ = ee_pose()
        hold(V3(p[0], p[1], 0.30), pq, OPEN, 0.8)
        if ok_push:
            break
        print(f"[solve] push round {rnd}: gate not met "
              f"(off={float((basket_pos()[:2] - tgt).norm())*1000:.0f}mm, "
              f"open={float(scene.opening()[0])*1000:.1f}mm) — re-approach", flush=True)
    hold(*ee_pose(), OPEN, 1.0)  # let the basket settle
    s_staged = readout("staged")
    if not ok_push and float((basket_pos()[:2] - tgt).norm()) > 0.035:
        # the drop is IRREVERSIBLE: never dispense onto an unstaged basket
        print("[solve] basket NOT staged — refusing to pull the tray", flush=True)
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(2)

    # --- PULL the tray by the yellow fin --------------------------------------------------------
    # Jaw along X (across the fin's 48 mm WIDTH): the hand's wide axis stays parallel
    # to the dispenser face — a y-axis jaw physically cannot descend beside the
    # wall/lid band. The -y pull rides on pad friction (~20 N capacity vs ~3 N needed).
    fq = jaw_quat(min((0.0, math.pi), key=lambda a: abs(wrap_pi(a - home_az))))
    pulled = False
    for attempt in range(3):
        dewind()

        def fin_tip_goal(z):
            f = fin_xy()
            zh = quat_apply(fq.unsqueeze(0), V3(0, 0, 1).unsqueeze(0))[0]
            return V3(f[0], f[1], z) - FINGER_LEN * zh, fq

        # stage high on the near side first (never cut through the lid box), then
        # hover: align precisely over the LIVE fin, jaw fully open (nothing near the
        # fingers along x)
        p, _ = ee_pose()
        hold(V3(p[0], p[1], 0.60), fq, OPEN, 0.8)
        f0 = fin_xy()
        hold(V3(float(f0[0]), float(f0[1]), 0.60), fq, OPEN, 1.0)
        if not run_phase(lambda: fin_tip_goal(0.47),
                         lambda: float((tip_pos()[:2] - fin_xy()).norm()) < 0.004,
                         OPEN, 8.0, tag=f"hoverF{attempt}"):
            park()
            continue
        # descend on a FROZEN snapshot: re-chasing a live fin while a fingertip rests
        # on it ratchets the tray along the slide (measured) — if aligned, the descent
        # touches nothing and the snapshot stays true
        snap = fin_xy().clone()

        def fin_snap_goal(z):
            zh = quat_apply(fq.unsqueeze(0), V3(0, 0, 1).unsqueeze(0))[0]
            return V3(snap[0], snap[1], z) - FINGER_LEN * zh, fq

        if not run_phase(lambda: fin_snap_goal(FIN_TIP_Z),
                         lambda: abs(float(tip_pos()[2]) - FIN_TIP_Z) < 0.006
                         and float((tip_pos()[:2] - snap).norm()) < 0.006,
                         OPEN, 6.0, tag=f"descendF{attempt}"):
            p, _ = ee_pose()
            hold(V3(p[0], p[1], 0.50), fq, OPEN, 0.8)  # straight up, then retry
            park()
            continue
        close_ramp(lambda: fin_snap_goal(FIN_TIP_Z)[0], lambda: fq, OPEN, GRIP_F,
                   secs=1.2)
        w = width()
        if not (WF_LO < w < WF_HI):
            print(f"[solve] fin grasp verdict failed (w={w*1000:.1f}mm) — retry",
                  flush=True)
            p, _ = ee_pose()
            hold(V3(p[0], p[1], 0.50), fq, OPEN, 0.8)
            park()
            continue
        print(f"[solve] FIN HELD (w={w*1000:.1f}mm) attempt {attempt}", flush=True)

        def pull_goal():
            h = ee_pose()[0]
            return V3(fin_xy()[0], float(h[1]) - 0.030, FIN_TIP_Z + FINGER_LEN), fq

        pulled = run_phase(pull_goal,
                           lambda: float(scene.opening()[0]) >= PULL_OPEN,
                           GRIP_F, 15.0, tag=f"pull{attempt}")
        if pulled:
            break
        print(f"[solve] pull attempt {attempt} stalled at "
              f"{float(scene.opening()[0])*1000:.0f}mm (w={width()*1000:.1f}mm) "
              f"— regrasp", flush=True)
        p, _ = ee_pose()
        hold(V3(p[0], p[1], 0.47), fq, OPEN, 0.8)

    p, _ = ee_pose()
    hold(p, ee_pose()[1], OPEN, 0.5)   # let go of the fin gently
    p, _ = ee_pose()
    hold(V3(p[0], p[1], 0.60), ee_pose()[1], OPEN, 0.9)
    park()
    s_pull = readout("pulled")
    if not pulled:
        print("[solve] tray never reached the open threshold", flush=True)
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(2)

    # --- VERIFY: butter lands, settles, sustained success ---------------------------------------
    deadline = sim_t["t"] + 8.0
    while sim_t["t"] < deadline and not bool(scene._delivered[0]):
        hold(*ee_pose(), OPEN, 0.5)
    s_del = readout("delivered")

    deadline = sim_t["t"] + 10.0
    while sim_t["t"] < deadline and not bool(scene.success()[0]):
        hold(*ee_pose(), OPEN, 0.5)
    s_ver = readout("verified")

    # persistence: keep simulating; success must not flicker off
    hold(*ee_pose(), OPEN, 3.0)
    ok = bool(scene.success()[0])
    s_last = readout("persisted")
    bu = (scene.butter.data.root_pos_w[0] - origin)
    bp = (basket_pos() - origin)
    print(f"[solve] final butter=({bu[0]:+.3f},{bu[1]:+.3f},{bu[2]:.3f}) "
          f"basket=({bp[0]:+.3f},{bp[1]:+.3f},{bp[2]:.3f}) "
          f"open={float(scene.opening()[0])*1000:.0f}mm", flush=True)
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

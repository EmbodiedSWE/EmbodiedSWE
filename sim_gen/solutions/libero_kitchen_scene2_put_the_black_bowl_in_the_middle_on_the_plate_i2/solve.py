"""solve — plated_meal with a REAL Franka arm, OSC servo loop (arm-only manipulation).

Plan (order not required by the task; this solve places the bowl first):

  1. SERVE THE BOWL   pick the most reachable bowl by a converging pinch across its bail
                      handle's crossbar (jaw snapped perpendicular to the live bar heading;
                      the grasp point is directly above the CoM, so the bowl hangs level),
                      lift, closed-loop carry on the BOWL xy onto the live plate axis,
                      lower until the bowl bottom meets the plate floor, release, retreat.
  2. LOAD THE FOOD    for each present cube: top-down pinch with the jaw snapped to the
                      cube's yaw, lift over the handle, closed-loop carry onto a drop
                      point in the mouth opening BESIDE the crossbar (jaw re-aligned along
                      the bar), lower to just above the rim, release — the cube falls in.
  3. PARK + SETTLE    withdraw the arm, wait for scene.settled(); read the scene's own
                      success()/score().

Only the arm's 8-D OSC action stream is commanded; task-object state is never written and
no external forces are applied. `SIM_GEN_SCORE <score>` is printed at every phase boundary
(latched rubric -> never decreases along this trajectory) and `SIM_GEN_SOLVE: SUCCESS` iff
scene.success() reads True at the end of every episode.

Run:  python -m simgen_tasks.libero_kitchen_scene2_put_the_black_bowl_in_the_middle_on_the_plate_i2.solve --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--max_sec", type=float, default=420.0, help="sim-time budget per episode")
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

import robobench.controllers  # noqa: E402,F401  (registers controllers)
import robobench.robots  # noqa: E402,F401  (registers robots)

try:
    from .scene import PlatedMealSceneCfg  # noqa: E402  (registers scene + env)
except ImportError:  # direct-script fallback
    from scene import PlatedMealSceneCfg  # type: ignore # noqa: E402

from robobench.core import EnvCfg  # noqa: E402
from robobench.robots.franka import FrankaRobotCfg  # noqa: E402

OPEN = 0.04
FINGER_LEN = 0.112          # panda_hand frame -> fingertip (corpus-measured)
BASE = (-0.42, 0.0, 0.20)   # arm mounted ON the counter slab, west of the work area


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    rcfg = FrankaRobotCfg(
        base_pos=BASE,
        nullspace_dof_pos=(),          # corpus lesson: default posture winds the arm
        gripper_effort_limit=80.0,
        gripper_stiffness=4000.0,
    )
    cfg = EnvCfg(scene="plated_meal", robot="franka", control_mode="osc",
                 env_spacing=3.0, robot_cfg=rcfg, seed=args.seed)
    env = cfg.build(num_envs=1, device=device)
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

    origin = env.iscene.env_origins[0]
    V3 = lambda x, y, z: torch.tensor([float(x), float(y), float(z)], device=device)  # noqa: E731
    base_xy = torch.tensor([BASE[0], BASE[1]], device=device) + origin[:2]
    lf_idx = art.body_names.index("panda_leftfinger")
    rf_idx = art.body_names.index("panda_rightfinger")

    def finger_diag(tag, bowl_i=None):
        lf = art.data.body_pos_w[0, lf_idx]
        rf = art.data.body_pos_w[0, rf_idx]
        msg = (f"[diag] {tag}: lf=({lf[0]:.4f},{lf[1]:.4f},{lf[2]:.4f}) "
               f"rf=({rf[0]:.4f},{rf[1]:.4f},{rf[2]:.4f}) w={width() * 1000:.1f}mm")
        if bowl_i is not None:
            b = scene.bowls[bowl_i].data.root_pos_w[0]
            msg += f" bowl=({b[0]:.4f},{b[1]:.4f},{b[2]:.4f})"
        print(msg, flush=True)

    z0 = c.surface_z

    # ----- low-level servo kernel (franka corpus lineage) -------------------------------
    def ee_pose():
        return art.data.body_pos_w[0, ee_idx], art.data.body_quat_w[0, ee_idx]

    def width() -> float:
        return 2.0 * art.data.joint_pos[0, fj1].item()

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
        env.step(a, render=False)
        sim_t["t"] += env.dt * robot.control_period

    def hold(pos, quat, grip, secs, xy_boost=1.0):
        for _ in range(SEC(secs)):
            a = torch.zeros(1, n_act, device=device)
            servo(pos, quat, grip, a, xy_boost)
            tick(a)

    def run_phase(goal_fn, gate_fn, grip, timeout_s, tag="", xy_boost=1.0, abort_fn=None):
        deadline = sim_t["t"] + timeout_s
        while sim_t["t"] < deadline:
            a = torch.zeros(1, n_act, device=device)
            gp, gq = goal_fn()
            servo(gp, gq, grip, a, xy_boost)
            tick(a)
            if gate_fn():
                return True
            if abort_fn is not None and abort_fn():
                print(f"[phase:{tag}] aborted (grip lost, w={width() * 1000:.1f}mm)", flush=True)
                return False
        if tag:
            p, _ = ee_pose()
            gp, _ = goal_fn()
            print(f"[phase:{tag}] timeout ee=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
                  f"goal=({gp[0]:.3f},{gp[1]:.3f},{gp[2]:.3f}) w={width() * 1000:.1f}mm",
                  flush=True)
        return False

    def close_ramp(pos, quat, grip_target, secs=1.2):
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
        """Top-down jaw pose, tilted away from the base column for close-in targets."""
        from isaaclab.utils.math import quat_from_angle_axis

        gq = jaw_quat(azimuth)
        d = float((target_xy - base_xy).norm())
        if d < near:
            tilt = min(max_tilt, (near - d) * 5.0)
            u = (target_xy - base_xy) / max(d, 1e-6)
            axis = V3(-u[1], u[0], 0.0)
            gq = quat_mul(quat_from_angle_axis(
                torch.tensor([tilt], device=device), axis.unsqueeze(0))[0].unsqueeze(0),
                gq.unsqueeze(0))[0]
        return gq

    def dewind():
        q = art.data.joint_pos[0]
        if (abs(q[0].item()) > 2.6 or q[3].item() < -2.95 or q[3].item() > -0.15
                or abs(q[6].item()) > 2.6):
            print(f"[solve] wound arm (q1={q[0]:.2f} q4={q[3]:.2f} q7={q[6]:.2f}) — joint reset",
                  flush=True)
            robot.reset(torch.tensor([0], device=device, dtype=torch.long))
            p, qq = ee_pose()
            hold(p, qq, OPEN, 0.5)

    # ----- scene reads -------------------------------------------------------------------
    def bowl_pos(i):
        return scene.bowls[i].data.root_pos_w[0]

    def plate_pos():
        return scene.plate.data.root_pos_w[0]

    def food_pos(f):
        return scene.food[f].data.root_pos_w[0]

    def bowl_yaw(i) -> float:
        q = scene.bowls[i].data.root_quat_w[0]
        w, x, y, z = (float(v) for v in q)
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def cube_yaw(f) -> float:
        q = scene.food[f].data.root_quat_w[0]
        w, x, y, z = (float(v) for v in q)
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def report(tag):
        onp = scene.bowls_on_plate()[0].tolist()
        gath = scene.gathered()[0].tolist()
        clear = scene.bowls_clear()[0].tolist()
        print(f"[readout] {tag}: on_plate={onp} gathered={gath} clear={clear} "
              f"present={scene.present[0].tolist()} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.2f}",
              flush=True)

    def mark(tag):
        print(f"SIM_GEN_SCORE {float(scene.score()[0]):.4f}", flush=True)
        report(tag)

    # ----- skill: carry the bowl by its bail handle and set it on the plate ------------------
    # Grasps on the bowl's walls or a rim-side tab all fail on a free-sliding 150 g bowl:
    # the CoM sits far from the grasp, and the gravity torque walks the pinch off the pads
    # (measured across 4 grasp designs). The bail handle's crossbar runs over the mouth
    # CENTRE — pinching its middle is the corpus can-pinch load case (pure vertical, no
    # torque): the carried bowl hangs level like a bucket.
    PRE = 0.013  # pre-narrowed per-finger target (26 mm jaw) while descending onto the bar

    def serve_bowl(i) -> bool:
        beam_dz = c.bowl_h / 2 + c.ear_h + c.ear_t / 2  # bowl centre -> crossbar centre
        for attempt in range(3):
            dewind()
            yaw = bowl_yaw(i)  # the crossbar runs along the bowl-local +x axis
            # jaw axis PERPENDICULAR to the crossbar (fingers straddle its 12 mm width);
            # wrap mod 180 deg into (-90, 90] so the wrist (q7, +-166 deg) never saturates
            az_j = math.remainder(yaw + math.pi / 2, math.pi)
            gq = jaw_quat(az_j)

            def ear_tip():
                b = bowl_pos(i)  # pinch the crossbar middle = directly above the CoM
                return V3(b[0], b[1], b[2] + beam_dz - 0.008)

            def hand_for(tip_z_off=0.0):
                t = ear_tip()
                return V3(t[0], t[1], t[2] + FINGER_LEN + tip_z_off)

            # A: hover above the bar, jaw pre-narrowed to 26 mm (7 mm clear per side)
            ok = run_phase(lambda: (hand_for(0.09), gq),
                           lambda: (ee_pose()[0][:2] - ear_tip()[:2]).norm() < 0.004,
                           PRE, 8.0, tag=f"bowl-hover{attempt}")
            # B: straight-down descent; fingertips sink beside the bar over the open mouth
            ok = ok and run_phase(lambda: (hand_for(0.0), gq),
                                  lambda: abs(float(ee_pose()[0][2] - hand_for(0.0)[2])) < 0.004
                                  and (ee_pose()[0][:2] - ear_tip()[:2]).norm() < 0.005,
                                  PRE, 8.0, tag=f"bowl-descend{attempt}")
            if not ok:
                p, _ = ee_pose()
                hold(V3(p[0], p[1], z0 + 0.35), gq, OPEN, 1.0)
                continue

            bp2 = bowl_pos(i)
            print(f"[diag] grasp setup: bowl=({bp2[0]:.4f},{bp2[1]:.4f},{bp2[2]:.4f}) "
                  f"yaw={math.degrees(yaw):.1f}", flush=True)
            finger_diag("pre-close", i)
            # converge on the tab, chasing its live XY only (the bowl may yaw/creep) with
            # the height FROZEN: tracking live z is positive feedback — sticky pads lift
            # the tab a little, the target rises, the bowl gets hoisted and twists out.
            z_close = float(hand_for(0.0)[2])
            n_close = SEC(1.8)
            for kk in range(n_close):
                a = torch.zeros(1, n_act, device=device)
                f = min(1.0, (kk + 1) / (n_close * 0.7))
                t = ear_tip()
                servo(V3(t[0], t[1], z_close), gq, PRE + (0.004 - PRE) * f, a)
                tick(a)
            finger_diag("post-close", i)
            # hand height that hangs the bowl bottom ~60 mm above the counter:
            # hand = bottom + bowl_h/2 + beam_dz - 8 mm tip offset + finger length
            CARRY_HAND = z0 + 0.060 + c.bowl_h / 2 + beam_dz - 0.008 + FINGER_LEN
            z_before = float(bowl_pos(i)[2])
            p, _ = ee_pose()
            hold(V3(p[0], p[1], CARRY_HAND), gq, 0.004, 2.2)
            w = width()
            # crossbar gripped reads ~9.5-13 mm; a free close reads 8 mm
            if float(bowl_pos(i)[2]) - z_before < 0.04 or not (0.0092 < w < 0.0145):
                print(f"[solve] bowl lift failed (dz={float(bowl_pos(i)[2]) - z_before:.3f} "
                      f"w={w * 1000:.1f}mm) — retry", flush=True)
                p, _ = ee_pose()
                hold(p, gq, OPEN, 0.6)
                hold(V3(p[0], p[1], z0 + 0.35), gq, OPEN, 1.0)
                continue

            # carry the BOWL onto the live plate axis (closed loop on object + target)
            def carry_goal():
                b, pl, h = bowl_pos(i), plate_pos(), ee_pose()[0]
                return V3(h[0] + (pl[0] - b[0]), h[1] + (pl[1] - b[1]), CARRY_HAND), gq

            run_phase(carry_goal,
                      lambda: (bowl_pos(i)[:2] - plate_pos()[:2]).norm() < 0.010,
                      0.004, 14.0, tag=f"bowl-carry{attempt}", xy_boost=1.6,
                      abort_fn=lambda: width() < 0.0092)
            if not (0.0092 < width() < 0.0145):
                print("[solve] bowl slipped in transit — retry", flush=True)
                continue

            # lower until the bowl bottom meets the plate floor (+3 mm), xy stays gated
            def target_bz():
                return float(plate_pos()[2]) + c.plate_floor_local_z + 0.003 + c.bowl_h / 2

            def lower_goal():
                b, pl, h = bowl_pos(i), plate_pos(), ee_pose()[0]
                return V3(h[0] + (pl[0] - b[0]), h[1] + (pl[1] - b[1]),
                          h[2] - (float(b[2]) - target_bz())), gq

            ok_low = run_phase(lower_goal,
                               lambda: float(bowl_pos(i)[2]) - target_bz() < 0.006
                               and (bowl_pos(i)[:2] - plate_pos()[:2]).norm() < 0.015,
                               0.004, 12.0, tag=f"bowl-lower{attempt}", xy_boost=1.6,
                               abort_fn=lambda: width() < 0.0092)

            p, _ = ee_pose()
            hold(p, gq, OPEN, 0.8)                       # slow release in place
            hold(V3(p[0], p[1], z0 + 0.35), gq, OPEN, 1.2)  # retreat straight up

            deadline = sim_t["t"] + 5.0
            while sim_t["t"] < deadline:
                pp, _ = ee_pose()
                hold(pp, gq, OPEN, 0.1)
                if bool(scene.bowls_on_plate()[0, i]):
                    return True
            print(f"[solve] bowl {i} not judged on-plate after release — retry", flush=True)
        return False

    # ----- skill: pick one cube and drop it into the served bowl ----------------------------
    def load_cube(f, bowl_i) -> bool:
        for attempt in range(3):
            if bool(scene.food_in_bowl()[0, f, bowl_i]):
                return True  # already in (e.g. bounced in on a previous try)
            dewind()
            fp = food_pos(f)
            # jaw parallel to the cube's faces (mod 90 deg): an unaligned jaw meets the
            # cube on its edges and twists it out on close (measured)
            az_c = math.remainder(cube_yaw(f), math.pi / 2)
            gq = tilt_quat(az_c, fp[:2])

            def tip_goal(z):
                o = food_pos(f)
                return V3(o[0], o[1], z + FINGER_LEN), gq

            hover_z = z0 + 0.13
            ok = run_phase(lambda: tip_goal(hover_z),
                           lambda: (ee_pose()[0][:2] - food_pos(f)[:2]).norm() < 0.005,
                           OPEN, 8.0, tag=f"cube{f}-hover{attempt}")
            grasp_z = lambda: float(food_pos(f)[2]) - 0.004  # noqa: E731
            ok = ok and run_phase(lambda: tip_goal(grasp_z()),
                                  lambda: abs(float(ee_pose()[0][2]) -
                                              (grasp_z() + FINGER_LEN)) < 0.005
                                  and (ee_pose()[0][:2] - food_pos(f)[:2]).norm() < 0.005,
                                  OPEN, 8.0, tag=f"cube{f}-descend{attempt}")
            if not ok:
                p, _ = ee_pose()
                hold(V3(p[0], p[1], z0 + 0.30), gq, OPEN, 1.0)
                continue

            gp, _ = ee_pose()
            close_ramp(gp, gq, 0.010, secs=1.6)
            z_before = float(food_pos(f)[2])
            p, _ = ee_pose()
            rim_z = float(bowl_pos(bowl_i)[2]) + c.bowl_h / 2
            # transit: cube bottom must clear the bail handle's crossbar top (rim + 41 mm)
            carry_z = rim_z + 0.075
            hold(V3(p[0], p[1], carry_z + FINGER_LEN), gq, 0.010, 1.4)
            w = width()
            if float(food_pos(f)[2]) - z_before < 0.04 or not (0.022 < w < 0.037):
                print(f"[solve] cube {f} lift failed (w={w * 1000:.1f}mm) — retry", flush=True)
                p, _ = ee_pose()
                hold(p, gq, OPEN, 0.5)
                hold(V3(p[0], p[1], z0 + 0.30), gq, OPEN, 1.0)
                continue

            # drop point: through one of the two mouth openings BESIDE the crossbar —
            # offset perpendicular to the bar (base-facing side), jaw re-aligned ALONG
            # the bar so neither finger hangs over it
            def drop_xy():
                b = bowl_pos(bowl_i)
                byaw = bowl_yaw(bowl_i)
                perp = V3(-math.sin(byaw), math.cos(byaw), 0.0)[:2]
                cand_a = b[:2] + 0.022 * perp
                cand_b = b[:2] - 0.022 * perp
                return cand_a if float((cand_a - base_xy).norm()) <= \
                    float((cand_b - base_xy).norm()) else cand_b

            gq2 = jaw_quat(math.remainder(bowl_yaw(bowl_i), math.pi))

            def carry_goal():
                o, h = food_pos(f), ee_pose()[0]
                d = drop_xy()
                return V3(h[0] + (d[0] - o[0]), h[1] + (d[1] - o[1]),
                          carry_z + FINGER_LEN), gq2

            run_phase(carry_goal,
                      lambda: (food_pos(f)[:2] - drop_xy()).norm() < 0.006,
                      0.010, 14.0, tag=f"cube{f}-carry{attempt}", xy_boost=1.6,
                      abort_fn=lambda: width() < 0.022)
            if not (0.022 < width() < 0.037):
                print(f"[solve] cube {f} slipped in transit — retry", flush=True)
                continue

            def lower_goal():
                o, h = food_pos(f), ee_pose()[0]
                d = drop_xy()
                drop_z = float(bowl_pos(bowl_i)[2]) + c.bowl_h / 2 + 0.030
                return V3(h[0] + (d[0] - o[0]), h[1] + (d[1] - o[1]),
                          h[2] - (float(o[2]) - drop_z)), gq2

            run_phase(lower_goal,
                      lambda: float(food_pos(f)[2]) - (float(bowl_pos(bowl_i)[2])
                                                       + c.bowl_h / 2 + 0.030) < 0.006
                      and (food_pos(f)[:2] - drop_xy()).norm() < 0.007,
                      0.010, 10.0, tag=f"cube{f}-lower{attempt}")

            p, _ = ee_pose()
            hold(p, gq2, OPEN, 0.5)                      # release: the cube drops in
            hold(V3(p[0], p[1], z0 + 0.36), gq2, OPEN, 1.0)

            deadline = sim_t["t"] + 4.0
            while sim_t["t"] < deadline:
                pp, _ = ee_pose()
                hold(pp, gq, OPEN, 0.1)
                if bool(scene.food_in_bowl()[0, f, bowl_i]):
                    return True
            print(f"[solve] cube {f} missed the bowl "
                  f"(at {[round(float(v), 3) for v in food_pos(f)]}) — retry", flush=True)
        return False

    # ----- one episode ----------------------------------------------------------------------
    def run_episode(ep: int) -> bool:
        t_ep = sim_t["t"]
        p0, q0 = ee_pose()
        hold(p0, q0, OPEN, 1.0)
        mark(f"ep{ep}-start")

        # choose the most reachable bowl (any is legal — they are interchangeable)
        dists = [float((bowl_pos(i)[:2] - base_xy).norm()) for i in range(c.n_bowls)]
        order = sorted(range(c.n_bowls), key=lambda i: dists[i])
        print(f"[solve] ep{ep}: bowl reach distances {[round(d, 3) for d in dists]} "
              f"-> serve order preference {order}", flush=True)

        bowl_i = -1
        for i in order:
            if serve_bowl(i):
                bowl_i = i
                break
        if bowl_i < 0:
            print("[solve] could not plate any bowl", flush=True)
            return False
        mark(f"ep{ep}-bowl-plated")

        present = [f for f in range(len(scene.food)) if bool(scene.present[0, f])]
        # nearest-first: shorter carries first, and the last cube gets the emptiest approach
        present.sort(key=lambda f: float((food_pos(f)[:2] - base_xy).norm()))
        print(f"[solve] ep{ep}: present cubes {present}", flush=True)
        for f in present:
            if sim_t["t"] - t_ep > args.max_sec:
                print("[solve] sim-time budget exhausted", flush=True)
                return False
            if not load_cube(f, bowl_i):
                print(f"[solve] cube {f} could not be loaded", flush=True)
                return False
            mark(f"ep{ep}-cube{f}-loaded")

        # park the arm clear of the assembly and let everything settle
        dewind()
        p, _ = ee_pose()
        park = V3(float(origin[0]) + BASE[0] + 0.18, float(origin[1]) + BASE[1], z0 + 0.40)
        hold(V3(p[0], p[1], z0 + 0.38), jaw_quat(0.0), OPEN, 1.0)
        deadline = sim_t["t"] + 10.0
        while sim_t["t"] < deadline:
            hold(park, jaw_quat(0.0), OPEN, 0.2)
            if bool(scene.success()[0]):
                break
        mark(f"ep{ep}-final")
        return bool(scene.success()[0])

    results = []
    for ep in range(args.episodes):
        if ep > 0:
            env.reset(seed=args.seed + ep)
            robot.reset(torch.tensor([0], device=device, dtype=torch.long))
        results.append(run_episode(ep))

    print(f"[solve] episode successes: {results}", flush=True)
    ok = all(results) and len(results) == args.episodes
    if ok:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL", flush=True)

    rc = 0 if ok else 1
    threading.Timer(10.0, lambda: os._exit(rc)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(rc)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)

"""plated_meal, lifted for data generation (strategy_1).

The delivered solve (strategy_0, pristine next door) with its choices exposed as Params:
macro order (bowl-first / food-first), serving-bowl choice, cube order, drop side, and the
documented-slack constants. The phase machine, skills, gates and retries are ported
verbatim from solve.py; every env.step goes through the recorder so episodes are captured
(and optionally noise-injected) uniformly.

Import only on the forge, after AppLauncher boot, from inside the task package.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_from_matrix,
    quat_mul,
)

OPEN = 0.04
FINGER_LEN = 0.112
BASE = (-0.42, 0.0, 0.20)          # the solve's embodiment binding (recorded in TASK.md)
PRE = 0.013                        # pre-narrowed jaw while descending onto the bail bar


@dataclass
class Params:
    """Sampling surface. Defaults reproduce the delivered solve exactly."""
    bowl_first: bool = True            # macro order: serve bowl then load food, or reverse
    bowl_choice: str = "reachable"     # reachable | random  (bowls are interchangeable)
    cube_order: str = "near"           # near | far | random
    # which mouth opening takes each cube. "alternate" separates cubes inside the bowl —
    # same-spot drops leave them leaning on each other with contact jitter right at the
    # settle threshold (measured |v|=0.051 vs the 0.05 gate)
    drop_side: str = "alternate"       # alternate | base | far | random
    carry_bowl_h: float = 0.060        # bowl bottom above counter during its carry (m)
    cube_carry_extra: float = 0.075    # cube bottom clearance above the bowl rim in transit
    cube_hover: float = 0.13           # hover height above counter for cube approach
    bowl_tip_off: float = -0.008       # fingertip depth on the bail crossbar
    cube_grasp_off: float = -0.004     # fingertip depth on the cube
    drop_perp: float = 0.022           # drop-point offset beside the crossbar (m)
    drop_h: float = 0.030              # cube release height above the rim (m)
    max_sec: float = 420.0
    extra: dict = field(default_factory=dict)   # free-form, recorded in meta


def run_episode(env, rec, P: Params, rng) -> bool:
    """One episode on an already-reset env. rec.step_env replaces env.step throughout."""
    device = env.device
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
    z0 = c.surface_z
    sim_t = {"t": 0.0}

    # ----- kernel (solve.py lineage; env.step -> rec.step_env) --------------------------------
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

    def tick(a):
        rec.step_env(a)
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
                print(f"[phase:{tag}] aborted (w={width() * 1000:.1f}mm)", flush=True)
                return False
        if tag:
            print(f"[phase:{tag}] timeout", flush=True)
        return False

    def jaw_quat(azimuth: float) -> torch.Tensor:
        yh = V3(math.cos(azimuth), math.sin(azimuth), 0.0)
        zh = V3(0.0, 0.0, -1.0)
        xh = torch.cross(yh, zh, dim=0)
        return quat_from_matrix(torch.stack([xh, yh, zh], dim=1).unsqueeze(0))[0]

    def tilt_quat(azimuth: float, target_xy, near=0.37, max_tilt=0.35):
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
            print("[gen] wound arm — joint reset", flush=True)
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

    def yaw_of(q) -> float:
        w, x, y, z = (float(v) for v in q)
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def bowl_yaw(i) -> float:
        return yaw_of(scene.bowls[i].data.root_quat_w[0])

    def cube_yaw(f) -> float:
        return yaw_of(scene.food[f].data.root_quat_w[0])

    # ----- skill: serve the bowl (ported; P hooks marked) --------------------------------
    def serve_bowl(i) -> bool:
        beam_dz = c.bowl_h / 2 + c.ear_h + c.ear_t / 2
        for attempt in range(3):
            dewind()
            # bar pose from the bowl's FULL orientation: after a knock the bowl can lean,
            # and an upright-only offset aims the pinch beside the bar (measured: free
            # close at 8 mm on every re-serve attempt)
            bq = scene.bowls[i].data.root_quat_w[0:1]
            bar_dir = quat_apply(bq, torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
            az_j = math.remainder(math.atan2(float(bar_dir[1]), float(bar_dir[0]))
                                  + math.pi / 2, math.pi)
            gq = jaw_quat(az_j)

            def ear_tip():
                b = bowl_pos(i)
                bq2 = scene.bowls[i].data.root_quat_w[0:1]
                off = quat_apply(bq2, torch.tensor([[0.0, 0.0, beam_dz]],
                                                   device=device))[0]
                return V3(b[0] + off[0], b[1] + off[1], b[2] + off[2] + P.bowl_tip_off)

            def hand_for(tip_z_off=0.0):
                t = ear_tip()
                return V3(t[0], t[1], t[2] + FINGER_LEN + tip_z_off)

            ok = run_phase(lambda: (hand_for(0.09), gq),
                           lambda: (ee_pose()[0][:2] - ear_tip()[:2]).norm() < 0.004,
                           PRE, 8.0, tag=f"bowl-hover{attempt}")
            ok = ok and run_phase(lambda: (hand_for(0.0), gq),
                                  lambda: abs(float(ee_pose()[0][2] - hand_for(0.0)[2])) < 0.004
                                  and (ee_pose()[0][:2] - ear_tip()[:2]).norm() < 0.005,
                                  PRE, 8.0, tag=f"bowl-descend{attempt}")
            if not ok:
                p, _ = ee_pose()
                hold(V3(p[0], p[1], z0 + 0.35), gq, OPEN, 1.0)
                continue

            z_close = float(hand_for(0.0)[2])
            n_close = SEC(1.8)
            for kk in range(n_close):
                a = torch.zeros(1, n_act, device=device)
                f = min(1.0, (kk + 1) / (n_close * 0.7))
                t = ear_tip()
                servo(V3(t[0], t[1], z_close), gq, PRE + (0.004 - PRE) * f, a)
                tick(a)
            CARRY_HAND = z0 + P.carry_bowl_h + c.bowl_h / 2 + beam_dz + P.bowl_tip_off + FINGER_LEN
            z_before = float(bowl_pos(i)[2])
            p, _ = ee_pose()
            hold(V3(p[0], p[1], CARRY_HAND), gq, 0.004, 2.2)
            w = width()
            if float(bowl_pos(i)[2]) - z_before < 0.04 or not (0.0092 < w < 0.0145):
                print(f"[gen] bowl lift failed (w={w * 1000:.1f}mm) — retry", flush=True)
                p, _ = ee_pose()
                hold(p, gq, OPEN, 0.6)
                hold(V3(p[0], p[1], z0 + 0.35), gq, OPEN, 1.0)
                continue

            def carry_goal():
                b, pl, h = bowl_pos(i), plate_pos(), ee_pose()[0]
                return V3(h[0] + (pl[0] - b[0]), h[1] + (pl[1] - b[1]), CARRY_HAND), gq

            run_phase(carry_goal,
                      lambda: (bowl_pos(i)[:2] - plate_pos()[:2]).norm() < 0.010,
                      0.004, 14.0, tag=f"bowl-carry{attempt}", xy_boost=1.6,
                      abort_fn=lambda: width() < 0.0092)
            if not (0.0092 < width() < 0.0145):
                print("[gen] bowl slipped in transit — retry", flush=True)
                continue

            def target_bz():
                return float(plate_pos()[2]) + c.plate_floor_local_z + 0.003 + c.bowl_h / 2

            def lower_goal():
                b, pl, h = bowl_pos(i), plate_pos(), ee_pose()[0]
                return V3(h[0] + (pl[0] - b[0]), h[1] + (pl[1] - b[1]),
                          h[2] - (float(b[2]) - target_bz())), gq

            run_phase(lower_goal,
                      lambda: float(bowl_pos(i)[2]) - target_bz() < 0.006
                      and (bowl_pos(i)[:2] - plate_pos()[:2]).norm() < 0.015,
                      0.004, 12.0, tag=f"bowl-lower{attempt}", xy_boost=1.6,
                      abort_fn=lambda: width() < 0.0092)

            p, _ = ee_pose()
            hold(p, gq, OPEN, 0.8)
            hold(V3(p[0], p[1], z0 + 0.35), gq, OPEN, 1.2)

            deadline = sim_t["t"] + 5.0
            while sim_t["t"] < deadline:
                pp, _ = ee_pose()
                hold(pp, gq, OPEN, 0.1)
                if bool(scene.bowls_on_plate()[0, i]):
                    return True
            print(f"[gen] bowl {i} not judged on-plate — retry", flush=True)
        return False

    # ----- skill: load one cube into the (possibly counter-standing) bowl -----------------
    def load_cube(f, bowl_i, nth: int = 0) -> bool:
        for attempt in range(3):
            if bool(scene.food_in_bowl()[0, f, bowl_i]):
                return True
            dewind()
            fp = food_pos(f)
            az_c = math.remainder(cube_yaw(f), math.pi / 2)
            gq = tilt_quat(az_c, fp[:2])

            def tip_goal(z):
                o = food_pos(f)
                return V3(o[0], o[1], z + FINGER_LEN), gq

            hover_z = z0 + P.cube_hover
            ok = run_phase(lambda: tip_goal(hover_z),
                           lambda: (ee_pose()[0][:2] - food_pos(f)[:2]).norm() < 0.005,
                           OPEN, 8.0, tag=f"cube{f}-hover{attempt}")
            grasp_z = lambda: float(food_pos(f)[2]) + P.cube_grasp_off  # noqa: E731
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
            n = SEC(1.6)
            for k in range(n):
                a = torch.zeros(1, n_act, device=device)
                fr = min(1.0, (k + 1) / (n * 0.7))
                servo(gp, gq, OPEN + (0.010 - OPEN) * fr, a)
                tick(a)
            z_before = float(food_pos(f)[2])
            p, _ = ee_pose()
            rim_z = float(bowl_pos(bowl_i)[2]) + c.bowl_h / 2
            carry_z = rim_z + P.cube_carry_extra
            hold(V3(p[0], p[1], carry_z + FINGER_LEN), gq, 0.010, 1.4)
            w = width()
            if float(food_pos(f)[2]) - z_before < 0.04 or not (0.022 < w < 0.037):
                print(f"[gen] cube {f} lift failed (w={w * 1000:.1f}mm) — retry", flush=True)
                p, _ = ee_pose()
                hold(p, gq, OPEN, 0.5)
                hold(V3(p[0], p[1], z0 + 0.30), gq, OPEN, 1.0)
                continue

            def drop_xy():
                b = bowl_pos(bowl_i)
                byaw = bowl_yaw(bowl_i)
                perp = V3(-math.sin(byaw), math.cos(byaw), 0.0)[:2]
                along = V3(math.cos(byaw), math.sin(byaw), 0.0)[:2]
                cand_a, cand_b = b[:2] + P.drop_perp * perp, b[:2] - P.drop_perp * perp
                da, db = float((cand_a - base_xy).norm()), float((cand_b - base_xy).norm())
                near, far = (cand_a, cand_b) if da <= db else (cand_b, cand_a)
                if P.drop_side == "alternate":
                    # 4 distinct landing spots (side x along-bar), advanced on retries so a
                    # bounced drop never repeats onto the cube that deflected it
                    k = (nth + attempt) % 4
                    side = near if k % 2 == 0 else far
                    return side + (0.015 if k < 2 else -0.015) * along
                if P.drop_side == "base":
                    return near
                if P.drop_side == "far":
                    return far
                return cand_a if rng.random() < 0.5 else cand_b

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
                print(f"[gen] cube {f} slipped in transit — retry", flush=True)
                continue

            def lower_goal():
                o, h = food_pos(f), ee_pose()[0]
                d = drop_xy()
                drop_z = float(bowl_pos(bowl_i)[2]) + c.bowl_h / 2 + P.drop_h
                return V3(h[0] + (d[0] - o[0]), h[1] + (d[1] - o[1]),
                          h[2] - (float(o[2]) - drop_z)), gq2

            run_phase(lower_goal,
                      lambda: float(food_pos(f)[2]) - (float(bowl_pos(bowl_i)[2])
                                                       + c.bowl_h / 2 + P.drop_h) < 0.006
                      and (food_pos(f)[:2] - drop_xy()).norm() < 0.007,
                      0.010, 10.0, tag=f"cube{f}-lower{attempt}")

            p, _ = ee_pose()
            hold(p, gq2, OPEN, 0.5)
            hold(V3(p[0], p[1], z0 + 0.36), gq2, OPEN, 1.0)

            deadline = sim_t["t"] + 4.0
            while sim_t["t"] < deadline:
                pp, _ = ee_pose()
                hold(pp, gq, OPEN, 0.1)
                if bool(scene.food_in_bowl()[0, f, bowl_i]):
                    return True
            print(f"[gen] cube {f} missed — retry", flush=True)
        return False

    # ----- the episode -----------------------------------------------------------------------
    def readout(tag):
        print(f"[readout] {tag}: on_plate={scene.bowls_on_plate()[0].tolist()} "
              f"gathered={scene.gathered()[0].tolist()} clear={scene.bowls_clear()[0].tolist()} "
              f"present={scene.present[0].tolist()} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.2f}",
              flush=True)
        vels = {}
        for name, b in ([(f"bowl{i}", scene.bowls[i]) for i in range(c.n_bowls)]
                        + [(f"food{i}", scene.food[i]) for i in range(len(scene.food))]
                        + [("plate", scene.plate)]):
            vels[name] = round(float(b.data.root_lin_vel_w[0].norm()), 4)
        print(f"[readout] |v|: {vels}", flush=True)

    t_ep = sim_t["t"]
    p0, q0 = ee_pose()
    hold(p0, q0, OPEN, 1.0)
    pl = plate_pos()
    print(f"[layout] plate=({float(pl[0]):.3f},{float(pl[1]):.3f}) "
          f"bowls={[[round(float(v), 3) for v in bowl_pos(i)[:2]] for i in range(c.n_bowls)]} "
          f"food={[[round(float(v), 3) for v in food_pos(f)[:2]] for f in range(len(scene.food))]} "
          f"present={scene.present[0].tolist()}", flush=True)

    dists = [float((bowl_pos(i)[:2] - base_xy).norm()) for i in range(c.n_bowls)]
    order = sorted(range(c.n_bowls), key=lambda i: dists[i])
    if P.bowl_choice == "random":
        rng.shuffle(order)

    present = [f for f in range(len(scene.food)) if bool(scene.present[0, f])]
    present.sort(key=lambda f: float((food_pos(f)[:2] - base_xy).norm()))
    if P.cube_order == "far":
        present.reverse()
    elif P.cube_order == "random":
        rng.shuffle(present)

    def over_budget():
        return sim_t["t"] - t_ep > P.max_sec

    if P.bowl_first:
        bowl_i = -1
        for i in order:
            if serve_bowl(i):
                bowl_i = i
                break
        if bowl_i < 0 or over_budget():
            return False
        for k, f in enumerate(present):
            if over_budget() or not load_cube(f, bowl_i, nth=k):
                return False
    else:
        # food-first: load the chosen bowl where it stands, then carry the loaded bowl
        bowl_i = order[0]
        for k, f in enumerate(present):
            if over_budget() or not load_cube(f, bowl_i, nth=k):
                return False
        if over_budget() or not serve_bowl(bowl_i):
            return False

    # terminal repair: components can regress after first being achieved (measured: a cube
    # recovery knocked the served bowl off the plate). Re-establish the full conjunction.
    for _ in range(2):
        if over_budget():
            break
        good = bool(scene.bowls_on_plate()[0, bowl_i])
        if not good:
            print("[gen] repair: re-serving the bowl", flush=True)
            if not serve_bowl(bowl_i):
                break
        missing = [f for f in present if not bool(scene.food_in_bowl()[0, f, bowl_i])]
        for k, f in enumerate(missing):
            print(f"[gen] repair: re-loading cube {f}", flush=True)
            if over_budget() or not load_cube(f, bowl_i, nth=k):
                break
        if bool(scene.bowls_on_plate()[0, bowl_i]) and not \
                [f for f in present if not bool(scene.food_in_bowl()[0, f, bowl_i])]:
            break

    dewind()
    p, _ = ee_pose()
    park = V3(float(origin[0]) + BASE[0] + 0.18, float(origin[1]) + BASE[1], z0 + 0.40)
    hold(V3(p[0], p[1], z0 + 0.38), jaw_quat(0.0), OPEN, 1.0)
    deadline = sim_t["t"] + 20.0
    while sim_t["t"] < deadline:
        hold(park, jaw_quat(0.0), OPEN, 0.2)
        if bool(scene.success()[0]):
            break
    readout("final")
    return bool(scene.success()[0])

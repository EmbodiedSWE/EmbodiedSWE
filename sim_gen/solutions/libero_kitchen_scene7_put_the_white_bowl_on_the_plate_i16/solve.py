"""solve — cloche_service with a REAL Franka arm, OSC servo loop (arm-only manipulation).

Plan (the execution order is enforced by physics — the plate is blocked until step 1):

  1. UNCOVER   pinch the dome's square grip knob top-down (jaw snapped to the knob's yaw
               mod 90 deg), lift the dome straight up off the plate, closed-loop carry to
               the park spot on the counter, set it down, release, retreat.
  2. SERVE     pinch the WHITE ramekin by spanning it — the jaw straddles the whole 56 mm
               bowl and closes on two opposite octagon flats (jaw snapped to the bowl yaw
               mod 45 deg; the proven can-pinch load case), lift high (clears the parked
               dome's knob), closed-loop carry onto the live plate axis, lower until the
               bowl bottom meets the plate floor, release, retreat.
  3. COVER     re-pinch the knob at the park spot, lift the dome high (its ring clears the
               served bowl), closed-loop carry the DOME axis onto the live plate axis,
               lower straight down — the ring drops inside the plate rim, enclosing the
               bowl — release, retreat.
  4. PARK      withdraw the arm, wait for scene.settled(); read success()/score().

Only the arm's 8-D OSC action stream is commanded; task-object state is never written and
no external forces are applied. `SIM_GEN_SCORE <score>` is printed at every phase boundary
(latched rubric -> never decreases along this trajectory) and `SIM_GEN_SOLVE: SUCCESS` iff
scene.success() reads True at the end of every episode.

Run:  python -m simgen_tasks.libero_kitchen_scene7_put_the_white_bowl_on_the_plate_i16.solve --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--max_sec", type=float, default=300.0, help="sim-time budget per episode")
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
    from .scene import ClocheServiceSceneCfg  # noqa: E402  (registers scene + env)
except ImportError:  # direct-script fallback
    from scene import ClocheServiceSceneCfg  # type: ignore # noqa: E402

from robobench.core import EnvCfg  # noqa: E402
from robobench.robots.franka import FrankaRobotCfg  # noqa: E402

OPEN = 0.04
FINGER_LEN = 0.112          # panda_hand frame -> fingertip (corpus-measured)
BASE = (-0.42, 0.0, 0.20)   # arm mounted ON the counter slab, west of the work area

# knob pinch: 18 mm square post; close cmd 6 mm -> free ~12 mm, gripped ~18 mm
KNOB_CLOSE = 0.006
KNOB_BAND = (0.0145, 0.027)
# bowl spanning pinch: outer flat-to-flat 56 mm; close cmd 24 mm -> free 48 mm, gripped ~56 mm
BOWL_CLOSE = 0.024
BOWL_BAND = (0.050, 0.064)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    rcfg = FrankaRobotCfg(
        base_pos=BASE,
        nullspace_dof_pos=(),          # corpus lesson: default posture winds the arm
        gripper_effort_limit=80.0,
        gripper_stiffness=4000.0,
    )
    cfg = EnvCfg(scene="cloche_service", robot="franka", control_mode="osc",
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
    def dome_pos():
        return scene.dome.data.root_pos_w[0]

    def plate_pos():
        return scene.plate.data.root_pos_w[0]

    def bowl_pos():
        return scene.bowl_white.data.root_pos_w[0]

    def body_yaw(body) -> float:
        q = body.data.root_quat_w[0]
        w, x, y, z = (float(v) for v in q)
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def plate_floor() -> float:
        return float(scene.plate_floor_top()[0])

    def report(tag):
        print(f"[readout] {tag}: uncovered={bool(scene.plate_uncovered()[0])} "
              f"on_plate={bool(scene.bowl_on_plate()[0])} seated={bool(scene.dome_seated()[0])} "
              f"enclosed={bool(scene.bowl_in_dome()[0])} red_clear={bool(scene.red_clear()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.2f}", flush=True)

    def mark(tag):
        print(f"SIM_GEN_SCORE {float(scene.score()[0]):.4f}", flush=True)
        report(tag)

    # knob pinch geometry: pinch the post 25 mm above the dome's top disc
    KNOB_TIP_DZ = c.dome_h / 2 + 0.025  # dome centre -> fingertip target z

    def grasp_knob(tag: str) -> bool:
        """Pinch the dome's knob top-down. Returns True with the dome held."""
        dewind()
        az = math.remainder(body_yaw(scene.dome), math.pi / 2)
        gq = jaw_quat(az)

        def tip():
            d = dome_pos()
            return V3(d[0], d[1], d[2] + KNOB_TIP_DZ)

        def hand(dz=0.0):
            t = tip()
            return V3(t[0], t[1], t[2] + FINGER_LEN + dz)

        ok = run_phase(lambda: (hand(0.09), gq),
                       lambda: (ee_pose()[0][:2] - tip()[:2]).norm() < 0.004,
                       OPEN, 8.0, tag=f"{tag}-hover")
        ok = ok and run_phase(lambda: (hand(0.0), gq),
                              lambda: abs(float(ee_pose()[0][2] - hand(0.0)[2])) < 0.004
                              and (ee_pose()[0][:2] - tip()[:2]).norm() < 0.005,
                              OPEN, 8.0, tag=f"{tag}-descend")
        if not ok:
            p, _ = ee_pose()
            hold(V3(p[0], p[1], z0 + 0.38), gq, OPEN, 1.0)
            return False
        gp, _ = ee_pose()
        close_ramp(gp, gq, KNOB_CLOSE, secs=1.2)
        z_before = float(dome_pos()[2])
        p, _ = ee_pose()
        # lift: dome base to z0 + 0.14 (ring clears plate rim, bowl, everything)
        lift_hand = z0 + 0.14 + c.dome_h / 2 + KNOB_TIP_DZ + FINGER_LEN
        hold(V3(p[0], p[1], lift_hand), gq, KNOB_CLOSE, 2.2)
        w = width()
        if float(dome_pos()[2]) - z_before < 0.05 or not (KNOB_BAND[0] < w < KNOB_BAND[1]):
            print(f"[solve] {tag}: dome lift failed (dz={float(dome_pos()[2]) - z_before:.3f} "
                  f"w={w * 1000:.1f}mm) — release", flush=True)
            p, _ = ee_pose()
            hold(p, gq, OPEN, 0.6)
            hold(V3(p[0], p[1], z0 + 0.40), gq, OPEN, 1.0)
            return False
        return True

    def carry_dome_to(target_xy_fn, tag: str, gate=0.006) -> bool:
        gq = jaw_quat(math.remainder(body_yaw(scene.dome), math.pi / 2))
        carry_hand = z0 + 0.14 + c.dome_h / 2 + KNOB_TIP_DZ + FINGER_LEN

        def goal():
            d, h = dome_pos(), ee_pose()[0]
            t = target_xy_fn()
            return V3(h[0] + (t[0] - d[0]), h[1] + (t[1] - d[1]), carry_hand), gq

        return run_phase(goal,
                         lambda: (dome_pos()[:2] - target_xy_fn()).norm() < gate,
                         KNOB_CLOSE, 14.0, tag=tag, xy_boost=1.6,
                         abort_fn=lambda: width() < KNOB_BAND[0])

    def lower_dome_to(base_z_fn, target_xy_fn, tag: str) -> bool:
        gq = jaw_quat(math.remainder(body_yaw(scene.dome), math.pi / 2))

        def goal():
            d, h = dome_pos(), ee_pose()[0]
            t = target_xy_fn()
            tz = base_z_fn() + c.dome_h / 2
            return V3(h[0] + (t[0] - d[0]), h[1] + (t[1] - d[1]),
                      h[2] - (float(d[2]) - tz)), gq

        ok = run_phase(goal,
                       lambda: float(dome_pos()[2]) - (base_z_fn() + c.dome_h / 2) < 0.006
                       and (dome_pos()[:2] - target_xy_fn()).norm() < 0.012,
                       KNOB_CLOSE, 12.0, tag=tag, xy_boost=1.6,
                       abort_fn=lambda: width() < KNOB_BAND[0])
        p, _ = ee_pose()
        hold(p, gq, OPEN, 0.8)                        # slow release in place
        hold(V3(p[0], p[1], z0 + 0.42), gq, OPEN, 1.2)  # retreat straight up
        return ok

    # ----- skill: uncover the plate --------------------------------------------------------
    def park_xy():
        side = float(scene.side[0])
        return V3(c.dome_park[0], c.dome_park[1] * side, 0)[:2] + origin[:2]

    def uncover() -> bool:
        for attempt in range(3):
            if bool(scene.plate_uncovered()[0]) and \
                    float((dome_pos()[:2] - plate_pos()[:2]).norm()) > c.uncover_dist:
                return True
            if not grasp_knob(f"uncover{attempt}"):
                continue
            if not carry_dome_to(park_xy, f"dome-park-carry{attempt}", gate=0.008):
                if not (KNOB_BAND[0] < width() < KNOB_BAND[1]):
                    continue  # dropped in transit — re-grasp wherever it fell
            lower_dome_to(lambda: z0 + 0.003, park_xy, f"dome-park-lower{attempt}")
            deadline = sim_t["t"] + 4.0
            while sim_t["t"] < deadline:
                pp, _ = ee_pose()
                hold(pp, jaw_quat(0.0), OPEN, 0.1)
                if float((dome_pos()[:2] - plate_pos()[:2]).norm()) > c.uncover_dist:
                    return True
            print(f"[solve] dome not clear of the plate after park (attempt {attempt})",
                  flush=True)
        return False

    # ----- skill: serve the white bowl onto the plate ---------------------------------------
    def serve_bowl() -> bool:
        for attempt in range(3):
            if bool(scene.bowl_on_plate()[0]):
                return True
            dewind()
            # jaw separation axis onto an octagon flat normal (mod 45 deg, |az| <= 22.5 deg)
            az = math.remainder(body_yaw(scene.bowl_white), math.pi / 4)
            gq = jaw_quat(az)

            def tip():
                b = bowl_pos()
                return V3(b[0], b[1], b[2] - 0.005)  # pads on the lower half of the wall

            def hand(dz=0.0):
                t = tip()
                return V3(t[0], t[1], t[2] + FINGER_LEN + dz)

            ok = run_phase(lambda: (hand(0.09), gq),
                           lambda: (ee_pose()[0][:2] - tip()[:2]).norm() < 0.004,
                           OPEN, 8.0, tag=f"bowl-hover{attempt}")
            ok = ok and run_phase(lambda: (hand(0.0), gq),
                                  lambda: abs(float(ee_pose()[0][2] - hand(0.0)[2])) < 0.004
                                  and (ee_pose()[0][:2] - tip()[:2]).norm() < 0.005,
                                  OPEN, 8.0, tag=f"bowl-descend{attempt}")
            if not ok:
                p, _ = ee_pose()
                hold(V3(p[0], p[1], z0 + 0.35), gq, OPEN, 1.0)
                continue

            # converge on the bowl chasing its live XY only, height FROZEN (live-z tracking
            # is positive feedback and hoists the object — corpus lesson)
            z_close = float(hand(0.0)[2])
            n_close = SEC(1.6)
            for kk in range(n_close):
                a = torch.zeros(1, n_act, device=device)
                f = min(1.0, (kk + 1) / (n_close * 0.7))
                t = tip()
                servo(V3(t[0], t[1], z_close), gq, OPEN + (BOWL_CLOSE - OPEN) * f, a)
                tick(a)
            z_before = float(bowl_pos()[2])
            p, _ = ee_pose()
            # carry height: bowl bottom at z0+0.17 — clears the parked dome's knob (z0+0.146)
            carry_hand = z0 + 0.17 + c.bowl_h / 2 - 0.005 + FINGER_LEN
            hold(V3(p[0], p[1], carry_hand), gq, BOWL_CLOSE, 2.2)
            w = width()
            if float(bowl_pos()[2]) - z_before < 0.04 or not (BOWL_BAND[0] < w < BOWL_BAND[1]):
                print(f"[solve] bowl lift failed (dz={float(bowl_pos()[2]) - z_before:.3f} "
                      f"w={w * 1000:.1f}mm) — retry", flush=True)
                p, _ = ee_pose()
                hold(p, gq, OPEN, 0.6)
                hold(V3(p[0], p[1], z0 + 0.35), gq, OPEN, 1.0)
                continue

            def carry_goal():
                b, pl, h = bowl_pos(), plate_pos(), ee_pose()[0]
                return V3(h[0] + (pl[0] - b[0]), h[1] + (pl[1] - b[1]), carry_hand), gq

            run_phase(carry_goal,
                      lambda: (bowl_pos()[:2] - plate_pos()[:2]).norm() < 0.008,
                      BOWL_CLOSE, 14.0, tag=f"bowl-carry{attempt}", xy_boost=1.6,
                      abort_fn=lambda: width() < BOWL_BAND[0])
            if not (BOWL_BAND[0] < width() < BOWL_BAND[1]):
                print("[solve] bowl slipped in transit — retry", flush=True)
                continue

            def target_bz():
                return plate_floor() + 0.003 + c.bowl_h / 2

            def lower_goal():
                b, pl, h = bowl_pos(), plate_pos(), ee_pose()[0]
                return V3(h[0] + (pl[0] - b[0]), h[1] + (pl[1] - b[1]),
                          h[2] - (float(b[2]) - target_bz())), gq

            run_phase(lower_goal,
                      lambda: float(bowl_pos()[2]) - target_bz() < 0.006
                      and (bowl_pos()[:2] - plate_pos()[:2]).norm() < 0.015,
                      BOWL_CLOSE, 12.0, tag=f"bowl-lower{attempt}", xy_boost=1.6,
                      abort_fn=lambda: width() < BOWL_BAND[0])

            p, _ = ee_pose()
            hold(p, gq, OPEN, 0.8)
            hold(V3(p[0], p[1], z0 + 0.38), gq, OPEN, 1.2)

            deadline = sim_t["t"] + 5.0
            while sim_t["t"] < deadline:
                pp, _ = ee_pose()
                hold(pp, gq, OPEN, 0.1)
                if bool(scene.bowl_on_plate()[0]):
                    return True
            print("[solve] bowl not judged on-plate after release — retry", flush=True)
        return False

    # ----- skill: cover the served bowl -------------------------------------------------------
    def cover() -> bool:
        for attempt in range(3):
            if bool(scene.dome_seated()[0]) and bool(scene.bowl_in_dome()[0]):
                return True
            if not grasp_knob(f"cover{attempt}"):
                continue

            def plate_xy():
                return plate_pos()[:2]

            if not carry_dome_to(plate_xy, f"dome-cover-carry{attempt}", gate=0.005):
                if not (KNOB_BAND[0] < width() < KNOB_BAND[1]):
                    continue
            lower_dome_to(lambda: plate_floor() + 0.003, plate_xy,
                          f"dome-cover-lower{attempt}")
            deadline = sim_t["t"] + 5.0
            while sim_t["t"] < deadline:
                pp, _ = ee_pose()
                hold(pp, jaw_quat(0.0), OPEN, 0.1)
                if bool(scene.dome_seated()[0]) and bool(scene.bowl_in_dome()[0]):
                    return True
            print(f"[solve] dome not seated after set-down (attempt {attempt})", flush=True)
        return False

    # ----- one episode ----------------------------------------------------------------------
    def run_episode(ep: int) -> bool:
        t_ep = sim_t["t"]
        p0, q0 = ee_pose()
        hold(p0, q0, OPEN, 1.0)
        d, pl, b = dome_pos(), plate_pos(), bowl_pos()
        print(f"[solve] ep{ep}: layout side={float(scene.side[0]):+.0f} "
              f"swap={int(scene.swap[0])} plate=({pl[0]:.3f},{pl[1]:.3f}) "
              f"dome=({d[0]:.3f},{d[1]:.3f}) white=({b[0]:.3f},{b[1]:.3f})", flush=True)
        mark(f"ep{ep}-start")

        if not uncover():
            print("[solve] could not uncover the plate", flush=True)
            return False
        mark(f"ep{ep}-uncovered")

        for round_ in range(2):
            if not serve_bowl():
                print("[solve] could not serve the bowl", flush=True)
                return False
            mark(f"ep{ep}-served-r{round_}")
            if cover():
                break
            # dome landed badly — if it knocked the bowl off, redo both
            if bool(scene.dome_seated()[0]) or round_ == 1:
                break
            print("[solve] cover round failed — clearing the dome and retrying", flush=True)
            if not uncover():
                break
        mark(f"ep{ep}-covered")

        # park the arm clear of the assembly and let everything settle
        dewind()
        p, _ = ee_pose()
        park = V3(float(origin[0]) + BASE[0] + 0.18, float(origin[1]) + BASE[1], z0 + 0.42)
        hold(V3(p[0], p[1], z0 + 0.40), jaw_quat(0.0), OPEN, 1.0)
        deadline = sim_t["t"] + 10.0
        while sim_t["t"] < deadline:
            hold(park, jaw_quat(0.0), OPEN, 0.2)
            if bool(scene.success()[0]):
                break
        mark(f"ep{ep}-final")
        if sim_t["t"] - t_ep > args.max_sec:
            print("[solve] NOTE: episode exceeded the sim-time budget", flush=True)
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

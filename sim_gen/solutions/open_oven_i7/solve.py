"""solve — REAL Franka-arm solution for `simgen.oven_dials` (the task's feasibility
certificate).

Env: scene "oven_dials" + robot "franka" (OSC), num_envs=1. Base pose chosen here and
recorded in TASK.md: base at (0, 0, 0) on the ground, facing +x; the control deck's near
edge is 0.16 m ahead, the two knobs 0.48 m ahead at heights ~0.22 m — the comfortable
top-down envelope.

Strategy (arm-only; NO task-object state writes, NO external forces — the scene's
`knob_drive`/`knob_force` probe buffers are never touched):
  per knob (right knob first, order is free):
    READ      the pointer angle (knob yaw) and the target setting (the amber lamp's tick;
              read here from scene state — physically the lamp on the deck);
    APPROACH  hover the closed fist above a point on the push circle (radius `PUSH_R`
              around the spindle), ~50 deg behind the grip bar's TAIL end;
    DESCEND   fingertips to bar mid-height (the fist stands beside the bar's side face);
    SWEEP     closed-loop push-turn: every tick, servo the fingertip toward a point on
              the push circle a small lead angle AHEAD of the live tail azimuth, in the
              direction that reduces the pointer error; the fist presses the bar's side
              face and the knob slews detent over detent. Chunked (<= ~110 deg per
              descent) so the fixed fist orientation never fights the arc; gate when the
              pointer is within the detent capture basin;
    RETREAT   straight up; the detent spring seats the pointer exactly on the setting;
    VERIFY    scene.at_target()[k] after settling; re-run the loop if not captured.

The detent basin (+-20 deg, measured by probe: 15 deg snaps in, 25 deg falls out) is the
tolerance buffer that makes the stop reliable: the sweep only has to stop within ~12 deg.

Prints scene readouts, `SIM_GEN_SCORE <score>` at every phase boundary (latched credit —
non-decreasing along this trajectory), and exactly `SIM_GEN_SOLVE: SUCCESS` on success.
Hard exit after the verdict (Kit teardown hangs).

Run (forge): python -u -m simgen_tasks.open_oven_i7.solve --headless [--seed N]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_sec", type=float, default=240.0, help="sim-time budget")
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
    quat_from_matrix,
    quat_mul,
)

import robobench  # noqa: E402
from robobench.core import EnvCfg  # noqa: E402

from . import scene as scene_mod  # noqa: E402  (registers "oven_dials")

assert scene_mod.OvenDialsScene is not None

# ----- Franka session constants (vendored from the corpus substrate) ---------------------------
OPEN = 0.04
CLOSED = 0.0
FINGER_LEN = 0.112          # [ROBOT] panda_hand frame -> fingertip along the approach axis
KP = (320.0, 320.0, 320.0, 600.0, 600.0, 600.0)  # [CORPUS] stiff-contact tier, kd critical

# ----- task-specific pinch-turn constants --------------------------------------------------------
# Runs 2-3 taught that PUSHING the bar with a closed fist is fragile at this lever arm:
# the fist climbs the 32 mm bar (run 2) or drags on the knob face while the chord-directed
# servo force goes radial, stalling against a 1.6 N detent (run 3). The robust primitive is
# a positively-engaged PINCH: straddle the bar's tail segment with open fingers (jaw
# perpendicular to the bar), pinch, and drag the grip point along the spindle-centred arc
# with the wrist tracking the bar's rotation.
GRIP_R = 0.034              # grip-circle radius around the spindle (on the bar, clear of cap)
PINCH = 0.006               # per-finger target: 12 mm on the 16 mm bar = firm pinch (a
                            # 20 mm loose cage was tried and is WORSE: sloppy engagement)
# Arc-goal lead ahead of the live tail azimuth. The pinched tip RIDES the bar, so the
# steady position error is lead*GRIP_R — that error IS the drive force (run 4: an 8 deg
# lead = 4 mm = 1.3 N stalled against the detent's 1.6 N). 14-26 deg = 2.7-5.2 N at kp 320.
LEAD_MIN_DEG = 14.0
LEAD_MAX_DEG = 26.0
STOP_ERR_DEG = 11.0         # stop sweeping here; the +-20 deg detent basin finishes the job
CHUNK_DEG = 100.0           # max arc per pinch (wrist roll centred on home stays comfortable)
TRAVEL_Z = 0.38             # hand transit height (well above bars at 0.235)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    robobench.discover()

    from robobench.robots.franka import FrankaRobotCfg

    rcfg = FrankaRobotCfg(base_pos=(0.0, 0.0, 0.0), nullspace_dof_pos=(),
                          gripper_effort_limit=120.0, gripper_stiffness=4000.0)
    env = EnvCfg(scene="oven_dials", robot="franka", control_mode="osc",
                 env_spacing=3, robot_cfg=rcfg).build(num_envs=1, device=device)
    # Seed AFTER build: the app/env build path reseeds the global RNG, so an earlier
    # manual_seed leaves every --seed with the identical episode (observed runs 5-7).
    torch.manual_seed(args.seed)
    env.reset()
    scene, robot = env.scene, env.robot
    c = scene.cfg
    art = robot.articulation
    ee_idx = art.body_names.index("panda_hand")
    osc = robot.controller.controllers[0]
    osc._kp = torch.tensor(list(KP), device=device)
    osc._kd = 2.0 * osc._kp.sqrt()
    osc.cfg.rot_scale = 0.15
    osc.cfg.kp_null = 3.0
    osc.cfg.kd_null = 2.0 * math.sqrt(3.0)
    n_act = robot.action_dim
    ctrl_hz = 1.0 / (env.dt * robot.control_period)
    origin = env.iscene.env_origins[0]

    V3 = lambda x, y, z: torch.tensor([float(x), float(y), float(z)], device=device)
    SEC = lambda s: max(1, round(s * ctrl_hz))
    sim_t = {"t": 0.0}

    # ----- low-level kernel (corpus substrate, vendored) ---------------------------------------
    def ee_pose():
        return art.data.body_pos_w[0, ee_idx], art.data.body_quat_w[0, ee_idx]

    def tip_pos():
        p, q = ee_pose()
        zh = quat_apply(q.unsqueeze(0), V3(0, 0, 1).unsqueeze(0))[0]
        return p + FINGER_LEN * zh

    def hand_for_tip(tip, quat):
        zh = quat_apply(quat.unsqueeze(0), V3(0, 0, 1).unsqueeze(0))[0]
        return tip - FINGER_LEN * zh

    def jaw_quat(azimuth: float):
        yh = V3(math.cos(azimuth), math.sin(azimuth), 0.0)
        zh = V3(0.0, 0.0, -1.0)
        xh = torch.cross(yh, zh, dim=0)
        return quat_from_matrix(torch.stack([xh, yh, zh], dim=1).unsqueeze(0))[0]

    def servo(goal_pos, goal_quat, grip, a):
        p, q = ee_pose()
        err = goal_pos - p
        a[0, 0:3] = (err / osc.cfg.pos_scale).clamp(-1.0, 1.0)
        qe = quat_mul(goal_quat.unsqueeze(0), quat_conjugate(q.unsqueeze(0)))
        a[0, 3:6] = (axis_angle_from_quat(qe)[0] / osc.cfg.rot_scale).clamp(-1.0, 1.0)
        a[0, 6:8] = grip

    def tick(a):
        env.step(a)
        sim_t["t"] += env.dt * robot.control_period

    def hold(pos, quat, grip, secs: float):
        for _ in range(SEC(secs)):
            a = torch.zeros(1, n_act, device=device)
            servo(pos, quat, grip, a)
            tick(a)

    def run_phase(goal_fn, gate_fn, grip, timeout_s: float, tag: str = "") -> bool:
        deadline = sim_t["t"] + timeout_s
        while sim_t["t"] < deadline:
            a = torch.zeros(1, n_act, device=device)
            gp, gq = goal_fn()
            servo(gp, gq, grip, a)
            tick(a)
            if gate_fn():
                return True
        if tag:
            p, _ = ee_pose()
            gp, _ = goal_fn()
            print(f"[phase:{tag}] timeout: ee=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
                  f"goal=({gp[0]:.3f},{gp[1]:.3f},{gp[2]:.3f}) "
                  f"err={float((gp - p).norm()) * 1000:.0f}mm", flush=True)
        return False

    def dewind(grip: float = OPEN) -> bool:
        q = art.data.joint_pos[0]
        if (abs(q[0].item()) > 2.6 or q[3].item() < -2.95 or q[3].item() > -0.15
                or abs(q[6].item()) > 2.6):
            print(f"[solve] wound arm (q1={q[0]:.2f} q4={q[3]:.2f} q7={q[6]:.2f}) — reset",
                  flush=True)
            robot.reset(torch.tensor([0], device=device, dtype=torch.long))
            p, qq = ee_pose()
            hold(p, qq, grip, 0.5)
            return True
        return False

    # ----- scene reads --------------------------------------------------------------------------
    def reading(k: int) -> float:
        return float(scene.readings_deg()[0, k])

    def target(k: int) -> float:
        return float(scene.target_deg()[0, k])

    def knob_xy(k: int):
        kc = c.knob_center(k)
        return origin[0] + kc[0], origin[1] + kc[1]

    # Fingertips engage the bar's LOWER half (room to ride up before disengaging), with a
    # slight down-bias while sweeping to counter any climb.
    push_z = float(origin[2]) + c.bar_z - 0.006

    def report(tag: str) -> None:
        r = scene.readings_deg()[0].tolist()
        t = scene.target_deg()[0].tolist()
        at = scene.at_target()[0].int().tolist()
        print(f"[solve] {tag:16s} read=({r[0]:7.2f},{r[1]:7.2f}) tgt=({t[0]:6.1f},{t[1]:6.1f}) "
              f"at={at} score={float(scene.score()[0]):.3f} "
              f"success={bool(scene.success()[0])}", flush=True)

    def score_print(tag: str) -> None:
        print(f"SIM_GEN_SCORE {float(scene.score()[0]):.4f}  ({tag})", flush=True)

    # ----- push-turn skill ------------------------------------------------------------------------
    def arc_point(k: int, azim_deg: float, z: float):
        kx, ky = knob_xy(k)
        a = math.radians(azim_deg)
        return V3(kx + GRIP_R * math.cos(a), ky + GRIP_R * math.sin(a), z)

    def wrap180(a: float) -> float:
        return (a + 180.0) % 360.0 - 180.0

    def sweep_chunk(k: int, sgn: float) -> bool:
        """One straddle + pinch + arc drag of at most CHUNK_DEG. Returns True when the
        pointer error is inside STOP_ERR_DEG (sweep done; the detent seats the pointer)."""
        theta0 = reading(k)
        chunk = min(CHUNK_DEG, abs(target(k) - theta0))
        # The grip bar is SYMMETRIC: grip whichever end keeps this chunk's swept arc
        # farthest from azimuth 180 (the base direction). Dragging the grip point back
        # toward the base is the arm's worst-conditioned pull — the commanded tangential
        # force leaks into a parasitic vertical push and the pads climb off the bar
        # (seed-0 stall: the tail swept -60 -> -160 deg, straight at the base).
        def path_clearance(e_off: float) -> float:
            az0 = theta0 + e_off
            return min(abs(wrap180(az0 + sgn * s - 180.0))
                       for s in (0.0, chunk * 0.25, chunk * 0.5, chunk * 0.75, chunk))
        end_off = max((180.0, 0.0), key=path_clearance)
        grip0 = theta0 + end_off
        # Jaw perpendicular to the bar; pick the 180-deg branch that centres this chunk's
        # wrist roll on the home azimuth (q7 stays far from its limits).
        base_az = grip0 + 90.0
        mid_want = math.degrees(home_az)
        off = min((0.0, 180.0, -180.0),
                  key=lambda o: abs(wrap180(base_az + o + sgn * chunk / 2 - mid_want)))
        print(f"[solve] k{k}: chunk {chunk:.0f} deg via {'tail' if end_off else 'pointer'} "
              f"end (clearance {path_clearance(end_off):.0f} deg)", flush=True)

        def jaw_of(grip_deg: float):
            return jaw_quat(math.radians(grip_deg + 90.0 + off))

        # APPROACH straddling the tail segment from above (open jaw flanks the bar); if the
        # realized wrist roll would leave no room for this chunk's rotation, flip the grip
        # 180 deg (same physical jaw line, opposite hand x) and re-approach.
        for _flip in range(2):
            gq0 = jaw_of(grip0)
            goal_hi = hand_for_tip(arc_point(k, grip0, float(origin[2]) + 0.34), gq0)
            if not run_phase(lambda: (goal_hi, gq0),
                             lambda: float((ee_pose()[0] - goal_hi).norm()) < 0.015,
                             OPEN, 8.0, tag=f"k{k} approach"):
                return False
            q7 = float(art.data.joint_pos[0, 6])
            q7_end = q7 + sgn * math.radians(chunk)
            if -2.5 < q7_end < 2.5:
                break
            off = wrap180(off + 180.0)
            print(f"[solve] k{k}: flip grip branch (q7={q7:.2f} would end at {q7_end:.2f})",
                  flush=True)
            dewind()
        # DESCEND: fingertip pads to grip height, live-tracking the chosen end
        def desc_goal():
            g_az = reading(k) + end_off
            return hand_for_tip(arc_point(k, g_az, push_z), jaw_of(g_az)), jaw_of(g_az)
        if not run_phase(desc_goal,
                         lambda: abs(float(tip_pos()[2]) - push_z) < 0.006
                         and float((tip_pos()[:2] - arc_point(k, reading(k) + end_off,
                                                              push_z)[:2]).norm()) < 0.008,
                         OPEN, 8.0, tag=f"k{k} descend"):
            return False
        # PINCH: slow ramp onto the 16 mm bar (12 mm command -> ~8 N per pad)
        n_close = SEC(0.8)
        for i in range(n_close):
            g = OPEN + (PINCH - OPEN) * min(1.0, (i + 1) / (n_close * 0.7))
            gp, gqq = desc_goal()
            a = torch.zeros(1, n_act, device=device)
            servo(gp, gqq, g, a)
            tick(a)

        # SWEEP: drag the pinched tail along the spindle-centred arc, wrist tracking
        deadline = sim_t["t"] + 16.0
        next_trace = sim_t["t"]
        boost = 0.0  # stall booster: no progress for 1.5 s -> +6 deg more lead (cap +18)
        last_th, last_move_t = reading(k), sim_t["t"]
        while sim_t["t"] < deadline:
            th = reading(k)
            err = target(k) - th
            if abs(err) <= STOP_ERR_DEG:
                break
            if abs(th - theta0) >= chunk + 15.0:
                break  # chunk exhausted — release, re-approach fresh
            if abs(th - last_th) > 1.5:
                last_th, last_move_t = th, sim_t["t"]
            elif sim_t["t"] - last_move_t > 1.5 and boost < 18.0:
                boost += 6.0
                last_move_t = sim_t["t"]
                print(f"[solve] k{k}: stall — lead boost {boost:.0f} deg", flush=True)
            lead = max(LEAD_MIN_DEG, min(LEAD_MAX_DEG, abs(err))) + boost
            phi = th + end_off + sgn * lead
            gqq = jaw_of(th + end_off)
            a = torch.zeros(1, n_act, device=device)
            servo(hand_for_tip(arc_point(k, phi, push_z - 0.002), gqq), gqq, PINCH, a)
            tick(a)
            if sim_t["t"] >= next_trace:
                next_trace = sim_t["t"] + 1.5
                tp = tip_pos()
                kx, ky = knob_xy(k)
                tip_az = math.degrees(math.atan2(float(tp[1]) - ky, float(tp[0]) - kx))
                jq = art.data.joint_pos[0]
                print(f"[trace] k{k} th={th:7.2f} err={err:+7.1f} tip_az={tip_az:7.1f} "
                      f"tip_z={float(tp[2]):.3f} w={2 * float(art.data.joint_pos[0, -1]):.3f} "
                      f"q1={jq[0]:+.2f} q4={jq[3]:+.2f} q7={jq[6]:+.2f}", flush=True)
        # RELEASE in place, then rise (open pads leave the bar undisturbed)
        gp, gqq = desc_goal()
        hold(gp, gqq, OPEN, 0.5)
        p, _ = ee_pose()
        hold(V3(p[0], p[1], float(origin[2]) + TRAVEL_Z), gqq, OPEN, 0.9)
        return abs(target(k) - reading(k)) <= STOP_ERR_DEG + 9.0  # capture basin margin

    def set_knob(k: int) -> bool:
        for attempt in range(5):
            th, tg = reading(k), target(k)
            err = tg - th
            if scene.at_target()[0, k]:
                print(f"[solve] knob{k}: at target (err={abs(err):.1f} deg)", flush=True)
                return True
            sgn = 1.0 if err > 0 else -1.0
            print(f"[solve] knob{k} attempt {attempt}: read={th:.1f} tgt={tg:.1f} "
                  f"delta={err:+.1f}", flush=True)
            dewind()
            sweep_chunk(k, sgn)
            dewind()
            # settle: the detent seats the pointer; judge the scene's own predicate
            p, q = ee_pose()
            for _ in range(6):
                hold(p, q, OPEN, 0.4)
                if scene.at_target()[0, k]:
                    break
            report(f"knob{k} a{attempt}")
            if scene.at_target()[0, k]:
                return True
        return bool(scene.at_target()[0, k])

    # ----- mission --------------------------------------------------------------------------------
    print(f"[solve] ctrl={ctrl_hz:.0f}Hz seed={args.seed} base=(0,0,0) facing +x", flush=True)
    print(env.describe(), flush=True)
    p0, q0 = ee_pose()
    hold(p0, q0, OPEN, 1.0)  # settle the fresh scene under the arm's own weight
    # the run-long fist orientation: the home jaw azimuth (never reorients the wrist)
    ey0 = quat_apply(ee_pose()[1].unsqueeze(0), V3(0, 1, 0).unsqueeze(0))[0]
    home_az = math.atan2(float(ey0[1]), float(ey0[0]))
    fist_q = jaw_quat(home_az)
    print(f"[solve] home jaw azimuth = {math.degrees(home_az):.1f} deg", flush=True)
    report("settled")
    score_print("start")

    ok = True
    for k in (0, 1):  # right knob (y<0) first; order is free
        got = set_knob(k)
        ok = ok and got
        score_print(f"knob{k} set" if got else f"knob{k} FAILED")
        if sim_t["t"] > args.max_sec:
            break

    # final: park up and away, settle, judge
    dewind()
    p, q = ee_pose()
    hold(V3(p[0], p[1], float(origin[2]) + TRAVEL_Z), q, OPEN, 0.8)
    hold(V3(float(origin[0]) + 0.25, float(origin[1]), float(origin[2]) + 0.45),
         jaw_quat(0.0), OPEN, 1.5)
    for _ in range(4):
        hold(*ee_pose(), OPEN, 0.5)
    report("final")
    score_print("final")
    # persistence spot-check: success must hold while the world keeps simulating
    persist = True
    for _ in range(6):
        hold(*ee_pose(), OPEN, 0.5)
        persist = persist and bool(scene.success()[0])
    report("persist+3s")

    success = bool(scene.success()[0]) and persist
    if success:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL", flush=True)
    threading.Timer(10.0, lambda: os._exit(0 if success else 1)).start()
    try:
        env.close()
        app.close()
    finally:
        os._exit(0 if success else 1)


if __name__ == "__main__":
    main()

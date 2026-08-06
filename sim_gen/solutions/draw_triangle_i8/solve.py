"""solve — real-robot solution for `bar_triangle` (sim_gen task draw_triangle_i8).

Single Franka arm, robobench OSC, base at (-0.52, 0, 0) facing +x (recorded in TASK.md).
The scene's staging spots fan out at 0.42-0.60 m from exactly this anchor, and the build
plan keeps every grasp/place target in the arm's comfortable 0.35-0.65 m band.

Strategy (arm-only manipulation; task objects are never written after the initial reset):
  PLAN     read the settled mat centre; compute the ideal frame layout (scene.frame_layout,
           uniform ~24 mm corner gaps, well under the 40 mm tolerance), centred slightly
           base-ward of the mat centre, rotation chosen to minimize the worst reach;
  per bar (farthest target first):
    PICK   hover over the bar's midpoint, jaw across the bar (branch nearest the wrist
           home azimuth), descend fingertips to 4 mm above the ground so the pads cover
           the bar's LOWER half (squeeze-down, the pen-corpus grasp), cage then ramp
           closed, lift, verdict = the BAR rose;
    PLACE  closed-loop carry on the live bar centre onto its planned centre while a yaw
           servo maps the bar's live long axis onto the planned side direction (mod pi);
           lower until the bar is just above rest height, wait for stillness, slow-release,
           retreat;
    a bar that lands off-plan (>12 mm / >4 deg) is re-picked with a compensated aim.
  REPAIR   after all three: any unclosed corner -> re-place the worse-off adjacent bar
           with a compensated aim (up to 2 rounds).

Prints the scene's own readouts (corner_report) and `SIM_GEN_SCORE <score>` at every phase
boundary (scores are latched by the scene, so the printed series never decreases), then
exactly `SIM_GEN_SOLVE: SUCCESS` iff scene.success() holds on the settled end state.

Run (forge): python -m simgen_tasks.draw_triangle_i8.solve --headless [--seed N]
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_sec", type=float, default=420.0, help="sim-time budget (s)")
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
    from simgen_tasks.draw_triangle_i8 import scene as scene_mod
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod

BASE_POS = (-0.52, 0.0, 0.0)  # the scene's stage_anchor (see TASK.md)
OPEN = 0.04          # per-finger position target, fully open (80 mm aperture)
FINGER_LEN = 0.112   # panda_hand frame -> fingertip midpoint along the approach axis
TRAVEL_HAND_Z = 0.24  # empty transit hand height (tips ~0.13; nothing in scene is taller)
CARRY_HAND_Z = 0.26   # loaded transit: an 18 mm bar pinched at the tips hangs at ~0.14
CAGE = 0.015          # per-finger pre-close: 30 mm aperture cages the 18 mm bar
GRIP = 0.005          # per-finger close target: firm squeeze on the 18 mm bar


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def wrap_half(a: float) -> float:
    """Wrap to (-pi/2, pi/2] — bar axis is direction-symmetric."""
    return (a + math.pi / 2) % math.pi - math.pi / 2


def main() -> None:  # noqa: PLR0915
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    robobench.discover()

    rcfg = FrankaRobotCfg(
        base_pos=BASE_POS,
        nullspace_dof_pos=(),          # default posture target winds the arm on long servos
        gripper_effort_limit=120.0,
        gripper_stiffness=4000.0,
    )
    env = EnvCfg(scene="bar_triangle", robot="franka", control_mode="osc",
                 robot_cfg=rcfg, env_spacing=3.0, seed=args.seed).build(
                     num_envs=1, device=device)
    env.reset()
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
    origin = env.iscene.env_origins[0]
    base_xy = art.data.root_pos_w[0, :2]

    def V3(x, y, z):
        return torch.tensor([float(x), float(y), float(z)], device=device)

    def SEC(s: float) -> int:
        return max(1, round(s * ctrl_hz))

    # ----- state reads ----------------------------------------------------------------------
    def ee_pose():
        return art.data.body_pos_w[0, ee_idx], art.data.body_quat_w[0, ee_idx]

    def width() -> float:
        return 2.0 * art.data.joint_pos[0, fj1].item()

    def bar_pos(i: int) -> torch.Tensor:
        return scene.bars[i].data.root_pos_w[0] - origin  # env-local

    def bar_yaw(i: int) -> float:
        q = scene.bars[i].data.root_quat_w[0]
        ax = quat_apply(q.unsqueeze(0), torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
        return math.atan2(float(ax[1]), float(ax[0]))

    def bar_speed(i: int) -> float:
        return float(scene.bars[i].data.root_lin_vel_w[0].norm())

    # ----- orientation ----------------------------------------------------------------------
    def jaw_quat(azimuth: float) -> torch.Tensor:
        yh = V3(math.cos(azimuth), math.sin(azimuth), 0.0)
        zh = V3(0.0, 0.0, -1.0)
        xh = torch.cross(yh, zh, dim=0)
        return quat_from_matrix(torch.stack([xh, yh, zh], dim=1).unsqueeze(0))[0]

    def jaw_az_of(q) -> float:
        ey = quat_apply(q.unsqueeze(0), torch.tensor([[0.0, 1.0, 0.0]], device=device))[0]
        return math.atan2(float(ey[1]), float(ey[0]))

    def tilted(gq: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
        """Lean the hand away from the base column for close-in targets (<0.37 m)."""
        d = float((target_xy - base_xy).norm())
        if d >= 0.37:
            return gq
        tilt = min(0.35, (0.37 - d) * 5.0)
        u = (target_xy - base_xy) / max(d, 1e-6)
        axis = V3(-float(u[1]), float(u[0]), 0.0)
        return quat_mul(quat_from_angle_axis(
            torch.tensor([tilt], device=device), axis.unsqueeze(0))[0].unsqueeze(0),
            gq.unsqueeze(0))[0]

    # ----- servo kernel ---------------------------------------------------------------------
    sim_t = {"t": 0.0}

    def servo(goal_pos, goal_quat, grip, a, xy_boost=1.0, rot_gain=1.0):
        p, q = ee_pose()
        err = goal_pos - p
        err = torch.cat([err[:2] * xy_boost, err[2:3]])
        a[0, 0:3] = (err / osc.cfg.pos_scale).clamp(-1.0, 1.0)
        qe = quat_mul(goal_quat.unsqueeze(0), quat_conjugate(q.unsqueeze(0)))
        a[0, 3:6] = (axis_angle_from_quat(qe)[0] / osc.cfg.rot_scale).clamp(-1.0, 1.0) * rot_gain
        a[0, 6:8] = grip

    def tick(a):
        env.step(a)
        sim_t["t"] += env.dt * robot.control_period

    def hold(pos, quat, grip, secs, xy_boost=1.0):
        for _ in range(SEC(secs)):
            a = torch.zeros(1, n_act, device=device)
            servo(pos, quat, grip, a, xy_boost)
            tick(a)

    def run_phase(goal_fn, gate_fn, grip, timeout_s, tag="", xy_boost=1.0, rot_gain=1.0):
        deadline = sim_t["t"] + timeout_s
        while sim_t["t"] < deadline:
            a = torch.zeros(1, n_act, device=device)
            gp, gq = goal_fn()
            servo(gp, gq, grip, a, xy_boost, rot_gain)
            tick(a)
            if gate_fn():
                return True
        if tag:
            p, _ = ee_pose()
            gp, _ = goal_fn()
            print(f"[phase:{tag}] timeout ee=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
                  f"goal=({gp[0]:.3f},{gp[1]:.3f},{gp[2]:.3f}) w={width() * 1000:.1f}mm",
                  flush=True)
        return False

    def dewind() -> bool:
        q = art.data.joint_pos[0]
        if (abs(q[0].item()) > 2.6 or q[3].item() < -2.95 or q[3].item() > -0.15
                or abs(q[6].item()) > 2.6):
            print(f"[solve] wound arm (q1={q[0]:.2f} q4={q[3]:.2f} q7={q[6]:.2f}) — joint reset",
                  flush=True)
            robot.reset(torch.tensor([0], device=device, dtype=torch.long))
            p, qq = ee_pose()
            hold(p, qq, OPEN, 0.5)
            return True
        return False

    def settle(secs: float = 1.0) -> None:
        p, q = ee_pose()
        hold(p, q, OPEN, secs)

    last_score = {"v": -1.0}

    def report(tag: str) -> None:
        s = float(scene.score()[0])
        print(f"[solve] {tag}: {scene.corner_report()}", flush=True)
        if s < last_score["v"] - 1e-6:
            print(f"[solve] WARNING: score regressed {last_score['v']:.3f} -> {s:.3f}",
                  flush=True)
        last_score["v"] = max(last_score["v"], s)
        print(f"SIM_GEN_SCORE {s:.3f}", flush=True)

    # ----- plan -----------------------------------------------------------------------------
    settle(1.5)
    mat = (scene.pad.data.root_pos_w[0] - origin)[:2]
    print(env.describe(), flush=True)
    print(f"[solve] seed={args.seed} mat=({mat[0]:.3f},{mat[1]:.3f})", flush=True)
    for i in range(3):
        p = bar_pos(i)
        print(f"[solve] bar_{i} L={c.bar_lengths[i] * 1000:.0f}mm at "
              f"({p[0]:.3f},{p[1]:.3f}) yaw={math.degrees(bar_yaw(i)):.0f}deg "
              f"d_base={float((p[:2] - (base_xy - origin[:2])).norm()):.2f}m", flush=True)
    report("initial")

    # build centre: slightly base-ward of the mat centre; rotation chosen to minimize the
    # worst endpoint reach from the base (keeps every target in the comfortable band)
    build_c = (float(mat[0]) - 0.03, float(mat[1]))
    bx, by = BASE_POS[0], BASE_POS[1]

    def worst_reach(rot: float) -> float:
        centers, yaws, _V, _g = scene_mod.frame_layout(c, build_c, rot)
        worst = 0.0
        for k in range(3):
            for sgn in (-1.0, 1.0):
                ex_ = centers[k][0] + sgn * math.cos(yaws[k]) * c.bar_lengths[k] / 2
                ey_ = centers[k][1] + sgn * math.sin(yaws[k]) * c.bar_lengths[k] / 2
                worst = max(worst, math.hypot(ex_ - bx, ey_ - by))
        return worst

    rot = min((math.radians(r) for r in range(0, 360, 10)), key=worst_reach)
    centers, yaws, verts, plan_gaps = scene_mod.frame_layout(c, build_c, rot)
    print(f"[solve] plan: rot={math.degrees(rot):.0f}deg worst_reach={worst_reach(rot):.3f}m "
          f"planned_gaps_mm={[round(g * 1000, 1) for g in plan_gaps]}", flush=True)

    home_az = jaw_az_of(ee_pose()[1])

    # ----- primitives -------------------------------------------------------------------------
    def pick(i: int) -> bool:
        """Grasp bar i across its width at the midpoint. True iff the bar rose with the hand."""
        dewind()
        p0 = bar_pos(i)
        az_bar = bar_yaw(i)
        cands = [az_bar + math.pi / 2, az_bar - math.pi / 2]
        jaw_az = min(cands, key=lambda a2: abs(wrap_pi(a2 - home_az)))
        gq = tilted(jaw_quat(jaw_az), p0[:2] + origin[:2])

        def tip_goal(z_local):
            """Hand pose putting the fingertip midpoint over the live bar centre at
            env-local height `z_local`."""
            o = bar_pos(i)
            tip_w = origin + V3(float(o[0]), float(o[1]), z_local)
            zh = quat_apply(gq.unsqueeze(0), V3(0, 0, 1).unsqueeze(0))[0]
            return tip_w - zh * FINGER_LEN, gq

        if not run_phase(lambda: tip_goal(TRAVEL_HAND_Z - FINGER_LEN),
                         lambda: float((ee_pose()[0][:2] - (bar_pos(i)[:2] + origin[:2]))
                                       .norm()) < 0.008,
                         OPEN, 8.0, tag=f"hover{i}"):
            return False
        # descend: fingertips to 4 mm above ground — pads cover the bar's lower half
        grasp_tip_z = c.surface_z + 0.004  # env-local

        def tip_z_now() -> float:
            p, q = ee_pose()
            zh = quat_apply(q.unsqueeze(0), V3(0, 0, 1).unsqueeze(0))[0]
            return float((p + zh * FINGER_LEN)[2])

        if not run_phase(lambda: tip_goal(grasp_tip_z),
                         lambda: abs(tip_z_now() - (float(origin[2]) + grasp_tip_z)) < 0.004
                         and float((ee_pose()[0][:2] - (bar_pos(i)[:2] + origin[:2]))
                                   .norm()) < 0.005,
                         CAGE, 8.0, tag=f"descend{i}"):
            return False
        # micro-stabilise, then quasi-static close ramp
        gp, _ = tip_goal(grasp_tip_z)
        hold(gp, gq, CAGE, 0.4)
        n_close = SEC(1.2)
        for k in range(n_close):
            a = torch.zeros(1, n_act, device=device)
            gp, _ = tip_goal(grasp_tip_z)
            g2 = CAGE + (GRIP - CAGE) * min(1.0, (k + 1) / (n_close * 0.7))
            servo(gp, gq, g2, a)
            tick(a)
        # lift + verdict: the BAR must rise
        z0 = float(bar_pos(i)[2])
        p, _ = ee_pose()
        for zt in (0.18, CARRY_HAND_Z):
            hold(V3(float(p[0]), float(p[1]), float(origin[2]) + zt), gq, GRIP, 0.8)
        rose = float(bar_pos(i)[2]) - z0
        w = width()
        ok = rose > 0.04 and 0.012 < w < 0.026
        print(f"[solve] pick bar_{i}: rose={rose * 1000:.0f}mm w={w * 1000:.1f}mm "
              f"{'HELD' if ok else 'FAILED'}", flush=True)
        if not ok:
            p, q = ee_pose()
            hold(V3(float(p[0]), float(p[1]), float(origin[2]) + TRAVEL_HAND_Z), q, OPEN, 0.8)
        return ok

    def place(i: int, tgt_xy, tgt_yaw: float) -> None:
        """Carry the held bar onto (tgt_xy, tgt_yaw) and release it there."""
        tgt = V3(tgt_xy[0], tgt_xy[1], 0.0)[:2]

        def yaw_err() -> float:
            return wrap_half(tgt_yaw - bar_yaw(i))

        def carry_goal():
            p, q = ee_pose()
            o = bar_pos(i)
            corr = (1.3 * (tgt - o[:2])).clamp(-0.012, 0.012)
            gq2 = jaw_quat(jaw_az_of(q) + max(-0.25, min(0.25, yaw_err())))
            return V3(float(p[0] + corr[0]), float(p[1] + corr[1]),
                      float(origin[2]) + CARRY_HAND_Z), gq2

        run_phase(carry_goal,
                  lambda: float((bar_pos(i)[:2] - tgt).norm()) < 0.004
                  and abs(yaw_err()) < math.radians(1.5),
                  GRIP, 16.0, tag=f"carry{i}", xy_boost=1.6, rot_gain=0.6)

        rest_z = c.surface_z + c.bar_w / 2

        def lower_goal():
            p, q = ee_pose()
            o = bar_pos(i)
            corr = (1.2 * (tgt - o[:2])).clamp(-0.008, 0.008)
            gq2 = jaw_quat(jaw_az_of(q) + max(-0.15, min(0.15, yaw_err())))
            dz = float(o[2]) - (rest_z + 0.003)
            return V3(float(p[0] + corr[0]), float(p[1] + corr[1]), float(p[2]) - dz), gq2

        run_phase(lower_goal,
                  lambda: float(bar_pos(i)[2]) < rest_z + 0.006,
                  GRIP, 10.0, tag=f"lower{i}", xy_boost=1.6, rot_gain=0.6)
        # pre-release stillness, then slow release and vertical retreat
        deadline = sim_t["t"] + 2.0
        while sim_t["t"] < deadline:
            a = torch.zeros(1, n_act, device=device)
            p, q = ee_pose()
            o = bar_pos(i)
            corr = (1.0 * (tgt - o[:2])).clamp(-0.004, 0.004)
            servo(V3(float(p[0] + corr[0]), float(p[1] + corr[1]), float(p[2])), q, GRIP, a)
            tick(a)
            if bar_speed(i) < 0.02 and float((o[:2] - tgt).norm()) < 0.006:
                break
        p, q = ee_pose()
        hold(p, q, 0.025, 0.5)
        hold(p, q, OPEN, 0.4)
        hold(V3(float(p[0]), float(p[1]), float(origin[2]) + TRAVEL_HAND_Z), q, OPEN, 0.9)

    def place_bar(i: int, aim_xy, aim_yaw: float, attempts: int = 3) -> bool:
        """pick + place with per-attempt aim compensation. True when the bar settled within
        12 mm / 4 deg of the PLAN pose (aim may differ — it absorbs systematic bias)."""
        aim = [float(aim_xy[0]), float(aim_xy[1])]
        ayaw = aim_yaw
        for att in range(attempts):
            if sim_t["t"] > args.max_sec:
                return False
            if not pick(i):
                continue
            place(i, aim, ayaw)
            settle(0.8)
            o = bar_pos(i)
            err_xy = math.hypot(float(o[0]) - centers[i][0], float(o[1]) - centers[i][1])
            err_yaw = abs(wrap_half(yaws[i] - bar_yaw(i)))
            print(f"[solve] placed bar_{i} attempt {att}: err_xy={err_xy * 1000:.1f}mm "
                  f"err_yaw={math.degrees(err_yaw):.1f}deg", flush=True)
            if err_xy < 0.012 and err_yaw < math.radians(4.0):
                return True
            # compensate: shift the aim by the measured miss (relative to the PLAN pose)
            aim[0] += centers[i][0] - float(o[0])
            aim[1] += centers[i][1] - float(o[1])
            ayaw = ayaw + wrap_half(yaws[i] - bar_yaw(i))
        return False

    # ----- mission --------------------------------------------------------------------------
    order = sorted(range(3), key=lambda k: -math.hypot(centers[k][0] - bx, centers[k][1] - by))
    print(f"[solve] placement order (far first): {order}", flush=True)
    for i in order:
        ok = place_bar(i, centers[i], yaws[i])
        dewind()
        settle(1.0)
        report(f"after bar_{i} ({'ok' if ok else 'off-plan'})")

    # ----- repair rounds ----------------------------------------------------------------------
    for rnd in range(2):
        settle(1.0)
        if bool(scene.success()[0]):
            break
        gaps, cnt, _xy, _use = scene.corners()
        bad = [k for k in range(3) if not bool(cnt[0, k])]
        if not bad:
            break
        # worst pair; re-place the adjacent bar whose pose deviates most from plan
        k = max(bad, key=lambda p: float(gaps[0, p]))
        a_, b_ = scene_mod.PAIRS[k]

        def dev(j: int) -> float:
            o = bar_pos(j)
            return (math.hypot(float(o[0]) - centers[j][0], float(o[1]) - centers[j][1])
                    + 0.09 * abs(wrap_half(yaws[j] - bar_yaw(j))))

        j = max((a_, b_), key=dev)
        print(f"[solve] repair round {rnd}: corner pair {k} open "
              f"(gap={float(gaps[0, k]) * 1000:.0f}mm) -> re-place bar_{j}", flush=True)
        place_bar(j, centers[j], yaws[j], attempts=2)
        dewind()
        settle(1.0)
        report(f"after repair {rnd}")

    # ----- verdict ------------------------------------------------------------------------
    settle(2.0)
    stable = True
    for _ in range(8):
        settle(0.3)
        if not bool(scene.success()[0]):
            stable = False
            break
    report("final")
    ok = stable and bool(scene.success()[0])
    print(f"[solve] sim_t={sim_t['t']:.0f}s", flush=True)
    if ok:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL", flush=True)
    code = 0 if ok else 1
    threading.Timer(15.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()

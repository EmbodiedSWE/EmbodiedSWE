"""solve — sear_and_serve (meat_off_grill_i18) with a REAL Franka arm, OSC servo loop.

Plan (base at BASE = the scene's stage_anchor; every grasp/place 0.40-0.64 m out):

  For each PRESENT patty (steak, then chicken), sequentially — the sear pad is one-slot:
  COOK    top-down pinch of the patty disc on the prep board (60 / 54 mm — proven
          parallel-jaw sizes), lift-verdict, closed-loop carry on the PATTY xy to the
          live pad center, lower until the scene's own on_pad() clause holds, then HOLD
          the patty against the pad — the cook clock is real accumulated pad contact —
          until cook_t crosses cook_min + margin (well short of burn_time), and lift
          straight off. The window is judged by the scene; the arm never releases on
          the pad, so the deadline is met by construction.
  SERVE   still gripped, carry to a free serving spot on the plate, lower to rest
          height, slow release, retreat; wait for served() (settled, flat, cooked,
          unburned, on the plate).

No task-object state is ever written; only arm + gripper joints are commanded through
the robobench OSC controller. Prints the scene readouts and `SIM_GEN_SCORE <score>` at
each phase boundary (latched credit is monotone along this trajectory) and exactly
`SIM_GEN_SOLVE: SUCCESS` once scene.success() holds and persists.

Run:  python -m simgen_tasks.meat_off_grill_i18.solve --headless [--seed N]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
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
    quat_from_angle_axis,
    quat_from_matrix,
    quat_mul,
)

import robobench  # noqa: E402
from robobench.core import EnvCfg  # noqa: E402
from robobench.robots.franka import FrankaRobotCfg  # noqa: E402

try:
    from . import scene as task_scene  # noqa: F401  (registers the scene)
except ImportError:  # direct-script fallback
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import scene as task_scene  # noqa: F401

# watchdog: never hang the forge — verdict or death
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(1))).start()

OPEN = 0.04          # per-finger target, fully open (80 mm aperture)
FINGER_LEN = 0.112   # panda_hand frame -> fingertip midpoint
BASE = (-0.55, 0.0, 0.0)  # arm base on the ground = scene cfg stage_anchor
CARRY_Z = 0.20       # patty-center transit height (clears the 108 mm grill by 80 mm)

# per-patty pinch parameters: grip command + width gates (proven 60/54 mm pinch numbers)
PINCH = {
    "steak": {"grip": 0.026, "w_lo": 0.050, "w_hi": 0.070},
    "chicken": {"grip": 0.022, "w_lo": 0.044, "w_hi": 0.064},
}


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    robobench.discover()
    rcfg = FrankaRobotCfg(
        base_pos=BASE,
        nullspace_dof_pos=(),      # default posture target winds the arm on long servos
        gripper_effort_limit=120.0,
        gripper_stiffness=4000.0,
    )
    env = EnvCfg(scene="sear_and_serve", robot="franka", control_mode="osc",
                 env_spacing=3.0, robot_cfg=rcfg).build(num_envs=1, device=device)
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
    osc.cfg.kd_null = 2.0 * math.sqrt(3.0)
    n_act = robot.action_dim
    ctrl_hz = 1.0 / (env.dt * robot.control_period)
    SEC = lambda s: max(1, round(s * ctrl_hz))  # noqa: E731

    env.reset(seed=args.seed)  # seed at reset time: build consumes RNG state
    print("[solve] describe():\n" + scene.describe(), flush=True)
    gp0 = scene.grill.data.root_pos_w[0]
    bp0 = scene.board.data.root_pos_w[0]
    pp0 = scene.plate.data.root_pos_w[0]
    print(f"[solve] layout: board=({bp0[0]:.3f},{bp0[1]:.3f}) "
          f"grill=({gp0[0]:.3f},{gp0[1]:.3f}) plate=({pp0[0]:.3f},{pp0[1]:.3f}) "
          f"present={scene.present[0].int().tolist()}", flush=True)

    V3 = lambda x, y, z: torch.tensor([float(x), float(y), float(z)], device=device)  # noqa: E731

    # ----- kernel (franka_session lineage) --------------------------------------------------
    def ee_pose():
        return art.data.body_pos_w[0, ee_idx], art.data.body_quat_w[0, ee_idx]

    def width() -> float:
        return 2.0 * art.data.joint_pos[0, fj1].item()

    def servo(goal_pos, goal_quat, grip, a, xy_boost: float = 1.0):
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
                  f"goal=({gp[0]:.3f},{gp[1]:.3f},{gp[2]:.3f}) "
                  f"err={float((gp - p).norm()) * 1000:.0f}mm w={width() * 1000:.1f}mm",
                  flush=True)
        return False

    def close_ramp(pos, quat, grip_target: float, secs: float = 1.2):
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

    def dewind(grip: float = OPEN) -> None:
        q = art.data.joint_pos[0]
        if (abs(q[0].item()) > 2.6 or q[3].item() < -2.95 or q[3].item() > -0.15
                or abs(q[6].item()) > 2.6):
            print(f"[solve] wound arm (q1={q[0]:.2f} q4={q[3]:.2f} q7={q[6]:.2f}) — reset",
                  flush=True)
            robot.reset(torch.tensor([0], device=device, dtype=torch.long))
            p, qq = ee_pose()
            hold(p, qq, grip, 0.5)

    # ----- scene readouts ---------------------------------------------------------------------
    score_trace: list[float] = []

    def boundary(tag: str) -> None:
        s = float(scene.score()[0])
        score_trace.append(s)
        print(f"[readout] {tag}: cook_t={[f'{v:.2f}' for v in scene.cook_t[0].tolist()]} "
              f"cooked={scene.cooked()[0].int().tolist()} "
              f"burned={scene.burned[0].int().tolist()} "
              f"served={scene.served()[0].int().tolist()} "
              f"present={scene.present[0].int().tolist()} "
              f"success={bool(scene.success()[0])}", flush=True)
        print(f"SIM_GEN_SCORE {s:.3f}", flush=True)

    # ----- geometry handles (ALL WORLD-frame) --------------------------------------------------
    def pad_center_xy() -> torch.Tensor:
        return scene.grill.data.root_pos_w[0][:2]  # pad is centered on the grill body

    def plate_slot_xy(slot: int) -> torch.Tensor:
        pp = scene.plate.data.root_pos_w[0]
        sx, sy = c.plate_slots[slot]
        return V3(pp[0] + sx, pp[1] + sy, 0.0)[:2]

    def bearing_from_base(xy: torch.Tensor) -> float:
        return math.atan2(float(xy[1]) - BASE[1], float(xy[0]) - BASE[0])

    # ----- per-patty routine --------------------------------------------------------------------
    def cook_and_serve(i: int, name: str, slot: int) -> bool:
        pinch = PINCH[name]
        grip, w_lo, w_hi = pinch["grip"], pinch["w_lo"], pinch["w_hi"]
        h = c.patties[i][2]
        obj = scene.patties[name]

        def obj_pos() -> torch.Tensor:
            return obj.data.root_pos_w[0]

        grill_z = float(scene.grill.data.root_pos_w[0][2])
        pad_rest_z = grill_z + c.pad_top_local + h / 2  # patty center resting on the pad
        plate_z = float(scene.plate.data.root_pos_w[0][2])
        plate_rest_z = plate_z + c.plate_t / 2 + h / 2
        cook_target = c.cook_min + 0.6  # lift at 2.1 s: >= cook_min, << burn_time

        for attempt in range(5):
            dewind()
            if bool(scene.burned[0, i]):
                print(f"[solve] {name}: BURNED — unrecoverable", flush=True)
                return False

            # ---- grasp at the patty's live position -------------------------------------
            az = bearing_from_base(obj_pos()[:2]) + (0.0, math.pi / 4, -math.pi / 4,
                                                     math.pi / 2, math.pi / 3)[attempt]
            gq = tilt_quat(az, obj_pos()[:2])

            def tip_goal(z, gq=gq):
                o = obj_pos()
                return V3(o[0], o[1], z + FINGER_LEN), gq

            if not run_phase(lambda: tip_goal(CARRY_Z),
                             lambda: (ee_pose()[0][:2] - obj_pos()[:2]).norm() < 0.006,
                             OPEN, 8.0, tag=f"{name}-hover{attempt}"):
                continue
            if not run_phase(lambda: tip_goal(float(obj_pos()[2])),
                             lambda: abs(float(ee_pose()[0][2])
                                         - (float(obj_pos()[2]) + FINGER_LEN)) < 0.005
                             and (ee_pose()[0][:2] - obj_pos()[:2]).norm() < 0.005,
                             OPEN, 8.0, tag=f"{name}-descend{attempt}"):
                p, _ = ee_pose()
                hold(V3(p[0], p[1], CARRY_Z + FINGER_LEN), gq, OPEN, 0.8)
                continue

            gp, _ = tip_goal(float(obj_pos()[2]))
            close_ramp(gp, gq, grip)

            z0 = float(obj_pos()[2])
            p, _ = ee_pose()
            run_phase(lambda p=p, gq=gq: (V3(p[0], p[1], CARRY_Z + FINGER_LEN), gq),
                      lambda: float(obj_pos()[2]) > z0 + 0.06,
                      grip, 6.0, tag=f"{name}-lift{attempt}")
            w = width()
            if float(obj_pos()[2]) - z0 < 0.04 or not (w_lo < w < w_hi):
                print(f"[solve] {name}: lift verdict failed (dz="
                      f"{float(obj_pos()[2]) - z0:.3f} w={w * 1000:.0f}mm) — retry",
                      flush=True)
                p, _ = ee_pose()
                hold(p, gq, OPEN, 0.6)
                hold(V3(p[0], p[1], CARRY_Z + FINGER_LEN + 0.05), gq, OPEN, 1.0)
                continue

            def carry_goal(target_xy_fn, gq=gq):
                o, hd = obj_pos(), ee_pose()[0]
                t = target_xy_fn()
                corr = (1.3 * (t - o[:2])).clamp(-0.015, 0.015)
                return V3(hd[0] + corr[0], hd[1] + corr[1], CARRY_Z + FINGER_LEN), gq

            # ---- COOK (skipped if the cook latch already fired on an earlier attempt) ----
            if not bool(scene.cooked()[0, i]):
                run_phase(lambda: carry_goal(pad_center_xy),
                          lambda: (obj_pos()[:2] - pad_center_xy()).norm() < 0.006,
                          grip, 14.0, tag=f"{name}-carry-pad{attempt}", xy_boost=1.6)
                if not (w_lo < width() < w_hi):
                    print(f"[solve] {name}: slipped in transit to the pad — retry",
                          flush=True)
                    continue

                def lower_pad_goal(gq=gq):
                    o, hd = obj_pos(), ee_pose()[0]
                    corr = (1.2 * (pad_center_xy() - o[:2])).clamp(-0.008, 0.008)
                    return V3(hd[0] + corr[0], hd[1] + corr[1],
                              pad_rest_z - 0.001 + FINGER_LEN), gq

                if not run_phase(lower_pad_goal,
                                 lambda: bool(scene.on_pad()[0, i]),
                                 grip, 10.0, tag=f"{name}-lower-pad{attempt}",
                                 xy_boost=1.6):
                    p, _ = ee_pose()
                    hold(V3(p[0], p[1], CARRY_Z + FINGER_LEN), gq, grip, 1.5)
                    continue
                boundary(f"{name}-on-pad")

                # hold against the pad; the scene's cook clock does the timing
                cooked_now = run_phase(
                    lower_pad_goal,
                    lambda: float(scene.cook_t[0, i]) >= cook_target,
                    grip, 8.0, tag=f"{name}-cook{attempt}")
                ct = float(scene.cook_t[0, i])
                print(f"[solve] {name}: leaving the pad at cook_t={ct:.2f}s "
                      f"(window [{c.cook_min:.1f}, {c.burn_time:.1f}))", flush=True)
                p, _ = ee_pose()
                run_phase(lambda p=p, gq=gq: (V3(p[0], p[1], CARRY_Z + FINGER_LEN), gq),
                          lambda: float(obj_pos()[2]) > pad_rest_z + 0.05,
                          grip, 6.0, tag=f"{name}-liftoff{attempt}")
                if not cooked_now or not bool(scene.cooked()[0, i]) \
                        or bool(scene.burned[0, i]):
                    print(f"[solve] {name}: cook window not met "
                          f"(cook_t={float(scene.cook_t[0, i]):.2f} "
                          f"burned={bool(scene.burned[0, i])}) — retry", flush=True)
                    continue
                boundary(f"{name}-cooked")
                if not (w_lo < width() < w_hi):
                    print(f"[solve] {name}: slipped at lift-off — retry", flush=True)
                    continue

            # ---- SERVE -------------------------------------------------------------------
            run_phase(lambda: carry_goal(lambda: plate_slot_xy(slot)),
                      lambda: (obj_pos()[:2] - plate_slot_xy(slot)).norm() < 0.006,
                      grip, 14.0, tag=f"{name}-carry-plate{attempt}", xy_boost=1.6)
            if not (w_lo < width() < w_hi):
                print(f"[solve] {name}: slipped in transit to the plate — retry",
                      flush=True)
                continue

            def lower_plate_goal(gq=gq):
                o, hd = obj_pos(), ee_pose()[0]
                corr = (1.2 * (plate_slot_xy(slot) - o[:2])).clamp(-0.008, 0.008)
                return V3(hd[0] + corr[0], hd[1] + corr[1],
                          plate_rest_z + 0.001 + FINGER_LEN), gq

            if not run_phase(lower_plate_goal,
                             lambda: abs(float(obj_pos()[2]) - plate_rest_z) < 0.006
                             and (obj_pos()[:2] - plate_slot_xy(slot)).norm() < 0.010,
                             grip, 10.0, tag=f"{name}-lower-plate{attempt}",
                             xy_boost=1.6):
                p, _ = ee_pose()
                hold(V3(p[0], p[1], CARRY_Z + FINGER_LEN), gq, grip, 1.5)
                continue

            p, _ = ee_pose()
            hold(p, gq, OPEN, 0.8)                                  # slow release
            hold(V3(p[0], p[1], p[2] + 0.12), gq, OPEN, 1.0)        # retreat up

            deadline = sim_t["t"] + 5.0
            while sim_t["t"] < deadline:
                pp, _ = ee_pose()
                hold(pp, gq, OPEN, env.dt * robot.control_period)
                if bool(scene.served()[0, i]):
                    return True
            print(f"[solve] {name}: placed but served() false — retry", flush=True)
        return False

    # =========================== run ===========================================================
    p0, q0 = ee_pose()
    hold(p0, q0, OPEN, 1.0)
    boundary("start")

    slot = 0
    for i, (name, _r, _h, _m, _rgb) in enumerate(c.patties):
        if not bool(scene.present[0, i]):
            continue
        if not cook_and_serve(i, name, slot):
            boundary(f"{name}-FAILED")
            print("SIM_GEN_SOLVE: FAIL", flush=True)
            os._exit(1)
        slot += 1
        boundary(f"{name}-served")

    # ---- persistence: park the arm, keep simulating, success must hold ------------------------
    dewind()
    p, _q = ee_pose()
    hold(V3(float(p[0]), float(p[1]), 0.45), jaw_quat(0.0), OPEN, 1.0)
    okp = True
    for _ in range(6):
        pp, _ = ee_pose()
        hold(pp, jaw_quat(0.0), OPEN, 0.5)
        okp = okp and bool(scene.success()[0])
    boundary("final")

    mono = all(score_trace[j] <= score_trace[j + 1] + 1e-6
               for j in range(len(score_trace) - 1))
    print(f"[solve] score trace: {[f'{v:.3f}' for v in score_trace]} monotone={mono}",
          flush=True)
    if okp and bool(scene.success()[0]) and float(scene.score()[0]) == 1.0 and mono:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
        os._exit(0)
    print("SIM_GEN_SOLVE: FAIL", flush=True)
    os._exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)

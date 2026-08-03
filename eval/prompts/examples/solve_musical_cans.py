"""solve — musical_cans (registry_i28) with a REAL Franka arm, OSC servo loop.

Strategy: read the sampled derangement from the scene (start_pad readback), compute the
forced move sequence through the gray buffer cup (pad 3), then execute each move as a
closed-loop pick / lift / carry / lower / release with the arm only:

  HOVER    top-down hand above the live can position
  DESCEND  fingertips to can-centre height (fingers straddle the can along the jaw axis)
  CLOSE    pinch the 60 mm cylinder; width gate
  LIFT     straight up until the can is clear of every collar and can top
  CARRY    closed-loop on the CAN xy onto the destination cup axis
  LOWER    can bottom to ~8 mm above the collar rim, xy gated < 5 mm
  RELEASE  open, retreat; poll the scene's own _seated_on/settled while the can seats

The counter is lava (irreversible `dropped` latch) so nothing is ever set down outside
a cup; transit heights clear all cans/collars. Success is scene.success() verbatim.

Run:  python -m simgen_tasks.musical_cans.solve --headless
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_sec", type=float, default=500.0, help="sim-time budget per episode (s)")
parser.add_argument("--episodes", type=int, default=1, help="episodes to run (each is a fresh reset)")
parser.add_argument("--video", action="store_true", help="record an mp4 of the run")
parser.add_argument("--video_path", type=str, default="simgen_tasks/musical_cans/solve_video.mp4")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True

app = AppLauncher(args).app

import torch  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    axis_angle_from_quat,
    quat_conjugate,
    quat_from_matrix,
    quat_mul,
)

import robobench.controllers  # noqa: E402,F401  (registers controllers)
import robobench.robots  # noqa: E402,F401  (registers robots)
import sim_gen.tasks.registry_i28.scene  # noqa: E402,F401  (registers the scene)
from robobench.core import EnvCfg  # noqa: E402
from robobench.robots.franka import FrankaRobotCfg  # noqa: E402

OPEN, CLOSED = 0.04, 0.020  # gentle pinch: hard close tunnels the 50 g can
FINGER_LEN = 0.112          # panda_hand frame -> fingertip (pen_holder-measured)
BASE = (-0.35, 0.0, 0.22)   # arm mounted ON the counter slab, west of the pad row


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    rcfg = FrankaRobotCfg(
        base_pos=BASE,
        nullspace_dof_pos=(),          # pen_holder lesson: default posture winds the arm
        gripper_effort_limit=25.0,
        gripper_stiffness=2000.0,
    )
    if args.video:  # inject a spectator camera into the scene's assets before build
        from isaaclab.sensors import CameraCfg
        import isaaclab.sim as sim_utils
        from sim_gen.tasks.registry_i28.scene import MusicalCansScene

        _orig_assets = MusicalCansScene.assets

        def _assets_with_cam(self):
            out = _orig_assets(self)
            out["video_cam"] = CameraCfg(
                prim_path="/World/video_cam",
                update_period=0.0,
                height=480, width=640,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=18.0, clipping_range=(0.05, 20.0)),
            )
            return out

        MusicalCansScene.assets = _assets_with_cam

    cfg = EnvCfg(scene="musical_cans", robot="franka", control_mode="osc",
                 env_spacing=3.0, robot_cfg=rcfg)
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
    CTRL_HZ = 1.0 / (env.dt * robot.control_period)
    SEC = lambda s: max(1, round(s * CTRL_HZ))  # noqa: E731

    env.reset()
    origin = env.iscene.env_origins[0]

    frames: list = []
    log_rows: list = []
    log_pads: list = []
    cam = None
    if args.video:
        cam = env.iscene["video_cam"]
        eye = origin + torch.tensor([0.95, -0.55, 0.95], device=device)
        tgt = origin + torch.tensor([-0.15, 0.0, 0.30], device=device)
        cam.set_world_poses_from_view(eye.unsqueeze(0), tgt.unsqueeze(0))
    V3 = lambda x, y, z: torch.tensor([float(x), float(y), float(z)], device=device)  # noqa: E731

    # ----- geometry -------------------------------------------------------------------
    SEAT_Z = c.seat_z                        # 0.3875 — seated can centre
    WALL_TOP = c.cup_floor_z + c.wall_h      # 0.362 — collar rim
    HAND_GRASP = SEAT_Z + FINGER_LEN         # hand z when fingertips are at can centre
    HOVER_HAND = 0.60
    CARRY_HAND = 0.645                       # can bottom ~0.485 > every top (0.435)
    PLACE_CENTER = WALL_TOP + 0.008 + c.can_h / 2  # release: can bottom 8 mm above rim

    def can_pos(i):
        return scene.cans[i].data.root_pos_w[0]

    def pad_xy(k):
        return origin[:2] + scene.pad_c[0, k]

    # ----- low-level servo (pen_holder lineage) ----------------------------------------
    def ee_pose():
        return art.data.body_pos_w[0, ee_idx], art.data.body_quat_w[0, ee_idx]

    def width():
        return 2.0 * art.data.joint_pos[0, fj1].item()

    def servo(goal_pos, goal_quat, grip, a):
        p, q = ee_pose()
        err = goal_pos - p
        err = torch.cat([err[:2] * 1.6, err[2:3]])
        a[0, 0:3] = (err / osc.cfg.pos_scale).clamp(-1.0, 1.0)
        qe = quat_mul(goal_quat.unsqueeze(0), quat_conjugate(q.unsqueeze(0)))
        a[0, 3:6] = (axis_angle_from_quat(qe)[0] / osc.cfg.rot_scale).clamp(-1.0, 1.0)
        a[0, 6:8] = grip

    def jaw_quat(azimuth: float) -> torch.Tensor:
        yh = V3(math.cos(azimuth), math.sin(azimuth), 0.0)
        zh = V3(0.0, 0.0, -1.0)
        xh = torch.cross(yh, zh, dim=0)
        return quat_from_matrix(torch.stack([xh, yh, zh], dim=1).unsqueeze(0))[0]

    DOWN_Q = jaw_quat(0.0)  # fingers open along world x (row runs along y)

    sim_t = {"t": 0.0}

    tick_n = {"n": 0}

    def log_state():
        row = [sim_t["t"]]
        hp = art.data.body_pos_w[0, ee_idx]
        row += [hp[0].item(), hp[1].item(), hp[2].item(),
                2.0 * art.data.joint_pos[0, fj1].item()]
        for b in scene.cans:
            row += b.data.root_pos_w[0].tolist() + b.data.root_quat_w[0].tolist()
        row.append(float(scene.score()[0].item()))
        log_rows.append(row)

    def tick(a):
        do_render = args.video and tick_n["n"] % 2 == 0
        env.step(a, render=do_render)
        if do_render and cam is not None:
            try:
                rgb = cam.data.output.get("rgb")
                if rgb is not None and rgb.ndim == 4 and rgb.shape[0] >= 1:
                    frames.append(rgb[0, :, :, :3].detach().cpu().numpy().copy())
            except Exception:  # noqa: BLE001 — video is best-effort, never fail the solve
                pass
        if tick_n["n"] % 3 == 0:
            log_state()
        tick_n["n"] += 1
        sim_t["t"] += env.dt * robot.control_period

    def hold(pos, quat, grip, n_ticks):
        for _ in range(n_ticks):
            a = torch.zeros(1, n_act, device=device)
            servo(pos, quat, grip, a)
            tick(a)

    def run_phase(goal_fn, gate_fn, grip, timeout_s, tag=""):
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
            print(f"[diag] {tag}: ee=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
                  f"goal=({gp[0]:.3f},{gp[1]:.3f},{gp[2]:.3f}) "
                  f"err={float((gp - p).norm()) * 1000:.0f}mm", flush=True)
        return False

    def dewind() -> None:
        q = art.data.joint_pos[0]
        if abs(q[0].item()) > 2.6 or q[3].item() < -2.95 or q[3].item() > -0.15 \
                or abs(q[6].item()) > 2.6:
            print("[solve] wound arm — joint reset", flush=True)
            robot.reset(torch.tensor([0], device=device, dtype=torch.long))
            for _ in range(SEC(0.5)):
                a = torch.zeros(1, n_act, device=device)
                p, qq = ee_pose()
                servo(p, qq, OPEN, a)
                tick(a)

    # (per-episode plan + execution live in run_episode below)

    # ----- one pick-transit-seat move ---------------------------------------------------
    def do_move(i: int, k: int) -> bool:
        dest = pad_xy(k)
        for attempt in range(3):
            if bool(scene.dropped[0]):
                return False
            dewind()
            cp0 = can_pos(i)
            print(f"[solve] move can{i} -> pad{k} attempt {attempt}: "
                  f"can at ({cp0[0]:.3f},{cp0[1]:.3f},{cp0[2]:.3f})", flush=True)

            # HOVER above the can
            run_phase(lambda: (V3(can_pos(i)[0], can_pos(i)[1], HOVER_HAND), DOWN_Q),
                      lambda: (ee_pose()[0][:2] - can_pos(i)[:2]).norm() < 0.006
                      and abs(ee_pose()[0][2] - HOVER_HAND) < 0.02,
                      OPEN, 8.0, tag=f"hover{i}")

            # DESCEND to grasp height
            def desc_goal():
                cp = can_pos(i)
                return V3(cp[0], cp[1], cp[2] + FINGER_LEN), DOWN_Q

            ok = run_phase(desc_goal,
                           lambda: (ee_pose()[0][:2] - can_pos(i)[:2]).norm() < 0.004
                           and abs(ee_pose()[0][2] - (can_pos(i)[2] + FINGER_LEN)) < 0.005,
                           OPEN, 8.0, tag=f"descend{i}")
            if not ok:
                p, q = ee_pose()
                hold(V3(p[0], p[1], HOVER_HAND), DOWN_Q, OPEN, SEC(1.0))
                continue

            # CLOSE
            p, _ = ee_pose()
            hold(p, DOWN_Q, CLOSED, SEC(1.0))
            w = width()
            if not (0.048 < w < 0.070):
                print(f"[solve] bad grasp width {w * 1000:.0f}mm — retry", flush=True)
                hold(p, DOWN_Q, OPEN, SEC(0.5))
                hold(V3(p[0], p[1], HOVER_HAND), DOWN_Q, OPEN, SEC(1.0))
                continue

            # LIFT straight up
            gx, gy = p[0].item(), p[1].item()
            run_phase(lambda: (V3(gx, gy, CARRY_HAND), DOWN_Q),
                      lambda: can_pos(i)[2] > 0.52,
                      CLOSED, 6.0, tag=f"lift{i}")
            if can_pos(i)[2] < 0.50 or not (0.048 < width() < 0.070):
                print(f"[solve] lift failed (can z={can_pos(i)[2]:.3f} w={width()*1000:.0f}mm)",
                      flush=True)
                p, _ = ee_pose()
                hold(p, DOWN_Q, OPEN, SEC(0.5))
                hold(V3(p[0], p[1], HOVER_HAND), DOWN_Q, OPEN, SEC(1.0))
                continue

            # CARRY: servo the CAN xy onto the destination cup axis
            def carry_goal():
                cp = can_pos(i)
                hp = ee_pose()[0]
                return V3(hp[0] + (dest[0] - cp[0]), hp[1] + (dest[1] - cp[1]),
                          CARRY_HAND), DOWN_Q

            run_phase(carry_goal,
                      lambda: (can_pos(i)[:2] - dest).norm() < 0.004
                      and can_pos(i)[2] > 0.52,
                      CLOSED, 12.0, tag=f"carry{i}")
            if not (0.048 < width() < 0.070):
                print("[solve] can slipped in transit", flush=True)
                continue

            # LOWER: can bottom to just above the rim, xy stays gated
            def lower_goal():
                cp = can_pos(i)
                hp = ee_pose()[0]
                return V3(hp[0] + (dest[0] - cp[0]), hp[1] + (dest[1] - cp[1]),
                          hp[2] + (PLACE_CENTER - cp[2])), DOWN_Q

            ok = run_phase(lower_goal,
                           lambda: abs(can_pos(i)[2] - PLACE_CENTER) < 0.004
                           and (can_pos(i)[:2] - dest).norm() < 0.005,
                           CLOSED, 10.0, tag=f"lower{i}")
            if not ok:
                # try to re-lift and re-align once more within this attempt
                p, _ = ee_pose()
                hold(V3(p[0], p[1], CARRY_HAND), DOWN_Q, CLOSED, SEC(1.5))
                continue

            # RELEASE + retreat
            p, _ = ee_pose()
            hold(p, DOWN_Q, OPEN, SEC(0.6))
            hold(V3(p[0], p[1], HOVER_HAND), DOWN_Q, OPEN, SEC(1.2))

            # SETTLE: poll the scene's own seat predicate
            deadline = sim_t["t"] + 4.0
            seated = False
            while sim_t["t"] < deadline:
                pp, qq = ee_pose()
                hold(pp, DOWN_Q, OPEN, 1)
                if bool((scene._seated_on(k) & scene.settled())[0, i]):
                    seated = True
                    break
            cp = can_pos(i)
            print(f"[solve] move can{i}->pad{k}: seated={seated} "
                  f"can=({cp[0]:.3f},{cp[1]:.3f},{cp[2]:.3f}) dropped={bool(scene.dropped[0])}",
                  flush=True)
            if seated:
                return True
            if bool(scene.dropped[0]):
                return False
            # not seated — if the can is still up somewhere retryable, loop re-picks it
        return False

    def run_episode(ep: int) -> bool:
        t_ep = sim_t["t"]
        # settle after reset
        p0, q0 = ee_pose()
        hold(p0, q0, OPEN, SEC(1.0))

        # plan: cycle decomposition through the buffer (pad 3)
        sp = scene.start_pad[0].tolist()
        present = [i for i in range(3) if sp[i] >= 0]
        print(f"[solve] ep{ep}: derangement start_pad={sp} present={present}", flush=True)
        a0 = present[0]
        moves = [(a0, 3)]
        cup = sp[a0]
        while cup != a0:
            moves.append((cup, cup))     # can `cup` goes home into the just-freed cup
            cup = sp[cup]
        moves.append((a0, a0))           # retrieve the parked can from the buffer
        print(f"[solve] ep{ep}: planned moves (can -> pad): {moves} "
              f"({len(moves)} = n_present+1)", flush=True)

        for (i, k) in moves:
            if sim_t["t"] - t_ep > args.max_sec:
                print("[solve] sim-time budget exhausted", flush=True)
                break
            if not do_move(i, k):
                print(f"[solve] MOVE FAILED can{i}->pad{k}", flush=True)
                break

        # park and let everything settle
        dewind()
        p, q = ee_pose()
        hold(V3(p[0], p[1], HOVER_HAND), DOWN_Q, OPEN, SEC(1.0))
        for _ in range(SEC(2.0)):
            pp, _ = ee_pose()
            hold(pp, DOWN_Q, OPEN, 1)

        succ = bool(scene.success()[0].item())
        score = float(scene.score()[0].item())
        print(f"[readout] ep{ep}: correct={scene.correct()[0].tolist()} "
              f"present={scene.present[0].tolist()} dropped={bool(scene.dropped[0])} "
              f"lifted={bool(scene.lifted[0])} buffer_used={bool(scene.buffer_used[0])}",
              flush=True)
        print(f"[readout] ep{ep}: success={succ} score={score:.3f}", flush=True)
        return succ

    results = []
    for ep in range(args.episodes):
        if ep > 0:
            env.reset()
        log_pads.append((origin[:2] + scene.pad_c[0]).cpu().numpy().copy())
        results.append(run_episode(ep))

    try:
        import numpy as np
        np.savez("simgen_tasks/musical_cans/solve_log.npz",
                 log=np.array(log_rows, dtype=np.float32),
                 pads=np.array(log_pads, dtype=np.float32))
        print(f"[solve] state log: simgen_tasks/musical_cans/solve_log.npz "
              f"({len(log_rows)} rows)", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[solve] log save failed: {exc!r}", flush=True)
    if args.video and frames:
        try:
            import numpy as np
            import imageio.v2 as iio
            arr = [f.astype("uint8") for f in frames]
            iio.mimsave(args.video_path, arr, fps=15, macro_block_size=1)
            print(f"[solve] video: {args.video_path} ({len(arr)} frames)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[solve] video save failed: {exc!r}", flush=True)

    print(f"[solve] episode successes: {results}", flush=True)
    print("RESULT: SUCCESS" if all(results) else "RESULT: FAIL", flush=True)

    import os
    os._exit(0)  # skip Kit teardown (it hangs headless); results are printed above


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        import os
        import traceback
        traceback.print_exc()
        print("RESULT: FAIL", flush=True)
        os._exit(1)

"""Scripted plug-insertion controller for Isaac-AutoMate-Assembly-Direct-v0.

Inserts the held plug (asset 00015: 8 mm peg) into its socket with the Franka robot
using a vectorized state machine on top of the env's task-space impedance controller:

    ALIGN  -> hover the plug above the socket and null XY error
    INSERT -> rate-limited compliant descent down the socket axis
              - spiral search if descent stalls with small XY error (rim catch)
              - retry from ALIGN if descent stalls with large XY error (grasp slip)
    SEAT   -> press to the seated height
    HOLD   -> keep position, latch success

The env action space (6,) is interpreted as fingertip pose deltas:
pos action * 0.1 m, rot action * 0.01 rad (roll/pitch forced upright, gripper closed),
EMA-smoothed with factor 0.2. Since the plug is rigidly grasped, commanding the
fingertip by the plug's position error servos the plug directly. The outer-loop gain
must stay < 1: the inner loop is a lagged position servo, and gain > 1 produces a
divergent limit cycle (observed: XY error oscillating 5 -> 45 mm, plug slipping in
the fingers).

Success uses the env's own criterion (automate_algo.check_plug_inserted_in_socket):
socket_z < plug_z < socket_z + disassembly_dist  AND  mean keypoint dist < 1.5 cm.

NOTE: never change env_cfg.sim.render_interval here — it sets rendering_dt, and the
env's reset loops (IK / gripper close) call sim.step(render=True), which advances
rendering_dt worth of physics per call. Changing it breaks the scripted grasp.

Run (records a video into Fable_5/AutoMate-Assembly-Direct-v0/videos):
    env_isaaclab/bin/python -u Fable_5/AutoMate-Assembly-Direct-v0/insert_plug.py --num_envs 4
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted AutoMate plug insertion.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_steps", type=int, default=110, help="Max policy steps (episode times out at 150).")
parser.add_argument("--hold_steps", type=int, default=20, help="Steps to hold after success before stopping.")
parser.add_argument("--video_name", type=str, default="plug_insertion")
parser.add_argument("--no_video", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
if not args_cli.no_video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.automate import automate_algo_utils as automate_algo
from isaaclab_tasks.utils import parse_env_cfg

TASK = "Isaac-AutoMate-Assembly-Direct-v0"
FABLE_DIR = os.path.dirname(os.path.abspath(__file__))

# controller constants (meters / policy steps @ 15 Hz)
ACTION_GAIN = 0.5  # outer-loop P gain on plug position error; MUST be < 1 (see module docstring)
ACTION_CAP = 0.3  # cap on |action| -> max 3 cm fingertip target offset
HOVER_ABOVE = 0.005  # hover height above the socket mouth during ALIGN
ALIGN_XY_TOL = 0.0015  # XY error to finish ALIGN
ALIGN_Z_TOL = 0.004  # Z error to finish ALIGN
DESCEND_RATE = 0.0010  # max target descent per step (~1.5 cm/s)
SEAT_OFFSET = 0.002  # target plug z above socket origin when seated (success needs z > socket_z)
STALL_WINDOW = 12  # steps without descent progress before stall handling kicks in
STALL_EPS = 0.0005  # required descent progress (m) over the window
SPIRAL_RADIUS = 0.0012  # max spiral search radius (clearance is ~0.11 mm; chamfer funnels ~1 mm)
SPIRAL_PERIOD = 10.0  # steps per spiral revolution
SPIRAL_MAX_T = 28  # spiral steps before giving up -> retry from ALIGN
RETRY_XY = 0.0025  # stall with XY error above this -> grasp slipped -> retry from ALIGN

ALIGN, INSERT, SEAT, HOLD = 0, 1, 2, 3
PHASE_NAMES = {ALIGN: "ALIGN", INSERT: "INSERT", SEAT: "SEAT", HOLD: "HOLD"}


class InsertionController:
    """Vectorized state machine that outputs env actions from privileged state."""

    def __init__(self, uenv):
        self.uenv = uenv
        n, dev = uenv.num_envs, uenv.device
        self.phase = torch.full((n,), ALIGN, dtype=torch.long, device=dev)
        self.z_target = uenv.held_pos[:, 2].clone()
        self.stall_anchor_z = uenv.held_pos[:, 2].clone()
        self.stall_count = torch.zeros(n, device=dev)
        self.spiral_t = torch.zeros(n, device=dev)
        self.retries = torch.zeros(n, dtype=torch.long, device=dev)

    def compute_action(self, successes: torch.Tensor) -> torch.Tensor:
        u = self.uenv
        n, dev = u.num_envs, u.device
        plug = u.held_pos  # (n, 3), env-local frame
        socket = u.fixed_pos
        hover_z = socket[:, 2] + u.disassembly_dists + HOVER_ABOVE
        seat_z = socket[:, 2] + SEAT_OFFSET
        xy_err = torch.norm(socket[:, :2] - plug[:, :2], dim=-1)

        # ---------- stall bookkeeping (INSERT only) ----------
        in_insert = self.phase == INSERT
        self.stall_count = torch.where(in_insert, self.stall_count + 1, torch.zeros_like(self.stall_count))
        window_done = self.stall_count >= STALL_WINDOW
        progressed = (self.stall_anchor_z - plug[:, 2]) > STALL_EPS
        stalled = in_insert & window_done & ~progressed
        # refresh window
        self.stall_anchor_z = torch.where(window_done, plug[:, 2], self.stall_anchor_z)
        self.stall_count = torch.where(window_done, torch.zeros_like(self.stall_count), self.stall_count)
        # spiral lifecycle: start on small-offset stall, advance while active, stop on progress
        start_spiral = stalled & (xy_err <= RETRY_XY) & (self.spiral_t == 0)
        self.spiral_t = torch.where(self.spiral_t > 0, self.spiral_t + 1, self.spiral_t)
        self.spiral_t = torch.where(start_spiral, torch.ones_like(self.spiral_t), self.spiral_t)
        self.spiral_t = torch.where(progressed & window_done, torch.zeros_like(self.spiral_t), self.spiral_t)

        # ---------- transitions ----------
        # INSERT -> ALIGN retry: grasp slipped sideways, or spiral exhausted
        retry = stalled & ((xy_err > RETRY_XY) | (self.spiral_t > SPIRAL_MAX_T))
        self.phase = torch.where(retry, torch.full_like(self.phase, ALIGN), self.phase)
        self.spiral_t = torch.where(retry, torch.zeros_like(self.spiral_t), self.spiral_t)
        self.retries += retry.long()

        # ALIGN -> INSERT: hovering on axis
        aligned = (
            (self.phase == ALIGN) & (xy_err < ALIGN_XY_TOL) & ((plug[:, 2] - hover_z).abs() < ALIGN_Z_TOL)
        )
        self.z_target = torch.where(aligned, plug[:, 2], self.z_target)
        self.stall_anchor_z = torch.where(aligned, plug[:, 2], self.stall_anchor_z)
        self.stall_count = torch.where(aligned, torch.zeros_like(self.stall_count), self.stall_count)
        self.spiral_t = torch.where(aligned, torch.zeros_like(self.spiral_t), self.spiral_t)
        self.phase = torch.where(aligned, torch.full_like(self.phase, INSERT), self.phase)

        # INSERT -> SEAT: nearly seated
        near_seat = (self.phase == INSERT) & (plug[:, 2] < seat_z + 0.004)
        self.phase = torch.where(near_seat, torch.full_like(self.phase, SEAT), self.phase)

        # any -> HOLD on success
        self.phase = torch.where(successes.bool(), torch.full_like(self.phase, HOLD), self.phase)

        # ---------- per-phase plug position targets ----------
        in_insert = self.phase == INSERT  # recompute after transitions
        target = torch.empty_like(plug)
        target[:, 0] = socket[:, 0]
        target[:, 1] = socket[:, 1]
        target[:, 2] = hover_z  # ALIGN default

        # INSERT: ratchet z target down, tracking-aware
        new_z = torch.maximum(torch.minimum(self.z_target, plug[:, 2]) - DESCEND_RATE, seat_z)
        self.z_target = torch.where(in_insert, new_z, self.z_target)
        target[:, 2] = torch.where(in_insert, self.z_target, target[:, 2])

        # spiral search offset while jammed (INSERT only)
        ang = 2.0 * math.pi * self.spiral_t / SPIRAL_PERIOD
        rad = torch.clamp(self.spiral_t / (2.0 * SPIRAL_PERIOD), max=1.0) * SPIRAL_RADIUS
        active_spiral = in_insert & (self.spiral_t > 0)
        target[:, 0] = torch.where(active_spiral, target[:, 0] + rad * torch.cos(ang), target[:, 0])
        target[:, 1] = torch.where(active_spiral, target[:, 1] + rad * torch.sin(ang), target[:, 1])

        # SEAT: press to seated height on axis
        target[:, 2] = torch.where(self.phase == SEAT, seat_z, target[:, 2])

        # HOLD: stay put
        target = torch.where((self.phase == HOLD).unsqueeze(-1), plug, target)

        # ---------- plug position error -> fingertip delta action ----------
        action = torch.zeros((n, 6), device=dev)
        action[:, :3] = torch.clamp(ACTION_GAIN * (target - plug) / u.pos_threshold, -ACTION_CAP, ACTION_CAP)
        return action


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    # eval-style hard start: plug always begins fully outside the socket (+ XY noise)
    env_cfg.tasks["insertion"].if_sbc = False
    # more time than the default 5 s episode; we stop early on success anyway
    env_cfg.episode_length_s = 10.0
    # camera: aimed precisely at env 0's socket after reset (origin_type env keeps it relative)
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.resolution = (1280, 720)

    render_mode = None if args_cli.no_video else "rgb_array"
    env = gym.make(TASK, cfg=env_cfg, render_mode=render_mode)
    if not args_cli.no_video:
        video_kwargs = {
            "video_folder": os.path.join(FABLE_DIR, "videos"),
            "name_prefix": args_cli.video_name,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.max_steps + args_cli.hold_steps,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    uenv = env.unwrapped
    env.reset()
    print(f"[CTRL] reset done. disassembly_dist={uenv.disassembly_dists[0].item():.4f} m", flush=True)
    print(f"[CTRL] socket pos (env 0): {uenv.fixed_pos[0].cpu().numpy()}", flush=True)
    print(f"[CTRL] plug pos   (env 0): {uenv.held_pos[0].cpu().numpy()}", flush=True)

    # frame env 0's socket close-up using its actual (randomized) position
    if uenv.viewport_camera_controller is not None:
        sock = uenv.fixed_pos[0].cpu().numpy()
        eye = (float(sock[0]) + 0.24, float(sock[1]) - 0.21, float(sock[2]) + 0.13)
        lookat = (float(sock[0]), float(sock[1]), float(sock[2]) + 0.025)
        uenv.viewport_camera_controller.update_view_location(eye=eye, lookat=lookat)

    ctrl = InsertionController(uenv)
    successes = torch.zeros(uenv.num_envs, dtype=torch.bool, device=uenv.device)
    hold_counter = 0

    for step in range(args_cli.max_steps + args_cli.hold_steps):
        with torch.inference_mode():
            action = ctrl.compute_action(successes)
            env.step(action)

            curr_success = automate_algo.check_plug_inserted_in_socket(
                uenv.held_pos,
                uenv.fixed_pos,
                uenv.disassembly_dists,
                uenv.keypoints_held,
                uenv.keypoints_fixed,
                uenv.cfg_task.close_error_thresh,
                uenv.episode_length_buf,
            )
        successes = successes | curr_success.bool()

        if step % 10 == 0 or bool(successes.all()):
            plug, socket = uenv.held_pos, uenv.fixed_pos
            xy_mm = (torch.norm(socket[:, :2] - plug[:, :2], dim=-1) * 1e3).cpu().numpy()
            z_mm = ((plug[:, 2] - socket[:, 2]) * 1e3).cpu().numpy()
            kp_mm = (uenv.keypoint_dist * 1e3).cpu().numpy()
            phases = [PHASE_NAMES[int(p)] for p in ctrl.phase.cpu()]
            print(
                f"[CTRL] step {step:3d} | xy_err(mm) {xy_mm.round(2)} | z_above(mm) {z_mm.round(1)} "
                f"| kp_dist(mm) {kp_mm.round(1)} | phase {phases} | success {successes.cpu().numpy()}",
                flush=True,
            )

        if successes.all():
            hold_counter += 1
            if hold_counter >= args_cli.hold_steps:
                break

    n_success = int(successes.sum())
    print(f"[CTRL] RESULT: {n_success}/{uenv.num_envs} envs inserted successfully "
          f"(success env ids: {successes.nonzero().squeeze(-1).cpu().tolist()}, "
          f"retries: {ctrl.retries.cpu().tolist()})", flush=True)
    print(f"[CTRL] env-internal latched success: {uenv.ep_succeeded.cpu().numpy()}", flush=True)

    env.close()
    if not args_cli.no_video:
        print(f"[CTRL] video saved under: {os.path.join(FABLE_DIR, 'videos')}", flush=True)
    return n_success


if __name__ == "__main__":
    main()
    simulation_app.close()

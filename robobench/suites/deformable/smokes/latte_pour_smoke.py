"""Pouring-suite smoke — the two Frankas LIFT both vessels to validate the simulation.

This is a capability check, not a solution: on the same registered env the pouring benchmark
uses (`deformable.latte.bimanual_franka.joint` — dynamic arms + dynamic vessels + the auto-weld
grasp contract + 1.5-way liquid feedback), each arm reaches its vessel's handle bar, closes
onto it (the scene's auto-weld engages, exactly as it would for an agent), lifts the vessel
clear of the table with its liquid aboard, holds, sets it back down, and releases. A PASS
demonstrates the pieces an actual pour needs: IK-driven reach to both handles, the grasp
contract engaging/releasing, vessels carrying their real (liquid-loaded) mass, and stable
coupled MPM dynamics through grasp/carry/release.

Verdict: during the hold BOTH vessels must have risen >= 30 mm, and at the end both stand
upright (tilt <= 5 deg) with the coffee retained (>= 0.99) and nothing spilled (<= 0.01).

Runs ONLY under the Newton venv. Do NOT record live (live rendering corrupts the coupled MPM
physics) — use `--dump_states` + the offline replay renderer:
  env_newton/bin/python -m robobench.suites.deformable.smokes.latte_pour_smoke --headless
"""

from __future__ import annotations

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1, help="MPM grid is scene-wide; keep 1")
parser.add_argument("--lift", type=float, default=0.06, help="vessel lift height [m]")
parser.add_argument("--hold", type=float, default=1.5, help="hold time at the top of the lift [s]")
parser.add_argument("--grasp_pitch", type=float, default=30.0, help="downward tilt of the horizontal side grasps [deg]")
parser.add_argument("--print_every", type=int, default=400, help="progress print period [steps]")
parser.add_argument("--max_steps", type=int, default=None, help="cap total steps (debugging)")
parser.add_argument("--dump_states", type=str, default=None,
                    help="record body_q + particle positions every --dump_every steps into this .npz "
                         "for offline replay rendering (LIVE rendering corrupts the coupled physics)")
parser.add_argument("--dump_every", type=int, default=7, help="state-dump cadence [steps]; 7 ~= 30 fps at 200 Hz")
AppLauncher.add_app_launcher_args(parser)
if "--enable_cameras" in sys.argv:
    sys.argv = [a for a in sys.argv if a != "--headless"]
    parser.set_defaults(headless=True, visualizer=["kit"])
args = parser.parse_args()
render_on = (args.livestream > 0) or bool(args.visualizer and "none" not in args.visualizer)

app = AppLauncher(args).app

# Disable the cubric GPU transform hierarchy (isaacsim 6.0.0.1 IAdapter version drift renders
# moving prims frozen/detached); the CPU update_world_xforms() fallback renders correctly.
from isaaclab_newton.physics import newton_manager as _nm  # noqa: E402


def _no_cubric(cls) -> None:
    cls._cubric = None


_nm.NewtonManager._setup_cubric_bindings = classmethod(_no_cubric)

import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.controllers.differential_ik import DifferentialIKController  # noqa: E402
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

FPS = 200  # must match MpmSimCfg.dt
MAX_DQ = 0.04  # per-tick joint reference step clamp [rad]
LEAD_MAX = 0.30  # max lead of the commanded reference over the ACTUAL joints [rad]
GRIP_OPEN = 0.04
# Two-stage close (cf. the grasp contract): pads stand off while the vessel is free, then
# finish toward the bar surface once the weld carries the load.
STANDOFF_MUG, GRIP_MUG_BAR = 0.011, 0.0085  # mug bar capsule r = 0.008
STANDOFF_PITCHER, GRIP_PITCHER_BAR = 0.009, 0.0065  # pitcher bar capsule r = 0.006


class Arm:
    """Per-arm DiffIK servo toward a world-frame hand pose; returns the 9-wide child action."""

    def __init__(self, robot, device: str) -> None:
        art = robot.articulation
        self.art = art
        self.device = device
        self.arm_ids = art.find_joints(["panda_joint[1-7]"])[0]
        self.hand_idx = art.find_bodies(["panda_hand"])[0][0]
        num_base = getattr(art, "num_base_dofs", 0)
        self.jacobi_body = self.hand_idx - 1 if art.is_fixed_base else self.hand_idx
        self.jacobi_joints = [j + num_base for j in self.arm_ids]
        self.grip = GRIP_OPEN
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": 0.05}
            ),
            num_envs=1,
            device=device,
        )
        self.limits = art.data.joint_pos_limits.torch[:, self.arm_ids, :]
        self.q_ref: torch.Tensor | None = None

    def hand_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.art.data.body_pos_w.torch[:, self.hand_idx], self.art.data.body_quat_w.torch[:, self.hand_idx]

    def solve(self, target_pos_w: torch.Tensor, target_quat_w: torch.Tensor) -> torch.Tensor:
        tp, tq = math_utils.subtract_frame_transforms(
            self.art.data.root_pos_w.torch, self.art.data.root_quat_w.torch, target_pos_w, target_quat_w
        )
        ee_p, ee_q = math_utils.subtract_frame_transforms(
            self.art.data.root_pos_w.torch, self.art.data.root_quat_w.torch, *self.hand_pose_w()
        )
        self.ik.set_command(torch.cat([tp, tq], dim=-1), ee_p, ee_q)
        J = self.art.data.body_link_jacobian_w.torch[:, self.jacobi_body, :, self.jacobi_joints]
        R = math_utils.matrix_from_quat(math_utils.quat_inv(self.art.data.root_quat_w.torch))
        J[:, :3, :] = torch.bmm(R, J[:, :3, :])
        J[:, 3:, :] = torch.bmm(R, J[:, 3:, :])
        q_arm = self.art.data.joint_pos.torch[:, self.arm_ids]
        q_des = self.ik.compute(ee_p, ee_q, J, q_arm)
        if self.q_ref is None:
            self.q_ref = q_arm.clone()
        q_ref = (self.q_ref + (q_des - self.q_ref).clamp(-MAX_DQ, MAX_DQ)).clamp(
            self.limits[..., 0], self.limits[..., 1]
        )
        q_ref = q_ref.clamp(q_arm - LEAD_MAX, q_arm + LEAD_MAX)
        self.q_ref = q_ref
        grip = torch.full((1, 2), self.grip, device=self.device)
        return torch.cat([q_ref, grip], dim=-1)


def _smoothstep(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


def _tilt_deg(pose_w: torch.Tensor) -> float:
    """Tilt of a vessel's +z axis off vertical [deg] from its (n, 7) world pose."""
    q = pose_w[0, 3:]
    x, y = float(q[0]), float(q[1])
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return math.degrees(math.acos(max(-1.0, min(1.0, up_z))))


class _Done(Exception):
    pass


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("deformable.latte.bimanual_franka.joint")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    env.reset()
    from robobench.suites.deformable.newton.coupled_manager import NewtonCoupledMJWarpMPMManager as Mgr

    Mgr.resync_collider_history()
    if render_on:
        scene.setup_particle_visuals()
    print(
        f"[lift] particles: coffee {scene.coffee.particles_per_object}, milk {scene.milk.particles_per_object}"
        f" | env: the benchmark (coupled + dynamic vessels + auto-weld + liquid feedback)",
        flush=True,
    )

    left, right = Arm(env.robot.robots["left"], device), Arm(env.robot.robots["right"], device)
    origin = env.iscene.env_origins[0]

    def t3(x: float, y: float, z: float) -> torch.Tensor:
        return torch.tensor([[x, y, z]], device=device) + origin

    def q_ry(deg: float) -> torch.Tensor:
        half = math.radians(deg) / 2.0
        return torch.tensor([[0.0, math.sin(half), 0.0, math.cos(half)]], device=device)

    Q_LEFT = q_ry(90.0 + args.grasp_pitch)
    Q_RIGHT = q_ry(-(90.0 + args.grasp_pitch))
    z_hat = torch.tensor([[0.0, 0.0, 1.0]], device=device)

    def hand_from_tip(tip_xyz: tuple, quat: torch.Tensor) -> torch.Tensor:
        return t3(*tip_xyz) - 0.113 * math_utils.quat_apply(quat, z_hat)

    # Pinch points at the handle bars (the bars' measured world spots).
    px, py = c.pitcher_pos
    lh_grasp = hand_from_tip((-0.080, 0.0, 0.093), Q_LEFT)
    rh_grasp = hand_from_tip((px + 0.060, py, 0.095), Q_RIGHT)
    lh_hover = lh_grasp + torch.tensor([-0.06, 0.0, 0.05], device=device)
    rh_hover = rh_grasp + torch.tensor([0.07, 0.0, 0.05], device=device)
    up = torch.tensor([0.0, 0.0, args.lift], device=device)

    tick = 0
    dump_frames: dict[str, list] = {"body_q": [], "coffee": [], "milk": []}
    lt: torch.Tensor | None = None
    rt: torch.Tensor | None = None

    def step() -> None:
        nonlocal tick
        action = torch.cat([left.solve(lt, Q_LEFT), right.solve(rt, Q_RIGHT)], dim=-1)
        env.step(action, render=render_on)
        tick += 1
        if render_on and tick % 5 == 0:
            scene.push_particle_visuals()
        if args.dump_states and tick % args.dump_every == 0:
            import numpy as _np
            import warp as _wp

            dump_frames["body_q"].append(
                _wp.to_torch(Mgr._state_0.body_q).detach().cpu().numpy().astype(_np.float32))
            dump_frames["coffee"].append(
                scene.coffee.data.nodal_pos_w.torch[0].detach().cpu().numpy().astype(_np.float16))
            dump_frames["milk"].append(
                scene.milk.data.nodal_pos_w.torch[0].detach().cpu().numpy().astype(_np.float16))
        if tick % args.print_every == 0:
            mz = float(scene.mug_pose_w[0, 2] - origin[2])
            pz = float(scene.pitcher_pose_w[0, 2] - origin[2])
            print(f"  step {tick:5d} t={tick / FPS:5.1f}s | mug z {mz:.3f} | pitcher z {pz:.3f}", flush=True)
        if args.max_steps is not None and tick >= args.max_steps:
            raise _Done

    def glide(fr_l: torch.Tensor, to_l: torch.Tensor, fr_r: torch.Tensor, to_r: torch.Tensor, dur: float):
        nonlocal lt, rt
        steps = max(1, int(round(dur * FPS)))
        for k in range(steps):
            s = _smoothstep((k + 1) / steps)
            lt = fr_l + (to_l - fr_l) * s
            rt = fr_r + (to_r - fr_r) * s
            step()
        return to_l, to_r

    def dwell(dur: float) -> None:
        for _ in range(int(round(dur * FPS))):
            step()

    mug_z0 = float(scene.mug_pose_w[0, 2] - origin[2])
    pit_z0 = float(scene.pitcher_pose_w[0, 2] - origin[2])
    rise = {"mug": 0.0, "pitcher": 0.0}

    try:
        lp0, _ = left.hand_pose_w()
        rp0, _ = right.hand_pose_w()
        lt, rt = lp0.clone(), rp0.clone()
        dwell(0.5)

        lcur, rcur = glide(lp0, lh_hover, rp0, rh_hover, 2.5)     # reach
        lcur, rcur = glide(lcur, lh_grasp, rcur, rh_grasp, 1.5)   # descend to the bars
        left.grip, right.grip = STANDOFF_MUG, STANDOFF_PITCHER    # close -> auto-weld engages
        dwell(1.0)
        left.grip, right.grip = GRIP_MUG_BAR, GRIP_PITCHER_BAR    # finish onto the bars
        lcur, rcur = glide(lcur, lh_grasp + up, rcur, rh_grasp + up, 1.5)  # lift both vessels
        for _ in range(int(round(args.hold * FPS))):
            step()
            rise["mug"] = max(rise["mug"], float(scene.mug_pose_w[0, 2] - origin[2]) - mug_z0)
            rise["pitcher"] = max(rise["pitcher"], float(scene.pitcher_pose_w[0, 2] - origin[2]) - pit_z0)
        print(f"  [lift] at hold: mug rose {rise['mug'] * 1e3:.0f} mm, pitcher rose "
              f"{rise['pitcher'] * 1e3:.0f} mm", flush=True)
        lcur, rcur = glide(lcur, lh_grasp, rcur, rh_grasp, 1.5)   # set back down
        left.grip = right.grip = GRIP_OPEN                        # open -> welds release
        dwell(1.0)
        lcur, rcur = glide(lcur, lh_hover, rcur, rh_hover, 1.2)   # retreat clear
        dwell(1.5)
    except _Done:
        print(f"  [cap] stopped at step {tick}", flush=True)

    retention = float(scene.retention_fraction().mean())
    spilled = float(scene.spilled_fraction().mean())
    mug_tilt = _tilt_deg(scene.mug_pose_w)
    pit_tilt = _tilt_deg(scene.pitcher_pose_w)
    ok = (rise["mug"] >= 0.03 and rise["pitcher"] >= 0.03 and mug_tilt <= 5.0 and pit_tilt <= 5.0
          and retention >= 0.99 and spilled <= 0.01)
    print(
        f"LATTE-LIFT {'PASS' if ok else 'FAIL'} | mug rose {rise['mug'] * 1e3:.0f} mm, pitcher rose "
        f"{rise['pitcher'] * 1e3:.0f} mm (gate >= 30) | final tilt mug {mug_tilt:.1f} deg, pitcher "
        f"{pit_tilt:.1f} deg (<= 5) | retention {retention:.3f} (>= 0.99) | spilled {spilled:.3f} (<= 0.01)",
        flush=True,
    )
    if args.dump_states and dump_frames["body_q"]:
        import numpy as _np

        _np.savez_compressed(
            args.dump_states,
            body_q=_np.stack(dump_frames["body_q"]),
            coffee=_np.stack(dump_frames["coffee"]),
            milk=_np.stack(dump_frames["milk"]),
            fps=float(FPS) / float(args.dump_every),
        )
        print(f"[dump] {len(dump_frames['body_q'])} frames -> {args.dump_states}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()

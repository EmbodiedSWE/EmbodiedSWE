"""Folding-suite smoke — a Franka LIFTS the T-shirt to validate the simulation.

This is a capability check, not a solution: on the same registered env the folding benchmark
uses (`deformable.tshirt.franka.joint` — Franka + VBD cloth on the coupled MJWarp substrate), the
arm reaches over the shirt, pinches the fabric with its real fingers (pure contact — no weld,
no constrained particles), lifts it clear of the table, holds, lowers, and releases. A PASS
demonstrates the pieces an actual folding attempt needs: IK-driven reach in the cloth
workspace, a force pinch that holds fabric under load, and stable coupled cloth dynamics
through grasp/carry/release.

Verdict: during the hold the cloth's highest point must rise >= 60 mm above its settled height
(the pinch genuinely carried fabric up), and every particle must stay in bounds.

Runs ONLY under the Newton venv (isaaclab develop runs headless unless a kit visualizer is
requested; an explicit --headless force-disables visualizers, so don't combine it with --viz):
  env_newton/bin/python -m robobench.suites.deformable.smokes.tshirt_fold_smoke --headless
  env_newton/bin/python -m robobench.suites.deformable.smokes.tshirt_fold_smoke --viz kit
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1, help="the demo driver is single-env; keep 1")
parser.add_argument("--lift", type=float, default=0.18, help="fingertip lift height above the table top [m]")
parser.add_argument("--hold", type=float, default=1.5, help="hold time at the top of the lift [s]")
parser.add_argument("--print_every", type=int, default=120, help="progress print period [steps]")
parser.add_argument("--max_steps", type=int, default=None, help="cap total steps (debugging)")
AppLauncher.add_app_launcher_args(parser)
if "--enable_cameras" in sys.argv:
    # Recording path (scripts/record_video.py): rendering on develop is pumped by visualizers, and
    # an explicit --headless force-disables them — so make headless+kit-visualizer the DEFAULTS.
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

FPS = 60  # must match the scene's NewtonSimCfg.dt
TABLE_TOP_Z = 0.20  # box table top (see TshirtFoldingSceneCfg)
MAX_DQ = 0.04  # per-tick joint reference step clamp [rad]
LEAD_MAX = 0.30  # max lead of the commanded reference over the ACTUAL joints [rad]
OPEN, CLOSE = 0.8 * 0.04, 0.0005  # finger targets [m]: near-zero close clamps the thin sheet
Q_DOWN = (0.92388, 0.0, 0.38268, 0.0)  # xyzw: gripper-down quat for +x-half grasps — 180 deg
# about (cos22.5, 0, sin22.5): the 45 deg wrist tilt keeps the far sleeve inside the reach
# envelope (a straight-down pinch there stalls short)
TIP_OFFSET = (0.0, 0.0, 0.113)  # fingertip point in the panda_hand frame

BOUNDS_LO = (-0.45, -1.00, -0.05)
BOUNDS_HI = (0.45, 0.10, 0.60)


class Arm:
    """DiffIK servo toward a world-frame hand pose; returns the 9-wide joint action."""

    def __init__(self, robot, device: str) -> None:
        art = robot.articulation
        self.art = art
        self.device = device
        self.arm_ids = art.find_joints(["panda_joint[1-7]"])[0]
        self.hand_idx = art.find_bodies(["panda_hand"])[0][0]
        num_base = getattr(art, "num_base_dofs", 0)
        self.jacobi_body = self.hand_idx - 1 if art.is_fixed_base else self.hand_idx
        self.jacobi_joints = [j + num_base for j in self.arm_ids]
        self.grip = OPEN
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


class _Done(Exception):
    pass


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("deformable.tshirt.franka.joint")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    env.reset()
    arm = Arm(env.robot, device)
    origin = env.iscene.env_origins[0]
    print(f"[lift] cloth particles: {scene.cloth.data.nodal_pos_w.torch.shape[1]}", flush=True)

    q_down = torch.tensor([Q_DOWN], device=device)
    tip_off = torch.tensor([TIP_OFFSET], device=device)

    def hand_for_tip(tip_w: torch.Tensor) -> torch.Tensor:
        """Hand position putting the fingertip point at `tip_w` under the down-grasp quat."""
        return tip_w - math_utils.quat_apply(q_down, tip_off)

    def cloth_local() -> torch.Tensor:
        return scene.nodal_pos_local()[0]

    tick = 0
    action = torch.zeros((1, 9), device=device)
    target_p: torch.Tensor | None = None
    hold_peak = [0.0]
    z_rest = 0.0

    def step() -> None:
        nonlocal tick
        action[:] = arm.solve(target_p, q_down)
        env.step(action, render=render_on)
        tick += 1
        if tick % 30 == 0 and not bool(torch.isfinite(scene.cloth.data.nodal_pos_w.torch).all()):
            print(f"  ABORT: cloth state went non-finite at step {tick}", flush=True)
            raise _Done
        if tick % args.print_every == 0:
            zmax = float(cloth_local()[:, 2].max())
            print(f"  step {tick:5d} t={tick / FPS:5.1f}s | cloth z_max {zmax:.3f} m", flush=True)
        if args.max_steps is not None and tick >= args.max_steps:
            raise _Done

    def glide(fr: torch.Tensor, to: torch.Tensor, dur: float) -> torch.Tensor:
        nonlocal target_p
        steps = max(1, int(round(dur * FPS)))
        for k in range(steps):
            target_p = fr + (to - fr) * _smoothstep((k + 1) / steps)
            step()
        return to

    def dwell(dur: float) -> None:
        for _ in range(int(round(dur * FPS))):
            step()

    try:
        # settle the cloth with the arm parked where it spawned
        p0, _ = arm.hand_pose_w()
        target_p = p0.clone()
        dwell(3.0)
        z_rest = float(cloth_local()[:, 2].max())

        # pinch point: the LEFT SLEEVE TIP, ~3 cm inside the max-x edge — two fabric layers
        # plus the seam, the thickest pinchable stack on the flat shirt (a flat mid-surface or
        # single-layer edge pinch closes on air).
        pl = cloth_local()
        tip = pl[pl[:, 0].argmax()]
        gx, gy = float(tip[0]) - 0.03, float(tip[1])
        tip_grasp = torch.tensor([[gx, gy, TABLE_TOP_Z]], device=device) + origin  # pad tips AT the tabletop: nothing slides under them before the close
        tip_hover = tip_grasp + torch.tensor([[0.0, 0.0, 0.15]], device=device)
        tip_lift = tip_grasp + torch.tensor([[0.0, 0.0, args.lift]], device=device)
        print(f"  [adapt] pinch at sleeve tip ({gx:+.3f}, {gy:+.3f}) | settled z_max {z_rest:.3f}", flush=True)

        cur = glide(p0, hand_for_tip(tip_hover), 2.5)     # reach over the shirt
        cur = glide(cur, hand_for_tip(tip_grasp), 1.5)    # descend to the fabric
        arm.grip = CLOSE                                   # pinch (real finger contact)
        dwell(2.0)
        hp, hq = arm.hand_pose_w()
        tip_now = hp + math_utils.quat_apply(hq, tip_off)
        fq = arm.art.data.joint_pos.torch[0, arm.art.find_joints(["panda_finger.*"])[0]]
        near = pl[((pl[:, 0] - gx) ** 2 + (pl[:, 1] - gy) ** 2) < 0.03 ** 2]
        print(f"  [close] hand err {float((hp - target_p).norm()) * 1e3:.0f} mm | tip world "
              f"({float(tip_now[0, 0] - origin[0]):+.3f}, {float(tip_now[0, 1] - origin[1]):+.3f}, "
              f"{float(tip_now[0, 2] - origin[2]):+.3f}) | fingers q {fq.tolist()} | cloth near pinch "
              f"n={near.shape[0]} zmax={float(near[:, 2].max()) if near.shape[0] else float('nan'):.3f}", flush=True)
        cur = glide(cur, hand_for_tip(tip_lift), 2.0)     # lift the shirt
        z_hold = 0.0
        for _ in range(int(round(args.hold * FPS))):
            step()
            z_hold = max(z_hold, float(cloth_local()[:, 2].max()))
        hold_peak[0] = z_hold
        print(f"  [lift] cloth z_max at hold {z_hold:.3f} m (settled {z_rest:.3f})", flush=True)
        cur = glide(cur, hand_for_tip(tip_grasp), 2.0)    # lower back to the table
        arm.grip = OPEN                                    # release
        dwell(1.0)
        cur = glide(cur, hand_for_tip(tip_hover), 1.5)    # retreat clear
        dwell(2.0)
    except _Done:
        print(f"  [cap] stopped at step {tick}", flush=True)

    p = scene.nodal_pos_local()
    lo = torch.tensor(BOUNDS_LO, device=p.device)
    hi = torch.tensor(BOUNDS_HI, device=p.device)
    in_bounds = bool(((p >= lo) & (p <= hi)).all())
    rise = hold_peak[0] - z_rest
    ok = rise >= 0.06 and in_bounds
    print(
        f"TSHIRT-LIFT {'PASS' if ok else 'FAIL'} | cloth rise at hold {rise * 1e3:+.0f} mm"
        f" (gate >= 60) | z_max {hold_peak[0]:.3f} vs settled {z_rest:.3f} | in_bounds={in_bounds}",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
    app.close()

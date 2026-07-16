"""Bimanual latte smoke — two kinematically-driven Frankas: left holds the mug handle, right
grasps the milk cup and pours it into the coffee mug.

KINEMATIC embodiment (Phase 2a): under the MPM manager the arms have no dynamics — each step this
script solves per-arm damped-least-squares DiffIK and *writes joint state* (positions + finite-
difference velocities); the Newton manager runs FK, so every link is a live MPM collider. Grasps
are attachments, not force closure: each vessel's pose is derived from its hand's ACTUAL pose
(`object = hand ∘ grasp_offset`, written kinematically with a finite-difference twist). The left
arm pinches the mug's handle bar and CARRIES the mug (coffee rides inside its moving collider);
the right arm takes the milk cup in a diametric body grasp (outer diameter 0.062 m < the 0.08 m
finger stroke). Real dynamics (MJWarp-coupled) is Phase 2b.

Sequence: reach (interpolated from the arms' actual start poses — no target jump) -> descend ->
close fingers -> the left arm lifts the mug to carry height -> the proven lip-anchored pour,
anchored to the LIFTED mug's live pose (lift, traverse, tilt to 118 deg, drain, recover, return,
set down) -> the mug is lowered back -> release.

Quats are **xyzw** (isaaclab develop / warp convention) throughout.

Verdict: same fluid gates as the scene-only smoke — milk transfer >= 0.70, coffee retention
>= 0.90, spilled <= 0.05 — plus an informational right-hand tracking-error trace.

Runs ONLY under the Newton venv:
  env_newton/bin/python -m robobench.suites.pouring.scripts.latte_bimanual_smoke              # headless
  env_newton/bin/python -m robobench.suites.pouring.scripts.latte_bimanual_smoke --viz kit    # GUI
"""

from __future__ import annotations

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--time_scale", type=float, default=1.0, help="multiply every phase duration")
parser.add_argument("--tilt_deg", type=float, default=118.0, help="full pour tilt of the milk cup [deg]")
parser.add_argument("--mug_tilt_deg", type=float, default=25.0, help="the mug tilts this far TOWARD the cup during the pour [deg]")
parser.add_argument("--lip_clear", type=float, default=0.032, help="cup-lip clearance above the mug's low rim [m]")
parser.add_argument("--hold", type=float, default=2.0, help="extra settle time after the trajectory [s]")
parser.add_argument("--print_every", type=int, default=200, help="progress print period [steps]")
parser.add_argument("--max_steps", type=int, default=None, help="cap total steps (debugging)")
parser.add_argument("--max_dq", type=float, default=0.04, help="per-tick joint step clamp [rad]")
parser.add_argument("--grasp_pitch", type=float, default=30.0, help="downward tilt of the right arm's HORIZONTAL side grasp [deg]: 0 = gripper level with the table, 30 = pointing 30 deg down at the cup")
parser.add_argument("--scene", nargs="*", default=None, metavar="K=V", help="scene cfg overrides")
AppLauncher.add_app_launcher_args(parser)
if "--enable_cameras" in sys.argv:
    # Recording path (scripts/record_video.py): rendering on develop is pumped by visualizers, and
    # an explicit --headless force-disables them — so make headless+kit-visualizer the DEFAULTS.
    sys.argv = [a for a in sys.argv if a != "--headless"]
    parser.set_defaults(headless=True, visualizer=["kit"])
args = parser.parse_args()
livestream_on = args.livestream > 0
render_on = livestream_on or bool(args.visualizer and "none" not in args.visualizer)

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

# Downward-pointing grasp quats, xyzw (see the folding smoke): QD = 180 deg about x (gripper
# down, fingers separating along world y); pre-rotate about world z to aim the finger axis.
QD = torch.tensor([1.0, 0.0, 0.0, 0.0])


def _smoothstep(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


class Arm:
    """Per-arm kinematic DiffIK driver: solves toward a world-frame hand target and writes joint
    state (positions + finite-difference velocities) directly — no dynamics under the MPM manager."""

    def __init__(self, robot, device: str, grip: float = 0.04) -> None:
        art = robot.articulation
        self.art = art
        self.device = device
        self.arm_ids = art.find_joints(["panda_joint[1-7]"])[0]
        self.finger_ids = art.find_joints(["panda_finger.*"])[0]
        self.hand_idx = art.find_bodies(["panda_hand"])[0][0]
        num_base = getattr(art, "num_base_dofs", 0)
        self.jacobi_body = self.hand_idx - 1 if art.is_fixed_base else self.hand_idx
        self.jacobi_joints = [j + num_base for j in self.arm_ids]
        self.grip = grip
        self._prev_q = art.data.joint_pos.torch.clone()
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": 0.05}
            ),
            num_envs=1,
            device=device,
        )
        self.limits = art.data.joint_pos_limits.torch[:, self.arm_ids, :]
        self.rot_err = 0.0  # orientation tracking error [rad], updated by track()

    def hand_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.art.data.body_pos_w.torch[:, self.hand_idx], self.art.data.body_quat_w.torch[:, self.hand_idx]

    def _hand_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        return math_utils.subtract_frame_transforms(
            self.art.data.root_pos_w.torch, self.art.data.root_quat_w.torch, *self.hand_pose_w()
        )

    def track(self, target_pos_w: torch.Tensor, target_quat_w: torch.Tensor, max_dq: float) -> float:
        """One DiffIK step toward a world-frame hand pose; returns the position error [m]."""
        tp, tq = math_utils.subtract_frame_transforms(
            self.art.data.root_pos_w.torch, self.art.data.root_quat_w.torch, target_pos_w, target_quat_w
        )
        ee_p, ee_q = self._hand_pose_b()
        self.ik.set_command(torch.cat([tp, tq], dim=-1), ee_p, ee_q)
        J = self.art.data.body_link_jacobian_w.torch[:, self.jacobi_body, :, self.jacobi_joints]
        R = math_utils.matrix_from_quat(math_utils.quat_inv(self.art.data.root_quat_w.torch))
        J[:, :3, :] = torch.bmm(R, J[:, :3, :])
        J[:, 3:, :] = torch.bmm(R, J[:, 3:, :])
        q_arm = self.art.data.joint_pos.torch[:, self.arm_ids]
        q_des = self.ik.compute(ee_p, ee_q, J, q_arm)
        q_new = (q_arm + (q_des - q_arm).clamp(-max_dq, max_dq)).clamp(self.limits[..., 0], self.limits[..., 1])
        q_full = self.art.data.joint_pos.torch.clone()
        q_full[:, self.arm_ids] = q_new
        q_full[:, self.finger_ids] = self.grip
        qd = (q_full - self._prev_q) * FPS
        self._prev_q = q_full.clone()
        self.art.write_joint_state_to_sim_index(position=q_full, velocity=qd)
        self.rot_err = float(math_utils.quat_error_magnitude(ee_q, tq).max())
        return float((ee_p - tp).norm(dim=-1).max())


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    cfg = ENVS.get("pouring.latte.bimanual_franka.joint")()
    kw: dict = {}
    for kv in args.scene or []:
        k, v = kv.split("=", 1)
        kw[k] = int(v) if v.lstrip("-").isdigit() else float(v)
    if kw:
        from robobench.suites.pouring.scenes import LatteSceneCfg

        cfg.scene_cfg = LatteSceneCfg(**kw)
    env = cfg.build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    if render_on:
        scene.setup_particle_visuals()
    left, right = Arm(env.robot["left"], device), Arm(env.robot["right"], device)
    origin = env.iscene.env_origins[0]
    print(
        f"[latte2] particles: coffee {scene.coffee.particles_per_object}, milk {scene.milk.particles_per_object}",
        flush=True,
    )

    from robobench.suites.pouring.scenes.latte import TABLE_TOP_Z

    def t3(x: float, y: float, z: float) -> torch.Tensor:
        return torch.tensor([[x, y, z]], device=device) + origin

    def q4(q: torch.Tensor | tuple) -> torch.Tensor:
        t = q if isinstance(q, torch.Tensor) else torch.tensor(q)
        return t.to(device).reshape(1, 4)

    def qlerp(qa: torch.Tensor, qb: torch.Tensor, s: float) -> torch.Tensor:
        qb = torch.where((qa * qb).sum(-1, keepdim=True) < 0, -qb, qb)
        q = qa + (qb - qa) * s
        return q / q.norm(dim=-1, keepdim=True)

    def q_ry(deg: float) -> torch.Tensor:
        half = math.radians(deg) / 2.0
        return q4((0.0, math.sin(half), 0.0, math.cos(half)))

    # BOTH hands grasp horizontally, pointing INWARD from opposite sides (gripper axes tilted
    # `grasp_pitch` deg down). During the pour the vessels rotate toward each other, so the hands
    # rotate OUTWARD — their swept volumes never cross the center line between the vessels.
    Q_LEFT = q_ry(90.0 + args.grasp_pitch)  # from -x, pointing +x-down at the mug handle
    Q_RIGHT = q_ry(-(90.0 + args.grasp_pitch))  # from +x, pointing -x-down at the milk cup

    # --- grasp geometry (world; see the scene cfg for the measured mug numbers) ---
    mx, my = c.milk_cup_pos
    z_hat = torch.tensor([[0.0, 0.0, 1.0]], device=device)

    def hand_from_tip(tip_xyz: tuple, quat: torch.Tensor) -> torch.Tensor:
        return t3(*tip_xyz) - 0.113 * math_utils.quat_apply(quat, z_hat)

    # Left: pinch the handle loop's outer vertical bar (x ~ -0.085, loop z ~0.02..0.08)
    lh_grasp = hand_from_tip((-0.080, 0.0, 0.093), Q_LEFT)
    # Right: diametric side grasp around the upper cup body (outer dia 0.062 < the 0.08 stroke)
    rh_grasp = hand_from_tip((mx, my, 0.085), Q_RIGHT)
    lh_hover = lh_grasp + torch.tensor([-0.06, 0.0, 0.05], device=device)
    rh_hover = rh_grasp + torch.tensor([0.07, 0.0, 0.05], device=device)
    GRIP_MUG_BAR, GRIP_CUP_BODY = 0.006, c.milk_cup_r + c.cup_wall + 0.001

    # --- phases ---
    phases: list[tuple[str, float]] = [
        ("reach", 2.5),
        ("descend", 1.5),
        ("grasp", 1.0),
        ("mug_lift", 1.5),
        ("lift", 1.2),
        ("traverse", 1.8),
        ("pour", 3.5),
        ("drain", 1.8),
        ("recover", 1.2),
        ("return", 1.8),
        ("set_down", 1.5),
        ("mug_down", 1.5),
        ("release", 1.5),
    ]
    CUP_PHASES = {"lift", "traverse", "pour", "drain", "recover", "return", "set_down"}
    MUG_ATTACHED = CUP_PHASES | {"mug_lift", "mug_down"}
    durs = [d * args.time_scale for _, d in phases]
    t_edges = [sum(durs[: i + 1]) for i in range(len(durs))]
    total_steps = int(round((t_edges[-1] + args.hold) * FPS))
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    env.reset()
    action = torch.zeros((1, env.robot.action_dim), device=device)

    lh_p0, lh_q0 = (x.clone() for x in left.hand_pose_w())
    rh_p0, rh_q0 = (x.clone() for x in right.hand_pose_w())

    off_cup = off_mug = None
    prev_cup_pose = prev_mug_pose = None
    rh_final = None
    theta_max = math.radians(args.tilt_deg)
    phi_max = math.radians(args.mug_tilt_deg)
    lip_local = torch.tensor([-(c.milk_cup_r + c.cup_wall), 0.0, c.milk_cup_h], device=device)
    rim_local = torch.tensor([[c.coffee_cup_r - 0.002, 0.0, c.coffee_cup_h]], device=device)  # low-rim pt
    mug_base = t3(0.0, 0.0, TABLE_TOP_Z)
    mug_carry = mug_base + torch.tensor([0.0, 0.0, 0.06], device=device)
    cup_start = t3(mx, my, TABLE_TOP_Z)
    last_phase = ""
    err_l = err_r = 0.0

    def fd_twist(pose: torch.Tensor, prev: torch.Tensor | None) -> torch.Tensor:
        if prev is None:
            return torch.zeros((1, 6), device=device)
        ang = math_utils.axis_angle_from_quat(math_utils.quat_mul(pose[:, 3:], math_utils.quat_inv(prev[:, 3:])))
        return torch.cat([(pose[:, :3] - prev[:, :3]) * FPS, ang * FPS], dim=-1)

    def lerp(a: torch.Tensor, b: torch.Tensor, s: float) -> torch.Tensor:
        return a + (b - a) * s

    def tilt_of(name: str, s: float) -> tuple[float, float]:
        """(cup theta, mug phi) for phase-synced counter-rotation."""
        if name == "pour":
            return theta_max * s, phi_max * s
        if name == "drain":
            return theta_max, phi_max
        if name == "recover":
            return theta_max * (1.0 - s), phi_max * (1.0 - s)
        return 0.0, 0.0

    def mug_target(name: str, s: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Scripted mug pose: lift to carry height, tilt +phi toward the cup during the pour."""
        _, phi = tilt_of(name, s)
        if name in ("reach", "descend", "grasp", "release"):
            return mug_base, q4((0.0, 0.0, 0.0, 1.0))
        if name == "mug_lift":
            return lerp(mug_base, mug_carry, s), q4((0.0, 0.0, 0.0, 1.0))
        if name == "mug_down":
            return lerp(mug_carry, mug_base, s), q4((0.0, 0.0, 0.0, 1.0))
        half = phi / 2.0
        return mug_carry, q4((0.0, math.sin(half), 0.0, math.cos(half)))

    def cup_target(name: str, s: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Scripted cup pose: the lip rides just above the (tilting) mug's LOW rim. The lip
        anchor starts OUTBOARD of the rim and slides inboard as theta grows — by the time the
        cup body has rotated up clear of the rim plane, the lip overhangs the mouth."""
        theta, _ = tilt_of(name, s)
        mp, mq = mug_target(name, s)
        low_rim = mp + math_utils.quat_apply(mq, rim_local)
        frac = theta / theta_max if theta_max > 0 else 0.0
        lip_dx = 0.015 - 0.023 * frac
        lip_target = low_rim[0] + torch.tensor([lip_dx, 0.0, args.lip_clear], device=device)
        st, ct = math.sin(-theta), math.cos(-theta)
        lx, lz = lip_local[0], lip_local[2]
        lip_rot = torch.tensor([ct * lx + st * lz, 0.0, -st * lx + ct * lz], device=device)
        anchor = lip_target - lip_rot
        half = -theta / 2.0
        quat = q4((0.0, math.sin(half), 0.0, math.cos(half)))
        if name == "lift":
            return lerp(cup_start, torch.stack([cup_start[0, 0], cup_start[0, 1], anchor[2] + 0.02]).unsqueeze(0), s), quat
        if name == "traverse":
            high = torch.stack([cup_start[0, 0], cup_start[0, 1], anchor[2] + 0.02]).unsqueeze(0)
            return lerp(high, anchor.unsqueeze(0), s), quat
        if name in ("pour", "drain", "recover"):
            return anchor.unsqueeze(0), quat
        if name == "return":
            high = torch.stack([cup_start[0, 0], cup_start[0, 1], anchor[2] + 0.02]).unsqueeze(0)
            return lerp(anchor.unsqueeze(0), high, s), quat
        if name == "set_down":
            high = torch.stack([cup_start[0, 0], cup_start[0, 1], anchor[2] + 0.02]).unsqueeze(0)
            return lerp(high, cup_start, s), quat
        return cup_start, quat

    def hand_target_from(obj_p: torch.Tensor, obj_q: torch.Tensor, off) -> tuple[torch.Tensor, torch.Tensor]:
        inv_q = math_utils.quat_inv(off[1])
        inv_p = -math_utils.quat_apply(inv_q, off[0])
        return math_utils.combine_frame_transforms(obj_p, obj_q, inv_p, inv_q)

    def status() -> str:
        return (
            f"transfer {float(scene.transfer_fraction().mean()):.3f}"
            f" | retention {float(scene.retention_fraction().mean()):.3f}"
            f" | spilled {float(scene.spilled_fraction().mean()):.3f}"
            f" | err L {err_l * 100:4.1f} R {err_r * 100:4.1f} cm"
            f" | rot L {left.rot_err:4.2f} R {right.rot_err:4.2f} rad"
        )

    for step in range(total_steps):
        t = step / FPS
        idx = next((i for i, edge in enumerate(t_edges) if t < edge), len(phases) - 1)
        name, _ = phases[idx]
        s = _smoothstep(1.0 - (t_edges[idx] - t) / max(durs[idx], 1e-9)) if t < t_edges[-1] else 1.0

        # fingers
        if name in ("reach", "descend"):
            left.grip = right.grip = 0.04
        elif name == "grasp":
            left.grip = max(GRIP_MUG_BAR, 0.04 - (0.04 - GRIP_MUG_BAR) * s)
            right.grip = max(GRIP_CUP_BODY, 0.04 - (0.04 - GRIP_CUP_BODY) * s)
        elif name == "release":
            left.grip = min(0.04, GRIP_MUG_BAR + (0.04 - GRIP_MUG_BAR) * s)
            right.grip = min(0.04, GRIP_CUP_BODY + (0.04 - GRIP_CUP_BODY) * s)

        # --- left arm: direct targets pre-attach, then derived from the scripted mug pose ---
        if name == "reach":
            err_l = left.track(lerp(lh_p0, lh_hover, s), qlerp(lh_q0, Q_LEFT, s), args.max_dq)
        elif name == "descend":
            err_l = left.track(lerp(lh_hover, lh_grasp, s), Q_LEFT, args.max_dq)
        elif name in ("grasp", "release"):
            err_l = left.track(lh_grasp, Q_LEFT, args.max_dq)
        else:
            if off_mug is None:
                hp, hq = left.hand_pose_w()
                off_mug = math_utils.subtract_frame_transforms(hp, hq, scene.mug_pose_w[:, :3], scene.mug_pose_w[:, 3:])
                print(f"  [attach] mug offset recorded | hand err L {err_l * 100:.1f} cm", flush=True)
            mp, mq = mug_target(name, s)
            lh_t, lh_q = hand_target_from(mp, mq, off_mug)
            err_l = left.track(lh_t, lh_q, args.max_dq)
            hp, hq = left.hand_pose_w()
            mpp, mqq = math_utils.combine_frame_transforms(hp, hq, off_mug[0], off_mug[1])
            mug_pose = torch.cat([mpp, mqq], dim=-1)
            scene.write_mug_pose(mug_pose, fd_twist(mug_pose, prev_mug_pose))
            prev_mug_pose = mug_pose.clone()

        # --- right arm: direct targets pre-attach, then derived from the scripted cup pose ---
        if name == "reach":
            err_r = right.track(lerp(rh_p0, rh_hover, s), qlerp(rh_q0, Q_RIGHT, s), args.max_dq)
        elif name == "descend":
            err_r = right.track(lerp(rh_hover, rh_grasp, s), Q_RIGHT, args.max_dq)
        elif name in ("grasp", "mug_lift"):
            err_r = right.track(rh_grasp, Q_RIGHT, args.max_dq)
        elif name in CUP_PHASES:
            if off_cup is None:
                hp, hq = right.hand_pose_w()
                off_cup = math_utils.subtract_frame_transforms(hp, hq, cup_start, q4((0.0, 0.0, 0.0, 1.0)))
                print("  [attach] cup offset recorded", flush=True)
            cp_t, cq_t = cup_target(name, s)
            rh_t, rh_q = hand_target_from(cp_t, cq_t, off_cup)
            err_r = right.track(rh_t, rh_q, args.max_dq)
            hp, hq = right.hand_pose_w()
            cpp, cqq = math_utils.combine_frame_transforms(hp, hq, off_cup[0], off_cup[1])
            pose = torch.cat([cpp, cqq], dim=-1)
            scene.milk_cup.write_root_link_pose_to_sim_index(root_pose=pose)
            scene.milk_cup.write_root_link_velocity_to_sim_index(root_velocity=fd_twist(pose, prev_cup_pose))
            prev_cup_pose = pose.clone()
        else:  # mug_down / release: hold where the cup was set down
            if rh_final is None:
                rh_final = tuple(x.clone() for x in right.hand_pose_w())
            err_r = right.track(rh_final[0], rh_final[1], args.max_dq)

        env.step(action, render=render_on)
        if render_on and step % 5 == 0:
            scene.push_particle_visuals()

        if name != last_phase:
            print(f"  t={t:5.1f}s phase {name:9s} | {status()}", flush=True)
            last_phase = name
        if step % args.print_every == 0:
            print(f"  step {step:5d} t={t:5.1f}s {name:9s} | {status()}", flush=True)

    transfer = float(scene.transfer_fraction().mean())
    retention = float(scene.retention_fraction().mean())
    spilled = float(scene.spilled_fraction().mean())
    ok = transfer >= 0.70 and retention >= 0.90 and spilled <= 0.05
    print(
        f"LATTE-BIMANUAL {'PASS' if ok else 'FAIL'} | transfer {transfer:.3f} (gate >= 0.70)"
        f" | retention {retention:.3f} (>= 0.90) | spilled {spilled:.3f} (<= 0.05)",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
    app.close()

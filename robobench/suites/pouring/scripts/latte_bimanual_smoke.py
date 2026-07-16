"""Bimanual latte smoke — two kinematically-driven Frankas: left holds the mug handle, right
grasps the milk cup and pours it into the coffee mug.

KINEMATIC embodiment (Phase 2a): under the MPM manager the arms have no dynamics — each step this
script solves per-arm damped-least-squares DiffIK and *writes joint state* (positions + finite-
difference velocities); the Newton manager runs FK, so every link is a live MPM collider. Grasps
are attachments, not force closure: after the grasp phase the milk cup's pose is derived from the
right hand's ACTUAL pose each step (`cup = hand ∘ grasp_offset`, written kinematically with a
finite-difference twist), and the mug stays the static collider it always was — the left arm's
handle grasp is visual. Real dynamics (MJWarp-coupled) is Phase 2b.

Sequence: reach (hands hover over their grasp points) -> descend -> close fingers + record the
grasp offset -> the proven lip-anchored pour (lift, traverse, tilt to 118 deg with the pouring
edge pinned above the mug mouth, drain, recover, return, set down) -> release + retreat.

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
parser.add_argument("--tilt_deg", type=float, default=118.0, help="full pour tilt [deg]")
parser.add_argument("--lip_height", type=float, default=0.06, help="lip anchor height above the mug rim [m]")
parser.add_argument("--lip_x", type=float, default=0.02, help="lip anchor x over the mug mouth [m]")
parser.add_argument("--hold", type=float, default=2.0, help="extra settle time after the trajectory [s]")
parser.add_argument("--print_every", type=int, default=200, help="progress print period [steps]")
parser.add_argument("--max_steps", type=int, default=None, help="cap total steps (debugging)")
parser.add_argument("--max_dq", type=float, default=0.04, help="per-tick joint step clamp [rad]")
parser.add_argument("--grasp_pitch", type=float, default=30.0, help="right grasp pre-tilt about +y [deg]: starts the wrist tilted AWAY from the pour so the 118 deg swing ends less past vertical")
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

    rz90 = torch.tensor([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)])
    Q_LEFT = q4(QD)  # fingers along world y — across the mug handle bar (bar runs along x)
    # Right: fingers along world x (straddle the rim wall radially), pre-tilted about +y so the
    # -118 deg pour swing ends less far past vertical (wrist-limit headroom).
    half_pitch = math.radians(args.grasp_pitch) / 2.0
    ry_pitch = torch.tensor([0.0, math.sin(half_pitch), 0.0, math.cos(half_pitch)])
    Q_RIGHT = q4(
        math_utils.quat_mul(
            ry_pitch.unsqueeze(0), math_utils.quat_mul(rz90.unsqueeze(0), QD.unsqueeze(0))
        ).squeeze(0)
    )

    # --- grasp geometry (world, env-local; see the scene cfg for the measured mug numbers) ---
    mx, my = c.milk_cup_pos
    z_hat = torch.tensor([[0.0, 0.0, 1.0]], device=device)

    def hand_from_tip(tip_xyz: tuple, quat: torch.Tensor) -> torch.Tensor:
        """Hand-origin target that puts the fingertip midpoint (0.113 m along the hand z-axis)
        at `tip_xyz`, for any hand orientation."""
        return t3(*tip_xyz) - 0.113 * math_utils.quat_apply(quat, z_hat)

    # Left: pinch the mug handle's top bar (bar along x at x ~[-0.092,-0.055], z ~0.075; 1 cm bite)
    handle_hand = hand_from_tip((-0.073, 0.0, 0.065), Q_LEFT)
    # Right: pinch the milk-cup rim wall on its +x side (wall mid-radius, ~2 cm below the rim)
    rim_hand = hand_from_tip((mx + c.milk_cup_r + c.cup_wall / 2, my, c.milk_cup_h - 0.022), Q_RIGHT)

    # --- the proven lip-anchored cup trajectory (from latte_pour_smoke) ---
    r_outer = c.milk_cup_r + c.cup_wall
    lip_local = torch.tensor([-r_outer, 0.0, c.milk_cup_h], device=device)
    rim_z = TABLE_TOP_Z + c.coffee_cup_h
    lip_target = torch.tensor([args.lip_x, my, rim_z + args.lip_height], device=device) + origin
    theta_max = math.radians(args.tilt_deg)
    p_start = t3(mx, my, TABLE_TOP_Z)[0]
    p_travel = t3(mx, my, rim_z + args.lip_height + 0.02)[0]

    def anchor_pos(theta: float) -> torch.Tensor:
        st, ct = math.sin(-theta), math.cos(-theta)
        lx, lz = lip_local[0], lip_local[2]
        lip_rot = torch.tensor([ct * lx + st * lz, 0.0, -st * lx + ct * lz], device=device)
        return lip_target - lip_rot

    p_anchor0 = anchor_pos(0.0)

    def lerp(a: torch.Tensor, b: torch.Tensor, s: float) -> torch.Tensor:
        return a + (b - a) * s

    # (name, duration, cup pose fn(alpha) -> (pos, theta)); fingers/attachment handled per phase.
    phases: list[tuple[str, float, object]] = [
        ("reach", 2.0, None),
        ("descend", 1.5, None),
        ("grasp", 1.0, None),
        ("lift", 1.2, lambda s: (lerp(p_start, p_travel, s), 0.0)),
        ("traverse", 1.8, lambda s: (lerp(p_travel, p_anchor0, s), 0.0)),
        ("pour", 3.5, lambda s: (anchor_pos(theta_max * s), theta_max * s)),
        ("drain", 1.8, lambda s: (anchor_pos(theta_max), theta_max)),
        ("recover", 1.2, lambda s: (anchor_pos(theta_max * (1.0 - s)), theta_max * (1.0 - s))),
        ("return", 1.8, lambda s: (lerp(p_anchor0, p_travel, s), 0.0)),
        ("set_down", 1.5, lambda s: (lerp(p_travel, p_start, s), 0.0)),
        ("release", 1.5, None),
    ]
    durs = [d * args.time_scale for _, d, _ in phases]
    t_edges = [sum(durs[: i + 1]) for i in range(len(durs))]
    total_steps = int(round((t_edges[-1] + args.hold) * FPS))
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    env.reset()
    action = torch.zeros((1, env.robot.action_dim), device=device)

    # left-hand targets (constant after grasp) and right-hand pre-grasp targets
    lift6 = torch.tensor([0.0, 0.0, 0.06], device=device)
    lh_hover, lh_grasp = handle_hand + lift6, handle_hand
    rh_hover, rh_grasp = rim_hand + lift6, rim_hand

    grasp_off: tuple[torch.Tensor, torch.Tensor] | None = None  # cup pose in the right-hand frame
    prev_cup_pose: torch.Tensor | None = None
    last_phase = ""
    err_l = err_r = 0.0

    def cup_pose_now() -> tuple[torch.Tensor, torch.Tensor]:
        hp, hq = right.hand_pose_w()
        return math_utils.combine_frame_transforms(hp, hq, grasp_off[0], grasp_off[1])

    def status() -> str:
        return (
            f"transfer {float(scene.transfer_fraction().mean()):.3f}"
            f" | retention {float(scene.retention_fraction().mean()):.3f}"
            f" | spilled {float(scene.spilled_fraction().mean()):.3f}"
            f" | err L {err_l * 100:4.1f} R {err_r * 100:4.1f} cm"
            f" | rot R {right.rot_err:4.2f} rad"
        )

    for step in range(total_steps):
        t = step / FPS
        idx = next((i for i, edge in enumerate(t_edges) if t < edge), len(phases) - 1)
        name, dur, cup_fn = phases[idx]
        s = _smoothstep(1.0 - (t_edges[idx] - t) / max(durs[idx], 1e-9)) if t < t_edges[-1] else 1.0

        # fingers
        if name in ("reach", "descend"):
            left.grip = right.grip = 0.04
        elif name == "grasp":
            left.grip = max(0.006, 0.04 - 0.034 * s)  # close on the ~1 cm handle bar
            right.grip = max(0.004, 0.04 - 0.036 * s)  # close on the 5 mm cup wall
        elif name == "release":
            left.grip = right.grip = min(0.04, 0.006 + 0.034 * s)

        # left hand target: hover -> grasp -> hold
        lh_target = lerp(lh_hover, lh_grasp, s) if name == "descend" else (lh_hover if name == "reach" else lh_grasp)
        err_l = left.track(lh_target, Q_LEFT, args.max_dq)

        # right hand target
        if name in ("reach", "descend", "grasp", "release"):
            rh_target, rh_quat = (
                (rh_hover, Q_RIGHT) if name == "reach" else (lerp(rh_hover, rh_grasp, s), Q_RIGHT)
            )
            if name in ("grasp", "release"):
                rh_target, rh_quat = rh_grasp, Q_RIGHT
            err_r = right.track(rh_target, rh_quat, args.max_dq)
        else:
            if grasp_off is None:  # first attached step: record the cup pose in the hand frame
                hp, hq = right.hand_pose_w()
                cup_p = t3(mx, my, TABLE_TOP_Z)  # the cup has not moved yet: its spawn pose
                cup_q = q4((0.0, 0.0, 0.0, 1.0))
                grasp_off = math_utils.subtract_frame_transforms(hp, hq, cup_p, cup_q)
                print(f"  [attach] grasp offset recorded | hand err R {err_r * 100:.1f} cm", flush=True)
            # attached: cup trajectory -> hand target = cup_target ∘ inv(offset)
            cup_tgt_p, theta = cup_fn(s)
            half = -theta / 2.0
            cup_tgt_q = q4((0.0, math.sin(half), 0.0, math.cos(half)))
            inv_q = math_utils.quat_inv(grasp_off[1])
            inv_p = -math_utils.quat_apply(inv_q, grasp_off[0])
            ht_p, ht_q = math_utils.combine_frame_transforms(cup_tgt_p.unsqueeze(0), cup_tgt_q, inv_p, inv_q)
            err_r = right.track(ht_p, ht_q, args.max_dq)
            # cup rides the hand's ACTUAL pose (attachment), with a finite-difference twist
            cp, cq = cup_pose_now()
            pose = torch.cat([cp, cq], dim=-1)
            twist = (
                torch.zeros((1, 6), device=device)
                if prev_cup_pose is None
                else torch.cat(
                    [
                        (pose[:, :3] - prev_cup_pose[:, :3]) * FPS,
                        math_utils.axis_angle_from_quat(
                            math_utils.quat_mul(pose[:, 3:], math_utils.quat_inv(prev_cup_pose[:, 3:]))
                        )
                        * FPS,
                    ],
                    dim=-1,
                )
            )
            scene.milk_cup.write_root_link_pose_to_sim_index(root_pose=pose)
            scene.milk_cup.write_root_link_velocity_to_sim_index(root_velocity=twist)
            prev_cup_pose = pose.clone()

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

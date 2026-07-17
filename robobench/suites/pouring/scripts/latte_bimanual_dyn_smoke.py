"""Bimanual latte smoke, Phase 2b — DYNAMIC Frankas on the coupled MJWarp+MPM substrate.

The arms are real now: each tick this script solves per-arm damped-least-squares DiffIK from the
arms' ACTUAL joint state and feeds the result as the joint-position ACTION — actuator PD
(kp 400 / kd 80, gravcomp 1.0, effort 300) tracks it under gravity inside SolverMuJoCo, with
MuJoCo-internal rigid contacts live (arm-table, arm-arm, arm-floor). The liquids run the same
implicit-MPM recipe, stepped once per tick against the POST-RIGID body poses (one-way coupling;
see `robobench.suites.pouring.coupled_manager`).

Grasps remain attachments (Phase 2a pattern): the vessels are kinematic ghosts — MPM colliders
ridden on each hand's ACTUAL (dynamics-lagged) pose via `object = hand ∘ grasp_offset`, written
AFTER the physics step; MuJoCo never sees their geometry. Real force closure is a later phase.

Timing: `FrankaRobot.JOINT_CONTROL_DT` is pinned to 1/FPS so `env.step` == exactly ONE physics
tick and `t = step / FPS` is exact. Phase 2a's control_period was 4, so its labeled durations
ran 4x longer in physics time — `--time_scale` defaults to 4.0 to reproduce the choreography
cadence that was actually validated.

Sequence and verdict gates are Phase 2a's: reach -> descend -> grasp -> mug carried in the air ->
lip-anchored partial pour with the departed-fraction trigger -> recover -> set down -> release.
PASS: transferred >= 0.15, kept-in-pitcher >= 0.15, retention >= 0.90, spilled <= 0.05.

Quats are **xyzw** (isaaclab develop / warp convention) throughout.

Runs ONLY under the Newton venv:
  env_newton/bin/python -m robobench.suites.pouring.scripts.latte_bimanual_dyn_smoke              # headless
  env_newton/bin/python -m robobench.suites.pouring.scripts.latte_bimanual_dyn_smoke --viz kit    # GUI
"""

from __future__ import annotations

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--time_scale", type=float, default=4.0, help="multiply every phase duration (4.0 reproduces Phase 2a's validated physical cadence — its control decimation stretched the labeled durations 4x)")
parser.add_argument("--tilt_deg", type=float, default=118.0, help="MAX pour tilt of the pitcher [deg]; the fill trigger usually stops it earlier")
parser.add_argument("--mug_tilt_deg", type=float, default=15.0, help="the mug tilts this far TOWARD the pitcher during the pour [deg]")
parser.add_argument("--lip_clear", type=float, default=0.032, help="pitcher-lip clearance above the mug's low rim [m]")
parser.add_argument("--pour_fraction", type=float, default=0.18, help="STOP pouring once this fraction of the milk has DEPARTED the pitcher (polled every step)")
parser.add_argument("--hold", type=float, default=2.0, help="extra settle time after the trajectory [s]")
parser.add_argument("--print_every", type=int, default=400, help="progress print period [steps]")
parser.add_argument("--max_steps", type=int, default=None, help="cap total steps (debugging)")
parser.add_argument("--max_dq", type=float, default=0.04, help="per-tick joint REFERENCE step clamp [rad] (8 rad/s slew at 200 Hz)")
parser.add_argument("--lead_max", type=float, default=0.30, help="max lead of the commanded reference over the ACTUAL joints [rad]: bounds PD force at kp*lead (120 N*m) and windup when blocked, while allowing sustained speed kp*lead/kd (~1.5 rad/s) — clamping the target to the actual joints instead would cap sustained speed at kp*max_dq/kd = 0.2 rad/s, too slow for the recover untilt")
parser.add_argument("--grasp_pitch", type=float, default=30.0, help="downward tilt of the horizontal side grasps [deg]")
parser.add_argument("--scene", nargs="*", default=None, metavar="K=V", help="scene cfg overrides")
parser.add_argument("--sim", nargs="*", default=None, metavar="K=V", help="sim cfg overrides (MpmSimCfg fields, e.g. use_cuda_graph=0 num_substeps=4)")
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
from robobench.robots.franka import FrankaRobot  # noqa: E402

FPS = 200  # must match MpmSimCfg.dt
RENDER_EVERY = 4  # render/particle-push cadence [ticks] (matches Phase 2a's effective cadence)

# One env.step == one physics tick: the 50 Hz default would make env.step run 4 ticks per call
# and quantize both the choreography clock and the vessel writes.
FrankaRobot.JOINT_CONTROL_DT = 1.0 / FPS


def _smoothstep(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


class Arm:
    """Per-arm DiffIK servo (Phase 2b): solves toward a world-frame hand target from the ACTUAL
    joint state and returns the 9-wide joint-position child action (7 arm + 2 fingers) for the
    actuator PD — no sim state writes."""

    def __init__(self, robot, device: str, grip: float = 0.04) -> None:
        art = robot.articulation
        self.art = art
        self.device = device
        self.arm_ids = art.find_joints(["panda_joint[1-7]"])[0]
        self.hand_idx = art.find_bodies(["panda_hand"])[0][0]
        num_base = getattr(art, "num_base_dofs", 0)
        self.jacobi_body = self.hand_idx - 1 if art.is_fixed_base else self.hand_idx
        self.jacobi_joints = [j + num_base for j in self.arm_ids]
        self.grip = grip
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": 0.05}
            ),
            num_envs=1,
            device=device,
        )
        self.limits = art.data.joint_pos_limits.torch[:, self.arm_ids, :]
        self.q_ref: torch.Tensor | None = None  # integrated joint reference (bounded-lead servo)
        self.tgt_p: torch.Tensor | None = None  # last world hand target (for post-step errors)
        self.tgt_q: torch.Tensor | None = None

    def hand_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.art.data.body_pos_w.torch[:, self.hand_idx], self.art.data.body_quat_w.torch[:, self.hand_idx]

    def _hand_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        return math_utils.subtract_frame_transforms(
            self.art.data.root_pos_w.torch, self.art.data.root_quat_w.torch, *self.hand_pose_w()
        )

    def solve(self, target_pos_w: torch.Tensor, target_quat_w: torch.Tensor, max_dq: float) -> torch.Tensor:
        """One DiffIK step toward a world-frame hand pose; returns the (1, 9) child action."""
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
        # Bounded-lead reference: integrate the command on the REFERENCE (slew-limited by max_dq)
        # rather than re-anchoring to the actual joints — anchoring caps sustained speed at
        # kp*max_dq/kd = 0.2 rad/s, far below the recover untilt's ~1 rad/s. The lead clamp keeps
        # the reference within `lead_max` of the actual joints: bounded PD force, no windup.
        if self.q_ref is None:
            self.q_ref = q_arm.clone()
        q_ref = (self.q_ref + (q_des - self.q_ref).clamp(-max_dq, max_dq)).clamp(
            self.limits[..., 0], self.limits[..., 1]
        )
        q_ref = q_ref.clamp(q_arm - args.lead_max, q_arm + args.lead_max)
        self.q_ref = q_ref
        self.tgt_p, self.tgt_q = target_pos_w, target_quat_w
        grip = torch.full((1, 2), self.grip, device=self.device)
        return torch.cat([q_ref, grip], dim=-1)

    def errors(self) -> tuple[float, float]:
        """Post-step hand tracking error vs the last target: (position [m], rotation [rad])."""
        if self.tgt_p is None:
            return 0.0, 0.0
        hp, hq = self.hand_pose_w()
        return (
            float((hp - self.tgt_p).norm(dim=-1).max()),
            float(math_utils.quat_error_magnitude(hq, self.tgt_q).max()),
        )


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    cfg = ENVS.get("pouring.latte_dyn.bimanual_franka.joint")()

    def kvparse(pairs) -> dict:
        out: dict = {}
        for kv in pairs or []:
            k, v = kv.split("=", 1)
            out[k] = int(v) if v.lstrip("-").isdigit() else float(v)
        return out

    build_kw: dict = {}
    scene_kw = kvparse(args.scene)
    if scene_kw:
        from robobench.suites.pouring.scenes import LatteSceneCfg

        cfg.scene_cfg = LatteSceneCfg(**scene_kw)
    sim_kw = kvparse(args.sim)
    if sim_kw:
        build_kw["sim_overrides"] = sim_kw
    env = cfg.build(num_envs=1, device=device, **build_kw)
    scene = env.scene
    c = scene.cfg
    if render_on:
        scene.setup_particle_visuals()
    left, right = Arm(env.robot["left"], device), Arm(env.robot["right"], device)
    assert env.robot.control_period == 1, f"expected 1 tick per env.step, got {env.robot.control_period}"
    origin = env.iscene.env_origins[0]
    print(
        f"[latte2b] particles: coffee {scene.coffee.particles_per_object}, milk {scene.milk.particles_per_object}",
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
    # `grasp_pitch` deg down); during the pour they rotate OUTWARD — swept volumes never cross
    # the center line between the vessels (Phase 2a choreography, unchanged).
    Q_LEFT = q_ry(90.0 + args.grasp_pitch)
    Q_RIGHT = q_ry(-(90.0 + args.grasp_pitch))

    # --- grasp geometry (world; see the scene cfg for the measured asset numbers) ---
    px, py = c.pitcher_pos
    z_hat = torch.tensor([[0.0, 0.0, 1.0]], device=device)

    def hand_from_tip(tip_xyz: tuple, quat: torch.Tensor) -> torch.Tensor:
        return t3(*tip_xyz) - 0.113 * math_utils.quat_apply(quat, z_hat)

    # Left: pinch the mug handle loop's outer vertical bar; Right: pinch the PITCHER handle's
    # outer vertical bar (both bars are VISUAL-only — kinematic-ghost vessels have no MuJoCo
    # geometry, so the fingers close through them; the attachment carries the vessel).
    lh_grasp = hand_from_tip((-0.080, 0.0, 0.093), Q_LEFT)
    rh_grasp = hand_from_tip((px + 0.060, py, 0.095), Q_RIGHT)
    lh_hover = lh_grasp + torch.tensor([-0.06, 0.0, 0.05], device=device)
    rh_hover = rh_grasp + torch.tensor([0.07, 0.0, 0.05], device=device)
    GRIP_MUG_BAR = GRIP_PITCHER_BAR = 0.006

    # --- phases ---
    phases: list[tuple[str, float]] = [
        ("reach", 2.5),
        ("descend", 1.5),
        ("grasp", 1.0),
        ("mug_lift", 1.5),
        ("lift", 1.2),
        ("traverse", 1.8),
        ("pour", 6.0),
        ("drain", 1.8),
        ("recover", 0.6),
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
    # Teleport hygiene: re-seed the MPM collider pose history so the reset is not read as a
    # (jump/dt) collider velocity by the backward finite difference (no-op here — reset poses
    # equal spawn poses — but the correct pattern for any episodic reset).
    from robobench.suites.pouring.coupled_manager import NewtonCoupledMJWarpMPMManager

    NewtonCoupledMJWarpMPMManager.resync_collider_history()

    lh_p0, lh_q0 = (x.clone() for x in left.hand_pose_w())
    rh_p0, rh_q0 = (x.clone() for x in right.hand_pose_w())

    off_cup = off_mug = None
    prev_cup_pose = prev_mug_pose = None
    rh_final = None
    theta_max = math.radians(args.tilt_deg)
    phi_max = math.radians(args.mug_tilt_deg)
    lip_local = torch.tensor([-(c.pitcher_r + c.pitcher_wall), 0.0, c.pitcher_h], device=device)
    rim_local = torch.tensor([[c.coffee_cup_r - 0.002, 0.0, c.coffee_cup_h]], device=device)  # low-rim pt
    mug_base = t3(0.0, 0.0, TABLE_TOP_Z)
    mug_carry = mug_base + torch.tensor([0.0, 0.0, 0.06], device=device)
    cup_start = t3(px, py, TABLE_TOP_Z)
    last_phase = ""
    err_l = err_r = rot_l = rot_r = 0.0
    # Pour-to-fraction: freeze the tilt and jump the clock to `recover` once enough milk has
    # DEPARTED the pitcher; recover untilts from the FROZEN angles.
    time_shift = 0.0
    tilt_frozen: tuple[float, float] | None = None
    fill_surface = float("-inf")
    transfer_now = 0.0

    def fd_twist(pose: torch.Tensor, prev: torch.Tensor | None) -> torch.Tensor:
        if prev is None:
            return torch.zeros((1, 6), device=device)
        ang = math_utils.axis_angle_from_quat(math_utils.quat_mul(pose[:, 3:], math_utils.quat_inv(prev[:, 3:])))
        return torch.cat([(pose[:, :3] - prev[:, :3]) * FPS, ang * FPS], dim=-1)

    def lerp(a: torch.Tensor, b: torch.Tensor, s: float) -> torch.Tensor:
        return a + (b - a) * s

    def tilt_of(name: str, s: float) -> tuple[float, float]:
        """(pitcher theta, mug phi) for phase-synced counter-rotation; frozen once triggered."""
        if tilt_frozen is not None and name in ("pour", "drain"):
            return tilt_frozen
        if name == "pour":
            return theta_max * s, phi_max * s
        if name == "drain":
            return theta_max, phi_max
        if name == "recover":
            th0, ph0 = tilt_frozen if tilt_frozen is not None else (theta_max, phi_max)
            return th0 * (1.0 - s), ph0 * (1.0 - s)
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
        """Scripted pitcher pose, barista-style: lip slightly INSIDE the (tilting) mug's LOW rim,
        starting HIGH and descending as theta grows (Phase 2a geometry, unchanged)."""
        theta, _ = tilt_of(name, s)
        mp, mq = mug_target(name, s)
        low_rim = mp + math_utils.quat_apply(mq, rim_local)
        lx, lz = float(lip_local[0]), float(lip_local[2])
        lip_up = max(args.lip_clear, 0.005 + math.sin(theta) * lx + math.cos(theta) * lz)
        lip_target = low_rim[0] + torch.tensor([-0.020, 0.0, lip_up], device=device)
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
            f"transfer {transfer_now:.3f} (target {args.pour_fraction:.2f}) | fill {fill_surface * 100:4.1f} cm"
            f" | milk-in-pitcher {float(scene.milk_in_pitcher_fraction().mean()):.3f}"
            f" | retention {float(scene.retention_fraction().mean()):.3f}"
            f" | spilled {float(scene.spilled_fraction().mean()):.3f}"
            f" | err L {err_l * 100:4.1f} R {err_r * 100:4.1f} cm | rot L {rot_l:.2f} R {rot_r:.2f} rad"
        )

    action = torch.zeros((1, env.robot.action_dim), device=device)

    for step in range(total_steps):
        t = step / FPS + time_shift
        idx = next((i for i, edge in enumerate(t_edges) if t < edge), len(phases) - 1)
        name, _ = phases[idx]
        s = _smoothstep(1.0 - (t_edges[idx] - t) / max(durs[idx], 1e-9)) if t < t_edges[-1] else 1.0

        # pour trigger: freeze the tilt and jump the clock to `recover` once enough milk has
        # DEPARTED the pitcher (every-step poll — the avalanche lasts ~1 s)
        if step % 10 == 0:
            fill_surface = scene.mug_surface_z()
            transfer_now = float(scene.transfer_fraction().mean())
        departed = 1.0 - float(scene.milk_in_pitcher_fraction().mean()) if name in ("pour", "drain") else 0.0
        if name in ("pour", "drain") and tilt_frozen is None and departed >= args.pour_fraction:
            tilt_frozen = tilt_of(name, s)
            drain_end = t_edges[[n for n, _ in phases].index("drain")]
            time_shift += drain_end - t
            t = step / FPS + time_shift
            name, s = "recover", 0.0
            print(
                f"  [fill] target reached at tilt {math.degrees(tilt_frozen[0]):.1f} deg -> recovering",
                flush=True,
            )

        # fingers
        if name in ("reach", "descend"):
            left.grip = right.grip = 0.04
        elif name == "grasp":
            left.grip = max(GRIP_MUG_BAR, 0.04 - (0.04 - GRIP_MUG_BAR) * s)
            right.grip = max(GRIP_PITCHER_BAR, 0.04 - (0.04 - GRIP_PITCHER_BAR) * s)
        elif name == "release":
            left.grip = min(0.04, GRIP_MUG_BAR + (0.04 - GRIP_MUG_BAR) * s)
            right.grip = min(0.04, GRIP_PITCHER_BAR + (0.04 - GRIP_PITCHER_BAR) * s)

        # --- hand targets (from the CURRENT actual state; physics advances below) ---
        if name == "reach":
            lh_t, lh_q = lerp(lh_p0, lh_hover, s), qlerp(lh_q0, Q_LEFT, s)
        elif name == "descend":
            lh_t, lh_q = lerp(lh_hover, lh_grasp, s), Q_LEFT
        elif name in ("grasp", "release"):
            lh_t, lh_q = lh_grasp, Q_LEFT
        else:
            if off_mug is None:
                hp, hq = left.hand_pose_w()
                off_mug = math_utils.subtract_frame_transforms(hp, hq, scene.mug_pose_w[:, :3], scene.mug_pose_w[:, 3:])
                print(f"  [attach] mug offset recorded | hand err L {err_l * 100:.1f} cm", flush=True)
            mp, mq = mug_target(name, s)
            lh_t, lh_q = hand_target_from(mp, mq, off_mug)

        if name == "reach":
            rh_t, rh_q = lerp(rh_p0, rh_hover, s), qlerp(rh_q0, Q_RIGHT, s)
        elif name == "descend":
            rh_t, rh_q = lerp(rh_hover, rh_grasp, s), Q_RIGHT
        elif name in ("grasp", "mug_lift"):
            rh_t, rh_q = rh_grasp, Q_RIGHT
        elif name in CUP_PHASES:
            if off_cup is None:
                hp, hq = right.hand_pose_w()
                off_cup = math_utils.subtract_frame_transforms(hp, hq, cup_start, q4((0.0, 0.0, 0.0, 1.0)))
                print("  [attach] pitcher offset recorded", flush=True)
            cp_t, cq_t = cup_target(name, s)
            rh_t, rh_q = hand_target_from(cp_t, cq_t, off_cup)
        else:  # mug_down / release: hold where the cup was set down
            if rh_final is None:
                rh_final = tuple(x.clone() for x in right.hand_pose_w())
            rh_t, rh_q = rh_final

        action[:, 0:9] = left.solve(lh_t, lh_q, args.max_dq)
        action[:, 9:18] = right.solve(rh_t, rh_q, args.max_dq)
        env.step(action, render=render_on and step % RENDER_EVERY == 0)

        # --- vessels ride the hands' ACTUAL (post-step) poses ---
        if name in MUG_ATTACHED and off_mug is not None:
            hp, hq = left.hand_pose_w()
            mpp, mqq = math_utils.combine_frame_transforms(hp, hq, off_mug[0], off_mug[1])
            mug_pose = torch.cat([mpp, mqq], dim=-1)
            scene.write_mug_pose(mug_pose, fd_twist(mug_pose, prev_mug_pose))
            prev_mug_pose = mug_pose.clone()
        if name in CUP_PHASES and off_cup is not None:
            hp, hq = right.hand_pose_w()
            cpp, cqq = math_utils.combine_frame_transforms(hp, hq, off_cup[0], off_cup[1])
            pose = torch.cat([cpp, cqq], dim=-1)
            scene.write_pitcher_pose(pose, fd_twist(pose, prev_cup_pose))
            prev_cup_pose = pose.clone()

        err_l, rot_l = left.errors()
        err_r, rot_r = right.errors()
        if render_on and step % RENDER_EVERY == 0:
            scene.push_particle_visuals()

        if step % 100 == 0 and bool(torch.isnan(left.art.data.joint_pos.torch).any() | torch.isnan(right.art.data.joint_pos.torch).any()):
            print(f"LATTE-BIMANUAL-DYN FAIL | NaN joint state at step {step} (rigid solver exploded)", flush=True)
            env.close()
            return

        if name != last_phase:
            print(f"  t={t:5.1f}s phase {name:9s} | {status()}", flush=True)
            last_phase = name
        if step % args.print_every == 0:
            print(f"  step {step:5d} t={t:5.1f}s {name:9s} | {status()}", flush=True)

    transfer = float(scene.transfer_fraction().mean())
    keep = float(scene.milk_in_pitcher_fraction().mean())
    retention = float(scene.retention_fraction().mean())
    spilled = float(scene.spilled_fraction().mean())
    ok = transfer >= 0.15 and keep >= 0.15 and retention >= 0.90 and spilled <= 0.05
    print(
        f"LATTE-BIMANUAL-DYN {'PASS' if ok else 'FAIL'} | milk transferred {transfer:.3f}"
        f" (>= 0.15) | milk kept in pitcher {keep:.3f} (>= 0.15)"
        f" | retention {retention:.3f} (>= 0.90) | spilled {spilled:.3f} (<= 0.05)"
        f" | final err L {err_l * 100:.1f} R {err_r * 100:.1f} cm",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
    app.close()

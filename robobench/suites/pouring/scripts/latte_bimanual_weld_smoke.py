"""Bimanual latte smoke — dynamic vessels carried by weld-at-grasp.

Dynamic Frankas (DiffIK -> actuator PD, MuJoCo rigid contacts) and DYNAMIC vessels (authored
mass, concave rigid proxies, one-way MPM liquids). Carrying is a MuJoCo equality WELD between
hand and vessel; metrics read the vessels' ACTUAL poses, so drops and topples score honestly.

Two modes:
  default -> scene `latte_weld`: this script engages/releases the welds (scripted grasp);
  --auto  -> scene `latte_auto`: the SCENE engages/releases welds from gripper proximity +
             closure (the agent-benchmark contract) — this script makes no weld calls.

Choreography: reach -> descend -> two-stage grasp close -> lifts -> traverse -> knee-crawl pour
with a departed-fraction fill trigger -> recover -> return -> set-down (4 mm high; release drops
the vessel onto its slab) -> release. PASS gates: transferred >= 0.15, kept >= 0.15,
retention >= 0.90, spilled <= 0.05.

Quats are **xyzw** (isaaclab develop / warp convention) throughout.

Runs ONLY under the Newton venv:
  env_newton/bin/python -m robobench.suites.pouring.scripts.latte_bimanual_weld_smoke              # headless
  env_newton/bin/python -m robobench.suites.pouring.scripts.latte_bimanual_weld_smoke --viz kit    # GUI
"""

from __future__ import annotations

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--time_scale", type=float, default=4.0, help="multiply every phase duration")
parser.add_argument("--tilt_deg", type=float, default=118.0, help="MAX pour tilt of the pitcher [deg]; the fill trigger usually stops it earlier")
parser.add_argument("--mug_tilt_deg", type=float, default=15.0, help="the mug tilts this far TOWARD the pitcher during the pour [deg]")
parser.add_argument("--lip_clear", type=float, default=0.021, help="pitcher-lip clearance above the mug's low rim [m]")
parser.add_argument("--pour_margin", type=float, default=0.022, help="early-pour clearance margin [m] in the lip-descent law")
parser.add_argument("--pour_fraction", type=float, default=0.14, help="STOP pouring once this fraction of the milk has DEPARTED the pitcher (polled every step)")
parser.add_argument("--hold", type=float, default=2.0, help="extra settle time after the trajectory [s]")
parser.add_argument("--print_every", type=int, default=400, help="progress print period [steps]")
parser.add_argument("--max_steps", type=int, default=None, help="cap total steps (debugging)")
parser.add_argument("--max_dq", type=float, default=0.04, help="per-tick joint REFERENCE step clamp [rad]")
parser.add_argument("--lead_max", type=float, default=0.30, help="max lead of the commanded reference over the ACTUAL joints [rad]")
parser.add_argument("--grasp_pitch", type=float, default=30.0, help="downward tilt of the horizontal side grasps [deg]")
parser.add_argument("--feed", action="store_true", help="run on scene latte_feed: latte_auto plus 1.5-way liquid feedback (vessels weigh what they hold; adds a fluid-force readout to the status line)")
parser.add_argument("--auto", action="store_true", help="run on scene latte_auto (agent-benchmark grasping): welds engage/release AUTOMATICALLY from gripper proximity + closure — this smoke then makes NO scripted weld calls, validating the mechanic end-to-end")
parser.add_argument("--dump_states", type=str, default=None, help="record body_q + particle positions every --dump_every steps into this .npz for scripts/replay_render.py (videos render from replayed states — LIVE rendering corrupts the coupled physics)")
parser.add_argument("--dump_every", type=int, default=7, help="state-dump cadence [steps]; 7 ~= 30 fps at 200 Hz")
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
import warp as wp  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.controllers.differential_ik import DifferentialIKController  # noqa: E402
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402
from robobench.robots.franka import FrankaRobot  # noqa: E402

FPS = 200  # must match MpmSimCfg.dt
RENDER_EVERY = 4  # render/particle-push cadence [ticks]
SETDOWN_DROP = 0.004  # set-down targets stop this far above the table; release drops the vessel

# One env.step == one physics tick (see the 2b smoke for the rationale).
FrankaRobot.JOINT_CONTROL_DT = 1.0 / FPS


def _smoothstep(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


class Arm:
    """Per-arm DiffIK servo: solves toward a world-frame hand
    target from the ACTUAL joint state and returns the 9-wide joint-position child action."""

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
        self.tgt_p: torch.Tensor | None = None
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
        # Bounded-lead reference (see the 2b smoke: anchoring to actual joints caps speed).
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
    env_name = "pouring.latte_weld.bimanual_franka.joint"
    if args.auto:
        env_name = "pouring.latte_auto.bimanual_franka.joint"
    if args.feed:
        env_name = "pouring.latte_feed.bimanual_franka.joint"
        args.auto = True  # latte_feed inherits the auto-grasp mechanic; make the weld calls hands-off
    cfg = ENVS.get(env_name)()

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
        f"[latte2c] particles: coffee {scene.coffee.particles_per_object}, milk {scene.milk.particles_per_object}"
        f" | vessels dynamic (mug {c.mug_mass} kg, pitcher {c.pitcher_mass} kg), weld-at-grasp",
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

    Q_LEFT = q_ry(90.0 + args.grasp_pitch)
    Q_RIGHT = q_ry(-(90.0 + args.grasp_pitch))

    # --- grasp geometry (world; see the scene cfg + rigidproxy constants for measured numbers) ---
    px, py = c.pitcher_pos
    z_hat = torch.tensor([[0.0, 0.0, 1.0]], device=device)

    def hand_from_tip(tip_xyz: tuple, quat: torch.Tensor) -> torch.Tensor:
        return t3(*tip_xyz) - 0.113 * math_utils.quat_apply(quat, z_hat)

    # Pinch points at the handle bars; the pads stop just short of the bar surfaces.
    lh_grasp = hand_from_tip((-0.080, 0.0, 0.093), Q_LEFT)
    rh_grasp = hand_from_tip((px + 0.060, py, 0.095), Q_RIGHT)
    lh_hover = lh_grasp + torch.tensor([-0.06, 0.0, 0.05], device=device)
    rh_hover = rh_grasp + torch.tensor([0.07, 0.0, 0.05], device=device)
    # Two-stage close: pads stand off while the vessel is free, then finish onto the bar
    # surface (+0.5 mm) during the first quarter of the lift, once the weld carries the load.
    STANDOFF_MUG, GRIP_MUG_BAR = 0.011, 0.0085  # mug bar capsule r = 0.008
    STANDOFF_PITCHER, GRIP_PITCHER_BAR = 0.009, 0.0065  # pitcher bar capsule r = 0.006

    # --- choreography phases ---
    phases: list[tuple[str, float]] = [
        ("reach", 2.5),
        ("descend", 1.5),
        ("grasp", 1.0),
        ("mug_lift", 1.5),
        ("lift", 1.2),
        ("traverse", 1.8),
        ("pour", 8.0),  # 2b ran 6.0; lengthened for the two-stage knee-crawl ramp (see tilt_of)
        ("drain", 1.8),
        ("recover", 1.0),  # 2b ran 0.6 -> a 38 deg/s untilt whip that flings the lip dribble to
        # the table; 23 deg/s keeps the tail over the mouth (in-flight milk is ~20% of the pour)
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
    # Teleport hygiene: re-seed the MPM collider pose history after the FULL reset (scene AND
    # robot writes) so the backward finite difference sees no phantom (jump/dt) velocity.
    from robobench.suites.pouring.coupled_manager import NewtonCoupledMJWarpMPMManager

    NewtonCoupledMJWarpMPMManager.resync_collider_history()

    # --feed: net vertical fluid force per vessel (should read ~ -liquid weight at rest)
    fb_bodies = {}
    if args.feed:
        labels = [str(b or "") for b in NewtonCoupledMJWarpMPMManager._model.body_label]
        for vessel, suffix in (("M", "CoffeeCup"), ("P", "Pitcher")):
            fb_bodies[vessel] = next(i for i, b in enumerate(labels) if b.endswith(suffix))

    lh_p0, lh_q0 = (x.clone() for x in left.hand_pose_w())
    rh_p0, rh_q0 = (x.clone() for x in right.hand_pose_w())

    off_cup = off_mug = None
    rh_final = None
    welds_released = False
    theta_max = math.radians(args.tilt_deg)
    phi_max = math.radians(args.mug_tilt_deg)
    lip_local = torch.tensor([-(c.pitcher_r + c.pitcher_wall), 0.0, c.pitcher_h], device=device)
    rim_local = torch.tensor([[c.coffee_cup_r - 0.002, 0.0, c.coffee_cup_h]], device=device)  # low-rim pt
    drop = torch.tensor([[0.0, 0.0, SETDOWN_DROP]], device=device)
    mug_base = t3(0.0, 0.0, TABLE_TOP_Z)
    mug_carry = mug_base + torch.tensor([0.0, 0.0, 0.06], device=device)
    cup_start = t3(px, py, TABLE_TOP_Z)
    last_phase = ""
    err_l = err_r = rot_l = rot_r = 0.0
    time_shift = 0.0
    tilt_frozen: tuple[float, float] | None = None
    fill_surface = float("-inf")
    transfer_now = 0.0

    def lerp(a: torch.Tensor, b: torch.Tensor, s: float) -> torch.Tensor:
        return a + (b - a) * s

    KNEE = math.radians(85.0)  # sub-avalanche tilt: outflow is a controlled ooze

    def tilt_of(name: str, s: float) -> tuple[float, float]:
        if tilt_frozen is not None and name in ("pour", "drain"):
            return tilt_frozen
        if name == "pour":
            # Two-stage ramp: sprint to the knee, then crawl through it so the fill trigger
            # freezes during controlled ooze (surge landings are chaotic).
            if s <= 0.6:
                theta = KNEE * (s / 0.6)
            else:
                theta = KNEE + (theta_max - KNEE) * ((s - 0.6) / 0.4)
            return theta, phi_max * s
        if name == "drain":
            return theta_max, phi_max
        if name == "recover":
            th0, ph0 = tilt_frozen if tilt_frozen is not None else (theta_max, phi_max)
            return th0 * (1.0 - s), ph0 * (1.0 - s)
        return 0.0, 0.0

    def mug_target(name: str, s: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Scripted mug CARRY pose (drives the left hand through the weld offset)."""
        _, phi = tilt_of(name, s)
        if name in ("reach", "descend", "grasp", "release"):
            return mug_base, q4((0.0, 0.0, 0.0, 1.0))
        if name == "mug_lift":
            return lerp(mug_base, mug_carry, s), q4((0.0, 0.0, 0.0, 1.0))
        if name == "mug_down":
            # stop 4 mm high; the release drops the mug onto its slab
            return lerp(mug_carry, mug_base + drop, s), q4((0.0, 0.0, 0.0, 1.0))
        half = phi / 2.0
        return mug_carry, q4((0.0, math.sin(half), 0.0, math.cos(half)))

    def cup_target(name: str, s: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Scripted pitcher pose, barista-style (2b geometry; margin is now args.pour_margin)."""
        theta, _ = tilt_of(name, s)
        mp, mq = mug_target(name, s)
        low_rim = mp + math_utils.quat_apply(mq, rim_local)
        lx, lz = float(lip_local[0]), float(lip_local[2])
        lip_up = max(args.lip_clear, args.pour_margin + math.sin(theta) * lx + math.cos(theta) * lz)
        # lip aims deep past the rim, near the mouth center: tolerates a stream-width of
        # landing scatter on both sides
        lip_target = low_rim[0] + torch.tensor([-0.038, 0.0, lip_up], device=device)
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
            return lerp(high, cup_start + drop, s), quat
        return cup_start, quat

    def hand_target_from(obj_p: torch.Tensor, obj_q: torch.Tensor, off) -> tuple[torch.Tensor, torch.Tensor]:
        inv_q = math_utils.quat_inv(off[1])
        inv_p = -math_utils.quat_apply(inv_q, off[0])
        return math_utils.combine_frame_transforms(obj_p, obj_q, inv_p, inv_q)

    def vessel_tilt_deg(pose_w: torch.Tensor) -> float:
        up = math_utils.quat_apply(pose_w[:, 3:], z_hat)
        return float(torch.rad2deg(torch.acos(up[0, 2].clamp(-1.0, 1.0))))

    def status() -> str:
        return (
            f"transfer {transfer_now:.3f} (target {args.pour_fraction:.2f}) | fill {fill_surface * 100:4.1f} cm"
            f" | milk-in-pitcher {float(scene.milk_in_pitcher_fraction().mean()):.3f}"
            f" | retention {float(scene.retention_fraction().mean()):.3f}"
            f" | spilled {float(scene.spilled_fraction().mean()):.3f}"
            f" | err L {err_l * 100:4.1f} R {err_r * 100:4.1f} cm | rot L {rot_l:.2f} R {rot_r:.2f} rad"
            # REAL vessel tilts (weld-sag / contact-fight observable — scripted tilt is not truth)
            f" | real tilt M {vessel_tilt_deg(scene.mug_pose_w):5.1f} P {vessel_tilt_deg(scene.pitcher_pose_w):5.1f} deg"
            + (
                " | Ffluid M {:5.2f} P {:5.2f} N".format(
                    *(
                        float(wp.to_torch(NewtonCoupledMJWarpMPMManager._fluid_forces)[fb_bodies[v], 2])
                        for v in ("M", "P")
                    )
                )
                if fb_bodies
                else ""
            )
        )

    action = torch.zeros((1, env.robot.action_dim), device=device)
    dump_frames: dict[str, list] = {"body_q": [], "coffee": [], "milk": []}

    for step in range(total_steps):
        t = step / FPS + time_shift
        idx = next((i for i, edge in enumerate(t_edges) if t < edge), len(phases) - 1)
        name, _ = phases[idx]
        s = _smoothstep(1.0 - (t_edges[idx] - t) / max(durs[idx], 1e-9)) if t < t_edges[-1] else 1.0

        # pour trigger: freeze the tilt and jump the clock to `recover` once enough milk departed
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

        # fingers + weld release
        if name in ("reach", "descend"):
            left.grip = right.grip = 0.04
        elif name == "grasp":
            left.grip = max(STANDOFF_MUG, 0.04 - (0.04 - STANDOFF_MUG) * s)
            right.grip = max(STANDOFF_PITCHER, 0.04 - (0.04 - STANDOFF_PITCHER) * s)
        elif name == "mug_lift":
            # weld engaged on this phase's first tick (below) -> finish the close onto the bar
            left.grip = STANDOFF_MUG + (GRIP_MUG_BAR - STANDOFF_MUG) * min(1.0, s / 0.25)
        elif name == "lift":
            right.grip = STANDOFF_PITCHER + (GRIP_PITCHER_BAR - STANDOFF_PITCHER) * min(1.0, s / 0.25)
        elif name == "release":
            if not welds_released:
                if not args.auto:  # latte_auto: opening the fingers below releases automatically
                    scene.weld_vessel("mug", False)
                    scene.weld_vessel("pitcher", False)
                welds_released = True
                print(
                    f"  [weld] released | mug tilt {vessel_tilt_deg(scene.mug_pose_w):.1f} deg"
                    f" | pitcher tilt {vessel_tilt_deg(scene.pitcher_pose_w):.1f} deg",
                    flush=True,
                )
            left.grip = min(0.04, GRIP_MUG_BAR + (0.04 - GRIP_MUG_BAR) * s)
            right.grip = min(0.04, GRIP_PITCHER_BAR + (0.04 - GRIP_PITCHER_BAR) * s)

        # --- hand targets ---
        if name == "reach":
            lh_t, lh_q = lerp(lh_p0, lh_hover, s), qlerp(lh_q0, Q_LEFT, s)
        elif name == "descend":
            lh_t, lh_q = lerp(lh_hover, lh_grasp, s), Q_LEFT
        elif name in ("grasp", "release"):
            lh_t, lh_q = lh_grasp, Q_LEFT
        else:
            if off_mug is None:
                hp, hq = left.hand_pose_w()
                # ride-along offsets anchor to SCRIPTED home poses, not read-back body poses
                # (root_link_quat_w carries per-body yaw offsets on free trimesh bodies)
                off_mug = math_utils.subtract_frame_transforms(hp, hq, mug_base, q4((0.0, 0.0, 0.0, 1.0)))
                if not args.auto:  # latte_auto: the scene's proximity+closure mechanic already engaged it
                    scene.weld_vessel("mug", True)
                print(f"  [weld] mug carried | hand err L {err_l * 100:.1f} cm", flush=True)
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
                # scripted home anchor, same rationale as off_mug
                off_cup = math_utils.subtract_frame_transforms(hp, hq, cup_start, q4((0.0, 0.0, 0.0, 1.0)))
                if not args.auto:
                    scene.weld_vessel("pitcher", True)
                print("  [weld] pitcher carried", flush=True)
            cp_t, cq_t = cup_target(name, s)
            rh_t, rh_q = hand_target_from(cp_t, cq_t, off_cup)
        else:  # mug_down / release: hold where the pitcher was set down
            if rh_final is None:
                rh_final = tuple(x.clone() for x in right.hand_pose_w())
            rh_t, rh_q = rh_final

        action[:, 0:9] = left.solve(lh_t, lh_q, args.max_dq)
        action[:, 9:18] = right.solve(rh_t, rh_q, args.max_dq)
        env.step(action, render=render_on and step % RENDER_EVERY == 0)
        # No vessel pose writes: the welds carry them, scene.post_step tracks their actual poses.

        if args.dump_states and step % args.dump_every == 0:
            import numpy as _np
            import warp as _wp

            dump_frames["body_q"].append(
                _wp.to_torch(NewtonCoupledMJWarpMPMManager._state_0.body_q).detach().cpu().numpy().astype(_np.float32)
            )
            dump_frames["coffee"].append(scene.coffee.data.nodal_pos_w.torch[0].detach().cpu().numpy().astype(_np.float16))
            dump_frames["milk"].append(scene.milk.data.nodal_pos_w.torch[0].detach().cpu().numpy().astype(_np.float16))

        err_l, rot_l = left.errors()
        err_r, rot_r = right.errors()
        if render_on and step % RENDER_EVERY == 0:
            scene.push_particle_visuals()

        if step % 100 == 0 and bool(torch.isnan(left.art.data.joint_pos.torch).any() | torch.isnan(right.art.data.joint_pos.torch).any()):
            print(f"LATTE-BIMANUAL-WELD FAIL | NaN joint state at step {step} (rigid solver exploded)", flush=True)
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
    mug_tilt = vessel_tilt_deg(scene.mug_pose_w)
    pitcher_tilt = vessel_tilt_deg(scene.pitcher_pose_w)
    ok = transfer >= 0.15 and keep >= 0.15 and retention >= 0.90 and spilled <= 0.05
    print(
        f"LATTE-BIMANUAL-WELD {'PASS' if ok else 'FAIL'} | milk transferred {transfer:.3f}"
        f" (>= 0.15) | milk kept in pitcher {keep:.3f} (>= 0.15)"
        f" | retention {retention:.3f} (>= 0.90) | spilled {spilled:.3f} (<= 0.05)"
        f" | final err L {err_l * 100:.1f} R {err_r * 100:.1f} cm"
        f" | vessels upright: mug {mug_tilt:.1f} deg, pitcher {pitcher_tilt:.1f} deg",
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

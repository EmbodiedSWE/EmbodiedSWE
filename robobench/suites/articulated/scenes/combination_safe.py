"""CombinationSafeScene — discover a hidden 3-number combination by FEEL, then open the safe.

The object world: a steel safe box (kinematic, open front) whose front is covered by a
hinged door. On the door: a spoked rotary DIAL (revolute, free-spinning with friction)
and a lever HANDLE (revolute). Inside: a prize block.
**Goal (carried here, no task layer): work out the secret combination from the dial's
mechanical clicks, enter it (right to A, left to B, right to C), turn the handle, open
the door, and take the prize block out onto the table.**

The combination is 3 distinct dial numbers, RANDOM EVERY EPISODE, and never exposed by
any query — `describe()` explains the mechanics but not the numbers. The only clue is
physical: when the dial reading passes a combination number while turning in that
number's direction, the mechanism BRAKES the dial for the few degrees around the number
(a detent "click"). A slow, steady sweep shows the clicks as sharp dips in rotation
rate; a fast spin smears them into control noise (deliberate speed/information
tradeoff, calibrated in the smoke).

Lock rules (post_step state machine, per env):
  - stage 0→1: dial comes to REST within `stop_tol_deg` of A, having approached
    turning RIGHT (reading decreasing).
  - stage 1→2: rest at B approached turning LEFT; stage 2→3: rest at C turning RIGHT.
  - in stages 1/2, sweeping PAST the stage target in its direction without stopping
    resets the sequence to stage 0 (as in a real safe).
  - resting at a WRONG number is harmless (exploration never punished at stage 0).
  - while stage < 3 the handle is bolted (strong spring to 0); once entered, the handle
    turns; while the handle is below `handle_open_deg` the door is bolted shut.

Bodies are plain rigid objects + authored USD joints (the proven lid-hinge pattern);
the dial/handle/door "friction" and the bolts are external torques applied in
`post_step` (always overwritten each substep). External drivers (NullRobot smoke, RL)
add torque through `scene.dial_drive` / `handle_drive` / `door_drive` tensors instead
of calling set_external_force_and_torque themselves — post_step owns that buffer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg, info, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


def _twist_deg(q_rel: torch.Tensor, axis: int) -> torch.Tensor:
    """Signed twist (deg, in [-180, 180]) of a relative quaternion about one local axis
    (swing-twist decomposition; axis: 0=x, 1=y, 2=z)."""
    w = q_rel[:, 0]
    a = q_rel[:, 1 + axis]
    ang = 2.0 * torch.atan2(a, w)
    ang = torch.rad2deg(ang)
    return (ang + 180.0) % 360.0 - 180.0


@dataclass
class CombinationSafeSceneCfg(BaseCfg):
    """Config for `CombinationSafeScene`."""

    # --- tunable: difficulty dials -----------------------------------------------------------
    dial_radius: float = tunable(0.07)  # bigger dial = easier fingertip work (curriculum knob)
    stop_tol_deg: float = tunable(6.0)  # rest-within-this of a target counts as "stopped at it"
    pass_slack_deg: float = tunable(14.0)  # overshoot past a stage target that triggers the reset
    # Measured (smoke 2026-07-14, brake 0.05/window 3.0): a 30 deg/s sweep STOPS DEAD in
    # the window (contrast 0.00 — unambiguous) but a 240 deg/s sweep still dipped to 0.11x
    # median rate. Window narrowed so fast sweeps smear closer to noise while slow stays
    # crisp; brake kept strong enough to be a real drag against a firm turner.
    detent_window_deg: float = tunable(2.5)  # half-width of the braking zone around each number
    detent_brake: float = tunable(0.05)  # brake torque magnitude (N*m) — the "click"
    # Plant scaling (v4 finding): with a light dial (I~4e-4) any torque that beats the
    # brake also spins the dial to 500+ deg/s in one substep. A heavy, well-damped dial
    # (mass 0.6 -> I~1.5e-3, friction 0.08) turns at ~36 deg/s under a 0.05 N*m push —
    # finger-scale torques give finger-scale speeds, and the detent is a crisp stall.
    dial_mass: float = tunable(0.6)
    dial_friction: float = tunable(0.08)  # viscous friction on the dial (N*m per rad/s)
    handle_open_deg: float = tunable(60.0)  # handle angle that withdraws the door bolt
    door_open_deg: float = tunable(70.0)  # door angle that counts as open
    # A "stop" must be truly stationary: at 0.08 rad/s (4.6 deg/s) threshold, slow entry
    # approaches and detent stall-creep (~2 deg/s) registered PHANTOM stops that reset
    # the sequence mid-entry (v2 smoke). 0.03 rad/s = 1.7 deg/s + 0.75 s dwell fixes it.
    rest_speed: float = tunable(0.03)  # |omega| below this (rad/s) counts toward "resting"
    rest_dwell_steps: int = tunable(90)  # consecutive resting substeps (~0.75 s) to register a stop

    # --- tunable: placement -------------------------------------------------------------------
    surface_z: float = tunable(0.0)  # work-surface height; 0 = on the ground (null smoke)
    safe_pos: tuple = tunable((0.0, 0.15))  # safe centre on the surface (door faces -y)

    # --- tunable: presentation ------------------------------------------------------------------
    # Demo-only stage lamps + visual hand (human-facing renders). NEVER enable for
    # agent evals — the lamps leak lock state.
    demo_lamps: bool = tunable(False)

    # --- info: structure ----------------------------------------------------------------------
    bench_size: tuple = info((1.1, 0.9))
    outer: tuple = info((0.42, 0.34, 0.40))  # safe outer (x, y, z)
    wall_t: float = info(0.02)
    door_t: float = info(0.02)
    door_gap: float = info(0.003)  # closed-door clearance off the front rim
    dial_th: float = info(0.03)  # dial cylinder thickness (its axis is y)
    spoke_len: float = info(0.16)  # full length of each cross spoke (tips clear the mark nub)
    spoke_w: float = info(0.016)
    handle_len: float = info(0.12)
    prize_size: float = info(0.07)
    combo_len: int = info(3)
    number_step: int = info(10)  # dial numbers live on multiples of this (deg)
    min_sep_deg: float = info(30.0)  # min separation between combination numbers

    # Derived (filled in __post_init__): dial/handle mount points in DOOR-local coords.
    dial_mount: tuple = field(default=None, init=False)
    handle_mount: tuple = field(default=None, init=False)

    def __post_init__(self) -> None:
        # Door local frame: x along width, y = thickness (normal), z = height.
        # Dial upper-centre, handle lower-right. Components hang off the -y (outer) face.
        self.dial_mount = (0.0, -(self.door_t / 2 + self.dial_th / 2 + 0.004), 0.07)
        self.handle_mount = (0.10, -(self.door_t / 2 + 0.015 + 0.004), -0.09)


@SCENES.register("safe")
class CombinationSafeScene(BaseScene):
    cfg: CombinationSafeSceneCfg

    def __init__(self, cfg: CombinationSafeSceneCfg | None = None) -> None:
        super().__init__(cfg or CombinationSafeSceneCfg())

    def _spoke_y(self, k: int) -> float:
        """Dial-local y offset of spoke k — each spoke in its OWN plane (see assets)."""
        c = self.cfg
        return -(c.dial_th / 2 + c.spoke_w / 2 + 0.002) - k * (c.spoke_w + 0.002)

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        W, D, H = c.outer
        t = c.wall_t
        cx, cy = c.safe_pos
        z0 = c.surface_z

        steel = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.37, 0.40))
        dark = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.17))
        brass = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.62, 0.25))

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
        }
        if z0 > 0:
            out["bench"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=sim_utils.CuboidCfg(
                    size=(c.bench_size[0], c.bench_size[1], z0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(cx, cy, z0 / 2)),
            )

        # Safe body = 5 kinematic walls, front (-y) open. Interior floor top at z0 + t.
        parts = {
            "bottom": ((W, D, t), (cx, cy, z0 + t / 2)),
            "top": ((W, D, t), (cx, cy, z0 + H - t / 2)),
            "back": ((W, t, H - 2 * t), (cx, cy + D / 2 - t / 2, z0 + H / 2)),
            "left": ((t, D - t, H - 2 * t), (cx - W / 2 + t / 2, cy - t / 2, z0 + H / 2)),
            "right": ((t, D - t, H - 2 * t), (cx + W / 2 - t / 2, cy - t / 2, z0 + H / 2)),
        }
        for name, (size, pos) in parts.items():
            out[f"safe_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Safe_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=steel,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
            )

        # Door: covers the front opening, hinged at its LEFT vertical edge (axis z).
        # Identity joint frames -> angle 0 = closed; opening swings outward = NEGATIVE.
        dy = cy - D / 2 - c.door_t / 2 - c.door_gap
        dz = z0 + H / 2
        out["door"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Door",
            spawn=sim_utils.CuboidCfg(
                size=(W, c.door_t, H - 0.01),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.5),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=dark,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(cx, dy, dz)),
        )

        # Dial: cylinder with its axis along y (the door normal), on the door's outer face.
        mx, my, mz = c.dial_mount
        out["dial"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Dial",
            spawn=sim_utils.CylinderCfg(
                radius=c.dial_radius, height=c.dial_th, axis="Y",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.dial_mass),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=brass,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(cx + mx, dy + my, dz + mz)),
        )
        # Two crossing spokes on the dial face (a "+" the fingertips can push on).
        # STACKED in separate y planes: co-planar they overlap at the crossing, and the
        # two bodies (each jointed to the dial but not to each other) grind on that
        # permanent penetration — measured as pseudo-random dial jams in the v5 smoke.
        for name, size, y_off in (
            ("spoke_x", (c.spoke_len, c.spoke_w, c.spoke_w), self._spoke_y(0)),
            ("spoke_z", (c.spoke_w, c.spoke_w, c.spoke_len), self._spoke_y(1)),
        ):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.capitalize(),
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=brass,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(cx + mx, dy + my + y_off, dz + mz)),
            )
        # Pointer mark: a small kinematic nub on the door just above the dial (reading
        # reference for look(); no joint).
        out["dial_mark"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Dial_mark",
            spawn=sim_utils.CuboidCfg(
                size=(0.012, 0.012, 0.03),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(cx + mx, dy + my, dz + mz + c.dial_radius + 0.022)),
        )

        # Handle: lever extending +x from its pivot on the door, axis y.
        hx, hy, hz = c.handle_mount
        out["handle"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Handle",
            spawn=sim_utils.CuboidCfg(
                size=(c.handle_len, 0.022, 0.022),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.12),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=brass,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(cx + hx + c.handle_len / 2 - 0.011, dy + hy, dz + hz)),
        )

        # Prize block on the safe floor.
        out["prize"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Prize",
            spawn=sim_utils.CuboidCfg(
                size=(c.prize_size,) * 3,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.15, 0.55)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(cx, cy, z0 + t + c.prize_size / 2 + 0.002)),
        )
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n = env.num_envs
        dev = env.device
        self.door: RigidObject = env.iscene["door"]
        self.dial: RigidObject = env.iscene["dial"]
        self.handle: RigidObject = env.iscene["handle"]
        self.prize: RigidObject = env.iscene["prize"]
        self.spokes = [env.iscene["spoke_x"], env.iscene["spoke_z"]]
        self.mark = env.iscene["dial_mark"]
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        self._build_dial_numbers()
        # Demo-only stage lamps (cfg.demo_lamps): visual indicators for humans watching
        # renders; NEVER enabled by the eval harness (would leak lock state).
        self._demo_lamps = self.cfg.demo_lamps
        if self._demo_lamps:
            self._build_stage_lamps()
            self._build_demo_hand()
        # External drive inputs (smoke / RL write these; post_step consumes them).
        self.dial_drive = torch.zeros(n, device=dev)
        self.handle_drive = torch.zeros(n, device=dev)
        self.door_drive = torch.zeros(n, device=dev)
        # Lock state.
        self._combo = torch.zeros(n, self.cfg.combo_len, device=dev)  # numbers (deg)
        self._stage = torch.zeros(n, dtype=torch.long, device=dev)
        self._theta = torch.zeros(n, device=dev)  # unwrapped dial angle (deg)
        self._prev_read = torch.zeros(n, device=dev)
        self._omega = torch.zeros(n, device=dev)  # deg/substep -> deg/s scaled
        self._rest_ctr = torch.zeros(n, dtype=torch.long, device=dev)
        self._last_dir = torch.zeros(n, device=dev)  # +1 left (increasing), -1 right
        self._primed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._primed_theta = torch.zeros(n, device=dev)
        # Reading of the last REGISTERED stop + armed flag: re-rests at (about) the same
        # reading are the same stop, not a new event (a stop is one decision, however long
        # it is held). Armed again once the dial clearly leaves that reading.
        self._stop_read = torch.zeros(n, device=dev)
        self._stop_armed = torch.ones(n, dtype=torch.bool, device=dev)
        # ENTERED latch: once the full combination registers, the bolt stays retracted
        # for the rest of the episode (real safes: wheel positions stop mattering once
        # the fence drops). Pre-latch, post-entry dial jiggle from handle/door contact
        # was re-bolting the handle mid-open and blanking the demo lamps.
        self._entered = torch.zeros(n, dtype=torch.bool, device=dev)
        # OPENED latch: once the entered safe's door has swung past door_open_deg the
        # episode counts as opened, even if the frictionless hinge later coasts the
        # door a few degrees back (demo5: success flaked exactly this way).
        self._opened = torch.zeros(n, dtype=torch.bool, device=dev)
        # Per-number detent-window flag (this substep): demo lamps flash amber on it.
        self._click_win = torch.zeros(n, self.cfg.combo_len, dtype=torch.bool, device=dev)
        # Lock event trace (env 0 only; smoke/debug): list of dicts, ring-buffered.
        self._trace: list[dict] = []
        self._trace_step = 0

    # 7-segment font for the dial-face digits (geometry prims: two textured-quad
    # attempts rendered as a blank disc under RTX, while plain displayColor prims
    # provably render — ticks, lamps, hand, ink).
    _SEGS = {"0": "ABCDEF", "1": "BC", "2": "ABGED", "3": "ABGCD", "4": "FGBC",
             "5": "AFGCD", "6": "AFGECD", "7": "ABC", "8": "ABCDEFG", "9": "ABCFGD"}

    def _build_dial_numbers(self) -> None:
        """Numbered dial face from geometry (children of Dial — rotate with it):
        black ticks every 10 deg, long ticks + 7-segment NUMBERS every 30 deg, so
        viewers watch the dial stop AT NUMBERS under the fixed red pointer. Number N
        sits at rest angle -N (measured +z -> +x): a +N twist about local y carries
        it under the top pointer exactly when the reading is N."""
        import omni.usd
        from pxr import Gf, UsdGeom

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        y0 = -c.dial_th / 2 - 0.0015
        dark = Gf.Vec3f(0.08, 0.08, 0.10)
        H, W_d, T = 0.011, 0.0062, 0.0020  # digit height/width, segment thickness
        seg_geo = {  # (du, dv, len_u, len_v) in digit-local coords (u tangent, v radial)
            "A": (0.0, H / 2, W_d, T), "G": (0.0, 0.0, W_d, T), "D": (0.0, -H / 2, W_d, T),
            "B": (W_d / 2, H / 4, T, H / 2), "C": (W_d / 2, -H / 4, T, H / 2),
            "E": (-W_d / 2, -H / 4, T, H / 2), "F": (-W_d / 2, H / 4, T, H / 2),
        }
        for i in range(self.env.num_envs):
            # Idempotency guard (multi-env server boot, 2026-07-17): with num_envs>1
            # isaaclab composes env_1.. from env_0 by reference, so env_0's decoration
            # prims are ALREADY visible under every env when we get here — Define
            # returns the composed prim and AddXformOp hard-fails on its existing ops.
            # The decorations are identical local offsets, so authoring env_0 only is
            # exactly right (they propagate through the reference).
            if stage.GetPrimAtPath(f"/World/envs/env_{i}/Dial/tick_0").IsValid():
                continue
            for d10 in range(36):  # ticks every 10 deg
                deg = d10 * 10
                th = math.radians(-deg)
                major = deg % 30 == 0
                ln = 0.016 if major else 0.008
                r_t = c.dial_radius - ln / 2 - 0.002
                tick = UsdGeom.Cube.Define(stage, f"/World/envs/env_{i}/Dial/tick_{d10}")
                tick.CreateSizeAttr(1.0)
                xf = UsdGeom.Xformable(tick)
                xf.AddTranslateOp().Set(Gf.Vec3d(r_t * math.sin(th), y0, r_t * math.cos(th)))
                xf.AddRotateYOp().Set(math.degrees(th))
                xf.AddScaleOp().Set(Gf.Vec3f(0.0035 if major else 0.002, 0.001, ln))
                tick.CreateDisplayColorAttr([dark])
            for n30 in range(12):  # 7-seg numbers every 30 deg
                num = n30 * 30
                th = math.radians(-num)
                sin, cos = math.sin(th), math.cos(th)
                r_c = c.dial_radius - 0.016 - 0.004 - 0.010  # inboard of the long tick
                chars = str(num)
                total_w = len(chars) * W_d + (len(chars) - 1) * 0.002
                for ci, ch in enumerate(chars):
                    cu = -total_w / 2 + W_d / 2 + ci * (W_d + 0.002)
                    for sname in self._SEGS[ch]:
                        du, dv, lu, lv = seg_geo[sname]
                        u = cu + du
                        v = r_c + dv
                        seg = UsdGeom.Cube.Define(
                            stage, f"/World/envs/env_{i}/Dial/num{num}_{ci}_{sname}")
                        seg.CreateSizeAttr(1.0)
                        xf = UsdGeom.Xformable(seg)
                        # Screen right = +x when viewing the -y-facing door (verified
                        # against the handle at handle_mount x=+0.10 rendering bottom-
                        # RIGHT), so +u maps STRAIGHT to +x. The one-run u-negation
                        # "fix" (demo7) was itself the mirror; demo6's order was right.
                        xf.AddTranslateOp().Set(Gf.Vec3d(
                            v * sin + u * cos, y0, v * cos - u * sin))
                        xf.AddRotateYOp().Set(math.degrees(th))
                        xf.AddScaleOp().Set(Gf.Vec3f(lu, 0.001, lv))
                        seg.CreateDisplayColorAttr([dark])

    def _build_demo_hand(self) -> None:
        """Demo-only visual 'hand' gripping the top spoke (children of Dial — they
        orbit with it, so the dial visibly is TURNED by something rather than by
        magic). Purely visual; gated by the same flag as the lamps."""
        import omni.usd
        from pxr import Gf, UsdGeom

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        grip_z = c.dial_radius * 0.78
        grey = Gf.Vec3f(0.42, 0.44, 0.48)
        for i in range(self.env.num_envs):
            # Same idempotency guard as _build_dial_numbers (env_1.. compose env_0).
            if stage.GetPrimAtPath(f"/World/envs/env_{i}/Dial/hand_palm").IsValid():
                continue
            for name, pos, size in (
                    ("finger_a", (0.016, -c.dial_th / 2 - 0.014, grip_z), (0.012, 0.034, 0.030)),
                    ("finger_b", (-0.016, -c.dial_th / 2 - 0.014, grip_z), (0.012, 0.034, 0.030)),
                    ("palm", (0.0, -c.dial_th / 2 - 0.043, grip_z), (0.05, 0.026, 0.040)),
                    ("wrist", (0.0, -c.dial_th / 2 - 0.075, grip_z), (0.024, 0.045, 0.024))):
                cube = UsdGeom.Cube.Define(stage, f"/World/envs/env_{i}/Dial/hand_{name}")
                cube.CreateSizeAttr(1.0)
                xf = UsdGeom.Xformable(cube)
                xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
                xf.AddScaleOp().Set(Gf.Vec3f(*size))
                cube.CreateDisplayColorAttr([grey])

    def _build_stage_lamps(self) -> None:
        """Three demo lamps on the safe top (children of Safe_top, visual only); lamp k
        turns green when combination stage k is entered. post_step refreshes colors."""
        import omni.usd
        from pxr import Gf, UsdGeom

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        self._lamps = []
        for i in range(self.env.num_envs):
            row = []
            for k in range(c.combo_len):
                path = f"/World/envs/env_{i}/Safe_top/lamp_{k}"
                # env_1.. compose env_0's lamps: keep the handle (color Sets author
                # per-env overrides) but only add xform ops on genuinely new prims.
                fresh = not stage.GetPrimAtPath(path).IsValid()
                lamp = UsdGeom.Sphere.Define(stage, path)
                lamp.CreateRadiusAttr(0.022)
                if fresh:
                    xf = UsdGeom.Xformable(lamp)
                    xf.AddTranslateOp().Set(Gf.Vec3d(
                        -0.07 + 0.07 * k, -c.outer[1] / 2 + 0.05, c.wall_t / 2 + 0.018))
                lamp.CreateDisplayColorAttr([Gf.Vec3f(0.30, 0.04, 0.04)])
                row.append(lamp)
            self._lamps.append(row)
        self._lamp_shown = [None] * self.env.num_envs

    def _refresh_stage_lamps(self) -> None:
        """Lamp language a naive viewer reads without any text:
        - lamp k FLASHES AMBER while the dial is dragging through secret number k's
          detent (the visible counterpart of the click the robot feels);
        - lamp k turns SOLID GREEN when number k registers (stage passes k);
        - a wrong stop / overshoot snaps the greens back to red — visibly a reset;
        - all three green (latched once entered) -> handle turns -> door opens."""
        from pxr import Gf

        for e in range(self.env.num_envs):
            if bool(self._entered[e]):
                key = "entered"
            else:
                s = int(self._stage[e])
                clicks = tuple(bool(v) for v in self._click_win[e])
                key = (s, clicks)
            if key == self._lamp_shown[e]:
                continue
            self._lamp_shown[e] = key
            for k, lamp in enumerate(self._lamps[e]):
                if key == "entered" or k < int(self._stage[e]):
                    col = Gf.Vec3f(0.05, 0.80, 0.10)  # registered: solid green
                elif bool(self._click_win[e, k]):
                    col = Gf.Vec3f(0.95, 0.65, 0.05)  # in this number's detent: amber
                else:
                    col = Gf.Vec3f(0.30, 0.04, 0.04)  # pending: dark red
                lamp.GetDisplayColorAttr().Set([col])

    def _author_joints(self) -> None:
        """Per env: door hinge (z, on the left front edge), dial spindle (y, on the door),
        handle pivot (y, on the door), and fixed joints dial<->spokes."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        W, D, H = c.outer
        t = c.wall_t
        stage = omni.usd.get_context().get_stage()
        mx, my, mz = c.dial_mount
        hx, hy, hz = c.handle_mount
        # Door-local hinge point: its left edge. Left-wall-local counterpart: front outer
        # edge (wall centre y = -t/2 relative to safe centre; door plane sits door_gap +
        # door_t/2 in front of y = -D/2).
        door_y_off = -D / 2 - c.door_t / 2 - c.door_gap  # door centre y, safe-local
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"

            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/door_hinge")
            j.CreateBody0Rel().SetTargets([f"{base}/Safe_left"])
            j.CreateBody1Rel().SetTargets([f"{base}/Door"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Z")
            # Left wall centre (safe-local): (-W/2 + t/2, -t/2, H/2). Hinge (safe-local):
            # (-W/2, door_y_off, H/2) -> wall-local:
            j.CreateLocalPos0Attr(Gf.Vec3f(-t / 2, door_y_off + t / 2, 0.0))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(-W / 2, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-115.0)
            j.CreateUpperLimitAttr(0.0)

            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/dial_spindle")
            j.CreateBody0Rel().SetTargets([f"{base}/Door"])
            j.CreateBody1Rel().SetTargets([f"{base}/Dial"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(mx, my, mz))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            # no limits: free spinning

            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/handle_pivot")
            j.CreateBody0Rel().SetTargets([f"{base}/Door"])
            j.CreateBody1Rel().SetTargets([f"{base}/Handle"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(hx, hy, hz))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            # Pivot at the lever's -x end.
            j.CreateLocalPos1Attr(Gf.Vec3f(-c.handle_len / 2 + 0.011, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(0.0)
            j.CreateUpperLimitAttr(90.0)

            for spoke, sname in ((0, "Spoke_x"), (1, "Spoke_z")):
                j = UsdPhysics.FixedJoint.Define(stage, f"{base}/spoke_fix_{spoke}")
                j.CreateBody0Rel().SetTargets([f"{base}/Dial"])
                j.CreateBody1Rel().SetTargets([f"{base}/{sname}"])
                j.CreateCollisionEnabledAttr(False)
                j.CreateLocalPos0Attr(Gf.Vec3f(0.0, self._spoke_y(spoke), 0.0))
                j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            # Fix the pointer mark to the door.
            j = UsdPhysics.FixedJoint.Define(stage, f"{base}/mark_fix")
            j.CreateBody0Rel().SetTargets([f"{base}/Door"])
            j.CreateBody1Rel().SetTargets([f"{base}/Dial_mark"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateLocalPos0Attr(Gf.Vec3f(mx, my, mz + c.dial_radius + 0.022))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # ----- reset --------------------------------------------------------------------------------
    def _closed_pose(self, name: str, m: int, env_ids: torch.Tensor) -> torch.Tensor:
        """Root state (m, 13) for a body in the everything-closed home layout."""
        c = self.cfg
        W, D, H = c.outer
        cx, cy = c.safe_pos
        z0 = c.surface_z
        dy = cy - D / 2 - c.door_t / 2 - c.door_gap
        dz = z0 + H / 2
        mx, my, mz = c.dial_mount
        hx, hy, hz = c.handle_mount
        pos = {
            "door": (cx, dy, dz),
            "dial": (cx + mx, dy + my, dz + mz),
            "spoke_x": (cx + mx, dy + my + self._spoke_y(0), dz + mz),
            "spoke_z": (cx + mx, dy + my + self._spoke_y(1), dz + mz),
            "mark": (cx + mx, dy + my, dz + mz + c.dial_radius + 0.022),
            "handle": (cx + hx + c.handle_len / 2 - 0.011, dy + hy, dz + hz),
            "prize": (cx, cy, z0 + c.wall_t + c.prize_size / 2 + 0.002),
        }[name]
        st = torch.zeros(m, 13, device=self.env.device)
        st[:, 0:3] = self.env_origins[env_ids] + torch.tensor(pos, device=self.env.device)
        st[:, 3] = 1.0
        return st

    def reset(self, env_ids: torch.Tensor) -> None:
        """Everything closed and at home; a fresh random combination per env."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        for name, body in (("door", self.door), ("dial", self.dial),
                           ("spoke_x", self.spokes[0]), ("spoke_z", self.spokes[1]),
                           ("mark", self.mark), ("handle", self.handle),
                           ("prize", self.prize)):
            body.write_root_state_to_sim(self._closed_pose(name, m, env_ids), env_ids)

        # Fresh combination: distinct numbers on the number grid, pairwise >= min_sep apart.
        n_choices = 360 // c.number_step
        for row, e in enumerate(env_ids.tolist()):
            while True:
                nums = torch.randperm(n_choices)[: c.combo_len].float() * c.number_step
                d = (nums.view(-1, 1) - nums.view(1, -1)).abs() % 360.0
                d = torch.minimum(d, 360.0 - d)
                d = d + torch.eye(c.combo_len) * 999.0
                if float(d.min()) >= c.min_sep_deg:
                    break
            self._combo[e] = nums.to(dev)
        self._stage[env_ids] = 0
        self._entered[env_ids] = False
        self._opened[env_ids] = False
        self._click_win[env_ids] = False
        self._theta[env_ids] = 0.0
        self._prev_read[env_ids] = 0.0
        self._omega[env_ids] = 0.0
        self._rest_ctr[env_ids] = 0
        self._last_dir[env_ids] = 0.0
        self._primed[env_ids] = False
        self._stop_read[env_ids] = 0.0
        self._stop_armed[env_ids] = True
        self.dial_drive[env_ids] = 0.0
        self.handle_drive[env_ids] = 0.0
        self.door_drive[env_ids] = 0.0

    # ----- angles -------------------------------------------------------------------------------
    def _rel_quat(self, child: RigidObject, parent: RigidObject) -> torch.Tensor:
        from isaaclab.utils.math import quat_inv, quat_mul

        return quat_mul(quat_inv(parent.data.root_quat_w), child.data.root_quat_w)

    def dial_reading_deg(self) -> torch.Tensor:
        """The number under the pointer, in [0, 360). Physical (a real dial shows this)."""
        return _twist_deg(self._rel_quat(self.dial, self.door), 1) % 360.0

    def handle_angle_deg(self) -> torch.Tensor:
        return _twist_deg(self._rel_quat(self.handle, self.door), 1).abs()

    def door_angle_deg(self) -> torch.Tensor:
        return _twist_deg(self.door.data.root_quat_w, 2).abs()

    # ----- lock mechanics (every substep) --------------------------------------------------------
    def post_step(self) -> None:
        c = self.cfg
        dev = self.env.device
        n = self.env.num_envs
        dt = self.env.dt

        read = self.dial_reading_deg()
        d = (read - self._prev_read + 180.0) % 360.0 - 180.0  # wrapped step delta (deg)
        self._prev_read = read
        self._theta = self._theta + d
        omega = d / dt  # deg/s
        self._omega = 0.7 * self._omega + 0.3 * omega  # light smoothing

        moving = self._omega.abs() > math.degrees(c.rest_speed)
        self._last_dir = torch.where(
            self._omega.abs() > 20.0, torch.sign(self._omega), self._last_dir)
        self._rest_ctr = torch.where(moving, torch.zeros_like(self._rest_ctr),
                                     self._rest_ctr + 1)
        # Re-arm the stop latch once the dial has clearly LEFT the last stop's reading;
        # until then, re-rests in the same spot are the same stop (one decision), not a
        # new event.
        away = (read - self._stop_read + 180.0) % 360.0 - 180.0
        self._stop_armed = self._stop_armed | (moving & (away.abs() > 1.5 * c.stop_tol_deg))
        rested = (self._rest_ctr == c.rest_dwell_steps) & self._stop_armed
        self._stop_read = torch.where(rested, read, self._stop_read)
        self._stop_armed = self._stop_armed & ~rested

        # Direction per stage: right (reading decreasing, dir=-1) for entries 0 and 2,
        # left (+1) for entry 1.
        stage_c = self._stage.clamp(max=c.combo_len - 1)
        target = self._combo.gather(1, stage_c.view(-1, 1)).squeeze(1)
        want_dir = torch.where(stage_c == 1,
                               torch.ones(n, device=dev), -torch.ones(n, device=dev))
        dist = (read - target + 180.0) % 360.0 - 180.0  # signed deg from target

        entering = self._stage < c.combo_len
        # STOPS drive the sequence (a rest event >= dwell). A stop at the current target
        # approached in the wanted direction advances. Any OTHER deliberate stop resets —
        # with re-entry (KMP-style): a wrong stop that itself matches the FIRST number
        # approached rightward restarts at stage 1. This keeps exploration honest: sweep
        # rests at arbitrary readings simply clear the lock instead of silently
        # accumulating stale progress (bug found by the first smoke run).
        adv = entering & rested & (dist.abs() <= c.stop_tol_deg) & (self._last_dir == want_dir)
        dist_a = (read - self._combo[:, 0] + 180.0) % 360.0 - 180.0
        matches_a = (dist_a.abs() <= c.stop_tol_deg) & (self._last_dir == -1.0)
        wrong = entering & rested & ~adv
        self._stage = torch.where(adv, self._stage + 1, self._stage)
        self._entered = self._entered | (self._stage >= c.combo_len)
        self._opened = self._opened | (self._entered
                                       & (self.door_angle_deg() >= c.door_open_deg))
        self._stage = torch.where(wrong & matches_a, torch.ones_like(self._stage), self._stage)
        self._stage = torch.where(wrong & ~matches_a, torch.zeros_like(self._stage), self._stage)
        self._primed = self._primed & ~(adv | wrong)
        self._trace_step += 1
        if bool(rested[0]) or bool(adv[0]):
            self._trace.append({"t": self._trace_step, "read": round(float(read[0]), 1),
                                "dir": float(self._last_dir[0]),
                                "stage": int(self._stage[0]),
                                "adv": bool(adv[0]), "wrong": bool(wrong[0])})
            if len(self._trace) > 60:
                self._trace.pop(0)

        # Pass-without-stop reset (stages 1..): prime on crossing the target in the wanted
        # direction, reset once overshoot exceeds pass_slack while still moving.
        prev_dist = (dist - d + 180.0) % 360.0 - 180.0
        crossed = entering & (self._stage > 0) & moving & (torch.sign(self._omega) == want_dir) \
            & (torch.sign(prev_dist) != torch.sign(dist)) & (prev_dist.abs() < 90.0)
        self._primed = self._primed | crossed
        self._primed_theta = torch.where(crossed, self._theta, self._primed_theta)
        blow = self._primed & moving & ((self._theta - self._primed_theta).abs() > c.pass_slack_deg)
        self._stage = torch.where(blow, torch.zeros_like(self._stage), self._stage)
        self._primed = self._primed & ~blow
        if self._demo_lamps:
            self._refresh_stage_lamps()
        if bool(blow[0]):
            self._trace.append({"t": self._trace_step, "read": round(float(read[0]), 1),
                                "stage": int(self._stage[0]), "blow": True})
            if len(self._trace) > 60:
                self._trace.pop(0)

        # ---- torques ----
        from isaaclab.utils.math import quat_apply

        ey = torch.tensor([0.0, 1.0, 0.0], device=dev).expand(n, 3)
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
        dial_axis = quat_apply(self.door.data.root_quat_w, ey)  # world dial/handle axis
        door_axis = ez

        # Dial: external drive + viscous friction + detent brake near ANY combo number
        # approached in that entry's direction.
        w_rad = torch.deg2rad(self._omega)
        tq = self.dial_drive - c.dial_friction * w_rad
        dirs = torch.tensor([-1.0, 1.0, -1.0], device=dev)  # per-entry click direction
        for k in range(c.combo_len):
            dk = (read - self._combo[:, k] + 180.0) % 360.0 - 180.0
            in_win = (dk.abs() <= c.detent_window_deg) & moving \
                & (torch.sign(self._omega) == dirs[k])
            self._click_win[:, k] = in_win
            tq = tq - torch.where(in_win, c.detent_brake * torch.sign(w_rad),
                                  torch.zeros(n, device=dev))
        self.dial.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=dev), (tq.view(n, 1, 1) * dial_axis.view(n, 1, 3)))

        # Handle: bolted (stiff spring to 0 + damping) until the combination is entered.
        h_ang = torch.deg2rad(_twist_deg(self._rel_quat(self.handle, self.door), 1))
        h_w = (self.handle.data.root_ang_vel_w * dial_axis).sum(dim=-1)
        locked = ~self._entered
        h_tq = self.handle_drive - 0.03 * h_w + torch.where(
            locked, -25.0 * h_ang, -0.05 * h_ang)
        self.handle.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=dev), (h_tq.view(n, 1, 1) * dial_axis.view(n, 1, 3)))

        # Door: bolted shut (stiff spring + damping) until the handle passes
        # handle_open_deg. The bolt only acts on a NEARLY-CLOSED door — a real bolt
        # blocks the door frame, it cannot reel an open door back in (v8 smoke: the
        # handle's return spring relaxed below 60 deg after opening and the old
        # unconditional bolt slammed the open door shut before the success check).
        d_ang = torch.deg2rad(_twist_deg(self.door.data.root_quat_w, 2))
        d_w = (self.door.data.root_ang_vel_w * door_axis).sum(dim=-1)
        bolted = (self.handle_angle_deg() < c.handle_open_deg) & (self.door_angle_deg() < 8.0)
        do_tq = self.door_drive - 0.15 * d_w + torch.where(
            bolted, -40.0 * d_ang, torch.zeros(n, device=dev))
        self.door.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=dev), (do_tq.view(n, 1, 1) * door_axis.view(n, 1, 3)))

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"door": self.door, "dial": self.dial, "handle": self.handle,
                  "prize": self.prize, "spoke_x": self.spokes[0], "spoke_z": self.spokes[1],
                  "mark": self.mark}
        return {
            "bodies": {n: b.data.root_state_w[env_ids].clone() for n, b in bodies.items()},
            "lock": {k: getattr(self, k)[env_ids].clone()
                     for k in ("_combo", "_stage", "_theta", "_prev_read", "_omega",
                               "_rest_ctr", "_last_dir", "_primed", "_primed_theta",
                               "_stop_read", "_stop_armed")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"door": self.door, "dial": self.dial, "handle": self.handle,
                  "prize": self.prize, "spoke_x": self.spokes[0], "spoke_z": self.spokes[1],
                  "mark": self.mark}
        for n, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][n], env_ids)
        for k, v in state["lock"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A steel safe (outer {c.outer[0]:.2f} x {c.outer[1]:.2f} x {c.outer[2]:.2f} m) "
            f"stands with its door facing -y. On the door: a spoked brass dial (radius "
            f"{c.dial_radius:.2f} m, numbers 0-359 increasing counter-clockwise, red pointer "
            f"mark above) and a lever handle. Inside, behind the locked door, sits a prize "
            f"block.\n"
            f"The lock takes a SECRET 3-number combination, different every episode, entered "
            f"as: turn the dial RIGHT (reading decreasing) and stop at the 1st number, then "
            f"LEFT (increasing) to the 2nd, then RIGHT to the 3rd (stops must be within "
            f"{c.stop_tol_deg:.0f} deg and held ~{c.rest_dwell_steps / 120:.1f} s). Sweeping "
            f"past the currently-needed number without stopping resets the sequence. Nobody "
            f"tells you the numbers: the only clue is mechanical — passing a combination "
            f"number while turning in its direction gives a brief detent 'click' (the dial "
            f"drags for ~{2 * c.detent_window_deg:.0f} deg). A slow steady sweep makes the "
            f"clicks stand out as sharp dips in the dial's rotation rate; spinning fast "
            f"smears them out.\n"
            f"Once the combination is entered the handle turns (>= {c.handle_open_deg:.0f} "
            f"deg withdraws the bolt), the door swings open, and the prize can be taken out.\n"
            f"Goal: open the safe and get the prize block fully OUT of it, onto the surface "
            f"in front."
        )

    # ----- progress -------------------------------------------------------------------------------
    def combo_entered(self) -> torch.Tensor:
        return self._entered.clone()

    def door_open(self) -> torch.Tensor:
        return self.door_angle_deg() >= self.cfg.door_open_deg

    def prize_out(self) -> torch.Tensor:
        """Prize centre outside the safe footprint (with margin) and settled near surface."""
        c = self.cfg
        W, D, _ = c.outer
        p = self.prize.data.root_pos_w - self.env_origins
        cx, cy = c.safe_pos
        outside = (((p[:, 0] - cx).abs() > W / 2 + 0.05)
                   | ((p[:, 1] - cy).abs() > D / 2 + 0.05))
        low = p[:, 2] < c.surface_z + 0.12
        settled = self.prize.data.root_lin_vel_w.norm(dim=-1) < 0.05
        return outside & low & settled

    def opened(self) -> torch.Tensor:
        return self._opened.clone()

    def success(self) -> torch.Tensor:
        return self.opened() & self.prize_out()


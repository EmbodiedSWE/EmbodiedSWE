"""GenevaVaultFeederScene — load balls through the roof port of a sealed rotary-airlock
feeder, then crank the external Geneva drive to index them, quarter-turn by quarter-turn,
over the internal drop hole and into the vault under the deck.

Derived from the LIBERO seed `libero_90/libero_kitchen_scene6_close_the_microwave` but
STRATEGICALLY DIFFERENT (see TASK.md): the seed is one uncontrolled push on a hinged
door judged by a joint angle. Here nothing closes at all — the plan is a two-phase
routing program through an INTERMITTENT-MOTION TRANSMISSION:

  (1) LOAD: drop each green ball through the only opening in the sealed drum — the
      yellow roof port — into the paddle compartment parked beneath it;
  (2) INDEX: turn the red crank. The crank's shaft carries a pin that engages radial
      slots on the paddle wheel's underdeck slot plate — a classic 4-slot GENEVA
      mechanism: one full crank revolution advances the wheel EXACTLY one quarter
      turn and parks it at the next station (the wheel's velocity is kinematically
      zero at pin entry and exit). Two quarter-turns carry a loaded compartment from
      the port to the diametrically opposite station, where an open hole in the deck
      drops the ball into the enclosed vault below. Delivering every present ball
      (the count is sampled per episode) is the goal.

The transmission is the only drive: the drum, vanes and slot plate are sealed away
(roof + shroud + skirts + the chute collar around the port keep fingers off every
rotating face while the wheel is parked — the parked vanes sit outside the port
footprint), so transport REQUIRES cranking, and cranking is quantized by the Geneva.

Mechanics: plain rigid bodies + two authored USD revolute joints (world-anchored,
continuous, no limits). `post_step` applies the `crank_torque` buffer (about the
driver's vertical axis — the body only yaws, so body z == world z) and the smoke-only
`ball_probe` world-frame forces, then latches rubric progress. Nothing else touches
the wrench slots.

Rubric (graded 0..1, latching, anchored in the demonstrated solve.py trajectory):
per present ball p_i = 0.28*loaded_latch + 0.22*mid_latch (carried to an intermediate
station) + 0.50*vault_latch (fell through the drop hole into the vault);
score = 0.90 * mean_present(p_i) + 0.10 * success(); success() = every present ball
inside the vault AND settled. Null policy: balls untouched on the staging ground ->
score 0. Loading without cranking caps at 0.252.

Per-episode randomization (readback-verified in smoke): ball count (1 or 2), ball
staging poses, the crank's starting phase (uniform over the 240 deg disengaged arc)
and the wheel's park jitter.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class GenevaVaultFeederSceneCfg(BaseCfg):
    """Config for `GenevaVaultFeederScene`. Geometry is derived + asserted in
    `__post_init__` so the scene, the solver and the smoke read the same numbers."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    settle_speed: float = tunable(0.08)  # max ball |lin vel| when judging (m/s)
    vault_z: float = tunable(0.20)  # ball centre below this = in the vault (deck bottom 0.24)
    lane_r_lo: float = tunable(0.20)  # loaded = ball centre radius in [lane_r_lo, lane_r_hi]
    lane_r_hi: float = tunable(0.34)
    lane_z_lo: float = tunable(0.262)  # ... and z in [lane_z_lo, lane_z_hi] (deck..roof)
    lane_z_hi: float = tunable(0.400)
    mid_band_deg: float = tunable(30.0)  # ball azimuth within this of 0 or 180 deg = mid station

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    max_balls: int = tunable(2)  # per-episode count k ~ U{1..max_balls}
    ball_jitter: float = tunable(0.05)  # +/- xy jitter on the staging poses (m)
    crank_arc_deg: float = tunable(120.0)  # crank phase ~ U(-arc, +arc) (disengaged region)
    park_jitter_deg: float = tunable(2.0)  # wheel park jitter (slot-mouth flare tolerates ~4)

    # --- tunable: plant ----------------------------------------------------------------------
    wheel_mass: float = tunable(3.0)
    wheel_damping: float = tunable(2.0)  # angular damping — kills the slot-clearance
    #   coast at pin exit and parks the wheel (engaged resisting torque I*d*w < 0.5 N*m)
    driver_mass: float = tunable(1.2)
    driver_damping: float = tunable(0.30)
    ball_mass: float = tunable(0.06)
    ball_radius: float = tunable(0.025)

    # --- info: layout (env-local; ground z=0; wheel axis at the origin) ----------------------
    d_axis: float = info(0.30)  # driver-axis distance from the wheel axis (+x side)
    deck_top: float = info(0.26)  # deck plate top (deck z 0.24..0.26)
    roof_lo: float = info(0.40)  # roof plate z 0.40..0.42
    roof_hi: float = info(0.42)
    drum_r: float = info(0.21)  # rotating drum (compartment inner wall)
    drum_top: float = info(0.373)  # drum/vane top (roof_lo - drum_top = 27mm < ball dia)
    vane_r_out: float = info(0.265)  # vane outer end — MUST clear the crank shaft's
    #                                  inner surface (d_axis - shaft_r = 0.28) plus offsets
    ring_apothem: float = info(0.28)  # inner guide ring (12-gon) inner face: the ball
    #                                  corridor is drum_r+ball_r .. ring_apothem-ball_r
    #                                  (0.235..0.255), entirely inside the crank shaft
    shroud_apothem: float = info(0.335)  # outer 12-gon casing wall inner face
    hole_x_half: float = info(0.06)  # deck drop hole: x in +/- this, y in [hole_y0, hole_y1]
    hole_y0: float = info(-0.335)
    hole_y1: float = info(-0.225)
    win_x_half: float = info(0.075)  # roof port: x in +/- this, y in [win_y0, win_y1]
    win_y0: float = info(0.21)  # port centred on the ball corridor centre (0.245)
    win_y1: float = info(0.28)
    chute_lo: float = info(0.378)  # port collar lower edge (5 mm above the drum top)
    ext_half: float = info(0.37)  # deck / skirt outer half-extent
    slit_z_lo: float = info(0.08)  # east-skirt pin slit (ball top is 0.05 — cannot exit)
    slit_z_hi: float = info(0.18)
    slit_y_half: float = info(0.23)
    # Geneva transmission (underdeck tier, z 0.11..0.15)
    slot_z_lo: float = info(0.11)
    slot_z_hi: float = info(0.15)
    pin_r: float = info(0.012)
    slot_half_gap: float = info(0.015)  # channel half-width (3 mm clearance per side)
    slot_wall_w: float = info(0.024)
    slot_r_in: float = info(0.07)  # channel walls span slot_r_in .. slot_r_out
    slot_r_out: float = info(0.225)
    hub_r: float = info(0.06)  # slot-plate hub (pin's closest approach is 0.088)
    # Bodies' root origins
    wheel_root_z: float = info(0.318)  # drum centre (drum z 0.263..0.373)
    driver_root_z: float = info(0.3225)  # driver shaft centre (shaft z 0.145..0.50)
    crank_r: float = info(0.11)  # knob orbit radius
    knob_z_lo: float = info(0.51)
    knob_z_hi: float = info(0.60)
    # Staging / depot
    stage_xy: tuple = info(((0.58, 0.28), (0.58, -0.28)))  # ball staging slots (ground)
    depot_xy: tuple = info((1.4, 1.4))  # absent balls park here (ground)
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    pin_orbit: float = field(default=None, init=False)  # pin radius on the driver = d*sin45
    ball_names: tuple = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.pin_orbit = self.d_axis * math.sin(math.pi / 4)  # 0.2121
        self.ball_names = tuple(f"ball_{i}" for i in range(self.max_balls))
        # ---- import-time geometry asserts (cheap forge iterations avoided) ----
        wheel_slot_r = self.d_axis * math.cos(math.pi / 4)  # pin enters the slot here
        assert self.slot_r_out > wheel_slot_r > self.slot_r_in + 0.01, "slot must span pin entry"
        assert self.d_axis - self.pin_orbit > self.hub_r + self.pin_r + 0.010, \
            "pin closest approach must clear the slot-plate hub"
        assert self.slot_half_gap > self.pin_r + 0.002, "pin needs channel clearance"
        ball_d = 2 * self.ball_radius
        assert self.roof_lo - self.drum_top < ball_d, "vane-roof gap must not pass a ball"
        assert self.chute_lo - self.drum_top >= 0.004, "port collar must clear the drum sweep"
        assert 2 * self.win_x_half > ball_d + 0.02, "port must admit a ball"
        assert self.hole_y1 - self.hole_y0 > ball_d + 0.02, "hole must span the ball lane"
        assert self.slit_z_lo > 2 * self.ball_radius + 0.01, "pin slit must not pass a ball"
        # ---- v3: guide-ring corridor (ring keeps balls off the crank shaft) ----
        corr_lo = self.drum_r + self.ball_radius  # innermost ball-centre radius (0.235)
        corr_hi = self.ring_apothem - self.ball_radius  # outermost (0.255)
        assert corr_hi - corr_lo >= 0.015, "ball corridor must have real width"
        assert self.vane_r_out >= corr_hi + 0.005, \
            "vane tip must reach past the outermost corridor ball centre"
        assert self.vane_r_out + 0.010 < self.d_axis - 0.02, \
            "vane tip sweep must clear the crank shaft inner surface"
        assert self.ring_apothem - self.vane_r_out < ball_d, \
            "vane-ring gap must not pass a ball"
        assert self.win_y1 <= self.ring_apothem, "port must sit inside the guide ring"
        assert self.win_y0 < 0.5 * (corr_lo + corr_hi) < self.win_y1, \
            "port must span the corridor centreline (the drop line)"
        assert self.drum_r + self.ball_radius > -self.hole_y1 + 0.005, \
            "hole inner edge must catch a ball hugging the drum (centre past the edge)"
        assert self.hole_y0 >= -self.shroud_apothem, "hole must stay inside the shroud"


def _qz(rad: float) -> tuple:
    return (math.cos(rad / 2), 0.0, 0.0, math.sin(rad / 2))


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("geneva_vault_feeder")
class GenevaVaultFeederScene(BaseScene):
    cfg: GenevaVaultFeederSceneCfg

    def __init__(self, cfg: GenevaVaultFeederSceneCfg | None = None) -> None:
        super().__init__(cfg or GenevaVaultFeederSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic rig root (base plate — every static box is authored
        as a child in bind), the two dynamic rotors (wheel root = drum, driver root = crank
        shaft; compound children authored in bind) and the balls."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        live = dict(sleep_threshold=0.0, stabilization_threshold=0.0)

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
            "rig": RigidObjectCfg(  # root = the vault floor plate; children in bind
                prim_path="{ENV_REGEX_NS}/Rig",
                spawn=sim_utils.CuboidCfg(
                    size=(0.80, 0.80, 0.02),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.42, 0.42, 0.45)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.01)),
            ),
            "wheel": RigidObjectCfg(  # root = the drum cylinder; vanes/shaft/slots in bind
                prim_path="{ENV_REGEX_NS}/Wheel",
                spawn=sim_utils.CylinderCfg(
                    radius=c.drum_r, height=0.110, axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        angular_damping=c.wheel_damping,
                        **live),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.wheel_mass),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.85, 0.45, 0.12)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.wheel_root_z)),
            ),
            "driver": RigidObjectCfg(  # root = the crank shaft; hub/pin/arm/knob in bind
                prim_path="{ENV_REGEX_NS}/Driver",
                spawn=sim_utils.CylinderCfg(
                    radius=0.02, height=0.355, axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        angular_damping=c.driver_damping,
                        **live),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.driver_mass),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.25, 0.25, 0.28)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.d_axis, 0.0, c.driver_root_z)),
            ),
        }
        for i, name in enumerate(c.ball_names):
            sx, sy = c.stage_xy[i % len(c.stage_xy)]
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball" + str(i),
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_radius,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        linear_damping=0.05,
                        angular_damping=0.05,
                        **live),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.4, dynamic_friction=0.35, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.10, 0.70, 0.20)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(sx, sy, c.ball_radius + 0.002)),
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
        c = self.cfg
        self.wheel: RigidObject = env.iscene["wheel"]
        self.driver: RigidObject = env.iscene["driver"]
        self.balls: dict[str, RigidObject] = {nm: env.iscene[nm] for nm in c.ball_names}
        self.env_origins = env.iscene.env_origins
        self._author_rig_children()
        self._author_wheel_children()
        self._author_driver_children()
        self._author_mass()
        self._author_joints()
        nb = len(c.ball_names)
        # Episode state.
        self.present = torch.ones(n, nb, dtype=torch.bool, device=dev)
        self.load_latch = torch.zeros(n, nb, device=dev)
        self.mid_latch = torch.zeros(n, nb, device=dev)
        self.vault_latch = torch.zeros(n, nb, device=dev)
        # External inputs (solve.py writes crank_torque; ball_probe is smoke-only
        # instrumentation; post_step consumes + owns the wrench slots).
        self.crank_torque = torch.zeros(n, device=dev)  # N*m about the driver axis (+ = CCW)
        self.ball_probe = torch.zeros(n, nb, 3, device=dev)  # world-frame probe forces (N)

    # -- USD authoring helpers (env_0 children; joints per env — the i360 pattern) --------------
    def _box(self, parent: str, name: str, lo: tuple, hi: tuple, color: tuple,
             root: tuple, yaw_deg: float = 0.0, center: tuple = None, size: tuple = None):
        """Author one collision box child. Either (lo, hi) world AABB or (center, size)
        world + yaw. `root` = the parent body's world origin."""
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        path = f"/World/envs/env_0/{parent}/{name}"
        if stage.GetPrimAtPath(path).IsValid():
            return
        if center is None:
            center = tuple((lo[k] + hi[k]) / 2 for k in range(3))
            size = tuple(hi[k] - lo[k] for k in range(3))
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(center[0] - root[0], center[1] - root[1],
                                         center[2] - root[2]))
        if yaw_deg:
            xf.AddRotateZOp().Set(yaw_deg)
        xf.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
        px.CreateContactOffsetAttr(self.cfg.contact_offset)
        px.CreateRestOffsetAttr(0.0)

    def _cyl(self, parent: str, name: str, center: tuple, radius: float, height: float,
             color: tuple, root: tuple, capsule: bool = False):
        """Author one collision cylinder/capsule child (axis Z), world `center`."""
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        path = f"/World/envs/env_0/{parent}/{name}"
        if stage.GetPrimAtPath(path).IsValid():
            return
        cls = UsdGeom.Capsule if capsule else UsdGeom.Cylinder
        geo = cls.Define(stage, path)
        geo.CreateRadiusAttr(radius)
        geo.CreateHeightAttr(height)
        geo.CreateAxisAttr("Z")
        xf = UsdGeom.Xformable(geo.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(center[0] - root[0], center[1] - root[1],
                                         center[2] - root[2]))
        geo.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdPhysics.CollisionAPI.Apply(geo.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(geo.GetPrim())
        px.CreateContactOffsetAttr(self.cfg.contact_offset)
        px.CreateRestOffsetAttr(0.0)

    def _author_rig_children(self) -> None:
        """All static geometry as children of the kinematic rig root (base plate at
        (0,0,0.01)): deck tiles around the drop hole + shaft openings, the 12-gon shroud,
        the roof tiles around the port, the yellow port collar and the vault skirts with
        the east pin slit."""
        c = self.cfg
        root = (0.0, 0.0, 0.01)
        e = c.ext_half
        tan_ = (0.72, 0.62, 0.46)  # deck
        slate = (0.22, 0.24, 0.28)  # roof
        gray = (0.35, 0.36, 0.38)  # skirts
        blue = (0.45, 0.55, 0.65)  # shroud
        yellow = (0.85, 0.75, 0.20)  # port collar
        z0, z1 = 0.24, c.deck_top
        hx, hy0, hy1 = c.hole_x_half, c.hole_y0, c.hole_y1
        # deck tiles (openings: centre shaft 0.05, hole, driver shaft x .25...35)
        deck = [
            ("deck_n", (-e, 0.05, z0), (e, e, z1)),
            ("deck_sband", (-e, hy1, z0), (e, -0.05, z1)),
            ("deck_sedge", (-e, -e, z0), (e, hy0, z1)),
            ("deck_hw", (-e, hy0, z0), (-hx, hy1, z1)),
            ("deck_he", (hx, hy0, z0), (e, hy1, z1)),
            ("deck_cw", (-e, -0.05, z0), (-0.05, 0.05, z1)),
            # driver shaft opening shrunk to x .274...326, y +/-.026 (6mm shaft clearance;
            # corridor balls, centre radius <= 0.255, stay >= 19mm from the west edge)
            ("deck_cd", (0.05, -0.05, z0), (0.274, 0.05, z1)),
            ("deck_de", (0.326, -0.05, z0), (e, 0.05, z1)),
            ("deck_dn", (0.274, 0.026, z0), (0.326, 0.05, z1)),
            ("deck_ds", (0.274, -0.05, z0), (0.326, -0.026, z1)),
        ]
        for nm, lo, hi in deck:
            self._box("Rig", nm, lo, hi, tan_, root)
        # shroud: 12 box segments, centreline apothem 0.345, z 0.26..0.40
        seg_len = 2 * 0.345 * math.tan(math.pi / 12) + 0.006
        for k in range(12):
            ang = k * 30.0
            a = math.radians(ang)
            cx, cy = 0.345 * math.cos(a), 0.345 * math.sin(a)
            self._box("Rig", f"shroud_{k}", None, None, blue, root, yaw_deg=ang + 90.0,
                      center=(cx, cy, 0.33), size=(seg_len, 0.02, 0.14))
        # inner guide ring: 12 box segments, inner apothem ring_apothem (0.28), thickness
        # 0.02, z 0.26..0.40 — walls the ball corridor (r 0.235..0.255) off the crank
        # shaft. The az-0 segment is split into two, leaving a y +/-0.026 notch where the
        # shaft's inner surface (x = 0.28) sits flush with the ring face: a bollard the
        # corridor ball rolls past, not a pocket it can wedge into.
        ring_c = c.ring_apothem + 0.01  # centreline
        ring_len = 2 * ring_c * math.tan(math.pi / 12) + 0.006
        for k in range(12):
            ang = k * 30.0
            a = math.radians(ang)
            cx, cy = ring_c * math.cos(a), ring_c * math.sin(a)
            if k == 0:
                half = (ring_len / 2 - 0.026)  # each split box's length
                yc = 0.026 + half / 2
                for tag, sy in (("a", +1), ("b", -1)):
                    self._box("Rig", f"ring_0{tag}", None, None, blue, root, yaw_deg=90.0,
                              center=(cx, sy * yc, 0.33), size=(half, 0.02, 0.14))
                continue
            self._box("Rig", f"ring_{k}", None, None, blue, root, yaw_deg=ang + 90.0,
                      center=(cx, cy, 0.33), size=(ring_len, 0.02, 0.14))
        # roof tiles around the port
        rl, rh = c.roof_lo, c.roof_hi
        roof = [
            ("roof_n", (-0.39, c.win_y1, rl), (0.39, 0.39, rh)),
            ("roof_s", (-0.39, -0.39, rl), (0.39, c.win_y0, rh)),
            ("roof_w", (-0.39, c.win_y0, rl), (-c.win_x_half, c.win_y1, rh)),
            ("roof_e", (c.win_x_half, c.win_y0, rl), (0.39, c.win_y1, rh)),
        ]
        for nm, lo, hi in roof:
            self._box("Rig", nm, lo, hi, slate, root)
        # port collar (chute): keeps fingers off the parked drum faces; z chute_lo..roof_hi
        wxh, wy0, wy1 = c.win_x_half, c.win_y0, c.win_y1
        cl = c.chute_lo
        chute = [
            ("chute_n", (-wxh - 0.02, wy1, cl), (wxh + 0.02, wy1 + 0.02, rh)),
            ("chute_s", (-wxh - 0.02, wy0 - 0.02, cl), (wxh + 0.02, wy0, rh)),
            ("chute_w", (-wxh - 0.02, wy0, cl), (-wxh, wy1, rh)),
            ("chute_e", (wxh, wy0, cl), (wxh + 0.02, wy1, rh)),
        ]
        for nm, lo, hi in chute:
            self._box("Rig", nm, lo, hi, yellow, root)
        # vault skirts (z 0.02..0.24); east side carries the pin slit
        sk = [
            ("skirt_w", (-e - 0.02, -e - 0.02, 0.02), (-e, e + 0.02, 0.24)),
            ("skirt_n", (-e, e, 0.02), (e, e + 0.02, 0.24)),
            ("skirt_s", (-e, -e - 0.02, 0.02), (e, -e, 0.24)),
            ("skirt_e_lo", (e, -e, 0.02), (e + 0.02, e, c.slit_z_lo)),
            ("skirt_e_hi", (e, -e, c.slit_z_hi), (e + 0.02, e, 0.24)),
            ("skirt_e_sn", (e, c.slit_y_half, c.slit_z_lo), (e + 0.02, e, c.slit_z_hi)),
            ("skirt_e_ss", (e, -e, c.slit_z_lo), (e + 0.02, -c.slit_y_half, c.slit_z_hi)),
        ]
        for nm, lo, hi in sk:
            self._box("Rig", nm, lo, hi, gray, root)

    def _author_wheel_children(self) -> None:
        """Wheel children (root = drum at z 0.318): connecting shaft, slot-plate hub, the
        4 Geneva slot channels (2 walls + 2 mouth flares each) and the 4 vanes."""
        c = self.cfg
        root = (0.0, 0.0, c.wheel_root_z)
        orange = (0.85, 0.45, 0.12)
        steel = (0.55, 0.56, 0.60)
        # connecting shaft through the deck opening
        self._cyl("Wheel", "shaft", (0.0, 0.0, 0.2015), 0.035, 0.123, steel, root)
        # slot-plate hub
        zc = (c.slot_z_lo + c.slot_z_hi) / 2
        zh = c.slot_z_hi - c.slot_z_lo
        self._cyl("Wheel", "slot_hub", (0.0, 0.0, zc), c.hub_r, zh, steel, root)
        # slot channels at 45 + k*90 (the parked orientation IS identity)
        wall_len = c.slot_r_out - c.slot_r_in
        wall_rc = (c.slot_r_in + c.slot_r_out) / 2
        off = c.slot_half_gap + c.slot_wall_w / 2
        for k in range(4):
            ang = 45.0 + k * 90.0
            a = math.radians(ang)
            ca, sa = math.cos(a), math.sin(a)
            for s, tag in ((+1, "a"), (-1, "b")):
                # wall centre: r=wall_rc on the channel axis, offset s*off laterally
                cx = wall_rc * ca - s * off * sa
                cy = wall_rc * sa + s * off * ca
                self._box("Wheel", f"slot{k}{tag}", None, None, steel, root, yaw_deg=ang,
                          center=(cx, cy, zc), size=(wall_len, c.slot_wall_w, zh))
                # mouth flare: from the wall's outer end, angled 25 deg outward.
                # Length 0.030 keeps the flare corner sweep radius <= 0.259 — clear of
                # the driver hub (r 0.03 at d=0.30) and of the pin's disengaged orbit.
                fa = math.radians(ang + s * 25.0)
                fx0 = c.slot_r_out * ca - s * off * sa  # wall outer-end centre
                fy0 = c.slot_r_out * sa + s * off * ca
                fx = fx0 + 0.015 * math.cos(fa)
                fy = fy0 + 0.015 * math.sin(fa)
                self._box("Wheel", f"flare{k}{tag}", None, None, steel, root,
                          yaw_deg=ang + s * 25.0,
                          center=(fx, fy, zc), size=(0.030, c.slot_wall_w, zh))
        # vanes (drum face .. vane_r_out), full drum height
        vane_len = c.vane_r_out - c.drum_r
        vane_rc = (c.drum_r + c.vane_r_out) / 2
        for k in range(4):
            ang = 45.0 + k * 90.0
            a = math.radians(ang)
            self._box("Wheel", f"vane{k}", None, None, orange, root, yaw_deg=ang,
                      center=(vane_rc * math.cos(a), vane_rc * math.sin(a), 0.318),
                      size=(vane_len, 0.024, 0.110))

    def _author_driver_children(self) -> None:
        """Driver children (root = shaft at (d,0,0.3225)): underdeck hub + Geneva pin
        (capsule), crank arm and the red knob."""
        c = self.cfg
        root = (c.d_axis, 0.0, c.driver_root_z)
        steel = (0.55, 0.56, 0.60)
        red = (0.80, 0.10, 0.10)
        dark = (0.25, 0.25, 0.28)
        # hub r 0.03: the wheel's flare corners sweep to r 0.259 from the wheel axis,
        # i.e. 0.041 from the driver axis — r 0.05 would jam the index at ~-24 deg.
        self._cyl("Driver", "hub", (c.d_axis, 0.0, 0.1275), 0.03, 0.035, steel, root)
        self._cyl("Driver", "pin", (c.d_axis + c.pin_orbit, 0.0, 0.13), c.pin_r, 0.05,
                  steel, root, capsule=True)
        self._box("Driver", "arm", None, None, dark, root,
                  center=(c.d_axis + 0.055, 0.0, 0.50), size=(0.13, 0.03, 0.02))
        self._cyl("Driver", "knob",
                  (c.d_axis + c.crank_r, 0.0, (c.knob_z_lo + c.knob_z_hi) / 2),
                  0.016, c.knob_z_hi - c.knob_z_lo, red, root)

    def _author_mass(self) -> None:
        """Explicit diagonal inertia + on-axis CoM for both rotors (the external-wrench
        plant needs authored inertia; CoM on the joint axis kills gravity torque)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        for path, mass, diag in (
            ("/World/envs/env_0/Wheel", self.cfg.wheel_mass, (0.10, 0.10, 0.09)),
            ("/World/envs/env_0/Driver", self.cfg.driver_mass, (0.030, 0.030, 0.030)),
        ):
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            m = UsdPhysics.MassAPI.Apply(prim)
            m.CreateMassAttr(mass)
            m.CreateDiagonalInertiaAttr(Gf.Vec3f(*diag))
            m.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))

    def _author_joints(self) -> None:
        """Per env: two continuous Z revolutes anchored on the kinematic rig — rig->wheel
        at the origin, rig->driver at (d,0). No limits (multi-turn crank); joint pairs
        collision-filtered."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            for name, body, anchor_w, root_z in (
                ("wheel_axle", "Wheel", (0.0, 0.0, c.wheel_root_z), c.wheel_root_z),
                ("driver_axle", "Driver", (c.d_axis, 0.0, c.driver_root_z), c.driver_root_z),
            ):
                j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/{name}")
                j.CreateBody0Rel().SetTargets([f"{base}/Rig"])
                j.CreateBody1Rel().SetTargets([f"{base}/{body}"])
                j.CreateCollisionEnabledAttr(False)
                j.CreateAxisAttr("Z")
                j.CreateLocalPos0Attr(Gf.Vec3f(anchor_w[0], anchor_w[1], anchor_w[2] - 0.01))
                j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the ball count, stage present balls on the ground with
        jitter (absent -> depot), park the wheel at a station with jitter, spin the crank
        to a uniform phase in the disengaged arc; clear latches and drive buffers."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        nb = len(c.ball_names)

        torch.rand(2, device=dev)  # burn the degenerate first post-seed draws
        k = torch.randint(1, c.max_balls + 1, (m,), device=dev)  # present count
        u = torch.rand(m, 2 + 2 * nb, device=dev)

        self.load_latch[env_ids] = 0.0
        self.mid_latch[env_ids] = 0.0
        self.vault_latch[env_ids] = 0.0
        self.crank_torque[env_ids] = 0.0
        self.ball_probe[env_ids] = 0.0

        # wheel parked at a station (identity = slots at 45+k*90) + jitter
        park = (u[:, 0] * 2 - 1) * math.radians(c.park_jitter_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 2] = c.wheel_root_z
        st[:, 3] = torch.cos(park / 2)
        st[:, 6] = torch.sin(park / 2)
        st[:, 0:3] += origin
        self.wheel.write_root_state_to_sim(st, env_ids)

        # crank at a uniform disengaged phase
        phi = (u[:, 1] * 2 - 1) * math.radians(c.crank_arc_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.d_axis
        st[:, 2] = c.driver_root_z
        st[:, 3] = torch.cos(phi / 2)
        st[:, 6] = torch.sin(phi / 2)
        st[:, 0:3] += origin
        self.driver.write_root_state_to_sim(st, env_ids)

        # balls: staged on the ground (present) or in the depot (absent)
        for i, nm in enumerate(c.ball_names):
            pres = (torch.arange(1, device=dev) + i < k.unsqueeze(1)).squeeze(1)  # i < k
            self.present[env_ids, i] = pres
            sx, sy = c.stage_xy[i % len(c.stage_xy)]
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = sx + (u[:, 2 + 2 * i] * 2 - 1) * c.ball_jitter
            st[:, 1] = sy + (u[:, 3 + 2 * i] * 2 - 1) * c.ball_jitter
            st[:, 2] = c.ball_radius + 0.002
            dx, dy = c.depot_xy
            st[:, 0] = torch.where(pres, st[:, 0], torch.full_like(st[:, 0], dx + 0.15 * i))
            st[:, 1] = torch.where(pres, st[:, 1], torch.full_like(st[:, 1], dy))
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            self.balls[nm].write_root_state_to_sim(st, env_ids)

    # ----- readings -----------------------------------------------------------------------------
    def _ball_pos(self) -> torch.Tensor:
        """(N, B, 3) env-local ball positions."""
        return torch.stack([b.data.root_pos_w for b in self.balls.values()], dim=1) \
            - self.env_origins[:, None, :]

    def _ball_vel(self) -> torch.Tensor:
        """(N, B) ball |lin vel|."""
        return torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                            for b in self.balls.values()], dim=1)

    def driver_angle(self) -> torch.Tensor:
        """(N,) crank angle in rad, wrapped to (-pi, pi] (the driver only yaws)."""
        q = self.driver.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 3], q[:, 0])

    def driver_rate(self) -> torch.Tensor:
        return self.driver.data.root_ang_vel_w[:, 2]

    def wheel_angle(self) -> torch.Tensor:
        """(N,) wheel angle in rad, wrapped (0 = parked-at-spawn station family)."""
        q = self.wheel.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 3], q[:, 0])

    def wheel_rate(self) -> torch.Tensor:
        return self.wheel.data.root_ang_vel_w[:, 2]

    def in_lane(self) -> torch.Tensor:
        """(N, B) bool: ball inside the compartment lane (loaded)."""
        c = self.cfg
        p = self._ball_pos()
        r = p[:, :, :2].norm(dim=-1)
        return (r > c.lane_r_lo) & (r < c.lane_r_hi) \
            & (p[:, :, 2] > c.lane_z_lo) & (p[:, :, 2] < c.lane_z_hi)

    def at_mid(self) -> torch.Tensor:
        """(N, B) bool: loaded AND azimuth within mid_band_deg of 0 or 180 deg (an
        intermediate station, either cranking direction)."""
        c = self.cfg
        p = self._ball_pos()
        az = torch.atan2(p[:, :, 1], p[:, :, 0])
        band = math.radians(c.mid_band_deg)
        near0 = az.abs() < band
        near180 = (az.abs() - math.pi).abs() < band
        return self.in_lane() & (near0 | near180)

    def in_vault(self) -> torch.Tensor:
        """(N, B) bool: ball below the deck inside the vault footprint."""
        c = self.cfg
        p = self._ball_pos()
        return (p[:, :, 2] < c.vault_z) \
            & (p[:, :, 0].abs() < c.ext_half) & (p[:, :, 1].abs() < c.ext_half)

    def settled(self) -> torch.Tensor:
        """(N, B) bool: ball |lin vel| below settle_speed."""
        return self._ball_vel() < self.cfg.settle_speed

    def success(self) -> torch.Tensor:
        """(N,) bool: every PRESENT ball is inside the vault and settled (current,
        physical state — the vault is enclosed, so this state is passively stable)."""
        ok = (self.in_vault() & self.settled()) | ~self.present
        return ok.all(dim=1)

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.90 * mean over present balls of
        (0.28*loaded + 0.22*mid + 0.50*vault latches) + 0.10 * success().
        Exactly 1.0 iff success() (success forces the latches full — the legit path
        implies them); ~0 for the null policy; load-without-crank caps at 0.252."""
        pres = self.present.float()
        npres = pres.sum(dim=1).clamp(min=1.0)
        vault_now = self.in_vault().float()
        load = torch.maximum(self.load_latch, vault_now)
        mid = torch.maximum(self.mid_latch, vault_now)
        vault = torch.maximum(self.vault_latch, vault_now)
        p = 0.28 * load + 0.22 * mid + 0.50 * vault
        base = (p * pres).sum(dim=1) / npres
        return 0.90 * base + 0.10 * self.success().float()

    # ----- step-coupled mechanics (every substep) ------------------------------------------------
    def post_step(self) -> None:
        """Apply the crank torque (driver body only yaws: body z == world z) and the
        smoke-only ball probe forces; then latch rubric progress (NaN-guarded)."""
        from isaaclab.utils.math import quat_apply_inverse

        n = self.env.num_envs
        dev = self.env.device
        zero3 = torch.zeros(n, 1, 3, device=dev)
        tq = torch.zeros(n, 1, 3, device=dev)
        tq[:, 0, 2] = self.crank_torque
        self.driver.set_external_force_and_torque(zero3, tq)
        for i, b in enumerate(self.balls.values()):
            # the wrench API takes BODY-frame forces and a rolling ball's body frame
            # spins — encode the WORLD-frame probe into the current body frame
            f_b = quat_apply_inverse(b.data.root_quat_w, self.ball_probe[:, i])
            b.set_external_force_and_torque(f_b.unsqueeze(1), zero3)

        vault_now = self.in_vault().float()
        loaded = torch.maximum(self.in_lane().float(), vault_now)
        mid = torch.maximum(self.at_mid().float(), vault_now)
        for buf, val in ((self.load_latch, loaded), (self.mid_latch, mid),
                         (self.vault_latch, vault_now)):
            val = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
            buf.copy_(torch.maximum(buf, val))

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in (("wheel", self.wheel), ("driver", self.driver),
                                     *self.balls.items())},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("present", "load_latch", "mid_latch", "vault_latch",
                               "crank_torque", "ball_probe")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = dict((("wheel", self.wheel), ("driver", self.driver), *self.balls.items()))
        for nm, st in state["bodies"].items():
            bodies[nm].write_root_state_to_sim(st, env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A sealed rotary-airlock feeder stands on a gray base plate: a blue twelve-"
            "sided shroud on a tan deck, covered by a dark flat roof. Inside, an orange "
            "paddle wheel (a central drum with four radial vanes) divides the narrow "
            "ring-shaped ball corridor between the drum and a blue inner guide ring into "
            "four compartments. The roof has exactly "
            "one opening — a square loading port with a YELLOW collar — above the "
            "compartment parked on the wheel's +y side. Diametrically OPPOSITE the port, "
            "hidden inside the shroud, an open hole in the deck drops anything that "
            "passes over it into the enclosed vault below the deck (gray skirts close "
            "the vault on all sides). On the east side a vertical shaft topped by a "
            "crank arm with a RED knob rises beside the shroud: under the deck it "
            "carries a pin that engages the paddle wheel's slotted plate as a GENEVA "
            "drive — one full crank revolution (either direction) advances the paddle "
            "wheel by EXACTLY one quarter turn and re-parks it; between indexes the "
            "wheel stays put. The wheel's rotating faces are out of reach: the crank is "
            "the drive. One or two GREEN balls (25 mm radius — count what you see) rest "
            "on the ground east of the rig.\n"
            "Goal: deliver every green ball into the vault below the deck. For each "
            "ball: place it into the loading port so it falls into the parked "
            "compartment, then crank the red knob through full revolutions — each "
            "revolution advances the wheel one quarter turn — until the ball's "
            "compartment has crossed the drop hole two quarter-turns from the port "
            "(either direction) and the ball falls through into the vault. With two "
            "balls you may load the second into the compartment newly parked under the "
            "port after any index. Balls left in compartments, on the roof, or outside "
            "score nothing beyond loading credit; success is every present ball resting "
            "inside the vault."
        )

    def instruction(self) -> str:
        return (
            "Drop each green ball through the yellow roof port into the parked "
            "compartment of the sealed paddle wheel, then turn the red crank in full "
            "revolutions — each revolution indexes the wheel one quarter turn — until "
            "every ball is carried over the internal drop hole (two quarter-turns from "
            "the port) and falls into the vault under the deck. The wheel can only be "
            "moved with the crank."
        )


# ----- runnable env: scene physics only (NullRobot) -> "simgen.geneva_vault_feeder" ------------
register_env("simgen", lambda: EnvCfg(scene="geneva_vault_feeder", robot="null"))

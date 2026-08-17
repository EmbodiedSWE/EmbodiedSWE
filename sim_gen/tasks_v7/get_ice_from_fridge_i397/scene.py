"""IceCaddyDockScene — carry a SEALED ice caddy to the dispensing dock, let its own
weight open the spring-loaded bottom valve on the dock's probe post, drain every ice
ball into the dock basin, then park the re-sealed caddy upright on the green pad.

Derived from the RLBench `get_ice_from_fridge` seed but STRATEGICALLY DIFFERENT (see
TASK.md): the seed's plan is "hold a cup against a fridge dispenser lever" — one
sustained press at a fixed machine, no transported container state and no ordering.
Here the ICE SOURCE ITSELF is the carried object: a closed-top caddy whose only
opening is a bottom port sealed by a spring-loaded poppet plug. NOTHING the hand does
to the caddy in free space can discharge it — tilt it, shake it, invert it: the roof
is solid and the spring (preload > plug weight, in any orientation) keeps the plug on
its seat. The ONLY discharge path is mechanical: seat the caddy on the dock so the
dock's fixed probe post reaches through the bottom port and pushes the plug open
against its spring — under the caddy's own weight, hands-off. A DECOY basin
(identical, no probe) seats the caddy but opens nothing. The plan skeleton becomes a
forced serial program with a passive interlock — lift, seat on the CORRECT dock,
dwell while gravity drains, lift off (the spring re-seals), park on the pad — instead
of the seed's single memoryless press.

Mechanics: the caddy is ONE dynamic compound body (roof root + walls, holed base
flange, 30-degree funnel plates, bridge handle authored as children); the plug is a
second dynamic body (head root + stem below, 45-degree ridge above) riding an
authored PRISMATIC joint (axis = caddy z, travel [0, 34] mm, 0 = sealed) with a
linear DriveAPI return spring (target below the lower stop = seating preload). Closed,
the plug head's rim sits BELOW the funnel surface line — the throat is walled shut.
The stem tip is recessed 2 mm above the base plane, so no floor and no finger can
reach it while the caddy stands; the dock's probe post (top 30 mm above the rim the
caddy seats on) is the one thing shaped to press it. `post_step` applies the
WORLD-frame `caddy_force_w`/`caddy_torque_w`/`plug_force_w` probe buffers (converted
to body frame each substep with the live quaternion) and owns both wrench slots.

Rubric (graded 0..1, latching transient achievement — anchored in solve.py):
  - `lift_latch`      (0.12): the caddy has ever been raised (root z >= 0.16);
  - `dock_open_latch` (0.18): the valve has ever been open (ext > 15 mm) WHILE the
    caddy is seated at the real dock (xy within 60 mm) — the only place it can open
    under the no-teleport physics;
  - delivered         (0.40): fraction of PRESENT balls that have ever settled inside
    the dock basin (per-ball latch);
  - current success   (0.30): every present ball settled in the dock basin AND the
    caddy parked upright on the pad with the valve re-sealed (ext <= 6 mm) AND
    everything still -> score == 1.0 iff success(). All latches are forced full on
    success (the legit path implies them). Null policy: caddy stands where it
    spawned, balls sealed inside — score ~0.

Per-episode randomization (readback-verified in smoke): which side the real dock is
on (the decoy mirrors it), dock/decoy/pad/caddy xy jitter, caddy yaw, and the number
of ice balls (2-4).

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
class IceCaddyDockSceneCfg(BaseCfg):
    """Config for `IceCaddyDockScene`. Geometry is derived once in `__post_init__` so the
    scene, the smoke AND the solver read the same numbers — and the interlock margins are
    ASSERTED there (fail at import, not after a forge round-trip)."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    lift_z: float = tunable(0.16)  # caddy root z above this = lifted (stand 0.108)
    open_ext: float = tunable(0.015)  # plug ext beyond this = valve open
    dock_xy_tol: float = tunable(0.060)  # caddy-over-dock radius for the open latch
    ball_xy_tol: float = tunable(0.077)  # ball within this of dock centre = in basin
    ball_z_lo: float = tunable(0.010)  # basin band (floor top 0.008 + r 0.010 = 0.018)
    ball_z_hi: float = tunable(0.075)  # below the rim (0.100): inside, not perched
    ball_latch_vel: float = tunable(0.08)  # ball speed gate when latching delivery
    park_xy_tol: float = tunable(0.070)  # caddy centre within this of the pad centre
    park_z_tol: float = tunable(0.012)  # |root z - park_root_z| when judging parked
    upright_cos: float = tunable(0.9848)  # cos(10 deg) tilt gate
    closed_ext: float = tunable(0.006)  # plug ext below this = re-sealed
    settle_lin: float = tunable(0.06)  # max |lin vel| (m/s) when judging (> GPU creep)
    settle_ang: float = tunable(0.30)  # max caddy |ang vel| (rad/s) when judging

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    n_balls_min: int = tunable(2)
    n_balls_max: int = tunable(4)
    dock_x: float = tunable(0.38)  # dock/decoy centre x (+- jitter)
    dock_y: float = tunable(0.22)  # dock at side*dock_y, decoy mirrored at -side*dock_y
    dock_jit: float = tunable(0.02)
    pad_x: float = tunable(0.12)
    pad_jit: float = tunable(0.02)
    pad_y_jit: float = tunable(0.03)
    caddy_x: float = tunable(0.13)
    caddy_y: float = tunable(0.30)  # caddy starts on the SAME side as the real dock
    caddy_jit: float = tunable(0.02)

    # --- tunable: plant (difficulty dials) ---------------------------------------------------
    caddy_mass: float = tunable(0.50)
    plug_mass: float = tunable(0.04)
    ball_mass: float = tunable(0.02)
    spring_k: float = tunable(75.0)  # N/m linear return spring on the plug
    spring_c: float = tunable(5.0)  # N*s/m
    spring_preload: float = tunable(0.008)  # drive target 8 mm BELOW the lower stop

    # --- info: caddy (local frame: origin = roof centre; base plane at z = -0.108) -----------
    roof_size: tuple = info((0.156, 0.156, 0.008))
    wall_t: float = info(0.008)
    wall_h: float = info(0.104)  # walls span z [-0.108, -0.004]
    flange_size: float = info(0.220)  # square base flange, 8 mm thick, hole in the middle
    flange_t: float = info(0.008)
    hole_half: float = info(0.036)  # bottom port: 72 mm square
    base_z: float = info(-0.108)  # caddy-local z of the base plane (flange bottom)
    funnel_len: float = info(0.0416)  # plate length along the 30-deg slope
    funnel_w: float = info(0.150)
    funnel_t: float = info(0.006)
    funnel_deg: float = info(30.0)
    funnel_lip_x: float = info(0.0345)  # throat lip (top surface low edge) |x| ...
    funnel_lip_z: float = info(-0.0995)  # ... and its caddy-local z
    post_size: tuple = info((0.012, 0.012, 0.030))  # handle posts at (+-0.036, 0)
    post_x: float = info(0.036)
    bar_size: tuple = info((0.096, 0.014, 0.012))  # crossbar, centre z +0.040
    bar_z: float = info(0.040)

    # --- info: plug (local frame: origin = head centre) --------------------------------------
    head_size: tuple = info((0.088, 0.088, 0.010))
    stem_size: tuple = info((0.012, 0.012, 0.010))
    stem_z: float = info(-0.010)  # stem centre (plug-local): tip at -0.015
    ridge_size: tuple = info((0.030, 0.080, 0.030))  # 45-deg roof ridge: balls shed off
    ridge_z: float = info(0.012)  # tent vertex -9.2 mm plug-local: ~8 mm above ground at spawn
    plug_closed_z: float = info(-0.091)  # head centre in the CADDY frame at ext = 0
    plug_stroke: float = info(0.034)  # prismatic travel; ext in [0, stroke]

    # --- info: dock / decoy / pad (kinematic; root = basin floor slab) -----------------------
    dock_floor_size: tuple = info((0.190, 0.190, 0.008))
    dock_floor_z: float = info(0.004)  # spawn z of the floor slab centre
    dock_wall_t: float = info(0.008)
    rim_z: float = info(0.100)  # world z of the basin rim the caddy flange seats on
    probe_size: tuple = info((0.014, 0.014, 0.122))
    probe_top_z: float = info(0.130)  # world z of the probe post top (rim + 30 mm)
    collar_inner: float = info(0.115)  # registration collar inner half-span (flange 0.110)
    collar_t: float = info(0.010)
    collar_top_z: float = info(0.124)  # world z of the collar top edge
    pad_size: tuple = info((0.240, 0.240, 0.006))
    pad_z: float = info(0.003)

    # --- info: balls -------------------------------------------------------------------------
    ball_r: float = info(0.010)
    ball_slots: tuple = info(((0.052, 0.0), (-0.052, 0.0), (0.0, 0.052), (0.0, -0.052)))
    ball_slot_z: float = info(-0.0744)  # hover ~5 mm above the funnel surface at |x|=0.052
    depot: tuple = info((0.85, 0.40))  # absent balls parked here (0.08 m y-spacing)
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    stand_root_z: float = field(default=None, init=False)
    seated_root_z: float = field(default=None, init=False)
    seated_ext: float = field(default=None, init=False)
    park_root_z: float = field(default=None, init=False)
    stem_tip_local: float = field(default=None, init=False)  # caddy-local z at ext = 0

    def __post_init__(self) -> None:
        c = self
        c.stand_root_z = -c.base_z  # 0.108
        c.seated_root_z = c.rim_z - c.base_z  # 0.208
        c.stem_tip_local = c.plug_closed_z + c.stem_z - c.stem_size[2] / 2  # -0.106
        c.seated_ext = c.probe_top_z - (c.seated_root_z + c.stem_tip_local)  # 0.028
        c.park_root_z = c.pad_z + c.pad_size[2] / 2 - c.base_z  # 0.114
        tan = math.tan(math.radians(c.funnel_deg))
        head_half = c.head_size[0] / 2  # 0.044
        head_bot = c.plug_closed_z - c.head_size[2] / 2  # -0.096
        surf_at_rim = c.funnel_lip_z + (head_half - c.funnel_lip_x) * tan  # funnel z there
        ball_d = 2 * c.ball_r
        # (1) closed = sealed: the head rim dips >= 1.5 mm below the funnel surface line.
        assert head_bot <= surf_at_rim - 0.0015, (head_bot, surf_at_rim)
        # (2) seated-open feed gap at the head rim >= ball_d + 6 mm.
        assert (head_bot + c.seated_ext) - surf_at_rim >= ball_d + 0.006
        # (3) throat: between funnel lips and around the stem/probe, >= ball_d + 7.5 mm.
        assert c.funnel_lip_x - c.stem_size[0] / 2 >= ball_d + 0.0075
        assert c.hole_half - c.probe_size[0] / 2 >= ball_d + 0.008
        # (4) seating force: caddy weight >= 1.5x the spring force at the seated lift.
        assert c.caddy_mass * 9.81 >= 1.5 * c.spring_k * (c.seated_ext + c.spring_preload)
        # (5) sealed in ANY orientation: spring preload alone out-pulls the plug's weight.
        assert c.spring_k * c.spring_preload >= 1.5 * c.plug_mass * 9.81
        # (6) stem recessed above the base plane: nothing reaches it while standing.
        assert c.stem_tip_local >= c.base_z + 0.0015
        # (7) the flange out-spans the basin: it seats on the rim, never falls in.
        assert c.flange_size / 2 >= c.dock_floor_size[0] / 2 + 0.010
        # (8) the seated lift stays clear of the joint's upper stop.
        assert 0.004 <= c.seated_ext <= c.plug_stroke - 0.005
        # (9) probe top clears the sealed stem tip while the caddy hovers 35 mm up.
        assert c.seated_root_z + 0.035 + c.stem_tip_local > c.probe_top_z + 0.005
        # (10) the 45-deg ridge tent's bottom vertex stays >= 4 mm above the ground
        # while the caddy stands sealed (a lower vertex gets ground-depenetrated OPEN).
        ridge_zext = (c.ridge_size[0] / 2 + c.ridge_size[2] / 2) * math.cos(math.pi / 4)
        assert c.stand_root_z + c.plug_closed_z + c.ridge_z - ridge_zext >= 0.004
        # (11) registration collar: 3-8 mm play around the flange; at max play the stem
        # still overlaps the probe top by >= 6 mm and the probe clears the base hole.
        play = c.collar_inner - c.flange_size / 2
        assert 0.003 <= play <= 0.008, play
        assert c.stem_size[0] / 2 + c.probe_size[0] / 2 - play >= 0.006
        assert c.hole_half - c.probe_size[0] / 2 - play >= 0.002
        # (12) the hovering flange bottom enters the collar from above with clearance.
        assert c.seated_root_z + 0.035 + c.base_z >= c.collar_top_z + 0.005


def _quat_z(rad: float) -> tuple:
    return (math.cos(rad / 2), 0.0, 0.0, math.sin(rad / 2))


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("ice_caddy_dock")
class IceCaddyDockScene(BaseScene):
    cfg: IceCaddyDockSceneCfg

    def __init__(self, cfg: IceCaddyDockSceneCfg | None = None) -> None:
        super().__init__(cfg or IceCaddyDockSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the dynamic caddy (roof root; compound authored in bind), the
        dynamic plug (head root; stem + ridge authored in bind), kinematic dock / decoy
        (floor roots; walls + probe authored in bind), the kinematic pad, and 4 balls."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        live = dict(sleep_threshold=0.0, stabilization_threshold=0.0,
                    solver_position_iteration_count=16, solver_velocity_iteration_count=4,
                    max_depenetration_velocity=0.5)

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
        # Kinematic fixtures: dock, decoy (floor slabs; superstructure authored in bind).
        for name, color in (("dock", (0.20, 0.30, 0.55)), ("decoy", (0.45, 0.30, 0.20))):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.title(),
                spawn=sim_utils.CuboidCfg(
                    size=c.dock_floor_size, rigid_props=kin, collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.55, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.3, c.dock_floor_z)),
            )
        out["pad"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Pad",
            spawn=sim_utils.CuboidCfg(
                size=c.pad_size, rigid_props=kin, collision_props=coll,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.7, dynamic_friction=0.65, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.60, 0.25))),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.12, 0.0, c.pad_z)),
        )
        # The caddy: roof cuboid is the ROOT (compound children authored in bind).
        out["caddy"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Caddy",
            spawn=sim_utils.CuboidCfg(
                size=c.roof_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=0.05, angular_damping=0.05, **live),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.caddy_mass),
                collision_props=coll,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.4, dynamic_friction=0.35, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.88, 0.92))),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.13, 0.30, c.stand_root_z)),
        )
        # The plug: head cuboid is the ROOT (stem + ridge authored in bind). Slick.
        out["plug"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plug",
            spawn=sim_utils.CuboidCfg(
                size=c.head_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=0.1, angular_damping=0.1, **live),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.plug_mass),
                collision_props=coll,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.04, dynamic_friction=0.04, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.35, 0.10))),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.13, 0.30, c.stand_root_z + c.plug_closed_z)),
        )
        for i in range(c.n_balls_max):
            out[f"ball_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball" + str(i),
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        linear_damping=0.2, angular_damping=0.3, **live),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.3, dynamic_friction=0.25, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.75, 0.92, 1.0))),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.depot[0], c.depot[1] + 0.08 * i, c.ball_r + 0.001)),
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

    # ----- lifecycle ---------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n = env.num_envs
        dev = env.device
        c = self.cfg
        self.caddy: RigidObject = env.iscene["caddy"]
        self.plug: RigidObject = env.iscene["plug"]
        self.dock: RigidObject = env.iscene["dock"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.pad: RigidObject = env.iscene["pad"]
        self.balls: list[RigidObject] = [env.iscene[f"ball_{i}"] for i in range(c.n_balls_max)]
        self.env_origins = env.iscene.env_origins
        self._author_children()
        self._author_joints()
        # Episode state.
        self.side = torch.ones(n, device=dev)  # +1 / -1: which y-side the real dock is on
        self.dock_xy = torch.zeros(n, 2, device=dev)  # env-local fixture centres
        self.decoy_xy = torch.zeros(n, 2, device=dev)
        self.pad_xy = torch.zeros(n, 2, device=dev)
        self.n_balls = torch.full((n,), c.n_balls_max, dtype=torch.long, device=dev)
        self.present = torch.zeros(n, c.n_balls_max, dtype=torch.bool, device=dev)
        self.lift_latch = torch.zeros(n, device=dev)
        self.dock_open_latch = torch.zeros(n, device=dev)
        self.ball_latch = torch.zeros(n, c.n_balls_max, device=dev)
        # External probe inputs (WORLD frame; post_step converts with the live quaternion
        # and owns both wrench slots — never call set_external_force_and_torque directly).
        self.caddy_force_w = torch.zeros(n, 3, device=dev)
        self.caddy_torque_w = torch.zeros(n, 3, device=dev)
        self.plug_force_w = torch.zeros(n, 3, device=dev)

    # -- authored geometry (env_0 only — env_1.. compose by reference; idempotent) -------------
    def _author_children(self) -> None:
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath("/World/envs/env_0/Caddy/wall_xp").IsValid():
            return

        def mat(path: str, static: float, dynamic: float):
            m = UsdShade.Material.Define(stage, path)
            pm = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
            pm.CreateStaticFrictionAttr(float(static))
            pm.CreateDynamicFrictionAttr(float(dynamic))
            pm.CreateRestitutionAttr(0.0)
            return m

        slick = mat("/World/envs/env_0/Looks/slick_phys", 0.04, 0.04)
        grippy = mat("/World/envs/env_0/Looks/grippy_phys", 0.5, 0.45)

        def box(path: str, size, center, color, quat=None, material=None) -> None:
            cube = UsdGeom.Cube.Define(stage, path)
            cube.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(cube.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(*center))
            if quat is not None:
                xf.AddOrientOp().Set(Gf.Quatf(quat[0], quat[1], quat[2], quat[3]))
            xf.AddScaleOp().Set(Gf.Vec3f(*size))
            cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            px = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
            px.CreateContactOffsetAttr(c.contact_offset)
            px.CreateRestOffsetAttr(0.0)
            if material is not None:
                UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
                    material, UsdShade.Tokens.weakerThanDescendants, "physics")

        # ---- caddy compound (root = roof at the local origin) ----
        C = "/World/envs/env_0/Caddy"
        body = (0.85, 0.88, 0.92)
        wz = -(c.roof_size[2] / 2 + c.wall_h / 2)  # -0.056
        box(f"{C}/wall_xp", (c.wall_t, c.roof_size[1], c.wall_h), (0.074, 0.0, wz), body)
        box(f"{C}/wall_xm", (c.wall_t, c.roof_size[1], c.wall_h), (-0.074, 0.0, wz), body)
        box(f"{C}/wall_yp", (0.140, c.wall_t, c.wall_h), (0.0, 0.074, wz), body)
        box(f"{C}/wall_ym", (0.140, c.wall_t, c.wall_h), (0.0, -0.074, wz), body)
        fz = c.base_z + c.flange_t / 2  # -0.104
        strip = (c.flange_size - 2 * c.hole_half) / 2  # 0.074
        fx = c.hole_half + strip / 2  # 0.073
        box(f"{C}/flange_xp", (strip, c.flange_size, c.flange_t), (fx, 0.0, fz), body)
        box(f"{C}/flange_xm", (strip, c.flange_size, c.flange_t), (-fx, 0.0, fz), body)
        box(f"{C}/flange_yp", (2 * c.hole_half, strip, c.flange_t), (0.0, fx, fz), body)
        box(f"{C}/flange_ym", (2 * c.hole_half, strip, c.flange_t), (0.0, -fx, fz), body)
        # Funnel plates: 30-deg slopes from the walls down to the throat lip. Slick.
        half = math.radians(c.funnel_deg) / 2
        qy_m = (math.cos(half), 0.0, -math.sin(half), 0.0)  # Ry(-30): rises toward +x
        qy_p = (math.cos(half), 0.0, +math.sin(half), 0.0)
        qx_p = (math.cos(half), +math.sin(half), 0.0, 0.0)  # Rx(+30): rises toward +y
        qx_m = (math.cos(half), -math.sin(half), 0.0, 0.0)
        fun = (0.55, 0.60, 0.70)
        fcx, fcz = 0.0540, -0.0917  # plate centre (derived from the lip point)
        sx = (c.funnel_len, c.funnel_w, c.funnel_t)
        sy = (c.funnel_w, c.funnel_len, c.funnel_t)
        box(f"{C}/funnel_xp", sx, (fcx, 0.0, fcz), fun, quat=qy_m, material=slick)
        box(f"{C}/funnel_xm", sx, (-fcx, 0.0, fcz), fun, quat=qy_p, material=slick)
        box(f"{C}/funnel_yp", sy, (0.0, fcx, fcz), fun, quat=qx_p, material=slick)
        box(f"{C}/funnel_ym", sy, (0.0, -fcx, fcz), fun, quat=qx_m, material=slick)
        # Bridge handle (the Franka grasp: 14 mm bar, 30 mm knuckle clearance).
        post_z = c.roof_size[2] / 2 + c.post_size[2] / 2  # 0.019
        bar = (0.20, 0.22, 0.26)
        box(f"{C}/post_p", c.post_size, (c.post_x, 0.0, post_z), bar, material=grippy)
        box(f"{C}/post_m", c.post_size, (-c.post_x, 0.0, post_z), bar, material=grippy)
        box(f"{C}/bar", c.bar_size, (0.0, 0.0, c.bar_z), bar, material=grippy)

        # ---- plug children (root = head) ----
        P = "/World/envs/env_0/Plug"
        orange = (0.80, 0.35, 0.10)
        box(f"{P}/stem", c.stem_size, (0.0, 0.0, c.stem_z), orange, material=slick)
        q45 = (math.cos(math.pi / 8), 0.0, math.sin(math.pi / 8), 0.0)  # Ry(45)
        box(f"{P}/ridge", c.ridge_size, (0.0, 0.0, c.ridge_z), orange,
            quat=q45, material=slick)

        # ---- dock / decoy superstructure (roots = floor slabs) ----
        for root, has_probe, col in (("Dock", True, (0.20, 0.30, 0.55)),
                                     ("Decoy", False, (0.45, 0.30, 0.20))):
            D = f"/World/envs/env_0/{root}"
            wall_h = c.rim_z - 2 * c.dock_floor_z  # 0.092 (floor top -> rim)
            wcz = 2 * c.dock_floor_z + wall_h / 2 - c.dock_floor_z  # rel z 0.050
            wx = c.dock_floor_size[0] / 2 - c.dock_wall_t / 2  # 0.091
            inner = c.dock_floor_size[0] - 2 * c.dock_wall_t  # 0.174
            box(f"{D}/wall_xp", (c.dock_wall_t, c.dock_floor_size[1], wall_h),
                (wx, 0.0, wcz), col)
            box(f"{D}/wall_xm", (c.dock_wall_t, c.dock_floor_size[1], wall_h),
                (-wx, 0.0, wcz), col)
            box(f"{D}/wall_yp", (inner, c.dock_wall_t, wall_h), (0.0, wx, wcz), col)
            box(f"{D}/wall_ym", (inner, c.dock_wall_t, wall_h), (0.0, -wx, wcz), col)
            # Registration collar: four slick guide bars OUTSIDE the rim capture the
            # descending flange with +-5 mm play — the caddy self-centres while it rides
            # the slick probe pin down (without them it tips and skates off the stem).
            gx = c.collar_inner + c.collar_t / 2  # bar centre half-span
            gh = c.collar_top_z - 0.092  # world 0.092 -> collar_top_z
            gcz = c.collar_top_z - gh / 2 - c.dock_floor_z  # rel z of bar centre
            glen = 2 * gx + c.collar_t
            box(f"{D}/guide_xp", (c.collar_t, glen, gh), (gx, 0.0, gcz), col,
                material=slick)
            box(f"{D}/guide_xm", (c.collar_t, glen, gh), (-gx, 0.0, gcz), col,
                material=slick)
            box(f"{D}/guide_yp", (glen, c.collar_t, gh), (0.0, gx, gcz), col,
                material=slick)
            box(f"{D}/guide_ym", (glen, c.collar_t, gh), (0.0, -gx, gcz), col,
                material=slick)
            if has_probe:
                pz = c.dock_floor_z + c.probe_size[2] / 2  # rel 0.065 -> top world 0.130
                box(f"{D}/probe", c.probe_size, (0.0, 0.0, pz), (0.9, 0.75, 0.2),
                    material=slick)

    def _author_joints(self) -> None:
        """Per env: Z-axis prismatic caddy -> plug, limits [0, stroke] (0 = sealed), with
        a linear DriveAPI return spring whose target sits BELOW the lower stop (preload:
        the plug presses onto its seat with spring_k * spring_preload, any orientation)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/plug_slide")
            j.CreateBody0Rel().SetTargets([f"{base}/Caddy"])
            j.CreateBody1Rel().SetTargets([f"{base}/Plug"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.plug_closed_z))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(0.0)
            j.CreateUpperLimitAttr(c.plug_stroke)
            drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
            drv.CreateTypeAttr("force")
            drv.CreateStiffnessAttr(float(c.spring_k))
            drv.CreateDampingAttr(float(c.spring_c))
            drv.CreateTargetPositionAttr(float(-c.spring_preload))

    # ----- reset -------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the dock side, fixture jitters, caddy pose (free yaw) and
        the ball count; write the kinematic fixtures, then the caddy + plug + balls as ONE
        consistent linkage (plug sealed, balls hovering ~5 mm over their funnel slots);
        clear latches and probe buffers."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        torch.rand(1, device=dev)  # burn the degenerate first post-seed draw
        u = torch.rand(m, 10, device=dev)
        side = torch.where(u[:, 0] < 0.5, -1.0, 1.0)
        self.side[env_ids] = side
        jit = lambda col: (2.0 * u[:, col] - 1.0)  # noqa: E731
        dock = torch.stack([c.dock_x + c.dock_jit * jit(1),
                            side * (c.dock_y + c.dock_jit * jit(2))], dim=1)
        decoy = torch.stack([c.dock_x + c.dock_jit * jit(3),
                             -side * (c.dock_y + c.dock_jit * jit(4))], dim=1)
        pad = torch.stack([c.pad_x + c.pad_jit * jit(5), c.pad_y_jit * jit(6)], dim=1)
        caddy = torch.stack([c.caddy_x + c.caddy_jit * jit(7),
                             side * (c.caddy_y + c.caddy_jit * jit(8))], dim=1)
        yaw = 2.0 * math.pi * u[:, 9]
        nb = c.n_balls_min + torch.clamp(
            (torch.rand(m, device=dev) * (c.n_balls_max - c.n_balls_min + 1)).long(),
            max=c.n_balls_max - c.n_balls_min)
        self.dock_xy[env_ids] = dock
        self.decoy_xy[env_ids] = decoy
        self.pad_xy[env_ids] = pad
        self.n_balls[env_ids] = nb
        for i in range(c.n_balls_max):
            self.present[env_ids, i] = i < nb
        self.lift_latch[env_ids] = 0.0
        self.dock_open_latch[env_ids] = 0.0
        self.ball_latch[env_ids] = 0.0
        self.caddy_force_w[env_ids] = 0.0
        self.caddy_torque_w[env_ids] = 0.0
        self.plug_force_w[env_ids] = 0.0

        def kin_write(body, xy, z: float) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        kin_write(self.dock, dock, c.dock_floor_z)
        kin_write(self.decoy, decoy, c.dock_floor_z)
        kin_write(self.pad, pad, c.pad_z)

        # Caddy 3 mm above its standing height (settles clean; no depenetration latch).
        cz = c.stand_root_z + 0.003
        cq = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw),
                          torch.zeros_like(yaw), torch.sin(yaw / 2)], dim=1)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = caddy
        st[:, 2] = cz
        st[:, 3:7] = cq
        st[:, 0:3] += origin
        self.caddy.write_root_state_to_sim(st, env_ids)
        # Plug sealed, SAME frame (the linkage moves as one).
        ps = st.clone()
        ps[:, 2] += c.plug_closed_z
        self.plug.write_root_state_to_sim(ps, env_ids)
        # Balls: present -> their funnel slots (rotated into the caddy frame, hovering);
        # absent -> the depot on open ground.
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        for i in range(c.n_balls_max):
            bx, by = c.ball_slots[i]
            bs = torch.zeros(m, 13, device=dev)
            bs[:, 3] = 1.0
            here = self.present[env_ids, i]
            bs[:, 0] = torch.where(here, caddy[:, 0] + bx * cy - by * sy,
                                   torch.full_like(cy, c.depot[0]))
            bs[:, 1] = torch.where(here, caddy[:, 1] + bx * sy + by * cy,
                                   torch.full_like(cy, c.depot[1] + 0.08 * i))
            bs[:, 2] = torch.where(here, cz + c.ball_slot_z,
                                   torch.full_like(cy, c.ball_r + 0.001))
            bs[:, 0:3] += origin
            self.balls[i].write_root_state_to_sim(bs, env_ids)

    # ----- readings ----------------------------------------------------------------------------
    def caddy_pos(self) -> torch.Tensor:
        """(N, 3) env-local caddy root position."""
        return self.caddy.data.root_pos_w - self.env_origins

    def plug_ext(self) -> torch.Tensor:
        """(N,) plug extension in m (0 = sealed): caddy-frame z of the plug root minus
        the sealed offset."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = self.plug.data.root_pos_w - self.caddy.data.root_pos_w
        return quat_apply_inverse(self.caddy.data.root_quat_w, rel)[:, 2] \
            - self.cfg.plug_closed_z

    def upright(self) -> torch.Tensor:
        """(N,) bool: caddy z-axis within `upright_cos` of world up."""
        q = self.caddy.data.root_quat_w
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return 1.0 - 2.0 * (x * x + y * y) >= self.cfg.upright_cos

    def ball_pos(self) -> torch.Tensor:
        """(N, n_max, 3) env-local ball positions."""
        return torch.stack([b.data.root_pos_w - self.env_origins for b in self.balls], dim=1)

    def ball_vel(self) -> torch.Tensor:
        """(N, n_max) ball speeds."""
        return torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.balls], dim=1)

    def in_basin(self, vel_gate: float) -> torch.Tensor:
        """(N, n_max) bool: ball inside the REAL dock's basin band, slower than
        `vel_gate` (the basin: walls to rim_z, floor top at 2 * dock_floor_z)."""
        c = self.cfg
        p = self.ball_pos()
        d = p[:, :, 0:2] - self.dock_xy.unsqueeze(1)
        return (d.abs().amax(dim=-1) <= c.ball_xy_tol) \
            & (p[:, :, 2] > c.ball_z_lo) & (p[:, :, 2] < c.ball_z_hi) \
            & (self.ball_vel() < vel_gate)

    def parked(self) -> torch.Tensor:
        """(N,) bool: caddy upright on the pad at its standing height."""
        c = self.cfg
        p = self.caddy_pos()
        on_pad = ((p[:, 0:2] - self.pad_xy).abs().amax(dim=-1) <= c.park_xy_tol) \
            & ((p[:, 2] - c.park_root_z).abs() <= c.park_z_tol)
        return on_pad & self.upright()

    def settled(self) -> torch.Tensor:
        """(N,) bool: caddy, plug and every present ball at rest."""
        c = self.cfg
        still = (self.caddy.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.caddy.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang) \
            & (self.plug.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
        ball_still = (self.ball_vel() < c.settle_lin) | ~self.present
        return still & ball_still.all(dim=1)

    def success(self) -> torch.Tensor:
        """(N,) bool, current physical state: every present ball rests inside the REAL
        dock basin, the caddy is parked upright on the pad with the valve re-sealed,
        and everything is still. A ball left inside (or dumped anywhere else), a caddy
        left on the dock, tipped, or held open — all fail."""
        c = self.cfg
        balls_ok = (self.in_basin(c.settle_lin) | ~self.present).all(dim=1)
        return balls_ok & self.parked() & (self.plug_ext() <= c.closed_ext) & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.12 * lift + 0.18 * dock-open + 0.40 * delivered
        fraction + 0.30 * current success. Exactly 1.0 iff success() (success forces all
        latches full); ~0 for the null policy; a seed-style press-at-the-machine strategy
        (no carried container) has nothing to press — every point routes through carrying
        the caddy to the dock."""
        succ = self.success().float()
        nb = self.n_balls.clamp(min=1).float()
        delivered = (self.ball_latch * self.present.float()).sum(dim=1) / nb
        return (0.12 * torch.maximum(self.lift_latch, succ)
                + 0.18 * torch.maximum(self.dock_open_latch, succ)
                + 0.40 * torch.maximum(delivered, succ)
                + 0.30 * succ)

    # ----- step-coupled mechanics (every substep) ----------------------------------------------
    def post_step(self) -> None:
        """Plant: the WORLD-frame probe buffers, converted to BODY frame with the live
        quaternion (set_external applies body-frame vectors on this stack), on the caddy
        and the plug. Then latch rubric progress (NaN earns nothing)."""
        from isaaclab.utils.math import quat_apply_inverse

        n = self.env.num_envs
        dev = self.env.device
        c = self.cfg
        qc = self.caddy.data.root_quat_w
        qp = self.plug.data.root_quat_w
        self.caddy.set_external_force_and_torque(
            quat_apply_inverse(qc, self.caddy_force_w).unsqueeze(1),
            quat_apply_inverse(qc, self.caddy_torque_w).unsqueeze(1))
        self.plug.set_external_force_and_torque(
            quat_apply_inverse(qp, self.plug_force_w).unsqueeze(1),
            torch.zeros(n, 1, 3, device=dev))

        pos = self.caddy_pos()
        lifted = (pos[:, 2] > c.lift_z).float()
        near_dock = (pos[:, 0:2] - self.dock_xy).norm(dim=-1) <= c.dock_xy_tol
        opened = ((self.plug_ext() > c.open_ext) & near_dock).float()
        in_b = self.in_basin(c.ball_latch_vel).float()
        succ = self.success().float()  # the legit path implies every latch
        lifted = torch.maximum(lifted, succ)
        opened = torch.maximum(opened, succ)
        in_b = torch.maximum(in_b, succ.unsqueeze(1))
        # A diverged substep must not latch: torch.maximum propagates NaN.
        lifted = torch.nan_to_num(lifted, nan=0.0, posinf=0.0, neginf=0.0)
        opened = torch.nan_to_num(opened, nan=0.0, posinf=0.0, neginf=0.0)
        in_b = torch.nan_to_num(in_b, nan=0.0, posinf=0.0, neginf=0.0)
        self.lift_latch = torch.maximum(self.lift_latch, lifted)
        self.dock_open_latch = torch.maximum(self.dock_open_latch, opened)
        self.ball_latch = torch.maximum(self.ball_latch, in_b)

    # ----- state (full, restorable) ------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {nm: b.data.root_state_w[env_ids].clone()
                  for nm, b in (("caddy", self.caddy), ("plug", self.plug),
                                ("dock", self.dock), ("decoy", self.decoy),
                                ("pad", self.pad))}
        for i, b in enumerate(self.balls):
            bodies[f"ball_{i}"] = b.data.root_state_w[env_ids].clone()
        return {
            "bodies": bodies,
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("side", "dock_xy", "decoy_xy", "pad_xy", "n_balls",
                               "present", "lift_latch", "dock_open_latch", "ball_latch",
                               "caddy_force_w", "caddy_torque_w", "plug_force_w")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        named = {"caddy": self.caddy, "plug": self.plug, "dock": self.dock,
                 "decoy": self.decoy, "pad": self.pad}
        for i, b in enumerate(self.balls):
            named[f"ball_{i}"] = b
        for nm, st in state["bodies"].items():
            named[nm].write_root_state_to_sim(st, env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    # ----- description -------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"On the floor stands a pale-blue ICE CADDY ({c.flange_size * 1000:.0f} mm "
            f"square base, {(-c.base_z + c.bar_z) * 1000:.0f} mm tall to its bridge "
            f"handle) holding {c.n_balls_min}-{c.n_balls_max} ice balls (count sampled "
            f"each episode). Its top is SEALED; the only opening is a bottom port closed "
            f"by an orange spring-loaded poppet plug whose push-stem is recessed above "
            f"the base plane — tilting, shaking or inverting the caddy releases nothing, "
            f"and nothing can reach the stem while the caddy stands. Nearby sit two "
            f"look-alike basins: the BLUE dispensing dock, which carries a yellow probe "
            f"post rising {(c.probe_top_z - c.rim_z) * 1000:.0f} mm above its rim at the "
            f"centre, and a BROWN decoy basin with no post. A green pad lies between "
            f"them. Which side (left/right) the real dock is on, all fixture positions, "
            f"the caddy's pose and the ball count are sampled fresh every episode.\n"
            f"Goal: seat the caddy squarely on the BLUE dock's rim — the probe post "
            f"passes through the bottom port and presses the plug open against its "
            f"spring under the caddy's own weight — wait for every ice ball to drain "
            f"through the funnel into the dock basin, then lift the caddy off (the "
            f"spring re-seals the port) and park it upright on the green pad. Success = "
            f"all balls resting inside the BLUE basin, the caddy standing upright on "
            f"the pad with its valve closed, everything still. Balls dumped anywhere "
            f"else, a ball left inside, or a caddy left on the dock all fail."
        )

    def instruction(self) -> str:
        return (
            "Get the ice out of the sealed caddy: carry the caddy to the blue dispensing "
            "dock and seat it on the rim so the dock's probe post opens the spring-loaded "
            "bottom valve, wait until every ice ball drains into the dock basin, then "
            "lift the caddy away — the valve snaps shut — and park it upright on the "
            "green pad. The brown basin is a decoy with no post: nothing opens there, "
            "and no amount of tilting or shaking gets ice past the sealed valve."
        )


# ----- runnable env: scene physics only (NullRobot smoke) -> "simgen.ice_caddy_dock" -----------
register_env("simgen", lambda: EnvCfg(scene="ice_caddy_dock", robot="null"))

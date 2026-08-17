"""PuddingCarouselScene — align a shrouded turntable, side-load the pudding box into
its bay, then rotate the loaded drum to the sealed stop.

Derived from the LIBERO scene10 seed "put the chocolate pudding in the top drawer of
the cabinet and close it" but STRATEGICALLY DIFFERENT (see TASK.md): the seed's plan
is pick the box, DROP it into an open PRISMATIC drawer from above, then push the
drawer shut — the receptacle is passive, top access is free, and the two goal clauses
are order-free. Here the receptacle is a REVOLUTE carousel drum inside a fixed shroud:
its single bay opens SIDEWAYS through the shroud's only window, and the same drum
rotation that closes the vault carries the payload with it. The plan skeleton becomes
ALIGN (rotate the drum down to its 0 deg stop so bay and window line up) -> LOAD
(slide the box horizontally through the window, across the sill, deep into the bay) ->
SEAL (rotate the loaded drum to its ~92 deg stop, where the bay faces a solid wall).
Both orderings are forced by geometry, not by rubric fiat: a misaligned drum presents
a curved rim to the window (nothing box-sized enters — the drum ROOF also kills the
seed's drop-from-above move), and a sealed drum blocks the window with its flank.

Mechanics (plain rigid bodies + authored USD joints, the fridge/oven-dials pattern):
the shroud is kinematic slabs; the drum is ONE dynamic body (disc root) with the bay
walls, full-disc roof, hub and overhead lever bar authored as compound collision
children; a Z-axis revolute joint (limits [0, 92] deg; 0 = bay aligned with the
window, 92 = sealed) hangs it on the kinematic plinth. `post_step` applies viscous
hinge friction and consumes two drive buffers — `drum_drive` (torque about the axis)
and `box_drive` (world-frame force on the pudding box, pre-encoded against the
rotation-since-write wrench frame) — nothing else touches those wrench slots.

Rubric (graded 0..1, latching transient achievement — anchored in the demonstrated
solve.py trajectory: align, load, seal):
  - `align_latch` (0.10): the drum has ever been at the aligned stop (theta <= 6 deg);
    also forced by a successful load (loading proves alignment happened);
  - `load_latch`  (0.35): the pudding box has ever rested inside the bay (drum frame);
  - `seal_latch`  (0.30): best sealing progress (theta / theta_closed) reached WHILE
    the box was in the bay; forced to 1.0 the moment the vault is closed with the box
    aboard;
  - current success (0.25): box in the bay AND drum within `closed_tol_deg` of the
    sealed stop AND everything settled -> score == 1.0 iff success(), ~0 for the null
    policy (the drum starts >= 48 deg from aligned, the box starts on the floor).

Per-episode randomization (readback-verified in smoke): drum start angle, which floor
slot the pudding vs the white decoy box starts in, per-box xy jitter and free yaw.

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
class PuddingCarouselSceneCfg(BaseCfg):
    """Config for `PuddingCarouselScene`. Geometry is derived once in `__post_init__` so
    the scene, the smoke AND the solver read the same numbers."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    closed_tol_deg: float = tunable(5.0)  # drum within this of the 92 deg stop = sealed
    aligned_tol_deg: float = tunable(6.0)  # drum within this of the 0 deg stop = aligned
    # Bay containment band, DRUM body frame (see the structure block for the geometry):
    # honest by construction — a box resting anywhere fully inside the bay has
    # x in [-0.0875, -0.0225], |y| <= 0.020, z ~= 0.035; one protruding past the disc
    # rim has x < -0.088, one outside the drum entirely is >= 0.10 away.
    bay_x_lo: float = tunable(-0.088)
    bay_x_hi: float = tunable(-0.012)
    bay_y_tol: float = tunable(0.032)
    bay_z_lo: float = tunable(0.018)
    bay_z_hi: float = tunable(0.065)
    settle_box: float = tunable(0.05)  # max box |lin vel| (m/s) when judging
    settle_drum: float = tunable(0.08)  # max drum |omega_z| (rad/s) when judging

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    theta0_min_deg: float = tunable(48.0)  # drum start angle sampled in [min, max];
    # 48 deg keeps the bay/window angular overlap <= 0 (half-angles 22.4 + 21.4 deg),
    # so nothing box-sized enters before an explicit align (verified in smoke).
    theta0_max_deg: float = tunable(75.0)
    box_jitter: float = tunable(0.035)  # uniform +- xy jitter of each floor box

    # --- tunable: drum plant (difficulty dials) ----------------------------------------------
    # Viscous-only hinge friction: a gentle lever push never stalls and the drum rests
    # wherever it stops (coast after a 0.45 rad/s cut is ~2 deg).
    drum_damping: float = tunable(0.40)  # viscous axis friction (N*m*s/rad)
    drum_mass: float = tunable(1.8)
    box_mass: float = tunable(0.10)
    box_size: float = tunable(0.055)  # pudding box edge (cube)

    # --- info: structure (env-local coordinates) ---------------------------------------------
    # Carousel axis at (0.50, 0.00). Drum: disc r=0.115 (top z=0.040 = bay floor), one
    # bay opening along drum LOCAL -x (worn world -x when aligned), interior x
    # [-0.115, +0.005], clear width 0.095, height 0.090, full-disc roof (top z=0.140),
    # hub + red lever bar at z~0.186 sweeping ABOVE the 0.15-tall shroud walls.
    # Shroud inner faces 0.14 from the axis; the only aperture is the front window
    # (y +-0.055, z 0.040..0.125) over a deep sill whose top is FLUSH with the bay
    # floor and whose inner edge stops 3 mm short of the disc rim. Max swept corner
    # radius of an in-bay box is 0.130 < 0.14 - contact offsets: a contained load
    # never jams the rotation.
    axis_xy: tuple = info((0.50, 0.0))
    disc_r: float = info(0.115)
    disc_h: float = info(0.015)
    disc_z: float = info(0.0325)  # disc centre height = drum root origin height
    bay_halfw: float = info(0.0475)  # clear half-width between the bay side walls
    bay_wall_t: float = info(0.015)
    bay_wall_h: float = info(0.090)
    # Bay walls (and everything above them) start 5 mm ABOVE the disc top: the wall
    # corners sweep at radius 0.131 > the sill's inner reach 0.118, and with wall
    # bottoms coincident with the sill top plane the drum drags on the shelf and the
    # 0.8 N*m lever authority stalls (measured). The 5 mm undercut clears the sill;
    # a 55 mm box cannot pass under it.
    bay_wall_lift: float = info(0.005)
    bay_back_x: float = info(0.0125)  # back wall centre, drum local x
    roof_h: float = info(0.010)
    hub_r: float = info(0.020)
    hub_h: float = info(0.035)
    lever_len: float = info(0.18)  # bar from local x -0.10 to -0.28, along the bay opening
    lever_sq: float = info(0.022)
    plinth_half: float = info(0.12)
    plinth_h: float = info(0.02)
    shroud_h: float = info(0.15)
    shroud_t: float = info(0.02)
    shroud_inner: float = info(0.14)  # inner faces this far from the axis
    win_half_y: float = info(0.055)  # window aperture y half-width
    win_z_top: float = info(0.125)  # aperture spans z [sill_top, win_z_top]
    sill_x: tuple = info((0.325, 0.382))  # deep sill under the window, top z = bay floor
    drum_limit_deg: float = info(92.0)
    # Floor slots the two boxes start in (target/decoy assignment sampled per episode).
    slot_a: tuple = info((0.19, 0.15))
    slot_b: tuple = info((0.19, -0.15))
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    bay_floor_z: float = field(default=None, init=False)  # world z of the bay floor / sill top
    box_half: float = field(default=None, init=False)
    box_rest_z: float = field(default=None, init=False)  # world z of a box on floor/sill/bay
    sill_center: tuple = field(default=None, init=False)  # world xy centre of the sill shelf
    bay_rest_local_z: float = field(default=None, init=False)  # in-bay box centre, drum frame

    def __post_init__(self) -> None:
        self.bay_floor_z = self.disc_z + self.disc_h / 2  # 0.040
        self.box_half = self.box_size / 2
        self.box_rest_z = self.bay_floor_z + self.box_half  # on sill or bay floor
        self.sill_center = ((self.sill_x[0] + self.sill_x[1]) / 2, 0.0)
        self.bay_rest_local_z = self.bay_floor_z + self.box_half - self.disc_z  # +0.035


def _quat_z(rad: float) -> tuple:
    return (math.cos(rad / 2), 0.0, 0.0, math.sin(rad / 2))


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("pudding_carousel")
class PuddingCarouselScene(BaseScene):
    cfg: PuddingCarouselSceneCfg

    def __init__(self, cfg: PuddingCarouselSceneCfg | None = None) -> None:
        super().__init__(cfg or PuddingCarouselSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic shroud (plinth, back/side walls, windowed front:
        sill + pillars + lintel), the dynamic drum disc (bay/roof/lever children are
        authored in bind), the pudding box and the white decoy box."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        ax, ay = c.axis_xy
        slate = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.40, 0.48))
        dark = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.16, 0.16, 0.18))
        wood = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.38, 0.20))
        brown = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.23, 0.10))
        white = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.93, 0.92, 0.88))
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        # post_step drives drum and box with external wrenches that do NOT wake a
        # sleeping body — sleep_threshold=0 keeps both plants live.
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
        }
        sh, st = c.shroud_h, c.shroud_t
        inner = c.shroud_inner
        sill_lo, sill_hi = c.sill_x
        slabs = {
            # plinth the drum hangs over (joint body0)
            "plinth": ((ax, ay, c.plinth_h / 2),
                       (2 * c.plinth_half, 2 * c.plinth_half, c.plinth_h), dark),
            "shroud_back": ((ax + inner + st / 2, ay, sh / 2), (st, 2 * inner + 2 * st, sh), slate),
            "shroud_left": ((ax, ay + inner + st / 2, sh / 2), (2 * inner, st, sh), slate),
            "shroud_right": ((ax, ay - inner - st / 2, sh / 2), (2 * inner, st, sh), slate),
            # windowed front (wall plane x = ax - inner - st/2 .. ax - inner):
            "front_sill": (((sill_lo + sill_hi) / 2, ay, c.bay_floor_z / 2),
                           (sill_hi - sill_lo, 2 * inner + 2 * st, c.bay_floor_z), slate),
            "front_pillar_l": ((ax - inner - st / 2, ay + (c.win_half_y + inner + st) / 2,
                                (sh + c.bay_floor_z) / 2),
                               (st, inner + st - c.win_half_y, sh - c.bay_floor_z), slate),
            "front_pillar_r": ((ax - inner - st / 2, ay - (c.win_half_y + inner + st) / 2,
                                (sh + c.bay_floor_z) / 2),
                               (st, inner + st - c.win_half_y, sh - c.bay_floor_z), slate),
            "front_lintel": ((ax - inner - st / 2, ay, (c.win_z_top + sh) / 2),
                             (st, 2 * c.win_half_y, sh - c.win_z_top), slate),
        }
        for name, (pos, size, mat) in slabs.items():
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.title().replace("_", ""),
                spawn=sim_utils.CuboidCfg(
                    size=size, rigid_props=kin, collision_props=coll, visual_material=mat),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
            )
        out["drum"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Drum",
            spawn=sim_utils.CylinderCfg(
                radius=c.disc_r,
                height=c.disc_h,
                axis="Z",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.5,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4,
                    **live),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.drum_mass),
                collision_props=coll,
                visual_material=wood,
            ),
            # spawned ALIGNED (theta = 0, bay opening along world -x); reset re-poses.
            init_state=RigidObjectCfg.InitialStateCfg(pos=(ax, ay, c.disc_z)),
        )
        for name, mat, slot in (("pudding", brown, c.slot_a), ("decoy", white, c.slot_b)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.title(),
                spawn=sim_utils.CuboidCfg(
                    size=(c.box_size,) * 3,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        linear_damping=0.05,
                        angular_damping=0.05,
                        **live),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.box_mass),
                    collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.50, dynamic_friction=0.45, restitution=0.05),
                    visual_material=mat,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(slot[0], slot[1], c.box_half + 0.002)),
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
                # both plants are driven by external wrenches every step
                "enable_external_forces_every_iteration": True,
            },
        )

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n = env.num_envs
        dev = env.device
        self.drum: RigidObject = env.iscene["drum"]
        self.pudding: RigidObject = env.iscene["pudding"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        self._author_drum_fixture()
        self._author_axle()
        # Episode state.
        self.theta0 = torch.full((n,), 1.0, device=dev)  # authored start angle (rad)
        self.align_latch = torch.zeros(n, device=dev)
        self.load_latch = torch.zeros(n, device=dev)
        self.seal_latch = torch.zeros(n, device=dev)
        # External drive inputs (solve.py and smoke probes write; post_step consumes and
        # owns both wrench slots — never call set_external_force_and_torque directly).
        self.drum_drive = torch.zeros(n, device=dev)  # torque about the axis (N*m, + seals)
        self.box_drive = torch.zeros(n, 3, device=dev)  # WORLD-frame force on the pudding (N)
        # Wrench-frame reference: the forge pod rotates applied wrenches by the body's
        # rotation-since-write; pre-encoding with q_ref * conj(q_now) undoes it. Updated
        # at reset and via mark_box_ref() after any teleport of the box.
        self._box_qref = torch.zeros(n, 4, device=dev)
        self._box_qref[:, 0] = 1.0

    def _author_drum_fixture(self) -> None:
        """Compound collision children on the drum disc (env_0 only — env_1.. compose
        from env_0 by reference; authored idempotently): bay side walls, bay back wall,
        the sealing flank wall, full-disc roof, hub and the red overhead lever bar
        (lever points along the bay opening, so it visually indicates the bay
        direction)."""
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath("/World/envs/env_0/Drum/bay_back").IsValid():
            return

        def _prep(prim) -> None:
            UsdPhysics.CollisionAPI.Apply(prim)
            px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            px.CreateContactOffsetAttr(c.contact_offset)
            px.CreateRestOffsetAttr(0.0)

        def box(name: str, center: tuple, size: tuple, color: tuple) -> None:
            cube = UsdGeom.Cube.Define(stage, f"/World/envs/env_0/Drum/{name}")
            cube.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(cube.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(*center))
            xf.AddScaleOp().Set(Gf.Vec3f(*size))
            cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            _prep(cube.GetPrim())

        def cyl(name: str, center_z: float, radius: float, height: float,
                color: tuple) -> None:
            cy = UsdGeom.Cylinder.Define(stage, f"/World/envs/env_0/Drum/{name}")
            cy.CreateRadiusAttr(radius)
            cy.CreateHeightAttr(height)
            cy.CreateAxisAttr("Z")
            cy.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                                 Gf.Vec3f(radius, radius, height / 2)])
            xf = UsdGeom.Xformable(cy.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, center_z))
            cy.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            _prep(cy.GetPrim())

        wood = (0.55, 0.38, 0.20)
        red = (0.85, 0.12, 0.10)
        disc_top = c.disc_h / 2 + c.bay_wall_lift  # walls start 5 mm above the bay floor
        wz = disc_top + c.bay_wall_h / 2
        wall_cx = (-c.disc_r + (c.bay_back_x - c.bay_wall_t / 2)) / 2  # spans rim..back face
        wall_len = (c.bay_back_x - c.bay_wall_t / 2) - (-c.disc_r)
        box("bay_wall_ypos", (wall_cx, c.bay_halfw + c.bay_wall_t / 2, wz),
            (wall_len, c.bay_wall_t, c.bay_wall_h), wood)
        box("bay_wall_yneg", (wall_cx, -c.bay_halfw - c.bay_wall_t / 2, wz),
            (wall_len, c.bay_wall_t, c.bay_wall_h), wood)
        box("bay_back", (c.bay_back_x, 0.0, wz),
            (c.bay_wall_t, 2 * c.bay_halfw + 2 * c.bay_wall_t + 0.01, c.bay_wall_h), wood)
        # Flank wall: a chord slab on the drum's local +y side, fully inside the rim
        # sweep (corners at radius sqrt(0.05^2 + 0.1025^2) = 0.114 < disc_r) so it adds
        # no clearance risk. When the drum is SEALED (+92 deg) this face turns toward
        # the window and physically closes it (the remaining sill gap is ~38 mm, less
        # than the box); when aligned it faces the side shroud wall, out of the way.
        box("flank", (0.0, 0.095, wz), (0.10, 0.015, c.bay_wall_h), wood)
        roof_z = disc_top + c.bay_wall_h + c.roof_h / 2
        cyl("roof", roof_z, c.disc_r, c.roof_h, wood)
        hub_z = roof_z + c.roof_h / 2 + c.hub_h / 2
        cyl("hub", hub_z, c.hub_r, c.hub_h, red)
        lever_z = hub_z + c.hub_h / 2 + c.lever_sq / 2
        box("lever", (-(0.10 + c.lever_len / 2), 0.0, lever_z),
            (c.lever_len, c.lever_sq, c.lever_sq), red)

    def _author_axle(self) -> None:
        """Per env: a Z-axis revolute joint plinth -> drum, limits [0, drum_limit_deg]
        (0 = bay aligned with the window, upper = sealed; the joint pair never
        collides — the stops are the joint's own limits)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        ax, ay = c.axis_xy
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/drum_axle")
            j.CreateBody0Rel().SetTargets([f"{base}/Plinth"])
            j.CreateBody1Rel().SetTargets([f"{base}/Drum"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.disc_z - c.plinth_h / 2))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(0.0)
            j.CreateUpperLimitAttr(c.drum_limit_deg)

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the drum at a sampled misaligned angle (pure joint-
        coordinate re-pose about the unchanged axle), sample which floor slot holds the
        pudding vs the decoy, drop both boxes with xy jitter + free yaw, clear latches
        and drives."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        ax, ay = c.axis_xy

        self.align_latch[env_ids] = 0.0
        self.load_latch[env_ids] = 0.0
        self.seal_latch[env_ids] = 0.0
        self.drum_drive[env_ids] = 0.0
        self.box_drive[env_ids] = 0.0

        # --- drum at the sampled start angle ---
        theta = torch.deg2rad(
            c.theta0_min_deg
            + (c.theta0_max_deg - c.theta0_min_deg) * torch.rand(m, device=dev))
        self.theta0[env_ids] = theta
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = ax
        st[:, 1] = ay
        st[:, 2] = c.disc_z
        st[:, 3] = torch.cos(theta / 2)
        st[:, 6] = torch.sin(theta / 2)
        st[:, 0:3] += origin
        self.drum.write_root_state_to_sim(st, env_ids)

        # --- boxes: slot assignment (torch.rand comparison — first randint after a
        # fresh manual_seed is near-constant), xy jitter, free yaw ---
        swap = torch.rand(m, device=dev) < 0.5
        slot_a = torch.tensor(c.slot_a, device=dev).expand(m, 2)
        slot_b = torch.tensor(c.slot_b, device=dev).expand(m, 2)
        for body, base_slot, other_slot in ((self.pudding, slot_a, slot_b),
                                            (self.decoy, slot_b, slot_a)):
            xy = torch.where(swap.unsqueeze(1), other_slot, base_slot).clone()
            xy = xy + (torch.rand(m, 2, device=dev) * 2 - 1) * c.box_jitter
            yaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = c.box_half + 0.002
            st[:, 3] = torch.cos(yaw / 2)
            st[:, 6] = torch.sin(yaw / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)
            if body is self.pudding:
                # Wrench-frame reference = the orientation just WRITTEN (readback right
                # after a write can be stale until the next step).
                self._box_qref[env_ids] = st[:, 3:7].clone()

    def mark_box_ref(self) -> None:
        """Capture the pudding box's current orientation as the wrench-frame reference.
        Call after any write_root_state teleport of the box (and after >= 1 step so the
        readback is fresh)."""
        self._box_qref = self.pudding.data.root_quat_w.clone()

    # ----- readings -----------------------------------------------------------------------------
    def drum_angle(self) -> torch.Tensor:
        """(N,) axle angle in rad (0 = aligned; the drum only ever rotates about z)."""
        q = self.drum.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 3], q[:, 0])

    def drum_rate(self) -> torch.Tensor:
        """(N,) signed axle rate (rad/s) — the drum's world-z angular velocity."""
        return self.drum.data.root_ang_vel_w[:, 2]

    def box_local(self) -> torch.Tensor:
        """(N, 3) pudding centre in the DRUM body frame (the bay lives in this frame,
        so a rotating drum judges its cargo identically to a parked one)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = self.pudding.data.root_pos_w - self.drum.data.root_pos_w
        return quat_apply_inverse(self.drum.data.root_quat_w, rel)

    def in_bay(self) -> torch.Tensor:
        """(N,) bool, geometric: pudding centre inside the bay containment band in the
        drum frame. No velocity clause (used while the drum is turning)."""
        c = self.cfg
        loc = self.box_local()
        return (loc[:, 0] > c.bay_x_lo) & (loc[:, 0] < c.bay_x_hi) \
            & (loc[:, 1].abs() < c.bay_y_tol) \
            & (loc[:, 2] > c.bay_z_lo) & (loc[:, 2] < c.bay_z_hi)

    def aligned(self) -> torch.Tensor:
        return self.drum_angle() <= math.radians(self.cfg.aligned_tol_deg)

    def sealed(self) -> torch.Tensor:
        return self.drum_angle() >= math.radians(
            self.cfg.drum_limit_deg - self.cfg.closed_tol_deg)

    def settled(self) -> torch.Tensor:
        """(N,) bool: pudding AND drum at rest."""
        c = self.cfg
        return (self.pudding.data.root_lin_vel_w.norm(dim=-1) < c.settle_box) \
            & (self.drum_rate().abs() < c.settle_drum)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.10 * align_latch + 0.35 * load_latch + 0.30 *
        seal_latch + 0.25 * current success. Exactly 1.0 iff success() (success forces
        all three latches to 1); ~0 for the null policy; a load lost after a full seal
        keeps the latched 0.75."""
        return (0.10 * self.align_latch + 0.35 * self.load_latch
                + 0.30 * self.seal_latch
                + 0.25 * (self.in_bay() & self.sealed() & self.settled()).float())

    def success(self) -> torch.Tensor:
        """(N,) bool: pudding contained in the bay, drum sealed against its stop,
        everything settled (current, physical state)."""
        return self.in_bay() & self.sealed() & self.settled()

    # ----- step-coupled mechanics (every substep) ------------------------------------------------
    def post_step(self) -> None:
        """Drum plant: viscous axle friction + the `drum_drive` torque buffer (world-z,
        invariant under the drum's own yaw). Box plant: the `box_drive` WORLD force,
        pre-encoded with q_ref * conj(q_now) against the pod's rotation-since-write
        wrench frame. Then latch rubric progress."""
        from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

        n = self.env.num_envs
        dev = self.env.device
        c = self.cfg
        tq = self.drum_drive - c.drum_damping * self.drum_rate()
        torque = torch.zeros(n, 1, 3, device=dev)
        torque[:, 0, 2] = tq
        self.drum.set_external_force_and_torque(torch.zeros(n, 1, 3, device=dev), torque)

        q_rel = quat_mul(self._box_qref, quat_conjugate(self.pudding.data.root_quat_w))
        f_in = quat_apply(q_rel, self.box_drive)
        self.pudding.set_external_force_and_torque(
            f_in.unsqueeze(1), torch.zeros(n, 1, 3, device=dev))

        # --- rubric latches (NaN-guarded: a diverged substep earns NO progress) ---
        in_bay = self.in_bay()
        box_slow = self.pudding.data.root_lin_vel_w.norm(dim=-1) < 0.08
        loaded = (in_bay & box_slow).float()
        alignd = self.aligned().float()
        theta = self.drum_angle().clamp(min=0.0)
        closed_rad = math.radians(c.drum_limit_deg - c.closed_tol_deg)
        prog = (theta / closed_rad).clamp(0.0, 1.0) * in_bay.float()
        prog = torch.where(in_bay & self.sealed(), torch.ones_like(prog), prog)
        for t in (loaded, alignd, prog):
            torch.nan_to_num_(t, nan=0.0, posinf=0.0, neginf=0.0)
        self.load_latch = torch.maximum(self.load_latch, loaded)
        # loading proves alignment happened (physically forced order)
        self.align_latch = torch.maximum(torch.maximum(self.align_latch, alignd),
                                         self.load_latch)
        self.seal_latch = torch.maximum(self.seal_latch, prog)

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"drum": self.drum, "pudding": self.pudding, "decoy": self.decoy}
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("theta0", "align_latch", "load_latch", "seal_latch",
                               "drum_drive", "box_drive", "_box_qref")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"drum": self.drum, "pudding": self.pudding, "decoy": self.decoy}
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A slate-gray shrouded carousel vault stands on the floor ahead of you "
            f"(axis at ({c.axis_xy[0]:.2f}, {c.axis_xy[1]:.2f})): square walls "
            f"{c.shroud_h * 100:.0f} cm tall with ONE window in the front face (the side "
            f"facing you), {2 * c.win_half_y * 100:.0f} cm wide, spanning heights "
            f"{c.bay_floor_z * 100:.0f}-{c.win_z_top * 100:.0f} cm, over a deep sill shelf "
            f"whose top is flush with the vault's internal floor. Inside spins a wooden "
            f"turntable drum with a single side-opening cargo bay (clear width "
            f"{2 * c.bay_halfw * 100:.1f} cm, height {c.bay_wall_h * 100:.0f} cm, roofed — "
            f"there is NO top access). A RED lever bar on top of the drum sweeps above the "
            f"shroud rim at height ~0.19 m and always points the same direction as the bay "
            f"opening. The drum turns on a vertical axle between two hard stops about "
            f"{c.drum_limit_deg:.0f} degrees apart: at the counterclockwise-most stop the "
            f"bay faces the window (LOAD position, lever pointing straight at you); at the "
            f"clockwise-most stop the bay faces a solid wall and the drum's flank seals the "
            f"window (SEALED position, lever pointing to your right, toward -y). The drum "
            f"starts somewhere in between ({c.theta0_min_deg:.0f}-{c.theta0_max_deg:.0f} "
            f"degrees from the load position) — misaligned, so nothing fits through the "
            f"window. On the floor between you and the vault lie two "
            f"{c.box_size * 100:.1f} cm cubes: the BROWN chocolate-pudding box and a WHITE "
            f"decoy box (positions swap randomly).\n"
            f"Goal: seal the brown pudding box inside the vault — first rotate the drum "
            f"(push the red lever) to the load stop so the bay faces the window, then "
            f"slide the pudding box through the window across the sill until it sits fully "
            f"inside the bay (fully within the drum's footprint), then rotate the drum to "
            f"the sealed stop with the box riding inside, and let everything come to rest. "
            f"The order is forced by the geometry: a misaligned or sealed drum blocks the "
            f"window, and the roof blocks dropping anything in from above. A box left "
            f"protruding from the bay, resting on the sill, on the drum roof, or anywhere "
            f"outside the bay does not count; sealing the white decoy instead of the brown "
            f"box does not count; the vault must end sealed (within "
            f"{c.closed_tol_deg:.0f} degrees of the sealed stop)."
        )

    def instruction(self) -> str:
        return (
            "Rotate the red lever to bring the carousel bay in line with the front "
            "window, slide the brown pudding box through the window fully into the bay, "
            "then rotate the drum to its opposite stop so the loaded bay is sealed inside "
            "the vault. Sealing the white decoy box, or leaving the drum short of the "
            "sealed stop, fails the task."
        )


# ----- runnable env: scene physics only (NullRobot smoke) -> "simgen.pudding_carousel" ---------
register_env("simgen", lambda: EnvCfg(scene="pudding_carousel", robot="null"))

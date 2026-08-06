"""FacingLaneScene — stock the red tin into a store-style FACING LANE: a cupboard bay
whose storage slot DOES NOT EXIST at rest, because a spring-loaded PUSHER PLATE sits
flush behind the front sill. Derived from rlbench/put_groceries_in_cupboard but the
receptacle fights back, so the seed's plan is dead on arrival.

Seed (rlbench/put_groceries_in_cupboard): identify a named grocery among distractors,
grasp it, carry it up, and set it down on a PASSIVE open cupboard shelf — the shelf
accepts anything lowered onto it from above. Here:

- The bay is a plinth-mounted lane with a raised SILL across the front and a ROOF over
  the rear two-thirds. There is NO resting surface to set the tin down on: at rest the
  orange PUSHER PLATE (riding a horizontal slide, sprung toward the front) sits 3 mm
  behind the sill — the lane has ZERO cavity. Lowering a tin onto the bay leaves it
  perched on the roof or bounced off the narrow front strip (smoke checks) — the
  seed's put-it-on-the-shelf end state scores nothing.
- The ONLY way to stow the tin is to CREATE the cavity: present the tin through the
  front window and press it HORIZONTALLY against the pusher face, driving the plate
  back against its spring (a real, load-bearing contact interaction — the spring
  pushes back the whole way), then lower the tin down behind the sill and release.
  The returning spring then CLAMPS the tin front-face against the sill's inner wall —
  the store-shelf "facing" state. Success reads back that clamp: plate compressed by
  ~the tin's depth AND the tin seated upright tight against the sill.
- A 25 mm open strip above the front zone is narrower than the 55 mm tin: nothing can
  be dropped in from above even where there is no roof (metric interlock).
- Two decoy tins (green, blue) of identical size force COLOR identification; a decoy
  in the lane fails the episode.

So a solver needs a different PLAN (press-against-a-spring insertion through a side
window, hold the compression, controlled lower-and-release into a slot that closes
itself) and different CODE STRUCTURE (force control against an opposing spring +
release timing, not lift-carry-lower onto a passive shelf). No execution order beyond
what the mechanism itself forces (the spring must be compressed BY the tin before the
tin can descend).

Assets are fully procedural (pen_holder-pattern compound spawner for the bay; plate
and tins are plain cuboids). The bay+plate mechanism is authored at its final pose and
NEVER re-posed (the jointed-pair teleport rule); per-episode randomization lives in
the free tins: slot PERMUTATION across three floor slots + xy jitter + yaw jitter —
all readback-verifiable.

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.15 * press  — latched max spring compression / tin depth (any presser earns it;
                  a bare hand press tops out here)
  0.25 * entry  — red tin CoM ever past the sill plane inside the lane window
  0.20 * seated — red tin ever seated upright on the lane floor behind the sill
  1.0 iff success() — red tin seated upright behind the sill, spring-CLAMPED against
                  the sill inner wall (plate compression AND front-gap readback), tin
                  and plate still, no decoy in the lane. Non-success cap 0.60.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- custom compound spawner -----------------------------------------------------------------
# One rigid body, several child colliders, authored with raw pxr APIs; only
# `isaaclab.sim.utils.clone` is borrowed (regex-resolve + per-env replication).

_SPAWNER_CACHE: dict[str, Any] = {}


def _add_box(stage, path: str, *, center, size, color, collide: Callable,
             material=None) -> None:
    """Author one axis-aligned box collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    if material is not None:
        from pxr import UsdShade

        UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(
            material, materialPurpose="physics")


def _make_collide(cfg: Any) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _make_material(stage, path: str, mu: float):
    """DEFINED-friction physics material (the sliding budget must be known, not
    backend-default: the return spring must beat floor friction to close the face
    gap). Restitution 0."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(mu))
    pm.CreateDynamicFrictionAttr(float(mu))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _root_xform(prim_path: str, translation, orientation):
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    return stage, xform.GetPrim()


def _spawn_bay(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the facing-lane bay: KINEMATIC compound on a plinth. Origin at the SILL
    OUTER FACE CENTRE at plinth-top height; local +u points INTO the lane. Interior:
    lane floor z in [0, roof], walls at |v| = lane_w/2, sill across u in [0, sill_t],
    roof over u in [roof_u0, bay_len], back wall closing the far end. The strip
    between the sill and the roof's leading edge is open sky — but only 25 mm across."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    mat = _make_material(stage, f"{prim_path}/bay_mat", cfg.mu)
    c = cfg
    half_w = c.lane_w / 2
    # plinth: from the ground to the lane floor, flush with the sill outer face
    _add_box(stage, f"{prim_path}/plinth",
             center=(c.bay_len / 2, 0.0, -c.plinth_h / 2),
             size=(c.bay_len, c.lane_w + 2 * c.wall_t, c.plinth_h),
             color=c.plinth_color, collide=collide, material=mat)
    # sill: the raised threshold across the lane front
    _add_box(stage, f"{prim_path}/sill",
             center=(c.sill_t / 2, 0.0, c.sill_h / 2),
             size=(c.sill_t, c.lane_w, c.sill_h),
             color=c.sill_color, collide=collide, material=mat)
    # side walls: full height, full depth
    for sgn, nm in ((1.0, "wall_l"), (-1.0, "wall_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(c.bay_len / 2, sgn * (half_w + c.wall_t / 2), c.wall_h / 2),
                 size=(c.bay_len, c.wall_t, c.wall_h),
                 color=c.bay_color, collide=collide, material=mat)
    # back wall
    _add_box(stage, f"{prim_path}/wall_back",
             center=(c.bay_len - c.wall_t / 2, 0.0, c.wall_h / 2),
             size=(c.wall_t, c.lane_w, c.wall_h),
             color=c.bay_color, collide=collide, material=mat)
    # roof: covers the rear of the lane from roof_u0 to the back; the open strip in
    # front of it (sill_t .. roof_u0 = 25 mm) is narrower than the 55 mm tin
    roof_len = c.bay_len - c.roof_u0
    _add_box(stage, f"{prim_path}/roof",
             center=(c.roof_u0 + roof_len / 2, 0.0, c.roof_z + c.roof_t / 2),
             size=(roof_len, c.lane_w + 2 * c.wall_t, c.roof_t),
             color=c.roof_color, collide=collide, material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclass (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "bay" not in _SPAWNER_CACHE:

        @configclass
        class BaySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bay)
            lane_w: float = 0.080
            bay_len: float = 0.129
            wall_t: float = 0.012
            wall_h: float = 0.162
            sill_t: float = 0.020
            sill_h: float = 0.030
            roof_u0: float = 0.045
            roof_z: float = 0.150
            roof_t: float = 0.012
            plinth_h: float = 0.12
            mu: float = 0.35
            bay_color: tuple = (0.42, 0.36, 0.28)
            plinth_color: tuple = (0.30, 0.26, 0.22)
            roof_color: tuple = (0.32, 0.27, 0.20)
            sill_color: tuple = (0.12, 0.12, 0.14)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["bay"] = BaySpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class FacingLaneSceneCfg(BaseCfg):
    """Config for `FacingLaneScene`. The interlocks are metric: at rest the pusher face
    sits 3 mm behind the sill (zero cavity — no drop, wedge or slide-in exists without
    compressing the spring THROUGH that gap), the roofed rear rejects lowering, the
    open front strip (25 mm) is narrower than the tin (55 mm), and the clamped success
    state requires the spring to be held compressed by the seated tin itself."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    settle_speed: float = tunable(0.04)  # max |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.60)  # max |ang vel| when judging (rad/s)
    upright_cos: float = tunable(0.94)  # tin local +z vs world +z (about 20 deg)
    clamp_comp_min: float = tunable(0.045)  # min plate compression at success (m)
    front_gap_max: float = tunable(0.012)  # max tin-front-face-to-sill gap at success

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    slot_jitter: float = tunable(0.025)  # per-tin spawn xy jitter (+/- m)
    yaw_jitter_deg: float = tunable(25.0)  # per-tin spawn yaw jitter (+/- deg)
    shuffle_slots: bool = tunable(True)  # random permutation of tins over slots

    # --- info: layout (single Franka base at the origin; radii 0.27-0.55 m) ---------------------
    bay_pos: tuple = info((0.48, 0.0, 0.12))  # bay origin (sill outer face, plinth top)
    slots: tuple = info(((0.30, 0.16), (0.27, 0.00), (0.30, -0.16)))  # tin floor slots

    # --- info: bay structure (bay-local frame: origin sill outer centre, +u into lane) ----------
    lane_w: float = info(0.080)  # interior width (v) — tin + 12.5 mm slack per side
    bay_len: float = info(0.129)  # sill outer face to back wall outer face
    wall_t: float = info(0.012)
    wall_h: float = info(0.162)
    sill_t: float = info(0.020)  # sill depth (u): interior starts at u = sill_t
    sill_h: float = info(0.030)  # sill top height above the lane floor
    roof_u0: float = info(0.045)  # roof leading edge (open strip = roof_u0 - sill_t)
    roof_z: float = info(0.150)  # roof underside height
    roof_t: float = info(0.012)
    plinth_h: float = info(0.12)  # lane floor height above the ground
    bay_mu: float = info(0.35)  # defined friction of every bay face

    # --- info: pusher plate + spring ------------------------------------------------------------
    plate_t: float = info(0.012)  # plate thickness (u)
    plate_w: float = info(0.068)  # plate width (v): 6 mm play per side (3x offsets)
    plate_h: float = info(0.110)  # plate height: top gap to roof = 40 mm < tin 55 mm
    plate_gap: float = info(0.002)  # float gap under the plate (gravity disabled)
    plate_home_u: float = info(0.029)  # plate CoM home: face 3 mm behind the sill
    plate_mass: float = info(0.15)
    plate_mu: float = info(0.30)
    travel: float = info(0.078)  # prismatic limit (back wall clearance 4 mm)
    spring_k: float = info(32.0)  # N/m -> ~1.8 N clamp at tin-depth compression
    spring_c: float = info(3.5)  # N*s/m, just under critical for the 0.15 kg plate
    spring_cap: float = info(6.0)  # spring force clamp (N)
    plate_color: tuple = info((0.95, 0.55, 0.10))  # orange

    # --- info: tins (target + two decoys, identical dims) ---------------------------------------
    tin_w: float = info(0.055)  # width (v) and depth (u)
    tin_h: float = info(0.075)
    tin_mass: float = info(0.25)
    tin_mu: float = info(0.50)
    red_color: tuple = info((0.85, 0.10, 0.08))  # TARGET
    green_color: tuple = info((0.10, 0.62, 0.18))  # decoy
    blue_color: tuple = info((0.12, 0.28, 0.85))  # decoy

    contact_offset: float = info(0.002)
    # rubric weights (0.15 + 0.25 + 0.20 = 0.60 = the non-success cap)
    w_press: float = info(0.15)
    w_entry: float = info(0.25)
    w_seat: float = info(0.20)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("facing_lane")
class FacingLaneScene(BaseScene):
    cfg: FacingLaneSceneCfg

    def __init__(self, cfg: FacingLaneSceneCfg | None = None) -> None:
        super().__init__(cfg or FacingLaneSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        bay_spawn = spawners["bay"](
            mass_props=sim_utils.MassPropertiesCfg(mass=12.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            lane_w=c.lane_w, bay_len=c.bay_len, wall_t=c.wall_t, wall_h=c.wall_h,
            sill_t=c.sill_t, sill_h=c.sill_h, roof_u0=c.roof_u0, roof_z=c.roof_z,
            roof_t=c.roof_t, plinth_h=c.plinth_h, mu=c.bay_mu,
            contact_offset=c.contact_offset,
        )

        def tin_cfg(name: str, color: tuple, slot: tuple) -> RigidObjectCfg:
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.CuboidCfg(
                    size=(c.tin_w, c.tin_w, c.tin_h),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5, angular_damping=0.30,
                        sleep_threshold=0.0, stabilization_threshold=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.tin_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.tin_mu, dynamic_friction=c.tin_mu,
                        restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(slot[0], slot[1], c.tin_h / 2 + 0.002)),
            )

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "bay": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bay",
                spawn=bay_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.bay_pos),
            ),
            # Gravity is DISABLED on the plate (it floats 2 mm over the lane floor and
            # never rubs it); the post_step spring about the authored home is its only
            # actuation, so home is the exact empty rest pose.
            "plate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plate",
                spawn=sim_utils.CuboidCfg(
                    size=(c.plate_t, c.plate_w, c.plate_h),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=True, max_depenetration_velocity=0.5,
                        sleep_threshold=0.0, stabilization_threshold=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.plate_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.plate_mu, dynamic_friction=c.plate_mu,
                        restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.plate_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bay_pos[0] + c.plate_home_u, c.bay_pos[1],
                         c.bay_pos[2] + c.plate_gap + c.plate_h / 2)),
            ),
            "red": tin_cfg("RedTin", c.red_color, c.slots[0]),
            "green": tin_cfg("GreenTin", c.green_color, c.slots[1]),
            "blue": tin_cfg("BlueTin", c.blue_color, c.slots[2]),
        }

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

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles, author the plate's prismatic slide, allocate latches."""
        super().bind(env)
        self.bay: RigidObject = env.iscene["bay"]
        self.plate: RigidObject = env.iscene["plate"]
        self.red: RigidObject = env.iscene["red"]
        self.green: RigidObject = env.iscene["green"]
        self.blue: RigidObject = env.iscene["blue"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._press_max = torch.zeros(n, device=dev)  # max compression / tin depth
        self._entry_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._seat_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._author_joints()

    def _author_joints(self) -> None:
        """Per env: the plate's horizontal prismatic slide on the bay — pair collision
        disabled, SYMMETRIC limits +/- travel (the weighbridge GPU lesson: a
        [0, travel] range can pin the plate under either joint-coordinate sign
        convention; the spring itself holds home). The mechanism is authored at its
        final pose and NEVER re-posed (the jointed-pair teleport rule)."""
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/plate_slide")
            j.CreateBody0Rel().SetTargets([f"{base}/Bay"])
            j.CreateBody1Rel().SetTargets([f"{base}/Plate"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(
                c.plate_home_u, 0.0, c.plate_gap + c.plate_h / 2))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-c.travel)
            j.CreateUpperLimitAttr(c.travel)
            lim = PhysxSchema.PhysxLimitAPI.Apply(j.GetPrim(), "linear")
            if hasattr(lim, "CreateContactDistanceAttr"):  # removed in Isaac Sim 5.1
                lim.CreateContactDistanceAttr(0.001)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: plate re-homed IN the fixed bay frame (follower-only write —
        the bay itself is never moved), stale external-force buffers zeroed, tins
        random-permuted over the three floor slots with xy + yaw jitter, latches
        cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- plate: follower-only re-home at the authored joint frame ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.bay_pos[0] + c.plate_home_u
        st[:, 1] = c.bay_pos[1]
        st[:, 2] = c.bay_pos[2] + c.plate_gap + c.plate_h / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.plate.write_root_state_to_sim(st, env_ids)

        # --- tins: permutation over slots + jitter + yaw ---
        slots = torch.tensor(c.slots, device=dev)  # (3, 2)
        if c.shuffle_slots:
            perm = torch.argsort(torch.rand(m, 3, device=dev), dim=1)
        else:
            perm = torch.arange(3, device=dev).expand(m, 3)
        for k, body in enumerate((self.red, self.green, self.blue)):
            xy = slots[perm[:, k]] \
                + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            yaw = (torch.rand(m, device=dev) * 2 - 1) \
                * math.radians(c.yaw_jitter_deg)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = c.tin_h / 2 + 0.002
            st[:, 3] = torch.cos(yaw / 2)
            st[:, 6] = torch.sin(yaw / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- clear latches + stale force buffers (set_external_force_and_torque does
        # not wake sleeping bodies and persists across resets) ---
        self._press_max[env_ids] = 0.0
        self._entry_ever[env_ids] = False
        self._seat_ever[env_ids] = False
        n = self.env.num_envs
        zero = torch.zeros(n, 1, 3, device=dev)
        for body in (self.plate, self.red, self.green, self.blue):
            body.set_external_force_and_torque(zero, zero)

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "plate": self.plate.data.root_state_w[env_ids].clone(),
            "red": self.red.data.root_state_w[env_ids].clone(),
            "green": self.green.data.root_state_w[env_ids].clone(),
            "blue": self.blue.data.root_state_w[env_ids].clone(),
            "press_max": self._press_max[env_ids].clone(),
            "entry_ever": self._entry_ever[env_ids].clone(),
            "seat_ever": self._seat_ever[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.plate.write_root_state_to_sim(state["plate"], env_ids)
        self.red.write_root_state_to_sim(state["red"], env_ids)
        self.green.write_root_state_to_sim(state["green"], env_ids)
        self.blue.write_root_state_to_sim(state["blue"], env_ids)
        self._press_max[env_ids] = state["press_max"]
        self._entry_ever[env_ids] = state["entry_ever"]
        self._seat_ever[env_ids] = state["seat_ever"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        strip = (c.roof_u0 - c.sill_t) * 1000
        return (
            f"A brown CUPBOARD BAY stands on a {c.plinth_h * 100:.0f} cm plinth, its "
            f"open side facing the robot. The bay is a single storage LANE "
            f"({c.lane_w * 100:.0f} cm wide) like a supermarket facing shelf: a black "
            f"SILL ({c.sill_h * 1000:.0f} mm tall, {c.sill_t * 1000:.0f} mm deep) runs "
            f"across the front at plinth height, a ROOF covers the rear of the lane "
            f"(underside {c.roof_z * 100:.0f} cm above the lane floor), and an ORANGE "
            f"PUSHER PLATE rides a horizontal slide inside, SPRING-LOADED toward the "
            f"front: at rest its face sits only 3 mm behind the sill, so THE LANE HAS "
            f"NO OPEN CAVITY — there is nowhere to set anything down. The only "
            f"opening over the front zone is a {strip:.0f} mm strip of sky between "
            f"sill and roof edge, narrower than any tin: nothing fits in from above, "
            f"and anything lowered onto the bay just rests on the roof. On the floor "
            f"in front stand THREE identical square tins "
            f"({c.tin_w * 100:.1f} x {c.tin_w * 100:.1f} x {c.tin_h * 100:.1f} cm), "
            f"distinguishable ONLY by color: RED, GREEN and BLUE; their positions are "
            f"shuffled every episode — identify by COLOR.\n"
            f"Goal: stock the RED tin into the lane, store-style. Pick up the red "
            f"tin, present it through the FRONT WINDOW just above the sill, and press "
            f"it HORIZONTALLY against the orange pusher face — the plate slides back "
            f"against its spring, opening a cavity exactly as deep as you press "
            f"(about 2 N and ~6 cm of travel). Keeping the spring compressed with "
            f"the tin itself, lower the tin down behind the sill onto the lane floor "
            f"and release it. The returning spring then clamps the tin's front face "
            f"tight against the sill's inner wall. SUCCESS requires exactly that "
            f"read-back state: the red tin seated UPRIGHT on the lane floor behind "
            f"the sill, pressed against the sill by the plate (plate compressed by "
            f"about the tin's depth, front gap under {c.front_gap_max * 1000:.0f} "
            f"mm), everything at rest, and BOTH decoy tins (green, blue) outside the "
            f"lane. Leaving the tin on the roof, on the sill, in front of the bay, "
            f"putting it in lying on its side, leaving it loose in the lane behind a "
            f"returned plate, or stocking a green/blue tin — all failure."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Stock the red tin into the spring-loaded cupboard lane: press it "
            "horizontally against the orange pusher plate to open the slot, lower it "
            "behind the black sill, and release so the spring clamps it upright "
            "against the sill. Leave the green and blue tins outside."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def _bay_local(self, body: RigidObject) -> torch.Tensor:
        """(N, 3) body CoM position in the bay frame (u into the lane, v across,
        z up from the lane floor)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = body.data.root_pos_w - self.bay.data.root_pos_w
        return quat_apply_inverse(self.bay.data.root_quat_w, rel)

    def compression(self) -> torch.Tensor:
        """(N,) plate spring compression (m): how far the plate face has been pressed
        back from its home 3 mm behind the sill. 0 = lane closed."""
        return (self._bay_local(self.plate)[:, 0] - self.cfg.plate_home_u).clamp(min=0.0)

    def _upright(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body local +z within ~20 deg of world +z."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        return (quat_apply(body.data.root_quat_w, ez)[:, 2] > self.cfg.upright_cos)

    def _entered(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body CoM past the sill plane inside the lane window (below the
        roof line — a tin perched on plate top or roof does not count)."""
        c = self.cfg
        loc = self._bay_local(body)
        return ((loc[:, 0] >= 0.030) & (loc[:, 0] <= c.bay_len - c.wall_t - 0.004)
                & (loc[:, 1].abs() <= 0.036)
                & (loc[:, 2] >= 0.004) & (loc[:, 2] <= 0.120))

    def _seated_now(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body seated UPRIGHT on the lane floor behind the sill."""
        c = self.cfg
        loc = self._bay_local(body)
        z0 = c.tin_h / 2
        return ((loc[:, 0] >= 0.023) & (loc[:, 0] <= 0.100)
                & (loc[:, 1].abs() <= 0.033)
                & (loc[:, 2] >= z0 - 0.014) & (loc[:, 2] <= z0 + 0.016)
                & self._upright(body))

    def _clamped_now(self) -> torch.Tensor:
        """(N,) bool: the spring is held compressed by the seated tin — plate pressed
        back by at least clamp_comp_min, tin front face within front_gap_max of the
        sill inner wall, plate still. This is the read-back store-facing clamp."""
        c = self.cfg
        comp = self.compression()
        front_gap = self._bay_local(self.red)[:, 0] - c.tin_w / 2 - c.sill_t
        plate_still = self.plate.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return ((comp >= c.clamp_comp_min) & (comp <= c.travel + 0.004)
                & (front_gap >= -0.006) & (front_gap <= c.front_gap_max)
                & plate_still)

    def _in_lane(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: generous lane-occupancy test (used for the decoy exclusion)."""
        c = self.cfg
        loc = self._bay_local(body)
        return ((loc[:, 0] >= 0.018) & (loc[:, 0] <= c.bay_len - 0.010)
                & (loc[:, 1].abs() <= 0.041)
                & (loc[:, 2] >= 0.0) & (loc[:, 2] <= 0.145))

    def _update_latches(self) -> None:
        c = self.cfg
        press = (self.compression() / c.tin_w).clamp(0.0, 1.0)
        press = torch.nan_to_num(press, nan=0.0, posinf=0.0, neginf=0.0)
        self._press_max = torch.maximum(self._press_max, press)
        self._entry_ever |= self._entered(self.red)
        self._seat_ever |= self._seated_now(self.red)

    # ----- step-coupled plant + bookkeeping (every substep) ----------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """The pusher spring lives here: f_u = -k*comp - c*v_u about the authored home
        along the bay axis (capped), applied every substep. Latches ride along."""
        c = self.cfg
        dev = self.env.device
        n = self.env.num_envs
        from isaaclab.utils.math import quat_apply

        u_hat = quat_apply(self.bay.data.root_quat_w,
                           torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3))
        disp = self._bay_local(self.plate)[:, 0] - c.plate_home_u
        v_u = (self.plate.data.root_lin_vel_w * u_hat).sum(dim=-1)
        f_u = (-c.spring_k * disp - c.spring_c * v_u).clamp(-c.spring_cap, c.spring_cap)
        self.plate.set_external_force_and_torque(
            (f_u.view(n, 1, 1) * u_hat.view(n, 1, 3)).contiguous(),
            torch.zeros(n, 1, 3, device=dev))
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: red tin seated upright behind the sill, spring-clamped against
        the sill inner wall, tin still, no decoy in the lane. Physical outcomes only."""
        c = self.cfg
        self._update_latches()
        still = ((self.red.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                 & (self.red.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega))
        decoy_in = self._in_lane(self.green) | self._in_lane(self.blue)
        return (self._seated_now(self.red) & self._clamped_now() & still & ~decoy_in)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*press + 0.25*entry + 0.20*seated — all latched,
        ~0 for doing nothing, capped 0.60 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_press * self._press_max + c.w_entry * self._entry_ever.float()
                + c.w_seat * self._seat_ever.float()).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through applied wrenches.
register_env("simgen", lambda: EnvCfg(scene="facing_lane", robot="null"))

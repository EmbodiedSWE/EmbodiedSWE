"""PanUnderLowShelfScene — SLIDE the frying pan under a low-clearance shelf (seed i4).

Derived from libero_90/kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf, but
STRATEGICALLY different: the seed is a free pick-and-place (grasp the pan, carry it
through the air, lower it into the shelf's open bottom compartment). Here the shelf is a
LOW alcove: its roof underside sits only ~18 mm above the pan body, and the roof fully
covers the goal region. Any lift-and-lower plan is physically blocked — the pan cannot
descend into the goal footprint from above, and there is no headroom for a gripper to
hold it from the top once inside. The only plan that works is NON-PREHENSILE: keep the
pan flat on the table and slide/push it (e.g. by the handle) through the front opening
until the whole disc is under the roof. Single skill, single stage — easy tier.

Judged in the SHELF'S body frame (its yaw is randomized): success iff the pan DISC
(handle excluded — it may stick out the front, as a real pan stored under a shelf would)
is fully inside the alcove footprint, the pan is flat on the table (tilt + base-height
gates), it is below the roof (a pan parked ON TOP of the roof is gated out by z), and it
is settled. Score is graded and latched: 0.2 * best approach toward the alcove +
0.6 * best insertion fraction of the disc past the front plane (only while under the
roof, roughly flat, laterally inside), 0.9 once fully in + flat, 1.0 iff success. Doing
nothing scores ~0 (approach is measured against the episode's own spawn distance).

Assets are fully procedural, one rigid body each, authored by custom compound spawners
(the pen_holder pattern — child colliders of one body never self-collide):
  - shelf: KINEMATIC alcove — roof slab + two side walls + back wall, open at the front
    (local -y). Gap height is the honesty knob: pan_h + ~18 mm.
  - pan: flat cylinder disc + a box handle along local +x, dark gray. Rest height =
    pan_h/2. Sleep/stabilization thresholds are zeroed at spawn so applied pushes always
    act (sleeping bodies ignore external wrenches).
  - bowl: a plain white cylinder distractor (the seed's white_bowl) — it also fits under
    the shelf, and putting IT under scores nothing (object identity matters).
Contact offsets are explicit and small (2 mm): the default ~2 cm offset would eat the
18 mm roof clearance and jam the slide with phantom contact.

Per-episode randomization: shelf xy jitter + yaw, pan spawn xy jitter + free yaw, bowl
xy jitter. Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- custom compound spawners ---------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, size, center, color, contact_offset: float) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _spawn_low_shelf(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC low alcove at `prim_path`: root Xform origin at the centre of
    the interior floor (z=0 plane = the table surface), roof slab whose underside is at
    `h_gap`, two side walls and a back wall; the front (local -y) is open."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(5.0)

    w, d, g, rt, wt = cfg.alcove_w, cfg.alcove_d, cfg.h_gap, cfg.roof_t, cfg.wall_t
    co, color = cfg.contact_offset, cfg.color
    # roof: covers the interior + both walls + the back wall footprint
    _box(stage, f"{prim_path}/roof", (w + 2 * wt, d + wt, rt),
         (0.0, wt / 2, g + rt / 2), color, co)
    # side walls
    _box(stage, f"{prim_path}/wall_l", (wt, d + wt, g),
         (-(w / 2 + wt / 2), wt / 2, g / 2), color, co)
    _box(stage, f"{prim_path}/wall_r", (wt, d + wt, g),
         (w / 2 + wt / 2, wt / 2, g / 2), color, co)
    # back wall
    _box(stage, f"{prim_path}/back", (w + 2 * wt, wt, g),
         (0.0, d / 2 + wt / 2, g / 2), color, co)
    return root


def _spawn_pan(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the frying pan at `prim_path`: one rigid body = flat cylinder disc (the pan
    body, axis +z) + a box handle along local +x. Depenetration capped + light damping
    (the pen_holder precedent) so pushes stay tame; sleep/stabilization thresholds are
    zeroed (a sleeping body silently ignores applied external forces)."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    disc = UsdGeom.Cylinder.Define(stage, f"{prim_path}/disc")
    disc.CreateRadiusAttr(cfg.pan_r)
    disc.CreateHeightAttr(cfg.pan_h)
    disc.CreateExtentAttr([Gf.Vec3f(-cfg.pan_r, -cfg.pan_r, -cfg.pan_h / 2),
                           Gf.Vec3f(cfg.pan_r, cfg.pan_r, cfg.pan_h / 2)])
    disc.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    _collide(disc.GetPrim(), cfg.contact_offset)

    _box(stage, f"{prim_path}/handle",
         (cfg.handle_l, cfg.handle_w, cfg.handle_t),
         (cfg.pan_r + cfg.handle_l / 2 - 0.006, 0.0, 0.003),
         cfg.handle_color, cfg.contact_offset)
    return root


def _shelf_spawner_cfg(*, alcove_w: float, alcove_d: float, h_gap: float, roof_t: float,
                       wall_t: float, color: tuple, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "shelf" not in _SPAWNER_CACHE:

        @configclass
        class LowShelfSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_low_shelf)
            alcove_w: float = 0.22
            alcove_d: float = 0.20
            h_gap: float = 0.048
            roof_t: float = 0.020
            wall_t: float = 0.020
            color: tuple = (0.45, 0.32, 0.18)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["shelf"] = LowShelfSpawnerCfg

    return _SPAWNER_CACHE["shelf"](
        mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        alcove_w=alcove_w, alcove_d=alcove_d, h_gap=h_gap, roof_t=roof_t,
        wall_t=wall_t, color=color, contact_offset=contact_offset,
    )


def _pan_spawner_cfg(*, pan_r: float, pan_h: float, handle_l: float, handle_w: float,
                     handle_t: float, mass: float, color: tuple, handle_color: tuple,
                     contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "pan" not in _SPAWNER_CACHE:

        @configclass
        class PanSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pan)
            pan_r: float = 0.080
            pan_h: float = 0.030
            handle_l: float = 0.110
            handle_w: float = 0.024
            handle_t: float = 0.012
            color: tuple = (0.16, 0.16, 0.18)
            handle_color: tuple = (0.05, 0.05, 0.05)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["pan"] = PanSpawnerCfg

    return _SPAWNER_CACHE["pan"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        pan_r=pan_r, pan_h=pan_h, handle_l=handle_l, handle_w=handle_w,
        handle_t=handle_t, color=color, handle_color=handle_color,
        contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class PanUnderLowShelfSceneCfg(BaseCfg):
    """Config for `PanUnderLowShelfScene`. `h_gap` is the strategic honesty knob: the
    roof underside must clear the flat pan body but leave no room to lower or hold the
    pan from above (enforced in `__post_init__`)."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    h_gap: float = tunable(0.048)  # roof underside above the table (m); pan_h + ~18 mm
    edge_tol: float = tunable(0.008)  # slack on "disc fully inside the footprint" (m)
    flat_tilt_deg: float = tunable(10.0)  # pan axis within this of world-up = "flat"
    z_tol: float = tunable(0.012)  # pan base within this of the table when judged (m)
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging success (m/s)
    gate_tilt_deg: float = tunable(30.0)  # rough-flat gate for insertion-progress credit

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    shelf_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the shelf at reset (m)
    shelf_yaw_deg: float = tunable(10.0)  # uniform +/- shelf yaw at reset (deg)
    pan_jitter: float = tunable(0.05)  # uniform +/- xy jitter of the pan spawn (m)
    pan_yaw_deg: float = tunable(180.0)  # uniform +/- pan yaw at reset (handle direction)
    bowl_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the bowl spawn (m)

    # --- tunable: placement ------------------------------------------------------------------
    shelf_pos: tuple = tunable((0.0, 0.20))  # alcove interior-floor centre, nominal
    pan_spawn: tuple = tunable((0.0, -0.22))  # pan spawn centre, in front of the alcove
    bowl_pos: tuple = tunable((0.32, -0.02))  # distractor bowl, off the slide corridor

    # --- info: structure ---------------------------------------------------------------------
    pan_r: float = info(0.080)  # disc radius
    pan_h: float = info(0.030)  # disc height; roof clearance = h_gap - pan_h ~= 18 mm
    handle_l: float = info(0.110)
    handle_w: float = info(0.024)
    handle_t: float = info(0.012)
    pan_mass: float = info(0.40)
    pan_color: tuple = info((0.16, 0.16, 0.18))
    handle_color: tuple = info((0.05, 0.05, 0.05))
    alcove_w: float = info(0.22)  # interior width; disc diameter 0.16 -> 3 cm side play
    alcove_d: float = info(0.20)  # interior depth; disc fits with a 4 cm centre band
    roof_t: float = info(0.020)
    wall_t: float = info(0.020)
    shelf_color: tuple = info((0.45, 0.32, 0.18))
    bowl_r: float = info(0.055)
    bowl_h: float = info(0.035)  # also fits under the roof — identity, not fit, rejects it
    bowl_mass: float = info(0.15)
    bowl_color: tuple = info((0.92, 0.92, 0.90))
    # Explicit small offsets: the ~2 cm default would eat the 18 mm roof clearance.
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    full_insert: float = field(default=None, init=False)  # disc diameter = full travel past front
    success_frac: float = field(default=None, init=False)  # predicted success onset fraction

    def __post_init__(self) -> None:
        clear = self.h_gap - self.pan_h
        assert 0.008 <= clear <= 0.035, (
            f"roof clearance {clear * 1000:.0f} mm breaks the task premise: it must clear the "
            f"sliding pan (>8 mm) but stay far too low for any lift-and-lower or top grasp (<35 mm)")
        self.full_insert = 2 * self.pan_r
        self.success_frac = 1.0 - self.edge_tol / self.full_insert


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("pan_lowshelf_slide")
class PanUnderLowShelfScene(BaseScene):
    cfg: PanUnderLowShelfSceneCfg

    def __init__(self, cfg: PanUnderLowShelfSceneCfg | None = None) -> None:
        super().__init__(cfg or PanUnderLowShelfSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
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
            "shelf": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shelf",
                spawn=_shelf_spawner_cfg(
                    alcove_w=c.alcove_w, alcove_d=c.alcove_d, h_gap=c.h_gap,
                    roof_t=c.roof_t, wall_t=c.wall_t, color=c.shelf_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.shelf_pos[0], c.shelf_pos[1], 0.0)),
            ),
            "pan": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pan",
                spawn=_pan_spawner_cfg(
                    pan_r=c.pan_r, pan_h=c.pan_h, handle_l=c.handle_l, handle_w=c.handle_w,
                    handle_t=c.handle_t, mass=c.pan_mass, color=c.pan_color,
                    handle_color=c.handle_color, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pan_spawn[0], c.pan_spawn[1], c.pan_h / 2 + 0.002)),
            ),
            "bowl": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl",
                spawn=sim_utils.CylinderCfg(
                    radius=c.bowl_r, height=c.bowl_h,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.05),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bowl_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.bowl_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bowl_pos[0], c.bowl_pos[1], c.bowl_h / 2 + 0.002)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.shelf: RigidObject = env.iscene["shelf"]
        self.pan: RigidObject = env.iscene["pan"]
        self.bowl: RigidObject = env.iscene["bowl"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.d0 = torch.full((n,), 1.0, device=dev)  # spawn->alcove distance, per episode
        self.approach_latch = torch.zeros(n, device=dev)
        self.frac_latch = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: shelf with xy jitter + yaw, pan flat in front with xy jitter +
        free yaw (handle direction random), bowl off to the side; latches zeroed and the
        approach baseline `d0` captured from the sampled poses."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        shelf_xy = torch.tensor(c.shelf_pos, device=dev).expand(m, 2).clone()
        shelf_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.shelf_jitter
        shelf_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.shelf_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = shelf_xy
        st[:, 3] = torch.cos(shelf_yaw / 2)
        st[:, 6] = torch.sin(shelf_yaw / 2)
        st[:, 0:3] += origin
        self.shelf.write_root_state_to_sim(st, env_ids)

        pan_xy = torch.tensor(c.pan_spawn, device=dev).expand(m, 2).clone()
        pan_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.pan_jitter
        pan_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.pan_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = pan_xy
        st[:, 2] = c.pan_h / 2 + 0.002
        st[:, 3] = torch.cos(pan_yaw / 2)
        st[:, 6] = torch.sin(pan_yaw / 2)
        st[:, 0:3] += origin
        self.pan.write_root_state_to_sim(st, env_ids)

        bowl_xy = torch.tensor(c.bowl_pos, device=dev).expand(m, 2).clone()
        bowl_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.bowl_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = bowl_xy
        st[:, 2] = c.bowl_h / 2 + 0.002
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.bowl.write_root_state_to_sim(st, env_ids)

        self.d0[env_ids] = (pan_xy - shelf_xy).norm(dim=-1).clamp(min=0.05)
        self.approach_latch[env_ids] = 0.0
        self.frac_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "shelf": self.shelf.data.root_state_w[env_ids].clone(),
            "pan": self.pan.data.root_state_w[env_ids].clone(),
            "bowl": self.bowl.data.root_state_w[env_ids].clone(),
            "d0": self.d0[env_ids].clone(),
            "approach_latch": self.approach_latch[env_ids].clone(),
            "frac_latch": self.frac_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.shelf.write_root_state_to_sim(state["shelf"], env_ids)
        self.pan.write_root_state_to_sim(state["pan"], env_ids)
        self.bowl.write_root_state_to_sim(state["bowl"], env_ids)
        self.d0[env_ids] = state["d0"]
        self.approach_latch[env_ids] = state["approach_latch"]
        self.frac_latch[env_ids] = state["frac_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A low wooden shelf alcove stands on the table: two side walls, a back wall and a "
            f"roof, open only at the front. The roof underside is just {c.h_gap * 1000:.0f} mm "
            f"above the table — barely above the {c.pan_h * 1000:.0f} mm tall frying pan body, "
            f"far too low to lower the pan in from above or to reach inside from the top. A dark "
            f"frying pan (disc {2 * c.pan_r * 1000:.0f} mm across with a straight handle) lies "
            f"flat in front of the opening, and a white bowl sits off to the side.\n"
            f"Goal: get the frying pan stored under the shelf — keep it flat on the table and "
            f"slide/push it through the front opening until its whole disc is inside the alcove "
            f"footprint, then leave it there at rest. The handle may stick out of the front. "
            f"Putting the bowl under instead counts for nothing, and the pan ending up ON TOP of "
            f"the shelf counts for nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the dark frying pan flat along the table through the front opening of the "
            "low shelf until its whole disc is inside the alcove, and leave it there at rest. "
            "Do not lift it — the pan on top of the shelf, or the white bowl inside, counts "
            "for nothing."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _pan_local(self) -> torch.Tensor:
        """(N,3) pan centre in the SHELF body frame (origin = interior-floor centre)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = self.pan.data.root_pos_w - self.shelf.data.root_pos_w
        return quat_apply_inverse(self.shelf.data.root_quat_w, rel)

    def _pan_up_z(self) -> torch.Tensor:
        """(N,) world-z of the pan's +z axis (1 = perfectly flat)."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(self.pan.data.root_quat_w, ez)[:, 2].clamp(-1.0, 1.0)

    def flat(self) -> torch.Tensor:
        """(N,) bool: pan within `flat_tilt_deg` of flat AND its base at table height."""
        c = self.cfg
        tilt_ok = self._pan_up_z() >= math.cos(math.radians(c.flat_tilt_deg))
        base_z = (self.pan.data.root_pos_w - self.env_origins)[:, 2] - c.pan_h / 2
        return tilt_ok & (base_z.abs() < c.z_tol)

    def insertion_frac(self) -> torch.Tensor:
        """(N,) in [0,1]: how far the disc's leading edge has travelled past the alcove's
        front plane — 0 at first contact with the plane, 1 when the whole disc is inside.
        Gated on being UNDER the roof (z), laterally inside, and roughly flat, so a pan on
        the roof or wedged on edge earns nothing."""
        c = self.cfg
        loc = self._pan_local()
        under = (loc[:, 2] < c.h_gap) & (loc[:, 2] > -0.02)
        lateral = loc[:, 0].abs() < c.alcove_w / 2
        rough_flat = self._pan_up_z() >= math.cos(math.radians(c.gate_tilt_deg))
        raw = ((loc[:, 1] + c.alcove_d / 2 + c.pan_r) / c.full_insert).clamp(0.0, 1.0)
        return raw * (under & lateral & rough_flat).float()

    def fully_in(self) -> torch.Tensor:
        """(N,) bool, geometric: the disc footprint inside the alcove interior (with
        `edge_tol` slack), under the roof."""
        c = self.cfg
        loc = self._pan_local()
        under = (loc[:, 2] < c.h_gap) & (loc[:, 2] > -0.02)
        in_depth = (loc[:, 1] >= -c.alcove_d / 2 + c.pan_r - c.edge_tol) & \
                   (loc[:, 1] <= c.alcove_d / 2 - c.pan_r + c.edge_tol)
        in_width = loc[:, 0].abs() <= c.alcove_w / 2 - c.pan_r + c.edge_tol
        return under & in_depth & in_width

    def settled(self) -> torch.Tensor:
        return self.pan.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch the best approach + best insertion each physics substep, so transient
        progress (a pan pushed in and then knocked back out) keeps its credit."""
        d = (self.pan.data.root_pos_w - self.shelf.data.root_pos_w)[:, :2].norm(dim=-1)
        approach = (1.0 - d / self.d0).clamp(0.0, 1.0)
        self.approach_latch = torch.maximum(self.approach_latch, approach)
        self.frac_latch = torch.maximum(self.frac_latch, self.insertion_frac())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: disc fully inside the alcove, pan flat on the table, settled."""
        return self.fully_in() & self.flat() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.2 * latched approach + 0.6 * latched insertion (max 0.8),
        0.9 once fully in + flat, 1.0 iff success. Doing nothing scores ~0."""
        base = (0.2 * self.approach_latch + 0.6 * self.frac_latch).clamp(0.0, 0.8)
        s = torch.where(self.fully_in() & self.flat(), torch.maximum(base, base.new_tensor(0.9)), base)
        return torch.where(self.success(), s.new_tensor(1.0), s)


register_env("simgen", lambda: EnvCfg(scene="pan_lowshelf_slide", robot="null"))

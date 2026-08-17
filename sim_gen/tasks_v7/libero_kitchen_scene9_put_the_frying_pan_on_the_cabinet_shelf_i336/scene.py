"""ComboShutterPantryScene — register two sliding shutter plates with a doorway, then
slide the pan through the opened gate into a roofed pantry alcove (sim_gen task
`libero_kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf_i336`).

Derived from libero_90/libero_kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf, but
STRATEGICALLY different: the seed is a single pick-and-place — grasp the frying pan, set
it down on the shelf's OPEN top region; its checker is pure position containment, so one
free-air transport ends the task. Here the goal volume is a pantry ALCOVE that is sealed
from above (roof), from the sides and the back (walls), and from the front by a wall
whose only DOORWAY is barred by TWO independent free-sliding SHUTTER PLATES, each with a
doorway-sized bottom-open notch, each spawned at a random misalignment large enough that
EACH PLATE ALONE blocks the pan. No placement — however precise — reaches the goal:
the solver must (1) slide shutter plate A along its guide slot until its notch registers
with the doorway, (2) do the same for shutter plate B (a COMBINATION: two independent
alignment subgoals that must HOLD SIMULTANEOUSLY), and then (3) push the pan FLAT along
the counter THROUGH the triple-aligned opening until it rests fully inside the alcove.
The doorway is deliberately shorter than the pan's diameter, so the pan cannot be rolled
on its rim through a partial gap, and the notch bridges pass over the sliding pan. A
smaller saucer (decoy) must stay outside. The plan (align a two-plate combination
mechanism, then thread cargo through the transient registered aperture) and the code
structure (per-plate alignment latches, a both-aligned latch, a transit-while-registered
pathway latch, judgment in the cabinet body frame) share nothing with the seed's
grasp-and-set-down, nor with sibling tasks (i11: open/load/re-close a sliding drawer —
one prismatic DoF, cavity carried BY the mover; here the movers are pure gatekeepers,
the goal is static architecture, there is no re-close; i9: extract from a cubby and
place onto a burner; i260: deploy a drop-leaf and brace it).

Judged in the CABINET's body frame (kinematic, xy + free yaw randomized: the push-through
direction must be read from the scene). success() iff:
  - the pan rests INSIDE the alcove (fully past the wall's inner face), FLAT (upright
    within `upright_deg`), settled;
  - the transit latch is set: the pan physically crossed the doorway slab WHILE both
    plates were registered — a pan written directly into the alcove earns nothing;
  - the saucer decoy is NOT in the alcove.
score() is latched every physics substep: 0.15 per plate ever registered + 0.20 both
registered simultaneously + 0.20 transit-while-registered, capped at 0.70; exactly 1.0
iff success(). Doing nothing scores ~0; the seed's strategy (pan set on top of the
fixture) scores ~0.

Assets are fully procedural (no external files):
  - cabinet (KINEMATIC compound): a counter plinth (0.80 x 1.00 x 0.24 m) carrying a
    front wall (doorway 0.17 wide x 0.055 tall) with a roofed alcove behind it
    (interior 0.25 deep x 0.26 wide x 0.10 tall), and in front of the wall a shutter
    ledge with two ridge-guided slots, both interrupted across the doorway span so the
    pan's floor path is bare, with end stops. Local +x = push-through direction.
  - two shutter plates (DYNAMIC compounds, 0.25 kg each): staple-shaped — two blocks
    joined by a top bridge over a doorway-sized bottom-open notch — each carrying one
    raised TAB post (13 x 32 mm cross-section — inside a Franka's 80 mm jaw span). The
    FRONT plate's tab sits at local +y (green), the REAR plate's at local -y (orange).
    They slide along cabinet y on the ledge, guided by ridges and the wall face.
  - pan: dark disc (r 65 mm, h 35 mm) with a short handle stub. decoy: smaller pale
    saucer disc (r 45 mm).
Moderate ledge/plate friction (the slide is real work but modest force moves it); the
pan is grippier so pushing it is controlled. Explicit small contact offsets.

Per-episode randomization (verified by readback in smoke): cabinet xy + free yaw, each
plate's offset magnitude in [off_lo, off_hi] with an independent random SIGN, pan spawn
xy + free yaw on the porch, decoy on a random +/-y side band — so a memorized fixed
trajectory fails. Heavy imports (isaaclab, pxr) are deferred so importing this module —
and registering the scene — stays app-free.
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


def _friction_material(stage, path: str, static: float, dynamic: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _collide(prim, contact_offset: float, material=None) -> None:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, size, center, color, contact_offset: float,
         material=None) -> None:
    """One collidable box child prim (translate -> scale, authored once — idempotent
    per prim, the duplicate-xformOp trap)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset, material)


def _cyl(stage, path: str, radius: float, height: float, center, color,
         contact_offset: float, material=None) -> None:
    """One collidable z-axis cylinder child prim."""
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr("Z")
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    r, h2 = float(radius), float(height) / 2
    cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h2), Gf.Vec3f(r, r, h2)])
    UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(cyl.GetPrim(), contact_offset, material)


def _spawn_cabinet(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC cabinet at `prim_path`. Origin = plinth-top centre
    (z = 0 at the plinth top); local +x = push-through direction (porch -> doorway ->
    alcove). The shutter ledge, both guide ridges and the front wall's doorway sill are
    all INTERRUPTED across the doorway span, so the pan's path is bare counter."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(60.0)

    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset

    # --- plinth (counter) under everything ---
    _box(stage, f"{prim_path}/plinth", cfg.plinth_size,
         (0.0, 0.0, -cfg.plinth_size[2] / 2), cfg.plinth_color, co, material=mat)

    wx0, wx1 = cfg.wall_x0, cfg.wall_x0 + cfg.wall_t
    dh = cfg.door_half

    # --- front wall: two piers + header over the doorway ---
    for tag, sgn in (("yp", 1.0), ("yn", -1.0)):
        _box(stage, f"{prim_path}/pier_{tag}",
             (cfg.wall_t, cfg.pier_y1 - dh, cfg.wall_h),
             ((wx0 + wx1) / 2, sgn * (dh + cfg.pier_y1) / 2, cfg.wall_h / 2),
             cfg.wall_color, co, material=mat)
    _box(stage, f"{prim_path}/header",
         (cfg.wall_t, 2 * dh, cfg.wall_h - cfg.door_h),
         ((wx0 + wx1) / 2, 0.0, (cfg.wall_h + cfg.door_h) / 2),
         cfg.wall_color, co, material=mat)

    # --- alcove behind the wall: side walls, back wall, roof ---
    ax1 = cfg.alc_x1
    for tag, sgn in (("yp", 1.0), ("yn", -1.0)):
        _box(stage, f"{prim_path}/alc_wall_{tag}",
             (ax1 - wx1, cfg.alc_wall_t, cfg.alc_h),
             ((wx1 + ax1) / 2, sgn * (cfg.alc_half_w + cfg.alc_wall_t / 2), cfg.alc_h / 2),
             cfg.body_color, co, material=mat)
    _box(stage, f"{prim_path}/alc_back",
         (cfg.back_t, 2 * (cfg.alc_half_w + cfg.alc_wall_t), cfg.alc_h),
         (ax1 + cfg.back_t / 2, 0.0, cfg.alc_h / 2), cfg.body_color, co, material=mat)
    _box(stage, f"{prim_path}/alc_roof",
         (ax1 + cfg.back_t - wx0, 2 * (cfg.alc_half_w + cfg.alc_wall_t), cfg.roof_t),
         ((wx0 + ax1 + cfg.back_t) / 2, 0.0, cfg.alc_h + cfg.roof_t / 2),
         cfg.roof_color, co, material=mat)

    # --- shutter ledge + guide ridges (both interrupted across the doorway span) ---
    gy0, gy1 = cfg.guide_y0, cfg.guide_y1
    for tag, sgn in (("yp", 1.0), ("yn", -1.0)):
        yc = sgn * (gy0 + gy1) / 2
        ln = gy1 - gy0
        _box(stage, f"{prim_path}/ledge_{tag}",
             (cfg.ledge_x1 - cfg.ledge_x0, ln, cfg.ledge_h),
             ((cfg.ledge_x0 + cfg.ledge_x1) / 2, yc, cfg.ledge_h / 2),
             cfg.guide_color, co, material=mat)
        _box(stage, f"{prim_path}/ridge_front_{tag}",
             (cfg.ridgeF_x1 - cfg.ridgeF_x0, ln, cfg.ridge_h),
             ((cfg.ridgeF_x0 + cfg.ridgeF_x1) / 2, yc, cfg.ridge_h / 2),
             cfg.guide_color, co, material=mat)
        _box(stage, f"{prim_path}/ridge_mid_{tag}",
             (cfg.ridgeM_x1 - cfg.ridgeM_x0, ln, cfg.ridge_h),
             ((cfg.ridgeM_x0 + cfg.ridgeM_x1) / 2, yc, cfg.ridge_h / 2),
             cfg.guide_color, co, material=mat)
        # end stop
        _box(stage, f"{prim_path}/stop_{tag}",
             (wx1 - cfg.ridgeF_x0, cfg.stop_y1 - cfg.stop_y0, cfg.stop_h),
             ((cfg.ridgeF_x0 + wx1) / 2, sgn * (cfg.stop_y0 + cfg.stop_y1) / 2,
              cfg.stop_h / 2),
             cfg.wall_color, co, material=mat)
    return root


def _spawn_plate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one DYNAMIC shutter plate at `prim_path`. Origin = plate bottom centre.
    Staple shape: two solid blocks joined by a top bridge over a bottom-open notch,
    plus one raised tab post (the graspable handle) above one block."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.20)
    px.CreateAngularDampingAttr(0.50)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(4)  # kills GPU cylinder/box phantom creep
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)

    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    t = cfg.plate_t
    nh = cfg.notch_half
    bl = cfg.block_len
    bh = cfg.block_h

    for tag, sgn in (("yp", 1.0), ("yn", -1.0)):
        _box(stage, f"{prim_path}/block_{tag}", (t, bl, bh),
             (0.0, sgn * (nh + bl / 2), bh / 2), cfg.color, co, material=mat)
    _box(stage, f"{prim_path}/bridge", (t, 2 * nh, cfg.bridge_t),
         (0.0, 0.0, bh - cfg.bridge_t / 2), cfg.color, co, material=mat)
    _box(stage, f"{prim_path}/tab", (t, cfg.tab_w, cfg.tab_h),
         (0.0, cfg.tab_y, bh + cfg.tab_h / 2), cfg.tab_color, co, material=mat)
    return root


def _spawn_pan(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC pan at `prim_path`. Origin = disc bottom centre. A flat dark
    disc with a short handle stub protruding from the rim along local +x."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.10)
    px.CreateAngularDampingAttr(0.30)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(4)  # kills GPU cylinder phantom creep (lateral veer)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)

    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    _cyl(stage, f"{prim_path}/disc", cfg.pan_r, cfg.pan_h,
         (0.0, 0.0, cfg.pan_h / 2), cfg.color, co, material=mat)
    _box(stage, f"{prim_path}/handle",
         (cfg.handle_len, cfg.handle_w, cfg.handle_t),
         (cfg.pan_r + cfg.handle_len / 2, 0.0, cfg.pan_h / 2), cfg.handle_color,
         co, material=mat)
    return root


def _cabinet_spawner_cfg(c: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "cabinet" not in _SPAWNER_CACHE:

        @configclass
        class PantryCabinetSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cabinet)
            plinth_size: tuple = (0.80, 1.00, 0.24)
            wall_x0: float = 0.060
            wall_t: float = 0.030
            wall_h: float = 0.130
            door_half: float = 0.085
            door_h: float = 0.055
            pier_y1: float = 0.44
            alc_x1: float = 0.340
            alc_half_w: float = 0.130
            alc_wall_t: float = 0.030
            alc_h: float = 0.100
            back_t: float = 0.030
            roof_t: float = 0.030
            ledge_x0: float = 0.008
            ledge_x1: float = 0.058
            ledge_h: float = 0.012
            ridgeF_x0: float = -0.005
            ridgeF_x1: float = 0.008
            ridgeM_x0: float = 0.027
            ridgeM_x1: float = 0.042
            ridge_h: float = 0.050
            guide_y0: float = 0.105
            guide_y1: float = 0.470
            stop_y0: float = 0.455
            stop_y1: float = 0.490
            stop_h: float = 0.100
            mu_static: float = 0.25
            mu_dynamic: float = 0.22
            plinth_color: tuple = (0.45, 0.45, 0.48)
            wall_color: tuple = (0.30, 0.32, 0.36)
            body_color: tuple = (0.26, 0.28, 0.33)
            roof_color: tuple = (0.22, 0.24, 0.30)
            guide_color: tuple = (0.38, 0.38, 0.42)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["cabinet"] = PantryCabinetSpawnerCfg

    return _SPAWNER_CACHE["cabinet"](
        mass_props=sim_utils.MassPropertiesCfg(mass=60.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        plinth_size=c.plinth_size, wall_x0=c.wall_x0, wall_t=c.wall_t, wall_h=c.wall_h,
        door_half=c.door_half, door_h=c.door_h, pier_y1=c.pier_y1, alc_x1=c.alc_x1,
        alc_half_w=c.alc_half_w, alc_wall_t=c.alc_wall_t, alc_h=c.alc_h,
        back_t=c.back_t, roof_t=c.roof_t, ledge_x0=c.ledge_x0, ledge_x1=c.ledge_x1,
        ledge_h=c.ledge_h, ridgeF_x0=c.ridgeF_x0, ridgeF_x1=c.ridgeF_x1,
        ridgeM_x0=c.ridgeM_x0, ridgeM_x1=c.ridgeM_x1, ridge_h=c.ridge_h,
        guide_y0=c.guide_y0, guide_y1=c.guide_y1, stop_y0=c.stop_y0, stop_y1=c.stop_y1,
        stop_h=c.stop_h, mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
        plinth_color=c.plinth_color, wall_color=c.wall_color, body_color=c.body_color,
        roof_color=c.roof_color, guide_color=c.guide_color,
        contact_offset=c.contact_offset,
    )


def _plate_spawner_cfg(c: Any, tab_y: float, color: tuple) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "plate" not in _SPAWNER_CACHE:

        @configclass
        class ShutterPlateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_plate)
            plate_t: float = 0.013
            notch_half: float = 0.085
            block_len: float = 0.200
            block_h: float = 0.083
            bridge_t: float = 0.021
            tab_w: float = 0.032
            tab_h: float = 0.070
            tab_y: float = 0.240
            mass: float = 0.25
            mu_static: float = 0.25
            mu_dynamic: float = 0.22
            color: tuple = (0.30, 0.55, 0.25)
            tab_color: tuple = (0.08, 0.08, 0.08)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["plate"] = ShutterPlateSpawnerCfg

    return _SPAWNER_CACHE["plate"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.plate_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        plate_t=c.plate_t, notch_half=c.door_half, block_len=c.block_len,
        block_h=c.block_h, bridge_t=c.bridge_t, tab_w=c.tab_w, tab_h=c.tab_h,
        tab_y=tab_y, mass=c.plate_mass, mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
        color=color, tab_color=c.tab_color, contact_offset=c.contact_offset,
    )


def _pan_spawner_cfg(c: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "pan" not in _SPAWNER_CACHE:

        @configclass
        class PanSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pan)
            pan_r: float = 0.065
            pan_h: float = 0.035
            handle_len: float = 0.050
            handle_w: float = 0.022
            handle_t: float = 0.012
            mass: float = 0.40
            mu_static: float = 0.45
            mu_dynamic: float = 0.40
            color: tuple = (0.10, 0.10, 0.12)
            handle_color: tuple = (0.15, 0.12, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["pan"] = PanSpawnerCfg

    return _SPAWNER_CACHE["pan"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.pan_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        pan_r=c.pan_r, pan_h=c.pan_h, handle_len=c.handle_len, handle_w=c.handle_w,
        handle_t=c.handle_t, mass=c.pan_mass, mu_static=c.pan_mu_static,
        mu_dynamic=c.pan_mu_dynamic, color=c.pan_color, handle_color=c.handle_color,
        contact_offset=c.pan_contact_offset,
    )


def _decoy_spawner_cfg(c: Any) -> Any:
    import isaaclab.sim as sim_utils

    return sim_utils.CylinderCfg(
        radius=c.decoy_r,
        height=c.decoy_h,
        axis="Z",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            linear_damping=0.10, angular_damping=0.30, max_depenetration_velocity=0.5,
            solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            sleep_threshold=0.0, stabilization_threshold=0.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=c.decoy_mass),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=c.pan_contact_offset, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=c.pan_mu_static, dynamic_friction=c.pan_mu_dynamic,
            restitution=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.decoy_color),
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ComboShutterPantrySceneCfg(BaseCfg):
    """Config for `ComboShutterPantryScene`. Honesty knobs asserted in `__post_init__`:
    a single misaligned plate blocks the pan, the registered gate passes it with real
    clearance, the doorway height forces the pan FLAT (an on-edge pan is taller than
    the header underside), the notch bridge passes over the sliding pan, the alcove is
    roofed against top entry, the plates are supported and guided over their whole
    travel, and tab + handle fit a Franka's 80 mm jaw."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    align_tol: float = tunable(0.015)  # |plate offset| below this = registered (m)
    pass_tol: float = tunable(0.020)  # transit latch: both offsets within this (m)
    entry_x: float = tunable(0.156)  # pan centre past this = fully inside (cab x, m)
    upright_deg: float = tunable(20.0)  # pan must rest flat within this tilt (deg)
    settle_lin: float = tunable(0.06)  # max |lin vel| (pan/decoy/plates) when judging

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    cab_jitter: float = tunable(0.05)  # uniform +/- xy jitter of the cabinet at reset (m)
    cab_yaw_deg: float = tunable(180.0)  # uniform +/- cabinet yaw (free — read the layout)
    off_lo: float = tunable(0.055)  # plate |offset| spawn band (m): each alone blocks
    off_hi: float = tunable(0.125)
    pan_x_lo: float = tunable(-0.26)  # pan spawn band, cabinet frame
    pan_x_hi: float = tunable(-0.16)
    pan_y_max: float = tunable(0.08)  # pan spawn |y| bound
    decoy_x_lo: float = tunable(-0.26)  # decoy spawn band (side sampled per episode)
    decoy_x_hi: float = tunable(-0.10)
    decoy_y_lo: float = tunable(0.25)
    decoy_y_hi: float = tunable(0.36)

    # --- info: cabinet structure (cabinet frame: z = 0 at plinth top, +x = push-through) -----
    plinth_size: tuple = info((0.80, 1.00, 0.24))
    wall_x0: float = info(0.060)  # front wall front face
    wall_t: float = info(0.030)
    wall_h: float = info(0.130)
    door_half: float = info(0.085)  # doorway half-width (width 0.17)
    door_h: float = info(0.055)  # doorway height (header underside)
    pier_y1: float = info(0.44)  # piers span |y| in [door_half, pier_y1]
    alc_x1: float = info(0.340)  # alcove interior back face
    alc_half_w: float = info(0.130)  # alcove interior half-width
    alc_wall_t: float = info(0.030)
    alc_h: float = info(0.100)  # alcove interior height (roof underside)
    back_t: float = info(0.030)
    roof_t: float = info(0.030)
    ledge_x0: float = info(0.008)  # shutter ledge (plates ride on its top)
    ledge_x1: float = info(0.058)
    ledge_h: float = info(0.012)
    ridgeF_x0: float = info(-0.005)  # front guide ridge
    ridgeF_x1: float = info(0.008)
    ridgeM_x0: float = info(0.027)  # middle guide ridge
    ridgeM_x1: float = info(0.042)
    ridge_h: float = info(0.050)
    guide_y0: float = info(0.105)  # ledge+ridges span |y| in [guide_y0, guide_y1]
    guide_y1: float = info(0.470)
    stop_y0: float = info(0.455)  # end stops
    stop_y1: float = info(0.490)
    stop_h: float = info(0.100)
    # --- info: shutter plates ------------------------------------------------------------------
    plate_t: float = info(0.013)
    block_len: float = info(0.200)  # each solid block flanking the notch
    block_h: float = info(0.083)
    bridge_t: float = info(0.021)  # top bridge over the notch
    tab_w: float = info(0.032)  # tab post cross-section 13 x 32 mm — Franka jaw target
    tab_h: float = info(0.070)
    tab_y: float = info(0.240)  # |tab centre| along the plate (front: +, rear: -)
    plate_mass: float = info(0.25)
    # --- info: pan + decoy ---------------------------------------------------------------------
    pan_r: float = info(0.065)
    pan_h: float = info(0.035)
    handle_len: float = info(0.050)
    handle_w: float = info(0.022)
    handle_t: float = info(0.012)
    pan_mass: float = info(0.40)
    decoy_r: float = info(0.045)
    decoy_h: float = info(0.020)
    decoy_mass: float = info(0.15)
    # --- info: friction + contact ----------------------------------------------------------------
    mu_static: float = info(0.25)  # cabinet + plates: the slide is real but modest work
    mu_dynamic: float = info(0.22)
    pan_mu_static: float = info(0.45)  # pan + decoy grip: pushing is controlled
    pan_mu_dynamic: float = info(0.40)
    contact_offset: float = info(0.0015)
    pan_contact_offset: float = info(0.002)
    # --- info: colors ----------------------------------------------------------------------------
    plinth_color: tuple = info((0.45, 0.45, 0.48))
    wall_color: tuple = info((0.30, 0.32, 0.36))
    body_color: tuple = info((0.26, 0.28, 0.33))
    roof_color: tuple = info((0.22, 0.24, 0.30))
    guide_color: tuple = info((0.38, 0.38, 0.42))
    plateF_color: tuple = info((0.30, 0.55, 0.25))
    plateR_color: tuple = info((0.80, 0.45, 0.10))
    tab_color: tuple = info((0.08, 0.08, 0.08))
    pan_color: tuple = info((0.10, 0.10, 0.12))
    handle_color: tuple = info((0.15, 0.12, 0.10))
    decoy_color: tuple = info((0.88, 0.86, 0.80))

    # Derived (filled in __post_init__).
    slot_xF: float = field(default=None, init=False)  # front plate seat x (cab frame)
    slot_xR: float = field(default=None, init=False)  # rear plate seat x
    plate_z0: float = field(default=None, init=False)  # plate origin z when seated
    slab_x0: float = field(default=None, init=False)  # transit slab (doorway crossing)
    slab_x1: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        wall_x1 = self.wall_x0 + self.wall_t
        self.slot_xF = (self.ridgeF_x1 + self.ridgeM_x0) / 2
        self.slot_xR = (self.ridgeM_x1 + self.wall_x0) / 2
        self.plate_z0 = self.ledge_h + 0.002
        self.slab_x0 = self.ridgeF_x0 + 0.005
        self.slab_x1 = wall_x1 + 0.010

        dia = 2 * self.pan_r
        door_w = 2 * self.door_half
        # registered gate passes the pan with real clearance
        assert door_w - 2 * self.align_tol >= dia + 0.005, "registered gate must pass the pan"
        # ONE misaligned plate blocks the pan
        assert door_w - self.off_lo <= dia - 0.010, "one misaligned plate must block the pan"
        # spawn offsets never read as registered
        assert self.off_lo > self.pass_tol + 0.020
        assert self.align_tol < self.pass_tol < self.off_lo < self.off_hi
        # doorway height passes the FLAT pan (handle included) but not the on-edge pan
        assert self.door_h >= self.pan_h + 0.015, "flat pan must clear the header"
        assert self.door_h >= self.pan_h / 2 + self.handle_t / 2 + 0.015
        assert dia > self.door_h + 0.05, "on-edge pan must NOT fit under the header"
        # notch bridge passes over the sliding pan
        bridge_z0 = self.plate_z0 + self.block_h - self.bridge_t
        assert bridge_z0 >= self.pan_h + 0.02, "bridge must clear the sliding pan"
        # plate notch matches the doorway; travel reaches the stops with margin
        assert self.stop_y0 - (self.door_half + self.block_len + self.off_hi) >= 0.02, \
            "plates must spawn clear of the end stops"
        # guide interruption leaves the doorway span bare but still under the piers
        assert self.guide_y0 >= self.door_half + 0.015
        assert self.guide_y0 >= self.pan_r + 0.030, "pan path must miss the ledge ends"
        # plates supported on the ledge at the worst spawn offset: at offset +off_hi
        # the -y block spans [-door_half-block_len+off_hi, -door_half+off_hi]; its
        # overlap with the -y ledge segment [-guide_y1, -guide_y0] must stay real
        worst_inner = -self.door_half - self.block_len + self.off_hi  # block's low end
        support = -self.guide_y0 - worst_inner
        assert support >= 0.04, "plate must keep a ledge support patch at max offset"
        assert self.door_half + self.block_len + self.off_hi <= self.guide_y1
        # slots: real but tight clearance; plates never touch each other
        slotF = self.ridgeM_x0 - self.ridgeF_x1
        slotR = self.wall_x0 - self.ridgeM_x1
        assert 0.004 <= slotF - self.plate_t <= 0.008
        assert 0.004 <= slotR - self.plate_t <= 0.008
        assert self.slot_xR - self.slot_xF >= self.plate_t + 0.004
        # alcove: pan + handle fit; interior reachable past entry_x; roof seals the top
        assert self.alc_x1 - wall_x1 >= dia + self.handle_len + 0.02
        assert 2 * self.alc_half_w >= dia + 0.02
        assert self.entry_x >= wall_x1 + self.pan_r - 1e-6, "entry_x = pan fully past the wall"
        assert self.alc_x1 - self.pan_r >= self.entry_x + 0.02
        assert self.alc_h >= self.pan_h + 0.02
        # embodiment: tab post and pan handle inside a Franka's 80 mm jaw span
        assert self.plate_t <= 0.075 and self.tab_w <= 0.075 and self.handle_w <= 0.075
        # tab rises above the wall guides so it is graspable from above
        assert self.plate_z0 + self.block_h + self.tab_h > self.stop_h + 0.04
        # spawn bands: everything on the plinth, pan reach clear of the front ridge,
        # pan + decoy bands disjoint even with the handle pointing at the decoy
        reach = self.pan_r + self.handle_len
        assert self.pan_x_lo - reach >= -self.plinth_size[0] / 2 + 0.02
        assert self.pan_x_hi + reach <= self.ridgeF_x0 - 0.03
        assert self.pan_y_max + reach <= self.decoy_y_lo - self.decoy_r - 0.01
        assert self.decoy_y_hi + self.decoy_r <= self.plinth_size[1] / 2 - 0.05
        assert self.decoy_x_lo - self.decoy_r >= -self.plinth_size[0] / 2 + 0.02
        assert self.decoy_x_hi + self.decoy_r <= self.ridgeF_x0 - 0.02
        # decoy is clearly the smaller, different disc
        assert self.decoy_r + 0.015 <= self.pan_r


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("combo_shutter_pantry")
class ComboShutterPantryScene(BaseScene):
    cfg: ComboShutterPantrySceneCfg

    def __init__(self, cfg: ComboShutterPantrySceneCfg | None = None) -> None:
        super().__init__(cfg or ComboShutterPantrySceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        top = c.plinth_size[2]
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
            "cabinet": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet",
                spawn=_cabinet_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, top)),
            ),
            "plate_f": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PlateF",
                spawn=_plate_spawner_cfg(c, +c.tab_y, c.plateF_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_xF, 0.09, top + c.plate_z0)),
            ),
            "plate_r": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PlateR",
                spawn=_plate_spawner_cfg(c, -c.tab_y, c.plateR_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_xR, -0.09, top + c.plate_z0)),
            ),
            "pan": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pan",
                spawn=_pan_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-0.20, 0.0, top + 0.002)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=_decoy_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-0.20, 0.30, top + c.decoy_h / 2 + 0.002)),
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
        self.cabinet: RigidObject = env.iscene["cabinet"]
        self.plate_f: RigidObject = env.iscene["plate_f"]
        self.plate_r: RigidObject = env.iscene["plate_r"]
        self.pan: RigidObject = env.iscene["pan"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.alignF_latch = torch.zeros(n, device=dev)
        self.alignR_latch = torch.zeros(n, device=dev)
        self.both_latch = torch.zeros(n, device=dev)
        self.transit_latch = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: cabinet with xy jitter + free yaw; each plate seated in its
        slot at a random offset (magnitude in [off_lo, off_hi], independent random
        sign); pan on the porch with xy jitter + free yaw; decoy on a random +/-y side
        band; latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        top = c.plinth_size[2]

        # Burn a draw: the FIRST post-seed draw is near-constant across seeds.
        _ = torch.rand(m, 4, device=dev)

        # --- cabinet: xy jitter + free yaw ---
        cab_xy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.cab_jitter
        cab_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.cab_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = cab_xy
        st[:, 2] = top
        st[:, 3] = torch.cos(cab_yaw / 2)
        st[:, 6] = torch.sin(cab_yaw / 2)
        st[:, 0:3] += origin
        self.cabinet.write_root_state_to_sim(st, env_ids)

        ch, sh = torch.cos(cab_yaw), torch.sin(cab_yaw)

        def to_world(lx: torch.Tensor, ly: torch.Tensor, lz: float) -> torch.Tensor:
            out = torch.zeros(m, 3, device=dev)
            out[:, 0] = cab_xy[:, 0] + lx * ch - ly * sh
            out[:, 1] = cab_xy[:, 1] + lx * sh + ly * ch
            out[:, 2] = top + lz
            return out

        def write_at(body, lx, ly, lz: float, yaw: torch.Tensor) -> None:
            stt = torch.zeros(m, 13, device=dev)
            stt[:, 0:3] = to_world(lx, ly, lz)
            stt[:, 3] = torch.cos(yaw / 2)
            stt[:, 6] = torch.sin(yaw / 2)
            stt[:, 0:3] += origin
            body.write_root_state_to_sim(stt, env_ids)

        # --- plates: seated in their slots at random offsets (independent signs) ---
        for body, slot_x in ((self.plate_f, c.slot_xF), (self.plate_r, c.slot_xR)):
            sign = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
            mag = c.off_lo + torch.rand(m, device=dev) * (c.off_hi - c.off_lo)
            write_at(body, torch.full((m,), slot_x, device=dev), sign * mag,
                     c.plate_z0, cab_yaw)

        # --- pan: porch band, free yaw ---
        px = c.pan_x_lo + torch.rand(m, device=dev) * (c.pan_x_hi - c.pan_x_lo)
        py = (torch.rand(m, device=dev) * 2 - 1) * c.pan_y_max
        pyaw = cab_yaw + (torch.rand(m, device=dev) * 2 - 1) * math.pi
        write_at(self.pan, px, py, 0.002, pyaw)

        # --- decoy: random +/-y side band ---
        side = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        dx = c.decoy_x_lo + torch.rand(m, device=dev) * (c.decoy_x_hi - c.decoy_x_lo)
        dy = side * (c.decoy_y_lo + torch.rand(m, device=dev)
                     * (c.decoy_y_hi - c.decoy_y_lo))
        write_at(self.decoy, dx, dy, c.decoy_h / 2 + 0.002, cab_yaw)

        # --- latches ---
        self.alignF_latch[env_ids] = 0.0
        self.alignR_latch[env_ids] = 0.0
        self.both_latch[env_ids] = 0.0
        self.transit_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "cabinet": self.cabinet.data.root_state_w[env_ids].clone(),
            "plate_f": self.plate_f.data.root_state_w[env_ids].clone(),
            "plate_r": self.plate_r.data.root_state_w[env_ids].clone(),
            "pan": self.pan.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "alignF_latch": self.alignF_latch[env_ids].clone(),
            "alignR_latch": self.alignR_latch[env_ids].clone(),
            "both_latch": self.both_latch[env_ids].clone(),
            "transit_latch": self.transit_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.cabinet.write_root_state_to_sim(state["cabinet"], env_ids)
        self.plate_f.write_root_state_to_sim(state["plate_f"], env_ids)
        self.plate_r.write_root_state_to_sim(state["plate_r"], env_ids)
        self.pan.write_root_state_to_sim(state["pan"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.alignF_latch[env_ids] = state["alignF_latch"]
        self.alignR_latch[env_ids] = state["alignR_latch"]
        self.both_latch[env_ids] = state["both_latch"]
        self.transit_latch[env_ids] = state["transit_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A gray counter ({c.plinth_size[0] * 100:.0f} x {c.plinth_size[1] * 100:.0f} cm, "
            f"{c.plinth_size[2] * 100:.0f} cm tall) carries a roofed PANTRY ALCOVE behind a "
            f"front wall. The wall's only DOORWAY ({2 * c.door_half * 100:.0f} cm wide, "
            f"{c.door_h * 100:.1f} cm tall) is barred by TWO free-sliding SHUTTER PLATES "
            f"riding in guide slots on a low ledge in front of the wall: a GREEN plate "
            f"(front slot, black tab handle on one side) and an ORANGE plate (rear slot, "
            f"tab on the other side). Each plate has a doorway-sized cut-out, but both "
            f"spawn SHIFTED sideways by several centimetres — in either direction, "
            f"independently — so each plate alone bars the doorway. On the open counter "
            f"in front of the plates lie a dark PAN (a {2 * c.pan_r * 100:.0f} cm disc "
            f"with a stub handle) and a smaller pale SAUCER. Positions, plate shifts and "
            f"the unit's orientation vary per episode: read the push-through direction "
            f"and each plate's misalignment from the scene.\n"
            f"Goal: the PAN stored inside the alcove. The doorway is the only way in — "
            f"the alcove is roofed and walled — and it is far shorter than the pan's "
            f"diameter, so the pan can only pass FLAT, sliding on the counter, and only "
            f"once BOTH plate cut-outs are registered with the doorway (each within "
            f"about {c.align_tol * 1000:.0f} mm). So: slide each plate by its black tab "
            f"along its slot until its cut-out lines up with the doorway, then push the "
            f"pan flat across the counter, under the plate bridges and through the "
            f"doorway, until it rests fully inside the alcove (handle in or out of the "
            f"doorway line — the pan's disc must be wholly past the wall). Leave the "
            f"SAUCER anywhere OUTSIDE the alcove.\n"
            f"Judged at rest: pan flat and fully inside the alcove, saucer outside. A "
            f"pan set on the roof, left on the porch, or jammed against a misaligned "
            f"plate counts for nothing; the pan must have crossed the doorway while "
            f"both plates were registered — there is no other way in."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the green and orange shutter plates by their black tabs until both "
            "cut-outs line up with the doorway, then push the pan flat through the "
            "opening until it rests inside the pantry alcove. Leave the saucer outside."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _local(self, body_q: torch.Tensor, body_p: torch.Tensor,
               p_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(body_q, p_w - body_p)

    def cab_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> cabinet body frame (origin = plinth-top centre)."""
        return self._local(self.cabinet.data.root_quat_w,
                           self.cabinet.data.root_pos_w, p_w)

    def _yaw_of(self, body) -> torch.Tensor:
        q = body.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 3], q[:, 0])

    # ----- predicates -------------------------------------------------------------------------
    def plate_offset(self, body) -> torch.Tensor:
        """(N,) signed offset of a plate's notch centre from the doorway centre, along
        the cabinet's y axis. 0 = registered."""
        return self.cab_local(body.data.root_pos_w)[:, 1]

    def plate_seated(self, body, slot_x: float) -> torch.Tensor:
        """(N,) bool: plate riding upright in its slot (x and z in place)."""
        c = self.cfg
        loc = self.cab_local(body.data.root_pos_w)
        return ((loc[:, 0] - slot_x).abs() < 0.012) & ((loc[:, 2] - c.plate_z0).abs() < 0.015)

    def registered(self, body, slot_x: float, tol: float) -> torch.Tensor:
        """(N,) bool: plate seated AND its notch within `tol` of the doorway."""
        return self.plate_seated(body, slot_x) & (self.plate_offset(body).abs() < tol)

    def pan_inside(self) -> torch.Tensor:
        """(N,) bool: pan disc fully past the wall's inner face, resting on the alcove
        floor (cabinet body frame)."""
        c = self.cfg
        loc = self.cab_local(self.pan.data.root_pos_w)
        return ((loc[:, 0] > c.entry_x) & (loc[:, 0] < c.alc_x1 - c.pan_r + 0.01)
                & (loc[:, 1].abs() < c.alc_half_w - c.pan_r + 0.015)
                & (loc[:, 2] > -0.01) & (loc[:, 2] < 0.02))

    def decoy_in_alcove(self) -> torch.Tensor:
        """(N,) bool: decoy centre anywhere in the alcove volume (to be REJECTED)."""
        c = self.cfg
        loc = self.cab_local(self.decoy.data.root_pos_w)
        return ((loc[:, 0] > c.wall_x0) & (loc[:, 0] < c.alc_x1 + c.back_t)
                & (loc[:, 1].abs() < c.alc_half_w + c.alc_wall_t)
                & (loc[:, 2] < c.alc_h))

    def pan_upright(self) -> torch.Tensor:
        """(N,) bool: pan resting flat within `upright_deg`."""
        from isaaclab.utils.math import quat_apply

        up = quat_apply(self.pan.data.root_quat_w,
                        torch.tensor([[0.0, 0.0, 1.0]],
                                     device=self.pan.data.root_quat_w.device
                                     ).expand(self.env.num_envs, 3))
        return up[:, 2] > math.cos(math.radians(self.cfg.upright_deg))

    def settled(self) -> torch.Tensor:
        """(N,) bool: pan, decoy and both plates |lin vel| below `settle_lin`."""
        c = self.cfg
        return ((self.pan.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.decoy.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.plate_f.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.plate_r.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin))

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch, each physics substep: each plate ever registered (seated, within
        `align_tol`), both registered SIMULTANEOUSLY, and the pan crossing the doorway
        slab WHILE both plates are registered within `pass_tol` — the pathway latch
        that rejects a pan written directly into the alcove."""
        c = self.cfg
        f_now = self.registered(self.plate_f, c.slot_xF, c.align_tol)
        r_now = self.registered(self.plate_r, c.slot_xR, c.align_tol)
        self.alignF_latch = torch.maximum(self.alignF_latch, f_now.float())
        self.alignR_latch = torch.maximum(self.alignR_latch, r_now.float())
        self.both_latch = torch.maximum(self.both_latch, (f_now & r_now).float())
        loc = self.cab_local(self.pan.data.root_pos_w)
        in_slab = ((loc[:, 0] > c.slab_x0) & (loc[:, 0] < c.slab_x1)
                   & (loc[:, 1].abs() < c.door_half) & (loc[:, 2] < 0.04))
        pass_f = self.registered(self.plate_f, c.slot_xF, c.pass_tol)
        pass_r = self.registered(self.plate_r, c.slot_xR, c.pass_tol)
        self.transit_latch = torch.maximum(self.transit_latch,
                                           (in_slab & pass_f & pass_r).float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: pan flat, settled and fully inside the alcove; the transit latch
        earned (it crossed the doorway while both plates were registered); the saucer
        decoy not in the alcove."""
        return (self.pan_inside() & self.pan_upright() & (self.transit_latch > 0.5)
                & ~self.decoy_in_alcove() & self.settled())

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15 per plate ever registered + 0.20 both registered
        simultaneously + 0.20 doorway transit while registered, capped at 0.70;
        exactly 1.0 iff success(). Doing nothing scores ~0; the seed's strategy (pan
        set on top of the fixture) scores ~0."""
        base = (0.15 * self.alignF_latch + 0.15 * self.alignR_latch
                + 0.20 * self.both_latch + 0.20 * self.transit_latch).clamp(0.0, 0.70)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="combo_shutter_pantry", robot="null"))

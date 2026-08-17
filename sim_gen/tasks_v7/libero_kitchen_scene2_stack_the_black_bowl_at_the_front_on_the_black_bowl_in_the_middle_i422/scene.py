"""ScreedPatchScene — inter the red keystone at the bottom of the deck pit, pave over it
with EXACTLY enough gray pavers to bring the patch flush, then screed the patch: drag the
bar straight across it at deck level and park it clear on the far side.

Derived from libero_90 kitchen_scene2 "stack the black bowl at the front on the black bowl
in the middle". The seed is ONE unconstrained grasp-carry-place: pick bowl_1, hover it over
bowl_2, release; success is a loose xy/z proximity between two free objects anywhere on the
counter. Here stacking is still the motor primitive — flat slabs are lowered one onto
another — but every property that made the seed easy is replaced by a constraint the rubric
actually measures:

  * the stack is built DOWN INSIDE A CAVITY, not up in the open: a rectangular pit sinks
    through the deck, and the first slab must come to rest ON THE PIT FLOOR. The stack has
    an absolute, episode-dependent target height (the deck plane), not a relative
    "A near B" relation;
  * the stack is ORDERED and the order is checked geometrically: the RED keystone slab must
    be the BOTTOM course (its rest z is pinned to the pit floor within 6 mm) and must end
    covered by at least one paver directly above it. Building the same-looking flush stack
    with the keystone on TOP — the seed's "stack A on B" plan — scores 0.00;
  * the stack is COUNTED, and the count changes per episode: the pit is two or three slab
    thicknesses deep (visible from the pit-floor depth), so exactly depth-1 pavers are
    needed on the keystone. One paver short leaves the patch top a full 20 mm low; one
    paver extra leaves it 20 mm proud; the flush band is +/-4 mm, so any miscount fails —
    and a proud patch also physically walls off the screed pass;
  * the goal has a second, physics-ordered clause the seed has no analogue of: after the
    patch is flush the SCREED BAR must be dragged across it AT DECK LEVEL (a real sliding
    pass — the latch requires the bar over the pit, in the deck-level z band, IN MOTION,
    while the patch below is already keystone-down, covered, and flush). A bar teleported
    to the far side has zero written velocity and can never earn the pass; a bar carried
    over in the air is outside the 8 mm z band; a pass swept BEFORE the patch is complete
    latches nothing. The bar is longer than the pit is wide, so the empty pit does not
    swallow it — an early sweep is possible, just worthless.

Success (simultaneous, settled): keystone interred (resting on the pit floor, flat) AND
covered by a paver AND the patch top flush with the deck (+/-4 mm, every in-pit slab flat)
AND the screed pass latched AND the bar parked clear PAST the pit on the far side at deck
level AND every dynamic body at rest.

Rubric (graded 0..1, latched in post_step, additive; 1.0 iff success()):
  0.00  nothing happened (all slabs still racked on the depot skids)
  +0.15 the keystone has rested interred at the bottom of the pit (velocity-gated)
  +0.35 the patch has been complete at rest: keystone interred + covered + flush
  +0.20 the screed pass: bar over the pit, deck-level, in real motion, patch complete below
  1.00  success() (overrides the 0.70 partial sum)

Assets are fully procedural (no external files):
  * bench: kinematic slab (top z=0.12);
  * deck: four kinematic plates on the bench (thickness 80 mm, top z=0.20) leaving a
    154 x 164 mm rectangular pit opening; a kinematic pit-floor block whose top sits
    depth*20 mm below the deck (depth in {2,3}, randomized per episode);
  * slabs: one RED keystone + three gray pavers, all 130 x 140 x 20 mm dynamic boxes,
    racked flat on two kinematic depot skid rails (24 mm tall — under-edge clearance for a
    parallel jaw) to one side of the pit;
  * screed bar: dynamic compound body — 50 x 240 x 15 mm blade plate + 24 mm square handle
    post (the pinch point) riding on two 6 mm end skid runners — parked on the deck on the
    near side of the pit. The 240 mm span out-reaches the 164 mm pit, so a dragged bar
    rides its runners on the flanking deck strips across the opening, the blade passing
    6 mm above the deck — clear of a flush patch, but any proud slab (one course = 20 mm)
    physically blocks the pass.

Per-episode randomization (verified by readback in the smoke): pit depth (2 or 3 courses),
pit xy, depot side (left/right), which rack slot holds the keystone, bar park x, slab yaws.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering the
scene — stays app-free.
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


# ----- custom compound spawner (screed bar) -------------------------------------------------------
# One rigid body: root Xform with RigidBodyAPI + explicit MassAPI (custom spawn funcs do NOT
# apply mass_props), child box colliders. Authored through `clone()` so per-env replication
# is idempotent.

_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_root(prim_path: str, translation, orientation, *, kinematic: bool, mass: float | None):
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    if mass is not None:
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
        px_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        px_rb.CreateMaxDepenetrationVelocityAttr(0.5)
        px_rb.CreateLinearDampingAttr(0.05)
        px_rb.CreateAngularDampingAttr(0.10)
    return root


def _child_box(prim_path: str, name: str, *, tx: float, ty: float, tz: float,
               sx: float, sy: float, sz: float, color: tuple, contact_offset: float):
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    box = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
    box.CreateSizeAttr(1.0)
    bxf = UsdGeom.Xformable(box.GetPrim())
    bxf.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))
    bxf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(box.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    return box


def _spawn_bar(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The screed bar: blade plate (origin at plate centre) + raised square handle post on
    top — the one pinch point — riding on two end skid runners. The runners put the blade
    bottom skid_h above the deck, so the blade CLEARS a flush-band-proud patch (the runners
    straddle the pit on the flanking deck strips) instead of catching its edge. CoM stays at
    the body origin (plate centre) so a horizontal drag force applied there barely tips it."""
    root = _apply_root(prim_path, translation, orientation, kinematic=False,
                       mass=cfg.mass_props.mass)
    _child_box(prim_path, "plate", tx=0.0, ty=0.0, tz=0.0,
               sx=cfg.plate_x, sy=cfg.plate_y, sz=cfg.plate_t,
               color=cfg.color, contact_offset=cfg.contact_offset)
    _child_box(prim_path, "handle", tx=0.0, ty=0.0,
               tz=cfg.plate_t / 2 + cfg.handle_h / 2,
               sx=cfg.handle_w, sy=cfg.handle_w, sz=cfg.handle_h,
               color=cfg.handle_color, contact_offset=cfg.contact_offset)
    for sgn, name in ((-1.0, "skid_s"), (1.0, "skid_n")):
        _child_box(prim_path, name, tx=0.0,
                   ty=sgn * (cfg.plate_y / 2 - cfg.skid_y / 2),
                   tz=-cfg.plate_t / 2 - cfg.skid_h / 2,
                   sx=cfg.skid_x, sy=cfg.skid_y, sz=cfg.skid_h,
                   color=cfg.color, contact_offset=cfg.contact_offset)
    return root


def _bar_spawner_cfg(defaults: dict, *, mass: float) -> Any:
    if "bar" not in _SPAWNER_CACHE:
        from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
        from isaaclab.sim.utils import clone
        from isaaclab.utils import configclass

        @configclass
        class BarSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bar)
            plate_x: float = 0.050
            plate_y: float = 0.240
            plate_t: float = 0.015
            handle_w: float = 0.024
            handle_h: float = 0.055
            skid_x: float = 0.050
            skid_y: float = 0.020
            skid_h: float = 0.006
            color: tuple = (0.30, 0.35, 0.55)
            handle_color: tuple = (0.90, 0.75, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["bar"] = BarSpawnerCfg

    import isaaclab.sim as sim_utils

    return _SPAWNER_CACHE["bar"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
        **defaults,
    )


# ----- scene cfg ------------------------------------------------------------------------------------
@dataclass
class ScreedPatchSceneCfg(BaseCfg):
    """Config for `ScreedPatchScene`. Honesty margins asserted in __post_init__."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    pit_x_tol: float = tunable(0.045)  # slab centre |x - pit_x| below this counts as in-pit
    pit_y_tol: float = tunable(0.050)  # slab centre |y - pit_y| below this counts as in-pit
    in_pit_below: float = tunable(0.004)  # slab centre must sit below deck_top - this (m)
    inter_z_tol: float = tunable(0.006)  # keystone rest z vs pit-floor seat, band (m)
    flat_max_deg: float = tunable(7.0)  # slab local +z within this of world up
    cover_dz_min: float = tunable(0.014)  # paver centre at least this above keystone centre
    cover_xy_tol: float = tunable(0.060)  # covering paver horizontal offset from keystone
    flush_tol: float = tunable(0.004)  # patch top vs deck plane band — a miscount is 20 mm
    sweep_y_tol: float = tunable(0.060)  # bar centre |y - pit_y| during the screed pass
    sweep_z_tol: float = tunable(0.008)  # bar centre vs deck-level rest z during the pass
    sweep_min_speed: float = tunable(0.03)  # bar |vx| must exceed this for the pass to latch
    sweep_vz_max: float = tunable(0.015)  # ... while |vz| stays below this (no drop transient)
    sweep_streak: int = tunable(5)  # consecutive qualifying steps required (~42 ms)
    bar_flat_max_deg: float = tunable(8.0)  # bar tilt gate (pass + park)
    clear_dx: float = tunable(0.13)  # bar centre past pit centre by this = parked clear
    clear_z_tol: float = tunable(0.010)  # parked bar must rest at deck level (not fallen off)
    settle_speed: float = tunable(0.05)  # max |v| of every dynamic body when judging (m/s)
    latch_speed: float = tunable(0.10)  # rest milestones only latch while this slow

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    pit_x_jitter: float = tunable(0.030)  # uniform +/- x jitter of the pit centre
    pit_y_jitter: float = tunable(0.040)  # uniform +/- y jitter of the pit centre
    depth_min: int = tunable(2)  # pit depth in slab courses, sampled uniform in
    depth_max: int = tunable(3)  # [depth_min, depth_max]
    mirror_depot: bool = tunable(True)  # per-episode: depot racks on +y or -y side
    bar_jitter: float = tunable(0.020)  # uniform +/- x jitter of the bar park
    slab_yaw_deg: float = tunable(5.0)  # uniform +/- yaw jitter of racked slabs

    # --- tunable: placement (deck frame; intended arm base at (pit_x-0.42, pit_y, 0.20)) --------
    depot_dist: float = tunable(0.30)  # depot rack centreline |y| offset from the pit centre
    depot_x0: float = tunable(-0.10)  # depot rack centre x offset from the pit centre
    slot_pitch: float = tunable(0.15)  # rack slot spacing along x (4 slots)
    bar_park_dx: float = tunable(-0.24)  # bar park x offset from the pit centre (near side)

    # --- info: structure -------------------------------------------------------------------------
    bench_size: tuple = info((1.5, 1.3))  # kinematic bench slab top (x, y)
    bench_h: float = info(0.12)  # bench top z
    deck_t: float = info(0.08)  # deck plate thickness
    deck_top: float = info(0.20)  # deck plane z = bench_h + deck_t (asserted)
    pit_lx: float = info(0.154)  # pit opening x span
    pit_ly: float = info(0.164)  # pit opening y span
    slab_lx: float = info(0.130)  # slab x — 12 mm side clearance in the pit
    slab_ly: float = info(0.140)  # slab y — 12 mm side clearance in the pit
    slab_t: float = info(0.020)  # slab thickness = one course = 5x the flush band
    key_mass: float = info(0.20)
    paver_mass: float = info(0.18)
    n_pavers: int = info(3)  # supply covers depth 3 (2 needed) AND the overfill probe
    floor_lx: float = info(0.150)  # pit-floor block nearly fills the opening (2 mm/side):
    floor_ly: float = info(0.160)  # nothing can slip beneath it
    floor_t: float = info(0.020)
    side_plate: tuple = info((0.42, 0.90))  # west/east deck plates (x, y)
    strip_plate: tuple = info((0.154, 0.368))  # north/south deck strips (x, y)
    rail_l: float = info(0.60)  # depot skid rail length (x)
    rail_w: float = info(0.014)
    rail_h: float = info(0.024)  # 24 mm under-slab clearance for a parallel-jaw edge grasp
    rail_gap: float = info(0.090)  # rail centreline spacing (slabs bridge both rails)
    bar_plate: tuple = info((0.050, 0.240, 0.015))  # 240 mm out-spans the 164 mm pit
    bar_handle: tuple = info((0.024, 0.055))  # square post (w, h): a Franka pinch point
    bar_skid: tuple = info((0.050, 0.020, 0.006))  # end runners (x, y, h): blade rides
    # skid_h above the deck, clearing a flush-band-proud patch instead of catching its edge
    bar_mass: float = info(0.12)
    key_color: tuple = info((0.75, 0.10, 0.10))
    paver_color: tuple = info((0.55, 0.55, 0.58))
    contact_offset: float = info(0.002)

    def __post_init__(self) -> None:
        assert abs(self.deck_top - (self.bench_h + self.deck_t)) < 1e-9
        # slabs drop into the pit with real (but not sloppy) side clearance
        assert 0.015 < self.pit_lx - self.slab_lx < 0.040
        assert 0.015 < self.pit_ly - self.slab_ly < 0.040
        # nothing can wedge under the pit-floor block
        assert 0.0 < (self.pit_lx - self.floor_lx) / 2 < 0.004
        assert 0.0 < (self.pit_ly - self.floor_ly) / 2 < 0.004
        # a one-slab miscount (20 mm) can never sneak through the flush band, even with the
        # ~8 mm max centre-rise a 7-degree-tilt slab could fake
        assert self.slab_t > 3.0 * self.flush_tol
        assert self.slab_t - 0.070 * math.sin(math.radians(self.flat_max_deg)) \
            > 2.0 * self.flush_tol
        # keystone-on-top is at least one full course (20 mm) out of the interred z band
        assert self.inter_z_tol < self.slab_t / 2
        # the bar out-spans the pit: it shoulders across the opening instead of falling in
        assert self.bar_plate[1] > self.pit_ly + 0.040
        # the blade rides on its skid runners with clearance over the WHOLE flush band, so a
        # legitimately flush (up to flush_tol proud) patch can never catch the blade edge
        assert self.bar_skid[2] > self.flush_tol + 0.001
        # the runners' inner edges stay on the deck strips flanking the pit (never dip in)
        assert self.bar_plate[1] / 2 - self.bar_skid[1] > self.pit_ly / 2 + 0.005
        # ... and their outer edges stay on the strips (strip spans the runner stance)
        assert self.bar_plate[1] / 2 < self.pit_ly / 2 + self.strip_plate[1]
        # every depth needs at least one covering paver, and the supply covers depth_max
        # plus the overfill probe (depth_max pavers on top of the keystone)
        assert self.depth_min >= 2
        assert self.n_pavers >= self.depth_max
        # the handle post is a comfortable parallel-jaw pinch (< 80 mm stroke)
        assert self.bar_handle[0] < 0.080
        # the depot racks stay clear of the bar's sweep lane
        assert self.depot_dist - self.slab_ly / 2 > self.bar_plate[1] / 2 + 0.050

    # derived (not fields to keep the dataclass frozen-simple)
    @property
    def bar_rest_z(self) -> float:
        return self.deck_top + self.bar_skid[2] + self.bar_plate[2] / 2

    @property
    def rack_z(self) -> float:
        return self.deck_top + self.rail_h + self.slab_t / 2


# ----- scene -----------------------------------------------------------------------------------------
@SCENES.register("screed_patch")
class ScreedPatchScene(BaseScene):
    cfg: ScreedPatchSceneCfg

    def __init__(self, cfg: ScreedPatchSceneCfg | None = None) -> None:
        super().__init__(cfg or ScreedPatchSceneCfg())

    # ----- assets ---------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, bench, four deck plates, pit floor, two depot rails, four slabs,
        the screed bar — at nominal poses (reset() re-places everything and samples the
        layout)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg

        def kin_box(name: str, size: tuple, pos: tuple, color: tuple) -> RigidObjectCfg:
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
            )

        def dyn_slab(name: str, color: tuple, mass: float, pos: tuple) -> RigidObjectCfg:
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.CuboidCfg(
                    size=(c.slab_lx, c.slab_ly, c.slab_t),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=False, max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.10),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.5, dynamic_friction=0.5, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
            )

        gray = (0.45, 0.45, 0.48)
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
        out["bench"] = kin_box("Bench", (c.bench_size[0], c.bench_size[1], c.bench_h),
                               (0.0, 0.0, c.bench_h / 2), (0.35, 0.35, 0.38))
        plate_z = c.bench_h + c.deck_t / 2
        out["deck_w"] = kin_box("DeckW", (c.side_plate[0], c.side_plate[1], c.deck_t),
                                (-c.pit_lx / 2 - c.side_plate[0] / 2, 0.0, plate_z), gray)
        out["deck_e"] = kin_box("DeckE", (c.side_plate[0], c.side_plate[1], c.deck_t),
                                (c.pit_lx / 2 + c.side_plate[0] / 2, 0.0, plate_z), gray)
        out["deck_n"] = kin_box("DeckN", (c.strip_plate[0], c.strip_plate[1], c.deck_t),
                                (0.0, c.pit_ly / 2 + c.strip_plate[1] / 2, plate_z), gray)
        out["deck_s"] = kin_box("DeckS", (c.strip_plate[0], c.strip_plate[1], c.deck_t),
                                (0.0, -c.pit_ly / 2 - c.strip_plate[1] / 2, plate_z), gray)
        out["pit_floor"] = kin_box("PitFloor", (c.floor_lx, c.floor_ly, c.floor_t),
                                   (0.0, 0.0, c.deck_top - 2 * c.slab_t - c.floor_t / 2),
                                   (0.22, 0.22, 0.25))
        rail_z = c.deck_top + c.rail_h / 2
        out["rail_a"] = kin_box("RailA", (c.rail_l, c.rail_w, c.rail_h),
                                (c.depot_x0, c.depot_dist - c.rail_gap / 2, rail_z),
                                (0.45, 0.32, 0.18))
        out["rail_b"] = kin_box("RailB", (c.rail_l, c.rail_w, c.rail_h),
                                (c.depot_x0, c.depot_dist + c.rail_gap / 2, rail_z),
                                (0.45, 0.32, 0.18))
        out["keystone"] = dyn_slab("Keystone", c.key_color, c.key_mass,
                                   (c.depot_x0 - 1.5 * c.slot_pitch, c.depot_dist, c.rack_z))
        for k in range(c.n_pavers):
            out[f"paver{k}"] = dyn_slab(
                f"Paver{k}", c.paver_color, c.paver_mass,
                (c.depot_x0 + (k - 0.5) * c.slot_pitch, c.depot_dist, c.rack_z))
        out["bar"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bar",
            spawn=_bar_spawner_cfg(
                dict(plate_x=c.bar_plate[0], plate_y=c.bar_plate[1], plate_t=c.bar_plate[2],
                     handle_w=c.bar_handle[0], handle_h=c.bar_handle[1],
                     skid_x=c.bar_skid[0], skid_y=c.bar_skid[1], skid_h=c.bar_skid[2],
                     contact_offset=c.contact_offset),
                mass=c.bar_mass),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(c.bar_park_dx, 0.0, c.bar_rest_z)),
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

    # ----- lifecycle --------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        self.deck_w: RigidObject = env.iscene["deck_w"]
        self.deck_e: RigidObject = env.iscene["deck_e"]
        self.deck_n: RigidObject = env.iscene["deck_n"]
        self.deck_s: RigidObject = env.iscene["deck_s"]
        self.pit_floor: RigidObject = env.iscene["pit_floor"]
        self.rail_a: RigidObject = env.iscene["rail_a"]
        self.rail_b: RigidObject = env.iscene["rail_b"]
        self.keystone: RigidObject = env.iscene["keystone"]
        self.pavers: list[RigidObject] = [env.iscene[f"paver{k}"] for k in range(c.n_pavers)]
        self.bar: RigidObject = env.iscene["bar"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # layout readback (local frame)
        self.pit_xy = torch.zeros(n, 2, device=dev)
        self.depth_units = torch.full((n,), 2.0, device=dev)
        self.depot_side = torch.ones(n, device=dev)
        self.key_slot = torch.zeros(n, device=dev)
        self.bar_park_x = torch.zeros(n, device=dev)
        # progress latches (post_step; cleared per reset)
        self.ever_interred = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ever_capped = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ever_swept = torch.zeros(n, dtype=torch.bool, device=dev)
        self.sweep_streak_ctr = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample pit centre + depth, depot side + keystone slot, bar park;
        re-place the deck plates / pit floor / rails around the sampled pit; rack the slabs;
        park the bar; clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def u(amp: float) -> torch.Tensor:
            return (torch.rand(m, device=dev) * 2 - 1) * amp

        def write(body: RigidObject, x: torch.Tensor, y: torch.Tensor,
                  z: torch.Tensor | float, yaw: torch.Tensor | None = None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = x
            st[:, 1] = y
            st[:, 2] = z if isinstance(z, torch.Tensor) else float(z)
            yw = yaw if yaw is not None else torch.zeros(m, device=dev)
            st[:, 3] = torch.cos(yw / 2)
            st[:, 6] = torch.sin(yw / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- pit centre + depth ---
        px = u(c.pit_x_jitter)
        py = u(c.pit_y_jitter)
        self.pit_xy[env_ids, 0] = px
        self.pit_xy[env_ids, 1] = py
        n_depths = c.depth_max - c.depth_min + 1
        depth = (c.depth_min
                 + torch.randint(n_depths, (m,), device=dev).float())
        self.depth_units[env_ids] = depth

        # --- deck plates + pit floor (kinematic; re-placed around the sampled pit) ---
        plate_z = c.bench_h + c.deck_t / 2
        zeros = torch.zeros(m, device=dev)
        write(self.deck_w, px - c.pit_lx / 2 - c.side_plate[0] / 2, zeros, plate_z)
        write(self.deck_e, px + c.pit_lx / 2 + c.side_plate[0] / 2, zeros, plate_z)
        write(self.deck_n, px, py + c.pit_ly / 2 + c.strip_plate[1] / 2, plate_z)
        write(self.deck_s, px, py - c.pit_ly / 2 - c.strip_plate[1] / 2, plate_z)
        floor_top = c.deck_top - depth * c.slab_t
        write(self.pit_floor, px, py, floor_top - c.floor_t / 2)

        # --- depot rails + racked slabs ---
        side = ((torch.rand(m, device=dev) < 0.5).float() * 2 - 1) if c.mirror_depot \
            else torch.ones(m, device=dev)
        self.depot_side[env_ids] = side
        y_d = py + side * c.depot_dist
        rail_z = c.deck_top + c.rail_h / 2
        write(self.rail_a, px + c.depot_x0, y_d - c.rail_gap / 2, rail_z)
        write(self.rail_b, px + c.depot_x0, y_d + c.rail_gap / 2, rail_z)
        slot = torch.randint(1 + c.n_pavers, (m,), device=dev).float()
        self.key_slot[env_ids] = slot

        def slot_x(idx: torch.Tensor) -> torch.Tensor:
            return px + c.depot_x0 + (idx - 0.5 * c.n_pavers) * c.slot_pitch

        yaw_amp = math.radians(c.slab_yaw_deg)
        write(self.keystone, slot_x(slot), y_d, c.rack_z + 0.001, u(yaw_amp))
        for k, paver in enumerate(self.pavers):
            idx = torch.full((m,), float(k), device=dev)
            idx = torch.where(idx >= slot, idx + 1.0, idx)  # skip the keystone's slot
            write(paver, slot_x(idx), y_d, c.rack_z + 0.001, u(yaw_amp))

        # --- bar park (near side of the pit) ---
        bx = px + c.bar_park_dx + u(c.bar_jitter)
        self.bar_park_x[env_ids] = bx
        write(self.bar, bx, py, c.bar_rest_z + 0.001)

        for latch in (self.ever_interred, self.ever_capped, self.ever_swept):
            latch[env_ids] = False
        self.sweep_streak_ctr[env_ids] = 0

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch progress milestones at sim rate. Rest milestones are velocity-gated; the
        screed pass demands the bar IN MOTION at deck level over an already-complete patch —
        a teleported bar (zero written velocity), a lifted bar (out of the z band), or a
        premature pass (patch incomplete) latches nothing. Latched credit survives later
        mishaps, so along a correct run the score never decreases."""
        c = self.cfg
        key_slow = self.keystone.data.root_lin_vel_w.norm(dim=-1) < c.latch_speed
        self.ever_interred |= self.interred() & key_slow
        slabs_slow = key_slow
        for p in self.pavers:
            slabs_slow &= p.data.root_lin_vel_w.norm(dim=-1) < c.latch_speed
        patch = self.interred() & self.covered() & self.flush()
        self.ever_capped |= patch & slabs_slow
        bar = self.bar.data.root_pos_w - self.env_origins
        over_pit = ((bar[:, 0] - self.pit_xy[:, 0]).abs() < c.pit_lx / 2) \
            & ((bar[:, 1] - self.pit_xy[:, 1]).abs() < c.sweep_y_tol)
        in_band = (bar[:, 2] - c.bar_rest_z).abs() < c.sweep_z_tol
        # the pass must be SUSTAINED HORIZONTAL sliding: |vx| above the floor, |vz| vetoed
        # (a bar teleport-dropped onto the patch has a vertical impact transient and zero
        # written vx; it can never accumulate the streak), for `sweep_streak` steps in a row
        vel = self.bar.data.root_lin_vel_w
        moving = (vel[:, 0].abs() > c.sweep_min_speed) & (vel[:, 2].abs() < c.sweep_vz_max)
        bar_flat = self._up_z(self.bar.data.root_quat_w).clamp(-1, 1) >= \
            math.cos(math.radians(c.bar_flat_max_deg))
        cond = patch & over_pit & in_band & moving & bar_flat
        self.sweep_streak_ctr = torch.where(cond, self.sweep_streak_ctr + 1,
                                            torch.zeros_like(self.sweep_streak_ctr))
        self.ever_swept |= self.sweep_streak_ctr >= c.sweep_streak

    # ----- state (full, restorable) ------------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = dict(deck_w=self.deck_w, deck_e=self.deck_e, deck_n=self.deck_n,
                      deck_s=self.deck_s, pit_floor=self.pit_floor, rail_a=self.rail_a,
                      rail_b=self.rail_b, keystone=self.keystone, bar=self.bar,
                      **{f"paver{k}": p for k, p in enumerate(self.pavers)})
        out = {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()}
        out["layout"] = torch.cat([
            self.pit_xy[env_ids], self.depth_units[env_ids, None],
            self.depot_side[env_ids, None], self.key_slot[env_ids, None],
            self.bar_park_x[env_ids, None]], dim=1)
        out["latches"] = torch.stack([self.ever_interred[env_ids], self.ever_capped[env_ids],
                                      self.ever_swept[env_ids]], dim=1)
        out["streak"] = self.sweep_streak_ctr[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = dict(deck_w=self.deck_w, deck_e=self.deck_e, deck_n=self.deck_n,
                      deck_s=self.deck_s, pit_floor=self.pit_floor, rail_a=self.rail_a,
                      rail_b=self.rail_b, keystone=self.keystone, bar=self.bar,
                      **{f"paver{k}": p for k, p in enumerate(self.pavers)})
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state[nm], env_ids)
        lay = state["layout"]
        self.pit_xy[env_ids] = lay[:, 0:2]
        self.depth_units[env_ids] = lay[:, 2]
        self.depot_side[env_ids] = lay[:, 3]
        self.key_slot[env_ids] = lay[:, 4]
        self.bar_park_x[env_ids] = lay[:, 5]
        lat = state["latches"]
        self.ever_interred[env_ids] = lat[:, 0]
        self.ever_capped[env_ids] = lat[:, 1]
        self.ever_swept[env_ids] = lat[:, 2]
        self.sweep_streak_ctr[env_ids] = state["streak"]

    # ----- description -------------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A raised gray work DECK (top {c.deck_top * 100:.0f} cm above the floor, "
            f"{c.deck_t * 1000:.0f} mm thick) with a rectangular PIT sunk through it — "
            f"opening {c.pit_lx * 1000:.0f} x {c.pit_ly * 1000:.0f} mm, near the deck "
            f"centre (exact spot changes every episode). The pit has a flat dark floor "
            f"whose depth changes every episode: it lies exactly TWO or THREE slab "
            f"thicknesses ({2 * c.slab_t * 1000:.0f} or {3 * c.slab_t * 1000:.0f} mm) "
            f"below the deck plane — judge the depth by looking into the pit. To one side "
            f"of the pit (left or right, changing every episode) four identical slabs "
            f"({c.slab_lx * 1000:.0f} x {c.slab_ly * 1000:.0f} x {c.slab_t * 1000:.0f} mm) "
            f"lie racked in a row on two low skid rails ({c.rail_h * 1000:.0f} mm tall, so "
            f"a gripper can reach under a slab's edge): ONE slab is RED (the keystone), "
            f"the other three are GRAY pavers; which rack slot holds the red one changes "
            f"every episode. On the near side of the pit a blue SCREED BAR (a "
            f"{c.bar_plate[0] * 1000:.0f} x {c.bar_plate[1] * 1000:.0f} x "
            f"{c.bar_plate[2] * 1000:.0f} mm plate with a yellow "
            f"{c.bar_handle[0] * 1000:.0f} mm square handle post — the grasp point — and "
            f"two {c.bar_skid[2] * 1000:.0f} mm skid runners under the plate ends) rests "
            f"on its runners on the deck; the bar is longer than the pit is wide, so it "
            f"slides across the opening riding its runners on the deck strips flanking the "
            f"pit, its blade passing {c.bar_skid[2] * 1000:.0f} mm above the deck plane — "
            f"clearing a flush patch but jamming on any slab standing a course proud.\n"
            f"Goal, in this order: (1) INTER the red keystone — lower it flat so it rests "
            f"directly ON the pit floor (it must end at the BOTTOM of the pit; a keystone "
            f"resting on top of pavers, or above the floor by more than "
            f"{c.inter_z_tol * 1000:.0f} mm, does not count). (2) PAVE over it with exactly "
            f"enough gray pavers — depth minus one, i.e. ONE paver for the shallow pit, TWO "
            f"for the deep pit — so the finished patch top is FLUSH with the deck plane "
            f"within +/-{c.flush_tol * 1000:.0f} mm, every slab lying flat; one paver too "
            f"few leaves the patch {c.slab_t * 1000:.0f} mm low, one too many leaves it "
            f"{c.slab_t * 1000:.0f} mm proud, and both fail — ANY slab sitting over the "
            f"pit above the deck plane breaks flush. Unused pavers stay on the "
            f"rack. (3) SCREED: only after the patch is complete, drag the bar by its "
            f"handle straight across the patch AT DECK LEVEL — a real sliding pass over "
            f"the pit (a bar lifted over in the air, or slid across before the patch is "
            f"finished, earns nothing) — and leave it parked flat on the deck CLEAR on the "
            f"far side of the pit (at least {c.clear_dx * 1000:.0f} mm past the pit "
            f"centre, on the side opposite its start). The task is complete only when the "
            f"keystone is interred and covered, the patch is flush, the screed pass has "
            f"been made, the bar is parked clear, and everything is at rest."
        )

    def instruction(self) -> str:
        return (
            "Lower the red keystone slab flat onto the pit floor first. Then stack exactly "
            "enough gray pavers on it — one for the shallow pit, two for the deep pit — so "
            "the patch is flush with the deck. Then drag the screed bar by its yellow "
            "handle straight across the finished patch, sliding at deck level, and park it "
            "clear on the far side."
        )

    # ----- geometric predicates ------------------------------------------------------------------------
    def _up_z(self, quat: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        shape = quat.shape[:-1]
        ez = torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(*shape, 3)
        return quat_apply(quat.reshape(-1, 4), ez.reshape(-1, 3)).reshape(*shape, 3)[..., 2]

    def _local(self, body: RigidObject) -> torch.Tensor:
        return body.data.root_pos_w - self.env_origins

    def _flat(self, body: RigidObject, max_deg: float) -> torch.Tensor:
        return self._up_z(body.data.root_quat_w).clamp(-1, 1) >= math.cos(math.radians(max_deg))

    def _in_pit(self, pos_local: torch.Tensor) -> torch.Tensor:
        """(N,) bool: a slab centre inside the pit volume — horizontal gates plus BELOW the
        deck plane. A slab lying on the deck sits slab_t/2 ABOVE the plane and fails."""
        c = self.cfg
        return ((pos_local[:, 0] - self.pit_xy[:, 0]).abs() < c.pit_x_tol) \
            & ((pos_local[:, 1] - self.pit_xy[:, 1]).abs() < c.pit_y_tol) \
            & (pos_local[:, 2] < c.deck_top - c.in_pit_below)

    def floor_top(self) -> torch.Tensor:
        """(N,) pit-floor top z (local): deck_top - depth * slab_t."""
        return self.cfg.deck_top - self.depth_units * self.cfg.slab_t

    def interred(self) -> torch.Tensor:
        """(N,) bool: keystone resting flat directly ON the pit floor — in the pit, flat,
        and its centre z pinned to the bottom-course seat within `inter_z_tol`. A keystone
        laid on top of D-1 pavers is a full course (20 mm) too high and always fails."""
        c = self.cfg
        k = self._local(self.keystone)
        seat = self.floor_top() + c.slab_t / 2
        return self._in_pit(k) & self._flat(self.keystone, c.flat_max_deg) \
            & ((k[:, 2] - seat).abs() < c.inter_z_tol)

    def covered(self) -> torch.Tensor:
        """(N,) bool: at least one paver lies flat in the pit directly ABOVE the keystone
        (a full course higher, horizontally on top of it)."""
        c = self.cfg
        k = self._local(self.keystone)
        out = torch.zeros(self.env.num_envs, dtype=torch.bool, device=k.device)
        for p in self.pavers:
            pp = self._local(p)
            out |= self._in_pit(pp) & self._flat(p, c.flat_max_deg) \
                & ((pp[:, 2] - k[:, 2]) > c.cover_dz_min) \
                & ((pp[:, :2] - k[:, :2]).norm(dim=-1) < c.cover_xy_tol)
        return out

    def flush(self) -> torch.Tensor:
        """(N,) bool: the patch top is flush with the deck plane — at least one slab truly
        in the pit, and over the pit's HORIZONTAL footprint (xy gates only, any height):
        every such slab flat and the highest slab top within `flush_tol` of the deck.
        Membership by footprint (not the in-pit z gate) is what catches the overfill: an
        extra slab perched a course proud has its centre ABOVE the deck plane, so a z-gated
        membership would silently drop it from the computation and read the courses below
        as flush. A one-slab miscount is off by 20 mm; the band is +/-4 mm."""
        c = self.cfg
        slabs = [self.keystone] + list(self.pavers)
        pos = [self._local(s) for s in slabs]
        over = torch.stack([  # over the pit footprint, regardless of height
            ((p[:, 0] - self.pit_xy[:, 0]).abs() < c.pit_x_tol)
            & ((p[:, 1] - self.pit_xy[:, 1]).abs() < c.pit_y_tol) for p in pos], dim=1)
        in_pit = torch.stack([self._in_pit(p) for p in pos], dim=1)  # (N,S)
        flats = torch.stack([self._flat(s, c.flat_max_deg) for s in slabs], dim=1)
        tops = torch.stack([p[:, 2] + c.slab_t / 2 for p in pos], dim=1)
        any_in = in_pit.any(dim=1)
        all_flat = (~over | flats).all(dim=1)
        top = torch.where(over, tops, torch.full_like(tops, -1.0)).max(dim=1).values
        return any_in & all_flat & ((top - c.deck_top).abs() < c.flush_tol)

    def bar_clear(self) -> torch.Tensor:
        """(N,) bool: the bar rests flat at deck level, parked clear PAST the pit on the far
        side (its start is always on the near side)."""
        c = self.cfg
        b = self._local(self.bar)
        return ((b[:, 0] - self.pit_xy[:, 0]) > c.clear_dx) \
            & ((b[:, 2] - c.bar_rest_z).abs() < c.clear_z_tol) \
            & self._flat(self.bar, c.bar_flat_max_deg)

    def settled(self) -> torch.Tensor:
        """(N,) bool: every dynamic body |lin vel| below `settle_speed`."""
        c = self.cfg
        still = self.keystone.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        for p in self.pavers:
            still &= p.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        still &= self.bar.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return still

    def success(self) -> torch.Tensor:
        """(N,) bool: patch complete (keystone interred + covered + flush) + the screed
        pass latched + bar parked clear + everything settled."""
        return (self.interred() & self.covered() & self.flush() & self.ever_swept
                & self.bar_clear() & self.settled())

    def score(self) -> torch.Tensor:
        """(N,) float 0..1 — additive latched milestones (monotone along a correct run):
        +0.15 keystone interred, +0.35 patch complete at rest, +0.20 screed pass;
        1.0 iff success()."""
        s = torch.zeros(self.env.num_envs, device=self.env.device)
        s = s + 0.15 * self.ever_interred.float()
        s = s + 0.35 * self.ever_capped.float()
        s = s + 0.20 * self.ever_swept.float()
        return torch.where(self.success(), torch.ones_like(s), s)


# Scene-level task (robot="null"): solve.py is the teleport certificate; the intended
# embodiment (single Franka + parallel jaw) is argued in TASK.md.
register_env("simgen", lambda: EnvCfg(scene="screed_patch", robot="null"))

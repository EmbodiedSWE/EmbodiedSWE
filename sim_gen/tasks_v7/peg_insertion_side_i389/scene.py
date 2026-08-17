"""LineBoreScene — seat the correct coupler block, then line-bore the headed shaft
through pylon window -> coupler channel -> pylon window (sim_gen task
`peg_insertion_side_i389`).

Derived from maniskill/peg_insertion_side, but STRATEGICALLY different: the seed is a
single-object precision insertion — pick one peg off the ground and push it sideways
into a hole in one fixed block (one grasp, one alignment, one insertion; the hole
exists from the start and judging is a bbox at the hole). Here the through-passage
does NOT exist until the solver BUILDS it, and the insertion is a chained multi-body
assembly:

- a kinematic FIXTURE holds two upright pylons, each pierced by a square WINDOW, with
  an empty keyed POCKET between them (floor rails + the pylon inner faces);
- two free blocks stand nearby: the SILVER COUPLER with a wide square channel that
  admits the shaft, and a COPPER DECOY whose channel is narrower than the shaft
  (16 mm vs 20 mm — it can never pass; asserted in cfg). Which start slot holds
  which block is randomized, so the choice must be made by LOOKING (color/channel),
  not by position;
- the solver must first SELECT and SEAT the coupler in the pocket — only then do the
  three bores line up into a continuous passage — and then THREAD the headed shaft
  through all three gates until its red head sits flush on the entry pylon;
- the order is TOPOLOGICALLY FORCED, not latch-declared: a shaft threaded through
  the two windows first lies across the pocket at bore height, and the coupler
  (closed channel profile — it cannot be slipped over a shaft sideways) can then
  only rest ON the shaft, far above the seat band (asserted in cfg). Seat-first is
  the only physical route.

A solver needs a different PLAN (inspect and choose between two candidate parts,
place-and-key a block into a pocket, then a chained three-gate threading with a
depth/flush end condition) and different CODE STRUCTURE (two manipulated objects,
an assembly predicate over their joint state) — not different insertion parameters.

success(): the coupler is seated in the pocket (keyed bands + upright) AND the shaft
is installed: axis-aligned with the bore line, inside the y/z bore band, spanning
both pylon outer faces, head flush on its entry pylon (within `flush_tol`), and the
shaft axis passes through the SEATED coupler's channel (encirclement) — all settled.

score() is graded and latched (credit never evaporates): 0.30 coupler ever seated;
+0.20 shaft ever engaged in a window while the coupler is seated; +0.30 x max
insertion-depth fraction while seated+engaged; 1.0 iff success(). Null policy ~0
(everything spawns scattered on the ground).

Per-episode randomization (readback-verified in smoke): fixture yaw + xy jitter,
which start slot holds the coupler vs the decoy (coin flip), block slot jitter +
free yaw, shaft slot jitter + free yaw. Assets fully procedural (boxes + cylinders,
compound rigid bodies). Heavy imports (isaaclab, pxr) deferred so importing this
module — and registering the scene — stays app-free.
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


def _material(stage, path: str, static: float = 0.6, dynamic: float = 0.5):
    """One USD physics material (explicit friction + zero restitution — custom-spawner
    colliders otherwise land on engine defaults)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _collide(prim, contact_offset: float, material) -> None:
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
    """Author one box child prim (translate -> scale, authored once — the duplicate-
    xformOp trap is avoided by never re-authoring an existing prim's ops)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset, material)


def _cyl_x(stage, path: str, radius: float, length: float, center, color,
           contact_offset: float, material=None) -> None:
    """One cylinder child prim with its axis along body +x."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateAxisAttr("X")
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(length))
    seg.CreateExtentAttr([Gf.Vec3f(-length / 2, -radius, -radius),
                          Gf.Vec3f(length / 2, radius, radius)])
    UsdGeom.Xformable(seg.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in center]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset, material)


def _rigid_root(stage, prim_path: str, translation, orientation, mass: float,
                kinematic: bool = False):
    """One rigid-body root Xform with the standard physics armor (zero sleep/
    stabilization thresholds: a sleeping body silently ignores applied wrenches;
    velocity iterations 4 — the phantom-creep fix)."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    return root, pxrb


def _spawn_fixture(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The KINEMATIC fixture: deck plate, two pierced pylons (4 boxes each around a
    square window), and two pocket guide rails on the deck between them."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 20.0,
                              kinematic=True)
    mat = _material(stage, f"{prim_path}/phys_mat")
    c = cfg
    blue = (0.16, 0.30, 0.62)
    grey = (0.55, 0.57, 0.60)
    dark = (0.25, 0.26, 0.28)
    deck_x, deck_y, deck_z = c.deck
    _box(stage, f"{prim_path}/deck", (deck_x, deck_y, deck_z),
         (0.0, 0.0, deck_z / 2), blue, c.contact_offset, mat)
    half_win = c.win / 2
    pyl_top = deck_z + c.pyl_h
    win_lo = c.win_z - half_win
    win_hi = c.win_z + half_win
    for side, xc in (("n", -(c.gap / 2 + c.pyl_t / 2)), ("f", c.gap / 2 + c.pyl_t / 2)):
        col_w = c.pyl_w / 2 - half_win
        col_yc = half_win + col_w / 2
        _box(stage, f"{prim_path}/pyl_{side}_col_p", (c.pyl_t, col_w, c.pyl_h),
             (xc, +col_yc, deck_z + c.pyl_h / 2), grey, c.contact_offset, mat)
        _box(stage, f"{prim_path}/pyl_{side}_col_n", (c.pyl_t, col_w, c.pyl_h),
             (xc, -col_yc, deck_z + c.pyl_h / 2), grey, c.contact_offset, mat)
        _box(stage, f"{prim_path}/pyl_{side}_sill", (c.pyl_t, c.win, win_lo - deck_z),
             (xc, 0.0, (deck_z + win_lo) / 2), grey, c.contact_offset, mat)
        _box(stage, f"{prim_path}/pyl_{side}_lintel", (c.pyl_t, c.win, pyl_top - win_hi),
             (xc, 0.0, (win_hi + pyl_top) / 2), grey, c.contact_offset, mat)
    rail_yc = c.rail_inner_y + c.rail_t / 2
    for sgn, nm in ((+1.0, "p"), (-1.0, "n")):
        _box(stage, f"{prim_path}/rail_{nm}", (c.gap, c.rail_t, c.rail_h),
             (0.0, sgn * rail_yc, deck_z + c.rail_h / 2), dark, c.contact_offset, mat)
    return root


def _spawn_block(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One free coupler block: 4 slabs around a square through-channel along body x
    (a closed profile — it cannot be slipped over a shaft sideways). `chan` sets the
    channel side (the coupler admits the shaft; the decoy refuses it)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    mat = _material(stage, f"{prim_path}/phys_mat")
    c = cfg
    half_ch = c.chan / 2
    slab_h = c.hz / 2 - half_ch  # top/bottom slab height
    side_w = c.wy / 2 - half_ch  # side slab thickness
    _box(stage, f"{prim_path}/bottom", (c.lx, c.chan, slab_h),
         (0.0, 0.0, -(half_ch + slab_h / 2)), c.color, c.contact_offset, mat)
    _box(stage, f"{prim_path}/top", (c.lx, c.chan, slab_h),
         (0.0, 0.0, +(half_ch + slab_h / 2)), c.color, c.contact_offset, mat)
    _box(stage, f"{prim_path}/side_p", (c.lx, side_w, c.hz),
         (0.0, +(half_ch + side_w / 2), 0.0), c.color, c.contact_offset, mat)
    _box(stage, f"{prim_path}/side_n", (c.lx, side_w, c.hz),
         (0.0, -(half_ch + side_w / 2), 0.0), c.color, c.contact_offset, mat)
    return root


def _spawn_shaft(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The headed shaft: green shank cylinder along body x plus a red head disc at
    the body -x end (the head is wider than every window and channel — it can never
    pass a gate; it is the flush stop AND the graspable feature)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)
    mat = _material(stage, f"{prim_path}/phys_mat")
    c = cfg
    _cyl_x(stage, f"{prim_path}/shank", c.r, c.shank, (0.0, 0.0, 0.0),
           (0.55, 0.75, 0.25), c.contact_offset, mat)
    _cyl_x(stage, f"{prim_path}/head", c.head_r, c.head_t,
           (-(c.shank / 2 + c.head_t / 2), 0.0, 0.0),
           (0.80, 0.15, 0.12), c.contact_offset, mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazily — app-free import)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "fixture" not in _SPAWNER_CACHE:

        @configclass
        class FixtureSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_fixture)
            contact_offset: float = 0.002
            deck: tuple = (0.30, 0.24, 0.02)
            gap: float = 0.058
            pyl_t: float = 0.012
            pyl_w: float = 0.16
            pyl_h: float = 0.16
            win: float = 0.036
            win_z: float = 0.075
            rail_t: float = 0.012
            rail_h: float = 0.030
            rail_inner_y: float = 0.033

        @configclass
        class BlockSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_block)
            mass: float = 0.25
            contact_offset: float = 0.002
            lx: float = 0.050
            wy: float = 0.060
            hz: float = 0.110
            chan: float = 0.036
            color: tuple = (0.75, 0.78, 0.82)

        @configclass
        class ShaftSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_shaft)
            mass: float = 0.15
            contact_offset: float = 0.002
            r: float = 0.010
            shank: float = 0.130
            head_r: float = 0.022
            head_t: float = 0.012

        _SPAWNER_CACHE.update(fixture=FixtureSpawnerCfg, block=BlockSpawnerCfg,
                              shaft=ShaftSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class LineBoreSceneCfg(BaseCfg):
    """Config for `LineBoreScene`. Every honesty premise is asserted in
    `__post_init__`: the decoy refuses the shaft, the head passes no gate, the seated
    coupler fully covers the window projection (no path around it), a crosswise block
    cannot enter the pocket, and a pre-spanned shaft makes the seat band unreachable
    (the topological order forcer)."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    seat_x_tol: float = tunable(0.009)  # coupler center |x| (pocket slack 4 mm + margin)
    seat_y_tol: float = tunable(0.008)  # coupler center |y| (rail slack 3 mm + margin)
    seat_z_tol: float = tunable(0.010)  # coupler center z about the seat height
    seat_up_min: float = tunable(0.98)  # coupler body-z alignment with fixture z
    seat_ax_min: float = tunable(0.98)  # |coupler channel axis . fixture x|
    inst_ax_min: float = tunable(0.97)  # |shaft axis . fixture x|
    bore_y_tol: float = tunable(0.012)  # shaft center |y| in the bore line
    bore_z_lo: float = tunable(0.062)  # shaft center z window in the bore
    bore_z_hi: float = tunable(0.088)
    flush_tol: float = tunable(0.012)  # head inner face within this of the entry face
    span_margin: float = tunable(0.005)  # tip beyond the far outer face by this
    encircle_tol: float = tunable(0.012)  # shaft axis offset from the channel axis
    settle_lin: float = tunable(0.08)  # max |lin vel| at judging (m/s)
    settle_ang: float = tunable(0.80)  # max |ang vel| at judging (rad/s)
    engage_depth: float = tunable(0.010)  # window engagement: tip this far past an outer face
    depth_den: float = tunable(0.10)  # flush-approach denominator for the depth fraction

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    yaw_jitter_deg: float = tunable(10.0)  # +- fixture yaw jitter
    pos_jitter: float = tunable(0.03)  # +- fixture xy jitter
    slot_jitter: float = tunable(0.02)  # +- object slot xy jitter
    block_slot: tuple = tunable((0.22, 0.17))  # canonical block slots at (x, +-y)
    shaft_slot: tuple = tunable((-0.28, 0.0))  # canonical shaft slot

    # --- info: structure (the geometry the spawners author) ----------------------------------
    deck: tuple = info((0.30, 0.24, 0.02))
    gap: float = info(0.058)  # pylon inner-face gap (= pocket x extent)
    pyl_t: float = info(0.012)  # pylon thickness
    pyl_w: float = info(0.16)
    pyl_h: float = info(0.16)  # pylon height above the deck
    win: float = info(0.036)  # square window side
    win_z: float = info(0.075)  # window center height (= deck top + block hz/2)
    rail_t: float = info(0.012)
    rail_h: float = info(0.030)
    rail_inner_y: float = info(0.033)  # rail inner face |y| (= block wy/2 + 3 mm)
    block_lx: float = info(0.050)  # block extent along the bore (x when seated)
    block_wy: float = info(0.060)
    block_hz: float = info(0.110)
    chan: float = info(0.036)  # coupler channel side (admits the shaft)
    decoy_chan: float = info(0.016)  # decoy channel side (REFUSES the shaft)
    shaft_r: float = info(0.010)
    shank: float = info(0.130)
    head_r: float = info(0.022)
    head_t: float = info(0.012)
    block_mass: float = info(0.25)
    shaft_mass: float = info(0.15)
    contact_offset: float = info(0.002)
    jaw_span: float = info(0.080)  # Franka parallel-jaw max opening (embodiment ref)

    def __post_init__(self) -> None:
        c = self
        deck_z = c.deck[2]
        x_out = c.gap / 2 + c.pyl_t  # pylon outer face |x|
        win_lo, win_hi = c.win_z - c.win / 2, c.win_z + c.win / 2
        # -- the gates are honest --
        assert c.win / 2 >= c.shaft_r + 0.006, "windows must admit the shaft freely"
        assert c.chan / 2 >= c.shaft_r + 0.006, "coupler channel must admit the shaft"
        assert c.decoy_chan / 2 < c.shaft_r - 0.001, "decoy channel must REFUSE the shaft"
        assert 2 * c.head_r > c.win + 0.006, "the head must pass no window"
        assert 2 * c.head_r > c.chan + 0.006, "the head must pass no channel"
        # -- the seated coupler covers the whole window projection (no path around) --
        rail_slack = c.rail_inner_y - c.block_wy / 2
        x_slack = c.gap - c.block_lx
        assert 0.002 <= rail_slack <= 0.006, "rails must key the coupler in y"
        assert 0.004 <= x_slack <= 0.012, "pylons must key the coupler in x"
        assert c.block_wy / 2 >= c.win / 2 + rail_slack + 0.004, \
            "seated coupler must cover the window in y despite slack"
        assert deck_z + c.block_hz >= win_hi + 0.015 and deck_z <= win_lo - 0.015, \
            "seated coupler must cover the window in z"
        assert abs((deck_z + c.block_hz / 2) - c.win_z) < 1e-9, \
            "channel center must line up with the window center when seated"
        # -- keying: a crosswise block cannot enter the pocket --
        assert c.block_wy > c.gap + 0.001, "a 90-degree-yawed block must not fit the pocket"
        # -- the shaft reaches: flush head => tip beyond the far face --
        assert c.shank >= 2 * x_out + 0.030, "shank must span both pylons with margin"
        assert c.shank - 2 * x_out >= c.span_margin + 0.020, \
            "flush head must leave the tip proud of the far face"
        # -- rubric bands sit on physics --
        rest_lo = win_lo + c.shaft_r - 0.003  # shaft resting on the sills
        assert c.bore_z_lo <= rest_lo and c.bore_z_hi >= c.win_z + 0.005, \
            "bore z window must accept every physical in-bore rest"
        assert c.chan / 2 - c.shaft_r <= c.encircle_tol, \
            "every physically-in-channel axis offset must be inside encircle_tol"
        # -- ORDER FORCER: with a spanning shaft in place the seat band is unreachable --
        shaft_rest_axis = win_lo + c.shaft_r  # spanning shaft rests on the sills
        block_on_shaft = shaft_rest_axis + c.shaft_r + c.block_hz / 2  # rest ON the shaft
        assert block_on_shaft > (deck_z + c.block_hz / 2) + c.seat_z_tol + 0.02, \
            "a coupler dropped onto a pre-spanned shaft must rest far above the seat band"
        # -- embodiment: everything fits the jaw --
        assert c.block_wy <= c.jaw_span - 0.015, "coupler must fit the Franka jaw"
        assert 2 * c.shaft_r <= c.jaw_span - 0.030, "shank must fit the Franka jaw"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("line_bore")
class LineBoreScene(BaseScene):
    cfg: LineBoreSceneCfg

    def __init__(self, cfg: LineBoreSceneCfg | None = None) -> None:
        super().__init__(cfg or LineBoreSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "fixture": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Fixture",
                spawn=sp["fixture"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    contact_offset=c.contact_offset, deck=c.deck, gap=c.gap,
                    pyl_t=c.pyl_t, pyl_w=c.pyl_w, pyl_h=c.pyl_h, win=c.win,
                    win_z=c.win_z, rail_t=c.rail_t, rail_h=c.rail_h,
                    rail_inner_y=c.rail_inner_y),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "coupler": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Coupler",
                spawn=sp["block"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.block_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.block_mass, contact_offset=c.contact_offset,
                    lx=c.block_lx, wy=c.block_wy, hz=c.block_hz, chan=c.chan,
                    color=(0.75, 0.78, 0.82)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.block_slot[0], c.block_slot[1], c.block_hz / 2 + 0.003)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=sp["block"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.block_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.block_mass, contact_offset=c.contact_offset,
                    lx=c.block_lx, wy=c.block_wy, hz=c.block_hz, chan=c.decoy_chan,
                    color=(0.72, 0.45, 0.20)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.block_slot[0], -c.block_slot[1], c.block_hz / 2 + 0.003)),
            ),
            "shaft": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shaft",
                spawn=sp["shaft"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.shaft_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.shaft_mass, contact_offset=c.contact_offset,
                    r=c.shaft_r, shank=c.shank, head_r=c.head_r, head_t=c.head_t),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.shaft_slot[0], c.shaft_slot[1], c.head_r + 0.002)),
            ),
        }
        return out

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
                # external-wrench plant recipe: without this the solve/smoke force
                # servos are under-applied across TGS iterations
                "enable_external_forces_every_iteration": True,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n, dev = env.num_envs, env.device
        self.fixture: RigidObject = env.iscene["fixture"]
        self.coupler: RigidObject = env.iscene["coupler"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.shaft: RigidObject = env.iscene["shaft"]
        self.env_origins = env.iscene.env_origins
        # +1: coupler starts at the +y slot; -1: at the -y slot (decoy at the other)
        self.slot_sign = torch.ones(n, device=dev)
        # progress latches (post_step)
        self.seat_latch = torch.zeros(n, device=dev)
        self.engage_latch = torch.zeros(n, device=dev)
        self.depth_latch = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: fixture at the workspace with yaw + xy jitter; the coupler
        and the decoy swap start slots on a coin flip (burn one draw first — the
        first post-seed draw is degenerate across seeds); blocks stand upright with
        free yaw + slot jitter; shaft lies flat with free yaw + slot jitter; latches
        zeroed. All object slots are FIXTURE-frame canonical, mapped through the
        sampled fixture pose so the layout tracks the jitter."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        _ = torch.rand(m, device=dev)  # burn the degenerate first post-seed draw
        self.slot_sign[env_ids] = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        ss = self.slot_sign[env_ids]

        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_jitter_deg) / 2
        cy, sy = torch.cos(2 * half), torch.sin(2 * half)
        fx = (torch.rand(m, device=dev) * 2 - 1) * c.pos_jitter
        fy = (torch.rand(m, device=dev) * 2 - 1) * c.pos_jitter
        fst = torch.zeros(m, 13, device=dev)
        fst[:, 0] = fx
        fst[:, 1] = fy
        fst[:, 3] = torch.cos(half)
        fst[:, 6] = torch.sin(half)
        fst[:, 0:3] += origin
        self.fixture.write_root_state_to_sim(fst, env_ids)

        def to_world_xy(px: torch.Tensor, py: torch.Tensor):
            return fx + cy * px - sy * py, fy + sy * px + cy * py

        def qz(yaw_half: torch.Tensor) -> tuple:
            return torch.cos(yaw_half), torch.sin(yaw_half)

        # blocks: standing upright at their (possibly swapped) slots
        for body, sign in ((self.coupler, ss), (self.decoy, -ss)):
            px = c.block_slot[0] + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
            py = sign * c.block_slot[1] + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
            wx, wy = to_world_xy(px, py)
            byaw = half + (torch.rand(m, device=dev) * 2 - 1) * math.pi
            qw, qzz = qz(byaw)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = wx
            st[:, 1] = wy
            st[:, 2] = c.block_hz / 2 + 0.003
            st[:, 3] = qw
            st[:, 6] = qzz
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # shaft: lying flat at its slot, free yaw
        px = c.shaft_slot[0] + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
        py = c.shaft_slot[1] + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
        wx, wy = to_world_xy(px, py)
        syaw = half + (torch.rand(m, device=dev) * 2 - 1) * math.pi
        qw, qzz = qz(syaw)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = wx
        st[:, 1] = wy
        st[:, 2] = c.head_r + 0.002
        st[:, 3] = qw
        st[:, 6] = qzz
        st[:, 0:3] += origin
        self.shaft.write_root_state_to_sim(st, env_ids)

        for latch in (self.seat_latch, self.engage_latch, self.depth_latch):
            latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def _bodies(self) -> dict[str, RigidObject]:
        return {"fixture": self.fixture, "coupler": self.coupler,
                "decoy": self.decoy, "shaft": self.shaft}

    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in self._bodies().items()},
            "slot_sign": self.slot_sign[env_ids].clone(),
            "latches": torch.stack([self.seat_latch[env_ids],
                                    self.engage_latch[env_ids],
                                    self.depth_latch[env_ids]], dim=1),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm, b in self._bodies().items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        self.slot_sign[env_ids] = state["slot_sign"]
        lt = state["latches"]
        self.seat_latch[env_ids] = lt[:, 0]
        self.engage_latch[env_ids] = lt[:, 1]
        self.depth_latch[env_ids] = lt[:, 2]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A blue-decked FIXTURE stands on the floor: two upright GREY PYLONS, each "
            f"pierced by a square WINDOW ({c.win * 1000:.0f} mm on a side, center "
            f"{c.win_z * 1000:.0f} mm up), face each other across an empty POCKET — a "
            "slot on the deck between the pylon inner faces, guided by two low dark "
            "rails. Nearby stand two look-alike blocks, each with a square channel "
            f"bored through it: the SILVER COUPLER (channel {c.chan * 1000:.0f} mm — "
            f"wide) and the COPPER DECOY (channel {c.decoy_chan * 1000:.0f} mm — "
            "visibly narrower). Which block stands at which start spot is randomized: "
            "identify them by COLOR and by the size of the channel opening, never by "
            "position. A SHAFT lies on the floor on the other side: a green "
            f"{2 * c.shaft_r * 1000:.0f} mm-thick shank with a wider RED HEAD disc at "
            "one end. The head is wider than every opening, so it can pass no window "
            "and no channel; the decoy's channel is narrower than the shank, so the "
            "shaft can NEVER pass through the decoy.\n"
            "Goal, in the only order physics allows: FIRST stand the silver coupler "
            "down inside the pocket between the pylons (the rails and pylon faces key "
            "it square; when seated, its channel lines up with the two windows into "
            "one continuous bore). THEN thread the shaft, head trailing, through the "
            "near pylon's window, straight through the seated coupler's channel, and "
            "out through the far pylon's window, pushing until the red head sits flush "
            "against the pylon it entered (either entry side works). Finish with "
            "everything at rest: coupler seated in the pocket, shaft spanning both "
            "pylons through the coupler's channel, head flush. A shaft threaded "
            "through the windows without the coupler seated, threaded short of flush, "
            "resting over the pylon tops, or stopped against the copper decoy counts "
            "for nothing at the end; seating the decoy dead-ends the task until it is "
            "removed. Note the pocket is exactly one block wide: a shaft threaded "
            "first will lie across the pocket and the coupler can then only rest on "
            "top of it — seat the coupler first."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Seat the silver wide-channel coupler block in the pocket between the two "
            "pylons, then thread the green shaft head-trailing through the near "
            "window, the coupler's channel, and the far window until its red head "
            "sits flush against the entry pylon and everything rests. The copper "
            "narrow-channel block is a decoy the shaft cannot pass; threading the "
            "windows without the coupler seated counts for nothing."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _fix_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.fixture.data.root_pos_w, self.fixture.data.root_quat_w

    def to_canon(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> fixture canonical frame."""
        from isaaclab.utils.math import quat_apply_inverse

        pos, quat = self._fix_pose()
        return quat_apply_inverse(quat, p_w - pos)

    def canon_to_world(self, canon: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        pos, quat = self._fix_pose()
        return pos + quat_apply(quat, canon)

    def _axis_canon(self, body: RigidObject, ax) -> torch.Tensor:
        """(N,3): a body-frame axis expressed in the fixture canonical frame."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        _pos, fq = self._fix_pose()
        v = torch.tensor(ax, device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply_inverse(fq, quat_apply(body.data.root_quat_w, v))

    # ----- predicates (fixture canonical frame) -----------------------------------------------
    def seated(self) -> torch.Tensor:
        """(N,) bool: the COUPLER is keyed into the pocket — center inside the seat
        bands, upright, channel axis along the bore."""
        c = self.cfg
        p = self.to_canon(self.coupler.data.root_pos_w)
        up = self._axis_canon(self.coupler, (0.0, 0.0, 1.0))
        ax = self._axis_canon(self.coupler, (1.0, 0.0, 0.0))
        seat_z = c.deck[2] + c.block_hz / 2
        return (p[:, 0].abs() <= c.seat_x_tol) & (p[:, 1].abs() <= c.seat_y_tol) \
            & ((p[:, 2] - seat_z).abs() <= c.seat_z_tol) \
            & (up[:, 2] >= c.seat_up_min) & (ax[:, 0].abs() >= c.seat_ax_min)

    def shaft_metrics(self) -> dict[str, torch.Tensor]:
        """Canonical-frame shaft geometry: alignment, end/tip/head-face coordinates,
        in-bore bands, engagement, spanning, flush, encirclement of the coupler."""
        c = self.cfg
        x_out = c.gap / 2 + c.pyl_t
        p = self.to_canon(self.shaft.data.root_pos_w)
        ax = self._axis_canon(self.shaft, (1.0, 0.0, 0.0))
        half = c.shank / 2
        tip = p + ax * half            # body +x end (away from the head)
        head_face = p - ax * half      # head inner face (body -x end of the shank)
        aligned = ax[:, 0].abs() >= c.inst_ax_min
        in_band = (p[:, 1].abs() <= c.bore_y_tol) \
            & (p[:, 2] >= c.bore_z_lo) & (p[:, 2] <= c.bore_z_hi)
        e_lo = torch.minimum(tip[:, 0], head_face[:, 0])
        e_hi = torch.maximum(tip[:, 0], head_face[:, 0])
        gate = x_out - c.engage_depth
        engaged = aligned & in_band \
            & (((e_lo <= -gate) & (e_hi >= -gate)) | ((e_lo <= gate) & (e_hi >= gate)))
        # entry side = the side the head is on
        head_neg = head_face[:, 0] < tip[:, 0]
        x_flush = torch.where(head_neg, -x_out, x_out)
        remaining = (head_face[:, 0] - x_flush).abs()
        flush = remaining <= c.flush_tol
        spanning = torch.where(head_neg,
                               tip[:, 0] >= x_out + c.span_margin,
                               tip[:, 0] <= -(x_out + c.span_margin))
        # encirclement: shaft axis offset from the seated coupler's channel axis,
        # evaluated at the coupler's x station
        b = self.to_canon(self.coupler.data.root_pos_w)
        t = (b[:, 0] - p[:, 0]) / ax[:, 0].clamp(min=1e-6).where(
            ax[:, 0] >= 0, ax[:, 0].clamp(max=-1e-6))
        py = p[:, 1] + ax[:, 1] * t
        pz = p[:, 2] + ax[:, 2] * t
        encircle = aligned & ((py - b[:, 1]).abs() <= c.encircle_tol) \
            & ((pz - b[:, 2]).abs() <= c.encircle_tol)
        return {"aligned": aligned, "in_band": in_band, "engaged": engaged,
                "remaining": remaining, "flush": flush, "spanning": spanning,
                "encircle": encircle, "tip": tip, "head_face": head_face, "p": p}

    def installed(self) -> torch.Tensor:
        """(N,) bool: shaft fully installed along the bore line (geometry only —
        seating of the coupler and settling judged separately)."""
        m = self.shaft_metrics()
        return m["aligned"] & m["in_band"] & m["spanning"] & m["flush"]

    def settled(self) -> torch.Tensor:
        """(N,) bool: shaft AND coupler at rest (thresholds above the GPU phantom-
        velocity band)."""
        c = self.cfg
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for b in (self.shaft, self.coupler):
            ok &= (b.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
                & (b.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
        return ok

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch (1) the coupler ever seated, (2) the shaft ever engaged in a window
        WHILE the coupler is seated, (3) the max flush-approach fraction while
        seated+engaged. Gating on the live seated() is what makes threading without
        the coupler worthless."""
        c = self.cfg
        seat_now = self.seated()
        m = self.shaft_metrics()
        self.seat_latch = torch.maximum(self.seat_latch, seat_now.float())
        eng = (seat_now & m["engaged"]).float()
        self.engage_latch = torch.maximum(self.engage_latch, eng)
        frac = (1.0 - m["remaining"] / c.depth_den).clamp(0.0, 1.0)
        self.depth_latch = torch.maximum(self.depth_latch,
                                         frac * (seat_now & m["engaged"]).float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: coupler seated in the pocket AND shaft installed along the bore
        AND the shaft axis passes through the seated coupler's channel AND both are
        settled. The interaction chain is physically necessary: the head passes no
        gate (so flush = full threading), the decoy refuses the shank, and a shaft
        spanned without the coupler leaves seated() False."""
        m = self.shaft_metrics()
        return self.seated() & self.installed() & m["encircle"] & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1], latched: 0.30 coupler ever seated; +0.20 shaft ever
        engaged in a window while seated; +0.30 x max flush-approach fraction while
        seated+engaged; 1.0 iff success(). Null policy ~0 (everything spawns
        scattered on the floor)."""
        base = 0.30 * self.seat_latch + 0.20 * self.engage_latch \
            + 0.30 * self.depth_latch
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="line_bore", robot="null", env_spacing=4.0))

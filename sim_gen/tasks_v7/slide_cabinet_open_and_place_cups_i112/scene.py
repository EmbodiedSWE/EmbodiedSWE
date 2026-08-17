"""ShutterCabinetScene — sort three cups into a two-bay top-loading cabinet through a
single captive BYPASS SHUTTER that can never uncover both openings at once, then park
the shutter covering the blue opening (sim_gen task `slide_cabinet_open_and_place_cups_i112`).

Derived from rlbench/slide_cabinet_open_and_place_cups, but STRATEGICALLY different:
the seed slides one cabinet door OPEN once — getting the panel out of the way is the
whole articulation act — and then places interchangeable cups into the single revealed
volume, judged by a recorded trajectory. Here the sliding panel can NEVER be gotten out
of the way: it is a captive shutter on a short prismatic track over TWO top-loading
openings, wide enough that (proved by geometry, asserted in `__post_init__`) whenever
one opening has a cup-sized gap the other is sealed below cup width. Access is
MULTIPLEXED: every deposit into one bay seals the other, so sorting the color-matched
cups (two RED, one BLUE) into their bays forces the solver to interleave shutter
shuttling with deposits, and success additionally requires the shutter parked back
FULLY COVERING the blue opening — which physically forces the blue deposit to happen
before the final parking move (a cup dropped on a covered opening rests on the
shutter). A solver needs a different PLAN (schedule deposits around a mutually
exclusive access resource and restore the mechanism's end state; the seed's
open-then-fill plan is impossible — there is no state where the cabinet is simply
"open") and different code STRUCTURE (per-bay color membership in a fixed cabinet
frame + a shutter-interval coverage predicate, not door-displacement + place-in-one-
volume).

success(): both RED cups upright on the RED bay floor, the BLUE cup upright on the
BLUE bay floor (cabinet-frame membership windows whose z band rejects cups resting on
the top plate, on the shutter, or stacked on another cup; an uprightness cone rejects
cups lying on their side), the shutter parked fully covering the blue opening, and
everything PERSISTENTLY still (stillness counter-latch — `settle_steps_min`
consecutive quiet steps).

score(), latched (credit never evaporates): 0.15 per cup ever seated in its own color
bay (0.45) + 0.15 once all three are seated simultaneously + 0.15 once all-seated
coincides with the shutter covering blue; 1.0 iff success(). Null policy scores ~0
(the shutter may even START covering blue: the coverage term is gated on all-seated,
so an untouched scene earns nothing).

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - case (KINEMATIC): floor-standing two-bay cabinet, origin at the center of the top
    plate's TOP surface. Top plate with two square openings (13 cm along the track x
    14 cm across it), outer walls, center divider, interior floor, one colored
    identification pad per bay floor (RED / BLUE). Never teleported (a joint's body0
    anchor is world-fixed; moving the case would tear the track) — the world anchor
    of the mechanism.
  - shutter (dynamic): one plate 26 cm along the track (wider than one opening plus
    the anti-sneak margin) with a yellow grip post, riding a spawn-authored prismatic
    joint (hard stops +/- `stroke`); it hovers 1.5 mm above the top plate (the joint
    is the track — no rubbing). At either stop it fully covers one opening and fully
    exposes a 10.5 cm strip of the other.
  - cups (x3, dynamic): solid cylinders, two RED and one BLUE, identical size; color
    is identity.
Friction materials are bound explicitly everywhere (the default-material trap);
contact offsets are explicit so the membership z bands stay real.

Per-episode randomization (readback-verified in smoke): the shutter's starting track
position (uniform over the whole stroke — episodes start blue-open, red-open, or
both-sealed, changing the required plan) and the cups' floor slots (3 of 4 slots,
shuffled) + xy jitter.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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
    """One USD physics material (explicit binding — the default-material ~0.5 trap)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, size, center, color, contact_offset: float, material=None) -> None:
    """Author one box child prim (translate -> scale, authored once — idempotent per
    prim, the duplicate-xformOp trap)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_case(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC two-bay cabinet; body origin at the CENTER of the top plate's TOP
    surface (the shutter joint anchors read from here)."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(20.0)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    hx, hy, h = cfg.half_x, cfg.half_y, cfg.height
    ax, ay, ac = cfg.ap_x, cfg.ap_y, cfg.ap_c  # aperture x-extent, y-extent, center |y|
    tp = cfg.plate_t
    grey = (0.42, 0.42, 0.46)
    dark = (0.30, 0.30, 0.34)
    # --- top plate: 5 strips framing the two square openings (plate top = z 0) ---
    fx = (hx - ax / 2) / 2 + ax / 2  # front/back strip x center
    _box(stage, f"{prim_path}/plate_front", (hx - ax / 2, 2 * hy, tp), (-fx, 0.0, -tp / 2),
         grey, co, material=mat)
    _box(stage, f"{prim_path}/plate_back", (hx - ax / 2, 2 * hy, tp), (fx, 0.0, -tp / 2),
         grey, co, material=mat)
    yo = (hy - (ac + ay / 2)) / 2 + ac + ay / 2  # outer y-strip center
    _box(stage, f"{prim_path}/plate_left", (ax, hy - (ac + ay / 2), tp), (0.0, -yo, -tp / 2),
         grey, co, material=mat)
    _box(stage, f"{prim_path}/plate_right", (ax, hy - (ac + ay / 2), tp), (0.0, yo, -tp / 2),
         grey, co, material=mat)
    _box(stage, f"{prim_path}/plate_mid", (ax, 2 * (ac - ay / 2), tp), (0.0, 0.0, -tp / 2),
         grey, co, material=mat)
    # --- walls (top plate bottom down to the ground), divider, interior floor ---
    wh = h - tp
    wz = -tp - wh / 2
    wt = cfg.wall_t
    _box(stage, f"{prim_path}/wall_xn", (wt, 2 * hy, wh), (-(hx - wt / 2), 0.0, wz),
         dark, co, material=mat)
    _box(stage, f"{prim_path}/wall_xp", (wt, 2 * hy, wh), (hx - wt / 2, 0.0, wz),
         dark, co, material=mat)
    _box(stage, f"{prim_path}/wall_yn", (2 * hx - 2 * wt, wt, wh), (0.0, -(hy - wt / 2), wz),
         dark, co, material=mat)
    _box(stage, f"{prim_path}/wall_yp", (2 * hx - 2 * wt, wt, wh), (0.0, hy - wt / 2, wz),
         dark, co, material=mat)
    _box(stage, f"{prim_path}/divider", (2 * hx - 2 * wt, wt, wh), (0.0, 0.0, wz),
         dark, co, material=mat)
    ft = cfg.floor_t
    _box(stage, f"{prim_path}/floor", (2 * hx - 2 * wt, 2 * hy - 2 * wt, ft),
         (0.0, 0.0, cfg.floor_z - ft / 2), grey, co, material=mat)
    # --- colored identification pads (RED bay at -y, BLUE bay at +y) ---
    pt = cfg.pad_t
    _box(stage, f"{prim_path}/pad_red", (cfg.pad_x, cfg.pad_y, pt),
         (0.0, -ac, cfg.floor_z + pt / 2), (0.80, 0.12, 0.12), co, material=mat)
    _box(stage, f"{prim_path}/pad_blue", (cfg.pad_x, cfg.pad_y, pt),
         (0.0, ac, cfg.floor_z + pt / 2), (0.12, 0.25, 0.85), co, material=mat)
    return root


def _spawn_shutter(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The bypass shutter, ONE dynamic rigid body riding a spawn-authored prismatic
    joint (axis Y, hard stops +/- stroke) against the sibling Case. It hovers 1.5 mm
    above the top plate — the joint is the track, nothing rubs. Mass/CoM/inertia are
    AUTHORED. The joint is authored IN THE SPAWNER so it exists BEFORE the physics
    parse (a joint authored in bind() comes too late and the body is baked free)."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.004))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.0016, 0.0010, 0.0025))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(cfg.lin_damping))
    pxrb.CreateAngularDampingAttr(2.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    _box(stage, f"{prim_path}/plate", (cfg.plate_x, cfg.plate_y, cfg.plate_t),
         (0.0, 0.0, 0.0), (0.16, 0.16, 0.18), co, material=mat)
    _box(stage, f"{prim_path}/handle", (0.022, 0.032, 0.05),
         (0.0, 0.0, cfg.plate_t / 2 + 0.025), (0.92, 0.78, 0.10), co, material=mat)

    # Prismatic track to the sibling Case (pair collision FILTERED by the joint —
    # the shutter hovers and never needs case contact; cups are separate bodies and
    # DO collide with both). Both anchors at the shutter's authored position.
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.PrismaticJoint.Define(stage, f"{prim_path}/track")
    j.CreateBody0Rel().SetTargets([f"{base}/Case"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateCollisionEnabledAttr(False)
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(cfg.hover_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-float(cfg.stroke))
    j.CreateUpperLimitAttr(float(cfg.stroke))
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazy: module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "case" not in _SPAWNER_CACHE:

        @configclass
        class CaseSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_case)
            half_x: float = 0.115
            half_y: float = 0.24
            height: float = 0.16
            plate_t: float = 0.012
            wall_t: float = 0.010
            floor_t: float = 0.012
            floor_z: float = -0.115
            pad_t: float = 0.003
            pad_x: float = 0.17
            pad_y: float = 0.16
            ap_x: float = 0.14
            ap_y: float = 0.13
            ap_c: float = 0.115
            mu_static: float = 0.6
            mu_dynamic: float = 0.5
            contact_offset: float = 0.0015

        @configclass
        class ShutterSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_shutter)
            plate_x: float = 0.20
            plate_y: float = 0.26
            plate_t: float = 0.012
            hover_z: float = 0.0075
            stroke: float = 0.055
            mass: float = 0.35
            lin_damping: float = 4.0
            mu_static: float = 0.6
            mu_dynamic: float = 0.5
            contact_offset: float = 0.0015

        _SPAWNER_CACHE.update(case=CaseSpawnerCfg, shutter=ShutterSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ShutterCabinetSceneCfg(BaseCfg):
    """Config for `ShutterCabinetScene`. `__post_init__` asserts the strategic honesty
    invariants: the shutter geometry makes cup-sized access to both openings mutually
    exclusive over the WHOLE track; the parked shutter fully covers the blue opening
    with slack; the membership z band accepts a floor-resting cup and rejects
    plate-top, shutter-top and stacked cups; two cups fit one bay and every grasp
    fits a parallel jaw."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    bay_x_tol: float = tunable(0.075)  # cup center |x| inside a bay (cabinet frame)
    bay_y_lo: float = tunable(0.034)  # cup center side*y band inside a bay ...
    bay_y_hi: float = tunable(0.200)  # ... (divider face + r  to  outer wall face - r)
    bay_z_tol: float = tunable(0.021)  # |cup center z - floor rest z| band (rejects
    # plate-top (+0.11 off), shutter-top (+0.13 off) and stacked (+0.065 off) cups)
    upright_max_deg: float = tunable(30.0)  # cup axis within this of world-up
    cover_c_min: float = tunable(0.048)  # shutter joint pos >= this fully seals the
    # blue opening (full geometric coverage needs 0.050; at 0.048 the residual slit is
    # 2 mm << cup — physically sealed; the stop sits at 0.055)
    settle_lin: float = tunable(0.05)  # max cup |lin vel| when judging (m/s)
    settle_ang: float = tunable(1.0)  # max cup |ang vel| when judging (rad/s)
    settle_shutter: float = tunable(0.02)  # max shutter |lin vel| counted as parked (m/s)
    settle_steps_min: int = tunable(30)  # stillness must PERSIST this many steps (0.25 s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    c0_range: float = tunable(0.054)  # shutter start joint pos ~ U(-r, +r) (full stroke)
    slot_x: float = tunable(0.30)  # cup staging row x (cups start on the floor here)
    slot_ys: tuple = tunable((-0.27, -0.09, 0.09, 0.27))  # 4 slots; 3 used, shuffled
    slot_jitter: float = tunable(0.03)  # uniform +/- xy jitter per cup at reset (m)

    # --- info: structure ---------------------------------------------------------------------
    case_pos: tuple = info((0.0, 0.0))  # cabinet origin xy in the env (NEVER teleported)
    case_h: float = info(0.16)  # top plate TOP height above the floor = case origin z
    half_x: float = info(0.115)  # cabinet footprint half-depth (x)
    half_y: float = info(0.24)  # cabinet footprint half-width (y, along the track)
    plate_t: float = info(0.012)
    wall_t: float = info(0.010)
    floor_z: float = info(-0.115)  # interior floor TOP, cabinet frame
    pad_t: float = info(0.003)
    ap_x: float = info(0.14)  # opening extent across the track (x)
    ap_y: float = info(0.13)  # opening extent along the track (y)
    ap_c: float = info(0.115)  # opening center |y| (RED bay -y, BLUE bay +y)
    shutter_y: float = info(0.26)  # shutter plate extent along the track
    shutter_x: float = info(0.20)
    shutter_t: float = info(0.012)
    shutter_hover: float = info(0.0075)  # shutter plate center above plate top (z)
    stroke: float = info(0.055)  # prismatic hard stops at +/- stroke
    shutter_mass: float = info(0.35)
    shutter_damping: float = info(4.0)
    cup_r: float = info(0.028)
    cup_h: float = info(0.065)
    cup_mass: float = info(0.06)
    cup_names: tuple = info(("red_a", "red_b", "blue"))
    cup_sides: tuple = info((-1.0, -1.0, +1.0))  # target bay per cup (RED -y, BLUE +y)
    cup_colors: tuple = info(((0.85, 0.13, 0.13), (0.85, 0.13, 0.13), (0.13, 0.28, 0.88)))
    cup_pass: float = info(0.060)  # min contiguous gap a cup can pass (2r + 4 mm)
    mu_static: float = info(0.6)
    mu_dynamic: float = info(0.5)
    contact_offset: float = info(0.0015)

    # Derived (filled in __post_init__).
    upright_cos: float = field(default=None, init=False)
    cup_rest_z: float = field(default=None, init=False)  # cup center on a pad, cabinet frame

    def __post_init__(self) -> None:
        self.upright_cos = math.cos(math.radians(self.upright_max_deg))
        self.cup_rest_z = self.floor_z + self.pad_t + self.cup_h / 2

        # -- mutual exclusion over the WHOLE track: whenever one opening has a
        #    cup-passable contiguous gap, the other's largest gap is below cup width.
        def gap(c: float, side: float) -> float:
            lo, hi = side * self.ap_c - self.ap_y / 2, side * self.ap_c + self.ap_y / 2
            pl, pr = c - self.shutter_y / 2, c + self.shutter_y / 2
            cl, cr = max(lo, pl), min(hi, pr)
            if cr <= cl:
                return hi - lo
            return max(cl - lo, hi - cr)

        for i in range(221):
            c = -self.stroke + (2 * self.stroke) * i / 220
            assert not (gap(c, -1.0) >= self.cup_pass and gap(c, +1.0) >= self.cup_pass), \
                f"mutual exclusion violated at shutter c={c:+.4f}"
        # -- the parked shutter fully covers the blue opening with slack, and still
        #    leaves the red opening fully open (the final state is reachable).
        assert self.stroke - (self.ap_c + self.ap_y / 2 - self.shutter_y / 2) >= 0.004, \
            "shutter at +stop must fully cover the blue opening with >= 4 mm slack"
        assert gap(self.stroke, -1.0) >= self.ap_y - 0.030, \
            "red opening must be (nearly) fully open at +stop"
        assert self.cover_c_min <= self.stroke - 0.004, "cover band must include the stop"
        # residual slit at the cover_c_min edge stays far below cup width
        assert gap(self.cover_c_min, +1.0) <= 0.006, "cover band edge must be sealed vs a cup"
        # -- z band honesty: accepts floor rest (pad AND bare floor), rejects everything above
        assert abs((self.floor_z + self.cup_h / 2) - self.cup_rest_z) < self.bay_z_tol, \
            "a cup off the pad on the bare floor must still be accepted"
        for wrong in (self.cup_h / 2,  # on the top plate
                      self.shutter_hover + self.shutter_t / 2 + self.cup_h / 2,  # on shutter
                      self.cup_rest_z + self.cup_h):  # stacked on another cup
            assert abs(wrong - self.cup_rest_z) > self.bay_z_tol + 0.005, \
                f"z band must reject resting height {wrong:+.4f}"
        # -- a lying cup inside the bay passes the z band and MUST be caught by uprightness
        assert abs((self.floor_z + self.cup_r) - self.cup_rest_z) < self.bay_z_tol, \
            "lying cup passes the z band -> the uprightness clause is load-bearing"
        # -- drops fit: two cups through one opening at x offsets +/-0.031
        assert 0.031 + self.cup_r <= self.ap_x / 2 - 0.008, "offset drop must clear the opening"
        assert 2 * 0.031 >= 2 * self.cup_r + 0.005, "the two red drop points must clear each other"
        # -- interior fits and jaw fits
        assert self.bay_y_hi - self.bay_y_lo >= 4 * self.cup_r, "bay must hold two cups"
        assert 2 * self.cup_r <= 0.075, "cup must fit a parallel jaw (~80 mm)"
        assert self.c0_range < self.stroke, "reset must not write the shutter into a stop"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("shutter_cabinet")
class ShutterCabinetScene(BaseScene):
    cfg: ShutterCabinetSceneCfg

    def __init__(self, cfg: ShutterCabinetSceneCfg | None = None) -> None:
        super().__init__(cfg or ShutterCabinetSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        px, py = c.case_pos

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_static, dynamic_friction=c.mu_dynamic,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "case": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Case",
                spawn=spawners["case"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    half_x=c.half_x, half_y=c.half_y, height=c.case_h,
                    plate_t=c.plate_t, wall_t=c.wall_t, floor_z=c.floor_z,
                    pad_t=c.pad_t, ap_x=c.ap_x, ap_y=c.ap_y, ap_c=c.ap_c,
                    mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, c.case_h)),
            ),
            "shutter": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shutter",
                spawn=spawners["shutter"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.shutter_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    plate_x=c.shutter_x, plate_y=c.shutter_y, plate_t=c.shutter_t,
                    hover_z=c.shutter_hover, stroke=c.stroke, mass=c.shutter_mass,
                    lin_damping=c.shutter_damping,
                    mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(px, py, c.case_h + c.shutter_hover)),
            ),
        }
        for i, nm in enumerate(c.cup_names):
            out[f"cup_{nm}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cup_" + nm,
                spawn=sim_utils.CylinderCfg(
                    radius=c.cup_r, height=c.cup_h, axis="Z",
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.10,
                        disable_gravity=False),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cup_mass),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_static, dynamic_friction=c.mu_dynamic,
                        restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.cup_colors[i]),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_x, c.slot_ys[i], c.cup_h / 2 + 0.003)),
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
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        self.case: RigidObject = env.iscene["case"]
        self.shutter: RigidObject = env.iscene["shutter"]
        self.cups: dict[str, RigidObject] = {
            nm: env.iscene[f"cup_{nm}"] for nm in c.cup_names}
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # cabinet origin, world (constant: the case is NEVER teleported)
        self.case_origin = self.env_origins + torch.tensor(
            [c.case_pos[0], c.case_pos[1], c.case_h], device=dev)
        self.seat_latch = torch.zeros(n, 3, device=dev)  # cup i ever seated in own bay
        self.all_latch = torch.zeros(n, device=dev)  # all three seated at once
        self.covered_latch = torch.zeros(n, device=dev)  # all seated & shutter covers blue
        self.still_count = torch.zeros(n, device=dev)  # consecutive still steps

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: shutter written to a uniform random track position (episodes
        start blue-open, red-open or both-sealed — the required plan changes), cups on
        3 of 4 shuffled floor slots with jitter; latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.case_origin[env_ids]

        # --- shutter: random joint position along the track ---
        c0 = (torch.rand(m, device=dev) * 2 - 1) * c.c0_range
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = origin
        st[:, 1] += c0
        st[:, 2] += c.shutter_hover
        st[:, 3] = 1.0
        self.shutter.write_root_state_to_sim(st, env_ids)

        # --- cups: 3 of the 4 slots, shuffled, + jitter, upright on the floor ---
        perm = torch.rand(m, len(c.slot_ys), device=dev).argsort(dim=1)  # (m, 4)
        ys = torch.tensor(c.slot_ys, device=dev)
        for i, nm in enumerate(c.cup_names):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = c.case_pos[0] + c.slot_x
            st[:, 1] = c.case_pos[1] + ys[perm[:, i]]
            st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            st[:, 2] = c.cup_h / 2 + 0.003
            st[:, 0:3] += self.env_origins[env_ids]
            st[:, 3] = 1.0
            self.cups[nm].write_root_state_to_sim(st, env_ids)

        self.seat_latch[env_ids] = 0.0
        self.all_latch[env_ids] = 0.0
        self.covered_latch[env_ids] = 0.0
        self.still_count[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "shutter": self.shutter.data.root_state_w[env_ids].clone(),
            "cups": {nm: b.data.root_state_w[env_ids].clone()
                     for nm, b in self.cups.items()},
            "seat_latch": self.seat_latch[env_ids].clone(),
            "all_latch": self.all_latch[env_ids].clone(),
            "covered_latch": self.covered_latch[env_ids].clone(),
            "still_count": self.still_count[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.shutter.write_root_state_to_sim(state["shutter"], env_ids)
        for nm, b in self.cups.items():
            b.write_root_state_to_sim(state["cups"][nm], env_ids)
        self.seat_latch[env_ids] = state["seat_latch"]
        self.all_latch[env_ids] = state["all_latch"]
        self.covered_latch[env_ids] = state["covered_latch"]
        self.still_count[env_ids] = state["still_count"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A low two-bay cabinet ({2 * c.half_y * 100:.0f} cm wide, "
            f"{2 * c.half_x * 100:.0f} cm deep, {c.case_h * 100:.0f} cm tall) stands on "
            f"the floor. Its TOP plate has two square loading openings "
            f"(~{c.ap_y * 100:.0f} cm), one over each bay; through an uncovered opening "
            f"you can see that bay's colored floor pad: the RED bay on one side, the "
            f"BLUE bay on the other. A single dark sliding SHUTTER with a yellow grip "
            f"post rides a short track over the openings. The shutter is captive and "
            f"{c.shutter_y * 100:.0f} cm long: at one end of its travel it fully covers "
            f"the blue opening (leaving the red one open), at the other end the "
            f"reverse, and at NO track position can a cup pass through both openings — "
            f"uncovering one seals the other. It may start anywhere on the track.\n"
            f"On the floor in front of the cabinet stand three identical-size cups "
            f"({2 * c.cup_r * 1000:.0f} mm wide, {c.cup_h * 1000:.0f} mm tall): two RED "
            f"and one BLUE (their starting spots change every episode).\n"
            f"Goal: store every cup UPRIGHT on the floor of the bay matching its color "
            f"— both red cups in the red bay, the blue cup in the blue bay, lowered in "
            f"through that bay's top opening (slide the shutter clear of it first) — "
            f"and then park the shutter fully covering the BLUE opening. Apart from "
            f"the obvious constraint that the blue cup must go in before its opening "
            f"is finally covered, the order is free. A cup in the wrong bay, lying on "
            f"its side, left on the floor, resting on the shutter or the cabinet top "
            f"does not count; success is judged with everything at rest and the "
            f"shutter parked over the blue opening."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Sort the three cups into the cabinet through its top openings — each cup "
            "upright on the floor of the bay matching its color, sliding the captive "
            "shutter as needed (it never uncovers both openings at once) — then park "
            "the shutter fully covering the blue opening. A cup in the wrong bay, "
            "lying on its side, or the shutter left off the blue opening fails."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def shutter_c(self) -> torch.Tensor:
        """(N,) shutter track position: its cabinet-frame y offset (the case never
        moves or rotates, so the cabinet frame is a world translation)."""
        return (self.shutter.data.root_pos_w - self.case_origin)[:, 1]

    def _cup_local(self, nm: str) -> torch.Tensor:
        """(N, 3) cup center in the cabinet frame."""
        return self.cups[nm].data.root_pos_w - self.case_origin

    def cup_upright(self, nm: str) -> torch.Tensor:
        """(N,) bool: cup axis within `upright_max_deg` of world-up."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        up = quat_apply(self.cups[nm].data.root_quat_w, ez)
        return up[:, 2] >= self.upright_cos_t()

    def upright_cos_t(self) -> float:
        return self.cfg.upright_cos

    def in_bay(self, nm: str, side: float) -> torch.Tensor:
        """(N,) bool: cup center inside bay `side` (+1 blue / -1 red), resting at
        floor height (the z band rejects plate-top, shutter-top and stacked cups)."""
        c = self.cfg
        loc = self._cup_local(nm)
        x_ok = loc[:, 0].abs() <= c.bay_x_tol
        yy = side * loc[:, 1]
        y_ok = (yy >= c.bay_y_lo) & (yy <= c.bay_y_hi)
        z_ok = (loc[:, 2] - c.cup_rest_z).abs() <= c.bay_z_tol
        return x_ok & y_ok & z_ok

    def seated(self, nm: str) -> torch.Tensor:
        """(N,) bool: cup upright inside ITS OWN color bay."""
        side = dict(zip(self.cfg.cup_names, self.cfg.cup_sides))[nm]
        return self.in_bay(nm, side) & self.cup_upright(nm)

    def all_seated(self) -> torch.Tensor:
        """(N,) bool: every cup seated in its own bay."""
        out = None
        for nm in self.cfg.cup_names:
            s = self.seated(nm)
            out = s if out is None else out & s
        return out

    def covers_blue(self) -> torch.Tensor:
        """(N,) bool: shutter parked far enough +y that the blue opening is sealed."""
        return self.shutter_c() >= self.cfg.cover_c_min

    def _still_now(self) -> torch.Tensor:
        """(N,) bool: every cup slow AND the shutter parked — INSTANTANEOUS (never
        judge on this alone; a cup mid-drop or the shutter mid-slide must not count)."""
        c = self.cfg
        ok = self.shutter.data.root_lin_vel_w.norm(dim=-1) < c.settle_shutter
        for b in self.cups.values():
            ok = ok & (b.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
                & (b.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
        return ok

    def settled(self) -> torch.Tensor:
        """(N,) bool: stillness has PERSISTED `settle_steps_min` consecutive steps."""
        return self.still_count >= self.cfg.settle_steps_min

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Stillness counter-latch + progress latches: per-cup ever-seated-in-own-bay;
        all three seated at once; all-seated WHILE the shutter covers blue (gated on
        all-seated so covering an unloaded cabinet — or the random start pose — earns
        nothing)."""
        self.still_count = (self.still_count + 1.0) * self._still_now().float()
        seats = torch.stack([self.seated(nm) for nm in self.cfg.cup_names], dim=-1)
        self.seat_latch = torch.maximum(self.seat_latch, seats.float())
        alls = seats.all(dim=-1)
        self.all_latch = torch.maximum(self.all_latch, alls.float())
        cov = alls & self.covers_blue()
        self.covered_latch = torch.maximum(self.covered_latch, cov.float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: both red cups upright in the red bay, the blue cup upright in
        the blue bay, the shutter parked fully covering the blue opening, everything
        persistently still."""
        return self.all_seated() & self.covers_blue() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15 per cup ever seated in its own bay + 0.15 all
        seated at once + 0.15 all-seated with the blue opening covered (all latched —
        credit never evaporates); 1.0 iff success(). Null policy ~0 (a shutter that
        STARTS covering blue earns nothing — the coverage term is gated on seats)."""
        base = (0.15 * self.seat_latch.sum(dim=1) + 0.15 * self.all_latch
                + 0.15 * self.covered_latch).clamp(0.0, 0.85)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="shutter_cabinet", robot="null", env_spacing=3.0))

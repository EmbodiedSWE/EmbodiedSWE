"""FerryDoorCabinetScene — ferry two cups into a sealed roofed gallery cabinet using the
sliding door itself as the cargo ram, then leave the door fully closed as the seal
(sim_gen task `slide_cabinet_open_and_place_cups_i432`).

Derived from rlbench/slide_cabinet_open_and_place_cups, but STRATEGICALLY different:
the seed slides one cabinet door OPEN once — getting the panel out of the way is the
whole articulation act — and then the arm reaches into the revealed volume to place
interchangeable cups, judged by a recorded trajectory. Here the interior can NEVER be
reached into: the cabinet is a low, fully roofed gallery whose storage bay lies at the
far end of a roofed corridor, and the only opening in the shell is a roof PORT above
the corridor, ~13 cm short of the bay (far beyond finger reach under a 7 cm roof). The
sliding door is a captive piston INSIDE that corridor: a cup dropped through the port
lands in the door's path, and the door's own CLOSING stroke is the only way to move it
— the leading face rams the cup down the corridor, past the bay line, into the bay.
The panel's role is INVERTED (it is the transport tool, not the obstacle: cargo moves
only while the door closes), it must be CYCLED (retract to re-expose the port, stage
the next cup, ram again — or stage both and ram the pair), and its fully-closed pose
is itself part of the goal state (it seals the corridor). A solver needs a different
PLAN (a stage-ram-retract cycle schedule; the seed's "open once, then place by hand"
plan is impossible — there is no state where the bay is hand-accessible) and different
code STRUCTURE (drive-the-door-to-deliver with bay membership judged at the far end of
a stroke, not door-displacement + place-in-one-volume).

success(): both cups upright at floor rest inside the bay (cabinet-frame membership
window whose z band rejects cups on the roof, on the door top, or stacked; an
uprightness cone rejects cups lying on their side), the door parked at its fully
CLOSED stop band, and everything PERSISTENTLY still (stillness counter-latch —
`settle_steps_min` consecutive quiet steps).

score(), latched (credit never evaporates): 0.10 per cup ever staged in the corridor
through the port + 0.20 per cup ever upright in the bay + 0.10 once both are in the
bay simultaneously + 0.05 once both-in coincides with the door closed (gated — an
empty cabinet with the door merely closed, or starting closed, earns nothing);
partial credit capped at 0.75; 1.0 iff success(). Null policy scores ~0.

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - case (KINEMATIC): a long, low roofed gallery (~41 x 9 x 9 cm) on the ground.
    Interior cross-section 72 x 70 mm. The bay is the y<0 end (bay line y=0, cabinet
    frame); the corridor (y>0) is roofed except the full-width roof PORT
    (y in [port_y0, port_y1]) and a narrow center handle SLOT running from the port
    to the front wall (too narrow for a cup by a wide margin). Never teleported (the
    door joint's body0 anchor is world-fixed) — the world anchor of the mechanism.
  - door (dynamic): a 12 cm piston plate filling the corridor cross-section to within
    2-4 mm on every side (asserted: no cup-sized gap anywhere around it), with a
    yellow handle post rising through the roof slot. It rides a spawn-authored
    prismatic joint (axis y, hard stops +/- `half_stroke`): at the OPEN stop its
    leading face clears the port; at the CLOSED stop the face sits 18 mm past the bay
    line. No spring, no detent — it stays where it is put (asserted in smoke).
  - cups (x2, dynamic): identical orange cylinders, on the ground beside the cabinet.
Friction materials are bound explicitly everywhere (the default-material trap);
contact offsets are explicit so the membership z bands stay real.

Per-episode randomization (readback-verified in smoke): the door's starting track
position (uniform over ~the whole stroke — episodes start anywhere from nearly closed,
port covered, to nearly open) and the cups' ground slots (2 of 4, shuffled) + jitter.

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
    """KINEMATIC roofed gallery; body origin at GROUND level on the bay line (y=0),
    centered in x. Interior: floor top z=fz, roof underside z=fz+ih. Bay y<0."""
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
    ihw, wt, fz, ih, rt = cfg.ihw, cfg.wall_t, cfg.floor_t, cfg.inner_h, cfg.roof_t
    y0, y1 = cfg.y_back_in, cfg.y_front_in  # interior y extent
    ohw = ihw + wt  # outer half width
    grey = (0.42, 0.42, 0.46)
    dark = (0.28, 0.28, 0.33)
    # --- floor plate (footprint incl. both end walls) ---
    _box(stage, f"{prim_path}/floor", (2 * ohw, (y1 + wt) - (y0 - wt), fz),
         (0.0, (y1 + y0) / 2, fz / 2), grey, co, material=mat)
    # --- side walls (full interior length) ---
    for sgn, nm in ((-1.0, "wall_xn"), (1.0, "wall_xp")):
        _box(stage, f"{prim_path}/{nm}", (wt, y1 - y0, ih),
             (sgn * (ihw + wt / 2), (y1 + y0) / 2, fz + ih / 2), dark, co, material=mat)
    # --- end walls: back (bay end, roof-covered) and front (tall, to roof top) ---
    _box(stage, f"{prim_path}/wall_back", (2 * ihw, wt, ih),
         (0.0, y0 - wt / 2, fz + ih / 2), dark, co, material=mat)
    _box(stage, f"{prim_path}/wall_front", (2 * ihw, wt, ih + rt),
         (0.0, y1 + wt / 2, fz + (ih + rt) / 2), dark, co, material=mat)
    # --- roof: solid over bay+corridor up to the port; then side strips beside the
    #     handle slot from the port to the front wall (the port and the slot are the
    #     ONLY openings in the shell) ---
    rz = fz + ih + rt / 2
    _box(stage, f"{prim_path}/roof_bay", (2 * ohw, cfg.port_y0 - (y0 - wt), rt),
         (0.0, (cfg.port_y0 + y0 - wt) / 2, rz), grey, co, material=mat)
    for sgn, nm in ((-1.0, "roof_sn"), (1.0, "roof_sp")):
        xin, xout = cfg.slot_hw, ohw
        _box(stage, f"{prim_path}/{nm}", (xout - xin, y1 - cfg.port_y1, rt),
             (sgn * (xin + xout) / 2, (y1 + cfg.port_y1) / 2, rz), grey, co, material=mat)
    return root


def _spawn_door(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The ferry door: ONE dynamic rigid body (piston plate + handle post) riding a
    spawn-authored prismatic joint (axis Y, hard stops +/- half_stroke) against the
    sibling Case. It hovers 2 mm above the corridor floor — the joint is the track,
    nothing rubs. Mass/CoM/inertia are AUTHORED. The joint is authored IN THE SPAWNER
    so it exists BEFORE the physics parse (a joint authored in bind() comes too late
    and the body is baked free)."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.0008, 0.0004, 0.0008))
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
    _box(stage, f"{prim_path}/plate", (cfg.door_w, cfg.door_l, cfg.door_h),
         (0.0, 0.0, 0.0), (0.16, 0.16, 0.18), co, material=mat)
    _box(stage, f"{prim_path}/handle", (0.016, 0.024, cfg.handle_h),
         (0.0, cfg.handle_dy, cfg.door_h / 2 + cfg.handle_h / 2),
         (0.92, 0.78, 0.10), co, material=mat)

    # Prismatic track to the sibling Case (pair collision FILTERED by the joint —
    # the door hovers inside the corridor and never needs case contact; the cups are
    # separate bodies and DO collide with both).
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.PrismaticJoint.Define(stage, f"{prim_path}/track")
    j.CreateBody0Rel().SetTargets([f"{base}/Case"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateCollisionEnabledAttr(False)
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, float(cfg.track_y0), float(cfg.track_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-float(cfg.half_stroke))
    j.CreateUpperLimitAttr(float(cfg.half_stroke))
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
            ihw: float = 0.036
            wall_t: float = 0.010
            floor_t: float = 0.010
            inner_h: float = 0.070
            roof_t: float = 0.010
            y_back_in: float = -0.150
            y_front_in: float = 0.245
            port_y0: float = 0.030
            port_y1: float = 0.098
            slot_hw: float = 0.010
            mu_static: float = 0.5
            mu_dynamic: float = 0.4
            contact_offset: float = 0.0015

        @configclass
        class DoorSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_door)
            door_w: float = 0.064
            door_l: float = 0.120
            door_h: float = 0.064
            handle_dy: float = 0.045
            handle_h: float = 0.064
            track_y0: float = 0.106
            track_z: float = 0.044
            half_stroke: float = 0.064
            mass: float = 0.35
            lin_damping: float = 4.0
            mu_static: float = 0.5
            mu_dynamic: float = 0.4
            contact_offset: float = 0.0015

        _SPAWNER_CACHE.update(case=CaseSpawnerCfg, door=DoorSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class FerryDoorCabinetSceneCfg(BaseCfg):
    """Config for `FerryDoorCabinetScene`. `__post_init__` asserts the strategic
    honesty invariants: no cup-sized gap ever exists around the door (the ram is the
    only way past it), the port passes a cup while the handle slot never does, the
    open door fully clears the port, the closed stroke geometrically delivers both
    cups inside the bay membership band, the z band accepts floor rest (and a LYING
    cup — uprightness is load-bearing) while rejecting door-top / roof-top / stacked
    cups, and a ram push can never tip a cup (mu * h < r)."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    bay_x_tol: float = tunable(0.020)  # cup center |x| inside the corridor (cabinet frame)
    bay_y_lo: float = tunable(-0.148)  # bay membership: cup center y band ...
    bay_y_hi: float = tunable(-0.040)  # ... (back wall face + r  to  closed face - r - 6 mm)
    stage_y_lo: float = tunable(-0.010)  # corridor (staged) membership y band ...
    stage_y_hi: float = tunable(0.102)  # ... (bay line - eps  to  open door face - r + eps)
    z_tol: float = tunable(0.015)  # |cup center z - floor rest z| band (rejects door-top
    # (+0.066 off), roof-top (+0.080 off) and stacked (+0.060 off) cups)
    upright_max_deg: float = tunable(30.0)  # cup axis within this of world-up
    closed_slack: float = tunable(0.006)  # door counts CLOSED within this of the -stop
    settle_lin: float = tunable(0.05)  # max cup |lin vel| when judging (m/s)
    settle_ang: float = tunable(1.0)  # max cup |ang vel| when judging (rad/s)
    settle_door: float = tunable(0.02)  # max door |lin vel| counted as parked (m/s)
    settle_steps_min: int = tunable(30)  # stillness must PERSIST this many steps (0.25 s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    d0_range: float = tunable(0.058)  # door start track pos ~ U(-r, +r) (stops at 0.064)
    slot_x: float = tunable(0.16)  # cup staging column x (cups start on the ground here)
    slot_ys: tuple = tunable((-0.06, 0.07, 0.20, 0.33))  # 4 ground slots; 2 used, shuffled
    slot_jitter: float = tunable(0.025)  # uniform +/- xy jitter per cup at reset (m)

    # --- info: structure ---------------------------------------------------------------------
    case_pos: tuple = info((0.0, 0.0))  # cabinet origin xy in the env (NEVER teleported)
    ihw: float = info(0.036)  # interior half width (x)
    wall_t: float = info(0.010)
    floor_t: float = info(0.010)  # floor plate; interior floor TOP = floor_t
    inner_h: float = info(0.070)  # interior clear height (floor top -> roof underside)
    roof_t: float = info(0.010)  # roof top = floor_t + inner_h + roof_t = 0.090
    y_back_in: float = info(-0.150)  # bay back wall interior face (bay line is y=0)
    y_front_in: float = info(0.245)  # front wall interior face
    port_y0: float = info(0.030)  # roof port (full interior width): y in [port_y0, port_y1]
    port_y1: float = info(0.098)
    slot_hw: float = info(0.010)  # handle slot half width (port_y1 -> front wall)
    door_w: float = info(0.064)  # door plate x extent (4 mm side gaps)
    door_l: float = info(0.120)  # door plate y extent
    door_h: float = info(0.064)  # door plate z extent (2 mm hover, 4 mm top slit)
    door_hover: float = info(0.002)
    handle_dy: float = info(0.045)  # handle post y offset from the door center
    handle_h: float = info(0.064)  # handle post height above the door top
    track_y0: float = info(0.106)  # door center y at track coordinate d = 0
    track_z: float = info(0.044)  # door center z (= floor_t + hover + door_h/2)
    half_stroke: float = info(0.064)  # prismatic hard stops at d = +/- half_stroke
    door_mass: float = info(0.35)
    door_damping: float = info(4.0)
    cup_r: float = info(0.028)
    cup_h: float = info(0.060)
    cup_mass: float = info(0.06)
    cup_names: tuple = info(("cup_a", "cup_b"))
    cup_color: tuple = info((0.90, 0.45, 0.10))
    mu_static: float = info(0.5)
    mu_dynamic: float = info(0.4)
    contact_offset: float = info(0.0015)

    # Derived (filled in __post_init__).
    upright_cos: float = field(default=None, init=False)
    cup_rest_z: float = field(default=None, init=False)  # cup center at floor rest, cab frame
    face_open: float = field(default=None, init=False)  # door leading face y at the +stop
    face_closed: float = field(default=None, init=False)  # door leading face y at the -stop

    def __post_init__(self) -> None:
        self.upright_cos = math.cos(math.radians(self.upright_max_deg))
        self.cup_rest_z = self.floor_t + self.cup_h / 2
        self.face_open = self.track_y0 + self.half_stroke - self.door_l / 2
        self.face_closed = self.track_y0 - self.half_stroke - self.door_l / 2

        r, h = self.cup_r, self.cup_h
        # -- the ram is the only way past the door: no cup-sized gap around it, ever
        side_gap = self.ihw - self.door_w / 2
        top_slit = (self.floor_t + self.inner_h) - (self.track_z + self.door_h / 2)
        assert max(side_gap, top_slit, self.door_hover) <= 0.006 < 2 * r, \
            "a cup must never fit around the door (side/top/bottom slits)"
        assert abs(self.track_z - self.door_h / 2 - self.floor_t - self.door_hover) < 1e-9, \
            "track_z inconsistent with hover"
        # -- the port passes a cup; the handle slot never does (and passes the handle)
        assert self.port_y1 - self.port_y0 >= 2 * r + 0.008, "port must pass a cup (y)"
        assert 2 * self.ihw >= 2 * r + 0.010, "port must pass a cup (x)"
        assert 2 * self.slot_hw <= 2 * r - 0.030, "handle slot must never pass a cup"
        assert 2 * self.slot_hw >= 0.016 + 0.003, "handle slot must pass the handle post"
        # -- the OPEN door fully clears the port (staging is possible)
        assert self.face_open >= self.port_y1 + 0.010, "open door must clear the port"
        assert self.y_front_in >= self.track_y0 + self.half_stroke + self.door_l / 2 + 0.010, \
            "front wall must clear the open door tail"
        # -- the CLOSED stroke geometrically delivers both cups inside the bay band
        last_center = self.face_closed - r  # cup rammed until the door parks at the -stop
        first_center = last_center - 2 * r  # the first cup, pushed on by the second
        assert last_center <= self.bay_y_hi - 0.004, "rammed cup must land in the bay band"
        assert first_center >= self.bay_y_lo + 0.004, "chained cup must stay in the bay band"
        assert first_center - r >= self.y_back_in + 0.010, "bay must hold both cups w/ slack"
        # door counted closed => face at least 8 mm past the bay line
        assert self.track_y0 - (self.half_stroke - self.closed_slack) - self.door_l / 2 \
            <= -0.008, "closed band must put the face well past the bay line"
        # -- z band honesty: accepts floor rest AND a lying cup (uprightness is
        #    load-bearing), rejects door-top / roof-top / stacked heights
        assert abs((self.floor_t + r) - self.cup_rest_z) < self.z_tol, \
            "lying cup passes the z band -> the uprightness clause is load-bearing"
        for wrong in (self.track_z + self.door_h / 2 + h / 2,  # on the door top
                      self.floor_t + self.inner_h + self.roof_t + h / 2,  # on the roof
                      self.cup_rest_z + h):  # stacked on another cup
            assert abs(wrong - self.cup_rest_z) > self.z_tol + 0.005, \
                f"z band must reject resting height {wrong:+.4f}"
        # -- a ram push can never tip a cup (worst case: force applied at the rim)
        assert self.mu_dynamic * h <= r, "ram push must slide, not tip, a cup"
        # -- staged band sits between the bay line and the open door face
        assert self.stage_y_hi <= self.face_open - r + 0.022, "staged band under the shell"
        assert self.stage_y_lo >= -0.016, "staged band must not reach into the bay"
        assert self.bay_y_hi < self.stage_y_lo + 0.032, "bay and staged bands must not blur"
        # -- fits and reset sanity
        assert 2 * r <= 0.075, "cup must fit a parallel jaw (~80 mm)"
        assert self.d0_range <= self.half_stroke - 0.005, \
            "reset must not write the door into a stop"
        assert min(abs(self.slot_ys[i] - self.slot_ys[j])
                   for i in range(len(self.slot_ys)) for j in range(i)) \
            >= 2 * r + 2 * self.slot_jitter + 0.012, "ground slots must never overlap"
        assert self.slot_x - self.slot_jitter - r >= self.ihw + self.wall_t + 0.010, \
            "ground slots must clear the cabinet footprint"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("ferry_door_cabinet")
class FerryDoorCabinetScene(BaseScene):
    cfg: FerryDoorCabinetSceneCfg

    def __init__(self, cfg: FerryDoorCabinetSceneCfg | None = None) -> None:
        super().__init__(cfg or FerryDoorCabinetSceneCfg())

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
                    ihw=c.ihw, wall_t=c.wall_t, floor_t=c.floor_t, inner_h=c.inner_h,
                    roof_t=c.roof_t, y_back_in=c.y_back_in, y_front_in=c.y_front_in,
                    port_y0=c.port_y0, port_y1=c.port_y1, slot_hw=c.slot_hw,
                    mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0)),
            ),
            "door": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Door",
                spawn=spawners["door"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.door_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    door_w=c.door_w, door_l=c.door_l, door_h=c.door_h,
                    handle_dy=c.handle_dy, handle_h=c.handle_h,
                    track_y0=c.track_y0, track_z=c.track_z, half_stroke=c.half_stroke,
                    mass=c.door_mass, lin_damping=c.door_damping,
                    mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(px, py + c.track_y0, c.track_z)),
            ),
        }
        for i, nm in enumerate(c.cup_names):
            out[nm] = RigidObjectCfg(
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
                        diffuse_color=c.cup_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(px + c.slot_x, py + c.slot_ys[i], c.cup_h / 2 + 0.003)),
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
        self.door: RigidObject = env.iscene["door"]
        self.cups: dict[str, RigidObject] = {nm: env.iscene[nm] for nm in c.cup_names}
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # cabinet origin, world (constant: the case is NEVER teleported)
        self.case_origin = self.env_origins + torch.tensor(
            [c.case_pos[0], c.case_pos[1], 0.0], device=dev)
        self.stage_latch = torch.zeros(n, 2, device=dev)  # cup i ever staged (corridor)
        self.bay_latch = torch.zeros(n, 2, device=dev)  # cup i ever upright in the bay
        self.both_latch = torch.zeros(n, device=dev)  # both in the bay at once
        self.closed_latch = torch.zeros(n, device=dev)  # both in AND door closed
        self.still_count = torch.zeros(n, device=dev)  # consecutive still steps

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: door written to a uniform random track position (episodes
        start anywhere from nearly closed — port covered — to nearly open), cups on
        2 of 4 shuffled ground slots with jitter; latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.case_origin[env_ids]

        # --- door: random position along its track ---
        d0 = (torch.rand(m, device=dev) * 2 - 1) * c.d0_range
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = origin
        st[:, 1] += c.track_y0 + d0
        st[:, 2] += c.track_z
        st[:, 3] = 1.0
        self.door.write_root_state_to_sim(st, env_ids)

        # --- cups: 2 of the 4 ground slots, shuffled, + jitter, upright ---
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

        self.stage_latch[env_ids] = 0.0
        self.bay_latch[env_ids] = 0.0
        self.both_latch[env_ids] = 0.0
        self.closed_latch[env_ids] = 0.0
        self.still_count[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "door": self.door.data.root_state_w[env_ids].clone(),
            "cups": {nm: b.data.root_state_w[env_ids].clone()
                     for nm, b in self.cups.items()},
            "stage_latch": self.stage_latch[env_ids].clone(),
            "bay_latch": self.bay_latch[env_ids].clone(),
            "both_latch": self.both_latch[env_ids].clone(),
            "closed_latch": self.closed_latch[env_ids].clone(),
            "still_count": self.still_count[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.door.write_root_state_to_sim(state["door"], env_ids)
        for nm, b in self.cups.items():
            b.write_root_state_to_sim(state["cups"][nm], env_ids)
        self.stage_latch[env_ids] = state["stage_latch"]
        self.bay_latch[env_ids] = state["bay_latch"]
        self.both_latch[env_ids] = state["both_latch"]
        self.closed_latch[env_ids] = state["closed_latch"]
        self.still_count[env_ids] = state["still_count"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        fp_y = (c.y_front_in + c.wall_t) - (c.y_back_in - c.wall_t)
        return (
            f"A long, low, fully ROOFED gallery cabinet ({fp_y * 100:.0f} cm long, "
            f"{2 * (c.ihw + c.wall_t) * 100:.0f} cm wide, ~9 cm tall) sits on the floor. "
            f"Its storage BAY is the far (back) end of the interior; the rest of the "
            f"interior is a straight roofed corridor. The shell has exactly two "
            f"openings, both in the roof: a rectangular loading PORT "
            f"(~{(c.port_y1 - c.port_y0) * 1000:.0f} x {2 * c.ihw * 1000:.0f} mm) over "
            f"the corridor, about {c.port_y0 * 100:.0f}-{c.port_y1 * 100:.0f} cm in "
            f"front of the bay, and a narrow slot (2 cm wide — far too narrow for a "
            f"cup) through which a yellow HANDLE post rises from the sliding door "
            f"inside. The door is a captive piston filling the corridor cross-section; "
            f"dragging its handle slides it between fully OPEN (retracted clear of the "
            f"port) and fully CLOSED (its leading face past the bay entrance). It has "
            f"no spring and stays where it is left. The bay itself is UNREACHABLE from "
            f"outside: no hand or tool fits under the roof, so the only way to move a "
            f"cup from the port to the bay is the door's own closing stroke — drop a "
            f"cup upright through the port (with the door retracted clear of it), then "
            f"slide the door closed so its face pushes the cup down the corridor into "
            f"the bay. Two identical orange cups "
            f"({2 * c.cup_r * 1000:.0f} mm wide, {c.cup_h * 1000:.0f} mm tall) stand "
            f"on the ground beside the cabinet (their spots change every episode; the "
            f"door may start anywhere on its track).\n"
            f"Goal: both cups stored UPRIGHT on the bay floor, and the door parked at "
            f"its fully CLOSED stop, sealing the corridor. You may ferry the cups one "
            f"per closing stroke (close, re-open, stage the second, close again) or "
            f"stage both in the corridor and ram them in with one final stroke; either "
            f"cup may go first. A cup dropped while the door still covers the port "
            f"just lands on the door's top. A cup lying on its side (it will be rammed "
            f"in lying and does not count), left in the corridor, on the roof or on "
            f"the ground does not count; success is judged with everything at rest and "
            f"the door fully closed."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Store both orange cups upright in the roofed cabinet's back bay: slide "
            "the door open by its yellow handle, drop a cup upright through the roof "
            "port, and slide the door closed so it rams the cup into the bay; repeat "
            "(or stage both, then ram once), and leave the door at its fully closed "
            "stop. A cup lying on its side, left outside the bay, or the door left "
            "open fails."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def door_d(self) -> torch.Tensor:
        """(N,) door track coordinate: cabinet-frame center y minus track_y0 (the
        case never moves or rotates, so the cabinet frame is a world translation).
        d = -half_stroke is fully CLOSED, +half_stroke is fully OPEN."""
        return (self.door.data.root_pos_w - self.case_origin)[:, 1] - self.cfg.track_y0

    def _cup_local(self, nm: str) -> torch.Tensor:
        """(N, 3) cup center in the cabinet frame."""
        return self.cups[nm].data.root_pos_w - self.case_origin

    def cup_upright(self, nm: str) -> torch.Tensor:
        """(N,) bool: cup axis within `upright_max_deg` of world-up."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        up = quat_apply(self.cups[nm].data.root_quat_w, ez)
        return up[:, 2] >= self.cfg.upright_cos

    def in_bay(self, nm: str) -> torch.Tensor:
        """(N,) bool: cup center inside the bay membership window at floor-rest
        height (the z band rejects door-top, roof-top and stacked cups)."""
        c = self.cfg
        loc = self._cup_local(nm)
        return (loc[:, 0].abs() <= c.bay_x_tol) \
            & (loc[:, 1] >= c.bay_y_lo) & (loc[:, 1] <= c.bay_y_hi) \
            & ((loc[:, 2] - c.cup_rest_z).abs() <= c.z_tol)

    def staged(self, nm: str) -> torch.Tensor:
        """(N,) bool: cup upright at floor rest in the CORRIDOR (between the bay
        line and the open door face) — i.e. successfully admitted through the port."""
        c = self.cfg
        loc = self._cup_local(nm)
        return (loc[:, 0].abs() <= c.bay_x_tol) \
            & (loc[:, 1] >= c.stage_y_lo) & (loc[:, 1] <= c.stage_y_hi) \
            & ((loc[:, 2] - c.cup_rest_z).abs() <= c.z_tol) & self.cup_upright(nm)

    def seated(self, nm: str) -> torch.Tensor:
        """(N,) bool: cup upright inside the bay."""
        return self.in_bay(nm) & self.cup_upright(nm)

    def both_seated(self) -> torch.Tensor:
        """(N,) bool: both cups upright in the bay."""
        out = None
        for nm in self.cfg.cup_names:
            s = self.seated(nm)
            out = s if out is None else out & s
        return out

    def door_closed(self) -> torch.Tensor:
        """(N,) bool: door parked within `closed_slack` of its fully-closed stop."""
        c = self.cfg
        return self.door_d() <= -(c.half_stroke - c.closed_slack)

    def _still_now(self) -> torch.Tensor:
        """(N,) bool: cups slow AND the door parked — INSTANTANEOUS (never judge on
        this alone; a cup mid-ram or the door mid-stroke must not count)."""
        c = self.cfg
        ok = self.door.data.root_lin_vel_w.norm(dim=-1) < c.settle_door
        for b in self.cups.values():
            ok = ok & (b.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
                & (b.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
        return ok

    def settled(self) -> torch.Tensor:
        """(N,) bool: stillness has PERSISTED `settle_steps_min` consecutive steps."""
        return self.still_count >= self.cfg.settle_steps_min

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Stillness counter-latch + progress latches: per-cup ever-staged (through
        the port), per-cup ever-in-the-bay, both-in-at-once, and both-in WHILE the
        door is closed (gated on both-in, so a door that merely starts — or is merely
        driven — closed over an empty cabinet earns nothing)."""
        self.still_count = (self.still_count + 1.0) * self._still_now().float()
        stages = torch.stack([self.staged(nm) for nm in self.cfg.cup_names], dim=-1)
        self.stage_latch = torch.maximum(self.stage_latch, stages.float())
        seats = torch.stack([self.seated(nm) for nm in self.cfg.cup_names], dim=-1)
        self.bay_latch = torch.maximum(self.bay_latch, seats.float())
        both = seats.all(dim=-1)
        self.both_latch = torch.maximum(self.both_latch, both.float())
        closed = both & self.door_closed()
        self.closed_latch = torch.maximum(self.closed_latch, closed.float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: both cups upright at floor rest in the bay, the door parked at
        its fully-closed stop, everything persistently still."""
        return self.both_seated() & self.door_closed() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10 per cup ever staged through the port + 0.20 per
        cup ever upright in the bay + 0.10 both in the bay at once + 0.05 both-in
        with the door closed (all latched — credit never evaporates); 1.0 iff
        success(). Null policy ~0 (a door that starts closed earns nothing — the
        closed term is gated on both cups being in the bay)."""
        base = (0.10 * self.stage_latch.sum(dim=1) + 0.20 * self.bay_latch.sum(dim=1)
                + 0.10 * self.both_latch + 0.05 * self.closed_latch).clamp(0.0, 0.75)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="ferry_door_cabinet", robot="null", env_spacing=3.0))

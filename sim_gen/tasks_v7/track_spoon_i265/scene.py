"""SpoonKnifeEdgeScene — load a serving spoon's bowl with a metal cube, then balance
the LOADED spoon level across a knife-edge fin (sim_gen task `track_spoon_i265`).

Derived from pick_place/track_spoon, but STRATEGICALLY different: the seed starts with
a spoon ALREADY rigidly grasped in the closed Franka gripper and rewards dense per-step
position+rotation tracking of a prescribed free-space waypoint path toward a basket —
pure transport fidelity of a held payload, with nothing to compute and no physical
consequence to any placement choice. Here the judged skill is COMPOSED CENTRE-OF-MASS
REASONING VERIFIED BY A STATIC-EQUILIBRIUM OUTCOME: a dense cube must first be seated
in the spoon's bowl pocket, and the loaded spoon must then be laid ACROSS a narrow
knife-edge fin so that it rests LEVEL with both ends hanging in free air. The fin
crest is only 20 mm wide, while every naive support anchor is far outside the balance
window: the spoon's geometric centre misses by ~48 mm, the EMPTY spoon's own CoM by
~28 mm, and the bowl centre by ~27 mm. Only the COMPOSED CoM of spoon+cube (the cube
weighs as much as the whole spoon) lies over the crest — so the solver must reason
about where the combined balance point sits, not where the object or its parts are.
There is no prescribed path and no per-step tracking; placement is judged solely by
the settled equilibrium physics produces. A solver therefore needs a different plan
(compose masses, compute a support line, verify by release) and different code
(equilibrium predicates instead of a waypoint follower).

Judged on the settled state, in body frames:
  success() iff  the cube is seated in the spoon's bowl pocket (spoon frame)  AND  the
  spoon rests across the fin crest at crest height with its long axis crossing the fin
  plane inside the fin span (stand frame)  AND  the spoon is LEVEL (axis tilt <=
  tilt_max_deg) and not rolled  AND  both spoon ends overhang beyond the crest on
  OPPOSITE sides and hang in free air  AND  everything is at rest with the stand
  upright.
score() is latched every physics substep: 0.20 * the cube was ever seated in the
pocket (quietly)  +  0.30 * the seated assembly ever rested spanning the crest at
height  +  0.20 * success() ever held, capped at 0.70; exactly 1.0 iff success() holds
now. Doing nothing scores ~0; the seed's strategy (carry the spoon somewhere and set
it down) also scores ~0.

Assets are fully procedural (no external meshes, boxes only — no spheres/capsules):
  - stand: DYNAMIC (6 kg — teleported at reset; a heavy slab so nudges don't walk it)
    base slab 0.20 x 0.16 x 0.020 with a knife-edge FIN plate on top (20 mm thick
    along stand x, 140 mm long along stand y, crest top 80 mm up), xy + yaw
    randomized per episode;
  - spoon: one rigid compound with a FULLY FLAT underside (bowl floor and handle
    bottom coplanar at local z=0, so it can rest flush on the crest anywhere along
    its length): rimmed square bowl pocket (48 mm outer, 12 mm rims) at the -x end, a
    150 x 16 x 8 mm handle, and a raised 16 x 16 x 22 mm tail KNOB for a parallel-jaw
    pinch. Mass ~46.5 g with CoM ~20 mm bowl-ward of the geometric centre — both
    DERIVED in cfg from the part dimensions (single source of truth) and authored
    explicitly via MassAPI (custom spawn funcs ignore cfg mass_props on this stack;
    MassAPI-only mass leaves the CoM at the body origin, so the CoM is authored too);
  - cube: a 26 mm aluminium-density cube (~47.5 g, a standard spawner that honors
    mass_props), fitting the 40 mm pocket with 7 mm slack per side.
Balance arithmetic (all asserted in cfg.__post_init__): loaded balance point
bal_x = (m_spoon*com_x + m_cube*bowl_cx) / (m_spoon + m_cube) ~ -48 mm; the support
window is the crest half-width (10 mm) and the worst-case CoM shift from cube slack in
the pocket is ~3.5 mm, so a correct placement balances robustly while every naive
anchor (centre / empty CoM / bowl centre, all >= 25 mm off) tips decisively — a
handle-side tip rests ~24 deg down on the slab, 6x the level tolerance.

Per-episode randomization (verified by readback in smoke): stand xy + yaw, spoon
floor slot xy + free yaw, cube floor slot xy. Heavy imports (isaaclab, pxr) are
deferred so importing this module — and registering the scene — stays app-free.
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


# ----- small torch quaternion helpers (wxyz) ---------------------------------------------------
def _qz(yaw: torch.Tensor) -> torch.Tensor:
    """(m,) yaw -> (m, 4) wxyz quaternion about world z."""
    q = torch.zeros(yaw.shape[0], 4, device=yaw.device)
    q[:, 0] = torch.cos(yaw / 2)
    q[:, 3] = torch.sin(yaw / 2)
    return q


# ----- custom compound spawners ----------------------------------------------------------------
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


def _rigid_dynamic(root, *, mass: float, com, lin_damp: float, ang_damp: float,
                   pos_iters: int = 16, vel_iters: int = 4) -> None:
    """Author a dynamic rigid body with EXPLICIT mass and CoM (MassAPI-only mass leaves
    the CoM at the body origin on this stack — always author both)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    m = UsdPhysics.MassAPI.Apply(root)
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(int(pos_iters))
    px.CreateSolverVelocityIterationCountAttr(int(vel_iters))
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Knife-edge stand: heavy base slab + fin plate whose flat top is the CREST.
    Local origin at the FLOOR under the slab centre; the fin is centred, 20 mm thick
    along local x and long along local y. DYNAMIC (heavy) — reset teleports it."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, mass=cfg.mass, com=(0.0, 0.0, 0.030), lin_damp=2.0, ang_damp=2.0)

    co = cfg.contact_offset
    _box(stage, f"{prim_path}/slab", (cfg.slab_x, cfg.slab_y, cfg.slab_t),
         (0.0, 0.0, cfg.slab_t / 2), cfg.slab_color, co)
    _box(stage, f"{prim_path}/fin", (cfg.fin_t, cfg.fin_len, cfg.fin_h),
         (0.0, 0.0, cfg.slab_t + cfg.fin_h / 2), cfg.fin_color, co)
    return root


def _stand_spawner_cfg(*, slab_x: float, slab_y: float, slab_t: float, fin_t: float,
                       fin_len: float, fin_h: float, mass: float, slab_color: tuple,
                       fin_color: tuple, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "stand" not in _SPAWNER_CACHE:

        @configclass
        class KnifeEdgeStandSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stand)
            slab_x: float = 0.20
            slab_y: float = 0.16
            slab_t: float = 0.020
            fin_t: float = 0.020
            fin_len: float = 0.140
            fin_h: float = 0.060
            mass: float = 6.0
            slab_color: tuple = (0.35, 0.33, 0.30)
            fin_color: tuple = (0.85, 0.45, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["stand"] = KnifeEdgeStandSpawnerCfg

    return _SPAWNER_CACHE["stand"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        slab_x=slab_x, slab_y=slab_y, slab_t=slab_t, fin_t=fin_t, fin_len=fin_len,
        fin_h=fin_h, mass=mass, slab_color=slab_color, fin_color=fin_color,
        contact_offset=contact_offset,
    )


def _spawn_spoon(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Serving spoon with a FULLY FLAT underside (local z=0): rimmed square bowl
    pocket at the -x end (floor slab + four rims), a thin handle whose bottom is
    coplanar with the bowl floor bottom, and a raised tail KNOB for a parallel-jaw
    pinch. Local origin: x = geometric centre of the overall length, z = the flat
    underside plane. Mass AND CoM are cfg-derived and authored explicitly."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, mass=cfg.mass, com=(cfg.com_x, 0.0, cfg.com_z),
                   lin_damp=0.2, ang_damp=1.5)

    co = cfg.contact_offset
    bl, bw, ft = cfg.bowl_l, cfg.bowl_w, cfg.floor_t
    rt, rh = cfg.rim_t, cfg.rim_h
    bcx = cfg.bowl_cx
    _box(stage, f"{prim_path}/bowl_floor", (bl, bw, ft), (bcx, 0.0, ft / 2),
         cfg.color, co)
    for s in (1.0, -1.0):
        _box(stage, f"{prim_path}/rim_x{'p' if s > 0 else 'n'}", (rt, bw, rh),
             (bcx + s * (bl - rt) / 2, 0.0, ft + rh / 2), cfg.rim_color, co)
        _box(stage, f"{prim_path}/rim_y{'p' if s > 0 else 'n'}", (bl - 2 * rt, rt, rh),
             (bcx, s * (bw - rt) / 2, ft + rh / 2), cfg.rim_color, co)
    _box(stage, f"{prim_path}/handle", (cfg.handle_l, cfg.handle_w, cfg.handle_t),
         (cfg.handle_cx, 0.0, cfg.handle_t / 2), cfg.color, co)
    _box(stage, f"{prim_path}/knob", (cfg.knob_w, cfg.knob_w, cfg.knob_h),
         (cfg.knob_cx, 0.0, cfg.handle_t + cfg.knob_h / 2), cfg.knob_color, co)
    return root


def _spoon_spawner_cfg(*, bowl_l: float, bowl_w: float, floor_t: float, rim_t: float,
                       rim_h: float, handle_l: float, handle_w: float, handle_t: float,
                       knob_w: float, knob_h: float, bowl_cx: float, handle_cx: float,
                       knob_cx: float, mass: float, com_x: float, com_z: float,
                       color: tuple, rim_color: tuple, knob_color: tuple,
                       contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "spoon" not in _SPAWNER_CACHE:

        @configclass
        class KnifeEdgeSpoonSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_spoon)
            bowl_l: float = 0.048
            bowl_w: float = 0.048
            floor_t: float = 0.008
            rim_t: float = 0.004
            rim_h: float = 0.012
            handle_l: float = 0.150
            handle_w: float = 0.016
            handle_t: float = 0.008
            knob_w: float = 0.016
            knob_h: float = 0.022
            bowl_cx: float = -0.075
            handle_cx: float = 0.024
            knob_cx: float = 0.091
            mass: float = 0.0465
            com_x: float = -0.0202
            com_z: float = 0.0073
            color: tuple = (0.80, 0.80, 0.85)
            rim_color: tuple = (0.62, 0.62, 0.70)
            knob_color: tuple = (0.20, 0.55, 0.25)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["spoon"] = KnifeEdgeSpoonSpawnerCfg

    return _SPAWNER_CACHE["spoon"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        bowl_l=bowl_l, bowl_w=bowl_w, floor_t=floor_t, rim_t=rim_t, rim_h=rim_h,
        handle_l=handle_l, handle_w=handle_w, handle_t=handle_t, knob_w=knob_w,
        knob_h=knob_h, bowl_cx=bowl_cx, handle_cx=handle_cx, knob_cx=knob_cx,
        mass=mass, com_x=com_x, com_z=com_z, color=color, rim_color=rim_color,
        knob_color=knob_color, contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SpoonKnifeEdgeCfg(BaseCfg):
    """Config for `SpoonKnifeEdgeScene`. All balance-defining quantities (spoon mass,
    CoM, cube mass, the loaded balance point) are DERIVED here from the part
    dimensions and densities — a single source of truth shared by the spawners, the
    rubric, and the solver — and the honesty of the task is asserted in
    `__post_init__`: the support window is real, every naive placement anchor lies
    far outside it, both ends genuinely overhang, the cube fits the pocket and the
    Franka jaw, and a tipped spoon reads MANY times the level tolerance."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    tilt_max_deg: float = tunable(4.0)  # spoon long-axis tilt from horizontal: LEVEL
    roll_max_deg: float = tunable(10.0)  # spoon body-z tilt from vertical (no flips)
    perp_max_deg: float = tunable(35.0)  # spoon axis vs the fin NORMAL, in the xy plane
    # Rest gate = velocity thresholds AND a pose-stillness window. GPU PhysX reads a
    # persistent phantom velocity on marginally-stable contacts (spoon on the crest:
    # |w| ~ 0.2 rad/s, cube in the pocket: |v| ~ 0.08 m/s, with positions frozen to
    # sub-mm), so the velocity thresholds sit ABOVE that artifact band and the pose
    # window carries the honesty: settled() also requires every body's pose to have
    # moved less than settle_pos / settle_rot_deg over the last settle_win substeps.
    # A teleport write trivially breaks the window (the old slot holds the pre-write
    # pose), so freshly-written states cannot transiently judge as settled.
    # (the crest-contact phantom ang-vel readback spikes to ~0.47 rad/s with the pose
    # frozen; 0.80 rad/s over the 0.25 s window would sweep 11.5 deg — the pose
    # window rejects real rotation long before this clause matters)
    settle_lin: float = tunable(0.15)  # max |lin vel| of spoon+cube when judging (m/s)
    settle_ang: float = tunable(0.80)  # max |spoon ang vel| when judging (rad/s)
    settle_pos: float = tunable(0.0015)  # max pose drift over the window (m)
    settle_rot_deg: float = tunable(0.5)  # max spoon orientation drift over the window
    settle_win: int = tunable(30)  # window length in substeps (0.25 s at 120 Hz)
    end_air_z: float = tunable(0.045)  # both spoon END points must hang above this (m)
    min_overhang: float = tunable(0.030)  # each end beyond the crest by at least this
    perch_lo: float = tunable(-0.005)  # spoon origin z band about crest_top (stand frame)
    perch_hi: float = tunable(0.010)
    edge_margin: float = tunable(0.010)  # axis must cross the fin this far from its ends
    pocket_xy_tol: float = tunable(0.011)  # cube centre vs bowl centre, spoon frame (m)
    cube_z_lo: float = tunable(0.010)  # cube centre z band in the spoon frame:
    cube_z_hi: float = tunable(0.036)  # seated = floor_t + cube/2 = 0.021

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    stand_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the stand (m)
    stand_yaw_deg: float = tunable(20.0)  # uniform +/- stand yaw (judge in stand frame)
    slot_jitter: float = tunable(0.03)  # +/- xy jitter of the spoon/cube floor slots

    # --- tunable: placement ------------------------------------------------------------------
    stand_pos: tuple = tunable((0.45, 0.0))  # stand centre, WORLD xy nominal
    spoon_slot: tuple = tunable((0.20, -0.20))  # spoon floor slot, WORLD xy nominal
    cube_slot: tuple = tunable((0.20, 0.20))  # cube floor slot, WORLD xy nominal

    # --- info: stand -------------------------------------------------------------------------
    slab_x: float = info(0.20)
    slab_y: float = info(0.16)
    slab_t: float = info(0.020)
    fin_t: float = info(0.020)  # crest width (the support window, along stand x)
    fin_len: float = info(0.140)  # crest length (along stand y)
    fin_h: float = info(0.060)  # fin height above the slab top
    stand_mass: float = info(6.0)
    # --- info: spoon (all lengths along local x; underside plane = local z 0) ---------------
    bowl_l: float = info(0.048)
    bowl_w: float = info(0.048)
    floor_t: float = info(0.008)
    rim_t: float = info(0.004)
    rim_h: float = info(0.012)
    handle_l: float = info(0.150)
    handle_w: float = info(0.016)
    handle_t: float = info(0.008)
    knob_w: float = info(0.016)
    knob_h: float = info(0.022)
    spoon_density: float = info(900.0)  # kg/m^3 (dense polymer)
    # --- info: cube --------------------------------------------------------------------------
    cube_s: float = info(0.026)
    cube_density: float = info(2700.0)  # kg/m^3 (aluminium)
    jaw_span: float = info(0.080)  # the Franka parallel jaw (embodiment argument)
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__) — the single source of balance truth.
    total_l: float = field(default=None, init=False)
    half_l: float = field(default=None, init=False)
    bowl_cx: float = field(default=None, init=False)
    handle_cx: float = field(default=None, init=False)
    knob_cx: float = field(default=None, init=False)
    crest_top: float = field(default=None, init=False)
    crest_half: float = field(default=None, init=False)
    pocket_half: float = field(default=None, init=False)
    m_spoon: float = field(default=None, init=False)
    com_x: float = field(default=None, init=False)
    com_z: float = field(default=None, init=False)
    m_cube: float = field(default=None, init=False)
    bal_x: float = field(default=None, init=False)  # loaded balance point, spoon frame

    def __post_init__(self) -> None:
        # ---- geometry ----
        self.total_l = self.bowl_l + self.handle_l  # bowl and handle abut at -half + bowl_l
        self.half_l = self.total_l / 2
        self.bowl_cx = -self.half_l + self.bowl_l / 2
        self.handle_cx = self.half_l - self.handle_l / 2
        self.knob_cx = self.half_l - self.knob_w / 2
        self.crest_top = self.slab_t + self.fin_h
        self.crest_half = self.fin_t / 2
        self.pocket_half = self.bowl_l / 2 - self.rim_t

        # ---- spoon mass + CoM composed from the parts (mirrors the spawner exactly) ----
        rho = self.spoon_density
        parts = []  # (mass, cx, cz)
        parts.append((rho * self.bowl_l * self.bowl_w * self.floor_t,
                      self.bowl_cx, self.floor_t / 2))
        rim_z = self.floor_t + self.rim_h / 2
        m_rx = rho * self.rim_t * self.bowl_w * self.rim_h
        for s in (1.0, -1.0):
            parts.append((m_rx, self.bowl_cx + s * (self.bowl_l - self.rim_t) / 2, rim_z))
        m_ry = rho * (self.bowl_l - 2 * self.rim_t) * self.rim_t * self.rim_h
        parts.append((2 * m_ry, self.bowl_cx, rim_z))
        parts.append((rho * self.handle_l * self.handle_w * self.handle_t,
                      self.handle_cx, self.handle_t / 2))
        parts.append((rho * self.knob_w * self.knob_w * self.knob_h,
                      self.knob_cx, self.handle_t + self.knob_h / 2))
        self.m_spoon = sum(m for m, _, _ in parts)
        self.com_x = sum(m * cx for m, cx, _ in parts) / self.m_spoon
        self.com_z = sum(m * cz for m, _, cz in parts) / self.m_spoon
        self.m_cube = self.cube_density * self.cube_s**3
        # a seated cube's CoM sits at the bowl centre (slack-shift asserted below)
        self.bal_x = ((self.m_spoon * self.com_x + self.m_cube * self.bowl_cx)
                      / (self.m_spoon + self.m_cube))

        # ---- honesty asserts ----
        # the balance window is REAL: worst-case CoM shift from cube slack in the
        # pocket stays well inside the crest half-width
        slack = self.pocket_half - self.cube_s / 2
        assert slack >= 0.005, "cube must drop into the pocket with real slack"
        com_shift = self.m_cube * slack / (self.m_spoon + self.m_cube)
        assert com_shift <= 0.4 * self.crest_half, (
            "cube slack must not blur the balance window")
        # every naive placement anchor lies FAR outside the window
        assert abs(self.bal_x) >= 0.030, "geometric-centre placement must tip"
        assert abs(self.bal_x - self.com_x) >= 0.025, "empty-CoM placement must tip"
        assert abs(self.bal_x - self.bowl_cx) >= 0.025, "bowl-centre placement must tip"
        # a correct perch really overhangs on both sides
        over_bowl = self.half_l - abs(self.bal_x)
        over_handle = self.half_l + abs(self.bal_x)
        assert over_bowl >= self.min_overhang + self.crest_half + 0.005, (
            "bowl end must genuinely overhang")
        assert over_handle >= self.min_overhang + self.crest_half + 0.005, (
            "handle end must genuinely overhang")
        # ends in free air is a discriminating clause: crest high enough, slab low enough
        assert self.crest_top >= self.end_air_z + 0.030, "crest must sit above the air band"
        assert self.slab_t + 0.010 <= self.end_air_z, "slab rest must fail the air band"
        # a handle-side tip rests on the slab MANY times past the level band
        tip_deg = math.degrees(math.asin((self.crest_top - self.slab_t) / over_handle))
        assert tip_deg >= 3.0 * self.tilt_max_deg, "a tipped spoon must read clearly tipped"
        # cube fits the pocket at ANY yaw, and the seated pose sits inside the z band
        assert self.cube_s * math.sqrt(2.0) / 2 <= self.pocket_half - 0.001, (
            "cube must fit the pocket at any yaw")
        seated_z = self.floor_t + self.cube_s / 2
        assert self.cube_z_lo + 0.005 <= seated_z <= self.cube_z_hi - 0.005, (
            "seated cube must sit inside the z band")
        assert self.pocket_xy_tol >= slack + 0.003, "a seated cube must always pass xy tol"
        # graspability (embodiment): cube and knob fit the parallel jaw
        assert self.cube_s + 0.010 <= self.jaw_span, "cube must fit the parallel jaw"
        assert self.knob_w + 0.010 <= self.jaw_span, "knob must fit the parallel jaw"
        assert self.knob_h >= 0.015, "knob must stand proud enough to pinch"
        # the axis-crossing clause has room: fin long enough for the yaw randomization
        assert self.fin_len / 2 - self.edge_margin >= 0.045, "fin span must be usable"
        # floor slots stay clear of the stand across all jitters and any spoon yaw
        slab_half_diag = math.hypot(self.slab_x / 2, self.slab_y / 2)
        need = slab_half_diag + self.half_l \
            + (self.stand_jitter + self.slot_jitter) * math.sqrt(2.0) + 0.005
        for slot in (self.spoon_slot, self.cube_slot):
            d = math.hypot(slot[0] - self.stand_pos[0], slot[1] - self.stand_pos[1])
            assert d >= need, "floor slots must stay clear of the stand"
        # the two slots stay clear of each other
        dd = math.hypot(self.spoon_slot[0] - self.cube_slot[0],
                        self.spoon_slot[1] - self.cube_slot[1])
        assert dd >= self.half_l + self.cube_s + 2 * self.slot_jitter * math.sqrt(2.0) + 0.01, (
            "spoon and cube slots must not collide")


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("spoon_knife_edge")
class SpoonKnifeEdgeScene(BaseScene):
    cfg: SpoonKnifeEdgeCfg

    def __init__(self, cfg: SpoonKnifeEdgeCfg | None = None) -> None:
        super().__init__(cfg or SpoonKnifeEdgeCfg())

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
            "stand": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stand",
                spawn=_stand_spawner_cfg(
                    slab_x=c.slab_x, slab_y=c.slab_y, slab_t=c.slab_t, fin_t=c.fin_t,
                    fin_len=c.fin_len, fin_h=c.fin_h, mass=c.stand_mass,
                    slab_color=(0.35, 0.33, 0.30), fin_color=(0.85, 0.45, 0.10),
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stand_pos[0], c.stand_pos[1], 0.0)),
            ),
            "spoon": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Spoon",
                spawn=_spoon_spawner_cfg(
                    bowl_l=c.bowl_l, bowl_w=c.bowl_w, floor_t=c.floor_t, rim_t=c.rim_t,
                    rim_h=c.rim_h, handle_l=c.handle_l, handle_w=c.handle_w,
                    handle_t=c.handle_t, knob_w=c.knob_w, knob_h=c.knob_h,
                    bowl_cx=c.bowl_cx, handle_cx=c.handle_cx, knob_cx=c.knob_cx,
                    mass=c.m_spoon, com_x=c.com_x, com_z=c.com_z,
                    color=(0.80, 0.80, 0.85), rim_color=(0.62, 0.62, 0.70),
                    knob_color=(0.20, 0.55, 0.25), contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.spoon_slot[0], c.spoon_slot[1], 0.002)),
            ),
            "cube": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube",
                spawn=sim_utils.CuboidCfg(
                    size=(c.cube_s, c.cube_s, c.cube_s),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.70, 0.16, 0.12)),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.m_cube),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        linear_damping=0.1, angular_damping=0.5,
                        max_depenetration_velocity=0.5,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cube_slot[0], c.cube_slot[1], c.cube_s / 2 + 0.001)),
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
        self.stand: RigidObject = env.iscene["stand"]
        self.spoon: RigidObject = env.iscene["spoon"]
        self.cube: RigidObject = env.iscene["cube"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self._stand_xy = torch.tensor(self.cfg.stand_pos, device=dev).repeat(n, 1)
        self._stand_yaw = torch.zeros(n, device=dev)
        self._loaded = torch.zeros(n, dtype=torch.bool, device=dev)
        self._perched = torch.zeros(n, dtype=torch.bool, device=dev)
        self._succ_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        # pose-stillness ring buffer: spoon pos(3)+quat(4), cube pos(3), stand pos(3)
        self._hist_len = int(self.cfg.settle_win)
        self._hist = torch.zeros(self._hist_len, n, 13, device=dev)
        self._hist_i = 0
        self._hist_fill = torch.zeros(n, dtype=torch.long, device=dev)
        self._still_pos = torch.zeros(n, dtype=torch.bool, device=dev)

    # ----- reset ------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: stand teleported to a jittered xy + yaw, spoon flat on the
        floor at its jittered slot with a FREE yaw, cube on the floor at its jittered
        slot, latches cleared. Draws use `torch.rand` only (the first randint after a
        manual seed is degenerate on this stack)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, wxy: torch.Tensor, z: float, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = wxy
            st[:, 2] = z
            st[:, 3:7] = quat
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        sxy = torch.tensor(c.stand_pos, device=dev).expand(m, 2) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.stand_jitter
        syaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.stand_yaw_deg)
        self._stand_xy[env_ids] = sxy
        self._stand_yaw[env_ids] = syaw
        write(self.stand, sxy, 0.0, _qz(syaw))

        pxy = torch.tensor(c.spoon_slot, device=dev).expand(m, 2) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        pyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        write(self.spoon, pxy, 0.002, _qz(pyaw))

        kxy = torch.tensor(c.cube_slot, device=dev).expand(m, 2) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        write(self.cube, kxy, c.cube_s / 2 + 0.001, _qz(torch.zeros(m, device=dev)))

        self._loaded[env_ids] = False
        self._perched[env_ids] = False
        self._succ_ever[env_ids] = False
        self._hist_fill[env_ids] = 0
        self._still_pos[env_ids] = False

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "stand": self.stand.data.root_state_w[env_ids].clone(),
            "spoon": self.spoon.data.root_state_w[env_ids].clone(),
            "cube": self.cube.data.root_state_w[env_ids].clone(),
            "stand_xy": self._stand_xy[env_ids].clone(),
            "stand_yaw": self._stand_yaw[env_ids].clone(),
            "loaded": self._loaded[env_ids].clone(),
            "perched": self._perched[env_ids].clone(),
            "succ_ever": self._succ_ever[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.stand.write_root_state_to_sim(state["stand"], env_ids)
        self.spoon.write_root_state_to_sim(state["spoon"], env_ids)
        self.cube.write_root_state_to_sim(state["cube"], env_ids)
        self._stand_xy[env_ids] = state["stand_xy"]
        self._stand_yaw[env_ids] = state["stand_yaw"]
        self._loaded[env_ids] = state["loaded"]
        self._perched[env_ids] = state["perched"]
        self._succ_ever[env_ids] = state["succ_ever"]
        self._hist_fill[env_ids] = 0  # restored poses must re-earn stillness
        self._still_pos[env_ids] = False

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A KNIFE-EDGE BALANCE STAND rests near ({c.stand_pos[0]:.2f}, "
            f"{c.stand_pos[1]:.2f}) (its xy and yaw change per episode): a dark "
            f"{c.slab_x * 100:.0f} x {c.slab_y * 100:.0f} cm base slab carrying an "
            f"orange fin plate — {c.fin_t * 1000:.0f} mm thick, "
            f"{c.fin_len * 100:.0f} cm long — whose flat top edge (the CREST) runs "
            f"{c.crest_top * 100:.0f} cm above the floor. On the floor lie a silver "
            f"SERVING SPOON ({c.total_l * 100:.1f} cm long, ~{c.m_spoon * 1000:.0f} g, "
            f"position and yaw change per episode) and a small red METAL CUBE "
            f"({c.cube_s * 1000:.0f} mm, ~{c.m_cube * 1000:.0f} g — as heavy as the "
            f"whole spoon). The spoon's underside is perfectly flat; at one end is a "
            f"square BOWL pocket ({(2 * c.pocket_half) * 1000:.0f} mm inside, "
            f"{c.rim_h * 1000:.0f} mm rims) whose centre lies "
            f"{abs(c.bowl_cx) * 1000:.0f} mm from the spoon's midpoint, and at the "
            f"other end a raised green KNOB for pinch-grasping. The empty spoon's own "
            f"centre of mass sits about {abs(c.com_x) * 1000:.0f} mm bowl-ward of its "
            f"midpoint.\n"
            f"Goal: seat the cube fully inside the bowl pocket, then lay the LOADED "
            f"spoon across the fin crest so that it BALANCES: spoon axis roughly "
            f"perpendicular to the fin (within {c.perp_max_deg:.0f} deg), level "
            f"within {c.tilt_max_deg:.0f} deg, not rolled, both ends overhanging the "
            f"crest by at least {c.min_overhang * 100:.0f} cm on opposite sides and "
            f"hanging in FREE AIR (above {c.end_air_z * 100:.1f} cm — touching the "
            f"slab or floor fails), with the cube still seated and everything at "
            f"rest. The crest is only {c.fin_t * 1000:.0f} mm wide, so the spoon "
            f"holds only if its COMBINED centre of mass (spoon + cube) sits over the "
            f"crest: that balance line lies between the empty spoon's CoM and the "
            f"bowl centre, far from the spoon's midpoint — supporting the spoon at "
            f"its midpoint, at its empty CoM, or under the bowl centre all tip it "
            f"off. Compute the loaded balance point from the masses and geometry "
            f"above, place, and let physics verdict."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Put the red metal cube into the spoon's bowl pocket, then balance the "
            "loaded spoon level across the orange knife-edge fin so both ends hang "
            "free in the air. The cube weighs as much as the spoon, so the balance "
            "point is far from the spoon's centre — near the bowl side."
        )

    # ----- readings / rubric ------------------------------------------------------------------
    def _spoon_in_stand(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(N,3) spoon origin in the stand frame; (N,3) spoon +x axis in the stand
        frame. Everything is judged relative to the LIVE stand pose."""
        from isaaclab.utils.math import quat_apply, quat_rotate_inverse

        n = self.env.num_envs
        ex = torch.zeros(n, 3, device=self.env.device)
        ex[:, 0] = 1.0
        d = quat_rotate_inverse(self.stand.data.root_quat_w,
                                self.spoon.data.root_pos_w - self.stand.data.root_pos_w)
        a = quat_rotate_inverse(self.stand.data.root_quat_w,
                                quat_apply(self.spoon.data.root_quat_w, ex))
        return d, a

    def cube_in_spoon(self) -> torch.Tensor:
        """(N,3) cube centre in the SPOON body frame."""
        from isaaclab.utils.math import quat_rotate_inverse

        return quat_rotate_inverse(self.spoon.data.root_quat_w,
                                   self.cube.data.root_pos_w - self.spoon.data.root_pos_w)

    def in_pocket(self) -> torch.Tensor:
        """(N,) bool: cube seated in the bowl pocket, judged in the spoon frame (valid
        at any spoon pose)."""
        c = self.cfg
        d = self.cube_in_spoon()
        return ((d[:, 0] - c.bowl_cx).abs() <= c.pocket_xy_tol) \
            & (d[:, 1].abs() <= c.pocket_xy_tol) \
            & (d[:, 2] >= c.cube_z_lo) & (d[:, 2] <= c.cube_z_hi)

    def spoon_tilt(self) -> torch.Tensor:
        """(N,) |spoon long-axis| elevation from horizontal (rad, stand frame)."""
        _, a = self._spoon_in_stand()
        return torch.asin(a[:, 2].abs().clamp(max=1.0))

    def perched(self) -> torch.Tensor:
        """(N,) bool: the spoon rests ACROSS the crest — origin z in the crest band,
        long axis within perp_max of the fin normal, the axis crossing the fin plane
        inside the fin span, and both ends beyond the crest on OPPOSITE sides."""
        c = self.cfg
        d, a = self._spoon_in_stand()
        z_ok = (d[:, 2] >= c.crest_top + c.perch_lo) & (d[:, 2] <= c.crest_top + c.perch_hi)
        ax, ay = a[:, 0], a[:, 1]
        perp_ok = torch.atan2(ay.abs(), ax.abs()) <= math.radians(c.perp_max_deg)
        ax_safe = torch.where(ax.abs() < 1e-4, torch.full_like(ax, 1e-4), ax)
        t_star = -d[:, 0] / ax_safe
        y_cross = d[:, 1] + t_star * ay
        span_ok = (t_star.abs() <= c.half_l) \
            & (y_cross.abs() <= c.fin_len / 2 - c.edge_margin)
        e_hi = d[:, 0] + c.half_l * ax
        e_lo = d[:, 0] - c.half_l * ax
        over_ok = ((e_hi >= c.min_overhang) & (e_lo <= -c.min_overhang)) \
            | ((e_lo >= c.min_overhang) & (e_hi <= -c.min_overhang))
        return z_ok & perp_ok & span_ok & over_ok

    def level(self) -> torch.Tensor:
        return self.spoon_tilt() <= math.radians(self.cfg.tilt_max_deg)

    def roll_ok(self) -> torch.Tensor:
        """(N,) bool: spoon body z stays near world-up (no rolls or flips)."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.zeros(n, 3, device=self.env.device)
        ez[:, 2] = 1.0
        bz = quat_apply(self.spoon.data.root_quat_w, ez)
        return bz[:, 2] >= math.cos(math.radians(self.cfg.roll_max_deg))

    def ends_air(self) -> torch.Tensor:
        """(N,) bool: BOTH spoon end points hang above end_air_z (world) — resting an
        end on the slab or the floor fails this."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        ex = torch.zeros(n, 3, device=self.env.device)
        ex[:, 0] = 1.0
        aw = quat_apply(self.spoon.data.root_quat_w, ex)
        pz = self.spoon.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        e1 = pz + c.half_l * aw[:, 2]
        e2 = pz - c.half_l * aw[:, 2]
        return (e1 >= c.end_air_z) & (e2 >= c.end_air_z)

    def stand_upright(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.zeros(n, 3, device=self.env.device)
        ez[:, 2] = 1.0
        sz = quat_apply(self.stand.data.root_quat_w, ez)
        return sz[:, 2] >= math.cos(math.radians(5.0))

    def settled(self) -> torch.Tensor:
        """At rest = velocities below thresholds (set above the GPU phantom-velocity
        artifact band) AND every body's pose frozen over the last settle_win substeps
        (the pose window is the honest gate: falling, tipping, rocking and
        freshly-teleported states all move; only a genuine equilibrium is still)."""
        c = self.cfg
        return (self.spoon.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.cube.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.stand.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.spoon.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang) \
            & self._still_pos

    def _update_still(self) -> None:
        """Per-substep pose-stillness window (ring buffer of the last settle_win
        poses). A teleport write breaks the window for a full settle_win substeps —
        the old slot holds the pre-write pose."""
        c = self.cfg
        cur = torch.cat([self.spoon.data.root_pos_w, self.spoon.data.root_quat_w,
                         self.cube.data.root_pos_w, self.stand.data.root_pos_w], dim=-1)
        old = self._hist[self._hist_i]
        full = self._hist_fill >= self._hist_len
        d_sp = (cur[:, 0:3] - old[:, 0:3]).norm(dim=-1)
        dq = 2.0 * torch.acos(
            (cur[:, 3:7] * old[:, 3:7]).sum(dim=-1).abs().clamp(max=1.0))
        d_cu = (cur[:, 7:10] - old[:, 7:10]).norm(dim=-1)
        d_st = (cur[:, 10:13] - old[:, 10:13]).norm(dim=-1)
        self._still_pos = full & (d_sp < c.settle_pos) & (d_cu < c.settle_pos) \
            & (d_st < c.settle_pos) & (dq < math.radians(c.settle_rot_deg))
        self._hist[self._hist_i] = cur
        self._hist_i = (self._hist_i + 1) % self._hist_len
        self._hist_fill += 1

    def _update_latches(self) -> None:
        quiet = (self.spoon.data.root_lin_vel_w.norm(dim=-1) < 0.15) \
            & (self.cube.data.root_lin_vel_w.norm(dim=-1) < 0.15)
        pocket = self.in_pocket()
        self._loaded |= pocket & quiet
        self._perched |= pocket & self.perched() & quiet
        self._succ_ever |= self._success_now()

    def _success_now(self) -> torch.Tensor:
        return (self.in_pocket() & self.perched() & self.level() & self.roll_ok()
                & self.ends_air() & self.stand_upright() & self.settled())

    # ----- step-coupled bookkeeping (every substep) -------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """The verdict is fully passive (gravity + contact do the work) — post_step
        only advances the pose-stillness window and latches partial credit so
        transient progress keeps its score."""
        self._update_still()
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: cube seated in the bowl pocket, loaded spoon resting ACROSS the
        crest, level and unrolled, both ends overhanging in free air on opposite
        sides, stand upright, everything at rest. Physical settled outcomes only —
        judged in body frames."""
        self._update_latches()
        return self._success_now()

    def score(self) -> torch.Tensor:
        """Latched, monotone. 0.20 the cube was ever seated in the pocket (quietly);
        +0.30 the seated assembly ever rested spanning the crest; +0.20 success()
        ever held; capped at 0.70. Exactly 1.0 iff success() holds NOW."""
        self._update_latches()
        base = (0.20 * self._loaded.float() + 0.30 * self._perched.float()
                + 0.20 * self._succ_ever.float()).clamp(max=0.70)
        return torch.where(self._success_now(), torch.ones_like(base), base)


register_env("simgen", lambda: EnvCfg(scene="spoon_knife_edge", robot="null"))

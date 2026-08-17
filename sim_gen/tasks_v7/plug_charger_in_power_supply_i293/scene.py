"""TotePourDockScene — pour the power bank out of a deep slick tote onto the green
charging mat, centre it flat, and park the emptied tote clear of the mat (sim_gen
task `plug_charger_in_power_supply_i293`).

Derived from rlbench/plug_charger_in_power_supply, but STRATEGICALLY different: the
seed is a terminal INSERTION — pick a free charger off the table, align its prongs,
and push it INTO a fixed wall socket; the charger is always directly graspable and
the episode is about reaching one inserted pose. Here there is no insertion, no
receptacle, no prongs, and — crucially — the judged object is NEVER directly
graspable at reset:

  1. the power bank starts flat on the floor of a deep, narrow, open-top tote
     (interior 120 x 82 mm, 130 mm deep). The mouth is narrower than a parallel-jaw
     hand's palm and the bank lies ~108 mm below the rim — far beyond finger length
     (asserted in cfg) — so the bank cannot be pinched, hooked, or poked in situ;
  2. the only way to free it is to manipulate the CONTAINER: grasp the tote by its
     rim, lift it over the charging mat, and invert it PAST vertical so the bank
     pours out under gravity (the tote interior is deliberately slick, so the pour
     releases at a knowable tilt — physics that must be exploited, not fought);
  3. the poured bank must end up lying FLAT near the mat centre (fine-positioned
     with pushes if the pour lands it off-centre), and
  4. the emptied tote must be PARKED well clear of the mat — leaving the tote on or
     over the mat fails, so "dump and drop" is not a solution.

So a solver needs a different PLAN (indirect manipulation of the payload through
its container, an airborne re-orientation past vertical, a placement, and a
tidy-away) and a different code structure (a tilt-progress latch gated on the bank
still being inside, a freed latch, a placement latch, and a container-clearance
predicate) — not "align a peg with a hole and push". The pour direction, the mat
location (a random ring around the tote), and every yaw vary per episode.

success() iff, settled (tote AND bank |v| < settle_lin):
  - the bank lies FLAT inside the mat's centre patch (mat frame, per-axis
    `place_xy`, origin height in the flat-rest band — an on-edge bank, a bank
    perched on the parked tote, or a bank off the mat all fail);
  - the tote is CLEAR of the mat: its centre's mat-frame Chebyshev distance
    exceeds `clear_cheb` (per-axis, so the footprints cannot overlap — asserted).

score() is latched every physics substep (credit never evaporates):
  0.20 * best tilt progress (tote tilt from upright toward `tip_ref_deg`, gated on
         the bank actually being inside the tote — tilting an empty tote earns 0)
+ 0.35 * freed (the bank has left the tote interior)
+ 0.30 * placed (the bank settled flat in the mat's centre patch),
capped at 0.85; exactly 1.0 iff success(). Doing nothing scores ~0, and the seed's
whole strategy (get the payload seated in a receptacle by direct grasp-and-push)
is physically unavailable: there is no receptacle and no direct grasp.

Assets are fully procedural (no external files):
  - tote: DYNAMIC open-top box shell, interior 120 x 82 x 130 mm, 8 mm walls and
    floor (outer 136 x 98 x 138 mm), 0.40 kg, SLICK physics material
    (mu 0.10/0.08) so the pour is low-friction and repeatable;
  - bank: DYNAMIC power bank, a single 90 x 55 x 22 mm brick, 0.20 kg, grippy
    (mu 0.45/0.40) so it stays where placed;
  - mat: green 200 x 200 mm charging mat, KINEMATIC and VISUAL-ONLY (no collider:
    an in-path goal marker with a proud collider edge would wall the final pushes).
Contact offsets are explicit (1 mm) and both materials have restitution 0.

Per-episode randomization (verified by readback in smoke): tote xy + free yaw,
bank position/yaw jitter inside the tote, mat on a random ring (azimuth, distance,
free yaw) around the tote. Heavy imports (isaaclab, pxr) are deferred so importing
this module — and registering the scene — stays app-free.
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


def _box(stage, path: str, size, center, color, contact_offset: float | None):
    """Axis-aligned box prim; collider only if `contact_offset` is not None."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        _collide(seg.GetPrim(), contact_offset)
    return seg.GetPrim()


def _phys_material(stage, path: str, mu_s: float, mu_d: float):
    """Author a UsdShade physics material (restitution 0). Custom spawn funcs get
    no cfg schemas applied, so friction MUST be authored here or PhysX falls back
    to defaults."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(mu_s))
    api.CreateDynamicFrictionAttr(float(mu_d))
    api.CreateRestitutionAttr(0.0)
    return mat


def _bind_phys(prim, mat) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _rigid_dynamic(root, mass: float) -> None:
    """Dynamic rigid-body armor on a compound root: explicit mass, damping so parts
    settle promptly, velocity iters 4 (kills slow contact-creep artifacts), no
    sleeping while we judge velocities, and the depenetration cap."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.10)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _spawn_tote(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC open-top tote at `prim_path`. Origin = outer base centre
    at floor level, local +z up, the mouth at the top. Floor slab + four walls
    around the interior; every part bound to the SLICK physics material."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.mu_s, cfg.mu_d)

    co, color = cfg.contact_offset, cfg.color
    ix, iy, dp = cfg.ix, cfg.iy, cfg.depth
    wt, ft = cfg.wall_t, cfg.floor_t
    ox, oy = ix + 2 * wt, iy + 2 * wt
    wz = ft + dp / 2  # wall centre z
    parts = [
        _box(stage, f"{prim_path}/floor", (ox, oy, ft), (0.0, 0.0, ft / 2), color, co),
        _box(stage, f"{prim_path}/wall_xp", (wt, oy, dp), (ix / 2 + wt / 2, 0.0, wz), color, co),
        _box(stage, f"{prim_path}/wall_xn", (wt, oy, dp), (-(ix / 2 + wt / 2), 0.0, wz), color, co),
        _box(stage, f"{prim_path}/wall_yp", (ix, wt, dp), (0.0, iy / 2 + wt / 2, wz), color, co),
        _box(stage, f"{prim_path}/wall_yn", (ix, wt, dp), (0.0, -(iy / 2 + wt / 2), wz), color, co),
    ]
    for p in parts:
        _bind_phys(p, mat)
    return root


def _spawn_bank(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC power bank at `prim_path`. Origin = BODY CENTRE (flat
    rest height = lz/2). One grippy brick + a small visual-only face stripe so its
    orientation reads in renders."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.mu_s, cfg.mu_d)

    body = _box(stage, f"{prim_path}/body", (cfg.lx, cfg.ly, cfg.lz),
                (0.0, 0.0, 0.0), cfg.color, cfg.contact_offset)
    _bind_phys(body, mat)
    # visual stripe (NO collider) on the local +z face
    _box(stage, f"{prim_path}/stripe", (cfg.lx * 0.6, cfg.ly * 0.25, 0.0006),
         (0.0, 0.0, cfg.lz / 2 + 0.0003), cfg.stripe_color, None)
    return root


def _spawn_mat(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC, VISUAL-ONLY charging mat at `prim_path`: a thin green
    square + a lighter centre patch, with NO colliders (a proud collider edge in
    the push path would wall the bank's final centimetres)."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(1.0)

    h = cfg.half
    _box(stage, f"{prim_path}/pad", (2 * h, 2 * h, cfg.thick),
         (0.0, 0.0, cfg.thick / 2), cfg.color, None)
    _box(stage, f"{prim_path}/centre", (0.06, 0.06, cfg.thick),
         (0.0, 0.0, cfg.thick / 2 + 0.0002), cfg.centre_color, None)
    return root


def _tote_spawner_cfg(**kw: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tote" not in _SPAWNER_CACHE:

        @configclass
        class ToteSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tote)
            ix: float = 0.120
            iy: float = 0.082
            depth: float = 0.130
            wall_t: float = 0.008
            floor_t: float = 0.008
            mass: float = 0.40
            mu_s: float = 0.10
            mu_d: float = 0.08
            color: tuple = (0.28, 0.32, 0.60)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["tote"] = ToteSpawnerCfg

    return _SPAWNER_CACHE["tote"](
        mass_props=sim_utils.MassPropertiesCfg(mass=kw["mass"]),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        **kw,
    )


def _bank_spawner_cfg(**kw: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "bank" not in _SPAWNER_CACHE:

        @configclass
        class BankSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bank)
            lx: float = 0.090
            ly: float = 0.055
            lz: float = 0.022
            mass: float = 0.20
            mu_s: float = 0.45
            mu_d: float = 0.40
            color: tuple = (0.12, 0.12, 0.14)
            stripe_color: tuple = (0.85, 0.55, 0.10)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["bank"] = BankSpawnerCfg

    return _SPAWNER_CACHE["bank"](
        mass_props=sim_utils.MassPropertiesCfg(mass=kw["mass"]),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        **kw,
    )


def _mat_spawner_cfg(**kw: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "mat" not in _SPAWNER_CACHE:

        @configclass
        class ChargeMatSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_mat)
            half: float = 0.100
            thick: float = 0.0016
            color: tuple = (0.16, 0.55, 0.22)
            centre_color: tuple = (0.55, 0.85, 0.45)

        _SPAWNER_CACHE["mat"] = ChargeMatSpawnerCfg

    return _SPAWNER_CACHE["mat"](
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        **kw,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class TotePourDockSceneCfg(BaseCfg):
    """Config for `TotePourDockScene`. Honesty knobs asserted in `__post_init__`:
    the bank is genuinely unreachable in situ (deep + narrow), the pour genuinely
    releases below the reference hold tilt (friction math), the placement band
    genuinely means "the whole bank flat on the mat", and a tote that satisfies
    `clear_cheb` genuinely cannot overlap the mat or touch a placed bank."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    place_xy: float = tunable(0.044)  # bank origin within this of the mat axis, per axis (m)
    place_z_lo: float = tunable(0.004)  # ... and origin height in the flat-rest band (m)
    place_z_hi: float = tunable(0.018)
    place_flat_deg: float = tunable(20.0)  # bank thin axis within this of world +/-z (either face)
    clear_cheb: float = tunable(0.19)  # tote centre's mat-frame per-axis (Chebyshev) clearance (m)
    settle_lin: float = tunable(0.06)  # max |lin vel| (tote AND bank) when judging (m/s)
    tip_ref_deg: float = tunable(100.0)  # tilt for full tip credit (below the pour-release tilt)
    tip_gate_expand: float = tunable(0.02)  # tip credit gated on bank inside (box grown by this)
    freed_expand: float = tunable(0.004)  # freed = bank origin OUTSIDE the box grown by this

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    tote_jitter: float = tunable(0.04)  # uniform +/- xy jitter of the tote at reset (m)
    tote_yaw_deg: float = tunable(180.0)  # uniform +/- tote yaw (free)
    bank_dx: float = tunable(0.006)  # bank in-tote local x jitter (m)
    bank_dy: float = tunable(0.004)  # bank in-tote local y jitter (m)
    bank_yaw_deg: float = tunable(6.0)  # bank yaw jitter relative to the tote (deg)
    mat_dist_lo: float = tunable(0.26)  # mat centre distance from the tote: U(lo, hi) (m)
    mat_dist_hi: float = tunable(0.32)
    mat_yaw_deg: float = tunable(180.0)  # uniform +/- mat yaw (free)

    # --- tunable: placement ------------------------------------------------------------------
    tote_pos: tuple = tunable((-0.05, 0.05))  # tote base centre, nominal

    # --- info: tote structure ----------------------------------------------------------------
    ix: float = info(0.120)  # interior x (the bank's 105.6 mm diagonal cannot wedge: asserted)
    iy: float = info(0.082)  # interior y (narrower than a palm: asserted)
    depth: float = info(0.130)  # interior depth (bank top ~108 mm below the rim: asserted)
    wall_t: float = info(0.008)
    floor_t: float = info(0.008)
    tote_mass: float = info(0.40)
    tote_mu_s: float = info(0.10)  # slick interior: the pour releases at a knowable tilt
    tote_mu_d: float = info(0.08)
    # --- info: bank structure ----------------------------------------------------------------
    bank_lx: float = info(0.090)
    bank_ly: float = info(0.055)
    bank_lz: float = info(0.022)  # thin axis = local z; flat rest height lz/2
    bank_mass: float = info(0.20)
    bank_mu_s: float = info(0.45)
    bank_mu_d: float = info(0.40)
    # --- info: mat + ground ------------------------------------------------------------------
    mat_half: float = info(0.100)
    mat_thick: float = info(0.0016)  # visual only — NO collider
    mat_reach: float = info(0.36)  # mat centre kept within this of the world origin (reach cap)
    ground_mu_s: float = info(0.55)
    ground_mu_d: float = info(0.45)
    # --- info: embodiment + solve references -------------------------------------------------
    finger_len: float = info(0.053)  # Franka finger length: cannot reach the bank (asserted)
    palm_w: float = info(0.088)  # Franka palm/knuckle width: cannot enter the mouth (asserted)
    pour_hold_deg: float = info(115.0)  # solve's hold tilt; above the release tilt (asserted)
    # --- info: colors + misc -----------------------------------------------------------------
    tote_color: tuple = info((0.28, 0.32, 0.60))
    bank_color: tuple = info((0.12, 0.12, 0.14))
    stripe_color: tuple = info((0.85, 0.55, 0.10))
    mat_color: tuple = info((0.16, 0.55, 0.22))
    mat_centre_color: tuple = info((0.55, 0.85, 0.45))
    contact_offset: float = info(0.001)

    # Derived (filled in __post_init__).
    outer_x: float = field(default=None, init=False)
    outer_y: float = field(default=None, init=False)
    tote_h: float = field(default=None, init=False)  # outer height = mouth (rim) z, tote frame
    tote_circ_xy: float = field(default=None, init=False)  # footprint circumradius
    bank_circ_xy: float = field(default=None, init=False)
    mu_pair_s: float = field(default=None, init=False)  # PhysX pair friction = mean of both

    def __post_init__(self) -> None:
        self.outer_x = self.ix + 2 * self.wall_t
        self.outer_y = self.iy + 2 * self.wall_t
        self.tote_h = self.floor_t + self.depth
        self.tote_circ_xy = math.hypot(self.outer_x / 2, self.outer_y / 2)
        self.bank_circ_xy = math.hypot(self.bank_lx / 2, self.bank_ly / 2)
        self.mu_pair_s = 0.5 * (self.tote_mu_s + self.bank_mu_s)

        bank_diag = math.hypot(self.bank_lx, self.bank_ly)
        # -- the pour cannot wedge, and the mouth genuinely passes the bank --
        assert bank_diag + 0.010 <= self.ix, (
            "bank footprint diagonal must clear the interior x (no wedging during the pour)")
        assert self.bank_lx + 0.006 <= self.ix and self.bank_ly + 0.006 <= self.iy, (
            "bank must pass the mouth flat with real clearance")
        # -- the bank is genuinely unreachable in situ (container manipulation is mandatory) --
        assert self.depth - self.bank_lz >= 0.100, (
            "bank top must lie far below the rim (deeper than any finger)")
        assert self.finger_len + 0.045 <= self.depth - self.bank_lz, (
            "finger length + margin must not reach the bank")
        assert self.iy < self.palm_w, "mouth must be narrower than the palm (no hand entry)"
        # -- the pour physics is honest: slick interior releases below the hold tilt --
        release_deg = 90.0 + math.degrees(math.atan(self.mu_pair_s))
        assert self.pour_hold_deg >= release_deg + 4.0, (
            "hold tilt must exceed the friction release tilt with margin")
        assert self.tip_ref_deg <= release_deg, (
            "full tip credit must be reachable while the bank is still inside")
        # -- placement band honesty: flat rest in-band, edge rest out, whole bank on the mat --
        assert self.place_z_lo < self.bank_lz / 2 < self.place_z_hi, (
            "flat-rest height must sit inside the placed band")
        assert self.place_z_hi < self.bank_ly / 2 - 0.004, (
            "placed band must reject an on-edge bank")
        assert self.place_xy + self.bank_circ_xy <= self.mat_half - 0.002, (
            "a placed bank must lie entirely on the mat")
        # -- clearance honesty: a clear tote cannot overlap the mat or touch a placed bank --
        assert self.clear_cheb >= self.mat_half + self.tote_circ_xy + 0.005, (
            "per-axis clearance must forbid tote/mat footprint overlap")
        assert self.clear_cheb - self.tote_circ_xy >= self.place_xy + self.bank_circ_xy + 0.005, (
            "a clear tote must be unable to touch a placed bank")
        # -- reset geometry: mat ring never overlaps the tote; fallback azimuth stays in reach --
        assert self.mat_dist_lo >= self.mat_half * math.sqrt(2.0) + self.tote_circ_xy + 0.01, (
            "mat ring must keep the mat clear of the tote at reset")
        assert self.mat_dist_hi <= self.mat_reach - 0.02, (
            "toward-origin fallback azimuth must keep the mat inside the reach cap")
        # -- bank spawn keeps ~5 mm to the tote walls at worst-case jitter + yaw --
        yaw = math.radians(self.bank_yaw_deg)
        ext_x = (self.bank_lx * math.cos(yaw) + self.bank_ly * math.sin(yaw)) / 2
        ext_y = (self.bank_ly * math.cos(yaw) + self.bank_lx * math.sin(yaw)) / 2
        assert ext_x + self.bank_dx <= self.ix / 2 - 0.005, "bank spawn x clearance to walls"
        assert ext_y + self.bank_dy <= self.iy / 2 - 0.004, "bank spawn y clearance to walls"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("tote_pour_dock")
class TotePourDockScene(BaseScene):
    cfg: TotePourDockSceneCfg

    def __init__(self, cfg: TotePourDockSceneCfg | None = None) -> None:
        super().__init__(cfg or TotePourDockSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.ground_mu_s, dynamic_friction=c.ground_mu_d,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "tote": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tote",
                spawn=_tote_spawner_cfg(
                    ix=c.ix, iy=c.iy, depth=c.depth, wall_t=c.wall_t, floor_t=c.floor_t,
                    mass=c.tote_mass, mu_s=c.tote_mu_s, mu_d=c.tote_mu_d, color=c.tote_color,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.tote_pos[0], c.tote_pos[1], 0.001)),
            ),
            "bank": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bank",
                spawn=_bank_spawner_cfg(
                    lx=c.bank_lx, ly=c.bank_ly, lz=c.bank_lz, mass=c.bank_mass,
                    mu_s=c.bank_mu_s, mu_d=c.bank_mu_d, color=c.bank_color,
                    stripe_color=c.stripe_color, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.tote_pos[0], c.tote_pos[1], c.floor_t + c.bank_lz / 2 + 0.002)),
            ),
            "mat": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Mat",
                spawn=_mat_spawner_cfg(half=c.mat_half, thick=c.mat_thick, color=c.mat_color,
                                       centre_color=c.mat_centre_color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.05, 0.0)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # external wrenches (the solve's pour hold / nudge, smoke's probes) must
                # act every solver iteration or the PD rings/stalls on this pod
                "enable_external_forces_every_iteration": True,
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
        self.tote: RigidObject = env.iscene["tote"]
        self.bank: RigidObject = env.iscene["bank"]
        self.mat: RigidObject = env.iscene["mat"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # progress latches
        self.tip_latch = torch.zeros(n, device=dev)
        self.freed_latch = torch.zeros(n, device=dev)
        self.place_latch = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: tote with xy jitter + free yaw, bank written flat on the
        tote floor with in-tote jitter + relative yaw, mat on a random ring
        (azimuth, distance, free yaw) around the tote — resampled (with a
        deterministic toward-origin fallback) to stay inside the reach cap;
        latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # first post-seed draws are near-degenerate across seeds — burn them
        _ = torch.rand(3, m, device=dev)

        # --- tote: xy jitter + free yaw ---
        tote_xy = torch.tensor(c.tote_pos, device=dev).expand(m, 2).clone()
        tote_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.tote_jitter
        tote_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.tote_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = tote_xy
        st[:, 2] = 0.001
        st[:, 3] = torch.cos(tote_yaw / 2)
        st[:, 6] = torch.sin(tote_yaw / 2)
        st[:, 0:3] += origin
        self.tote.write_root_state_to_sim(st, env_ids)

        # --- bank: flat on the tote floor, in-tote jitter + relative yaw ---
        lx = (torch.rand(m, device=dev) * 2 - 1) * c.bank_dx
        ly = (torch.rand(m, device=dev) * 2 - 1) * c.bank_dy
        byaw = tote_yaw + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.bank_yaw_deg)
        ca, sa = torch.cos(tote_yaw), torch.sin(tote_yaw)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = tote_xy[:, 0] + ca * lx - sa * ly
        st[:, 1] = tote_xy[:, 1] + sa * lx + ca * ly
        st[:, 2] = 0.001 + c.floor_t + c.bank_lz / 2 + 0.0015
        st[:, 3] = torch.cos(byaw / 2)
        st[:, 6] = torch.sin(byaw / 2)
        st[:, 0:3] += origin
        self.bank.write_root_state_to_sim(st, env_ids)

        # --- mat: random ring around the tote, kept inside the reach cap ---
        dist = c.mat_dist_lo + torch.rand(m, device=dev) * (c.mat_dist_hi - c.mat_dist_lo)
        az = torch.rand(m, device=dev) * 2 * math.pi
        mat_xy = tote_xy + dist[:, None] * torch.stack([torch.cos(az), torch.sin(az)], dim=-1)
        for _ in range(12):
            bad = mat_xy.norm(dim=-1) > c.mat_reach
            if not bad.any():
                break
            k = int(bad.sum())
            az_new = torch.rand(k, device=dev) * 2 * math.pi
            mat_xy[bad] = tote_xy[bad] + dist[bad, None] * torch.stack(
                [torch.cos(az_new), torch.sin(az_new)], dim=-1)
        bad = mat_xy.norm(dim=-1) > c.mat_reach
        if bad.any():  # deterministic fallback: place the mat on the origin side of the tote
            u = -tote_xy[bad] / tote_xy[bad].norm(dim=-1, keepdim=True).clamp_min(1e-6)
            mat_xy[bad] = tote_xy[bad] + dist[bad, None] * u
        mat_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.mat_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = mat_xy
        st[:, 3] = torch.cos(mat_yaw / 2)
        st[:, 6] = torch.sin(mat_yaw / 2)
        st[:, 0:3] += origin
        self.mat.write_root_state_to_sim(st, env_ids)

        # --- latches ---
        self.tip_latch[env_ids] = 0.0
        self.freed_latch[env_ids] = 0.0
        self.place_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "tote": self.tote.data.root_state_w[env_ids].clone(),
            "bank": self.bank.data.root_state_w[env_ids].clone(),
            "mat": self.mat.data.root_state_w[env_ids].clone(),
            "tip_latch": self.tip_latch[env_ids].clone(),
            "freed_latch": self.freed_latch[env_ids].clone(),
            "place_latch": self.place_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.tote.write_root_state_to_sim(state["tote"], env_ids)
        self.bank.write_root_state_to_sim(state["bank"], env_ids)
        self.mat.write_root_state_to_sim(state["mat"], env_ids)
        self.tip_latch[env_ids] = state["tip_latch"]
        self.freed_latch[env_ids] = state["freed_latch"]
        self.place_latch[env_ids] = state["place_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A deep blue open-top tote ({c.outer_x * 1000:.0f} x {c.outer_y * 1000:.0f} mm "
            f"footprint, {c.tote_h * 1000:.0f} mm tall, slick {c.wall_t * 1000:.0f} mm walls) "
            f"stands on the floor. A black power bank "
            f"({c.bank_lx * 1000:.0f} x {c.bank_ly * 1000:.0f} x {c.bank_lz * 1000:.0f} mm, "
            f"orange stripe on top) lies FLAT on the tote's floor, about "
            f"{(c.depth - c.bank_lz) * 1000:.0f} mm below the rim. The mouth "
            f"({c.ix * 1000:.0f} x {c.iy * 1000:.0f} mm) is too narrow for a hand and the "
            f"bank lies too deep for fingers, so the bank CANNOT be grasped, hooked, or "
            f"poked where it is — the only way to get it out is to pick the tote up by its "
            f"rim and POUR: tilt it past vertical over the target until the bank slides out "
            f"of the mouth under gravity. A flat green charging mat "
            f"({2 * c.mat_half * 1000:.0f} mm square, light-green centre patch) lies on the "
            f"floor {c.mat_dist_lo * 100:.0f}-{c.mat_dist_hi * 100:.0f} cm away in a random "
            f"direction.\n"
            f"Goal: the bank must end up lying FLAT on the mat with its centre within "
            f"{c.place_xy * 1000:.0f} mm of the mat centre (either face up; on-edge fails), "
            f"and the tote must be set down CLEAR of the mat — its centre at least "
            f"{c.clear_cheb * 100:.0f} cm from the mat centre along every mat axis. Pour the "
            f"bank out over the mat, nudge it to the centre patch if the pour lands it "
            f"off-centre, and park the empty tote well away.\n"
            f"Judged only when everything has settled: a bank still in the tote, on edge, "
            f"off the mat, or off-centre counts for nothing; so does a perfectly placed "
            f"bank with the tote left on or over the mat."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pick up the tote, pour the power bank out onto the green charging mat, "
            "nudge it flat onto the centre patch, then set the empty tote down well "
            "clear of the mat."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _tote_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> tote body frame (origin = outer base centre)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.tote.data.root_quat_w,
                                  p_w - self.tote.data.root_pos_w)

    def _mat_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> mat body frame (origin = pad centre, floor level)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.mat.data.root_quat_w,
                                  p_w - self.mat.data.root_pos_w)

    def _up_z(self, body) -> torch.Tensor:
        """(N,) world-z component of the body's local +z."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(body.data.root_quat_w, ez)[:, 2]

    # ----- predicates -------------------------------------------------------------------------
    def bank_in_tote(self, expand: float) -> torch.Tensor:
        """(N,) bool: the bank's origin inside the tote's interior box grown by
        `expand` on every face (tote body frame — valid at any tote tilt)."""
        c = self.cfg
        loc = self._tote_local(self.bank.data.root_pos_w)
        return ((loc[:, 0].abs() < c.ix / 2 + expand)
                & (loc[:, 1].abs() < c.iy / 2 + expand)
                & (loc[:, 2] > c.floor_t - 0.004 - expand)
                & (loc[:, 2] < c.tote_h + expand))

    def tote_tilt(self) -> torch.Tensor:
        """(N,) rad: tote tilt from upright (angle of local +z from world +z)."""
        return torch.acos(self._up_z(self.tote).clamp(-1.0, 1.0))

    def tip_frac(self) -> torch.Tensor:
        """(N,) in [0,1]: tilt progress toward `tip_ref_deg`, gated on the bank
        actually being inside the tote — tilting an empty tote earns nothing."""
        c = self.cfg
        frac = (self.tote_tilt() / math.radians(c.tip_ref_deg)).clamp(0.0, 1.0)
        return frac * self.bank_in_tote(c.tip_gate_expand).float()

    def bank_freed(self) -> torch.Tensor:
        """(N,) bool: the bank has left the tote interior."""
        return ~self.bank_in_tote(self.cfg.freed_expand)

    def bank_flat(self) -> torch.Tensor:
        """(N,) bool: bank thin axis within `place_flat_deg` of world up/down."""
        return self._up_z(self.bank).abs() >= math.cos(math.radians(self.cfg.place_flat_deg))

    def bank_placed(self) -> torch.Tensor:
        """(N,) bool: the bank lying FLAT in the mat's centre patch (mat frame,
        per-axis `place_xy`, origin height in the flat-rest band — on-edge banks,
        banks perched on the tote, and airborne banks all fail)."""
        c = self.cfg
        loc = self._mat_local(self.bank.data.root_pos_w)
        return ((loc[:, 0].abs() < c.place_xy) & (loc[:, 1].abs() < c.place_xy)
                & (loc[:, 2] > c.place_z_lo) & (loc[:, 2] < c.place_z_hi)
                & self.bank_flat())

    def tote_clear(self) -> torch.Tensor:
        """(N,) bool: the tote centre's mat-frame per-axis (Chebyshev) distance
        exceeds `clear_cheb` — by the cfg asserts this forbids any tote/mat
        footprint overlap and any contact with a placed bank."""
        loc = self._mat_local(self.tote.data.root_pos_w)
        return torch.maximum(loc[:, 0].abs(), loc[:, 1].abs()) > self.cfg.clear_cheb

    def settled(self) -> torch.Tensor:
        """(N,) bool: tote AND bank |lin vel| below `settle_lin`."""
        c = self.cfg
        return ((self.tote.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.bank.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin))

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch tilt progress, pour-out, and settled placement each physics
        substep, so transient progress keeps its credit."""
        self.tip_latch = torch.maximum(self.tip_latch, self.tip_frac())
        self.freed_latch = torch.maximum(self.freed_latch, self.bank_freed().float())
        placed_now = self.bank_placed() \
            & (self.bank.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin)
        self.place_latch = torch.maximum(self.place_latch, placed_now.float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: bank settled flat in the mat's centre patch + tote parked
        clear of the mat, everything settled."""
        return self.bank_placed() & self.tote_clear() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.20 * latched gated tilt + 0.35 * freed +
        0.30 * placed, capped at 0.85; exactly 1.0 iff success(). Doing nothing
        scores ~0; a perfect pour-and-place with the tote dumped on the mat caps
        at 0.85."""
        base = (0.20 * self.tip_latch + 0.35 * self.freed_latch
                + 0.30 * self.place_latch).clamp(0.0, 0.85)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="tote_pour_dock", robot="null"))

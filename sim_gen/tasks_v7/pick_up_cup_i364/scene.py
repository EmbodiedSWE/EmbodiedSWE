"""CooperCurveScene — identify the TAPERED roller-cup, stage it, and deliver it through a
curved roofed rolling gallery by taper-steered rolling (sim_gen task `pick_up_cup_i364`).

Derived from rlbench/pick_up_cup ("pick up the cup": grasp the one free cup and lift it
straight up, judged by height). Here NOTHING is lifted to succeed and grasp-and-raise is
worth zero: the manipulation model is replaced wholesale by ROLLING TRANSPORT THROUGH A
CURVE THAT ONLY THE OBJECT'S OWN TAPER CAN STEER. The cup lies on its side and is a
conical wheelset — a small BASE rim (r=30 mm) and a large MOUTH rim (r=45 mm) joined by
a thin waist. Rolling on its rims it cannot go straight: it orbits the apex of its own
cone (220 mm beyond the base rim). The arena exploits exactly that: a START PEN and a
CATCH BAY are connected only by a 150 deg annular GALLERY whose roof (underside 115 mm)
is far below the cup's standing height (185 mm) and admits only a body rolling on its
side (rolling profile 90 mm). The gallery's curvature matches the cup's natural turning
circle. A same-length DECOY roller with EQUAL rims (37.5 mm / 37.5 mm) rolls dead
straight — launched into the gallery it wedges against the outer wall long before the
bay (verified as a smoke probe): the taper is load-bearing, not decoration.

What the solver must bring, none of which exists in the seed:
  (1) perception of TAPER identity — cup and decoy swap spawn slots per episode and
      differ only in rim profile;
  (2) STAGING: park the cup lying at the gallery mouth with its BASE toward the arc
      centre, so its cone apex falls on the gallery's centre — a pose, not a height;
  (3) an aimed, force-capped LAUNCH: one tangential push below the friction budget,
      then hands-off — the taper steers the coast through 150 deg of curve;
  (4) restraint: the decoy must stay out of the catch bay.

Assets are fully procedural (compound spawners; children of one body never collide):
  - arena: STATIC ring-segment walls (300 mm tall) enclosing start pen (azimuth -160
    to -75 deg), gallery (-75..75 deg, roofed 115..135 mm), catch bay (75..115 deg,
    radial arrest wall at 115 deg); channel between wall radii 155 / 420 mm.
  - cup: DYNAMIC wheelset — base sphere r 30 mm at x=-55 mm, mouth sphere r 45 mm at
    x=+55 mm, capsule waist r 20 mm (250 g, authored mass/inertia/material).
  - decoy: same construction, both rims r 37.5 mm (250 g).

Per-episode randomization (readback-verifiable): cup/decoy permuted over two pen
slots, polar jitter (+-3 deg azimuth, +-12 mm radius) and free yaw each.

Rubric (0..1; ordered latched credit anchored in the demonstrated solve trajectory):
  0.10 * staged — cup ever parked in the staging zone, lying, base toward the centre
  0.20 * cp1    — staged, then cup crossed gallery azimuth -45 deg at floor level
  0.20 * cp2    — ... then crossed 0 deg          (latch chain: each needs the last)
  0.15 * cp3    — ... then crossed +45 deg
  0.10 * bay    — ... then entered the catch bay
  1.0 iff success() — full latch chain AND cup at rest lying in the catch bay, decoy
                    out of the bay, all finite. Non-success capped at 0.75.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
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


# ----- USD authoring helpers ---------------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _root_xform(prim_path: str, translation, orientation):
    """Define an Xform root and author its (idempotent, single) translate/orient ops."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    xf = UsdGeom.Xformable(xform)
    # canonical op triple [translate, orient, scale] (XformPrimView requires it)
    t = translation if translation is not None else (0.0, 0.0, 0.0)
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in t]))
    w, x, y, z = (float(v) for v in (orientation if orientation is not None
                                     else (1.0, 0.0, 0.0, 0.0)))
    xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    return stage, xform.GetPrim()


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _phys_material(stage, path: str, mu_s: float, mu_d: float, restitution: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(mu_s))
    api.CreateDynamicFrictionAttr(float(mu_d))
    api.CreateRestitutionAttr(float(restitution))
    return mat


def _bind_material(prim, mat) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _add_box(stage, path: str, *, center, size, color, yaw_deg: float = 0.0,
             collide: Callable | None = None, mat=None):
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    half = math.radians(yaw_deg) / 2.0
    xf.AddOrientOp().Set(Gf.Quatf(math.cos(half), Gf.Vec3f(0.0, 0.0, math.sin(half))))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide is not None:
        collide(box.GetPrim())
        if mat is not None:
            _bind_material(box.GetPrim(), mat)
    return box.GetPrim()


# ----- arena spawner (static ring-segment walls + roof) ------------------------------------------
def _spawn_arena(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the arena: STATIC colliders only (no rigid body). Polar frame centred at
    the root origin: start pen [phi_lo..roof_a], roofed gallery [roof_a..roof_b],
    catch bay [roof_b..phi_hi] closed by a radial arrest wall at phi_hi."""
    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    wall_mat = _phys_material(stage, f"{prim_path}/wallMat", cfg.wall_mu, cfg.wall_mu, 0.0)
    c = cfg

    def ring(tag: str, radius: float, pa: float, pb: float, height: float, color) -> None:
        nseg = max(1, int(math.ceil((pb - pa) / c.seg_deg)))
        step = (pb - pa) / nseg
        length = 2.0 * radius * math.tan(math.radians(step) / 2.0) + 0.012
        for i in range(nseg):
            a = pa + (i + 0.5) * step
            ar = math.radians(a)
            _add_box(stage, f"{prim_path}/{tag}_{i}",
                     center=(radius * math.cos(ar), radius * math.sin(ar), height / 2.0),
                     size=(c.wall_t, length, height), color=color, yaw_deg=a,
                     collide=collide, mat=wall_mat)

    def radial_wall(tag: str, phi: float) -> None:
        rm = (c.R_in + c.R_out) / 2.0
        ar = math.radians(phi)
        _add_box(stage, f"{prim_path}/{tag}",
                 center=(rm * math.cos(ar), rm * math.sin(ar), c.wall_h / 2.0),
                 size=(c.R_out - c.R_in + 0.030, c.wall_t, c.wall_h),
                 color=c.wall_color, yaw_deg=phi, collide=collide, mat=wall_mat)

    # walls: full sweep on both radii
    ring("inner", c.R_in, c.phi_lo, c.phi_hi, c.wall_h, c.wall_color)
    ring("outer", c.R_out, c.phi_lo, c.phi_hi, c.wall_h, c.wall_color)
    # roof: gallery span only, low slabs across the channel
    rm = (c.R_in + c.R_out) / 2.0
    nseg = max(1, int(math.ceil((c.roof_b - c.roof_a) / c.seg_deg)))
    step = (c.roof_b - c.roof_a) / nseg
    length = 2.0 * (c.R_out + 0.02) * math.tan(math.radians(step) / 2.0) + 0.012
    for i in range(nseg):
        a = c.roof_a + (i + 0.5) * step
        ar = math.radians(a)
        _add_box(stage, f"{prim_path}/roof_{i}",
                 center=(rm * math.cos(ar), rm * math.sin(ar), c.roof_lo + c.roof_t / 2.0),
                 size=(c.R_out - c.R_in + 0.030, length, c.roof_t),
                 color=c.roof_color, yaw_deg=a, collide=collide, mat=wall_mat)
    # end walls
    radial_wall("pen_end", c.phi_lo)
    radial_wall("arrest", c.phi_hi)
    # catch-bay goal marker: VISUAL ONLY (a flush collider would wall the rolling cup)
    bm = math.radians((c.roof_b + c.phi_hi) / 2.0)
    _add_box(stage, f"{prim_path}/bayMark",
             center=(rm * math.cos(bm), rm * math.sin(bm), 0.0005),
             size=(c.R_out - c.R_in - 0.06, 0.10, 0.001),
             color=(0.15, 0.65, 0.20), yaw_deg=(c.roof_b + c.phi_hi) / 2.0,
             collide=None)
    return root


# ----- roller spawner (wheelset: two rim spheres + capsule waist) --------------------------------
def _spawn_roller(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a roller at `prim_path`: DYNAMIC compound. Body frame: +x from BASE rim
    (radius `r_a`, at x=-L/2) toward MOUTH rim (radius `r_b`, at x=+L/2); capsule
    waist along x. Mass, CoM (origin), diagonal inertia and material are authored
    here explicitly (custom spawn funcs apply no cfg schemas)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.02)
    px.CreateAngularDampingAttr(0.02)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    c = cfg
    # mass: 45/45/10 split over base rim / mouth rim / waist, CoM at the body origin
    m1, m2, mc = 0.45 * c.mass, 0.45 * c.mass, 0.10 * c.mass
    h = c.length / 2.0
    ixx = 0.4 * m1 * c.r_a**2 + 0.4 * m2 * c.r_b**2 + 0.5 * mc * c.r_core**2
    iyy = (0.4 * m1 * c.r_a**2 + m1 * h**2 + 0.4 * m2 * c.r_b**2 + m2 * h**2
           + mc * (c.length**2 / 12.0))
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(float(c.mass))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(float(ixx), float(iyy), float(iyy)))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

    collide = _make_collide(c.contact_offset)
    mat = _phys_material(stage, f"{prim_path}/rimMat", c.mu, c.mu - 0.05, 0.0)

    def sphere(name: str, x: float, radius: float, color) -> None:
        sp = UsdGeom.Sphere.Define(stage, f"{prim_path}/{name}")
        sp.CreateRadiusAttr(float(radius))
        UsdGeom.Xformable(sp.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(float(x), 0.0, 0.0))
        sp.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        collide(sp.GetPrim())
        _bind_material(sp.GetPrim(), mat)

    sphere("base", -h, c.r_a, c.color_a)
    sphere("mouth", +h, c.r_b, c.color_b)
    cap = UsdGeom.Capsule.Define(stage, f"{prim_path}/waist")
    cap.CreateRadiusAttr(float(c.r_core))
    cap.CreateHeightAttr(float(c.length - 2.0 * c.r_core - 0.020))
    cap.CreateAxisAttr("X")
    cap.CreateDisplayColorAttr([Gf.Vec3f(*c.color_b)])
    collide(cap.GetPrim())
    _bind_material(cap.GetPrim(), mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg, SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "arena" not in _SPAWNER_CACHE:

        @configclass
        class ArenaSpawnerCfg(SpawnerCfg):
            func: Callable = clone(_spawn_arena)
            R_in: float = 0.155
            R_out: float = 0.420
            wall_t: float = 0.022
            wall_h: float = 0.30
            roof_lo: float = 0.115
            roof_t: float = 0.020
            seg_deg: float = 12.5
            phi_lo: float = -160.0
            phi_hi: float = 115.0
            roof_a: float = -75.0
            roof_b: float = 75.0
            wall_mu: float = 0.15
            wall_color: tuple = (0.42, 0.42, 0.46)
            roof_color: tuple = (0.30, 0.30, 0.36)
            contact_offset: float = 0.002

        @configclass
        class RollerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_roller)
            length: float = 0.110
            r_a: float = 0.030
            r_b: float = 0.045
            r_core: float = 0.020
            mass: float = 0.25
            mu: float = 0.80
            color_a: tuple = (0.75, 0.30, 0.15)
            color_b: tuple = (0.85, 0.45, 0.25)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(arena=ArenaSpawnerCfg, roller=RollerSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg ---------------------------------------------------------------------------------
@dataclass
class CooperCurveSceneCfg(BaseCfg):
    """Config for `CooperCurveScene`. The geometric claims the task rests on are
    asserted in __post_init__: the staged cup's rolling footprint clears both gallery
    walls; the roof admits the rolling profile but refuses standing/carried passage;
    the equal-rim decoy launched straight must wedge on the outer wall well before
    the bay; spawn slots are wall-clear and disjoint from the staging zone."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    stage_phi_lo: float = tunable(-90.0)   # staging zone azimuth window (deg)
    stage_phi_hi: float = tunable(-76.0)
    stage_rho_tol: float = tunable(0.045)  # staging radius tolerance around the apex-true radius
    stage_align_deg: float = tunable(25.0)  # cup axis (base->mouth) within this of radially OUT
    stage_calm: float = tunable(0.30)      # max |lin vel| for the staging latch (m/s)
    zone_half_deg: float = tunable(10.0)   # checkpoint azimuth half-width (deg)
    zone_z_max: float = tunable(0.060)     # checkpoint/bay: cup centre below this (floor transit)
    zone_rho_lo: float = tunable(0.190)    # checkpoint/bay: cup centre radius window (m)
    zone_rho_hi: float = tunable(0.400)
    bay_phi_lo: float = tunable(88.0)      # catch-bay azimuth window (deg)
    bay_phi_hi: float = tunable(114.0)
    lying_axis_z: float = tunable(0.35)    # |axis z-component| below this = lying on its side
    settle_lin: float = tunable(0.05)      # max |lin vel| when judging success (m/s)
    settle_ang: float = tunable(0.80)      # max |ang vel| of the cup when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    slot_shuffle: bool = tunable(True)     # permute cup/decoy over the two pen slots
    jit_phi_deg: float = tunable(3.0)      # per-body azimuth jitter at its slot (+- deg)
    jit_rho: float = tunable(0.010)        # per-body radius jitter (+- m)
    spawn_yaw_deg: float = tunable(180.0)  # per-body free yaw (+- deg)

    # --- info: rollers ---------------------------------------------------------------------------
    length: float = info(0.110)            # rim-sphere centre separation
    r_base: float = info(0.030)            # cup base rim radius
    r_mouth: float = info(0.045)           # cup mouth rim radius
    r_core: float = info(0.020)            # waist capsule radius
    r_decoy: float = info(0.0375)          # decoy rim radius (both ends)
    roller_mass: float = info(0.25)
    roller_mu: float = info(0.80)
    # --- info: arena (wall CENTRE radii / azimuths in deg) ---------------------------------------
    R_in: float = info(0.155)
    R_out: float = info(0.420)
    wall_t: float = info(0.022)
    wall_h: float = info(0.30)
    roof_lo: float = info(0.115)           # roof underside height
    roof_t: float = info(0.020)
    phi_lo: float = info(-160.0)           # pen end wall
    phi_hi: float = info(115.0)            # bay arrest wall
    roof_a: float = info(-75.0)            # gallery (roofed) span
    roof_b: float = info(75.0)
    cp_phis: tuple = info((-45.0, 0.0, 45.0))  # ordered checkpoint azimuths
    # --- info: spawn slots (polar, deg / m) ------------------------------------------------------
    slot_phis: tuple = info((-100.0, -134.0))
    slot_rho: float = info(0.2875)
    floor_mu: float = info(0.80)
    contact_offset: float = info(0.002)
    # rubric weights (0.10 + 0.20 + 0.20 + 0.15 + 0.10 = 0.75 = the non-success cap)
    w_staged: float = info(0.10)
    w_cp1: float = info(0.20)
    w_cp2: float = info(0.20)
    w_cp3: float = info(0.15)
    w_bay: float = info(0.10)

    # ----- derived geometry ----------------------------------------------------------------------
    def d_apex(self) -> float:
        """Distance from the BASE rim centre to the cone apex (the turn centre)."""
        return self.length * self.r_base / (self.r_mouth - self.r_base)

    def rest_pitch(self) -> float:
        """Axis pitch of the resting cup (mouth end higher), radians."""
        return math.asin((self.r_mouth - self.r_base) / self.length)

    def rho_stage(self) -> float:
        """Body-origin radius that puts the cup's apex exactly on the arena centre."""
        return self.d_apex() + 0.5 * self.length * math.cos(self.rest_pitch())

    def clear_in(self) -> float:
        return self.R_in + self.wall_t / 2.0

    def clear_out(self) -> float:
        return self.R_out - self.wall_t / 2.0

    def __post_init__(self) -> None:
        # Honesty-by-construction asserts (the claims the task rests on).
        ci, co = self.clear_in(), self.clear_out()
        cosp = math.cos(self.rest_pitch())
        inner_edge = self.d_apex() - self.r_base
        outer_edge = self.d_apex() + self.length * cosp + self.r_mouth
        assert inner_edge > ci + 0.015, "staged cup base rim must clear the inner wall"
        assert outer_edge < co - 0.015, "staged cup mouth rim must clear the outer wall"
        assert 2.0 * self.r_mouth + 0.020 <= self.roof_lo, \
            "the roof must admit the rolling profile with margin"
        assert self.length * cosp + self.r_base + self.r_mouth >= self.roof_lo + 0.045, \
            "a standing/carried cup must NOT fit under the roof"
        assert 2.0 * self.r_decoy + 0.020 <= self.roof_lo, \
            "the decoy's rolling profile must also fit (the gallery, not the roof, stops it)"
        # decoy launched straight from the staging radius wedges before the bay
        rho_out_sph = self.rho_stage() + self.length / 2.0
        s_jam = math.sqrt(max((co - self.r_decoy) ** 2 - rho_out_sph**2, 1e-9))
        rho_mid = (ci + co) / 2.0
        s_bay = rho_mid * math.radians(self.bay_phi_lo - self.stage_phi_hi)
        assert s_jam < 0.45 * s_bay, \
            "the straight-rolling decoy must wedge on the outer wall well before the bay"
        # spawn slots: wall-clear (>= 8 mm at max jitter) and disjoint from staging
        ext = self.length / 2.0 * cosp + max(self.r_mouth, self.r_decoy)
        rho_min = self.slot_rho - self.jit_rho
        rho_max = self.slot_rho + self.jit_rho
        assert rho_min - ext > ci + 0.008, "spawn slots must clear the inner wall"
        assert rho_max + ext < co - 0.008, "spawn slots must clear the outer wall"
        for sp in self.slot_phis:
            assert sp + self.jit_phi_deg < self.stage_phi_lo - 2.0, \
                "spawn slots must sit outside the staging zone"
            wall_gap = self.slot_rho * math.radians((sp - self.jit_phi_deg) - self.phi_lo)
            assert wall_gap > ext + 0.008, "spawn slots must clear the pen end wall"
        assert abs(self.w_staged + self.w_cp1 + self.w_cp2 + self.w_cp3
                   + self.w_bay - 0.75) < 1e-9, "weights must sum to the 0.75 cap"


# ----- small quaternion helpers (wxyz, torch, batched) -------------------------------------------
def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -------------------------------------------------------------------------------------
@SCENES.register("cooper_curve")
class CooperCurveScene(BaseScene):
    cfg: CooperCurveSceneCfg

    def __init__(self, cfg: CooperCurveSceneCfg | None = None) -> None:
        super().__init__(cfg or CooperCurveSceneCfg())

    # ----- assets --------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        arena_spawn = cls["arena"](
            R_in=c.R_in, R_out=c.R_out, wall_t=c.wall_t, wall_h=c.wall_h,
            roof_lo=c.roof_lo, roof_t=c.roof_t, phi_lo=c.phi_lo, phi_hi=c.phi_hi,
            roof_a=c.roof_a, roof_b=c.roof_b, contact_offset=c.contact_offset)
        cup_spawn = cls["roller"](
            length=c.length, r_a=c.r_base, r_b=c.r_mouth, r_core=c.r_core,
            mass=c.roller_mass, mu=c.roller_mu, contact_offset=c.contact_offset,
            color_a=(0.75, 0.30, 0.15), color_b=(0.88, 0.48, 0.28))
        decoy_spawn = cls["roller"](
            length=c.length, r_a=c.r_decoy, r_b=c.r_decoy, r_core=c.r_core,
            mass=c.roller_mass, mu=c.roller_mu, contact_offset=c.contact_offset,
            color_a=(0.35, 0.42, 0.60), color_b=(0.42, 0.50, 0.68))
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.floor_mu, dynamic_friction=c.floor_mu - 0.05,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "arena": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Arena",
                spawn=arena_spawn,
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "cup": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cup",
                spawn=cup_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 1.0, 0.05)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=decoy_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.3, 1.0, 0.05)),
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
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
                # the solve drives the cup with per-step external wrenches
                "enable_external_forces_every_iteration": True,
            },
        )

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.cup: RigidObject = env.iscene["cup"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self.cup_slot = torch.zeros(n, dtype=torch.long, device=dev)  # which slot the cup got
        # ordered latch chain (partial credit survives transients; success judged live)
        self._staged = torch.zeros(n, dtype=torch.bool, device=dev)
        self._cp1 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._cp2 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._cp3 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._bay = torch.zeros(n, dtype=torch.bool, device=dev)

    def _rest_state(self, rho: torch.Tensor, phi: torch.Tensor, yaw: torch.Tensor,
                    tapered: bool) -> torch.Tensor:
        """Root state (m,13) for a roller resting on its rims at polar (rho, phi) with
        body yaw `yaw` (axis heading), 2 mm settle drop. `tapered`=False -> decoy."""
        c = self.cfg
        m = rho.shape[0]
        dev = rho.device
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = rho * torch.cos(phi)
        st[:, 1] = rho * torch.sin(phi)
        if tapered:
            st[:, 2] = (c.r_base + c.r_mouth) / 2.0 + 0.002
            pitch = torch.full((m,), c.rest_pitch(), device=dev)
        else:
            st[:, 2] = c.r_decoy + 0.002
            pitch = torch.zeros(m, device=dev)
        st[:, 3:7] = _qmul(_qz(yaw), _qy(-pitch))
        return st

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: permute cup/decoy over the two pen slots, polar jitter + free
        yaw each, overlap-rejected (deterministic radial-yaw fallback), clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        torch.rand(3, device=dev)  # burn draws: first post-seed draws are degenerate

        if c.slot_shuffle:
            swap = torch.rand(m, device=dev) < 0.5
        else:
            swap = torch.zeros(m, dtype=torch.bool, device=dev)
        self.cup_slot[env_ids] = swap.long()
        phis = torch.tensor([math.radians(p) for p in c.slot_phis], device=dev)
        cup_phi0 = torch.where(swap, phis[1], phis[0])
        dec_phi0 = torch.where(swap, phis[0], phis[1])

        def sample(phi0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            rho = c.slot_rho + (torch.rand(m, device=dev) * 2 - 1) * c.jit_rho
            phi = phi0 + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.jit_phi_deg)
            yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.spawn_yaw_deg)
            return rho, phi, yaw

        def seg_gap(ra, pa, ya, rb, pb, yb) -> torch.Tensor:
            """Min distance between the two axis segments (half-length L/2), (m,)."""
            ca = torch.stack([ra * torch.cos(pa), ra * torch.sin(pa)], dim=-1)
            cb = torch.stack([rb * torch.cos(pb), rb * torch.sin(pb)], dim=-1)
            ua = torch.stack([torch.cos(ya), torch.sin(ya)], dim=-1)
            ub = torch.stack([torch.cos(yb), torch.sin(yb)], dim=-1)
            h = c.length / 2.0
            best = torch.full_like(ra, torch.inf)
            for f in (-1.0, -0.5, 0.0, 0.5, 1.0):
                p = ca + ua * (f * h)
                t = ((p - cb) * ub).sum(-1).clamp(-h, h)
                best = torch.minimum(best, (p - cb - ub * t.unsqueeze(-1)).norm(dim=-1))
                q = cb + ub * (f * h)
                t = ((q - ca) * ua).sum(-1).clamp(-h, h)
                best = torch.minimum(best, (q - ca - ua * t.unsqueeze(-1)).norm(dim=-1))
            return best

        need = c.r_mouth + c.r_decoy + 0.010
        c_rho, c_phi, c_yaw = sample(cup_phi0)
        d_rho, d_phi, d_yaw = sample(dec_phi0)
        for _try in range(12):
            bad = seg_gap(c_rho, c_phi, c_yaw, d_rho, d_phi, d_yaw) < need
            if not bad.any():
                break
            nb = int(bad.sum())
            d_yaw[bad] = (torch.rand(nb, device=dev) * 2 - 1) * math.radians(c.spawn_yaw_deg)
            c_yaw[bad] = (torch.rand(nb, device=dev) * 2 - 1) * math.radians(c.spawn_yaw_deg)
        bad = seg_gap(c_rho, c_phi, c_yaw, d_rho, d_phi, d_yaw) < need
        if bad.any():  # deterministic fallback: both axes radial -> tangential extent is minimal
            c_yaw[bad] = c_phi[bad]
            d_yaw[bad] = d_phi[bad]

        st = self._rest_state(c_rho, c_phi, c_yaw, tapered=True)
        st[:, 0:3] += origin
        self.cup.write_root_state_to_sim(st, env_ids)
        st = self._rest_state(d_rho, d_phi, d_yaw, tapered=False)
        st[:, 0:3] += origin
        self.decoy.write_root_state_to_sim(st, env_ids)

        for latch in (self._staged, self._cp1, self._cp2, self._cp3, self._bay):
            latch[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "cup": self.cup.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "cup_slot": self.cup_slot[env_ids].clone(),
            "staged": self._staged[env_ids].clone(),
            "cp1": self._cp1[env_ids].clone(),
            "cp2": self._cp2[env_ids].clone(),
            "cp3": self._cp3[env_ids].clone(),
            "bay": self._bay[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.cup.write_root_state_to_sim(state["cup"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.cup_slot[env_ids] = state["cup_slot"]
        self._staged[env_ids] = state["staged"]
        self._cp1[env_ids] = state["cp1"]
        self._cp2[env_ids] = state["cp2"]
        self._cp3[env_ids] = state["cp3"]
        self._bay[env_ids] = state["bay"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A walled rolling arena stands on the floor: an open-topped START PEN "
            f"curves into a low ROOFED GALLERY (a {c.roof_b - c.roof_a:.0f} deg "
            f"annular corridor between wall radii {c.R_in * 100:.0f} and "
            f"{c.R_out * 100:.0f} cm, roof underside just {c.roof_lo * 1000:.0f} mm "
            f"up) which opens into a walled CATCH BAY marked by a green strip and "
            f"closed by an arrest wall. In the pen lie TWO rollers of equal length "
            f"({c.length * 1000:.0f} mm between rim centres) whose positions are "
            f"swapped and scattered every episode: the terracotta CUP is TAPERED — a "
            f"small base rim ({c.r_base * 1000:.0f} mm radius) and a large mouth rim "
            f"({c.r_mouth * 1000:.0f} mm) — while the blue-grey DECOY has EQUAL rims "
            f"({c.r_decoy * 1000:.1f} mm) and rolls dead straight. A tapered roller "
            f"cannot roll straight: it orbits the apex of its own cone, "
            f"{c.d_apex() * 100:.0f} cm beyond its base rim — and the gallery's "
            f"curvature matches exactly that turning circle. The roof admits only a "
            f"roller on its side (rolling profile {2 * c.r_mouth * 1000:.0f} mm); "
            f"the cup standing or carried upright ({(c.length + c.r_base + c.r_mouth) * 1000:.0f} "
            f"mm) cannot pass.\n"
            f"Goal: deliver the CUP into the catch bay through the gallery, by "
            f"rolling. (1) Identify the tapered roller by its rim profile. (2) Stage "
            f"it at the gallery mouth (azimuth {c.stage_phi_lo:.0f}..{c.stage_phi_hi:.0f} "
            f"deg) lying on its rims with its BASE toward the arena centre, so its "
            f"cone apex falls on the gallery's arc centre. (3) Launch it with a "
            f"single tangential push and let it coast: the taper steers it around "
            f"the full curve, past the gallery checkpoints in order, and into the "
            f"bay, where the arrest wall stops it. The decoy must stay OUT of the "
            f"catch bay. Success: the cup at rest on its side inside the catch bay, "
            f"having rolled the gallery end to end."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Find the tapered terracotta roller-cup, lay it at the gallery mouth "
            "with its small base rim toward the arena centre, and push it so it "
            "rolls on its rims all the way around the curved roofed gallery into "
            "the green catch bay. Leave the straight-rolling blue decoy out of "
            "the bay."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _polar(self, pos_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """World positions (N,3) -> (rho (N,), phi_deg (N,)) about the arena centre."""
        rel = pos_w[:, :2] - self.env_origins[:, :2]
        return rel.norm(dim=-1), torch.rad2deg(torch.atan2(rel[:, 1], rel[:, 0]))

    def _axis_w(self, body) -> torch.Tensor:
        """(N,3) world direction of the body +x axis (base -> mouth)."""
        from isaaclab.utils.math import quat_apply

        q = body.data.root_quat_w
        ex = torch.tensor([1.0, 0.0, 0.0], device=q.device).expand(q.shape[0], 3)
        return quat_apply(q, ex)

    def _rel_z(self, body) -> torch.Tensor:
        return body.data.root_pos_w[:, 2] - self.env_origins[:, 2]

    def staged_now(self) -> torch.Tensor:
        """(N,) bool: cup lying in the staging zone, base toward the arena centre,
        apex-true radius, calm."""
        c = self.cfg
        rho, phi = self._polar(self.cup.data.root_pos_w)
        ax = self._axis_w(self.cup)
        rel = self.cup.data.root_pos_w[:, :2] - self.env_origins[:, :2]
        u_r = rel / rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        ax_xy = ax[:, :2] / ax[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)
        out_ok = (ax_xy * u_r).sum(-1) > math.cos(math.radians(c.stage_align_deg))
        lying = ax[:, 2].abs() < c.lying_axis_z
        calm = self.cup.data.root_lin_vel_w.norm(dim=-1) < c.stage_calm
        return ((phi > c.stage_phi_lo) & (phi < c.stage_phi_hi)
                & ((rho - c.rho_stage()).abs() < c.stage_rho_tol)
                & (self._rel_z(self.cup) < c.zone_z_max) & lying & out_ok & calm)

    def _zone(self, body, phi_c: float) -> torch.Tensor:
        """(N,) bool: body centre at floor level inside the checkpoint sector."""
        c = self.cfg
        rho, phi = self._polar(body.data.root_pos_w)
        return ((phi - phi_c).abs() < c.zone_half_deg) \
            & (rho > c.zone_rho_lo) & (rho < c.zone_rho_hi) \
            & (self._rel_z(body) < c.zone_z_max)

    def in_bay(self, body) -> torch.Tensor:
        """(N,) bool: body centre at floor level inside the catch bay."""
        c = self.cfg
        rho, phi = self._polar(body.data.root_pos_w)
        return ((phi > c.bay_phi_lo) & (phi < c.bay_phi_hi)
                & (rho > c.zone_rho_lo) & (rho < c.zone_rho_hi)
                & (self._rel_z(body) < c.zone_z_max))

    def lying(self) -> torch.Tensor:
        return self._axis_w(self.cup)[:, 2].abs() < self.cfg.lying_axis_z

    def settled(self) -> torch.Tensor:
        c = self.cfg
        return (self.cup.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.cup.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang) \
            & (self.decoy.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)

    def _update_latches(self) -> None:
        c = self.cfg
        self._staged |= self.staged_now()
        self._cp1 |= self._staged & self._zone(self.cup, c.cp_phis[0])
        self._cp2 |= self._cp1 & self._zone(self.cup, c.cp_phis[1])
        self._cp3 |= self._cp2 & self._zone(self.cup, c.cp_phis[2])
        self._bay |= self._cp3 & self.in_bay(self.cup)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the cup rolled the gallery end to end (full ordered latch chain)
        and now rests on its side inside the catch bay; the decoy is out of the bay;
        everything finite."""
        self._update_latches()
        finite = torch.isfinite(self.cup.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.decoy.data.root_pos_w).all(dim=-1)
        return (self._staged & self._cp1 & self._cp2 & self._cp3 & self._bay
                & self.in_bay(self.cup) & self.lying() & self.settled()
                & ~self.in_bay(self.decoy) & finite)

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: ordered latched credit (staged -> cp1 -> cp2 -> cp3 ->
        bay; ~0 for doing nothing, and no credit for any leg without the previous
        legs), capped at 0.75 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        part = (c.w_staged * self._staged.float() + c.w_cp1 * self._cp1.float()
                + c.w_cp2 * self._cp2.float() + c.w_cp3 * self._cp3.float()
                + c.w_bay * self._bay.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(part), part)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="cooper_curve", robot="null"))

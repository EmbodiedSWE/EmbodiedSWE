"""ChockRampScene — park the RED cube at rest ON a slick 25-degree ramp, inside a
randomized YELLOW target band, by first keying the BLUE CHOCK into the pocket
directly downhill of the band so it arrests the cube's slide.

Derived from mujoco_playground/pick ("bring the box to the target": one grasp of a
4 cm cube and one guided free-space carry to a floating, always-reachable target
pose; the target is trivially STABLE — wherever you place the box, it stays).
Here the goal pose is on a surface where NOTHING rests unsupported: the ramp face
is slick (friction angle of every object/ramp pair is ~11 deg below the 25-deg
slope, asserted), so the seed's place-at-the-target primitive scores zero — the
cube placed on the band simply slides off the foot of the ramp (the smoke battery
proves it). The plan the seed never needs:
  (1) INSTALL A FIXTURE FIRST: the blue CHOCK has a keying TAB on its underside;
      drop it into the recessed POCKET immediately downhill of the yellow band.
      The tab-in-pocket anchors the chock against the slide load (an unkeyed
      chock resting anywhere on the face slides away like everything else);
  (2) then LAY the red cube on the ramp above the band and LET GO: gravity
      delivers it — the cube slides down the slick face and is arrested by the
      seated chock's plate exactly inside the band.
Execution order is FORCED BY PHYSICS, not by the rubric: with no chock seated
there is no rest state on the face, so cube-first ends with the cube on the floor
past the ramp foot. Station choice matters: three identical pockets 90 mm apart,
but only the one just downhill of the band parks the cube IN the band — a full,
physically stable park at a wrong station is judged 0 (smoke constructs it).
Only the settled final state is judged.

Assets are fully procedural: one KINEMATIC compound (the ramp) + one dynamic
compound (the chock, custom spawner) + one dynamic cube + one visual-only band
marker (kinematic, collision DISABLED — it can never arrest anything).
  ramp (kinematic; SLOPE frame: origin = face centre, +x upslope, +z face normal):
    face   : 0.48 x 0.24 slick top layer, 12 mm thick, with three rectangular
             POCKET cutouts (24 x 52 mm, 12 mm deep) on the centreline at
             x in {-0.075, 0.015, 0.105};
    base   : dark under-layer whose top is the pocket floors;
    pedestal: cosmetic support block underneath.
  chock (dynamic, 0.15 kg, grippy): upright PLATE (14 x 60 x 45 mm) with a
    downhill anti-tip FOOT and an underside TAB (14 x 40 x 11 mm) that keys into
    a pocket. Origin = centre of the plate-bottom/tab-top plane.
  cube  (dynamic, RED, 50 mm, 0.10 kg, grippy). Friction pairs with the slick
    face still give a friction angle far below the slope angle (PhysX averages
    the two materials; asserted in __post_init__).
  marker: thin YELLOW plate on the face, centred on the target band; visual only.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  0.375 * seated — the CHOCK EVER seated in the STARRED pocket (keyed, aligned,
                   slow)                                              (latched)
  0.375 * parked — the cube EVER braced against the seated chock inside the
                   band, slow                                         (latched)
  1.0 iff success() — cube braced in the band by the seated chock AND the pair
  still for >= settle_steps consecutive physics steps (a stillness STREAK,
  ticked only in post_step), finite. Non-success capped at 0.75 (exact in
  float32: 0.375 + 0.375). Null policy latches nothing (score ~0).

Honesty geometry (asserted in `__post_init__`):
  - slick face: atan(pair-averaged mu) < slope - 6 deg for BOTH the cube and the
    chock, so no unsupported rest exists on the face (the seed strategy fails);
  - tab/pocket: real play (>= 8 mm along-slope, >= 10 mm across) so a hand can
    find the socket, tab shallower than the pocket so the plate lands flush on
    the face (an unkeyed chock sits ~11 mm PROUD — the seated z-window rejects it);
  - the cube cannot drop into a pocket (opening far smaller than the cube);
  - anti-tip: the seated chock's restoring moment about its downhill foot edge
    exceeds the cube's slide load moment by >= 2x;
  - plate tall enough that the cube pushes face-on and cannot ride over;
  - stations far enough apart that a wrong-station park is far outside the band;
  - the release span above every band stays on the face, and ground spawn slots
    clear the ramp's foot by a real margin.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
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


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qinv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    w = aw * bw - (av * bv).sum(-1, keepdim=True)
    v = aw * bv + bw * av + torch.cross(av, bv, dim=-1)
    return torch.cat([w, v], dim=-1)


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4), pure torch."""
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(qv, t, dim=-1)


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- custom compound spawners -----------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _root_xform(prim_path: str, translation, orientation):
    """Define an Xform root and author its (idempotent, single) translate/orient ops."""
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


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, *, center, size, color, contact_offset: float):
    """One collidable box child: translate + scale, displayColor."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(box.GetPrim(), contact_offset)
    return box.GetPrim()


def _phys_material(stage, path: str, static: float, dynamic: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _bind_material(prim, mat) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _rigid_kinematic(root) -> None:
    """Kinematic rigid-body armor on the compound root (teleportable at reset)."""
    from pxr import UsdPhysics

    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(60.0)


def _rigid_dynamic(root, mass: float) -> None:
    """Dynamic rigid-body armor. NOTE: a custom spawn func gets NO schema help from
    the asset cfg (mass_props / rigid_props are ignored) — author everything here.
    MassAPI mass alone puts the CoM at the body origin; for the chock the origin is
    the plate-bottom centre, i.e. a LOW CoM — conservative for anti-tip."""
    from pxr import PhysxSchema, UsdPhysics

    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(False)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.20)
    px.CreateMaxDepenetrationVelocityAttr(0.5)


def _spawn_ramp(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC ramp at `prim_path`, entirely in the SLOPE frame
    (origin = face centre, +x upslope, +z face normal; the world tilt lives in the
    root pose). Children: dark base layer (its top = the pocket floors), slick top
    face split into strips around the three pocket cutouts, cosmetic pedestal."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_kinematic(root)
    c = cfg
    co = c.contact_offset
    face_mat = _phys_material(stage, f"{prim_path}/facemat", c.mu_face, c.mu_face_dyn)
    hl, hw = c.slope_len / 2, c.slope_w / 2
    kids = []

    def box(name, center, size, color):
        kids.append(_box(stage, f"{prim_path}/{name}", center=center, size=size,
                         color=color, contact_offset=co))

    # base layer: top face at z = -top_t (the pocket floors), dark
    box("base", (0.0, 0.0, -c.top_t - c.base_t / 2),
        (c.slope_len, c.slope_w, c.base_t), c.base_color)
    # top face: full-width strips between/outside the pockets ...
    xs = sorted(c.pocket_xs)
    edges = [-hl]
    for xk in xs:
        edges += [xk - c.pocket_len / 2, xk + c.pocket_len / 2]
    edges.append(hl)
    for i in range(0, len(edges), 2):
        x0, x1 = edges[i], edges[i + 1]
        box(f"face_x{i // 2}", ((x0 + x1) / 2, 0.0, -c.top_t / 2),
            (x1 - x0, c.slope_w, c.top_t), c.face_color)
    # ... plus, per pocket, two side strips flanking the cutout
    yin = c.pocket_w / 2
    for j, xk in enumerate(xs):
        for s, tag in ((1.0, "p"), (-1.0, "n")):
            box(f"face_s{j}{tag}", (xk, s * (yin + (hw - yin) / 2), -c.top_t / 2),
                (c.pocket_len, hw - yin, c.top_t), c.face_color)
    # cosmetic pedestal under the face (embedded into the ground; kinematic)
    box("pedestal", (0.10, 0.0, -0.115), (0.26, 0.20, 0.17), c.base_color)
    for k in kids:
        _bind_material(k, face_mat)
    return root


def _spawn_chock(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC chock at `prim_path`. Body frame: origin = centre of the
    plate-bottom / tab-top plane, +x upslope when seated. Children: underside TAB
    (keys into a pocket), upright PLATE (the arrest face, upslope face at
    +plate_t/2), downhill anti-tip FOOT."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_dynamic(root, cfg.mass)
    c = cfg
    co = c.contact_offset
    mat = _phys_material(stage, f"{prim_path}/mat", c.mu, c.mu_dyn)
    kids = [
        _box(stage, f"{prim_path}/tab",
             center=(0.0, 0.0, -c.tab_h / 2),
             size=(c.plate_t, c.tab_w, c.tab_h), color=c.color, contact_offset=co),
        _box(stage, f"{prim_path}/plate",
             center=(0.0, 0.0, c.plate_h / 2),
             size=(c.plate_t, c.plate_w, c.plate_h), color=c.color, contact_offset=co),
        _box(stage, f"{prim_path}/foot",
             center=(-(c.plate_t / 2 + c.foot_len / 2), 0.0, c.foot_h / 2),
             size=(c.foot_len, c.plate_w, c.foot_h), color=c.color, contact_offset=co),
    ]
    for k in kids:
        _bind_material(k, mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "ramp" not in _SPAWNER_CACHE:

        @configclass
        class RampSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_ramp)
            slope_len: float = 0.48
            slope_w: float = 0.24
            top_t: float = 0.012
            base_t: float = 0.020
            pocket_xs: tuple = (-0.075, 0.015, 0.105)
            pocket_len: float = 0.024
            pocket_w: float = 0.052
            mu_face: float = 0.10
            mu_face_dyn: float = 0.08
            face_color: tuple = (0.62, 0.66, 0.72)
            base_color: tuple = (0.18, 0.20, 0.24)
            contact_offset: float = 0.002

        @configclass
        class ChockSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_chock)
            plate_t: float = 0.014
            plate_w: float = 0.060
            plate_h: float = 0.045
            tab_w: float = 0.040
            tab_h: float = 0.011
            foot_len: float = 0.030
            foot_h: float = 0.008
            mass: float = 0.15
            mu: float = 0.40
            mu_dyn: float = 0.35
            color: tuple = (0.10, 0.20, 0.85)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["ramp"] = RampSpawnerCfg
        _SPAWNER_CACHE["chock"] = ChockSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ChockRampSceneCfg(BaseCfg):
    """Config for `ChockRampScene`. The forced order (fixture before cargo) and the
    station choice are enforced by the slick face and the tab/pocket key — the
    rubric only reads out states that geometry makes meaningful."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    band_half: float = tunable(0.030)       # in-band: |cube_x - band_centre| <= this (m)
    seat_tol_x: float = tunable(0.008)      # chock seated: |x - pocket_x| < this (m)
    seat_tol_y: float = tunable(0.012)
    seat_z_lo: float = tunable(-0.004)      # seated chock origin z window (unkeyed rest
    seat_z_hi: float = tunable(0.005)       # sits ~tab_h PROUD and is rejected)
    seat_ang_deg: float = tunable(15.0)     # chock alignment to the slope frame (deg)
    gap_lo: float = tunable(-0.008)         # braced: cube-face-to-plate-face gap window (m)
    gap_hi: float = tunable(0.015)
    settle_lin: float = tunable(0.10)       # stillness gates (above the GPU phantom band)
    settle_ang: float = tunable(0.90)
    settle_steps: int = tunable(12)         # consecutive still steps required by success
    seat_vel: float = tunable(0.15)         # latch gates: max |v| when latching seated
    park_vel: float = tunable(0.25)         # ... and parked

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    ramp_jitter: float = tunable(0.03)      # ramp xy jitter (+/- m)
    ramp_yaw_deg: float = tunable(15.0)     # ramp yaw jitter (+/- deg)
    slot_jitter: float = tunable(0.03)      # ground spawn slot xy jitter (+/- m)

    # --- info: ramp (SLOPE frame: origin = face centre, +x upslope, +z face normal) --------------
    slope_deg: float = info(25.0)
    slope_len: float = info(0.48)
    slope_w: float = info(0.24)
    top_t: float = info(0.012)              # face layer thickness = pocket depth
    base_t: float = info(0.020)
    pocket_xs: tuple = info((-0.075, 0.015, 0.105))  # pocket centres (slope x, on centreline)
    pocket_len: float = info(0.024)         # pocket opening along-slope
    pocket_w: float = info(0.052)           # ... and across
    mu_face: float = info(0.10)             # slick face (pair-averaged with the objects)
    ramp_pos: tuple = info((0.46, 0.0))     # world xy of the face centre
    # --- info: chock (origin = plate-bottom / tab-top centre) ------------------------------------
    plate_t: float = info(0.014)
    plate_w: float = info(0.060)
    plate_h: float = info(0.045)
    tab_w: float = info(0.040)
    tab_h: float = info(0.011)
    foot_len: float = info(0.030)
    foot_h: float = info(0.008)
    chock_mass: float = info(0.15)
    chock_mu: float = info(0.40)
    # --- info: cube / band / marker / misc -------------------------------------------------------
    cube_s: float = info(0.05)
    cube_mass: float = info(0.10)
    cube_mu: float = info(0.40)
    band_off: float = info(0.032)           # band centre = pocket_x + this (= plate_t/2 + s/2 + 0.0)
    marker_size: tuple = info((0.060, 0.20, 0.003))
    slot_x: float = info(-0.32)             # ground spawn slots, ramp-YAW frame (m)
    slot_y: float = info(0.18)
    ground_mu: float = info(0.35)
    contact_offset: float = info(0.002)
    red_color: tuple = info((0.85, 0.10, 0.10))
    marker_color: tuple = info((0.95, 0.85, 0.10))
    # rubric weights (0.375 + 0.375 = 0.75 — exact in float32)
    w_seated: float = info(0.375)
    w_parked: float = info(0.375)

    # Derived (filled in __post_init__).
    ramp_pz: float = field(default=0.0, init=False)     # world z of the face centre
    bands: tuple = field(default=(), init=False)        # band centres (slope x)

    def __post_init__(self) -> None:
        th = math.radians(self.slope_deg)
        s = self.cube_s
        self.ramp_pz = (self.slope_len / 2) * math.sin(th) + 0.001
        self.bands = tuple(x + self.band_off for x in self.pocket_xs)
        # band centre sits exactly where a cube touching a centred chock's plate rests:
        assert abs(self.band_off - (self.plate_t / 2 + s / 2)) < 1e-9, \
            "band_off must equal plate_t/2 + cube_s/2"
        # slick face: no unsupported rest for EITHER object (PhysX averages materials):
        for mu_obj, tag in ((self.cube_mu, "cube"), (self.chock_mu, "chock")):
            pair = (mu_obj + self.mu_face) / 2
            assert math.degrees(math.atan(pair)) < self.slope_deg - 6.0, \
                f"face must be steeper than the {tag} friction angle by >= 6 deg"
        # tab/pocket: real play for a hand, tab shallower than the pocket (flush seat),
        # and the unkeyed rest (~tab_h proud) far outside the seated z-window:
        assert self.pocket_len - self.plate_t >= 0.008, "pocket along-slope play >= 8 mm"
        assert self.pocket_w - self.tab_w >= 0.010, "pocket across play >= 10 mm"
        assert self.tab_h < self.top_t - 0.0005, "tab must be shallower than the pocket"
        assert self.seat_z_lo < 0.0 < self.seat_z_hi < self.tab_h - 0.004, \
            "seated z-window must exclude the unkeyed (proud) rest"
        # the cube can never drop into a pocket:
        assert self.pocket_len < s - 0.015, "pockets must be far smaller than the cube"
        # anti-tip: restoring moment about the downhill foot edge >= 2x the cube's
        # slide-load moment (CoM at the body origin, i.e. at foot height ~0):
        g, cth, sth = 9.81, math.cos(th), math.sin(th)
        lever = self.plate_t / 2 + self.foot_len
        m_restore = self.chock_mass * g * cth * lever
        m_tip = self.cube_mass * g * sth * (s / 2)
        assert m_restore >= 2.0 * m_tip, \
            f"anti-tip margin {m_restore / m_tip:.2f}x < 2x"
        # the cube pushes the plate face-on and cannot ride over it:
        assert self.plate_h > 0.7 * s, "plate must face most of the cube"
        # impact capture: the pocket's downhill wall blocks any pitch-out until the
        # tab LIFTS clear (tab_h), so un-keying the chock costs m_ch*g*cos(th)*tab_h;
        # the sliding cube's plastic-collision energy share must fall >= 2x short:
        mu_pair = (self.cube_mu + self.mu_face) / 2
        a_slide = g * (sth - mu_pair * cth)
        ke_hit = self.cube_mass * a_slide * 0.045          # worst release-to-plate run
        e_chock = ke_hit * self.cube_mass / (self.cube_mass + self.chock_mass)
        e_lift = self.chock_mass * g * cth * self.tab_h
        assert e_lift >= 2.0 * e_chock, \
            f"impact-capture margin {e_lift / e_chock:.2f}x < 2x"
        # wrong-station parks land far outside the band:
        gaps = [b - a for a, b in zip(self.pocket_xs, self.pocket_xs[1:])]
        assert min(gaps) > 2 * self.band_half + 0.02, "stations must separate the bands"
        # a release just above ANY band stays on the face, and the lowest pocket
        # leaves run-out to the foot edge:
        assert max(self.bands) + self.band_half + s < self.slope_len / 2, \
            "release span above the top band must stay on the face"
        assert min(self.pocket_xs) - self.pocket_len / 2 > -self.slope_len / 2 + 0.05, \
            "lowest pocket must sit clear of the foot edge"
        # ground spawn slots clear the ramp's foot-edge footprint by a real margin:
        foot_reach = (self.slope_len / 2) * math.cos(th)
        assert abs(self.slot_x) - self.slot_jitter - 0.05 > foot_reach, \
            "spawn slots must clear the ramp foot"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("chock_ramp")
class ChockRampScene(BaseScene):
    cfg: ChockRampSceneCfg

    def __init__(self, cfg: ChockRampSceneCfg | None = None) -> None:
        super().__init__(cfg or ChockRampSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        ramp_spawn = cls["ramp"](
            slope_len=c.slope_len, slope_w=c.slope_w, top_t=c.top_t, base_t=c.base_t,
            pocket_xs=c.pocket_xs, pocket_len=c.pocket_len, pocket_w=c.pocket_w,
            mu_face=c.mu_face, mu_face_dyn=max(c.mu_face - 0.02, 0.02),
            contact_offset=c.contact_offset)
        chock_spawn = cls["chock"](
            plate_t=c.plate_t, plate_w=c.plate_w, plate_h=c.plate_h,
            tab_w=c.tab_w, tab_h=c.tab_h, foot_len=c.foot_len, foot_h=c.foot_h,
            mass=c.chock_mass, mu=c.chock_mu, mu_dyn=c.chock_mu - 0.05,
            contact_offset=c.contact_offset)

        th2 = math.radians(-c.slope_deg) / 2  # ramp default tilt: qy(-slope)

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.ground_mu, dynamic_friction=c.ground_mu - 0.05,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "ramp": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ramp",
                spawn=ramp_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.ramp_pos[0], c.ramp_pos[1], c.ramp_pz),
                    rot=(math.cos(th2), 0.0, math.sin(th2), 0.0)),
            ),
            "chock": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Chock",
                spawn=chock_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.ramp_pos[0] + c.slot_x, c.ramp_pos[1] - c.slot_y,
                         c.plate_t / 2 + 0.001),
                    rot=(math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)),  # lying on the +x face
            ),
            "cube": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube",
                spawn=sim_utils.CuboidCfg(
                    size=(c.cube_s, c.cube_s, c.cube_s),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05,
                        angular_damping=0.20,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.cube_mu, dynamic_friction=c.cube_mu - 0.05,
                        restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.red_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.ramp_pos[0] + c.slot_x, c.ramp_pos[1] + c.slot_y,
                         c.cube_s / 2 + 0.002)),
            ),
            # visual-only band marker: NO collision props => no collider is authored;
            # it can never arrest or park anything (the smoke battery slides the cube
            # straight across it).
            "marker": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BandMarker",
                spawn=sim_utils.CuboidCfg(
                    size=c.marker_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
                    collision_props=None,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.marker_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.ramp_pos[0], c.ramp_pos[1], c.ramp_pz + 0.05)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,  # quasi-static: a short slick slide and a keyed drop
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
        super().bind(env)
        self.ramp: RigidObject = env.iscene["ramp"]
        self.chock: RigidObject = env.iscene["chock"]
        self.cube: RigidObject = env.iscene["cube"]
        self.marker: RigidObject = env.iscene["marker"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        c = self.cfg
        self.pockets_t = torch.tensor(c.pocket_xs, device=dev)
        self.bands_t = torch.tensor(c.bands, device=dev)
        self.k_star = torch.zeros(n, dtype=torch.long, device=dev)
        self.swap = torch.ones(n, device=dev)  # +1: chock spawns on the -y slot
        # latches (partial credit survives transients); stillness streak (post_step only)
        self._l_seated = torch.zeros(n, dtype=torch.bool, device=dev)
        self._l_parked = torch.zeros(n, dtype=torch.bool, device=dev)
        self._streak = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: ramp pose (xy jitter + yaw), starred station k* in {0,1,2}
        (marker re-posed onto that band), chock and cube on jittered, randomly
        swapped ground slots downhill of the foot. All draws via torch.rand /
        torch.randint (readback-verified by the smoke battery); a burn draw guards
        the degenerate-first-draw quirk before the discrete station choice."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        th = math.radians(c.slope_deg)

        _ = torch.rand(m, 4, device=dev)  # burn (first post-seed draws are degenerate)

        def rnd(k: float) -> torch.Tensor:
            return (torch.rand(m, device=dev) * 2 - 1) * k

        def write(body, xy, z, q) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            st[:, 3:7] = q
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        gx = c.ramp_pos[0] + rnd(c.ramp_jitter)
        gy = c.ramp_pos[1] + rnd(c.ramp_jitter)
        gyaw = rnd(math.radians(c.ramp_yaw_deg))
        gq = _qmul(_qz(gyaw), _qy(torch.full((m,), -th, device=dev)))
        gxy = torch.stack([gx, gy], dim=-1)
        gpos3 = torch.cat([gxy, torch.full((m, 1), c.ramp_pz, device=dev)], dim=-1)
        write(self.ramp, gxy, torch.full((m,), c.ramp_pz, device=dev), gq)

        # starred station + marker onto that band (slope frame -> world)
        k = torch.randint(0, 3, (m,), device=dev)
        self.k_star[env_ids] = k
        band_local = torch.zeros(m, 3, device=dev)
        band_local[:, 0] = self.bands_t[k]
        band_local[:, 2] = c.marker_size[2] / 2 + 0.0006
        mpos = gpos3 + _qapply(gq, band_local)
        write(self.marker, mpos[:, :2], mpos[:, 2], gq)

        # ground slots in the ramp-YAW frame (downhill of the foot), swapped sides
        swap = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
        self.swap[env_ids] = swap
        qy90 = _qy(torch.full((m,), math.pi / 2, device=dev))
        yawq = _qz(gyaw)
        for body, s, z, q in (
                (self.chock, swap, c.plate_t / 2 + 0.001,
                 _qmul(_qz(gyaw + rnd(math.radians(180.0))), qy90)),
                (self.cube, -swap, c.cube_s / 2 + 0.002,
                 _qz(gyaw + rnd(math.radians(180.0))))):
            sl = torch.stack([
                c.slot_x + rnd(c.slot_jitter),
                -s * c.slot_y + rnd(c.slot_jitter),
                torch.zeros(m, device=dev)], dim=-1)
            w = gxy + _qapply(yawq, sl)[:, :2]
            write(body, w, torch.full((m,), z, device=dev), q)

        self._l_seated[env_ids] = False
        self._l_parked[env_ids] = False
        self._streak[env_ids] = 0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "ramp": self.ramp.data.root_state_w[env_ids].clone(),
            "chock": self.chock.data.root_state_w[env_ids].clone(),
            "cube": self.cube.data.root_state_w[env_ids].clone(),
            "marker": self.marker.data.root_state_w[env_ids].clone(),
            "k_star": self.k_star[env_ids].clone(),
            "swap": self.swap[env_ids].clone(),
            "l_seated": self._l_seated[env_ids].clone(),
            "l_parked": self._l_parked[env_ids].clone(),
            "streak": self._streak[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.ramp.write_root_state_to_sim(state["ramp"], env_ids)
        self.chock.write_root_state_to_sim(state["chock"], env_ids)
        self.cube.write_root_state_to_sim(state["cube"], env_ids)
        self.marker.write_root_state_to_sim(state["marker"], env_ids)
        self.k_star[env_ids] = state["k_star"]
        self.swap[env_ids] = state["swap"]
        self._l_seated[env_ids] = state["l_seated"]
        self._l_parked[env_ids] = state["l_parked"]
        self._streak[env_ids] = state["streak"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        mm = 1000
        return (
            f"A {c.slope_deg:.0f}-degree RAMP stands on the floor: a slick grey face "
            f"{c.slope_len * mm:.0f} mm long and {c.slope_w * mm:.0f} mm wide, low edge "
            f"near the floor, high edge ~{(c.ramp_pz + 0.1):.2f} m up. The face is far "
            f"too slippery for anything to rest on unsupported — a cube or the chock "
            f"laid anywhere on it slides straight off the low edge. Down the face's "
            f"centreline are three identical dark rectangular POCKETS (sockets "
            f"{c.pocket_len * mm:.0f} x {c.pocket_w * mm:.0f} mm, {c.top_t * mm:.0f} mm "
            f"deep, ~90 mm apart). A thin YELLOW BAND is painted across the face just "
            f"UPHILL of one pocket — which pocket varies per episode, as do the ramp's "
            f"position and heading. On the floor near the ramp lie a RED CUBE "
            f"({c.cube_s * mm:.0f} mm) and a BLUE CHOCK: an upright plate with a wide "
            f"foot on one side and a small rectangular TAB ({c.plate_t * mm:.0f} x "
            f"{c.tab_w * mm:.0f} x {c.tab_h * mm:.0f} mm) protruding from its underside. "
            f"The tab keys into any pocket; seated, the plate stands flush on the face, "
            f"upright plate uphill-facing, foot pointing downhill.\n"
            f"Goal: the RED CUBE must end up at rest ON the ramp face, centred on the "
            f"yellow band. Nothing rests there unsupported, so FIRST seat the chock: "
            f"drop its tab into the pocket immediately DOWNHILL of the yellow band "
            f"(the tab-in-pocket anchors the chock; a chock merely laid on the face "
            f"slides away). THEN lay the red cube on the face a little uphill of the "
            f"band and release it: it slides down and is caught by the chock's plate, "
            f"coming to rest inside the band. The order is forced — with no chock "
            f"seated the cube cannot stop on the face — and the station matters: a "
            f"chock in either other pocket parks the cube ~90 mm outside the band, "
            f"which scores nothing. Only the settled final state is judged: cube still, "
            f"braced against the seated chock, inside the band."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Seat the blue chock's tab in the ramp pocket just downhill of the yellow "
            "band, then lay the red cube on the ramp above the band and release it so "
            "it slides down and comes to rest against the chock, inside the band. The "
            "ramp is too slippery for the cube to stop anywhere unsupported, so the "
            "chock must be seated first, and in that pocket."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points (N,3) -> the SLOPE frame (face centre, +x upslope, +z normal)."""
        return _qapply(_qinv(self.ramp.data.root_quat_w),
                       pos_w - self.ramp.data.root_pos_w)

    def chock_seated(self) -> torch.Tensor:
        """(N,) bool, LIVE: chock keyed into the STARRED pocket — origin over the
        pocket centre, FLUSH on the face (the seated z-window rejects the ~11 mm
        proud unkeyed rest), aligned with the slope frame."""
        c = self.cfg
        p = self._local(self.chock.data.root_pos_w)
        xk = self.pockets_t[self.k_star]
        dq = _qmul(_qinv(self.ramp.data.root_quat_w), self.chock.data.root_quat_w)
        ang = 2.0 * torch.acos(dq[:, 0].abs().clamp(max=1.0))
        return ((p[:, 0] - xk).abs() < c.seat_tol_x) \
            & (p[:, 1].abs() < c.seat_tol_y) \
            & (p[:, 2] > c.seat_z_lo) & (p[:, 2] < c.seat_z_hi) \
            & (ang < math.radians(c.seat_ang_deg))

    def on_band(self) -> torch.Tensor:
        """(N,) bool, LIVE: cube centre resting ON the face inside the starred band
        (z window = on-face rest; a cube on top of a fallen chock sits proud)."""
        c = self.cfg
        p = self._local(self.cube.data.root_pos_w)
        b = self.bands_t[self.k_star]
        return ((p[:, 0] - b).abs() <= c.band_half) \
            & (p[:, 1].abs() < 0.06) \
            & ((p[:, 2] - c.cube_s / 2).abs() < 0.008)

    def braced(self) -> torch.Tensor:
        """(N,) bool, LIVE: cube on the face with its downhill face at the seated
        chock's plate (gap in a small window), laterally engaging the plate."""
        c = self.cfg
        pc = self._local(self.cube.data.root_pos_w)
        pk = self._local(self.chock.data.root_pos_w)
        gap = pc[:, 0] - (pk[:, 0] + c.plate_t / 2 + c.cube_s / 2)
        return self.chock_seated() \
            & ((pc[:, 2] - c.cube_s / 2).abs() < 0.008) \
            & (gap > c.gap_lo) & (gap < c.gap_hi) \
            & ((pc[:, 1] - pk[:, 1]).abs() < 0.045)

    def _still(self) -> torch.Tensor:
        sl, sa = self.cfg.settle_lin, self.cfg.settle_ang
        ok = torch.ones_like(self._l_seated)
        for b in (self.cube, self.chock):
            ok &= (b.data.root_lin_vel_w.norm(dim=-1) < sl) \
                & (b.data.root_ang_vel_w.norm(dim=-1) < sa)
        return ok

    def _finite(self) -> torch.Tensor:
        p = torch.stack([self.cube.data.root_pos_w, self.chock.data.root_pos_w,
                         self.ramp.data.root_pos_w], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _ok(self) -> torch.Tensor:
        """The success condition sans stillness-duration: braced in the band, finite."""
        return self.braced() & self.on_band() & self._finite()

    def _update_latches(self) -> None:
        """Idempotent OR-latches only (safe to call from success/score between steps)."""
        fin = self._finite()
        seat_now = fin & self.chock_seated() \
            & (self.chock.data.root_lin_vel_w.norm(dim=-1) < self.cfg.seat_vel)
        park_now = fin & self.braced() & self.on_band() \
            & (self.cube.data.root_lin_vel_w.norm(dim=-1) < self.cfg.park_vel)
        self._l_parked |= park_now
        self._l_seated |= seat_now | self._l_parked

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latches + the stillness STREAK. The streak is non-idempotent, so it ticks
        HERE ONLY (success()/score() never advance it)."""
        self._update_latches()
        good = self._ok() & self._still()
        self._streak = torch.where(good, self._streak + 1,
                                   torch.zeros_like(self._streak))

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the cube braced by the seated chock inside the band, finite, and
        STILL for >= settle_steps consecutive physics steps. A live outcome — pull
        the chock or nudge the cube out and this returns False again."""
        self._update_latches()
        return self._ok() & self._still() & (self._streak >= self.cfg.settle_steps)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.375*seated + 0.375*parked (latched; ~0 for doing
        nothing), capped at 0.75 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_seated * self._l_seated.float()
                + c.w_parked * self._l_parked.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="chock_ramp", robot="null"))

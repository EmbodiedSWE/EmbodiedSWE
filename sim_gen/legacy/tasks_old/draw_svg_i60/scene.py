"""GalleryEaselScene — flip the finished canvases face-up and display them leaning on an
easel rack; keep the blank canvas off the rack (sim_gen task `draw_svg_i60`, derived from
maniskill/draw_svg).

The seed is CONTINUOUS TRAJECTORY DRAWING: a Franka drags a red-cube marker along a
prescribed SVG path on a flat canvas — one held tool, one long guarded sweep, judged on
the traced path history. This task keeps the seed's world (finished 2D artworks on flat
canvas plaques) and discards the entire plan class:

  - NOTHING is drawn and nothing traces. The artworks arrive FINISHED — flat rigid canvas
    plaques lying FACE-DOWN scattered on the studio floor (art surface against the ground,
    blank canvas-back up).
  - The goal is a static DISPLAY arrangement: every artwork must end LEANING on a 3-slot
    gallery easel rack — bottom edge on the slot shelf behind the retaining lip, back
    resting against the tilted backrest, art surface facing OUT toward the viewer, each
    artwork in its OWN slot, settled. Solving means per-plaque 180-degree FLIP
    reorientation (face-down -> face-out) plus an INCLINED TWO-CONTACT EQUILIBRIUM
    placement — skills the seed never touches (its marker stays in one flat pose all
    episode).
  - A BLANK plaque (no art on either face, otherwise identical) is sometimes present and
    must be WITHHELD: a blank parked in any slot region blocks success and is penalized.
  - The seed's own strategy is expressible only as its terminal disposition — artwork
    lying flat, art-up, on the work surface (exactly how a drawing episode ends) — and
    that state scores 0 here (tested negative control): flat fails the tilt band, the
    floor fails the slot region, and no trace/path term exists to reward the sweep.

Judged on PHYSICAL outcomes only (settled poses read back from sim, in the easel's frame
so the jittered/yawed rack judges identically):
  - `displayed()`: plaque bottom seated in a slot (xy band — honest by construction: the
    slot dividers cap the physical lateral slack at 15 mm << the 35 mm gate), centre in
    the leaning x/z band (rejects flat-on-shelf, on-the-ground, perched-on-lip poses),
    height axis tilted BACK 6-48 degrees from vertical (a flat plaque reads 90), art
    normal pointing OUT (face-in lean rejected), settled.
  - success(): every present artwork displayed, in pairwise DISTINCT slots, and no blank
    plaque anywhere in the rack region.
  - score(): 0 for doing nothing; w_first (0.10, latched in post_step) once any artwork
    was ever genuinely displayed; + w_frac (0.60) * displayed fraction; + w_allnow (0.15)
    when all are up in distinct slots; - blank_penalty (0.10) while a blank squats in the
    rack; exactly 1.0 iff success. Max non-success 0.85 (blank-strayed full rack: 0.75).

Per-episode randomization: easel pose (xy jitter + yaw — slot axes rotate), plaque
scatter poses (arc slot + jitter + free yaw, always face-down), and subset sampling of
both the third artwork AND the blank distractor (solve what you see). Absent plaques park
in an off-camera depot.

Assets are fully procedural (no external files): a KINEMATIC easel rack (shelf slab +
tilted backrest + retaining lip + slot fins, one body, re-posed per reset by pose
writes) and four dynamic plaques (box collider + visual-only art slab on the +z face —
the pen-tip pattern). Heavy imports (isaaclab, pxr) are deferred so importing this
module — and registering the scene — stays app-free.
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


# ----- custom compound spawners ----------------------------------------------------------------
# One rigid body each, authored with raw pxr APIs; `isaaclab.sim.utils.clone` supplies the
# regex-resolve + per-env replication (each cloned prim is authored fresh — idempotent, no
# duplicate-xformOp trap).

_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_easel(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC 3-slot easel rack at `prim_path`. Body origin = centre of the
    SHELF TOP face (z=0 local): the shelf slab hangs below (resting on the ground), the
    retaining lip rises at the front edge (+x = viewer side), the backrest panel rises
    from the shelf's rear edge tilted BACK by `beta_deg` (top leans -x), and vertical
    fins split the shelf into `n_slots` slots along y. The backrest front face passes
    through (rest_x0, z=0) with outward normal (cos b, 0, sin b): a RotateYOp of -beta
    maps the panel's local long axis +z onto (-sin b, 0, cos b) and its thickness axis +x
    onto that outward normal."""
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
    rb.CreateKinematicEnabledAttr(True)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    def box(name: str, center, scale, color, rot_y_deg: float = 0.0) -> None:
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(*center))
        if rot_y_deg:
            sxf.AddRotateYOp().Set(rot_y_deg)
        sxf.AddScaleOp().Set(Gf.Vec3f(*scale))
        seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        collide(seg.GetPrim())

    b = math.radians(cfg.beta_deg)
    width = cfg.n_slots * (cfg.slot_w + cfg.fin_t) + cfg.fin_t
    # shelf slab: top face at local z=0, resting on the ground
    box("shelf", (0.0, 0.0, -cfg.shelf_t / 2), (cfg.shelf_d, width, cfg.shelf_t),
        cfg.wood_color)
    # backrest panel: front face through (rest_x0, 0), leaning back by beta
    cx = cfg.rest_x0 - math.sin(b) * cfg.back_len / 2 - math.cos(b) * cfg.back_t / 2
    cz = math.cos(b) * cfg.back_len / 2 - math.sin(b) * cfg.back_t / 2
    box("backrest", (cx, 0.0, cz), (cfg.back_t, width, cfg.back_len),
        cfg.back_color, rot_y_deg=-cfg.beta_deg)
    # retaining lip along the front edge
    box("lip", ((cfg.lip_in + cfg.lip_in + cfg.lip_t) / 2, 0.0, cfg.lip_h / 2),
        (cfg.lip_t, width, cfg.lip_h), cfg.trim_color)
    # slot fins: n_slots+1 boundaries along y
    pitch = cfg.slot_w + cfg.fin_t
    for m in range(cfg.n_slots + 1):
        yb = (m - cfg.n_slots / 2) * pitch
        box(f"fin_{m}", (0.0, yb, cfg.fin_h / 2), (cfg.shelf_d, cfg.fin_t, cfg.fin_h),
            cfg.trim_color)
    return root


def _spawn_plaque(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one canvas plaque at `prim_path`: root Xform with RigidBodyAPI + explicit
    MassAPI, a single box collider (LOCAL AXES: +x = height axis, length `ph`; +y =
    width `pw`; +z = art normal, thickness `pt`), and — for artworks only — a
    VISUAL-ONLY thin colored art slab on the +z face (no CollisionAPI, the pen-tip
    pattern), so art vs canvas-back is visible while contact stays a clean box."""
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
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    # Gentle overlap resolution: the oracle authors the plaque 1.5 mm off the backrest
    # inside the 2 mm contact offset — the depenetration solver must nudge, not eject.
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.30)  # kill the lean-in rock so settle gates close fast

    body = UsdGeom.Cube.Define(stage, f"{prim_path}/body")
    body.CreateSizeAttr(1.0)
    bxf = UsdGeom.Xformable(body.GetPrim())
    bxf.AddScaleOp().Set(Gf.Vec3f(cfg.ph, cfg.pw, cfg.pt))
    body.CreateDisplayColorAttr([Gf.Vec3f(*cfg.back_color)])
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(body.GetPrim())
    px.CreateContactOffsetAttr(float(cfg.contact_offset))
    px.CreateRestOffsetAttr(0.0)

    if cfg.has_art:  # visual only — NO CollisionAPI
        art = UsdGeom.Cube.Define(stage, f"{prim_path}/art")
        art.CreateSizeAttr(1.0)
        axf = UsdGeom.Xformable(art.GetPrim())
        axf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, cfg.pt / 2 + 0.001))
        axf.AddScaleOp().Set(Gf.Vec3f(cfg.ph - 0.024, cfg.pw - 0.024, 0.002))
        art.CreateDisplayColorAttr([Gf.Vec3f(*cfg.art_color)])
    return root


def _easel_spawner_cfg(*, beta_deg: float, rest_x0: float, shelf_d: float, shelf_t: float,
                       lip_in: float, lip_t: float, lip_h: float, slot_w: float,
                       fin_t: float, fin_h: float, n_slots: int, back_len: float,
                       back_t: float, wood_color: tuple, back_color: tuple,
                       trim_color: tuple, contact_offset: float) -> Any:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "easel" not in _SPAWNER_CACHE:

        @configclass
        class EaselSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_easel)
            beta_deg: float = 18.0
            rest_x0: float = -0.045
            shelf_d: float = 0.11
            shelf_t: float = 0.03
            lip_in: float = 0.043
            lip_t: float = 0.012
            lip_h: float = 0.014
            slot_w: float = 0.15
            fin_t: float = 0.010
            fin_h: float = 0.05
            n_slots: int = 3
            back_len: float = 0.26
            back_t: float = 0.012
            wood_color: tuple = (0.45, 0.32, 0.20)
            back_color: tuple = (0.55, 0.42, 0.28)
            trim_color: tuple = (0.30, 0.21, 0.13)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["easel"] = EaselSpawnerCfg

    return _SPAWNER_CACHE["easel"](
        beta_deg=beta_deg, rest_x0=rest_x0, shelf_d=shelf_d, shelf_t=shelf_t,
        lip_in=lip_in, lip_t=lip_t, lip_h=lip_h, slot_w=slot_w, fin_t=fin_t, fin_h=fin_h,
        n_slots=n_slots, back_len=back_len, back_t=back_t, wood_color=wood_color,
        back_color=back_color, trim_color=trim_color, contact_offset=contact_offset,
    )


def _plaque_spawner_cfg(*, ph: float, pw: float, pt: float, mass: float, has_art: bool,
                        art_color: tuple, back_color: tuple, contact_offset: float) -> Any:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "plaque" not in _SPAWNER_CACHE:

        @configclass
        class PlaqueSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_plaque)
            ph: float = 0.16
            pw: float = 0.12
            pt: float = 0.018
            mass: float = 0.10
            has_art: bool = True
            art_color: tuple = (0.75, 0.12, 0.15)
            back_color: tuple = (0.82, 0.78, 0.70)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["plaque"] = PlaqueSpawnerCfg

    return _SPAWNER_CACHE["plaque"](
        ph=ph, pw=pw, pt=pt, mass=mass, has_art=has_art, art_color=art_color,
        back_color=back_color, contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class GalleryEaselSceneCfg(BaseCfg):
    """Config for `GalleryEaselScene`. Easel local frame: origin at the centre of the
    shelf TOP face, +x toward the viewer (front), slots along y, +z up. Manifest order:
    plaques 0-2 are artworks (crimson / teal / amber art faces), plaque 3 is the blank."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    tilt_min_deg: float = tunable(6.0)   # displayed lean band, degrees back from vertical.
    tilt_max_deg: float = tunable(48.0)  # Physical rest angles span ~18-40 deg (backrest 18
    # deg; bottom edge pulled to the lip steepens the chord); a flat plaque reads 90 —
    # rejected; upright-frozen would read 0 — rejected (and is not a physical rest pose).
    face_gate: float = tunable(0.45)     # min dot(art normal, easel outward +x): a correct
    # lean scores ~+0.95, a face-in lean ~-0.95 — the gate splits them with huge margin.
    lean_back_gate: float = tunable(0.05)  # the top must lean TOWARD the backrest (w_x <=
    # -gate in the easel frame): rejects a plaque propped forward against the lip.
    slot_dy_gate: float = tunable(0.035)  # |y - slot centre| in the easel frame. Honest by
    # construction: the slot fins cap the physical bottom-in-slot slack at
    # (slot_w - pw)/2 = 15 mm < 35 mm, so any plaque genuinely seated in a slot counts.
    x_band: tuple = tunable((-0.105, 0.045))  # centre x band (easel frame): all physical
    # leans (theta 8-45 deg, bottom edge anywhere between backrest base and lip) land
    # inside; a plaque on the ground in front of the rack (x ~ +0.2) is far outside.
    z_band: tuple = tunable((0.045, 0.098))   # centre z band above the shelf top: leans
    # span 0.063-0.081; flat-on-shelf reads 0.009, flat-on-ground reads -0.021 — rejected.
    settle_lin: float = tunable(0.05)   # max |lin vel| when judging settled (m/s)
    settle_ang: float = tunable(0.8)    # max |ang vel| (rad/s)

    # --- tunable: score weights --------------------------------------------------------------
    w_first: float = tunable(0.10)      # latched: some artwork was once genuinely displayed
    w_frac: float = tunable(0.60)       # x fraction of present artworks displayed now
    w_allnow: float = tunable(0.15)     # all displayed in distinct slots (pre-success state)
    blank_penalty: float = tunable(0.10)  # while a blank plaque squats in the rack region

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    subset_sample: bool = tunable(True)  # sample presence of artwork 2 and the blank
    easel_jitter: float = tunable(0.05)  # uniform +/- xy jitter of the easel at reset
    easel_yaw_deg: float = tunable(45.0)  # uniform +/- yaw of the easel (slot axes rotate)
    scatter_jitter: float = tunable(0.03)  # per-plaque xy jitter on the scatter arc
    scatter_yaw_deg: float = tunable(180.0)  # free yaw of each face-down plaque

    # --- info: structure ---------------------------------------------------------------------
    beta_deg: float = info(18.0)   # backrest tilt back from vertical
    rest_x0: float = info(-0.045)  # backrest front face crosses the shelf top plane here
    shelf_d: float = info(0.11)    # shelf depth (x span -0.055..+0.055)
    shelf_t: float = info(0.03)    # shelf slab thickness (top face = local z 0, on ground)
    lip_in: float = info(0.043)    # retaining lip inner face x
    lip_t: float = info(0.012)
    lip_h: float = info(0.014)
    slot_w: float = info(0.15)     # slot inner width; plaque pw=0.12 -> 15 mm lateral slack
    fin_t: float = info(0.010)
    fin_h: float = info(0.05)
    n_slots: int = info(3)
    back_len: float = info(0.26)   # backrest panel length (top at z ~ 0.247 — a leaning
    # plaque's top contact lands at z ~ 0.15, well inside the panel)
    back_t: float = info(0.012)
    ph: float = info(0.16)         # plaque height (local +x = height axis)
    pw: float = info(0.12)         # plaque width — 0.16 diagonal keying: a landscape lean
    # cannot seat (ph=0.16 > slot_w=0.15), so portrait orientation is enforced physically
    pt: float = info(0.018)        # plaque thickness (local +z = art normal)
    plaque_mass: float = info(0.10)
    art_colors: tuple = info(((0.75, 0.12, 0.15), (0.10, 0.55, 0.55), (0.90, 0.65, 0.10)))
    back_color: tuple = info((0.82, 0.78, 0.70))  # raw canvas — both faces of the blank
    wood_color: tuple = info((0.45, 0.32, 0.20))
    backrest_color: tuple = info((0.55, 0.42, 0.28))
    trim_color: tuple = info((0.30, 0.21, 0.13))
    easel_pos: tuple = info((0.20, 0.0))
    scatter_center: tuple = info((-0.20, 0.0))
    scatter_r: float = info(0.20)
    scatter_angles: tuple = info((140.0, 205.0, 265.0, 325.0))  # arc slots, degrees
    friction: float = info(0.60)
    contact_offset: float = info(0.002)
    parking_pos: tuple = info((1.1, 1.1))  # off-camera ground depot for absent plaques
    # rack region for the blank-stray clause (easel frame; generously covers every slot,
    # the shelf, the lip and the lean volume — but NOT the floor in front of the rack)
    region_x: tuple = info((-0.11, 0.055))
    region_z: tuple = info((-0.01, 0.28))

    # Derived (filled in __post_init__).
    n_plaques: int = field(default=None, init=False)
    slot_y: tuple = field(default=None, init=False)
    shelf_top_z: float = field(default=None, init=False)
    region_y: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.n_plaques = 4  # 3 artworks + 1 blank
        pitch = self.slot_w + self.fin_t
        self.slot_y = tuple(round((i - (self.n_slots - 1) / 2) * pitch, 4)
                            for i in range(self.n_slots))
        self.shelf_top_z = self.shelf_t  # easel root z: slab bottom on the ground
        self.region_y = round(self.n_slots * pitch / 2 + self.fin_t, 4)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("gallery_easel")
class GalleryEaselScene(BaseScene):
    cfg: GalleryEaselSceneCfg

    def __init__(self, cfg: GalleryEaselSceneCfg | None = None) -> None:
        super().__init__(cfg or GalleryEaselSceneCfg())

    # ----- assets ---------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic easel rack, three artwork plaques and one blank.
        reset() re-poses everything."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
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
            "easel": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Easel",
                spawn=_easel_spawner_cfg(
                    beta_deg=c.beta_deg, rest_x0=c.rest_x0, shelf_d=c.shelf_d,
                    shelf_t=c.shelf_t, lip_in=c.lip_in, lip_t=c.lip_t, lip_h=c.lip_h,
                    slot_w=c.slot_w, fin_t=c.fin_t, fin_h=c.fin_h, n_slots=c.n_slots,
                    back_len=c.back_len, back_t=c.back_t, wood_color=c.wood_color,
                    back_color=c.backrest_color, trim_color=c.trim_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.easel_pos[0], c.easel_pos[1], c.shelf_top_z)),
            ),
        }
        for i in range(c.n_plaques):
            is_art = i < 3
            ang = math.radians(c.scatter_angles[i])
            out[self._name(i)] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plaque_" + self._name(i),
                spawn=_plaque_spawner_cfg(
                    ph=c.ph, pw=c.pw, pt=c.pt, mass=c.plaque_mass, has_art=is_art,
                    art_color=c.art_colors[i] if is_art else c.back_color,
                    back_color=c.back_color, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.scatter_center[0] + c.scatter_r * math.cos(ang),
                         c.scatter_center[1] + c.scatter_r * math.sin(ang),
                         c.pt / 2 + 0.002),
                    rot=(0.0, 1.0, 0.0, 0.0),  # face-down
                ),
            )
        return out

    @staticmethod
    def _name(i: int) -> str:
        return f"art_{i}" if i < 3 else "blank"

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles + allocate per-episode buffers."""
        super().bind(env)
        n, dev = env.num_envs, env.device
        c = self.cfg
        self.easel: RigidObject = env.iscene["easel"]
        self.plaques: list[RigidObject] = [env.iscene[self._name(i)]
                                           for i in range(c.n_plaques)]
        self.env_origins = env.iscene.env_origins
        self.easel_c = torch.zeros(n, 2, device=dev)   # easel centre (env-local xy)
        self.easel_yaw = torch.zeros(n, device=dev)
        self.present = torch.ones(n, c.n_plaques, dtype=torch.bool, device=dev)
        self.any_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self.slot_y_t = torch.tensor(c.slot_y, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: re-pose the kinematic easel (jitter + yaw), sample presence of
        artwork 2 and the blank, scatter present plaques FACE-DOWN on the arc (jitter +
        free yaw), park absent plaques in the ground depot, clear the latch."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- easel: kinematic pose-only write (jitter + yaw) ---
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.easel_yaw_deg)
        ec = torch.tensor(c.easel_pos, device=dev).expand(m, 2).clone()
        ec += (torch.rand(m, 2, device=dev) * 2 - 1) * c.easel_jitter
        self.easel_c[env_ids] = ec
        self.easel_yaw[env_ids] = yaw
        st = torch.zeros(m, 7, device=dev)
        st[:, 0:2] = ec
        st[:, 2] = c.shelf_top_z
        st[:, 3] = torch.cos(yaw / 2)
        st[:, 6] = torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.easel.write_root_pose_to_sim(st, env_ids)

        # --- presence: artworks 0-1 always; artwork 2 and the blank sampled ---
        pres = torch.ones(m, c.n_plaques, dtype=torch.bool, device=dev)
        if c.subset_sample:
            pres[:, 2] = torch.rand(m, device=dev) < 0.5
            pres[:, 3] = torch.rand(m, device=dev) < 0.5
        self.present[env_ids] = pres

        # --- plaques: face-down on the scatter arc; absent -> parking depot ---
        for i in range(c.n_plaques):
            ang = math.radians(c.scatter_angles[i])
            scat = torch.zeros(m, 3, device=dev)
            scat[:, 0] = c.scatter_center[0] + c.scatter_r * math.cos(ang)
            scat[:, 1] = c.scatter_center[1] + c.scatter_r * math.sin(ang)
            scat[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.scatter_jitter
            scat[:, 2] = c.pt / 2 + 0.002
            park = torch.zeros(m, 3, device=dev)
            park[:, 0] = c.parking_pos[0] + (i % 2) * 0.25
            park[:, 1] = c.parking_pos[1] + (i // 2) * 0.25
            park[:, 2] = c.pt / 2 + 0.002
            pr = pres[:, i].unsqueeze(1)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.where(pr, scat, park)
            # face-down with free yaw: q = q_z(yaw) * q_x(pi) = (0, cos y/2, sin y/2, 0)
            half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.scatter_yaw_deg) / 2
            st[:, 4] = torch.cos(half)
            st[:, 5] = torch.sin(half)
            self.plaques[i].write_root_state_to_sim(st, env_ids)

        self.any_ever[env_ids] = False

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch bookkeeping every physics substep: ANY artwork genuinely displayed once
        (settled lean, judged on live buffers) latches the first-display credit."""
        d = self.displayed() & self.present
        self.any_ever |= d[:, :3].any(dim=1)

    # ----- state (full, restorable) ----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "plaques": [b.data.root_state_w[env_ids].clone() for b in self.plaques],
            "easel": self.easel.data.root_state_w[env_ids].clone(),
            "easel_c": self.easel_c[env_ids].clone(),
            "easel_yaw": self.easel_yaw[env_ids].clone(),
            "present": self.present[env_ids].clone(),
            "any_ever": self.any_ever[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for b, st in zip(self.plaques, state["plaques"]):
            b.write_root_state_to_sim(st, env_ids)
        self.easel.write_root_pose_to_sim(state["easel"][:, 0:7], env_ids)
        self.easel_c[env_ids] = state["easel_c"]
        self.easel_yaw[env_ids] = state["easel_yaw"]
        self.present[env_ids] = state["present"]
        self.any_ever[env_ids] = state["any_ever"]

    # ----- description -----------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wooden gallery easel rack stands on the floor at a random heading: a low "
            f"shelf with a retaining lip, split by fins into {c.n_slots} display slots "
            f"({c.slot_w * 1000:.0f} mm wide), backed by a panel leaning back "
            f"{c.beta_deg:.0f} degrees. Scattered face-DOWN on the floor across from it "
            f"lie finished canvas plaques ({c.ph * 1000:.0f} x {c.pw * 1000:.0f} x "
            f"{c.pt * 1000:.0f} mm, painted on exactly one face — crimson, teal or amber; "
            f"the raw canvas back is beige). Two or three artworks are present, and "
            f"sometimes an entirely BLANK plaque (beige on both faces) lies among them — "
            f"count and check what you see.\n"
            f"Goal: put every artwork on display — flip it art-side-out and lean it in its "
            f"own slot (bottom edge on the shelf behind the lip, back against the panel, "
            f"painted face toward the viewer), all settled. A plaque displayed art-inward, "
            f"lying flat, or sharing a slot does not count, and the blank plaque must stay "
            f"OFF the rack — parking it in a slot blocks the goal until removed."
        )

    # ----- progress / rubric ------------------------------------------------------------------
    def _plaque_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos env-local (N,P,3), quat (N,P,4)) for all plaques, manifest order."""
        pos = torch.stack([b.data.root_pos_w - self.env_origins for b in self.plaques],
                          dim=1)
        quat = torch.stack([b.data.root_quat_w for b in self.plaques], dim=1)
        return pos, quat

    def _easel_frame(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-plaque geometry in the EASEL frame: (le (N,P,3) centre position with z
        measured above the shelf top, u (N,P,3) plaque height axis, n (N,P,3) art
        normal — axes rotated into the easel frame)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        pos, quat = self._plaque_tensors()
        n_env, p = pos.shape[0], pos.shape[1]
        qf = quat.reshape(n_env * p, 4)
        ex = torch.tensor([1.0, 0.0, 0.0], device=pos.device).expand(n_env * p, 3)
        ez = torch.tensor([0.0, 0.0, 1.0], device=pos.device).expand(n_env * p, 3)
        u = quat_apply(qf, ex).reshape(n_env, p, 3)
        nrm = quat_apply(qf, ez).reshape(n_env, p, 3)
        cy = torch.cos(self.easel_yaw)[:, None]
        sy = torch.sin(self.easel_yaw)[:, None]

        def to_easel(v: torch.Tensor) -> torch.Tensor:
            vx = v[:, :, 0] * cy + v[:, :, 1] * sy
            vy = -v[:, :, 0] * sy + v[:, :, 1] * cy
            return torch.stack([vx, vy, v[:, :, 2]], dim=-1)

        d = pos.clone()
        d[:, :, 0] -= self.easel_c[:, None, 0]
        d[:, :, 1] -= self.easel_c[:, None, 1]
        d[:, :, 2] -= c.shelf_top_z
        return to_easel(d), to_easel(u), to_easel(nrm)

    def settled(self) -> torch.Tensor:
        """(N, P) bool: per-plaque lin + ang velocity below the settle gates."""
        c = self.cfg
        cols = []
        for b in self.plaques:
            lin = b.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
            ang = b.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang
            cols.append(lin & ang)
        return torch.stack(cols, dim=1)

    def slot_index(self) -> torch.Tensor:
        """(N, P) long: nearest slot by easel-frame y."""
        le, _u, _n = self._easel_frame()
        dy = (le[:, :, 1:2] - self.slot_y_t[None, None, :]).abs()
        return dy.argmin(dim=-1)

    def displayed(self) -> torch.Tensor:
        """(N, P) bool, geometric + settled: bottom seated in a slot (y band), centre in
        the leaning x/z band, height axis tilted BACK 6-48 deg from vertical, art normal
        facing OUT, settled. All in the easel frame — the jittered/yawed rack judges
        identically. (The blank passes this predicate too if leaned; the rubric only
        counts artworks and flags the blank via `blank_stray`.)"""
        c = self.cfg
        le, u, nrm = self._easel_frame()
        # slot y band
        dy = (le[:, :, 1:2] - self.slot_y_t[None, None, :]).abs()
        slot_ok = (dy < c.slot_dy_gate).any(dim=-1)
        # leaning centre band
        x_ok = (le[:, :, 0] > c.x_band[0]) & (le[:, :, 0] < c.x_band[1])
        z_ok = (le[:, :, 2] > c.z_band[0]) & (le[:, :, 2] < c.z_band[1])
        # height axis, flipped so the "up" end is up; tilt back from vertical
        sgn = torch.where(u[:, :, 2] >= 0, 1.0, -1.0)
        w = u * sgn.unsqueeze(-1)
        wz = w[:, :, 2].clamp(-1.0, 1.0)
        tilt_ok = ((wz >= math.cos(math.radians(c.tilt_max_deg)))
                   & (wz <= math.cos(math.radians(c.tilt_min_deg))))
        lean_back = w[:, :, 0] <= -c.lean_back_gate
        # art normal out toward the viewer
        face_ok = nrm[:, :, 0] >= c.face_gate
        return slot_ok & x_ok & z_ok & tilt_ok & lean_back & face_ok & self.settled()

    def blank_stray(self) -> torch.Tensor:
        """(N,) bool: the blank plaque is present and its centre sits anywhere in the
        rack region (slots, shelf, lip or lean volume)."""
        c = self.cfg
        le, _u, _n = self._easel_frame()
        b = le[:, 3]
        in_region = ((b[:, 0] > c.region_x[0]) & (b[:, 0] < c.region_x[1])
                     & (b[:, 1].abs() < c.region_y)
                     & (b[:, 2] > c.region_z[0]) & (b[:, 2] < c.region_z[1]))
        return in_region & self.present[:, 3]

    def _arts_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(disp (N,3) displayed-and-present artworks, all_now (N,), distinct (N,))."""
        disp = self.displayed()[:, :3] & self.present[:, :3]
        all_now = disp.sum(dim=1) == self.present[:, :3].sum(dim=1)
        idx = self.slot_index()[:, :3]
        one_hot = torch.nn.functional.one_hot(idx, self.cfg.n_slots)
        counts = (one_hot * disp.unsqueeze(-1).long()).sum(dim=1)
        distinct = (counts <= 1).all(dim=-1)
        return disp, all_now, distinct

    def success(self) -> torch.Tensor:
        """(N,) bool: every present artwork displayed, in pairwise distinct slots, and no
        blank plaque in the rack region — the physical gallery goal."""
        _disp, all_now, distinct = self._arts_state()
        return all_now & distinct & ~self.blank_stray()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0 for doing nothing; w_first once any artwork was ever
        displayed (latched); + w_frac * displayed fraction; + w_allnow when all are up in
        distinct slots; - blank_penalty while a blank squats in the rack; exactly 1.0 iff
        success. Max non-success 0.85; blank-strayed full rack 0.75."""
        c = self.cfg
        disp, all_now, distinct = self._arts_state()
        k = self.present[:, :3].sum(dim=1).clamp(min=1).float()
        frac = disp.sum(dim=1).float() / k
        stray = self.blank_stray()
        base = (c.w_first * self.any_ever.float() + c.w_frac * frac
                + c.w_allnow * (all_now & distinct).float()
                - c.blank_penalty * stray.float())
        base = base.clamp(0.0, 0.90)
        succ = all_now & distinct & ~stray
        return torch.where(succ, torch.ones_like(base), base)


# ----- env registration ------------------------------------------------------------------------
register_env("simgen", lambda: EnvCfg(scene="gallery_easel", robot="null", env_spacing=4.0))

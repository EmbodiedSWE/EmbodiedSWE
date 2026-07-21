"""PenHolderScene — fill an open pen holder with scattered pens, tip-up (port).

The object world for the RoboDojo "fill-pen-holder" port: an open cup (the pen holder)
standing on the work surface and up to four pens lying flat, scattered on the other side.
**Goal (carried here, no task layer): put every present pen into the holder tip-up, then
leave the holder standing upright on the surface.**

This is a DIFFICULTY-FLOOR tier task (by design): meant to be SOLVABLE — a graded
0-100 rubric so weak agents rank instead of flatlining — while still honest bimanual
manipulation (the source robot picks the holder up with one hand and fills it with the
other; every insertion targets a compliant, moving cup).

Judged by the ported source rubric with verbatim thresholds, computed in the HOLDER'S
BODY FRAME so a held / tilted holder judges identically to a standing one (the source
fills a held holder): a pen counts when its bottom end is within `xy_tol` (3.5 cm,
source) of the holder axis, its depth below the rim exceeds `depth_min` (3.5 cm, source
depth-into-holder), its axis points tip-up along the holder axis (within
`pen_align_max_deg`), the holder itself is within `holder_tilt_max_deg` (45 deg, source)
of world-up, and pen + holder are settled. Transition scores [10, 25, 40] for 1/2/3 pens,
90 for ALL present pens in the (possibly still held) holder, 100 additionally requires
the loaded holder standing upright ON the work surface (`holder_placed` — the port of the
source's set-down; a top-heavy loaded cup tips easily, which is the real final stage).
NOTE the source's 100-score also requires both grippers >= 80% open and both
end-effectors back at their episode-start pose; those are EMBODIMENT clauses, checked at
the robot-binding/harness layer, deliberately not here (the scene is robot-agnostic —
the stacking-toy return-to-origin precedent).

Assets are fully procedural, one rigid body each, authored by custom compound spawners
(the stacking-toy stacking-piece pattern — child colliders of one body never self-collide):
  - holder: a bottom disc + 8 box wall segments forming an open octagonal cup
    (inner inradius 44 mm, the xy_tol honesty limit: any pen physically inside the cup
    counts, so the 3.5 cm tolerance is honest by construction, like stacking-toy's peg-enforced
    concentricity; sized so FOUR pens fit on the floor, not just one). Grasp the 8 mm rim
    with any jaw, or palm the ~10 cm body.
  - pen: a cylinder barrel collider (r >= 10 mm — the thin-cylinder pinch audit knob)
    plus a VISUAL-ONLY dark cone tip (the balance_scale needle pattern), so tip vs
    bottom is visible to a skimming viewer and the rubric's tip-up clause is honest.
Two families (2 `pen` + 2 `oil_pen`, the source's object set) differing in color and
diameter only — identity is oracle-visible; there is no hidden state in the floor tier.

Per-episode randomization (task-family knobs): holder pose (xy jitter + yaw), pen
scatter poses (arc slot + xy jitter + free yaw, lying FLAT — every pen must be
reoriented to vertical), AND pen-count subset sampling per family, so success is judged
on the sampled subset and a memorized fixed sequence fails. Absent pens park in an
off-camera ground depot (InteractiveScene cannot despawn).

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg, info, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- custom compound spawners ---------------------------------------------------------------
# One rigid body per object, several child colliders + visual-only decoration, authored with raw
# pxr APIs; only `isaaclab.sim.utils.clone` is borrowed (the regex-resolve + per-env replicate
# machinery every CuboidCfg spawn uses). Same fallback as the stacking piece: author into a /tmp
# USD and return a UsdFileCfg if this ever fights the platform.

_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_pen_holder(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the open cup at `prim_path`: root Xform with RigidBodyAPI + explicit MassAPI
    (overlapping wall segments would double-count density; the symmetric layout keeps the
    auto-CoM on the axis, slightly low because of the bottom disc — good for stability), a
    bottom cylinder collider, and 8 box wall segments forming an octagonal shell whose inner
    aperture is a regular octagon of inradius `inner_r`. Explicit small contact offsets (the
    pc_gpu precedent: the ~2 cm default would produce phantom wall contact on a 3 cm funnel)."""
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
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    # Cap the contact-solver pop (default max depenetration velocity is 3 m/s): an end-on
    # pen impact penetrates a few mm in one 120 Hz step and the solver would otherwise eject
    # it ballistically — the measured bounce-outs of the first two smoke runs. The factory-env
    # insertion trick: resolve overlap gently.
    PhysxSchema.PhysxRigidBodyAPI.Apply(root).CreateMaxDepenetrationVelocityAttr(0.5)

    color = Gf.Vec3f(*cfg.color)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    # bottom disc: floor of the cup, top face at (-h/2 + bot_t) in body frame
    outer_r = cfg.inner_r + cfg.wall_t
    bot = UsdGeom.Cylinder.Define(stage, f"{prim_path}/bottom")
    bot.CreateRadiusAttr(outer_r)
    bot.CreateHeightAttr(cfg.bot_t)
    bot.CreateExtentAttr([Gf.Vec3f(-outer_r, -outer_r, -cfg.bot_t / 2),
                          Gf.Vec3f(outer_r, outer_r, cfg.bot_t / 2)])
    UsdGeom.Xformable(bot.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, -cfg.height / 2 + cfg.bot_t / 2))
    bot.CreateDisplayColorAttr([color])
    collide(bot.GetPrim())

    # 8 wall boxes: mid-plane at inradius inner_r + wall_t/2; segment length closes the OUTER
    # octagon (adjacent segments overlap toward the outside — harmless inside one body); the
    # aperture stays the exact intersection of the 8 inner half-planes (a regular octagon of
    # inradius `inner_r`, circumradius inner_r/cos(pi/8) = 1.082*inner_r).
    n = cfg.n_segments
    r_mid = cfg.inner_r + cfg.wall_t / 2
    seg_len = 2 * (cfg.inner_r + cfg.wall_t) * math.tan(math.pi / n) + 0.002
    for k in range(n):
        ang = 2 * math.pi * k / n
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/wall_{k}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(r_mid * math.cos(ang), r_mid * math.sin(ang), 0.0))
        sxf.AddRotateZOp().Set(math.degrees(ang))
        sxf.AddScaleOp().Set(Gf.Vec3f(cfg.wall_t, seg_len, cfg.height))
        seg.CreateDisplayColorAttr([color])
        collide(seg.GetPrim())
    return root


def _spawn_pen(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one pen at `prim_path`: root Xform with RigidBodyAPI + explicit MassAPI, a
    cylinder barrel collider along local +z, and a VISUAL-ONLY cone tip past the +z end (no
    CollisionAPI — the balance_scale needle pattern; base radius = barrel radius, so a pen
    lying flat shows no penetration). Pen local frame: axis = +z, tip end = +z."""
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
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    # Same depenetration cap as the holder (see there) + a whiff of damping so a 20 g pen
    # rattling in the cup crosses the 0.05 m/s settle gate promptly instead of ringing.
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)

    barrel = UsdGeom.Cylinder.Define(stage, f"{prim_path}/barrel")
    barrel.CreateRadiusAttr(cfg.pen_r)
    barrel.CreateHeightAttr(cfg.barrel_l)
    barrel.CreateExtentAttr([Gf.Vec3f(-cfg.pen_r, -cfg.pen_r, -cfg.barrel_l / 2),
                             Gf.Vec3f(cfg.pen_r, cfg.pen_r, cfg.barrel_l / 2)])
    barrel.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    UsdPhysics.CollisionAPI.Apply(barrel.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(barrel.GetPrim())
    px.CreateContactOffsetAttr(float(cfg.contact_offset))
    px.CreateRestOffsetAttr(0.0)

    tip = UsdGeom.Cone.Define(stage, f"{prim_path}/tip")  # visual only — NO CollisionAPI
    tip.CreateRadiusAttr(cfg.pen_r)
    tip.CreateHeightAttr(cfg.tip_h)
    tip.CreateExtentAttr([Gf.Vec3f(-cfg.pen_r, -cfg.pen_r, -cfg.tip_h / 2),
                          Gf.Vec3f(cfg.pen_r, cfg.pen_r, cfg.tip_h / 2)])
    UsdGeom.Xformable(tip.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, cfg.barrel_l / 2 + cfg.tip_h / 2))
    tip.CreateDisplayColorAttr([Gf.Vec3f(*cfg.tip_color)])
    return root


def _holder_spawner_cfg(*, inner_r: float, wall_t: float, height: float, bot_t: float,
                        mass: float, color: tuple, n_segments: int,
                        contact_offset: float) -> Any:
    """Build (lazily, app required) the holder spawner cfg — `clone` wraps
    `_spawn_pen_holder` exactly like `spawn_cuboid` is wrapped."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "holder" not in _SPAWNER_CACHE:

        @configclass
        class PenHolderSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pen_holder)
            inner_r: float = 0.04  # inner octagon INRADIUS (m)
            wall_t: float = 0.008
            height: float = 0.11
            bot_t: float = 0.008
            color: tuple = (0.25, 0.45, 0.45)
            n_segments: int = 8
            contact_offset: float = 0.002

        _SPAWNER_CACHE["holder"] = PenHolderSpawnerCfg

    return _SPAWNER_CACHE["holder"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        inner_r=inner_r, wall_t=wall_t, height=height, bot_t=bot_t,
        color=color, n_segments=n_segments, contact_offset=contact_offset,
    )


def _pen_spawner_cfg(*, pen_r: float, barrel_l: float, tip_h: float, mass: float,
                     color: tuple, tip_color: tuple, contact_offset: float) -> Any:
    """Build (lazily, app required) the pen spawner cfg."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "pen" not in _SPAWNER_CACHE:

        @configclass
        class PenSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pen)
            pen_r: float = 0.010
            barrel_l: float = 0.135
            tip_h: float = 0.015
            color: tuple = (0.2, 0.35, 0.85)
            tip_color: tuple = (0.08, 0.08, 0.08)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["pen"] = PenSpawnerCfg

    return _SPAWNER_CACHE["pen"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        pen_r=pen_r, barrel_l=barrel_l, tip_h=tip_h,
        color=color, tip_color=tip_color, contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class PenHolderSceneCfg(BaseCfg):
    """Config for `PenHolderScene`. The source thresholds (`xy_tol`, `depth_min`,
    `holder_tilt_max_deg`) are ported verbatim and stay SOFT by design — this is a floor
    task; harden only for curriculum variants."""

    # --- tunable: rubric thresholds (source values, verbatim) --------------------------------
    xy_tol: float = tunable(0.035)  # pen bottom within this of the holder axis (source 3.5 cm).
    # Honest by construction: max physical in-cup offset = inner_r - min pen_r = 3.4 cm
    # < xy_tol, so any pen physically inside counts; a pen leaning OUTSIDE is >= 5 cm away.
    depth_min: float = tunable(0.035)  # pen bottom below the rim by more than this (source 3.5 cm)
    holder_tilt_max_deg: float = tunable(45.0)  # holder axis within this of world-up (source 45)
    pen_align_max_deg: float = tunable(45.0)  # pen axis within this of the HOLDER axis, tip-up
    # (port interpretation of the source's tip-below-root clause, in our tip-up convention;
    # geometry already bounds an in-cup pen's lean — this clause rejects tip-DOWN insertions).
    settle_speed: float = tunable(0.05)  # max |v| (pen AND holder) when judging (m/s)
    placed_tilt_deg: float = tunable(10.0)  # "standing upright" gate for the 100-score set-down
    placed_z_tol: float = tunable(0.010)  # holder bottom within this of the surface (m)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    reset_pos_jitter: float = tunable(0.04)  # uniform +/- xy jitter (holder AND pens) at reset
    reset_yaw_deg: float = tunable(180.0)  # uniform +/- yaw per body at reset (pens lie flat)
    subset_sample: bool = tunable(True)  # per-episode pen-count sampling (demo sets False)
    min_present: int = tunable(1)  # per-family lower bound of sampled pen count

    # --- tunable: placement (robot embodiments raise the work onto a bench) ------------------
    surface_z: float = tunable(0.0)  # work-surface height; 0 = on the ground (null smoke)
    holder_pos: tuple = tunable((0.22, 0.0))  # holder centre on the surface (source: right half)
    pens_center: tuple = tunable((-0.10, 0.0))  # scatter-arc centre (source: pens on left half)
    spawn_radii: tuple = tunable((0.18,))  # scatter arc radii (robot cfgs: front arc)
    spawn_arc: tuple = tunable((90.0, 270.0))  # scatter arc (deg) around pens_center

    # --- info: structure ----------------------------------------------------------------------
    bench_size: tuple = info((1.1, 0.9))  # procedural bench top (x, y), used when surface_z > 0
    # Inner inradius sized for FOUR pens, not one: at 40 mm (run 3/4) the fourth drop had no
    # floor left — it rested on the pile of three, too shallow / past the tilt cone until a
    # shake-down seated it. 44 mm is the honesty limit: inner_r - min pen_r = 34 mm < the
    # 35 mm xy_tol, so the cup still enforces the tolerance by construction.
    holder_inner_r: float = info(0.044)  # inner octagon inradius; funnel = inner_r - pen_r
    holder_wall_t: float = info(0.008)  # rim width — the universal pinch-grasp affordance
    # Height sized against LEAN: a pen with its bottom at the wall and shaft on the opposite
    # rim leans atan((34+44)/108) ~= 36 deg — inside the 45 deg tip-up cone with margin.
    holder_h: float = info(0.120)
    # Floor thickness sized against TUNNELING (GPU PhysX has no CCD): a pen dropped end-on
    # from the mouth hits at ~1.6 m/s = 13 mm/step at 120 Hz — an 8 mm floor was punched
    # through in the first smoke run (pens ejected); 12 mm + the 5 mm contact offset gives
    # ~17 mm of capture per step.
    holder_bot_t: float = info(0.012)
    holder_mass: float = info(0.20)
    holder_color: tuple = info((0.25, 0.45, 0.45))
    n_segments: int = info(8)
    # Contact offset trades phantom contact against fast-contact capture: the 28 mm funnel
    # tolerates a generous 5 mm speculative margin (unlike stacking's 5 mm clearance, which
    # forced 2 mm), and the margin is what catches a 13 mm/step end-on pen impact.
    contact_offset: float = info(0.005)
    tip_h: float = info(0.015)  # visual cone tip past the +z barrel end
    tip_color: tuple = info((0.08, 0.08, 0.08))
    pen_mass: float = info(0.02)
    # (family name, count, barrel radius, barrel length, rgb) — the source's 2 pens + 2 oil
    # pens; radii pass the thin-cylinder pinch audit (scale-up knob lives here).
    families: tuple = info((
        ("pen", 2, 0.010, 0.135, (0.20, 0.35, 0.85)),
        ("oil_pen", 2, 0.012, 0.125, (0.85, 0.25, 0.20)),
    ))
    # Off-camera ground depot for absent pens; grid extent 1.0 + 2*0.14 + pen 0.15 < half of
    # env_spacing 3 (the stacking-toy depot analysis).
    parking_pos: tuple = info((1.0, 1.0))

    # Derived (filled in __post_init__).
    holder_outer_r: float = field(default=None, init=False)
    floor_local_z: float = field(default=None, init=False)  # cup floor, holder body frame
    manifest: tuple = field(default=None, init=False)  # ((name, fam_idx, pen_r, barrel_l), ...)

    def __post_init__(self) -> None:
        self.holder_outer_r = round(self.holder_inner_r + self.holder_wall_t, 4)
        self.floor_local_z = round(-self.holder_h / 2 + self.holder_bot_t, 4)
        flat = []
        for f, (fname, count, pen_r, barrel_l, _rgb) in enumerate(self.families):
            for j in range(count):
                flat.append((f"{fname}_{j}", f, pen_r, barrel_l))
        self.manifest = tuple(flat)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("pen_holder")
class PenHolderScene(BaseScene):
    cfg: PenHolderSceneCfg

    def __init__(self, cfg: PenHolderSceneCfg | None = None) -> None:
        super().__init__(cfg or PenHolderSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, optional bench, the free-standing cup, and the pens lying flat at
        their nominal scatter slots (reset() re-places everything)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.surface_z

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
        if z0 > 0:  # procedural workbench (crate pattern): kinematic slab, top at surface_z
            out["bench"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=sim_utils.CuboidCfg(
                    size=(c.bench_size[0], c.bench_size[1], z0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, z0 / 2)),
            )

        out["holder"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Holder",
            spawn=_holder_spawner_cfg(
                inner_r=c.holder_inner_r, wall_t=c.holder_wall_t, height=c.holder_h,
                bot_t=c.holder_bot_t, mass=c.holder_mass, color=c.holder_color,
                n_segments=c.n_segments, contact_offset=c.contact_offset,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.holder_pos[0], c.holder_pos[1], z0 + c.holder_h / 2 + 0.002)),
        )

        a0, a1 = (math.radians(v) for v in c.spawn_arc)
        n_pens = len(c.manifest)
        cx, cy = c.pens_center
        for i, (name, f, pen_r, barrel_l) in enumerate(c.manifest):
            rgb = c.families[f][4]
            ang = a0 + (a1 - a0) * (i + 0.5) / n_pens
            r = c.spawn_radii[i % len(c.spawn_radii)]
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pen_" + name,
                spawn=_pen_spawner_cfg(
                    pen_r=pen_r, barrel_l=barrel_l, tip_h=c.tip_h, mass=c.pen_mass,
                    color=rgb, tip_color=c.tip_color, contact_offset=0.003,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(cx + r * math.cos(ang), cy + r * math.sin(ang), z0 + pen_r + 0.003),
                    rot=(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0),  # lying flat
                ),
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

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles + allocate the presence mask + per-pen constant tensors. No joints to
        author, no post_step mechanics — the task is pure passive physics."""
        super().bind(env)
        c = self.cfg
        self.holder: RigidObject = env.iscene["holder"]
        self.pens: dict[str, RigidObject] = {
            name: env.iscene[name] for name, _f, _r, _l in c.manifest}
        self.env_origins = env.iscene.env_origins
        # present[e, i]: pen i participates in episode e (sampled at reset; judged subset).
        self.present = torch.ones(env.num_envs, len(c.manifest),
                                  dtype=torch.bool, device=env.device)
        # per-pen constants, manifest order
        self._half_l = torch.tensor([l / 2 for _n, _f, _r, l in c.manifest], device=env.device)
        self._pen_r = torch.tensor([r for _n, _f, r, _l in c.manifest], device=env.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the present pen subset per family, place the holder upright
        with xy jitter + yaw, scatter present pens lying flat on the arc with jitter + free
        yaw, park absent pens in the ground depot."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- subset sampling (the task-family knob): per family, k ~ U{min_present..n} ---
        col0 = 0
        for _fname, count, _r, _l, _rgb in c.families:
            if c.subset_sample:
                k = torch.randint(c.min_present, count + 1, (m,), device=dev)
            else:
                k = torch.full((m,), count, dtype=torch.long, device=dev)
            rank = torch.rand(m, count, device=dev).argsort(dim=1).argsort(dim=1)
            self.present[env_ids.unsqueeze(1), torch.arange(col0, col0 + count, device=dev)] = (
                rank < k.unsqueeze(1))
            col0 += count

        yaw_amp = math.radians(c.reset_yaw_deg)

        # --- holder: upright at holder_pos + jitter, random yaw ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.holder_pos[0]
        st[:, 1] = c.holder_pos[1]
        st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
        st[:, 2] = c.surface_z + c.holder_h / 2 + 0.002
        half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        st[:, 0:3] += origin
        self.holder.write_root_state_to_sim(st, env_ids)

        # --- pens: arc slot + jitter, lying FLAT with free yaw; absent -> parking depot ---
        a0, a1 = (math.radians(v) for v in c.spawn_arc)
        n_pens = len(c.manifest)
        cx, cy = c.pens_center
        c45 = math.cos(math.pi / 4)  # q_pitch = 90 deg about y: pen local +z -> world +x
        for i, (name, _f, pen_r, _l) in enumerate(c.manifest):
            ang = a0 + (a1 - a0) * (i + 0.5) / n_pens
            r = c.spawn_radii[i % len(c.spawn_radii)]
            scat = torch.zeros(m, 3, device=dev)
            scat[:, 0] = cx + r * math.cos(ang)
            scat[:, 1] = cy + r * math.sin(ang)
            scat[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
            scat[:, 2] = c.surface_z + pen_r + 0.003
            park = torch.zeros(m, 3, device=dev)
            park[:, 0] = c.parking_pos[0] + (i % 2) * 0.14
            park[:, 1] = c.parking_pos[1] + (i // 2) * 0.14
            park[:, 2] = pen_r + 0.003

            pres = self.present[env_ids, i].unsqueeze(1)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.where(pres, scat, park)
            # lying flat: q = qz(yaw) * qy(90 deg)  ->  (cy*c45, -sy*c45, cy*c45, sy*c45)
            # with cy=cos(yaw/2), sy=sin(yaw/2)  [qz=(cy,0,0,sy), qy=(c45,0,c45,0) components]
            half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            st[:, 3] = torch.cos(half) * c45
            st[:, 4] = -torch.sin(half) * c45
            st[:, 5] = torch.cos(half) * c45
            st[:, 6] = torch.sin(half) * c45
            self.pens[name].write_root_state_to_sim(st, env_ids)

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "holder": self.holder.data.root_state_w[env_ids].clone(),
            "pens": {n: b.data.root_state_w[env_ids].clone() for n, b in self.pens.items()},
            "present": self.present[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.holder.write_root_state_to_sim(state["holder"], env_ids)
        for n, b in self.pens.items():
            b.write_root_state_to_sim(state["pens"][n], env_ids)
        self.present[env_ids] = state["present"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        where = "on the ground" if c.surface_z <= 0 else "on a workbench"
        fams = " and ".join(
            f"up to {count} {name.replace('_', ' ')}{'s' if count > 1 else ''} "
            f"({2 * r * 1000:.0f} mm thick)"
            for name, count, r, _l, _rgb in c.families)
        return (
            f"An open cylindrical pen holder (a cup, ~{2 * c.holder_outer_r * 1000:.0f} mm wide, "
            f"{c.holder_h * 1000:.0f} mm tall) stands {where}. Scattered on its other side lie "
            f"pens, flat on the surface: {fams}, each with a dark cone tip at one end. Between "
            f"one and all pens of each kind are present in any episode: count what you see.\n"
            f"Goal: put every pen into the holder tip-up (dark tip pointing out of the cup), "
            f"then leave the loaded holder standing upright on the surface. You may hold the "
            f"holder while filling it; a pen dropped tip-down or left leaning outside the cup "
            f"does not count, and a tipped-over holder scores nothing until it is stood up."
        )

    # ----- progress / rubric ----------------------------------------------------------------------
    def _pen_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(pos_w (N,P,3), quat (N,P,4), |lin_vel| (N,P)) for all pens, manifest order."""
        pos = torch.stack([b.data.root_pos_w for b in self.pens.values()], dim=1)
        quat = torch.stack([b.data.root_quat_w for b in self.pens.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.pens.values()], dim=1)
        return pos, quat, vel

    def _pen_ends_local(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Pen bottom / tip points in the HOLDER'S BODY FRAME, shapes (N, P, 3). The rubric
        lives in this frame so a held, moving, tilted holder judges identically to a standing
        one (the source fills a held holder)."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        pos, quat, _v = self._pen_tensors()
        n, p = pos.shape[0], pos.shape[1]
        ez = torch.tensor([0.0, 0.0, 1.0], device=pos.device).expand(n * p, 3)
        axis = quat_apply(quat.reshape(n * p, 4), ez).reshape(n, p, 3)
        bottom = pos - axis * self._half_l[None, :, None]
        tip = pos + axis * (self._half_l + self.cfg.tip_h)[None, :, None]
        hq = self.holder.data.root_quat_w[:, None, :].expand(n, p, 4).reshape(n * p, 4)
        hp = self.holder.data.root_pos_w[:, None, :]
        b_loc = quat_apply_inverse(hq, (bottom - hp).reshape(n * p, 3)).reshape(n, p, 3)
        t_loc = quat_apply_inverse(hq, (tip - hp).reshape(n * p, 3)).reshape(n, p, 3)
        return b_loc, t_loc

    def holder_up(self) -> torch.Tensor:
        """(N,) bool: holder axis within `holder_tilt_max_deg` of world-up (source 45 deg)."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        up = quat_apply(self.holder.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.holder_tilt_max_deg))

    def inserted(self) -> torch.Tensor:
        """(N, P) bool, geometric: pen bottom within `xy_tol` of the holder axis, deeper than
        `depth_min` below the rim (and above the cup floor), axis tip-up along the holder axis
        — all in the holder frame — with the holder itself `holder_up`."""
        c = self.cfg
        b_loc, t_loc = self._pen_ends_local()
        near_axis = b_loc[:, :, :2].norm(dim=-1) < c.xy_tol
        depth = c.holder_h / 2 - b_loc[:, :, 2]
        deep = (depth > c.depth_min) & (b_loc[:, :, 2] > c.floor_local_z - 0.005)
        axis_loc = t_loc - b_loc
        axis_loc = axis_loc / axis_loc.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        tip_up = axis_loc[:, :, 2] >= math.cos(math.radians(c.pen_align_max_deg))
        return near_axis & deep & tip_up & self.holder_up().unsqueeze(-1)

    def settled(self) -> torch.Tensor:
        """(N, P) bool: pen AND holder |lin vel| below `settle_speed`."""
        _p, _q, vel = self._pen_tensors()
        holder_still = self.holder.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed
        return (vel < self.cfg.settle_speed) & holder_still.unsqueeze(-1)

    def counted(self) -> torch.Tensor:
        """(N, P) bool: inserted, settled AND present — what the rubric counts."""
        return self.inserted() & self.settled() & self.present

    def all_inserted(self) -> torch.Tensor:
        """(N,) bool: every PRESENT pen counts — judged on the sampled subset (90-state)."""
        return (self.counted() | ~self.present).all(dim=1)

    def holder_placed(self) -> torch.Tensor:
        """(N,) bool: the loaded holder set down standing upright ON the work surface —
        within `placed_tilt_deg` of vertical, bottom within `placed_z_tol` of the surface,
        settled. The scene-level port of the source's set-down clause."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        up = quat_apply(self.holder.data.root_quat_w, ez)
        upright = up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(c.placed_tilt_deg))
        bottom_z = (self.holder.data.root_pos_w - self.env_origins)[:, 2] - up[:, 2] * c.holder_h / 2
        on_surface = (bottom_z - c.surface_z).abs() < c.placed_z_tol
        still = self.holder.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return upright & on_surface & still

    def score(self) -> torch.Tensor:
        """(N,) int: the ported transition rubric — [0, 10, 25, 40] for 0/1/2/3 pens counted,
        90 for ALL present pens in the (possibly held) holder, 100 for all-in + the holder
        standing upright on the surface. NOTE the source's 100 additionally requires both
        grippers >= 80% open and both end-effectors back at their start pose; those are
        EMBODIMENT clauses, checked at the robot-binding/harness layer, deliberately not here
        (the scene is robot-agnostic — the stacking-toy return-to-origin precedent)."""
        table = torch.tensor([0, 10, 25, 40], device=self.env.device)
        k = self.counted().sum(dim=1)
        base = table[k.clamp(max=3)]
        all_in = self.all_inserted()
        full = torch.where(self.holder_placed(), 100, 90)
        return torch.where(all_in, full, base)

    def success(self) -> torch.Tensor:
        """(N,) bool: all present pens inserted tip-up + holder standing upright on the
        surface (scene-level success; the NullRobot oracle's target)."""
        return self.all_inserted() & self.holder_placed()

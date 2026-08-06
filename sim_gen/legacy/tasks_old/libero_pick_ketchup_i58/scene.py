"""HeightLineupScene — stand the bottles on the display shelf in a HEIGHT-SORTED lineup
(sim_gen task `libero_pick_ketchup_i58`, derived from libero/libero_pick_ketchup).

The seed is a single-target prehensile CONTAINMENT task: identify the one named object
(the ketchup) among six grocery items, grasp it, carry it, drop it inside a basket; its
checker is literally a relative-bbox containment test against the basket, and the five
distractors never matter.

This task keeps the LIBERO grocery-tabletop flavor (a clutter of bottle-like items and a
basket in the scene) but replaces "identify one target + contain it" with a RELATIONAL
ARRANGEMENT goal that has no single target, no containment, and no fixed slots:

  - Five colored bottles of visibly different heights (95..215 mm) are scattered — some
    fallen on their side, some standing — with 3..5 of them PRESENT per episode
    (subset-sampled). EVERY present bottle is a goal object.
  - A free-standing display shelf (kinematic, re-posed every reset: xy jitter + free
    yaw) carries a green DATUM POST at one end. The goal: every present bottle stands
    upright on the shelf top in a single lineup ordered by height — shortest nearest
    the post, height strictly increasing along the shelf away from it. There are no
    marked slots and no physical keying: the order is a pure pairwise x-relation in
    the SHELF's body frame, so the correct arrangement changes with the sampled subset
    and the shelf pose. The solver must COMPARE the bottles and plan an assignment.
  - The seed's basket is present as SEED-BAIT: an open kinematic basket that scores
    nothing. Executing the seed's plan (drop bottles into the basket) is a tested
    negative control at ~0.

Judged on PHYSICAL outcomes only, in the shelf body frame: a bottle counts when its
bottom-face center is on the shelf top (inside the top's margins, within `placed_z_tol`
of the surface), its axis is upright (cap up, within `upright_max_deg` of world-up),
and it is settled; an adjacent height-pair counts when both bottles count and the
taller one's x exceeds the shorter one's by `pair_gap_min`. Lying a bottle at its slot,
standing it on another bottle, or standing it in the basket all fail by construction
(upright / z / on-shelf gates).

Rubric (graded, [0, 1], 1.0 iff success):
  0.10 * (latched `ever_placed` fraction)   — transient achievement latch: a bottle
        that once stood counted on the shelf keeps this slice even if later knocked off
  0.40 * (fraction of present bottles counted NOW)
  0.35 * (fraction of adjacent height-pairs in correct x-order NOW)
  1.0  iff success(): all present bottles counted AND every adjacent pair ordered.

Assets are fully procedural, compound spawners (pen-holder pattern): bottle = cylinder
barrel collider + visual-only cap; shelf = kinematic slab + back wall + green datum
post; basket = kinematic open box. Heavy imports (isaaclab, pxr) are deferred so
importing this module stays app-free.
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


# ----- compound spawners -----------------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _add_root(stage, prim_path: str, translation, orientation):
    """Author the root Xform with optional translate/orient ops; return (root, xformable)."""
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xform.Define(stage, prim_path)
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    return xform.GetPrim()


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, size, pos, color, contact_offset: float | None):
    from pxr import Gf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        _collide(cube.GetPrim(), contact_offset)
    return cube


def _spawn_bottle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One dynamic bottle: cylinder barrel collider along local +z plus a VISUAL-ONLY
    narrower cap cylinder past the +z end (the pen cone-tip pattern), so up vs down is
    visible and the rubric's cap-up clause is honest."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = _add_root(stage, prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)

    barrel = UsdGeom.Cylinder.Define(stage, f"{prim_path}/barrel")
    barrel.CreateRadiusAttr(cfg.bottle_r)
    barrel.CreateHeightAttr(cfg.bottle_h)
    barrel.CreateExtentAttr([Gf.Vec3f(-cfg.bottle_r, -cfg.bottle_r, -cfg.bottle_h / 2),
                             Gf.Vec3f(cfg.bottle_r, cfg.bottle_r, cfg.bottle_h / 2)])
    barrel.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    _collide(barrel.GetPrim(), cfg.contact_offset)

    cap_r = cfg.bottle_r * 0.62
    cap = UsdGeom.Cylinder.Define(stage, f"{prim_path}/cap")  # visual only — NO CollisionAPI
    cap.CreateRadiusAttr(cap_r)
    cap.CreateHeightAttr(cfg.cap_h)
    cap.CreateExtentAttr([Gf.Vec3f(-cap_r, -cap_r, -cfg.cap_h / 2),
                          Gf.Vec3f(cap_r, cap_r, cfg.cap_h / 2)])
    UsdGeom.Xformable(cap.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, cfg.bottle_h / 2 + cfg.cap_h / 2))
    cap.CreateDisplayColorAttr([Gf.Vec3f(*cfg.cap_color)])
    return root


def _spawn_shelf(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The KINEMATIC display shelf. Root at the slab center: slab top at local
    z = +shelf_h/2, ground at local z = -shelf_h/2. Back wall along local +y edge,
    green datum post standing on the ground just past the local -x end (the SHORT end
    of the target lineup)."""
    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = _add_root(stage, prim_path, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)

    _box(stage, f"{prim_path}/slab", (cfg.shelf_len, cfg.shelf_dep, cfg.shelf_h),
         (0.0, 0.0, 0.0), cfg.slab_color, cfg.contact_offset)
    _box(stage, f"{prim_path}/backwall",
         (cfg.shelf_len, cfg.shelf_wall_t, cfg.shelf_wall_h),
         (0.0, cfg.shelf_dep / 2 - cfg.shelf_wall_t / 2,
          cfg.shelf_h / 2 + cfg.shelf_wall_h / 2), cfg.wall_color, cfg.contact_offset)
    _box(stage, f"{prim_path}/post", (0.028, 0.028, cfg.post_h),
         (-(cfg.shelf_len / 2 + cfg.post_gap), 0.0, -cfg.shelf_h / 2 + cfg.post_h / 2),
         cfg.post_color, cfg.contact_offset)
    return root


def _spawn_basket(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The KINEMATIC seed-bait basket: open box, root at the floor-slab center."""
    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = _add_root(stage, prim_path, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)

    o, t, h, ft = cfg.outer, cfg.wall_t, cfg.wall_h, cfg.floor_t
    _box(stage, f"{prim_path}/floor", (o, o, ft), (0.0, 0.0, 0.0),
         cfg.color, cfg.contact_offset)
    for k, (dx, dy, sx, sy) in enumerate((
            (o / 2 - t / 2, 0.0, t, o), (-(o / 2 - t / 2), 0.0, t, o),
            (0.0, o / 2 - t / 2, o, t), (0.0, -(o / 2 - t / 2), o, t))):
        _box(stage, f"{prim_path}/wall_{k}", (sx, sy, h),
             (dx, dy, ft / 2 + h / 2), cfg.color, cfg.contact_offset)
    return root


def _bottle_spawner_cfg(*, bottle_r, bottle_h, cap_h, mass, color, cap_color,
                        contact_offset) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "bottle" not in _SPAWNER_CACHE:

        @configclass
        class BottleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bottle)
            bottle_r: float = 0.022
            bottle_h: float = 0.15
            cap_h: float = 0.018
            color: tuple = (0.5, 0.5, 0.5)
            cap_color: tuple = (0.12, 0.12, 0.12)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["bottle"] = BottleSpawnerCfg

    return _SPAWNER_CACHE["bottle"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        bottle_r=bottle_r, bottle_h=bottle_h, cap_h=cap_h, color=color,
        cap_color=cap_color, contact_offset=contact_offset,
    )


def _shelf_spawner_cfg(*, shelf_len, shelf_dep, shelf_h, shelf_wall_t, shelf_wall_h,
                       post_h, post_gap, slab_color, wall_color, post_color,
                       contact_offset) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "shelf" not in _SPAWNER_CACHE:

        @configclass
        class ShelfSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_shelf)
            shelf_len: float = 0.60
            shelf_dep: float = 0.16
            shelf_h: float = 0.06
            shelf_wall_t: float = 0.012
            shelf_wall_h: float = 0.10
            post_h: float = 0.16
            post_gap: float = 0.035
            slab_color: tuple = (0.45, 0.42, 0.38)
            wall_color: tuple = (0.35, 0.32, 0.30)
            post_color: tuple = (0.10, 0.75, 0.25)
            contact_offset: float = 0.003

        _SPAWNER_CACHE["shelf"] = ShelfSpawnerCfg

    return _SPAWNER_CACHE["shelf"](
        mass_props=sim_utils.MassPropertiesCfg(mass=6.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        shelf_len=shelf_len, shelf_dep=shelf_dep, shelf_h=shelf_h,
        shelf_wall_t=shelf_wall_t, shelf_wall_h=shelf_wall_h, post_h=post_h,
        post_gap=post_gap, slab_color=slab_color, wall_color=wall_color,
        post_color=post_color, contact_offset=contact_offset,
    )


def _basket_spawner_cfg(*, outer, wall_t, wall_h, floor_t, color, contact_offset) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "basket" not in _SPAWNER_CACHE:

        @configclass
        class BasketSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_basket)
            outer: float = 0.22
            wall_t: float = 0.012
            wall_h: float = 0.12
            floor_t: float = 0.015
            color: tuple = (0.55, 0.38, 0.20)
            contact_offset: float = 0.003

        _SPAWNER_CACHE["basket"] = BasketSpawnerCfg

    return _SPAWNER_CACHE["basket"](
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        outer=outer, wall_t=wall_t, wall_h=wall_h, floor_t=floor_t, color=color,
        contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class HeightLineupSceneCfg(BaseCfg):
    """Config for `HeightLineupScene`. Shelf local frame: +x = lineup direction (post at
    the -x end marks the SHORT end), slab top at local z = +shelf_h/2."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    upright_max_deg: float = tunable(10.0)  # bottle axis within this of world-up (cap up).
    # The tallest bottle's static tip angle is atan(r / (h/2)) = atan(22/107.5) ~= 11.6 deg:
    # any bottle leaning MORE than the gate would topple on its own anyway — honest gate.
    placed_z_tol: float = tunable(0.012)  # bottom-face center within this of the shelf top.
    # Min bottle height 95 mm >> this: standing on ANOTHER bottle (or on the basket floor,
    # 15 mm) never passes — no stacking / no basket credit by construction.
    edge_margin: float = tunable(0.020)  # bottom center inside shelf top x-extent by this
    y_margin: float = tunable(0.060)  # |y_local| of the bottom center on the 160 mm-deep top
    pair_gap_min: float = tunable(0.020)  # taller.x - shorter.x must exceed this (m); two
    # touching bottles are 2r = 44 mm apart in x when actually side by side in the lineup
    settle_lin: float = tunable(0.05)  # max |lin vel| when judging settled (m/s)
    settle_ang: float = tunable(0.50)  # max |ang vel| when judging settled (rad/s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    subset_sample: bool = tunable(True)  # per-episode bottle-count sampling
    min_present: int = tunable(3)  # lower bound of the sampled bottle count
    shelf_pos: tuple = tunable((0.40, 0.0))  # shelf center nominal (xy)
    shelf_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the shelf at reset
    shelf_yaw_deg: float = tunable(180.0)  # uniform +/- shelf yaw at reset (free heading)
    scatter_center: tuple = tunable((-0.30, 0.0))  # bottle scatter-arc center
    scatter_radius: float = tunable(0.16)  # scatter arc radius
    scatter_arc: tuple = tunable((80.0, 280.0))  # scatter arc (deg): faces AWAY from shelf
    scatter_jitter: float = tunable(0.04)  # uniform +/- xy jitter per bottle
    bottle_yaw_deg: float = tunable(180.0)  # uniform +/- yaw per bottle
    stand_prob: float = tunable(0.5)  # per bottle: spawn standing (else fallen on its side)
    basket_pos: tuple = tunable((-0.05, 0.48))  # basket nominal (x, |y|); y-side sampled
    basket_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the basket

    # --- info: structure ---------------------------------------------------------------------
    shelf_len: float = info(0.60)
    shelf_dep: float = info(0.16)
    shelf_h: float = info(0.06)  # slab height; top face at this above the ground
    shelf_wall_t: float = info(0.012)
    shelf_wall_h: float = info(0.10)
    post_h: float = info(0.16)
    post_gap: float = info(0.035)  # post center past the slab -x end
    bottle_r: float = info(0.022)  # all bottles share one radius (only HEIGHT differs)
    cap_h: float = info(0.018)  # visual-only cap past the +z barrel end
    bottle_mass: float = info(0.09)
    basket_outer: float = info(0.22)
    basket_wall_t: float = info(0.012)
    basket_wall_h: float = info(0.12)
    basket_floor_t: float = info(0.015)
    slab_color: tuple = info((0.45, 0.42, 0.38))
    wall_color: tuple = info((0.35, 0.32, 0.30))
    post_color: tuple = info((0.10, 0.75, 0.25))
    basket_color: tuple = info((0.55, 0.38, 0.20))
    cap_color: tuple = info((0.12, 0.12, 0.12))
    contact_offset: float = info(0.002)
    # (name, barrel height, rgb) in ASCENDING height order — the manifest order IS the
    # target lineup order. Heights step 30 mm: unambiguous to a viewer.
    families: tuple = info((
        ("blue", 0.095, (0.25, 0.45, 0.90)),
        ("green", 0.125, (0.20, 0.68, 0.30)),
        ("yellow", 0.155, (0.90, 0.80, 0.20)),
        ("orange", 0.185, (0.95, 0.55, 0.15)),
        ("red", 0.215, (0.85, 0.20, 0.15)),
    ))
    parking_pos: tuple = info((1.3, 1.3))  # off-camera ground depot for absent bottles

    # Derived (filled in __post_init__).
    manifest: tuple = field(default=None, init=False)  # ((name, height), ...) ascending

    def __post_init__(self) -> None:
        self.manifest = tuple((f"bottle_{name}", h) for name, h, _rgb in self.families)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("height_lineup")
class HeightLineupScene(BaseScene):
    cfg: HeightLineupSceneCfg

    def __init__(self, cfg: HeightLineupSceneCfg | None = None) -> None:
        super().__init__(cfg or HeightLineupSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic shelf + basket (re-posed by reset), and the five
        bottles at nominal scatter slots (reset re-places everything)."""
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
            "shelf": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shelf",
                spawn=_shelf_spawner_cfg(
                    shelf_len=c.shelf_len, shelf_dep=c.shelf_dep, shelf_h=c.shelf_h,
                    shelf_wall_t=c.shelf_wall_t, shelf_wall_h=c.shelf_wall_h,
                    post_h=c.post_h, post_gap=c.post_gap, slab_color=c.slab_color,
                    wall_color=c.wall_color, post_color=c.post_color,
                    contact_offset=0.003,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.shelf_pos[0], c.shelf_pos[1], c.shelf_h / 2)),
            ),
            "basket": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Basket",
                spawn=_basket_spawner_cfg(
                    outer=c.basket_outer, wall_t=c.basket_wall_t, wall_h=c.basket_wall_h,
                    floor_t=c.basket_floor_t, color=c.basket_color, contact_offset=0.003,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.basket_pos[0], c.basket_pos[1], c.basket_floor_t / 2)),
            ),
        }
        a0, a1 = (math.radians(v) for v in c.scatter_arc)
        nb = len(c.manifest)
        cx, cy = c.scatter_center
        for i, ((name, _h), (_nm, _hh, rgb)) in enumerate(zip(c.manifest, c.families)):
            ang = a0 + (a1 - a0) * (i + 0.5) / nb
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bottle_" + name,
                spawn=_bottle_spawner_cfg(
                    bottle_r=c.bottle_r, bottle_h=_hh, cap_h=c.cap_h,
                    mass=c.bottle_mass, color=rgb, cap_color=c.cap_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(cx + c.scatter_radius * math.cos(ang),
                         cy + c.scatter_radius * math.sin(ang), c.bottle_r + 0.003),
                    rot=(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)),
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
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles + allocate the presence mask, latches and per-bottle constants."""
        super().bind(env)
        c = self.cfg
        n, dev = env.num_envs, env.device
        self.shelf: RigidObject = env.iscene["shelf"]
        self.basket: RigidObject = env.iscene["basket"]
        self.bottles: dict[str, RigidObject] = {
            name: env.iscene[name] for name, _h in c.manifest}
        self.env_origins = env.iscene.env_origins
        nb = len(c.manifest)
        self.present = torch.ones(n, nb, dtype=torch.bool, device=dev)
        self.ever_placed = torch.zeros(n, nb, dtype=torch.bool, device=dev)
        self._half_h = torch.tensor([h / 2 for _nm, h in c.manifest], device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the present subset (3..5), re-pose the kinematic shelf
        (xy jitter + free yaw) and basket (side + jitter + yaw), scatter present bottles
        on the arc — each standing OR fallen on its side (sampled) — park absent bottles
        in the ground depot, clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        nb = len(c.manifest)

        # --- subset sampling: k ~ U{min_present..nb} present bottles per episode ---
        if c.subset_sample:
            k = torch.randint(c.min_present, nb + 1, (m,), device=dev)
        else:
            k = torch.full((m,), nb, dtype=torch.long, device=dev)
        rank = torch.rand(m, nb, device=dev).argsort(dim=1).argsort(dim=1)
        self.present[env_ids] = rank < k.unsqueeze(1)

        # --- shelf: nominal pos + jitter, free yaw (kinematic re-pose) ---
        st = torch.zeros(m, 7, device=dev)
        st[:, 0] = c.shelf_pos[0]
        st[:, 1] = c.shelf_pos[1]
        st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.shelf_jitter
        st[:, 2] = c.shelf_h / 2
        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.shelf_yaw_deg) / 2
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        st[:, 0:3] += origin
        self.shelf.write_root_pose_to_sim(st, env_ids)

        # --- basket: sampled side, jitter, free yaw ---
        st = torch.zeros(m, 7, device=dev)
        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           torch.ones(m, device=dev), -torch.ones(m, device=dev))
        st[:, 0] = c.basket_pos[0]
        st[:, 1] = side * c.basket_pos[1]
        st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.basket_jitter
        st[:, 2] = c.basket_floor_t / 2
        half = (torch.rand(m, device=dev) * 2 - 1) * math.pi / 2
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        st[:, 0:3] += origin
        self.basket.write_root_pose_to_sim(st, env_ids)

        # --- bottles: arc slot + jitter; standing OR lying (sampled); absent -> depot ---
        a0, a1 = (math.radians(v) for v in c.scatter_arc)
        cx, cy = c.scatter_center
        yaw_amp = math.radians(c.bottle_yaw_deg)
        c45 = math.cos(math.pi / 4)
        for i, (name, h) in enumerate(c.manifest):
            ang = a0 + (a1 - a0) * (i + 0.5) / nb
            scat = torch.zeros(m, 3, device=dev)
            scat[:, 0] = cx + c.scatter_radius * math.cos(ang)
            scat[:, 1] = cy + c.scatter_radius * math.sin(ang)
            scat[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.scatter_jitter
            park = torch.zeros(m, 3, device=dev)
            park[:, 0] = c.parking_pos[0] + (i % 2) * 0.15
            park[:, 1] = c.parking_pos[1] + (i // 2) * 0.15

            stand = torch.rand(m, device=dev) < c.stand_prob
            half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            cy_, sy_ = torch.cos(half), torch.sin(half)
            # standing: q = qz(yaw); lying flat: q = qz(yaw) * qy(90 deg)
            q_stand = torch.stack([cy_, torch.zeros_like(cy_), torch.zeros_like(cy_), sy_],
                                  dim=1)
            q_lie = torch.stack([cy_ * c45, -sy_ * c45, cy_ * c45, sy_ * c45], dim=1)

            pres = self.present[env_ids, i]
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = torch.where(pres.unsqueeze(1), scat[:, 0:2], park[:, 0:2])
            z_scat = torch.where(stand, torch.full((m,), h / 2 + 0.003, device=dev),
                                 torch.full((m,), c.bottle_r + 0.003, device=dev))
            st[:, 2] = torch.where(pres, z_scat,
                                   torch.full((m,), h / 2 + 0.003, device=dev))
            q_scat = torch.where(stand.unsqueeze(1), q_stand, q_lie)
            st[:, 3:7] = torch.where(pres.unsqueeze(1), q_scat, q_stand)
            st[:, 0:3] += origin
            self.bottles[name].write_root_state_to_sim(st, env_ids)

        self.ever_placed[env_ids] = False

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Every physics substep: latch `ever_placed` per bottle (stood counted on the
        shelf at some point — the transient-achievement slice of the rubric)."""
        self.ever_placed |= self.counted()

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "shelf": self.shelf.data.root_state_w[env_ids].clone(),
            "basket": self.basket.data.root_state_w[env_ids].clone(),
            "bottles": {n: b.data.root_state_w[env_ids].clone()
                        for n, b in self.bottles.items()},
            "present": self.present[env_ids].clone(),
            "ever_placed": self.ever_placed[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.shelf.write_root_pose_to_sim(state["shelf"][:, 0:7], env_ids)
        self.basket.write_root_pose_to_sim(state["basket"][:, 0:7], env_ids)
        for n, b in self.bottles.items():
            b.write_root_state_to_sim(state["bottles"][n], env_ids)
        self.present[env_ids] = state["present"]
        self.ever_placed[env_ids] = state["ever_placed"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        fams = ", ".join(f"{name} ({h * 1000:.0f} mm)" for name, h, _rgb in c.families)
        return (
            f"A display shelf (a {c.shelf_len * 1000:.0f} mm-long raised slab with a low "
            f"back wall) stands free on the floor at a random heading; a GREEN POST rises "
            f"at one end of it. Scattered on the other side of the workspace lie colored "
            f"bottles of clearly different heights — {fams} — some standing, some fallen "
            f"over; between 3 and all 5 are present in any episode (count what you see). "
            f"An open basket also sits nearby: it is NOT a goal.\n"
            f"Goal: stand every present bottle upright (cap up) on the shelf top in a "
            f"single line-up SORTED BY HEIGHT — the shortest bottle nearest the green "
            f"post, each taller bottle placed further from the post along the shelf. "
            f"Bottles left on the floor, laid flat on the shelf, stacked on each other, "
            f"or dropped in the basket do not count; the lineup is judged on the settled "
            f"final poses in the shelf's own frame."
        )

    # ----- progress / rubric ------------------------------------------------------------------
    def _bottle_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(pos_w (N,B,3), quat (N,B,4), |lin_vel| (N,B), |ang_vel| (N,B))."""
        pos = torch.stack([b.data.root_pos_w for b in self.bottles.values()], dim=1)
        quat = torch.stack([b.data.root_quat_w for b in self.bottles.values()], dim=1)
        lin = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.bottles.values()], dim=1)
        ang = torch.stack([b.data.root_ang_vel_w.norm(dim=-1)
                           for b in self.bottles.values()], dim=1)
        return pos, quat, lin, ang

    def _axes_world(self) -> torch.Tensor:
        """(N,B,3) bottle body +z axis in world frame."""
        from isaaclab.utils.math import quat_apply

        _p, quat, _l, _a = self._bottle_tensors()
        n, b = quat.shape[0], quat.shape[1]
        ez = torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(n * b, 3)
        return quat_apply(quat.reshape(n * b, 4), ez).reshape(n, b, 3)

    def bottoms_shelf_local(self) -> torch.Tensor:
        """(N,B,3) bottle bottom-face centers in the SHELF body frame (the judging frame:
        the lineup direction is shelf-local +x, the top face is local z = shelf_h/2)."""
        from isaaclab.utils.math import quat_apply_inverse

        pos, _q, _l, _a = self._bottle_tensors()
        axis = self._axes_world()
        bottom = pos - axis * self._half_h[None, :, None]
        n, b = pos.shape[0], pos.shape[1]
        sq = self.shelf.data.root_quat_w[:, None, :].expand(n, b, 4).reshape(n * b, 4)
        sp = self.shelf.data.root_pos_w[:, None, :]
        return quat_apply_inverse(sq, (bottom - sp).reshape(n * b, 3)).reshape(n, b, 3)

    def placed_geom(self) -> torch.Tensor:
        """(N,B) bool, geometric: bottom center on the shelf top (x/y margins + z band)
        with the bottle upright, cap up."""
        c = self.cfg
        loc = self.bottoms_shelf_local()
        dz = loc[:, :, 2] - c.shelf_h / 2
        on_top = ((loc[:, :, 0].abs() <= c.shelf_len / 2 - c.edge_margin)
                  & (loc[:, :, 1].abs() <= c.y_margin)
                  & (dz > -0.006) & (dz < c.placed_z_tol))
        upright = self._axes_world()[:, :, 2] >= math.cos(math.radians(c.upright_max_deg))
        return on_top & upright

    def settled(self) -> torch.Tensor:
        """(N,B) bool: bottle lin AND ang velocity below the settle gates."""
        _p, _q, lin, ang = self._bottle_tensors()
        return (lin < self.cfg.settle_lin) & (ang < self.cfg.settle_ang)

    def counted(self) -> torch.Tensor:
        """(N,B) bool: standing on the shelf, settled AND present — what the rubric counts."""
        return self.placed_geom() & self.settled() & self.present

    def sorted_pairs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """((N,) correct pairs, (N,) total pairs): over PRESENT bottles in ascending height
        order (= manifest order), each consecutive pair counts when both bottles are
        counted and the taller one sits at least `pair_gap_min` further along shelf +x."""
        c = self.cfg
        cnt = self.counted()
        x = self.bottoms_shelf_local()[:, :, 0]
        n, b = cnt.shape
        ok = torch.zeros(n, device=cnt.device)
        tot = torch.zeros(n, device=cnt.device)
        for i in range(b):
            for j in range(i + 1, b):
                between = (self.present[:, i + 1:j].any(dim=1) if j > i + 1
                           else torch.zeros(n, dtype=torch.bool, device=cnt.device))
                adjacent = self.present[:, i] & self.present[:, j] & ~between
                good = adjacent & cnt[:, i] & cnt[:, j] & (x[:, j] - x[:, i] >= c.pair_gap_min)
                tot += adjacent.float()
                ok += good.float()
        return ok, tot

    def success(self) -> torch.Tensor:
        """(N,) bool: every present bottle stands counted on the shelf AND every adjacent
        height-pair is in correct x-order (scene-level success; the oracle's target)."""
        all_in = (self.counted() | ~self.present).all(dim=1)
        ok, tot = self.sorted_pairs()
        return all_in & (ok >= tot) & (tot >= 1.0)

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10 * latched ever-placed fraction + 0.40 * currently-
        counted fraction + 0.35 * correctly-ordered adjacent-pair fraction; exactly 1.0
        iff success() now; ~0 for doing nothing (and for the seed's basket dump)."""
        k = self.present.float().sum(dim=1).clamp(min=1.0)
        lat = 0.10 * (self.ever_placed & self.present).float().sum(dim=1) / k
        now = 0.40 * self.counted().float().sum(dim=1) / k
        ok, tot = self.sorted_pairs()
        order = 0.35 * ok / tot.clamp(min=1.0)
        s = lat + now + order
        return torch.where(self.success(), torch.ones_like(s), s)


# ----- env registration ------------------------------------------------------------------------
register_env("simgen", lambda: EnvCfg(scene="height_lineup", robot="null", env_spacing=4.0))

"""PantryRackOrderScene — slide three cans into a one-ended roofed rack so their
depth order matches the episode's randomized target permutation.

Derived from libero_90/living_room_scene1 "pick up the alphabet soup and put it in
the basket" (grasp ONE target can among distractors, carry it over a passive open
basket, release, bbox containment check). Here there is no distractor and no drop-in
container at all — ALL THREE cans are targets and the container is a PANTRY RACK: a
low channel with a floor, two walls, a ROOF and a closed back end, open only at its
front MOUTH (with a flared loading apron). What is judged is not "is the can in the
container" but the ARRANGEMENT: from the closed end outward, the identity order of
the three cans must equal a per-episode random permutation.

Why the arrangement is a genuine sequencing puzzle, enforced by geometry:
  - the roof means cans can only ENTER through the mouth, pushed along the floor —
    nothing can be lowered in from above;
  - the channel is too narrow for any two cans to pass each other inside (for every
    pair, the maximum lateral centre separation the walls allow is far below the sum
    of the radii), so cans inside keep their relative depth order forever;
  - hence the final arrangement IS the insertion history: the solver must derive the
    insertion sequence from the goal (deepest-first) and execute it. A wrong first
    insertion cannot be repaired by reordering inside.

The target permutation is grounded VISUALLY: three colored order tiles sit on the
rack roof, running from the closed end to the mouth; the tile nearest the closed end
names (by color) the can that must end up deepest, and so on outward. The tiles are
kinematic bodies re-posed per episode from the sampled permutation.

Cans differ in COLOR and DIAMETER (same height): RED tomato 40 mm, GREEN peas 50 mm,
BLUE soup 60 mm. All procedural (compound spawners, raw pxr authoring).

Rubric (0..1, latched stages anchored in the demonstrated push-insertion solution):
  0.30  stage 1 (latched, 8-step still streak): the can required deepest is fully
        inside and strictly deeper than every other can inside
  0.30  stage 2 (latched): additionally the second-required can is fully inside in
        rank 2 (third can absent or shallower)
  cap 0.60 without success; 1.0 iff success() live: all three cans fully inside,
  upright, in the exact target depth order (strict 30 mm rank margins — contiguous
  cans are >= 45 mm apart by geometry), everything still.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
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


def _add_box(stage, path: str, *, center, size, color, collide=None, yaw_deg: float = 0.0) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw_deg:
        half = math.radians(yaw_deg) / 2
        xf.AddOrientOp().Set(Gf.Quatf(math.cos(half), Gf.Vec3f(0.0, 0.0, math.sin(half))))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide is not None:
        collide(box.GetPrim())


def _add_cyl(stage, path: str, *, center, radius, height, color, collide=None) -> None:
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr("Z")
    r, h = float(radius), float(height)
    cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
    from pxr import UsdGeom as _ug

    _ug.Xformable(cyl.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide is not None:
        collide(cyl.GetPrim())


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


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


def _root_xform(prim_path: str, translation, orientation):
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


def _dyn_props(root, *, lin_damp: float, ang_damp: float) -> None:
    """Dynamic-body physics armor (custom spawners apply NO cfg schemas)."""
    from pxr import PhysxSchema

    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(lin_damp))
    pxrb.CreateAngularDampingAttr(float(ang_damp))
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)


def _spawn_rack(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the pantry rack: KINEMATIC compound. Local frame: origin at the MOUTH
    centre at ground level, +x pointing INWARD (depth), +z up. Floor slab (with a
    loading apron in front of the mouth, both slick), two side walls, closed back
    wall, a ROOF over the whole channel, and two flared guide walls at the mouth."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    mat = _phys_material(stage, f"{prim_path}/slickMat", c.floor_mu, c.floor_mu * 0.9)

    out_half = c.interior_w / 2 + c.wall_t
    x_hi = c.interior_len + c.wall_t  # outer face of the back wall
    # floor slab: apron (in front of the mouth) + channel floor, one slick slab.
    # Kept NARROW (a strip under the channel + on-axis approach) so the jittered
    # side staging slots on the ground can never spawn flush against its edge.
    fl_len = x_hi + c.apron_len
    _add_box(stage, f"{prim_path}/floor",
             center=((x_hi - c.apron_len) / 2, 0.0, c.floor_t / 2),
             size=(fl_len, 2 * out_half + 0.06, c.floor_t),
             color=c.floor_color, collide=collide)
    _bind_material(stage.GetPrimAtPath(f"{prim_path}/floor"), mat)

    wall_cz = c.floor_t + c.wall_h / 2
    for sgn, nm in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/wall_{nm}",
                 center=(x_hi / 2, sgn * (c.interior_w / 2 + c.wall_t / 2), wall_cz),
                 size=(x_hi, c.wall_t, c.wall_h), color=c.wall_color, collide=collide)
    _add_box(stage, f"{prim_path}/back",
             center=(c.interior_len + c.wall_t / 2, 0.0, wall_cz),
             size=(c.wall_t, c.interior_w, c.wall_h), color=c.wall_color,
             collide=collide)
    _add_box(stage, f"{prim_path}/roof",
             center=(x_hi / 2, 0.0, c.floor_t + c.wall_h + c.roof_t / 2),
             size=(x_hi, 2 * out_half, c.roof_t), color=c.roof_color, collide=collide)
    # flared mouth guides (funnel pushes into the mouth)
    fd = math.radians(c.flare_deg)
    for sgn, nm in ((1.0, "p"), (-1.0, "n")):
        dirx, diry = -math.cos(fd), sgn * math.sin(fd)
        cx = dirx * c.flare_len / 2
        cy = sgn * (c.interior_w / 2 + c.wall_t / 2) + diry * c.flare_len / 2
        _add_box(stage, f"{prim_path}/flare_{nm}",
                 center=(cx, cy, wall_cz),
                 size=(c.flare_len, c.wall_t, c.wall_h), color=c.wall_color,
                 yaw_deg=sgn * -c.flare_deg + 180.0, collide=collide)
    return root


def _spawn_can(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a food can: DYNAMIC single cylinder collider + a visual-only lid disc
    (no collision — a proud collider on top would snag the roof probes)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    _dyn_props(root, lin_damp=0.25, ang_damp=0.8)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_cyl(stage, f"{prim_path}/body",
             center=(0.0, 0.0, 0.0), radius=c.radius, height=c.height,
             color=c.color, collide=collide)
    mat = _phys_material(stage, f"{prim_path}/canMat", c.can_mu, c.can_mu * 0.9)
    _bind_material(stage.GetPrimAtPath(f"{prim_path}/body"), mat)
    _add_cyl(stage, f"{prim_path}/lid",
             center=(0.0, 0.0, c.height / 2 - 0.0015), radius=c.radius * 0.92,
             height=0.002, color=(0.78, 0.78, 0.80), collide=None)  # visual only
    return root


def _spawn_token(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author an order tile: KINEMATIC thin colored box (pure marker; it sits on the
    roof, outside every can path)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    _add_box(stage, f"{prim_path}/tile",
             center=(0.0, 0.0, 0.0), size=(cfg.size_x, cfg.size_y, cfg.size_z),
             color=cfg.color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rack" not in _SPAWNER_CACHE:

        @configclass
        class RackSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rack)
            interior_len: float = 0.185
            interior_w: float = 0.066
            wall_t: float = 0.010
            wall_h: float = 0.080
            floor_t: float = 0.008
            roof_t: float = 0.008
            apron_len: float = 0.16
            flare_len: float = 0.10
            flare_deg: float = 30.0
            floor_mu: float = 0.10
            floor_color: tuple = (0.30, 0.26, 0.22)
            wall_color: tuple = (0.55, 0.40, 0.24)
            roof_color: tuple = (0.45, 0.32, 0.19)
            contact_offset: float = 0.002

        @configclass
        class CanSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_can)
            radius: float = 0.025
            height: float = 0.050
            mass: float = 0.2
            can_mu: float = 0.25
            color: tuple = (0.5, 0.5, 0.5)
            contact_offset: float = 0.002

        @configclass
        class TokenSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_token)
            size_x: float = 0.036
            size_y: float = 0.055
            size_z: float = 0.006
            color: tuple = (0.5, 0.5, 0.5)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(rack=RackSpawnerCfg, can=CanSpawnerCfg,
                              token=TokenSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class PantryRackOrderSceneCfg(BaseCfg):
    """Config for `PantryRackOrderScene`.

    No-passing proof (why depth order == insertion order): interior width 66 mm; a
    can of radius r can offset its centre at most (33 - r) mm from the axis. Worst
    pair (red 20 + green 25): max centre separation 13 + 8 = 21 mm << 45 mm = sum of
    radii. Every other pair is worse. So cans inside can NEVER swap depth ranks."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.70)  # max |ang vel| when judging (rad/s)
    in_margin: float = tunable(0.008)  # fully inside: depth >= radius + this
    order_margin: float = tunable(0.030)  # strict rank margin (contiguous >= 45 mm)
    z_tol: float = tunable(0.015)  # can centre height band about floor_top + h/2
    upright_max_deg: float = tunable(20.0)  # can axis within this of vertical
    streak_n: int = tunable(8)  # consecutive still steps to latch a stage

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    rack_x_jit: float = tunable(0.03)  # rack x jitter (+/- m)
    rack_y_max: float = tunable(0.05)  # rack lateral offset (+/- m)
    rack_yaw_deg: float = tunable(20.0)  # rack yaw jitter (+/- deg, mouth toward base)
    slot_jitter: float = tunable(0.03)  # per-can staging xy jitter (+/- m)
    shuffle_perm: bool = tunable(True)  # random target permutation (False -> identity)
    shuffle_slots: bool = tunable(True)  # random can->staging-slot assignment

    # --- info: layout (single Franka base at the origin) -----------------------------------------
    rack_x: float = info(0.44)  # mouth distance from the base
    stage_slots: tuple = info(((0.16, -0.20), (0.13, 0.0), (0.16, 0.20)))  # ground

    # --- info: rack structure --------------------------------------------------------------------
    interior_len: float = info(0.185)  # mouth plane (x=0) to closed-end inner face
    interior_w: float = info(0.066)
    wall_t: float = info(0.010)
    wall_h: float = info(0.080)
    floor_t: float = info(0.008)
    roof_t: float = info(0.008)
    apron_len: float = info(0.16)
    flare_len: float = info(0.10)
    flare_deg: float = info(30.0)
    floor_mu: float = info(0.10)  # slick channel floor (pair-averaged with can_mu)
    can_mu: float = info(0.25)

    # --- info: cans (name, radius, height, mass, color, color word) ------------------------------
    can_specs: tuple = info((
        ("tomato", 0.020, 0.050, 0.12, (0.80, 0.10, 0.08), "RED"),
        ("peas", 0.025, 0.050, 0.20, (0.10, 0.55, 0.15), "GREEN"),
        ("soup", 0.030, 0.050, 0.30, (0.12, 0.25, 0.78), "BLUE"),
    ))

    # --- info: order tiles on the roof (x slots, deepest first) ----------------------------------
    token_x: tuple = info((0.150, 0.0955, 0.041))
    token_size: tuple = info((0.036, 0.055, 0.006))

    contact_offset: float = info(0.002)
    w_stage: float = info(0.30)  # per-stage latched credit (cap 0.60)

    @property
    def floor_top(self) -> float:
        return self.floor_t

    @property
    def roof_top(self) -> float:
        return self.floor_t + self.wall_h + self.roof_t


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("pantry_rack_order")
class PantryRackOrderScene(BaseScene):
    cfg: PantryRackOrderSceneCfg

    def __init__(self, cfg: PantryRackOrderSceneCfg | None = None) -> None:
        super().__init__(cfg or PantryRackOrderSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        rack_spawn = spawners["rack"](
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            interior_len=c.interior_len, interior_w=c.interior_w, wall_t=c.wall_t,
            wall_h=c.wall_h, floor_t=c.floor_t, roof_t=c.roof_t,
            apron_len=c.apron_len, flare_len=c.flare_len, flare_deg=c.flare_deg,
            floor_mu=c.floor_mu, contact_offset=c.contact_offset,
        )
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
            "rack": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rack",
                spawn=rack_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.rack_x, 0.0, 0.0)),
            ),
        }
        for i, (name, r, h, m, color, _cw) in enumerate(c.can_specs):
            out[f"can_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Can_" + name,
                spawn=spawners["can"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=m),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    radius=r, height=h, mass=m, can_mu=c.can_mu, color=color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stage_slots[i][0], c.stage_slots[i][1], h / 2 + 0.002)),
            )
            out[f"token_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Token_" + name,
                spawn=spawners["token"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    size_x=c.token_size[0], size_y=c.token_size[1],
                    size_z=c.token_size[2], color=color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rack_x + c.token_x[i], 0.0,
                         c.roof_top + c.token_size[2] / 2 + 0.001)),
            )
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "enable_external_forces_every_iteration": True,
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
        c = self.cfg
        self.rack: RigidObject = env.iscene["rack"]
        self.cans: list[RigidObject] = [env.iscene[f"can_{n}"]
                                        for n, *_ in c.can_specs]
        self.tokens: list[RigidObject] = [env.iscene[f"token_{n}"]
                                          for n, *_ in c.can_specs]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self._radii = torch.tensor([s[1] for s in c.can_specs], device=dev)
        self._height = float(c.can_specs[0][2])
        # target permutation: perm[e] = can indices, DEEPEST FIRST
        self._perm = torch.arange(3, device=dev).unsqueeze(0).expand(n, 3).clone()
        self._s1_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._s2_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._s1_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._s2_ever = torch.zeros(n, dtype=torch.bool, device=dev)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: rack re-posed (x/y jitter + yaw), target permutation
        sampled and DISPLAYED by re-posing the roof tiles (deepest-first slots),
        cans assigned to shuffled staging slots on the ground with xy jitter,
        latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        _ = torch.rand(m, 4, device=dev)  # burn draws (first post-seed draw trap)

        # --- rack (kinematic): x/y jitter + yaw, mouth toward the base ---
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rack_yaw_deg)
        rx = c.rack_x + (torch.rand(m, device=dev) * 2 - 1) * c.rack_x_jit
        ry = (torch.rand(m, device=dev) * 2 - 1) * c.rack_y_max
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = rx, ry
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.rack.write_root_state_to_sim(st, env_ids)

        # --- target permutation (deepest-first) + roof tiles at the rank slots ---
        if c.shuffle_perm:
            perm = torch.rand(m, 3, device=dev).argsort(dim=1)
        else:
            perm = torch.arange(3, device=dev).unsqueeze(0).expand(m, 3).clone()
        self._perm[env_ids] = perm
        rank_of = perm.argsort(dim=1)  # rank_of[e, can] = rank (0 = deepest)
        tok_x = torch.tensor(c.token_x, device=dev)
        tz = c.roof_top + c.token_size[2] / 2 + 0.001
        for i in range(3):
            lx = tok_x[rank_of[:, i]]  # local x of can i's tile
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = rx + cy * lx
            st[:, 1] = ry + sy * lx
            st[:, 2] = tz
            st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
            st[:, 0:3] += origin
            self.tokens[i].write_root_state_to_sim(st, env_ids)

        # --- cans: shuffled staging slots on the ground + xy jitter, standing ---
        if c.shuffle_slots:
            slot_perm = torch.rand(m, 3, device=dev).argsort(dim=1)
        else:
            slot_perm = torch.arange(3, device=dev).unsqueeze(0).expand(m, 3).clone()
        slots = torch.tensor(c.stage_slots, device=dev, dtype=torch.float)  # (3,2)
        for i in range(3):
            xy = slots[slot_perm[:, i]] \
                + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = self._height / 2 + 0.002
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            self.cans[i].write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._s1_streak[env_ids] = 0
        self._s2_streak[env_ids] = 0
        self._s1_ever[env_ids] = False
        self._s2_ever[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "rack": self.rack.data.root_state_w[env_ids].clone(),
            "cans": [b.data.root_state_w[env_ids].clone() for b in self.cans],
            "tokens": [b.data.root_state_w[env_ids].clone() for b in self.tokens],
            "perm": self._perm[env_ids].clone(),
            "s1_streak": self._s1_streak[env_ids].clone(),
            "s2_streak": self._s2_streak[env_ids].clone(),
            "s1_ever": self._s1_ever[env_ids].clone(),
            "s2_ever": self._s2_ever[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.rack.write_root_state_to_sim(state["rack"], env_ids)
        for b, s in zip(self.cans, state["cans"]):
            b.write_root_state_to_sim(s, env_ids)
        for b, s in zip(self.tokens, state["tokens"]):
            b.write_root_state_to_sim(s, env_ids)
        self._perm[env_ids] = state["perm"]
        self._s1_streak[env_ids] = state["s1_streak"]
        self._s2_streak[env_ids] = state["s2_streak"]
        self._s1_ever[env_ids] = state["s1_ever"]
        self._s2_ever[env_ids] = state["s2_ever"]

    # ----- description ---------------------------------------------------------------------------
    def _perm_words(self) -> list[str] | None:
        if not hasattr(self, "_perm"):
            return None
        c = self.cfg
        p = self._perm[0].tolist()
        return [f"{c.can_specs[i][5]} {c.can_specs[i][0]} can "
                f"({2 * c.can_specs[i][1] * 1000:.0f} mm)" for i in p]

    def describe(self) -> str:
        c = self.cfg
        words = self._perm_words()
        order_txt = ""
        if words is not None:
            order_txt = (f" In this episode the tiles read, closed end first: "
                         f"1) {words[0]}, 2) {words[1]}, 3) {words[2]}.")
        return (
            f"A wooden PANTRY RACK stands on the floor: a low channel "
            f"({c.interior_w * 1000:.0f} mm wide, {c.wall_h * 1000:.0f} mm tall inside, "
            f"{c.interior_len * 1000:.0f} mm deep) with a slick floor, two side walls, "
            f"a closed back end and a ROOF. Its only opening is the front MOUTH, "
            f"facing you, with a flared loading apron in front. Because of the roof, "
            f"cans can only get inside by SLIDING through the mouth along the floor — "
            f"nothing can be lowered in from above. The channel is too narrow for two "
            f"cans to pass one another inside, so cans inside keep their front-to-back "
            f"order forever: the first can pushed in stays deepest.\n"
            f"Three food cans stand on the ground nearby (same height "
            f"{c.can_specs[0][2] * 1000:.0f} mm, identified by color and width): a "
            f"RED tomato can (40 mm wide), a GREEN peas can (50 mm) and a BLUE soup "
            f"can (60 mm). Their starting spots are shuffled every episode.\n"
            f"Goal: load ALL THREE cans fully into the rack, standing upright, so "
            f"that their order from the CLOSED END outward matches the three colored "
            f"ORDER TILES on the rack roof: the tile nearest the closed end names "
            f"the can that must end up deepest, the middle tile the middle can, and "
            f"the tile nearest the mouth the outermost can.{order_txt}\n"
            f"Plan the insertion sequence before you start: push in the "
            f"deepest-required can first, then the middle one, then the outermost — "
            f"each new can pushes the earlier ones deeper. A can inserted out of "
            f"turn cannot be reordered inside the channel. A can counts only when it "
            f"is fully past the mouth, upright and at rest; only the final settled "
            f"arrangement is judged."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        words = self._perm_words()
        order_txt = (f" — here: {words[0].split()[0]} deepest, then {words[1].split()[0]}, "
                     f"then {words[2].split()[0]} at the mouth" if words else "")
        return (
            "Slide the three cans upright through the mouth of the roofed pantry "
            "rack, deepest-required first, so their order from the closed end "
            "matches the colored order tiles on the roof" + order_txt +
            ". All three must end fully inside, upright and at rest; a can pushed "
            "in out of order cannot be reordered."
        )

    # ----- readings ------------------------------------------------------------------------------
    def _rack_local(self, p_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.rack.data.root_quat_w,
                                  p_w - self.rack.data.root_pos_w)

    def can_locals(self) -> torch.Tensor:
        """(N, 3, 3): rack-local positions of the three cans (x = depth)."""
        return torch.stack([self._rack_local(b.data.root_pos_w)
                            for b in self.cans], dim=1)

    def depths(self) -> torch.Tensor:
        """(N, 3): can-centre depth past the mouth plane (rack-local x)."""
        return self.can_locals()[:, :, 0]

    def upright(self) -> torch.Tensor:
        """(N, 3) bool: can axis within `upright_max_deg` of world-up."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        ups = torch.stack([quat_apply(b.data.root_quat_w, ez)[:, 2]
                           for b in self.cans], dim=1)
        return ups.clamp(-1.0, 1.0) >= math.cos(math.radians(c.upright_max_deg))

    def inside(self) -> torch.Tensor:
        """(N, 3) bool: can fully past the mouth plane (depth >= radius +
        `in_margin`), between the walls, standing on the channel floor, upright."""
        c = self.cfg
        loc = self.can_locals()
        d = loc[:, :, 0]
        fully = d >= self._radii.unsqueeze(0) + c.in_margin
        in_y = loc[:, :, 1].abs() <= c.interior_w / 2
        z0 = c.floor_top + self._height / 2
        in_z = (loc[:, :, 2] - z0).abs() <= c.z_tol
        return fully & in_y & in_z & self.upright()

    def still(self) -> torch.Tensor:
        """(N, 3) bool per can."""
        c = self.cfg
        lin = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.cans],
                          dim=1)
        ang = torch.stack([b.data.root_ang_vel_w.norm(dim=-1) for b in self.cans],
                          dim=1)
        return (lin < c.settle_speed) & (ang < c.settle_omega)

    # ----- rubric --------------------------------------------------------------------------------
    def _stage_conds(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(s1, s2, order_ok), each (N,) bool — pure geometry, no stillness.

        s1: the can required deepest (t0) is fully inside and strictly deeper (by
            `order_margin`) than every OTHER can that is inside.
        s2: s1 AND the second-required can (t1) is fully inside, below t0, and the
            third can (t2) is either not inside or shallower than t1.
        order_ok: all three inside and depth(t0) > depth(t1) > depth(t2) with the
            strict margins.
        """
        c = self.cfg
        d = self.depths()
        ins = self.inside()
        p = self._perm
        d0 = d.gather(1, p[:, 0:1]).squeeze(1)
        d1 = d.gather(1, p[:, 1:2]).squeeze(1)
        d2 = d.gather(1, p[:, 2:3]).squeeze(1)
        i0 = ins.gather(1, p[:, 0:1]).squeeze(1)
        i1 = ins.gather(1, p[:, 1:2]).squeeze(1)
        i2 = ins.gather(1, p[:, 2:3]).squeeze(1)
        m = c.order_margin
        s1 = i0 & (~i1 | (d0 >= d1 + m)) & (~i2 | (d0 >= d2 + m))
        s2 = s1 & i1 & (~i2 | (d1 >= d2 + m))
        order_ok = i0 & i1 & i2 & (d0 >= d1 + m) & (d1 >= d2 + m)
        return s1, s2, order_ok

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch bookkeeping every step (streaks demand stillness of every can, so a
        can flying past a rank never latches a stage). Streaks only advance HERE —
        success()/score() are read-only."""
        s1, s2, _ = self._stage_conds()
        all_still = self.still().all(dim=1)
        self._s1_streak = torch.where(s1 & all_still, self._s1_streak + 1,
                                      torch.zeros_like(self._s1_streak))
        self._s1_ever |= self._s1_streak >= self.cfg.streak_n
        self._s2_streak = torch.where(s2 & all_still, self._s2_streak + 1,
                                      torch.zeros_like(self._s2_streak))
        self._s2_ever |= self._s2_streak >= self.cfg.streak_n

    def success(self) -> torch.Tensor:
        """(N,) bool: all three cans fully inside, upright, in the exact target
        depth order (strict margins), everything still — judged LIVE on the settled
        physical arrangement."""
        _s1, _s2, order_ok = self._stage_conds()
        return order_ok & self.still().all(dim=1)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.30 per latched stage (cap 0.60), exactly 1.0 iff
        success() holds live. Doing nothing scores 0; a full insertion in a wrong
        order latches at most the stages whose prefix it actually honored."""
        c = self.cfg
        base = (c.w_stage * self._s1_ever.float()
                + c.w_stage * self._s2_ever.float()).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through pose writes and
# applied forces.
register_env("simgen", lambda: EnvCfg(scene="pantry_rack_order", robot="null"))

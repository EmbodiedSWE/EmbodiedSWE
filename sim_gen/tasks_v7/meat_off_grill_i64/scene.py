"""KebabSpitScene — meter the COOKED pieces off the overhanging tip of a captive kebab
spit into the serving tray below, leaving every RAW piece threaded. Derived from
rlbench/meat_off_grill but the meat is CAPTIVE ON A SPIT, so the seed's plan is dead
on arrival.

Seed (rlbench/meat_off_grill): two free meat pieces REST on a grill; the whole plan is
grasp-a-free-body, lift it off the grill, set it beside — an unordered pick-and-place
with nothing constraining the motion. Here nothing rests free and nothing is carried:

- Every piece is a RING (a cube with a square through-channel) THREADED on a long
  square SPIT RAIL, cantilevered from a mast over a grill bed. A piece cannot be
  lifted off, tipped off, or pulled sideways off the rail (smoke's captivity probe
  hauls one straight up at ~6x its weight and it stays threaded) — the ONLY exit is
  travelling AXIALLY past the rail's free TIP, which overhangs a serving tray.
- The pieces form a train: they cannot pass each other, so the serving order is
  physically FIFO from the tip inward. The BROWN (cooked) pieces are the outboard
  group next to the tip; the PINK (raw) pieces are inboard next to the mast.
- The goal is a METERED PARTIAL UNLOAD: slide every cooked piece off the tip so it
  drops and settles INSIDE the tray, while every raw piece REMAINS threaded on the
  rail. Pushing the train from the butt end (the naive move) shoves the raw pieces
  toward the tip too and overruns; the correct plan is to find the cooked/raw
  BOUNDARY, reach into the gap there, and push only the cooked group.

So a solver needs a different PLAN (identify a color boundary in a captive train,
push at the boundary through a finger-sized gap, convey the group axially off an
overhang, let gravity capture the pieces in the tray, and STOP with the raw group
untouched) and a different CODE STRUCTURE (axial push control with a stop condition
instead of grasp-lift-place). No piece is ever grasped or carried; the judged bodies
travel only along the rail and by falling.

Assets are fully procedural (hockey/pen_holder-pattern compound spawners; child
colliders of one body never self-collide):
  - spit fixture: KINEMATIC compound — mast, cantilevered square rail (the free span
    the rings ride), a dark grill bed under the span, and the serving tray (floor +
    4 walls) under the rail's overhanging tip. Origin on the ground under the rail;
    local +x runs from the mast to the tip.
  - pieces: DYNAMIC rings — 42 mm cubes with a 26 mm square channel (5 mm clearance
    per side around the 16 mm rail; the square-on-square section cannot spin). Brown
    = cooked, pink = raw. Absent pieces park in an off-scene ground depot.

Per-episode randomization (readback-verifiable): cooked count 1-3 and raw count 1-2,
fixture xy jitter + yaw, tip standoff and every inter-piece gap resampled (the gaps
are where a fingertip reaches in — they move every episode).

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.15 * advance    — cooked group ever conveyed along +x WHILE THREADED (running
                      max over on-rail displacement, ramp 8 cm; ~0 for doing nothing)
  0.35 * off-tip    — fraction of cooked pieces ever past the tip (latched; a captive
                      ring can only get there by travelling the rail)
  0.25 * in-tray    — fraction of cooked pieces ever settled inside the tray (latched)
  1.0 iff success() — every cooked piece settled INSIDE the tray, every raw piece
                      still threaded on the rail, no absent piece in the tray, all
                      still. Non-success cap 0.75.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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


def _add_box(stage, path: str, *, center, size, color, collide: Callable) -> None:
    """Author one axis-aligned box collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _make_collide(cfg: Any) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _bind_mat(prim_path: str, children: list[str], static: float, dynamic: float) -> None:
    """Author one physics material under the body root and bind it to the child
    colliders (custom spawner colliders otherwise get the ~0.5 default silently)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    mat_path = f"{prim_path}/physMat"
    sim_utils.spawn_rigid_body_material(
        mat_path,
        sim_utils.RigidBodyMaterialCfg(static_friction=static, dynamic_friction=dynamic,
                                       restitution=0.0))
    for child in children:
        bind_physics_material(child, mat_path)


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


def _spawn_spit(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the spit fixture: KINEMATIC compound. Origin on the GROUND under the
    rail; local +x from mast to tip. Children: mast, rail (square bar, tip at
    x = tip_x, free span for the rings), grill bed under the span, serving tray
    (floor + 4 walls) under the overhanging tip."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    a = c.rail_a
    # mast (holds the butt end of the rail)
    _add_box(stage, f"{prim_path}/mast",
             center=(c.mast_x, 0.0, c.mast_h / 2),
             size=(0.05, 0.06, c.mast_h), color=c.mast_color, collide=collide)
    # rail: from inside the mast to the free tip
    x0 = c.mast_x + 0.01
    _add_box(stage, f"{prim_path}/rail",
             center=((x0 + c.tip_x) / 2, 0.0, c.rail_z),
             size=(c.tip_x - x0, a, a), color=c.rail_color, collide=collide)
    # grill bed under the span (theme + a floor the rings never touch)
    _add_box(stage, f"{prim_path}/bed",
             center=(-0.06, 0.0, 0.03),
             size=(0.44, 0.14, 0.06), color=c.bed_color, collide=collide)
    _add_box(stage, f"{prim_path}/bed_glow",
             center=(-0.06, 0.0, 0.0615),
             size=(0.40, 0.10, 0.003), color=(0.85, 0.30, 0.08), collide=collide)
    # serving tray under the overhanging tip: floor + 4 walls
    tx0, tx1 = c.tray_x0, c.tray_x1
    ty, wt, wh = c.tray_hw, 0.010, c.tray_wall_h
    _add_box(stage, f"{prim_path}/tray_floor",
             center=((tx0 + tx1) / 2, 0.0, 0.006),
             size=(tx1 - tx0, 2 * ty, 0.012), color=c.tray_color, collide=collide)
    _add_box(stage, f"{prim_path}/tray_front",
             center=(tx0 - wt / 2, 0.0, wh / 2),
             size=(wt, 2 * ty + 2 * wt, wh), color=c.tray_color, collide=collide)
    _add_box(stage, f"{prim_path}/tray_back",
             center=(tx1 + wt / 2, 0.0, wh / 2),
             size=(wt, 2 * ty + 2 * wt, wh), color=c.tray_color, collide=collide)
    for sgn, nm in ((1.0, "tray_l"), (-1.0, "tray_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=((tx0 + tx1) / 2, sgn * (ty + wt / 2), wh / 2),
                 size=(tx1 - tx0, wt, wh), color=c.tray_color, collide=collide)
    _bind_mat(prim_path, [f"{prim_path}/rail"], c.rail_mu, max(c.rail_mu - 0.03, 0.02))
    return root


def _spawn_ring(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one ring piece: DYNAMIC cube with a square through-channel along local
    +x (4 child boxes forming a tube). Origin at the volumetric centre — which is the
    channel axis, so the explicit MassAPI mass keeps the CoM on the rail axis."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    # High angular damping: kills the edge-contact spin limit cycle a piece can enter
    # when it lands on top of a pile (constant phantom |w| that never settles). Meat
    # is soft — a spinning kebab cube is not a behaviour worth preserving.
    pxrb.CreateAngularDampingAttr(1.2)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg)
    s, ch = cfg.piece_s, cfg.chan_half
    w = s / 2 - ch  # wall thickness of the tube
    for sgn, nm in ((1.0, "wall_l"), (-1.0, "wall_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(0.0, sgn * (ch + w / 2), 0.0),
                 size=(s, w, s), color=cfg.color, collide=collide)
    for sgn, nm in ((1.0, "plate_t"), (-1.0, "plate_b")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(0.0, 0.0, sgn * (ch + w / 2)),
                 size=(s, 2 * ch, w), color=cfg.color, collide=collide)
    _bind_mat(prim_path,
              [f"{prim_path}/{nm}" for nm in ("wall_l", "wall_r", "plate_t", "plate_b")],
              cfg.mu, max(cfg.mu - 0.03, 0.02))
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "spit" not in _SPAWNER_CACHE:

        @configclass
        class SpitSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_spit)
            rail_a: float = 0.016
            rail_z: float = 0.16
            mast_x: float = -0.335
            mast_h: float = 0.20
            tip_x: float = 0.24
            tray_x0: float = 0.21
            tray_x1: float = 0.41
            tray_hw: float = 0.096
            tray_wall_h: float = 0.082
            rail_mu: float = 0.20
            mast_color: tuple = (0.25, 0.25, 0.28)
            rail_color: tuple = (0.75, 0.76, 0.80)
            bed_color: tuple = (0.13, 0.12, 0.12)
            tray_color: tuple = (0.85, 0.87, 0.90)
            contact_offset: float = 0.002

        @configclass
        class RingSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_ring)
            piece_s: float = 0.042
            chan_half: float = 0.013
            mu: float = 0.35
            color: tuple = (0.45, 0.26, 0.12)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(spit=SpitSpawnerCfg, ring=RingSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class KebabSpitSceneCfg(BaseCfg):
    """Config for `KebabSpitScene`. The interlocks are metric: the 26 mm square
    channel around the 16 mm square rail is captive everywhere except past the tip
    (max escape tilt ~13 deg while any channel length overlaps the rail), the rail's
    tip overhangs the tray's inner region by 30 mm (a piece that clears the tip falls
    inside), and the pieces cannot pass each other on the rail (train order is
    physical FIFO)."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.80)  # max |ang vel| when judging (rad/s)
    adv_ramp: float = tunable(0.08)  # advance credit ramp (m of +x conveyance)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    fix_jitter: float = tunable(0.03)  # fixture xy jitter (+/- m)
    fix_yaw_deg: float = tunable(12.0)  # fixture yaw about nominal (+/- deg)
    standoff_rng: tuple = tunable((0.05, 0.09))  # outermost piece face to tip
    gap_rng: tuple = tunable((0.040, 0.058))  # inter-piece gaps (fingertip room)
    n_cooked_max: int = tunable(3)  # cooked count sampled in {1..n_cooked_max}
    n_raw_max: int = tunable(2)  # raw count sampled in {1..n_raw_max}

    # --- info: layout (single Franka base at the origin; nominal yaw 90 deg puts the
    # rail roughly along world +y, every interaction point at radius 0.35-0.55 m) ----------------
    fix_pos: tuple = info((0.42, 0.02))  # fixture origin (world xy, nominal)
    fix_yaw_nom_deg: float = info(90.0)
    park_pos: tuple = info((1.10, 1.10))  # ground depot for absent pieces

    # --- info: fixture structure -----------------------------------------------------------------
    rail_a: float = info(0.016)  # rail square cross-section
    rail_z: float = info(0.16)  # rail axis height
    mast_x: float = info(-0.335)  # mast centre (local x)
    mast_face_x: float = info(-0.31)  # inner mast face — butt end of the free span
    tip_x: float = info(0.24)  # the rail's free tip (local x)
    tray_x0: float = info(0.21)  # tray inner region x in [tray_x0, tray_x1]
    tray_x1: float = info(0.41)
    tray_hw: float = info(0.096)  # tray inner half-width (|y|)
    tray_wall_h: float = info(0.082)
    rail_mu: float = info(0.20)

    # --- info: pieces ----------------------------------------------------------------------------
    piece_s: float = info(0.042)  # ring outer cube size
    chan_half: float = info(0.013)  # channel half-width (26 mm hole on 16 mm rail)
    piece_mass: float = info(0.05)
    piece_mu: float = info(0.35)
    cooked_color: tuple = info((0.45, 0.26, 0.12))  # brown
    raw_color: tuple = info((0.93, 0.52, 0.55))  # pink
    contact_offset: float = info(0.002)

    # rubric weights (0.15 + 0.35 + 0.25 = 0.75 = the non-success cap)
    w_adv: float = info(0.15)
    w_off: float = info(0.35)
    w_tray: float = info(0.25)

    def __post_init__(self) -> None:
        # honesty geometry: worst-case train + gaps + standoff fits the free span
        worst = (self.standoff_rng[1]
                 + (self.n_cooked_max + self.n_raw_max) * self.piece_s
                 + (self.n_cooked_max + self.n_raw_max - 1) * self.gap_rng[1])
        span = self.tip_x - self.mast_face_x
        assert worst <= span - 0.01, f"train does not fit the rail span ({worst} > {span})"
        # a piece that clears the tip falls inside the tray inner region
        assert self.tray_x0 + 0.02 <= self.tip_x <= self.tray_x1 - 0.10, \
            "tip must overhang the tray's inner region with fall room"
        # the smallest gap admits a fingertip (embodiment argument)
        assert self.gap_rng[0] >= 0.035, "gaps must admit a Franka fingertip"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("kebab_spit")
class KebabSpitScene(BaseScene):
    cfg: KebabSpitSceneCfg

    N_COOKED, N_RAW = 3, 2  # spawned bodies (per-episode counts sample subsets)

    def __init__(self, cfg: KebabSpitSceneCfg | None = None) -> None:
        super().__init__(cfg or KebabSpitSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        spit_spawn = spawners["spit"](
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            rail_a=c.rail_a, rail_z=c.rail_z, mast_x=c.mast_x, tip_x=c.tip_x,
            tray_x0=c.tray_x0, tray_x1=c.tray_x1, tray_hw=c.tray_hw,
            tray_wall_h=c.tray_wall_h, rail_mu=c.rail_mu,
            contact_offset=c.contact_offset,
        )

        def ring_spawn(color):
            return spawners["ring"](
                mass_props=sim_utils.MassPropertiesCfg(mass=c.piece_mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=32,
                    solver_velocity_iteration_count=4),
                piece_s=c.piece_s, chan_half=c.chan_half, mu=c.piece_mu,
                color=color, contact_offset=c.contact_offset,
            )

        yaw0 = math.radians(c.fix_yaw_nom_deg)
        q0 = (math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2))
        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.55, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "spit": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Spit",
                spawn=spit_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.fix_pos[0], c.fix_pos[1], 0.0), rot=q0),
            ),
        }
        # nominal spawn poses: threaded along the rail at the nominal fixture pose
        # (reset() re-threads everything; these just avoid a spawn-inside-mast state)
        z_rest = c.rail_z + c.chan_half - c.rail_a / 2  # channel bottom on rail bottom
        for i in range(self.N_COOKED + self.N_RAW):
            cooked = i < self.N_COOKED
            name = f"cooked_{i}" if cooked else f"raw_{i - self.N_COOKED}"
            xl = c.tip_x - 0.07 - (c.piece_s + 0.05) * i - c.piece_s / 2
            wx = c.fix_pos[0] + xl * math.cos(yaw0)
            wy = c.fix_pos[1] + xl * math.sin(yaw0)
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ring_" + name,
                spawn=ring_spawn(c.cooked_color if cooked else c.raw_color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(wx, wy, z_rest), rot=q0),
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

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.spit: RigidObject = env.iscene["spit"]
        self.cooked: list[RigidObject] = [env.iscene[f"cooked_{i}"] for i in range(self.N_COOKED)]
        self.raw: list[RigidObject] = [env.iscene[f"raw_{i}"] for i in range(self.N_RAW)]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.n_cooked = torch.full((n,), self.N_COOKED, dtype=torch.long, device=dev)
        self.n_raw = torch.full((n,), self.N_RAW, dtype=torch.long, device=dev)
        # latches: partial progress survives transient achievements (rubric requirement)
        self._adv_max = torch.zeros(n, device=dev)
        self._off_ever = torch.zeros(n, self.N_COOKED, dtype=torch.bool, device=dev)
        self._tray_ever = torch.zeros(n, self.N_COOKED, dtype=torch.bool, device=dev)
        self._cooked_x0 = torch.zeros(n, self.N_COOKED, device=dev)  # reset local x

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: fixture re-posed (xy jitter + yaw), cooked/raw counts
        sampled, the present train re-threaded from the tip inward — cooked group
        outboard, raw group inboard — with fresh standoff + gaps; absent pieces park
        in the ground depot; latches cleared."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- fixture (kinematic): xy jitter + yaw ---
        yaw = (math.radians(c.fix_yaw_nom_deg)
               + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.fix_yaw_deg))
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.fix_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.fix_jitter
        st[:, 1] = c.fix_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.fix_jitter
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.spit.write_root_state_to_sim(st, env_ids)
        f_pos, f_quat = st[:, 0:3].clone(), st[:, 3:7].clone()

        # --- counts ---
        nc = torch.randint(1, c.n_cooked_max + 1, (m,), device=dev)
        nr = torch.randint(1, c.n_raw_max + 1, (m,), device=dev)
        self.n_cooked[env_ids], self.n_raw[env_ids] = nc, nr

        # --- train layout from the tip inward: standoff + per-slot gaps ---
        s = c.piece_s
        d0 = (c.standoff_rng[0]
              + torch.rand(m, device=dev) * (c.standoff_rng[1] - c.standoff_rng[0]))
        gaps = (c.gap_rng[0]
                + torch.rand(m, self.N_COOKED + self.N_RAW - 1, device=dev)
                * (c.gap_rng[1] - c.gap_rng[0]))
        # sequence position j (0 = outermost): x_j = tip - d0 - s/2 - j*s - sum(gaps[:j])
        gcum = torch.cat([torch.zeros(m, 1, device=dev), torch.cumsum(gaps, dim=1)], dim=1)
        j_idx = torch.arange(self.N_COOKED + self.N_RAW, device=dev).unsqueeze(0)
        x_seq = c.tip_x - d0.unsqueeze(1) - s / 2 - j_idx * s - gcum  # (m, 5)

        z_rest = c.rail_z + c.chan_half - c.rail_a / 2 + 0.001
        park = torch.tensor(c.park_pos, device=dev)

        def write_piece(body, seq_idx: torch.Tensor, present: torch.Tensor,
                        park_slot: int) -> torch.Tensor:
            xl = x_seq.gather(1, seq_idx.clamp(0, x_seq.shape[1] - 1).unsqueeze(1)).squeeze(1)
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0], loc[:, 2] = xl, z_rest
            on_rail_pos = f_pos + quat_apply(f_quat, loc)
            park_pos = torch.zeros(m, 3, device=dev)
            park_pos[:, 0] = park[0] + 0.12 * park_slot
            park_pos[:, 1] = park[1]
            park_pos[:, 2] = s / 2 + 0.002
            park_pos += origin
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = torch.where(present.unsqueeze(1), on_rail_pos, park_pos)
            ident = torch.zeros(m, 4, device=dev)
            ident[:, 0] = 1.0
            st[:, 3:7] = torch.where(present.unsqueeze(1), f_quat, ident)
            body.write_root_state_to_sim(st, env_ids)
            return xl

        for i, body in enumerate(self.cooked):
            xl = write_piece(body, torch.full((m,), i, dtype=torch.long, device=dev),
                             nc > i, i)
            self._cooked_x0[env_ids, i] = xl
        for i, body in enumerate(self.raw):
            write_piece(body, nc + i, nr > i, self.N_COOKED + i)

        # --- clear latches ---
        self._adv_max[env_ids] = 0.0
        self._off_ever[env_ids] = False
        self._tray_ever[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "spit": self.spit.data.root_state_w[env_ids].clone(),
            "cooked": [b.data.root_state_w[env_ids].clone() for b in self.cooked],
            "raw": [b.data.root_state_w[env_ids].clone() for b in self.raw],
            "n_cooked": self.n_cooked[env_ids].clone(),
            "n_raw": self.n_raw[env_ids].clone(),
            "adv_max": self._adv_max[env_ids].clone(),
            "off_ever": self._off_ever[env_ids].clone(),
            "tray_ever": self._tray_ever[env_ids].clone(),
            "cooked_x0": self._cooked_x0[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.spit.write_root_state_to_sim(state["spit"], env_ids)
        for b, st in zip(self.cooked, state["cooked"]):
            b.write_root_state_to_sim(st, env_ids)
        for b, st in zip(self.raw, state["raw"]):
            b.write_root_state_to_sim(st, env_ids)
        self.n_cooked[env_ids] = state["n_cooked"]
        self.n_raw[env_ids] = state["n_raw"]
        self._adv_max[env_ids] = state["adv_max"]
        self._off_ever[env_ids] = state["off_ever"]
        self._tray_ever[env_ids] = state["tray_ever"]
        self._cooked_x0[env_ids] = state["cooked_x0"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A kebab SPIT stands over a dark grill bed: a horizontal square steel "
            f"rail ({c.rail_a * 1000:.0f} mm across, axis {c.rail_z * 100:.0f} cm above "
            f"the ground) clamped at one end into a dark mast and FREE at the other "
            f"end — the TIP — which overhangs a white SERVING TRAY "
            f"({(c.tray_x1 - c.tray_x0) * 100:.0f} x {2 * c.tray_hw * 100:.0f} cm inside, "
            f"{c.tray_wall_h * 100:.0f} cm walls) sitting on the ground below it. "
            f"Threaded on the rail is a row of {c.piece_s * 1000:.0f} mm cube-shaped "
            f"kebab pieces, each with a square hole through its middle: the piece "
            f"count varies by episode — next to the TIP sit one to three BROWN "
            f"(cooked) pieces, and behind them, toward the mast, one to two PINK "
            f"(raw) pieces. Count and locate them by color. Every piece is CAPTIVE: "
            f"the rail runs through its hole, so it cannot be lifted off or tipped "
            f"off anywhere along the span — the ONLY way off is to slide it along "
            f"the rail PAST THE FREE TIP, where it drops toward the tray. Pieces "
            f"cannot pass each other, so they come off strictly in order from the "
            f"tip inward. There are finger-sized gaps ({c.gap_rng[0] * 100:.0f}-"
            f"{c.gap_rng[1] * 100:.0f} cm) between neighbouring pieces.\n"
            f"Goal: serve the cooked kebab — slide EVERY BROWN piece along the rail "
            f"and off the tip so that each one ends up resting INSIDE the serving "
            f"tray, while EVERY PINK piece stays threaded on the rail. Push the "
            f"brown group from the brown/pink boundary (reach into the gap between "
            f"the innermost brown piece and the outermost pink piece); pushing the "
            f"whole row from the mast end shoves the pink pieces along too and "
            f"drives them off the tip — any pink piece off the rail is failure, and "
            f"a brown piece that lands outside the tray is failure."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide every brown cooked piece along the spit rail and off its free tip "
            "so each lands inside the serving tray below. Every pink raw piece must "
            "stay threaded on the rail."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def _fix_local(self, body: RigidObject) -> torch.Tensor:
        """(N, 3) body CoM position in the fixture frame (x mast->tip, z up)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = body.data.root_pos_w - self.spit.data.root_pos_w
        return quat_apply_inverse(self.spit.data.root_quat_w, rel)

    def _on_rail(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: piece still threaded — CoM within the captive tube around the
        rail axis (the 26 mm channel bounds |y|, z to ~6 mm; generous 15/14 mm here)
        and short of the tip."""
        c = self.cfg
        loc = self._fix_local(body)
        z0 = c.rail_z + c.chan_half - c.rail_a / 2
        return ((loc[:, 1].abs() <= 0.015)
                & ((loc[:, 2] - z0).abs() <= 0.014)
                & (loc[:, 0] >= c.mast_face_x - 0.005) & (loc[:, 0] <= c.tip_x - 0.004))

    def _in_tray(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: piece CoM inside the tray's inner volume (on the floor or on a
        settled pile — two-high is z ~0.075)."""
        c = self.cfg
        loc = self._fix_local(body)
        return ((loc[:, 0] >= c.tray_x0 + 0.005) & (loc[:, 0] <= c.tray_x1 - 0.005)
                & (loc[:, 1].abs() <= c.tray_hw - 0.003)
                & (loc[:, 2] > 0.012) & (loc[:, 2] < 0.135))

    def _still(self, body: RigidObject) -> torch.Tensor:
        c = self.cfg
        return ((body.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                & (body.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega))

    def _present_cooked(self) -> torch.Tensor:
        """(N, 3) bool: cooked piece i participates this episode."""
        idx = torch.arange(self.N_COOKED, device=self.env.device).unsqueeze(0)
        return idx < self.n_cooked.unsqueeze(1)

    def _present_raw(self) -> torch.Tensor:
        idx = torch.arange(self.N_RAW, device=self.env.device).unsqueeze(0)
        return idx < self.n_raw.unsqueeze(1)

    def _update_latches(self) -> None:
        c = self.cfg
        pres = self._present_cooked()
        loc_x = torch.stack([self._fix_local(b)[:, 0] for b in self.cooked], dim=1)
        # advance: running max of cooked +x conveyance WHILE THREADED (present pieces
        # only). Gating on _on_rail keeps the credit honest: +x displacement of a
        # piece that is not riding the rail is not rail conveyance.
        on_rail = torch.stack([self._on_rail(b) for b in self.cooked], dim=1)
        disp = (loc_x - self._cooked_x0).where(pres & on_rail, torch.zeros_like(loc_x))
        adv = (disp.max(dim=1).values / c.adv_ramp).clamp(0.0, 1.0)
        adv = torch.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)
        self._adv_max = torch.maximum(self._adv_max, adv)
        # off-tip and settled-in-tray, per cooked piece
        self._off_ever |= (loc_x > c.tip_x + 0.003) & pres
        in_tray = torch.stack([self._in_tray(b) & self._still(b) for b in self.cooked], dim=1)
        self._tray_ever |= in_tray & pres

    # ----- step-coupled bookkeeping (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No plant (the fixture is kinematic and jointless) — just latch rubric
        progress every step so transient achievements keep credit."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool, physical outcomes only: every present cooked piece settled
        INSIDE the tray, every present raw piece still THREADED on the rail and
        still, no absent piece in the tray, states finite."""
        self._update_latches()
        pc, pr = self._present_cooked(), self._present_raw()
        cooked_ok = torch.stack(
            [self._in_tray(b) & self._still(b) for b in self.cooked], dim=1)
        raw_ok = torch.stack(
            [self._on_rail(b) & self._still(b) for b in self.raw], dim=1)
        absent_clear = torch.stack(
            [~self._in_tray(b) for b in self.cooked + self.raw], dim=1)
        absent_mask = torch.cat([~pc, ~pr], dim=1)
        finite = torch.stack(
            [torch.isfinite(b.data.root_state_w).all(dim=-1)
             for b in [self.spit] + self.cooked + self.raw], dim=1).all(dim=1)
        return ((cooked_ok | ~pc).all(dim=1) & (raw_ok | ~pr).all(dim=1)
                & (absent_clear | ~absent_mask).all(dim=1) & finite)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*advance + 0.35*off-tip fraction + 0.25*in-tray
        fraction — all latched, ~0 for doing nothing, capped 0.75 — and exactly 1.0
        iff success() holds live."""
        c = self.cfg
        self._update_latches()
        pres = self._present_cooked().float()
        k = pres.sum(dim=1).clamp(min=1.0)
        off_frac = (self._off_ever.float() * pres).sum(dim=1) / k
        tray_frac = (self._tray_ever.float() * pres).sum(dim=1) / k
        base = (c.w_adv * self._adv_max + c.w_off * off_frac
                + c.w_tray * tray_frac).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through applied wrenches.
register_env("simgen", lambda: EnvCfg(scene="kebab_spit", robot="null"))

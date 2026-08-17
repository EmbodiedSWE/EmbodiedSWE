"""RailHangerScene — hang the salad-dressing bottle from the overhead rail rack.

Derived from libero/libero_pick_salad_dressing ("pick the salad dressing and place it
in the basket": grasp the amber bottle among grocery distractors, carry it through
free air and DROP it into an open basket — one pick-and-place judged by a containment
bbox). Here the goal is inverted from containment to SUSPENSION, and the basket is
demoted to a decoy: the amber bottle must end HANGING IN MID-AIR from an overhead
rack, and the only physical path to that state is a constrained lateral slide.

The rack is a gallows fixture: a heavy base slab, a column at one end, and TWO
PARALLEL HORIZONTAL RAILS cantilevered from the column at height `rail_top`,
forming a slot that is OPEN at the far end (flared mouth) and CLOSED at the column.
The bottle has a narrow NECK and a wide CAP FLANGE: the neck fits between the rails,
the cap does not, and the body does not fit below them — so the bottle can hang from
the rails by its cap, and the ONLY way in is to thread the neck through the open
mouth (cap above the rail plane, body below) and slide it along the slot. The goal
seat is at the CLOSED end: the hanging bottle's body against the full-height column
(root x = -0.102, measured on the forge). Nothing about dropping into a
receptacle survives from the seed — the end state is the bottle dangling in free
air, its weight carried by the cap-on-rails contact, and reaching it requires an
approach + insertion + constrained horizontal travel, not a vertical drop.

Decoys: a RED bottle of identical shape (hanging it instead fails) and the seed's
own receptacle — an open basket on the ground (placing the bottle in it earns
nothing; smoke proves it).

Assets are fully procedural (compound-spawner pattern; children of one rigid
compound never self-collide):
  - rack: heavy DYNAMIC compound (25 kg — dynamic, not kinematic, so teleports at
    reset behave; ground friction keeps it planted). Local frame: origin at the
    slab centre on the ground, slot along +x, mouth at +x. Children: base slab
    380x240x20 mm; column 40x100x300 mm spanning x [-0.17,-0.13], top flush with
    the rail plane; two rails 260x20x20 mm spanning x [-0.13,+0.13], tops at
    z=0.32, inner faces at y=+/-0.016 (gap 32 mm); two flare beams yawed +/-30 deg
    at the mouth (capture widens ~16 -> ~40 mm). Rails/flares/column are SLICK
    (min-combine) so the cap slides; the slab keeps normal friction so the rack
    itself stays put.
  - bottles (amber target / red decoy): compound, origin at the body centre:
    body cylinder r 28 x 140 mm; neck r 10 x 40 mm (local z 0.07..0.11); cap
    flange r 26 x 12 mm (local z 0.11..0.122), cap child SLICK. 0.25 kg. Hanging:
    cap underside on the rail tops -> root z = 0.21, body bottom 140 mm off the
    ground. Geometry gates the mechanism: neck dia 20 < gap 32 < cap dia 52 <
    body dia 56, and the body top hangs 20 mm below the rail undersides.
  - basket: open box (160 mm square floor, 90 mm walls), 1.0 kg — the seed's
    receptacle, present purely as a decoy.

Per-episode randomization (readback-verifiable): rack yaw +/-20 deg + xy jitter,
bottle station swap + per-piece xy jitter + free yaw, basket jitter.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  0.25 hooked — the bottle hanging from the rails ANYWHERE along the slot
  0.35 seated — hanging AT the closed end (neck at the column stop)
  capped at 0.60; 1.0 iff success(): the amber bottle hanging at the seat, upright,
  the red decoy NOT on the rack, everything settled (consecutive-still counter)
  and finite. Null policy ~0 (both credits require the bottle carried off the
  ground, threaded into the slot, and slid along it).

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


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qinv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[:, 1:] = -out[:, 1:]
    return out


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = torch.cat([torch.zeros_like(q[:, :1]), v], dim=-1)
    return _qmul(_qmul(q, qv), _qinv(q))[:, 1:]


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
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


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _add_box(stage, path: str, *, center, size, color, collide: Callable,
             yaw_deg: float = 0.0):
    """One box child: translate (+ optional z-rotation) + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw_deg:
        xf.AddRotateZOp().Set(float(yaw_deg))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _add_cyl(stage, path: str, *, center, radius, height, color, collide: Callable):
    """One z-axis cylinder child."""
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr("Z")
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())
    return cyl.GetPrim()


def _mk_material(prim_path: str, name: str, mu_s: float, mu_d: float, combine: str) -> str:
    import isaaclab.sim as sim_utils

    mat_path = f"{prim_path}/{name}"
    sim_utils.spawn_rigid_body_material(mat_path, sim_utils.RigidBodyMaterialCfg(
        static_friction=float(mu_s), dynamic_friction=float(mu_d), restitution=0.0,
        friction_combine_mode=combine))
    return mat_path


def _dyn_body(root, mass: float, lin_damp: float, ang_damp: float) -> None:
    """Standard dynamic compound body physics (32/4 iters, no sleep, damped)."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _spawn_rack(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the hanger rack: heavy DYNAMIC compound. Local frame: origin at the
    slab centre on the ground, slot along +x, open mouth at +x. Rail tops (and the
    column top, flush) form the hang plane at z = rail_top; the rail inner faces
    at y = +/- half_gap form the neck slot; the column face at x = -0.13 is the
    end stop. Rails, flares and column are bound to a SLICK min-combine material
    (the cap slides on them); the slab keeps default friction (the rack must not
    skate on the ground)."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    _dyn_body(root, cfg.rack_mass, 0.5, 0.5)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    ht = c.rail_top  # 0.32: rail/column top plane

    _add_box(stage, f"{prim_path}/slab", center=(-0.02, 0.0, 0.010),
             size=(0.380, 0.240, 0.020), color=c.slab_color, collide=collide)
    slick_parts = [
        _add_box(stage, f"{prim_path}/column", center=(-0.15, 0.0, 0.020 + (ht - 0.020) / 2),
                 size=(0.040, 0.100, ht - 0.020), color=c.column_color, collide=collide),
        _add_box(stage, f"{prim_path}/rail_p", center=(0.0, 0.026, ht - 0.010),
                 size=(0.260, 0.020, 0.020), color=c.rail_color, collide=collide),
        _add_box(stage, f"{prim_path}/rail_n", center=(0.0, -0.026, ht - 0.010),
                 size=(0.260, 0.020, 0.020), color=c.rail_color, collide=collide),
        # flare beams: continue the rail inner faces outward at +/-30 deg, tops flush
        _add_box(stage, f"{prim_path}/flare_p", center=(0.1488, 0.0384, ht - 0.010),
                 size=(0.055, 0.020, 0.020), color=c.rail_color, collide=collide,
                 yaw_deg=30.0),
        _add_box(stage, f"{prim_path}/flare_n", center=(0.1488, -0.0384, ht - 0.010),
                 size=(0.055, 0.020, 0.020), color=c.rail_color, collide=collide,
                 yaw_deg=-30.0),
    ]
    slick = _mk_material(prim_path, "slick", c.slick_mu, c.slick_mu, "min")
    for prim in slick_parts:
        bind_physics_material(str(prim.GetPath()), slick)
    return root


def _spawn_bottle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a hanger bottle: DYNAMIC compound, origin at the BODY centre.
    Children: body cylinder (r 28, h 140), neck (r 10, h 40, z 0.07..0.11), cap
    flange (r 26, h 12, z 0.11..0.122). The cap is bound SLICK (min-combine) so
    cap-on-rail contact is slick-on-slick; body and neck keep default friction so
    the bottle stands normally on the ground."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    _dyn_body(root, cfg.bottle_mass, 0.10, 0.30)
    collide = _make_collide(cfg.contact_offset)
    c = cfg

    _add_cyl(stage, f"{prim_path}/body", center=(0.0, 0.0, 0.0),
             radius=c.body_r, height=c.body_h, color=c.color, collide=collide)
    _add_cyl(stage, f"{prim_path}/neck", center=(0.0, 0.0, 0.090),
             radius=c.neck_r, height=c.neck_h, color=c.neck_color, collide=collide)
    cap = _add_cyl(stage, f"{prim_path}/cap", center=(0.0, 0.0, 0.116),
                   radius=c.cap_r, height=c.cap_h, color=c.cap_color, collide=collide)
    slick = _mk_material(prim_path, "slick", c.slick_mu, c.slick_mu, "min")
    bind_physics_material(str(cap.GetPath()), slick)
    return root


def _spawn_basket(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the decoy basket: DYNAMIC compound, origin at the floor centre on the
    ground. Open box: 160 mm square floor + four 90 mm walls."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _dyn_body(root, cfg.basket_mass, 0.10, 0.20)
    collide = _make_collide(cfg.contact_offset)
    c = cfg

    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, 0.004),
             size=(0.160, 0.160, 0.008), color=c.basket_color, collide=collide)
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/wall_x{tag}", center=(sgn * 0.076, 0.0, 0.053),
                 size=(0.008, 0.160, 0.090), color=c.basket_color, collide=collide)
        _add_box(stage, f"{prim_path}/wall_y{tag}", center=(0.0, sgn * 0.076, 0.053),
                 size=(0.144, 0.008, 0.090), color=c.basket_color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rack" not in _SPAWNER_CACHE:

        @configclass
        class RackSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rack)
            rack_mass: float = 25.0
            rail_top: float = 0.32
            slick_mu: float = 0.06
            slab_color: tuple = (0.35, 0.36, 0.40)
            column_color: tuple = (0.22, 0.24, 0.28)
            rail_color: tuple = (0.75, 0.76, 0.80)
            contact_offset: float = 0.002

        @configclass
        class BottleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bottle)
            bottle_mass: float = 0.25
            body_r: float = 0.028
            body_h: float = 0.140
            neck_r: float = 0.010
            neck_h: float = 0.040
            cap_r: float = 0.026
            cap_h: float = 0.012
            slick_mu: float = 0.06
            color: tuple = (0.90, 0.62, 0.12)
            neck_color: tuple = (0.85, 0.85, 0.85)
            cap_color: tuple = (0.92, 0.92, 0.95)
            contact_offset: float = 0.002

        @configclass
        class BasketSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_basket)
            basket_mass: float = 1.0
            basket_color: tuple = (0.52, 0.36, 0.18)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["rack"] = RackSpawnerCfg
        _SPAWNER_CACHE["bottle"] = BottleSpawnerCfg
        _SPAWNER_CACHE["basket"] = BasketSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class RailHangerSceneCfg(BaseCfg):
    """Config for `RailHangerScene`. The mechanism is honest by construction:
    neck dia 20 mm < rail gap 32 mm < cap dia 52 mm < body dia 56 mm, so a bottle
    inside the hang band can only be supported by its cap on the rails, and the
    seat (closed end) is reachable only by sliding along the slot from the mouth
    — the cap cannot pass down through the gap, the body cannot pass up, and the
    column blocks the closed end."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    hang_y_tol: float = tunable(0.014)     # root |y| off the slot centreline when hanging (m)
    hang_z: tuple = tunable((0.195, 0.222))  # root height band when hanging (rack-local, m)
    hook_x: tuple = tunable((-0.135, 0.135))  # slot span within which "hooked" counts
    seat_x: float = tunable(-0.095)        # hanging at x < this = seated (stop at -0.102)
    upright_deg: float = tunable(15.0)     # max bottle tilt from vertical when judged
    settle_speed: float = tunable(0.05)    # max |lin vel| of loose pieces when judging (m/s)
    slow_gate: float = tunable(0.08)       # credit latches only while the bottle is this slow
    still_steps: int = tunable(60)         # consecutive still substeps required (0.5 s)

    # --- tunable: randomization (the task-family knobs) -------------------------------------------
    rack_yaw_deg: float = tunable(20.0)    # rack yaw about its nominal heading (+/- deg)
    rack_jitter: float = tunable(0.03)     # rack xy jitter (+/- m)
    station_jitter: float = tunable(0.03)  # per-piece xy jitter at its station (+/- m)
    station_swap: bool = tunable(True)     # shuffle dressing/decoy start stations

    # --- info: layout (rack-local stations on the open ground) ------------------------------------
    rack_pos: tuple = info((0.45, 0.0))    # rack origin on the ground (nominal)
    rack_yaw_nom_deg: float = info(0.0)    # nominal heading (mouth toward world +x)
    st_bottle_1: tuple = info((0.30, -0.10))
    st_bottle_2: tuple = info((0.14, -0.30))
    st_basket: tuple = info((0.10, 0.30))
    # --- info: rack geometry (rack-local; see _spawn_rack) ----------------------------------------
    rail_top: float = info(0.32)           # rail/column top plane (the hang plane)
    half_gap: float = info(0.016)          # rail inner faces at y = +/- 0.016
    rail_x: tuple = info((-0.13, 0.13))    # rail span; mouth at +0.13, column face at -0.13
    flare_tip_x: float = info(0.178)       # outermost flare reach
    seat_x_nom: float = info(-0.102)       # root x with the bottle body against the column
    rack_mass: float = info(25.0)
    slick_mu: float = info(0.06)
    # --- info: bottle geometry (compound, origin at the body centre) ------------------------------
    body_r: float = info(0.028)
    body_h: float = info(0.140)
    neck_r: float = info(0.010)
    neck_h: float = info(0.040)            # neck local z 0.07..0.11
    cap_r: float = info(0.026)
    cap_h: float = info(0.012)             # cap local z 0.11..0.122; underside at +0.11
    cap_under: float = info(0.110)         # cap underside height above the body centre
    bottle_mass: float = info(0.25)
    hang_root_z: float = info(0.210)       # rail_top - cap_under: root height when hanging
    basket_mass: float = info(1.0)
    dressing_color: tuple = info((0.90, 0.62, 0.12))
    decoy_color: tuple = info((0.78, 0.10, 0.10))
    contact_offset: float = info(0.002)
    # --- info: rubric weights (0.25 + 0.35 = 0.60 = the non-success cap) --------------------------
    w_hook: float = info(0.25)
    w_seat: float = info(0.35)
    # decoy-exclusion volume (rack-local): the decoy must not be on/in the rack slot
    excl_y: float = info(0.05)
    excl_x: tuple = info((-0.17, 0.20))
    excl_z: tuple = info((0.15, 0.45))


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("rail_hanger")
class RailHangerScene(BaseScene):
    cfg: RailHangerSceneCfg

    def __init__(self, cfg: RailHangerSceneCfg | None = None) -> None:
        super().__init__(cfg or RailHangerSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        px, py = c.rack_pos
        yaw0 = math.radians(c.rack_yaw_nom_deg)
        qz0 = (math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2))

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
                spawn=cls["rack"](rack_mass=c.rack_mass, rail_top=c.rail_top,
                                  slick_mu=c.slick_mu, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0), rot=qz0),
            ),
            "basket": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Basket",
                spawn=cls["basket"](basket_mass=c.basket_mass,
                                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px + 0.10, py + 0.30, 0.001)),
            ),
        }
        for name, color, x0 in (("dressing", c.dressing_color, 0.75),
                                ("decoy", c.decoy_color, 0.59)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bottle_" + name,
                spawn=cls["bottle"](bottle_mass=c.bottle_mass, body_r=c.body_r,
                                    body_h=c.body_h, neck_r=c.neck_r, neck_h=c.neck_h,
                                    cap_r=c.cap_r, cap_h=c.cap_h, slick_mu=c.slick_mu,
                                    color=color, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(x0, py - 0.10, c.body_h / 2)),
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
        self.rack: RigidObject = env.iscene["rack"]
        self.dressing: RigidObject = env.iscene["dressing"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.basket: RigidObject = env.iscene["basket"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # dressing_station[e] = 1 / 2: which scatter station holds the DRESSING bottle
        self.dressing_station = torch.ones(n, dtype=torch.long, device=dev)
        # credit latches (survive transients; success is judged live)
        self._hooked = torch.zeros(n, dtype=torch.bool, device=dev)
        self._seated = torch.zeros(n, dtype=torch.bool, device=dev)
        # settle bookkeeping
        self._still = torch.zeros(n, dtype=torch.long, device=dev)
        self._steps = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the rack (nominal heading + yaw + xy jitter), stand
        the two bottles at their (possibly swapped) rack-local stations with jitter
        + free yaw, drop the basket at its station, clear all latches. Stations are
        fixed rack-local anchors spaced so pieces cannot overlap by construction."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        yaw = math.radians(c.rack_yaw_nom_deg) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rack_yaw_deg)
        q_r = _qz(yaw)
        pp = torch.zeros(m, 3, device=dev)
        pp[:, 0] = c.rack_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.rack_jitter
        pp[:, 1] = c.rack_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.rack_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pp + origin
        st[:, 3:7] = q_r
        self.rack.write_root_state_to_sim(st, env_ids)

        def place(station: tuple, z: float) -> torch.Tensor:
            """(m,13) upright state at a rack-local station with jitter + free yaw."""
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = station[0] + (torch.rand(m, device=dev) * 2 - 1) * c.station_jitter
            loc[:, 1] = station[1] + (torch.rand(m, device=dev) * 2 - 1) * c.station_jitter
            out = torch.zeros(m, 13, device=dev)
            out[:, 0:3] = pp + _qapply(q_r, loc) + origin
            out[:, 2] = z + origin[:, 2]
            out[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi)
            return out

        if c.station_swap:
            stn = torch.where(torch.rand(m, device=dev) < 0.5,
                              torch.ones(m, dtype=torch.long, device=dev),
                              torch.full((m,), 2, dtype=torch.long, device=dev))
        else:
            stn = torch.ones(m, dtype=torch.long, device=dev)
        self.dressing_station[env_ids] = stn
        s1 = torch.tensor(c.st_bottle_1, device=dev)
        s2 = torch.tensor(c.st_bottle_2, device=dev)
        for body, own in ((self.dressing, stn == 1), (self.decoy, stn == 2)):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0:2] = torch.where(own.unsqueeze(-1), s1, s2)
            loc[:, 0] += (torch.rand(m, device=dev) * 2 - 1) * c.station_jitter
            loc[:, 1] += (torch.rand(m, device=dev) * 2 - 1) * c.station_jitter
            stt = torch.zeros(m, 13, device=dev)
            stt[:, 0:3] = pp + _qapply(q_r, loc) + origin
            stt[:, 2] = c.body_h / 2 + 0.002 + origin[:, 2]
            stt[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi)
            body.write_root_state_to_sim(stt, env_ids)

        self.basket.write_root_state_to_sim(place(c.st_basket, 0.002), env_ids)

        self._hooked[env_ids] = False
        self._seated[env_ids] = False
        self._still[env_ids] = 0
        self._steps[env_ids] = 0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "rack": self.rack.data.root_state_w[env_ids].clone(),
            "dressing": self.dressing.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "basket": self.basket.data.root_state_w[env_ids].clone(),
            "dressing_station": self.dressing_station[env_ids].clone(),
            "hooked": self._hooked[env_ids].clone(),
            "seated": self._seated[env_ids].clone(),
            "still": self._still[env_ids].clone(),
            "steps": self._steps[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.rack.write_root_state_to_sim(state["rack"], env_ids)
        self.dressing.write_root_state_to_sim(state["dressing"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.basket.write_root_state_to_sim(state["basket"], env_ids)
        self.dressing_station[env_ids] = state["dressing_station"]
        self._hooked[env_ids] = state["hooked"]
        self._seated[env_ids] = state["seated"]
        self._still[env_ids] = state["still"]
        self._steps[env_ids] = state["steps"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A HANGER RACK stands on the ground: a grey base slab, a dark column at "
            f"one end, and two parallel silver RAILS cantilevered horizontally from "
            f"the column top at {c.rail_top * 1000:.0f} mm height. The rails form a "
            f"narrow SLOT ({2 * c.half_gap * 1000:.0f} mm wide) that is OPEN at the "
            f"far end — its entrance flared into a funnel mouth — and CLOSED where "
            f"the rails meet the column. Nearby, TWO bottles stand upright on the "
            f"ground, identical in shape (body {2 * c.body_r * 1000:.0f} mm wide, "
            f"{c.body_h * 1000:.0f} mm tall, with a narrow {2 * c.neck_r * 1000:.0f} mm "
            f"neck and a wide {2 * c.cap_r * 1000:.0f} mm white CAP flange on top): "
            f"one AMBER (the salad dressing — the target) and one RED (a decoy). "
            f"Which bottle stands where is shuffled per episode — the color is the "
            f"only cue. An open brown BASKET also sits on the ground; it is a decoy "
            f"receptacle and plays no part in the goal.\n"
            f"Goal: HANG the AMBER bottle from the rack, at the CLOSED end of the "
            f"slot. The neck fits between the rails and the cap flange does not, so "
            f"the bottle hangs from the rails by its cap. The only way in is through "
            f"the OPEN mouth: hold the bottle upright beside the open end with its "
            f"cap just above the rail tops and its neck level with the slot, thread "
            f"the neck in through the flared mouth, slide the bottle along the slot "
            f"until its body rests against the column, and release it there. Finish "
            f"with the amber bottle hanging freely at the closed end (body dangling "
            f"in mid-air, touching nothing but the rails), the RED bottle NOT on the "
            f"rack, and everything at rest. Dropping the bottle into the basket or "
            f"standing it anywhere on the ground scores nothing; a bottle hooked "
            f"into the slot but left near the open end is only partial credit."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Hang the amber salad-dressing bottle on the overhead rail rack: thread "
            "its neck into the open flared end of the slot between the two rails so "
            "the cap flange rides on the rail tops, slide it along the slot to the "
            "closed end against the column, and leave it hanging there at rest. Do "
            "not hang the red bottle, and do not use the basket."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _rack_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the rack frame, (N,3) -> (N,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.rack.data.root_quat_w,
                                  pos_w - self.rack.data.root_pos_w)

    def _up_z(self, body: RigidObject) -> torch.Tensor:
        """(N,) world-z component of the body's +z axis (1 = upright)."""
        n = body.data.root_quat_w.shape[0]
        ez = torch.tensor([[0.0, 0.0, 1.0]], device=self.env.device).expand(n, 3)
        return _qapply(body.data.root_quat_w, ez)[:, 2]

    def hanging(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool (live): `body` hanging from the rails — root on the slot
        centreline within `hang_y_tol`, height in the hang band (cap on the rail
        tops; the band excludes both standing anywhere and lying ON TOP of the
        rails), within the slot span, near-vertical. Geometry makes this state
        supportable only by the cap-on-rails contact."""
        c = self.cfg
        loc = self._rack_local(body.data.root_pos_w)
        return (loc[:, 1].abs() < c.hang_y_tol) \
            & (loc[:, 2] > c.hang_z[0]) & (loc[:, 2] < c.hang_z[1]) \
            & (loc[:, 0] > c.hook_x[0]) & (loc[:, 0] < c.hook_x[1]) \
            & (self._up_z(body) > math.cos(math.radians(c.upright_deg)))

    def seated(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool (live): hanging AT the closed end (neck at the column stop)."""
        return self.hanging(body) & \
            (self._rack_local(body.data.root_pos_w)[:, 0] < self.cfg.seat_x)

    def decoy_off_rack(self) -> torch.Tensor:
        """(N,) bool (live): the decoy is nowhere on/in the rack slot volume."""
        c = self.cfg
        loc = self._rack_local(self.decoy.data.root_pos_w)
        on_rack = (loc[:, 1].abs() < c.excl_y) \
            & (loc[:, 0] > c.excl_x[0]) & (loc[:, 0] < c.excl_x[1]) \
            & (loc[:, 2] > c.excl_z[0]) & (loc[:, 2] < c.excl_z[1])
        return ~on_rack

    def settled(self) -> torch.Tensor:
        """(N,) bool: consecutive-still counter satisfied (not an instantaneous
        velocity gate — teleports zero velocities; the counter runs in post_step)."""
        return (self._still >= self.cfg.still_steps) & (self._steps >= self.cfg.still_steps)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w for b in
                         (self.rack, self.dressing, self.decoy, self.basket)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        slow = self.dressing.data.root_lin_vel_w.norm(dim=-1) < c.slow_gate
        self._hooked |= self.hanging(self.dressing) & slow & fin
        self._seated |= self.seated(self.dressing) & slow & fin
        still = (self.dressing.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.decoy.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.basket.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.rack.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
        self._still = torch.where(still, self._still + 1, torch.zeros_like(self._still))
        self._steps = self._steps + 1

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the AMBER bottle hanging at the closed-end seat (live physical
        outcome: suspended by its cap on the rails, upright, at the column stop),
        the RED decoy not on the rack, everything settled and finite."""
        return self.seated(self.dressing) & self.decoy_off_rack() \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.25*hooked + 0.35*seated (latched; ~0 for doing
        nothing — both credits require the bottle lifted off the ground, threaded
        into the slot and slid along it), capped at 0.60 — and exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        base = (c.w_hook * self._hooked.float()
                + c.w_seat * self._seated.float()).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="rail_hanger", robot="null"))

"""HanoiRingsScene — move the three-ring tower to the GREEN post, never larger on smaller.

Derived from libero_90/libero_kitchen_scene1_open_drawer_put_bowl ("open the top drawer
of the cabinet and put the bowl in it"), but the PLAN is replaced wholesale. The seed's
strategy is: actuate one articulated container open (a prismatic drawer pull), then a
single pick-and-place of one object into the revealed containment region — two
independent subgoals, no ordering pressure beyond "open before insert", judged by one
final containment box. Here there is NO container, NO articulated fixture and NO
single placement: three RINGS of graduated size start threaded on one of three fixed
POSTS, and the goal is to rebuild the tower on the GREEN post under the classic Tower
of Hanoi rule — a ring may only be moved by lifting it clear over its post's tip, and
a LARGER ring must never come to rest on top of a SMALLER one (the scene watches every
settled configuration and latches a permanent violation). Solving requires a computed
7-move recursion through the spare post (small->target, mid->spare, small->spare,
large->target, small->source, mid->target, small->target); a solver needs multi-step
look-ahead state planning the seed task has no analogue of, and every single move is a
thread-over-post insertion executed by gravity, not a free placement.

Assets are fully procedural (compound-spawner pattern; child colliders of one body
never self-collide):
  - board: KINEMATIC slab the whole puzzle stands on (jittered pose per episode).
  - posts: three separate KINEMATIC bodies — square base pad + vertical shaft
    (r=6 mm, 85 mm) + guiding cone tip (20 mm). One pad is GREEN (the target), two
    are GRAY. Which post stands on which of the three board slots is a per-episode
    permutation (readback-verifiable).
  - rings: three DYNAMIC octagonal annuli (8 box colliders each) with a common
    28 mm-across-flats hole and graduated outer size/color: LARGE red (70 mm across
    flats), MID yellow (57 mm), SMALL white (44 mm). All start threaded on one gray
    post, largest at the bottom.

Rubric (0..1; latched partial credit anchored in the demonstrated solve trajectory):
  0.15 * first_move — the SMALL ring ever threaded on a post other than its start
                      post (the mandatory opening move; 0 for the null policy)
  0.35 * large_home — the LARGE ring ever seated at the BOTTOM of the green post,
                      settled (move 4 of the recursion)
  0.30 * mid_home   — large_home AND the MID ring threaded directly on top of it on
                      the green post, settled (move 6)
  1.0 iff success() — all three rings threaded on the green post, bottom-to-top
                      LARGE/MID/SMALL, seated and contiguous, everything at rest,
                      and the Hanoi rule never violated. Non-success capped at 0.80;
                      a latched violation caps the score at 0.15 forever.

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


def _add_box(stage, path: str, *, center, size, color, collide: Callable, orient=None):
    """One box child: translate [+ orient] + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _add_cyl(stage, path: str, *, center, radius, height, color, collide: Callable):
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr("Z")
    r, h = float(radius), float(height)
    cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())
    return cyl.GetPrim()


def _add_cone(stage, path: str, *, center, radius, height, color, collide: Callable):
    from pxr import Gf, UsdGeom

    cone = UsdGeom.Cone.Define(stage, path)
    cone.CreateRadiusAttr(float(radius))
    cone.CreateHeightAttr(float(height))
    cone.CreateAxisAttr("Z")
    r, h = float(radius), float(height)
    cone.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
    xf = UsdGeom.Xformable(cone.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cone.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cone.GetPrim())
    return cone.GetPrim()


def _spawn_post(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one post at `prim_path`: KINEMATIC compound. Local frame: origin at the
    centre of the base pad's BOTTOM face. Children: colored square base pad, vertical
    shaft cylinder, guiding cone tip."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_box(stage, f"{prim_path}/pad", center=(0.0, 0.0, c.pad_t / 2),
             size=(c.pad_w, c.pad_w, c.pad_t), color=c.pad_color, collide=collide)
    _add_cyl(stage, f"{prim_path}/shaft",
             center=(0.0, 0.0, c.pad_t + c.shaft_h / 2),
             radius=c.shaft_r, height=c.shaft_h, color=c.post_color, collide=collide)
    _add_cone(stage, f"{prim_path}/tip",
              center=(0.0, 0.0, c.pad_t + c.shaft_h + c.cone_h / 2),
              radius=c.shaft_r, height=c.cone_h, color=c.post_color, collide=collide)
    return root


def _spawn_ring(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one ring at `prim_path`: DYNAMIC octagonal annulus of 8 box colliders.
    Local frame: origin at the ring centre (== CoM by symmetry, so the authored
    MassAPI mass is honest). Inner faces form an octagon with apothem `r_in`
    (the hole), outer flats sit at apothem `r_out` (the parallel-jaw grip faces)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(1.0)
    px.CreateAngularDampingAttr(0.8)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    t = c.r_out - c.r_in                        # radial wall thickness
    rc = (c.r_out + c.r_in) / 2                 # wall centreline radius
    w = 2 * c.r_out * math.tan(math.pi / 8) + 0.002   # tangential width (overlap-closed)
    for i in range(8):
        ang = i * math.pi / 4
        q = (math.cos(ang / 2), 0.0, 0.0, math.sin(ang / 2))
        _add_box(stage, f"{prim_path}/seg_{i}",
                 center=(rc * math.cos(ang), rc * math.sin(ang), 0.0),
                 size=(t, w, c.height), color=c.color, collide=collide, orient=q)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "post" not in _SPAWNER_CACHE:

        @configclass
        class PostSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_post)
            pad_w: float = 0.09
            pad_t: float = 0.008
            shaft_r: float = 0.006
            shaft_h: float = 0.085
            cone_h: float = 0.020
            pad_color: tuple = (0.45, 0.45, 0.45)
            post_color: tuple = (0.35, 0.35, 0.38)
            contact_offset: float = 0.002

        @configclass
        class RingSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_ring)
            r_out: float = 0.035
            r_in: float = 0.014
            height: float = 0.016
            mass: float = 0.10
            color: tuple = (0.85, 0.10, 0.10)
            contact_offset: float = 0.004

        _SPAWNER_CACHE.update(post=PostSpawnerCfg, ring=RingSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class HanoiRingsSceneCfg(BaseCfg):
    """Config for `HanoiRingsScene`. The threaded gate is honest by construction:
    a ring resting threaded on a post can be off-axis by at most
    hole apothem - shaft radius = 8 mm < the 12 mm gate, while a ring resting
    anywhere NOT threaded (leaning on the shaft, lying on the pad beside it) has its
    centre >= outer apothem + shaft radius >= 28 mm from the axis."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    thread_xy_tol: float = tunable(0.012)  # post-frame |xy| gate for "threaded" (rest bound 8 mm)
    thread_z_lo: float = tunable(0.010)    # post-frame z band for "threaded": above the pad ...
    thread_z_hi: float = tunable(0.095)    # ... and below the shaft top (a carried ring is out)
    seat_z_max: float = tunable(0.028)     # bottom ring centre below this = seated on the pad
    gap_max: float = tunable(0.027)        # max centre-to-centre z gap between stacked rings (nom 16)
    flat_max_deg: float = tunable(30.0)    # ring axis within this of world-up to count as threaded
    settle_speed: float = tunable(0.05)    # max |lin vel| of every ring when judging (m/s)
    viol_speed: float = tunable(0.08)      # the offender must be at most this fast (resting, not flying)
    viol_gap: float = tunable(0.040)       # offender within this z of the smaller ring (resting ON it)
    viol_steps: int = tunable(12)          # consecutive substeps before the violation latches

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    board_jitter: float = tunable(0.04)    # board xy jitter (+/- m)
    board_yaw_deg: float = tunable(30.0)   # board yaw about nominal (+/- deg)
    slot_shuffle: bool = tunable(True)     # permute the three posts over the three board slots
    stack_jitter: float = tunable(0.002)   # per-ring xy jitter when spawning the start tower (+/- m)

    # --- info: layout ---------------------------------------------------------------------------
    board_pos: tuple = info((0.36, 0.0))   # board centre on the ground (nominal)
    board_size: tuple = info((0.62, 0.22, 0.03))
    slot_dx: float = info(0.15)            # slots at board-local x = -dx, 0, +dx
    # --- info: post structure (local frame: origin at the pad bottom centre) --------------------
    pad_w: float = info(0.09)
    pad_t: float = info(0.008)
    shaft_r: float = info(0.006)
    shaft_h: float = info(0.085)
    cone_h: float = info(0.020)
    z_release: float = info(0.129)         # free-space release point over the cone apex (post local)
    # --- info: rings (index order LARGE, MID, SMALL) --------------------------------------------
    hole_r: float = info(0.014)            # hole apothem (all rings share it — classic Hanoi)
    ring_r_out: tuple = info((0.035, 0.0285, 0.022))
    ring_h: float = info(0.016)
    ring_mass: tuple = info((0.10, 0.08, 0.06))
    ring_colors: tuple = info(((0.85, 0.10, 0.10), (0.92, 0.78, 0.10), (0.92, 0.92, 0.88)))
    green_pad: tuple = info((0.10, 0.62, 0.12))
    gray_pad: tuple = info((0.45, 0.45, 0.45))
    board_color: tuple = info((0.32, 0.24, 0.16))
    contact_offset: float = info(0.004)
    # rubric weights (0.15 + 0.35 + 0.30 = 0.80 = the non-success cap)
    w_move1: float = info(0.15)
    w_large: float = info(0.35)
    w_mid: float = info(0.30)
    viol_cap: float = info(0.15)           # a latched Hanoi violation caps the score here


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


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("hanoi_rings")
class HanoiRingsScene(BaseScene):
    cfg: HanoiRingsSceneCfg

    RING_NAMES = ("ring_large", "ring_mid", "ring_small")   # size order, big -> small
    POST_NAMES = ("post_green", "post_gray0", "post_gray1")  # green = target
    TARGET = 0        # index into POST_NAMES
    SOURCE = 1        # the start tower always spawns on post_gray0 (its SLOT varies)

    def __init__(self, cfg: HanoiRingsSceneCfg | None = None) -> None:
        super().__init__(cfg or HanoiRingsSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
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
            "board": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Board",
                spawn=sim_utils.CuboidCfg(
                    size=c.board_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.board_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.board_pos[0], c.board_pos[1], c.board_size[2] / 2)),
            ),
        }
        for i, name in enumerate(self.POST_NAMES):
            pad = c.green_pad if i == self.TARGET else c.gray_pad
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Post_" + name,
                spawn=cls["post"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    pad_w=c.pad_w, pad_t=c.pad_t, shaft_r=c.shaft_r,
                    shaft_h=c.shaft_h, cone_h=c.cone_h, pad_color=pad,
                    contact_offset=0.002),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.board_pos[0] + (i - 1) * c.slot_dx, c.board_pos[1],
                         c.board_size[2])),
            )
        for i, name in enumerate(self.RING_NAMES):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ring_" + name,
                spawn=cls["ring"](
                    r_out=c.ring_r_out[i], r_in=c.hole_r, height=c.ring_h,
                    mass=c.ring_mass[i], color=c.ring_colors[i],
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.board_pos[0], c.board_pos[1],
                         c.board_size[2] + c.pad_t + (i + 0.5) * c.ring_h + 0.05)),
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
        self.board: RigidObject = env.iscene["board"]
        self.posts: dict[str, RigidObject] = {n: env.iscene[n] for n in self.POST_NAMES}
        self.rings: dict[str, RigidObject] = {n: env.iscene[n] for n in self.RING_NAMES}
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self.slot_of_post = torch.zeros(n, 3, dtype=torch.long, device=dev)
        # latches (partial credit + the Hanoi rule; success is judged live except ~violated)
        self._m1 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._lg = torch.zeros(n, dtype=torch.bool, device=dev)
        self._md = torch.zeros(n, dtype=torch.bool, device=dev)
        self._viol = torch.zeros(n, dtype=torch.bool, device=dev)
        self._viol_ctr = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the board (xy jitter + yaw), permute the three posts
        over the three board slots, spawn the ring tower threaded on `post_gray0`
        (largest at the bottom, per-ring yaw + xy jitter), clear all latches."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- board: kinematic, yaw + xy jitter ---
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.board_yaw_deg)
        q_board = _qz(yaw)
        bp = torch.zeros(m, 3, device=dev)
        bp[:, 0] = c.board_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.board_jitter
        bp[:, 1] = c.board_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.board_jitter
        bp[:, 2] = c.board_size[2] / 2
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = bp + origin
        st[:, 3:7] = q_board
        self.board.write_root_state_to_sim(st, env_ids)

        # --- posts: permute over the slots (board-local x = -dx, 0, +dx) ---
        if c.slot_shuffle:
            perm = torch.rand(m, 3, device=dev).argsort(dim=1)
        else:
            perm = torch.arange(3, device=dev).expand(m, 3).clone()
        self.slot_of_post[env_ids] = perm
        top = torch.zeros(m, 3, device=dev)
        top[:, 2] = c.board_size[2] / 2  # board centre -> board top
        post_base = {}
        for i, name in enumerate(self.POST_NAMES):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = (perm[:, i].float() - 1.0) * c.slot_dx
            pos = bp + top + quat_apply(q_board, loc)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = q_board
            self.posts[name].write_root_state_to_sim(st, env_ids)
            post_base[name] = pos

        # --- rings: threaded tower on post_gray0, largest at the bottom ---
        src = post_base[self.POST_NAMES[self.SOURCE]]
        for k, name in enumerate(self.RING_NAMES):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = src[:, 0:2] \
                + (torch.rand(m, 2, device=dev) * 2 - 1) * c.stack_jitter
            st[:, 2] = src[:, 2] + c.pad_t + (k + 0.5) * c.ring_h + 0.002 * (k + 1)
            st[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi)
            st[:, 0:3] += origin
            self.rings[name].write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._m1[env_ids] = False
        self._lg[env_ids] = False
        self._md[env_ids] = False
        self._viol[env_ids] = False
        self._viol_ctr[env_ids] = 0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "board": self.board.data.root_state_w[env_ids].clone(),
            "posts": {n: b.data.root_state_w[env_ids].clone() for n, b in self.posts.items()},
            "rings": {n: b.data.root_state_w[env_ids].clone() for n, b in self.rings.items()},
            "slot_of_post": self.slot_of_post[env_ids].clone(),
            "m1": self._m1[env_ids].clone(), "lg": self._lg[env_ids].clone(),
            "md": self._md[env_ids].clone(), "viol": self._viol[env_ids].clone(),
            "viol_ctr": self._viol_ctr[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.board.write_root_state_to_sim(state["board"], env_ids)
        for n, b in self.posts.items():
            b.write_root_state_to_sim(state["posts"][n], env_ids)
        for n, b in self.rings.items():
            b.write_root_state_to_sim(state["rings"][n], env_ids)
        self.slot_of_post[env_ids] = state["slot_of_post"]
        self._m1[env_ids] = state["m1"]
        self._lg[env_ids] = state["lg"]
        self._md[env_ids] = state["md"]
        self._viol[env_ids] = state["viol"]
        self._viol_ctr[env_ids] = state["viol_ctr"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        d = [int(2 * r * 1000) for r in c.ring_r_out]
        return (
            f"A wooden board on the ground carries THREE upright POSTS in a row (slender "
            f"{2 * c.shaft_r * 1000:.0f} mm shafts, {(c.shaft_h + c.cone_h) * 1000:.0f} mm "
            f"tall, each with a pointed cone tip and a square base pad, "
            f"{c.slot_dx * 1000:.0f} mm apart). Exactly one base pad is GREEN — that post "
            f"is the TARGET. The other two pads are gray. Which post stands where, and the "
            f"board's position and heading, change every episode — look at the colors.\n"
            f"Threaded on one gray post stands a tower of THREE RINGS, largest at the "
            f"bottom: a RED ring ({d[0]} mm across), a YELLOW ring ({d[1]} mm) and a WHITE "
            f"ring ({d[2]} mm) on top. All rings share the same "
            f"{2 * c.hole_r * 1000:.0f} mm hole: any ring fits on any post, and a ring can "
            f"only leave or join a post by passing vertically over that post's cone tip.\n"
            f"Goal: rebuild the whole tower on the GREEN post — red at the bottom, then "
            f"yellow, then white — with every ring threaded on the post and the stack at "
            f"rest. RULE (Tower of Hanoi): move one ring at a time, and NEVER let a larger "
            f"ring come to rest on top of a smaller ring, on any post, at any moment. The "
            f"scene watches continuously; one settled larger-on-smaller configuration "
            f"permanently voids the episode even if the final tower looks right. With one "
            f"spare gray post this forces the classic 7-move recursion: white to the green "
            f"post, yellow to the spare, white onto yellow, red to the green post, white "
            f"back to the start post, yellow onto red, white on top."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Move the tower of three rings, one ring at a time, from its gray post to "
            "the post with the GREEN base pad, using the spare gray post as needed: "
            "finish with red at the bottom, yellow in the middle, white on top. Never "
            "rest a larger ring on a smaller one — that voids the task."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _ring_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(pos_w (N,3,3), quat (N,3,4), |lin_vel| (N,3)) for all rings, size order."""
        pos = torch.stack([b.data.root_pos_w for b in self.rings.values()], dim=1)
        quat = torch.stack([b.data.root_quat_w for b in self.rings.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.rings.values()], dim=1)
        return pos, quat, vel

    def _post_local(self) -> torch.Tensor:
        """(N, R=3, P=3, 3): every ring position in every post's local frame."""
        from isaaclab.utils.math import quat_apply_inverse

        pos, _q, _v = self._ring_tensors()               # (N,3,3)
        n = pos.shape[0]
        pp = torch.stack([b.data.root_pos_w for b in self.posts.values()], dim=1)   # (N,3,3)
        pq = torch.stack([b.data.root_quat_w for b in self.posts.values()], dim=1)  # (N,3,4)
        rel = pos[:, :, None, :] - pp[:, None, :, :]     # (N,R,P,3)
        q = pq[:, None, :, :].expand(n, 3, 3, 4)
        return quat_apply_inverse(q.reshape(-1, 4), rel.reshape(-1, 3)).reshape(n, 3, 3, 3)

    def flat(self) -> torch.Tensor:
        """(N, 3) bool: ring axis within `flat_max_deg` of world-up."""
        from isaaclab.utils.math import quat_apply

        _p, quat, _v = self._ring_tensors()
        n = quat.shape[0]
        ez = torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(n * 3, 3)
        up = quat_apply(quat.reshape(n * 3, 4), ez).reshape(n, 3, 3)
        return up[:, :, 2].abs() >= math.cos(math.radians(self.cfg.flat_max_deg))

    def threaded(self) -> torch.Tensor:
        """(N, R=3, P=3) bool, geometric: ring hole around post shaft — post-frame
        |xy| inside the annular-clearance gate, z between the pad and the shaft top,
        ring lying flat. Honest by construction (see cfg docstring)."""
        c = self.cfg
        loc = self._post_local()
        near = loc[:, :, :, 0:2].norm(dim=-1) < c.thread_xy_tol
        band = (loc[:, :, :, 2] > c.thread_z_lo) & (loc[:, :, :, 2] < c.thread_z_hi)
        return near & band & self.flat()[:, :, None]

    def ring_post_z(self) -> torch.Tensor:
        """(N, R=3, P=3): ring z in each post frame (for order/gap/seat checks)."""
        return self._post_local()[:, :, :, 2]

    def release_point_w(self, post_name: str) -> torch.Tensor:
        """(N, 3) world free-space release point above `post_name`'s cone apex."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        off = torch.tensor([0.0, 0.0, self.cfg.z_release],
                           device=self.env.device).expand(n, 3)
        p = self.posts[post_name]
        return p.data.root_pos_w + quat_apply(p.data.root_quat_w, off)

    # ----- rubric stages -------------------------------------------------------------------------
    def _stages_now(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(m1, lg, md) live. m1: SMALL threaded on any post but its start post.
        lg: LARGE seated at the bottom of the green post, settled. md: lg AND MID
        threaded directly on top of the large, settled."""
        c = self.cfg
        th = self.threaded()
        z = self.ring_post_z()
        _p, _q, vel = self._ring_tensors()
        tgt = self.TARGET
        m1 = th[:, 2, tgt] | th[:, 2, 2]                     # small on green or spare gray1
        lg = th[:, 0, tgt] & (z[:, 0, tgt] < c.seat_z_max) & (vel[:, 0] < c.settle_speed)
        md = lg & th[:, 1, tgt] & (z[:, 1, tgt] > z[:, 0, tgt]) \
            & ((z[:, 1, tgt] - z[:, 0, tgt]) < c.gap_max) & (vel[:, 1] < c.settle_speed)
        return m1, lg, md

    def _violation_now(self) -> torch.Tensor:
        """(N,) bool: some LARGER ring currently rests on top of a SMALLER one —
        both threaded on the same post, larger above and within `viol_gap`, larger
        near rest. Latching happens in post_step after `viol_steps` consecutive hits
        (the turning-point/stillness pattern: never latch a transient)."""
        c = self.cfg
        th = self.threaded()
        z = self.ring_post_z()
        _p, _q, vel = self._ring_tensors()
        out = torch.zeros(th.shape[0], dtype=torch.bool, device=th.device)
        for a in range(3):           # a = bigger ring index (size order big->small)
            for b in range(a + 1, 3):
                both = th[:, a, :] & th[:, b, :]                     # same post (N,P)
                above = (z[:, a, :] > z[:, b, :]) \
                    & ((z[:, a, :] - z[:, b, :]) < c.viol_gap)
                out |= (both & above).any(dim=1) & (vel[:, a] < c.viol_speed)
        return out

    def _update_latches(self) -> None:
        m1, lg, md = self._stages_now()
        self._m1 |= m1
        self._lg |= lg
        self._md |= md
        v = self._violation_now()
        self._viol_ctr = torch.where(v, self._viol_ctr + 1,
                                     torch.zeros_like(self._viol_ctr))
        self._viol |= self._viol_ctr >= self.cfg.viol_steps

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: all three rings threaded on the GREEN post, bottom-to-top
        LARGE/MID/SMALL (strict z order, seated, contiguous), everything at rest and
        finite — and the Hanoi rule was never violated (latched watchdog)."""
        c = self.cfg
        self._update_latches()
        th = self.threaded()
        z = self.ring_post_z()
        pos, _q, vel = self._ring_tensors()
        tgt = self.TARGET
        all_on = th[:, :, tgt].all(dim=1)
        zt = z[:, :, tgt]
        ordered = (zt[:, 0] < zt[:, 1]) & (zt[:, 1] < zt[:, 2])
        seated = zt[:, 0] < c.seat_z_max
        contiguous = ((zt[:, 1] - zt[:, 0]) < c.gap_max) & ((zt[:, 2] - zt[:, 1]) < c.gap_max)
        still = (vel < c.settle_speed).all(dim=1)
        finite = torch.isfinite(pos).all(dim=-1).all(dim=-1)
        return all_on & ordered & seated & contiguous & still & finite & ~self._viol

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: latched 0.15*first_move + 0.35*large_home +
        0.30*mid_home (~0 for the null policy), capped at 0.80; a latched Hanoi
        violation caps everything at `viol_cap`; exactly 1.0 iff success() holds."""
        c = self.cfg
        self._update_latches()
        base = (c.w_move1 * self._m1.float() + c.w_large * self._lg.float()
                + c.w_mid * self._md.float()).clamp(max=0.80)
        base = torch.where(self._viol, base.clamp(max=c.viol_cap), base)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="hanoi_rings", robot="null"))

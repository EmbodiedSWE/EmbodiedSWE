"""HanoiPlatesScene — relay a graduated three-plate tower to the marked post, Hanoi rules.

Derived from libero_90/kitchen_scene2_put_the_black_bowl_at_the_back_on_the_plate ("put
the black bowl at the back on the plate": identify one bowl among three identical ones
and set it on a plate — a single unordered pick-and-place). The MANIPULATION MODEL is
replaced wholesale: nothing here is "put object A on fixture B once". A rack carries
three vertical posts; a tower of three square ring-plates (graduated sizes, distinct
colors) starts threaded on one post, and the goal is that tower rebuilt on the post
marked by a green pad — under the classic Tower-of-Hanoi contract:
  - move ONE plate at a time (never two plates off the posts at once);
  - a plate may only be put down THREADED ON A POST — never parked on the rack, the
    ground, or anywhere else;
  - never rest a larger plate on a smaller one.
Because a larger plate can never sit above a smaller one and plates cannot be parked,
the tower cannot be carried over as a stack and cannot be shuttled through the table:
the only way is the 7-move Hanoi recursion through the spare post. The solver's plan is
a constraint-governed SEQUENCE with a buffer, not a placement: a different plan and a
different code structure from the seed's single pick-and-place. The rules are monitored
over the whole trajectory and LATCH: any violation caps the episode at score 0 and
success stays False forever, even if the final arrangement looks perfect (the smoke
battery constructs exactly that and proves rejection).

Assets are fully procedural (compound-spawner pattern; child colliders of one body
never self-collide):
  - rack: KINEMATIC base plank (460 x 200 x 30 mm) with three steel posts (r 7 mm,
    115 mm tall) in a row, 150 mm apart;
  - marker: KINEMATIC green pad (130 x 130 x 4 mm) placed around the TARGET post's
    foot at reset (it identifies the goal post visually; plates land on it);
  - plates: three DYNAMIC square ring-plates ("picture frames" of 4 boxes, 16 mm
    thick) with a raised square boss ring (12 mm) on top. The boss spaces stacked
    plates apart (finger clearance) and carries the next plate. Sizes/colors:
    RED 100 mm (largest, 40 mm aperture), YELLOW 80 mm (36 mm), BLUE 60 mm (32 mm).
    Post-in-aperture slack is 13/11/9 mm — loose, drop-threadable, arm-friendly.

Per-episode randomization (readback-verifiable): rack xy jitter + yaw, the START post
(where the tower stands) and the TARGET post (where the green pad lies) sampled over
all 6 ordered pairs.

Rubric (0..1; latched partial credit anchored in the demonstrated 7-move solve):
  0.15 * s1 — any plate ever threaded, at rest, on the TARGET post (move 1; 0 for null)
  0.35 * s2 — the RED (largest) plate ever seated at the BOTTOM of the target post
              (move 4 — the move the whole recursion exists to enable)
  0.25 * s3 — RED seated and YELLOW threaded above it on the target post (move 6)
  1.0 iff success() — all three plates threaded on the target post in size order
              (red under yellow under blue), settled, finite, and NO rule violation
              ever latched. Violation forces score 0 and success False, permanently.
Non-success is capped at 0.75.

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


def _add_box(stage, path: str, *, center, size, color, collide: Callable):
    """One box child: translate + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _add_cylinder(stage, path: str, *, center, radius, height, color, collide: Callable):
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


def _spawn_rack(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC compound: base plank + three posts in a row along local x.
    Local frame: origin at the plank footprint centre on the ground."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_box(stage, f"{prim_path}/plank", center=(0.0, 0.0, c.base_h / 2),
             size=(c.base_l, c.base_w, c.base_h), color=c.plank_color, collide=collide)
    for i, name in enumerate(("post_a", "post_b", "post_c")):
        _add_cylinder(stage, f"{prim_path}/{name}",
                      center=((i - 1) * c.post_dx, 0.0, c.base_h + c.post_h / 2),
                      radius=c.post_r, height=c.post_h,
                      color=c.post_color, collide=collide)
    return root


def _spawn_plate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC compound: a square ring-plate (picture frame of 4 boxes, thickness t)
    with a raised square boss ring (height boss_h, outer half boss) on top. The
    aperture (half-width aper) runs through frame AND boss so a post threads the whole
    body. Local frame: origin at the FRAME's centre (mid-thickness); MassAPI mass
    keeps the CoM there."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    rb.CreateSolverPositionIterationCountAttr(16)
    rb.CreateSolverVelocityIterationCountAttr(1)
    rb.CreateMaxDepenetrationVelocityAttr(0.5)
    rb.CreateLinearDampingAttr(0.10)
    rb.CreateAngularDampingAttr(0.50)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    a, h, t = c.outer, c.aper, c.t
    b, bh = c.boss, c.boss_h
    # frame: two full-width bars along y at x = +/-(a+h)/2, two short bars closing it
    for sgn in (1.0, -1.0):
        s = "p" if sgn > 0 else "n"
        _add_box(stage, f"{prim_path}/fx_{s}", center=(sgn * (a + h) / 2, 0.0, 0.0),
                 size=(a - h, 2 * a, t), color=c.color, collide=collide)
        _add_box(stage, f"{prim_path}/fy_{s}", center=(0.0, sgn * (a + h) / 2, 0.0),
                 size=(2 * h, a - h, t), color=c.color, collide=collide)
        # boss ring on top (spacer + carrier for the next plate)
        _add_box(stage, f"{prim_path}/bx_{s}",
                 center=(sgn * (b + h) / 2, 0.0, t / 2 + bh / 2),
                 size=(b - h, 2 * b, bh), color=c.color, collide=collide)
        _add_box(stage, f"{prim_path}/by_{s}",
                 center=(0.0, sgn * (b + h) / 2, t / 2 + bh / 2),
                 size=(2 * h, b - h, bh), color=c.color, collide=collide)
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
            base_l: float = 0.46
            base_w: float = 0.20
            base_h: float = 0.03
            post_dx: float = 0.15
            post_r: float = 0.007
            post_h: float = 0.115
            plank_color: tuple = (0.45, 0.33, 0.20)
            post_color: tuple = (0.55, 0.57, 0.60)
            contact_offset: float = 0.002

        @configclass
        class PlateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_plate)
            outer: float = 0.050
            aper: float = 0.020
            t: float = 0.016
            boss: float = 0.026
            boss_h: float = 0.012
            mass: float = 0.12
            color: tuple = (0.8, 0.2, 0.2)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["rack"] = RackSpawnerCfg
        _SPAWNER_CACHE["plate"] = PlateSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class HanoiPlatesSceneCfg(BaseCfg):
    """Config for `HanoiPlatesScene`. The threaded-tolerances are honest by
    construction: a plate physically threaded on a post can sit at most
    sqrt(2)*(aperture - post_r) ~= 18 mm off the post axis (largest plate), inside the
    30 mm gate, while a plate resting anywhere off a post is >= 45 mm from every
    axis or outside the z span."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    thread_xy_tol: float = tunable(0.030)   # plate centre within this of a post axis
    thread_z_slack: float = tunable(0.006)  # threaded z span: base top .. post top - this
    thread_tilt_max_deg: float = tunable(15.0)  # plate normal within this of vertical
    settle_speed: float = tunable(0.05)     # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.5)        # max |ang vel| when judging (rad/s)
    still_steps: int = tunable(12)          # consecutive still steps -> "at rest"
    multi_off_steps: int = tunable(10)      # >=2 plates off-post this long -> violation
    parked_steps: int = tunable(30)         # a plate at rest off-post this long -> violation
    warmup_steps: int = tunable(40)         # latch grace after reset (spawn settle)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    rack_jitter: float = tunable(0.03)      # rack xy jitter (+/- m)
    rack_yaw_deg: float = tunable(20.0)     # rack yaw about nominal (+/- deg)
    shuffle_posts: bool = tunable(True)     # sample start/target posts (demo sets False)

    # --- info: layout (nominal; rack long axis nominally along world y) --------------------------
    rack_pos: tuple = info((0.42, 0.0))
    rack_yaw_nom_deg: float = info(90.0)
    # --- info: rack structure --------------------------------------------------------------------
    base_l: float = info(0.46)
    base_w: float = info(0.20)
    base_h: float = info(0.03)
    post_dx: float = info(0.15)
    post_r: float = info(0.007)
    post_h: float = info(0.115)
    # --- info: marker pad ------------------------------------------------------------------------
    pad_s: float = info(0.13)
    pad_t: float = info(0.004)
    pad_color: tuple = info((0.10, 0.70, 0.20))
    # --- info: plates (largest -> smallest; outer/aperture HALF-widths) --------------------------
    plate_t: float = info(0.016)
    boss_h: float = info(0.012)
    boss_half: float = info(0.026)
    plate_outer: tuple = info((0.050, 0.040, 0.030))
    plate_aper: tuple = info((0.020, 0.018, 0.016))
    plate_mass: float = info(0.12)
    plate_colors: tuple = info(((0.85, 0.15, 0.15), (0.90, 0.75, 0.10), (0.15, 0.35, 0.90)))
    plate_names: tuple = info(("red", "yellow", "blue"))
    contact_offset: float = info(0.002)
    # rubric weights (0.15 + 0.35 + 0.25 = 0.75 = the non-success cap)
    w_first: float = info(0.15)
    w_large: float = info(0.35)
    w_mid: float = info(0.25)


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("hanoi_plates")
class HanoiPlatesScene(BaseScene):
    cfg: HanoiPlatesSceneCfg

    def __init__(self, cfg: HanoiPlatesSceneCfg | None = None) -> None:
        super().__init__(cfg or HanoiPlatesSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        rack_spawn = cls["rack"](
            base_l=c.base_l, base_w=c.base_w, base_h=c.base_h, post_dx=c.post_dx,
            post_r=c.post_r, post_h=c.post_h, contact_offset=c.contact_offset)

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
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rack_pos[0], c.rack_pos[1], 0.0)),
            ),
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad",
                spawn=sim_utils.CuboidCfg(
                    size=(c.pad_s, c.pad_s, c.pad_t),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pad_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 1.0, c.pad_t / 2)),
            ),
        }
        for i, name in enumerate(c.plate_names):
            out[f"plate_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plate_" + name,
                spawn=cls["plate"](
                    outer=c.plate_outer[i], aper=c.plate_aper[i], t=c.plate_t,
                    boss=c.boss_half, boss_h=c.boss_h, mass=c.plate_mass,
                    color=c.plate_colors[i], contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.3, 1.0 + 0.2 * i, 0.05)),
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
        c = self.cfg
        self.rack: RigidObject = env.iscene["rack"]
        self.pad: RigidObject = env.iscene["pad"]
        self.plates: list[RigidObject] = [
            env.iscene[f"plate_{n}"] for n in c.plate_names]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self.start_post = torch.zeros(n, dtype=torch.long, device=dev)
        self.target_post = torch.full((n,), 2, dtype=torch.long, device=dev)
        # post local xy offsets, rack frame (3, 2)
        self._post_xy = torch.tensor(
            [[-c.post_dx, 0.0], [0.0, 0.0], [c.post_dx, 0.0]], device=dev)
        # trajectory monitors
        self._warmup = torch.zeros(n, dtype=torch.long, device=dev)
        self._still = torch.zeros(n, 3, dtype=torch.long, device=dev)
        self._multi = torch.zeros(n, dtype=torch.long, device=dev)
        self._parked = torch.zeros(n, 3, dtype=torch.long, device=dev)
        self._violated = torch.zeros(n, dtype=torch.bool, device=dev)
        # progress latches (the 7-move solve's stage boundaries)
        self._s1 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._s2 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._s3 = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the rack (xy jitter + yaw), sample START and TARGET
        posts, lay the green pad around the target post's foot, thread the tower
        (red/yellow/blue bottom-up) onto the start post, clear all monitors."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- rack: kinematic, nominal heading + yaw + xy jitter ---
        yaw = math.radians(c.rack_yaw_nom_deg) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rack_yaw_deg)
        q_rack = _qz(yaw)
        rp = torch.zeros(m, 3, device=dev)
        rp[:, 0] = c.rack_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.rack_jitter
        rp[:, 1] = c.rack_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.rack_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = rp + origin
        st[:, 3:7] = q_rack
        self.rack.write_root_state_to_sim(st, env_ids)

        # --- start / target posts (all 6 ordered pairs) ---
        if c.shuffle_posts:
            start = torch.randint(0, 3, (m,), device=dev)
            target = (start + torch.randint(1, 3, (m,), device=dev)) % 3
        else:
            start = torch.zeros(m, dtype=torch.long, device=dev)
            target = torch.full((m,), 2, dtype=torch.long, device=dev)
        self.start_post[env_ids] = start
        self.target_post[env_ids] = target

        def place(body, loc: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = rp + quat_apply(q_rack, loc) + origin
            st[:, 3:7] = q_rack
            body.write_root_state_to_sim(st, env_ids)

        # --- green pad around the target post's foot ---
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0:2] = self._post_xy[target]
        loc[:, 2] = c.base_h + c.pad_t / 2
        place(self.pad, loc)

        # --- tower on the start post: red (largest) at the bottom, 2 mm drop gaps ---
        level = c.plate_t + c.boss_h
        for i in range(3):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0:2] = self._post_xy[start]
            loc[:, 2] = c.base_h + c.pad_t + c.plate_t / 2 + i * level + 0.002 * (i + 1)
            place(self.plates[i], loc)

        # --- clear monitors ---
        self._warmup[env_ids] = c.warmup_steps
        self._still[env_ids] = 0
        self._multi[env_ids] = 0
        self._parked[env_ids] = 0
        self._violated[env_ids] = False
        self._s1[env_ids] = False
        self._s2[env_ids] = False
        self._s3[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "rack": self.rack.data.root_state_w[env_ids].clone(),
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "plates": [b.data.root_state_w[env_ids].clone() for b in self.plates],
            "start": self.start_post[env_ids].clone(),
            "target": self.target_post[env_ids].clone(),
            "warmup": self._warmup[env_ids].clone(),
            "still": self._still[env_ids].clone(),
            "multi": self._multi[env_ids].clone(),
            "parked": self._parked[env_ids].clone(),
            "violated": self._violated[env_ids].clone(),
            "s1": self._s1[env_ids].clone(),
            "s2": self._s2[env_ids].clone(),
            "s3": self._s3[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.rack.write_root_state_to_sim(state["rack"], env_ids)
        self.pad.write_root_state_to_sim(state["pad"], env_ids)
        for b, s in zip(self.plates, state["plates"]):
            b.write_root_state_to_sim(s, env_ids)
        self.start_post[env_ids] = state["start"]
        self.target_post[env_ids] = state["target"]
        self._warmup[env_ids] = state["warmup"]
        self._still[env_ids] = state["still"]
        self._multi[env_ids] = state["multi"]
        self._parked[env_ids] = state["parked"]
        self._violated[env_ids] = state["violated"]
        self._s1[env_ids] = state["s1"]
        self._s2[env_ids] = state["s2"]
        self._s3[env_ids] = state["s3"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wooden rack ({c.base_l * 1000:.0f} x {c.base_w * 1000:.0f} mm plank) "
            f"lies on the ground with three vertical steel posts in a row "
            f"({2 * c.post_r * 1000:.0f} mm thick, {c.post_h * 1000:.0f} mm tall, "
            f"{c.post_dx * 1000:.0f} mm apart). Three square ring-plates — flat frames "
            f"with a hole through the middle and a raised square collar on top — start "
            f"stacked as a tower on ONE post, biggest at the bottom: a RED plate "
            f"({2 * c.plate_outer[0] * 1000:.0f} mm wide), a YELLOW plate "
            f"({2 * c.plate_outer[1] * 1000:.0f} mm), and a BLUE plate "
            f"({2 * c.plate_outer[2] * 1000:.0f} mm) on top. A flat GREEN PAD lies "
            f"around the foot of another post: that is the TARGET post. Which post "
            f"holds the tower and which is marked change every episode, as do the "
            f"rack's position and heading — look for the tower and the green pad.\n"
            f"Goal: rebuild the whole tower on the marked post — red at the bottom, "
            f"then yellow, then blue, every plate threaded over the post — obeying "
            f"three rules that are monitored at every moment and are UNFORGIVING: "
            f"(1) move only ONE plate at a time; (2) a plate may only be set down "
            f"threaded on one of the three posts — never laid on the plank, the "
            f"ground, or anywhere else; (3) never rest a larger plate on top of a "
            f"smaller one, on any post. Breaking any rule at any time fails the "
            f"episode permanently, even if the tower ends up looking right. Because "
            f"of rule 3 you cannot carry the tower over in one piece, and because of "
            f"rule 2 you cannot lay plates aside: relay them through the spare post "
            f"(the classic 7-move sequence). Plates thread loosely (9-13 mm of slack "
            f"around the post) and their collars leave a "
            f"{c.boss_h * 1000:.0f} mm finger gap between stacked plates."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Move the three-plate tower from its post to the post marked by the green "
            "pad, using the spare post as a buffer. Move one plate at a time, set "
            "plates down only threaded on a post, and never rest a larger plate on a "
            "smaller one — breaking any rule fails the task. Finish with red at the "
            "bottom, yellow, then blue on the marked post."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _rack_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the (kinematic, live-read) rack frame, (N,3) -> (N,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.rack.data.root_quat_w,
                                  pos_w - self.rack.data.root_pos_w)

    def _plate_loc(self) -> torch.Tensor:
        """(N, 3, 3): plate centres in the rack frame."""
        return torch.stack([self._rack_local(b.data.root_pos_w) for b in self.plates],
                           dim=1)

    def _plate_upright(self) -> torch.Tensor:
        """(N, 3) bool: plate normal within `thread_tilt_max_deg` of world up."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        cos_max = math.cos(math.radians(self.cfg.thread_tilt_max_deg))
        ups = [quat_apply(b.data.root_quat_w, ez)[:, 2] for b in self.plates]
        return torch.stack(ups, dim=1).clamp(-1.0, 1.0) >= cos_max

    def threaded(self) -> torch.Tensor:
        """(N, 3 plates, 3 posts) bool, geometric: plate centre within `thread_xy_tol`
        of the post axis (rack frame), centre z inside the post's span (above the
        plank top, below the post tip — a plate balanced ON the tip is excluded),
        and the plate near-flat. Honest by construction: a threaded plate can sit at
        most ~18 mm off-axis; anything resting off the posts is >= 45 mm away or
        outside the z span."""
        c = self.cfg
        loc = self._plate_loc()                                   # (N, 3, 3)
        d = (loc[:, :, None, :2] - self._post_xy[None, None]).norm(dim=-1)  # (N,3,3)
        z = loc[:, :, 2]
        z_ok = (z > c.base_h - 0.005) & (z < c.base_h + c.post_h - c.thread_z_slack)
        return (d < c.thread_xy_tol) & z_ok[:, :, None] & self._plate_upright()[:, :, None]

    def _vels(self) -> tuple[torch.Tensor, torch.Tensor]:
        lin = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.plates], dim=1)
        ang = torch.stack([b.data.root_ang_vel_w.norm(dim=-1) for b in self.plates], dim=1)
        return lin, ang

    def at_rest(self) -> torch.Tensor:
        """(N, 3) bool: still for `still_steps` consecutive steps (latched counter —
        instantaneous velocity gates false-fire at swing turning points)."""
        return self._still >= self.cfg.still_steps

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w for b in self.plates], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _large_seated(self, thr: torch.Tensor) -> torch.Tensor:
        """(N,) bool: RED threaded on the target post, at rest, seated at the BOTTOM
        (centre within 30 mm of the plank top — one level up would be >= 40 mm)."""
        c = self.cfg
        n = self.env.num_envs
        ar = torch.arange(n, device=self.env.device)
        on_target = thr[ar, :, self.target_post]                  # (N, 3)
        z = self._plate_loc()[:, :, 2] - c.base_h                 # above plank top
        return on_target[:, 0] & self.at_rest()[:, 0] & (z[:, 0] < 0.030)

    # ----- trajectory monitors -------------------------------------------------------------------
    def _update_monitors(self) -> None:
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        ar = torch.arange(n, device=dev)

        lin, ang = self._vels()
        still_now = (lin < c.settle_speed) & (ang < c.settle_ang)
        self._still = torch.where(still_now, self._still + 1,
                                  torch.zeros_like(self._still))

        warm = self._warmup > 0
        self._warmup = (self._warmup - 1).clamp(min=0)

        thr = self.threaded()                                     # (N, 3, 3)
        on_any = thr.any(dim=-1)                                  # (N, 3)
        rest = self.at_rest()                                     # (N, 3)

        # rule 1: at most ONE plate off the posts at a time
        off = (~on_any).sum(dim=-1)
        self._multi = torch.where(off >= 2, self._multi + 1,
                                  torch.zeros_like(self._multi))
        # rule 2: a plate may not come to rest anywhere off the posts
        parked_now = (~on_any) & rest
        self._parked = torch.where(parked_now, self._parked + 1,
                                   torch.zeros_like(self._parked))
        # rule 3: a larger plate resting above a smaller one on the same post
        z = self._plate_loc()[:, :, 2]
        big_on_small = torch.zeros(n, dtype=torch.bool, device=dev)
        for i in range(3):            # i = larger
            for j in range(i + 1, 3):  # j = smaller
                same = (thr[:, i] & thr[:, j]).any(dim=-1)
                big_on_small |= same & (z[:, i] > z[:, j]) & rest[:, i] & rest[:, j]

        viol = (self._multi >= c.multi_off_steps) \
            | (self._parked >= c.parked_steps).any(dim=-1) | big_on_small
        self._violated |= viol & ~warm

        # progress latches (only while un-violated; frozen during warmup)
        live = ~warm & ~self._violated
        on_target = thr[ar, :, self.target_post]                  # (N, 3)
        self._s1 |= live & (on_target & rest).any(dim=-1)
        seated = self._large_seated(thr)
        self._s2 |= live & seated
        self._s3 |= live & seated & on_target[:, 1] & rest[:, 1]

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_monitors()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: all three plates threaded on the TARGET post in size order (red
        under yellow under blue), all at rest, finite, and NO rule violation ever
        latched since reset. The arrangement clauses are live physical outcomes; the
        rule clauses are trajectory facts."""
        self._update_monitors()
        n = self.env.num_envs
        ar = torch.arange(n, device=self.env.device)
        thr = self.threaded()
        on_target = thr[ar, :, self.target_post]                  # (N, 3)
        z = self._plate_loc()[:, :, 2]
        order = (z[:, 0] < z[:, 1]) & (z[:, 1] < z[:, 2])
        return on_target.all(dim=-1) & order & self.at_rest().all(dim=-1) \
            & self._finite() & ~self._violated & (self._warmup == 0)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*s1 + 0.35*s2 + 0.25*s3 (latched stage credit,
        ~0 for the null policy since every stage needs a plate on the TARGET post),
        capped at 0.75; a latched rule violation forces 0; exactly 1.0 iff success()
        holds live."""
        c = self.cfg
        self._update_monitors()
        base = (c.w_first * self._s1.float() + c.w_large * self._s2.float()
                + c.w_mid * self._s3.float()).clamp(max=0.75)
        base = torch.where(self._violated, torch.zeros_like(base), base)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="hanoi_plates", robot="null"))

"""TrayPackScene — jointly ARRANGE three rectangular blocks so all of them lie flat
on the floor of a walled tray at once (sim_gen task `native_tasks_i37`).

Derived from maniskill/native_tasks (the MetaSim-native ManiSkill single-primitive
tabletop suite: PickCube / PushCube / PullCube / StackCube / LiftPegUpright ... — each
task moves ONE free primitive to ONE independent free-space goal pose), but the
MANIPULATION MODEL is replaced wholesale. The seed's plan is: a single object, a
single goal site in open space, judged by a point distance — placements never
interact. Here NO block has a goal pose of its own and no single placement earns
success: the goal is a JOINT FEASIBILITY constraint. Three blocks with footprints
120x40, 80x80 and 80x40 mm must all lie flat ON THE FLOOR of a tray whose interior
is only 135x135 mm — the blocks tile 144 cm^2 of a 182 cm^2 floor, so they only all
fit in deliberate tilings (long bar along one wall, big slab square in a corner
beside it, small brick dropped into the one remaining pocket, up to symmetry). A
block placed thoughtlessly (e.g. the slab in the middle) makes the remaining blocks
IMPOSSIBLE to seat: physics (rigid non-overlap, the walls) enforces the constraint,
so the solver must plan an ARRANGEMENT and typically re-arrange, sliding blocks
flush against walls and against each other to open the last pocket. On top of that
the big slab starts INSIDE the tray STANDING ON ITS SIDE (scoring nothing) at a
random spot — a blocker that must first be laid flat and then packed like the rest.
Nothing in the seed suite ever couples one object's placement to another's
feasibility; that coupling IS this task.

Assets are fully procedural (the compound-spawner pattern — child colliders of one
body never self-collide):
  - tray: KINEMATIC open box on the ground — 20 mm floor plate + four 12 mm walls,
    50 mm tall, interior 135 x 135 mm. Dark grey body, amber wall tops.
  - blocks, DYNAMIC boxes, all 40 mm tall: RED BAR 120x40 mm, BLUE SLAB 80x80 mm,
    GREEN BRICK 80x40 mm.

Per-episode randomization (readback-verifiable): tray xy jitter + FREE yaw
(+/-180 deg); the blue slab's standing spot and heading inside the tray; the bar's
and brick's scatter poses (xy jitter + free yaw) on the ground outside.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  0.25 * bar_seated    — the red bar ever lies flat on the tray floor, fully inside
  0.25 * slab_seated   — the blue slab ever lies flat on the tray floor, fully inside
  0.25 * brick_seated  — the green brick ever lies flat on the tray floor, fully inside
  (each latched; ~0 for the null policy — the standing slab does NOT count)
  1.0 iff success()    — ALL THREE blocks simultaneously flat on the tray floor,
  fully inside the walls, at rest. Non-success is capped at 0.75.
"seated" is live-geometric: block bottom face at floor height (rejects resting on
another block, on the wall tops, or on the ground outside), block flat (rejects the
standing/leaning slab), all four bottom corners inside the interior footprint.

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


# ----- custom compound spawner ------------------------------------------------------------------
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


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the tray at `prim_path`: KINEMATIC compound. Local frame: origin at the
    centre of the floor plate on the ground; interior rises along local +z.

    Children: floor plate (top at z_floor) and four walls (wall_h tall above the
    floor top), interior inner_x x inner_y between them."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    ix, iy = c.inner_x / 2, c.inner_y / 2
    wt, wh, zf = c.wall_t, c.wall_h, c.z_floor
    zc = zf + wh / 2
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, zf / 2),
             size=(2 * ix + 2 * wt, 2 * iy + 2 * wt, zf),
             color=c.floor_color, collide=collide)
    for sgn in (1.0, -1.0):  # walls along y (+/-x sides), full y span
        _add_box(stage, f"{prim_path}/wall_x{'p' if sgn > 0 else 'n'}",
                 center=(sgn * (ix + wt / 2), 0.0, zc),
                 size=(wt, 2 * iy + 2 * wt, wh), color=c.wall_color, collide=collide)
        _add_box(stage, f"{prim_path}/wall_y{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * (iy + wt / 2), zc),
                 size=(2 * ix + 2 * wt, wt, wh), color=c.wall_color, collide=collide)
        # amber rim strips (visual identification of the wall tops; thin, collidable)
        _add_box(stage, f"{prim_path}/rim_x{'p' if sgn > 0 else 'n'}",
                 center=(sgn * (ix + wt / 2), 0.0, zf + wh + 0.0015),
                 size=(wt, 2 * iy + 2 * wt, 0.003), color=c.rim_color, collide=collide)
        _add_box(stage, f"{prim_path}/rim_y{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * (iy + wt / 2), zf + wh + 0.0015),
                 size=(2 * ix + 2 * wt, wt, 0.003), color=c.rim_color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclass (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tray" not in _SPAWNER_CACHE:

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            inner_x: float = 0.135
            inner_y: float = 0.135
            wall_t: float = 0.012
            wall_h: float = 0.050
            z_floor: float = 0.020
            floor_color: tuple = (0.25, 0.25, 0.28)
            wall_color: tuple = (0.33, 0.33, 0.38)
            rim_color: tuple = (0.88, 0.66, 0.12)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["tray"] = TraySpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class TrayPackSceneCfg(BaseCfg):
    """Config for `TrayPackScene`. The seated gates are honest by construction: the
    walls bound a floor-resting block's corners to the interior footprint, the
    20 mm floor plate separates in-tray floor height from the outside ground by a
    full block-half-height, and a block resting on another sits 40 mm too high."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    flat_max_deg: float = tunable(10.0)   # block thickness axis within this of world up
    z_tol: float = tunable(0.010)         # |centre z - (floor top + 20 mm)| gate (m)
    foot_margin: float = tunable(0.004)   # bottom-corner overshoot allowed past the interior (m)
    settle_speed: float = tunable(0.05)   # max |lin vel| of every block when judging (m/s)
    settle_omega: float = tunable(0.5)    # max |ang vel| of every block when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    tray_yaw_deg: float = tunable(180.0)  # tray yaw (+/- deg, FREE)
    tray_jitter: float = tunable(0.05)    # tray xy jitter (+/- m)
    slab_spot: float = tunable(0.020)     # slab standing-spot xy range inside the tray (+/- m)
    slab_yaw_deg: float = tunable(180.0)  # slab standing heading (+/- deg, FREE)
    scatter_jitter: float = tunable(0.04) # bar/brick ground xy jitter (+/- m)
    scatter_yaw_deg: float = tunable(180.0)  # bar/brick ground yaw (+/- deg, FREE)

    # --- info: layout ---------------------------------------------------------------------------
    tray_pos: tuple = info((0.42, 0.02))    # tray centre on the ground (nominal)
    bar_pos: tuple = info((0.28, -0.30))    # red bar scatter spot on the ground (nominal)
    brick_pos: tuple = info((0.24, 0.30))   # green brick scatter spot on the ground (nominal)
    # --- info: tray structure (local frame: origin at floor-plate centre on the ground) ---------
    inner_x: float = info(0.135)
    inner_y: float = info(0.135)
    wall_t: float = info(0.012)
    wall_h: float = info(0.050)   # wall height above the floor top
    z_floor: float = info(0.020)  # floor plate TOP height
    # --- info: blocks (full sizes; all 40 mm tall) ----------------------------------------------
    bar_size: tuple = info((0.120, 0.040, 0.040))
    slab_size: tuple = info((0.080, 0.080, 0.040))
    brick_size: tuple = info((0.080, 0.040, 0.040))
    bar_mass: float = info(0.24)
    slab_mass: float = info(0.32)
    brick_mass: float = info(0.16)
    bar_color: tuple = info((0.85, 0.10, 0.10))
    slab_color: tuple = info((0.12, 0.25, 0.85))
    brick_color: tuple = info((0.10, 0.70, 0.20))
    contact_offset: float = info(0.002)
    # rubric weights (3 x 0.25 = 0.75 = the non-success cap)
    w_seat: float = info(0.25)


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


def _qx(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 1] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("tray_pack")
class TrayPackScene(BaseScene):
    cfg: TrayPackSceneCfg

    BLOCK_NAMES = ("bar", "slab", "brick")

    def __init__(self, cfg: TrayPackSceneCfg | None = None) -> None:
        super().__init__(cfg or TrayPackSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        tray_spawn = cls["tray"](
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            inner_x=c.inner_x, inner_y=c.inner_y, wall_t=c.wall_t, wall_h=c.wall_h,
            z_floor=c.z_floor, contact_offset=c.contact_offset)

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
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=tray_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.tray_pos[0], c.tray_pos[1], 0.0)),
            ),
        }
        sizes = {"bar": c.bar_size, "slab": c.slab_size, "brick": c.brick_size}
        masses = {"bar": c.bar_mass, "slab": c.slab_mass, "brick": c.brick_mass}
        colors = {"bar": c.bar_color, "slab": c.slab_color, "brick": c.brick_color}
        for i, name in enumerate(self.BLOCK_NAMES):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Block_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=sizes[name],
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.2, angular_damping=0.5,
                        sleep_threshold=0.0, stabilization_threshold=0.0,
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=1),
                    mass_props=sim_utils.MassPropertiesCfg(mass=masses[name]),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.003, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.5, dynamic_friction=0.4,
                        restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=colors[name]),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.7 + 0.15 * i, -0.7, 0.05)),
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
        self.tray: RigidObject = env.iscene["tray"]
        self.blocks: dict[str, RigidObject] = {
            n: env.iscene[n] for n in self.BLOCK_NAMES}
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # per-block footprint half extents (local x, y) and half heights
        sizes = (c.bar_size, c.slab_size, c.brick_size)
        self._half_xy = torch.tensor([[s[0] / 2, s[1] / 2] for s in sizes], device=dev)
        self._half_z = torch.tensor([s[2] / 2 for s in sizes], device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._seat = torch.zeros(n, 3, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the tray (free yaw + xy jitter); stand the blue slab
        ON ITS SIDE inside the tray at a random spot/heading; scatter the bar and
        brick flat on the ground outside (jitter + free yaw); clear the latches."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- tray: kinematic, free yaw + xy jitter ---
        t_psi = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.tray_yaw_deg)
        q_tray = _qz(t_psi)
        tp = torch.zeros(m, 3, device=dev)
        tp[:, 0] = c.tray_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.tray_jitter
        tp[:, 1] = c.tray_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.tray_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = tp + origin
        st[:, 3:7] = q_tray
        self.tray.write_root_state_to_sim(st, env_ids)

        # --- slab: standing on a side face INSIDE the tray (local y up), random
        # spot + heading. Base 80x40 half-diag 44.7 mm; |xy| <= 20 mm keeps every
        # heading clear of the walls (67.5 mm half interior).
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0:2] = (torch.rand(m, 2, device=dev) * 2 - 1) * c.slab_spot
        loc[:, 2] = c.z_floor + c.slab_size[0] / 2 + 0.003
        s_psi = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.slab_yaw_deg)
        q_slab = _qmul(q_tray, _qmul(_qz(s_psi),
                                     _qx(torch.full((m,), math.pi / 2, device=dev))))
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = tp + quat_apply(q_tray, loc) + origin
        st[:, 3:7] = q_slab
        self.blocks["slab"].write_root_state_to_sim(st, env_ids)

        # --- bar + brick: flat on the ground outside, jitter + free yaw ---
        for name, nom in (("bar", c.bar_pos), ("brick", c.brick_pos)):
            psi = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.scatter_yaw_deg)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = nom[0] + (torch.rand(m, device=dev) * 2 - 1) * c.scatter_jitter
            st[:, 1] = nom[1] + (torch.rand(m, device=dev) * 2 - 1) * c.scatter_jitter
            st[:, 2] = 0.020 + 0.002
            st[:, 3:7] = _qz(psi)
            st[:, 0:3] += origin
            self.blocks[name].write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._seat[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "blocks": {n: b.data.root_state_w[env_ids].clone()
                       for n, b in self.blocks.items()},
            "seat": self._seat[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        for n, b in self.blocks.items():
            b.write_root_state_to_sim(state["blocks"][n], env_ids)
        self._seat[env_ids] = state["seat"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A dark-grey open TRAY with an amber rim stands on the ground: a square "
            f"box, interior {c.inner_x * 1000:.0f} x {c.inner_y * 1000:.0f} mm, walls "
            f"{c.wall_h * 1000:.0f} mm tall, open on top. There are three solid "
            f"blocks, each {c.bar_size[2] * 1000:.0f} mm thick: a RED BAR "
            f"({c.bar_size[0] * 1000:.0f} x {c.bar_size[1] * 1000:.0f} mm) and a "
            f"GREEN BRICK ({c.brick_size[0] * 1000:.0f} x "
            f"{c.brick_size[1] * 1000:.0f} mm) lying flat on the ground outside the "
            f"tray, and a BLUE SLAB ({c.slab_size[0] * 1000:.0f} x "
            f"{c.slab_size[1] * 1000:.0f} mm) which starts INSIDE the tray standing "
            f"upright on one of its narrow side faces, poking above the rim. The "
            f"tray's position and heading, the slab's standing spot, and the two "
            f"outside blocks' poses vary every episode.\n"
            f"Goal: pack the tray — end with ALL THREE blocks lying FLAT on the tray "
            f"floor at the same time, every block fully inside the walls, nothing "
            f"resting on another block or on the rim. The three footprints "
            f"(120x40, 80x80, 80x40 mm) only just fit the 135x135 mm floor, so they "
            f"must be arranged deliberately: lay the long red bar flush along one "
            f"wall, seat the blue slab square into a corner beside it, and the green "
            f"brick drops into the one remaining pocket (any rotation or mirror "
            f"image of that arrangement works). A block left standing, leaning, "
            f"overlapping the rim, or lying on top of another does not count — the "
            f"standing blue slab must be laid flat and packed like the rest. Blocks "
            f"may be picked up, slid, or pushed; there is no required order, but a "
            f"block placed mid-floor will have to be moved again. Success: all three "
            f"blocks flat on the tray floor inside the walls, everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lay the standing blue slab flat and pack all three blocks — red bar, "
            "blue slab, green brick — so they all lie flat on the tray floor at "
            "once, each fully inside the walls, none resting on another block or "
            "on the rim."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _tray_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the (kinematic, live-read) tray frame. Accepts (N,3) or
        (N,P,3); returns the same shape."""
        from isaaclab.utils.math import quat_apply_inverse

        tp = self.tray.data.root_pos_w
        tq = self.tray.data.root_quat_w
        if pos_w.dim() == 3:
            n, p = pos_w.shape[0], pos_w.shape[1]
            rel = (pos_w - tp[:, None, :]).reshape(n * p, 3)
            q = tq[:, None, :].expand(n, p, 4).reshape(n * p, 4)
            return quat_apply_inverse(q, rel).reshape(n, p, 3)
        return quat_apply_inverse(tq, pos_w - tp)

    def _block_tensors(self) -> tuple[torch.Tensor, ...]:
        """(pos_w (N,3,3), quat (N,3,4), |lin vel| (N,3), |ang vel| (N,3))."""
        pos = torch.stack([b.data.root_pos_w for b in self.blocks.values()], dim=1)
        quat = torch.stack([b.data.root_quat_w for b in self.blocks.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.blocks.values()], dim=1)
        omg = torch.stack([b.data.root_ang_vel_w.norm(dim=-1)
                           for b in self.blocks.values()], dim=1)
        return pos, quat, vel, omg

    def flat(self) -> torch.Tensor:
        """(N, 3) bool: block thickness axis (local +z) within `flat_max_deg` of
        world up — rejects the standing/leaning slab."""
        from isaaclab.utils.math import quat_apply

        _p, quat, _v, _w = self._block_tensors()
        n = quat.shape[0]
        ez = torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(n * 3, 3)
        up = quat_apply(quat.reshape(n * 3, 4), ez).reshape(n, 3, 3)
        return up[:, :, 2] >= math.cos(math.radians(self.cfg.flat_max_deg))

    def seated(self) -> torch.Tensor:
        """(N, 3) bool, live-geometric: block flat, centre at floor height + half
        thickness (rejects on-another-block / on-the-rim / on-the-ground-outside),
        all four bottom corners inside the interior footprint (tray frame)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        pos, quat, _v, _w = self._block_tensors()
        n = pos.shape[0]
        loc_c = self._tray_local(pos)
        z_ok = (loc_c[:, :, 2] - (c.z_floor + self._half_z[None, :])).abs() < c.z_tol
        # bottom corners: local (+/-hx, +/-hy, -hz) -> world -> tray frame
        sx = torch.tensor([1.0, 1.0, -1.0, -1.0], device=pos.device)
        sy = torch.tensor([1.0, -1.0, 1.0, -1.0], device=pos.device)
        corn = torch.zeros(n, 3, 4, 3, device=pos.device)
        corn[:, :, :, 0] = self._half_xy[None, :, 0, None] * sx[None, None, :]
        corn[:, :, :, 1] = self._half_xy[None, :, 1, None] * sy[None, None, :]
        corn[:, :, :, 2] = -self._half_z[None, :, None]
        q = quat[:, :, None, :].expand(n, 3, 4, 4).reshape(-1, 4)
        cw = quat_apply(q, corn.reshape(-1, 3)).reshape(n, 3, 4, 3) \
            + pos[:, :, None, :]
        cl = self._tray_local(cw.reshape(n, 12, 3)).reshape(n, 3, 4, 3)
        in_x = cl[:, :, :, 0].abs() <= c.inner_x / 2 + c.foot_margin
        in_y = cl[:, :, :, 1].abs() <= c.inner_y / 2 + c.foot_margin
        inside = (in_x & in_y).all(dim=-1)
        return self.flat() & z_ok & inside

    def settled(self) -> torch.Tensor:
        """(N, 3) bool: block lin/ang velocity below the settle gates."""
        _p, _q, vel, omg = self._block_tensors()
        return (vel < self.cfg.settle_speed) & (omg < self.cfg.settle_omega)

    def _update_latches(self) -> None:
        self._seat |= self.seated() & self.settled()

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: ALL THREE blocks simultaneously seated (flat on the tray floor,
        fully inside the walls) and at rest, states finite. All clauses are live
        physical outcomes."""
        self._update_latches()
        pos, _q, _v, _w = self._block_tensors()
        finite = torch.isfinite(pos).all(dim=-1).all(dim=-1)
        return self.seated().all(dim=1) & self.settled().all(dim=1) & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.25 per block ever seated (latched; ~0 for doing
        nothing — the standing slab does not count), capped at 0.75 — and exactly
        1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_seat * self._seat.float().sum(dim=1)).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="tray_pack", robot="null"))

"""DropGateOvenScene — unload the blue can, then drop the gravity gate to seal the oven.

Derived from rlbench/close_microwave ("close microwave": push the open hinged door of a
microwave until it shuts), but the MANIPULATION MODEL is replaced wholesale. The seed's
plan is a single unordered act: one pushing contact on a swinging panel, rotating it
about its hinge until flush — no perception demands, no ordering, no other object. Here
the arm NEVER pushes anything shut: the "door" is a guillotine DROP-GATE riding in
vertical side slots, held raised by a yellow PROP POST standing under its bottom edge.
Closing is performed by GRAVITY the instant the loaded post is extracted — the solver's
contribution to closure is a mechanism RELEASE (pull a strut out from under a load), not
a door push. Before that release the solver must PERCEIVE which of the two cans inside
the chamber is blue (their arrangement is shuffled per episode), extract the blue one
through the open doorway and seat it on the green pad outside, while the red can must
REMAIN inside and end up sealed behind the gate. The order is physically forced: once
the gate is down the doorway is covered (a 3x-weight shove cannot push a can out — the
smoke battery proves it), and a single arm cannot hold the gate up and reach inside at
the same time, so extraction must precede release.

Assets are fully procedural (the compound-spawner pattern — child colliders of one body
never self-collide):
  - oven: KINEMATIC chamber (interior 160 x 240 x 180 mm) open on its local +x face
    (doorway 200 mm wide, 30..150 mm tall). The doorway is flanked by vertical slot
    guides: the gate slides between the chamber's front face (behind) and two outer
    guide strips (in front), with slot end caps; an apron plate extends forward as the
    gate seat and the post's footing.
  - gate: DYNAMIC orange panel (240 x 135 x 10 mm) riding in the slots. Raised, its
    bottom edge rests on the prop post at 144 mm; seated, its bottom rests on the apron
    at 30 mm, fully covering the doorway.
  - prop post: DYNAMIC yellow square post (25 x 25 x 114 mm) standing on the apron in
    the doorway plane, under the gate's bottom edge, carrying the gate's weight.
  - cans: two DYNAMIC cylinders (r 25 mm, h 60 mm), one BLUE one RED, standing on the
    chamber floor; which stands where is shuffled per episode.
  - pad: KINEMATIC green disc (r 60 mm, 8 mm thick) on the ground outside.

Per-episode randomization (readback-verifiable): oven yaw +/- 20 deg + xy jitter, pad
xy jitter, prop post lateral position in the doorway (+/- 50 mm), and the blue/red can
arrangement swap + per-can jitter.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve trajectory):
  0.15 * out     — the blue can ever outside the chamber volume (latched; 0 for null)
  0.30 * placed  — the blue can ever seated upright on the green pad, settled (latched)
  0.30 * sealed  — the gate ever fully seated in its slots while the blue can is out
                   and the red can is in (latched; "just close the gate" does NOT
                   latch it: the blue can must already be out)
  1.0 iff success() — gate fully seated in the doorway slots (position AND orientation
                   in the oven frame), blue can upright on the pad, red can inside the
                   chamber, everything settled and finite. Non-success capped at 0.75.

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


def _spawn_oven(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the oven at `prim_path`: KINEMATIC compound. Local frame: origin at the
    chamber footprint centre on the ground; the doorway faces local +x.

    Children: apron/base plate (chamber floor + gate seat + post footing), back wall,
    two side walls, top plate, two front side strips (doorway flanks, the REAR faces of
    the gate slots), a header above the doorway, two outer guide strips (the FRONT
    faces of the slots), two slot end caps, and two rear upper retainers that back the
    raised gate above the header."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    body, guide = c.body_color, c.guide_color
    # apron / base plate: chamber floor, gate seat, post footing (x -0.10 .. +0.17)
    _add_box(stage, f"{prim_path}/apron", center=(0.035, 0.0, c.zf / 2),
             size=(0.27, 0.30, c.zf), color=body, collide=collide)
    # chamber shell
    _add_box(stage, f"{prim_path}/back", center=(-0.086, 0.0, 0.12),
             size=(0.012, 0.264, 0.18), color=body, collide=collide)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/side_{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * 0.126, 0.12),
                 size=(0.184, 0.012, 0.18), color=body, collide=collide)
    _add_box(stage, f"{prim_path}/top", center=(0.0, 0.0, 0.216),
             size=(0.184, 0.264, 0.012), color=body, collide=collide)
    # doorway flanks (rear slot faces) + header
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/front_{'p' if sgn > 0 else 'n'}",
                 center=(0.086, sgn * 0.113, 0.12),
                 size=(0.012, 0.026, 0.18), color=body, collide=collide)
    _add_box(stage, f"{prim_path}/header", center=(0.086, 0.0, 0.18),
             size=(0.012, 0.20, 0.06), color=body, collide=collide)
    # outer guide strips (front slot faces) + slot end caps + rear upper retainers
    for sgn in (1.0, -1.0):
        s = "p" if sgn > 0 else "n"
        _add_box(stage, f"{prim_path}/guide_{s}",
                 center=(0.116, sgn * 0.120, 0.175),
                 size=(0.012, 0.030, 0.29), color=guide, collide=collide)
        _add_box(stage, f"{prim_path}/cap_{s}",
                 center=(0.101, sgn * 0.132, 0.175),
                 size=(0.030, 0.006, 0.29), color=guide, collide=collide)
        _add_box(stage, f"{prim_path}/ret_{s}",
                 center=(0.086, sgn * 0.113, 0.265),
                 size=(0.012, 0.026, 0.11), color=body, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclass (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "oven" not in _SPAWNER_CACHE:

        @configclass
        class OvenSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_oven)
            zf: float = 0.030
            body_color: tuple = (0.28, 0.30, 0.34)
            guide_color: tuple = (0.42, 0.45, 0.50)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["oven"] = OvenSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class DropGateOvenSceneCfg(BaseCfg):
    """Config for `DropGateOvenScene`. The gate-seated tolerances are honest by
    construction: the slot guides bound a seated gate centre to within ~4 mm of the
    nominal plane in x and ~9 mm in y, so any gate physically seated in its slots
    passes the (20/30 mm) gates, while a gate lying anywhere else cannot."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    gate_z_tol: float = tunable(0.012)      # gate centre within this above its seated height
    gate_x_tol: float = tunable(0.020)      # |oven-frame x - gate plane| gate (slots bound 4 mm)
    gate_y_tol: float = tunable(0.030)      # |oven-frame y| gate (end caps bound 9 mm)
    gate_tilt_max_deg: float = tunable(10.0)  # gate height axis within this of oven +z
    pad_xy_tol: float = tunable(0.045)      # blue can centre within this of the pad axis
    pad_z_tol: float = tunable(0.015)       # blue can centre height above the pad within this
    can_upright_max_deg: float = tunable(15.0)  # blue can axis within this of world-up
    settle_speed: float = tunable(0.05)     # max |lin vel| (gate + both cans) when judging (m/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    oven_yaw_deg: float = tunable(20.0)     # oven yaw about its nominal heading (+/- deg)
    oven_jitter: float = tunable(0.04)      # oven xy jitter (+/- m)
    pad_jitter: float = tunable(0.05)       # pad xy jitter (+/- m)
    post_y_range: float = tunable(0.05)     # prop post lateral slot in the doorway (+/- m)
    can_swap: bool = tunable(True)          # shuffle which interior slot holds the blue can
    can_jitter: float = tunable(0.012)      # per-can xy jitter (+/- m)

    # --- info: layout (world nominal; oven doorway faces its local +x) --------------------------
    oven_pos: tuple = info((0.44, -0.04))   # oven origin on the ground (nominal)
    oven_yaw_nom_deg: float = info(180.0)   # nominal heading: doorway faces world -x
    pad_pos: tuple = info((0.20, 0.30))     # pad centre on the ground (nominal)
    # --- info: oven structure (local frame: origin at chamber footprint centre, ground) ---------
    zf: float = info(0.030)        # chamber floor / apron top height
    int_d: float = info(0.16)      # chamber interior depth  (local x: -0.08 .. +0.08)
    int_w: float = info(0.24)      # chamber interior width  (local y)
    int_h: float = info(0.18)      # chamber interior height (floor top .. ceiling)
    door_w: float = info(0.20)     # doorway width
    door_top: float = info(0.15)   # doorway top (header bottom)
    x_front: float = info(0.092)   # front wall OUTER face = rear slot plane
    x_gate: float = info(0.097)    # gate centre plane (slot: 0.092 .. 0.110)
    # --- info: gate ------------------------------------------------------------------------------
    gate_t: float = info(0.010)
    gate_w: float = info(0.240)
    gate_h: float = info(0.135)
    gate_mass: float = info(0.15)
    z_seat: float = info(0.0975)   # seated gate CENTRE height (bottom on the apron at zf)
    gate_color: tuple = info((0.90, 0.45, 0.10))
    # --- info: prop post -------------------------------------------------------------------------
    post_xy: float = info(0.025)   # square cross-section
    post_h: float = info(0.114)    # top at 0.144 = raised gate bottom
    post_x: float = info(0.100)    # post centre x (under the gate plane, clear of the header)
    post_mass: float = info(0.05)
    post_color: tuple = info((0.92, 0.80, 0.12))
    # --- info: cans ------------------------------------------------------------------------------
    can_r: float = info(0.025)
    can_h: float = info(0.060)
    can_mass: float = info(0.10)
    can_slot_x: float = info(0.02)   # interior slots at (can_slot_x, +/- can_slot_y)
    can_slot_y: float = info(0.06)
    blue_color: tuple = info((0.10, 0.30, 0.85))
    red_color: tuple = info((0.85, 0.10, 0.10))
    # --- info: pad -------------------------------------------------------------------------------
    pad_r: float = info(0.060)
    pad_t: float = info(0.008)
    pad_color: tuple = info((0.10, 0.65, 0.20))
    contact_offset: float = info(0.002)
    # rubric weights (0.15 + 0.30 + 0.30 = 0.75 = the non-success cap)
    w_out: float = info(0.15)
    w_placed: float = info(0.30)
    w_sealed: float = info(0.30)


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


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("drop_gate_oven")
class DropGateOvenScene(BaseScene):
    cfg: DropGateOvenSceneCfg

    def __init__(self, cfg: DropGateOvenSceneCfg | None = None) -> None:
        super().__init__(cfg or DropGateOvenSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        oven_spawn = cls["oven"](
            mass_props=sim_utils.MassPropertiesCfg(mass=25.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            zf=c.zf, contact_offset=c.contact_offset)

        dyn_props = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.2, angular_damping=0.2,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5, dynamic_friction=0.4, restitution=0.0),
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
            "oven": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Oven",
                spawn=oven_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.oven_pos[0], c.oven_pos[1], 0.0)),
            ),
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad",
                spawn=sim_utils.CylinderCfg(
                    radius=c.pad_r, height=c.pad_t, axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pad_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pad_pos[0], c.pad_pos[1], c.pad_t / 2)),
            ),
            "gate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Gate",
                spawn=sim_utils.CuboidCfg(
                    size=(c.gate_t, c.gate_w, c.gate_h),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.gate_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.gate_color),
                    **dyn_props,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 1.0, 0.10)),
            ),
            "post": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Post",
                spawn=sim_utils.CuboidCfg(
                    size=(c.post_xy, c.post_xy, c.post_h),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.post_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.post_color),
                    **dyn_props,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 1.3, 0.06)),
            ),
        }
        for name in ("blue_can", "red_can"):
            color = c.blue_color if name == "blue_can" else c.red_color
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Can_" + name,
                spawn=sim_utils.CylinderCfg(
                    radius=c.can_r, height=c.can_h, axis="Z",
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.can_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    **dyn_props,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.3, 1.0, 0.05)),
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
        self.oven: RigidObject = env.iscene["oven"]
        self.pad: RigidObject = env.iscene["pad"]
        self.gate: RigidObject = env.iscene["gate"]
        self.post: RigidObject = env.iscene["post"]
        self.blue: RigidObject = env.iscene["blue_can"]
        self.red: RigidObject = env.iscene["red_can"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # blue_slot[e] = +1 / -1: sign of the interior y slot the BLUE can occupies
        self.blue_slot = torch.ones(n, dtype=torch.float, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._out = torch.zeros(n, dtype=torch.bool, device=dev)
        self._placed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._sealed = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the oven (yaw + xy jitter) and pad (xy jitter), stand
        the prop post on the apron at a random lateral slot in the doorway, rest the
        gate raised on the post, place the two cans on the chamber floor (arrangement
        swap + jitter), clear the latches."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- oven: kinematic, nominal heading + yaw + xy jitter ---
        yaw = math.radians(c.oven_yaw_nom_deg) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.oven_yaw_deg)
        q_oven = _qz(yaw)
        op = torch.zeros(m, 3, device=dev)
        op[:, 0] = c.oven_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.oven_jitter
        op[:, 1] = c.oven_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.oven_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = op + origin
        st[:, 3:7] = q_oven
        self.oven.write_root_state_to_sim(st, env_ids)

        # --- pad: kinematic, xy jitter ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.pad_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.pad_jitter
        st[:, 1] = c.pad_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.pad_jitter
        st[:, 2] = c.pad_t / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.pad.write_root_state_to_sim(st, env_ids)

        def place(body, loc: torch.Tensor, q_extra: torch.Tensor | None = None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = op + quat_apply(q_oven, loc) + origin
            st[:, 3:7] = q_oven if q_extra is None else _qmul(q_oven, q_extra)
            body.write_root_state_to_sim(st, env_ids)

        # --- prop post: standing on the apron in the doorway plane, random y slot ---
        post_y = (torch.rand(m, device=dev) * 2 - 1) * c.post_y_range
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.post_x
        loc[:, 1] = post_y
        loc[:, 2] = c.zf + c.post_h / 2
        place(self.post, loc)

        # --- gate: raised, bottom resting on the post top (spawn 1 mm above) ---
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.x_gate
        loc[:, 2] = c.zf + c.post_h + 0.001 + c.gate_h / 2
        place(self.gate, loc)

        # --- cans: interior slots, arrangement swap + jitter ---
        if c.can_swap:
            swap = torch.where(torch.rand(m, device=dev) < 0.5,
                               torch.ones(m, device=dev), -torch.ones(m, device=dev))
        else:
            swap = torch.ones(m, device=dev)
        self.blue_slot[env_ids] = swap
        for body, sgn in ((self.blue, swap), (self.red, -swap)):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = c.can_slot_x \
                + (torch.rand(m, device=dev) * 2 - 1) * c.can_jitter
            loc[:, 1] = sgn * c.can_slot_y \
                + (torch.rand(m, device=dev) * 2 - 1) * c.can_jitter
            loc[:, 2] = c.zf + c.can_h / 2 + 0.002
            place(body, loc)

        # --- clear latches ---
        self._out[env_ids] = False
        self._placed[env_ids] = False
        self._sealed[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "oven": self.oven.data.root_state_w[env_ids].clone(),
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "gate": self.gate.data.root_state_w[env_ids].clone(),
            "post": self.post.data.root_state_w[env_ids].clone(),
            "blue": self.blue.data.root_state_w[env_ids].clone(),
            "red": self.red.data.root_state_w[env_ids].clone(),
            "blue_slot": self.blue_slot[env_ids].clone(),
            "out": self._out[env_ids].clone(),
            "placed": self._placed[env_ids].clone(),
            "sealed": self._sealed[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.oven.write_root_state_to_sim(state["oven"], env_ids)
        self.pad.write_root_state_to_sim(state["pad"], env_ids)
        self.gate.write_root_state_to_sim(state["gate"], env_ids)
        self.post.write_root_state_to_sim(state["post"], env_ids)
        self.blue.write_root_state_to_sim(state["blue"], env_ids)
        self.red.write_root_state_to_sim(state["red"], env_ids)
        self.blue_slot[env_ids] = state["blue_slot"]
        self._out[env_ids] = state["out"]
        self._placed[env_ids] = state["placed"]
        self._sealed[env_ids] = state["sealed"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A dark-grey OVEN BOX stands on the ground, its open doorway "
            f"({c.door_w * 1000:.0f} mm wide, ~{(c.door_top - c.zf) * 1000:.0f} mm tall) "
            f"facing you. The doorway has no hinged door: an ORANGE DROP-GATE (a "
            f"{c.gate_w * 1000:.0f} x {c.gate_h * 1000:.0f} mm panel) rides in vertical "
            f"slots at the doorway's sides, and it is currently RAISED, held up only by "
            f"a YELLOW PROP POST ({c.post_xy * 1000:.0f} mm square, "
            f"{c.post_h * 1000:.0f} mm tall) standing under its bottom edge in the "
            f"doorway. If the post is pulled or knocked out from under the gate, the "
            f"gate falls in its slots and seals the doorway; once down it sits between "
            f"guides and the chamber cannot be reached. Inside the chamber, two metal "
            f"cans (each {2 * c.can_r * 1000:.0f} mm across, {c.can_h * 1000:.0f} mm "
            f"tall) stand on the floor: one BLUE and one RED. Which can stands on which "
            f"side is shuffled every episode — look at the colors. On the ground beside "
            f"the oven lies a flat GREEN PAD (a disc {2 * c.pad_r * 1000:.0f} mm "
            f"across). The oven's position and heading, the pad's position, and the "
            f"post's lateral spot in the doorway also vary per episode.\n"
            f"Goal: take the BLUE can out of the chamber through the open doorway and "
            f"stand it UPRIGHT on the green pad; leave the RED can inside; then remove "
            f"the yellow prop post so the orange gate drops and seals the doorway. The "
            f"final state must have: the gate fully seated in its slots (bottom resting "
            f"on the oven's apron, covering the doorway), the blue can upright on the "
            f"pad, and the red can still inside the sealed chamber. Mind the order: "
            f"once the gate is down you cannot reach the cans anymore, so take the blue "
            f"can out FIRST. Taking out the red can, leaving the blue can inside, "
            f"tipping the blue can over, or leaving the gate off its seat all fail."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Take the blue can out of the oven and stand it upright on the green pad, "
            "leaving the red can inside. Then pull the yellow prop post out from under "
            "the raised orange gate so the gate drops and seals the doorway. The blue "
            "can must be out before the gate falls."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _oven_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the (kinematic, live-read) oven frame, (N,3) -> (N,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.oven.data.root_quat_w,
                                  pos_w - self.oven.data.root_pos_w)

    def _in_chamber(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,) bool: world point inside the chamber interior volume (oven frame)."""
        c = self.cfg
        loc = self._oven_local(pos_w)
        return (loc[:, 0] > -c.int_d / 2) & (loc[:, 0] < c.x_front) \
            & (loc[:, 1].abs() < c.int_w / 2) \
            & (loc[:, 2] > c.zf - 0.01) & (loc[:, 2] < c.zf + c.int_h)

    def _upright(self, body, max_deg: float) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        up = quat_apply(body.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(max_deg))

    def gate_seated(self) -> torch.Tensor:
        """(N,) bool, geometric: the gate fully seated in its doorway slots — oven-frame
        centre within (`gate_x_tol`, `gate_y_tol`) of the slot plane, centre height
        within `gate_z_tol` above the seated height, and the gate's height axis within
        `gate_tilt_max_deg` of oven +z. Honest by construction: the slot guides bound a
        seated gate far inside these gates; a gate lying flat on the ground, leaning
        elsewhere, or hanging above its seat cannot pass."""
        c = self.cfg
        loc = self._oven_local(self.gate.data.root_pos_w)
        pos_ok = (loc[:, 0] - c.x_gate).abs() < c.gate_x_tol
        pos_ok &= loc[:, 1].abs() < c.gate_y_tol
        pos_ok &= (loc[:, 2] - c.z_seat) < c.gate_z_tol
        pos_ok &= loc[:, 2] > c.z_seat - 0.02
        return pos_ok & self._upright(self.gate, c.gate_tilt_max_deg)

    def blue_on_pad(self) -> torch.Tensor:
        """(N,) bool: blue can standing upright on the green pad — xy within
        `pad_xy_tol` of the pad axis, centre at pad-top + can half-height within
        `pad_z_tol`, axis within `can_upright_max_deg` of world-up."""
        c = self.cfg
        d_xy = (self.blue.data.root_pos_w[:, :2] - self.pad.data.root_pos_w[:, :2]).norm(dim=-1)
        z_rel = self.blue.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        z_tgt = c.pad_t + c.can_h / 2
        return (d_xy < c.pad_xy_tol) & ((z_rel - z_tgt).abs() < c.pad_z_tol) \
            & self._upright(self.blue, c.can_upright_max_deg)

    def red_inside(self) -> torch.Tensor:
        """(N,) bool: red can centre inside the chamber interior volume."""
        return self._in_chamber(self.red.data.root_pos_w)

    def blue_outside(self) -> torch.Tensor:
        """(N,) bool: blue can centre OUTSIDE the chamber interior volume."""
        return ~self._in_chamber(self.blue.data.root_pos_w)

    def settled(self) -> torch.Tensor:
        """(N,) bool: gate and both cans |lin vel| below `settle_speed`."""
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                         for b in (self.gate, self.blue, self.red)], dim=1)
        return (v < self.cfg.settle_speed).all(dim=1)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w
                         for b in (self.gate, self.post, self.blue, self.red)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        self._out |= self.blue_outside() & self._finite()
        self._placed |= self.blue_on_pad() & self.settled()
        self._sealed |= self.gate_seated() & self.blue_outside() & self.red_inside()

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the gate fully seated in its doorway slots, the blue can upright
        on the green pad, the red can inside the (sealed) chamber, everything settled
        and finite. All clauses are live physical outcomes."""
        self._update_latches()
        return self.gate_seated() & self.blue_on_pad() & self.red_inside() \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*out + 0.30*placed + 0.30*sealed (all latched;
        ~0 for doing nothing — note `sealed` requires the blue can OUT, so merely
        dropping the gate scores ~0), capped at 0.75 — and exactly 1.0 iff success()
        holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_out * self._out.float() + c.w_placed * self._placed.float()
                + c.w_sealed * self._sealed.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="drop_gate_oven", robot="null"))

"""LatchDrawerScene — press-to-open latch drawer, then retrieve the stashed cube
(libero_kitchen_scene1_open_top_drawer_i15).

Derived from libero_90 kitchen_scene1 "open the top drawer of the cabinet", where the
whole task is: grasp the drawer handle and PULL the prismatic drawer open (goal = the
drawer joint past a threshold). Here that plan is impossible and the goal moves:

  - the drawer has NO handle: its front plate sits flush inside the cabinet face (a
    5 mm seam all round) — there is nothing to hook or pinch, so grasp-and-pull is off
    the table by construction;
  - the drawer is a PUSH-LATCH mechanism (like a real push-to-open kitchen fitting):
    while latched, a stiff detent spring anchors it shut, and even a state-teleported
    "pulled open" drawer snaps straight back closed. The only way to open it is the
    INVERSE of the seed's motion: press the front plate INWARD past the arming depth
    (~14 mm, quasi-static), then withdraw — the latch clicks off and an ejection
    spring drives the drawer out toward the robot on its own;
  - the drawer being open is NOT the goal, only transient access: sealed inside is a
    small RED cube, and success is that cube settled on the GREEN floor pad beside the
    cabinet. A same-size BLUE decoy cube already lies on the floor on the opposite
    flank; moving it achieves nothing.

So a solver needs a different PLAN, not different numbers: push away from itself
against a flush face (no grasp at all), let the mechanism do the opening, then a
pick-out-and-place — versus the seed's single grasp-handle-and-pull. The execution
order is enforced by physics: the cube is enclosed until the drawer ejects.

Success is judged on the PHYSICAL terminal state only: the red cube resting ON the pad
(xy within tolerance, z at pad-top rest height, settled). Partial credit latches the
stages the real solution passes through: latch armed (pressed deep enough) -> drawer
ejected -> cube lifted clear -> cube on the pad.

Mechanism notes (proven robobench cribs):
  - Cabinet carcass is kinematic and NEVER teleported; the drawer is one compound rigid
    body on a per-env authored USD prismatic joint (axis X, body0 = cabinet, body1 =
    drawer, pair collision disabled — authored clearances are real clearances).
    Symmetric joint limits (the GPU sign-convention hedge); the real end stops are
    one-sided stiff springs in post_step.
  - The latch is a post_step force law on the drawer (world x = the joint axis):
    latched -> capped spring anchored at CLOSED (press resistance ~3.4 N at the arming
    depth, snap-back damped hard so a yanked-open drawer cannot self-arm on the
    rebound; arming additionally requires the press to be quasi-static);
    released -> capped spring toward the OPEN stop (gentle ~4 m/s^2 ejection, below the
    cube's friction slip threshold, so the cargo rides the drawer out).
  - Reset re-poses only the FOLLOWER at its authored closed pose (pure re-pin + 2
    substep grace); randomization lives in the pad (polar about the planned robot
    base), the cube-in-drawer offset/yaw and the decoy.

Everything is procedural. Heavy imports (isaaclab, pxr) are deferred so importing this
module stays app-free.
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


# ----- custom compound spawners -------------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_root(prim_path: str, translation, orientation, *, kinematic: bool, mass: float,
                max_depen: float = 0.5):
    """Root Xform + rigid-body APIs, xform ops authored fresh (no duplicate-op clones)."""
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
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(max_depen)
    return stage, root, pxrb


def _box_part(stage, prim_path: str, name: str, size, center, color, contact_offset) -> None:
    """One box child; collider iff contact_offset is not None."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    seg = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)


def _spawn_cabinet(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Kinematic cabinet carcass, root at the front-face center on the ground: bottom
    panel, two side panels, back panel, top cover, and a face frame (two stiles + a
    bottom rail) forming the drawer aperture. Local +x runs INTO the cabinet."""
    stage, root, _ = _apply_root(prim_path, translation, orientation, kinematic=True, mass=5.0)
    c, co = cfg, cfg.contact_offset
    d, w2 = c.depth, c.half_w  # depth into +x, half width
    ap_y, ap_lo, ap_hi = c.ap_half_y, c.ap_z_lo, c.ap_z_hi
    top_t, side_t, back_t, face_t, bot_top = c.top_t, c.side_t, c.back_t, c.face_t, c.bot_top
    # bottom panel (starts behind the face frame)
    _box_part(stage, prim_path, "bottom", (d - face_t, 2 * w2, bot_top),
              (face_t + (d - face_t) / 2, 0.0, bot_top / 2), c.color, co)
    # side panels
    for sy in (-1.0, 1.0):
        _box_part(stage, prim_path, f"side_{'p' if sy > 0 else 'n'}",
                  (d - face_t, side_t, ap_hi - bot_top),
                  (face_t + (d - face_t) / 2, sy * (w2 - side_t / 2),
                   bot_top + (ap_hi - bot_top) / 2), c.color, co)
    # back panel
    _box_part(stage, prim_path, "back", (back_t, 2 * (w2 - side_t), ap_hi - bot_top),
              (d - back_t / 2, 0.0, bot_top + (ap_hi - bot_top) / 2), c.color, co)
    # top cover (spans the full footprint, seals the cavity from above)
    _box_part(stage, prim_path, "top", (d, 2 * w2, top_t), (d / 2, 0.0, ap_hi + top_t / 2),
              c.color, co)
    # face frame: stiles left/right of the aperture + bottom rail below it
    for sy in (-1.0, 1.0):
        _box_part(stage, prim_path, f"stile_{'p' if sy > 0 else 'n'}",
                  (face_t, w2 - ap_y, ap_hi), (face_t / 2, sy * (ap_y + (w2 - ap_y) / 2),
                                               ap_hi / 2), c.face_color, co)
    _box_part(stage, prim_path, "rail_bottom", (face_t, 2 * ap_y, ap_lo),
              (face_t / 2, 0.0, ap_lo / 2), c.face_color, co)
    return root


def _spawn_drawer(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The drawer: one dynamic compound body. Body frame: origin at the CAVITY FLOOR TOP
    CENTER; the front plate extends toward local -x (the robot side), the back wall
    toward +x. Sleep threshold zero (driven by post_step forces)."""
    stage, root, pxrb = _apply_root(prim_path, translation, orientation, kinematic=False,
                                    mass=cfg.mass_props.mass)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    c, co = cfg, cfg.contact_offset
    lx, ly = c.cav_x, c.cav_y  # cavity inner extents
    wt, wh, ft = c.wall_t, c.wall_h, c.floor_t
    _box_part(stage, prim_path, "floor", (lx, ly, ft), (0.0, 0.0, -ft / 2), c.color, co)
    for sy in (-1.0, 1.0):
        _box_part(stage, prim_path, f"wall_{'p' if sy > 0 else 'n'}",
                  (lx, wt, wh), (0.0, sy * (ly / 2 + wt / 2), wh / 2), c.color, co)
    _box_part(stage, prim_path, "back", (wt, ly + 2 * wt, wh), (lx / 2 + wt / 2, 0.0, wh / 2),
              c.color, co)
    # front plate: taller and wider than the cavity, the visible flush face
    _box_part(stage, prim_path, "plate", (c.plate_t, c.plate_w, c.plate_h),
              (-lx / 2 - c.plate_t / 2, 0.0, c.plate_cz), c.plate_color, co)
    return root


def _cabinet_spawner_cfg(c: LatchDrawerSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "cabinet" not in _SPAWNER_CACHE:

        @configclass
        class CabinetSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cabinet)
            depth: float = 0.26
            half_w: float = 0.13
            ap_half_y: float = 0.10
            ap_z_lo: float = 0.030
            ap_z_hi: float = 0.126
            top_t: float = 0.022
            side_t: float = 0.025
            back_t: float = 0.030
            face_t: float = 0.015
            bot_top: float = 0.033
            color: tuple = (0.52, 0.36, 0.20)
            face_color: tuple = (0.44, 0.30, 0.16)
            contact_offset: float = 0.003

        _SPAWNER_CACHE["cabinet"] = CabinetSpawnerCfg

    return _SPAWNER_CACHE["cabinet"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        depth=c.cab_depth, half_w=c.cab_half_w, ap_half_y=c.ap_half_y,
        ap_z_lo=c.ap_z_lo, ap_z_hi=c.ap_z_hi, top_t=c.cab_top_t, side_t=c.cab_side_t,
        back_t=c.cab_back_t, face_t=c.cab_face_t, bot_top=c.cab_bot_top,
        color=c.cab_color, face_color=c.face_color, contact_offset=c.contact_offset,
    )


def _drawer_spawner_cfg(c: LatchDrawerSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "drawer" not in _SPAWNER_CACHE:

        @configclass
        class DrawerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_drawer)
            cav_x: float = 0.16
            cav_y: float = 0.15
            wall_t: float = 0.010
            wall_h: float = 0.045
            floor_t: float = 0.010
            plate_t: float = 0.014
            plate_w: float = 0.19
            plate_h: float = 0.084
            plate_cz: float = 0.030
            color: tuple = (0.62, 0.46, 0.28)
            plate_color: tuple = (0.72, 0.54, 0.32)
            contact_offset: float = 0.003

        _SPAWNER_CACHE["drawer"] = DrawerSpawnerCfg

    return _SPAWNER_CACHE["drawer"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.drawer_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        cav_x=c.cav_x, cav_y=c.cav_y, wall_t=c.drw_wall_t, wall_h=c.drw_wall_h,
        floor_t=c.drw_floor_t, plate_t=c.plate_t, plate_w=c.plate_w, plate_h=c.plate_h,
        plate_cz=c.plate_cz, color=c.drw_color, plate_color=c.plate_color,
        contact_offset=c.contact_offset,
    )


# ----- scene cfg -----------------------------------------------------------------------------------
@dataclass
class LatchDrawerSceneCfg(BaseCfg):
    """Config for `LatchDrawerScene`. World frame (per env origin): the cabinet stands at
    +x with its flush face toward the robot side (-x); the drawer slides along x (INWARD
    is +x = pressing, OUTWARD is -x = opening toward the robot); the pad and decoy lie on
    the floor on opposite lateral flanks. Laid out for a Franka base at `base_anchor`."""

    # --- tunable: rubric thresholds ----------------------------------------------------------------
    pad_tol: float = tunable(0.050)  # cube center xy within this of the pad center
    on_pad_z_lo: float = tunable(0.026)  # cube center z band = resting ON the pad top
    on_pad_z_hi: float = tunable(0.042)  # (floor rest is ~0.0235 -> rejected)
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    lift_z: float = tunable(0.16)  # cube center above this latches "lifted"
    open_gate: float = tunable(0.090)  # drawer opening that latches "ejected" (m)

    # --- tunable: latch mechanism ------------------------------------------------------------------
    press_arm: float = tunable(0.014)  # press depth that arms the latch (m)
    press_release: float = tunable(0.004)  # armed + back above this -> latch releases
    press_stop: float = tunable(0.020)  # inward hard stop (one-sided stiff spring)
    open_stop: float = tunable(0.130)  # outward hard stop = full opening (m)
    arm_speed: float = tunable(0.03)  # arming requires |v_x| below this (quasi-static)
    k_latch: float = tunable(240.0)  # latch spring (N/m), force-capped
    latch_cap: float = tunable(6.0)  # |F| cap of the latch spring (N)
    c_press: float = tunable(8.0)  # latch damping while pressed side (N*s/m)
    c_snap: float = tunable(30.0)  # latch damping while yanked open (kills rebound)
    k_eject: float = tunable(60.0)  # ejection spring toward the open stop (N/m)
    eject_cap: float = tunable(2.5)  # |F| cap of the ejection spring (N)
    c_eject: float = tunable(7.0)  # ejection damping (N*s/m)
    k_stop: float = tunable(2500.0)  # one-sided end-stop stiffness (N/m)

    # --- tunable: randomization ----------------------------------------------------------------
    pad_bearing: tuple = tunable((48.0, 72.0))  # |bearing| from base +x (deg), side sampled
    pad_radius: tuple = tunable((0.44, 0.54))  # pad center distance from the base anchor
    decoy_bearing: tuple = tunable((25.0, 55.0))  # decoy polar band, MIRROR flank of the pad
    decoy_radius: tuple = tunable((0.40, 0.48))
    cube_x_band: tuple = tunable((-0.048, -0.020))  # cube center in the drawer frame (front)
    cube_y_abs: float = tunable(0.022)  # keeps finger descent clear of the cavity walls
    cube_yaw_deg: float = tunable(12.0)
    decoy_jitter: float = tunable(0.02)

    # --- info: placement (laid out for the solve.py base pose) --------------------------------------
    base_anchor: tuple = info((-0.28, 0.0))  # planned Franka base xy (solve.py uses this)
    cab_face_x: float = info(0.30)  # cabinet front face plane
    drawer_closed_x: float = info(0.394)  # drawer body origin, closed (= cab_face_x + face_t
    #                                       + cav_x/2 - ... : plate front face flush at 0.300)

    # --- info: cabinet carcass (kinematic, FIXED — never teleported) ------------------------------
    cab_depth: float = info(0.26)
    cab_half_w: float = info(0.13)
    ap_half_y: float = info(0.10)
    ap_z_lo: float = info(0.030)
    ap_z_hi: float = info(0.126)
    cab_top_t: float = info(0.022)
    cab_side_t: float = info(0.025)
    cab_back_t: float = info(0.030)
    cab_face_t: float = info(0.015)
    cab_bot_top: float = info(0.033)
    cab_color: tuple = info((0.52, 0.36, 0.20))
    face_color: tuple = info((0.44, 0.30, 0.16))

    # --- info: drawer ------------------------------------------------------------------------------
    drawer_mass: float = info(0.45)
    cav_x: float = info(0.16)  # cavity inner length (x)
    cav_y: float = info(0.15)  # cavity inner width (y)
    drw_wall_t: float = info(0.010)
    drw_wall_h: float = info(0.045)
    drw_floor_t: float = info(0.010)
    drw_floor_z: float = info(0.048)  # cavity floor TOP height (world z, drawer origin)
    plate_t: float = info(0.014)
    plate_w: float = info(0.19)
    plate_h: float = info(0.084)
    plate_cz: float = info(0.030)  # plate center z in the drawer frame
    drw_color: tuple = info((0.62, 0.46, 0.28))
    plate_color: tuple = info((0.72, 0.54, 0.32))

    # --- info: cube, decoy, pad --------------------------------------------------------------------
    cube_size: float = info(0.045)
    cube_mass: float = info(0.12)
    cube_mu: float = info(0.9)
    cube_color: tuple = info((0.85, 0.12, 0.10))  # RED — the cargo
    decoy_color: tuple = info((0.12, 0.25, 0.85))  # BLUE — the decoy
    pad_half: float = info(0.070)
    pad_t: float = info(0.008)
    pad_color: tuple = info((0.10, 0.72, 0.20))  # GREEN — the goal pad
    contact_offset: float = info(0.003)

    # Derived (filled in __post_init__).
    plate_face_x: float = field(default=None, init=False)  # plate front face, closed
    cube_rest_z: float = field(default=None, init=False)  # cube center resting in the drawer

    def __post_init__(self) -> None:
        self.plate_face_x = round(self.drawer_closed_x - self.cav_x / 2 - self.plate_t, 4)
        self.cube_rest_z = round(self.drw_floor_z + self.cube_size / 2, 4)


# ----- scene ----------------------------------------------------------------------------------------
@SCENES.register("latch_drawer")
class LatchDrawerScene(BaseScene):
    cfg: LatchDrawerSceneCfg

    def __init__(self, cfg: LatchDrawerSceneCfg | None = None) -> None:
        super().__init__(cfg or LatchDrawerSceneCfg())

    # ----- assets ---------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        tight = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset,
                                                 rest_offset=0.0)
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
        out["cabinet"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cabinet",
            spawn=_cabinet_spawner_cfg(c),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(c.cab_face_x, 0.0, 0.0)),
        )
        out["drawer"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Drawer",
            spawn=_drawer_spawner_cfg(c),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.drawer_closed_x, 0.0, c.drw_floor_z)),
        )
        for name, color in (("cube", c.cube_color), ("decoy", c.decoy_color)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.capitalize(),
                spawn=sim_utils.CuboidCfg(
                    size=(c.cube_size,) * 3,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
                    collision_props=tight,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.cube_mu, dynamic_friction=c.cube_mu - 0.15,
                        restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.36 if name == "cube" else 0.10,
                         0.0 if name == "cube" else -0.30, c.cube_size / 2 + 0.002)),
            )
        out["pad"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Pad",
            spawn=sim_utils.CuboidCfg(
                size=(2 * c.pad_half, 2 * c.pad_half, c.pad_t),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=tight,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pad_color),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.10, 0.40, c.pad_t / 2)),
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

    # ----- lifecycle --------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n = env.num_envs
        dev = env.device
        self.cabinet: RigidObject = env.iscene["cabinet"]
        self.drawer: RigidObject = env.iscene["drawer"]
        self.cube: RigidObject = env.iscene["cube"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.pad: RigidObject = env.iscene["pad"]
        self.env_origins = env.iscene.env_origins
        self._author_joint()
        # mechanism state
        self._armed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._released = torch.zeros(n, dtype=torch.bool, device=dev)
        # latched progress
        self._ejected_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._lifted_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._placed_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._pad_xy = torch.zeros(n, 2, device=dev)
        self._grace = torch.zeros(n, dtype=torch.long, device=dev)

    def _author_joint(self) -> None:
        """Per env: the drawer's X prismatic slide on the cabinet — pair collision
        disabled, SYMMETRIC generous limits (the GPU sign-convention hedge); the real
        travel stops are one-sided stiff springs in post_step."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/drawer_slide")
            j.CreateBody0Rel().SetTargets([f"{base}/Cabinet"])
            j.CreateBody1Rel().SetTargets([f"{base}/Drawer"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(c.drawer_closed_x - c.cab_face_x, 0.0,
                                           c.drw_floor_z))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-0.16)
            j.CreateUpperLimitAttr(0.16)

    # ----- geometry queries ---------------------------------------------------------------------------
    def drawer_disp(self) -> torch.Tensor:
        """(N,) drawer displacement from CLOSED along the slide (m): positive = pressed
        inward, negative = open toward the robot."""
        return (self.drawer.data.root_pos_w[:, 0] - self.env_origins[:, 0]
                - self.cfg.drawer_closed_x)

    def drawer_open(self) -> torch.Tensor:
        """(N,) opening distance (m), >= 0."""
        return (-self.drawer_disp()).clamp(min=0.0)

    def drawer_pose_at(self, disp: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """(M,13) drawer root state at displacement `disp` (zero velocity), world frame."""
        c = self.cfg
        m = disp.shape[0]
        st = torch.zeros(m, 13, device=disp.device)
        st[:, 0] = c.drawer_closed_x + disp
        st[:, 2] = c.drw_floor_z
        st[:, 3] = 1.0
        st[:, 0:3] += self.env_origins[env_ids]
        return st

    def cube_local(self) -> torch.Tensor:
        """(N,3) cube center in the DRAWER frame (drawer never rotates)."""
        return self.cube.data.root_pos_w - self.drawer.data.root_pos_w

    def cube_in_drawer(self) -> torch.Tensor:
        """(N,) bool: cube center inside the drawer cavity volume."""
        c = self.cfg
        loc = self.cube_local()
        return ((loc[:, 0].abs() < c.cav_x / 2) & (loc[:, 1].abs() < c.cav_y / 2)
                & (loc[:, 2] > -0.005) & (loc[:, 2] < c.drw_wall_h + c.cube_size))

    def cube_on_pad(self) -> torch.Tensor:
        """(N,) bool, geometric: RED cube centered on the pad, resting at pad-top height."""
        c = self.cfg
        p = self.cube.data.root_pos_w - self.env_origins
        near = (p[:, :2] - self._pad_xy).norm(dim=-1) < c.pad_tol
        return near & (p[:, 2] > c.on_pad_z_lo) & (p[:, 2] < c.on_pad_z_hi)

    def cube_settled(self) -> torch.Tensor:
        return self.cube.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    def drawer_still(self) -> torch.Tensor:
        return self.drawer.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    # ----- mechanism (every substep) ------------------------------------------------------------------
    def post_step(self) -> None:
        c = self.cfg
        dev = self.env.device
        n = self.env.num_envs

        # reset grace: re-pin the freshly closed drawer while write timing settles
        gids = (self._grace > 0).nonzero(as_tuple=False).squeeze(-1)
        if len(gids):
            st = self.drawer_pose_at(torch.zeros(len(gids), device=dev), gids)
            self.drawer.write_root_state_to_sim(st, gids)
            self._grace[gids] -= 1

        disp = self.drawer_disp()
        v = self.drawer.data.root_lin_vel_w[:, 0]

        # latch state machine
        arm_now = (~self._released) & (disp > c.press_arm) & (v.abs() < c.arm_speed)
        self._armed |= arm_now
        rel_now = self._armed & (~self._released) & (disp < c.press_release)
        self._released |= rel_now

        # force law along the slide (world x; the drawer cannot rotate)
        lat = torch.clamp(-c.k_latch * disp, -c.latch_cap, c.latch_cap) \
            - torch.where(disp < -0.01, c.c_snap, c.c_press) * v
        ej = torch.clamp(c.k_eject * (-c.open_stop - disp), -c.eject_cap, c.eject_cap) \
            - c.c_eject * v
        f = torch.where(self._released, ej, lat)
        # one-sided end stops
        f = f - c.k_stop * (disp - c.press_stop).clamp(min=0.0)
        f = f - c.k_stop * (disp + c.open_stop).clamp(max=0.0)
        wrench = torch.zeros(n, 1, 3, device=dev)
        wrench[:, 0, 0] = f
        self.drawer.set_external_force_and_torque(wrench,
                                                  torch.zeros(n, 1, 3, device=dev))

        # latched progress (all physical)
        self._ejected_ever |= self._released & (disp < -c.open_gate)
        cz = (self.cube.data.root_pos_w - self.env_origins)[:, 2]
        self._lifted_ever |= cz > c.lift_z
        self._placed_ever |= self.cube_on_pad()

    # ----- reset --------------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Cabinet stays put (jointed pair, never teleported); drawer re-pinned CLOSED
        (follower-only, pure translation). Sample: pad polar slot + side, cube offset/yaw
        inside the drawer, decoy polar slot on the mirror flank + yaw."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # drawer: closed, zero velocity
        self.drawer.write_root_state_to_sim(
            self.drawer_pose_at(torch.zeros(m, device=dev), env_ids), env_ids)

        # pad: polar about the planned robot base, side sampled
        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           torch.ones(m, device=dev), -torch.ones(m, device=dev))
        b0, b1 = (math.radians(x) for x in c.pad_bearing)
        r0, r1 = c.pad_radius
        th = side * (b0 + (b1 - b0) * torch.rand(m, device=dev))
        r = r0 + (r1 - r0) * torch.rand(m, device=dev)
        ax, ay = c.base_anchor
        px = ax + r * torch.cos(th)
        py = ay + r * torch.sin(th)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = px
        st[:, 1] = py
        st[:, 2] = c.pad_t / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.pad.write_root_state_to_sim(st, env_ids)
        self._pad_xy[env_ids, 0] = px
        self._pad_xy[env_ids, 1] = py

        # red cube: inside the drawer cavity, near the front plate
        x0, x1 = c.cube_x_band
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.drawer_closed_x + x0 + (x1 - x0) * torch.rand(m, device=dev)
        st[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.cube_y_abs
        st[:, 2] = c.cube_rest_z + 0.002
        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.cube_yaw_deg) / 2
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        st[:, 0:3] += origin
        self.cube.write_root_state_to_sim(st, env_ids)

        # decoy: mirror flank of the pad, free yaw
        d0, d1 = (math.radians(x) for x in c.decoy_bearing)
        rd0, rd1 = c.decoy_radius
        thd = -side * (d0 + (d1 - d0) * torch.rand(m, device=dev))
        rd = rd0 + (rd1 - rd0) * torch.rand(m, device=dev)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = ax + rd * torch.cos(thd)
        st[:, 1] = ay + rd * torch.sin(thd)
        st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.decoy_jitter
        st[:, 2] = c.cube_size / 2 + 0.002
        half = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        st[:, 0:3] += origin
        self.decoy.write_root_state_to_sim(st, env_ids)

        self._armed[env_ids] = False
        self._released[env_ids] = False
        self._ejected_ever[env_ids] = False
        self._lifted_ever[env_ids] = False
        self._placed_ever[env_ids] = False
        self._grace[env_ids] = 2
        # zero the drawer's force buffer (a stale eject force would kick the fresh episode)
        n = self.env.num_envs
        self.drawer.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=dev), torch.zeros(n, 1, 3, device=dev))

    # ----- state (full, restorable) ---------------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"drawer": self.drawer, "cube": self.cube, "decoy": self.decoy,
                  "pad": self.pad}
        return {
            "bodies": {k: b.data.root_state_w[env_ids].clone() for k, b in bodies.items()},
            "pad_xy": self._pad_xy[env_ids].clone(),
            "latches": torch.stack([self._armed[env_ids], self._released[env_ids],
                                    self._ejected_ever[env_ids], self._lifted_ever[env_ids],
                                    self._placed_ever[env_ids]], dim=1).clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"drawer": self.drawer, "cube": self.cube, "decoy": self.decoy,
                  "pad": self.pad}
        for k, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][k], env_ids)
        self._pad_xy[env_ids] = state["pad_xy"]
        lat = state["latches"]
        self._armed[env_ids] = lat[:, 0]
        self._released[env_ids] = lat[:, 1]
        self._ejected_ever[env_ids] = lat[:, 2]
        self._lifted_ever[env_ids] = lat[:, 3]
        self._placed_ever[env_ids] = lat[:, 4]

    # ----- description ----------------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A low wooden cabinet stands on the floor. Its front is a single flush "
            "drawer face: a light-brown plate sitting inside the cabinet face with only "
            "a thin seam around it — NO handle, NO knob, nothing to hook or pinch, so "
            "the drawer cannot be pulled open. It is a PUSH-LATCH drawer (push-to-open): "
            f"press the plate straight INWARD about {c.press_arm * 1000:.0f}-"
            f"{c.press_stop * 1000:.0f} mm with a steady push, then withdraw; the latch "
            "clicks off and a spring slides the drawer out toward you on its own (about "
            f"{c.open_stop * 100:.0f} cm). If you merely tug the drawer it stays shut, "
            "and a shallow poke does not arm the latch.\n"
            "Sealed inside the drawer lies a small RED cube "
            f"({c.cube_size * 100:.1f} cm). On the floor to one side of the cabinet lies "
            "a flat GREEN square pad; on the opposite side lies a BLUE cube of the same "
            "size — a decoy, leave it alone.\n"
            "Goal: the RED cube resting centered ON the green pad, everything at rest. "
            "The drawer's final position does not matter. The order is enforced by "
            "physics: the cube is enclosed until the drawer ejects, so you must press "
            "the drawer face, let it spring open, then pick the red cube out of the "
            "open drawer and set it on the green pad."
        )

    # ----- rubric ---------------------------------------------------------------------------------------
    def score(self) -> torch.Tensor:
        """(N,) float in [0,1], latched stages of the demonstrated solution: 0.15 latch
        ever armed (a real deep press) + 0.15 drawer ever ejected + 0.20 cube ever lifted
        clear + 0.25 cube ever on the pad; exactly 1.0 iff success(). Null policy ~0; a
        yanked-open drawer earns nothing (arming requires the quasi-static press and
        ejection requires the released latch)."""
        s = (0.15 * self._armed.float()
             + 0.15 * self._ejected_ever.float()
             + 0.20 * self._lifted_ever.float()
             + 0.25 * self._placed_ever.float())
        return torch.where(self.success(),
                           torch.ones(self.env.num_envs, device=self.env.device),
                           s.clamp(0.0, 0.95))

    def success(self) -> torch.Tensor:
        """(N,) bool: the RED cube resting ON the green pad (xy within tolerance, z at
        pad-top rest height), settled — the physical terminal state. Unreachable without
        operating the latch: the cube starts sealed inside the closed drawer."""
        return self.cube_on_pad() & self.cube_settled()


register_env("simgen", lambda: EnvCfg(scene="latch_drawer", robot="null"))

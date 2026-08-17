"""TwistlockUnplugScene — turn the desk lamp OFF by unplugging it at the wall:
twist its childproof plug to the slot angle, pull it out — and leave the fan powered.

Derived from the RLBench `lamp_off` seed but STRATEGICALLY DIFFERENT (see TASK.md):
the seed's plan is one poke on a button that sits ON the lamp — one contact, one
straight push, no discrimination, no mechanism. Here the lamp HAS no switch: it is
powered through a CHILDPROOF TWIST-LOCK plug seated in a two-socket outlet box, and
the only way to turn it off is to cut power at the wall. That flips every element of
the plan: (1) the robot acts on the OUTLET, not the lamp — pressing the lamp does
nothing; (2) there are two identical black plugs, and WHICH one is the lamp's is
sampled fresh every episode, identified only by tracing the red cord from the lamp's
base (the blue cord belongs to the fan, which must STAY plugged in — a keep-alive
constraint the seed has no analogue of); (3) the extraction itself is a two-DOF
mechanism, not a push: each plug carries a key bar behind a slotted faceplate, and
the bar only passes when the plug has been ROTATED to the slot's angle (sampled
per episode, either direction) — pull without twist and the bar jams on the plate,
which is real collision geometry, not a rubric clause.

Mechanics (plain rigid bodies + authored USD joints, the door-hinge pattern): each
plug is ONE dynamic compound body (square cap + round shaft + key bar + prong
visuals) hung on a D6 joint (outlet -> plug) that frees exactly two DOF: translation
along the socket axis x (travel [0, 55 mm]) and rotation about it (+-60 deg). The
slotted faceplate is four SEPARATE kinematic boxes per socket, re-posed at reset to
the episode's slot angle; the joint pair (plug vs outlet) is collision-filtered but
the plate is a separate body, so bar-vs-plate contact is live — the interlock is
physical. `post_step` owns both plugs' wrench slots: it applies the `drive_f` /
`drive_t` buffers (solve.py's stand-in for the gripper's axial pull / wrist twist;
both are along the free axis, so the wrench never drifts with plug rotation) and
latches rubric progress.

Rubric (graded 0..1, latching transient achievement — anchored in the demonstrated
solve.py trajectory: align, pass, extract):
  - `align_latch` (0.25): best angular alignment of the LAMP plug's bar with its
    slot (1.0 within `align_tol_deg`); forced to 1.0 when the bar passes the plate;
  - `pass_latch`  (0.15): the bar has ever physically cleared the faceplate;
  - `ext_latch`   (0.30): best extraction fraction of the lamp plug (ext/ext_goal);
  - current success (0.30): lamp plug out >= `ext_goal`, fan plug still seated,
    everything settled -> score == 1.0 iff success(); ~0 for the null policy.
  - Unseating the FAN's plug (current state) multiplies the non-success score by
    0.25 — credit only evaporates under incorrect behavior.

Per-episode randomization (readback-verified in smoke): which socket powers the lamp
(cord routing swaps), each faceplate's slot angle (magnitude AND sign), lamp / fan
positions on the desk, and the cord's lay.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable
from robobench.core.registries import ENVS

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class TwistlockUnplugSceneCfg(BaseCfg):
    """Config for `TwistlockUnplugScene`. Geometry is derived once in `__post_init__` so the
    scene, the smoke AND the solver read the same numbers."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    ext_goal: float = tunable(0.040)  # lamp plug pulled out at least this far = unplugged
    ext_pass: float = tunable(0.024)  # bar has fully cleared the faceplate beyond this
    seat_tol: float = tunable(0.008)  # fan plug must stay within this of fully seated
    align_tol_deg: float = tunable(10.0)  # |twist - slot angle| within this = aligned
    settle_lin: float = tunable(0.03)  # max plug |lin vel| (m/s) when judging
    settle_ang: float = tunable(0.5)  # max plug |ang vel| (rad/s) when judging

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    slot_min_deg: float = tunable(30.0)  # slot angle magnitude sampled in [min, max]...
    slot_max_deg: float = tunable(50.0)  # ...with a random sign, per socket, per episode
    dev_x_range: tuple = tunable((-0.02, 0.16))  # lamp/fan base x band on the desk
    dev_y_lo: tuple = tunable((-0.34, -0.10))  # one device in the y<0 band...
    dev_y_hi: tuple = tunable((0.10, 0.34))  # ...the other in y>0 (sides randomized)
    cord_bulge: float = tunable(0.06)  # max sideways bow of a cord's lay (m)

    # --- tunable: plant ----------------------------------------------------------------------
    plug_mass: float = tunable(0.06)
    plug_lin_damp: float = tunable(4.0)  # plug body damping — parks the plug where released
    plug_ang_damp: float = tunable(6.0)

    # --- info: structure (env-local coordinates; sockets face +x, robot side) ----------------
    desk_center: tuple = info((0.05, 0.0, 0.36))
    desk_size: tuple = info((0.95, 1.00, 0.08))  # top at z = 0.40
    outlet_center_xy: tuple = info((-0.28, 0.0))
    outlet_size: tuple = info((0.06, 0.30, 0.16))  # face plane at x = -0.25
    socket_y: tuple = info((-0.07, 0.07))  # socket 0 / socket 1 axes
    socket_z_above_desk: float = info(0.11)  # socket axis height above the desk top
    # Plug, in its own frame (origin = cap centre, axis = +x = extraction direction):
    cap_size: float = info(0.026)  # square cap (the jaw target)
    shaft_r: float = info(0.008)  # round shaft — clears the 11 mm slot at ANY angle
    shaft_len: float = info(0.032)
    bar_half_len: float = info(0.030)  # key bar, along the plug-local z when untwisted
    bar_wt: float = info(0.008)  # bar width and thickness
    cap_from_face: float = info(0.043)  # seated cap centre sits this far off the face
    bar_from_face: float = info(0.007)  # seated bar centre sits this far off the face
    prong_len: float = info(0.020)  # visual prongs (collision-free), into the socket
    # Faceplate (four kinematic boxes; slot long axis at the episode's angle from vertical):
    plate_rear_from_face: float = info(0.014)  # 3 mm free travel before a locked bar jams
    plate_t: float = info(0.008)
    slot_half_len: float = info(0.036)
    slot_half_w: float = info(0.011)  # bar passes iff |twist - slot| <~ 13 deg (geometry)
    travel_max: float = info(0.110)  # D6 translation limit — generous headroom past
    # ext_goal: a limit bound near the release point injects push-back impulses via the
    # limit's contactDistance (the plug is "leashed", never fully free — see TASK.md)
    twist_lim_deg: float = info(60.0)  # D6 rotation limit (both signs)
    n_cord_segs: int = info(8)  # flat lay segments per cord (+1 fixed riser)
    cord_seg_len: float = info(0.10)
    cord_r: float = info(0.006)
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    desk_top_z: float = field(default=None, init=False)
    face_x: float = field(default=None, init=False)
    socket_z: float = field(default=None, init=False)
    cap_x0: float = field(default=None, init=False)  # seated cap centre x (env-local)
    plate_cx: float = field(default=None, init=False)  # plate centre x

    def __post_init__(self) -> None:
        self.desk_top_z = self.desk_center[2] + self.desk_size[2] / 2
        self.face_x = self.outlet_center_xy[0] + self.outlet_size[0] / 2
        self.socket_z = self.desk_top_z + self.socket_z_above_desk
        self.cap_x0 = self.face_x + self.cap_from_face
        self.plate_cx = self.face_x + self.plate_rear_from_face + self.plate_t / 2

    # -- shared geometry helpers (scene + smoke + solver read the same numbers) ---------------
    def plate_box_locals(self) -> list[tuple[tuple, tuple]]:
        """(centre, size) of the four faceplate boxes in the UNROTATED plate frame
        (slot long axis along z). Rotate about x by the slot angle at reset."""
        side_y = self.slot_half_w + 0.006
        end_z = self.slot_half_len + 0.006
        return [
            ((0.0, +side_y, 0.0), (self.plate_t, 0.012, 2 * self.slot_half_len + 0.032)),
            ((0.0, -side_y, 0.0), (self.plate_t, 0.012, 2 * self.slot_half_len + 0.032)),
            ((0.0, 0.0, +end_z), (self.plate_t, 2 * self.slot_half_w, 0.012)),
            ((0.0, 0.0, -end_z), (self.plate_t, 2 * self.slot_half_w, 0.012)),
        ]


def _quat_x(rad: torch.Tensor) -> torch.Tensor:
    """(N,) angle about +x -> (N, 4) wxyz."""
    half = rad / 2
    q = torch.zeros(rad.shape[0], 4, device=rad.device)
    q[:, 0] = torch.cos(half)
    q[:, 1] = torch.sin(half)
    return q


# ----- scene -----------------------------------------------------------------------------------
class TwistlockUnplugScene(BaseScene):
    cfg: TwistlockUnplugSceneCfg

    def __init__(self, cfg: TwistlockUnplugSceneCfg | None = None) -> None:
        super().__init__(cfg or TwistlockUnplugSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, kinematic desk + outlet box, two dynamic twist-lock plugs (compound
        children authored in bind), eight kinematic faceplate boxes (re-posed at reset to the
        episode's slot angles), the dynamic lamp and fan props, and the kinematic cord-lay
        segments (visual identification, collision-free)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        wood = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.48, 0.35, 0.20))
        cream = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.89, 0.84))
        black = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.10, 0.11))
        plate_gray = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.36, 0.38))
        red = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.60, 0.08, 0.08))
        blue = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.15, 0.60))
        brass = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.58, 0.22))
        teal = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.45, 0.45))
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        # post_step drives the plugs with external wrenches that do NOT wake a sleeping
        # body — sleep_threshold=0 keeps the plant live.
        live = dict(sleep_threshold=0.0, stabilization_threshold=0.0)

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
            "desk": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Desk",
                spawn=sim_utils.CuboidCfg(
                    size=c.desk_size, rigid_props=kin, collision_props=coll,
                    visual_material=wood),
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.desk_center),
            ),
            "outlet": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Outlet",
                spawn=sim_utils.CuboidCfg(
                    size=c.outlet_size, rigid_props=kin, collision_props=coll,
                    visual_material=cream),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.outlet_center_xy[0], c.outlet_center_xy[1],
                         c.desk_top_z + c.outlet_size[2] / 2)),
            ),
        }
        # --- the two plugs: root prim = the square cap (fixture children authored in bind) ---
        for s in (0, 1):
            out[f"plug{s}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + f"Plug{s}",
                spawn=sim_utils.CuboidCfg(
                    size=(c.cap_size,) * 3,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        linear_damping=c.plug_lin_damp,
                        angular_damping=c.plug_ang_damp,
                        **live),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.plug_mass),
                    collision_props=coll,
                    visual_material=black,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cap_x0, c.socket_y[s], c.socket_z)),
            )
            # --- faceplate: four kinematic boxes per socket (re-posed at reset) ---
            for k, (_loc, size) in enumerate(c.plate_box_locals()):
                out[f"plate{s}_{k}"] = RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/" + f"Plate{s}_{k}",
                    spawn=sim_utils.CuboidCfg(
                        size=size, rigid_props=kin, collision_props=coll,
                        visual_material=plate_gray),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(c.plate_cx, c.socket_y[s], c.socket_z)),
                )
        # --- the two appliances (dynamic props; fixture children authored in bind) ---
        for name, mat in (("lamp", brass), ("fan", teal)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.title(),
                spawn=sim_utils.CylinderCfg(
                    radius=0.055, height=0.020,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.5, angular_damping=0.5, **live),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.2),
                    collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8, dynamic_friction=0.7, restitution=0.05),
                    visual_material=mat,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.08, -0.22 if name == "lamp" else 0.22, c.desk_top_z + 0.011)),
            )
        # --- cord segments: kinematic, COLLISION-FREE (visual identification only) ---
        for cord, mat in (("cord_lamp", red), ("cord_fan", blue)):
            for k in range(c.n_cord_segs + 1):  # +1: the fixed riser up the outlet face
                out[f"{cord}_{k}"] = RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/" + f"{cord.title().replace('_', '')}{k}",
                    spawn=sim_utils.CuboidCfg(
                        size=(c.cord_r, c.cord_seg_len, c.cord_r),
                        rigid_props=kin, visual_material=mat),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(0.0, 0.0, c.desk_top_z + c.cord_r / 2)),
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
        super().bind(env)
        n = env.num_envs
        dev = env.device
        c = self.cfg
        self.plugs: list[RigidObject] = [env.iscene[f"plug{s}"] for s in (0, 1)]
        self.plates: list[list[RigidObject]] = [
            [env.iscene[f"plate{s}_{k}"] for k in range(4)] for s in (0, 1)]
        self.lamp: RigidObject = env.iscene["lamp"]
        self.fan: RigidObject = env.iscene["fan"]
        self.cords: dict[str, list[RigidObject]] = {
            nm: [env.iscene[f"{nm}_{k}"] for k in range(c.n_cord_segs + 1)]
            for nm in ("cord_lamp", "cord_fan")}
        self.env_origins = env.iscene.env_origins
        self._author_plug_fixture()
        self._author_d6()
        # Episode state.
        self.lamp_socket = torch.zeros(n, dtype=torch.long, device=dev)  # lamp's socket idx
        self.slot_deg = torch.zeros(n, 2, device=dev)  # slot angle per socket (signed deg)
        self.align_latch = torch.zeros(n, device=dev)
        self.pass_latch = torch.zeros(n, device=dev)
        self.ext_latch = torch.zeros(n, device=dev)
        # External drive input (solve.py and smoke probes write; post_step consumes + owns
        # both plugs' wrench slots — never call set_external_force_and_torque directly).
        self.drive_f = torch.zeros(n, 2, device=dev)  # axial pull (N, + = out) per socket
        self.drive_t = torch.zeros(n, 2, device=dev)  # twist torque (N*m about +x) per socket

    def _author_plug_fixture(self) -> None:
        """Compound children on each plug cap (env_0 only — env_1.. compose from env_0 by
        reference; authored idempotently): round shaft (collides — clears the slot at any
        angle), key bar (collides — THE interlock), two prong visuals (collision-free)."""
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath("/World/envs/env_0/Plug0/bar").IsValid():
            return

        def box(root: str, name: str, center: tuple, size: tuple, color: tuple,
                collide: bool = True) -> None:
            cube = UsdGeom.Cube.Define(stage, f"{root}/{name}")
            cube.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(cube.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(*center))
            xf.AddScaleOp().Set(Gf.Vec3f(*size))
            cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            if collide:
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                px = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
                px.CreateContactOffsetAttr(c.contact_offset)
                px.CreateRestOffsetAttr(0.0)

        def cyl_x(root: str, name: str, center: tuple, radius: float, length: float,
                  color: tuple) -> None:
            cy = UsdGeom.Cylinder.Define(stage, f"{root}/{name}")
            cy.CreateAxisAttr("X")
            cy.CreateRadiusAttr(radius)
            cy.CreateHeightAttr(length)
            xf = UsdGeom.Xformable(cy.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(*center))
            cy.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            UsdPhysics.CollisionAPI.Apply(cy.GetPrim())
            px = PhysxSchema.PhysxCollisionAPI.Apply(cy.GetPrim())
            px.CreateContactOffsetAttr(c.contact_offset)
            px.CreateRestOffsetAttr(0.0)

        dark = (0.10, 0.10, 0.11)
        steel = (0.75, 0.75, 0.78)
        # shaft spans plug-local x [-cap_from_face - 0.002, -cap_from_face - 0.002 + shaft_len]
        # = [-0.045, -0.013]: from 2 mm behind the face plane (joint pair is collision-
        # filtered, so it may enter the outlet) through the plate hole to the cap rear.
        shaft_cx = -(c.cap_from_face + 0.002) + c.shaft_len / 2
        for s in (0, 1):
            root = f"/World/envs/env_0/Plug{s}"
            cyl_x(root, "shaft", (shaft_cx, 0.0, 0.0), c.shaft_r, c.shaft_len, dark)
            # bar: the key — along plug-local z when untwisted
            box(root, "bar", (c.bar_from_face - c.cap_from_face, 0.0, 0.0),
                (c.bar_wt, c.bar_wt, 2 * c.bar_half_len), dark)
            # prongs: visual only, at the bar's azimuth (+-8 mm along local z)
            for pz in (+0.008, -0.008):
                box(root, f"prong{'p' if pz > 0 else 'n'}",
                    (-c.cap_from_face - c.prong_len / 2, 0.0, pz),
                    (c.prong_len, 0.004, 0.004), steel, collide=False)
        # --- appliance dressing (visual-only children; roots are the base cylinders) ---
        lamp_root = "/World/envs/env_0/Lamp"
        box(lamp_root, "pole", (0.0, 0.0, 0.15), (0.018, 0.018, 0.28), (0.72, 0.58, 0.22),
            collide=False)
        box(lamp_root, "shade", (0.0, 0.0, 0.31), (0.11, 0.11, 0.07), (0.85, 0.70, 0.35),
            collide=False)
        sp = UsdGeom.Sphere.Define(stage, f"{lamp_root}/bulb")
        sp.CreateRadiusAttr(0.025)
        UsdGeom.Xformable(sp.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.265))
        sp.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.95, 0.45)])
        fan_root = "/World/envs/env_0/Fan"
        box(fan_root, "column", (0.0, 0.0, 0.10), (0.024, 0.024, 0.18), (0.10, 0.45, 0.45),
            collide=False)
        fd = UsdGeom.Cylinder.Define(stage, f"{fan_root}/disc")
        fd.CreateAxisAttr("X")
        fd.CreateRadiusAttr(0.085)
        fd.CreateHeightAttr(0.02)
        UsdGeom.Xformable(fd.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.26))
        fd.CreateDisplayColorAttr([Gf.Vec3f(0.16, 0.55, 0.55)])

    def _author_d6(self) -> None:
        """Per env and per socket: a D6 joint outlet -> plug freeing exactly transX
        (travel [0, travel_max]) and rotX (+- twist_lim_deg). The joint pair never
        collides; the faceplate is a SEPARATE body, so the bar-vs-plate interlock stays
        live contact."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        oc = (c.outlet_center_xy[0], c.outlet_center_xy[1], c.desk_top_z + c.outlet_size[2] / 2)
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            for s in (0, 1):
                j = UsdPhysics.Joint.Define(stage, f"{base}/plug_joint{s}")
                j.CreateBody0Rel().SetTargets([f"{base}/Outlet"])
                j.CreateBody1Rel().SetTargets([f"{base}/Plug{s}"])
                j.CreateCollisionEnabledAttr(False)
                j.CreateLocalPos0Attr(Gf.Vec3f(
                    c.face_x - oc[0], c.socket_y[s] - oc[1], c.socket_z - oc[2]))
                j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLocalPos1Attr(Gf.Vec3f(-c.cap_from_face, 0.0, 0.0))
                j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                for axis in ("transY", "transZ", "rotY", "rotZ"):
                    lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), axis)
                    lim.CreateLowAttr(1.0)  # low > high = locked
                    lim.CreateHighAttr(-1.0)
                lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), "transX")
                lim.CreateLowAttr(0.0)
                lim.CreateHighAttr(c.travel_max)
                lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), "rotX")
                lim.CreateLowAttr(-c.twist_lim_deg)
                lim.CreateHighAttr(c.twist_lim_deg)

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the lamp's socket, both slot angles (magnitude + sign),
        appliance positions; seat both plugs (twist 0); pose the faceplates at their slot
        angles; lay the cords; clear latches and drives."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # NOTE: the FIRST randint draw after manual_seed is degenerate on this stack
        # (seeds 0..4 all gave 1); the rand stream varies fine, so the socket choice
        # is derived from a later rand draw instead.
        sign = torch.where(torch.rand(m, 2, device=dev) < 0.5, -1.0, 1.0)
        mag = c.slot_min_deg + (c.slot_max_deg - c.slot_min_deg) * torch.rand(m, 2, device=dev)
        self.lamp_socket[env_ids] = (torch.rand(m, device=dev) < 0.5).long()
        self.slot_deg[env_ids] = sign * mag
        self.align_latch[env_ids] = 0.0
        self.pass_latch[env_ids] = 0.0
        self.ext_latch[env_ids] = 0.0
        self.drive_f[env_ids] = 0.0
        self.drive_t[env_ids] = 0.0

        # --- plugs: seated, untwisted ---
        for s in (0, 1):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = c.cap_x0
            st[:, 1] = c.socket_y[s]
            st[:, 2] = c.socket_z
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            self.plugs[s].write_root_state_to_sim(st, env_ids)

        # --- faceplates: four boxes per socket, rotated to the slot angle about +x ---
        for s in (0, 1):
            a = torch.deg2rad(self.slot_deg[env_ids, s])
            q = _quat_x(a)
            ca, sa = torch.cos(a), torch.sin(a)
            for k, (loc, _size) in enumerate(c.plate_box_locals()):
                ly, lz = loc[1], loc[2]
                st = torch.zeros(m, 13, device=dev)
                st[:, 0] = c.plate_cx + loc[0]
                st[:, 1] = c.socket_y[s] + ca * ly - sa * lz
                st[:, 2] = c.socket_z + sa * ly + ca * lz
                st[:, 3:7] = q
                st[:, 0:3] += origin
                self.plates[s][k].write_root_state_to_sim(st, env_ids)

        # --- appliances: lamp and fan on opposite y bands (sides randomized) ---
        lamp_hi = torch.rand(m, device=dev) < 0.5
        xr = c.dev_x_range
        for name, body in (("lamp", self.lamp), ("fan", self.fan)):
            hi = lamp_hi if name == "lamp" else ~lamp_hi
            ylo = torch.tensor(c.dev_y_lo, device=dev)
            yhi = torch.tensor(c.dev_y_hi, device=dev)
            band = torch.where(hi.unsqueeze(1), yhi, ylo)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = xr[0] + (xr[1] - xr[0]) * torch.rand(m, device=dev)
            st[:, 1] = band[:, 0] + (band[:, 1] - band[:, 0]) * torch.rand(m, device=dev)
            st[:, 2] = c.desk_top_z + 0.011
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- cords: flat quadratic lay from each appliance base to ITS socket + riser ---
        for nm, body in (("cord_lamp", self.lamp), ("cord_fan", self.fan)):
            sock = self.lamp_socket[env_ids] if nm == "cord_lamp" \
                else 1 - self.lamp_socket[env_ids]
            sy = torch.where(sock == 0,
                             torch.full((m,), c.socket_y[0], device=dev),
                             torch.full((m,), c.socket_y[1], device=dev))
            p0 = body.data.root_pos_w[env_ids, 0:2] - origin[:, 0:2]  # appliance base xy
            p2 = torch.stack([torch.full((m,), c.face_x + 0.008, device=dev), sy], dim=1)
            mid = (p0 + p2) / 2
            d = p2 - p0
            L = d.norm(dim=1, keepdim=True).clamp(min=1e-6)
            perp = torch.stack([-d[:, 1], d[:, 0]], dim=1) / L
            bulge = (torch.rand(m, 1, device=dev) * 2 - 1) * c.cord_bulge
            p1 = mid + perp * bulge
            zc = c.desk_top_z + c.cord_r / 2 + 0.001
            for k in range(c.n_cord_segs):
                t = (k + 0.5) / c.n_cord_segs
                pt = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
                tan = 2 * (1 - t) * (p1 - p0) + 2 * t * (p2 - p1)
                yaw = torch.atan2(tan[:, 1], tan[:, 0]) - math.pi / 2  # box long axis = +y
                st = torch.zeros(m, 13, device=dev)
                st[:, 0:2] = pt
                st[:, 2] = zc
                st[:, 3] = torch.cos(yaw / 2)
                st[:, 6] = torch.sin(yaw / 2)
                st[:, 0:3] += origin
                self.cords[nm][k].write_root_state_to_sim(st, env_ids)
            # riser: fixed vertical segment up the outlet face to below the socket
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = c.face_x + 0.004
            st[:, 1] = sy
            st[:, 2] = c.desk_top_z + 0.05
            st[:, 3] = math.cos(math.pi / 4)  # rot x by 90 deg: long axis y -> z
            st[:, 4] = math.sin(math.pi / 4)
            st[:, 0:3] += origin
            self.cords[nm][c.n_cord_segs].write_root_state_to_sim(st, env_ids)

    # ----- readings -----------------------------------------------------------------------------
    def plug_ext(self) -> torch.Tensor:
        """(N, 2) extraction of each plug along +x (0 = seated)."""
        e = [(p.data.root_pos_w[:, 0] - self.env_origins[:, 0]) - self.cfg.cap_x0
             for p in self.plugs]
        return torch.stack(e, dim=1)

    def plug_twist(self) -> torch.Tensor:
        """(N, 2) twist about +x in rad (plug orientation is pure x-rotation by the D6)."""
        t = []
        for p in self.plugs:
            q = p.data.root_quat_w
            t.append(2.0 * torch.atan2(q[:, 1], q[:, 0]))
        return torch.stack(t, dim=1)

    def _gather_lamp(self, v2: torch.Tensor) -> torch.Tensor:
        return v2.gather(1, self.lamp_socket.unsqueeze(1)).squeeze(1)

    def lamp_ext(self) -> torch.Tensor:
        return self._gather_lamp(self.plug_ext())

    def decoy_ext(self) -> torch.Tensor:
        return self.plug_ext().gather(1, (1 - self.lamp_socket).unsqueeze(1)).squeeze(1)

    def lamp_align_err(self) -> torch.Tensor:
        """(N,) |twist - slot angle| of the LAMP plug (rad)."""
        tw = self._gather_lamp(self.plug_twist())
        a = torch.deg2rad(self._gather_lamp(self.slot_deg))
        return (tw - a).abs()

    def settled(self) -> torch.Tensor:
        """(N,) bool: both plugs at rest (lin + ang)."""
        c = self.cfg
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for p in self.plugs:
            ok &= (p.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
                & (p.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
        return ok

    def decoy_seated(self) -> torch.Tensor:
        return self.decoy_ext() < self.cfg.seat_tol

    def success(self) -> torch.Tensor:
        """(N,) bool: LAMP plug pulled out past `ext_goal`, FAN plug still seated,
        everything settled (current, physical state)."""
        return (self.lamp_ext() >= self.cfg.ext_goal) & self.decoy_seated() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.25 * align_latch + 0.15 * pass_latch + 0.30 * ext_latch
        + 0.30 * current success; the non-success part is quartered while the FAN's plug
        is unseated. Exactly 1.0 iff success() (success implies all three latches are 1);
        ~0 for the null policy."""
        base = 0.25 * self.align_latch + 0.15 * self.pass_latch + 0.30 * self.ext_latch
        base = torch.where(self.decoy_seated(), base, base * 0.25)
        return base + 0.30 * self.success().float()

    # ----- step-coupled mechanics (every substep) ------------------------------------------------
    def post_step(self) -> None:
        """Plug plant: consume the `drive_f` / `drive_t` buffers (owns both plugs' wrench
        slots; drive is along the free axis, so plug rotation never skews it), then latch
        rubric progress for the LAMP plug."""
        n = self.env.num_envs
        dev = self.env.device
        c = self.cfg
        for s in (0, 1):
            f = torch.zeros(n, 1, 3, device=dev)
            t = torch.zeros(n, 1, 3, device=dev)
            f[:, 0, 0] = self.drive_f[:, s]
            t[:, 0, 0] = self.drive_t[:, s]
            self.plugs[s].set_external_force_and_torque(f, t)

        ext = self.lamp_ext()
        err = self.lamp_align_err()
        a_mag = torch.deg2rad(self._gather_lamp(self.slot_deg)).abs()
        tol = math.radians(c.align_tol_deg)
        align = ((a_mag - err) / (a_mag - tol).clamp(min=1e-6)).clamp(0.0, 1.0)
        passed = (ext > c.ext_pass).float()
        align = torch.maximum(align, passed)  # a bar that PASSED was aligned when it mattered
        extf = (ext / c.ext_goal).clamp(0.0, 1.0)
        # A diverged substep must not latch: torch.maximum propagates NaN. Garbage earns 0.
        align = torch.nan_to_num(align, nan=0.0, posinf=0.0, neginf=0.0)
        passed = torch.nan_to_num(passed, nan=0.0, posinf=0.0, neginf=0.0)
        extf = torch.nan_to_num(extf, nan=0.0, posinf=0.0, neginf=0.0)
        self.align_latch = torch.maximum(self.align_latch, align)
        self.pass_latch = torch.maximum(self.pass_latch, passed)
        self.ext_latch = torch.maximum(self.ext_latch, extf)

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = self._bodies()
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("lamp_socket", "slot_deg", "align_latch", "pass_latch",
                               "ext_latch", "drive_f", "drive_t")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = self._bodies()
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    def _bodies(self) -> dict[str, Any]:
        out = {"plug0": self.plugs[0], "plug1": self.plugs[1],
               "lamp": self.lamp, "fan": self.fan}
        for s in (0, 1):
            for k in range(4):
                out[f"plate{s}_{k}"] = self.plates[s][k]
        for nm, segs in self.cords.items():
            for k, b in enumerate(segs):
                out[f"{nm}_{k}"] = b
        return out

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A desk holds a glowing brass lamp (yellow bulb, no switch anywhere on it), a "
            f"teal desk fan, and — at the rear of the desk, facing you — a cream outlet box "
            f"with two identical childproof sockets side by side. Each socket holds a black "
            f"plug with a square {c.cap_size * 100:.1f} cm cap pointing at you; behind each "
            f"cap, a dark key bar rides behind a gray slotted faceplate. A RED cord runs "
            f"along the desk from the LAMP's base to one socket; a BLUE cord runs from the "
            f"FAN's base to the other. Which socket powers the lamp changes every episode — "
            f"trace the red cord from the lamp to find its plug.\n"
            f"Goal: turn the lamp OFF by unplugging it at the outlet. The plugs are "
            f"twist-locked: grasp the lamp's plug cap and ROTATE it about the socket axis "
            f"until its key bar lines up with the faceplate's angled slot (the slot's tilt "
            f"— up to {c.slot_max_deg:.0f} degrees clockwise or counter-clockwise, sampled "
            f"per episode — is visible on the faceplate, and the bar stays parallel to the "
            f"cap's flat sides; alignment within about {c.align_tol_deg:.0f} degrees is "
            f"enough), then PULL the plug straight out at least "
            f"{c.ext_goal * 100:.0f} cm and let go. Pulling without rotating jams the bar "
            f"against the plate — the plug will not come out. The FAN must stay powered: "
            f"its plug has to remain fully seated in its socket. Pulling the wrong plug, "
            f"leaving the lamp's plug less than {c.ext_goal * 100:.0f} cm out, or unseating "
            f"the fan's plug fails the task. Nothing on the lamp itself needs touching."
        )

    def instruction(self) -> str:
        return (
            "Unplug the desk lamp at the outlet box: trace the red cord from the lamp to "
            "its plug, rotate that plug until its key bar aligns with the slot in the "
            "faceplate, then pull it straight out of the socket. Do not unplug the fan — "
            "its plug must stay seated."
        )


# Guarded registration: the forge may import this module under two names.
if "twistlock_unplug" not in SCENES.list():
    SCENES.register("twistlock_unplug", TwistlockUnplugScene)
if "simgen.twistlock_unplug" not in ENVS.list():
    register_env("simgen", lambda: EnvCfg(scene="twistlock_unplug", robot="null"))

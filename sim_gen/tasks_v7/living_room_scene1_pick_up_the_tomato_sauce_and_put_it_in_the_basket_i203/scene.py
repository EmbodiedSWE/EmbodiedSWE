"""PortDropboxScene — post the red sauce can end-on through the drop-box's side port.

Derived from the LIBERO seed `living_room_scene1_pick_up_the_tomato_sauce_and_put_it_in
_the_basket` but STRATEGICALLY DIFFERENT (see TASK.md): the seed is a one-stage free
pick-and-place — grasp a can standing in the open, carry it over an OPEN-topped basket,
let go, bounding-box check. Here the container is SEALED on top (a full roof) and its
only entry is a square SIDE PORT partway up one wall, fronted by an exterior loading
tray with guide rails. The port admits the can END-ON only: the aperture is 80 mm
square, the can is 64 mm across but 110 mm long, so it passes along its axis and is
geometrically rejected side-on; the roof rejects the seed's top-drop outright (a can
released above the box settles ON the roof — smoke constructs exactly that). The plan
is therefore reorientation + threaded insertion: lay the upright can HORIZONTAL, stage
it on the tray between the rails aligned with the port axis, and push it axially
through the port tunnel until its centre of mass crosses the inner wall face — at which
point it tips and gravity drops it irreversibly onto the box floor. A beige distractor
can of identical shape must be left OUTSIDE (wrong-object clause).

Mechanics: the box + tray is ONE kinematic compound body (base-slab root, wall / roof /
tray / rail child colliders authored in bind — idempotent, per env), re-posed per
episode with sampled yaw + xy jitter. The two cans are plain dynamic cylinders.
`post_step` owns both cans' wrench slots: it applies the `push_f` buffer (solve.py's
stand-in for the Franka fingertip pushing the can's rear face) and latches rubric
progress.

Rubric (graded 0..1, anchored in the demonstrated solve.py trajectory):
  - 0.25 * staged_latch: the red can has RESTED (slow-gated) lying horizontal on the
    tray, aligned with the port axis — the reorientation stage;
  - 0.35 * insert_latch: best axial penetration of the red can's CoM toward the
    commitment line 2 mm past the inner wall face (gated: aligned, in the port band,
    slower than `insert_speed`), latched running max — full credit only once the can
    is tipping-committed and the insertion is irreversible;
  - 1.0 iff success(): the red can PHYSICALLY inside the box (bin-frame footprint,
    centre BELOW the port sill — a can on the roof, on the tray or straddling the sill
    never counts), the beige can NOT inside, red can settled. Otherwise the partial
    credit is capped at 0.60. ~0 for the null policy; latched credit never evaporates.

Per-episode randomization (readback-verified in smoke): box yaw (+-15 deg) + xy jitter,
and both cans' ground poses — the red/beige SIDES SWAP 50/50 and each position is
jittered, so "the can on the left" is not memorizable.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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
class PortDropboxSceneCfg(BaseCfg):
    """Config for `PortDropboxScene`. Geometry is derived once in `__post_init__` so the
    scene, the smoke AND the solver read the same numbers. All heights are in the BIN
    GROUND frame (z above the ground plane); the kinematic root's body origin sits at
    the base-slab centre, z = slab_t / 2."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    # Settledness is judged by a POSITION-DELTA streak, not velocity readback: under an
    # external-wrench plant the root velocity readback is phantom (noisy while positions
    # are frozen), so a velocity gate would never pass. 0.3 mm/substep at 120 Hz bounds
    # real motion at 0.036 m/s; the streak requires it held ~0.4 s straight.
    still_tol: float = tunable(0.0003)  # red-can max |dpos| per substep to count as still (m)
    still_steps: int = tunable(45)  # consecutive still substeps required
    staged_speed: float = tunable(0.10)  # staged latch only below this speed (slow gate)
    insert_speed: float = tunable(0.60)  # insert latch only below this speed (no ballistic
    # credit; generous enough that the sill-edge contact jolt near commitment cannot
    # drop the final gated sample — the positional gates do the threading work)
    inside_margin: float = tunable(0.03)  # inside = centre below sill_z - this (m)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    bin_yaw_deg: float = tunable(15.0)  # box yaw sampled in +-this
    bin_jitter: float = tunable(0.025)  # box xy jitter (m)
    bin_pos: tuple = tunable((0.42, 0.0))  # nominal box centre (env-local xy)
    can_x_range: tuple = tunable((-0.08, 0.03))  # can ground zone, x (clear of the tray)
    can_y_band: tuple = tunable((0.07, 0.20))  # |y| band; red/beige sides swap 50/50

    # --- tunable: plant ----------------------------------------------------------------------
    can_mass: float = tunable(0.35)  # a full 400 g-class sauce can
    can_lin_damp: float = tunable(0.05)
    can_ang_damp: float = tunable(0.05)

    # --- info: structure (bin ground frame; port faces bin-local -x) -------------------------
    can_r: float = info(0.032)
    can_l: float = info(0.110)
    wall_t: float = info(0.02)
    in_half: float = info(0.12)  # interior half-extent (x and y)
    slab_t: float = info(0.02)  # base slab (the root prim); interior floor top
    wall_top: float = info(0.30)  # walls end here; roof slab sits on top
    roof_t: float = info(0.02)
    port_half_w: float = info(0.04)  # aperture |y| < this (80 mm wide)
    port_z0: float = info(0.14)  # aperture bottom (the sill)
    port_z1: float = info(0.22)  # aperture top (the lintel)
    tray_len: float = info(0.17)  # exterior loading tray, along -x from the outer face
    tray_w: float = info(0.12)
    tray_top: float = info(0.141)  # 1 mm above the sill: a step DOWN into the port, no catch
    rail_t: float = info(0.02)
    rail_h: float = info(0.03)
    rail_gap_half: float = info(0.041)  # rail inner faces at +-this (channel 82 mm)
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    out_half: float = field(default=None, init=False)  # outer half-extent = in_half + wall_t
    sill_z: float = field(default=None, init=False)
    port_cz: float = field(default=None, init=False)
    outer_face: float = field(default=None, init=False)  # bin-local x of the outer wall face
    inner_face: float = field(default=None, init=False)
    tray_cx: float = field(default=None, init=False)
    com_start: float = field(default=None, init=False)  # insert progress 0 (CoM local x)
    com_full: float = field(default=None, init=False)  # insert progress 1: tipping-committed
    stage_x: tuple = field(default=None, init=False)  # staged window for the can CoM (local x)
    z_inside_max: float = field(default=None, init=False)
    can_stage_z: float = field(default=None, init=False)  # CoM height lying on the tray

    def __post_init__(self) -> None:
        self.out_half = self.in_half + self.wall_t
        self.sill_z = self.port_z0
        self.port_cz = (self.port_z0 + self.port_z1) / 2
        self.outer_face = -self.out_half
        self.inner_face = -self.in_half
        self.tray_cx = -(self.out_half + self.tray_len / 2)
        self.com_start = -0.20  # nose (= CoM + half_l) at the tunnel mouth region
        # Full insert credit once the CoM is past the support edge at the inner wall
        # face — the can is tipping and the insertion is irreversible. Kept just 2 mm
        # past the edge: further in, the tipping pitch breaks the `lying` gate before
        # the last sample can latch.
        self.com_full = self.inner_face + 0.002
        self.stage_x = (-self.out_half - self.tray_len + 0.01, -0.16)
        self.z_inside_max = self.sill_z - self.inside_margin
        self.can_stage_z = self.tray_top + self.can_r


def _quat_z(rad: torch.Tensor) -> torch.Tensor:
    """(N,) angle about +z -> (N, 4) wxyz."""
    half = rad / 2
    q = torch.zeros(rad.shape[0], 4, device=rad.device)
    q[:, 0] = torch.cos(half)
    q[:, 3] = torch.sin(half)
    return q


def _quat_lying(yaw: torch.Tensor) -> torch.Tensor:
    """(N,) yaw -> (N, 4) wxyz for qz(yaw) * qy(90 deg): cylinder local +z lies along the
    world direction (cos yaw, sin yaw, 0) — a can lying flat, axis at that bearing."""
    c45 = math.cos(math.pi / 4)
    half = yaw / 2
    q = torch.zeros(yaw.shape[0], 4, device=yaw.device)
    q[:, 0] = torch.cos(half) * c45
    q[:, 1] = -torch.sin(half) * c45
    q[:, 2] = torch.cos(half) * c45
    q[:, 3] = torch.sin(half) * c45
    return q


# ----- scene -----------------------------------------------------------------------------------
class PortDropboxScene(BaseScene):
    cfg: PortDropboxSceneCfg

    def __init__(self, cfg: PortDropboxSceneCfg | None = None) -> None:
        super().__init__(cfg or PortDropboxSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic drop-box root (base slab; walls / roof / tray /
        rails are child colliders authored in bind), and the two dynamic cans."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        green = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.38, 0.16))
        # post_step drives the cans with external wrenches that do NOT wake a sleeping
        # body — sleep_threshold=0 keeps them live.
        can_props = dict(
            max_depenetration_velocity=0.5,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
            linear_damping=c.can_lin_damp,
            angular_damping=c.can_ang_damp,
            sleep_threshold=0.0,
            stabilization_threshold=0.0,
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
            # The drop-box root: the base slab. Everything else rides on it as authored
            # child colliders, so one root-state write re-poses the whole fixture.
            "bin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bin",
                spawn=sim_utils.CuboidCfg(
                    size=(2 * c.out_half, 2 * c.out_half, c.slab_t),
                    rigid_props=kin, collision_props=coll, visual_material=green),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bin_pos[0], c.bin_pos[1], c.slab_t / 2)),
            ),
        }
        for name, color, y0 in (("red_can", (0.75, 0.06, 0.06), 0.14),
                                ("beige_can", (0.82, 0.72, 0.55), -0.14)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + ("RedCan" if name == "red_can" else "BeigeCan"),
                spawn=sim_utils.CylinderCfg(
                    radius=c.can_r, height=c.can_l, axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(**can_props),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.can_mass),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-0.02, y0, c.can_l / 2 + 0.003)),
            )
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # Wrench plant: without this, external forces under-apply across TGS
                # iterations (IsaacLab's startup warning names this flag — treat as fatal).
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

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n = env.num_envs
        dev = env.device
        c = self.cfg
        self.bin: RigidObject = env.iscene["bin"]
        self.red: RigidObject = env.iscene["red_can"]
        self.beige: RigidObject = env.iscene["beige_can"]
        self.env_origins = env.iscene.env_origins
        self._author_box()
        # Episode state (sampled at reset; the kinematic box never moves afterwards).
        self.bin_xy = torch.zeros(n, 2, device=dev)
        self.bin_xy[:, 0] = c.bin_pos[0]
        self.bin_xy[:, 1] = c.bin_pos[1]
        self.bin_yaw = torch.zeros(n, device=dev)
        self.red0 = torch.zeros(n, 2, device=dev)  # sampled ground xy (readback checks)
        self.beige0 = torch.zeros(n, 2, device=dev)
        self.staged_latch = torch.zeros(n, device=dev)
        self.insert_latch = torch.zeros(n, device=dev)
        # Position-delta stillness streak for the red can (velocity readback is phantom
        # under the wrench plant; teleports reset the streak automatically via the jump).
        self.still_streak = torch.zeros(n, device=dev)
        self.red_prev = torch.zeros(n, 3, device=dev)
        # External drive input (solve.py and smoke probes write; post_step consumes + owns
        # both cans' wrench slots — never call set_external_force_and_torque directly).
        self.push_f = torch.zeros(n, 2, 3, device=dev)  # [red, beige] world-frame force (N)

    def _author_box(self) -> None:
        """Child colliders of the bin root, per env, authored idempotently: walls (the
        front one split around the port aperture), roof, exterior loading tray + guide
        rails, plus a white visual frame around the port. Offsets are in the ROOT BODY
        frame (origin at the base-slab centre, slab_t/2 above ground)."""
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        zb = c.slab_t / 2  # ground -> body-frame z shift
        green = (0.10, 0.38, 0.16)
        gray = (0.55, 0.55, 0.58)
        dark = (0.20, 0.20, 0.22)
        white = (0.95, 0.95, 0.95)
        wt2 = c.wall_t / 2
        wall_cx = c.in_half + wt2  # wall centreline |x| or |y|
        wall_h = c.wall_top - c.slab_t
        wall_cz = c.slab_t + wall_h / 2 - zb
        jamb_w = c.out_half - c.port_half_w  # width of each front-wall jamb piece
        port_h = c.port_z1 - c.port_z0

        parts: list[tuple[str, tuple, tuple, tuple, bool]] = [
            # name, centre (body frame), size, color, collide
            ("wall_back", (wall_cx, 0.0, wall_cz), (c.wall_t, 2 * c.out_half, wall_h),
             green, True),
            ("wall_left", (0.0, wall_cx, wall_cz), (2 * c.in_half, c.wall_t, wall_h),
             green, True),
            ("wall_right", (0.0, -wall_cx, wall_cz), (2 * c.in_half, c.wall_t, wall_h),
             green, True),
            ("front_below", (-wall_cx, 0.0, (c.slab_t + c.port_z0) / 2 - zb),
             (c.wall_t, 2 * c.out_half, c.port_z0 - c.slab_t), green, True),
            ("front_above", (-wall_cx, 0.0, (c.port_z1 + c.wall_top) / 2 - zb),
             (c.wall_t, 2 * c.out_half, c.wall_top - c.port_z1), green, True),
            ("front_jamb_l", (-wall_cx, c.port_half_w + jamb_w / 2, c.port_cz - zb),
             (c.wall_t, jamb_w, port_h), green, True),
            ("front_jamb_r", (-wall_cx, -(c.port_half_w + jamb_w / 2), c.port_cz - zb),
             (c.wall_t, jamb_w, port_h), green, True),
            ("roof", (0.0, 0.0, c.wall_top + c.roof_t / 2 - zb),
             (2 * c.out_half, 2 * c.out_half, c.roof_t), green, True),
            ("tray", (c.tray_cx, 0.0, c.tray_top / 2 - zb),
             (c.tray_len, c.tray_w, c.tray_top), gray, True),
            ("rail_l", (c.tray_cx, c.rail_gap_half + c.rail_t / 2,
                        c.tray_top + c.rail_h / 2 - zb),
             (c.tray_len, c.rail_t, c.rail_h), dark, True),
            ("rail_r", (c.tray_cx, -(c.rail_gap_half + c.rail_t / 2),
                        c.tray_top + c.rail_h / 2 - zb),
             (c.tray_len, c.rail_t, c.rail_h), dark, True),
            # White visual frame around the port on the outer face (identification cue).
            ("port_top", (-c.out_half - 0.0025, 0.0, c.port_z1 + 0.006 - zb),
             (0.005, 2 * c.port_half_w + 0.024, 0.012), white, False),
            ("port_side_l", (-c.out_half - 0.0025, c.port_half_w + 0.006, c.port_cz - zb),
             (0.005, 0.012, port_h + 0.024), white, False),
            ("port_side_r", (-c.out_half - 0.0025, -(c.port_half_w + 0.006), c.port_cz - zb),
             (0.005, 0.012, port_h + 0.024), white, False),
        ]

        for i in range(self.env.num_envs):
            root = f"/World/envs/env_{i}/Bin"
            if stage.GetPrimAtPath(f"{root}/wall_back").IsValid():
                continue  # idempotent (never author a duplicate xformOp)
            for name, center, size, color, collide in parts:
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

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the box pose (yaw +- bin_yaw_deg, xy jitter), sample the
        two cans' ground poses with a 50/50 SIDE SWAP (red left / beige right or the
        mirror — disjoint |y| bands keep them >= 9 cm apart by construction), stand both
        cans upright, clear latches and drives. torch.rand throughout (the first randint
        after manual_seed is degenerate on this stack)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.bin_yaw_deg)
        bxy = torch.tensor(c.bin_pos, device=dev).expand(m, 2) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.bin_jitter
        self.bin_yaw[env_ids] = yaw
        self.bin_xy[env_ids] = bxy
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = bxy
        st[:, 2] = c.slab_t / 2
        st[:, 3:7] = _quat_z(yaw)
        st[:, 0:3] += origin
        self.bin.write_root_state_to_sim(st, env_ids)

        # Cans: side swap + jitter. |y| in [band0, band1] on opposite sides.
        side = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        x01 = torch.rand(m, 2, device=dev)  # x draws for (red, beige)
        y01 = torch.rand(m, 2, device=dev)
        xs = c.can_x_range[0] + x01 * (c.can_x_range[1] - c.can_x_range[0])
        ys = c.can_y_band[0] + y01 * (c.can_y_band[1] - c.can_y_band[0])
        red_xy = torch.stack([xs[:, 0], side * ys[:, 0]], dim=1)
        beige_xy = torch.stack([xs[:, 1], -side * ys[:, 1]], dim=1)
        self.red0[env_ids] = red_xy
        self.beige0[env_ids] = beige_xy
        for body, xy in ((self.red, red_xy), (self.beige, beige_xy)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = c.can_l / 2 + 0.003
            st[:, 3] = 1.0  # upright (cylinder axis = +z)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        self.staged_latch[env_ids] = 0.0
        self.insert_latch[env_ids] = 0.0
        self.still_streak[env_ids] = 0.0
        # Seed the streak reference from the freshly WRITTEN pose (the data buffer may
        # be stale right after write_root_state_to_sim).
        self.red_prev[env_ids, 0:2] = red_xy + origin[:, 0:2]
        self.red_prev[env_ids, 2] = c.can_l / 2 + 0.003 + origin[:, 2]
        self.push_f[env_ids] = 0.0

    # ----- readings -----------------------------------------------------------------------------
    def bin_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N, 3) world points -> bin GROUND frame: xy in the box's yaw frame about its
        centre, z above the ground plane (the box only yaws)."""
        d = p_w - self.env_origins
        dx = d[:, 0] - self.bin_xy[:, 0]
        dy = d[:, 1] - self.bin_xy[:, 1]
        cy, sy = torch.cos(self.bin_yaw), torch.sin(self.bin_yaw)
        return torch.stack([cy * dx + sy * dy, -sy * dx + cy * dy, d[:, 2]], dim=1)

    def can_axis(self, body: RigidObject) -> torch.Tensor:
        """(N, 3) the can's cylinder axis in world."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        return quat_apply(body.data.root_quat_w, ez)

    def axis_local_x(self, body: RigidObject) -> torch.Tensor:
        """(N,) component of the can axis along the bin-local +x (the port axis)."""
        a = self.can_axis(body)
        return torch.cos(self.bin_yaw) * a[:, 0] + torch.sin(self.bin_yaw) * a[:, 1]

    def inside(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool, geometric: the can's CENTRE within the interior footprint and
        genuinely BELOW the port sill (a can on the roof sits at ~0.35 m, on the tray at
        ~0.17 m, straddling the sill at ~0.17 m — none counts)."""
        c = self.cfg
        p = self.bin_local(body.data.root_pos_w)
        return (p[:, 0].abs() < c.in_half - 0.005) & (p[:, 1].abs() < c.in_half - 0.005) \
            & (p[:, 2] > c.slab_t + 0.005) & (p[:, 2] < c.z_inside_max)

    def red_settled(self) -> torch.Tensor:
        """(N,) bool: red can at rest, judged by the position-delta streak maintained in
        post_step (velocity readback is phantom under the wrench plant; positions are
        authoritative). Teleports break the streak via the position jump."""
        return self.still_streak >= self.cfg.still_steps

    def success(self) -> torch.Tensor:
        """(N,) bool: the red can physically inside the box (below the sill, on the
        floor), the beige distractor NOT inside, red can settled. Judged live on the
        current physical state."""
        return self.inside(self.red) & ~self.inside(self.beige) & self.red_settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.25 * staged_latch + 0.35 * insert_latch, capped at
        0.60 unless success() — 1.0 iff success(). ~0 for the null policy (both latches
        need the can lying ON the tray / in the port band); latched credit never
        evaporates."""
        partial = (0.25 * self.staged_latch + 0.35 * self.insert_latch).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(partial), partial)

    # ----- step-coupled mechanics (every substep) ------------------------------------------------
    def post_step(self) -> None:
        """Can plant: consume the `push_f` buffer (owns both cans' wrench slots), then
        latch rubric progress: staged (rested lying + aligned on the tray, slow-gated)
        and insertion (best gated CoM advance toward the commitment line)."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        for k, body in ((0, self.red), (1, self.beige)):
            f = torch.zeros(n, 1, 3, device=dev)
            # push_f is WORLD-frame. The default wrench call applies the vector in the
            # BODY frame (effective_world = R_now . f_given on this stack), so encode
            # per substep. For the axial push on the lying can the force is along the
            # can's own roll axis, so rolling cannot rotate it off course.
            f[:, 0, :] = quat_apply_inverse(body.data.root_quat_w, self.push_f[:, k, :])
            body.set_external_force_and_torque(f, torch.zeros(n, 1, 3, device=dev))

        # Stillness streak (position-delta; velocity readback is phantom here).
        pos = self.red.data.root_pos_w
        still = (pos - self.red_prev).norm(dim=-1) < c.still_tol
        self.still_streak = torch.where(still, self.still_streak + 1,
                                        torch.zeros_like(self.still_streak))
        self.red_prev = pos.clone()

        p = self.bin_local(self.red.data.root_pos_w)
        ax = self.axis_local_x(self.red)
        az = self.can_axis(self.red)[:, 2]
        speed = self.red.data.root_lin_vel_w.norm(dim=-1)
        lying = az.abs() < 0.35
        aligned = ax.abs() > 0.80
        in_channel = p[:, 1].abs() < 0.05
        staged = lying & aligned & in_channel \
            & (p[:, 0] > c.stage_x[0]) & (p[:, 0] < c.stage_x[1]) \
            & (p[:, 2] > c.tray_top) & (p[:, 2] < c.tray_top + 0.07) \
            & (speed < c.staged_speed)
        self.staged_latch = torch.maximum(self.staged_latch, staged.float())

        prog = ((p[:, 0] - c.com_start) / (c.com_full - c.com_start)).clamp(0.0, 1.0)
        gate = lying & aligned & in_channel \
            & (p[:, 2] > c.tray_top) & (p[:, 2] < c.port_z1) & (speed < c.insert_speed)
        prog = torch.nan_to_num(prog * gate.float(), nan=0.0, posinf=0.0, neginf=0.0)
        self.insert_latch = torch.maximum(self.insert_latch, prog)

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = self._bodies()
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("bin_xy", "bin_yaw", "red0", "beige0", "staged_latch",
                               "insert_latch", "still_streak", "red_prev", "push_f")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = self._bodies()
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    def _bodies(self) -> dict[str, Any]:
        return {"bin": self.bin, "red_can": self.red, "beige_can": self.beige}

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A dark-green DROP-BOX ({2 * c.out_half * 100:.0f} cm square, "
            f"{(c.wall_top + c.roof_t) * 100:.0f} cm tall) stands on the ground, completely "
            f"SEALED on top by a roof. Its only opening is a square PORT "
            f"({2 * c.port_half_w * 1000:.0f} mm wide and {(c.port_z1 - c.port_z0) * 1000:.0f} mm "
            f"tall, outlined in WHITE) partway up one side wall, at the inner end of a gray "
            f"exterior LOADING TRAY with two dark guide rails "
            f"({2 * c.rail_gap_half * 1000:.0f} mm apart) leading straight into it. On the "
            f"ground on the far side of the tray stand two cans, both "
            f"{2 * c.can_r * 1000:.0f} mm across and {c.can_l * 1000:.0f} mm tall, upright: one "
            f"RED (the tomato-sauce can — the target) and one BEIGE (a distractor). Their "
            f"left/right positions are shuffled every episode; the box's heading also varies, "
            f"so read the tray direction from the scene.\n"
            f"Goal: get the RED can INSIDE the drop-box, resting on the box floor, and leave "
            f"the BEIGE can outside. Dropping the can from above cannot work — the roof is "
            f"closed, and a can released over the box just sits on the roof. The port admits "
            f"the can END-ON ONLY: it is wider than the can's {2 * c.can_r * 1000:.0f} mm "
            f"diameter but far shorter than its {c.can_l * 1000:.0f} mm length, so a can "
            f"presented sideways or upright jams at the opening. Lay the red can down "
            f"HORIZONTAL on the tray between the rails with its axis pointing at the port, "
            f"then push it straight along the tray through the port; once its middle passes "
            f"the wall it tips and falls to the box floor by itself. Success is judged "
            f"physically: the red can's body at rest inside the box, below the port opening, "
            f"with the beige can not inside."
        )

    def instruction(self) -> str:
        return (
            "Lay the red can down on the loading tray between the rails, axis toward the "
            "white-framed port, and push it end-on through the port so it drops inside the "
            "green box. Leave the beige can outside."
        )


# Guarded registration: the forge may import this module under two names.
if "port_dropbox" not in SCENES.list():
    SCENES.register("port_dropbox", PortDropboxScene)
if "simgen.port_dropbox" not in ENVS.list():
    register_env("simgen", lambda: EnvCfg(scene="port_dropbox", robot="null"))

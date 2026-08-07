"""CheckerSiloScene — load the gravity silo so the stack reads RED, WHITE, RED.

Derived from rlbench/setup_checkers ("place the checkers on the board in the starting
arrangement"), but the MANIPULATION MODEL is replaced wholesale. The seed's plan is:
repeatedly pick identical flat cylinders off the table and lay each one FLAT onto a
marked square of a horizontal board — 24 independent, unordered, tolerance-loose
tabletop placements onto an open surface. Here NOTHING is ever laid flat onto anything:
the destination is a narrow vertical SILO (an enclosed channel with a flared funnel
mouth) whose interior no gripper can enter. Checkers must be plucked from a supply rack
where they stand ON EDGE, carried upright, and RELEASED over the funnel so gravity
threads them down the channel onto the stack. The channel is one disc wide, so the
final bottom-to-top color sequence IS the insertion order: the goal stack (red, white,
red) forces a strict execution order the seed has no analogue of. One extra white
checker is supplied and must NOT be inserted (exactly three discs in the silo) — a
count/decoy constraint, again absent from the seed. Placement precision is replaced by
release timing/altitude over an aperture; per-piece pose accuracy is replaced by
sequencing and restraint.

Assets are fully procedural (the compound-spawner pattern — child colliders of one
body never self-collide):
  - silo: KINEMATIC tower on a foot plate. Interior channel 32 x 52 mm, 150 mm tall,
    closed on three sides; the FRONT face is two vertical strips leaving a 24 mm
    viewing slit (a 40 mm disc cannot escape, but the stack colors are visible).
    A flared YELLOW funnel (40 mm tall, mouth ~110 x 130 mm) tops the channel.
  - supply rack: KINEMATIC grooved tray — base, two rails 28 mm apart, stop nubs —
    holding four checkers standing on edge in a row (slot order randomized).
  - checkers: four DYNAMIC cylinders, r=20 mm, h=18 mm: two RED, two WHITE.

Per-episode randomization (readback-verifiable): silo yaw +/- 30 deg + xy jitter,
rack yaw +/- 25 deg + xy jitter, and the rack slot PERMUTATION of the four checkers
(plus per-slot jitter) — which disc is where must be perceived, not memorized.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve trajectory):
  0.15 * approach   — some checker ever carried within `approach_r` of the funnel
                      mouth (latched; 0 for the null policy)
  0.30 * prefix1    — the stack ever reads [RED] seated on the channel floor (latched)
  0.30 * prefix2    — the stack ever reads [RED, WHITE] contiguously (latched)
  1.0 iff success() — EXACTLY three checkers inside the silo, settled, upright in the
                      channel, contiguous from the floor, bottom-to-top colors
                      red/white/red; the spare white left outside. Non-success capped
                      at 0.75.

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


def _spawn_silo(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the silo at `prim_path`: KINEMATIC compound. Local frame: origin at the
    centre of the foot plate on the ground; the channel rises along local +z, the
    viewing slit faces local +x.

    Children: foot plate, channel floor block, two side walls (+/-y), a full back
    wall (-x), two front strips (+x, the slit between them), and four slanted funnel
    plates flaring from the channel top to the mouth."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    ix, iy = c.chan_x / 2, c.chan_y / 2          # interior half extents
    wt = c.wall_t
    zf = c.z_floor                               # channel floor TOP
    zt = zf + c.chan_h                           # channel top = funnel throat
    # foot plate + channel floor block
    _add_box(stage, f"{prim_path}/foot", center=(0.0, 0.0, 0.006),
             size=(0.16, 0.14, 0.012), color=c.body_color, collide=collide)
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, (0.012 + zf) / 2),
             size=(2 * ix + 2 * wt + 0.02, 2 * iy + 2 * wt + 0.02, zf - 0.012),
             color=c.body_color, collide=collide)
    zc = (zf + zt) / 2
    # side walls (+/-y), full x span
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/side_{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * (iy + wt / 2), zc),
                 size=(2 * ix + 2 * wt, wt, c.chan_h), color=c.body_color,
                 collide=collide)
    # back wall (-x), full y span
    _add_box(stage, f"{prim_path}/back", center=(-(ix + wt / 2), 0.0, zc),
             size=(wt, 2 * iy + 2 * wt, c.chan_h), color=c.body_color,
             collide=collide)
    # front strips (+x): viewing slit of width slit_w between them
    strip_w = iy - c.slit_w / 2
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/front_{'p' if sgn > 0 else 'n'}",
                 center=(ix + wt / 2, sgn * (c.slit_w / 2 + strip_w / 2), zc),
                 size=(wt, strip_w, c.chan_h), color=c.front_color,
                 collide=collide)
    # funnel: four slanted plates, interior opening -> flared mouth
    fh = c.funnel_h
    run_x = c.mouth_x / 2 - ix
    run_y = c.mouth_y / 2 - iy
    ax = math.atan2(run_x, fh)
    ay = math.atan2(run_y, fh)
    lx = math.hypot(run_x, fh) + 0.004
    ly = math.hypot(run_y, fh) + 0.004
    zfc = zt + fh / 2
    qx_p = (math.cos(ax / 2), 0.0, math.sin(ax / 2), 0.0)     # about +y: lean +x
    qx_n = (math.cos(ax / 2), 0.0, -math.sin(ax / 2), 0.0)
    qy_p = (math.cos(ay / 2), -math.sin(ay / 2), 0.0, 0.0)    # about -x: lean +y
    qy_n = (math.cos(ay / 2), math.sin(ay / 2), 0.0, 0.0)
    _add_box(stage, f"{prim_path}/fun_xp", center=((ix + c.mouth_x / 2) / 2, 0.0, zfc),
             size=(0.008, c.mouth_y + 0.01, lx), color=c.funnel_color,
             collide=collide, orient=qx_p)
    _add_box(stage, f"{prim_path}/fun_xn", center=(-(ix + c.mouth_x / 2) / 2, 0.0, zfc),
             size=(0.008, c.mouth_y + 0.01, lx), color=c.funnel_color,
             collide=collide, orient=qx_n)
    _add_box(stage, f"{prim_path}/fun_yp", center=(0.0, (iy + c.mouth_y / 2) / 2, zfc),
             size=(c.mouth_x + 0.01, 0.008, ly), color=c.funnel_color,
             collide=collide, orient=qy_p)
    _add_box(stage, f"{prim_path}/fun_yn", center=(0.0, -(iy + c.mouth_y / 2) / 2, zfc),
             size=(c.mouth_x + 0.01, 0.008, ly), color=c.funnel_color,
             collide=collide, orient=qy_n)
    return root


def _spawn_rack(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the supply rack: KINEMATIC grooved tray. Local frame: origin at the
    centre of the base plate on the ground; the groove runs along local +x, discs
    stand on edge with their axes along local +y."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_box(stage, f"{prim_path}/base", center=(0.0, 0.0, c.base_t / 2),
             size=(c.length, c.width, c.base_t), color=c.color, collide=collide)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/rail_{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * (c.gap / 2 + c.rail_t / 2),
                         c.base_t + c.rail_h / 2),
                 size=(c.length, c.rail_t, c.rail_h), color=c.color,
                 collide=collide)
    n_slots = 4
    pitch = c.length / n_slots
    for j in range(n_slots + 1):
        x = -c.length / 2 + j * pitch
        _add_box(stage, f"{prim_path}/nub_{j}",
                 center=(x, 0.0, c.base_t + c.nub_h / 2),
                 size=(0.006, c.gap, c.nub_h), color=c.color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "silo" not in _SPAWNER_CACHE:

        @configclass
        class SiloSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_silo)
            chan_x: float = 0.032
            chan_y: float = 0.052
            chan_h: float = 0.150
            z_floor: float = 0.032
            wall_t: float = 0.010
            slit_w: float = 0.024
            funnel_h: float = 0.040
            mouth_x: float = 0.110
            mouth_y: float = 0.130
            body_color: tuple = (0.20, 0.28, 0.45)
            front_color: tuple = (0.30, 0.40, 0.58)
            funnel_color: tuple = (0.88, 0.72, 0.16)
            contact_offset: float = 0.002

        @configclass
        class RackSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rack)
            length: float = 0.24
            width: float = 0.060
            base_t: float = 0.010
            rail_t: float = 0.008
            rail_h: float = 0.020
            gap: float = 0.028
            nub_h: float = 0.014
            color: tuple = (0.45, 0.32, 0.18)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(silo=SiloSpawnerCfg, rack=RackSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CheckerSiloSceneCfg(BaseCfg):
    """Config for `CheckerSiloScene`. The in-silo tolerances are honest by
    construction: the channel walls bound a resting disc centre to |x| <= 7 mm,
    |y| <= 6 mm, so any disc physically inside the channel passes the (20/30 mm)
    gates, while a disc outside the walls cannot."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    in_x_tol: float = tunable(0.020)   # |silo-frame x| gate for "inside" (walls bound 7 mm)
    in_y_tol: float = tunable(0.030)   # |silo-frame y| gate for "inside" (walls bound 6 mm)
    in_z_lo: float = tunable(0.027)    # silo-frame z above this = inside band (floor top 32 mm)
    in_z_hi: float = tunable(0.222)    # up to the funnel mouth: a disc in the funnel still counts
    bottom_z_max: float = tunable(0.067)  # bottom disc centre below this = seated on the floor
    gap_max: float = tunable(0.055)    # max centre-to-centre z gap between stacked discs (nom 40)
    align_max_deg: float = tunable(40.0)  # disc axis within this of the silo slit axis (local +x)
    settle_speed: float = tunable(0.05)   # max |lin vel| of every disc when judging (m/s)
    approach_r: float = tunable(0.12)  # latched approach credit: disc within this of the mouth

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    silo_yaw_deg: float = tunable(30.0)   # silo yaw about nominal (+/- deg)
    silo_jitter: float = tunable(0.05)    # silo xy jitter (+/- m)
    rack_yaw_deg: float = tunable(25.0)   # rack yaw about nominal (+/- deg)
    rack_jitter: float = tunable(0.04)    # rack xy jitter (+/- m)
    slot_shuffle: bool = tunable(True)    # randomize which checker stands in which rack slot
    slot_jitter: float = tunable(0.004)   # per-disc jitter along the groove (+/- m)

    # --- info: layout ---------------------------------------------------------------------------
    silo_pos: tuple = info((0.36, 0.14))   # silo foot centre on the ground (nominal)
    rack_pos: tuple = info((0.28, -0.26))  # rack base centre on the ground (nominal)
    # --- info: silo structure (local frame: origin at the foot centre on the ground) ------------
    chan_x: float = info(0.032)   # channel interior along local x (disc thickness direction)
    chan_y: float = info(0.052)   # channel interior along local y (disc diameter direction)
    chan_h: float = info(0.150)   # channel height above its floor
    z_floor: float = info(0.032)  # channel floor TOP height
    wall_t: float = info(0.010)
    slit_w: float = info(0.024)   # front viewing slit width (< disc diameter)
    funnel_h: float = info(0.040)
    mouth_x: float = info(0.110)  # funnel mouth opening at the top
    mouth_y: float = info(0.130)
    z_mouth: float = info(0.235)  # release point above the mouth (silo local)
    # --- info: rack -----------------------------------------------------------------------------
    rack_len: float = info(0.24)
    rack_gap: float = info(0.028)
    rack_base_t: float = info(0.010)
    # --- info: checkers -------------------------------------------------------------------------
    disc_r: float = info(0.020)
    disc_h: float = info(0.018)
    disc_mass: float = info(0.030)
    red_color: tuple = info((0.85, 0.08, 0.08))
    white_color: tuple = info((0.93, 0.93, 0.90))
    contact_offset: float = info(0.002)
    # target pattern bottom-up: True=red. Fixed task definition (red, white, red).
    pattern: tuple = info((True, False, True))
    # rubric weights (0.15 + 0.30 + 0.30 = 0.75 = the non-success cap)
    w_appr: float = info(0.15)
    w_p1: float = info(0.30)
    w_p2: float = info(0.30)


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


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("checker_silo")
class CheckerSiloScene(BaseScene):
    cfg: CheckerSiloSceneCfg

    DISC_NAMES = ("red_0", "red_1", "white_0", "white_1")
    IS_RED = (True, True, False, False)

    def __init__(self, cfg: CheckerSiloSceneCfg | None = None) -> None:
        super().__init__(cfg or CheckerSiloSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        silo_spawn = cls["silo"](
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            chan_x=c.chan_x, chan_y=c.chan_y, chan_h=c.chan_h, z_floor=c.z_floor,
            wall_t=c.wall_t, slit_w=c.slit_w, funnel_h=c.funnel_h,
            mouth_x=c.mouth_x, mouth_y=c.mouth_y, contact_offset=c.contact_offset)
        rack_spawn = cls["rack"](
            mass_props=sim_utils.MassPropertiesCfg(mass=3.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            length=c.rack_len, gap=c.rack_gap, base_t=c.rack_base_t,
            contact_offset=c.contact_offset)

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
            "silo": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Silo",
                spawn=silo_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.silo_pos[0], c.silo_pos[1], 0.0)),
            ),
            "rack": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rack",
                spawn=rack_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rack_pos[0], c.rack_pos[1], 0.0)),
            ),
        }
        for i, name in enumerate(self.DISC_NAMES):
            color = c.red_color if self.IS_RED[i] else c.white_color
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Disc_" + name,
                spawn=sim_utils.CylinderCfg(
                    radius=c.disc_r, height=c.disc_h, axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=1.0, angular_damping=0.5,
                        sleep_threshold=0.0, stabilization_threshold=0.0,
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=1),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.disc_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.005, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5,
                        restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.6 + 0.1 * i, -0.6, 0.05)),
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
        self.silo: RigidObject = env.iscene["silo"]
        self.rack: RigidObject = env.iscene["rack"]
        self.discs: dict[str, RigidObject] = {
            n: env.iscene[n] for n in self.DISC_NAMES}
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self.is_red = torch.tensor(self.IS_RED, dtype=torch.bool, device=dev)
        self.slot_of = torch.zeros(n, 4, dtype=torch.long, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._appr = torch.zeros(n, dtype=torch.bool, device=dev)
        self._p1 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._p2 = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place silo and rack (yaw + xy jitter each), shuffle the
        four checkers over the rack slots (standing on edge, axis along the groove
        normal, per-slot jitter), clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- silo: kinematic, yaw + xy jitter ---
        s_psi = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.silo_yaw_deg)
        q_silo = _qz(s_psi)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.silo_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.silo_jitter
        st[:, 1] = c.silo_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.silo_jitter
        st[:, 3:7] = q_silo
        st[:, 0:3] += origin
        self.silo.write_root_state_to_sim(st, env_ids)

        # --- rack: kinematic, yaw + xy jitter ---
        r_psi = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rack_yaw_deg)
        q_rack = _qz(r_psi)
        rp = torch.zeros(m, 3, device=dev)
        rp[:, 0] = c.rack_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.rack_jitter
        rp[:, 1] = c.rack_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.rack_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = rp + origin
        st[:, 3:7] = q_rack
        self.rack.write_root_state_to_sim(st, env_ids)

        # --- checkers: random slot permutation, standing on edge in the groove ---
        from isaaclab.utils.math import quat_apply

        if c.slot_shuffle:
            perm = torch.rand(m, 4, device=dev).argsort(dim=1)
        else:
            perm = torch.arange(4, device=dev).expand(m, 4).clone()
        self.slot_of[env_ids] = perm
        pitch = c.rack_len / 4
        slot_x = torch.tensor([-1.5, -0.5, 0.5, 1.5], device=dev) * pitch
        q_disc = _qmul(q_rack, _qx(torch.full((m,), math.pi / 2, device=dev)))
        for i in range(4):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = slot_x[perm[:, i]] \
                + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
            loc[:, 2] = c.rack_base_t + c.disc_r + 0.002
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = rp + quat_apply(q_rack, loc) + origin
            st[:, 3:7] = q_disc
            self.discs[self.DISC_NAMES[i]].write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._appr[env_ids] = False
        self._p1[env_ids] = False
        self._p2[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "silo": self.silo.data.root_state_w[env_ids].clone(),
            "rack": self.rack.data.root_state_w[env_ids].clone(),
            "discs": {n: b.data.root_state_w[env_ids].clone()
                      for n, b in self.discs.items()},
            "slot_of": self.slot_of[env_ids].clone(),
            "appr": self._appr[env_ids].clone(),
            "p1": self._p1[env_ids].clone(),
            "p2": self._p2[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.silo.write_root_state_to_sim(state["silo"], env_ids)
        self.rack.write_root_state_to_sim(state["rack"], env_ids)
        for n, b in self.discs.items():
            b.write_root_state_to_sim(state["discs"][n], env_ids)
        self.slot_of[env_ids] = state["slot_of"]
        self._appr[env_ids] = state["appr"]
        self._p1[env_ids] = state["p1"]
        self._p2[env_ids] = state["p2"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A dark-blue SILO TOWER stands on the ground: a narrow vertical channel "
            f"(interior {c.chan_x * 1000:.0f} x {c.chan_y * 1000:.0f} mm, "
            f"{c.chan_h * 1000:.0f} mm tall) topped by a flared YELLOW FUNNEL whose "
            f"open mouth (~{c.mouth_x * 1000:.0f} x {c.mouth_y * 1000:.0f} mm) is the "
            f"ONLY way in. The channel is exactly one checker wide, so checkers "
            f"dropped through the funnel fall down and stack on edge, in the order "
            f"they are dropped; a narrow front slit lets you see the stack colors. "
            f"No hand or finger fits inside the channel or the slit, and a checker "
            f"that has dropped in cannot be taken back out. Nearby on the ground, a "
            f"wooden SUPPLY RACK (a grooved tray) holds FOUR checkers standing "
            f"upright on edge in a row: two RED and two WHITE (each a disc "
            f"{2 * c.disc_r * 1000:.0f} mm across, {c.disc_h * 1000:.0f} mm thick). "
            f"Which slot holds which color is shuffled every episode — look at the "
            f"colors. The silo's and rack's positions and headings also vary.\n"
            f"Goal: load the silo so that its final stack reads, from BOTTOM to TOP: "
            f"RED, then WHITE, then RED — exactly three checkers, with the remaining "
            f"white checker left anywhere OUTSIDE the silo. Because the silo fills "
            f"bottom-up in drop order and cannot be unloaded, you MUST insert a red "
            f"checker first, then a white one, then the second red one; dropping a "
            f"wrong color, or dropping the fourth checker in, ruins the stack "
            f"permanently. Take checkers from the rack (they already stand upright "
            f"there), keep each one roughly upright, and release it over the yellow "
            f"funnel mouth so it falls in. Success: exactly the red-white-red stack "
            f"resting on the channel floor, everything at rest, the spare white "
            f"checker outside the silo."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Drop checkers from the rack into the silo through its yellow funnel so "
            "the stack reads red, white, red from bottom to top: insert a red "
            "checker first, then a white one, then the other red one. Leave the "
            "remaining white checker outside the silo — exactly three checkers go in."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _silo_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the (kinematic, live-read) silo frame. Accepts (N,3) or
        (N,P,3); returns the same shape."""
        from isaaclab.utils.math import quat_apply_inverse

        sp = self.silo.data.root_pos_w
        sq = self.silo.data.root_quat_w
        if pos_w.dim() == 3:
            n, p = pos_w.shape[0], pos_w.shape[1]
            rel = (pos_w - sp[:, None, :]).reshape(n * p, 3)
            q = sq[:, None, :].expand(n, p, 4).reshape(n * p, 4)
            return quat_apply_inverse(q, rel).reshape(n, p, 3)
        return quat_apply_inverse(sq, pos_w - sp)

    def mouth_point_w(self) -> torch.Tensor:
        """(N, 3) world release point above the funnel mouth: silo local (0,0,z_mouth)."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        off = torch.tensor([0.0, 0.0, self.cfg.z_mouth],
                           device=self.env.device).expand(n, 3)
        return self.silo.data.root_pos_w + quat_apply(self.silo.data.root_quat_w, off)

    def _disc_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(pos_w (N,4,3), quat (N,4,4), |lin_vel| (N,4)) for all discs, name order."""
        pos = torch.stack([b.data.root_pos_w for b in self.discs.values()], dim=1)
        quat = torch.stack([b.data.root_quat_w for b in self.discs.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.discs.values()], dim=1)
        return pos, quat, vel

    def in_silo(self) -> torch.Tensor:
        """(N, 4) bool, geometric: disc centre inside the silo interior column
        (channel + funnel), silo frame. Honest by construction: the walls bound a
        resting disc centre far inside these gates."""
        c = self.cfg
        pos, _q, _v = self._disc_tensors()
        loc = self._silo_local(pos)
        return (loc[:, :, 0].abs() < c.in_x_tol) \
            & (loc[:, :, 1].abs() < c.in_y_tol) \
            & (loc[:, :, 2] > c.in_z_lo) & (loc[:, :, 2] < c.in_z_hi)

    def aligned(self) -> torch.Tensor:
        """(N, 4) bool: disc axis within `align_max_deg` of the silo slit axis
        (silo local +x) — a properly stacked disc stands on edge facing the slit."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        _p, quat, _v = self._disc_tensors()
        n = quat.shape[0]
        ez = torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(n * 4, 3)
        axis_w = quat_apply(quat.reshape(n * 4, 4), ez)
        sq = self.silo.data.root_quat_w[:, None, :].expand(n, 4, 4).reshape(n * 4, 4)
        axis_s = quat_apply_inverse(sq, axis_w).reshape(n, 4, 3)
        return axis_s[:, :, 0].abs() >= math.cos(math.radians(self.cfg.align_max_deg))

    def stack_readout(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sort the inside discs by silo-frame height. Returns (count (N,),
        zs (N,4) sorted with +inf for outside, red_sorted (N,4) bool)."""
        pos, _q, _v = self._disc_tensors()
        loc = self._silo_local(pos)
        inside = self.in_silo()
        z = torch.where(inside, loc[:, :, 2],
                        torch.full_like(loc[:, :, 2], torch.inf))
        zs, idx = z.sort(dim=1)
        red_sorted = self.is_red.expand_as(inside).gather(1, idx)
        return inside.sum(dim=1), zs, red_sorted

    def _prefix_now(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(p1, p2, p3) live: the stack currently reads [R], [R,W], [R,W,R] —
        seated on the floor, contiguous, exact count for p3."""
        c = self.cfg
        k, zs, red = self.stack_readout()
        seated = zs[:, 0] < c.bottom_z_max
        p1 = (k >= 1) & red[:, 0] & seated
        p2 = p1 & (k >= 2) & ~red[:, 1] & ((zs[:, 1] - zs[:, 0]) < c.gap_max)
        p3 = p2 & (k == 3) & red[:, 2] & ((zs[:, 2] - zs[:, 1]) < c.gap_max)
        return p1, p2, p3

    def _update_latches(self) -> None:
        c = self.cfg
        pos, _q, _v = self._disc_tensors()
        d = (pos - self.mouth_point_w()[:, None, :]).norm(dim=-1)
        self._appr |= (d < c.approach_r).any(dim=1)
        p1, p2, _p3 = self._prefix_now()
        self._p1 |= p1
        self._p2 |= p2

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: EXACTLY three checkers inside the silo, stacked contiguously
        from the channel floor, bottom-to-top colors red/white/red, every inside
        disc on edge (aligned with the slit axis), all four discs at rest and
        finite. All clauses are live physical outcomes."""
        c = self.cfg
        self._update_latches()
        _p1, _p2, p3 = self._prefix_now()
        inside = self.in_silo()
        ok_align = (self.aligned() | ~inside).all(dim=1)
        pos, _q, vel = self._disc_tensors()
        still = (vel < c.settle_speed).all(dim=1)
        finite = torch.isfinite(pos).all(dim=-1).all(dim=-1)
        return p3 & ok_align & still & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*approach + 0.30*prefix1 + 0.30*prefix2 (all
        latched; ~0 for doing nothing), capped at 0.75 — and exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_appr * self._appr.float() + c.w_p1 * self._p1.float()
                + c.w_p2 * self._p2.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="checker_silo", robot="null"))

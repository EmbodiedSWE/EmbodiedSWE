"""KerfChopScene — sever the marked middle of a rod with a knife pressed through a kerf slot.

Derived from rlbench/put_knife_in_knife_block ("pick the knife off the chopping board
and slide it into a slot of the knife block": the knife is the OBJECT of the task and
the plan is grasp-handle -> carry -> align -> vertical insertion into a passive
receptacle; nothing resists, nothing changes but the knife's pose). Here the knife's
role is inverted wholesale: the knife is a FORCE TOOL and the goal state is a change
in ANOTHER object — the task cannot be completed by putting the knife anywhere.

A heavy CHOPPING STATION stands on the ground: two grey anvil towers bridged by a
workpiece — a square rod made of three butted segments (beige / RED / beige), the two
outer segments resting on the anvil tops, the RED middle spanning the open WELL
between them. The seams are breakable fixed joints: physical welds that shear only
above a real force/torque threshold. The rod is caged by side fences, end stops and a
COVER plate 12 mm above it, so no gripper can touch it — the only access is a narrow
KERF SLOT in the cover (13 mm wide), directly over the red span. On the ground lie two
tools: a thin-bladed KNIFE (4 mm blade — fits the slot) and a wide CLEAVER (20 mm
blade — geometrically excluded). The task: drive the knife blade through the slot and
press down hard enough to snap BOTH seams, so the red section drops into the well and
settles there, while the beige ends stay seated on their anvils.

Success is a physical outcome, live-read: both seams severed (relative seam-point
separation readback), the RED segment settled INSIDE the well, both beige segments
still seated on the anvils, the station upright, everything settled and finite.

Assets are fully procedural (compound-spawner pattern; the seam joints are authored at
spawn on the middle segment, referencing its sibling segments, with explicit
physics:breakForce / physics:breakTorque).

Per-episode randomization (readback-verifiable): station yaw +/- 20 deg + xy jitter,
rod x-offset along the cradle, tool slot swap (which side the knife lies on) +
per-tool xy jitter + free yaw.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve trajectory):
  0.15 * engaged — the knife blade tip ever inside the kerf slot below the cover top
                   (latched; only the thin blade fits)
  0.20 * pressed — engaged AND the blade tip ever driven below the rod's rest top
                   (latched; requires deflecting or severing the rod, not just resting
                   on it)
  0.15 * sev_a + 0.15 * sev_b — each seam severed (latched; physically irreversible)
  1.0 iff success(). Non-success capped at 0.65.

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


def _qx(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 1] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def encode_force(mode: int, q_ref: torch.Tensor, q_now: torch.Tensor,
                 f_world: torch.Tensor) -> torch.Tensor:
    """Pre-encode a desired WORLD-frame force for `set_external_force_and_torque`.

    Some pods rotate an applied wrench by the body's rotation since its reference
    orientation (applied = R_now * R_ref^T * arg). mode 0 passes the world force
    through unchanged; mode 1 pre-encodes with R_ref * R_now^T so the applied force
    comes out as the desired world force. Callers PROBE which mode moves the body the
    right way and lock it in (`q_ref` = readback at the reference instant)."""
    if mode == 0:
        return f_world
    return _qapply(_qmul(q_ref, _qinv(q_now)), f_world)


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


def _rigid_root(prim_path: str, translation, orientation, mass: float, *,
                ang_damping: float = 0.2, lin_damping: float = 0.1,
                com=None):
    """Author a dynamic rigid root with the standard physics armor."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(float(mass))
    if com is not None:
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(lin_damping))
    pxrb.CreateAngularDampingAttr(float(ang_damping))
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(1)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    return stage, root


def _bind_mat(prim_path: str, mat_path: str, static: float, dynamic: float) -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    sim_utils.spawn_rigid_body_material(
        mat_path,
        sim_utils.RigidBodyMaterialCfg(static_friction=static, dynamic_friction=dynamic,
                                       restitution=0.0))
    bind_physics_material(prim_path, mat_path)


def _spawn_station(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the chopping station: heavy DYNAMIC compound (25 kg). Local frame:
    origin at the footprint centre on the ground; the cradle runs along local x.

    Children: two anvil towers (rod supports flanking the WELL), the well floor and
    its two y-walls, two rod fences, two end stops, and the COVER plate (12 mm above
    the rod top) pierced by the KERF SLOT: 2 side strips + 2 end caps leaving a
    slot_x_half*2 x slot_y_half*2 aperture centred over the well."""
    c = cfg
    stage, root = _rigid_root(prim_path, translation, orientation, 25.0,
                              ang_damping=0.5, lin_damping=0.5)
    collide = _make_collide(c.contact_offset)
    grey, lite, dark = c.body_color, c.cover_color, c.well_color

    az = c.anvil_top
    # anvil towers: x in +/-[well_x_half, cradle_x_half+0.004], ground..anvil_top
    ax0, ax1 = c.well_x_half, c.cradle_x_half + 0.004
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/anvil_{tag}",
                 center=(sgn * (ax0 + ax1) / 2, 0.0, az / 2),
                 size=(ax1 - ax0, 0.060, az), color=grey, collide=collide)
    # well floor + y walls (the catch tray between the anvils)
    _add_box(stage, f"{prim_path}/well_floor", center=(0.0, 0.0, 0.005),
             size=(2 * c.well_x_half, 2 * c.well_y_half + 0.010, 0.010),
             color=dark, collide=collide)
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/well_wall_{tag}",
                 center=(0.0, sgn * (c.well_y_half + 0.0025), c.well_wall_z / 2),
                 size=(2 * c.well_x_half, 0.005, c.well_wall_z),
                 color=dark, collide=collide)
    # rod fences: from the anvil top to the cover underside, full cradle length
    fz0, fz1 = az, c.cover_under
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/fence_{tag}",
                 center=(0.0, sgn * (c.fence_y_inner + 0.0025), (fz0 + fz1) / 2),
                 size=(2 * (c.cradle_x_half + 0.004), 0.005, fz1 - fz0),
                 color=grey, collide=collide)
    # end stops
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/stop_{tag}",
                 center=(sgn * (c.cradle_x_half + 0.0025), 0.0, (fz0 + fz1) / 2),
                 size=(0.005, 2 * c.fence_y_inner, fz1 - fz0),
                 color=grey, collide=collide)
    # cover plate with the kerf slot
    ct = (c.cover_under + c.cover_top) / 2
    cth = c.cover_top - c.cover_under
    cover_y1 = c.fence_y_inner + 0.005  # cover spans to the fence outer face
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/cover_strip_{tag}",
                 center=(0.0, sgn * (c.slot_y_half + cover_y1) / 2, ct),
                 size=(2 * (c.cradle_x_half + 0.005), cover_y1 - c.slot_y_half, cth),
                 color=lite, collide=collide)
    capx0, capx1 = c.slot_x_half, c.cradle_x_half + 0.005
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/cover_cap_{tag}",
                 center=(sgn * (capx0 + capx1) / 2, 0.0, ct),
                 size=(capx1 - capx0, 2 * c.slot_y_half, cth),
                 color=lite, collide=collide)
    # friction: anvils grip the rod; the well floor is grippy so the fallen piece stays
    _bind_mat(f"{prim_path}/anvil_p", f"{prim_path}/anvilMat", 0.60, 0.55)
    _bind_mat(f"{prim_path}/anvil_n", f"{prim_path}/anvilMat", 0.60, 0.55)
    _bind_mat(f"{prim_path}/well_floor", f"{prim_path}/wellMat", 0.80, 0.75)
    return root


def _spawn_segment(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rod segment: a single dynamic box. Local frame at the segment centre."""
    c = cfg
    stage, root = _rigid_root(prim_path, translation, orientation, c.seg_mass,
                              ang_damping=0.2, lin_damping=0.1)
    collide = _make_collide(c.contact_offset)
    _add_box(stage, f"{prim_path}/body",
             center=(0.0, 0.0, 0.0),
             size=(c.seg_len, 2 * c.rod_half, 2 * c.rod_half),
             color=c.color, collide=collide)
    _bind_mat(f"{prim_path}/body", f"{prim_path}/segMat", 0.60, 0.55)
    if c.with_joints:
        # breakable seam welds to the sibling outer segments (authored at spawn —
        # post-play joints are dead). Body0 = outer, Body1 = this (middle) segment.
        from pxr import Gf, UsdPhysics

        base = prim_path.rsplit("/", 1)[0]
        for sib, sgn, tag in ((c.sib_a, -1.0, "a"), (c.sib_b, 1.0, "b")):
            j = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/seam_{tag}")
            j.CreateBody0Rel().SetTargets([f"{base}/{sib}"])
            j.CreateBody1Rel().SetTargets([prim_path])
            # seam plane: at +/- seg_len/2 of this (middle) segment
            j.CreateLocalPos0Attr(Gf.Vec3f(-sgn * c.seg_len / 2, 0.0, 0.0))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(sgn * c.seg_len / 2, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateBreakForceAttr(float(c.break_force))
            j.CreateBreakTorqueAttr(float(c.break_torque))
    return root


def _spawn_tool(prim_path: str, cfg: Any, translation=None, orientation=None):
    """A knife-like tool: handle box + blade box, collinear (handle along the blade
    spine). Local frame: origin at the handle/blade junction, blade extends +x, blade
    edge at local z = -blade_h (use pose = blade DOWN). CoM authored blade-heavy so a
    press near the origin lands close to the contact line."""
    c = cfg
    stage, root = _rigid_root(prim_path, translation, orientation, c.mass,
                              ang_damping=0.8, lin_damping=0.1, com=c.com)
    collide = _make_collide(c.contact_offset)
    _add_box(stage, f"{prim_path}/handle",
             center=(-0.055, 0.0, 0.010),
             size=(0.110, 0.022, 0.020), color=c.handle_color, collide=collide)
    _add_box(stage, f"{prim_path}/blade",
             center=(c.blade_len / 2, 0.0, -c.blade_h / 2),
             size=(c.blade_len, c.blade_t, c.blade_h),
             color=c.blade_color, collide=collide)
    _bind_mat(f"{prim_path}/blade", f"{prim_path}/bladeMat", 0.30, 0.25)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "station" not in _SPAWNER_CACHE:

        @configclass
        class StationSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_station)
            anvil_top: float = 0.060
            well_x_half: float = 0.052
            well_y_half: float = 0.032
            well_wall_z: float = 0.050
            fence_y_inner: float = 0.0140
            cradle_x_half: float = 0.128
            cover_under: float = 0.096
            cover_top: float = 0.104
            slot_x_half: float = 0.050
            slot_y_half: float = 0.0065
            body_color: tuple = (0.35, 0.38, 0.45)
            cover_color: tuple = (0.62, 0.64, 0.68)
            well_color: tuple = (0.20, 0.21, 0.24)
            contact_offset: float = 0.002

        @configclass
        class SegmentSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_segment)
            seg_len: float = 0.080
            rod_half: float = 0.012
            seg_mass: float = 0.12
            color: tuple = (0.82, 0.75, 0.60)
            with_joints: bool = False
            sib_a: str = "RodA"
            sib_b: str = "RodB"
            break_force: float = 60.0
            break_torque: float = 0.16
            contact_offset: float = 0.002

        @configclass
        class ToolSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tool)
            blade_len: float = 0.050
            blade_t: float = 0.004
            blade_h: float = 0.040
            mass: float = 0.15
            com: tuple = (0.020, 0.0, -0.010)
            handle_color: tuple = (0.45, 0.30, 0.15)
            blade_color: tuple = (0.85, 0.87, 0.90)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["station"] = StationSpawnerCfg
        _SPAWNER_CACHE["segment"] = SegmentSpawnerCfg
        _SPAWNER_CACHE["tool"] = ToolSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class KerfChopSceneCfg(BaseCfg):
    """Config for `KerfChopScene`. The severing threshold is real physics: the seam
    joints carry `break_torque` = 0.16 N m / `break_force` = 60 N — ~20x the static
    load of the resting rod (self-weight end moments ~0.008 N m), and reachable with
    a ~15-30 N blade press on the spanning middle (measured on the forge: a 0.08 N m
    seam broke at a 10 N press; 0.16 N m holds the resting blade with wide margin)."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    sever_sep: float = tunable(0.020)     # seam counts as severed when the two seam
    # anchor points (segment-frame, on the weld plane) separate beyond this (m)
    settle_speed: float = tunable(0.05)   # max |lin vel| (rod segments) when judging (m/s)
    seat_z_tol: float = tunable(0.010)    # outer segment still seated: |z - rest| below this
    seat_y_tol: float = tunable(0.014)    # outer segment still seated: |y| below this
    well_z_max: float = tunable(0.048)    # middle centre below this = down in the well
    upright_min: float = tunable(0.90)    # station up-axis dot world-up above this

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    station_yaw_deg: float = tunable(20.0)  # station yaw about nominal (+/- deg)
    station_jitter: float = tunable(0.04)   # station xy jitter (+/- m)
    rod_x_jitter: float = tunable(0.006)    # rod x-offset along the cradle (+/- m)
    tool_swap: bool = tunable(True)         # shuffle which side slot holds the knife
    tool_jitter: float = tunable(0.030)     # per-tool xy jitter (+/- m)
    tool_yaw_deg: float = tunable(180.0)    # per-tool free yaw (+/- deg)

    # --- info: layout (world nominal) ------------------------------------------------------------
    station_pos: tuple = info((0.45, 0.0))   # station origin on the ground (nominal)
    station_yaw_nom_deg: float = info(0.0)   # nominal heading (cradle along world x)
    tool_slot: tuple = info((0.16, 0.24))    # tool start slots: (x, +/-y), world nominal
    # --- info: station structure (local frame: origin at footprint centre, ground) ---------------
    anvil_top: float = info(0.060)    # anvil top surface (rod rest support)
    well_x_half: float = info(0.052)  # well opening half-extent (anvil inner faces)
    well_y_half: float = info(0.032)  # well interior half-width
    well_wall_z: float = info(0.050)  # well y-wall top height
    fence_y_inner: float = info(0.0140)  # rod fence inner face |y| (2 mm rod clearance — blade can't pass)
    cradle_x_half: float = info(0.128)   # end-stop inner face |x|
    cover_under: float = info(0.096)  # cover underside height
    cover_top: float = info(0.104)    # cover top height
    slot_x_half: float = info(0.050)  # kerf slot half-length (local x)
    slot_y_half: float = info(0.0065)  # kerf slot half-width (local y) — 13 mm aperture
    # --- info: rod --------------------------------------------------------------------------------
    seg_len: float = info(0.080)      # each segment length (rod total 240 mm)
    rod_half: float = info(0.012)     # square cross-section half-extent (24 mm)
    seg_mass: float = info(0.12)
    break_force: float = info(60.0)   # seam weld: physics:breakForce (N)
    break_torque: float = info(0.16)  # seam weld: physics:breakTorque (N m)
    mid_color: tuple = info((0.85, 0.10, 0.10))
    outer_color: tuple = info((0.82, 0.75, 0.60))
    # --- info: tools ------------------------------------------------------------------------------
    blade_len: float = info(0.050)
    knife_blade_t: float = info(0.004)   # fits the 13 mm slot
    cleaver_blade_t: float = info(0.020)  # does NOT fit the 13 mm slot
    blade_h: float = info(0.040)
    tool_mass: float = info(0.15)
    contact_offset: float = info(0.002)
    # rubric weights (0.15 + 0.25 + 0.25 = 0.65 = the non-success cap)
    w_engaged: float = info(0.15)
    w_sever: float = info(0.25)

    @property
    def rod_rest_z(self) -> float:
        """Rod centre rest height (anvil top + half cross-section)."""
        return self.anvil_top + self.rod_half

    @property
    def rod_top_z(self) -> float:
        """Rod top face rest height."""
        return self.anvil_top + 2 * self.rod_half


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("kerf_chop")
class KerfChopScene(BaseScene):
    cfg: KerfChopSceneCfg

    def __init__(self, cfg: KerfChopSceneCfg | None = None) -> None:
        super().__init__(cfg or KerfChopSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        station_spawn = cls["station"](contact_offset=c.contact_offset)
        seg_kw = dict(seg_len=c.seg_len, rod_half=c.rod_half, seg_mass=c.seg_mass,
                      contact_offset=c.contact_offset)
        knife_spawn = cls["tool"](blade_len=c.blade_len, blade_t=c.knife_blade_t,
                                  blade_h=c.blade_h, mass=c.tool_mass,
                                  handle_color=(0.45, 0.30, 0.15),
                                  blade_color=(0.85, 0.87, 0.90),
                                  contact_offset=c.contact_offset)
        cleaver_spawn = cls["tool"](blade_len=c.blade_len, blade_t=c.cleaver_blade_t,
                                    blade_h=c.blade_h, mass=c.tool_mass,
                                    handle_color=(0.12, 0.12, 0.12),
                                    blade_color=(0.28, 0.28, 0.32),
                                    contact_offset=c.contact_offset)

        px, py = c.station_pos
        z_rod = c.rod_rest_z

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
            "station": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Station",
                spawn=station_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0)),
            ),
            # spawn order matters: the seam joints on the middle segment reference
            # the sibling outer segments, so RodA / RodB are declared first.
            "out_a": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/RodA",
                spawn=cls["segment"](color=c.outer_color, with_joints=False, **seg_kw),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px - 0.080, py, z_rod)),
            ),
            "out_b": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/RodB",
                spawn=cls["segment"](color=c.outer_color, with_joints=False, **seg_kw),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px + 0.080, py, z_rod)),
            ),
            "mid": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/RodMid",
                spawn=cls["segment"](color=c.mid_color, with_joints=True,
                                     sib_a="RodA", sib_b="RodB",
                                     break_force=c.break_force,
                                     break_torque=c.break_torque, **seg_kw),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, z_rod)),
            ),
            "knife": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Knife",
                spawn=knife_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.tool_slot[0], c.tool_slot[1], 0.030),
                    rot=(0.70711, 0.70711, 0.0, 0.0)),  # lying flat, blade sideways
            ),
            "cleaver": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cleaver",
                spawn=cleaver_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.tool_slot[0], -c.tool_slot[1], 0.030),
                    rot=(0.70711, 0.70711, 0.0, 0.0)),
            ),
        }
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
        self.station: RigidObject = env.iscene["station"]
        self.out_a: RigidObject = env.iscene["out_a"]
        self.out_b: RigidObject = env.iscene["out_b"]
        self.mid: RigidObject = env.iscene["mid"]
        self.knife: RigidObject = env.iscene["knife"]
        self.cleaver: RigidObject = env.iscene["cleaver"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # knife_slot[e] = +1 / -1: sign of the start slot (world y) the KNIFE occupies
        self.knife_slot = torch.ones(n, dtype=torch.float, device=dev)
        # rod x-offset along the cradle (station frame), per episode
        self.rod_off = torch.zeros(n, dtype=torch.float, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._engaged = torch.zeros(n, dtype=torch.bool, device=dev)
        self._sev_a = torch.zeros(n, dtype=torch.bool, device=dev)
        self._sev_b = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the station (yaw + xy jitter), seat the three rod
        segments in the cradle at a random x-offset in one consistent write (zero
        joint error at the seams), scatter the tools on their ground slots (slot swap
        + jitter + free yaw), clear the latches.

        NOTE: a seam weld, once broken, is gone for the lifetime of the stage —
        PhysX cannot re-create it. Resets restore poses, not broken welds; test
        batteries must order their intact-seam probes before any deliberate break."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- station: nominal heading + yaw + xy jitter ---
        yaw = math.radians(c.station_yaw_nom_deg) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.station_yaw_deg)
        q_st = _qz(yaw)
        pp = torch.zeros(m, 3, device=dev)
        pp[:, 0] = c.station_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.station_jitter
        pp[:, 1] = c.station_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.station_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pp + origin
        st[:, 3:7] = q_st
        self.station.write_root_state_to_sim(st, env_ids)

        # --- rod: three segments seated in one consistent arrangement ---
        off = (torch.rand(m, device=dev) * 2 - 1) * c.rod_x_jitter
        self.rod_off[env_ids] = off
        for body, dx in ((self.out_a, -c.seg_len), (self.mid, 0.0), (self.out_b, c.seg_len)):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = off + dx
            loc[:, 2] = c.rod_rest_z + 0.001
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pp + _qapply(q_st, loc) + origin
            st[:, 3:7] = q_st
            body.write_root_state_to_sim(st, env_ids)

        # --- tools: ground start slots, slot swap + jitter + free yaw, lying flat ---
        if c.tool_swap:
            swap = torch.where(torch.rand(m, device=dev) < 0.5,
                               torch.ones(m, device=dev), -torch.ones(m, device=dev))
        else:
            swap = torch.ones(m, device=dev)
        self.knife_slot[env_ids] = swap
        half_pi = torch.full((m,), math.pi / 2, device=dev)
        for body, sgn in ((self.knife, swap), (self.cleaver, -swap)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = c.tool_slot[0] + (torch.rand(m, device=dev) * 2 - 1) * c.tool_jitter
            st[:, 1] = sgn * c.tool_slot[1] \
                + (torch.rand(m, device=dev) * 2 - 1) * c.tool_jitter
            st[:, 2] = 0.030
            qyaw = _qz((torch.rand(m, device=dev) * 2 - 1)
                       * math.radians(c.tool_yaw_deg))
            st[:, 3:7] = _qmul(qyaw, _qx(half_pi))  # lying flat, blade sideways
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._engaged[env_ids] = False
        self._sev_a[env_ids] = False
        self._sev_b[env_ids] = False

    # ----- state (full, restorable poses; broken welds are NOT restorable) -----------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "station": self.station.data.root_state_w[env_ids].clone(),
            "out_a": self.out_a.data.root_state_w[env_ids].clone(),
            "out_b": self.out_b.data.root_state_w[env_ids].clone(),
            "mid": self.mid.data.root_state_w[env_ids].clone(),
            "knife": self.knife.data.root_state_w[env_ids].clone(),
            "cleaver": self.cleaver.data.root_state_w[env_ids].clone(),
            "knife_slot": self.knife_slot[env_ids].clone(),
            "rod_off": self.rod_off[env_ids].clone(),
            "engaged": self._engaged[env_ids].clone(),
            "sev_a": self._sev_a[env_ids].clone(),
            "sev_b": self._sev_b[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for name, body in (("station", self.station), ("out_a", self.out_a),
                           ("out_b", self.out_b), ("mid", self.mid),
                           ("knife", self.knife), ("cleaver", self.cleaver)):
            body.write_root_state_to_sim(state[name], env_ids)
        self.knife_slot[env_ids] = state["knife_slot"]
        self.rod_off[env_ids] = state["rod_off"]
        self._engaged[env_ids] = state["engaged"]
        self._sev_a[env_ids] = state["sev_a"]
        self._sev_b[env_ids] = state["sev_b"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A heavy CHOPPING STATION stands on the ground: two grey-blue anvil "
            f"towers ({c.anvil_top * 1000:.0f} mm tall) bridged by a square rod "
            f"({2 * c.rod_half * 1000:.0f} x {2 * c.rod_half * 1000:.0f} mm cross-"
            f"section, 240 mm long). The rod is made of three butted sections welded "
            f"at two seams: BEIGE ends resting on the anvil tops and a RED middle "
            f"section ({c.seg_len * 1000:.0f} mm) spanning the open WELL between the "
            f"anvils — a dark catch pit whose floor lies near the ground. The rod is "
            f"caged: side fences, end stops, and a light-grey COVER plate fixed "
            f"{(c.cover_under - c.rod_top_z) * 1000:.0f} mm above it close it in from "
            f"above, so no finger or gripper can reach the rod. The only access is "
            f"the KERF SLOT cut through the cover directly over the red span: "
            f"{2 * c.slot_x_half * 1000:.0f} mm long and only "
            f"{2 * c.slot_y_half * 1000:.0f} mm wide. On the ground in front lie two "
            f"tools of the same handle size, one on each side (which side is "
            f"shuffled per episode): a KNIFE with a bright steel blade only "
            f"{c.knife_blade_t * 1000:.0f} mm thick — it fits through the slot — and "
            f"a dark CLEAVER whose blade is {c.cleaver_blade_t * 1000:.0f} mm thick "
            f"and cannot enter the slot. The station's position and heading, the "
            f"rod's offset along its cradle, and both tools' poses vary per episode.\n"
            f"Goal: CHOP OUT the RED middle section. Pick up the KNIFE (brown "
            f"handle, thin bright blade), point the blade straight down, insert it "
            f"through the kerf slot, and press down FIRMLY on the red span — the "
            f"seam welds shear only under a deliberate press through the blade; "
            f"merely resting the blade on the rod does nothing. If one seam snaps "
            f"first, keep pressing on the dropped side until the second seam gives. "
            f"When both seams snap, the red section drops into the well. "
            f"Finish with the red section lying settled INSIDE the well, both beige "
            f"end sections still seated on their anvils, and the station upright. "
            f"Prying the station, using the cleaver, or leaving the red section "
            f"anywhere but inside the well fails. The physical sequence is "
            f"insert-then-press; no other ordering constraints apply."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pick up the thin-bladed knife, insert its blade down through the narrow "
            "slot in the chopping station's cover, and press hard on the red middle "
            "section of the rod until both seams snap and the red piece falls into "
            "the well below. Leave the beige end sections seated on their supports."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _station_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the station frame, (N,3) -> (N,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.station.data.root_quat_w,
                                  pos_w - self.station.data.root_pos_w)

    def seam_sep(self) -> torch.Tensor:
        """(N,2): distance between the two seam anchor points (weld-plane points in
        each segment's own frame) for seam A (out_a|mid) and seam B (mid|out_b)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        ex = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3)
        half = c.seg_len / 2
        a_end = self.out_a.data.root_pos_w \
            + quat_apply(self.out_a.data.root_quat_w, ex) * half
        b_end = self.out_b.data.root_pos_w \
            - quat_apply(self.out_b.data.root_quat_w, ex) * half
        m_a = self.mid.data.root_pos_w - quat_apply(self.mid.data.root_quat_w, ex) * half
        m_b = self.mid.data.root_pos_w + quat_apply(self.mid.data.root_quat_w, ex) * half
        return torch.stack([(a_end - m_a).norm(dim=-1), (b_end - m_b).norm(dim=-1)],
                           dim=1)

    def severed(self) -> torch.Tensor:
        """(N,2) bool, live: each seam's anchor points separated beyond `sever_sep`."""
        return self.seam_sep() > self.cfg.sever_sep

    def blade_tip_local(self) -> torch.Tensor:
        """(N,3): the knife blade's bottom-edge midpoint, station frame."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        tip = torch.tensor([c.blade_len / 2, 0.0, -c.blade_h],
                           device=self.env.device).expand(n, 3)
        tip_w = self.knife.data.root_pos_w + quat_apply(self.knife.data.root_quat_w, tip)
        return self._station_local(tip_w)

    def blade_engaged(self) -> torch.Tensor:
        """(N,) bool, live: the knife blade tip inside the kerf slot footprint,
        below the cover top and above the well floor."""
        c = self.cfg
        t = self.blade_tip_local()
        return (t[:, 0].abs() < c.slot_x_half) \
            & (t[:, 1].abs() < c.slot_y_half + 0.004) \
            & (t[:, 2] < c.cover_top) & (t[:, 2] > 0.010)

    def mid_in_well(self) -> torch.Tensor:
        """(N,) bool, live: the RED segment centre down inside the well."""
        c = self.cfg
        t = self._station_local(self.mid.data.root_pos_w)
        return (t[:, 0].abs() < c.well_x_half) & (t[:, 1].abs() < c.well_y_half + 0.004) \
            & (t[:, 2] > 0.004) & (t[:, 2] < c.well_z_max)

    def outers_seated(self) -> torch.Tensor:
        """(N,) bool, live: both beige segments still seated on their anvils."""
        c = self.cfg
        ok = None
        for body, sgn in ((self.out_a, -1.0), (self.out_b, 1.0)):
            t = self._station_local(body.data.root_pos_w)
            good = ((t[:, 0] * sgn) > c.well_x_half - c.seg_len / 2 - 0.004) \
                & ((t[:, 0] * sgn) < c.cradle_x_half) \
                & (t[:, 1].abs() < c.seat_y_tol) \
                & ((t[:, 2] - c.rod_rest_z).abs() < c.seat_z_tol)
            ok = good if ok is None else (ok & good)
        return ok

    def station_upright(self) -> torch.Tensor:
        """(N,) bool: station up-axis within tolerance of world-up, base near ground."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        up = quat_apply(self.station.data.root_quat_w, ez)
        near_ground = self.station.data.root_pos_w[:, 2].abs() < 0.02
        return (up[:, 2] > self.cfg.upright_min) & near_ground

    def settled(self) -> torch.Tensor:
        """(N,) bool: all rod segments |lin vel| below `settle_speed`."""
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                         for b in (self.out_a, self.out_b, self.mid)], dim=1)
        return (v < self.cfg.settle_speed).all(dim=1)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w for b in
                         (self.station, self.out_a, self.out_b, self.mid,
                          self.knife, self.cleaver)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        fin = self._finite()
        eng = self.blade_engaged() & fin
        self._engaged |= eng
        sev = self.severed()
        self._sev_a |= sev[:, 0] & fin
        self._sev_b |= sev[:, 1] & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: both seams severed (live seam-point readback), the RED segment
        settled inside the well, both beige segments seated on their anvils, the
        station upright, everything settled and finite. All physical outcomes."""
        self._update_latches()
        sev = self.severed()
        return sev[:, 0] & sev[:, 1] & self.mid_in_well() & self.outers_seated() \
            & self.station_upright() & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*engaged + 0.25*sev_a + 0.25*sev_b (all
        latched; ~0 for doing nothing — `engaged` requires the thin blade inside
        the kerf slot, the sev latches require real seam separation), capped at
        0.65 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_engaged * self._engaged.float()
                + c.w_sever * self._sev_a.float()
                + c.w_sever * self._sev_b.float()).clamp(max=0.65)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="kerf_chop", robot="null"))

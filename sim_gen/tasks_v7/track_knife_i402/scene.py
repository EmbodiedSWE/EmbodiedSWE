"""KnifeStandScene — BUILD the display stand, then rest the knife on it (sim_gen task
`track_knife_i402`, scene `knife_stand`, env `simgen.knife_stand`).

Derived from pick_place/track_knife, but STRATEGICALLY different: the seed starts with
the knife ALREADY RIGIDLY GRASPED in the Franka gripper and grades dense tracking of a
prescribed free-space waypoint path toward a plate — pure transport of a held object,
no contact event, no construction. Here the destination for the knife DOES NOT EXIST
at reset: it must be BUILT first. Two slotted panels form a cross-lap joint — the BLUE
base panel stands on its foot with a 16 mm slot cut down from its top edge; the RED
cap panel (no foot; it lies flat on a rack) has the mirror slot cut up from its bottom
edge. Pressing the cap panel down over the base at right angles meshes the two slots
into a free-standing X-stand whose top edges carry two cut NOTCHES. Only then can the
knife be laid across the pair of notches, level and diagonal, ~11 cm off the floor.
The physics forces the order: a knife has no resting place at the judged height until
the stand is meshed (on-end it is too low, leaned it is not level, perched on a lone
panel it fails the mesh clauses). Nothing is tracked, nothing starts in the hand, and
the load-bearing interactions are a press-fit insertion and a gravity seat — contact
events the seed never has.

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - panel_a (BLUE base): DYNAMIC slab 160 x 8 x 120 mm on a 160 x 110 x 10 mm foot
    (two pads, 24 mm centre channel); slot (16 mm) from the top edge down to
    mid-height at local x = 0; a 44 mm notch (12 mm deep) in the top edge at +45 mm.
  - panel_b (RED cap): DYNAMIC slab 160 x 8 x 120 mm, NO foot; slot from the bottom
    edge up to mid-height; same top-edge notch. It spawns LYING FLAT on the rack.
  - knife: DYNAMIC blade 140 x 20 x 5 mm + offset grip block (root/CoM at the blade
    centre); spawns flat on a low pedestal block so a parallel jaw can reach it.
  - rack: KINEMATIC two-rail cradle, 40 mm tall (elevates the cap panel so its edge
    overhangs for a jaw grasp).
  - block: KINEMATIC pedestal, 30 mm tall, under the knife.

Slot/slab faces are slick (polished joinery): the press-fit is FORM guidance, never a
friction fight. Per-episode randomization (readback-verifiable): base panel xy + free
yaw, rack xy + yaw with the cap panel's slide-position on it, pedestal xy + knife yaw.

Rubric (0..1, latched partial credit that never evaporates):
  0.20 cap panel ever ENGAGED in the base slot (upright over the base, >= 65 mm down)
  0.25 stand ever fully MESHED (cap seated to the floor, panels crossed ~90 deg)
  0.30 knife ever SEATED across both notches of the live meshed stand
  cap 0.75; exactly 1.0 iff success(): stand meshed and the knife seated in the notch
  band (level, on the live notch line) with everything at rest.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
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


# ----- custom compound spawners ---------------------------------------------------------------

_SPAWNER_CACHE: dict[str, Any] = {}


def _add_box(stage, path: str, *, center, size, color, collide: Callable, yaw: float = 0.0) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw != 0.0:
        xf.AddRotateZOp().Set(math.degrees(yaw))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _slick_material(stage, root_path: str):
    """Low-friction physics material (waxed joinery faces). The cross-lap press-fit is
    guided by FORM (slot walls), never won by friction; PhysX pair-averages friction,
    so both panels' slab faces carry this so slab-in-slot contact is slick-on-slick."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, f"{root_path}/slickMat")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(0.06)
    api.CreateDynamicFrictionAttr(0.05)
    api.CreateRestitutionAttr(0.0)
    return mat


def _bind_slick(stage, mat, *prim_paths: str) -> None:
    from pxr import UsdShade

    for p in prim_paths:
        prim = stage.GetPrimAtPath(p)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _root_xform(prim_path: str, translation, orientation):
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


def _rigid_dynamic(root, *, mass: float, lin_damp: float, ang_damp: float,
                   pos_iters: int = 16, vel_iters: int = 4,
                   com: tuple = (0.0, 0.0, 0.0)) -> None:
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    m = UsdPhysics.MassAPI.Apply(root)
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(lin_damp))
    pxrb.CreateAngularDampingAttr(float(ang_damp))
    pxrb.CreateSolverPositionIterationCountAttr(pos_iters)
    pxrb.CreateSolverVelocityIterationCountAttr(vel_iters)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)


def _spawn_panel(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one cross-lap panel. Root origin at the BOTTOM CENTRE of the slab in its
    UPRIGHT pose; slab plane = local x-z. `slot_top` True = base panel (foot, slot cut
    down from the top edge); False = cap panel (no foot, slot cut up from the bottom
    edge). Both carry the knife notch in the top edge at local x = +notch_x."""
    stage, root = _root_xform(prim_path, translation, orientation)
    # CoM inset up from the bottom edge: on the slab boundary (0,0,0) a flat rest can
    # never be statically stable (support hull cannot extend past the CoM edge) — the
    # cap panel tipped off the rack every episode.  20 mm puts the lying CoM 10 mm
    # inside the rail hull while keeping the press tilt damping discrete-stable
    # (I_tilt ~ 7.9e-4 -> c*dt/I = 0.79 < 1).
    _rigid_dynamic(root, mass=float(cfg.mass), lin_damp=0.06, ang_damp=0.60, pos_iters=24,
                   com=(0.0, 0.0, float(cfg.com_z)))
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    L, H, t = c.length, c.height, c.thick
    hw = c.slot_w / 2
    n0, n1 = c.notch_x - c.notch_w / 2, c.notch_x + c.notch_w / 2
    col = c.color
    dark = tuple(v * 0.55 for v in col)
    names = []

    def box(nm, x0, x1, z0, z1, cc):
        _add_box(stage, f"{prim_path}/{nm}", center=((x0 + x1) / 2, 0.0, (z0 + z1) / 2),
                 size=(x1 - x0, t, z1 - z0), color=cc, collide=collide)
        names.append(f"{prim_path}/{nm}")

    def notched_half(z0, z1):
        """Full-length band z0..z1 with the notch sunk notch_d below z1."""
        box("na", -L / 2, n0, z0, z1, col)
        box("nf", n0, n1, z0, z1 - c.notch_d, dark)
        box("nb", n1, L / 2, z0, z1, col)

    def slotted_half(z0, z1, notched: bool):
        """Band z0..z1 split by the centre slot; +x wing optionally notched at the top."""
        box("wm", -L / 2, -hw, z0, z1, col)
        if notched:
            box("wa", hw, n0, z0, z1, col)
            box("wf", n0, n1, z0, z1 - c.notch_d, dark)
            box("wb", n1, L / 2, z0, z1, col)
        else:
            box("wp", hw, L / 2, z0, z1, col)

    if c.slot_top:  # base: solid lower half + slotted (and notched) upper half + foot
        box("low", -L / 2, L / 2, 0.0, H / 2, col)
        slotted_half(H / 2, H, notched=True)
        # foot = two pads with a centre CHANNEL: the meshed cap panel's slab crosses
        # the foot footprint, so it must pass between the pads down to the floor
        for s, nm in ((1.0, "foot_p"), (-1.0, "foot_m")):
            fx0 = c.foot_gap / 2
            _add_box(stage, f"{prim_path}/{nm}",
                     center=(s * (fx0 + L / 2) / 2, 0.0, c.foot_t / 2),
                     size=(L / 2 - fx0, c.foot_w, c.foot_t), color=dark, collide=collide)
    else:  # cap: slotted lower half + solid notched upper half, no foot
        slotted_half(0.0, H / 2, notched=False)
        notched_half(H / 2, H)
    mat = _slick_material(stage, prim_path)
    _bind_slick(stage, mat, *names)  # foot keeps default friction: it anchors the base
    return root


def _spawn_knife(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the knife: DYNAMIC blade slab + offset grip block, root/CoM at the blade
    centre (the grip is outboard of the notch span so the blade alone takes the seat)."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_dynamic(root, mass=float(cfg.mass), lin_damp=0.08, ang_damp=0.40)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_box(stage, f"{prim_path}/blade", center=(0.0, 0.0, 0.0),
             size=(c.blade_l, c.blade_w, c.blade_t), color=(0.80, 0.82, 0.86),
             collide=collide)
    gx = -(c.blade_l / 2 + c.grip_l / 2 - 0.006)
    _add_box(stage, f"{prim_path}/grip", center=(gx, 0.0, c.grip_t / 2 - c.blade_t / 2),
             size=(c.grip_l, c.grip_w, c.grip_t), color=(0.20, 0.12, 0.06),
             collide=collide)
    return root


def _spawn_rack(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the rack: KINEMATIC two-rail cradle (rails run along local x)."""
    stage, root = _root_xform(prim_path, translation, orientation)
    from pxr import UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    for s, nm in ((1.0, "rail_p"), (-1.0, "rail_m")):
        _add_box(stage, f"{prim_path}/{nm}", center=(0.0, s * c.rail_dy, c.rack_h / 2),
                 size=(c.rail_len, c.rail_w, c.rack_h), color=(0.45, 0.36, 0.22),
                 collide=collide)
    return root


def _spawn_block(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the pedestal: KINEMATIC block the knife lies across (long axis local y)."""
    stage, root = _root_xform(prim_path, translation, orientation)
    from pxr import UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_box(stage, f"{prim_path}/top", center=(0.0, 0.0, c.block_h / 2),
             size=(c.block_x, c.block_y, c.block_h), color=(0.30, 0.30, 0.34),
             collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "panel" not in _SPAWNER_CACHE:

        @configclass
        class PanelSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_panel)
            slot_top: bool = True
            mass: float = 1.2
            color: tuple = (0.15, 0.35, 0.80)
            length: float = 0.160
            height: float = 0.120
            thick: float = 0.008
            slot_w: float = 0.016
            notch_x: float = 0.045
            notch_w: float = 0.044
            notch_d: float = 0.012
            foot_w: float = 0.110
            foot_t: float = 0.010
            foot_gap: float = 0.024
            com_z: float = 0.020
            contact_offset: float = 0.0012

        @configclass
        class KnifeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_knife)
            mass: float = 0.07
            blade_l: float = 0.140
            blade_w: float = 0.020
            blade_t: float = 0.005
            grip_l: float = 0.040
            grip_w: float = 0.026
            grip_t: float = 0.014
            contact_offset: float = 0.0012

        @configclass
        class RackSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rack)
            rack_h: float = 0.040
            rail_dy: float = 0.035
            rail_len: float = 0.150
            rail_w: float = 0.020
            contact_offset: float = 0.0015

        @configclass
        class BlockSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_block)
            block_h: float = 0.030
            block_x: float = 0.030
            block_y: float = 0.080
            contact_offset: float = 0.0015

        _SPAWNER_CACHE.update(panel=PanelSpawnerCfg, knife=KnifeSpawnerCfg,
                              rack=RackSpawnerCfg, block=BlockSpawnerCfg)
    return _SPAWNER_CACHE


def _qz(yaw: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(yaw.shape[0], 4, device=yaw.device)
    q[:, 0] = torch.cos(yaw / 2)
    q[:, 3] = torch.sin(yaw / 2)
    return q


def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], dim=-1)


def _yaw_of(q: torch.Tensor) -> torch.Tensor:
    return torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                       1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def _up_z(q: torch.Tensor) -> torch.Tensor:
    """World z-component of the body z-axis (1 = upright)."""
    return 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)


def _axis_x(q: torch.Tensor) -> torch.Tensor:
    """(N, 3) world direction of the body x-axis."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([1.0 - 2.0 * (y ** 2 + z ** 2),
                        2.0 * (x * y + w * z),
                        2.0 * (x * z - w * y)], dim=-1)


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class KnifeStandSceneCfg(BaseCfg):
    """Config for `KnifeStandScene`. Honesty asserted in __post_init__: the slot passes
    the slab with real clearance (net of contact offsets); the notch admits the blade
    and sinks it below the rim; the judged knife band is above every stand-free rest
    height yet below a rim-perch; the blade spans the notch line with the grip clear of
    both seats; panel edges and the blade fit the Franka jaw; all work is in reach."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    mesh_xy_tol: float = tunable(0.012)   # cap-centre xy distance from base centre when meshed
    seat_z_tol: float = tunable(0.010)    # cap root z above the floor when fully seated
    cross_yaw_deg: float = tunable(20.0)  # |panel yaw difference - 90 deg| tolerance
    upright_deg: float = tunable(12.0)    # panel tilt tolerance from vertical
    engage_z: float = tunable(0.055)      # cap root z below this while aligned = engaged
    kz_lo: float = tunable(0.100)         # knife-centre z band (seated in the notches)
    kz_hi: float = tunable(0.117)
    knife_xy_tol: float = tunable(0.015)  # knife centre from the live notch midpoint
    knife_yaw_deg: float = tunable(20.0)  # knife axis from the live notch line (mod 180)
    knife_flat_deg: float = tunable(8.0)  # knife axis from horizontal; a skew half-perch
    #   (one end in a notch, one on a rim) tilts ~10.7 deg and must be refused
    settle_speed: float = tunable(0.05)   # max |lin vel| of judged bodies when judging (m/s)
    settle_omega: float = tunable(0.60)   # max |ang vel| of judged bodies when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    a_jitter: float = tunable(0.035)      # base panel xy jitter (+/- m); yaw is free
    rack_jitter: float = tunable(0.030)   # rack xy jitter (+/- m)
    rack_yaw_deg: float = tunable(15.0)   # rack yaw (+/- deg)
    b_slide: float = tunable(0.010)       # cap panel slide along the rails (+/- m)
    block_jitter: float = tunable(0.030)  # pedestal xy jitter (+/- m)
    knife_yaw: float = tunable(0.25)      # knife yaw on the pedestal (+/- rad)
    knife_jit: float = tunable(0.005)     # knife xy jitter on the pedestal (+/- m)

    # --- info: layout (single Franka base at the origin) ----------------------------------------
    a_pos: tuple = info((0.42, 0.02))     # base panel footprint centre (before jitter)
    rack_pos: tuple = info((0.16, -0.30)) # rack centre (before jitter)
    block_pos: tuple = info((0.14, 0.30)) # pedestal centre (before jitter)

    # --- info: geometry (mirrors the spawner defaults) ------------------------------------------
    length: float = info(0.160)
    height: float = info(0.120)
    thick: float = info(0.008)
    slot_w: float = info(0.016)
    notch_x: float = info(0.045)
    notch_w: float = info(0.044)  # wide: the blade crosses each panel at ~45 deg, so its
    #   plan footprint through the 8 mm wall is blade_w/sin45 + thick ~ 36.3 mm
    notch_d: float = info(0.012)
    foot_w: float = info(0.110)
    foot_t: float = info(0.010)
    foot_gap: float = info(0.024)  # centre channel between the foot pads
    a_mass: float = info(1.2)  # weighted base: it must anchor against insertion drag
    b_mass: float = info(0.28)
    blade_l: float = info(0.140)
    blade_w: float = info(0.020)
    blade_t: float = info(0.005)
    grip_l: float = info(0.040)
    knife_mass: float = info(0.07)
    rack_h: float = info(0.040)
    rail_dy: float = info(0.035)
    rail_w: float = info(0.020)
    panel_com_z: float = info(0.020)  # CoM inset from the panel bottom edge (stability)
    b_rack_off: float = info(0.055)   # cap-panel root offset along rack-local +y
    block_h: float = info(0.030)
    contact_offset: float = info(0.0012)
    blue: tuple = info((0.15, 0.35, 0.80))
    red: tuple = info((0.80, 0.15, 0.12))

    # rubric weights (0.20 + 0.25 + 0.30 = 0.75 = the non-success cap)
    w_engage: float = info(0.20)
    w_mesh: float = info(0.25)
    w_knife: float = info(0.30)

    # Derived (filled in __post_init__).
    notch_r: float = field(default=None, init=False)    # notch offset from the panel centre
    knife_rest_z: float = field(default=None, init=False)
    b_lie_z: float = field(default=None, init=False)    # cap panel root z lying on the rack
    knife_lie_z: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.notch_r = self.notch_x
        self.knife_rest_z = self.height - self.notch_d + self.blade_t / 2
        self.b_lie_z = self.rack_h + self.thick / 2 + 0.0008
        self.knife_lie_z = self.block_h + self.blade_t / 2 + 0.0008
        # press-fit honesty: the slot passes the slab with clearance net of both
        # contact offsets, and the notch admits (and sinks) the blade
        assert self.slot_w - self.thick > 2 * self.contact_offset + 0.0008, \
            "slot does not pass the slab"
        # the blade crosses each panel at ~45 deg: its plan-view footprint through the
        # wall is blade_w/sin45 + thick; the notch must pass that with clearance
        diag = self.blade_w / math.sin(math.pi / 4) + self.thick
        assert self.notch_w > diag + 0.005, "notch does not admit the 45-deg blade crossing"
        assert self.notch_d > self.blade_t + 0.004, "notch does not sink the blade below the rim"
        # the notch must stay inside the +x top-edge wing on both panel variants
        assert self.notch_x - self.notch_w / 2 > self.slot_w / 2 + 0.008, \
            "notch breaks into the centre slot"
        assert self.notch_x + self.notch_w / 2 < self.length / 2 - 0.008, \
            "notch breaks the panel end"
        # the judged knife band: strictly above every stand-free rest (flat on the
        # floor/pedestal/lying cap; on-end on the grip tip reaches ~0.082) and strictly
        # below a rim-perch across the un-notched top edges (0.1225)
        on_end = self.blade_l / 2 + 0.014
        assert self.kz_lo > on_end + 0.004, "knife band reachable standing on end"
        assert self.kz_hi < self.height + self.blade_t / 2 - 0.003, \
            "knife band admits a rim-perch on the un-notched top edges"
        assert self.kz_lo < self.knife_rest_z < self.kz_hi, "true notch seat outside the band"
        # the blade spans the diagonal notch line; the grip stays clear of both seats
        # (along the knife axis, each wall contact zone extends thick/sin45 about the
        # notch centre at +/- span/2 from the midpoint)
        span = self.notch_r * math.sqrt(2)
        seat_out = span / 2 + self.thick / math.sin(math.pi / 4) / 2 + 0.004
        assert self.blade_l / 2 > seat_out + 0.006, "blade too short for the notch line"
        grip_inner = self.blade_l / 2 - 0.006  # grip starts here (spawner gx geometry)
        assert grip_inner > seat_out + 0.004, "grip lands on a notch seat"
        # cross-lap honesty: engaged threshold is genuinely below half-mesh; a seated
        # cap rests its wing bottoms on the floor at ~0
        assert self.engage_z < self.height / 2, "engage threshold above half-mesh"
        # the foot channel passes the cap slab at its worst in-slot lateral play
        assert self.foot_gap / 2 > self.slot_w / 2 + 0.002, "foot pads block the seating cap"
        # an upright-but-unmeshed cap beside the base keeps centres >= ~half a panel
        # apart (slabs must not intersect), far outside mesh_xy_tol
        assert self.mesh_xy_tol < self.length / 4, "mesh tolerance admits a beside-park"
        # embodiment: 8 mm panel edges and the 5 mm blade fit the 80 mm Franka jaw;
        # the rack/pedestal elevate their payloads for a top grasp
        assert self.thick < 0.078 and self.blade_t < 0.078, "jaw cannot pass the stock"
        assert self.rack_h >= 0.025 and self.block_h >= 0.02, "payload too low to grasp"
        # rack rest honesty: lying flat, the cap panel's authored CoM must sit inside
        # the rail support hull (CoM at the slab boundary edge can never be stable)
        com_y = self.b_rack_off - self.panel_com_z          # rack-local y of the lying CoM
        assert com_y < self.rail_dy + self.rail_w / 2 - 0.008, "lying cap CoM overhangs the rails"
        assert com_y > -(self.rail_dy + self.rail_w / 2) + 0.008, "lying cap CoM overhangs the rails"
        # ... and the slab still covers both rails
        assert self.b_rack_off - self.height < -(self.rail_dy + self.rail_w / 2), \
            "cap slab does not span the far rail"
        # reach: the farthest work point stays inside the 0.855 m Franka envelope
        far = math.hypot(self.a_pos[0], self.a_pos[1]) + self.a_jitter + self.length / 2
        assert far < 0.62, "base panel work exceeds the reach envelope"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("knife_stand")
class KnifeStandScene(BaseScene):
    cfg: KnifeStandSceneCfg

    def __init__(self, cfg: KnifeStandSceneCfg | None = None) -> None:
        super().__init__(cfg or KnifeStandSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        rigid = sim_utils.RigidBodyPropertiesCfg()
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "panel_a": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PanelA",
                spawn=sp["panel"](mass_props=sim_utils.MassPropertiesCfg(mass=c.a_mass),
                                  rigid_props=rigid, slot_top=True, mass=c.a_mass,
                                  color=c.blue, length=c.length, height=c.height,
                                  thick=c.thick, slot_w=c.slot_w, notch_x=c.notch_x,
                                  notch_w=c.notch_w, notch_d=c.notch_d,
                                  foot_w=c.foot_w, foot_t=c.foot_t, foot_gap=c.foot_gap,
                                  com_z=c.panel_com_z, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.a_pos[0], c.a_pos[1], 0.0)),
            ),
            "panel_b": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PanelB",
                spawn=sp["panel"](mass_props=sim_utils.MassPropertiesCfg(mass=c.b_mass),
                                  rigid_props=rigid, slot_top=False, mass=c.b_mass,
                                  color=c.red, length=c.length, height=c.height,
                                  thick=c.thick, slot_w=c.slot_w, notch_x=c.notch_x,
                                  notch_w=c.notch_w, notch_d=c.notch_d,
                                  com_z=c.panel_com_z, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rack_pos[0], c.rack_pos[1] + c.b_rack_off, c.b_lie_z)),
            ),
            "knife": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Knife",
                spawn=sp["knife"](mass_props=sim_utils.MassPropertiesCfg(mass=c.knife_mass),
                                  rigid_props=rigid, mass=c.knife_mass,
                                  blade_l=c.blade_l, blade_w=c.blade_w, blade_t=c.blade_t,
                                  grip_l=c.grip_l, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.block_pos[0], c.block_pos[1], c.knife_lie_z)),
            ),
            "rack": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rack",
                spawn=sp["rack"](mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
                                 rigid_props=kin, rack_h=c.rack_h, rail_dy=c.rail_dy,
                                 rail_w=c.rail_w),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rack_pos[0], c.rack_pos[1], 0.0)),
            ),
            "block": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Block",
                spawn=sp["block"](mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
                                  rigid_props=kin, block_h=c.block_h),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.block_pos[0], c.block_pos[1], 0.0)),
            ),
        }

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
        self.panel_a: RigidObject = env.iscene["panel_a"]
        self.panel_b: RigidObject = env.iscene["panel_b"]
        self.knife: RigidObject = env.iscene["knife"]
        self.rack: RigidObject = env.iscene["rack"]
        self.block: RigidObject = env.iscene["block"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self._a_xy = torch.tensor(self.cfg.a_pos, device=dev).repeat(n, 1)
        self._a_yaw = torch.zeros(n, device=dev)
        self._rack_xy = torch.tensor(self.cfg.rack_pos, device=dev).repeat(n, 1)
        self._rack_yaw = torch.zeros(n, device=dev)
        self._block_xy = torch.tensor(self.cfg.block_pos, device=dev).repeat(n, 1)
        self._engaged_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._meshed_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._knife_ever = torch.zeros(n, dtype=torch.bool, device=dev)

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: base panel upright with xy jitter + FREE yaw; rack with xy
        jitter + yaw and the cap panel LYING FLAT across its rails (slide jitter);
        pedestal with xy jitter and the knife flat on it (yaw + xy jitter); clear
        latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        axy = torch.tensor(c.a_pos, device=dev).expand(m, 2) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.a_jitter
        ayaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = axy
        st[:, 2] = 0.0005
        st[:, 3:7] = _qz(ayaw)
        st[:, 0:3] += origin
        self.panel_a.write_root_state_to_sim(st, env_ids)
        self._a_xy[env_ids] = axy
        self._a_yaw[env_ids] = ayaw

        rxy = torch.tensor(c.rack_pos, device=dev).expand(m, 2) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.rack_jitter
        ryaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rack_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = rxy
        st[:, 3:7] = _qz(ryaw)
        st[:, 0:3] += origin
        self.rack.write_root_state_to_sim(st, env_ids)
        self._rack_xy[env_ids] = rxy
        self._rack_yaw[env_ids] = ryaw

        # cap panel lying flat: root at rack-local (slide, +b_rack_off); upright local z
        # maps to rack-local -y (Rx(+90 deg)), slab 8 mm thick over both rails
        qx90 = torch.zeros(m, 4, device=dev)
        qx90[:, 0] = math.cos(math.pi / 4)
        qx90[:, 1] = math.sin(math.pi / 4)
        slide = (torch.rand(m, device=dev) * 2 - 1) * c.b_slide
        cy, sy = torch.cos(ryaw), torch.sin(ryaw)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = rxy[:, 0] + cy * slide - sy * c.b_rack_off
        st[:, 1] = rxy[:, 1] + sy * slide + cy * c.b_rack_off
        st[:, 2] = c.b_lie_z
        st[:, 3:7] = _qmul(_qz(ryaw), qx90)
        st[:, 0:3] += origin
        self.panel_b.write_root_state_to_sim(st, env_ids)

        bxy = torch.tensor(c.block_pos, device=dev).expand(m, 2) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.block_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = bxy
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.block.write_root_state_to_sim(st, env_ids)
        self._block_xy[env_ids] = bxy

        kyaw = (torch.rand(m, device=dev) * 2 - 1) * c.knife_yaw
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = bxy + (torch.rand(m, 2, device=dev) * 2 - 1) * c.knife_jit
        st[:, 2] = c.knife_lie_z
        st[:, 3:7] = _qz(kyaw)
        st[:, 0:3] += origin
        self.knife.write_root_state_to_sim(st, env_ids)

        self._engaged_ever[env_ids] = False
        self._meshed_ever[env_ids] = False
        self._knife_ever[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "panel_a": self.panel_a.data.root_state_w[env_ids].clone(),
            "panel_b": self.panel_b.data.root_state_w[env_ids].clone(),
            "knife": self.knife.data.root_state_w[env_ids].clone(),
            "rack": self.rack.data.root_state_w[env_ids].clone(),
            "block": self.block.data.root_state_w[env_ids].clone(),
            "a_xy": self._a_xy[env_ids].clone(),
            "a_yaw": self._a_yaw[env_ids].clone(),
            "rack_xy": self._rack_xy[env_ids].clone(),
            "rack_yaw": self._rack_yaw[env_ids].clone(),
            "block_xy": self._block_xy[env_ids].clone(),
            "engaged_ever": self._engaged_ever[env_ids].clone(),
            "meshed_ever": self._meshed_ever[env_ids].clone(),
            "knife_ever": self._knife_ever[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.panel_a.write_root_state_to_sim(state["panel_a"], env_ids)
        self.panel_b.write_root_state_to_sim(state["panel_b"], env_ids)
        self.knife.write_root_state_to_sim(state["knife"], env_ids)
        self.rack.write_root_state_to_sim(state["rack"], env_ids)
        self.block.write_root_state_to_sim(state["block"], env_ids)
        self._a_xy[env_ids] = state["a_xy"]
        self._a_yaw[env_ids] = state["a_yaw"]
        self._rack_xy[env_ids] = state["rack_xy"]
        self._rack_yaw[env_ids] = state["rack_yaw"]
        self._block_xy[env_ids] = state["block_xy"]
        self._engaged_ever[env_ids] = state["engaged_ever"]
        self._meshed_ever[env_ids] = state["meshed_ever"]
        self._knife_ever[env_ids] = state["knife_ever"]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A BLUE base panel ({c.length * 100:.0f} x {c.height * 100:.0f} cm, "
            f"{c.thick * 1000:.0f} mm thick) stands upright on its foot near "
            f"({c.a_pos[0]:.2f}, {c.a_pos[1]:.2f}) (xy and yaw change per episode). A "
            f"{c.slot_w * 1000:.1f} mm SLOT is cut DOWN from the middle of its top edge to "
            f"half height, and a {c.notch_w * 1000:.0f} mm knife NOTCH "
            f"({c.notch_d * 1000:.0f} mm deep) is cut into its top edge {c.notch_x * 100:.1f} cm "
            f"to one side of the slot. A RED cap panel of the same size lies FLAT on a two-rail "
            f"rack near ({c.rack_pos[0]:.2f}, {c.rack_pos[1]:.2f}); it has NO foot, the mirror "
            f"slot cut UP from the middle of its bottom edge to half height, and the same "
            f"top-edge notch. A steel KNIFE (blade {c.blade_l * 100:.0f} x "
            f"{c.blade_w * 100:.0f} cm, {c.blade_t * 1000:.0f} mm thick, dark grip at one end, "
            f"root at the blade centre) lies flat on a low pedestal near "
            f"({c.block_pos[0]:.2f}, {c.block_pos[1]:.2f}). Rack and pedestal move per episode.\n"
            f"Goal: BUILD the knife stand, then use it. Stand the red cap panel over the blue "
            f"base at RIGHT ANGLES and press it straight down so the two half-slots mesh into a "
            f"cross-lap joint — fully seated, its bottom edges rest on the floor (root z under "
            f"{c.seat_z_tol * 100:.0f} cm) with panel centres within {c.mesh_xy_tol * 100:.1f} cm "
            f"and the crossing within {c.cross_yaw_deg:.0f} deg of square. That puts the two "
            f"notches {c.notch_x * 100:.1f} cm out from the joint on perpendicular top edges. "
            f"Then lay the knife LEVEL across BOTH notches — blade centre within "
            f"{c.knife_xy_tol * 100:.1f} cm of the notch midpoint, aligned with the notch line "
            f"within {c.knife_yaw_deg:.0f} deg, sunk in the notch band "
            f"({c.kz_lo * 100:.1f}-{c.kz_hi * 100:.1f} cm up) — and leave everything at rest. "
            f"The knife has no legal resting place until the stand is meshed: on end it is too "
            f"low, leaning it is not level, and perched anywhere else it is off the notch line."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Build the knife stand first: take the red slotted panel off the rack, hold it "
            "upright at right angles over the blue standing panel, and press it straight down "
            "so their half-slots mesh into a cross-lap X that sits flat on the floor. Then "
            "take the knife from the pedestal and lay it level across the two notches in the "
            "panels' top edges, and leave everything at rest."
        )

    # ----- readings / rubric ---------------------------------------------------------------------
    def notch_points(self) -> tuple[torch.Tensor, torch.Tensor]:
        """((N, 2), (N, 2)) LIVE world xy (env-local) of the two notch centres."""
        c = self.cfg
        out = []
        for p in (self.panel_a, self.panel_b):
            pos = p.data.root_pos_w - self.env_origins
            ax = _axis_x(p.data.root_quat_w)
            out.append(pos[:, 0:2] + c.notch_r * ax[:, 0:2])
        return out[0], out[1]

    def meshed(self) -> torch.Tensor:
        """(N,) bool: the cross-lap joint is live — both panels upright, centres
        coincident in xy, the cap seated to the floor, crossing ~90 deg."""
        c = self.cfg
        pa = self.panel_a.data.root_pos_w - self.env_origins
        pb = self.panel_b.data.root_pos_w - self.env_origins
        qa, qb = self.panel_a.data.root_quat_w, self.panel_b.data.root_quat_w
        up = math.cos(math.radians(c.upright_deg))
        dyaw = _wrap(_yaw_of(qb) - _yaw_of(qa)).abs()
        cross = (dyaw - math.pi / 2).abs() < math.radians(c.cross_yaw_deg)
        return ((pa[:, 0:2] - pb[:, 0:2]).norm(dim=-1) <= c.mesh_xy_tol) \
            & (_up_z(qa) > up) & (_up_z(qb) > up) \
            & (pb[:, 2] <= c.seat_z_tol) & (pa[:, 2] <= c.seat_z_tol) & cross

    def engaged(self) -> torch.Tensor:
        """(N,) bool: the cap panel is upright over the base with its slab well into
        the base slot (root descended past engage_z)."""
        c = self.cfg
        pa = self.panel_a.data.root_pos_w - self.env_origins
        pb = self.panel_b.data.root_pos_w - self.env_origins
        up = math.cos(math.radians(c.upright_deg))
        return ((pa[:, 0:2] - pb[:, 0:2]).norm(dim=-1) <= c.mesh_xy_tol) \
            & (_up_z(self.panel_b.data.root_quat_w) > up) & (pb[:, 2] <= c.engage_z)

    def knife_seated(self) -> torch.Tensor:
        """(N,) bool: the knife lies level in the judged band, centred on the LIVE
        notch midpoint and aligned with the live notch line (mod 180 deg)."""
        c = self.cfg
        na, nb = self.notch_points()
        mid = (na + nb) / 2
        line = nb - na
        p = self.knife.data.root_pos_w - self.env_origins
        ax = _axis_x(self.knife.data.root_quat_w)
        flat = ax[:, 2].abs() < math.sin(math.radians(c.knife_flat_deg))
        d = (p[:, 0:2] - mid).norm(dim=-1) <= c.knife_xy_tol
        band = (p[:, 2] >= c.kz_lo) & (p[:, 2] <= c.kz_hi)
        la = torch.atan2(line[:, 1], line[:, 0])
        ka = torch.atan2(ax[:, 1], ax[:, 0])
        dline = _wrap(ka - la).abs()
        aligned = torch.minimum(dline, math.pi - dline) < math.radians(c.knife_yaw_deg)
        return band & flat & d & aligned

    def _still(self) -> torch.Tensor:
        c = self.cfg
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for b in (self.panel_a, self.panel_b, self.knife):
            ok &= ((b.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                   & (b.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega))
        return ok

    def _update_latches(self) -> None:
        mesh = self.meshed()
        self._engaged_ever |= self.engaged()
        self._meshed_ever |= mesh
        # knife credit only counts on the LIVE meshed stand: a blade perched on a
        # lone panel (or held at height) can never earn it
        self._knife_ever |= mesh & self.knife_seated()

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: the cross-lap stand is meshed AND the knife rests seated across
        both notches AND everything is at rest. Physical outcomes only."""
        self._update_latches()
        return self.meshed() & self.knife_seated() & self._still()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: latched 0.20 (cap ever engaged in the base slot)
        + 0.25 (stand ever fully meshed) + 0.30 (knife ever seated on the live meshed
        stand) — ~0 for doing nothing, capped 0.75 — exactly 1.0 iff success()."""
        c = self.cfg
        self._update_latches()
        base = (c.w_engage * self._engaged_ever.float()
                + c.w_mesh * self._meshed_ever.float()
                + c.w_knife * self._knife_ever.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="knife_stand", robot="null"))

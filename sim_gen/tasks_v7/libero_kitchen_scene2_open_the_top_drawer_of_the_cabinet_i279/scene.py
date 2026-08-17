"""GateHopperScene — lift the sealed hopper, TILT it over the blue bin so its flush,
handleless floor gate slides open under gravity and the captive ball drops in, then set
the hopper back down (sim_gen task
`libero_kitchen_scene2_open_the_top_drawer_of_the_cabinet_i279`).

Derived from libero_90/libero_kitchen_scene2_open_the_top_drawer_of_the_cabinet ("open
the top drawer of the cabinet": walk up to an ANCHORED cabinet, grasp the drawer's
HANDLE, and pull the prismatic joint past a threshold — one direct pulling contact on
the sliding member itself, judged by a joint reading). Here the ACTUATION MODEL of the
sliding member is inverted and the slide is demoted from goal to gate:

  - The "drawer" is the floor GATE of a free-standing, fully enclosed HOPPER: a slick
    captive plate riding in rails, its only exposed face FLUSH with the hopper's front
    wall. There is NO handle and no graspable lip — the gate physically cannot be
    opened by pulling on it. The seed's entire plan (grasp the sliding member, pull)
    is impossible by construction.
  - The gate is actuated INDIRECTLY, by reorienting its housing: lift the hopper off
    the ground (roof handle) and pitch it nose-down; gravity slides the plate out
    through its slot until the rear stop-tab catches. Level the hopper again and the
    gate merely stays put — orientation of the WHOLE housing is the only control input
    the mechanism responds to.
  - Opening the gate is not the goal but the MEANS: the hopper imprisons an orange
    ball (walls + roof + gate — the only exit is the floor opening the plate vacates).
    The goal is DELIVERY: hold the hopper above the blue catch bin when the floor
    opens so the ball falls INTO the bin, then park the emptied hopper upright on the
    ground, clear of the bin. A hopper tilted while it still stands on the ground
    dumps the ball onto the ground — unrecoverable by the gate route — so the lift
    must come first and the aim must be right when the floor lets go.

A solver therefore needs: (1) a carry grasp on the housing (not on the mechanism);
(2) a positioning judgment (hover the bottom opening over the bin BEFORE tilting);
(3) gravity-actuation of a captive slide through housing orientation; (4) a set-down.
None of these exist in the seed, and the seed's one skill (pull the slide directly) is
useless here.

Assets are fully procedural (compound-spawner pattern):
  - hopper housing: DYNAMIC compound body (side/rear walls, front sill + upper front
    wall leaving a plate-height slot, slick internal rails and keeper strips that
    guide the plate through its full stroke, full roof, and a roof HANDLE bar sized
    for a parallel jaw). Interior cavity 100 x 80 x 59 mm above the plate.
  - gate plate: DYNAMIC slick plate (107 x 77.6 x 6 mm) with a rear stop-tab; captive
    between rails, keepers, slot and rear wall — max extension 93 mm, at which the
    floor opening (64 mm wide, 99 mm long) has fully vacated the ball's footprint.
  - ball: orange sphere r 24 mm — fits the floor opening with 16 mm slack, cannot
    pass the 7.5 mm plate slot, cannot be grasped (fully enclosed until delivered).
  - bin: KINEMATIC blue square catch bin (115 mm inner, 35 mm wall) on the ground.

Per-episode randomization (readback-verifiable): hopper xy + full yaw, ball position
inside the cavity, bin xy + full yaw.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  0.15 * lift  — hopper base ever above `lift_z` with the ball still inside it
  0.25 * gate  — gate extension past `gate_open_ext` while the hopper is aloft
  0.30 * drop  — the ball inside the bin (bin frame, below the rim)
  1.0 iff success() — ball settled inside the bin AND the hopper parked upright on
                    the ground with its centre at least `clear_min` from the bin,
                    everything settled and finite. Non-success is capped at 0.70.

Honesty by construction (asserted in __post_init__): the ball fits through the floor
opening but not the slot; the stop-tab travel exceeds the travel that vacates the
ball's footprint; `clear_min` exceeds the hopper/bin no-touch distance; the spawn
layout parks the hopper far beyond `clear_min` with the ball inside it, so the null
policy scores ~0 and only the ball's delivery is ever at stake.

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


# ----- custom compound spawners ------------------------------------------------------------------
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


def _rigid_body(root, mass: float, *, com=None, lin_damp=0.1, ang_damp=0.3,
                kinematic=False) -> None:
    """Apply RigidBody + Mass + Physx armor to a compound root (explicit CoM —
    MassAPI mass alone leaves the CoM at the body origin, which is wanted here only
    because we author the CoM explicitly)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    m = UsdPhysics.MassAPI.Apply(root)
    m.CreateMassAttr(float(mass))
    if com is not None:
        m.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(lin_damp))
    pxrb.CreateAngularDampingAttr(float(ang_damp))
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)


def _friction_material(stage, path: str, static: float, dynamic: float):
    """Author a UsdPhysics material (custom-spawner colliders otherwise fall back to
    the ~0.5 default no matter what the Cfg says)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _make_collide(contact_offset: float, material=None) -> Callable:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)
        if material is not None:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics")

    return collide


def _add_box(stage, path: str, *, center, size, color, collide: Callable):
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _spawn_hopper(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the hopper housing: one DYNAMIC compound body. Local frame: origin at
    the footprint centre, z = 0 at the base plane (rest on the ground = pos.z ~ 0);
    the gate slides toward local +x (the nose). Two materials: structural (walls,
    roof, handle — grips the ground when parked) and SLICK (rails, sill, keepers —
    every face the plate rubs, so the pair-averaged contact friction stays low and
    a modest nose-down pitch slides the gate)."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    hl, hw, t = c.cav_l / 2, c.cav_w / 2, c.wall_t
    _rigid_body(root, float(c.hopper_mass), com=(0.0, 0.0, 0.045),
                lin_damp=0.2, ang_damp=0.5)
    mat_s = _friction_material(stage, f"{prim_path}/mat_struct", 0.70, 0.60)
    mat_k = _friction_material(stage, f"{prim_path}/mat_slick",
                               c.slick_mu, max(c.slick_mu - 0.02, 0.02))
    cs = _make_collide(c.contact_offset, mat_s)
    ck = _make_collide(c.contact_offset, mat_k)
    wood, dark = c.body_color, c.trim_color
    wall_h = c.roof_z
    slot_top = c.rail_h + c.plate_t + c.slot_gap
    # side walls (full height, full outer length)
    for s in (-1.0, 1.0):
        _add_box(stage, f"{prim_path}/side_{'p' if s > 0 else 'n'}",
                 center=(0.0, s * (hw + t / 2), wall_h / 2),
                 size=(2 * hl + 2 * t, t, wall_h), color=wood, collide=cs)
    # rear wall
    _add_box(stage, f"{prim_path}/rear", center=(-(hl + t / 2), 0.0, wall_h / 2),
             size=(t, 2 * hw, wall_h), color=wood, collide=cs)
    # front sill (below the slot; its top is the slot floor — slick)
    _add_box(stage, f"{prim_path}/sill", center=(hl + t / 2, 0.0, c.rail_h / 2),
             size=(t, 2 * hw, c.rail_h), color=dark, collide=ck)
    # front upper wall (above the slot)
    _add_box(stage, f"{prim_path}/front_up",
             center=(hl + t / 2, 0.0, (slot_top + wall_h) / 2),
             size=(t, 2 * hw, wall_h - slot_top), color=wood, collide=cs)
    # rails (the plate rides on their tops — slick)
    for s in (-1.0, 1.0):
        _add_box(stage, f"{prim_path}/rail_{'p' if s > 0 else 'n'}",
                 center=(0.0, s * (hw - c.rail_w / 2), c.rail_h / 2),
                 size=(2 * hl, c.rail_w, c.rail_h), color=dark, collide=ck)
    # keeper strips (above the plate edges: stop the extended plate from seesawing)
    keep_z0 = c.rail_h + c.plate_t + c.keeper_gap
    for s in (-1.0, 1.0):
        _add_box(stage, f"{prim_path}/keep_{'p' if s > 0 else 'n'}",
                 center=(0.0, s * (hw - c.keeper_w / 2), keep_z0 + 0.003),
                 size=(2 * hl, c.keeper_w, 0.006), color=dark, collide=ck)
    # roof (seals the top — the gate opening is the ONLY way out)
    _add_box(stage, f"{prim_path}/roof", center=(0.0, 0.0, wall_h + 0.003),
             size=(2 * hl + 2 * t, 2 * hw + 2 * t, 0.006), color=wood, collide=cs)
    # handle: two posts + a jaw-sized bar across the roof
    for s in (-1.0, 1.0):
        _add_box(stage, f"{prim_path}/post_{'p' if s > 0 else 'n'}",
                 center=(0.0, s * 0.030, wall_h + 0.006 + 0.012),
                 size=(0.010, 0.010, 0.024), color=dark, collide=cs)
    _add_box(stage, f"{prim_path}/bar",
             center=(0.0, 0.0, wall_h + 0.006 + 0.024 + 0.006),
             size=(0.012, 0.072, 0.012), color=dark, collide=cs)
    return root


def _spawn_plate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the gate plate: DYNAMIC slick plate + rear stop-tab, origin at the
    plate slab's centre. The tab hangs BELOW the plate's rear edge, down into the
    floor opening (between the rails, clear of the ball's cavity), and catches the
    front sill's inner face at full travel — an interior tab would ram the ball
    that gravity wedges against the front wall and jam the gate."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_body(root, float(c.plate_mass), com=(0.0, 0.0, 0.0),
                lin_damp=0.02, ang_damp=0.1)
    mat = _friction_material(stage, f"{prim_path}/mat",
                             c.slick_mu, max(c.slick_mu - 0.02, 0.02))
    ck = _make_collide(c.contact_offset, mat)
    _add_box(stage, f"{prim_path}/slab", center=(0.0, 0.0, 0.0),
             size=(c.plate_l, c.plate_w, c.plate_t), color=c.plate_color, collide=ck)
    _add_box(stage, f"{prim_path}/tab",
             center=(-c.plate_l / 2 + c.tab_lx / 2, 0.0, -(c.plate_t / 2 + c.tab_h / 2)),
             size=(c.tab_lx, c.tab_w, c.tab_h), color=c.plate_color, collide=ck)
    return root


def _spawn_bin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the catch bin: KINEMATIC square tray (floor + 4 walls), origin at the
    footprint centre, z = 0 at the ground plane."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_body(root, 5.0, kinematic=True)
    mat = _friction_material(stage, f"{prim_path}/mat", 0.8, 0.7)
    cs = _make_collide(0.002, mat)
    hi, t, wh = c.bin_in / 2, c.bin_t, c.bin_wall_h
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, c.bin_floor_t / 2),
             size=(c.bin_in + 2 * t, c.bin_in + 2 * t, c.bin_floor_t),
             color=c.bin_color, collide=cs)
    for ax in ("x", "y"):
        for s in (-1.0, 1.0):
            cx = s * (hi + t / 2) if ax == "x" else 0.0
            cy = s * (hi + t / 2) if ax == "y" else 0.0
            sx = t if ax == "x" else c.bin_in + 2 * t
            sy = t if ax == "y" else c.bin_in
            _add_box(stage, f"{prim_path}/wall_{ax}{'p' if s > 0 else 'n'}",
                     center=(cx, cy, c.bin_floor_t + wh / 2),
                     size=(sx, sy, wh), color=c.bin_color, collide=cs)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "hopper" not in _SPAWNER_CACHE:

        @configclass
        class HopperSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_hopper)
            cav_l: float = 0.100
            cav_w: float = 0.080
            wall_t: float = 0.008
            rail_h: float = 0.010
            rail_w: float = 0.008
            keeper_w: float = 0.006
            keeper_gap: float = 0.0012
            slot_gap: float = 0.0015
            plate_t: float = 0.006
            roof_z: float = 0.075
            hopper_mass: float = 0.30
            slick_mu: float = 0.11
            contact_offset: float = 0.001
            body_color: tuple = (0.55, 0.38, 0.20)
            trim_color: tuple = (0.20, 0.20, 0.22)

        @configclass
        class PlateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_plate)
            plate_l: float = 0.107
            plate_w: float = 0.0776
            plate_t: float = 0.006
            tab_lx: float = 0.006
            tab_w: float = 0.050
            tab_h: float = 0.008
            plate_mass: float = 0.035
            slick_mu: float = 0.11
            contact_offset: float = 0.001
            plate_color: tuple = (0.75, 0.76, 0.78)

        @configclass
        class BinSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bin)
            bin_in: float = 0.115
            bin_t: float = 0.006
            bin_wall_h: float = 0.035
            bin_floor_t: float = 0.008
            bin_color: tuple = (0.12, 0.30, 0.85)

        _SPAWNER_CACHE["hopper"] = HopperSpawnerCfg
        _SPAWNER_CACHE["plate"] = PlateSpawnerCfg
        _SPAWNER_CACHE["bin"] = BinSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg ---------------------------------------------------------------------------------
@dataclass
class GateHopperSceneCfg(BaseCfg):
    """Config for `GateHopperScene` (see module docstring for the honesty asserts)."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    settle_speed: float = tunable(0.05)    # max |lin vel| (ball AND hopper) when judging (m/s)
    upright_max_deg: float = tunable(10.0)  # hopper local +z within this of world up when parked
    base_z_tol: float = tunable(0.020)     # hopper base height above the ground when parked (m)
    clear_min: float = tunable(0.18)       # min hopper-centre..bin-centre distance when parked (m)
    ball_in_margin: float = tunable(0.010)  # bin-frame |xy| slack: inside = |xy| < bin_in/2 - this
    lift_z: float = tunable(0.06)          # "hopper aloft" once its base is above this (m)
    gate_open_ext: float = tunable(0.050)  # "gate open" once the plate extends past this (m)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    hopper_jitter: float = tunable(0.05)   # hopper spawn xy jitter (+/- m)
    hopper_yaw_deg: float = tunable(180.0)  # hopper spawn free yaw (+/- deg)
    bin_jitter: float = tunable(0.05)      # bin xy jitter (+/- m)
    bin_yaw_deg: float = tunable(180.0)    # bin free yaw (+/- deg)
    ball_x_rng: tuple = tunable((-0.016, 0.012))  # ball spawn range in the cavity (hopper frame)
    ball_y_rng: tuple = tunable((-0.010, 0.010))

    # --- info: layout (world nominal) ------------------------------------------------------------
    hopper_pos: tuple = info((0.0, 0.0))
    bin_pos: tuple = info((0.55, 0.0))
    # --- info: hopper geometry (local frame: origin at footprint centre, z=0 at the base) --------
    cav_l: float = info(0.100)             # cavity length (x, the slide direction)
    cav_w: float = info(0.080)             # cavity width (y)
    wall_t: float = info(0.008)
    rail_h: float = info(0.010)            # rail top = the plate's riding plane
    rail_w: float = info(0.008)            # rails protrude this far inward at each side
    keeper_w: float = info(0.006)
    keeper_gap: float = info(0.0012)       # vertical slop above the plate under the keepers
    slot_gap: float = info(0.0015)         # vertical slop above the plate in the front slot
    roof_z: float = info(0.075)            # interior roof height (walls run 0..roof_z)
    hopper_mass: float = info(0.30)
    # --- info: gate plate ------------------------------------------------------------------------
    plate_l: float = info(0.107)
    plate_w: float = info(0.0776)
    plate_t: float = info(0.006)
    tab_lx: float = info(0.006)
    tab_w: float = info(0.050)          # tab spans this in y — must clear the rails
    tab_h: float = info(0.008)          # tab hangs this far BELOW the plate (in the opening)
    plate_mass: float = info(0.035)
    slick_mu: float = info(0.11)           # rails/sill/keepers AND plate: pair-averaged slickness
    # --- info: ball / bin ------------------------------------------------------------------------
    ball_r: float = info(0.024)
    ball_mass: float = info(0.030)
    bin_in: float = info(0.115)            # bin inner width (square)
    bin_t: float = info(0.006)
    bin_wall_h: float = info(0.035)
    bin_floor_t: float = info(0.008)
    contact_offset: float = info(0.001)
    body_color: tuple = info((0.55, 0.38, 0.20))
    trim_color: tuple = info((0.20, 0.20, 0.22))
    plate_color: tuple = info((0.75, 0.76, 0.78))
    ball_color: tuple = info((0.95, 0.45, 0.05))
    bin_color: tuple = info((0.12, 0.30, 0.85))
    # rubric weights (0.15 + 0.25 + 0.30 = 0.70 = the non-success cap)
    w_lift: float = info(0.15)
    w_gate: float = info(0.25)
    w_drop: float = info(0.30)

    # Derived (filled in __post_init__).
    plate_closed_x: float = info(0.0)      # plate-centre local x when the gate is closed (flush)
    travel_max: float = info(0.0)          # stop-tab travel
    bin_top_z: float = info(0.0)

    def __post_init__(self) -> None:
        hl = self.cav_l / 2
        front_out = hl + self.wall_t
        self.plate_closed_x = front_out - self.plate_l / 2
        plate_rear = front_out - self.plate_l
        self.travel_max = hl - (plate_rear + self.tab_lx)
        self.bin_top_z = self.bin_floor_t + self.bin_wall_h
        # Honesty-by-construction asserts (the geometric claims the task rests on).
        opening_w = self.cav_w - 2 * self.rail_w
        assert opening_w > 2 * self.ball_r + 0.012, \
            "the ball must fit through the floor opening with slack"
        assert self.rail_h + self.plate_t + self.slot_gap < 2 * self.ball_r, \
            "the ball must NOT fit through the plate slot"
        drop_travel = (hl - self.ball_r) - plate_rear
        assert self.travel_max > drop_travel + 0.010, \
            "the stop-tab travel must vacate the ball's footprint with margin"
        assert self.tab_w / 2 < self.cav_w / 2 - self.rail_w - 0.004, \
            "the under-plate tab must ride inside the floor opening, clear of the rails"
        assert self.tab_h < self.rail_h - 0.001, \
            "the under-plate tab must clear the ground while the plate rides the rails"
        assert self.gate_open_ext < self.travel_max - 0.02, \
            "the gate-open latch must be reachable inside the tab travel"
        bin_hd = math.hypot(self.bin_in / 2 + self.bin_t, self.bin_in / 2 + self.bin_t)
        hop_hd = math.hypot(hl + self.wall_t, self.cav_w / 2 + self.wall_t)
        assert self.clear_min > bin_hd + hop_hd + 0.004, \
            "clear_min must exceed the hopper/bin no-touch distance"
        assert self.bin_in / 2 - self.ball_in_margin > self.ball_r + 0.010, \
            "a centred ball must satisfy the in-bin xy clause with margin"
        d = math.dist(self.hopper_pos, self.bin_pos)
        assert d - self.hopper_jitter - self.bin_jitter > self.clear_min + 0.05, \
            "the hopper must spawn parked far beyond clear_min (null scores ~0)"
        x0, x1 = self.ball_x_rng
        y0, y1 = self.ball_y_rng
        assert -hl + self.ball_r + 0.005 < x0 and x1 < hl - self.ball_r - 0.005 \
            and -self.cav_w / 2 + self.ball_r + 0.004 < y0 \
            and y1 < self.cav_w / 2 - self.ball_r - 0.004, \
            "the ball spawn band must keep clearance from the cavity walls"


# ----- small quaternion helpers (wxyz, torch, batched) -------------------------------------------
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
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device)


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = torch.cat([torch.zeros_like(q[:, :1]), v], dim=-1)
    return _qmul(_qmul(q, qv), _qinv(q))[:, 1:]


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -------------------------------------------------------------------------------------
@SCENES.register("gate_hopper")
class GateHopperScene(BaseScene):
    cfg: GateHopperSceneCfg

    def __init__(self, cfg: GateHopperSceneCfg | None = None) -> None:
        super().__init__(cfg or GateHopperSceneCfg())

    # ----- assets --------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        hx, hy = c.hopper_pos
        bx, by = c.bin_pos
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
            "hopper": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Hopper",
                spawn=cls["hopper"](
                    cav_l=c.cav_l, cav_w=c.cav_w, wall_t=c.wall_t, rail_h=c.rail_h,
                    rail_w=c.rail_w, keeper_w=c.keeper_w, keeper_gap=c.keeper_gap,
                    slot_gap=c.slot_gap, plate_t=c.plate_t, roof_z=c.roof_z,
                    hopper_mass=c.hopper_mass, slick_mu=c.slick_mu,
                    contact_offset=c.contact_offset, body_color=c.body_color,
                    trim_color=c.trim_color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(hx, hy, 0.001)),
            ),
            "plate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/GatePlate",
                spawn=cls["plate"](
                    plate_l=c.plate_l, plate_w=c.plate_w, plate_t=c.plate_t,
                    tab_lx=c.tab_lx, tab_w=c.tab_w, tab_h=c.tab_h,
                    plate_mass=c.plate_mass, slick_mu=c.slick_mu,
                    contact_offset=c.contact_offset, plate_color=c.plate_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(hx + c.plate_closed_x, hy, c.rail_h + c.plate_t / 2 + 0.001)),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.ball_color),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.15),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.5, dynamic_friction=0.45, restitution=0.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(hx, hy, c.rail_h + c.plate_t + c.ball_r + 0.001)),
            ),
            "bin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/CatchBin",
                spawn=cls["bin"](bin_in=c.bin_in, bin_t=c.bin_t, bin_wall_h=c.bin_wall_h,
                                 bin_floor_t=c.bin_floor_t, bin_color=c.bin_color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(bx, by, 0.0)),
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

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.hopper: RigidObject = env.iscene["hopper"]
        self.plate: RigidObject = env.iscene["plate"]
        self.ball: RigidObject = env.iscene["ball"]
        self.bin: RigidObject = env.iscene["bin"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches (partial credit survives transients; success is judged live)
        self._lift = torch.zeros(n, dtype=torch.bool, device=dev)
        self._gate = torch.zeros(n, dtype=torch.bool, device=dev)
        self._drop = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: hopper at its slot (xy jitter + full yaw) with the gate
        closed (flush) and the ball at a random spot on the plate inside the cavity
        (poses composed with the hopper frame); bin at its slot (xy jitter + full
        yaw, kinematic); clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.hopper_yaw_deg)
        qh = _qz(yaw)
        hp = torch.zeros(m, 3, device=dev)
        hp[:, 0] = c.hopper_pos[0]
        hp[:, 1] = c.hopper_pos[1]
        hp[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.hopper_jitter
        hp[:, 2] = 0.001
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = hp + origin
        st[:, 3:7] = qh
        self.hopper.write_root_state_to_sim(st, env_ids)

        # plate: closed (flush), riding on the rails, composed with the hopper frame
        pl = torch.zeros(m, 3, device=dev)
        pl[:, 0] = c.plate_closed_x
        pl[:, 2] = c.rail_h + c.plate_t / 2 + 0.0005
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = hp + _qapply(qh, pl) + origin
        st[:, 3:7] = qh
        self.plate.write_root_state_to_sim(st, env_ids)

        # ball: random spot on the plate inside the cavity
        bl = torch.zeros(m, 3, device=dev)
        x0, x1 = c.ball_x_rng
        y0, y1 = c.ball_y_rng
        bl[:, 0] = x0 + torch.rand(m, device=dev) * (x1 - x0)
        bl[:, 1] = y0 + torch.rand(m, device=dev) * (y1 - y0)
        bl[:, 2] = c.rail_h + c.plate_t + c.ball_r + 0.001
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = hp + _qapply(qh, bl) + origin
        st[:, 3] = 1.0
        self.ball.write_root_state_to_sim(st, env_ids)

        # bin: kinematic, jittered slot + free yaw
        byaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.bin_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.bin_pos[0]
        st[:, 1] = c.bin_pos[1]
        st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.bin_jitter
        st[:, 0:3] += origin
        st[:, 3:7] = _qz(byaw)
        self.bin.write_root_state_to_sim(st, env_ids)

        self._lift[env_ids] = False
        self._gate[env_ids] = False
        self._drop[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "hopper": self.hopper.data.root_state_w[env_ids].clone(),
            "plate": self.plate.data.root_state_w[env_ids].clone(),
            "ball": self.ball.data.root_state_w[env_ids].clone(),
            "bin": self.bin.data.root_state_w[env_ids].clone(),
            "lift": self._lift[env_ids].clone(),
            "gate": self._gate[env_ids].clone(),
            "drop": self._drop[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.hopper.write_root_state_to_sim(state["hopper"], env_ids)
        self.plate.write_root_state_to_sim(state["plate"], env_ids)
        self.ball.write_root_state_to_sim(state["ball"], env_ids)
        self.bin.write_root_state_to_sim(state["bin"], env_ids)
        self._lift[env_ids] = state["lift"]
        self._gate[env_ids] = state["gate"]
        self._drop[env_ids] = state["drop"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A small wooden HOPPER (a closed box, {(c.cav_l + 2 * c.wall_t) * 100:.1f} x "
            f"{(c.cav_w + 2 * c.wall_t) * 100:.1f} cm footprint, roof at "
            f"{c.roof_z * 100:.1f} cm, with a dark HANDLE BAR across its roof) stands on "
            f"the ground; its position and heading are randomized. Sealed inside it is an "
            f"ORANGE BALL ({c.ball_r * 200:.1f} cm diameter): the box has four walls and a "
            f"full roof, and its floor is a sliding steel-gray GATE PLATE riding in slick "
            f"internal rails. The gate has NO handle — its only exposed face sits FLUSH "
            f"with the hopper's front wall (the front face showing a thin gray sliver at "
            f"the bottom is the hopper's NOSE) — so it cannot be gripped or pulled open. "
            f"It responds only to gravity: lift the whole hopper and pitch it NOSE-DOWN "
            f"by roughly 15-30 degrees and the plate slides out through its slot until an "
            f"internal stop catches it (about {c.travel_max * 100:.0f} cm out), opening "
            f"the floor; the ball then falls out of the bottom. While the hopper stands "
            f"on the ground or is held level, the gate stays shut and the ball is "
            f"trapped. Across the workspace sits a fixed BLUE square catch bin "
            f"({c.bin_in * 100:.1f} cm inner width, {c.bin_top_z * 100:.1f} cm rim), "
            f"position and heading randomized.\n"
            f"Goal: deliver the orange ball INTO the blue bin, then park the hopper. "
            f"Lift the hopper by its handle, hold it a little above the bin so the "
            f"bottom of the box is over the bin's mouth, and tilt it nose-down until the "
            f"gate slides open and the ball drops inside the bin (below the rim). Then "
            f"set the hopper back down UPRIGHT on open ground, well clear of the bin — "
            f"its centre at least {c.clear_min * 100:.0f} cm from the bin's centre (a "
            f"few cm of daylight between them) — and let everything come to rest. "
            f"Tilting the hopper while it still stands on the ground just dumps the "
            f"ball onto the ground, which counts for nothing: the ball must end up "
            f"resting inside the bin, the hopper parked upright and clear, everything "
            f"settled."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lift the wooden hopper by its roof handle, hold it just above the blue "
            "bin, and tilt it nose-down so its flush floor gate slides open and the "
            "orange ball inside drops into the bin. Then set the hopper down upright "
            "on the ground, clear of the bin, and let everything come to rest."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _hopper_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.hopper.data.root_pos_w, self.hopper.data.root_quat_w

    def gate_ext(self) -> torch.Tensor:
        """(N,) gate extension (m): plate-centre travel past its closed pose, in the
        hopper's frame."""
        hp, hq = self._hopper_frame()
        loc = _qapply(_qinv(hq), self.plate.data.root_pos_w - hp)
        return loc[:, 0] - self.cfg.plate_closed_x

    def ball_local(self) -> torch.Tensor:
        """(N,3) ball centre in the hopper's frame."""
        hp, hq = self._hopper_frame()
        return _qapply(_qinv(hq), self.ball.data.root_pos_w - hp)

    def ball_in_hopper(self) -> torch.Tensor:
        """(N,) bool: ball centre inside the hopper cavity (hopper frame)."""
        c = self.cfg
        bl = self.ball_local()
        return (bl[:, 0].abs() < c.cav_l / 2 + 0.01) \
            & (bl[:, 1].abs() < c.cav_w / 2 + 0.01) \
            & (bl[:, 2] > 0.0) & (bl[:, 2] < c.roof_z + 0.01)

    def ball_in_bin(self) -> torch.Tensor:
        """(N,) bool: ball centre inside the bin's inner box, below the rim (bin
        frame — a ball perched on the rim or leaning outside fails the xy bound)."""
        c = self.cfg
        bq = self.bin.data.root_quat_w
        bl = _qapply(_qinv(bq), self.ball.data.root_pos_w - self.bin.data.root_pos_w)
        half = c.bin_in / 2 - c.ball_in_margin
        return (bl[:, 0].abs() < half) & (bl[:, 1].abs() < half) \
            & (bl[:, 2] > 0.0) & (bl[:, 2] < c.bin_top_z + 0.005)

    def hopper_up(self) -> torch.Tensor:
        """(N,) bool: hopper local +z within `upright_max_deg` of world up."""
        _hp, hq = self._hopper_frame()
        n = hq.shape[0]
        ez = torch.tensor([0.0, 0.0, 1.0], device=hq.device).expand(n, 3)
        up = _qapply(hq, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.upright_max_deg))

    def hopper_parked(self) -> torch.Tensor:
        """(N,) bool: hopper upright ON the ground, its centre at least `clear_min`
        from the bin's centre."""
        c = self.cfg
        hp, _hq = self._hopper_frame()
        base_z = hp[:, 2] - self.env_origins[:, 2]
        d = (hp[:, :2] - self.bin.data.root_pos_w[:, :2]).norm(dim=-1)
        return self.hopper_up() & (base_z.abs() < c.base_z_tol) & (d > c.clear_min)

    def settled(self) -> torch.Tensor:
        """(N,) bool: ball AND hopper |lin vel| below the settle gate."""
        c = self.cfg
        return (self.ball.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.hopper.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)

    def _update_latches(self) -> None:
        c = self.cfg
        hp, _hq = self._hopper_frame()
        aloft = (hp[:, 2] - self.env_origins[:, 2]) > c.lift_z
        self._lift |= aloft & self.ball_in_hopper()
        self._gate |= aloft & (self.gate_ext() > c.gate_open_ext)
        self._drop |= self.ball_in_bin()

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the ball rests inside the bin AND the hopper is parked upright
        on the ground clear of the bin, everything settled and finite. All clauses
        are live physical outcomes (no latch can substitute for them)."""
        self._update_latches()
        fin = torch.isfinite(self.ball.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.hopper.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.plate.data.root_pos_w).all(dim=-1)
        return self.ball_in_bin() & self.hopper_parked() & self.settled() & fin

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15*lift + 0.25*gate + 0.30*drop (all latched; ~0
        for doing nothing), capped at 0.70 — and exactly 1.0 iff success()."""
        c = self.cfg
        self._update_latches()
        base = (c.w_lift * self._lift.float() + c.w_gate * self._gate.float()
                + c.w_drop * self._drop.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="gate_hopper", robot="null"))

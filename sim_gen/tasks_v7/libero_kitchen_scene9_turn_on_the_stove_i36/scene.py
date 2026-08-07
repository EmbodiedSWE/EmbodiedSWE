"""GapFerryScene — build the bridge, then ferry the ungraspable ball across the chasm.

Derived from libero_90/libero_kitchen_scene9_turn_on_the_stove ("turn on the stove": an
articulated stove USD whose knob the robot rotates past a joint threshold — one
actuation of a BUILT mechanism, judged by a joint readout). Here NOTHING is articulated
and there is no joint to read: the scene is two elevated, walled DOCKS facing each
other across a fixed 140 mm chasm, a loose rail-guided channel PLANK parked on a low
storage rack beside them, and a 90 mm BALL waiting in the start dock. The "mechanism"
the task turns on is one the solver must CONSTRUCT: seat the plank into the rebate
ledges cut into both dock edges so it bridges the chasm flush with the deck tops, then
ROLL the ball across it into the far dock. The ball is deliberately wider than a
parallel jaw opens (90 mm > 80 mm): it cannot be carried, only rolled — so the bridge
is physically load-bearing, not decorative, and pushing the ball anywhere without the
bridge drops it into the chasm, where it is unrecoverable.

Assets are fully procedural:
  - docks: two KINEMATIC compound bodies (solid pedestal, deck slab, a 55 mm deep x
    15 mm drop rebate ledge at the chasm edge, 60 mm perimeter walls open toward the
    chasm). The far dock's deck is BLUE, the start dock's deck GREY; the whole
    assembly's position and heading randomize per episode.
  - plank: a DYNAMIC channel plank (210 x 130 x 15 mm base + two 14 mm guide rails),
    parked rails-up on a two-sleeper rack (25 mm tall — finger clearance underneath).
    Seated in the rebates its top is FLUSH with both decks and the backstops box it
    lengthwise (a plank inside both rebates physically cannot be off by more than
    20 mm, so a "barely hanging on" seating cannot even be constructed).
  - ball: a DYNAMIC orange sphere, 90 mm diameter, 450 g, damped so it settles.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve trajectory):
  0.30 * bridged  — the plank ever seated across the chasm: both ends supported on the
                    rebate ledges, at seat height, level, calm            (latched)
  0.15 * embarked — the ball ever rolling ON the plank over the chasm     (latched)
  0.25 * crossed  — the ball ever inside the far dock's deck volume       (latched)
  1.0 iff success() — the ball resting ON the far dock's deck (inside its walls, at
                    deck+radius height — not on the ledge, not on the plank, not on
                    the ground), settled and finite. Non-success capped at 0.70.
All success clauses are live physical outcomes; latches only preserve credit for
stages genuinely passed through. The null policy latches nothing (plank on the rack
never bridges; a ball in the start dock is in no window).

Honesty geometry (asserted in `__post_init__`):
  - the plank spans the chasm with >= 30 mm overlap per side when centred, and the
    rebate backstops bound its lengthwise error to the seating slack;
  - the rail channel clears the ball; the ball exceeds the 80 mm jaw; the dock walls
    out-reach the ball's centre height (a fast arrival cannot hop out);
  - the success height window excludes a ball resting on the bare ledge, and the
    success x window excludes a ball still on the plank's far end.

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
    out[..., 1:] = -out[..., 1:]
    return out


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4), pure torch."""
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(qv, t, dim=-1)


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def encode_force(mode: int, q_ref: torch.Tensor, q_now: torch.Tensor,
                 f_world: torch.Tensor) -> torch.Tensor:
    """Pre-encode a desired WORLD-frame force for `set_external_force_and_torque`.

    Some pods rotate an applied wrench by the body's rotation since its reference
    orientation (applied = R_now * R_ref^T * arg). mode 0 passes the world force
    through unchanged; mode 1 pre-encodes with R_ref * R_now^T so the applied force
    comes out as the desired world force. Callers PROBE which mode moves the body the
    right way and lock it in (`q_ref` = readback at the reference instant). For a
    ROLLING ball q_now changes every step, so mode-1 encoding must be recomputed with
    a fresh readback each step."""
    if mode == 0:
        return f_world
    return _qapply(_qmul(q_ref, _qinv(q_now)), f_world)


# ----- procedural compound spawners -------------------------------------------------------------
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


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, *, center, size, color, contact_offset: float):
    """One collidable box child: translate + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(box.GetPrim(), contact_offset)
    return box.GetPrim()


def _phys_material(stage, path: str, static: float, dynamic: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _bind_material(prim, mat) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _rigid_dynamic(root, mass: float) -> None:
    """Dynamic rigid-body armor on a compound root: explicit mass, mild damping so
    the body settles promptly, no sleeping while velocities are judged, depenetration
    cap, iterated solver."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.15)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _rigid_kinematic(root, mass: float) -> None:
    """Kinematic rigid-body armor: pose is authored/written, never simulated."""
    from pxr import UsdPhysics

    api = UsdPhysics.RigidBodyAPI.Apply(root)
    api.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))


def _spawn_dock(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one KINEMATIC dock. Root origin: centre of the chasm-facing edge at
    GROUND level, body +x pointing INTO the chasm (the far dock is placed with a
    180 deg yaw so both rebates face each other). Children (one body, no
    self-collision): solid pedestal, deck slab (top = deck_h, coloured), rebate
    ledge (top = deck_h - rebate_drop), two side walls and a back wall rising
    wall_h above the deck; the chasm side stays open."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_kinematic(root, 50.0)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.mu_static, cfg.mu_dynamic)
    c = cfg
    co = c.contact_offset
    dl, dw, hh = c.dock_len, c.dock_w, c.deck_h
    rd, rt = c.rebate_d, c.rebate_drop
    base_h = hh - c.slab_t
    deck_len = dl - rd  # x extent of the raised deck slab (walls run along it)
    kids = [
        _box(stage, f"{prim_path}/pedestal", center=(-dl / 2, 0.0, base_h / 2),
             size=(dl, dw, base_h), color=c.pedestal_color, contact_offset=co),
        _box(stage, f"{prim_path}/deck", center=(-(rd + deck_len / 2), 0.0,
             base_h + c.slab_t / 2),
             size=(deck_len, dw, c.slab_t), color=c.deck_color, contact_offset=co),
        _box(stage, f"{prim_path}/ledge", center=(-rd / 2, 0.0,
             base_h + (c.slab_t - rt) / 2),
             size=(rd, dw, c.slab_t - rt), color=c.ledge_color, contact_offset=co),
        _box(stage, f"{prim_path}/back_wall",
             center=(-dl + c.wall_t / 2, 0.0, hh + c.wall_h / 2),
             size=(c.wall_t, dw, c.wall_h), color=c.wall_color, contact_offset=co),
    ]
    for sgn in (1.0, -1.0):
        s = "p" if sgn > 0 else "n"
        kids.append(_box(stage, f"{prim_path}/wall_y{s}",
                         center=(-(rd + deck_len / 2), sgn * (dw / 2 - c.wall_t / 2),
                                 hh + c.wall_h / 2),
                         size=(deck_len, c.wall_t, c.wall_h), color=c.wall_color,
                         contact_offset=co))
    for k in kids:
        _bind_material(k, mat)
    return root


def _spawn_plank(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC channel plank. Root origin at the centre of the base slab;
    two guide rails run the full length on top (rails-up = channel open upward)."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_dynamic(root, cfg.mass)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.mu_static, cfg.mu_dynamic)
    c = cfg
    co = c.contact_offset
    kids = [
        _box(stage, f"{prim_path}/base", center=(0.0, 0.0, 0.0),
             size=(c.length, c.width, c.thick), color=c.base_color, contact_offset=co),
    ]
    for sgn in (1.0, -1.0):
        s = "p" if sgn > 0 else "n"
        kids.append(_box(stage, f"{prim_path}/rail_y{s}",
                         center=(0.0, sgn * (c.width / 2 - c.rail_t / 2),
                                 c.thick / 2 + c.rail_h / 2),
                         size=(c.length, c.rail_t, c.rail_h), color=c.rail_color,
                         contact_offset=co))
    for k in kids:
        _bind_material(k, mat)
    return root


def _spawn_rack(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC two-sleeper storage rack (25 mm tall — the parked plank
    keeps finger clearance underneath)."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_kinematic(root, 20.0)
    mat = _phys_material(stage, f"{prim_path}/physmat", 0.6, 0.5)
    c = cfg
    for sgn in (1.0, -1.0):
        s = "p" if sgn > 0 else "n"
        k = _box(stage, f"{prim_path}/sleeper_x{s}",
                 center=(sgn * c.spacing / 2, 0.0, c.height / 2),
                 size=(c.sleeper_w, c.sleeper_l, c.height), color=c.color,
                 contact_offset=c.contact_offset)
        _bind_material(k, mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "dock" not in _SPAWNER_CACHE:

        @configclass
        class DockSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_dock)
            dock_len: float = 0.21
            dock_w: float = 0.34
            deck_h: float = 0.12
            slab_t: float = 0.03
            rebate_d: float = 0.055
            rebate_drop: float = 0.015
            wall_t: float = 0.012
            wall_h: float = 0.06
            mu_static: float = 0.60
            mu_dynamic: float = 0.50
            pedestal_color: tuple = (0.45, 0.45, 0.48)
            deck_color: tuple = (0.62, 0.62, 0.64)
            ledge_color: tuple = (0.35, 0.35, 0.38)
            wall_color: tuple = (0.52, 0.52, 0.55)
            contact_offset: float = 0.002

        @configclass
        class PlankSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_plank)
            length: float = 0.21
            width: float = 0.13
            thick: float = 0.015
            rail_t: float = 0.010
            rail_h: float = 0.014
            mass: float = 0.15
            mu_static: float = 0.50
            mu_dynamic: float = 0.45
            base_color: tuple = (0.76, 0.60, 0.35)
            rail_color: tuple = (0.55, 0.40, 0.20)
            contact_offset: float = 0.002

        @configclass
        class RackSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rack)
            spacing: float = 0.12
            sleeper_w: float = 0.03
            sleeper_l: float = 0.20
            height: float = 0.025
            color: tuple = (0.30, 0.28, 0.26)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["dock"] = DockSpawnerCfg
        _SPAWNER_CACHE["plank"] = PlankSpawnerCfg
        _SPAWNER_CACHE["rack"] = RackSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class GapFerrySceneCfg(BaseCfg):
    """Config for `GapFerryScene`. The seating slack is bounded by the rebate
    backstops themselves, the crossing tolerance by the guide rails, and the arrival
    windows by the far dock's walls (see the honesty asserts)."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    seat_z_tol: float = tunable(0.006)       # |plank centre z - seat height| when bridged
    end_overlap_min: float = tunable(0.008)  # each plank end this far over its ledge (m)
    plank_tilt_max_deg: float = tunable(6.0)  # plank +z within this of world-up when bridged
    embark_margin: float = tunable(0.012)    # ball-over-chasm x window inset (m)
    dock_x_lo: float = tunable(-0.160)       # success window, far-dock frame x (back wall side)
    dock_x_hi: float = tunable(-0.060)       # success window, far-dock frame x (past the ledge)
    dock_y_tol: float = tunable(0.115)       # success window, far-dock frame |y|
    ball_z_lo: float = tunable(0.157)        # success window, ball centre height (deck+r-8mm)
    ball_z_hi: float = tunable(0.175)        # success window, ball centre height (deck+r+10mm)
    settle_speed: float = tunable(0.05)      # max |lin vel| (ball AND plank) when judging (m/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    asm_jitter: float = tunable(0.030)       # assembly-centre xy jitter (+/- m)
    asm_yaw_deg: float = tunable(25.0)       # assembly heading (+/- deg)
    rack_side_swap: bool = tunable(True)     # rack spawns left OR right of the docks
    plank_yaw_deg: float = tunable(20.0)     # plank yaw jitter on the rack (+/- deg)
    plank_flip: bool = tunable(True)         # coin: plank parked end-for-end
    plank_x_jitter: float = tunable(0.030)   # plank lengthwise jitter on the rack (+/- m)
    ball_x_range: tuple = tunable((-0.150, -0.105))  # ball spawn, start-dock frame x
    ball_y_range: tuple = tunable((-0.080, 0.080))   # ball spawn, start-dock frame y

    # --- info: layout (world nominal, ground z = 0) ----------------------------------------------
    asm_center: tuple = info((0.40, 0.0))    # chasm centre (assembly frame origin)
    gap: float = info(0.14)                  # chasm width between the two rebate edges
    rack_dy: float = info(0.33)              # rack centre |y| in the assembly frame
    # --- info: dock structure --------------------------------------------------------------------
    dock_len: float = info(0.21)
    dock_w: float = info(0.34)
    deck_h: float = info(0.12)               # deck top height
    slab_t: float = info(0.03)
    rebate_d: float = info(0.055)            # rebate ledge depth (x extent)
    rebate_drop: float = info(0.015)         # ledge sits this far below the deck top
    wall_t: float = info(0.012)
    wall_h: float = info(0.06)               # wall rise above the deck (> ball radius)
    start_deck_color: tuple = info((0.62, 0.62, 0.64))
    far_deck_color: tuple = info((0.15, 0.35, 0.80))
    # --- info: plank -----------------------------------------------------------------------------
    plank_l: float = info(0.21)
    plank_w: float = info(0.13)
    plank_t: float = info(0.015)             # base thickness == rebate_drop -> flush when seated
    rail_t: float = info(0.010)
    rail_h: float = info(0.014)
    plank_mass: float = info(0.15)
    # --- info: ball ------------------------------------------------------------------------------
    ball_r: float = info(0.045)              # 90 mm dia > the 80 mm parallel jaw: roll, not carry
    ball_mass: float = info(0.45)
    ball_color: tuple = info((0.90, 0.45, 0.10))
    # --- info: rack ------------------------------------------------------------------------------
    rack_h: float = info(0.025)
    rack_spacing: float = info(0.12)
    jaw_max: float = info(0.080)             # Franka parallel-jaw max opening (embodiment bound)
    contact_offset: float = info(0.002)
    # rubric weights (0.30 + 0.15 + 0.25 = 0.70 = the non-success cap)
    w_bridge: float = info(0.30)
    w_embark: float = info(0.15)
    w_cross: float = info(0.25)

    def __post_init__(self) -> None:
        # the plank spans the chasm with real overlap, and the backstops box it in
        assert self.plank_l >= self.gap + 0.06, "plank must span with >=30 mm/side centred"
        slack = (self.gap + 2 * self.rebate_d) - self.plank_l
        assert 0.02 <= slack <= 0.06, "seating slack must be small but constructible"
        # a seated plank is flush with the decks (the ball rolls on/off with no step)
        assert abs(self.plank_t - self.rebate_drop) < 1e-6, "seated plank must be flush"
        # the rail channel clears the ball
        channel = self.plank_w - 2 * self.rail_t
        assert channel >= 2 * self.ball_r + 0.015, "rail channel must clear the ball"
        # the ball is ungraspable by the parallel jaw (the transport-forcing bound)
        assert 2 * self.ball_r > self.jaw_max + 0.008, "ball must exceed the jaw opening"
        # the walls out-reach the ball centre (an arriving ball cannot hop out)
        assert self.wall_h > self.ball_r + 0.010, "walls must retain the ball"
        # the success height window excludes a ball resting on the bare ledge
        ledge_rest = self.deck_h - self.rebate_drop + self.ball_r
        assert ledge_rest < self.ball_z_lo - 0.004, "ledge rest must fail the z window"
        # the success x window excludes a ball still over the ledge / plank end
        assert self.dock_x_hi <= -self.rebate_d, "success x window must clear the ledge"
        # ball spawn zone sits fully on the start deck, inside the walls
        bx, by = self.ball_x_range, self.ball_y_range
        assert bx[0] > -(self.dock_len - self.wall_t) + self.ball_r, "spawn hits back wall"
        assert bx[1] < -self.rebate_d - self.ball_r, "spawn zone must stay off the ledge"
        assert abs(by[0]) < self.dock_w / 2 - self.wall_t - self.ball_r, "spawn hits wall"
        # the rack stays clear of the dock footprint under all jitters
        assert self.rack_dy - self.asm_jitter > self.dock_w / 2 + self.plank_w / 2 + 0.04, \
            "rack could spawn against a dock"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("gap_ferry")
class GapFerryScene(BaseScene):
    cfg: GapFerrySceneCfg

    def __init__(self, cfg: GapFerrySceneCfg | None = None) -> None:
        super().__init__(cfg or GapFerrySceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        dock_kw = dict(
            dock_len=c.dock_len, dock_w=c.dock_w, deck_h=c.deck_h, slab_t=c.slab_t,
            rebate_d=c.rebate_d, rebate_drop=c.rebate_drop, wall_t=c.wall_t,
            wall_h=c.wall_h, contact_offset=c.contact_offset)
        ax, ay = c.asm_center
        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "start_dock": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/StartDock",
                spawn=cls["dock"](deck_color=c.start_deck_color, **dock_kw),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(ax - c.gap / 2, ay, 0.0)),
            ),
            "far_dock": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/FarDock",
                spawn=cls["dock"](deck_color=c.far_deck_color, **dock_kw),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(ax + c.gap / 2, ay, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
            ),
            "rack": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rack",
                spawn=cls["rack"](spacing=c.rack_spacing, height=c.rack_h,
                                  contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(ax, ay + c.rack_dy, 0.0)),
            ),
            "plank": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plank",
                spawn=cls["plank"](length=c.plank_l, width=c.plank_w, thick=c.plank_t,
                                   rail_t=c.rail_t, rail_h=c.rail_h, mass=c.plank_mass,
                                   contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(ax, ay + c.rack_dy, c.rack_h + c.plank_t / 2 + 0.003)),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.ball_color),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.25, angular_damping=0.25,
                        sleep_threshold=0.0, stabilization_threshold=0.0,
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=1),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.7, dynamic_friction=0.6, restitution=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(ax - c.gap / 2 - 0.12, ay, c.deck_h + c.ball_r + 0.003)),
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
        self.start_dock: RigidObject = env.iscene["start_dock"]
        self.far_dock: RigidObject = env.iscene["far_dock"]
        self.rack: RigidObject = env.iscene["rack"]
        self.plank: RigidObject = env.iscene["plank"]
        self.ball: RigidObject = env.iscene["ball"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # rack_side[e] = +1: rack at +y of the assembly / -1: at -y
        self.rack_side = torch.ones(n, dtype=torch.float, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._bridged = torch.zeros(n, dtype=torch.bool, device=dev)
        self._embarked = torch.zeros(n, dtype=torch.bool, device=dev)
        self._crossed = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: the whole dock assembly takes a new centre + heading, the
        rack flips to a random side, the plank parks on it with yaw jitter + an
        end-for-end coin, the ball spawns in a window of the start deck. Latches
        cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def rnd(k: float) -> torch.Tensor:
            return (torch.rand(m, device=dev) * 2 - 1) * k

        theta = rnd(math.radians(c.asm_yaw_deg))
        ax = c.asm_center[0] + rnd(c.asm_jitter)
        ay = c.asm_center[1] + rnd(c.asm_jitter)
        cth, sth = torch.cos(theta), torch.sin(theta)

        def asm_xy(lx: torch.Tensor, ly: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Assembly-frame xy -> world xy."""
            return ax + lx * cth - ly * sth, ay + lx * sth + ly * cth

        def write(body, x, y, z, q) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0], st[:, 1], st[:, 2] = x, y, z
            st[:, 3:7] = q
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        zeros = torch.zeros(m, device=dev)
        # docks: rebate edges facing each other across the chasm
        sx, sy = asm_xy(torch.full((m,), -c.gap / 2, device=dev), zeros)
        write(self.start_dock, sx, sy, zeros, _qz(theta))
        fx, fy = asm_xy(torch.full((m,), c.gap / 2, device=dev), zeros)
        write(self.far_dock, fx, fy, zeros, _qz(theta + math.pi))

        # rack + plank: a random side of the docks
        if c.rack_side_swap:
            side = torch.where(torch.rand(m, device=dev) < 0.5,
                               torch.ones(m, device=dev), -torch.ones(m, device=dev))
        else:
            side = torch.ones(m, device=dev)
        self.rack_side[env_ids] = side
        rx, ry = asm_xy(zeros, side * c.rack_dy)
        write(self.rack, rx, ry, zeros, _qz(theta))
        pdx = rnd(c.plank_x_jitter)
        pdy = rnd(0.015)
        px, py = asm_xy(pdx, side * c.rack_dy + pdy)
        pyaw = theta + rnd(math.radians(c.plank_yaw_deg))
        if c.plank_flip:
            flip = (torch.rand(m, device=dev) < 0.5).float() * math.pi
            pyaw = pyaw + flip
        write(self.plank, px, py,
              torch.full((m,), c.rack_h + c.plank_t / 2 + 0.003, device=dev), _qz(pyaw))

        # ball: a window of the start deck (start-dock frame; that frame's +x points
        # into the chasm and its origin is the chasm edge)
        blx = c.ball_x_range[0] + torch.rand(m, device=dev) \
            * (c.ball_x_range[1] - c.ball_x_range[0])
        bly = c.ball_y_range[0] + torch.rand(m, device=dev) \
            * (c.ball_y_range[1] - c.ball_y_range[0])
        bx, by = asm_xy(-c.gap / 2 + blx, bly)
        write(self.ball, bx, by,
              torch.full((m,), c.deck_h + c.ball_r + 0.003, device=dev),
              _qz(torch.zeros(m, device=dev)))

        self._bridged[env_ids] = False
        self._embarked[env_ids] = False
        self._crossed[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "start_dock": self.start_dock.data.root_state_w[env_ids].clone(),
            "far_dock": self.far_dock.data.root_state_w[env_ids].clone(),
            "rack": self.rack.data.root_state_w[env_ids].clone(),
            "plank": self.plank.data.root_state_w[env_ids].clone(),
            "ball": self.ball.data.root_state_w[env_ids].clone(),
            "rack_side": self.rack_side[env_ids].clone(),
            "bridged": self._bridged[env_ids].clone(),
            "embarked": self._embarked[env_ids].clone(),
            "crossed": self._crossed[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.start_dock.write_root_state_to_sim(state["start_dock"], env_ids)
        self.far_dock.write_root_state_to_sim(state["far_dock"], env_ids)
        self.rack.write_root_state_to_sim(state["rack"], env_ids)
        self.plank.write_root_state_to_sim(state["plank"], env_ids)
        self.ball.write_root_state_to_sim(state["ball"], env_ids)
        self.rack_side[env_ids] = state["rack_side"]
        self._bridged[env_ids] = state["bridged"]
        self._embarked[env_ids] = state["embarked"]
        self._crossed[env_ids] = state["crossed"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"Two raised platforms (DOCKS, deck tops {c.deck_h * 1000:.0f} mm above the "
            f"floor, {c.wall_h * 1000:.0f} mm walls on three sides) face each other "
            f"across an open {c.gap * 1000:.0f} mm chasm; the pair's position and "
            f"heading vary per episode. The START dock has a GREY deck and holds a "
            f"large ORANGE BALL ({2 * c.ball_r * 1000:.0f} mm across — wider than a "
            f"parallel gripper opens, so it cannot be picked up, only rolled or "
            f"pushed). The TARGET dock, across the chasm, has a BLUE deck and is "
            f"empty. Each dock's chasm edge carries a recessed LEDGE (a shelf "
            f"{c.rebate_d * 1000:.0f} mm deep, {c.rebate_drop * 1000:.0f} mm below "
            f"the deck top). On a low rack beside the docks lies a tan CHANNEL PLANK "
            f"({c.plank_l * 1000:.0f} x {c.plank_w * 1000:.0f} x "
            f"{c.plank_t * 1000:.0f} mm, with two raised guide rails along its "
            f"edges); the rack's side of the docks, and the plank's exact pose on it, "
            f"vary per episode.\n"
            f"Goal: get the orange ball resting on the BLUE deck of the target dock. "
            f"The only way across is to BRIDGE the chasm first: lay the plank, rails "
            f"up, across the gap so each end seats on a ledge — seated, its top lies "
            f"flush with both decks and the rails form a guide channel — then roll "
            f"the ball across the bridge into the target dock. A ball pushed toward "
            f"the chasm without a seated bridge falls in and cannot be recovered. "
            f"Success requires the ball fully ON the blue deck (inside its walls, at "
            f"deck height — not on the ledge, not still on the plank, not on the "
            f"floor) and at rest; the plank may stay in place. The bridge must come "
            f"before the crossing — that order is physics, not decree."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lay the tan channel plank across the chasm so both ends seat on the edge "
            "ledges, then roll the big orange ball over the bridge until it rests on "
            "the blue deck of the far dock. Do not let the ball fall into the chasm."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _dock_local(self, dock, pos_w: torch.Tensor) -> torch.Tensor:
        """World points (N,3) -> the dock's body frame (origin: chasm-edge centre at
        ground; +x into the chasm)."""
        return _qapply(_qinv(dock.data.root_quat_w), pos_w - dock.data.root_pos_w)

    def _plank_axis(self) -> torch.Tensor:
        """(N,3) world direction of the plank's long (+x body) axis."""
        n = self.env.num_envs
        ex = torch.tensor([1.0, 0.0, 0.0], device=self.env.device).expand(n, 3)
        return _qapply(self.plank.data.root_quat_w, ex)

    def _plank_ends(self) -> tuple[torch.Tensor, torch.Tensor]:
        """((N,3), (N,3)) world positions of the plank's two end centres."""
        axis = self._plank_axis()
        ctr = self.plank.data.root_pos_w
        h = self.cfg.plank_l / 2
        return ctr + axis * h, ctr - axis * h

    def plank_up_z(self) -> torch.Tensor:
        """(N,) world-z component of the plank's body +z axis (+1 = rails up, level)."""
        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        return _qapply(self.plank.data.root_quat_w, ez)[:, 2].clamp(-1.0, 1.0)

    def bridged(self) -> torch.Tensor:
        """(N,) bool: the plank seated across the chasm — each end at least
        `end_overlap_min` over its rebate ledge, plank centre at seat height, level
        (rails up), and calm. Every clause is a live physical outcome."""
        c = self.cfg
        e1, e2 = self._plank_ends()
        # per dock: the nearer end must lie over the ledge (frame x in [-rebate_d, 0])
        s1 = self._dock_local(self.start_dock, e1)[:, 0]
        s2 = self._dock_local(self.start_dock, e2)[:, 0]
        f1 = self._dock_local(self.far_dock, e1)[:, 0]
        f2 = self._dock_local(self.far_dock, e2)[:, 0]
        start_ok = (torch.minimum(s1, s2) <= -c.end_overlap_min) \
            & (torch.minimum(s1, s2) >= -(c.rebate_d + 0.02))
        far_ok = (torch.minimum(f1, f2) <= -c.end_overlap_min) \
            & (torch.minimum(f1, f2) >= -(c.rebate_d + 0.02))
        seat_z = c.deck_h - c.rebate_drop + c.plank_t / 2
        z_rel = self.plank.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        seated = (z_rel - seat_z).abs() < c.seat_z_tol
        level = self.plank_up_z() >= math.cos(math.radians(c.plank_tilt_max_deg))
        calm = self.plank.data.root_lin_vel_w.norm(dim=-1) < 0.08
        return start_ok & far_ok & seated & level & calm

    def _plank_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        return _qapply(_qinv(self.plank.data.root_quat_w),
                       pos_w - self.plank.data.root_pos_w)

    def on_plank(self) -> torch.Tensor:
        """(N,) bool: the ball riding the plank channel (inside the rails in the
        plank frame, centre at rolling height)."""
        c = self.cfg
        loc = self._plank_local(self.ball.data.root_pos_w)
        z_rel = self.ball.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        return (loc[:, 0].abs() < c.plank_l / 2 + 0.010) \
            & (loc[:, 1].abs() < c.plank_w / 2 - c.rail_t) \
            & (z_rel > c.deck_h + c.ball_r - 0.012) \
            & (z_rel < c.deck_h + c.ball_r + 0.020)

    def over_chasm(self) -> torch.Tensor:
        """(N,) bool: the ball centre horizontally over the open gap (start-dock
        frame x in the chasm span, inset by `embark_margin`)."""
        c = self.cfg
        x = self._dock_local(self.start_dock, self.ball.data.root_pos_w)[:, 0]
        return (x > c.embark_margin) & (x < c.gap - c.embark_margin)

    def in_far_dock(self) -> torch.Tensor:
        """(N,) bool: ball centre inside the far dock's deck volume — past the ledge,
        inside the walls, at deck+radius height (a ball on the bare ledge, on the
        plank's far end, or on the floor all fail a window)."""
        c = self.cfg
        loc = self._dock_local(self.far_dock, self.ball.data.root_pos_w)
        z_rel = self.ball.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        return (loc[:, 0] > c.dock_x_lo) & (loc[:, 0] < c.dock_x_hi) \
            & (loc[:, 1].abs() < c.dock_y_tol) \
            & (z_rel > c.ball_z_lo) & (z_rel < c.ball_z_hi)

    def ball_in_chasm(self) -> torch.Tensor:
        """(N,) bool: the ball fell to the floor (unrecoverable)."""
        z_rel = self.ball.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        return z_rel < self.cfg.ball_r + 0.02

    def settled(self) -> torch.Tensor:
        """(N,) bool: ball AND plank |lin vel| below `settle_speed`."""
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                         for b in (self.ball, self.plank)], dim=1)
        return (v < self.cfg.settle_speed).all(dim=1)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w for b in (self.ball, self.plank)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        fin = self._finite()
        br = self.bridged() & fin
        self._bridged |= br
        self._embarked |= self.on_plank() & self.over_chasm() & fin
        self._crossed |= self.in_far_dock() & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the ball resting on the far dock's deck — inside the walls,
        past the ledge, at deck height — with ball and plank settled and finite.
        All clauses are live physical outcomes."""
        self._update_latches()
        return self.in_far_dock() & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.30*bridged + 0.15*embarked + 0.25*crossed (all
        latched; ~0 for doing nothing — a parked plank never bridges and a ball in
        the start dock is in no window), capped at 0.70 — and exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_bridge * self._bridged.float() + c.w_embark * self._embarked.float()
                + c.w_cross * self._crossed.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="gap_ferry", robot="null"))

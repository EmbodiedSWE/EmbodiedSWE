"""HookHangBasketScene — hang the basket on the high wall peg by its handle, then put
the red ketchup bottle into the HANGING basket.

Derived from libero_90/living_room_scene1 "pick up the ketchup and put it in the basket"
(grasp the standing ketchup bottle, carry it over a PASSIVE open basket resting on the
surface, release, bbox containment check). Here the receptacle is NOT usable where it
lies: success requires the basket to be SUSPENDED first, and the deposit happens into a
compliant, swinging container.

- A wall STAND carries two horizontal pegs at different heights; WHICH SIDE is the high
  one flips per episode. The basket carries a bail HANDLE (a bar above its opening).
  Success requires the handle bar resting ON a peg with the basket hanging CLEAR OF THE
  FLOOR. Hang depth (bar to basket bottom) is 186 mm: the HIGH peg (~0.36 m) leaves
  ~0.17 m of air below the hanging basket; the LOW peg (~0.13 m) is a DECOY — a basket
  "hung" there stands on the ground and is rejected (smoke #7).
- The seed's entire plan — put the ketchup in the basket where the basket lies — is a
  rejected end state here (smoke #6): containment in a grounded basket earns only the
  small shaping credit, never success.
- The deposit itself is a new interaction: the hanging basket is a pendulum on a point
  contact (bar across peg). Dropping the 0.30 kg bottle into the 0.22 kg basket swings
  and tilts it; success is judged on the SETTLED suspended state (bar still on the peg,
  bottle inside, everything still) — a hard release can knock the basket off the peg
  or eject the bottle, so the impulse must be managed.
- A same-shape brown BBQ-sauce bottle (slot-swapped per episode) must stay OUT
  (smoke #8, #9).

So a solver needs a different plan (prepare the receptacle by suspending it at height —
choosing the peg that gives ground clearance — then load a moving, compliant container)
and different code structure (a hang-then-fill program around a pendulum contact),
not different parameters on grasp-carry-drop into a passive box.

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - stand: KINEMATIC compound — floor base slab + vertical board. Local +x points OUT
    of the board face (toward the workspace after the reset yaw ~ 180 deg).
  - peg_a / peg_b: two KINEMATIC compound bodies (cylinder along local +x, r 8 mm,
    156 mm long, plus a raised tip stub that retains the bar) posed at reset on the
    board face at (y = +/-0.11) with randomized high/low assignment + z jitter.
  - basket: DYNAMIC compound — open box (160 x 120 x 110 mm outer) + two handle struts
    + the handle BAR (12 mm square section) 200 mm above the basket bottom. Origin at
    the basket floor bottom centre => authored mass puts the CoM there, so the basket
    hangs level and pendulum-stable.
  - ketchup / bbq: DYNAMIC squeeze bottles (body cylinder 55 mm dia + cap), standing.

Per-episode randomization (readback-verifiable): stand y offset + yaw, HIGH/LOW side
swap + per-peg z jitter, basket xy jitter + free yaw, Bernoulli slot swap of the two
bottles + per-bottle xy jitter.

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.25 * hung    — basket ever ENGAGED-HUNG (bar on a peg, hang-depth band, clear of
                   the floor, upright) for >= 10 consecutive steps (latched)
  0.10 * in      — ketchup ever contained in the basket, anywhere (the seed's plan
                   tops out here) (latched)
  0.30 * loaded  — ketchup contained WHILE the basket is engaged-hung, >= 10
                   consecutive steps (latched)
  1.0 iff success() — basket engaged-hung AND ketchup contained AND bbq NOT contained
                   AND basket + ketchup still. Non-success cap 0.65.

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
# One rigid body per object, several child colliders, authored with raw pxr APIs; only
# `isaaclab.sim.utils.clone` is borrowed (regex-resolve + per-env replication).

_SPAWNER_CACHE: dict[str, Any] = {}


def _add_box(stage, path: str, *, center, size, color, collide: Callable) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _add_cyl(stage, path: str, *, center, radius, height, axis, color,
             collide: Callable) -> None:
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr(axis)
    r, h = float(radius), float(height)
    if axis == "X":
        ext = [Gf.Vec3f(-h / 2, -r, -r), Gf.Vec3f(h / 2, r, r)]
    else:
        ext = [Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)]
    cyl.CreateExtentAttr(ext)
    UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())


def _make_collide(cfg: Any) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


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


def _dyn_props(root, *, lin_damp: float, ang_damp: float) -> None:
    """Dynamic-body physics armor (custom spawners apply NO cfg schemas, so author
    everything here): depenetration cap, damping, zero sleep (force-driven bodies are
    judged for stillness), iterated solver (vel iters capped at 4 — TGS)."""
    from pxr import PhysxSchema

    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(lin_damp))
    pxrb.CreateAngularDampingAttr(float(ang_damp))
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the wall stand: KINEMATIC base slab + vertical board. Origin at the board
    centreline on the floor; the board FRONT face is at local x = +board_t/2."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    _add_box(stage, f"{prim_path}/base",
             center=(-0.02, 0.0, c.base_t / 2),
             size=(c.base_d, c.base_w, c.base_t), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/board",
             center=(0.0, 0.0, c.base_t + c.board_h / 2),
             size=(c.board_t, c.board_w, c.board_h), color=c.color, collide=collide)
    return root


def _spawn_peg(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one peg: KINEMATIC cylinder along local +x (root at the board face) with
    a raised tip STUB that retains the handle bar."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    _add_cyl(stage, f"{prim_path}/rod",
             center=(c.peg_len / 2, 0.0, 0.0), radius=c.peg_r, height=c.peg_len,
             axis="X", color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/stub",
             center=(c.peg_len + 0.006, 0.0, 0.012),
             size=(0.012, 0.020, 0.032), color=c.stub_color, collide=collide)
    return root


def _spawn_basket(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the basket: DYNAMIC open box + handle struts + handle BAR. Origin at the
    basket floor bottom centre (authored mass => CoM there: hangs level, low CoM)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    # Heavy damping: a bar rocking on a round peg is a near-lossless limit cycle in
    # PhysX (edge contact) — model a lossy wicker pendulum so it can actually settle.
    _dyn_props(root, lin_damp=0.25, ang_damp=2.5)
    collide = _make_collide(cfg)
    c = cfg
    ox, oy = c.out_x / 2, c.out_y / 2  # outer half extents
    _add_box(stage, f"{prim_path}/floor",
             center=(0.0, 0.0, c.floor_t / 2),
             size=(c.out_x, c.out_y, c.floor_t), color=c.color, collide=collide)
    wz = c.floor_t + c.wall_h / 2
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/wall_x{'p' if sgn > 0 else 'n'}",
                 center=(sgn * (ox - c.wall_t / 2), 0.0, wz),
                 size=(c.wall_t, c.out_y, c.wall_h), color=c.color, collide=collide)
        _add_box(stage, f"{prim_path}/wall_y{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * (oy - c.wall_t / 2), wz),
                 size=(c.out_x - 2 * c.wall_t, c.wall_t, c.wall_h),
                 color=c.color, collide=collide)
    rim_z = c.floor_t + c.wall_h
    strut_h = c.bar_z - c.bar_t / 2 - rim_z
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/strut_{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * (oy - c.wall_t / 2), rim_z + strut_h / 2),
                 size=(c.bar_t, c.wall_t, strut_h), color=c.handle_color,
                 collide=collide)
    _add_box(stage, f"{prim_path}/bar",
             center=(0.0, 0.0, c.bar_z),
             size=(c.bar_t, c.out_y + 0.004, c.bar_t), color=c.handle_color,
             collide=collide)
    return root


def _spawn_bottle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a squeeze bottle: DYNAMIC body cylinder + thinner cap cylinder along
    local +z. Origin at the body cylinder's centre."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    # Rolling in the swinging basket couples into its pendulum mode — damp it out.
    _dyn_props(root, lin_damp=0.12, ang_damp=1.2)
    collide = _make_collide(cfg)
    c = cfg
    _add_cyl(stage, f"{prim_path}/body",
             center=(0.0, 0.0, 0.0), radius=c.body_r, height=c.body_h, axis="Z",
             color=c.color, collide=collide)
    _add_cyl(stage, f"{prim_path}/cap",
             center=(0.0, 0.0, c.body_h / 2 + c.cap_h / 2), radius=c.cap_r,
             height=c.cap_h, axis="Z", color=c.cap_color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "stand" not in _SPAWNER_CACHE:

        @configclass
        class StandSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stand)
            base_d: float = 0.24
            base_w: float = 0.44
            base_t: float = 0.024
            board_t: float = 0.024
            board_w: float = 0.36
            board_h: float = 0.48
            color: tuple = (0.52, 0.36, 0.22)
            contact_offset: float = 0.002

        @configclass
        class PegSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_peg)
            peg_r: float = 0.008
            peg_len: float = 0.156
            color: tuple = (0.62, 0.63, 0.66)
            stub_color: tuple = (0.75, 0.20, 0.15)
            contact_offset: float = 0.002

        @configclass
        class BasketSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_basket)
            out_x: float = 0.16
            out_y: float = 0.12
            floor_t: float = 0.010
            wall_h: float = 0.10
            wall_t: float = 0.008
            bar_z: float = 0.20
            bar_t: float = 0.012
            mass: float = 0.22
            color: tuple = (0.16, 0.45, 0.20)
            handle_color: tuple = (0.22, 0.58, 0.28)
            contact_offset: float = 0.002

        @configclass
        class BottleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bottle)
            body_r: float = 0.0275
            body_h: float = 0.105
            cap_r: float = 0.0255
            cap_h: float = 0.028
            mass: float = 0.30
            color: tuple = (0.5, 0.5, 0.5)
            cap_color: tuple = (0.95, 0.95, 0.92)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(stand=StandSpawnerCfg, peg=PegSpawnerCfg,
                              basket=BasketSpawnerCfg, bottle=BottleSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class HookHangBasketSceneCfg(BaseCfg):
    """Config for `HookHangBasketScene`. The decoy interlock is metric: hang depth
    (bar centre to basket bottom) is bar_z - bar_t/2 - peg_r = 186 mm; the LOW peg
    (0.13 m) leaves that below the floor, so a basket hung there stands on the ground
    and fails the hang-band + floor-clearance gates in every randomized layout."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.70)  # max |ang vel| when judging (rad/s)
    upright_max_deg: float = tunable(28.0)  # basket up-axis cone (loaded tilt ~9 deg)
    clear_z: float = tunable(0.06)  # basket origin min height: suspended, not grounded
    hang_lo: float = tunable(-0.215)  # basket origin band in the peg frame (hang depth
    hang_hi: float = tunable(-0.145)  # is -0.186 nominal)
    streak_n: int = tunable(10)  # consecutive steps to latch hung / loaded

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    stand_dy_max: float = tunable(0.05)  # stand lateral offset (+/- m)
    stand_yaw_jit_deg: float = tunable(10.0)  # stand yaw jitter about facing the robot
    peg_z_jit: float = tunable(0.012)  # per-peg height jitter (+/- m)
    side_swap: bool = tunable(True)  # Bernoulli high/low side swap
    basket_jit: float = tunable(0.03)  # basket spawn xy jitter (+/- m)
    slot_jitter: float = tunable(0.03)  # per-bottle spawn xy jitter (+/- m)
    swap_slots: bool = tunable(True)  # Bernoulli ketchup/bbq slot swap

    # --- info: layout (single Franka base at the origin; radii 0.20-0.55 m) ---------------------
    stand_x: float = info(0.52)  # stand board distance from the base
    peg_y: float = info(0.11)  # peg lateral offset on the board (+/-)
    high_z: float = info(0.36)  # high peg root height
    low_z: float = info(0.13)  # low peg root height (the decoy)
    basket_start: tuple = info((0.26, 0.0))  # basket spawn on the floor
    slot_a: tuple = info((0.24, 0.20))  # bottle spawn slot A (left)
    slot_b: tuple = info((0.24, -0.20))  # bottle spawn slot B (right)

    # --- info: stand / peg structure -------------------------------------------------------------
    base_d: float = info(0.24)
    base_w: float = info(0.44)
    base_t: float = info(0.024)
    board_t: float = info(0.024)
    board_w: float = info(0.36)
    board_h: float = info(0.48)
    peg_r: float = info(0.008)
    peg_len: float = info(0.156)  # rod length; retaining stub just past the tip
    stand_color: tuple = info((0.52, 0.36, 0.22))  # warm wood
    peg_color: tuple = info((0.62, 0.63, 0.66))  # steel gray
    stub_color: tuple = info((0.75, 0.20, 0.15))  # red tip stub

    # --- info: basket ----------------------------------------------------------------------------
    out_x: float = info(0.16)  # outer footprint
    out_y: float = info(0.12)
    floor_t: float = info(0.010)
    wall_h: float = info(0.10)  # rim at floor_t + wall_h = 0.11
    wall_t: float = info(0.008)
    bar_z: float = info(0.20)  # handle bar centre above the basket bottom
    bar_t: float = info(0.012)  # bar square section (parallel-jaw graspable)
    basket_mass: float = info(0.22)
    basket_color: tuple = info((0.16, 0.45, 0.20))  # green
    handle_color: tuple = info((0.22, 0.58, 0.28))

    # --- info: bottles ---------------------------------------------------------------------------
    body_r: float = info(0.0275)  # 55 mm body dia
    body_h: float = info(0.105)
    cap_r: float = info(0.0255)
    cap_h: float = info(0.028)
    ketchup_mass: float = info(0.30)
    bbq_mass: float = info(0.28)
    ketchup_color: tuple = info((0.72, 0.07, 0.05))  # red, white cap
    ketchup_cap: tuple = info((0.95, 0.95, 0.92))
    bbq_color: tuple = info((0.30, 0.14, 0.07))  # dark brown, black cap
    bbq_cap: tuple = info((0.08, 0.08, 0.08))

    contact_offset: float = info(0.002)
    # rubric weights (0.25 + 0.10 + 0.30 = 0.65 = the non-success cap)
    w_hang: float = info(0.25)
    w_in: float = info(0.10)
    w_load: float = info(0.30)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("hook_hang_basket")
class HookHangBasketScene(BaseScene):
    cfg: HookHangBasketSceneCfg

    def __init__(self, cfg: HookHangBasketSceneCfg | None = None) -> None:
        super().__init__(cfg or HookHangBasketSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        stand_spawn = spawners["stand"](
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            base_d=c.base_d, base_w=c.base_w, base_t=c.base_t, board_t=c.board_t,
            board_w=c.board_w, board_h=c.board_h, color=c.stand_color,
            contact_offset=c.contact_offset,
        )

        def peg_spawn():
            return spawners["peg"](
                mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                peg_r=c.peg_r, peg_len=c.peg_len, color=c.peg_color,
                stub_color=c.stub_color, contact_offset=c.contact_offset,
            )

        basket_spawn = spawners["basket"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.basket_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            out_x=c.out_x, out_y=c.out_y, floor_t=c.floor_t, wall_h=c.wall_h,
            wall_t=c.wall_t, bar_z=c.bar_z, bar_t=c.bar_t, mass=c.basket_mass,
            color=c.basket_color, handle_color=c.handle_color,
            contact_offset=c.contact_offset,
        )

        def bottle_spawn(mass, color, cap_color):
            return spawners["bottle"](
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                body_r=c.body_r, body_h=c.body_h, cap_r=c.cap_r, cap_h=c.cap_h,
                mass=mass, color=color, cap_color=cap_color,
                contact_offset=c.contact_offset,
            )

        # initial poses are placeholders; reset() writes the real randomized layout
        qpi = (0.0, 0.0, 0.0, 1.0)  # yaw pi: stand local +x faces the robot
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
            "stand": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stand",
                spawn=stand_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.stand_x, 0.0, 0.0),
                                                          rot=qpi),
            ),
            "peg_a": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PegA",
                spawn=peg_spawn(),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stand_x - c.board_t / 2, -c.peg_y, c.high_z), rot=qpi),
            ),
            "peg_b": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PegB",
                spawn=peg_spawn(),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stand_x - c.board_t / 2, c.peg_y, c.low_z), rot=qpi),
            ),
            "basket": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Basket",
                spawn=basket_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.basket_start[0], c.basket_start[1], 0.002)),
            ),
            "ketchup": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ketchup",
                spawn=bottle_spawn(c.ketchup_mass, c.ketchup_color, c.ketchup_cap),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_a[0], c.slot_a[1], c.body_h / 2 + 0.002)),
            ),
            "bbq": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bbq",
                spawn=bottle_spawn(c.bbq_mass, c.bbq_color, c.bbq_cap),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_b[0], c.slot_b[1], c.body_h / 2 + 0.002)),
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
        self.stand: RigidObject = env.iscene["stand"]
        self.peg_a: RigidObject = env.iscene["peg_a"]
        self.peg_b: RigidObject = env.iscene["peg_b"]
        self.basket: RigidObject = env.iscene["basket"]
        self.ketchup: RigidObject = env.iscene["ketchup"]
        self.bbq: RigidObject = env.iscene["bbq"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._hang_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._load_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._hung_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._in_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._loaded_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._high_is_a = torch.zeros(n, dtype=torch.bool, device=dev)  # readback aid

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: stand re-posed (lateral offset + yaw about facing the robot),
        HIGH/LOW peg side assignment Bernoulli-swapped with per-peg z jitter, basket
        upright on the floor (xy jitter + free yaw), bottles randomly ASSIGNED to the
        two spawn slots (+ xy jitter) standing up, latches cleared."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- stand (kinematic): yaw ~ pi (board face toward the robot) + jitter ---
        dy = (torch.rand(m, device=dev) * 2 - 1) * c.stand_dy_max
        yaw = math.pi + (torch.rand(m, device=dev) * 2 - 1) \
            * math.radians(c.stand_yaw_jit_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = c.stand_x, dy
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.stand.write_root_state_to_sim(st, env_ids)
        s_pos, s_quat = st[:, 0:3].clone(), st[:, 3:7].clone()

        # --- pegs (kinematic): posed on the board face; high/low side swap + z jitter ---
        if c.side_swap:
            high_a = torch.rand(m, device=dev) < 0.5
        else:
            high_a = torch.ones(m, dtype=torch.bool, device=dev)
        self._high_is_a[env_ids] = high_a
        zj = (torch.rand(m, 2, device=dev) * 2 - 1) * c.peg_z_jit
        z_a = torch.where(high_a, c.high_z + zj[:, 0], c.low_z + zj[:, 0])
        z_b = torch.where(high_a, c.low_z + zj[:, 1], c.high_z + zj[:, 1])
        for peg, y_sgn, z in ((self.peg_a, 1.0, z_a), (self.peg_b, -1.0, z_b)):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0], loc[:, 1], loc[:, 2] = c.board_t / 2, y_sgn * c.peg_y, z
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = s_pos + quat_apply(s_quat, loc)
            st[:, 3:7] = s_quat
            peg.write_root_state_to_sim(st, env_ids)

        # --- basket: upright on the floor, xy jitter + free yaw ---
        byaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.basket_start[0]
        st[:, 1] = c.basket_start[1]
        st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.basket_jit
        st[:, 2] = 0.002
        st[:, 3], st[:, 6] = torch.cos(byaw / 2), torch.sin(byaw / 2)
        st[:, 0:3] += origin
        self.basket.write_root_state_to_sim(st, env_ids)

        # --- bottles: Bernoulli slot swap + jitter, standing up ---
        if c.swap_slots:
            swap = torch.rand(m, device=dev) < 0.5
        else:
            swap = torch.zeros(m, dtype=torch.bool, device=dev)
        slot_a = torch.tensor(c.slot_a, device=dev).expand(m, 2)
        slot_b = torch.tensor(c.slot_b, device=dev).expand(m, 2)
        k_xy = torch.where(swap.unsqueeze(1), slot_b, slot_a) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        q_xy = torch.where(swap.unsqueeze(1), slot_a, slot_b) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        for body, xy in ((self.ketchup, k_xy), (self.bbq, q_xy)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = c.body_h / 2 + 0.002
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._hang_streak[env_ids] = 0
        self._load_streak[env_ids] = 0
        self._hung_ever[env_ids] = False
        self._in_ever[env_ids] = False
        self._loaded_ever[env_ids] = False

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "stand": self.stand.data.root_state_w[env_ids].clone(),
            "peg_a": self.peg_a.data.root_state_w[env_ids].clone(),
            "peg_b": self.peg_b.data.root_state_w[env_ids].clone(),
            "basket": self.basket.data.root_state_w[env_ids].clone(),
            "ketchup": self.ketchup.data.root_state_w[env_ids].clone(),
            "bbq": self.bbq.data.root_state_w[env_ids].clone(),
            "hang_streak": self._hang_streak[env_ids].clone(),
            "load_streak": self._load_streak[env_ids].clone(),
            "hung_ever": self._hung_ever[env_ids].clone(),
            "in_ever": self._in_ever[env_ids].clone(),
            "loaded_ever": self._loaded_ever[env_ids].clone(),
            "high_is_a": self._high_is_a[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.stand.write_root_state_to_sim(state["stand"], env_ids)
        self.peg_a.write_root_state_to_sim(state["peg_a"], env_ids)
        self.peg_b.write_root_state_to_sim(state["peg_b"], env_ids)
        self.basket.write_root_state_to_sim(state["basket"], env_ids)
        self.ketchup.write_root_state_to_sim(state["ketchup"], env_ids)
        self.bbq.write_root_state_to_sim(state["bbq"], env_ids)
        self._hang_streak[env_ids] = state["hang_streak"]
        self._load_streak[env_ids] = state["load_streak"]
        self._hung_ever[env_ids] = state["hung_ever"]
        self._in_ever[env_ids] = state["in_ever"]
        self._loaded_ever[env_ids] = state["loaded_ever"]
        self._high_is_a[env_ids] = state["high_is_a"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        hang_depth = c.bar_z - c.bar_t / 2 - c.peg_r
        return (
            f"On the floor stands a wooden wall STAND (a {c.board_w * 100:.0f} cm wide, "
            f"{(c.base_t + c.board_h) * 100:.0f} cm tall board on a base) with TWO steel "
            f"PEGS sticking out horizontally toward you ({c.peg_len * 100:.0f} cm long, "
            f"{2 * c.peg_r * 1000:.0f} mm thick, each ending in a small red retaining "
            f"stub): one HIGH (~{c.high_z * 100:.0f} cm up) and one LOW "
            f"(~{c.low_z * 100:.0f} cm up) — WHICH SIDE is the high one varies between "
            f"episodes, so look. In front of it a GREEN open-top BASKET "
            f"({c.out_x * 100:.0f} x {c.out_y * 100:.0f} cm, rim at "
            f"{(c.floor_t + c.wall_h) * 100:.0f} cm) sits on the floor; above its "
            f"opening arches a HANDLE with a horizontal grab BAR "
            f"{c.bar_z * 100:.0f} cm above the basket bottom, so a basket hanging by "
            f"its bar dangles {hang_depth * 100:.1f} cm below the peg it rests on. "
            f"Nearby stand two squeeze bottles (their positions swap between episodes "
            f"— identify by COLOR): a RED ketchup bottle with a white cap and a dark "
            f"BROWN barbecue-sauce bottle with a black cap, both "
            f"{2 * c.body_r * 100:.1f} cm dia x ~{(c.body_h + c.cap_h) * 100:.1f} cm "
            f"tall.\n"
            f"Goal: the basket must end up HANGING by its handle bar from a peg, "
            f"SUSPENDED CLEAR OF THE FLOOR, with the RED ketchup bottle resting INSIDE "
            f"it; the BROWN bottle stays out, and everything must come to rest. Note "
            f"the hang depth: only the HIGH peg leaves the basket airborne — on the "
            f"LOW peg the basket still touches the ground, which does not count. "
            f"Either order works: you may hang the empty basket and then lower the "
            f"ketchup in through the opening (drop it gently — a hard drop swings the "
            f"basket off the peg or bounces the bottle out), or load the basket on "
            f"the floor first and hang the loaded basket by its bar. Ketchup in a "
            f"basket left on the floor, a basket on the low peg or propped against "
            f"the stand, the brown bottle in the basket (alone or additionally), or "
            f"anything still moving — all failure."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Hang the green basket by its handle bar on the higher wall peg so it "
            "hangs clear of the floor, then put the red ketchup bottle inside the "
            "hanging basket and let everything settle. Keep the brown sauce bottle "
            "out of the basket."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def _bar_center_w(self) -> torch.Tensor:
        """(N, 3) world position of the handle bar centre."""
        from isaaclab.utils.math import quat_apply

        off = torch.tensor([0.0, 0.0, self.cfg.bar_z], device=self.env.device)
        return self.basket.data.root_pos_w + quat_apply(
            self.basket.data.root_quat_w, off.expand(self.env.num_envs, 3))

    def _peg_local(self, peg: RigidObject, p_w: torch.Tensor) -> torch.Tensor:
        """(N, 3) world points expressed in the peg's frame (+x along the peg)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(peg.data.root_quat_w, p_w - peg.data.root_pos_w)

    def _basket_upright(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        up = quat_apply(self.basket.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.upright_max_deg))

    def _engaged_on(self, peg: RigidObject) -> torch.Tensor:
        """(N,) bool: handle bar resting ON this peg with the basket dangling below at
        hang depth, clear of the floor, upright. Peg-frame math, valid under stand yaw
        randomization. The z window's top (0.034) admits bar-on-peg (nominal 0.014)
        and the x window's top (peg_len - 0.006) excludes a bar balanced on the stub."""
        c = self.cfg
        b = self._peg_local(peg, self._bar_center_w())
        o = self._peg_local(peg, self.basket.data.root_pos_w)
        in_x = (b[:, 0] >= 0.015) & (b[:, 0] <= c.peg_len - 0.006)
        in_y = b[:, 1].abs() <= (c.out_y + 0.004) / 2 - 0.010  # bar must CROSS the peg
        in_z = (b[:, 2] >= 0.002) & (b[:, 2] <= 0.034)
        hang = (o[:, 2] >= c.hang_lo) & (o[:, 2] <= c.hang_hi)
        clear = (self.basket.data.root_pos_w - self.env_origins)[:, 2] >= c.clear_z
        return in_x & in_y & in_z & hang & clear & self._basket_upright()

    def engaged_hung(self) -> torch.Tensor:
        """(N,) bool: basket hanging by its bar on EITHER peg, suspended. Only the
        high peg can physically satisfy this (hang depth 186 mm > low peg height)."""
        return self._engaged_on(self.peg_a) | self._engaged_on(self.peg_b)

    def _basket_local(self, p_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.basket.data.root_quat_w,
                                  p_w - self.basket.data.root_pos_w)

    def contained(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body CoM inside the basket cavity, judged in the BASKET frame
        (a hanging tilted basket judges identically): within the floor extents, above
        the floor, and BELOW THE RIM minus margin (a bottle perched on the rim or bar
        does not count)."""
        c = self.cfg
        loc = self._basket_local(body.data.root_pos_w)
        rim = c.floor_t + c.wall_h
        return ((loc[:, 0].abs() <= c.out_x / 2 - c.wall_t - 0.007)
                & (loc[:, 1].abs() <= c.out_y / 2 - c.wall_t - 0.007)
                & (loc[:, 2] >= 0.004) & (loc[:, 2] <= rim - 0.015))

    def _still(self) -> torch.Tensor:
        c = self.cfg
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for body in (self.basket, self.ketchup):
            ok &= body.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
            ok &= body.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega
        return ok

    # ----- step-coupled bookkeeping (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No plant (the stand and pegs are kinematic) — just run the streak counters
        and latch rubric progress every step so transient achievements keep credit.
        Streaks only advance HERE (success()/score() are read-only)."""
        eng = self.engaged_hung()
        self._hang_streak = torch.where(eng, self._hang_streak + 1,
                                        torch.zeros_like(self._hang_streak))
        self._hung_ever |= self._hang_streak >= self.cfg.streak_n
        cont = self.contained(self.ketchup)
        self._in_ever |= cont
        loaded = eng & cont
        self._load_streak = torch.where(loaded, self._load_streak + 1,
                                        torch.zeros_like(self._load_streak))
        self._loaded_ever |= self._load_streak >= self.cfg.streak_n

    def success(self) -> torch.Tensor:
        """(N,) bool: basket ENGAGED-HUNG on a peg (suspended clear of the floor) AND
        ketchup contained in it AND bbq NOT contained AND basket + ketchup still.
        Physical outcomes only — supported through the bar-peg contact, judged live."""
        return (self.engaged_hung() & self.contained(self.ketchup)
                & ~self.contained(self.bbq) & self._still())

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.25*hung + 0.10*in + 0.30*loaded — all latched, ~0
        for doing nothing, capped 0.65 — and exactly 1.0 iff success() holds live.
        The seed's plan (ketchup into the basket where it lies) tops out at 0.10."""
        c = self.cfg
        base = (c.w_hang * self._hung_ever.float() + c.w_in * self._in_ever.float()
                + c.w_load * self._loaded_ever.float()).clamp(max=0.65)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through applied wrenches.
register_env("simgen", lambda: EnvCfg(scene="hook_hang_basket", robot="null"))

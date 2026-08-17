"""MarbleRouterScene — set the switch fins to spell the route to the marked
bin, pull the release gate, and let gravity deliver the sealed marble
(sim_gen task `scene_b_i305`).

Derived from calvin/scene_B but STRATEGICALLY different: the CALVIN table is a
menu of independent binary single-DOF primitives (button, switch, slider,
drawer — each one "done" when pushed to an end-stop) plus free blocks the arm
picks or pushes on an open tabletop. Here every DOF the robot may touch is
purely INSTRUMENTAL: a pitched, roofed switchyard seals a marble the arm can
never grasp once released, and the goal is a place-outcome of that untouchable
payload — the marble at rest in the one bin singled out by a floor marker.
Two railroad-switch fins (revolutes about the board normal, gravity-bistable
between hard stops) form a 2-bit routing program that must be WRITTEN FIRST;
a lateral sliding gate (prismatic, gravity-neutral) then commits the plan
irreversibly: the marble rolls the tree hands-off and lands wherever the fins
pointed. No CALVIN objective conditions one DOF's meaning on another, none is
irreversible, and none is judged by where a never-touched object ends up.

No stored energy beyond the marble's own gravity on the pitched board: fins
are plain damped revolutes between stops (gravity presses them into whichever
stop they are near — bistable, no springs), the gate a damped horizontal
prismatic between stops. Every intermediate configuration persists hands-off.

Assets are fully procedural: one KINEMATIC compound rig (pitched board, walls,
junction islands, roof plates with fin-post slots and bin viewing slits, four
legs, four colored bin flags), three DYNAMIC fins and one DYNAMIC gate joined
to the rig by bind-time per-env USD joints (collision filtering applies only
to each jointed pair, so every marble contact stays live), a 40 mm marble,
and a white KINEMATIC floor marker teleported in front of the target bin at
reset. The rig pose is FIXED (joints anchored to a kinematic body stay
world-fixed), so randomization lives in the target/marker, the fin start
angles, and the marble's start pose.

Per-episode randomization (readback-verified by smoke): target bin t in 0..3
(both path fins start at the WRONG stop; the off-path fin starts at a random
stop), marker lateral jitter, marble start jitter.

Rubric (0..1; latched partial credit anchored in the demonstrated solve):
  0.20  routed   — both path fins at their correct stops while the gate is
                   still shut and the marble still waits behind it
  0.20  released — gate pulled past the open threshold while the path fins
                   are correct
  0.15  branched — the marble crosses junction A into the correct half
capped at 0.55; exactly 1.0 iff success(): the marble at rest inside the
TARGET bin (board-frame box), settled and finite — a LIVE outcome with no
memory. Null policy ~0 (the marble waits behind the shut gate forever). The
seed reflex — yank the slider end-stop without setting the switches — sends
the marble to a wrong bin: 0 credit, irreversibly failed.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
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

_G = 9.81


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


def _span(stage, path: str, *, x, y, z, color, collide: Callable):
    """Box child from axis spans (x0, x1), (y0, y1), (z0, z1)."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2))
    xf.AddScaleOp().Set(Gf.Vec3f(x[1] - x[0], y[1] - y[0], z[1] - z[0]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _seg(stage, path: str, *, p0, p1, t, z, color, collide: Callable, side: float = 0.0):
    """Oriented wall box whose INNER FACE runs p0 -> p1 (board-frame xy).

    `side` = which way the body of the wall is offset from the p0->p1 line:
    +1 offsets toward +normal (left of travel), -1 toward -normal, 0 centers
    the wall on the line. The box is extended t/2 at each end so consecutive
    segments seal their corners.
    """
    from pxr import Gf, UsdGeom

    x0, y0 = p0
    x1, y1 = p1
    length = math.hypot(x1 - x0, y1 - y0) + t
    yaw = math.atan2(y1 - y0, x1 - x0)
    nx, ny = -math.sin(yaw), math.cos(yaw)
    cx = (x0 + x1) / 2 + side * nx * t / 2
    cy = (y0 + y1) / 2 + side * ny * t / 2
    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, (z[0] + z[1]) / 2))
    half = yaw / 2
    xf.AddOrientOp().Set(Gf.Quatf(math.cos(half), Gf.Vec3f(0.0, 0.0, math.sin(half))))
    xf.AddScaleOp().Set(Gf.Vec3f(length, t, z[1] - z[0]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _mk_material(prim_path: str, name: str, mu_s: float, mu_d: float, combine: str) -> str:
    import isaaclab.sim as sim_utils

    mat_path = f"{prim_path}/{name}"
    sim_utils.spawn_rigid_body_material(mat_path, sim_utils.RigidBodyMaterialCfg(
        static_friction=float(mu_s), dynamic_friction=float(mu_d), restitution=0.0,
        friction_combine_mode=combine))
    return mat_path


def _dyn_body(root, mass: float, lin_damp: float, ang_damp: float,
              com=None, inertia=None) -> None:
    """Standard dynamic compound body physics (32/4 iters, no sleep, damped).
    Custom spawners must author MassAPI explicitly (density-mass trap) — and
    mass alone leaves the CoM at the body origin, so CoM / diagonal inertia
    are authored here too when given."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    mapi = UsdPhysics.MassAPI.Apply(root)
    mapi.CreateMassAttr(float(mass))
    if com is not None:
        mapi.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    if inertia is not None:
        mapi.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(4)
    prb.CreateLinearDampingAttr(float(lin_damp))
    prb.CreateAngularDampingAttr(float(ang_damp))
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)


def _spawn_rig(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the switchyard rig: one KINEMATIC compound in the BOARD frame
    (x downhill along the pitched board, y lateral, z board-normal; z=0 is the
    board top surface — the root xform carries the world pitch). Floor, chute
    walls (with the gate slot cut out of the +y wall), Y-junction flares and
    islands, bin dividers, end walls, a roof with fin-post slots and bin
    viewing slits, four legs, and a colored flag over each bin."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    hw = c.chan_hw
    t = c.wall_t
    w0, w1 = 0.0, c.wall_h
    yc = c.branch_y                      # junction-B channel centre offset
    dy = c.child_dy                      # bin centre offset from its junction centre
    out1 = yc + hw                       # branch channel outer edge (0.1425)
    inn1 = yc - hw                       # branch channel inner edge (0.0725)
    out2 = yc + dy + hw                  # outermost bin edge (0.200)
    inn2 = yc - dy - hw                  # innermost bin edge (0.015)
    bi = dy - hw                         # B-island half width (0.0225)
    half_w = out2 + t                    # board half width (0.212)
    wc, ic, rc = c.wall_color, c.island_color, c.roof_color

    def span(name, x, y, z, color=None):
        _span(stage, f"{prim_path}/{name}", x=x, y=y, z=z, color=color or wc, collide=collide)

    def seg(name, p0, p1, side, color=None):
        _seg(stage, f"{prim_path}/{name}", p0=p0, p1=p1, t=t, z=(w0, w1),
             color=color or wc, collide=collide, side=side)

    # floor / end walls / legs
    span("floor", (c.floor_x0, c.board_end), (-half_w, half_w), (-c.floor_t, 0.0),
         color=c.floor_color)
    span("wall_start", (c.floor_x0, c.floor_x0 + t), (-hw - t, hw + t), (w0, w1))
    span("wall_end", (c.bins_end, c.board_end), (-half_w, half_w), (w0, w1))
    for i, (lx, ly) in enumerate(((0.04, 0.18), (0.04, -0.18), (0.95, 0.18), (0.95, -0.18))):
        span(f"leg_{i}", (lx - 0.02, lx + 0.02), (ly - 0.02, ly + 0.02),
             (-c.leg_drop, -c.floor_t), color=c.floor_color)

    # start chute walls (inner faces +/-hw); the gate slot is cut from BOTH
    # walls (the +y side passes the stem/plate, the -y side receives the
    # plate tip) — each slot is narrower than the marble
    x0 = c.floor_x0 + t
    span("chute_yn_a", (x0, c.gate_x0), (-hw - t, -hw), (w0, w1))
    span("chute_yn_b", (c.gate_x1, c.chute_end + 0.005), (-hw - t, -hw), (w0, w1))
    span("chute_yp_a", (x0, c.gate_x0), (hw, hw + t), (w0, w1))
    span("chute_yp_b", (c.gate_x1, c.chute_end + 0.005), (hw, hw + t), (w0, w1))

    # junction A flares (outer), islands, junction B flares
    seg("flare1_yp", (c.chute_end, hw), (c.a_body_end, out1), +1.0)
    seg("flare1_yn", (c.chute_end, -hw), (c.a_body_end, -out1), -1.0)
    seg("flare2_yp", (c.a_body_end, out1), (c.b_end, out2), +1.0)
    seg("flare2_yn", (c.a_body_end, -out1), (c.b_end, -out2), -1.0)
    span("outer_yp", (c.b_end, c.bins_end), (out2, out2 + t), (w0, w1))
    span("outer_yn", (c.b_end, c.bins_end), (-out2 - t, -out2), (w0, w1))
    # A island: flank starts pulled downstream of the apex so the fin pivot
    # region stays clear (the residual apex gap is far smaller than the ball)
    a0 = c.a_pull
    fr = inn1 / (c.a_end - c.a_x)        # A flank slope
    seg("a_flank_yp", (c.a_x + a0, a0 * fr), (c.a_end, inn1), -1.0, color=ic)
    seg("a_flank_yn", (c.a_x + a0, -a0 * fr), (c.a_end, -inn1), +1.0, color=ic)
    span("a_body", (c.a_end, c.a_body_end), (-inn1, inn1), (w0, w1), color=ic)
    seg("b_inner_yp", (c.a_body_end, inn1), (c.b_end, inn2), -1.0, color=ic)
    seg("b_inner_yn", (c.a_body_end, -inn1), (c.b_end, -inn2), +1.0, color=ic)
    span("mid_body", (c.b_end, c.bins_end), (-inn2, inn2), (w0, w1), color=ic)
    b0 = c.b_pull
    br = bi / (c.b_end - c.b_x)          # B flank slope
    for tag, s in (("l", 1.0), ("r", -1.0)):
        cy = s * yc
        seg(f"b{tag}_flank_out", (c.b_x + b0, cy + s * b0 * br), (c.b_end, cy + s * bi),
            -s, color=ic)
        seg(f"b{tag}_flank_in", (c.b_x + b0, cy - s * b0 * br), (c.b_end, cy - s * bi),
            +s, color=ic)
        span(f"b{tag}_body", (c.b_end, c.bins_end), (min(cy - s * bi, cy + s * bi),
             max(cy - s * bi, cy + s * bi)), (w0, w1), color=ic)

    # roof (z roof_z0..roof_z1) with fin-post slots and bin viewing slits
    r0, r1 = c.wall_h, c.roof_z1
    ha0, ha1, hah = c.hole_a          # A post slot: x span + y half width
    hb0, hb1, hbh = c.hole_b          # B post slots
    sl = c.slit_hw
    bys = (yc + dy, yc - dy, -yc + dy, -yc - dy)

    def roof(name, x, y):
        span(name, x, y, (r0, r1), color=rc)

    roof("roof_1", (c.roof_x0, ha0), (-half_w, half_w))
    roof("roof_2a", (ha0, ha1), (-half_w, -hah))
    roof("roof_2b", (ha0, ha1), (hah, half_w))
    roof("roof_3", (ha1, hb0), (-half_w, half_w))
    roof("roof_4a", (hb0, hb1), (-half_w, -yc - hbh))
    roof("roof_4b", (hb0, hb1), (-yc + hbh, yc - hbh))
    roof("roof_4c", (hb0, hb1), (yc + hbh, half_w))
    roof("roof_5", (hb1, c.bins_x0), (-half_w, half_w))
    edges = [-half_w]
    for by in sorted(bys):
        edges += [by - sl, by + sl]
    edges += [half_w]
    for i in range(0, len(edges), 2):
        roof(f"roof_6_{i // 2}", (c.bins_x0, c.bins_end), (edges[i], edges[i + 1]))
    roof("roof_7", (c.bins_end, c.board_end), (-half_w, half_w))

    # bin flags (identity colors, visible above the roof)
    for i, by in enumerate(bys):
        span(f"flag_{i}", c.flag_x, (by - c.flag_hw, by + c.flag_hw),
             (c.roof_z1, c.flag_top), color=c.flag_colors[i])

    mat = _mk_material(prim_path, "board", c.mu_s, c.mu_d, "average")
    bind_physics_material(prim_path, mat)
    return root


def _spawn_fin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one switch fin with root at its PIVOT (the revolute anchor):
    a thin blade extending UPSTREAM (-x, so board-plane gravity makes the fin
    bistable about its stops) and a square post rising through the roof slot
    for the gripper. Mass/CoM/inertia authored explicitly."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _span(stage, f"{prim_path}/blade", x=(-c.blade_l, 0.0), y=(-c.blade_ht, c.blade_ht),
          z=(c.blade_z0, c.blade_z1), color=c.color, collide=collide)
    _span(stage, f"{prim_path}/post", x=(-c.post_r - c.post_hw, -c.post_r + c.post_hw),
          y=(-c.post_hw, c.post_hw), z=(c.blade_z0, c.post_top), color=c.post_color,
          collide=collide)
    izz = c.mass * c.blade_l ** 2 / 3.0
    _dyn_body(root, c.mass, 0.0, c.ang_damp,
              com=(-0.45 * c.blade_l, 0.0, 0.022), inertia=(2e-5, izz, izz))
    mat = _mk_material(prim_path, "fin", c.mu_s, c.mu_d, "average")
    bind_physics_material(prim_path, mat)
    return root


def _spawn_gate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the release gate with root at the PLATE CENTRE (the prismatic
    anchor): a blocking plate spanning the chute, a stem reaching out through
    the wall slot, and a grip knob at its end. Travel is along +y (lateral,
    gravity-neutral on the pitched board)."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _span(stage, f"{prim_path}/plate", x=(-c.plate_ht, c.plate_ht),
          y=(-c.plate_hw, c.plate_hw), z=(-c.plate_hz, c.plate_hz),
          color=c.color, collide=collide)
    _span(stage, f"{prim_path}/stem", x=(-0.004, 0.004), y=(c.plate_hw - 0.005, c.knob_y0),
          z=(-0.006, 0.006), color=c.color, collide=collide)
    _span(stage, f"{prim_path}/knob", x=(-0.015, 0.015), y=(c.knob_y0, c.knob_y0 + 0.036),
          z=(-0.015, 0.015), color=c.knob_color, collide=collide)
    _dyn_body(root, c.mass, c.lin_damp, 1.0,
              com=(0.0, 0.03, 0.0), inertia=(9e-4, 2e-4, 1e-3))
    mat = _mk_material(prim_path, "gate", c.mu_s, c.mu_d, "average")
    bind_physics_material(prim_path, mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rig" not in _SPAWNER_CACHE:

        @configclass
        class RigSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rig)
            floor_x0: float = -0.022
            board_end: float = 1.004
            floor_t: float = 0.012
            chan_hw: float = 0.035
            wall_t: float = 0.012
            wall_h: float = 0.055
            roof_z1: float = 0.070
            roof_x0: float = 0.132
            gate_x0: float = 0.132
            gate_x1: float = 0.168
            chute_end: float = 0.380
            a_x: float = 0.520
            a_end: float = 0.600
            a_body_end: float = 0.640
            b_x: float = 0.720
            b_end: float = 0.800
            bins_x0: float = 0.820
            bins_end: float = 0.980
            branch_y: float = 0.1075
            child_dy: float = 0.0575
            a_pull: float = 0.012
            b_pull: float = 0.020
            hole_a: tuple = (0.456, 0.476, 0.024)
            hole_b: tuple = (0.663, 0.683, 0.024)
            slit_hw: float = 0.006
            flag_x: tuple = (0.870, 0.910)
            flag_hw: float = 0.014
            flag_top: float = 0.125
            leg_drop: float = 0.35
            floor_color: tuple = (0.55, 0.42, 0.28)
            wall_color: tuple = (0.35, 0.38, 0.44)
            island_color: tuple = (0.28, 0.30, 0.36)
            roof_color: tuple = (0.72, 0.74, 0.78)
            flag_colors: tuple = ((0.78, 0.12, 0.15), (0.10, 0.62, 0.25),
                                  (0.15, 0.35, 0.82), (0.92, 0.68, 0.10))
            contact_offset: float = 0.003
            mu_s: float = 0.45
            mu_d: float = 0.40

        @configclass
        class FinSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_fin)
            blade_l: float = 0.115
            blade_ht: float = 0.005
            blade_z0: float = 0.004
            blade_z1: float = 0.050
            post_r: float = 0.055
            post_hw: float = 0.005
            post_top: float = 0.098
            mass: float = 0.025
            ang_damp: float = 0.01
            color: tuple = (0.85, 0.30, 0.10)
            post_color: tuple = (0.95, 0.85, 0.20)
            contact_offset: float = 0.0015
            mu_s: float = 0.20
            mu_d: float = 0.18

        @configclass
        class GateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_gate)
            plate_ht: float = 0.012
            plate_hw: float = 0.045
            plate_hz: float = 0.025
            knob_y0: float = 0.240
            mass: float = 0.12
            lin_damp: float = 3.0
            color: tuple = (0.20, 0.55, 0.60)
            knob_color: tuple = (0.95, 0.85, 0.20)
            contact_offset: float = 0.0015
            mu_s: float = 0.30
            mu_d: float = 0.28

        _SPAWNER_CACHE["rig"] = RigSpawnerCfg
        _SPAWNER_CACHE["fin"] = FinSpawnerCfg
        _SPAWNER_CACHE["gate"] = GateSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class MarbleRouterSceneCfg(BaseCfg):
    """Config for `MarbleRouterScene`. The margin contract is asserted in
    `__post_init__`: the marble fits every corridor with real clearance but no
    aperture (gate slot, roof post-slots, viewing slits, fin seal gaps, apex
    gaps) passes it; a wrong fin seals its branch; the marble cannot tunnel
    any wall at terminal speed; and the success box separates adjacent bins."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    y_tol: float = tunable(0.030)         # |ball y - bin y| board-frame for success (m)
    x_lo: float = tunable(0.870)          # success box, board-frame x (m)
    x_hi: float = tunable(0.975)
    z_lo: float = tunable(0.006)          # success box, board-frame z (m)
    z_hi: float = tunable(0.045)
    settle_lin: float = tunable(0.060)    # ball |lin vel| when judging success (m/s)
    fin_tol: float = tunable(0.060)       # |fin angle - correct stop| for "routed" (rad)
    gate_shut_y: float = tunable(0.020)   # gate y below this = still shut (m)
    gate_open_y: float = tunable(0.054)   # gate y above this = released (m)
    wait_x: float = tunable(0.120)        # ball board-x below this = still waiting (m)
    branch_x: float = tunable(0.560)      # ball board-x past this = through junction A (m)
    branch_dy: float = tunable(0.015)     # |ball y| margin for the "correct half" (m)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    ball_x_min: float = tunable(0.040)    # marble start band, board frame (m)
    ball_x_max: float = tunable(0.090)
    ball_y_jit: float = tunable(0.010)
    marker_jit: float = tunable(0.008)    # marker lateral jitter (m)

    # --- info: world placement (FIXED - bind-time joints anchor to the rig) ----------------------
    pitch_deg: float = info(10.0)         # board pitch about +y (+x downhill)
    rig_pos: tuple = info((-0.10, 0.0, 0.245))
    # --- info: board layout (board frame: x downhill, y lateral, z normal, z=0 board top) --------
    floor_x0: float = info(-0.022)
    board_end: float = info(1.004)
    floor_t: float = info(0.012)
    chan_hw: float = info(0.035)          # channel inner half width
    wall_t: float = info(0.012)
    wall_h: float = info(0.055)           # wall top = roof underside
    roof_z1: float = info(0.070)
    gate_x0: float = info(0.132)          # gate slot in the +y chute wall
    gate_x1: float = info(0.168)
    gate_x: float = info(0.150)           # gate plate centre plane
    gate_z: float = info(0.027)           # gate root height (plate centre)
    gate_travel: float = info(0.10)       # prismatic limits [0, travel] along +y
    chute_end: float = info(0.380)        # chute walls end / flares begin
    a_x: float = info(0.520)              # junction A apex = fin A pivot
    a_end: float = info(0.600)            # A island flank end
    a_body_end: float = info(0.640)       # A island body end / B flares begin
    b_x: float = info(0.720)              # junction B apex = fin B pivot
    b_end: float = info(0.800)            # B flares end / bin dividers begin
    bins_x0: float = info(0.820)
    bins_end: float = info(0.980)         # end wall inner face
    branch_y: float = info(0.1075)        # B junction channel centre (+/-)
    child_dy: float = info(0.0575)        # bin centre offset from its B centre
    a_pull: float = info(0.012)           # island flank start pulled off the apex
    b_pull: float = info(0.020)
    hole_a: tuple = info((0.456, 0.476, 0.024))   # fin-post roof slots (x0, x1, y half)
    hole_b: tuple = info((0.663, 0.683, 0.024))
    slit_hw: float = info(0.006)          # bin viewing slit half width
    # --- info: fins / gate / ball / marker -------------------------------------------------------
    blade_la: float = info(0.115)         # fin A blade length (upstream of pivot)
    blade_lb: float = info(0.090)         # fin B blade length
    tip_dy_a: float = info(0.030)         # blade tip lateral throw at the stop
    tip_dy_b: float = info(0.024)
    post_r_a: float = info(0.055)         # grip post radius from pivot (upstream)
    post_r_b: float = info(0.048)
    post_hw: float = info(0.005)
    blade_ht: float = info(0.005)
    fin_mass: float = info(0.025)
    fin_write_off: float = info(0.012)    # reset writes fins this far inside the stop (rad)
    plate_hw: float = info(0.045)         # gate plate half width (y)
    plate_ht: float = info(0.012)         # gate plate half thickness (x)
    gate_mass: float = info(0.12)
    gate_write_y: float = info(0.002)     # reset writes the gate here (just off limit 0)
    ball_r: float = info(0.020)
    ball_mass: float = info(0.060)
    marker_x: float = info(1.00)          # marker tile centre, world x (on the ground)
    marker_size: float = info(0.070)
    contact_offset: float = info(0.003)   # rig/ball contact offset (anti-tunnel budget)
    # --- info: rubric weights (0.20 + 0.20 + 0.15 = 0.55 = the non-success cap) ------------------
    w_routed: float = info(0.20)
    w_released: float = info(0.20)
    w_branched: float = info(0.15)

    # ----- derived helpers ------------------------------------------------------------------
    def psi_a(self) -> float:
        """Fin A stop angle (rad) from its tip throw."""
        return math.asin(self.tip_dy_a / self.blade_la)

    def psi_b(self) -> float:
        return math.asin(self.tip_dy_b / self.blade_lb)

    def bin_ys(self) -> tuple:
        """Bin centre y's, t = 0..3 (+y outermost first)."""
        yc, dy = self.branch_y, self.child_dy
        return (yc + dy, yc - dy, -yc + dy, -yc - dy)

    def __post_init__(self) -> None:
        ball_d = 2 * self.ball_r
        psi_a, psi_b = self.psi_a(), self.psi_b()
        pitch = math.radians(self.pitch_deg)
        # -- corridors pass the marble with real clearance
        assert 2 * self.chan_hw - ball_d >= 0.024, "channel must pass the marble freely"
        assert self.wall_h - ball_d >= 0.014, "marble must roll under the roof freely"
        # -- but no aperture passes it (the yard is SEALED once the marble is released)
        assert self.gate_x1 - self.gate_x0 <= ball_d - 0.004, "gate wall slot must seal"
        assert self.hole_a[1] - self.hole_a[0] <= ball_d - 0.004, "A post slot must seal"
        assert self.hole_b[1] - self.hole_b[0] <= ball_d - 0.004, "B post slot must seal"
        assert 2 * self.slit_hw <= ball_d - 0.008, "viewing slits must seal"
        # -- gate: shut covers the chute; open threshold really passes the marble
        assert self.plate_hw >= self.chan_hw + 0.008, "shut plate must cover the chute"
        assert self.gate_write_y + self.plate_hw >= self.chan_hw + 0.006
        assert self.gate_open_y - (self.plate_hw - self.chan_hw) >= ball_d + 0.004, \
            "open threshold must pass the marble"
        assert self.gate_travel - (self.plate_hw - self.chan_hw) >= ball_d + 0.015
        assert self.gate_shut_y + 0.010 < self.gate_open_y
        assert 0.0 < self.gate_write_y < self.gate_shut_y
        # -- fin seal gaps: at the stop the blade tip nearly touches the upstream wall
        #    end, leaving a gap far smaller than the marble (both junctions, both stops)
        tip_a = (self.a_x - self.blade_la * math.cos(psi_a), self.tip_dy_a)
        gap_a = math.hypot(tip_a[0] - self.chute_end, self.chan_hw - tip_a[1])
        assert gap_a <= ball_d - 0.008, f"A seal gap {gap_a:.4f} must block the marble"
        assert tip_a[0] > self.chute_end + 0.004, "A blade must not overlap the chute wall"
        inn1 = self.branch_y - self.chan_hw
        tip_b = (self.b_x - self.blade_lb * math.cos(psi_b), self.branch_y - self.tip_dy_b)
        gap_b = math.hypot(tip_b[0] - self.a_body_end, tip_b[1] - inn1)
        assert gap_b <= ball_d - 0.008, f"B seal gap {gap_b:.4f} must block the marble"
        out1 = self.branch_y + self.chan_hw
        gap_b2 = math.hypot(tip_b[0] - self.a_body_end,
                            (self.branch_y + self.tip_dy_b) - out1)
        assert gap_b2 <= ball_d - 0.008, f"B outer seal gap {gap_b2:.4f} must block"
        # -- the B blade at its OUTER stop must NOT rest on the flare wall (a
        #    flush prop would mask the joint stop): clear the flare-1 inner
        #    line by >= 2 mm at the tip
        flare1_at_tip = self.chan_hw + (tip_b[0] - self.chute_end) \
            * (out1 - self.chan_hw) / (self.a_body_end - self.chute_end)
        tip_face = self.branch_y + self.tip_dy_b + self.blade_ht * math.cos(psi_b)
        assert tip_face <= flare1_at_tip - 0.002, \
            f"B blade would prop on the flare wall ({tip_face:.4f} vs {flare1_at_tip:.4f})"
        # -- island apex gaps (flank starts pulled off the pivot) still block the marble
        fr_a = inn1 / (self.a_end - self.a_x)
        assert 2 * self.a_pull * fr_a <= ball_d - 0.008, "A apex gap must block"
        bi = self.child_dy - self.chan_hw
        fr_b = bi / (self.b_end - self.b_x)
        assert 2 * self.b_pull * fr_b <= ball_d - 0.008, "B apex gap must block"
        assert self.a_pull >= self.wall_t / 2 + 0.004, "A flank clear of the blade sweep"
        assert self.b_pull >= self.wall_t / 2 + 0.004, "B flank clear of the blade sweep"
        # -- routed-side passages stay wide at the pinch (pivot plane)
        out_at_a = self.chan_hw + (self.a_x - self.chute_end) \
            * (out1 - self.chan_hw) / (self.a_body_end - self.chute_end)
        assert out_at_a - self.blade_ht >= ball_d + 0.008, "A routed passage too tight"
        out_at_b = out1 + (self.b_x - self.a_body_end) \
            * ((self.branch_y + self.child_dy + self.chan_hw) - out1) \
            / (self.b_end - self.a_body_end)
        assert out_at_b - self.branch_y - self.blade_ht >= ball_d + 0.008, \
            "B routed passage too tight"
        # -- anti-tunnel: terminal rolling speed x one step must not clear a wall
        v_max = math.sqrt(2 * (5.0 / 7.0) * _G * math.sin(pitch)
                          * (self.bins_end - self.ball_x_min))
        assert v_max / 120.0 <= self.wall_t + self.contact_offset, \
            f"marble would tunnel walls at {v_max:.2f} m/s"
        assert self.board_end - self.bins_end >= 2 * self.wall_t, "end wall doubly thick"
        # -- success box: separates adjacent bins, contains the physical rest point
        bys = self.bin_ys()
        for i in range(3):
            assert bys[i] - bys[i + 1] >= 2 * self.y_tol + 0.02, "bins must not overlap"
        assert self.y_tol >= (self.chan_hw - self.ball_r) + 0.005, \
            "any physical in-bin rest must pass the y test"
        rest_x = self.bins_end - self.ball_r - self.contact_offset
        assert self.x_lo + 0.02 < rest_x < self.x_hi - 0.003, "rest point inside x band"
        assert self.x_lo > self.bins_x0 + 0.02, "x band excludes the bin mouth"
        assert self.z_lo < self.ball_r < self.z_hi, "rest height inside z band"
        assert self.z_hi < self.wall_h - 0.005, "z band excludes anything above the walls"
        # -- fins: bistable (blade points upstream -> board-plane gravity presses the
        #    fin into whichever stop it is near), writes inside stops, tol < throw
        tau_g = self.fin_mass * _G * math.sin(pitch) * 0.45 * self.blade_la \
            * math.sin(psi_a)
        assert tau_g > 1e-4, "fin A gravity bias too weak to hold its stop"
        assert self.fin_write_off < min(psi_a, psi_b) / 4
        assert self.fin_tol < min(psi_a, psi_b) / 2, "fin_tol must separate the stops"
        # -- roof post slots really contain the post sweep (with margin)
        for (h0, h1, hh), pr, psi in ((self.hole_a, self.post_r_a, psi_a),
                                      (self.hole_b, self.post_r_b, psi_b)):
            px = self.a_x if pr == self.post_r_a else self.b_x
            x_min = px - pr - self.post_hw
            x_max = px - pr * math.cos(psi) + self.post_hw
            assert h0 + 0.003 <= x_min and x_max <= h1 - 0.003, "post slot x too tight"
            assert pr * math.sin(psi) + self.post_hw <= hh - 0.004, "post slot y too tight"
        # -- start band: marble spawns clear of the gate and both chute walls
        assert self.ball_x_max + self.ball_r <= self.gate_x0 - 0.010
        assert self.ball_x_min - self.ball_r >= self.floor_x0 + self.wall_t + 0.005
        assert self.ball_y_jit + self.ball_r <= self.chan_hw - 0.004
        assert self.wait_x >= self.ball_x_max + self.ball_r and self.wait_x < self.gate_x1
        # -- branched gate sits inside the A diverge, past the apex
        assert self.a_x + 0.02 < self.branch_x < self.a_body_end
        assert self.branch_dy < inn1
        # -- marker: distinct per bin, clear of the rig, jaw-graspable knobs
        assert bys[0] - bys[1] >= self.marker_size + 0.02, "markers must disambiguate"
        assert self.marker_jit <= 0.012
        assert self.marker_size <= 0.075 and 2 * self.post_hw <= 0.075
        # -- rubric weights
        assert abs(self.w_routed + self.w_released + self.w_branched - 0.55) < 1e-9


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("marble_router")
class MarbleRouterScene(BaseScene):
    cfg: MarbleRouterSceneCfg

    def __init__(self, cfg: MarbleRouterSceneCfg | None = None) -> None:
        super().__init__(cfg or MarbleRouterSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def _pitch_quat(self) -> tuple:
        p = math.radians(self.cfg.pitch_deg)
        return (math.cos(p / 2), 0.0, math.sin(p / 2), 0.0)

    def _b2w_f(self, xl: float, yl: float, zl: float) -> tuple:
        """Board frame -> world (floats, for authored poses)."""
        c = self.cfg
        p = math.radians(c.pitch_deg)
        cp, sp = math.cos(p), math.sin(p)
        px, py, pz = c.rig_pos
        return (px + xl * cp + zl * sp, py + yl, pz - xl * sp + zl * cp)

    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        rig_spawn = cls["rig"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            floor_x0=c.floor_x0, board_end=c.board_end, floor_t=c.floor_t,
            chan_hw=c.chan_hw, wall_t=c.wall_t, wall_h=c.wall_h, roof_z1=c.roof_z1,
            roof_x0=c.gate_x0, gate_x0=c.gate_x0, gate_x1=c.gate_x1,
            chute_end=c.chute_end, a_x=c.a_x, a_end=c.a_end, a_body_end=c.a_body_end,
            b_x=c.b_x, b_end=c.b_end, bins_x0=c.bins_x0, bins_end=c.bins_end,
            branch_y=c.branch_y, child_dy=c.child_dy, a_pull=c.a_pull, b_pull=c.b_pull,
            hole_a=c.hole_a, hole_b=c.hole_b, slit_hw=c.slit_hw,
            contact_offset=c.contact_offset)
        fin_a_spawn = cls["fin"](blade_l=c.blade_la, blade_ht=c.blade_ht,
                                 post_r=c.post_r_a, post_hw=c.post_hw, mass=c.fin_mass)
        fin_b_spawn = cls["fin"](blade_l=c.blade_lb, blade_ht=c.blade_ht,
                                 post_r=c.post_r_b, post_hw=c.post_hw, mass=c.fin_mass)
        gate_spawn = cls["gate"](plate_ht=c.plate_ht, plate_hw=c.plate_hw,
                                 mass=c.gate_mass)

        q = self._pitch_quat()
        rigid = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.05, angular_damping=0.05,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=32, solver_velocity_iteration_count=4)
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)
        mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.45, dynamic_friction=0.40, restitution=0.0)

        ball_pos = self._b2w_f((c.ball_x_min + c.ball_x_max) / 2, 0.0, c.ball_r + 0.001)
        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "rig": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rig", spawn=rig_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.rig_pos, rot=q)),
            # fins/gate are authored at their JOINT-ZERO poses: the bind-time
            # joints anchor against these authored positions.
            "fin_a": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/FinA", spawn=fin_a_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=self._b2w_f(c.a_x, 0.0, 0.0), rot=q)),
            "fin_bl": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/FinBL", spawn=fin_b_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=self._b2w_f(c.b_x, c.branch_y, 0.0), rot=q)),
            "fin_br": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/FinBR", spawn=fin_b_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=self._b2w_f(c.b_x, -c.branch_y, 0.0), rot=q)),
            "gate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Gate", spawn=gate_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=self._b2w_f(c.gate_x, 0.0, c.gate_z), rot=q)),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    rigid_props=rigid, collision_props=coll, physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.90, 0.92, 0.95))),
                init_state=RigidObjectCfg.InitialStateCfg(pos=ball_pos)),
            "marker": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Marker",
                spawn=sim_utils.CuboidCfg(
                    size=(c.marker_size, c.marker_size, 0.008),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.97, 0.97, 0.97))),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.marker_x, 0.0, 0.004))),
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
        from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

        self._quat_apply = quat_apply
        self._quat_apply_inv = quat_apply_inverse
        self._quat_mul = quat_mul
        c = self.cfg
        self.rig: RigidObject = env.iscene["rig"]
        self.fin_a: RigidObject = env.iscene["fin_a"]
        self.fin_bl: RigidObject = env.iscene["fin_bl"]
        self.fin_br: RigidObject = env.iscene["fin_br"]
        self.gate: RigidObject = env.iscene["gate"]
        self.ball: RigidObject = env.iscene["ball"]
        self.marker: RigidObject = env.iscene["marker"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        p = math.radians(c.pitch_deg)
        self._cp, self._sp = math.cos(p), math.sin(p)
        self._q_pitch = torch.tensor(self._pitch_quat(), device=dev).expand(n, 4)
        self._rig_pos = torch.tensor(c.rig_pos, device=dev)
        # readbacks (verified by smoke): [t, half, thA0, thBL0, thBR0, ball_x,
        # ball_y, marker_y]
        self.layout = torch.zeros(n, 8, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._routed_l = torch.zeros(n, dtype=torch.bool, device=dev)
        self._released_l = torch.zeros(n, dtype=torch.bool, device=dev)
        self._branched_l = torch.zeros(n, dtype=torch.bool, device=dev)
        self._author_joints()

    def _author_joints(self) -> None:
        """Per-env joints, authored ONCE at bind time against the AUTHORED
        (joint-zero) poses: three revolutes about the board normal for the
        fins (hard stops at the routing angles) and one horizontal prismatic
        for the gate (hard stops shut/open). Collision filtering disables only
        each rig<->body pair, so every marble contact stays live."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        c = self.cfg
        psi_a_deg = math.degrees(c.psi_a())
        psi_b_deg = math.degrees(c.psi_b())
        fins = (("FinA", (c.a_x, 0.0, 0.0), psi_a_deg),
                ("FinBL", (c.b_x, c.branch_y, 0.0), psi_b_deg),
                ("FinBR", (c.b_x, -c.branch_y, 0.0), psi_b_deg))
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            for name, pivot, lim in fins:
                j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/{name.lower()}_pivot")
                j.CreateBody0Rel().SetTargets([f"{base}/Rig"])
                j.CreateBody1Rel().SetTargets([f"{base}/{name}"])
                j.CreateCollisionEnabledAttr(False)
                j.CreateAxisAttr("Z")
                j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in pivot]))
                j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLowerLimitAttr(-float(lim))
                j.CreateUpperLimitAttr(float(lim))
            k = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/gate_slide")
            k.CreateBody0Rel().SetTargets([f"{base}/Rig"])
            k.CreateBody1Rel().SetTargets([f"{base}/Gate"])
            k.CreateCollisionEnabledAttr(False)
            k.CreateAxisAttr("Y")
            k.CreateLocalPos0Attr(Gf.Vec3f(float(c.gate_x), 0.0, float(c.gate_z)))
            k.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            k.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            k.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            k.CreateLowerLimitAttr(0.0)
            k.CreateUpperLimitAttr(float(c.gate_travel))

    # ----- frames --------------------------------------------------------------------------------
    def _b2w(self, local: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """(M,3) board frame -> world, including env origins."""
        out = torch.empty_like(local)
        out[:, 0] = self._rig_pos[0] + local[:, 0] * self._cp + local[:, 2] * self._sp
        out[:, 1] = self._rig_pos[1] + local[:, 1]
        out[:, 2] = self._rig_pos[2] - local[:, 0] * self._sp + local[:, 2] * self._cp
        return out + self.env_origins[env_ids]

    def ball_b(self) -> torch.Tensor:
        """(N,3) marble centre in the board frame."""
        d = self.ball.data.root_pos_w - self.env_origins - self._rig_pos
        out = torch.empty_like(d)
        out[:, 0] = d[:, 0] * self._cp - d[:, 2] * self._sp
        out[:, 1] = d[:, 1]
        out[:, 2] = d[:, 0] * self._sp + d[:, 2] * self._cp
        return out

    def _qz(self, ang: torch.Tensor) -> torch.Tensor:
        q = torch.zeros(ang.shape[0], 4, device=ang.device)
        q[:, 0] = torch.cos(ang / 2)
        q[:, 3] = torch.sin(ang / 2)
        return q

    def fin_angle(self, fin) -> torch.Tensor:
        """(N,) fin angle about the board normal (rad; + = tip toward -y,
        routing the marble to the +y child)."""
        qp = self._q_pitch.clone()
        qp[:, 1:] = -qp[:, 1:]
        rel = self._quat_mul(qp, fin.data.root_quat_w)
        raw = 2.0 * torch.atan2(rel[:, 3], rel[:, 0])
        return torch.atan2(torch.sin(raw), torch.cos(raw))

    def gate_y(self) -> torch.Tensor:
        """(N,) gate opening (m; 0 = shut, gate_travel = fully open)."""
        return (self.gate.data.root_pos_w - self.env_origins)[:, 1]

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: rig re-asserted at its FIXED pose (bind-time joints
        anchor to it); target bin t sampled and the white marker teleported in
        front of it; BOTH path fins written at their WRONG stops and the
        off-path fin at a random stop (all just inside the limits); the gate
        written shut; the marble dropped at a jittered start pose behind the
        gate; latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        psi_a, psi_b = c.psi_a(), c.psi_b()
        wa = psi_a - c.fin_write_off
        wb = psi_b - c.fin_write_off

        torch.rand(m, device=dev)                       # burn (first-draw degeneracy)
        u = torch.rand(m, 6, device=dev)
        ones = torch.ones(m, device=dev)
        t = (u[:, 0] * 4).long().clamp(max=3).float()
        half = torch.where(t <= 1, ones, -ones)
        # correct stops: theta_A* = half * psi_a ; theta_B* = +psi_b iff t in {0, 2}
        sgn_b = torch.where((t == 0) | (t == 2), ones, -ones)
        th_a0 = -half * wa                              # path fin A: WRONG stop
        th_path_b0 = -sgn_b * wb                        # path fin B: WRONG stop
        th_off_b0 = torch.where(u[:, 1] < 0.5, -wb * ones, wb * ones)  # off-path: random
        th_bl0 = torch.where(half > 0, th_path_b0, th_off_b0)
        th_br0 = torch.where(half > 0, th_off_b0, th_path_b0)
        ball_x = c.ball_x_min + u[:, 2] * (c.ball_x_max - c.ball_x_min)
        ball_y = (u[:, 3] * 2 - 1) * c.ball_y_jit
        bys = torch.tensor(c.bin_ys(), device=dev)
        marker_y = bys[t.long()] + (u[:, 4] * 2 - 1) * c.marker_jit

        self.layout[env_ids, 0] = t
        self.layout[env_ids, 1] = half
        self.layout[env_ids, 2] = th_a0
        self.layout[env_ids, 3] = th_bl0
        self.layout[env_ids, 4] = th_br0
        self.layout[env_ids, 5] = ball_x
        self.layout[env_ids, 6] = ball_y
        self.layout[env_ids, 7] = marker_y

        def write(body, pos_w, quat) -> None:
            s = torch.zeros(m, 13, device=dev)
            s[:, :3] = pos_w
            s[:, 3:7] = quat
            body.write_root_state_to_sim(s, env_ids)

        qp = self._q_pitch[:m]
        loc = torch.zeros(m, 3, device=dev)
        write(self.rig, self._b2w(loc, env_ids), qp)
        for fin, piv_y, piv_x, th in ((self.fin_a, 0.0, c.a_x, th_a0),
                                      (self.fin_bl, c.branch_y, c.b_x, th_bl0),
                                      (self.fin_br, -c.branch_y, c.b_x, th_br0)):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = piv_x
            loc[:, 1] = piv_y
            write(fin, self._b2w(loc, env_ids), self._quat_mul(qp, self._qz(th)))
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.gate_x
        loc[:, 1] = c.gate_write_y
        loc[:, 2] = c.gate_z
        write(self.gate, self._b2w(loc, env_ids), qp)
        loc = torch.stack([ball_x, ball_y,
                           torch.full((m,), c.ball_r + 0.001, device=dev)], dim=1)
        write(self.ball, self._b2w(loc, env_ids), qp)
        mk = torch.zeros(m, 13, device=dev)
        mk[:, 0] = self.env_origins[env_ids, 0] + c.marker_x
        mk[:, 1] = self.env_origins[env_ids, 1] + marker_y
        mk[:, 2] = self.env_origins[env_ids, 2] + 0.004
        mk[:, 3] = 1.0
        self.marker.write_root_state_to_sim(mk, env_ids)

        self._routed_l[env_ids] = False
        self._released_l[env_ids] = False
        self._branched_l[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {"layout": self.layout[env_ids].clone(),
               "routed_l": self._routed_l[env_ids].clone(),
               "released_l": self._released_l[env_ids].clone(),
               "branched_l": self._branched_l[env_ids].clone()}
        for name, body in self._bodies().items():
            out[name] = body.data.root_state_w[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for name, body in self._bodies().items():
            body.write_root_state_to_sim(state[name], env_ids)
        self.layout[env_ids] = state["layout"]
        self._routed_l[env_ids] = state["routed_l"]
        self._released_l[env_ids] = state["released_l"]
        self._branched_l[env_ids] = state["branched_l"]

    def _bodies(self) -> dict[str, Any]:
        return {"rig": self.rig, "fin_a": self.fin_a, "fin_bl": self.fin_bl,
                "fin_br": self.fin_br, "gate": self.gate, "ball": self.ball,
                "marker": self.marker}

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        psi_a_deg = math.degrees(c.psi_a())
        psi_b_deg = math.degrees(c.psi_b())
        return (
            f"A MARBLE SWITCHYARD stands on four legs: a board about "
            f"{c.board_end * 100:.0f} cm long, pitched {c.pitch_deg:.0f} degrees downhill, "
            f"with walled channels under a grey ROOF — a sealed gravity maze. At the high "
            f"end a {2 * c.ball_r * 1000:.0f} mm steel MARBLE waits in an open (unroofed) "
            f"chute behind a teal sliding GATE; its yellow grip KNOB sticks out of the "
            f"chute's side wall. Past the gate the channel forks at junction A, and each "
            f"branch forks again at a junction B, ending in FOUR parallel BINS at the low "
            f"end. A colored flag stands over each bin: crimson, green, blue, amber from "
            f"the +y side inward. Narrow roof slits let you see into the bins, but no "
            f"opening in the yard passes the marble — once released it can only be steered "
            f"by the track itself.\n"
            f"Each fork holds a RAILROAD-SWITCH FIN: an orange blade hinged at the fork's "
            f"island tip, swinging between two hard stops (+/-{psi_a_deg:.0f} degrees at A, "
            f"+/-{psi_b_deg:.0f} at B) about the board normal. A yellow square POST rises "
            f"from each blade through a slot in the roof — push or pinch the post to throw "
            f"the switch. At either stop the blade seals one branch and guides the marble "
            f"into the other: blade tip swung toward -y sends the marble to the +y branch, "
            f"and vice versa. The board's slope holds a thrown fin against its stop, and "
            f"the rolling marble presses it harder shut — but the fins start at the WRONG "
            f"stops for the goal route (the third, off-route fin starts at a random stop).\n"
            f"On the ground just past the low end lies a WHITE MARKER TILE, lined up with "
            f"one bin: that bin is the TARGET (which one varies by episode, as do the "
            f"marble's exact start spot and the marker's exact offset).\n"
            f"Goal: FIRST throw the fins along the route so the forks spell the path from "
            f"the chute to the marked bin (junction A, then the B on that side; the third "
            f"fin is irrelevant), THEN slide the gate open by its knob (it moves sideways, "
            f"about {c.gate_travel * 100:.0f} cm to its stop) and let go. The marble rolls "
            f"the yard hands-off and must come to REST inside the marked bin. Releasing "
            f"first and steering later cannot work: the marble outruns you into the wrong "
            f"bin, and there is no way to touch it again. Only the marble's final resting "
            f"bin is judged."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Find the bin the white floor tile marks. Throw the two switch fins on the "
            "route to that bin by their yellow posts — first the fork near the gate's "
            "side, then the second fork — so each blade seals the wrong branch. Then "
            "pull the teal gate open by its knob and let the marble roll: it must stop "
            "inside the marked bin."
        )

    # ----- live predicates -----------------------------------------------------------------------
    def _theta_star(self) -> tuple:
        """(N,) correct stop angles for fin A and the PATH fin B."""
        c = self.cfg
        ones = torch.ones_like(self.layout[:, 0])
        return (self.layout[:, 1] * c.psi_a(),
                torch.where((self.layout[:, 0] == 0) | (self.layout[:, 0] == 2),
                            ones, -ones) * c.psi_b())

    def fins_correct(self) -> torch.Tensor:
        """(N,) bool: both PATH fins within fin_tol of their correct stops."""
        c = self.cfg
        th_a_star, th_b_star = self._theta_star()
        half = self.layout[:, 1]
        th_b = torch.where(half > 0, self.fin_angle(self.fin_bl),
                           self.fin_angle(self.fin_br))
        return ((self.fin_angle(self.fin_a) - th_a_star).abs() < c.fin_tol) \
            & ((th_b - th_b_star).abs() < c.fin_tol)

    def ball_waiting(self) -> torch.Tensor:
        return self.ball_b()[:, 0] < self.cfg.wait_x

    def in_target_bin(self) -> torch.Tensor:
        """(N,) bool: marble centre inside the TARGET bin's board-frame box."""
        c = self.cfg
        bys = torch.tensor(c.bin_ys(), device=self.env.device)
        y_t = bys[self.layout[:, 0].long()]
        b = self.ball_b()
        return ((b[:, 1] - y_t).abs() < c.y_tol) \
            & (b[:, 0] > c.x_lo) & (b[:, 0] < c.x_hi) \
            & (b[:, 2] > c.z_lo) & (b[:, 2] < c.z_hi)

    def settled(self) -> torch.Tensor:
        return self.ball.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin

    def _finite(self) -> torch.Tensor:
        ps = [b.data.root_pos_w for b in self._bodies().values()]
        return torch.isfinite(torch.stack(ps, dim=1)).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        ok = self.fins_correct() & fin
        self._routed_l |= ok & (self.gate_y() < c.gate_shut_y) & self.ball_waiting()
        # order-aware chain: released needs routed already latched, branched needs
        # released — gate-first play collects nothing, however it ends.
        self._released_l |= self._routed_l & ok & (self.gate_y() > c.gate_open_y)
        b = self.ball_b()
        self._branched_l |= self._released_l & ok & (b[:, 0] > c.branch_x) \
            & (b[:, 0] < c.bins_x0) \
            & (b[:, 1] * self.layout[:, 1] > c.branch_dy)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the marble is AT REST inside the TARGET bin (board-frame
        box around the marked bin, between floor and wall top), finite — a
        LIVE outcome with no memory. The yard is sealed, so the marble can
        only have arrived there through the forks the fins spelled."""
        self._update_latches()
        return self.in_target_bin() & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.20 routed + 0.20 released + 0.15 branched,
        latched, capped at 0.55; exactly 1.0 iff success() holds live. Doing
        nothing scores ~0 (the marble waits behind the gate forever); the
        seed's yank-the-slider reflex without setting the fins scores ~0 and
        irreversibly fails the episode."""
        c = self.cfg
        self._update_latches()
        base = (c.w_routed * self._routed_l.float()
                + c.w_released * self._released_l.float()
                + c.w_branched * self._branched_l.float()).clamp(max=0.55)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="marble_router", robot="null"))

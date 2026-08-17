"""RelayCascadeScene — relay the tile-named cube through a sealed two-stage gravity
cascade into the bottom bin (sim_gen task `franka_handover_i356`).

Derived from bimanual/franka_handover, but STRATEGICALLY different: the seed's whole
skill is a direct arm-to-arm free-space transfer — two Frankas reach one cube and
pass it between grippers. Here there is ONE arm and NO free-space path to the goal:
the delivery bin is the floor of a fully SEALED tower (walls, roof, 6 mm internal
gaps, lever slots all thinner than the cube), so the cube physically cannot be
carried, dropped, or slid into the bin. The only route is the tower's own BUCKET
BRIGADE: drop the cube through the roof intake onto see-saw TRAY 1, press lever L1
(a rod protruding through a narrow wall slot) to tilt tray 1 and discharge the cube
down onto see-saw TRAY 2, then press lever L2 to tilt tray 2 and discharge it into
the bin. Both trays are counterweighted: they gravity-return to their rest stops
when released, so each press is a momentary, fingertip-force interaction — the
"handover" is re-interpreted as a chained mechanism-mediated relay whose stage
ORDER is topologically forced (L2 moves an empty tray unless stage 1 has already
delivered the cube onto it; the intake feeds only tray 1).

The scene also forces PERCEPTION: two cubes (red / blue) rest on the front apron;
the colour tile on the apron pedestal names the one to relay. The decoy must be
left alone (a decoy in the bin at judging time fails the episode).

Assets are fully procedural (compound-spawner pattern; children of one body never
self-collide): housing (heavy DYNAMIC compound — a kinematic root would orphan the
tray joint anchors when reset teleports it: base, side/end walls with two lever
slots, roof plate with intake mouth, internal seal wall, bin partition, front apron
with cube slots + tile pedestal), two counterweighted trays (DYNAMIC compounds on
authored limited revolute X joints: plate, retainer, ballast, lever rod; explicit
MassAPI mass + CoM + diagonal inertia so gravity return and the plant's discrete
damping are auditable), two cubes, two kinematic colour tiles. Each tray carries a
post_step torque plant (viscous damping + external `lever_drive` clamp) with a
finite-difference hinge rate — `root_ang_vel_w` is phantom under external wrenches
on this stack.

Per-episode randomization (readback-verified by smoke): housing yaw FREE (+/-180
deg) + xy jitter, WHICH colour is the target, the cube->slot permutation, cube
jitter + yaw. All discrete draws derive from torch.rand (the first torch.randint
after manual_seed is degenerate on this stack).

Rubric (0..1; latched credit anchored in the demonstrated solve trajectory):
  0.20  intake — the TARGET cube ever aboard tray 1 (latched)
  0.25  stage  — the TARGET cube ever aboard tray 2 (latched; reachable only by a
                 tray-1 discharge)
  0.30  bin    — the TARGET cube ever in the bin box (latched)
capped at 0.75; exactly 1.0 iff success(): target cube settled in the bin, no decoy
in the bin, trays at rest, everything finite. Null policy ~0. The seed's strategy —
carry the cube to the destination — earns ~0: a cube carried to the sealed tower
just rests on the roof plate.

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


def _author_mass(root, mass: float, com, inertia) -> None:
    """Explicit MassAPI mass + CoM + diagonal inertia. On this stack, MassAPI mass on a
    compound root leaves the CoM at the body ORIGIN and the shape-derived inertia is
    unknown — author all three so gravity return and plant stability are auditable."""
    from pxr import Gf, UsdPhysics

    api = UsdPhysics.MassAPI.Apply(root)
    api.CreateMassAttr(float(mass))
    api.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))


def _mk_material(prim_path: str, name: str, mu_s: float, mu_d: float, combine: str) -> str:
    import isaaclab.sim as sim_utils

    mat_path = f"{prim_path}/{name}"
    sim_utils.spawn_rigid_body_material(mat_path, sim_utils.RigidBodyMaterialCfg(
        static_friction=float(mu_s), dynamic_friction=float(mu_d), restitution=0.0,
        friction_combine_mode=combine))
    return mat_path


def _spawn_housing(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the sealed relay tower at `prim_path`: heavy DYNAMIC compound. Local
    frame: origin on the ground under the tower centre; +x is the FRONT (robot /
    apron / lever side); the cascade runs along y (intake high at +y, bin floor at
    +y, sump dead-pocket at -y). The +x side wall carries the two lever slots."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    xo = c.in_x + c.wall_t          # outer x half-extent
    ylo, yhi = c.in_y0 - c.wall_t, c.in_y1 + c.wall_t

    # --- base plate + end walls + solid -x side wall ----------------------------------
    _span(stage, f"{prim_path}/base", x=(-xo, xo), y=(ylo, yhi), z=(0.0, c.base_t),
          color=c.shell_color, collide=collide)
    _span(stage, f"{prim_path}/end_n", x=(-xo, xo), y=(ylo, c.in_y0),
          z=(0.0, c.top_z1), color=c.shell_color, collide=collide)
    _span(stage, f"{prim_path}/end_p", x=(-xo, xo), y=(c.in_y1, yhi),
          z=(0.0, c.top_z1), color=c.shell_color, collide=collide)
    _span(stage, f"{prim_path}/side_n", x=(-xo, -c.in_x), y=(c.in_y0, c.in_y1),
          z=(0.0, c.top_z1), color=c.side_color, collide=collide)

    # --- +x side wall: 5 z-bands leaving the two lever slots --------------------------
    xs = (c.in_x, xo)
    bands = [
        ("f0", (0.0, c.slot2_z0), None),
        ("f1", (c.slot2_z0, c.slot2_z1), (c.slot2_y0, c.slot2_y1)),
        ("f2", (c.slot2_z1, c.slot1_z0), None),
        ("f3", (c.slot1_z0, c.slot1_z1), (c.slot1_y0, c.slot1_y1)),
        ("f4", (c.slot1_z1, c.top_z1), None),
    ]
    for tag, (z0, z1), hole in bands:
        if hole is None:
            _span(stage, f"{prim_path}/side_p_{tag}", x=xs, y=(c.in_y0, c.in_y1),
                  z=(z0, z1), color=c.side_color, collide=collide)
        else:
            _span(stage, f"{prim_path}/side_p_{tag}a", x=xs, y=(c.in_y0, hole[0]),
                  z=(z0, z1), color=c.side_color, collide=collide)
            _span(stage, f"{prim_path}/side_p_{tag}b", x=xs, y=(hole[1], c.in_y1),
                  z=(z0, z1), color=c.side_color, collide=collide)

    # --- roof plate with the intake mouth (full interior x width) ---------------------
    _span(stage, f"{prim_path}/roof_n", x=(-xo, xo), y=(c.in_y0, c.mouth_y0),
          z=(c.top_z0, c.top_z1), color=c.roof_color, collide=collide)
    _span(stage, f"{prim_path}/roof_p", x=(-xo, xo), y=(c.mouth_y1, c.in_y1),
          z=(c.top_z0, c.top_z1), color=c.roof_color, collide=collide)

    # --- internal seal wall A (blocks the tray1 +y edge channel down to the bin) ------
    _span(stage, f"{prim_path}/wall_a", x=(-c.in_x, c.in_x), y=(c.wa_y0, c.wa_y1),
          z=(c.wa_z0, c.top_z0), color=c.side_color, collide=collide)

    # --- bin partition (bin y > part_y1; sump dead-pocket y < part_y0) ----------------
    _span(stage, f"{prim_path}/partition", x=(-c.in_x, c.in_x), y=(c.part_y0, c.part_y1),
          z=(c.base_t, c.part_z1), color=c.part_color, collide=collide)

    # --- front apron: cube slots platform + tile pedestal -----------------------------
    _span(stage, f"{prim_path}/apron", x=(xo, c.ap_x1), y=(-c.ap_y, c.ap_y),
          z=(0.0, c.ap_z1), color=c.apron_color, collide=collide)
    px, py = c.ped_center
    P = c.ped_half
    _span(stage, f"{prim_path}/pedestal", x=(px - P, px + P), y=(py - P, py + P),
          z=(c.ap_z1, c.ap_z1 + c.ped_h), color=c.ped_color, collide=collide)

    # --- dynamic root (NOT kinematic: tray joint anchors must follow teleports) ------
    UsdPhysics.RigidBodyAPI.Apply(root)
    _author_mass(root, c.housing_mass, (0.02, 0.02, 0.10), c.housing_inertia)
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(1)
    prb.CreateLinearDampingAttr(0.5)
    prb.CreateAngularDampingAttr(2.0)
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)
    base = _mk_material(prim_path, "shell_mat", c.base_mu_s, c.base_mu_d, "average")
    bind_physics_material(prim_path, base)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one see-saw tray at `prim_path`: DYNAMIC compound. Local frame: origin
    at the tray's PIVOT (hinge axis = local x). Children: plate (y span py0..py1),
    retainer wall (ry0..ry1, up rz0..rz1), ballast block (by0..by1, under-plate
    bz0..bz1), lever rod through the wall slot (rod_x0..rod_x1 at rod_y)."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    W = c.half_w
    _span(stage, f"{prim_path}/plate", x=(-W, W), y=(c.py0, c.py1),
          z=(-c.plate_t / 2, c.plate_t / 2), color=c.plate_color, collide=collide)
    _span(stage, f"{prim_path}/retainer", x=(-W, W), y=(c.ry0, c.ry1),
          z=(c.plate_t / 2, c.rz1), color=c.ret_color, collide=collide)
    _span(stage, f"{prim_path}/ballast", x=(-0.030, 0.030), y=(c.by0, c.by1),
          z=(c.bz0, -c.plate_t / 2), color=c.bal_color, collide=collide)
    _span(stage, f"{prim_path}/rod", x=(c.rod_x0, c.rod_x1),
          y=(c.rod_y - c.rod_h, c.rod_y + c.rod_h),
          z=(-c.rod_h, c.rod_h), color=c.rod_color, collide=collide)

    UsdPhysics.RigidBodyAPI.Apply(root)
    _author_mass(root, c.mass, c.com, c.inertia)
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(1)
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)
    slick = _mk_material(prim_path, "deck", c.mu_s, c.mu_d, "average")
    bind_physics_material(prim_path, slick)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "housing" not in _SPAWNER_CACHE:

        @configclass
        class HousingSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_housing)
            in_x: float = 0.060
            in_y0: float = -0.19
            in_y1: float = 0.28
            wall_t: float = 0.012
            base_t: float = 0.012
            top_z0: float = 0.44
            top_z1: float = 0.452
            mouth_y0: float = 0.055
            mouth_y1: float = 0.155
            wa_y0: float = 0.155
            wa_y1: float = 0.167
            wa_z0: float = 0.20
            part_y0: float = 0.005
            part_y1: float = 0.015
            part_z1: float = 0.10
            slot1_y0: float = -0.033
            slot1_y1: float = 0.003
            slot1_z0: float = 0.311
            slot1_z1: float = 0.384
            slot2_y0: float = 0.014
            slot2_y1: float = 0.052
            slot2_z0: float = 0.166
            slot2_z1: float = 0.244
            ap_x1: float = 0.29
            ap_y: float = 0.10
            ap_z1: float = 0.10
            ped_center: tuple = (0.245, 0.0)
            ped_half: float = 0.030
            ped_h: float = 0.060
            housing_mass: float = 40.0
            housing_inertia: tuple = (2.5, 2.5, 3.0)
            shell_color: tuple = (0.42, 0.40, 0.36)
            side_color: tuple = (0.35, 0.38, 0.45)
            roof_color: tuple = (0.30, 0.33, 0.40)
            part_color: tuple = (0.25, 0.27, 0.30)
            apron_color: tuple = (0.50, 0.48, 0.44)
            ped_color: tuple = (0.30, 0.30, 0.32)
            contact_offset: float = 0.0015
            base_mu_s: float = 0.50
            base_mu_d: float = 0.45

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            half_w: float = 0.056
            plate_t: float = 0.010
            py0: float = -0.08
            py1: float = 0.10
            ry0: float = 0.088
            ry1: float = 0.100
            rz1: float = 0.027
            by0: float = 0.040
            by1: float = 0.080
            bz0: float = -0.035
            rod_y: float = -0.07
            rod_h: float = 0.008
            rod_x0: float = 0.045
            rod_x1: float = 0.105
            mass: float = 0.50
            com: tuple = (0.0, 0.045, -0.010)
            inertia: tuple = (0.0045, 0.0045, 0.0045)
            plate_color: tuple = (0.60, 0.60, 0.64)
            ret_color: tuple = (0.20, 0.22, 0.26)
            bal_color: tuple = (0.15, 0.15, 0.17)
            rod_color: tuple = (0.85, 0.75, 0.15)
            contact_offset: float = 0.0015
            mu_s: float = 0.32
            mu_d: float = 0.28

        _SPAWNER_CACHE["housing"] = HousingSpawnerCfg
        _SPAWNER_CACHE["tray"] = TraySpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class RelayCascadeSceneCfg(BaseCfg):
    """Config for `RelayCascadeScene`. The seal + cascade contract is asserted in
    `__post_init__`: every gap/slot in the shell is thinner than the cube; each
    tray's sweep clears the shell, the other tray and the partition by explicit
    margins (joint-pair collision filtering is not trusted either way); ballast
    gravity-return dominates the worst cargo torque; lever press forces stay
    fingertip-scale; the tray plants are discretely stable; the tray-2 discharge
    (quasi-static AND ballistic) lands inside the bin box."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    settle_lin: float = tunable(0.05)      # max cube/housing |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.15)    # max |FD tray rate| when judging (rad/s)
    bin_z_max: float = tunable(0.09)       # cube-centre max height inside the bin (m)
    box_x_tol: float = tunable(0.050)      # |x| half-width of all stage boxes (m)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    yaw_deg: float = tunable(180.0)        # housing yaw uniform +/- (FREE heading)
    xy_jitter: float = tunable(0.05)       # housing xy jitter (+/- m)
    randomize_target: bool = tunable(True)  # sample WHICH colour is the target
    randomize_slots: bool = tunable(True)   # permute cube -> apron slot
    cube_jitter: float = tunable(0.010)    # cube xy jitter in its slot (+/- m)

    # --- tunable: lever drive plants -------------------------------------------------------------
    # Discrete-stability audit (explicit external torques at 120 Hz, authored Ix=0.0045):
    # b*dt/I = 0.02/(120*0.0045) ~ 0.037 << 1; tau_max/I*dt = 0.5/0.0045/120 ~ 0.93 rad/s
    # per-step worst kick < 1. post_step wrenches act one substep late, so solve gains
    # stay well under K*dt/I = 1 (the solve uses bang-bang clamped torque + joint
    # limits as stops, no outer loop needed).
    tau_max: float = tunable(0.50)         # |lever_drive| clamp per tray (N*m)
    lever_b: float = tunable(0.020)        # viscous plant damping (N*m*s/rad)
    rate_clamp: float = tunable(20.0)      # FD rate clamp (rad/s) — teleport transients

    # --- info: housing (local frame: origin on the ground under the tower, +x = front) -----------
    base_z: float = info(0.001)            # root spawn height (1 mm, settles onto ground)
    in_x: float = info(0.060)              # interior x half-width
    in_y0: float = info(-0.19)             # interior y span
    in_y1: float = info(0.28)
    wall_t: float = info(0.012)
    base_t: float = info(0.012)
    top_z0: float = info(0.44)             # roof plate underside
    top_z1: float = info(0.452)
    mouth_y0: float = info(0.055)          # intake mouth (hole in the roof, full int. x)
    mouth_y1: float = info(0.155)
    wa_y0: float = info(0.155)             # internal seal wall A (z wa_z0..top_z0)
    wa_y1: float = info(0.167)
    wa_z0: float = info(0.20)
    part_y0: float = info(0.005)           # bin partition (bin y > part_y1)
    part_y1: float = info(0.015)
    part_z1: float = info(0.10)
    slot1_y0: float = info(-0.033)         # lever slots in the +x wall
    slot1_y1: float = info(0.003)
    slot1_z0: float = info(0.311)
    slot1_z1: float = info(0.384)
    slot2_y0: float = info(0.014)
    slot2_y1: float = info(0.052)
    slot2_z0: float = info(0.166)
    slot2_z1: float = info(0.244)
    ap_x1: float = info(0.29)              # apron platform (x from the outer wall face)
    ap_y: float = info(0.10)
    ap_z1: float = info(0.10)
    ped_center: tuple = info((0.245, 0.0))
    ped_half: float = info(0.030)
    ped_h: float = info(0.060)
    housing_mass: float = info(40.0)       # heavy DYNAMIC fixture: the joint anchors must
    housing_inertia: tuple = info((2.5, 2.5, 3.0))  # follow reset teleports
    # --- info: trays (local frame at each pivot; hinge = local x) --------------------------------
    t1_pivot: tuple = info((0.05, 0.36))   # (y, z) in the housing frame
    t2_pivot: tuple = info((-0.03, 0.22))
    t1_lim: tuple = info((-8.0, 30.0))     # joint limits (deg); rest at the LOWER/UPPER
    t2_lim: tuple = info((-35.0, 8.0))     #   stop resp. (ballast-held)
    t1_rest: float = info(-8.0)            # gravity rest angle (deg) = a joint stop
    t2_rest: float = info(8.0)
    tray_half_w: float = info(0.056)
    plate_t: float = info(0.010)
    t1_py: tuple = info((-0.08, 0.10))     # plate y span (local)
    t2_py: tuple = info((-0.12, 0.09))
    t1_ry: tuple = info((0.088, 0.100))    # retainer y span
    t2_ry: tuple = info((-0.120, -0.108))
    t1_rz1: float = info(0.027)            # retainer top (local z)
    t2_rz1: float = info(0.060)
    t1_by: tuple = info((0.040, 0.080))    # ballast y span (under-plate)
    t2_by: tuple = info((-0.115, -0.075))
    bz0: float = info(-0.035)
    t1_rod_y: float = info(-0.07)          # lever rod centre (local y)
    t2_rod_y: float = info(0.07)
    rod_h: float = info(0.008)
    rod_x1: float = info(0.105)
    t1_mass: float = info(0.50)
    t2_mass: float = info(0.45)
    t1_com: tuple = info((0.0, 0.045, -0.010))
    t2_com: tuple = info((0.0, -0.035, -0.010))
    tray_inertia: tuple = info((0.0045, 0.0045, 0.0045))
    tray_mu_s: float = info(0.32)          # deck friction: parks at 8 deg, slides at 30+
    tray_mu_d: float = info(0.28)
    # --- info: cubes / tiles / slots -------------------------------------------------------------
    cube_size: float = info(0.050)
    cube_mass: float = info(0.10)
    cube_mu_s: float = info(0.50)
    cube_mu_d: float = info(0.45)
    cube_colors: tuple = info(((0.85, 0.10, 0.10), (0.12, 0.30, 0.85)))
    color_names: tuple = info(("red", "blue"))
    tile_side: float = info(0.070)
    tile_t: float = info(0.010)
    slot_xy: tuple = info(((0.150, 0.060), (0.150, -0.060)))  # on the apron (housing frame)
    park_xy: tuple = info(((1.7, 0.9), (1.7, -0.9)))  # off-field tile parking (world)
    # --- info: materials / rubric ----------------------------------------------------------------
    contact_offset: float = info(0.0015)
    base_mu_s: float = info(0.50)
    base_mu_d: float = info(0.45)
    w_intake: float = info(0.20)
    w_stage: float = info(0.25)
    w_bin: float = info(0.30)
    score_cap: float = info(0.75)
    # stage boxes (housing frame): intake/tray1, tray2, bin (see _stage_flags)
    box1_y: tuple = info((0.00, 0.16))
    box1_z: tuple = info((0.33, 0.44))
    box2_y: tuple = info((-0.17, 0.05))
    box2_z: tuple = info((0.19, 0.30))
    bin_y: tuple = info((0.030, 0.265))

    def __post_init__(self) -> None:
        c = self
        s = c.cube_size

        def rot(y, z, phi):
            return (y * math.cos(phi) - z * math.sin(phi),
                    y * math.sin(phi) + z * math.cos(phi))

        # --- seal: every gap/slot in or into the shell is thinner than the cube ------------------
        assert (c.slot1_y1 - c.slot1_y0) < s - 0.008, "lever slot 1 must reject the cube"
        assert (c.slot2_y1 - c.slot2_y0) < s - 0.008, "lever slot 2 must reject the cube"
        assert c.in_x - c.tray_half_w < s / 2, "tray side gaps must reject the cube"
        # tray1 +y edge to wall A: a thin vertical channel the cube cannot enter
        t1y, t1z = c.t1_pivot
        gap_a = c.wa_y0 - (t1y + c.t1_py[1])  # max reach at phi=0
        assert 0.002 < gap_a < 0.010, "tray1/wall-A gap must be a few mm"
        assert abs(c.wa_y1 - c.mouth_y1 - c.wall_t) < 0.02 or c.wa_y0 >= c.mouth_y1, \
            "wall A must start at the intake mouth far edge"
        # tray2 -y edge to the end wall: gap thinner than the cube
        t2y, t2z = c.t2_pivot
        gap_n = (t2y + c.t2_py[0] * math.cos(math.radians(c.t2_rest))) - c.in_y0
        assert gap_n < s - 0.008, "tray2/end-wall gap must reject the cube"
        # intake mouth passes the cube and feeds ONLY tray1
        assert c.mouth_y1 - c.mouth_y0 >= s + 0.045, "mouth must pass the cube with slack"
        assert 2 * c.in_x >= s + 0.045
        assert c.mouth_y0 >= t1y - 0.06 and c.mouth_y1 <= c.wa_y0 + 0.001, \
            "mouth must sit over tray1"

        # --- tray sweeps: clearances at the extreme joint angles ---------------------------------
        lo1, hi1 = (math.radians(a) for a in c.t1_lim)
        lo2, hi2 = (math.radians(a) for a in c.t2_lim)
        # tray1 retainer top corner stays below the roof at full press
        _, rz = rot(c.t1_ry[1], c.t1_rz1, hi1)
        assert t1z + rz < c.top_z0 - 0.005, "tray1 retainer must clear the roof"
        _, ez = rot(c.t1_py[1], c.plate_t / 2, hi1)
        assert t1z + ez < c.top_z0 - 0.005, "tray1 plate edge must clear the roof"
        # tray1 lowest sweep point vs tray2 highest sweep point (y-separated anyway)
        t1_low = t1z + min(rot(c.t1_py[0], -c.plate_t / 2, a)[1] for a in (lo1, hi1))
        t2_high = t2z + max(rot(c.t2_ry[0], c.t2_rz1, a)[1] for a in (lo2, hi2))
        t2_high_y = t2y + rot(c.t2_ry[0], c.t2_rz1, lo2)[0]
        t1_min_y = t1y + min(c.t1_py[0] * math.cos(a) for a in (lo1, hi1, 0.0))
        assert (t2_high < t1_low - 0.005) or (t2_high_y < t1_min_y - 0.010), \
            "tray sweeps must not intersect"
        # tray2 lowest sweep point clears the partition and the bin airspace
        t2_low = t2z + min(rot(c.t2_py[1], -c.plate_t / 2, lo2)[1],
                           rot(c.t2_by[0], c.bz0, hi2)[1],
                           rot(c.t2_by[1], c.bz0, lo2)[1])
        assert t2_low > c.part_z1 + 0.005, "tray2 sweep must clear the partition"
        # lever rods stay inside their slots across the full travel (+ rod-half margin)
        for rod_y, piv, lim, sy0, sy1, sz0, sz1 in (
            (c.t1_rod_y, c.t1_pivot, (lo1, hi1), c.slot1_y0, c.slot1_y1,
             c.slot1_z0, c.slot1_z1),
            (c.t2_rod_y, c.t2_pivot, (lo2, hi2), c.slot2_y0, c.slot2_y1,
             c.slot2_z0, c.slot2_z1),
        ):
            for a in lim:
                ry, rz2 = rot(rod_y, 0.0, a)
                assert sy0 + c.rod_h < piv[0] + ry < sy1 - c.rod_h, "rod must clear slot y"
                assert sz0 + c.rod_h < piv[1] + rz2 < sz1 - c.rod_h, "rod must clear slot z"
        assert c.rod_x1 > c.in_x + c.wall_t + 0.02, "rod must protrude for pressing"

        # --- ballast: gravity return dominates the worst cargo torque ----------------------------
        g = 9.81
        # tray1 restoring torque at full press (CoM offset), worst cargo assist at the exit edge
        yc, zc = c.t1_com[1], c.t1_com[2]
        tau_bal1 = c.t1_mass * g * (yc * math.cos(hi1) - zc * math.sin(hi1))
        tau_cargo1 = c.cube_mass * g * abs(c.t1_py[0]) * math.cos(hi1)
        assert tau_bal1 > 2.5 * tau_cargo1, "tray1 ballast must dominate cargo 2.5x"
        yc2, zc2 = c.t2_com[1], c.t2_com[2]
        tau_bal2 = -c.t2_mass * g * (yc2 * math.cos(lo2) - zc2 * math.sin(lo2))
        assert tau_bal2 > 0, "tray2 ballast must gravity-return from full press"
        # press torque budget: tau_max overcomes ballast + cargo lift with >=1.5x margin
        press1 = tau_bal1 + tau_cargo1
        press2 = tau_bal2 + c.cube_mass * g * abs(c.t2_ry[0])
        assert c.tau_max > 1.3 * press1 and c.tau_max > 1.3 * press2, \
            "tau_max must overcome ballast+cargo with margin"
        # fingertip-scale lever forces
        assert c.tau_max / abs(c.t1_rod_y) < 10.0 and c.tau_max / abs(c.t2_rod_y) < 10.0

        # --- deck friction: cube parks at rest tilt, slides at full press ------------------------
        mu_s_pair = (c.tray_mu_s + c.cube_mu_s) / 2
        mu_d_pair = (c.tray_mu_d + c.cube_mu_d) / 2
        assert mu_s_pair > math.tan(math.radians(abs(c.t1_rest))) + 0.10, \
            "cube must park on the resting tray"
        assert math.tan(hi1) > mu_s_pair + 0.10 and math.tan(-lo2) > mu_s_pair + 0.10, \
            "cube must slide at full press"

        # --- tray2 discharge lands in the bin (quasi-static AND ballistic) ----------------------
        edge_y = t2y + c.t2_py[1] * math.cos(lo2)  # exit edge at full press
        edge_z = t2z + c.t2_py[1] * math.sin(lo2)
        y0 = edge_y + (s / 2) * math.cos(lo2) * 0.8  # cube-centre overhang at tip-off
        assert y0 > c.part_y1 + s / 2 + 0.002, "quasi-static drop must clear the partition"
        assert edge_z > c.part_z1 + 0.02, "exit edge must clear the partition"
        v = math.sqrt(2 * g * (math.sin(-lo2) - mu_d_pair * math.cos(lo2))
                      * (c.t2_py[1] - c.t2_py[0]))
        vy, vz = v * math.cos(lo2), v * math.sin(lo2)  # vz < 0
        zc0 = edge_z + (s / 2) * math.cos(lo2)
        # time to fall to the resting cube-centre height
        z_rest = c.base_t + s / 2
        tt = (-(-vz) + math.sqrt(vz * vz + 2 * g * (zc0 - z_rest))) / g
        y_land = y0 + vy * tt
        slide = (vy * vy) / (2 * (c.base_mu_d + c.cube_mu_d) / 2 * g)
        assert c.bin_y[0] + 0.005 < y_land < c.in_y1 - s / 2, "ballistic landing in the bin"
        assert y_land + slide < c.in_y1 + 0.05, "landing slide arrested by the end wall"
        # the flight passes UNDER wall A into the bin
        t_wa = (c.wa_y0 - y0) / vy
        z_wa = zc0 + vz * t_wa - 0.5 * g * t_wa * t_wa
        assert z_wa + s / 2 < c.wa_z0 - 0.01, "discharge flight must pass under wall A"

        # --- stage boxes: z-disjoint; bin box inside the interior --------------------------------
        assert c.box2_z[1] < c.box1_z[0] and c.bin_z_max < c.box2_z[0]
        assert c.bin_y[0] >= c.part_y1 + 0.010 and c.bin_y[1] <= c.in_y1 - 0.010
        assert c.box_x_tol < c.in_x

        # --- apron: slots + pedestal reachable, on the platform ----------------------------------
        xo = c.in_x + c.wall_t
        for sx, sy in c.slot_xy:
            assert xo + s / 2 + c.cube_jitter < sx < c.ap_x1 - s / 2 - c.cube_jitter
            assert abs(sy) + s / 2 + c.cube_jitter < c.ap_y
            px, py = c.ped_center
            assert math.hypot(sx - px, sy - py) > c.ped_half * math.sqrt(2) \
                + s * math.sqrt(2) / 2 + c.cube_jitter + 0.005
        assert c.ped_center[0] + c.ped_half < c.ap_x1

        # --- plant discrete stability ------------------------------------------------------------
        dt = 1.0 / 120.0
        ix = c.tray_inertia[0]
        assert c.lever_b * dt / ix < 0.5, "plant damping must be discretely stable"
        assert c.tau_max / ix * dt < 1.0, "per-step torque kick must stay bounded"
        assert abs(c.w_intake + c.w_stage + c.w_bin - c.score_cap) < 1e-9


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qx(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 1] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qconj(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[:, 1:] = -out[:, 1:]
    return out


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("relay_cascade")
class RelayCascadeScene(BaseScene):
    cfg: RelayCascadeSceneCfg

    def __init__(self, cfg: RelayCascadeSceneCfg | None = None) -> None:
        super().__init__(cfg or RelayCascadeSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        housing_spawn = cls["housing"](contact_offset=c.contact_offset)
        tray1_spawn = cls["tray"](
            py0=c.t1_py[0], py1=c.t1_py[1], ry0=c.t1_ry[0], ry1=c.t1_ry[1],
            rz1=c.t1_rz1, by0=c.t1_by[0], by1=c.t1_by[1], bz0=c.bz0,
            rod_y=c.t1_rod_y, rod_x1=c.rod_x1, mass=c.t1_mass, com=c.t1_com,
            inertia=c.tray_inertia, contact_offset=c.contact_offset,
            mu_s=c.tray_mu_s, mu_d=c.tray_mu_d)
        tray2_spawn = cls["tray"](
            py0=c.t2_py[0], py1=c.t2_py[1], ry0=c.t2_ry[0], ry1=c.t2_ry[1],
            rz1=c.t2_rz1, by0=c.t2_by[0], by1=c.t2_by[1], bz0=c.bz0,
            rod_y=c.t2_rod_y, rod_x1=c.rod_x1, mass=c.t2_mass, com=c.t2_com,
            inertia=c.tray_inertia, contact_offset=c.contact_offset,
            mu_s=c.tray_mu_s, mu_d=c.tray_mu_d,
            plate_color=(0.55, 0.58, 0.55))

        rigid = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.20, angular_damping=0.20,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=32, solver_velocity_iteration_count=1)
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "housing": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Housing", spawn=housing_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.base_z))),
            "tray1": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray1", spawn=tray1_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, c.t1_pivot[0], c.t1_pivot[1] + c.base_z))),
            "tray2": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray2", spawn=tray2_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, c.t2_pivot[0], c.t2_pivot[1] + c.base_z))),
        }
        for k in range(2):
            out[f"cube_{k}"] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Cube_{k}",
                spawn=sim_utils.CuboidCfg(
                    size=(c.cube_size, c.cube_size, c.cube_size),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
                    rigid_props=rigid, collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.cube_mu_s, dynamic_friction=c.cube_mu_d,
                        restitution=0.0, friction_combine_mode="average"),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.cube_colors[k])),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2 + 0.2 * k, 0.8, 0.05)))
            out[f"tile_{k}"] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Tile_{k}",
                spawn=sim_utils.CuboidCfg(
                    size=(c.tile_side, c.tile_side, c.tile_t),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.cube_colors[k])),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2 + 0.2 * k, -0.8, 0.02)))
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                # Without this, external wrenches are under-applied across TGS
                # iterations and the lever plants stall far below equilibrium.
                "enable_external_forces_every_iteration": True,
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.housing: RigidObject = env.iscene["housing"]
        self.trays: list[RigidObject] = [env.iscene["tray1"], env.iscene["tray2"]]
        self.cubes: list[RigidObject] = [env.iscene[f"cube_{k}"] for k in range(2)]
        self.tiles: list[RigidObject] = [env.iscene[f"tile_{k}"] for k in range(2)]
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        n = env.num_envs
        dev = env.device
        # episode readbacks (verified by smoke)
        self.target_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self.slot_of = torch.zeros(n, 2, dtype=torch.long, device=dev)  # cube k -> slot
        # drive plant state (post_step OWNS the trays' external-wrench slots; solve/smoke
        # write lever_drive only — never call set_external_force_and_torque on a tray)
        self.lever_drive = torch.zeros(n, 2, device=dev)  # commanded torque (N*m, clamped)
        self.rate_fd = torch.zeros(n, 2, device=dev)      # FD hinge rates (root_ang_vel_w
        self._phi_prev = torch.zeros(n, 2, device=dev)    # is phantom under ext. wrenches)
        # latches (partial credit survives transients; success is judged live)
        self._intake = torch.zeros(n, dtype=torch.bool, device=dev)
        self._stage = torch.zeros(n, dtype=torch.bool, device=dev)
        self._bin = torch.zeros(n, dtype=torch.bool, device=dev)

    def _author_joints(self) -> None:
        """Per env: LIMITED revolute X joints housing->tray at each pivot. Body0 is the
        heavy DYNAMIC housing root so the anchors follow reset teleports (a kinematic
        body0 anchor stays world-fixed at the spawn pose on this stack). The joint
        limits ARE the tray stops (rest + full press); joint-pair collision keeps its
        default (filtered) — no tray-housing contact is ever load-bearing."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            for name, piv, lim in (("Tray1", c.t1_pivot, c.t1_lim),
                                   ("Tray2", c.t2_pivot, c.t2_lim)):
                j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/{name.lower()}_pivot")
                j.CreateBody0Rel().SetTargets([f"{base}/Housing"])
                j.CreateBody1Rel().SetTargets([f"{base}/{name}"])
                j.CreateAxisAttr("X")
                j.CreateLocalPos0Attr(Gf.Vec3f(0.0, float(piv[0]), float(piv[1])))
                j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLowerLimitAttr(float(lim[0]))
                j.CreateUpperLimitAttr(float(lim[1]))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the housing (free yaw + xy jitter) and both trays
        together at their gravity-rest stop angles (whole-linkage write), permute the
        two cubes over the apron slots, sample the target colour, stand its tile on
        the pedestal (other parked off-field), sync the FD rate references, clear
        drives and latches. All discrete draws derive from torch.rand — the first
        torch.randint after manual_seed is degenerate on this stack."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        from isaaclab.utils.math import quat_apply

        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_deg)
        q_h = _qz(yaw)
        dp = torch.zeros(m, 3, device=dev)
        dp[:, 0] = (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        dp[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        dp[:, 2] = c.base_z
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = dp + origin
        st[:, 3:7] = q_h
        self.housing.write_root_state_to_sim(st, env_ids)

        # trays: pivots in the housing frame, rest ANGLE = the ballast-held stop
        # (write the whole linkage together — teleporting one body of a joint pair
        # gets depenetrated back by the other)
        rests = (c.t1_rest, c.t2_rest)
        for k, tray in enumerate(self.trays):
            piv = (c.t1_pivot, c.t2_pivot)[k]
            loc = torch.tensor([0.0, piv[0], piv[1]], device=dev).expand(m, 3)
            phi = torch.full((m,), math.radians(rests[k]), device=dev)
            s = torch.zeros(m, 13, device=dev)
            s[:, 0:3] = dp + origin + quat_apply(q_h, loc)
            s[:, 3:7] = _qmul(q_h, _qx(phi))
            tray.write_root_state_to_sim(s, env_ids)
            self._phi_prev[env_ids, k] = phi

        # target colour + cube->slot permutation (rand-derived, readback-verified)
        tgt = (torch.rand(m, device=dev) * 2).long().clamp(max=1)
        if not c.randomize_target:
            tgt = torch.zeros(m, dtype=torch.long, device=dev)
        self.target_idx[env_ids] = tgt
        if c.randomize_slots:
            perm = torch.argsort(torch.rand(m, 2, device=dev), dim=1)
        else:
            perm = torch.arange(2, device=dev).expand(m, 2).contiguous()
        self.slot_of[env_ids] = perm

        slots = torch.tensor(c.slot_xy, device=dev)
        for k in range(2):
            sxy = slots[perm[:, k]]
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = sxy[:, 0] + (torch.rand(m, device=dev) * 2 - 1) * c.cube_jitter
            loc[:, 1] = sxy[:, 1] + (torch.rand(m, device=dev) * 2 - 1) * c.cube_jitter
            loc[:, 2] = c.ap_z1 + c.cube_size / 2 + 0.003
            yaw_c = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            s = torch.zeros(m, 13, device=dev)
            s[:, 0:3] = dp + origin + quat_apply(q_h, loc)
            s[:, 3:7] = _qmul(q_h, _qz(yaw_c))
            self.cubes[k].write_root_state_to_sim(s, env_ids)

        # tiles: the target's tile stands on the pedestal; the other parks off-field
        px, py = c.ped_center
        ped = torch.tensor([px, py, c.ap_z1 + c.ped_h + c.tile_t / 2 + 0.001], device=dev)
        for k in range(2):
            s = torch.zeros(m, 13, device=dev)
            on_ped = tgt == k
            loc = ped.expand(m, 3)
            s[:, 0:3] = dp + origin + quat_apply(q_h, loc)
            s[:, 3:7] = q_h
            park_k = c.park_xy[k]
            park = torch.tensor([park_k[0], park_k[1], c.tile_t / 2 + 0.002], device=dev)
            s[:, 0:3] = torch.where(on_ped.unsqueeze(-1), s[:, 0:3], origin + park)
            s[:, 3] = torch.where(on_ped, s[:, 3], torch.ones(m, device=dev))
            s[:, 4:7] = torch.where(on_ped.unsqueeze(-1), s[:, 4:7],
                                    torch.zeros(m, 3, device=dev))
            self.tiles[k].write_root_state_to_sim(s, env_ids)

        self.lever_drive[env_ids] = 0.0
        self.rate_fd[env_ids] = 0.0
        self._intake[env_ids] = False
        self._stage[env_ids] = False
        self._bin[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {
            "housing": self.housing.data.root_state_w[env_ids].clone(),
            "target_idx": self.target_idx[env_ids].clone(),
            "slot_of": self.slot_of[env_ids].clone(),
            "drive": self.lever_drive[env_ids].clone(),
            "rate_fd": self.rate_fd[env_ids].clone(),
            "phi_prev": self._phi_prev[env_ids].clone(),
            "intake": self._intake[env_ids].clone(),
            "stage": self._stage[env_ids].clone(),
            "bin": self._bin[env_ids].clone(),
        }
        for k in range(2):
            out[f"tray_{k}"] = self.trays[k].data.root_state_w[env_ids].clone()
            out[f"cube_{k}"] = self.cubes[k].data.root_state_w[env_ids].clone()
            out[f"tile_{k}"] = self.tiles[k].data.root_state_w[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.housing.write_root_state_to_sim(state["housing"], env_ids)
        for k in range(2):
            self.trays[k].write_root_state_to_sim(state[f"tray_{k}"], env_ids)
            self.cubes[k].write_root_state_to_sim(state[f"cube_{k}"], env_ids)
            self.tiles[k].write_root_state_to_sim(state[f"tile_{k}"], env_ids)
        self.target_idx[env_ids] = state["target_idx"]
        self.slot_of[env_ids] = state["slot_of"]
        self.lever_drive[env_ids] = state["drive"]
        self.rate_fd[env_ids] = state["rate_fd"]
        self._phi_prev[env_ids] = state["phi_prev"]
        self._intake[env_ids] = state["intake"]
        self._stage[env_ids] = state["stage"]
        self._bin[env_ids] = state["bin"]

    def resync_rate(self, env_ids: torch.Tensor | None = None) -> None:
        """Re-anchor the FD rate references to the CURRENT tray angles (call after
        any manual tray teleport, once the sim buffers reflect it)."""
        ph = self.phi()
        if env_ids is None:
            self._phi_prev[:] = ph
            self.rate_fd[:] = 0.0
        else:
            self._phi_prev[env_ids] = ph[env_ids]
            self.rate_fd[env_ids] = 0.0

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A SEALED RELAY TOWER stands on the ground: a tall narrow cabinet whose "
            f"bottom compartment — the DELIVERY BIN, behind a low internal partition — is "
            f"completely enclosed by walls and a roof. Nothing can be carried, dropped, or "
            f"slid into the bin from outside: every gap and slot in the shell is thinner "
            f"than the {1000 * c.cube_size:.0f} mm cubes. The only opening is the INTAKE "
            f"MOUTH in the roof, which feeds the upper of two counterweighted SEE-SAW "
            f"TRAYS inside. Each tray has a yellow LEVER ROD protruding through a narrow "
            f"slot in the front wall: pressing lever L1 (the upper rod) down tilts tray 1 "
            f"and slides its load down onto tray 2; pressing lever L2 (the lower rod) "
            f"down tilts tray 2 and slides its load out over the partition into the bin. "
            f"Released levers gravity-return: the ballast under each tray restores it to "
            f"its resting tilt, which cradles the load against a retainer wall.\n"
            f"Two cubes — one red, one blue — rest on the front apron platform. The "
            f"COLOUR TILE standing on the apron pedestal names the ONE cube to deliver "
            f"(which colour, and which cube sits in which slot, varies by episode; the "
            f"tower's heading varies too). Goal: drop the TILE-COLOURED cube through the "
            f"roof intake onto tray 1, then press lever L1, then lever L2, relaying the "
            f"cube stage by stage into the sealed bin. Leave the other cube alone: a "
            f"wrong cube in the bin at the end fails the episode. The stage order is "
            f"physically forced — the intake feeds only tray 1, and pressing L2 early "
            f"just tips an empty tray."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Read the colour tile on the apron pedestal, drop the matching cube through "
            "the roof intake onto the upper tray, then press the upper lever rod down "
            "and release it, then press the lower lever rod down and release it, so the "
            "cube cascades stage by stage into the sealed bottom bin."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def phi(self) -> torch.Tensor:
        """(N,2) tray hinge angles relative to the housing (rad, wrapped): rotation of
        the relative quaternion about local x."""
        out = []
        for tray in self.trays:
            q = _qmul(_qconj(self.housing.data.root_quat_w), tray.data.root_quat_w)
            w, x, y, z = q.unbind(-1)
            out.append(torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        return torch.stack(out, dim=1)

    def _housing_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.housing.data.root_quat_w,
                                  pos_w - self.housing.data.root_pos_w)

    def _boxes(self, pos_w: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """(N,)x3 bools: a cube-centre world position inside the tray-1 / tray-2 /
        bin stage boxes (housing frame; z-disjoint bands)."""
        c = self.cfg
        loc = self._housing_local(pos_w)
        xok = loc[:, 0].abs() < c.box_x_tol
        b1 = xok & (loc[:, 1] > c.box1_y[0]) & (loc[:, 1] < c.box1_y[1]) \
            & (loc[:, 2] > c.box1_z[0]) & (loc[:, 2] < c.box1_z[1])
        b2 = xok & (loc[:, 1] > c.box2_y[0]) & (loc[:, 1] < c.box2_y[1]) \
            & (loc[:, 2] > c.box2_z[0]) & (loc[:, 2] < c.box2_z[1])
        bb = xok & (loc[:, 1] > c.bin_y[0]) & (loc[:, 1] < c.bin_y[1]) \
            & (loc[:, 2] < c.bin_z_max)
        return b1, b2, bb

    def in_bin(self, pos_w: torch.Tensor) -> torch.Tensor:
        return self._boxes(pos_w)[2]

    def _cube_flags(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(N,) target-in-bin and (N,) decoy-in-bin."""
        inb = torch.stack([self.in_bin(cb.data.root_pos_w) for cb in self.cubes], dim=1)
        onehot = torch.nn.functional.one_hot(self.target_idx, 2).bool()
        return (inb & onehot).any(dim=1), (inb & ~onehot).any(dim=1)

    def settled(self) -> torch.Tensor:
        """(N,) bool: cubes + housing slow, tray FD rates slow."""
        c = self.cfg
        ok = self.housing.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        for cb in self.cubes:
            ok = ok & (cb.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
        return ok & (self.rate_fd.abs() < c.settle_omega).all(dim=1)

    def _finite(self) -> torch.Tensor:
        ps = [self.housing.data.root_pos_w] + [t.data.root_pos_w for t in self.trays] \
            + [cb.data.root_pos_w for cb in self.cubes]
        return torch.isfinite(torch.stack(ps, dim=1)).all(dim=-1).all(dim=-1)

    # ----- step-coupled mechanics (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Tray plants: clamped external drive torque + viscous damping on the FD hinge
        rates, applied as BODY-frame x torques (the hinge axis is each tray's local x,
        so the body frame is immune to the stack's wrench frame drag); then latch
        rubric credit. Owns the trays' external-wrench slots."""
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        dt = self.env.dt

        ph = self.phi()
        fin_ph = torch.isfinite(ph)
        raw = _wrap(ph - self._phi_prev) / dt
        self.rate_fd = torch.where(
            fin_ph, raw.clamp(-c.rate_clamp, c.rate_clamp), torch.zeros_like(raw))
        self._phi_prev = torch.where(fin_ph, ph, self._phi_prev)

        tau = self.lever_drive.clamp(-c.tau_max, c.tau_max) - c.lever_b * self.rate_fd
        tau = torch.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)
        zero = torch.zeros(n, 1, 3, device=dev)
        for k, tray in enumerate(self.trays):
            tq = torch.zeros(n, 1, 3, device=dev)
            tq[:, 0, 0] = tau[:, k]
            tray.set_external_force_and_torque(zero, tq)

        self._update_latches()

    def _update_latches(self) -> None:
        fin = self._finite()
        tgt = self.target_idx
        pos = torch.stack([cb.data.root_pos_w for cb in self.cubes], dim=1)
        tpos = pos.gather(1, tgt.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
        b1, b2, bb = self._boxes(tpos)
        self._intake |= b1 & fin
        self._stage |= b2 & fin
        self._bin |= bb & fin

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the TARGET cube inside the sealed bin, no decoy in the bin,
        trays at rest, everything settled and finite — all live physical outcomes.
        The cube got there only by riding the two-stage cascade: the sealed shell
        physically excludes any direct placement (smoke proves this with settled
        constructs)."""
        self._update_latches()
        tgt_in, dec_in = self._cube_flags()
        return tgt_in & ~dec_in & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.20 target-ever-on-tray-1 + 0.25 target-ever-on-
        tray-2 + 0.30 target-ever-in-bin (all latched), capped at 0.75; exactly 1.0
        iff success() holds live. Doing nothing scores ~0; pressing levers with no
        cube aboard scores ~0; the seed's strategy (carry the cube to the goal)
        leaves it on the roof plate and scores ~0."""
        c = self.cfg
        self._update_latches()
        base = (c.w_intake * self._intake.float() + c.w_stage * self._stage.float()
                + c.w_bin * self._bin.float()).clamp(max=c.score_cap)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="relay_cascade", robot="null"))

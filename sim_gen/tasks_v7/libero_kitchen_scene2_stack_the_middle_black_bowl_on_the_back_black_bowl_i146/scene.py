"""CounterweightVaultScene — weight the pan to hold the gravity gate open, then deliver
the sugar cube through the opened vault mouth.

Derived from libero_90 kitchen_scene2 "stack the middle black bowl on the back black bowl".
The seed is ONE unconstrained grasp-carry-place: pick a black bowl, set it down on another
black bowl, success = a single xy/z proximity relation between two bowl origins. Here the
black bowl is never a payload to be stacked — it is a COUNTERWEIGHT, a tool whose WEIGHT
(not its position relative to another bowl) is what the task consumes:

  * the centre of the scene is a self-closing GRAVITY GATE: a rigid rotor on a revolute
    joint whose axis is tilted 6 deg from vertical. A wide steel flap on the long arm of
    the rotor parks over the mouth of a walled VAULT; a brass basket (the pan) rides the
    short arm. Because the axis is tilted, every point of the rotor's orbit has a height:
    the flap rests at the LOW point of its orbit (over the vault) and anything that swings
    it away must fight its way uphill. Empty, the rotor always falls back to closed;
  * setting the BLACK BOWL (0.32-0.48 kg, re-randomized every episode) into the pan makes
    the pan side heavier than the flap side over the whole travel, so the gate swings
    ~100 deg open BY GRAVITY and — the crucial part — STAYS open only while the bowl stays
    in the pan. The bowl is placed once and never "stacked on" anything: it ends up
    hanging off the rotor, nowhere near the other bowl;
  * only through the opened mouth can the red sugar cube be dropped into the WHITE bowl
    that sits inside the vault. The flap covers the mouth whenever the pan is unloaded
    (verified: the largest crescent that ever opens without a real counterweight is ~3 mm,
    a 26 mm cube cannot pass), so the delivery is physically gated on the counterweight
    step — an ordered two-stage goal the seed does not have;
  * success is a HELD state, not an event: bowl seated in the pan, gate open at its stop,
    cube at rest inside the white bowl, white bowl undisturbed in its recess, everything
    settled. Remove the bowl and the gate closes — the end state is load-bearing.

Success (simultaneous, settled):
  * the black bowl rests IN the pan (pan-frame xy/z bands, upright);
  * the gate is open: rotor swing angle >= `open_min_deg` (90; the hard stop is 100);
  * the cube rests INSIDE the white bowl (bowl-frame xy/z bands, not on its rim);
  * the white bowl is still seated in the vault recess, upright;
  * every dynamic body settled (the rotor rests against its stop, held by the bowl).

Rubric (graded 0..1, latched in post_step, additive; 1.0 iff success()):
  0.00  nothing happened
  +0.20 loaded  — the black bowl has rested in the pan (any gate angle)
  +0.30 held    — bowl in pan AND gate open AND rotor at rest, simultaneously
  +0.30 delivered — cube at rest in the white bowl, white bowl home, WHILE the gate is
        open (a cube teleported into the closed vault can never latch this: the flap
        covers the mouth exactly when this clause is false)
  1.00  success() (overrides the 0.80 partial sum)

Honesty of the gates:
  * `held` cannot be teleport-faked: writing the rotor open without a counterweight
    lets physics close it again within a second (gravity return), and the latch demands
    the bowl seated in the pan at the same instant;
  * `delivered` demands gate-open at the latch instant — the same aperture condition
    physics enforces (closed flap leaves at most a ~3 mm crescent);
  * every latch carries a `latch_speed` velocity gate so a body flying through a band
    latches nothing; latched credit survives later mishaps (never evaporates).

Assets are fully procedural (no external files):
  * frame (kinematic, one body): hinge post + octagon-walled vault (inner apothem 65 mm,
    rim 100 mm above the counter) with a floor recess ring that locates the white bowl;
  * rotor (ONE dynamic compound on an authored tilted-axis revolute joint): steel flap
    disc (r 125 mm) on a 190 mm arm, brass pan basket (inner ~122 mm across, 22 mm walls)
    on a 130 mm arm at 155 deg from the flap; mass/CoM/diagonal inertia are authored
    AND re-enforced through the PhysX view at bind (a custom-spawned compound gets NO
    auto mass properties, and a CoM left on the hinge axis makes the gate a neutral
    pendulum); it spawns/resets AT its empty gravity rest so episodes start swing-free;
  * black bowl (counterweight, free): octagon-walled open bowl, 100 mm across, 60 mm tall,
    7 mm rim wall (pinchable); per-episode hidden-ish mass via set_masses + readback;
  * white bowl (delivery target, free): open bowl seated in the vault recess;
  * cube: 26 mm red sugar/candy cube.

Per-episode randomization (verified by readback in the smoke): black-bowl park side and
jitter and yaw, cube park side and jitter and yaw, black-bowl MASS U[0.32, 0.48] kg,
white-bowl yaw (+ sub-mm recess play). The gate fixture itself is FIXED: its revolute
joint is authored against the kinematic frame at build time, and PhysX keeps
kinematic-body joint anchors world-fixed — so the fixture cannot be pose-randomized
per episode (declared limitation, same for every jointed-fixture task in this corpus).

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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


# ----- custom compound spawners -------------------------------------------------------------------
# One rigid body per object: root Xform with RigidBodyAPI (+ EXPLICIT MassAPI mass, CoM and
# diagonal inertia on dynamics — custom spawners get no auto-computed mass properties and
# an authored mass alone leaves the CoM at the body origin, which would kill every gravity
# torque this mechanism lives on). Authored through `clone()` so per-env replication is
# idempotent.

_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_root(prim_path: str, translation, orientation, *, kinematic: bool,
                mass: float | None, com=(0.0, 0.0, 0.0), inertia=None,
                no_sleep: bool = False, ang_damp: float = 0.10, lin_damp: float = 0.05):
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    if mass is not None:
        mapi = UsdPhysics.MassAPI.Apply(root)
        mapi.CreateMassAttr(float(mass))
        mapi.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
        if inertia is not None:
            mapi.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
        px_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        # Cap the contact-solver pop on landings; light damping so landed bodies cross
        # the settle gate promptly instead of ringing.
        px_rb.CreateMaxDepenetrationVelocityAttr(0.5)
        px_rb.CreateLinearDampingAttr(float(lin_damp))
        px_rb.CreateAngularDampingAttr(float(ang_damp))
        if no_sleep:
            # The rotor must respond the instant a counterweight lands, even after
            # resting closed for a long time.
            px_rb.CreateSleepThresholdAttr(0.0)
            px_rb.CreateStabilizationThresholdAttr(0.0)
    return root


def _child_cyl(prim_path: str, name: str, *, radius: float, height: float,
               tx: float = 0.0, ty: float = 0.0, tz: float = 0.0,
               color: tuple, contact_offset: float | None):
    """Child cylinder; collider only when `contact_offset` is not None."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/{name}")
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(cyl.GetPrim())
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)
    return cyl


def _child_box(prim_path: str, name: str, *, tx: float, ty: float, tz: float,
               sx: float, sy: float, sz: float, yaw_deg: float = 0.0,
               color: tuple, contact_offset: float | None):
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    box = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
    box.CreateSizeAttr(1.0)
    bxf = UsdGeom.Xformable(box.GetPrim())
    bxf.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))
    if yaw_deg != 0.0:
        bxf.AddRotateZOp().Set(float(yaw_deg))
    bxf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(box.GetPrim())
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)
    return box


def _octagon_wall(prim_path: str, tag: str, *, cx: float, cy: float, z_center: float,
                  r_mid: float, thick: float, height: float, yaw0_deg: float,
                  color: tuple, contact_offset: float):
    """8 box segments forming a closed octagon wall of mid-line radius `r_mid`.
    Each segment's local x axis is radial (so its size is (thick, seg_len, height))."""
    seg_len = 2.0 * r_mid * math.tan(math.pi / 8.0) + 0.004  # slight overlap: no slits
    for k in range(8):
        ang = yaw0_deg + k * 45.0
        rad = math.radians(ang)
        _child_box(prim_path, f"{tag}{k}",
                   tx=cx + r_mid * math.cos(rad), ty=cy + r_mid * math.sin(rad),
                   tz=z_center, sx=thick, sy=seg_len, sz=height, yaw_deg=ang,
                   color=color, contact_offset=contact_offset)


def _spawn_frame(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The kinematic gate frame, ONE body: hinge post + vault (floor disc, octagon wall,
    recess ring). Root origin at the hinge axis on the counter top."""
    root = _apply_root(prim_path, translation, orientation, kinematic=True, mass=None)
    # hinge post (top 22 mm below the rotor hub: the jointed pair NEVER touches,
    # at any joint angle — see the clearance note in _spawn_rotor)
    _child_cyl(prim_path, "post", radius=cfg.post_r, height=cfg.post_h,
               tz=cfg.post_h / 2, color=cfg.steel, contact_offset=cfg.contact_offset)
    vx, vy = 0.0, -cfg.flap_arm  # vault centre: downhill of the hinge
    _child_cyl(prim_path, "vault_floor", radius=cfg.vault_wall_rmid + cfg.vault_wall_t / 2 + 0.004,
               height=cfg.vault_floor_t, tx=vx, ty=vy, tz=cfg.vault_floor_t / 2,
               color=cfg.wood, contact_offset=cfg.contact_offset)
    wall_h = cfg.vault_h - cfg.vault_floor_t
    _octagon_wall(prim_path, "vwall", cx=vx, cy=vy,
                  z_center=cfg.vault_floor_t + wall_h / 2,
                  r_mid=cfg.vault_wall_rmid, thick=cfg.vault_wall_t, height=wall_h,
                  yaw0_deg=0.0, color=cfg.wood, contact_offset=cfg.contact_offset)
    _octagon_wall(prim_path, "ring", cx=vx, cy=vy,
                  z_center=cfg.vault_floor_t + cfg.ring_h / 2,
                  r_mid=cfg.ring_rmid, thick=cfg.ring_t, height=cfg.ring_h,
                  yaw0_deg=22.5, color=cfg.wood_dark, contact_offset=cfg.contact_offset)
    return root


def _spawn_rotor(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The gate rotor, ONE dynamic body: hub sleeve, flap arm + steel flap disc at
    local -y (over the vault when the joint angle is 0), pan arm + drop rod + brass pan
    basket at `pan_local`. Root origin ON the hinge axis. Mass/CoM/inertia authored."""
    root = _apply_root(prim_path, translation, orientation, kinematic=False,
                       mass=cfg.mass_props.mass, com=cfg.com, inertia=cfg.inertia,
                       no_sleep=True, ang_damp=cfg.ang_damp, lin_damp=0.05)
    px, py, pz = cfg.pan_local  # pan FLOOR-TOP centre in rotor frame
    az = math.degrees(math.atan2(py, px))
    # hub cap above the post. GEOMETRIC clearance, not the joint collision filter:
    # the filter is unreliable for compound children (forge-verified on i99), and the
    # hub rim wobbles ~5 mm vertically as the rotor precesses about the tilted axis —
    # so the hub bottom (z -0.008, world +0.302) stays >= 17 mm above the post top
    # (world +0.280) at EVERY joint angle.
    _child_cyl(prim_path, "hub", radius=cfg.hub_r, height=0.028, tz=0.006,
               color=cfg.steel, contact_offset=cfg.contact_offset)
    # flap arm + flap disc (flap raised 3 mm above the arm plane: its low edge dips
    # up to ~14 mm while sweeping the vault rim; keeps >= 6 mm dynamic clearance while
    # the closed 10 mm rim gap stays far below the 26 mm cube)
    _child_box(prim_path, "flap_arm", tx=0.0, ty=-(cfg.hub_r + cfg.flap_arm) / 2, tz=0.0,
               sx=0.020, sy=cfg.flap_arm - cfg.hub_r + 0.02, sz=0.008,
               color=cfg.steel, contact_offset=cfg.contact_offset)
    _child_cyl(prim_path, "flap", radius=cfg.flap_r, height=cfg.flap_t,
               ty=-cfg.flap_arm, tz=0.003,
               color=cfg.steel_dark, contact_offset=cfg.contact_offset)
    # pan arm (radial at the pan azimuth) + drop rod: both STOP SHORT of the pan mouth,
    # hugging the pan's near OUTER wall face so the whole opening stays clear for the
    # dropped counterweight (v1 ran the arm over the mouth and the rod down its centre:
    # the bowl PERCHED on the rod top 42 mm above the floor and could never seat)
    wall_out = math.hypot(px, py) - (cfg.pan_wall_rmid + cfg.pan_wall_t / 2)  # 0.063
    arm_r0, arm_r1 = cfg.hub_r, wall_out - 0.005
    arm_mid = (arm_r0 + arm_r1) / 2
    _child_box(prim_path, "pan_arm",
               tx=arm_mid * math.cos(math.radians(az)), ty=arm_mid * math.sin(math.radians(az)),
               tz=-0.004, sx=arm_r1 - arm_r0 + 0.02, sy=0.020, sz=0.008,
               yaw_deg=az, color=cfg.brass, contact_offset=cfg.contact_offset)
    rod_rad = wall_out - 0.003  # 16 mm rod: radial span 0.052..0.068, embedded ~5 mm
    _child_box(prim_path, "pan_rod",  # into the wall (inner face at 0.069: never pokes in)
               tx=rod_rad * math.cos(math.radians(az)), ty=rod_rad * math.sin(math.radians(az)),
               tz=-0.015, sx=0.016, sy=0.016, sz=0.030,
               color=cfg.brass, contact_offset=cfg.contact_offset)
    _child_cyl(prim_path, "pan_floor", radius=cfg.pan_floor_r, height=cfg.pan_floor_t,
               tx=px, ty=py, tz=pz - cfg.pan_floor_t / 2,
               color=cfg.brass, contact_offset=cfg.contact_offset)
    _octagon_wall(prim_path, "pwall", cx=px, cy=py,
                  z_center=pz + cfg.pan_wall_h / 2,
                  r_mid=cfg.pan_wall_rmid, thick=cfg.pan_wall_t, height=cfg.pan_wall_h,
                  yaw0_deg=az, color=cfg.brass, contact_offset=cfg.contact_offset)
    return root


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """An open-top bowl: floor disc + octagon wall. Root origin at the bottom centre."""
    root = _apply_root(prim_path, translation, orientation, kinematic=False,
                       mass=cfg.mass_props.mass, com=(0.0, 0.0, cfg.com_z),
                       inertia=cfg.inertia)
    _child_cyl(prim_path, "floor", radius=cfg.floor_r, height=cfg.floor_t,
               tz=cfg.floor_t / 2, color=cfg.color, contact_offset=cfg.contact_offset)
    _octagon_wall(prim_path, "wall", cx=0.0, cy=0.0,
                  z_center=cfg.floor_t + cfg.wall_h / 2,
                  r_mid=cfg.wall_rmid, thick=cfg.wall_t, height=cfg.wall_h,
                  yaw0_deg=0.0, color=cfg.color, contact_offset=cfg.contact_offset)
    return root


def _spawner_cfgs() -> dict[str, Any]:
    """Lazily-built @configclass spawner cfg types (heavy imports deferred)."""
    if "frame" not in _SPAWNER_CACHE:
        from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
        from isaaclab.sim.utils import clone
        from isaaclab.utils import configclass

        @configclass
        class FrameSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_frame)
            post_r: float = 0.018
            post_h: float = 0.080
            flap_arm: float = 0.19
            vault_wall_rmid: float = 0.0725
            vault_wall_t: float = 0.015
            vault_h: float = 0.10
            vault_floor_t: float = 0.008
            ring_rmid: float = 0.0595
            ring_t: float = 0.006
            ring_h: float = 0.010
            steel: tuple = (0.42, 0.44, 0.48)
            wood: tuple = (0.55, 0.42, 0.25)
            wood_dark: tuple = (0.38, 0.28, 0.16)
            contact_offset: float = 0.002

        @configclass
        class RotorSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rotor)
            hub_r: float = 0.030
            flap_arm: float = 0.19
            flap_r: float = 0.125
            flap_t: float = 0.006
            pan_local: tuple = (0.05494, 0.11782, -0.042)
            pan_floor_r: float = 0.068
            pan_floor_t: float = 0.006
            pan_wall_rmid: float = 0.064
            pan_wall_t: float = 0.006
            pan_wall_h: float = 0.022
            com: tuple = (0.0147, -0.0629, -0.0090)
            inertia: tuple = (0.0054, 0.0008, 0.0062)
            ang_damp: float = 0.40
            steel: tuple = (0.42, 0.44, 0.48)
            steel_dark: tuple = (0.30, 0.32, 0.36)
            brass: tuple = (0.72, 0.60, 0.28)
            contact_offset: float = 0.002

        @configclass
        class BowlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bowl)
            floor_r: float = 0.048
            floor_t: float = 0.008
            wall_rmid: float = 0.044
            wall_t: float = 0.007
            wall_h: float = 0.052
            com_z: float = 0.012
            inertia: tuple = (0.0004, 0.0004, 0.0005)
            color: tuple = (0.05, 0.05, 0.06)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(frame=FrameSpawnerCfg, rotor=RotorSpawnerCfg, bowl=BowlSpawnerCfg)
    return _SPAWNER_CACHE


def _compound_spawner_cfg(kind: str, defaults: dict, *, mass: float, kinematic: bool) -> Any:
    import isaaclab.sim as sim_utils

    return _spawner_cfgs()[kind](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic),
        **defaults,
    )


# ----- scene cfg ------------------------------------------------------------------------------------
@dataclass
class CounterweightVaultSceneCfg(BaseCfg):
    """Config for `CounterweightVaultScene`. The torque budget behind the numbers:
    flap-side moment 0.0269 kg*m vs empty pan-side 0.0099 (gate rests ~13 deg from
    closed, flap still covers the whole mouth with ~11 mm to spare); the lightest
    counterweight (0.32 kg at 0.13 m) makes the pan side 1.6x the flap side even at the
    open stop, so the gate opens fully and stays open for every sampled mass; nothing
    lighter than ~0.15 kg in the pan can reach the open band at all (the 15 g cube
    moves the gate ~4 deg)."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    open_min_deg: float = tunable(90.0)  # gate counts as open at/above this (stop = 100)
    closed_max_deg: float = tunable(25.0)  # readback band: gate "closed" below this
    pan_xy_tol: float = tunable(0.020)  # bowl centre to pan centre, pan frame (m)
    pan_z_lo: float = tunable(-0.006)  # bowl bottom relative to pan floor top (m)
    pan_z_hi: float = tunable(0.012)
    bowl_up_max_deg: float = tunable(18.0)  # bowl axis vs up (the open pan tilts ~9 deg)
    cube_xy_tol: float = tunable(0.030)  # cube centre in white-bowl frame (inner clear r 0.040)
    cube_z_tol: float = tunable(0.008)  # |cube centre - (floor_t + cube/2)| in bowl frame
    cube_tilt_max_deg: float = tunable(30.0)  # cube roughly flat in the bowl
    home_xy_tol: float = tunable(0.008)  # white bowl still in its recess
    home_z_tol: float = tunable(0.006)
    home_up_max_deg: float = tunable(10.0)
    settle_speed: float = tunable(0.05)  # max |v| of every dynamic body when judging (m/s)
    settle_rotor_w: float = tunable(0.08)  # max rotor |omega| about the gate axis (rad/s)
    settle_streak: int = tunable(20)  # `settled` = still for this many CONSECUTIVE steps
    # (~0.17 s at 120 Hz): a decaying rotor rock reads instantaneously-still at every
    # velocity zero-crossing, and a single-poll judge at a turning point calls a moving
    # scene settled (seed-1 forge run: success flickered off 0.3 s after such a read)
    latch_speed: float = tunable(0.10)  # milestones latch only while the body is this slow
    latch_rotor_w: float = tunable(0.20)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    bowl_jitter: float = tunable(0.020)  # uniform +/- xy jitter of the black-bowl park
    cube_jitter: float = tunable(0.015)
    mirror_bowl: bool = tunable(True)  # per-episode: black bowl parks on +y or -y
    mirror_cube: bool = tunable(True)
    mass_lo: float = tunable(0.32)  # black-bowl mass range (kg), re-sampled per episode
    mass_hi: float = tunable(0.48)
    vbowl_jitter: float = tunable(0.001)  # white-bowl recess play (sub-mm; yaw is free)

    # --- tunable: placement (counter frame; intended arm base at (-0.42, 0, surface_z)) ---------
    surface_z: float = tunable(0.20)  # counter height; the arm base is mounted on the counter
    hinge_xy: tuple = tunable((0.10, 0.14))  # gate hinge (BUILD-time knob: the joint is
    # authored against the kinematic frame, so this must NOT vary per episode)
    bowl_park: tuple = tunable((-0.10, 0.24))  # black-bowl park (y side mirrored per episode)
    cube_park: tuple = tunable((0.32, 0.15))  # cube park (y side mirrored per episode)

    # --- info: mechanism structure ----------------------------------------------------------------
    bench_size: tuple = info((1.5, 1.3))
    tilt_deg: float = info(6.0)  # gate axis tilt from vertical (downhill = -y = vault side)
    hinge_h: float = info(0.110)  # hinge axis height above the counter
    pan_phase_deg: float = info(155.0)  # pan azimuth ahead of the flap (155 = 25 past anti-flap)
    flap_arm: float = info(0.19)  # hinge->flap-centre = hinge->vault-centre
    pan_arm: float = info(0.13)
    flap_r: float = info(0.125)
    flap_t: float = info(0.006)
    joint_lo_deg: float = info(-100.0)  # opening is NEGATIVE joint rotation; -100 = open stop
    joint_hi_deg: float = info(5.0)
    rest_open_deg: float = info(13.15)  # empty gravity rest (combined CoM at downhill);
    # the rotor SPAWNS and resets at this pose so episodes start swing-free
    rotor_mass: float = info(0.285)
    rotor_com: tuple = info((0.0147, -0.0629, -0.0090))  # authored AND runtime-enforced:
    # every gravity torque in the mechanism hangs on this off-axis CoM
    pan_floor_r: float = info(0.068)
    pan_wall_rmid: float = info(0.064)  # inner apothem 0.061: the ~103 mm bowl drops in with ~9 mm play
    pan_wall_t: float = info(0.006)
    pan_wall_h: float = info(0.022)
    pan_drop: float = info(0.042)  # pan floor top this far below the hinge axis
    vault_wall_rmid: float = info(0.0725)  # inner apothem 0.065 = the mouth
    vault_wall_t: float = info(0.015)
    vault_h: float = info(0.10)  # rim top above the counter
    vault_floor_t: float = info(0.008)
    ring_rmid: float = info(0.0595)  # recess ring: inner apothem 0.052 locates the white bowl
    ring_t: float = info(0.006)
    ring_h: float = info(0.010)
    post_r: float = info(0.018)
    post_h: float = info(0.080)
    hub_r: float = info(0.030)
    rotor_ang_damp: float = info(0.40)
    # bowls + cube
    bowl_floor_r: float = info(0.048)
    bowl_floor_t: float = info(0.008)
    cw_wall_rmid: float = info(0.044)  # black bowl: 7 mm wall, 52 mm deep -> rim pinch
    cw_wall_t: float = info(0.007)
    cw_wall_h: float = info(0.052)
    cw_mass0: float = info(0.40)  # authored nominal; reset() re-randomizes via set_masses
    vb_wall_rmid: float = info(0.043)  # white bowl: inner clear r 0.040, outer < recess play
    vb_wall_t: float = info(0.006)
    vb_wall_h: float = info(0.030)
    vb_mass: float = info(0.12)
    cube_size: float = info(0.026)
    cube_mass: float = info(0.015)
    black: tuple = info((0.05, 0.05, 0.06))  # matte black, like the seed's akita bowls
    white: tuple = info((0.92, 0.92, 0.90))
    red: tuple = info((0.85, 0.15, 0.12))
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    pan_local: tuple = field(default=None, init=False)  # pan floor-top centre, rotor frame
    vault_xy: tuple = field(default=None, init=False)  # vault centre, counter frame
    cw_h: float = field(default=None, init=False)  # black bowl total height
    vb_h: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        az = math.radians(self.pan_phase_deg - 90.0)  # flap sits at azimuth -90 (downhill)
        self.pan_local = (round(self.pan_arm * math.cos(az), 5),
                          round(self.pan_arm * math.sin(az), 5), -self.pan_drop)
        self.vault_xy = (self.hinge_xy[0], self.hinge_xy[1] - self.flap_arm)
        self.cw_h = round(self.bowl_floor_t + self.cw_wall_h, 4)
        self.vb_h = round(self.bowl_floor_t + self.vb_wall_h, 4)

    def rest_quat(self) -> tuple:
        """(w,x,y,z) rotor orientation at the empty gravity rest: joint rotation
        -rest_open_deg about the tilted gate axis. The rotor spawns AND resets here so
        no episode starts with a multi-second pendulum transient (the joint's angle
        coordinate stays anchored to the authored LocalRot frames, so the authored
        limit numbers are unaffected by the spawn pose)."""
        b = math.radians(self.tilt_deg)
        ax = (0.0, -math.sin(b), math.cos(b))
        half = math.radians(-self.rest_open_deg) / 2.0
        s = math.sin(half)
        return (math.cos(half), ax[0] * s, ax[1] * s, ax[2] * s)


# ----- scene -----------------------------------------------------------------------------------------
@SCENES.register("counterweight_vault")
class CounterweightVaultScene(BaseScene):
    cfg: CounterweightVaultSceneCfg

    def __init__(self, cfg: CounterweightVaultSceneCfg | None = None) -> None:
        super().__init__(cfg or CounterweightVaultSceneCfg())

    # ----- assets ---------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, counter slab, gate frame + rotor, black bowl, white bowl, cube,
        at nominal poses (reset() re-places the free bodies and samples the episode)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.surface_z
        hx, hy = c.hinge_xy

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
            "bench": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=sim_utils.CuboidCfg(
                    size=(c.bench_size[0], c.bench_size[1], z0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.08, 0.0, z0 / 2)),
            ),
        }

        frame_defaults = dict(post_r=c.post_r, post_h=c.post_h, flap_arm=c.flap_arm,
                              vault_wall_rmid=c.vault_wall_rmid, vault_wall_t=c.vault_wall_t,
                              vault_h=c.vault_h, vault_floor_t=c.vault_floor_t,
                              ring_rmid=c.ring_rmid, ring_t=c.ring_t, ring_h=c.ring_h,
                              contact_offset=c.contact_offset)
        out["frame"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Frame",
            spawn=_compound_spawner_cfg("frame", frame_defaults, mass=1.0, kinematic=True),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(hx, hy, z0)),
        )

        rotor_defaults = dict(hub_r=c.hub_r, flap_arm=c.flap_arm, flap_r=c.flap_r,
                              flap_t=c.flap_t, pan_local=c.pan_local,
                              pan_floor_r=c.pan_floor_r, pan_wall_rmid=c.pan_wall_rmid,
                              pan_wall_t=c.pan_wall_t, pan_wall_h=c.pan_wall_h,
                              com=c.rotor_com,
                              ang_damp=c.rotor_ang_damp, contact_offset=c.contact_offset)
        out["rotor"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Rotor",
            spawn=_compound_spawner_cfg("rotor", rotor_defaults,
                                        mass=c.rotor_mass, kinematic=False),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(hx, hy, z0 + c.hinge_h),
                                                      rot=c.rest_quat()),
        )

        out["cw_bowl"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cw_bowl",
            spawn=_compound_spawner_cfg(
                "bowl",
                dict(floor_r=c.bowl_floor_r, floor_t=c.bowl_floor_t, wall_rmid=c.cw_wall_rmid,
                     wall_t=c.cw_wall_t, wall_h=c.cw_wall_h, com_z=0.012,
                     inertia=(0.0004, 0.0004, 0.0005), color=c.black,
                     contact_offset=c.contact_offset),
                mass=c.cw_mass0, kinematic=False),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.bowl_park[0], c.bowl_park[1], z0 + 0.001)),
        )

        out["v_bowl"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/V_bowl",
            spawn=_compound_spawner_cfg(
                "bowl",
                dict(floor_r=0.047, floor_t=c.bowl_floor_t, wall_rmid=c.vb_wall_rmid,
                     wall_t=c.vb_wall_t, wall_h=c.vb_wall_h, com_z=0.010,
                     inertia=(0.0001, 0.0001, 0.00015), color=c.white,
                     contact_offset=c.contact_offset),
                mass=c.vb_mass, kinematic=False),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.vault_xy[0], c.vault_xy[1], z0 + c.vault_floor_t + 0.001)),
        )

        out["cube"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(c.cube_size, c.cube_size, c.cube_size),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=0.5),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.002, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.red),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.6, dynamic_friction=0.5, restitution=0.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.cube_park[0], c.cube_park[1], z0 + c.cube_size / 2 + 0.001)),
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

    # ----- lifecycle --------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles, author the gate joint, allocate layout readbacks + latches."""
        super().bind(env)
        c = self.cfg
        self.rotor: RigidObject = env.iscene["rotor"]
        self.cw_bowl: RigidObject = env.iscene["cw_bowl"]
        self.v_bowl: RigidObject = env.iscene["v_bowl"]
        self.cube: RigidObject = env.iscene["cube"]
        self.env_origins = env.iscene.env_origins
        self._author_joint()
        n = env.num_envs
        dev = env.device
        b = math.radians(c.tilt_deg)
        # gate axis in world (identical in every env: the frame never rotates)
        self._axis = torch.tensor([0.0, -math.sin(b), math.cos(b)], device=dev)
        f0 = torch.tensor([0.0, -1.0, 0.0], device=dev)
        self._flap0 = f0 - (f0 @ self._axis) * self._axis  # closed flap dir, in-plane part
        self._pan_local = torch.tensor(c.pan_local, device=dev)
        self._rest_quat = torch.tensor(c.rest_quat(), device=dev)
        self._enforce_rotor_mass_props()
        self.bowl_mass = torch.full((n,), c.cw_mass0, device=dev)
        self.bowl_side = torch.ones(n, device=dev)
        self.cube_side = torch.ones(n, device=dev)
        # progress latches (post_step; cleared per reset)
        self.ever_loaded = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ever_held = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ever_delivered = torch.zeros(n, dtype=torch.bool, device=dev)
        # consecutive-steps-still counter behind settled() (post_step; cleared per reset)
        self.rest_streak = torch.zeros(n, dtype=torch.long, device=dev)

    def _enforce_rotor_mass_props(self) -> None:
        """Write the rotor CoM into the PhysX view and VERIFY mass + CoM by readback.
        The whole mechanism (self-closing gate, counterweight hold) lives on this
        off-axis CoM; a rotor whose CoM silently lands on the hinge axis is a
        NEUTRAL pendulum and voids the task, so this is enforced at runtime on top
        of the USD authoring and printed once for the log."""
        c = self.cfg
        view = self.rotor.root_physx_view
        n = self.env.num_envs
        coms = view.get_coms().clone()
        flat = coms.view(n, 7)
        flat[:, :3] = torch.tensor(c.rotor_com, dtype=flat.dtype)
        view.set_coms(coms, torch.arange(n))
        com_back = view.get_coms().view(n, 7)[0, :3].tolist()
        mass_back = float(view.get_masses().view(n, -1)[0, 0])
        print(f"[counterweight_vault] rotor readback: mass={mass_back:.3f} "
              f"(authored {c.rotor_mass}) com={[round(v, 4) for v in com_back]} "
              f"(authored {c.rotor_com})", flush=True)
        if abs(mass_back - c.rotor_mass) > 0.02:
            print("[counterweight_vault] ROTOR MASS APPLY FAILED — torque budget void",
                  flush=True)

    def _author_joint(self) -> None:
        """The gate: a revolute joint, frame (kinematic) -> rotor, axis tilted
        `tilt_deg` from vertical with the downhill side toward the vault (-y).
        LocalRot Rx(+tilt) maps joint-Z onto (0, -sin b, cos b). Opening is the
        NEGATIVE rotation (the loaded pan swings toward the downhill azimuth)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        half = math.radians(c.tilt_deg) / 2.0
        qw, qx = math.cos(half), math.sin(half)
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/gate_pivot")
            j.CreateBody0Rel().SetTargets([f"{base}/Frame"])
            j.CreateBody1Rel().SetTargets([f"{base}/Rotor"])
            # CRITICAL: the rotor hub sleeves the post; without this the default
            # joint-pair collision welds the hinge solid.
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.hinge_h))
            j.CreateLocalRot0Attr(Gf.Quatf(qw, qx, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(qw, qx, 0.0, 0.0))
            j.CreateLowerLimitAttr(c.joint_lo_deg)
            j.CreateUpperLimitAttr(c.joint_hi_deg)

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: rotor back to the closed spawn pose (it settles ~13 deg onto
        its gravity rest), white bowl re-seated (fresh yaw), black bowl and cube parked
        on random sides with jitter, FRESH black-bowl mass; latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        z0 = c.surface_z
        hx, hy = c.hinge_xy

        def u(amp: float, shape=(1,)) -> torch.Tensor:
            return (torch.rand(m, *shape, device=dev) * 2 - 1) * amp

        def side(enabled: bool) -> torch.Tensor:
            if not enabled:
                return torch.ones(m, device=dev)
            # torch.rand comparison, NOT randint: the first randint draw after a fresh
            # manual_seed is near-constant across seeds.
            return torch.where(torch.rand(m, device=dev) > 0.5,
                               torch.ones(m, device=dev), -torch.ones(m, device=dev))

        def write(body: RigidObject, xy: torch.Tensor, z: float, yaw: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            st[:, 3] = torch.cos(yaw / 2)
            st[:, 6] = torch.sin(yaw / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # rotor: back to the empty gravity rest (open ~13 deg, flap covering the mouth)
        # — resetting AT equilibrium avoids a multi-second pendulum transient
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = hx, hy, z0 + c.hinge_h
        st[:, 3:7] = self._rest_quat
        st[:, 0:3] += origin
        self.rotor.write_root_state_to_sim(st, env_ids)

        # white bowl: seated in the recess, fresh yaw, sub-mm play
        vxy = torch.tensor(c.vault_xy, device=dev).expand(m, 2) + u(c.vbowl_jitter, (2,))
        write(self.v_bowl, vxy, z0 + c.vault_floor_t + 0.001,
              u(math.pi).squeeze(-1))

        # black bowl + cube: parked on random sides
        bs = side(c.mirror_bowl)
        self.bowl_side[env_ids] = bs
        bxy = torch.stack([torch.full((m,), c.bowl_park[0], device=dev),
                           c.bowl_park[1] * bs], dim=1) + u(c.bowl_jitter, (2,))
        write(self.cw_bowl, bxy, z0 + 0.001, u(math.pi).squeeze(-1))

        cs = side(c.mirror_cube)
        self.cube_side[env_ids] = cs
        cxy = torch.stack([torch.full((m,), c.cube_park[0], device=dev),
                           c.cube_park[1] * cs], dim=1) + u(c.cube_jitter, (2,))
        write(self.cube, cxy, z0 + c.cube_size / 2 + 0.001, u(math.pi).squeeze(-1))

        # fresh counterweight mass
        self.bowl_mass[env_ids] = (torch.rand(m, device=dev)
                                   * (c.mass_hi - c.mass_lo) + c.mass_lo)
        self._apply_masses(env_ids)

        for latch in (self.ever_loaded, self.ever_held, self.ever_delivered):
            latch[env_ids] = False
        self.rest_streak[env_ids] = 0

    def _apply_masses(self, env_ids: torch.Tensor) -> None:
        """Write the black-bowl mass into PhysX and VERIFY by readback — a silent no-op
        voids the counterweight physics the whole task hangs on."""
        view = self.cw_bowl.root_physx_view
        all_ids = torch.arange(self.env.num_envs)
        masses = view.get_masses().clone().cpu().view(self.env.num_envs, -1)
        masses[env_ids.cpu(), :] = self.bowl_mass[env_ids].cpu().view(-1, 1)
        view.set_masses(masses.view_as(view.get_masses()), all_ids)
        back = view.get_masses().cpu().view(self.env.num_envs, -1)[:, 0]
        err = (back.to(self.bowl_mass.device) - self.bowl_mass).abs().max()
        if float(err) > 1e-4:
            print(f"[counterweight_vault] MASS APPLY FAILED: max err {float(err):.4f} "
                  f"— the gate physics is void", flush=True)

    # ----- gate readings ------------------------------------------------------------------------
    def open_angle_deg(self) -> torch.Tensor:
        """(N,) gate swing from the closed spawn pose, POSITIVE toward open (the joint
        rotation is negative; this returns its negation). 0 = authored closed pose,
        ~13 = empty gravity rest, 100 = open stop."""
        from isaaclab.utils.math import quat_apply

        q = self.rotor.data.root_quat_w
        n = q.shape[0]
        f0 = torch.tensor([0.0, -1.0, 0.0], device=q.device).expand(n, 3)
        f = quat_apply(q, f0)
        a = self._axis
        pf = f - (f @ a).unsqueeze(-1) * a
        p0 = self._flap0.expand(n, 3)
        s = (torch.cross(p0, pf, dim=-1) @ a)
        cth = (p0 * pf).sum(dim=-1)
        return -torch.rad2deg(torch.atan2(s, cth))

    def rotor_axis_w(self) -> torch.Tensor:
        """(N,) rotor angular speed about the gate axis (rad/s, absolute)."""
        return (self.rotor.data.root_ang_vel_w @ self._axis).abs()

    def gate_open(self) -> torch.Tensor:
        return self.open_angle_deg() >= self.cfg.open_min_deg

    def pan_center_w(self) -> torch.Tensor:
        """(N,3) pan floor-top centre in world."""
        from isaaclab.utils.math import quat_apply

        q = self.rotor.data.root_quat_w
        return self.rotor.data.root_pos_w + quat_apply(
            q, self._pan_local.expand(q.shape[0], 3))

    # ----- geometric predicates -----------------------------------------------------------------
    def _up_z(self, quat: torch.Tensor) -> torch.Tensor:
        """z-component of a body's local +z in world, for tilt gates."""
        from isaaclab.utils.math import quat_apply

        n = quat.shape[0]
        ez = torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(n, 3)
        return quat_apply(quat, ez)[:, 2]

    def bowl_in_pan(self) -> torch.Tensor:
        """(N,) bool: the black bowl rests IN the pan — pan-frame xy within `pan_xy_tol`,
        bottom within [pan_z_lo, pan_z_hi] of the pan floor top, upright. Valid at any
        gate angle (the pan tilts with the rotor; the bands are rotor-frame)."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        q = self.rotor.data.root_quat_w
        d = self.cw_bowl.data.root_pos_w - self.pan_center_w()
        loc = quat_apply_inverse(q, d)
        in_xy = loc[:, :2].norm(dim=-1) < c.pan_xy_tol
        in_z = (loc[:, 2] > c.pan_z_lo) & (loc[:, 2] < c.pan_z_hi)
        up = self._up_z(self.cw_bowl.data.root_quat_w).clamp(-1, 1) \
            >= math.cos(math.radians(c.bowl_up_max_deg))
        return in_xy & in_z & up

    def cube_in_vbowl(self) -> torch.Tensor:
        """(N,) bool: cube rests INSIDE the white bowl (bowl frame): xy within
        `cube_xy_tol` (any physically-inside rest is <= 0.027; a rim perch is >= 0.043),
        centre z within `cube_z_tol` of floor_t + cube/2 (a rim perch is ~30 mm high),
        roughly flat."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        q = self.v_bowl.data.root_quat_w
        d = self.cube.data.root_pos_w - self.v_bowl.data.root_pos_w
        loc = quat_apply_inverse(q, d)
        in_xy = loc[:, :2].norm(dim=-1) < c.cube_xy_tol
        z_tgt = c.bowl_floor_t + c.cube_size / 2
        in_z = (loc[:, 2] - z_tgt).abs() < c.cube_z_tol
        flat = self._up_z(self.cube.data.root_quat_w).abs().clamp(max=1.0) \
            >= math.cos(math.radians(c.cube_tilt_max_deg))
        return in_xy & in_z & flat

    def vbowl_home(self) -> torch.Tensor:
        """(N,) bool: white bowl still seated in the vault recess, upright."""
        c = self.cfg
        p = self.v_bowl.data.root_pos_w - self.env_origins
        tgt = torch.tensor(c.vault_xy, device=p.device)
        in_xy = (p[:, :2] - tgt).norm(dim=-1) < c.home_xy_tol
        in_z = (p[:, 2] - (c.surface_z + c.vault_floor_t)).abs() < c.home_z_tol
        up = self._up_z(self.v_bowl.data.root_quat_w).clamp(-1, 1) \
            >= math.cos(math.radians(c.home_up_max_deg))
        return in_xy & in_z & up

    def _still_now(self) -> torch.Tensor:
        """(N,) bool: every dynamic body slow RIGHT NOW; rotor still about its axis."""
        c = self.cfg
        still = self.cw_bowl.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        still &= self.v_bowl.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        still &= self.cube.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        still &= self.rotor.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        still &= self.rotor_axis_w() < c.settle_rotor_w
        return still

    def settled(self) -> torch.Tensor:
        """(N,) bool: still for `settle_streak` CONSECUTIVE steps (counter maintained in
        post_step) — a decaying oscillation is instantaneously slow at every turning
        point, so a single-instant read would call a moving scene settled."""
        return self.rest_streak >= self.cfg.settle_streak

    def success(self) -> torch.Tensor:
        """(N,) bool: counterweight seated, gate held open, cube delivered into the
        undisturbed white bowl, everything at rest — a HELD state, not an event."""
        return (self.bowl_in_pan() & self.gate_open() & self.cube_in_vbowl()
                & self.vbowl_home() & self.settled())

    # ----- progress latching ----------------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch milestones at sim rate, velocity-gated: a body flying through a band
        latches nothing; a teleported-open EMPTY gate cannot latch `held` (no bowl in
        the pan — and gravity re-closes it); a cube written into the CLOSED vault
        cannot latch `delivered` (the gate-open clause is false whenever the flap
        covers the mouth)."""
        c = self.cfg
        now = self._still_now()
        self.rest_streak = torch.where(now, self.rest_streak + 1,
                                       torch.zeros_like(self.rest_streak))
        bowl_slow = self.cw_bowl.data.root_lin_vel_w.norm(dim=-1) < c.latch_speed
        in_pan = self.bowl_in_pan()
        self.ever_loaded |= in_pan & bowl_slow
        rotor_still = self.rotor_axis_w() < c.latch_rotor_w
        self.ever_held |= in_pan & bowl_slow & self.gate_open() & rotor_still
        cube_slow = self.cube.data.root_lin_vel_w.norm(dim=-1) < c.latch_speed
        self.ever_delivered |= (self.cube_in_vbowl() & self.vbowl_home()
                                & self.gate_open() & cube_slow)

    def score(self) -> torch.Tensor:
        """(N,) float 0..1 — additive latched milestones (monotone along a correct run):
        +0.20 loaded, +0.30 held open, +0.30 delivered; 1.0 iff success()."""
        s = torch.zeros(self.env.num_envs, device=self.env.device)
        s = s + 0.20 * self.ever_loaded.float()
        s = s + 0.30 * self.ever_held.float()
        s = s + 0.30 * self.ever_delivered.float()
        return torch.where(self.success(), torch.ones_like(s), s)

    # ----- state (full, restorable) ------------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"rotor": self.rotor, "cw_bowl": self.cw_bowl,
                  "v_bowl": self.v_bowl, "cube": self.cube}
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "bowl_mass": self.bowl_mass[env_ids].clone(),
            "sides": torch.stack([self.bowl_side[env_ids], self.cube_side[env_ids]], dim=1),
            "latches": torch.stack([self.ever_loaded[env_ids], self.ever_held[env_ids],
                                    self.ever_delivered[env_ids]], dim=1),
            "rest_streak": self.rest_streak[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"rotor": self.rotor, "cw_bowl": self.cw_bowl,
                  "v_bowl": self.v_bowl, "cube": self.cube}
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        self.bowl_mass[env_ids] = state["bowl_mass"]
        self._apply_masses(env_ids)
        sides = state["sides"]
        self.bowl_side[env_ids] = sides[:, 0]
        self.cube_side[env_ids] = sides[:, 1]
        lat = state["latches"]
        self.ever_loaded[env_ids] = lat[:, 0]
        self.ever_held[env_ids] = lat[:, 1]
        self.ever_delivered[env_ids] = lat[:, 2]
        self.rest_streak[env_ids] = state["rest_streak"]

    # ----- description -------------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wide gray kitchen counter. Near ({c.hinge_xy[0]:.2f}, {c.hinge_xy[1]:.2f}) "
            f"stands a GRAVITY GATE: a steel post carrying a rigid rotor on a hinge whose "
            f"axis leans {c.tilt_deg:.0f} deg from vertical. The rotor's LONG arm ends in a "
            f"wide dark-steel FLAP disc ({2 * c.flap_r * 1000:.0f} mm across) that parks over "
            f"the mouth of a walled wooden VAULT ({2 * 0.065 * 1000:.0f} mm inner mouth, rim "
            f"{c.vault_h * 1000:.0f} mm above the counter) just downhill of the hinge; the "
            f"SHORT arm carries an open brass PAN (inner ~{2 * 0.061 * 1000:.0f} mm across, "
            f"{c.pan_wall_h * 1000:.0f} mm walls, floor about {(c.hinge_h - c.pan_drop) * 1000:.0f} mm "
            f"above the counter). Because the axis leans, the flap always falls back over the "
            f"mouth: the EMPTY gate self-closes, and nothing lighter than about 0.15 kg in "
            f"the pan can hold it open. Inside the vault, a WHITE bowl sits in a shallow "
            f"recess. A matte-BLACK bowl ({2 * (c.cw_wall_rmid + c.cw_wall_t / 2) * 1000 / 0.9239:.0f} mm "
            f"across, {c.cw_h * 1000:.0f} mm tall, {c.cw_wall_t * 1000:.0f} mm rim wall — its "
            f"weight is re-randomized every episode but always ample) waits on one front "
            f"side of the counter, and a small RED sugar cube ({c.cube_size * 1000:.0f} mm) "
            f"on the other; sides and exact spots change every episode.\n"
            f"Goal: use the black bowl as a COUNTERWEIGHT, then deliver the cube. Set the "
            f"black bowl into the brass pan (it drops in with ~6 mm play and must sit flat); "
            f"its weight swings the gate ~100 deg open and HOLDS it open — the flap clears "
            f"the vault mouth only while the pan stays loaded. Then drop the red cube "
            f"through the opened mouth so it lands INSIDE the white bowl. Finish with the "
            f"bowl still seated in the pan, the gate resting open at its stop, the cube at "
            f"rest in the white bowl, and the white bowl undisturbed in its recess. A cube "
            f"on the flap, on a rim, or anywhere but inside the white bowl does not count; "
            f"the closed flap physically blocks the mouth, so do not try to poke the cube "
            f"past it. Everything must come to rest."
        )

    def instruction(self) -> str:
        return (
            "Set the black bowl into the brass pan so its weight swings the tilted gate "
            "open and holds it open, then drop the red sugar cube through the opened "
            "vault mouth so it rests inside the white bowl. Leave the bowl in the pan, "
            "the gate open at its stop, and the white bowl seated in its recess."
        )


# Scene-level task (robot="null"): solve.py is the teleport certificate; the intended
# embodiment (single Franka + parallel jaw) is argued in TASK.md.
register_env("simgen", lambda: EnvCfg(scene="counterweight_vault", robot="null"))

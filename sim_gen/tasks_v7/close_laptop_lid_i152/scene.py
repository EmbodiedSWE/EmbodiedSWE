"""BallastLidChestScene — drop steel blocks into the lid tray until their weight
swings the counterweighted lid shut.

Derived from rlbench/close_laptop_lid ("close laptop lid": push the raised hinged
screen of a laptop down through its arc until it lies shut), but the MANIPULATION
MODEL is replaced wholesale. The seed's plan is a single unordered act: one pushing
contact on a hinged panel, rotating it about its hinge until flush. Here PUSHING THE
LID SHUT DOES NOT WORK — the lid carries a rear COUNTERWEIGHT behind its hinge that
biases it OPEN with ~1.4 N.m to spare, so a lid pushed closed and released swings
right back up to its 46 deg stop (the smoke battery constructs exactly the seed's
strategy and proves the physics rejects it). Closing is instead performed by
BALLAST: the lid's top face carries a shallow walled TRAY near its front edge, and
each steel block laid in the tray adds ~0.10 kg.m of closing moment. One block is
never enough (>= 0.35 N.m short); two blocks always overcome the counterweight
(>= 0.27 N.m of net closing torque at every angle of the arc), so gravity swings
the lid down and HOLDS it down (~0.54 N.m). The solver's contribution to closure is
placing removable masses into a receptacle on the mechanism — pick-and-place onto a
tilted target — not a door push, and the final state is self-maintaining only
because the ballast stays in the tray.

Assets are fully procedural (the compound-spawner pattern — child colliders of one
body never self-collide):
  - chest: heavy DYNAMIC open-top box (outer 360 x 300 x 200 mm, 12 mm walls,
    25 kg). It must be dynamic, NOT kinematic: on this stack a joint anchored to a
    kinematic body0 stays world-fixed at its spawn pose when the body is teleported
    at reset (probe-verified), which would break the hinge under randomization.
  - lid: DYNAMIC compound whose body origin sits ON the hinge axis (back top edge
    of the chest, axis = chest local y). In front of the hinge: a 335 x 300 x 12 mm
    slab that covers the chest mouth, carrying the block tray (interior 70 x 200 mm,
    50 mm walls) near the front edge. Behind the hinge: a short bar and a dense
    counterweight bar (mass authored via per-child DENSITY — root-level mass_props
    on a custom spawner cfg is silently ignored on this stack, and per-child density
    also yields the true CoM, which is load-bearing here: lid CoM x ~= -44 mm,
    i.e. BEHIND the hinge).
  - blocks: three DYNAMIC dark steel blocks (50 x 50 x 36 mm, 0.40 kg), scattered
    on the ground in front of the chest.

Static torque budget about the hinge (moments in kg.m, worst cases over the block
rest window x in [0.25, 0.27] and the full arc 0..46 deg; asserted in
`__post_init__`):
  lid-forward (closing)  C_L = 0.157      counterweight (opening)  C_W = 0.302
  empty lid at 0 deg:  opens with C_W - C_L               = 0.145  (1.42 N.m)
  one block, worst:    still opens with C_W - C_L - 0.108 = 0.037  (0.36 N.m)
  two blocks, worst:   closes with >= 0.028 at every angle          (0.27 N.m)
  two blocks at 0 deg: held shut with >= 0.055                      (0.54 N.m)

Per-episode randomization (readback-verifiable): chest yaw +/- 20 deg + xy jitter,
per-block ground scatter + free yaw.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  0.25 * ballast1 — ever >= 1 block settled inside the lid tray (latched)
  0.25 * ballast2 — ever >= 2 blocks settled inside the lid tray (latched)
  0.20 * closing  — lid opening angle ever < 20 deg while >= 2 blocks ride in the
                    tray (latched; pushing the empty lid down does NOT fire it)
  1.0 iff success() — lid opening angle <= 3 deg, >= 2 blocks inside the tray,
                    everything settled and finite. Non-success capped at 0.70.

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


# ----- custom compound spawner ------------------------------------------------------------------
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


def _spawn_chest(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the chest at `prim_path`: heavy DYNAMIC open-top box (root mass 25 kg;
    the root-mass CoM lands at the body origin on the ground, which only makes the
    chest harder to tip). Local frame: origin at the footprint centre on the ground;
    the front (block-scatter side) faces local +x, the lid hinge runs along local y
    at the back top edge (x = -0.18, z = 0.213)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(25.0)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    body = c.body_color
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, 0.006),
             size=(0.336, 0.276, 0.012), color=body, collide=collide)
    for sgn, name in ((1.0, "front"), (-1.0, "back")):
        _add_box(stage, f"{prim_path}/{name}", center=(sgn * 0.174, 0.0, 0.100),
                 size=(0.012, 0.300, 0.200), color=body, collide=collide)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/side_{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * 0.144, 0.100),
                 size=(0.336, 0.012, 0.200), color=body, collide=collide)
    return root


def _spawn_lid(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the lid at `prim_path`: DYNAMIC compound whose body origin sits ON the
    hinge axis, plus the REVOLUTE joint to the sibling chest (joints must be
    authored at spawn — post-play joints are dead).

    All child masses are authored via per-child DENSITY (root-level `mass_props` on
    a custom spawner cfg is silently ignored on this stack; per-child density works
    and yields the true CoM + inertia, both load-bearing for the counterweight).

    Children (local frame: +x = forward over the chest mouth when closed):
      slab   x  0.025..0.360  (the 25 mm standoff keeps the swept slab edge >= 5 mm
             clear of the chest back-wall plane at every hinge angle:
             0.025*cos(46 deg) = 17.4 mm > wall 12 mm)
      tray   hinge-side wall at x 0.210..0.225, front lip at 0.295..0.310, side
             walls between them; interior 70 x 200 mm, walls 50 mm tall
      bar    x -0.080..-0.030  (light link to the counterweight; the empty band
             x -0.030..+0.025 is the hinge cutout)
      cw     x -0.160..-0.080  (dense counterweight bar, ~2.50 kg at x = -0.12)
    """
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(float(c.lid_ang_damping))
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(1)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)

    def boxd(path, center, size, color, density):
        prim = _add_box(stage, f"{prim_path}/{path}", center=center, size=size,
                        color=color, collide=collide)
        UsdPhysics.MassAPI.Apply(prim).CreateDensityAttr(float(density))

    wood, trim, dense = c.lid_color, c.tray_color, c.cw_color
    d = c.wood_density
    # slab (forward of the hinge; top face at local z = 0)
    boxd("slab", (0.1925, 0.0, -0.006), (0.335, 0.300, 0.012), wood, d)
    # tray walls (block interior x 0.225..0.295, |y| < 0.100, 50 mm tall)
    boxd("tray_hinge", (0.2175, 0.0, 0.025), (0.015, 0.230, 0.050), trim, d)
    boxd("tray_lip", (0.3025, 0.0, 0.025), (0.015, 0.230, 0.050), trim, d)
    for sgn in (1.0, -1.0):
        boxd(f"tray_side_{'p' if sgn > 0 else 'n'}",
             (0.260, sgn * 0.1075, 0.025), (0.070, 0.015, 0.050), trim, d)
    # rear link bar + counterweight
    boxd("bar", (-0.055, 0.0, 0.0), (0.050, 0.060, 0.020), wood, d)
    boxd("cw", (-0.120, 0.0, 0.0), (0.080, 0.100, 0.042), dense, c.cw_density)

    # revolute hinge to the sibling chest, axis along the chest back top edge (y)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([f"{base}/Chest"])
    j.CreateBody1Rel().SetTargets([prim_path])
    # collision between the lid and the chest MUST stay ON (the USD default for a
    # joint pair is filtered): the closed rest IS slab-on-wall-rim contact
    j.CreateCollisionEnabledAttr(True)
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(float(c.hinge_x), 0.0, float(c.hinge_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    # opening rotation (front edge up) is NEGATIVE about local +y: joint angle
    # -46 deg = fully open stop, +2 deg = slight over-close slack (the physical
    # closed stop is the slab resting on the wall rims at ~ -0.2 deg opening)
    j.CreateLowerLimitAttr(-float(c.open_stop_deg))
    j.CreateUpperLimitAttr(2.0)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "chest" not in _SPAWNER_CACHE:

        @configclass
        class ChestSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_chest)
            body_color: tuple = (0.30, 0.33, 0.40)
            contact_offset: float = 0.002

        @configclass
        class LidSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lid)
            hinge_x: float = -0.18
            hinge_z: float = 0.213
            open_stop_deg: float = 46.0
            lid_ang_damping: float = 3.0
            wood_density: float = 450.0
            cw_density: float = 7450.0
            lid_color: tuple = (0.72, 0.54, 0.30)
            tray_color: tuple = (0.85, 0.68, 0.20)
            cw_color: tuple = (0.35, 0.35, 0.38)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["chest"] = ChestSpawnerCfg
        _SPAWNER_CACHE["lid"] = LidSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BallastLidChestSceneCfg(BaseCfg):
    """Config for `BallastLidChestScene`. The closed gate (opening angle <=
    `closed_max_deg`) is honest by construction: the counterweight holds an
    under-ballasted lid >= 0.35 N.m away from closing at every angle, so the only
    settled state that passes is a lid held down by >= 2 tray blocks; the physical
    rim stop bounds a ballast-closed lid to ~ -0.2..0 deg, far inside the 3 deg
    tolerance."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    closed_max_deg: float = tunable(3.0)     # lid opening angle at/below this counts as closed
    close_latch_deg: float = tunable(20.0)   # partial-credit latch: lid ever below this w/ ballast
    settle_speed: float = tunable(0.05)      # max block |lin vel| when judging (m/s)
    lid_settle_avel: float = tunable(0.30)   # max lid |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    chest_yaw_deg: float = tunable(20.0)     # chest yaw about its nominal heading (+/- deg)
    chest_jitter: float = tunable(0.04)      # chest xy jitter (+/- m)
    block_jitter: float = tunable(0.03)      # per-block ground xy jitter (+/- m)
    block_yaw_deg: float = tunable(180.0)    # per-block free yaw (+/- deg)

    # --- info: layout (world nominal; chest front faces its local +x) ---------------------------
    chest_pos: tuple = info((0.44, 0.0))     # chest origin on the ground (nominal)
    chest_yaw_nom_deg: float = info(180.0)   # nominal heading: front faces world -x (the robot)
    block_slots: tuple = info(((0.30, -0.14), (0.38, 0.0), (0.30, 0.14)))  # chest-local, ground
    # --- info: chest structure (local frame: origin at footprint centre, ground) ----------------
    wall_t: float = info(0.012)
    chest_dx: float = info(0.360)   # outer footprint x
    chest_dy: float = info(0.300)   # outer footprint y
    wall_h: float = info(0.200)     # wall top height
    hinge_x: float = info(-0.18)    # hinge axis, chest frame (back outer face)
    hinge_z: float = info(0.213)
    # --- info: lid (local frame: origin ON the hinge axis, +x forward when closed) --------------
    slab_x0: float = info(0.025)    # slab inner edge (hinge cutout boundary)
    slab_x1: float = info(0.360)    # slab front edge (flush over the front wall)
    slab_t: float = info(0.012)
    lid_w: float = info(0.300)
    tray_x0: float = info(0.225)    # tray interior (block rest window x 0.25..0.27)
    tray_x1: float = info(0.295)
    tray_hy: float = info(0.100)    # tray interior half width
    tray_wall_h: float = info(0.050)
    open_stop_deg: float = info(46.0)   # joint limit; the lid rests open here
    lid_ang_damping: float = info(3.0)  # jointFriction is inert on non-articulation joints;
    #                                     body angular damping is what stills the swing
    wood_density: float = info(450.0)
    cw_density: float = info(7450.0)    # counterweight ~2.50 kg at x -0.12 -> 0.300 kg.m opening
    lid_mass_ref: float = info(3.28)    # expected PhysX-derived lid mass (readback guard)
    lid_com_x_max: float = info(-0.020)  # lid CoM must sit BEHIND the hinge (readback guard)
    # --- info: blocks ----------------------------------------------------------------------------
    block_size: tuple = info((0.05, 0.05, 0.036))
    block_mass: float = info(0.40)
    block_color: tuple = info((0.25, 0.26, 0.30))
    contact_offset: float = info(0.002)
    # rubric weights (0.25 + 0.25 + 0.20 = 0.70 = the non-success cap)
    w_ballast1: float = info(0.25)
    w_ballast2: float = info(0.25)
    w_closing: float = info(0.20)

    def __post_init__(self) -> None:
        """Audit the static torque budget and the swept-clearance bounds (moments in
        kg.m about the hinge; masses derived from the authored densities)."""
        d = self.wood_density
        m_slab = 0.335 * 0.300 * 0.012 * d              # 0.543 kg
        m_wall = 0.015 * 0.230 * 0.050 * d              # 0.078 kg (hinge wall, lip)
        m_side = 0.070 * 0.015 * 0.050 * d              # 0.024 kg
        m_bar = 0.050 * 0.060 * 0.020 * d               # 0.027 kg
        m_cw = 0.080 * 0.100 * 0.042 * self.cw_density  # 2.50 kg
        c_l = m_slab * 0.1925 + m_wall * (0.2175 + 0.3025) + 2 * m_side * 0.260
        c_w = m_cw * 0.120 + m_bar * 0.055
        m_b = self.block_mass
        x_min, x_max = self.tray_x0 + 0.025, self.tray_x1 - 0.025  # lying-block rest window
        z_b = self.block_size[2] / 2
        th = math.radians(self.open_stop_deg)
        # one block anywhere in the tray must NOT close the lid (worst at 0 deg)
        assert c_w - c_l - m_b * x_max > 0.030, (c_w, c_l)
        # two blocks at the hinge-side wall must close it from the stop (worst angle)
        close_46 = math.cos(th) * (c_l + 2 * m_b * x_min - c_w) - math.sin(th) * 2 * m_b * z_b
        assert close_46 > 0.020, close_46
        # the empty lid must reopen decisively from closed (the seed-strategy rejection)
        assert c_w - c_l > 0.100, (c_w, c_l)
        # swept clearance: slab inner edge stays clear of the back-wall plane at all angles
        assert self.slab_x0 * math.cos(th) > self.wall_t + 0.004
        # counterweight ground clearance at the open stop
        assert self.hinge_z - (0.160 * math.sin(th) + 0.021 * math.cos(th)) > 0.05


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


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("ballast_lid_chest")
class BallastLidChestScene(BaseScene):
    cfg: BallastLidChestSceneCfg

    def __init__(self, cfg: BallastLidChestSceneCfg | None = None) -> None:
        super().__init__(cfg or BallastLidChestSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        chest_spawn = cls["chest"](
            mass_props=sim_utils.MassPropertiesCfg(mass=25.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            contact_offset=c.contact_offset)
        lid_spawn = cls["lid"](
            hinge_x=c.hinge_x, hinge_z=c.hinge_z, open_stop_deg=c.open_stop_deg,
            lid_ang_damping=c.lid_ang_damping, wood_density=c.wood_density,
            cw_density=c.cw_density, contact_offset=c.contact_offset)

        dyn_props = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.2, angular_damping=0.2,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5, dynamic_friction=0.4, restitution=0.0),
        )

        # template poses: the lid MUST spawn consistent with its authored joint
        # frames (chest at nominal yaw -> hinge world pose, lid closed = angle 0)
        px, py = c.chest_pos
        yaw0 = math.radians(c.chest_yaw_nom_deg)
        cy, sy = math.cos(yaw0), math.sin(yaw0)
        hx_w = px + cy * c.hinge_x
        hy_w = py + sy * c.hinge_x
        q0 = (math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2))

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
            "chest": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Chest",
                spawn=chest_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0), rot=q0),
            ),
            "lid": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lid",
                spawn=lid_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(hx_w, hy_w, c.hinge_z), rot=q0),
            ),
        }
        for i in range(3):
            out[f"block{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Block_" + str(i),
                spawn=sim_utils.CuboidCfg(
                    size=c.block_size,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.block_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.block_color),
                    **dyn_props,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0 + 0.2 * i, 1.0, 0.05)),
            )
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # smoke probes shove bodies via set_external_force_and_torque;
                # without this flag wrenches are under-applied across TGS iterations
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

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.chest: RigidObject = env.iscene["chest"]
        self.lid: RigidObject = env.iscene["lid"]
        self.blocks: list[RigidObject] = [env.iscene[f"block{i}"] for i in range(3)]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches (partial credit survives transients; success is judged live)
        self._ballast1 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._ballast2 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._closing = torch.zeros(n, dtype=torch.bool, device=dev)
        # consecutive-still counters guarding the ballast latches: a velocity
        # threshold alone fires vacuously at bounce turning points, so a block that
        # skips through the tray and leaves must not latch credit
        self._still1 = torch.zeros(n, dtype=torch.long, device=dev)
        self._still2 = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the chest (yaw + xy jitter), hang the lid on its
        hinge at the OPEN stop (pose consistent with the joint frames — the lid
        origin sits on the hinge axis, so any opening angle is a pure quaternion
        change), scatter the three blocks on the ground in front (jitter + free
        yaw), clear the latches."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- chest: heavy dynamic, nominal heading + yaw + xy jitter ---
        yaw = math.radians(c.chest_yaw_nom_deg) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.chest_yaw_deg)
        q_chest = _qz(yaw)
        cp = torch.zeros(m, 3, device=dev)
        cp[:, 0] = c.chest_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.chest_jitter
        cp[:, 1] = c.chest_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.chest_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = cp + origin
        st[:, 3:7] = q_chest
        self.chest.write_root_state_to_sim(st, env_ids)

        # --- lid: on its hinge, just inside the open stop (settles onto it) ---
        hinge = torch.zeros(m, 3, device=dev)
        hinge[:, 0] = c.hinge_x
        hinge[:, 2] = c.hinge_z
        open0 = torch.full((m,), math.radians(c.open_stop_deg - 1.0), device=dev)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = cp + quat_apply(q_chest, hinge) + origin
        st[:, 3:7] = _qmul(q_chest, _qy(-open0))
        self.lid.write_root_state_to_sim(st, env_ids)

        # --- blocks: ground scatter in front of the chest (jitter + free yaw) ---
        for i, body in enumerate(self.blocks):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = c.block_slots[i][0] \
                + (torch.rand(m, device=dev) * 2 - 1) * c.block_jitter
            loc[:, 1] = c.block_slots[i][1] \
                + (torch.rand(m, device=dev) * 2 - 1) * c.block_jitter
            qb = _qz((torch.rand(m, device=dev) * 2 - 1) * math.radians(c.block_yaw_deg))
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = cp + quat_apply(q_chest, loc) + origin
            st[:, 2] = c.block_size[2] / 2 + 0.002 + origin[:, 2]
            st[:, 3:7] = _qmul(q_chest, qb)
            body.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._ballast1[env_ids] = False
        self._ballast2[env_ids] = False
        self._closing[env_ids] = False
        self._still1[env_ids] = 0
        self._still2[env_ids] = 0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {
            "chest": self.chest.data.root_state_w[env_ids].clone(),
            "lid": self.lid.data.root_state_w[env_ids].clone(),
            "ballast1": self._ballast1[env_ids].clone(),
            "ballast2": self._ballast2[env_ids].clone(),
            "closing": self._closing[env_ids].clone(),
            "still1": self._still1[env_ids].clone(),
            "still2": self._still2[env_ids].clone(),
        }
        for i, body in enumerate(self.blocks):
            out[f"block{i}"] = body.data.root_state_w[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.chest.write_root_state_to_sim(state["chest"], env_ids)
        self.lid.write_root_state_to_sim(state["lid"], env_ids)
        for i, body in enumerate(self.blocks):
            body.write_root_state_to_sim(state[f"block{i}"], env_ids)
        self._ballast1[env_ids] = state["ballast1"]
        self._ballast2[env_ids] = state["ballast2"]
        self._closing[env_ids] = state["closing"]
        self._still1[env_ids] = state["still1"]
        self._still2[env_ids] = state["still2"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A blue-grey open-top CHEST ({c.chest_dx * 1000:.0f} x "
            f"{c.chest_dy * 1000:.0f} mm footprint, {c.wall_h * 1000:.0f} mm walls) "
            f"stands on the ground. Its wooden LID is hinged along the chest's BACK "
            f"top edge and is currently held OPEN at about {c.open_stop_deg:.0f} "
            f"degrees by a dark COUNTERWEIGHT bar that sticks out behind the hinge: "
            f"the lid is heavier behind the hinge than in front, so an empty lid "
            f"always swings back up to its open stop — pushing it closed and letting "
            f"go does NOT keep it closed. On the lid's top face, near its front "
            f"edge, sits a shallow yellow-walled TRAY (interior "
            f"{(c.tray_x1 - c.tray_x0) * 1000:.0f} x {2 * c.tray_hy * 1000:.0f} mm, "
            f"{c.tray_wall_h * 1000:.0f} mm walls). Because the lid is tilted, the "
            f"tray floor slopes toward the hinge and anything laid in it rests "
            f"against the tray's hinge-side wall. Three dark STEEL BLOCKS "
            f"({c.block_size[0] * 1000:.0f} x {c.block_size[1] * 1000:.0f} x "
            f"{c.block_size[2] * 1000:.0f} mm, {c.block_mass:.1f} kg each) lie on "
            f"the ground in front of the chest. Each block laid in the tray adds "
            f"closing leverage: ONE block is not enough to overcome the "
            f"counterweight, but TWO (or all three) are — the lid then swings shut "
            f"under gravity and their weight keeps it shut. The chest's position "
            f"and heading and the blocks' scatter vary per episode.\n"
            f"Goal: get the lid fully closed (lying flat on the chest rim, within "
            f"{c.closed_max_deg:.0f} degrees) with at least TWO steel blocks "
            f"sitting inside the lid tray, everything at rest. Blocks dropped "
            f"inside the chest cavity, left on the ground, or balanced anywhere "
            f"else do not count toward success — only blocks resting INSIDE the lid "
            f"tray count, and blocks in the cavity or on the ground add no closing "
            f"leverage. Pushing the lid shut without the ballast fails because it "
            f"reopens on its own."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Place at least two of the steel blocks into the yellow tray on the "
            "chest's raised lid. Their weight overcomes the counterweight behind "
            "the hinge and swings the lid shut. Do not just push the lid down — "
            "without the ballast it springs back open. Finish with the lid fully "
            "closed and at least two blocks resting in the lid tray."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def open_angle_deg(self) -> torch.Tensor:
        """(N,) float: lid opening angle in degrees (0 = closed flat on the rim,
        `open_stop_deg` = at the open stop), extracted from the chest->lid relative
        quaternion about the hinge (local y) axis. There is no joint-state API on a
        plain spawn-authored USD joint, so this is the hinge readout."""
        qc = self.chest.data.root_quat_w
        ql = self.lid.data.root_quat_w
        qc_inv = qc * torch.tensor([1.0, -1.0, -1.0, -1.0], device=qc.device)
        rel = _qmul(qc_inv, ql)
        # pure y rotation: rel ~ (cos(a/2), 0, sin(a/2), 0); opening = -a
        ang = 2.0 * torch.atan2(rel[:, 2], rel[:, 0])
        ang = torch.rad2deg(ang)
        ang = torch.where(ang > 180.0, ang - 360.0, ang)
        ang = torch.where(ang < -180.0, ang + 360.0, ang)
        return -ang

    def _lid_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the lid body frame (origin on the hinge axis)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.lid.data.root_quat_w,
                                  pos_w - self.lid.data.root_pos_w)

    def _chest_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.chest.data.root_quat_w,
                                  pos_w - self.chest.data.root_pos_w)

    def in_tray(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,) bool: world point inside the lid-tray interior volume (lid frame,
        slightly padded)."""
        c = self.cfg
        loc = self._lid_local(pos_w)
        return (loc[:, 0] > c.tray_x0 - 0.005) & (loc[:, 0] < c.tray_x1 + 0.005) \
            & (loc[:, 1].abs() < c.tray_hy + 0.005) \
            & (loc[:, 2] > 0.002) & (loc[:, 2] < c.tray_wall_h + 0.020)

    def in_cavity(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,) bool: world point inside the chest interior (the WRONG place)."""
        c = self.cfg
        loc = self._chest_local(pos_w)
        return (loc[:, 0].abs() < c.chest_dx / 2 - c.wall_t) \
            & (loc[:, 1].abs() < c.chest_dy / 2 - c.wall_t) \
            & (loc[:, 2] > c.wall_t) & (loc[:, 2] < c.wall_h)

    def tray_count(self) -> torch.Tensor:
        """(N,) int: number of blocks whose centre rides inside the lid tray."""
        return torch.stack([self.in_tray(b.data.root_pos_w)
                            for b in self.blocks], dim=1).sum(dim=1)

    def settled(self) -> torch.Tensor:
        """(N,) bool: all blocks below `settle_speed`, lid below `lid_settle_avel`."""
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                         for b in self.blocks], dim=1)
        w = self.lid.data.root_ang_vel_w.norm(dim=-1)
        return (v < self.cfg.settle_speed).all(dim=1) & (w < self.cfg.lid_settle_avel)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w for b in (self.chest, self.lid, *self.blocks)],
                        dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Advance the still counters and latches ONCE per physics step (calling
        this from success()/score() too would double-count the counters)."""
        cnt = self.tray_count()
        ok = self.settled() & self._finite()
        zero = torch.zeros_like(self._still1)
        self._still1 = torch.where((cnt >= 1) & ok, self._still1 + 1, zero)
        self._still2 = torch.where((cnt >= 2) & ok, self._still2 + 1, zero)
        self._ballast1 |= self._still1 >= 6   # 6 steps = 50 ms of true rest in the tray
        self._ballast2 |= self._still2 >= 6
        self._closing |= (cnt >= 2) & (self.open_angle_deg() < self.cfg.close_latch_deg) \
            & self._finite()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: lid closed (opening angle <= `closed_max_deg`), at least two
        blocks inside the lid tray, everything settled and finite. All clauses are
        live physical outcomes — the counterweight guarantees the closed clause can
        only hold in a settled state if the tray ballast is actually doing the
        holding."""
        return (self.open_angle_deg() <= self.cfg.closed_max_deg) \
            & (self.tray_count() >= 2) & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.25*ballast1 + 0.25*ballast2 + 0.20*closing (all
        latched; ~0 for doing nothing — pushing the empty lid down latches nothing
        because `closing` requires two blocks riding in the tray), capped at 0.70 —
        and exactly 1.0 iff success() holds live."""
        c = self.cfg
        base = (c.w_ballast1 * self._ballast1.float()
                + c.w_ballast2 * self._ballast2.float()
                + c.w_closing * self._closing.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="ballast_lid_chest", robot="null"))

"""RockerTwinDrawersScene — put the chocolate pudding in the SHUT drawer of an
anti-phase twin-drawer chest and shut it again (sim_gen task
`libero_kitchen_scene10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it_i424`).

Derived from the LIBERO scene10 seed "put the chocolate pudding in the top drawer of
the cabinet and close it" but STRATEGICALLY DIFFERENT (see TASK.md): the seed's plan
is grasp the drawer HANDLE, pull the drawer open, drop the pudding in from above,
push it shut — the drawer is an independent 1-DOF receptacle the hand actuates
directly in both directions. Here the chest has TWO side-by-side drawers with
smooth, knobless fronts that finish FLUSH with the cabinet face when shut: a shut
drawer offers nothing to hook, pinch or pull, and it already rests on its inner hard
stop, so pressing it does nothing either. The two drawers are coupled by a hidden
WALKING BEAM (a vertical-axis rocker whose two downward pins ride fork slots on each
drawer's tail): the transmission is a captive 1:1 INVERTER that enforces
ext_A + ext_B ~= travel at every instant. Consequences that reshape the whole plan:
(1) the goal drawer (the one that starts SHUT — which side is sampled per episode)
can only be opened by pushing the OTHER, open drawer fully IN; (2) at most one
drawer can be shut at a time, so closing the loaded goal drawer necessarily POPS THE
TWIN BACK OUT — "leave the chest fully closed", the seed's implicit end state, is
physically impossible, and the episode ends with the decoy drawer standing open;
(3) the transmission is exercised in BOTH directions in one episode (twin-in to open
the goal, goal-in to re-eject the twin). The seed's strategy transplanted here —
pull open / drop / push shut on one independent drawer — dies at step one (nothing
to pull) and its drop-from-above dies against the chest roof (both constructed and
rejected in smoke).

Mechanics (plain rigid bodies + authored USD joints): the chest is kinematic slabs
(plinth, windowed front wall, sides, rear, roof). Each drawer is ONE dynamic body
(tray floor root) with panel, tray walls and tail fork plates as compound collision
children, riding an X-axis prismatic joint (limits [-travel, 0]; 0 = shut flush).
The rocker is ONE dynamic body (overhead bar root) with two pin cylinders as
children, on a free Z revolute over the plinth; each pin sits captive between a
drawer's fork plates (4 mm backlash) for the drawer's whole travel — it can never
disengage. `post_step` applies viscous damping to both drawers and the rocker and
consumes two drive buffers — `drawer_drive` (N,2 world-X forces on [left, right])
and `cargo_drive` (N,3 world force on the pudding, wrench-frame pre-encoded);
nothing else touches those wrench slots.

Rubric (0..1, latched credit anchored in the demonstrated solve trajectory:
open-via-twin -> load -> shut):
  - `open_latch` (0.15): the goal drawer has ever been >= `open_thresh` out (only
    reachable through the transmission); also forced by a successful load;
  - `load_latch` (0.35): the pudding has ever rested inside the GOAL drawer's tray
    (drawer-frame band, velocity-gated);
  - current success (0.50): pudding in the goal tray AND the goal drawer within
    `closed_tol` of flush AND everything settled -> score == 1.0 iff success().
Null policy ~0 (goal drawer starts shut, pudding on the floor). Loading the WRONG
(initially open) drawer and shutting it — a plausible lazy reading — scores at most
the 0.15 open latch and never succeeds.

Per-episode randomization (readback-verified in smoke): WHICH side is the shut goal
drawer (left/right, with the rocker and both drawers posed consistently), plus the
pudding's floor position and free yaw.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class RockerTwinDrawersSceneCfg(BaseCfg):
    """Config for `RockerTwinDrawersScene`. Geometry is derived once in `__post_init__`
    so the scene, the smoke AND the solver read the same numbers."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    closed_tol: float = tunable(0.008)  # goal drawer extension below this = shut flush
    open_thresh: float = tunable(0.085)  # goal drawer extension above this = opened
    # Tray containment band, GOAL-DRAWER body frame (root = tray floor centre).
    # Honest by construction: a pudding resting anywhere on the tray floor has
    # x in [-0.056, +0.046], |y| <= 0.031, z ~= +0.028 (interior x [-0.080, 0.070],
    # y +-0.055, minus the 24 mm cube half). The band is only reachable ON the tray
    # floor: below it is the floor slab, above 0.060 is a cube perched on the walls.
    tray_x_lo: float = tunable(-0.058)
    tray_x_hi: float = tunable(0.048)
    tray_y_tol: float = tunable(0.033)
    tray_z_lo: float = tunable(0.010)
    tray_z_hi: float = tunable(0.060)
    settle_cargo: float = tunable(0.05)  # max pudding |lin vel| (m/s) when judging
    settle_drawer: float = tunable(0.02)  # max drawer |vx| (m/s) when judging

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    cargo_x: tuple = tunable((0.06, 0.15))  # pudding floor spawn x band
    cargo_y_half: float = tunable(0.15)  # pudding floor spawn |y| bound

    # --- tunable: plant (difficulty dials) ---------------------------------------------------
    drawer_damping: float = tunable(3.0)  # viscous slide friction per drawer (N*s/m)
    rocker_damping: float = tunable(0.05)  # viscous pivot friction (N*m*s/rad);
    # explicit-damping stability: c*dt/I ~ 0.05*(1/120)/0.0015 = 0.28 < 1
    drawer_mass: float = tunable(0.5)
    rocker_mass: float = tunable(0.25)
    cargo_mass: float = tunable(0.08)
    cargo_size: float = tunable(0.048)  # pudding box edge (cube)

    # --- info: structure (env-local coordinates; chest faces -x, ground z=0) -----------------
    # Drawer slide: X prismatic, limits [-travel, 0]; extension e = -q (0 shut flush,
    # travel fully out). Shut drawer root at closed_x; drawer centres at y +-drawer_y.
    travel: float = info(0.110)
    drawer_y: float = info(0.105)
    closed_x: float = info(0.410)
    floor_z: float = info(0.038)  # drawer root (tray floor centre) height
    floor_size: tuple = info((0.160, 0.130, 0.008))
    # Drawer children (drawer local frame). Panel finishes flush in the front wall
    # plane x [0.306, 0.330] when shut, with 4 mm side / 5 mm top / 4 mm bottom
    # clearance to the aperture (> the 3 mm summed contact offsets: no phantom drag).
    panel_c: tuple = info((-0.092, 0.0, 0.0525))
    panel_s: tuple = info((0.024, 0.155, 0.113))
    traywall_y: float = info(0.060)  # tray side wall centreline
    traywall_s: tuple = info((0.160, 0.010, 0.093))
    trayrear_c: tuple = info((0.075, 0.0, 0.0425))
    trayrear_s: tuple = info((0.010, 0.110, 0.093))
    # Tail fork: two plates normal to X; the rocker pin rides the 20 mm gap between
    # them (16 mm pin -> 4 mm backlash). Fork centreline offset +-fork_dy toward the
    # chest midline mirror (pin y = arm*cos(phi) walks 0.105..0.1185).
    fork_front_x: float = info(0.086)
    fork_rear_x: float = info(0.114)
    fork_dy: float = info(0.007)
    fork_s: tuple = info((0.008, 0.066, 0.060))
    fork_z: float = info(0.029)
    # Rocker: bar root at the pivot, pins hang to z [0.037, 0.135].
    pivot: tuple = info((0.455, 0.0, 0.143))
    arm_len: float = info(0.1185)
    bar_s: tuple = info((0.024, 0.267, 0.016))
    pin_r: float = info(0.008)
    pin_h: float = info(0.098)
    pin_dz: float = info(-0.057)
    rocker_limit_deg: float = info(33.0)  # safety only; forks bind first (~29.9 deg)
    # Chest shell (kinematic slabs).
    plinth_x: tuple = info((0.300, 0.620))
    plinth_y_half: float = info(0.215)
    plinth_h: float = info(0.030)
    front_x: tuple = info((0.306, 0.330))
    aperture_dy: float = info(0.0815)  # aperture y = drawer_y +- aperture_dy
    aperture_z_top: float = info(0.152)
    lintel_z_top: float = info(0.165)
    side_y: tuple = info((0.191, 0.215))
    rear_x: tuple = info((0.596, 0.620))
    roof_z: tuple = info((0.165, 0.185))
    post_r: float = info(0.012)
    post_h: float = info(0.100)
    drop_dx: float = info(-0.038)  # drop point, goal-drawer local x (exposed tray)
    contact_offset: float = info(0.0015)

    # Derived (filled in __post_init__).
    phi_max: float = field(default=None, init=False)  # rocker angle at the stops (rad)
    throw: float = field(default=None, init=False)  # pin x-throw = travel / 2
    cargo_half: float = field(default=None, init=False)
    open_x: float = field(default=None, init=False)  # fully-out drawer root x
    tray_rest_local_z: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.throw = self.travel / 2  # 0.055
        self.phi_max = math.asin(self.throw / self.arm_len)  # 0.4828 rad = 27.66 deg
        self.cargo_half = self.cargo_size / 2
        self.open_x = self.closed_x - self.travel  # 0.300
        # pudding centre resting on the tray floor, drawer frame
        self.tray_rest_local_z = self.floor_size[2] / 2 + self.cargo_half  # +0.028


def _quat_z(rad: float) -> tuple:
    return (math.cos(rad / 2), 0.0, 0.0, math.sin(rad / 2))


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("rocker_twin_drawers")
class RockerTwinDrawersScene(BaseScene):
    cfg: RockerTwinDrawersSceneCfg

    def __init__(self, cfg: RockerTwinDrawersSceneCfg | None = None) -> None:
        super().__init__(cfg or RockerTwinDrawersSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic chest shell (plinth, windowed front wall,
        sides, rear, roof), the two dynamic drawer trays and the rocker bar (compound
        children are authored in bind), and the pudding cube."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        slate = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.40, 0.48))
        dark = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.16, 0.16, 0.18))
        wood = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.38, 0.20))
        brown = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.23, 0.10))
        red = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.12, 0.10))
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        # post_step drives the drawers/rocker/cargo with external wrenches that do
        # NOT wake a sleeping body — sleep_threshold=0 keeps every plant live.
        live = dict(sleep_threshold=0.0, stabilization_threshold=0.0)
        dyn = dict(
            max_depenetration_velocity=0.5,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
            **live,
        )

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
        }
        # --- chest shell slabs (name: (pos, size, material)) ---
        px_lo, px_hi = c.plinth_x
        fx_lo, fx_hi = c.front_x
        fx_c, fx_t = (fx_lo + fx_hi) / 2, fx_hi - fx_lo
        sy_lo, sy_hi = c.side_y
        rx_lo, rx_hi = c.rear_x
        rz_lo, rz_hi = c.roof_z
        ph = c.plinth_h
        ap_lo = c.drawer_y - c.aperture_dy  # 0.0235
        ap_hi = c.drawer_y + c.aperture_dy  # 0.1865
        wall_z = (ph + c.aperture_z_top) / 2  # pillar centre z
        wall_h = c.aperture_z_top - ph
        shell_z = (ph + c.lintel_z_top) / 2
        shell_h = c.lintel_z_top - ph
        slabs = {
            # plinth the drawers/rocker hang over (joint body0; joint pair is
            # collision-filtered — drawer floors ride 4 mm above it by design)
            "plinth": (((px_lo + px_hi) / 2, 0.0, ph / 2),
                       (px_hi - px_lo, 2 * c.plinth_y_half, ph), dark),
            # windowed front wall: three pillars + full-width lintel; the two
            # apertures are exactly the drawers' passage holes
            "front_pillar_c": ((fx_c, 0.0, wall_z), (fx_t, 2 * ap_lo, wall_h), slate),
            "front_pillar_l": ((fx_c, (ap_hi + c.plinth_y_half) / 2, wall_z),
                               (fx_t, c.plinth_y_half - ap_hi, wall_h), slate),
            "front_pillar_r": ((fx_c, -(ap_hi + c.plinth_y_half) / 2, wall_z),
                               (fx_t, c.plinth_y_half - ap_hi, wall_h), slate),
            "front_lintel": ((fx_c, 0.0, (c.aperture_z_top + c.lintel_z_top) / 2),
                             (fx_t, 2 * c.plinth_y_half, c.lintel_z_top - c.aperture_z_top),
                             slate),
            "side_wall_l": (((fx_hi + rx_hi) / 2, (sy_lo + sy_hi) / 2, shell_z),
                            (rx_hi - fx_hi, sy_hi - sy_lo, shell_h), slate),
            "side_wall_r": (((fx_hi + rx_hi) / 2, -(sy_lo + sy_hi) / 2, shell_z),
                            (rx_hi - fx_hi, sy_hi - sy_lo, shell_h), slate),
            "rear_wall": (((rx_lo + rx_hi) / 2, 0.0, shell_z),
                          (rx_hi - rx_lo, 2 * c.plinth_y_half, shell_h), slate),
            "roof": (((px_lo + px_hi) / 2, 0.0, (rz_lo + rz_hi) / 2),
                     (px_hi - px_lo, 2 * c.plinth_y_half, rz_hi - rz_lo), slate),
        }
        for name, (pos, size, mat) in slabs.items():
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.title().replace("_", ""),
                spawn=sim_utils.CuboidCfg(
                    size=size, rigid_props=kin, collision_props=coll, visual_material=mat),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
            )
        # --- dynamic bodies (roots; children authored in bind) ---
        # Authored consistent assembly: LEFT drawer shut, RIGHT fully out, rocker at
        # -phi_max (reset re-poses per episode).
        for name, y, x in (("drawer_l", c.drawer_y, c.closed_x),
                           ("drawer_r", -c.drawer_y, c.open_x)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.title().replace("_", ""),
                spawn=sim_utils.CuboidCfg(
                    size=c.floor_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(**dyn),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.drawer_mass),
                    collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.50, dynamic_friction=0.45, restitution=0.0),
                    visual_material=wood,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, c.floor_z)),
            )
        out["rocker"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Rocker",
            spawn=sim_utils.CuboidCfg(
                size=c.bar_s,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(**dyn),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.rocker_mass),
                collision_props=coll,
                visual_material=red,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=c.pivot, rot=_quat_z(-self.cfg.phi_max)),
        )
        out["cargo"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cargo",
            spawn=sim_utils.CuboidCfg(
                size=(c.cargo_size,) * 3,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=0.05, angular_damping=0.05, **dyn),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.cargo_mass),
                collision_props=coll,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.50, dynamic_friction=0.45, restitution=0.05),
                visual_material=brown,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.10, 0.0, c.cargo_half + 0.002)),
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
                # every plant is driven by external wrenches every step
                "enable_external_forces_every_iteration": True,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n = env.num_envs
        dev = env.device
        self.drawer_l: RigidObject = env.iscene["drawer_l"]
        self.drawer_r: RigidObject = env.iscene["drawer_r"]
        self.rocker: RigidObject = env.iscene["rocker"]
        self.cargo: RigidObject = env.iscene["cargo"]
        self.env_origins = env.iscene.env_origins
        self._author_children()
        self._author_joints()
        # Episode state: 0 = LEFT (+y) drawer is the shut goal, 1 = RIGHT (-y).
        self.target_side = torch.zeros(n, dtype=torch.long, device=dev)
        self.open_latch = torch.zeros(n, device=dev)
        self.load_latch = torch.zeros(n, device=dev)
        # External drive inputs (solve.py and smoke probes write; post_step consumes
        # and owns every wrench slot — never call set_external_force_and_torque
        # directly).
        self.drawer_drive = torch.zeros(n, 2, device=dev)  # world-X force [left, right] (N)
        self.cargo_drive = torch.zeros(n, 3, device=dev)  # WORLD-frame force on the pudding
        # Wrench-frame reference: the forge pod rotates applied wrenches by the
        # body's rotation-since-write; pre-encoding with q_ref * conj(q_now) undoes
        # it. Drawers never rotate (prismatic, identity quat) so only the cargo needs
        # this. Updated at reset and via mark_cargo_ref() after teleports.
        self._cargo_qref = torch.zeros(n, 4, device=dev)
        self._cargo_qref[:, 0] = 1.0

    def _author_children(self) -> None:
        """Compound collision children (env_0 only — env_1.. compose from env_0 by
        reference; authored idempotently): per drawer the front panel, tray walls and
        tail fork plates; on the rocker the two hanging pins; on the plinth a centre
        post that visually carries the rocker bar."""
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath("/World/envs/env_0/DrawerL/panel").IsValid():
            return

        def _prep(prim) -> None:
            UsdPhysics.CollisionAPI.Apply(prim)
            px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            px.CreateContactOffsetAttr(c.contact_offset)
            px.CreateRestOffsetAttr(0.0)

        def box(parent: str, name: str, center: tuple, size: tuple, color: tuple) -> None:
            cube = UsdGeom.Cube.Define(stage, f"/World/envs/env_0/{parent}/{name}")
            cube.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(cube.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(*center))
            xf.AddScaleOp().Set(Gf.Vec3f(*size))
            cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            _prep(cube.GetPrim())

        def cyl(parent: str, name: str, center: tuple, radius: float, height: float,
                color: tuple) -> None:
            cy = UsdGeom.Cylinder.Define(stage, f"/World/envs/env_0/{parent}/{name}")
            cy.CreateRadiusAttr(radius)
            cy.CreateHeightAttr(height)
            cy.CreateAxisAttr("Z")
            cy.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                                 Gf.Vec3f(radius, radius, height / 2)])
            xf = UsdGeom.Xformable(cy.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(*center))
            cy.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            _prep(cy.GetPrim())

        wood = (0.55, 0.38, 0.20)
        teal = (0.10, 0.55, 0.55)
        orange = (0.85, 0.45, 0.10)
        red = (0.85, 0.12, 0.10)
        dark = (0.16, 0.16, 0.18)
        for parent, s, panel_col in (("DrawerL", +1.0, teal), ("DrawerR", -1.0, orange)):
            box(parent, "panel", c.panel_c, c.panel_s, panel_col)
            box(parent, "wall_yp", (0.0, +c.traywall_y, 0.0425), c.traywall_s, wood)
            box(parent, "wall_yn", (0.0, -c.traywall_y, 0.0425), c.traywall_s, wood)
            box(parent, "wall_rear", c.trayrear_c, c.trayrear_s, wood)
            # fork centreline is offset toward the rocker pin's arc (sign mirrors)
            box(parent, "fork_front", (c.fork_front_x, s * c.fork_dy, c.fork_z),
                c.fork_s, wood)
            box(parent, "fork_rear", (c.fork_rear_x, s * c.fork_dy, c.fork_z),
                c.fork_s, wood)
        for name, sy in (("pin_l", +1.0), ("pin_r", -1.0)):
            cyl("Rocker", name, (0.0, sy * c.arm_len, c.pin_dz), c.pin_r, c.pin_h, red)
        # cosmetic pivot post under the bar (child of the plinth; the plinth<->rocker
        # joint pair is collision-filtered anyway)
        px_c = (c.plinth_x[0] + c.plinth_x[1]) / 2
        cyl("Plinth", "post",
            (c.pivot[0] - px_c, 0.0, c.post_h / 2 + c.plinth_h / 2), c.post_r,
            c.post_h, dark)

    def _author_joints(self) -> None:
        """Per env: two X-axis prismatic joints plinth -> drawer (limits [-travel, 0]
        metres; 0 = shut flush, joint pairs never collide — the stops are the joint's
        own limits) and one free-ish Z revolute plinth -> rocker (generous +-33 deg
        safety limits; the captive forks bind first at ~29.9 deg)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        px_c = (c.plinth_x[0] + c.plinth_x[1]) / 2
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            for name, prim, y in (("slide_l", "DrawerL", c.drawer_y),
                                  ("slide_r", "DrawerR", -c.drawer_y)):
                j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/{name}")
                j.CreateBody0Rel().SetTargets([f"{base}/Plinth"])
                j.CreateBody1Rel().SetTargets([f"{base}/{prim}"])
                j.CreateCollisionEnabledAttr(False)
                j.CreateAxisAttr("X")
                # anchor = the drawer's SHUT root pose -> q = 0 shut, q = -travel out
                j.CreateLocalPos0Attr(
                    Gf.Vec3f(c.closed_x - px_c, y, c.floor_z - c.plinth_h / 2))
                j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLowerLimitAttr(-c.travel)
                j.CreateUpperLimitAttr(0.0)
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/rocker_pivot")
            j.CreateBody0Rel().SetTargets([f"{base}/Plinth"])
            j.CreateBody1Rel().SetTargets([f"{base}/Rocker"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(
                Gf.Vec3f(c.pivot[0] - px_c, 0.0, c.pivot[2] - c.plinth_h / 2))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-c.rocker_limit_deg)
            j.CreateUpperLimitAttr(c.rocker_limit_deg)

    # ----- reset ------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample WHICH side is the shut goal drawer, pose goal shut /
        twin fully out / rocker at the matching stop (a consistent assembly — pins
        centred in their fork gaps), drop the pudding on the floor with xy jitter and
        free yaw, clear latches and drives."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        self.open_latch[env_ids] = 0.0
        self.load_latch[env_ids] = 0.0
        self.drawer_drive[env_ids] = 0.0
        self.cargo_drive[env_ids] = 0.0

        # --- goal side (torch.rand comparison — first randint after a fresh
        # manual_seed is near-constant) ---
        side = (torch.rand(m, device=dev) < 0.5).long()  # 0 = left goal, 1 = right
        self.target_side[env_ids] = side
        # rocker at -phi_max <=> left shut / right out; +phi_max <=> mirrored
        phi = torch.where(side == 0, -c.phi_max, c.phi_max)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.pivot[0]
        st[:, 2] = c.pivot[2]
        st[:, 3] = torch.cos(phi / 2)
        st[:, 6] = torch.sin(phi / 2)
        st[:, 0:3] += origin
        self.rocker.write_root_state_to_sim(st, env_ids)
        # drawers: goal at ext 0 (root closed_x), twin at ext travel (root open_x)
        for body, y, shut in ((self.drawer_l, c.drawer_y, side == 0),
                              (self.drawer_r, -c.drawer_y, side == 1)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = torch.where(shut, torch.full_like(phi, c.closed_x),
                                   torch.full_like(phi, c.open_x))
            st[:, 1] = y
            st[:, 2] = c.floor_z
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- pudding on the floor: xy jitter + free yaw ---
        x = c.cargo_x[0] + (c.cargo_x[1] - c.cargo_x[0]) * torch.rand(m, device=dev)
        y = (torch.rand(m, device=dev) * 2 - 1) * c.cargo_y_half
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = x
        st[:, 1] = y
        st[:, 2] = c.cargo_half + 0.002
        st[:, 3] = torch.cos(yaw / 2)
        st[:, 6] = torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.cargo.write_root_state_to_sim(st, env_ids)
        # Wrench-frame reference = the orientation just WRITTEN (readback right after
        # a write can be stale until the next step).
        self._cargo_qref[env_ids] = st[:, 3:7].clone()

    def mark_cargo_ref(self) -> None:
        """Capture the pudding's current orientation as the wrench-frame reference.
        Call after any write_root_state teleport of the cargo (and after >= 1 step so
        the readback is fresh)."""
        self._cargo_qref = self.cargo.data.root_quat_w.clone()

    # ----- readings ---------------------------------------------------------------------------
    def ext(self) -> torch.Tensor:
        """(N, 2) drawer extensions [left, right] in metres (0 = shut flush,
        travel = fully out; drawers only ever translate along -x to open)."""
        c = self.cfg
        el = c.closed_x - (self.drawer_l.data.root_pos_w[:, 0] - self.env_origins[:, 0])
        er = c.closed_x - (self.drawer_r.data.root_pos_w[:, 0] - self.env_origins[:, 0])
        return torch.stack((el, er), dim=1)

    def target_ext(self) -> torch.Tensor:
        """(N,) extension of the GOAL drawer (the one that starts shut)."""
        e = self.ext()
        return torch.where(self.target_side == 0, e[:, 0], e[:, 1])

    def twin_ext(self) -> torch.Tensor:
        """(N,) extension of the NON-goal drawer."""
        e = self.ext()
        return torch.where(self.target_side == 0, e[:, 1], e[:, 0])

    def phi(self) -> torch.Tensor:
        """(N,) rocker pivot angle in rad (0 = both drawers half-out; the bar only
        ever rotates about z)."""
        q = self.rocker.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 3], q[:, 0])

    def _target_root(self) -> torch.Tensor:
        """(N, 3) GOAL drawer root world position."""
        return torch.where((self.target_side == 0).unsqueeze(1),
                           self.drawer_l.data.root_pos_w,
                           self.drawer_r.data.root_pos_w)

    def cargo_local(self) -> torch.Tensor:
        """(N, 3) pudding centre in the GOAL drawer's frame (root = tray floor
        centre; drawers keep identity orientation on their prismatic slides, so a
        plain offset is exact — and the tray judges its cargo identically shut or
        out)."""
        return self.cargo.data.root_pos_w - self._target_root()

    def in_tray(self) -> torch.Tensor:
        """(N,) bool, geometric: pudding centre inside the GOAL tray band (drawer
        frame). No velocity clause (used while the drawer is moving)."""
        c = self.cfg
        loc = self.cargo_local()
        return (loc[:, 0] > c.tray_x_lo) & (loc[:, 0] < c.tray_x_hi) \
            & (loc[:, 1].abs() < c.tray_y_tol) \
            & (loc[:, 2] > c.tray_z_lo) & (loc[:, 2] < c.tray_z_hi)

    def settled(self) -> torch.Tensor:
        """(N,) bool: pudding AND both drawers at rest."""
        c = self.cfg
        return (self.cargo.data.root_lin_vel_w.norm(dim=-1) < c.settle_cargo) \
            & (self.drawer_l.data.root_lin_vel_w[:, 0].abs() < c.settle_drawer) \
            & (self.drawer_r.data.root_lin_vel_w[:, 0].abs() < c.settle_drawer)

    def success(self) -> torch.Tensor:
        """(N,) bool: pudding contained in the GOAL drawer's tray, that drawer shut
        flush against its stop, everything settled and finite (current, physical
        state). The twin drawer necessarily stands open — it is not judged."""
        fin = torch.isfinite(self.cargo.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.ext()).all(dim=-1)
        return self.in_tray() & (self.target_ext() < self.cfg.closed_tol) \
            & self.settled() & fin

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15 * open_latch + 0.35 * load_latch + 0.50 *
        current success. Exactly 1.0 iff success() (success => in-tray & slow =>
        load latch; load forces the open latch); ~0 for the null policy; re-opening
        a finished chest drops back to the latched 0.50."""
        return (0.15 * self.open_latch + 0.35 * self.load_latch
                + 0.50 * self.success().float())

    # ----- step-coupled mechanics (every substep) ---------------------------------------------
    def post_step(self) -> None:
        """Drawer plants: viscous slide friction + the `drawer_drive` world-X force
        buffers (drawers never rotate, so no wrench-frame encoding is needed). Rocker
        plant: viscous pivot friction (torque about world z — invariant under the
        bar's own yaw). Cargo plant: the `cargo_drive` WORLD force, pre-encoded with
        q_ref * conj(q_now) against the pod's rotation-since-write wrench frame.
        Then latch rubric progress."""
        from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

        n = self.env.num_envs
        dev = self.env.device
        c = self.cfg
        zero3 = torch.zeros(n, 1, 3, device=dev)
        for i, body in ((0, self.drawer_l), (1, self.drawer_r)):
            f = torch.zeros(n, 1, 3, device=dev)
            f[:, 0, 0] = self.drawer_drive[:, i] \
                - c.drawer_damping * body.data.root_lin_vel_w[:, 0]
            body.set_external_force_and_torque(f, zero3.clone())
        tq = torch.zeros(n, 1, 3, device=dev)
        tq[:, 0, 2] = -c.rocker_damping * self.rocker.data.root_ang_vel_w[:, 2]
        self.rocker.set_external_force_and_torque(zero3.clone(), tq)

        q_rel = quat_mul(self._cargo_qref, quat_conjugate(self.cargo.data.root_quat_w))
        f_in = quat_apply(q_rel, self.cargo_drive)
        self.cargo.set_external_force_and_torque(f_in.unsqueeze(1), zero3.clone())

        # --- rubric latches (NaN-guarded: a diverged substep earns NO progress) ---
        cargo_slow = self.cargo.data.root_lin_vel_w.norm(dim=-1) < 0.08
        loaded = (self.in_tray() & cargo_slow).float()
        opened = (self.target_ext() >= c.open_thresh).float()
        for t in (loaded, opened):
            torch.nan_to_num_(t, nan=0.0, posinf=0.0, neginf=0.0)
        self.load_latch = torch.maximum(self.load_latch, loaded)
        # a load proves the goal drawer was opened first (physically forced order)
        self.open_latch = torch.maximum(torch.maximum(self.open_latch, opened),
                                        self.load_latch)

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"drawer_l": self.drawer_l, "drawer_r": self.drawer_r,
                  "rocker": self.rocker, "cargo": self.cargo}
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("target_side", "open_latch", "load_latch",
                               "drawer_drive", "cargo_drive", "_cargo_qref")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"drawer_l": self.drawer_l, "drawer_r": self.drawer_r,
                  "rocker": self.rocker, "cargo": self.cargo}
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A slate-gray chest stands on a low plinth ahead of you, its front face "
            f"(the side facing you, at x = {c.front_x[0]:.3f}) pierced by two "
            f"drawer apertures side by side at y = +{c.drawer_y:.3f} and "
            f"-{c.drawer_y:.3f}. It holds two sliding drawers with smooth, KNOBLESS "
            f"fronts: the LEFT (+y) drawer's panel is TEAL, the RIGHT (-y) one's is "
            f"ORANGE. A shut drawer finishes perfectly flush with the chest face — "
            f"nothing protrudes to hook, pinch or pull — and it already rests on its "
            f"inner hard stop, so pressing it does nothing. The chest is roofed and "
            f"walled; the only openings are the two drawer apertures, which the "
            f"drawers themselves fill. Hidden inside, an overhead red walking beam "
            f"couples the two drawers ANTI-PHASE through pins riding forks on their "
            f"tails: the two extensions always sum to the {c.travel * 100:.0f} cm "
            f"travel, so at most one drawer can be shut at a time — pushing the open "
            f"drawer IN drives the shut one OUT by the same amount, and vice versa. "
            f"One drawer starts fully shut and one fully out (WHICH side is which is "
            f"random each episode). The shut one is the GOAL drawer. A brown "
            f"{c.cargo_size * 100:.1f} cm chocolate-pudding cube lies on the floor "
            f"between you and the chest.\n"
            f"Goal: get the pudding into the GOAL drawer's tray and end with that "
            f"drawer shut flush again. The only way to open it is the transmission: "
            f"push the OTHER (open) drawer fully in, which slides the goal drawer's "
            f"tray out through its aperture. Drop the pudding into the exposed tray, "
            f"then push the goal drawer shut — the twin necessarily pops back out; "
            f"that is fine, only the goal drawer is judged. Dropping the pudding "
            f"onto the roof, into the WRONG (initially open) drawer, or leaving the "
            f"goal drawer more than {c.closed_tol * 1000:.0f} mm from flush does "
            f"not count."
        )

    def instruction(self) -> str:
        return (
            "Two knobless drawers share a hidden anti-phase linkage: pushing one in "
            "slides the other out, and at most one can be shut at a time. Put the "
            "brown pudding cube into the drawer that STARTS SHUT and end with that "
            "drawer shut flush again: push the open drawer fully in to drive the "
            "shut one out, drop the pudding into its exposed tray, then push it "
            "shut (the other drawer will pop back out — that is expected). Loading "
            "the initially-open drawer instead, or leaving the goal drawer ajar, "
            "fails the task."
        )


# ----- runnable env: scene physics only (NullRobot smoke) -> "simgen.rocker_twin_drawers" ------
register_env("simgen", lambda: EnvCfg(scene="rocker_twin_drawers", robot="null"))

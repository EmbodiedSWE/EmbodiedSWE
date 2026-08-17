"""CarouselDispatchScene — rotate a free-spinning carousel until its cargo bay faces the
RED garage, then push the cargo off the deck so it slides fully inside.

Derived from rlbench/reach_and_drag ("reach and drag": grasp a loose stick and use it as
a tool to DRAG a cube across the open table onto a colored target square among
distractors), but the MANIPULATION MODEL is replaced wholesale. The seed's plan is
tool-mediated planar dragging: acquire an elongated tool, sweep the cube freely across
the table to a flat target. Here there is NO tool and NO free dragging: the cargo (a
90 mm cube, wider than a parallel jaw) rides in a rail-guarded BAY on a free-spinning
CAROUSEL platter, and the only way to any garage is (1) ROTATE the mechanism — push its
rim/handle so the platter turns on its pivot bearing — until the bay's outward opening
faces the one correct garage, then (2) EJECT the cargo radially outward through the
garage's mouth. The garages are covered (roof) and plinth-mounted (floor at deck
height, a 66 mm blank face below), so the cube can never be dragged along the ground
into one, never dropped in from above, and never carried (it does not fit the jaw): the
carousel is load-bearing, and the order (align, then eject) is physically forced.

Assets are fully procedural (compound-spawner pattern; child colliders of one body
never self-collide):
  - pedestal: KINEMATIC square plate (330 x 330 x 20 mm) with a square bearing WELL
    (inner 66 mm, walls 36 mm tall) at its centre.
  - platter: DYNAMIC compound. A deck disc (r 160 mm, 12 mm thick) resting on the well
    wall tops; a round stub (r 30 mm) hanging into the well (2 mm play — the radial
    bearing; deck-on-wall-top is the thrust bearing, so dry friction both centres the
    platter and brakes it). On the deck: the cargo BAY (two rails + an inner backstop,
    open ONLY radially outward) at local +x, and a yellow HANDLE PEG (r 12 mm, 80 mm
    tall) at local -x for pushing the platter around.
  - cargo: DYNAMIC red cube, 90 mm (> the 80 mm Franka jaw: pushable, not graspable).
  - garages: three KINEMATIC covered docks (RED / GREEN / BLUE), mouths facing the
    platter, floor top 2 mm below deck top, interior 150 mm wide x 114 mm tall, roof
    on top, solid plinth below. Which color sits at which of the three azimuth slots
    (base-local 0 / 90 / 270 deg) is SHUFFLED per episode.

Per-episode randomization (readback-verifiable): pedestal xy jitter + yaw, garage color
permutation over the slots, platter start angle (guaranteed 30..150 deg away from the
red garage, either sign), cargo jitter in the bay.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve trajectory):
  0.25 * aligned   — the bay ever faces the red garage (error < `align_tol_deg`) with
                     the cargo still in the bay and the platter near-still (latched;
                     a fast flyby or an empty bay does not count)
  0.35 * delivered — the cargo centre ever inside the red garage past the full-inside
                     plane (latched; reachable only through the mouth)
  1.0 iff success() — cargo settled ON the red garage floor, fully inside, everything
                     finite. Non-success capped at 0.60.

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


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

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


def _add_cyl(stage, path: str, *, center, radius, height, color, collide: Callable):
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr("Z")
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())
    return cyl.GetPrim()


def _spawn_pedestal(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC pedestal: square plate + 4 bearing-well walls (inner square 66 mm,
    z 0.020..0.056). Local origin: plate footprint centre on the ground."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_box(stage, f"{prim_path}/plate", center=(0.0, 0.0, c.plate_h / 2),
             size=(2 * c.plate_hw, 2 * c.plate_hw, c.plate_h), color=c.plate_color,
             collide=collide)
    wt, wi, wh = c.well_t, c.well_inner, c.well_wall_h  # thickness, inner half-w, height
    zc = c.plate_h + wh / 2
    _add_box(stage, f"{prim_path}/well_n", center=(0.0, wi + wt / 2, zc),
             size=(2 * wi + 2 * wt, wt, wh), color=c.well_color, collide=collide)
    _add_box(stage, f"{prim_path}/well_s", center=(0.0, -(wi + wt / 2), zc),
             size=(2 * wi + 2 * wt, wt, wh), color=c.well_color, collide=collide)
    _add_box(stage, f"{prim_path}/well_e", center=(wi + wt / 2, 0.0, zc),
             size=(wt, 2 * wi, wh), color=c.well_color, collide=collide)
    _add_box(stage, f"{prim_path}/well_w", center=(-(wi + wt / 2), 0.0, zc),
             size=(wt, 2 * wi, wh), color=c.well_color, collide=collide)
    return root


def _spawn_platter(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC platter compound. Local origin: deck disc CENTRE (mid-thickness), so the
    authored MassAPI mass (which pins the CoM at the body origin) puts the CoM on the
    spin axis at deck height. Children: deck disc, bearing stub (below), bay rails +
    backstop and the handle peg (above)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.05)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    dt2 = c.deck_t / 2
    _add_cyl(stage, f"{prim_path}/deck", center=(0.0, 0.0, 0.0),
             radius=c.deck_r, height=c.deck_t, color=c.deck_color, collide=collide)
    _add_cyl(stage, f"{prim_path}/stub", center=(0.0, 0.0, -dt2 - c.stub_len / 2),
             radius=c.stub_r, height=c.stub_len, color=c.deck_color, collide=collide)
    # bay: backstop (inner) + two rails; open radially outward (local +x) only
    rz = dt2 + c.rail_h / 2
    _add_box(stage, f"{prim_path}/backstop", center=(c.bay_x0 - 0.005, 0.0, rz),
             size=(0.010, 2 * c.bay_hw + 0.020, c.rail_h), color=c.rail_color,
             collide=collide)
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/rail_{tag}",
                 center=((c.bay_x0 + c.bay_x1) / 2, sgn * (c.bay_hw + 0.005), rz),
                 size=(c.bay_x1 - c.bay_x0, 0.010, c.rail_h), color=c.rail_color,
                 collide=collide)
    _add_cyl(stage, f"{prim_path}/handle",
             center=(-c.handle_x, 0.0, dt2 + c.handle_h / 2),
             radius=c.handle_r, height=c.handle_h, color=c.handle_color, collide=collide)
    return root


def _spawn_dock(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC covered garage. Local origin: footprint centre on the ground; the
    MOUTH faces local -x. Solid plinth (floor top at `plinth_top`, blank face below),
    two side walls, back wall, roof."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    hx = c.half_depth                      # footprint half-depth (x)
    wall_h = c.roof_z - c.plinth_top       # interior height
    zc = (c.roof_z + c.plinth_top) / 2
    _add_box(stage, f"{prim_path}/plinth", center=(0.0, 0.0, c.plinth_top / 2),
             size=(2 * hx, 2 * c.y_half + 0.020, c.plinth_top), color=c.color,
             collide=collide)
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/wall_{tag}", center=(0.0, sgn * (c.y_half + 0.005), zc),
                 size=(2 * hx, 0.010, wall_h), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/back", center=(c.x_back + 0.006, 0.0, zc),
             size=(0.012, 2 * c.y_half, wall_h), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/roof", center=(0.0, 0.0, c.roof_z + 0.005),
             size=(2 * hx, 2 * c.y_half + 0.020, 0.010), color=c.color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "pedestal" not in _SPAWNER_CACHE:

        @configclass
        class PedestalSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pedestal)
            plate_hw: float = 0.165
            plate_h: float = 0.020
            well_inner: float = 0.033
            well_t: float = 0.010
            well_wall_h: float = 0.036
            plate_color: tuple = (0.35, 0.36, 0.40)
            well_color: tuple = (0.25, 0.26, 0.30)
            contact_offset: float = 0.002

        @configclass
        class PlatterSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_platter)
            deck_r: float = 0.160
            deck_t: float = 0.012
            stub_r: float = 0.030
            stub_len: float = 0.034
            bay_x0: float = 0.060
            bay_x1: float = 0.150
            bay_hw: float = 0.049
            rail_h: float = 0.030
            handle_x: float = 0.135
            handle_r: float = 0.012
            handle_h: float = 0.080
            mass: float = 1.5
            deck_color: tuple = (0.55, 0.56, 0.60)
            rail_color: tuple = (0.30, 0.31, 0.35)
            handle_color: tuple = (0.95, 0.82, 0.10)
            contact_offset: float = 0.002

        @configclass
        class DockSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_dock)
            half_depth: float = 0.085
            y_half: float = 0.075
            plinth_top: float = 0.066
            x_back: float = 0.070
            roof_z: float = 0.180
            color: tuple = (0.8, 0.1, 0.1)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(pedestal=PedestalSpawnerCfg, platter=PlatterSpawnerCfg,
                              dock=DockSpawnerCfg)
    return _SPAWNER_CACHE


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


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CarouselDispatchSceneCfg(BaseCfg):
    """Config for `CarouselDispatchScene`. The full-inside plane (`inside_x_min`) is
    honest by construction: a cargo centre past it has its rear face at/behind the
    garage mouth plane, i.e. the cube is entirely within the covered interior; the
    interior is reachable only through the mouth (roof above, walls aside, solid
    plinth below), which sits at deck height."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    align_tol_deg: float = tunable(8.0)   # bay-to-red-garage azimuth error for `aligned`
    spin_still: float = tunable(0.30)     # max |platter w_z| (rad/s) for `aligned` (no flybys)
    settle_speed: float = tunable(0.05)   # max cargo |lin vel| when judging (m/s)
    settle_spin: float = tunable(0.50)    # max platter |w_z| when judging (rad/s)
    inside_x_min: float = tunable(-0.030)  # dock-frame x threshold: cargo centre past this
    # = rear face at/behind the mouth plane (mouth -0.085, cargo half 0.045)
    rest_z_max: float = tunable(0.135)    # dock-frame cargo centre max z for "on the floor"

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    base_jitter: float = tunable(0.025)   # pedestal xy jitter (+/- m)
    base_yaw_deg: float = tunable(12.0)   # pedestal yaw (+/- deg)
    start_err_min_deg: float = tunable(30.0)   # platter start angle away from the red garage
    start_err_max_deg: float = tunable(150.0)
    cargo_jitter: float = tunable(0.004)  # cargo xy jitter in the bay (+/- m)
    shuffle_docks: bool = tunable(True)   # per-episode garage color permutation

    # --- info: layout ----------------------------------------------------------------------------
    base_pos: tuple = info((0.40, 0.0))   # pedestal centre on the ground (nominal)
    slot_az_deg: tuple = info((0.0, 90.0, 270.0))  # garage slots, base-local azimuths
    r_dock: float = info(0.255)           # garage footprint centre radius from the spin axis
    # --- info: platter / bearing (rest heights MEASURED by the geometry below) -------------------
    deck_r: float = info(0.160)
    deck_t: float = info(0.012)
    z_platter: float = info(0.062)        # deck-centre rest height (wall top 0.056 + deck_t/2)
    platter_mass: float = info(1.5)
    bay_x0: float = info(0.060)           # bay inner edge (platter local x)
    bay_x1: float = info(0.150)           # bay outer (open) edge
    bay_hw: float = info(0.049)           # bay half-width (rail inner faces)
    handle_x: float = info(0.135)         # handle peg radius (local -x side)
    # --- info: cargo -----------------------------------------------------------------------------
    cargo_size: float = info(0.090)       # cube edge (> the 80 mm Franka jaw: push-only)
    cargo_mass: float = info(0.25)
    cargo_slot_x: float = info(0.105)     # cargo centre radius in the bay
    cargo_color: tuple = info((0.85, 0.10, 0.10))
    # --- info: garages (dock local frame: origin footprint centre, mouth faces -x) ---------------
    dock_mouth_x: float = info(-0.085)    # mouth plane (front face)
    dock_x_back: float = info(0.070)      # back wall inner face
    dock_y_half: float = info(0.075)      # side wall inner faces
    plinth_top: float = info(0.066)       # garage floor top (2 mm below deck top 0.068)
    roof_z: float = info(0.180)           # roof underside
    dock_colors: tuple = info((("red", (0.80, 0.08, 0.08)),
                               ("green", (0.08, 0.62, 0.15)),
                               ("blue", (0.10, 0.25, 0.85))))
    contact_offset: float = info(0.002)
    # rubric weights (0.25 + 0.35 = 0.60 = the non-success cap)
    w_aligned: float = info(0.25)
    w_delivered: float = info(0.35)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("cargo_carousel")
class CarouselDispatchScene(BaseScene):
    cfg: CarouselDispatchSceneCfg

    def __init__(self, cfg: CarouselDispatchSceneCfg | None = None) -> None:
        super().__init__(cfg or CarouselDispatchSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        bx, by = c.base_pos
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
            "pedestal": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pedestal",
                spawn=cls["pedestal"](contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(bx, by, 0.0)),
            ),
            "platter": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Platter",
                spawn=cls["platter"](mass=c.platter_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(bx, by, c.z_platter + 0.002)),
            ),
            "cargo": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cargo",
                spawn=sim_utils.CuboidCfg(
                    size=(c.cargo_size,) * 3,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cargo_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.05,
                        sleep_threshold=0.0, stabilization_threshold=0.0,
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=4),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.5, dynamic_friction=0.4, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.cargo_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(bx + c.cargo_slot_x, by, c.z_platter + c.deck_t / 2 + c.cargo_size / 2 + 0.004)),
            ),
        }
        for i, (name, rgb) in enumerate(c.dock_colors):
            az = math.radians(c.slot_az_deg[i])
            out[f"dock_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Dock_" + name,
                spawn=cls["dock"](color=rgb, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(bx + c.r_dock * math.cos(az), by + c.r_dock * math.sin(az), 0.0),
                    rot=(math.cos(az / 2), 0.0, 0.0, math.sin(az / 2))),
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

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        self.pedestal: RigidObject = env.iscene["pedestal"]
        self.platter: RigidObject = env.iscene["platter"]
        self.cargo: RigidObject = env.iscene["cargo"]
        self.docks: dict[str, RigidObject] = {
            name: env.iscene[f"dock_{name}"] for name, _rgb in c.dock_colors}
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # red_slot[e]: index into slot_az_deg where the RED garage stands this episode
        self.red_slot = torch.zeros(n, dtype=torch.long, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._aligned = torch.zeros(n, dtype=torch.bool, device=dev)
        self._delivered = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the pedestal (xy jitter + yaw), shuffle the garage
        colors over the three slots, drop the platter onto its bearing at a start
        angle 30..150 deg (either sign) away from the red garage, seat the cargo in
        the bay (xy jitter), clear the latches."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- pedestal: kinematic, nominal pos + jitter + yaw ---
        byaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.base_yaw_deg)
        q_base = _qz(byaw)
        bp = torch.zeros(m, 3, device=dev)
        bp[:, 0] = c.base_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.base_jitter
        bp[:, 1] = c.base_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.base_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = bp + origin
        st[:, 3:7] = q_base
        self.pedestal.write_root_state_to_sim(st, env_ids)

        # --- garages: color permutation over the slots (argsort-of-rand: no first-randint
        # degeneracy), placed radially, mouths facing the spin axis ---
        if c.shuffle_docks:
            perm = torch.rand(m, 3, device=dev).argsort(dim=1)  # perm[e, i] = slot of color i
        else:
            perm = torch.arange(3, device=dev).expand(m, 3).clone()
        self.red_slot[env_ids] = perm[:, 0]
        slot_az = torch.tensor([math.radians(a) for a in c.slot_az_deg], device=dev)
        for i, (name, _rgb) in enumerate(c.dock_colors):
            az = byaw + slot_az[perm[:, i]]
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = bp[:, 0] + c.r_dock * torch.cos(az)
            st[:, 1] = bp[:, 1] + c.r_dock * torch.sin(az)
            st[:, 3:7] = _qz(az)
            st[:, 0:3] += origin
            self.docks[name].write_root_state_to_sim(st, env_ids)

        # --- platter: on the bearing (2 mm drop), start angle away from the red garage ---
        az_red = byaw + slot_az[perm[:, 0]]
        sign = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        off = math.radians(c.start_err_min_deg) + torch.rand(m, device=dev) * \
            math.radians(c.start_err_max_deg - c.start_err_min_deg)
        pyaw = az_red + sign * off
        q_plat = _qz(pyaw)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = bp[:, 0:2]
        st[:, 2] = c.z_platter + 0.002
        st[:, 3:7] = q_plat
        st[:, 0:3] += origin
        self.platter.write_root_state_to_sim(st, env_ids)

        # --- cargo: in the bay (platter frame), xy jitter ---
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.cargo_slot_x + (torch.rand(m, device=dev) * 2 - 1) * c.cargo_jitter
        loc[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.cargo_jitter
        loc[:, 2] = c.deck_t / 2 + c.cargo_size / 2 + 0.004
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = bp[:, 0:2]
        st[:, 2] = c.z_platter + 0.002
        st[:, 0:3] += quat_apply(q_plat, loc)
        st[:, 3:7] = q_plat
        self.cargo.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._aligned[env_ids] = False
        self._delivered[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "pedestal": self.pedestal.data.root_state_w[env_ids].clone(),
            "platter": self.platter.data.root_state_w[env_ids].clone(),
            "cargo": self.cargo.data.root_state_w[env_ids].clone(),
            "docks": {n: b.data.root_state_w[env_ids].clone() for n, b in self.docks.items()},
            "red_slot": self.red_slot[env_ids].clone(),
            "aligned": self._aligned[env_ids].clone(),
            "delivered": self._delivered[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.pedestal.write_root_state_to_sim(state["pedestal"], env_ids)
        self.platter.write_root_state_to_sim(state["platter"], env_ids)
        self.cargo.write_root_state_to_sim(state["cargo"], env_ids)
        for n, b in self.docks.items():
            b.write_root_state_to_sim(state["docks"][n], env_ids)
        self.red_slot[env_ids] = state["red_slot"]
        self._aligned[env_ids] = state["aligned"]
        self._delivered[env_ids] = state["delivered"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A CAROUSEL stands on a low pedestal on the ground: a grey platter disc "
            f"({2 * c.deck_r * 1000:.0f} mm across, deck top ~68 mm up) that spins freely "
            f"about its centre on a pivot bearing — push its rim or the YELLOW HANDLE PEG "
            f"(12 mm thick, 80 mm tall) near the rim and it rotates; friction stops it when "
            f"you stop pushing. On the deck sits one RED CARGO CUBE "
            f"({c.cargo_size * 1000:.0f} mm — too wide for a parallel-jaw gripper: it must "
            f"be PUSHED, it cannot be picked up), held in a shallow BAY between two low "
            f"rails and an inner backstop; the bay is open on ONE side only, facing "
            f"radially OUTWARD, directly opposite the handle peg. Around the carousel "
            f"stand three covered GARAGES — one RED, one GREEN, one BLUE; which garage "
            f"stands where changes every episode, so look at the colors. Each garage's "
            f"open mouth faces the platter, its floor is level with the deck (a blank face "
            f"below — a cube on the ground cannot enter), and its roof prevents dropping "
            f"anything in from above.\n"
            f"Goal: deliver the red cargo cube INTO THE RED GARAGE. Rotate the carousel "
            f"(push the handle peg or the rim; rotate GENTLY — spinning fast flings the "
            f"cargo off) until the bay's open side points at the red garage's mouth, then "
            f"push the cargo radially outward: it slides out of the bay, across the small "
            f"gap, through the mouth, and into the garage. The cube must end FULLY inside "
            f"the red garage (its trailing face past the mouth plane), resting on the "
            f"garage floor, at rest. Pushing it into the green or blue garage, dropping it "
            f"on the ground, or leaving it sticking out of the mouth all fail."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Rotate the carousel gently until its cargo bay opening faces the red "
            "garage, then push the red cargo cube out of the bay so it slides through "
            "the mouth and rests fully inside the red garage. Delivering it into any "
            "other garage or leaving it outside or half-in fails."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _dock_local(self, name: str, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the (kinematic, live-read) dock frame, (N,3) -> (N,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        d = self.docks[name]
        return quat_apply_inverse(d.data.root_quat_w, pos_w - d.data.root_pos_w)

    def _platter_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.platter.data.root_quat_w,
                                  pos_w - self.platter.data.root_pos_w)

    def align_err_rad(self) -> torch.Tensor:
        """(N,) signed angle (rad) from the bay's outward direction (platter local +x)
        to the red garage's direction, about world z. Rotating the platter by +err
        aligns the bay with the red mouth."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ex = torch.tensor([1.0, 0.0, 0.0], device=self.env.device).expand(n, 3)
        tray = quat_apply(self.platter.data.root_quat_w, ex)[:, :2]
        tray = tray / tray.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        rd = (self.docks["red"].data.root_pos_w[:, :2]
              - self.platter.data.root_pos_w[:, :2])
        rd = rd / rd.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return torch.atan2(tray[:, 0] * rd[:, 1] - tray[:, 1] * rd[:, 0],
                           (tray * rd).sum(-1))

    def cargo_in_bay(self) -> torch.Tensor:
        """(N,) bool: cargo centre inside the bay volume, platter frame."""
        c = self.cfg
        loc = self._platter_local(self.cargo.data.root_pos_w)
        return (loc[:, 0] > c.bay_x0 - 0.010) & (loc[:, 0] < c.bay_x1 + 0.012) \
            & (loc[:, 1].abs() < c.bay_hw + 0.010) \
            & (loc[:, 2] > 0.020) & (loc[:, 2] < 0.085)

    def aligned_now(self) -> torch.Tensor:
        """(N,) bool: bay pointing at the red garage within `align_tol_deg`, cargo
        still aboard, platter near-still (a fast flyby does not count)."""
        c = self.cfg
        err_ok = self.align_err_rad().abs() < math.radians(c.align_tol_deg)
        still = self.platter.data.root_ang_vel_w[:, 2].abs() < c.spin_still
        return err_ok & self.cargo_in_bay() & still

    def in_red_zone(self) -> torch.Tensor:
        """(N,) bool: cargo centre in the red garage interior PAST the full-inside
        plane (`inside_x_min`): rear face at/behind the mouth. Reachable only through
        the mouth (roof above, walls aside, solid plinth below)."""
        c = self.cfg
        loc = self._dock_local("red", self.cargo.data.root_pos_w)
        return (loc[:, 0] > c.inside_x_min) & (loc[:, 0] < c.dock_x_back) \
            & (loc[:, 1].abs() < c.dock_y_half - 0.005) \
            & (loc[:, 2] > c.plinth_top - 0.005) & (loc[:, 2] < c.roof_z)

    def resting_in_red(self) -> torch.Tensor:
        """(N,) bool: in the red zone AND down on the garage floor (not perched)."""
        loc = self._dock_local("red", self.cargo.data.root_pos_w)
        return self.in_red_zone() & (loc[:, 2] < self.cfg.rest_z_max)

    def settled(self) -> torch.Tensor:
        """(N,) bool: cargo slow and the platter not spinning."""
        c = self.cfg
        return (self.cargo.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.platter.data.root_ang_vel_w[:, 2].abs() < c.settle_spin)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([self.platter.data.root_pos_w, self.cargo.data.root_pos_w], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        fin = self._finite()
        self._aligned |= self.aligned_now() & fin
        self._delivered |= self.in_red_zone() & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the cargo cube settled on the red garage floor, fully inside
        (rear face past the mouth plane), everything finite. Live physical outcome."""
        self._update_latches()
        return self.resting_in_red() & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.25*aligned + 0.35*delivered (latched; ~0 for the
        null policy — the start angle is guaranteed >= `start_err_min_deg` off), capped
        at 0.60 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_aligned * self._aligned.float()
                + c.w_delivered * self._delivered.float()).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="cargo_carousel", robot="null"))

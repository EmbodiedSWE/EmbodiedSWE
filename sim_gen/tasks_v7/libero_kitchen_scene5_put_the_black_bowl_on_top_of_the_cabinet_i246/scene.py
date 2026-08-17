"""CrownSocketScene — evict the red plug from the rooftop socket, then seat the black
bowl in the vacated socket at the crown of the cabinet's shedding roof.

Derived from libero_90 kitchen_scene5 "put the black bowl on top of the cabinet" (a
Franka picks a black bowl off the table and sets it on the cabinet's FIXED FLAT top).
Here the cabinet still exists and the goal phrase still holds — the black bowl must end
up on top of the cabinet — but the top is no longer a surface that accepts placement:

  - the whole top is a slick pitched GABLE ROOF (22 deg, low-friction sheet) that
    SHEDS anything set down on it — the seed's entire plan (carry the bowl up and
    release it on the top) ends with the bowl sliding off the eave onto the floor;
  - the only sanctioned rest is a sunken square SOCKET at the roof's crown (a raised
    rim tower whose well floor sits below the rim) — and at reset the socket is
    OCCUPIED by a red cylindrical PLUG with a grasp knob;
  - the winning plan is EVICTION + SUBSTITUTION: pull the plug vertically out of the
    socket by its knob (a contact-dynamics extraction), discard it, and only then
    lower the black bowl through the rim aperture so it drops and seats on the well
    floor. The order is forced VOLUMETRICALLY — while the plug occupies the well
    there is physically no room for the bowl (it can only stack on the plug, which
    the rubric's z band rejects). No joints, no counterweights, no mechanism to
    actuate: the interlock is pure rigid-body occupancy and shedding geometry.

Assets are fully procedural. The cabinet is 8 KINEMATIC boxes (carcass, two roof
slabs, four rim-tower walls, socket floor) that are RE-POSED every reset from a
sampled cabinet frame (xy jitter + yaw), so the socket's world location changes per
episode; the plug (cylinder body + knob, custom compound spawner) and the black bowl
(a squat black cylinder) are dynamic.

Rubric (0..1, state-based — credit persists under correct behavior because the states
persist, and honestly reflects the CURRENT socket state):
  0.40 * vacated      — the plug is out of the socket volume (lifted clear or away)
  0.30 * bowl seated  — the bowl rests in the well: xy within tolerance of the socket
                        axis, root z inside the seated band (rejects hovering, rim
                        rests, and stacking on the plug), upright
  1.00 iff success()  — bowl seated AND settled. Null policy scores 0 (plug seated
                        occupies the socket; the bowl starts on the floor).

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


# ----- custom compound spawner (the plug: body cylinder + grasp knob, one rigid body) -----------
_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_plug(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the plug: DYNAMIC compound of two z-aligned cylinders — the socket-filling
    body and a slender grasp knob on top. Root origin at the BOTTOM centre (root z ==
    well-floor height when seated). Custom spawners apply no cfg schemas, so mass,
    damping, solver iterations and sleep thresholds are authored here explicitly."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    root = xform.GetPrim()

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.20)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)

    def cyl(name: str, radius: float, height: float, z: float) -> None:
        c = UsdGeom.Cylinder.Define(stage, f"{prim_path}/{name}")
        c.CreateRadiusAttr(float(radius))
        c.CreateHeightAttr(float(height))
        c.CreateAxisAttr("Z")
        c.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                            Gf.Vec3f(radius, radius, height / 2)])
        UsdGeom.Xformable(c.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(z)))
        c.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(c.GetPrim())
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    cyl("body", cfg.body_r, cfg.body_h, cfg.body_h / 2)
    cyl("knob", cfg.knob_r, cfg.knob_h, cfg.body_h + cfg.knob_h / 2)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the plug spawner configclass (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "plug" not in _SPAWNER_CACHE:

        @configclass
        class PlugSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_plug)
            body_r: float = 0.046
            body_h: float = 0.05
            knob_r: float = 0.016
            knob_h: float = 0.035
            color: tuple = (0.75, 0.10, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["plug"] = PlugSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CrownSocketSceneCfg(BaseCfg):
    """Config for `CrownSocketScene`. The interlock is volumetric: the socket is the
    only place on the cabinet that holds an object (the roof sheds everything else),
    and it starts full — nothing can be seated until the plug is extracted."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    seat_xy_tol: float = tunable(0.022)  # bowl centre within this of the socket axis, per axis
    # (honest by construction: max physical in-well offset = well_half - bowl_r = 1.8 cm)
    seat_z_lo: float = tunable(0.450)  # seated bowl-root z band (root = cylinder CENTRE;
    seat_z_hi: float = tunable(0.478)  # seated centre = floor_top + bowl_h/2 = 0.460; the hi
    # bound rejects rim rests (0.500), stacking on the plug body (0.510+) and hovers)
    up_min: float = tunable(0.85)  # min body-z . world-z for "upright"
    settle_speed: float = tunable(0.10)  # max bowl |lin vel| when judging (above the known
    # GPU phantom-velocity artifact band; the position band + persistence carry honesty)
    settle_omega: float = tunable(1.0)  # max bowl |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    cab_jitter: float = tunable(0.025)  # cabinet-frame xy jitter (+/- m) per episode
    cab_yaw_deg: float = tunable(10.0)  # cabinet-frame yaw (+/- deg) per episode
    bowl_jitter: float = tunable(0.03)  # bowl spawn xy jitter (+/- m)
    bowl_side_random: bool = tunable(True)  # Bernoulli left/right bowl spawn side

    # --- info: layout (single Franka base at the origin, facing +x) ------------------------------
    cab_pos: tuple = info((0.55, 0.0))  # nominal cabinet-frame centre (jittered per episode)
    bowl_slot: tuple = info((0.21, 0.27))  # bowl spawn (x, |y|); side sampled per episode

    # --- info: cabinet structure (all cabinet-local; 8 kinematic parts, re-posed per reset) ------
    carcass: tuple = info((0.42, 0.42, 0.34))  # solid base box; top face at 0.34
    roof_eave_z: float = info(0.345)  # roof TOP surface height at the eave (y = +/-0.21)
    ridge_z: float = info(0.430)  # roof TOP surface height at the ridge (y = 0)
    roof_t: float = info(0.012)
    roof_mu: float = info(0.05)  # slick sheet: pair-averaged with the bowl's 0.3 ->
    # effective mu ~0.175, friction angle ~10 deg << the 22 deg pitch: the roof sheds
    well_half: float = info(0.05)  # socket well half-width (square, inner 10 x 10 cm)
    wall_t: float = info(0.02)
    rim_z: float = info(0.475)  # rim-tower top
    wall_z0: float = info(0.35)  # rim-tower wall bottom (overlaps the roof: sealed)
    floor_top: float = info(0.435)  # well floor top (socket depth = rim_z - floor_top = 4 cm)
    floor_t: float = info(0.02)

    # --- info: plug / bowl ------------------------------------------------------------------------
    plug_body_r: float = info(0.046)  # 4 mm/side clearance in the 10 cm well
    plug_body_h: float = info(0.05)  # seated body top at 0.485 — 1 cm proud of the rim
    plug_knob_r: float = info(0.016)  # parallel-jaw grasp knob (3.2 cm dia)
    plug_knob_h: float = info(0.035)
    plug_mass: float = info(0.35)
    bowl_r: float = info(0.032)  # 6.4 cm dia — inside the Franka jaw; 1.8 cm/side in-well
    bowl_h: float = info(0.05)
    bowl_mass: float = info(0.25)
    bowl_mu: float = info(0.30)

    # --- info: colors -----------------------------------------------------------------------------
    carcass_color: tuple = info((0.45, 0.30, 0.15))  # brown wooden cabinet
    roof_color: tuple = info((0.35, 0.37, 0.42))  # dark slate "sheet metal" roof
    tower_color: tuple = info((0.65, 0.65, 0.68))  # light gray rim tower
    plug_color: tuple = info((0.75, 0.10, 0.10))  # RED plug
    bowl_color: tuple = info((0.04, 0.04, 0.04))  # BLACK bowl

    # --- info: rubric weights ---------------------------------------------------------------------
    w_vacated: float = info(0.40)
    w_seated: float = info(0.30)

    # --- info: occupancy band (plug "in the socket" volume, cabinet-local) ----------------------
    occ_xy: float = info(0.07)  # plug root within this of the socket axis, per axis, AND
    occ_z_lo: float = info(0.40)  # root (plug bottom) z inside this band -> socket occupied
    occ_z_hi: float = info(0.49)  # lifted clear = bottom 1.5 cm above the rim -> vacated

    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__): {name: (local_pos, local_quat, size, color, slick)}
    parts: dict = field(default=None, init=False)
    pitch: float = field(default=None, init=False)
    seat_z: float = field(default=None, init=False)  # seated bowl-root (centre) height

    def __post_init__(self) -> None:
        half_w = self.carcass[1] / 2  # 0.21
        rise = self.ridge_z - self.roof_eave_z  # 0.085
        self.pitch = math.atan2(rise, half_w)  # ~22 deg
        sp, cp = math.sin(self.pitch), math.cos(self.pitch)
        slope_len = math.hypot(half_w, rise)
        # roof slab centre = top-surface midpoint minus half thickness along the normal
        mid_y = half_w / 2
        mid_z = (self.ridge_z + self.roof_eave_z) / 2
        cy = mid_y - (self.roof_t / 2) * sp
        cz = mid_z - (self.roof_t / 2) * cp
        h2 = self.pitch / 2
        wh = self.well_half
        wt = self.wall_t
        wall_h = self.rim_z - self.wall_z0
        wall_zc = (self.rim_z + self.wall_z0) / 2
        self.seat_z = self.floor_top + self.bowl_h / 2
        cx, cyl_ = self.carcass[0], self.carcass[1]
        self.parts = {
            "carcass": ((0.0, 0.0, self.carcass[2] / 2), (1.0, 0.0, 0.0, 0.0),
                        (cx, cyl_, self.carcass[2]), self.carcass_color, False),
            # +y half descends toward +y: rotate about x by -pitch; -y half mirrors
            "roof_p": ((0.0, +cy, cz), (math.cos(h2), -math.sin(h2), 0.0, 0.0),
                       (cx, slope_len, self.roof_t), self.roof_color, True),
            "roof_m": ((0.0, -cy, cz), (math.cos(h2), +math.sin(h2), 0.0, 0.0),
                       (cx, slope_len, self.roof_t), self.roof_color, True),
            "wall_xp": ((+(wh + wt / 2), 0.0, wall_zc), (1.0, 0.0, 0.0, 0.0),
                        (wt, 2 * (wh + wt), wall_h), self.tower_color, False),
            "wall_xm": ((-(wh + wt / 2), 0.0, wall_zc), (1.0, 0.0, 0.0, 0.0),
                        (wt, 2 * (wh + wt), wall_h), self.tower_color, False),
            "wall_yp": ((0.0, +(wh + wt / 2), wall_zc), (1.0, 0.0, 0.0, 0.0),
                        (2 * wh, wt, wall_h), self.tower_color, False),
            "wall_ym": ((0.0, -(wh + wt / 2), wall_zc), (1.0, 0.0, 0.0, 0.0),
                        (2 * wh, wt, wall_h), self.tower_color, False),
            "sockfloor": ((0.0, 0.0, self.floor_top - self.floor_t / 2),
                          (1.0, 0.0, 0.0, 0.0),
                          (2 * (wh + wt), 2 * (wh + wt), self.floor_t),
                          self.tower_color, False),
        }


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product, (N,4) x (N,4) -> (N,4), wxyz."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("crown_socket")
class CrownSocketScene(BaseScene):
    cfg: CrownSocketSceneCfg

    def __init__(self, cfg: CrownSocketSceneCfg | None = None) -> None:
        super().__init__(cfg or CrownSocketSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cx, cy = c.cab_pos
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
        # --- the 8 kinematic cabinet parts (re-posed per reset from the sampled frame) ---
        for name, (lp, lq, size, color, slick) in c.parts.items():
            mat = sim_utils.RigidBodyMaterialCfg(
                static_friction=c.roof_mu if slick else 0.5,
                dynamic_friction=c.roof_mu if slick else 0.5,
                restitution=0.0,
            )
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cab_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(cx + lp[0], cy + lp[1], lp[2]), rot=lq),
            )
        # --- plug (custom compound spawner authors its own physics armor) ---
        plug_spawn = _spawner_classes()["plug"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.plug_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            body_r=c.plug_body_r, body_h=c.plug_body_h,
            knob_r=c.plug_knob_r, knob_h=c.plug_knob_h,
            color=c.plug_color, contact_offset=c.contact_offset,
        )
        out["plug"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plug",
            spawn=plug_spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(cx, cy, c.floor_top + 0.0005)),
        )
        # --- black bowl (squat cylinder) ---
        out["bowl"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bowl",
            spawn=sim_utils.CylinderCfg(
                radius=c.bowl_r, height=c.bowl_h, axis="Z",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4,
                    max_depenetration_velocity=0.5,
                    linear_damping=0.05,
                    angular_damping=0.20,
                    sleep_threshold=0.0,
                    stabilization_threshold=0.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.bowl_mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=c.bowl_mu, dynamic_friction=c.bowl_mu,
                    restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.bowl_color),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.bowl_slot[0], c.bowl_slot[1], c.bowl_h / 2 + 0.002)),
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
        self.parts: dict[str, RigidObject] = {n: env.iscene[n] for n in c.parts}
        self.plug: RigidObject = env.iscene["plug"]
        self.bowl: RigidObject = env.iscene["bowl"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self._cab_xy = torch.tensor(c.cab_pos, device=dev).expand(n, 2).clone()
        self._cab_yaw = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the cabinet frame (xy jitter + yaw) and re-pose all 8
        kinematic parts in it; seat the plug in the socket; spawn the bowl on a random
        floor side with jitter (with a deterministic outward clamp so it can never
        spawn intersecting the carcass — flush spawns get depenetration-shoved)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        dxy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.cab_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.cab_yaw_deg)
        cab = torch.tensor(c.cab_pos, device=dev) + dxy
        self._cab_xy[env_ids] = cab
        self._cab_yaw[env_ids] = yaw
        cosy, siny = torch.cos(yaw), torch.sin(yaw)
        qyaw = torch.stack([torch.cos(yaw / 2), torch.zeros(m, device=dev),
                            torch.zeros(m, device=dev), torch.sin(yaw / 2)], dim=-1)

        # --- the 8 kinematic cabinet parts ---
        for name, (lp, lq, _size, _color, _slick) in c.parts.items():
            wx = cab[:, 0] + cosy * lp[0] - siny * lp[1]
            wy = cab[:, 1] + siny * lp[0] + cosy * lp[1]
            lqt = torch.tensor(lq, device=dev).expand(m, 4)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0], st[:, 1], st[:, 2] = wx, wy, lp[2]
            st[:, 3:7] = _quat_mul(qyaw, lqt)
            st[:, 0:3] += origin
            self.parts[name].write_root_state_to_sim(st, env_ids)

        # --- plug: seated in the socket (root = bottom centre on the well floor) ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = cab[:, 0], cab[:, 1]
        st[:, 2] = c.floor_top + 0.0005
        st[:, 3:7] = qyaw
        st[:, 0:3] += origin
        self.plug.write_root_state_to_sim(st, env_ids)

        # --- bowl: random side + jitter on the floor, clamped outside the carcass ---
        if c.bowl_side_random:
            side = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        else:
            side = torch.ones(m, device=dev)
        bx = c.bowl_slot[0] + (torch.rand(m, device=dev) * 2 - 1) * c.bowl_jitter
        by = side * c.bowl_slot[1] + (torch.rand(m, device=dev) * 2 - 1) * c.bowl_jitter
        # deterministic outward clamp (cabinet-local Chebyshev >= carcass half + bowl r + 8mm)
        lx = cosy * (bx - cab[:, 0]) + siny * (by - cab[:, 1])
        ly = -siny * (bx - cab[:, 0]) + cosy * (by - cab[:, 1])
        ches = torch.maximum(lx.abs(), ly.abs()).clamp(min=1e-6)
        need = c.carcass[0] / 2 + c.bowl_r + 0.008
        scale = torch.where(ches < need, (need + 0.02) / ches, torch.ones_like(ches))
        lx, ly = lx * scale, ly * scale
        bx = cab[:, 0] + cosy * lx - siny * ly
        by = cab[:, 1] + siny * lx + cosy * ly
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = bx, by
        st[:, 2] = c.bowl_h / 2 + 0.002
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.bowl.write_root_state_to_sim(st, env_ids)

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "parts": {n: b.data.root_state_w[env_ids].clone() for n, b in self.parts.items()},
            "plug": self.plug.data.root_state_w[env_ids].clone(),
            "bowl": self.bowl.data.root_state_w[env_ids].clone(),
            "cab_xy": self._cab_xy[env_ids].clone(),
            "cab_yaw": self._cab_yaw[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for n, b in self.parts.items():
            b.write_root_state_to_sim(state["parts"][n], env_ids)
        self.plug.write_root_state_to_sim(state["plug"], env_ids)
        self.bowl.write_root_state_to_sim(state["bowl"], env_ids)
        self._cab_xy[env_ids] = state["cab_xy"]
        self._cab_yaw[env_ids] = state["cab_yaw"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A BROWN wooden cabinet ({c.carcass[0] * 100:.0f} x {c.carcass[1] * 100:.0f} cm "
            f"footprint, body {c.carcass[2] * 100:.0f} cm tall) stands on the floor, its "
            f"position and heading slightly different each episode. Its whole top is a "
            f"DARK-SLATE pitched gable roof (two slick sheet-metal slopes at "
            f"{math.degrees(self.cfg.pitch):.0f} deg, ridge at {c.ridge_z * 100:.0f} cm): "
            f"anything set down on the roof slides off the eave onto the floor — the roof "
            f"holds nothing. At the crown of the roof, breaking through the ridge, sits a "
            f"LIGHT-GRAY square rim tower (outer {2 * (c.well_half + c.wall_t) * 100:.0f} cm, "
            f"rim top at {c.rim_z * 100:.0f} cm) enclosing a sunken SOCKET well "
            f"({2 * c.well_half * 100:.0f} x {2 * c.well_half * 100:.0f} cm, "
            f"{(c.rim_z - c.floor_top) * 100:.0f} cm deep) — the only spot on the cabinet "
            f"that holds an object. At the start the socket is OCCUPIED by a RED "
            f"cylindrical plug ({2 * c.plug_body_r * 100:.1f} cm dia body filling the well, "
            f"topped by a {2 * c.plug_knob_r * 100:.1f} cm dia grasp knob reaching "
            f"{(c.floor_top + c.plug_body_h + c.plug_knob_h) * 100:.0f} cm). A BLACK squat "
            f"cylinder — the bowl ({2 * c.bowl_r * 100:.1f} cm dia, {c.bowl_h * 100:.0f} cm "
            f"tall) — stands on the floor beside the cabinet (left or right varies per "
            f"episode).\n"
            f"Goal: put the black bowl on top of the cabinet — which here means SEATED in "
            f"the rooftop socket: first pull the red plug straight up and out of the socket "
            f"by its knob and set it aside anywhere on the floor away from the cabinet, "
            f"then lower the black bowl centered over the vacated socket and let it drop "
            f"in, so it rests upright on the well floor (centre within "
            f"{c.seat_xy_tol * 100:.1f} cm of the socket axis) with everything at rest. "
            f"The order is forced by physics: while the plug fills the well there is no "
            f"room for the bowl (it can only stack on the plug, which does not count). A "
            f"bowl left anywhere on the roof slides off; a bowl on the floor, on the rim, "
            f"or resting on the plug counts for nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pull the red plug up out of the socket at the crown of the cabinet's roof by "
            "its knob and set it aside on the floor, then seat the black cylindrical bowl "
            "in the vacated socket so it rests upright inside the well. The slick pitched "
            "roof sheds anything placed elsewhere on top; a bowl not seated in the socket "
            "does not count."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def _local(self, body: RigidObject) -> torch.Tensor:
        """(N,3) body root position in the CABINET frame (xy rotated by -yaw, z as-is)."""
        p = body.data.root_pos_w - self.env_origins
        d = p[:, :2] - self._cab_xy
        cosy, siny = torch.cos(self._cab_yaw), torch.sin(self._cab_yaw)
        lx = cosy * d[:, 0] + siny * d[:, 1]
        ly = -siny * d[:, 0] + cosy * d[:, 1]
        return torch.stack([lx, ly, p[:, 2]], dim=-1)

    def plug_occupies(self) -> torch.Tensor:
        """(N,) bool: the plug is inside the socket volume (root/bottom within the
        occupancy band). Seated -> True; lifted clear of the rim or moved away -> False."""
        c = self.cfg
        l = self._local(self.plug)
        return ((l[:, 0].abs() <= c.occ_xy) & (l[:, 1].abs() <= c.occ_xy)
                & (l[:, 2] >= c.occ_z_lo) & (l[:, 2] <= c.occ_z_hi))

    def vacated(self) -> torch.Tensor:
        """(N,) bool: the socket is free (state-based — honestly reflects the CURRENT
        socket state; re-occupying the socket revokes the credit)."""
        return ~self.plug_occupies()

    def _upright(self, body: RigidObject) -> torch.Tensor:
        q = body.data.root_quat_w
        r33 = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        return r33 >= self.cfg.up_min

    def bowl_seated(self) -> torch.Tensor:
        """(N,) bool, geometric: bowl root (cylinder centre) on the socket axis within
        `seat_xy_tol` per axis, z inside the seated band, upright. The z band rejects
        hovering, rim rests, and stacking on the (still-seated) plug; being physically
        inside the well implies the plug is out (there is no room for both)."""
        c = self.cfg
        l = self._local(self.bowl)
        return ((l[:, 0].abs() <= c.seat_xy_tol) & (l[:, 1].abs() <= c.seat_xy_tol)
                & (l[:, 2] >= c.seat_z_lo) & (l[:, 2] <= c.seat_z_hi)
                & self._upright(self.bowl))

    def bowl_still(self) -> torch.Tensor:
        return ((self.bowl.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed)
                & (self.bowl.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_omega))

    def success(self) -> torch.Tensor:
        """(N,) bool: the black bowl SEATED in the rooftop socket and at rest — a
        physical outcome (a settled pose inside a well the plug had to vacate)."""
        return self.bowl_seated() & self.bowl_still()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.40 * socket-vacated + 0.30 * bowl-seated (state-based:
        the states persist under correct behavior, so credit does not evaporate along
        the solution trajectory), exactly 1.0 iff success(). Null policy: the plug
        occupies the socket and the bowl stands on the floor -> 0."""
        c = self.cfg
        base = c.w_vacated * self.vacated().float() + c.w_seated * self.bowl_seated().float()
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="crown_socket", robot="null"))

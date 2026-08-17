"""RelayTunnelScene — seat the RED cube in a sunken well at the far end of a low,
roofed, single-lane tunnel, by driving it with the BLUE cube as a relay pusher.

Derived from mujoco_playground/pick ("bring the box to the target": one grasp of a
4 cm cube and one guided free-space carry to a floating, always-reachable target
pose). Here nothing is carried to the goal: the goal region is the interior of a
SUNKEN WELL cut into the floor at the far end of a low roofed tunnel (a "garage").
The roof forbids any top-down placement over the well (the smoke battery presses a
cube onto the roof to prove it), and the well lies ~90 mm past the mouth — beyond
what a parallel-jaw hand can reach into the 66 mm-tall opening (~60 mm of closed
finger prong; the embodiment argument in TASK.md). The plan the seed never needs:
  (1) stage the RED cube on the open apron at the tunnel mouth and finger-push it
      inside as far as the hand can follow (~45 mm);
  (2) stage the BLUE cube behind it and push BLUE — the blue cube becomes the
      TOOL, transmitting the push through block-on-block contact and driving RED
      the final unreachable distance until it drops 16 mm into the well and seats.
The channel is single-lane (one cube wide): blocks cannot pass each other inside,
so the RED cube MUST enter first — a blue-first end state (blue in the well, red
stuck behind) is judged a failure. The 16 mm step captures the seated cube against
horizontal pushes; only the settled final state is judged.

Assets are fully procedural: one KINEMATIC compound (the garage) + two dynamic
cubes (identical physics, different colors).
  garage (kinematic, garage frame: mouth plane x=0, channel +x, ground z=0):
    slab   : raised floor, x in [-apron_len, well_x0], thickness `step` (16 mm) —
             its front part is the open APRON where cubes are staged;
    walls  : +/-y, x in [0, back_x1], full height — the single-lane channel;
    well   : x in [well_x0, well_x1] has NO slab: its floor is the ground,
             `step` lower — the sunken target region;
    backwl : x in [well_x1, back_x1], full height — the channel dead-ends;
    roof   : x in [0, back_x1] — the whole channel (and the entire well) is
             covered; interior height above the slab is 66 mm (cube 50 mm).
  red / blue cubes: 50 mm, 0.08 kg, moderate friction (walls are slick so pushed
  cubes square up against them instead of wedging).

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  0.25 * entered — the RED cube EVER inside the roofed channel            (latched)
  0.25 * deep    — the RED cube EVER past half-depth (x > deep_x — beyond a
                   direct fingertip push; the relay zone)                 (latched)
  0.25 * seated  — the RED cube EVER seated in the sunken well            (latched)
  1.0 iff success() — the RED cube seated in the well, settled, finite.
  Non-success capped at 0.75. Null policy latches nothing (score ~0).

Honesty geometry (asserted in `__post_init__`):
  - the roof covers the entire well and channel: no top-down entry anywhere past
    the mouth; the interior height passes ONE cube with clearance but forbids
    stacking (a cube cannot ride over another inside);
  - the well admits exactly ONE cube (well_len < 2 cubes): the wrong cube seated
    is an exclusive, unrecoverable wrong outcome;
  - the seated z-window separates cleanly from both a cube resting ON the slab
    and a cube straddling the lip (the step is what makes z discriminative);
  - seating requires the red cube's trailing face to pass ~90 mm — far past the
    documented fingertip reach, so the relay (or a calibrated flick) is required;
  - spawn slots clear the garage walls by a real margin (no depenetration nudges).

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
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


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
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


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, *, center, size, color, contact_offset: float):
    """One collidable box child: translate + scale, displayColor."""
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


def _rigid_kinematic(root) -> None:
    """Kinematic rigid-body armor on the compound root (teleportable at reset)."""
    from pxr import UsdPhysics

    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(60.0)


def _spawn_garage(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC garage at `prim_path`. Root origin = mouth plane centre
    at ground level; channel extends +x. Children: slab (apron + in-channel floor),
    two side walls, back wall, roof. Floor and walls get separate frictions (slick
    walls so pushed cubes square up instead of wedging)."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_kinematic(root)
    floor_mat = _phys_material(stage, f"{prim_path}/floormat",
                               cfg.mu_floor, max(cfg.mu_floor - 0.05, 0.02))
    wall_mat = _phys_material(stage, f"{prim_path}/wallmat",
                              cfg.mu_wall, max(cfg.mu_wall - 0.02, 0.02))
    c = cfg
    co = c.contact_offset
    hw = c.chan_w / 2                           # channel interior half-width
    wall_h = c.roof_z                            # walls reach the roof underside
    slab_x0, slab_x1 = -c.apron_len, c.well_x0
    floors, walls = [], []

    def box(kids, name, center, size, color):
        kids.append(_box(stage, f"{prim_path}/{name}", center=center, size=size,
                         color=color, contact_offset=co))

    # slab: raised floor from the open apron up to the well lip
    box(floors, "slab", ((slab_x0 + slab_x1) / 2, 0.0, c.step / 2),
        (slab_x1 - slab_x0, c.chan_w, c.step), c.slab_color)
    # side walls: the single-lane channel
    for s, tag in ((1.0, "yp"), (-1.0, "yn")):
        box(walls, f"wall_{tag}", (c.back_x1 / 2, s * (hw + c.wall_t / 2), wall_h / 2),
            (c.back_x1, c.wall_t, wall_h), c.wall_color)
    # back wall: the channel dead-ends just past the well
    box(walls, "backwall", ((c.well_x1 + c.back_x1) / 2, 0.0, wall_h / 2),
        (c.back_x1 - c.well_x1, c.chan_w, wall_h), c.wall_color)
    # roof: covers the whole channel including the entire well
    box(walls, "roof", (c.back_x1 / 2, 0.0, c.roof_z + c.roof_t / 2),
        (c.back_x1, c.chan_w + 2 * c.wall_t, c.roof_t), c.roof_color)
    for k in floors:
        _bind_material(k, floor_mat)
    for k in walls:
        _bind_material(k, wall_mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclass (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "garage" not in _SPAWNER_CACHE:

        @configclass
        class GarageSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_garage)
            apron_len: float = 0.12
            chan_w: float = 0.068
            wall_t: float = 0.02
            step: float = 0.016
            well_x0: float = 0.090
            well_x1: float = 0.175
            back_x1: float = 0.195
            roof_z: float = 0.082
            roof_t: float = 0.014
            mu_floor: float = 0.30
            mu_wall: float = 0.06
            slab_color: tuple = (0.72, 0.70, 0.62)
            wall_color: tuple = (0.42, 0.46, 0.55)
            roof_color: tuple = (0.16, 0.42, 0.24)
            contact_offset: float = 0.003

        _SPAWNER_CACHE["garage"] = GarageSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class RelayTunnelSceneCfg(BaseCfg):
    """Config for `RelayTunnelScene`. The single-lane relay and the capture are
    enforced by the garage's own geometry; the rubric only reads out states the
    geometry makes meaningful (see the honesty asserts)."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    deep_x: float = tunable(0.080)          # `deep` latch: red centre past this (m)
    seat_x_lo: float = tunable(0.105)       # seated window, garage-frame x (m)
    seat_x_hi: float = tunable(0.162)
    seat_z_hi: float = tunable(0.0285)      # seated cube centre BELOW this (on-slab is 0.041)
    settle_lin: float = tunable(0.05)       # max red |lin vel| when judging success (m/s)
    settle_ang: float = tunable(1.0)        # max red |ang vel| when judging success (rad/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    garage_jitter: float = tunable(0.03)    # garage xy jitter (+/- m)
    garage_yaw_deg: float = tunable(20.0)   # garage yaw jitter (+/- deg)
    cube_jitter: float = tunable(0.03)      # cube spawn xy jitter (+/- m)
    cube_yaw_deg: float = tunable(180.0)    # cube spawn yaw (+/- deg)

    # --- info: garage structure (garage frame: mouth plane x=0, channel +x, ground z=0) ----------
    apron_len: float = info(0.12)           # open staging slab in front of the mouth
    chan_w: float = info(0.068)             # channel interior width
    wall_t: float = info(0.02)
    step: float = info(0.016)               # slab thickness = well drop (the capture)
    well_x0: float = info(0.090)            # well near lip (slab ends here)
    well_x1: float = info(0.175)            # well far edge (back wall starts)
    back_x1: float = info(0.195)            # garage far extent
    roof_z: float = info(0.082)             # roof underside height
    roof_t: float = info(0.014)
    garage_pos: tuple = info((0.42, 0.0))   # world xy of the garage frame origin
    mu_floor: float = info(0.30)
    mu_wall: float = info(0.06)
    # --- info: cubes -----------------------------------------------------------------------------
    cube_s: float = info(0.05)
    cube_mass: float = info(0.08)
    cube_mu: float = info(0.30)
    red_color: tuple = info((0.85, 0.10, 0.10))
    blue_color: tuple = info((0.10, 0.20, 0.85))
    spawn_local_x: float = info(-0.20)      # cube spawn slots, garage frame
    spawn_local_y: float = info(0.13)
    ground_mu: float = info(0.35)
    contact_offset: float = info(0.003)
    finger_reach: float = info(0.060)       # documented closed-jaw prong depth (embodiment)
    # rubric weights (0.25 * 3 = 0.75 <= the non-success cap 0.75)
    w_entered: float = info(0.25)
    w_deep: float = info(0.25)
    w_seated: float = info(0.25)

    # Derived (filled in __post_init__).
    z_on_slab: float = field(default=0.0, init=False)   # cube centre resting on the slab
    z_seated: float = field(default=0.0, init=False)    # cube centre seated in the well
    z_roof_top: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        s = self.cube_s
        self.z_on_slab = self.step + s / 2
        self.z_seated = s / 2
        self.z_roof_top = self.roof_z + self.roof_t
        well_len = self.well_x1 - self.well_x0
        # the well admits exactly ONE cube (exclusive occupancy):
        assert s + 0.006 < well_len < 2 * s, "well must admit exactly one cube"
        # the roof covers the whole channel and the whole well (no top-down entry):
        assert self.back_x1 >= self.well_x1, "roof/garage must cover the well fully"
        # one cube passes with clearance; two never stack inside:
        assert self.roof_z - self.step > s + 0.010, "channel must pass one cube"
        assert self.roof_z < 2 * s + self.step, "channel must forbid stacking inside"
        assert self.chan_w > s + 0.012, "channel must pass a (slightly yawed) cube"
        assert self.chan_w < 2 * s, "channel must be single-lane (no passing)"
        # the seated z-window discriminates: seated < window < lip-straddle < on-slab
        lean = math.asin(min(1.0, self.step / s))       # lip-straddle tilt
        z_lean = self.step / 2 + (s / 2) * math.cos(lean)
        assert self.z_seated + 0.002 < self.seat_z_hi < z_lean - 0.002 < self.z_on_slab, \
            "seated z-window must separate seated / lip-straddle / on-slab"
        # seating is beyond a direct fingertip push (the relay is load-bearing): even
        # the shallowest seat puts the cube's trailing face at the well lip.
        assert self.well_x0 > self.finger_reach + 0.02, \
            "well must lie beyond fingertip reach (trailing face at the lip)"
        # during the lip tumble the cube's bottom front corner must clear the back
        # wall, INCLUDING when the relay over-pushes the drop ~12 mm past the lip
        # (a moving cube leaves the slab before it starts tipping):
        reach = self.well_x0 + 0.012 + s * (math.cos(lean) + math.sin(lean))
        assert reach < self.well_x1 - 0.004, \
            f"tumbling cube corner must clear the back wall (reach {reach})"
        # spawn slots clear the garage walls by a real margin:
        assert (self.spawn_local_y - self.cube_jitter - s / 2
                - (self.chan_w / 2 + self.wall_t)) > 0.005, "spawn slots must clear the walls"
        assert self.spawn_local_x + self.cube_jitter + s / 2 < -self.apron_len - 0.01, \
            "spawn slots must sit on the ground, clear of the apron"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("relay_tunnel")
class RelayTunnelScene(BaseScene):
    cfg: RelayTunnelSceneCfg

    def __init__(self, cfg: RelayTunnelSceneCfg | None = None) -> None:
        super().__init__(cfg or RelayTunnelSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        garage_spawn = cls["garage"](
            apron_len=c.apron_len, chan_w=c.chan_w, wall_t=c.wall_t, step=c.step,
            well_x0=c.well_x0, well_x1=c.well_x1, back_x1=c.back_x1,
            roof_z=c.roof_z, roof_t=c.roof_t, mu_floor=c.mu_floor, mu_wall=c.mu_wall,
            contact_offset=c.contact_offset)

        def cube(color, x0, y0) -> RigidObjectCfg:
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/CubeTmp",
                spawn=sim_utils.CuboidCfg(
                    size=(c.cube_s, c.cube_s, c.cube_s),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05,
                        angular_damping=0.20,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.cube_mu, dynamic_friction=c.cube_mu - 0.05,
                        restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(x0, y0, c.cube_s / 2 + 0.002)),
            )

        red = cube(c.red_color, c.garage_pos[0] + c.spawn_local_x,
                   c.garage_pos[1] - c.spawn_local_y)
        red.prim_path = "{ENV_REGEX_NS}/RedCube"
        blue = cube(c.blue_color, c.garage_pos[0] + c.spawn_local_x,
                    c.garage_pos[1] + c.spawn_local_y)
        blue.prim_path = "{ENV_REGEX_NS}/BlueCube"

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.ground_mu, dynamic_friction=c.ground_mu - 0.05,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "garage": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Garage",
                spawn=garage_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.garage_pos[0], c.garage_pos[1], 0.0)),
            ),
            "red": red,
            "blue": blue,
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,  # quasi-static pushing; the 16 mm well drop is slow
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
        self.garage: RigidObject = env.iscene["garage"]
        self.red: RigidObject = env.iscene["red"]
        self.blue: RigidObject = env.iscene["blue"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self.swap = torch.ones(n, device=dev)  # +1: red spawns on the -y slot
        # latches (partial credit survives transients; success is judged live)
        self._l_entered = torch.zeros(n, dtype=torch.bool, device=dev)
        self._l_deep = torch.zeros(n, dtype=torch.bool, device=dev)
        self._l_seated = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: garage pose (xy jitter + yaw), red/blue cubes on jittered,
        randomly swapped floor slots with free yaw, latches cleared. All draws via
        torch.rand (uniform, readback-verified by the smoke battery)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def rnd(k: float) -> torch.Tensor:
            return (torch.rand(m, device=dev) * 2 - 1) * k

        def write(body, local_xy, z, q) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = local_xy
            st[:, 2] = z
            st[:, 3:7] = q
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        gx = c.garage_pos[0] + rnd(c.garage_jitter)
        gy = c.garage_pos[1] + rnd(c.garage_jitter)
        gyaw = rnd(math.radians(c.garage_yaw_deg))
        gq = _qz(gyaw)
        gxy = torch.stack([gx, gy], dim=-1)
        write(self.garage, gxy, torch.zeros(m, device=dev), gq)

        def to_world(local_xy: torch.Tensor) -> torch.Tensor:
            l3 = torch.cat([local_xy, torch.zeros(m, 1, device=dev)], dim=-1)
            return gxy + _qapply(gq, l3)[:, :2]

        swap = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
        self.swap[env_ids] = swap
        for body, s in ((self.red, swap), (self.blue, -swap)):
            bl = torch.stack([
                c.spawn_local_x + rnd(c.cube_jitter),
                -s * c.spawn_local_y + rnd(c.cube_jitter)], dim=-1)
            yaw = gyaw + rnd(math.radians(c.cube_yaw_deg))
            write(body, to_world(bl),
                  torch.full((m,), c.cube_s / 2 + 0.002, device=dev), _qz(yaw))

        self._l_entered[env_ids] = False
        self._l_deep[env_ids] = False
        self._l_seated[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "garage": self.garage.data.root_state_w[env_ids].clone(),
            "red": self.red.data.root_state_w[env_ids].clone(),
            "blue": self.blue.data.root_state_w[env_ids].clone(),
            "swap": self.swap[env_ids].clone(),
            "l_entered": self._l_entered[env_ids].clone(),
            "l_deep": self._l_deep[env_ids].clone(),
            "l_seated": self._l_seated[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.garage.write_root_state_to_sim(state["garage"], env_ids)
        self.red.write_root_state_to_sim(state["red"], env_ids)
        self.blue.write_root_state_to_sim(state["blue"], env_ids)
        self.swap[env_ids] = state["swap"]
        self._l_entered[env_ids] = state["l_entered"]
        self._l_deep[env_ids] = state["l_deep"]
        self._l_seated[env_ids] = state["l_seated"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        mm = 1000
        return (
            f"A low GARAGE stands on the floor: a dead-end tunnel one cube wide "
            f"({c.chan_w * mm:.0f} mm between blue-grey walls), {c.back_x1 * mm:.0f} mm "
            f"deep, fully covered by a GREEN ROOF whose underside is only "
            f"{c.roof_z * mm:.0f} mm up — nothing can be lowered in from above; the open "
            f"MOUTH is the only way in. A sand-colored floor slab {c.step * mm:.0f} mm "
            f"thick runs from an open apron in front of the mouth to {c.well_x0 * mm:.0f} mm "
            f"inside, where it ends: the last {(c.well_x1 - c.well_x0) * mm:.0f} mm before "
            f"the back wall is a SUNKEN WELL, {c.step * mm:.0f} mm lower, sized for exactly "
            f"one cube. On the floor in front lie two {c.cube_s * mm:.0f} mm cubes, one RED "
            f"and one BLUE. The garage's position and heading and both cube poses vary per "
            f"episode.\n"
            f"Goal: the RED cube must end up resting flat on the well floor, fully inside "
            f"the sunken well, settled. The well lies deeper inside than a hand can reach "
            f"through the low mouth, so slide the red cube in along the slab and use the "
            f"BLUE cube as a pusher behind it: pushing the blue cube into the mouth drives "
            f"the red cube ahead of it until the red cube drops off the slab edge into the "
            f"well. ORDER IS FORCED: the red cube must enter FIRST — the channel is a "
            f"single lane, cubes cannot pass each other inside, and the well takes only "
            f"one cube, so if the blue cube goes in first and seats in the well the "
            f"episode is failed. The blue cube may end up anywhere; only the red cube "
            f"seated in the well is judged, on its settled final state."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the red cube through the garage mouth and seat it in the sunken well "
            "at the far end, pushing the blue cube in behind it to drive it the final "
            "distance. The red cube must go in first: if the blue cube ends up in the "
            "well instead, the task is failed."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _garage_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points (N,3) -> the garage frame."""
        return _qapply(_qinv(self.garage.data.root_quat_w),
                       pos_w - self.garage.data.root_pos_w)

    def entered(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool, LIVE: body centre inside the roofed channel (past the mouth,
        between the walls, below the roof — a cube ON the roof does not count)."""
        c = self.cfg
        p = self._garage_local(body.data.root_pos_w)
        return (p[:, 0] > 0.015) & (p[:, 0] < c.back_x1) \
            & (p[:, 1].abs() < c.chan_w / 2) & (p[:, 2] > 0.0) & (p[:, 2] < 0.075)

    def deep(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool, LIVE: entered AND past `deep_x` (beyond a direct fingertip push)."""
        p = self._garage_local(body.data.root_pos_w)
        return self.entered(body) & (p[:, 0] > self.cfg.deep_x)

    def in_well(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool, LIVE: body centre inside the seated window — x within the well
        span, on the channel axis, and LOW (the 16 mm drop is what z discriminates:
        on-slab rest and lip-straddle rest both sit above `seat_z_hi`)."""
        c = self.cfg
        p = self._garage_local(body.data.root_pos_w)
        return (p[:, 0] > c.seat_x_lo) & (p[:, 0] < c.seat_x_hi) \
            & (p[:, 1].abs() < c.chan_w / 2) \
            & (p[:, 2] > 0.016) & (p[:, 2] < c.seat_z_hi)

    def settled_red(self) -> torch.Tensor:
        """(N,) bool: red cube linear AND angular velocity below the settle gates."""
        return (self.red.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin) \
            & (self.red.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([self.red.data.root_pos_w, self.blue.data.root_pos_w,
                         self.garage.data.root_pos_w], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        fin = self._finite()
        self._l_seated |= fin & self.in_well(self.red)
        self._l_deep |= fin & (self.deep(self.red) | self._l_seated)
        self._l_entered |= fin & (self.entered(self.red) | self._l_deep)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the RED cube seated in the sunken well, settled, finite. A live
        physical outcome — a cube on the slab short of the lip, straddling the lip,
        on the roof, or the BLUE cube in the well all judge False by geometry."""
        self._update_latches()
        return self.in_well(self.red) & self.settled_red() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.25*entered + 0.25*deep + 0.25*seated (all latched
        on the RED cube; ~0 for doing nothing), capped at 0.75 — and exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_entered * self._l_entered.float() + c.w_deep * self._l_deep.float()
                + c.w_seated * self._l_seated.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="relay_tunnel", robot="null"))

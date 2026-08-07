"""StopperVaultScene — push the green cube into the roofed vault through its open
doorway, then CLOSE the vault by seating the wide BLUE stopper into the doorway pocket.

Derived from rlbench/put_item_in_drawer, but the receptacle's whole access model is
inverted. The seed's drawer cabinet is CLOSED until acted on: the plan is grasp the
drawer handle, pull the prismatic joint open, then lower the item in from above.
Here the vault is jointless and its side doorway is ALWAYS open — there is no handle,
no articulation, and no opening move at all. The interior is fully roofed (no top
access), so the deposit is a horizontal ground-level PUSH-THROUGH of the item along
the doorway tunnel. What replaces the seed's (optional) closing motion is a second,
harder insertion: the doorway must be sealed by fetching a LOOSE STOPPER and fitting
it into the doorway pocket, where inner stop ribs arrest it flush. Two stoppers stand
outside: the wide BLUE one fits the pocket and closes the aperture; the narrow RED
one is a decoy that could never cover the opening and must be left out. Execution
order is physically forced: a seated stopper blocks the only entry, so the cube MUST
go in first (roof gap, side gaps and header gap around an accepted stopper pose are
all far smaller than the cube).

Assets are fully procedural, authored by custom compound spawners (the pen_holder /
chill_rack pattern — child colliders of one body never self-collide):
  - vault: KINEMATIC compound — back/side walls, roof, and a THICK front wall pierced
    by a doorway tunnel (104 mm wide x 70 mm tall x 30 mm deep) with two dark stop
    ribs at the tunnel's inner end (they narrow the inner exit to 80 mm: the 50 mm
    cube passes; the 84 mm blue stopper is arrested → the pocket). Local frame:
    origin at the interior floor centre on the GROUND, +x = OUT through the doorway.
  - cube (target): green 50 mm cube, dynamic.
  - blue stopper: dynamic compound — slab 84 x 40 x 56 mm standing upright + a grip
    boss on its outer face; seated, the slab's outer 16 mm and the whole boss stand
    proud of the mouth (boss face ~48 mm outside — the stopper can always be pulled
    back out; it is never captive).
  - red stopper (decoy): same shape family but slab only 40 mm wide — visibly too
    narrow to close the 104 mm doorway.

Per-episode randomization (readback-verifiable): vault yaw + xy jitter, cube ground
slot with free yaw, the cube band and the stopper band SWAP sides 50/50, and the
blue/red slot order within the stopper band swaps 50/50; all bodies get xy jitter.

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.15 * approach — latched max of the cube's progress toward the doorway mouth,
                    normalized by its own spawn distance (exactly 0 for null policy)
  0.30 * deposited — cube ever fully inside the vault interior (latched)
  0.30 * seal     — latched max of the BLUE stopper's progress toward its seat,
                    normalized by its own spawn distance, counted ONLY while the
                    cube is currently fully inside (sealing an empty vault, or
                    seating the stopper before the deposit, earns nothing)
  1.0 iff success() — cube settled fully inside + BLUE stopper settled seated in the
                    pocket (position + upright + axis-aligned) + RED decoy clear of
                    the doorway and interior. Non-success is capped at 0.85.

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


# ----- custom compound spawners ---------------------------------------------------------------
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


def _make_collide(cfg: Any) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _add_box(stage, path: str, *, center, size, color, collide: Callable) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _spawn_vault(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the vault at `prim_path`: KINEMATIC rigid body (repositionable at reset
    via write_root_state, immovable to contacts). Local frame: origin at the interior
    floor centre on the GROUND; +x = OUT through the doorway. The ground plane IS the
    interior floor and the doorway sill (no step to catch a sliding cube).

    Children: back wall, 2 side walls, roof, and the thick front wall in three pieces
    (2 side segments + header) leaving the doorway tunnel, plus 2 stop ribs at the
    tunnel's inner end that arrest the blue stopper (the pocket)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg

    def box(name, center, size, color=None):
        _add_box(stage, f"{prim_path}/{name}", center=center, size=size,
                 color=color or c.color, collide=collide)

    ix, iy, wt = c.in_hx, c.in_hy, c.wall_t
    fx0, fx1 = ix, ix + c.front_t          # front wall inner / outer face
    h = c.roof_z                            # wall height (roof underside)
    ox = (fx1 + (-ix - wt)) / 2             # full-footprint x centre
    olen = fx1 - (-ix - wt)                 # full-footprint x length
    # back wall
    box("wall_back", (-ix - wt / 2, 0.0, h / 2), (wt, 2 * (iy + wt), h))
    # side walls (full depth, meet the front wall's outer face)
    for sgn, nm in ((1.0, "wall_l"), (-1.0, "wall_r")):
        box(nm, (ox, sgn * (iy + wt / 2), h / 2), (olen, wt, h))
    # roof (full footprint) — the interior has NO top access
    box("roof", (ox, 0.0, h + c.roof_t / 2), (olen, 2 * (iy + wt), c.roof_t),
        color=c.roof_color)
    # front wall: two side segments + header over the doorway
    seg_w = iy + wt - c.ap_hw
    for sgn, nm in ((1.0, "front_l"), (-1.0, "front_r")):
        box(nm, ((fx0 + fx1) / 2, sgn * (c.ap_hw + seg_w / 2), h / 2),
            (c.front_t, seg_w, h), color=c.front_color)
    box("header", ((fx0 + fx1) / 2, 0.0, (c.ap_h + h) / 2),
        (c.front_t, 2 * c.ap_hw, h - c.ap_h), color=c.front_color)
    # stop ribs at the tunnel's inner end: narrow the inner exit to 2*stop_hw
    for sgn, nm in ((1.0, "stop_l"), (-1.0, "stop_r")):
        box(nm, (fx0 + c.stop_t / 2, sgn * (c.stop_hw + (c.ap_hw - c.stop_hw) / 2),
                 c.ap_h / 2),
            (c.stop_t, c.ap_hw - c.stop_hw, c.ap_h), color=c.stop_color)
    return root


def _spawn_stopper(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a stopper: DYNAMIC compound — upright slab (its local +x is the OUT
    direction when seated) + grip boss centred on the outer face. Local origin at the
    SLAB CENTRE (standing rest height = slab_h/2). Sleep/stabilization thresholds
    zeroed (the solve drives it with external forces; a sleeping body silently
    ignores them)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.10)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg)
    c = cfg
    _add_box(stage, f"{prim_path}/slab", center=(0.0, 0.0, 0.0),
             size=(c.slab_d, c.slab_w, c.slab_h), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/boss",
             center=(c.slab_d / 2 + c.boss_d / 2, 0.0, c.boss_zc),
             size=(c.boss_d, c.boss_w, c.boss_h), color=c.boss_color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "vault" not in _SPAWNER_CACHE:

        @configclass
        class VaultSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_vault)
            in_hx: float = 0.105
            in_hy: float = 0.105
            wall_t: float = 0.012
            roof_z: float = 0.150
            roof_t: float = 0.012
            front_t: float = 0.030
            ap_hw: float = 0.052
            ap_h: float = 0.070
            stop_t: float = 0.006
            stop_hw: float = 0.040
            color: tuple = (0.76, 0.78, 0.82)
            front_color: tuple = (0.45, 0.50, 0.58)
            roof_color: tuple = (0.58, 0.62, 0.68)
            stop_color: tuple = (0.12, 0.13, 0.16)
            contact_offset: float = 0.002

        @configclass
        class StopperSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stopper)
            slab_d: float = 0.040
            slab_w: float = 0.084
            slab_h: float = 0.056
            boss_d: float = 0.032
            boss_w: float = 0.030
            boss_h: float = 0.024
            boss_zc: float = 0.012
            color: tuple = (0.15, 0.35, 0.75)
            boss_color: tuple = (0.10, 0.24, 0.55)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(vault=VaultSpawnerCfg, stopper=StopperSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class StopperVaultSceneCfg(BaseCfg):
    """Config for `StopperVaultScene`. The doorway tunnel is 104 mm wide x 70 mm tall
    x 30 mm deep through the thick front wall; the stop ribs narrow its inner exit to
    80 mm, so the 50 mm cube passes but the 84 mm blue stopper is arrested — that
    recess is the stopper pocket. An accepted seated stopper leaves side gaps
    <= 10 mm and a header gap of 14 mm: the 50 mm cube can NEVER enter past it,
    which physically forces cube-first execution order. The roof forbids drop-in;
    the ground plane is the interior floor and the doorway sill."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    inside_x_max: float = tunable(0.075)  # cube centre vault-x below this = fully inside (m)
    # (rear face then < 0.100, fully past the front wall's inner face at 0.105; a
    # Franka fingertip reaching ~40 mm through the aperture can leave the cube here)
    inside_y_max: float = tunable(0.078)  # |cube centre vault-y| below this (m)
    inside_z_max: float = tunable(0.120)  # under the roof (a cube on the roof is ~0.19)
    seat_x_tol: float = tunable(0.012)  # |stopper x - seat_x| below this = seated depth (m)
    seat_y_tol: float = tunable(0.015)  # |stopper y| in the vault frame (m)
    seat_z_tol: float = tunable(0.010)  # |stopper z - slab_h/2| (standing on the sill) (m)
    upright_min: float = tunable(0.978)  # stopper local +z . world up >= this (~12 deg)
    align_min: float = tunable(0.94)  # |stopper local +x . vault local +x| >= this (~20 deg)
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    vault_yaw_deg: float = tunable(8.0)  # vault yaw jitter about nominal (+/- deg)
    vault_jitter: float = tunable(0.02)  # vault xy jitter (+/- m)
    cube_x_range: tuple = tunable((0.14, 0.24))  # cube world x band (m)
    cube_y_band: tuple = tunable((0.20, 0.32))  # |cube world y| band (sign = its side) (m)
    stop_x_range: tuple = tunable((0.13, 0.21))  # stoppers world x band (m)
    slot_a_band: tuple = tunable((0.15, 0.20))  # stopper slot A |world y| band (m)
    slot_b_band: tuple = tunable((0.33, 0.38))  # stopper slot B |world y| band (m)
    swap_sides: bool = tunable(True)  # 50%: cube band and stopper band swap sides
    swap_slots: bool = tunable(True)  # 50%: blue/red swap slots A/B
    cube_yaw_deg: float = tunable(180.0)  # cube free yaw (+/- deg)
    stopper_yaw_deg: float = tunable(180.0)  # stopper free yaw (+/- deg)

    # --- info: structure (vault local frame: origin interior floor centre, +x = OUT) ------------
    vault_pos: tuple = info((0.50, 0.0))  # interior floor centre on the ground
    vault_yaw_nominal: float = info(180.0)  # deg; local +x (doorway) -> world -x
    in_hx: float = info(0.105)  # interior half-depth (back wall face -x, front face +x)
    in_hy: float = info(0.105)  # interior half-width
    wall_t: float = info(0.012)
    roof_z: float = info(0.150)  # roof UNDERSIDE (interior height)
    roof_t: float = info(0.012)
    front_t: float = info(0.030)  # front wall thickness = doorway tunnel depth
    ap_hw: float = info(0.052)  # doorway aperture half-width (104 mm)
    ap_h: float = info(0.070)  # doorway aperture height
    stop_t: float = info(0.006)  # stop rib thickness (tunnel inner end)
    stop_hw: float = info(0.040)  # inner-exit half-width between the ribs (80 mm)
    cube_s: float = info(0.050)  # green cube edge
    cube_mass: float = info(0.12)
    slab_d: float = info(0.040)  # stopper slab depth (x when seated)
    slab_w: float = info(0.084)  # BLUE slab width — fits the 96 mm aperture
    slab_h: float = info(0.056)  # slab height (14 mm header gap when seated)
    decoy_w: float = info(0.040)  # RED slab width — leaves >= 22 mm gaps, never seals
    boss_d: float = info(0.032)  # grip boss (its face ~48 mm outside the mouth, seated)
    boss_w: float = info(0.030)
    boss_h: float = info(0.024)
    boss_zc: float = info(0.012)  # boss centre above slab centre (world z ~40 mm)
    stopper_mass: float = info(0.18)
    vault_color: tuple = info((0.76, 0.78, 0.82))
    front_color: tuple = info((0.45, 0.50, 0.58))
    roof_color: tuple = info((0.58, 0.62, 0.68))
    stop_color: tuple = info((0.12, 0.13, 0.16))
    cube_color: tuple = info((0.10, 0.62, 0.15))
    blue_color: tuple = info((0.15, 0.35, 0.75))
    blue_boss_color: tuple = info((0.10, 0.24, 0.55))
    red_color: tuple = info((0.80, 0.10, 0.08))
    red_boss_color: tuple = info((0.55, 0.07, 0.06))
    contact_offset: float = info(0.002)
    # rubric weights (0.15 + 0.30 + 0.30 = 0.75 <= the 0.85 non-success cap)
    w_approach: float = info(0.15)
    w_deposit: float = info(0.30)
    w_seal: float = info(0.30)

    # Derived (filled in __post_init__).
    seat_x: float = field(default=None, init=False)  # seated slab-centre x, vault frame
    mouth_x: float = field(default=None, init=False)  # doorway outer face x, vault frame

    def __post_init__(self) -> None:
        # slab arrested when its inner face meets the stop ribs' outer face
        self.seat_x = round(self.in_hx + self.stop_t + self.slab_d / 2, 4)   # 0.131
        self.mouth_x = round(self.in_hx + self.front_t, 4)                   # 0.135


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("stopper_vault")
class StopperVaultScene(BaseScene):
    cfg: StopperVaultSceneCfg

    def __init__(self, cfg: StopperVaultSceneCfg | None = None) -> None:
        super().__init__(cfg or StopperVaultSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        vault_spawn = cls["vault"](
            mass_props=sim_utils.MassPropertiesCfg(mass=15.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            in_hx=c.in_hx, in_hy=c.in_hy, wall_t=c.wall_t, roof_z=c.roof_z,
            roof_t=c.roof_t, front_t=c.front_t, ap_hw=c.ap_hw, ap_h=c.ap_h,
            stop_t=c.stop_t, stop_hw=c.stop_hw, color=c.vault_color,
            front_color=c.front_color, roof_color=c.roof_color,
            stop_color=c.stop_color, contact_offset=c.contact_offset)
        blue_spawn = cls["stopper"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.stopper_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            slab_d=c.slab_d, slab_w=c.slab_w, slab_h=c.slab_h, boss_d=c.boss_d,
            boss_w=c.boss_w, boss_h=c.boss_h, boss_zc=c.boss_zc, color=c.blue_color,
            boss_color=c.blue_boss_color, contact_offset=c.contact_offset)
        red_spawn = cls["stopper"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.stopper_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            slab_d=c.slab_d, slab_w=c.decoy_w, slab_h=c.slab_h, boss_d=c.boss_d,
            boss_w=c.boss_w, boss_h=c.boss_h, boss_zc=c.boss_zc, color=c.red_color,
            boss_color=c.red_boss_color, contact_offset=c.contact_offset)
        cube_spawn = sim_utils.CuboidCfg(
            size=(c.cube_s, c.cube_s, c.cube_s),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5, linear_damping=0.05,
                angular_damping=0.05, sleep_threshold=0.0, stabilization_threshold=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=c.contact_offset, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.cube_color),
        )
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
            "vault": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Vault",
                spawn=vault_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.vault_pos[0], c.vault_pos[1], 0.0),
                    rot=(0.0, 0.0, 0.0, 1.0)),  # nominal yaw 180
            ),
            "cube": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/GreenCube",
                spawn=cube_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.19, -0.26, c.cube_s / 2 + 0.002)),
            ),
            "blue": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BlueStopper",
                spawn=blue_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.17, 0.175, c.slab_h / 2 + 0.002)),
            ),
            "red": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/RedStopper",
                spawn=red_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.17, 0.355, c.slab_h / 2 + 0.002)),
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
        self.vault: RigidObject = env.iscene["vault"]
        self.cube: RigidObject = env.iscene["cube"]
        self.blue: RigidObject = env.iscene["blue"]
        self.red: RigidObject = env.iscene["red"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        # latches: partial progress survives transient achievements (rubric requirement)
        self._approach_max = torch.zeros(n, device=env.device)
        self._deposited = torch.zeros(n, dtype=torch.bool, device=env.device)
        self._seal_max = torch.zeros(n, device=env.device)
        self._d0_cube = torch.full((n,), 0.4, device=env.device)  # spawn dist to mouth
        self._d0_blue = torch.full((n,), 0.5, device=env.device)  # spawn dist to seat

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the vault (yaw + xy jitter), stand the cube and the
        two stoppers in their (possibly side-swapped / slot-swapped) ground bands with
        free yaw; clear latches; store each mover's own spawn distance to its target
        (the normalizers that make null-policy credit exactly zero)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- vault: kinematic, yaw + xy jitter ---
        yaw = math.radians(c.vault_yaw_nominal) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.vault_yaw_deg)
        vx = c.vault_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.vault_jitter
        vy = c.vault_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.vault_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = vx, vy
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.vault.write_root_state_to_sim(st, env_ids)

        # --- side / slot swaps ---
        side = torch.where(  # cube side sign; stoppers get the other side
            (torch.rand(m, device=dev) < 0.5) if c.swap_sides
            else torch.zeros(m, dtype=torch.bool, device=dev),
            torch.ones(m, device=dev), -torch.ones(m, device=dev))
        slot_swap = (torch.rand(m, device=dev) < 0.5) if c.swap_slots \
            else torch.zeros(m, dtype=torch.bool, device=dev)

        def band(lo_hi: tuple, mm: int) -> torch.Tensor:
            return lo_hi[0] + torch.rand(mm, device=dev) * (lo_hi[1] - lo_hi[0])

        # --- cube: standing on the ground on its side of the vault axis, free yaw ---
        cx = band(c.cube_x_range, m)
        cy = side * band(c.cube_y_band, m)
        cyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.cube_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = cx, cy, c.cube_s / 2 + 0.002
        st[:, 3], st[:, 6] = torch.cos(cyaw / 2), torch.sin(cyaw / 2)
        st[:, 0:3] += origin
        self.cube.write_root_state_to_sim(st, env_ids)

        # --- stoppers: standing upright in slots A/B on the other side, free yaw ---
        ya = -side * band(c.slot_a_band, m)
        yb = -side * band(c.slot_b_band, m)
        blue_y = torch.where(slot_swap, yb, ya)
        red_y = torch.where(slot_swap, ya, yb)
        blue_x = torch.zeros(m, device=dev)
        for obj, oy in ((self.blue, blue_y), (self.red, red_y)):
            ox = band(c.stop_x_range, m)
            oyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.stopper_yaw_deg)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0], st[:, 1], st[:, 2] = ox, oy, c.slab_h / 2 + 0.002
            st[:, 3], st[:, 6] = torch.cos(oyaw / 2), torch.sin(oyaw / 2)
            st[:, 0:3] += origin
            obj.write_root_state_to_sim(st, env_ids)
            if obj is self.blue:
                blue_x = ox

        # --- clear latches; store the spawn-distance normalizers (vault frame) ---
        self._approach_max[env_ids] = 0.0
        self._deposited[env_ids] = False
        self._seal_max[env_ids] = 0.0
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

        def to_vault(wx: torch.Tensor, wy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            dx, dy = wx - vx, wy - vy
            return cos_y * dx + sin_y * dy, -sin_y * dx + cos_y * dy

        cxl, cyl = to_vault(cx, cy)
        self._d0_cube[env_ids] = torch.sqrt(
            (cxl - c.mouth_x) ** 2 + cyl ** 2 + (c.cube_s / 2 - 0.025) ** 2).clamp(min=0.05)
        bxl, byl = to_vault(blue_x, blue_y)
        self._d0_blue[env_ids] = torch.sqrt(
            (bxl - c.seat_x) ** 2 + byl ** 2).clamp(min=0.05)

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "vault": self.vault.data.root_state_w[env_ids].clone(),
            "cube": self.cube.data.root_state_w[env_ids].clone(),
            "blue": self.blue.data.root_state_w[env_ids].clone(),
            "red": self.red.data.root_state_w[env_ids].clone(),
            "approach_max": self._approach_max[env_ids].clone(),
            "deposited": self._deposited[env_ids].clone(),
            "seal_max": self._seal_max[env_ids].clone(),
            "d0_cube": self._d0_cube[env_ids].clone(),
            "d0_blue": self._d0_blue[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.vault.write_root_state_to_sim(state["vault"], env_ids)
        self.cube.write_root_state_to_sim(state["cube"], env_ids)
        self.blue.write_root_state_to_sim(state["blue"], env_ids)
        self.red.write_root_state_to_sim(state["red"], env_ids)
        self._approach_max[env_ids] = state["approach_max"]
        self._deposited[env_ids] = state["deposited"]
        self._seal_max[env_ids] = state["seal_max"]
        self._d0_cube[env_ids] = state["d0_cube"]
        self._d0_blue[env_ids] = state["d0_blue"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A grey VAULT stands on the ground: a roofed box (interior "
            f"{2 * c.in_hy * 100:.0f} cm wide, {2 * c.in_hx * 100:.0f} cm deep, "
            f"{c.roof_z * 100:.0f} cm tall) with no lid and no door — its ONLY access is an "
            f"open DOORWAY through its thick slate-blue front wall: an aperture "
            f"{2 * c.ap_hw * 100:.1f} cm wide and {c.ap_h * 100:.0f} cm tall forming a "
            f"{c.front_t * 100:.0f} cm deep tunnel at ground level (the ground is the sill — "
            f"no step). Two dark STOP RIBS line the tunnel's inner end, narrowing its exit "
            f"to {2 * c.stop_hw * 100:.0f} cm: this recess is the stopper POCKET. On the "
            f"ground outside stand three loose bodies: a GREEN cube "
            f"({c.cube_s * 100:.0f} cm) on one side of the doorway axis, and on the other "
            f"side two upright STOPPERS, each an upright slab with a protruding grip boss on "
            f"one face: a BLUE one ({c.slab_w * 100:.1f} cm wide — it fits the doorway) and "
            f"a RED one (only {c.decoy_w * 100:.1f} cm wide — far too narrow to close the "
            f"{2 * c.ap_hw * 100:.1f} cm doorway; it is a decoy and must be left out of the "
            f"vault and its doorway).\n"
            f"Goal, in this order: FIRST move the GREEN cube fully inside the vault — the "
            f"roof means nothing can be dropped in from above, so slide/push it along the "
            f"ground in through the doorway tunnel until it is completely past the front "
            f"wall. THEN close the vault: bring the BLUE stopper to the doorway, upright "
            f"with its slab facing the tunnel and its grip boss pointing outward, and slide "
            f"it into the pocket until the stop ribs arrest it (slab centre within "
            f"{c.seat_x_tol * 100:.1f} cm of the seat depth, upright and square to the "
            f"doorway; its boss remains outside as a handle). The order is forced: a seated "
            f"stopper leaves gaps far smaller than the cube, so the cube can no longer "
            f"enter. Success: the green cube at rest fully inside, the BLUE stopper at rest "
            f"seated in the pocket, and the RED stopper clear of the doorway and interior — "
            f"everything settled. Sealing an empty vault, sealing with the red stopper, or "
            f"leaving the cube in the tunnel all fail."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Push the green cube along the ground through the vault's open doorway until "
            "it is fully inside, then seal the doorway by sliding the wide BLUE stopper "
            "upright into the doorway pocket until the inner stop ribs arrest it. The cube "
            "must go in before the stopper; the narrow RED stopper is a decoy — leave it "
            "out."
        )

    # ----- progress / rubric ------------------------------------------------------------------------
    def _local(self, obj) -> torch.Tensor:
        """Object centre in the VAULT'S body frame, (N, 3) — containment and seating
        live in this frame so a yawed/jittered vault judges identically."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = obj.data.root_pos_w - self.vault.data.root_pos_w
        return quat_apply_inverse(self.vault.data.root_quat_w, rel)

    def _cube_inside_now(self) -> torch.Tensor:
        """(N,) bool: cube centre fully inside the interior (past the front wall's
        inner face with a margin of half a cube + 1 cm), under the roof."""
        c = self.cfg
        loc = self._local(self.cube)
        return (loc[:, 0] < c.inside_x_max) & (loc[:, 1].abs() < c.inside_y_max) \
            & (loc[:, 2] < c.inside_z_max) & (loc[:, 2] > 0.005)

    def _blue_seated_now(self) -> torch.Tensor:
        """(N,) bool: blue stopper standing in the pocket — slab centre at seat depth
        (arrested by the stop ribs), centred, standing on the sill, upright, and its
        slab square to the doorway (a sideways slab straddling the aperture fails the
        axis clause)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        loc = self._local(self.blue)
        ez = torch.tensor([0.0, 0.0, 1.0], device=loc.device).expand(n, 3)
        ex = torch.tensor([1.0, 0.0, 0.0], device=loc.device).expand(n, 3)
        up = quat_apply(self.blue.data.root_quat_w, ez)
        px = quat_apply(self.blue.data.root_quat_w, ex)
        vxw = quat_apply(self.vault.data.root_quat_w, ex)
        upright = up[:, 2] >= c.upright_min
        aligned = (px * vxw).sum(-1).abs() >= c.align_min
        pos_ok = ((loc[:, 0] - c.seat_x).abs() < c.seat_x_tol) \
            & (loc[:, 1].abs() < c.seat_y_tol) \
            & ((loc[:, 2] - c.slab_h / 2).abs() < c.seat_z_tol)
        return pos_ok & upright & aligned

    def _red_clear(self) -> torch.Tensor:
        """(N,) bool: red decoy NOT in the interior, tunnel or pocket region."""
        c = self.cfg
        loc = self._local(self.red)
        intruding = (loc[:, 0] < c.mouth_x) & (loc[:, 1].abs() < c.in_hy + 0.02) \
            & (loc[:, 2] < c.roof_z + 0.005)
        return ~intruding

    def _update_latches(self) -> None:
        """Refresh the latches: `approach` is the running max of the cube's progress
        toward the doorway mouth (normalized by its own spawn distance — exactly 0
        for the null policy); `deposited` once the cube is fully inside; `seal` is the
        running max of the blue stopper's progress toward its seat, counted ONLY
        while the cube is currently fully inside (order gating: seating the stopper
        first, or sealing an empty vault, earns nothing)."""
        c = self.cfg
        cl = self._local(self.cube)
        d_cube = torch.sqrt((cl[:, 0] - c.mouth_x) ** 2 + cl[:, 1] ** 2
                            + (cl[:, 2] - 0.025) ** 2)
        appr = (1.0 - d_cube / self._d0_cube).clamp(0.0, 1.0)
        self._approach_max = torch.maximum(self._approach_max, appr)
        inside = self._cube_inside_now()
        self._deposited |= inside
        bl = self._local(self.blue)
        d_blue = torch.sqrt((bl[:, 0] - c.seat_x) ** 2 + bl[:, 1] ** 2)
        seal = (1.0 - d_blue / self._d0_blue).clamp(0.0, 1.0)
        self._seal_max = torch.maximum(self._seal_max, seal * inside.float())

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: sealed containment — cube settled fully inside, BLUE stopper
        settled seated in the pocket, RED decoy clear of doorway + interior."""
        c = self.cfg
        self._update_latches()
        still = (self.cube.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.blue.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
        return self._cube_inside_now() & self._blue_seated_now() & self._red_clear() & still

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*approach + 0.30*deposited + 0.30*seal (all
        latched; seal gated on the cube being inside; ~0 for doing nothing) — capped
        at 0.85 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_approach * self._approach_max + c.w_deposit * self._deposited.float()
                + c.w_seal * self._seal_max).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="stopper_vault", robot="null"))

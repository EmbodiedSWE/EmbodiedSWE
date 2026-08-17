"""StoneDoorVaultScene — roll the blue millstone along its channel into the doorway
pocket to close the vault.

Derived from rlbench/close_microwave ("close microwave": push the open hinged door of a
microwave about its hinge until it shuts), but the CLOSURE MECHANISM and the required
plan are replaced wholesale. The seed's door is an attached 1-DoF panel and its entire
skill is one push toward the frame. Here the "door" is a FREE BODY — a heavy upright
disc (a rolling millstone, tomb-door style) standing in a guide channel that runs along
the FRONT of the vault wall. Closing means a controlled LONG-RANGE ROLL parallel to the
wall: drive the stone ~15-25 cm down its channel until it reaches the recessed POCKET
cut into the channel bed directly in front of the doorway, where it drops ~10 mm and
seats between the bed edges — a gravity detent produced purely by contact. Pushing the
stone toward the wall (the seed's motion) does nothing; the stone must be transported
ALONG the wall and captured by the pocket. A small RED DECOY block stands on the
opposite side of the track: seating the decoy earns nothing (and physically blocks the
pocket), so the solver must first discriminate the closure element by color and size.

Assets are fully procedural (the compound-spawner pattern — child colliders of one
kinematic body never self-collide):
  - vault: KINEMATIC wall (240 mm tall, 30 mm thick) with a doorway aperture (80 mm
    wide, 150 mm tall) at its center, plus the guide channel along its front face: two
    guide rails (45 mm tall) flanking a raised channel BED (10 mm tall, 48 mm wide)
    that runs the full track EXCEPT for a 90 mm gap — the POCKET — directly in front
    of the doorway. The pocket floor is the ground itself, 10 mm below the bed top.
    End caps close the channel at both ends.
  - stone: DYNAMIC BLUE disc (r 60 mm, 40 mm thick, 0.5 kg) standing upright in the
    channel on one side of the pocket. Rolled over the bed it cannot pass the pocket:
    at any quasi-static speed it drops in and seats (center bounded by construction to
    within ~12 mm of the pocket center; extraction needs a ~0.65 mg horizontal pull).
  - decoy: DYNAMIC RED square block (70 x 70 mm face, 40 mm thick, 0.2 kg) standing
    in the channel on the OTHER side of the pocket. It fits the pocket but is not the
    vault door. A block, not a disc: a standing disc is subject to a damping-immune
    GPU friction-solver creep (~8 mm/s along the track, seed-dependent) that could
    carry an untouched decoy into the pocket; a face-resting block cannot roll and
    stays put. (The stone never exhibited the creep in forge runs — larger contact
    line, 2.5x the mass — and it is driven anyway.)

Per-episode randomization (readback-verifiable): vault yaw +/- 30 deg + xy jitter,
WHICH SIDE of the pocket the stone starts on (the decoy takes the other), and both
discs' distances along the track.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve trajectory):
  0.15 * moved — the stone ever driven >= 70 mm along the track toward the pocket
    (the threshold sits ABOVE the free-stone wander band: an untouched standing disc
    exhibits a bounded, seed-dependent GPU-solver wander of up to ~45 mm before
    arresting — measured over 10 s on 5 seeds — so no credit accrues force-free)
  0.25 * near  — the stone ever within 75 mm of the pocket center (in-channel)
  0.20 * in    — the stone ever dropped into the pocket (in-channel, below bed height)
  1.0 iff success() — the stone SEATED in the pocket: vault-frame |y| within pos_tol
    of the pocket center, center height in the seated band (on the pocket floor, not
    perched on the bed), rolling plane upright between the rails, settled and finite.
  Non-success capped at 0.60. All credit is measured on the BLUE stone only.

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


def _spawn_vault(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the vault at `prim_path`: KINEMATIC compound. Local frame: origin on the
    ground at the wall center; the wall runs along local y, its FRONT faces local +x.

    Children: two wall slabs flanking the doorway aperture, the lintel above it, the
    inner and outer guide rails (full track length), the two channel-bed segments
    (raised 10 mm, interrupted by the pocket gap in front of the doorway — the pocket
    floor is the world ground), and two channel end caps."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    wall, guide = c.wall_color, c.guide_color
    half = c.track_half
    # wall slabs (doorway aperture |y| < door_hw, z < door_h) + lintel
    slab_len = half - c.door_hw
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/wall_{'p' if sgn > 0 else 'n'}",
                 center=(-c.wall_t / 2, sgn * (c.door_hw + slab_len / 2), c.wall_h / 2),
                 size=(c.wall_t, slab_len, c.wall_h), color=wall, collide=collide)
    _add_box(stage, f"{prim_path}/lintel",
             center=(-c.wall_t / 2, 0.0, (c.door_h + c.wall_h) / 2),
             size=(c.wall_t, 2 * c.door_hw, c.wall_h - c.door_h), color=wall,
             collide=collide)
    # guide rails (full track length, both sides of the channel)
    _add_box(stage, f"{prim_path}/rail_in",
             center=((c.rail_in_x0 + c.rail_in_x1) / 2, 0.0, c.rail_h / 2),
             size=(c.rail_in_x1 - c.rail_in_x0, 2 * half, c.rail_h), color=guide,
             collide=collide)
    _add_box(stage, f"{prim_path}/rail_out",
             center=((c.rail_out_x0 + c.rail_out_x1) / 2, 0.0, c.rail_h / 2),
             size=(c.rail_out_x1 - c.rail_out_x0, 2 * half, c.rail_h), color=guide,
             collide=collide)
    # channel bed segments (raised bed interrupted by the pocket gap at y = 0)
    bed_len = half - c.pocket_hl
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/bed_{'p' if sgn > 0 else 'n'}",
                 center=((c.rail_in_x1 + c.rail_out_x0) / 2,
                         sgn * (c.pocket_hl + bed_len / 2), c.bed_h / 2),
                 size=(c.rail_out_x0 - c.rail_in_x1, bed_len, c.bed_h), color=guide,
                 collide=collide)
    # channel end caps
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/cap_{'p' if sgn > 0 else 'n'}",
                 center=((c.rail_in_x1 + c.rail_out_x0) / 2, sgn * c.cap_y, c.rail_h / 2),
                 size=(c.rail_out_x0 - c.rail_in_x1, 0.012, c.rail_h), color=guide,
                 collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclass (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "vault" not in _SPAWNER_CACHE:

        @configclass
        class VaultSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_vault)
            wall_t: float = 0.030
            wall_h: float = 0.240
            door_hw: float = 0.040
            door_h: float = 0.150
            rail_in_x0: float = 0.002
            rail_in_x1: float = 0.012
            rail_out_x0: float = 0.060
            rail_out_x1: float = 0.070
            rail_h: float = 0.045
            bed_h: float = 0.010
            track_half: float = 0.42
            cap_y: float = 0.404
            pocket_hl: float = 0.045
            wall_color: tuple = (0.30, 0.30, 0.34)
            guide_color: tuple = (0.55, 0.55, 0.58)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["vault"] = VaultSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class StoneDoorVaultSceneCfg(BaseCfg):
    """Config for `StoneDoorVaultScene`. The seated tolerances are honest by
    construction: a stone physically resting on the pocket floor is bounded by the bed
    edges to within ~12 mm of the pocket center (pos_tol 20 mm) and rests with its
    center at ~stone_r above the ground (seated band 45..65 mm rejects both a stone
    perched on the 10 mm bed, ~70 mm, and one lying flat, ~20 mm)."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    pos_tol: float = tunable(0.020)        # |vault-frame y| of the seated stone (bed bounds 12 mm)
    seat_z_min: float = tunable(0.045)     # seated stone center height band (m, above ground) —
    seat_z_max: float = tunable(0.065)     # on the pocket floor: ~60; perched on the bed: ~70
    chan_x_tol: float = tunable(0.020)     # |vault-frame x - channel center| (rails bound ~4 mm)
    upright_max_deg: float = tunable(15.0)  # rolling plane within this of vertical (rails bound ~5)
    settle_lin: float = tunable(0.08)      # max |lin vel| when judging (above the GPU creep band)
    settle_ang: float = tunable(0.60)      # max |ang vel| when judging (above the phantom band)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    vault_yaw_deg: float = tunable(30.0)   # vault yaw about its nominal heading (+/- deg)
    vault_jitter: float = tunable(0.05)    # vault xy jitter (+/- m)
    stone_dist: tuple = tunable((0.14, 0.26))  # stone start |y| along the track (m)
    decoy_dist: tuple = tunable((0.14, 0.24))  # decoy start |y| along the track (m)
    side_swap: bool = tunable(True)        # randomize WHICH side of the pocket the stone starts on

    # --- info: layout (world nominal; the vault front faces its local +x) -----------------------
    vault_pos: tuple = info((0.42, 0.0))   # vault origin on the ground (nominal)
    vault_yaw_nom_deg: float = info(180.0)  # nominal heading: front faces world -x (the robot)
    # --- info: vault structure (local frame: origin on the ground at the wall center) -----------
    wall_t: float = info(0.030)
    wall_h: float = info(0.240)
    door_hw: float = info(0.040)   # doorway aperture half-width
    door_h: float = info(0.150)    # doorway aperture height
    rail_h: float = info(0.045)    # guide rail height
    bed_h: float = info(0.010)     # channel bed height (= the pocket drop)
    chan_x: float = info(0.036)    # channel centerline (local x)
    track_half: float = info(0.42)
    pocket_hl: float = info(0.045)  # pocket half-length along the track
    # --- info: discs -----------------------------------------------------------------------------
    stone_r: float = info(0.060)
    stone_t: float = info(0.040)
    stone_mass: float = info(0.50)
    decoy_s: float = info(0.070)   # decoy block face edge (y and z)
    decoy_t: float = info(0.040)   # decoy block thickness (across the channel)
    decoy_mass: float = info(0.20)
    stone_color: tuple = info((0.10, 0.30, 0.85))
    decoy_color: tuple = info((0.85, 0.10, 0.10))
    contact_offset: float = info(0.002)
    # --- info: latch thresholds and rubric weights (0.15 + 0.25 + 0.20 = 0.60 cap) ---------------
    moved_d: float = info(0.070)   # `moved` latch: driven this far along the track toward y=0
    #                                (above the ~45 mm bounded free-stone wander band)
    near_y: float = info(0.075)    # `near` latch: |y| below this, in-channel
    in_y: float = info(0.030)      # `in` latch: |y| below this AND center below in_z (dropped in)
    in_z: float = info(0.066)
    w_moved: float = info(0.15)
    w_near: float = info(0.25)
    w_in: float = info(0.20)


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
@SCENES.register("stone_door_vault")
class StoneDoorVaultScene(BaseScene):
    cfg: StoneDoorVaultSceneCfg

    def __init__(self, cfg: StoneDoorVaultSceneCfg | None = None) -> None:
        super().__init__(cfg or StoneDoorVaultSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        vault_spawn = cls["vault"](
            mass_props=sim_utils.MassPropertiesCfg(mass=40.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            wall_h=c.wall_h, wall_t=c.wall_t, door_hw=c.door_hw, door_h=c.door_h,
            rail_h=c.rail_h, bed_h=c.bed_h, track_half=c.track_half,
            pocket_hl=c.pocket_hl, contact_offset=c.contact_offset)

        # Shared dynamic-body props.
        # - velocity iterations 4: mitigates the GPU creep artifact on free rolling
        #   convex bodies; damping 0.35 bounds any set-down-jolt free roll of the
        #   stone to ~25 mm (a level bed is otherwise loss-free for a rolling disc).
        # - sleep disabled: all gates read live velocities.
        rigid = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5,
            linear_damping=0.35, angular_damping=0.35,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4)
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)
        mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.6, dynamic_friction=0.5, restitution=0.0)

        def disc_cfg(radius: float, height: float, mass: float, color: tuple):
            return sim_utils.CylinderCfg(
                radius=radius, height=height, axis="Z",
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                rigid_props=rigid, collision_props=coll, physics_material=mat,
            )

        def block_cfg(size: tuple, mass: float, color: tuple):
            # The decoy is a face-resting BLOCK: a standing disc picks up a
            # damping-immune GPU friction-solver creep (~8 mm/s along the track,
            # seed-dependent) that could carry the untouched decoy into the pocket;
            # a block cannot roll and stays put.
            return sim_utils.CuboidCfg(
                size=size,
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                rigid_props=rigid, collision_props=coll, physics_material=mat,
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
                    pos=(c.vault_pos[0], c.vault_pos[1], 0.0)),
            ),
            "stone": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stone",
                spawn=disc_cfg(c.stone_r, c.stone_t, c.stone_mass, c.stone_color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 1.0, 0.10)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=block_cfg((c.decoy_t, c.decoy_s, c.decoy_s), c.decoy_mass,
                                c.decoy_color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 1.3, 0.10)),
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
        self.stone: RigidObject = env.iscene["stone"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # stone_side[e] = +1 / -1: sign of the track side the STONE starts on
        self.stone_side = torch.ones(n, dtype=torch.float, device=dev)
        self._dist0 = torch.zeros(n, dtype=torch.float, device=dev)  # stone start |y|
        # latches (partial credit survives transients; success is judged live)
        self._moved = torch.zeros(n, dtype=torch.bool, device=dev)
        self._near = torch.zeros(n, dtype=torch.bool, device=dev)
        self._in = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the vault (yaw + xy jitter), stand the stone upright in
        the channel on a random side of the pocket at a random distance, stand the
        decoy on the OTHER side, clear the latches."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        _ = torch.rand(3, m, device=dev)  # burn post-seed draws (first draw degenerate)

        # --- vault: kinematic, nominal heading + yaw + xy jitter ---
        yaw = math.radians(c.vault_yaw_nom_deg) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.vault_yaw_deg)
        q_vault = _qz(yaw)
        vp = torch.zeros(m, 3, device=dev)
        vp[:, 0] = c.vault_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.vault_jitter
        vp[:, 1] = c.vault_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.vault_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = vp + origin
        st[:, 3:7] = q_vault
        self.vault.write_root_state_to_sim(st, env_ids)

        # --- side assignment + distances ---
        if c.side_swap:
            side = torch.where(torch.rand(m, device=dev) < 0.5,
                               torch.ones(m, device=dev), -torch.ones(m, device=dev))
        else:
            side = torch.ones(m, device=dev)
        self.stone_side[env_ids] = side
        d_stone = c.stone_dist[0] + torch.rand(m, device=dev) \
            * (c.stone_dist[1] - c.stone_dist[0])
        d_decoy = c.decoy_dist[0] + torch.rand(m, device=dev) \
            * (c.decoy_dist[1] - c.decoy_dist[0])
        self._dist0[env_ids] = d_stone

        # --- stone upright in the channel (cylinder axis across the channel = local x),
        #     decoy block face-resting on the bed on the other side;
        #     0.5 mm set-down gap (larger drops jolt the free stone into long rolls) ---
        q_disc = _qmul(q_vault, _qy(torch.full((m,), math.pi / 2, device=dev)))
        for body, dist, sgn, half_h, q in (
                (self.stone, d_stone, side, c.stone_r, q_disc),
                (self.decoy, d_decoy, -side, c.decoy_s / 2, q_vault)):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = c.chan_x
            loc[:, 1] = sgn * dist
            loc[:, 2] = c.bed_h + half_h + 0.0005
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = vp + quat_apply(q_vault, loc) + origin
            st[:, 3:7] = q
            body.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._moved[env_ids] = False
        self._near[env_ids] = False
        self._in[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "vault": self.vault.data.root_state_w[env_ids].clone(),
            "stone": self.stone.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "stone_side": self.stone_side[env_ids].clone(),
            "dist0": self._dist0[env_ids].clone(),
            "moved": self._moved[env_ids].clone(),
            "near": self._near[env_ids].clone(),
            "in": self._in[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.vault.write_root_state_to_sim(state["vault"], env_ids)
        self.stone.write_root_state_to_sim(state["stone"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.stone_side[env_ids] = state["stone_side"]
        self._dist0[env_ids] = state["dist0"]
        self._moved[env_ids] = state["moved"]
        self._near[env_ids] = state["near"]
        self._in[env_ids] = state["in"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A dark-grey VAULT WALL ({c.wall_h * 1000:.0f} mm tall) stands on the "
            f"ground, facing you. At its center is an open DOORWAY "
            f"({2 * c.door_hw * 1000:.0f} mm wide, {c.door_h * 1000:.0f} mm tall). The "
            f"doorway has no hinged door: along the FRONT of the wall runs a straight "
            f"GUIDE CHANNEL — two light-grey rails ({c.rail_h * 1000:.0f} mm tall) "
            f"flanking a raised bed strip {c.bed_h * 1000:.0f} mm high — and directly "
            f"in front of the doorway the bed is interrupted by a RECESSED POCKET (a "
            f"{2 * c.pocket_hl * 1000:.0f} mm-long gap down to ground level). In the "
            f"channel, one on each side of the pocket, stand two objects: a large "
            f"round BLUE STONE (the rolling vault door, a disc "
            f"{2 * c.stone_r * 1000:.0f} mm across, {c.stone_t * 1000:.0f} mm thick) "
            f"and a small square RED BLOCK ({c.decoy_s * 1000:.0f} mm) that is NOT "
            f"the door. Which side each object starts on, their distances from the "
            f"pocket, and the wall's position and heading all vary per episode — "
            f"look at the colors and shapes.\n"
            f"Goal: close the vault by rolling the BLUE stone along the channel — "
            f"moving it parallel to the wall, toward the doorway — until it reaches "
            f"the pocket, drops in, and SEATS on the pocket floor in front of the "
            f"doorway. The final state must have the blue stone resting upright in "
            f"the pocket (down on the pocket floor between the bed edges, its faces "
            f"between the rails), at rest. The stone is heavy: drive it steadily "
            f"along the track; the pocket will capture it. Putting the RED block in "
            f"the pocket earns nothing and blocks the door seat — leave it where it "
            f"is. A stone left short of the pocket, driven past onto the far bed, "
            f"perched on the bed edge, or lying flat earns no success."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Roll the large blue stone disc along the guide channel in front of the "
            "wall until it drops into the recessed pocket before the doorway, and "
            "leave it seated upright there. Do not put the small red block in the "
            "pocket."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _vault_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the (kinematic, live-read) vault frame, (N,3) -> (N,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.vault.data.root_quat_w,
                                  pos_w - self.vault.data.root_pos_w)

    def _in_channel(self, loc: torch.Tensor, x_tol: float) -> torch.Tensor:
        """(N,) bool: vault-local point laterally within the channel."""
        return (loc[:, 0] - self.cfg.chan_x).abs() < x_tol

    def _upright(self) -> torch.Tensor:
        """(N,) bool: the stone's rolling plane vertical — its cylinder axis (body +z)
        horizontal within `upright_max_deg`."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        ax = quat_apply(self.stone.data.root_quat_w, ez)
        return ax[:, 2].abs() <= math.sin(math.radians(self.cfg.upright_max_deg))

    def stone_seated(self) -> torch.Tensor:
        """(N,) bool, geometric: the stone SEATED in the doorway pocket — vault-frame
        |y| within `pos_tol` of the pocket center, laterally in the channel, center
        height in the seated band (resting on the pocket floor: ~stone_r; a stone
        perched on the bed sits ~10 mm higher, one lying flat ~40 mm lower), rolling
        plane upright. Honest by construction: the bed edges bound a stone physically
        on the pocket floor to |y| <~ 12 mm."""
        c = self.cfg
        loc = self._vault_local(self.stone.data.root_pos_w)
        z = self.stone.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        return (loc[:, 1].abs() < c.pos_tol) & self._in_channel(loc, c.chan_x_tol) \
            & (z > c.seat_z_min) & (z < c.seat_z_max) & self._upright()

    def settled(self) -> torch.Tensor:
        """(N,) bool: stone lin AND ang velocity below the settle gates."""
        lin = self.stone.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin
        ang = self.stone.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang
        return lin & ang

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w for b in (self.stone, self.decoy)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        loc = self._vault_local(self.stone.data.root_pos_w)
        z = self.stone.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        absy = loc[:, 1].abs()
        fin = self._finite()
        self._moved |= fin & ((self._dist0 - absy) > c.moved_d)
        self._near |= fin & (absy < c.near_y) & self._in_channel(loc, 0.05)
        self._in |= fin & (absy < c.in_y) & (z < c.in_z) \
            & self._in_channel(loc, c.chan_x_tol)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the blue stone seated in the doorway pocket, settled and finite.
        All clauses are live physical outcomes."""
        self._update_latches()
        return self.stone_seated() & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*moved + 0.25*near + 0.20*in (all latched on the
        BLUE stone; ~0 for doing nothing, ~0 for driving the decoy), capped at 0.60 —
        and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_moved * self._moved.float() + c.w_near * self._near.float()
                + c.w_in * self._in.float()).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="stone_door_vault", robot="null"))

"""JengaQuarryScene — slide the two buried red blocks out of a pre-built crisscross
tower (Jenga-style, under the load of the layers above) and deposit them in the blue
tray, WITHOUT disturbing any cream block. Derived from rlbench/block_pyramid, with
the seed's strategy inverted.

Seed (rlbench/block_pyramid): six loose green cubes among red distractors must be
STACKED UP into a 3-2-1 pyramid — a constructive task: repeated free-space
pick-and-place, judged on the assembled end pose, nothing starts in contact with
anything and no existing structure constrains the moves.

Here everything is inverted, so a solver needs a different PLAN and different CODE
STRUCTURE:

- The structure ALREADY EXISTS at reset: a 3-layer x 3-block crisscross tower
  (alternating x/y layers, like a Jenga stack) on a plinth. Nothing is built; two
  marked blocks are EXTRACTED from it.
- The judged interaction is a SLIDE UNDER LOAD, not a placement: each red target
  block is buried in the bottom or middle layer, fully roofed by the perpendicular
  layer above. It cannot be lifted out (raising it must raise the whole
  superstructure, which displaces cream blocks and fails the task) — it can only
  be slid out lengthwise along its own axis, Jenga-style, with the weight of the
  blocks above riding on it the whole way.
- A PRESERVATION constraint replaces the seed's assembly goal: success requires
  every one of the seven cream blocks to still sit in its original slot (position,
  height, and tilt tolerances) when the episode is judged. Toppling the tower to
  get at the reds — the "just knock it over and pick from the rubble" shortcut —
  is rejected by construction, and every partial-credit latch is gated on the
  tower still standing at the moment the credit is earned.
- Selection is positional, not just color: WHICH slot of each layer holds a red
  block is randomized per episode, so a memorized extraction sequence fails.

Apparatus (fully procedural):
  - kinematic PLINTH (9.3 cm square, 6 cm tall) carrying the tower;
  - 9 dynamic blocks (13 x 3 x 1.8 cm, 100 g): 2 red targets, 7 cream. Layer 0
    (bottom, on the plinth) runs along x, layer 1 along y, layer 2 along x; each
    layer holds 3 blocks on a 3.1 cm pitch (1 mm side gaps). Block ends protrude
    ~1.85 cm past the tower span on both sides — the grasp/push feature;
  - kinematic blue TRAY (16 cm square inner, 4 cm walls) on the table, offset from
    the tower.

Per-episode randomization (readback-verifiable): rig xy + yaw, WHICH slot of layer
0 and of layer 1 is red, tray xy jitter.

Rubric (0..1; partial credit latched, every latch gated on tower_standing() at the
moment it is earned):
  0.05 / red  — slid: the red displaced > 3 cm along its slot axis while still in
                its channel (same height, same cross-line) — the Jenga slide;
  0.10 / red  — extracted: the red fully clear of the tower footprint;
  0.20 / red  — banked: the red inside the tray volume;
  1.0 iff success() — both reds settled in the tray, all 7 cream blocks still in
                their slots, everything still. Non-success cap 0.70. Null ~0.

Heavy imports (isaaclab) are deferred so importing this module stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class JengaQuarrySceneCfg(BaseCfg):
    """Config for `JengaQuarryScene`. The interlock is gravitational: each red block
    is roofed by the perpendicular layer above, so the only extraction that leaves
    the tower standing is the lengthwise Jenga slide; prying a red upward must lift
    the whole superstructure and displaces cream blocks (preservation failure)."""

    # --- tunable: randomization (task-family knobs) ---------------------------------------------
    rig_jitter: float = tunable(0.05)  # +/- xy jitter on the whole rig
    rig_yaw_deg: float = tunable(30.0)  # +/- rig yaw (limited: one Franka base pose stays valid)
    tray_jitter: float = tunable(0.03)  # +/- xy jitter on the tray (rig-local)

    # --- tunable: rubric thresholds -------------------------------------------------------------
    settle_speed: float = tunable(0.10)  # max |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.90)  # max |ang vel| when judging (rad/s)
    stand_xy_tol: float = tunable(0.030)  # cream block xy drift bound (m)
    stand_z_tol: float = tunable(0.012)  # cream block height drift bound (< one block height)
    stand_tilt_deg: float = tunable(15.0)  # cream block tilt bound
    slid_min: float = tunable(0.030)  # in-channel displacement that earns the slide latch
    slid_cross_max: float = tunable(0.012)  # ...while within this of the slot centreline
    slid_dz_max: float = tunable(0.008)  # ...and at slot height (teleports/pry-ups do not count)
    ext_dist: float = tunable(0.135)  # horizontal clearance from the tower axis = extracted
    tray_xy_tol: float = tunable(0.070)  # red centre within this box of the tray centre
    tray_z: tuple = tunable((0.010, 0.046))  # red centre resting band inside the tray

    # --- info: geometry (rig-local frame: origin = tower axis at table level) -------------------
    block_size: tuple = info((0.130, 0.030, 0.018))  # l x w x h
    block_gap: float = info(0.001)  # side gap between neighbouring blocks in a layer
    plinth_size: tuple = info((0.093, 0.093, 0.060))
    tray_pos: tuple = info((0.34, 0.0))  # tray centre (rig-local, before jitter)
    tray_inner: float = info(0.160)  # square inner side
    tray_wall_t: float = info(0.008)
    tray_wall_h: float = info(0.040)  # above the tray floor top
    tray_floor_t: float = info(0.008)

    # --- info: dynamics -------------------------------------------------------------------------
    block_mass: float = info(0.10)
    block_friction: float = info(0.18)  # block/block (pair-averaged); low, like waxed jenga wood
    # NOTE: the plinth + tray carry the GROUND material, so block/plinth ends up at
    # (0.18+0.60)/2 ~ 0.39 pair-averaged — the stationary bottom blocks anchor to the
    # plinth harder than the sliding red drags its roof, keeping tower drift low.
    ground_friction: float = info(0.60)
    contact_offset: float = info(0.002)

    # --- info: colors ---------------------------------------------------------------------------
    red_color: tuple = info((0.90, 0.08, 0.08))
    cream_color: tuple = info((0.87, 0.78, 0.58))
    plinth_color: tuple = info((0.45, 0.45, 0.50))
    tray_color: tuple = info((0.15, 0.30, 0.90))

    # --- info: rubric weights -------------------------------------------------------------------
    w_slid: float = info(0.05)
    w_ext: float = info(0.10)
    w_tray: float = info(0.20)  # 2*(0.05+0.10+0.20) = 0.70 = the non-success cap

    # derived landmarks --------------------------------------------------------------------------
    @property
    def pitch(self) -> float:
        return self.block_size[1] + self.block_gap  # 0.031

    @property
    def span_half(self) -> float:
        return self.pitch + self.block_size[1] / 2  # 0.046: outer edge of a 3-slot layer

    @property
    def plinth_top(self) -> float:
        return self.plinth_size[2]

    def layer_z(self, layer: int) -> float:
        h = self.block_size[2]
        return self.plinth_top + h / 2 + layer * h + 0.0004 * (layer + 1)

    @property
    def tray_floor_top(self) -> float:
        return self.tray_floor_t

    def __post_init__(self) -> None:
        length, w, h = self.block_size
        # grasp/push feature: block ends protrude past the layer span AND the plinth
        protrusion = length / 2 - self.span_half
        assert protrusion >= 0.015, f"end protrusion {protrusion:.4f} too small to grasp"
        assert length / 2 - self.plinth_size[0] / 2 >= 0.015, \
            "bottom-layer ends must overhang the plinth for a pinch grasp"
        # plinth supports the whole bottom layer footprint
        assert self.plinth_size[1] / 2 >= self.span_half, "plinth narrower than the layer span"
        # a red still pinned in its channel can never earn the extraction latch: the
        # slide leaves the roof (and its support) at span_half + l/2 travel
        slide_out = self.span_half + length / 2
        assert self.ext_dist >= slide_out + 0.02, \
            f"ext_dist {self.ext_dist} inside the pinned slide range ({slide_out:.3f})"
        # the tray is well outside the extraction fall zone
        tray_near = self.tray_pos[0] - self.tray_inner / 2 - self.tray_wall_t - self.tray_jitter
        assert tray_near > self.ext_dist + length / 2, "tray overlaps the extraction zone"
        # both blocks fit flat in the tray, and the containment box is honest: any
        # physically-inside rest pose passes (max centre offset = inner/2 - w/2)
        assert self.tray_inner >= length + 0.02, "tray too small for a block"
        assert self.tray_inner / 2 - w / 2 <= self.tray_xy_tol + 1e-6, \
            "tray_xy_tol rejects a physically-inside rest pose"
        # the z band rejects a block perched on the tray rim
        rim_rest = self.tray_floor_top + self.tray_wall_h + h / 2
        assert rim_rest > self.tray_z[1] + 0.005, "tray z band admits a rim perch"
        # ...but accepts flat and stacked-on-one-block rests on the tray floor
        assert self.tray_z[0] < self.tray_floor_top + h / 2 < self.tray_z[1], "flat rest outside band"
        assert self.tray_z[0] < self.tray_floor_top + 1.5 * h < self.tray_z[1], "stacked rest outside band"
        # preservation tolerances: a one-slot shift or a one-layer drop is a failure
        assert self.stand_xy_tol < self.pitch, "xy tolerance admits a one-slot shift"
        assert self.stand_z_tol < h, "z tolerance admits a one-layer drop"
        # slide latch fires only inside the channel
        assert self.slid_min < protrusion + w, "slide latch needs too much travel"
        assert self.slid_cross_max < self.pitch - w / 2, "cross tolerance reaches the next slot"


# ----- kinematic furniture ---------------------------------------------------------------------
def _tray_pieces(c: JengaQuarrySceneCfg) -> list[tuple[str, tuple, tuple]]:
    """Tray pieces as (name, offset-from-tray-centre, size). z offsets from the table."""
    inner, t, wh, ft = c.tray_inner, c.tray_wall_t, c.tray_wall_h, c.tray_floor_t
    wz = ft + wh / 2
    off = inner / 2 + t / 2
    span = inner + 2 * t
    return [
        ("tray_floor", (0.0, 0.0, ft / 2), (span, span, ft)),
        ("tray_n", (0.0, off, wz), (span, t, wh)),
        ("tray_s", (0.0, -off, wz), (span, t, wh)),
        ("tray_e", (off, 0.0, wz), (t, inner, wh)),
        ("tray_w", (-off, 0.0, wz), (t, inner, wh)),
    ]


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("jenga_quarry")
class JengaQuarryScene(BaseScene):
    cfg: JengaQuarrySceneCfg

    # block order: 0-1 the red targets (red0 -> layer 0, red1 -> layer 1), 2-8 cream
    BLOCKS = ("red0", "red1", "cream0", "cream1", "cream2", "cream3",
              "cream4", "cream5", "cream6")

    def __init__(self, cfg: JengaQuarrySceneCfg | None = None) -> None:
        super().__init__(cfg or JengaQuarrySceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        # dynamic << static: the slipping interface (the slid red vs its roof) drags
        # at HALF what the stationary supports hold statically, so a quasi-static
        # Jenga slide leaves the rest of the tower stuck (waxed-wood behaviour).
        block_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.block_friction, dynamic_friction=c.block_friction * 0.5,
            restitution=0.0)
        floor_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.ground_friction, dynamic_friction=c.ground_friction * 0.9,
            restitution=0.0)

        def kin_box(name: str, pos: tuple, size: tuple, color: tuple) -> RigidObjectCfg:
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=floor_mat,  # anchor: high-friction furniture
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
            )

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(physics_material=floor_mat),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "plinth": kin_box("Plinth", (0.0, 0.0, c.plinth_size[2] / 2), c.plinth_size,
                              c.plinth_color),
        }
        for name, off, size in _tray_pieces(c):
            out[name] = kin_box(name.capitalize(),
                                (c.tray_pos[0] + off[0], c.tray_pos[1] + off[1], off[2]),
                                size, c.tray_color)
        for i, name in enumerate(self.BLOCKS):
            color = c.red_color if i < 2 else c.cream_color
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Block_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=c.block_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.15,
                        angular_damping=0.15,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.block_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=block_mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, (i - 4) * 0.05, self.cfg.layer_z(0))),
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
        c = self.cfg
        self.blocks: list[RigidObject] = [env.iscene[nm] for nm in self.BLOCKS]
        self.plinth: RigidObject = env.iscene["plinth"]
        self.tray: dict[str, RigidObject] = {
            nm: env.iscene[nm] for nm, _off, _sz in _tray_pieces(c)}
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self._rig_xy = torch.zeros(n, 2, device=dev)
        self._rig_yaw = torch.zeros(n, device=dev)
        self._tray_xy = torch.zeros(n, 2, device=dev)  # rig-local tray centre
        self._slot = torch.zeros(n, 2, dtype=torch.long, device=dev)  # red slot per layer
        self._rst_local = torch.zeros(n, 9, 3, device=dev)  # per-block reset slot (rig-local)
        # red slot axes are fixed by layer parity: red0 slides along local x, red1 along y
        self._axis = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=dev)
        # latches (gated on tower_standing() at earn time)
        self._slid = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self._ext = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self._tray_in = torch.zeros(n, 2, dtype=torch.bool, device=dev)

    # ----- frame helpers -------------------------------------------------------------------------
    def _rig_quat(self, env_ids: torch.Tensor | None = None, extra_yaw: float = 0.0) -> torch.Tensor:
        yaw = (self._rig_yaw if env_ids is None else self._rig_yaw[env_ids]) + extra_yaw
        half = yaw / 2
        q = torch.zeros(len(yaw), 4, device=yaw.device)
        q[:, 0] = torch.cos(half)
        q[:, 3] = torch.sin(half)
        return q

    def to_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world -> rig-local (origin at tower axis, table z=0)."""
        p = (pos_w - self.env_origins).clone()
        p[:, 0:2] -= self._rig_xy
        cy, sy = torch.cos(self._rig_yaw), torch.sin(self._rig_yaw)
        out = torch.empty_like(p)
        out[:, 0] = cy * p[:, 0] + sy * p[:, 1]
        out[:, 1] = -sy * p[:, 0] + cy * p[:, 1]
        out[:, 2] = p[:, 2]
        return out

    def to_world(self, pos_l: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """(M,3) rig-local -> world (without env origins)."""
        yaw = self._rig_yaw[env_ids]
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        out = torch.empty_like(pos_l)
        out[:, 0] = cy * pos_l[:, 0] - sy * pos_l[:, 1] + self._rig_xy[env_ids, 0]
        out[:, 1] = sy * pos_l[:, 0] + cy * pos_l[:, 1] + self._rig_xy[env_ids, 1]
        out[:, 2] = pos_l[:, 2]
        return out

    def _write_furniture(self, env_ids: torch.Tensor) -> None:
        """Re-pose the kinematic plinth + tray from the stored rig pose."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        q = self._rig_quat(env_ids)
        st = torch.zeros(m, 13, device=dev)
        local = torch.tensor([0.0, 0.0, c.plinth_size[2] / 2], device=dev).expand(m, 3).clone()
        st[:, 0:3] = self.to_world(local, env_ids) + origin
        st[:, 3:7] = q
        self.plinth.write_root_state_to_sim(st, env_ids)
        for nm, off, _sz in _tray_pieces(c):
            local = torch.zeros(m, 3, device=dev)
            local[:, 0] = self._tray_xy[env_ids, 0] + off[0]
            local[:, 1] = self._tray_xy[env_ids, 1] + off[1]
            local[:, 2] = off[2]
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = self.to_world(local, env_ids) + origin
            st[:, 3:7] = q
            self.tray[nm].write_root_state_to_sim(st, env_ids)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: rig re-posed (xy jitter + yaw), red slots sampled per
        layer, tower rebuilt block by block, tray jittered; latches cleared. One
        rand draw is burned first (the first post-seed draw is near-degenerate)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        torch.rand(8, device=dev)  # burn the near-degenerate first draw
        torch.randint(0, 3, (8,), device=dev)  # ...and the first discrete draw

        # --- rig + tray pose ---
        self._rig_xy[env_ids] = (torch.rand(m, 2, device=dev) * 2 - 1) * c.rig_jitter
        self._rig_yaw[env_ids] = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rig_yaw_deg)
        self._tray_xy[env_ids, 0] = c.tray_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.tray_jitter
        self._tray_xy[env_ids, 1] = c.tray_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.tray_jitter
        self._write_furniture(env_ids)

        # --- sample which slot of layer 0 / layer 1 is red ---
        s0 = torch.randint(0, 3, (m,), device=dev)
        s1 = torch.randint(0, 3, (m,), device=dev)
        self._slot[env_ids, 0] = s0
        self._slot[env_ids, 1] = s1

        # --- build the slot table: block index -> (layer, slot) ---
        # layer 0 (axis x): red0 at s0, cream0/cream1 at the two other slots
        # layer 1 (axis y): red1 at s1, cream2/cream3 at the two other slots
        # layer 2 (axis x): cream4/5/6 at slots 0/1/2
        o00 = torch.where(s0 == 0, torch.ones_like(s0), torch.zeros_like(s0))
        o01 = torch.where(s0 == 2, torch.ones_like(s0), torch.full_like(s0, 2))
        o10 = torch.where(s1 == 0, torch.ones_like(s1), torch.zeros_like(s1))
        o11 = torch.where(s1 == 2, torch.ones_like(s1), torch.full_like(s1, 2))
        two = torch.full_like(s0, 2)
        layers = torch.stack([torch.zeros_like(s0), torch.ones_like(s0),
                              torch.zeros_like(s0), torch.zeros_like(s0),
                              torch.ones_like(s0), torch.ones_like(s0),
                              two, two, two], dim=1)  # (m, 9)
        slots = torch.stack([s0, s1, o00, o01, o10, o11,
                             torch.zeros_like(s0), torch.ones_like(s0), two], dim=1)

        # --- write the blocks ---
        for i, block in enumerate(self.blocks):
            layer = layers[:, i]
            slot = slots[:, i]
            along_x = (layer != 1)  # layers 0 and 2 run along x, layer 1 along y
            off = (slot.float() - 1.0) * c.pitch
            local = torch.zeros(m, 3, device=dev)
            local[:, 0] = torch.where(along_x, torch.zeros_like(off), off)
            local[:, 1] = torch.where(along_x, off, torch.zeros_like(off))
            zs = torch.tensor([c.layer_z(k) for k in range(3)], device=dev)
            local[:, 2] = zs[layer]
            self._rst_local[env_ids, i] = local
            yaw = self._rig_yaw[env_ids] + torch.where(
                along_x, torch.zeros_like(off), torch.full_like(off, math.pi / 2))
            half = yaw / 2
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = self.to_world(local, env_ids) + origin
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            block.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._slid[env_ids] = False
        self._ext[env_ids] = False
        self._tray_in[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "blocks": [b.data.root_state_w[env_ids].clone() for b in self.blocks],
            "rig_xy": self._rig_xy[env_ids].clone(),
            "rig_yaw": self._rig_yaw[env_ids].clone(),
            "tray_xy": self._tray_xy[env_ids].clone(),
            "slot": self._slot[env_ids].clone(),
            "rst_local": self._rst_local[env_ids].clone(),
            "slid": self._slid[env_ids].clone(),
            "ext": self._ext[env_ids].clone(),
            "tray_in": self._tray_in[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self._rig_xy[env_ids] = state["rig_xy"]
        self._rig_yaw[env_ids] = state["rig_yaw"]
        self._tray_xy[env_ids] = state["tray_xy"]
        self._write_furniture(env_ids)
        for b, st in zip(self.blocks, state["blocks"]):
            b.write_root_state_to_sim(st, env_ids)
        self._slot[env_ids] = state["slot"]
        self._rst_local[env_ids] = state["rst_local"]
        self._slid[env_ids] = state["slid"]
        self._ext[env_ids] = state["ext"]
        self._tray_in[env_ids] = state["tray_in"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        length, w, h = (v * 100 for v in c.block_size)
        return (
            f"On a gray PLINTH ({c.plinth_size[0] * 100:.1f} cm square, "
            f"{c.plinth_top * 100:.0f} cm tall) stands a small Jenga-style TOWER: three "
            f"layers of three wooden blocks each ({length:.0f} x {w:.0f} x {h:.1f} cm), "
            f"laid side by side, each layer perpendicular to the one below. Seven blocks "
            f"are CREAM; TWO are RED — one somewhere in the BOTTOM layer, one somewhere "
            f"in the MIDDLE layer (which slot is red varies per episode). Every block's "
            f"ends stick out ~{(c.block_size[0] / 2 - c.span_half) * 1000:.0f} mm past "
            f"the stack on both sides, so any block can be gripped or pushed by its "
            f"protruding end. A blue TRAY ({c.tray_inner * 100:.0f} cm square inside, "
            f"{c.tray_wall_h * 100:.0f} cm walls) sits on the table beside the plinth.\n"
            f"Goal: get BOTH red blocks to rest inside the tray while ALL SEVEN cream "
            f"blocks remain exactly where they started (each within "
            f"{c.stand_xy_tol * 100:.0f} cm / {c.stand_tilt_deg:.0f} deg of its original "
            f"slot). Each red block is roofed by the layer above it: it cannot be lifted "
            f"out — pulling it upward just heaves the blocks above and wrecks the tower. "
            f"It must be SLID OUT LENGTHWISE along its own axis (either direction), "
            f"Jenga-style, with the layers above riding on it, then carried to the tray. "
            f"A red block left anywhere but inside the tray does not count; if any cream "
            f"block ends up shifted, dropped a layer, tilted, or knocked off, the task "
            f"fails — toppling the tower and picking the reds from the rubble scores "
            f"nothing. The two reds may be extracted in either order."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the two red blocks lengthwise out of the block tower, Jenga-style, "
            "and place them in the blue tray. Every cream block must stay in its "
            "original position — if the tower shifts or topples, the task fails."
        )

    # ----- readings ------------------------------------------------------------------------------
    def block_local(self, i: int) -> torch.Tensor:
        """(N,3) block i centre in the rig-local frame."""
        return self.to_local(self.blocks[i].data.root_pos_w)

    def _up_z(self, i: int) -> torch.Tensor:
        """(N,) world-z component of block i's body +z axis (1 = flat)."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(self.blocks[i].data.root_quat_w, ez)[:, 2]

    def tower_standing(self) -> torch.Tensor:
        """(N,) bool: every CREAM block still in its reset slot — xy within
        `stand_xy_tol`, height within `stand_z_tol`, tilt within `stand_tilt_deg`."""
        c = self.cfg
        cos_tilt = math.cos(math.radians(c.stand_tilt_deg))
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for i in range(2, 9):
            d = self.block_local(i) - self._rst_local[:, i]
            ok &= (d[:, 0:2].norm(dim=-1) < c.stand_xy_tol) \
                & (d[:, 2].abs() < c.stand_z_tol) & (self._up_z(i) > cos_tilt)
        return ok

    def in_tray(self, k: int) -> torch.Tensor:
        """(N,) bool: red k centre inside the tray volume, at floor rest height."""
        c = self.cfg
        p = self.block_local(k)
        dx = (p[:, 0] - self._tray_xy[:, 0]).abs()
        dy = (p[:, 1] - self._tray_xy[:, 1]).abs()
        return ((dx < c.tray_xy_tol) & (dy < c.tray_xy_tol)
                & (p[:, 2] > c.tray_z[0]) & (p[:, 2] < c.tray_z[1]))

    def settled(self) -> torch.Tensor:
        """(N,) bool: every block below the settle gates."""
        c = self.cfg
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for b in self.blocks:
            ok &= (b.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
                & (b.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega)
        return ok

    # ----- rubric --------------------------------------------------------------------------------
    def _update_latches(self) -> None:
        c = self.cfg
        standing = self.tower_standing()
        for k in range(2):
            p = self.block_local(k)
            d = p - self._rst_local[:, k]
            along = (d[:, 0:2] * self._axis[k]).sum(dim=-1).abs()
            cross = (d[:, 0:2] * self._axis[1 - k]).sum(dim=-1).abs()
            in_channel = (cross < c.slid_cross_max) & (d[:, 2].abs() < c.slid_dz_max)
            self._slid[:, k] |= standing & in_channel & (along > c.slid_min)
            hdist = p[:, 0:2].norm(dim=-1)
            self._ext[:, k] |= standing & (hdist > c.ext_dist) & (p[:, 2] < 0.30)
            nearly_still = self.blocks[k].data.root_lin_vel_w.norm(dim=-1) < 0.25
            self._tray_in[:, k] |= standing & self.in_tray(k) & nearly_still

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No mechanism plant (furniture kinematic, blocks free) — just latch
        rubric progress every step so transient achievements keep credit."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: both reds resting inside the tray, every cream block still in
        its slot, everything settled — live physical outcome only (no latches)."""
        self._update_latches()
        return (self.in_tray(0) & self.in_tray(1) & self.tower_standing()
                & self.settled())

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: latched slide/extract/bank credit (each latch gated
        on the tower standing at the moment it was earned), cap 0.70 — and 1.0 iff
        success() holds live. Null policy ~0."""
        c = self.cfg
        self._update_latches()
        base = (c.w_slid * self._slid.float().sum(dim=1)
                + c.w_ext * self._ext.float().sum(dim=1)
                + c.w_tray * self._tray_in.float().sum(dim=1)).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; the reds are driven through applied wrenches.
register_env("simgen", lambda: EnvCfg(scene="jenga_quarry", robot="null"))

"""FireCribScene — build a log-cabin kindling crib on the hearth plate, then rest the
griddle level on top (sim_gen task `close_grill_i8`).

Derived from rlbench/close_grill ("close the grill": push the open hinged lid of a
barbecue grill about its hinge until it shuts), but the MANIPULATION MODEL is replaced
wholesale. The seed's plan is ONE unordered pushing contact on a panel that is already
attached to the fixture — no transport, no second object, no structure, judged by a
door joint. Here NOTHING is hinged and nothing is pushed shut; there is no lid and no
joint anywhere. The task is STRUCTURE BUILDING: the solver must erect a two-layer
log-cabin FIRE CRIB from four loose wooden splits on a marked hearth plate — two
splits laid parallel with an open air gap between them, two more laid across them at
right angles — and then prove the structure by resting a steel GRIDDLE PLATE flat and
level on top. The placed objects themselves become the support for every later
placement: no fixture guides anything, stability is emergent, and the final griddle
height (~49 mm above the hearth) is reachable ONLY through the completed crib. A
too-short OFFCUT decoy physically cannot serve as a crib member (shorter than the
smallest legal air gap — dropped across the gap it falls straight through) and must be
left OFF the hearth plate.

What the solver must bring, none of which exists in the seed:
  (1) perception of LENGTH identity (four 170 mm splits vs a 48 mm offcut, spawn
      slots permuted per episode) and of the hearth plate's randomized position;
  (2) an ordered BUILD (bottom layer, then top layer, then the griddle — each layer
      rests on the one below, so the order is physically inherent);
  (3) structural judgment: parallel spacing inside the 78..128 mm window (the air
      gap / "chimney" a fire lay needs), orthogonal crossings that actually land on
      BOTH lower splits, and a final load the structure must carry;
  (4) restraint: the offcut is scrap and must stay off the hearth plate.

Assets are fully procedural (the compound-spawner pattern — child colliders of one
body never self-collide):
  - hearth plate: KINEMATIC dark slab 260 x 260 x 12 mm (xy + yaw randomized).
  - splits: four DYNAMIC wooden bars 170 x 22 x 22 mm (60 g).
  - offcut: one DYNAMIC bar 48 x 22 x 22 mm — same stock, obviously shorter.
  - griddle: DYNAMIC compound — steel plate 150 x 150 x 10 mm with a square grasp
    knob (24 x 24 x 30 mm) at its centre (220 g).

Per-episode randomization (readback-verifiable): hearth xy jitter + full yaw, the
FIVE bars permuted over five ground scatter slots (overlap-rejected) with xy jitter
and free yaw, griddle xy jitter + free yaw.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve trajectory):
  0.10 * approach — a split ever carried within `approach_r` of the hearth centre
  0.25 * base     — a legal bottom pair ever formed (two splits flat ON the hearth,
                    parallel, spacing in the window) (latched)
  0.20 * span1    — a first spanner ever rested across BOTH bottom splits (latched)
  0.20 * crib     — the full two-layer crib ever complete (latched)
  1.0 iff success() — crib complete, griddle resting flat and level on the top pair
                    (centre 41..60 mm above the hearth surface), offcut off the
                    hearth plate, everything settled and finite. Non-success capped
                    at 0.75.

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


# ----- custom compound spawner (griddle plate + grasp knob) -------------------------------------
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


def _spawn_griddle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the griddle at `prim_path`: DYNAMIC compound. Local frame: origin at the
    PLATE CENTRE (plate spans z -5..+5 mm); the square grasp knob rises above."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_box(stage, f"{prim_path}/plate", center=(0.0, 0.0, 0.0),
             size=(c.plate_size, c.plate_size, c.plate_t), color=c.plate_color,
             collide=collide)
    _add_box(stage, f"{prim_path}/knob", center=(0.0, 0.0, c.plate_t / 2 + c.knob_h / 2),
             size=(c.knob_xy, c.knob_xy, c.knob_h), color=c.knob_color,
             collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclass (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "griddle" not in _SPAWNER_CACHE:

        @configclass
        class GriddleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_griddle)
            plate_size: float = 0.150
            plate_t: float = 0.010
            knob_xy: float = 0.024
            knob_h: float = 0.030
            plate_color: tuple = (0.34, 0.36, 0.40)
            knob_color: tuple = (0.06, 0.06, 0.06)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(griddle=GriddleSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class FireCribSceneCfg(BaseCfg):
    """Config for `FireCribScene`. The structural claims are honest by construction
    (asserted in __post_init__): the offcut is shorter than the smallest legal air
    gap (it cannot span), the griddle and the top splits both reach across the
    largest legal gap with real overlap, and the griddle's success height band is
    reachable only above a two-layer structure."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    bot_z_lo: float = tunable(0.004)     # bottom split centre above hearth top: band lo (nom 11 mm)
    bot_z_hi: float = tunable(0.020)     # band hi
    top_z_lo: float = tunable(0.026)     # top split centre above hearth top: band lo (nom 33 mm)
    top_z_hi: float = tunable(0.042)     # band hi
    plate_z_lo: float = tunable(0.041)   # griddle centre above hearth top: band lo (nom 49 mm)
    plate_z_hi: float = tunable(0.060)   # band hi
    bar_tilt_max_deg: float = tunable(10.0)   # split long axis within this of horizontal
    plate_tilt_max_deg: float = tunable(12.0)  # griddle face within this of level
    par_max_deg: float = tunable(20.0)   # in-layer pair: axes parallel within this
    ortho_tol_deg: float = tunable(20.0)  # cross-layer: axes orthogonal within this
    gap_min: float = tunable(0.078)      # pair spacing window (centreline distance, m)
    gap_max: float = tunable(0.128)
    cross_margin: float = tunable(0.008)  # crossing point inside both bars by this from the ends
    plate_xy_tol: float = tunable(0.055)  # griddle centre within this of the crib centroid (xy)
    pad_edge_margin: float = tunable(0.010)  # bottom splits inside the hearth by this margin
    settle_speed: float = tunable(0.05)  # max |lin vel| of every dynamic body when judging (m/s)
    latch_speed: float = tunable(0.15)   # max split |lin vel| for a structure latch to arm
    approach_r: float = tunable(0.20)    # latched approach credit: split within this of hearth ctr

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    pad_jitter: float = tunable(0.04)    # hearth xy jitter (+/- m)
    pad_yaw_deg: float = tunable(180.0)  # hearth yaw (+/- deg, visual + footprint frame)
    slot_shuffle: bool = tunable(True)   # permute the five bars over the five scatter slots
    slot_jitter: float = tunable(0.030)  # per-bar xy jitter at its slot (+/- m)
    bar_yaw_deg: float = tunable(180.0)  # per-bar free yaw (+/- deg)
    griddle_jitter: float = tunable(0.030)  # griddle xy jitter (+/- m)

    # --- info: layout (world nominal) ------------------------------------------------------------
    pad_pos: tuple = info((0.42, 0.16))     # hearth plate centre (nominal)
    griddle_pos: tuple = info((0.50, -0.14))  # griddle spawn (nominal)
    bar_slots: tuple = info(((-0.20, -0.26), (-0.08, -0.34), (0.05, -0.27),
                             (0.18, -0.35), (0.30, -0.27)))  # five scatter slots
    # --- info: hearth plate ----------------------------------------------------------------------
    pad_size: float = info(0.260)
    pad_t: float = info(0.012)
    pad_color: tuple = info((0.15, 0.15, 0.16))
    # --- info: bars ------------------------------------------------------------------------------
    bar_len: float = info(0.170)
    bar_w: float = info(0.022)
    bar_mass: float = info(0.060)
    offcut_len: float = info(0.048)
    offcut_mass: float = info(0.020)
    split_color: tuple = info((0.55, 0.38, 0.20))
    offcut_color: tuple = info((0.62, 0.45, 0.26))
    # --- info: griddle ---------------------------------------------------------------------------
    plate_size: float = info(0.150)
    plate_t: float = info(0.010)
    knob_xy: float = info(0.024)
    knob_h: float = info(0.030)
    plate_mass: float = info(0.220)
    contact_offset: float = info(0.002)
    # rubric weights (0.10 + 0.25 + 0.20 + 0.20 = 0.75 = the non-success cap)
    w_appr: float = info(0.10)
    w_base: float = info(0.25)
    w_span1: float = info(0.20)
    w_crib: float = info(0.20)

    def __post_init__(self) -> None:
        # Honesty-by-construction asserts (the geometric claims the task rests on).
        assert self.offcut_len <= self.gap_min - self.bar_w - 0.004, \
            "offcut must be shorter than the smallest legal air gap (cannot span)"
        assert self.bar_len / 2 >= self.gap_max / 2 + self.cross_margin + 0.004, \
            "a top split must reach across the widest legal bottom pair"
        assert self.plate_size / 2 >= self.gap_max / 2 + 0.008, \
            "the griddle must reach both top splits at the widest legal spacing"
        assert self.plate_z_lo > self.bar_w + self.plate_t / 2 + 0.004, \
            "the griddle success band must be unreachable from a single layer"
        assert self.bot_z_hi < self.top_z_lo, "layer bands must be disjoint"


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


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("fire_crib")
class FireCribScene(BaseScene):
    cfg: FireCribSceneCfg

    SPLIT_NAMES = ("split_0", "split_1", "split_2", "split_3")
    BAR_NAMES = SPLIT_NAMES + ("offcut",)

    def __init__(self, cfg: FireCribSceneCfg | None = None) -> None:
        super().__init__(cfg or FireCribSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()

        dyn_props = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.2, angular_damping=0.3,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7, dynamic_friction=0.6, restitution=0.0),
        )

        griddle_spawn = cls["griddle"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.plate_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.2, angular_damping=0.3,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1),
            plate_size=c.plate_size, plate_t=c.plate_t,
            knob_xy=c.knob_xy, knob_h=c.knob_h,
            contact_offset=c.contact_offset)

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
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Hearth",
                spawn=sim_utils.CuboidCfg(
                    size=(c.pad_size, c.pad_size, c.pad_t),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8, dynamic_friction=0.7, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pad_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pad_pos[0], c.pad_pos[1], c.pad_t / 2)),
            ),
            "griddle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Griddle",
                spawn=griddle_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.griddle_pos[0], c.griddle_pos[1], c.plate_t / 2 + 0.002)),
            ),
        }
        for i, name in enumerate(self.BAR_NAMES):
            is_split = name != "offcut"
            length = c.bar_len if is_split else c.offcut_len
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bar_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=(length, c.bar_w, c.bar_w),
                    mass_props=sim_utils.MassPropertiesCfg(
                        mass=c.bar_mass if is_split else c.offcut_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.split_color if is_split else c.offcut_color),
                    **dyn_props,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0 + 0.1 * i, 1.0, 0.03)),
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
        self.pad: RigidObject = env.iscene["pad"]
        self.griddle: RigidObject = env.iscene["griddle"]
        self.bars: dict[str, RigidObject] = {n: env.iscene[n] for n in self.BAR_NAMES}
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self.slot_of = torch.zeros(n, 5, dtype=torch.long, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._appr = torch.zeros(n, dtype=torch.bool, device=dev)
        self._base = torch.zeros(n, dtype=torch.bool, device=dev)
        self._span1 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._crib = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the hearth plate (xy jitter + full yaw), permute the
        five bars over the five scatter slots (flat on the ground, xy jitter + free
        yaw, overlap-rejected), place the griddle (xy jitter + free yaw), clear the
        latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- hearth plate: kinematic, xy jitter + yaw ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.pad_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.pad_jitter
        st[:, 1] = c.pad_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.pad_jitter
        st[:, 2] = c.pad_t / 2
        st[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1) * math.radians(c.pad_yaw_deg))
        st[:, 0:3] += origin
        self.pad.write_root_state_to_sim(st, env_ids)

        # --- bars: slot permutation, overlap-rejected scatter, flat with free yaw ---
        if c.slot_shuffle:
            perm = torch.rand(m, 5, device=dev).argsort(dim=1)
        else:
            perm = torch.arange(5, device=dev).expand(m, 5).clone()
        self.slot_of[env_ids] = perm
        slots = torch.tensor(c.bar_slots, device=dev)  # (5, 2)
        yaw_amp = math.radians(c.bar_yaw_deg)
        half_len = [c.bar_len / 2] * 4 + [c.offcut_len / 2]

        def seg_dist(p_c, p_y, p_h, q_c, q_y, q_h):
            """Min distance between 2D segments (centre, yaw, half-length), batched (m,)."""
            su = torch.stack([torch.cos(p_y), torch.sin(p_y)], dim=-1)
            tv = torch.stack([torch.cos(q_y), torch.sin(q_y)], dim=-1)
            best = torch.full_like(p_y, torch.inf)
            for fa in (-1.0, -0.5, 0.0, 0.5, 1.0):
                pa = p_c + su * (fa * p_h)
                w = pa - q_c
                t = (w * tv).sum(-1).clamp(-q_h, q_h)
                best = torch.minimum(best, (w - tv * t.unsqueeze(-1)).norm(dim=-1))
            for fb in (-1.0, -0.5, 0.0, 0.5, 1.0):
                qb = q_c + tv * (fb * q_h)
                w = qb - p_c
                t = (w * su).sum(-1).clamp(-p_h, p_h)
                best = torch.minimum(best, (w - su * t.unsqueeze(-1)).norm(dim=-1))
            return best

        placed: list[tuple[torch.Tensor, torch.Tensor, float]] = []
        for i, name in enumerate(self.BAR_NAMES):
            slot_xy = slots[perm[:, i]]  # (m, 2)
            ctr = slot_xy + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            yaw = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp
            for _try in range(12):
                bad = torch.zeros(m, dtype=torch.bool, device=dev)
                for (qc, qy, qh) in placed:
                    d = seg_dist(ctr, yaw, half_len[i], qc, qy, qh)
                    bad |= d < (c.bar_w + 0.008)
                if not bad.any():
                    break
                nb = int(bad.sum())
                ctr[bad] = slot_xy[bad] + (torch.rand(nb, 2, device=dev) * 2 - 1) * c.slot_jitter
                yaw[bad] = (torch.rand(nb, device=dev) * 2 - 1) * yaw_amp
            placed.append((ctr, yaw, half_len[i]))
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = ctr
            st[:, 2] = c.bar_w / 2 + 0.002
            st[:, 3:7] = _qz(yaw)
            st[:, 0:3] += origin
            self.bars[name].write_root_state_to_sim(st, env_ids)

        # --- griddle: own spawn zone, xy jitter + free yaw, flat on the ground ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.griddle_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.griddle_jitter
        st[:, 1] = c.griddle_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.griddle_jitter
        st[:, 2] = c.plate_t / 2 + 0.002
        st[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi)
        st[:, 0:3] += origin
        self.griddle.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._appr[env_ids] = False
        self._base[env_ids] = False
        self._span1[env_ids] = False
        self._crib[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "griddle": self.griddle.data.root_state_w[env_ids].clone(),
            "bars": {n: b.data.root_state_w[env_ids].clone() for n, b in self.bars.items()},
            "slot_of": self.slot_of[env_ids].clone(),
            "appr": self._appr[env_ids].clone(),
            "base": self._base[env_ids].clone(),
            "span1": self._span1[env_ids].clone(),
            "crib": self._crib[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.pad.write_root_state_to_sim(state["pad"], env_ids)
        self.griddle.write_root_state_to_sim(state["griddle"], env_ids)
        for n, b in self.bars.items():
            b.write_root_state_to_sim(state["bars"][n], env_ids)
        self.slot_of[env_ids] = state["slot_of"]
        self._appr[env_ids] = state["appr"]
        self._base[env_ids] = state["base"]
        self._span1[env_ids] = state["span1"]
        self._crib[env_ids] = state["crib"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A square charcoal-gray HEARTH PLATE ({c.pad_size * 100:.0f} cm across, "
            f"{c.pad_t * 1000:.0f} mm thick) lies flat on the ground; its position and "
            f"heading vary per episode. Scattered on the opposite side of the workspace "
            f"lie FIVE wooden bars of identical {c.bar_w * 1000:.0f} mm square "
            f"cross-section: FOUR long KINDLING SPLITS ({c.bar_len * 100:.0f} cm long) "
            f"and ONE short OFFCUT ({c.offcut_len * 100:.1f} cm long) — their positions "
            f"and headings are shuffled every episode, so tell them apart by LENGTH. "
            f"Nearby lies a flat steel GRIDDLE PLATE ({c.plate_size * 100:.0f} cm "
            f"square) with a small square grasp knob at its centre.\n"
            f"Goal: build a log-cabin FIRE CRIB on the hearth plate and rest the "
            f"griddle level on top. (1) Lay TWO long splits flat ON the hearth plate, "
            f"parallel to each other, with {c.gap_min * 100:.1f} to "
            f"{c.gap_max * 100:.1f} cm between their centrelines so an open air gap "
            f"remains between them. (2) Lay the OTHER TWO long splits on top of that "
            f"pair, at right angles to it, each resting across BOTH lower splits, again "
            f"{c.gap_min * 100:.1f}-{c.gap_max * 100:.1f} cm apart from each other. "
            f"(3) Rest the griddle plate flat and LEVEL on the upper pair, roughly "
            f"centred over the crib (its centre ends up about "
            f"{(c.plate_z_lo + c.plate_z_hi) * 500:.0f} mm above the hearth surface). "
            f"Build bottom layer first, then top layer, then the griddle — each stage "
            f"rests on the one below. The short OFFCUT is scrap: it is too short to "
            f"span the air gap, and it must be left OFF the hearth plate, on the "
            f"ground. Success: the completed two-layer crossed crib standing on the "
            f"hearth plate with the griddle resting level on top, the offcut off the "
            f"plate, and everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Build a log-cabin fire crib on the gray hearth plate: lay two long wooden "
            "splits parallel with an open air gap between them, lay the other two long "
            "splits across them at right angles, then rest the steel griddle plate "
            "flat and level on top. Leave the short offcut off the hearth plate."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _pad_top_z(self) -> torch.Tensor:
        """(N,) world z of the hearth plate top surface."""
        return self.pad.data.root_pos_w[:, 2] + self.cfg.pad_t / 2

    def _pad_local_xy(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points (N,3) or (N,P,3) -> hearth-frame xy (same leading shape, last=2)."""
        from isaaclab.utils.math import quat_apply_inverse

        pp = self.pad.data.root_pos_w
        pq = self.pad.data.root_quat_w
        if pos_w.dim() == 3:
            n, p = pos_w.shape[0], pos_w.shape[1]
            rel = (pos_w - pp[:, None, :]).reshape(n * p, 3)
            q = pq[:, None, :].expand(n, p, 4).reshape(n * p, 4)
            return quat_apply_inverse(q, rel).reshape(n, p, 3)[..., :2]
        return quat_apply_inverse(pq, pos_w - pp)[..., :2]

    def _split_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(pos_w (N,4,3), axis_w (N,4,3), |lin_vel| (N,4)) for the four long splits."""
        from isaaclab.utils.math import quat_apply

        pos = torch.stack([self.bars[n].data.root_pos_w for n in self.SPLIT_NAMES], dim=1)
        quat = torch.stack([self.bars[n].data.root_quat_w for n in self.SPLIT_NAMES], dim=1)
        vel = torch.stack([self.bars[n].data.root_lin_vel_w.norm(dim=-1)
                           for n in self.SPLIT_NAMES], dim=1)
        n = pos.shape[0]
        ex = torch.tensor([1.0, 0.0, 0.0], device=pos.device).expand(n * 4, 3)
        axis = quat_apply(quat.reshape(n * 4, 4), ex).reshape(n, 4, 3)
        return pos, axis, vel

    @staticmethod
    def _dir_xy(axis: torch.Tensor) -> torch.Tensor:
        """Normalized horizontal direction of bar axes; axis (..., 3) -> (..., 2)."""
        d = axis[..., :2]
        return d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    def _layer_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(in_bot (N,4), in_top (N,4)): split is horizontal and its centre height above
        the hearth top lies in the bottom/top layer band; bottom additionally requires
        the centre inside the hearth footprint."""
        c = self.cfg
        pos, axis, _v = self._split_tensors()
        dz = pos[:, :, 2] - self._pad_top_z()[:, None]
        horiz = axis[:, :, 2].abs() < math.sin(math.radians(c.bar_tilt_max_deg))
        loc = self._pad_local_xy(pos)
        half = c.pad_size / 2 - c.pad_edge_margin
        on_pad = (loc[:, :, 0].abs() < half) & (loc[:, :, 1].abs() < half)
        in_bot = horiz & on_pad & (dz > c.bot_z_lo) & (dz < c.bot_z_hi)
        in_top = horiz & (dz > c.top_z_lo) & (dz < c.top_z_hi)
        return in_bot, in_top

    def _pair_ok(self, pos, dxy, i: int, j: int) -> torch.Tensor:
        """(N,) bool: splits i, j are parallel within `par_max_deg` and their
        centreline distance lies in the [gap_min, gap_max] window."""
        c = self.cfg
        ui, uj = dxy[:, i], dxy[:, j]
        cross = (ui[:, 0] * uj[:, 1] - ui[:, 1] * uj[:, 0]).abs()
        par = cross < math.sin(math.radians(c.par_max_deg))
        d = pos[:, j, :2] - pos[:, i, :2]
        ni = torch.stack([-ui[:, 1], ui[:, 0]], dim=-1)
        gap = (d * ni).sum(dim=-1).abs()
        return par & (gap > c.gap_min) & (gap < c.gap_max)

    def _cross_ok(self, pos, dxy, t: int, b: int) -> torch.Tensor:
        """(N,) bool: top split t is orthogonal to bottom split b (within tol) and
        their 2D axis lines intersect inside BOTH bars (margin from the ends)."""
        c = self.cfg
        u, v = dxy[:, t], dxy[:, b]
        cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        ortho = cross.abs() > math.cos(math.radians(c.ortho_tol_deg))
        d = pos[:, b, :2] - pos[:, t, :2]
        cr = torch.where(cross.abs() < 1e-4, torch.full_like(cross, 1e-4), cross)
        s = (d[:, 0] * v[:, 1] - d[:, 1] * v[:, 0]) / cr    # param along top bar t
        w = (d[:, 0] * u[:, 1] - d[:, 1] * u[:, 0]) / cr    # param along bottom bar b
        lim = c.bar_len / 2 - c.cross_margin
        return ortho & (s.abs() < lim) & (w.abs() < lim)

    def _structure_now(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(base_now, span1_now, crib_now), each (N,) bool, purely geometric.
        base: some legal bottom pair. span1: base + at least one top split crossing
        both members. crib: a full partition of the four splits into a legal bottom
        pair and a legal top pair, every top member crossing both bottom members."""
        pos, axis, _v = self._split_tensors()
        dxy = self._dir_xy(axis)
        in_bot, in_top = self._layer_masks()
        n = pos.shape[0]
        dev = pos.device
        base = torch.zeros(n, dtype=torch.bool, device=dev)
        span1 = torch.zeros(n, dtype=torch.bool, device=dev)
        crib = torch.zeros(n, dtype=torch.bool, device=dev)
        import itertools

        for i, j in itertools.combinations(range(4), 2):
            pij = in_bot[:, i] & in_bot[:, j] & self._pair_ok(pos, dxy, i, j)
            base |= pij
            rest = [k for k in range(4) if k not in (i, j)]
            spans = {}
            for k in rest:
                spans[k] = (in_top[:, k] & self._cross_ok(pos, dxy, k, i)
                            & self._cross_ok(pos, dxy, k, j))
                span1 |= pij & spans[k]
            k, l = rest
            crib |= (pij & spans[k] & spans[l] & self._pair_ok(pos, dxy, k, l))
        return base, span1, crib

    def _crib_centroid_xy(self) -> torch.Tensor:
        """(N,2) mean xy of the four splits (the crib centre when the crib stands)."""
        pos, _a, _v = self._split_tensors()
        return pos[:, :, :2].mean(dim=1)

    def griddle_on_crib(self) -> torch.Tensor:
        """(N,) bool, geometric: griddle centre in the success height band above the
        hearth top, face level, xy within `plate_xy_tol` of the crib centroid."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        gp = self.griddle.data.root_pos_w
        gq = self.griddle.data.root_quat_w
        n = gp.shape[0]
        dz = gp[:, 2] - self._pad_top_z()
        ez = torch.tensor([0.0, 0.0, 1.0], device=gp.device).expand(n, 3)
        up = quat_apply(gq, ez)
        level = up[:, 2] > math.cos(math.radians(c.plate_tilt_max_deg))
        near = (gp[:, :2] - self._crib_centroid_xy()).norm(dim=-1) < c.plate_xy_tol
        return (dz > c.plate_z_lo) & (dz < c.plate_z_hi) & level & near

    def offcut_clear(self) -> torch.Tensor:
        """(N,) bool: the offcut is OFF the hearth plate (centre outside the footprint)
        and down at ground level (not perched on the structure)."""
        c = self.cfg
        p = self.bars["offcut"].data.root_pos_w
        loc = self._pad_local_xy(p)
        half = c.pad_size / 2
        outside = (loc[:, 0].abs() > half + 0.005) | (loc[:, 1].abs() > half + 0.005)
        low = (p[:, 2] - self._pad_top_z()) < 0.020
        return outside & low

    def settled(self) -> torch.Tensor:
        """(N,) bool: every dynamic body |lin vel| below `settle_speed`."""
        c = self.cfg
        _p, _a, vel = self._split_tensors()
        vo = self.bars["offcut"].data.root_lin_vel_w.norm(dim=-1)
        vg = self.griddle.data.root_lin_vel_w.norm(dim=-1)
        return (vel < c.settle_speed).all(dim=1) & (vo < c.settle_speed) \
            & (vg < c.settle_speed)

    def _update_latches(self) -> None:
        c = self.cfg
        pos, _a, vel = self._split_tensors()
        d = (pos[:, :, :2] - self.pad.data.root_pos_w[:, None, :2]).norm(dim=-1)
        self._appr |= (d < c.approach_r).any(dim=1)
        calm = (vel < c.latch_speed).all(dim=1)
        base, span1, crib = self._structure_now()
        self._base |= base & calm
        self._span1 |= span1 & calm
        self._crib |= crib & calm

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the full two-layer crib stands on the hearth plate, the griddle
        rests level on top in the success height band, the offcut is off the plate,
        everything is settled and finite. All clauses are live physical outcomes."""
        self._update_latches()
        _b, _s, crib = self._structure_now()
        pos, _a, _v = self._split_tensors()
        finite = torch.isfinite(pos).all(dim=-1).all(dim=-1) \
            & torch.isfinite(self.griddle.data.root_pos_w).all(dim=-1)
        return crib & self.griddle_on_crib() & self.offcut_clear() \
            & self.settled() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10*approach + 0.25*base + 0.20*span1 + 0.20*crib
        (all latched; ~0 for doing nothing), capped at 0.75 — and exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_appr * self._appr.float() + c.w_base * self._base.float()
                + c.w_span1 * self._span1.float()
                + c.w_crib * self._crib.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="fire_crib", robot="null"))

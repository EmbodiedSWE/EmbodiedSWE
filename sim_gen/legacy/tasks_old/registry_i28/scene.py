"""MusicalCansScene — a buffer-constrained rearrangement puzzle (sim_gen task
`registry_i28`, derived from the simpler_env registry / task suite).

The seed is the whole SimplerEnv registry: 25 tasks whose plans are all one free
primitive — pick an object, move object A near object B, put A on B, open/close a
drawer, open-then-place. In every one of them the target location is FREE: the solver
picks the object up and puts it where the instruction says, and the world never
objects. This task keeps the seed's object vocabulary (upright drink cans on a kitchen
counter, "move/put the can at X") but makes every direct placement IMPOSSIBLE, turning
the episode into a small state-space puzzle:

  - Three colored cans (red / green / blue) stand seated in the cups of three
    color-matched pedestals — but on the WRONG pedestals: the starting arrangement is a
    sampled DERANGEMENT (no can on its own pad; either both 3-cycles when all three
    cans are present, or a 2-can swap when one can is sampled absent).
  - Each pedestal cup physically fits exactly ONE can (aperture 76 mm, can 60 mm) — a
    second can set on an occupied pad ends up stacked or toppled, never seated.
  - The counter is OFF LIMITS: a present can that ever comes to counter level latches
    an irreversible `dropped` (the narrative: the counter is soaking wet, a can set
    down there is ruined). So the seed's implicit escape hatch — "just set it aside on
    the table for a moment" — is a tested failing strategy.
  - One spare GRAY pedestal (the buffer) is the only legal parking spot.

Goal: every present can seated upright and settled in its color-matched cup, nothing
ever dropped. Because the start is a derangement, no can can go home directly — its
home is occupied. The solver must decompose the permutation through the buffer:
`n_present + 1` moves (3 misplaced -> 4 moves, 2 misplaced -> 3 moves), and the order
of moves is FORCED (a move is only possible into an empty cup). A memorized fixed
sequence fails: which cycle / which pair is misplaced changes every episode.

Judged on PHYSICAL outcomes only:
  - `correct[i]`: can i's centre within `pad_xy_tol` of ITS pad's cup axis, standing at
    seat height (`seat_z_tol`), upright within `upright_max_deg` (15 deg — chosen ABOVE
    the ~10 deg maximum physical lean of a can inside the cup, so anything genuinely in
    the cup counts; a can lying across the collar mouth is at 90 deg and never counts),
    and settled (lin + ang velocity gates).
  - success(): all present cans correct AND `dropped` never latched.
  - score(): 0 for doing nothing (the deranged start has zero correct cans by
    construction) — else 0.05 for the latched first lift + 0.10 for the latched use of
    the buffer cup + 0.55 * (fraction of present cans correct); hard-capped at 0.25
    forever once `dropped` latched; exactly 1.0 iff success. Max non-success = 0.70
    (capped states lower).

Per-episode randomization: which derangement (2 cycles x present-subset x absent
identity), pedestal xy jitter, can seating jitter — the move sequence a solver must
produce differs episode to episode in both length and order.

Assets are fully procedural: a kinematic counter slab, four kinematic compound
pedestals (column + octagonal collar cup, authored by a custom `clone()` spawner — the
pen_holder pattern, kinematic variant), three dynamic cylinder cans. No external
files. Heavy imports (isaaclab, pxr) are deferred so importing this module — and
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


# ----- custom compound spawner -----------------------------------------------------------------
# One KINEMATIC rigid body per pedestal: a support column whose flat top is the cup floor,
# ringed by 8 box wall segments forming a shallow octagonal collar (regular-octagon aperture of
# inradius `inner_r`). Authored with raw pxr APIs; `isaaclab.sim.utils.clone` supplies the
# regex-resolve + per-env replication every CuboidCfg spawn uses (idempotent decoration — the
# duplicate-xformOp trap does not arise because each cloned prim is authored fresh).

_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_pedestal(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one pedestal at `prim_path`. Body origin = cup-floor centre (column top face);
    the column hangs below (embedded a few mm into the counter slab — kinematic-in-kinematic,
    no dynamics), the collar walls rise above. Kinematic: re-posed per reset by pose writes."""
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
    rb.CreateKinematicEnabledAttr(True)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    # column: cylinder, top face at local z=0 (the cup floor the can stands on)
    col = UsdGeom.Cylinder.Define(stage, f"{prim_path}/column")
    col.CreateRadiusAttr(cfg.column_r)
    col.CreateHeightAttr(cfg.column_h)
    col.CreateExtentAttr([Gf.Vec3f(-cfg.column_r, -cfg.column_r, -cfg.column_h / 2),
                          Gf.Vec3f(cfg.column_r, cfg.column_r, cfg.column_h / 2)])
    UsdGeom.Xformable(col.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, -cfg.column_h / 2))
    col.CreateDisplayColorAttr([Gf.Vec3f(*cfg.column_color)])
    collide(col.GetPrim())

    # 8 collar wall boxes: shallow octagonal cup of inner inradius `inner_r`, colored with
    # the pad's identity color (the pen_holder wall layout, short walls)
    n = 8
    r_mid = cfg.inner_r + cfg.wall_t / 2
    seg_len = 2 * (cfg.inner_r + cfg.wall_t) * math.tan(math.pi / n) + 0.002
    color = Gf.Vec3f(*cfg.color)
    for k in range(n):
        ang = 2 * math.pi * k / n
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/wall_{k}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(r_mid * math.cos(ang), r_mid * math.sin(ang),
                                          cfg.wall_h / 2))
        sxf.AddRotateZOp().Set(math.degrees(ang))
        sxf.AddScaleOp().Set(Gf.Vec3f(cfg.wall_t, seg_len, cfg.wall_h))
        seg.CreateDisplayColorAttr([color])
        collide(seg.GetPrim())
    return root


def _pedestal_spawner_cfg(*, column_r: float, column_h: float, inner_r: float, wall_t: float,
                          wall_h: float, color: tuple, column_color: tuple,
                          contact_offset: float) -> Any:
    """Build (lazily, app required) the pedestal spawner cfg — `clone` wraps
    `_spawn_pedestal` exactly like `spawn_cuboid` is wrapped."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "pedestal" not in _SPAWNER_CACHE:

        @configclass
        class PedestalSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pedestal)
            column_r: float = 0.045
            column_h: float = 0.125
            inner_r: float = 0.038
            wall_t: float = 0.008
            wall_h: float = 0.022
            color: tuple = (0.5, 0.5, 0.5)
            column_color: tuple = (0.35, 0.33, 0.30)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["pedestal"] = PedestalSpawnerCfg

    return _SPAWNER_CACHE["pedestal"](
        column_r=column_r, column_h=column_h, inner_r=inner_r, wall_t=wall_t, wall_h=wall_h,
        color=color, column_color=column_color, contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class MusicalCansSceneCfg(BaseCfg):
    """Config for `MusicalCansScene`. Env-local frame: counter centred at the origin, its
    top at z = `counter_top`; pedestals in a row along y; pad index 0/1/2 = the red/green/
    blue home pads (can i's home is pad i), pad index 3 = the gray buffer."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    pad_xy_tol: float = tunable(0.015)  # can centre within this of the cup axis. Honest by
    # construction: max physical in-cup offset = inner_r - can_r = 8 mm < 15 mm, so any can
    # genuinely seated counts; the NEXT pad is 200 mm away.
    seat_z_tol: float = tunable(0.010)  # can centre within this of seat height (cup floor +
    # can_h/2). A can stacked on an occupant sits ~95 mm high — rejected by two orders.
    upright_max_deg: float = tunable(15.0)  # can axis within this of vertical. Max physical
    # in-cup lean ~ asin(2*(inner_r-can_r)/can_h) ~ 10 deg < 15, so a leaning-but-seated can
    # counts; a can lying across the collar mouth (90 deg) never does.
    settle_lin: float = tunable(0.04)   # max |lin vel| when judging settled (m/s)
    settle_ang: float = tunable(0.60)   # max |ang vel| when judging settled (rad/s)
    drop_margin: float = tunable(0.08)  # `dropped` latches when a PRESENT can centre falls
    # below counter_top + this: any resting pose ON the counter (standing 0.048, lying 0.030
    # above it) is inside the band; a can seated on a pad is 87 mm above the threshold.
    lift_above: float = tunable(0.11)   # `lifted` latches when a present can centre rises
    # above cup_floor + this (a seated can's TOP is 47 mm lower — no false latch).

    # --- tunable: score weights --------------------------------------------------------------
    w_lift: float = tunable(0.05)    # latched: some can was genuinely lifted clear
    w_buffer: float = tunable(0.10)  # latched: some can was parked seated in the buffer cup
    w_frac: float = tunable(0.55)    # x fraction of present cans currently correct
    drop_cap: float = tunable(0.25)  # hard score ceiling forever once `dropped` latched

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    p_three: float = tunable(0.5)       # P(all three cans present); else one absent, pair swap
    pad_jitter: float = tunable(0.03)   # uniform +/- xy jitter of every pedestal at reset
    can_seat_jitter: float = tunable(0.004)  # uniform +/- xy jitter of the can inside its cup

    # --- info: structure ---------------------------------------------------------------------
    counter_size: tuple = info((0.85, 1.05))
    counter_top: float = info(0.22)   # counter slab top height (the forbidden surface)
    ped_h: float = info(0.12)         # cup floor height above the counter top
    column_r: float = info(0.045)
    inner_r: float = info(0.038)      # collar aperture inradius; funnel = inner_r - can_r = 8 mm
    wall_t: float = info(0.008)
    wall_h: float = info(0.022)       # shallow collar: holds a standing can, hides nothing
    can_r: float = info(0.030)        # 60 mm can in a 76 mm aperture: ONE can per cup, ever
    can_h: float = info(0.095)
    can_mass: float = info(0.05)
    pad_nominal: tuple = info(((0.0, -0.30), (0.0, -0.10), (0.0, 0.10), (0.0, 0.30)))
    # (red, green, blue, buffer) pedestal centres; 200 mm apart >> jitter + diameters.
    can_colors: tuple = info(((0.85, 0.15, 0.15), (0.15, 0.62, 0.20), (0.20, 0.35, 0.85)))
    buffer_color: tuple = info((0.45, 0.45, 0.45))
    column_color: tuple = info((0.35, 0.33, 0.30))
    friction: float = info(0.60)
    contact_offset: float = info(0.002)  # small: the 8 mm radial funnel must not be eaten
    # by speculative contact (the pinch trap) — 2+2 mm combined < the centred 8 mm gap.
    parking_pos: tuple = info((1.1, 1.1))  # ground depot for the absent can (off camera,
    # off the counter — masked out of every latch and every rubric term)

    # Derived (filled in __post_init__).
    cup_floor_z: float = field(default=None, init=False)
    seat_z: float = field(default=None, init=False)
    n_cans: int = field(default=3, init=False)

    def __post_init__(self) -> None:
        self.cup_floor_z = round(self.counter_top + self.ped_h, 4)
        self.seat_z = round(self.cup_floor_z + self.can_h / 2, 4)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("musical_cans")
class MusicalCansScene(BaseScene):
    cfg: MusicalCansSceneCfg

    def __init__(self, cfg: MusicalCansSceneCfg | None = None) -> None:
        super().__init__(cfg or MusicalCansSceneCfg())

    # ----- assets ---------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic counter slab, four kinematic pedestals (three home
        pads colored like their cans + the gray buffer), three dynamic cans. reset()
        re-poses everything."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.friction, dynamic_friction=c.friction, restitution=0.0)
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)

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
            "counter": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Counter",
                spawn=sim_utils.CuboidCfg(
                    size=(c.counter_size[0], c.counter_size[1], c.counter_top),
                    rigid_props=kin, collision_props=coll, physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.32, 0.38)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.counter_top / 2)),
            ),
        }
        pad_colors = list(c.can_colors) + [c.buffer_color]
        for k in range(4):
            out[f"pad_{k}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad_" + str(k),
                spawn=_pedestal_spawner_cfg(
                    column_r=c.column_r, column_h=c.ped_h + 0.005, inner_r=c.inner_r,
                    wall_t=c.wall_t, wall_h=c.wall_h, color=pad_colors[k],
                    column_color=c.column_color, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pad_nominal[k][0], c.pad_nominal[k][1], c.cup_floor_z)),
            )
        for i in range(c.n_cans):
            out[f"can_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Can_" + str(i),
                spawn=sim_utils.CylinderCfg(
                    radius=c.can_r, height=c.can_h,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.10),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.can_mass),
                    collision_props=coll, physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.can_colors[i]),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pad_nominal[i][0], c.pad_nominal[i][1], c.seat_z + 0.003)),
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
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles + allocate the per-episode arrangement/latch buffers."""
        super().bind(env)
        n, dev = env.num_envs, env.device
        c = self.cfg
        self.pads: list[RigidObject] = [env.iscene[f"pad_{k}"] for k in range(4)]
        self.cans: list[RigidObject] = [env.iscene[f"can_{i}"] for i in range(c.n_cans)]
        self.env_origins = env.iscene.env_origins
        self.pad_c = torch.zeros(n, 4, 2, device=dev)     # pedestal centres (env-local xy)
        self.start_pad = torch.zeros(n, c.n_cans, dtype=torch.long, device=dev)  # -1 = absent
        self.present = torch.ones(n, c.n_cans, dtype=torch.bool, device=dev)
        self.dropped = torch.zeros(n, dtype=torch.bool, device=dev)      # touched the counter
        self.lifted = torch.zeros(n, dtype=torch.bool, device=dev)       # rose clear of a cup
        self.buffer_used = torch.zeros(n, dtype=torch.bool, device=dev)  # parked in the buffer

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter the four pedestals (kinematic pose writes), sample the
        derangement (all-3 cycle or 2-can swap with one absent), seat each present can in
        its sampled START pad with tiny jitter, park the absent can in the ground depot,
        clear all latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- pedestals: nominal + jitter, pose-only writes (kinematic decoration) ---
        nominal = torch.tensor(c.pad_nominal, device=dev)  # (4, 2)
        centres = nominal.unsqueeze(0) + (torch.rand(m, 4, 2, device=dev) * 2 - 1) * c.pad_jitter
        self.pad_c[env_ids] = centres
        for k in range(4):
            st = torch.zeros(m, 7, device=dev)
            st[:, 0:2] = centres[:, k]
            st[:, 2] = c.cup_floor_z
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            self.pads[k].write_root_pose_to_sim(st, env_ids)

        # --- derangement: start_pad[i] != i for every present can ---
        idx = torch.arange(c.n_cans, device=dev)
        three = torch.rand(m, device=dev) < c.p_three
        shift = torch.randint(1, 3, (m, 1), device=dev)           # 1 or 2: the two 3-cycles
        p3 = (idx.unsqueeze(0) + shift) % 3
        absent = torch.randint(0, 3, (m, 1), device=dev)          # the sampled missing can
        p2 = (3 - absent - idx.unsqueeze(0))                      # partner swap of the pair
        p2 = torch.where(idx.unsqueeze(0) == absent, torch.full_like(p2, -1), p2)
        sp = torch.where(three.unsqueeze(1), p3, p2)
        self.start_pad[env_ids] = sp
        self.present[env_ids] = sp >= 0

        # --- cans: seated in their start cups (present) or ground depot (absent) ---
        for i in range(c.n_cans):
            pad_i = sp[:, i].clamp(min=0)
            seat = centres[torch.arange(m, device=dev), pad_i]
            seat = seat + (torch.rand(m, 2, device=dev) * 2 - 1) * c.can_seat_jitter
            park = torch.tensor([c.parking_pos[0] + 0.12 * i, c.parking_pos[1]], device=dev)
            pres = (sp[:, i] >= 0).unsqueeze(1)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = torch.where(pres, seat, park.expand(m, 2))
            st[:, 2] = torch.where(pres.squeeze(1),
                                   torch.full((m,), c.seat_z + 0.003, device=dev),
                                   torch.full((m,), c.can_h / 2 + 0.002, device=dev))
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            self.cans[i].write_root_state_to_sim(st, env_ids)

        self.dropped[env_ids] = False
        self.lifted[env_ids] = False
        self.buffer_used[env_ids] = False

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch bookkeeping every physics substep (buffers fresh): DROPPED when any
        present can's centre falls to counter level (real fall physics — the cups are
        87 mm higher); LIFTED when any present can rises clear above cup height;
        BUFFER_USED when any present can is settled seated in the buffer cup."""
        c = self.cfg
        pos = self._can_pos_local()                       # (n, 3cans, 3)
        z = pos[:, :, 2]
        self.dropped |= ((z < c.counter_top + c.drop_margin) & self.present).any(dim=1)
        self.lifted |= ((z > c.cup_floor_z + c.lift_above) & self.present).any(dim=1)
        on_buf = self._seated_on(3) & self.settled() & self.present
        self.buffer_used |= on_buf.any(dim=1)

    # ----- state (full, restorable) ----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "cans": [b.data.root_state_w[env_ids].clone() for b in self.cans],
            "pads": [b.data.root_state_w[env_ids].clone() for b in self.pads],
            "pad_c": self.pad_c[env_ids].clone(),
            "start_pad": self.start_pad[env_ids].clone(),
            "present": self.present[env_ids].clone(),
            "dropped": self.dropped[env_ids].clone(),
            "lifted": self.lifted[env_ids].clone(),
            "buffer_used": self.buffer_used[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for b, st in zip(self.cans, state["cans"]):
            b.write_root_state_to_sim(st, env_ids)
        for b, st in zip(self.pads, state["pads"]):
            b.write_root_pose_to_sim(st[:, 0:7], env_ids)
        self.pad_c[env_ids] = state["pad_c"]
        self.start_pad[env_ids] = state["start_pad"]
        self.present[env_ids] = state["present"]
        self.dropped[env_ids] = state["dropped"]
        self.lifted[env_ids] = state["lifted"]
        self.buffer_used[env_ids] = state["buffer_used"]

    # ----- description -----------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A kitchen counter (top {c.counter_top * 100:.0f} cm high) carries a row of four "
            f"pedestal cups, each a {c.ped_h * 100:.0f} cm column topped by a shallow collar "
            f"cup that seats exactly ONE drink can ({2 * c.can_r * 1000:.0f} mm cans in "
            f"{2 * c.inner_r * 1000:.0f} mm cups). Three cups are color-coded — red, green, "
            f"blue — and the fourth, gray one is a spare. Matching colored cans stand seated "
            f"in the colored cups, but SHUFFLED: no can is in the cup of its own color (some "
            f"episodes have only two cans, swapped; some have all three, cycled). The spare "
            f"gray cup is empty.\n"
            f"Goal: rearrange until every can stands seated, upright and at rest, in the cup "
            f"of its own color. Constraints that shape the plan: a cup holds only one can, so "
            f"a can cannot be put into its home cup while the wrong can still sits there — "
            f"and the counter itself is soaking wet, so a can may NEVER be set down on it "
            f"(a can that touches counter level is ruined and the episode cannot fully "
            f"succeed). The only legal waypoint is the spare gray cup: move one can there "
            f"first, then cascade the others home, and finish by moving the parked can out "
            f"of the spare into its freed home."
        )

    # ----- progress / rubric ------------------------------------------------------------------
    def _can_pos_local(self) -> torch.Tensor:
        """(N, 3cans, 3) can centres in env-local coords."""
        return torch.stack(
            [b.data.root_pos_w - self.env_origins for b in self.cans], dim=1)

    def upright(self) -> torch.Tensor:
        """(N, 3cans) bool: can axis within `upright_max_deg` of vertical (either end)."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        cos_max = math.cos(math.radians(self.cfg.upright_max_deg))
        cols = []
        for b in self.cans:
            axis = quat_apply(b.data.root_quat_w, ez)
            cols.append(axis[:, 2].abs() >= cos_max)
        return torch.stack(cols, dim=1)

    def settled(self) -> torch.Tensor:
        """(N, 3cans) bool: per-can lin + ang velocity below the settle gates."""
        c = self.cfg
        cols = []
        for b in self.cans:
            lin = b.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
            ang = b.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang
            cols.append(lin & ang)
        return torch.stack(cols, dim=1)

    def _seated_on(self, pad_k: int) -> torch.Tensor:
        """(N, 3cans) bool, geometric: can centre within `pad_xy_tol` of pad k's cup axis,
        standing at seat height, upright. (Settled and present are judged separately.)"""
        c = self.cfg
        pos = self._can_pos_local()
        near = (pos[:, :, :2] - self.pad_c[:, pad_k].unsqueeze(1)).norm(dim=-1) <= c.pad_xy_tol
        z_ok = (pos[:, :, 2] - c.seat_z).abs() <= c.seat_z_tol
        return near & z_ok & self.upright()

    def correct(self) -> torch.Tensor:
        """(N, 3cans) bool: can i seated in ITS OWN color pad (pad i), settled, present."""
        cols = [self._seated_on(i)[:, i] for i in range(self.cfg.n_cans)]
        return torch.stack(cols, dim=1) & self.settled() & self.present

    def all_correct(self) -> torch.Tensor:
        """(N,) bool: every PRESENT can correct (judged on the sampled subset)."""
        return (self.correct() | ~self.present).all(dim=1)

    def success(self) -> torch.Tensor:
        """(N,) bool: all present cans seated home, settled — and nothing was ever dropped
        to counter level. Current-state physical predicate gated by the drop history."""
        return self.all_correct() & ~self.dropped

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0 for doing nothing (the deranged start has zero correct
        cans); else w_lift for the latched lift + w_buffer for the latched buffer visit +
        w_frac * fraction-correct; capped at `drop_cap` forever once dropped; exactly 1.0
        iff success. Max non-success = w_lift + w_buffer + w_frac * (n-1)/n ~= 0.52."""
        c = self.cfg
        n_pres = self.present.sum(dim=1).clamp(min=1).float()
        frac = self.correct().sum(dim=1).float() / n_pres
        base = c.w_lift * self.lifted.float() + c.w_buffer * self.buffer_used.float() \
            + c.w_frac * frac
        base = torch.where(self.dropped, base.clamp(max=c.drop_cap), base)
        return torch.where(self.success(), torch.ones_like(base), base)


# ----- env registration ------------------------------------------------------------------------
register_env("simgen", lambda: EnvCfg(scene="musical_cans", robot="null", env_spacing=4.0))

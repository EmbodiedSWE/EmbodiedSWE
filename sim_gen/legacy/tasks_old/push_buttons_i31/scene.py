"""CircuitBridgeScene — repair three broken circuits by laying conductor bars across
their terminal-post pairs, in the commanded color order (seed: rlbench/push_buttons,
strategically inverted).

The seed task is a sequence of momentary fingertip presses: identify the commanded
colored buttons and poke each in order — no object is moved, nothing persists, the only
skill is ordered reach-and-touch. Here the three colored "buttons" become three BROKEN
CIRCUITS: each is a pair of rigid terminal posts separated by an open trench, and
nothing on the console is pressable — the fixtures are kinematic, so any amount of
ordered poking accomplishes exactly nothing. To energize a circuit the solver must
FETCH the matching colored conductor bar from the scatter area on the far side, carry
it over, and LAY IT ACROSS the two posts so that it genuinely spans the gap: resting
elevated on BOTH post tops, aligned with the pair axis, level, and settled. A bar
balanced on one post, dropped into the trench, or laid on the wrong-color pair earns
nothing. The circuits must be energized in the fixed commanded order (red, then green,
then blue — printed in describe()); completing a circuit while an earlier one is still
dead trips a permanent interlock that voids success. The solver's plan is therefore
transport + two-point-support precision placement under an ordering constraint — a
different plan from the seed's ordered pokes, not different numbers.

Mechanism (all procedural primitives): each terminal pair is ONE kinematic compound
body (base strip + two square posts, authored by a custom spawner — child colliders of
one body never self-collide), re-posed per episode with `write_root_state_to_sim` (the
proven kinematic-furniture randomization pattern). The bars are plain dynamic cuboids.
A lamp dome on each fixture glows in the circuit color while its bridge is live (the
visible outcome for video). Geometry makes the rubric honest by construction: posts are
40 mm deep with inner faces 70 mm apart and the bar is 160 mm long, so an aligned bar
keeps both-post support up to ~30 mm of along-axis offset, loses the far post entirely
at 45 mm (it tips into the trench), and a perpendicular bar (30 mm wide) falls straight
through the 70 mm gap onto the base strip 60 mm below the post tops.

Judged on PHYSICAL outcome only: a circuit is `bridged_now` when its OWN bar's center
is over the gap center (along/lateral tolerances in the fixture frame), at post-top
height (elevation is the spanning test), axis-aligned with the pair, level, and settled
— sustained `hold_steps` consecutive substeps before the completion latches (the
anti-transient gate). success() = all three circuits currently held bridged AND their
first completions happened in the commanded order. score() in [0, 1]: 0.1 latched once
any bar is lifted clear of the floor, +0.25 latched per circuit completed in sequence,
capped at 0.45 forever once the order interlock trips, 1.0 iff success.

Per-episode randomization: fixture-to-station permutation + per-fixture xy jitter +
free yaw (the commanded order is fixed but WHERE each color sits is not), bar-to-slot
permutation on the scatter arc + per-bar xy jitter + free yaw.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
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


# ----- custom compound spawner ------------------------------------------------------------------
# One KINEMATIC rigid body per terminal pair: a base strip lying on the ground and two square
# posts on top of it, several child colliders authored with raw pxr APIs; only
# `isaaclab.sim.utils.clone` is borrowed (the per-env replicate machinery every CuboidCfg spawn
# uses). The lamp dome is authored later, in bind(), per cloned env prim (decorations authored
# after cloning — the duplicate-xformOp trap).

_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_terminal_pair(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one terminal-pair fixture at `prim_path`: root Xform (kinematic rigid body)
    with a base strip along local x and two posts at local x = +/- post_dx. Root origin
    sits at ground level, fixture centre. Explicit small contact offsets (the pc_gpu
    precedent: the default speculative margin would eat the trench clearance)."""
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
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(1.0)

    def box(name: str, size: tuple, pos: tuple, rgb: tuple) -> None:
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        sxf.AddScaleOp().Set(Gf.Vec3f(*size))
        seg.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
        UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    bx, by, bz = cfg.base_size
    px_, py, pz = cfg.post_size
    box("base", (bx, by, bz), (0.0, 0.0, bz / 2), cfg.base_color)
    for tag, sgn in (("post_a", -1.0), ("post_b", 1.0)):
        box(tag, (px_, py, pz), (sgn * cfg.post_dx, 0.0, bz + pz / 2), cfg.post_color)
    return root


def _pair_spawner_cfg(*, base_size: tuple, post_size: tuple, post_dx: float,
                      base_color: tuple, post_color: tuple, contact_offset: float) -> Any:
    """Build (lazily, app required) the terminal-pair spawner cfg — `clone` wraps
    `_spawn_terminal_pair` exactly like `spawn_cuboid` is wrapped."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "pair" not in _SPAWNER_CACHE:

        @configclass
        class TerminalPairSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_terminal_pair)
            base_size: tuple = (0.17, 0.064, 0.012)
            post_size: tuple = (0.04, 0.04, 0.06)
            post_dx: float = 0.055
            base_color: tuple = (0.24, 0.24, 0.28)
            post_color: tuple = (0.5, 0.5, 0.5)
            contact_offset: float = 0.004

        _SPAWNER_CACHE["pair"] = TerminalPairSpawnerCfg

    return _SPAWNER_CACHE["pair"](
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        base_size=base_size, post_size=post_size, post_dx=post_dx,
        base_color=base_color, post_color=post_color, contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CircuitBridgeSceneCfg(BaseCfg):
    """Config for `CircuitBridgeScene`. Tolerances are sized against the fixture geometry
    (see __doc__): the rubric's 35 mm along-axis window sits inside the ~40 mm physical
    both-post-support window, so "bridged" is honest — anything the rubric accepts is a
    real two-point span."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    along_tol: float = tunable(0.035)  # bar centre along the pair axis, from gap centre (m)
    lat_tol: float = tunable(0.018)  # bar centre lateral offset from the pair axis (m)
    z_tol: float = tunable(0.008)  # bar centre height error vs. resting-on-posts height (m)
    align_max_deg: float = tunable(20.0)  # bar long axis vs. pair axis (either direction)
    level_max_deg: float = tunable(12.0)  # bar long axis vs. horizontal
    settle_speed: float = tunable(0.05)  # max |v| of the bar when judging (m/s)
    hold_steps: int = tunable(30)  # consecutive substeps a bridge must persist to latch
    lift_h: float = tunable(0.05)  # bar bottom above this height latches "lifted"

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    fix_jitter: float = tunable(0.025)  # per-fixture xy jitter (m)
    fix_yaw_deg: float = tunable(180.0)  # per-fixture yaw range (+/-)
    shuffle_fixtures: bool = tunable(True)  # permute circuit -> station assignment per episode
    bar_jitter: float = tunable(0.04)  # per-bar xy jitter (m)
    bar_yaw_deg: float = tunable(180.0)  # per-bar yaw range (+/-)
    shuffle_bars: bool = tunable(True)  # permute bar -> scatter-slot assignment per episode

    # --- tunable: placement ------------------------------------------------------------------
    surface_z: float = tunable(0.0)  # work-surface height; 0 = on the ground (null smoke)
    fix_slots: tuple = tunable(((0.30, 0.0), (0.28, -0.28), (0.28, 0.28)))  # fixture stations
    bars_center: tuple = tunable((-0.16, 0.0))  # scatter-arc centre (bars on the far side)
    spawn_radius: float = tunable(0.20)  # scatter arc radius
    spawn_arc: tuple = tunable((90.0, 270.0))  # scatter arc (deg) around bars_center

    # --- info: structure ----------------------------------------------------------------------
    base_size: tuple = info((0.17, 0.064, 0.012))  # fixture base strip (x, y, z)
    post_size: tuple = info((0.04, 0.04, 0.06))  # each terminal post (x, y, z)
    post_dx: float = info(0.055)  # post centres at +/- this along the pair axis
    # Bar 160 mm long over posts whose inner faces are 70 mm apart: centred, it overlaps
    # each 40 mm post top fully; at 45 mm along-axis offset the near end reaches exactly
    # the far post's inner face (zero overlap) and the bar tips into the trench.
    bar_size: tuple = info((0.16, 0.03, 0.02))
    bar_mass: float = info(0.08)
    base_color: tuple = info((0.24, 0.24, 0.28))
    contact_offset: float = info(0.004)
    # (name, bar rgb, post rgb): commanded order = tuple order (red, green, blue).
    circuits: tuple = info((
        ("red", (0.88, 0.15, 0.12), (0.45, 0.10, 0.08)),
        ("green", (0.15, 0.75, 0.20), (0.08, 0.38, 0.12)),
        ("blue", (0.18, 0.35, 0.90), (0.08, 0.16, 0.45)),
    ))
    lamp_r: float = info(0.010)
    lamp_dim: tuple = info((0.35, 0.35, 0.35))

    # Derived (filled in __post_init__).
    post_top_z: float = field(default=None, init=False)  # post top above the surface (m)
    bridge_z: float = field(default=None, init=False)  # bar centre height when spanning (m)
    gap_inner: float = field(default=None, init=False)  # trench width between inner faces (m)

    def __post_init__(self) -> None:
        self.post_top_z = round(self.base_size[2] + self.post_size[2], 4)
        self.bridge_z = round(self.post_top_z + self.bar_size[2] / 2, 4)
        self.gap_inner = round(2 * self.post_dx - self.post_size[0], 4)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("circuit_bridge")
class CircuitBridgeScene(BaseScene):
    cfg: CircuitBridgeSceneCfg

    def __init__(self, cfg: CircuitBridgeSceneCfg | None = None) -> None:
        super().__init__(cfg or CircuitBridgeSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the three kinematic terminal-pair fixtures at their nominal
        stations, and the bars at their nominal scatter slots (reset() re-places all)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.surface_z

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
        for k, (name, _bar_rgb, post_rgb) in enumerate(c.circuits):
            sx, sy = c.fix_slots[k]
            out[f"pair_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pair_" + name,
                spawn=_pair_spawner_cfg(
                    base_size=c.base_size, post_size=c.post_size, post_dx=c.post_dx,
                    base_color=c.base_color, post_color=post_rgb,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(sx, sy, z0)),
            )

        a0, a1 = (math.radians(v) for v in c.spawn_arc)
        cx, cy = c.bars_center
        nb = len(c.circuits)
        for k, (name, bar_rgb, _post_rgb) in enumerate(c.circuits):
            ang = a0 + (a1 - a0) * (k + 0.5) / nb
            out[f"bar_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bar_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=c.bar_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bar_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=bar_rgb),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(cx + c.spawn_radius * math.cos(ang),
                         cy + c.spawn_radius * math.sin(ang),
                         z0 + c.bar_size[2] / 2 + 0.003)),
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

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles, allocate latches / hold counters, author the lamp domes."""
        super().bind(env)
        c = self.cfg
        n = env.num_envs
        dev = env.device
        self.names = [name for name, _b, _p in c.circuits]
        self.pairs: dict[str, RigidObject] = {
            name: env.iscene[f"pair_{name}"] for name in self.names}
        self.bars: dict[str, RigidObject] = {
            name: env.iscene[f"bar_{name}"] for name in self.names}
        self.env_origins = env.iscene.env_origins
        nk = len(self.names)
        # Ordering state: hold counters, first-completion latches, in-sequence credit,
        # the tripped interlock, and the lift latch.
        self._hold = torch.zeros(n, nk, dtype=torch.long, device=dev)
        self._first = torch.zeros(n, nk, dtype=torch.bool, device=dev)
        self._in_seq = torch.zeros(n, nk, dtype=torch.bool, device=dev)
        self._violated = torch.zeros(n, dtype=torch.bool, device=dev)
        self._lifted = torch.zeros(n, dtype=torch.bool, device=dev)
        self._build_lamps()

    def _build_lamps(self) -> None:
        """A lamp dome on each fixture's first post (visual child of the CLONED prim, no
        collision): glows in the circuit color while its bridge is live."""
        import omni.usd
        from pxr import Gf, UsdGeom

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        self._lamps = {}
        for name in self.names:
            lamps = []
            for i in range(self.env.num_envs):
                lamp = UsdGeom.Sphere.Define(
                    stage, f"/World/envs/env_{i}/Pair_{name}/lamp")
                lamp.CreateRadiusAttr(c.lamp_r)
                UsdGeom.Xformable(lamp.GetPrim()).AddTranslateOp().Set(
                    Gf.Vec3d(-c.post_dx, 0.0, c.post_top_z + c.lamp_r * 0.7))
                lamp.CreateDisplayColorAttr([Gf.Vec3f(*c.lamp_dim)])
                lamps.append(lamp)
            self._lamps[name] = lamps
        self._lamp_shown = {name: [False] * self.env.num_envs for name in self.names}

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: fixtures permuted over the stations with xy jitter + free yaw
        (kinematic re-posing — readback-verified pattern), bars permuted over the scatter
        arc with jitter + free yaw, all latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        nk = len(self.names)

        # --- fixtures: station permutation + jitter + yaw ---
        if c.shuffle_fixtures:
            station = torch.rand(m, nk, device=dev).argsort(dim=1)  # circuit k -> station
        else:
            station = torch.arange(nk, device=dev).expand(m, nk)
        slots = torch.tensor(c.fix_slots, device=dev, dtype=torch.float)  # (nk, 2)
        for k, name in enumerate(self.names):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = slots[station[:, k]]
            st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.fix_jitter
            st[:, 2] = c.surface_z
            half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.fix_yaw_deg) / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            self.pairs[name].write_root_state_to_sim(st, env_ids)

        # --- bars: scatter-slot permutation + jitter + free yaw ---
        a0, a1 = (math.radians(v) for v in c.spawn_arc)
        cx, cy = c.bars_center
        if c.shuffle_bars:
            slot = torch.rand(m, nk, device=dev).argsort(dim=1)
        else:
            slot = torch.arange(nk, device=dev).expand(m, nk)
        for k, name in enumerate(self.names):
            ang = a0 + (a1 - a0) * (slot[:, k].float() + 0.5) / nk
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = cx + c.spawn_radius * torch.cos(ang)
            st[:, 1] = cy + c.spawn_radius * torch.sin(ang)
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.bar_jitter
            st[:, 2] = c.surface_z + c.bar_size[2] / 2 + 0.003
            half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.bar_yaw_deg) / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            self.bars[name].write_root_state_to_sim(st, env_ids)

        self._hold[env_ids] = 0
        self._first[env_ids] = False
        self._in_seq[env_ids] = False
        self._violated[env_ids] = False
        self._lifted[env_ids] = False

    # ----- geometry queries ---------------------------------------------------------------------
    def _tensors(self) -> tuple[torch.Tensor, ...]:
        """(bar_pos, bar_quat, bar_vel, fix_pos, fix_quat), stacked (N, K, ...)."""
        bp = torch.stack([b.data.root_pos_w for b in self.bars.values()], dim=1)
        bq = torch.stack([b.data.root_quat_w for b in self.bars.values()], dim=1)
        bv = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                          for b in self.bars.values()], dim=1)
        fp = torch.stack([p.data.root_pos_w for p in self.pairs.values()], dim=1)
        fq = torch.stack([p.data.root_quat_w for p in self.pairs.values()], dim=1)
        return bp, bq, bv, fp, fq

    def bridged_now(self) -> torch.Tensor:
        """(N, K) bool, geometric + settled: circuit k's OWN bar spans its post pair —
        centre over the gap centre (fixture frame), at post-top height (the elevation
        clause is the spanning test), axis-aligned, level, and still."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        c = self.cfg
        bp, bq, bv, fp, fq = self._tensors()
        n, k = bp.shape[0], bp.shape[1]
        dp = quat_apply_inverse(fq.reshape(n * k, 4),
                                (bp - fp).reshape(n * k, 3)).reshape(n, k, 3)
        near = (dp[:, :, 0].abs() < c.along_tol) & (dp[:, :, 1].abs() < c.lat_tol) \
            & ((dp[:, :, 2] - c.bridge_z).abs() < c.z_tol)
        ex = torch.tensor([1.0, 0.0, 0.0], device=bp.device).expand(n * k, 3)
        bar_ax = quat_apply(bq.reshape(n * k, 4), ex).reshape(n, k, 3)
        fix_ax = quat_apply(fq.reshape(n * k, 4), ex).reshape(n, k, 3)
        aligned = (bar_ax * fix_ax).sum(dim=-1).abs() > math.cos(math.radians(c.align_max_deg))
        level = bar_ax[:, :, 2].abs() < math.sin(math.radians(c.level_max_deg))
        return near & aligned & level & (bv < c.settle_speed)

    # ----- mechanics (every substep) --------------------------------------------------------------
    def post_step(self) -> None:
        c = self.cfg
        now = self.bridged_now()
        self._hold = torch.where(now, self._hold + 1, torch.zeros_like(self._hold))

        # First-completion latches + the ordering interlock, ranks ascending so two
        # circuits crossing on the same substep still count as in-order.
        done = self._hold >= c.hold_steps
        for k in range(len(self.names)):
            cross = done[:, k] & ~self._first[:, k]
            earlier = self._first[:, :k].all(dim=1) if k > 0 else \
                torch.ones_like(self._violated)
            self._in_seq[:, k] |= cross & earlier
            self._violated |= cross & ~earlier
            self._first[:, k] |= cross

        # Lift latch: any bar clear of the floor.
        bp = torch.stack([b.data.root_pos_w for b in self.bars.values()], dim=1)
        bottom = bp[:, :, 2] - self.env_origins[:, None, 2] - c.bar_size[2] / 2
        self._lifted |= (bottom > c.surface_z + c.lift_h).any(dim=1)

        # Lamp refresh (env-wise, on change only).
        from pxr import Gf

        for k, name in enumerate(self.names):
            lit_col = c.circuits[k][1]
            for e in range(self.env.num_envs):
                lit = bool(now[e, k])
                if lit != self._lamp_shown[name][e]:
                    self._lamp_shown[name][e] = lit
                    col = lit_col if lit else c.lamp_dim
                    self._lamps[name][e].GetDisplayColorAttr().Set([Gf.Vec3f(*col)])

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "pairs": {n: p.data.root_state_w[env_ids].clone()
                      for n, p in self.pairs.items()},
            "bars": {n: b.data.root_state_w[env_ids].clone()
                     for n, b in self.bars.items()},
            "latches": {k: getattr(self, k)[env_ids].clone()
                        for k in ("_hold", "_first", "_in_seq", "_violated", "_lifted")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for n, p in self.pairs.items():
            p.write_root_state_to_sim(state["pairs"][n], env_ids)
        for n, b in self.bars.items():
            b.write_root_state_to_sim(state["bars"][n], env_ids)
        for k, v in state["latches"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        order = " -> ".join(self.names if hasattr(self, "names")
                            else [n for n, _b, _p in c.circuits])
        return (
            f"Three broken circuits stand on the surface, one per color (red, green, "
            f"blue): each is a pair of {c.post_size[2] * 1000:.0f} mm terminal posts on a "
            f"dark base strip, separated by a {c.gap_inner * 1000:.0f} mm open trench; a "
            f"grey lamp sits on each fixture. Nothing on the fixtures moves or can be "
            f"pressed. Scattered on the far side lie three conductor bars "
            f"({c.bar_size[0] * 1000:.0f}x{c.bar_size[1] * 1000:.0f}x"
            f"{c.bar_size[2] * 1000:.0f} mm), colored to match.\n"
            f"Goal: energize all three circuits IN ORDER ({order}) by laying each color's "
            f"bar ACROSS its own pair of posts so it genuinely spans the trench — resting "
            f"level on both post tops, aligned with the pair, and left there (the lamp "
            f"lights while a bridge is live). A bar balanced on a single post, dropped "
            f"into the trench, or laid on the wrong color's posts counts for nothing, and "
            f"energizing a circuit while an earlier one in the order is still dead trips "
            f"the interlock permanently — order matters."
        )

    # ----- progress / rubric ----------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: all three circuits currently held bridged (each hold counter past
        `hold_steps` — settled spans, live physical state) and no ordering violation."""
        return (self._hold >= self.cfg.hold_steps).all(dim=1) & ~self._violated

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0 for nothing; 0.1 latched once any bar was lifted clear
        of the floor; +0.25 latched per circuit completed in the commanded sequence;
        capped at 0.45 forever once the ordering interlock trips; 1.0 iff success."""
        n = self.env.num_envs
        dev = self.env.device
        s = torch.where(self._lifted, torch.full((n,), 0.1, device=dev),
                        torch.zeros(n, device=dev))
        s = s + 0.25 * self._in_seq.float().sum(dim=1)
        s = torch.where(self._violated, s.clamp(max=0.45), s)
        return torch.where(self.success(), torch.ones_like(s), s)


register_env("simgen", lambda: EnvCfg(scene="circuit_bridge", robot="null", env_spacing=3))

"""CatwalkBridgeScene — bridge a roofed trench with a plank, then push the red cube
across into the island shelter (pick_cube_i295).

Derived from the ManiSkill `pick_cube` seed but STRATEGICALLY DIFFERENT (see TASK.md):
the seed's whole plan is ONE grasp of a red cube and a 0.1 m free-space LIFT — no other
object matters, no structure, no order. Here the red cube's destination is a low ROOFED
island deck on the far side of an open TRENCH. The roof (7.2 cm of interior headroom)
makes every carry/drop strategy impossible — the seed's plan lands the cube on the roof
and earns nothing (reproduced and rejected in smoke). The only way across is to BUILD
THE PATH FIRST: a yellow plank must be inserted through the tunnel mouth and seated on
the sills on BOTH sides of the trench (its half-length 12 cm is shorter than the 14 cm
gap, so shoving it in lengthwise tips it into the trench — it must be carried in level
and set down; smoke proves the shove fails), and only then can the cube be pushed along
the plank, over the drop-off at its far end, onto the island floor. The push itself must
happen deep inside the tunnel, beyond fingertip reach — the blue rod is the poling tool
the embodiment uses for it (TASK.md).

No joints, no springs — the plant is pure passive rigid-body physics. post_step owns
the three dynamic bodies' wrench slots (solve.py's stand-in for fingertip/rod-tip
pushes), pre-encoded against the wrench frame drag (applied wrenches rotate with
rotation-since-reset on this stack), plus the two rubric latches.

Rubric (graded 0..1, anchored in the demonstrated solve.py trajectory):
  - 0.25 * latch_bridge: the plank was seen SPANNING — level, at sill height, its
    x-extent covering both sill edges, still (latched);
  - 0.35 * latch_transit: the cube was seen riding the spanning plank OVER the trench
    (on-plank height band, past mid-gap, moving slower than 0.5 m/s) (latched);
  - 1.0 iff success(): the cube rests ON THE ISLAND FLOOR (deck-height band — the
    on-plank band is 12 mm higher and excluded), well inside the island, everything
    settled. ~0 for the null policy; latched credit never evaporates.

Per-episode randomization (readback-verified in smoke): plank and cube are dealt to the
two bench slots by a sampled swap, each with xy jitter and free yaw; the rod gets its
own floor-slot jitter and +-30 deg yaw.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable
from robobench.core.registries import ENVS

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CatwalkBridgeSceneCfg(BaseCfg):
    """Config for `CatwalkBridgeScene`. Geometry is derived once in `__post_init__` so the
    scene, the smoke AND the solver read the same numbers."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    settle_lin: float = tunable(0.05)  # max |lin vel| on every dynamic body when judging (m/s)
    bridge_cover_lo: float = tunable(-0.085)  # plank x-extent must reach below this (start sill)
    bridge_cover_hi: float = tunable(0.065)  # ... and above this (island sill)
    bridge_z_band: tuple = tunable((0.081, 0.092))  # plank CENTRE z when seated on the sills
    bridge_up_min: float = tunable(0.98)  # plank up-axis z (level) when spanning
    transit_z_band: tuple = tunable((0.103, 0.126))  # cube CENTRE z riding the plank
    transit_x_band: tuple = tunable((0.0, 0.12))  # past mid-gap, before deep island
    transit_v_max: float = tunable(0.5)  # riding, not ballistic (m/s)
    goal_x_band: tuple = tunable((0.10, 0.25))  # cube CENTRE fully inside the island
    goal_y_half: float = tunable(0.14)
    goal_z_band: tuple = tunable((0.093, 0.107))  # ON THE ISLAND FLOOR (on-plank = 0.112+)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    slot_jitter: float = tunable(0.025)  # uniform +- xy jitter (plank/cube slots) at reset
    yaw_deg: float = tunable(180.0)  # uniform +- yaw (plank/cube) at reset
    rod_jitter: float = tunable(0.04)  # uniform +- xy jitter on the rod's floor slot
    rod_yaw_deg: float = tunable(30.0)  # uniform +- yaw on the rod (stays clear of the walls)

    # --- tunable: plant ----------------------------------------------------------------------
    plank_size: tuple = tunable((0.24, 0.10, 0.012))  # half-length 0.12 < gap 0.14 < 0.24
    plank_mass: float = tunable(0.15)
    cube_size: float = tunable(0.04)  # the seed's red cube
    cube_mass: float = tunable(0.05)
    rod_size: tuple = tunable((0.35, 0.02, 0.02))  # the poling tool (embodiment affordance)
    rod_mass: float = tunable(0.08)
    friction: float = tunable(0.30)  # dynamic friction, every body + fixture (pair mean = mu)

    # --- info: structure (env-local; ground plane at z = 0, deck tops at z0) -----------------
    z0: float = info(0.08)  # deck height (trench depth)
    mouth_x: float = info(-0.08)  # start-deck rear edge = tunnel mouth plane
    island_x0: float = info(0.06)  # island front edge (gap = island_x0 - mouth_x = 0.14)
    island_x1: float = info(0.26)  # island rear edge (back wall behind)
    bench_x0: float = info(-0.46)  # start-deck front edge
    deck_y_half: float = info(0.16)  # deck half-width (tunnel interior half-width)
    wall_t: float = info(0.03)
    tunnel_h: float = info(0.072)  # interior headroom above the decks (roof underside)
    roof_t: float = info(0.03)
    slots: tuple = info(((-0.34, 0.07), (-0.19, -0.07)))  # bench slots (plank/cube, dealt)
    rod_slot: tuple = info((-0.20, 0.36))  # rod floor slot (beside the bench, on the ground)
    span_center_x: float = info(-0.01)  # spanning pose: plank x in [-0.13, 0.11], 5 cm seats
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    gap: float = field(default=None, init=False)
    roof_z0: float = field(default=None, init=False)  # roof underside
    plank_seat_z: float = field(default=None, init=False)  # plank CENTRE z when seated
    cube_ride_z: float = field(default=None, init=False)  # cube CENTRE z riding the plank
    cube_deck_z: float = field(default=None, init=False)  # cube CENTRE z on a deck

    def __post_init__(self) -> None:
        self.gap = self.island_x0 - self.mouth_x
        assert self.plank_size[0] / 2 < self.gap < self.plank_size[0], \
            "plank must span the gap but be un-shovable (half-length < gap)"
        self.roof_z0 = self.z0 + self.tunnel_h
        self.plank_seat_z = self.z0 + self.plank_size[2] / 2
        self.cube_ride_z = self.z0 + self.plank_size[2] + self.cube_size / 2
        self.cube_deck_z = self.z0 + self.cube_size / 2


def _quat_z(rad: torch.Tensor) -> torch.Tensor:
    """(N,) angle about +z -> (N, 4) wxyz."""
    half = rad / 2
    q = torch.zeros(rad.shape[0], 4, device=rad.device)
    q[:, 0] = torch.cos(half)
    q[:, 3] = torch.sin(half)
    return q


# ----- scene -----------------------------------------------------------------------------------
class CatwalkBridgeScene(BaseScene):
    cfg: CatwalkBridgeSceneCfg

    def __init__(self, cfg: CatwalkBridgeSceneCfg | None = None) -> None:
        super().__init__(cfg or CatwalkBridgeSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic catwalk fixture (start bench, island deck, side
        walls, back wall, roof — the trench is the ABSENCE of floor between the decks),
        and the three dynamic bodies: yellow plank, red cube, blue rod."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        concrete = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.50))
        dark = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.28))
        yellow = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.78, 0.10))
        red_m = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.08, 0.08))
        blue_m = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.30, 0.88))
        mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.friction + 0.05, dynamic_friction=c.friction, restitution=0.0)
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)

        def kin_box(name: str, center: tuple, size: tuple, vis) -> RigidObjectCfg:
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.CuboidCfg(size=size, rigid_props=kin, collision_props=coll,
                                          physics_material=mat, visual_material=vis),
                init_state=RigidObjectCfg.InitialStateCfg(pos=center),
            )

        yh = c.deck_y_half
        roof_top = c.roof_z0 + c.roof_t
        wall_x0, wall_x1 = c.mouth_x, c.island_x1 + c.wall_t  # walls run mouth -> behind back wall
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
            "bench": kin_box(
                "Bench", ((c.bench_x0 + c.mouth_x) / 2, 0.0, c.z0 / 2),
                (c.mouth_x - c.bench_x0, 2 * yh, c.z0), concrete),
            "island": kin_box(
                "Island", ((c.island_x0 + c.island_x1) / 2, 0.0, c.z0 / 2),
                (c.island_x1 - c.island_x0, 2 * yh, c.z0), concrete),
            "wall_left": kin_box(
                "WallLeft", ((wall_x0 + wall_x1) / 2, -(yh + c.wall_t / 2), roof_top / 2),
                (wall_x1 - wall_x0, c.wall_t, roof_top), concrete),
            "wall_right": kin_box(
                "WallRight", ((wall_x0 + wall_x1) / 2, yh + c.wall_t / 2, roof_top / 2),
                (wall_x1 - wall_x0, c.wall_t, roof_top), concrete),
            "back_wall": kin_box(
                "BackWall", (c.island_x1 + c.wall_t / 2, 0.0, roof_top / 2),
                (c.wall_t, 2 * (yh + c.wall_t), roof_top), concrete),
            "roof": kin_box(
                "Roof", ((c.mouth_x + c.island_x1) / 2, 0.0, c.roof_z0 + c.roof_t / 2),
                (c.island_x1 - c.mouth_x, 2 * (yh + c.wall_t), c.roof_t), dark),
        }

        # post_step drives the bodies with external wrenches that do NOT wake a sleeping
        # body — sleep_threshold=0 keeps the plant live.
        dyn = dict(max_depenetration_velocity=0.5, solver_position_iteration_count=16,
                   solver_velocity_iteration_count=4, sleep_threshold=0.0,
                   stabilization_threshold=0.0)
        for name, size, mass, vis, damp in (
                ("plank", c.plank_size, c.plank_mass, yellow, 0.10),
                ("cube", (c.cube_size,) * 3, c.cube_mass, red_m, 0.10),
                ("rod", c.rod_size, c.rod_mass, blue_m, 0.15)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.title(),
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        linear_damping=damp, angular_damping=damp, **dyn),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                    collision_props=coll, physics_material=mat, visual_material=vis),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slots[0][0], c.slots[0][1], c.z0 + 0.05)),
            )
        out["rod"].init_state = type(out["rod"].init_state)(
            pos=(c.rod_slot[0], c.rod_slot[1], c.rod_size[2] / 2 + 0.003))
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
        super().bind(env)
        n = env.num_envs
        dev = env.device
        self.plank: RigidObject = env.iscene["plank"]
        self.cube: RigidObject = env.iscene["cube"]
        self.rod: RigidObject = env.iscene["rod"]
        self.bodies = [self.plank, self.cube, self.rod]  # push_f order: 0=plank 1=cube 2=rod
        self.env_origins = env.iscene.env_origins
        # Episode state.
        self.spawn_xy = torch.zeros(n, 3, 2, device=dev)  # sampled poses (readback reference)
        self.spawn_yaw = torch.zeros(n, 3, device=dev)
        self.spawn_swap = torch.zeros(n, dtype=torch.bool, device=dev)  # plank<->cube slot deal
        self.latch_bridge = torch.zeros(n, dtype=torch.bool, device=dev)
        self.latch_transit = torch.zeros(n, dtype=torch.bool, device=dev)
        # External push input (solve.py and smoke probes write WORLD-frame forces; post_step
        # consumes + owns the wrench slots — never call set_external_* directly).
        self.push_f = torch.zeros(n, 3, 3, device=dev)
        self._q_ref = torch.zeros(n, 3, 4, device=dev)  # quats at reset (wrench-drag pre-encode)
        self._q_ref[:, :, 0] = 1.0

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: deal plank and cube to the two bench slots by a sampled swap
        (position never identifies a body — shape/color does), each with xy jitter and free
        yaw; rod on its floor slot with jitter and +-rod_yaw_deg; clear latches and pushes.
        Burns one rand draw first (the first post-seed draw is degenerate on this stack)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        _ = torch.rand(m, 2, device=dev)  # burn: first post-seed draw is near-constant

        swap = torch.rand(m, device=dev) < 0.5
        slots = torch.tensor(c.slots, device=dev)  # (2, 2)
        plank_slot = torch.where(swap.unsqueeze(1), slots[1], slots[0])
        cube_slot = torch.where(swap.unsqueeze(1), slots[0], slots[1])
        yaw_amp = math.radians(c.yaw_deg)

        specs = (
            (self.plank, plank_slot, c.slot_jitter, yaw_amp, c.z0 + c.plank_size[2] / 2 + 0.003),
            (self.cube, cube_slot, c.slot_jitter, yaw_amp, c.cube_deck_z + 0.003),
            (self.rod, torch.tensor(c.rod_slot, device=dev).expand(m, 2), c.rod_jitter,
             math.radians(c.rod_yaw_deg), c.rod_size[2] / 2 + 0.003),
        )
        for i, (body, slot, jit, amp, z) in enumerate(specs):
            xy = slot + (torch.rand(m, 2, device=dev) * 2 - 1) * jit
            yaw = (torch.rand(m, device=dev) * 2 - 1) * amp
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            st[:, 3:7] = _quat_z(yaw)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)
            self.spawn_xy[env_ids, i] = xy
            self.spawn_yaw[env_ids, i] = yaw
            self._q_ref[env_ids, i] = st[:, 3:7]
        self.spawn_swap[env_ids] = swap
        self.latch_bridge[env_ids] = False
        self.latch_transit[env_ids] = False
        self.push_f[env_ids] = 0.0

    # ----- readings -----------------------------------------------------------------------------
    def _local(self, body) -> torch.Tensor:
        return body.data.root_pos_w - self.env_origins

    def bridge_ok(self) -> torch.Tensor:
        """(N,) bool, geometric: the plank is SPANNING — level (up-axis vertical), centre at
        sill height, its x-extent covering both sill edges, and still. Computed from the
        plank's live pose (a plank resting only on the island, tipped into the trench, or
        still on the bench all fail one clause or another)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        p = self._local(self.plank)
        q = self.plank.data.root_quat_w
        ex = torch.tensor([1.0, 0.0, 0.0], device=p.device).expand(n, 3)
        ey = torch.tensor([0.0, 1.0, 0.0], device=p.device).expand(n, 3)
        ez = torch.tensor([0.0, 0.0, 1.0], device=p.device).expand(n, 3)
        ax = quat_apply(q, ex)  # long axis
        ay = quat_apply(q, ey)
        up = quat_apply(q, ez)
        half_x = c.plank_size[0] / 2 * ax[:, 0].abs() + c.plank_size[1] / 2 * ay[:, 0].abs()
        level = up[:, 2] > c.bridge_up_min
        z_ok = (p[:, 2] > c.bridge_z_band[0]) & (p[:, 2] < c.bridge_z_band[1])
        cover = ((p[:, 0] - half_x) < c.bridge_cover_lo) & ((p[:, 0] + half_x) > c.bridge_cover_hi)
        still = self.plank.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin
        return level & z_ok & cover & still

    def cube_on_island(self) -> torch.Tensor:
        """(N,) bool, geometric: cube centre ON THE ISLAND FLOOR — deck-height z band (a cube
        still on the plank's far seat sits 12 mm higher and is excluded), fully inside the
        island interior."""
        c = self.cfg
        p = self._local(self.cube)
        return (p[:, 0] > c.goal_x_band[0]) & (p[:, 0] < c.goal_x_band[1]) \
            & (p[:, 1].abs() < c.goal_y_half) \
            & (p[:, 2] > c.goal_z_band[0]) & (p[:, 2] < c.goal_z_band[1])

    def settled(self) -> torch.Tensor:
        """(N,) bool: every dynamic body |lin vel| below settle_lin."""
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for b in self.bodies:
            ok &= b.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin
        return ok

    def success(self) -> torch.Tensor:
        """(N,) bool: the cube rests on the island floor, everything settled (current,
        physical state — how it got there is judged by physics: the only floor between the
        decks is the one the solver built)."""
        return self.cube_on_island() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 1.0 iff success(); else 0.25 * latch_bridge (a spanning
        bridge was built) + 0.35 * latch_transit (the cube crossed the trench riding it).
        Latched credit never evaporates; ~0 for the null policy."""
        partial = 0.25 * self.latch_bridge.float() + 0.35 * self.latch_transit.float()
        return torch.where(self.success(), torch.ones_like(partial), partial)

    # ----- step-coupled mechanics (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """The plant: consume the three bodies' push buffers (world-frame intent pre-encoded
        against the wrench frame drag) and update the rubric latches."""
        from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        zt = torch.zeros(n, 1, 3, device=dev)
        for i, b in enumerate(self.bodies):
            q_now = b.data.root_quat_w
            q_fix = quat_mul(self._q_ref[:, i, :], quat_conjugate(q_now))
            fc = quat_apply(q_fix, self.push_f[:, i, :])
            b.set_external_force_and_torque(fc.unsqueeze(1), zt)

        # Latches (bool; NaN-state comparisons are False, so garbage latches nothing).
        bridge = self.bridge_ok()
        self.latch_bridge |= bridge
        p = self._local(self.cube)
        v = self.cube.data.root_lin_vel_w.norm(dim=-1)
        riding = (p[:, 2] > c.transit_z_band[0]) & (p[:, 2] < c.transit_z_band[1]) \
            & (p[:, 0] > c.transit_x_band[0]) & (p[:, 0] < c.transit_x_band[1]) \
            & (p[:, 1].abs() < c.goal_y_half) & (v < c.transit_v_max)
        self.latch_transit |= riding & bridge

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = self._named_bodies()
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("spawn_xy", "spawn_yaw", "spawn_swap", "latch_bridge",
                               "latch_transit", "push_f", "_q_ref")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = self._named_bodies()
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    def _named_bodies(self) -> dict[str, Any]:
        return {"plank": self.plank, "cube": self.cube, "rod": self.rod}

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"Two concrete decks, {c.z0 * 100:.0f} cm high, face each other across an open "
            f"TRENCH {c.gap * 100:.0f} cm wide that drops to the floor: a long open BENCH on "
            f"the near side, and an ISLAND deck on the far side. From the bench's rear edge "
            f"onward, everything — the trench and the whole island — is covered by a flat "
            f"dark ROOF leaving only {c.tunnel_h * 100:.1f} cm of headroom above the decks, "
            f"walled on both sides and at the back: a low tunnel whose only opening is its "
            f"mouth at the bench edge. Nothing can be lowered in from above and no hand fits "
            f"inside. On the bench lie a YELLOW plank "
            f"({c.plank_size[0] * 100:.0f} x {c.plank_size[1] * 100:.0f} x "
            f"{c.plank_size[2] * 1000:.0f} mm) and a RED cube ({c.cube_size * 100:.0f} cm), "
            f"dealt to random spots each episode; a BLUE square rod "
            f"({c.rod_size[0] * 100:.0f} cm long, {c.rod_size[1] * 100:.0f} cm thick) lies "
            f"on the floor beside the bench.\n"
            f"Goal: the RED cube at rest ON THE ISLAND FLOOR, well inside the tunnel. The "
            f"trench has no floor, so the path must be BUILT first: seat the yellow plank "
            f"level across the trench so it rests on both deck edges (it is "
            f"{c.plank_size[0] * 100:.0f} cm long — long enough to span the "
            f"{c.gap * 100:.0f} cm gap with a seat on each sill; but note its half-length is "
            f"SHORTER than the gap, so shoving it forward along the deck tips it into the "
            f"trench — carry it in level through the mouth and set it down). Then move the "
            f"red cube across the plank bridge, over the step-down at its far end, onto the "
            f"island floor, and leave everything at rest. The far half of the crossing is "
            f"beyond arm's reach inside the tunnel — the blue rod is a poling tool: push the "
            f"cube ahead of the rod tip through the mouth. A cube dropped into the trench, "
            f"left on the plank, or parked on the roof does not count."
        )

    def instruction(self) -> str:
        return (
            "Bridge the open trench with the yellow plank so it rests level on both deck "
            "edges, then push the red cube across the plank — using the blue rod to reach "
            "inside the low tunnel — until the cube rests on the island floor beyond the "
            "plank. The cube must end at rest on the island deck itself; a cube in the "
            "trench, on the plank, or on the roof fails."
        )


# Guarded registration: the forge may import this module under two names.
if "catwalk_bridge" not in SCENES.list():
    SCENES.register("catwalk_bridge", CatwalkBridgeScene)
if "simgen.catwalk_bridge" not in ENVS.list():
    register_env("simgen", lambda: EnvCfg(scene="catwalk_bridge", robot="null"))

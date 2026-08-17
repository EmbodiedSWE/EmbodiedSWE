"""TunnelRelayScene — feed four blue blocks into a raised tunnel so the growing
column pushes an unreachable red cube off the far cliff into a green pen below.
Derived from rlbench/slide_block_to_target, but the judged object can never be
touched: it is moved only BY PROXY through a column of intermediate blocks.

Seed (rlbench/slide_block_to_target): push a red cube across the floor until it
sits on a flat target marker — one direct planar push on the goal object itself.
Here the red cube survives, and it still has to travel to a target region, but:

- The cube starts INSIDE a square bore (4.8 cm wide x 5.2 cm tall) running along
  the top of a raised deck. The bore is barely wider than the cube — no gripper,
  finger, or tool fits beside it, so the seed's strategy (contact the goal object
  and push it) is impossible BY CONSTRUCTION. The only usable contact surface is
  the column of feeder blocks inserted through the tunnel mouth.
- The target is no longer a flat marker to slide onto: it is a walled pen on the
  FLOOR, one deck-height below the bore. The cube reaches it only by being driven
  off the cliff at the tunnel's far end and dropping in — the final approach is a
  fall, not a slide, and a cube slid to any reachable floor location outside the
  pen scores nothing.
- The tunnel length is chosen so the task has an exact resource arithmetic: a
  column of THREE feeders pushed flush to the mouth leaves the cube short of the
  cliff (rubric rejects), while FOUR feeders eject it with the last feeder still
  proud of the mouth — the solver must relay ALL FOUR blocks, one behind another,
  through the same mouth, and never needs (nor is able) to reach inside.

So a solver needs a different PLAN (collect and sequence four proxy objects; the
goal object is never contacted) and different CODE STRUCTURE (a repeated
fetch/stage/insert cycle with column-depth readback, instead of one position servo
on the goal object). Execution order is NOT rubric-constrained: only the final
settled pose of the red cube is judged; physics itself forces sequential feeding.

Assets are fully procedural:
  - 8 KINEMATIC gray/green cuboids (deck, 2 bore walls, 2 roof strips leaving a
    1 cm sight slot, 3 pen walls) — a rigid rig re-posed per episode with xy
    jitter + yaw.
  - red cargo cube (4 cm, dynamic) spawned inside the bore.
  - 4 blue feeder blocks (5.5 x 4 x 4 cm, dynamic) scattered on the floor.

Per-episode randomization (readback-verifiable): rig xy + yaw, cargo depth inside
the bore + lateral jitter, feeder scatter positions + free yaw.

Rubric (0..1; partial progress latched):
  0.45 * prog   — latched running max of the cargo's normalized advance along the
                  bore toward the ejection line (gated to the bore lane / pen
                  region, so off-lane probe placements do not latch it)
  0.25 * eject  — cargo ever past the cliff plane AND below deck height (latched)
  1.0 iff success() — cargo settled on the floor INSIDE the pen, live.
  Non-success cap 0.70. Null policy ~0.

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
class TunnelRelaySceneCfg(BaseCfg):
    """Config for `TunnelRelayScene`. The interlock is geometric: the bore admits
    the 4 cm cube with 8 mm total lateral clearance and 12 mm headroom — no jaw or
    tool fits beside/above the cargo, so it can only be driven by a column of
    feeder blocks inserted through the mouth. Feeder arithmetic (asserted below):
    three feeders flush to the mouth leave the cargo centre 15 mm short of the
    cliff; four eject it with the last feeder still 40 mm proud of the mouth."""

    # --- tunable: randomization (task-family knobs) ---------------------------------------------
    rig_jitter: float = tunable(0.05)  # +/- xy jitter on the whole rig
    rig_yaw_deg: float = tunable(45.0)  # +/- rig yaw (limited: one Franka base pose stays valid)
    cargo_depth: tuple = tunable((0.03, 0.07))  # cargo centre x band inside the bore (rig-local)
    cargo_y_jitter: float = tunable(0.003)  # +/- lateral cargo jitter inside the bore
    feeder_jitter: float = tunable(0.025)  # +/- xy jitter on each feeder floor slot

    # --- tunable: rubric thresholds -------------------------------------------------------------
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.60)  # max |ang vel| when judging (rad/s)
    pen_x: tuple = tunable((0.158, 0.272))  # cargo centre x band inside the pen (rig-local)
    pen_y_tol: float = tunable(0.050)  # cargo centre |y| bound inside the pen
    pen_z: tuple = tunable((0.010, 0.050))  # cargo centre resting band (floor of the pen)

    # --- info: rig geometry (rig-local frame: origin = deck centre xy at floor level) -----------
    deck_size: tuple = info((0.30, 0.14, 0.12))  # deck; top at z=0.12, cliff face at x=+0.15
    tunnel_len: float = info(0.20)  # walls+roof span x in [-0.05, +0.15]; apron is [-0.15, -0.05]
    wall_t: float = info(0.010)
    bore_w: float = info(0.048)  # bore width (y)
    bore_h: float = info(0.052)  # bore height above the deck top
    roof_t: float = info(0.008)
    slot_w: float = info(0.010)  # open sight slot along the roof centreline
    pen_len: float = info(0.13)  # pen inner x span: cliff face 0.15 -> back wall 0.28
    pen_half_w: float = info(0.065)  # pen inner |y| bound
    pen_wall_t: float = info(0.012)
    pen_h: float = info(0.075)

    # --- info: dynamic bodies -------------------------------------------------------------------
    cargo_size: float = info(0.040)  # red cube edge
    cargo_mass: float = info(0.08)
    feeder_size: tuple = info((0.055, 0.040, 0.040))  # blue feeder block (x=length along bore)
    feeder_mass: float = info(0.10)
    # feeder floor slots (rig-local xy), scattered behind/off the mouth side of the deck
    feeder_slots: tuple = info(((-0.28, -0.15), (-0.42, -0.08), (-0.42, 0.08), (-0.28, 0.15)))

    # --- info: materials / colors ---------------------------------------------------------------
    static_friction: float = info(0.12)  # slick rig: a light push slides the column
    block_friction: float = info(0.30)
    ground_friction: float = info(0.60)  # grippy floor: the cargo stays where it lands
    contact_offset: float = info(0.002)
    deck_color: tuple = info((0.55, 0.55, 0.58))
    wall_color: tuple = info((0.45, 0.45, 0.50))
    roof_color: tuple = info((0.30, 0.30, 0.34))
    pen_color: tuple = info((0.10, 0.65, 0.20))  # green
    cargo_color: tuple = info((0.90, 0.10, 0.10))  # red
    feeder_color: tuple = info((0.15, 0.30, 0.90))  # blue

    # --- info: rubric geometry / weights --------------------------------------------------------
    x_eject: float = info(0.17)  # cargo-centre full-credit line (= cliff + cargo/2)
    w_prog: float = info(0.45)
    w_eject: float = info(0.25)  # 0.45 + 0.25 = 0.70 = the non-success cap

    # derived landmarks -------------------------------------------------------------------------
    @property
    def x_cliff(self) -> float:
        return self.deck_size[0] / 2  # +0.15

    @property
    def x_mouth(self) -> float:
        return self.x_cliff - self.tunnel_len  # -0.05

    @property
    def deck_top(self) -> float:
        return self.deck_size[2]  # 0.12

    def __post_init__(self) -> None:
        b = self.cargo_size
        fl, fw, fh = self.feeder_size
        # bore admits the blocks with clearance, but no tool beside/above them
        assert self.bore_w - b >= 0.006, "bore too tight for the cargo"
        assert self.bore_w - fw >= 0.006, "bore too tight for the feeders"
        assert self.bore_h - b >= 0.010, "bore headroom too small for the cargo"
        assert self.bore_h - fh >= 0.010, "bore headroom too small for the feeders"
        # tip-off: while pivoting over the cliff edge the cargo's top-face centre
        # rises to sqrt((b/2)^2 + b^2) above the deck; it must clear the roof end
        rise = math.sqrt((b / 2) ** 2 + b**2)
        assert rise <= self.bore_h - 0.006, f"tip-off corner rise {rise:.4f} fouls the roof"
        # resource arithmetic: 3 feeders flush to the mouth leave the cargo short...
        reach3 = self.x_mouth + 3 * fl + b / 2
        assert reach3 <= self.x_cliff - 0.005, f"3 feeders already eject (reach {reach3:.3f})"
        # ...4 feeders eject it, with the last feeder still proud of the mouth
        reach4 = self.x_mouth + 4 * fl + b / 2
        assert reach4 >= self.x_cliff + b / 2, f"4 feeders cannot eject (reach {reach4:.3f})"
        rear_at_eject = (self.x_cliff - b / 2) - 4 * fl
        assert rear_at_eject <= self.x_mouth - 0.015, (
            f"last feeder not proud of the mouth at ejection (rear {rear_at_eject:.3f})")
        # apron holds a feeder with room for the fingertip behind it
        apron = self.x_mouth - (-self.deck_size[0] / 2)
        assert apron >= fl + 0.03, "apron too short to stage a feeder"
        # cargo spawn keeps both faces >= 5 cm from both openings
        assert self.cargo_depth[0] - b / 2 - self.x_mouth >= 0.05, "cargo spawns too near the mouth"
        assert self.x_cliff - (self.cargo_depth[1] + b / 2) >= 0.05, "cargo spawns too near the cliff"
        # pen catches the drop; stacked-on-feeder rest sits above the pen z band
        assert self.pen_len >= 2 * b + 0.02, "pen too short"
        assert self.pen_z[1] < fh + b / 2, "pen z band admits a cargo stacked on a feeder"
        assert abs(self.x_eject - (self.x_cliff + b / 2)) < 1e-6, \
            "x_eject must sit at the ejection line"
        # feeder slots: pairwise clearance beats jitter + block diagonal
        need = 2 * self.feeder_jitter * math.sqrt(2) + math.hypot(fl, fw)
        slots = self.feeder_slots
        for i in range(len(slots)):
            for j in range(i + 1, len(slots)):
                d = math.hypot(slots[i][0] - slots[j][0], slots[i][1] - slots[j][1])
                assert d >= need, f"feeder slots {i},{j} can collide ({d:.3f} < {need:.3f})"


def _pieces(c: TunnelRelaySceneCfg) -> list[tuple[str, tuple, tuple, tuple]]:
    """The 8 kinematic rig pieces: (name, rig-local centre, size, color)."""
    dl, dw, dh = c.deck_size
    wx = c.x_cliff - c.tunnel_len / 2  # 0.05
    wall_y = c.bore_w / 2 + c.wall_t / 2
    wall_z = dh + c.bore_h / 2
    roof_w = (c.bore_w / 2 + c.wall_t - c.slot_w / 2)  # each strip: slot edge -> outer edge
    roof_y = c.slot_w / 2 + roof_w / 2
    roof_z = dh + c.bore_h + c.roof_t / 2
    pen_cx = c.x_cliff + c.pen_len / 2
    back_x = c.x_cliff + c.pen_len + c.pen_wall_t / 2
    side_y = c.pen_half_w + c.pen_wall_t / 2
    back_w = 2 * (c.pen_half_w + c.pen_wall_t)
    pz = c.pen_h / 2
    return [
        ("deck", (0.0, 0.0, dh / 2), (dl, dw, dh), c.deck_color),
        ("wall_l", (wx, wall_y, wall_z), (c.tunnel_len, c.wall_t, c.bore_h), c.wall_color),
        ("wall_r", (wx, -wall_y, wall_z), (c.tunnel_len, c.wall_t, c.bore_h), c.wall_color),
        ("roof_l", (wx, roof_y, roof_z), (c.tunnel_len, roof_w, c.roof_t), c.roof_color),
        ("roof_r", (wx, -roof_y, roof_z), (c.tunnel_len, roof_w, c.roof_t), c.roof_color),
        ("pen_back", (back_x, 0.0, pz), (c.pen_wall_t, back_w, c.pen_h), c.pen_color),
        ("pen_l", (pen_cx, side_y, pz), (c.pen_len, c.pen_wall_t, c.pen_h), c.pen_color),
        ("pen_r", (pen_cx, -side_y, pz), (c.pen_len, c.pen_wall_t, c.pen_h), c.pen_color),
    ]


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("tunnel_relay")
class TunnelRelayScene(BaseScene):
    cfg: TunnelRelaySceneCfg

    def __init__(self, cfg: TunnelRelaySceneCfg | None = None) -> None:
        super().__init__(cfg or TunnelRelaySceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        slick = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.static_friction, dynamic_friction=c.static_friction * 0.9,
            restitution=0.0)
        grip = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.block_friction, dynamic_friction=c.block_friction * 0.9,
            restitution=0.0)
        floor = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.ground_friction, dynamic_friction=c.ground_friction * 0.9,
            restitution=0.0)

        def kin_box(name: str, center: tuple, size: tuple, color: tuple) -> RigidObjectCfg:
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=slick,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=center),
            )

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(physics_material=floor),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
        }
        for name, center, size, color in _pieces(c):
            out[name] = kin_box(name.capitalize(), center, size, color)
        out["cargo"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cargo",
            spawn=sim_utils.CuboidCfg(
                size=(c.cargo_size, c.cargo_size, c.cargo_size),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.5, disable_gravity=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.cargo_mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                physics_material=grip,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.cargo_color),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.05, 0.0, c.deck_top + 0.022)),
        )
        for i in range(4):
            out[f"feeder{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Feeder" + str(i),
                spawn=sim_utils.CuboidCfg(
                    size=c.feeder_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5, disable_gravity=False),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.feeder_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=grip,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.feeder_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.feeder_slots[i][0], c.feeder_slots[i][1], c.feeder_size[2] / 2 + 0.002)),
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
        self.cargo: RigidObject = env.iscene["cargo"]
        self.feeders: list[RigidObject] = [env.iscene[f"feeder{i}"] for i in range(4)]
        self.rig: dict[str, RigidObject] = {
            name: env.iscene[name] for name, _, _, _ in _pieces(self.cfg)}
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self._rig_xy = torch.zeros(n, 2, device=dev)
        self._rig_yaw = torch.zeros(n, device=dev)
        self._x0 = torch.zeros(n, device=dev)
        self._need = torch.full((n,), self.cfg.x_eject, device=dev)
        # latches: partial progress survives transient achievements
        self._prog_max = torch.zeros(n, device=dev)
        self._eject_ever = torch.zeros(n, dtype=torch.bool, device=dev)

    # ----- frame helpers -------------------------------------------------------------------------
    def _rig_quat(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        yaw = self._rig_yaw if env_ids is None else self._rig_yaw[env_ids]
        half = yaw / 2
        q = torch.zeros(len(yaw), 4, device=yaw.device)
        q[:, 0] = torch.cos(half)
        q[:, 3] = torch.sin(half)
        return q

    def to_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world -> rig-local (origin at deck centre xy, floor z=0)."""
        p = pos_w - self.env_origins
        p = p.clone()
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

    def _write_rig(self, env_ids: torch.Tensor) -> None:
        """Re-pose the 8 kinematic rig pieces from the stored rig xy + yaw."""
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        q = self._rig_quat(env_ids)
        for name, center, _size, _color in _pieces(self.cfg):
            local = torch.tensor(center, device=dev).expand(m, 3).clone()
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = self.to_world(local, env_ids) + origin
            st[:, 3:7] = q
            self.rig[name].write_root_state_to_sim(st, env_ids)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: rig re-posed (xy jitter + yaw), cargo re-spawned inside
        the bore at a sampled depth, feeders scattered on their floor slots with
        free yaw; latches cleared. One rand draw is burned first (the first
        post-seed draw is near-degenerate)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        torch.rand(8, device=dev)  # burn the near-degenerate first draw

        # --- rig pose ---
        self._rig_xy[env_ids, 0] = (torch.rand(m, device=dev) * 2 - 1) * c.rig_jitter
        self._rig_xy[env_ids, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.rig_jitter
        self._rig_yaw[env_ids] = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rig_yaw_deg)
        self._write_rig(env_ids)
        q_rig = self._rig_quat(env_ids)

        # --- cargo: sampled depth inside the bore, lateral jitter, rig yaw ---
        x0 = c.cargo_depth[0] + torch.rand(m, device=dev) * (c.cargo_depth[1] - c.cargo_depth[0])
        y0 = (torch.rand(m, device=dev) * 2 - 1) * c.cargo_y_jitter
        z0 = torch.full((m,), c.deck_top + c.cargo_size / 2 + 0.002, device=dev)
        self._x0[env_ids] = x0
        self._need[env_ids] = c.x_eject - x0
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = self.to_world(torch.stack([x0, y0, z0], dim=1), env_ids) + origin
        st[:, 3:7] = q_rig
        self.cargo.write_root_state_to_sim(st, env_ids)

        # --- feeders: slots + jitter, free yaw ---
        for i, feeder in enumerate(self.feeders):
            sx = c.feeder_slots[i][0] + (torch.rand(m, device=dev) * 2 - 1) * c.feeder_jitter
            sy = c.feeder_slots[i][1] + (torch.rand(m, device=dev) * 2 - 1) * c.feeder_jitter
            sz = torch.full((m,), c.feeder_size[2] / 2 + 0.002, device=dev)
            yaw = self._rig_yaw[env_ids] + (torch.rand(m, device=dev) * 2 - 1) * math.pi
            half = yaw / 2
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = self.to_world(torch.stack([sx, sy, sz], dim=1), env_ids) + origin
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            feeder.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._prog_max[env_ids] = 0.0
        self._eject_ever[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "cargo": self.cargo.data.root_state_w[env_ids].clone(),
            "feeders": [f.data.root_state_w[env_ids].clone() for f in self.feeders],
            "rig_xy": self._rig_xy[env_ids].clone(),
            "rig_yaw": self._rig_yaw[env_ids].clone(),
            "x0": self._x0[env_ids].clone(),
            "need": self._need[env_ids].clone(),
            "prog_max": self._prog_max[env_ids].clone(),
            "eject_ever": self._eject_ever[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self._rig_xy[env_ids] = state["rig_xy"]
        self._rig_yaw[env_ids] = state["rig_yaw"]
        self._write_rig(env_ids)
        self.cargo.write_root_state_to_sim(state["cargo"], env_ids)
        for f, st in zip(self.feeders, state["feeders"]):
            f.write_root_state_to_sim(st, env_ids)
        self._x0[env_ids] = state["x0"]
        self._need[env_ids] = state["need"]
        self._prog_max[env_ids] = state["prog_max"]
        self._eject_ever[env_ids] = state["eject_ever"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        fl = c.feeder_size[0]
        return (
            f"A gray DECK ({c.deck_size[0] * 100:.0f} x {c.deck_size[1] * 100:.0f} cm, "
            f"{c.deck_top * 100:.0f} cm tall) stands on the floor. Along the front "
            f"{c.tunnel_len * 100:.0f} cm of its top runs a straight square TUNNEL: two "
            f"walls and a roof enclose a bore {c.bore_w * 100:.1f} cm wide and "
            f"{c.bore_h * 100:.1f} cm tall (a {c.slot_w * 1000:.0f} mm sight slot runs along "
            f"the roof centreline so the inside stays visible). The rear "
            f"{(c.x_mouth + c.deck_size[0] / 2) * 100:.0f} cm of the deck top is an open "
            f"APRON leading into the tunnel MOUTH; the far end of the tunnel is flush "
            f"with the deck's front face — a CLIFF dropping to the floor. On the floor "
            f"directly under the cliff sits a GREEN PEN (three walls "
            f"{c.pen_h * 100:.1f} cm tall; inner area {c.pen_len * 100:.0f} x "
            f"{2 * c.pen_half_w * 100:.0f} cm; the deck's front face closes its fourth "
            f"side; the floor is its bottom).\n"
            f"A RED CUBE ({c.cargo_size * 100:.0f} cm) rests INSIDE the tunnel, several "
            f"centimetres from both openings. The bore is barely wider and taller than "
            f"the cube: no gripper, finger, or tool fits beside or above it, so the red "
            f"cube can NEVER be touched, grasped, or pushed directly. FOUR BLUE BLOCKS "
            f"({fl * 100:.1f} x {c.feeder_size[1] * 100:.0f} x {c.feeder_size[2] * 100:.0f} cm) "
            f"lie scattered on the floor around the rear of the deck.\n"
            f"Goal: the red cube must end SETTLED ON THE FLOOR INSIDE THE GREEN PEN. "
            f"The only way to move it is by PROXY: slide blue blocks in through the "
            f"tunnel mouth, long side leading, one behind another, so the growing "
            f"column drives the red cube ahead of it until the cube tips off the cliff "
            f"and drops into the pen. The lengths are exact: a column of THREE blocks "
            f"pushed flush to the mouth still leaves the cube short of the cliff — ALL "
            f"FOUR blocks are needed, and when the cube falls the last block is still "
            f"sticking out of the mouth (nothing ever needs to enter the bore). A red "
            f"cube slid, dropped, or left anywhere else — on the deck, on the floor "
            f"beside the pen, on top of a block — does not count.\n"
            f"Execution order is NOT constrained: only the final settled state of the "
            f"red cube is judged (physics itself forces the blocks to be fed "
            f"sequentially through the same mouth)."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Feed the four blue blocks one after another into the tunnel mouth so the "
            "column pushes the red cube off the far cliff; the red cube must end "
            "resting on the floor inside the green pen."
        )

    # ----- readings / rubric ---------------------------------------------------------------------
    def cargo_local(self) -> torch.Tensor:
        """(N,3) cargo centre in the rig-local frame."""
        return self.to_local(self.cargo.data.root_pos_w)

    def feeder_local(self, i: int) -> torch.Tensor:
        """(N,3) feeder i centre in the rig-local frame."""
        return self.to_local(self.feeders[i].data.root_pos_w)

    def in_pen(self) -> torch.Tensor:
        """(N,) bool: cargo centre inside the pen, resting at floor height."""
        c = self.cfg
        p = self.cargo_local()
        return ((p[:, 0] > c.pen_x[0]) & (p[:, 0] < c.pen_x[1])
                & (p[:, 1].abs() <= c.pen_y_tol)
                & (p[:, 2] > c.pen_z[0]) & (p[:, 2] < c.pen_z[1]))

    def settled(self) -> torch.Tensor:
        """(N,) bool: cargo linear AND angular velocity below the settle gates."""
        c = self.cfg
        return ((self.cargo.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                & (self.cargo.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega))

    def _update_latches(self) -> None:
        c = self.cfg
        p = self.cargo_local()
        # progress counts only in the bore lane (riding the deck top) or already
        # past the cliff at floor height — off-lane placements cannot latch it
        in_lane = ((p[:, 2] > c.deck_top + 0.010) & (p[:, 2] < c.deck_top + 0.035)
                   & (p[:, 1].abs() < 0.030) & (p[:, 0] > self._x0 - 0.02))
        past_cliff = (p[:, 0] > c.x_cliff) & (p[:, 1].abs() < 0.060) & (p[:, 2] < 0.060)
        x_credit = torch.where(past_cliff, torch.full_like(p[:, 0], c.x_eject), p[:, 0])
        prog = ((x_credit - self._x0) / self._need).clamp(0.0, 1.0)
        prog = torch.nan_to_num(prog, nan=0.0, posinf=0.0, neginf=0.0)
        prog = torch.where(in_lane | past_cliff, prog, torch.zeros_like(prog))
        self._prog_max = torch.maximum(self._prog_max, prog)
        self._eject_ever |= ((p[:, 0] > c.x_cliff + 0.01) & (p[:, 1].abs() < 0.060)
                             & (p[:, 2] < c.deck_top - 0.005))

    # ----- step-coupled bookkeeping (every substep) ----------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No mechanism plant (rig kinematic, blocks free) — just latch rubric
        progress every step so transient achievements keep credit."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: cargo settled on the floor inside the pen — live physical
        outcome only (no latches)."""
        self._update_latches()
        return self.in_pen() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.45*prog + 0.25*eject_ever, latched, ~0 for the
        null policy, capped 0.70 — and 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_prog * self._prog_max
                + c.w_eject * self._eject_ever.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; the feeders are driven through applied wrenches.
register_env("simgen", lambda: EnvCfg(scene="tunnel_relay", robot="null"))

"""QuarryScene — uncover a buried gold cube and enthrone it, keeping all rubble in the
pit (sim_gen task `scoop_with_spatula_i387`).

Derived from rlbench/scoop_with_spatula ("scoop up the cube and lift it with the
spatula": grasp a thin-bladed TOOL, slide the blade under one free-standing cube,
carry it aloft on the blade). The MANIPULATION MODEL is replaced wholesale. The seed's
plan is tool-mediated scooping of an unobstructed object, judged on a transient,
hand-supported carry. Here there is NO tool, nothing is slid under anything, and
nothing is judged in the air. The task is OCCLUSION-DRIVEN EXCAVATION UNDER A
CONTAINMENT INVARIANT:

  A walled, open-top QUARRY PIT stands on the ground. At its west end a small GOLD
  cube (the prize) lies on the pit floor, BURIED under a pile of six larger gray
  RUBBLE stones — one capping stone rests directly on top of it, a ring of stones
  hems it in. East of centre the pit floor is empty. Outside the pit stands a blue
  PEDESTAL with a shallow square pocket on top.

  Goal: the gold cube seated in the pedestal pocket — with ALL SIX rubble stones
  still inside the pit walls at the end. The overburden must therefore be cleared
  WITHIN the pit (the empty east half exists for exactly this), the prize uncovered,
  extracted, and enthroned.

What the solver must bring, none of which exists in the seed:
  (1) occlusion reasoning: the prize is not directly accessible — the capping stone
      (and whatever the pile settles into) must be removed first; the uncover step
      is physically forced, not declared;
  (2) a containment invariant: rubble may only be relocated INSIDE the pit — the
      rubric's keep-in clause turns "fling the clutter anywhere" into failure;
  (3) targeted retrieval: the prize (gold, 35 mm) must be told from the rubble
      (gray, 40 mm) and is the only object with a destination outside the pit.

Assets are fully procedural:
  - pit: STATIC colliders — four walls on the ground plane (the ground is the pit
    floor). Fixed pose on purpose (the layout randomization lives in the contents).
  - pedestal: STATIC column with a pocket on top (floor = column top + four low
    walls). A cube physically inside the pocket is always within the seat tolerance
    (honest by construction); a cube perched on the pocket rim is rejected by BOTH
    the xy and the z clause (asserted in `__post_init__`).
  - prize: one 35 mm gold cube (0.10 kg). rubble: six 40 mm gray stones (0.12 kg).

Per-episode randomization (readback-verifiable): prize xy inside the burial zone,
the six stones permuted over the six pile slots (cap + 5-ring), ring rotation angle,
per-stone jitter and free yaw.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve trajectory):
  0.15 clear  — >= `need_clear` stones ever calm inside the pit, farther than
                `clear_r` from the prize's spawn point (all six in the pit)  (latched)
  0.20 expose — the prize ever UNCOVERED (no stone overhanging its top), stones all
                in the pit and calm                                          (latched)
  0.20 lift   — the prize ever above `lift_z` (out of the pit depth), stones all in
                the pit; gated on `expose`                                   (latched)
  0.15 near   — the prize ever within `near_r` of the seat point, stones all in the
                pit; gated on `lift`                                         (latched)
  1.0 iff success(): prize seated in the pedestal pocket + ALL six stones inside the
  pit + everything settled and finite. Non-success capped at 0.70.
Every latch requires all six stones inside the pit at that instant, so ejecting
rubble from the pit earns nothing — and success() re-checks containment live.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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


# ----- small quaternion helper (wxyz, torch, batched) -------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class QuarrySceneCfg(BaseCfg):
    """Config for `QuarryScene`. The geometric claims the rubric rests on (pocket
    honesty, rim-perch rejection, burial zone fits inside the walls, the clear
    threshold clears the pile footprint) are asserted in `__post_init__`."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    seat_tol: float = tunable(0.020)      # seated: prize xy within this of the pocket centre
    seat_z_tol: float = tunable(0.008)    # seated: prize centre z within this of the seat height
    settle_speed: float = tunable(0.05)   # max |lin vel| (prize AND stones) when judging (m/s)
    latch_speed: float = tunable(0.15)    # max stone |lin vel| for latches to arm
    cover_xy: float = tunable(0.038)      # covered: stone within this of the prize axis (xy)...
    cover_dz: float = tunable(0.026)      # ...AND stone centre above prize centre by this
    clear_r: float = tunable(0.10)        # 'clear' latch: stone farther than this from prize spawn
    need_clear: int = tunable(3)          # 'clear' latch: at least this many stones relocated
    lift_z: float = tunable(0.10)         # 'lift' latch: prize centre above this (above the rim)
    near_r: float = tunable(0.10)         # 'near' latch: prize within this (3D) of the seat point
    in_pit_z: float = tunable(0.14)       # in-pit: stone centre z below this (piles allowed)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    bury_x: tuple = tunable((0.17, 0.23))   # prize spawn x range (west / burial end)
    bury_y: float = tunable(0.042)          # prize spawn |y| bound
    cap_jitter: float = tunable(0.005)      # capping stone xy jitter on the prize top (+/- m)
    ring_r: tuple = tunable((0.056, 0.062)) # ring stone radius range around the prize
    ring_jit_deg: float = tunable(3.0)      # ring stone angle jitter (+/- deg)
    yaw_deg: float = tunable(180.0)         # free yaw for prize and stones (+/- deg)

    # --- info: pit (STATIC, fixed pose — the ground plane is the pit floor) ----------------------
    pit_center: tuple = info((0.28, 0.0))
    pit_inner: tuple = info((0.40, 0.26))   # inner span (x, y)
    wall_t: float = info(0.012)
    wall_h: float = info(0.055)
    wall_color: tuple = info((0.30, 0.24, 0.18))
    # --- info: pedestal (STATIC, fixed pose) -----------------------------------------------------
    ped_pos: tuple = info((0.28, 0.30))
    ped_xy: float = info(0.090)             # column footprint
    ped_h: float = info(0.12)               # column top = pocket floor
    pocket_in: float = info(0.052)          # pocket inner span
    pocket_wall_t: float = info(0.010)
    pocket_wall_h: float = info(0.014)
    ped_color: tuple = info((0.15, 0.35, 0.75))
    pocket_color: tuple = info((0.35, 0.55, 0.90))
    # --- info: bodies ----------------------------------------------------------------------------
    token_size: float = info(0.035)         # the gold prize cube
    rubble_size: float = info(0.040)        # the gray stones
    token_mass: float = info(0.10)
    rubble_mass: float = info(0.12)
    token_color: tuple = info((0.95, 0.78, 0.10))
    rubble_color: tuple = info((0.42, 0.42, 0.45))
    contact_offset: float = info(0.002)
    # --- info: clear-half drop slots (shared by solve/smoke; all inside the east half) -----------
    clear_slots: tuple = info(((0.34, -0.07), (0.34, 0.0), (0.34, 0.07),
                               (0.41, -0.07), (0.41, 0.0), (0.41, 0.07)))
    # rubric weights (0.15 + 0.20 + 0.20 + 0.15 = 0.70 = the non-success cap)
    w_clear: float = info(0.15)
    w_expose: float = info(0.20)
    w_lift: float = info(0.20)
    w_near: float = info(0.15)

    def __post_init__(self) -> None:
        px, py = self.pit_center
        ix, iy = self.pit_inner
        th, rh = self.token_size / 2, self.rubble_size / 2
        # pocket admits the prize with real slack, and is honest by construction:
        # any prize physically inside the pocket is within seat_tol of the centre
        slack = (self.pocket_in - self.token_size) / 2
        assert slack >= 0.006, f"pocket must admit the prize with slack, got {slack:.4f}"
        assert slack < self.seat_tol, "any in-pocket seat must count (honesty bound)"
        # a prize perched on the pocket rim is rejected by BOTH clauses
        seat_z = self.ped_h + th
        rim_z = self.ped_h + self.pocket_wall_h + th
        rim_xy = self.pocket_in / 2 + self.pocket_wall_t / 2
        assert rim_z - seat_z > self.seat_z_tol + 0.002, "rim perch must fail the z clause"
        assert rim_xy > self.seat_tol + 0.005, "rim perch must fail the xy clause"
        # burial zone (prize + stone ring) fits inside the west walls with margin,
        # and never reaches the drop slots in the east half
        reach = self.ring_r[1] + rh + 0.002
        assert self.bury_x[0] - reach >= px - ix / 2, "ring must fit west of the prize"
        assert self.bury_y + reach <= iy / 2 - 0.003, "ring must fit inside the y walls"
        min_slot_x = min(s[0] for s in self.clear_slots)
        assert self.bury_x[1] + reach < min_slot_x - rh - 0.004, \
            "the pile footprint must never reach the drop slots"
        # the ring never crowds two stones into overlap (worst yaw = diagonal contact)
        worst_gap = 2 * self.ring_r[0] * math.sin(math.radians(36 - self.ring_jit_deg))
        assert worst_gap > self.rubble_size * math.sqrt(2), "ring stones must not overlap"
        # cover predicate: the capping stone IS covered, a floor-level neighbour is NOT
        assert th + rh > self.cover_dz, "a capping stone must register as cover"
        assert rh - th < self.cover_dz, "a floor-level neighbour must not register"
        assert self.cover_xy > self.cap_jitter + 0.010, "jittered cap stays inside cover_xy"
        # clear threshold genuinely clears the pile footprint, and every drop slot is
        # inside the east half of the pit, beyond clear_r from any burial-zone prize
        assert self.clear_r > self.ring_r[1] + 0.03, "clear_r must clear the ring"
        for sx, sy in self.clear_slots:
            assert px < sx - rh and sx + rh < px + ix / 2 - 0.003, "slot in east half"
            assert abs(sy) + rh < iy / 2 - 0.003, "slot inside the y walls"
            # min distance from the slot to ANY prize spawn in the burial zone
            d = math.hypot(sx - self.bury_x[1], max(0.0, abs(sy) - self.bury_y))
            assert d > self.clear_r + 0.003, f"slot ({sx},{sy}) must clear clear_r"
        # a stone perched on a wall top rests its CoM over the wall-top patch, i.e.
        # centre |dx| >= ix/2 — outside the in-pit box (ix/2 - 0.005) by construction
        assert self.lift_z > self.wall_h + 0.02, "lift means clear of the pit depth"
        assert self.in_pit_z > self.wall_h + 2 * self.rubble_size, "in-pit allows piles"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("quarry")
class QuarryScene(BaseScene):
    cfg: QuarrySceneCfg

    RUBBLE_NAMES = tuple(f"rubble_{i}" for i in range(6))

    def __init__(self, cfg: QuarrySceneCfg | None = None) -> None:
        super().__init__(cfg or QuarrySceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        px, py = c.pit_center
        ix, iy = c.pit_inner
        mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.7, dynamic_friction=0.6, restitution=0.0)

        def static_box(prim, pos, size, color):
            return AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/" + prim,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
            )

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
        # pit walls (static; the ground plane is the pit floor)
        wz = c.wall_h / 2
        wall_specs = {
            "wall_xn": ((px - ix / 2 - c.wall_t / 2, py, wz),
                        (c.wall_t, iy + 2 * c.wall_t, c.wall_h)),
            "wall_xp": ((px + ix / 2 + c.wall_t / 2, py, wz),
                        (c.wall_t, iy + 2 * c.wall_t, c.wall_h)),
            "wall_yn": ((px, py - iy / 2 - c.wall_t / 2, wz),
                        (ix, c.wall_t, c.wall_h)),
            "wall_yp": ((px, py + iy / 2 + c.wall_t / 2, wz),
                        (ix, c.wall_t, c.wall_h)),
        }
        for key, (pos, size) in wall_specs.items():
            out[key] = static_box(key.capitalize(), pos, size, c.wall_color)
        # pedestal: column + pocket walls (static)
        bx, by = c.ped_pos
        out["ped_col"] = static_box("PedColumn", (bx, by, c.ped_h / 2),
                                    (c.ped_xy, c.ped_xy, c.ped_h), c.ped_color)
        wo = c.pocket_in / 2 + c.pocket_wall_t / 2
        span = c.pocket_in + 2 * c.pocket_wall_t
        pz = c.ped_h + c.pocket_wall_h / 2
        pocket_specs = {
            "ped_wxn": ((bx - wo, by, pz), (c.pocket_wall_t, span, c.pocket_wall_h)),
            "ped_wxp": ((bx + wo, by, pz), (c.pocket_wall_t, span, c.pocket_wall_h)),
            "ped_wyn": ((bx, by - wo, pz), (span, c.pocket_wall_t, c.pocket_wall_h)),
            "ped_wyp": ((bx, by + wo, pz), (span, c.pocket_wall_t, c.pocket_wall_h)),
        }
        for key, (pos, size) in pocket_specs.items():
            out[key] = static_box(key.capitalize(), pos, size, c.pocket_color)

        body_props = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.05, angular_damping=0.05,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=c.contact_offset, rest_offset=0.0),
            physics_material=mat,
        )
        out["token"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Token",
            spawn=sim_utils.CuboidCfg(
                size=(c.token_size,) * 3,
                mass_props=sim_utils.MassPropertiesCfg(mass=c.token_mass),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=c.token_color, roughness=0.25),
                **body_props,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.20, 0.0, 0.03)),
        )
        for i, name in enumerate(self.RUBBLE_NAMES):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rubble_" + str(i),
                spawn=sim_utils.CuboidCfg(
                    size=(c.rubble_size,) * 3,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.rubble_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.rubble_color, roughness=0.85),
                    **body_props,
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
        self.token: RigidObject = env.iscene["token"]
        self.rubble: dict[str, RigidObject] = {n: env.iscene[n] for n in self.RUBBLE_NAMES}
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # readback-able randomization record: prize spawn xy + stone->pile-slot map
        self.token_spawn = torch.zeros(n, 2, device=dev)
        self.pile_slot = torch.zeros(n, 6, dtype=torch.long, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._clear = torch.zeros(n, dtype=torch.bool, device=dev)
        self._expose = torch.zeros(n, dtype=torch.bool, device=dev)
        self._lift = torch.zeros(n, dtype=torch.bool, device=dev)
        self._near = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: prize at a random spot in the burial zone; the six stones
        permuted over the six pile slots — one CAP directly on the prize top, a
        5-stone ring around it (random rotation, jitter, free yaw). East half empty."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        _ = torch.rand(m, 4, device=dev)  # burn: first post-seed draws are degenerate

        yaw_amp = math.radians(c.yaw_deg)
        tx = c.bury_x[0] + torch.rand(m, device=dev) * (c.bury_x[1] - c.bury_x[0])
        ty = (torch.rand(m, device=dev) * 2 - 1) * c.bury_y
        self.token_spawn[env_ids, 0] = tx
        self.token_spawn[env_ids, 1] = ty

        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = tx, ty
        st[:, 2] = c.token_size / 2 + 0.003
        st[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1) * yaw_amp)
        st[:, 0:3] += origin
        self.token.write_root_state_to_sim(st, env_ids)

        # stones: permute over slots (slot 0 = cap, 1..5 = ring at 72 deg spacing)
        perm = torch.rand(m, 6, device=dev).argsort(dim=1)
        self.pile_slot[env_ids] = perm
        ang0 = torch.rand(m, device=dev) * 2 * math.pi
        for i, name in enumerate(self.RUBBLE_NAMES):
            slot = perm[:, i]
            is_cap = slot == 0
            k = (slot - 1).clamp(min=0).float()
            ang = ang0 + k * math.radians(72.0) \
                + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.ring_jit_deg)
            rr = c.ring_r[0] + torch.rand(m, device=dev) * (c.ring_r[1] - c.ring_r[0])
            ring_x = tx + rr * torch.cos(ang)
            ring_y = ty + rr * torch.sin(ang)
            cap_x = tx + (torch.rand(m, device=dev) * 2 - 1) * c.cap_jitter
            cap_y = ty + (torch.rand(m, device=dev) * 2 - 1) * c.cap_jitter
            cap_z = c.token_size + 0.003 + c.rubble_size / 2 + 0.005
            ring_z = c.rubble_size / 2 + 0.004
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = torch.where(is_cap, cap_x, ring_x)
            st[:, 1] = torch.where(is_cap, cap_y, ring_y)
            st[:, 2] = torch.where(is_cap, torch.full_like(cap_x, cap_z),
                                   torch.full_like(cap_x, ring_z))
            st[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1) * yaw_amp)
            st[:, 0:3] += origin
            self.rubble[name].write_root_state_to_sim(st, env_ids)

        self._clear[env_ids] = False
        self._expose[env_ids] = False
        self._lift[env_ids] = False
        self._near[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "token": self.token.data.root_state_w[env_ids].clone(),
            "rubble": {n: b.data.root_state_w[env_ids].clone()
                       for n, b in self.rubble.items()},
            "token_spawn": self.token_spawn[env_ids].clone(),
            "pile_slot": self.pile_slot[env_ids].clone(),
            "clear": self._clear[env_ids].clone(),
            "expose": self._expose[env_ids].clone(),
            "lift": self._lift[env_ids].clone(),
            "near": self._near[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.token.write_root_state_to_sim(state["token"], env_ids)
        for n, b in self.rubble.items():
            b.write_root_state_to_sim(state["rubble"][n], env_ids)
        self.token_spawn[env_ids] = state["token_spawn"]
        self.pile_slot[env_ids] = state["pile_slot"]
        self._clear[env_ids] = state["clear"]
        self._expose[env_ids] = state["expose"]
        self._lift[env_ids] = state["lift"]
        self._near[env_ids] = state["near"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        ix, iy = c.pit_inner
        return (
            f"A rectangular QUARRY PIT stands on the ground: four dark-brown walls "
            f"({c.wall_h * 1000:.0f} mm tall, open on top) enclosing a "
            f"{ix * 100:.0f} x {iy * 100:.0f} cm floor. In the WEST end of the pit a "
            f"small GOLD cube ({c.token_size * 1000:.0f} mm, the prize) lies on the pit "
            f"floor, BURIED under a pile of six larger GRAY rubble stones "
            f"({c.rubble_size * 1000:.0f} mm each): one capping stone rests directly on "
            f"top of the gold cube and the others hem it in. The EAST half of the pit "
            f"floor is empty. Outside the pit, to the north, stands a BLUE PEDESTAL "
            f"({c.ped_h * 100:.0f} cm tall) with a shallow square pocket on its top "
            f"({c.pocket_in * 1000:.0f} mm inner span, light-blue rim).\n"
            f"Goal: set the GOLD cube inside the pedestal pocket, resting on the pocket "
            f"floor — while ALL SIX gray stones end up INSIDE the pit walls. The gold "
            f"cube is buried, so first clear the stones off and away from it; put the "
            f"stones down in the empty east half of the pit (anywhere inside the walls "
            f"is fine — a stone left outside the pit at the end fails the task). Then "
            f"pick the uncovered gold cube out of the pit and seat it in the pocket. "
            f"Only the gold cube belongs on the pedestal; a gray stone in the pocket "
            f"counts for nothing. Success: the gold cube seated in the pocket and all "
            f"six stones inside the pit, everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Dig out the buried gold cube: move the gray stones off it and set them "
            "down in the empty half of the pit — every stone must stay inside the pit "
            "walls. Then place the gold cube into the pocket on top of the blue "
            "pedestal."
        )

    # ----- live predicates -----------------------------------------------------------------------
    def _rubble_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(local pos (N,6,3), |lin vel| (N,6)) for the six stones."""
        pos = torch.stack([b.data.root_pos_w for b in self.rubble.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.rubble.values()], dim=1)
        return pos - self.env_origins[:, None, :], vel

    def _token_local(self) -> torch.Tensor:
        return self.token.data.root_pos_w - self.env_origins

    def _in_pit(self, loc: torch.Tensor) -> torch.Tensor:
        """(..., 3) local positions -> bool: centre inside the pit box (a stone perched
        on a wall top is outside; stacked stones up to `in_pit_z` are inside)."""
        c = self.cfg
        px, py = c.pit_center
        ix, iy = c.pit_inner
        return ((loc[..., 0] - px).abs() < ix / 2 - 0.005) \
            & ((loc[..., 1] - py).abs() < iy / 2 - 0.005) \
            & (loc[..., 2] > 0.005) & (loc[..., 2] < c.in_pit_z)

    def rubble_in_pit(self) -> torch.Tensor:
        """(N,6) bool: each stone inside the pit."""
        loc, _v = self._rubble_tensors()
        return self._in_pit(loc)

    def all_in_pit(self) -> torch.Tensor:
        """(N,) bool: ALL six stones inside the pit — the containment invariant."""
        return self.rubble_in_pit().all(dim=1)

    def covered(self) -> torch.Tensor:
        """(N,) bool: some stone overhangs the prize (within `cover_xy` of its axis
        AND `cover_dz` above its centre) — the occlusion predicate."""
        c = self.cfg
        loc, _v = self._rubble_tensors()
        tl = self._token_local()[:, None, :]
        near = (loc[:, :, :2] - tl[:, :, :2]).norm(dim=-1) < c.cover_xy
        above = (loc[:, :, 2] - tl[:, :, 2]) > c.cover_dz
        return (near & above).any(dim=1)

    def token_seated(self) -> torch.Tensor:
        """(N,) bool: prize resting on the pocket floor — xy within `seat_tol` of the
        pocket centre, centre z within `seat_z_tol` of the seat height, still. A rim
        perch fails BOTH position clauses (asserted in cfg)."""
        c = self.cfg
        tl = self._token_local()
        seat = torch.tensor([c.ped_pos[0], c.ped_pos[1], c.ped_h + c.token_size / 2],
                            device=tl.device)
        xy_ok = (tl[:, :2] - seat[:2]).norm(dim=-1) < c.seat_tol
        z_ok = (tl[:, 2] - seat[2]).abs() < c.seat_z_tol
        still = self.token.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return xy_ok & z_ok & still

    def settled(self) -> torch.Tensor:
        """(N,) bool: prize AND all stones below `settle_speed`."""
        _p, vel = self._rubble_tensors()
        tok = self.token.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed
        return (vel < self.cfg.settle_speed).all(dim=1) & tok

    def _update_latches(self) -> None:
        c = self.cfg
        loc, vel = self._rubble_tensors()
        inpit = self._in_pit(loc)
        all_in = inpit.all(dim=1)
        calm = vel < c.latch_speed
        d_spawn = (loc[:, :, :2] - self.token_spawn[:, None, :]).norm(dim=-1)
        n_cleared = (inpit & calm & (d_spawn > c.clear_r)).sum(dim=1)
        self._clear |= all_in & (n_cleared >= c.need_clear)
        self._expose |= all_in & ~self.covered() & calm.all(dim=1)
        tl = self._token_local()
        self._lift |= self._expose & all_in & (tl[:, 2] > c.lift_z)
        seat = torch.tensor([c.ped_pos[0], c.ped_pos[1], c.ped_h + c.token_size / 2],
                            device=tl.device)
        self._near |= self._lift & all_in & ((tl - seat).norm(dim=-1) < c.near_r)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: prize seated in the pedestal pocket + ALL six stones inside the
        pit walls + everything settled and finite. All clauses are live physical
        outcomes on settled poses."""
        self._update_latches()
        loc, _v = self._rubble_tensors()
        finite = torch.isfinite(loc).all(dim=-1).all(dim=-1) \
            & torch.isfinite(self.token.data.root_pos_w).all(dim=-1)
        return self.token_seated() & self.all_in_pit() & self.settled() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15*clear + 0.20*expose + 0.20*lift + 0.15*near (all
        latched, all gated on the containment invariant; ~0 for doing nothing), capped
        at 0.70 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_clear * self._clear.float() + c.w_expose * self._expose.float()
                + c.w_lift * self._lift.float() + c.w_near * self._near.float()
                ).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="quarry", robot="null"))

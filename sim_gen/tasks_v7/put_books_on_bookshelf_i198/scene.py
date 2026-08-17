"""TiltShelfButtressScene — build a leaning book row against a bookend on a slanted shelf.

Derived from the RLBench `put_books_on_bookshelf` seed but STRATEGICALLY different (see
TASK.md): the seed stands loose books upright on a LEVEL, empty shelf — independent
pick-and-place moves, order-free, each book stable the moment it is released. Here the
display plank is SLANTED along its length (13.5–16.5 deg, and WHICH end is downhill
flips per episode) — steeper than every book's topple angle, so NO book can stand alone
on it: the seed's plan (stand a book on the shelf) physically fails, toppling the book
off the open downhill end. The task is a CONSTRUCTION UNDER INSTABILITY with a forced
execution order: first install the heavy charcoal BOOKEND BLOCK on the plank (the only
thing that rests stably on the slope), then stand the four books one at a time in a
single tight row UPSLOPE of it, each new book leaning on the structure built so far
(bookend, then book on book). The finished row is one mutually-supporting lean-to
anchored by the bookend; remove the anchor and the whole row cascades off the shelf
(a smoke negative).

Mechanics: plain rigid-body contact — no joints. Table, back wall and the slanted
plank are kinematic bodies re-posed at reset (tilt angle, tilt SIGN and plank offset
are sampled). The bookend and the four books are dynamic boxes. `post_step` owns the
external-wrench slot of one selectable book (`drive_sel`/`drive_f`, the fingertip
stand-in for the snug press) and advances the rubric latches, gated on the whole
plant being slow (no fly-through credit).

Rubric (graded 0..1, anchored in the demonstrated solve.py trajectory, judged in the
PLANK FRAME so both tilt signs and all sampled angles are one code path):
  - 0.25 * but_latch : bookend resting on the plank (in bounds, aligned with the
    plank normal, at rest height), latched while the plant is slow;
  - 0.15 * k_latch   : k = length of the contiguous chain of standing books
    (upright within `upright_max_deg` of the PLANK NORMAL, on the plank, at rest
    height) growing upslope from the bookend with face gaps < `chain_gap` —
    latched max, gated on the plant being slow;
  - success() (=> score exactly 1.0): bookend ok AND all four books ok AND the
    chain spans all four AND everything settled — live physical state. Staged sum
    caps at 0.85; null policy ~0; latched credit never evaporates.

Per-episode randomization (readback-verified in smoke): tilt sign (which end is
downhill), tilt magnitude, plank y-offset, the table slot permutation of the four
books + per-book jitter/yaw, and the bookend's start pose on the table.

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
class TiltShelfButtressSceneCfg(BaseCfg):
    """Config for `TiltShelfButtressScene`. Geometry is derived once in `__post_init__` so the
    scene, the smoke AND the solver read the same numbers."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    upright_max_deg: float = tunable(12.0)  # book height-axis within this of the PLANK NORMAL
    but_align_max_deg: float = tunable(10.0)  # bookend height-axis within this of the normal
    chain_gap: float = tunable(0.015)  # max face-to-face slope gap inside the row (m)
    edge_margin: float = tunable(0.020)  # on-plank bounds inset from the plank edges (m)
    rest_z_tol: float = tunable(0.012)  # body rest height vs plank surface (m)
    settle_speed: float = tunable(0.05)  # max |lin vel| on every dynamic body when judging
    settle_ang: float = tunable(0.50)  # max |ang vel| on every dynamic body when judging
    latch_speed: float = tunable(0.08)  # latch gate: plant counts only while this slow

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    tilt_min_deg: float = tunable(13.5)  # sampled tilt magnitude range; min stays >= 2.9 deg
    tilt_max_deg: float = tunable(16.5)  # above the steepest book topple angle (10.6 deg)
    plank_y_jitter: float = tunable(0.05)  # uniform +- y shift of the plank (m)
    slot_jitter: float = tunable(0.02)  # uniform +- xy jitter of table start slots (m)
    slot_yaw_deg: float = tunable(30.0)  # uniform +- yaw of books at their table slots
    but_y_range: float = tunable(0.30)  # bookend table start y ~ U(-this, +this)

    # --- tunable: plant ----------------------------------------------------------------------
    book_density: float = tunable(500.0)  # kg/m^3 -> masses 0.21..0.23 kg
    but_mass: float = tunable(1.2)  # heavy bookend: friction anchor of the whole row
    friction: float = tunable(0.6)  # static friction everywhere (dyn = 0.9x)

    # --- info: structure (env-local coordinates; shelf slope runs along y) -------------------
    table_center: tuple = info((0.05, 0.0, 0.36))
    table_size: tuple = info((1.00, 1.10, 0.08))  # top at z0 = 0.40
    wall_x: float = info(-0.31)
    wall_size: tuple = info((0.02, 0.90, 0.50))
    plank_center: tuple = info((-0.18, 0.0, 0.53))  # y replaced by the sampled offset
    plank_size: tuple = info((0.16, 0.42, 0.02))  # slope runs along y (length 0.42)
    n_books: int = info(4)
    book_d: float = info(0.11)  # depth (x), all books
    book_t: tuple = info((0.024, 0.028, 0.022, 0.026))  # thickness (slope axis) per book
    book_h: tuple = info((0.165, 0.150, 0.175, 0.160))  # height per book
    book_colors: tuple = info((
        ("red", (0.78, 0.10, 0.10)),
        ("blue", (0.12, 0.22, 0.78)),
        ("green", (0.10, 0.55, 0.15)),
        ("yellow", (0.85, 0.75, 0.10)),
    ))
    but_size: tuple = info((0.09, 0.055, 0.085))  # bookend block (x, slope, height)
    book_slot_x: float = info(0.10)  # table start row of books
    book_slot_y: tuple = info((-0.24, -0.08, 0.08, 0.24))
    but_slot_x: float = info(0.30)  # bookend table start x
    contact_offset: float = info(0.001)  # < placement clearance — no phantom row contact

    # Derived (filled in __post_init__).
    z0: float = field(default=None, init=False)  # table top
    book_mass: tuple = field(default=None, init=False)
    topple_deg: tuple = field(default=None, init=False)  # atan(t/h) per book

    def __post_init__(self) -> None:
        self.z0 = self.table_center[2] + self.table_size[2] / 2
        self.book_mass = tuple(self.book_density * self.book_d * t * h
                               for t, h in zip(self.book_t, self.book_h))
        self.topple_deg = tuple(math.degrees(math.atan2(t, h))
                                for t, h in zip(self.book_t, self.book_h))
        assert self.tilt_min_deg >= max(self.topple_deg) + 2.5, \
            "the tilt must defeat every book's topple angle with margin"


# ----- scene -----------------------------------------------------------------------------------
class TiltShelfButtressScene(BaseScene):
    cfg: TiltShelfButtressSceneCfg

    def __init__(self, cfg: TiltShelfButtressSceneCfg | None = None) -> None:
        super().__init__(cfg or TiltShelfButtressSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, kinematic table + back wall + slanted plank, the dynamic bookend
        block and four dynamic book boxes (reset() re-places everything; nominal poses here
        use a +tilt_min plank and zero jitter)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        wood = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.32, 0.18))
        wall_col = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.72, 0.66))
        plank_col = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.20, 0.11))
        charcoal = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.12, 0.14))
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.friction, dynamic_friction=0.9 * c.friction, restitution=0.0)
        dyn = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
            linear_damping=0.05,
            angular_damping=0.10,
            sleep_threshold=0.0,  # external wrenches + latch gating need live velocities
            stabilization_threshold=0.0,
        )

        phi0 = math.radians(c.tilt_min_deg)
        q0 = (math.cos(phi0 / 2), math.sin(phi0 / 2), 0.0, 0.0)
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
            "table": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                spawn=sim_utils.CuboidCfg(size=c.table_size, rigid_props=kin,
                                          collision_props=coll, physics_material=mat,
                                          visual_material=wood),
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.table_center),
            ),
            "wall": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Wall",
                spawn=sim_utils.CuboidCfg(size=c.wall_size, rigid_props=kin,
                                          collision_props=coll, physics_material=mat,
                                          visual_material=wall_col),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.wall_x, 0.0, c.z0 + c.wall_size[2] / 2)),
            ),
            "plank": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plank",
                spawn=sim_utils.CuboidCfg(size=c.plank_size, rigid_props=kin,
                                          collision_props=coll, physics_material=mat,
                                          visual_material=plank_col),
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.plank_center, rot=q0),
            ),
            "buttress": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Buttress",
                spawn=sim_utils.CuboidCfg(
                    size=c.but_size, rigid_props=dyn,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.but_mass),
                    collision_props=coll, physics_material=mat, visual_material=charcoal),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.but_slot_x, 0.0, c.z0 + c.but_size[2] / 2 + 0.002)),
            ),
        }
        for b in range(c.n_books):
            t, h = c.book_t[b], c.book_h[b]
            _cname, rgb = c.book_colors[b]
            out[f"book{b}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + f"Book{b}",
                spawn=sim_utils.CuboidCfg(
                    size=(c.book_d, t, h), rigid_props=dyn,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.book_mass[b]),
                    collision_props=coll, physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.book_slot_x, c.book_slot_y[b], c.z0 + h / 2 + 0.002)),
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
                "enable_external_forces_every_iteration": True,  # wrench presses under-apply otherwise
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        n = env.num_envs
        dev = env.device
        self.books: list[RigidObject] = [env.iscene[f"book{b}"] for b in range(c.n_books)]
        self.buttress: RigidObject = env.iscene["buttress"]
        self.kin_parts: dict[str, RigidObject] = {
            nm: env.iscene[nm] for nm in ("table", "wall", "plank")}
        self.env_origins = env.iscene.env_origins
        self._t = torch.tensor(c.book_t, device=dev)
        self._h = torch.tensor(c.book_h, device=dev)
        # Episode state (sampled at reset).
        self.tilt_sign = torch.ones(n, device=dev)  # +1: downhill = -y; -1: downhill = +y
        self.tilt = torch.zeros(n, device=dev)  # tilt magnitude (rad)
        self.phi = torch.zeros(n, device=dev)  # signed plank roll about x = sign * tilt
        self.plank_pos = torch.zeros(n, 3, device=dev)  # env-local plank center
        self.plank_quat = torch.zeros(n, 4, device=dev)
        self.book_start = torch.zeros(n, c.n_books, 2, device=dev)  # table xy per book
        self.but_start = torch.zeros(n, 2, device=dev)
        self.but_latch = torch.zeros(n, device=dev)
        self.k_latch = torch.zeros(n, device=dev)
        # External drive input on ONE selectable book (world-frame force at the CoM; solve.py
        # writes; post_step consumes + owns the wrench slots).
        self.drive_sel = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.drive_f = torch.zeros(n, 3, device=dev)

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample tilt sign + magnitude + plank offset, the table slot
        permutation of the books (+ jitter, yaw) and the bookend's table pose; write the
        kinematic parts and the dynamic bodies. Uses torch.rand throughout (the first
        randint after manual_seed is degenerate on this stack)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        sign = torch.where(torch.rand(m, device=dev) < 0.5,
                           -torch.ones(m, device=dev), torch.ones(m, device=dev))
        tilt = math.radians(c.tilt_min_deg) + torch.rand(m, device=dev) \
            * math.radians(c.tilt_max_deg - c.tilt_min_deg)
        phi = sign * tilt
        plank_y = (torch.rand(m, device=dev) * 2 - 1) * c.plank_y_jitter
        perm = torch.rand(m, c.n_books, device=dev).argsort(dim=1)  # book -> table slot
        slot_y = torch.tensor(c.book_slot_y, device=dev)

        self.tilt_sign[env_ids] = sign
        self.tilt[env_ids] = tilt
        self.phi[env_ids] = phi
        self.plank_pos[env_ids, 0] = c.plank_center[0]
        self.plank_pos[env_ids, 1] = plank_y
        self.plank_pos[env_ids, 2] = c.plank_center[2]
        pq = torch.zeros(m, 4, device=dev)
        pq[:, 0] = torch.cos(phi / 2)
        pq[:, 1] = torch.sin(phi / 2)
        self.plank_quat[env_ids] = pq
        self.but_latch[env_ids] = 0.0
        self.k_latch[env_ids] = 0.0
        self.drive_sel[env_ids] = -1
        self.drive_f[env_ids] = 0.0

        def write(body: RigidObject, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        qid = torch.zeros(m, 4, device=dev)
        qid[:, 0] = 1.0
        write(self.kin_parts["table"],
              torch.tensor(c.table_center, device=dev).expand(m, 3), qid)
        write(self.kin_parts["wall"],
              torch.tensor([c.wall_x, 0.0, c.z0 + c.wall_size[2] / 2],
                           device=dev).expand(m, 3), qid)
        write(self.kin_parts["plank"], self.plank_pos[env_ids], pq)

        # bookend on the table (yaw jitter only)
        bx = c.but_slot_x + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
        by = (torch.rand(m, device=dev) * 2 - 1) * c.but_y_range
        self.but_start[env_ids, 0] = bx
        self.but_start[env_ids, 1] = by
        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(20.0) / 2
        bq = torch.zeros(m, 4, device=dev)
        bq[:, 0] = torch.cos(half)
        bq[:, 3] = torch.sin(half)
        bpos = torch.stack([bx, by, torch.full((m,), c.z0 + c.but_size[2] / 2 + 0.002,
                                               device=dev)], dim=1)
        write(self.buttress, bpos, bq)

        # books upright at their permuted table slots (+ jitter, yaw)
        for b in range(c.n_books):
            sy = slot_y[perm[:, b]]
            px = c.book_slot_x + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
            py = sy + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
            self.book_start[env_ids, b, 0] = px
            self.book_start[env_ids, b, 1] = py
            half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.slot_yaw_deg) / 2
            q = torch.zeros(m, 4, device=dev)
            q[:, 0] = torch.cos(half)
            q[:, 3] = torch.sin(half)
            pos = torch.stack([px, py, torch.full((m,), c.z0 + c.book_h[b] / 2 + 0.002,
                                                  device=dev)], dim=1)
            write(self.books[b], pos, q)

    # ----- readings -----------------------------------------------------------------------------
    def _body_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(pos env-local (N,5,3), quat (N,5,4), |lin vel| (N,5), |ang vel| (N,5)) for the
        four books then the bookend."""
        bodies = self.books + [self.buttress]
        pos = torch.stack([b.data.root_pos_w for b in bodies], dim=1) \
            - self.env_origins[:, None, :]
        quat = torch.stack([b.data.root_quat_w for b in bodies], dim=1)
        lv = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in bodies], dim=1)
        av = torch.stack([b.data.root_ang_vel_w.norm(dim=-1) for b in bodies], dim=1)
        return pos, quat, lv, av

    def _plank_local(self, pos: torch.Tensor) -> torch.Tensor:
        """(N,K,3) env-local positions -> plank-frame coordinates (x depth, y slope, z normal)."""
        from isaaclab.utils.math import quat_apply_inverse

        n, k = pos.shape[0], pos.shape[1]
        rel = pos - self.plank_pos[:, None, :]
        pq = self.plank_quat[:, None, :].expand(n, k, 4).reshape(n * k, 4)
        return quat_apply_inverse(pq, rel.reshape(n * k, 3)).reshape(n, k, 3)

    def _axes(self, quat: torch.Tensor, axis: tuple) -> torch.Tensor:
        """(N,K,3) world direction of a body axis."""
        from isaaclab.utils.math import quat_apply

        n, k = quat.shape[0], quat.shape[1]
        v = torch.tensor(axis, device=quat.device, dtype=quat.dtype).expand(n * k, 3)
        return quat_apply(quat.reshape(n * k, 4), v).reshape(n, k, 3)

    def normal_w(self) -> torch.Tensor:
        """(N,3) world plank surface normal."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device) \
            .expand(self.env.num_envs, 3)
        return quat_apply(self.plank_quat, ez)

    def _ok_parts(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(books_ok (N,4), but_ok (N,), u_books (N,4), u_but (N,)) — the plank-frame
        standing predicates and UPHILL slope coordinates (uphill positive)."""
        c = self.cfg
        pos, quat, _lv, _av = self._body_tensors()
        loc = self._plank_local(pos)  # (N,5,3)
        up = self._axes(quat, (0.0, 0.0, 1.0))  # body height axes, world
        nrm = self.normal_w()[:, None, :]  # (N,1,3)
        align = (up * nrm).sum(-1)  # cos(angle to plank normal)

        in_x = loc[:, :, 0].abs() < (c.plank_size[0] / 2 - c.edge_margin)
        in_y = loc[:, :, 1].abs() < (c.plank_size[1] / 2 - c.edge_margin)
        rest_z = torch.cat([c.plank_size[2] / 2 + self._h[None, :].expand(pos.shape[0], -1) / 2,
                            torch.full((pos.shape[0], 1), c.plank_size[2] / 2
                                       + c.but_size[2] / 2, device=pos.device)], dim=1)
        at_z = (loc[:, :, 2] - rest_z).abs() < c.rest_z_tol

        books_ok = (align[:, :4] >= math.cos(math.radians(c.upright_max_deg))) \
            & in_x[:, :4] & in_y[:, :4] & at_z[:, :4]
        but_ok = (align[:, 4] >= math.cos(math.radians(c.but_align_max_deg))) \
            & in_x[:, 4] & in_y[:, 4] & at_z[:, 4]

        u = self.tilt_sign[:, None] * loc[:, :, 1]  # uphill slope coordinate
        return books_ok, but_ok, u[:, :4], u[:, 4]

    def chain_count(self) -> torch.Tensor:
        """(N,) float 0..4: length of the contiguous chain of standing books growing UPSLOPE
        from the bookend — each link a books_ok book whose downslope face is within
        `chain_gap` of the previous face. Kills the no-buttress and A-frame cheats: credit
        exists only for structure actually anchored on the bookend."""
        c = self.cfg
        books_ok, but_ok, u, ub = self._ok_parts()
        ok = books_ok & (u > ub)  # must sit upslope of the bookend center
        uu = torch.where(ok, u, torch.full_like(u, 1e6))
        order = uu.argsort(dim=1)
        us = uu.gather(1, order)
        ts = self._t[None, :].expand_as(uu).gather(1, order)

        prev_face = ub + c.but_size[1] / 2
        alive = but_ok
        count = torch.zeros_like(ub)
        for k in range(c.n_books):
            gap = us[:, k] - ts[:, k] / 2 - prev_face
            link = alive & (us[:, k] < 1e5) & (gap < c.chain_gap) & (gap > -0.025)
            count = count + link.float()
            prev_face = torch.where(link, us[:, k] + ts[:, k] / 2, prev_face)
            alive = link
        return count

    def buttress_ok(self) -> torch.Tensor:
        """(N,) bool: bookend resting on the plank, aligned with the plank normal."""
        _bo, but_ok, _u, _ub = self._ok_parts()
        return but_ok

    def books_ok(self) -> torch.Tensor:
        """(N,4) bool: per-book standing-on-plank predicate (plank frame)."""
        books_ok, _b, _u, _ub = self._ok_parts()
        return books_ok

    def settled(self) -> torch.Tensor:
        """(N,) bool: every dynamic body slow (lin and ang)."""
        c = self.cfg
        _p, _q, lv, av = self._body_tensors()
        return (lv < c.settle_speed).all(dim=1) & (av < c.settle_ang).all(dim=1)

    def success(self) -> torch.Tensor:
        """(N,) bool: bookend on the plank, ALL four books standing in one contiguous row
        upslope of it, everything settled (physical, live state)."""
        books_ok, but_ok, _u, _ub = self._ok_parts()
        return but_ok & books_ok.all(dim=1) \
            & (self.chain_count() >= self.cfg.n_books - 0.5) & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.25 * but_latch + 0.15 * k_latch (latched max chain length,
        slow-gated), capped at 0.85; exactly 1.0 iff success(). Null policy ~0; latched
        credit never evaporates."""
        staged = (0.25 * self.but_latch + 0.15 * self.k_latch).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(staged), staged)

    # ----- step-coupled mechanics (every substep) ------------------------------------------------
    def post_step(self) -> None:
        """Apply the world-frame `drive_f` CoM force to the `drive_sel` book (converted to
        the body frame with the FRESH quat — this stack applies wrenches in the body's
        current frame), then advance the latches, gated on the whole plant being slow."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        zt = torch.zeros(n, 1, 3, device=dev)
        for b in range(c.n_books):
            sel = self.drive_sel == b
            f = torch.zeros(n, 1, 3, device=dev)
            if sel.any():
                q = self.books[b].data.root_quat_w[sel]
                f[sel, 0, :] = quat_apply_inverse(q, self.drive_f[sel])
            self.books[b].set_external_force_and_torque(f, zt)

        _p, _q, lv, _av = self._body_tensors()
        slow = (lv < c.latch_speed).all(dim=1)
        but_ok = self.buttress_ok()
        self.but_latch = torch.maximum(self.but_latch,
                                       torch.nan_to_num((but_ok & slow).float()))
        k = torch.nan_to_num(self.chain_count() * slow.float())
        self.k_latch = torch.maximum(self.k_latch, k)

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "books": [b.data.root_state_w[env_ids].clone() for b in self.books],
            "buttress": self.buttress.data.root_state_w[env_ids].clone(),
            "kin": {nm: b.data.root_state_w[env_ids].clone()
                    for nm, b in self.kin_parts.items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("tilt_sign", "tilt", "phi", "plank_pos", "plank_quat",
                               "book_start", "but_start", "but_latch", "k_latch",
                               "drive_sel", "drive_f")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for b, st in zip(self.books, state["books"]):
            b.write_root_state_to_sim(st, env_ids)
        self.buttress.write_root_state_to_sim(state["buttress"], env_ids)
        for nm, st in state["kin"].items():
            self.kin_parts[nm].write_root_state_to_sim(st, env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        inv = ", ".join(f"a {nm} one ({t * 1000:.0f} mm thick, {h * 1000:.0f} mm tall)"
                        for (nm, _rgb), t, h in zip(c.book_colors, c.book_t, c.book_h))
        return (
            f"A wooden display shelf is mounted on the wall at the back of a table: a single "
            f"plank {c.plank_size[1] * 100:.0f} cm long and {c.plank_size[0] * 100:.0f} cm "
            f"deep, VISIBLY SLANTED along its length by {c.tilt_min_deg:.0f}-"
            f"{c.tilt_max_deg:.0f} degrees. Which end is the low end varies per episode — "
            f"look at the plank. Both ends are open: anything that topples slides off. "
            f"Standing upright on the table are four hardcover books: {inv}. A heavy dark "
            f"charcoal bookend block ({c.but_size[1] * 1000:.0f} mm wide along the shelf, "
            f"{c.but_size[2] * 1000:.0f} mm tall, {c.but_mass:.1f} kg) also sits on the "
            f"table.\n"
            f"Goal: shelve all four books standing upright on the slanted plank in ONE tight "
            f"row, leaning against the bookend. The slant is steeper than any book's tipping "
            f"angle, so NO book can stand on the plank alone — a book simply stood on the "
            f"bare plank topples and slides off the low end. The only build that works, in "
            f"this order: (1) place the bookend flat on the plank near its LOW end (the "
            f"block is heavy and squat — it alone rests stably on the slope); (2) stand the "
            f"books one at a time on the UPHILL side of the bookend, each new book placed "
            f"directly against the structure so it leans on it (first book on the bookend, "
            f"each next book on the previous one); nudge the row snug so every face gap is "
            f"under ~{c.chain_gap * 1000:.0f} mm. Success is judged on the settled scene: "
            f"bookend resting flat on the plank, all four books standing within "
            f"{c.upright_max_deg:.0f} degrees of the plank's surface normal in a contiguous "
            f"row that starts at the bookend and runs uphill, nothing moving. Books lying "
            f"flat, standing off the row, or fallen off the plank do not count."
        )

    def instruction(self) -> str:
        return (
            "Place the heavy charcoal bookend flat on the slanted shelf near the shelf's "
            "low end, then stand all four books on the shelf in one tight row on the "
            "uphill side of the bookend, each book leaning against the row. The shelf is "
            "too steep for a book to stand alone: any book left unsupported, lying flat, "
            "or off the shelf fails the task."
        )


# Guarded registration: the forge may import this module under two names.
if "tilt_shelf_buttress" not in SCENES.list():
    SCENES.register("tilt_shelf_buttress", TiltShelfButtressScene)
if "simgen.tilt_shelf_buttress" not in ENVS.list():
    register_env("simgen", lambda: EnvCfg(scene="tilt_shelf_buttress", robot="null"))

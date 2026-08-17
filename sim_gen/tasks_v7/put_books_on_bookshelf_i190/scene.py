"""LibrarianExtractScene — pull ONE requested book out of a packed shelf, lay it in the tray.

Derived from the RLBench `put_books_on_bookshelf` seed but STRATEGICALLY INVERTED (see
TASK.md): the seed transports loose books from the table ONTO an open shelf — additive
pick-and-place with no selection problem and no constraint on anything else. Here the
books already fill the shelf: five hardcovers of different colors/sizes stand in a
tight row inside a one-slot case (side gaps ~3 mm — no room to pinch a spine; the side
cheeks run all the way to the ceiling — no way to lift a book clear of the case). An ORANGE REQUEST CARD lies
on the table directly in front of ONE slot; the book above it is the requested one,
sampled fresh every episode along with the whole row order. The task is a SUBTRACTIVE,
selective extraction under a do-not-disturb obligation: tip the requested book's top
edge forward so it pivots on its bottom edge out of the row (the librarian's move —
the only contact the geometry admits), grip its exposed covers, and lay it FLAT,
covers down, inside the open tray on the table. Every other book must remain standing
upright in its own slot. Execution order is forced by geometry: tip first, then
extract, then place.

Mechanics: plain rigid-body contact — no joints. The case (floor / back / ceiling /
two cheeks), the tray (floor + 4 walls) and the request card are kinematic bodies
re-posed at reset (row center and tray position jitter). The books are dynamic boxes.
`post_step` owns the target book's wrench slot: it applies the `drive_f`/`drive_t`
buffers (world-frame; converted to the body frame with the FRESH quat every substep —
the fingertip stand-in for the tip maneuver) and advances the rubric latches.

Rubric (graded 0..1, anchored in the demonstrated solve.py trajectory):
  - 0.35 * tip_latch: best forward-pivot progress clamp(pitch/30deg) of the target
    book, latched only while it is still IN its slot and every other book is
    undisturbed (a book tilted after being carried out latches nothing);
  - 0.25 * out_latch: target body carried clear of the shelf while the row stands;
  - 0.30 * tray_now: live "flat inside the tray" predicate;
  - success() (=> score 1.0 exactly): target settled flat in the tray AND all four
    other books upright in their own slots, everything settled. Null policy ~0;
    latched credit does not evaporate under correct behavior.

Per-episode randomization (readback-verified in smoke): the row permutation of the
five books, the requested-book index (card physically re-posed under that slot), the
row center y, and the tray y position.

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
class LibrarianExtractSceneCfg(BaseCfg):
    """Config for `LibrarianExtractScene`. Geometry is derived once in `__post_init__` so the
    scene, the smoke AND the solver read the same numbers."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    tip_grab_deg: float = tunable(30.0)  # forward pivot that fully latches the tip stage
    upright_max_deg: float = tunable(12.0)  # a non-target book must stay within this of vertical
    slot_y_tol: float = tunable(0.020)  # ...and within this of its own slot center (m)
    flat_max_deg: float = tunable(15.0)  # target cover-normal within this of vertical = flat
    tray_xy_tol: float = tunable(0.055)  # target center within this of the tray center (m)
    tray_z_tol: float = tunable(0.010)  # target center height vs floor+t/2 (m)
    settle_speed: float = tunable(0.05)  # max |lin vel| on every book when judging (m/s)
    settle_ang: float = tunable(0.50)  # max target |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    row_c_jitter: float = tunable(0.03)  # uniform +- y shift of the whole case (m)
    tray_y_jitter: float = tunable(0.08)  # uniform +- y shift of the tray (m)

    # --- tunable: plant ----------------------------------------------------------------------
    book_density: float = tunable(500.0)  # kg/m^3 -> masses 0.25..0.48 kg
    friction: float = tunable(0.6)  # book/shelf/tray static friction (dyn = 0.9x)

    # --- info: structure (env-local coordinates; shelf front faces +x) -----------------------
    table_center: tuple = info((0.05, 0.0, 0.36))
    table_size: tuple = info((1.00, 1.10, 0.08))  # top at z0 = 0.40
    n_books: int = info(5)
    book_d: float = info(0.13)  # depth (x), all books
    book_t: tuple = info((0.040, 0.034, 0.030, 0.026, 0.022))  # thickness (y) per book
    book_h: tuple = info((0.185, 0.170, 0.190, 0.160, 0.178))  # height (z) per book
    book_colors: tuple = info((
        ("crimson", (0.75, 0.08, 0.08)),
        ("blue", (0.10, 0.20, 0.75)),
        ("green", (0.10, 0.55, 0.15)),
        ("yellow", (0.85, 0.75, 0.10)),
        ("violet", (0.45, 0.12, 0.60)),
    ))
    gap: float = info(0.003)  # side gap between neighboring books
    side_slack: float = info(0.002)  # row-end slack against each cheek
    shelf_x: float = info(-0.205)  # case center x (boards)
    shelf_depth: float = info(0.15)  # boards x-size: x in [-0.28, -0.13]
    shelf_front_x: float = info(-0.13)
    book_x: float = info(-0.21)  # book center x (front face at -0.145)
    shelf_floor_top: float = info(0.42)  # zs — books stand here
    ceil_clear: float = info(0.24)  # ceiling bottom = zs + this (0.66); tallest book's
    # top-rear corner rises h*cos(th)+d*sin(th)-h ~ 40 mm during the 33 deg tip sweep —
    # 0.24 leaves >= 10 mm of ceiling margin at full tip (0.22 would jam at ~17 deg).
    board_t: float = info(0.02)  # floor board thickness
    ceil_t: float = info(0.015)
    back_t: float = info(0.015)
    cheek_t: float = info(0.015)
    tray_x: float = info(0.16)
    tray_floor_size: tuple = info((0.24, 0.30, 0.012))  # top at 0.412
    tray_wall_h: float = info(0.03)
    tray_wall_t: float = info(0.01)
    card_size: tuple = info((0.055, 0.035, 0.004))
    card_x: float = info(-0.085)
    contact_offset: float = info(0.001)  # < gap/2 per face — no phantom neighbor contact

    # Derived (filled in __post_init__).
    z0: float = field(default=None, init=False)  # table top
    zs: float = field(default=None, init=False)  # shelf floor top
    zc: float = field(default=None, init=False)  # ceiling bottom
    inner_span: float = field(default=None, init=False)  # cheek-to-cheek clear width
    tray_floor_top: float = field(default=None, init=False)
    tray_inner: tuple = field(default=None, init=False)  # (x, y) clear inner spans
    book_mass: tuple = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.z0 = self.table_center[2] + self.table_size[2] / 2
        self.zs = self.shelf_floor_top
        self.zc = self.zs + self.ceil_clear
        self.inner_span = (sum(self.book_t) + (self.n_books - 1) * self.gap
                           + 2 * self.side_slack)
        self.tray_floor_top = self.z0 + self.tray_floor_size[2]
        self.tray_inner = (self.tray_floor_size[0] - 2 * self.tray_wall_t,
                           self.tray_floor_size[1] - 2 * self.tray_wall_t)
        self.book_mass = tuple(self.book_density * self.book_d * t * h
                               for t, h in zip(self.book_t, self.book_h))


# ----- scene -----------------------------------------------------------------------------------
class LibrarianExtractScene(BaseScene):
    cfg: LibrarianExtractSceneCfg

    def __init__(self, cfg: LibrarianExtractSceneCfg | None = None) -> None:
        super().__init__(cfg or LibrarianExtractSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, kinematic table + case boards + tray + request card, five dynamic
        book boxes (reset() re-places everything; nominal poses here use zero jitter)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        wood = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.32, 0.18))
        case_col = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32, 0.22, 0.12))
        tray_col = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.58))
        orange = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.45, 0.05))
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.friction, dynamic_friction=0.9 * c.friction, restitution=0.0)

        def kin_box(size: tuple, pos: tuple, color: Any) -> RigidObjectCfg:
            return RigidObjectCfg(
                prim_path=None,  # filled by caller
                spawn=sim_utils.CuboidCfg(size=size, rigid_props=kin, collision_props=coll,
                                          physics_material=mat, visual_material=color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
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

        cheek_y = c.inner_span / 2 + c.cheek_t / 2
        parts = {
            "table": (c.table_size, c.table_center, wood),
            "case_floor": ((c.shelf_depth, 0.21, c.board_t),
                           (c.shelf_x, 0.0, c.zs - c.board_t / 2), case_col),
            "case_ceil": ((c.shelf_depth, 0.21, c.ceil_t),
                          (c.shelf_x, 0.0, c.zc + c.ceil_t / 2), case_col),
            "case_back": ((c.back_t, 0.21, c.zc + c.ceil_t + 0.015 - c.z0),
                          (c.shelf_x - c.shelf_depth / 2 - c.back_t / 2, 0.0,
                           (c.zc + c.ceil_t + 0.015 + c.z0) / 2), case_col),
            "case_cheek_l": ((c.shelf_depth, c.cheek_t, c.ceil_clear),
                             (c.shelf_x, -cheek_y, (c.zs + c.zc) / 2), case_col),
            "case_cheek_r": ((c.shelf_depth, c.cheek_t, c.ceil_clear),
                             (c.shelf_x, cheek_y, (c.zs + c.zc) / 2), case_col),
            "tray_floor": (c.tray_floor_size,
                           (c.tray_x, 0.0, c.z0 + c.tray_floor_size[2] / 2), tray_col),
            "tray_wall_xn": ((c.tray_wall_t, c.tray_floor_size[1], c.tray_wall_h),
                             (c.tray_x - c.tray_floor_size[0] / 2 + c.tray_wall_t / 2, 0.0,
                              c.tray_floor_top + c.tray_wall_h / 2), tray_col),
            "tray_wall_xp": ((c.tray_wall_t, c.tray_floor_size[1], c.tray_wall_h),
                             (c.tray_x + c.tray_floor_size[0] / 2 - c.tray_wall_t / 2, 0.0,
                              c.tray_floor_top + c.tray_wall_h / 2), tray_col),
            "tray_wall_yn": ((c.tray_floor_size[0] - 2 * c.tray_wall_t, c.tray_wall_t,
                              c.tray_wall_h),
                             (c.tray_x, -c.tray_floor_size[1] / 2 + c.tray_wall_t / 2,
                              c.tray_floor_top + c.tray_wall_h / 2), tray_col),
            "tray_wall_yp": ((c.tray_floor_size[0] - 2 * c.tray_wall_t, c.tray_wall_t,
                              c.tray_wall_h),
                             (c.tray_x, c.tray_floor_size[1] / 2 - c.tray_wall_t / 2,
                              c.tray_floor_top + c.tray_wall_h / 2), tray_col),
            "card": (c.card_size, (c.card_x, 0.0, c.z0 + c.card_size[2] / 2), orange),
        }
        for name, (size, pos, color) in parts.items():
            cfg = kin_box(size, pos, color)
            cfg.prim_path = "{ENV_REGEX_NS}/" + name.title().replace("_", "")
            out[name] = cfg

        # --- the five books (dynamic; nominal row order = index order, zero jitter) ---
        y = -c.inner_span / 2 + c.side_slack
        for b in range(c.n_books):
            t, h = c.book_t[b], c.book_h[b]
            _cname, rgb = c.book_colors[b]
            out[f"book{b}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + f"Book{b}",
                spawn=sim_utils.CuboidCfg(
                    size=(c.book_d, t, h),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        linear_damping=0.05,
                        angular_damping=0.10,
                        sleep_threshold=0.0,  # external wrenches must always act
                        stabilization_threshold=0.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.book_mass[b]),
                    collision_props=coll,
                    physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.book_x, y + t / 2, c.zs + h / 2 + 0.001)),
            )
            y += t + c.gap
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "enable_external_forces_every_iteration": True,  # wrench plants under-apply otherwise
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
        self.kin_parts: dict[str, RigidObject] = {
            nm: env.iscene[nm] for nm in
            ("table", "case_floor", "case_ceil", "case_back", "case_cheek_l", "case_cheek_r",
             "tray_floor", "tray_wall_xn", "tray_wall_xp", "tray_wall_yn", "tray_wall_yp",
             "card")}
        self.env_origins = env.iscene.env_origins
        self._t = torch.tensor(c.book_t, device=dev)
        self._h = torch.tensor(c.book_h, device=dev)
        # Episode state.
        self.target_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self.slot_y = torch.zeros(n, c.n_books, device=dev)  # slot center per BOOK INDEX
        self.row_c = torch.zeros(n, device=dev)
        self.tray_y = torch.zeros(n, device=dev)
        self.tip_latch = torch.zeros(n, device=dev)
        self.out_latch = torch.zeros(n, device=dev)
        # External drive input on the TARGET book (world frame; solve.py and smoke probes
        # write; post_step consumes + owns the wrench slots — never call
        # set_external_force_and_torque directly).
        self.drive_f = torch.zeros(n, 3, device=dev)
        self.drive_t = torch.zeros(n, 3, device=dev)

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the row permutation, the requested-book index, the case
        row center and the tray position; write the case boards, tray, card and books.
        Uses torch.rand throughout (the first randint after manual_seed is degenerate on
        this stack)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        perm = torch.rand(m, c.n_books, device=dev).argsort(dim=1)  # row order, left -> right
        tgt = (torch.rand(m, device=dev) * c.n_books).long().clamp(max=c.n_books - 1)
        row_c = (torch.rand(m, device=dev) * 2 - 1) * c.row_c_jitter
        tray_y = (torch.rand(m, device=dev) * 2 - 1) * c.tray_y_jitter

        # slot centers by BOOK INDEX (invert the permutation while accumulating widths)
        slot = torch.zeros(m, c.n_books, device=dev)
        y_cursor = row_c - c.inner_span / 2 + c.side_slack
        for k in range(c.n_books):
            b = perm[:, k]  # book index sitting at row position k
            tk = self._t[b]
            center = y_cursor + tk / 2
            slot.scatter_(1, b.unsqueeze(1), center.unsqueeze(1))
            y_cursor = y_cursor + tk + c.gap

        self.target_idx[env_ids] = tgt
        self.slot_y[env_ids] = slot
        self.row_c[env_ids] = row_c
        self.tray_y[env_ids] = tray_y
        self.tip_latch[env_ids] = 0.0
        self.out_latch[env_ids] = 0.0
        self.drive_f[env_ids] = 0.0
        self.drive_t[env_ids] = 0.0

        def write_kin(name: str, x: float, y: torch.Tensor, z: float) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = x
            st[:, 1] = y
            st[:, 2] = z
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            self.kin_parts[name].write_root_state_to_sim(st, env_ids)

        cheek_y = c.inner_span / 2 + c.cheek_t / 2
        write_kin("table", c.table_center[0], torch.zeros(m, device=dev), c.table_center[2])
        write_kin("case_floor", c.shelf_x, row_c, c.zs - c.board_t / 2)
        write_kin("case_ceil", c.shelf_x, row_c, c.zc + c.ceil_t / 2)
        write_kin("case_back", c.shelf_x - c.shelf_depth / 2 - c.back_t / 2, row_c,
                  (c.zc + c.ceil_t + 0.015 + c.z0) / 2)
        write_kin("case_cheek_l", c.shelf_x, row_c - cheek_y, (c.zs + c.zc) / 2)
        write_kin("case_cheek_r", c.shelf_x, row_c + cheek_y, (c.zs + c.zc) / 2)
        write_kin("tray_floor", c.tray_x, tray_y, c.z0 + c.tray_floor_size[2] / 2)
        wz = c.tray_floor_top + c.tray_wall_h / 2
        wx = c.tray_floor_size[0] / 2 - c.tray_wall_t / 2
        wy = c.tray_floor_size[1] / 2 - c.tray_wall_t / 2
        write_kin("tray_wall_xn", c.tray_x - wx, tray_y, wz)
        write_kin("tray_wall_xp", c.tray_x + wx, tray_y, wz)
        write_kin("tray_wall_yn", c.tray_x, tray_y - wy, wz)
        write_kin("tray_wall_yp", c.tray_x, tray_y + wy, wz)
        card_y = slot.gather(1, tgt.unsqueeze(1)).squeeze(1)
        write_kin("card", c.card_x, card_y, c.z0 + c.card_size[2] / 2)

        for b in range(c.n_books):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = c.book_x
            st[:, 1] = slot[:, b]
            st[:, 2] = c.zs + c.book_h[b] / 2 + 0.001
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            self.books[b].write_root_state_to_sim(st, env_ids)

    # ----- readings -----------------------------------------------------------------------------
    def _book_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(pos_w-origin (N,B,3), quat (N,B,4), |lin vel| (N,B), |ang vel| (N,B))."""
        pos = torch.stack([b.data.root_pos_w for b in self.books], dim=1) \
            - self.env_origins[:, None, :]
        quat = torch.stack([b.data.root_quat_w for b in self.books], dim=1)
        lv = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.books], dim=1)
        av = torch.stack([b.data.root_ang_vel_w.norm(dim=-1) for b in self.books], dim=1)
        return pos, quat, lv, av

    def _gather_target(self, x: torch.Tensor) -> torch.Tensor:
        """Select the target book's row from a (N, B, ...) tensor -> (N, ...)."""
        idx = self.target_idx.view(-1, 1, *([1] * (x.dim() - 2))).expand(-1, 1, *x.shape[2:])
        return x.gather(1, idx).squeeze(1)

    def _axes(self, quat: torch.Tensor, axis: tuple) -> torch.Tensor:
        """(N, B, 3) world direction of a body axis for all books."""
        from isaaclab.utils.math import quat_apply

        n, bk = quat.shape[0], quat.shape[1]
        v = torch.tensor(axis, device=quat.device, dtype=quat.dtype).expand(n * bk, 3)
        return quat_apply(quat.reshape(n * bk, 4), v).reshape(n, bk, 3)

    def pitch_fwd(self) -> torch.Tensor:
        """(N,) forward pivot angle of the TARGET book (rad): angle of its height axis from
        vertical, signed + toward the shelf opening (+x)."""
        _p, quat, _lv, _av = self._book_tensors()
        up = self._axes(quat, (0.0, 0.0, 1.0))
        u = self._gather_target(up)
        return torch.atan2(u[:, 0], u[:, 2].clamp(min=-1.0, max=1.0))

    def others_ok(self) -> torch.Tensor:
        """(N,) bool: every NON-target book upright within `upright_max_deg`, in its own
        slot (y within `slot_y_tol`, x/z at the shelf rest pose)."""
        c = self.cfg
        pos, quat, _lv, _av = self._book_tensors()
        up = self._axes(quat, (0.0, 0.0, 1.0))
        upright = up[:, :, 2] >= math.cos(math.radians(c.upright_max_deg))
        in_slot = (pos[:, :, 1] - self.slot_y).abs() < c.slot_y_tol
        at_x = (pos[:, :, 0] - c.book_x).abs() < 0.03
        at_z = (pos[:, :, 2] - (c.zs + self._h[None, :] / 2)).abs() < 0.02
        ok = upright & in_slot & at_x & at_z
        not_target = torch.ones_like(ok, dtype=torch.bool)
        not_target.scatter_(1, self.target_idx.unsqueeze(1), False)
        return (ok | ~not_target).all(dim=1)

    def tray_ok(self) -> torch.Tensor:
        """(N,) bool: TARGET book flat (cover-normal vertical within `flat_max_deg`) with
        its center inside the tray, resting at floor height."""
        c = self.cfg
        pos, quat, _lv, _av = self._book_tensors()
        p = self._gather_target(pos)
        ey = self._gather_target(self._axes(quat, (0.0, 1.0, 0.0)))
        t_half = self._t[self.target_idx] / 2
        flat = ey[:, 2].abs() >= math.cos(math.radians(c.flat_max_deg))
        in_x = (p[:, 0] - c.tray_x).abs() < c.tray_xy_tol
        in_y = (p[:, 1] - self.tray_y).abs() < c.tray_xy_tol
        at_z = (p[:, 2] - (c.tray_floor_top + t_half)).abs() < c.tray_z_tol
        return flat & in_x & in_y & at_z

    def settled(self) -> torch.Tensor:
        """(N,) bool: every book slow, and the target rotationally quiet."""
        c = self.cfg
        _p, _q, lv, av = self._book_tensors()
        return (lv < c.settle_speed).all(dim=1) & (self._gather_target(av) < c.settle_ang)

    def success(self) -> torch.Tensor:
        """(N,) bool: the requested book lies flat inside the tray, all four other books
        stand upright in their own slots, everything settled (physical, live state)."""
        return self.tray_ok() & self.others_ok() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: staged credit anchored in the demonstrated solution —
        0.35 * tip_latch (pivot progress IN the slot, row undisturbed) + 0.25 * out_latch
        (carried clear, row undisturbed) + 0.30 * live tray placement, capped at 0.90;
        exactly 1.0 iff success(). Null policy ~0; latches never evaporate."""
        staged = (0.35 * self.tip_latch + 0.25 * self.out_latch
                  + 0.30 * self.tray_ok().float()).clamp(max=0.90)
        return torch.where(self.success(), torch.ones_like(staged), staged)

    # ----- step-coupled mechanics (every substep) ------------------------------------------------
    def post_step(self) -> None:
        """Apply the world-frame `drive_f`/`drive_t` buffers to the TARGET book (converted
        to the body frame with the FRESH quat each substep — the default call applies
        wrenches in the body's current frame on this stack), then advance the latches."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        for b in range(c.n_books):
            sel = self.target_idx == b
            f = torch.zeros(n, 1, 3, device=dev)
            t = torch.zeros(n, 1, 3, device=dev)
            if sel.any():
                q = self.books[b].data.root_quat_w[sel]
                f[sel, 0, :] = quat_apply_inverse(q, self.drive_f[sel])
                t[sel, 0, :] = quat_apply_inverse(q, self.drive_t[sel])
            self.books[b].set_external_force_and_torque(f, t)

        pos, _q, _lv, _av = self._book_tensors()
        p = self._gather_target(pos)
        ok = self.others_ok()
        grab = math.radians(c.tip_grab_deg)
        in_slot = (p[:, 0] < c.book_x + 0.072) & (p[:, 2] > c.zs) & (p[:, 2] < c.zc) \
            & ((p[:, 1] - self.slot_y.gather(1, self.target_idx.unsqueeze(1)).squeeze(1)).abs()
               < 0.04)
        prog = (self.pitch_fwd() / grab).clamp(0.0, 1.0) * (ok & in_slot).float()
        prog = torch.nan_to_num(prog, nan=0.0, posinf=0.0, neginf=0.0)
        self.tip_latch = torch.maximum(self.tip_latch, prog)
        out_now = (p[:, 0] > -0.10) & (p[:, 2] > c.z0) & ok
        self.out_latch = torch.maximum(self.out_latch, out_now.float())

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "books": [b.data.root_state_w[env_ids].clone() for b in self.books],
            "kin": {nm: b.data.root_state_w[env_ids].clone()
                    for nm, b in self.kin_parts.items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("target_idx", "slot_y", "row_c", "tray_y", "tip_latch",
                               "out_latch", "drive_f", "drive_t")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for b, st in zip(self.books, state["books"]):
            b.write_root_state_to_sim(st, env_ids)
        for nm, st in state["kin"].items():
            self.kin_parts[nm].write_root_state_to_sim(st, env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        inv = ", ".join(f"{nm} ({t * 1000:.0f} mm thick, {h * 1000:.0f} mm tall)"
                        for (nm, _rgb), t, h in zip(c.book_colors, c.book_t, c.book_h))
        return (
            f"A small one-slot bookcase stands at the back of a table, its open side facing "
            f"you. Inside it, five hardcover books stand upright in one tight row between two "
            f"fixed side cheeks, spines out: {inv}. Their row order is shuffled every episode. "
            f"The clearance between the books' tops and the case ceiling is only "
            f"{c.ceil_clear * 100 - 19:.0f}-{c.ceil_clear * 100 - 16:.0f} cm and the side gaps "
            f"between neighboring books are ~{c.gap * 1000:.0f} mm — far too tight to pinch a "
            f"book's covers in place or to lift it straight up out of the case. On the table "
            f"directly in front of ONE slot lies a small ORANGE REQUEST CARD: the book standing "
            f"straight above that card is the requested book (a different one each episode). An "
            f"open shallow gray tray also sits on the table in front of the case.\n"
            f"Goal: take ONLY the requested book out of the bookcase and lay it FLAT inside the "
            f"tray, covers down. Every other book must remain standing upright in its own slot, "
            f"undisturbed — toppling, extracting or shifting any other book fails the task. The "
            f"geometry forces the librarian's move, in this order: first tip the requested "
            f"book's top edge forward (hook a fingertip over the top of its spine, or press and "
            f"drag it) so the book pivots on its bottom edge and leans out of the row by about "
            f"{c.tip_grab_deg:.0f} degrees; its covers are then exposed beyond the case front, "
            f"so grip them near the top edge, draw the book out, carry it over the tray and lay "
            f"it down flat. Success is judged on the settled scene: requested book flat inside "
            f"the tray, the other four still standing in place."
        )

    def instruction(self) -> str:
        return (
            "Take the book standing directly above the orange request card out of the "
            "bookcase and lay it flat, covers down, inside the gray tray. Do not topple "
            "or displace any of the other books."
        )


# Guarded registration: the forge may import this module under two names.
if "librarian_extract" not in SCENES.list():
    SCENES.register("librarian_extract", LibrarianExtractScene)
if "simgen.librarian_extract" not in ENVS.list():
    register_env("simgen", lambda: EnvCfg(scene="librarian_extract", robot="null"))

"""IkeaTableAssemblyScene — an IKEA-style table with threaded studs and loose legs.

The object world for the flagship assembly task: a tabletop slab carrying four threaded studs at
its corners, and four loose legs (each a threaded nut + a graspable shaft) resting on the slab.
**Goal (carried here, no task layer): screw every leg down onto its stud until it seats.**

Embodiment-agnostic — it knows nothing about who turns the legs. A Franka grips and screws them; a
floating-leg debug robot drives them by force; either way the legs are *scene* objects the robot
reaches through `env.scene`.

Mechanic — auto-weld on seat: a leg threading onto a stud is contact-rich and slow to settle, so the
instant a leg reaches seat depth the scene welds it to the table (a pre-authored, normally-disabled
FixedJoint, just toggled on — no mid-sim prim create/remove). The tabletop is ALWAYS a dynamic body
(a runtime FixedJoint binds only two dynamic bodies). The weld state is reconciled against one
criterion every step and after every reset / set_state — see `_weld_targets` + `_reconcile_welds`.

Lifted from `legacy/four_leg_env.py`. Heavy imports (isaaclab, pxr) are deferred so importing this
module — and registering the scene — stays app-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv

@dataclass
class IkeaTableAssemblySceneCfg(BaseCfg):
    """Config for `IkeaTableAssemblyScene`. Nothing is locked — a variant is just a copy with a few
    fields changed. Nothing is locked — a curriculum/debug variant is just a `.copy()` with a few changed.
    """

    # --- the curriculum / difficulty dials -----------------------------------------------
    # A leg is "seated on a stud" when — all measured in the tabletop's own frame — it is at/below
    # `seat_z` above the slab, within `align_xy` of some stud, and tilted <= `align_axis_deg` off the
    # stud axis (order-independent: any leg may seat on any stud). It then welds once it is also
    # descending slower than `seat_speed` (a debounce so a transient bounce doesn't weld too early).
    seat_z: float = 0.012  # max height above the slab top (m)
    align_xy: float = 0.02  # max horizontal distance (m) from the nearest stud
    align_axis_deg: float = 10.0  # max tilt of the leg's screw axis off the stud axis (deg)
    seat_speed: float = 0.05  # max descent speed (m/s) at the moment of welding
    reset_pos_jitter: float = 0.01  # uniform +/- xy jitter per leg at reset (m); 0 = none

    # --- structure, reset layout, masses, workbench, asset paths (fixed) --------------------
    num_legs: int = 4
    # Four leg slots, inset ~2.5 cm from the edges of the 0.55 m top (matches the baked studs).
    slots: tuple[tuple[float, float], ...] = ((0.25, 0.25), (-0.25, 0.25), (0.25, -0.25), (-0.25, -0.25))
    leg_start_z: float = 0.026  # leg height above the slab top at spawn (nut origin)
    table_thickness: float = 0.05  # furniture slab thickness (its bottom rests on the surface)
    leg_mass: float = 0.12  # informational; the real value is baked into leg.usd
    light_intensity: float = 2500.0
    # Reset layout — the legs' start pose. Default: laid down on their side (90° about x), in a row on
    # ONE side of the centre line (+x), from leg0 (nearest the centre) outward to the last leg — the
    # tabletop sits on the OPPOSITE side (see `table_offset`). Tune the row + lying pose below; a
    # curriculum/robot may set `leg_init_xy` to override the row.
    leg_init_xy: tuple[tuple[float, float], ...] = ()  # per-leg start xy (workbench-rel.); () -> the row below
    leg_row_x0: float = -0.05  # [TUNE] x of leg0; smaller = closer to centre/inside, larger = toward the +x edge
    leg_row_y: float = -0.03  # [TUNE: reach] more negative = toward the robot
    leg_spacing: float = 0.12  # [TUNE: spread] x gap between adjacent legs (leg k at x0 + k*spacing, +x)
    leg_init_z: float = 0.022  # [TUNE: to the asset] leg-origin height above the surface when lying (~radius)
    leg_init_quat: tuple[float, float, float, float] = (2 ** -0.5, 2 ** -0.5, 0.0, 0.0)  # [TUNE] wxyz; 90° about x -> lying
    # Tabletop: slid to the OPPOSITE side of the centre line from the legs (default -x), by `table_offset`
    # (studs move with it; `seated()` is measured in the table frame, so the offset is transparent).
    table_offset: tuple[float, float] = (-0.45, 0.0)  # [TUNE] tabletop xy offset from the workbench centre
    # Workbench: the furniture lies on the vendored Heavy-Duty PackingTable (a static work surface).
    # `surface_z` is the DESIRED working-surface height; the workbench is placed so its top lands there
    # — i.e. sunk below (or raised above) the floor by `surface_z - workbench_height`. Lower it for a
    # shorter robot's reach (Isaac likewise sinks this same table to ~0.7 m for the G1; at 0.994 the
    # bench sits flush on the floor as before). The sunk part clips below the ground — purely cosmetic
    # (the bench is kinematic), exactly as in the Isaac pick-place env.
    surface_z: float = 0.994  # working-surface height (m); the furniture spawns here
    workbench_height: float = 0.994  # the PackingTable's intrinsic top-above-its-base (at workbench_scale)
    workbench_pos: tuple[float, float] = (0.0, 0.0)  # xy the workbench (and furniture) sit at
    workbench_usd: str = ""  # empty -> vendored Heavy-Duty packing table under assets/props/
    workbench_scale: float = 0.01  # the table USD is authored in cm; scale to metres
    # Asset files. Empty -> the packaged ikea_table assets (leg.usd + table.usd, which reference the
    # shared Factory nut/bolt under ../factory/ relatively, so the tree stays relocatable).
    asset_dir: str = ""
    leg_usd: str = ""
    table_usd: str = ""

    def __post_init__(self) -> None:
        if not self.leg_init_xy:  # default: legs lying in a row, leg0 at the centre, going outward (+x)
            self.leg_init_xy = tuple((self.leg_row_x0 + k * self.leg_spacing, self.leg_row_y) for k in range(self.num_legs))
        assets = Path(__file__).resolve().parents[1] / "assets"
        self.asset_dir = self.asset_dir or str(assets / "ikea_table")
        self.leg_usd = self.leg_usd or str(Path(self.asset_dir) / "leg.usd")
        self.table_usd = self.table_usd or str(Path(self.asset_dir) / "table.usd")
        self.workbench_usd = self.workbench_usd or str(
            assets / "props" / "packing_table" / "SM_HeavyDutyPackingTable_C02_01_physics.usd"
        )


@SCENES.register("ikea_table")
class IkeaTableAssemblyScene(BaseScene):
    cfg: IkeaTableAssemblySceneCfg

    def __init__(self, cfg: IkeaTableAssemblySceneCfg | None = None) -> None:
        super().__init__(cfg or IkeaTableAssemblySceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Floor, a dome light, the raised workbench, the free tabletop (slab + four SDF studs as one
        body), and `num_legs` loose legs at the slots. The furniture spawns on the workbench surface
        (slab bottom on the table top)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        wx, wy = c.workbench_pos
        tx, ty = wx + c.table_offset[0], wy + c.table_offset[1]  # tabletop centre (slid off the workbench centre)
        wb_z = c.surface_z - c.workbench_height  # place the bench so its top lands at surface_z (sinks if low)
        slab_top = c.surface_z + c.table_thickness  # the tabletop body's origin sits at the slab's top

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(usd_path=str(
                    Path(__file__).resolve().parents[1] / "assets" / "props" / "ground" / "default_ground.usd")),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=c.light_intensity, color=(0.9, 0.9, 0.9)),
            ),
            "workbench": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Workbench",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(wx, wy, wb_z)),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.workbench_usd,
                    scale=(c.workbench_scale,) * 3,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                ),
            ),
        }
        table_spawn = sim_utils.UsdFileCfg(usd_path=c.table_usd, rigid_props=sim_utils.RigidBodyPropertiesCfg())
        out["tabletop"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Tabletop",
            spawn=table_spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(tx, ty, slab_top)),
        )
        for i in range(c.num_legs):
            sx, sy = c.slots[i]  # spawn pose only: on the (offset) studs; reset() lays out the real start
            out[f"leg_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Leg_%d" % i,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.leg_usd,
                    activate_contact_sensors=True,
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=192,
                        solver_velocity_iteration_count=1,
                        max_depenetration_velocity=5.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.leg_mass),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(tx + sx, ty + sy, slab_top + c.leg_start_z)),
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
        """Grab the table + leg handles, cache env origins, and pre-author the (disabled) weld
        joints. Called once after the scene is built."""
        super().bind(env)
        self.table: RigidObject = env.iscene["tabletop"]
        self.legs: list[RigidObject] = [env.iscene[f"leg_{i}"] for i in range(self.cfg.num_legs)]
        self.env_origins = env.iscene.env_origins
        self.welded = torch.zeros(env.num_envs, self.cfg.num_legs, dtype=torch.bool, device=env.device)
        self._precreate_weld_joints()

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh, unassembled start (all welds released): the tabletop sits upright on the workbench
        (slid off-centre by `table_offset`) and the legs lie on their side in a row near the centre."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]  # (m, 3)
        wx, wy = c.workbench_pos
        tx, ty = wx + c.table_offset[0], wy + c.table_offset[1]
        slab_top = c.surface_z + c.table_thickness

        # Tabletop: upright, slid off the workbench centre by `table_offset`.
        tbl = torch.zeros(m, 13, device=dev)
        tbl[:, 0:3] = origin + torch.tensor((tx, ty, slab_top), device=dev)
        tbl[:, 3] = 1.0  # identity quat
        self.table.write_root_state_to_sim(tbl, env_ids)

        # Legs: each at its configured start pose (leg_init_xy/_z/_quat, workbench-relative) + a small
        # xy jitter. Default is lying in a row near the centre; tune the cfg's leg_row_* / leg_init_*.
        quat = torch.tensor(c.leg_init_quat, device=dev)
        for k, leg in enumerate(self.legs):
            x, y = c.leg_init_xy[k]
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.tensor((wx + x, wy + y, c.surface_z + c.leg_init_z), device=dev)
            st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
            st[:, 3:7] = quat
            leg.write_root_state_to_sim(st, env_ids)

        self._reconcile_welds(env_ids, torch.zeros(m, c.num_legs, dtype=torch.bool, device=dev))

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Reconcile every weld against the seat criterion (auto-weld on seat). Runs each step."""
        ids = torch.arange(self.env.num_envs, device=self.env.device) if env_ids is None else env_ids
        self._reconcile_welds(ids, self._weld_targets()[ids])

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        """Restorable scene state for `env_ids`: world root states (pos+quat+lin/ang vel, 13) of the
        table and each leg, plus the per-leg weld flags."""
        return {
            "table": self.table.data.root_state_w[env_ids].clone(),
            "legs": torch.stack([leg.data.root_state_w[env_ids].clone() for leg in self.legs], dim=1),
            "welded": self.welded[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        """Restore what `get_state` returned: write the bodies, then reconcile the welds to exactly
        the recorded flags — from the restored poses, since handle `.data` is stale after a write."""
        self.table.write_root_state_to_sim(state["table"], env_ids)
        for i, leg in enumerate(self.legs):
            leg.write_root_state_to_sim(state["legs"][:, i], env_ids)
        self._reconcile_welds(env_ids, state["welded"], table_state=state["table"], leg_states=state["legs"])

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        corners = ", ".join(f"({x:+.2f}, {y:+.2f})" for x, y in c.slots[: c.num_legs])
        return (
            f"An IKEA-style side table, partly assembled, lying on a sturdy waist-high workbench. "
            f"The free-standing tabletop slab "
            f"(~0.55 x 0.55 m, {c.table_thickness * 100:.0f} cm thick) carries {c.num_legs} upright "
            f"threaded studs at its corners (xy = {corners} m, relative to the slab centre). "
            f"{c.num_legs} loose legs — each a threaded nut on a graspable cylindrical shaft — lie on "
            f"their side in a row near the centre of the workbench, ready to be picked up and fitted.\n"
            f"Goal: screw every leg down onto its stud (turn it clockwise while pressing down) until "
            f"it seats. A seated leg locks in place. "
            f"The table is assembled once all {c.num_legs} legs are seated."
        )

    # ----- progress (public: seated(); reads how far the assembly has got) -------------
    def seated(self) -> torch.Tensor:
        """Whether each leg is seated on a stud, shape (num_envs, num_legs). In the tabletop's own
        frame the leg must be: threaded down to seat depth, within `align_xy` of *some* stud, and its
        screw axis within `align_axis_deg` of the stud axis. Order-independent — a leg may seat on
        ANY stud. ('Seated' = a part has reached its final resting position.) An env is assembled
        when every entry of its row is True."""
        import math

        off = self._leg_offsets_in_table()  # (n, nlegs, 3): leg position in the slab frame
        studs = torch.tensor(self.cfg.slots, device=off.device, dtype=off.dtype)  # (nstuds, 2)
        near = (off[:, :, None, :2] - studs[None, None]).norm(dim=-1).amin(dim=-1)  # to nearest stud
        depth_ok = off[..., 2] <= self.cfg.seat_z
        axis_ok = self._leg_axis_cos() >= math.cos(math.radians(self.cfg.align_axis_deg))
        return depth_ok & (near <= self.cfg.align_xy) & axis_ok

    def success(self) -> torch.Tensor:
        """(N,) bool: every leg seated on a stud — this scene's assembled state
        (scene-level success alias, matching the other suites' surface)."""
        return self.seated().all(dim=1)

    def _leg_offsets_in_table(self) -> torch.Tensor:
        """Each leg's position in the tabletop's local frame, shape (num_envs, num_legs, 3): xy = its
        place in the slab plane (compare to the stud xy), z = height above the slab top. Measured in
        the table frame, so it is correct however the assembly is oriented (tilted, flipped, …)."""
        from isaaclab.utils.math import quat_apply_inverse

        tp, tq = self.table.data.root_pos_w, self.table.data.root_quat_w
        return torch.stack([quat_apply_inverse(tq, leg.data.root_pos_w - tp) for leg in self.legs], dim=1)

    def _leg_axis_cos(self) -> torch.Tensor:
        """cos of the angle between each leg's screw axis (its local +z) and the stud axis (the
        table's local +z), shape (num_envs, num_legs); 1.0 = perfectly aligned."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        table_up = quat_apply(self.table.data.root_quat_w, ez)  # (n, 3)
        legs_up = torch.stack([quat_apply(leg.data.root_quat_w, ez) for leg in self.legs], dim=1)
        return (legs_up * table_up[:, None, :]).sum(dim=-1)

    # ----- weld machinery (private; the auto-weld-on-seat sim-hack, not an agent action) --------
    # Every leg<->table FixedJoint is authored DISABLED before play and only toggled on/off. A
    # runtime joint binds two DYNAMIC bodies (the tabletop is always dynamic). To weld: set the
    # joint frames to the current relative pose and enable it; to release: disable it. No mid-sim
    # create/remove (RemovePrim resyncs physics and blows up the batch). Each (env, leg) is async.
    def _weld_targets(self) -> torch.Tensor:
        """THE STANDARD — which (env, leg) SHOULD be welded right now, shape (num_envs, num_legs).
        Monotonic: a leg welds once it is seated AND settling, and stays welded until reset (so a
        welded leg follows the table through any later motion, e.g. a flip). This is the single
        place that decides *when to auto-weld* — refine the criterion here."""
        settling = torch.stack([leg.data.root_lin_vel_w[:, 2].abs() for leg in self.legs], dim=-1)
        return self.welded | (self.seated() & (settling < self.cfg.seat_speed))

    def _reconcile_welds(self, env_ids, target, *, table_state=None, leg_states=None) -> None:
        """Modular weld/unweld: bring every (env, leg) joint for `env_ids` into line with `target`
        (bool, rows aligned to `env_ids`) — weld those that should be welded and aren't, release
        those that are and shouldn't be. Welds use `table_state`/`leg_states` poses when given (a
        set_state restore, where handle `.data` is stale right after a write), else the live poses.
        Only the changes are touched, so it stays cheap at large num_envs."""
        have = self.welded[env_ids]
        to_weld = target & ~have
        to_unweld = have & ~target
        if to_weld.any():
            if table_state is not None and leg_states is not None:  # set_state restore: saved poses
                tpos, tquat = table_state[:, 0:3], table_state[:, 3:7]
                lpos, lquat = leg_states[:, :, 0:3], leg_states[:, :, 3:7]
            else:  # live poses from the handles
                tpos, tquat = self.table.data.root_pos_w[env_ids], self.table.data.root_quat_w[env_ids]
                lpos = torch.stack([leg.data.root_pos_w[env_ids] for leg in self.legs], dim=1)
                lquat = torch.stack([leg.data.root_quat_w[env_ids] for leg in self.legs], dim=1)
            for row, k in to_weld.nonzero(as_tuple=False).tolist():
                self._weld_pair(int(env_ids[row]), k, tpos[row], tquat[row], lpos[row, k], lquat[row, k])
        for row, k in to_unweld.nonzero(as_tuple=False).tolist():
            self._unweld_pair(int(env_ids[row]), k)

    def _precreate_weld_joints(self) -> None:
        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()
        self._weld_paths: list[list[str]] = []
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            paths = []
            for k in range(self.cfg.num_legs):
                jp = f"{base}/rweld_{k}"
                j = UsdPhysics.FixedJoint.Define(stage, jp)
                j.CreateBody0Rel().SetTargets([f"{base}/Tabletop"])
                j.CreateBody1Rel().SetTargets([f"{base}/Leg_{k}/factory_nut_loose"])
                j.CreateJointEnabledAttr(False)
                paths.append(jp)
            self._weld_paths.append(paths)

    def _weld_pair(self, env_i: int, leg_k: int, tp, tq, lp, lq) -> None:
        """Lock (env_i, leg_k) to the table at the relative pose implied by world poses tp/tq (table)
        and lp/lq (leg). Each arg is a length-3 (pos) / length-4 (quat, wxyz) tensor."""
        from pxr import Gf, UsdPhysics
        from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

        q1c = quat_conjugate(lq.unsqueeze(0))
        rel_pos = quat_apply(q1c, (tp - lp).unsqueeze(0))[0]
        rel_rot = quat_mul(q1c, tq.unsqueeze(0))[0]
        stage = self.env.stage
        j = UsdPhysics.FixedJoint.Get(stage, self._weld_paths[env_i][leg_k])
        j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        j.CreateLocalPos1Attr(Gf.Vec3f(*(float(v) for v in rel_pos.tolist())))
        w, x, y, z = (float(v) for v in rel_rot.tolist())
        j.CreateLocalRot1Attr(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
        j.GetJointEnabledAttr().Set(True)
        self.welded[env_i, leg_k] = True

    def _unweld_pair(self, env_i: int, leg_k: int) -> None:
        """Release one (env, leg) weld by disabling its joint (re-weldable later)."""
        from pxr import UsdPhysics

        UsdPhysics.FixedJoint.Get(self.env.stage, self._weld_paths[env_i][leg_k]).GetJointEnabledAttr().Set(False)
        self.welded[env_i, leg_k] = False

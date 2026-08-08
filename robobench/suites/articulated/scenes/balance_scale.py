"""BalanceScaleScene — five identical boxes, hidden masses, one beam scale; sort by weight.

The object world: a two-tray BEAM SCALE (kinematic post + dynamic beam on a damped
revolute pivot, rigid trays fixed to the beam ends, a pointer needle over a tick
plate), five visually IDENTICAL boxes whose masses are re-randomized every episode,
and a ranked SHELF with five slots.
**Goal (carried here, no task layer): discover the weight order by weighing boxes
against each other on the scale, then place them on the shelf lightest -> heaviest
(slot 0 = lightest). The final order is judged against the hidden masses.**

The masses are HIDDEN: `describe()` never states them, and they are never exposed by
any agent-facing query — the tilting beam is the only instrument. The beam is honest
physics: torque = m*g*lever, so an off-centre placement biases the lever arm and can
produce a MISLEADING reading (a designed property, verified by the calibration sweep
in the smoke). The pivot carries viscous damping (post_step, the dial pattern) so
readings settle in bounded time.

Ported patterns: authored revolute + fixed joints (safe dial/spokes), visual-only
children for the needle/ticks (dial ticks), post_step-owned external torques.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


@dataclass
class BalanceScaleSceneCfg(BaseCfg):
    """Config for `BalanceScaleScene`."""

    # --- difficulty dials (mass band + pivot damping) ---------------
    n_boxes: int = 5
    mass_lo: float = 0.25  # per-episode masses sampled in [lo, hi] (kg)
    mass_hi: float = 0.95
    mass_min_gap: float = 0.08  # min pairwise gap — the instrument's resolution target
    beam_damping: float = 0.6  # pivot viscous damping (N*m per rad/s)
    beam_limit_deg: float = 8.0  # hard stops; a verdict = beam resting at a stop
    settle_speed: float = 0.03  # |omega| (rad/s) below which the beam counts as settled
    read_min_deg: float = 2.0  # |angle| below this at rest = "balanced/unreadable"

    # --- placement ------------------------------------------------------------------------------
    surface_z: float = 0.0
    scale_pos: tuple = (0.0, 0.10)  # post centre on the surface
    shelf_pos: tuple = (0.0, 0.42)  # shelf front-centre
    box_row_y: float = -0.25  # boxes start in a row on this line

    # --- structure ------------------------------------------------------------------------------
    bench_size: tuple = (1.4, 1.1)
    beam_len: float = 0.36
    beam_sec: tuple = (0.03, 0.02)  # beam cross-section (y, z)
    post_h: float = 0.16  # post top above the surface
    pivot_gap: float = 0.006  # AIR between post top and beam bottom: jointed
    # pairs must NEVER overlap — the pivot probe proved the beam was RESTING on the
    # post (rose 1 cm at boot, contact-welded 'hinge') through v1-v7.
    post_sq: float = 0.05
    tray_size: tuple = (0.13, 0.13, 0.008)
    tray_drop: float = 0.06  # trays hang this far below the beam axis (pendulum stability)
    box_size: tuple = (0.09, 0.09, 0.07)
    box_gap: float = 0.16  # spacing of the start row
    shelf_slot_w: float = 0.13
    shelf_depth: float = 0.16

    # Derived: tray centre x offset (the lever arm).
    lever: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        # Trays hang OUTBOARD of the beam tips (v3 bug: the beam passed directly over
        # the trays with 6 mm of clearance — no box could ever sit there).
        self.lever = self.beam_len / 2 + self.tray_size[0] / 2 - 0.01


@SCENES.register("scale")
class BalanceScaleScene(BaseScene):
    cfg: BalanceScaleSceneCfg

    # (name, rgb) identities for the boxes — visual only; masses stay hidden.
    BOX_COLORS = (("red", (0.75, 0.15, 0.15)), ("blue", (0.15, 0.30, 0.80)),
                  ("green", (0.15, 0.60, 0.20)), ("yellow", (0.85, 0.75, 0.15)),
                  ("purple", (0.55, 0.20, 0.65)))

    def __init__(self, cfg: BalanceScaleSceneCfg | None = None) -> None:
        super().__init__(cfg or BalanceScaleSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.surface_z
        sx, sy = c.scale_pos
        pivot_z = z0 + c.post_h + c.pivot_gap + c.beam_sec[1] / 2

        steel = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.40, 0.42, 0.46))
        brass = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.60, 0.28))
        wood = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.42, 0.25))

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
        }
        if z0 > 0:
            out["bench"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=sim_utils.CuboidCfg(
                    size=(c.bench_size[0], c.bench_size[1], z0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, z0 / 2)),
            )

        # Post (kinematic) + tick plate for the needle (visual child added in bind()).
        out["scale_post"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Scale_post",
            spawn=sim_utils.CuboidCfg(
                size=(c.post_sq, c.post_sq, c.post_h),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=steel,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(sx, sy, z0 + c.post_h / 2)),
        )

        # Beam (dynamic, long axis x) at the pivot height.
        out["beam"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Beam",
            spawn=sim_utils.CuboidCfg(
                size=(c.beam_len, c.beam_sec[0], c.beam_sec[1]),
                # NEVER SLEEPS: all forces on the beam arrive via the tensor API
                # (damping/arrest torques) which does NOT wake a sleeping body — the
                # pivot probe showed the beam frozen with stale velocity readings
                # (the exact dose-plunger signature) through v1-v6.
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    sleep_threshold=0.0, stabilization_threshold=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.30),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=brass,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(sx, sy, pivot_z)),
        )
        # Trays: flat plates fixed to the beam ends, dropped below the axis. Friction
        # holds a box at the <=12 deg beam limit (mu 0.6 > tan 12 deg) — no rims needed.
        for name, side in (("tray_l", -1.0), ("tray_r", 1.0)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.capitalize(),
                spawn=sim_utils.CuboidCfg(
                    size=c.tray_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        sleep_threshold=0.0, stabilization_threshold=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.0, dynamic_friction=0.9),
                    visual_material=steel,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(sx + side * c.lever, sy, pivot_z - c.tray_drop)),
            )

        # Shelf: kinematic plank + 6 dividers forming 5 ranked slots.
        shx, shy = c.shelf_pos
        shelf_w = c.n_boxes * c.shelf_slot_w + 0.02
        out["shelf"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Shelf",
            spawn=sim_utils.CuboidCfg(
                size=(shelf_w, c.shelf_depth, 0.012),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=wood,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(shx, shy, z0 + 0.006)),
        )
        for k in range(c.n_boxes + 1):
            dx = shx - shelf_w / 2 + 0.01 + k * c.shelf_slot_w
            out[f"divider_{k}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Divider_" + str(k),
                spawn=sim_utils.CuboidCfg(
                    size=(0.008, c.shelf_depth, 0.05),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=wood,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(dx, shy, z0 + 0.012 + 0.025)),
            )

        # Boxes: IDENTICAL size/shape, DISTINCT colors. The colors are identities, not
        # information: masses are still hidden and resampled every episode, so "the red
        # one" can never be memorized as heavy — but a viewer (and the agent's notes)
        # can now track which box went where across weighings. (Review 2026-07-15:
        # five indistinguishable cubes made the demo unfollowable.)
        for k in range(c.n_boxes):
            bx = (k - (c.n_boxes - 1) / 2) * c.box_gap
            out[f"box_{k}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Box_" + str(k),
                spawn=sim_utils.CuboidCfg(
                    size=c.box_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        sleep_threshold=0.0, stabilization_threshold=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.55),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=self.BOX_COLORS[k % len(self.BOX_COLORS)][1]),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(bx, c.box_row_y, z0 + c.box_size[2] / 2)),
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
        super().bind(env)
        c = self.cfg
        n = env.num_envs
        dev = env.device
        self.beam: RigidObject = env.iscene["beam"]
        self.trays = [env.iscene["tray_l"], env.iscene["tray_r"]]
        self.boxes = [env.iscene[f"box_{k}"] for k in range(c.n_boxes)]
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        self._build_needle()
        # Hidden per-episode masses (kg), shape (n, n_boxes). NEVER exposed by queries.
        self._masses = torch.full((n, c.n_boxes), 0.5, device=dev)
        # Beam ARREST (the transit-lock knob every precision balance has): while True,
        # post_step servoes the beam hard toward level so boxes can be loaded without
        # slamming the beam into its stops. Weigh with the arrest RELEASED.
        self.arrested = torch.ones(n, dtype=torch.bool, device=dev)
        self.dbg_torque = torch.zeros(n, device=dev)  # smoke diagnostic: extra pivot torque

    def _author_joints(self) -> None:
        """Beam pivot (revolute, axis y, limited) + tray fixed joints."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/beam_pivot")
            j.CreateBody0Rel().SetTargets([f"{base}/Scale_post"])
            j.CreateBody1Rel().SetTargets([f"{base}/Beam"])
            # CRITICAL: the beam overlaps the post at the pivot; without this the
            # default joint-pair collision welds the "hinge" solid (v1-v5).
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(
                0.0, 0.0, c.post_h / 2 + c.pivot_gap + c.beam_sec[1] / 2))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-c.beam_limit_deg)
            j.CreateUpperLimitAttr(c.beam_limit_deg)

            for name, prim, side in (("trayl_fix", "Tray_l", -1.0), ("trayr_fix", "Tray_r", 1.0)):
                jf = UsdPhysics.FixedJoint.Define(stage, f"{base}/{name}")
                jf.CreateBody0Rel().SetTargets([f"{base}/Beam"])
                jf.CreateBody1Rel().SetTargets([f"{base}/{prim}"])
                jf.CreateCollisionEnabledAttr(False)
                jf.CreateLocalPos0Attr(Gf.Vec3f(side * c.lever, 0.0, -c.tray_drop))
                jf.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                jf.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                jf.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    def _build_needle(self) -> None:
        """Visual pointer needle (child of the Beam prim — rotates with it, no physics)
        + tick marks on the post so the reading is legible in renders."""
        import omni.usd
        from pxr import Gf, UsdGeom

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            # Idempotency guard (multi-env boot, 2026-07-17): env_1.. INHERIT env_0's
            # subtree (cloner copy_from_source=False), so env_0's decorations already
            # compose under every env — AddXformOp on them would hard-fail. These are
            # static identical offsets: author env_0 only, inheritance shows them
            # everywhere.
            if stage.GetPrimAtPath(f"/World/envs/env_{i}/Beam/needle").IsValid():
                continue
            needle = UsdGeom.Cube.Define(stage, f"/World/envs/env_{i}/Beam/needle")
            needle.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(needle)
            xf.AddTranslateOp().Set(Gf.Vec3d(0.0, -c.beam_sec[0] / 2 - 0.003, -0.05))
            xf.AddScaleOp().Set(Gf.Vec3f(0.008, 0.004, 0.10))
            needle.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.1, 0.1)])
            tick = UsdGeom.Cube.Define(stage, f"/World/envs/env_{i}/Scale_post/tick0")
            tick.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(tick)
            xf.AddTranslateOp().Set(Gf.Vec3d(0.0, -c.post_sq / 2 - 0.004, c.post_h / 2 - 0.045))
            xf.AddScaleOp().Set(Gf.Vec3f(0.004, 0.004, 0.03))
            tick.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.1, 0.1)])
            for tag, side in (("hl", -1.0), ("hr", 1.0)):
                strut = UsdGeom.Cube.Define(stage, f"/World/envs/env_{i}/Beam/hanger_{tag}")
                strut.CreateSizeAttr(1.0)
                xf = UsdGeom.Xformable(strut)
                xf.AddTranslateOp().Set(Gf.Vec3d(side * (c.beam_len / 2 - 0.006), 0.0,
                                                 -c.tray_drop / 2))
                xf.AddScaleOp().Set(Gf.Vec3f(0.008, 0.008, c.tray_drop))
                strut.CreateDisplayColorAttr([Gf.Vec3f(0.72, 0.60, 0.28)])

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Level beam, boxes back to the start row (small jitter), FRESH HIDDEN MASSES."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        z0 = c.surface_z
        sx, sy = c.scale_pos
        pivot_z = z0 + c.post_h + c.pivot_gap + c.beam_sec[1] / 2

        def put(body: RigidObject, pos) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = self.env_origins[env_ids] + torch.tensor(pos, device=dev)
            st[:, 3] = 1.0
            body.write_root_state_to_sim(st, env_ids)

        put(self.beam, (sx, sy, pivot_z))
        put(self.trays[0], (sx - c.lever, sy, pivot_z - c.tray_drop))
        put(self.trays[1], (sx + c.lever, sy, pivot_z - c.tray_drop))
        for k in range(c.n_boxes):
            bx = (k - (c.n_boxes - 1) / 2) * c.box_gap
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = self.env_origins[env_ids] + torch.tensor(
                [bx, c.box_row_y, z0 + c.box_size[2] / 2 + 0.002], device=dev)
            st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * 0.015
            st[:, 3] = 1.0
            self.boxes[k].write_root_state_to_sim(st, env_ids)

        self.arrested[env_ids] = True
        # Fresh hidden masses: distinct, pairwise gap >= mass_min_gap.
        for e in env_ids.tolist():
            while True:
                ms = (torch.rand(c.n_boxes) * (c.mass_hi - c.mass_lo) + c.mass_lo)
                d = (ms.view(-1, 1) - ms.view(1, -1)).abs() + torch.eye(c.n_boxes) * 9.0
                if float(d.min()) >= c.mass_min_gap:
                    break
            self._masses[e] = ms.to(dev)
        self._apply_masses(env_ids)

    def _apply_masses(self, env_ids: torch.Tensor) -> None:
        """Write the hidden masses into PhysX (per-box RigidObject views) and VERIFY by
        readback — a silent no-op here voids the whole task (v1 smoke: every weighing
        balanced because all boxes stayed at the spawn mass)."""
        all_ids = torch.arange(self.env.num_envs)
        for k, box in enumerate(self.boxes):
            view = box.root_physx_view
            masses = view.get_masses().clone().cpu().view(self.env.num_envs, -1)
            masses[env_ids.cpu(), :] = self._masses[env_ids, k].cpu().view(-1, 1)
            view.set_masses(masses.view_as(view.get_masses()), all_ids)
        back = self.boxes[0].root_physx_view.get_masses().cpu().view(-1)[0]
        expect = float(self._masses[0, 0])
        if abs(float(back) - expect) > 1e-4:
            print(f"[balance_scale] MASS APPLY FAILED: readback {float(back):.3f} "
                  f"!= {expect:.3f} — verdicts are void", flush=True)

    # ----- readings -----------------------------------------------------------------------------
    def beam_angle_deg(self) -> torch.Tensor:
        """Signed beam tilt (deg): positive = RIGHT (+x) side down. Physical — the agent
        can equally read it off the needle in look() images."""
        q = self.beam.data.root_quat_w
        ang = 2.0 * torch.atan2(q[:, 2], q[:, 0])  # twist about y
        # Sign MEASURED in the v8 smoke (heavier right -> ang > 0 -> right down = +).
        return torch.rad2deg(ang)

    def beam_settled(self) -> torch.Tensor:
        # Pivot-axis component only: full-norm jitter from box/tray contacts kept the
        # v1 smoke "unsettled" forever even though the reading was long since stable.
        w = self.beam.data.root_ang_vel_w[:, 1].abs()
        return w < self.cfg.settle_speed

    def verdict(self) -> torch.Tensor:
        """Per env: +1 right-heavier, -1 left-heavier, 0 unreadable/level (only valid
        while settled — the caller must check beam_settled())."""
        a = self.beam_angle_deg()
        out = torch.zeros_like(a)
        out = torch.where(a > self.cfg.read_min_deg, torch.ones_like(a), out)
        out = torch.where(a < -self.cfg.read_min_deg, -torch.ones_like(a), out)
        return out

    # ----- mechanics ----------------------------------------------------------------------------
    def arrest(self, on: bool = True) -> None:
        """Engage/release the beam arrest (transit lock). Load boxes arrested; weigh
        released."""
        self.arrested[:] = bool(on)

    def post_step(self) -> None:
        """Pivot damping (the beam must settle, not oscillate); strong level-hold PD
        while the arrest is engaged."""
        dev = self.env.device
        n = self.env.num_envs
        ey = torch.tensor([0.0, 1.0, 0.0], device=dev).expand(n, 3)
        w = (self.beam.data.root_ang_vel_w * ey).sum(dim=-1)
        # beam_angle_deg() returns +deg(twist_y) (its sign convention: right side down =
        # positive). The level-hold PD drives the twist to zero with NEGATIVE position
        # feedback: tau_y = -k*twist_y - d*w_y.
        twist_y = torch.deg2rad(self.beam_angle_deg())
        free = -self.cfg.beam_damping * w
        hold = (-8.0 * twist_y - 1.0 * w).clamp(-6.0, 6.0)
        tq = torch.where(self.arrested, hold, free) + self.dbg_torque
        self.beam.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=dev), tq.view(n, 1, 1) * ey.view(n, 1, 3))

    # ----- state (masses INCLUDED for restore; agent-facing queries never read this) -------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"beam": self.beam, "tray_l": self.trays[0], "tray_r": self.trays[1],
                  **{f"box_{k}": b for k, b in enumerate(self.boxes)}}
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "masses": self._masses[env_ids].clone(),
            "arrested": self.arrested[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"beam": self.beam, "tray_l": self.trays[0], "tray_r": self.trays[1],
                  **{f"box_{k}": b for k, b in enumerate(self.boxes)}}
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        self._masses[env_ids] = state["masses"]
        self.arrested[env_ids] = state["arrested"]
        self._apply_masses(env_ids)

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A two-tray beam scale stands at ({c.scale_pos[0]:.2f}, {c.scale_pos[1]:.2f}): "
            f"a brass beam on a damped pivot with a flat tray on each end ({2 * c.lever:.2f} m "
            f"apart) and a red pointer needle over the post's tick mark. Five boxes of "
            f"IDENTICAL size ({c.box_size[0]:.2f} m cubes-ish) wait in a row at y="
            f"{c.box_row_y:.2f}, colored (left to right) "
            f"{', '.join(nm for nm, _ in self.BOX_COLORS[:c.n_boxes])} — the colors are "
            f"only identities: each box has a DIFFERENT hidden weight, re-randomized "
            f"every episode, and no query reveals them. Behind the scale, a shelf with "
            f"{c.n_boxes} labelled slots (slot 0 leftmost).\n"
            f"The scale is the only instrument: put one box on each tray, let the beam "
            f"settle, and the needle tilts toward the heavier side (a level, settled beam "
            f"within {c.read_min_deg:.0f} deg means it cannot tell). Place a box well "
            f"OFF-CENTRE on its tray and the lever arm changes — sloppy placement can give "
            f"a WRONG reading, so centre the boxes (or re-weigh when in doubt). The beam "
            f"starts ARRESTED (transit-locked level, like any precision balance): load "
            f"boxes while arrested, call arrest(False) to weigh, arrest(True) before "
            f"unloading — a released beam slams under a one-sided load.\n"
            f"Goal: place all {c.n_boxes} boxes into the shelf slots ordered by weight — "
            f"lightest in slot 0, heaviest in slot {c.n_boxes - 1}. Judged on the final "
            f"order only; the number of weighings is your efficiency score."
        )

    # ----- progress -----------------------------------------------------------------------------
    def slot_x(self, k: int) -> float:
        c = self.cfg
        shelf_w = c.n_boxes * c.shelf_slot_w + 0.02
        return c.shelf_pos[0] - shelf_w / 2 + 0.01 + (k + 0.5) * c.shelf_slot_w

    def slot_of_box(self) -> torch.Tensor:
        """(num_envs, n_boxes) slot index per box, -1 if not on the shelf."""
        c = self.cfg
        n = self.env.num_envs
        out = torch.full((n, c.n_boxes), -1, dtype=torch.long, device=self.env.device)
        shy = c.shelf_pos[1]
        for k, box in enumerate(self.boxes):
            p = box.data.root_pos_w - self.env_origins
            on_shelf = ((p[:, 1] - shy).abs() < c.shelf_depth / 2) & \
                (p[:, 2] > c.surface_z) & (p[:, 2] < c.surface_z + 0.15)
            for s in range(c.n_boxes):
                in_slot = on_shelf & ((p[:, 0] - self.slot_x(s)).abs() < c.shelf_slot_w / 2 - 0.005)
                out[:, k] = torch.where(in_slot, torch.full_like(out[:, k], s), out[:, k])
        return out

    def success(self) -> torch.Tensor:
        """All boxes on the shelf, one per slot, slot order == hidden mass order."""
        c = self.cfg
        n = self.env.num_envs
        slots = self.slot_of_box()  # (n, n_boxes)
        ok = (slots >= 0).all(dim=1)
        # each slot used exactly once
        for s in range(c.n_boxes):
            ok = ok & ((slots == s).sum(dim=1) == 1)
        # rank of each box's mass must equal its slot
        rank = self._masses.argsort(dim=1).argsort(dim=1)  # 0 = lightest
        ok = ok & (slots == rank).all(dim=1)
        settled = torch.ones(n, dtype=torch.bool, device=self.env.device)
        for b in self.boxes:
            settled = settled & (b.data.root_lin_vel_w.norm(dim=-1) < 0.05)
        return ok & settled


"""PressurePlateScene — hold a spring-loaded pressure plate DOWN by loading it with a
sufficiently heavy block (seed: rlbench/push_button, strategically inverted).

The seed task is a momentary fingertip press: reach the button, poke it, done — no object
interaction at all. Here the "button" is an industrial floor-scale pressure plate that a
poke CANNOT actuate: it rides a vertical prismatic slide with a spring return (the
microwave-button mechanism from robobench.suites.articulated), so any transient press
springs straight back, and success is gated on the plate being HELD at depth by a resting
object — the rubric requires a block physically supported by the plate, so even an
infinitely patient end-effector press never scores. The solver's plan is therefore a
different skill entirely: choose the heavy block among light foam distractors, pick it up,
and set it down ON the plate so its weight keeps the plate bottomed after the hand leaves.

Mechanism (all procedural primitives; the jointed-button pattern proven in
microwave_meal.py): a kinematic pedestal, a dynamic plate joined to it by a Z prismatic
joint (collision between the pair disabled; symmetric limits +/- travel — the GPU
sign-convention hedge), and a post_step spring `f_z = k*(home - z) - c*v_z` with gravity
disabled on the plate so the empty plate rests exactly at home. A load M depresses the
plate by M*g/k until the -travel stop: with k = 80 N/m and travel = 35 mm the plate
bottoms out under >= ~0.29 kg. The dark-red load block (0.5 kg) holds it down with ~75%
margin; one pale foam block (60 g) sags it only ~7 mm and even both foams stacked
(~15 mm) stay well under the 28 mm press threshold — the near-miss controls are built
into the object set. A lamp on the pedestal glows green while the plate is pressed (the
visible outcome for video/viewer).

Judged on PHYSICAL outcome only: live plate depth (>= press_frac * travel), a block
geometrically resting on the plate (center over the plate, bottom at plate-top height),
both settled, SUSTAINED for `press_hold_steps` consecutive substeps (a post_step counter
— the anti-poke gate). score() in [0, 1]: 0.2 latched once any block has been lifted
clear of the floor, 0.5 latched once a block has reached the plate top, 0.5..0.9 live
with a block on the plate scaling with depth, 1.0 iff success. Latches make transient
progress stick; success itself is the current sustained physical state.

Per-episode randomization: pedestal xy jitter + yaw (plate teleported consistently with
it — the joint sees an unchanged relative pose), block-to-slot permutation on the scatter
arc, per-block xy jitter + free yaw.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class PressurePlateSceneCfg(BaseCfg):
    """Config for `PressurePlateScene`. Spring constants are sized for 120 Hz stability
    (m=0.08 kg, k=80 N/m -> ~5 Hz, ~24 substeps/period; the microwave-button precedent)."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    press_frac: float = tunable(0.80)  # depth fraction of travel that counts as pressed
    press_hold_steps: int = tunable(90)  # consecutive substeps the press must be sustained
    on_plate_r: float = tunable(0.050)  # block center within this of the plate axis (m)
    on_plate_z_tol: float = tunable(0.020)  # block bottom within this of the plate top (m)
    lift_h: float = tunable(0.050)  # block bottom above this height latches "lifted"
    settle_speed: float = tunable(0.05)  # max |v| when judging load/plate (m/s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    reset_pos_jitter: float = tunable(0.04)  # uniform +/- xy jitter per block at reset
    reset_yaw_deg: float = tunable(180.0)  # uniform +/- yaw per block at reset
    ped_jitter: float = tunable(0.03)  # pedestal xy jitter (plate follows rigidly)
    ped_yaw_deg: float = tunable(180.0)  # pedestal yaw range
    shuffle_slots: bool = tunable(True)  # permute block -> arc-slot assignment per episode

    # --- tunable: placement ------------------------------------------------------------------
    surface_z: float = tunable(0.0)  # work-surface height; 0 = on the ground (null smoke)
    ped_pos: tuple = tunable((0.22, 0.0))  # pedestal centre on the surface
    blocks_center: tuple = tunable((-0.12, 0.0))  # scatter-arc centre (blocks on the far side)
    spawn_radius: float = tunable(0.18)  # scatter arc radius
    spawn_arc: tuple = tunable((90.0, 270.0))  # scatter arc (deg) around blocks_center

    # --- info: structure ----------------------------------------------------------------------
    ped_size: tuple = info((0.18, 0.18, 0.06))  # kinematic pedestal (x, y, z)
    plate_size: tuple = info((0.10, 0.10, 0.024))  # the sliding plate (bright red)
    travel: float = info(0.035)  # plate vertical travel, home -> bottomed (m)
    spring_k: float = info(80.0)  # spring return (N/m); bottom-out load = k*travel = 2.8 N
    spring_c: float = info(4.0)  # damping (N*s/m), ~0.7 critical for the empty plate
    plate_mass: float = info(0.08)
    plate_color: tuple = info((0.80, 0.16, 0.12))
    ped_color: tuple = info((0.28, 0.28, 0.32))
    contact_offset: float = info(0.004)
    # (name, cube edge (m), mass (kg), rgb): ONE load block heavy enough to bottom the
    # plate (4.9 N >> 2.8 N) and two pale foam distractors (0.59 N each -> ~7 mm sag;
    # both stacked ~15 mm — still far under the 28 mm press threshold).
    blocks: tuple = info((
        ("load", 0.075, 0.50, (0.45, 0.10, 0.09)),
        ("foam_0", 0.045, 0.06, (0.88, 0.83, 0.55)),
        ("foam_1", 0.045, 0.06, (0.62, 0.76, 0.90)),
    ))
    lamp_dim: tuple = info((0.05, 0.22, 0.07))
    lamp_lit: tuple = info((0.10, 0.92, 0.15))

    # Derived (filled in __post_init__).
    press_depth: float = field(default=None, init=False)  # depth that counts as pressed (m)
    plate_home_lz: float = field(default=None, init=False)  # plate centre z above surface, home
    trigger_kg: float = field(default=None, init=False)  # load that bottoms the plate (kg)

    def __post_init__(self) -> None:
        self.press_depth = round(self.press_frac * self.travel, 4)
        self.plate_home_lz = round(self.ped_size[2] + self.travel + self.plate_size[2] / 2, 4)
        self.trigger_kg = round(self.spring_k * self.travel / 9.81, 3)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("pressure_plate")
class PressurePlateScene(BaseScene):
    cfg: PressurePlateSceneCfg

    def __init__(self, cfg: PressurePlateSceneCfg | None = None) -> None:
        super().__init__(cfg or PressurePlateSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, kinematic pedestal, the sliding plate at home, and the blocks at
        their nominal scatter slots (reset() re-places everything)."""
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
            "pedestal": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pedestal",
                spawn=sim_utils.CuboidCfg(
                    size=c.ped_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.ped_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.ped_pos[0], c.ped_pos[1], z0 + c.ped_size[2] / 2)),
            ),
            # Gravity is DISABLED on the plate: the post_step spring force about `home`
            # then makes home the exact rest pose with no preload bookkeeping (the spring
            # preload is the disabled weight).
            "plate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plate",
                spawn=sim_utils.CuboidCfg(
                    size=c.plate_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=True, max_depenetration_velocity=0.5),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.plate_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.plate_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.ped_pos[0], c.ped_pos[1], z0 + c.plate_home_lz)),
            ),
        }

        a0, a1 = (math.radians(v) for v in c.spawn_arc)
        cx, cy = c.blocks_center
        nb = len(c.blocks)
        for i, (name, edge, mass, rgb) in enumerate(c.blocks):
            ang = a0 + (a1 - a0) * (i + 0.5) / nb
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Block_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=(edge, edge, edge),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(cx + c.spawn_radius * math.cos(ang),
                         cy + c.spawn_radius * math.sin(ang), z0 + edge / 2 + 0.003)),
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
        """Grab handles, author the plate slide joints, build the lamps, allocate latches."""
        super().bind(env)
        c = self.cfg
        n = env.num_envs
        dev = env.device
        self.pedestal: RigidObject = env.iscene["pedestal"]
        self.plate: RigidObject = env.iscene["plate"]
        self.blocks: dict[str, RigidObject] = {
            name: env.iscene[name] for name, _e, _m, _rgb in c.blocks}
        self.env_origins = env.iscene.env_origins
        self._half = torch.tensor([e / 2 for _n, e, _m, _rgb in c.blocks], device=dev)
        # Plate home z is a constant of the flat world (pedestal jitter is xy-only).
        self._home_z = self.env_origins[:, 2] + c.surface_z + c.plate_home_lz
        # Latches + the sustained-press counter (the anti-poke gate).
        self._lifted = torch.zeros(n, dtype=torch.bool, device=dev)
        self._placed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._hold = torch.zeros(n, dtype=torch.long, device=dev)
        self._author_joints()
        self._build_lamps()

    def _author_joints(self) -> None:
        """Per env: the plate's Z prismatic slide on the pedestal — pair collision
        disabled (the joint limit is the mechanical stop), SYMMETRIC limits +/- travel
        (the microwave GPU lesson: a [0, travel] range can pin the plate at zero under
        either joint-coordinate sign convention)."""
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/plate_slide")
            j.CreateBody0Rel().SetTargets([f"{base}/Pedestal"])
            j.CreateBody1Rel().SetTargets([f"{base}/Plate"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(
                0.0, 0.0, c.ped_size[2] / 2 + c.travel + c.plate_size[2] / 2))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-c.travel)
            j.CreateUpperLimitAttr(c.travel)
            lim = PhysxSchema.PhysxLimitAPI.Apply(j.GetPrim(), "linear")
            if hasattr(lim, "CreateContactDistanceAttr"):  # removed in Isaac Sim 5.1 schema
                lim.CreateContactDistanceAttr(0.001)

    def _build_lamps(self) -> None:
        """A lamp dome on the pedestal top corner (visual child, no collision): green
        while the plate is pressed — the appliance's visible outcome."""
        import omni.usd
        from pxr import Gf, UsdGeom

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        self._lamps = []
        off = c.ped_size[0] / 2 - 0.022
        for i in range(self.env.num_envs):
            lamp = UsdGeom.Sphere.Define(stage, f"/World/envs/env_{i}/Pedestal/lamp")
            lamp.CreateRadiusAttr(0.012)
            UsdGeom.Xformable(lamp.GetPrim()).AddTranslateOp().Set(
                Gf.Vec3d(off, off, c.ped_size[2] / 2 + 0.006))
            lamp.CreateDisplayColorAttr([Gf.Vec3f(*c.lamp_dim)])
            self._lamps.append(lamp)
        self._lamp_shown = [False] * self.env.num_envs

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pedestal (with the plate riding rigidly at home above it) at
        ped_pos + jitter with a random yaw; blocks permuted over the scatter-arc slots
        with xy jitter + free yaw; latches cleared; force buffers zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- pedestal + plate: a COMMON planar transform, so the authored joint sees an
        # unchanged relative pose (both bodies share the vertical joint axis).
        pxy = torch.zeros(m, 2, device=dev)
        pxy[:, 0] = c.ped_pos[0]
        pxy[:, 1] = c.ped_pos[1]
        pxy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.ped_jitter
        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.ped_yaw_deg) / 2
        qw, qz = torch.cos(half), torch.sin(half)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = pxy
        st[:, 2] = c.surface_z + c.ped_size[2] / 2
        st[:, 3], st[:, 6] = qw, qz
        st[:, 0:3] += origin
        self.pedestal.write_root_state_to_sim(st, env_ids)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = pxy
        st[:, 2] = c.surface_z + c.plate_home_lz
        st[:, 3], st[:, 6] = qw, qz
        st[:, 0:3] += origin
        self.plate.write_root_state_to_sim(st, env_ids)

        # --- blocks: slot permutation + jitter + free yaw ---
        a0, a1 = (math.radians(v) for v in c.spawn_arc)
        nb = len(c.blocks)
        cx, cy = c.blocks_center
        if c.shuffle_slots:
            slot = torch.rand(m, nb, device=dev).argsort(dim=1)  # block i -> slot[:, i]
        else:
            slot = torch.arange(nb, device=dev).expand(m, nb)
        for i, (name, edge, _mass, _rgb) in enumerate(c.blocks):
            ang = a0 + (a1 - a0) * (slot[:, i].float() + 0.5) / nb
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = cx + c.spawn_radius * torch.cos(ang)
            st[:, 1] = cy + c.spawn_radius * torch.sin(ang)
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
            st[:, 2] = c.surface_z + edge / 2 + 0.003
            bhalf = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.reset_yaw_deg) / 2
            st[:, 3] = torch.cos(bhalf)
            st[:, 6] = torch.sin(bhalf)
            st[:, 0:3] += origin
            self.blocks[name].write_root_state_to_sim(st, env_ids)

        self._lifted[env_ids] = False
        self._placed[env_ids] = False
        self._hold[env_ids] = 0
        # Zero the plate's external-force buffer: a stale spring force from the previous
        # episode would kick the gravity-free plate for one substep before post_step runs.
        n = self.env.num_envs
        self.plate.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=dev), torch.zeros(n, 1, 3, device=dev))

    # ----- geometry queries ---------------------------------------------------------------------
    def plate_depth(self) -> torch.Tensor:
        """(N,) plate depression below home (m), 0 = fully up."""
        return (self._home_z - self.plate.data.root_pos_w[:, 2]).clamp(min=0.0)

    def _block_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos (N,B,3), |lin_vel| (N,B)) for all blocks, manifest order."""
        pos = torch.stack([b.data.root_pos_w for b in self.blocks.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.blocks.values()], dim=1)
        return pos, vel

    def on_plate(self) -> torch.Tensor:
        """(N, B) bool, geometric: block centre within `on_plate_r` of the plate axis and
        block bottom at the LIVE plate-top height (tracks the plate through its travel)."""
        c = self.cfg
        pos, _v = self._block_tensors()
        pxy = self.plate.data.root_pos_w[:, None, :2]
        near = (pos[:, :, :2] - pxy).norm(dim=-1) < c.on_plate_r
        plate_top = (self.plate.data.root_pos_w[:, 2] + c.plate_size[2] / 2).unsqueeze(1)
        bottom = pos[:, :, 2] - self._half[None, :]
        return near & ((bottom - plate_top).abs() < c.on_plate_z_tol)

    def loaded(self) -> torch.Tensor:
        """(N,) bool: some SETTLED block rests on the plate (the anti-poke clause — a
        bare end-effector press has no block to show)."""
        _p, vel = self._block_tensors()
        return (self.on_plate() & (vel < self.cfg.settle_speed)).any(dim=1)

    def pressed_now(self) -> torch.Tensor:
        """(N,) bool, instantaneous: plate at depth, loaded by a settled block, plate still."""
        c = self.cfg
        plate_still = self.plate.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return (self.plate_depth() >= c.press_depth) & self.loaded() & plate_still

    # ----- mechanics (every substep) --------------------------------------------------------------
    def post_step(self) -> None:
        c = self.cfg
        dev = self.env.device
        n = self.env.num_envs

        # Spring return about home (gravity is disabled on the plate, so home is the
        # exact empty rest pose): f_z = k*(home - z) - c*v_z.
        d = self._home_z - self.plate.data.root_pos_w[:, 2]
        v_z = self.plate.data.root_lin_vel_w[:, 2]
        f_z = c.spring_k * d - c.spring_c * v_z
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
        self.plate.set_external_force_and_torque(
            f_z.view(n, 1, 1) * ez.view(n, 1, 3), torch.zeros(n, 1, 3, device=dev))

        # Latches + the sustained-press counter.
        pos, _vel = self._block_tensors()
        bottom = pos[:, :, 2] - self.env_origins[:, None, 2] - self._half[None, :]
        self._lifted |= (bottom > c.surface_z + c.lift_h).any(dim=1)
        self._placed |= self.on_plate().any(dim=1)
        now = self.pressed_now()
        self._hold = torch.where(now, self._hold + 1, torch.zeros_like(self._hold))

        # Lamp refresh (env-wise, on change only).
        from pxr import Gf

        for e in range(n):
            lit = bool(now[e])
            if lit != self._lamp_shown[e]:
                self._lamp_shown[e] = lit
                col = c.lamp_lit if lit else c.lamp_dim
                self._lamps[e].GetDisplayColorAttr().Set([Gf.Vec3f(*col)])

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "pedestal": self.pedestal.data.root_state_w[env_ids].clone(),
            "plate": self.plate.data.root_state_w[env_ids].clone(),
            "blocks": {n: b.data.root_state_w[env_ids].clone()
                       for n, b in self.blocks.items()},
            "latches": {k: getattr(self, k)[env_ids].clone()
                        for k in ("_lifted", "_placed", "_hold")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.pedestal.write_root_state_to_sim(state["pedestal"], env_ids)
        self.plate.write_root_state_to_sim(state["plate"], env_ids)
        for n, b in self.blocks.items():
            b.write_root_state_to_sim(state["blocks"][n], env_ids)
        for k, v in state["latches"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A pedestal ({c.ped_size[0] * 100:.0f} cm square, dark grey) stands on the "
            f"surface, carrying a bright-red spring-loaded pressure plate "
            f"({c.plate_size[0] * 100:.0f} cm square) that rides {c.travel * 1000:.0f} mm "
            f"of vertical travel; a lamp on the pedestal corner glows green while the "
            f"plate is held fully down. Scattered on the other side lie three loose "
            f"cubes: one dark-red LOAD block ({c.blocks[0][1] * 100:.1f} cm, "
            f"~{c.blocks[0][2] * 1000:.0f} g) and two pale foam blocks "
            f"({c.blocks[1][1] * 100:.1f} cm, ~{c.blocks[1][2] * 1000:.0f} g each).\n"
            f"Goal: keep the plate pressed all the way down by WEIGHTING it — place a "
            f"heavy enough object on the plate and leave it there. The spring needs "
            f"about {c.trigger_kg * 1000:.0f} g on the plate to bottom out, so only the "
            f"dark-red load block can do it; a foam block only sags the plate a little, "
            f"and pressing the plate by hand does not count — it must be held down by a "
            f"resting object, sustained, with everything settled."
        )

    # ----- progress / rubric ----------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the plate has been held at depth by a settled resting block for
        `press_hold_steps` consecutive substeps (live physical state — remove the load
        and success reverts as the plate springs back)."""
        return self._hold >= self.cfg.press_hold_steps

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0 for nothing; 0.2 latched once any block was lifted
        clear of the floor; 0.5 latched once a block reached the plate; 0.5..0.9 live
        with a block on the plate, scaling with depth toward the press threshold;
        1.0 iff success."""
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        s = torch.zeros(n, device=dev)
        s = torch.where(self._lifted, torch.full_like(s, 0.2), s)
        s = torch.where(self._placed, torch.maximum(s, torch.full_like(s, 0.5)), s)
        frac = (self.plate_depth() / c.press_depth).clamp(0.0, 1.0)
        live = torch.where(self.on_plate().any(dim=1), 0.5 + 0.4 * frac, torch.zeros(n, device=dev))
        s = torch.maximum(s, live)
        return torch.where(self.success(), torch.ones_like(s), s)


register_env("sim_gen", lambda: EnvCfg(scene="pressure_plate", robot="null", env_spacing=3))

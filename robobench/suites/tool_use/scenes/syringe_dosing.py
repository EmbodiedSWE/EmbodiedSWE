"""SyringeDosingScene — draw a full syringe, dispense exactly one third into each of 3 wells.

The object world: a syringe (barrel + prismatic plunger, with finger-loop flanges and a
thumb ring), standing upright in a 4-post stand; a well plate with one wide RESERVOIR
and three small SAMPLE wells; everything on a bench.
**Goal (carried here, no task layer): draw a full load from the reservoir, then dispense
33% +/- 5% of capacity into each sample well, and park the syringe back in its stand.**

No fluid is simulated. "Liquid moved" = plunger travel while the nozzle tip is SEATED on
a well (within `seat_tol` laterally, hovering within the seating band above the rim):
  - plunger OUT (travel increasing) while seated on the reservoir -> liquid drawn;
  - plunger OUT anywhere else -> air (nothing gained);
  - plunger IN (travel decreasing) -> liquid leaves first; it lands in whichever well
    the tip is seated on, otherwise it is SPILLED (spills are counted and shown).
Over-dispensing is IRREVERSIBLE per well: wells only fill (no negative doses), so a well
pushed past the band can never come back — one careless push voids the episode. (The
reservoir itself is unlimited: a short draw can be topped back up by re-seating on it.)

The plunger has deliberate friction (viscous + Coulomb, applied in post_step) so it
moves smoothly under drive and STAYS PUT when released — metering is a slow-push
control problem, not a flick. External drivers (smoke / RL / scripted hands) write
`scene.plunger_drive` (N, +z draws) instead of touching the force buffer directly;
post_step owns `set_external_force_and_torque` for the plunger.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg, info, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


@dataclass
class SyringeDosingSceneCfg(BaseCfg):
    """Config for `SyringeDosingScene`."""

    # --- tunable: difficulty dials -----------------------------------------------------------
    seat_tol: float = tunable(0.006)  # max lateral tip offset that still counts as seated (m)
    seat_band: float = tunable(0.012)  # tip must hover within this above the well rim (m)
    dose_band: tuple = tunable((0.28, 0.38))  # per-well acceptance band (fraction of capacity)
    draw_min: float = tunable(0.95)  # min drawn fraction for the draw stage
    # O-ring friction = INTERNAL action-reaction PAIR between plunger and barrel:
    # a position lock (anchor spring) while not firmly driven, viscous drag while
    # driven. MEASURED (v9): applying these as one-sided external forces levitated the
    # whole rig off its stand — the "brake" pushed the plunger up with nothing pushing
    # the barrel down, and 3.1 N of net lift beat the rig's 2.35 N weight. Internal
    # pairs cannot exert a net force on the assembly, by construction.
    # Small viscous drag (explicit force, clamped in post_step). STABILITY: must
    # stay far under 2*m/dt = 2*0.05*120 = 12 N*s/m.
    plunger_visc: float = tunable(3.0)  # viscous drag while driven (N per m/s)
    plunger_break_free: float = tunable(1.5)  # o-ring joint friction: drive above this slides (N)
    reset_jitter: float = tunable(0.015)  # +/- xy jitter of the well plate per episode (m)
    debug_forensics: bool = tunable(False)  # per-substep draw-loss ledgers (smoke diagnosis only)

    # --- tunable: placement -------------------------------------------------------------------
    surface_z: float = tunable(0.0)
    # Stand moved +x/-y (was (0.16,-0.02)): its collar sat 11 cm from well 2 and
    # blocked the glide corridor — v37/38 never converged on well 2's seat and the
    # jammed nozzle bled the remaining load to `spilled`.
    stand_pos: tuple = tunable((0.22, -0.06))  # syringe stand centre on the surface
    plate_pos: tuple = tunable((-0.06, 0.10))  # well-plate centre

    # --- info: structure ----------------------------------------------------------------------
    bench_size: tuple = info((1.1, 0.9))
    barrel_r: float = info(0.025)
    barrel_l: float = info(0.20)
    nozzle_r: float = info(0.006)
    nozzle_l: float = info(0.05)
    stroke: float = info(0.12)  # plunger travel = syringe capacity
    plunger_r: float = info(0.010)
    # Plunger length/seat chosen so its lower end NEVER reaches the nozzle: plunger and
    # nozzle are not jointed to each other, so an overlap makes PhysX fight the fixed
    # joints (measured 2026-07-14: the assembly sank 19 mm and crept sideways).
    plunger_l: float = info(0.18)
    plunger_seat: float = info(0.03)  # plunger-centre offset above barrel centre at travel 0
    plunger_mass: float = info(0.05)  # single source for assets() AND gravity comp
    ring_mass: float = info(0.01)
    flange_z: float = info(0.09)  # barrel-local height of the finger flanges
    post_h: float = info(0.26)
    post_sq: float = info(0.014)
    pocket_r: float = info(0.037)  # stand pocket radius (posts at +/- this)
    res_r: float = info(0.035)  # reservoir puck radius
    res_h: float = info(0.05)
    well_r: float = info(0.018)  # sample-well puck radius
    well_h: float = info(0.012)
    # sample wells in a row along x, plate-local:
    well_dx: tuple = info(((-0.03, -0.05), (0.05, -0.05), (0.13, -0.05)))
    res_d: tuple = info((-0.03, 0.07))  # reservoir, plate-local

    # Derived: home barrel-centre height over the surface.
    barrel_home_h: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        # Flanges rest on the post tops: barrel centre = post_h - flange_z above the
        # surface, spawned 6 mm high so the flanges settle DOWN onto the posts instead of
        # spawning interpenetrated.
        self.barrel_home_h = self.post_h - self.flange_z + 0.006


@SCENES.register("syringe")
class SyringeDosingScene(BaseScene):
    cfg: SyringeDosingSceneCfg

    def __init__(self, cfg: SyringeDosingSceneCfg | None = None) -> None:
        super().__init__(cfg or SyringeDosingSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.surface_z
        sx, sy = c.stand_pos
        px, py = c.plate_pos
        bz = z0 + c.barrel_home_h  # barrel centre at home

        white = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.92, 0.95))
        blue = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.45, 0.85))
        grey = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.50))

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

        # --- syringe stand: a COLLAR (square frame) the barrel hangs through, flanges
        # resting on its continuous top face. Nothing constrains the barrel's yaw while
        # it hangs, so the support must be yaw-symmetric — discrete posts are not:
        # measured (2026-07-14, twice) the syringe slowly yawed, the flanges rotated off
        # the posts, and the whole rig slipped through and toppled. ---
        a = 2 * c.barrel_r + 0.004  # square aperture, 2 mm clearance around the barrel
        b = 0.05  # collar bar width
        ct = 0.02  # collar thickness; top face at post_h
        cz = z0 + c.post_h - ct / 2
        for name, size, pos in (
            ("stand_yp", (a + 2 * b, b, ct), (sx, sy + a / 2 + b / 2, cz)),
            ("stand_yn", (a + 2 * b, b, ct), (sx, sy - a / 2 - b / 2, cz)),
            ("stand_xp", (b, a, ct), (sx + a / 2 + b / 2, sy, cz)),
            ("stand_xn", (b, a, ct), (sx - a / 2 - b / 2, sy, cz)),
        ):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.capitalize(),
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=grey,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
            )
        for k in range(4):  # legs at the frame corners (kinematic, mostly visual)
            lx = sx + (a / 2 + b - 0.012) * (1 if k in (0, 1) else -1)
            ly = sy + (a / 2 + b - 0.012) * (1 if k in (0, 2) else -1)
            out[f"post_{k}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Post_" + str(k),
                spawn=sim_utils.CuboidCfg(
                    size=(c.post_sq, c.post_sq, c.post_h - ct),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=grey,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(lx, ly, z0 + (c.post_h - ct) / 2)),
            )

        # --- well plate: kinematic base + solid pucks (reservoir + 3 wells) ---
        out["plate"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plate",
            spawn=sim_utils.CuboidCfg(
                size=(0.30, 0.24, 0.008),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=white,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(px + 0.05, py, z0 + 0.004)),
        )
        out["reservoir"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Reservoir",
            spawn=sim_utils.CylinderCfg(
                radius=c.res_r, height=c.res_h,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=blue,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(px + c.res_d[0], py + c.res_d[1], z0 + 0.008 + c.res_h / 2)),
        )
        for k, (dx, dy) in enumerate(c.well_dx):
            out[f"well_{k}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Well_" + str(k),
                spawn=sim_utils.CylinderCfg(
                    radius=c.well_r, height=c.well_h,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.75, 0.2)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(px + dx, py + dy, z0 + 0.008 + c.well_h / 2)),
            )

        # --- syringe: barrel (+ nozzle + 2 flanges fixed) and plunger (+ thumb ring fixed) ---
        out["barrel"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Barrel",
            spawn=sim_utils.CylinderCfg(
                radius=c.barrel_r, height=c.barrel_l,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                # translucent: the liquid column inside must be VISIBLE (the doses
                # were pure bookkeeping for 36 runs — nobody could see the water)
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.88, 0.90, 0.95), opacity=0.35),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(sx, sy, bz)),
        )
        out["nozzle"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Nozzle",
            spawn=sim_utils.CylinderCfg(
                radius=c.nozzle_r, height=c.nozzle_l,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=blue,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(sx, sy, bz - c.barrel_l / 2 - c.nozzle_l / 2)),
        )
        for k, side in enumerate((-1.0, 1.0)):
            out[f"flange_{k}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Flange_" + str(k),
                spawn=sim_utils.CuboidCfg(
                    size=(0.035, 0.020, 0.008),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=white,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(sx + side * (c.barrel_r + 0.0175), sy, bz + c.flange_z)),
            )
        # Plunger: NO collider — its rod lives inside the solid barrel bore, and
        # solid-solid depenetration is exactly what wrecked the rig (the hand pushes
        # the thumb RING, which does collide). Its motion is owned by the post_step
        # slide constraint.
        out["plunger"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plunger",
            spawn=sim_utils.CylinderCfg(
                radius=c.plunger_r, height=c.plunger_l,
                # NEVER SLEEPS: teleport-carries (velocity zeroed every step) put the
                # plunger to sleep, after which drive forces were silently ignored —
                # the v14 signature was a frozen position with a stale constant
                # velocity reading. Actuated-by-force bodies must not doze.
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    sleep_threshold=0.0, stabilization_threshold=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.plunger_mass),
                visual_material=grey,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(sx, sy, bz + c.plunger_seat)),
        )
        # Thumb ring width: MUST clear the flanges. At 0.05 wide its edges sat exactly
        # flush with the flange inner faces (x = +/-0.025) — a permanent rail contact
        # whose friction dominated every plunger motion for six smoke runs (slow creep
        # to -20mm where the ring rested ON the flange tops, 10x-viscous drag under
        # drive). 0.036 leaves 7 mm of air per side.
        out["thumb_ring"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Thumb_ring",
            spawn=sim_utils.CuboidCfg(
                size=(0.036, 0.02, 0.012),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.ring_mass),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=grey,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(sx, sy, bz + c.plunger_seat + c.plunger_l / 2 + 0.006)),
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
        n = env.num_envs
        dev = env.device
        c = self.cfg
        self.barrel: RigidObject = env.iscene["barrel"]
        self.plunger: RigidObject = env.iscene["plunger"]
        self.nozzle: RigidObject = env.iscene["nozzle"]
        self.flanges = [env.iscene["flange_0"], env.iscene["flange_1"]]
        self.ring: RigidObject = env.iscene["thumb_ring"]
        self.reservoir: RigidObject = env.iscene["reservoir"]
        self.wells = [env.iscene[f"well_{k}"] for k in range(len(c.well_dx))]
        self.plate: RigidObject = env.iscene["plate"]
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        self.plunger_drive = torch.zeros(n, device=dev)  # +N draws (plunger out)
        self._travel_prev = torch.zeros(n, device=dev)
        self._hold_at = torch.zeros(n, device=dev)  # parking-brake anchor (travel, m)
        self._hold_on = torch.ones(n, dtype=torch.bool, device=dev)
        self._acc = torch.zeros(n, device=dev)  # dose-ledger deadband accumulator
        # Draw forensics (cfg.debug_forensics): per-substep ledgers for diagnosing where
        # committed travel goes. Allocated always (cheap), updated only when enabled.
        self._dbg_pos_seated = torch.zeros(n, device=dev)
        self._dbg_pos_unseated = torch.zeros(n, device=dev)
        self._dbg_neg = torch.zeros(n, device=dev)
        self._x_led = torch.zeros(n, device=dev)  # committed-travel ledger (m)
        self._res_recent = torch.zeros(n, dtype=torch.long, device=dev)
        self._well_recent = torch.zeros(n, len(c.well_dx), dtype=torch.long, device=dev)
        self._fix_on = torch.zeros(n, dtype=torch.bool, device=dev)  # compliant fixture
        self._fix_pos = torch.zeros(n, 3, device=dev)
        self._fix_ax = torch.zeros(n, 3, device=dev)
        self._build_liquid_viz()
        self._viz_step = 0
        self._viz_shown = [None] * n  # last (liquid, dose0, dose1, dose2) rendered
        self._liquid = torch.zeros(n, device=dev)  # fraction of capacity in the barrel
        self._doses = torch.zeros(n, len(c.well_dx), device=dev)
        self._spilled = torch.zeros(n, device=dev)
        self._drawn_ok = torch.zeros(n, dtype=torch.bool, device=dev)  # draw stage passed

    def _author_joints(self) -> None:
        """Per env: prismatic barrel<->plunger (axis z, limits [0, stroke]) + fixed joints
        for the nozzle, flanges and thumb ring."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/plunger_slide")
            j.CreateBody0Rel().SetTargets([f"{base}/Barrel"])
            j.CreateBody1Rel().SetTargets([f"{base}/Plunger"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.plunger_seat))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            # MEASURED (pinned probe v10, collider-free rig): the joint coordinate
            # equals travel directly — with limits [-stroke, 0] the plunger pinned at
            # travel ~0 going UP and slid freely DOWN, i.e. allowed range was
            # [-stroke, 0] in travel. (v7's sag that framed [0, stroke] was the
            # since-removed plunger-collider fight, and v8/v9 probes were corrupted by
            # rig toppling.) Correct range for travel in [0, stroke]:
            j.CreateLowerLimitAttr(0.0)
            j.CreateUpperLimitAttr(c.stroke)
            # The o-ring parking brake IS the joint's Coulomb friction — solver-level,
            # unconditionally stable. Every explicit-force emulation of it (spring
            # latch v13-v17, clamped spring v18) limit-cycled or NaN'ed the island:
            # 200 N/m at 120 Hz on an 80 g body has no stability margin.
            from pxr import PhysxSchema

            pj = PhysxSchema.PhysxJointAPI.Apply(j.GetPrim())
            pj.CreateJointFrictionAttr(c.plunger_break_free)

            fixes = (
                ("nozzle_fix", "Barrel", "Nozzle",
                 (0.0, 0.0, -c.barrel_l / 2 - c.nozzle_l / 2)),
                ("flange0_fix", "Barrel", "Flange_0",
                 (-(c.barrel_r + 0.0175), 0.0, c.flange_z)),
                ("flange1_fix", "Barrel", "Flange_1",
                 (c.barrel_r + 0.0175, 0.0, c.flange_z)),
                ("ring_fix", "Plunger", "Thumb_ring",
                 (0.0, 0.0, c.plunger_l / 2 + 0.006)),
            )
            for name, b0, b1, lp0 in fixes:
                j = UsdPhysics.FixedJoint.Define(stage, f"{base}/{name}")
                j.CreateBody0Rel().SetTargets([f"{base}/{b0}"])
                j.CreateBody1Rel().SetTargets([f"{base}/{b1}"])
                j.CreateCollisionEnabledAttr(False)
                j.CreateLocalPos0Attr(Gf.Vec3f(*lp0))
                j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        z0 = c.surface_z
        sx, sy = c.stand_pos
        bz = z0 + c.barrel_home_h

        def put(body: RigidObject, pos) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = self.env_origins[env_ids] + torch.tensor(pos, device=dev)
            st[:, 3] = 1.0
            body.write_root_state_to_sim(st, env_ids)

        put(self.barrel, (sx, sy, bz))
        put(self.nozzle, (sx, sy, bz - c.barrel_l / 2 - c.nozzle_l / 2))
        put(self.flanges[0], (sx - (c.barrel_r + 0.0175), sy, bz + c.flange_z))
        put(self.flanges[1], (sx + (c.barrel_r + 0.0175), sy, bz + c.flange_z))
        put(self.plunger, (sx, sy, bz + c.plunger_seat))
        put(self.ring, (sx, sy, bz + c.plunger_seat + c.plunger_l / 2 + 0.006))

        # Per-episode well-plate placement jitter (a common xy offset for the plate and
        # everything on it; kinematic bodies accept root-state writes).
        px, py = c.plate_pos
        jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_jitter

        def put_j(body: RigidObject, pos) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = self.env_origins[env_ids] + torch.tensor(pos, device=dev)
            st[:, 0:2] += jit
            st[:, 3] = 1.0
            body.write_root_state_to_sim(st, env_ids)

        put_j(self.plate, (px + 0.05, py, z0 + 0.004))
        put_j(self.reservoir, (px + c.res_d[0], py + c.res_d[1], z0 + 0.008 + c.res_h / 2))
        for k, (dx, dy) in enumerate(c.well_dx):
            put_j(self.wells[k], (px + dx, py + dy, z0 + 0.008 + c.well_h / 2))

        self.plunger_drive[env_ids] = 0.0
        self._travel_prev[env_ids] = 0.0
        self._hold_at[env_ids] = 0.0
        self._hold_on[env_ids] = True
        for e in env_ids.tolist():
            self._viz_shown[e] = None
        self._acc[env_ids] = 0.0
        self._x_led[env_ids] = 0.0
        self._liquid[env_ids] = 0.0
        self._doses[env_ids] = 0.0
        self._spilled[env_ids] = 0.0
        self._drawn_ok[env_ids] = False
        # A fixture left engaged across reset would drag the re-homed barrel toward its
        # stale pre-reset capture point; the seat-debounce counters and forensic
        # ledgers are episode state too.
        self._fix_on[env_ids] = False
        self._res_recent[env_ids] = 0
        self._well_recent[env_ids] = 0
        for t in (self._dbg_pos_seated, self._dbg_pos_unseated, self._dbg_neg):
            t[env_ids] = 0.0

    # ----- geometry helpers ---------------------------------------------------------------------
    def barrel_axis(self) -> torch.Tensor:
        """The barrel's +z (plunger-out) direction in world."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(self.barrel.data.root_quat_w, ez)

    def travel(self) -> torch.Tensor:
        """SIGNED plunger travel (m): + = drawn out, 0 = seated, negative = below the
        seat. (An early clamp-at-0 here blinded three probe runs in a row — the joint
        looked 'locked' when the plunger was actually parked below the seat.) Dose
        arithmetic clamps where it consumes this."""
        ax = self.barrel_axis()
        rel = self.plunger.data.root_pos_w - self.barrel.data.root_pos_w
        return (rel * ax).sum(dim=-1) - self.cfg.plunger_seat

    def tip_pos(self) -> torch.Tensor:
        """The nozzle tip point (world)."""
        c = self.cfg
        return self.barrel.data.root_pos_w - self.barrel_axis() * (
            c.barrel_l / 2 + c.nozzle_l)

    def _seated_on(self, target_pos: torch.Tensor, rim_z: torch.Tensor) -> torch.Tensor:
        """Tip within seat_tol laterally of target, inside the vertical seating band, and
        the syringe roughly UPRIGHT (a sideways tip is not seated — bug found when the
        first smoke run dosed through a fallen-over syringe)."""
        c = self.cfg
        tip = self.tip_pos()
        lat = (tip[:, :2] - target_pos[:, :2]).norm(dim=-1)
        dz = tip[:, 2] - rim_z
        upright = self.barrel_axis()[:, 2] > 0.85
        # Lower bound is generous: the compliant fixture bobs a few mm during
        # metering and a tip a few mm INSIDE the mouth is still seated (v20 lost
        # half the draw to flicker at dz < -2 mm).
        return (lat <= c.seat_tol) & (dz >= -0.012) & (dz <= c.seat_band) & upright

    def seated_reservoir(self) -> torch.Tensor:
        r = self.reservoir.data.root_pos_w
        return self._seated_on(r, r[:, 2] + self.cfg.res_h / 2)

    def seated_well(self) -> torch.Tensor:
        """(num_envs, n_wells) bool."""
        cols = []
        for w in self.wells:
            p = w.data.root_pos_w
            cols.append(self._seated_on(p, p[:, 2] + self.cfg.well_h / 2))
        return torch.stack(cols, dim=1)

    # ----- dosing mechanics (every substep) --------------------------------------------------------
    def _build_liquid_viz(self) -> None:
        """VISIBLE WATER (visual-only, no physics — the whiteboard-ink pattern): a blue
        column inside the translucent barrel showing the drawn liquid, and a blue fill
        column on each well showing its received dose. For 36 runs the liquid was pure
        bookkeeping and the videos were unreadable — every dose is now watchable."""
        import omni.usd
        from pxr import Gf, UsdGeom

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        # Multi-env note (2026-07-17): env_1.. INHERIT env_0's subtree (cloner
        # copy_from_source=False), so env_0's viz prims already compose under every
        # env when bind() runs — AddXformOp there hard-fails ("op already exists").
        # For inherited prims we grab the EXISTING xformOp attributes instead; Set()
        # on them authors a per-env override, so per-env liquid levels still render.
        def _make(path, radius, t0, s0):
            """Define the viz cylinder at `path` and return its (translate, scale) ops.
            The existence check MUST precede Define (Define itself creates the prim)."""
            fresh = not stage.GetPrimAtPath(path).IsValid()
            geom = UsdGeom.Cylinder.Define(stage, path)
            geom.CreateRadiusAttr(radius)
            geom.CreateHeightAttr(1.0)
            geom.CreateAxisAttr("Z")
            geom.CreateDisplayColorAttr([Gf.Vec3f(0.15, 0.45, 0.95)])
            if fresh:
                xf = UsdGeom.Xformable(geom)
                t_op, s_op = xf.AddTranslateOp(), xf.AddScaleOp()
            else:
                p = geom.GetPrim()
                t_op = p.GetAttribute("xformOp:translate")
                s_op = p.GetAttribute("xformOp:scale")
            t_op.Set(t0)
            s_op.Set(s0)
            return t_op, s_op

        self._viz_barrel, self._viz_wells = [], []
        for i in range(self.env.num_envs):
            self._viz_barrel.append(_make(
                f"/World/envs/env_{i}/Barrel/liquid", c.barrel_r * 0.92,
                Gf.Vec3d(0.0, 0.0, -c.barrel_l / 2 + 0.001), Gf.Vec3f(1.0, 1.0, 0.002)))
            self._viz_wells.append([
                _make(f"/World/envs/env_{i}/Well_{k}/fill", c.well_r * 0.95,
                      Gf.Vec3d(0.0, 0.0, c.well_h / 2 + 0.001), Gf.Vec3f(1.0, 1.0, 0.002))
                for k in range(len(self.wells))])

    def _update_liquid_viz(self) -> None:
        from pxr import Gf, Vt

        c = self.cfg
        self._viz_step += 1
        if self._viz_step % 10 != 0:
            return
        for e in range(self.env.num_envs):
            liq = float(self._liquid[e])
            doses = [float(v) for v in self._doses[e]]
            key = (round(liq, 3), *(round(d, 3) for d in doses))
            if key == self._viz_shown[e]:
                continue
            self._viz_shown[e] = key
            # barrel column: liquid fraction of the interior length
            h = max(liq, 0.002) * c.barrel_l * 0.85
            t_op, s_op = self._viz_barrel[e]
            t_op.Set(Gf.Vec3d(0.0, 0.0, -c.barrel_l / 2 + 0.001 + h / 2))
            s_op.Set(Gf.Vec3f(1.0, 1.0, h))
            # well fills: a full 1/3 dose reads as a ~33 mm column above the rim
            for k, (tw, sw) in enumerate(self._viz_wells[e]):
                hw = max(doses[k], 0.002) * 0.14
                tw.Set(Gf.Vec3d(0.0, 0.0, c.well_h / 2 + 0.001 + hw / 2))
                sw.Set(Gf.Vec3f(1.0, 1.0, hw))

    def post_step(self) -> None:
        c = self.cfg
        dev = self.env.device
        n = self.env.num_envs

        x = self.travel()
        dx_raw = x - self._travel_prev
        self._travel_prev = x
        # DEADBAND accumulator: commit travel to the dose ledger in 2 mm quanta so
        # sub-mm vibration can't pump liquid in/out of existence (v22).
        self._acc = self._acc + dx_raw
        commit = torch.where(self._acc.abs() > 0.002, self._acc,
                             torch.zeros_like(self._acc))
        self._acc = self._acc - commit
        dx = commit
        self._x_led = self._x_led + commit

        frac = dx / c.stroke
        drawing = frac > 0
        # DEBOUNCED seat flags for the dose ledger: the instantaneous flag flickers
        # for single substeps mid-solve while the loop-level duty cycle is 100%
        # (v26 forensics: 37 mm of a 115 mm fully-seated draw discounted). Seated
        # within the last 10 substeps counts; >10 substeps unseated = drawing air,
        # exactly the task's semantics.
        seated_res_now = self.seated_reservoir()
        self._res_recent = torch.where(
            seated_res_now, torch.full_like(self._res_recent, 10),
            (self._res_recent - 1).clamp(min=0))
        seated_res = self._res_recent > 0
        seated_w_now = self.seated_well()
        self._well_recent = torch.where(
            seated_w_now, torch.full_like(self._well_recent, 10),
            (self._well_recent - 1).clamp(min=0))
        seated_w = self._well_recent > 0

        # Draw: gain liquid only while seated on the reservoir (else it's air). Liquid
        # can never exceed the committed drawn volume: the GAIN is capped by the
        # committed-travel ledger (capping the running total instead double-deducted
        # the first push quantum after a full draw — once by the cap, once by the
        # dispense below). The ledger cap itself exists because raw sub-quantum dips
        # (pulsed metering ripple) ratcheted the liquid away ~1 mm at a time while
        # the deadband ledger never saw a dispense (v24).
        gain = torch.where(drawing & seated_res, frac, torch.zeros(n, device=dev))
        gain_cap = (self._x_led.clamp(min=0.0) / c.stroke + 1e-6 - self._liquid).clamp(min=0.0)
        gain = torch.minimum(gain, gain_cap)
        if c.debug_forensics:  # draw-loss diagnosis: where does committed travel go?
            self._dbg_pos_seated = self._dbg_pos_seated + torch.where(
                drawing & seated_res, dx, torch.zeros_like(dx))
            self._dbg_pos_unseated = self._dbg_pos_unseated + torch.where(
                drawing & ~seated_res, dx, torch.zeros_like(dx))
            self._dbg_neg = self._dbg_neg + torch.where(
                dx < 0, dx, torch.zeros_like(dx))
        self._liquid = self._liquid + gain
        # Draw stage passes on actual LIQUID, not travel: pulling air off-well and then
        # tapping the reservoir must not count as a draw.
        self._drawn_ok = self._drawn_ok | (seated_res & (self._liquid >= c.draw_min))

        # Dispense: liquid leaves first; it lands in the seated well or is spilled.
        push = (-frac).clamp(min=0.0)
        out_liq = torch.minimum(push, self._liquid)
        self._liquid = self._liquid - out_liq
        any_w = seated_w.any(dim=1)
        widx = seated_w.float().argmax(dim=1)
        add = torch.zeros(n, len(self.wells), device=dev)
        add[torch.arange(n, device=dev), widx] = torch.where(
            any_w, out_liq, torch.zeros(n, device=dev))
        self._doses = self._doses + add
        self._spilled = self._spilled + torch.where(any_w, torch.zeros(n, device=dev), out_liq)

        # Plunger o-ring along the barrel axis, applied as an INTERNAL action-reaction
        # pair (equal and opposite on plunger and barrel — the pair can never lift or
        # sink the assembly as a whole). While the external drive is below break_free
        # the o-ring position-locks the plunger to its anchor; a firm push/pull
        # releases the lock (anchor re-latches where the drive stops). The drive force
        # itself is one-sided by design: it stands in for a hand pushing the ring.
        ax = self.barrel_axis()
        v = ((self.plunger.data.root_lin_vel_w - self.barrel.data.root_lin_vel_w) * ax).sum(dim=-1)
        # Parking brake: the authored PhysX jointFriction is a NO-OP for joints
        # outside articulations (runtime warning "Joint friction attribute is only
        # applied for joints in articulations") — so the brake must be explicit. A
        # GENTLE clamped position lock is stable (k*dt^2/m ~ 0.07, unlike the 200 N/m
        # spring that limit-cycled in v13-v17): engages when the drive backs off,
        # holds the 15% uncompensated weight, and kills the post-meter coast that
        # overdosed every well by ~0.06 (v20-v36).
        engaged = self.plunger_drive.abs() < 0.3
        newly = engaged & ~self._hold_on
        self._hold_at = torch.where(newly, x, self._hold_at)
        self._hold_on = engaged
        brake = (50.0 * (self._hold_at - x) - 1.0 * v).clamp(-1.2, 1.2)
        visc = (-c.plunger_visc * v).clamp(-3.0, 3.0)
        oring = torch.where(engaged, brake + visc, visc)
        f_plunger = self.plunger_drive + oring
        self._dbg_force = f_plunger  # debug: last applied axial force (N)
        m_moving = c.plunger_mass + c.ring_mass  # the fixed-jointed moving pair
        comp = torch.zeros(n, 3, device=dev)
        # 85% compensation: a small net downward bias keeps the parked plunger parked
        # (full comp made it neutrally buoyant and it crept off the seat at boot, v21).
        comp[:, 2] = 0.85 * m_moving * 9.81
        # All force vectors here are WORLD-frame (ax is the world barrel axis, comp is
        # world-up): is_global=True. Body-local application (the default) is only
        # correct while the barrel quat is near identity — a yawed/tilted barrel would
        # get a rotated force.
        self.plunger.set_external_force_and_torque(
            (f_plunger.view(n, 1, 1) * ax.view(n, 1, 3) + comp.view(n, 1, 3)),
            torch.zeros(n, 1, 3, device=dev), is_global=True)

        # Barrel = o-ring reaction + optional COMPLIANT FIXTURE (a PD "hand" holding
        # the barrel where fixture() captured it). The fixture replaced the smoke's
        # per-step teleport pin during metering: teleporting a jointed body against a
        # driven plunger ignites a violent solver limit-cycle (v16: barrel z thrashing
        # +-5 mm, 48 N phantom viscous spikes, plunger travel bouncing 30-70 mm).
        # While the fixture holds, the o-ring reaction is ROUTED TO GROUND through
        # the "hand" instead of shaking the barrel: the +-3 N clamped viscous chatter
        # resonated the compliant fixture (9 Hz, ~10 mm) and the seat flag + travel
        # ledger thrashed in sync (v26/v27 forensics: 219 mm of "unseated" travel on
        # a 115 mm draw).
        f_bar = torch.where(self._fix_on.view(n, 1),
                            torch.zeros(n, 3, device=dev),
                            (-oring).view(n, 1) * ax)
        tq_bar = torch.zeros(n, 3, device=dev)
        if bool(self._fix_on.any()):
            bp = self.barrel.data.root_pos_w
            bv = self.barrel.data.root_lin_vel_w
            bw = self.barrel.data.root_ang_vel_w
            # explicit-force stability: kp*dt^2/m ~ 0.22, kd*dt/m ~ 0.4 (v22's
            # 1500/40 vibrated the barrel and the seat flag flickered in phase
            # with dx — the draw registered 1% liquid at full travel)
            f_fix = 800.0 * (self._fix_pos - bp) - 12.0 * bv
            # keep upright: torque steers the barrel axis back to the captured axis
            tq_fix = 3.0 * torch.cross(ax, self._fix_ax, dim=-1) - 0.15 * bw
            on = self._fix_on.view(n, 1).float()
            f_bar = f_bar + on * f_fix
            tq_bar = on * tq_fix
        # World-frame forces/torques (PD on world positions, cross of world axes):
        # is_global=True, same reasoning as the plunger force above.
        self.barrel.set_external_force_and_torque(
            f_bar.view(n, 1, 3), tq_bar.view(n, 1, 3), is_global=True)
        self._update_liquid_viz()

    def fixture(self, on: bool = True, env_ids: torch.Tensor | None = None) -> None:
        """Engage/release the compliant barrel fixture at the CURRENT pose (the
        stand-in for a steady hand: an 800 N/m / 12 N*s/m PD plus an uprighting
        torque). Metering must run with the fixture, never with kinematic re-pinning."""
        ids = slice(None) if env_ids is None else env_ids
        self._fix_on[ids] = bool(on)
        if on:
            self._fix_pos[ids] = self.barrel.data.root_pos_w[ids]
            self._fix_ax[ids] = self.barrel_axis()[ids]

    # ----- state ---------------------------------------------------------------------------------
    # Every episode-relevant tensor, so a restored snapshot behaves exactly like the
    # moment it was taken (an earlier version missed the travel/hold/fixture state and
    # the ledger cap silently emptied a restored full draw on the next substep).
    _STATE_KEYS = ("_travel_prev", "_liquid", "_doses", "_spilled", "_drawn_ok",
                   "plunger_drive", "_hold_at", "_hold_on", "_acc", "_x_led",
                   "_res_recent", "_well_recent", "_fix_on", "_fix_pos", "_fix_ax")

    def _bodies(self) -> dict[str, RigidObject]:
        out = {"barrel": self.barrel, "plunger": self.plunger, "nozzle": self.nozzle,
               "flange_0": self.flanges[0], "flange_1": self.flanges[1], "ring": self.ring,
               "plate": self.plate, "reservoir": self.reservoir}
        for k, w in enumerate(self.wells):
            out[f"well_{k}"] = w
        return out

    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {n: b.data.root_state_w[env_ids].clone()
                       for n, b in self._bodies().items()},
            "dose": {k: getattr(self, k)[env_ids].clone() for k in self._STATE_KEYS},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for n, b in self._bodies().items():
            b.write_root_state_to_sim(state["bodies"][n], env_ids)
        for k, v in state["dose"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        lo, hi = c.dose_band
        return (
            f"A large syringe stands upright in a 4-post stand: a white barrel (radius "
            f"{c.barrel_r:.3f} m) with two finger-loop flanges, a grey plunger with a thumb "
            f"ring on top (travel {c.stroke:.2f} m = its full capacity), and a blue nozzle "
            f"pointing down. Nearby, a well plate: one wide blue RESERVOIR full of liquid "
            f"and three small yellow SAMPLE wells in a row.\n"
            f"Liquid moves with the plunger, but ONLY while the nozzle tip is seated on a "
            f"well: hovering within {c.seat_tol * 1000:.0f} mm laterally and {c.seat_band * 1000:.0f} "
            f"mm above its rim. Pulling the plunger while seated on the reservoir draws "
            f"liquid; pushing it dispenses into whichever well the tip is seated on — "
            f"anywhere else the liquid is WASTED. Wells never un-fill: a well pushed past "
            f"its band is ruined for the episode (the reservoir can be re-drawn from, but "
            f"over-dispensing cannot be undone). The water is VISIBLE: a blue column "
            f"inside the translucent barrel shows the drawn liquid, and each well shows "
            f"its received dose as a blue fill column. The plunger has deliberate "
            f"friction: push/pull it slowly and steadily.\n"
            f"Goal: draw a full load (>= {c.draw_min:.0%} of capacity, tip seated in the "
            f"reservoir), dispense between {lo:.0%} and {hi:.0%} of capacity into EACH of "
            f"the three sample wells, then park the syringe upright back in its stand."
        )

    # ----- progress -------------------------------------------------------------------------------
    def liquid(self) -> torch.Tensor:
        return self._liquid.clone()

    def doses(self) -> torch.Tensor:
        """(num_envs, 3) dispensed fraction of capacity per sample well."""
        return self._doses.clone()

    def spilled(self) -> torch.Tensor:
        return self._spilled.clone()

    def drawn(self) -> torch.Tensor:
        return self._drawn_ok.clone()

    def doses_ok(self) -> torch.Tensor:
        lo, hi = self.cfg.dose_band
        return ((self._doses >= lo) & (self._doses <= hi)).all(dim=1)

    def parked(self) -> torch.Tensor:
        """Syringe back in the stand: over the pocket, upright, settled."""
        c = self.cfg
        p = self.barrel.data.root_pos_w - self.env_origins
        sx, sy = c.stand_pos
        near = (p[:, :2] - torch.tensor([sx, sy], device=self.env.device)).norm(dim=-1) < 0.03
        upright = self.barrel_axis()[:, 2] > 0.9
        settled = self.barrel.data.root_lin_vel_w.norm(dim=-1) < 0.05
        return near & upright & settled

    def success(self) -> torch.Tensor:
        return self.drawn() & self.doses_ok() & self.parked()


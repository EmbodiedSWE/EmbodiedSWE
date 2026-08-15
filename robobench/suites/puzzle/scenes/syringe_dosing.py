"""SyringeDosingScene — draw a full syringe, dispense exactly one third into each of 3 tubes.

The object world: a scanned syringe (translucent barrel with flanges and luer tip +
prismatic plunger with thumb plate — two vendored rigid bodies from the A23D asset)
LYING FLAT on the shelf, no stand; a rack of three GLASS TEST TUBES (the sample wells)
and a wide open RESERVOIR beaker — all set up on the top shelf of a MEDICAL CART
(vendored scan: two shelves, guard rails, peel-packed syringes and a basin as authentic
decor; wheels welded — a parked cart, it must not roll under the arm).
The cart's own packaged syringes are scenery only: they are sealed flat peel-packs fused
into the cart body, with no plunger geometry and an 8 mm barrel far below the metering
mechanic's tolerances — the dosing instrument is this scene's articulated syringe.
**Goal (carried here, no task layer): draw a full load from the reservoir, then dispense
33% +/- 10% of capacity into each tube, and lay the syringe back down at its spot.**

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
    dose_band: tuple = tunable((0.23, 0.43))  # per-tube acceptance band: 33% +/- 10%
    # of capacity (relaxed from +/-5% — the tight band priced in kinematic-oracle
    # metering precision, not an arm pressing the plunger through o-ring friction)
    draw_min: float = tunable(0.95)  # min drawn fraction for the draw stage
    # O-ring feel: the prismatic joint's Coulomb friction + viscous damping (the
    # plunger is a REAL body now — an arm hooks the thumb plate and pulls). The
    # friction holds the 0.06 kg plunger wherever it lands, including vertical.
    plunger_friction: float = tunable(1.5)  # joint Coulomb friction (N)
    plunger_damping: float = tunable(4.0)  # joint viscous damping (N*s/m)
    reset_jitter: float = tunable(0.015)  # +/- xy jitter of the well plate per episode (m)
    debug_forensics: bool = tunable(False)  # per-substep draw-loss ledgers (smoke diagnosis only)

    # --- tunable: placement -------------------------------------------------------------------
    surface_z: float = tunable(0.0)  # 0 + medical_cart -> auto-set to the shelf top
    # Cart-shelf layout: everything in the shelf's free WEST half (the east half is
    # occupied by the fused cloth/instrument decor, the two vials sit at
    # (-0.126/-0.066, -0.046) and have colliders). No stand: the syringe LIES FLAT
    # on the shelf at home_pos (tip west, thumb east) and is parked by laying it
    # back down there.
    home_pos: tuple = tunable((-0.06, -0.08))  # lying syringe centre on the surface.
    # The syringe lies ALONG X (tip west, plate east): a y-lying syringe forces
    # its plate to y>=+0.09 where every wrist column hits the north guard rail —
    # unreachable by both arms in all orientation families (jobs 19-27,
    # 2026-08-11). Lying along x keeps the whole syringe south of the rail wall.
    rack_pos: tuple = tunable((0.0, 0.14))  # tube-rack centre (north strip, east
    # of the home column; the 1.02 m glass tubes must sit outside both arms'
    # LOW-reach sweeps — park carries dogleg south of them)
    res_pos: tuple = tunable((0.14, -0.05))  # reservoir beaker centre: EAST of
    # the home footprint (17 mm clear of the lying plate end — at x=0.02 the
    # beaker overlapped the boot pose) at the PROVEN south latitude (y=+0.03
    # sat in the rack shadow that kills the two-hand seat, job_0038/pod-8;
    # x=-0.16 jammed the left hand past its own base, job_0034). Warm-pod
    # validated: boot rest clean, grasp 8.0 mm, seat corridor 2.8 mm.

    # --- info: workbench ------------------------------------------------------------------------
    workbench: str = info("medical_cart")  # "medical_cart" (vendored cart) | "bench" (slab)
    cart_shelf_z: float = info(0.7492)  # cart top-shelf surface (vendor measurement)
    bench_size: tuple = info((1.1, 0.9))
    # --- info: arm-side table (along the cart's NORTH long side, below the shelf) --------------
    # The cart's shelf has no room for robot bases: the packing table (the chair /
    # tool-packing general-purpose bench) runs along the cart's long side, and the
    # bimanual arms mount spread out on its cart-side edge, working the shelf from
    # above the north guard rail.
    side_table: bool = info(True)
    table_pos: tuple = info((0.0, 0.65))  # table centre (long axis along x)
    table_top_z: float = info(0.71)  # tabletop height (~4 cm below the shelf)
    table_height: float = info(0.994)  # packing-table intrinsic top height (scale 0.01)

    # --- info: tube rack (vendor measurements, rack frame) --------------------------------------
    tube_mouths: tuple = info(((-0.094, -0.036), (0.0, -0.036), (0.094, -0.036)))
    tube_rim_dz: float = info(0.2724)  # mouth rim above the rack origin
    tube_inner_r: float = info(0.017)
    # --- info: syringe (A23D scan, vendor measurements — barrel + plunger USDs) ----------------
    barrel_r: float = info(0.030)
    barrel_l: float = info(0.2222)
    nozzle_l: float = info(0.0288)  # luer tip below the barrel tube bottom (no needle:
    # the scan's hypodermic was a loose prop staged beside the syringe, dropped)
    stroke: float = info(0.12)  # plunger travel = syringe capacity (geometric max 0.18)
    plunger_l: float = info(0.1813)
    plunger_seat: float = info(-0.0084)  # plunger-centre offset along the barrel axis
    # at travel 0 (fully pushed): the rod is buried, its centre just below the barrel's
    plunger_mass: float = info(0.06)  # single source for assets() AND gravity comp
    res_r: float = info(0.035)  # reservoir beaker radius
    res_h: float = info(0.05)
    well_r: float = info(0.018)  # seat-target radius at each tube mouth

    # Derived: lying barrel-centre height over the surface (spawned slightly high on
    # the flange radius; it settles onto shelf contact in the first steps).
    barrel_home_h: float = field(default=None, init=False)
    asset_dir: str = info("")

    def __post_init__(self) -> None:
        from pathlib import Path

        self.barrel_home_h = 0.045
        self.asset_dir = self.asset_dir or str(
            Path(__file__).resolve().parents[1] / "assets")
        if self.workbench == "medical_cart" and self.surface_z == 0.0:
            self.surface_z = self.cart_shelf_z


@SCENES.register("syringe")
class SyringeDosingScene(BaseScene):
    cfg: SyringeDosingSceneCfg

    def __init__(self, cfg: SyringeDosingSceneCfg | None = None) -> None:
        super().__init__(cfg or SyringeDosingSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        from pathlib import Path

        c = self.cfg
        z0 = c.surface_z
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
        if c.workbench == "medical_cart":
            # STATIC colliders (AssetBaseCfg, no rigid body): a kinematic rigid
            # cart that nobody writes goes DORMANT once nothing touches it and
            # then generates no new contact pairs (GPU pipeline) — the parked
            # syringe fell through the shelf after any airborne phase (bisected
            # 2026-08-10). Static colliders cannot sleep. The vendored USD
            # authors no RigidBodyAPI.
            out["cart"] = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Cart",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(Path(c.asset_dir) / "medical_cart/medical_cart.usd"),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            )
        if c.workbench == "medical_cart" and c.side_table:
            # the general-purpose packing table (chair-assembly convention), SUNK so
            # its 0.994 m top lands at table_top_z — legs disappear into the floor,
            # the proven pattern for mounting this bench at arbitrary heights.
            # AssetBaseCfg + kinematic spawn props (the tool_packing pattern):
            # RigidObjectCfg fails to RESOLVE a root rigid body on this USD.
            table_spawn = sim_utils.UsdFileCfg(
                usd_path=str(
                    Path(c.asset_dir).parents[1] / "assembly/assets/props/"
                    "packing_table/SM_HeavyDutyPackingTable_C02_01_physics.usd"),
                scale=(0.01, 0.01, 0.01),
            )
            table_spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True)
            out["side_table"] = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Side_table",
                spawn=table_spawn,
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(c.table_pos[0], c.table_pos[1],
                         c.table_top_z - c.table_height)),
            )
        elif z0 > 0:
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

        # --- sample wells: the 3-tube glass rack (kinematic, tubes ARE the wells) ---
        rx, ry = c.rack_pos
        out["rack"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Rack",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(Path(c.asset_dir) / "tube_rack/tube_rack.usd"),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(rx, ry, z0)),
        )
        # --- reservoir: a wide open beaker beside the rack (NEUTRAL container;
        # the water is a separate wrap band authored in _build_liquid_viz whose
        # level DROPS as liquid is drawn — a solid blue cylinder read as a beaker
        # that never empties) ---
        beaker = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.75, 0.78))
        out["reservoir"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Reservoir",
            spawn=sim_utils.CylinderCfg(
                radius=c.res_r, height=c.res_h,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=beaker,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.res_pos[0], c.res_pos[1], z0 + c.res_h / 2)),
        )

        # --- syringe: a TWO-LINK ARTICULATION (vendored syringe_artic.usd) —
        # barrel root link + dynamic plunger on a PRISMATIC joint [0, stroke].
        # The plunger is a real body with a rod + thumb-plate collider: a second
        # arm hooks under the plate and PULLS to draw (the earlier kinematic
        # plunger metered through a scene scalar, which made the pressing arm
        # cosmetic — user directive 2026-08-10: the fill must be physical).
        # Joint friction (actuator cfg + reset-time PhysX write, the toolbox-door
        # pattern) makes the plunger stay wherever it lands — the o-ring feel.
        # sleep_threshold 0 on the root: a stationary-held body whose broadphase
        # entry goes dormant falls through everything on release (bisected
        # 2026-08-10 on the rigid barrel; same class applies to articulations).
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg

        lie = (0.5, -0.5, 0.5, -0.5)  # body +z (thumb) -> world +x (tip WEST)
        hx, hy = c.home_pos
        bz = z0 + c.barrel_home_h
        out["syringe"] = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Syringe",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(Path(c.asset_dir) / "syringe/syringe_artic.usd"),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(sleep_threshold=0.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    sleep_threshold=0.0,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(hx, hy, bz), rot=lie, joint_pos={"joint_plunger": 0.0}),
            actuators={
                "plunger": ImplicitActuatorCfg(
                    joint_names_expr=["joint_plunger"],
                    stiffness=0.0,
                    damping=c.plunger_damping,  # viscous o-ring feel under pull
                    friction=c.plunger_friction,  # Coulomb: holds it anywhere
                    effort_limit_sim=40.0,
                ),
            },
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
        self.syringe = env.iscene["syringe"]  # two-link articulation (barrel root)
        self.reservoir: RigidObject = env.iscene["reservoir"]
        self.rack: RigidObject = env.iscene["rack"]
        # cart is STATIC scenery (AssetBaseCfg) — no RigidObject handle, and it
        # must not be in the kinematic-refresh loop
        self.n_wells = len(c.tube_mouths)
        self.env_origins = env.iscene.env_origins
        self._plunger_body = self.syringe.find_bodies(["plunger"])[0][0]
        self._joint_id = self.syringe.find_joints(["joint_plunger"])[0]
        self.plunger_drive = torch.zeros(n, device=dev)  # +N draws (plunger out)
        self._travel_prev = torch.zeros(n, device=dev)
        self._dbg_force = torch.zeros(n, device=dev)  # last commanded drive (N)
        self._acc = torch.zeros(n, device=dev)  # dose-ledger deadband accumulator
        # Draw forensics (cfg.debug_forensics): per-substep ledgers for diagnosing where
        # committed travel goes. Allocated always (cheap), updated only when enabled.
        self._dbg_pos_seated = torch.zeros(n, device=dev)
        self._dbg_pos_unseated = torch.zeros(n, device=dev)
        self._dbg_neg = torch.zeros(n, device=dev)
        self._x_led = torch.zeros(n, device=dev)  # committed-travel ledger (m)
        self._res_recent = torch.zeros(n, dtype=torch.long, device=dev)
        self._well_recent = torch.zeros(n, len(c.tube_mouths), dtype=torch.long, device=dev)
        self._fix_on = torch.zeros(n, dtype=torch.bool, device=dev)  # compliant fixture
        self._fix_pos = torch.zeros(n, 3, device=dev)
        self._fix_ax = torch.zeros(n, 3, device=dev)
        self._build_liquid_viz()
        self._viz_step = 0
        self._viz_shown = [None] * n  # last (liquid, dose0, dose1, dose2) rendered
        self._liquid = torch.zeros(n, device=dev)  # fraction of capacity in the barrel
        self._doses = torch.zeros(n, len(c.tube_mouths), device=dev)
        self._spilled = torch.zeros(n, device=dev)
        # net liquid TAKEN from the reservoir (syringe-capacity units) — drives the
        # beaker's visible water level (one full draw drops it a quarter)
        self._res_taken = torch.zeros(n, device=dev)
        self._drawn_ok = torch.zeros(n, dtype=torch.bool, device=dev)  # draw stage passed

    # (Visuals and colliders — tube, flange box, luer capsule, plunger rod and
    # thumb plate — are all authored in the vendored syringe_artic.usd. The
    # barrel<->plunger pair is filtered by the articulation's self-collision
    # setting; the prismatic joint IS the metering mechanism now: a second arm
    # hooks the plate and pulls, joint friction holds it wherever it lands.)

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        z0 = c.surface_z
        hx, hy = c.home_pos
        bz = z0 + c.barrel_home_h
        lie = (0.5, -0.5, 0.5, -0.5)  # body +z (thumb) -> world +x (tip WEST)

        # lying home: tip west, thumb east; joint 0 = plunger fully pushed (its
        # rod still stands proud of the rim — the hook-grasp handle). Spawned a
        # touch high, settles onto shelf contact.
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = self.env_origins[env_ids] + torch.tensor(
            (hx, hy, bz), device=dev)
        st[:, 3:7] = torch.tensor(lie, device=dev)
        self.syringe.write_root_state_to_sim(st, env_ids)
        jp = torch.zeros(m, 1, device=dev)
        self.syringe.write_joint_state_to_sim(jp, jp.clone(), env_ids=env_ids)
        self.syringe.set_joint_effort_target(jp.clone(), env_ids=env_ids)
        # Joint friction + damping THROUGH THE PHYSX VIEW: the actuator cfg's
        # values never reach PhysX for this articulation (first articulated run:
        # the plunger slammed limit-to-limit at 1.76 m/s under a 1.5 N drive —
        # zero effective friction OR damping). Same pathology and same fix as the
        # toolbox doors (2026-08-07): write dof friction/damping at reset via
        # root_physx_view with CPU tensors.
        n_all = self.env.num_envs
        view = self.syringe.root_physx_view
        all_cpu = torch.arange(n_all, device="cpu", dtype=torch.int32)
        view.set_dof_friction_coefficients(
            torch.full((n_all, 1), c.plunger_friction, device="cpu"), all_cpu)
        view.set_dof_dampings(
            torch.full((n_all, 1), c.plunger_damping, device="cpu"), all_cpu)
        try:  # verify the values actually landed in PhysX (print once per reset)
            fr_live = view.get_dof_friction_coefficients()
            dp_live = view.get_dof_dampings()
            print(f"[syringe-scene] joint live: friction="
                  f"{float(fr_live.reshape(-1)[0]):.2f} "
                  f"damping={float(dp_live.reshape(-1)[0]):.2f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[syringe-scene] joint param readback failed: {exc!r}",
                  flush=True)

        # Per-episode placement jitter: the rack and the reservoir each get their own
        # xy offset (they are independent pieces of labware on the shelf).
        def put_j(body: RigidObject, pos) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = self.env_origins[env_ids] + torch.tensor(pos, device=dev)
            st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_jitter
            st[:, 3] = 1.0
            body.write_root_state_to_sim(st, env_ids)

        put_j(self.rack, (c.rack_pos[0], c.rack_pos[1], z0))
        put_j(self.reservoir, (c.res_pos[0], c.res_pos[1], z0 + c.res_h / 2))

        self.plunger_drive[env_ids] = 0.0
        self._travel_prev[env_ids] = 0.0
        for e in env_ids.tolist():
            self._viz_shown[e] = None
        self._acc[env_ids] = 0.0
        self._x_led[env_ids] = 0.0
        self._liquid[env_ids] = 0.0
        self._doses[env_ids] = 0.0
        self._spilled[env_ids] = 0.0
        self._res_taken[env_ids] = 0.0
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
        return quat_apply(self.syringe.data.root_quat_w, ez)

    def travel(self) -> torch.Tensor:
        """Plunger travel (m): + = drawn out, 0 = fully pushed. MEASURED — the
        prismatic joint position (a second arm pulling the plate moves it; joint
        friction holds it wherever it lands)."""
        return self.syringe.data.joint_pos[:, self._joint_id[0]].clone()

    def tip_pos(self) -> torch.Tensor:
        """The nozzle tip point (world)."""
        c = self.cfg
        return self.syringe.data.root_pos_w - self.barrel_axis() * (
            c.barrel_l / 2 + c.nozzle_l)

    def plunger_top_w(self) -> torch.Tensor:
        """Thumb-plate centre (world) — the hook-grasp handle on the plunger link."""
        return (self.syringe.data.body_pos_w[:, self._plunger_body]
                + self.barrel_axis() * (self.cfg.plunger_l / 2))

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

    def well_mouths(self) -> torch.Tensor:
        """(num_envs, n_wells, 3) tube mouth centres in world (rim plane)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        rp = self.rack.data.root_pos_w
        rq = self.rack.data.root_quat_w
        out = []
        for mx, my in c.tube_mouths:
            off = torch.tensor([mx, my, c.tube_rim_dz], device=self.env.device)
            out.append(rp + quat_apply(rq, off.expand(rp.shape[0], 3)))
        return torch.stack(out, dim=1)

    def seated_well(self) -> torch.Tensor:
        """(num_envs, n_wells) bool: tip seated on a tube mouth."""
        mouths = self.well_mouths()
        cols = [self._seated_on(mouths[:, k], mouths[:, k, 2])
                for k in range(self.n_wells)]
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

        self._viz_barrel, self._viz_wells, self._viz_res = [], [], []
        # tube fill indicators WRAP the tubes (radius just outside the glass): a
        # rising liquid-level band readable at any glass opacity (interior columns
        # were invisible behind the frosted glass)
        self._fill_base = c.tube_rim_dz - 0.165
        for i in range(self.env.num_envs):
            # the barrel level wraps OUTSIDE the (opaque) visual tube, like the
            # tube fill bands — an interior column would be hidden
            self._viz_barrel.append(_make(
                f"/World/envs/env_{i}/Syringe/barrel/liquid", c.barrel_r + 0.003,
                Gf.Vec3d(0.0, 0.0, -c.barrel_l / 2 + 0.001), Gf.Vec3f(1.0, 1.0, 0.002)))
            self._viz_wells.append([
                _make(f"/World/envs/env_{i}/Rack/fill_{k}", 0.0208,
                      Gf.Vec3d(mx, my, self._fill_base), Gf.Vec3f(1.0, 1.0, 0.002))
                for k, (mx, my) in enumerate(c.tube_mouths)])
            # beaker water: wrap band whose TOP EDGE is the live level (starts
            # full; a full syringe draw drops it a quarter of the beaker)
            h0 = c.res_h * 0.9
            self._viz_res.append(_make(
                f"/World/envs/env_{i}/Reservoir/water", c.res_r + 0.003,
                Gf.Vec3d(0.0, 0.0, -c.res_h / 2 + h0 / 2), Gf.Vec3f(1.0, 1.0, h0)))

    def _update_liquid_viz(self) -> None:
        from pxr import Gf, Vt

        c = self.cfg
        self._viz_step += 1
        if self._viz_step % 10 != 0:
            return
        for e in range(self.env.num_envs):
            liq = float(self._liquid[e])
            doses = [float(v) for v in self._doses[e]]
            taken = float(self._res_taken[e])
            key = (round(liq, 3), round(taken, 3), *(round(d, 3) for d in doses))
            if key == self._viz_shown[e]:
                continue
            self._viz_shown[e] = key
            # barrel column: liquid fraction of the interior length
            h = max(liq, 0.002) * c.barrel_l * 0.85
            t_op, s_op = self._viz_barrel[e]
            t_op.Set(Gf.Vec3d(0.0, 0.0, -c.barrel_l / 2 + 0.001 + h / 2))
            s_op.Set(Gf.Vec3f(1.0, 1.0, h))
            # tube fills: a full 1/3 dose reads as a ~47 mm column inside the glass
            for k, (tw, sw) in enumerate(self._viz_wells[e]):
                hw = max(doses[k], 0.002) * 0.14
                mx, my = c.tube_mouths[k]
                tw.Set(Gf.Vec3d(mx, my, self._fill_base + hw / 2))
                sw.Set(Gf.Vec3f(1.0, 1.0, hw))
            # beaker water: one full syringe draw drains a quarter of the beaker
            hr = c.res_h * 0.9 * max(1.0 - 0.25 * taken, 0.06)
            tr, sr = self._viz_res[e]
            tr.Set(Gf.Vec3d(0.0, 0.0, -c.res_h / 2 + hr / 2))
            sr.Set(Gf.Vec3f(1.0, 1.0, hr))

    def post_step(self) -> None:
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        n = self.env.num_envs
        all_ids = torch.arange(n, device=dev)

        bp_f = self.syringe.data.root_pos_w
        bv_f = self.syringe.data.root_lin_vel_w
        bw_f = self.syringe.data.root_ang_vel_w
        ax = self.barrel_axis()

        # --- plunger_drive: a REAL joint effort now (N along the prismatic axis;
        # + draws the plunger out). The o-ring feel comes from the joint's
        # Coulomb friction + damping (actuator cfg), not a scene-side rate law —
        # an arm hooking the plate and pulling moves the SAME joint.
        self.syringe.set_joint_effort_target(
            self.plunger_drive.view(n, 1), env_ids=all_ids)
        self._dbg_force = self.plunger_drive  # debug surface: last commanded drive

        # Kinematic-target refresh for the rack + reservoir (they stay kinematic

        # Kinematic-target refresh for the rack + reservoir (they stay kinematic
        # rigid bodies because reset jitters their poses): a kinematic actor that
        # is never written FALLS ASLEEP once nothing touches it, and a sleeping
        # kinematic actor on the GPU pipeline generates NO new contact pairs
        # (bisected 2026-08-10 on the cart, which is now STATIC instead). The
        # no-op pose write each step is the wake signal PhysX expects.
        for body in (self.rack, self.reservoir):
            st = body.data.root_state_w.clone()
            st[:, 7:13] = 0.0
            body.write_root_state_to_sim(st, all_ids)

        x = self.travel().clamp(0.0, c.stroke)
        dx_raw = x - self._travel_prev
        self._travel_prev = x
        # DEADBAND accumulator: commit travel to the dose ledger in 0.5 mm quanta.
        # (2 mm guarded the old force-driven rig's vibration; the scene-owned travel
        # is exact and monotone under drive, so the quantum only sets ledger lag.)
        self._acc = self._acc + dx_raw
        commit = torch.where(self._acc.abs() > 0.0005, self._acc,
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
        add = torch.zeros(n, self.n_wells, device=dev)
        add[torch.arange(n, device=dev), widx] = torch.where(
            any_w, out_liq, torch.zeros(n, device=dev))
        self._doses = self._doses + add
        self._spilled = self._spilled + torch.where(any_w, torch.zeros(n, device=dev), out_liq)
        # beaker water ledger: draws take, pushes while seated on the reservoir return
        self._res_taken = (self._res_taken + gain - torch.where(
            seated_res, out_liq, torch.zeros(n, device=dev))).clamp(min=0.0)

        # Optional COMPLIANT FIXTURE on the barrel link (a PD "hand" holding it
        # where fixture() captured it) — kept for drivers without a gripper.
        f_bar = torch.zeros(n, 3, device=dev)
        tq_bar = torch.zeros(n, 3, device=dev)
        if bool(self._fix_on.any()):
            f_fix = (800.0 * (self._fix_pos - bp_f) - 12.0 * bv_f).clamp(-30.0, 30.0)
            # keep upright: torque steers the barrel axis back to the captured axis
            tq_fix = (3.0 * torch.cross(ax, self._fix_ax, dim=-1)
                      - 0.15 * bw_f).clamp(-2.0, 2.0)
            on = self._fix_on.view(n, 1).float()
            f_bar = f_bar + on * f_fix
            tq_bar = on * tq_fix
        self.syringe.set_external_force_and_torque(
            f_bar.view(n, 1, 3), tq_bar.view(n, 1, 3), body_ids=[0], is_global=True)
        self._update_liquid_viz()

    def fixture(self, on: bool = True, env_ids: torch.Tensor | None = None) -> None:
        """Engage/release the compliant barrel fixture at the CURRENT pose (the
        stand-in for a steady hand: an 800 N/m / 12 N*s/m PD plus an uprighting
        torque)."""
        ids = slice(None) if env_ids is None else env_ids
        self._fix_on[ids] = bool(on)
        if on:
            self._fix_pos[ids] = self.syringe.data.root_pos_w[ids]
            self._fix_ax[ids] = self.barrel_axis()[ids]

    # ----- state ---------------------------------------------------------------------------------
    # Every episode-relevant tensor, so a restored snapshot behaves exactly like the
    # moment it was taken (an earlier version missed the travel/hold/fixture state and
    # the ledger cap silently emptied a restored full draw on the next substep).
    _STATE_KEYS = ("_travel_prev", "_liquid", "_doses", "_spilled",
                   "_res_taken", "_drawn_ok", "plunger_drive", "_acc", "_x_led",
                   "_res_recent", "_well_recent", "_fix_on", "_fix_pos", "_fix_ax")

    def _bodies(self) -> dict[str, RigidObject]:
        return {"rack": self.rack, "reservoir": self.reservoir}

    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {n: b.data.root_state_w[env_ids].clone()
                       for n, b in self._bodies().items()},
            "syringe": {
                "root": self.syringe.data.root_state_w[env_ids].clone(),
                "joint_pos": self.syringe.data.joint_pos[env_ids].clone(),
                "joint_vel": self.syringe.data.joint_vel[env_ids].clone(),
            },
            "dose": {k: getattr(self, k)[env_ids].clone() for k in self._STATE_KEYS},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for n, b in self._bodies().items():
            b.write_root_state_to_sim(state["bodies"][n], env_ids)
        s = state["syringe"]
        self.syringe.write_root_state_to_sim(s["root"], env_ids)
        self.syringe.write_joint_state_to_sim(
            s["joint_pos"], s["joint_vel"], env_ids=env_ids)
        for k, v in state["dose"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        lo, hi = c.dose_band
        return (
            f"On the top shelf of a medical cart: a large syringe LYING FLAT (a scanned "
            f"model — translucent barrel with graduation marks and finger flanges, a "
            f"plunger with a thumb plate, travel {c.stroke:.2f} m = its full capacity, and "
            f"a luer nozzle at the tip), a rack of three GLASS TEST TUBES, and a wide "
            f"RESERVOIR beaker of water (its blue level visibly drops as liquid is "
            f"drawn). The cart's lower shelf holds packaged syringes and a basin — "
            f"scenery only.\n"
            f"Liquid moves with the plunger, but ONLY while the syringe is roughly "
            f"upright with the nozzle tip seated on a mouth: hovering within "
            f"{c.seat_tol * 1000:.0f} mm laterally and {c.seat_band * 1000:.0f} mm above the rim. "
            f"Pulling the plunger while seated on the reservoir draws liquid; pushing it "
            f"dispenses into whichever tube the tip is seated on — anywhere else the "
            f"liquid is WASTED. Tubes never un-fill: a tube pushed past its band is "
            f"ruined for the episode (the reservoir can be re-drawn from, but "
            f"over-dispensing cannot be undone). The water is VISIBLE: a blue column "
            f"inside the translucent barrel shows the drawn liquid, and each tube fills "
            f"with a blue column as it receives its dose. The plunger has deliberate "
            f"friction: push/pull it slowly and steadily.\n"
            f"Goal: draw a full load (>= {c.draw_min:.0%} of capacity, tip seated in the "
            f"reservoir), dispense between {lo:.0%} and {hi:.0%} of capacity into EACH of "
            f"the three tubes, then lay the syringe back down at its spot on the shelf."
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
        """Syringe laid back down at its home spot: near home_pos, LYING (no stand),
        on the shelf, settled."""
        c = self.cfg
        p = self.syringe.data.root_pos_w - self.env_origins
        hx, hy = c.home_pos
        near = (p[:, :2] - torch.tensor([hx, hy], device=self.env.device)).norm(dim=-1) < 0.06
        lying = self.barrel_axis()[:, 2].abs() < 0.35
        on_shelf = (p[:, 2] - c.surface_z).abs() < 0.08
        settled = self.syringe.data.root_lin_vel_w.norm(dim=-1) < 0.05
        return near & lying & on_shelf & settled

    def success(self) -> torch.Tensor:
        return self.drawn() & self.doses_ok() & self.parked()


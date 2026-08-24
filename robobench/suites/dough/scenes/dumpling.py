"""DumplingScene — roll the dough out (registered `dumpling`): flatten a ball into a wrapper.

ONE scene class, one cfg, one registration — every run builds this same scene. It composes,
in one place:

  - Newton implicit **MPM elastoplastic dough**: the dough ball is an `MPMObject` (jittered
    particle lattice) with finite Young's modulus + von-Mises yield stress + full cohesion
    (`tensile_yield_ratio=1`) — permanent deformation under the pin. Derived from Newton's
    "mud" recipe (`examples/mpm/example_mpm_multi_material.py`), stiffened so a rolled wrapper
    HOLDS shape instead of slumping.
  - the COUPLED MJWarp+MPM substrate (suite-local `coupled_manager`, a verbatim twin of the
    pouring suite's — suites must not import each other): an arm
    gets real dynamics (gravity, actuator PD, MuJoCo rigid contacts) while the dough follows the
    post-rigid body poses (one-way rigid -> dough; feedback OFF — a pin needn't feel the dough).
    The robot-less binding runs the pure-MPM manager with a KINEMATIC pin instead.
  - a DYNAMIC rolling pin: one free rigid body (capsule barrel along +x, square grip stub on
    top, square END-CAPS over the barrel domes) nested in a static two-block cradle; barrel MPM
    friction is LOW (a sliding pass must squeeze the dough, not drag it) while the table is
    grippy (floured board). The end-caps rest flat across the cradle slot edges — a geometric
    roll-lock (a bare barrel is an inverted pendulum under the top-heavy stub, and regularized
    contact friction lets it creep over until the stub lies sideways).
  - the AUTO-WELD grasp contract (the pouring suite's mechanic): close the gripper on the grip
    stub within `auto_weld_dist` and the pin welds on at the measured pose; open past
    `auto_weld_release` to let go. No scripted attach calls anywhere.

Task story (env-local meters, table top at `surface_z`): a dough ball rests at (0, 0);
ROLL IT OUT — low sliding passes flatten it into a wrapper (thin + wide:
`flatten_h95_max` / `spread_r90_min`), with nothing smeared off the board or punched through
the table. `success()` gates the rolled sheet plus those guards.

Requires the Newton venv (`env_newton`, see the README) to build; single-env only (the MPM
fixed grid spans the whole scene). Heavy imports are deferred so importing stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from robobench.core import SCENES, BaseCfg, BaseScene, info, tunable
from robobench.suites.dough.newton_sim import DoughSimCfg

if TYPE_CHECKING:
    import torch

    from robobench.core import BaseEnv


# ----- procedural particle seeding (numpy only; app-free) ------------------------------------------
def sphere_lattice(
    radius: float,
    z_lo: float,
    voxel: float,
    particles_per_cell: float,
    density: float,
    seed: int,
) -> tuple[np.ndarray, float, float]:
    """Jittered particle lattice filling a local-space ball whose BOTTOM sits at `z_lo`
    (center at (0, 0, z_lo + radius)). Returns (points, particle_radius, particle_mass)."""
    lo = np.array([-radius, -radius, z_lo], dtype=np.float32)
    hi = np.array([radius, radius, z_lo + 2.0 * radius], dtype=np.float32)
    resolution = np.maximum(np.ceil(particles_per_cell * (hi - lo) / voxel), 1).astype(int)
    cell = (hi - lo) / resolution
    cell_volume = float(np.prod(cell))
    p_radius = float(cell.max() * 0.45)
    p_mass = float(cell_volume * density)
    axes = [np.arange(int(n) + 1) * c for n, c in zip(resolution, cell)]
    points = np.stack(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    rng = np.random.default_rng(seed)
    points += (rng.random(points.shape) - 0.5) * (0.10 * float(cell.max()))
    points += lo
    center = np.array([0.0, 0.0, z_lo + radius], dtype=np.float32)
    keep = ((points - center) ** 2).sum(axis=1) < radius**2
    points = points[keep]
    if points.shape[0] == 0:
        raise RuntimeError("sphere_lattice produced no particles; shrink voxel_size or grow the ball.")
    return points.astype(np.float32, copy=False), p_radius, p_mass


@dataclass
class DumplingSceneCfg(BaseCfg):
    """Dough, pin, and layout dials. The dough recipe starts from Newton's cohesive "mud"
    material (`example_mpm_multi_material.py`: yield_pressure 1e10, yield_stress 300,
    tensile_yield_ratio 1, friction 0, viscosity 100, rho 1000), stiffened so the rolled
    wrapper holds shape instead of slumping."""

    # --- MPM solver / seeding ---
    voxel_size: float = tunable(0.0025)  # [TUNE] MPM grid voxel [m]; 2 voxels through the 5 mm target
    # sheet (pouring runs 0.003); finer resolves the wrapper cross-section, slower
    particles_per_cell: float = tunable(2.0)  # pouring's floor — below 2.0 under-resolves the
    # constitutive model (its 1.6 note: a deep narrow fill collapsed into a sticky blob)
    max_iterations: int = tunable(100)  # rheology iterations (pouring's proven value)
    # --- dough material (wheat dough: cohesive elasto-viscoplastic) ---
    dough_density: float = tunable(1100.0)  # wheat dough ~1.05-1.2 g/cm^3
    dough_young_modulus: float = tunable(2.0e5)  # [TUNE] finite E so passes MATE the sheet smoothly; the
    # mud recipe keeps the solver's rigid-plastic 1e15 default — too crisp
    dough_poisson: float = tunable(0.45)  # near-incompressible
    dough_yield_stress: float = tunable(2.0e3)  # [TUNE] von-Mises tau_y [Pa]; mud's 300 self-slumps in
    # seconds; self-weight stress ~rho*g*t ~= 54 Pa << 2 kPa << pin-contact kPa..10s kPa, so the
    # wrapper holds shape at rest yet yields under the pass
    dough_yield_pressure: float = tunable(1.0e10)  # mud value: never yields as a granular (no crumble)
    dough_tensile_ratio: float = tunable(1.0)  # mud value: FULL cohesion (tension cutoff never truncates)
    dough_viscosity: float = tunable(20.0)  # [TUNE] rate resistance; mud's 100 is honey-drag
    dough_friction: float = tunable(0.0)  # mud value: dough strength is cohesive, not frictional
    dough_damping: float = tunable(0.05)  # kills post-pass ringing (pouring liquids run 0.02)
    # hardening/dilatancy stay 0: snow's hardening=10 packs stiff — dough must stay re-workable
    # --- geometry [m] ---
    dough_ball_radius: float = tunable(0.022)  # d=4.4 cm ball; its volume rolls into a ~5 mm sheet of
    # radius ~5.3 cm (a 10-11 cm wrapper) — ~23k particles at voxel 0.0025 / ppc 2
    # --- layout (the folding suite's ambient frame: table top at 0.2, visual ground sunk) ---
    surface_z: float = info(0.2, doc="table-top height [m]; record_video.py anchors the camera on this")
    table_size: tuple[float, float, float] = info((0.8, 0.8, 0.2), doc="table box extents [m]")
    table_center_x: float = info(0.05, doc="table center x [m]; top spans x in [-0.35, 0.45]")
    table_friction: float = tunable(0.8)  # [TUNE] floured-board grip: the dough must NOT ride the
    # sliding pin — pass flattening lives on table mu >> barrel mu
    light_intensity: float = tunable(3000.0)
    # --- rolling pin: ONE free rigid body (capsule barrel along +x + grip stub + end-caps) ---
    pin_barrel_radius: float = tunable(0.020)
    pin_barrel_half_len: float = tunable(0.070)  # 14 cm cylindrical section spans the ~11 cm wrapper
    pin_grip_width: float = info(0.022)  # square stub cross-section = the pouring MUG bar's grip_w
    # (fingers engage with PD headroom at aperture ~11 mm)
    pin_grip_height: float = tunable(0.05)  # stub top clears the hand body during the top-down pinch
    pin_mass: float = tunable(0.35)  # wooden pin [kg]; carried by the weld, so mostly cosmetic
    pin_cap_cross: float = info(0.032)  # square END-CAP cross-section [m]. The caps rest FLAT
    # across the cradle slot edges, locking the parked pin against axis-spin GEOMETRICALLY: a
    # bare barrel in the slot is an inverted pendulum (top-heavy stub), and MuJoCo's regularized
    # contact friction only creeps, never sticks — the stub would end up sideways, unpinchable
    # for a top-down grasp. Sized so the resting barrel floats ~2 mm clear of the slot edges
    # while the cap corners stay above the lowest commanded pass floor.
    pin_cap_len: float = info(0.020)  # end-cap axial length [m]; caps cover the capsule domes
    pin_friction: float = tunable(0.05)  # [TUNE] barrel MPM/table friction — LOW: a welded pin SLIDES
    # over the dough (one-way coupling can't spin a passive roller), and a sliding pass must
    # squeeze the sheet flat, not plow it sideways
    pin_grip_friction: float = tunable(0.8)  # stub faces the fingers (rigid grip)
    pin_stand: tuple[float, float] = info((-0.14, -0.20), doc="cradle center xy [m]; also the park spot,"
                                          " offset clear of the rolling strokes over the dough at x~0")
    cradle_gap: float = info(0.015, doc="cradle inner-face half-gap [m]; the end-caps bridge it")
    cradle_height: float = tunable(0.025)  # cradle block height [m]
    pin_dynamic: bool = info(True, doc="False in the robot-less binding: kinematic pin, scripted "
                             "pose writes (the pure-MPM manager ghosts kinematic bodies into colliders)")
    # --- auto-weld grasp contract (the pouring suite's proven values) ---
    auto_weld_dist: float = tunable(0.03)  # pinch-point-to-stub-center engage radius [m]
    auto_weld_close_margin: float = tunable(0.003)  # engage when aperture < stub half-width + this [m]
    auto_weld_release: float = tunable(0.02)  # release when aperture opens past this [m] (hysteresis)
    # --- success gates (sample over a settle window and gate medians; instantaneous in success()) ---
    flatten_h95_max: float = tunable(0.009)  # wrapper q95 height above the table [m]; ball starts ~0.044
    spread_r90_min: float = tunable(0.038)  # q90 radius about the dough centroid [m]. An ideal-volume
    # 5.3 cm disc would give sqrt(0.9)*R ~= 0.050, but a really rolled sheet rounds its rim:
    # kinematic-pin runs land ~0.042, franka runs 0.040-0.047; the settled BALL reads ~0.020 —
    # 0.038 keeps wide separation from the failure mode on both sides
    extent_max: float = tunable(0.14)  # final max xy-extent of the dough [m] (a rolled wrapper spans
    # ~0.11; smears and squirts blow past this)
    conserve_min: float = tunable(0.995)  # fraction of the dough still in the work zone
    floor_z_slack: float = tunable(0.008)  # min particle z >= surface_z - this (nothing punched through)
    # --- rendering (the pouring suite's Fabric workaround values) ---
    dough_color: tuple[float, float, float] = info((0.93, 0.87, 0.70), doc="dough particle display color")
    table_color: tuple[float, float, float] = info((0.04, 0.04, 0.045), doc="table-top display color; "
                                                   "metallic black (the allen_bolt lab-table look) so the "
                                                   "pale dough reads against it")
    visual_update_frequency: int = info(4, doc="Kit particle visual update period [render frames]")
    visual_width_scale: float = tunable(2.2)  # display width vs physical particle diameter

    @property
    def pin_home_z(self) -> float:
        """Barrel-center rest height above the table: the square end-caps sit flat across the
        cradle slot edges (the barrel itself floats ~2 mm clear — the caps' geometric roll-lock
        is what holds the top-heavy pin upright between grasps)."""
        return self.cradle_height + self.pin_cap_cross / 2.0


@SCENES.register("dumpling")
class DumplingScene(BaseScene):
    """THE dough roll-out benchmark scene — see the module docstring for the mechanic stack
    (MPM dough, coupled substrate, dynamic pin, auto-weld grasping). Handles after bind:
    `self.dough` (MPMObject) and `self.pin` (RigidObject); `success()` gates the rolled
    wrapper plus the conservation guards."""

    cfg: DumplingSceneCfg

    def __init__(self, cfg: DumplingSceneCfg | None = None) -> None:
        super().__init__(cfg or DumplingSceneCfg())

    # ----- assets ---------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        from collections.abc import Callable
        from dataclasses import MISSING

        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
        from isaaclab.sim.utils import clone
        from isaaclab.utils.configclass import configclass
        from isaaclab_newton.assets.mpm_object import MPMObjectCfg
        from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg, MPMPointsCfg

        c = self.cfg
        s = c.surface_z

        @configclass
        class PinCfg(sim_utils.MeshCfg):
            """Suite-local rolling-pin spawner: capsule barrel (axis +x) + square grip stub +
            square end-caps on one Xform rigid body, separate physics materials per part (low-mu
            barrel faces the dough, grippy stub faces the fingers)."""

            func: Callable | str = clone(_spawn_pin)
            barrel_radius: float = MISSING
            barrel_half_len: float = MISSING
            grip_width: float = MISSING
            grip_height: float = MISSING
            cap_cross: float = MISSING
            cap_len: float = MISSING
            mass: float = MISSING
            grip_physics_material: sim_utils.NewtonMaterialPropertiesCfg | None = None

        # Dough: a ball resting just above the table (settles at t=0).
        dough_pts, dough_pr, dough_pm = sphere_lattice(
            c.dough_ball_radius, 0.002, c.voxel_size, c.particles_per_cell, c.dough_density, seed=0
        )

        static_collision = sim_utils.NewtonCollisionPropertiesCfg(collision_enabled=True, contact_margin=0.0003)
        table_mat = sim_utils.NewtonMaterialPropertiesCfg(
            static_friction=c.table_friction, dynamic_friction=c.table_friction
        )

        def block(pos: tuple[float, float, float], size: tuple[float, float, float]) -> AssetBaseCfg:
            return AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Cradle" + f"{'A' if pos[1] < c.pin_stand[1] else 'B'}",
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    collision_props=static_collision,
                    physics_material=table_mat,
                    physics_material_path="physicsMaterial",
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.35, 0.25)),
                    visual_material_path="visualMaterial",
                ),
            )

        sx, sy = c.pin_stand
        # long enough in x that the barrel END-CAPS (at +-[70, 90] mm) rest on the slot edges
        cradle_size = (0.20, 0.02, c.cradle_height)
        cradle_dy = c.cradle_gap + cradle_size[1] / 2.0  # inner faces at +-cradle_gap from the stand
        return {
            # Visual ground SUNK to z=-1.05 (folding-suite landmine: a ground-plane collider at
            # exactly z=0 goes haywire in the Newton->MuJoCo conversion under the coupled
            # substrate — phantom kN*m contact forces on the arm joints). The invisible Floor box
            # below carries the actual z=0 surface for anything that falls off the table.
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
                spawn=sim_utils.GroundPlaneCfg(),
            ),
            "floor": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Floor",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
                spawn=sim_utils.CuboidCfg(
                    size=(2.0, 2.0, 0.02),
                    collision_props=static_collision,
                    physics_material=table_mat,
                    physics_material_path="physicsMaterial",
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35)),
                    visual_material_path="visualMaterial",
                ),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=c.light_intensity, color=(0.85, 0.85, 0.85)),
            ),
            "table": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(c.table_center_x, 0.0, s - c.table_size[2] / 2)),
                spawn=sim_utils.CuboidCfg(
                    size=c.table_size,
                    collision_props=static_collision,
                    physics_material=table_mat,
                    physics_material_path="physicsMaterial",
                    # Metallic black bench (the allen_bolt lab-table look): the pale dough reads
                    # against it. Display-only — friction is the physics material.
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.table_color, metallic=0.9, roughness=0.35
                    ),
                    visual_material_path="visualMaterial",
                ),
            ),
            # Static cradle: two blocks along x, spaced in y — the pin's square end-caps rest
            # FLAT across the slot edges (geometric roll-lock; the bare barrel would creep-spin
            # under the top-heavy stub) with the grip stub standing proud for the pinch.
            "cradle_a": block((sx, sy - cradle_dy, s + c.cradle_height / 2.0), cradle_size),
            "cradle_b": block((sx, sy + cradle_dy, s + c.cradle_height / 2.0), cradle_size),
            "pin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pin",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(sx, sy, s + c.pin_home_z + 0.001)),
                spawn=PinCfg(
                    barrel_radius=c.pin_barrel_radius,
                    barrel_half_len=c.pin_barrel_half_len,
                    grip_width=c.pin_grip_width,
                    grip_height=c.pin_grip_height,
                    cap_cross=c.pin_cap_cross,
                    cap_len=c.pin_cap_len,
                    mass=c.pin_mass,
                    rigid_props=sim_utils.NewtonRigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        kinematic_enabled=not c.pin_dynamic,
                        disable_gravity=False,
                    ),
                    collision_props=sim_utils.NewtonCollisionPropertiesCfg(
                        collision_enabled=True, contact_margin=0.001
                    ),
                    physics_material=sim_utils.NewtonMaterialPropertiesCfg(
                        static_friction=c.pin_friction, dynamic_friction=c.pin_friction
                    ),
                    physics_material_path="physicsMaterial",
                    grip_physics_material=sim_utils.NewtonMaterialPropertiesCfg(
                        static_friction=c.pin_grip_friction, dynamic_friction=c.pin_grip_friction
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.65, 0.45, 0.25)),
                    visual_material_path="visualMaterial",
                ),
            ),
            "dough": MPMObjectCfg(
                prim_path="{ENV_REGEX_NS}/Dough",
                spawn=MPMPointsCfg(
                    positions=dough_pts.tolist(),
                    mass=dough_pm,
                    radius=dough_pr,
                    material=MPMParticleMaterialCfg(
                        density=c.dough_density,
                        young_modulus=c.dough_young_modulus,
                        poisson_ratio=c.dough_poisson,
                        yield_stress=c.dough_yield_stress,
                        yield_pressure=c.dough_yield_pressure,
                        tensile_yield_ratio=c.dough_tensile_ratio,
                        viscosity=c.dough_viscosity,
                        friction=c.dough_friction,
                        damping=c.dough_damping,
                    ),
                    visual_color=c.dough_color,
                    visual_update_frequency=c.visual_update_frequency,
                ),
                init_state=MPMObjectCfg.InitialStateCfg(pos=(0.0, 0.0, s)),
            ),
        }

    def sim_cfg(self) -> DoughSimCfg:
        return DoughSimCfg(
            voxel_size=self.cfg.voxel_size,
            max_iterations=self.cfg.max_iterations,
            coupled=True,
            # The pin's grasp-contract weld row (disabled until grasp; ignored by the uncoupled
            # substrate). The contract currently binds panda-handed arms — the suffix names the
            # hand body the weld attaches to.
            welds=[("weld_pin", "Robot/panda_hand", "Pin")],
        )

    # ----- lifecycle ------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        import torch

        super().bind(env)
        self.dough = env.iscene["dough"]
        self.pin = env.iscene["pin"]
        # Snapshot the spawn state as the reset target (the settled-ball start).
        self._default_state = (
            self.dough.data.nodal_pos_w.torch.clone(),
            self.dough.data.nodal_vel_w.torch.clone().zero_(),
        )
        origins = env.iscene.env_origins
        c = self.cfg
        quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=origins.device).expand(origins.shape[0], 4)
        pin_local = torch.tensor(
            [c.pin_stand[0], c.pin_stand[1], c.surface_z + c.pin_home_z + 0.001], device=origins.device
        )
        self._default_pin_pose = torch.cat([origins + pin_local, quat], dim=-1)
        # Latest actual pin pose (world) — pass scheduling reads it (honest under a dropped pin).
        self.pin_pose_w = self._default_pin_pose.clone()
        self._fabric_particle_attrs: list[tuple[Any, Any]] = []
        self._auto_state: dict[str, bool] = {}
        self._auto_ready = False
        # Whether a grasp contract exists is decided LAZILY in _auto_setup: the robot binds
        # AFTER the scene (BaseEnv.__init__ order), so robot-side probes here see a stub.
        self._auto_dead = False

    def resync_colliders(self) -> None:
        """Re-seed the MPM collider pose history after any TELEPORTING write (an episodic
        `env.reset()`): under `collider_velocity_mode="backward"` an unsynced teleport reads as
        a (jump/dt) collider velocity that violently kicks the dough for one tick. Handles both
        substrates; never call per scripted per-tick pose write."""
        from robobench.suites.dough.coupled_manager import NewtonCoupledMJWarpMPMManager as Mgr

        if Mgr._mpm_solver is not None:
            Mgr.resync_collider_history()
            return
        # Pure-MPM substrate: the implicit-MPM solver IS the canonical solver.
        from newton import eval_fk
        from isaaclab_newton.physics.newton_manager import NewtonManager

        solver, state = NewtonManager._solver, NewtonManager._state_0
        if solver is not None and state is not None and hasattr(solver, "_last_step_data"):
            eval_fk(NewtonManager._model, state.joint_q, state.joint_qd, state, None)
            solver._last_step_data.save_collider_current_position(state.body_q)

    # ----- auto-weld grasp contract + actual-pose tracking ----------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Track the pin's actual pose (pass scheduling + metrics are honest under a dropped
        pin), then run the auto-weld grasp state machine."""
        import torch

        self.pin_pose_w = torch.cat(
            [self.pin.data.root_link_pos_w.torch, self.pin.data.root_link_quat_w.torch], dim=-1
        )
        if self._auto_dead:
            return
        try:
            if not self._auto_ready:
                self._auto_setup()
            if not self._auto_dead:
                self._auto_tick()
        except Exception as e:  # noqa: BLE001 — a broken grasp mechanic must be loud, not fatal
            print(f"[auto-weld] DISABLED after error: {e!r}", flush=True)
            self._auto_dead = True

    def _auto_setup(self) -> None:
        """Resolve body indices, the grip-stub local center, and the finger joints. Runs once,
        lazily (the Newton model exists post-build, and the robot is bound by the first
        post_step). Robot-less binding / uncoupled substrate: no grasp contract — go dead."""
        import torch

        from robobench.suites.dough.coupled_manager import NewtonCoupledMJWarpMPMManager as Mgr

        if Mgr._mpm_solver is None or getattr(self.env.robot, "articulation", None) is None:
            self._auto_dead = True
            self._auto_ready = True
            return
        body_labels = [str(b or "") for b in Mgr._model.body_label]

        def body_idx(suffix: str) -> int:
            matches = [i for i, b in enumerate(body_labels) if b.endswith(suffix)]
            assert len(matches) == 1, (suffix, matches)
            return matches[0]

        c = self.cfg
        self._pin_body = body_idx("Pin")
        self._hand_body = body_idx("Robot/panda_hand")
        art = self.env.robot.articulation
        self._finger_ids = art.find_joints(["panda_finger.*"])[0]
        # Grip-stub center in the pin body frame (body origin = barrel center).
        self._stub_local = torch.tensor(
            [[0.0, 0.0, c.pin_barrel_radius + c.pin_grip_height / 2.0]], device=self.env.device
        )
        # Grasp-target table: (weld label, body index, grip center in body frame, pinch
        # half-width) — one contract, N tools; the machine engages the NEAREST inside the window.
        self._tools = [("weld_pin", self._pin_body, self._stub_local, c.pin_grip_width / 2.0)]
        self._auto_ready = True

    def _auto_tick(self) -> None:
        import torch
        import warp as wp

        import isaaclab.utils.math as math_utils

        from robobench.suites.dough.coupled_manager import NewtonCoupledMJWarpMPMManager as Mgr

        c = self.cfg
        art = self.env.robot.articulation
        aperture = float(art.data.joint_pos.torch[0, self._finger_ids].mean())
        held = [lb for lb, _, _, _ in self._tools if self._auto_state.get(lb, False)]
        if held:
            if aperture > c.auto_weld_release:
                for lb in held:
                    Mgr.set_weld(lb, False)
                    self._auto_state[lb] = False
                    print(f"  [auto-weld] RELEASED {lb[5:]} (aperture {aperture * 1000:.1f} mm)", flush=True)
            return
        if aperture >= max(hw for _, _, _, hw in self._tools) + c.auto_weld_close_margin:
            return  # fingers not squeezing — cheap early-out before any pose math
        # fresh view every tick — never cache it: the solver ping-pongs state_0/state_1, so a
        # cached view pins one buffer and the machine reads frozen poses
        body_q = wp.to_torch(Mgr._state_0.body_q)
        tip_local = torch.tensor([[0.0, 0.0, 0.113]], device=body_q.device)
        pinch = body_q[self._hand_body, :3] + math_utils.quat_apply(body_q[self._hand_body, 3:][None], tip_local)[0]
        best = None
        for lb, body, grip_local, half_w in self._tools:
            if aperture >= half_w + c.auto_weld_close_margin:
                continue
            grip = body_q[body, :3] + math_utils.quat_apply(body_q[body, 3:][None], grip_local.to(body_q.device))[0]
            dist = float((pinch - grip).norm())
            if dist < c.auto_weld_dist and (best is None or dist < best[1]):
                best = (lb, dist)
        if best is not None:
            lb, dist = best
            Mgr.set_weld(lb, True)
            self._auto_state[lb] = True
            print(
                f"  [auto-weld] GRIPPED {lb[5:]} (dist {dist * 100:.1f} cm, aperture {aperture * 1000:.1f} mm)",
                flush=True,
            )

    # ----- Kit particle visuals ---------------------------------------------------------------
    # Same Fabric workaround as the pouring suite: the backend's plain-USD particle sync is
    # ignored by the Fabric scene delegate, so positions are pushed through usdrt. Display-only.
    def setup_particle_visuals(self) -> None:
        """Prepare Kit particle rendering: scale display widths by `cfg.visual_width_scale` and
        attach the Fabric stage for `push_particle_visuals`. Idempotent; silent no-op without
        the kit visualizer."""
        if self._fabric_particle_attrs:
            return
        try:
            import usdrt
            from pxr import UsdGeom, Vt

            import isaaclab.sim as sim_utils
            from isaaclab.sim.utils.stage import get_current_stage

            stage = sim_utils.get_current_stage()
            if not stage.GetPrimAtPath("/World/Visuals/MPMParticles").IsValid():
                return
            vis_prims = [
                p
                for p in stage.Traverse()
                if p.GetTypeName() == "Points" and str(p.GetPath()).startswith("/World/Visuals/MPMParticles")
            ]
            scale = float(self.cfg.visual_width_scale)
            rt_stage = get_current_stage(fabric=True)
            for prim in vis_prims:
                scaled = np.array(UsdGeom.Points(prim).GetWidthsAttr().Get(), dtype=np.float32) * scale
                UsdGeom.Points(prim).GetWidthsAttr().Set(Vt.FloatArray.FromNumpy(scaled))
                path = str(prim.GetPath())
                obj = self.dough if "Dough" in path else None
                rt_prim = rt_stage.GetPrimAtPath(path)
                if obj is not None and rt_prim:
                    try:
                        w_attr = rt_prim.GetAttribute("widths") or rt_prim.CreateAttribute(
                            "widths", usdrt.Sdf.ValueTypeNames.FloatArray, False
                        )
                        w_attr.Set(usdrt.Vt.FloatArray(scaled.reshape(-1, 1)))  # usdrt arrays are 2-D
                    except Exception as e:  # noqa: BLE001
                        print(f"[dumpling] fabric widths write skipped ({e}); USD widths still set", flush=True)
                    self._fabric_particle_attrs.append((rt_prim.GetAttribute("points"), obj))
            self._usdrt_vt = usdrt.Vt
        except Exception as e:  # noqa: BLE001 — display sugar must never kill a run
            print(f"[dumpling] Kit particle visual setup skipped: {e}", flush=True)

    def push_particle_visuals(self) -> None:
        """Write current particle positions into Fabric so the Kit render shows live dough.
        Call at render cadence; no-op if setup found no prims."""
        for attr, obj in self._fabric_particle_attrs:
            pts = obj.data.nodal_pos_w.torch[0].cpu().numpy().astype(np.float32)
            attr.Set(self._usdrt_vt.Vec3fArray(pts))

    def reset(self, env_ids: torch.Tensor) -> None:
        import torch

        # Weld off FIRST: the pin teleports home while the hand still holds its last pose — an
        # active weld would read that as a violent constraint violation.
        if not self._auto_dead and self._auto_state.get("weld_pin"):
            from robobench.suites.dough.coupled_manager import NewtonCoupledMJWarpMPMManager as Mgr

            Mgr.set_weld("weld_pin", False)
        self._auto_state = {}
        pos, vel = self._default_state
        self.dough.write_nodal_pos_to_sim_index(pos[env_ids].contiguous(), env_ids=env_ids)
        self.dough.write_nodal_velocity_to_sim_index(vel[env_ids].contiguous(), env_ids=env_ids)
        self.pin.write_root_link_pose_to_sim_index(
            root_pose=self._default_pin_pose[env_ids].contiguous(), env_ids=env_ids
        )
        zero_twist = torch.zeros((len(env_ids), 6), device=self._default_pin_pose.device)
        self.pin.write_root_link_velocity_to_sim_index(root_velocity=zero_twist, env_ids=env_ids)
        self.pin_pose_w = self._default_pin_pose.clone()

    # ----- state ----------------------------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "dough_pos": self.dough.data.nodal_pos_w.torch[env_ids].clone(),
            "dough_vel": self.dough.data.nodal_vel_w.torch[env_ids].clone(),
            "pin_pose": self.pin_pose_w[env_ids].clone(),
            "pin_vel": self.pin.data.root_link_vel_w.torch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.dough.write_nodal_pos_to_sim_index(state["dough_pos"].contiguous(), env_ids=env_ids)
        self.dough.write_nodal_velocity_to_sim_index(state["dough_vel"].contiguous(), env_ids=env_ids)
        self.pin.write_root_link_pose_to_sim_index(root_pose=state["pin_pose"].contiguous(), env_ids=env_ids)
        self.pin.write_root_link_velocity_to_sim_index(
            root_velocity=state["pin_vel"].contiguous(), env_ids=env_ids
        )

    # ----- metrics --------------------------------------------------------------------------------
    # Torch reductions over the dough cloud in the env-local frame, heights relative to the
    # table top. Sample them over a settle window and gate medians; success() gates the
    # instantaneous values.
    def _local(self, obj) -> torch.Tensor:
        """Particle positions with the env origin removed (env-local frame): (num_envs, P, 3)."""
        return obj.data.nodal_pos_w.torch - self.env.iscene.env_origins[:, None, :]

    def flatten_height(self) -> torch.Tensor:
        """(N,) wrapper thickness proxy: q95 dough height above the table [m]."""
        import torch

        return torch.nanquantile(self._local(self.dough)[..., 2], 0.95, dim=1) - self.cfg.surface_z

    def spread_radius(self) -> torch.Tensor:
        """(N,) wrapper size proxy: q90 radial distance from the dough xy centroid [m]."""
        import torch

        p = self._local(self.dough)[..., :2]
        return torch.nanquantile((p - p.nanmean(dim=1, keepdim=True)).norm(dim=-1), 0.90, dim=1)

    def final_extent(self) -> torch.Tensor:
        """(N,) xy-extent of the dough [m], per-axis q99.5-q0.5 span — compactness of the
        RESULT. Quantile span, not max-min: a pass can squirt a few crumbs sideways and a
        handful of outliers must not flip a compactness verdict (strays are policed by
        `conservation()` instead)."""
        import torch

        p = self._local(self.dough)[..., :2]
        ext = torch.nanquantile(p, 0.995, dim=1) - torch.nanquantile(p, 0.005, dim=1)
        return ext.max(dim=-1).values

    def conservation(self) -> torch.Tensor:
        """(N,) fraction of dough particles still in the work zone: inside a 0.30 m cylinder
        about the board origin, z in [table - 1 cm, table + 30 cm]."""
        c = self.cfg
        p = self._local(self.dough)
        inside = (
            (p[..., :2].norm(dim=-1) < 0.30)
            & (p[..., 2] > c.surface_z - 0.01)
            & (p[..., 2] < c.surface_z + 0.30)
        )
        return inside.float().mean(dim=1)

    def min_particle_z(self) -> torch.Tensor:
        """(N,) lowest dough particle height [m] (q0.2%) — table punch-through guard."""
        import torch

        return torch.nanquantile(self._local(self.dough)[..., 2], 0.002, dim=1)

    def success(self) -> torch.Tensor:
        """(N,) bool: the dough is ROLLED OUT — thin and wide — and nothing was smeared off the
        board or punched through the table. Sample over a settle window and gate medians for a
        robust verdict; this is the instantaneous alias matching the other suites' surface."""
        c = self.cfg
        return (
            (self.flatten_height() <= c.flatten_h95_max)
            & (self.spread_radius() >= c.spread_r90_min)
            & (self.final_extent() <= c.extent_max)
            & (self.conservation() >= c.conserve_min)
            & (self.min_particle_z() >= c.surface_z - c.floor_z_slack)
        )

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A ball of dough (radius {c.dough_ball_radius * 100:.1f} cm, elastoplastic — it holds whatever"
            f" shape you press it into) rests at (0, 0) on a dark steel worktop, floured so the dough"
            f" grips without sticking (table top at z={c.surface_z})."
            f" A wooden rolling pin (barrel radius {c.pin_barrel_radius * 100:.0f} cm, axis along +x, flat"
            f" square end-caps) rests in a cradle at ({c.pin_stand[0]}, {c.pin_stand[1]}) with a square grip"
            f" stub on top. GRASPING: move the gripper's pinch point within {c.auto_weld_dist * 100:.0f} cm"
            " of the stub and CLOSE the fingers onto it — the pin attaches rigidly; OPEN the gripper to"
            " release it. Goal: ROLL THE DOUGH OUT into a wrapper with low sliding passes of the barrel"
            f" (dough q95 height <= {c.flatten_h95_max * 1000:.0f} mm, q90 radius >="
            f" {c.spread_r90_min * 1000:.0f} mm), keeping the sheet compact (xy extent <="
            f" {c.extent_max * 1000:.0f} mm) with nothing flung off the board or punched through the"
            " table. NOTE: the barrel is deliberately low-friction — a sliding pass SQUEEZES the sheet"
            " flat; pushing moves this dough reliably, lifting it by a pinch does not."
        )


# ----- suite-local pin spawner (module level so configclass `func` can reference it) ---------------
def _spawn_pin(
    prim_path: str,
    cfg: Any,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: Any,
):
    """Spawn the rolling pin as a standard Isaac Lab asset: Xform root + Capsule barrel (rotated
    onto local +x) + Cube grip stub + square end-caps (the cradle roll-lock), rigid-body + mass
    + collision schemas from the cfg, and separate physics materials for barrel (dough-facing)
    and stub (finger-facing); the caps share the barrel's material."""
    from pxr import Gf, Sdf, UsdPhysics

    from isaaclab.sim import schemas
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material, create_prim, get_current_stage

    stage = get_current_stage()
    create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation, stage=stage)
    geom_path = f"{prim_path}/geometry"
    create_prim(geom_path, prim_type="Xform", stage=stage)

    barrel_path = f"{geom_path}/barrel"
    # UsdGeom.Capsule extends along its local Z; rotate 90 deg about Y so the barrel runs along
    # the BODY's +x (the shape transform carries the rotation into both MuJoCo and MPM).
    create_prim(
        barrel_path,
        prim_type="Capsule",
        attributes={"radius": float(cfg.barrel_radius), "height": 2.0 * float(cfg.barrel_half_len)},
        orientation=(0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)),  # xyzw, Ry(90)
        stage=stage,
    )
    grip_path = f"{geom_path}/grip"
    create_prim(
        grip_path,
        prim_type="Cube",
        attributes={"size": 1.0},
        translation=(0.0, 0.0, float(cfg.barrel_radius) + float(cfg.grip_height) / 2.0),
        scale=(float(cfg.grip_width), float(cfg.grip_width), float(cfg.grip_height)),
        stage=stage,
    )
    # Square end-caps over the capsule domes: parked, they rest FLAT across the cradle slot
    # edges — a geometric roll-lock. A bare barrel spins in the slot under the top-heavy stub's
    # gravity torque (regularized contact friction only CREEPS, never sticks) and the stub ends
    # up sideways, unreachable for the top-down re-grasp.
    cap_paths = []
    for side, xs in (("capL", -1.0), ("capR", 1.0)):
        cap_path = f"{geom_path}/{side}"
        create_prim(
            cap_path,
            prim_type="Cube",
            attributes={"size": 1.0},
            translation=(xs * (float(cfg.barrel_half_len) + float(cfg.cap_len) / 2.0), 0.0, 0.0),
            scale=(float(cfg.cap_len), float(cfg.cap_cross), float(cfg.cap_cross)),
            stage=stage,
        )
        cap_paths.append(cap_path)
    # Stiff contact on the grip stub (mjc:solref, read by the newton importer) — same dial as
    # the pouring suite's handle bars, so the pinch doesn't wobble in the weld window.
    stage.GetPrimAtPath(grip_path).CreateAttribute("mjc:solref", Sdf.ValueTypeNames.Float2, custom=True).Set(
        Gf.Vec2f(0.004, 1.0)
    )

    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)
    if getattr(cfg, "mass", None):
        UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(prim_path)).GetMassAttr().Set(float(cfg.mass))
    if cfg.collision_props is not None:
        schemas.define_collision_properties(barrel_path, cfg.collision_props, stage=stage)
        schemas.define_collision_properties(grip_path, cfg.collision_props, stage=stage)
        for cap_path in cap_paths:
            schemas.define_collision_properties(cap_path, cfg.collision_props, stage=stage)
    if cfg.visual_material is not None:
        material_path = cfg.visual_material_path
        if not material_path.startswith("/"):
            material_path = f"{geom_path}/{material_path}"
        cfg.visual_material.func(material_path, cfg.visual_material)
        bind_visual_material(barrel_path, material_path, stage=stage)
        bind_visual_material(grip_path, material_path, stage=stage)
        for cap_path in cap_paths:
            bind_visual_material(cap_path, material_path, stage=stage)
    if cfg.physics_material is not None:  # barrel: dough-facing low mu
        material_path = cfg.physics_material_path
        if not material_path.startswith("/"):
            material_path = f"{geom_path}/{material_path}"
        cfg.physics_material.func(material_path, cfg.physics_material)
        bind_physics_material(barrel_path, material_path, stage=stage)
        for cap_path in cap_paths:
            bind_physics_material(cap_path, material_path, stage=stage)
    if getattr(cfg, "grip_physics_material", None) is not None:  # stub: finger-facing grip
        grip_mat_path = f"{geom_path}/gripMaterial"
        cfg.grip_physics_material.func(grip_mat_path, cfg.grip_physics_material)
        bind_physics_material(grip_path, grip_mat_path, stage=stage)

    return stage.GetPrimAtPath(prim_path)

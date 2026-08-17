"""CaromCourtScene — aim the deflector, then release the ball; it banks into the bay.

Derived from the RLBench `basketball_in_hoop` seed but STRATEGICALLY DIFFERENT (see
TASK.md): the seed's plan is grasp the ball, carry it above the hoop, open the jaw —
gravity delivers, the hand touches the ball the whole way and aiming is just "hold it
over the ring". Here the robot NEVER touches the ball and can never carry it: the ball
is sealed inside a fully ROOFED circular court (the only roof opening is a 26 mm shaft
slot a 60 mm ball cannot pass) and delivery is INDIRECT and ballistic-by-rolling. Two
controls stand outside/above the sealed volume:

  1. an overhead POINTER BAR at the court centre, rigid with a hidden deflector blade
     under the roof (one D6 revolute, vertical axis, +-80 deg stops, joint damper) —
     rotating the bar aims the blade to a CONTINUOUS sampled bearing; the aim cue is a
     red BEACON standing above the goal bay, placed exactly on the pointer ray;
  2. a spring-less gravity PORTCULLIS gate on an external launch chute (D6 vertical
     slide, 85 mm travel, joint damper, gravity re-closes) — lifting it by its knob
     releases the ball, which rolls down the 15 deg chute, enters the court through a
     wall port, CAROMS off the aimed blade and rolls out through the orange doorway
     into a sealed exterior bay.

The execution ORDER is physically forced (declared in TASK.md): aim FIRST, release
SECOND. A ball released before aiming crosses the court on the port line, misses the
blade (parked in the opposite quadrant), dies against the far wall at radius ~0.27 m —
outside the blade's 0.17 m sweep — and no contact strategy can reach it under the roof.
Nothing re-cocks the chute; the episode is spent.

Mechanics: plain rigid bodies + authored USD D6 joints (the proven pattern). post_step
owns both wrench slots and consumes the `drive_vane_t` / `drive_gate_f` buffers
(solve.py's stand-ins for the Franka's wrist-roll on the pointer bar and jaw lift on
the gate knob). NOTHING is teleported during an episode; every joule the ball carries
comes from the chute drop, and every degree of aim from the driven vane.

Rubric (graded 0..1, latched, anchored in the demonstrated solve.py trajectory):
  - 0.20 * aim_latch: vane within aim_tol of the sampled bearing AND slow for
    latch_steps consecutive substeps (a fast sweep through the bearing latches nothing);
  - 0.25 * launch_latch: ball inside the court (under the roof) — gated on aim_latch,
    so releasing first earns launch credit only if the aim is completed anyway;
  - 0.25 * approach_latch: ball within 0.12 m of the computed exit point M on the wall
    circle — gated on launch_latch;
  - 0.30 * success(): ball settled INSIDE the bay volume (bay-frame box, on the floor,
    slow). score == 1.0 iff success; ~0 for the null policy (ball parked behind the
    closed gate); latched credit never evaporates under correct behaviour.

Per-episode randomization (readback-verified in smoke): the required blade bearing
theta_star (continuous, [-58, -34] deg — the whole delivery geometry moves: bay, posts,
lintel, roof, beacon and the two wall gaps are re-posed from it) and the vane's start
angle theta0 (continuous, [22, 70] deg, always >= 56 deg from every legal target).

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
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

_TAN15 = math.tan(math.radians(15.0))


def _exit_point(theta: float, y_line: float, off: float, r_in: float) -> tuple:
    """Carom geometry for one bearing (python floats — used by __post_init__ audits).

    The ball rolls along y = y_line toward +x, meets the blade line through the origin
    at angle theta, and (restitution 0) exits along the blade direction u = (cos, sin)
    displaced by `off` = ball_r + blade half-thickness along the blade normal. Returns
    (M, az_mouth) where M is the exit point on the wall circle radius r_in.
    """
    u = (math.cos(theta), math.sin(theta))
    n = (math.sin(theta), -math.cos(theta))  # toward the incoming-ball side
    ax = y_line / math.tan(theta)
    c = (ax + off * n[0], y_line + off * n[1])
    cu = c[0] * u[0] + c[1] * u[1]
    s = -cu + math.sqrt(cu * cu + r_in * r_in - (c[0] ** 2 + c[1] ** 2))
    m = (c[0] + s * u[0], c[1] + s * u[1])
    return m, math.atan2(m[1], m[0])


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CaromCourtSceneCfg(BaseCfg):
    """Config for `CaromCourtScene`. All delivery geometry is derived in `__post_init__`
    and AUDITED there (cheap float math), so the scene, the smoke and the solver read
    the same numbers and a bad tune fails at import time, not after a forge run."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    aim_tol_deg: float = tunable(7.0)  # |vane - theta_star| within this = aimed
    slow_gate: float = tunable(0.30)  # rad/s: vane counts as slow below this
    latch_steps: int = tunable(24)  # consecutive slow substeps before aim latches (0.2 s)
    launch_r: float = tunable(0.28)  # ball horizontal radius below this = in the court
    approach_d: float = tunable(0.12)  # ball within this of the exit point M
    goal_x: tuple = tunable((0.035, 0.170))  # success band, bay-frame x' (m)
    goal_y: float = tunable(0.055)  # success band, bay-frame |y'| (m)
    goal_z: float = tunable(0.090)  # ball centre at/below this = on the bay floor
    settle_lin: float = tunable(0.10)  # max ball |lin vel| when judging (above GPU creep)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    theta_star_deg: tuple = tunable((-58.0, -34.0))  # required blade bearing range
    theta0_deg: tuple = tunable((22.0, 70.0))  # vane start range (>= 56 deg from target)

    # --- tunable: plant ----------------------------------------------------------------------
    ball_mass: float = tunable(0.15)
    ball_lin_damp: float = tunable(0.05)
    ball_ang_damp: float = tunable(0.05)
    vane_mass: float = tunable(0.25)
    vane_damping: float = tunable(0.35)  # N*m*s/rad joint damper (holds aim vs impact)
    gate_mass: float = tunable(0.05)
    gate_damping: float = tunable(3.0)  # N*s/m joint damper (gravity still re-closes)

    # --- info: court (env-local coordinates; ground top at z = 0) ----------------------------
    r_in: float = info(0.30)  # inner wall radius
    wall_rc: float = info(0.31)  # wall segment centre radius
    wall_seg: tuple = info((0.088, 0.020, 0.118))  # tangential x thickness x height
    n_walls: int = info(20)
    roof_half: tuple = info((0.70, 0.335, 0.012))  # two half slabs, slot between them
    roof_zc: float = info(0.125)  # roof under 0.119, top 0.131
    roof_yc: float = info(0.1805)  # half-slab centres at +-this (slot |y| < 0.013)
    slot_w: float = info(0.026)  # roof slot width — ball (0.060) can NOT pass

    # --- info: ball --------------------------------------------------------------------------
    ball_r: float = info(0.030)
    ball_start: tuple = info((-0.567, -0.05, 0.084))  # on the incline, uphill of the gate

    # --- info: chute (fixed, along +x at y = y_line) -----------------------------------------
    y_line: float = info(-0.05)  # launch line (= ball path into the court)
    chan_half: float = info(0.036)  # rail inner faces at y_line +- this (channel 0.072)
    incline_x: tuple = info((-0.630, -0.385))  # 15 deg slope footprint
    flat_x: tuple = info((-0.400, -0.285))  # flat run-out, top z = flat_top
    flat_top: float = info(0.003)  # 3 mm step DOWN onto the court ground
    rail_th: float = info(0.010)
    rail_h: float = info(0.032)

    # --- info: gate (portcullis on the incline) ----------------------------------------------
    gate_x: float = info(-0.52)
    tab_size: tuple = info((0.012, 0.060, 0.050))
    gate_travel: float = info(0.085)  # D6 transZ limit [0, travel]
    knob_size: float = info(0.028)  # jaw target cube on top of the stem

    # --- info: vane (pointer bar + hidden blade, one body) -----------------------------------
    shaft_r: float = info(0.008)
    shaft_z: tuple = info((0.012, 0.200))  # cylinder extent (centre z = 0.106)
    blade_size: tuple = info((0.170, 0.012, 0.100))  # under the roof, z 0.005..0.105
    blade_local: tuple = info((0.085, 0.0, -0.051))
    pointer_size: tuple = info((0.170, 0.020, 0.020))  # above the roof, z 0.165..0.185
    pointer_local: tuple = info((0.085, 0.0, 0.069))
    rot_lim_deg: float = info(80.0)  # D6 rotZ end stops

    # --- info: bay + beacon (re-posed per episode from theta_star) ---------------------------
    mouth_half_w: float = info(0.085)  # wall-gap half chord at the mouth
    port_half_w: float = info(0.060)  # wall-gap half chord at the chute port
    bay_parts: tuple = info((
        # name           size (x', y', z)          local centre (x', y', z)
        ("bay_side_l", (0.160, 0.012, 0.118), (0.105, 0.066, 0.059)),
        ("bay_side_r", (0.160, 0.012, 0.118), (0.105, -0.066, 0.059)),
        ("bay_back", (0.012, 0.144, 0.118), (0.176, 0.0, 0.059)),
        ("bay_roof", (0.200, 0.180, 0.012), (0.085, 0.0, 0.124)),
        ("post_l", (0.045, 0.025, 0.118), (0.0025, 0.0725, 0.059)),
        ("post_r", (0.045, 0.025, 0.118), (0.0025, -0.0725, 0.059)),
        ("lintel", (0.045, 0.145, 0.022), (0.0025, 0.0, 0.106)),
    ))
    beacon_r: float = info(0.012)
    beacon_h: float = info(0.220)
    beacon_zc: float = info(0.245)  # bottom 0.135 — clears both roofs
    beacon_range: float = info(0.38)  # beacon stands at this radius ON the pointer ray

    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    carom_off: float = field(default=None, init=False)  # ball_r + blade half-thickness
    az_port: float = field(default=None, init=False)
    port_half: float = field(default=None, init=False)  # rad
    mouth_half: float = field(default=None, init=False)  # rad
    seg_ang: float = field(default=None, init=False)  # rad covered by one wall segment
    tab_bot: float = field(default=None, init=False)  # closed-gate tab bottom z
    gate_zc: float = field(default=None, init=False)  # closed-gate root centre z
    vane_zc: float = field(default=None, init=False)

    def floortop(self, x: float) -> float:
        """Chute surface height at x (incline section)."""
        return self.flat_top + (self.incline_x[1] - x) * _TAN15

    def __post_init__(self) -> None:
        c = self
        c.carom_off = c.ball_r + c.blade_size[1] / 2
        c.az_port = math.atan2(c.y_line, -math.sqrt(c.r_in**2 - c.y_line**2))
        c.port_half = math.asin(c.port_half_w / c.r_in)
        c.mouth_half = math.asin(c.mouth_half_w / c.r_in)
        c.seg_ang = 2 * math.asin(c.wall_seg[0] / 2 / c.wall_rc)
        c.tab_bot = c.floortop(c.gate_x) + 0.004
        c.gate_zc = c.tab_bot + c.tab_size[2] / 2
        c.vane_zc = (c.shaft_z[0] + c.shaft_z[1]) / 2

        # ---- audits: fail at import, not after a forge run ----------------------------------
        ball_d = 2 * c.ball_r
        # Passages: ball fits the channel, the port, the mouth, under the lintel; NOT the slot.
        assert ball_d < 2 * c.chan_half - 0.008, "ball must roll freely in the chute channel"
        assert ball_d < 2 * c.port_half_w - 0.04, "ball must pass the wall port"
        assert ball_d < 0.12 - 0.02, "ball must pass between the mouth posts"
        assert ball_d < (0.106 - 0.022 / 2) - 0.005, "ball must pass under the lintel"
        assert ball_d > c.slot_w + 0.02, "ball must NOT pass the roof slot (seed strategy)"
        # Vane: blade under the roof, pointer above it, stops inside the wrap-safe range.
        blade_top = c.vane_zc + c.blade_local[2] + c.blade_size[2] / 2
        assert blade_top <= c.roof_zc - c.roof_half[2] / 2 - 0.004, "blade clears roof underside"
        p_bot = c.vane_zc + c.pointer_local[2] - c.pointer_size[2] / 2
        assert p_bot >= c.roof_zc + c.roof_half[2] / 2 + 0.02, "pointer bar clears roof top"
        assert 2 * c.shaft_r < c.slot_w - 0.008, "shaft spins freely in the roof slot"
        assert c.rot_lim_deg <= 87.0, "keep the D6 well clear of the 180-deg wrap"
        lo, hi = c.theta_star_deg
        assert -c.rot_lim_deg + 2 < lo < hi < 0, "targets inside the stops, below the port line"
        assert c.theta0_deg[0] - hi >= 8 * c.aim_tol_deg, "start >> aim band from every target"
        assert c.theta0_deg[1] < c.rot_lim_deg - 2
        # Gate: full lift clears the ball; closed tab seals the channel.
        ft = c.floortop(c.gate_x)
        ball_top = ft + c.ball_r * math.cos(math.radians(15.0)) + c.ball_r
        assert c.tab_bot + c.gate_travel >= ball_top + 0.012, "lifted gate clears the ball"
        assert c.tab_bot - ft < 0.006, "closed gate leaves no gap the ball could wedge under"
        assert c.tab_size[1] < 2 * c.chan_half - 0.004, "tab slides freely between the rails"
        assert 2 * c.chan_half - c.tab_size[1] < ball_d, "ball cannot squeeze past the tab"
        # Carom geometry at both bearing extremes: blade covers the contact, the exit point
        # exists, walls cover both arcs, and the bay/beacon stay consistent.
        for th_deg in (lo, hi):
            th = math.radians(th_deg)
            a_r = abs(c.y_line / math.tan(th)) * math.sqrt(1 + math.tan(th) ** 2)
            assert a_r + c.carom_off < c.blade_size[0] - 0.02, "blade covers the carom point"
            assert a_r > 2 * c.shaft_r + c.ball_r, "carom point clear of the shaft"
            m, az_m = _exit_point(th, c.y_line, c.carom_off, c.r_in)
            # Wall arcs (port gap -> mouth gap, both directions) must be coverable.
            # az_port ~ -170 deg sits below every az_m (~ -43..-67 deg): arc 1 runs
            # directly between them; arc 2 wraps through +y (hence the +2*pi).
            a1 = (az_m - c.mouth_half) - (c.az_port + c.port_half)
            a2 = (c.az_port - c.port_half + 2 * math.pi) - (az_m + c.mouth_half)
            n1 = min(max(round(c.n_walls * a1 / (a1 + a2)), 4), 8)
            assert n1 * c.seg_ang >= a1 and (c.n_walls - n1) * c.seg_ang >= a2, \
                "wall segments must cover both arcs"
            # Beacon on the pointer ray sits inside the bay roof footprint (visual anchor).
            u = (math.cos(th), math.sin(th))
            b = (c.beacon_range * u[0], c.beacon_range * u[1])
            bx = (b[0] - m[0]) * u[0] + (b[1] - m[1]) * u[1]
            assert 0.0 < bx < c.bay_parts[2][2][0], "beacon stands over the bay"
        assert c.beacon_zc - c.beacon_h / 2 >= c.roof_zc + c.roof_half[2] / 2 + 0.003
        # Direct-shot audit: the released ball starts from rest and is guided by the
        # channel from the gate to the rail end (the incline is pitched purely along x,
        # so gravity adds no lateral velocity); the extreme reachable wall azimuth must
        # stay >= 6 deg from the nearest mouth edge at the WORST bearing (hi, mouth
        # closest to the port line). The spread bound (lateral play over the guided
        # gate->exit length) is already generous for a from-rest release.
        spread = math.atan((2 * c.chan_half - ball_d) / (c.flat_x[1] - c.gate_x))
        y0 = c.y_line - (c.chan_half - c.ball_r)
        x0 = -math.sqrt(c.r_in**2 - c.y_line**2)
        d = (math.cos(-spread), math.sin(-spread))
        cu = x0 * d[0] + y0 * d[1]
        t = -cu + math.sqrt(cu * cu + c.r_in**2 - (x0 * x0 + y0 * y0))
        az_reach = math.atan2(y0 + t * d[1], x0 + t * d[0])
        _, az_hi = _exit_point(math.radians(hi), c.y_line, c.carom_off, c.r_in)
        margin = math.degrees((az_hi + c.mouth_half) - az_reach) * -1.0
        assert margin >= 6.0, f"direct hand-rolled shot must miss the mouth ({margin:.1f} deg)"
        # Physically forced order: a stranded (unaimed) ball dies beyond the blade sweep.
        assert c.r_in - c.ball_r - 0.02 > c.blade_size[0], "stranded ball is out of blade reach"


def _quat_z(rad: torch.Tensor) -> torch.Tensor:
    """(N,) angle about +z -> (N, 4) wxyz."""
    half = rad / 2
    q = torch.zeros(rad.shape[0], 4, device=rad.device)
    q[:, 0] = torch.cos(half)
    q[:, 3] = torch.sin(half)
    return q


def _wrap(a: torch.Tensor) -> torch.Tensor:
    """Wrap angles to (-pi, pi]."""
    return torch.atan2(torch.sin(a), torch.cos(a))


# ----- scene -----------------------------------------------------------------------------------
class CaromCourtScene(BaseScene):
    cfg: CaromCourtSceneCfg

    def __init__(self, cfg: CaromCourtSceneCfg | None = None) -> None:
        super().__init__(cfg or CaromCourtSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, a fixed kinematic rig (roof halves + chute, authored in bind),
        20 kinematic wall segments + 7 bay slabs + beacon (all re-posed at reset from the
        sampled bearing), and three dynamic bodies: ball, vane, gate."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        wallmat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.16, 0.17, 0.20))
        orange = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.45, 0.05))
        red = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.06, 0.06))
        steel = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.58))
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        live = dict(sleep_threshold=0.0, stabilization_threshold=0.0)
        dead = sim_utils.RigidBodyMaterialCfg(  # dead-impact surfaces: carom keeps tangential
            static_friction=0.35, dynamic_friction=0.30, restitution=0.0)

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.35, dynamic_friction=0.30, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            # Fixed rig root (buried marker; the visible rig is authored in bind).
            "rig": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rig",
                spawn=sim_utils.CuboidCfg(
                    size=(0.01, 0.01, 0.01), rigid_props=kin, collision_props=coll,
                    visual_material=wallmat),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.03)),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        linear_damping=c.ball_lin_damp,
                        angular_damping=c.ball_ang_damp,
                        **live),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    collision_props=coll,
                    physics_material=dead,
                    visual_material=orange,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.ball_start),
            ),
            "vane": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Vane",
                spawn=sim_utils.CylinderCfg(
                    radius=c.shaft_r, height=c.shaft_z[1] - c.shaft_z[0], axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        **live),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.vane_mass),
                    collision_props=coll,
                    physics_material=dead,
                    visual_material=steel,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.vane_zc)),
            ),
            "gate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Gate",
                spawn=sim_utils.CuboidCfg(
                    size=c.tab_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        **live),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.gate_mass),
                    collision_props=coll,
                    physics_material=dead,
                    visual_material=steel,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.gate_x, c.y_line, c.gate_zc)),
            ),
            "beacon": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beacon",
                spawn=sim_utils.CylinderCfg(
                    radius=c.beacon_r, height=c.beacon_h, axis="Z",
                    rigid_props=kin, collision_props=coll, visual_material=red),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.beacon_range, 0.0, c.beacon_zc)),
            ),
        }
        for i in range(c.n_walls):
            out[f"wall{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + f"Wall{i}",
                spawn=sim_utils.CuboidCfg(
                    size=c.wall_seg, rigid_props=kin, collision_props=coll,
                    physics_material=dead, visual_material=wallmat),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.wall_rc, 0.05 * i - 0.5, c.wall_seg[2] / 2)),
            )
        for name, size, _local in c.bay_parts:
            mat = orange if name.startswith(("post", "lintel")) else wallmat
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.title().replace("_", ""),
                spawn=sim_utils.CuboidCfg(
                    size=size, rigid_props=kin, collision_props=coll,
                    physics_material=dead, visual_material=mat),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, size[2] / 2)),
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
        self.ball: RigidObject = env.iscene["ball"]
        self.vane: RigidObject = env.iscene["vane"]
        self.gate: RigidObject = env.iscene["gate"]
        self.beacon: RigidObject = env.iscene["beacon"]
        self.walls: list[RigidObject] = [env.iscene[f"wall{i}"] for i in range(c.n_walls)]
        self.bay: dict[str, RigidObject] = {nm: env.iscene[nm] for nm, _s, _l in c.bay_parts}
        self.env_origins = env.iscene.env_origins
        self._author_rig()
        self._author_compounds()
        self._author_joints()
        # Episode state.
        self.theta_star = torch.zeros(n, device=dev)  # required blade bearing (rad)
        self.theta0 = torch.zeros(n, device=dev)  # vane start angle (rad)
        self.exit_m = torch.zeros(n, 2, device=dev)  # exit point M on the wall circle
        self.aim_latch = torch.zeros(n, device=dev)
        self.launch_latch = torch.zeros(n, device=dev)
        self.approach_latch = torch.zeros(n, device=dev)
        self.slow_ctr = torch.zeros(n, dtype=torch.long, device=dev)
        # External drive input (solve.py and smoke probes write; post_step consumes and
        # OWNS both wrench slots — never call set_external_force_and_torque directly).
        self.drive_vane_t = torch.zeros(n, device=dev)  # torque about the vane axis (N*m)
        self.drive_gate_f = torch.zeros(n, device=dev)  # vertical force on the gate (N)
        # WORLD-frame force on the ball (smoke probes only): the ball spins while it
        # rolls, so post_step re-encodes this into the body frame every substep.
        self.drive_ball_f = torch.zeros(n, 3, device=dev)

    def _author_rig(self) -> None:
        """Fixed rig, authored once as children of env_*/Rig (idempotent): the two roof
        half-slabs and the launch chute (incline, flat run-out, rails, backstop)."""
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath("/World/envs/env_0/Rig/roof_p").IsValid():
            return
        root_pos = (0.0, 0.0, -0.03)  # rig root spawn centre (locals are world - this)

        def box(root: str, name: str, world_c: tuple, size: tuple, color: tuple,
                pitch_deg: float = 0.0, collide: bool = True) -> None:
            cube = UsdGeom.Cube.Define(stage, f"{root}/{name}")
            cube.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(cube.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(world_c[0] - root_pos[0],
                                             world_c[1] - root_pos[1],
                                             world_c[2] - root_pos[2]))
            if pitch_deg:
                xf.AddRotateYOp().Set(pitch_deg)
            xf.AddScaleOp().Set(Gf.Vec3f(*size))
            cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            if collide:
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                px = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
                px.CreateContactOffsetAttr(c.contact_offset)
                px.CreateRestOffsetAttr(0.0)

        gray = (0.35, 0.36, 0.40)
        wood = (0.48, 0.35, 0.20)
        x_lo, x_hi = c.incline_x
        mid = (x_lo + x_hi) / 2
        slab_l = (x_hi - x_lo) / math.cos(math.radians(15.0)) + 0.006
        top_mid = (c.floortop(x_lo) + c.flat_top) / 2
        nrm = (math.sin(math.radians(15.0)), 0.0, math.cos(math.radians(15.0)))
        for i in range(self.env.num_envs):
            root = f"/World/envs/env_{i}/Rig"
            # roof halves (slot |y| < slot_w/2 between them, for the vane shaft)
            for sgn, nm in ((1.0, "roof_p"), (-1.0, "roof_n")):
                box(root, nm, (0.0, sgn * c.roof_yc, c.roof_zc), c.roof_half, gray)
            # chute: incline slab (pitched +15 deg: +x end lower), flat run-out
            box(root, "incline",
                (mid - 0.006 * nrm[0], c.y_line, top_mid - 0.006 * nrm[2]),
                (slab_l, 0.100, 0.012), wood, pitch_deg=15.0)
            box(root, "flat",
                ((c.flat_x[0] + c.flat_x[1]) / 2, c.y_line, c.flat_top - 0.003),
                (c.flat_x[1] - c.flat_x[0], 0.100, 0.006), wood)
            # rails: incline-following + flat sections, inner faces at y_line +- chan_half
            for sgn in (1.0, -1.0):
                ry = c.y_line + sgn * (c.chan_half + c.rail_th / 2)
                rc = (mid + (0.006 + c.rail_h / 2) * nrm[0], ry,
                      top_mid + (0.006 + c.rail_h / 2) * nrm[2] - 0.006)
                box(root, f"rail_i{'p' if sgn > 0 else 'n'}", rc,
                    (slab_l, c.rail_th, c.rail_h), gray, pitch_deg=15.0)
                box(root, f"rail_f{'p' if sgn > 0 else 'n'}",
                    ((c.flat_x[0] + c.flat_x[1]) / 2, ry, c.flat_top + c.rail_h / 2 - 0.002),
                    (c.flat_x[1] - c.flat_x[0], c.rail_th, c.rail_h), gray)
            # backstop at the top of the incline
            box(root, "stop", (x_lo - 0.018, c.y_line, c.floortop(x_lo - 0.006) + 0.02),
                (0.024, 0.100, 0.080), gray)

    def _author_compounds(self) -> None:
        """Compound children (idempotent, per env): the vane's hidden blade + pointer bar
        + red tip, and the gate's stem + knob."""
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath("/World/envs/env_0/Vane/blade").IsValid():
            return

        def box(root: str, name: str, center: tuple, size: tuple, color: tuple,
                collide: bool = True) -> None:
            cube = UsdGeom.Cube.Define(stage, f"{root}/{name}")
            cube.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(cube.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(*center))
            xf.AddScaleOp().Set(Gf.Vec3f(*size))
            cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            if collide:
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                px = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
                px.CreateContactOffsetAttr(c.contact_offset)
                px.CreateRestOffsetAttr(0.0)

        for i in range(self.env.num_envs):
            vroot = f"/World/envs/env_{i}/Vane"
            box(vroot, "blade", c.blade_local, c.blade_size, (0.25, 0.25, 0.28))
            box(vroot, "pointer", c.pointer_local, c.pointer_size, (0.80, 0.06, 0.06))
            box(vroot, "tip",
                (c.pointer_local[0] + 0.073, 0.0, c.pointer_local[2] + 0.017),
                (0.024, 0.026, 0.014), (0.95, 0.10, 0.10), collide=False)
            groot = f"/World/envs/env_{i}/Gate"
            box(groot, "stem", (0.0, 0.0, c.tab_size[2] / 2 + 0.047),
                (0.012, 0.012, 0.094), (0.55, 0.55, 0.58))
            box(groot, "knob", (0.0, 0.0, c.tab_size[2] / 2 + 0.094 + c.knob_size / 2),
                (c.knob_size,) * 3, (0.95, 0.45, 0.05))

    def _author_joints(self) -> None:
        """Per env: two world-anchored D6 joints (body1-only; LocalPos0 is the WORLD
        anchor incl. the env origin). Vane: rotZ free within +-rot_lim_deg, joint damper
        holds the aim against ball impact. Gate: transZ free within [0, travel], damper
        smooths the slide, gravity re-closes it."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        origins = self.env_origins.cpu().numpy()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            ox, oy, oz = (float(v) for v in origins[i])
            # --- vane revolute ---
            j = UsdPhysics.Joint.Define(stage, f"{base}/vane_joint")
            j.CreateBody1Rel().SetTargets([f"{base}/Vane"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateLocalPos0Attr(Gf.Vec3f(ox, oy, oz + c.vane_zc))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            for axis in ("transX", "transY", "transZ", "rotX", "rotY"):
                lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), axis)
                lim.CreateLowAttr(1.0)  # low > high = locked
                lim.CreateHighAttr(-1.0)
            lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), "rotZ")
            lim.CreateLowAttr(-c.rot_lim_deg)
            lim.CreateHighAttr(c.rot_lim_deg)
            drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "rotZ")
            drv.CreateTypeAttr("force")
            drv.CreateStiffnessAttr(0.0)
            # USD angular drives are per-DEGREE: convert the cfg's N*m*s/rad value.
            drv.CreateDampingAttr(c.vane_damping / 57.29578)
            drv.CreateTargetVelocityAttr(0.0)
            # --- gate prismatic ---
            j = UsdPhysics.Joint.Define(stage, f"{base}/gate_joint")
            j.CreateBody1Rel().SetTargets([f"{base}/Gate"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateLocalPos0Attr(Gf.Vec3f(ox + c.gate_x, oy + c.y_line, oz + c.gate_zc))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            for axis in ("transX", "transY", "rotX", "rotY", "rotZ"):
                lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), axis)
                lim.CreateLowAttr(1.0)
                lim.CreateHighAttr(-1.0)
            lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), "transZ")
            lim.CreateLowAttr(0.0)
            lim.CreateHighAttr(c.gate_travel)
            drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "transZ")
            drv.CreateTypeAttr("force")
            drv.CreateStiffnessAttr(0.0)
            drv.CreateDampingAttr(c.gate_damping)
            drv.CreateTargetVelocityAttr(0.0)

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the bearing theta_star and the vane start theta0
        (torch.rand only), derive the exit point M, and pose EVERYTHING from it: the
        20 wall segments (leaving the port gap and the mouth gap), the 7 bay slabs and
        the beacon (bay frame = (M, yaw = theta_star)), the vane at theta0, the gate
        closed, the ball parked on the incline behind the gate. Clear latches/drives."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        lo, hi = (math.radians(v) for v in c.theta_star_deg)
        th = lo + torch.rand(m, device=dev) * (hi - lo)
        s_lo, s_hi = (math.radians(v) for v in c.theta0_deg)
        th0 = s_lo + torch.rand(m, device=dev) * (s_hi - s_lo)

        # Exit point M (vectorized mirror of _exit_point).
        u = torch.stack([torch.cos(th), torch.sin(th)], dim=1)
        nrm = torch.stack([torch.sin(th), -torch.cos(th)], dim=1)
        ax = c.y_line / torch.tan(th)
        cpt = torch.stack([ax, torch.full_like(ax, c.y_line)], dim=1) + c.carom_off * nrm
        cu = (cpt * u).sum(dim=1)
        s = -cu + torch.sqrt(cu * cu + c.r_in**2 - (cpt * cpt).sum(dim=1))
        mm = cpt + s.unsqueeze(1) * u
        az_m = torch.atan2(mm[:, 1], mm[:, 0])

        self.theta_star[env_ids] = th
        self.theta0[env_ids] = th0
        self.exit_m[env_ids] = mm
        for k in (self.aim_latch, self.launch_latch, self.approach_latch):
            k[env_ids] = 0.0
        self.slow_ctr[env_ids] = 0
        self.drive_vane_t[env_ids] = 0.0
        self.drive_gate_f[env_ids] = 0.0
        self.drive_ball_f[env_ids] = 0.0

        def pose(body: RigidObject, px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor,
                 yaw: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = px
            st[:, 1] = py
            st[:, 2] = pz
            st[:, 3:7] = _quat_z(yaw)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- walls: two arcs between the port gap and the mouth gap ---
        two_pi = 2 * math.pi
        a1s = c.az_port + c.port_half  # port -> mouth (directly, no wrap)
        a1 = (az_m - c.mouth_half) - a1s
        a2s = az_m + c.mouth_half  # mouth -> port (through +y side)
        a2 = (c.az_port - c.port_half + two_pi) - a2s
        n1 = torch.round(c.n_walls * a1 / (a1 + a2)).clamp(4, 8).long()
        zc = torch.full((m,), c.wall_seg[2] / 2, device=dev)
        for i in range(c.n_walls):
            i_t = torch.full((m,), float(i), device=dev)
            in1 = i_t < n1.float()
            az = torch.where(
                in1,
                a1s + (i_t + 0.5) * a1 / n1.float(),
                a2s + (i_t - n1.float() + 0.5) * a2 / (c.n_walls - n1).float())
            pose(self.walls[i], c.wall_rc * torch.cos(az), c.wall_rc * torch.sin(az),
                 zc, az + math.pi / 2)

        # --- bay slabs + beacon in the (M, yaw=theta_star) frame ---
        for name, _size, loc in c.bay_parts:
            px = mm[:, 0] + loc[0] * u[:, 0] - loc[1] * u[:, 1]
            py = mm[:, 1] + loc[0] * u[:, 1] + loc[1] * u[:, 0]
            pose(self.bay[name], px, py, torch.full((m,), loc[2], device=dev), th)
        pose(self.beacon, c.beacon_range * u[:, 0], c.beacon_range * u[:, 1],
             torch.full((m,), c.beacon_zc, device=dev), th)

        # --- dynamic bodies: vane at theta0, gate closed, ball parked on the incline ---
        z = torch.zeros(m, device=dev)
        pose(self.vane, z, z, z + c.vane_zc, th0)
        pose(self.gate, z + c.gate_x, z + c.y_line, z + c.gate_zc, z)
        pose(self.ball, z + c.ball_start[0], z + c.ball_start[1], z + c.ball_start[2], z)

    # ----- readings -----------------------------------------------------------------------------
    def vane_yaw(self) -> torch.Tensor:
        """(N,) vane rotation about +z in rad (the D6 frees only rotZ)."""
        q = self.vane.data.root_quat_w
        return _wrap(2.0 * torch.atan2(q[:, 3], q[:, 0]))

    def aim_err(self) -> torch.Tensor:
        """(N,) |vane bearing - theta_star| (rad)."""
        return _wrap(self.vane_yaw() - self.theta_star).abs()

    def aimed_now(self) -> torch.Tensor:
        """(N,) bool: CURRENT |aim error| < tol and the vane is slow."""
        return (self.aim_err() < math.radians(self.cfg.aim_tol_deg)) \
            & (self.vane.data.root_ang_vel_w[:, 2].abs() < self.cfg.slow_gate)

    def ball_local(self) -> torch.Tensor:
        """(N, 3) ball position in env-local coordinates."""
        return self.ball.data.root_pos_w - self.env_origins

    def bay_coords(self) -> torch.Tensor:
        """(N, 2) ball position in the bay frame (x' along the exit direction from M)."""
        p = self.ball_local()[:, :2] - self.exit_m
        cth = torch.cos(self.theta_star)
        sth = torch.sin(self.theta_star)
        return torch.stack([p[:, 0] * cth + p[:, 1] * sth,
                            -p[:, 0] * sth + p[:, 1] * cth], dim=1)

    def gate_lift(self) -> torch.Tensor:
        """(N,) gate lift above the closed position (m)."""
        return (self.gate.data.root_pos_w[:, 2] - self.env_origins[:, 2]) - self.cfg.gate_zc

    def success(self) -> torch.Tensor:
        """(N,) bool: ball resting INSIDE the bay — bay-frame box on the bay floor,
        slow (current, physical state; nothing about how it got there is waived —
        the geometry makes the doorway the only way in)."""
        c = self.cfg
        b = self.bay_coords()
        p = self.ball_local()
        v = self.ball.data.root_lin_vel_w.norm(dim=-1)
        return (b[:, 0] >= c.goal_x[0]) & (b[:, 0] <= c.goal_x[1]) \
            & (b[:, 1].abs() <= c.goal_y) & (p[:, 2] <= c.goal_z) & (v < c.settle_lin)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.20*aim_latch + 0.25*launch_latch + 0.25*approach_latch
        + 0.30*success. The latch chain is order-gated (launch needs aim, approach needs
        launch); ~0 for the null policy; exactly 1.0 once success holds."""
        return 0.20 * self.aim_latch + 0.25 * self.launch_latch \
            + 0.25 * self.approach_latch + 0.30 * self.success().float()

    # ----- step-coupled mechanics (every substep) ------------------------------------------------
    def post_step(self) -> None:
        """Consume the drive buffers (owns the wrench slots; the vane only yaws and the
        gate only heaves, so body frame == world frame for both wrenches), then advance
        the order-gated latch chain through the aim slow-gate counter."""
        n = self.env.num_envs
        dev = self.env.device
        c = self.cfg
        f = torch.zeros(n, 1, 3, device=dev)
        t = torch.zeros(n, 1, 3, device=dev)
        t[:, 0, 2] = self.drive_vane_t
        self.vane.set_external_force_and_torque(f, t)
        f = torch.zeros(n, 1, 3, device=dev)
        f[:, 0, 2] = self.drive_gate_f
        self.gate.set_external_force_and_torque(f, torch.zeros(n, 1, 3, device=dev))
        # Ball probe force is WORLD-frame; the wrench API is body-frame and the ball
        # spins as it rolls — re-encode per substep with the current orientation.
        q = self.ball.data.root_quat_w  # (N, 4) wxyz
        qv = q[:, 1:]
        tt = 2.0 * torch.cross(qv, self.drive_ball_f, dim=-1)
        fb = self.drive_ball_f - q[:, :1] * tt + torch.cross(qv, tt, dim=-1)
        f = torch.zeros(n, 1, 3, device=dev)
        f[:, 0, :] = fb
        self.ball.set_external_force_and_torque(f, torch.zeros(n, 1, 3, device=dev))

        # Aim latch: slow-gate counter — sweeping through the bearing latches nothing.
        near = self.aim_err() < math.radians(c.aim_tol_deg)
        slow = self.vane.data.root_ang_vel_w[:, 2].abs() < c.slow_gate
        self.slow_ctr = torch.where(near & slow, self.slow_ctr + 1,
                                    torch.zeros_like(self.slow_ctr))
        aim_ok = (self.slow_ctr >= c.latch_steps).float()
        aim_ok = torch.nan_to_num(aim_ok, nan=0.0)  # a diverged substep must not latch
        self.aim_latch = torch.maximum(self.aim_latch, aim_ok)
        # Launch latch (gated on aim): ball inside the court, under the roof.
        p = self.ball_local()
        r_xy = p[:, :2].norm(dim=-1)
        in_court = (r_xy < c.launch_r) & (p[:, 2] < 0.115) & (p[:, 2] > -0.05)
        lv = torch.nan_to_num((in_court.float() * self.aim_latch), nan=0.0)
        self.launch_latch = torch.maximum(self.launch_latch, lv)
        # Approach latch (gated on launch): ball near the exit point M.
        d = (p[:, :2] - self.exit_m).norm(dim=-1)
        av = torch.nan_to_num(((d < c.approach_d).float() * self.launch_latch), nan=0.0)
        self.approach_latch = torch.maximum(self.approach_latch, av)

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = self._bodies()
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("theta_star", "theta0", "exit_m", "aim_latch", "launch_latch",
                               "approach_latch", "slow_ctr", "drive_vane_t", "drive_gate_f",
                               "drive_ball_f")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = self._bodies()
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    def _bodies(self) -> dict[str, Any]:
        out = {"ball": self.ball, "vane": self.vane, "gate": self.gate, "beacon": self.beacon}
        out.update({f"wall{i}": w for i, w in enumerate(self.walls)})
        out.update(self.bay)
        return out

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A round court ({c.r_in * 200:.0f} cm across) is completely sealed: dark walls "
            f"all around and a flat gray roof over the whole floor — the only roof opening "
            f"is a narrow slot for the centre shaft, far too small for the ball. The orange "
            f"BALL ({c.ball_r * 200:.0f} cm) waits OUTSIDE the court, parked on a raised "
            f"wooden launch chute behind a steel PORTCULLIS GATE; the gate slides straight "
            f"up by its orange KNOB and falls closed again when released. Where the chute "
            f"meets the court wall there is a port; directly across the court, one section "
            f"of wall is replaced by an orange DOORWAY (two posts and a lintel) leading into "
            f"a small roofed BAY, and a tall red BEACON stands above the bay. At the court "
            f"centre, a POINTER BAR with a red tip rides above the roof on the shaft; the "
            f"shaft also carries a hidden deflector blade under the roof, rigid with the "
            f"bar. The bar spins freely between end stops and stays where it is left.\n"
            f"Goal: put the ball inside the bay. The hand can never touch the ball — it is "
            f"behind the gate and then under the roof — so the delivery is indirect: FIRST "
            f"rotate the pointer bar until its red tip points at the red beacon (the beacon "
            f"stands exactly on the correct pointer ray; both the beacon bearing and the "
            f"bar's starting angle are sampled fresh every episode), THEN lift the gate and "
            f"hold it up. The ball rolls down the chute, enters through the port, banks off "
            f"the hidden blade and rolls through the doorway into the bay. The order is "
            f"forced: a ball released before aiming crosses to the far wall and dies there, "
            f"out of the blade's reach, and nothing can re-cock the chute. Sweeping the bar "
            f"past the beacon without stopping counts for nothing — only where it RESTS "
            f"aims the blade."
        )

    def instruction(self) -> str:
        return (
            "Rotate the overhead pointer bar until its red tip rests pointing at the red "
            "beacon, then lift the portcullis gate by its knob and hold it up so the ball "
            "rolls down the chute, banks off the hidden blade, and stops inside the bay "
            "behind the orange doorway."
        )


# Guarded registration: the forge may import this module under two names.
if "carom_court" not in SCENES.list():
    SCENES.register("carom_court", CaromCourtScene)
if "simgen.carom_court" not in ENVS.list():
    register_env("simgen", lambda: EnvCfg(scene="carom_court", robot="null"))

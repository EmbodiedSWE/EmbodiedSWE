"""ChannelFlipboardScene — flip the TV's channel cards until the target color faces front.

Derived from the RLBench `change_channel` seed but STRATEGICALLY DIFFERENT (see TASK.md):
the seed's plan is "pick the hand-held TV remote off the table, aim it at the TV, press one
channel button" — one grasp, one transport, one discrete press on a free-standing device.
Here NOTHING is hand-held and nothing is pressed: the channel indicator is a mechanical
FLIP-BOARD on top of the TV cabinet — four rigid colored cards hinged on a stack of
parallel horizontal axles over a ridge, resting like a card tent (front slope / back slope).
The channel currently "playing" is the color of the OUTERMOST card hanging on the FRONT of
the board; a colored tile in the program-guide frame on the cabinet top names the target.
**Goal (carried here, no task layer): flip cards over the top rail — in the physically
forced outer-card-first order, forward or backward — until the target color faces front,
and leave every card settled on its side.** A solver needs an ordered sequence of
gravity-assisted over-center flips of stacked hinged plates, not a grasp-aim-press.

Mechanics (plain rigid bodies + authored USD revolute joints, the proven oven_dials
lineage): card i hangs from its own Y-axis revolute joint anchored to the kinematic
cabinet, axes stacked vertically (card 0 highest = outermost on BOTH slopes). Each slab
hangs on end pins at a RADIAL GAP `r_gap` below its axle (the joint anchor sits r_gap
beyond the slab's upper edge), so a card sweeping over the top clears the resting cards
of HIGHER axles: their slabs' inner-edge corners fall inside the swept annulus HOLE
(clearance audited per axle pair in __post_init__ — without the gap a crossing card jams
on the axle stack; measured: a gapless card stalls at +21 deg). Cards of LOWER axles
hang entirely outside the sweep sector (|direction| > rest_deg), so the current
outermost card of a slope always crosses cleanly. Cards rest ON their joint limits at
+-`rest_deg` from vertical-up (or propped well shy of the stop on a card already on
that slope — still categorical); between the limits the only equilibrium is the
unstable apex, so every card ends categorically on the front (+x) or the back (-x).
Flipping an inner card is impossible alone: lifting it presses its slab face out-and-up
into the slab of every card resting outside it (contact from ~15 deg of travel) and
drags them along (proved by a smoke probe at the solve's own torque cap). And a card
that another card has PROPPED ON is locked the other way: lifting it carries the rider
geometrically — from ~99 deg up to the apex the lifter's root corner sits inside the
rider's swept annulus band (cos(phi) > -axis_dz/(2*r_gap)), so the rider cannot shed
and would be shoved over the top (verified in sim: the pair limit-cycles below the
apex). Together these make the board PLAN-LEVEL ONE-WAY: a higher channel is reached
by peeling the front slope back-ward outer-first; the only reachable lower channel is
1 (clear the whole back slope front-ward). reset() samples only reachable (start,
target) pairs, and an overshot card is recoverable only while nothing has landed on
top of it — so a solver must read the goal tile FIRST and flip exactly the required
cards. `post_step` applies
viscous hinge friction and consumes external drive/force buffers (`card_drive` about each
hinge axis, `card_force` at each card's CoM); writers are solve.py's fingertip-scale flip
drive and smoke's probes.

Rubric (graded 0..1, latched flip credit — anchored in the demonstrated solve.py
trajectory, which flips the required cards one at a time and settles):
  - per episode, channels start(s) -> target(t) define the REQUIRED MOVERS: cards
    [s, t) flipping front->back if t > s, cards [t, s) flipping back->front if t < s;
  - `flip_latch[i]` latches the first time a required mover is observed past
    `latch_deg` on its GOAL side (crossing the apex commits the card — gravity finishes);
  - success(): every card rests on its partition side for channel t (cards < t on the
    back, cards >= t on the front, all beyond `side_min_deg`) AND everything is settled;
  - score = 0.55 * (latched movers / required movers) + 0.45 * success() -> ~0 for the
    null policy, exactly 1.0 iff success(), 0.55 retained if a finished board is later
    knocked off (latched credit does not evaporate).

Per-episode randomization: start channel uniform; target channel uniform over the
REACHABLE set for that start (all higher channels, plus channel 1) — so the flip
direction AND the number of flips vary; card start jitter into the stops; and the guide
tile follows the target color — a memorized fixed flip sequence fails.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering the
scene — stays app-free.
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
class ChannelFlipboardSceneCfg(BaseCfg):
    """Config for `ChannelFlipboardScene`. Geometry is derived once in `__post_init__` so the
    scene, the smoke AND the solver read the same numbers."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    side_min_deg: float = tunable(30.0)  # a card counts as ON a side beyond this from apex
    latch_deg: float = tunable(35.0)  # flip credit latches past this on the goal side
    settle_omega: float = tunable(0.60)  # max |ang vel| (rad/s) when judging settled
    settle_speed: float = tunable(0.08)  # max |lin vel| (m/s) when judging settled

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    start_jitter_deg: float = tunable(2.0)  # cards start 1..1+this INSIDE their stop
    # (small on purpose: same-slope cards jittered further apart than ~4 deg would
    # interpenetrate at spawn — the planes converge over the axle offset)

    # --- tunable: card plant (difficulty dials) ----------------------------------------------
    # Hinge viscous friction sized against the fall from the apex: gravity torque peak
    # m*g*(r_gap + L/2) = 0.059 N*m, so terminal fall rate ~ 0.059/visc ~ 5 rad/s — a
    # firm but tunnel-safe landing on the stops (edge speed ~0.9 m/s). Stability at
    # 120 Hz: visc*dt/I = 0.012/120/7.8e-4 ~ 0.13 < 1.
    hinge_visc: float = tunable(0.012)  # viscous hinge friction (N*m*s/rad)
    card_mass: float = tunable(0.05)

    # --- info: structure ----------------------------------------------------------------------
    n_cards: int = info(4)  # channels 1..4, card index 0..3 (card 0 outermost)
    rest_deg: float = info(125.0)  # joint limits; cards rest ON the stops (+front/-back)
    card_l: float = info(0.12)  # slab length (radial extent of the solid card)
    r_gap: float = info(0.06)  # axle-to-slab radial gap (end-pin suspension; puts the
    # other cards' inner corners inside the swept annulus HOLE during a crossing)
    card_w: float = info(0.10)  # width along the hinge axis
    card_th: float = info(0.006)
    axis_x: float = info(0.45)  # all hinge axes on this vertical line
    axis_z0: float = info(0.36)  # LOWEST axis (innermost card, index n-1)
    axis_dz: float = info(0.018)  # vertical spacing between consecutive axes
    cabinet_pos: tuple = info((0.45, 0.0, 0.10))  # kinematic TV cabinet centre
    cabinet_size: tuple = info((0.30, 0.44, 0.20))  # front face (+x) carries the dark screen
    cheek_w: float = info(0.012)  # the two axle-carrier cheek plates
    cheek_dy: float = info(0.075)  # cheek centre |y|; cards (half-w 0.05) clear by 2 cm
    tile_size: float = info(0.035)  # program-guide tile (kinematic, colored like a card)
    tile_y: float = info(0.16)  # guide frame centre on the cabinet top, +y side
    contact_offset: float = info(0.002)
    colors: tuple = info((
        ("red", (0.82, 0.10, 0.10)),
        ("green", (0.10, 0.62, 0.16)),
        ("blue", (0.15, 0.32, 0.85)),
        ("yellow", (0.92, 0.80, 0.10)),
    ))
    # Rubric weights: 0.55 latched flip credit + 0.45 live goal state = 1.0 iff success.
    w_flip: float = info(0.55)
    w_goal: float = info(0.45)

    # Derived (filled in __post_init__).
    cabinet_top_z: float = field(default=None, init=False)
    axis_zs: tuple = field(default=None, init=False)  # per card, index order (0 = highest)
    tile_pos: tuple = field(default=None, init=False)
    tile_depot: tuple = field(default=None, init=False)  # spare tiles hide INSIDE the cabinet

    def __post_init__(self) -> None:
        n = self.n_cards
        self.cabinet_top_z = self.cabinet_pos[2] + self.cabinet_size[2] / 2
        self.axis_zs = tuple(self.axis_z0 + (n - 1 - i) * self.axis_dz for i in range(n))
        self.tile_pos = (self.axis_x, self.tile_y,
                         self.cabinet_top_z + self.tile_size / 2 + 0.001)
        self.tile_depot = (self.cabinet_pos[0], -0.10, self.cabinet_pos[2])
        # -- design invariants (audited once; smoke re-checks the physical versions) --
        rest = math.radians(self.rest_deg)
        # resting cards are parallel planes separated by axis_dz*sin(rest): they must NOT
        # touch (a touching stack would prop cards off their joint stops)
        gap = self.axis_dz * math.sin(rest) - self.card_th
        assert gap > 0.003, f"resting cards would touch (gap {gap * 1000:.1f} mm)"
        # top-crossing audit, per axle pair (the reason r_gap exists). A card sweeps the
        # annulus [r_gap, r_gap+card_l] around its own axle across the whole sector
        # |phi| <= rest. It crosses cleanly iff for every OTHER card k:
        #  (a) axle_k ABOVE: card k's slab inner-edge corner (radius r_gap from axle_k,
        #      hanging at +-rest) falls inside the sweeping card's annulus HOLE with
        #      thickness margin — the sweep passes AROUND it;
        #  (b) axle_k BELOW: card k's whole resting slab hangs outside the sweep sector.
        # radial clearance vs the resting corner's half-thickness + 3 mm air
        need = self.card_th / 2 + 0.003
        cb = math.cos(rest)  # < 0
        for k in range(1, self.n_cards):
            dz = k * self.axis_dz
            # (a) corner-in-hole clearance (law of cosines)
            corner_r = math.sqrt(self.r_gap**2 + dz**2 + 2.0 * self.r_gap * dz * cb)
            assert self.r_gap - corner_r > need, \
                f"axle pair dz={dz * 1000:.0f}mm: corner-in-hole clearance " \
                f"{(self.r_gap - corner_r) * 1000:.1f} mm too small"
            # the higher slab becomes touchable only from its hole-exit direction on —
            # that must sit well down the slope, so slab-on-slab contact is ONLY the
            # intended pile-prop / drag-interlock near the slopes, never near the top
            u_exit = (-2 * dz * cb + math.sqrt(4 * dz * dz * cb * cb
                                               - 4 * (dz * dz - self.r_gap**2))) / 2
            exit_dir = math.degrees(math.atan2(u_exit * math.sin(rest),
                                               dz + u_exit * cb))  # signed z: in (0,180)
            assert exit_dir > 65.0, \
                f"axle pair dz={dz * 1000:.0f}mm: slab contact possible at {exit_dir:.0f} deg"
            # (b) lower resting slab outside the sweep sector. Its direction seen from
            # the upper axle approaches rest_deg as radius grows, so the worst point is
            # the slab's OUTER edge.
            big_r = self.r_gap + self.card_l
            ang = math.degrees(math.atan2(big_r * math.sin(rest),
                                          big_r * abs(math.cos(rest)) + dz))
            assert 180.0 - ang > self.rest_deg + 2.0, \
                f"axle pair dz={dz * 1000:.0f}mm: lower card enters the sweep sector"
        # the free edges (the lowest swept points) stay clear above the cabinet top
        edge_z = self.axis_z0 + (self.r_gap + self.card_l) * math.cos(rest)
        assert edge_z - self.cabinet_top_z > 0.05, "no finger room under the card edges"
        # the guide tile sits outside the cards' sweep corridor
        assert self.tile_y - self.tile_size / 2 > self.card_w / 2 + 0.02, "tile in card sweep"
        assert self.tile_y + self.tile_size / 2 < self.cabinet_size[1] / 2, "tile off cabinet"

    # -- shared geometry helpers (scene + smoke + solver read the same numbers) ----------------
    def hinge_point(self, i: int) -> tuple:
        return (self.axis_x, 0.0, self.axis_zs[i])

    def card_pose(self, i: int, phi_rad: float) -> tuple:
        """(pos, quat) of card i's BODY (cuboid centre) at hinge angle `phi` (0 = top,
        + toward the front/+x). Card local +z runs axle -> free edge; axle at local
        (0, 0, -(r_gap + L/2)); the joint frames are identity, so the joint angle IS
        phi. The slab centre rides at radius r_gap + L/2 from the axle along e(phi) —
        the r_gap air band between axle and slab is what lets a card cross the top."""
        hx, hy, hz = self.hinge_point(i)
        rc = self.r_gap + self.card_l / 2
        e = (math.sin(phi_rad), 0.0, math.cos(phi_rad))
        pos = (hx + e[0] * rc, hy, hz + e[2] * rc)
        h = phi_rad / 2
        return pos, (math.cos(h), 0.0, math.sin(h), 0.0)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("channel_flipboard")
class ChannelFlipboardScene(BaseScene):
    cfg: ChannelFlipboardSceneCfg

    def __init__(self, cfg: ChannelFlipboardSceneCfg | None = None) -> None:
        super().__init__(cfg or ChannelFlipboardSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic TV cabinet + axle cheeks, four hinged channel cards
        (one visual color each), four kinematic guide tiles (one per color; reset parks the
        target's tile in the guide frame and hides the rest inside the cabinet)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        grey = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.36, 0.38))
        light_grey = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.63, 0.66))
        # dead-soft landings: cards slam their stops and each other, never bounce back over
        mat = sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.5,
                                             restitution=0.0)
        # post_step drives the cards with external wrenches that do NOT wake a sleeping
        # body — sleep_threshold=0 keeps the plant live (oven_dials lesson).
        live = dict(sleep_threshold=0.0, stabilization_threshold=0.0,
                    solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                    max_depenetration_velocity=0.5, angular_damping=0.2, linear_damping=0.05)

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
            "cabinet": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet",
                spawn=sim_utils.CuboidCfg(
                    size=c.cabinet_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    physics_material=mat,
                    visual_material=grey,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.cabinet_pos),
            ),
        }
        cheek_h = 0.24
        for side, sy in (("l", -1.0), ("r", 1.0)):
            out[f"cheek_{side}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cheek_" + side,
                spawn=sim_utils.CuboidCfg(
                    size=(0.08, c.cheek_w, cheek_h),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    physics_material=mat,
                    visual_material=light_grey,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.axis_x, sy * c.cheek_dy, c.cabinet_top_z + cheek_h / 2)),
            )
        for i, (name, rgb) in enumerate(c.colors):
            pos, quat = c.card_pose(i, math.radians(c.rest_deg))
            out[f"card_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Card_" + str(i),
                spawn=sim_utils.CuboidCfg(
                    size=(c.card_th, c.card_w, c.card_l),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(**live),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.card_mass),
                    collision_props=coll,
                    physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=quat),
            )
            out[f"tile_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tile_" + str(i),
                spawn=sim_utils.CuboidCfg(
                    size=(c.tile_size,) * 3,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.tile_depot[0], c.tile_depot[1] + 0.05 * i, c.tile_depot[2])),
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
        n, p = env.num_envs, c.n_cards
        dev = env.device
        self.cabinet: RigidObject = env.iscene["cabinet"]
        self.cards: list[RigidObject] = [env.iscene[f"card_{i}"] for i in range(p)]
        self.tiles: list[RigidObject] = [env.iscene[f"tile_{i}"] for i in range(p)]
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        self._author_decorations()
        # Episode state.
        self.start_ch = torch.zeros(n, dtype=torch.long, device=dev)
        self.target_ch = torch.zeros(n, dtype=torch.long, device=dev)
        self.goal_sign = torch.zeros(n, p, device=dev)  # +1 front / -1 back / 0 non-mover
        self.flip_latch = torch.zeros(n, p, dtype=torch.bool, device=dev)
        self.k_req = torch.ones(n, dtype=torch.long, device=dev)
        # External drive inputs (solve.py's flip drive and smoke probes write; post_step
        # consumes + owns the cards' wrench slot — never call set_external_force_and_torque
        # on the cards directly).
        self.card_drive = torch.zeros(n, p, device=dev)  # torque about each hinge (+y, N*m)
        self.card_force = torch.zeros(n, p, 3, device=dev)  # world force at each card CoM

    def _author_joints(self) -> None:
        """Per env: one Y-axis revolute joint cabinet->card per card, anchored on the card's
        own axle line, limits at +-rest_deg (the cards REST on these stops). Joint frames
        are identity, so the joint angle equals the card angle phi (0 = apex)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        cx, cy, cz = c.cabinet_pos
        for e in range(self.env.num_envs):
            base = f"/World/envs/env_{e}"
            for i in range(c.n_cards):
                hx, hy, hz = c.hinge_point(i)
                j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/card_hinge_{i}")
                j.CreateBody0Rel().SetTargets([f"{base}/Cabinet"])
                j.CreateBody1Rel().SetTargets([f"{base}/Card_{i}"])
                j.CreateCollisionEnabledAttr(False)
                j.CreateAxisAttr("Y")
                j.CreateLocalPos0Attr(Gf.Vec3f(hx - cx, hy - cy, hz - cz))
                j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, -(c.r_gap + c.card_l / 2)))
                j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLowerLimitAttr(-c.rest_deg)
                j.CreateUpperLimitAttr(c.rest_deg)

    def _author_decorations(self) -> None:
        """Visual-only prims (displayColor, NO CollisionAPI), env_0 only (isaaclab composes
        env_1.. from env_0 by reference — the oven_dials idempotency precedent): the dark TV
        screen on the cabinet's FRONT (+x) face, the guide-frame border around the tile
        nest on the cabinet top, and one axle rod per card between the cheeks."""
        import omni.usd
        from pxr import Gf, UsdGeom

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath("/World/envs/env_0/Cabinet/screen").IsValid():
            return
        cx, cy, cz = c.cabinet_pos
        scr = UsdGeom.Cube.Define(stage, "/World/envs/env_0/Cabinet/screen")
        scr.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(scr.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(c.cabinet_size[0] / 2 + 0.001, 0.0, 0.0))
        xf.AddScaleOp().Set(Gf.Vec3f(0.004, 0.34, 0.13))
        scr.CreateDisplayColorAttr([Gf.Vec3f(0.03, 0.03, 0.05)])
        # guide frame: four dark strips around the tile nest
        tx, ty, tz = c.tile_pos
        s = c.tile_size / 2 + 0.006
        for k, (ox, oy, sx, sy) in enumerate((
                (s, 0.0, 0.004, 2 * s), (-s, 0.0, 0.004, 2 * s),
                (0.0, s, 2 * s, 0.004), (0.0, -s, 2 * s, 0.004))):
            bar = UsdGeom.Cube.Define(stage, f"/World/envs/env_0/Cabinet/guide_{k}")
            bar.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(bar.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(tx - cx + ox, ty - cy + oy,
                                             c.cabinet_top_z - cz + 0.003))
            xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, 0.006))
            bar.CreateDisplayColorAttr([Gf.Vec3f(0.05, 0.05, 0.06)])
        for i in range(c.n_cards):
            hx, hy, hz = c.hinge_point(i)
            rod = UsdGeom.Cylinder.Define(stage, f"/World/envs/env_0/Cabinet/axle_{i}")
            rod.CreateAxisAttr("Y")
            rod.CreateRadiusAttr(0.0035)
            rod.CreateHeightAttr(2 * c.cheek_dy)
            xf = UsdGeom.Xformable(rod.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(hx - cx, hy - cy, hz - cz))
            rod.CreateDisplayColorAttr([Gf.Vec3f(0.75, 0.76, 0.78)])

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample start channel s and a REACHABLE target t != s (any t > s,
        or t == 0 — the board is plan-level one-way), hang cards for channel s (cards < s
        on the back, rest on the front — jittered a few degrees INSIDE their stops so they
        settle onto them), park the target's guide tile in the frame and hide the other
        tiles inside the cabinet."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        p = c.n_cards
        origin = self.env_origins[env_ids]
        torch.rand(8, device=dev)  # burn draws: the FIRST post-seed draw is degenerate

        s = torch.randint(0, p, (m,), device=dev)
        # REACHABLE targets only (the board is plan-level one-way, see module docstring):
        # any t > s (peel the front slope back-ward), or t == 0 (clear the whole back
        # slope front-ward). 0 < t < s would require pulling a helper card out from
        # UNDER the mover propped on it — geometrically locked: from ~99 deg up to the
        # apex the lifter's root corner sits inside the rider's swept annulus band
        # (cos(phi) > -axis_dz/(2*r_gap)), so the rider is carried, at any force.
        has0 = (s > 0).long()
        n_all = (p - 1 - s) + has0  # allowed set = ({0} if s>0) + {s+1 .. p-1}
        r = (torch.rand(m, device=dev) * n_all.float()).long()
        r = torch.minimum(r, n_all - 1)
        t = torch.where((r == 0) & (s > 0), torch.zeros_like(s), s + r + 1 - has0)
        self.start_ch[env_ids] = s
        self.target_ch[env_ids] = t

        idx = torch.arange(p, device=dev).unsqueeze(0)  # (1, p)
        lo = torch.minimum(s, t).unsqueeze(1)
        hi = torch.maximum(s, t).unsqueeze(1)
        mover = (idx >= lo) & (idx < hi)  # cards between the channels move
        sign = torch.where(t.unsqueeze(1) > s.unsqueeze(1), -1.0, 1.0)  # goal side of movers
        self.goal_sign[env_ids] = torch.where(mover, sign, torch.zeros_like(sign))
        self.k_req[env_ids] = mover.sum(dim=1).clamp(min=1)
        self.flip_latch[env_ids] = False
        self.card_drive[env_ids] = 0.0
        self.card_force[env_ids] = 0.0

        jit = torch.rand(m, p, device=dev) * c.start_jitter_deg + 1.0  # 1..1+jitter deg inside
        side = torch.where(idx < s.unsqueeze(1), -1.0, 1.0)  # start partition for channel s
        phi = torch.deg2rad(side * (c.rest_deg - jit))
        for i in range(p):
            st = torch.zeros(m, 13, device=dev)
            for row in range(m):
                pos, quat = c.card_pose(i, float(phi[row, i]))
                st[row, 0:3] = origin[row] + torch.tensor(pos, device=dev)
                st[row, 3:7] = torch.tensor(quat, device=dev)
            self.cards[i].write_root_state_to_sim(st, env_ids)
            # tiles: target's color into the guide frame, the rest hidden in the cabinet
            st = torch.zeros(m, 13, device=dev)
            st[:, 3] = 1.0
            in_frame = t == i
            frame = torch.tensor(c.tile_pos, device=dev)
            depot = torch.tensor((c.tile_depot[0], c.tile_depot[1] + 0.05 * i,
                                  c.tile_depot[2]), device=dev)
            st[:, 0:3] = origin + torch.where(in_frame.unsqueeze(1), frame, depot)
            self.tiles[i].write_root_state_to_sim(st, env_ids)

    # ----- readings ----------------------------------------------------------------------------
    def angles_deg(self) -> torch.Tensor:
        """(N, P) card angle phi in deg: 0 = apex (straight up), + = front (+x side),
        - = back. The +-125 deg range cannot wrap."""
        out = []
        for b in self.cards:
            q = b.data.root_quat_w
            out.append(torch.rad2deg(2.0 * torch.atan2(q[:, 2], q[:, 0])))
        ang = torch.stack(out, dim=1)
        return (ang + 180.0) % 360.0 - 180.0

    def hinge_rate(self) -> torch.Tensor:
        """(N, P) signed hinge rate (rad/s) — the world-y angular velocity."""
        return torch.stack([b.data.root_ang_vel_w[:, 1] for b in self.cards], dim=1)

    def visible_channel(self) -> torch.Tensor:
        """(N,) index of the card showing on the FRONT of the board (the outermost card
        past +side_min_deg), or n_cards if the front is bare."""
        on_front = self.angles_deg() > self.cfg.side_min_deg  # (N, P)
        p = self.cfg.n_cards
        idx = torch.where(on_front, torch.arange(p, device=on_front.device).expand_as(on_front),
                          torch.full_like(on_front, p, dtype=torch.long))
        return idx.min(dim=1).values

    def partition_ok(self) -> torch.Tensor:
        """(N,) bool: every card rests categorically on its side for the TARGET channel —
        cards < t beyond side_min on the back, cards >= t beyond side_min on the front."""
        c = self.cfg
        ang = self.angles_deg()
        idx = torch.arange(c.n_cards, device=ang.device).unsqueeze(0)
        want_back = idx < self.target_ch.unsqueeze(1)
        ok = torch.where(want_back, ang < -c.side_min_deg, ang > c.side_min_deg)
        return ok.all(dim=1)

    def settled(self) -> torch.Tensor:
        """(N,) bool: every card slow in both angular and linear velocity."""
        c = self.cfg
        w = torch.stack([b.data.root_ang_vel_w.norm(dim=-1) for b in self.cards], dim=1)
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.cards], dim=1)
        return ((w < c.settle_omega) & (v < c.settle_speed)).all(dim=1)

    def success(self) -> torch.Tensor:
        """(N,) bool: the board shows the target channel — goal partition, settled."""
        return self.partition_ok() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: w_flip * latched fraction of required movers that crossed
        to their goal side + w_goal * success(). ~0 for the null policy (no mover has
        crossed), exactly 1.0 iff success(), latched credit survives a later knock-off."""
        c = self.cfg
        frac = self.flip_latch.float().sum(dim=1) / self.k_req.float()
        return c.w_flip * frac + c.w_goal * self.success().float()

    # ----- step-coupled mechanics (every substep) ------------------------------------------------
    def post_step(self) -> None:
        """Card plant: viscous hinge friction + external drive/force buffers about each
        card's world-y hinge axis (the kinematic cabinet never moves, so hinge axes stay
        world-fixed); then latch flip credit for required movers past latch_deg on their
        goal side. Owns the cards' external-wrench slot."""
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        ey = torch.tensor([0.0, 1.0, 0.0], device=dev)

        rate = self.hinge_rate()  # (N, P)
        tq = self.card_drive - c.hinge_visc * rate
        for i in range(c.n_cards):
            # reshape, not view: column slices of (N, P[, 3]) buffers are non-contiguous
            self.cards[i].set_external_force_and_torque(
                self.card_force[:, i].reshape(n, 1, 3),
                tq[:, i].reshape(n, 1, 1) * ey.view(1, 1, 3))

        # flip credit: mover observed past latch_deg on its goal side. NaN never latches
        # (comparisons with NaN are False), so a diverged frame earns no credit.
        ang = self.angles_deg()
        crossed = ang * self.goal_sign > c.latch_deg  # goal_sign 0 -> never latches
        self.flip_latch |= crossed

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {f"card_{i}": b for i, b in enumerate(self.cards)}
        bodies.update({f"tile_{i}": b for i, b in enumerate(self.tiles)})
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("start_ch", "target_ch", "goal_sign", "flip_latch", "k_req",
                               "card_drive", "card_force")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {f"card_{i}": b for i, b in enumerate(self.cards)}
        bodies.update({f"tile_{i}": b for i, b in enumerate(self.tiles)})
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        order = ", ".join(nm for nm, _rgb in c.colors)
        return (
            f"A TV cabinet (a grey box, top at {c.cabinet_top_z:.2f} m; the dark screen "
            f"marks its FRONT face) carries a mechanical flip-board between two upright "
            f"cheek plates: {c.n_cards} rigid colored channel cards "
            f"({c.card_w * 100:.0f} x {c.card_l * 100:.0f} cm), each hanging "
            f"{c.r_gap * 100:.0f} cm below its own horizontal axle on end pins (an open "
            f"air band separates axle and card). The axles are stacked a little above "
            f"one another, so the cards nest like a card tent over a ridge: on either "
            f"slope only the OUTERMOST card can be flipped by itself — lifting an inner "
            f"card presses into the cards outside it and just drags them along. A card "
            f"dropping onto a slope may rest propped on an outer card's face instead of "
            f"reaching its own stop — that still counts as being on the slope. From outermost to innermost the cards are {order} "
            f"(channels 1..{c.n_cards}). The channel now playing is the card you see "
            f"facing the FRONT of the board (the front slope's outermost card). On the "
            f"cabinet top, in the small dark guide frame, sits a colored tile: that is "
            f"this episode's TARGET channel color (sampled fresh every episode — read it "
            f"from the scene).\n"
            f"Goal: flip cards over the top of the board — a card is flipped by lifting "
            f"its free lower edge up over the apex; gravity drops it onto the other "
            f"slope — until the front of the board shows the guide tile's color: every "
            f"card ahead of the target card hangs on the BACK slope, the target card and "
            f"all cards behind it hang on the FRONT slope. Depending on the start you may "
            f"have to flip cards front-to-back or back-to-front; only the outermost card "
            f"of a slope can move alone, so work outer-first. Mind the one-way trap: a "
            f"card that lands on a slope rests ON TOP of the cards dropped there before "
            f"it, and a card with another propped on top of it can NEVER be flipped back "
            f"— the rider gets carried toward the apex and wedges the pair. An overshot "
            f"card is only recoverable while nothing has landed on it, so read the guide "
            f"tile FIRST and flip exactly the required cards (the requested target is "
            f"always reachable that way). The board only counts when every card rests "
            f"fully on a slope, nothing balanced on top."
        )

    def instruction(self) -> str:
        return (
            "Flip the hinged channel cards over the top of the flip-board, outermost card "
            "first, until the card facing the board's front is the same color as the tile "
            "in the guide frame on the cabinet top; leave every card resting fully on a "
            "slope. A board left showing any other color fails."
        )


# ----- runnable env: scene physics only (NullRobot) -> "simgen.channel_flipboard" --------------
register_env("simgen", lambda: EnvCfg(scene="channel_flipboard", robot="null"))

"""ChuteSwitchScene — route color-coded balls through a two-way gravity chute whose
sliding switch gate opens exactly one branch at a time
(sim_gen task `libero_kitchen_scene1_open_top_drawer_i14`).

Derived from libero_90/libero_kitchen_scene1_open_top_drawer, but STRATEGICALLY
different: the seed is one prehensile articulation act — grasp the drawer handle and
PULL the panel along its built prismatic joint until a joint readout crosses a
threshold. Here that exact motor primitive (slide a prismatic part between two stops)
still exists — the switch GATE — but it is judged NOWHERE and earns NOTHING by itself
(the smoke battery flips it back and forth and shows the score stays ~0). The judged
outcome is DELIVERIES: two GREEN balls must end up inside the GREEN bin and one RED
ball inside the RED bin, and the only physical path into either bin is down a roofed
twin-channel gravity chute whose single sliding gate always BLOCKS one channel and
opens the other. So the plan is a routing program, not a pull: read which bin sits at
which outlet (randomized), read the gate's current park (randomized), set the switch,
feed the balls that match the open route, then RECONFIGURE the switch mid-task and
feed the rest. A ball fed into the blocked channel jams against the gate; a ball
routed to the WRONG bin is unrecoverable (bins are roofed, mouths face the chute);
the BLUE decoy ball must be left out of both bins entirely.

Plan-level contrast with the seed: the seed's whole skill is one guided translation
of the judged part; here the translation is a mere sub-skill, the judged parts are
free bodies that gravity carries along a path SELECTED by mechanism state, the task
interleaves perception (bin sides, gate park, ball colors) with at least one
mandatory mid-task reconfiguration, and wrong routing is punished by irreversibility
instead of being retryable.

Assets are fully procedural (compound-spawner pattern; children of one body never
self-collide):
  - router (KINEMATIC compound): an inclined twin-channel chute on legs. Local frame:
    origin at the centre of the floor's TOP face, +x runs downhill, channels LEFT
    (+y) and RIGHT (-y) separated by a divider wall; side walls carry a through-slot
    at the gate station; the run downstream of the gate is ROOFED (the only free-space
    entries are the open-top inlet segment upstream of the gate and the two outlet
    faces, which mate to the bin mouths with sub-ball clearance). Uphill end wall,
    outrigger floor strip and two end-stop posts bound the gate's travel to exactly
    two parks: LEFT (blocks the left channel) and RIGHT (blocks the right channel).
  - gate (DYNAMIC, plain cuboid): a steel slider bar resting on the chute floor
    through the wall slots. Ball impacts push it squarely against the slot's
    downstream faces (no lateral component), so balls cannot re-route the switch;
    a push along the bar axis (its tail sticks out of the side wall) slides it
    between parks.
  - bin_green / bin_red (KINEMATIC compounds): roofed boxes with a single mouth that
    faces the chute outlet across a sub-ball gap; a sill ridge keeps a delivered ball
    from rolling back out. Once a ball is inside, no free-space path leads back out.
  - balls (DYNAMIC spheres): two GREEN, one RED (the cargo) and one BLUE decoy,
    scattered on a ground ring around the router.

Per-episode randomization (readback-verifiable): router yaw FREE (+/-180 deg) + xy
jitter; WHICH bin sits at WHICH outlet (bins_swapped); the gate's initial park side;
the balls' ring slots by random permutation with angle/radius jitter.

Rubric (0..1; latched credit anchored in the demonstrated solve trajectory):
  0.10 * routed — ever any cargo ball inside a channel DOWNSTREAM of the gate station
  0.20 * green1 — ever >= 1 green ball settled inside the green bin (latched)
  0.20 * green2 — ever both green balls settled inside the green bin (latched)
  0.20 * red    — ever the red ball settled inside the red bin (latched)
  capped at 0.70; exactly 1.0 iff success(): both greens in the green bin, red in the
  red bin, blue in NEITHER bin, bins still seated at their outlets, everything
  settled and finite. Null policy scores ~0; flipping the gate alone latches nothing;
  a wrong-bin delivery latches nothing and (being irreversible) forfeits success.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- custom compound spawners -----------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _root_xform(prim_path: str, translation, orientation):
    """Define an Xform root and author its (idempotent, single) translate/orient ops."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    return stage, xform.GetPrim()


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _add_box(stage, path: str, *, center, size, color, collide: Callable):
    """One box child: translate + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _spawn_router(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the router at `prim_path`: KINEMATIC compound. Local frame: origin at
    the centre of the floor slab's TOP face; +x downhill; channels at y = +/-ch_off.
    The body is mounted tilted by the caller (root orientation), so every child here
    is an axis-aligned box in the inclined frame."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    hl = c.length / 2
    wall_top = c.wall_h
    # floor slab + outrigger strip at the gate station (carries the end-stop posts)
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, -c.floor_t / 2),
             size=(c.length, c.total_w + 0.024, c.floor_t), color=c.body_color,
             collide=collide)
    _add_box(stage, f"{prim_path}/outrig", center=(c.gate_x, 0.0, -c.floor_t / 2),
             size=(0.10, 2 * c.post_y + 0.04, c.floor_t), color=c.body_color,
             collide=collide)
    # uphill end wall
    _add_box(stage, f"{prim_path}/endwall", center=(-hl + 0.006, 0.0, wall_top / 2),
             size=(0.012, c.total_w, wall_top), color=c.body_color, collide=collide)
    # outer walls + centre divider, each split by the gate slot
    slot0, slot1 = c.gate_x - c.slot_w / 2, c.gate_x + c.slot_w / 2
    for name, yc, wt in (("wl", c.outer_y, c.wall_t), ("wr", -c.outer_y, c.wall_t),
                         ("div", 0.0, c.div_t)):
        for tag, x0, x1 in (("u", -hl, slot0), ("d", slot1, hl)):
            _add_box(stage, f"{prim_path}/{name}_{tag}",
                     center=((x0 + x1) / 2, yc, wall_top / 2),
                     size=(x1 - x0, wt, wall_top), color=c.body_color, collide=collide)
    # roof: downstream of the gate only (the inlet run stays open-top)
    _add_box(stage, f"{prim_path}/roof",
             center=((c.roof_x0 + hl) / 2, 0.0, wall_top + c.roof_t / 2),
             size=(hl - c.roof_x0, c.total_w + 0.024, c.roof_t),
             color=c.roof_color, collide=collide)
    # gate guide rails: hang from roof height down to rail_bot, flanking the bar's
    # top 8 mm across the FULL travel so the bar cannot yaw and catch a slot edge;
    # balls (crest at 2r < rail_bot) pass underneath in the open channel
    for tag, xc in (("u", c.gate_x - c.slot_w / 2 - 0.005),
                    ("d", c.gate_x + c.slot_w / 2 + 0.005)):
        _add_box(stage, f"{prim_path}/rail_{tag}",
                 center=(xc, 0.0, (c.rail_bot + wall_top) / 2),
                 size=(0.010, c.total_w + 0.024, wall_top - c.rail_bot),
                 color=c.roof_color, collide=collide)
    # gate end-stop posts on the outrigger
    for sy in (1.0, -1.0):
        t = "p" if sy > 0 else "n"
        _add_box(stage, f"{prim_path}/post_{t}",
                 center=(c.gate_x, sy * c.post_y, 0.035),
                 size=(0.036, 0.012, 0.070), color=c.roof_color, collide=collide)
    # legs (visual grounding; kinematic, so exact ground contact is irrelevant)
    for sx, leg_l in ((-1.0, 0.24), (1.0, 0.14)):
        t = "u" if sx < 0 else "d"
        _add_box(stage, f"{prim_path}/leg_{t}",
                 center=(sx * (hl - 0.06), 0.0, -c.floor_t - leg_l / 2),
                 size=(0.030, c.total_w * 0.8, leg_l), color=c.body_color,
                 collide=collide)
    return root


def _spawn_bin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a roofed bin at `prim_path`: KINEMATIC compound. Local frame: origin at
    the centre of the bin footprint ON THE GROUND; the mouth faces local -x. The only
    opening is the mouth (sill at the bottom, lintel + roof above)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    hx, hy = c.bin_d / 2, c.bin_w / 2   # deep along the channel (x), narrow across (y)
    color = c.color
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, c.bfloor_t / 2),
             size=(c.bin_d, c.bin_w, c.bfloor_t), color=color, collide=collide)
    wz = c.bwall_h / 2
    _add_box(stage, f"{prim_path}/back", center=(hx - c.bwall_t / 2, 0.0, wz),
             size=(c.bwall_t, c.bin_w, c.bwall_h), color=color, collide=collide)
    for sy in (1.0, -1.0):
        t = "p" if sy > 0 else "n"
        _add_box(stage, f"{prim_path}/side_{t}",
                 center=(0.0, sy * (hy - c.bwall_t / 2), wz),
                 size=(c.bin_d - 2 * c.bwall_t, c.bwall_t, c.bwall_h),
                 color=color, collide=collide)
    # front face: sill strip, two pillars flanking the mouth, lintel above it
    fx = -hx + c.bwall_t / 2
    _add_box(stage, f"{prim_path}/sill", center=(fx, 0.0, c.sill_top / 2),
             size=(c.bwall_t, c.bin_w, c.sill_top), color=color, collide=collide)
    pw = (c.bin_w - c.mouth_w) / 2
    mz = c.sill_top + c.mouth_h
    for sy in (1.0, -1.0):
        t = "p" if sy > 0 else "n"
        _add_box(stage, f"{prim_path}/pillar_{t}",
                 center=(fx, sy * (c.bin_w - pw) / 2, (c.sill_top + mz) / 2),
                 size=(c.bwall_t, pw, c.mouth_h + 0.001), color=color, collide=collide)
    _add_box(stage, f"{prim_path}/lintel",
             center=(fx, 0.0, (mz + c.bwall_h) / 2),
             size=(c.bwall_t, c.bin_w, c.bwall_h - mz), color=color, collide=collide)
    _add_box(stage, f"{prim_path}/roof", center=(0.0, 0.0, c.bwall_h + c.roof_t / 2),
             size=(c.bin_d, c.bin_w, c.roof_t), color=color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "router" not in _SPAWNER_CACHE:

        @configclass
        class RouterSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_router)
            length: float = 0.60
            ch_w: float = 0.090
            ch_off: float = 0.051
            outer_y: float = 0.102
            total_w: float = 0.216
            wall_t: float = 0.012
            div_t: float = 0.012
            wall_h: float = 0.075
            floor_t: float = 0.020
            roof_t: float = 0.010
            roof_x0: float = -0.030
            gate_x: float = -0.010
            slot_w: float = 0.036
            rail_bot: float = 0.050
            post_y: float = 0.152
            body_color: tuple = (0.30, 0.32, 0.36)
            roof_color: tuple = (0.42, 0.44, 0.50)
            contact_offset: float = 0.0015

        @configclass
        class BinSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bin)
            bin_d: float = 0.130
            bin_w: float = 0.098
            bwall_t: float = 0.010
            bwall_h: float = 0.125
            bfloor_t: float = 0.012
            sill_top: float = 0.030
            mouth_w: float = 0.070
            mouth_h: float = 0.080
            roof_t: float = 0.010
            color: tuple = (0.5, 0.5, 0.5)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["router"] = RouterSpawnerCfg
        _SPAWNER_CACHE["bin"] = BinSpawnerCfg
    return _SPAWNER_CACHE

# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ChuteSwitchSceneCfg(BaseCfg):
    """Config for `ChuteSwitchScene`. Every opening a ball is NOT meant to pass is
    sized below the ball diameter (asserted below), so the roofed run downstream of
    the gate and the roofed bins are physically closed to any path except the
    intended one: inlet -> open channel -> outlet -> bin mouth."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    settle_lin: float = tunable(0.05)      # max |lin vel| of balls + gate when judging (m/s)
    ball_slow: float = tunable(0.25)       # per-ball speed gate for the delivery latches (m/s)
    park_tol: float = tunable(0.012)       # gate counts as parked within this of a park (m)
    bin_seat_tol: float = tunable(0.008)   # bin may not have moved more than this (m)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    yaw_deg: float = tunable(180.0)        # router yaw uniform +/- (FREE heading)
    xy_jitter: float = tunable(0.05)       # router xy jitter (+/- m)
    swap_bins: bool = tunable(True)        # randomize which bin serves which outlet
    random_gate: bool = tunable(True)      # randomize the gate's initial park side
    slot_jitter_deg: float = tunable(12.0)  # per-ball ring-slot angle jitter (+/- deg)
    ring_r_min: float = tunable(0.50)      # ball scatter ring radii (m)
    ring_r_max: float = tunable(0.62)

    # --- info: layout ----------------------------------------------------------------------------
    router_pos: tuple = info((0.0, 0.0))   # router origin on the ground plane (nominal)
    # Ring slots (router frame; 0 deg = downhill). All four sit in the UPHILL half so
    # a single fixed manipulator base uphill of the chute reaches every ball, the
    # inlet and the gate tail (embodiment argument in TASK.md).
    slot_angles_deg: tuple = info((115.0, 165.0, 215.0, 265.0))
    tilt_deg: float = info(12.0)           # chute incline
    exit_h: float = info(0.060)            # chute floor-top height at the outlet edge

    # --- info: router structure (local frame: origin at floor-top centre, +x downhill) -----------
    length: float = info(0.60)
    ch_w: float = info(0.090)              # each channel's clear width
    ch_off: float = info(0.051)            # channel centreline at y = +/- this
    outer_y: float = info(0.102)           # outer wall centreline
    total_w: float = info(0.216)
    wall_t: float = info(0.012)
    div_t: float = info(0.012)
    wall_h: float = info(0.075)
    floor_t: float = info(0.020)
    roof_t: float = info(0.010)
    roof_x0: float = info(-0.030)          # roof starts UPSTREAM of the gate slot
    past_x: float = info(0.010)            # "routed past the switch" x threshold
    gate_x: float = info(-0.010)           # gate station (slot centre)
    slot_w: float = info(0.036)            # wall through-slot width (< ball diameter)
    rail_bot: float = info(0.050)          # guide-rail underside (balls pass under)
    post_y: float = info(0.152)            # end-stop post centreline
    # --- info: gate (plain dynamic cuboid) -------------------------------------------------------
    gate_size: tuple = info((0.028, 0.144, 0.058))
    gate_mass: float = info(0.50)
    gate_park: float = info(0.072)         # park centres at y = +/- this (router frame)
    # --- info: bins (deep along the channel, narrow across so neighbours clear) ------------------
    bin_d: float = info(0.130)             # footprint depth along the channel heading
    bin_w: float = info(0.098)             # footprint width across (2*ch_off - 4 mm)
    bwall_t: float = info(0.010)
    bwall_h: float = info(0.125)
    bfloor_t: float = info(0.012)
    sill_top: float = info(0.030)
    mouth_w: float = info(0.070)
    mouth_h: float = info(0.080)
    bin_gap: float = info(0.012)           # chute outlet edge -> bin front face (< ball)
    green: tuple = info((0.10, 0.65, 0.15))
    red: tuple = info((0.75, 0.10, 0.10))
    # --- info: balls -----------------------------------------------------------------------------
    ball_r: float = info(0.020)
    ball_mass: float = info(0.060)
    blue: tuple = info((0.15, 0.25, 0.85))
    contact_offset: float = info(0.0015)
    # rubric weights (0.10 + 3 x 0.20 = 0.70 = the non-success cap)
    w_route: float = info(0.10)
    w_del: float = info(0.20)

    # Derived (filled in __post_init__).
    root_z: float = field(default=0.0, init=False)     # router root height for exit_h
    bin_run: float = field(default=0.0, init=False)    # bin centre offset along heading

    def __post_init__(self) -> None:
        th = math.radians(self.tilt_deg)
        self.root_z = self.exit_h + math.sin(th) * self.length / 2
        self.bin_run = (self.length / 2) * math.cos(th) + self.bin_gap + self.bin_d / 2

        d = 2 * self.ball_r
        # Geometry consistency of the switch itself.
        assert abs((self.div_t / 2 + self.ch_w / 2) - self.ch_off) < 1e-9
        assert abs((self.div_t / 2 + self.ch_w + self.wall_t / 2) - self.outer_y) < 1e-9
        assert abs((2 * self.outer_y + self.wall_t) - self.total_w) < 1e-9
        gy = self.gate_size[1]
        lo, hi = self.gate_park - gy / 2, self.gate_park + gy / 2
        assert lo <= self.div_t / 2 + 1e-9 and hi >= self.outer_y + self.wall_t / 2, \
            "parked gate must cover its channel wall-to-wall"
        assert lo >= -self.div_t / 2 - 1e-9, \
            "parked gate must not intrude into the open channel"
        assert abs((self.post_y - 0.006) - hi) <= 0.004, \
            "end-stop posts must arrest the gate exactly at its parks"
        # Every unintended opening is smaller than the ball.
        assert self.slot_w < d, "wall slot must not pass a ball"
        assert self.wall_h - self.gate_size[2] < d, "gap over the parked gate must not pass a ball"
        assert self.bin_gap < d, "chute->bin gap must not pass a ball"
        # Every intended opening is comfortably larger than the ball.
        assert self.ch_w > d + 0.03, "channel must pass a ball with slack"
        assert self.mouth_w > d + 0.02 and self.mouth_h > d + 0.01, "bin mouth too small"
        assert self.mouth_w <= self.bin_w - 2 * self.bwall_t, "mouth wider than bin front"
        # The two bins sit at y = +/- ch_off: they must not touch, let alone overlap.
        assert self.bin_w <= 2 * self.ch_off - 0.004 + 1e-9, \
            "adjacent bins must clear each other at the outlets"
        # Both greens must fit in one bin interior with slack (along the deep axis).
        assert self.bin_d - 2 * self.bwall_t > 2 * d + 0.02, \
            "bin interior must hold two balls along its depth"
        assert self.bin_w - 2 * self.bwall_t > d + 0.03, \
            "bin interior must pass a ball across its width"
        # The whole ball (bottom to top) must clear the mouth opening on arrival.
        assert self.exit_h - self.ball_r > self.sill_top + 0.005, \
            "arriving ball must clear the sill"
        assert self.exit_h + self.ball_r < self.sill_top + self.mouth_h - 0.005, \
            "arriving ball must clear the lintel"
        # The roof must cover the gate so nothing can be dropped in past the switch.
        assert self.roof_x0 < self.gate_x - self.slot_w / 2 - 1e-9, \
            "roof must start upstream of the gate slot"
        assert self.past_x > self.gate_x + self.slot_w / 2, \
            "past-the-switch threshold must sit downstream of the slot"
        # Guide rails: above a rolling ball, but overlapping the bar's top.
        assert self.rail_bot > d + 0.008, "balls must pass under the guide rails"
        assert self.rail_bot < self.gate_size[2] - 0.005, \
            "guide rails must overlap the bar to prevent yaw"
        assert self.sill_top - self.bfloor_t >= 0.8 * self.ball_r, \
            "sill must be tall enough to keep a settled ball in"


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("chute_switch")
class ChuteSwitchScene(BaseScene):
    cfg: ChuteSwitchSceneCfg

    BALLS = ("green_0", "green_1", "red_0", "blue_0")
    CARGO = ("green_0", "green_1", "red_0")

    def __init__(self, cfg: ChuteSwitchSceneCfg | None = None) -> None:
        super().__init__(cfg or ChuteSwitchSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        router_spawn = cls["router"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            length=c.length, ch_w=c.ch_w, ch_off=c.ch_off, outer_y=c.outer_y,
            total_w=c.total_w, wall_t=c.wall_t, div_t=c.div_t, wall_h=c.wall_h,
            floor_t=c.floor_t, roof_t=c.roof_t, roof_x0=c.roof_x0, gate_x=c.gate_x,
            slot_w=c.slot_w, rail_bot=c.rail_bot, post_y=c.post_y,
            contact_offset=c.contact_offset)

        def bin_spawn(color):
            return cls["bin"](
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                bin_d=c.bin_d, bin_w=c.bin_w, bwall_t=c.bwall_t, bwall_h=c.bwall_h,
                bfloor_t=c.bfloor_t, sill_top=c.sill_top, mouth_w=c.mouth_w,
                mouth_h=c.mouth_h, roof_t=c.roof_t, color=color,
                contact_offset=c.contact_offset)

        rigid = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.05, angular_damping=0.05,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=32, solver_velocity_iteration_count=1)
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "router": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Router", spawn=router_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.router_pos[0], c.router_pos[1], c.root_z))),
            "bin_green": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BinGreen", spawn=bin_spawn(c.green),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 1.5, 0.0))),
            "bin_red": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BinRed", spawn=bin_spawn(c.red),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, -1.5, 0.0))),
            "gate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Gate",
                spawn=sim_utils.CuboidCfg(
                    size=c.gate_size,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.gate_mass),
                    rigid_props=rigid, collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.30, dynamic_friction=0.25, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.80, 0.78, 0.20)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.5, 0.0, 0.05))),
        }
        for name, color in (("green_0", c.green), ("green_1", c.green),
                            ("red_0", c.red), ("blue_0", c.blue)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball_" + name,
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    rigid_props=rigid, collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.60, dynamic_friction=0.50, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2, 1.2, c.ball_r)))
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

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.router: RigidObject = env.iscene["router"]
        self.bins: dict[str, RigidObject] = {
            "green": env.iscene["bin_green"], "red": env.iscene["bin_red"]}
        self.gate: RigidObject = env.iscene["gate"]
        self.balls: dict[str, RigidObject] = {n: env.iscene[n] for n in self.BALLS}
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # readbacks (verified by smoke)
        self.bins_swapped = torch.zeros(n, dtype=torch.bool, device=dev)
        self.gate_init_left = torch.zeros(n, dtype=torch.bool, device=dev)
        self.slot_perm = torch.zeros(n, len(self.BALLS), dtype=torch.long, device=dev)
        # expected bin poses (success requires the bins to still sit at their outlets)
        self._bin_tgt = {k: torch.zeros(n, 3, device=dev) for k in self.bins}
        # latches (partial credit survives transients; success is judged live)
        self._route = torch.zeros(n, dtype=torch.bool, device=dev)
        self._g1 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._g2 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._r1 = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the inclined router (free yaw + xy jitter), park the
        gate on a random side, seat the two bins at the outlets in a random
        left/right arrangement, deal the four balls to ring slots by a random
        permutation, clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_deg)
        q_yaw = _qz(yaw)
        tilt = torch.full((m,), math.radians(c.tilt_deg), device=dev)
        q_router = _qmul(q_yaw, _qy(tilt))
        rp = torch.zeros(m, 3, device=dev)
        rp[:, 0] = c.router_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        rp[:, 1] = c.router_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        rp[:, 2] = c.root_z
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = rp + origin
        st[:, 3:7] = q_router
        self.router.write_root_state_to_sim(st, env_ids)

        from isaaclab.utils.math import quat_apply

        # gate: parked on a random side, resting on the inclined floor
        left = (torch.rand(m, device=dev) < 0.5) if c.random_gate \
            else torch.ones(m, dtype=torch.bool, device=dev)
        self.gate_init_left[env_ids] = left
        gy = torch.where(left, torch.full((m,), c.gate_park, device=dev),
                         torch.full((m,), -c.gate_park, device=dev))
        loc = torch.stack([torch.full((m,), c.gate_x, device=dev), gy,
                           torch.full((m,), c.gate_size[2] / 2, device=dev)], dim=-1)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = rp + origin + quat_apply(q_router, loc)
        st[:, 3:7] = q_router
        self.gate.write_root_state_to_sim(st, env_ids)

        # bins: seat at the two outlets; which color serves which outlet is random
        swap = (torch.rand(m, device=dev) < 0.5) if c.swap_bins \
            else torch.zeros(m, dtype=torch.bool, device=dev)
        self.bins_swapped[env_ids] = swap
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        for name in self.bins:
            is_green = name == "green"
            # green serves the LEFT (+y) outlet unless swapped
            side = torch.where(swap,
                               torch.full((m,), -1.0 if is_green else 1.0, device=dev),
                               torch.full((m,), 1.0 if is_green else -1.0, device=dev))
            lx = torch.full((m,), c.bin_run, device=dev)
            ly = side * c.ch_off
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = rp[:, 0] + lx * cy - ly * sy
            st[:, 1] = rp[:, 1] + lx * sy + ly * cy
            st[:, 2] = 0.0
            st[:, 3:7] = q_yaw
            st[:, 0:3] += origin
            self.bins[name].write_root_state_to_sim(st, env_ids)
            self._bin_tgt[name][env_ids] = st[:, 0:3]

        # balls: ring slots, random permutation, jitter
        nb = len(self.BALLS)
        perm = torch.rand(m, nb, device=dev).argsort(dim=1)
        self.slot_perm[env_ids] = perm
        base = torch.tensor(c.slot_angles_deg, device=dev)
        for i, name in enumerate(self.BALLS):
            ang = base[perm[:, i]] + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter_deg
            ang = yaw + torch.deg2rad(ang)
            r = c.ring_r_min + torch.rand(m, device=dev) * (c.ring_r_max - c.ring_r_min)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = rp[:, 0] + r * torch.cos(ang)
            st[:, 1] = rp[:, 1] + r * torch.sin(ang)
            st[:, 2] = c.ball_r + 0.002
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            self.balls[name].write_root_state_to_sim(st, env_ids)

        self._route[env_ids] = False
        self._g1[env_ids] = False
        self._g2[env_ids] = False
        self._r1[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "router": self.router.data.root_state_w[env_ids].clone(),
            "gate": self.gate.data.root_state_w[env_ids].clone(),
            "bins": {k: b.data.root_state_w[env_ids].clone()
                     for k, b in self.bins.items()},
            "balls": {k: b.data.root_state_w[env_ids].clone()
                      for k, b in self.balls.items()},
            "bin_tgt": {k: v[env_ids].clone() for k, v in self._bin_tgt.items()},
            "bins_swapped": self.bins_swapped[env_ids].clone(),
            "gate_init_left": self.gate_init_left[env_ids].clone(),
            "slot_perm": self.slot_perm[env_ids].clone(),
            "route": self._route[env_ids].clone(), "g1": self._g1[env_ids].clone(),
            "g2": self._g2[env_ids].clone(), "r1": self._r1[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.router.write_root_state_to_sim(state["router"], env_ids)
        self.gate.write_root_state_to_sim(state["gate"], env_ids)
        for k, b in self.bins.items():
            b.write_root_state_to_sim(state["bins"][k], env_ids)
        for k, b in self.balls.items():
            b.write_root_state_to_sim(state["balls"][k], env_ids)
        for k in self._bin_tgt:
            self._bin_tgt[k][env_ids] = state["bin_tgt"][k]
        self.bins_swapped[env_ids] = state["bins_swapped"]
        self.gate_init_left[env_ids] = state["gate_init_left"]
        self.slot_perm[env_ids] = state["slot_perm"]
        self._route[env_ids] = state["route"]
        self._g1[env_ids] = state["g1"]
        self._g2[env_ids] = state["g2"]
        self._r1[env_ids] = state["r1"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"An inclined BALL CHUTE stands on legs: two parallel roofed channels run "
            f"side by side down the slope and end above two covered BINS — one GREEN, "
            f"one RED (which bin sits under which channel varies per episode). The "
            f"only open stretch is the inlet at the TOP of the chute; everything "
            f"downhill of the yellow SWITCH GATE is covered, and each bin's only "
            f"opening is the mouth facing its channel outlet. The gate is a sliding "
            f"bar crossing both channels through slots in the walls: it always BLOCKS "
            f"exactly one channel and leaves the other open, and it can be pushed "
            f"along its axis (its tail sticks out of the side wall) between the two "
            f"stops. A ball released into the OPEN channel's inlet rolls down and "
            f"falls into that channel's bin; a ball released into the BLOCKED channel "
            f"jams against the gate. Once a ball is inside a bin it cannot be taken "
            f"out (the bins are covered and a sill blocks the mouth). On the ground "
            f"around the chute lie four loose balls "
            f"({2 * c.ball_r * 1000:.0f} mm): two GREEN, one RED, one BLUE. The "
            f"chute's position and heading, the bins' arrangement, the gate's initial "
            f"side and the balls' positions vary every episode; read the layout, do "
            f"not memorize it.\n"
            f"Goal: deliver BOTH green balls into the GREEN bin and the RED ball into "
            f"the RED bin, using the chute (there is no other way in). The BLUE ball "
            f"belongs in NEITHER bin — leave it outside. Because both colors' "
            f"channels share the one switch gate, you must set the gate for one "
            f"color's channel, feed those balls, then slide the gate to the other "
            f"side and feed the rest. A ball sent down the wrong channel ends up in "
            f"the wrong bin and cannot be recovered. Final state: two green balls "
            f"inside the green bin, the red ball inside the red bin, the blue ball "
            f"outside both, bins undisturbed, everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Use the inclined chute to deliver the two green balls into the green bin "
            "and the red ball into the red bin. Slide the yellow switch gate to open "
            "the channel you need before each delivery. Leave the blue ball out of "
            "both bins."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _router_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.router.data.root_quat_w,
                                  pos_w - self.router.data.root_pos_w)

    def gate_y(self) -> torch.Tensor:
        """(N,) gate centre's y in the ROUTER frame: + = toward the LEFT channel."""
        return self._router_local(self.gate.data.root_pos_w)[:, 1]

    def gate_side(self) -> torch.Tensor:
        """(N,) int: +1 gate parked LEFT (left channel BLOCKED), -1 parked RIGHT,
        0 not parked (mid-travel, tipped, or out of the pocket)."""
        c = self.cfg
        loc = self._router_local(self.gate.data.root_pos_w)
        in_pocket = (loc[:, 0] - c.gate_x).abs() < 0.02
        in_pocket &= (loc[:, 2] - c.gate_size[2] / 2).abs() < 0.02
        left = in_pocket & ((loc[:, 1] - c.gate_park).abs() < c.park_tol)
        right = in_pocket & ((loc[:, 1] + c.gate_park).abs() < c.park_tol)
        return left.long() - right.long()

    def _ball_pos(self) -> torch.Tensor:
        return torch.stack([b.data.root_pos_w for b in self.balls.values()], dim=1)

    def balls_in_channel(self, downstream_only: bool = True) -> torch.Tensor:
        """(N, 4) bool: ball centre inside either channel's clear volume, in the
        ROUTER body frame; by default only DOWNSTREAM of the gate station (i.e. the
        ball was actually routed past the switch)."""
        c = self.cfg
        pos = self._ball_pos()
        n, k = pos.shape[0], pos.shape[1]
        loc = self._router_local(pos.reshape(n * k, 3)).reshape(n, k, 3)
        in_y = ((loc[:, :, 1] - c.ch_off).abs() < c.ch_w / 2) \
            | ((loc[:, :, 1] + c.ch_off).abs() < c.ch_w / 2)
        x_lo = c.past_x if downstream_only else -c.length / 2
        in_x = (loc[:, :, 0] > x_lo) & (loc[:, :, 0] < c.length / 2)
        in_z = (loc[:, :, 2] > 0.0) & (loc[:, :, 2] < c.wall_h)
        return in_x & in_y & in_z

    def balls_in_bin(self, name: str) -> torch.Tensor:
        """(N, 4) bool: ball centre inside bin `name`'s interior (bin body frame)."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        b = self.bins[name]
        pos = self._ball_pos()
        n, k = pos.shape[0], pos.shape[1]
        loc = quat_apply_inverse(
            b.data.root_quat_w.unsqueeze(1).expand(n, k, 4).reshape(n * k, 4),
            (pos - b.data.root_pos_w.unsqueeze(1)).reshape(n * k, 3)).reshape(n, k, 3)
        half_x = c.bin_d / 2 - c.bwall_t - 0.002
        half_y = c.bin_w / 2 - c.bwall_t - 0.002
        return (loc[:, :, 0].abs() < half_x) & (loc[:, :, 1].abs() < half_y) \
            & (loc[:, :, 2] > c.bfloor_t - 0.002) & (loc[:, :, 2] < c.bwall_h)

    def bins_seated(self) -> torch.Tensor:
        """(N,) bool: both bins still sit where reset mated them to the outlets."""
        tol = self.cfg.bin_seat_tol
        ok = None
        for k, b in self.bins.items():
            d = (b.data.root_pos_w - self._bin_tgt[k]).norm(dim=-1) < tol
            ok = d if ok is None else ok & d
        return ok

    def settled(self) -> torch.Tensor:
        """(N,) bool: every ball and the gate below the velocity gate."""
        c = self.cfg
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                         for b in self.balls.values()]
                        + [self.gate.data.root_lin_vel_w.norm(dim=-1)], dim=1)
        return (v < c.settle_lin).all(dim=1)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([self.gate.data.root_pos_w]
                        + [b.data.root_pos_w for b in self.balls.values()], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        in_ch = self.balls_in_channel()          # (N,4), downstream of the gate
        self._route |= in_ch[:, 0:3].any(dim=1) & fin   # cargo balls only
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                         for b in self.balls.values()], dim=1)
        slow = v < c.ball_slow
        in_g = self.balls_in_bin("green") & slow
        in_r = self.balls_in_bin("red") & slow
        ng = in_g[:, 0:2].sum(dim=1)             # green balls in the green bin
        self._g1 |= (ng >= 1) & fin
        self._g2 |= (ng >= 2) & fin
        self._r1 |= in_r[:, 2] & fin             # red ball in the red bin
        # (blue never latches anything; wrong-bin deliveries never latch anything)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: both greens inside the green bin, red inside the red bin, blue
        in NEITHER bin, bins still seated at their outlets, everything settled and
        finite. All clauses are live physical outcomes: the roofed chute and bins
        leave routing through the switch as the only path in."""
        self._update_latches()
        in_g = self.balls_in_bin("green")
        in_r = self.balls_in_bin("red")
        greens_ok = in_g[:, 0] & in_g[:, 1]
        red_ok = in_r[:, 2]
        blue_out = ~in_g[:, 3] & ~in_r[:, 3]
        return greens_ok & red_ok & blue_out & self.bins_seated() \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.10 once any cargo ball is routed past the gate +
        0.20 per correct delivery (latched), capped at 0.70; exactly 1.0 iff
        success() holds live. Doing nothing scores ~0; sliding the gate back and
        forth (the seed's prismatic-pull skill) latches nothing; wrong-bin
        deliveries latch nothing."""
        c = self.cfg
        self._update_latches()
        base = (c.w_route * self._route.float()
                + c.w_del * (self._g1.float() + self._g2.float() + self._r1.float())
                ).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="chute_switch", robot="null"))

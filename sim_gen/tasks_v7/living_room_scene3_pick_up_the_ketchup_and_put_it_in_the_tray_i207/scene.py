"""QueueDispenserScene — dispense the ketchup carton from a sealed gravity-feed column
into the catch tray staged over the reject pit.

Derived from libero_90/living_room_scene3_pick_up_the_ketchup_and_put_it_in_the_tray
("pick up the ketchup and put it in the tray": grasp one bottle among distractors on a
table, carry it through free air, lower it into an open static tray — one pick-and-place
judged the instant the bottle enters the tray's bounding box). Here the ketchup can
NEVER be picked up at all: it sits somewhere in a vertical QUEUE of three color-coded
cartons inside a sealed DISPENSER COLUMN (roofed, walls closed; a 16 mm sight slot shows
the queue colors, far too narrow for any carton or grasp). Cartons leave the column only
through its bottom OUTLET: the queue rests on an 11-degree FEED BED behind a 16 mm
RETENTION LIP, and a firm push through the 44 mm REAR SLOT tips the bottom carton over
the lip, after which gravity slides it down the slick launch tongue and drops it off the
tip — one carton per push: the next carton drops shielded by the front wall (outlet
edge at z 0.420 covers it for ~80 of its 88 mm fall) and its residual slide energy is
far below the lip's pivot barrier, so the queue re-seats itself. The tongue tip overhangs a deep REJECT
PIT: whatever is dispensed with nothing staged below is lost in the pit. The catch TRAY
starts loose on the ground; seated on the pit rim it intercepts the drop instead.

So the seed's single pick-and-place becomes queue sequencing with indirect acquisition:
read the queue through the sight slot, eject every carton BELOW the ketchup into the
reject pit (forced by the queue order — the ketchup physically cannot come out first),
stage the tray on the pit rim, then eject the ketchup so it lands in the tray. Ejecting
the ketchup before staging the tray drops it into the pit — unrecoverable. Success is
the settled end state: the ketchup carton at rest INSIDE the tray, the tray SEATED on
the pit rim, and NO other carton in the tray.

Structure (all procedural; the STATION is 18 standard-schema KINEMATIC cuboids posed
coherently from one station pose at every reset — yaw + xy randomization with zero
custom spawner surface; only the tray is a custom dynamic compound):
  station-local frame: origin at the column's rear wall outer face on the ground,
  dispense direction = +x, z up. Feed ramp surface z(x) = 0.315 - tan(11 deg) * x.
  - column: side walls (inner faces y +/-0.028), rear wall with the push slot
    (44 x ~60 mm, sill = the ramp surface), front wall strips above the outlet
    (lower edge z 0.410) leaving a full-height 16 mm sight slot, roof at z 0.600,
    pedestal below the ramp;
  - feed ramp, pitched 11 deg, carrying an 8 mm FEED BED (grippier, so the dropped
    follower lands soft) behind the retention LIP (16 mm proud of the bed) at the
    shaft outlet, and a slick launch TONGUE (guarded by low side rails) whose tip
    (x 0.117, z 0.292) overhangs the pit mouth;
  - reject pit: inner 150 x 110 mm, rim top z 0.200, floor z 0.012 — a dispensed
    carton comes to rest ~150 mm below the rim, out of play;
  - tray: dynamic compound (custom spawner: MassAPI 0.25 kg + material authored in
    the func), outer 180 x 150 mm, walls 40 mm — seated on the pit rim it covers the
    mouth and its interior sits under the tongue tip; it starts loose on the ground.
  - cartons: three 50 x 50 x 80 mm boxes, 0.15 kg: KETCHUP (red), BUTTER (yellow),
    CREAM CHEESE (white), stacked in a per-episode-shuffled queue.

Geometry facts the task rests on (verified by the smoke battery):
  - captive queue: roof + walls sealed, sight slot 16 mm and rear slot 44 mm are both
    < the 50 mm carton — the only way out is the outlet, bottom carton first;
  - retention lip: a light nudge (~1.5 N on the stacked queue) cannot tip a carton
    over the 16 mm lip; dispensing takes a deliberate firm push (the null policy and
    incidental contact leave the queue seated);
  - one-per-push metering: the follower drops shielded by the front wall (outlet
    lower edge z 0.420) and lands on the grippy bed with too little energy to vault
    the lip — it re-seats instead of following the ejected carton out;
  - drop point: the tongue tip overhangs the pit mouth interior; with the tray seated
    the tip lies over the tray's interior with >= 35 mm wall clearance a side.

Per-episode randomization (readback-verifiable): station yaw +/-20 deg + xy jitter,
queue permutation (which carton sits at which level — the number of forced rejects
varies 0..2), tray start pose on the open ground (position + yaw).

Rubric (0..1; latched partial credit anchored in the demonstrated solve):
  0.15 * (rejected/needed) — cartons queued BELOW the ketchup ejected from the column
                             (latched each; 0 when the ketchup starts at the bottom)
  0.15 * staged            — the tray ever at rest seated on the pit rim (latched)
  0.35 * served            — the ketchup carton ever at rest inside the tray (latched)
  1.0 iff success()        — ketchup in the tray AND tray seated on the rim AND no
                             other carton in the tray, all settled + finite.
  Non-success is capped at 0.65; the null policy scores ~0.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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


def _qinv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[:, 1:] = -out[:, 1:]
    return out


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = torch.cat([torch.zeros_like(q[:, :1]), v], dim=-1)
    return _qmul(_qmul(q, qv), _qinv(q))[:, 1:]


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qy_const(deg: float, device) -> torch.Tensor:
    """(1,4) constant pitch quaternion about +y (positive pitches +x downward)."""
    h = math.radians(deg) / 2
    return torch.tensor([[math.cos(h), 0.0, math.sin(h), 0.0]], device=device)


def encode_force(mode: int, q_ref: torch.Tensor, q_now: torch.Tensor,
                 f_world: torch.Tensor) -> torch.Tensor:
    """Pre-encode a desired WORLD-frame force for `set_external_force_and_torque`.

    Some pods rotate an applied wrench by the body's rotation since its reference
    orientation (applied = R_now * R_ref^T * arg). mode 0 passes the world force
    through unchanged; mode 1 pre-encodes with R_ref * R_now^T so the applied force
    comes out as the desired world force. Callers PROBE which mode moves the body the
    right way and lock it in (`q_ref` = readback at the reference instant)."""
    if mode == 0:
        return f_world
    return _qapply(_qmul(q_ref, _qinv(q_now)), f_world)


# ----- custom compound spawner: the catch tray --------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the catch tray: DYNAMIC compound (floor + 4 walls under one rigid root;
    child colliders of one body never self-collide). Root/CoM at the floor-bottom
    centre. Custom spawn funcs apply NO cfg schemas, so mass (MassAPI), damping and
    the friction material are authored here; solve.py asserts the mass readback."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    root = xform.GetPrim()

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.tray_mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.2)
    pxrb.CreateAngularDampingAttr(0.2)
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    hx, hy = cfg.tray_outer[0] / 2, cfg.tray_outer[1] / 2  # 0.090, 0.075
    t, wh, ft = cfg.tray_wall_t, cfg.tray_wall_h, cfg.tray_floor_t

    def box(name, center, size):
        b = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        b.CreateSizeAttr(1.0)
        bx = UsdGeom.Xformable(b.GetPrim())
        bx.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        bx.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
        b.CreateDisplayColorAttr([Gf.Vec3f(*cfg.tray_color)])
        UsdPhysics.CollisionAPI.Apply(b.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(b.GetPrim())
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    box("floor", (0.0, 0.0, ft / 2), (2 * hx, 2 * hy, ft))
    for tag, sx in (("back", -1.0), ("front", 1.0)):
        box(f"wall_{tag}", (sx * (hx - t / 2), 0.0, ft + wh / 2), (t, 2 * hy, wh))
    for tag, sy in (("left", -1.0), ("right", 1.0)):
        box(f"wall_{tag}", (0.0, sy * (hy - t / 2), ft + wh / 2), (2 * hx - 2 * t, t, wh))

    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    mat_path = f"{prim_path}/trayMat"
    sim_utils.spawn_rigid_body_material(
        mat_path,
        sim_utils.RigidBodyMaterialCfg(static_friction=cfg.tray_friction,
                                       dynamic_friction=0.9 * cfg.tray_friction,
                                       restitution=0.0))
    for child in ("floor", "wall_back", "wall_front", "wall_left", "wall_right"):
        bind_physics_material(f"{prim_path}/{child}", mat_path)
    return root


def _tray_spawner_cls() -> Any:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tray" not in _SPAWNER_CACHE:

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            tray_mass: float = 0.25
            tray_outer: tuple = (0.180, 0.150)
            tray_wall_t: float = 0.008
            tray_wall_h: float = 0.040
            tray_floor_t: float = 0.008
            tray_color: tuple = (0.16, 0.45, 0.20)
            tray_friction: float = 0.45
            contact_offset: float = 0.002

        _SPAWNER_CACHE["tray"] = TraySpawnerCfg
    return _SPAWNER_CACHE["tray"]


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class QueueDispenserSceneCfg(BaseCfg):
    """Config for `QueueDispenserScene`. Station geometry is generated into `parts`
    (name, local pos, size, pitch_deg, color) by `__post_init__` from the constants
    below; every predicate gate is stated in the station or tray body frame."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    settle_speed: float = tunable(0.06)      # max |lin vel| (tray + cartons) when judging (m/s)
    tray_settle_avel: float = tunable(0.50)  # max tray |ang vel| when judging (rad/s)
    seat_xy_tol: float = tunable(0.020)      # tray origin within this of the seat centre (m)
    seat_tilt_deg: float = tunable(15.0)     # tray "upright" cone when seated

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    station_yaw_deg: float = tunable(20.0)   # station yaw about nominal (+/- deg)
    station_jitter: float = tunable(0.04)    # station xy jitter (+/- m)
    tray_x_range: tuple = tunable((0.31, 0.41))  # tray start x (station-local, on the ground)
    tray_y_range: tuple = tunable((-0.15, 0.15))  # tray start y
    tray_yaw_deg: float = tunable(180.0)     # tray free yaw (+/- deg)

    # --- info: world placement -------------------------------------------------------------------
    station_pos: tuple = info((0.0, 0.0))    # station origin on the ground (nominal)
    station_yaw_nom_deg: float = info(0.0)   # nominal heading: dispense direction = world +x

    # --- info: feed geometry (station-local; ramp surface z(x) = ramp_z0 - tan(theta)*x) ---------
    ramp_deg: float = info(11.0)
    ramp_z0: float = info(0.315)             # ramp surface height at x = 0 (the rear wall)
    bed_t: float = info(0.008)               # feed-bed plate thickness (queue rests on it)
    lip_x: float = info(0.0596)              # retention lip centre x (bar 8 x 56 mm)
    lip_h: float = info(0.016)               # lip prominence above the FEED BED surface
    tongue_tip_x: float = info(0.117)        # launch tongue tip (drop point)
    shaft_x: tuple = info((0.000, 0.056))    # shaft inner faces (rear, front)
    shaft_half_y: float = info(0.028)        # shaft inner half width
    outlet_top_z: float = info(0.420)        # front wall lower edge (outlet clear height)
    slot_half_y: float = info(0.022)         # rear push slot half width (44 mm)
    slot_top_z: float = info(0.376)          # rear push slot top (sill = ramp surface, 0.315)
    sight_half_y: float = info(0.008)        # sight slot half width (16 mm)
    roof_z: float = info(0.600)              # column inner ceiling
    # --- info: reject pit (station-local) --------------------------------------------------------
    pit_inner_x: tuple = info((0.085, 0.235))
    pit_half_y: float = info(0.055)
    pit_rim_z: float = info(0.200)
    pit_floor_z: float = info(0.012)
    # --- info: tray + seat -----------------------------------------------------------------------
    tray_mass: float = info(0.25)
    tray_outer: tuple = info((0.180, 0.150))
    tray_wall_t: float = info(0.008)
    tray_wall_h: float = info(0.040)
    tray_floor_t: float = info(0.008)
    seat_x: float = info(0.160)              # tray origin at the seat (pit mouth centre)
    seat_z: tuple = info((0.190, 0.215))     # tray origin z window when seated on the rim
    # --- info: in-tray gate (tray body frame, on a carton's centre) ------------------------------
    in_x: float = info(0.072)
    in_y: float = info(0.058)
    in_z: tuple = info((0.004, 0.064))
    # --- info: dirty gate (tray frame; a DISTRACTOR anywhere in the tray's column of
    # space — inside OR piled on top of the contents — spoils the serve) ------------------------
    over_x: float = info(0.095)
    over_y: float = info(0.080)
    over_z: tuple = info((0.004, 0.180))
    # --- info: in-magazine / in-pit gates (station frame, on a carton's centre) ------------------
    mag_x: tuple = info((-0.015, 0.075))
    mag_y: float = info(0.040)
    mag_z: tuple = info((0.280, 0.620))
    pit_gate_z: tuple = info((0.020, 0.160))
    # --- info: cartons ---------------------------------------------------------------------------
    carton_size: tuple = info((0.050, 0.050, 0.080))
    carton_mass: float = info(0.15)
    ketchup_color: tuple = info((0.82, 0.08, 0.06))
    butter_color: tuple = info((0.93, 0.78, 0.12))
    cream_color: tuple = info((0.93, 0.93, 0.90))
    # --- info: materials / colors ----------------------------------------------------------------
    body_color: tuple = info((0.35, 0.37, 0.42))
    trim_color: tuple = info((0.55, 0.58, 0.62))
    pit_color: tuple = info((0.16, 0.17, 0.20))
    tray_color: tuple = info((0.16, 0.45, 0.20))
    ramp_friction: float = info(0.06)        # slick tongue: post-lip slide accelerates
    bed_friction: float = info(0.14)         # feed bed: still self-feeds (mu < tan 11deg)
    carton_friction: float = info(0.25)
    contact_offset: float = info(0.002)
    # --- rubric weights (0.15 + 0.15 + 0.35 = 0.65 = the non-success cap) ------------------------
    w_reject: float = info(0.15)
    w_staged: float = info(0.15)
    w_served: float = info(0.35)

    # Derived (filled in __post_init__): ((name, pos, size, pitch_deg, color), ...)
    parts: tuple = field(default=None, init=False)

    def __post_init__(self) -> None:
        th = math.radians(self.ramp_deg)
        tan, sin, cos = math.tan(th), math.sin(th), math.cos(th)

        def surf(x: float) -> float:
            return self.ramp_z0 - tan * x

        def on_ramp(x: float, half_h: float) -> tuple:
            """Centre of a pitched box whose BOTTOM face rides the ramp surface at x."""
            return (x + half_h * sin, 0.0, surf(x) + half_h * cos)

        body, trim, pit = self.body_color, self.trim_color, self.pit_color
        deg = self.ramp_deg
        p: list = []
        # pedestal under the ramp (clear of the tray seat x >= 0.070 and the rear slot)
        p.append(("pedestal", (0.0175, 0.0, 0.148), (0.055, 0.072, 0.296), 0.0, body))
        # feed ramp: top surface passes through (0.057, surf(0.057)); spans x ~ -0.003..0.117
        rc = (0.057 - 0.006 * sin, 0.0, surf(0.057) - 0.006 * cos)
        p.append(("ramp", rc, (0.122, 0.076, 0.012), deg, trim))
        # feed bed: grippier plate ON the ramp, behind the lip (the queue rests on it)
        p.append(("bed", on_ramp(0.0265, self.bed_t / 2), (0.058, 0.056, self.bed_t),
                  deg, body))
        # retention lip at the shaft outlet: one bar whose bottom rides the RAMP and
        # whose crest stands `lip_h` proud of the BED surface (bar height = bed + lip)
        p.append(("lip", on_ramp(self.lip_x, (self.bed_t + self.lip_h) / 2),
                  (0.008, 0.056, self.bed_t + self.lip_h), deg, trim))
        # launch-tongue side rails
        for tag, sy in (("rail_l", -1.0), ("rail_r", 1.0)):
            cx, _cy, cz = on_ramp(0.089, 0.010)
            p.append((tag, (cx, sy * 0.032, cz), (0.052, 0.008, 0.020), deg, trim))
        # column side walls (inner faces y +/-0.028), z 0.280..0.600
        for tag, sy in (("side_l", -1.0), ("side_r", 1.0)):
            p.append((tag, (0.028, sy * 0.032, 0.440), (0.080, 0.008, 0.320), 0.0, body))
        # rear wall: two strips flanking the push slot, then solid above it
        for tag, sy in (("rear_l", -1.0), ("rear_r", 1.0)):
            p.append((tag, (-0.005, sy * 0.029, 0.328), (0.010, 0.014, 0.096), 0.0, body))
        p.append(("rear_top", (-0.005, 0.0, 0.488), (0.010, 0.072, 0.224), 0.0, body))
        # front wall strips above the outlet, leaving the central sight slot
        # (lower edge = outlet_top_z: it shields the dropping follower carton)
        fz = (self.outlet_top_z + 0.600) / 2
        for tag, sy in (("front_l", -1.0), ("front_r", 1.0)):
            p.append((tag, (0.060, sy * 0.022, fz),
                      (0.008, 0.028, 0.600 - self.outlet_top_z), 0.0, body))
        # roof
        p.append(("roof", (0.026, 0.0, 0.605), (0.088, 0.080, 0.010), 0.0, body))
        # reject pit
        p.append(("pit_floor", (0.160, 0.0, 0.006), (0.166, 0.126, 0.012), 0.0, pit))
        p.append(("pit_back", (0.081, 0.0, 0.100), (0.008, 0.126, 0.200), 0.0, pit))
        p.append(("pit_front", (0.239, 0.0, 0.100), (0.008, 0.126, 0.200), 0.0, pit))
        for tag, sy in (("pit_l", -1.0), ("pit_r", 1.0)):
            p.append((tag, (0.160, sy * 0.059, 0.100), (0.166, 0.008, 0.200), 0.0, pit))
        self.parts = tuple(p)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("queue_dispenser")
class QueueDispenserScene(BaseScene):
    cfg: QueueDispenserSceneCfg

    ITEMS = ("ketchup", "butter", "cream_cheese")  # item 0 is the target

    def __init__(self, cfg: QueueDispenserSceneCfg | None = None) -> None:
        super().__init__(cfg or QueueDispenserSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
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

        # --- station: standard-schema KINEMATIC cuboids, one per part -------------------------
        # (cfg schemas all apply — friction, contact offsets; reset() poses them coherently)
        for name, pos, size, pitch, color in c.parts:
            if name in ("ramp", "lip"):
                fr = c.ramp_friction
            elif name == "bed":
                fr = c.bed_friction
            else:
                fr = 0.40
            h = math.radians(pitch) / 2
            out["st_" + name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/St_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=fr, dynamic_friction=0.9 * fr, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(pos[0], pos[1], pos[2]),
                    rot=(math.cos(h), 0.0, math.sin(h), 0.0)),
            )

        # --- tray: dynamic compound (custom spawner authors mass + material) ------------------
        tray_cls = _tray_spawner_cls()
        out["tray"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Tray",
            spawn=tray_cls(tray_mass=c.tray_mass, tray_outer=c.tray_outer,
                           tray_wall_t=c.tray_wall_t, tray_wall_h=c.tray_wall_h,
                           tray_floor_t=c.tray_floor_t, tray_color=c.tray_color,
                           contact_offset=c.contact_offset),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.36, 0.0, 0.002)),
        )

        # --- cartons --------------------------------------------------------------------------
        colors = {"ketchup": c.ketchup_color, "butter": c.butter_color,
                  "cream_cheese": c.cream_color}
        for i, name in enumerate(self.ITEMS):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Carton_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=c.carton_size,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.carton_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.10,
                        sleep_threshold=0.0, stabilization_threshold=0.0,
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=4),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.carton_friction,
                        dynamic_friction=0.9 * c.carton_friction, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colors[name]),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0 + 0.3 * i, 1.0, 0.05)),
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

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        self.station_parts: dict[str, RigidObject] = {
            name: env.iscene["st_" + name] for name, *_rest in c.parts}
        self.tray: RigidObject = env.iscene["tray"]
        self.items: list[RigidObject] = [env.iscene[n] for n in self.ITEMS]
        self.ketchup: RigidObject = self.items[0]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # station pose (authoritative: the parts are kinematic, they never drift)
        self._st_pos = torch.zeros(n, 3, device=dev)
        self._st_quat = torch.zeros(n, 4, device=dev)
        self._st_quat[:, 0] = 1.0
        # levels[e, i]: queue level (0 = bottom) of item i this episode
        self.levels = torch.zeros(n, 3, dtype=torch.long, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._rejected = torch.zeros(n, 3, dtype=torch.bool, device=dev)
        self._staged = torch.zeros(n, dtype=torch.bool, device=dev)
        self._served = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the station (yaw + xy jitter) part by part, stack the
        cartons in a freshly shuffled queue on the feed ramp, drop the tray loose on
        the open ground, clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        th = math.radians(c.ramp_deg)

        # --- station pose ---
        yaw = math.radians(c.station_yaw_nom_deg) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.station_yaw_deg)
        q_st = _qz(yaw)
        pp = torch.zeros(m, 3, device=dev)
        pp[:, 0] = c.station_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.station_jitter
        pp[:, 1] = c.station_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.station_jitter
        self._st_pos[env_ids] = pp
        self._st_quat[env_ids] = q_st

        for name, pos, _size, pitch, _color in c.parts:
            loc = torch.tensor(pos, device=dev, dtype=torch.float).expand(m, 3)
            qp = _qy_const(pitch, dev).expand(m, 4) if pitch else None
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pp + _qapply(q_st, loc) + origin
            st[:, 3:7] = _qmul(q_st, qp) if qp is not None else q_st
            self.station_parts[name].write_root_state_to_sim(st, env_ids)

        # --- queue: shuffle which carton sits at which level ---
        # (torch.rand + argsort, not randint: the first randint after manual_seed is
        # near-constant across seeds on this stack)
        levels = torch.rand(m, 3, device=dev).argsort(dim=1)
        self.levels[env_ids] = levels
        # Spawn UPRIGHT (x-extent 50 mm < the 56 mm shaft depth: no wall contact at
        # the write) with 4 mm inter-carton gaps; the 120-step settle drops the
        # stack onto the pitched feed bed and slides it against the lip.
        bed_top = (c.ramp_z0 - math.tan(th) * 0.028) + c.bed_t / math.cos(th)
        for i, body in enumerate(self.items):
            lev = levels[:, i].float()
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = 0.028
            loc[:, 2] = bed_top + 0.042 + lev * 0.084
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pp + _qapply(q_st, loc) + origin
            st[:, 3:7] = q_st
            body.write_root_state_to_sim(st, env_ids)

        # --- tray: loose on the open ground, clear of the station ---
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.tray_x_range[0] + torch.rand(m, device=dev) \
            * (c.tray_x_range[1] - c.tray_x_range[0])
        loc[:, 1] = c.tray_y_range[0] + torch.rand(m, device=dev) \
            * (c.tray_y_range[1] - c.tray_y_range[0])
        loc[:, 2] = 0.003
        q_tr = _qz((torch.rand(m, device=dev) * 2 - 1) * math.radians(c.tray_yaw_deg))
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pp + _qapply(q_st, loc) + origin
        st[:, 3:7] = _qmul(q_st, q_tr)
        self.tray.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._rejected[env_ids] = False
        self._staged[env_ids] = False
        self._served[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out: dict[str, Any] = {
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "st_pos": self._st_pos[env_ids].clone(),
            "st_quat": self._st_quat[env_ids].clone(),
            "levels": self.levels[env_ids].clone(),
            "rejected": self._rejected[env_ids].clone(),
            "staged": self._staged[env_ids].clone(),
            "served": self._served[env_ids].clone(),
        }
        for name, body in zip(self.ITEMS, self.items):
            out[name] = body.data.root_state_w[env_ids].clone()
        for name, body in self.station_parts.items():
            out["st_" + name] = body.data.root_state_w[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        for name, body in zip(self.ITEMS, self.items):
            body.write_root_state_to_sim(state[name], env_ids)
        for name, body in self.station_parts.items():
            body.write_root_state_to_sim(state["st_" + name], env_ids)
        self._st_pos[env_ids] = state["st_pos"]
        self._st_quat[env_ids] = state["st_quat"]
        self.levels[env_ids] = state["levels"]
        self._rejected[env_ids] = state["rejected"]
        self._staged[env_ids] = state["staged"]
        self._served[env_ids] = state["served"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A grey DISPENSER COLUMN (~90 mm square shaft, 610 mm tall) stands on the "
            "ground. Sealed inside it, three cartons of identical shape (50 x 50 x 80 "
            "mm) sit stacked in a vertical QUEUE on a tilted feed ramp: red KETCHUP, "
            "yellow BUTTER, white CREAM CHEESE — which carton sits at which level is "
            "shuffled every episode. Read the queue order through the narrow vertical "
            "SIGHT SLOT (16 mm) in the column's front face; the column is roofed and "
            "closed everywhere else, so no carton can be grasped, lifted out, or "
            "pulled through a slot. Cartons leave ONLY through the bottom OUTLET on "
            "the front side: the lowest carton rests against a small RETENTION LIP, "
            "and a firm push on its back face through the 44 mm REAR PUSH SLOT (low "
            "on the column's back face) pops it over the lip — gravity then slides it "
            "down the short launch tongue and drops it off the tip. One push ejects "
            "exactly one carton (the queue slides down and re-seats); a gentle nudge "
            "does not defeat the lip. The tongue tip overhangs a deep dark REJECT "
            "PIT (mouth ~150 x 110 mm, 200 mm deep): anything dispensed with nothing "
            "staged below falls into the pit and is LOST — the pit is too deep and "
            "narrow to retrieve from. A green open-top TRAY (180 x 150 mm, 40 mm "
            "walls) lies loose on the open ground nearby. Seated flat on the pit's "
            "rim it covers the mouth and catches whatever drops off the tongue; it "
            "can be slid in level from the front, under the tongue (32 mm of "
            "clearance over its walls). The column's position and heading, the queue "
            "order, and the tray's start pose vary per episode.\n"
            "Goal: end with the red KETCHUP carton at rest INSIDE the tray, the tray "
            "still seated on the pit rim, and NO other carton in or on the tray. "
            "Since the "
            "queue only dispenses bottom-first, every carton stacked BELOW the "
            "ketchup must first be ejected into the reject pit (before the tray is "
            "staged); then seat the tray on the rim and eject the ketchup so it "
            "lands in the tray. Ejecting the ketchup before the tray is staged drops "
            "it into the pit — unrecoverable. A wrong carton left in (or piled on "
            "top of) the tray, the ketchup anywhere but inside the seated tray, or "
            "the tray off its seat all fail."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Serve the red ketchup carton from the dispenser column into the green "
            "tray: push out and discard into the reject pit every carton queued below "
            "the ketchup, seat the tray on the pit rim under the outlet, then push "
            "the ketchup out so it drops into the tray. Only the ketchup may end up "
            "in the tray, and the tray must stay seated on the rim."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def station_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the station frame, (N,3) -> (N,3)."""
        return _qapply(_qinv(self._st_quat),
                       pos_w - self._st_pos - self.env_origins)

    def station_world(self, loc: torch.Tensor) -> torch.Tensor:
        """Station-local points -> world, (N,3) -> (N,3)."""
        return self._st_pos + _qapply(self._st_quat, loc) + self.env_origins

    def _tray_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.tray.data.root_quat_w,
                                  pos_w - self.tray.data.root_pos_w)

    def in_tray(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: carton centre inside the tray's interior volume (tray frame)."""
        c = self.cfg
        loc = self._tray_local(body.data.root_pos_w)
        return (loc[:, 0].abs() < c.in_x) & (loc[:, 1].abs() < c.in_y) \
            & (loc[:, 2] > c.in_z[0]) & (loc[:, 2] < c.in_z[1])

    def over_tray(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: carton centre anywhere in the tray's column of space — inside
        the tray OR piled on top of its contents / resting across its walls. Used
        for the DIRTY clause so a follower stacked on the served carton still
        spoils the serve (the plain `in_tray` z-gate would miss it)."""
        c = self.cfg
        loc = self._tray_local(body.data.root_pos_w)
        return (loc[:, 0].abs() < c.over_x) & (loc[:, 1].abs() < c.over_y) \
            & (loc[:, 2] > c.over_z[0]) & (loc[:, 2] < c.over_z[1])

    def in_magazine(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: carton centre inside the column shaft (station frame)."""
        c = self.cfg
        loc = self.station_local(body.data.root_pos_w)
        return (loc[:, 0] > c.mag_x[0]) & (loc[:, 0] < c.mag_x[1]) \
            & (loc[:, 1].abs() < c.mag_y) \
            & (loc[:, 2] > c.mag_z[0]) & (loc[:, 2] < c.mag_z[1])

    def in_pit(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: carton centre down inside the reject pit (station frame)."""
        c = self.cfg
        loc = self.station_local(body.data.root_pos_w)
        return (loc[:, 0] > c.pit_inner_x[0]) & (loc[:, 0] < c.pit_inner_x[1]) \
            & (loc[:, 1].abs() < c.pit_half_y) \
            & (loc[:, 2] > c.pit_gate_z[0]) & (loc[:, 2] < c.pit_gate_z[1])

    def tray_seated(self) -> torch.Tensor:
        """(N,) bool: tray origin at the seat (pit mouth centre) on the rim, upright."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        loc = self.station_local(self.tray.data.root_pos_w)
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        up = quat_apply(self.tray.data.root_quat_w, ez)
        upright = up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(c.seat_tilt_deg))
        return ((loc[:, 0] - c.seat_x).abs() < c.seat_xy_tol) \
            & (loc[:, 1].abs() < c.seat_xy_tol) \
            & (loc[:, 2] > c.seat_z[0]) & (loc[:, 2] < c.seat_z[1]) & upright

    def needed_ahead(self) -> torch.Tensor:
        """(N,3) bool: item i is a distractor queued BELOW the ketchup — it must be
        dispensed (rejected) before the ketchup can come out."""
        klev = self.levels[:, 0].unsqueeze(1)
        out = self.levels < klev
        out[:, 0] = False
        return out

    def settled(self) -> torch.Tensor:
        """(N,) bool: tray + all cartons |lin vel| below `settle_speed`, tray ang slow."""
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                         for b in (self.tray, *self.items)], dim=1)
        tray_still = self.tray.data.root_ang_vel_w.norm(dim=-1) < self.cfg.tray_settle_avel
        return (v < self.cfg.settle_speed).all(dim=1) & tray_still

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w for b in (self.tray, *self.items)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        fin = self._finite()
        slow = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) < 0.10
                            for b in self.items], dim=1)
        out_mag = torch.stack([~self.in_magazine(b) for b in self.items], dim=1)
        self._rejected |= self.needed_ahead() & out_mag & slow & fin.unsqueeze(1)
        tray_slow = self.tray.data.root_lin_vel_w.norm(dim=-1) < 0.10
        self._staged |= self.tray_seated() & tray_slow & fin
        self._served |= self.in_tray(self.ketchup) & slow[:, 0] & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the ketchup carton inside the tray AND the tray seated on the
        pit rim AND no other carton in the tray, everything settled and finite. All
        clauses are live physical outcomes; the required ordering is enforced by the
        queue and the pit geometry, not by rubric fiat."""
        self._update_latches()
        clean = ~(self.over_tray(self.items[1]) | self.over_tray(self.items[2]))
        return self.in_tray(self.ketchup) & self.tray_seated() & clean \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: latched partial credit — rejects (share of the
        cartons queued below the ketchup ejected from the column), tray staged on the
        rim, ketchup at rest in the tray — capped at 0.65; exactly 1.0 iff success()
        holds live. The null policy scores ~0 (every credit requires driving a body
        somewhere it does not start)."""
        c = self.cfg
        self._update_latches()
        need = self.needed_ahead()
        n_need = need.sum(dim=1)
        rej = (self._rejected & need).sum(dim=1).float()
        s = torch.where(n_need > 0,
                        c.w_reject * rej / n_need.clamp(min=1).float(),
                        torch.zeros_like(rej))
        s = s + c.w_staged * self._staged.float() + c.w_served * self._served.float()
        s = s.clamp(max=0.65)
        return torch.where(self.success(), torch.ones_like(s), s)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="queue_dispenser", robot="null"))

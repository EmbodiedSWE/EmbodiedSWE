"""ChairFoldawayScene — fold each folding chair flat, then slot it into the storage rack
(sim_gen task `stack_chairs_i326`).

Derived from rlbench/stack_chairs, but STRATEGICALLY different: the seed is transport +
STACKING — pick each chair up and pile it on the previous one, success being a vertical
chair-on-chair tower of rigid chairs. Here every chair is an ARTICULATED two-body
mechanism (a base frame + a seat on a real revolute hinge with hard stops), and the goal
container is a slotted storage rack whose slots are NARROWER than any deployed chair.
The task is RECONFIGURE-then-store: swing each chair's seat up past its gravity balance
point so it rests folded against the backrest (a gravity-bistable click — deployed and
folded are both stable, nothing in between is), which shrinks the chair's footprint from
197 mm to 95 mm, then lower the folded chair down into a rack slot (115 mm gap) until it
stands on the rack floor — one chair per slot, all three slots filled. A solver needs a
different PLAN (no piling — piled chairs score nothing; per object it must first operate
an internal DOF, and only the folded configuration can pass the slot, so "transport
harder" can never substitute for the fold) and a different code structure (a hinge-angle
gate read from the RELATIVE pose of two bodies of one object + slot occupancy in the
rack's randomized frame, instead of any on-top-of / pile-height test). It also differs
from the sibling desk-tuck task (`stack_chairs_i110`): that is horizontal threading of
rigid chairs under an overhang with a width-matching subproblem; here the chairs are
identical, insertion is VERTICAL through a top opening, and the physical gate is the
object's own articulation state, not a size-to-aperture assignment.

Geometry (procedural; the rack is 7 KINEMATIC cuboids re-posed every reset; each chair
is TWO dynamic rigid bodies joined by a spawn-authored revolute joint):
  - chair frame: foot plate 95 x 140 x 12 mm + backrest panel 14 x 140 x 190 mm, body
    origin 40 mm above the foot bottom, mass 0.5 kg (CoM authored at the origin).
  - chair seat: 130 x 130 x 12 mm panel on a hinge at frame-local (18, 0, 60) mm, axis
    Y, joint limits [-100 deg, 0 deg]. At 0 deg the seat sticks out horizontally
    (+x, reach 152 mm from the origin) and gravity presses it onto the 0-deg stop; at
    -100 deg (readback +100 deg) the seat CoM is ~5 deg PAST vertical, so gravity
    presses it onto the -100 deg stop — a bistable fold with no stable state between.
  - rack: floor plate 260 x 420 x 12 mm, two side walls, four fins forming THREE
    identical top-open slots, each 115 mm wide and 130 mm deep (fin top 142 mm up).

Footprint arithmetic (asserted in `__post_init__`): folded chair x-extent 95 mm passes
the 115 mm slot gap with 20 mm clearance; a DEPLOYED chair's smallest horizontal extent
is its 140 mm width > gap, so the WHOLE chair can never pass below the fin tops in any
yaw — at most its 95 mm foot drops in while the fin CATCHES the protruding seat and
props it well below the fold gate (a geometric cap, asserted). Two folded chairs cannot
share a slot (2 x 95 = 190 mm > gap; and both inside the x band would overlap). So
"every slot holds exactly one folded chair" is forced by PHYSICS, and the fold gate is
forced by the aperture, not by rubric fiat.

success(): every slot covered — exactly one chair whose seat reads folded (hinge angle
>= `fold_min_deg`), origin inside that slot's x/y band (rack frame) and in the on-rack-
floor height band, upright within `upright_max_deg`, stillness SUSTAINED
`settle_steps` substeps.

score(): per chair, 1.0 if racked NOW (in some slot, settled), else `fold_credit` once
its seat has ever held the folded angle `fold_hold_steps` consecutive substeps (latched
in post_step — fold credit never evaporates); mean over chairs; 1.0 iff success().
Null policy scores ~0 (chairs spawn deployed on the floor in front of the rack).

Per-episode randomization (readback-verified in smoke): rack position + yaw, chair
scatter slot permutation + xy jitter + free yaw (overlap-rejected with a deterministic
fallback). Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv

# chairs are IDENTICAL — colors only make describe() and telemetry unambiguous
CHAIRS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("red", (0.85, 0.10, 0.10)),
    ("green", (0.10, 0.70, 0.15)),
    ("blue", (0.12, 0.30, 0.85)),
)


# ----- custom compound spawners (chair frame / chair seat) -------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _material(stage, path: str, static: float, dynamic: float):
    """One USD physics material (explicit friction + zero restitution — custom-spawner
    colliders otherwise land on engine defaults)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, size, center, color, contact_offset: float, material) -> None:
    """One axis-aligned box child prim (translate -> scale, authored exactly once — the
    duplicate-xformOp trap is avoided by never re-authoring an existing prim's ops)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(seg.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
        material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _rigid_setup(root, mass: float, com, inertia, lin_damp: float, ang_damp: float) -> None:
    """RigidBody + AUTHORED mass/CoM/inertia + PhysX solver settings. Zero sleep /
    stabilization thresholds: a sleeping body silently ignores applied wrenches, which
    the solve's fold-torque servo depends on."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    m = UsdPhysics.MassAPI.Apply(root)
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _spawn_chair_frame(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The chair FRAME: foot plate + backrest panel, one dynamic rigid body. Body
    origin `origin_h` above the foot bottom, on the chair centerline; local +x = the
    side the deployed seat sticks out toward (the chair's front)."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_setup(root, cfg.mass, (0.0, 0.0, 0.0), (0.0030, 0.0025, 0.0012), 0.05, 0.20)
    mat = _material(stage, f"{prim_path}/phys_mat", 0.50, 0.45)
    co = cfg.contact_offset
    r, g, b = cfg.color
    dark = (r * 0.55, g * 0.55, b * 0.55)
    _box(stage, f"{prim_path}/foot", cfg.foot_size, cfg.foot_center, dark, co, mat)
    _box(stage, f"{prim_path}/panel", cfg.panel_size, cfg.panel_center, cfg.color, co, mat)
    return root


def _spawn_chair_seat(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The chair SEAT: one panel, one dynamic rigid body, on a spawn-authored revolute
    joint (axis Y) against the sibling frame body. Body origin AT THE HINGE; the panel
    extends +x (deployed). Limits [-fold_stop_deg (folded, seat up past vertical), 0
    (deployed, horizontal hard stop)] — gravity presses onto whichever stop the seat is
    nearer, so the fold is bistable. The joint is authored IN THE SPAWNER so it exists
    before the physics parse; the joint filters frame<->seat collisions."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_setup(root, cfg.mass, cfg.com, (9.0e-5, 9.0e-5, 1.7e-4), 0.05, 0.15)
    mat = _material(stage, f"{prim_path}/phys_mat", 0.50, 0.45)
    r, g, b = cfg.color
    lite = (min(1.0, r * 1.15 + 0.10), min(1.0, g * 1.15 + 0.10), min(1.0, b * 1.15 + 0.10))
    _box(stage, f"{prim_path}/panel", cfg.panel_size, cfg.com, lite, cfg.contact_offset, mat)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([f"{base}/{cfg.frame_name}"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateCollisionEnabledAttr(False)
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in cfg.anchor_frame]))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-float(cfg.fold_stop_deg))
    j.CreateUpperLimitAttr(0.0)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazy: module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "frame" not in _SPAWNER_CACHE:

        @configclass
        class FrameSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_chair_frame)
            mass: float = 0.5
            color: tuple = (0.8, 0.1, 0.1)
            foot_size: tuple = (0.095, 0.140, 0.012)
            foot_center: tuple = (0.0025, 0.0, -0.034)
            panel_size: tuple = (0.014, 0.140, 0.190)
            panel_center: tuple = (-0.014, 0.0, 0.055)
            contact_offset: float = 0.002

        @configclass
        class SeatSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_chair_seat)
            mass: float = 0.06
            color: tuple = (0.8, 0.1, 0.1)
            panel_size: tuple = (0.130, 0.130, 0.012)
            com: tuple = (0.069, 0.0, -0.006)
            frame_name: str = "ChairFrame_red"
            anchor_frame: tuple = (0.010, 0.0, 0.060)
            fold_stop_deg: float = 100.0
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(frame=FrameSpawnerCfg, seat=SeatSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ChairFoldawaySceneCfg(BaseCfg):
    """Config for `ChairFoldawayScene`. The footprint arithmetic that makes the fold
    gate physical (folded passes / deployed never passes / one chair per slot) is
    asserted in `__post_init__`."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    fold_min_deg: float = tunable(80.0)  # seat hinge angle that counts as FOLDED (rest = 100)
    upright_max_deg: float = tunable(15.0)  # chair up-axis within this of world-up
    band_x: float = tunable(0.045)  # |rack-frame x| of a racked chair origin (along the slot)
    band_y: float = tunable(0.045)  # |rack-frame y - slot center| of a racked chair origin
    z_lo: float = tunable(0.038)  # chair-origin height band: standing ON THE RACK FLOOR
    z_hi: float = tunable(0.068)  # (a chair standing ON the fin tops reads ~0.182)
    settle_lin: float = tunable(0.05)  # max frame |lin vel| at judging (m/s)
    settle_ang: float = tunable(0.60)  # max frame/seat |ang vel| at judging (rad/s)
    seat_lin: float = tunable(0.08)  # max seat |lin vel| at judging (m/s)
    settle_steps: int = tunable(20)  # substeps of SUSTAINED stillness
    fold_hold_steps: int = tunable(10)  # substeps the fold must hold to latch fold credit
    fold_credit: float = tunable(0.35)  # per-chair latched credit once ever folded

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    rack_pos: tuple = tunable((0.47, 0.0))  # rack-frame origin on the floor
    rack_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the rack
    rack_yaw_deg: float = tunable(14.0)  # uniform +/- rack yaw
    scatter_x: float = tunable(0.05)  # chair scatter row x (in front of the rack)
    scatter_dy: float = tunable(0.20)  # scatter slot pitch along y
    scatter_jitter: float = tunable(0.025)  # uniform +/- xy jitter per chair
    chair_yaw_deg: float = tunable(180.0)  # uniform +/- yaw per scattered chair

    # --- info: structure (what the spawners author) ------------------------------------------
    slot_gap: float = info(0.115)  # clear width of each rack slot
    fin_t: float = info(0.012)  # fin thickness
    fin_len: float = info(0.200)  # fin length along the slot (rack x)
    fin_h: float = info(0.130)  # fin height above the rack floor plate
    wall_off: float = info(0.106)  # side-wall center |x| (inner faces at +/-0.100)
    wall_t: float = info(0.012)
    plate_size: tuple = info((0.260, 0.420, 0.012))  # rack floor plate
    foot_size: tuple = info((0.095, 0.140, 0.012))  # chair foot plate
    foot_center: tuple = info((0.0025, 0.0, -0.034))
    panel_size: tuple = info((0.014, 0.140, 0.190))  # backrest panel
    panel_center: tuple = info((-0.014, 0.0, 0.055))
    seat_size: tuple = info((0.130, 0.130, 0.012))  # seat panel
    seat_com: tuple = info((0.069, 0.0, -0.006))  # seat CoM / panel center, seat frame
    hinge: tuple = info((0.018, 0.0, 0.060))  # hinge anchor, frame-local
    fold_stop_deg: float = info(100.0)  # folded hard stop (readback rest angle)
    origin_h: float = info(0.040)  # frame origin above the foot bottom
    frame_mass: float = info(0.5)
    seat_mass: float = info(0.06)
    contact_offset: float = info(0.002)

    # ----- derived geometry -----------------------------------------------------------------
    @property
    def fin_y(self) -> tuple:  # fin center y, rack frame, left -> right
        p = self.slot_gap + self.fin_t
        return (-1.5 * p, -0.5 * p, 0.5 * p, 1.5 * p)

    @property
    def slot_y(self) -> tuple:  # slot center y, rack frame
        p = self.slot_gap + self.fin_t
        return (-p, 0.0, p)

    @property
    def foot_x(self) -> tuple:  # chair foot plate x extent, frame-local
        return (self.foot_center[0] - self.foot_size[0] / 2,
                self.foot_center[0] + self.foot_size[0] / 2)

    @property
    def deployed_reach(self) -> float:  # deployed seat tip x, frame-local
        return self.hinge[0] + self.seat_com[0] + self.seat_size[0] / 2

    @property
    def chair_w(self) -> float:  # chair y width (foot = panel = seat + margin)
        return self.foot_size[1]

    @property
    def fin_top(self) -> float:  # fin top height above the floor
        return self.plate_size[2] + self.fin_h

    @property
    def rack_floor_z(self) -> float:  # racked chair origin height (on the rack floor)
        return self.plate_size[2] + self.origin_h

    def _folded_x_extent(self) -> tuple:
        """Chair x extent (frame-local) with the seat folded onto the -fold_stop stop."""
        th = math.radians(self.fold_stop_deg)
        hx = self.hinge[0]
        lo, hi = self.foot_x
        for dx in (self.seat_com[0] - self.seat_size[0] / 2, self.seat_com[0] + self.seat_size[0] / 2):
            for dz in (self.seat_com[2] - self.seat_size[2] / 2, self.seat_com[2] + self.seat_size[2] / 2):
                x = hx + dx * math.cos(th) - dz * math.sin(th)
                lo, hi = min(lo, x), max(hi, x)
        return (lo, hi)

    def __post_init__(self) -> None:
        c = self
        # -- the fold is gravity-bistable: seat CoM tilt puts the balance ~5 deg short of
        #    the folded stop, so both stops are pressed-on rest states --
        com_tilt = math.degrees(math.atan2(-c.seat_com[2], c.seat_com[0]))
        balance = 90.0 + com_tilt
        assert c.fold_stop_deg >= balance + 4.0, "folded stop must sit past the gravity balance"
        assert c.fold_min_deg <= c.fold_stop_deg - 15.0, "fold gate must clear the rest angle"
        assert c.fold_min_deg >= balance - 20.0
        # -- folded chair passes the slot gap; deployed chair NEVER does (any yaw) --
        flo, fhi = c._folded_x_extent()
        folded_x = fhi - flo
        assert c.slot_gap - folded_x >= 0.018, "folded chair must pass the slot with >= 18 mm"
        assert min(c.chair_w, c.deployed_reach - c.foot_x[0]) >= c.slot_gap + 0.020, \
            "a deployed chair's smallest horizontal extent must exceed the slot gap"
        # -- folded seat stays inside the foot span (folded footprint = the foot plate) --
        assert flo >= c.foot_x[0] - 1e-6 and fhi <= c.foot_x[1] + 1e-6
        # -- folded seat does not cross the backrest panel below the panel top --
        th = math.radians(c.fold_stop_deg)
        panel_face = c.panel_center[0] + c.panel_size[0] / 2
        panel_top = c.panel_center[2] + c.panel_size[2] / 2
        r_at_top = (panel_top - c.hinge[2]) / math.sin(th)
        assert c.hinge[0] + r_at_top * math.cos(th) - c.seat_size[2] / 2 * math.sin(th) \
            >= panel_face + 0.002, "folded seat (back face) must clear the backrest panel"
        # -- one chair per slot is physical --
        assert 2.0 * folded_x >= c.slot_gap + 0.020, "two folded chairs must not share a slot"
        assert c.chair_w >= 2.0 * c.band_x + 0.020, "two in-band chairs in one slot would overlap"
        # -- height band: on the rack floor inside; perched on the fin tops far outside --
        assert c.z_lo < c.rack_floor_z < c.z_hi
        assert c.fin_top + c.origin_h > c.z_hi + 0.020, \
            "a chair perched on the fin tops must fail the height band"
        # -- a chair standing IN a slot cannot deploy its seat: the fin props it well
        #    below the fold gate (steepest prop = seat underside on the fin-top inner
        #    corner; +12 deg pessimism for panel thickness) --
        corner_x = c.slot_gap / 2 - c.hinge[0]
        corner_z = c.fin_top - c.origin_h - c.hinge[2]
        propped = math.degrees(math.atan2(corner_z, corner_x)) + 12.0
        assert propped <= c.fold_min_deg - 10.0, "an in-slot propped seat must fail the fold gate"
        # -- slots identical and disjoint; bands inside the slot --
        assert c.band_y <= c.slot_gap / 2
        assert c.band_x <= c.fin_len / 2 - 0.040
        # -- scatter zone clear of the rack at worst-case jitter+yaw --
        half_diag = (c.plate_size[0] / 2 * math.cos(math.radians(c.rack_yaw_deg))
                     + c.plate_size[1] / 2 * math.sin(math.radians(c.rack_yaw_deg)))
        worst_edge = c.rack_pos[0] - c.rack_jitter - half_diag
        worst_reach = c.scatter_x + c.scatter_jitter \
            + math.hypot(c.deployed_reach, c.chair_w / 2)
        assert worst_reach + 0.015 < worst_edge, "scattered chairs must never spawn in the rack"
        assert c.settle_steps >= 12 and c.fold_hold_steps >= 6
        assert 0.0 < c.fold_credit < 1.0


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("chair_foldaway")
class ChairFoldawayScene(BaseScene):
    cfg: ChairFoldawaySceneCfg

    def __init__(self, cfg: ChairFoldawaySceneCfg | None = None) -> None:
        super().__init__(cfg or ChairFoldawaySceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        steel = (0.52, 0.54, 0.58)
        steel2 = (0.40, 0.42, 0.46)

        def piece(name: str, size, color) -> RigidObjectCfg:
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.30, dynamic_friction=0.25, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                # nominal spawn pose; reset() writes the real per-episode poses
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.rack_pos[0], 0.0, 0.5)),
            )

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.50, dynamic_friction=0.45, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "plate": piece("Plate", c.plate_size, steel2),
            "wall_lo": piece("Wall_lo", (c.wall_t, c.plate_size[1] - 0.015, c.fin_h), steel),
            "wall_hi": piece("Wall_hi", (c.wall_t, c.plate_size[1] - 0.015, c.fin_h), steel),
        }
        for k in range(4):
            out[f"fin_{k}"] = piece(f"Fin_{k}", (c.fin_len, c.fin_t, c.fin_h), steel)
        for i, (name, rgb) in enumerate(CHAIRS):
            fx = c.scatter_x
            fy = (i - 1) * c.scatter_dy
            fz = c.origin_h + 0.003
            out[f"frame_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/ChairFrame_" + name,
                spawn=sp["frame"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.frame_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.frame_mass, color=rgb, foot_size=c.foot_size,
                    foot_center=c.foot_center, panel_size=c.panel_size,
                    panel_center=c.panel_center, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(fx, fy, fz)),
            )
            out[f"seat_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/ChairSeat_" + name,
                spawn=sp["seat"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.seat_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.seat_mass, color=rgb, panel_size=c.seat_size,
                    com=c.seat_com, frame_name="ChairFrame_" + name,
                    anchor_frame=c.hinge, fold_stop_deg=c.fold_stop_deg,
                    contact_offset=c.contact_offset),
                # spawn at the deployed joint pose relative to the frame
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(fx + c.hinge[0], fy + c.hinge[1], fz + c.hinge[2])),
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
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n, dev = env.num_envs, env.device
        self.frames: list[RigidObject] = [env.iscene[f"frame_{nm}"] for nm, _ in CHAIRS]
        self.seats: list[RigidObject] = [env.iscene[f"seat_{nm}"] for nm, _ in CHAIRS]
        self.pieces: dict[str, RigidObject] = {
            nm: env.iscene[nm] for nm in ("plate", "wall_lo", "wall_hi",
                                          "fin_0", "fin_1", "fin_2", "fin_3")}
        self.env_origins = env.iscene.env_origins
        # per-episode rack pose (written at reset, readable by rubric + solver)
        self.rack_xy = torch.zeros(n, 2, device=dev)
        self.rack_yaw = torch.zeros(n, device=dev)
        # progress latches (post_step): chair c has ever HELD the folded angle
        self.fold_streak = torch.zeros(n, 3, device=dev)
        self.fold_latch = torch.zeros(n, 3, device=dev)
        # sustained per-chair stillness counter (consecutive still substeps)
        self.still_count = torch.zeros(n, 3, device=dev)

    @staticmethod
    def _seg_dist(p0: torch.Tensor, p1: torch.Tensor,
                  q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
        """Batched exact min distance between 2D segments [p0,p1] and [q0,q1], (m,)."""
        d1, d2, r = p1 - p0, q1 - q0, p0 - q0
        a = (d1 * d1).sum(-1).clamp(min=1e-12)
        e = (d2 * d2).sum(-1).clamp(min=1e-12)
        b = (d1 * d2).sum(-1)
        c_ = (d1 * r).sum(-1)
        f = (d2 * r).sum(-1)
        denom = (a * e - b * b)
        s = torch.where(denom.abs() > 1e-12, (b * f - c_ * e) / denom.clamp(min=1e-12),
                        torch.zeros_like(denom)).clamp(0.0, 1.0)
        t = ((b * s + f) / e).clamp(0.0, 1.0)
        s = ((b * t - c_) / a).clamp(0.0, 1.0)
        cp = p0 + d1 * s.unsqueeze(-1)
        cq = q0 + d2 * t.unsqueeze(-1)
        return (cp - cq).norm(dim=-1)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the rack pose + yaw, write the 7 kinematic rack pieces;
        scatter the chairs DEPLOYED in front (slot permutation + jitter + free yaw,
        overlap-rejected with a deterministic fallback); zero the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        torch.rand(m, 7, device=dev)  # burn: first post-seed draws are degenerate

        # --- rack pose ---
        rxy = torch.tensor(c.rack_pos, device=dev).expand(m, 2).clone()
        rxy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.rack_jitter
        ryaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rack_yaw_deg)
        self.rack_xy[env_ids] = rxy
        self.rack_yaw[env_ids] = ryaw
        cy, sy = torch.cos(ryaw), torch.sin(ryaw)

        def write_piece(body, lx, ly, lz: float) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = rxy[:, 0] + lx * cy - ly * sy
            st[:, 1] = rxy[:, 1] + lx * sy + ly * cy
            st[:, 2] = lz
            st[:, 3] = torch.cos(ryaw / 2)
            st[:, 6] = torch.sin(ryaw / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        zero = torch.zeros(m, device=dev)
        wall_z = c.plate_size[2] + c.fin_h / 2
        write_piece(self.pieces["plate"], zero, zero, c.plate_size[2] / 2)
        write_piece(self.pieces["wall_lo"], zero - c.wall_off, zero, wall_z)
        write_piece(self.pieces["wall_hi"], zero + c.wall_off, zero, wall_z)
        for k, fy in enumerate(c.fin_y):
            write_piece(self.pieces[f"fin_{k}"], zero, zero + fy, wall_z)

        # --- chairs: slot permutation + jitter + free yaw, overlap-rejected ---
        slot_perm = torch.rand(m, 3, device=dev).argsort(dim=1)
        slots = torch.tensor([-c.scatter_dy, 0.0, c.scatter_dy], device=dev)
        yaw_amp = math.radians(c.chair_yaw_deg) / 2
        # chair as a 2D capsule: segment [p - 0.030 d, p + 0.082 d], radius 0.072
        SEG_A, SEG_B, MIN_SEP = 0.030, 0.082, 0.144
        pos_i: list[torch.Tensor] = []
        yaw_i: list[torch.Tensor] = []
        ends: list[tuple[torch.Tensor, torch.Tensor]] = []
        fallback = torch.zeros(m, dtype=torch.bool, device=dev)
        for i in range(3):
            base = torch.stack([torch.full((m,), c.scatter_x, device=dev),
                                slots[slot_perm[:, i]]], dim=1)
            pos = base + (torch.rand(m, 2, device=dev) * 2 - 1) * c.scatter_jitter
            yaw = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp
            bad = torch.zeros(m, dtype=torch.bool, device=dev)
            for _try in range(12):
                d = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1)
                e0, e1 = pos - SEG_A * d, pos + SEG_B * d
                bad = torch.zeros(m, dtype=torch.bool, device=dev)
                for q0, q1 in ends:
                    bad |= self._seg_dist(e0, e1, q0, q1) < MIN_SEP
                if not bad.any():
                    break
                nb = int(bad.sum())
                pos[bad] = base[bad] + (torch.rand(nb, 2, device=dev) * 2 - 1) * c.scatter_jitter
                yaw[bad] = (torch.rand(nb, device=dev) * 2 - 1) * yaw_amp
            fallback |= bad
            ends.append((e0, e1))
            pos_i.append(pos)
            yaw_i.append(yaw)
        if fallback.any():
            # deterministic by-construction fallback: exact slots, yaw 0 (60 mm gaps)
            for i in range(3):
                pos_i[i][fallback, 0] = c.scatter_x
                pos_i[i][fallback, 1] = slots[slot_perm[fallback, i]]
                yaw_i[i][fallback] = 0.0

        hx, _hy, hz = c.hinge
        for i in range(3):
            pos, yaw = pos_i[i], yaw_i[i]
            cyi, syi = torch.cos(yaw), torch.sin(yaw)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = pos[:, 0]
            st[:, 1] = pos[:, 1]
            st[:, 2] = c.origin_h + 0.003
            st[:, 3] = torch.cos(yaw / 2)
            st[:, 6] = torch.sin(yaw / 2)
            st[:, 0:3] += origin
            self.frames[i].write_root_state_to_sim(st, env_ids)
            # seat: deployed joint pose, written CONSISTENTLY with the frame (linkage)
            ss = st.clone()
            ss[:, 0] += hx * cyi
            ss[:, 1] += hx * syi
            ss[:, 2] += hz
            self.seats[i].write_root_state_to_sim(ss, env_ids)

        self.fold_streak[env_ids] = 0.0
        self.fold_latch[env_ids] = 0.0
        self.still_count[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "frames": [b.data.root_state_w[env_ids].clone() for b in self.frames],
            "seats": [b.data.root_state_w[env_ids].clone() for b in self.seats],
            "pieces": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in self.pieces.items()},
            "rack_xy": self.rack_xy[env_ids].clone(),
            "rack_yaw": self.rack_yaw[env_ids].clone(),
            "fold_streak": self.fold_streak[env_ids].clone(),
            "fold_latch": self.fold_latch[env_ids].clone(),
            "still_count": self.still_count[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for b, st in zip(self.frames, state["frames"]):
            b.write_root_state_to_sim(st, env_ids)
        for b, st in zip(self.seats, state["seats"]):
            b.write_root_state_to_sim(st, env_ids)
        for nm, b in self.pieces.items():
            b.write_root_state_to_sim(state["pieces"][nm], env_ids)
        self.rack_xy[env_ids] = state["rack_xy"]
        self.rack_yaw[env_ids] = state["rack_yaw"]
        self.fold_streak[env_ids] = state["fold_streak"]
        self.fold_latch[env_ids] = state["fold_latch"]
        self.still_count[env_ids] = state["still_count"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A steel storage rack stands on the floor: a flat base plate "
            f"({c.plate_size[0] * 1000:.0f} x {c.plate_size[1] * 1000:.0f} mm) with two "
            "side walls and four upright fins that divide it into THREE identical "
            f"slots, each {c.slot_gap * 1000:.0f} mm wide, open only at the TOP (the "
            f"fins are {c.fin_top * 1000:.0f} mm tall). The rack's position and heading "
            "change every episode: read them by looking. Scattered on the floor in "
            "front of the rack stand three identical FOLDING chairs (red, green, "
            "blue), each a base frame (foot plate + tall backrest) with a seat panel "
            "on a real hinge. Every chair starts DEPLOYED: the seat sticks out "
            f"horizontally, making the chair {(c.deployed_reach - c.foot_x[0]) * 1000:.0f} mm "
            f"deep by {c.chair_w * 1000:.0f} mm wide — too big for a slot in ANY "
            "orientation. Folding the seat up past vertical clicks it flat against "
            f"the backrest (it rests there at ~{c.fold_stop_deg:.0f} deg and stays by "
            f"gravity), shrinking the chair to {(c.foot_x[1] - c.foot_x[0]) * 1000:.0f} mm "
            "deep — narrow enough to pass a slot with room to spare.\n"
            "Goal: put every chair away — for each chair, fold its seat fully up "
            "against the backrest, then lower the folded chair down through a slot's "
            "top opening until it stands upright on the rack floor between the fins — "
            "so that each of the three slots ends up holding exactly one folded, "
            "upright chair, and everything rests still. Any order and any "
            "chair-to-slot assignment is fine (the slots are identical).\n"
            "What does NOT count: a chair left on the floor, folded or not; a "
            "deployed chair balanced on top of the rack or wedged in a slot mouth; a "
            f"chair in a slot with its seat not fully folded (less than "
            f"{c.fold_min_deg:.0f} deg); a toppled or hanging chair; chairs stacked "
            "on one another; two chairs forced into one slot."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Fold each chair's seat up flat against its backrest, then lower the "
            "folded chair down into a rack slot until it stands upright on the rack "
            "floor — one chair per slot, all three slots filled. A chair left "
            "outside, not fully folded, perched on the rack, toppled, or sharing a "
            "slot does not count."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _frame_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(frame pos (N,C,3) env-local, frame quat (N,C,4)), world frame."""
        pos = torch.stack([b.data.root_pos_w for b in self.frames], dim=1)
        quat = torch.stack([b.data.root_quat_w for b in self.frames], dim=1)
        return pos - self.env_origins[:, None, :], quat

    def fold_deg(self) -> torch.Tensor:
        """(N,C): seat fold angle in degrees — 0 deployed, `fold_stop_deg` folded.
        Read from the RELATIVE orientation of the seat and frame bodies (the hinge
        keeps the seat +x axis in the frame's x-z plane)."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        _p, fq = self._frame_tensors()
        sq = torch.stack([b.data.root_quat_w for b in self.seats], dim=1)
        n, cc = fq.shape[0], fq.shape[1]
        ex = torch.tensor([1.0, 0.0, 0.0], device=fq.device).expand(n * cc, 3)
        a_w = quat_apply(sq.reshape(n * cc, 4), ex)
        a_f = quat_apply_inverse(fq.reshape(n * cc, 4), a_w).reshape(n, cc, 3)
        return torch.rad2deg(torch.atan2(a_f[:, :, 2], a_f[:, :, 0]))

    def chair_rack_xy(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(x_r (N,C), y_r (N,C)): chair-frame origins in the RACK frame (x along the
        slots, y across them, origin at the rack center)."""
        pos, _q = self._frame_tensors()
        d = pos[:, :, 0:2] - self.rack_xy[:, None, :]
        cy = torch.cos(self.rack_yaw)[:, None]
        sy = torch.sin(self.rack_yaw)[:, None]
        x_r = d[:, :, 0] * cy + d[:, :, 1] * sy
        y_r = -d[:, :, 0] * sy + d[:, :, 1] * cy
        return x_r, y_r

    def chair_up_z(self) -> torch.Tensor:
        """(N,C): world-z component of each chair frame's body +z axis (+1 = upright)."""
        from isaaclab.utils.math import quat_apply

        _p, quat = self._frame_tensors()
        n, cc = quat.shape[0], quat.shape[1]
        ez = torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(n * cc, 3)
        return quat_apply(quat.reshape(n * cc, 4), ez).reshape(n, cc, 3)[:, :, 2]

    def racked(self) -> torch.Tensor:
        """(N,C,S) bool, geometric: chair c counts as racked in slot s — seat folded
        NOW, origin inside the slot's x/y band (rack frame), in the on-rack-floor
        height band, upright."""
        c = self.cfg
        x_r, y_r = self.chair_rack_xy()
        pos, _q = self._frame_tensors()
        fold_ok = self.fold_deg() >= c.fold_min_deg
        x_ok = x_r.abs() < c.band_x
        z_ok = (pos[:, :, 2] > c.z_lo) & (pos[:, :, 2] < c.z_hi)
        up_ok = self.chair_up_z() >= math.cos(math.radians(c.upright_max_deg))
        ok_c = fold_ok & x_ok & z_ok & up_ok  # (N,C)
        slot_y = torch.tensor(c.slot_y, device=y_r.device)
        lat = (y_r[:, :, None] - slot_y[None, None, :]).abs() < c.band_y  # (N,C,S)
        return ok_c[:, :, None] & lat

    def _still_now(self) -> torch.Tensor:
        """(N,C) bool: chair (frame AND seat) instantaneously below the stillness
        thresholds."""
        c = self.cfg
        fv = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.frames], dim=1)
        fw = torch.stack([b.data.root_ang_vel_w.norm(dim=-1) for b in self.frames], dim=1)
        sv = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.seats], dim=1)
        sw = torch.stack([b.data.root_ang_vel_w.norm(dim=-1) for b in self.seats], dim=1)
        return (fv < c.settle_lin) & (fw < c.settle_ang) \
            & (sv < c.seat_lin) & (sw < c.settle_ang)

    def settled(self) -> torch.Tensor:
        """(N,C) bool: chair stillness SUSTAINED `settle_steps` consecutive substeps."""
        return self.still_count >= float(self.cfg.settle_steps)

    def covered(self) -> torch.Tensor:
        """(N,S) bool: slot s holds exactly one racked, settled chair. Two chairs
        cannot physically be racked in one slot (footprint arithmetic in cfg)."""
        occ = self.racked() & self.settled()[:, :, None]
        return occ.sum(dim=1) == 1

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch per-chair fold progress (seat has ever HELD the folded angle
        `fold_hold_steps` consecutive substeps) and run the sustained-stillness
        counters, every physics substep."""
        folded = self.fold_deg() >= self.cfg.fold_min_deg
        self.fold_streak = torch.where(folded, self.fold_streak + 1.0,
                                       torch.zeros_like(self.fold_streak))
        self.fold_latch = torch.maximum(
            self.fold_latch, (self.fold_streak >= float(self.cfg.fold_hold_steps)).float())
        self.still_count = torch.where(self._still_now(), self.still_count + 1.0,
                                       torch.zeros_like(self.still_count))

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: every slot holds exactly one folded, settled, upright chair."""
        return self.covered().all(dim=1)

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: per chair, 1.0 if racked NOW (some slot, settled) else
        `fold_credit` once its seat has ever held the fold (latched — credit never
        evaporates); mean over chairs; capped at 0.98 unless success(). Null policy ~0
        (chairs spawn deployed on the floor; nothing folds or moves them)."""
        racked_now = (self.racked() & self.settled()[:, :, None]).any(dim=2).float()
        per = torch.maximum(racked_now, self.cfg.fold_credit * self.fold_latch)
        mean = per.mean(dim=1)
        return torch.where(self.success(), torch.ones_like(mean), mean.clamp(max=0.98))


register_env("simgen", lambda: EnvCfg(scene="chair_foldaway", robot="null", env_spacing=3.0))

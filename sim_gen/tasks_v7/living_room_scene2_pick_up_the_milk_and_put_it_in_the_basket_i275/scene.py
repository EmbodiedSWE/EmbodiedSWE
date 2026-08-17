"""DropChuteScene — feed the milk carton into a SEALED drop-bin through its one-way
swing-door slot (sim_gen task `living_room_scene2_pick_up_the_milk_and_put_it_in_the_basket_i275`).

Derived from libero_90/living_room_scene2 "pick up the milk and put it in the basket",
but STRATEGICALLY inverted at the destination: the seed's whole delivery is a top-drop —
grasp the free-standing milk carton, carry it OVER the open basket, and release it so it
falls in from above; success is a bounding-box containment readout. Here the container
is a ROOFED bin whose top entry is physically denied: its only entrance is a low letter-
slot in the front wall, guarded by a gravity-closed swing flap that only opens INWARD.
A carton standing up cannot pass the slot (too tall) and nothing can be dropped in from
above (full roof). The solver must instead lay the WHITE milk carton on its side on the
YELLOW loading shelf outside the slot and PUSH it horizontally through the flap — a
sustained contact push against a self-closing door — until it tips over the inner sill
edge and falls to the bin floor; the flap then swings shut behind it on its own. The
load-bearing interaction is a horizontal push-through insertion (no counterpart in the
seed's grasp-carry-release-from-above), the approach direction is horizontal instead of
vertical, and the door is a passive one-way mechanism that is never held or actuated as
a separate step — it yields to the cargo itself.

Strategy vs what was read while building: the sibling task i177 (`dump_hopper`) cages
the ITEM (untouchable milk) and has the solver carry the CONTAINER and hold a lever
through a gravity discharge. This task is its inverse: the milk is directly manipulated
throughout, the container never moves, no mechanism is held — the flap is passive and
self-closing, and the denial is at the DESTINATION (sealed roof, one-way slot), not at
the item. pen_holder (exemplar) is many-object tip-up insertion into a carriable cup
with no mechanism at all. The seed itself is the open-top drop this bin makes
impossible.

success(): the milk carton rests INSIDE the bin below the slot sill (bin-frame windows;
the z window is BELOW the aperture, so a carton straddling the slot can never pass),
the flap hangs closed (also rejects a carton propping the door), the carton is settled,
and the red juice carton (decoy) is nowhere in or against the bin. score(): stateless,
monotone along the solution — 0.20 carton lying on the loading shelf, 0.55 carton
entered past the doorway plane, 0.80 carton deep inside on the bin floor, combined by
max; 1.0 iff success(). Null policy scores ~0 (both cartons spawn standing on the
floor well away from the shelf and the bin).

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - bin (dynamic, 30 kg — heavy DYNAMIC fixture, never kinematic, so the flap hinge's
    anchor follows the reset teleport): floor plate, back and side walls, FULL roof,
    and a front wall made of a sill panel (up to the slot bottom), two pillars, and a
    lintel — leaving a 120 mm tall x 200 mm wide letter slot. A YELLOW slick loading
    shelf sticks out under the slot at exactly sill height, with a support leg.
  - flap (dynamic, 60 g, authored CoM + diagonal inertia): a thin BLUE door plate
    hanging from a RevoluteJoint just under the lintel, inboard of the front wall.
    Limits [-1 deg, +80 deg]: it can swing INWARD (positive, pushed by cargo) but not
    outward — a one-way door. The authored CoM sits below the hinge so gravity returns
    it shut on its own. Joint-pair collision stays FILTERED (the USD default): the
    flap may sweep past the bin shell; its travel is bounded by the joint limits.
  - milk carton (target): 60 x 60 x 160 mm white box, 350 g, spawned STANDING on the
    floor away from the bin.
  - juice carton (decoy): identical red box standing on the floor on the mirrored
    side; it must stay out of / off the bin.
Per-episode randomization (readback-verified in smoke): bin xy jitter + FREE yaw, milk
and juice on mirrored jittered arcs around the bin (side, bearing, radius, free yaw).
Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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


# ----- geometry constants (single source of truth: spawners + cfg asserts + rubric) ------------
BIN_HX = 0.180  # bin outer half-extent, x (+x = out through the slot)
BIN_HY = 0.180  # bin outer half-extent, y
BIN_H = 0.320  # bin outer height (roof top)
WALL_T = 0.010
FLOOR_T = 0.010  # bin floor plate (interior floor top = z 0.010)
ROOF_T = 0.010
SILL = 0.120  # slot bottom = sill-panel top = shelf top (bin frame z)
SLOT_TOP = 0.240  # slot top = lintel underside
SLOT_HW = 0.100  # slot half-width (y)
SHELF_X0 = BIN_HX  # loading shelf x extent (bin frame): 0.180 .. 0.380
SHELF_X1 = 0.380
SHELF_HW = 0.120  # shelf half-width (y)
SHELF_T = 0.030  # shelf slab thickness (top at SILL)
FLAP_HINGE_X = 0.160  # hinge line, inboard of the front wall inner face (0.170)
FLAP_HINGE_Z = 0.238  # hinge line height (just under the lintel underside)
FLAP_LEN = 0.127  # flap length below the hinge (closed bottom edge at z 0.111)
FLAP_W = 0.190  # flap width (y) — covers the slot
FLAP_T = 0.006
FLAP_MIN_DEG = -1.0  # one-way: no outward swing
FLAP_MAX_DEG = 80.0  # inward travel stop
MILK_W = 0.060  # carton square cross-section
MILK_H = 0.160  # carton long dimension


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (N,3) by quaternions q (N,4), wxyz."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _qinv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[:, 1:] = -out[:, 1:]
    return out


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


# ----- custom compound spawners ----------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _friction_material(stage, path: str, static: float, dynamic: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, size, center, color, contact_offset: float,
         material=None) -> None:
    """Author one box child prim (translate -> scale, authored once — idempotent per
    prim, the duplicate-xformOp trap)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_bin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the drop-bin as ONE heavy DYNAMIC compound body (never kinematic: the
    flap hinge's anchor must follow the reset teleport — a kinematic body0's anchor
    stays world-fixed at the spawn pose). Origin = base center on the ground;
    +x = out through the slot, over the loading shelf."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    # ZERO sleep/stabilization: a sleeping bin would freeze the flap-hinge anchor.
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    grip = _friction_material(stage, f"{prim_path}/grip_mat", cfg.mu_body_s, cfg.mu_body_d)
    slick = _friction_material(stage, f"{prim_path}/slick_mat", cfg.mu_slide_s, cfg.mu_slide_d)
    gray = (0.45, 0.48, 0.50)
    dark = (0.30, 0.32, 0.35)
    yellow = (0.95, 0.85, 0.10)
    wall_h = BIN_H - FLOOR_T - ROOF_T  # 0.300, walls from z 0.010 to 0.310
    wall_cz = FLOOR_T + wall_h / 2  # 0.160
    # floor plate (interior floor top = z 0.010) — grippy: the delivered carton parks
    _box(stage, f"{prim_path}/floor", (2 * BIN_HX, 2 * BIN_HY, FLOOR_T),
         (0.0, 0.0, FLOOR_T / 2), dark, 0.0015, material=grip)
    # back wall (inner face x = -0.170)
    _box(stage, f"{prim_path}/back", (WALL_T, 2 * BIN_HY, wall_h),
         (-(BIN_HX - WALL_T / 2), 0.0, wall_cz), gray, 0.0015, material=grip)
    # side walls (inner faces |y| = 0.170)
    for tag, sy in (("l", 1.0), ("r", -1.0)):
        _box(stage, f"{prim_path}/wall_{tag}", (2 * BIN_HX, WALL_T, wall_h),
             (0.0, sy * (BIN_HY - WALL_T / 2), wall_cz), gray, 0.0015, material=grip)
    # FULL roof — top entry is denied by construction
    _box(stage, f"{prim_path}/roof", (2 * BIN_HX, 2 * BIN_HY, ROOF_T),
         (0.0, 0.0, BIN_H - ROOF_T / 2), gray, 0.0015, material=grip)
    # front wall: sill panel below the slot (top = SILL, slick — the carton slides
    # across it), pillars flanking the slot, lintel above it
    _box(stage, f"{prim_path}/sill", (WALL_T, 2 * BIN_HY, SILL - FLOOR_T),
         (BIN_HX - WALL_T / 2, 0.0, FLOOR_T + (SILL - FLOOR_T) / 2),
         gray, 0.0015, material=slick)
    for tag, sy in (("l", 1.0), ("r", -1.0)):
        _box(stage, f"{prim_path}/pillar_{tag}",
             (WALL_T, BIN_HY - SLOT_HW, SLOT_TOP - SILL),
             (BIN_HX - WALL_T / 2, sy * (SLOT_HW + (BIN_HY - SLOT_HW) / 2),
              SILL + (SLOT_TOP - SILL) / 2), gray, 0.0015, material=slick)
    _box(stage, f"{prim_path}/lintel", (WALL_T, 2 * BIN_HY, BIN_H - ROOF_T - SLOT_TOP),
         (BIN_HX - WALL_T / 2, 0.0, SLOT_TOP + (BIN_H - ROOF_T - SLOT_TOP) / 2),
         gray, 0.0015, material=slick)
    # YELLOW loading shelf outside the slot, top flush with the sill — slick, so a
    # light push slides the carton; support leg at the outer end
    _box(stage, f"{prim_path}/shelf", (SHELF_X1 - SHELF_X0, 2 * SHELF_HW, SHELF_T),
         ((SHELF_X0 + SHELF_X1) / 2, 0.0, SILL - SHELF_T / 2),
         yellow, 0.0015, material=slick)
    _box(stage, f"{prim_path}/shelf_leg", (0.030, 2 * SHELF_HW, SILL - SHELF_T),
         (SHELF_X1 - 0.015, 0.0, (SILL - SHELF_T) / 2), dark, 0.0015, material=grip)
    return root


def _spawn_flap(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the one-way swing flap: a thin BLUE plate hanging from a RevoluteJoint
    into the sibling bin, with AUTHORED CoM below the hinge (gravity self-close) and
    diagonal inertia. Origin = the hinge line; the plate is the child at -z."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass_props.mass))
    # Authored CoM (MassAPI mass alone would pin the CoM at the body origin = the
    # hinge, killing the gravity self-close) + diagonal inertia (plate estimates).
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, -float(FLAP_LEN) / 2))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(2.6e-4, 8.5e-5, 1.8e-4))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(1)
    pxrb.CreateAngularDampingAttr(1.5)  # soften the swing shut (no clatter)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    body = _friction_material(stage, f"{prim_path}/body_mat", cfg.mu_flap_s, cfg.mu_flap_d)
    blue = (0.15, 0.35, 0.90)
    _box(stage, f"{prim_path}/plate", (FLAP_T, FLAP_W, FLAP_LEN),
         (0.0, 0.0, -FLAP_LEN / 2), blue, 0.0015, material=body)

    # revolute hinge to the sibling bin, axis = bin y, on the hinge line.
    # Positive rotation about +y swings the plate bottom toward -x = INTO the bin.
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([f"{base}/Bin"])
    j.CreateBody1Rel().SetTargets([prim_path])
    # joint-pair collision stays FILTERED (the USD default): the flap may sweep past
    # the bin shell; its travel is bounded by the joint limits alone.
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(float(FLAP_HINGE_X), 0.0, float(FLAP_HINGE_Z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(float(FLAP_MIN_DEG))  # one-way: no outward swing
    j.CreateUpperLimitAttr(float(FLAP_MAX_DEG))  # inward stop
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (explicit @configclass subclasses
    of RigidObjectSpawnerCfg, defined lazily so the module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "bin" not in _SPAWNER_CACHE:

        @configclass
        class BinSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bin)
            mu_body_s: float = 0.45
            mu_body_d: float = 0.40
            mu_slide_s: float = 0.15
            mu_slide_d: float = 0.12

        @configclass
        class FlapSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_flap)
            mu_flap_s: float = 0.20
            mu_flap_d: float = 0.15

        _SPAWNER_CACHE.update(bin=BinSpawnerCfg, flap=FlapSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class DropChuteSceneCfg(BaseCfg):
    """Config for `DropChuteScene`. The honesty knobs are asserted in `__post_init__`:
    a LYING carton fits through the slot with margin while a STANDING one physically
    cannot, the roof seals top entry, the flap genuinely covers the slot yet clears a
    delivered carton and self-closes under authored gravity bias, the push force sits
    in a fingertip band, the "inside" z window lies BELOW the aperture (a slot-
    straddling carton can never pass), and both cartons spawn beyond the shelf and the
    bin so the null policy scores ~0."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    in_x_max: float = tunable(0.160)  # |x| of the carton center, bin frame
    in_y_max: float = tunable(0.145)
    in_z_win: tuple = tunable((0.013, 0.095))  # BELOW the sill (aperture) by >= 20 mm
    flap_closed_deg: float = tunable(15.0)  # door shut (also: nothing propping it)
    settle_lin: float = tunable(0.05)  # settle gates (m/s, rad/s)
    settle_ang: float = tunable(1.0)
    shelf_x_win: tuple = tunable((0.150, 0.400))  # staged-on-shelf window (bin frame)
    shelf_y_max: float = tunable(0.115)
    shelf_z_win: tuple = tunable((0.130, 0.215))  # excludes floor-standing cartons
    ent_x_max: float = tunable(0.165)  # entered-past-the-doorway box
    ent_y_max: float = tunable(0.160)
    ent_z_win: tuple = tunable((0.005, 0.260))
    decoy_x_max: float = tunable(0.185)  # decoy-in/at-the-bin violation box
    decoy_y_max: float = tunable(0.175)
    decoy_z_win: tuple = tunable((-0.020, 0.330))

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    bin_jitter: float = tunable(0.05)  # uniform +/- xy jitter of the bin (m)
    bin_yaw_max: float = tunable(180.0)  # uniform +/- bin yaw (deg; FREE heading)
    arc_bearing: tuple = tunable((35.0, 75.0))  # carton bearing band off bin +x (deg)
    arc_radius: tuple = tunable((0.50, 0.62))  # ... radius band around the bin (m)

    # --- info: structure -----------------------------------------------------------------------
    milk_w: float = info(MILK_W)
    milk_h: float = info(MILK_H)
    milk_mass: float = info(0.35)
    juice_mass: float = info(0.30)
    flap_mass: float = info(0.06)
    bin_mass: float = info(30.0)  # heavy dynamic fixture (hinge anchor follows teleports)
    mu_milk_s: float = info(0.30)
    mu_milk_d: float = info(0.25)
    mu_slide_s: float = info(0.15)  # shelf + sill + doorway (the carton must slide in)
    mu_slide_d: float = info(0.12)
    mu_body_s: float = info(0.45)  # bin interior (the delivered carton parks)
    mu_body_d: float = info(0.40)
    mu_flap_s: float = info(0.20)
    mu_flap_d: float = info(0.15)
    mu_ground_s: float = info(0.60)
    mu_ground_d: float = info(0.50)

    def __post_init__(self) -> None:
        # -- the slot is the bin's only entrance, and only for a LYING carton ------------------
        assert SLOT_TOP - SILL >= MILK_W + 0.040, "a LYING carton must pass the slot with margin"
        assert SILL + MILK_H >= SLOT_TOP + 0.030, "a STANDING carton must NOT pass the slot"
        assert 2 * SLOT_HW >= MILK_W + 0.100, "slot wide enough for a yawed lying carton"
        assert 2 * (BIN_HY - WALL_T) >= math.hypot(MILK_W, MILK_H) + 0.060, \
            "bin interior must swallow a tumbled carton with margin"
        # -- the flap covers the slot, clears the delivered carton, and can ride over cargo ----
        flap_bot = FLAP_HINGE_Z - FLAP_LEN
        assert FLAP_W >= 2 * SLOT_HW - 0.020, "flap must cover the slot width"
        assert flap_bot <= SILL - 0.005, "closed flap must reach below the sill (seals the slot)"
        assert flap_bot >= FLOOR_T + MILK_W + 0.030, \
            "closed flap must hang clear of a carton lying on the bin floor"
        top_at_max = FLAP_HINGE_Z - FLAP_LEN * math.cos(math.radians(FLAP_MAX_DEG))
        assert top_at_max >= SILL + MILK_W + 0.020, \
            "at full travel the flap must ride clear over a lying carton crossing the sill"
        # -- gravity self-close is real (authored CoM below the hinge) -------------------------
        bias = self.flap_mass * 9.81 * (FLAP_LEN / 2)
        assert 0.010 <= bias <= 0.200, f"flap gravity-close bias {bias:.3f} N*m out of band"
        # -- the push is a fingertip force -----------------------------------------------------
        mu_comb = (self.mu_slide_s + self.mu_milk_s) / 2  # PhysX default combine: average
        f_push = mu_comb * self.milk_mass * 9.81 + bias / max(FLAP_HINGE_Z - SILL - MILK_W / 2, 0.02)
        assert 0.3 <= f_push <= 5.0, f"push force {f_push:.2f} N out of the fingertip band"
        # -- "inside" means BELOW the aperture: a slot-straddler can never pass ----------------
        assert self.in_z_win[1] <= SILL - 0.020, "inside z window must sit below the sill"
        straddle_z = SILL + MILK_W / 2  # CoM of a carton lying across the sill
        assert straddle_z - self.in_z_win[1] >= 0.045, \
            "slot-straddling carton CoM must sit well above the inside z window"
        assert self.in_z_win[0] < FLOOR_T + MILK_W / 2 < self.in_z_win[1], \
            "a carton lying on the bin floor must be inside"
        assert self.in_z_win[0] < FLOOR_T + MILK_H / 2 < self.in_z_win[1], \
            "a carton standing on the bin floor must be inside"
        assert self.in_x_max <= BIN_HX - WALL_T - 0.005
        assert self.in_y_max <= BIN_HY - WALL_T - 0.020
        # -- shelf credit is real staging, not floor-standing ----------------------------------
        assert SHELF_X1 - SHELF_X0 >= MILK_H + 0.030, "shelf long enough to lay the carton on"
        assert 2 * SHELF_HW >= MILK_H + 0.040, "shelf wide enough for a yawed lying carton"
        assert self.shelf_z_win[0] >= MILK_H / 2 + 0.023, \
            "shelf z window must exclude a carton standing on the ground"
        assert self.shelf_z_win[0] <= SILL + MILK_W / 2 - 0.005 <= self.shelf_z_win[1], \
            "a carton lying on the shelf must be in the shelf window"
        # -- entered box: inside the doorway plane, disjoint from shelf rests ------------------
        assert self.ent_x_max < SHELF_X0 - 0.010, "entered box must stop short of the shelf"
        assert self.ent_x_max >= self.shelf_x_win[0], \
            "entered and shelf boxes must overlap so the score is monotone through the door"
        # -- null policy scores 0: cartons spawn beyond the shelf and the bin ------------------
        y_min = self.arc_radius[0] * math.sin(math.radians(self.arc_bearing[0]))
        assert y_min > BIN_HY + 0.080, "spawn arc must start well clear of the bin sides"
        assert self.arc_radius[0] > SHELF_X1 + math.hypot(MILK_W, MILK_H) / 2 + 0.020, \
            "spawn arc must start beyond the shelf tip"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("drop_chute")
class DropChuteScene(BaseScene):
    cfg: DropChuteSceneCfg

    def __init__(self, cfg: DropChuteSceneCfg | None = None) -> None:
        super().__init__(cfg or DropChuteSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        carton_rigid = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            max_depenetration_velocity=0.5, sleep_threshold=0.0,
            stabilization_threshold=0.0, linear_damping=0.05, angular_damping=0.1)
        carton_coll = sim_utils.CollisionPropertiesCfg(contact_offset=0.0015, rest_offset=0.0)
        carton_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.mu_milk_s, dynamic_friction=c.mu_milk_d, restitution=0.0)

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_ground_s, dynamic_friction=c.mu_ground_d,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            # NOTE: the bin MUST spawn before the flap (the hinge targets it).
            "bin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bin",
                spawn=spawners["bin"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bin_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mu_body_s=c.mu_body_s, mu_body_d=c.mu_body_d,
                    mu_slide_s=c.mu_slide_s, mu_slide_d=c.mu_slide_d),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.003)),
            ),
            "flap": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Flap",
                spawn=spawners["flap"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.flap_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mu_flap_s=c.mu_flap_s, mu_flap_d=c.mu_flap_d),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(FLAP_HINGE_X, 0.0, FLAP_HINGE_Z + 0.003)),
            ),
            "milk": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Milk",
                spawn=sim_utils.CuboidCfg(
                    size=(c.milk_w, c.milk_w, c.milk_h),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.95, 0.95, 0.97)),
                    physics_material=carton_mat, rigid_props=carton_rigid,
                    collision_props=carton_coll,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.milk_mass)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.55, 0.30, c.milk_h / 2 + 0.003)),
            ),
            "juice": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Juice",
                spawn=sim_utils.CuboidCfg(
                    size=(c.milk_w, c.milk_w, c.milk_h),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.85, 0.12, 0.12)),
                    physics_material=carton_mat, rigid_props=carton_rigid,
                    collision_props=carton_coll,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.juice_mass)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.55, -0.30, c.milk_h / 2 + 0.003)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # External-wrench push plant: without this the body-frame push force
                # is under-applied across TGS iterations and the insertion stalls.
                "enable_external_forces_every_iteration": True,
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
        self.bin: RigidObject = env.iscene["bin"]
        self.flap: RigidObject = env.iscene["flap"]
        self.milk: RigidObject = env.iscene["milk"]
        self.juice: RigidObject = env.iscene["juice"]
        self.env_origins = env.iscene.env_origins

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter + free-yaw the bin (heavy DYNAMIC teleport — the flap
        hinge anchor follows), write the flap CONSISTENTLY shut on its hinge line in
        the bin's new frame, and stand the milk and juice cartons on mirrored jittered
        arcs around the bin (side, bearing, radius, free yaw)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        # --- bin: xy jitter + free yaw (write the WHOLE linkage together) ---
        bxy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.bin_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.bin_yaw_max)
        q_bin = _qz(yaw)
        z0 = torch.full((m, 1), 0.003, device=dev)
        bin_pos = torch.cat([bxy, z0], dim=-1)
        write(self.bin, bin_pos, q_bin)

        # --- flap: SHUT on its hinge line, in the bin's new frame ---
        hinge_local = torch.tensor([FLAP_HINGE_X, 0.0, FLAP_HINGE_Z], device=dev).expand(m, 3)
        write(self.flap, bin_pos + _qapply(q_bin, hinge_local), q_bin)

        # --- milk + juice: STANDING on mirrored jittered arcs around the bin ---
        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           torch.tensor(-1.0, device=dev), torch.tensor(1.0, device=dev))
        b0, b1 = (math.radians(v) for v in c.arc_bearing)
        r0, r1 = c.arc_radius
        for body, sgn in ((self.milk, side), (self.juice, -side)):
            bear = (b0 + torch.rand(m, device=dev) * (b1 - b0)) * sgn
            rad = r0 + torch.rand(m, device=dev) * (r1 - r0)
            local = torch.stack([rad * torch.cos(bear), rad * torch.sin(bear),
                                 torch.full((m,), MILK_H / 2 + 0.003, device=dev)], dim=-1)
            byaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            write(body, bin_pos + _qapply(q_bin, local), _qmul(q_bin, _qz(byaw)))

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {nm: getattr(self, nm).data.root_state_w[env_ids].clone()
                for nm in ("bin", "flap", "milk", "juice")}

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm in ("bin", "flap", "milk", "juice"):
            getattr(self, nm).write_root_state_to_sim(state[nm], env_ids)

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A sealed gray drop-bin ({2 * BIN_HX * 100:.0f} x {2 * BIN_HY * 100:.0f} cm, "
            f"{BIN_H * 100:.0f} cm tall) stands on the floor. Its top is a FULL roof — "
            f"nothing can be dropped in from above. Its only entrance is a letter slot "
            f"in the front wall ({(SLOT_TOP - SILL) * 100:.0f} cm tall, "
            f"{2 * SLOT_HW * 100:.0f} cm wide, bottom edge {SILL * 100:.0f} cm up), "
            f"guarded on the inside by a BLUE swing flap hanging from a hinge under "
            f"the slot's top edge. The flap only swings INWARD — push cargo against "
            f"it and it yields, then falls shut again on its own; it cannot swing "
            f"outward, so nothing comes back out. A YELLOW loading shelf sticks out "
            f"under the slot, its top exactly flush with the slot's bottom edge. A "
            f"WHITE milk carton ({c.milk_w * 100:.0f} x {c.milk_w * 100:.0f} x "
            f"{c.milk_h * 100:.0f} cm) and a RED juice carton stand on the floor on "
            f"opposite sides of the bin, out of reach of the shelf. A carton standing "
            f"upright is TALLER than the slot: it only fits through lying on its "
            f"side. The bin's position and heading and both cartons' poses change "
            f"every episode — read the scene by looking.\n"
            f"Goal: get the WHITE milk carton inside the bin. Pick it up, lay it on "
            f"its side on the YELLOW shelf pointing at the slot, and push it "
            f"horizontally through the BLUE flap until it tips over the inner edge "
            f"and drops to the bin floor; the flap swings shut behind it. Finish "
            f"with the milk resting on the bin floor, the flap hanging closed, and "
            f"everything at rest. The RED juice carton must stay out: any part of it "
            f"in the doorway, against the bin, or inside fails the task."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lay the white milk carton on its side on the yellow loading shelf and "
            "push it through the blue one-way flap into the sealed bin, so it drops "
            "to the bin floor and the flap swings shut behind it. Leave the milk "
            "resting inside and keep the red juice carton away from the bin."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def bin_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> bin frame (origin = base center on the ground)."""
        return _qapply(_qinv(self.bin.data.root_quat_w),
                       pos_w - self.bin.data.root_pos_w)

    def flap_deg(self) -> torch.Tensor:
        """(N,) flap swing angle in degrees (positive = swung INWARD), read back from
        the flap's orientation relative to the bin about the hinge (bin y) axis."""
        rel = _qmul(_qinv(self.bin.data.root_quat_w), self.flap.data.root_quat_w)
        return torch.rad2deg(2.0 * torch.atan2(rel[:, 2], rel[:, 0]))

    def flap_closed(self) -> torch.Tensor:
        """(N,) bool: the flap hangs shut (nothing propping the doorway)."""
        return self.flap_deg().abs() <= self.cfg.flap_closed_deg

    def milk_inside(self) -> torch.Tensor:
        """(N,) bool: carton center inside the bin, BELOW the slot sill."""
        c = self.cfg
        loc = self.bin_local(self.milk.data.root_pos_w)
        return (loc[:, 0].abs() <= c.in_x_max) & (loc[:, 1].abs() <= c.in_y_max) \
            & (loc[:, 2] >= c.in_z_win[0]) & (loc[:, 2] <= c.in_z_win[1])

    def milk_on_shelf(self) -> torch.Tensor:
        """(N,) bool: carton lying on the loading shelf (staged for the push)."""
        c = self.cfg
        loc = self.bin_local(self.milk.data.root_pos_w)
        return (loc[:, 0] >= c.shelf_x_win[0]) & (loc[:, 0] <= c.shelf_x_win[1]) \
            & (loc[:, 1].abs() <= c.shelf_y_max) \
            & (loc[:, 2] >= c.shelf_z_win[0]) & (loc[:, 2] <= c.shelf_z_win[1])

    def milk_entered(self) -> torch.Tensor:
        """(N,) bool: carton center past the doorway plane, within the bin footprint."""
        c = self.cfg
        loc = self.bin_local(self.milk.data.root_pos_w)
        return (loc[:, 0].abs() <= c.ent_x_max) & (loc[:, 1].abs() <= c.ent_y_max) \
            & (loc[:, 2] >= c.ent_z_win[0]) & (loc[:, 2] <= c.ent_z_win[1])

    def decoy_out(self) -> torch.Tensor:
        """(N,) bool: the red juice carton is nowhere in / at the bin or its doorway."""
        c = self.cfg
        loc = self.bin_local(self.juice.data.root_pos_w)
        inside = (loc[:, 0].abs() <= c.decoy_x_max) & (loc[:, 1].abs() <= c.decoy_y_max) \
            & (loc[:, 2] >= c.decoy_z_win[0]) & (loc[:, 2] <= c.decoy_z_win[1])
        return ~inside

    def settled(self, body) -> torch.Tensor:
        return (body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang)

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: milk carton resting inside the bin below the slot sill, flap
        hanging closed, carton settled, decoy nowhere in / at the bin."""
        return self.milk_inside() & self.flap_closed() & self.settled(self.milk) \
            & self.decoy_out()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: max of 0.20 (carton staged lying on the shelf), 0.55
        (carton entered past the doorway plane), 0.80 (carton deep inside, below the
        sill); 1.0 iff success(). Stateless and monotone along the intended solution
        (stage -> push through -> drop -> flap shuts -> settle); the null policy
        scores ~0 (both cartons spawn standing on the floor beyond the shelf)."""
        shelf = self.milk_on_shelf().float()
        ent = self.milk_entered().float()
        deep = self.milk_inside().float()
        base = torch.maximum(0.20 * shelf, torch.maximum(0.55 * ent, 0.80 * deep))
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="drop_chute", robot="null", env_spacing=3.0))

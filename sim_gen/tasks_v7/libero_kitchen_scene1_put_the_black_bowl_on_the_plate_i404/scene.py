"""CaliberVaultScene — sort two marbles into a roofed vault through the only aperture
each one fits (libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i404).

Derived from libero_90 kitchen_scene1 "put the black bowl on the plate", where the whole
task is one blind pick-and-place of a payload onto an open, passive goal surface.  Here
the goal region is a SEALED VAULT and the two payloads are marbles of different caliber
whose ROUTES INTO IT are decided by size interference, not by the hand:

  - The vault is fully roofed (canopy) and walled: nothing can be placed into it from
    above.  It has exactly two ways in.
  - Route 1 — the RAIL RUN: two horizontally DIVERGING rails enter under the canopy.
    Their gap starts at 62 mm and widens linearly to 92 mm.  The BIG marble (d=60)
    bridges the gap and rides the rails; because the gap widens, its center height
    h(x) = sqrt((R+rr)^2 - (s(x)/2)^2) falls covertly, so it self-accelerates on
    visually level rails and DROPS THROUGH the gap right over the vault (geometric
    release where s = 2(R+rr), x ~ 0.32, inside the vault span).  The SMALL marble
    (d=32) never bridges the gap (62 > 2(r+rr) = 56): dropped on the rails it falls
    straight through onto a return ramp that rolls it back OUT of the structure.
  - Route 2 — the SIDE PORT: a 44 x 44 mm doorway low in the vault's side wall, fed by
    an external open-top runway lane (inner width 45 mm) with a 10 mm drop-in sill.
    The small marble rolls down the lane and through the port; the big marble does not
    even fit INTO the lane (45 < 60) and is refused by the port.

  Goal: BOTH marbles at rest inside the vault.

STRATEGIC DIFFERENCE from the seed: the seed's plan — carry the payload to the goal and
set it down — is physically impossible here: the goal surface is under a roof.  Success
requires routing each object through the one size-gated machine path that admits it
(ride-and-release for the big, lane-and-port for the small), i.e. matching objects to
apertures and letting the structure's physics finish the delivery.  Nothing here is a
container-content relation (i281), a hidden-mass measurement (i187), or an uncover/
re-cover ordering (i221); the deliverable states of the seed and siblings are rejected
states or unreachable states of this scene.

Everything is procedural: one KINEMATIC compound rig (rails, canopy, fascia, ramp,
bulkhead, vault, port, lane) so the whole structure can be re-posed per reset (xy
jitter + full yaw), plus two free spheres.  No joints, no hidden state.

Mechanism notes (proven corpus cribs):
  - Custom spawners silently ignore cfg mass_props / rigid_props — the rig authors
    kinematicEnabled and MassAPI mass inside the spawn func; smoke asserts readback.
  - All predicates are computed in the RIG BODY FRAME via quat_apply_inverse, so the
    reset-time yaw randomization is load-bearing, not cosmetic.
  - Latched progress (ride / canopy / vault / lane) updates in post_step every step —
    latched credit never evaporates; settle gates sit at 0.06 m/s, above the GPU
    phantom-creep band.
  - Geometry honesty is asserted at import time in __post_init__ (pass/never-pass
    margins, release point inside the vault, rider clearances) — the interference
    arithmetic is checked before any GPU minute is spent.
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


# ----- custom compound spawner (the whole rig is ONE kinematic rigid body) -------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_rig(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One KINEMATIC rigid body: the caliber-vault rig.  Rig frame: +x along the rails
    from the open loading bay toward the vault, +z up, floor at local z = 0.

    Children: two diverging rail capsules, canopy slab, entrance fascia shoulders,
    return ramp + curbs, bulkhead (center wall + shoulders), vault floor/walls with the
    side port cut into the +y wall (sill + header), and the external runway lane
    (floor, two guide walls, end stop)."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    # rigid body: KINEMATIC (cfg rigid_props is ignored for custom funcs — author here)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        pc = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        pc.CreateContactOffsetAttr(float(cfg.contact_offset))
        pc.CreateRestOffsetAttr(0.0)

    def box(name: str, center, dims, color) -> None:
        cube = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        cube.CreateSizeAttr(1.0)
        bxf = UsdGeom.Xformable(cube.GetPrim())
        bxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        bxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in dims]))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        collide(cube.GetPrim())

    steel = (0.62, 0.64, 0.68)
    slate = (0.35, 0.38, 0.44)
    roofc = (0.16, 0.22, 0.42)
    amber = (0.80, 0.58, 0.18)
    rampc = (0.45, 0.42, 0.38)

    # --- rails: two capsules diverging in yaw about the rig x-axis ------------------------------
    phi = math.atan(cfg.rail_half_slope)
    span = cfg.rail_len_x / math.cos(phi)
    y_mid = cfg.rail_s0 / 2 + cfg.rail_half_slope * (cfg.rail_len_x / 2)
    for sgn, name in ((1.0, "rail_p"), (-1.0, "rail_m")):
        cap = UsdGeom.Capsule.Define(stage, f"{prim_path}/{name}")
        cap.CreateAxisAttr("X")
        cap.CreateRadiusAttr(float(cfg.rail_r))
        cap.CreateHeightAttr(float(span))
        e = span / 2 + cfg.rail_r
        cap.CreateExtentAttr([Gf.Vec3f(-e, -cfg.rail_r, -cfg.rail_r),
                              Gf.Vec3f(e, cfg.rail_r, cfg.rail_r)])
        rxf = UsdGeom.Xformable(cap.GetPrim())
        rxf.AddTranslateOp().Set(Gf.Vec3d(cfg.rail_len_x / 2, sgn * y_mid, cfg.rail_z))
        rxf.AddRotateZOp().Set(sgn * math.degrees(phi))
        cap.CreateDisplayColorAttr([Gf.Vec3f(*steel)])
        collide(cap.GetPrim())

    # --- canopy (roofs the whole run + vault; the loading bay x<canopy_x0 is open sky) ----------
    cx0, cx1 = cfg.canopy_x0, cfg.wall_far_x1
    box("canopy", ((cx0 + cx1) / 2, 0.0, cfg.canopy_z + 0.005),
        (cx1 - cx0, 2 * cfg.side_half_w_out, 0.010), roofc)

    # --- entrance fascia: shoulders leaving a center slot for the riding big marble -------------
    fy = (cfg.fascia_slot_half_w + cfg.side_half_w_out) / 2
    fw = cfg.side_half_w_out - cfg.fascia_slot_half_w
    fz = (cfg.fascia_z0 + cfg.canopy_z) / 2
    fh = cfg.canopy_z - cfg.fascia_z0
    box("fascia_p", (0.005, fy, fz), (0.020, fw, fh), slate)
    box("fascia_m", (0.005, -fy, fz), (0.020, fw, fh), slate)

    # --- return ramp under the rails (slopes out the open end) + curbs --------------------------
    rx0, rx1 = cfg.ramp_x0, cfg.bulk_x0
    r_len = rx1 - rx0
    pitch = math.degrees(math.atan((cfg.ramp_z_hi - cfg.ramp_z_lo) / r_len))
    ramp = UsdGeom.Cube.Define(stage, f"{prim_path}/ramp")
    ramp.CreateSizeAttr(1.0)
    rxf = UsdGeom.Xformable(ramp.GetPrim())
    rxf.AddTranslateOp().Set(Gf.Vec3d((rx0 + rx1) / 2, 0.0,
                                      (cfg.ramp_z_hi + cfg.ramp_z_lo) / 2 - 0.004))
    rxf.AddRotateYOp().Set(-pitch)  # +x end raised: surface drains toward the open end
    rxf.AddScaleOp().Set(Gf.Vec3f(r_len / math.cos(math.radians(pitch)), 0.190, 0.008))
    ramp.CreateDisplayColorAttr([Gf.Vec3f(*rampc)])
    collide(ramp.GetPrim())
    box("curb_p", ((rx0 + rx1) / 2, 0.1015, 0.042), (r_len, 0.013, 0.076), rampc)
    box("curb_m", ((rx0 + rx1) / 2, -0.1015, 0.042), (r_len, 0.013, 0.076), rampc)

    # --- bulkhead: low center wall the rider clears + full-height shoulders ---------------------
    bx = (cfg.bulk_x0 + cfg.bulk_x1) / 2
    bt = cfg.bulk_x1 - cfg.bulk_x0
    box("bulk_c", (bx, 0.0, cfg.bulk_top / 2), (bt, 2 * cfg.bulk_notch_half_w, cfg.bulk_top),
        slate)
    sy = (cfg.bulk_notch_half_w + cfg.side_half_w_out) / 2
    sw = cfg.side_half_w_out - cfg.bulk_notch_half_w
    box("bulk_p", (bx, sy, cfg.canopy_z / 2), (bt, sw, cfg.canopy_z), slate)
    box("bulk_m", (bx, -sy, cfg.canopy_z / 2), (bt, sw, cfg.canopy_z), slate)

    # --- vault: floor, far wall, -y wall, +y wall with the port cut -----------------------------
    vx0, vx1 = cfg.bulk_x0, cfg.wall_far_x1
    box("vfloor", ((vx0 + vx1) / 2, 0.0, cfg.vault_floor_top / 2),
        (vx1 - vx0, 2 * cfg.side_half_w_in + 0.020, cfg.vault_floor_top), slate)
    box("vwall_far", ((cfg.wall_far_x0 + vx1) / 2, 0.0, cfg.canopy_z / 2),
        (vx1 - cfg.wall_far_x0, 2 * cfg.side_half_w_out, cfg.canopy_z), slate)
    wy = (cfg.side_half_w_in + cfg.side_half_w_out) / 2
    wt = cfg.side_half_w_out - cfg.side_half_w_in
    box("vwall_ny", ((vx0 + vx1) / 2, -wy, cfg.canopy_z / 2),
        (vx1 - vx0, wt, cfg.canopy_z), slate)
    px0, px1 = cfg.port_x0, cfg.port_x1
    box("vwall_py_a", ((vx0 + px0) / 2, wy, cfg.canopy_z / 2), (px0 - vx0, wt, cfg.canopy_z),
        slate)
    box("vwall_py_b", ((px1 + vx1) / 2, wy, cfg.canopy_z / 2), (vx1 - px1, wt, cfg.canopy_z),
        slate)
    box("vwall_sill", ((px0 + px1) / 2, wy, cfg.lane_floor_top / 2),
        (px1 - px0, wt, cfg.lane_floor_top), amber)
    hz = (cfg.port_z1 + cfg.canopy_z) / 2
    box("vwall_hdr", ((px0 + px1) / 2, wy, hz), (px1 - px0, wt, cfg.canopy_z - cfg.port_z1),
        slate)

    # --- runway lane: floor + guide walls + end stop (open top) ---------------------------------
    lx = (px0 + px1) / 2
    ly0, ly1 = cfg.side_half_w_in + 0.004, cfg.lane_y_end
    lw_out = cfg.lane_half_w + cfg.lane_wall_t
    box("lane_floor", (lx, (ly0 + ly1) / 2, cfg.lane_floor_top - 0.004),
        (2 * lw_out, ly1 - ly0, 0.008), amber)
    for sgn, name in ((1.0, "lane_wall_a"), (-1.0, "lane_wall_b")):
        box(name, (lx + sgn * (cfg.lane_half_w + cfg.lane_wall_t / 2), (ly0 + ly1) / 2, 0.040),
            (cfg.lane_wall_t, ly1 - ly0, 0.050), amber)
    box("lane_end", (lx, ly1 - 0.004, 0.035), (2 * lw_out, 0.008, 0.040), amber)
    return root


def _rig_spawner_cfg(c: CaliberVaultSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rig" not in _SPAWNER_CACHE:

        @configclass
        class RigSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rig)
            mass: float = 60.0
            contact_offset: float = 0.002
            rail_r: float = 0.012
            rail_z: float = 0.140
            rail_len_x: float = 0.44
            rail_s0: float = 0.062
            rail_half_slope: float = 0.0341
            canopy_x0: float = 0.14
            canopy_z: float = 0.205
            fascia_slot_half_w: float = 0.035
            fascia_z0: float = 0.125
            ramp_x0: float = -0.08
            ramp_z_lo: float = 0.004
            ramp_z_hi: float = 0.030
            bulk_x0: float = 0.28
            bulk_x1: float = 0.30
            bulk_top: float = 0.108
            bulk_notch_half_w: float = 0.058
            side_half_w_in: float = 0.088
            side_half_w_out: float = 0.098
            wall_far_x0: float = 0.443
            wall_far_x1: float = 0.463
            vault_floor_top: float = 0.005
            port_x0: float = 0.353
            port_x1: float = 0.397
            port_z1: float = 0.059
            lane_floor_top: float = 0.015
            lane_half_w: float = 0.0225
            lane_wall_t: float = 0.008
            lane_y_end: float = 0.302

        _SPAWNER_CACHE["rig"] = RigSpawnerCfg

    return _SPAWNER_CACHE["rig"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.rig_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass=c.rig_mass, contact_offset=c.contact_offset,
        rail_r=c.rail_r, rail_z=c.rail_z, rail_len_x=c.rail_len_x, rail_s0=c.rail_s0,
        rail_half_slope=c.rail_half_slope, canopy_x0=c.canopy_x0, canopy_z=c.canopy_z,
        fascia_slot_half_w=c.fascia_slot_half_w, fascia_z0=c.fascia_z0,
        ramp_x0=c.ramp_x0, ramp_z_lo=c.ramp_z_lo, ramp_z_hi=c.ramp_z_hi,
        bulk_x0=c.bulk_x0, bulk_x1=c.bulk_x1, bulk_top=c.bulk_top,
        bulk_notch_half_w=c.bulk_notch_half_w, side_half_w_in=c.side_half_w_in,
        side_half_w_out=c.side_half_w_out, wall_far_x0=c.wall_far_x0,
        wall_far_x1=c.wall_far_x1, vault_floor_top=c.vault_floor_top,
        port_x0=c.port_x0, port_x1=c.port_x1, port_z1=c.port_z1,
        lane_floor_top=c.lane_floor_top, lane_half_w=c.lane_half_w,
        lane_wall_t=c.lane_wall_t, lane_y_end=c.lane_y_end,
    )


# ----- scene cfg -----------------------------------------------------------------------------------
@dataclass
class CaliberVaultSceneCfg(BaseCfg):
    """Config for `CaliberVaultScene`.  The rig is placed at reset with xy jitter and a
    full-circle yaw; the two marbles stage on the open floor in two rig-frame slots
    whose assignment swaps per episode.  All rubric bands are in the RIG FRAME."""

    # --- tunable: rubric thresholds ----------------------------------------------------------------
    settle_lin: float = tunable(0.06)  # max marble |lin vel| when judging (above GPU creep)
    vault_x0: float = tunable(0.302)  # vault-interior band (rig frame): x low
    vault_x1: float = tunable(0.441)  # ... x high
    vault_half_w: float = tunable(0.086)  # ... |y| max
    vault_z0: float = tunable(0.008)  # ... marble-center z low (on/near the vault floor)
    vault_z1: float = tunable(0.110)  # ... z high (excludes rail rides and canopy perches)
    ride_x0: float = tunable(0.040)  # ride band: big-marble center x low
    ride_x1: float = tunable(0.310)  # ... x high (release happens past here)
    ride_half_w: float = tunable(0.020)  # ... |y| max (centered between the rails)
    ride_z0: float = tunable(0.132)  # ... z low
    ride_z1: float = tunable(0.185)  # ... z high
    canopy_ride_x: float = tunable(0.160)  # ride beyond this x latches "under the canopy"
    lane_y0: float = tunable(0.100)  # lane band: small-marble center y low
    lane_y1: float = tunable(0.250)  # ... y high (below start: genuinely advanced)
    lane_z0: float = tunable(0.022)  # ... z low
    lane_z1: float = tunable(0.060)  # ... z high

    # --- tunable: randomization --------------------------------------------------------------------
    rig_base: tuple = tunable((0.05, -0.08))  # nominal rig root xy (env frame)
    rig_jitter: float = tunable(0.050)  # uniform +/- xy jitter of the rig at reset
    slot_a: tuple = tunable((-0.12, 0.26))  # marble staging slot A (rig frame)
    slot_b: tuple = tunable((0.10, 0.30))  # marble staging slot B (rig frame)
    slot_jitter: float = tunable(0.030)  # uniform +/- xy jitter of each marble slot

    # --- info: marbles -----------------------------------------------------------------------------
    big_r: float = info(0.030)
    big_mass: float = info(0.120)
    big_color: tuple = info((0.72, 0.10, 0.12))
    small_r: float = info(0.016)
    small_mass: float = info(0.030)
    small_color: tuple = info((0.92, 0.92, 0.95))

    # --- info: rig geometry (rig frame, floor at z=0) ----------------------------------------------
    rig_mass: float = info(60.0)
    rail_r: float = info(0.012)
    rail_z: float = info(0.140)  # rail axis height
    rail_len_x: float = info(0.44)  # rails span x in [0, rail_len_x]
    rail_s0: float = info(0.062)  # axis separation at x=0
    rail_half_slope: float = info(0.0341)  # d(s/2)/dx — the divergence
    canopy_x0: float = info(0.14)  # canopy starts here; x<this is the open loading bay
    canopy_z: float = info(0.205)  # canopy underside
    fascia_slot_half_w: float = info(0.035)
    fascia_z0: float = info(0.125)
    ramp_x0: float = info(-0.08)
    ramp_z_lo: float = info(0.004)
    ramp_z_hi: float = info(0.030)
    bulk_x0: float = info(0.28)
    bulk_x1: float = info(0.30)
    bulk_top: float = info(0.108)  # center-wall top the rider must clear
    bulk_notch_half_w: float = info(0.058)
    side_half_w_in: float = info(0.088)  # vault interior half-width
    side_half_w_out: float = info(0.098)  # side wall exterior
    wall_far_x0: float = info(0.443)  # vault far wall interior face
    wall_far_x1: float = info(0.463)
    vault_floor_top: float = info(0.005)
    port_x0: float = info(0.353)  # port cut in the +y wall
    port_x1: float = info(0.397)
    port_z1: float = info(0.059)  # port headroom (opening spans lane_floor_top..port_z1)
    lane_floor_top: float = info(0.015)  # 10 mm above the vault floor: drop-in sill
    lane_half_w: float = info(0.0225)  # lane inner half-width (45 mm inner width)
    lane_wall_t: float = info(0.008)
    lane_y_end: float = info(0.302)
    drop_x: float = info(0.050)  # intended rig-frame x where the big marble is set on the rails
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    x_release: float = field(default=None, init=False)  # geometric fall-through x of the big
    ride_h0: float = field(default=None, init=False)  # big center height above rail axis at drop_x

    def _sep(self, x: float) -> float:
        return self.rail_s0 + 2 * self.rail_half_slope * x

    def __post_init__(self) -> None:
        a_big = self.big_r + self.rail_r
        a_small = self.small_r + self.rail_r
        # -- rails: pass/never-pass interference margins --------------------------------------------
        assert self._sep(0.0) >= 2 * a_small + 0.004, (
            f"small marble must ALWAYS fall through: s0 {self._sep(0.0)} vs {2 * a_small}")
        assert self._sep(self.drop_x) <= 2 * a_big - 0.016, (
            "big marble must bridge the rails with margin at the drop point")
        self.x_release = round((2 * a_big - self.rail_s0) / (2 * self.rail_half_slope), 4)
        assert self.bulk_x1 + 0.012 < self.x_release < self.wall_far_x0 - self.big_r - 0.05, (
            f"big-marble release x {self.x_release} must land it inside the vault")
        assert self._sep(self.rail_len_x) > 2 * a_big + 0.006, (
            "rails must widen past the big marble's bridge before they end")
        # -- rider clearances -----------------------------------------------------------------------
        self.ride_h0 = round(math.sqrt(a_big**2 - (self._sep(self.drop_x) / 2) ** 2), 4)
        top0 = self.rail_z + self.ride_h0 + self.big_r
        assert top0 <= self.canopy_z - 0.006, f"riding big marble grazes the canopy ({top0})"
        h_bulk = math.sqrt(max(a_big**2 - (self._sep(self.bulk_x1) / 2) ** 2, 0.0))
        belly = self.rail_z + h_bulk - self.big_r
        assert belly >= self.bulk_top + 0.006, (
            f"rider belly {belly:.4f} must clear the bulkhead center wall {self.bulk_top}")
        assert self.fascia_slot_half_w >= self.big_r + 0.004, "fascia slot pinches the rider"
        assert self.fascia_z0 <= self.rail_z + self.ride_h0 - self.big_r - 0.008, (
            "fascia sill must sit below the riding marble's underbelly")
        # -- port / lane: pass the small, refuse the big --------------------------------------------
        port_w = self.port_x1 - self.port_x0
        port_h = self.port_z1 - self.lane_floor_top
        assert port_w >= 2 * self.small_r + 0.008 and port_h >= 2 * self.small_r + 0.008, (
            "port must pass the small marble with margin")
        assert port_w <= 2 * self.big_r - 0.012, "port must refuse the big marble"
        assert 2 * self.lane_half_w <= 2 * self.big_r - 0.012, (
            "the big marble must not even fit into the lane")
        assert 2 * self.lane_half_w >= 2 * self.small_r + 0.008, "lane pinches the small marble"
        # any marble in the lane is aligned with the port (no funnel needed)
        assert self.lane_half_w - self.small_r <= (port_w / 2 - self.small_r) + 0.001, (
            "lane must confine the small marble to the port's admission window")
        assert self.lane_floor_top - self.vault_floor_top >= 0.006, (
            "sill must drop INTO the vault so the small marble cannot roll back out")
        # -- canopy seals the vault from above ------------------------------------------------------
        assert self.canopy_x0 <= self.bulk_x0 - 0.10, "canopy must roof the whole approach"
        # -- staging slots: clear of the rig footprint and of each other at worst-case jitter -------
        j = self.slot_jitter
        for name, (sx, sy) in (("slot_a", self.slot_a), ("slot_b", self.slot_b)):
            # clear of the rail/vault corridor (|y| <= side_half_w_out, x in [ramp_x0, far wall])
            assert abs(sy) - j > self.side_half_w_out + 0.03, (
                f"{name} inside the rig corridor at worst-case jitter")
            # clear of the runway-lane footprint (x around the port, y up to lane_y_end)
            assert sx + j < self.port_x0 - self.lane_wall_t - self.lane_half_w - 0.03, (
                f"{name} too close to the lane at worst-case jitter")
        d = math.hypot(self.slot_a[0] - self.slot_b[0], self.slot_a[1] - self.slot_b[1])
        assert d - 2 * math.sqrt(2) * j > self.big_r + self.small_r + 0.02, (
            "staging slots can collide at worst-case jitter")
        # -- rubric bands consistent with the geometry ----------------------------------------------
        assert self.vault_x0 >= self.bulk_x1 and self.vault_x1 <= self.wall_far_x0, "vault band"
        assert self.vault_z1 < self.ride_z0, "vault band overlaps the ride band"
        assert self.vault_z0 < self.vault_floor_top + self.small_r < self.vault_z1, (
            "a small marble resting on the vault floor must be inside the vault z band")
        assert self.lane_z0 < self.lane_floor_top + self.small_r < self.lane_z1, "lane z band"
        assert self.lane_y1 < self.lane_y_end - 0.03, "lane 'advanced' band must be down-lane"


# ----- scene ----------------------------------------------------------------------------------------
@SCENES.register("caliber_vault")
class CaliberVaultScene(BaseScene):
    cfg: CaliberVaultSceneCfg

    def __init__(self, cfg: CaliberVaultSceneCfg | None = None) -> None:
        super().__init__(cfg or CaliberVaultSceneCfg())

    # ----- assets ---------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg

        def marble(radius: float, mass: float, color: tuple) -> Any:
            return sim_utils.SphereCfg(
                radius=radius,
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4,
                    max_depenetration_velocity=0.5,
                    linear_damping=0.02, angular_damping=0.05),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.7, dynamic_friction=0.6, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "rig": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rig",
                spawn=_rig_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.rig_base[0], c.rig_base[1], 0.0)),
            ),
            "big": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/MarbleBig",
                spawn=marble(c.big_r, c.big_mass, c.big_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rig_base[0] + c.slot_a[0], c.rig_base[1] + c.slot_a[1],
                         c.big_r + 0.003)),
            ),
            "small": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/MarbleSmall",
                spawn=marble(c.small_r, c.small_mass, c.small_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rig_base[0] + c.slot_b[0], c.rig_base[1] + c.slot_b[1],
                         c.small_r + 0.003)),
            ),
        }

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

    # ----- lifecycle ------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n = env.num_envs
        dev = env.device
        self.rig: RigidObject = env.iscene["rig"]
        self.big: RigidObject = env.iscene["big"]
        self.small: RigidObject = env.iscene["small"]
        self.env_origins = env.iscene.env_origins
        # per-episode staging assignment: big marble in slot A (True) or slot B (False)
        self.big_in_a = torch.ones(n, dtype=torch.bool, device=dev)
        # latched progress (updated in post_step every step)
        self._ride_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._canopy_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._big_vault_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._lane_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._small_vault_ever = torch.zeros(n, dtype=torch.bool, device=dev)

    # ----- rig-frame readouts ---------------------------------------------------------------------
    def local(self, body: RigidObject) -> torch.Tensor:
        """(N, 3) body center in the RIG FRAME (all rubric bands live here)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = body.data.root_pos_w - self.rig.data.root_pos_w
        return quat_apply_inverse(self.rig.data.root_quat_w, rel)

    def _in_box(self, p: torch.Tensor, x0: float, x1: float, half_w: float,
                z0: float, z1: float) -> torch.Tensor:
        return ((p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1].abs() < half_w)
                & (p[:, 2] > z0) & (p[:, 2] < z1))

    # ----- predicates -----------------------------------------------------------------------------
    def in_vault(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: marble center inside the vault interior (rig frame), down at
        floor level — excludes rail rides above and canopy perches outside."""
        c = self.cfg
        return self._in_box(self.local(body), c.vault_x0, c.vault_x1, c.vault_half_w,
                            c.vault_z0, c.vault_z1)

    def riding(self) -> torch.Tensor:
        """(N,) bool: the big marble bridging the rails in the ride corridor."""
        c = self.cfg
        return self._in_box(self.local(self.big), c.ride_x0, c.ride_x1, c.ride_half_w,
                            c.ride_z0, c.ride_z1)

    def in_lane(self) -> torch.Tensor:
        """(N,) bool: the small marble advanced down the runway lane."""
        c = self.cfg
        p = self.local(self.small)
        lane_cx = (c.port_x0 + c.port_x1) / 2
        return (((p[:, 0] - lane_cx).abs() < c.lane_half_w)
                & (p[:, 1] > c.lane_y0) & (p[:, 1] < c.lane_y1)
                & (p[:, 2] > c.lane_z0) & (p[:, 2] < c.lane_z1))

    def settled(self) -> torch.Tensor:
        """(N,) bool: both marbles translationally still (above the GPU creep band)."""
        c = self.cfg
        return ((self.big.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.small.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin))

    # ----- mechanism (every step) -----------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        c = self.cfg
        ride = self.riding()
        self._ride_ever |= ride
        self._canopy_ever |= ride & (self.local(self.big)[:, 0] > c.canopy_ride_x)
        self._big_vault_ever |= self.in_vault(self.big)
        self._lane_ever |= self.in_lane()
        self._small_vault_ever |= self.in_vault(self.small)

    # ----- reset ----------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the kinematic rig (xy jitter + full-circle yaw), stage
        the two marbles on the open floor in the two rig-frame slots with per-episode
        assignment swap + jitter."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # rig pose
        rxy = torch.tensor(c.rig_base, device=dev).expand(m, 2).clone()
        rxy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.rig_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = rxy
        st[:, 3] = torch.cos(yaw / 2)
        st[:, 6] = torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.rig.write_root_state_to_sim(st, env_ids)

        # staging: swap assignment per episode (torch.rand — first-randint is degenerate)
        self.big_in_a[env_ids] = torch.rand(m, device=dev) < 0.5
        cy, sy = torch.cos(yaw), torch.sin(yaw)

        def place(body: RigidObject, slot_xy: torch.Tensor, radius: float) -> None:
            jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            p = slot_xy + jit
            wx = rxy[:, 0] + cy * p[:, 0] - sy * p[:, 1]
            wy = rxy[:, 1] + sy * p[:, 0] + cy * p[:, 1]
            bst = torch.zeros(m, 13, device=dev)
            bst[:, 0] = wx
            bst[:, 1] = wy
            bst[:, 2] = radius + 0.003
            bst[:, 3] = 1.0
            bst[:, 0:3] += origin
            body.write_root_state_to_sim(bst, env_ids)

        a = torch.tensor(c.slot_a, device=dev).expand(m, 2)
        b = torch.tensor(c.slot_b, device=dev).expand(m, 2)
        sel = self.big_in_a[env_ids].unsqueeze(1)
        place(self.big, torch.where(sel, a, b), c.big_r)
        place(self.small, torch.where(sel, b, a), c.small_r)

        self._ride_ever[env_ids] = False
        self._canopy_ever[env_ids] = False
        self._big_vault_ever[env_ids] = False
        self._lane_ever[env_ids] = False
        self._small_vault_ever[env_ids] = False

    # ----- state (full, restorable) ---------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "rig": self.rig.data.root_state_w[env_ids].clone(),
            "big": self.big.data.root_state_w[env_ids].clone(),
            "small": self.small.data.root_state_w[env_ids].clone(),
            "big_in_a": self.big_in_a[env_ids].clone(),
            "latches": torch.stack([self._ride_ever[env_ids], self._canopy_ever[env_ids],
                                    self._big_vault_ever[env_ids], self._lane_ever[env_ids],
                                    self._small_vault_ever[env_ids]], dim=1),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.rig.write_root_state_to_sim(state["rig"], env_ids)
        self.big.write_root_state_to_sim(state["big"], env_ids)
        self.small.write_root_state_to_sim(state["small"], env_ids)
        self.big_in_a[env_ids] = state["big_in_a"]
        lt = state["latches"]
        self._ride_ever[env_ids] = lt[:, 0]
        self._canopy_ever[env_ids] = lt[:, 1]
        self._big_vault_ever[env_ids] = lt[:, 2]
        self._lane_ever[env_ids] = lt[:, 3]
        self._small_vault_ever[env_ids] = lt[:, 4]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "On the floor stands a track-and-vault RIG. At one end is an OPEN LOADING "
            "BAY where two steel RAILS begin, running side by side under a long dark "
            "CANOPY (roof). The rails look parallel but secretly DIVERGE: their gap "
            f"starts at {c.rail_s0 * 1000:.0f} mm and widens along the run. Under the "
            "canopy, past a low interior wall, the rails pass over a fully roofed, "
            "walled VAULT chamber sunk to floor level; below the rails before the "
            "vault, a return ramp slopes back out of the bay. In the vault's side "
            f"wall there is a small square PORT ({(c.port_x1 - c.port_x0) * 1000:.0f} mm "
            "wide) at floor height, fed from outside by an open-top RUNWAY LANE "
            f"({2 * c.lane_half_w * 1000:.0f} mm inner width) with a small drop-in "
            "sill at the doorway.\n"
            f"Two marbles rest on the open floor: a RED one {2 * c.big_r * 1000:.0f} mm "
            f"across and a WHITE one {2 * c.small_r * 1000:.0f} mm across.\n"
            "Goal: BOTH marbles must end up at rest INSIDE the vault. The vault is "
            "roofed — nothing can be dropped in from above. The red marble is too "
            "big for the port (and for the lane itself); its only way in is the rail "
            "run: set it onto the rails in the open bay and it will bridge the gap, "
            "roll along under the canopy, and fall through the widening gap right "
            "into the vault. The white marble is too small for the rails — it falls "
            "straight through the gap onto the return ramp and rolls back out; its "
            "only way in is the runway lane: roll it down the lane, through the "
            "port, and over the sill. Either marble may be delivered first."
        )

    def instruction(self) -> str:
        return (
            "Get both marbles into the roofed vault: send the big red marble down "
            "the diverging rails from the open loading bay so it drops through into "
            "the vault, and roll the small white marble down the side runway lane "
            "through the little port; the vault's roof blocks any drop-in from "
            "above, the port and lane are too small for the red marble, and the "
            "rails cannot hold the white one."
        )

    # ----- rubric ---------------------------------------------------------------------------------
    def score(self) -> torch.Tensor:
        """(N,) float in [0,1], latched stages of the demonstrated solution: 0.10 big
        ever riding the rails + 0.15 ever riding under the canopy + 0.25 big ever in
        the vault + 0.15 small ever advanced down the lane + 0.25 small ever in the
        vault (sum 0.90); exactly 1.0 iff success(). Null policy scores 0; latched
        credit never evaporates."""
        s = (0.10 * self._ride_ever.float() + 0.15 * self._canopy_ever.float()
             + 0.25 * self._big_vault_ever.float() + 0.15 * self._lane_ever.float()
             + 0.25 * self._small_vault_ever.float())
        return torch.where(self.success(),
                           torch.ones(self.env.num_envs, device=self.env.device),
                           s.clamp(0.0, 0.90))

    def success(self) -> torch.Tensor:
        """(N,) bool, all physical: both marbles at rest inside the vault interior."""
        return self.in_vault(self.big) & self.in_vault(self.small) & self.settled()


register_env("simgen", lambda: EnvCfg(scene="caliber_vault", robot="null"))

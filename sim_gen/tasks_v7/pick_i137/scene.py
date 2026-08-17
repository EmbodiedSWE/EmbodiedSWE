"""GravityVaultScene — the target region is SEALED: the red ball must be delivered by
GRAVITY down a guarded feed tower, after the sliding gate interlock is withdrawn.

Derived from mujoco_playground/pick ("bring the box to the target": one grasp of a 4 cm
cube and one guided free-space carry to a floating target pose — the target is an
arbitrary reachable point and the whole plan is pick-carry-place). Here the "target" is
the interior of a closed grey CHAMBER whose only opening is a roofed feed TOWER rising
from its roof to an open funnel MOUTH 0.42 m up: no arm can place, push or drop the ball
into the chamber directly (the smoke battery parks a ball on the roof and against the
walls to prove it). Midway down the tower a YELLOW GATE — a free tongue plate riding in
through-slots in the tower walls, with a grasp knob protruding to one side — blocks the
channel: a ball dropped early just parks on the gate. The plan the seed never needs:
(1) slide the gate OUT of its slots by its knob (a contact-rich captive-slide
extraction), (2) carry the RED ball above the funnel and RELEASE it — gravity and the
funnel walls do the final transport into the chamber. A same-size BLUE decoy ball pays
nothing.

Assets are fully procedural, one kinematic compound + one dynamic compound + two spheres:
  - vault (KINEMATIC, one body, vault frame: chamber centre at xy 0, ground z 0):
      chamber: floor plate 0.22 sq x 0.03; interior 0.18 sq x 0.14 tall; walls 0.02;
               a 30 mm viewing slit in the +x wall (z 0.06..0.09 — a 50 mm ball cannot
               pass); roof z 0.17..0.19 with a central 0.09 sq feed hole;
      tower:   interior 0.09 sq, walls 0.02, z 0.19..0.36; the +/-y walls carry a
               full-width gate slot z 0.245..0.269 (24 mm — the ball cannot pass);
      funnel:  four tilted plates z 0.36..0.42 flaring to a ~0.17 sq mouth.
  - gate (DYNAMIC, one body): tongue 0.084 x 0.215 x 0.012 spanning both slots, plus a
    knob tab on the outer end; per episode it is inserted from +y or -y (mirrored).
  - red ball / blue decoy: r 0.025, 0.10 kg, identical physics, different colors.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve trajectory):
  0.30 * gate_out — the gate tongue EVER fully clear of the tower channel   (latched)
  0.25 * entered  — the RED ball EVER inside the tower channel (descent begun; the
                    only way in for an arm is over the funnel mouth)        (latched)
  0.20 * chamber  — the RED ball EVER inside the chamber interior           (latched)
  1.0 iff success() — the RED ball inside the chamber, settled, finite.
  Non-success capped at 0.75. Null policy latches nothing (score ~0).

Honesty geometry (asserted in `__post_init__`):
  - the tower-wall side gaps beside the tongue, the gate slot height and the viewing
    slit are all SMALLER than the ball: a spanning gate really blocks, and the chamber
    really is sealed from everywhere but the mouth;
  - a ball resting ON the roof or ON the gate sits OUTSIDE the chamber z-band;
  - at max insertion jitter the tongue still spans the channel;
  - funnel mouth and tower interior clear the ball with real margins;
  - knob protrusion leaves finger room between knob and tower wall.

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
    out[..., 1:] = -out[..., 1:]
    return out


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4), pure torch."""
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(qv, t, dim=-1)


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


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


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, *, center, size, color, contact_offset: float, orient=None):
    """One collidable box child: translate (+ optional wxyz orient) + scale, displayColor."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(box.GetPrim(), contact_offset)
    return box.GetPrim()


def _phys_material(stage, path: str, static: float, dynamic: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _bind_material(prim, mat) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _rigid_dynamic(root, mass: float) -> None:
    """Dynamic rigid-body armor on the compound root: explicit mass (custom spawners
    apply NO cfg schemas — author everything here), mild damping, depenetration cap,
    iterated solver, no sleeping while velocities are judged."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.10)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _rigid_kinematic(root) -> None:
    """Kinematic rigid-body armor on the compound root (teleportable at reset)."""
    from pxr import UsdPhysics

    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(80.0)


def _spawn_vault(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC vault at `prim_path`. Root origin = chamber centre at
    ground level. Children: chamber floor/walls/roof frame, tower walls (the +/-y
    pair split around the gate slot), four tilted funnel plates."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_kinematic(root)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.mu_static, cfg.mu_dynamic)
    c = cfg
    co = c.contact_offset
    hi = c.chamber_in / 2                      # chamber interior half (0.09)
    ho = hi + c.wall_t                         # chamber outer half (0.11)
    z0, z1 = c.floor_t, c.floor_t + c.chamber_h          # chamber interior z band
    z_roof_top = z1 + c.roof_t                            # 0.19
    si = c.shaft_in / 2                        # tower interior half (0.045)
    so = si + c.wall_t                         # tower outer half (0.065)
    z_shaft_top = z_roof_top + c.shaft_h                  # 0.36
    slot_lo = z_roof_top + c.slot_lift                    # 0.245
    slot_hi = slot_lo + c.slot_gap                        # 0.269
    kids = []

    def box(name, center, size, color, orient=None):
        kids.append(_box(stage, f"{prim_path}/{name}", center=center, size=size,
                         color=color, contact_offset=co, orient=orient))

    # chamber floor + walls (viewing slit in the +x wall)
    box("floor", (0, 0, c.floor_t / 2), (2 * ho, 2 * ho, c.floor_t), c.base_color)
    box("wall_xn", (-hi - c.wall_t / 2, 0, (z0 + z1) / 2),
        (c.wall_t, 2 * ho, c.chamber_h), c.base_color)
    box("wall_xp_lo", (hi + c.wall_t / 2, 0, (z0 + c.slit_lo) / 2),
        (c.wall_t, 2 * ho, c.slit_lo - z0), c.base_color)
    box("wall_xp_hi", (hi + c.wall_t / 2, 0, (c.slit_hi + z1) / 2),
        (c.wall_t, 2 * ho, z1 - c.slit_hi), c.base_color)
    for s in (1.0, -1.0):
        box(f"wall_y{'p' if s > 0 else 'n'}", (0, s * (hi + c.wall_t / 2), (z0 + z1) / 2),
            (2 * hi, c.wall_t, c.chamber_h), c.base_color)
    # roof frame around the central shaft hole
    ry = (si + ho) / 2
    for s in (1.0, -1.0):
        box(f"roof_y{'p' if s > 0 else 'n'}", (0, s * ry, z1 + c.roof_t / 2),
            (2 * ho, ho - si, c.roof_t), c.roof_color)
        box(f"roof_x{'p' if s > 0 else 'n'}", (s * ry, 0, z1 + c.roof_t / 2),
            (ho - si, 2 * si, c.roof_t), c.roof_color)
    # tower: +/-x plain walls, +/-y walls split around the full-width gate slot
    zc = (z_roof_top + z_shaft_top) / 2
    for s in (1.0, -1.0):
        box(f"shaft_x{'p' if s > 0 else 'n'}", (s * (si + c.wall_t / 2), 0, zc),
            (c.wall_t, 2 * so, c.shaft_h), c.tower_color)
        box(f"shaft_y{'p' if s > 0 else 'n'}_lo",
            (0, s * (si + c.wall_t / 2), (z_roof_top + slot_lo) / 2),
            (2 * si, c.wall_t, slot_lo - z_roof_top), c.tower_color)
        box(f"shaft_y{'p' if s > 0 else 'n'}_hi",
            (0, s * (si + c.wall_t / 2), (slot_hi + z_shaft_top) / 2),
            (2 * si, c.wall_t, z_shaft_top - slot_hi), c.tower_color)
    # funnel: four plates tilting outward from the tower rim to the mouth
    mo = c.mouth_in / 2
    run = mo - si
    tilt = math.atan2(run, c.funnel_h)
    slope = math.hypot(run, c.funnel_h) + 0.004
    cxy = (si + mo) / 2 + 0.005                # nudged outward: no inward ledge
    zf = z_shaft_top + c.funnel_h / 2
    ch, sh = math.cos(tilt / 2), math.sin(tilt / 2)
    wide = c.mouth_in + 0.015
    box("funnel_xp", (cxy, 0, zf), (c.funnel_t, wide, slope), c.funnel_color,
        orient=(ch, 0.0, sh, 0.0))
    box("funnel_xn", (-cxy, 0, zf), (c.funnel_t, wide, slope), c.funnel_color,
        orient=(ch, 0.0, -sh, 0.0))
    box("funnel_yp", (0, cxy, zf), (wide, c.funnel_t, slope), c.funnel_color,
        orient=(ch, -sh, 0.0, 0.0))
    box("funnel_yn", (0, -cxy, zf), (wide, c.funnel_t, slope), c.funnel_color,
        orient=(ch, sh, 0.0, 0.0))
    for k in kids:
        _bind_material(k, mat)
    return root


def _spawn_gate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC gate at `prim_path`. Root origin = tongue centre; body +y
    points toward the knob end. Children: tongue plate + knob tab."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_dynamic(root, cfg.mass)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.mu_static, cfg.mu_dynamic)
    c = cfg
    kids = [
        _box(stage, f"{prim_path}/tongue", center=(0.0, 0.0, 0.0),
             size=(c.tongue_w, c.tongue_len, c.tongue_t), color=c.color,
             contact_offset=c.contact_offset),
        _box(stage, f"{prim_path}/knob",
             center=(0.0, c.tongue_len / 2 - c.knob_d / 2,
                     c.tongue_t / 2 + c.knob_h / 2),
             size=(c.knob_w, c.knob_d, c.knob_h), color=c.color,
             contact_offset=c.contact_offset),
    ]
    for k in kids:
        _bind_material(k, mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "vault" not in _SPAWNER_CACHE:

        @configclass
        class VaultSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_vault)
            chamber_in: float = 0.18
            chamber_h: float = 0.14
            wall_t: float = 0.02
            floor_t: float = 0.03
            roof_t: float = 0.02
            slit_lo: float = 0.06
            slit_hi: float = 0.09
            shaft_in: float = 0.09
            shaft_h: float = 0.17
            slot_lift: float = 0.055
            slot_gap: float = 0.024
            funnel_h: float = 0.06
            mouth_in: float = 0.17
            funnel_t: float = 0.008
            mu_static: float = 0.35
            mu_dynamic: float = 0.30
            base_color: tuple = (0.45, 0.47, 0.52)
            tower_color: tuple = (0.55, 0.58, 0.64)
            roof_color: tuple = (0.36, 0.38, 0.44)
            funnel_color: tuple = (0.66, 0.69, 0.74)
            contact_offset: float = 0.004

        @configclass
        class GateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_gate)
            tongue_w: float = 0.084
            tongue_len: float = 0.215
            tongue_t: float = 0.012
            knob_w: float = 0.032
            knob_d: float = 0.022
            knob_h: float = 0.05
            mass: float = 0.12
            mu_static: float = 0.30
            mu_dynamic: float = 0.25
            color: tuple = (0.92, 0.78, 0.10)
            contact_offset: float = 0.003

        _SPAWNER_CACHE["vault"] = VaultSpawnerCfg
        _SPAWNER_CACHE["gate"] = GateSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class GravityVaultSceneCfg(BaseCfg):
    """Config for `GravityVaultScene`. The interlock and the seal are enforced by the
    vault's own geometry; the rubric only reads out states the geometry makes
    meaningful (see the honesty asserts)."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    chamber_xy_tol: float = tunable(0.075)   # |ball centre| xy bound of the chamber region (m)
    settle_lin: float = tunable(0.05)        # max ball |lin vel| when judging success (m/s)
    settle_ang: float = tunable(1.5)         # max ball |ang vel| when judging success (rad/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    vault_jitter: float = tunable(0.035)     # vault xy jitter (+/- m)
    vault_yaw_deg: float = tunable(30.0)     # vault yaw jitter (+/- deg)
    gate_retract_jitter: float = tunable(0.012)  # gate inserted-depth jitter (0..this, m)
    ball_jitter: float = tunable(0.03)       # ball spawn xy jitter (+/- m)

    # --- info: vault structure (vault frame: chamber centre xy 0, ground z 0) --------------------
    chamber_in: float = info(0.18)           # chamber interior width
    chamber_h: float = info(0.14)            # chamber interior height
    wall_t: float = info(0.02)
    floor_t: float = info(0.03)
    roof_t: float = info(0.02)
    slit_lo: float = info(0.06)              # viewing slit z band in the +x wall
    slit_hi: float = info(0.09)
    shaft_in: float = info(0.09)             # tower interior width
    shaft_h: float = info(0.17)              # tower height (roof top -> funnel base)
    slot_lift: float = info(0.055)           # gate slot bottom above the roof top
    slot_gap: float = info(0.024)            # gate slot height (< ball diameter)
    funnel_h: float = info(0.06)
    mouth_in: float = info(0.17)             # funnel mouth interior width
    funnel_t: float = info(0.008)
    vault_pos: tuple = info((0.45, 0.0))     # world xy of the vault frame origin
    vault_mu_static: float = info(0.35)
    vault_mu_dynamic: float = info(0.30)
    # --- info: gate ------------------------------------------------------------------------------
    tongue_w: float = info(0.084)            # tongue width (tower interior minus 6 mm)
    tongue_len: float = info(0.215)
    tongue_t: float = info(0.012)
    knob_w: float = info(0.032)
    knob_d: float = info(0.022)
    knob_h: float = info(0.05)
    gate_mass: float = info(0.12)
    gate_mu_static: float = info(0.30)
    gate_mu_dynamic: float = info(0.25)
    # --- info: balls -----------------------------------------------------------------------------
    ball_r: float = info(0.025)
    ball_mass: float = info(0.10)
    ball_mu: float = info(0.5)
    red_color: tuple = info((0.85, 0.10, 0.10))
    blue_color: tuple = info((0.10, 0.20, 0.85))
    ball_local_x: float = info(0.30)         # ball spawn slots, vault frame
    ball_local_y: float = info(0.14)
    ground_mu: float = info(0.40)
    contact_offset: float = info(0.004)
    # rubric weights (0.30 + 0.25 + 0.20 = 0.75 <= the non-success cap 0.75)
    w_gate: float = info(0.30)
    w_entered: float = info(0.25)
    w_chamber: float = info(0.20)

    # Derived (filled in __post_init__).
    z_roof_top: float = field(default=0.0, init=False)   # chamber roof top
    z_shaft_top: float = field(default=0.0, init=False)  # tower top / funnel base
    z_mouth: float = field(default=0.0, init=False)      # funnel mouth rim
    slot_lo: float = field(default=0.0, init=False)      # gate slot bottom
    gate_rest_z: float = field(default=0.0, init=False)  # gate tongue-centre rest height
    gate_in_y: float = field(default=0.0, init=False)    # |gate centre y| when inserted

    def __post_init__(self) -> None:
        self.z_roof_top = self.floor_t + self.chamber_h + self.roof_t
        self.z_shaft_top = self.z_roof_top + self.shaft_h
        self.z_mouth = self.z_shaft_top + self.funnel_h
        self.slot_lo = self.z_roof_top + self.slot_lift
        self.gate_rest_z = self.slot_lo + self.tongue_t / 2 + 0.001
        self.gate_in_y = self.tongue_len / 2 - (self.shaft_in / 2 + self.wall_t)
        d = 2 * self.ball_r
        # the interlock really blocks and the chamber really is sealed:
        assert self.shaft_in - self.tongue_w < d - 0.02, \
            "side gaps beside the tongue must not pass the ball"
        assert self.slot_gap < d - 0.02, "the gate slot must not pass the ball"
        assert self.slit_hi - self.slit_lo < d - 0.015, \
            "the viewing slit must not pass the ball"
        # a spanning gate spans at max insertion jitter (inner tip past the -y interior
        # face by a real margin):
        tip_y_max = (self.gate_in_y + self.gate_retract_jitter) - self.tongue_len / 2
        assert tip_y_max < -(self.shaft_in / 2) - 0.005, \
            "at max insertion jitter the tongue must still span the channel"
        # a ball resting on the roof / on the gate sits OUTSIDE the chamber z-band:
        assert self.z_roof_top + self.ball_r > self.floor_t + self.chamber_h, \
            "roof-top rest pose must be outside the chamber band"
        assert self.slot_lo + self.tongue_t + self.ball_r > self.floor_t + self.chamber_h, \
            "on-gate rest pose must be outside the chamber band"
        # ball clearances (drop tolerance for the arm):
        assert self.mouth_in > d + 0.10, "funnel mouth must clear the ball generously"
        assert self.shaft_in > d + 0.03, "tower interior must clear the ball"
        assert self.chamber_in > d + 0.10, "chamber interior must clear the ball"
        assert self.chamber_xy_tol < self.chamber_in / 2, "xy tol must sit inside the walls"
        # knob finger room: knob inner face to tower outer wall when fully inserted
        knob_inner = self.gate_in_y + self.tongue_len / 2 - self.knob_d
        assert knob_inner - (self.shaft_in / 2 + self.wall_t) > 0.045, \
            "need finger room between knob and tower wall"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("gravity_vault")
class GravityVaultScene(BaseScene):
    cfg: GravityVaultSceneCfg

    def __init__(self, cfg: GravityVaultSceneCfg | None = None) -> None:
        super().__init__(cfg or GravityVaultSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        vault_spawn = cls["vault"](
            chamber_in=c.chamber_in, chamber_h=c.chamber_h, wall_t=c.wall_t,
            floor_t=c.floor_t, roof_t=c.roof_t, slit_lo=c.slit_lo, slit_hi=c.slit_hi,
            shaft_in=c.shaft_in, shaft_h=c.shaft_h, slot_lift=c.slot_lift,
            slot_gap=c.slot_gap, funnel_h=c.funnel_h, mouth_in=c.mouth_in,
            funnel_t=c.funnel_t, mu_static=c.vault_mu_static,
            mu_dynamic=c.vault_mu_dynamic, contact_offset=c.contact_offset)
        gate_spawn = cls["gate"](
            tongue_w=c.tongue_w, tongue_len=c.tongue_len, tongue_t=c.tongue_t,
            knob_w=c.knob_w, knob_d=c.knob_d, knob_h=c.knob_h, mass=c.gate_mass,
            mu_static=c.gate_mu_static, mu_dynamic=c.gate_mu_dynamic,
            contact_offset=0.003)

        def ball(color) -> RigidObjectCfg:
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BallTmp",  # overwritten per asset below
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.10,
                        angular_damping=0.20,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.ball_mu, dynamic_friction=c.ball_mu - 0.05,
                        restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.ball_r + 0.003)),
            )

        red = ball(c.red_color)
        red.prim_path = "{ENV_REGEX_NS}/RedBall"
        red.init_state.pos = (c.vault_pos[0] + c.ball_local_x,
                              c.vault_pos[1] - c.ball_local_y, c.ball_r + 0.003)
        blue = ball(c.blue_color)
        blue.prim_path = "{ENV_REGEX_NS}/BlueBall"
        blue.init_state.pos = (c.vault_pos[0] + c.ball_local_x,
                               c.vault_pos[1] + c.ball_local_y, c.ball_r + 0.003)

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.ground_mu, dynamic_friction=c.ground_mu - 0.05,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "vault": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Vault",
                spawn=vault_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.vault_pos[0], c.vault_pos[1], 0.0)),
            ),
            "gate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Gate",
                spawn=gate_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.vault_pos[0], c.vault_pos[1] + c.gate_in_y, c.gate_rest_z)),
            ),
            "red": red,
            "blue": blue,
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 240.0,  # a ball dropped 0.42 m lands at ~2.9 m/s: 12 mm/step,
            physx={          # caught by the 4 mm speculative contact offsets
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
        self.vault: RigidObject = env.iscene["vault"]
        self.gate: RigidObject = env.iscene["gate"]
        self.red: RigidObject = env.iscene["red"]
        self.blue: RigidObject = env.iscene["blue"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self.gate_side = torch.ones(n, device=dev)  # +1: knob toward vault +y
        # latches (partial credit survives transients; success is judged live)
        self._l_gate = torch.zeros(n, dtype=torch.bool, device=dev)
        self._l_entered = torch.zeros(n, dtype=torch.bool, device=dev)
        self._l_chamber = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: vault pose (xy jitter + yaw), gate inserted from a random
        side at a random depth, red/blue balls on jittered slots with a random side
        swap, latches cleared. All draws via torch.rand (uniform, readback-verified)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def rnd(k: float) -> torch.Tensor:
            return (torch.rand(m, device=dev) * 2 - 1) * k

        def write(body, local_xy, z, q) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = local_xy
            st[:, 2] = z
            st[:, 3:7] = q
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        vx = c.vault_pos[0] + rnd(c.vault_jitter)
        vy = c.vault_pos[1] + rnd(c.vault_jitter)
        vyaw = rnd(math.radians(c.vault_yaw_deg))
        vq = _qz(vyaw)
        vxy = torch.stack([vx, vy], dim=-1)
        write(self.vault, vxy, torch.zeros(m, device=dev), vq)

        def to_world(local_xy: torch.Tensor) -> torch.Tensor:
            l3 = torch.cat([local_xy, torch.zeros(m, 1, device=dev)], dim=-1)
            return vxy + _qapply(vq, l3)[:, :2]

        # gate: random side (mirrored insertion) + random retraction depth
        side = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
        self.gate_side[env_ids] = side
        retract = torch.rand(m, device=dev) * c.gate_retract_jitter
        gy = side * (c.gate_in_y + retract)
        g_local = torch.stack([torch.zeros(m, device=dev), gy], dim=-1)
        gyaw = vyaw + torch.where(side < 0, torch.full((m,), math.pi, device=dev),
                                  torch.zeros(m, device=dev))
        write(self.gate, to_world(g_local),
              torch.full((m,), c.gate_rest_z, device=dev), _qz(gyaw))

        # balls: jittered slots, random side swap
        swap = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
        for body, s in ((self.red, swap), (self.blue, -swap)):
            bl = torch.stack([
                c.ball_local_x + rnd(c.ball_jitter),
                s * c.ball_local_y + rnd(c.ball_jitter)], dim=-1)
            write(body, to_world(bl),
                  torch.full((m,), c.ball_r + 0.003, device=dev),
                  _qz(torch.zeros(m, device=dev)))

        self._l_gate[env_ids] = False
        self._l_entered[env_ids] = False
        self._l_chamber[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "vault": self.vault.data.root_state_w[env_ids].clone(),
            "gate": self.gate.data.root_state_w[env_ids].clone(),
            "red": self.red.data.root_state_w[env_ids].clone(),
            "blue": self.blue.data.root_state_w[env_ids].clone(),
            "gate_side": self.gate_side[env_ids].clone(),
            "l_gate": self._l_gate[env_ids].clone(),
            "l_entered": self._l_entered[env_ids].clone(),
            "l_chamber": self._l_chamber[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.vault.write_root_state_to_sim(state["vault"], env_ids)
        self.gate.write_root_state_to_sim(state["gate"], env_ids)
        self.red.write_root_state_to_sim(state["red"], env_ids)
        self.blue.write_root_state_to_sim(state["blue"], env_ids)
        self.gate_side[env_ids] = state["gate_side"]
        self._l_gate[env_ids] = state["l_gate"]
        self._l_entered[env_ids] = state["l_entered"]
        self._l_chamber[env_ids] = state["l_chamber"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A grey VAULT stands on the floor: a closed box chamber "
            f"({c.chamber_in * 1000:.0f} mm square inside, fully walled and roofed, with "
            f"only a narrow {1000 * (c.slit_hi - c.slit_lo):.0f} mm viewing slit in one "
            f"wall) with a square feed TOWER rising from the middle of its roof to an "
            f"open FUNNEL mouth ~{c.mouth_in * 1000:.0f} mm wide at "
            f"{c.z_mouth * 1000:.0f} mm height. The funnel mouth is the ONLY way into "
            f"the chamber. Midway up the tower a YELLOW GATE — a flat tongue riding in "
            f"slots through the tower walls, with a knob sticking out one side — blocks "
            f"the channel: anything dropped into the funnel just rests on the gate. On "
            f"the floor nearby lie two {2 * c.ball_r * 1000:.0f} mm balls, one RED and "
            f"one BLUE. The vault's position and heading, which side the gate's knob "
            f"faces, and the ball positions vary per episode.\n"
            f"Goal: get the RED ball to rest inside the chamber. Slide the yellow gate "
            f"fully out of the tower by its knob (in the direction the knob points; it "
            f"cannot be pushed in), then drop the red ball into the funnel mouth so it "
            f"falls down the tower into the chamber. The BLUE ball is a decoy: it "
            f"counts for nothing, and putting it in the chamber neither helps nor "
            f"hurts. There is no required order — but while the gate spans the tower, "
            f"a dropped ball parks on it and scores nothing. Only the settled final "
            f"state of the red ball is judged."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pull the yellow gate out of the tower by its knob, then pick up the red "
            "ball and drop it into the funnel on top so it falls into the closed "
            "chamber below. Only the red ball inside the chamber counts — the blue "
            "ball is a decoy."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _vault_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points (N,3) -> the vault frame."""
        return _qapply(_qinv(self.vault.data.root_quat_w),
                       pos_w - self.vault.data.root_pos_w)

    def gate_pts_local(self) -> torch.Tensor:
        """(N,9,3) vault-frame positions of nine points along the tongue centreline."""
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        s = torch.linspace(-c.tongue_len / 2, c.tongue_len / 2, 9, device=dev)
        offs = torch.zeros(len(s), 3, device=dev)
        offs[:, 1] = s
        q = self.gate.data.root_quat_w.unsqueeze(1).expand(n, len(s), 4)
        pts_w = self.gate.data.root_pos_w.unsqueeze(1) \
            + _qapply(q, offs.unsqueeze(0).expand(n, len(s), 3))
        vq = _qinv(self.vault.data.root_quat_w).unsqueeze(1).expand(n, len(s), 4)
        return _qapply(vq, pts_w - self.vault.data.root_pos_w.unsqueeze(1))

    def pin_clear(self) -> torch.Tensor:
        """(N,) bool, LIVE: no tongue point inside the tower channel's blocking box
        (|x|,|y| < shaft_in/2, z in the slot band) — the channel is open."""
        c = self.cfg
        p = self.gate_pts_local()
        si = c.shaft_in / 2
        blocking = (p[:, :, 0].abs() < si) & (p[:, :, 1].abs() < si) \
            & (p[:, :, 2] > c.slot_lo - 0.012) \
            & (p[:, :, 2] < c.slot_lo + c.slot_gap + 0.014)
        return ~blocking.any(dim=1)

    def in_chamber(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool, LIVE: body centre inside the chamber interior region."""
        c = self.cfg
        p = self._vault_local(body.data.root_pos_w)
        return (p[:, 0].abs() < c.chamber_xy_tol) & (p[:, 1].abs() < c.chamber_xy_tol) \
            & (p[:, 2] > c.floor_t + 0.005) & (p[:, 2] < c.floor_t + c.chamber_h - 0.01)

    def in_shaft(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool, LIVE: body centre inside the tower channel (descent begun)."""
        c = self.cfg
        p = self._vault_local(body.data.root_pos_w)
        si = c.shaft_in / 2
        return (p[:, 0].abs() < si) & (p[:, 1].abs() < si) \
            & (p[:, 2] > c.z_roof_top) & (p[:, 2] < c.z_mouth - 0.02)

    def settled_red(self) -> torch.Tensor:
        """(N,) bool: red ball linear AND angular velocity below the settle gates."""
        return (self.red.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin) \
            & (self.red.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([self.red.data.root_pos_w, self.blue.data.root_pos_w,
                         self.gate.data.root_pos_w, self.vault.data.root_pos_w], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        fin = self._finite()
        self._l_gate |= fin & self.pin_clear()
        self._l_chamber |= fin & self.in_chamber(self.red)
        self._l_entered |= fin & (self.in_shaft(self.red) | self._l_chamber)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the RED ball inside the chamber interior, settled, finite. A
        live physical outcome — a ball parked on the roof, on the gate, in the tower
        or anywhere outside judges False by geometry."""
        self._update_latches()
        return self.in_chamber(self.red) & self.settled_red() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.30*gate_out + 0.25*entered + 0.20*chamber (all
        latched; ~0 for doing nothing — a spanning gate and an outside ball latch
        nothing), capped at 0.75 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_gate * self._l_gate.float() + c.w_entered * self._l_entered.float()
                + c.w_chamber * self._l_chamber.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="gravity_vault", robot="null"))

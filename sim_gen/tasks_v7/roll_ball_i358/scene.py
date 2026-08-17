"""WindmillTollgateScene — a ROOFED courtyard with one doorway is patrolled by a
motor-driven two-blade windmill sweeping the interior; the red ball must come to
rest on the GREEN corner pad inside, and the mill must be SILENCED (held stalled
against its still-running motor) by threading a brake beam through a guide bore in
the right wall until it intrudes into the blade sweep. Derived from
maniskill/roll_ball, but the seed's entire plan (push a ball across open ground
into a flat goal region) is only the harmless last mile here; the load-bearing work
is defeating an ACTIVE mechanism that the seed has no concept of.

Seed (maniskill/roll_ball): one free ball on an open table, roll it into a marked
region — a single unobstructed conveyance. Strategic differences here:

- The goal region is INSIDE a walled, ROOFED courtyard whose only opening is a
  doorway — no crane-in from above, no straight-line push: the route is through the
  door, and the interior floor is tilted so gravity finishes the delivery.
- A two-blade WINDMILL on a free vertical axle spins over the interior, driven
  every step by a weak but tireless motor (velocity-servo torque, capped). Its
  sweep covers the doorway vestibule and the corridor to the pad: entrants get
  batted around indefinitely while it runs. Success REQUIRES the mill be stalled
  — |rotation| below a small bound over a 1 s window while the motor still drives —
  which no amount of ball-pushing achieves (the pad is outside the sweep circle, so
  a delivered ball cannot be the thing jamming it).
- The mill is silenced with a separate tool: a steel BRAKE BEAM threaded from
  OUTSIDE through a square guide bore in the right wall (exterior guide sleeve ->
  30 mm port -> interior sleeve) until its tip crosses the sweep circle; the
  advancing blade then slams into the beam flank and the motor stalls against it.
  The bore is at blade height, far above the rolling ball (which passes underneath),
  and is far too small (30 mm) for the 60 mm ball — the seed's object cannot do the
  beam's job, and the beam route cannot smuggle the ball in.
- Only then does the seed's skill matter: bowl the ball up the outside apron ramp,
  through the doorway; the tilted floor carries it along the stalled blade's flank,
  around its tip, into the downhill corner pad where it settles between the walls.

So the solver needs a different PLAN (disable an active mechanism with a non-target
tool, then deliver) and different CODE (peg-in-bore beam insertion at height +
stall detection + a gravity-assisted doorway bowl), not one flat push.

Assets are fully procedural (compound spawners; child colliders of one rigid body
never self-collide):
  - court: KINEMATIC compound — 4 walls (0.16 m tall, doorway in the front wall,
    bore through the right wall), full roof, tilted interior floor slab (2.3 deg,
    downhill toward the back-right corner), exterior apron ramp with guide rails,
    interior + exterior bore sleeves, and a GREEN goal pad (visual only — no
    collider, nothing to park on). Bound material restitution 0, min-combine.
  - mill: DYNAMIC blade+hub compound on a spawn-authored free RevoluteJoint
    (vertical axis at the court-local axle point). Driven in post_step by a
    torque-capped velocity servo (body-z torque — invariant under yaw, so the
    force-frame pod quirk cannot touch it). Spin direction is randomized.
  - beam: DYNAMIC 26 mm square steel bar, 200 mm long (fits the 80 mm jaw).
  - ball: DYNAMIC 60 mm red sphere (fits the 80 mm jaw; too big for the bore).

Per-episode randomization (readback-verifiable): mill initial angle, ball spawn on
the open ground in front of the apron, beam spawn pose on the ground right of the
courtyard, motor spin direction (readback via early angular velocity).

Rubric (0..1; latched partial credit, cleared on reset; ~0 for the null policy):
  0.25 engaged  — beam tip ever in the bore lane past the engagement depth
  0.25 silenced — mill ever stalled (pose-window) while the beam is engaged
  0.25 entered  — ball inside the courtyard while the mill is stalled
  1.0 iff success() — mill stalled now AND ball settled on the corner pad, finite.
  Non-success cap 0.75.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
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


# ----- pure-torch quaternion helpers (shared with solve.py / smoke.py) --------------------------
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


def _qz(yaw: torch.Tensor) -> torch.Tensor:
    """(N,) yaw -> (N, 4) wxyz quaternions about +z."""
    h = yaw * 0.5
    q = torch.zeros(yaw.shape[0], 4, device=yaw.device)
    q[:, 0] = torch.cos(h)
    q[:, 3] = torch.sin(h)
    return q


def encode_force(mode: int, q_ref: torch.Tensor, q_now: torch.Tensor,
                 f_world: torch.Tensor) -> torch.Tensor:
    """Pre-encode a desired WORLD-frame force for `set_external_force_and_torque`.

    Some pods rotate an applied wrench by the body's rotation since its reference
    orientation (applied = R_now * R_ref^T * arg). mode 0 passes the world force
    through unchanged; mode 1 pre-encodes with R_ref * R_now^T so the applied force
    comes out as the desired world force. Callers PROBE which mode moves the body
    the right way (critical for ROLLING bodies) and lock it in."""
    if mode == 0:
        return f_world
    return _qapply(_qmul(q_ref, _qinv(q_now)), f_world)


# ----- custom compound spawners ------------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _root_xform(prim_path: str, translation, orientation):
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


def _add_box(stage, path: str, *, center, size, color, collide: Callable | None,
             quat=None, density: float | None = None):
    """One box child: translate (+ optional orient) + scale, displayColor; collider
    only when `collide` is given (the goal pad is deliberately visual-only)."""
    from pxr import Gf, UsdGeom, UsdPhysics

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if quat is not None:
        w, x, y, z = (float(v) for v in quat)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide is not None:
        collide(box.GetPrim())
    if density is not None:
        UsdPhysics.MassAPI.Apply(box.GetPrim()).CreateDensityAttr(float(density))
    return box.GetPrim()


def _add_cyl(stage, path: str, *, center, radius, height, color, collide: Callable,
             density: float | None = None):
    """One z-axis cylinder child."""
    from pxr import Gf, UsdGeom, UsdPhysics

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr("Z")
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())
    if density is not None:
        UsdPhysics.MassAPI.Apply(cyl.GetPrim()).CreateDensityAttr(float(density))
    return cyl.GetPrim()


def _spawn_court(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the courtyard: KINEMATIC compound. Origin at the courtyard centre on
    the ground; +x right, +y back. Walls 0.16 m tall with a doorway in the front
    wall and a square bore through the right wall; full roof; interior floor slab
    tilted `tilt_deg` downhill toward (+x, +y); interior + exterior bore sleeves;
    exterior apron ramp with rails; GREEN goal pad (visual only, no collider)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    wi, wt, wh = c.wall_in, c.wall_t, c.wall_h        # 0.17, 0.03, 0.16
    wc = wi + wt / 2                                   # wall centreline 0.185
    wall, steel = c.wall_color, c.steel_color

    # front wall (y = -wc) with doorway x in [door_x0, door_x1], full height
    _add_box(stage, f"{prim_path}/w_front_l",
             center=((-wi - wt + c.door_x0) / 2, -wc, wh / 2),
             size=(c.door_x0 + wi + wt, wt, wh), color=wall, collide=collide)
    _add_box(stage, f"{prim_path}/w_front_r",
             center=((c.door_x1 + wi + wt) / 2, -wc, wh / 2),
             size=(wi + wt - c.door_x1, wt, wh), color=wall, collide=collide)
    # back and left walls: full runs
    _add_box(stage, f"{prim_path}/w_back", center=(0.0, wc, wh / 2),
             size=(2 * wi + 2 * wt, wt, wh), color=wall, collide=collide)
    _add_box(stage, f"{prim_path}/w_left", center=(-wc, 0.0, wh / 2),
             size=(wt, 2 * wi + 2 * wt, wh), color=wall, collide=collide)
    # right wall (x = +wc) with the bore: lane y = chan_y, square inner chan_half
    cy, ch = c.chan_y, c.chan_half                     # -0.04, 0.015
    z0, z1 = c.chan_z0, c.chan_z1                      # 0.09, 0.12
    _add_box(stage, f"{prim_path}/w_right_lo", center=(wc, 0.0, z0 / 2),
             size=(wt, 2 * wi + 2 * wt, z0), color=wall, collide=collide)
    _add_box(stage, f"{prim_path}/w_right_hi", center=(wc, 0.0, (z1 + wh) / 2),
             size=(wt, 2 * wi + 2 * wt, wh - z1), color=wall, collide=collide)
    _add_box(stage, f"{prim_path}/w_right_s1",
             center=(wc, (-wi - wt + cy - ch) / 2, (z0 + z1) / 2),
             size=(wt, cy - ch + wi + wt, z1 - z0), color=wall, collide=collide)
    _add_box(stage, f"{prim_path}/w_right_s2",
             center=(wc, (cy + ch + wi + wt) / 2, (z0 + z1) / 2),
             size=(wt, wi + wt - cy - ch, z1 - z0), color=wall, collide=collide)
    # roof
    _add_box(stage, f"{prim_path}/roof", center=(0.0, 0.0, wh + c.roof_t / 2),
             size=(2 * wi + 2 * wt + 0.02, 2 * wi + 2 * wt + 0.02, c.roof_t),
             color=c.roof_color, collide=collide)

    # tilted interior floor slab: top plane z = slab_top_c - kx * (x + y)
    a = math.radians(c.tilt_deg)
    h = a / 2
    s2 = math.sqrt(2.0)
    # rotation about the horizontal axis (-1, 1, 0)/sqrt(2) by `a`: the surface
    # normal tilts toward (+x, +y), so the top plane DESCENDS toward (+x, +y) —
    # z_top = slab_top_c - kx*(x + y), matching the __post_init__ audit convention
    q_slab = (math.cos(h), -math.sin(h) / s2, math.sin(h) / s2, 0.0)
    _add_box(stage, f"{prim_path}/slab",
             center=(0.0, 0.0, c.slab_top_c - c.slab_t / 2 * math.cos(a)),
             size=(c.slab_len, c.slab_len, c.slab_t), color=c.slab_color,
             collide=collide, quat=q_slab)
    # GREEN goal pad: VISUAL ONLY (no collider — a flush collider would be a wall
    # to a rolling ball). Sunk so only ~1 mm stands proud of the slab.
    kx = math.tan(a) / s2
    pad_z = c.slab_top_c - kx * (c.pad_cx + c.pad_cy)
    _add_box(stage, f"{prim_path}/pad",
             center=(c.pad_cx, c.pad_cy, pad_z - 0.001),
             size=(2 * c.pad_half, 2 * c.pad_half, 0.004),
             color=(0.10, 0.75, 0.20), collide=None, quat=q_slab)

    # interior bore sleeve: inner square chan_half about (y=cy, z mid), x in
    # [sleeve_in_x0, wall_in] — supports the cantilevered beam near the sweep
    sx0, sx1 = c.sleeve_in_x0, wi
    sxc, sxl = (sx0 + sx1) / 2, sx1 - sx0
    zm = (z0 + z1) / 2
    st = 0.012
    _add_box(stage, f"{prim_path}/sl_in_bot", center=(sxc, cy, z0 - st / 2),
             size=(sxl, 2 * ch + 2 * st, st), color=steel, collide=collide)
    _add_box(stage, f"{prim_path}/sl_in_top", center=(sxc, cy, z1 + st / 2),
             size=(sxl, 2 * ch + 2 * st, st), color=steel, collide=collide)
    _add_box(stage, f"{prim_path}/sl_in_yn", center=(sxc, cy - ch - st / 2, zm),
             size=(sxl, st, z1 - z0), color=steel, collide=collide)
    _add_box(stage, f"{prim_path}/sl_in_yp", center=(sxc, cy + ch + st / 2, zm),
             size=(sxl, st, z1 - z0), color=steel, collide=collide)
    # exterior guide sleeve: slightly larger inner square (mouth_half) so hand
    # insertion has slack; x in [wall_out, wall_out + sleeve_out_len]
    mo = c.mouth_half
    ox0 = wi + wt
    oxc, oxl = ox0 + c.sleeve_out_len / 2, c.sleeve_out_len
    _add_box(stage, f"{prim_path}/sl_out_bot", center=(oxc, cy, zm - mo - st / 2),
             size=(oxl, 2 * mo + 2 * st, st), color=steel, collide=collide)
    _add_box(stage, f"{prim_path}/sl_out_top", center=(oxc, cy, zm + mo + st / 2),
             size=(oxl, 2 * mo + 2 * st, st), color=steel, collide=collide)
    _add_box(stage, f"{prim_path}/sl_out_yn", center=(oxc, cy - mo - st / 2, zm),
             size=(oxl, st, 2 * mo), color=steel, collide=collide)
    _add_box(stage, f"{prim_path}/sl_out_yp", center=(oxc, cy + mo + st / 2, zm),
             size=(oxl, st, 2 * mo), color=steel, collide=collide)

    # exterior apron ramp: top runs z=0 at y = -(apron_y1 + apron_len) up to
    # apron_top at y = -apron_y1 (under the doorway, ~1 mm proud of the slab seam)
    th = math.atan2(c.apron_top, c.apron_len)
    hh = th / 2
    q_ap = (math.cos(hh), math.sin(hh), 0.0, 0.0)      # about +x: +y end rises
    yc = -(c.apron_y1 + c.apron_len / 2)
    zc = c.apron_top / 2
    n = (0.0, -math.sin(th), math.cos(th))
    axc = (c.door_x0 + c.door_x1) / 2
    _add_box(stage, f"{prim_path}/apron",
             center=(axc, yc - n[1] * c.apron_t / 2, zc - n[2] * c.apron_t / 2),
             size=(c.apron_w, c.apron_len / math.cos(th), c.apron_t),
             color=c.slab_color, collide=collide, quat=q_ap)
    for sgn, nm in ((1.0, "rail_p"), (-1.0, "rail_n")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(axc + sgn * (c.apron_w / 2 - 0.006),
                         yc + n[1] * 0.02, zc + n[2] * 0.02),
                 size=(0.012, c.apron_len / math.cos(th), 0.05),
                 color=wall, collide=collide, quat=q_ap)

    # bound material: restitution 0, min-combine (kills bounce on every contact)
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    sim_utils.spawn_rigid_body_material(
        f"{prim_path}/physmat",
        sim_utils.RigidBodyMaterialCfg(
            static_friction=float(c.mu_static), dynamic_friction=float(c.mu_dynamic),
            restitution=0.0, friction_combine_mode="min",
            restitution_combine_mode="min"))
    bind_physics_material(prim_path, f"{prim_path}/physmat")
    return root


def _spawn_mill(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The windmill: DYNAMIC blade+hub compound, body origin ON the axle axis at
    ground level, plus the spawn-authored free RevoluteJoint (vertical axis) to the
    sibling court. Per-child DENSITY gives the true rotor inertia."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.2)
    pxrb.CreateAngularDampingAttr(float(c.ang_damping))
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    zc = (c.blade_z0 + c.blade_z1) / 2
    _add_box(stage, f"{prim_path}/blade",
             center=(0.0, 0.0, zc),
             size=(2 * c.sweep_r, c.blade_t, c.blade_z1 - c.blade_z0),
             color=c.blade_color, collide=collide, density=c.blade_density)
    # hub: radius == blade half-thickness, so it stays FLUSH with the blade faces
    # (a prouder hub would snag the ball rolling along a stalled blade's flank)
    _add_cyl(stage, f"{prim_path}/hub", center=(0.0, 0.0, zc),
             radius=c.blade_t / 2, height=c.blade_z1 - c.blade_z0 + 0.03,
             color=(0.25, 0.25, 0.28), collide=collide, density=c.blade_density)

    # free revolute joint to the sibling court: vertical axis through the axle
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/axle")
    j.CreateBody0Rel().SetTargets([f"{base}/Court"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateCollisionEnabledAttr(True)
    j.CreateAxisAttr("Z")
    j.CreateLocalPos0Attr(Gf.Vec3f(float(c.axle_x), float(c.axle_y), zc))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, zc))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    # no limits: a free, continuously rotating axle

    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    sim_utils.spawn_rigid_body_material(
        f"{prim_path}/physmat",
        sim_utils.RigidBodyMaterialCfg(
            static_friction=0.35, dynamic_friction=0.30, restitution=0.0,
            friction_combine_mode="min", restitution_combine_mode="min"))
    bind_physics_material(prim_path, f"{prim_path}/physmat")
    return root


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The brake beam: DYNAMIC square steel bar, long axis local +x, origin at the
    bar centre. Mass via density (true CoM/inertia)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.2)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    _add_box(stage, f"{prim_path}/bar", center=(0.0, 0.0, 0.0),
             size=(c.beam_len, c.beam_s, c.beam_s), color=c.beam_color,
             collide=collide, density=c.beam_density)
    return root


def _spawn_ball(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The red ball: DYNAMIC sphere. Damping so it rolls to rest; sleep thresholds
    zeroed; velocity iterations 4 (kills the GPU sphere phantom-creep artifact)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.10)
    pxrb.CreateAngularDampingAttr(0.50)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    r = float(cfg.radius)
    sph = UsdGeom.Sphere.Define(stage, f"{prim_path}/ball")
    sph.CreateRadiusAttr(r)
    sph.CreateExtentAttr([Gf.Vec3f(-r, -r, -r), Gf.Vec3f(r, r, r)])
    sph.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    _make_collide(cfg.contact_offset)(sph.GetPrim())
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "court" not in _SPAWNER_CACHE:

        @configclass
        class CourtSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_court)
            wall_in: float = 0.17
            wall_t: float = 0.03
            wall_h: float = 0.16
            roof_t: float = 0.03
            door_x0: float = -0.11
            door_x1: float = 0.0
            chan_y: float = -0.04
            chan_half: float = 0.015
            chan_z0: float = 0.09
            chan_z1: float = 0.12
            sleeve_in_x0: float = 0.10
            sleeve_out_len: float = 0.05
            mouth_half: float = 0.017
            tilt_deg: float = 2.3
            slab_top_c: float = 0.014
            slab_t: float = 0.02
            slab_len: float = 0.36
            pad_cx: float = 0.125
            pad_cy: float = 0.125
            pad_half: float = 0.045
            apron_y1: float = 0.175
            apron_len: float = 0.16
            apron_top: float = 0.023
            apron_w: float = 0.20
            apron_t: float = 0.012
            mu_static: float = 0.35
            mu_dynamic: float = 0.30
            wall_color: tuple = (0.52, 0.50, 0.46)
            roof_color: tuple = (0.35, 0.34, 0.32)
            slab_color: tuple = (0.72, 0.68, 0.60)
            steel_color: tuple = (0.55, 0.57, 0.60)
            contact_offset: float = 0.0015

        @configclass
        class MillSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_mill)
            sweep_r: float = 0.112
            blade_t: float = 0.024
            blade_z0: float = 0.056
            blade_z1: float = 0.128
            blade_density: float = 400.0
            ang_damping: float = 0.3
            axle_x: float = -0.04
            axle_y: float = -0.04
            blade_color: tuple = (0.90, 0.55, 0.10)
            contact_offset: float = 0.0015

        @configclass
        class BeamSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beam)
            beam_len: float = 0.20
            beam_s: float = 0.026
            beam_density: float = 7800.0
            beam_color: tuple = (0.20, 0.35, 0.85)
            contact_offset: float = 0.0015

        @configclass
        class BallSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_ball)
            radius: float = 0.030
            mass: float = 0.25
            color: tuple = (0.85, 0.10, 0.10)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE.update(court=CourtSpawnerCfg, mill=MillSpawnerCfg,
                              beam=BeamSpawnerCfg, ball=BallSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg ---------------------------------------------------------------------------------
@dataclass
class WindmillTollgateSceneCfg(BaseCfg):
    """Config for `WindmillTollgateScene`. The gate geometry is honest by
    construction — every clause is asserted numerically in __post_init__: the ball
    fits the doorway and the corridor around a stalled blade but NOT the bore; the
    blade clears every wall, the roof, a floor-lying beam, and the sleeves, yet
    strikes both the rolling ball (low) and the inserted beam (high); the pad is
    outside the sweep circle so a delivered ball can never be what stalls the mill."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    stall_window: int = tunable(120)      # steps per stall-check window (1 s @ 120 Hz)
    stall_max_deg: float = tunable(2.0)   # |yaw change| per window at/below = stalled
    settle_speed: float = tunable(0.08)   # max ball |lin vel| when judging (m/s)
    pad_lo: float = tunable(0.085)        # pad band (court-local x AND y), ball centre
    pad_hi: float = tunable(0.165)
    pad_z_hi: float = tunable(0.09)       # ball centre below this = on the floor
    engage_x: float = tunable(0.065)      # beam tip court-x at/below = engaged
    lane_y_tol: float = tunable(0.035)    # beam-tip lane tolerance about chan_y
    lane_z0: float = tunable(0.075)       # beam-tip lane z band
    lane_z1: float = tunable(0.135)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    ball_x_rng: tuple = tunable((-0.12, 0.01))    # ball spawn (court-local)
    ball_y_rng: tuple = tunable((-0.46, -0.38))
    beam_x_rng: tuple = tunable((0.315, 0.38))    # beam spawn (court-local)
    beam_y_rng: tuple = tunable((0.04, 0.15))
    beam_yaw_max: float = tunable(0.25)           # beam spawn yaw (+/- rad)

    # --- info: motor -----------------------------------------------------------------------------
    drive_omega: float = info(2.5)    # motor target speed (rad/s; sign randomized)
    drive_k: float = info(0.05)       # velocity-servo gain (N*m*s)
    drive_tau_max: float = info(0.05)  # motor torque cap (N*m)
    ang_damping: float = info(0.3)

    # --- info: courtyard layout (single Franka base near the world origin) -----------------------
    court_x: float = info(0.38)       # courtyard centre (world)
    court_y: float = info(0.06)
    wall_in: float = info(0.17)       # interior half extent
    wall_t: float = info(0.03)
    wall_h: float = info(0.16)
    roof_t: float = info(0.03)
    door_x0: float = info(-0.11)      # doorway span (front wall, court-local x)
    door_x1: float = info(0.0)
    tilt_deg: float = info(2.3)       # floor tilt, downhill toward (+x, +y)
    slab_top_c: float = info(0.014)   # slab top height at the courtyard centre
    slab_t: float = info(0.02)
    slab_len: float = info(0.36)
    pad_cx: float = info(0.125)       # green pad centre
    pad_cy: float = info(0.125)
    pad_half: float = info(0.045)
    apron_y1: float = info(0.175)     # apron high end |y| (runs under the doorway)
    apron_len: float = info(0.16)
    apron_top: float = info(0.023)
    apron_w: float = info(0.20)
    mu_static: float = info(0.35)
    mu_dynamic: float = info(0.30)

    # --- info: bore / sleeves --------------------------------------------------------------------
    chan_y: float = info(-0.04)       # bore lane (court-local y; aims at the axle)
    chan_half: float = info(0.015)    # 30 mm square bore
    chan_z0: float = info(0.09)
    chan_z1: float = info(0.12)
    sleeve_in_x0: float = info(0.10)  # interior sleeve reach (toward the axle)
    sleeve_out_len: float = info(0.05)
    mouth_half: float = info(0.017)   # 34 mm exterior mouth (8 mm slack on the beam)

    # --- info: mill ------------------------------------------------------------------------------
    axle_x: float = info(-0.04)       # axle (court-local)
    axle_y: float = info(-0.04)
    sweep_r: float = info(0.112)      # blade half length = sweep radius
    blade_t: float = info(0.024)
    blade_z0: float = info(0.056)
    blade_z1: float = info(0.128)
    blade_density: float = info(400.0)

    # --- info: free bodies -----------------------------------------------------------------------
    beam_len: float = info(0.20)      # 26 mm square bar — fits the 80 mm jaw
    beam_s: float = info(0.026)
    beam_density: float = info(7800.0)
    ball_r: float = info(0.030)       # 60 mm ball — fits the jaw, NOT the bore
    ball_mass: float = info(0.25)

    contact_offset: float = info(0.0015)
    # rubric weights (0.25 + 0.25 + 0.25 = 0.75 = the non-success cap)
    w_engage: float = info(0.25)
    w_silence: float = info(0.25)
    w_enter: float = info(0.25)

    def __post_init__(self) -> None:
        """Audit the toll-gate geometry (metres)."""
        kx = math.tan(math.radians(self.tilt_deg)) / math.sqrt(2.0)
        ball_d = 2 * self.ball_r

        def slab_top(x: float, y: float) -> float:
            return self.slab_top_c - kx * (x + y)

        # doorway passes the ball; the bore does NOT (the ball cannot do the beam's job)
        assert self.door_x1 - self.door_x0 > ball_d + 0.02
        assert self.wall_h > ball_d + 0.02
        assert 2 * self.chan_half < ball_d - 0.01
        # blade clears every wall at every angle
        for d in (self.wall_in + self.axle_x, self.wall_in - self.axle_x,
                  self.wall_in + self.axle_y, self.wall_in - self.axle_y):
            assert d - self.sweep_r >= 0.012, (d, self.sweep_r)
        # blade clears the roof and the highest floor point inside the sweep
        assert self.wall_h - self.blade_z1 >= 0.02
        floor_hi = slab_top(self.axle_x - self.sweep_r / math.sqrt(2),
                            self.axle_y - self.sweep_r / math.sqrt(2))
        assert self.blade_z0 - floor_hi >= 0.03
        # blade clears a beam LYING anywhere on the slab (a dropped beam cannot
        # trivially stall the mill from the floor)
        assert self.blade_z0 - (floor_hi + self.beam_s) >= 0.005
        # ...yet strikes the rolling ball everywhere in the sweep (lowest floor pt)
        floor_lo = slab_top(self.axle_x + self.sweep_r / math.sqrt(2),
                            self.axle_y + self.sweep_r / math.sqrt(2))
        assert (floor_lo + ball_d) - self.blade_z0 >= 0.010
        # ...and overlaps the inserted beam's z-band by >= 20 mm
        beam_zc = (self.chan_z0 + self.chan_z1) / 2
        z_over = min(self.blade_z1, beam_zc + self.beam_s / 2) \
            - max(self.blade_z0, beam_zc - self.beam_s / 2)
        assert z_over >= 0.020, z_over
        # the bore lane crosses the sweep circle with real depth, and the interior
        # sleeve tip stays clear of the sweep
        dy = abs(self.chan_y - self.axle_y)
        assert dy < self.sweep_r - 0.03
        x_sweep = self.axle_x + math.sqrt(self.sweep_r ** 2 - dy ** 2)
        assert x_sweep - self.engage_x >= 0.005, (x_sweep, self.engage_x)
        assert self.sleeve_in_x0 - x_sweep >= 0.02
        # beam long enough: tip at engage_x keeps the tail supported in the sleeves
        assert self.engage_x + self.beam_len >= self.wall_in + 0.03
        # bore slack sane; exterior mouth passes the beam with hand slack
        assert 0.003 <= 2 * self.chan_half - self.beam_s <= 0.008
        assert 2 * self.mouth_half - self.beam_s >= 0.006
        # the ball passes UNDER the interior sleeve (floor there is downhill)
        z_ball_top = slab_top(self.sleeve_in_x0, self.chan_y) + ball_d
        assert (self.chan_z0 - 0.012) - z_ball_top >= 0.004
        # corridor around a stalled blade: blade flank to front wall, and blade tip
        # to right wall, both pass the ball
        assert (self.axle_y - self.blade_t / 2) - (-self.wall_in) > ball_d + 0.02
        assert self.wall_in - x_sweep > ball_d + 0.015
        # pad band lies OUTSIDE the sweep circle (a delivered ball cannot stall the
        # mill) and inside the walls
        d_pad = math.hypot(self.pad_lo - self.axle_x, self.pad_lo - self.axle_y)
        assert d_pad - self.ball_r > self.sweep_r + 0.01
        assert self.pad_hi <= self.wall_in
        # jaw feasibility: beam and ball both fit the 80 mm parallel jaw
        assert self.beam_s < 0.080 and ball_d < 0.080
        # apron seam stands proud of the slab across the whole doorway (no step UP)
        for x in (self.door_x0, self.door_x1):
            assert self.apron_top >= slab_top(x, -self.apron_y1) - 0.0005, x
        # stall detection cannot alias: a free mill turns far more than the stall
        # bound per window, far less than a full turn
        turn = self.drive_omega * self.stall_window / 120.0
        assert math.radians(self.stall_max_deg) * 10 < turn < math.radians(300.0)


# ----- scene -------------------------------------------------------------------------------------
@SCENES.register("windmill_tollgate")
class WindmillTollgateScene(BaseScene):
    cfg: WindmillTollgateSceneCfg

    def __init__(self, cfg: WindmillTollgateSceneCfg | None = None) -> None:
        super().__init__(cfg or WindmillTollgateSceneCfg())

    # ----- assets --------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        court_spawn = cls["court"](
            mass_props=sim_utils.MassPropertiesCfg(mass=80.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            wall_in=c.wall_in, wall_t=c.wall_t, wall_h=c.wall_h, roof_t=c.roof_t,
            door_x0=c.door_x0, door_x1=c.door_x1, chan_y=c.chan_y,
            chan_half=c.chan_half, chan_z0=c.chan_z0, chan_z1=c.chan_z1,
            sleeve_in_x0=c.sleeve_in_x0, sleeve_out_len=c.sleeve_out_len,
            mouth_half=c.mouth_half, tilt_deg=c.tilt_deg, slab_top_c=c.slab_top_c,
            slab_t=c.slab_t, slab_len=c.slab_len, pad_cx=c.pad_cx, pad_cy=c.pad_cy,
            pad_half=c.pad_half, apron_y1=c.apron_y1, apron_len=c.apron_len,
            apron_top=c.apron_top, apron_w=c.apron_w, mu_static=c.mu_static,
            mu_dynamic=c.mu_dynamic, contact_offset=c.contact_offset)
        mill_spawn = cls["mill"](
            sweep_r=c.sweep_r, blade_t=c.blade_t, blade_z0=c.blade_z0,
            blade_z1=c.blade_z1, blade_density=c.blade_density,
            ang_damping=c.ang_damping, axle_x=c.axle_x, axle_y=c.axle_y,
            contact_offset=c.contact_offset)
        beam_spawn = cls["beam"](
            beam_len=c.beam_len, beam_s=c.beam_s, beam_density=c.beam_density,
            contact_offset=c.contact_offset)
        ball_spawn = cls["ball"](
            radius=c.ball_r, mass=c.ball_mass, contact_offset=c.contact_offset)

        px, py = c.court_x, c.court_y
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
            # the court NEVER moves (its pose is not randomized): the mill's joint
            # anchor on a kinematic body0 stays world-fixed, so a court teleport
            # would tear the axle. All randomization is on the free bodies + rotor.
            "court": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Court",
                spawn=court_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0)),
            ),
            "mill": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Mill",
                spawn=mill_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(px + c.axle_x, py + c.axle_y, 0.0)),
            ),
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beam",
                spawn=beam_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(px + 0.35, py + 0.10, 0.015)),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=ball_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(px - 0.06, py - 0.42, 0.031)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # the motor (and solve/smoke pushes) run through
                # set_external_force_and_torque; without this flag wrenches are
                # under-applied across TGS iterations
                "enable_external_forces_every_iteration": True,
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
        self.court: RigidObject = env.iscene["court"]
        self.mill: RigidObject = env.iscene["mill"]
        self.beam: RigidObject = env.iscene["beam"]
        self.ball: RigidObject = env.iscene["ball"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self.spin_sign = torch.ones(n, device=dev)     # motor direction (+/-1)
        # stall detector: pose-window (velocity readbacks have phantom spikes)
        self._yaw_anchor = torch.zeros(n, device=dev)
        self._win_cnt = torch.zeros(n, dtype=torch.long, device=dev)
        self._stalled = torch.zeros(n, dtype=torch.bool, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._engaged = torch.zeros(n, dtype=torch.bool, device=dev)
        self._silenced = torch.zeros(n, dtype=torch.bool, device=dev)
        self._entered = torch.zeros(n, dtype=torch.bool, device=dev)

    def _court_pos(self, env_ids: torch.Tensor | slice) -> torch.Tensor:
        c = self.cfg
        p = self.env_origins[env_ids].clone()
        p[:, 0] += c.court_x
        p[:, 1] += c.court_y
        return p

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: spin the rotor to a random angle about its FIXED axle (the
        rotor origin lies on the axis, so any yaw is a pure pose write consistent
        with the joint), scatter the ball on the ground before the apron and the
        beam on the ground right of the courtyard, flip the motor direction coin,
        clear the latches and the stall detector."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        base = self._court_pos(env_ids)

        _ = torch.rand(m, device=dev)  # burn: first post-seed draw is degenerate
        yaw0 = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = base[:, 0] + c.axle_x
        st[:, 1] = base[:, 1] + c.axle_y
        st[:, 2] = base[:, 2]
        st[:, 3:7] = _qz(yaw0)
        self.mill.write_root_state_to_sim(st, env_ids)

        bx = c.ball_x_rng[0] + torch.rand(m, device=dev) \
            * (c.ball_x_rng[1] - c.ball_x_rng[0])
        by = c.ball_y_rng[0] + torch.rand(m, device=dev) \
            * (c.ball_y_rng[1] - c.ball_y_rng[0])
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = base[:, 0] + bx
        st[:, 1] = base[:, 1] + by
        st[:, 2] = base[:, 2] + c.ball_r + 0.001
        st[:, 3] = 1.0
        self.ball.write_root_state_to_sim(st, env_ids)

        kx = c.beam_x_rng[0] + torch.rand(m, device=dev) \
            * (c.beam_x_rng[1] - c.beam_x_rng[0])
        ky = c.beam_y_rng[0] + torch.rand(m, device=dev) \
            * (c.beam_y_rng[1] - c.beam_y_rng[0])
        kyaw = (torch.rand(m, device=dev) * 2 - 1) * c.beam_yaw_max
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = base[:, 0] + kx
        st[:, 1] = base[:, 1] + ky
        st[:, 2] = base[:, 2] + c.beam_s / 2 + 0.002
        st[:, 3:7] = _qz(kyaw)
        self.beam.write_root_state_to_sim(st, env_ids)

        # motor direction: coin AFTER the continuous draws (first-draw degeneracy)
        self.spin_sign[env_ids] = torch.where(
            torch.rand(m, device=dev) < 0.5,
            torch.ones(m, device=dev), -torch.ones(m, device=dev))

        self._yaw_anchor[env_ids] = yaw0
        self._win_cnt[env_ids] = 0
        self._stalled[env_ids] = False
        self._engaged[env_ids] = False
        self._silenced[env_ids] = False
        self._entered[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "mill": self.mill.data.root_state_w[env_ids].clone(),
            "beam": self.beam.data.root_state_w[env_ids].clone(),
            "ball": self.ball.data.root_state_w[env_ids].clone(),
            "spin_sign": self.spin_sign[env_ids].clone(),
            "yaw_anchor": self._yaw_anchor[env_ids].clone(),
            "win_cnt": self._win_cnt[env_ids].clone(),
            "stalled": self._stalled[env_ids].clone(),
            "engaged": self._engaged[env_ids].clone(),
            "silenced": self._silenced[env_ids].clone(),
            "entered": self._entered[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.mill.write_root_state_to_sim(state["mill"], env_ids)
        self.beam.write_root_state_to_sim(state["beam"], env_ids)
        self.ball.write_root_state_to_sim(state["ball"], env_ids)
        self.spin_sign[env_ids] = state["spin_sign"]
        self._yaw_anchor[env_ids] = state["yaw_anchor"]
        self._win_cnt[env_ids] = state["win_cnt"]
        self._stalled[env_ids] = state["stalled"]
        self._engaged[env_ids] = state["engaged"]
        self._silenced[env_ids] = state["silenced"]
        self._entered[env_ids] = state["entered"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A square walled COURTYARD (interior 340 x 340 mm, walls 160 mm tall, "
            "30 mm thick) stands on the ground, fully covered by a ROOF — the only "
            "way in is a 110 mm-wide DOORWAY in the front wall (its left portion). "
            "An APRON RAMP with low side rails climbs the outside ground up through "
            "the doorway. The interior floor is a slab tilted 2.3 degrees, downhill "
            "toward the BACK-RIGHT corner, where a GREEN PAD (90 x 90 mm, flush "
            "with the floor) marks the goal. Inside, a two-blade WINDMILL — one "
            "orange bar, 224 mm tip to tip, spanning heights 56-128 mm — turns on "
            "a free vertical axle offset toward the door-side corner. A motor "
            "drives it continuously (direction varies per episode) toward 2.5 rad/s "
            "with a small capped torque: left alone it NEVER stops, and its sweep "
            "covers the doorway vestibule and the corridor to the pad, batting "
            "anything that enters. The blade passes above a bar lying on the floor "
            "and below the wall bore, but square through both a rolling ball and "
            "an inserted bar. Through the RIGHT wall, at blade height (90-120 mm), "
            "runs a square guide BORE (30 mm inner) aimed at the axle: a steel "
            "exterior mouth (34 mm) and an interior sleeve guide a bar pushed in "
            "from outside. A blue steel BRAKE BEAM (26 x 26 x 200 mm) lies on the "
            "ground right of the courtyard, and a red BALL (60 mm diameter — too "
            "fat for the bore) rests on the open ground in front of the apron. "
            "Their spawn spots and the mill's angle vary per episode.\n"
            "Goal: SILENCE the mill and DELIVER the ball. Push the brake beam "
            "through the bore until its tip crosses the blade circle — the next "
            "blade pass slams into the beam's flank and the motor stalls against "
            "it (the mill counts as silenced only while its rotation stays within "
            f"{c.stall_max_deg:.0f} degrees over a full second, motor still "
            "driving). Then bowl the ball up the apron through the doorway: the "
            "tilted floor carries it along the stalled blade's flank, around the "
            "blade tip, and down into the back-right corner. Success: the mill "
            "stalled AND the ball at rest ON the green pad (court-local x and y "
            f"both in [{c.pad_lo:.3f}, {c.pad_hi:.3f}] m). Either order works, "
            "but while the mill runs, the blade sweeps the delivery corridor — "
            "and a ball on the pad sits OUTSIDE the blade circle, so parking the "
            "ball can never be what stops the mill."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Stop the windmill inside the roofed courtyard: slide the blue brake "
            "beam into the square bore in the right wall until the blade jams "
            "against it and the mill stalls. Then bowl the red ball up the apron "
            "ramp through the doorway so it rolls down the tilted floor, around "
            "the stalled blade, and rests on the green corner pad."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World -> court-local (the court never moves and has identity yaw)."""
        return pos_w - self._court_pos(slice(None))

    def mill_yaw(self) -> torch.Tensor:
        """(N,) rotor yaw (rad, wrapped) — the joint has no state API; this is the
        hinge readout from the root quaternion (yaw-only body)."""
        q = self.mill.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 3], q[:, 0])

    @staticmethod
    def _wrap(a: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(a), torch.cos(a))

    def beam_tip_local(self) -> torch.Tensor:
        """(N, 3) court-local position of the beam end nearer the axle (min-x end)."""
        c = self.cfg
        half = torch.zeros(self.env.num_envs, 3, device=self.env.device)
        half[:, 0] = c.beam_len / 2
        ax = _qapply(self.beam.data.root_quat_w, half)
        p = self.beam.data.root_pos_w
        e1 = self._local(p + ax)
        e2 = self._local(p - ax)
        return torch.where((e1[:, 0] < e2[:, 0]).unsqueeze(-1), e1, e2)

    def beam_engaged_now(self) -> torch.Tensor:
        """(N,) bool: beam tip in the bore lane, past the engagement depth (inside
        the sweep circle)."""
        c = self.cfg
        tip = self.beam_tip_local()
        return ((tip[:, 1] - c.chan_y).abs() < c.lane_y_tol) \
            & (tip[:, 2] > c.lane_z0) & (tip[:, 2] < c.lane_z1) \
            & (tip[:, 0] < c.engage_x)

    def ball_on_pad(self) -> torch.Tensor:
        c = self.cfg
        p = self._local(self.ball.data.root_pos_w)
        return (p[:, 0] > c.pad_lo) & (p[:, 0] < c.pad_hi) \
            & (p[:, 1] > c.pad_lo) & (p[:, 1] < c.pad_hi) \
            & (p[:, 2] < c.pad_z_hi)

    def ball_inside(self) -> torch.Tensor:
        c = self.cfg
        p = self._local(self.ball.data.root_pos_w)
        return (p[:, 0].abs() < c.wall_in - 0.005) \
            & (p[:, 1].abs() < c.wall_in - 0.005) & (p[:, 2] < c.wall_h - 0.02)

    def stalled(self) -> torch.Tensor:
        """(N,) bool: the mill is currently held stalled (pose-window verdict)."""
        return self._stalled

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w
                         for b in (self.mill, self.beam, self.ball)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Drive the motor (every step) and advance the stall detector + latches."""
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        # motor: torque-capped velocity servo about body z (yaw-invariant frame)
        w = self.mill.data.root_ang_vel_w[:, 2]
        tau = (c.drive_k * (self.spin_sign * c.drive_omega - w)) \
            .clamp(-c.drive_tau_max, c.drive_tau_max)
        t3 = torch.zeros(n, 1, 3, device=dev)
        t3[:, 0, 2] = tau
        self.mill.set_external_force_and_torque(
            forces=torch.zeros(n, 1, 3, device=dev), torques=t3)

        fin = self._finite()
        # stall detector: yaw progress over a pose window (teleport-safe: reset
        # re-anchors; velocity phantom spikes cannot fake or break it)
        self._win_cnt += 1
        yaw = self.mill_yaw()
        done = self._win_cnt >= c.stall_window
        if bool(done.any()):
            dyaw = self._wrap(yaw - self._yaw_anchor).abs()
            verdict = dyaw < math.radians(c.stall_max_deg)
            self._stalled = torch.where(done & fin, verdict, self._stalled)
            self._yaw_anchor = torch.where(done, yaw, self._yaw_anchor)
            self._win_cnt = torch.where(done, torch.zeros_like(self._win_cnt),
                                        self._win_cnt)
        # latches
        eng = self.beam_engaged_now() & fin
        self._engaged |= eng
        self._silenced |= self._stalled & eng
        self._entered |= self._stalled & self.ball_inside() & fin

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the mill is held stalled against its running motor AND the
        red ball rests on the green pad, everything finite. Both clauses are live
        physical outcomes: the pad lies outside the sweep circle, so the ball can
        never be the obstruction — silencing takes the beam."""
        c = self.cfg
        still = self.ball.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return self._stalled & self.ball_on_pad() & still & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.25*engaged + 0.25*silenced + 0.25*entered (all
        latched; ~0 for the null policy — the mill spins forever on its own and
        both free bodies start outside), capped at 0.75 — and exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        base = (c.w_engage * self._engaged.float()
                + c.w_silence * self._silenced.float()
                + c.w_enter * self._entered.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="windmill_tollgate", robot="null"))

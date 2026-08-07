"""KeystoneCascadeScene — stand the tall red keystone on the trigger pad, poke it over,
and let its fall run a guarded domino cascade that ends with a ball knocked into a
walled pit (sim_gen task `poke_cube_i83`).

Derived from maniskill/poke_cube, but STRATEGICALLY different: the seed grasps a peg
tool and pokes a red cube a few centimeters across the plane into a painted goal
region — one planar push, judged by an xy readout on the poked object itself. Here
the poked object is not the goal at all, and the poke is not transport: the keystone
must first be PLACED standing on a trigger pad (a latched precondition read while the
whole downstream run is still intact), and the poke's only job is to convert the
stored potential energy of placement into a CAUSAL CHAIN the robot can never touch:
the falling keystone bridges through the mouth of a tunnel too narrow for the hand,
topples a six-domino run inside it, the last domino fells a tall hammer piece, the
hammer swings through a letterbox hole in a walled kiosk, and its shaft sweeps a ball
off a perch into the kiosk pit. The rubric reads the END of that chain (every run
piece down, ball at rest on the pit floor) plus the latched precondition; the seed's
own move — sliding the poked object along the floor to a goal spot — is constructed
in smoke and scores ~0 (there is no floor goal region at all).

Strategy vs the corpus: the beam-scale task converts PLACED MASS into torque on an
untouched body; here nothing is banked — the goal is a one-shot self-propagating
event chain behind a hand-proof guard, armed by placement and released by a poke.
Pick-place tasks read containment of the carried object (the keystone's final pose is
scored only as "toppled"); articulation tasks rotate built joints by direct grasp;
the skittle-gallery task throws a projectile through a fixed funnel — its chain is
ballistic flight, not a mechanical relay of stored energy through five stages. No
corpus task read latches a PRECONDITION (placement while the run stands) and then
requires the chain to actually fire: doing the cascade first and placing the keystone
afterwards earns nothing (the latch only arms while all seven run pieces stand).

Guard honesty (asserted in __post_init__): the tunnel mouth (58 mm) is narrower than
the Franka hand (~63 mm) and the first domino stands 69 mm behind the mouth plane —
beyond fingertip protrusion (~54 mm) — so no run piece is directly reachable; the
kiosk walls + roof seal the ball off except for the letterbox. A SHORT decoy domino
stood on the pad cannot bridge to the run (tip circle 82 mm < 95 mm gap to the first
domino); only the tall keystone can (its tip passes the first domino's top corner
with >= 8 mm to spare) — and it clears under the tunnel roof and is too short to
lean on the roof edge, so it must fall THROUGH the mouth onto the run.

success(): the `armed` latch is set (keystone once stood settled on the pad window
while all 7 run pieces stood), the keystone and all 7 run pieces (6 dominoes +
hammer) are toppled, and the ball rests on the kiosk pit floor. score(): stateless
partial credit 0.25*armed + 0.30*(run pieces toppled / 7) + 0.10*(ball in pit),
capped at 0.65; 1.0 iff success(). Null policy ~0 (everything stands, ball on perch).

Assets are fully procedural (one kinematic compound arcade + simple dynamic boxes):
  - arcade (KINEMATIC): blue trigger pad; a roofed tunnel (inner 58 mm wide, 150 mm
    tall) from the mouth to the kiosk; the kiosk (walls + roof + letterbox front
    52 x 145 mm) with an interior perch column carrying the ball behind a low lip
    open toward the pit.
  - run dominoes d1..d6 (18 x 40 x 80 mm, 40 g) standing inside the tunnel; hammer
    (22 x 40 x 140 mm, 100 g) standing at the tunnel end before the letterbox.
  - keystone (22 x 40 x 140 mm, 100 g, RED) lying loose on the open floor among two
    SAME-SIZE-AS-RUN decoy dominoes (tan); which scatter slot holds the keystone is
    shuffled per episode.
  - ball (r = 14 mm, 8 g) on the perch inside the kiosk.

Per-episode randomization (readback-verified in smoke): the whole arcade jitters in
xy and yaws +/-25 deg (all rubric geometry is alley-frame), run dominoes d2..d5
jitter along the run, the keystone's slot among the three scatter slots is shuffled,
and every loose piece gets slot jitter + free yaw. Heavy imports (isaaclab, pxr) are
deferred so importing this module — and registering the scene — stays app-free.
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


# ----- geometry constants (single source of truth: spawner + cfg asserts + rubric) -------------
_T, _W, _H = 0.018, 0.040, 0.080  # run/decoy domino (thickness, width, height)
_KT, _KW, _KH = 0.022, 0.040, 0.140  # keystone & hammer (tall pieces)
_SDIAG = math.hypot(_T, _H)  # 0.0820: short-domino tip circle about its base edge
_KDIAG = math.hypot(_KT, _KH)  # 0.1417: tall-piece tip circle about its base edge

_PAD_X = 0.50  # trigger pad center (alley frame; alley +x runs downstream)
_MOUTH_X = 0.535  # tunnel mouth plane (upstream wall/roof start)
_TUN_HW = 0.029  # tunnel inner half-width -> 58 mm opening (< 63 mm hand)
_TUN_WALL = 0.012
_TUN_ROOF_Z = 0.150  # roof underside (tall piece 140 mm stands under it)
_D1_X = 0.613  # first run domino center (face 69 mm behind the mouth)
_PITCH = 0.0564  # run spacing d1..d6
_D6_X = _D1_X + 5 * _PITCH  # 0.895
_HAM_X = 0.963  # hammer center (tall piece at the tunnel end)
_KIOSK_F = 1.002  # kiosk front OUTER face = tunnel end
_HOLE_HW = 0.026  # letterbox half-width (52 mm; hammer 40 mm passes)
_HOLE_SILL = 0.010  # letterbox sill height
_HOLE_TOP = 0.155  # letterbox top (opening height 145 mm)
_KIOSK_IX0, _KIOSK_IX1 = 1.014, 1.154  # kiosk interior x span (the pit floor)
_KIOSK_IYH = 0.065  # kiosk interior y half-width
_KIOSK_H = 0.180  # kiosk wall height (roof on top)
_PERCH_X = 1.069  # perch column center (inside the pit)
_PERCH_S = 0.024  # perch column square side
_PERCH_H = 0.048  # perch column height (ball center rests at 0.062)
_BALL_R = 0.014

_SLOTS = ((0.18, -0.25), (0.28, 0.25), (0.44, -0.18))  # loose-piece scatter slots


# ----- custom compound spawner -----------------------------------------------------------------
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


def _box(stage, path: str, size, center, color, contact_offset: float, material=None) -> None:
    """Author one colliding box child prim (translate -> scale, authored once)."""
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


def _spawn_arcade(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC arcade: trigger pad, roofed tunnel, kiosk with letterbox
    front, and the interior perch (column + 3-sided lip open downstream). Origin =
    alley origin on the ground; +x runs downstream toward the kiosk."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(10.0)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)

    pad_m = _friction_material(stage, f"{prim_path}/pad_mat", cfg.mu_pad_s, cfg.mu_pad_d)
    wall_m = _friction_material(stage, f"{prim_path}/wall_mat", cfg.mu_wall_s, cfg.mu_wall_d)
    gray = (0.45, 0.45, 0.50)
    dark = (0.32, 0.32, 0.38)
    blue = (0.15, 0.35, 0.85)

    # --- trigger pad (thin plate; visual target + grippy footing for the poke) ---
    _box(stage, f"{prim_path}/pad", (0.060, 0.055, 0.002), (_PAD_X, 0.0, 0.001), blue,
         0.001, material=pad_m)

    # --- tunnel: two side walls + roof, mouth at _MOUTH_X, ending at the kiosk face ---
    tun_len = _KIOSK_F - _MOUTH_X  # 0.467
    tun_cx = (_MOUTH_X + _KIOSK_F) / 2
    wall_cy = _TUN_HW + _TUN_WALL / 2  # 0.035
    _box(stage, f"{prim_path}/tun_wn", (tun_len, _TUN_WALL, _TUN_ROOF_Z),
         (tun_cx, wall_cy, _TUN_ROOF_Z / 2), gray, 0.001, material=wall_m)
    _box(stage, f"{prim_path}/tun_ws", (tun_len, _TUN_WALL, _TUN_ROOF_Z),
         (tun_cx, -wall_cy, _TUN_ROOF_Z / 2), gray, 0.001, material=wall_m)
    _box(stage, f"{prim_path}/tun_roof", (tun_len, 2 * (_TUN_HW + _TUN_WALL), 0.012),
         (tun_cx, 0.0, _TUN_ROOF_Z + 0.006), dark, 0.001, material=wall_m)

    # --- kiosk front (letterbox): sill + two side sections + header ---
    fx = _KIOSK_F + 0.006  # front wall center (12 mm thick, outer face at _KIOSK_F)
    side_w = (_KIOSK_IYH + _TUN_WALL) - _HOLE_HW  # 0.051 per side
    side_cy = _HOLE_HW + side_w / 2  # 0.0515
    _box(stage, f"{prim_path}/kio_sill", (0.012, 2 * _HOLE_HW, _HOLE_SILL),
         (fx, 0.0, _HOLE_SILL / 2), gray, 0.001, material=wall_m)
    _box(stage, f"{prim_path}/kio_fn", (0.012, side_w, _KIOSK_H),
         (fx, side_cy, _KIOSK_H / 2), gray, 0.001, material=wall_m)
    _box(stage, f"{prim_path}/kio_fs", (0.012, side_w, _KIOSK_H),
         (fx, -side_cy, _KIOSK_H / 2), gray, 0.001, material=wall_m)
    _box(stage, f"{prim_path}/kio_head", (0.012, 2 * _HOLE_HW, _KIOSK_H - _HOLE_TOP),
         (fx, 0.0, (_KIOSK_H + _HOLE_TOP) / 2), gray, 0.001, material=wall_m)

    # --- kiosk shell: side walls, back wall, roof ---
    kio_len = (_KIOSK_IX1 + 0.012) - _KIOSK_F  # 0.164
    kio_cx = (_KIOSK_F + _KIOSK_IX1 + 0.012) / 2  # 1.084
    _box(stage, f"{prim_path}/kio_wn", (kio_len, 0.012, _KIOSK_H),
         (kio_cx, _KIOSK_IYH + 0.006, _KIOSK_H / 2), gray, 0.001, material=wall_m)
    _box(stage, f"{prim_path}/kio_ws", (kio_len, 0.012, _KIOSK_H),
         (kio_cx, -(_KIOSK_IYH + 0.006), _KIOSK_H / 2), gray, 0.001, material=wall_m)
    _box(stage, f"{prim_path}/kio_back", (0.012, 2 * _KIOSK_IYH + 0.024, _KIOSK_H),
         (_KIOSK_IX1 + 0.006, 0.0, _KIOSK_H / 2), gray, 0.001, material=wall_m)
    _box(stage, f"{prim_path}/kio_roof", (kio_len, 2 * _KIOSK_IYH + 0.024, 0.012),
         (kio_cx, 0.0, _KIOSK_H + 0.006), dark, 0.001, material=wall_m)

    # --- perch: column + 3-sided retaining lip (open toward +x, the pit) ---
    _box(stage, f"{prim_path}/perch", (_PERCH_S, _PERCH_S, _PERCH_H),
         (_PERCH_X, 0.0, _PERCH_H / 2), dark, 0.001, material=wall_m)
    lip_z = _PERCH_H + 0.005  # lip spans z 0.048..0.058
    _box(stage, f"{prim_path}/lip_x", (0.004, 0.044, 0.010),
         (_PERCH_X - 0.020, 0.0, lip_z), dark, 0.001, material=wall_m)
    _box(stage, f"{prim_path}/lip_yn", (0.044, 0.004, 0.010),
         (_PERCH_X, 0.020, lip_z), dark, 0.001, material=wall_m)
    _box(stage, f"{prim_path}/lip_ys", (0.044, 0.004, 0.010),
         (_PERCH_X, -0.020, lip_z), dark, 0.001, material=wall_m)
    return root


def _spawner_classes() -> dict[str, Any]:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "arcade" not in _SPAWNER_CACHE:

        @configclass
        class ArcadeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_arcade)
            mu_pad_s: float = 0.60
            mu_pad_d: float = 0.55
            mu_wall_s: float = 0.10
            mu_wall_d: float = 0.08

        _SPAWNER_CACHE["arcade"] = ArcadeSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class KeystoneCascadeSceneCfg(BaseCfg):
    """Config for `KeystoneCascadeScene`. The guard geometry is asserted in
    `__post_init__`: the run is hand-proof (mouth < hand, d1 beyond fingertips), a
    short decoy on the pad cannot bridge to the run, and the tall keystone can —
    passing under the roof and striking d1's top corner."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    stand_upz: float = tunable(0.90)  # up-axis z above this = standing (arming gate)
    topple_upz: float = tunable(0.60)  # up-axis z below this = toppled (fallen chain
    #   pieces rest leaning at up_z ~0.3-0.45; an intact leaning pair sits ~0.86)
    hammer_topple_upz: float = tunable(0.78)  # the felled hammer rests propped on the
    #   perch lip at up_z ~0.63; a struck-but-not-felled hammer rocks back to ~1.0,
    #   so its felled threshold can sit higher without accepting a near-miss
    pad_win_x: tuple = tunable((0.455, 0.550))  # armed window, alley x (capability edge)
    pad_win_y: float = tunable(0.040)  # ... |alley y| <= this
    armed_lin: float = tunable(0.05)  # keystone must be settled to arm
    armed_ang: float = tunable(0.60)
    armed_streak: int = tunable(12)  # substeps the arm condition must hold (anti-flicker)
    pit_x: tuple = tunable((1.020, 1.150))  # ball-in-pit window, alley frame
    pit_y: float = tunable(0.061)
    pit_z: float = tunable(0.035)  # ball CENTER below this = on the pit floor (perch
    #   keeps it at 0.062, so a perched ball can never count)
    ball_settle: float = tunable(0.05)  # ball lin speed below this = at rest

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    anchor_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the alley anchor (m)
    yaw_max: float = tunable(25.0)  # uniform +/- alley yaw (deg)
    run_jitter: float = tunable(0.003)  # d2..d5 jitter along the run (m)
    slot_jitter: float = tunable(0.03)  # loose-piece scatter jitter (m)

    # --- info: structure ---------------------------------------------------------------------
    dom_size: tuple = info((_T, _W, _H))  # run + decoy dominoes
    dom_mass: float = info(0.040)
    key_size: tuple = info((_KT, _KW, _KH))  # keystone & hammer
    key_mass: float = info(0.100)
    ball_radius: float = info(_BALL_R)
    ball_mass: float = info(0.008)
    mu_piece_s: float = info(0.40)  # dominoes / keystone / hammer material
    mu_piece_d: float = info(0.35)
    mu_ball_s: float = info(0.30)
    mu_ball_d: float = info(0.25)
    mu_pad_s: float = info(0.60)  # pad is grippy: the poked keystone tips, not slides
    mu_pad_d: float = info(0.55)
    mu_wall_s: float = info(0.10)  # tunnel/kiosk faces are slick: pieces don't catch
    mu_wall_d: float = info(0.08)
    mu_ground_s: float = info(0.50)
    mu_ground_d: float = info(0.45)
    hand_width: float = info(0.063)  # Franka hand body width (guard sizing)
    finger_reach: float = info(0.054)  # fingertip protrusion beyond the hand body

    # Derived (filled in __post_init__).
    n_run: int = field(default=7, init=False)  # 6 dominoes + hammer

    def __post_init__(self) -> None:
        # -- the guard is hand-proof --
        assert 2 * _TUN_HW < self.hand_width, "tunnel mouth must be narrower than the hand"
        assert (_D1_X - _T / 2) - _MOUTH_X > self.finger_reach + 0.01, \
            "d1 must stand beyond fingertip reach behind the mouth plane"
        assert 2 * _HOLE_HW < self.hand_width, "letterbox must be narrower than the hand"
        # -- a SHORT decoy on the pad cannot bridge to the run --
        assert _PAD_X + _T / 2 + _SDIAG + 0.006 < _D1_X - _T / 2, \
            "a short domino on the pad must fall short of d1"
        # -- the keystone CAN bridge: tip circle covers d1's top corner with margin --
        reach_corner = math.hypot((_D1_X - _T / 2) - (_PAD_X + _KT / 2), _H)
        assert reach_corner <= _KDIAG - 0.008, "keystone must reach d1's top corner"
        # -- ... passing under the roof, and too short to lean on the roof edge --
        tip_at_mouth = math.sqrt(_KDIAG**2 - (_MOUTH_X - (_PAD_X + _KT / 2))**2)
        assert tip_at_mouth < _TUN_ROOF_Z - 0.004, "keystone tip must clear the roof"
        assert math.hypot(_MOUTH_X - (_PAD_X + _KT / 2), _TUN_ROOF_Z) > _KDIAG + 0.004, \
            "keystone must be too short to hang on the roof edge"
        # -- run relay margins --
        assert _SDIAG > _PITCH + 0.02, "each domino must strike the next well up its face"
        gap = (_HAM_X - _KT / 2) - (_D6_X + _T / 2)
        assert math.sqrt(_SDIAG**2 - gap**2) > 0.055, "d6 must strike the hammer high"
        # -- hammer through the letterbox, sweeping the ball --
        piv = _HAM_X + _KT / 2
        assert _KW < 2 * _HOLE_HW - 0.008, "hammer must pass the letterbox width"
        tip_at_face = math.sqrt(_KDIAG**2 - (_KIOSK_F - piv)**2)
        assert tip_at_face < _HOLE_TOP - 0.004, "hammer tip must clear the letterbox top"
        assert math.hypot(_PERCH_X - piv, _PERCH_H + _BALL_R) < _KDIAG - 0.008, \
            "the hammer shaft must sweep through the perched ball's center"
        assert piv + _KDIAG > _PERCH_X + _BALL_R + 0.02, "hammer must reach past the ball"
        # -- pit window vs perch: a perched ball can never count as in-pit --
        assert _PERCH_H + _BALL_R > self.pit_z + 0.02
        assert self.pit_x[0] > _KIOSK_IX0 and self.pit_x[1] < _KIOSK_IX1 + 0.001
        assert self.pit_y < _KIOSK_IYH
        # -- ball fits its lip cradle and the letterbox is irrelevant to it (walled in) --
        assert 2 * _BALL_R < 0.036 - 0.003, "ball must sit inside the perch lip"
        # -- embodiment: the keystone is graspable; the pad window is inside capability --
        assert _KT < 0.078, "keystone must fit the parallel jaw across its thickness"
        assert self.pad_win_x[0] >= 0.451, \
            "armed window must not extend below the bridge-capability boundary"
        assert self.pad_win_x[0] < _PAD_X < self.pad_win_x[1]
        # -- toppled/standing thresholds leave a dead band --
        assert self.topple_upz + 0.25 < self.stand_upz
        assert self.hammer_topple_upz + 0.10 < self.stand_upz


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("keystone_cascade")
class KeystoneCascadeScene(BaseScene):
    cfg: KeystoneCascadeSceneCfg

    def __init__(self, cfg: KeystoneCascadeSceneCfg | None = None) -> None:
        super().__init__(cfg or KeystoneCascadeSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        arcade_cls = _spawner_classes()["arcade"]
        piece_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.mu_piece_s, dynamic_friction=c.mu_piece_d, restitution=0.0)
        piece_rigid = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=8, solver_velocity_iteration_count=1,
            max_depenetration_velocity=0.5, sleep_threshold=0.0,
            stabilization_threshold=0.0)
        piece_coll = sim_utils.CollisionPropertiesCfg(contact_offset=0.0015, rest_offset=0.0)

        def box_body(name: str, size, mass, color, pos):
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    physics_material=piece_mat, rigid_props=piece_rigid,
                    collision_props=piece_coll,
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
            )

        tan = (0.75, 0.62, 0.42)
        out: dict[str, Any] = {
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
            "arcade": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Arcade",
                spawn=arcade_cls(
                    mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mu_pad_s=c.mu_pad_s, mu_pad_d=c.mu_pad_d,
                    mu_wall_s=c.mu_wall_s, mu_wall_d=c.mu_wall_d),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.95, 0.95, 0.95)),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_ball_s, dynamic_friction=c.mu_ball_d,
                        restitution=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=8,
                        # 4 velocity iterations: kills the GPU sphere-on-box creep
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5, linear_damping=0.02,
                        sleep_threshold=0.0, stabilization_threshold=0.0),
                    collision_props=piece_coll,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(_PERCH_X, 0.0, _PERCH_H + _BALL_R + 0.002)),
            ),
            "keystone": box_body("Keystone", c.key_size, c.key_mass, (0.85, 0.10, 0.10),
                                 (_SLOTS[0][0], _SLOTS[0][1], _KT / 2 + 0.002)),
            "hammer": box_body("Hammer", c.key_size, c.key_mass, (0.35, 0.25, 0.15),
                               (_HAM_X, 0.0, _KH / 2 + 0.002)),
        }
        for i in range(6):
            out[f"d{i + 1}"] = box_body(f"D{i + 1}", c.dom_size, c.dom_mass, tan,
                                        (_D1_X + i * _PITCH, 0.0, _H / 2 + 0.002))
        for i in range(2):
            out[f"decoy_{i}"] = box_body(f"Decoy{i}", c.dom_size, c.dom_mass, tan,
                                         (_SLOTS[i + 1][0], _SLOTS[i + 1][1], _T / 2 + 0.002))
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
        self.arcade: RigidObject = env.iscene["arcade"]
        self.run: list[RigidObject] = [env.iscene[f"d{i + 1}"] for i in range(6)]
        self.run.append(env.iscene["hammer"])  # run = d1..d6 + hammer (7 pieces)
        self.keystone: RigidObject = env.iscene["keystone"]
        self.decoys: list[RigidObject] = [env.iscene[f"decoy_{i}"] for i in range(2)]
        self.ball: RigidObject = env.iscene["ball"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.anchor = torch.zeros(n, 2, device=dev)  # alley anchor (env-local xy)
        self.yaw = torch.zeros(n, device=dev)  # alley yaw
        self.key_slot = torch.zeros(n, dtype=torch.long, device=dev)
        self.armed = torch.zeros(n, dtype=torch.bool, device=dev)  # the placement latch
        self._streak = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter/yaw the whole arcade (kinematic teleport), stand the
        run + hammer inside it, seat the ball on the perch, and scatter keystone +
        decoys lying flat at shuffled jittered slots. Clears the armed latch."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        axy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.anchor_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_max)
        self.anchor[env_ids] = axy
        self.yaw[env_ids] = yaw
        self.armed[env_ids] = False
        self._streak[env_ids] = 0
        half = yaw / 2
        zeros = torch.zeros(m, device=dev)
        q_yaw = torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)
        cy, sy = torch.cos(yaw), torch.sin(yaw)

        def to_world(local: torch.Tensor) -> torch.Tensor:
            wx = axy[:, 0] + local[:, 0] * cy - local[:, 1] * sy
            wy = axy[:, 1] + local[:, 0] * sy + local[:, 1] * cy
            return torch.stack([wx, wy, local[:, 2]], dim=-1)

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        def col(x): return torch.full((m,), x, device=dev)

        write(self.arcade, to_world(torch.stack([zeros, zeros, zeros], dim=-1)), q_yaw)
        # --- run dominoes + hammer, standing, aligned with the alley ---
        for i in range(6):
            x = col(_D1_X + i * _PITCH)
            if 1 <= i <= 4:  # d2..d5 jitter along the run
                x = x + (torch.rand(m, device=dev) * 2 - 1) * c.run_jitter
            write(self.run[i], to_world(torch.stack([x, zeros, col(_H / 2 + 0.002)],
                                                    dim=-1)), q_yaw)
        write(self.run[6], to_world(torch.stack([col(_HAM_X), zeros,
                                                 col(_KH / 2 + 0.002)], dim=-1)), q_yaw)
        # --- ball on the perch ---
        write(self.ball, to_world(torch.stack(
            [col(_PERCH_X), zeros, col(_PERCH_H + _BALL_R + 0.002)], dim=-1)), q_yaw)

        # --- loose pieces: keystone + 2 decoys, lying flat at shuffled jittered slots ---
        perm = torch.rand(m, 3, device=dev).argsort(dim=-1)  # slot permutation per env
        self.key_slot[env_ids] = perm[:, 0]
        slots = torch.tensor(_SLOTS, device=dev)  # (3, 2)
        c45 = math.sqrt(0.5)
        for j, (body, hz) in enumerate([(self.keystone, _KT / 2), (self.decoys[0], _T / 2),
                                        (self.decoys[1], _T / 2)]):
            sxy = slots[perm[:, j]] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            wyaw = yaw + (torch.rand(m, device=dev) * 2 - 1) * math.pi
            ch, sh = torch.cos(wyaw / 2), torch.sin(wyaw / 2)
            # lying flat: qz(wyaw) * qy(90 deg) -> long axis horizontal
            q_flat = torch.stack([ch * c45, -sh * c45, ch * c45, sh * c45], dim=-1)
            pos = to_world(torch.stack([sxy[:, 0], sxy[:, 1], col(hz + 0.002)], dim=-1))
            write(body, pos, q_flat)

    def post_step(self) -> None:
        """Latch `armed` when the keystone stands settled in the pad window while ALL
        7 run pieces still stand — held for `armed_streak` consecutive substeps (a
        teleport-write or a fly-through cannot flicker-latch it)."""
        c = self.cfg
        loc = self.to_alley(self.keystone.data.root_pos_w)
        on_pad = (loc[:, 0] >= c.pad_win_x[0]) & (loc[:, 0] <= c.pad_win_x[1]) \
            & (loc[:, 1].abs() <= c.pad_win_y)
        standing = self.up_z(self.keystone) > c.stand_upz
        slow = (self.keystone.data.root_lin_vel_w.norm(dim=-1) < c.armed_lin) \
            & (self.keystone.data.root_ang_vel_w.norm(dim=-1) < c.armed_ang)
        run_up = self.run_standing()
        cond = on_pad & standing & slow & run_up
        self._streak = torch.where(cond, self._streak + 1,
                                   torch.zeros_like(self._streak))
        self.armed = self.armed | (self._streak >= c.armed_streak)

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "arcade": self.arcade.data.root_state_w[env_ids].clone(),
            "run": [b.data.root_state_w[env_ids].clone() for b in self.run],
            "keystone": self.keystone.data.root_state_w[env_ids].clone(),
            "decoys": [b.data.root_state_w[env_ids].clone() for b in self.decoys],
            "ball": self.ball.data.root_state_w[env_ids].clone(),
            "anchor": self.anchor[env_ids].clone(),
            "yaw": self.yaw[env_ids].clone(),
            "key_slot": self.key_slot[env_ids].clone(),
            "armed": self.armed[env_ids].clone(),
            "streak": self._streak[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.arcade.write_root_state_to_sim(state["arcade"], env_ids)
        for b, st in zip(self.run, state["run"]):
            b.write_root_state_to_sim(st, env_ids)
        self.keystone.write_root_state_to_sim(state["keystone"], env_ids)
        for b, st in zip(self.decoys, state["decoys"]):
            b.write_root_state_to_sim(st, env_ids)
        self.ball.write_root_state_to_sim(state["ball"], env_ids)
        self.anchor[env_ids] = state["anchor"]
        self.yaw[env_ids] = state["yaw"]
        self.key_slot[env_ids] = state["key_slot"]
        self.armed[env_ids] = state["armed"]
        self._streak[env_ids] = state["streak"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A domino arcade stands on the floor: a small blue trigger pad, and right "
            "behind it the mouth of a roofed tunnel (about 6 cm wide and 15 cm tall "
            "inside — far too narrow for the hand). Inside the tunnel a run of six "
            "dominoes leads to a tall dark hammer block, and the tunnel ends at a "
            "walled, roofed kiosk whose only opening is a letterbox slot in its front "
            "wall. Inside the kiosk a white ball rests on a raised perch. Nothing "
            "inside the tunnel or the kiosk can be reached or touched directly. On "
            "the open floor lie three loose pieces: two short dominoes (the same size "
            "as the run pieces) and one tall RED block — the keystone, about "
            f"{_KH * 100:.0f} cm long and {_KT * 1000:.0f} mm thick, easy to grasp. "
            "Which scatter spot holds the keystone, the whole arcade's position and "
            "heading, and the run spacing change every episode — read the scene by "
            "looking.\n"
            "Goal: pick up the tall red keystone, stand it upright on the blue pad "
            "(centered, long axis vertical, facing the tunnel mouth), let it settle, "
            "then poke its upper half toward the mouth so it topples INTO the tunnel. "
            "Only the keystone is tall enough to bridge through the mouth and strike "
            "the first domino; a short domino stood on the pad falls short. The "
            "falling keystone starts the chain: the dominoes topple one into the "
            "next, the last one fells the hammer, the hammer swings through the "
            "letterbox, and its shaft sweeps the ball off its perch onto the kiosk "
            "floor. Finish with every run piece and the keystone down and the ball "
            "at rest on the kiosk floor. The keystone must be standing on the pad — "
            "with the whole run still intact — before it is poked; toppling the run "
            "any other way, or at any other time, does not count. Touch only the "
            "loose pieces on the open floor."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Stand the tall red keystone block upright on the blue pad, then poke it "
            "so it topples into the tunnel mouth and sets off the domino run — the "
            "chain must end with the white ball knocked off its perch onto the kiosk "
            "floor. Touch only the loose pieces."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def to_alley(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N, 3) world position -> alley frame (anchor + yaw removed)."""
        p = pos_w - self.env_origins
        cy, sy = torch.cos(self.yaw), torch.sin(self.yaw)
        dx = p[:, 0] - self.anchor[:, 0]
        dy = p[:, 1] - self.anchor[:, 1]
        return torch.stack([dx * cy + dy * sy, -dx * sy + dy * cy, p[:, 2]], dim=-1)

    def alley_dir(self, local_vec) -> torch.Tensor:
        """(N, 3) world direction of an alley-frame vector (rotation only)."""
        vx, vy, vz = (float(v) for v in local_vec)
        cy, sy = torch.cos(self.yaw), torch.sin(self.yaw)
        return torch.stack([vx * cy - vy * sy, vx * sy + vy * cy,
                            torch.full_like(cy, vz)], dim=-1)

    def up_z(self, body) -> torch.Tensor:
        """(N,) world-z component of the body's +z (long/up) axis."""
        from isaaclab.utils.math import quat_apply

        q = body.data.root_quat_w
        ez = torch.tensor([0.0, 0.0, 1.0], device=q.device).expand(q.shape[0], 3)
        return quat_apply(q, ez)[:, 2]

    def run_standing(self) -> torch.Tensor:
        """(N,) bool: ALL 7 run pieces upright."""
        out = None
        for b in self.run:
            up = self.up_z(b) > self.cfg.stand_upz
            out = up if out is None else out & up
        return out

    def run_toppled_frac(self) -> torch.Tensor:
        """(N,) float: fraction of the 7 run pieces toppled (dominoes vs the
        `topple_upz` lean gate; the hammer vs its own propped-rest gate)."""
        c = self.cfg
        out = sum((self.up_z(b) < c.topple_upz).float() for b in self.run[:6])
        out = out + (self.up_z(self.run[6]) < c.hammer_topple_upz).float()
        return out / float(c.n_run)

    def ball_in_pit(self) -> torch.Tensor:
        """(N,) bool: ball center on the kiosk pit floor (alley frame)."""
        c = self.cfg
        loc = self.to_alley(self.ball.data.root_pos_w)
        return (loc[:, 0] >= c.pit_x[0]) & (loc[:, 0] <= c.pit_x[1]) \
            & (loc[:, 1].abs() <= c.pit_y) & (loc[:, 2] <= c.pit_z)

    def ball_settled(self) -> torch.Tensor:
        return self.ball.data.root_lin_vel_w.norm(dim=-1) < self.cfg.ball_settle

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: armed latch set (keystone once stood settled on the pad while
        the whole run stood), keystone toppled, all 7 run pieces toppled, ball at
        rest on the pit floor."""
        c = self.cfg
        key_down = self.up_z(self.keystone) < c.topple_upz
        all_down = self.run_toppled_frac() >= 1.0 - 1e-6
        return self.armed & key_down & all_down & self.ball_in_pit() & self.ball_settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.25 * armed + 0.30 * run-toppled fraction + 0.10 *
        ball-in-pit, capped at 0.65; 1.0 iff success(). Null policy ~0 (nothing
        toppled, latch clear, ball on its perch above the pit-z gate)."""
        base = (0.25 * self.armed.float() + 0.30 * self.run_toppled_frac()
                + 0.10 * self.ball_in_pit().float()).clamp(max=0.65)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="keystone_cascade", robot="null",
                                      env_spacing=3.0))

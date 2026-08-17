"""GaugeSortScene — sort four bars through a go/no-go gauge slot: thin bars drop
THROUGH the slot into a covered bin, thick bars go to the open reject tray
(sim_gen task `pick_single_egad_i216`).

Derived from maniskill/pick_single_egad, but STRATEGICALLY different: the seed
grasps ONE odd object and lifts it 7.5 cm — a single pick judged by a z-shift on
the picked object itself; no other object, no destination, no decision. Here
nothing is judged by lifting at all: FOUR bars of identical color and length but
different thickness must each be CLASSIFIED against a physical go/no-go gauge —
a slot channel of per-episode-randomized gap width, the only opening of an
otherwise closed bin — and ROUTED to one of two destinations: bars thinner than
the gap must pass through the slot (a real aperture transit, the bin is sealed
everywhere else) and rest inside the bin; bars thicker than the gap must be laid
in a walled reject tray. Which of the middle bars fits flips with the sampled
gap, so a memorized fixed routing fails: the plan is measure-then-route, and its
code shape is a per-object decision loop, not a single lift.

Strategy vs the corpus tasks used as format references: the pen-holder exemplar
fills ONE open cup with all pens tip-up (single destination, orientation rubric,
no classification); the keystone-cascade task arms a latch and fires a hands-off
domino chain (causal relay, no routing decision). Here there is no chain and no
latch — the core skill is physical gauging: the same motion (drop through slot)
succeeds or is mechanically refused depending on a per-episode dimension, and
the rubric requires BOTH routes to be used correctly (dumping everything in the
tray fails on the fitting bars; everything at the slot fails on the thick ones).

Geometry honesty (asserted in __post_init__): the sampled gap is always >= 5 mm
away from every bar thickness, so "fits" is crisp — a fitting bar clears the
slot by >= 6 mm total, a rejected bar interferes by >= 6 mm; each bar's minimum
dimension IS its thickness (t < w < l), so no orientation sneaks a thick bar
through; the bin is closed (4 walls + two gauge plates flush with the wall top)
so any bar inside it physically transited the slot; a bar (95 mm) is shorter
than the bin walls (140 mm) so it submerges fully below the sill.

success(): every bar settled in its correct destination — fitting bars inside
the bin (center below sill - 20 mm, inside the inner walls), thick bars inside
the tray. score(): 0.9 * (correctly routed fraction); 1.0 iff success(). Null
policy ~0 (all bars on the open floor).

Assets are fully procedural:
  - station (KINEMATIC compound): bin walls (inner 200 x 140 mm, 140 mm tall)
    and the reject tray (inner 200 x 200 mm floor, 32 mm lip) side by side;
  - gauge plates (2 separate KINEMATIC boxes): the bin roof, repositioned at
    reset so the slot channel between them has the sampled gap W;
  - bars bar_a..bar_d (dynamic): 95 mm long, thickness/width 16/40, 30/48,
    44/56, 58/66 mm, all the same orange — identity is SIZE, not color.

Per-episode randomization (readback-verified in smoke): gauge gap W = one of
three class midpoints (23 / 37 / 51 mm) + jitter (so 1, 2 or 3 bars fit — at
least one bar always fits and bar_d never does), station anchor xy + yaw, bar
scatter slot permutation + jitter + free yaw (overlap-rejected placement).
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


# ----- geometry constants (single source of truth: spawner + cfg asserts + rubric) -------------
_PL = 0.095  # bar length (all bars)
# (name, thickness t, width w): t < w < l; the MIN dimension is t, so the slot
# classifies orientation-free.
_PARTS = (("bar_a", 0.016, 0.040),
          ("bar_b", 0.030, 0.048),
          ("bar_c", 0.044, 0.056),
          ("bar_d", 0.058, 0.066))
_THINS = tuple(t for _n, t, _w in _PARTS)
_MIDS = (0.023, 0.037, 0.051)  # gauge-gap class midpoints (1 / 2 / 3 bars fit)

_ANCHOR = (0.42, 0.0)  # nominal station anchor (world; robot base at origin)
_BIN_C = (0.06, -0.17)  # bin center, station frame
_TRAY_C = (0.06, 0.17)  # tray center, station frame
_BIN_IX = 0.10  # bin inner half-length (x, along the slot channel)
_BIN_IY = 0.07  # bin inner half-width (y, across the gap)
_WALL_T = 0.012
_SILL = 0.140  # bin wall top = gauge plate underside (the aperture sill)
_PLATE_T = 0.008
_PLATE_L = 2 * (_BIN_IX + _WALL_T)  # 0.224: plates cover the walls end to end
_PLATE_W = 0.10
_PLATE_Z = _SILL + _PLATE_T / 2  # 0.144
_TRAY_IH = 0.10  # tray inner half-size (square)
_TRAY_WH = 0.032  # tray lip height
_TRAY_FT = 0.004  # tray floor thickness
_SLOTS4 = ((-0.20, -0.17), (-0.23, -0.055), (-0.23, 0.06), (-0.20, 0.175))


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


def _spawn_station(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC sorting station: the bin's four walls (roofless here —
    the two gauge plates are separate bodies) and the walled reject tray. Origin =
    station anchor on the ground; +x runs away from the robot."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(20.0)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)

    mat = _friction_material(stage, f"{prim_path}/station_mat", cfg.mu_station_s,
                             cfg.mu_station_d)
    gray = (0.42, 0.44, 0.50)
    red = (0.62, 0.16, 0.14)

    bx, by = _BIN_C
    # --- bin: two end walls (normal x) + two side walls (normal y), 140 mm tall ---
    end_sz = (_WALL_T, 2 * _BIN_IY + 2 * _WALL_T, _SILL)
    _box(stage, f"{prim_path}/bin_e0", end_sz,
         (bx - _BIN_IX - _WALL_T / 2, by, _SILL / 2), gray, 0.001, material=mat)
    _box(stage, f"{prim_path}/bin_e1", end_sz,
         (bx + _BIN_IX + _WALL_T / 2, by, _SILL / 2), gray, 0.001, material=mat)
    side_sz = (2 * _BIN_IX, _WALL_T, _SILL)
    _box(stage, f"{prim_path}/bin_s0", side_sz,
         (bx, by - _BIN_IY - _WALL_T / 2, _SILL / 2), gray, 0.001, material=mat)
    _box(stage, f"{prim_path}/bin_s1", side_sz,
         (bx, by + _BIN_IY + _WALL_T / 2, _SILL / 2), gray, 0.001, material=mat)

    tx, ty = _TRAY_C
    # --- tray: floor plate + four low lip walls ---
    _box(stage, f"{prim_path}/tray_floor",
         (2 * _TRAY_IH + 2 * _WALL_T, 2 * _TRAY_IH + 2 * _WALL_T, _TRAY_FT),
         (tx, ty, _TRAY_FT / 2), red, 0.001, material=mat)
    lz = _TRAY_FT + _TRAY_WH / 2
    lip_x = (_WALL_T, 2 * _TRAY_IH + 2 * _WALL_T, _TRAY_WH)
    _box(stage, f"{prim_path}/tray_e0", lip_x,
         (tx - _TRAY_IH - _WALL_T / 2, ty, lz), red, 0.001, material=mat)
    _box(stage, f"{prim_path}/tray_e1", lip_x,
         (tx + _TRAY_IH + _WALL_T / 2, ty, lz), red, 0.001, material=mat)
    lip_y = (2 * _TRAY_IH, _WALL_T, _TRAY_WH)
    _box(stage, f"{prim_path}/tray_s0", lip_y,
         (tx, ty - _TRAY_IH - _WALL_T / 2, lz), red, 0.001, material=mat)
    _box(stage, f"{prim_path}/tray_s1", lip_y,
         (tx, ty + _TRAY_IH + _WALL_T / 2, lz), red, 0.001, material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "station" not in _SPAWNER_CACHE:

        @configclass
        class StationSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_station)
            mu_station_s: float = 0.45
            mu_station_d: float = 0.40

        _SPAWNER_CACHE["station"] = StationSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class GaugeSortSceneCfg(BaseCfg):
    """Config for `GaugeSortScene`. The gauge honesty margins are asserted in
    `__post_init__`: every sampled gap is >= 5 mm away from every bar thickness,
    each bar's MIN dimension is its thickness, the bin is closed except for the
    slot, and every judged threshold clears the geometry by construction."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    settle_speed: float = tunable(0.12)  # |lin vel| below this = settled (above the
    #   GPU phantom-velocity readback artifact band)
    bin_xy_x: float = tunable(0.095)  # |station x - bin cx| below this = inside (5 mm
    #   inside the inner wall face; a lying bar's center is always deeper)
    bin_xy_y: float = tunable(0.065)
    bin_z: float = tunable(0.120)  # bar center BELOW this = below the sill (aperture
    #   containment: sill 0.140 - 20 mm margin; a settled bar tops out ~0.105)
    tray_xy: float = tunable(0.095)  # |station xy - tray c| below this = inside
    tray_z: float = tunable(0.080)  # bar center below this = down in the tray (a bar
    #   balanced on the 32 mm lip or on the bin roof never passes)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    anchor_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the station anchor
    yaw_max: float = tunable(20.0)  # uniform +/- station yaw (deg)
    gauge_jitter: float = tunable(0.001)  # uniform +/- jitter on the sampled gap (m)
    slot_jitter: float = tunable(0.025)  # bar scatter jitter (m)

    # --- info: structure ---------------------------------------------------------------------
    bar_len: float = info(_PL)
    parts: tuple = info(_PARTS)  # (name, thickness, width) per bar
    mids: tuple = info(_MIDS)  # gauge-gap class midpoints
    bar_mass: float = info(0.060)  # 60 g per bar (uniform; size is the identity cue)
    mu_bar_s: float = info(0.40)
    mu_bar_d: float = info(0.35)
    mu_station_s: float = info(0.45)
    mu_station_d: float = info(0.40)
    hand_jaw: float = info(0.078)  # Franka parallel-jaw max usable opening

    def __post_init__(self) -> None:
        thins = [t for _n, t, _w in self.parts]
        wides = [w for _n, _t, w in self.parts]
        # -- bars: min dimension is the thickness; ascending, well separated --
        for (_n, t, w) in self.parts:
            assert t < w < self.bar_len, "each bar must have t < w < l (min dim = t)"
        for a, b in zip(thins, thins[1:]):
            assert b - a >= 0.012, "bar thicknesses must differ by >= 12 mm"
        # -- gauge classes: every sampled gap >= 5 mm from every thickness --
        for k, mid in enumerate(self.mids):
            lo, hi = mid - self.gauge_jitter, mid + self.gauge_jitter
            assert lo - thins[k] >= 0.005, "largest fitting bar must clear the gap"
            assert thins[k + 1] - hi >= 0.005, "smallest thick bar must interfere"
        assert thins[0] < min(self.mids) - self.gauge_jitter, "bar_a always fits"
        assert thins[-1] > max(self.mids) + self.gauge_jitter, "bar_d never fits"
        # -- the bin is closed except the slot; plates cover the walls --
        w_min = min(self.mids) - self.gauge_jitter
        assert w_min / 2 + _PLATE_W >= _BIN_IY + _WALL_T + 0.02, \
            "gauge plates must cover the bin walls at the narrowest gap"
        assert abs(_PLATE_L - 2 * (_BIN_IX + _WALL_T)) < 1e-9, \
            "plates must span the bin end to end"
        # -- containment thresholds clear the geometry --
        assert self.bin_z <= _SILL - 0.02, "in-bin gate must sit >= 20 mm below the sill"
        assert self.bar_len < _SILL, "a bar must submerge fully below the sill"
        assert self.bar_len < 2 * _BIN_IX - 0.01, "a bar must lie flat inside the bin"
        assert self.bin_xy_y <= _BIN_IY - 0.004 and self.bin_xy_x <= _BIN_IX - 0.004
        assert self.tray_xy <= _TRAY_IH - 0.004
        assert self.tray_z > _TRAY_FT + max(thins) / 2 + 0.01, \
            "a bar lying on the tray floor must pass the tray-z gate"
        assert self.tray_z < _SILL + max(thins) / 2, \
            "a bar on the bin roof must fail the tray-z gate"
        # -- capacity: three thick bars fit the tray, three thin bars fit the bin --
        assert sum(sorted(wides)[-3:]) <= 2 * _TRAY_IH - 0.02
        # -- embodiment: every bar is graspable across its thickness --
        assert max(thins) <= self.hand_jaw - 0.015, "jaw must span the thickest bar"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("gauge_sort")
class GaugeSortScene(BaseScene):
    cfg: GaugeSortSceneCfg

    def __init__(self, cfg: GaugeSortSceneCfg | None = None) -> None:
        super().__init__(cfg or GaugeSortSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        station_cls = _spawner_classes()["station"]
        bar_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.mu_bar_s, dynamic_friction=c.mu_bar_d, restitution=0.0)
        bar_rigid = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=8, solver_velocity_iteration_count=4,
            max_depenetration_velocity=0.5, linear_damping=0.05, angular_damping=0.05,
            sleep_threshold=0.0, stabilization_threshold=0.0)
        bar_coll = sim_utils.CollisionPropertiesCfg(contact_offset=0.0015, rest_offset=0.0)
        plate_spawn = dict(
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.22, 0.28)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=c.mu_station_s, dynamic_friction=c.mu_station_d,
                restitution=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0))
        orange = (0.90, 0.45, 0.10)

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
            "station": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Station",
                spawn=station_cls(
                    mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mu_station_s=c.mu_station_s, mu_station_d=c.mu_station_d),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(_ANCHOR[0], _ANCHOR[1], 0.0)),
            ),
        }
        for side, y0 in (("n", 1.0), ("s", -1.0)):
            out[f"plate_{side}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plate_" + side,
                spawn=sim_utils.CuboidCfg(size=(_PLATE_L, _PLATE_W, _PLATE_T),
                                          **plate_spawn),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(_ANCHOR[0] + _BIN_C[0],
                         _ANCHOR[1] + _BIN_C[1] + y0 * (0.018 + _PLATE_W / 2),
                         _PLATE_Z)),
            )
        for i, (name, t, w) in enumerate(c.parts):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bar_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=(w, t, c.bar_len),  # local x=width, y=thickness, z=length
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=orange),
                    physics_material=bar_mat, rigid_props=bar_rigid,
                    collision_props=bar_coll,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bar_mass)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(_ANCHOR[0] + _SLOTS4[i][0], _ANCHOR[1] + _SLOTS4[i][1],
                         t / 2 + 0.003),
                    rot=(math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)),  # lying flat
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
        c = self.cfg
        self.station: RigidObject = env.iscene["station"]
        self.plates: list[RigidObject] = [env.iscene["plate_s"], env.iscene["plate_n"]]
        self.bars: list[RigidObject] = [env.iscene[name] for name, _t, _w in c.parts]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.anchor = torch.zeros(n, 2, device=dev)  # station anchor (env-local xy)
        self.yaw = torch.zeros(n, device=dev)  # station yaw
        self.W = torch.zeros(n, device=dev)  # sampled gauge gap
        self.slot_perm = torch.zeros(n, 4, dtype=torch.long, device=dev)
        self._thins = torch.tensor([t for _n, t, _w in c.parts], device=dev)
        self._wides = torch.tensor([w for _n, _t, w in c.parts], device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter/yaw the station (kinematic teleport), sample the
        gauge gap and reposition the two plates to open exactly that slot, scatter
        the four bars lying flat at shuffled slots (overlap-rejected)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        axy = torch.tensor(_ANCHOR, device=dev) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.anchor_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_max)
        self.anchor[env_ids] = axy
        self.yaw[env_ids] = yaw
        # gauge gap: class from a rand comparison (first-randint degeneracy trap),
        # then +/- jitter around the class midpoint
        k = (torch.rand(m, device=dev) * len(c.mids)).long().clamp(max=len(c.mids) - 1)
        mids = torch.tensor(c.mids, device=dev)
        wgap = mids[k] + (torch.rand(m, device=dev) * 2 - 1) * c.gauge_jitter
        self.W[env_ids] = wgap

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

        write(self.station, to_world(torch.stack([zeros, zeros, zeros], dim=-1)), q_yaw)
        # --- gauge plates: open a slot of width W centered on the bin axis ---
        for sgn, body in ((-1.0, self.plates[0]), (1.0, self.plates[1])):
            py = _BIN_C[1] + sgn * (wgap / 2 + _PLATE_W / 2)
            write(body, to_world(torch.stack([col(_BIN_C[0]), py, col(_PLATE_Z)],
                                             dim=-1)), q_yaw)

        # --- bars: shuffled scatter slots, lying flat, free yaw, overlap-rejected ---
        perm = torch.rand(m, 4, device=dev).argsort(dim=-1)
        self.slot_perm[env_ids] = perm
        slots = torch.tensor(_SLOTS4, device=dev)
        c45 = math.sqrt(0.5)

        def seg_dist(p_c, p_y, p_h, q_c, q_y, q_h):
            """Min distance between 2D segments (centre, yaw, half-length), batched."""
            su = torch.stack([torch.cos(p_y), torch.sin(p_y)], dim=-1)
            tv = torch.stack([torch.cos(q_y), torch.sin(q_y)], dim=-1)
            best = torch.full_like(p_y, torch.inf)
            for fa in (-1.0, -0.5, 0.0, 0.5, 1.0):
                pa = p_c + su * (fa * p_h)
                wv = pa - q_c
                tt = (wv * tv).sum(-1).clamp(-q_h, q_h)
                best = torch.minimum(best, (wv - tv * tt.unsqueeze(-1)).norm(dim=-1))
            for fb in (-1.0, -0.5, 0.0, 0.5, 1.0):
                qb = q_c + tv * (fb * q_h)
                wv = qb - p_c
                tt = (wv * su).sum(-1).clamp(-p_h, p_h)
                best = torch.minimum(best, (wv - su * tt.unsqueeze(-1)).norm(dim=-1))
            return best

        placed: list[tuple[torch.Tensor, torch.Tensor, float]] = []
        for j, (name, t, w) in enumerate(c.parts):
            slot = slots[perm[:, j]]
            ctr = slot + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            v = (torch.rand(m, device=dev) * 2 - 1) * math.pi  # local (station) yaw
            for _try in range(12):
                bad = torch.zeros(m, dtype=torch.bool, device=dev)
                for (qc, qv, qw) in placed:
                    d = seg_dist(ctr, v, _PL / 2, qc, qv, _PL / 2)
                    bad |= d < (w / 2 + qw / 2 + 0.006)
                if not bad.any():
                    break
                nb = int(bad.sum())
                ctr[bad] = slot[bad] + (torch.rand(nb, 2, device=dev) * 2 - 1) * c.slot_jitter
                v[bad] = (torch.rand(nb, device=dev) * 2 - 1) * math.pi
            placed.append((ctr, v, w))

            u = yaw + v  # world yaw of the bar's flat-lying frame
            ch, sh = torch.cos(u / 2), torch.sin(u / 2)
            # lying flat on the w x l face: q = qz(u) * qx(90 deg)
            q_flat = torch.stack([ch * c45, ch * c45, sh * c45, sh * c45], dim=-1)
            pos = to_world(torch.stack([ctr[:, 0], ctr[:, 1], col(t / 2 + 0.003)], dim=-1))
            write(self.bars[j], pos, q_flat)

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "station": self.station.data.root_state_w[env_ids].clone(),
            "plates": [b.data.root_state_w[env_ids].clone() for b in self.plates],
            "bars": [b.data.root_state_w[env_ids].clone() for b in self.bars],
            "anchor": self.anchor[env_ids].clone(),
            "yaw": self.yaw[env_ids].clone(),
            "W": self.W[env_ids].clone(),
            "slot_perm": self.slot_perm[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.station.write_root_state_to_sim(state["station"], env_ids)
        for b, st in zip(self.plates, state["plates"]):
            b.write_root_state_to_sim(st, env_ids)
        for b, st in zip(self.bars, state["bars"]):
            b.write_root_state_to_sim(st, env_ids)
        self.anchor[env_ids] = state["anchor"]
        self.yaw[env_ids] = state["yaw"]
        self.W[env_ids] = state["W"]
        self.slot_perm[env_ids] = state["slot_perm"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        return (
            "A sorting station stands on the floor: a closed gray bin (inner footprint "
            "about 20 x 14 cm, walls 14 cm tall) whose only opening is a straight slot "
            "channel running the length of its dark roof — the gap between the two roof "
            "plates is the GAUGE, and its width changes every episode (it is one of "
            "roughly 23, 37 or 51 mm). Beside the bin sits an open red tray (about "
            "20 x 20 cm inside, 3 cm lip). Scattered on the open floor in front lie "
            "FOUR orange bars, all 95 mm long and all the same color, differing only in "
            "cross-section: 16 x 40, 30 x 48, 44 x 56 and 58 x 66 mm (thickness x "
            "width). The station's position and heading, the gauge gap, and the bars' "
            "resting spots and headings change every episode — read the scene by "
            "looking, and identify each bar by its SIZE.\n"
            "Goal: sort all four bars by the gauge. Every bar THINNER than the slot gap "
            "must be dropped through the slot so it rests inside the bin (hold it with "
            "its long axis vertical and its thin side across the gap, then release it "
            "over the slot; the bin is closed everywhere else, so through the slot is "
            "the only way in). Every bar TOO THICK for the gap must be laid inside the "
            "red tray, within its lip. The thinnest bar (16 mm) always fits and the "
            "thickest (58 mm) never does; whether the middle bars fit depends on this "
            "episode's gap — gauge them against the slot by eye or by trying. Bars may "
            "be sorted in any order. The task is done when every bar rests settled in "
            "its correct destination: a fitting bar left in the tray, on the roof, or "
            "on the floor counts against you, as does a thick bar left anywhere but "
            "the tray."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Sort the four orange bars with the roof slot as a go/no-go gauge: drop "
            "every bar thin enough to pass through the slot into the closed bin, and "
            "lay every bar too thick for the slot inside the red tray. Every bar must "
            "end up in its correct place."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def to_station(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N, 3) world position -> station frame (anchor + yaw removed)."""
        p = pos_w - self.env_origins
        cy, sy = torch.cos(self.yaw), torch.sin(self.yaw)
        dx = p[:, 0] - self.anchor[:, 0]
        dy = p[:, 1] - self.anchor[:, 1]
        return torch.stack([dx * cy + dy * sy, -dx * sy + dy * cy, p[:, 2]], dim=-1)

    def station_dir(self, local_vec) -> torch.Tensor:
        """(N, 3) world direction of a station-frame vector (rotation only)."""
        vx, vy, vz = (float(v) for v in local_vec)
        cy, sy = torch.cos(self.yaw), torch.sin(self.yaw)
        return torch.stack([vx * cy - vy * sy, vx * sy + vy * cy,
                            torch.full_like(cy, vz)], dim=-1)

    def bar_local(self) -> torch.Tensor:
        """(N, 4, 3) bar centers in the station frame."""
        return torch.stack([self.to_station(b.data.root_pos_w) for b in self.bars],
                           dim=1)

    def fits(self) -> torch.Tensor:
        """(N, 4) bool: bar thickness below this episode's gauge gap (>= 5 mm margin
        either way by construction)."""
        return self._thins[None, :] < self.W[:, None]

    def in_bin(self) -> torch.Tensor:
        """(N, 4) bool: bar center inside the bin's inner walls and BELOW the sill
        gate (the bin is closed except for the slot, so this is a real transit)."""
        c = self.cfg
        loc = self.bar_local()
        return ((loc[:, :, 0] - _BIN_C[0]).abs() < c.bin_xy_x) \
            & ((loc[:, :, 1] - _BIN_C[1]).abs() < c.bin_xy_y) \
            & (loc[:, :, 2] < c.bin_z)

    def in_tray(self) -> torch.Tensor:
        """(N, 4) bool: bar center inside the tray lip, down at floor level."""
        c = self.cfg
        loc = self.bar_local()
        return ((loc[:, :, 0] - _TRAY_C[0]).abs() < c.tray_xy) \
            & ((loc[:, :, 1] - _TRAY_C[1]).abs() < c.tray_xy) \
            & (loc[:, :, 2] < c.tray_z)

    def settled(self) -> torch.Tensor:
        """(N, 4) bool: bar |lin vel| below `settle_speed`."""
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.bars],
                          dim=1)
        return vel < self.cfg.settle_speed

    def correct(self) -> torch.Tensor:
        """(N, 4) bool: bar settled in its gauge-correct destination."""
        f = self.fits()
        return ((f & self.in_bin()) | (~f & self.in_tray())) & self.settled()

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: every bar settled in its correct destination."""
        return self.correct().all(dim=1)

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.9 * (correctly routed fraction); 1.0 iff
        success(). Null policy ~0 (all bars loose on the open floor). Credit is
        state-based and stable: a routed bar rests in a walled destination it
        cannot leave without intervention."""
        base = 0.9 * self.correct().float().mean(dim=1)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="gauge_sort", robot="null",
                                      env_spacing=3.0))

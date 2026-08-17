"""QCChuteScene — find the ONE polished (slippery) disc among three identical-looking
discs by physically probing them on an inclined test ramp, then let it slide through the
covered tunnel so it flies into the sealed acceptance cup.

Derived from robosuite/robosuite_env (Lift / Stack / Door / PickPlaceCan /
NutAssemblySquare), but the DECISION PROBLEM is replaced wholesale. Every seed task has
the same shape: the goal object is KNOWN a priori and fully characterized by sight, and
the whole task is a single free grasp-and-carry to a goal pose — the manipulated
object's own pose IS the predicate, and perception is passive (localize, then move).
Here the central unknown is the goal object's IDENTITY, and it is invisible: the three
gray discs are geometrically and visually identical; exactly one is POLISHED (near-
frictionless base) and which one is sampled per episode. The only way to find it is
INTERACTIVE PERCEPTION — place a disc on the open upper section of a steep ramp and
watch the physics: a rough disc sits still (its friction holds it, 3.3x margin), the
polished disc immediately slides away (4.2x margin). And the goal region is SEALED:
the acceptance cup is closed on all sides and roofed; its only opening faces the mouth
of a low covered tunnel (the lower half of the ramp, 12 mm of headroom over the disc,
~0.20 m deep — far beyond any gripper's reach). So grasp-and-carry, the seed's one
skill, provably cannot finish the task: the disc must be DELIVERED BY THE MECHANISM,
sliding the roofed run on its own low friction and flying off the lip into the cup.
The plan is: probe (physics experiment per disc), classify, commit — with a real cost
for guessing wrong: a rough disc shoved down the tunnel jams it or fouls the cup, and
the cup keys on the polished disc only.

Structure (all procedural; one KINEMATIC compound "rig" + three dynamic discs):
  - rig: a chute assembly pitched 12 deg (deck 0.50 m long, inner width 64 mm, side
    walls 34 mm tall) whose UPPER half is open-topped (the probe / loading zone) and
    whose LOWER half is covered by an opaque roof (34 mm underside clearance, disc is
    22 mm tall); past the exit lip a fully enclosed acceptance CUP (raised floor, entry
    sill, far wall, side walls, roof) catches the flying disc; three yellow STAGING
    PADS on the ground in front of the rig hold the discs at reset.
  - discs (x3): identical gray cylinders (r 20 mm, h 22 mm, 100 g). Their friction
    materials use multiply-combine, and every rig face has mu = 1.0, so the pair
    friction IS the disc's own coefficient: polished 0.05/0.04, rough 0.70/0.65
    (tan 12 deg = 0.213 sits between them with >2.5x margin each way — asserted).

Per-episode randomization (readback-verifiable): rig xy jitter + yaw, the PAD SLOT of
each disc (a hidden permutation — which pad holds the polished disc is the discrete
secret, torch.rand-driven), per-disc pad jitter.

Rubric (0..1, anchored in the demonstrated solve; latched credit):
  0.30 track  — the polished disc has been on the ramp deck (the probe/commit zone);
  0.30 tunnel — the polished disc entered the covered tunnel section (only possible by
        sliding: the roof forbids carrying, its friction forbids staying);
  1.0 iff success(): the polished disc rests INSIDE the cup (box gates well inside the
        walls), NO rough disc is in the cup, all three discs settled, finite.
  Non-success is capped at 0.60.

The interlocks are honest by construction: the cup is sealed by geometry (not a rubric
clause), a rough disc physically cannot idle down the tunnel (friction, not fiat), and
identity is enforced by which asset carries the polished material. smoke.py proves each
physically.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
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


# ----- USD authoring helpers (idempotent, single-op) ---------------------------------------------
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


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, *, center, size, color, contact_offset: float):
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
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


# ----- rig geometry (slab lists) ------------------------------------------------------------------
def _level_slabs(c: Any) -> list[tuple[str, float, float, float, float, float, float, tuple]]:
    """(name, x0, x1, y0, y1, z0, z1, color) — rig-root frame (level) slabs: the sealed
    acceptance cup (solid pedestal to the ground, entry sill, far wall, side walls,
    roof) and the three staging pads."""
    cup = c.cup_color
    wy0, wy1 = c.cup_wall_y0, c.cup_wall_y1
    slabs: list = [
        ("pedestal", c.ped_x0, c.ped_x1, -wy1, wy1, 0.0, c.cup_floor_z, cup),
        ("sill", c.sill_x0, c.sill_x1, -wy0, wy0, c.cup_floor_z, c.sill_top, cup),
        ("farwall", c.farw_x0, c.ped_x1, -wy0, wy0, c.cup_floor_z, c.cup_top, cup),
        ("cupwp", c.ped_x0, c.ped_x1, wy0, wy1, c.cup_floor_z, c.cup_top, cup),
        ("cupwn", c.ped_x0, c.ped_x1, -wy1, -wy0, c.cup_floor_z, c.cup_top, cup),
        ("cuproof", c.ped_x0, c.ped_x1, -wy1, wy1, c.cup_top, c.cup_top + c.cup_roof_t, cup),
    ]
    for k, (px, py) in enumerate(c.pad_xy):
        slabs.append((f"pad{k}", px - c.pad_hw, px + c.pad_hw, py - c.pad_hw, py + c.pad_hw,
                      0.0, c.pad_t, c.pad_color))
    return slabs


def _slope_slabs(c: Any) -> list[tuple[str, float, float, float, float, float, float, tuple]]:
    """Chute-frame slabs (x = downslope from the deck-top midpoint, z = deck normal;
    the deck TOP is the z=0 plane): deck, side walls, top cap, tunnel roof."""
    hw, t = c.chute_hw, c.wall_t
    s0, s1 = c.deck_s0, c.deck_s1
    slabs: list = [
        ("deck", s0, s1, -(hw + t), hw + t, -c.deck_t, 0.0, c.deck_color),
        ("wallp", s0, s1, hw, hw + t, 0.0, c.wall_h, c.wall_color),
        ("walln", s0, s1, -(hw + t), -hw, 0.0, c.wall_h, c.wall_color),
        ("topcap", s0, s0 + t, -hw, hw, 0.0, c.wall_h, c.wall_color),
        ("roof", c.roof_s0, c.roof_s1, -(hw + t), hw + t,
         c.wall_h, c.wall_h + c.roof_t, c.roof_color),
    ]
    return slabs


def _spawn_rig(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC rig compound: level cup + pads at the root, plus a pitched
    child Xform carrying the chute (deck / walls / tunnel roof)."""
    from pxr import Gf, UsdGeom, UsdPhysics

    c = cfg.geo
    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.mu_static, cfg.mu_dynamic)
    for name, x0, x1, y0, y1, z0, z1, color in _level_slabs(c):
        p = _box(stage, f"{prim_path}/{name}",
                 center=((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2),
                 size=(x1 - x0, y1 - y0, z1 - z0), color=color,
                 contact_offset=cfg.contact_offset)
        _bind_material(p, mat)
    # pitched chute sub-frame: +x tips DOWN by pitch_deg (rotation about +y)
    chute = UsdGeom.Xform.Define(stage, f"{prim_path}/chute")
    xf = UsdGeom.Xformable(chute)
    xf.AddTranslateOp().Set(Gf.Vec3d(float(c.mid_x), 0.0, float(c.mid_z)))
    half = math.radians(c.pitch_deg) / 2
    xf.AddOrientOp().Set(Gf.Quatf(math.cos(half), Gf.Vec3f(0.0, math.sin(half), 0.0)))
    for name, x0, x1, y0, y1, z0, z1, color in _slope_slabs(c):
        p = _box(stage, f"{prim_path}/chute/{name}",
                 center=((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2),
                 size=(x1 - x0, y1 - y0, z1 - z0), color=color,
                 contact_offset=cfg.contact_offset)
        _bind_material(p, mat)
    return root


_SPAWNER_CACHE: dict[str, Any] = {}


def _spawner_classes() -> dict[str, Any]:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rig" not in _SPAWNER_CACHE:

        @configclass
        class RigSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rig)
            geo: Any = None
            mu_static: float = 1.0
            mu_dynamic: float = 1.0
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["rig"] = RigSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg ---------------------------------------------------------------------------------
@dataclass
class QCChuteSceneCfg(BaseCfg):
    """Config for `QCChuteScene`. Margin ledger (why the mechanism is honest by
    construction, all asserted in __post_init__): tan(12 deg) = 0.213 sits between the
    polished pair-friction (0.05, slides with 4.2x margin) and the rough pair-friction
    (0.70, holds with 3.3x margin); multiply-combine against mu=1.0 rig faces makes the
    pair coefficient exactly the disc's own. The tunnel headroom (12 mm over the disc)
    and depth (~0.20 m) forbid carrying or reaching; the slowest possible transit
    (released at the tunnel mouth) still leaves the lip at ~0.81 m/s and clears the
    entry sill by ~19 mm; the fastest (full ramp) lands well inside the far wall and
    stays ~12 mm under the cup roof."""

    # --- tunable: rubric thresholds ---------------------------------------------------------------
    chute_band_y: float = tunable(0.040)   # |y| bound of the on-deck band (rig local)
    chute_s_lo: float = tunable(-0.26)     # downslope coord band of the deck
    chute_s_hi: float = tunable(0.26)
    chute_h_lo: float = tunable(-0.005)    # height-above-deck band (disc centre ~0.011)
    chute_h_hi: float = tunable(0.035)
    tunnel_s: float = tunable(0.06)        # s beyond this = inside the covered tunnel
    cup_x0: float = tunable(0.262)         # success box (rig local), past the sill span
    cup_x1: float = tunable(0.344)
    cup_y: float = tunable(0.043)
    cup_z_lo: float = tunable(0.026)       # resting on the cup floor (centre ~0.034)
    cup_z_hi: float = tunable(0.048)
    settle_speed: float = tunable(0.10)    # max |lin vel| (all discs) when judging

    # --- tunable: randomization (the task-family knobs) -------------------------------------------
    rig_jitter: float = tunable(0.04)      # rig xy jitter (+/- m)
    rig_yaw_deg: float = tunable(25.0)     # rig yaw about the nominal (+/- deg)
    slot_jitter: float = tunable(0.020)    # per-disc xy jitter on its pad (+/- m)

    # --- info: nominal placement ------------------------------------------------------------------
    rig_pos: tuple = info((0.40, 0.0))
    rig_yaw0: float = info(0.0)            # ramp climbs away to the robot's left

    # --- info: chute structure (chute frame; deck top = z0 plane; see _slope_slabs) ---------------
    pitch_deg: float = info(12.0)
    mid_x: float = info(0.0)               # deck-top midpoint, rig local
    mid_z: float = info(0.107)
    deck_s0: float = info(-0.262)          # deck span along the slope (s1 = exit lip)
    deck_s1: float = info(0.25)
    deck_t: float = info(0.010)
    chute_hw: float = info(0.032)          # inner half width -> 64 mm channel
    wall_t: float = info(0.012)
    wall_h: float = info(0.034)            # wall top = tunnel-roof underside
    roof_s0: float = info(0.05)            # tunnel roof span (lower half + lip overhang)
    roof_s1: float = info(0.253)
    roof_t: float = info(0.008)

    # --- info: cup structure (rig local, level; see _level_slabs) ---------------------------------
    ped_x0: float = info(0.240)
    ped_x1: float = info(0.364)
    cup_floor_z: float = info(0.023)       # pedestal top = cup floor
    sill_x0: float = info(0.246)
    sill_x1: float = info(0.258)
    sill_top: float = info(0.032)
    farw_x0: float = info(0.346)
    cup_wall_y0: float = info(0.045)       # side wall inner face
    cup_wall_y1: float = info(0.057)
    cup_top: float = info(0.089)           # side/far wall top = cup-roof underside
    cup_roof_t: float = info(0.008)

    # --- info: staging pads / discs ---------------------------------------------------------------
    pad_xy: tuple = info(((-0.15, -0.20), (0.0, -0.20), (0.15, -0.20)))
    pad_hw: float = info(0.040)
    pad_t: float = info(0.003)
    puck_r: float = info(0.020)
    puck_h: float = info(0.022)
    puck_mass: float = info(0.10)
    mu_polished_s: float = info(0.05)
    mu_polished_d: float = info(0.04)
    mu_rough_s: float = info(0.70)
    mu_rough_d: float = info(0.65)
    mu_rig: float = info(1.0)

    # --- info: colors / misc ----------------------------------------------------------------------
    deck_color: tuple = info((0.55, 0.58, 0.66))
    wall_color: tuple = info((0.40, 0.42, 0.48))
    roof_color: tuple = info((0.20, 0.20, 0.24))
    cup_color: tuple = info((0.10, 0.52, 0.20))
    pad_color: tuple = info((0.85, 0.75, 0.10))
    puck_color: tuple = info((0.55, 0.55, 0.58))
    contact_offset: float = info(0.0015)
    # rubric weights (0.30 + 0.30 = 0.60 = the non-success cap)
    w_track: float = info(0.30)
    w_tunnel: float = info(0.30)

    def __post_init__(self) -> None:
        g = 9.81
        th = math.radians(self.pitch_deg)
        sl, co = math.sin(th), math.cos(th)
        # friction margins: the pitch separates the two materials with >2.5x each way
        assert math.tan(th) > 2.5 * self.mu_polished_s * self.mu_rig
        assert self.mu_rough_s * self.mu_rig > 2.5 * math.tan(th)
        # tunnel headroom over the disc
        assert self.wall_h - self.puck_h > 0.009
        # slowest transit (released at the tunnel mouth) still clears the entry sill
        a = g * (sl - self.mu_polished_d * self.mu_rig * co)
        v_min = math.sqrt(2.0 * a * (self.deck_s1 - self.roof_s0))
        x_lip = self.mid_x + self.deck_s1 * co
        z_lip = self.mid_z - self.deck_s1 * sl
        t_sill = (self.sill_x1 - x_lip) / (v_min * co)
        drop = v_min * sl * t_sill + 0.5 * g * t_sill**2
        bottom = z_lip + self.puck_h / 2 - drop - self.puck_h / 2
        assert bottom - self.sill_top > 0.004, "sill clips the slowest transit"
        # flight stays under the cup roof
        assert z_lip + self.puck_h > 0.0 and z_lip + self.puck_h < self.cup_top - 0.005
        # cup interior admits the disc and the two rest poses sit inside the success box
        assert self.farw_x0 - self.sill_x1 > 2 * self.puck_r + 0.010
        assert self.cup_x1 - (self.farw_x0 - self.puck_r) > 0.010
        assert (self.sill_x1 + self.puck_r) - self.cup_x0 > 0.010
        assert self.cup_x0 - self.sill_x1 > 0.003          # sill-perch excluded
        assert self.cup_wall_y0 - self.puck_r > self.cup_y - 0.020  # y gate inside walls


# ----- quaternion helper (wxyz, batched, yaw about z) ---------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene ---------------------------------------------------------------------------------------
@SCENES.register("qc_chute")
class QCChuteScene(BaseScene):
    cfg: QCChuteSceneCfg

    PUCK_NAMES = ("puck_0", "puck_1", "puck_2")   # puck_0 carries the polished material

    def __init__(self, cfg: QCChuteSceneCfg | None = None) -> None:
        super().__init__(cfg or QCChuteSceneCfg())

    # ----- assets --------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()

        def puck_spawn(mu_s: float, mu_d: float):
            return sim_utils.CylinderCfg(
                radius=c.puck_r, height=c.puck_h, axis="Z",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4,
                    max_depenetration_velocity=0.5,
                    linear_damping=0.0, angular_damping=0.05,
                    sleep_threshold=0.0, stabilization_threshold=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.puck_mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=mu_s, dynamic_friction=mu_d, restitution=0.0,
                    friction_combine_mode="multiply", restitution_combine_mode="min"),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.puck_color),
            )

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
            "rig": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rig",
                spawn=sp["rig"](geo=c, mu_static=c.mu_rig, mu_dynamic=c.mu_rig,
                                contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rig_pos[0], c.rig_pos[1], 0.0)),
            ),
        }
        mus = [(c.mu_polished_s, c.mu_polished_d),
               (c.mu_rough_s, c.mu_rough_d), (c.mu_rough_s, c.mu_rough_d)]
        for name, (mu_s, mu_d) in zip(self.PUCK_NAMES, mus):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.title().replace("_", ""),
                spawn=puck_spawn(mu_s, mu_d),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.9, 0.9, 0.05)),
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

    # ----- lifecycle -------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.rig: RigidObject = env.iscene["rig"]
        self.pucks: dict[str, RigidObject] = {n: env.iscene[n] for n in self.PUCK_NAMES}
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # per-episode sampled facts: _slot[:, j] = pad index of puck j (col 0 = polished)
        self._slot = torch.zeros(n, 3, dtype=torch.long, device=dev)
        # latches
        self._track = torch.zeros(n, dtype=torch.bool, device=dev)
        self._tunnel = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the rig (yaw + xy jitter), deal the three discs onto
        the three pads through a hidden random permutation (which pad holds the
        polished disc is the per-episode secret), clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        yaw = math.radians(c.rig_yaw0) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rig_yaw_deg)
        qr = _qz(yaw)
        rp = torch.zeros(m, 3, device=dev)
        rp[:, 0] = c.rig_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.rig_jitter
        rp[:, 1] = c.rig_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.rig_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = rp + origin
        st[:, 3:7] = qr
        self.rig.write_root_state_to_sim(st, env_ids)

        def to_world(loc: torch.Tensor) -> torch.Tensor:
            from isaaclab.utils.math import quat_apply

            return rp + origin + quat_apply(qr, loc)

        # hidden permutation (torch.rand + argsort, not randint: first-randint degeneracy)
        order = torch.rand(m, 3, device=dev).argsort(dim=1)
        self._slot[env_ids] = order
        pads = torch.tensor(c.pad_xy, device=dev, dtype=torch.float32)
        for j, name in enumerate(self.PUCK_NAMES):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0:2] = pads[order[:, j]] \
                + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            loc[:, 2] = c.pad_t + c.puck_h / 2 + 0.0015
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = to_world(loc)
            st[:, 3:7] = qr
            self.pucks[name].write_root_state_to_sim(st, env_ids)

        self._track[env_ids] = False
        self._tunnel[env_ids] = False

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "rig": self.rig.data.root_state_w[env_ids].clone(),
            "pucks": {n: p.data.root_state_w[env_ids].clone()
                      for n, p in self.pucks.items()},
            "slot": self._slot[env_ids].clone(),
            "track": self._track[env_ids].clone(),
            "tunnel": self._tunnel[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.rig.write_root_state_to_sim(state["rig"], env_ids)
        for n, p in self.pucks.items():
            p.write_root_state_to_sim(state["pucks"][n], env_ids)
        self._slot[env_ids] = state["slot"]
        self._track[env_ids] = state["track"]
        self._tunnel[env_ids] = state["tunnel"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        return (
            "A quality-control RIG stands on the ground (position and heading vary "
            "per episode). It is a straight RAMP, 0.50 m long and 64 mm wide between "
            "low side walls, tilted 12 degrees. The UPPER half of the ramp is open "
            "from above — this is the TEST SECTION. The LOWER half is covered by a "
            "dark opaque ROOF, forming a low TUNNEL (only 12 mm of clearance over a "
            "disc, about 0.20 m deep — no gripper can reach inside). The tunnel ends "
            "at a drop-off lip; just past it sits the green ACCEPTANCE CUP, which is "
            "completely CLOSED — raised floor, entry sill, side walls, far wall and "
            "a green roof. Its ONLY opening faces the tunnel mouth: the one way "
            "anything gets into the cup is to slide down the ramp, through the "
            "tunnel, and fly off the lip.\n"
            "On three yellow STAGING PADS in front of the rig lie three gray DISCS "
            "(40 mm diameter, 22 mm tall). They look absolutely identical, but "
            "exactly ONE of them is POLISHED: its base is nearly frictionless, "
            "while the other two are ROUGH. Which pad holds the polished disc is "
            "random every episode and CANNOT be seen — it can only be discovered by "
            "experiment: set a disc on the open test section of the ramp and let "
            "go. A rough disc stays put (its friction holds it on the 12-degree "
            "slope with a wide margin); the polished disc immediately slides away "
            "downhill, shoots through the tunnel, and flies into the cup. The discs "
            "may be freely picked up and carried anywhere in the open — but the cup "
            "is sealed, so carrying can never finish the job.\n"
            "Goal: deliver the POLISHED disc into the acceptance cup, and keep the "
            "two rough discs out of it (a rough disc shoved into the tunnel or cup "
            "spoils the batch — success requires the cup to contain the polished "
            "disc and ONLY the polished disc). Success: the polished disc rests on "
            "the cup floor between its walls, no rough disc inside, everything "
            "settled."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Find the one polished (slippery) disc by placing discs on the open "
            "upper test section of the ramp: rough discs stay put, the polished one "
            "slides away on its own. Return any rough disc to its yellow pad, and "
            "let the polished disc slide down through the covered tunnel so it "
            "flies off the lip into the sealed green acceptance cup."
        )

    # ----- frames / predicates ---------------------------------------------------------------------
    def rig_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.rig.data.root_quat_w,
                                  pos_w - self.rig.data.root_pos_w)

    def puck_locs(self) -> torch.Tensor:
        """(N, 3, 3) rig-local disc centres, puck order (index 0 = polished)."""
        return torch.stack(
            [self.rig_local(self.pucks[n].data.root_pos_w) for n in self.PUCK_NAMES],
            dim=1)

    def chute_coords(self, loc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rig-local -> chute coords: s (downslope from the deck-top midpoint),
        y (lateral), h (height above the deck-top plane)."""
        c = self.cfg
        th = math.radians(c.pitch_deg)
        rx = loc[..., 0] - c.mid_x
        rz = loc[..., 2] - c.mid_z
        s = rx * math.cos(th) - rz * math.sin(th)
        h = rx * math.sin(th) + rz * math.cos(th)
        return s, loc[..., 1], h

    def on_deck(self, loc: torch.Tensor) -> torch.Tensor:
        """Bool: the point is riding the ramp deck (in-channel, on-surface band)."""
        c = self.cfg
        s, y, h = self.chute_coords(loc)
        return ((y.abs() < c.chute_band_y) & (s > c.chute_s_lo) & (s < c.chute_s_hi)
                & (h > c.chute_h_lo) & (h < c.chute_h_hi))

    def in_cup(self, loc: torch.Tensor) -> torch.Tensor:
        """Bool: the point rests inside the acceptance cup (box well inside walls)."""
        c = self.cfg
        return ((loc[..., 0] > c.cup_x0) & (loc[..., 0] < c.cup_x1)
                & (loc[..., 1].abs() < c.cup_y)
                & (loc[..., 2] > c.cup_z_lo) & (loc[..., 2] < c.cup_z_hi))

    def settled(self) -> torch.Tensor:
        vels = torch.stack(
            [p.data.root_lin_vel_w.norm(dim=-1) for p in self.pucks.values()], dim=1)
        return (vels < self.cfg.settle_speed).all(dim=1)

    def _update_latches(self) -> None:
        c = self.cfg
        loc = self.rig_local(self.pucks["puck_0"].data.root_pos_w)
        band = self.on_deck(loc)
        s, _, _ = self.chute_coords(loc)
        self._track |= band
        self._tunnel |= band & (s > c.tunnel_s)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric ----------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the POLISHED disc rests inside the sealed cup, NO rough disc is
        in the cup, all three discs settled, finite. Live (revocable) containment."""
        self._update_latches()
        locs = self.puck_locs()
        finite = torch.ones_like(self._track)
        for p in self.pucks.values():
            finite &= torch.isfinite(p.data.root_pos_w).all(dim=-1)
        pol_in = self.in_cup(locs[:, 0])
        rough_in = self.in_cup(locs[:, 1]) | self.in_cup(locs[:, 2])
        return pol_in & ~rough_in & self.settled() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.30 once the polished disc has been on the ramp
        (probe/commit) + 0.30 once it entered the covered tunnel (only reachable by
        sliding), capped at 0.60 — and exactly 1.0 iff success()."""
        c = self.cfg
        self._update_latches()
        base = (c.w_track * self._track.float()
                + c.w_tunnel * self._tunnel.float()).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; discs are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="qc_chute", robot="null"))

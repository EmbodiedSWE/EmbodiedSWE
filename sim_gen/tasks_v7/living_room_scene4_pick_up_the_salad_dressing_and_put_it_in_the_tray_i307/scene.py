"""CrateDockScene — slide the ungraspable amber crate through a walled corridor into a
roofed dock, after clearing the tall blocker crate into the siding (sim_gen task
`living_room_scene4_pick_up_the_salad_dressing_and_put_it_in_the_tray_i307`).

Derived from libero_90 living_room_scene4 "pick up the salad dressing and put it in the
tray", but STRATEGICALLY different: the seed is a direct prehensile transport — grasp
the target bottle among distractors, carry it through free space, drop it inside a
tray's open containment box. Here BOTH the grasp and the drop are removed by design:

  - the payload (the amber "salad-dressing crate", 110 x 110 mm footprint) is WIDER in
    every horizontal direction than a parallel jaw's 80 mm stroke — it cannot be
    grasped or lifted, only PUSHED on its faces;
  - the goal container (the DOCK) is ROOFED: its only entrance is the corridor mouth,
    so nothing can be dropped in from above — the payload must be slid in at deck
    level, through the mouth, along the floor;
  - an ORDERING CONSTRAINT is enforced by geometry, not by rubric fiat: a taller gray
    BLOCKER crate stands in the corridor junction. The blocker is taller than the dock
    mouth (135 mm vs the 112 mm roof underside), so it can never be pushed through and
    out of the way forward — it must first be pushed sideways into the SIDING branch.
    Only then can the payload cross the junction and enter the dock.

A solver therefore needs a different PLAN (route two ungraspable bodies through a
corridor topology in the right order, purely by pushing) and a different CODE
STRUCTURE (rig-frame push-axis control for two different objects on two different
axes, jam detection, a mouth pathway latch) — not a grasp-carry-release routine.

success(): the payload settled FULLY INSIDE the dock (rig-frame containment window,
deck-rest height, velocity gates) AND it got there THROUGH THE MOUTH (latched pathway:
the payload was observed crossing the mouth window — teleporting it through the roof
earns no success). score() is latched credit that never evaporates: 0.15 blocker ever
cleared into the siding + 0.15 payload ever crossed the junction band + 0.20 payload
ever in the mouth window + 0.25 payload ever inside the dock (cap 0.75); exactly 1.0
iff success() live. Null policy scores 0.

Fully procedural geometry (boxes only; no external assets). Per-episode randomization
(readback-verified in smoke): rig yaw +/-30 deg + xy jitter (the lane and siding push
axes must be READ from the scene), payload start slot along the entry lane + small
yaw, blocker jitter in the junction. Heavy imports (isaaclab, pxr) are deferred so
importing this module stays app-free.
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


# ----- USD authoring helpers (kinematic rig compound) -------------------------------------------
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
    """Author one axis-aligned box child prim (translate -> scale; authored once)."""
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


def _spawn_rig(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC rig. Local frame: origin on the GROUND under the entry-lane axis;
    +x = lane direction (entry -> junction -> dock mouth -> dock), +y = siding
    direction. Deck top at cfg.deck_top. All interior faces named in comments."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(60.0)

    co = cfg.contact_offset
    slick = _friction_material(stage, f"{prim_path}/mat_slick",
                               cfg.mu_deck, cfg.mu_deck * 0.85)
    wallm = _friction_material(stage, f"{prim_path}/mat_wall", 0.30, 0.25)
    zt = cfg.deck_top
    gray = (0.45, 0.45, 0.48)
    dark = (0.26, 0.26, 0.30)
    roofc = (0.20, 0.42, 0.28)

    # deck slab: x [-0.36, 0.36], y [-0.15, 0.37], top at deck_top (slick)
    _box(stage, f"{prim_path}/deck", (0.72, 0.52, zt), (0.0, 0.11, zt / 2),
         gray, co, material=slick)
    # south wall, full length: interior face y = -0.070
    _box(stage, f"{prim_path}/wall_south", (0.72, 0.012, 0.055),
         (0.0, -0.076, zt + 0.0275), dark, co, material=wallm)
    # north entry wall: x [-0.36, +0.020], interior face y = +0.070
    _box(stage, f"{prim_path}/wall_north", (0.38, 0.012, 0.055),
         (-0.17, 0.076, zt + 0.0275), dark, co, material=wallm)
    # siding west wall: interior face x = +0.020, y [0.076, 0.326]
    _box(stage, f"{prim_path}/siding_w", (0.012, 0.25, 0.055),
         (0.014, 0.201, zt + 0.0275), dark, co, material=wallm)
    # siding east wall: interior face x = +0.160 (the dock mouth plane), y [0.076, 0.326]
    _box(stage, f"{prim_path}/siding_e", (0.012, 0.25, 0.055),
         (0.166, 0.201, zt + 0.0275), dark, co, material=wallm)
    # siding end wall: interior face y = +0.324
    _box(stage, f"{prim_path}/siding_end", (0.164, 0.012, 0.055),
         (0.09, 0.330, zt + 0.0275), dark, co, material=wallm)
    # dock north wall: x [0.160, 0.360], tall to the roof
    _box(stage, f"{prim_path}/dock_n", (0.20, 0.012, cfg.roof_under - zt),
         (0.26, 0.076, (zt + cfg.roof_under) / 2), dark, co, material=wallm)
    # dock back wall: interior face x = +0.360
    _box(stage, f"{prim_path}/dock_back", (0.012, 0.164, cfg.roof_under - zt),
         (0.366, 0.0, (zt + cfg.roof_under) / 2), dark, co, material=wallm)
    # south-wall topper over the dock span (fills wall-top .. roof gap)
    _box(stage, f"{prim_path}/dock_s_top", (0.20, 0.012, cfg.roof_under - zt - 0.055),
         (0.26, -0.076, (zt + 0.055 + cfg.roof_under) / 2), dark, co, material=wallm)
    # ROOF: x [0.150, 0.366] (front edge 10 mm proud of the mouth), underside at
    # roof_under — the payload passes with clearance, the blocker jams against it
    _box(stage, f"{prim_path}/roof", (0.216, 0.196, 0.014),
         (0.258, 0.0, cfg.roof_under + 0.007), roofc, co, material=wallm)
    return root


def _spawner_classes() -> dict[str, Any]:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rig" not in _SPAWNER_CACHE:

        @configclass
        class RigSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rig)
            deck_top: float = 0.040
            roof_under: float = 0.112
            mu_deck: float = 0.15
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["rig"] = RigSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CrateDockSceneCfg(BaseCfg):
    """Config for `CrateDockScene`. Honesty asserted in __post_init__: the payload
    passes under the roof with clearance, the blocker cannot, neither crate can be
    grasped by an 80 mm jaw, neither can spin inside the lane, and the containment
    window is reachable (back-wall rest inside it)."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    settle_lin: float = tunable(0.05)   # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.8)    # max |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    rig_yaw_deg: float = tunable(30.0)  # rig yaw, +/- deg (push axes must be read)
    rig_jitter: float = tunable(0.04)   # rig xy jitter (+/- m)
    start_x_span: float = tunable(0.035)  # payload start slot along the lane (+/- m)
    start_y_span: float = tunable(0.010)  # payload lateral start (+/- m; 5 mm wall clear)
    start_yaw_deg: float = tunable(6.0)   # payload start yaw (+/- deg)
    blocker_x_jit: float = tunable(0.008)  # blocker junction jitter (+/- m)
    blocker_y_jit: float = tunable(0.010)

    # --- info: rig geometry (rig frame: origin on the ground, +x lane, +y siding) ---------------
    deck_top: float = info(0.040)       # deck surface height
    lane_half: float = info(0.070)      # lane interior half-width (interior 140 mm)
    mouth_x: float = info(0.160)        # dock mouth plane (siding east wall interior face)
    roof_under: float = info(0.112)     # roof underside height
    roof_front_x: float = info(0.150)   # roof front edge (proud of the mouth)
    dock_back_x: float = info(0.360)    # dock back wall interior face
    siding_x: tuple = info((0.020, 0.160))  # siding interior x span
    siding_end_y: float = info(0.324)   # siding end wall interior face
    mu_deck: float = info(0.15)
    # --- info: crates ---------------------------------------------------------------------------
    payload_w: float = info(0.110)      # payload footprint (square)
    payload_h: float = info(0.055)
    payload_mass: float = info(0.40)
    blocker_w: float = info(0.110)
    blocker_h: float = info(0.095)      # deck_top + 0.095 = 0.135 > roof_under: jams
    blocker_mass: float = info(0.70)
    mu_crate: float = info(0.15)        # pair-averaged with the slick deck -> 0.15
    payload_color: tuple = info((0.95, 0.62, 0.10))
    blocker_color: tuple = info((0.30, 0.34, 0.42))
    contact_offset: float = info(0.0015)
    # --- info: nominal starts (rig frame) -------------------------------------------------------
    payload_start: tuple = info((-0.27, 0.0))
    blocker_start: tuple = info((0.091, 0.0))
    # --- info: rubric windows (rig frame) -------------------------------------------------------
    junction_band: tuple = info((0.08, 0.14))  # payload must CROSS this band (60 mm wide,
    #                                            uncrossable in one 120 Hz step at the push caps)
    mouth_band: tuple = info((0.13, 0.21))     # mouth pathway window
    dock_x: tuple = info((0.225, 0.315))       # payload center: fully inside the dock
    dock_y_tol: float = info(0.060)
    cleared_y: float = info(0.15)              # blocker center past this = out of the corridor
    # rubric weights (0.15 + 0.15 + 0.20 + 0.25 = 0.75 = the non-success cap)
    w_cleared: float = info(0.15)
    w_junction: float = info(0.15)
    w_mouth: float = info(0.20)
    w_dock: float = info(0.25)

    def __post_init__(self) -> None:
        zt = self.deck_top
        # payload passes under the roof with real clearance; the blocker cannot
        assert zt + self.payload_h + 0.010 < self.roof_under, "payload does not fit the mouth"
        assert zt + self.blocker_h > self.roof_under + 0.015, "blocker would fit the mouth"
        # both crates are UNGRASPABLE by an 80 mm parallel jaw (min horizontal span)
        assert min(self.payload_w, self.blocker_w) >= 0.100, "crate would fit the jaw"
        # the lane admits a crate with slack but cannot let it spin (diagonal > width)
        lane_w = 2 * self.lane_half
        for w in (self.payload_w, self.blocker_w):
            assert lane_w - w >= 0.020, "lane too tight"
            assert w * math.sqrt(2.0) > lane_w + 0.010, "crate could spin in the lane"
        # containment window: nonempty, past the mouth by a full crate + 10 mm, and the
        # back-wall rest pose lands inside it
        assert self.dock_x[0] > self.mouth_x + self.payload_w / 2 + 0.008
        rest_x = self.dock_back_x - self.payload_w / 2 - 0.004
        assert self.dock_x[0] + 0.01 < rest_x < self.dock_x[1] - 0.005, "back-wall rest outside window"
        # blocker cleared: reachable inside the siding with margin
        assert self.cleared_y + 0.04 < self.siding_end_y - self.blocker_w / 2
        # junction band wider than one step of travel at generous speed (no skip)
        assert self.junction_band[1] - self.junction_band[0] > 0.03
        # spawn clearances: payload 5 mm off the walls, blocker 5 mm off the siding walls
        assert self.payload_w / 2 + self.start_y_span + 0.004 < self.lane_half
        assert self.blocker_start[0] + self.blocker_x_jit + self.blocker_w / 2 + 0.005 \
            < self.siding_x[1]
        assert self.blocker_start[0] - self.blocker_x_jit - self.blocker_w / 2 - 0.005 \
            > self.siding_x[0]


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


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("crate_dock")
class CrateDockScene(BaseScene):
    cfg: CrateDockSceneCfg

    def __init__(self, cfg: CrateDockSceneCfg | None = None) -> None:
        super().__init__(cfg or CrateDockSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        crate_props = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.10, angular_damping=0.50,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=16, solver_velocity_iteration_count=1)
        crate_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.mu_crate, dynamic_friction=c.mu_crate * 0.85,
            restitution=0.0)

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.80, dynamic_friction=0.70, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "rig": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rig",
                spawn=sp["rig"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=60.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    deck_top=c.deck_top, roof_under=c.roof_under,
                    mu_deck=c.mu_deck, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "payload": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Payload",
                spawn=sim_utils.CuboidCfg(
                    size=(c.payload_w, c.payload_w, c.payload_h),
                    rigid_props=crate_props.replace(),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=crate_mat.replace(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.payload_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.payload_color)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.payload_start[0], c.payload_start[1],
                         c.deck_top + c.payload_h / 2 + 0.002)),
            ),
            "blocker": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Blocker",
                spawn=sim_utils.CuboidCfg(
                    size=(c.blocker_w, c.blocker_w, c.blocker_h),
                    rigid_props=crate_props.replace(),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=crate_mat.replace(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.blocker_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.blocker_color)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.blocker_start[0], c.blocker_start[1],
                         c.deck_top + c.blocker_h / 2 + 0.002)),
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

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.rig: RigidObject = env.iscene["rig"]
        self.payload: RigidObject = env.iscene["payload"]
        self.blocker: RigidObject = env.iscene["blocker"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # latches (partial credit survives transients; success needs the mouth latch)
        self._cleared = torch.zeros(n, dtype=torch.bool, device=dev)
        self._junction = torch.zeros(n, dtype=torch.bool, device=dev)
        self._mouth = torch.zeros(n, dtype=torch.bool, device=dev)
        self._dock_ever = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the rig (yaw + xy jitter), seat the payload in its
        entry-lane start slot (x span + small lateral + small yaw), seat the blocker
        in the junction (small jitter), clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        from isaaclab.utils.math import quat_apply

        _ = torch.rand(m, device=dev)  # burn the first post-seed draw (degenerate)

        psi = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rig_yaw_deg)
        q_rig = _qz(psi)
        rp = torch.zeros(m, 3, device=dev)
        rp[:, 0] = (torch.rand(m, device=dev) * 2 - 1) * c.rig_jitter
        rp[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.rig_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = rp + origin
        st[:, 3:7] = q_rig
        self.rig.write_root_state_to_sim(st, env_ids)

        # payload: entry-lane slot, 2 mm drop, small yaw about the rig heading
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.payload_start[0] + (torch.rand(m, device=dev) * 2 - 1) * c.start_x_span
        loc[:, 1] = c.payload_start[1] + (torch.rand(m, device=dev) * 2 - 1) * c.start_y_span
        loc[:, 2] = c.deck_top + c.payload_h / 2 + 0.002
        dyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.start_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = rp + origin + quat_apply(q_rig, loc)
        st[:, 3:7] = _qmul(q_rig, _qz(dyaw))
        self.payload.write_root_state_to_sim(st, env_ids)

        # blocker: junction, aligned with the rig, small jitter
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.blocker_start[0] + (torch.rand(m, device=dev) * 2 - 1) * c.blocker_x_jit
        loc[:, 1] = c.blocker_start[1] + (torch.rand(m, device=dev) * 2 - 1) * c.blocker_y_jit
        loc[:, 2] = c.deck_top + c.blocker_h / 2 + 0.002
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = rp + origin + quat_apply(q_rig, loc)
        st[:, 3:7] = q_rig
        self.blocker.write_root_state_to_sim(st, env_ids)

        self._cleared[env_ids] = False
        self._junction[env_ids] = False
        self._mouth[env_ids] = False
        self._dock_ever[env_ids] = False

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {n: getattr(self, n).data.root_state_w[env_ids].clone()
               for n in ("rig", "payload", "blocker")}
        out["latches"] = torch.stack(
            [self._cleared[env_ids], self._junction[env_ids],
             self._mouth[env_ids], self._dock_ever[env_ids]], dim=1).clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for n in ("rig", "payload", "blocker"):
            getattr(self, n).write_root_state_to_sim(state[n], env_ids)
        lat = state["latches"]
        self._cleared[env_ids] = lat[:, 0]
        self._junction[env_ids] = lat[:, 1]
        self._mouth[env_ids] = lat[:, 2]
        self._dock_ever[env_ids] = lat[:, 3]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A flat gray DECK ({c.deck_top * 1000:.0f} mm high) stands on the floor; "
            f"its position and heading change every episode, so read its geometry from "
            f"the scene. Low dark walls on the deck form a straight ENTRY LANE "
            f"({2 * c.lane_half * 1000:.0f} mm wide) that runs into a JUNCTION, where "
            f"a SIDING branch opens to one side and the lane continues forward into a "
            f"DOCK — a bay covered by a green ROOF whose underside is only "
            f"{c.roof_under * 1000:.0f} mm above the floor level of the deck's "
            f"{c.deck_top * 1000:.0f} mm surface (opening height "
            f"{(c.roof_under - c.deck_top) * 1000:.0f} mm). The roofed dock's ONLY "
            f"entrance is its mouth at the end of the lane.\n"
            f"Two crates sit on the deck. The AMBER crate (the payload, "
            f"{c.payload_w * 1000:.0f} x {c.payload_w * 1000:.0f} x "
            f"{c.payload_h * 1000:.0f} mm) waits in the entry lane. The taller GRAY "
            f"crate ({c.blocker_w * 1000:.0f} x {c.blocker_w * 1000:.0f} x "
            f"{c.blocker_h * 1000:.0f} mm) stands in the junction, blocking the way "
            f"to the dock. Both crates are wider than any parallel-jaw gripper can "
            f"open — they cannot be grasped or lifted, only PUSHED on their side "
            f"faces. The gray crate is taller than the dock opening, so it can never "
            f"go into the dock; the amber crate fits under the roof with room to "
            f"spare.\n"
            f"Goal: get the AMBER crate to rest FULLY INSIDE the covered dock. First "
            f"push the gray blocker sideways out of the junction into the siding "
            f"branch (past the corridor), then push the amber crate along the lane, "
            f"through the junction and the mouth, under the roof, until it sits "
            f"completely inside the dock (pushing it gently against the back wall is "
            f"fine). The amber crate must enter through the mouth — the roof makes "
            f"dropping it in impossible. Success: the amber crate at rest fully "
            f"inside the dock."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Push the tall gray crate out of the junction into the siding, then push "
            "the amber crate along the walled lane, through the mouth, under the "
            "green roof until it rests fully inside the covered dock. The crates are "
            "too wide to grasp — slide them on the deck; the amber crate must enter "
            "the dock through its mouth."
        )

    # ----- frames / live predicates -------------------------------------------------------------
    def _rig_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.rig.data.root_quat_w,
                                  pos_w - self.rig.data.root_pos_w)

    def _settled(self, body) -> torch.Tensor:
        c = self.cfg
        return (body.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    def payload_in_dock(self) -> torch.Tensor:
        """(N,) bool: payload center inside the dock containment window (rig frame),
        at deck-rest height — live geometric containment, fully past the mouth."""
        c = self.cfg
        loc = self._rig_local(self.payload.data.root_pos_w)
        z0 = c.deck_top + c.payload_h / 2
        return (loc[:, 0] > c.dock_x[0]) & (loc[:, 0] < c.dock_x[1]) \
            & (loc[:, 1].abs() < c.dock_y_tol) \
            & ((loc[:, 2] - z0).abs() < 0.010)

    def blocker_cleared(self) -> torch.Tensor:
        """(N,) bool: blocker center well inside the siding, out of the corridor."""
        c = self.cfg
        loc = self._rig_local(self.blocker.data.root_pos_w)
        return (loc[:, 1] > c.cleared_y) \
            & (loc[:, 0] > c.siding_x[0] - 0.01) & (loc[:, 0] < c.siding_x[1] + 0.01)

    def _payload_bands(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(junction_crossing, mouth_window) booleans for the payload center."""
        c = self.cfg
        loc = self._rig_local(self.payload.data.root_pos_w)
        zt = (loc[:, 2] > c.deck_top) & (loc[:, 2] < c.roof_under)
        iny = loc[:, 1].abs() < c.lane_half + 0.01
        junc = (loc[:, 0] > c.junction_band[0]) & (loc[:, 0] < c.junction_band[1]) \
            & iny & zt
        mouth = (loc[:, 0] > c.mouth_band[0]) & (loc[:, 0] < c.mouth_band[1]) \
            & iny & zt
        return junc, mouth

    def _update_latches(self) -> None:
        junc, mouth = self._payload_bands()
        self._cleared |= self.blocker_cleared()
        self._junction |= junc
        self._mouth |= mouth
        self._dock_ever |= self.payload_in_dock()

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric -------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the payload settled fully inside the roofed dock AND it got
        there through the mouth (latched pathway) — finite. The judged containment is
        live physics; the pathway latch rejects any through-the-roof bypass."""
        self._update_latches()
        finite = torch.isfinite(self.payload.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.blocker.data.root_pos_w).all(dim=-1)
        return self.payload_in_dock() & self._settled(self.payload) & self._mouth & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: latched stage credit (never evaporates) — blocker
        cleared 0.15, junction crossed 0.15, mouth window 0.20, ever in the dock 0.25
        (cap 0.75); exactly 1.0 iff success()."""
        c = self.cfg
        self._update_latches()
        base = (c.w_cleared * self._cleared.float()
                + c.w_junction * self._junction.float()
                + c.w_mouth * self._mouth.float()
                + c.w_dock * self._dock_ever.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="crate_dock", robot="null"))

"""FalseworkTentScene — build a free-standing card-tent from two panels using the green
pillar as TEMPORARY falsework, then remove and park the pillar.

Derived from maniskill/stack_pyramid ("pick up the red cube, place it next to the green
cube, stack the blue cube on top of both"), but the STRUCTURAL PRINCIPLE is inverted.
The seed's plan is repeated free-space pick-and-place: every intermediate state is
independently stable (a cube sits wherever you put it), the goal is a pose relation
checked by bounding boxes, and nothing ever has to be supported while being built.
Here the goal structure is a MUTUALLY-SUPPORTING pair — two thin panels leaning
against each other in an inverted V ("card tent") on the white build pad. Neither
panel can stand alone at the required lean (8..55 deg off vertical): released
unsupported, it falls. A single parallel-jaw arm cannot hold one panel while placing
the other, so the tent CANNOT be assembled by direct sequential placement. The scene
provides the classical civil-engineering answer: FALSEWORK. A heavy green pillar
stands at the tent site; each panel can be leaned against a pillar face (each such
intermediate IS stable), and lifting the pillar straight up out of the pair lets the
panels fall inward onto each other and settle into the self-supporting tent. Success
additionally requires the pillar to be REMOVED (>= `clear_r` from both panels — the
tent must carry itself) and parked on the magenta pad. The seed's strategy — stack the
panels flat on top of each other — is a settled state this rubric rejects (smoke #4).

Bodies (all procedural, no external assets):
  - panel_a: BLUE panel 170 x 100 x 14 mm (local z = long axis), 80 g.
  - panel_b: RED panel 150 x 100 x 14 mm, 70 g.
  - pillar: GREEN falsework pillar — dynamic compound body: column 80 x 24 x 190 mm
    plus a YELLOW T-handle bar 160 x 24 x 24 mm on top (grasp the handle ends, which
    overhang the panels' 100 mm width, and lift straight up). ~0.7 kg, CoM authored
    at 90 mm: stable under one-sided panel loads (10x tipping margin). Its faces are
    bound to a LOW-friction material with combine mode "min", so the vertical
    extraction slides freely and does not drag the tent apart.
  - build_pad: WHITE kinematic plate 400 x 400 x 8 mm — the tent site (high
    friction: the panel bases must not creep outward).
  - park_pad: MAGENTA kinematic plate 220 x 220 x 8 mm — where the pillar ends up.

Per-episode randomization (readback-verified in smoke): build pad xy + yaw (the tent
axis heading varies), park pad xy, panel scatter xy + free yaw, and which scatter slot
each panel occupies.

Rubric (0..1; latched partial credit anchored in the demonstrated solve trajectory):
  0.25 * lean  — some panel held a settled lean (tilt in band, base down, in the
                 build zone) — the falsework skill (latched, counter-debounced)
  0.30 * pair  — BOTH panels simultaneously held settled opposed leans (latched)
  0.30 * free  — the full tent predicate held with the pillar CLEAR (latched)
  1.0 iff success() — tent standing (both panels tilted 8..55 deg, bases down inside
                 the build zone, top edges within `apex_gap`, apex above `apex_min_h`,
                 lean directions opposed, pillar >= `clear_r` from both panels) AND
                 pillar resting on the park pad, everything settled and finite.
                 Non-success is capped at 0.85.

The mutual-support claim is honest physics, not fiat: a lone panel at an in-band lean
has its CoM outside its 14 mm footprint and falls (the tilt band plus the >= 3 s
persistence window make a faked lean unreachable), and `clear_r` exceeds the pillar's
maximum horizontal reach (80 mm half-handle) plus the apex offset of any in-band
panel (~60 mm), so no pillar placement outside `clear_r` can prop a counted tent.

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


# ----- custom compound spawner (pillar) ---------------------------------------------------------
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


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _add_box(stage, path: str, *, center, size, color, collide: Callable):
    """One box child: translate + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _bind_phys_material(stage, mat_path: str, prims, *, static: float, dynamic: float,
                        restitution: float, combine: str) -> None:
    """Author a UsdShade physics material and bind it to `prims`. Custom-spawner
    colliders otherwise fall back to a ~0.5-friction default; the pillar NEEDS a low,
    combine-min material so the vertical extraction slides instead of dragging."""
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, mat_path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(static))
    api.CreateDynamicFrictionAttr(float(dynamic))
    api.CreateRestitutionAttr(float(restitution))
    pxm = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    pxm.CreateFrictionCombineModeAttr(combine)
    pxm.CreateRestitutionCombineModeAttr("min")
    for prim in prims:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics")


def _spawn_pillar(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the falsework pillar at `prim_path`: DYNAMIC compound (column + handle).
    Local frame: origin at the base centre on the ground. Everything physical (rigid
    body, mass/CoM, damping, solver iters, low-friction combine-min material) is
    authored here — the clone-wrapped custom func gets no schema help."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.1)
    px.CreateAngularDampingAttr(0.1)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    # explicit CoM (mid-column): predictable statics regardless of MassAPI defaults
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, float(cfg.com_z)))

    collide = _make_collide(cfg.contact_offset)
    c = cfg
    col = _add_box(stage, f"{prim_path}/column",
                   center=(0.0, 0.0, c.col_h / 2),
                   size=(c.col_x, c.col_y, c.col_h),
                   color=c.column_color, collide=collide)
    han = _add_box(stage, f"{prim_path}/handle",
                   center=(0.0, 0.0, c.col_h + c.handle_t / 2),
                   size=(c.handle_len, c.handle_w, c.handle_t),
                   color=c.handle_color, collide=collide)
    _bind_phys_material(stage, f"{prim_path}/physmat", (col, han),
                        static=c.friction, dynamic=c.friction * 0.9,
                        restitution=0.0, combine="min")
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclass (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "pillar" not in _SPAWNER_CACHE:

        @configclass
        class PillarSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pillar)
            col_x: float = 0.08
            col_y: float = 0.024
            col_h: float = 0.19
            handle_len: float = 0.16
            handle_w: float = 0.024
            handle_t: float = 0.024
            mass: float = 0.7
            com_z: float = 0.09
            friction: float = 0.12
            column_color: tuple = (0.10, 0.55, 0.20)
            handle_color: tuple = (0.92, 0.80, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["pillar"] = PillarSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg --------------------------------------------------------------------------------
@dataclass
class FalseworkTentSceneCfg(BaseCfg):
    """Config for `FalseworkTentScene`. Honesty bounds: `clear_r` (0.15) > pillar max
    horizontal reach (0.08 half-handle) + in-band apex-to-centre offset of a panel
    ((L/2)*sin(55 deg) ~= 0.07 worst case, ~0.06 at demonstrated tilts), so no pillar
    placement that satisfies the clearance can physically prop a counted tent; the
    tilt band (8..55 deg) excludes flat panels (90 deg), flat stacks (90 deg), and
    half-collapses (> 55 deg); a lone unsupported panel inside the band falls (14 mm
    footprint), so persistence forces real mutual support."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    tilt_lo_deg: float = tunable(8.0)    # min lean of each panel's long axis off vertical
    tilt_hi_deg: float = tunable(55.0)   # max lean (beyond = collapsed / ramped)
    apex_gap: float = tunable(0.06)      # top-edge midpoints of the panels within this (m)
    apex_min_h: float = tunable(0.10)    # apex height above the build-pad top (m)
    opposed_max_dot: float = tunable(-0.5)  # lean-direction cosine: opposed = dot < this
    base_z_max: float = tunable(0.030)   # panel bottom-edge midpoint within this of the pad top
    zone_r: float = tunable(0.17)        # panel bottoms within this of the build-pad centre
    clear_r: float = tunable(0.15)       # pillar horiz distance to EACH panel centre (freedom)
    park_tol: float = tunable(0.11)      # pillar xy within this of the park-pad centre
    park_z_max: float = tunable(0.15)    # pillar root below this = resting, not carried
    settle_speed: float = tunable(0.06)  # max |lin vel| of every dynamic body when judging
    latch_steps: int = tunable(5)        # consecutive still steps before a latch fires

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    build_jitter: float = tunable(0.05)   # build pad xy jitter (+/- m)
    build_yaw_deg: float = tunable(40.0)  # build pad yaw (+/- deg): tent axis heading varies
    park_jitter: float = tunable(0.05)    # park pad xy jitter (+/- m)
    panel_jitter: float = tunable(0.04)   # panel scatter xy jitter (+/- m)
    panel_yaw_deg: float = tunable(180.0)  # panel scatter yaw (+/- deg)
    slot_swap: bool = tunable(True)       # randomly swap which scatter slot each panel takes

    # --- info: layout (nominal, world xy) -------------------------------------------------------
    build_pos: tuple = info((0.45, 0.00))
    park_pos: tuple = info((0.15, -0.38))
    panel_slots: tuple = info(((0.12, 0.32), (0.42, 0.34)))
    # --- info: panels (local frame: x=width, y=thickness, z=length) -----------------------------
    panel_l: tuple = info((0.17, 0.15))   # (panel_a BLUE long, panel_b RED short)
    panel_w: float = info(0.10)
    panel_t: float = info(0.014)
    panel_mass: tuple = info((0.08, 0.07))
    # --- info: pillar (see PillarSpawnerCfg; duplicated here for geometry math) -----------------
    col_x: float = info(0.08)
    col_y: float = info(0.024)  # THIN: the released panels have only a small arc to close
    col_h: float = info(0.19)
    handle_len: float = info(0.16)
    pillar_mass: float = info(0.7)
    # --- info: pads -----------------------------------------------------------------------------
    build_pad_size: float = info(0.40)
    park_pad_size: float = info(0.22)
    pad_t: float = info(0.008)
    # --- info: colors / misc --------------------------------------------------------------------
    panel_a_color: tuple = info((0.12, 0.25, 0.90))   # blue
    panel_b_color: tuple = info((0.88, 0.10, 0.10))   # red
    build_pad_color: tuple = info((0.92, 0.92, 0.92))  # white
    park_pad_color: tuple = info((0.80, 0.10, 0.80))   # magenta
    contact_offset: float = info(0.002)
    # rubric weights (0.25 + 0.30 + 0.30 = 0.85 = the non-success cap)
    w_lean: float = info(0.25)
    w_pair: float = info(0.30)
    w_free: float = info(0.30)


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


def _qx(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 1] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene ------------------------------------------------------------------------------------
@SCENES.register("falsework_tent")
class FalseworkTentScene(BaseScene):
    cfg: FalseworkTentSceneCfg

    PANEL_NAMES = ("panel_a", "panel_b")

    def __init__(self, cfg: FalseworkTentSceneCfg | None = None) -> None:
        super().__init__(cfg or FalseworkTentSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        pillar_spawn = _spawner_classes()["pillar"](
            col_x=c.col_x, col_y=c.col_y, col_h=c.col_h,
            handle_len=c.handle_len, mass=c.pillar_mass,
            contact_offset=c.contact_offset)

        def dyn_props():
            return sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.1, angular_damping=0.2,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1)

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
            "build_pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BuildPad",
                spawn=sim_utils.CuboidCfg(
                    size=(c.build_pad_size, c.build_pad_size, c.pad_t),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.0, dynamic_friction=0.95, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.build_pad_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.build_pos[0], c.build_pos[1], c.pad_t / 2)),
            ),
            "park_pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/ParkPad",
                spawn=sim_utils.CuboidCfg(
                    size=(c.park_pad_size, c.park_pad_size, c.pad_t),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.7, dynamic_friction=0.6, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.park_pad_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.park_pos[0], c.park_pos[1], c.pad_t / 2)),
            ),
            "pillar": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pillar",
                spawn=pillar_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.build_pos[0], c.build_pos[1], c.pad_t + 0.002)),
            ),
        }
        for i, name in enumerate(self.PANEL_NAMES):
            color = c.panel_a_color if i == 0 else c.panel_b_color
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Panel_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=(c.panel_w, c.panel_t, c.panel_l[i]),
                    rigid_props=dyn_props(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.panel_mass[i]),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.003, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.85, dynamic_friction=0.75, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.panel_slots[i][0], c.panel_slots[i][1],
                         c.panel_t / 2 + 0.002)),
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

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        self.pillar: RigidObject = env.iscene["pillar"]
        self.build_pad: RigidObject = env.iscene["build_pad"]
        self.park_pad: RigidObject = env.iscene["park_pad"]
        self.panels: dict[str, RigidObject] = {
            n: env.iscene[n] for n in self.PANEL_NAMES}
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self._len = torch.tensor(c.panel_l, device=dev)  # (2,)
        self.build_xy = torch.zeros(n, 2, device=dev)    # world xy (incl. origins)
        self.build_yaw = torch.zeros(n, device=dev)
        self.park_xy = torch.zeros(n, 2, device=dev)
        self.slot_swapped = torch.zeros(n, dtype=torch.bool, device=dev)
        # latches (partial credit survives transients; success is judged live) + debounce counters
        self._lean = torch.zeros(n, dtype=torch.bool, device=dev)
        self._pair = torch.zeros(n, dtype=torch.bool, device=dev)
        self._free = torch.zeros(n, dtype=torch.bool, device=dev)
        self._lean_ct = torch.zeros(n, dtype=torch.long, device=dev)
        self._pair_ct = torch.zeros(n, dtype=torch.long, device=dev)
        self._free_ct = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the build pad (xy jitter + yaw) with the pillar
        standing at its centre, the park pad (xy jitter), and the two panels lying
        FLAT at their (possibly swapped) scatter slots with free yaw. Clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- build pad: kinematic, xy jitter + yaw ---
        psi = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.build_yaw_deg)
        bxy = torch.zeros(m, 2, device=dev)
        bxy[:, 0] = c.build_pos[0]
        bxy[:, 1] = c.build_pos[1]
        bxy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.build_jitter
        bxy += origin[:, 0:2]
        self.build_xy[env_ids] = bxy
        self.build_yaw[env_ids] = psi
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = bxy
        st[:, 2] = origin[:, 2] + c.pad_t / 2
        st[:, 3:7] = _qz(psi)
        self.build_pad.write_root_state_to_sim(st, env_ids)

        # --- park pad: kinematic, xy jitter ---
        pxy = torch.zeros(m, 2, device=dev)
        pxy[:, 0] = c.park_pos[0]
        pxy[:, 1] = c.park_pos[1]
        pxy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.park_jitter
        pxy += origin[:, 0:2]
        self.park_xy[env_ids] = pxy
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = pxy
        st[:, 2] = origin[:, 2] + c.pad_t / 2
        st[:, 3] = 1.0
        self.park_pad.write_root_state_to_sim(st, env_ids)

        # --- pillar: standing at the build-pad centre, aligned with the pad yaw ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = bxy
        st[:, 2] = origin[:, 2] + c.pad_t + 0.002
        st[:, 3:7] = _qz(psi)
        self.pillar.write_root_state_to_sim(st, env_ids)

        # --- panels: lying FLAT at scatter slots (possibly swapped), free yaw ---
        swap = torch.rand(m, device=dev) < 0.5 if c.slot_swap \
            else torch.zeros(m, dtype=torch.bool, device=dev)
        self.slot_swapped[env_ids] = swap
        slots = torch.tensor(c.panel_slots, device=dev)  # (2, 2)
        for i, name in enumerate(self.PANEL_NAMES):
            slot_idx = torch.where(swap, torch.tensor(1 - i, device=dev),
                                   torch.tensor(i, device=dev))
            xy = slots[slot_idx] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.panel_jitter
            yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.panel_yaw_deg)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy + origin[:, 0:2]
            st[:, 2] = origin[:, 2] + c.panel_t / 2 + 0.002
            # flat: local y (thickness) -> world z; q = qz(yaw) * qx(90 deg)
            st[:, 3:7] = _qmul(_qz(yaw), _qx(torch.full((m,), math.pi / 2, device=dev)))
            self.panels[name].write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        for t in (self._lean, self._pair, self._free):
            t[env_ids] = False
        for t in (self._lean_ct, self._pair_ct, self._free_ct):
            t[env_ids] = 0

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "pillar": self.pillar.data.root_state_w[env_ids].clone(),
            "build_pad": self.build_pad.data.root_state_w[env_ids].clone(),
            "park_pad": self.park_pad.data.root_state_w[env_ids].clone(),
            "panels": {n: b.data.root_state_w[env_ids].clone()
                       for n, b in self.panels.items()},
            "build_xy": self.build_xy[env_ids].clone(),
            "build_yaw": self.build_yaw[env_ids].clone(),
            "park_xy": self.park_xy[env_ids].clone(),
            "slot_swapped": self.slot_swapped[env_ids].clone(),
            "lean": self._lean[env_ids].clone(),
            "pair": self._pair[env_ids].clone(),
            "free": self._free[env_ids].clone(),
            "lean_ct": self._lean_ct[env_ids].clone(),
            "pair_ct": self._pair_ct[env_ids].clone(),
            "free_ct": self._free_ct[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.pillar.write_root_state_to_sim(state["pillar"], env_ids)
        self.build_pad.write_root_state_to_sim(state["build_pad"], env_ids)
        self.park_pad.write_root_state_to_sim(state["park_pad"], env_ids)
        for n, b in self.panels.items():
            b.write_root_state_to_sim(state["panels"][n], env_ids)
        self.build_xy[env_ids] = state["build_xy"]
        self.build_yaw[env_ids] = state["build_yaw"]
        self.park_xy[env_ids] = state["park_xy"]
        self.slot_swapped[env_ids] = state["slot_swapped"]
        self._lean[env_ids] = state["lean"]
        self._pair[env_ids] = state["pair"]
        self._free[env_ids] = state["free"]
        self._lean_ct[env_ids] = state["lean_ct"]
        self._pair_ct[env_ids] = state["pair_ct"]
        self._free_ct[env_ids] = state["free_ct"]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"On the ground lie two loose panels: a BLUE panel "
            f"({c.panel_l[0] * 1000:.0f} x {c.panel_w * 1000:.0f} x "
            f"{c.panel_t * 1000:.0f} mm) and a RED panel ({c.panel_l[1] * 1000:.0f} x "
            f"{c.panel_w * 1000:.0f} x {c.panel_t * 1000:.0f} mm), both flat, "
            f"scattered with random headings. A WHITE square pad "
            f"({c.build_pad_size * 1000:.0f} mm) is the BUILD SITE; a GREEN pillar "
            f"(column {c.col_x * 1000:.0f} x {c.col_y * 1000:.0f} x "
            f"{c.col_h * 1000:.0f} mm with a YELLOW T-handle bar on top, ~"
            f"{c.pillar_mass * 1000:.0f} g) stands at its centre. A MAGENTA square "
            f"pad ({c.park_pad_size * 1000:.0f} mm) lies elsewhere. Pad positions "
            f"and the build pad's heading vary per episode.\n"
            f"Goal: build a FREE-STANDING TENT on the white pad — the blue and red "
            f"panels leaning against each other in an inverted V, each tilted "
            f"between {c.tilt_lo_deg:.0f} and {c.tilt_hi_deg:.0f} degrees off "
            f"vertical in OPPOSITE directions, bottom edges resting on the pad "
            f"within {c.zone_r * 100:.0f} cm of its centre, top edges together "
            f"(within {c.apex_gap * 1000:.0f} mm) with the ridge at least "
            f"{c.apex_min_h * 100:.0f} cm above the pad — and then move the green "
            f"pillar AWAY (at least {c.clear_r * 100:.0f} cm from both panels) and "
            f"leave it resting on the MAGENTA pad. A panel this thin cannot stand "
            f"leaning on its own, and you have only one gripper — but a panel CAN "
            f"rest leaning against the pillar's flat faces, so the pillar can serve "
            f"as temporary falsework: lean one panel on each side of the column, "
            f"lift the pillar straight up by its yellow handle so the panels settle "
            f"onto each other, then park the pillar. The finished tent must support "
            f"itself: nothing may touch or prop it, and a tent that collapses, a "
            f"panel left flat, panels merely stacked on top of each other, or the "
            f"pillar left near the tent or off the magenta pad all fail. Success is "
            f"judged with everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lean the blue and red panels against each other on the white pad so "
            "they form a free-standing tent (each tilted 8-55 degrees, opposite "
            "ways, top edges together, ridge at least 10 cm up). Use the green "
            "pillar as temporary support, then lift it out by its yellow handle and "
            "leave it resting on the magenta pad, at least 15 cm from the panels. "
            "The tent must stand on its own; a collapsed tent, flat or stacked "
            "panels, or a pillar left near the tent fails."
        )

    # ----- live predicates ----------------------------------------------------------------------
    def _panel_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(pos_w (N,2,3), up-axis u (N,2,3) sign-corrected upward, |lin vel| (N,2))."""
        from isaaclab.utils.math import quat_apply

        pos = torch.stack([b.data.root_pos_w for b in self.panels.values()], dim=1)
        quat = torch.stack([b.data.root_quat_w for b in self.panels.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.panels.values()], dim=1)
        n = pos.shape[0]
        ez = torch.tensor([0.0, 0.0, 1.0], device=pos.device).expand(n * 2, 3)
        u = quat_apply(quat.reshape(n * 2, 4), ez).reshape(n, 2, 3)
        u = u * torch.sign(u[:, :, 2:3] + 1e-12)  # point up
        return pos, u, vel

    def panel_tilt_deg(self) -> torch.Tensor:
        """(N,2) tilt of each panel's long axis off vertical, degrees (flat ~= 90)."""
        _p, u, _v = self._panel_tensors()
        return torch.rad2deg(torch.acos(u[:, :, 2].clamp(-1.0, 1.0)))

    def _ends(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(top (N,2,3), bottom (N,2,3)): panel long-axis end midpoints, world."""
        pos, u, _v = self._panel_tensors()
        half = self._len[None, :, None] / 2
        return pos + u * half, pos - u * half

    def lean_ok(self) -> torch.Tensor:
        """(N,2) bool: panel tilt in band, base down near the pad top, base in zone."""
        c = self.cfg
        tilt = self.panel_tilt_deg()
        top, bot = self._ends()
        band = (tilt >= c.tilt_lo_deg) & (tilt <= c.tilt_hi_deg)
        base_z = bot[:, :, 2] - self.env_origins[:, None, 2]
        based = base_z < c.pad_t + c.base_z_max
        in_zone = (bot[:, :, 0:2] - self.build_xy[:, None, :]).norm(dim=-1) < c.zone_r
        return band & based & in_zone

    def opposed(self) -> torch.Tensor:
        """(N,) bool: the two panels lean in opposite horizontal directions."""
        _p, u, _v = self._panel_tensors()
        h = u[:, :, 0:2]
        h = h / h.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return (h[:, 0] * h[:, 1]).sum(dim=-1) < self.cfg.opposed_max_dot

    def apex_ok(self) -> torch.Tensor:
        """(N,) bool: top edges together and the ridge high enough above the pad."""
        c = self.cfg
        top, _bot = self._ends()
        close = (top[:, 0] - top[:, 1]).norm(dim=-1) < c.apex_gap
        apex_z = top[:, :, 2].max(dim=1).values - self.env_origins[:, 2]
        return close & (apex_z > c.pad_t + c.apex_min_h)

    def pillar_clear(self) -> torch.Tensor:
        """(N,) bool: pillar horizontally >= clear_r from BOTH panel centres — the
        tent carries itself (clear_r exceeds the pillar's max horizontal reach plus
        any in-band panel's apex offset, so a clear pillar cannot prop the tent)."""
        pos, _u, _v = self._panel_tensors()
        d = (pos[:, :, 0:2] - self.pillar.data.root_pos_w[:, None, 0:2]).norm(dim=-1)
        return (d > self.cfg.clear_r).all(dim=1)

    def tent_standing(self) -> torch.Tensor:
        """(N,) bool: the full free-standing-tent predicate (geometry + freedom)."""
        return self.lean_ok().all(dim=1) & self.opposed() & self.apex_ok() \
            & self.pillar_clear()

    def pillar_parked(self) -> torch.Tensor:
        """(N,) bool: pillar resting on/at the park pad (down, in tolerance)."""
        c = self.cfg
        pp = self.pillar.data.root_pos_w
        d = (pp[:, 0:2] - self.park_xy).norm(dim=-1)
        z = pp[:, 2] - self.env_origins[:, 2]
        return (d < c.park_tol) & (z < c.park_z_max)

    def still(self) -> torch.Tensor:
        """(N,) bool: every dynamic body below the settle speed."""
        _p, _u, vel = self._panel_tensors()
        pv = self.pillar.data.root_lin_vel_w.norm(dim=-1)
        return (vel < self.cfg.settle_speed).all(dim=1) & (pv < self.cfg.settle_speed)

    def _finite(self) -> torch.Tensor:
        pos, _u, _v = self._panel_tensors()
        return torch.isfinite(pos).all(dim=-1).all(dim=-1) \
            & torch.isfinite(self.pillar.data.root_pos_w).all(dim=-1)

    # ----- latches ------------------------------------------------------------------------------
    def _update_latches(self) -> None:
        """Debounced (counter) latching: a state must hold `latch_steps` consecutive
        post-steps while STILL before it counts — a panel sweeping through the tilt
        band mid-fall carries velocity and never latches (turning-point trap)."""
        c = self.cfg
        _p, _u, vel = self._panel_tensors()
        panel_still = vel < c.settle_speed
        lean_now = (self.lean_ok() & panel_still).any(dim=1)
        pair_now = (self.lean_ok() & panel_still).all(dim=1) & self.opposed()
        free_now = self.tent_standing() & panel_still.all(dim=1)
        for now, ct, latch in ((lean_now, self._lean_ct, self._lean),
                               (pair_now, self._pair_ct, self._pair),
                               (free_now, self._free_ct, self._free)):
            ct[:] = torch.where(now, ct + 1, torch.zeros_like(ct))
            latch |= ct >= c.latch_steps

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric -------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: free-standing tent + pillar parked, everything at rest and
        finite. All clauses are live physical outcomes — a collapsed tent or a
        propping pillar fails on the spot, so success must persist on its own."""
        self._update_latches()
        return self.tent_standing() & self.pillar_parked() & self.still() \
            & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.25*lean + 0.30*pair + 0.30*free (latched, ~0 for
        the null policy), capped at 0.85 — and exactly 1.0 iff success()."""
        c = self.cfg
        self._update_latches()
        base = (c.w_lean * self._lean.float() + c.w_pair * self._pair.float()
                + c.w_free * self._free.float()).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="falsework_tent", robot="null"))

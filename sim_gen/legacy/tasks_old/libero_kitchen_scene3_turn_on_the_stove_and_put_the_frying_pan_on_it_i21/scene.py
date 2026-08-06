"""ShortOrderScene — cook every patty on the ALREADY-HOT burner, in time, one at a time
(sim_gen task `libero_kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it_i21`,
derived from libero_90/kitchen_scene3 "turn on the stove and put the frying pan on it").

The seed's plan is: toggle the stove ON, then pick-and-place the pan ONTO it — success is
a STATIC terminal relation (thing resting on the powered stove, forever). This task
inverts that relation into a HAZARD: the burner is already on (glowing, nothing to
toggle), and "resting on the stove" is a state every object must PASS THROUGH and LEAVE.
Each raw patty must sit on the one-patty burner disc long enough to cook (>= `cook_min`
substeps of settled residence) but be taken off before it burns (`burn_steps`,
irreversible latch), then be set down on the serving plate. The burner disc is sized so
only ONE patty can accrue heat at a time (geometric single-slot: two non-overlapping
patties can never both have their centres inside the cook radius; a stacked patty sits
above the height band), so with 2-3 patties the solver must SCHEDULE items through a
shared station under residence-time windows. The seed's own terminal state — put the
object on the hot stove and stop — burns the patty and permanently caps the score: the
seed's goal state is this task's canonical failure, which the smoke tests explicitly.

Judged on PHYSICAL outcomes + latched achievements:
  - cook/burn state derives ONLY from real settled residence on the burner, accumulated
    every physics substep (`post_step`): centre within `cook_xy_tol` of the burner axis,
    resting height on the burner disc, near-zero speed. Teleporting a patty across the
    stove accrues nothing; parking it beside the burner or on the stove body (12 mm
    below the disc top) accrues nothing.
  - success(): every PRESENT patty is cooked, NOT burnt, and rests settled on the
    serving plate (current-state pose check).
  - score(): per present patty — 0.15 once seared (latched, >= `sear_steps` on the
    burner), 0.70 once cooked (latched), 1.0 while ALSO delivered (cooked & settled on
    the plate); a BURNT patty is capped at 0.05 forever. Mean over present patties;
    exactly 1.0 iff success; ~0 for doing nothing (all credit starts at the burner).
    NOTE score is deliberately NOT monotone in time — burning loses credit; the
    monotonicity contract holds along the correct plan's milestones.
  - patties recolor when their state flips (raw pink -> cooked brown -> burnt black),
    so doneness is perceivable, not hidden state.

Per-episode randomization: patty count (2 or 3, judged on the sampled subset), patty
spawn poses on the prep board, stove and plate positions INCLUDING which side of the
counter each sits on — a memorized fixed shuttle trajectory fails.

Assets are fully procedural: kinematic counter/board/plate (plain shape cfgs), a
kinematic compound stove (body box + glowing burner disc + visual lamp), dynamic patty
discs via a custom compound spawner. Heavy imports (isaaclab, pxr) are deferred so
importing this module — and registering the scene — stays app-free.
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


# ----- custom compound spawners ----------------------------------------------------------------
# Same pattern as the pen_holder exemplar: one rigid body per object, child colliders +
# visual-only decoration authored with raw pxr APIs, `isaaclab.sim.utils.clone` for the
# per-env replicate machinery. Child prim names are FIXED ("disc", "burner") so the scene
# can address per-env copies for recoloring / geometry queries.

_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_stove(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the stove at `prim_path`: a KINEMATIC root (RigidBodyAPI + kinematic flag —
    the proven re-posable-furniture recipe), a dark body box (root frame at the body
    centre), the glowing burner disc centred on its top, and a small visual-only 'ON'
    lamp on the front edge. Explicit small contact offsets (the default ~2 cm would
    float patties visibly above the 12 mm disc)."""
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
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    bx, by, bh = cfg.body_size
    body = UsdGeom.Cube.Define(stage, f"{prim_path}/body")
    body.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(body.GetPrim())
    sxf.AddScaleOp().Set(Gf.Vec3f(bx, by, bh))
    body.CreateDisplayColorAttr([Gf.Vec3f(*cfg.body_color)])
    collide(body.GetPrim())

    burner = UsdGeom.Cylinder.Define(stage, f"{prim_path}/burner")
    burner.CreateRadiusAttr(cfg.ped_r)
    burner.CreateHeightAttr(cfg.ped_h)
    burner.CreateExtentAttr([Gf.Vec3f(-cfg.ped_r, -cfg.ped_r, -cfg.ped_h / 2),
                             Gf.Vec3f(cfg.ped_r, cfg.ped_r, cfg.ped_h / 2)])
    UsdGeom.Xformable(burner.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, bh / 2 + cfg.ped_h / 2))
    burner.CreateDisplayColorAttr([Gf.Vec3f(*cfg.burner_color)])
    collide(burner.GetPrim())

    # visual-only 'ON' lamp — NO CollisionAPI (the pen-tip decoration pattern)
    lamp = UsdGeom.Cube.Define(stage, f"{prim_path}/lamp")
    lamp.CreateSizeAttr(1.0)
    lxf = UsdGeom.Xformable(lamp.GetPrim())
    lxf.AddTranslateOp().Set(Gf.Vec3d(bx / 2 - 0.015, -by / 2 + 0.02, bh / 2 + 0.006))
    lxf.AddScaleOp().Set(Gf.Vec3f(0.018, 0.018, 0.012))
    lamp.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.10, 0.05)])
    return root


def _spawn_patty(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one patty at `prim_path`: dynamic root Xform (RigidBodyAPI + explicit
    MassAPI + depenetration cap + damping so a 50 g disc crosses the settle gate
    promptly), one cylinder collider child named 'disc' carrying the displayColor the
    scene rewrites when the patty's cook state flips."""
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
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.10)
    pxrb.CreateAngularDampingAttr(0.20)

    disc = UsdGeom.Cylinder.Define(stage, f"{prim_path}/disc")
    disc.CreateRadiusAttr(cfg.patty_r)
    disc.CreateHeightAttr(cfg.patty_h)
    disc.CreateExtentAttr([Gf.Vec3f(-cfg.patty_r, -cfg.patty_r, -cfg.patty_h / 2),
                           Gf.Vec3f(cfg.patty_r, cfg.patty_r, cfg.patty_h / 2)])
    disc.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    UsdPhysics.CollisionAPI.Apply(disc.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(disc.GetPrim())
    px.CreateContactOffsetAttr(float(cfg.contact_offset))
    px.CreateRestOffsetAttr(0.0)
    return root


def _stove_spawner_cfg(*, body_size: tuple, ped_r: float, ped_h: float, body_color: tuple,
                       burner_color: tuple, contact_offset: float) -> Any:
    """Build (lazily, app required) the stove spawner cfg."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "stove" not in _SPAWNER_CACHE:

        @configclass
        class StoveSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stove)
            body_size: tuple = (0.20, 0.20, 0.05)
            ped_r: float = 0.045
            ped_h: float = 0.012
            body_color: tuple = (0.12, 0.12, 0.14)
            burner_color: tuple = (0.90, 0.25, 0.08)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["stove"] = StoveSpawnerCfg

    return _SPAWNER_CACHE["stove"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        body_size=body_size, ped_r=ped_r, ped_h=ped_h, body_color=body_color,
        burner_color=burner_color, contact_offset=contact_offset,
    )


def _patty_spawner_cfg(*, patty_r: float, patty_h: float, mass: float, color: tuple,
                       contact_offset: float) -> Any:
    """Build (lazily, app required) the patty spawner cfg."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "patty" not in _SPAWNER_CACHE:

        @configclass
        class PattySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_patty)
            patty_r: float = 0.030
            patty_h: float = 0.016
            color: tuple = (0.88, 0.55, 0.50)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["patty"] = PattySpawnerCfg

    return _SPAWNER_CACHE["patty"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        patty_r=patty_r, patty_h=patty_h, color=color, contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ShortOrderSceneCfg(BaseCfg):
    """Config for `ShortOrderScene`. Counter frame: counter centred at the env origin,
    counter top at z = `counter_t`; prep board on -x, stove and plate on +x, on opposite
    y sides (which side is which is sampled per episode)."""

    # --- tunable: cook window (physics substeps @ 120 Hz) ------------------------------------
    sear_steps: int = tunable(30)     # residence to latch 'seared' (first partial credit)
    cook_min_steps: int = tunable(180)  # residence to latch 'cooked' (1.5 s)
    burn_steps: int = tunable(480)    # residence to latch 'burnt' — irreversible (4.0 s)

    # --- tunable: rubric thresholds ----------------------------------------------------------
    cook_xy_tol: float = tunable(0.028)  # centre distance from burner axis that accrues heat;
    # < patty diameter (0.060), so two non-overlapping patties can never both accrue.
    cook_z_tol: float = tunable(0.008)   # height band (about resting-on-burner) that accrues
    cook_speed_max: float = tunable(0.08)  # |lin vel| gate for accrual (resting, not passing)
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging delivery (m/s)
    plate_margin: float = tunable(0.012)  # delivered: centre within plate_r - margin of axis

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    stove_x_range: tuple = tunable((0.10, 0.24))  # stove centre x band on the counter
    stove_y_range: tuple = tunable((0.10, 0.20))  # |y| band; the SIDE is sampled per episode
    plate_x_range: tuple = tunable((0.10, 0.24))  # plate centre x band (opposite y side)
    plate_y_range: tuple = tunable((0.10, 0.20))
    patty_jitter: float = tunable(0.020)  # uniform +/- xy jitter on the prep-board slots
    # (slots are 110 mm apart and patties 60 mm wide, so +/-20 mm can never overlap them)
    subset_sample: bool = tunable(True)   # per-episode patty-count sampling (2..3)
    min_present: int = tunable(2)

    # --- info: structure ---------------------------------------------------------------------
    counter_size: tuple = info((1.10, 0.80))  # kinematic counter slab (x, y)
    counter_t: float = info(0.04)             # slab thickness; counter top at this z
    stove_body: tuple = info((0.20, 0.20, 0.05))  # stove body box (x, y, h)
    ped_r: float = info(0.045)   # burner disc radius — the one-patty slot
    ped_h: float = info(0.012)   # burner disc height above the stove body
    plate_r: float = info(0.080)  # serving plate radius (3 patties fit side by side)
    plate_h: float = info(0.010)
    board_size: tuple = info((0.18, 0.38, 0.006))  # prep board (patty spawn zone)
    board_x: float = info(-0.30)
    patty_r: float = info(0.030)
    patty_h: float = info(0.016)
    patty_mass: float = info(0.05)
    n_patties: int = info(3)
    raw_color: tuple = info((0.88, 0.55, 0.50))     # raw patty pink
    cooked_color: tuple = info((0.45, 0.26, 0.10))  # cooked brown
    burnt_color: tuple = info((0.07, 0.06, 0.05))   # burnt near-black
    stove_body_color: tuple = info((0.12, 0.12, 0.14))
    burner_color: tuple = info((0.90, 0.25, 0.08))  # glowing = ON, always
    plate_color: tuple = info((0.92, 0.92, 0.90))
    board_color: tuple = info((0.55, 0.40, 0.22))
    contact_offset: float = info(0.002)
    parking_pos: tuple = info((0.85, 0.60))  # off-counter ground depot for absent patties

    # Derived (filled in __post_init__).
    counter_top: float = field(default=None, init=False)
    burner_top: float = field(default=None, init=False)  # top of the glowing disc (z)
    plate_top: float = field(default=None, init=False)
    board_top: float = field(default=None, init=False)
    patty_names: tuple = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.counter_top = round(self.counter_t, 4)
        self.burner_top = round(self.counter_top + self.stove_body[2] + self.ped_h, 4)
        self.plate_top = round(self.counter_top + self.plate_h, 4)
        self.board_top = round(self.counter_top + self.board_size[2], 4)
        self.patty_names = tuple(f"patty_{i}" for i in range(self.n_patties))


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("short_order")
class ShortOrderScene(BaseScene):
    cfg: ShortOrderSceneCfg

    def __init__(self, cfg: ShortOrderSceneCfg | None = None) -> None:
        super().__init__(cfg or ShortOrderSceneCfg())

    # ----- assets ---------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, kinematic counter + prep board, the kinematic compound stove,
        the kinematic serving plate, and the dynamic patties (reset() re-places the
        stove, plate and patties)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)

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
            "counter": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Counter",
                spawn=sim_utils.CuboidCfg(
                    size=(c.counter_size[0], c.counter_size[1], c.counter_t),
                    rigid_props=kin, collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.38, 0.38, 0.42)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.counter_t / 2)),
            ),
            "board": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Board",
                spawn=sim_utils.CuboidCfg(
                    size=c.board_size,
                    rigid_props=kin, collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.board_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.board_x, 0.0, c.counter_top + c.board_size[2] / 2)),
            ),
            "stove": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stove",
                spawn=_stove_spawner_cfg(
                    body_size=c.stove_body, ped_r=c.ped_r, ped_h=c.ped_h,
                    body_color=c.stove_body_color, burner_color=c.burner_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.17, 0.15, c.counter_top + c.stove_body[2] / 2)),
            ),
            "plate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plate",
                spawn=sim_utils.CylinderCfg(
                    radius=c.plate_r, height=c.plate_h,
                    rigid_props=kin, collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.plate_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.17, -0.15, c.counter_top + c.plate_h / 2)),
            ),
        }
        for i, name in enumerate(c.patty_names):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Patty_" + str(i),
                spawn=_patty_spawner_cfg(
                    patty_r=c.patty_r, patty_h=c.patty_h, mass=c.patty_mass,
                    color=c.raw_color, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.board_x, -0.11 + 0.11 * i,
                         c.board_top + c.patty_h / 2 + 0.002)),
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
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles + allocate the per-episode buffers the rubric depends on."""
        super().bind(env)
        c = self.cfg
        n, dev = env.num_envs, env.device
        self.stove: RigidObject = env.iscene["stove"]
        self.plate: RigidObject = env.iscene["plate"]
        self.patties: dict[str, RigidObject] = {nm: env.iscene[nm] for nm in c.patty_names}
        self.env_origins = env.iscene.env_origins
        p = len(c.patty_names)
        self.present = torch.ones(n, p, dtype=torch.bool, device=dev)
        self.stove_xy = torch.zeros(n, 2, device=dev)   # burner axis, env-local
        self.plate_xy = torch.zeros(n, 2, device=dev)   # plate axis, env-local
        self.cook_steps = torch.zeros(n, p, dtype=torch.long, device=dev)
        self.seared = torch.zeros(n, p, dtype=torch.bool, device=dev)
        self.cooked = torch.zeros(n, p, dtype=torch.bool, device=dev)
        self.burnt = torch.zeros(n, p, dtype=torch.bool, device=dev)

    def _patty_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(env-local pos (N,P,3), |lin vel| (N,P)) for all patties, manifest order."""
        pos = torch.stack([b.data.root_pos_w for b in self.patties.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.patties.values()], dim=1)
        return pos - self.env_origins[:, None, :], vel

    def _recolor(self, e: int, i: int, rgb: tuple) -> None:
        """Rewrite the displayColor of patty i's disc in env e (the microwave-lamp
        refresh pattern: per-env cloned child prims are addressable by name)."""
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(f"/World/envs/env_{e}/Patty_{i}/disc")
        UsdGeom.Cylinder(prim).CreateDisplayColorAttr([Gf.Vec3f(*rgb)])

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the present patty subset, the stove/plate positions
        (including which y side each sits on), scatter present patties on the prep
        board, park absent ones off-counter, clear all cook state, repaint raw."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        p = len(c.patty_names)

        def u(lo: float, hi: float, shape=(1,)) -> torch.Tensor:
            return lo + (hi - lo) * torch.rand(m, *shape, device=dev).squeeze(-1)

        # --- present subset: k ~ U{min_present..n} ---
        if c.subset_sample:
            k = torch.randint(c.min_present, p + 1, (m,), device=dev)
        else:
            k = torch.full((m,), p, dtype=torch.long, device=dev)
        rank = torch.rand(m, p, device=dev).argsort(dim=1).argsort(dim=1)
        self.present[env_ids] = rank < k.unsqueeze(1)

        # --- stove and plate: sampled x, sampled |y|, opposite sides, side sign sampled ---
        side = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        sxy = torch.stack([u(*c.stove_x_range), side * u(*c.stove_y_range)], dim=1)
        pxy = torch.stack([u(*c.plate_x_range), -side * u(*c.plate_y_range)], dim=1)
        self.stove_xy[env_ids] = sxy
        self.plate_xy[env_ids] = pxy

        st = torch.zeros(m, 7, device=dev)
        st[:, 0:2] = sxy
        st[:, 2] = c.counter_top + c.stove_body[2] / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.stove.write_root_pose_to_sim(st, env_ids)

        st = torch.zeros(m, 7, device=dev)
        st[:, 0:2] = pxy
        st[:, 2] = c.counter_top + c.plate_h / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.plate.write_root_pose_to_sim(st, env_ids)

        # --- patties: board slots + jitter, free yaw; absent -> off-counter ground depot ---
        for i, name in enumerate(c.patty_names):
            slot = torch.zeros(m, 3, device=dev)
            slot[:, 0] = c.board_x
            slot[:, 1] = -0.11 + 0.11 * i
            slot[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.patty_jitter
            slot[:, 2] = c.board_top + c.patty_h / 2 + 0.002
            park = torch.zeros(m, 3, device=dev)
            park[:, 0] = c.parking_pos[0] + 0.10 * i
            park[:, 1] = c.parking_pos[1]
            park[:, 2] = c.patty_h / 2 + 0.002
            pres = self.present[env_ids, i].unsqueeze(1)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.where(pres, slot, park)
            half = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            self.patties[name].write_root_state_to_sim(st, env_ids)

        # --- clear cook state + repaint raw ---
        self.cook_steps[env_ids] = 0
        self.seared[env_ids] = False
        self.cooked[env_ids] = False
        self.burnt[env_ids] = False
        for e in env_ids.tolist():
            for i in range(p):
                self._recolor(e, i, c.raw_color)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Heat accrual, every physics substep: a patty accrues one step of heat while
        its centre is within `cook_xy_tol` of the burner axis, at resting height on the
        burner disc, and near-still. Latches seared/cooked/burnt (irreversible) and
        repaints flipped patties. A teleport THROUGH the burner region accrues at most
        one step; residence is physical."""
        c = self.cfg
        pos, vel = self._patty_tensors()
        rest_z = c.burner_top + c.patty_h / 2
        xy_ok = (pos[:, :, :2] - self.stove_xy[:, None, :]).norm(dim=-1) < c.cook_xy_tol
        z_ok = (pos[:, :, 2] - rest_z).abs() < c.cook_z_tol
        on = xy_ok & z_ok & (vel < c.cook_speed_max) & self.present
        self.cook_steps += on.long()
        new_ck = (self.cook_steps >= c.cook_min_steps) & ~self.cooked
        new_bt = (self.cook_steps >= c.burn_steps) & ~self.burnt
        self.seared |= self.cook_steps >= c.sear_steps
        self.cooked |= new_ck
        self.burnt |= new_bt
        if bool(new_ck.any()) or bool(new_bt.any()):
            for e, i in torch.nonzero(new_bt).tolist():
                self._recolor(e, i, c.burnt_color)
            for e, i in torch.nonzero(new_ck & ~self.burnt).tolist():
                self._recolor(e, i, c.cooked_color)

    # ----- state (full, restorable) ----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "stove": self.stove.data.root_state_w[env_ids].clone(),
            "plate": self.plate.data.root_state_w[env_ids].clone(),
            "patties": {nm: b.data.root_state_w[env_ids].clone()
                        for nm, b in self.patties.items()},
            "present": self.present[env_ids].clone(),
            "stove_xy": self.stove_xy[env_ids].clone(),
            "plate_xy": self.plate_xy[env_ids].clone(),
            "cook_steps": self.cook_steps[env_ids].clone(),
            "seared": self.seared[env_ids].clone(),
            "cooked": self.cooked[env_ids].clone(),
            "burnt": self.burnt[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.stove.write_root_pose_to_sim(state["stove"][:, 0:7], env_ids)
        self.plate.write_root_pose_to_sim(state["plate"][:, 0:7], env_ids)
        for nm, b in self.patties.items():
            b.write_root_state_to_sim(state["patties"][nm], env_ids)
        self.present[env_ids] = state["present"]
        self.stove_xy[env_ids] = state["stove_xy"]
        self.plate_xy[env_ids] = state["plate_xy"]
        self.cook_steps[env_ids] = state["cook_steps"]
        self.seared[env_ids] = state["seared"]
        self.cooked[env_ids] = state["cooked"]
        self.burnt[env_ids] = state["burnt"]

    # ----- description -----------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        dt = 1.0 / 120.0
        return (
            f"A kitchen counter. On one side sits a small stove whose single burner disc "
            f"(radius {c.ped_r * 1000:.0f} mm) is ALREADY ON — it glows orange and its red "
            f"lamp is lit; there is nothing to switch. On the opposite side sits a white "
            f"serving plate; both positions (and which side each is on) change every "
            f"episode. On the wooden prep board lie 2-3 raw pink patties "
            f"({2 * c.patty_r * 1000:.0f} mm discs): count what you see.\n"
            f"Goal: cook EVERY patty and serve it. A patty cooks only while it rests on "
            f"the glowing disc: after about {c.cook_min_steps * dt:.1f} s it turns brown "
            f"(cooked); if it stays past {c.burn_steps * dt:.1f} s total it turns black — "
            f"BURNT, which is permanent and ruins the dish. The disc holds one patty at a "
            f"time (a second patty beside or on top does not cook). Move each cooked patty "
            f"off in time and set it down on the serving plate; finish with all patties "
            f"cooked (none burnt) resting on the plate. Leaving a patty parked on the hot "
            f"stove is the one sure way to fail."
        )

    # ----- progress / rubric ------------------------------------------------------------------
    def delivered(self) -> torch.Tensor:
        """(N, P) bool, current-state: cooked, NOT burnt, resting settled on the serving
        plate (centre within `plate_r - plate_margin` of the plate axis, at plate-stack
        height — up to three patties stacked count — and slower than `settle_speed`)."""
        c = self.cfg
        pos, vel = self._patty_tensors()
        xy_ok = (pos[:, :, :2] - self.plate_xy[:, None, :]).norm(dim=-1) \
            < c.plate_r - c.plate_margin
        z_lo = c.plate_top + c.patty_h / 2 - 0.006
        z_hi = c.plate_top + 2.5 * c.patty_h + 0.006
        z_ok = (pos[:, :, 2] > z_lo) & (pos[:, :, 2] < z_hi)
        still = vel < c.settle_speed
        return self.cooked & ~self.burnt & xy_ok & z_ok & still

    def success(self) -> torch.Tensor:
        """(N,) bool: every PRESENT patty is cooked (never burnt) and rests settled on
        the serving plate."""
        return (self.delivered() | ~self.present).all(dim=1) & self.present.any(dim=1)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: per present patty — 0.15 seared (latched), 0.70 cooked
        (latched), 1.0 delivered (current-state); a burnt patty is capped at 0.05
        forever. Mean over present patties, capped at 0.95 unless success (1.0 iff
        success). Doing nothing scores 0 (all credit starts at the burner)."""
        per = torch.where(
            self.burnt,
            torch.full_like(self.seared, 0.05, dtype=torch.float32),
            0.15 * self.seared.float() + 0.55 * self.cooked.float()
            + 0.30 * self.delivered().float(),
        )
        per = per * self.present.float()
        raw = per.sum(dim=1) / self.present.float().sum(dim=1).clamp(min=1.0)
        return torch.where(self.success(), torch.ones_like(raw), raw.clamp(max=0.95))


# ----- env registration ------------------------------------------------------------------------
register_env("simgen", lambda: EnvCfg(scene="short_order", robot="null", env_spacing=4.0))

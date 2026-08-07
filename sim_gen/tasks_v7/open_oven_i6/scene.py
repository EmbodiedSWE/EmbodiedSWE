"""DiceTumbleScene — tumble each oversized die onto a colored floor pad, matching face up
(sim_gen task `open_oven_i6`).

Derived from rlbench/open_oven, but STRATEGICALLY different: the seed is one prehensile
articulation act — grasp the oven door handle and rotate the panel about its BUILT
hinge until open; one grasp, one guided rotation on a fixed axis, judged by a door
joint. Here there is no hinge, no handle, no appliance and no grasp at all: the
rotation the task demands must be MANUFACTURED by the solver, one quarter-turn at a
time, by tipping a free cube over its own ground edge — the transient "hinge" is a
contact line that exists only while the cube pivots on it. Each die is a 100 mm cube
(wider than any parallel jaw — nonprehensile by construction) with six distinctly
colored faces. Two thin square pads lie on the floor, each painted one color of the
same palette; per episode the two pad colors are SAMPLED and each die spawns with an
up-face that matches NEITHER pad (resampled orientation + free yaw). A solver needs a
different PLAN each episode (read both pad colors and both dice orientations; for each
die compute a tumble sequence — one tip if the needed face is on a side, two same-way
tips if it is underneath — pick tip directions that respect the cube's own face grid;
then push LOW to slide without tipping and park the die centered on a pad) and a
different code structure (an SO(3) face-up predicate + per-pad latched credit — not a
door-joint readout). The mechanics are height-keyed: a push above ~h/2 divided by the
friction coefficient tips the die over the far bottom edge, a push below it slides the
die flat — the same contact, two different verbs, and the task needs both.

The seed's end state (a door swung open) is not expressible here — documented N/A in
TASK.md; the nearest naive plan, "transport the die to the pad without reorienting it"
(the seed family's put-tray-in-oven move), is constructed in smoke and scores ~0.

success(): every ACTIVE pad carries a settled die whose face of that pad's color
points straight up — die center inside the pad's inner tolerance window (pad body
frame), resting AT pad height (a die stacked on another die reads ~100 mm high and is
rejected), face-up within `face_max_deg`, velocities below the settle gates. The two
pads are farther apart than twice the window, so one die can never serve both
(asserted in __post_init__); both pads served implies two dice placed.

score() is graded and latched (credit never evaporates), per active pad: 0.10 * some
die EVER showed this pad's color face-up (the reorientation is done) + 0.10 * ever
face-up within `near_r` of this pad (transported while reoriented) + 0.25 * ever
face-up AND in the pad window at pad height (placed); sum over the two pads = 0.90;
1.0 iff success(). The null policy scores ~0 (spawn up-faces exclude both pad colors).

Assets are fully procedural, one rigid body each (compound spawners; child colliders
of one body never self-collide):
  - die (x2, dynamic, 350 g): a hollow 100 mm cube shell of six 8 mm plates, each
    plate carrying one palette color — the color IS the face. High-friction physics
    material (0.9/0.8) bound to every plate: friction is load-bearing (it is what
    makes a high push TIP instead of slide; the tip-before-slide inequality is
    asserted in __post_init__).
  - pad (x6, KINEMATIC, one per palette color): a thin 220 x 220 x 4 mm plate with
    NO collider — a painted floor marking, recessed so only a fraction of a
    millimeter stands proud (for rendering). The die slides on the ground plane
    straight across it: an earlier colliding-lip design stalled the slide — even a
    0.2 mm step plus the contact offsets forms a wall whose speculative contact
    kills the incoming momentum, and a quasi-static climb needs more push moment
    than the no-tip budget allows. The kinematic body's pose still anchors the
    rubric's pad-frame window. Per episode the two SAMPLED colors' pads are placed at the two arena
    slots (jitter + free yaw); the other four park in a ground depot outside the
    arena.
Contact offsets are explicit (1.5 mm) so the pad-height window and the flat-resting
face-up geometry stay real.

Per-episode randomization (readback-verified in smoke): the two pad colors (15 color
pairs), pad slot jitter + free yaw, per-die spawn xy jitter + free yaw + up-face
sampled among the four non-target colors (the tumble count and directions change).
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


# ----- custom compound spawners ---------------------------------------------------------------
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
    """Author one USD physics material (friction is load-bearing here: it decides
    whether a push tips or slides the die)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, size, center, color, contact_offset: float, material=None,
         collide: bool = True) -> None:
    """Author one box child prim (translate -> scale, authored once — idempotent per
    prim, the duplicate-xformOp trap). `collide=False` authors a purely visual box."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if not collide:
        return
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_die(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one dynamic die at `prim_path`: a hollow cube shell of six colored
    plates (one rigid body; the color IS the face). Origin = cube center."""
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    from pxr import UsdGeom

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.15)
    pxrb.CreateSolverPositionIterationCountAttr(8)
    pxrb.CreateSolverVelocityIterationCountAttr(1)
    # ZERO sleep/stabilization thresholds: a sleeping body silently ignores applied
    # external wrenches (the solve/smoke force probes depend on this).
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    s, t, co = cfg.size, cfg.plate_t, cfg.contact_offset
    inner = s - 2 * t
    # face order matches the cfg palette/axis convention: +x, -x, +y, -y, +z, -z
    plates = [
        ((t, s, inner), (+(s - t) / 2, 0.0, 0.0)),
        ((t, s, inner), (-(s - t) / 2, 0.0, 0.0)),
        ((inner, t, inner), (0.0, +(s - t) / 2, 0.0)),
        ((inner, t, inner), (0.0, -(s - t) / 2, 0.0)),
        ((s, s, t), (0.0, 0.0, +(s - t) / 2)),
        ((s, s, t), (0.0, 0.0, -(s - t) / 2)),
    ]
    for i, (size, center) in enumerate(plates):
        _box(stage, f"{prim_path}/face_{i}", size, center, cfg.colors[i], co, material=mat)
    return root


def _spawn_pad(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one KINEMATIC pad at `prim_path`: a thin colored square plate with NO
    collider — a painted floor marking. The die slides on the ground plane straight
    across it (no lip, no seam to catch on); the kinematic body's pose still anchors
    the rubric's pad-frame window. Origin = plate center."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(2.0)
    _box(stage, f"{prim_path}/plate", (cfg.size, cfg.size, cfg.thickness),
         (0.0, 0.0, 0.0), cfg.color, cfg.contact_offset, collide=False)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (explicit @configclass subclasses
    of RigidObjectSpawnerCfg, defined lazily so the module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "die" not in _SPAWNER_CACHE:

        @configclass
        class DieSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_die)
            size: float = 0.100
            plate_t: float = 0.008
            colors: tuple = ()
            mu_static: float = 0.9
            mu_dynamic: float = 0.8
            contact_offset: float = 0.0015

        @configclass
        class PadSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pad)
            size: float = 0.22
            thickness: float = 0.004
            color: tuple = (0.5, 0.5, 0.5)
            contact_offset: float = 0.0015  # unused (no collider) — kept for _box's signature

        _SPAWNER_CACHE.update(die=DieSpawnerCfg, pad=PadSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class DiceTumbleSceneCfg(BaseCfg):
    """Config for `DiceTumbleScene`. The strategic honesty knobs are asserted in
    `__post_init__`: the die is too wide for a parallel jaw (nonprehensile by
    construction), friction makes a high push tip before it slides (the tumble
    mechanism is physically real), and the two pads are far enough apart that one die
    can never serve both."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    face_max_deg: float = tunable(15.0)  # target face normal within this of world-up
    pad_xy_tol: float = tunable(0.055)  # |die center - pad center| per axis, pad frame (m);
    # with die half 0.050 and pad half 0.110 this window keeps the die FULLY on the pad
    pad_dz_lo: float = tunable(-0.012)  # die center height minus (pad_t + die/2), window
    pad_dz_hi: float = tunable(0.020)  # ... a die stacked on a die reads ~ +0.10 -> rejected
    settle_lin: float = tunable(0.08)  # max |lin vel| when judging success (m/s)
    settle_ang: float = tunable(1.0)  # max |ang vel| when judging success (rad/s)
    near_r: float = tunable(0.18)  # face-up die within this of the pad center -> near latch

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    pad_slots: tuple = tunable(((0.30, 0.17), (0.30, -0.17)))  # the two arena pad slots (xy)
    pad_jitter: float = tunable(0.03)  # uniform +/- xy jitter per placed pad at reset (m)
    die_slots: tuple = tunable(((0.02, 0.12), (0.02, -0.12)))  # the two die spawn slots (xy)
    die_jitter: float = tunable(0.04)  # uniform +/- xy jitter per die at reset (m)

    # --- info: structure ---------------------------------------------------------------------
    die_size: float = info(0.100)  # cube edge (100 mm > the ~80 mm Franka jaw: no grasp)
    die_plate_t: float = info(0.008)
    die_mass: float = info(0.35)
    pad_size: float = info(0.22)
    pad_t: float = info(0.004)
    pad_lip: float = info(0.0002)  # pad top proud of the floor — purely VISUAL (the
    # pad has no collider; the die slides on the ground plane straight across it)
    mu_static: float = info(0.9)  # die/ground materials (combine mode: average)
    mu_dynamic: float = info(0.8)
    contact_offset: float = info(0.0015)
    parking: tuple = info((1.25, -0.55))  # ground depot for the 4 absent pads (row, dy 0.28)
    # (name, display color) per face, index = body axis (+x,-x,+y,-y,+z,-z). Both dice
    # carry the SAME layout; the pads reuse these colors.
    palette: tuple = info((
        ("red", (0.85, 0.10, 0.10)),
        ("green", (0.10, 0.65, 0.15)),
        ("blue", (0.10, 0.25, 0.85)),
        ("yellow", (0.92, 0.85, 0.10)),
        ("purple", (0.55, 0.15, 0.75)),
        ("white", (0.92, 0.92, 0.92)),
    ))

    # Derived (filled in __post_init__).
    tip_h: float = field(default=None, init=False)  # push height above which a push tips

    def __post_init__(self) -> None:
        s, mu = self.die_size, self.mu_dynamic
        # -- the mechanism must be real (geometry/physics asserts) --
        assert s > 0.085, "die must be wider than a parallel jaw (nonprehensile by construction)"
        self.tip_h = (s / 2) / mu  # push above this height -> tips; below -> slides
        assert self.tip_h < 0.95 * s, \
            "friction too low: no push height on the die face can tip it before it slides"
        assert self.tip_h > 0.25 * s, \
            "friction too high: any push tips — the slide verb would not exist"
        assert self.pad_xy_tol + s / 2 <= self.pad_size / 2 + 1e-9, \
            "the pad window must keep the die fully on the pad"
        d = math.dist(self.pad_slots[0], self.pad_slots[1])
        assert d - 2 * self.pad_jitter > 2 * self.pad_xy_tol + 0.05, \
            "pads must be far enough apart that one die can never serve both"
        assert self.pad_dz_hi < s / 2, "the height window must reject a die stacked on a die"
        assert len(self.palette) == 6


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("dice_tumble")
class DiceTumbleScene(BaseScene):
    cfg: DiceTumbleSceneCfg

    def __init__(self, cfg: DiceTumbleSceneCfg | None = None) -> None:
        super().__init__(cfg or DiceTumbleSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        die_cls, pad_cls = spawners["die"], spawners["pad"]

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_static, dynamic_friction=c.mu_dynamic,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
        }
        for i, nm in enumerate(("die_a", "die_b")):
            out[nm] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Die_" + nm[-1],
                spawn=die_cls(
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.die_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    size=c.die_size, plate_t=c.die_plate_t,
                    colors=tuple(col for _n, col in c.palette),
                    mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.die_slots[i][0], c.die_slots[i][1], c.die_size / 2 + 0.003)),
            )
        for k, (nm, col) in enumerate(c.palette):
            out[f"pad_{nm}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad_" + nm,
                spawn=pad_cls(
                    mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    size=c.pad_size, thickness=c.pad_t, color=col,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.parking[0], c.parking[1] + 0.28 * (k % 3),
                         c.pad_lip - c.pad_t / 2 + 0.28 * (k // 3))),
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
        self.die_names = ("die_a", "die_b")
        self.dice: dict[str, RigidObject] = {nm: env.iscene[nm] for nm in self.die_names}
        self.pads: dict[str, RigidObject] = {
            nm: env.iscene[f"pad_{nm}"] for nm, _c in c.palette}
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # pad_color[e, p]: palette index of the color required at arena slot p.
        self.pad_color = torch.zeros(n, 2, dtype=torch.long, device=dev)
        self.show_latch = torch.zeros(n, 2, device=dev)  # some die ever face-up in this color
        self.near_latch = torch.zeros(n, 2, device=dev)  # ... within near_r of the pad
        self.placed_latch = torch.zeros(n, 2, device=dev)  # ... in the pad window at pad height

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the two pad colors (distinct), place those pads at the
        jittered/yawed arena slots and park the rest in the depot; drop each die at its
        jittered slot with a free yaw and an up-face sampled among the four NON-target
        colors (so the null policy shows neither pad color and every serve needs at
        least one tumble); zero the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        # --- pad colors: 2 distinct palette indices per env ---
        colors = torch.rand(m, 6, device=dev).argsort(dim=1)[:, :2]
        self.pad_color[env_ids] = colors

        # --- pads: sampled colors to the arena slots (jitter + free yaw); rest parked ---
        slots = torch.tensor(c.pad_slots, device=dev)  # (2, 2)
        for k, (nm, _col) in enumerate(c.palette):
            park = torch.tensor([c.parking[0], c.parking[1] + 0.28 * (k % 3),
                                 c.pad_lip - c.pad_t / 2 + 0.28 * (k // 3)],
                                device=dev).expand(m, 3)
            pos = park.clone()
            yaw = torch.zeros(m, device=dev)
            for p in range(2):
                sel = colors[:, p] == k
                if sel.any():
                    xy = slots[p].expand(m, 2) \
                        + (torch.rand(m, 2, device=dev) * 2 - 1) * c.pad_jitter
                    pos[sel, 0:2] = xy[sel]
                    pos[sel, 2] = c.pad_lip - c.pad_t / 2
                    yaw[sel] = (torch.rand(int(sel.sum()), device=dev) * 2 - 1) * math.pi
            half = yaw / 2
            quat = torch.stack([torch.cos(half), torch.zeros(m, device=dev),
                                torch.zeros(m, device=dev), torch.sin(half)], dim=-1)
            write(self.pads[nm], pos, quat)

        # --- dice: jittered slots, free yaw, up-face sampled among non-target colors ---
        from isaaclab.utils.math import quat_mul

        # base orientation putting face f up (f indexes +x,-x,+y,-y,+z,-z), (6, 4) wxyz
        r = math.sqrt(0.5)
        q_face = torch.tensor([
            [r, 0.0, -r, 0.0],  # +x up: rot_y(-90)
            [r, 0.0, r, 0.0],   # -x up: rot_y(+90)
            [r, r, 0.0, 0.0],   # +y up: rot_x(+90)
            [r, -r, 0.0, 0.0],  # -y up: rot_x(-90)
            [1.0, 0.0, 0.0, 0.0],  # +z up
            [0.0, 1.0, 0.0, 0.0],  # -z up: rot_x(180)
        ], device=dev)
        allowed = torch.ones(m, 6, device=dev)
        allowed.scatter_(1, colors, 0.0)  # exclude both pad colors from spawn up-faces
        for i, nm in enumerate(self.die_names):
            face = torch.multinomial(allowed, 1).squeeze(1)  # (m,)
            yaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            half = yaw / 2
            q_yaw = torch.stack([torch.cos(half), torch.zeros(m, device=dev),
                                 torch.zeros(m, device=dev), torch.sin(half)], dim=-1)
            quat = quat_mul(q_yaw, q_face[face])
            xy = torch.tensor(c.die_slots[i], device=dev).expand(m, 2) \
                + (torch.rand(m, 2, device=dev) * 2 - 1) * c.die_jitter
            pos = torch.cat([xy, torch.full((m, 1), c.die_size / 2 + 0.003, device=dev)],
                            dim=-1)
            write(self.dice[nm], pos, quat)

        self.show_latch[env_ids] = 0.0
        self.near_latch[env_ids] = 0.0
        self.placed_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "dice": {nm: b.data.root_state_w[env_ids].clone() for nm, b in self.dice.items()},
            "pads": {nm: b.data.root_state_w[env_ids].clone() for nm, b in self.pads.items()},
            "pad_color": self.pad_color[env_ids].clone(),
            "show_latch": self.show_latch[env_ids].clone(),
            "near_latch": self.near_latch[env_ids].clone(),
            "placed_latch": self.placed_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm, b in self.dice.items():
            b.write_root_state_to_sim(state["dice"][nm], env_ids)
        for nm, b in self.pads.items():
            b.write_root_state_to_sim(state["pads"][nm], env_ids)
        self.pad_color[env_ids] = state["pad_color"]
        self.show_latch[env_ids] = state["show_latch"]
        self.near_latch[env_ids] = state["near_latch"]
        self.placed_latch[env_ids] = state["placed_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        names = ", ".join(nm.upper() for nm, _c in c.palette)
        return (
            f"Two large cubes rest on the floor, each {c.die_size * 1000:.0f} mm on a "
            f"side — wider than a parallel-jaw gripper opens, so they cannot be picked "
            f"up; push them to slide, or push them high up on a face to tip them over "
            f"an edge. Every face of a cube is painted a different color of the same "
            f"six-color set ({names}); both cubes carry the same six colors. Also on "
            f"the floor lie two thin square pads ({c.pad_size * 1000:.0f} mm, "
            f"{c.pad_t * 1000:.0f} mm thick), each painted ONE color of that set; the "
            f"two pad colors change every episode, as do the pads' exact positions and "
            f"the cubes' starting positions and orientations — read all of it by "
            f"looking. The pads are flat markings lying flush with the floor, so a cube "
            f"can be slid straight onto one.\n"
            f"Goal: for each pad, leave one cube resting fully ON the pad, centered on "
            f"it (within about {c.pad_xy_tol * 100:.0f} cm of the pad center), with the "
            f"cube face matching THAT pad's color pointing straight up (within "
            f"{c.face_max_deg:.0f} degrees). One cube per pad, both pads served at the "
            f"same time, everything at rest. Neither cube starts with either pad color "
            f"facing up, so each cube must be tipped over its edges — a quarter-turn "
            f"per tip — until the right face is up (a face on the side needs one tip "
            f"away from the side it faces; a face underneath needs two tips the same "
            f"way), then slid onto its pad without tipping further. Either cube may "
            f"serve either pad; no ordering is required. A cube on a pad with the "
            f"wrong face up, off the pad center, stacked on the other cube, or still "
            f"moving counts for nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Tip and slide the two oversized cubes onto the two colored floor pads: "
            "one cube per pad, resting centered on the pad with the cube face that "
            "matches the pad's color turned straight up, both pads at once. A wrong "
            "face up or a cube off its pad center counts for nothing."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _face_dots(self, nm: str) -> torch.Tensor:
        """(N, 6): world-up component of each face normal of die `nm`, palette order
        (+x,-x,+y,-y,+z,-z)."""
        from isaaclab.utils.math import quat_apply

        q = self.dice[nm].data.root_quat_w
        n = q.shape[0]
        eye = torch.eye(3, device=q.device).unsqueeze(0).expand(n, 3, 3).reshape(n * 3, 3)
        qq = q.unsqueeze(1).expand(n, 3, 4).reshape(n * 3, 4)
        axes_z = quat_apply(qq, eye).reshape(n, 3, 3)[:, :, 2]  # (N,3): z of body x,y,z
        return torch.stack([axes_z[:, 0], -axes_z[:, 0], axes_z[:, 1], -axes_z[:, 1],
                            axes_z[:, 2], -axes_z[:, 2]], dim=-1)

    def _pad_state(self, p: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos (N,3), quat (N,4)) of the ACTIVE pad at arena slot `p` (gathered by the
        per-env sampled color)."""
        pos = torch.stack([b.data.root_pos_w for b in self.pads.values()], dim=1)  # (N,6,3)
        quat = torch.stack([b.data.root_quat_w for b in self.pads.values()], dim=1)
        idx = self.pad_color[:, p]
        ar = torch.arange(pos.shape[0], device=pos.device)
        return pos[ar, idx], quat[ar, idx]

    def face_up(self, nm: str, p: int) -> torch.Tensor:
        """(N,) bool: die `nm` shows arena-pad `p`'s color face-up (within face_max_deg)."""
        dots = self._face_dots(nm)
        ar = torch.arange(dots.shape[0], device=dots.device)
        return dots[ar, self.pad_color[:, p]] >= math.cos(math.radians(self.cfg.face_max_deg))

    def on_pad(self, nm: str, p: int) -> torch.Tensor:
        """(N,) bool: die `nm` centered in pad `p`'s window (pad body frame) AT pad
        height (a die stacked on the other die reads ~0.10 too high and is rejected)."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        pp, pq = self._pad_state(p)
        rel = quat_apply_inverse(pq, self.dice[nm].data.root_pos_w - pp)
        dz = rel[:, 2] - (c.pad_t / 2 + c.die_size / 2)
        return (rel[:, 0].abs() <= c.pad_xy_tol) & (rel[:, 1].abs() <= c.pad_xy_tol) \
            & (dz >= c.pad_dz_lo) & (dz <= c.pad_dz_hi)

    def settled(self, nm: str) -> torch.Tensor:
        d = self.dice[nm]
        return (d.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin) \
            & (d.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang)

    def served(self, p: int) -> torch.Tensor:
        """(N,) bool: some settled die sits in pad `p`'s window with `p`'s color up."""
        out = None
        for nm in self.die_names:
            s = self.face_up(nm, p) & self.on_pad(nm, p) & self.settled(nm)
            out = s if out is None else out | s
        return out

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch, per arena pad: reorientation done (some die ever face-up in the pad's
        color), transported (face-up within near_r of the pad), placed (face-up in the
        pad window at pad height), each physics substep."""
        c = self.cfg
        for p in range(2):
            pp, _pq = self._pad_state(p)
            show = None
            near = None
            placed = None
            for nm in self.die_names:
                fu = self.face_up(nm, p)
                d_xy = (self.dice[nm].data.root_pos_w[:, :2] - pp[:, :2]).norm(dim=-1)
                nr = fu & (d_xy < c.near_r)
                pl = fu & self.on_pad(nm, p)
                show = fu if show is None else show | fu
                near = nr if near is None else near | nr
                placed = pl if placed is None else placed | pl
            self.show_latch[:, p] = torch.maximum(self.show_latch[:, p], show.float())
            self.near_latch[:, p] = torch.maximum(self.near_latch[:, p], near.float())
            self.placed_latch[:, p] = torch.maximum(self.placed_latch[:, p], placed.float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: both arena pads served — each carries a settled die with the
        pad's color face-up in the pad window. The pads are farther apart than the
        window allows one die to straddle (asserted), so this implies two dice."""
        return self.served(0) & self.served(1)

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: per pad 0.10 * show latch + 0.10 * near latch + 0.25 *
        placed latch (sum = 0.90); 1.0 iff success(). Latched — credit never
        evaporates; the null policy scores ~0 (spawn up-faces exclude both pad
        colors)."""
        base = (0.10 * self.show_latch + 0.10 * self.near_latch
                + 0.25 * self.placed_latch).sum(dim=1).clamp(0.0, 0.90)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="dice_tumble", robot="null", env_spacing=3.0))

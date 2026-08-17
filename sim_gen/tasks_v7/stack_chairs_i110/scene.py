"""ChairTuckScene — slide each chair under the kneehole desk into the one bay it fits
(sim_gen task `stack_chairs_i110`).

Derived from rlbench/stack_chairs, but STRATEGICALLY different: the seed is transport +
STACKING — pick each chair up and pile it on top of the previous one, success being a
vertical chair-on-chair tower. Here no chair ever touches another chair and piling
counts for NOTHING. A kneehole desk stands on the floor: a top slab on four partition
walls forming THREE open bays of DIFFERENT widths (left-to-right order shuffled every
episode), closed at the back. Three chairs of DIFFERENT widths lie scattered in front.
The goal state has every chair slid HORIZONTALLY under the desk top into a bay — seat
under the slab, backrest docked against the desk's front edge — one chair per bay, all
upright and settled. A solver needs a different PLAN (no vertical stacking; a
size-to-bay matching that must be read from the scene — the wide chair fits ONLY the
wide bay, so a greedy wrong assignment dead-ends the episode; every placement is a
lateral insertion under an overhang, impossible to do by dropping from above) and a
different code structure (a per-bay occupancy predicate in the DESK's frame — depth
behind the front edge + lateral containment + upright/facing + floor-height band —
instead of any on-top-of / pile-height test).

Geometry (procedural; the desk is 6 KINEMATIC cuboids so bay order is re-randomized
every reset; each chair is one rigid compound body from a custom spawner):
  - desk: top slab 140x308x12 mm with its underside `clear_h` = 80 mm above the floor,
    4 partition walls 10 mm thick, a full-width back wall; bays 116 / 90 / 62 mm wide
    (order permuted per episode).
  - chair k: seat 80 mm deep x `chair_widths[k]` wide (100 / 72 / 48 mm; red / green /
    blue), seat top at 55 mm, backrest panel up to 120 mm. Seat (55 mm) passes under
    the slab (80 mm); the backrest (120 mm) CANNOT — so a tucked chair necessarily
    faces the desk and docks with its backrest against the slab's front edge (max seat-
    center depth = 40 mm behind the edge). Body origin at seat center, 50 mm above the
    floor (~ the natural CoM; spawn-authored MassAPI leaves CoM at the body origin).

Width arithmetic (asserted in `__post_init__`): chair k fits its matched bay with
>= 12 mm total clearance, does NOT fit any narrower bay (>= 8 mm interference), and no
two chairs fit side-by-side in the widest bay — so "every bay holds exactly one tucked
chair" forces the unique size matching through PHYSICS, not through the rubric (the
rubric never checks chair identity).

success(): every bay covered — exactly one chair whose seat center is >= `depth_min`
behind the desk's front edge (desk frame), laterally inside that bay's span, in the
floor-height band, upright within `upright_max_deg`, facing the desk within
`facing_max_deg`, with stillness SUSTAINED `settle_steps` substeps.

score(): per bay, 1.0 if covered NOW, else `tuck_credit` = 0.4 once ANY chair has ever
read tucked in that bay (latched in post_step — credit never evaporates); mean over
the 3 bays; 1.0 iff success(). Null policy scores ~0 (chairs scatter in FRONT of the
desk; nothing ever slides them under).

Per-episode randomization (readback-verified in smoke): desk position + yaw, the
left-to-right ORDER of the three bay widths, chair slot permutation + xy jitter + free
yaw. Heavy imports (isaaclab, pxr) are deferred so importing this module — and
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

# chair k is judged nowhere by identity — colors only make describe() unambiguous
CHAIRS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("red", (0.85, 0.10, 0.10)),
    ("green", (0.10, 0.70, 0.15)),
    ("blue", (0.12, 0.30, 0.85)),
)


# ----- custom compound spawner (the chair) -----------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _material(stage, path: str, static: float, dynamic: float):
    """One USD physics material (explicit friction + zero restitution — custom-spawner
    colliders otherwise land on engine defaults)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, size, center, color, contact_offset: float, material) -> None:
    """One axis-aligned box child prim (translate -> scale, authored exactly once — the
    duplicate-xformOp trap is avoided by never re-authoring an existing prim's ops)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(seg.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
        material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_chair(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The chair: 4 legs + seat slab + backrest panel — one rigid body. Origin at the
    seat center, `origin_h` above the leg bottoms (~ the natural CoM; MassAPI mass
    leaves the CoM at the body origin). Local +x = the chair's FRONT (the side that
    slides under the desk first); backrest at local -x. Zero sleep / stabilization
    thresholds: a sleeping body silently ignores applied wrenches, which the solve
    force-push depends on."""
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
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.20)

    mat = _material(stage, f"{prim_path}/phys_mat", 0.45, 0.40)
    co = cfg.contact_offset
    col = tuple(cfg.color)
    w, sd, st = cfg.width, cfg.seat_depth, cfg.seat_t
    oh, ls = cfg.origin_h, cfg.leg_s
    seat_lo = cfg.seat_top - oh - st  # seat underside, local z
    leg_h = (cfg.seat_top - st) - 0.003  # floor -> seat underside
    # legs (flush with the seat corners)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            _box(stage, f"{prim_path}/leg_{'f' if sx > 0 else 'b'}{'l' if sy > 0 else 'r'}",
                 (ls, ls, leg_h),
                 (sx * (sd / 2 - ls / 2), sy * (w / 2 - ls / 2), seat_lo - leg_h / 2),
                 col, co, mat)
    # seat slab
    _box(stage, f"{prim_path}/seat", (sd, w, st),
         (0.0, 0.0, seat_lo + st / 2), col, co, mat)
    # backrest panel (local -x, rises to back_top above the floor)
    bh = cfg.back_top - cfg.seat_top
    _box(stage, f"{prim_path}/back", (cfg.back_t, w, bh),
         (-(sd / 2 + cfg.back_t / 2), 0.0, (cfg.seat_top - oh) + bh / 2), col, co, mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg class (lazily — app-free import)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "chair" not in _SPAWNER_CACHE:

        @configclass
        class ChairSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_chair)
            mass: float = 0.15
            color: tuple = (0.8, 0.1, 0.1)
            width: float = 0.100
            seat_depth: float = 0.080
            seat_t: float = 0.008
            seat_top: float = 0.055
            leg_s: float = 0.008
            back_t: float = 0.008
            back_top: float = 0.120
            origin_h: float = 0.050
            contact_offset: float = 0.002

        _SPAWNER_CACHE["chair"] = ChairSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ChairTuckSceneCfg(BaseCfg):
    """Config for `ChairTuckScene`. The width arithmetic that makes the size matching
    physical (fits / misfits / no-two-in-one) is asserted in `__post_init__`."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    depth_min: float = tunable(0.020)  # seat center this far behind the desk front edge (m)
    upright_max_deg: float = tunable(15.0)  # chair up-axis within this of world-up
    facing_max_deg: float = tunable(35.0)  # chair front axis within this of the tuck direction
    z_lo: float = tunable(0.030)  # chair-origin height band: ON THE FLOOR under the desk
    z_hi: float = tunable(0.070)  # (a chair on the desk TOP reads 0.142 and fails)
    settle_lin: float = tunable(0.05)  # max chair |lin vel| at judging (m/s)
    settle_ang: float = tunable(0.60)  # max chair |ang vel| at judging (rad/s)
    settle_steps: int = tunable(20)  # substeps of SUSTAINED stillness
    tuck_credit: float = tunable(0.4)  # per-bay latched credit once ANY chair ever tucked there

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    desk_pos: tuple = tunable((0.45, 0.0))  # desk-frame origin on the floor
    desk_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the desk
    desk_yaw_deg: float = tunable(14.0)  # uniform +/- desk yaw
    scatter_x: float = tunable(0.08)  # chair scatter row x (in front of the desk)
    scatter_dy: float = tunable(0.16)  # scatter slot pitch along y
    scatter_jitter: float = tunable(0.02)  # uniform +/- xy jitter per chair
    chair_yaw_deg: float = tunable(180.0)  # uniform +/- yaw per scattered chair

    # --- info: structure (what the spawners author) ------------------------------------------
    bay_widths: tuple = info((0.116, 0.090, 0.062))  # matched to chair_widths, index-wise
    chair_widths: tuple = info((0.100, 0.072, 0.048))  # red, green, blue
    wall_t: float = info(0.010)  # partition wall thickness
    desk_depth: float = info(0.140)  # slab (and bay) depth
    clear_h: float = info(0.080)  # floor -> slab underside
    top_t: float = info(0.012)  # slab thickness
    seat_depth: float = info(0.080)
    seat_t: float = info(0.008)
    seat_top: float = info(0.055)  # floor -> seat top (must pass under clear_h)
    back_t: float = info(0.008)
    back_top: float = info(0.120)  # floor -> backrest top (must NOT pass under clear_h)
    leg_s: float = info(0.008)
    origin_h: float = info(0.050)  # body origin above the leg bottoms (~ CoM height)
    chair_mass: float = info(0.15)
    contact_offset: float = info(0.002)

    @property
    def desk_width(self) -> float:
        return sum(self.bay_widths) + 4 * self.wall_t

    def __post_init__(self) -> None:
        c = self
        bw, cw = c.bay_widths, c.chair_widths
        assert len(bw) == 3 and len(cw) == 3
        assert all(bw[i] > bw[i + 1] for i in range(2)) and all(cw[i] > cw[i + 1] for i in range(2))
        # -- chair k fits its matched bay with sliding clearance --
        assert all(bw[i] - cw[i] >= 0.012 for i in range(3)), \
            "each chair must fit its matched bay with >= 12 mm total clearance"
        # -- chair k does NOT fit any narrower bay (physical misfit, not a rubric clause) --
        assert all(cw[i] - bw[j] >= 0.008 for i in range(3) for j in range(3) if j > i), \
            "a chair must physically not fit any narrower bay"
        # -- no two chairs fit side-by-side in the widest bay: exactly-one is physical --
        assert cw[1] + cw[2] >= bw[0] + 0.002, \
            "two chairs must not fit side-by-side in the widest bay"
        # -- the dock (backrest on the slab edge) is deeper than the depth gate --
        assert c.seat_depth / 2 >= c.depth_min + 0.012, \
            "the docked seat center must clear depth_min with margin"
        # -- the seat passes under the slab; the backrest cannot --
        assert c.seat_top <= c.clear_h - 0.020, "the seat must pass under the slab"
        assert c.back_top >= c.clear_h + 0.030, "the backrest must NOT pass under the slab"
        # -- height band: on-floor origin inside, on-the-slab origin far outside --
        assert c.z_lo < c.origin_h < c.z_hi
        assert c.clear_h + c.top_t + c.origin_h > c.z_hi + 0.05, \
            "a chair standing ON the desk top must fail the height band"
        # -- scatter zone clear of the desk at worst-case jitter+yaw --
        worst_edge = (c.desk_pos[0] - c.desk_jitter - c.desk_depth / 2
                      - math.sin(math.radians(c.desk_yaw_deg)) * c.desk_width / 2)
        assert c.scatter_x + c.scatter_jitter + 0.07 < worst_edge, \
            "scattered chairs must never spawn inside the desk"
        # -- lateral gates are per-bay disjoint (separated by a wall) --
        assert c.wall_t > 0.0
        assert c.settle_steps >= 12
        assert 0.0 < c.tuck_credit < 1.0


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("chair_tuck")
class ChairTuckScene(BaseScene):
    cfg: ChairTuckSceneCfg

    def __init__(self, cfg: ChairTuckSceneCfg | None = None) -> None:
        super().__init__(cfg or ChairTuckSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        wood = (0.48, 0.32, 0.18)
        wood2 = (0.56, 0.40, 0.24)

        def piece(name: str, size, color) -> RigidObjectCfg:
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.30, dynamic_friction=0.25, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                # nominal spawn pose; reset() writes the real per-episode poses
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.desk_pos[0], 0.0, 0.5)),
            )

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.45, dynamic_friction=0.40, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "slab": piece("Slab", (c.desk_depth, c.desk_width, c.top_t), wood),
            "backwall": piece("BackWall", (c.wall_t, c.desk_width, c.clear_h), wood2),
        }
        for k in range(4):
            out[f"wall_{k}"] = piece(f"Wall_{k}",
                                     (c.desk_depth - c.wall_t, c.wall_t, c.clear_h), wood2)
        for i, (name, rgb) in enumerate(CHAIRS):
            out[f"chair_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Chair_" + name,
                spawn=sp["chair"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.chair_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.chair_mass, color=rgb, width=c.chair_widths[i],
                    seat_depth=c.seat_depth, seat_t=c.seat_t, seat_top=c.seat_top,
                    leg_s=c.leg_s, back_t=c.back_t, back_top=c.back_top,
                    origin_h=c.origin_h, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.scatter_x, (i - 1) * c.scatter_dy, c.origin_h + 0.003)),
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
        n, dev = env.num_envs, env.device
        self.chairs: list[RigidObject] = [env.iscene[f"chair_{nm}"] for nm, _ in CHAIRS]
        self.pieces: dict[str, RigidObject] = {
            nm: env.iscene[nm] for nm in ("slab", "backwall", "wall_0", "wall_1",
                                          "wall_2", "wall_3")}
        self.env_origins = env.iscene.env_origins
        # per-episode desk layout (written at reset, readable by rubric + solver)
        self.desk_xy = torch.zeros(n, 2, device=dev)
        self.desk_yaw = torch.zeros(n, device=dev)
        self.bay_w = torch.zeros(n, 3, device=dev)  # bay widths, left->right
        self.bay_y = torch.zeros(n, 3, device=dev)  # bay center y, DESK frame, left->right
        # progress latch (post_step): bay b has ever had a chair tucked in it
        self.tuck_latch = torch.zeros(n, 3, device=dev)
        # sustained per-chair stillness counter (consecutive still substeps)
        self.still_count = torch.zeros(n, 3, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the desk pose + yaw and the left-to-right ORDER of the
        three bay widths, write the 6 kinematic desk pieces accordingly; scatter the
        chairs in front (slot permutation + jitter + free yaw); zero the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- desk pose ---
        dxy = torch.tensor(c.desk_pos, device=dev).expand(m, 2).clone()
        dxy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.desk_jitter
        dyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.desk_yaw_deg)
        self.desk_xy[env_ids] = dxy
        self.desk_yaw[env_ids] = dyaw

        # --- bay order: permutation of the three widths (rand-argsort, not randint) ---
        perm = torch.rand(m, 3, device=dev).argsort(dim=1)  # perm[:, slot] = width index
        widths = torch.tensor(c.bay_widths, device=dev)
        bw = widths[perm]  # (m, 3) left->right
        self.bay_w[env_ids] = bw
        # wall/bay y positions (desk frame): edge, wall, bay, wall, bay, wall, bay, wall
        wall_y = torch.zeros(m, 4, device=dev)
        bay_y = torch.zeros(m, 3, device=dev)
        y = torch.full((m,), -self.cfg.desk_width / 2, device=dev)
        for k in range(3):
            wall_y[:, k] = y + c.wall_t / 2
            bay_y[:, k] = y + c.wall_t + bw[:, k] / 2
            y = y + c.wall_t + bw[:, k]
        wall_y[:, 3] = y + c.wall_t / 2
        self.bay_y[env_ids] = bay_y

        cy, sy = torch.cos(dyaw), torch.sin(dyaw)

        def write_piece(body, lx, ly, lz: float) -> None:
            """Place one kinematic desk piece from its desk-frame offset (lx, ly, lz)."""
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = dxy[:, 0] + lx * cy - ly * sy
            st[:, 1] = dxy[:, 1] + lx * sy + ly * cy
            st[:, 2] = lz
            st[:, 3] = torch.cos(dyaw / 2)
            st[:, 6] = torch.sin(dyaw / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        zero = torch.zeros(m, device=dev)
        write_piece(self.pieces["slab"], zero, zero, c.clear_h + c.top_t / 2)
        write_piece(self.pieces["backwall"],
                    zero + (c.desk_depth / 2 - c.wall_t / 2), zero, c.clear_h / 2)
        for k in range(4):
            # walls sit flush with the front edge, ending where the back wall begins
            write_piece(self.pieces[f"wall_{k}"], zero - c.wall_t / 2, wall_y[:, k],
                        c.clear_h / 2)

        # --- chairs: slot permutation + jitter + free yaw, in front of the desk ---
        slot_perm = torch.rand(m, 3, device=dev).argsort(dim=1)
        slots = torch.tensor([-c.scatter_dy, 0.0, c.scatter_dy], device=dev)
        yaw_amp = math.radians(c.chair_yaw_deg)
        for i in range(3):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = c.scatter_x
            st[:, 1] = slots[slot_perm[:, i]]
            st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.scatter_jitter
            st[:, 2] = c.origin_h + 0.003
            half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            self.chairs[i].write_root_state_to_sim(st, env_ids)

        self.tuck_latch[env_ids] = 0.0
        self.still_count[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "chairs": [b.data.root_state_w[env_ids].clone() for b in self.chairs],
            "pieces": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in self.pieces.items()},
            "desk_xy": self.desk_xy[env_ids].clone(),
            "desk_yaw": self.desk_yaw[env_ids].clone(),
            "bay_w": self.bay_w[env_ids].clone(),
            "bay_y": self.bay_y[env_ids].clone(),
            "tuck_latch": self.tuck_latch[env_ids].clone(),
            "still_count": self.still_count[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for b, st in zip(self.chairs, state["chairs"]):
            b.write_root_state_to_sim(st, env_ids)
        for nm, b in self.pieces.items():
            b.write_root_state_to_sim(state["pieces"][nm], env_ids)
        self.desk_xy[env_ids] = state["desk_xy"]
        self.desk_yaw[env_ids] = state["desk_yaw"]
        self.bay_w[env_ids] = state["bay_w"]
        self.bay_y[env_ids] = state["bay_y"]
        self.tuck_latch[env_ids] = state["tuck_latch"]
        self.still_count[env_ids] = state["still_count"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        bw = [f"{v * 1000:.0f}" for v in c.bay_widths]
        cw = [f"{v * 1000:.0f}" for v in c.chair_widths]
        return (
            "A small kneehole desk stands on the floor: a flat wooden top "
            f"({c.desk_width * 1000:.0f} mm wide, {c.desk_depth * 1000:.0f} mm deep, its "
            f"underside {c.clear_h * 1000:.0f} mm above the floor) resting on four "
            "partition walls that divide the space under it into THREE open bays of "
            f"different widths — {bw[0]}, {bw[1]} and {bw[2]} mm — closed at the back, "
            "open only at the desk's front edge. The desk's position, its heading and "
            "the LEFT-TO-RIGHT ORDER of the three bay widths change every episode: read "
            "them by looking. Scattered on the floor in front of the desk stand three "
            "chairs of different widths (each a seat on four legs with a tall backrest; "
            f"seat top {c.seat_top * 1000:.0f} mm, backrest top {c.back_top * 1000:.0f} mm "
            f"high): a RED one {cw[0]} mm wide, a GREEN one {cw[1]} mm wide and a BLUE "
            f"one {cw[2]} mm wide, at random positions and headings.\n"
            "Goal: tuck every chair under the desk — for each chair, slide it seat-first "
            "in through a bay's front opening until the seat is under the desk top and "
            "the backrest meets the desk's front edge — so that each of the three bays "
            "ends up holding exactly one upright chair, and all three chairs rest still. "
            "The backrest is taller than the opening, so a chair only goes in facing the "
            "desk, and it cannot be dropped in from above through the top. Each chair "
            "only fits through a bay wider than itself: the wide red chair fits ONLY the "
            "widest bay, the green chair does not fit the narrowest bay — putting a "
            "chair in a bay another chair needs leaves that chair with no bay that fits "
            "it. Any tucking order is fine.\n"
            "What does NOT count: a chair left in front of, beside, or on TOP of the "
            "desk; a chair only partially slid in (seat center less than "
            f"{c.depth_min * 1000:.0f} mm behind the front edge); a toppled or sideways-"
            "lying chair pushed underneath; chairs stacked on one another; two chairs "
            "forced toward the same bay."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide each chair seat-first under the desk into a bay it fits through, "
            "until its backrest meets the desk edge and every bay holds exactly one "
            "upright chair. A chair left outside, on top of the desk, only partially "
            "tucked, toppled, or in a bay too narrow for another chair's only fit "
            "does not count."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _chair_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(chair pos (N,C,3) env-local, chair quat (N,C,4)), world frame."""
        pos = torch.stack([b.data.root_pos_w for b in self.chairs], dim=1)
        quat = torch.stack([b.data.root_quat_w for b in self.chairs], dim=1)
        return pos - self.env_origins[:, None, :], quat

    def chair_desk_xy(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(x_t (N,C), y_t (N,C)): chair origins in the DESK frame (+x = tuck
        direction, origin at the desk center; the front edge is x_t = -depth/2)."""
        pos, _q = self._chair_tensors()
        d = pos[:, :, 0:2] - self.desk_xy[:, None, :]
        cy = torch.cos(self.desk_yaw)[:, None]
        sy = torch.sin(self.desk_yaw)[:, None]
        x_t = d[:, :, 0] * cy + d[:, :, 1] * sy
        y_t = -d[:, :, 0] * sy + d[:, :, 1] * cy
        return x_t, y_t

    def chair_up_z(self) -> torch.Tensor:
        """(N,C): world-z component of each chair's body +z axis (+1 = upright)."""
        from isaaclab.utils.math import quat_apply

        _p, quat = self._chair_tensors()
        n, cc = quat.shape[0], quat.shape[1]
        ez = torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(n * cc, 3)
        return quat_apply(quat.reshape(n * cc, 4), ez).reshape(n, cc, 3)[:, :, 2]

    def chair_facing(self) -> torch.Tensor:
        """(N,C): cosine between the chair's front axis (horizontal projection) and
        the desk's tuck direction (+x_t)."""
        from isaaclab.utils.math import quat_apply

        _p, quat = self._chair_tensors()
        n, cc = quat.shape[0], quat.shape[1]
        ex = torch.tensor([1.0, 0.0, 0.0], device=quat.device).expand(n * cc, 3)
        f = quat_apply(quat.reshape(n * cc, 4), ex).reshape(n, cc, 3)
        fh = f[:, :, 0:2] / f[:, :, 0:2].norm(dim=-1, keepdim=True).clamp(min=1e-9)
        cy = torch.cos(self.desk_yaw)[:, None]
        sy = torch.sin(self.desk_yaw)[:, None]
        return fh[:, :, 0] * cy + fh[:, :, 1] * sy

    def chair_depth(self) -> torch.Tensor:
        """(N,C): seat-center depth BEHIND the desk's front edge (negative = outside)."""
        x_t, _y = self.chair_desk_xy()
        return x_t + self.cfg.desk_depth / 2

    def tucked(self) -> torch.Tensor:
        """(N,C,B) bool, geometric: chair c counts as tucked in bay b — deep enough
        behind the front edge, laterally inside bay b's span (desk frame), origin in
        the on-floor height band, upright, facing the desk."""
        c = self.cfg
        x_t, y_t = self.chair_desk_xy()
        pos, _q = self._chair_tensors()
        depth_ok = (self.chair_depth() > c.depth_min) & (x_t < c.desk_depth / 2)
        z_ok = (pos[:, :, 2] > c.z_lo) & (pos[:, :, 2] < c.z_hi)
        up_ok = self.chair_up_z() >= math.cos(math.radians(c.upright_max_deg))
        face_ok = self.chair_facing() >= math.cos(math.radians(c.facing_max_deg))
        ok_c = depth_ok & z_ok & up_ok & face_ok  # (N,C)
        lat = (y_t[:, :, None] - self.bay_y[:, None, :]).abs() \
            < self.bay_w[:, None, :] / 2  # (N,C,B)
        return ok_c[:, :, None] & lat

    def _still_now(self) -> torch.Tensor:
        """(N,C) bool: chair instantaneously below the stillness thresholds."""
        c = self.cfg
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.chairs], dim=1)
        w = torch.stack([b.data.root_ang_vel_w.norm(dim=-1) for b in self.chairs], dim=1)
        return (v < c.settle_lin) & (w < c.settle_ang)

    def settled(self) -> torch.Tensor:
        """(N,C) bool: chair stillness SUSTAINED `settle_steps` consecutive substeps."""
        return self.still_count >= float(self.cfg.settle_steps)

    def covered(self) -> torch.Tensor:
        """(N,B) bool: bay b holds exactly one tucked, settled chair. Two chairs
        cannot physically be tucked in one bay (width arithmetic asserted in cfg)."""
        occ = self.tucked() & self.settled()[:, :, None]
        return occ.sum(dim=1) == 1

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch per-bay tuck progress (any chair ever geometrically tucked in bay b)
        and run the sustained-stillness counters, every physics substep."""
        self.tuck_latch = torch.maximum(self.tuck_latch,
                                        self.tucked().any(dim=1).float())
        self.still_count = torch.where(self._still_now(), self.still_count + 1.0,
                                       torch.zeros_like(self.still_count))

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: every bay holds exactly one tucked, settled, upright chair."""
        return self.covered().all(dim=1)

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: per bay, 1.0 if covered NOW else `tuck_credit` once a
        chair has ever been tucked there (latched — credit never evaporates); mean
        over bays; equals 1.0 iff success(). Null policy ~0 (chairs scatter in FRONT
        of the desk; nothing ever slides them under)."""
        per = torch.where(self.covered(), torch.ones_like(self.tuck_latch),
                          self.cfg.tuck_credit * self.tuck_latch)
        return per.mean(dim=1)


register_env("simgen", lambda: EnvCfg(scene="chair_tuck", robot="null", env_spacing=3.0))

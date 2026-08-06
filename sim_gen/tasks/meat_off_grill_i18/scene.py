"""SearServeScene — cook each patty on the one-slot sear pad for a bounded time, then serve.

Derived from rlbench/meat_off_grill, but STRATEGICALLY INVERTED: the seed's whole plan is a
single free pick-and-place — the meat already sits on the grill and success is merely moving
it off. Here transport is the trivial part and the judged quantity is a TIME INTEGRAL the
seed never had: each raw patty (starting on a prep board, NOT on the grill) must rest on the
grill's sear pad long enough to cook (`cook_min` seconds of accumulated pad contact), but be
taken off before it burns (`burn_time` — a PERMANENT spoil latch), and only then placed on
the serving plate. Raw meat delivered to the plate — a flawless execution of the seed's
transport plan — scores exactly zero, and grabbing the meat off the grill promptly (the
seed's literal move) fails the cook gate. The sear pad is deliberately sized so only ONE
patty fits at a time (two non-overlapping patties cannot both be fully inside it — asserted
at config time), so with two patties present the episode is a genuine scheduling problem:
cook A, serve A, cook B, serve B, with a per-patty deadline between "cooked" and "burned".

Judging is physical and final-state-plus-history: pad contact is accumulated per physics
substep from real settled poses (post_step latch — the transient-achievement pattern);
"served" is a current-state predicate (settled flat on the plate, cooked, not burned).
Rubric per present patty: 0.2 once it has ever rested on the pad (latched), 0.55 once
cooked (latched via the cook-time integral), 1.0 while served; a burned patty is capped at
0.05 forever (spoilage is the one intentional non-monotonicity — a correct schedule never
burns). Score = 0.9 * mean over present patties, forced to 1.0 iff success (all present
patties served). Doing nothing scores 0.

Embodiment (the v2 contract): the geometry is laid out base-polar for a single Franka on
the ground at `stage_anchor` (solve.py's base pose). Patty diameters (60 / 54 mm) are the
proven parallel-jaw pinch sizes; every grasp/place radius stays inside the measured
0.40-0.64 m ground-level envelope; the fixtures cannot overlap under worst-case jitter+yaw.

Per-episode randomization: board/grill/plate poses (xy jitter + yaw), patty slots on the
board (xy jitter), AND patty-count subset sampling (1-2 present; the absent patty parks in
an off-camera depot), so a memorized fixed schedule fails. All assets are procedural: the
grill is a kinematic compound body (charcoal box + raised near-black sear pad), the plate a
kinematic white disc, the board a kinematic slab, the patties plain dynamic cylinders.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering the
scene — stays app-free.
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


# ----- custom compound spawner (grill) ----------------------------------------------------------
# One kinematic rigid body: charcoal body box + raised near-black sear pad box. Authored with
# raw pxr APIs; `isaaclab.sim.utils.clone` provides the per-env replicate machinery. Fresh
# Define per prim -> xformOps authored once (idempotent under clone).

_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_grill(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the grill at `prim_path`: a KINEMATIC rigid body (re-placed per episode by
    root-state writes). Body frame: the body box spans local z in [-h/2, +h/2]; the sear pad
    (a 2*pad_half square, pad_t thick, near-black) sits centered on top, its top face at
    local z = h/2 + pad_t."""
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

    def box(name: str, size, center, color) -> None:
        b = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        b.CreateSizeAttr(1.0)
        bxf = UsdGeom.Xformable(b.GetPrim())
        bxf.AddTranslateOp().Set(Gf.Vec3d(*center))
        bxf.AddScaleOp().Set(Gf.Vec3f(*size))
        b.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdPhysics.CollisionAPI.Apply(b.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(b.GetPrim())
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    sx, sy, sz = cfg.body_size
    box("body", (sx, sy, sz), (0.0, 0.0, 0.0), cfg.body_color)
    box("pad", (2 * cfg.pad_half, 2 * cfg.pad_half, cfg.pad_t),
        (0.0, 0.0, sz / 2 + cfg.pad_t / 2), cfg.pad_color)
    return root


def _grill_spawner_cfg(*, body_size: tuple, pad_half: float, pad_t: float, body_color: tuple,
                       pad_color: tuple, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "grill" not in _SPAWNER_CACHE:

        @configclass
        class GrillSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_grill)
            body_size: tuple = (0.24, 0.20, 0.10)
            pad_half: float = 0.0455
            pad_t: float = 0.008
            body_color: tuple = (0.28, 0.28, 0.30)
            pad_color: tuple = (0.09, 0.05, 0.05)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["grill"] = GrillSpawnerCfg

    return _SPAWNER_CACHE["grill"](
        mass_props=sim_utils.MassPropertiesCfg(mass=3.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        body_size=body_size, pad_half=pad_half, pad_t=pad_t,
        body_color=body_color, pad_color=pad_color, contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SearServeSceneCfg(BaseCfg):
    """Config for `SearServeScene`. The one-slot property (two patties cannot both be fully
    inside the sear pad) and the plate capacity (both patties fit, non-overlapping) are
    asserted geometrically in __post_init__ — honesty by construction."""

    # --- tunable: cook window (the rubric's core) ----------------------------------------------
    cook_min: float = tunable(1.5)  # accumulated pad-seconds to count as cooked
    burn_time: float = tunable(4.0)  # accumulated pad-seconds after which the patty is SPOILED
    pad_z_tol: float = tunable(0.008)  # patty must rest at pad-top height within this (m)
    plate_z_tol: float = tunable(0.008)  # served: patty bottom at plate top within this (m)
    upright_max_deg: float = tunable(15.0)  # patty axis within this of world-up (flat side down)
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging served (m/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    reset_pos_jitter: float = tunable(0.035)  # uniform +/- xy jitter for board/grill/plate
    patty_jitter: float = tunable(0.012)  # uniform +/- xy jitter per patty around its board slot
    reset_yaw_deg: float = tunable(180.0)  # uniform +/- yaw per fixture at reset
    subset_sample: bool = tunable(True)  # per-episode patty-count sampling
    min_present: int = tunable(1)  # lower bound of sampled patty count

    # --- tunable: placement (base-polar staging: everything 0.40-0.64 m from stage_anchor) ------
    board_pos: tuple = tunable((-0.22, -0.40))  # prep board centre
    grill_pos: tuple = tunable((0.00, 0.00))  # grill centre
    plate_pos: tuple = tunable((-0.22, 0.40))  # serving plate centre
    surface_z: float = tunable(0.0)  # work-surface height; 0 = on the ground

    # --- info: intended robot base (solve.py places the Franka here) ----------------------------
    stage_anchor: tuple = info((-0.55, 0.0))

    # --- info: grill (kinematic compound) --------------------------------------------------------
    grill_size: tuple = info((0.24, 0.20, 0.10))  # body box (x, y, height)
    pad_half: float = info(0.0455)  # sear pad half-extent — the ONE-SLOT knob (see assert)
    pad_t: float = info(0.008)  # pad thickness (raised so the hot zone is visually distinct)
    grill_color: tuple = info((0.28, 0.28, 0.30))
    pad_color: tuple = info((0.09, 0.05, 0.05))

    # --- info: plate + board (kinematic) ---------------------------------------------------------
    plate_r: float = info(0.11)
    plate_t: float = info(0.012)
    plate_color: tuple = info((0.95, 0.95, 0.93))
    board_size: tuple = info((0.26, 0.20, 0.014))
    board_color: tuple = info((0.72, 0.55, 0.34))
    plate_slots: tuple = info(((0.048, 0.0), (-0.048, 0.0)))  # serving spots (world-frame offsets)

    # --- info: patties (name, radius, height, mass, rgb) — the seed's steak + chicken ------------
    # Diameters 60 / 54 mm: the proven Franka parallel-jaw pinch sizes (80 mm max aperture).
    patties: tuple = info((
        ("steak", 0.030, 0.022, 0.15, (0.48, 0.13, 0.10)),
        ("chicken", 0.027, 0.020, 0.12, (0.93, 0.80, 0.60)),
    ))
    board_slots: tuple = info(((-0.055, 0.0), (0.055, 0.0)))  # board-frame patty start slots
    contact_offset: float = info(0.002)
    parking_pos: tuple = info((1.0, 1.0))  # off-camera ground depot for the absent patty

    # Derived (filled in __post_init__).
    pad_top_local: float = field(default=None, init=False)  # pad top face, grill body frame
    n_patties: int = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.pad_top_local = round(self.grill_size[2] / 2 + self.pad_t, 4)
        self.n_patties = len(self.patties)
        radii = [p[1] for p in self.patties]
        # One-slot honesty: with every patty fully inside the pad (centre within
        # pad_half - r per axis), the max possible centre separation is the diagonal of the
        # combined slack boxes — it must be SMALLER than the touching distance r_i + r_j
        # (with an 8 mm buffer: both-fully-inside would need >= 8 mm of interpenetration).
        for i in range(len(radii)):
            for j in range(i + 1, len(radii)):
                mx = (self.pad_half - radii[i]) + (self.pad_half - radii[j])
                assert math.hypot(mx, mx) < radii[i] + radii[j] - 0.008, \
                    "sear pad is not one-slot: two patties could cook simultaneously"
        # Every patty individually fits on the pad and the plate with margin.
        for _n, r, _h, _m, _c in self.patties:
            assert self.pad_half - r >= 0.014, "patty barely fits the pad"
            assert self.plate_r - r >= 0.06, "patty barely fits the plate"
        # Both plate slots hold their patties fully inside, non-overlapping.
        (x0, y0), (x1, y1) = self.plate_slots
        assert math.hypot(x0 - x1, y0 - y1) > radii[0] + radii[1] + 0.008
        for (sx, sy), r in zip(self.plate_slots, radii):
            assert math.hypot(sx, sy) + r < self.plate_r - 0.005
        # Board slots keep the patties apart even at worst-case jitter.
        (bx0, by0), (bx1, by1) = self.board_slots
        assert math.hypot(bx0 - bx1, by0 - by1) > radii[0] + radii[1] + 2 * self.patty_jitter


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("sear_and_serve")
class SearServeScene(BaseScene):
    cfg: SearServeSceneCfg

    def __init__(self, cfg: SearServeSceneCfg | None = None) -> None:
        super().__init__(cfg or SearServeSceneCfg())

    # ----- assets ------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the three kinematic fixtures (prep board, grill, plate — nominal
        poses; reset() re-places everything) and the dynamic patty cylinders on the board."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.surface_z

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
        }

        out["board"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Board",
            spawn=sim_utils.CuboidCfg(
                size=c.board_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002,
                                                                 rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.board_color),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.board_pos[0], c.board_pos[1], z0 + c.board_size[2] / 2)),
        )

        out["grill"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Grill",
            spawn=_grill_spawner_cfg(
                body_size=c.grill_size, pad_half=c.pad_half, pad_t=c.pad_t,
                body_color=c.grill_color, pad_color=c.pad_color,
                contact_offset=c.contact_offset,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.grill_pos[0], c.grill_pos[1], z0 + c.grill_size[2] / 2)),
        )

        out["plate"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plate",
            spawn=sim_utils.CylinderCfg(
                radius=c.plate_r, height=c.plate_t,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002,
                                                                 rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.plate_color),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.plate_pos[0], c.plate_pos[1], z0 + c.plate_t / 2)),
        )

        for i, (name, r, h, mass, rgb) in enumerate(c.patties):
            sx, sy = c.board_slots[i]
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Patty_" + name,
                spawn=sim_utils.CylinderCfg(
                    radius=r, height=h,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        linear_damping=0.10, angular_damping=0.40,
                        max_depenetration_velocity=0.5),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.7, dynamic_friction=0.6, restitution=0.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.board_pos[0] + sx, c.board_pos[1] + sy,
                         z0 + c.board_size[2] + h / 2 + 0.003)),
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

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        self.board: RigidObject = env.iscene["board"]
        self.grill: RigidObject = env.iscene["grill"]
        self.plate: RigidObject = env.iscene["plate"]
        self.patties: dict[str, RigidObject] = {
            name: env.iscene[name] for name, _r, _h, _m, _c in c.patties}
        self.env_origins = env.iscene.env_origins
        n, p = env.num_envs, c.n_patties
        # present[e, i]: patty i participates in episode e (sampled at reset; judged subset).
        self.present = torch.ones(n, p, dtype=torch.bool, device=env.device)
        # cook_t[e, i]: accumulated seconds patty i has rested on the sear pad (post_step).
        self.cook_t = torch.zeros(n, p, device=env.device)
        # burned[e, i]: cook_t exceeded burn_time at least once — PERMANENT spoil latch.
        self.burned = torch.zeros(n, p, dtype=torch.bool, device=env.device)
        # ever_on_pad[e, i]: transient-achievement latch paying the first rubric step.
        self.ever_on_pad = torch.zeros(n, p, dtype=torch.bool, device=env.device)
        self._r = torch.tensor([r for _n, r, _h, _m, _c in c.patties], device=env.device)
        self._h = torch.tensor([h for _n, _r, h, _m, _c in c.patties], device=env.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the present patty subset, place board / grill / plate with
        xy jitter + yaw, seat present patties RAW on their board slots (board-frame, so the
        slots follow the board's yaw), park the absent patty, clear all cook state."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        yaw_amp = math.radians(c.reset_yaw_deg)

        self.cook_t[env_ids] = 0.0
        self.burned[env_ids] = False
        self.ever_on_pad[env_ids] = False

        # --- subset sampling: k ~ U{min_present..n_patties} present patties ---
        if c.subset_sample:
            k = torch.randint(c.min_present, c.n_patties + 1, (m,), device=dev)
        else:
            k = torch.full((m,), c.n_patties, dtype=torch.long, device=dev)
        rank = torch.rand(m, c.n_patties, device=dev).argsort(dim=1).argsort(dim=1)
        self.present[env_ids] = rank < k.unsqueeze(1)

        def yawed_state(base_xy: tuple, z: float) -> tuple[torch.Tensor, torch.Tensor]:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = base_xy[0]
            st[:, 1] = base_xy[1]
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
            st[:, 2] = z
            yaw = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp
            st[:, 3] = torch.cos(yaw / 2)
            st[:, 6] = torch.sin(yaw / 2)
            st[:, 0:3] += origin
            return st, yaw

        board_st, board_yaw = yawed_state(c.board_pos, c.surface_z + c.board_size[2] / 2)
        self.board.write_root_state_to_sim(board_st, env_ids)
        grill_st, _ = yawed_state(c.grill_pos, c.surface_z + c.grill_size[2] / 2)
        self.grill.write_root_state_to_sim(grill_st, env_ids)
        plate_st, _ = yawed_state(c.plate_pos, c.surface_z + c.plate_t / 2)
        self.plate.write_root_state_to_sim(plate_st, env_ids)

        cy, sy = torch.cos(board_yaw), torch.sin(board_yaw)
        for i, (name, _r, h, _mass, _rgb) in enumerate(c.patties):
            lx, ly = c.board_slots[i]
            seat = torch.zeros(m, 3, device=dev)
            seat[:, 0] = board_st[:, 0] - origin[:, 0] + (cy * lx - sy * ly)
            seat[:, 1] = board_st[:, 1] - origin[:, 1] + (sy * lx + cy * ly)
            seat[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.patty_jitter
            seat[:, 2] = c.surface_z + c.board_size[2] + h / 2 + 0.003
            park = torch.zeros(m, 3, device=dev)
            park[:, 0] = c.parking_pos[0] + i * 0.14
            park[:, 1] = c.parking_pos[1]
            park[:, 2] = h / 2 + 0.003
            pres = self.present[env_ids, i].unsqueeze(1)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.where(pres, seat, park)
            st[:, 3] = 1.0
            self.patties[name].write_root_state_to_sim(st, env_ids)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Accumulate real pad contact at sim rate: while a patty rests flat on the sear pad
        (geometric clause on the live pose), its cook clock advances by dt; crossing
        `burn_time` latches the permanent spoil flag."""
        onpad = self.on_pad()
        self.ever_on_pad |= onpad & self.present
        self.cook_t += onpad.float() * self.env.dt
        self.burned |= self.cook_t >= self.cfg.burn_time

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "board": self.board.data.root_state_w[env_ids].clone(),
            "grill": self.grill.data.root_state_w[env_ids].clone(),
            "plate": self.plate.data.root_state_w[env_ids].clone(),
            "patties": {n: b.data.root_state_w[env_ids].clone()
                        for n, b in self.patties.items()},
            "present": self.present[env_ids].clone(),
            "cook_t": self.cook_t[env_ids].clone(),
            "burned": self.burned[env_ids].clone(),
            "ever_on_pad": self.ever_on_pad[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.board.write_root_state_to_sim(state["board"], env_ids)
        self.grill.write_root_state_to_sim(state["grill"], env_ids)
        self.plate.write_root_state_to_sim(state["plate"], env_ids)
        for n, b in self.patties.items():
            b.write_root_state_to_sim(state["patties"][n], env_ids)
        self.present[env_ids] = state["present"]
        self.cook_t[env_ids] = state["cook_t"]
        self.burned[env_ids] = state["burned"]
        self.ever_on_pad[env_ids] = state["ever_on_pad"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A cookout station on the ground: a wooden prep board holding one or two raw "
            f"patties (a dark-red steak disc, {2 * c.patties[0][1] * 1000:.0f} mm across, "
            f"and/or a pale chicken disc, {2 * c.patties[1][1] * 1000:.0f} mm across — count "
            f"what you see), a charcoal grill (a {c.grill_size[2] * 1000:.0f} mm tall dark "
            f"box) whose top carries a raised near-black SEAR PAD "
            f"({2 * c.pad_half * 1000:.0f} mm square — room for only one patty at a time), "
            f"and a round white serving plate.\n"
            f"Goal: cook and serve every patty. A patty must rest flat on the sear pad for at "
            f"least {c.cook_min:.1f} s of accumulated contact to be cooked, but it BURNS — and "
            f"is spoiled forever — if its total pad time reaches {c.burn_time:.1f} s. Take each "
            f"patty off the grill inside that window and set it down flat on the plate. Raw or "
            f"burned meat on the plate counts for nothing; only cooked, unspoiled patties "
            f"resting on the plate score, and all present patties must end there. The pad "
            f"holds one patty at a time, so with two patties cook and serve them one after "
            f"the other."
        )

    # ----- predicates / rubric --------------------------------------------------------------------
    def _patty_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(pos_w (N,P,3), quat (N,P,4), |lin vel| (N,P)) for all patties, cfg order."""
        pos = torch.stack([b.data.root_pos_w for b in self.patties.values()], dim=1)
        quat = torch.stack([b.data.root_quat_w for b in self.patties.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.patties.values()], dim=1)
        return pos, quat, vel

    def _local_to(self, body, points: torch.Tensor) -> torch.Tensor:
        """Express world points (N,P,3) in `body`'s frame -> (N,P,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        n, p = points.shape[0], points.shape[1]
        q = body.data.root_quat_w[:, None, :].expand(n, p, 4).reshape(n * p, 4)
        pw = body.data.root_pos_w[:, None, :]
        return quat_apply_inverse(q, (points - pw).reshape(n * p, 3)).reshape(n, p, 3)

    def upright(self) -> torch.Tensor:
        """(N,P) bool: patty axis within `upright_max_deg` of world-up (flat side down)."""
        from isaaclab.utils.math import quat_apply

        pos, quat, _v = self._patty_tensors()
        n, p = pos.shape[0], pos.shape[1]
        ez = torch.tensor([0.0, 0.0, 1.0], device=pos.device).expand(n * p, 3)
        axis = quat_apply(quat.reshape(n * p, 4), ez).reshape(n, p, 3)
        return axis[:, :, 2] >= math.cos(math.radians(self.cfg.upright_max_deg))

    def on_pad(self) -> torch.Tensor:
        """(N,P) bool, grill body frame: patty FULLY inside the sear pad footprint
        (centre within pad_half - r per axis), resting at pad-top height, flat. The
        fully-inside clause is what makes the pad one-slot (see cfg assert)."""
        c = self.cfg
        loc = self._local_to(self.grill, self._patty_tensors()[0])
        slack = (c.pad_half - self._r)[None, :]
        xy_ok = (loc[:, :, 0].abs() < slack) & (loc[:, :, 1].abs() < slack)
        rest_z = c.pad_top_local + self._h[None, :] / 2
        z_ok = (loc[:, :, 2] - rest_z).abs() < c.pad_z_tol
        return xy_ok & z_ok & self.upright()

    def on_plate(self) -> torch.Tensor:
        """(N,P) bool, plate body frame: patty fully inside the plate disc, resting at
        plate-top height, flat. Purely geometric — a RAW patty can be on_plate (the seed-
        strategy control's sanity leg) without ever being served."""
        c = self.cfg
        loc = self._local_to(self.plate, self._patty_tensors()[0])
        xy_ok = loc[:, :, :2].norm(dim=-1) < (c.plate_r - self._r)[None, :]
        rest_z = c.plate_t / 2 + self._h[None, :] / 2
        z_ok = (loc[:, :, 2] - rest_z).abs() < c.plate_z_tol
        return xy_ok & z_ok & self.upright()

    def cooked(self) -> torch.Tensor:
        """(N,P) bool: accumulated pad time reached `cook_min` (monotone integral latch)."""
        return self.cook_t >= self.cfg.cook_min

    def settled(self) -> torch.Tensor:
        """(N,P) bool: patty |lin vel| below `settle_speed`."""
        return self._patty_tensors()[2] < self.cfg.settle_speed

    def served(self) -> torch.Tensor:
        """(N,P) bool: present, COOKED, NOT burned, settled flat on the plate."""
        return (self.present & self.cooked() & ~self.burned
                & self.on_plate() & self.settled())

    def all_served(self) -> torch.Tensor:
        """(N,) bool: every PRESENT patty served — judged on the sampled subset."""
        return (self.served() | ~self.present).all(dim=1)

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: per present patty 0.2 once it has rested on the pad (latch),
        0.55 once cooked (latch), 1.0 while served; a burned patty is capped at 0.05 forever.
        Score = 0.9 * mean over present patties, 1.0 iff success. Doing nothing — or
        delivering RAW meat to the plate (the seed's plan) — scores 0."""
        p = torch.zeros_like(self.cook_t)
        p = torch.where(self.ever_on_pad, torch.full_like(p, 0.20), p)
        p = torch.where(self.cooked(), torch.full_like(p, 0.55), p)
        p = torch.where(self.served(), torch.full_like(p, 1.00), p)
        p = torch.where(self.burned, torch.full_like(p, 0.05), p)
        tot = self.present.sum(dim=1).clamp(min=1).float()
        s = 0.9 * (p * self.present.float()).sum(dim=1) / tot
        return torch.where(self.all_served(), torch.ones_like(s), s)

    def success(self) -> torch.Tensor:
        """(N,) bool: every present patty cooked within the window and settled on the plate."""
        return self.all_served()


# Scene-level env binding (robot embodiments are a later stage; solve.py builds franka itself).
register_env("simgen", lambda: EnvCfg(scene="sear_and_serve", robot="null", env_spacing=3))

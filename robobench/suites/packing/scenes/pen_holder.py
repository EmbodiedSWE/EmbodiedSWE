"""PenHolderScene — fill an open pen holder with scattered pens, tip-up (port).

The object world for the RoboDojo "fill-pen-holder" port: an open cup (the pen holder)
standing on the work surface and up to four pens lying flat, scattered on the other side.
**Goal (carried here, no task layer): put every present pen into the holder tip-up, then
leave the holder standing upright on the surface.**

This task is deliberately SOLVABLE — a graded 0-100 rubric so weak agents rank
instead of flatlining — while still honest bimanual manipulation (the source robot
picks the holder up with one hand and fills it with the other; every insertion
targets a compliant, moving cup).

Judged by the ported source rubric with verbatim thresholds, computed in the HOLDER'S
BODY FRAME so a held / tilted holder judges identically to a standing one (the source
fills a held holder): a pen counts when its bottom end is within `xy_tol` (3.5 cm,
source) of the holder axis, its depth below the rim exceeds `depth_min` (3.5 cm, source
depth-into-holder), its axis points tip-up along the holder axis (within
`pen_align_max_deg`), the holder itself is within `holder_tilt_max_deg` (45 deg, source)
of world-up, and pen + holder are settled. Transition scores [10, 25, 40] for 1/2/3 pens,
90 for ALL present pens in the (possibly still held) holder, 100 additionally requires
the loaded holder standing upright ON the work surface (`holder_placed` — the port of the
source's set-down; a top-heavy loaded cup tips easily, which is the real final stage).
NOTE the source's 100-score also requires both grippers >= 80% open and both
end-effectors back at their episode-start pose; those are EMBODIMENT clauses, checked at
the robot-binding/harness layer, deliberately not here (the scene is robot-agnostic —
the stacking-toy return-to-origin precedent).

Assets are vendored product models (one rigid body each, baked by
scripts/vendor_pen_holder_assets.py — measured constants, canonical task frames):
  - holder: a hexagonal cup (textured shell, outer flat-to-flat ~69 mm, corner width
    80 mm, 120 mm tall). The shell is visual-only; physics is an invisible floor plate
    + one wall box per hexagon flat at the measured inner surface (inradius 33.1 mm,
    corner reach 38.2 mm — the xy_tol honesty limit: any pencil physically inside the
    cup counts, so the 3.5 cm tolerance stays honest by construction; four pencils fit
    on the floor). Grasp the ~1.5 mm rim with any jaw, or palm the ~69 mm body.
  - pen: a mechanical pencil (150 mm long, 12 mm dia — the source model is a 211 mm
    drafting pencil, scaled 0.71 at bake to standard pencil size; full-length pencils
    protrude ~95 mm from this cup and pile transients eject/wedge neighbours), click
    button frozen at rest, convexDecomposition collision true to the visual — the
    graphite tip end IS the asset's +z, so the rubric's tip-up clause is honest with
    tip_h = 0.
One family of four identical pencils (the vendored set has one pencil model); identity
is oracle-visible; there is no hidden state anywhere in the task.

Per-episode randomization (task-family knobs): holder pose (xy jitter + yaw), pen
scatter poses (arc slot + xy jitter + free yaw, lying FLAT — every pen must be
reoriented to vertical), AND pen-count subset sampling per family, so success is judged
on the sampled subset and a memorized fixed sequence fails. Absent pens park in an
off-camera ground depot (InteractiveScene cannot despawn).

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from robobench.core.assets import asset_path
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class PenHolderSceneCfg(BaseCfg):
    """Config for `PenHolderScene`. The source thresholds (`xy_tol`, `depth_min`,
    `holder_tilt_max_deg`) are ported verbatim and stay SOFT by design — this is a floor
    task; harden only for curriculum variants."""

    # --- rubric thresholds (source values, verbatim) --------------------------------
    xy_tol: float = 0.035  # pen bottom within this of the holder axis (source 3.5 cm).
    # Honest by construction: max physical in-cup offset = corner reach - pen r
    # = 38.2 - 5.8 mm = 3.2 cm < xy_tol, so any pencil physically inside counts; a pen
    # leaning OUTSIDE the shell is >= 4 cm away.
    depth_min: float = 0.035  # pen bottom below the rim by more than this (source 3.5 cm)
    holder_tilt_max_deg: float = 45.0  # holder axis within this of world-up (source 45)
    pen_align_max_deg: float = 45.0  # pen axis within this of the HOLDER axis, tip-up
    # (port interpretation of the source's tip-below-root clause, in our tip-up convention;
    # geometry already bounds an in-cup pen's lean — this clause rejects tip-DOWN insertions).
    settle_speed: float = 0.05  # max |v| (pen AND holder) when judging (m/s)
    placed_tilt_deg: float = 10.0  # "standing upright" gate for the 100-score set-down
    placed_z_tol: float = 0.010  # holder bottom within this of the surface (m)

    # --- randomization (the task-family knobs) --------------------------------------
    reset_pos_jitter: float = 0.04  # uniform +/- xy jitter (holder AND pens) at reset
    reset_yaw_deg: float = 180.0  # uniform +/- yaw per body at reset (pens lie flat)
    subset_sample: bool = True  # per-episode pen-count sampling (demo sets False)
    min_present: int = 1  # per-family lower bound of sampled pen count

    # --- placement (table-relative xy; the table itself sits at TABLES pos) ---------
    surface_z: float | None = None  # work-surface height (m); None -> the preset's
    holder_pos: tuple = (0.22, 0.0)  # holder centre on the surface (source: right half)
    pens_center: tuple = (-0.10, 0.0)  # scatter-arc centre (source: pens on left half)
    spawn_radii: tuple = (0.18,)  # scatter arc radii (robot cfgs: front arc)
    spawn_arc: tuple = (90.0, 270.0)  # scatter arc (deg) around pens_center

    # --- work surface (the ikea/motherboard/tool_packing vendored-table pattern) ---------
    # Default = the general-purpose packing table (the ikea/microwave bench; the lab
    # table is an industrial GPU-assembly bench and stays available as a preset).
    table: str = "packing"  # which work surface: "lab_table" | "packing"
    table_depth_scale: float = 1.5  # y-stretch: the packing top is 2.47 x 0.76 m,
    # and scatter + holder + an on-table robot base need ~0.9 m of depth (the microwave
    # task's deepening, adopted with it)
    workbench_pos: tuple[float, float] | None = None  # xy the table sits at; None -> preset
    workbench_usd: str = ""  # empty -> the preset's vendored USD
    # Same presets as the sibling scenes, with ONE local default position. The lab
    # table's collision behaves as if rotated 180 deg from its authored orient (MEASURED
    # by a pen ladder: the supported region is x in [pos-1.03, pos+0.25],
    # y in [pos-0.46, pos+0.46] — pens beyond fell 1 m through the visual top; the
    # motherboard/tool_packing layouts happen to fit that footprint either way, which is
    # why it never showed there). (0.40, -0.03) puts the supported region at
    # x [-0.63, 0.65], y [-0.49, 0.43]: the scatter/holder zone, the franka base
    # (0, -0.40) and both multi bases (+/-0.55, 0) all land on real collision.
    TABLES: ClassVar[dict[str, dict[str, Any]]] = {
        "lab_table": {"usd": ("lab_table", "table_instanceable.usd"), "scale": 1.0,
                      "orient": (0.70711, 0.0, 0.0, 0.70711), "surface_z": 0.0,
                      "pos": (0.40, -0.03), "top_offset": 0.0, "height": 1.05,
                      "kinematic": False},
        "packing": {"usd": ("packing_table", "SM_HeavyDutyPackingTable_C02_01_physics.usd"),
                    "scale": 0.01, "orient": (1.0, 0.0, 0.0, 0.0), "surface_z": 0.994,
                    "pos": (0.0, 0.0), "top_offset": 0.994, "height": 0.994,
                    "kinematic": True},
    }

    # --- structure (holder/pencil constants MEASURED at bake time by ---------------------
    # scripts/vendor_pen_holder_assets.py — keep in sync with its printed "scene constants")
    # The vendored hexagonal cup: wall-collider inner inradius (flats) and the corner
    # reach (= inradius / cos 30). Funnel = inner_r - pen_r ~= 27 mm; honesty bound
    # = corner reach - pen_r = 32 mm < the 35 mm xy_tol, so the shell still enforces
    # the tolerance by construction. Four pencils fit on the floor.
    holder_inner_r: float = 0.0331  # hexagon flat inradius (wall-collider inner face)
    holder_corner_r: float = 0.0382  # hexagon corner reach from the axis
    holder_outer_r: float = 0.0400  # outer corner radius (side-lying rest height)
    # Height/lean: a pencil with its bottom at a wall and shaft on the opposite rim leans
    # atan((33+38)/116) ~= 32 deg — inside the 45 deg tip-up cone with margin.
    holder_h: float = 0.120
    floor_local_z: float = -0.0559  # cup floor top, holder body frame (4.1 mm plate)
    holder_mass: float = 0.20
    # Contact offset trades phantom contact against fast-contact capture: the ~25 mm funnel
    # tolerates a generous 5 mm speculative margin, and the margin is what catches a
    # 13 mm/step end-on pen impact before the thin walls could be tunneled.
    contact_offset: float = 0.005
    tip_h: float = 0.0  # the graphite tip IS the asset's +z end (no add-on cone)
    pen_mass: float = 0.012  # a 150 mm mechanical pencil with its mechanism
    # (family name, count, barrel radius, full length) — one family of four identical
    # vendored mechanical pencils (r/l measured at bake time).
    families: tuple = (
        ("pencil", 4, 0.0058, 0.1500),
    )
    # Off-camera ground depot for absent pens; grid extent 1.0 + 2*0.14 + pen 0.21 < half of
    # env_spacing 3 (the stacking-toy depot analysis).
    parking_pos: tuple = (1.0, 1.0)
    # Asset USDs; empty -> the vendored assets committed under `assets/pen_holder/`.
    asset_dir: str = ""
    holder_usd: str = ""
    pencil_usd: str = ""

    # Derived (filled in __post_init__).
    manifest: tuple = field(default=None, init=False)  # ((name, fam_idx, pen_r, barrel_l), ...)

    def __post_init__(self) -> None:
        assets = asset_path(Path(__file__).resolve().parents[1] / "assets")
        self.asset_dir = self.asset_dir or str(assets / "pen_holder")
        self.holder_usd = self.holder_usd or str(Path(self.asset_dir) / "pen_holder.usd")
        self.pencil_usd = self.pencil_usd or str(Path(self.asset_dir) / "pencil.usd")
        preset = self.TABLES[self.table]
        if self.surface_z is None:
            self.surface_z = preset["surface_z"]
        if self.workbench_pos is None:
            self.workbench_pos = preset["pos"]
        self.workbench_usd = self.workbench_usd or str(
            assets.parents[1] / "assembly" / "assets" / "props" / preset["usd"][0] / preset["usd"][1])
        flat = []
        for f, (fname, count, pen_r, barrel_l) in enumerate(self.families):
            for j in range(count):
                flat.append((f"{fname}_{j}", f, pen_r, barrel_l))
        self.manifest = tuple(flat)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("pen_holder")
class PenHolderScene(BaseScene):
    cfg: PenHolderSceneCfg

    def __init__(self, cfg: PenHolderSceneCfg | None = None) -> None:
        super().__init__(cfg or PenHolderSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Floor, light, the vendored work table (ikea/motherboard pattern), the
        free-standing cup, and the pens lying flat at their nominal scatter slots
        (reset() re-places everything). Both vendored USDs get the physics armor at
        spawn (depenetration cap; damping on the pens so a 12 g pencil rattling in the
        cup crosses the 0.05 m/s settle gate promptly instead of ringing)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.surface_z
        for usd in (c.holder_usd, c.pencil_usd):
            if not Path(usd).is_file():
                raise FileNotFoundError(
                    f"{usd} not found — the pen_holder assets ship with the repo under "
                    f"`suites/packing/assets/pen_holder/` (rebake: "
                    f"scripts/vendor_pen_holder_assets.py)"
                )
        preset = c.TABLES[c.table]
        wx, wy = c.workbench_pos
        table_z = z0 - preset["top_offset"]
        ground_z = z0 - preset["height"]
        s = preset["scale"]
        table_spawn = sim_utils.UsdFileCfg(usd_path=c.workbench_usd,
                                           scale=(s, s * c.table_depth_scale, s))
        if preset["kinematic"]:
            table_spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(usd_path=str(
                    Path(c.asset_dir).parents[2] / "assembly" / "assets" / "props"
                    / "ground" / "default_ground.usd")),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, ground_z)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "workbench": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(wx, wy, table_z), rot=preset["orient"]),
                spawn=table_spawn,
            ),
        }
        if c.table == "lab_table":
            # The lab table's collision covers only part of its visual top and sits
            # 180 deg from the VISUAL, which follows the authored +90 (both measured —
            # see the TABLES comment). Everything visual-but-unsupported is a phantom
            # surface objects fall a metre through. Back the ENTIRE visual footprint
            # (x [wx-1.20, wx+0.46], y [wy-0.315, wy+1.104]) with one invisible static
            # slab whose top sits 0.5 mm below the real collision top: shadowed where
            # real collision exists, a safety net where it does not.
            out["table_backing"] = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/TableBacking",
                spawn=sim_utils.CuboidCfg(
                    size=(1.66, 1.42, 0.05),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visible=False,
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(wx - 0.37, wy + 0.395, z0 - 0.0035 - 0.025)),
            )

        out["holder"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Holder",
            spawn=sim_utils.UsdFileCfg(
                usd_path=c.holder_usd,
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=0.5),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.holder_mass),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.holder_pos[0], c.holder_pos[1], z0 + c.holder_h / 2 + 0.002)),
        )

        a0, a1 = (math.radians(v) for v in c.spawn_arc)
        n_pens = len(c.manifest)
        cx, cy = c.pens_center
        for i, (name, _f, pen_r, _barrel_l) in enumerate(c.manifest):
            ang = a0 + (a1 - a0) * (i + 0.5) / n_pens
            r = c.spawn_radii[i % len(c.spawn_radii)]
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pen_" + name,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.pencil_usd,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.003, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        # crowded-cup piles lever the thin shafts hard; iterate the
                        # solver like the other contact-rich suites do
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=1,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05,
                        angular_damping=0.05,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.pen_mass),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(cx + r * math.cos(ang), cy + r * math.sin(ang), z0 + pen_r + 0.003),
                    rot=(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0),  # lying flat
                ),
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
        """Grab handles + allocate the presence mask + per-pen constant tensors. No joints to
        author, no post_step mechanics — the task is pure passive physics."""
        super().bind(env)
        c = self.cfg
        self.holder: RigidObject = env.iscene["holder"]
        self.pens: dict[str, RigidObject] = {
            name: env.iscene[name] for name, _f, _r, _l in c.manifest}
        self.env_origins = env.iscene.env_origins
        # present[e, i]: pen i participates in episode e (sampled at reset; judged subset).
        self.present = torch.ones(env.num_envs, len(c.manifest),
                                  dtype=torch.bool, device=env.device)
        # per-pen constants, manifest order
        self._half_l = torch.tensor([l / 2 for _n, _f, _r, l in c.manifest], device=env.device)
        self._pen_r = torch.tensor([r for _n, _f, r, _l in c.manifest], device=env.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the present pen subset per family, place the holder upright
        with xy jitter + yaw, scatter present pens lying flat on the arc with jitter + free
        yaw, park absent pens in the ground depot."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- subset sampling (the task-family knob): per family, k ~ U{min_present..n} ---
        col0 = 0
        for _fname, count, _r, _l in c.families:
            if c.subset_sample:
                k = torch.randint(c.min_present, count + 1, (m,), device=dev)
            else:
                k = torch.full((m,), count, dtype=torch.long, device=dev)
            rank = torch.rand(m, count, device=dev).argsort(dim=1).argsort(dim=1)
            self.present[env_ids.unsqueeze(1), torch.arange(col0, col0 + count, device=dev)] = (
                rank < k.unsqueeze(1))
            col0 += count

        yaw_amp = math.radians(c.reset_yaw_deg)

        # --- holder: upright at holder_pos + jitter, random yaw ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.holder_pos[0]
        st[:, 1] = c.holder_pos[1]
        st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
        st[:, 2] = c.surface_z + c.holder_h / 2 + 0.002
        half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        st[:, 0:3] += origin
        self.holder.write_root_state_to_sim(st, env_ids)

        # --- pens: arc slot + jitter, lying FLAT with free yaw; absent -> parking depot ---
        # Overlap-rejected placement: a lying pen is a barrel_l-long segment, and tight
        # robot-binding arcs put slots closer than a pen length — free yaws can overlap
        # tip-to-tip at spawn and the depenetration pop hurls a pen off the table
        # (measured: 1 m displacement inside 0.1 s). Place pens sequentially; where a
        # new pen's segment comes too close to an already-placed one, resample its yaw
        # and jitter for that env.
        a0, a1 = (math.radians(v) for v in c.spawn_arc)
        n_pens = len(c.manifest)
        cx, cy = c.pens_center
        c45 = math.cos(math.pi / 4)  # q_pitch = 90 deg about y: pen local +z -> world +x

        def seg_dist(p_c, p_y, p_h, q_c, q_y, q_h):
            """Min distance between 2D segments (centre, yaw, half-length), batched (m,)."""
            su = torch.stack([torch.cos(p_y), torch.sin(p_y)], dim=-1)
            tv = torch.stack([torch.cos(q_y), torch.sin(q_y)], dim=-1)
            best = torch.full_like(p_y, torch.inf)
            for fa in (-1.0, -0.5, 0.0, 0.5, 1.0):  # sampled points on segment A
                pa = p_c + su * (fa * p_h)
                w = pa - q_c
                t = (w * tv).sum(-1).clamp(-q_h, q_h)  # closest point on segment B
                best = torch.minimum(best, (w - tv * t.unsqueeze(-1)).norm(dim=-1))
            for fb in (-1.0, -0.5, 0.0, 0.5, 1.0):  # and the reverse direction
                qb = q_c + tv * (fb * q_h)
                w = qb - p_c
                t = (w * su).sum(-1).clamp(-p_h, p_h)
                best = torch.minimum(best, (w - su * t.unsqueeze(-1)).norm(dim=-1))
            return best

        placed: list[tuple[torch.Tensor, torch.Tensor, float, float]] = []  # (ctr, yaw, half, r)
        for i, (name, _f, pen_r, barrel_l) in enumerate(c.manifest):
            ang = a0 + (a1 - a0) * (i + 0.5) / n_pens
            r = c.spawn_radii[i % len(c.spawn_radii)]
            slot = torch.tensor([cx + r * math.cos(ang), cy + r * math.sin(ang)], device=dev)
            ctr = slot + (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
            yaw = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp
            for _try in range(12):
                bad = torch.zeros(m, dtype=torch.bool, device=dev)
                for (qc, qy, qh, qr) in placed:
                    d = seg_dist(ctr, yaw, barrel_l / 2, qc, qy, qh)
                    bad |= d < (pen_r + qr + 0.006)
                if not bad.any():
                    break
                ctr[bad] = slot + (torch.rand(int(bad.sum()), 2, device=dev) * 2 - 1) * c.reset_pos_jitter
                yaw[bad] = (torch.rand(int(bad.sum()), device=dev) * 2 - 1) * yaw_amp
            placed.append((ctr, yaw, barrel_l / 2, pen_r))

            scat = torch.zeros(m, 3, device=dev)
            scat[:, 0:2] = ctr
            scat[:, 2] = c.surface_z + pen_r + 0.003
            park = torch.zeros(m, 3, device=dev)
            park[:, 0] = c.parking_pos[0] + (i % 2) * 0.14
            park[:, 1] = c.parking_pos[1] + (i // 2) * 0.14
            # depot sits on the GROUND, which lies a full table height below the work
            # surface (parking at pen_r above z=0 left pens airborne on tall presets)
            park[:, 2] = c.surface_z - c.TABLES[c.table]["height"] + pen_r + 0.003

            pres = self.present[env_ids, i].unsqueeze(1)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.where(pres, scat, park)
            # lying flat: q = qz(yaw) * qy(90 deg)  ->  (cy*c45, -sy*c45, cy*c45, sy*c45)
            # with cy=cos(yaw/2), sy=sin(yaw/2)  [qz=(cy,0,0,sy), qy=(c45,0,c45,0) components]
            half = yaw / 2
            st[:, 3] = torch.cos(half) * c45
            st[:, 4] = -torch.sin(half) * c45
            st[:, 5] = torch.cos(half) * c45
            st[:, 6] = torch.sin(half) * c45
            self.pens[name].write_root_state_to_sim(st, env_ids)

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "holder": self.holder.data.root_state_w[env_ids].clone(),
            "pens": {n: b.data.root_state_w[env_ids].clone() for n, b in self.pens.items()},
            "present": self.present[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.holder.write_root_state_to_sim(state["holder"], env_ids)
        for n, b in self.pens.items():
            b.write_root_state_to_sim(state["pens"][n], env_ids)
        self.present[env_ids] = state["present"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        where = "on a work table"
        fams = " and ".join(
            f"up to {count} {name.replace('_', ' ')}{'s' if count > 1 else ''} "
            f"({2 * r * 1000:.0f} mm thick, {l * 1000:.0f} mm long)"
            for name, count, r, l in c.families)
        return (
            f"An open hexagonal pen holder (a cup, ~{2 * c.holder_outer_r * 1000:.0f} mm wide "
            f"across its corners, {c.holder_h * 1000:.0f} mm tall) stands {where}. Scattered on "
            f"its other side lie mechanical pencils, flat on the surface: {fams}, each with a "
            f"pointed writing tip at one end and a click button at the other. Between one and "
            f"all pencils are present in any episode: count what you see.\n"
            f"Goal: put every pencil into the holder tip-up (writing tip pointing out of the "
            f"cup), then leave the loaded holder standing upright on the surface. You may hold "
            f"the holder while filling it; a pencil dropped tip-down or left leaning outside "
            f"the cup does not count, and a tipped-over holder scores nothing until it is "
            f"stood up."
        )

    # ----- progress / rubric ----------------------------------------------------------------------
    def _pen_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(pos_w (N,P,3), quat (N,P,4), |lin_vel| (N,P)) for all pens, manifest order."""
        pos = torch.stack([b.data.root_pos_w for b in self.pens.values()], dim=1)
        quat = torch.stack([b.data.root_quat_w for b in self.pens.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.pens.values()], dim=1)
        return pos, quat, vel

    def _pen_ends_local(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Pen bottom / tip points in the HOLDER'S BODY FRAME, shapes (N, P, 3). The rubric
        lives in this frame so a held, moving, tilted holder judges identically to a standing
        one (the source fills a held holder)."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        pos, quat, _v = self._pen_tensors()
        n, p = pos.shape[0], pos.shape[1]
        ez = torch.tensor([0.0, 0.0, 1.0], device=pos.device).expand(n * p, 3)
        axis = quat_apply(quat.reshape(n * p, 4), ez).reshape(n, p, 3)
        bottom = pos - axis * self._half_l[None, :, None]
        tip = pos + axis * (self._half_l + self.cfg.tip_h)[None, :, None]
        hq = self.holder.data.root_quat_w[:, None, :].expand(n, p, 4).reshape(n * p, 4)
        hp = self.holder.data.root_pos_w[:, None, :]
        b_loc = quat_apply_inverse(hq, (bottom - hp).reshape(n * p, 3)).reshape(n, p, 3)
        t_loc = quat_apply_inverse(hq, (tip - hp).reshape(n * p, 3)).reshape(n, p, 3)
        return b_loc, t_loc

    def holder_up(self) -> torch.Tensor:
        """(N,) bool: holder axis within `holder_tilt_max_deg` of world-up (source 45 deg)."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        up = quat_apply(self.holder.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.holder_tilt_max_deg))

    def inserted(self) -> torch.Tensor:
        """(N, P) bool, geometric: pen bottom within `xy_tol` of the holder axis, deeper than
        `depth_min` below the rim (and above the cup floor), axis tip-up along the holder axis
        — all in the holder frame — with the holder itself `holder_up`."""
        c = self.cfg
        b_loc, t_loc = self._pen_ends_local()
        near_axis = b_loc[:, :, :2].norm(dim=-1) < c.xy_tol
        depth = c.holder_h / 2 - b_loc[:, :, 2]
        deep = (depth > c.depth_min) & (b_loc[:, :, 2] > c.floor_local_z - 0.005)
        axis_loc = t_loc - b_loc
        axis_loc = axis_loc / axis_loc.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        tip_up = axis_loc[:, :, 2] >= math.cos(math.radians(c.pen_align_max_deg))
        return near_axis & deep & tip_up & self.holder_up().unsqueeze(-1)

    def settled(self) -> torch.Tensor:
        """(N, P) bool: pen AND holder |lin vel| below `settle_speed`."""
        _p, _q, vel = self._pen_tensors()
        holder_still = self.holder.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed
        return (vel < self.cfg.settle_speed) & holder_still.unsqueeze(-1)

    def counted(self) -> torch.Tensor:
        """(N, P) bool: inserted, settled AND present — what the rubric counts."""
        return self.inserted() & self.settled() & self.present

    def all_inserted(self) -> torch.Tensor:
        """(N,) bool: every PRESENT pen counts — judged on the sampled subset (90-state)."""
        return (self.counted() | ~self.present).all(dim=1)

    def holder_placed(self) -> torch.Tensor:
        """(N,) bool: the loaded holder set down standing upright ON the work surface —
        within `placed_tilt_deg` of vertical, bottom within `placed_z_tol` of the surface,
        settled. The scene-level port of the source's set-down clause."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        up = quat_apply(self.holder.data.root_quat_w, ez)
        upright = up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(c.placed_tilt_deg))
        bottom_z = (self.holder.data.root_pos_w - self.env_origins)[:, 2] - up[:, 2] * c.holder_h / 2
        on_surface = (bottom_z - c.surface_z).abs() < c.placed_z_tol
        still = self.holder.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return upright & on_surface & still

    def score(self) -> torch.Tensor:
        """(N,) int: the ported transition rubric — [0, 10, 25, 40] for 0/1/2/3 pens counted,
        90 for ALL present pens in the (possibly held) holder, 100 for all-in + the holder
        standing upright on the surface. NOTE the source's 100 additionally requires both
        grippers >= 80% open and both end-effectors back at their start pose; those are
        EMBODIMENT clauses, checked at the robot-binding/harness layer, deliberately not here
        (the scene is robot-agnostic — the stacking-toy return-to-origin precedent)."""
        table = torch.tensor([0, 10, 25, 40], device=self.env.device)
        k = self.counted().sum(dim=1)
        base = table[k.clamp(max=3)]
        all_in = self.all_inserted()
        full = torch.where(self.holder_placed(), 100, 90)
        return torch.where(all_in, full, base)

    def success(self) -> torch.Tensor:
        """(N,) bool: all present pens inserted tip-up + holder standing upright on the
        surface (scene-level success; the NullRobot oracle's target)."""
        return self.all_inserted() & self.holder_placed()

"""WhiteboardWordScene — write a word with a marker, then erase and redo the worst letter.

The object world: a whiteboard standing upright (writing face toward -y), a TRAY under
it holding a chunky MARKER (pinch-sized cylinder) and an ERASER block.
**Goal (carried here, no task layer): write the episode's 3-letter WORD inside the
letter boxes; if a letter comes out badly, erase EXACTLY that letter (neighbours are
a few cm away) and rewrite it. Success = every letter's stroke corridor >= 85% inked
with <= 10% of all ink stray, judged by the simulator's own ink grid.**

INK is simulator-side ground truth (no VLM in the verdict): a 2 mm occupancy grid over
the writing area. A cell is INKED when the marker TIP is pressed within `ink_depth` of
the board plane over that cell (penetration-depth proxy — robust to contact chatter);
cells are CLEARED under the eraser's face when it is pressed flat against the board.
Letters are stroke polylines from a built-in segment font, expanded to corridors of
width `corridor_w`; scoring is set arithmetic per letter (coverage / stray /
worst-letter argmin / selectivity), all deterministic.

The per-episode word is sampled at reset from cfg.words (a letter-box layout makes it a
task family). `describe()` states the word — the challenge is fine motor control, not
reading the goal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg, info, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv

# Stroke font: unit-box polylines (x right, y up), straight segments only.
FONT: dict[str, list[list[tuple[float, float]]]] = {
    "A": [[(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)], [(0.2, 0.4), (0.8, 0.4)]],
    "C": [[(1.0, 0.0), (0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]],
    "E": [[(1.0, 0.0), (0.0, 0.0), (0.0, 1.0), (1.0, 1.0)], [(0.0, 0.5), (0.7, 0.5)]],
    "H": [[(0.0, 0.0), (0.0, 1.0)], [(1.0, 0.0), (1.0, 1.0)], [(0.0, 0.5), (1.0, 0.5)]],
    "I": [[(0.5, 0.0), (0.5, 1.0)]],
    "K": [[(0.0, 0.0), (0.0, 1.0)], [(1.0, 1.0), (0.0, 0.5), (1.0, 0.0)]],
    "L": [[(0.0, 1.0), (0.0, 0.0), (1.0, 0.0)]],
    "N": [[(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]],
    "O": [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]],
    "T": [[(0.0, 1.0), (1.0, 1.0)], [(0.5, 1.0), (0.5, 0.0)]],
    "U": [[(0.0, 1.0), (0.0, 0.1), (0.5, 0.0), (1.0, 0.1), (1.0, 1.0)]],
    "V": [[(0.0, 1.0), (0.5, 0.0), (1.0, 1.0)]],
    "W": [[(0.0, 1.0), (0.25, 0.0), (0.5, 0.6), (0.75, 0.0), (1.0, 1.0)]],
    "X": [[(0.0, 0.0), (1.0, 1.0)], [(0.0, 1.0), (1.0, 0.0)]],
    "Y": [[(0.0, 1.0), (0.5, 0.5), (1.0, 1.0)], [(0.5, 0.5), (0.5, 0.0)]],
    "Z": [[(0.0, 1.0), (1.0, 1.0), (0.0, 0.0), (1.0, 0.0)]],
}

@dataclass
class WhiteboardWordSceneCfg(BaseCfg):
    """Config for `WhiteboardWordScene`."""

    # --- tunable: difficulty dials -----------------------------------------------------------
    ink_depth: float = tunable(0.004)  # tip-to-plane distance that still inks (m)
    # Geometry needs MARGIN on both scores (measured over 2 smoke runs):
    #  - coverage: the tip band (2*tip_r) must overhang the corridor, or a perfect trace
    #    caps below coverage_min (8mm band/16mm corridor -> 0.51; equal 10/10 -> 0.81 on
    #    thin letters from mm-level sag + cell quantization). 12mm band / 10mm corridor
    #    saturates a clean trace.
    #  - stray: ink is only "stray" beyond corridor + stray_margin (nobody calls 1mm of
    #    edge bleed scribbling); the overhang of a clean trace lands inside the margin,
    #    while the negative-control scribble (cm from any stroke) stays fully stray.
    tip_r: float = tunable(0.006)  # ink dot radius (m)
    corridor_w: float = tunable(0.010)  # full stroke-corridor width (m)
    stray_margin: float = tunable(0.004)  # bleed allowance beyond the corridor edge (m)
    coverage_min: float = tunable(0.85)
    stray_max: float = tunable(0.10)
    erase_gap: float = tunable(0.006)  # eraser-face-to-plane distance that clears (m)

    # --- tunable: placement -------------------------------------------------------------------
    surface_z: float = tunable(0.0)
    board_y: float = tunable(0.24)  # writing plane y

    # --- tunable: randomization ----------------------------------------------------------------
    # Per-episode word pool (was a module constant; scene init parameters belong in the
    # scene cfg — a curriculum variant can now swap the pool per binding).
    words: tuple = tunable(("CAT", "HEX", "VAN", "WIN", "YAK", "ZOO", "NUT", "OIL", "LAW", "KIT"))

    # --- info: structure ----------------------------------------------------------------------
    bench_size: tuple = info((1.2, 0.9))
    board_size: tuple = info((0.7, 0.03, 0.5))
    cell: float = info(0.002)
    area: tuple = info((0.42, 0.22))  # writing area (u along x, v up)
    area_v0: float = info(0.16)  # writing-area bottom above the surface
    letter_box: tuple = info((0.08, 0.12))  # letter size in the slot
    letter_pitch: float = info(0.13)  # slot-to-slot spacing (>= 4cm gaps)
    n_letters: int = info(3)
    marker_r: float = info(0.0125)  # chunky marker per the brief
    marker_l: float = info(0.12)
    eraser_size: tuple = info((0.10, 0.06, 0.04))
    tray_y: float = info(0.10)  # tray front offset from the board plane

    grid_n: tuple = field(default=None, init=False)  # (nu, nv)

    def __post_init__(self) -> None:
        self.grid_n = (int(round(self.area[0] / self.cell)),
                       int(round(self.area[1] / self.cell)))


@SCENES.register("whiteboard")
class WhiteboardWordScene(BaseScene):
    cfg: WhiteboardWordSceneCfg

    def __init__(self, cfg: WhiteboardWordSceneCfg | None = None) -> None:
        super().__init__(cfg or WhiteboardWordSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.surface_z
        by = c.board_y
        bw, bt, bh = c.board_size

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
        }
        if z0 > 0:
            out["bench"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=sim_utils.CuboidCfg(
                    size=(c.bench_size[0], c.bench_size[1], z0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, z0 / 2)),
            )

        out["board"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Board",
            spawn=sim_utils.CuboidCfg(
                size=(bw, bt, bh),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.97)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, by + bt / 2, z0 + bh / 2 + 0.08)),
        )
        # Tray: a kinematic shelf in front of/below the board.
        out["tray"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Tray",
            spawn=sim_utils.CuboidCfg(
                size=(bw, c.tray_y, 0.012),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.55)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, by - c.tray_y / 2, z0 + 0.075)),
        )
        out["marker"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Marker",
            spawn=sim_utils.CylinderCfg(
                radius=c.marker_r, height=c.marker_l, axis="X",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.2, 0.2)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(-0.15, by - c.tray_y / 2, z0 + 0.081 + c.marker_r)),
        )
        out["eraser"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Eraser",
            spawn=sim_utils.CuboidCfg(
                size=c.eraser_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.3, 0.7)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.15, by - c.tray_y / 2, z0 + 0.081 + c.eraser_size[2] / 2)),
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
        n = env.num_envs
        dev = env.device
        self.marker: RigidObject = env.iscene["marker"]
        self.eraser: RigidObject = env.iscene["eraser"]
        self.board: RigidObject = env.iscene["board"]
        self.env_origins = env.iscene.env_origins
        nu, nv = c.grid_n
        self.ink = torch.zeros(n, nu, nv, dtype=torch.bool, device=dev)
        self._words: list[str] = ["CAT"] * n
        # corridor masks: (n, n_letters, nu, nv) + union
        self._corr = torch.zeros(n, c.n_letters, nu, nv, dtype=torch.bool, device=dev)
        # dilated corridor union (corridor + stray_margin): ink inside it is never stray
        self._corr_dil = torch.zeros(n, nu, nv, dtype=torch.bool, device=dev)
        # cell centres (u, v) for fast lookup
        us = (torch.arange(nu, device=dev) + 0.5) * c.cell - c.area[0] / 2
        vs = (torch.arange(nv, device=dev) + 0.5) * c.cell
        self._uu, self._vv = torch.meshgrid(us, vs, indexing="ij")
        self._ink_dirty = torch.zeros(n, dtype=torch.bool, device=dev)
        self._viz_step = 0
        self._build_ink_viz()

    def _build_ink_viz(self) -> None:
        """Per-env UsdGeomPointInstancer that shows the ink grid as dark dots hovering
        just off the board face — purely visual (no physics), updated in post_step so
        the writing is actually VISIBLE in renders and look() images."""
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        self._viz = []
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            # Idempotency guard (multi-env boot, 2026-07-17): env_1.. inherit env_0's
            # subtree, so the proto (and its scale op) may already compose here —
            # AddXformOp on it would hard-fail. Existence check MUST precede Define.
            proto_fresh = not stage.GetPrimAtPath(f"{base}/InkViz/proto").IsValid()
            proto = UsdGeom.Cube.Define(stage, f"{base}/InkViz/proto")
            proto.CreateSizeAttr(1.0)
            if proto_fresh:
                xf = UsdGeom.Xformable(proto)
                xf.AddScaleOp().Set(Gf.Vec3f(c.cell * 0.95, 0.0015, c.cell * 0.95))
            proto.CreateDisplayColorAttr([Gf.Vec3f(0.08, 0.08, 0.35)])
            inst = UsdGeom.PointInstancer.Define(stage, f"{base}/InkViz/points")
            inst.CreatePrototypesRel().SetTargets([Sdf.Path(f"{base}/InkViz/proto")])
            inst.CreatePositionsAttr([])
            inst.CreateProtoIndicesAttr([])
            self._viz.append(inst)

    def _update_ink_viz(self, e: int) -> None:
        from pxr import Gf, Vt

        c = self.cfg
        idx = self.ink[e].nonzero(as_tuple=False)
        if len(idx) > 0:
            us = ((idx[:, 0].float() + 0.5) * c.cell - c.area[0] / 2)
            vs = ((idx[:, 1].float() + 0.5) * c.cell)
            o = self.env_origins[e]
            xs = (us + o[0]).tolist()
            ys = float(o[1] + c.board_y - 0.0015)
            zs = (vs + o[2] + c.surface_z + c.area_v0).tolist()
            pos = Vt.Vec3fArray([Gf.Vec3f(x, ys, z) for x, z in zip(xs, zs)])
        else:
            pos = Vt.Vec3fArray([])
        self._viz[e].GetPositionsAttr().Set(pos)
        self._viz[e].GetProtoIndicesAttr().Set(Vt.IntArray([0] * len(pos)))

    # ----- letters ------------------------------------------------------------------------------
    def letter_origin(self, slot: int) -> tuple[float, float]:
        """Board coords (u, v) of letter slot's lower-left corner."""
        c = self.cfg
        u0 = -(c.n_letters - 1) / 2 * c.letter_pitch - c.letter_box[0] / 2
        return (u0 + slot * c.letter_pitch, 0.04)

    def letter_strokes(self, slot: int, letter: str) -> list[list[tuple[float, float]]]:
        """Stroke polylines for `letter` in `slot`, in board (u, v) metres."""
        c = self.cfg
        ox, oy = self.letter_origin(slot)
        w, h = c.letter_box
        return [[(ox + px * w, oy + py * h) for px, py in seg] for seg in FONT[letter]]

    def _rasterize_corridors(self, e: int) -> None:
        c = self.cfg
        dev = self.env.device
        dil = torch.zeros_like(self._corr_dil[e])
        r_dil = c.corridor_w / 2 + c.stray_margin
        for slot, letter in enumerate(self._words[e]):
            mask = torch.zeros_like(self._corr[e, slot])
            for seg in self.letter_strokes(slot, letter):
                for (x1, y1), (x2, y2) in zip(seg[:-1], seg[1:]):
                    seg_len = max(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, 1e-9)
                    m = int(seg_len / 0.001) + 1
                    ts = torch.linspace(0, 1, m, device=dev)
                    px = x1 + (x2 - x1) * ts
                    py = y1 + (y2 - y1) * ts
                    d2 = ((self._uu.unsqueeze(-1) - px) ** 2
                          + (self._vv.unsqueeze(-1) - py) ** 2).min(dim=-1).values
                    mask |= d2 <= (c.corridor_w / 2) ** 2
                    dil |= d2 <= r_dil ** 2
            self._corr[e, slot] = mask
        self._corr_dil[e] = dil

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        z0 = c.surface_z
        by = c.board_y

        def put(body: RigidObject, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = self.env_origins[env_ids] + torch.tensor(pos, device=dev)
            st[:, 3:7] = torch.tensor(quat, device=dev)
            body.write_root_state_to_sim(st, env_ids)

        put(self.marker, (-0.15, by - c.tray_y / 2, z0 + 0.081 + c.marker_r))
        put(self.eraser, (0.15, by - c.tray_y / 2, z0 + 0.081 + c.eraser_size[2] / 2))
        self.ink[env_ids] = False
        pool = self.cfg.words
        for e in env_ids.tolist():
            w = pool[int(torch.randint(len(pool), (1,)))]
            self._words[e] = w
            self._rasterize_corridors(e)
            self._update_ink_viz(e)

    # ----- ink mechanics (every substep) ----------------------------------------------------------
    def tip_pos(self) -> torch.Tensor:
        """Marker tip (its +x end) in world."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        ex = torch.tensor([c.marker_l / 2, 0.0, 0.0], device=self.env.device).expand(n, 3)
        return self.marker.data.root_pos_w + quat_apply(self.marker.data.root_quat_w, ex)

    def _board_uv(self, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """World points -> (u, v, plane distance). u along x, v up from area bottom."""
        c = self.cfg
        rel = p - self.env_origins
        u = rel[:, 0]
        v = rel[:, 2] - (c.surface_z + c.area_v0)
        dist = c.board_y - rel[:, 1]  # >0 in front of the plane
        return u, v, dist

    def post_step(self) -> None:
        c = self.cfg
        n = self.env.num_envs

        # Marker inking.
        u, v, dist = self._board_uv(self.tip_pos())
        writing = (dist.abs() <= c.ink_depth)
        if bool(writing.any()):
            for e in writing.nonzero(as_tuple=True)[0].tolist():
                du = self._uu - u[e]
                dv = self._vv - v[e]
                self.ink[e] |= (du * du + dv * dv) <= c.tip_r ** 2
            self._ink_dirty = self._ink_dirty | writing

        # Eraser clearing: face-on (its +y face toward the board plane) and pressed.
        from isaaclab.utils.math import quat_apply

        ey = torch.tensor([0.0, 1.0, 0.0], device=self.env.device).expand(n, 3)
        face_n = quat_apply(self.eraser.data.root_quat_w, ey)
        face_c = self.eraser.data.root_pos_w + face_n * (c.eraser_size[1] / 2)
        ue, ve, de = self._board_uv(face_c)
        flat = face_n[:, 1] > 0.9  # facing the board
        erasing = flat & (de.abs() <= c.erase_gap)
        if bool(erasing.any()):
            hx, hz = c.eraser_size[0] / 2, c.eraser_size[2] / 2
            for e in erasing.nonzero(as_tuple=True)[0].tolist():
                box = ((self._uu - ue[e]).abs() <= hx) & ((self._vv - ve[e]).abs() <= hz)
                self.ink[e] &= ~box
            self._ink_dirty = self._ink_dirty | erasing

        # Ink visualization refresh (throttled: USD writes are not free).
        self._viz_step += 1
        if self._viz_step % 10 == 0 and bool(self._ink_dirty.any()):
            for e in self._ink_dirty.nonzero(as_tuple=True)[0].tolist():
                self._update_ink_viz(e)
            self._ink_dirty[:] = False

    # ----- state ---------------------------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in (("marker", self.marker), ("eraser", self.eraser))},
            "ink": self.ink[env_ids].clone(),
            "words": [self._words[int(e)] for e in env_ids],
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm, b in (("marker", self.marker), ("eraser", self.eraser)):
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        self.ink[env_ids] = state["ink"]
        self._ink_dirty[env_ids] = True
        for row, e in enumerate(env_ids.tolist()):
            if self._words[e] != state["words"][row]:
                self._words[e] = state["words"][row]
                self._rasterize_corridors(e)

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        slots = ", ".join(
            f"slot {i} at u={self.letter_origin(i)[0]:.2f}" for i in range(c.n_letters))
        return (
            f"A whiteboard stands upright (writing plane at y={c.board_y:.2f}, writing "
            f"area {c.area[0]:.2f} x {c.area[1]:.2f} m starting {c.area_v0:.2f} m above "
            f"the surface). On the tray: a chunky red MARKER (grab it and press its TIP "
            f"within {c.ink_depth * 1000:.0f} mm of the plane to ink) and a blue ERASER "
            f"(press its big face flat on the board to wipe everything under it).\n"
            f"THE WORD TO WRITE: '{self._words[0]}'. Letters go in {c.n_letters} boxes "
            f"({c.letter_box[0]:.2f} x {c.letter_box[1]:.2f} m; {slots}; box bottoms "
            f"{self.letter_origin(0)[1]:.2f} m above the area bottom). Trace each "
            f"letter's strokes; a letter scores by how much of its stroke corridor you "
            f"inked (need >= {c.coverage_min:.0%}) and ink outside all corridors counts "
            f"as stray (total stray must stay <= {c.stray_max:.0%}). You can query your "
            f"letter scores, erase your worst letter WITHOUT touching its neighbours "
            f"(they are only ~4 cm away), and rewrite it."
        )

    # ----- progress -------------------------------------------------------------------------------
    def word(self) -> str:
        return self._words[0]

    def letter_scores(self) -> torch.Tensor:
        """(num_envs, n_letters) corridor coverage in [0, 1]."""
        inked = self.ink.unsqueeze(1) & self._corr
        return (inked.sum(dim=(2, 3)).float()
                / self._corr.sum(dim=(2, 3)).clamp(min=1).float())

    def stray_frac(self) -> torch.Tensor:
        """Fraction of all inked cells lying beyond every corridor + stray_margin."""
        total = self.ink.sum(dim=(1, 2)).float()
        stray = (self.ink & ~self._corr_dil).sum(dim=(1, 2)).float()
        return torch.where(total > 0, stray / total.clamp(min=1), torch.zeros_like(total))

    def worst_letter(self) -> torch.Tensor:
        return self.letter_scores().argmin(dim=1)

    def ink_image(self) -> torch.Tensor:
        """(num_envs, nu, nv) bool — the raw ink grid (for look()-style rendering)."""
        return self.ink.clone()

    def success(self) -> torch.Tensor:
        c = self.cfg
        cov_ok = (self.letter_scores() >= c.coverage_min).all(dim=1)
        return cov_ok & (self.stray_frac() <= c.stray_max)


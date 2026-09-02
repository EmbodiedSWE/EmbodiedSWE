"""DiceFoodScene — grid dicing: the slice scene's world (board, knife rest, arm stand, edge-level
knife) on a food baked as an N x M grid (x-planes and y-planes), default tomato 3 x 3.
Env name: cutting.dice (`food="potato"` for the potato).

The dice scene owns its own cut mechanics and leaves `SliceFoodScene` untouched:

- Planes come in two families; a y-plane needs the blade yawed 90 deg.
- Every grid-neighbour pair is one weld (FixedJoint), gated ON ITS OWN, live: the pair's cut
  point and plane normal are carried rigidly by its two pieces (their bake-frame offsets
  rotated by their live orientations). Once earlier cuts have let columns drift apart, a pair
  releases only when the blade is between THOSE two pieces — a plane-level gate would free
  untouched pairs across the whole row.
- `pair_cut` (n, P) is the truth; `cut` (n, planes) is the plane-level view drivers plan
  against (all of a plane's pairs released). `plane_aim_w` / `plane_axis_w` give drivers the
  live aim point and blade yaw for a plane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from robobench.core import SCENES, BaseScene

from .slice_food import SliceFoodScene, SliceFoodSceneCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


@dataclass
class DiceFoodSceneCfg(SliceFoodSceneCfg):
    food: str = "tomato"
    # round whole foods are baked with a flat resting patch (bake --flat-bottom) so they do not
    # roll under an off-centre press (a 3 x 3 has no centre plane); a little extra damping
    # keeps the freed cubes from rolling off
    piece_linear_damping: float = 0.3
    piece_angular_damping: float = 1.0

    FOOD_PRESETS: ClassVar[dict[str, dict]] = {
        "tomato": {"depth_past_center": 0.0},  # release as the edge reaches the flesh centre
        "potato": {"depth_past_center": 0.0},
    }


@SCENES.register("dice")
class DiceFoodScene(SliceFoodScene):
    """Grid dicing (tomato / potato, 3 x 3). Env name: cutting.dice"""

    cfg: DiceFoodSceneCfg

    def __init__(self, cfg: DiceFoodSceneCfg | None = None) -> None:
        super().__init__(cfg or DiceFoodSceneCfg())
        # planes: ("x", value) then ("y", value); cells (i, j) index the grid
        self.planes = [("x", p) for p in self.manifest["planes_x"]] + \
                      [("y", p) for p in self.manifest["planes_y"]]
        self.cells = [tuple(m["cell"]) for m in self.manifest["pieces"] if m is not None]

    def _plane_index(self, axis: str, k: int) -> int:
        """Index into `self.planes` of the k-th plane along `axis`."""
        return [i for i, (a, _) in enumerate(self.planes) if a == axis][k]

    # ----- assets: the slice world, pieces with the dice damping -------------------------------
    def assets(self) -> dict[str, Any]:
        out = super().assets()
        c = self.cfg
        for k, meta in enumerate(self.manifest["pieces"]):
            if meta is not None:
                rp = out[f"piece_{k}"].spawn.rigid_props
                rp.linear_damping = c.piece_linear_damping
                rp.angular_damping = c.piece_angular_damping
        return out

    # ----- lifecycle ------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        BaseScene.bind(self, env)  # not the slice bind: it welds a chain
        man = self.manifest
        self.pieces: list[RigidObject] = []
        self._cents: list[tuple[float, float, float]] = []
        for k, meta in enumerate(man["pieces"]):
            if meta is None:
                continue
            self.pieces.append(env.iscene[f"piece_{k}"])
            self._cents.append(tuple(meta["centroid"]))
        self.knife: RigidObject = env.iscene["knife"]
        self.env_origins = env.iscene.env_origins
        self._board_top = self.cfg.surface_z
        self._edge_local = torch.tensor(self.knife_edge(), device=env.device)  # (N, 3) knife frame
        # WELD PAIRS: (a, b, plane_idx) for every grid-neighbour pair, plus each pair's cut
        # point as a rigid offset from EACH of its pieces (bake frame) and its plane normal.
        # Centroids are vertex means, not geometric centres, so a->b is not the normal.
        idx_of = {cell: i for i, cell in enumerate(self.cells)}
        self.pairs: list[tuple[int, int, int]] = []
        off_a, off_b, normals = [], [], []
        for i, (ci, cj) in enumerate(self.cells):
            for di, dj, axis in ((1, 0, "x"), (0, 1, "y")):
                nb = (ci + di, cj + dj)
                if nb not in idx_of:
                    continue
                ax = 0 if axis == "x" else 1
                plane_idx = self._plane_index(axis, ci if axis == "x" else cj)
                p = self.planes[plane_idx][1]
                ca, cb = self._cents[i], self._cents[idx_of[nb]]
                cut_pt = [(ca[d] + cb[d]) / 2 for d in range(3)]
                cut_pt[ax] = p
                self.pairs.append((i, idx_of[nb], plane_idx))
                off_a.append([cut_pt[d] - ca[d] for d in range(3)])
                off_b.append([cut_pt[d] - cb[d] for d in range(3)])
                normals.append([1.0, 0.0, 0.0] if axis == "x" else [0.0, 1.0, 0.0])
        self._plane_pairs = [[k for k, pr in enumerate(self.pairs) if pr[2] == idx]
                             for idx in range(len(self.planes))]
        dev = env.device
        self._pair_a = torch.tensor([pr[0] for pr in self.pairs], device=dev)
        self._pair_b = torch.tensor([pr[1] for pr in self.pairs], device=dev)
        self._pair_off_a = torch.tensor(off_a, device=dev)  # (P, 3) bake frame
        self._pair_off_b = torch.tensor(off_b, device=dev)
        self._pair_n = torch.tensor(normals, device=dev)  # (P, 3) bake frame
        self.pair_cut = torch.zeros(env.num_envs, len(self.pairs), dtype=torch.bool, device=dev)
        self.cut = torch.zeros(env.num_envs, len(self.planes), dtype=torch.bool, device=dev)
        self._precreate_weld_joints()

    # ----- live cut geometry (world frame) -----------------------------------------------------
    def _pair_geometry(self):
        """Per weld pair, LIVE: its cut point (n, P, 3) and cut-plane normal (n, P, 3), each
        carried rigidly by the pair's two pieces and averaged (identical while still welded)."""
        from isaaclab.utils.math import quat_apply

        n, P = self.env.num_envs, len(self.pairs)
        pos = torch.stack([pc.data.root_pos_w for pc in self.pieces], dim=1)  # (n, K, 3)
        quat = torch.stack([pc.data.root_quat_w for pc in self.pieces], dim=1)  # (n, K, 4)
        qa = quat[:, self._pair_a].reshape(-1, 4)
        qb = quat[:, self._pair_b].reshape(-1, 4)
        oa = self._pair_off_a.unsqueeze(0).expand(n, P, 3).reshape(-1, 3)
        ob = self._pair_off_b.unsqueeze(0).expand(n, P, 3).reshape(-1, 3)
        nn = self._pair_n.unsqueeze(0).expand(n, P, 3).reshape(-1, 3)
        pt_a = pos[:, self._pair_a] + quat_apply(qa, oa).reshape(n, P, 3)
        pt_b = pos[:, self._pair_b] + quat_apply(qb, ob).reshape(n, P, 3)
        axis = (quat_apply(qa, nn) + quat_apply(qb, nn)).reshape(n, P, 3)
        return 0.5 * (pt_a + pt_b), axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def plane_aim_w(self, idx: int) -> torch.Tensor:
        """(n, 3) world: where to aim the blade for plane `idx` — the mean of its pairs' live
        cut points (for a still-welded row this is the scored plane's flesh centre)."""
        aim, _ = self._pair_geometry()
        return aim[:, self._plane_pairs[idx]].mean(dim=1)

    def plane_axis_w(self, idx: int) -> torch.Tensor:
        """(n, 3) world: the plane normal for plane `idx` (mean of its pairs' live normals)."""
        _, axis = self._pair_geometry()
        m = axis[:, self._plane_pairs[idx]].mean(dim=1)
        return m / m.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    # ----- state ----------------------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "pieces": torch.stack([p.data.root_state_w[env_ids].clone() for p in self.pieces], dim=1),
            "knife": self.knife.data.root_state_w[env_ids].clone(),
            "cut": self.pair_cut[env_ids].clone(),  # per weld pair
        }

    def describe(self) -> str:
        c = self.cfg
        nx = sum(1 for a, _ in self.planes if a == "x")
        return (
            f"A {c.food} lies on a chopping board on a kitchen island; a chef knife lies on a "
            f"knife rest beside the board. The {c.food} is scored on a grid: {nx} planes across "
            f"and {len(self.planes) - nx} along it. Goal: pick up the knife and press its edge "
            f"down through each scored plane (blade aligned with the plane — turn the blade 90 "
            f"deg for the second set) to dice the {c.food}. It is diced once every plane has "
            f"been pressed through and the pieces separate, all of them staying on the board."
        )

    def pieces_count(self) -> torch.Tensor:
        """(N,) long: connected components of the live weld graph per env."""
        counts = torch.zeros(self.env.num_envs, dtype=torch.long)
        pair_cut = self.pair_cut.cpu()
        for e in range(self.env.num_envs):
            parent = list(range(len(self.cells)))

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            for k, (a, b, _) in enumerate(self.pairs):
                if not bool(pair_cut[e, k]):
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[ra] = rb
            counts[e] = len({find(i) for i in range(len(self.cells))})
        return counts.to(self.env.device)

    # ----- the cut gate (private; the weld-release sim-hack, not an agent action) --------------
    def _cut_targets(self) -> torch.Tensor:
        """THE STANDARD — which weld PAIRS should be released now (sticky, monotone), each
        measured against its own two pieces, live: the knife's REAL edge, at the sample nearest
        the pair's cut point, must be within `plane_tol` of the pair's cut plane, within
        `flesh_radius` of the cut point along the blade, with the blade normal aligned to the
        pair's plane normal, pressed below the release depth (the cut point minus
        `depth_past_center`, or bottomed on the board), both pieces settled, and the knife
        pressing (not rising)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        n = self.env.num_envs
        kp, kq = self.knife.data.root_pos_w, self.knife.data.root_quat_w
        N = self._edge_local.shape[0]
        kq_rep = kq.unsqueeze(1).expand(n, N, 4).reshape(-1, 4)
        el_rep = self._edge_local.unsqueeze(0).expand(n, N, 3).reshape(-1, 3)
        edge_pts_w = kp.unsqueeze(1) + quat_apply(kq_rep, el_rep).reshape(n, N, 3)
        bn_w = quat_apply(kq, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3))  # blade normal

        aim, axis = self._pair_geometry()  # (n, P, 3) each
        dh = (edge_pts_w[:, None, :, :2] - aim[:, :, None, :2]).norm(dim=-1)  # (n, P, N)
        j = dh.argmin(dim=-1)
        ep = torch.gather(edge_pts_w, 1, j.unsqueeze(-1).expand(-1, -1, 3))  # (n, P, 3)
        rel = ep - aim
        perp = torch.stack([-axis[..., 1], axis[..., 0], torch.zeros_like(axis[..., 0])], dim=-1)
        near = (rel * axis).sum(-1).abs() < c.plane_tol
        at_flesh = (rel * perp).sum(-1).abs() < c.flesh_radius
        aligned = (bn_w.unsqueeze(1) * axis).sum(-1).abs() > math.cos(math.radians(c.blade_align_deg))
        depth_thresh = torch.maximum(aim[..., 2] - c.depth_past_center,
                                     torch.full_like(aim[..., 2], self._board_top + 0.0015))
        pressed = ep[..., 2] < depth_thresh
        speed = torch.stack([pc.data.root_lin_vel_w.norm(dim=-1) for pc in self.pieces], dim=1)
        settled = (speed[:, self._pair_a] < c.food_settle_speed) & \
                  (speed[:, self._pair_b] < c.food_settle_speed)
        # release only while the blade PRESSES, never while it rises out of the kerf
        pressing = (self.knife.data.root_lin_vel_w[:, 2] < c.press_vz_max).unsqueeze(1)
        return self.pair_cut | (near & at_flesh & aligned & pressed & settled & pressing)

    def _reconcile_cuts(self, env_ids, target) -> None:
        """Apply a release target — per pair (n_ids, P), or per plane (n_ids, planes) as the
        slice reset hands over — and refresh the plane-level view."""
        if target.shape[1] == len(self.planes) and len(self.planes) != len(self.pairs):
            expanded = torch.zeros(target.shape[0], len(self.pairs), dtype=torch.bool, device=target.device)
            for k, (_, _, plane_idx) in enumerate(self.pairs):
                expanded[:, k] = target[:, plane_idx]
            target = expanded
        have = self.pair_cut[env_ids]
        for row, k in (target & ~have).nonzero(as_tuple=False).tolist():
            self._set_pair(int(env_ids[row]), k, enabled=False)
        for row, k in (have & ~target).nonzero(as_tuple=False).tolist():
            self._set_pair(int(env_ids[row]), k, enabled=True)
        for idx, ks in enumerate(self._plane_pairs):
            self.cut[env_ids, idx] = self.pair_cut[env_ids][:, ks].all(dim=1)

    def _precreate_weld_joints(self) -> None:
        import omni.usd
        from pxr import Gf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        b = self.manifest["bounds"]
        prim_idx = [k for k, meta in enumerate(self.manifest["pieces"]) if meta is not None]
        self._joint_paths: list[list[str]] = []  # [env][pair] -> joint path
        for e in range(self.env.num_envs):
            base = f"/World/envs/env_{e}"
            per_pair: list[str] = []
            for a, bb, plane_idx in self.pairs:
                axis, p = self.planes[plane_idx]
                jp = f"{base}/cutweld_{axis}_{self.cells[a][0]}_{self.cells[a][1]}"
                jt = UsdPhysics.FixedJoint.Define(stage, jp)
                jt.CreateBody0Rel().SetTargets([f"{base}/Piece_{prim_idx[a]}"])
                jt.CreateBody1Rel().SetTargets([f"{base}/Piece_{prim_idx[bb]}"])
                ca, cb = self._cents[a], self._cents[bb]
                mid = [(ca[0] + cb[0]) / 2, (ca[1] + cb[1]) / 2, (b[0][2] + b[1][2]) / 2]
                mid[0 if axis == "x" else 1] = p
                jt.CreateLocalPos0Attr(Gf.Vec3f(*[mid[d] - ca[d] for d in range(3)]))
                jt.CreateLocalPos1Attr(Gf.Vec3f(*[mid[d] - cb[d] for d in range(3)]))
                jt.CreateJointEnabledAttr(True)
                per_pair.append(jp)
            self._joint_paths.append(per_pair)

    def _set_pair(self, env_i: int, pair_idx: int, *, enabled: bool) -> None:
        from pxr import UsdPhysics

        UsdPhysics.FixedJoint.Get(
            self.env.stage, self._joint_paths[env_i][pair_idx]).GetJointEnabledAttr().Set(enabled)
        self.pair_cut[env_i, pair_idx] = not enabled

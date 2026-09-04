"""A splat model = ONE object (GSWorld's unit) as a PLY + its ``_poses.json``:

  - ``<name>.ply``        the scan in the metric sim frame; optional per-gaussian ``semantics``
                          property = link index into ``link_names`` (absent -> one link)
  - ``<name>_poses.json`` where every sim body the model rides on WAS at scan time:
                          ``body_names``, ``pos (B,3)``, ``quat (B,4) wxyz``, and ``link_names``
                          (label index -> body name), plus ``qpos``/``source`` notes. Not labels —
                          those are in the PLY.

A robot is a model with many links (the articulation's bodies at the scan qpos); a table, a cup,
a room piece is a model with one link (its rigid body's pose at scan time). At load, each link's
gaussians are expressed in that link's frame (``inv(T_link_scan)``); at runtime they are re-posed
from the live sim: ``world = T_link_now @ local`` — GSWorld's
``sim2gs @ link_now @ inv(link_scan) @ inv(sim2gs)`` with the alignment already applied. No URDF /
FK is needed: link poses come straight from the sim, so every model is manipulable the same way.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from gsworld.splat import GaussianSet, read_ply_property


def load_scan_poses(path: str | Path) -> dict:
    """Read ``<name>_poses.json``."""
    import json

    d = json.loads(Path(path).read_text())
    get = lambda k, default=None: d.get(k, default)  # noqa: E731
    out = {"body_names": [str(s) for s in get("body_names")],
           "pos": np.asarray(get("pos"), dtype=np.float64), "quat": np.asarray(get("quat"), dtype=np.float64)}
    ln = get("link_names")
    out["link_names"] = [str(s) for s in ln] if ln is not None else out["body_names"]
    q = get("qpos")
    out["qpos"] = None if q is None or len(q) == 0 else np.asarray(q, dtype=np.float64)
    return out


def save_scan_poses(path: str | Path, body_names, pos, quat, link_names, qpos=None, source: str = "") -> None:
    """Write ``<name>_poses.json`` (readable, diffable)."""
    import json

    Path(path).write_text(json.dumps({
        "body_names": [str(s) for s in body_names], "link_names": [str(s) for s in link_names],
        "pos": np.asarray(pos, dtype=float).round(6).tolist(), "quat": np.asarray(quat, dtype=float).round(6).tolist(),
        "qpos": [] if qpos is None else np.asarray(qpos, dtype=float).round(6).tolist(), "source": source,
    }, indent=1))


class SplatModel:
    def __init__(self, ply: str | Path, scan_poses: str | Path | dict, device: str = "cuda") -> None:
        import torch
        from scipy.spatial.transform import Rotation

        poses = scan_poses if isinstance(scan_poses, dict) else load_scan_poses(scan_poses)
        g = GaussianSet.from_ply(ply)
        try:
            labels = np.rint(read_ply_property(ply, "semantics")).astype(np.int64)
        except KeyError:  # unlabeled object: everything rides the model's single body
            labels = np.zeros(g.n, dtype=np.int64)
        parts, idx, kept = [], [], []
        for i, ln in enumerate(poses["link_names"]):
            m = labels == i
            if not m.any():
                continue
            b = poses["body_names"].index(ln)
            T = np.eye(4)
            q = poses["quat"][b]
            T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            T[:3, 3] = poses["pos"][b]
            parts.append(g.select(m).transform(np.linalg.inv(T)))  # scan frame -> link-local
            idx.append(np.full(int(m.sum()), len(kept), dtype=np.int64))
            kept.append(ln)
        self.local = GaussianSet.concat(parts)
        self.link_idx = np.concatenate(idx) if idx else np.zeros(0, np.int64)
        self.link_names: list[str] = kept
        self.scan_qpos = poses["qpos"]
        self.device = device
        self._t = self.local.to_torch(device)
        self._link_idx_t = torch.as_tensor(self.link_idx, device=device)
        self._body_order: list[int] | None = None

    # ----------------------------------------------------------------- binding to a sim articulation
    def bind_bodies(self, body_names: list[str], strict: bool = True) -> list[str]:
        """Map the model's links onto the sim's body ordering (an articulation's ``body_names``, or
        ``[name]`` for a rigid object). With ``strict`` a missing link raises; otherwise links the sim
        lacks are dropped (partial model, e.g. a Franka+Robotiq scan on a Panda-hand Franka renders
        the arm only). Returns the bound links."""
        import torch

        missing = [ln for ln in self.link_names if ln not in body_names]
        if missing and strict:
            raise KeyError(f"robot splat links not in articulation bodies: {missing}")
        if missing:
            keep = np.array([ln not in missing for ln in self.link_names])
            mask = keep[self.link_idx]
            self.local = self.local.select(mask)
            self.link_idx = (np.cumsum(keep) - 1)[self.link_idx[mask]]
            self.link_names = [ln for ln in self.link_names if ln not in missing]
            self._t = self.local.to_torch(self.device)
            self._link_idx_t = torch.as_tensor(self.link_idx, device=self.device)
        self._body_order = [body_names.index(ln) for ln in self.link_names]
        return list(self.link_names)

    def posed_torch(self, body_pos, body_quat) -> dict:
        """Activated world-frame tensors for the renderer from the sim body poses (``(B,3)`` positions,
        ``(B,4)`` wxyz quaternions — an articulation's ``body_link_state_w`` or a rigid object's root)."""
        import torch

        assert self._body_order is not None, "call bind_bodies(body_names) first"
        order = torch.as_tensor(self._body_order, device=self.device)
        pos = torch.as_tensor(body_pos, device=self.device, dtype=torch.float32)[order][self._link_idx_t]
        q = torch.as_tensor(body_quat, device=self.device, dtype=torch.float32)[order][self._link_idx_t]
        q = q / q.norm(dim=-1, keepdim=True)
        w, xyz = q[:, :1], q[:, 1:]
        v = self._t["means"]
        t = 2.0 * torch.cross(xyz, v, dim=-1)
        means = v + w * t + torch.cross(xyz, t, dim=-1) + pos
        w1, x1, y1, z1 = q.unbind(-1)
        w2, x2, y2, z2 = self._t["quats"].unbind(-1)
        quats = torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                             w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], dim=-1)
        return dict(means=means, quats=quats, scales=self._t["scales"], opacities=self._t["opacities"],
                    colors=self._t["colors"])

    def posed_numpy(self, body_pos: np.ndarray, body_quat: np.ndarray, body_names: list[str]) -> GaussianSet:
        """CPU variant (previews / exports): world-frame GaussianSet."""
        parts = []
        for i, ln in enumerate(self.link_names):
            b = body_names.index(ln)
            parts.append(self.local.select(self.link_idx == i).posed(body_pos[b], body_quat[b]))
        return GaussianSet.concat(parts)

    def labels(self) -> np.ndarray:
        """Per-gaussian link index, aligned with ``local`` / ``posed_numpy`` output."""
        return self.link_idx.copy()


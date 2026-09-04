"""Gaussian sets: a plain numpy container for 3D Gaussians with PLY I/O and rigid/similarity
transforms. A PLY is the unit of exchange — one object (a robot scan with per-link ``semantics``,
a table, a room) per file, as in GSWorld. Conventions follow the Inria / gsplat PLY export:

  - ``quats`` are wxyz (not necessarily normalized — rasterizers normalize),
  - ``log_scales`` are log-space (activate with exp), ``logit_opacities`` logit-space (sigmoid),
  - ``sh0`` is the DC color term ``(N, 3)``, ``shN`` the rest ``(N, K-1, 3)`` (``K = (deg+1)^2``).

2DGS surfel PLYs (two scales) are loaded as flat 3D gaussians with a thin third axis.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814  # DC SH basis; rgb = sh0 * SH_C0 + 0.5


# --------------------------------------------------------------------------- quaternion helpers (wxyz)
def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 * q2, both wxyz, broadcastable ``(..., 4)``."""
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v (..., 3)`` by unit quaternions ``q (..., 4)`` wxyz."""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion wxyz (scipy convention, converted)."""
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def decompose_similarity(T: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """4x4 similarity -> (scale, rotation 3x3, translation 3). Rotation via polar decomposition, so a
    slightly non-orthogonal ICP result still yields a proper rotation."""
    T = np.asarray(T, dtype=np.float64)
    s = float(np.cbrt(np.linalg.det(T[:3, :3])))
    U, _, Vt = np.linalg.svd(T[:3, :3] / s)
    R = U @ Vt
    if np.linalg.det(R) < 0:  # reflection guard
        U[:, -1] *= -1
        R = U @ Vt
    return s, R, T[:3, 3].copy()


def read_ply_property(path: str | Path, name: str) -> np.ndarray:
    """One float vertex property from a binary PLY (e.g. GSWorld's per-gaussian ``semantics``)."""
    data = Path(path).read_bytes()
    hdr, body = data.split(b"end_header\n", 1)
    lines = hdr.split(b"\n")
    n = int(next(l for l in lines if l.startswith(b"element vertex")).split()[-1])
    props = [l.split()[-1].decode() for l in lines if l.startswith(b"property")]
    if name not in props:
        raise KeyError(f"{path}: no '{name}' property (has {props})")
    return np.frombuffer(body, dtype="<f4", count=n * len(props)).reshape(n, len(props))[:, props.index(name)].copy()


# --------------------------------------------------------------------------- the container
@dataclass
class GaussianSet:
    means: np.ndarray            # (N, 3)
    quats: np.ndarray            # (N, 4) wxyz
    log_scales: np.ndarray       # (N, 3)
    logit_opacities: np.ndarray  # (N,)
    sh0: np.ndarray              # (N, 3)
    shN: np.ndarray              # (N, K-1, 3)

    # ----------------------------------------------------------------- construction / io
    @property
    def n(self) -> int:
        return int(self.means.shape[0])

    @property
    def sh_degree(self) -> int:
        return int(round(np.sqrt(self.shN.shape[1] + 1))) - 1

    @classmethod
    def empty(cls, sh_degree: int = 3) -> "GaussianSet":
        k = (sh_degree + 1) ** 2 - 1
        z = lambda *s: np.zeros(s, dtype=np.float32)  # noqa: E731
        return cls(z(0, 3), z(0, 4), z(0, 3), z(0), z(0, 3), z(0, k, 3))

    @classmethod
    def from_ply(cls, path: str | Path) -> "GaussianSet":
        """Read an Inria/gsplat-style binary PLY (property names, not order, are relied on)."""
        data = Path(path).read_bytes()
        hdr, body = data.split(b"end_header\n", 1)
        lines = hdr.split(b"\n")
        assert any(l.startswith(b"format binary_little_endian") for l in lines), f"{path}: need binary_little_endian ply"
        n = int(next(l for l in lines if l.startswith(b"element vertex")).split()[-1])
        props = [l.split()[-1].decode() for l in lines if l.startswith(b"property")]
        arr = np.frombuffer(body, dtype="<f4", count=n * len(props)).reshape(n, len(props))
        ix = {p: i for i, p in enumerate(props)}
        col = lambda names: arr[:, [ix[p] for p in names]].copy()  # noqa: E731

        n_rest = sum(1 for p in props if p.startswith("f_rest_"))
        shN = col([f"f_rest_{i}" for i in range(n_rest)]).reshape(n, 3, n_rest // 3).transpose(0, 2, 1) if n_rest else np.zeros((n, 0, 3), np.float32)
        if "scale_2" in ix:
            log_scales = col(["scale_0", "scale_1", "scale_2"])
        else:  # 2DGS surfel: thin third axis (log space)
            s2 = col(["scale_0", "scale_1"])
            log_scales = np.concatenate([s2, s2.min(axis=1, keepdims=True) - 3.0], axis=1)
        return cls(
            means=col(["x", "y", "z"]),
            quats=col(["rot_0", "rot_1", "rot_2", "rot_3"]),
            log_scales=log_scales.astype(np.float32),
            logit_opacities=arr[:, ix["opacity"]].copy(),
            sh0=col(["f_dc_0", "f_dc_1", "f_dc_2"]),
            shN=np.ascontiguousarray(shN, dtype=np.float32),
        )

    def to_ply(self, path: str | Path, semantics: np.ndarray | None = None) -> None:
        """Write an Inria/gsplat-style binary PLY (viewable in SuperSplat etc.). ``semantics`` adds a
        per-gaussian float property of that name — GSWorld's labeled-model convention."""
        n = self.n
        props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
        props += [f"f_rest_{i}" for i in range(self.shN.shape[1] * 3)]
        props += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        cols = [self.means, np.zeros((n, 3), np.float32), self.sh0,
                self.shN.transpose(0, 2, 1).reshape(n, -1),  # inria layout: (N, 3, K-1) flattened
                self.logit_opacities[:, None], self.log_scales, self.quats]
        if semantics is not None:
            props.append("semantics")
            cols.append(np.asarray(semantics, dtype=np.float32)[:, None])
        arr = np.concatenate([np.asarray(c, dtype=np.float32) for c in cols], axis=1).astype("<f4")
        hdr = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n" + "".join(f"property float {p}\n" for p in props) + "end_header\n"
        with open(path, "wb") as f:
            f.write(hdr.encode())
            f.write(arr.tobytes())

    @staticmethod
    def concat(sets: list["GaussianSet"]) -> "GaussianSet":
        sets = [s for s in sets if s.n > 0]
        if not sets:
            return GaussianSet.empty()
        cat = lambda k: np.concatenate([getattr(s, k) for s in sets], axis=0)  # noqa: E731
        return GaussianSet(cat("means"), cat("quats"), cat("log_scales"), cat("logit_opacities"), cat("sh0"), cat("shN"))

    # ----------------------------------------------------------------- selection
    def select(self, mask: np.ndarray) -> "GaussianSet":
        return GaussianSet(self.means[mask], self.quats[mask], self.log_scales[mask],
                           self.logit_opacities[mask], self.sh0[mask], self.shN[mask])

    def crop_box(self, lo, hi) -> "GaussianSet":
        lo, hi = np.asarray(lo, dtype=np.float32), np.asarray(hi, dtype=np.float32)
        return self.select(np.all((self.means > lo) & (self.means < hi), axis=1))

    def opacity_filter(self, min_opacity: float) -> "GaussianSet":
        return self.select(1.0 / (1.0 + np.exp(-self.logit_opacities)) >= min_opacity)

    # ----------------------------------------------------------------- transforms
    def transform(self, T: np.ndarray) -> "GaussianSet":
        """Apply a 4x4 rigid or similarity transform (uniform scale absorbed into log_scales)."""
        s, R, t = decompose_similarity(T)
        q = quat_from_matrix(R).astype(np.float32)
        return GaussianSet(
            means=((np.asarray(T)[:3, :3] @ self.means.astype(np.float64).T).T + t).astype(np.float32),
            quats=quat_mul(np.broadcast_to(q, self.quats.shape), self.quats).astype(np.float32),
            log_scales=(self.log_scales + np.log(s)).astype(np.float32),
            logit_opacities=self.logit_opacities,
            sh0=self.sh0, shN=self.shN,
        )

    def posed(self, pos: np.ndarray, quat_wxyz: np.ndarray) -> "GaussianSet":
        """Rigidly place this set (in its local frame) at world pose (pos, quat)."""
        q = np.asarray(quat_wxyz, dtype=np.float32)
        q = q / np.linalg.norm(q)
        return GaussianSet(
            means=(quat_rotate(q, self.means) + np.asarray(pos, dtype=np.float32)).astype(np.float32),
            quats=quat_mul(np.broadcast_to(q, self.quats.shape), self.quats).astype(np.float32),
            log_scales=self.log_scales, logit_opacities=self.logit_opacities, sh0=self.sh0, shN=self.shN,
        )

    # ----------------------------------------------------------------- views
    def rgb(self) -> np.ndarray:
        """Base color per gaussian in [0, 1] (DC term only) — for point-cloud previews."""
        return np.clip(self.sh0 * SH_C0 + 0.5, 0.0, 1.0)

    def to_point_ply(self, path: str | Path, min_opacity: float = 0.3) -> None:
        """Colored xyz point cloud (previews, external picking tools)."""
        g = self.opacity_filter(min_opacity)
        c = (g.rgb() * 255).astype(np.uint8)
        body = np.zeros(g.n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        body["x"], body["y"], body["z"] = g.means.T
        body["r"], body["g"], body["b"] = c.T
        with open(path, "wb") as f:
            f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {g.n}\nproperty float x\nproperty float y\n"
                     "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
            body.tofile(f)

    def to_torch(self, device: str = "cuda") -> dict:
        """Activated tensors in the layout gsplat wants: scales=exp, opacities=sigmoid, colors=(N,K,3) SH."""
        import torch

        t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)  # noqa: E731
        return dict(
            means=t(self.means), quats=t(self.quats), scales=torch.exp(t(self.log_scales)),
            opacities=torch.sigmoid(t(self.logit_opacities)),
            colors=torch.cat([t(self.sh0)[:, None, :], t(self.shN)], dim=1),
        )

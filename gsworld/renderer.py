"""gsplat rasterization wrapper: render a set of activated gaussian tensors from a PinholeCamera,
plus alpha compositing over a background plate."""
from __future__ import annotations

import numpy as np

from gsworld.camera import PinholeCamera


def concat_gaussians(parts: list[dict]) -> dict:
    """Concatenate activated gaussian dicts (as produced by ``GaussianSet.to_torch`` / ``SplatModel.posed_torch``)."""
    import torch

    parts = [p for p in parts if p is not None and p["means"].shape[0] > 0]
    if len(parts) == 1:
        return parts[0]
    return {k: torch.cat([p[k] for p in parts], dim=0) for k in ("means", "quats", "scales", "opacities", "colors")}


class SplatRenderer:
    def __init__(self, device: str = "cuda", near_plane: float = 0.05, background=None) -> None:
        self.device = device
        self.near_plane = near_plane
        self.background = background  # None -> transparent (alpha reported); else (3,) rgb in [0,1]

    def render(self, gaussians: dict, camera: PinholeCamera) -> tuple[np.ndarray, np.ndarray]:
        """-> (rgb (H, W, 3) float32 in [0,1], alpha (H, W, 1) float32)."""
        import torch

        try:
            from gsplat import rasterization
        except ImportError as exc:  # the gs layer is an optional extra
            raise ImportError("gsworld rendering needs gsplat: `pip install -e .[gs]` "
                              "(or `pip install gsplat`)") from exc

        w2c, K = camera.to_torch(self.device)
        sh_degree = int(round(np.sqrt(gaussians["colors"].shape[1]))) - 1
        bg = None if self.background is None else torch.as_tensor(self.background, dtype=torch.float32, device=self.device)[None]
        rgb, alpha, _ = rasterization(
            gaussians["means"], gaussians["quats"], gaussians["scales"], gaussians["opacities"], gaussians["colors"],
            w2c, K, camera.width, camera.height, sh_degree=sh_degree, packed=False, near_plane=self.near_plane,
            backgrounds=bg,
        )
        return rgb[0].clamp(0, 1).detach().cpu().numpy(), alpha[0].detach().cpu().numpy()

    @staticmethod
    def composite(rgb: np.ndarray, alpha: np.ndarray, plate: np.ndarray) -> np.ndarray:
        """Alpha-over the splat render onto an ``(H, W, 3)`` uint8 plate (e.g. the sim render with the
        splat-covered robot hidden). Returns uint8."""
        p = plate.astype(np.float32) / 255.0
        out = rgb * alpha + p * (1.0 - alpha)
        return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)

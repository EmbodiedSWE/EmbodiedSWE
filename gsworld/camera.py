"""Pinhole cameras for the splat renderer, taken from Isaac.

Conventions: ``K`` is the OpenCV intrinsic matrix for a ``width x height`` image; ``w2c`` is the
OpenCV world-to-camera transform (+Z forward, +Y down). USD cameras look down -Z with +Y up
(OpenGL), hence the axis flip. Kit fits the *horizontal* aperture to the render resolution and
keeps square pixels, so ``fy == fx`` regardless of the authored vertical aperture.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass
class PinholeCamera:
    K: np.ndarray      # (3, 3)
    w2c: np.ndarray    # (4, 4) OpenCV world-to-camera
    width: int
    height: int

    @property
    def c2w(self) -> np.ndarray:
        return np.linalg.inv(self.w2c)

    @classmethod
    def from_opencv_pose(cls, K: np.ndarray, c2w_cv: np.ndarray, width: int, height: int) -> "PinholeCamera":
        return cls(np.asarray(K, dtype=np.float64), np.linalg.inv(np.asarray(c2w_cv, dtype=np.float64)), width, height)

    @classmethod
    def from_usd_prim(cls, stage, prim_path: str, width: int, height: int, origin=(0.0, 0.0, 0.0)) -> "PinholeCamera":
        """Read focal/aperture + world transform of a USD camera prim (e.g. ``/OmniverseKit_Persp``).

        ``origin`` is subtracted from the camera position (Isaac Lab env origins -> env-local frame).
        """
        from pxr import UsdGeom

        prim = stage.GetPrimAtPath(prim_path)
        cam = UsdGeom.Camera(prim)
        f = cam.GetFocalLengthAttr().Get()
        h_ap = cam.GetHorizontalApertureAttr().Get()
        fx = f / h_ap * width
        K = np.array([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]])
        c2w_gl = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T  # usd row-major -> column
        c2w_cv = c2w_gl @ _GL_TO_CV
        c2w_cv[:3, 3] -= np.asarray(origin, dtype=np.float64)
        return cls.from_opencv_pose(K, c2w_cv, width, height)

    @classmethod
    def from_isaaclab_camera(cls, sensor, env_index: int = 0, origin=(0.0, 0.0, 0.0)) -> "PinholeCamera":
        """From an Isaac Lab ``Camera`` sensor (uses its intrinsic matrix + world pose, 'world' convention)."""
        import isaaclab.utils.math as lab_math

        K = sensor.data.intrinsic_matrices[env_index].detach().cpu().numpy().astype(np.float64)
        pos = sensor.data.pos_w[env_index].detach().cpu().numpy().astype(np.float64) - np.asarray(origin)
        quat = sensor.data.quat_w_ros[env_index]  # ROS/OpenCV: +Z forward, +Y down
        R = lab_math.matrix_from_quat(quat[None])[0].detach().cpu().numpy().astype(np.float64)
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = R, pos
        h, w = sensor.image_shape
        return cls.from_opencv_pose(K, c2w, int(w), int(h))

    def to_torch(self, device: str = "cuda") -> tuple:
        import torch

        return (torch.as_tensor(self.w2c, dtype=torch.float32, device=device)[None],
                torch.as_tensor(self.K, dtype=torch.float32, device=device)[None])

"""gsworld — optional Gaussian-splat rendering for robobench envs (GSWorld style).

Independent of robobench's core: it imports robobench, never the other way round. Install the extras
with `pip install -e .[gs]`; without this package every robobench env still runs.

The sim stays the source of truth for physics, contacts, and proprioception; this package only
replaces camera RGB with a photoreal 3D-Gaussian-Splatting render whose robot gaussians are
re-posed every step from the articulation's link poses:

  - :mod:`gsworld.splat`        gaussian sets: one object per PLY (3DGS / 2DGS, semantics), transforms, crops
  - :mod:`gsworld.model`        SplatModel: any object (robot, table, cup) as PLY + poses, re-posed from the sim
  - :mod:`gsworld.camera`       pinhole camera (K + OpenCV world-to-cam) from an Isaac camera prim
  - :mod:`gsworld.renderer`     thin gsplat rasterization wrapper + compositing
  - :mod:`gsworld.wrapper`      ``SplatEnv``: wraps a robobench env, renders splat RGB per step

Heavy imports (torch, gsplat, Isaac) are deferred so ``import gsworld`` stays app-free.
"""
from __future__ import annotations

from gsworld.splat import GaussianSet

__all__ = ["GaussianSet"]

"""``SplatEnv`` — wrap a built robobench env so every step can return a photoreal splat render.

Mechanism (GSWorld): physics, contacts and proprioception stay in the sim; the robot's
per-link gaussians are re-posed from the articulation's live body poses, static splats (scene,
table, objects — all already in the sim frame) are appended,
and the result is rasterized from a pinhole camera. Pixels the splats don't cover fall back to the
sim render with the splat-covered robot hidden (so the mesh never bleeds through).

    from gsworld.wrapper import SplatEnv
    env = EnvCfg(scene="table", robot="franka_robotiq", control_mode="joint").build()
    senv = SplatEnv(env, robot_splat=SplatModel(ply, poses), static_splats=[...])
    senv.attach_viewport_camera(eye, target)
    obs = senv.step(action)                 # dict: rgb (splat), sim_rgb, alpha
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from gsworld.camera import PinholeCamera
from gsworld.renderer import SplatRenderer, concat_gaussians
from gsworld.model import SplatModel
from gsworld.splat import GaussianSet


class SplatEnv:
    def __init__(self, env, robot_splat: SplatModel | None = None,
                 static_splats: list[str | Path | GaussianSet] | None = None,
                 object_splats: dict[str, SplatModel] | None = None,
                 device: str | None = None, hide_robot_in_plate: bool = True) -> None:
        self.env = env
        self.device = device or str(env.device)
        self.renderer = SplatRenderer(device=self.device)
        self.camera: PinholeCamera | None = None
        self._annot = None
        self._hidden_prims: list = []
        self.hide_robot_in_plate = hide_robot_in_plate
        self.origin = env.iscene.env_origins[0].detach().cpu().numpy().astype(np.float64)
        import torch

        self._origin_t = torch.as_tensor(self.origin, dtype=torch.float32, device=self.device)

        self.robot_splat: SplatModel | None = None
        self.splat_links: list[str] = []
        if robot_splat is not None:
            self.robot_splat = robot_splat
            bodies = list(env.robot.articulation.body_names)
            self.splat_links = self.robot_splat.bind_bodies(bodies, strict=False)
            if len(self.splat_links) < len(bodies):
                print(f"[gsworld] partial robot splat: {len(self.splat_links)}/{len(bodies)} links covered "
                      f"(uncovered links stay raytraced)", flush=True)

        parts = []
        for s in static_splats or []:
            g = s if isinstance(s, GaussianSet) else GaussianSet.from_ply(s)
            parts.append(g.to_torch(self.device))
        self._static = concat_gaussians(parts) if parts else None
        # object models: bound to a rigid object (one link, its root pose) or an articulation (its bodies)
        self._objects: dict[str, SplatModel] = {}
        for name, m in (object_splats or {}).items():
            if name in env.iscene.rigid_objects:
                m.bind_bodies([name], strict=False)
            elif name in env.iscene.articulations:
                m.bind_bodies(list(env.iscene[name].body_names), strict=False)
            else:
                raise KeyError(f"object model '{name}' has no rigid object / articulation in the scene")
            self._objects[name] = m

    @classmethod
    def from_config(cls, env, config, **kw) -> "SplatEnv":
        """Wrap an existing env from a scene JSON path or a loaded ``SplatSceneCfg`` (see :mod:`gsworld.config`)."""
        from gsworld.config import SplatSceneCfg

        cfg = config if isinstance(config, SplatSceneCfg) else SplatSceneCfg.load(config)
        kw = dict(kw)
        device = kw.pop("device", None) or str(env.device)
        return cls(env, robot_splat=cfg.robot_splat(device), static_splats=cfg.load_static(),
                   object_splats=cfg.load_objects(device), device=device, **kw)

    @property
    def n_static(self) -> int:
        return 0 if self._static is None else int(self._static["means"].shape[0])

    # ----------------------------------------------------------------- camera
    def attach_viewport_camera(self, eye, target, width: int = 960, height: int = 600,
                               prim_path: str = "/OmniverseKit_Persp") -> "SplatEnv":
        """Point the Kit viewport camera and attach an RGB annotator; derive the splat camera from it."""
        import omni.replicator.core as rep

        env = self.env
        env.sim.set_camera_view(tuple(np.asarray(eye) + self.origin), tuple(np.asarray(target) + self.origin),
                                camera_prim_path=prim_path)
        rp = rep.create.render_product(prim_path, (width, height))
        self._annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self._annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        self.camera = PinholeCamera.from_usd_prim(env.stage, prim_path, width, height, origin=self.origin)
        self._resolve_hidden_prims()
        return self

    def _resolve_hidden_prims(self) -> None:
        """Prims hidden when grabbing the composite plate: one per splat-covered link (partial splats
        keep the uncovered links raytraced); with no robot splat nothing is hidden."""
        import isaaclab.sim as sim_utils
        from pxr import Usd, UsdGeom

        self._hidden_prims = []
        if self.robot_splat is None:
            return
        paths = sim_utils.find_matching_prim_paths(self.env.robot.articulation.cfg.prim_path)
        if not paths:
            return
        root = self.env.stage.GetPrimAtPath(paths[0])
        wanted = set(self.splat_links)
        for prim in Usd.PrimRange(root):
            if prim.GetName() in wanted:
                self._hidden_prims.append(UsdGeom.Imageable(prim))
                wanted.discard(prim.GetName())
        if wanted:
            print(f"[gsworld] no prim found for splat links {sorted(wanted)}; they will bleed through the plate", flush=True)

    def use_viewport(self, annot, width: int, height: int, prim_path: str = "/OmniverseKit_Persp") -> "SplatEnv":
        """Adopt an existing viewport render product + RGB annotator (e.g. scripts/record_video.py's)
        instead of creating one; the splat camera is re-read from the prim on every render so a
        camera moved by the driver script is followed."""
        self._annot = annot
        self._viewport = (prim_path, width, height)
        self.camera = PinholeCamera.from_usd_prim(self.env.stage, prim_path, width, height, origin=self.origin)
        self._resolve_hidden_prims()
        return self

    def set_camera(self, camera: PinholeCamera) -> "SplatEnv":
        """Render the splats from an arbitrary camera (e.g. a calibrated real external camera)."""
        self.camera = camera
        return self

    # ----------------------------------------------------------------- stepping / rendering
    def _grab(self) -> np.ndarray:
        for _ in range(2):
            self.env.sim.render()
        return np.asarray(self._annot.get_data())[..., :3].astype(np.uint8).copy()

    def grab_sim(self) -> np.ndarray:
        """The sim's own RGB for the current state (uint8)."""
        return self._grab()

    def grab_plate(self) -> np.ndarray:
        """The sim RGB with the splat-covered links hidden — the background the splats go over."""
        prims = self._hidden_prims if self.hide_robot_in_plate else []
        if not prims:
            return self._grab()
        for p in prims:
            p.MakeInvisible()
        try:
            return self._grab()
        finally:
            for p in prims:
                p.MakeVisible()

    def gaussians(self) -> dict | None:
        """All activated gaussians for the current sim state (robot re-posed + static), or None."""
        parts = []
        if self.robot_splat is not None:
            st = self.env.robot.articulation.data.body_link_state_w[0]
            parts.append(self.robot_splat.posed_torch(st[:, :3], st[:, 3:7]))
        if self._static is not None:
            parts.append(self._static)
        for name, m in self._objects.items():
            asset = self.env.iscene[name]
            if name in self.env.iscene.articulations:
                st = asset.data.body_link_state_w[0]
                parts.append(m.posed_torch(st[:, :3] - self._origin_t, st[:, 3:7]))
            else:
                st = asset.data.root_state_w[0]
                parts.append(m.posed_torch(st[None, :3] - self._origin_t, st[None, 3:7]))
        return concat_gaussians(parts) if parts else None

    def render(self, sim_rgb: bool = True) -> dict:
        """Render the current sim state -> ``rgb`` (uint8 splat composite), ``alpha``, ``sim_rgb``."""
        assert self.camera is not None, "attach_viewport_camera() or set_camera() first"
        if getattr(self, "_viewport", None):  # follow a viewport camera someone else may have moved
            p, w, h = self._viewport
            self.camera = PinholeCamera.from_usd_prim(self.env.stage, p, w, h, origin=self.origin)
        out = {}
        plate = None
        if self._annot is not None and sim_rgb:
            out["sim_rgb"] = self.grab_sim()
            plate = self.grab_plate()
        g = self.gaussians()
        if g is not None:
            rgb, alpha = self.renderer.render(g, self.camera)
            out["alpha"] = alpha
            out["rgb"] = self.renderer.composite(rgb, alpha, plate) if plate is not None else (rgb * 255).astype(np.uint8)
        else:
            out["rgb"] = plate
        return out

    def step(self, action, render: bool = True) -> dict | None:
        """Step the env; render the splat view when ``render`` (returns None otherwise)."""
        self.env.step(action, render=render)
        return self.render() if render else None

    def reset(self) -> dict:
        self.env.reset()
        return self.render()

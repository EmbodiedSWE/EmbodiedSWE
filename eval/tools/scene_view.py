"""Scene view — see what your code actually did, instead of inferring it from numbers.

    from scene_view import Viewer

    viewer = Viewer(env)                    # once, after env.reset()
    viewer.snapshot("before")               # -> /workspace/.footage/0001_before.png
    for t in range(steps):
        env.step(action)
        viewer.grab()                       # buffer a frame (cheap; every 2nd call renders)
    viewer.snapshot("after grasp")
    print(viewer.save_frames(6, "approach"))  # 6 stills sampled evenly across the run
    viewer.save_video("approach")           # the same footage as one H.264 mp4

Open the PNGs with your file tools and LOOK at them: a policy that reports success while the
part lies on the floor is caught in one glance and in no number of printed distances.

The capture recipe:

  * the default viewport camera (`/OmniverseKit_Persp`) with a replicator rgb annotator — the
    same path Isaac Lab's own `render()` uses, not a Camera sensor added to the scene;
  * FXAA instead of temporal AA: TAA/DLSS blends frames across time, so teleported state
    (`set_states` restores, resets) leaves ghost trails in every capture no matter how many
    flush renders run. FXAA is single-frame — no accumulation, no ghosts;
  * replica envs are hidden from the render when `num_envs > 1` (visibility only, physics
    untouched): they are RL-batch machinery, not scenery, and RTX would otherwise pay to trace
    rows of ghost benches in every frame;
  * the view is anchored to env 0's origin, so with replicated envs it films the work area
    rather than empty floor.

Boot requirement: the capture path only exists when the app is launched with cameras enabled —
`AppLauncher(headless=True, enable_cameras=True)`. Without it `Viewer` raises at construction
(better than silently recording nothing).
"""

from __future__ import annotations

import time
from pathlib import Path

# Self-declaration, read by eval/tools/__init__.py::discover(). A plain dict on purpose: this
# file is also installed standalone on the agent's PYTHONPATH, where a relative import of the
# tools package would fail.
TOOL = {
    "name": "scene_view",
    "description": (
        "Render what happened: snapshots of the current state and evenly-sampled stills / an "
        "mp4 of a run, captured ghost-free from the viewport camera, saved under /workspace "
        "for the agent to open and look at."
    ),
    "exports": ("Viewer",),
    "prompt_doc": "tools/scene_view.md",
}

# Default framing suits a tabletop workspace (the common case in this bench); callers with
# other scene layouts pass their own eye/target.
DEFAULT_EYE = (1.6, 1.6, 1.9)
DEFAULT_TARGET = (-0.25, 0.0, 1.0)


class Viewer:
    """Viewport RGB capture for one env, writing PNGs (and optionally MP4s) under `out`."""

    def __init__(self, env, out: str | Path = "/workspace/.footage",
                 size: tuple[int, int] = (1280, 720),
                 eye: tuple[float, float, float] | None = None,
                 target: tuple[float, float, float] | None = None,
                 every: int = 2, frame: str = "env"):
        """`eye`/`target` are relative to env 0's origin by default; pass `frame="world"`
        to aim in world coordinates (e.g. straight at an object's `root_pos_w`)."""
        import numpy as np

        self.env = env
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.every = max(1, int(every))
        self._frames: list = []       # in-memory tail of the footage (chunks spill to disk)
        self._parts: list[Path] = []  # spilled npz chunks, in order — zero truncation
        self._n_frames = 0            # frames captured EVER (parts + buffer)
        self._grab_calls = 0
        self._n_saved = 0
        self._orig_step = None        # env.step before auto() hooked it
        import os as _os

        # 400 = the original recorder's chunk size (CAPX_VIDEO_CHUNK), kept env-overridable
        self._CHUNK = int(_os.environ.get("CAPX_VIDEO_CHUNK", "400"))

        import omni.replicator.core as rep

        # rgb_array capture requires partial rendering; FXAA kills temporal ghosting.
        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        try:
            import carb.settings

            carb.settings.get_settings().set("/rtx/post/aa/op", 2)  # 2 = FXAA
        except Exception as exc:  # noqa: BLE001 -- a quality tweak must not block capture
            print(f"[scene_view] could not force FXAA ({exc!r}); captures may ghost after "
                  f"teleports", flush=True)

        # Replicas are batch machinery, not scenery: hide envs 1..N-1 from the renderer.
        try:
            if env.num_envs > 1:
                import omni.usd
                from pxr import UsdGeom

                stage = omni.usd.get_context().get_stage()
                for i in range(1, int(env.num_envs)):
                    prim = stage.GetPrimAtPath(f"/World/envs/env_{i}")
                    if prim and prim.IsValid():
                        UsdGeom.Imageable(prim).MakeInvisible()
        except Exception as exc:  # noqa: BLE001
            print(f"[scene_view] could not hide replica envs ({exc!r}); expect rows of "
                  f"duplicate scenes in frame", flush=True)

        # Anchor the view to env 0's origin so replicated envs film the work area (world-frame
        # callers already aimed at real coordinates and get them unshifted).
        origin = (np.zeros(3) if frame == "world"
                  else env.iscene.env_origins[0].detach().cpu().numpy().astype(float))
        eye = np.asarray(eye if eye is not None else DEFAULT_EYE, dtype=float) + origin
        target = np.asarray(target if target is not None else DEFAULT_TARGET, dtype=float) + origin
        env.sim.set_camera_view(tuple(eye), tuple(target),
                                camera_prim_path="/OmniverseKit_Persp")

        product = rep.create.render_product("/OmniverseKit_Persp",
                                            (int(size[0]), int(size[1])))
        self._annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self._annot.attach([product])
        for _ in range(6):  # warm up so the first real frame is populated
            env.sim.render()
        probe = self._annot.get_data()
        shape = getattr(probe, "shape", None)
        if not shape or len(shape) < 3 or shape[0] == 0:
            raise RuntimeError(
                "viewport capture returned no pixels — the app must be launched with "
                "AppLauncher(headless=True, enable_cameras=True)")
        print(f"[scene_view] capture ready, rgb shape {tuple(shape)}, writing to {self.out}",
              flush=True)

    # ----- capture -------------------------------------------------------------------------
    @property
    def annotator(self):
        """The raw rgb annotator, for callers that manage their own frame pipeline (e.g. a
        dataset recorder writing straight into a video encoder)."""
        return self._annot

    def frame(self):
        """Render the current state and return it as an (H,W,3) uint8 array (None on failure)."""
        return self._render_once()

    def _render_once(self):
        import numpy as np

        self.env.sim.render()
        data = self._annot.get_data()
        arr = np.asarray(data)
        if arr.ndim != 3 or arr.shape[0] == 0:
            return None
        return arr[:, :, :3].astype(np.uint8).copy()

    def auto(self, every: int | None = None) -> None:
        """Record automatically: every `every`-th `env.step(...)` captures a frame, no grab()
        calls needed — recording just happens, like the original harness. Call `stop_auto()`
        to unhook (e.g. before a parameter search, whose footage is N overlaid candidates)."""
        if every:
            self.every = max(1, int(every))
        if self._orig_step is not None:
            return
        orig = self.env.step

        def stepped(*args, **kwargs):
            out = orig(*args, **kwargs)
            self.grab()
            return out

        self._orig_step = orig
        self.env.step = stepped
        print(f"[scene_view] auto-recording every {self.every} step(s)", flush=True)

    def stop_auto(self) -> None:
        if self._orig_step is not None:
            self.env.step = self._orig_step
            self._orig_step = None

    def grab(self) -> bool:
        """Buffer one frame of the run (or let `auto()` call this for you). Only every
        `every`-th call actually renders, so a per-step loop stays cheap; full chunks spill
        to disk, so memory stays bounded and no frame is ever dropped."""
        self._grab_calls += 1
        if (self._grab_calls - 1) % self.every:
            return False
        frame = self._render_once()
        if frame is not None:
            self._frames.append(frame)
            self._n_frames += 1
            if len(self._frames) >= self._CHUNK:
                self._spill()
        return frame is not None

    def _spill(self) -> None:
        """Write the in-memory buffer to an npz part (kept, never dropped). On a write
        failure the buffer is KEPT in memory so no frame is lost silently."""
        import numpy as np

        if not self._frames:
            return
        try:
            parts_dir = self.out / ".parts"
            parts_dir.mkdir(parents=True, exist_ok=True)
            path = parts_dir / f"part_{len(self._parts):05d}.npz"
            np.savez_compressed(path, frames=np.stack(self._frames, axis=0))
            self._parts.append(path)
            self._frames = []
        except Exception as exc:  # noqa: BLE001
            print(f"[scene_view] chunk spill FAILED ({exc!r}) — keeping frames in memory "
                  f"(no loss)", flush=True)

    def _all_frames(self):
        """Every frame captured so far, parts + live buffer, in order."""
        import numpy as np

        out = []
        for p in self._parts:
            out.extend(np.load(p)["frames"])
        out.extend(self._frames)
        return out

    def snapshot(self, label: str = "") -> str:
        """Render the CURRENT state to a PNG and return its path."""
        frame = self._render_once()
        if frame is None:
            return "(snapshot failed: renderer returned no pixels)"
        return self._write_png(frame, label or "snapshot")

    # ----- reading the footage back --------------------------------------------------------
    def __len__(self) -> int:
        return self._n_frames

    def save_frames(self, k: int = 6, label: str = "run") -> list[str]:
        """Sample `k` frames evenly across everything captured so far and write PNGs."""
        import numpy as np

        frames = self._all_frames()
        if not frames:
            print("[scene_view] no frames captured — call auto() once, or grab() inside "
                  "the stepping loop", flush=True)
            return []
        idxs = np.linspace(0, len(frames) - 1, min(int(k), len(frames))).astype(int)
        return [self._write_png(frames[i], f"{label}_{j:02d}") for j, i in enumerate(idxs)]

    def save_video(self, label: str = "run", fps: int = 15) -> str:
        """Write everything captured so far as one H.264/yuv420p mp4 (the codec that actually
        plays in the IDE) and return its path."""
        frames = self._all_frames()
        if not frames:
            return "(no frames captured — call auto() once, or grab() inside the loop)"
        path = self.out / f"{self._next_idx():04d}_{_safe(label)}.mp4"
        try:
            import imageio

            with imageio.get_writer(str(path), fps=int(fps), codec="libx264",
                                    pixelformat="yuv420p", macro_block_size=2) as w:
                for f in frames:
                    w.append_data(f)
        except Exception as exc:  # noqa: BLE001 -- stills must survive a codec problem
            print(f"[scene_view] mp4 write failed ({exc!r}); falling back to stills", flush=True)
            return str(self.save_frames(8, label))
        print(f"[scene_view] wrote {path} ({len(frames)} frames @ {fps} fps)", flush=True)
        return str(path)

    def clear(self) -> None:
        """Start the NEXT recording segment: drops the buffer and spilled chunks (the PNGs
        and MP4s already written stay), so each video covers exactly one attempt."""
        self._frames = []
        for p in self._parts:
            p.unlink(missing_ok=True)
        self._parts = []
        self._n_frames = 0
        self._grab_calls = 0

    # ----- internals -----------------------------------------------------------------------
    def _next_idx(self) -> int:
        self._n_saved += 1
        return self._n_saved

    def _write_png(self, frame, label: str) -> str:
        from PIL import Image

        path = self.out / f"{self._next_idx():04d}_{_safe(label)}.png"
        Image.fromarray(frame).save(path)
        print(f"[scene_view] wrote {path}", flush=True)
        return str(path)


def _safe(label: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_" else "_" for c in label.strip())
    return (keep or f"view_{int(time.time())}")[:60]

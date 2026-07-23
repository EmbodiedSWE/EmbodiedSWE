"""Video recorder for sim_gen scenes.

House rule: EVERY smoke/validation run records a video.  Frames are captured through
MuJoCo's offscreen renderer and encoded H.264 + yuv420p via imageio-ffmpeg's bundled
ffmpeg — never OpenCV's mp4v, which renders as a green screen in the IDE player.

The offscreen GL backend defaults to osmesa (software, works on any headless box) but
is overridable via SIM_GEN_MUJOCO_GL for boxes where a GPU path is preferable/required
(e.g. SIM_GEN_MUJOCO_GL=egl when libOSMesa is unavailable but a GPU + EGL is).
"""

from __future__ import annotations

import os
import subprocess

os.environ.setdefault("MUJOCO_GL", os.environ.get("SIM_GEN_MUJOCO_GL", "osmesa"))

import imageio_ffmpeg
import mujoco
import numpy as np


class Recorder:
    def __init__(self, width: int = 960, height: int = 600, every: int = 8,
                 max_frames: int = 3000):
        self.width, self.height = width, height
        self.every = every                # capture every N physics steps
        self.max_frames = max_frames
        self.frames: list[np.ndarray] = []
        self._renderer = None
        self._count = 0

    def maybe_capture(self, scene) -> None:
        self._count += 1
        if self._count % self.every or len(self.frames) >= self.max_frames:
            return
        if self._renderer is None:
            self._renderer = mujoco.Renderer(scene.model, self.height, self.width)
        cam = mujoco.MjvCamera()
        cam.distance = scene.cfg.camera_distance
        cam.azimuth = scene.cfg.camera_azimuth
        cam.elevation = scene.cfg.camera_elevation
        cam.lookat[:] = scene.cfg.camera_lookat
        self._renderer.update_scene(scene.data, camera=cam)
        self.frames.append(self._renderer.render().copy())

    def save(self, path: str, fps: int = 30) -> str:
        """Encode captured frames to H.264 mp4 at ``path``."""
        if not self.frames:
            raise RuntimeError("recorder captured no frames")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        h, w, _ = self.frames[0].shape
        cmd = [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for f in self.frames:
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}) for {path}")
        return path

"""Record one episode: the trajectory, and the video of that same rollout.

Both come from the SAME env.step loop on purpose. GPU PhysX in a contact-rich scene like this
(192 solver iterations, SDF threads) is not run-to-run identical, so a trajectory replayed from
its parameters afterwards is a different trajectory — and a video rendered from a replay would
not be footage of the rollout the verifier passed. Recording inline costs a camera capture every
`video_every` control steps and nothing else: rendering does not touch physics.

Per control step it keeps what a policy-learning consumer needs: the action, the robot's joint
state, the EE pose, the gripper width, and every scene object's full 13-dim root state (pos,
quat, lin vel, ang vel) — which for this scene is the nut's threaded height, the thing the task
is scored on.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class _View:
    """What build_record_camera reads its eye/target from (it consults the live api's VIEWS)."""

    def __init__(self, eye, target):
        self.VIEWS = {"default": (tuple(eye), tuple(target))}


class EpisodeRecorder:
    def __init__(self, env, video_path: str | None = None, video_every: int = 16,
                 fps: int = 30, width: int = 1280, height: int = 720):
        self.env = env
        self.video_every = int(video_every)
        self.render = video_path is not None
        self.n = 0
        self.cam = self.writer = None
        self.frames = 0
        self.video_path = video_path
        self._assets: dict | None = None
        self.rows: dict[str, list] = {k: [] for k in (
            "t", "action", "joint_pos", "joint_vel", "ee_pos", "ee_quat", "grip", "phase", "tstate")}
        self.objects: dict[str, list] = {}
        art = env.robot.articulation
        self._ee = art.body_names.index("panda_hand")
        self._fj = art.find_joints(["panda_finger_joint1"])[0]
        self._wh = (int(width), int(height))
        if video_path:
            import imageio.v2 as imageio
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            # H.264 + yuv420p: OpenCV's default mp4v renders as a green screen in the IDE player.
            self.writer = imageio.get_writer(video_path, fps=fps, codec="libx264",
                                             pixelformat="yuv420p", macro_block_size=1)

    def aim(self, eye=None, target=None) -> None:
        """Attach the capture pipeline and aim it at the bolt. Call after env.reset().

        Uses the repo's own build_record_camera, which is the capture path PROVEN on these L20
        pods: a replicator render product plus an rgb annotator on the viewport camera, with the
        sim in PARTIAL_RENDERING and FXAA. An isaaclab Camera *sensor* — what the suite's smoke
        scripts use — returns an empty [1, 0] buffer in this app configuration no matter how many
        frames are pumped through it, which is what the first three smoke batches died on.
        """
        if self.writer is None:
            return
        # The capture pipeline moved into the tools library (eval/tools/scene_view.py) when the
        # cosigen harness was retired; Viewer is the same replicator+FXAA recipe behind a class.
        from tools.scene_view import Viewer

        if eye is None:
            eye, target = self.frame_scene()
        self.cam = Viewer(self.env, out="/tmp/div_footage", size=self._wh,
                          eye=eye, target=target, frame="world").annotator
        probe = self.cam.get_data()
        shape = getattr(probe, "shape", None)
        if not shape or len(shape) < 3 or shape[0] < 8:
            raise RuntimeError(f"capture pipeline yielded {shape} — no usable frame")
        print(f"[rec] capture ready rgb_shape={shape}", flush=True)

    def observe(self, env, action, phase: str = "", tstate: str = "") -> bool:
        """Record (state, action) BEFORE the step, and say whether this step must render.

        Called before env.step so the stored state is the one the action was computed from. Doing
        it after the step would pair a_t with s_{t+1}, which silently shifts every state-action
        pair a consumer trains on by one control period.

        Rendering happens only on capture steps, as the suite's smoke scripts do: rendering all
        144k steps of an episode would cost far more than the frames are worth.
        """
        self._capture_now = self.writer is not None and (self.n % self.video_every == 0)
        self._record(env, action, phase, tstate)
        return self._capture_now

    def _record(self, env, action, phase: str = "", tstate: str = "") -> None:
        art = env.robot.articulation
        self.rows["t"].append(self.n)
        self.rows["action"].append(action[0].detach().cpu().numpy().copy())
        self.rows["joint_pos"].append(art.data.joint_pos[0].detach().cpu().numpy().copy())
        self.rows["joint_vel"].append(art.data.joint_vel[0].detach().cpu().numpy().copy())
        self.rows["ee_pos"].append(art.data.body_pos_w[0, self._ee].detach().cpu().numpy().copy())
        self.rows["ee_quat"].append(art.data.body_quat_w[0, self._ee].detach().cpu().numpy().copy())
        self.rows["grip"].append(float(art.data.joint_pos[0, self._fj].item()))
        self.rows["phase"].append(phase)
        self.rows["tstate"].append(tstate)
        for name, asset in self._scene_objects().items():
            self.objects.setdefault(name, []).append(
                asset.data.root_state_w[0].detach().cpu().numpy().copy())
        self.n += 1

    def capture(self, env) -> None:
        """Grab the frame for this step. Called after env.step, so the pixels show the result."""
        if getattr(self, "_capture_now", False):
            img = np.asarray(self.cam.get_data())
            if img.dtype != np.uint8:
                img = (img.clip(0, 1) * 255).astype(np.uint8)
            if img.ndim != 3 or img.shape[0] < 8 or img.shape[1] < 8:
                raise RuntimeError(f"capture returned {img.shape} {img.dtype} — no usable frame")
            if self.frames == 0 and not img.any():
                raise RuntimeError("the first captured frame is entirely black — refusing to "
                                   "record a batch of black videos")
            self.writer.append_data(img[..., :3])
            self.frames += 1

    def _scene_objects(self) -> dict:
        """Every rigid asset the scene exposes, by name. Scenes hold them either as lists
        (nut_thread: `nuts`, `bolts`) or as dicts (pen_holder: `pens`, plus a single `holder`),
        so discover rather than hardcode."""
        if self._assets is not None:
            return self._assets
        scene, out = self.env.scene, {}
        for attr in dir(scene):
            if attr.startswith("_") or attr in ("cfg", "num_envs"):
                continue
            try:
                val = getattr(scene, attr)
            except Exception:  # noqa: BLE001 -- properties that need a running sim
                continue
            items = (val.items() if isinstance(val, dict)
                     else enumerate(val) if isinstance(val, (list, tuple))
                     else [("", val)])
            for key, obj in items:
                if hasattr(obj, "data") and hasattr(obj.data, "root_state_w"):
                    out[f"{attr}_{key}" if key != "" else attr] = obj
        self._assets = out
        print(f"[rec] recording {len(out)} assets: {', '.join(sorted(out))}", flush=True)
        return out

    def save(self, npz_path: str, meta: dict) -> None:
        arrays = {k: np.asarray(v) for k, v in self.rows.items() if k not in ("phase", "tstate")}
        arrays["phase"] = np.asarray(self.rows["phase"], dtype=object)
        arrays["tstate"] = np.asarray(self.rows["tstate"], dtype=object)
        for name, seq in self.objects.items():
            arrays[f"obj_{name}"] = np.asarray(seq)
        Path(npz_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, meta=np.asarray([repr(meta)], dtype=object), **arrays)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

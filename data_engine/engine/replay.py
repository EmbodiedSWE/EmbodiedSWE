"""replay — L5 visual: re-render recorded episodes kinematically, in parallel envs.

Generation records full restorable states per sim tick (`ep_NNNN/traj.npz`, keys =
the flattened `env.get_states()` tree). Replay rebuilds the env from the CELL's
LOCAL scene (same as generation), restores E episodes at once — one per env slot —
through `env.set_states`, and renders every env in a single pass with a TiledCamera.
Physics never decides anything: every rendered frame is a restored recorded state
(plus the one sim tick that syncs PhysX -> renderer), so visual changes (camera,
lighting, textures) apply retroactively to every episode ever recorded.

The parts that bite (all handled here):
  - env-origin shift: recorded states are world-frame on the SOURCE batch's env grid;
    each replay slot shifts them onto its own origin, anchored on the robot base
    (deterministic, never jittered): shift_e = replay_base_e - recorded robot/root[0].
  - scene.post_step() runs on the restored state BEFORE the render, so state-coupled
    visuals (the bulb-glow mechanic) land in the same frame, not one frame late.
  - the RTX temporal denoiser accumulates across frames ("visual cache"): a teleport
    leaves ghost trails, and the first frames of a boot render dark/noisy. Warmup
    renders at every chunk start flush it before any frame is kept.
  - Kit renders the state PhysX had at its last step: `sim.step(render=True)` (one
    tick from the restored state, velocities as recorded) is what syncs transforms
    into the renderer — `sim.render()` alone would show the previous frame's poses.

Outputs, per episode: `imgs/<cam>/frame_%06d.jpg` (the dataset frames, README's
`imgs/` slot), `imgs/render_<cam>.json` (frame indices + camera + joint names +
visual draw — the export contract, per cam so K looks coexist), `imgs/preview.mp4`
(time-lapsed, for eyeballing — shared, last run wins). Per batch:
`replay_sheet.png` (episodes x time contact sheet).

Visual diversification is SCENE-OWNED, like world physics: the cell's scene.py
declares `VISUAL_PARAMS` bands (same grammar as `PHYSICAL_PARAMS`, cfg default =
the nominal look). `visual_draw=k` samples the bands at Halton index k into ONE
stage-wide look for the whole render pass (lights and ground are shared prims, so a
pass has a single look; K looks of the same episodes = K render runs, distinct
`--cam` names keep them all). The draw is applied at BOTH moments, so a knob needs
no tag saying which kind it is: sampled values are written onto the scene cfg
before the build (build-consumed fields — a table preset, a backdrop usd — take
effect there with no extra code), then handed to `scene.apply_visual_params` after
it (the LIVE subset: attribute writes, material rebinds; fields with no branch
there already did their work at build). The draw index and values land in
render_<cam>.json. The `--visual file.py` hook stays as the escape hatch for looks
that don't band-ify (`setup(env)` once after the build, `per_frame(env, t)` before
each rendered frame); this module stays generic.

Needs a running AppLauncher (see scripts/render.py). One scene per process — the
CLI splits multi-scene inputs into one subprocess per scene group.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .generation import _load, build_env

WARMUP_DEFAULT = 12  # throwaway renders per chunk start (temporal-denoiser flush)
_FAR_CLIP = 40.0  # camera far clip (m); env_spacing beyond it isolates envs pixel-exactly


# ----- episode discovery -------------------------------------------------------------------------
def batch_scene(batch_dir: Path) -> str:
    """The scene cell a batch was generated on (from its meta's `cell`)."""
    return json.loads((batch_dir / "meta.json").read_text())["cell"].split("/")[0]


def collect_episodes(gen_root: Path, batches: list[str] | None = None,
                     episodes: list[str] | None = None) -> list[Path]:
    """Episode dirs to replay: explicit `episodes`, else every ep of `batches`
    (names under data/), else every ep of every batch."""
    if episodes:
        out = [Path(e) for e in episodes]
        missing = [p for p in out if not (p / "traj.npz").is_file()]
        if missing:
            raise SystemExit(f"not episode dirs (no traj.npz): {missing}")
        return out
    data = gen_root / "data"
    dirs = [data / b for b in batches] if batches else sorted(d for d in data.iterdir() if d.is_dir())
    return [ep for d in dirs for ep in sorted(d.glob("ep_*")) if (ep / "traj.npz").is_file()]


def group_by_scene(eps: list[Path]) -> dict[str, list[Path]]:
    """{scene: [ep dirs]} — replay builds one env per scene, so inputs split here."""
    groups: dict[str, list[Path]] = {}
    for ep in eps:
        groups.setdefault(batch_scene(ep.parent), []).append(ep)
    return groups


# ----- state plumbing ----------------------------------------------------------------------------
def _unflatten(flat: dict) -> dict:
    """{"scene/bulbs": t, "robot/controller/0/prev_action": t} -> the nested set_states tree."""
    out: dict = {}
    for key, v in flat.items():
        node = out
        *parents, leaf = key.split("/")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = v
    return out


def _lookat_quat(eye, target):
    """World-convention (+X fwd, +Z up) look-at orientation as (w, x, y, z)."""
    import torch

    from isaaclab.utils.math import quat_from_matrix

    f = torch.tensor([t - e for t, e in zip(target, eye)], dtype=torch.float32)
    f = f / f.norm().clamp_min(1e-9)
    up = torch.tensor([0.0, 0.0, 1.0])
    left = torch.linalg.cross(up, f)
    if left.norm() < 1e-6:  # looking straight up/down: pick an arbitrary left
        left = torch.tensor([0.0, 1.0, 0.0])
    left = left / left.norm()
    u = torch.linalg.cross(f, left)
    return tuple(quat_from_matrix(torch.stack([f, left, u], dim=1)).tolist())


def _camera_cfg(name: str, eye, target, size, surface_z: float, focal: float):
    """Per-env TiledCamera under {ENV_REGEX_NS}: eye/target are env-origin-relative on
    the work surface (record_video.py's convention — the script adds surface_z itself).
    `focal` is USD mm on the default 20.955 aperture (24 ≈ 47° hFOV, 16 ≈ 66°)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg

    eye = (eye[0], eye[1], eye[2] + surface_z)
    target = (target[0], target[1], target[2] + surface_z)
    return name, TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        width=size[0], height=size[1],
        data_types=["rgb"],
        update_period=0.0,
        offset=TiledCameraCfg.OffsetCfg(pos=eye, rot=_lookat_quat(eye, target), convention="world"),
        spawn=sim_utils.PinholeCameraCfg(focal_length=focal, clipping_range=(0.05, _FAR_CLIP)),
    )


def build_replay_env(scene_dir: Path, num_envs: int, device: str, cam_name: str,
                     eye, target, size, focal: float, env_spacing: float,
                     visual_draw: int | None = None):
    """generation.build_env on the cell's LOCAL scene (nominal world — physics is
    overwritten every frame anyway), with the tiled camera injected into the scene's
    assets before the build.

    RTX renders ONE shared stage (no per-env world isolation like Madrona/ManiSkill),
    so standalone-robot frames come from geometry: `env_spacing` spreads the replay
    grid wider than the camera's far clip (`_FAR_CLIP`) — neighbors are clipped before
    rasterization, which is pixel-identical to independent worlds. The recorded states
    shift onto whatever grid replay builds, so the spacing is free. The shared ground
    plane is enlarged to keep a floor under the far-flung envs."""
    import math
    import re

    from robobench.core.registries import SCENES

    _load("datagen_local_scene", scene_dir / "scene" / "scene.py")
    scene_name = re.search(r'@SCENES\.register\("([\w.]+)"\)',
                           (scene_dir / "scene" / "scene.py").read_text()).group(1)
    scene_cls = SCENES.get(scene_name)
    surface_z = float(getattr(scene_cls().cfg, "surface_z", 0.0) or 0.0)
    # the visual draw is sampled BEFORE the build and written onto the scene cfg, so
    # build-consumed knobs (a table preset, a backdrop usd) take effect with no extra
    # code; live knobs are re-applied through scene.apply_visual_params after the build
    visual_values: dict = {}
    if visual_draw is not None:
        from .sampler import sample, scene_bands

        bands = scene_bands(scene_cls, scene_cls().cfg, attr="VISUAL_PARAMS")
        if not bands:
            raise SystemExit(f"--visual_draw: {scene_cls.__name__} declares no VISUAL_PARAMS bands")
        visual_values = sample(bands, visual_draw)
    name, cam_cfg = _camera_cfg(cam_name, eye, target, size, surface_z, focal)
    # grid extent + far-clip margin on every side
    ground_xy = (math.ceil(math.sqrt(num_envs)) - 1) * env_spacing + 2 * (_FAR_CLIP + 10.0)

    def assets_with_camera(self, _orig=scene_cls.assets):
        from isaaclab.sim import GroundPlaneCfg

        out = {**_orig(self), name: cam_cfg}
        for asset in out.values():
            spawn = getattr(asset, "spawn", None)
            if isinstance(spawn, GroundPlaneCfg) and max(spawn.size) < ground_xy:
                spawn.size = (ground_xy, ground_xy)
        return out

    scene_cls.assets = assets_with_camera
    try:
        env, gen, _, _ = build_env(scene_dir, num_envs, device, seed=0, nominal=True,
                                   env_spacing=env_spacing,
                                   scene_overrides=visual_values or None)
    finally:
        scene_cls.assets = assets_with_camera.__defaults__[0]
    return env, gen, visual_values


def _load_shifted(ep_dir: Path, base_pos, device):
    """Episode arrays as device tensors, positions shifted onto the replay slot's
    origin. Every last-dim-13 array is a world-frame root state (the engine's own
    convention: pos 0:3, quat 3:7, vels 7:13) — only those get the shift."""
    import numpy as np
    import torch

    data = {k: torch.as_tensor(v, device=device) for k, v in np.load(ep_dir / "traj.npz").items()}
    shift = base_pos - data["robot/root"][0, 0:3]
    for k, v in data.items():
        if k != "action" and v.shape[-1] == 13:
            v[..., 0:3] += shift
    meta = json.loads((ep_dir / "meta.json").read_text())
    meta["_ep_dir"] = str(ep_dir)
    return data, meta


# ----- the replay itself -------------------------------------------------------------------------
def replay_scene(gen_root: Path, scene: str, eps: list[Path], *, num_envs: int = 8,
                 fps: int = 30, size=(640, 480), eye=(1.0, -0.7, 0.5),
                 target=(0.22, 0.1, 0.18), focal: float = 16.0,
                 cam_name: str = "cam", warmup: int = WARMUP_DEFAULT,
                 save_frames: bool = True, preview: bool = True, preview_speed: float = 6.0,
                 crf: int = 26, max_frames: int = 0, visual: str | None = None,
                 visual_draw: int | None = None, env_spacing: float = 50.0,
                 device: str = "cuda:0") -> list[Path]:
    """Replay `eps` (all from `scene`) in chunks of `num_envs`, writing frames/previews
    into each episode dir. Returns the episode dirs rendered."""
    import imageio
    import numpy as np
    import torch

    scene_dir = gen_root / "scenes" / scene
    if num_envs > 1 and env_spacing <= _FAR_CLIP:
        print(f"[replay] WARNING: env_spacing {env_spacing} <= far clip {_FAR_CLIP} — "
              f"neighbor envs will appear in frames", flush=True)
    env, gen, visual_values = build_replay_env(scene_dir, num_envs, device, cam_name, eye,
                                               target, size, focal, env_spacing, visual_draw)
    cam = env.iscene.sensors[cam_name]
    if visual_values:
        # the draw already sat on the scene cfg through the build (build-consumed knobs
        # took effect there); the hook now applies the LIVE subset — attribute writes,
        # material rebinds — before the escape-hatch hook and before any warmup
        env.scene.apply_visual_params(env, visual_values)
        print(f"[replay {scene}] visual draw {visual_draw}: "
              + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in sorted(visual_values.items())), flush=True)
    hook = _load("datagen_visual_hook", Path(visual)) if visual else None
    if hook and hasattr(hook, "setup"):
        hook.setup(env)
    env.reset(seed=0)
    base_pos = env.robot.articulation.data.root_pos_w.clone()  # (E, 3) — the shift anchor
    stride = max(1, round(1.0 / (fps * env.dt)))
    pool = ThreadPoolExecutor(max_workers=8)

    def render_once():
        """One restored-state frame: post_step BEFORE the render (glow in-frame), then
        the sim tick that carries the transforms into the renderer."""
        env.iscene.update(env.dt)
        env.robot.post_step()
        env.scene.post_step()
        env.sim.step(render=True)
        cam.update(0.0, force_recompute=True)
        return cam.data.output["rgb"]

    # same-T episodes chunk together (a batch shares T); mixed chunks pad with the last state
    eps = sorted(eps, key=lambda p: (json.loads((p / "meta.json").read_text())["steps"], str(p)))
    done: list[Path] = []
    for lo in range(0, len(eps), num_envs):
        chunk = eps[lo:lo + num_envs]
        loaded = [_load_shifted(ep, base_pos[e], device) for e, ep in enumerate(chunk)]
        # controller state is dropped: stateless leaves flatten to nothing (a composite
        # can't restore a partial tree), and kinematic replay never applies an action
        keys = [k for k in loaded[0][0]
                if k != "action" and not k.startswith("robot/controller")]
        T = [d["robot/joint_pos"].shape[0] for d, _ in loaded]
        t_max = max(T)
        n_frames = len(range(0, t_max, stride))
        if max_frames:
            n_frames = min(n_frames, max_frames)
        print(f"[replay {scene}] chunk {lo // num_envs + 1}: {len(chunk)} eps, "
              f"{t_max} steps @ {1 / env.dt:.0f}Hz -> {n_frames} frames each", flush=True)

        writers, frame_dirs, indices = [], [], []
        for d, meta in loaded:
            ep_dir = Path(meta["_ep_dir"])
            img_dir = ep_dir / "imgs"
            img_dir.mkdir(exist_ok=True)
            frame_dirs.append(img_dir / cam_name)
            if save_frames:
                frame_dirs[-1].mkdir(exist_ok=True)
                for stale in frame_dirs[-1].glob("frame_*.jpg"):  # a re-render must not leave old tails
                    stale.unlink()
            w = None
            if preview:
                w = imageio.get_writer(str(img_dir / "preview.mp4"), fps=max(1, round(fps * preview_speed)),
                                       codec="libx264", quality=None, pixelformat="yuv420p",
                                       output_params=["-crf", str(crf), "-preset", "medium"])
            writers.append(w)
            indices.append([])

        def compose(t: int) -> dict:
            flat = {}
            for k in keys:
                rows = [d[k][min(t, T[i] - 1)] for i, (d, _) in enumerate(loaded)]
                rows += [rows[-1]] * (num_envs - len(rows))  # pad unused slots
                flat[k] = torch.stack(rows)
            return _unflatten(flat)

        env.set_states(compose(0))
        for _ in range(max(0, warmup)):
            render_once()

        for fi, t in enumerate(range(0, t_max, stride)):
            if max_frames and fi >= max_frames:
                break
            env.set_states(compose(t))
            if hook and hasattr(hook, "per_frame"):
                hook.per_frame(env, t)
            rgb = render_once()  # (E, H, W, 3) uint8, device
            frames = rgb.cpu().numpy()
            for i in range(len(chunk)):
                if t >= T[i]:  # this episode already ended — freeze, don't record
                    continue
                frame = frames[i]
                if save_frames:
                    pool.submit(imageio.imwrite, str(frame_dirs[i] / f"frame_{fi:06d}.jpg"),
                                frame, quality=90)
                if writers[i] is not None:
                    writers[i].append_data(frame)
                indices[i].append(t)

        pool.shutdown(wait=True)
        pool = ThreadPoolExecutor(max_workers=8)
        for i, (d, meta) in enumerate(loaded):
            if writers[i] is not None:
                writers[i].close()
            ep_dir = Path(meta["_ep_dir"])
            # per-cam contract, so K looks under K cam names coexist; drop a legacy
            # single-name render.json only if it was this camera's (now superseded)
            legacy = ep_dir / "imgs" / "render.json"
            if legacy.exists() and json.loads(legacy.read_text()).get("camera") == cam_name:
                legacy.unlink()
            (ep_dir / "imgs" / f"render_{cam_name}.json").write_text(json.dumps({
                "camera": cam_name, "size": list(size), "fps": fps, "stride": stride,
                "sim_dt": env.dt, "frame_indices": indices[i],
                "eye": list(eye), "target": list(target), "focal": focal,
                "env_spacing": env_spacing,
                "joint_names": list(env.robot.articulation.joint_names),
                "scene_description": env.scene.describe(),
                "success": meta.get("success"), "cell": meta.get("cell"),
                "visual_hook": visual, "visual_draw": visual_draw, "visual_values": visual_values,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }, indent=2) + "\n")
            done.append(ep_dir)
        del loaded
        torch.cuda.empty_cache() if device.startswith("cuda") else None
    pool.shutdown(wait=True)
    return done


def contact_sheet(batch_dir: Path, cam_name: str = "cam", cols: int = 5) -> Path | None:
    """episodes x time grid from the saved frames -> <batch>/replay_sheet.png."""
    import imageio
    import numpy as np

    rows = []
    for ep in sorted(batch_dir.glob("ep_*")):
        frames = sorted((ep / "imgs" / cam_name).glob("frame_*.jpg"))
        if not frames:
            continue
        picks = [frames[min(int(i * (len(frames) - 1) / (cols - 1)), len(frames) - 1)] for i in range(cols)]
        rows.append(np.concatenate([imageio.imread(p) for p in picks], axis=1))
    if not rows:
        return None
    sheet = batch_dir / "replay_sheet.png"
    imageio.imwrite(str(sheet), np.concatenate(rows, axis=0))
    return sheet

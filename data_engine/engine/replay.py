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

Cameras are DECLARED, two kinds with two owners: the scene's `CAMERAS` are external
views (env-origin-relative on the work surface — where to stand to see THIS
geometry), the robot's `CAMERAS` are ego views (a `link` key mounts the camera on
that body, eye/target in the link frame — it rides the link through the replayed
motion). All declared views render simultaneously in one pass, one TiledCamera
each; `--cams` selects a subset, and `--eye/--target` adds a one-time ad-hoc view
(named by `--cam`) for probing before numbers get written into a declaration.
Per-view `bands` randomize the pose with THE sampling grammar (engine/sampler.py)
on {eye,target}_{x,y,z}: nominal = the declared value, drawn PER EPISODE (Halton
index = the episode's global position in the run), each episode holding its own
fixed camera — set before the chunk's warmup, so the denoiser never sees it move.

Outputs, per (episode, view): `imgs/<view>/frame_%06d.jpg` (the dataset frames,
README's `imgs/` slot), `imgs/render_<view>.json` (frame indices + the ACTUAL
per-episode camera + joint names + visual draw — the export contract, per view so
several views/looks coexist), `imgs/preview_<view>.mp4` (time-lapsed, for
eyeballing). Per batch: `replay_sheet_<view>.png` (episodes x time contact sheet).

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


# ----- cameras: declared views + the ad-hoc override ----------------------------------------------
_CAM_BAND_KEYS = ("eye_x", "eye_y", "eye_z", "target_x", "target_y", "target_z")


def resolve_views(scene_cls, robot_cls, cams: list[str] | None, adhoc: dict | None) -> dict[str, dict]:
    """The views to render: the SCENE's `CAMERAS` (external, env-origin-relative on the work
    surface) merged with the ROBOT's `CAMERAS` (ego — a `link` key mounts the camera on that
    body, eye/target in the link frame), plus the CLI's one-time ad-hoc view. `cams` selects
    by name (None = all declared + the ad-hoc). Per-view `bands` use THE sampling grammar
    (engine/sampler.py) on {eye,target}_{x,y,z} — nominal = the declared value, drawn per
    episode; ego views can't band (their pose is the link's)."""
    from .sampler import _check_spec

    views = {n: {**spec, "link": None} for n, spec in getattr(scene_cls, "CAMERAS", {}).items()}
    for n, spec in getattr(robot_cls, "CAMERAS", {}).items():
        if n in views:
            raise SystemExit(f"camera '{n}' declared by both {scene_cls.__name__} and {robot_cls.__name__}")
        views[n] = {**spec, "robot_prim": robot_cls().prim_name}
    if adhoc:
        views[adhoc["name"]] = {**adhoc, "link": None}
    for n, v in views.items():
        bands = v.get("bands") or {}
        if bands and v.get("link"):
            raise SystemExit(f"camera '{n}': ego views (link-mounted) can't declare bands")
        unknown = set(bands) - set(_CAM_BAND_KEYS)
        if unknown:
            raise SystemExit(f"camera '{n}': bands on {sorted(unknown)} (allowed: {_CAM_BAND_KEYS})")
        v["bands"] = {k: _check_spec(f"CAMERAS['{n}']", k, s) for k, s in bands.items()}
        v.setdefault("focal", 16.0)
    if cams:
        missing = sorted(set(cams) - set(views))
        if missing:
            raise SystemExit(f"unknown cameras {missing}; declared: {sorted(views)}")
        views = {n: views[n] for n in cams}
    if not views:
        raise SystemExit("no cameras: the scene/robot declare none — pass --eye/--target for a one-time view")
    return views


def _camera_cfg(name: str, view: dict, size, surface_z: float):
    """Per-env TiledCamera for one view. External: under {ENV_REGEX_NS}/<name>, eye/target
    env-origin-relative on the work surface (surface_z added here). Ego: under the robot
    link ({ENV_REGEX_NS}/<Robot>/<link>/<name>), eye/target in the LINK frame — the camera
    rides the link through the replayed motion. `focal` is USD mm on the default 20.955
    aperture (24 ≈ 47° hFOV, 16 ≈ 66°)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg

    eye, target = view["eye"], view["target"]
    if view.get("link"):
        prim = "{ENV_REGEX_NS}/%s/%s/%s" % (view["robot_prim"], view["link"], name)
    else:
        prim = "{ENV_REGEX_NS}/" + name
        eye = (eye[0], eye[1], eye[2] + surface_z)
        target = (target[0], target[1], target[2] + surface_z)
    return TiledCameraCfg(
        prim_path=prim,
        width=size[0], height=size[1],
        data_types=["rgb"],
        update_period=0.0,
        offset=TiledCameraCfg.OffsetCfg(pos=tuple(eye), rot=_lookat_quat(eye, target), convention="world"),
        spawn=sim_utils.PinholeCameraCfg(focal_length=view["focal"], clipping_range=(0.05, _FAR_CLIP)),
    )


def build_replay_env(scene_dir: Path, num_envs: int, device: str,
                     cams: list[str] | None, adhoc: dict | None, size,
                     env_spacing: float, visual_draw: int | None = None):
    """generation.build_env on the cell's LOCAL scene (nominal world — physics is
    overwritten every frame anyway), with one tiled camera PER RESOLVED VIEW injected
    into the scene's assets before the build.

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
    import robobench
    import yaml

    robobench.discover()
    from robobench.core.registries import ENVS, ROBOTS

    scene_cls = SCENES.get(scene_name)
    surface_z = float(getattr(scene_cls().cfg, "surface_z", 0.0) or 0.0)
    gen = yaml.safe_load((scene_dir.parents[1] / "gen.yaml").read_text())
    robot_cls = ROBOTS.get(ENVS.get(gen["preset"])().robot)
    views = resolve_views(scene_cls, robot_cls, cams, adhoc)
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
    cam_cfgs = {n: _camera_cfg(n, v, size, surface_z) for n, v in views.items()}
    # entities spawn in insertion order (scene assets, then robot assets): ego cameras
    # need the robot LINK prim to exist, so they inject into the ROBOT's assets — after
    # the Robot entry — while external views ride the scene's
    static_cfgs = {n: c for n, c in cam_cfgs.items() if not views[n].get("link")}
    ego_cfgs = {n: c for n, c in cam_cfgs.items() if views[n].get("link")}
    # grid extent + far-clip margin on every side
    ground_xy = (math.ceil(math.sqrt(num_envs)) - 1) * env_spacing + 2 * (_FAR_CLIP + 10.0)

    def assets_with_camera(self, _orig=scene_cls.assets):
        from isaaclab.sim import GroundPlaneCfg

        out = {**_orig(self), **static_cfgs}
        for asset in out.values():
            spawn = getattr(asset, "spawn", None)
            if isinstance(spawn, GroundPlaneCfg) and max(spawn.size) < ground_xy:
                spawn.size = (ground_xy, ground_xy)
        return out

    def assets_with_ego(self, _orig=robot_cls.assets):
        return {**_orig(self), **ego_cfgs}

    scene_cls.assets = assets_with_camera
    robot_cls.assets = assets_with_ego
    try:
        env, gen, _, _ = build_env(scene_dir, num_envs, device, seed=0, nominal=True,
                                   env_spacing=env_spacing,
                                   scene_overrides=visual_values or None)
    finally:
        scene_cls.assets = assets_with_camera.__defaults__[0]
        robot_cls.assets = assets_with_ego.__defaults__[0]
    return env, gen, visual_values, views, surface_z


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
                 fps: int | None = None, size=(640, 480), cams: list[str] | None = None,
                 adhoc: dict | None = None, warmup: int = WARMUP_DEFAULT,
                 save_frames: bool = True, preview: bool = True, preview_speed: float = 6.0,
                 crf: int = 26, max_frames: int = 0, visual: str | None = None,
                 visual_draw: int | None = None, env_spacing: float = 50.0,
                 device: str = "cuda:0") -> list[Path]:
    """Replay `eps` (all from `scene`) in chunks of `num_envs`, rendering every resolved
    view each frame and writing frames/previews into each episode dir. Returns the
    episode dirs rendered.

    Traj rows are one per env.step = one per control latch (row_dt = sim_dt x the
    recorded decimation, both stamped in each ep's meta.json). fps=None (the default)
    renders one frame PER LATCH — fps = the batch's control rate, the matched regime
    for VLA export; pass fps to subsample (e.g. --fps 30 on a 240 Hz batch)."""
    import imageio
    import numpy as np
    import torch

    scene_dir = gen_root / "scenes" / scene
    if num_envs > 1 and env_spacing <= _FAR_CLIP:
        print(f"[replay] WARNING: env_spacing {env_spacing} <= far clip {_FAR_CLIP} — "
              f"neighbor envs will appear in frames", flush=True)
    env, gen, visual_values, views, surface_z = build_replay_env(
        scene_dir, num_envs, device, cams, adhoc, size, env_spacing, visual_draw)
    sensors = {n: env.iscene.sensors[n] for n in views}
    print(f"[replay {scene}] views: " + ", ".join(
        f"{n} (ego on {v['link']})" if v.get("link") else n for n, v in views.items()), flush=True)
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
    pool = ThreadPoolExecutor(max_workers=8)

    def render_once():
        """One restored-state frame: post_step BEFORE the render (glow in-frame), then
        the sim tick that carries the transforms into the renderer."""
        env.iscene.update(env.dt)
        env.robot.post_step()
        env.scene.post_step()
        env.sim.step(render=True)
        out = {}
        for n, cam in sensors.items():
            cam.update(0.0, force_recompute=True)
            out[n] = cam.data.output["rgb"]
        return out

    env_origins = env.iscene.env_origins  # (E, 3), device

    def place_banded_views(lo: int, n_eps: int) -> dict[str, list]:
        """Draw each banded view's per-episode pose (THE sampler, index = the episode's
        global position in this run) and set every env slot's camera before the chunk's
        warmup — each episode holds its own fixed camera for its whole duration. Returns
        {view: [(eye, target) per episode]} for the render contract."""
        from .sampler import sample

        placed: dict[str, list] = {}
        for n, v in views.items():
            if not v["bands"]:
                continue
            poses, quats, actual = [], [], []
            for i in range(num_envs):
                g = lo + min(i, n_eps - 1)  # padded slots reuse the last episode's draw
                draw = sample(v["bands"], g)
                e = [draw.get(f"eye_{a}", x) for a, x in zip("xyz", v["eye"])]
                t = [draw.get(f"target_{a}", x) for a, x in zip("xyz", v["target"])]
                if i < n_eps:
                    actual.append((e, t))
                e_w = [e[0], e[1], e[2] + surface_z]
                poses.append(env_origins[i] + torch.tensor(e_w, device=device))
                quats.append(torch.tensor(_lookat_quat(e, t), device=device))
            sensors[n].set_world_poses(torch.stack(poses), torch.stack(quats),
                                       convention="world")
            placed[n] = actual
        return placed

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
        # a traj row = one env.step = one control latch; the recorded control rate sets
        # the frame clock, NOT the rebuilt env's physics dt (the solve may have decimated)
        row_dts = {m.get("sim_dt", env.dt) * m.get("decimation", 1) for _, m in loaded}
        if len(row_dts) > 1:
            raise SystemExit(f"[replay {scene}] chunk mixes control rates "
                             f"({sorted(1 / d for d in row_dts)}) — render the batches separately")
        row_dt = row_dts.pop()
        stride = max(1, round(1.0 / (fps * row_dt))) if fps else 1
        fps_out = int(round(1.0 / (row_dt * stride)))
        n_frames = len(range(0, t_max, stride))
        if max_frames:
            n_frames = min(n_frames, max_frames)
        print(f"[replay {scene}] chunk {lo // num_envs + 1}: {len(chunk)} eps, "
              f"{t_max} rows @ {1 / row_dt:.0f}Hz control -> {n_frames} frames each "
              f"@ {fps_out}fps", flush=True)

        # per (episode, view): frame dir, preview writer; frame indices are per episode
        writers, frame_dirs, indices = [], [], []
        for d, meta in loaded:
            ep_dir = Path(meta["_ep_dir"])
            img_dir = ep_dir / "imgs"
            img_dir.mkdir(exist_ok=True)
            fd, ws = {}, {}
            for n in views:
                fd[n] = img_dir / n
                if save_frames:
                    fd[n].mkdir(exist_ok=True)
                    for stale in fd[n].glob("frame_*.jpg"):  # a re-render must not leave old tails
                        stale.unlink()
                ws[n] = None
                if preview:
                    ws[n] = imageio.get_writer(str(img_dir / f"preview_{n}.mp4"),
                                               fps=max(1, round(fps_out * preview_speed)),
                                               codec="libx264", quality=None, pixelformat="yuv420p",
                                               output_params=["-crf", str(crf), "-preset", "medium"])
            frame_dirs.append(fd)
            writers.append(ws)
            indices.append([])

        def compose(t: int) -> dict:
            flat = {}
            for k in keys:
                rows = [d[k][min(t, T[i] - 1)] for i, (d, _) in enumerate(loaded)]
                rows += [rows[-1]] * (num_envs - len(rows))  # pad unused slots
                flat[k] = torch.stack(rows)
            return _unflatten(flat)

        placed = place_banded_views(lo, len(chunk))  # per-episode camera draws, pre-warmup
        env.set_states(compose(0))
        for _ in range(max(0, warmup)):
            render_once()

        for fi, t in enumerate(range(0, t_max, stride)):
            if max_frames and fi >= max_frames:
                break
            env.set_states(compose(t))
            if hook and hasattr(hook, "per_frame"):
                hook.per_frame(env, t)
            rgbs = {n: r.cpu().numpy() for n, r in render_once().items()}  # each (E, H, W, 3) uint8
            for i in range(len(chunk)):
                if t >= T[i]:  # this episode already ended — freeze, don't record
                    continue
                for n in views:
                    frame = rgbs[n][i]
                    if save_frames:
                        pool.submit(imageio.imwrite, str(frame_dirs[i][n] / f"frame_{fi:06d}.jpg"),
                                    frame, quality=90)
                    if writers[i][n] is not None:
                        writers[i][n].append_data(frame)
                indices[i].append(t)

        pool.shutdown(wait=True)
        pool = ThreadPoolExecutor(max_workers=8)
        for i, (d, meta) in enumerate(loaded):
            ep_dir = Path(meta["_ep_dir"])
            for n, v in views.items():
                if writers[i][n] is not None:
                    writers[i][n].close()
                eye, target = (placed[n][i] if n in placed else (v["eye"], v["target"]))
                (ep_dir / "imgs" / f"render_{n}.json").write_text(json.dumps({
                    "camera": n, "size": list(size), "fps": fps_out, "stride": stride,
                    "sim_dt": env.dt, "step_dt": row_dt, "frame_indices": indices[i],
                    "eye": list(eye), "target": list(target), "focal": v["focal"],
                    "link": v.get("link"), "cam_draw": (lo + i if n in placed else None),
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
    return done, list(views)


def contact_sheet(batch_dir: Path, view: str, cols: int = 5) -> Path | None:
    """episodes x time grid from the saved frames -> <batch>/replay_sheet_<view>.png."""
    import imageio
    import numpy as np

    rows = []
    for ep in sorted(batch_dir.glob("ep_*")):
        frames = sorted((ep / "imgs" / view).glob("frame_*.jpg"))
        if not frames:
            continue
        picks = [frames[min(int(i * (len(frames) - 1) / (cols - 1)), len(frames) - 1)] for i in range(cols)]
        rows.append(np.concatenate([imageio.imread(p) for p in picks], axis=1))
    if not rows:
        return None
    sheet = batch_dir / f"replay_sheet_{view}.png"
    imageio.imwrite(str(sheet), np.concatenate(rows, axis=0))
    return sheet

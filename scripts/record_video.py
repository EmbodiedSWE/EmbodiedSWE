"""Record any smoke / experiment run to an mp4, without modifying it.

Runs the target headless and grabs the viewport camera every ~1/fps of sim time into the mp4,
so the video plays at real-time speed. Fully headless — safe to run next to another Isaac Sim.

The target can be given three ways:
  - a bare smoke name          -> resolved under --package (default: the assembly smokes pkg)
  - a full dotted module path  -> run as `python -m <module>`
  - a path to a .py file       -> run as a standalone script (e.g. an experiments/ solver)

Run from the repo root (the repo is pip-installed editable, so no PYTHONPATH needed):

    # an assembly smoke (bare name):
    .venv/bin/python scripts/record_video.py so101_smoke

    # an experiment solver (.py path), only the first 20s, higher quality:
    .venv/bin/python scripts/record_video.py \
        experiments/../solve.py --max-seconds 20 --quality 9

    # a smoke in another suite (dotted module), custom camera + output:
    .venv/bin/python scripts/record_video.py \
        robobench.suites.assembly.smokes.pc_gpu_smoke \
        --video videos/gpu.mp4 --eye 0.9 -1.1 0.65 --target-at 0.3 -0.05 0.1
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import runpy
import sys

import numpy as np

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("target", help="smoke name (bare, resolved under --package), a dotted module "
                                    "path, or a path to a standalone .py script")
parser.add_argument("--package", default="robobench.suites.assembly.smokes",
                    help="package a bare smoke name is resolved under")
parser.add_argument("--video", default=None,
                    help="output mp4 path (default: videos/<target>-<timestamp>.mp4)")
parser.add_argument("--fps", type=int, default=30, help="capture rate in sim time = playback rate; 0 = auto "
                                                         "(the control rate, capped at 30, so clips play in real time)")
parser.add_argument("--quality", type=int, default=8, metavar="0-10",
                    help="imageio mp4 quality (0=worst/smallest, 10=best/largest); "
                         "resolution (--size) is the other big lever")
parser.add_argument("--size", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
parser.add_argument("--max-seconds", type=float, default=None, dest="max_seconds",
                    help="stop recording and exit after this many seconds of VIDEO (playback = "
                         "sim time); omit to record the whole run")
parser.add_argument("--eye", type=float, nargs=3, default=(0.9, -1.1, 0.65),
                    help="camera eye, relative to the env origin on the work surface")
parser.add_argument("--target-at", type=float, nargs=3, default=(0.30, -0.05, 0.10), dest="target_at",
                    help="camera look-at point, same frame")
parser.add_argument("--camera-frame", choices=("surface", "origin"), default="surface", dest="camera_frame",
                    help="what --eye/--target-at are relative to: the env origin lifted to the scene's "
                         "surface_z (default) or the bare env origin (the backdrop specs' convention)")
a, rest = parser.parse_known_args()  # extra args pass through to the target

_base = os.path.splitext(os.path.basename(a.target))[0]  # strip dir + .py
smoke_name = _base.rsplit(".", 1)[-1]                     # last part of a dotted module
if a.video:
    video_path = os.path.abspath(a.video)
else:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    video_path = os.path.abspath(os.path.join("videos", f"{smoke_name}-{stamp}.mp4"))
os.makedirs(os.path.dirname(video_path), exist_ok=True)

import robobench.core.env as _envmod  # noqa: E402  (app-free, safe to import now)
import robobench.core.config as _cfgmod  # noqa: E402

# COSIGEN_SEED=<int> forces the env seed of every EnvCfg.build in the run (repeat a script over seeds
# without editing it).
if os.environ.get("COSIGEN_SEED"):
    _forced_seed = int(os.environ["COSIGEN_SEED"])
    _orig_build = _cfgmod.EnvCfg.build

    def _build_seeded(self, **overrides):
        overrides["seed"] = _forced_seed
        print(f"[record] env seed forced to {_forced_seed} (COSIGEN_SEED)", flush=True)
        return _orig_build(self, **overrides)

    _cfgmod.EnvCfg.build = _build_seeded

# Kit can mis-decode a newer driver and silently disable RTX; with RTX off, replicator's
# render graph cannot be wired and annot.attach() dies with "Unable to write from unknown
# dtype, kind=f, size=0". The egg_carton smoke — the recording path that works on this
# build — passes this Kit flag for exactly that reason, but a solution builds its own
# AppLauncher and cannot know about it, so inject it for every AppLauncher in this process.
_RTX_FIX = "--/rtx/verifyDriverVersion/enabled=false"

try:
    from robobench.core.isaaclab_compat import is_isaaclab_30 as _is30
    _IS_30 = bool(_is30())
except Exception:  # noqa: BLE001  (main's 2.3 tree has no compat module)
    _IS_30 = False


def _bind_fabric_30(env) -> bool:
    """Isaac Lab 3.0 publishes physics poses to the renderer only through a visualizer's Fabric
    binding (KitVisualizer.initialize -> FabricBackend.bind_transforms); a headless run has no
    visualizer, so without this the rendered stage never moves. Bind the sim's scene-data provider
    the same way."""
    if not _IS_30 or _rec.get("fabric") is not None:
        return _rec.get("fabric") is not None
    try:
        from isaaclab_physx.renderers.fabric import FabricBackendCfg

        sim = env.sim
        provider = sim.get_scene_data_provider()
        fabric = sim.get_or_create_backend(FabricBackendCfg(stage=sim.stage, device=sim.device))
        fabric.bind_transforms(provider)
        _rec["fabric"] = (fabric, provider, sim)
        print("[record] Fabric transform publication bound (Isaac Lab 3.0 headless)", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[record] Fabric binding not ready yet: {exc!r}", flush=True)
        return False


def _pump_kit() -> None:
    """Isaac Lab 3.0: `sim.render()` only drives registered visualizers, and a headless run has
    none, so nothing publishes poses to Fabric or pumps the Kit app loop that RTX + the
    replicator annotator render on. Do both explicitly around every capture (what
    KitVisualizer.step + render_rgb_array do)."""
    if _IS_30:
        fp = _rec.get("fabric")
        if fp is None and _rec.get("env") is not None:
            _bind_fabric_30(_rec["env"])
            fp = _rec.get("fabric")
        if fp is not None:
            fabric, provider, sim = fp
            fabric.update_transforms(provider)
            fabric.update_geometries(provider, sim.render_generation)
        import omni.kit.app

        omni.kit.app.get_app().update()


def _patch_app_launcher() -> None:
    # pinocchio MUST register its eigenpy converters before anything pulls in Kit; importing
    # isaaclab.app only defines the class (Kit starts on construction), but keep the proven
    # order anyway so this patch can never be the thing that breaks pink_ik.
    try:
        import pinocchio  # noqa: F401
    except ImportError:
        pass
    # isaaclab.app is a namespace package with no attributes until the submodule loads, so
    # import the CLASS. Patching the class object is what matters: the target's own
    # `from isaaclab.app import AppLauncher` binds this same object.
    from isaaclab.app import AppLauncher

    _orig_init = AppLauncher.__init__

    def _init(self, launcher_args=None, **kw):
        # a target built with `add_app_launcher_args` already carries `kit_args` in its
        # Namespace; AppLauncher refuses the same key in both places, so merge into
        # whichever side has it
        ns_has = launcher_args is not None and not isinstance(launcher_args, dict) \
            and hasattr(launcher_args, "kit_args")
        if ns_has:
            existing = getattr(launcher_args, "kit_args") or ""
            if _RTX_FIX not in existing:
                launcher_args.kit_args = f"{existing} {_RTX_FIX}".strip()
        else:
            existing = kw.get("kit_args") or ""
            if _RTX_FIX not in existing:
                kw["kit_args"] = f"{existing} {_RTX_FIX}".strip()
        # Cameras on, on both stacks: a script that builds `AppLauncher(headless=True)` without a
        # parser never sees the injected --enable_cameras argv, and without cameras headless Kit
        # never loads omni.replicator (3.0 has no such flag at all; launch_simulation reads the key).
        if ns_has:
            setattr(launcher_args, "enable_cameras", True)
        elif launcher_args is not None and not isinstance(launcher_args, dict):
            setattr(launcher_args, "enable_cameras", True)
        else:
            kw["enable_cameras"] = True
        ret = _orig_init(self, launcher_args, **kw)
        # Scripts that end with `app.close(); os._exit(0)`: on Isaac Lab 3.0 the close tears the
        # process down before any Python exit hook runs, so flush the mp4 and print the verdict
        # BEFORE the close.
        try:
            app = self.app
            _orig_close = app.close

            def _close_flushed(*args_, **kwargs_):
                _flush_writer()
                return _orig_close(*args_, **kwargs_)

            app.close = _close_flushed
        except Exception as exc:  # noqa: BLE001  (2.3's pybind IApp refuses new attributes; os._exit covers it there)
            print(f"[record] app.close not wrapped ({exc!r})", flush=True)
        return ret

    AppLauncher.__init__ = _init
    print(f"[record] AppLauncher patched with {_RTX_FIX}", flush=True)


_patch_app_launcher()

_rec = {"writer": None, "annot": None, "every": 1, "count": 0, "frames": 0, "max_frames": None}


_RECORD_CAM = "/World/RecordCam"


def _record_camera_path_30(env, eye, target) -> str:
    """Isaac Lab 3.0: `sim.set_camera_view` only forwards to registered visualizers (a headless
    run has none) and the pre-existing Kit viewport camera ignores USD pose edits, so render
    from a dedicated camera prim posed the way KitVisualizer's `_set_usd_camera_pose` does
    (its 3.0 quaternions are xyzw). Returns the camera prim path for the render product."""
    if not _IS_30:
        return "/OmniverseKit_Persp"
    import torch
    from pxr import Gf, UsdGeom
    from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

    stage = env.sim.stage
    eye_t = torch.as_tensor(eye, dtype=torch.float32).reshape(1, 3)
    look_t = torch.as_tensor(target, dtype=torch.float32).reshape(1, 3)
    rot = create_rotation_matrix_from_view(eye_t, look_t, up_axis=UsdGeom.GetStageUpAxis(stage), device="cpu")
    q = quat_from_matrix(rot)[0]  # xyzw
    cam = UsdGeom.Camera.Define(stage, _RECORD_CAM)
    persp = UsdGeom.Camera(stage.GetPrimAtPath("/OmniverseKit_Persp"))
    if persp:  # same lens as the Kit viewport camera the 2.3 recordings used
        for get, set_ in ((persp.GetFocalLengthAttr, cam.CreateFocalLengthAttr),
                          (persp.GetHorizontalApertureAttr, cam.CreateHorizontalApertureAttr),
                          (persp.GetVerticalApertureAttr, cam.CreateVerticalApertureAttr),
                          (persp.GetClippingRangeAttr, cam.CreateClippingRangeAttr)):
            if get().HasValue():
                set_(get().Get())
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.ClearXformOpOrder()
    xf.SetResetXformStack(True)
    prim = cam.GetPrim()
    t_attr = prim.GetAttribute("xformOp:translate")
    t_op = UsdGeom.XformOp(t_attr) if t_attr else xf.AddTranslateOp()
    o_attr = prim.GetAttribute("xformOp:orient")
    o_op = UsdGeom.XformOp(o_attr) if o_attr else xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    xf.SetXformOpOrder([t_op, o_op], xf.GetResetXformStack())
    t_op.Set(Gf.Vec3d(*[float(v) for v in eye]))
    o_op.Set(Gf.Quatd(float(q[3]), Gf.Vec3d(float(q[0]), float(q[1]), float(q[2]))))
    return _RECORD_CAM


def _ensure(env) -> None:
    """Set up on the first step, once the app and env exist."""
    if _rec["writer"] is not None:
        return
    import imageio
    import omni.replicator.core as rep

    anchor = env.iscene.env_origins[0].tolist()
    if a.camera_frame == "surface":
        anchor[2] += float(getattr(env.scene.cfg, "surface_z", 0.0))
    # Order below mirrors the packing.egg_carton smoke, which is the recording path proven
    # to work on this Isaac build (isaacsim 5.1 + omni.replicator.core 1.12.27). Attaching
    # the annotator without first selecting a render mode and warming the graph dies with
    # "TypeError: Unable to write from unknown dtype, kind=f, size=0" inside
    # omni.syntheticdata's intergraph dependency wiring (measured 2026-08-30).
    if hasattr(env.sim, "set_render_mode"):  # absent on isaaclab develop (Newton venv) — attach works without it there
        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
    eye_w = [anchor[i] + a.eye[i] for i in range(3)]
    target_w = [anchor[i] + a.target_at[i] for i in range(3)]
    env.sim.set_camera_view(eye=eye_w, target=target_w)
    cam_path = _record_camera_path_30(env, eye_w, target_w)
    _rec["env"] = env
    _bind_fabric_30(env)
    rp = rep.create.render_product(cam_path, tuple(a.size))
    annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    annot.attach([rp])
    for _ in range(6):
        env.sim.render()
        _pump_kit()
    warm = np.asarray(annot.get_data())
    print(f"[record] camera warmup frame shape={getattr(warm, 'shape', None)}", flush=True)
    sim_dt = env.dt * env.robot.control_period  # sim time of one step() call
    if a.fps <= 0:
        a.fps = max(1, min(30, int(round(1.0 / sim_dt))))
    _rec["annot"] = annot
    _rec["every"] = max(1, round(1.0 / (sim_dt * a.fps)))
    if a.max_seconds is not None:
        _rec["max_frames"] = max(1, round(a.max_seconds * a.fps))
    _rec["writer"] = imageio.get_writer(video_path, fps=a.fps, quality=a.quality)
    cap = f", stop @ {_rec['max_frames']} frames ({a.max_seconds}s)" if _rec["max_frames"] else ""
    print(f"[record] {video_path}: {a.size[0]}x{a.size[1]} @ {a.fps} fps q{a.quality} "
          f"(1 frame / {_rec['every']} steps){cap}", flush=True)


_orig_step = _envmod.BaseEnv.step


def _step(self, action, render: bool = False) -> None:
    _ensure(self)
    _rec["env"] = self
    _rec["count"] += 1
    grab = _rec["count"] % _rec["every"] == 0
    _orig_step(self, action, render=render or grab)
    if grab:
        _pump_kit()
        frame = np.asarray(_rec["annot"].get_data())
        if frame.size:  # the first few frames can come back empty
            _rec["writer"].append_data(frame[..., :3])
            _rec["frames"] += 1
            if _rec["max_frames"] is not None and _rec["frames"] >= _rec["max_frames"]:
                print(f"[record] --max-seconds reached at {_rec['frames']} frames, stopping.",
                      flush=True)
                _exit_flush(0)  # flush the mp4, then hard-exit the whole run


_envmod.BaseEnv.step = _step

_orig_reset = _envmod.BaseEnv.reset


def _reset(self, *args_, **kwargs_):
    """Attach the recorder BEFORE the first reset runs.

    omni.syntheticdata refuses to wire its render graph once the simulation has advanced:
    attaching from inside the first step() raises "Unable to write from unknown dtype".
    The egg_carton smoke — the recording path that works on this build — attaches while the
    env is freshly built and un-reset, so do the same here.
    """
    _ensure(self)
    return _orig_reset(self, *args_, **kwargs_)


_envmod.BaseEnv.reset = _reset

_orig_close = _envmod.BaseEnv.close


def _close(self) -> None:
    """Flush the mp4 before teardown (all frames are in by now), then guard against Kit's
    close() sometimes hanging with a 30s force-exit watchdog."""
    _flush_writer()
    import threading

    watchdog = threading.Timer(30.0, lambda: _os_exit(0))
    watchdog.daemon = True
    watchdog.start()
    _orig_close(self)


_envmod.BaseEnv.close = _close

_os_exit = os._exit


def _print_verdict() -> None:
    """One uniform verdict line per run, read off the last env stepped (`scene.success()` /
    `scene.score()`). Printed once, before the mp4 is flushed."""
    env = _rec.get("env")
    if env is None or _rec.get("verdict_done"):
        return
    _rec["verdict_done"] = True
    try:
        success = env.scene.success().tolist()
    except Exception as exc:  # noqa: BLE001
        success = f"n/a({type(exc).__name__})"
    try:
        score = env.scene.score().tolist()
    except Exception as exc:  # noqa: BLE001
        score = f"n/a({type(exc).__name__})"
    print(f"[record] VERDICT success={success} score={score} steps={_rec['count']} frames={_rec['frames']}",
          flush=True)


def _flush_writer() -> None:
    _print_verdict()
    w, _rec["writer"] = _rec["writer"], None
    if w is not None:
        w.close()
        print(f"[record] video written: {video_path}", flush=True)


def _exit_flush(code: int) -> None:
    try:
        _flush_writer()
    finally:
        _os_exit(code)


os._exit = _exit_flush

sys.argv = [smoke_name, "--headless", "--enable_cameras"] + rest
try:
    if a.target.endswith(".py"):
        runpy.run_path(a.target, run_name="__main__")
    elif "." in a.target:
        runpy.run_module(a.target, run_name="__main__")
    else:
        runpy.run_module(f"{a.package}.{a.target}", run_name="__main__")
except BaseException:
    # Crash path. Kit's teardown can wedge indefinitely after an exception, which turns a
    # failed render into a full pod-timeout of wasted GPU time (measured: 60 min for a
    # fault that surfaced in seconds). Print the fault, flush whatever was captured, and
    # hard-exit so the failure is cheap and legible.
    import traceback

    traceback.print_exc()
    _flush_writer()
    sys.stdout.flush()
    _os_exit(1)
else:
    _flush_writer()  # normal path also exits through _exit_flush above

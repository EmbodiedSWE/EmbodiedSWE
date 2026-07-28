"""CoSiGen session layer: suite registry, env/camera construction, prompt
assembly (make_prompt), the per-turn namespace, and run_policy — the exec
entrypoint the render server drives."""
from __future__ import annotations

import contextlib
import copy
import io
import os
import re as _re
import subprocess
import time as _time
import traceback
import uuid as _uuid

import numpy as np
import torch

import robobench
from robobench.core import ENVS

from cosigen_config import _CKPT_FNS, _FEATURES_OFF, _OPT_FNS
from cosigen_prompts import _strip_doc_section
from cosigen_apis import *  # noqa: F401,F403 -- suite classes
from cosigen_apis import (AssemblyApi, GenericSceneApi, PackingApi)


SUITES: dict[str, type] = {"assembly": AssemblyApi, "packing": PackingApi,
                           "articulated": GenericSceneApi, "tool_use": GenericSceneApi}


def _suite_of(env_name: str) -> str:
    return env_name.split(".", 1)[0]


def _load_legacy_null_api() -> type:
    """Load the retired no-robot api from eval/legacy by file path (legacy is not a
    package and does not hot-reload; it ships with the tarball only)."""
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parent / "legacy" / "cosigen_null.py"
    if not p.is_file():
        raise RuntimeError(
            "null-embodiment sessions were retired to eval/legacy/cosigen_null.py "
            f"(2026-07-26) and this deployment does not ship it ({p} missing)")
    spec = importlib.util.spec_from_file_location("cosigen_legacy_null", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NullSceneApi


# Assembly scenes whose objects come in the loose/fixed pair FrankaTabletopApi reads
# (see its _pairs): nuts+bolts, bulbs+sockets. Bimanual ikea has its own api below.
_PAIR_SCENES = {"nut_thread", "bulb"}


def resolve_api(env_name: str) -> type:
    """Pick the control-API class for a registered env name
    '<suite>.<scene>.<robot>.<mode>' (embodiment-specific subclasses first,
    then the suite mapping)."""
    parts = env_name.split(".")
    suite = _suite_of(env_name)
    scene = parts[1] if len(parts) >= 2 else ""
    robot = parts[2] if len(parts) >= 3 else ""
    # Null embodiment (sim_gen constructed problems, '<suite>.<scene>' with no robot
    # segment): RETIRED to legacy/cosigen_null.py 2026-07-26 (user decision) — no
    # session ever ran on it. Robot-bound eval variants ('sim_gen.<scene>.franka')
    # carry a robot segment and fall through to the arm apis below.
    if robot == "null" or (suite in ("sim_gen", "simgen") and len(parts) < 3):
        return _load_legacy_null_api()
    # Single gripper ARMS all share the Franka-style virtual-target api (they expose
    # EE_BODY / GRIPPER_JOINTS and an osc/impedance task-space mode): franka, piper, wxai.
    # Bimanual MultiRobot presets (bimanual_franka / aloha / bimanual_piper) need a
    # dedicated api (two EE targets over concatenated action slices) — add it HERE
    # (additively) before pointing a server at a bimanual env.
    # franka_ped = the same FrankaRobot class registered under a second name for the
    # pedestal whiteboard binding (2026-07-17, whiteboard agent) — same api family.
    if robot == "kuka_allegro":
        return KukaAllegroApi
    if robot in ("franka", "piper", "wxai", "franka_ped", "franka_ped2", "franka_ped3"):
        if suite == "packing":
            return FrankaPackingApi
        if suite == "assembly":
            # FrankaTabletopApi reads objects as a loose/fixed PAIR (nuts+bolts,
            # bulbs+sockets) and refuses anything else: booting
            # assembly.pc_ram.franka.osc died on "unsupported franka scene
            # PcRamAssemblyScene". Every other assembly scene — the PC builds, chair,
            # stacking toy, allen bolt, so101, and ikea under one arm — goes through the
            # generic surface, which enumerates the scene's own assets and uses the
            # scene's success predicate.
            return FrankaTabletopApi if scene in _PAIR_SCENES else FrankaGenericApi
        return FrankaGenericApi
    if robot in ("bimanual_franka", "aloha", "bimanual_piper") and suite == "packing":
        return BimanualFrankaPackingApi
    if robot in ("bimanual_franka", "aloha", "bimanual_piper") and suite in (
            "articulated", "tool_use"):
        return FrankaBimanualApi
    if robot in ("bimanual_franka", "aloha", "bimanual_piper") and suite == "assembly":
        # IkeaBimanualApi reads legs+studs, so it only fits the ikea scene; the other
        # assembly scenes get the bimanual generic surface rather than a wrong binding.
        return IkeaBimanualApi if scene == "ikea_table" else FrankaBimanualApi
    if robot == "gr1t2":
        if suite == "packing":
            return Gr1t2PackingApi
        if suite in ("articulated", "tool_use"):
            return Gr1t2GenericApi
    return SUITES[suite]


# --------------------------- harness helpers ---------------------------
def build_env(env_name: str, max_steps: int = 3000, num_envs: int = 1):
    """Build a robobench env by registered name + its suite control API. Requires the
    Isaac app to be already booted. Returns (env, api). num_envs > 1 adds mirrored
    replica envs used as the RL-training batch (env 0 stays authoritative)."""
    robobench.discover()
    if env_name not in ENVS.list() and env_name.split(".")[0] in ("sim_gen", "simgen"):
        # sim_gen constructed problems live OUTSIDE robobench/suites (CoSiGen/sim_gen/
        # tasks/<dir>/scene.py, dir name != scene name); importing a task's scene module
        # runs its @SCENES.register + register_env. Import every task package (import-
        # light per robobench conventions), skipping broken/incomplete ones.
        import importlib
        from pathlib import Path
        tasks_root = Path(__file__).resolve().parents[1] / "sim_gen" / "tasks"
        for d in sorted(p for p in tasks_root.iterdir() if (p / "scene.py").is_file()):
            try:
                importlib.import_module(f"sim_gen.tasks.{d.name}.scene")
            except Exception as exc:  # noqa: BLE001 -- rejected/incomplete packages exist
                print(f"[build_env] skipping sim_gen task {d.name}: {exc!r}", flush=True)
    if (env_name not in ENVS.list() and env_name.endswith(".franka")
            and env_name.split(".")[0] in ("sim_gen", "simgen")):
        # Franka-bound EVAL variant of a constructed problem (2026-07-23 franka eval
        # pipeline): the task's scene.py registered only the null binding
        # ('<suite>.<scene>'); clone that EnvCfg and swap the embodiment. robobench
        # composes any scene with any registered robot, so no scene edits are needed.
        base_name = env_name.rsplit(".", 1)[0]
        if base_name in ENVS.list():
            import dataclasses

            from robobench.core import register_env

            base_factory = ENVS.get(base_name)
            register_env(env_name.split(".")[0],
                         lambda: dataclasses.replace(base_factory(), robot="franka",
                                                     robot_cfg=None, control_mode=""))
    if env_name not in ENVS.list():
        raise ValueError(f"unknown robobench env {env_name!r}; known={ENVS.list()}")
    suite = _suite_of(env_name)
    if suite not in SUITES and suite not in ("sim_gen", "simgen"):  # constructed: null api, no suite entry
        raise ValueError(f"no control API for suite {suite!r}; known suites={list(SUITES)}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # NOTE (2026-07-26): a gpu_max_num_partitions=8 override for batch pods was trialed
    # here and REMOVED — the A/B on the slow-state recipe showed no effect (548s vs
    # 555s); the batch slowdown is seeded by residual velocities in replicated states
    # (see zero_root_velocities), not solver partitioning.
    env = ENVS.get(env_name)().build(num_envs=int(num_envs), device=device)
    env.reset()
    api = resolve_api(env_name)(env, max_steps=max_steps)
    global _LAST_API
    _LAST_API = api  # build_record_camera anchors the boot-time camera on this api's views
    return env, api


_LAST_API = None


def build_record_camera(env, width: int = 1280, height: int = 720):
    """Set up RGB capture from the VIEWPORT camera and return an rgb annotator (or ``None`` on
    failure). Mirrors isaaclab ManagerBasedRLEnv.render()'s rgb_array path EXACTLY (the proven
    mechanism the Fable_5 G1 video uses): a replicator render product + rgb annotator on the
    default viewport camera (/OmniverseKit_Persp), NOT an isaaclab Camera *sensor*.

    Requires AppLauncher(enable_cameras=True) + the RTX launch recipe. We force the sim render
    mode to PARTIAL_RENDERING (required for rgb_array) and aim the viewport at the work area."""
    import sys as _sys
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        # GHOST-FREE BY CONSTRUCTION (2026-07-24): temporal AA (TAA/DLSS) blends frames
        # across time, so teleported state (goto/restore/RL resets) always leaves
        # translucent trails no matter how many flush renders run. FXAA is single-frame:
        # no accumulation, no ghosts. (CAPX_CAPTURE_AA_OP overrides; 2 = FXAA.)
        try:
            import carb.settings

            carb.settings.get_settings().set(
                "/rtx/post/aa/op", int(os.environ.get("CAPX_CAPTURE_AA_OP", "2")))
        except Exception:  # noqa: BLE001 -- capture quality tweak must never break boot
            traceback.print_exc()
        # Anchor the view to ENV 0's origin (with num_envs>1 the grid offsets env 0 away
        # from the world origin; a fixed world-frame eye would film empty floor) AND to the
        # suite's workspace (the api's default view -- franka tabletop != ikea workbench).
        # Replicas are RL-batch machinery, not scenery: hide envs 1..N-1 from the
        # renderer (visibility only; physics untouched). Every video used to show
        # rows of ghost benches receding into the distance — those were the replica
        # cells, and RTX was paying to trace all of them every captured frame.
        try:
            if env.num_envs > 1 and os.environ.get("CAPX_HIDE_REPLICA_ENVS", "1") == "1":
                import omni.usd
                from pxr import UsdGeom
                stage = omni.usd.get_context().get_stage()
                hidden = 0
                for i in range(1, int(env.num_envs)):
                    prim = stage.GetPrimAtPath(f"/World/envs/env_{i}")
                    if prim and prim.IsValid():
                        UsdGeom.Imageable(prim).MakeInvisible()
                        hidden += 1
                print(f"[server] hid {hidden} replica envs from the render", flush=True)
        except Exception:  # noqa: BLE001 -- cosmetic/perf tweak must never break boot
            traceback.print_exc()
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        eye, target = (_LAST_API.VIEWS["default"] if _LAST_API is not None
                       else ((1.6, 1.6, 1.9), (-0.25, 0.0, 1.0)))
        env.sim.set_camera_view(tuple(np.array(eye, dtype=float) + o),
                                tuple(np.array(target, dtype=float) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        render_product = rep.create.render_product("/OmniverseKit_Persp", (int(width), int(height)))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([render_product])
        for _ in range(6):  # warm up the renderer so the first real frame is populated
            env.sim.render()
        data = annot.get_data()
        shape = getattr(data, "shape", None)
        print(f"[server] record camera ready rgb_shape={shape}", flush=True)
        return annot
    except Exception:
        print("record camera build FAILED (video disabled, grading unaffected):\n"
              + traceback.format_exc(), file=_sys.stderr, flush=True)
        return None


def make_prompt(env, api) -> str:
    head = ("You are solving a physics manipulation task (Isaac Lab simulation)."
            if getattr(api, "IS_NULL_EMBODIMENT", False) else
            "You are controlling a robot to complete a manipulation task (Isaac Lab simulation).")
    directive = getattr(api, "CODE_DIRECTIVE", CODE_DIRECTIVE)
    doc = api.api_doc()
    _tails = ["\nThe robot base", "\nTWO fixed-base", "\nNOTE:", "\nNote:",
              "\nSESSION PROTOCOL"]
    if "checkpoint" in _FEATURES_OFF:
        doc = _strip_doc_section(doc, "\n== Checkpoint tree",
                                 ["\n== Tuning the numbers"] + _tails)
        doc = doc.replace(
            "Note: steps_used counts simulated time and is monotonic -- goto() does not "
            "refund steps\n(jumping back in the tree is itself a step forward in time).",
            "Note: steps_used counts simulated time and is monotonic.")
        directive = directive.replace(
            "marked\ncompleted when the stage's program is checkpointed.",
            "marked\ncompleted when the stage verifiably succeeds.")
        directive = directive.replace(
            "marked completed when the stage's program is checkpointed.",
            "marked completed when the stage verifiably succeeds.")
        directive = directive.replace(
            " You can also consider splitting your work and program into smaller "
            "pieces / stages, and checkpoint at stages you want, so that you could "
            "save progress more frequently.",
            "")
        directive = directive.replace(
            "3. Every run that moves the world comes back with an image of where it "
            "ended and the log it printed; read both and assess them before the next "
            "run. Keep the states you would not want to re-derive: a kept state becomes "
            "a node your later work starts from and can return to.\n",
            "3. Every run that moves the world comes back with an image of where it "
            "ended and the log it printed; read both and assess them before the next "
            "run.\n")
        directive = directive.replace(
            "5. Before going back to a node, look at what it holds: the goto tool shows "
            "that node's image, log and program, and asks you to confirm. Read the "
            "'previously tried from here' digest when you land, and change approach "
            "rather than repeating a branch that already failed the same way.\n",
            "\n")
    if "opt" in _FEATURES_OFF:
        doc = _strip_doc_section(doc, "\n== Tuning the numbers in your program",
                                 ["\n== Checkpoint tree"] + _tails)
        directive = directive.replace(
            "4. Use the optimize tool to find optimal values of parameter settings in "
            "your programs: whenever a program has parameters that you need to decide "
            "with trial and error (offsets, depths, angles, timings, thresholds), call "
            "optimize on that program instead of hand-guessing values run after run.\n",
            "4. If a stage keeps failing, change the approach rather than repeating it.\n")
    return f"{head}\nTASK:\n{env.describe()}\n\n{doc}\n" + directive


def reset_episode(env, api) -> dict:
    env.reset()
    api.reset_state()
    api._snapshots.clear()
    api._event_log.clear()
    api._ckpt_nodes.clear()
    api._ckpt_seq = 0
    api._ckpt_current = None
    api._turn_no = 0
    # Frame indexing is GLOBAL and storage is zero-truncation: a new episode continues
    # the capture-index space (old parts stay on disk), it does not rewind it.
    api._turn_frame_start = api._rec_total
    api._edge_frame_start = api._rec_total
    return api.obs_snapshot()


def make_namespace(env, api) -> dict:
    """Fresh agent namespace for a session: RAW simulator access + the toolkit.

    The agent writes Isaac Lab-flavored code (the pick_place_wheel.py style): `env` is
    the live robobench env (env.scene, env.robot, env.sim, env.step(actions)), `api` is
    the control-API object, `torch`/`np` are in scope, and `import isaaclab...` works —
    the Isaac app is already booted. The curated toolkit functions (move_to, grippers,
    look, checkpoint/goto, ...) are ALSO injected and remain the recommended path for
    control, backtracking and vision: they keep the step budget, video recording and
    the checkpoint tree consistent. Direct env.step()/state writes bypass the budget
    and the recorder — use them for reads and advanced control, not to teleport task
    objects (success is graded on the physical task)."""
    g = {"__name__": "__policy__", "np": np, "torch": torch,
         "env": env, "api": api,
         "obs": copy.deepcopy(api.obs_snapshot())}
    fns = api.functions()
    if "checkpoint" in _FEATURES_OFF:
        fns = {k: v for k, v in fns.items() if k not in _CKPT_FNS}
    if "opt" in _FEATURES_OFF:
        fns = {k: v for k, v in fns.items() if k not in _OPT_FNS}
    g.update(fns)
    return g


def run_policy(env, api, code: str, ns: dict | None = None) -> tuple[int, str, str]:
    """Exec one policy program/turn. `ns` is the persistent session namespace (multi-turn:
    variables/policies from earlier turns stay in scope); None -> fresh single-shot namespace.
    Checkpointing is agent-chosen (checkpoint()); begin/end_turn only do turn bookkeeping."""
    g = ns if ns is not None else make_namespace(env, api)
    g["obs"] = copy.deepcopy(api.obs_snapshot())
    api.begin_turn(code)
    out, err = io.StringIO(), io.StringIO()
    rc = 0
    api._turn_stdout_buf = out  # lets checkpoint() snapshot the log up to its call
    opt_req = getattr(api, "_pending_optimize", None)
    api._pending_optimize = None
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            if opt_req:
                # Parameter-search turn (driver's optimize tool): the shipped program
                # runs as one copy per env; no session code executes this turn.
                res = api.optimize_program(
                    opt_req["program"], opt_req["objective"],
                    opt_req.get("space") or {}, **(opt_req.get("options") or {}))
                api._last_optimize_result = res
                print(f"[optimize] best: {res['best']}  score: {res['best_score']:.6g} "
                      f"({res['n_evals']} evaluations in {res['seconds']}s)")
                print(f"[optimize] sensitivity: {res['sensitivity']}")
            else:
                exec(compile(code, "<policy>", "exec"), g, g)
        except SystemExit:
            # policy code calling exit()/SystemExit must NOT propagate: it would kill the
            # render server's main job loop (BaseException passes its `except Exception`).
            print("[policy raised SystemExit — treated as normal end of program]", file=err)
        except Exception:
            traceback.print_exc(file=err)
            rc = 1
    api._turn_stdout_buf = None
    stdout_val = out.getvalue()
    # One image of the end state whenever the program actually moved the world, so the
    # turn's result can always show what it did. Frames were captured during the run, so
    # "did it step" is just whether any were added.
    if getattr(api, "_rec_total", 0) > getattr(api, "_turn_frame_start", 0) and not opt_req:
        api.capture_end_state()
    api.end_turn(code, rc, stdout=stdout_val)
    return rc, stdout_val, err.getvalue()


def check_success(env, api) -> bool:
    """Whether the task is done, from privileged state the agent never sees.

    Prefers the suite's grader when robobench ships one for this scene (the graders
    replaced BaseVerifier upstream, 2026-07-27): its check_success is the same predicate
    the container-based eval grades against, so a run here and a graded delivery there
    agree on what "solved" means. Falls back to the scene's own success when the suite
    has no grader yet.
    """
    grader = _suite_grader(env)
    if grader is not None:
        try:
            return bool(grader.check_success())
        except Exception:  # noqa: BLE001 -- a grader bug must not fail the turn
            traceback.print_exc()
    return api.success()


_GRADER_CACHE: dict[int, object] = {}


def _suite_grader(env):
    """The suite grader for this env's scene, built once per env and reused.

    Graders live in `robobench/suites/<suite>/grader/` and are keyed by their SCENE
    class; a suite with no grader for this scene simply yields None. Never reachable
    from a program's namespace — the agent must not read its own rubric.
    """
    key = id(env)
    if key in _GRADER_CACHE:
        return _GRADER_CACHE[key]
    grader = None
    try:
        import importlib

        # robobench.suites.<suite>.scenes.<module> -> the suite this scene belongs to
        parts = type(env.scene).__module__.split(".")
        suite = parts[2] if len(parts) > 3 and parts[1] == "suites" else ""
        if not suite:
            _GRADER_CACHE[key] = None
            return None
        mod = importlib.import_module(f"robobench.suites.{suite}.grader")
        for name in getattr(mod, "__all__", dir(mod)):
            cls = getattr(mod, name, None)
            scene_cls = getattr(cls, "SCENE", None)
            if isinstance(cls, type) and scene_cls and isinstance(env.scene, scene_cls):
                grader = cls(env)
                print(f"[grader] {name} attached for scoring", flush=True)
                break
    except ModuleNotFoundError:
        pass
    except Exception:  # noqa: BLE001 -- scoring is best-effort, the session goes on
        traceback.print_exc()
    _GRADER_CACHE[key] = grader
    return grader


def scene_summary(env, api) -> str:
    return api.scene_summary()

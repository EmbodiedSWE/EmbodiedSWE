"""Evaluate pi05_base zero-shot on a sim_gen task, with a REAL Franka arm.

Runs inside the Isaac venv and talks to an openpi policy server over websockets (openpi needs
jax[cuda]; Isaac needs its own torch, so they stay in separate venvs on the same GPU — which is
how openpi is meant to be used anyway, see examples/libero/main.py).

Observation/action contract is LIBERO's, because that is the convention closest to this robot:
  observation/image        224x224 uint8, third-person
  observation/wrist_image  224x224 uint8, from the gripper
  observation/state        8 = EE position (3) + EE axis-angle (3) + finger joints (2)
  prompt                   the task's own instruction text
  -> action chunk, each row [dx, dy, dz, drx, dry, drz, gripper]
Robobench's OSC action is [dpos(3), drot(3), finger(2)] with the deltas already normalised to
[-1, 1], so the first six numbers pass through and the gripper scalar becomes a finger target.

Success is the scene's own `success()`. Nothing here scores anything.

    python pi05_eval.py --task-dir base_i25 --scene topple_order --host 127.0.0.1 --port 8000 \
                        --episodes 5 --headless --enable_cameras
"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import time
from pathlib import Path

from isaaclab.app import AppLauncher

_ap = argparse.ArgumentParser()
_ap.add_argument("--task-dir", required=True, help="package under sim_gen.tasks")
_ap.add_argument("--scene", required=True, help="registered scene name")
_ap.add_argument("--host", default="127.0.0.1")
_ap.add_argument("--port", type=int, default=8000)
_ap.add_argument("--episodes", type=int, default=5)
_ap.add_argument("--max-steps", type=int, default=600, help="control steps per episode")
_ap.add_argument("--replan", type=int, default=5, help="actions consumed before re-querying")
_ap.add_argument("--out", default="/home/tiger/workspace/pi05")
_ap.add_argument("--hdfs", default="")
_ap.add_argument("--video-every", type=int, default=1,
                 help="capture cadence in control steps; 1 is free (the base view "
                      "is rendered every step for the policy anyway)")
_ap.add_argument("--prompt", default="", help="VLA instruction; empty falls back to describe()")
# ---- BC-checkpoint parity flags (2026-08-04). Defaults preserve the original
# zero-shot pi05_base behavior byte-for-byte; the plated_meal BC eval sets all
# of them to match how the training data was RECORDED (convention.json +
# gen_batch.py of data_gen/plated_meal/gen_v1). ----
_ap.add_argument("--task-module", default="",
                 help="full dotted module registering the scene (overrides "
                      "sim_gen.tasks.<task_dir>.scene); 'local:<file.py>' imports a file "
                      "beside this script")
_ap.add_argument("--scene-eye", default="", help="comma xyz, env-origin-relative")
_ap.add_argument("--scene-tgt", default="", help="comma xyz, env-origin-relative")
_ap.add_argument("--wrist-mode", default="behind", choices=("behind", "side6"),
                 help="side6 = gen_batch's probed placement (fingertips top-center)")
_ap.add_argument("--state-origin-relative", action="store_true",
                 help="EE position relative to env origin (training convention)")
_ap.add_argument("--gripper-mode", default="libero", choices=("libero", "meters"),
                 help="meters = policy outputs a per-finger target in m (bc_v1 convention)")
_ap.add_argument("--hold", type=int, default=1,
                 help="control steps per policy action (data was recorded at 10fps; "
                      "-1 = auto: round(ctrl_hz/10) like the generator's Recorder)")
_ap.add_argument("--osc-preset", default="", choices=("", "plated_meal"),
                 help="apply the generator's OSC gains/scales")
_ap.add_argument("--robot-preset", default="", choices=("", "plated_meal"),
                 help="apply the generator's FrankaRobotCfg (base pos, gripper limits)")
_ap.add_argument("--obs-res", type=int, default=224,
                 help="capture resolution (training data was 256; server resizes)")
_ap.add_argument("--seed0", type=int, default=-1,
                 help="episode ep resets with seed seed0+ep (like the generator); -1 keeps "
                      "the unseeded legacy behavior. Sharded sweeps pass disjoint ranges so "
                      "every checkpoint sees the same layouts exactly once")
AppLauncher.add_app_launcher_args(_ap)
ARGS = _ap.parse_args()
ARGS.headless = True
ARGS.enable_cameras = True
ARGS.enable_pinocchio = True
if not getattr(ARGS, "kit_args", None):
    # kit mis-decodes the L20 driver version and silently rejects RTX -> empty frames.
    ARGS.kit_args = "--/rtx/verifyDriverVersion/enabled=false"
app = AppLauncher(ARGS).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import axis_angle_from_quat  # noqa: E402

import robobench  # noqa: E402
from robobench.core import EnvCfg  # noqa: E402
from openpi_client import image_tools  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402

RES = ARGS.obs_res


def np3(t) -> np.ndarray:
    return t.detach().cpu().numpy().astype(float)


class Views:
    """One replicator render product on the viewport, aimed twice per step: the scene, then the
    gripper. Two 224x224 renders are cheap, and it avoids authoring a second camera prim."""

    def __init__(self, env):
        import omni.replicator.core as rep

        self.env = env
        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        try:
            import carb.settings
            carb.settings.get_settings().set("/rtx/post/aa/op", 2)   # FXAA: no temporal ghosting
        except Exception as exc:  # noqa: BLE001
            print(f"[pi05] AA setting skipped: {exc}", flush=True)
        self.origin = np3(env.iscene.env_origins[0])
        rp = rep.create.render_product("/OmniverseKit_Persp", (RES, RES))
        self.annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self.annot.attach([rp])
        art = env.robot.articulation
        self.ee = art.body_names.index("panda_hand")
        for _ in range(6):
            env.sim.render()
        print(f"[pi05] views ready, probe={np.asarray(self.annot.get_data()).shape}", flush=True)

    def _grab(self, eye, target) -> np.ndarray:
        self.env.sim.set_camera_view(tuple(eye), tuple(target),
                                     camera_prim_path="/OmniverseKit_Persp")
        for _ in range(3):          # flush: the camera jumped, temporal history is stale
            self.env.sim.render()
        img = np.asarray(self.annot.get_data())
        if img.dtype != np.uint8:
            img = (img.clip(0, 1) * 255).astype(np.uint8)
        return image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img[..., :3], RES, RES))

    def scene_view(self) -> np.ndarray:
        o = self.origin
        eye = (np.array([float(x) for x in ARGS.scene_eye.split(",")])
               if ARGS.scene_eye else np.array([1.1, -1.1, 0.9]))
        tgt = (np.array([float(x) for x in ARGS.scene_tgt.split(",")])
               if ARGS.scene_tgt else np.array([0.0, 0.0, 0.15]))
        return self._grab(o + eye, o + tgt)

    def wrist_view(self) -> np.ndarray:
        """Camera on the hand; placement mode must match how the data was recorded."""
        art = self.env.robot.articulation
        p = np3(art.data.body_pos_w[0, self.ee])
        q = art.data.body_quat_w[0, self.ee:self.ee + 1]
        from isaaclab.utils.math import quat_apply
        fwd = np3(quat_apply(q, torch.tensor([[0.0, 0.0, 1.0]], device=q.device))[0])
        if ARGS.wrist_mode == "side6":
            side = np3(quat_apply(q, torch.tensor([[1.0, 0.0, 0.0]], device=q.device))[0])
            return self._grab(p - 0.02 * fwd + 0.06 * side, p + 0.22 * fwd)
        return self._grab(p - 0.12 * fwd, p + 0.25 * fwd)


def main() -> int:
    out = Path(ARGS.out)
    out.mkdir(parents=True, exist_ok=True)
    if ARGS.task_module.startswith("local:"):
        # a scene file shipped beside this script (e.g. the data_gen plated_meal scene)
        import importlib.util
        import sys as _sys
        p = Path(__file__).resolve().parent / ARGS.task_module[len("local:"):]
        spec = importlib.util.spec_from_file_location("bc_eval_scene", p)
        m = importlib.util.module_from_spec(spec)
        # dataclasses looks the module up by name at class-creation time;
        # exec without registration dies with NoneType.__dict__
        _sys.modules[spec.name] = m
        spec.loader.exec_module(m)
    elif ARGS.task_module:
        importlib.import_module(ARGS.task_module)
    else:
        importlib.import_module(f"sim_gen.tasks.{ARGS.task_dir}.scene")   # registers the scene
    robobench.discover()
    robot_cfg = None
    if ARGS.robot_preset == "plated_meal":
        # the generator's robot config (gen_batch.py, gen_strategy.BASE):
        # state/action parity depends on it
        from robobench.robots.franka import FrankaRobotCfg
        robot_cfg = FrankaRobotCfg(base_pos=(-0.42, 0.0, 0.20), nullspace_dof_pos=(),
                                   gripper_effort_limit=80.0, gripper_stiffness=4000.0)
    env = EnvCfg(scene=ARGS.scene, robot="franka", control_mode="osc",
                 env_spacing=3, **({"robot_cfg": robot_cfg} if robot_cfg else {})).build(
        num_envs=1, device="cuda:0" if torch.cuda.is_available() else "cpu")
    scene, robot = env.scene, env.robot
    art = robot.articulation
    ee = art.body_names.index("panda_hand")
    fj = art.find_joints(["panda_finger_joint1", "panda_finger_joint2"])[0]
    dim = robot.action_dim
    env.reset()
    if ARGS.osc_preset == "plated_meal":
        # the generator's controller tuning (gen_strategy.py) -- the recorded
        # actions are normalized against THESE scales
        osc = robot.controller.controllers[0]
        osc._kp = torch.tensor([220.0, 220.0, 220.0, 600.0, 600.0, 600.0], device=env.device)
        osc._kd = 2.0 * osc._kp.sqrt()
        osc.cfg.rot_scale = 0.15
        osc.cfg.kp_null = 3.0
        osc.cfg.kd_null = 3.46
    views = Views(env)
    ctrl_hz = 1.0 / (env.dt * robot.control_period)
    if ARGS.hold == -1:
        ARGS.hold = max(1, round(ctrl_hz / 10.0))
        print(f"[pi05] hold=auto -> {ARGS.hold} (ctrl_hz={ctrl_hz:.1f})", flush=True)
    # real-time playback: frames are captured every `video_every` control steps,
    # so fps = ctrl_hz/video_every makes the mp4 run at wall-clock speed (the
    # old hardcoded fps=20 played 1100-step episodes in 5s -- unwatchable)
    video_fps = max(1.0, ctrl_hz / max(1, ARGS.video_every))
    prompt = ARGS.prompt or (scene.describe() if hasattr(scene, "describe")
                             else ARGS.scene.replace("_", " "))
    print(f"[pi05] prompt: {prompt[:600]}", flush=True)

    client = websocket_client_policy.WebsocketClientPolicy(host=ARGS.host, port=ARGS.port)
    print(f"[pi05] policy server metadata: {client.get_server_metadata()}", flush=True)

    import imageio.v2 as imageio
    results = []
    for ep in range(ARGS.episodes):
        if ARGS.seed0 >= 0:
            env.reset(seed=ARGS.seed0 + ep)
        else:
            env.reset()
        plan: list[np.ndarray] = []
        writer = imageio.get_writer(str(out / f"{ARGS.scene}_ep{ep}.mp4"), fps=video_fps,
                                    codec="libx264", pixelformat="yuv420p", macro_block_size=1)
        ok, t0, steps = False, time.time(), 0
        for t in range(ARGS.max_steps):
            base = views.scene_view()
            wrist = views.wrist_view()
            if t % ARGS.video_every == 0:
                writer.append_data(base)
            if not plan:
                p = np3(art.data.body_pos_w[0, ee])
                if ARGS.state_origin_relative:
                    p = p - views.origin
                aa = np3(axis_angle_from_quat(art.data.body_quat_w[0, ee:ee + 1])[0])
                grip = np3(art.data.joint_pos[0, fj])
                obs = {"observation/image": base, "observation/wrist_image": wrist,
                       "observation/state": np.concatenate([p, aa, grip]).astype(np.float32),
                       "prompt": prompt}
                chunk = client.infer(obs)["actions"]
                plan = [np.asarray(a, dtype=float) for a in chunk[:ARGS.replan]]
            a = plan.pop(0)
            act = torch.zeros(1, dim, device=env.device)
            act[0, 0:3] = torch.tensor(np.clip(a[0:3], -1, 1), device=env.device)
            act[0, 3:6] = torch.tensor(np.clip(a[3:6], -1, 1), device=env.device)
            if ARGS.gripper_mode == "meters":
                # bc_v1 convention: gripper is a per-finger position target in metres
                act[0, 6:8] = float(np.clip(a[6], 0.0, 0.04))
            else:
                # LIBERO gripper: +1 closes, -1 opens. Ours takes finger targets in metres.
                act[0, 6:8] = 0.0 if float(a[6]) > 0 else 0.04
            for _ in range(max(1, ARGS.hold)):
                env.step(act, render=False)
            steps = t + 1
            if bool(scene.success()[0]):
                ok = True
                break
        writer.close()
        # graded progress (0.15 plated / 0.25 loaded / 0.5 gathered / 0.75 assembled / 1.0):
        # the scene latches milestones in post_step, so one read at episode end suffices.
        # Binary success hides everything a BC policy does short of full completion.
        score = float(scene.score()[0]) if hasattr(scene, "score") else None
        results.append({"episode": ep, "success": ok, "score": score, "steps": steps,
                        "seed": (ARGS.seed0 + ep if ARGS.seed0 >= 0 else None),
                        "wall_s": round(time.time() - t0, 1)})
        print(f"[pi05] ep{ep} success={ok} score={score} steps={steps} "
              f"wall={results[-1]['wall_s']}s", flush=True)

    n_ok = sum(r["success"] for r in results)
    scores = [r["score"] for r in results if r["score"] is not None]
    summary = {"scene": ARGS.scene, "task_dir": ARGS.task_dir, "model": "pi05_base",
               "episodes": ARGS.episodes, "successes": n_ok,
               "rate": n_ok / max(1, ARGS.episodes),
               "mean_score": (sum(scores) / len(scores)) if scores else None,
               "prompt": prompt, "runs": results}
    (out / f"{ARGS.scene}_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"[pi05] RESULT {ARGS.scene}: {n_ok}/{ARGS.episodes}", flush=True)
    if ARGS.hdfs:
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", ARGS.hdfs], capture_output=True)
        for f in sorted(out.glob(f"{ARGS.scene}*")):
            subprocess.run(["hdfs", "dfs", "-put", "-f", str(f), f"{ARGS.hdfs}/{f.name}"],
                           capture_output=True)
    env.close()
    return 0


if __name__ == "__main__":
    import os
    rc = 1
    try:
        rc = main()
    except BaseException as exc:  # noqa: BLE001 -- a crash must leave a record
        import traceback
        traceback.print_exc()
        Path(ARGS.out).mkdir(parents=True, exist_ok=True)
        (Path(ARGS.out) / f"{ARGS.scene}_error.json").write_text(json.dumps(
            {"scene": ARGS.scene, "error": f"{type(exc).__name__}: {exc}"[:400],
             "traceback": traceback.format_exc()[-1500:]}, indent=1) + "\n")
        if ARGS.hdfs:
            subprocess.run(["hdfs", "dfs", "-mkdir", "-p", ARGS.hdfs], capture_output=True)
            subprocess.run(["hdfs", "dfs", "-put", "-f",
                            str(Path(ARGS.out) / f"{ARGS.scene}_error.json"),
                            f"{ARGS.hdfs}/{ARGS.scene}_error.json"], capture_output=True)
        # exit NONZERO immediately: app.close() must not mask the failure into
        # rc=0 (it did -- pods reported "eval exited rc=0" on crashed runs and
        # kept the job looking alive; user directive 2026-08-04: failed jobs
        # must FAIL).
        os._exit(1)
    finally:
        import threading
        threading.Timer(20.0, lambda: os._exit(rc)).start()
    app.close()
    os._exit(rc)

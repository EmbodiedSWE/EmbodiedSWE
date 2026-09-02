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
_ap.add_argument("--video-every", type=int, default=2)
_ap.add_argument("--prompt", default="", help="VLA instruction; empty falls back to describe()")
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

RES = 224


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
        return self._grab(o + np.array([1.1, -1.1, 0.9]), o + np.array([0.0, 0.0, 0.15]))

    def wrist_view(self) -> np.ndarray:
        """Look down the gripper's approach axis from just behind the hand."""
        art = self.env.robot.articulation
        p = np3(art.data.body_pos_w[0, self.ee])
        q = art.data.body_quat_w[0, self.ee:self.ee + 1]
        from isaaclab.utils.math import quat_apply
        fwd = np3(quat_apply(q, torch.tensor([[0.0, 0.0, 1.0]], device=q.device))[0])
        return self._grab(p - 0.12 * fwd, p + 0.25 * fwd)


def main() -> int:
    out = Path(ARGS.out)
    out.mkdir(parents=True, exist_ok=True)
    importlib.import_module(f"sim_gen.tasks.{ARGS.task_dir}.scene")   # registers the scene
    robobench.discover()
    env = EnvCfg(scene=ARGS.scene, robot="franka", control_mode="osc",
                 env_spacing=3).build(num_envs=1,
                                      device="cuda:0" if torch.cuda.is_available() else "cpu")
    scene, robot = env.scene, env.robot
    art = robot.articulation
    ee = art.body_names.index("panda_hand")
    fj = art.find_joints(["panda_finger_joint1", "panda_finger_joint2"])[0]
    dim = robot.action_dim
    env.reset()
    views = Views(env)
    prompt = ARGS.prompt or (scene.describe() if hasattr(scene, "describe")
                             else ARGS.scene.replace("_", " "))
    print(f"[pi05] prompt: {prompt[:600]}", flush=True)

    client = websocket_client_policy.WebsocketClientPolicy(host=ARGS.host, port=ARGS.port)
    print(f"[pi05] policy server metadata: {client.get_server_metadata()}", flush=True)

    import imageio.v2 as imageio
    results = []
    for ep in range(ARGS.episodes):
        env.reset()
        plan: list[np.ndarray] = []
        writer = imageio.get_writer(str(out / f"{ARGS.scene}_ep{ep}.mp4"), fps=20,
                                    codec="libx264", pixelformat="yuv420p", macro_block_size=1)
        ok, t0, steps = False, time.time(), 0
        for t in range(ARGS.max_steps):
            base = views.scene_view()
            wrist = views.wrist_view()
            if t % ARGS.video_every == 0:
                writer.append_data(base)
            if not plan:
                p = np3(art.data.body_pos_w[0, ee])
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
            # LIBERO gripper: +1 closes, -1 opens. Ours takes finger targets in metres.
            act[0, 6:8] = 0.0 if float(a[6]) > 0 else 0.04
            env.step(act, render=False)
            steps = t + 1
            # success is grader-defined for this scene; seated() is the progress read
            if bool(scene.seated().all(dim=1)[0]):
                ok = True
                break
        writer.close()
        results.append({"episode": ep, "success": ok, "steps": steps,
                        "wall_s": round(time.time() - t0, 1)})
        print(f"[pi05] ep{ep} success={ok} steps={steps} "
              f"wall={results[-1]['wall_s']}s", flush=True)

    n_ok = sum(r["success"] for r in results)
    summary = {"scene": ARGS.scene, "task_dir": ARGS.task_dir, "model": "pi05_base",
               "episodes": ARGS.episodes, "successes": n_ok,
               "rate": n_ok / max(1, ARGS.episodes), "prompt": prompt, "runs": results}
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
    finally:
        import threading
        threading.Timer(20.0, lambda: os._exit(rc)).start()
    app.close()
    os._exit(rc)

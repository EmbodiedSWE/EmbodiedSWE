"""plated_meal BC policy evaluation — one batch of rollouts on a forge GPU.

Runs as a module inside the submitted task package (`--module eval_batch`),
exactly like the data generator's gen_batch: same Isaac boot, same scene
(scene.py), same OSC controller bring-up, same camera placements and
state assembly as recording time (DATA_SPEC / convention.json parity). The
ONLY difference from generation: actions come from a trained pi05 policy
served over HTTP (policy_http_server.py on an H20 job) instead of the
scripted strategy.

Protocol per episode: reset(seed) -> at every 10fps tick that starts an
action chunk, render both cameras + read state, POST npz to the server,
receive a (horizon, 7) action chunk, execute each action with zero-order
hold at the control rate; episode ends on scene.success() or the tick cap.

Success = the scene's own success() (same gate the generator used).
Emits "EVAL_BATCH: DONE k/N" and pushes a results JSON to HDFS.
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--server", required=True, help="policy server base url")
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--seed0", type=int, default=50000,
                    help="eval seeds; keep DISJOINT from training data seeds")
parser.add_argument("--max-ticks", type=int, default=1200, help="10fps ticks (120s)")
parser.add_argument("--horizon", type=int, default=10, help="action chunk length")
parser.add_argument("--tag", default="eval")
parser.add_argument("--hdfs-out",
                    default="hdfs://haruna/tmp/zeyu.shen/simgen_bc/plated_meal_eval_v1")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"
args.enable_cameras = True
app = AppLauncher(args).app

import io  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import axis_angle_from_quat, quat_apply  # noqa: E402

import robobench.controllers  # noqa: E402,F401
import robobench.robots  # noqa: E402,F401

try:
    from .scene import PlatedMealSceneCfg  # noqa: E402,F401  (registers the scene)
    from . import gen_strategy as S  # noqa: E402
except ImportError:
    from scene import PlatedMealSceneCfg  # type: ignore # noqa: E402,F401
    import gen_strategy as S  # type: ignore # noqa: E402

from robobench.core import EnvCfg  # noqa: E402
from robobench.robots.franka import FrankaRobotCfg  # noqa: E402

RES = 256
FPS = 10
INSTRUCTION = ("Gather all the food cubes into one bowl and set that bowl on the plate; "
               "keep the other two bowls off the plate.")
BASE_EYE = (0.55, -0.85, 0.75)   # deploy-time scene view (convention.json)
BASE_TGT = (0.05, 0.05, 0.28)


class ObsRig:
    """Camera + state capture identical to gen_batch.Recorder's placements."""

    def __init__(self, env):
        import omni.replicator.core as rep

        self.env = env
        self.art = env.robot.articulation
        self.ee_idx = self.art.body_names.index("panda_hand")
        self.fj = self.art.find_joints(["panda_finger_joint1", "panda_finger_joint2"])[0]
        self.origin = env.iscene.env_origins[0]
        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        try:
            import carb.settings
            carb.settings.get_settings().set("/rtx/post/aa/op", 2)
        except Exception as exc:  # noqa: BLE001
            print(f"[rig] AA setting skipped: {exc}", flush=True)
        rp = rep.create.render_product("/OmniverseKit_Persp", (RES, RES))
        self.annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self.annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        probe = np.asarray(self.annot.get_data())
        assert probe.size and probe.shape[0] == RES, f"camera probe failed: {probe.shape}"
        self.ctrl_hz = 1.0 / (env.dt * env.robot.control_period)
        self.every = max(1, round(self.ctrl_hz / FPS))
        print(f"[rig] camera ready; ctrl_hz={self.ctrl_hz:.1f} every={self.every}", flush=True)

    def _grab(self, eye_w, tgt_w) -> np.ndarray:
        self.env.sim.set_camera_view(tuple(eye_w), tuple(tgt_w),
                                     camera_prim_path="/OmniverseKit_Persp")
        for _ in range(3):
            self.env.sim.render()
        img = np.asarray(self.annot.get_data())
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        return img[..., :3]

    def observe(self) -> dict:
        o = self.origin.detach().cpu().numpy()
        base = self._grab(o + np.array(BASE_EYE), o + np.array(BASE_TGT))
        p = self.art.data.body_pos_w[0, self.ee_idx]
        q = self.art.data.body_quat_w[0, self.ee_idx:self.ee_idx + 1]
        fwd = quat_apply(q, torch.tensor([[0.0, 0.0, 1.0]], device=q.device))[0]
        side = quat_apply(q, torch.tensor([[1.0, 0.0, 0.0]], device=q.device))[0]
        pn, fn = p.detach().cpu().numpy(), fwd.detach().cpu().numpy()
        sn = side.detach().cpu().numpy()
        wrist = self._grab(pn - 0.02 * fn + 0.06 * sn, pn + 0.22 * fn)
        aa = axis_angle_from_quat(q)[0]
        grip = self.art.data.joint_pos[0, self.fj]
        state = np.concatenate([(p - self.origin).detach().cpu().numpy(),
                                aa.detach().cpu().numpy(),
                                grip.detach().cpu().numpy()]).astype(np.float32)
        return {"image": base, "wrist_image": wrist, "state": state}


def query_policy(server: str, obs: dict) -> np.ndarray:
    buf = io.BytesIO()
    np.savez_compressed(buf, image=obs["image"], wrist_image=obs["wrist_image"],
                        state=obs["state"], prompt=np.array(INSTRUCTION))
    req = urllib.request.Request(server.rstrip("/") + "/act", data=buf.getvalue(),
                                 headers={"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=60) as r:
        out = np.load(io.BytesIO(r.read()))
        return np.asarray(out["actions"], dtype=np.float32)   # (horizon, 7)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    rcfg = FrankaRobotCfg(base_pos=S.BASE, nullspace_dof_pos=(),
                          gripper_effort_limit=80.0, gripper_stiffness=4000.0)
    env = EnvCfg(scene="plated_meal", robot="franka", control_mode="osc",
                 env_spacing=3.0, robot_cfg=rcfg, seed=args.seed0).build(
        num_envs=1, device=device)
    env.reset()
    rig = ObsRig(env)

    # same OSC gains/scales the demonstrations were generated with
    osc = env.robot.controller.controllers[0]
    osc._kp = torch.tensor([220.0, 220.0, 220.0, 600.0, 600.0, 600.0], device=device)
    osc._kd = 2.0 * osc._kp.sqrt()
    osc.cfg.rot_scale = 0.15
    osc.cfg.kp_null = 3.0
    osc.cfg.kd_null = 3.46

    # policy server must be alive before burning sim time
    with urllib.request.urlopen(args.server.rstrip("/") + "/health", timeout=30) as r:
        print("[eval] server health:", r.read().decode()[:200], flush=True)

    results = []
    for ep in range(args.episodes):
        seed = args.seed0 + ep
        torch.manual_seed(seed)
        np.random.seed(seed)
        env.reset(seed=seed)
        env.robot.reset(torch.tensor([0], device=device, dtype=torch.long))
        zero6 = torch.zeros(1, 6, device=device)
        ids = torch.tensor([0], device=device, dtype=torch.long)
        for b in list(env.scene.bowls) + list(env.scene.food) + [env.scene.plate]:
            b.write_root_velocity_to_sim(zero6, ids)

        t0 = time.time()
        ok = False
        ticks = 0
        queries = 0
        try:
            chunk = None
            for tick in range(args.max_ticks):
                ticks = tick + 1
                if tick % args.horizon == 0:
                    chunk = query_policy(args.server, rig.observe())
                    queries += 1
                a7 = chunk[min(tick % args.horizon, len(chunk) - 1)]
                a = torch.tensor(a7, device=device, dtype=torch.float32).unsqueeze(0)
                for _ in range(rig.every):
                    env.step(a, render=False)
                if tick % 10 == 9 and bool(env.scene.success()[0]):
                    ok = True
                    break
            if not ok:
                ok = bool(env.scene.success()[0])
        except Exception as exc:  # noqa: BLE001 -- one bad episode must not kill the batch
            import traceback
            traceback.print_exc()
            print(f"[eval] ep{ep} crashed: {exc!r}", flush=True)
            ok = False
        results.append({"ep": ep, "seed": seed, "success": bool(ok),
                        "ticks": int(ticks), "queries": int(queries),
                        "wall_s": round(time.time() - t0, 1)})
        print(f"[eval] ep{ep} seed={seed} success={ok} ticks={ticks} "
              f"wall={results[-1]['wall_s']}s", flush=True)

    n_ok = sum(r["success"] for r in results)
    summary = {"tag": args.tag, "server": args.server, "episodes": args.episodes,
               "seed0": args.seed0, "successes": n_ok,
               "success_rate": round(n_ok / max(1, args.episodes), 4),
               "instruction": INSTRUCTION, "runs": results}
    out = Path(f"eval_{args.tag}.json")
    out.write_text(json.dumps(summary, indent=1))
    if args.hdfs_out:
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", args.hdfs_out], capture_output=True)
        r = subprocess.run(["hdfs", "dfs", "-put", "-f", str(out), args.hdfs_out],
                           capture_output=True, text=True)
        print(f"[eval] hdfs push rc={r.returncode}", flush=True)
    print(f"EVAL_BATCH: DONE {n_ok}/{args.episodes}", flush=True)
    import os
    import threading
    threading.Timer(10.0, lambda: os._exit(0)).start()
    os._exit(0)


if __name__ == "__main__":
    main()

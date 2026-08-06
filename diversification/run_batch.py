"""Run a whole diversification batch on ONE GPU: N envs, one episode of theta each.

    python run_batch.py --envs 100 --total 100 --out /tmp/div --hdfs <dir> --headless --enable_cameras

What is per-env and what is not, because the scene config is shared by every clone:
  per-env   — all 11 policy dimensions, and the nut's spawn xy (written into the sim after reset,
              which is how per-env randomisation is done here)
  per-batch — nut_friction and bolt_slot_x, which live in the scene cfg and in the bolt's spawn.
              They stay a pod-level choice: a few pods cover that dimension.
The verdicts record what was actually used, so nothing is implied that was not applied.

Trajectories for every env are kept subsampled (default every 8th control step, 60 Hz); at full
480 Hz, 100 envs x 144k steps would be several GB per batch for data nobody reads at that rate.
The video is env 0's, because the capture pipeline films one viewport; `film_replay.py` renders
the rest from their recorded states.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from isaaclab.app import AppLauncher

_ap = argparse.ArgumentParser()
_ap.add_argument("--envs", type=int, default=100)
_ap.add_argument("--total", type=int, default=100, help="theta batch size (indices 0..total-1)")
_ap.add_argument("--start", type=int, default=0, help="first theta index this pod runs")
_ap.add_argument("--out", default="/tmp/div")
_ap.add_argument("--hdfs", default="")
_ap.add_argument("--max-sec", type=float, default=300.0)
_ap.add_argument("--record-every", type=int, default=8)
_ap.add_argument("--video-every", type=int, default=16)
_ap.add_argument("--no-video", action="store_true")
_ap.add_argument("--friction", type=float, default=None, help="batch-level nut friction")
_ap.add_argument("--bolt-slot-x", type=float, default=None, help="batch-level bolt slot x")
_ap.add_argument("--tag", default="")
_ap.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(_ap)
ARGS = _ap.parse_args()

ARGS.headless = True
ARGS.enable_cameras = True
ARGS.enable_pinocchio = True
if not getattr(ARGS, "kit_args", None):
    # kit mis-decodes driver 535.261 as 535.5 and rejects it; without this RTX creates no scene
    # renderer and every capture buffer comes back empty.
    ARGS.kit_args = "--/rtx/verifyDriverVersion/enabled=false"
app = AppLauncher(ARGS).app

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import EnvCfg  # noqa: E402
from robobench.robots.franka import FrankaRobot, FrankaRobotCfg  # noqa: E402
from robobench.suites.assembly.scenes import NutThreadAssemblySceneCfg  # noqa: E402

import nut_thread_batch as P  # noqa: E402
from sample_theta import theta_for  # noqa: E402

DT = 1.0 / 480.0


class BatchRecorder:
    """Per-env subsampled trajectories, plus env 0's video."""

    def __init__(self, env, video_path=None, video_every=16, record_every=8, fps=30):
        self.env, self.video_every, self.record_every = env, int(video_every), int(record_every)
        self.n = 0
        self.frames = 0
        self.cam = self.writer = None
        self.rows = {k: [] for k in ("t", "action", "joint_pos", "ee_pos", "grip")}
        self.objects: dict[str, list] = {}
        art = env.robot.articulation
        self._ee = art.body_names.index("panda_hand")
        self._fj = art.find_joints(["panda_finger_joint1"])[0]
        if video_path:
            import imageio.v2 as imageio
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            self.writer = imageio.get_writer(video_path, fps=fps, codec="libx264",
                                             pixelformat="yuv420p", macro_block_size=1)

    def aim(self):
        if self.writer is None:
            return
        from tools.scene_view import Viewer  # the retired harness's capture recipe, as a tool

        b = self.env.scene.bolts[0].data.root_pos_w[0].detach().cpu().numpy().astype(float)
        self.cam = Viewer(self.env, out="/tmp/div_footage", size=(1280, 720),
                          eye=(b[0] - 0.26, b[1] - 0.22, b[2] + 0.20),
                          target=(b[0], b[1], b[2] + 0.02), frame="world").annotator
        print(f"[rec] capture ready rgb_shape={getattr(self.cam.get_data(), 'shape', None)}",
              flush=True)

    def observe(self, env, action, *_):
        self._capture = self.writer is not None and (self.n % self.video_every == 0)
        if self.n % self.record_every == 0:
            art = env.robot.articulation
            self.rows["t"].append(self.n)
            self.rows["action"].append(action.detach().cpu().numpy().copy())
            self.rows["joint_pos"].append(art.data.joint_pos.detach().cpu().numpy().copy())
            self.rows["ee_pos"].append(art.data.body_pos_w[:, self._ee].detach().cpu().numpy().copy())
            self.rows["grip"].append(art.data.joint_pos[:, self._fj].detach().cpu().numpy().copy())
            for i, nut in enumerate(env.scene.nuts):
                self.objects.setdefault(f"nut_{i}", []).append(
                    nut.data.root_state_w.detach().cpu().numpy().copy())
            for i, bolt in enumerate(env.scene.bolts):
                self.objects.setdefault(f"bolt_{i}", []).append(
                    bolt.data.root_state_w.detach().cpu().numpy().copy())
        self.n += 1
        return self._capture

    def capture(self, env):
        if not getattr(self, "_capture", False):
            return
        img = np.asarray(self.cam.get_data())
        if img.dtype != np.uint8:
            img = (img.clip(0, 1) * 255).astype(np.uint8)
        if img.ndim != 3:
            raise RuntimeError(f"capture returned {img.shape} — no usable frame")
        self.writer.append_data(img[..., :3])
        self.frames += 1

    def save(self, path, meta):
        arrays = {k: np.asarray(v) for k, v in self.rows.items()}   # (steps, n_envs, ...)
        for name, seq in self.objects.items():
            arrays[f"obj_{name}"] = np.asarray(seq)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, meta=np.asarray([json.dumps(meta)], dtype=object), **arrays)

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def main() -> int:
    n = int(ARGS.envs)
    idx = list(range(ARGS.start, ARGS.start + n))
    thetas = [theta_for(i, ARGS.total) for i in idx]
    out = Path(ARGS.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = ARGS.tag or f"batch{ARGS.start:04d}_{n}"

    # Batch-level dims (scene cfg is shared by every clone): take the flags, else episode 0's.
    friction = ARGS.friction if ARGS.friction is not None else float(thetas[0]["nut_friction"])
    slot_x = ARGS.bolt_slot_x if ARGS.bolt_slot_x is not None else float(thetas[0]["bolt_slot_x"])
    for t in thetas:
        t["nut_friction"], t["bolt_slot_x"] = friction, slot_x

    torch.manual_seed(int(ARGS.seed))
    robobench.discover()
    FrankaRobot.TORQUE_CONTROL_DT = DT
    env = EnvCfg(
        scene="nut_thread",
        scene_cfg=NutThreadAssemblySceneCfg(surface_z=0.0, bolt_slots=((slot_x, 0.0),),
                                           nut_init_xy=((float(thetas[0]["nut_x"]),
                                                         float(thetas[0]["nut_y"])),),
                                           nut_friction=friction),
        robot="franka",
        robot_cfg=FrankaRobotCfg(base_pos=(0.0, 0.0, 0.0), nullspace_dof_pos=(),
                                 gripper_stiffness=P.GRIP_KP),
        control_mode="osc", env_spacing=2, sim_overrides={"dt": DT}).build(num_envs=n,
                                                                          device="cuda:0")
    video = None if ARGS.no_video else str(out / f"{tag}_env0.mp4")
    rec = BatchRecorder(env, video, ARGS.video_every, ARGS.record_every)
    env.reset()

    # Per-env nut spawn. The cfg gave every clone episode 0's xy, so shift each env by its own
    # delta from that. Shifting (rather than recomputing a world position) cannot disagree with the
    # table's frame, and it leaves the scene's own +/-10 mm reset jitter in place on top of theta.
    nut = env.scene.nuts[0]
    st = nut.data.root_state_w.clone()
    dx = torch.tensor([float(t["nut_x"]) - float(thetas[0]["nut_x"]) for t in thetas],
                      device=env.device)
    dy = torch.tensor([float(t["nut_y"]) - float(thetas[0]["nut_y"]) for t in thetas],
                      device=env.device)
    st[:, 0] += dx
    st[:, 1] += dy
    st[:, 7:13] = 0.0
    nut.write_root_state_to_sim(st)
    for _ in range(int(0.3 / DT)):        # let the re-placed nuts settle before the policy looks
        env.step(torch.zeros(n, env.robot.action_dim, device=env.device), render=False)
    rec.aim()

    print(f"[batch] {n} envs, theta {idx[0]}..{idx[-1]}, friction={friction} slot_x={slot_x}",
          flush=True)
    t0 = time.time()
    verdicts = P.solve_batch(env, thetas, rec, max_sec=ARGS.max_sec)
    wall = round(time.time() - t0, 1)
    rec.close()

    ok = sum(1 for v in verdicts if v["success"])
    for v in verdicts:
        v.update(wall_s=wall, envs=n, friction=friction, bolt_slot_x=slot_x, batch_tag=tag)
    (out / f"{tag}.json").write_text(json.dumps(verdicts, indent=1) + "\n")
    for v in verdicts:
        (out / f"ep{v['index']:04d}.json").write_text(json.dumps(v, indent=1, sort_keys=True) + "\n")
    rec.save(str(out / f"{tag}.npz"), {"thetas": thetas, "verdicts": verdicts})
    print(f"[batch] RESULT {ok}/{n} succeeded in {wall}s ({wall / max(1, n):.1f}s/episode) "
          f"frames={rec.frames}", flush=True)

    if ARGS.hdfs:
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", ARGS.hdfs], capture_output=True)
        for f in sorted(out.glob("*")):
            if f.is_file():
                r = subprocess.run(["hdfs", "dfs", "-put", "-f", str(f), f"{ARGS.hdfs}/{f.name}"],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    print(f"[batch] upload {f.name} FAILED: {r.stderr.strip()[:160]}", flush=True)
        print(f"[batch] uploaded -> {ARGS.hdfs}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    import os
    rc = 1
    try:
        rc = main()
    except BaseException as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        Path(ARGS.out).mkdir(parents=True, exist_ok=True)
        (Path(ARGS.out) / "batch_error.json").write_text(json.dumps(
            {"error": f"{type(exc).__name__}: {exc}"[:400],
             "traceback": traceback.format_exc()[-1500:]}, indent=1) + "\n")
        if ARGS.hdfs:
            subprocess.run(["hdfs", "dfs", "-mkdir", "-p", ARGS.hdfs], capture_output=True)
            subprocess.run(["hdfs", "dfs", "-put", "-f", str(Path(ARGS.out) / "batch_error.json"),
                            f"{ARGS.hdfs}/batch_error.json"], capture_output=True)
        rc = 1
    finally:
        import threading
        threading.Timer(20.0, lambda: os._exit(rc)).start()
    app.close()
    os._exit(rc)

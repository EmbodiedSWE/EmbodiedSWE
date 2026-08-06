"""plated_meal data generation — one batch of recorded episodes on a forge GPU.

Runs as a module inside the submitted task package (`--module gen_batch`). Boots Isaac
once, builds the env once, then loops episodes: sample (seed, Params, noise/physics/visual
condition) for the batch's mode, run strategy_1, record at 10 fps, gate on the scene's own
success(), dump DATA_SPEC Option-B episode dirs, push the batch to HDFS.

Modes (per sim_gen/diversification.html):
  nominal   the delivered behavior, seeds only
  params    Params sampled within bands (macro order, bowl choice, cube order, slack constants)
  noise     nominal params + OU actuation noise on the 6 pose dims; executed action is
            perturbed, the clean action is the label (both recorded)
  physics   per-episode mass/friction jitter on the task objects, no action noise
  visual    nominal behavior, camera pose + lighting jittered per episode

Frames follow simgen_bc/DATA_SPEC.md: image/wrist_image (256,256,3) u8 at the SAME tick as
state (8,) f32, actions (7,) f32 = [dpose(6), gripper(1)] as passed to env.step; fps=10.
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=2)
parser.add_argument("--seed0", type=int, default=0)
parser.add_argument("--mode", default="nominal",
                    choices=("nominal", "params", "noise", "physics", "visual"))
parser.add_argument("--sigma", type=float, default=0.08, help="OU noise scale (noise mode)")
parser.add_argument("--force-bowl-first", type=int, default=-1, choices=(-1, 0, 1),
                    help="params mode: pin the macro order (-1 = sample)")
parser.add_argument("--batch", default="")
parser.add_argument("--out", default="datagen_out")
parser.add_argument("--hdfs-out", default="hdfs://haruna/tmp/zeyu.shen/simgen_bc/plated_meal_gen_v1")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"
args.enable_cameras = True
app = AppLauncher(args).app

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import axis_angle_from_quat, quat_apply  # noqa: E402

import robobench.controllers  # noqa: E402,F401
import robobench.robots  # noqa: E402,F401

from . import gen_strategy as S  # noqa: E402
from .gen_strategy import Params  # noqa: E402

try:
    from .scene import PlatedMealSceneCfg  # noqa: E402,F401  (registers the scene)
except ImportError:
    from scene import PlatedMealSceneCfg  # type: ignore # noqa: E402,F401

from robobench.core import EnvCfg  # noqa: E402
from robobench.robots.franka import FrankaRobotCfg  # noqa: E402

RES = 256
FPS = 10
INSTRUCTION = ("Gather all the food cubes into one bowl and set that bowl on the plate; "
               "keep the other two bowls off the plate.")
BASE_EYE = (0.55, -0.85, 0.75)     # deploy-time scene view, relative to env origin
BASE_TGT = (0.05, 0.05, 0.28)


class Recorder:
    """Captures (image, wrist_image, state, actions) at 10 fps; injects OU noise if armed.

    state[t] and both renders are taken BEFORE actions[t] executes (DATA_SPEC alignment).
    actions[t] is the clean 7-D label; actions_exec[t] is what env.step actually received.
    """

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
            carb.settings.get_settings().set("/rtx/post/aa/op", 2)  # FXAA — no ghosting
        except Exception as exc:  # noqa: BLE001
            print(f"[rec] AA setting skipped: {exc}", flush=True)
        rp = rep.create.render_product("/OmniverseKit_Persp", (RES, RES))
        self.annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self.annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        probe = np.asarray(self.annot.get_data())
        assert probe.size and probe.shape[0] == RES, f"camera probe failed: {probe.shape}"
        print(f"[rec] camera ready {probe.shape}", flush=True)

        self.ctrl_hz = 1.0 / (env.dt * env.robot.control_period)
        self.every = max(1, round(self.ctrl_hz / FPS))
        self.eye = np.array(BASE_EYE, dtype=float)
        self.tgt = np.array(BASE_TGT, dtype=float)
        self.noise_sigma = 0.0
        self.noise_alpha = 0.0
        self._eps = torch.zeros(6, device=env.device)
        self._buf = None
        self._k = 0

    # ----- per-episode ------------------------------------------------------------------
    def begin(self, noise_sigma=0.0, corr_s=0.3, eye=None, tgt=None):
        self.noise_sigma = float(noise_sigma)
        self.noise_alpha = math.exp(-1.0 / (self.ctrl_hz * corr_s))
        self._eps.zero_()
        if eye is not None:
            self.eye = np.array(eye, dtype=float)
        if tgt is not None:
            self.tgt = np.array(tgt, dtype=float)
        self._buf = {"image": [], "wrist_image": [], "state": [],
                     "actions": [], "actions_exec": []}
        self._k = 0

    def _grab(self, eye_w, tgt_w) -> np.ndarray:
        self.env.sim.set_camera_view(tuple(eye_w), tuple(tgt_w),
                                     camera_prim_path="/OmniverseKit_Persp")
        for _ in range(3):
            self.env.sim.render()
        img = np.asarray(self.annot.get_data())
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        return img[..., :3]

    def _capture(self, a_clean_7, a_exec_7):
        o = self.origin.detach().cpu().numpy()
        base = self._grab(o + self.eye, o + self.tgt)
        p = self.art.data.body_pos_w[0, self.ee_idx]
        q = self.art.data.body_quat_w[0, self.ee_idx:self.ee_idx + 1]
        fwd = quat_apply(q, torch.tensor([[0.0, 0.0, 1.0]], device=q.device))[0]
        side = quat_apply(q, torch.tensor([[1.0, 0.0, 0.0]], device=q.device))[0]
        pn, fn = p.detach().cpu().numpy(), fwd.detach().cpu().numpy()
        sn = side.detach().cpu().numpy()
        # "side6" placement (probed): clears the wrist mesh, fingers + workspace in frame
        wrist = self._grab(pn - 0.02 * fn + 0.06 * sn, pn + 0.22 * fn)
        aa = axis_angle_from_quat(q)[0]
        grip = self.art.data.joint_pos[0, self.fj]
        state = np.concatenate([(p - self.origin).detach().cpu().numpy(),
                                aa.detach().cpu().numpy(),
                                grip.detach().cpu().numpy()]).astype(np.float32)
        self._buf["image"].append(base)
        self._buf["wrist_image"].append(wrist)
        self._buf["state"].append(state)
        self._buf["actions"].append(a_clean_7)
        self._buf["actions_exec"].append(a_exec_7)

    def step_env(self, a_clean: torch.Tensor):
        """The strategy's only path into env.step."""
        a_exec = a_clean
        if self.noise_sigma > 0.0:
            w = torch.randn(6, device=a_clean.device)
            self._eps = self.noise_alpha * self._eps + \
                self.noise_sigma * math.sqrt(1 - self.noise_alpha ** 2) * w
            a_exec = a_clean.clone()
            a_exec[0, 0:6] = (a_exec[0, 0:6] + self._eps).clamp(-1.0, 1.0)
        if self._k % self.every == 0:
            c7 = np.concatenate([a_clean[0, 0:6].detach().cpu().numpy(),
                                 a_clean[0, 6:7].detach().cpu().numpy()]).astype(np.float32)
            e7 = np.concatenate([a_exec[0, 0:6].detach().cpu().numpy(),
                                 a_exec[0, 6:7].detach().cpu().numpy()]).astype(np.float32)
            self._capture(c7, e7)
        self._k += 1
        self.env.step(a_exec, render=False)

    def finish(self) -> dict:
        out = {k: np.stack(v) if v else np.zeros(0) for k, v in self._buf.items()}
        self._buf = None
        return out


def set_light_scale(scale: float) -> bool:
    """Scale every UsdLux light intensity by `scale` (visual mode)."""
    import omni.usd
    from pxr import UsdLux

    stage = omni.usd.get_context().get_stage()
    n = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdLux.LightAPI) or prim.GetTypeName().endswith("Light"):
            attr = prim.GetAttribute("inputs:intensity")
            if attr and attr.Get() is not None:
                attr.Set(float(attr.Get()) * scale)
                n += 1
    print(f"[gen] scaled {n} lights by {scale:.2f}", flush=True)
    return n > 0


def jitter_physics(scene, rng) -> dict:
    """Per-episode mass/friction jitter on task objects, verified by readback."""
    applied = {}
    bodies = {f"bowl{i}": b for i, b in enumerate(scene.bowls)}
    bodies.update({f"food{i}": b for i, b in enumerate(scene.food)})
    for name, b in bodies.items():
        view = b.root_physx_view
        m = view.get_masses().clone()
        f = float(rng.uniform(0.7, 1.3))
        view.set_masses(m * f, torch.arange(m.shape[0]))
        m2 = view.get_masses()
        assert abs(float(m2[0]) - float(m[0]) * f) < 1e-6 * max(1.0, float(m[0])), \
            f"mass jitter readback failed for {name}"
        applied[name] = {"mass_scale": round(f, 3)}
    return applied


def main() -> None:
    # full log survives with the data (the forge returns only a 12 KB stdout tail)
    import sys

    class Tee:
        def __init__(self, path):
            self.f = open(path, "w", buffering=1)
            self.o = sys.stdout

        def write(self, x):
            self.o.write(x)
            self.f.write(x)

        def flush(self):
            self.o.flush()
            self.f.flush()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    rcfg = FrankaRobotCfg(base_pos=S.BASE, nullspace_dof_pos=(),
                          gripper_effort_limit=80.0, gripper_stiffness=4000.0)
    env = EnvCfg(scene="plated_meal", robot="franka", control_mode="osc",
                 env_spacing=3.0, robot_cfg=rcfg, seed=args.seed0).build(
        num_envs=1, device=device)
    env.reset()
    rec = Recorder(env)

    batch = args.batch or f"{args.mode}_{time.strftime('%m%d_%H%M')}"
    out_root = Path(args.out) / batch
    out_root.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out_root / "gen.log")
    results = []

    for ep in range(args.episodes):
        seed = args.seed0 + ep
        rng = random.Random(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        env.reset(seed=seed)
        env.robot.reset(torch.tensor([0], device=device, dtype=torch.long))
        # kill residual velocities from the previous episode: reset restores poses but
        # inherited motion makes late-episode settle() flaky (measured: identical layout,
        # settled=False only in batch position)
        zero6 = torch.zeros(1, 6, device=device)
        ids = torch.tensor([0], device=device, dtype=torch.long)
        for b in list(env.scene.bowls) + list(env.scene.food) + [env.scene.plate]:
            b.write_root_velocity_to_sim(zero6, ids)

        P = Params()
        cond: dict = {"mode": args.mode, "seed": seed}
        if args.mode == "params":
            P.bowl_first = (rng.random() < 0.6 if args.force_bowl_first < 0
                            else bool(args.force_bowl_first))
            P.bowl_choice = rng.choice(["reachable", "random"])
            P.cube_order = rng.choice(["near", "far", "random"])
            P.drop_side = rng.choice(["base", "far", "random"])
            P.carry_bowl_h = rng.uniform(0.045, 0.085)
            P.cube_carry_extra = rng.uniform(0.055, 0.095)
            P.cube_hover = rng.uniform(0.10, 0.16)
            P.bowl_tip_off = rng.uniform(-0.010, -0.006)
            P.cube_grasp_off = rng.uniform(-0.006, -0.002)
            P.drop_perp = rng.uniform(0.018, 0.026)
            P.drop_h = rng.uniform(0.022, 0.042)
        sigma = 0.0
        eye = tgt = None
        if args.mode == "noise":
            sigma = rng.uniform(0.5, 1.0) * args.sigma
            cond["sigma"] = round(sigma, 4)
        if args.mode == "physics":
            cond["physics"] = jitter_physics(env.scene, rng)
        if args.mode == "visual":
            eye = (np.array(BASE_EYE) +
                   np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15),
                             rng.uniform(-0.10, 0.15)]))
            tgt = (np.array(BASE_TGT) +
                   np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05),
                             rng.uniform(-0.04, 0.04)]))
            scale = rng.uniform(0.6, 1.6)
            set_light_scale(scale)
            cond["light_scale"] = round(scale, 3)
            cond["eye"] = [round(float(v), 3) for v in eye]
            cond["tgt"] = [round(float(v), 3) for v in tgt]

        rec.begin(noise_sigma=sigma, eye=eye, tgt=tgt)
        t0 = time.time()
        try:
            ok = S.run_episode(env, rec, P, rng)
        except Exception as exc:  # noqa: BLE001 -- one bad episode must not kill the batch
            import traceback
            traceback.print_exc()
            print(f"[gen] ep{ep} crashed: {exc!r}", flush=True)
            ok = False
        data = rec.finish()
        if args.mode == "visual":
            set_light_scale(1.0 / cond["light_scale"])  # restore for the next episode

        n = len(data["state"]) if ok else 0
        finite = ok and all(np.isfinite(data[k]).all() for k in ("state", "actions"))
        ok = ok and finite and n >= 10
        results.append({"ep": ep, "seed": seed, "success": bool(ok), "frames": int(n),
                        "wall_s": round(time.time() - t0, 1), **cond,
                        "params": {k: (round(v, 4) if isinstance(v, float) else v)
                                   for k, v in vars(P).items() if k != "extra"}})
        print(f"[gen] ep{ep} seed={seed} success={ok} frames={n} "
              f"wall={results[-1]['wall_s']}s", flush=True)
        if ok:
            ep_dir = out_root / f"ep_{ep:04d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                ep_dir / "frames.npz",
                image=data["image"], wrist_image=data["wrist_image"],
                state=data["state"], actions=data["actions"],
                actions_exec=data["actions_exec"])
            (ep_dir / "meta.json").write_text(json.dumps(
                {"task": INSTRUCTION, "task_dir": "plated_meal", "seed": seed,
                 "success": True, "fps": FPS, "scene": "scene_0",
                 "strategy": "strategy_1", **cond,
                 "params": results[-1]["params"]}, indent=1))

    n_ok = sum(r["success"] for r in results)
    (out_root / "meta.json").write_text(json.dumps(
        {"mode": args.mode, "episodes": args.episodes, "successes": n_ok,
         "yield": round(n_ok / max(1, args.episodes), 3), "seed0": args.seed0,
         "instruction": INSTRUCTION, "runs": results}, indent=1))
    print(f"[gen] batch {batch}: {n_ok}/{args.episodes} verified episodes", flush=True)

    if args.hdfs_out:
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", args.hdfs_out], capture_output=True)
        r = subprocess.run(["hdfs", "dfs", "-put", "-f", str(out_root), args.hdfs_out],
                           capture_output=True, text=True)
        print(f"[gen] hdfs push rc={r.returncode} {r.stderr[-200:] if r.returncode else ''}",
              flush=True)
    print(f"GEN_BATCH: DONE {n_ok}/{args.episodes}", flush=True)
    threading.Timer(10.0, lambda: os._exit(0 if n_ok else 1)).start()
    os._exit(0 if n_ok else 1)


if __name__ == "__main__":
    main()

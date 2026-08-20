"""replay_actions — re-drive recorded episodes' actions through the eval executor (boots Isaac).

    # joint-PD condition: the bake's joint_pos labels through the joint tracker
    .venv/bin/python vla/eval/replay_actions.py bulb_jointpd_60hz --headless \\
        --episodes vla/example_data/stamp_check/ep_0000 vla/example_data/stamp_check/ep_0001

    # matched-controller condition: the SAME episodes' verbatim recorded commands
    # through the controller they ran under (law from each episode's stamped meta.json)
    .venv/bin/python vla/eval/replay_actions.py assembly.bulb.franka.osc --matched-controller \\
        --headless --batch <…/data/<batch>>

    # start points are free: --t0 9000 (all), --t0 9000 8700 (per episode, in the
    # order passed), or --t0-frac 0.85 (fraction of each episode's own length).
    # Late starts are the strongest sanity anchor: little room for divergence.

The executor certification: init each env slot from an episode's recorded state
at its start tick (mid-starts are exact — the traj stores full state every
tick), feed each episode's own actions on one global clock, and measure how
faithfully the executor reproduces the demos — per-joint tracking error against
the recorded q and the grader's success verdict. Episodes chunk into groups of
num_envs (sorted by length); a finished episode's slot freezes on a hold action
while the rest run, and its metrics stop accumulating. Each episode's recorded
PHYSICAL_PARAMS draw (meta parameters.physical) is re-applied by DEFAULT; a
batch mixing draws is refused. If the demos' own actions can't re-succeed, no
policy trained on them will; --matched-controller is the sanity anchor (the
exact controller the demos ran under, so it should re-succeed).

Conventions (mirroring vla/convert/conventions.py at stride 1):
  joint_pos  action_t = [q_rec[t+1], closedness_rec[t+1]]          (absolute)
  joint_vel  action_t = [(q_rec[t+1]-q_rec[t])*rate, closed[t+1]]  (integrated
             from LIVE q — honest, drift compounds; --integrate dataset targets
             q_rec[t+1] instead = the pure tracker test)
  raw_cmd    action_t = traj["action"][t] verbatim (--matched-controller;
             closed-loop by nature: the controller re-anchors on the live EE pose)

Writes report.json (per-episode + aggregate) and, for the first --video-slots
slots of the first chunk, per-view mp4s under --out/<tag>/.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="re-drive recorded episodes through the eval executor")
parser.add_argument("source", help="registered sim name | ENVS preset | bake.json path")
parser.add_argument("--episodes", nargs="*", default=[], help="episode dirs")
parser.add_argument("--batch", default="", help="batch dir — every ep_* inside")
parser.add_argument("--matched-controller", dest="matched_controller", action="store_true",
                    help="replay traj['action'] verbatim under the episodes' stamped controller law")
parser.add_argument("--num_envs", type=int, default=4, help="episodes replayed in parallel")
parser.add_argument("--t0", nargs="*", default=[],
                    help="start time(s) in SIM SECONDS ('130', '250.5', or clock '4:10' / "
                         "'4:10.10'): one value for all episodes, or one per episode in the "
                         "order passed (default 0). Dataset videos run at one frame per tick, "
                         "so a video timestamp is directly a sim time")
parser.add_argument("--t0-frac", type=float, default=None, dest="t0_frac",
                    help="start at this fraction of EACH episode's length (overrides --t0)")
parser.add_argument("--cap", type=int, default=0, help="cap replayed ticks per episode (0 = all)")
parser.add_argument("--extra-hold", type=int, default=60, dest="extra_hold",
                    help="hold steps after the last action (settle before the final verdict)")
parser.add_argument("--integrate", choices=("live", "dataset"), default="live",
                    help="joint_vel only: integrate targets from live q (honest) or dataset q")
parser.add_argument("--video-slots", type=int, default=1, dest="video_slots",
                    help="record mp4s for this many slots of the FIRST chunk (0 = none)")
parser.add_argument("--grip-margin", type=float, default=None, dest="grip_margin",
                    help="metres of finger closure commanded beyond the closedness label "
                         "(squeeze-force restoration for joint conventions)")
parser.add_argument("--out", default=str(Path(__file__).parent / "_out"))
parser.add_argument("--tag", default="", help="output dir name (default: derived)")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app = AppLauncher(args).app

import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim import load_sim  # noqa: E402  (importing sim also puts vla/convert on sys.path)

import episode as _convert  # noqa: E402  (vla/convert's convention math)

# ----- episodes + start points --------------------------------------------------------------------
eps = [Path(e) for e in args.episodes]
if args.batch:
    eps += sorted(p for p in Path(args.batch).glob("ep_*") if (p / "traj.npz").is_file())
if not eps:
    raise SystemExit("no episodes: pass --episodes and/or --batch")
metas = {e: json.loads((e / "meta.json").read_text()) for e in eps}
m0 = metas[eps[0]]
rec_rate = 1.0 / (m0["sim_dt"] * m0.get("decimation", 1))


def parse_t0(v: str) -> int:
    """Sim time -> tick: '130' / '250.5' seconds, or clock '4:10' / '4:10.10' / '1:04:10'."""
    sec = 0.0
    for part in str(v).split(":"):
        sec = sec * 60.0 + float(part)
    return round(sec * rec_rate)


if args.t0_frac is not None:
    t0_of = {e: int(args.t0_frac * (metas[e]["steps"] - 1)) for e in eps}
elif len(args.t0) == 1:
    t0_of = {e: parse_t0(args.t0[0]) for e in eps}
elif len(args.t0) == len(eps):
    t0_of = {e: parse_t0(v) for e, v in zip(eps, args.t0)}  # in the order passed, BEFORE sorting
elif args.t0:
    raise SystemExit(f"--t0 takes 1 value or {len(eps)} (one per episode); got {len(args.t0)}")
else:
    t0_of = {e: 0 for e in eps}

eps.sort(key=lambda e: (metas[e]["steps"], str(e)))
phys = {e: (metas[e].get("parameters") or {}).get("physical") or {} for e in eps}
for e, m in metas.items():
    if abs(1.0 / (m["sim_dt"] * m.get("decimation", 1)) - rec_rate) > 1e-6:
        raise SystemExit(f"{e}: control rate differs from {eps[0]} — replay batches separately")
    if args.matched_controller and m.get("controller") != m0.get("controller"):
        raise SystemExit(f"{e}: stamped controller law differs from {eps[0]} — mixed laws")
    if phys[e] != phys[eps[0]]:
        raise SystemExit(f"{e}: PHYSICAL_PARAMS draw {phys[e]} differs from {eps[0]}'s "
                         f"{phys[eps[0]]} — replay per-draw groups separately")

# ----- sim ----------------------------------------------------------------------------------------
overrides: dict = {}
if phys[eps[0]]:
    overrides["physical_params"] = phys[eps[0]]  # match the recorded world by default
if args.grip_margin is not None:
    overrides["grip_margin"] = args.grip_margin
if args.matched_controller:
    if not m0.get("controller"):
        raise SystemExit("--matched-controller needs episodes with a stamped controller "
                         "block in meta.json")
    overrides.update(control_space="raw_cmd", control_freq_hz=rec_rate,
                     stamp={"controller": m0["controller"], "origin": "episode-meta"})
device = "cuda:0" if torch.cuda.is_available() else "cpu"
sim = load_sim(args.source, num_envs=min(args.num_envs, len(eps)), device=device, **overrides)
cs = sim.spec.control_space
if cs is None:
    raise SystemExit("replay needs a control law (joint_pos/joint_vel/raw_cmd) — "
                     "a bare preset has nothing to certify (add --matched-controller?)")
if abs(sim.rate_hz - rec_rate) > 1e-6:
    raise SystemExit(f"episode rows at {rec_rate:.1f} Hz but executor latches at "
                     f"{sim.rate_hz:.1f} Hz — stride replay not implemented (use the native rate)")

tag = args.tag or re.sub(r"[^\w.-]+", "_",
                         f"{args.source}{'_matched' if args.matched_controller else ''}").strip("_")
out = Path(args.out) / tag
out.mkdir(parents=True, exist_ok=True)
E = sim.env.num_envs
n_arm = len(sim.arm_names)


def hold_rows(last_actions: np.ndarray) -> torch.Tensor:
    """Per-slot freeze action for matched-controller replay: zeros for task-space leaves,
    the last recorded command for joint leaves (holds the grip without inventing motion)."""
    from robobench.controllers import JointController

    rows = torch.zeros((E, last_actions.shape[1]), device=device)
    i = 0
    for leaf in getattr(sim.env.robot.controller, "controllers", [sim.env.robot.controller]):
        if isinstance(leaf, JointController):
            rows[:, i:i + leaf.action_dim] = torch.as_tensor(
                last_actions[:, i:i + leaf.action_dim], dtype=torch.float32, device=device)
        i += leaf.action_dim
    return rows


# ----- replay -------------------------------------------------------------------------------------
results = []
for lo in range(0, len(eps), E):
    chunk = eps[lo:lo + E]
    n = len(chunk)
    q_rec, closed_rec, raw_act, T, S = [], [], [], [], []
    for e in chunk:
        traj = np.load(e / "traj.npz")
        arm_joints, q, closed = _convert._split_gripper(
            list(sim.env.robot.articulation.joint_names), traj["robot/joint_pos"].astype(np.float32))
        if arm_joints != sim.arm_names:
            raise SystemExit(f"{e}: arm joints {arm_joints} != sim {sim.arm_names}")
        q_rec.append(q); closed_rec.append(closed); T.append(len(q))
        S.append(min(t0_of[e], len(q) - 1))
        raw_act.append(traj["action"].astype(np.float32) if args.matched_controller else None)
    pad = E - n
    q_rec += [q_rec[-1]] * pad; closed_rec += [closed_rec[-1]] * pad
    raw_act += [raw_act[-1]] * pad; T += [T[-1]] * pad; S += [S[-1]] * pad
    k_end = max(T[s] - 1 - S[s] for s in range(E))
    if args.cap:
        k_end = min(k_end, args.cap)
    print(f"[replay] chunk {lo // E + 1}: {n} eps, start ticks {S[:n]}, up to {k_end} ticks "
          f"@ {rec_rate:.0f} Hz ({cs}{'/' + args.integrate if cs == 'joint_vel' else ''})", flush=True)

    obs = sim.init_from_episode(chunk, t0=S[:n])
    writers = {}
    if lo == 0 and args.video_slots > 0:
        for s in range(min(args.video_slots, n)):
            for view in sim.sensors:
                writers[(s, view)] = imageio.get_writer(
                    str(out / f"{chunk[s].name}_{view}.mp4"), fps=int(round(rec_rate)),
                    codec="libx264", quality=None, pixelformat="yuv420p",
                    output_params=["-crf", "18", "-preset", "medium"])

    err_max = np.zeros((E, n_arm)); err_sum = np.zeros(E); err_n = np.zeros(E)
    grip_max = np.zeros(E); first_success = [None] * E
    prog_peak = np.zeros(E); prog = np.zeros(E)  # grader rubric progress (0..1), if a grader exists
    idx = lambda s, k: min(S[s] + k, T[s] - 1)  # noqa: E731  this slot's traj row at clock k
    for k in range(k_end):
        active = np.array([S[s] + k + 1 < T[s] for s in range(E)])
        if cs == "joint_pos" or (cs == "joint_vel" and args.integrate == "dataset"):
            q_t = np.stack([q_rec[s][idx(s, k + 1)] for s in range(E)])
            c_t = np.array([closed_rec[s][idx(s, k + 1)] for s in range(E)], np.float32)
            obs = sim.step_targets(torch.as_tensor(q_t, device=device),
                                   torch.as_tensor(c_t, device=device))
        elif cs == "joint_vel":
            v = np.stack([(q_rec[s][idx(s, k + 1)] - q_rec[s][idx(s, k)]) * rec_rate
                          if active[s] else np.zeros(n_arm, np.float32) for s in range(E)])
            c_t = np.array([closed_rec[s][idx(s, k + 1)] for s in range(E)], np.float32)
            obs = sim.step(np.concatenate([v, c_t[:, None]], axis=1))
        else:  # raw_cmd
            a = np.stack([raw_act[s][idx(s, k)] for s in range(E)])
            rows = torch.as_tensor(a, dtype=torch.float32, device=device)
            frozen = ~active
            if frozen.any():
                fr = torch.as_tensor(frozen, device=device)
                rows[fr] = hold_rows(np.stack([raw_act[s][max(T[s] - 2, 0)]
                                               for s in range(E)]))[fr]
            obs = sim.step(rows)
        if sim.grader is not None:
            prog = sim.grader.progress().cpu().numpy()
            prog_peak = np.maximum(prog_peak, prog)
        tgt = np.stack([q_rec[s][idx(s, k + 1)] for s in range(E)])
        e_t = np.abs(obs["state"][:, :-1] - tgt)
        c_ref = np.array([closed_rec[s][idx(s, k + 1)] for s in range(E)])
        for s in range(E):
            if active[s]:
                err_max[s] = np.maximum(err_max[s], e_t[s])
                err_sum[s] += e_t[s].mean(); err_n[s] += 1
                grip_max[s] = max(grip_max[s], abs(obs["state"][s, -1] - c_ref[s]))
                if first_success[s] is None and bool(obs["success"][s]):
                    first_success[s] = S[s] + k
        for (s, view), w in writers.items():
            w.append_data(obs["images"][view][s])
        if k % 600 == 0:
            print(f"[replay]   k={k}/{k_end} active={int(active[:n].sum())}/{n} "
                  f"err_max={e_t[active].max() if active.any() else 0:.4f} "
                  f"progress={[round(float(p), 3) for p in prog[:n]]} "
                  f"success={obs['success'][:n].tolist()}", flush=True)

    for _ in range(args.extra_hold):
        obs = sim.step(sim.hold_action())
        for (s, view), w in writers.items():
            w.append_data(obs["images"][view][s])
    for w in writers.values():
        w.close()

    for s, e in enumerate(chunk):
        if first_success[s] is None and bool(obs["success"][s]):
            first_success[s] = T[s] - 1
        results.append({
            "episode": str(e), "ticks": T[s], "t0": S[s], "t0_s": round(S[s] / rec_rate, 3),
            "recorded_success": metas[e].get("success"),
            "replay_success": bool(obs["success"][s]), "first_success_tick": first_success[s],
            "err_max": round(float(err_max[s].max()), 5),
            "err_mean": round(float(err_sum[s] / max(err_n[s], 1)), 5),
            "grip_err_max": round(float(grip_max[s]), 5),
            "progress_final": round(float(prog[s]), 4),
            "progress_peak": round(float(prog_peak[s]), 4),
        })
        r = results[-1]
        print(f"[replay] {e.parent.name}/{e.name} (t0={S[s]}): "
              f"replay_success={r['replay_success']} (recorded {r['recorded_success']}) "
              f"err_max={r['err_max']} first_success={r['first_success_tick']}", flush=True)

# ----- report -------------------------------------------------------------------------------------
n_ok = sum(r["replay_success"] for r in results)
n_rec = sum(bool(r["recorded_success"]) for r in results)
summary = {
    "source": args.source, "control_space": cs, "matched_controller": args.matched_controller,
    "t0_frac": args.t0_frac, "integrate": args.integrate if cs == "joint_vel" else None,
    "rate_hz": rec_rate, "physical_params": phys[eps[0]], "episodes": len(results),
    "replay_success": n_ok, "recorded_success": n_rec,
    "err_max": max(r["err_max"] for r in results),
    "per_episode": results,
}
(out / "report.json").write_text(json.dumps(summary, indent=1) + "\n")
print(f"[replay] AGGREGATE: {n_ok}/{len(results)} replay success "
      f"(recorded {n_rec}/{len(results)}), err_max {summary['err_max']}", flush=True)
print(f"[replay] DONE -> {out}", flush=True)

import os  # noqa: E402
import threading  # noqa: E402

watchdog = threading.Timer(10.0, lambda: os._exit(0))
watchdog.daemon = True
watchdog.start()
app.close()
os._exit(0)

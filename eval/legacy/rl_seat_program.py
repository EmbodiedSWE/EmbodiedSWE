#!/usr/bin/env python3
"""Seat-policy RL program on the 512-env bimanual pod (real-benefit target).

Attaches to the durable ikea-bifranka-v7 tree on the fresh pod, restores the braced
threading state (n9), then chains rl_train bursts whose reward comes from the scene's
OWN seat margins (depth/xy/axis — the seated() predicate). Continues each improving
policy with init_from; stops when the trained policy actually seats leg_0 in rollouts
(or after max bursts). Policies persist to HDFS under the session, so the next opus
agent can rl_load them. Video recorded every turn.
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from pathlib import Path

ART = Path("/home/tiger/cap-x/CoSiGen/cosigen_eval_artifacts/rlfix_validation")
REG = "hdfs://haruna/tmp/zeyu.shen/cosigen_render/assembly.ikea_table.bimanual_franka.osc.txt"

SETUP = r"""
import numpy as np
print(list_checkpoints())
print(goto('n9')[:200])
cfg = env.scene.cfg
SEAT_Z, ALIGN_XY, AXIS_DEG = float(cfg.seat_z), float(cfg.align_xy), float(cfg.align_axis_deg)
SLOTS = [np.array(s, dtype=float) for s in cfg.slots]
print("seat_z", SEAT_Z, "align_xy", ALIGN_XY, "axis_deg", AXIS_DEG, "slots", SLOTS)

def _q2R(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])

def _margins(v):
    lp, lq = v.object_pose('leg_0'); tp, tq = v.object_pose('table')
    Rt = _q2R(np.asarray(tq, dtype=float))
    off = Rt.T @ (np.asarray(lp, dtype=float) - np.asarray(tp, dtype=float))
    near = min(float(np.linalg.norm(off[:2] - s)) for s in SLOTS)
    axis = float(np.dot(_q2R(np.asarray(lq, dtype=float))[:, 2], Rt[:, 2]))
    return float(off[2]), near, axis

def _seated0(v):
    try:
        return bool(v.seated()[0])
    except Exception as exc:
        print("seated read failed:", repr(exc))
        return False

def _ruined(offz, near, axis):
    return offz > 0.12 or offz < -0.05 or near > 0.12 or axis < 0.90

def obs_fn(v):
    return v.rich_obs(objects=['leg_0'])

_prev_offz = {}

def reward_fn(v):
    offz, near, axis = _margins(v)
    delta = _prev_offz.get(v._i, offz) - offz   # potential shaping: reward DESCENT
    _prev_offz[v._i] = offz
    r = (SEAT_Z - offz) * 30.0 - near * 20.0 + (axis - 0.99) * 5.0 + delta * 400.0
    if _seated0(v):
        r += 50.0
    if _ruined(offz, near, axis):
        r -= 20.0
    return r

def done_fn(v):
    offz, near, axis = _margins(v)
    return _seated0(v) or _ruined(offz, near, axis)

def probe_fn(v):
    offz, near, axis = _margins(v)
    return {'offz': offz, 'near': near, 'axis': axis,
            'seated': 1.0 if _seated0(v) else 0.0,
            'depth_margin': offz - SEAT_Z}

globals()['_prev_offz'] = _prev_offz
for f in ['_q2R', '_margins', '_seated0', '_ruined', 'obs_fn', 'reward_fn', 'done_fn',
          'probe_fn']:
    globals()[f] = eval(f)
globals()['SEAT_Z'], globals()['ALIGN_XY'], globals()['SLOTS'] = SEAT_Z, ALIGN_XY, SLOTS

# ---- scripted top-cage grasp: episodes must START with the leg IN GRIP ----
# (n9 is checkpointed post-release: the right gripper sits closed ~30 cm ABOVE the
# leg. Training from that state never touches the leg — 2026-07-24 diagnosis.)
lp = np.array(get_object_pose('leg_0')[0])
p1, q1 = get_eef_pose(1)
open_gripper(1); step(10)
# n9 leaves the right wrist in a tilted mid-ratchet orientation (near joint limits;
# holding it made position moves no-ops — xy error frozen at 0.0376 across
# corrections). Reorient FIRST: lift, then a clean straight-down grasp quat.
q_down = np.array([0.0, 1.0, 0.0, 0.0])
p1 = np.array(p1)
move_to(1, [p1[0], p1[1], p1[2] + 0.12], q_down, max_steps=150, pos_tol=0.02)
ep = np.array(get_eef_pose(1)[0]); eq = np.array(get_eef_pose(1)[1])
print("post-reorient eef", np.round(ep, 4).tolist(), "quat", np.round(eq, 3).tolist())
# descend caged around the vertical shaft with iterative re-centering
tgt_xy = lp[:2].copy()
for hover_z, tol in ((lp[2] + 0.32, 0.01), (lp[2] + 0.24, 0.004),
                     (lp[2] + 0.24, 0.003), (lp[2] + 0.24, 0.003)):
    move_to(1, [tgt_xy[0], tgt_xy[1], hover_z], q_down, max_steps=120, pos_tol=tol)
    lp_now = np.array(get_object_pose('leg_0')[0])
    ep_now = np.array(get_eef_pose(1)[0])
    err = lp_now[:2] - ep_now[:2]
    print(f"approach z={hover_z:.3f}: eef z={ep_now[2]:.3f} xy err "
          f"{float(np.linalg.norm(err)):.4f}")
    tgt_xy = tgt_xy + err  # feed-forward the residual
close_gripper(1); step(15)
lp2 = np.array(get_object_pose('leg_0')[0])
ep = np.array(get_eef_pose(1)[0])
fingers = get_link_positions('right.*finger.*')
fd = min(float(np.linalg.norm(np.array(v)[:2] - lp2[:2])) for v in fingers.values())
print("grasp check: leg moved", round(float(np.linalg.norm(lp2 - lp)), 4),
      "eef-leg xy", round(float(np.linalg.norm(ep[:2] - lp2[:2])), 4),
      "min finger-shaft xy dist", round(fd, 4))
from cosigen_loop import AssemblyView
offz, near, axis = _margins(AssemblyView(api, 0))
print("start margins env0: offz", round(offz, 4), "near", round(near, 4),
      "axis", round(axis, 4), "seated", get_seated())
assert fd < 0.03 and near < 0.05, "grasp construction failed -- fingers not at shaft"
"""

BURST = r"""
import numpy as np
res = rl_train(obs_fn, reward_fn, done_fn=done_fn, probe_fn=probe_fn,
               action_spec={'arm': 1, 'wrist_pos': 0.004, 'wrist_rot': 0.10, 'hand': 0.08},
               randomize={'leg_0': {'yaw': 0.1}},
               hidden_sizes=(256, 256), iters=100, budget_s=600, horizon=96,
               minibatches=16, plateau_patience=10{INIT})
print("improved", res.get('improved'), "pid", res.get('pid'))
for c in res.get('curve', []):
    print(c)
print("probe", res.get('probe'))
globals()['last_pid'] = res.get('pid')
rl_save(res.get('pid'), 'seat_leg0_latest')
"""

ROLLOUT = r"""
import numpy as np
print(goto('n9')[:120])
out = rl_run(last_pid, max_steps=600)
print("rl_run:", out)
print("seated:", get_seated())
from cosigen_loop import AssemblyView
offz, near, axis = _margins(AssemblyView(api, 0))
print("final margins: offz", round(offz, 4), "near", round(near, 4), "axis", round(axis, 4))
"""


def post(url, code, meta=None, timeout=7000):
    payload = {"code": code, "reset": False, "max_steps": 100000, "num_frames": 300,
               "meta": dict(meta or {}, **{"arm_recorder": {"every": 2, "max_frames": 1200}})}
    req = urllib.request.Request(url + "/run_policy", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> None:
    url = subprocess.run(["hdfs", "dfs", "-cat", REG], capture_output=True,
                         text=True).stdout.strip()
    ART.mkdir(parents=True, exist_ok=True)
    print(f"[seat] pod {url}")
    res = post(url, SETUP, meta={"session_id": "ikea-bifranka-v7", "resume_tree": True})
    print("[seat] setup rc", res.get("rc"))
    print(res.get("stdout", ""))
    print(res.get("stderr", ""))
    if res.get("rc") != 0:
        raise SystemExit("setup failed")

    best = None
    pid = None  # fresh: the earlier chain trained from an ungripped state (worthless)
    for burst in range(12):
        code = BURST.replace("{INIT}", f", init_from='{pid}'" if pid else "")
        t0 = time.time()
        res = post(url, code)
        out = res.get("stdout", "")
        print(f"===== burst {burst} ({time.time()-t0:.0f}s) rc={res.get('rc')} =====")
        print(out)
        print(res.get("stderr", "")[:1500])
        if res.get("rc") != 0:
            print("[seat] burst errored; stopping for inspection")
            break
        pid = None
        for line in out.splitlines():
            if line.startswith("improved") and "pid" in line:
                pid = line.split()[-1]
        rewards = [float(l.split("'ep_reward': ")[1].split(",")[0])
                   for l in out.splitlines() if "'ep_reward'" in l]
        seated_frac = 0.0
        for line in out.splitlines():
            if line.startswith("probe") and "'seated'" in line:
                try:
                    seated_frac = float(line.split("'seated': {'mean': ")[1].split(",")[0])
                except (IndexError, ValueError):
                    pass
        cur_best = max(rewards) if rewards else None
        print(f"[seat] burst {burst}: pid={pid} best_ep={cur_best} "
              f"probe_seated_mean={seated_frac}")
        (ART / "seat_program_log.jsonl").open("a").write(json.dumps(
            {"burst": burst, "pid": pid, "best_ep": cur_best,
             "seated_probe": seated_frac, "t": time.time()}) + "\n")
        if seated_frac >= 0.25:
            print("[seat] seated fraction target reached — running policy rollout")
            break
        if cur_best is not None and best is not None and cur_best <= best + 1e-6:
            print("[seat] no improvement this burst — one more chance with same policy")
        best = cur_best if best is None else max(best, cur_best)

    res = post(url, ROLLOUT)
    print("===== ROLLOUT =====")
    print(res.get("stdout", ""))
    print(res.get("stderr", "")[:1500])


if __name__ == "__main__":
    main()

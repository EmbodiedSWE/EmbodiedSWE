# Native scripted policy (run via: cosigen_harness.py --policy <this file>):
# 1) fresh-episode arm-tracking check on the bimanual env (the live session's arm went
#    unresponsive after ad-hoc probing; verify a clean episode is healthy),
# 2) an rl_train burst called exactly the way an agent does in-session.
import numpy as np

# --- 1. arm tracking ---------------------------------------------------------------
p0 = np.array(get_eef_pose(1)[0])
d = move_to(1, [p0[0], p0[1] + 0.06, p0[2] - 0.08], None, max_steps=80, pos_tol=0.005)
p1 = np.array(get_eef_pose(1)[0])
moved = float(np.linalg.norm(p1 - p0))
print("ARM CHECK: eef1", np.round(p0, 4).tolist(), "->", np.round(p1, 4).tolist(),
      "| moved", round(moved, 4), "| move_to final dist", round(float(d), 4))
assert moved > 0.03, "fresh-episode arm does not track either -- pod-level problem"

# --- 2. native rl_train burst (agent-style reach MDP on leg_1) ----------------------
def obs_fn(v):
    return v.rich_obs(objects=['leg_1'])

def _d(v):
    e = np.asarray(v.eef_pose(1)[0], dtype=float)
    l = np.asarray(v.object_pose('leg_1')[0], dtype=float)
    return float(np.linalg.norm(e - (l + np.array([0.0, 0.0, 0.15]))))

def reward_fn(v):
    return -_d(v) * 10.0

def done_fn(v):
    return _d(v) > 1.0

def probe_fn(v):
    return {'dist': _d(v)}

res = rl_train(obs_fn, reward_fn, done_fn=done_fn, probe_fn=probe_fn,
               action_spec={'arm': 1, 'wrist_pos': 0.01, 'wrist_rot': 0.0, 'hand': 0.05},
               hidden_sizes=(128, 128), iters=8, budget_s=150, horizon=48)
print("RL CHECK: improved", res.get('improved'), "pid", res.get('pid'))
for c in res.get('curve', []):
    print(c)
print("probe", res.get('probe'))

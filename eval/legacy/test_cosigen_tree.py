#!/usr/bin/env python3
"""Multi-turn client selftest for the CoSiGen checkpoint-tree (backtracking) skill.

Drives the live render server through a scripted multi-turn session and asserts:
  - auto-checkpoint per turn + manual checkpoint(label)
  - goto() restores the WORLD (object poses match the node's recorded scene)
    while the session NAMESPACE persists (variables survive the jump)
  - steps are monotonic (goto does NOT rewind steps_used)
  - list_checkpoints/get_checkpoint_scene/get_checkpoint_code work
  - meta side-channel annotations land in the tree
  - review_rollout/render_checkpoint attach turn_images (when video on)

Usage:  python scripts/tests/test_cosigen_tree.py [--server URL]
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cosigen_harness import discover_server, ping, run_policy  # noqa: E402


def turn(server, code, reset=False, meta=None, num_frames=150):
    r = run_policy(server, code, max_steps=3000, num_frames=num_frames,
                   reset=reset, meta=meta, timeout=900)
    if r.get("rc") != 0:
        print("STDERR:", r.get("stderr", "")[-1500:])
    assert r.get("rc") == 0, "turn crashed"
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=None)
    ap.add_argument("--env", default="assembly.ikea_table.g1.pink_ik")
    args = ap.parse_args()
    server = discover_server(args.env, args.server)
    meta0 = ping(server)
    assert meta0.get("booted"), "server not booted"
    print(f"[test] server={server} num_envs={meta0.get('num_envs')}")

    # -- turn 1: fresh episode; move a bit; stash a variable
    r1 = turn(server, """
# plan: inspect the scene and raise the left wrist a little
stash = {'note': 'survives-jumps'}
p, _ = get_eef_pose(0)
move_to(0, [p[0], p[1], p[2] + 0.08], max_steps=120)
print('steps after t1:', get_robot_state()['steps_used'])
""", reset=True)
    t1 = r1["tree"]
    print(f"[t1] tree={t1}")
    assert t1["n_nodes"] == 2, f"expected root+turn1, got {t1}"  # n1=start, n2=turn1

    # -- turn 2: manual checkpoint, then perturb the world
    r2 = turn(server, """
# plan: checkpoint, then push the wrist far right to perturb state
cid_keep = checkpoint('before-perturb')
leg_before = get_state('leg_0')[:3].tolist()
p, _ = get_eef_pose(0)
move_to(0, [p[0], p[1] - 0.25, p[2]], max_steps=200)
print('kept:', cid_keep, '| leg_before:', leg_before)
""")
    t2 = r2["tree"]
    assert t2["n_nodes"] == 4, f"expected 4 nodes (root,t1,manual,t2), got {t2}"

    # -- turn 3 (with meta annotation for turn 2's node): goto + assertions
    ann = {"annotations": {t2["new_node"]: "perturbed wrist to the right (test annotation)"}}
    r3 = turn(server, """
# plan: jump back to the manual checkpoint and verify world vs namespace
steps_before_goto = get_robot_state()['steps_used']
msg = goto(cid_keep)
print(msg)
steps_after_goto = get_robot_state()['steps_used']
assert steps_after_goto >= steps_before_goto, 'steps must be monotonic'
assert stash['note'] == 'survives-jumps', 'namespace must survive goto'
node_scene = get_checkpoint_scene(cid_keep)
leg_now = get_state('leg_0')[:3]
leg_rec = np.asarray(node_scene['objects']['legs']['leg_0'])
err = float(np.linalg.norm(leg_now - leg_rec))
print('restore err vs recorded scene: %.5f m' % err)
assert err < 5e-3, 'goto must restore recorded object poses'
print('code of kept node has', len(get_checkpoint_code(cid_keep)), 'chars (manual -> 0)')
print('tree now:')
print(list_checkpoints())
print(review_rollout(2))
print(render_checkpoint(cid_keep))
""", meta=ann)
    t3 = r3["tree"]
    print(f"[t3] tree={t3}")
    assert "YOU ARE HERE" in r3["stdout"], "tree printout must mark current node"
    assert "test annotation" in r3["stdout"], "meta annotation must appear in the tree"
    n_imgs = len(r3.get("turn_images") or [])
    print(f"[t3] turn_images={n_imgs}")
    assert n_imgs >= 1, "review_rollout/render_checkpoint should attach images"

    # -- turn 4: branch from the checkpoint (tree gains a sibling branch)
    r4 = turn(server, """
# plan: act differently from the restored state (branch)
p, _ = get_eef_pose(0)
move_to(0, [p[0], p[1] + 0.15, p[2]], max_steps=120)
print(list_checkpoints())
""")
    out = r4["stdout"]
    # after goto(n3)+acting, n3 has two children (turn-2's node and turn-3's node) at depth 3
    assert out.count(" d3 ") >= 2, f"expected a branch (2 nodes at depth 3):\n{out}"
    print("TREE SELFTEST PASSED")


if __name__ == "__main__":
    main()

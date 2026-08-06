# Scripted self-test for the AGENT-TRAINED RL SUB-POLICY skill (phase-4a validation).
# Exercises, end to end, exactly what a model would do across turns -- but scripted, so it
# validates the machinery independent of model quality:
#   introspection -> scripted grasp+carry of leg_0 to above stud_0 -> snapshot/restore
#   fidelity check -> tiny rl_train (seat/press MDP, small budget) -> rl_run -> report.
import time

import numpy as np


def solve_task():
    t0 = time.time()
    print("== introspection ==")
    print("objects:", list_objects())
    rs = get_robot_state()
    print("right wrist:", np.round(rs["right_wrist_pose"][:3], 3).tolist(),
          "hand_frac:", round(rs["hand_frac"], 3), "steps:", rs["steps_used"])
    print("seated:", get_seated())

    print("== scripted grasp + carry (leg_0 -> above stud_0) ==")
    open_gripper(0)
    lp, _ = get_object_pose("leg_0")
    move_to(0, [lp[0], lp[1], lp[2] + 0.12], max_steps=400)
    move_to(0, [lp[0], lp[1], lp[2] + 0.02], max_steps=200)
    close_gripper(0)
    sp, _ = get_object_pose("stud_0")
    move_to(0, [sp[0], sp[1], sp[2] + 0.10], max_steps=400)
    print("carry done: leg_0", np.round(get_object_pose("leg_0")[0], 3).tolist(),
          "stud_0", np.round(sp, 3).tolist())

    print("== checkpoint/goto fidelity ==")
    cid = checkpoint("selftest-fidelity")
    before = get_state("leg_0")
    step(80)  # let things drift
    moved = float(np.linalg.norm(get_state("leg_0")[:3] - before[:3]))
    goto(cid)
    after = get_state("leg_0")
    err = float(np.linalg.norm(after[:3] - before[:3]))
    print(f"checkpoint {cid}: drift_during_perturb={moved:.4f}m restore_err={err:.6f}m")
    assert err < 5e-3, f"goto fidelity FAILED (err={err})"

    print("== tiny rl_train (seat leg_0 on stud_0) ==")

    def obs_fn(v):
        leg_p, leg_q = v.object_pose("leg_0")
        stud_p, _ = v.object_pose("stud_0")
        eef_p, _ = v.eef_pose(0)
        return np.concatenate([leg_p - stud_p, leg_q, eef_p - leg_p,
                               v.object_vel("leg_0"), [v.hand_frac()]])

    def reward_fn(v):
        leg_p, _ = v.object_pose("leg_0")
        stud_p, _ = v.object_pose("stud_0")
        d = float(np.linalg.norm(leg_p - stud_p))
        return -d + (5.0 if v.seated()[0] else 0.0)

    def done_fn(v):
        return v.seated()[0]

    res = rl_train(obs_fn, reward_fn, done_fn=done_fn,
                   action_spec={"arm": 0, "wrist_pos": 0.008, "hand": 0.05},
                   randomize={"leg_0": {"pos": 0.005}},
                   hidden_sizes=(64, 64), iters=6, budget_s=120, horizon=32)
    print("rl_train:", {k: res[k] for k in ("pid", "iters_run", "obs_dim", "act_dim", "seconds")})
    print("curve:", res["curve"])
    assert res["iters_run"] >= 1, "no training iterations completed"

    print("== rl_run ==")
    out = rl_run(res["pid"], max_steps=60, stop_when=lambda v: v.seated()[0])
    print("rl_run:", out, "seated now:", get_seated())

    print("== rl_save/rl_load ==")
    dst = rl_save(res["pid"], "selftest_seat_leg0")
    pid2 = rl_load("selftest_seat_leg0")
    out2 = rl_run(pid2, max_steps=10, obs_fn=obs_fn)
    print("saved->", dst, "| loaded pid:", pid2, "| rerun:", out2)

    print(f"RL SELFTEST PASSED in {time.time() - t0:.0f}s")


solve_task()

# Scripted self-test policy for the CoSiGen assembly env (validates the control API):
# grab leg_0 and lower it onto stud_0. Not expected to fully seat (naive), but exercises
# get_object_pose / move_to / open_close_gripper / get_seated / step end-to-end.
import numpy as np


def solve_task():
    print("seated start:", get_seated())
    open_gripper(0)
    lp, _ = get_object_pose("leg_0")
    print("leg_0:", np.round(lp, 3).tolist())
    move_to(0, [lp[0], lp[1], lp[2] + 0.12], max_steps=400)
    move_to(0, [lp[0], lp[1], lp[2] + 0.02], max_steps=200)
    close_gripper(0)
    sp, _ = get_object_pose("stud_0")
    print("stud_0:", np.round(sp, 3).tolist())
    move_to(0, [sp[0], sp[1], sp[2] + 0.10], max_steps=400)
    move_to(0, [sp[0], sp[1], sp[2] + 0.01], max_steps=300)
    step(60)
    print("seated end:", get_seated())


solve_task()

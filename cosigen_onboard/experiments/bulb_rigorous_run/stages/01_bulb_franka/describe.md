## Scene

1 lamp socket standing upright, fixed on a sturdy table, and 1 loose light bulb lying on the table beside it, ready to be picked up and fitted. Each socket carries a real internal thread.
Goal: pick up the bulb, set it on the socket, and screw it down (turn it clockwise while pressing down) until it seats. A seated bulb is held by its thread, and GLOWS ever brighter as it screws home — from a faint orange at first electrical contact to fully bright warm white exactly when seated. The task is complete once the bulb is seated.

## Robot

A Franka Emika Panda arm with a parallel-jaw gripper, fixed to the table. Control mode 'osc': 7 arm joints by operational-space control (joint torque); the action is 6 end-effector pose deltas, plus 2 gripper fingers by direct position target. Action dim 8.

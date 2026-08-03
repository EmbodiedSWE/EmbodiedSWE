# Skill: you may replace the controller

This world does not pin the preset's controller. If the actuation channel is what is holding the
task back, change it rather than fighting it.

- **Read the controller source first.** It ships in `/bench` under `robobench/controllers/`, and
  the preset's mode is named in the task text. Know what you are replacing before you replace it.
- **Diagnose before switching.** A pose the arm reaches but cannot hold, a command that saturates
  several millimetres short, a wrist that rings after a step change — these are controller
  symptoms. A missed grasp usually is not.
- **The options are real.** `osc` shapes inertia and is smooth on this arm; `impedance` is
  Jacobian-transpose task-space; `joint` takes joint position targets directly and is the most
  predictable when you want exact configurations. You can also write your own and drive
  `env.step()` yourself.
- **Gravity is the usual culprit.** These controllers carry no gravity compensation, so a
  position servo sags configuration-dependently at reach. Measure the sag and correct for it
  instead of raising gains until the arm rings.
- **Change one thing at a time**, and print the tracking error before and after. "It feels
  better" is not a measurement.

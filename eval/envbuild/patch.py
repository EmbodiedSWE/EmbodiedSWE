"""Ablation-arm patches, applied to an extracted tree — enforcement by construction.

Append-override style (never edits function bodies): robust to robobench
internals changing. The /bench mount is read-only in the container, so patches
happen here, at build time, and the agent cannot revert them.
"""

from __future__ import annotations

from pathlib import Path


def disable_set_states(tree: Path) -> None:
    """Make `BaseEnv.set_states` raise, so the agent cannot restore snapshots.

    The task must then be solved from the initial condition `reset()`
    establishes, stepping forward — no jumping to a saved mid-task state.
    `get_states` and the asset handles stay readable.
    """
    envpy = tree / "robobench" / "core" / "env.py"
    block = [
        "",
        "",
        "# --- EXPERIMENT ARM PATCH: set_states disabled (applied at build time) ---",
        "def _eval_arm_blocked(self, *args, **kwargs):",
        "    raise PermissionError('env.set_states is disabled in this experiment arm')",
        "BaseEnv.set_states = _eval_arm_blocked",
    ]
    with envpy.open("a") as f:
        f.write("\n".join(block) + "\n")


def freeze_control_mode(tree: Path) -> None:
    """Make the preset's controller the only reachable one, at both chokepoints:

    - `EnvCfg.build` refuses a `control_mode` override, so every env built in
      this world carries the preset's mode;
    - `BaseRobot.set_controller` refuses once a controller is bound, so a live
      env's controller cannot be swapped (matters at grading time, when
      solve(env) receives a live env).

    Controller SOURCE stays readable — the ablation is the actuation channel,
    not knowledge.
    """
    cfgpy = tree / "robobench" / "core" / "config.py"
    with cfgpy.open("a") as f:
        f.write("\n".join([
            "",
            "",
            "# --- EXPERIMENT PATCH: control mode frozen to the preset (applied at build time) ---",
            "_eval_orig_build = EnvCfg.build",
            "def _eval_frozen_build(self, **overrides):",
            "    if 'control_mode' in overrides and overrides['control_mode'] != self.control_mode:",
            "        raise PermissionError('control_mode is frozen to the preset in this experiment')",
            "    return _eval_orig_build(self, **overrides)",
            "EnvCfg.build = _eval_frozen_build",
        ]) + "\n")

    robotpy = tree / "robobench" / "core" / "robot.py"
    with robotpy.open("a") as f:
        f.write("\n".join([
            "",
            "",
            "# --- EXPERIMENT PATCH: controller locked after the initial bind (applied at build time) ---",
            "_eval_orig_set_controller = BaseRobot.set_controller",
            "def _eval_locked_set_controller(self, controller):",
            "    if getattr(self, 'controller', None) is not None:",
            "        raise PermissionError('the controller is frozen to the preset in this experiment')",
            "    return _eval_orig_set_controller(self, controller)",
            "BaseRobot.set_controller = _eval_locked_set_controller",
        ]) + "\n")

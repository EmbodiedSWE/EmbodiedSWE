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

"""Runnable scripts for the pouring suite (under the Newton venv, from the repo root):

  - `latte_bimanual_weld_smoke.py` — THE bimanual latte smoke: two dynamic Frankas grasp both
    vessels and pour milk into the coffee, with a transfer/retention/spill verdict. Modes:
    default = scene `latte_weld` (scripted welds), `--auto` = `latte_auto` (grasps engage from
    gripper proximity + closure — the agent benchmark), `--feed` = `latte_feed` (`--auto` plus
    1.5-way liquid feedback).

Invocation: `env_newton/bin/python -m robobench.suites.pouring.scripts.latte_bimanual_weld_smoke [flags]`.
"""

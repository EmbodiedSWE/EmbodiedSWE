"""Runnable scripts for the pouring suite (under the Newton venv, from the repo root):

  - `latte_bimanual_weld_smoke.py` — THE bimanual latte smoke on the benchmark env
    (`pouring.latte.bimanual_franka.joint`): two dynamic Frankas grasp both vessels
    through the scene's auto-weld contract (proximity + closure engages, opening releases) and
    pour milk into the coffee under 1.5-way liquid feedback, with a transfer/retention/spill
    verdict.

Invocation: `env_newton/bin/python -m robobench.suites.pouring.scripts.latte_bimanual_weld_smoke [flags]`.
"""

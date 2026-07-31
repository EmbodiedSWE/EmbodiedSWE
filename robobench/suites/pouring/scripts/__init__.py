"""Runnable scripts for the pouring suite (under the Newton venv, from the repo root):

  - `latte_pour_smoke.py` — simulation capability check on the benchmark env
    (`pouring.latte.bimanual_franka.joint`): both Frankas grasp the vessels through the scene's
    auto-weld contract and lift them, with rise/upright/spill verdicts. Not a solution — the
    bimanual-Franka solution lives outside the benchmark tree (gitignored `experiments/`).

Invocation: `env_newton/bin/python -m robobench.suites.pouring.scripts.latte_pour_smoke [flags]`.
"""

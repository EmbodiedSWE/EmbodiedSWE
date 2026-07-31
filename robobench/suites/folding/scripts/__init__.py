"""Runnable scripts for the folding suite (under the Newton venv, from the repo root):

  - `tshirt_fold_smoke.py` — simulation capability check on the benchmark env
    (`folding.tshirt.franka.joint`): the Franka pinches the shirt with its real fingers and
    lifts it clear of the table, with a cloth-rise verdict. Not a solution — the Franka folding
    solution lives outside the benchmark tree (gitignored `experiments/`).

Invocation: `env_newton/bin/python -m robobench.suites.folding.scripts.tshirt_fold_smoke [flags]`.
"""

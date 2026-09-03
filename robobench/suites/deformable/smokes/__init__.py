"""Runnable smokes for the deformable suite (under the Newton venv, from the repo root):

  - `tshirt_fold_smoke.py` — simulation capability check on `deformable.tshirt.franka.joint`:
    the Franka pinches the shirt with its real fingers and lifts it clear of the table, with a
    cloth-rise verdict. Not a solution.
  - `latte_pour_smoke.py`  — the coupled-substrate pour on `deformable.latte.bimanual_franka.joint`.
  - `knot_smoke.py`        — the four-beat half knot on `deformable.knot` (kinematic handles).

Invocation: `env_newton/bin/python -m robobench.suites.deformable.smokes.<name> [flags]`.
"""

"""Canonical runnable env configs for the shoe_tying suite — registered in `ENVS` by name.

These run ONLY under the Newton venv (`env_newton`, newton >= 1.6 — see the README); registration
itself stays app-free in any venv.
"""

from __future__ import annotations

from robobench.core import EnvCfg, register_env

SUITE = "shoe_tying"

# Two separate laces rooted at the sneaker's top eyelets, ends free; tie a self-holding half
# knot. Scene physics only: the rod ends are kinematic handles driven straight through the
# Newton manager (no robot in the loop yet), the way the standalone scripts drove them.
# Robot-less -> standalone VBD manager (cable joints are not MuJoCo-convertible).
# -> "shoe_tying.knot"
register_env(SUITE, lambda: EnvCfg(scene="knot", robot="null", env_spacing=2.0))

"""sampler — per-env world-physics draws from the scene's own PHYSICAL_PARAMS bands.

THE sampling declaration lives in the scene class (no side files): `PHYSICAL_PARAMS` on a
`BaseScene` subclass maps each post-build-appliable cfg field to a distribution spec —
pre-baked reasonable bands, authored next to the knowledge that justifies them (the
scene's own comments document the knees):

    PHYSICAL_PARAMS = {
        "bulb_glass_friction": {"dist": "uniform", "lo": 0.30, "hi": 0.45},
        "part_mass":           {"dist": "gaussian", "mean": 0.05, "std": 0.01, "lo": 0.02},
        "socket_friction":     None,   # appliable but NOT sampled (bind still applies nominal)
    }

`dist` is one of uniform | loguniform | gaussian (optional lo/hi truncation) | choice;
`reason` is optional free text carried into the meta. A cell adjusts bands by editing
its own scene.py copy — the same way it edits any other part of its world.

Nominal is IMPLICIT — a band never repeats it: the cfg default IS the nominal, and
env slot 0 of every batch keeps it (the in-batch canary; see engine/generation.py).
`--nominal` skips sampling entirely (the baseline batch). Draws are a pure function of
the index: Halton low-discrepancy per dimension (sorted name order), `choice` cycled,
so draw k is the same value on any machine, forever — sharding and reruns need no
coordination and no stored state. Reset/initialization randomization is NOT sampled
here; it belongs to the scene's own reset and the phase reset conditions. Grading
thresholds and controller gains are never banded.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

DISTS = ("uniform", "loguniform", "gaussian", "choice")
# per-dist required/allowed spec keys (besides "dist"); "reason" is allowed everywhere
_KEYS = {
    "uniform": ({"lo", "hi"}, set()),
    "loguniform": ({"lo", "hi"}, set()),
    "gaussian": ({"mean", "std"}, {"lo", "hi"}),
    "choice": ({"options"}, set()),
}
_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71)


def solve_bands(solve_module: Any) -> dict[str, dict[str, Any]]:
    """The solve's validated sampling declaration: `SOLVE_PARAMS` maps module-level
    CONSTANTS of solve.py to distribution specs (None = declared, not sampled). The
    value assigned in the file IS the nominal; `solve(env)` never changes signature —
    the engine writes the drawn values onto the module before calling it. One set is
    drawn PER BATCH (`--solve_draw`). No `SOLVE_PARAMS` -> nominal, solves run
    unchanged."""
    src = f"{getattr(solve_module, '__name__', 'solve')}.SOLVE_PARAMS"
    declared = getattr(solve_module, "SOLVE_PARAMS", {}) or {}
    bands: dict[str, dict[str, Any]] = {}
    for name, spec in declared.items():
        if not hasattr(solve_module, name):
            raise ValueError(f"{src}: '{name}' is not a module-level constant of the solve")
        if spec is None:
            continue
        bands[name] = _check_spec(src, name, spec)
    return bands


def scene_bands(scene_cls: type, scene_cfg: Any) -> dict[str, dict[str, Any]]:
    """The scene's validated sampling declaration: the `PHYSICAL_PARAMS` entries that carry
    a spec (None entries are appliable-but-not-sampled). Fails loudly — a bad band must
    kill the batch before the expensive build, never silently go nominal. Every name
    must be a real field of the scene's cfg (the nominal source)."""
    src = f"{scene_cls.__name__}.PHYSICAL_PARAMS"
    bands: dict[str, dict[str, Any]] = {}
    for name, spec in getattr(scene_cls, "PHYSICAL_PARAMS", {}).items():
        if not hasattr(scene_cfg, name):
            raise ValueError(f"{src}: '{name}' is not a field of {type(scene_cfg).__name__}")
        if spec is None:
            continue
        bands[name] = _check_spec(src, name, spec)
    return bands


def _check_spec(src: str, name: str, spec: Any) -> dict[str, Any]:
    if not isinstance(spec, dict) or "dist" not in spec:
        raise ValueError(f"{src}: '{name}' must be a mapping with a `dist` key (or None)")
    dist = spec["dist"]
    if dist not in DISTS:
        raise ValueError(f"{src}: '{name}': unknown dist '{dist}' (one of {', '.join(DISTS)})")
    required, optional = _KEYS[dist]
    keys = set(spec) - {"dist", "reason"}
    if not required <= keys:
        raise ValueError(f"{src}: '{name}' ({dist}) missing {sorted(required - keys)}")
    if keys - required - optional:
        raise ValueError(f"{src}: '{name}' ({dist}) has unknown keys {sorted(keys - required - optional)}")
    if dist != "choice":
        for k in keys:
            try:
                spec[k] = float(spec[k])
            except (TypeError, ValueError):
                raise ValueError(f"{src}: '{name}': `{k}` must be a number, got {spec[k]!r}") from None
    if dist in ("uniform", "loguniform"):
        lo, hi = spec["lo"], spec["hi"]
        if not lo < hi:
            raise ValueError(f"{src}: '{name}': lo must be < hi")
        if dist == "loguniform" and lo <= 0:
            raise ValueError(f"{src}: '{name}': loguniform needs lo > 0")
    elif dist == "gaussian":
        if spec["std"] <= 0:
            raise ValueError(f"{src}: '{name}': std must be > 0")
        if "lo" in spec and "hi" in spec and not spec["lo"] < spec["hi"]:
            raise ValueError(f"{src}: '{name}': lo must be < hi")
    elif not spec["options"]:
        raise ValueError(f"{src}: '{name}': choice needs non-empty options")
    return dict(spec)


def sample(bands: dict[str, dict[str, Any]], index: int) -> dict[str, Any]:
    """{name: value} for one draw index — a pure function of (bands, index); {} for
    empty bands. Halton low-discrepancy per dimension in sorted name order; `choice`
    cycles its options so coverage is even at any count."""
    if not bands:
        return {}
    if index < 0:
        raise ValueError("index must be >= 0")
    out: dict[str, Any] = {}
    for dim, name in enumerate(sorted(bands)):
        spec = bands[name]
        if spec["dist"] == "choice":
            out[name] = spec["options"][index % len(spec["options"])]
            continue
        u = _halton(index + 1, _PRIMES[dim % len(_PRIMES)])  # +1: radical-inverse(0)=0 would pin u=0
        if spec["dist"] == "uniform":
            out[name] = spec["lo"] + u * (spec["hi"] - spec["lo"])
        elif spec["dist"] == "loguniform":
            out[name] = math.exp(math.log(spec["lo"]) + u * (math.log(spec["hi"]) - math.log(spec["lo"])))
        else:  # gaussian, optionally truncated by clipping
            v = statistics.NormalDist(spec["mean"], spec["std"]).inv_cdf(u)
            if "lo" in spec:
                v = max(v, spec["lo"])
            if "hi" in spec:
                v = min(v, spec["hi"])
            out[name] = v
    return out


def _halton(index: int, base: int) -> float:
    """Radical-inverse of `index` in `base` — the Halton sequence, in (0, 1) for index >= 1."""
    u, f = 0.0, 1.0
    while index > 0:
        f /= base
        u += f * (index % base)
        index //= base
    return u

"""sampler — per-episode/per-batch parameters drawn from the cells' params.yaml files.

THE sampling declaration is data, not code: an optional, FLAT `params.yaml` per cell —
the file's location says which layer it feeds, so there are no section headers:

    scenes/<scene_cell>/params.yaml                 world physics -> scene-cfg field overrides
    .../strategies/<strategy_N>/params.yaml         solve parameters -> solve(...) kwargs

Each top-level key is a parameter name mapped to a distribution spec:

    bulb_glass_friction: {dist: uniform, lo: 0.30, hi: 0.45, reason: "knee at 0.3"}
    nut_mass:            {dist: gaussian, mean: 0.03, std: 0.008, lo: 0.015}
    approach_side:       {dist: choice, options: ["+y", "-y"]}

`dist` is one of uniform | loguniform | gaussian | choice; `reason` is optional free
text carried into the meta. One reserved key: `frozen:` — a {name: note} map of
parameters deliberately out of reach (a name both frozen and banded is an error).

Nominal is IMPLICIT — the yaml never states it. The scene cfg's default is the env
nominal; the solve kwarg's default is the solve nominal. A cell WITHOUT a params.yaml
is nominal at every index (existing cells run unchanged), and the nominal baseline
batch stays what it always was — a batch run without (or ignoring) the yamls. When a
yaml IS present, every index >= 0 is a genuine draw: no reserved canary index.

Draws are a pure function of the index: Halton low-discrepancy per dimension
(dimensions in sorted name order), `choice` cycled. Env parameters vary per BATCH
(the env is built once per batch; the cfg is patched before build), solve parameters
per ROLLOUT (one batched solve() call parameterizes all its envs together).
Reset/initialization randomization is NOT sampled here; it belongs to the scene's
own reset and the phase reset conditions.

Grading/feedback law is never declared here — that is what `frozen:` is for.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PARAMS_FILE = "params.yaml"
DISTS = ("uniform", "loguniform", "gaussian", "choice")
# per-dist required/allowed spec keys (besides "dist"); "reason" is allowed everywhere
_KEYS = {
    "uniform": ({"lo", "hi"}, set()),
    "loguniform": ({"lo", "hi"}, set()),
    "gaussian": ({"mean", "std"}, {"lo", "hi"}),
    "choice": ({"options"}, set()),
}
_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71)


# ----- declaration loading ----------------------------------------------------------------------
@dataclass
class Declaration:
    """One cell's parsed params.yaml: {name: spec} bands + the frozen {name: note} map."""

    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    frozen: dict[str, str] = field(default_factory=dict)
    source: str = ""  # the file it came from ("" -> absent: nominal-only)

    def to_meta(self) -> dict[str, Any]:
        """The declaration verbatim, for the batch meta.json (provenance)."""
        return {"source": self.source, "params": self.params, "frozen": self.frozen}


def load_declaration(cell_dir: Path) -> Declaration:
    """Parse `<cell_dir>/params.yaml`; absent file -> the empty (nominal-only) declaration.
    Validation fails loudly — a typo in a band must kill the batch, not silently go nominal."""
    path = Path(cell_dir) / PARAMS_FILE
    if not path.is_file():
        return Declaration()
    import yaml

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping of param name -> spec")
    frozen = raw.pop("frozen", {}) or {}
    if not isinstance(frozen, dict) or not all(isinstance(v, str) for v in frozen.values()):
        raise ValueError(f"{path}: `frozen` must map names to a short note (string)")
    params: dict[str, dict[str, Any]] = {}
    for name, spec in raw.items():
        params[name] = _check_spec(path, name, spec)
        if name in frozen:
            raise ValueError(f"{path}: '{name}' is both banded and frozen")
    return Declaration(params, dict(frozen), str(path))


def _check_spec(path: Path, name: str, spec: Any) -> dict[str, Any]:
    if not isinstance(spec, dict) or "dist" not in spec:
        raise ValueError(f"{path}: '{name}' must be a mapping with a `dist` key")
    dist = spec["dist"]
    if dist not in DISTS:
        raise ValueError(f"{path}: '{name}': unknown dist '{dist}' (one of {', '.join(DISTS)})")
    required, optional = _KEYS[dist]
    keys = set(spec) - {"dist", "reason"}
    if not required <= keys:
        raise ValueError(f"{path}: '{name}' ({dist}) missing {sorted(required - keys)}")
    if keys - required - optional:
        raise ValueError(f"{path}: '{name}' ({dist}) has unknown keys {sorted(keys - required - optional)}")
    if dist != "choice":  # coerce numerics — pyyaml reads unsigned-exponent floats ("1.0e5") as str
        for k in keys:
            try:
                spec[k] = float(spec[k])
            except (TypeError, ValueError):
                raise ValueError(f"{path}: '{name}': `{k}` must be a number, got {spec[k]!r}") from None
    if dist in ("uniform", "loguniform"):
        lo, hi = spec["lo"], spec["hi"]
        if not lo < hi:
            raise ValueError(f"{path}: '{name}': lo must be < hi")
        if dist == "loguniform" and lo <= 0:
            raise ValueError(f"{path}: '{name}': loguniform needs lo > 0")
    elif dist == "gaussian":
        if spec["std"] <= 0:
            raise ValueError(f"{path}: '{name}': std must be > 0")
        if "lo" in spec and "hi" in spec and not spec["lo"] < spec["hi"]:
            raise ValueError(f"{path}: '{name}': lo must be < hi")
    elif not spec["options"]:
        raise ValueError(f"{path}: '{name}': choice needs non-empty options")
    return dict(spec)


def validate_env_keys(decl: Declaration, scene_cfg: Any) -> None:
    """Every env param (and frozen name) must be a real field of the cell's scene cfg —
    called by generation once the cfg exists, so typos fail before any rollout."""
    for name in (*decl.params, *decl.frozen):
        if not hasattr(scene_cfg, name):
            raise ValueError(f"{decl.source}: '{name}' is not a field of {type(scene_cfg).__name__}")


# ----- index-addressed draws --------------------------------------------------------------------
def sample(decl: Declaration, index: int) -> dict[str, Any]:
    """{name: value} for one index — a pure function of (declaration, index). Every
    index >= 0 is a genuine draw (an empty declaration -> {} = nominal; the nominal
    BASELINE is a batch run without yamls). Halton low-discrepancy per dimension in
    sorted name order; `choice` cycles its options so coverage is even at any count."""
    if not decl.params:
        return {}
    if index < 0:
        raise ValueError("index must be >= 0")
    out: dict[str, Any] = {}
    for dim, name in enumerate(sorted(decl.params)):
        spec = decl.params[name]
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

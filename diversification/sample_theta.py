"""Draw one theta from params_nut_thread.yaml, deterministically from an episode index.

Index-addressed on purpose: a pod is told "you are episode 37" and derives the same theta every
time, so a re-run of episode 37 samples the same parameters and no coordinator has to hand out
values. Continuous bands are drawn from a low-discrepancy sequence rather than uniform noise —
at ~100 points uniform draws leave visible clumps and holes in a 12-dimensional box. Discrete
strategy choices are cycled by index so every one of them appears a fair number of times instead
of being left to chance.

Episode 0 is always the nominal theta: the parameters the script was solved at. It is the
control, and if it fails, the harness is broken rather than the sampling being too wide.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

SPEC = Path(__file__).resolve().parent / "params_nut_thread.yaml"


def spec_path(task: str) -> Path:
    return Path(__file__).resolve().parent / f"params_{task}.yaml"


def load_spec(path: str | Path = SPEC) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _lowdisc(n_dims: int, index: int) -> list[float]:
    """One point of a low-discrepancy sequence, in [0,1)^n_dims, from its index alone.

    The additive-recurrence (Kronecker) sequence with the generalised golden-ratio bases: point k
    is frac(k * a_j) with a_j = phi_d^-(j+1), phi_d the root of x^(d+1) = x + 1. It fills the box
    far more evenly than uniform draws at ~100 points, needs no dependency (scipy is absent here
    and I will not have theta depend on which machine drew it), and is a pure function of the
    index, so episode 37 is the same parameters everywhere and forever.
    """
    phi = 2.0
    for _ in range(64):  # Newton on x^(d+1) - x - 1
        phi -= (phi ** (n_dims + 1) - phi - 1.0) / ((n_dims + 1) * phi ** n_dims - 1.0)
    return [((index + 1) * phi ** -(j + 1)) % 1.0 for j in range(n_dims)]


def theta_for(index: int, n_total: int, spec: dict | None = None) -> dict[str, Any]:
    """Theta for episode `index` out of `n_total`."""
    spec = spec or load_spec()
    cont: list[tuple[str, str, list]] = []
    for section in ("env", "policy"):
        for name, d in spec[section].items():
            if "range" in d and name != "seed":
                cont.append((section, name, d["range"]))
    theta: dict[str, Any] = {"_index": index, "_nominal": index == 0}

    theta.update(spec.get("constants") or {})   # frozen values the program still reads from theta

    if index == 0:
        for section, name, _ in cont:
            theta[name] = spec[section][name]["nominal"]
        for name, d in spec["strategy"].items():
            theta[name] = d["nominal"]
        theta["seed"] = 0
        return theta

    u = _lowdisc(len(cont), index)
    for j, (section, name, rng) in enumerate(cont):
        lo, hi = float(rng[0]), float(rng[1])
        theta[name] = lo + float(u[j]) * (hi - lo)
    # Strategy dims come from FURTHER coordinates of the same sequence. Cycling them with a
    # multiplied divisor (div *= len(choices)) starves the trailing dimensions: with 5x3x3x3x2
    # choices and 100 episodes the last divisor reached 135, so `all_pens` was False in all 100
    # episodes of pen_b1 and `slide_step_deg` took one value per 45-episode block.
    keys = list(spec["strategy"].keys())
    ud = _lowdisc(len(cont) + len(keys), index)[len(cont):]
    for j, name in enumerate(keys):
        choices = spec["strategy"][name]["choices"]
        theta[name] = choices[min(len(choices) - 1, int(ud[j] * len(choices)))]
    theta["seed"] = 1_000 + index
    return theta


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--show", type=int, default=4, help="print the first N thetas")
    a = ap.parse_args()
    spec = load_spec()
    for i in range(min(a.show, a.n)):
        print(json.dumps(theta_for(i, a.n, spec), sort_keys=True))
    # Coverage report over the whole batch: how often each strategy combination appears.
    from collections import Counter
    combos = Counter()
    for i in range(a.n):
        t = theta_for(i, a.n, spec)
        combos[(t["sweep_deg"], t["nut_center"], t["w_land_deg"])] += 1
    print(f"\n{a.n} episodes -> {len(combos)} strategy combinations")
    for k, v in sorted(combos.items()):
        print(f"  sweep={k[0]:>5} nut_center={str(k[1]):>5} w_land={k[2]:>4}: {v}")

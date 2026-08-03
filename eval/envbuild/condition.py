"""One experimental condition, read from one yaml file.

Design note §1: which skills, tools and code features an agent gets is selected by a single
declaration per condition, and everything downstream — prompt text, skill installation, tool
registration, method-level gating — derives from it. This module is that declaration's only
reader, so no consumer parses a config itself and no consumer invents a default.

    rules:    [autonomous_operation]      # must follow; markdown under prompts/rules/
    skills:   [save_checkpoint]           # may follow; markdown under prompts/skills/
    tools:    [checkpoint_tree]           # callable tools from the eval/tools/ library
    features: {set_states: false}         # code gating; every feature defaults to TRUE

Two invariants are enforced at load, because both are cheap here and expensive at run time:

  * a name must exist in its library (a typo'd rule silently ungranted is an unnoticed
    condition change, which makes an experiment uncitable);
  * a granted tool keeps the features it runs on (§5) — `evolutionary_parameter_search` with
    `set_states: false` is contradictory, and refusing at load beats failing mid-run.

`facts` validation stays where it was: a prompt may WITHHOLD what the world does, never
contradict it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tools as tool_lib

EVAL_DIR = Path(__file__).resolve().parents[1]
CONFIGS_DIR = EVAL_DIR / "configs"
PROMPTS_DIR = EVAL_DIR / "prompts"
RULES_DIR = PROMPTS_DIR / "rules"
SKILLS_DIR = PROMPTS_DIR / "skills"

# Each feature maps to the patch that ENFORCES its absence, applied to the extracted tree at
# build time (envbuild/patch.py). A feature not listed here is declarative only — recorded in
# the receipt and honoured by the driver, with no build-time patch to apply.
FEATURE_PATCHES = {
    "set_states": "disable_set_states",
    "control_mode": "freeze_control_mode",
}

# A selected rule/skill may presume something about the world; the value the fact must have for
# the text to be true. Withholding is a valid experimental variant, contradicting is a bug.
NEEDS = {
    "save_checkpoint": ("set_states", True),
    "no_checkpoint": ("set_states", False),
    "frozen_controller": ("control_mode_frozen", True),
}


@dataclass(frozen=True)
class Condition:
    """A loaded condition. `path` is kept so a run record can cite the exact file."""

    path: Path
    rules: tuple[str, ...] = ()
    skills: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    features: dict[str, bool] = field(default_factory=dict)

    # ----- features ------------------------------------------------------------------------
    def enabled(self, feature: str) -> bool:
        """Every feature defaults to TRUE; only an explicit false blocks it."""
        return bool(self.features.get(feature, True))

    def patches(self) -> list[str]:
        """`patch.py` function names to apply for the features this condition blocks."""
        return [fn for feat, fn in FEATURE_PATCHES.items() if not self.enabled(feat)]

    # ----- tools ---------------------------------------------------------------------------
    def tool_specs(self) -> list[tool_lib.ToolSpec]:
        library = tool_lib.discover()
        return [library[name] for name in self.tools]

    def as_record(self) -> dict:
        """The condition as it goes into a run/build receipt — one citable summary."""
        return {
            "config": str(self.path),
            "rules": list(self.rules),
            "skills": list(self.skills),
            "tools": list(self.tools),
            "features": dict(self.features),
        }


def _library(directory: Path) -> list[str]:
    return sorted(p.stem for p in directory.glob("*.md"))


def list_rules() -> list[str]:
    return _library(RULES_DIR)


def list_skills() -> list[str]:
    return _library(SKILLS_DIR)


def list_tools() -> list[str]:
    return sorted(tool_lib.discover())


def resolve_config(name_or_path: str | Path) -> Path:
    """Accept a bare condition name or a path. Bare names resolve under eval/configs/."""
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p.resolve()
    candidate = CONFIGS_DIR / f"{p.name if p.suffix else p}.yaml"
    if candidate.exists():
        return candidate.resolve()
    known = ", ".join(sorted(c.stem for c in CONFIGS_DIR.glob("*.yaml"))) or "(none)"
    raise SystemExit(f"unknown condition '{name_or_path}'; eval/configs/ has: {known}")


def load(name_or_path: str | Path) -> Condition:
    """Read + validate one condition file. Raises SystemExit on anything questionable."""
    import yaml

    path = resolve_config(name_or_path)
    loaded = yaml.safe_load(path.read_text())
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path}: a condition must be a yaml mapping, got {type(loaded).__name__}")

    unknown_keys = set(loaded) - {"rules", "skills", "tools", "features"}
    # Infra options (agent, budget, gpu, ...) live on the runner's CLI, not in a condition: a
    # condition names WHAT the agent is granted, so a stray key here is usually a typo that
    # would otherwise be silently ignored.
    if unknown_keys:
        raise SystemExit(
            f"{path}: unknown condition keys {sorted(unknown_keys)}; "
            f"a condition declares only rules/skills/tools/features"
        )

    features = loaded.get("features") or {}
    if not isinstance(features, dict) or any(not isinstance(v, bool) for v in features.values()):
        raise SystemExit(f"{path}: features must be a mapping of name -> true/false")

    cond = Condition(
        path=path,
        rules=tuple(loaded.get("rules") or ()),
        skills=tuple(loaded.get("skills") or ()),
        tools=tuple(loaded.get("tools") or ()),
        features=dict(features),
    )
    _validate_libraries(cond)
    _validate_tool_features(cond)
    return cond


def _validate_libraries(cond: Condition) -> None:
    for kind, selected, library in (
        ("rule", cond.rules, list_rules()),
        ("skill", cond.skills, list_skills()),
        ("tool", cond.tools, list_tools()),
    ):
        for name in selected:
            if name not in library:
                raise SystemExit(
                    f"{cond.path}: unknown {kind} '{name}'; library: {', '.join(library) or '(empty)'}"
                )
    unknown_features = set(cond.features) - set(FEATURE_PATCHES)
    if unknown_features:
        raise SystemExit(
            f"{cond.path}: unknown features {sorted(unknown_features)}; "
            f"known: {', '.join(sorted(FEATURE_PATCHES))}"
        )


def _validate_tool_features(cond: Condition) -> None:
    """Design note §5: a granted tool keeps the features it runs on."""
    for spec in cond.tool_specs():
        for feature in spec.requires_features:
            if not cond.enabled(feature):
                raise SystemExit(
                    f"{cond.path}: invalid condition — tool '{spec.name}' runs on "
                    f"'{feature}', which this config sets false. Either drop the tool or "
                    f"leave the feature enabled (cheating is policed by rule and grading)."
                )


def check_facts(cond: Condition, facts: dict) -> None:
    """A prompt may WITHHOLD a fact about the world, never contradict it."""
    for name in (*cond.rules, *cond.skills):
        if name not in NEEDS:
            continue
        fact, wanted = NEEDS[name]
        if facts.get(fact) != wanted:
            raise SystemExit(
                f"{cond.path}: '{name}' presumes {fact}={wanted}, but this world has "
                f"{fact}={facts.get(fact)} — the prompt may not contradict the world"
            )

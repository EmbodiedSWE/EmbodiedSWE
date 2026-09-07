"""Assessment — write down what a run actually did before starting the next one.

    from assessment import assess, history

    assess(scene="part_0 in place; part_1 tipped over next to its goal",
           log="goal metric 3.1mm of 5mm; gripper lost contact at t=210",
           failure_modes="grasp slips during the transfer — approach angle too steep",
           keep=True, label="part_0 in place", env=env,
           action="transfer part_0", outcome="partial", program=__file__)

    print(history())        # in a LATER script: what did I already conclude?

The point is the discipline the old harness enforced between runs, kept as a callable record:
say what the images show, what the numbers say, and what failed — in words — before touching
the code again. Two things make this worth calling rather than just thinking:

  * the record PERSISTS (``/workspace/.assessments/reviews.jsonl``). Each of your scripts is a
    fresh process with no memory; `history()` is how a later run recalls what was already tried
    and concluded, instead of rediscovering it;
  * `keep=True` couples the verdict to the checkpoint tree: a state judged worth building on is
    saved AS PART OF the judgment (pass `tree`, or `env` to open one loudly), so "good state"
    and "saved state" cannot drift apart.

`assess` refuses empty prose — "see log" recorded three runs in a row is how an agent loops on
the same failure. If the outcome rests on constants you picked by hand, say in `constants_plan`
whether you are handing them to parameter_search, or why not.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# Self-declaration, read by eval/tools/__init__.py::discover(). A plain dict on purpose: this
# file is also installed standalone on the agent's PYTHONPATH, where a relative import of the
# tools package would fail.
TOOL = {
    "name": "assessment",
    "description": (
        "Structured per-run self-review persisted to disk: what the images show, what the "
        "numbers say, what failed — with keep=True saving the state to the checkpoint tree, "
        "and history() recalling past conclusions in later scripts."
    ),
    "exports": ("assess", "history"),
    "prompt_doc": "tools/assessment.md",
}

DEFAULT_ROOT = "/workspace/.assessments"
OUTCOMES = ("ok", "partial", "failed", "crashed")


def _path(root: str | Path | None) -> Path:
    p = Path(root or DEFAULT_ROOT)
    p.mkdir(parents=True, exist_ok=True)
    return p / "reviews.jsonl"


def _open_checkpoint_tree(env, checkpoint_root: str | Path | None, reason: str):
    if __package__:
        from .checkpoint_tree import CheckpointTree
    else:
        from checkpoint_tree import CheckpointTree

    chosen_root = checkpoint_root or getattr(env, "checkpoint_root", None)
    print(
        f"[assessment] AUTO-OPENING CheckpointTree(env) to {reason}"
        + (f" at {chosen_root}" if chosen_root else ""),
        flush=True,
    )
    # assess() is called at the END of a world-moving run. The live state must not be
    # mislabeled as canonical n0 merely because this is the first tree construction.
    kwargs = {"assume_reset": False}
    return (CheckpointTree(env, root=chosen_root, **kwargs)
            if chosen_root else CheckpointTree(env, **kwargs))


def assess(
    scene: str,
    log: str,
    failure_modes: str,
    *,
    keep: bool = False,
    label: str = "",
    tree=None,
    env=None,
    program: str | Path | None = None,
    action: str = "",
    outcome: str = "ok",
    constants_plan: str = "",
    root: str | Path | None = None,
    checkpoint_root: str | Path | None = None,
) -> dict:
    """Record one run's review. Returns the record (with `checkpoint` when one was saved).

    scene           what the end-state image(s) show — where things actually ended up
    log             what the printed numbers say, including the measurements that matter
    failure_modes   what went wrong or blocks progress; say plainly if the run went as intended
    keep            True when this end state is worth building on
    label           the state reached (required with keep) — it becomes the checkpoint label
    tree            a checkpoint_tree.CheckpointTree; with keep=True the state is saved to it
    env             live environment; opens CheckpointTree(env) when one is needed and tree is absent
    program         complete source provenance copied by the checkpoint tree
    action          action attempted (defaults to label, then failure_modes)
    outcome         ok | partial | failed | crashed
    constants_plan  when the outcome rests on hand-picked numbers: 'searching' or why not
    """
    missing = [k for k, v in (("scene", scene), ("log", log),
                              ("failure_modes", failure_modes)) if not str(v or "").strip()]
    if missing:
        raise ValueError(f"say something for: {', '.join(missing)} — an empty review is how "
                         f"the same failure gets attempted twice")
    if keep and not str(label or "").strip():
        raise ValueError("keep=True needs a label: name the STATE reached, "
                         "e.g. 'part_0 secured in its mount'")
    outcome = str(outcome or "").lower()
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
    if keep and tree is None and env is None:
        raise ValueError(
            "keep=True requires tree=CheckpointTree(env) or env=...; "
            "a kept verdict without a saved state is not allowed")

    keep_requested = bool(keep)
    failed = outcome in ("failed", "crashed")
    keep_state = keep_requested and not failed
    resolved_action = str(action or label or failure_modes).strip()

    record = {
        "t": time.time(),
        "scene": str(scene).strip(),
        # Exact log, including leading/trailing whitespace and every character.
        "log": str(log),
        "failure_modes": str(failure_modes).strip(),
        "keep": keep_state,
        "keep_requested": keep_requested,
        "label": str(label or "").strip(),
        "action": resolved_action,
        "outcome": outcome,
        "program": str(program) if program is not None else "",
        "constants_plan": str(constants_plan or "").strip(),
    }
    provenance_metrics = {
        "assessment_scene": record["scene"],
        "assessment_failure_modes": record["failure_modes"],
        "constants_plan": record["constants_plan"],
        "keep_requested": keep_requested,
    }

    if keep_state:
        if tree is None:
            tree = _open_checkpoint_tree(env, checkpoint_root, "save the kept assessment")
        # Keep the default branch digest concise: failure prose is the note, while the dedicated
        # log field/file carries the exact full log and metrics carry the complete assessment.
        as_root = getattr(tree, "current", None) is None
        if as_root:
            print(
                "[assessment] no verified active origin is available; storing this kept "
                "state as an independent root with origin=unknown-live-root, not inventing "
                "a parent edge",
                flush=True,
            )
        record["checkpoint"] = tree.save(
            record["label"], note=record["failure_modes"], action=record["action"],
            program=program, log=record["log"], outcome=outcome,
            metrics=provenance_metrics, as_root=as_root,
        )
    elif tree is not None or env is not None:
        if tree is None:
            tree = _open_checkpoint_tree(env, checkpoint_root, "record the non-kept attempt")
        if keep_requested and failed:
            print(
                f"[assessment] outcome={outcome} cannot be a reusable kept checkpoint; "
                "recording it as a non-reusable attempt instead",
                flush=True,
            )
        record["attempt"] = tree.record_attempt(
            record["action"], outcome, note=record["failure_modes"],
            log=record["log"], program=program, metrics=provenance_metrics,
        )

    path = _path(root)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    n = sum(1 for _ in path.open())
    print(f"[assessment] recorded review #{n}"
          + (f" (checkpoint {record['checkpoint']})" if record.get("checkpoint") else "")
          + (f" (attempt {record['attempt']})" if record.get("attempt") else ""),
          flush=True)
    return record


def history(n: int = 10, root: str | Path | None = None) -> str:
    """The last `n` reviews, oldest first — read this at the START of a new script."""
    path = _path(root)
    if not path.is_file():
        return "(no assessments recorded yet)"
    lines = path.read_text().strip().splitlines()[-int(n):]
    out = []
    for i, line in enumerate(lines, 1):
        try:
            r = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"[assessment] invalid history JSON on selected line {i}: {exc!r}", flush=True)
            continue
        mark = " [KEPT" + (f" -> {r['checkpoint']}" if r.get("checkpoint") else "") + "]" \
            if r.get("keep") else ""
        if r.get("attempt"):
            mark += f" [ATTEMPT {r['attempt']} {r.get('outcome', '')}]"
        out.append(f"{i}. {r.get('label') or '(no label)'}{mark}\n"
                   f"   scene: {r.get('scene', '')}\n"
                   f"   log: {r.get('log', '')}\n"
                   f"   failed: {r.get('failure_modes', '')}")
    return "\n".join(out) or "(no assessments recorded yet)"

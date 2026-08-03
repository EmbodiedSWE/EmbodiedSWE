"""Assessment — write down what a run actually did before starting the next one.

    from assessment import assess, history

    assess(scene="part_0 in place; part_1 tipped over next to its goal",
           log="goal metric 3.1mm of 5mm; gripper lost contact at t=210",
           failure_modes="grasp slips during the transfer — approach angle too steep",
           keep=True, label="part_0 in place", tree=tree)   # tree: optional CheckpointTree

    print(history())        # in a LATER script: what did I already conclude?

The point is the discipline the old harness enforced between runs, kept as a callable record:
say what the images show, what the numbers say, and what failed — in words — before touching
the code again. Two things make this worth calling rather than just thinking:

  * the record PERSISTS (``/workspace/.assessments/reviews.jsonl``). Each of your scripts is a
    fresh process with no memory; `history()` is how a later run recalls what was already tried
    and concluded, instead of rediscovering it;
  * `keep=True` couples the verdict to the checkpoint tree: a state judged worth building on is
    saved AS PART OF the judgment (pass the tree), so "good state" and "saved state" cannot
    drift apart.

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


def _path(root: str | Path | None) -> Path:
    p = Path(root or DEFAULT_ROOT)
    p.mkdir(parents=True, exist_ok=True)
    return p / "reviews.jsonl"


def assess(
    scene: str,
    log: str,
    failure_modes: str,
    *,
    keep: bool = False,
    label: str = "",
    tree=None,
    constants_plan: str = "",
    root: str | Path | None = None,
) -> dict:
    """Record one run's review. Returns the record (with `checkpoint` when one was saved).

    scene           what the end-state image(s) show — where things actually ended up
    log             what the printed numbers say, including the measurements that matter
    failure_modes   what went wrong or blocks progress; say plainly if the run went as intended
    keep            True when this end state is worth building on
    label           the state reached (required with keep) — it becomes the checkpoint label
    tree            a checkpoint_tree.CheckpointTree; with keep=True the state is saved to it
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

    record = {
        "t": time.time(),
        "scene": str(scene).strip(),
        "log": str(log).strip(),
        "failure_modes": str(failure_modes).strip(),
        "keep": bool(keep),
        "label": str(label or "").strip(),
        "constants_plan": str(constants_plan or "").strip(),
    }
    if keep:
        if tree is not None:
            # the note carries the log IN FULL — a [:200] slice here (removed 2026-08-01) was
            # a silent truncation; anything long is still one field in one node
            record["checkpoint"] = tree.save(record["label"], note=record["log"])
        else:
            print("[assessment] keep=True but no tree passed — the verdict is recorded, the "
                  "state is NOT saved (pass tree=CheckpointTree(env) to save it)", flush=True)

    path = _path(root)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    n = sum(1 for _ in path.open())
    print(f"[assessment] recorded review #{n}"
          + (f" (checkpoint {record['checkpoint']})" if record.get("checkpoint") else ""),
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
        except json.JSONDecodeError:
            continue
        mark = " [KEPT" + (f" -> {r['checkpoint']}" if r.get("checkpoint") else "") + "]" \
            if r.get("keep") else ""
        out.append(f"{i}. {r.get('label') or '(no label)'}{mark}\n"
                   f"   scene: {r.get('scene', '')}\n"
                   f"   log: {r.get('log', '')}\n"
                   f"   failed: {r.get('failure_modes', '')}")
    return "\n".join(out) or "(no assessments recorded yet)"

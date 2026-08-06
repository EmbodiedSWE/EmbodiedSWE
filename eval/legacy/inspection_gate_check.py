#!/usr/bin/env python3
"""Offline check of the inspection loop: the gate, the assessment, and the goto handshake.

No pod and no LLM: CoSiGenToolState.execute is replaced by a stub that reports whether a
run moved the world, which is the only thing the gate keys off. Covers the behaviour that
matters:
  * a run that moved the world blocks the next execute until it is assessed,
  * a run that moved nothing does not,
  * an assessment missing its observations is refused, and one that skips the optimize
    question is refused only while the judge has flagged hand-picked constants,
  * keep=True checkpoints the program, keep=False does not,
  * goto shows the node first and refuses to move until the same node is confirmed,
  * internal turns (view frames, node inspection) never trip the gate.

Usage:  python3 scripts/tests/inspection_gate_check.py
"""
import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/tiger/cap-x/CoSiGen/eval")
import cosigen_agentic_loop as L  # noqa: E402

CHECKS: list[tuple[str, bool]] = []


def check(label: str, ok: bool) -> None:
    CHECKS.append((label, bool(ok)))


class FakeState(L.CoSiGenToolState):
    """Real gate logic, fake simulator."""

    def __init__(self, tmp: str, *, trial_mode=True):
        super().__init__("http://fake", tmp, trial_mode=trial_mode,
                         session_id="gate-check")
        self.calls: list[dict] = []
        self.moves = True          # does the next run move the world?
        self.files = {"/workspace/p.py": "move_to(1, [0, 0, 1])\n"}

    async def execute(self, code, stage_summary="", *, name="", checkpoint=False,
                      label="", optimize=None, internal=False, goto_allowed=False):
        self.executions += 1
        self.calls.append({"code": code, "name": name, "checkpoint": checkpoint,
                           "internal": internal, "goto_allowed": goto_allowed})
        moved = self.moves and not internal
        result = {
            "rc": 0, "success": False, "stdout": "leg_2 z=1.05 seated=[False]*4",
            "stderr": "", "scene_summary": "seated=[False]  steps_used=100/1e9",
            "n_frames": 40 if moved else 0,
            "turn_images": ([{"view": "end of run", "jpeg_b64": "QUJD"}] if moved else []),
            "tree": {"current": "n2", "new_node": ("n3" if checkpoint else None)},
        }
        self.last_result = result
        if not internal and int(result["n_frames"]) > 0:
            self.pending_review = {"run": self.executions, "name": name,
                                   "node": "n2", "checkpointed": bool(checkpoint)}
        return result

    async def read_workspace_file(self, path):
        return self.files[path]


def out(resp) -> dict:
    try:
        return json.loads(resp.output)
    except json.JSONDecodeError:
        return {}


async def main() -> None:
    tmp = tempfile.mkdtemp()
    st = FakeState(tmp)
    ex, asr, goto = L.CoSiGenExecuteTool(st), L.CoSiGenAssessTool(st), L.CoSiGenGotoTool(st)

    # a run that moves the world, then the gate
    await ex.execute({"code": "move_to(1, [0, 0, 1])"})
    check("moving run leaves an assessment pending", st.pending_review is not None)
    blocked = out(await ex.execute({"code": "print(1)"}))
    check("next execute is refused until assessed", blocked.get("error") is not None)

    # a half-filled assessment is refused
    bad = out(await asr.execute({"scene": "leg upright", "log": "", "failure_modes": "x",
                                 "keep": False}))
    check("assessment without the log is refused", bad.get("error") is not None)

    # the optimize question, only while the judge has flagged constants
    st.tunable_flagged = "GRASP_Z 0.12, HOVER 0.18"
    need_opt = out(await asr.execute({"scene": "leg upright", "log": "z=1.05",
                                      "failure_modes": "grip slips", "keep": False}))
    check("assessment must answer the optimize question when constants were flagged",
          need_opt.get("error") is not None)
    ok = out(await asr.execute({"scene": "leg upright", "log": "z=1.05",
                                "failure_modes": "grip slips", "keep": False,
                                "optimize": "searching"}))
    check("assessment accepted once it answers", ok.get("recorded") is True)
    check("gate reopens after the assessment", st.pending_review is None)
    check("flag cleared with it", st.tunable_flagged == "")
    ran = out(await ex.execute({"code": "print(1)"}))
    check("execute runs again after assessing", "error" not in ran)

    # keep=True checkpoints the program; keep=False does not
    before = sum(1 for c in st.calls if c["checkpoint"])
    kept = out(await asr.execute({"scene": "s", "log": "l", "failure_modes": "f",
                                  "keep": True, "label": "leg upright",
                                  "path": "/workspace/p.py"}))
    after = sum(1 for c in st.calls if c["checkpoint"])
    check("keep=True checkpoints the program", after == before + 1
          and "saved as n3" in str(kept.get("checkpoint")))
    check("keeping a state does not demand a second assessment of the same program",
          st.pending_review is None)
    check("execute is free right after keeping",
          "error" not in out(await ex.execute({"code": "print(3)"})))
    await ex.execute({"code": "move_to(1, [0, 0, 1])"})
    before = sum(1 for c in st.calls if c["checkpoint"])
    await asr.execute({"scene": "s", "log": "l", "failure_modes": "f", "keep": False})
    check("keep=False checkpoints nothing",
          sum(1 for c in st.calls if c["checkpoint"]) == before)

    # every tool that moves the world respects the gate, not just execute
    ckpt = L.CoSiGenCheckpointTool(st)
    await ex.execute({"code": "move_to(1, [0, 0, 1])"})
    check("checkpoint is refused while an assessment is pending",
          out(await ckpt.execute({"label": "x", "path": "/workspace/p.py"}))
          .get("error") is not None)
    check("goto is refused while an assessment is pending",
          out(await goto.execute({"node": "n1"})).get("error") is not None)
    await asr.execute({"scene": "s", "log": "l", "failure_modes": "f", "keep": False})
    check("checkpoint works once the run is assessed",
          "error" not in out(await ckpt.execute({"label": "x", "path": "/workspace/p.py"})))
    await asr.execute({"scene": "s", "log": "l", "failure_modes": "f", "keep": False})

    # a run that moves nothing does not gate
    st.moves = False
    await ex.execute({"code": "print(get_object_pose('leg_0'))"})
    check("read-only run leaves no assessment pending", st.pending_review is None)
    check("execute after a read-only run is allowed",
          "error" not in out(await ex.execute({"code": "print(2)"})))

    # goto: show first, then confirm, and only for the node that was shown
    early = out(await goto.execute({"node": "n1", "confirm": True}))
    check("goto refuses to move before showing the node", early.get("error") is not None)
    shown = out(await goto.execute({"node": "n1"}))
    check("goto shows the node's log and scene", "state_and_log" in shown)
    check("showing a node is an internal turn (no gate)", st.pending_review is None)
    wrong = out(await goto.execute({"node": "n2", "confirm": True}))
    check("confirming a different node is refused", wrong.get("error") is not None)
    moved = out(await goto.execute({"node": "n1", "confirm": True}))
    check("confirming the shown node moves the world", moved.get("node") == "n1")
    check("the move is flagged as coming from the tool",
          any(c["goto_allowed"] for c in st.calls))

    # the assessments are recorded next to the run
    rows = [json.loads(l) for l in
            (Path(tmp) / "assessments.jsonl").read_text().splitlines() if l.strip()]
    check("assessments are written to the artifact dir", len(rows) >= 3)
    check("each row carries the agent's own words",
          all(r.get("failure_modes") for r in rows))

    for label, ok_ in CHECKS:
        print(f"   {'PASS' if ok_ else 'FAIL'}  {label}")
    print("INSPECTION GATE " + ("PASSED" if all(o for _, o in CHECKS) else "FAILED"))
    sys.exit(0 if all(o for _, o in CHECKS) else 1)


asyncio.run(main())

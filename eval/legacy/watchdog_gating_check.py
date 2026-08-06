#!/usr/bin/env python3
"""Regression check: the watchdog must never mention a facility its arm does not have.

The ckpt arm has no optimize tool, but 30 of its 53 reminders offered optimize() because
the judge prompt named the facility outside the gated block. That is silent: the arm keeps
running, the agent is pointed at a tool it cannot call, and the ablation between arms
stops being an ablation. Two layers are checked here — the assembled prompt, and the
drop-guard that catches a judge which names a missing facility anyway.

Usage:  python3 scripts/tests/watchdog_gating_check.py
"""
import asyncio
import json
import pathlib
import re
import sys
import tempfile
import time

SRC = pathlib.Path("/home/tiger/cap-x/CoSiGen/eval/cosigen_watchdog.py").read_text()
ns = {"json": json, "re": re, "time": time, "Path": pathlib.Path, "annotations": None}
exec(compile(SRC, "cosigen_watchdog", "exec"), ns)
HarnessWatchdog = ns["HarnessWatchdog"]


class _Caller:
    """Stub swalm caller returning a fixed judge verdict."""

    def __init__(self, message: str):
        self.message = message

    async def _call_llm(self, _messages):
        body = json.dumps({"intervene": True, "reason": "stub", "message": self.message})
        return type("R", (), {"content": body})()


def advise(has_opt: bool, has_ckpt: bool, message: str):
    with tempfile.TemporaryDirectory() as d:
        w = HarnessWatchdog(_Caller(message), d, has_opt=has_opt, has_ckpt=has_ckpt)
        w.observe({"run": 7, "name": "p", "code": "Z = 0.1\n", "rc": 0,
                   "runs_since_checkpoint": 5, "runs_since_optimize": 5, "node": "n2"})
        out = asyncio.run(w.maybe_advise())
        rows = [json.loads(l) for l in
                (pathlib.Path(d) / "watchdog.jsonl").read_text().splitlines() if l.strip()]
        return out, rows[-1]


OPT_MSG = "you could call optimize(program=..., space=..., objective=...) for these"
CKPT_MSG = "you could checkpoint(label=..., path=...) the state this run reached"
checks = []

# 1. the assembled prompt carries only the arm's own facilities
ck = HarnessWatchdog(None, tempfile.mkdtemp(), has_opt=False, has_ckpt=True)
full = HarnessWatchdog(None, tempfile.mkdtemp(), has_opt=True, has_ckpt=True)
opt_only = HarnessWatchdog(None, tempfile.mkdtemp(), has_opt=True, has_ckpt=False)
checks += [
    ("ckpt-arm prompt never says optimize", "optimize" not in ck.prompt.lower()),
    ("ckpt-arm prompt does explain checkpoint", "checkpoint(" in ck.prompt),
    ("opt-only prompt never says checkpoint", "checkpoint" not in opt_only.prompt.lower()),
    ("both-arm prompt explains both",
     "optimize(" in full.prompt and "checkpoint(" in full.prompt),
    ("both-arm prompt keeps the one-message clause",
     "both facilities fit" in full.prompt),
    ("single-facility prompt drops the one-message clause",
     "both facilities fit" not in ck.prompt),
]

# 2. the guard drops a judge verdict that names a facility the arm lacks
out, row = advise(False, True, OPT_MSG)
checks += [("optimize nudge dropped on a no-optimize arm", out is None),
           ("drop is logged with a reason", "dropped" in row)]
out, row = advise(True, False, CKPT_MSG)
checks += [("checkpoint nudge dropped on a no-checkpoint arm", out is None)]

# 3. legitimate nudges still pass through untouched
out, _ = advise(True, True, OPT_MSG)
checks += [("optimize nudge delivered when the arm has it", out == OPT_MSG)]
out, _ = advise(False, True, CKPT_MSG)
checks += [("checkpoint nudge delivered when the arm has it", out == CKPT_MSG)]

for label, ok in checks:
    print(f"   {'PASS' if ok else 'FAIL'}  {label}")
print("WATCHDOG GATING " + ("PASSED" if all(ok for _, ok in checks) else "FAILED"))
sys.exit(0 if all(ok for _, ok in checks) else 1)

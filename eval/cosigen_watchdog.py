"""Driver-side harness watchdog (user-requested, 2026-07-26).

Reads the agent's recent executed programs after every turn and asks an LLM judge
whether one of the harness facilities fits what the agent
is doing right now: the `optimize` tool (its program's outcome rests on hand-picked
numbers) or `checkpoint` (its program advanced the world and the state is unsaved).
When one fits, a short reminder rides that turn's tool feedback — phrased as an
option the agent may take, never an instruction, and never a verdict on whether the
agent's progress is good enough: that judgment stays with the agent.

Checkpointing was added to the remit after v19: the ckpt arm checkpointed 4 times in
168 runs, all of them in the milestone-rich opening, and by run 66 it wanted to "step
back to the one real success" — a state it had never saved, so it tried to re-derive
it instead of returning to it.

No mechanical rate limiting at all (user decisions 2026-07-26; the old 8-turn cooldown
let the v18 agent grind a hand-tuning loop unchallenged, the old 6-intervention session
cap muted the v19 watchdog for the rest of its session, and the post-checkpoint/optimize
quiet period is likewise gone): the judge is consulted after every clean turn and
decides itself whether to speak. It sees its own last reminder so it does not repeat
the same advice every turn, and the session counters (runs since last checkpoint /
optimize, 0 = just happened) tell it when the agent is already behaving. Every
consultation — fired or not — is appended to watchdog.jsonl in the artifact dir,
so the paper can audit when and why it spoke.

Background for the design: v14/v16/v17 agents each entered long hand-tuning loops
without ever calling optimize, yet when asked out-of-band WHY (2026-07-25 probe),
the v17 agent correctly explained when the tool applies and named its own tunable
constants. The knowledge is present; the in-flow pause to apply it is not. The
watchdog supplies that pause.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

# Everything facility-specific — the description, the example reminder and the "fits
# when" rule — is assembled per arm from the blocks below. Nothing about a facility may
# sit in the static text: the ckpt arm has no optimize tool, yet 30 of its 53 reminders
# offered optimize() because the example call and the fits-optimize rule lived here
# unconditionally, telling the judge about a tool the gating had removed (2026-07-27).
WATCHDOG_PROMPT = """You are a watchdog for a robot-control coding agent's harness. The agent iterates on
control programs against a simulator. These harness facilities are available to it and
easy to forget in the middle of debugging:
<FACILITIES>
Below are the agent's most recent executed programs, plus a few session counters. If
<ONE_OF> fits what the agent is doing right
now, write one short reminder addressed to the agent, offering it as an option — for
example <EXAMPLES>. Keep it
as a soft reminder, and let the agent decide.

<FITS>
Stay quiet when <NO_FIT>: the program is only reading state or exploring, it
crashed, or the agent is already using the facility. Name the constants or the state
you actually see in its program.<BOTH_FIT>

Your previous reminder: <LAST_MESSAGE>

Reply with only a JSON object:
{"intervene": true or false, "reason": "one line", "message": "the reminder, or empty",
 "tunable": "the constants in the latest program whose values were picked by hand and
 decide its outcome, named plainly; empty when there are none"}
Fill in "tunable" from what the program shows regardless of whether you intervene, and
leave the ranges to the agent: it knows the scene.

Session counters: <COUNTERS>

<TURNS>"""

EXAMPLE_OPTIMIZE = ('"when you want to optimize <these parameters> for <this goal>, you '
                    'can call\noptimize(program=..., space=..., objective=...)"')

EXAMPLE_CHECKPOINT = ('"if you want to save the state this\nprogram reached, you can '
                      'checkpoint it with checkpoint(label=..., path=...)"')

FITS_OPTIMIZE = """A situation fits optimize when the program's outcome depends on numeric values the
agent picked by hand (grasp offsets, heights, angles, stroke sizes, step counts,
rates). It does not have to be repeating itself, and trying a new strategy does not
exempt that strategy's hand-picked numbers.
"""

FITS_CHECKPOINT = """A situation fits checkpoint when the program moved the world somewhere that is not
saved yet. No need to judge whether the progress is good enough to keep. Your job here
is just to offer a soft reminder that the agent could checkpoint useful states.
"""

BOTH_FIT = (" And if both facilities fit, mention them in one\nbrief message rather "
            "than two.")


FACILITY_OPTIMIZE = """- optimize: runs many copies of one of the agent's programs in parallel, each copy
  with different values substituted for chosen top-level numeric constants, scores
  every end state with a small objective file, and searches for the best values. It is
  the way to set any parameter the agent would otherwise pick by trial and error
  (offsets, depths, angles, timings, thresholds, stroke counts, rates). It runs in the
  background, so the agent keeps working while it searches. The copies act through the
  toolkit only, so if the program also configures the simulator through raw env/api
  handles (controller gains, for instance), that part belongs in the separate setup
  file the tool takes — worth saying when the program in front of you does both."""

FACILITY_CHECKPOINT = """- checkpoint(label=..., path=...): saves the state a program reaches as a node in the
  agent's checkpoint tree, so later work starts from there and goto() can return to it.
  Unsaved states are lost when the next program runs from the current node."""


class HarnessWatchdog:
    """Consulted once per executed turn; the judge itself decides when speaking again
    is worthwhile (it sees its own last reminder — no mechanical cooldown). `facilities`
    limits the remit to what this pod actually offers, so a reminder never points at a
    tool the agent does not have. `caller` is a swalm LLM caller (agent's model)."""

    def __init__(self, caller, artifact_dir, window: int = 3, *,
                 has_opt: bool = True, has_ckpt: bool = True):
        self.caller = caller
        self.log_path = Path(artifact_dir) / "watchdog.jsonl"
        self.window = int(window)
        self.turns: list[dict] = []
        self.interventions = 0  # counted for the log only; no session cap (2026-07-26)
        self.last_message = ""  # the judge's previous reminder, shown back to it
        self.last_tunable = ""  # constants the judge saw in the latest program
        self.has_opt, self.has_ckpt = bool(has_opt), bool(has_ckpt)
        parts = ([FACILITY_OPTIMIZE] if has_opt else []) + \
                ([FACILITY_CHECKPOINT] if has_ckpt else [])
        self.facilities = "\n".join(parts)
        self.enabled = bool(parts)
        both = has_opt and has_ckpt
        self.prompt = (
            WATCHDOG_PROMPT
            .replace("<FACILITIES>", self.facilities)
            .replace("<ONE_OF>", "one of those facilities" if both else "it")
            .replace("<EXAMPLES>", ", or ".join(
                ([EXAMPLE_OPTIMIZE] if has_opt else [])
                + ([EXAMPLE_CHECKPOINT] if has_ckpt else [])))
            .replace("<FITS>", "\n".join(
                ([FITS_OPTIMIZE] if has_opt else [])
                + ([FITS_CHECKPOINT] if has_ckpt else [])))
            .replace("<NO_FIT>", "neither fits" if both else "it does not fit")
            .replace("<BOTH_FIT>", BOTH_FIT if both else ""))
        # Names the arm does not have. A reminder pointing at one is worse than silence:
        # it sends the agent after a tool that is not in its tool list, and it quietly
        # breaks the ablation by telling a no-optimize arm that optimize exists.
        self.forbidden = ([] if has_opt else ["optimize"]) + \
                         ([] if has_ckpt else ["checkpoint", "goto"])

    def observe(self, record: dict) -> None:
        self.turns.append(record)
        self.turns = self.turns[-self.window:]

    async def maybe_advise(self) -> str | None:
        decision: dict = {"consulted": False, "intervene": False}
        try:
            if not self.turns or self.turns[-1].get("rc") not in (0, None):
                decision["skipped"] = "last turn errored (agent is debugging)"
                return None
            decision["consulted"] = True
            verdict = await self._judge()
            decision.update(verdict)
            # What the driver's assessment gate reads: named constants, no ranges.
            self.last_tunable = str(verdict.get("tunable") or "") if self.has_opt else ""
            leaked = [n for n in self.forbidden
                      if n in str(verdict.get("message", "")).lower()]
            if leaked:
                decision["intervene"] = False
                decision["dropped"] = f"named a facility this arm lacks: {leaked}"
                return None
            if verdict.get("intervene") and verdict.get("message"):
                self.interventions += 1
                self.last_message = str(verdict["message"])
                return self.last_message
            return None
        finally:
            decision["run"] = (self.turns[-1] or {}).get("run") if self.turns else None
            decision["t"] = time.time()
            try:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(decision, ensure_ascii=False, default=str) + "\n")
            except OSError:
                pass

    async def _judge(self) -> dict:
        # Program text in full, never truncated: the constants the judge reasons about
        # are in the code. The programs' printed output is deliberately NOT included --
        # neither verdict rests on it (hand-picked numbers are visible in the source;
        # "moved the world and did not save it" comes from rc + the checkpoint counters),
        # while its size is set by whatever the agent chose to print. A single stage walk
        # over 512 envs printed 25k lines / 1.9M chars, which put the judge call over the
        # model's context limit and cost the arm every reminder for that turn.
        blocks = []
        for r in self.turns:
            blocks.append(
                f"--- turn {r.get('run')} (program {r.get('name') or 'inline'}, "
                f"rc={r.get('rc')})\n{r.get('code') or ''}")
        counters = {
            "runs_since_last_checkpoint": self.turns[-1].get("runs_since_checkpoint"),
            "runs_since_last_optimize": self.turns[-1].get("runs_since_optimize"),
            "reminders_already_sent": self.interventions,
            "current_node": self.turns[-1].get("node"),
        }
        prompt = (self.prompt
                  .replace("<LAST_MESSAGE>", self.last_message or "(none yet)")
                  .replace("<COUNTERS>", json.dumps(counters))
                  .replace("<TURNS>", "\n\n".join(blocks)))
        resp = await self.caller._call_llm([{"role": "user", "content": prompt}])
        m = re.search(r"\{.*\}", resp.content or "", re.DOTALL)
        if not m:
            return {"intervene": False, "reason": "judge returned no JSON"}
        try:
            out = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"intervene": False, "reason": "judge JSON unparseable"}
        return {"intervene": bool(out.get("intervene")),
                "reason": str(out.get("reason", "")),
                "message": str(out.get("message", "")),
                "tunable": str(out.get("tunable", ""))}

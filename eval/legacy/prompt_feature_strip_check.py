#!/usr/bin/env python3
"""Check that the per-arm prompt stripping still matches the prompt text.

make_prompt tailors the directive for arms without checkpointing or without optimize by
replacing exact sentences. Those literals live in a different file from the sentences
themselves, so editing the prompt silently breaks the tailoring: the no-optimize arm would
keep being told to use optimize. This parses the replace() calls and checks every search
string is really present.

Usage:  python3 scripts/tests/prompt_feature_strip_check.py
"""
import ast
import re
import sys
from pathlib import Path

EVAL = Path("/home/tiger/cap-x/CoSiGen/eval")
prompts_src = (EVAL / "cosigen_prompts.py").read_text()
session_src = (EVAL / "cosigen_session.py").read_text()

# the text being tailored: the directive constants, as the pod composes them
consts = {}
for node in ast.parse(prompts_src).body:
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
            and isinstance(node.value.value, str):
        consts[node.targets[0].id] = node.value.value
directive = "".join(consts.get(k, "") for k in
                    ("DIRECTIVE_SESSION_PROTOCOL", "DIRECTIVE_PLANNING",
                     "DIRECTIVE_CLOSING"))
# An api class may ship its own directive (legacy null embodiment), and make_prompt
# tailors whichever one it gets, so a literal counts as live if any of them contains it.
for extra in EVAL.glob("legacy/*.py"):
    for node in ast.parse(extra.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str) \
                and "DIRECTIVE" in getattr(node.targets[0], "id", ""):
            directive += "\n" + node.value.value

bad = []
for node in ast.walk(ast.parse(session_src)):
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "replace" and node.args):
        continue
    target = getattr(node.func.value, "id", "")
    needle = node.args[0]
    if not isinstance(needle, ast.Constant) or not isinstance(needle.value, str):
        continue
    text = directive if target == "directive" else None
    if text is None:
        continue
    if needle.value not in text:
        bad.append(needle.value)

print(f"directive replacements checked; {len(bad)} no longer match")
for b in bad:
    print("   MISSING from the directive:", re.sub(r"\s+", " ", b)[:110])
print("PROMPT FEATURE STRIP " + ("PASSED" if not bad else "FAILED"))
sys.exit(0 if not bad else 1)

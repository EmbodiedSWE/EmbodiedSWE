#!/usr/bin/env python3
"""Wait for the orphaned turn on the bimanual pod to finish draining, then relaunch the
v7 opus driver with --resume (restores the durable tree; fresh conversation)."""
from __future__ import annotations

import json
import subprocess
import time
import urllib.request

URL = "http://[2605:340:cd51:7700:48b7:5e1c:63f0:eeb]:10600"

EXTRA = (
    "You have RAW simulator access (env, api, torch, isaaclab imports live in your "
    "namespace — see RAW SIMULATOR ACCESS in the API doc); write Isaac Lab-flavored "
    "code when it helps, alongside the toolkit (move_to, look, checkpoint/goto, "
    "rl_train). STUDY THE SOLVED EXAMPLES FIRST: /home/tiger/CoSiGen/eval/examples/ on "
    "this simulator host contains complete verified solvers for related tasks — read "
    "them from your executed code before planning. IMPORTANT UPDATE (2026-07-24): "
    "rl_train was BROKEN for this bimanual embodiment during your earlier turns "
    "(hand-action crash + a dead-replica bug + wrong per-arm wrist reads in rich_obs) — "
    "your conclusion that 'RL is unavailable here' NO LONGER HOLDS. All three defects "
    "are fixed and validated on this exact pod: a reach policy trains cleanly (reward "
    "rises, 512-replica batch healthy). Use rl_train for the contact-rich seat/thread "
    "steps (cosigen-rl-subpolicy skill: rich_obs, short bursts, init_from continuation, "
    "follow the recommendation field); start episodes AT the difficult moment (leg IN "
    "GRIP at contact). NOTE the wrist joint limit allows only ~0.45 revolutions per "
    "twist — full seating needs ratchet cycles (twist-release-unwind-regrip); consider "
    "training the per-cycle stroke and looping it with rl_run. checkpoint() before "
    "risky maneuvers; goto() instead of repeating failures. Do NOT teleport task "
    "objects into goal states — success is graded on physical manipulation."
)


def n_runs() -> int:
    try:
        with urllib.request.urlopen(URL + "/ping", timeout=10) as r:
            return int(json.loads(r.read()).get("n_runs", -1))
    except Exception as exc:  # noqa: BLE001
        print(f"ping failed: {exc!r}", flush=True)
        return -1


def main() -> None:
    base = n_runs()
    print(f"waiting for orphan turn to drain (n_runs now {base})", flush=True)
    for _ in range(240):
        n = n_runs()
        print(f"n_runs={n}", flush=True)
        if n > base >= 0 or (base < 0 and n >= 0):
            break
        base = max(base, n)
        time.sleep(60)
    print("orphan drained; relaunching v7 driver with --resume", flush=True)
    cmd = ["python3.11", "scripts/cosigen_harness.py",
           "--env", "assembly.ikea_table.bimanual_franka.osc",
           "--model", "claude-opus-48-relay",
           "--session-id", "ikea-bifranka-v7",
           "--max-steps", "100000", "--resume",
           "--transcript",
           "CoSiGen/cosigen_eval_artifacts/ikea_bifranka_v7/transcript.json",
           "--extra-prompt", EXTRA]
    log = open("CoSiGen/cosigen_eval_artifacts/ikea_bifranka_v7/run.log", "a")
    subprocess.Popen(cmd, cwd="/home/tiger/cap-x", stdout=log, stderr=log,
                     start_new_session=True)
    print("v7 driver relaunched", flush=True)


if __name__ == "__main__":
    main()

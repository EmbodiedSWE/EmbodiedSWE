#!/usr/bin/env python3
"""Driver-side half of the async optimize path, which the pod-level e2e cannot cover.

opt_async_e2e.py drives the SEARCH POD directly. This one drives CoSiGenToolState the
way the agent's tool call does, so it exercises the pieces that live in the driver:
  1. start_optimize returns immediately with a handle (the call must not block),
  2. the dedicated search pod is first sent to the agent's node (goto prep),
  3. _poll_optimize turns the pod's /ping progress into agent-visible notes,
  4. optimize_status.json in the artifact dir tracks the search and its result,
  5. a second start_optimize while one runs is refused with a readable reason.

Usage:  opt_driver_async_check.py <turn_pod_url> <search_pod_url>
"""
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/tiger/cap-x/CoSiGen/eval")
from cosigen_agentic_loop import CoSiGenToolState  # noqa: E402

TURN, SEARCH = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/")
ART = Path("/home/tiger/cap-x/eval_result/v21_monitor/driver_async_check")
ART.mkdir(parents=True, exist_ok=True)

PROGRAM = """
HOVER_DZ = 0.18
p, _ = get_object_pose('leg_0')
open_gripper(1, 10)
move_to(1, [p[0], p[1] - 0.12, p[2] + HOVER_DZ], None, max_steps=110, pos_tol=0.01)
step(10)
"""
OBJECTIVE = """
import numpy as np
def objective(v):
    p, _ = v.object_pose('leg_0')
    e, _ = v.eef_pose(1)
    goal = np.asarray([p[0], p[1] - 0.12, p[2] + 0.12])
    return float(np.linalg.norm(np.asarray(e) - goal))
"""


async def main() -> None:
    state = CoSiGenToolState(TURN, str(ART), trial_mode=True,
                             session_id="driver-async-check", opt_server=SEARCH)
    request = {"program": PROGRAM, "objective": OBJECTIVE,
               "space": {"HOVER_DZ": [0.05, 0.30]},
               "options": {"generations": 2, "budget_s": 900,
                           "max_steps_per_eval": 250},
               "progress_meta": {"program": "/workspace/probe.py",
                                 "program_version": "probe__opt0001"}}
    t0 = time.time()
    started = await state.start_optimize(request, program_path="/workspace/probe.py",
                                         program_version="probe__opt0001", node=None)
    launch_s = time.time() - t0
    print(f"[driver] start_optimize returned in {launch_s:.2f}s: {started}", flush=True)

    second = await state.start_optimize(dict(request), program_path="/workspace/probe.py",
                                        program_version="probe__opt0002", node=None)
    print(f"[driver] second call while running: {second}", flush=True)

    notes: list[str] = []
    while not state.opt_job["task"].done():
        await asyncio.sleep(20)
        new = state.drain_opt_notes()
        for n in new:
            print(f"[driver] note -> {n}", flush=True)
        notes += new
    notes += state.drain_opt_notes()
    await state.opt_job["task"]

    status = {}
    sf = ART / "optimize_status.json"
    if sf.exists():
        status = json.loads(sf.read_text())
    print(f"\n[driver] final status file: {json.dumps(status)[:400]}", flush=True)

    checks = {
        "start_optimize returned immediately": launch_s < 20,
        "search ran on the dedicated pod": bool(started.get("dedicated")),
        "concurrent search refused with a reason": (
            not second.get("started") and bool(second.get("reason"))),
        "progress notes reached the agent": any("progress" in n for n in notes),
        "result note carries the best values": any("best" in n.lower() for n in notes),
        "status file holds the finished search": status.get("state") == "done",
    }
    print("\n[driver] checks:", flush=True)
    for name, ok in checks.items():
        print(f"   {'PASS' if ok else 'FAIL'}  {name}", flush=True)
    print("DRIVER ASYNC " + ("PASSED" if all(checks.values()) else "FAILED"), flush=True)


asyncio.run(main())

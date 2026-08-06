#!/usr/bin/env python3
"""RL-fix validation against a GIVEN pod URL (old bimanual pod, post-queue-drain).

Same MDP as scripts/tests/rl_fix_validation.py (arm-1 reach to 15 cm above leg_1,
rich_obs, residual wrist deltas, hand block ENABLED); assumes the pod is idle. First
re-POSTs /reload (fast when the queue is empty) so the fixed cosigen_loop/cosigen_rl
are definitely live, then runs the validation turn as session "rlfix-validation".
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ART = Path("/home/tiger/cap-x/CoSiGen/cosigen_eval_artifacts/rlfix_validation")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl_fix_validation import TURN_CODE  # noqa: E402  (same MDP, single source)


def _post(url: str, path: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> None:
    url = sys.argv[1].rstrip("/")
    print(f"[oldpod] target {url}")
    r = _post(url, "/reload", {}, 150)
    print(f"[oldpod] reload -> {r}")
    if not r.get("ok"):
        raise SystemExit("reload did not apply; aborting (fixes must be live first)")

    ART.mkdir(parents=True, exist_ok=True)
    payload = {"code": TURN_CODE, "reset": True, "max_steps": 100000,
               "num_frames": 300,
               "meta": {"session_id": "rlfix-validation",
                        "arm_recorder": {"every": 2, "max_frames": 1200}}}
    print("[oldpod] submitting validation turn ...")
    t0 = time.time()
    res = _post(url, "/run_policy", payload, 3600)
    print(f"[oldpod] done in {time.time() - t0:.0f}s rc={res.get('rc')}")
    (ART / "validation_result_oldpod.json").write_text(json.dumps(
        {k: v for k, v in res.items() if k != "turn_images"}, indent=1, default=str))
    print("===== FULL STDOUT =====")
    print(res.get("stdout", ""))
    print("===== STDERR =====")
    print(res.get("stderr", ""))


if __name__ == "__main__":
    main()

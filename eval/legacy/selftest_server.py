"""Self-test client for the CoSiGen render server (runs ON the same pod via localhost).

Polls GET /ping until booted, prints the served prompt, then POSTs the scripted
policy to /run_policy and prints rc/success/scene -- validating the full server
harness (boot + prompt + exec control API + grade) before any RL wiring.
"""
import argparse
import json
import os
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int,
                default=int(os.environ.get("ARNOLD_WORKER_0_PORT", "10355").split(",")[0]))
ap.add_argument("--policy", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "selftest_policy.py"))
ap.add_argument("--boot-wait", type=int, default=180)
args = ap.parse_args()

base = f"http://localhost:{args.port}"
meta: dict = {}
for _ in range(args.boot_wait):
    try:
        meta = json.load(urllib.request.urlopen(base + "/ping", timeout=20))
        if meta.get("booted") or meta.get("boot_error"):
            break
    except Exception as exc:  # noqa: BLE001
        print("ping retry:", repr(exc), flush=True)
    time.sleep(10)

print("BOOTED:", meta.get("booted"), "BOOT_ERROR:", meta.get("boot_error"), flush=True)
print("PROMPT[:700]:\n", (meta.get("prompt") or "")[:700], flush=True)

if meta.get("booted"):
    # Trivial, fast policy (a few steps) so the boot self-test isolates the render path quickly
    # instead of running the long scripted policy. num_frames>0 exercises the recording camera.
    code = "for _ in range(4):\n    step(20)\n"
    req = urllib.request.Request(
        base + "/run_policy",
        data=json.dumps({"code": code, "max_steps": 3000, "num_frames": 20}).encode(),
        headers={"Content-Type": "application/json"},
    )
    r = json.load(urllib.request.urlopen(req, timeout=200))
    print("SELFTEST rc=", r.get("rc"), "success=", r.get("success"), flush=True)
    print("STDOUT tail:\n", (r.get("stdout") or "")[-1200:], flush=True)
    print("SCENE:\n", r.get("scene_summary"), flush=True)
    print("SELFTEST_VIDEO n_frames=", r.get("n_frames"), "frames_npz=", r.get("frames_npz"), flush=True)
    print("COSIGEN_SELFTEST_DONE", flush=True)
else:
    print("COSIGEN_SELFTEST_BOOT_FAILED", flush=True)

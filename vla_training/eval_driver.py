"""H20-pod driver for one checkpoint's BC eval: serve the policy, run the
forge-side eval batch against it, propagate the verdict.

Runs inside the eval job pod (see mlx_bc_eval.yaml). Steps:
  1. start policy_http_server.py on this pod (venv python, 1 GPU);
  2. wait for /health;
  3. stage the forge package (eval_batch.py + scene.py + gen_strategy.py,
     shipped on the FUSE mount under simgen_bc/eval_pkg) and submit it to the
     assigned forge via forge_client.py (task namespace plated_meal_eval);
  4. run --module eval_batch with this pod's address; the module prints
     "EVAL_BATCH: DONE k/N" and pushes its results JSON to HDFS itself;
  5. exit 0 iff the run completed (any success count), nonzero on transport
     or module failure so the job surfaces red.

  python3 eval_driver.py --step 19000 --forge-url http://[...]:PORT \
      --episodes 20 --venv /home/tiger/cap-x/relevant_repos/openpi/.venv
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

FUSE = Path("/mnt/hdfs/__MERLIN_USER_DIR__/simgen_bc")


def pod_ipv6() -> str:
    out = subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()
    v6 = [a for a in out if ":" in a and not a.startswith("fe80")]
    if not v6:
        raise SystemExit(f"no global ipv6 on pod: {out}")
    return v6[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--forge-url", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=50000)
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--venv", default="/home/tiger/cap-x/relevant_repos/openpi/.venv")
    ap.add_argument("--ckpt", default="/opt/tiger/bc_ckpt")
    ap.add_argument("--config", default="pi05_simgen_bc_v1")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent

    # 1-2: policy server up on this pod
    srv = subprocess.Popen(
        [f"{args.venv}/bin/python", str(here / "policy_http_server.py"),
         "--config", args.config, "--ckpt", args.ckpt, "--port", str(args.port)],
        stdout=sys.stdout, stderr=sys.stderr)
    base = f"http://[{pod_ipv6()}]:{args.port}"
    deadline = time.time() + 900   # model load + jit can take minutes
    while True:
        if srv.poll() is not None:
            raise SystemExit(f"policy server died rc={srv.returncode}")
        try:
            with urllib.request.urlopen(f"http://[::1]:{args.port}/health", timeout=5):
                break
        except Exception:
            if time.time() > deadline:
                raise SystemExit("policy server never became healthy")
            time.sleep(5)
    print(f"[driver] policy server healthy at {base}", flush=True)

    # 3: forge package
    pkg = Path("/tmp/pm_eval_submit")
    if pkg.exists():
        shutil.rmtree(pkg)
    shutil.copytree(FUSE / "eval_pkg", pkg)
    client = pkg / "forge_client.py"
    sub = subprocess.run(
        ["python3", str(client), "--forge-url", args.forge_url, "submit",
         "--task", "plated_meal_eval", "--dir", str(pkg)],
        capture_output=True, text=True, timeout=300)
    print(f"[driver] submit rc={sub.returncode} {(sub.stderr or sub.stdout)[-200:]}", flush=True)
    if sub.returncode != 0:
        raise SystemExit("forge submit failed")

    # 4: run the eval module (forge executes one module at a time)
    mod_args = (f"--headless --server {base} --episodes {args.episodes} "
                f"--seed0 {args.seed0} --tag step{args.step}")
    run = subprocess.run(
        ["python3", str(client), "--forge-url", args.forge_url, "run",
         "--task", "plated_meal_eval", "--module", "eval_batch",
         f"--args={mod_args}", "--timeout", "10000"],
        capture_output=True, text=True, timeout=10300)
    out = run.stdout + run.stderr
    print(out[-3000:], flush=True)
    m = re.search(r"EVAL_BATCH: DONE (\d+)/(\d+)", out)
    srv.terminate()
    if not m:
        raise SystemExit(f"eval module did not complete (rc={run.returncode})")
    k, n = int(m.group(1)), int(m.group(2))
    print(json.dumps({"step": args.step, "successes": k, "episodes": n,
                      "success_rate": round(k / max(1, n), 4)}), flush=True)
    print(f"EVAL_DRIVER: DONE step={args.step} {k}/{n}", flush=True)


if __name__ == "__main__":
    main()

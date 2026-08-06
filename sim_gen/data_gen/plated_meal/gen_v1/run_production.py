"""Production driver: the full plated_meal generation campaign across the forge pool.

One thread per forge; each submits the package once and works through its run queue
sequentially (a forge executes one module at a time). Every run is a gen_batch invocation;
episodes land on HDFS from the pod. A run that errors or yields zero is retried once.

  python run_production.py            # wave 1: the 10-forge plan, log to production.log
  python run_production.py --wave 3   # reconcile from HDFS, run whatever is missing

Wave 3 is the ground-truth replan: it lists the prod_* batches that actually landed on
HDFS (batch meta.json carries seed0), maps them back to plan entries, and re-runs only
the entries with no delivered batch, spread one-run-per-forge over all live forges.
Runs are keyed by seed0, unique per run across the whole campaign (retry shift +37 < 100
spacing). Use it after any driver mishap; it is safe at quiescence (idle forges).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
COSIGEN = HERE.parents[3]
CLIENT = COSIGEN / "sim_gen" / "pipeline" / "forge_client.py"
SUBMIT_DIR = "/tmp/pm_gen_submit"
REG = "hdfs://haruna/tmp/zeyu.shen/simgen_forge"
HDFS_BASE = "hdfs://haruna/tmp/zeyu.shen/simgen_bc/plated_meal_gen_v1"

# (mode, episodes, seed0, extra_args) — seeds disjoint across every run and every smoke
PLAN: dict[int, list[tuple[str, int, int, str]]] = {}


def add(forges, mode, total, seed_base, per_run=15, extra=""):
    runs_needed = (total + per_run - 1) // per_run
    for k in range(runs_needed):
        f = forges[k % len(forges)]
        n = min(per_run, total - k * per_run)
        PLAN.setdefault(f, []).append(
            (mode, n, seed_base + k * 100, extra))


add(forges=[0, 1, 2, 3], mode="nominal", total=60, seed_base=1000)
add(forges=list(range(10)), mode="params", total=280, seed_base=2000, per_run=14)
add(forges=[4, 5, 6, 7, 8, 9], mode="noise", total=130, seed_base=3000, per_run=22)
add(forges=[0, 1, 2, 3], mode="physics", total=70, seed_base=4000, per_run=18)
add(forges=[4, 5, 6, 7, 8, 9], mode="visual", total=60, seed_base=5000, per_run=10)

# extra runs to cover the success-rate gap (~74% observed), fresh seed block
TOPUP: list[tuple[str, int, int, str]] = [
    ("nominal", 15, 6000, ""), ("params", 14, 6100, ""), ("params", 14, 6200, ""),
    ("params", 14, 6300, ""), ("noise", 22, 6400, ""), ("noise", 22, 6500, ""),
    ("physics", 18, 6600, ""), ("visual", 10, 6700, ""),
]

LOCK = threading.Lock()
TALLY = {"attempted": 0, "succeeded": 0, "runs": []}
TAG = ""  # batch-name wave tag, set by main()


def delivered_seed0s() -> set[int]:
    """seed0 of every prod_* batch that actually landed on HDFS (batch meta.json)."""
    out = subprocess.run(["hdfs", "dfs", "-ls", HDFS_BASE], capture_output=True, text=True)
    batches = [l.split()[-1] for l in out.stdout.splitlines()
               if "/prod_" in l and l.startswith("d")]
    seeds = set()
    for b in batches:
        r = subprocess.run(["hdfs", "dfs", "-cat", f"{b}/meta.json"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            try:
                seeds.add(int(json.loads(r.stdout)["seed0"]))
            except (ValueError, KeyError) as exc:
                print(f"[prod] WARN: bad meta in {b}: {exc!r}", flush=True)
        else:
            print(f"[prod] WARN: no meta.json yet in {b} (in flight?)", flush=True)
    return seeds


def wave3_plan(live: list[int]) -> dict[int, list[tuple[str, int, int, str]]]:
    """Ground truth from HDFS: re-run every plan entry with no delivered batch."""
    got = delivered_seed0s()
    entries = [e for q in PLAN.values() for e in q] + TOPUP
    pending = [e for e in entries
               if not any(e[2] <= s < e[2] + 100 for s in got)]
    print(f"[prod] wave3: {len(got)} batches delivered, {len(pending)} of "
          f"{len(entries)} runs still missing -> {len(live)} forges", flush=True)
    plan: dict[int, list[tuple[str, int, int, str]]] = {}
    for k, e in enumerate(pending):
        plan.setdefault(live[k % len(live)], []).append(e)
    return plan


def forge_url(i: int) -> str:
    out = subprocess.run(["hdfs", "dfs", "-cat", f"{REG}/forge_{i}.txt"],
                         capture_output=True, text=True)
    return [l for l in out.stdout.splitlines() if l.startswith("http")][-1]


def run_one(url: str, mode: str, n: int, seed0: int, extra: str, batch: str) -> dict:
    args = f"--headless --episodes {n} --seed0 {seed0} --mode {mode} --batch {batch}"
    if extra:
        args += f" {extra}"
    p = subprocess.run(
        ["python", str(CLIENT), "--forge-url", url, "run", "--task", "plated_meal",
         "--module", "gen_batch", f"--args={args}", "--timeout", "3500"],
        capture_output=True, text=True, timeout=3700)
    out = p.stdout + p.stderr
    ok_n = 0
    for line in out.splitlines():
        if line.startswith("GEN_BATCH: DONE"):
            ok_n = int(line.split()[2].split("/")[0])
    return {"batch": batch, "mode": mode, "episodes": n, "seed0": seed0,
            "successes": ok_n, "rc": p.returncode}


def worker(fi: int, queue) -> None:
    url = forge_url(fi)
    for attempt in range(2):
        sub = subprocess.run(["python", str(CLIENT), "--forge-url", url, "submit",
                              "--task", "plated_meal", "--dir", SUBMIT_DIR],
                             capture_output=True, text=True, timeout=120)
        if sub.returncode == 0:
            break
        print(f"[prod] f{fi}: submit rc={sub.returncode} "
              f"({(sub.stderr or sub.stdout)[-150:].strip()}) — "
              f"{'retrying' if attempt == 0 else 'ABANDONING forge'}", flush=True)
        time.sleep(15)
    else:
        return
    for j, (mode, n, seed0, extra) in enumerate(queue):
        batch = f"prod_{mode}_{TAG}f{fi}_{j}"
        for attempt in range(2):
            try:
                r = run_one(url, mode, n, seed0 + attempt * 37, extra, batch)
            except Exception as exc:  # noqa: BLE001 -- a dead run must not kill the queue
                r = {"batch": batch, "mode": mode, "episodes": n, "seed0": seed0,
                     "successes": 0, "rc": -1, "error": repr(exc)[:200]}
            with LOCK:
                TALLY["attempted"] += r["episodes"]
                TALLY["succeeded"] += r["successes"]
                TALLY["runs"].append(r)
                print(f"[prod] f{fi} {batch}: {r['successes']}/{r['episodes']} "
                      f"(total {TALLY['succeeded']}/{TALLY['attempted']})", flush=True)
                (HERE / f"production_tally{('_' + TAG) if TAG else ''}.json").write_text(
                    json.dumps(TALLY, indent=1))
            if r["successes"] > 0:
                break
            print(f"[prod] f{fi} {batch}: zero yield (rc={r['rc']}) — "
                  f"{'retrying' if attempt == 0 else 'giving up'}", flush=True)


def live_forges(candidates=range(50)) -> list[int]:
    """Forge indices whose registry entry answers /ping."""
    import urllib.request
    live = []
    for i in candidates:
        try:
            url = forge_url(i)
            with urllib.request.urlopen(f"{url}/ping", timeout=8) as resp:
                if resp.status == 200:
                    live.append(i)
        except Exception:  # noqa: BLE001 -- not-yet-booted forges are expected here
            pass
    return live


def main() -> None:
    global TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, default=1)
    args = ap.parse_args()
    plan = PLAN
    if args.wave >= 2:
        TAG = f"w{args.wave}"
        forges = live_forges()
        print(f"[prod] {TAG}: {len(forges)} live forges: {forges}", flush=True)
        plan = wave3_plan(forges)
    threads = [threading.Thread(target=worker, args=(fi, q), daemon=True)
               for fi, q in sorted(plan.items())]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"[prod] DONE in {(time.time() - t0) / 60:.0f} min: "
          f"{TALLY['succeeded']}/{TALLY['attempted']} verified episodes", flush=True)


if __name__ == "__main__":
    main()

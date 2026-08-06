"""Pull a batch's results back and report the two deliverables: the videos, and the success rate.

Reads what the pods pushed to HDFS (one .json verdict, one .mp4, one .npz per episode), mirrors
it locally, and prints the rate plus a per-parameter breakdown. The breakdown is the input to the
repair round: it is where you see which band or which strategy choice is costing yield.

Episodes that never reported are listed separately and NOT counted as failures — a pod that was
preempted before it finished says nothing about the policy, and folding it into the rate would
quietly understate it.

  python collect.py --batch nut_v1 --n 100
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

OUT_ROOT = "hdfs://haruna/tmp/zeyu.shen/cosigen_div"
LOCAL_ROOT = Path("/home/tiger/cap-x/eval_result/diversification")


def fetch(batch: str) -> Path:
    local = LOCAL_ROOT / batch
    local.mkdir(parents=True, exist_ok=True)
    for pat in ("*.json", "*.mp4", "*.npz"):
        subprocess.run(["hdfs", "dfs", "-get", "-f", f"{OUT_ROOT}/{batch}/{pat}", str(local)],
                       capture_output=True, text=True)
    return local


def report(local: Path, n_expected: int) -> dict:
    verdicts = []
    for f in sorted(local.glob("ep*.json")):
        try:
            verdicts.append(json.loads(f.read_text()))
        except ValueError:
            print(f"  ! {f.name} is not readable JSON")
    got = {v["index"] for v in verdicts}
    missing = [i for i in range(n_expected) if i not in got]
    ok = [v for v in verdicts if v.get("success")]
    videos = sorted(local.glob("ep*.mp4"))

    print(f"\n=== batch {local.name}")
    print(f"episodes reported : {len(verdicts)} / {n_expected}")
    print(f"successes         : {len(ok)}")
    if verdicts:
        print(f"SUCCESS RATE      : {len(ok)}/{len(verdicts)} = {100.0 * len(ok) / len(verdicts):.1f}% "
              f"(of reported episodes)")
    print(f"videos            : {len(videos)}  ({sum(f.stat().st_size for f in videos) / 1e6:.0f} MB)")
    if missing:
        print(f"no result yet     : {len(missing)} -> {missing[:12]}{' ...' if len(missing) > 12 else ''}")

    nominal = [v for v in verdicts if v["theta"].get("_nominal")]
    if nominal:
        print(f"control (ep0000)  : success={nominal[0]['success']} dz={nominal[0]['dz_mm']}mm "
              f"strokes={nominal[0]['strokes']}")

    # Where the yield goes: strategy choices first (they are the big lever), then which phase
    # failures died in.
    if verdicts:
        print("\nby strategy:")
        by = defaultdict(lambda: [0, 0])
        for v in verdicts:
            for k in ("sweep_deg", "nut_center", "w_land_deg"):
                cell = by[(k, v["theta"][k])]
                cell[0] += 1
                cell[1] += bool(v.get("success"))
        for (k, val), (tot, good) in sorted(by.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
            print(f"  {k:>11} = {str(val):>5}: {good}/{tot}")
        fails = [v for v in verdicts if not v.get("success")]
        if fails:
            print("\nfailures by final phase:")
            ph = defaultdict(int)
            for v in fails:
                ph[v.get("final_phase", "?")] += 1
            for k, c in sorted(ph.items(), key=lambda kv: -kv[1]):
                print(f"  {k:>8}: {c}")
            print("\nworst-case geometry of failures (dz above seat, lateral):")
            for v in sorted(fails, key=lambda v: -v["dz_mm"])[:5]:
                print(f"  ep{v['index']:04d} dz={v['dz_mm']:6.1f}mm lat={v['lat_mm']:5.1f}mm "
                      f"strokes={v['strokes']:3d} phase={v.get('final_phase')}")

    summary = {"batch": local.name, "expected": n_expected, "reported": len(verdicts),
               "successes": len(ok), "videos": len(videos), "missing": missing,
               "rate": (len(ok) / len(verdicts)) if verdicts else None}
    (local / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args()
    local = (LOCAL_ROOT / a.batch) if a.no_fetch else fetch(a.batch)
    report(local, a.n)
    print(f"\nvideos + trajectories: {local}")


if __name__ == "__main__":
    main()

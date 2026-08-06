"""Are the successful trajectories actually different, or the same motion re-run?

Sampling diverse parameters does not by itself make diverse trajectories: a closed loop can absorb
a perturbation and converge to the same path. This measures the trajectories themselves.

Three numbers, each with a physical yardstick so the answer is not "0.043, is that a lot?":

  pairwise path distance  — every successful episode's end-effector path resampled to a common
                            length, then mean RMS distance between each pair. Yardstick: the task
                            scale (workspace span) and the part size (a 24 mm nut, a 10-12 mm pen).
  joint-space spread      — std over episodes of each arm joint at matched progress, in radians.
                            Two runs of the same motion differ by ~0 here even if timing differs.
  feature spread          — duration, path length, and where in the workspace the work happened.

Also writes an overlay plot of the end-effector paths, which is the version you can eyeball.

  python diversity_report.py --batch pen_b2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/tiger/cap-x/eval_result/diversification")


def resample(arr: np.ndarray, n: int = 200) -> np.ndarray:
    """Resample a (T, d) path to (n, d) by normalised index — time-warp invariant, so two runs of
    the same motion at different speeds still compare as identical."""
    t = np.linspace(0, 1, len(arr))
    tn = np.linspace(0, 1, n)
    return np.stack([np.interp(tn, t, arr[:, j]) for j in range(arr.shape[1])], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--points", type=int, default=200)
    a = ap.parse_args()
    d = ROOT / a.batch

    eps, paths, joints, feats = [], [], [], []
    for f in sorted(d.glob("ep*.npz")):
        idx = int(f.stem[2:6])
        vj = d / f"ep{idx:04d}.json"
        if not vj.exists():
            continue
        v = json.loads(vj.read_text())
        if not v.get("success"):
            continue
        z = np.load(f, allow_pickle=True)
        ee = np.asarray(z["ee_pos"], dtype=float)
        jp = np.asarray(z["joint_pos"], dtype=float)[:, :7]
        eps.append((idx, v))
        paths.append(resample(ee, a.points))
        joints.append(resample(jp, a.points))
        step = np.linalg.norm(np.diff(ee, axis=0), axis=1)
        feats.append({"idx": idx, "steps": len(ee), "path_m": float(step.sum()),
                      "span_xy_m": float(np.ptp(ee[:, :2], axis=0).max()),
                      "z_range_m": float(np.ptp(ee[:, 2]))})
    if len(paths) < 2:
        print(f"{len(paths)} successful episodes with trajectories — need >= 2")
        return

    P = np.stack(paths)          # (E, n, 3)
    J = np.stack(joints)         # (E, n, 7)
    E = len(P)
    dist = np.zeros((E, E))
    for i in range(E):
        for j in range(E):
            dist[i, j] = np.sqrt(((P[i] - P[j]) ** 2).sum(axis=1).mean())
    off = dist[~np.eye(E, dtype=bool)]

    workspace = float(np.ptp(P.reshape(-1, 3), axis=0).max())
    print(f"=== {a.batch}: {E} successful trajectories\n")
    print("pairwise end-effector path distance (RMS over the resampled path):")
    print(f"  mean {off.mean() * 100:6.1f} cm   median {np.median(off) * 100:6.1f} cm   "
          f"min {off.min() * 100:6.1f} cm   max {off.max() * 100:6.1f} cm")
    print(f"  yardstick: workspace span covered by these paths = {workspace * 100:.1f} cm")
    print(f"  closest pair: ep{eps[np.unravel_index(np.argmin(dist + np.eye(E) * 9), dist.shape)[0]][0]:04d} "
          f"/ ep{eps[np.unravel_index(np.argmin(dist + np.eye(E) * 9), dist.shape)[1]][0]:04d}")

    spread = J.std(axis=0).mean(axis=0)   # (7,) mean over path of per-joint std across episodes
    print("\njoint-space spread across episodes (rad, mean over the path):")
    print("  " + "  ".join(f"q{k + 1}={spread[k]:.3f}" for k in range(7))
          + f"   | mean {spread.mean():.3f} rad = {np.degrees(spread.mean()):.1f} deg")

    print("\nper-episode features:")
    print(f"  {'ep':>6} {'steps':>7} {'path (m)':>9} {'xy span':>8} {'z range':>8}")
    for f in feats:
        print(f"  {f['idx']:6d} {f['steps']:7d} {f['path_m']:9.2f} {f['span_xy_m']:8.3f} "
              f"{f['z_range_m']:8.3f}")
    arr = np.array([[f["steps"], f["path_m"]] for f in feats], dtype=float)
    print(f"  duration  min/max = {arr[:, 0].min():.0f}/{arr[:, 0].max():.0f} steps "
          f"({arr[:, 0].max() / max(1, arr[:, 0].min()):.1f}x)")
    print(f"  path len  min/max = {arr[:, 1].min():.2f}/{arr[:, 1].max():.2f} m "
          f"({arr[:, 1].max() / max(1e-6, arr[:, 1].min()):.1f}x)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), dpi=130)
        for (idx, _v), p in zip(eps, P):
            ax[0].plot(p[:, 0], p[:, 1], lw=1.2, alpha=0.85, label=f"ep{idx:04d}")
            ax[1].plot(np.linspace(0, 1, len(p)), p[:, 2], lw=1.2, alpha=0.85)
        ax[0].set_xlabel("x (m)"), ax[0].set_ylabel("y (m)")
        ax[0].set_title(f"{a.batch}: end-effector paths, {E} successful episodes")
        ax[0].set_aspect("equal", adjustable="datalim")
        ax[0].legend(fontsize=6, ncol=2)
        ax[1].set_xlabel("normalised episode progress"), ax[1].set_ylabel("end-effector z (m)")
        ax[1].set_title("height profiles")
        for x in ax:
            x.grid(alpha=0.3)
        out = d / "diversity_paths.png"
        fig.tight_layout()
        fig.savefig(out)
        print(f"\nplot: {out}")
    except ImportError:
        print("\n(matplotlib absent — skipped the overlay plot)")


if __name__ == "__main__":
    main()

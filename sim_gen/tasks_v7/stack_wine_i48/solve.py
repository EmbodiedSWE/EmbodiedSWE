"""Teleport solution for CaskWeightSortScene (sim_gen task `stack_wine_i48`) — the
task's legitimacy certificate.

The task is epistemic: three visually identical kegs carry a shuffled mass
permutation (0.3 / 0.9 / 2.7 kg), and success requires racking them SORTED BY
WEIGHT into color-coded cradles. This solver therefore has a measurement phase
that no amount of teleporting can replace:

  P1 — PROBES (pure contact dynamics): each keg in turn gets the SAME short force
  pulse (`set_external_force_and_torque`, world -y, 10 substeps from rest) and is
  left to roll out and coast. Displacement ∝ v0 ∝ 1/m (identical pulse, identical
  damping), so the light keg rolls ~3x farther than the middle one and ~9x farther
  than the heavy one. The kegs are RANKED BY MEASURED DISPLACEMENT — the ground
  truth `scene.rank_of_cask` is printed for audit but NEVER used for control. If
  adjacent displacement ratios are ambiguous (< 1.5) the whole battery re-runs
  with a stronger pulse.

  P2 — PLACEMENT (teleport = TRANSPORT ONLY, ending in FREE SPACE): each keg is
  teleported to a point 60 mm ABOVE its measured-rank cradle (orientation
  preserved — rolling only changed the spin about its own axis — velocities
  zeroed) and then DROPS under gravity, wedging between the two rails through
  real contact. Seating is verified by readback; a failed seat is retried from a
  re-centred hover. The keg is never teleported into the wedge itself.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds
after success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds.

Run (forge): python -u -m simgen_tasks.stack_wine_i48.solve --headless [--seed N]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import os
import threading

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

ROLE_NAMES = scene_mod.ROLE_NAMES

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cask_weight_sort")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def cask_xy(i: int) -> torch.Tensor:
        """(2,) keg i centre, env-local."""
        return (scene.casks[i].data.root_pos_w - scene.env_origins)[0, :2]

    def clear_wrench(i: int) -> None:
        scene.casks[i].set_external_force_and_torque(
            zero_wrench, zero_wrench, env_ids=all_ids)

    def report(tag: str) -> None:
        seated = scene.seated_correct()[0]
        print(f"[solve] {tag:16s} | seated_correct={[bool(v) for v in seated]} "
              f"settled={[bool(v) for v in scene.settled()[0]]} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, layout readback -------------------------------
    step(90)
    bay_p = scene._bay_pos()[0]  # (3,3) row j = role j (RED/YELLOW/GREEN)
    order = torch.argsort(bay_p[:, 0]).tolist()
    print(f"[solve] layout readback (seed {args.seed}): bays left-to-right = "
          + " ".join(f"{ROLE_NAMES[j]}@({float(bay_p[j, 0]):+.3f},{float(bay_p[j, 1]):+.3f})"
                     for j in order), flush=True)
    print("[solve] kegs: " + " ".join(
        f"cask_{i}@({float(cask_xy(i)[0]):+.3f},{float(cask_xy(i)[1]):+.3f})"
        for i in range(3)), flush=True)
    gt_rank = scene.rank_of_cask[0].tolist()
    gt_mass = [float(v) for v in scene.mass_of_cask[0]]
    print(f"[solve] ground truth (AUDIT ONLY, not used for control): "
          f"rank_of_cask={gt_rank} masses={[f'{m:.1f}' for m in gt_mass]}", flush=True)
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: probes (contact dynamics, the measurement) -------------------
    def probe_all(force: float) -> list[float]:
        """Give every keg the same -y pulse from rest and return xy displacements."""
        disps: list[float] = []
        for i in range(3):
            step(60)  # make sure this keg starts from (near) rest
            p0 = cask_xy(i).clone()
            f = torch.tensor([0.0, -force, 0.0], device=device)
            for _ in range(10):
                scene.casks[i].set_external_force_and_torque(
                    f.view(1, 1, 3).expand(n, 1, 3).contiguous(), zero_wrench,
                    env_ids=all_ids, is_global=True)
                env.step(no_action)
            clear_wrench(i)
            step(300)  # coast 2.5 s — same fraction of the full roll for every keg
            d = cask_xy(i) - p0
            disps.append(float(d.norm()))
            print(f"[solve] probe cask_{i}: F={force:.2f} N x 10 steps -> "
                  f"d=({float(d[0]):+.3f},{float(d[1]):+.3f}) |d|={disps[-1]:.3f} m",
                  flush=True)
        return disps

    force = 0.8
    disp = probe_all(force)
    for attempt in range(2):
        srt = sorted(disp)
        ratios = (srt[1] / max(srt[0], 1e-6), srt[2] / max(srt[1], 1e-6))
        print(f"[solve] probe ranking: disps={[f'{d:.3f}' for d in disp]} "
              f"adjacent ratios={ratios[0]:.2f},{ratios[1]:.2f}", flush=True)
        if srt[0] > 0.008 and ratios[0] >= 1.5 and ratios[1] >= 1.5:
            break
        force *= 1.6
        print(f"[solve] probe ambiguous — re-probing with F={force:.2f} N", flush=True)
        disp = probe_all(force)
    else:
        srt = sorted(disp)
        ratios = (srt[1] / max(srt[0], 1e-6), srt[2] / max(srt[1], 1e-6))
        if not (srt[0] > 0.008 and ratios[0] >= 1.5 and ratios[1] >= 1.5):
            print("SIM_GEN_SOLVE: FAIL (probes never separated)", flush=True)
            os._exit(1)

    # Smallest displacement = heaviest = rank 0 (RED). Measured, not ground truth.
    idx_by_disp = sorted(range(3), key=lambda i: disp[i])
    meas_rank = [0, 0, 0]
    for r, i in enumerate(idx_by_disp):
        meas_rank[i] = r
    match = "MATCH" if meas_rank == gt_rank else "MISMATCH"
    print(f"[solve] measured rank_of_cask={meas_rank} (ground truth {gt_rank}: {match})",
          flush=True)
    s1 = print_score("P1 probes")
    assert s1 >= s0 - 1e-6, "score decreased across the probes"

    # ---------------- phase 2: placement (teleport transport + contact drop) ----------------
    def seat(i: int, role: int) -> bool:
        """Teleport keg i to a hover ABOVE role's cradle (free space, velocities
        zeroed), let it DROP and wedge through contact; verify by readback."""
        for attempt in range(3):
            tgt = scene._bay_pos()[0, role]  # env-local plate origin
            st = torch.zeros(n, 13, device=device)
            st[:, 0] = tgt[0]
            st[:, 1] = tgt[1]
            st[:, 2] = c.seat_z + 0.060  # hover: free space above the rails
            if attempt == 0:
                st[:, 3:7] = scene.casks[i].data.root_quat_w[0]  # orientation preserved
            else:
                st[:, 3] = 1.0  # retry: axis exactly along the trough
            st[:, 0:3] += scene.env_origins
            scene.casks[i].write_root_state_to_sim(st, all_ids)
            step(240)  # drop + wedge + settle (2 s)
            ok = bool(scene.seated_matrix()[0, i, role])
            rel = scene._cask_pos()[0, i] - scene._bay_pos()[0, role]
            print(f"[solve] seat cask_{i} -> {ROLE_NAMES[role]} attempt {attempt}: "
                  f"rel=({float(rel[0]):+.3f},{float(rel[1]):+.3f},{float(rel[2]):+.3f}) "
                  f"seated={ok}", flush=True)
            if ok:
                return True
        return False

    s_prev = s1
    for i in range(3):
        if not seat(i, meas_rank[i]):
            report("FAIL-state")
            print(f"SIM_GEN_SOLVE: FAIL (cask_{i} never seated)", flush=True)
            os._exit(1)
        s_now = print_score(f"P2 seat cask_{i} in {ROLE_NAMES[meas_rank[i]]}")
        assert s_now >= s_prev - 1e-6, "score decreased across a seating"
        s_prev = s_now

    step(120)  # 1 s hands-off settle before judging
    report("racked")
    s2 = print_score("P2 all racked + settle")
    assert s2 >= s_prev - 1e-6, "score decreased across the final settle"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after racking)", flush=True)
        os._exit(1)

    # ---------------- phase 3: persistence (>= 3.3 simulated seconds, no intervention) ------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s3 = print_score("P3 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s3 >= s2 - 1e-6
    if ok:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)

    code = 0 if ok else 1
    # Hard exit: Kit teardown hangs — watchdog then die.
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001 — die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)

"""Teleport solution for CupShellsScene (sim_gen task `stack_cups_i92`) — the task's
legitimacy certificate.

Teleportation handles TRANSPORT ONLY. The load-bearing interaction — capping each die
with its inverted cup so the rim seals on the floor around it — goes through contact
dynamics, one color pair at a time:
  1. FLIP+CARRY (transport): the cup is teleported from its rim-up row pose to a
     hover pose ALREADY INVERTED (a 180-deg reorientation is part of transport),
     mouth ~23 mm above the matching die's top. Nothing is in contact; the die is
     outside the cup.
  2. CAP (dynamics): a velocity-regulated vertical force (gravity feed-forward + PD
     to -0.06 m/s, force at the CoM only — no torque, no orientation pinning) lowers
     the cup so the die passes through the open mouth and the rim descends to the
     floor; an xy PD keeps the cup centered over the die. The wrench is DROPPED as
     soon as the rim reads sealed (or descent stalls in contact); the final seal is
     rim-on-floor contact carrying the cup's weight.
  3. Repeat for the other two colors; nothing is ever teleported into contact, never
     welded, never held at the end.
After all three pairs read covered, hands off: `SIM_GEN_SCORE` is printed at every
phase boundary (non-decreasing — flip credit is latched, covered credit is physical
and stays), success() must hold through a >= 3.3 simulated-second persistence window
with no intervention, and only then `SIM_GEN_SOLVE: SUCCESS` is printed.

Run (forge): python -u -m simgen_tasks.stack_cups_i92.solve --headless [--seed N]
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

from_scene = scene_mod
COLORS = from_scene.COLORS

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cup_shells")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def wrench(body, f_w: torch.Tensor) -> None:
        """Apply a WORLD force (n,3) to `body`, expressed in its CURRENT link frame
        (`is_global=True` silently drops torques on this stack — transform manually,
        the house convention). Re-set every step while pushing."""
        from isaaclab.utils.math import quat_apply_inverse

        q = body.data.root_link_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f_w).unsqueeze(1),
            torch.zeros(n, 1, 3, device=device),
            env_ids=all_ids)

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        cov = scene.covered()[0]
        rad = scene.die_radial()[0]
        mh = scene.mouth_height()[0]
        up = scene.cup_up_z()[0]
        dh = scene.die_height()[0]
        pairs = " ".join(
            f"{nm}:cov={bool(cov[i])} rad={float(rad[i]) * 1000:.0f}mm "
            f"mouth={float(mh[i]) * 1000:.0f}mm up_z={float(up[i]):+.2f} "
            f"die_h={float(dh[i]) * 1000:.0f}mm"
            for i, (nm, _) in enumerate(COLORS))
        print(f"[solve] {tag:14s} | {pairs} | success={bool(scene.success()[0])}"
              f" score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    for i, (nm, _) in enumerate(COLORS):
        cu, di = rel(scene.cups[i]), rel(scene.dice[i])
        print(f"[solve] layout (seed {args.seed}) {nm}: "
              f"cup=({float(cu[0]):+.3f},{float(cu[1]):+.3f},{float(cu[2]):.3f}) "
              f"die=({float(di[0]):+.3f},{float(di[1]):+.3f},{float(di[2]):.3f})",
              flush=True)
    report("reset")
    last = print_score("P0 reset+settle")
    assert last <= 0.02, "null credit at reset — rubric leak"

    # ---------------- per pair: FLIP+CARRY (transport) then CAP (dynamics) -----------------
    q_inv = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)  # 180 deg about x
    for i, (nm, _) in enumerate(COLORS):
        cup, die = scene.cups[i], scene.dice[i]

        # --- FLIP+CARRY: hover inverted above the die; mouth ~23 mm over the die top ---
        d_pos = die.data.root_pos_w.clone()
        hover_mouth = c.die_s + 0.023  # mouth-plane height at hover
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = d_pos[:, 0:2]
        st[:, 2] = scene.env_origins[:, 2] + hover_mouth + c.height / 2
        st[:, 3:7] = q_inv
        cup.write_root_state_to_sim(st, all_ids)
        # regulated zero-velocity hold while the flip latch reads the pose
        for _ in range(12):
            v = cup.data.root_lin_vel_w
            f = torch.zeros(n, 3, device=device)
            f[:, 2] = (c.cup_mass * 9.81 + 5.0 * (0.0 - v[:, 2])).clamp(
                0.0, 2 * c.cup_mass * 9.81)
            f[:, 0:2] = (3.0 * (d_pos[:, 0:2] - cup.data.root_pos_w[:, 0:2])
                         - 1.5 * v[:, 0:2]).clamp(-0.8, 0.8)
            wrench(cup, f)
            env.step(no_action)
        report(f"hover-{nm}")
        assert float(scene.mouth_height()[0, i]) > c.rim_z_max + 0.02, \
            "hover must start with the rim well OFF the floor"
        s = print_score(f"P{2 * i + 1} {nm} cup flipped + carried to hover (transport)")
        assert s >= last - 1e-6, "score decreased at flip"
        last = s

        # --- CAP: velocity-regulated descent until the rim seals on the floor ---
        sealed = False
        for k in range(600):
            mh = float(scene.mouth_height()[0, i])
            vz = float(cup.data.root_lin_vel_w[0, 2])
            if mh < 0.004 or (mh < 0.012 and abs(vz) < 0.004 and k > 40):
                sealed = True
                break
            v = cup.data.root_lin_vel_w
            f = torch.zeros(n, 3, device=device)
            f[:, 2] = (c.cup_mass * 9.81 + 5.0 * (-0.06 - v[:, 2])).clamp(
                0.0, 2 * c.cup_mass * 9.81)
            f[:, 0:2] = (3.0 * (die.data.root_pos_w[:, 0:2]
                                - cup.data.root_pos_w[:, 0:2])
                         - 1.5 * v[:, 0:2]).clamp(-0.8, 0.8)
            wrench(cup, f)
            env.step(no_action)
        wrench(cup, torch.zeros(n, 3, device=device))
        assert sealed, f"{nm} cup descent never reached the floor seal"
        step(60)  # hands off: rim-on-floor contact carries the cup; pair settles
        report(f"capped-{nm}")
        assert bool(scene.covered()[0, i]), f"{nm} pair did not read covered"
        s = print_score(f"P{2 * i + 2} {nm} die capped (contact dynamics)")
        assert s >= last - 1e-6, "score decreased across capping"
        last = s

    # ---------------- final: hands off, success + persistence ------------------------------
    step(60)
    report("all-capped")
    s = print_score("P7 all three pairs sealed, hands off")
    assert s >= last - 1e-6
    last = s
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after capping all pairs)", flush=True)
        os._exit(1)

    hold, flickers = True, 0
    for k in range(400):  # 400 substeps = 3.33 s at 120 Hz, no intervention
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:
                cov = scene.covered()[0]
                print(f"[solve] persist flicker @step {k}: covered="
                      f"{[bool(v) for v in cov]}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s = print_score("P8 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s >= last - 1e-6
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)

"""solve — demonstration solution for StoveDialInterlockScene (…scene9_turn_on_the_stove_i338).

Scene-level env (robot="null"). This solve uses NO teleports at all: every
interaction is a cranking torque on a dial's z-bearing (the stand-in for the
Franka gripping the colored peg and orbiting it, TASK.md), applied through the
scene's `rotor_tau` wrench slot. The goal event — the gas fence dropping ~22 mm
into the three aligned notches — is entirely passive gravity + contact:
  - ALIGN L, M, R (any order works; we go left to right): a velocity-cascade
    torque servo turns each dial until its notch bearing (yaw -> 0) sits under
    its fence foot, released only when close AND slow;
  - DROP: when the third notch clears its foot, the fence FALLS on its
    prismatic joint and lands by contact on the drum tops inside the notches
    (never commanded, never touched).

Prints `SIM_GEN_SCORE <v>` at phase boundaries (asserted non-decreasing:
latched/held credit must not evaporate). After success() first holds, keeps
simulating >= 3.5 more simulated seconds with all drive buffers zero
(asserted); only if success() still holds prints exactly `SIM_GEN_SOLVE:
SUCCESS`. Hard exit (os._exit) after the verdict, watchdog Timer as backstop —
Kit teardown hangs otherwise.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene9_turn_on_the_stove_i338.solve --headless [--seed N]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_sec", type=float, default=600.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.libero_kitchen_scene9_turn_on_the_stove_i338 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls (daemon Timer).
_wd = threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                             os._exit(3)))
_wd.daemon = True
_wd.start()

# Dial crank servo (hand-scale numbers; rotor Iz = 1.1e-3 kg*m^2, ang damping 0.8/s).
# Discrete stability against the one-substep wrench delay:
#   KW*dt/Iz = 0.08*(1/120)/1.1e-3 ~ 0.61 < 1.
K_A = 4.0  # 1/s outer angle->rate gain
W_CAP = 1.2  # rad/s rate cap while cranking
KW = 0.08  # N*m*s/rad inner rate gain
TAU_MAX = 0.30  # N*m hard cap (an easy two-finger crank on the 22 mm-orbit peg)
E_DONE = math.radians(2.0)  # rad: angle band for servo exit...
W_DONE = 0.06  # rad/s: ...and slow (never release a spinning dial)


def wrap(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.stove_dial_interlock")().build(num_envs=1, device=device)
    scene = env.scene
    no_action = torch.empty(0, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        e = scene.align_err_deg()[0]
        al = scene.aligned()[0]
        print(f"[solve] {tag:12s} err=({float(e[0]):6.1f}, {float(e[1]):6.1f}, "
              f"{float(e[2]):6.1f})deg aligned=({int(al[0])},{int(al[1])},{int(al[2])}) "
              f"drop={float(scene.fence_drop()[0]) * 1000:5.1f}mm "
              f"dropped={bool(scene.dropped()[0])} settled={bool(scene.settled()[0])} "
              f"score={sc():.3f} success={bool(scene.success()[0])}", flush=True)

    last_score = -1.0

    def phase_score(tag: str) -> None:
        nonlocal last_score
        s = sc()
        assert s >= last_score - 1e-6, f"score decreased at {tag}: {last_score} -> {s}"
        last_score = s
        print(f"SIM_GEN_SCORE {s:.3f}", flush=True)

    def verdict(ok: bool) -> None:
        print("SIM_GEN_SOLVE: SUCCESS" if ok else "SIM_GEN_SOLVE: FAIL", flush=True)
        code = 0 if ok else 1
        t = threading.Timer(10.0, lambda: os._exit(code))
        t.daemon = True
        t.start()
        try:
            env.close()
            app.close()
        except Exception:  # noqa: BLE001
            pass
        os._exit(code)

    def crank_to_zero(i: int, budget: int = 1800) -> bool:
        """Velocity-cascade torque servo on dial i's bearing: turn the notch bearing
        (yaw) to 0 through the live revolute joint against damping; release only when
        close AND slow. Shortest way around — either crank direction is legal."""
        for k in range(budget):
            yaw = float(scene.yaws()[0, i])
            w = float(scene.rotors[i].data.root_ang_vel_w[0, 2])
            e = wrap(-yaw)
            if abs(e) < E_DONE and abs(w) < W_DONE:
                scene.rotor_tau[0, i] = 0.0
                return True
            w_des = max(-W_CAP, min(W_CAP, K_A * e))
            scene.rotor_tau[0, i] = max(-TAU_MAX, min(TAU_MAX, KW * (w_des - w)))
            step(1)
            if k and k % 240 == 0:
                report(f"crank[{i}]")
        scene.rotor_tau[0, i] = 0.0
        return False

    # ================= reset + settle ============================================================
    env.reset(seed=args.seed)
    step(60)
    y0 = scene.yaw0[0]
    print(f"[solve] seed={args.seed} start yaws="
          f"({math.degrees(float(y0[0])):.1f}, {math.degrees(float(y0[1])):.1f}, "
          f"{math.degrees(float(y0[2])):.1f})deg "
          f"phys_window=+-{scene.cfg.phys_window_deg:.1f}deg", flush=True)
    report("reset")
    assert float(scene.rotor_tau.abs().max()) == 0.0 and float(scene.fence_f.abs().max()) == 0.0
    assert not bool(scene.dropped()[0]), "fence must start riding the walls"
    phase_score("reset")  # ~0.000

    # ================= PHASES 1..3: crank each dial into alignment ==============================
    for n, i in enumerate((0, 1, 2), start=1):
        if not crank_to_zero(i):
            report(f"align{i}-fail")
            print(f"[solve] PHASE {n} FAILED: dial {i} did not reach alignment", flush=True)
            verdict(False)
        step(30)
        report(f"aligned[{i}]")
        if n < 3:
            assert bool(scene.aligned()[0, i]), f"dial {i} must read aligned after the crank"
            assert not bool(scene.dropped()[0]), \
                "fence must NOT drop before all three dials align"
            phase_score(f"phase{n}-align")  # ~0.2 * n

    # ================= DROP: passive gravity follower ============================================
    ok = False
    for _ in range(60):  # up to 5 s for fall + settle + success to hold
        if bool(scene.success()[0]):
            ok = True
            break
        step(10)
    report("dropped")
    if not ok:
        print("[solve] PHASE 3 FAILED: fence did not drop into the aligned notches", flush=True)
        verdict(False)
    phase_score("phase3-drop")  # 1.000

    # ================= PHASE 4: persistence (>= 3.5 simulated seconds, hands off) ================
    assert float(scene.rotor_tau.abs().max()) == 0.0 and float(scene.fence_f.abs().max()) == 0.0, \
        "drives must be zero for persistence"
    persist = int(round(3.5 / env.dt))  # 420 physics steps at 1/120 s
    step(persist)
    report("final")
    phase_score("final")
    still_ok = bool(scene.success()[0]) and abs(sc() - 1.0) < 1e-3
    print(f"[solve] persistence: {persist} steps ({persist * env.dt:.2f} s) hands-off, "
          f"success={still_ok}", flush=True)
    verdict(still_ok)


if __name__ == "__main__":
    main()

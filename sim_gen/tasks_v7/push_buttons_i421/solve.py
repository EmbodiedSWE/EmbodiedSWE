"""solve — force-probe solution for DropSequencerScene (push_buttons_i421).

Scene-level env (robot="null"). NO teleports at all: nothing needs transporting — the
marbles start on their perches and gravity does the transport. Every action is a
contact-dynamics press through the scene's world-frame probe-force buffers
(frame-encoded plant-side in scene.post_step).

  PHASE 0  reset + settle; read the per-episode marble->chimney permutation from
           scene.chim (the readback the real agent gets by looking through the slits)
           and derive the press order: crimson's chimney, then amber's, then azure's.
  PHASE k  (x3, one per marble, in goal order) press-release-settle:
             press: capped velocity-servo force on that chimney's plunger cap along the
                    vault press axis (+y), feedforward = spring load (80 q + 0.6 N), to
                    the 20 mm hard stop; hold until the marble leaves its perch (rel-z
                    readback), then force off — the spring returns the plunger.
             settle: hands off while the marble falls the sealed shaft and rolls down
                    the 3 deg slope to the white stop; verify by readback that the
                    settled queue prefix advanced to k (a wrong drop can never fix
                    itself, so this is the whole ballgame).
           Gain audited: 6*dt/m = 6/(120*0.06) = 0.83 < 1; cap 4.5 N.
  PHASE 4  all forces off, wait for the sustained-success counter.
  PHASE 5  keep simulating >= 3 s more, hands-off; success() must still hold.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (asserted non-decreasing) and
exactly `SIM_GEN_SOLVE: SUCCESS` only if success() holds after the persistence window.
Hard exit (os._exit) with a daemon watchdog Timer backstop.

Run (forge): python -u -m simgen_tasks.push_buttons_i421.solve --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_sec", type=float, default=900.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import os  # noqa: E402
import threading  # noqa: E402

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.push_buttons_i421 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

G = scene_mod.G
NAMES = scene_mod.MARBLE_NAMES

# Watchdog: never leave a GPU zombie if anything below stalls.
_wd = threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                             os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.drop_sequencer")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        q = scene.rod_q()[0]
        rel = scene.marble_rel()[0]
        print(f"[solve] {tag:9s} q=({q[0] * 1000:5.1f},{q[1] * 1000:5.1f},"
              f"{q[2] * 1000:5.1f})mm "
              + " ".join(f"{nm[:2]}=({rel[i, 0] * 1000:+6.1f},{rel[i, 1] * 1000:+6.1f},"
                         f"{rel[i, 2] * 1000:6.1f})" for i, nm in enumerate(NAMES))
              + f" drop={int(scene._dropped[0, 0])}{int(scene._dropped[0, 1])}"
                f"{int(scene._dropped[0, 2])} p={int(scene.queue_prefix_now()[0])}"
                f"/{int(scene._prefix[0])} hold={int(scene._hold[0])} "
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

    def press_and_drop(rod: int, mi: int) -> bool:
        """Velocity-servo press to the hard stop; hold until marble mi leaves its perch;
        release (spring return). Gain audit: 6*dt/m = 6/(120*0.06) = 0.83 < 1."""
        key = f"rod{rod}"

        def servo(v_des: float) -> None:
            axis = scene.press_axis_w()[0]
            v = float((scene.rods[rod].data.root_lin_vel_w[0] * axis).sum())
            q = float(scene.rod_q()[0, rod])
            ff = c.spring_k * max(0.0, q) + 0.6
            f = max(0.0, min(4.5, 6.0 * (v_des - v) + ff))
            scene.push_w[key][0] = axis * f

        dropped = False
        at = 0
        for _ in range(900):
            servo(0.06 if at < 8 else 0.0)
            step(1)
            if float(scene.rod_q()[0, rod]) >= G.STROKE - 0.0025:
                at += 1
            else:
                at = max(0, at - 1)
            if bool(scene._dropped[0, mi]):
                dropped = True
                if at >= 8:
                    break
        scene.push_w[key][0] = 0.0
        return dropped

    def wait_prefix(k: int) -> bool:
        """Hands off: wait for the dropped marble to roll to the stop and the settled
        correct-prefix latch to reach k."""
        for _ in range(90):
            step(10)
            if int(scene._prefix[0]) >= k:
                return True
        return False

    # ================= PHASE 0: reset + settle + read the permutation =======================
    env.reset(seed=args.seed)
    step(60)
    report("reset")
    phase_score("reset")  # ~0.000

    chim = [int(v) for v in scene.chim[0]]  # chim[m] = chimney holding marble m
    print(f"[solve] permutation (slit readback): "
          + ", ".join(f"{nm} in chimney {chim[i]}" for i, nm in enumerate(NAMES))
          + f" -> press order rods {chim}", flush=True)

    # ================= PHASES 1-3: press-release-settle, marble by marble ===================
    expect = (0.35, 0.60, 0.85)
    for k, mi in enumerate((0, 1, 2)):  # goal order IS marble order: crimson, amber, azure
        if not press_and_drop(chim[mi], mi):
            report(f"press{k + 1}")
            print(f"[solve] PHASE {k + 1} FAILED: {NAMES[mi]} never left its perch",
                  flush=True)
            verdict(False)
        if not wait_prefix(k + 1):
            report(f"queue{k + 1}")
            print(f"[solve] PHASE {k + 1} FAILED: settled prefix never reached {k + 1}",
                  flush=True)
            verdict(False)
        step(30)  # extra settle
        report(f"queue{k + 1}")
        phase_score(f"queue{k + 1}")  # expect >= expect[k]
        assert sc() >= expect[k] - 1e-6, f"queue{k + 1} score {sc():.3f} < {expect[k]}"

    # ================= PHASE 4: wait for the sustained-success counter ======================
    for kk in scene.push_w:
        scene.push_w[kk].zero_()
    okd = False
    for _ in range(60):
        step(10)
        if bool(scene.success()[0]):
            okd = True
            break
    report("success")
    if not okd:
        print("[solve] PHASE 4 FAILED: success never sustained", flush=True)
        verdict(False)
    phase_score("phase4")  # 1.000

    # ================= PHASE 5: persistence (>= 3 simulated seconds, hands off) =============
    persist = int(round(3.5 / env.dt))  # 420 substeps at 1/120 s
    step(persist)
    report("final")
    phase_score("final")
    still_ok = bool(scene.success()[0]) and sc() == 1.0
    print(f"[solve] persistence: {persist} steps ({persist * env.dt:.2f} s) hands-off, "
          f"success={still_ok}", flush=True)
    verdict(still_ok)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)

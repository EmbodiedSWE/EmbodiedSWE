"""solve — force-probe solution for ShutterTrapScene (push_buttons_i273).

Scene-level env (robot="null"). This solve uses NO teleports at all: every action goes
through contact dynamics via the scene's world-frame probe-force buffers (frame-encoded
plant-side in scene.post_step). The plate starts captive in its corridor, so there is
nothing to transport.

  PHASE 0  reset + settle; read WHICH corridor end the plate spawned at (randomized) and
           order the buttons nearest-shutter-first (the geometry forces this order).
  PHASE k  (x3, one per button) press-release-chase:
             press: capped velocity-servo force straight down on the plunger head to the
                    24 mm hard stop (against the ~0.6-0.9 N spring return), short hold;
             chase: force off the plunger, then a capped velocity-servo push on the PLATE
                    toward a staging target that covers the just-pressed head (with the
                    judged 5 mm margin) while stopping >= 9 mm short of the next raised
                    head; brake, settle, verify the head is trapped at depth by readback.
           Gains audited: press 20*dt/m = 20/(120*0.2) = 0.83 < 1; plate 25*dt/m =
           25/(120*0.4) = 0.52 < 1. The rise window is ~3 s; each chase takes < 1.5 s.
  PHASE 4  all forces off, wait for the sustained-success counter.
  PHASE 5  keep simulating >= 3 s more, hands-off; success() must still hold.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (asserted non-decreasing) and
exactly `SIM_GEN_SOLVE: SUCCESS` only if success() holds after the persistence window.
Hard exit (os._exit) with a daemon watchdog Timer backstop.

Run (forge): python -u -m simgen_tasks.push_buttons_i273.solve --headless
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
    from simgen_tasks.push_buttons_i273 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

G = scene_mod.G

# Watchdog: never leave a GPU zombie if anything below stalls.
_wd = threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                             os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    from isaaclab.utils.math import quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.shutter_trap")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    ex = torch.zeros(1, 3, device=device)
    ex[0, 0] = 1.0

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        d = scene.depth()[0]
        cov = scene.covered()[0]
        print(f"[solve] {tag:10s} d=({d[0] * 1000:5.1f},{d[1] * 1000:5.1f},"
              f"{d[2] * 1000:5.1f})mm xp={float(scene.plate_x()[0]) * 1000:+7.1f}mm "
              f"cov={int(cov[0])}{int(cov[1])}{int(cov[2])} "
              f"trap={int(scene._trapped[0, 0])}{int(scene._trapped[0, 1])}"
              f"{int(scene._trapped[0, 2])} hold={int(scene._hold[0])} "
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

    def push_off() -> None:
        for k in scene.push_w:
            scene.push_w[k].zero_()

    def press(i: int) -> bool:
        """Velocity-servo press straight down to the hard stop, then a short hold."""
        key = f"p{i}"
        reached, at = False, 0
        for _ in range(600):
            up = scene.up_w()[0]
            v_dn = float(-(scene.plungers[i].data.root_lin_vel_w[0] * up).sum())
            f = max(0.0, min(3.5, 20.0 * (0.08 - v_dn) + 1.0))
            scene.push_w[key][0] = -up * f
            step(1)
            if float(scene.depth()[0, i]) >= G.D_MAX - 0.0015:
                at += 1
                if at >= 10:
                    reached = True
                    break
            else:
                at = 0
        for _ in range(10):  # hold at the bottom (v_des 0)
            up = scene.up_w()[0]
            v_dn = float(-(scene.plungers[i].data.root_lin_vel_w[0] * up).sum())
            f = max(0.0, min(3.5, 20.0 * (0.0 - v_dn) + 1.0))
            scene.push_w[key][0] = -up * f
            step(1)
        scene.push_w[key][0] = 0.0
        return reached

    def chase(t_x: float, s: float) -> bool:
        """Velocity-servo push on the plate to console-frame x target t_x (travel toward
        the corridor centre, i.e. along -s * console-x), then brake."""
        u = -s * quat_apply(scene.console.data.root_quat_w, ex)[0]
        ok = False
        for _ in range(500):
            dist = (float(scene.plate_x()[0]) - t_x) * s  # remaining travel, positive
            if dist <= 0.003:
                ok = True
                break
            v_des = max(0.02, min(0.11, 4.0 * dist))
            v = float((scene.plate.data.root_lin_vel_w[0] * u).sum())
            # gain 40 (40*dt/m = 0.83 < 1) + 0.5 N feedforward: stall authority 1.3 N,
            # well above the ~0.55 N rail static-friction threshold
            scene.push_w["plate"][0] = u * max(-4.0, min(4.0, 40.0 * (v_des - v) + 0.5))
            step(1)
        for _ in range(20):  # brake (no feedforward)
            v = float((scene.plate.data.root_lin_vel_w[0] * u).sum())
            scene.push_w["plate"][0] = u * max(-4.0, min(4.0, 40.0 * (0.0 - v)))
            step(1)
        scene.push_w["plate"][0] = 0.0
        return ok

    # ================= PHASE 0: reset + settle + read the spawn side ========================
    env.reset(seed=args.seed)
    step(60)
    report("reset")
    phase_score("reset")  # ~0.000

    s = 1.0 if float(scene.plate_x()[0]) > 0 else -1.0
    order = [2, 1, 0] if s > 0 else [0, 1, 2]  # nearest-shutter-first
    targets = [s * 0.149, s * 0.0915, 0.0]  # staged plate-x after each trap
    print(f"[solve] plate spawned at side {s:+.0f}; button order {order}", flush=True)

    # ================= PHASES 1-3: press-release-chase, button by button ====================
    expect = (0.30, 0.55, 0.80)
    for k, (i, t_x) in enumerate(zip(order, targets)):
        if not press(i):
            report(f"press{k + 1}")
            print(f"[solve] PHASE {k + 1} FAILED: p{i} never reached the hard stop", flush=True)
            verdict(False)
        if k == 0:
            phase_score("press1")  # 0.050 (deep-press latch)
        if not chase(t_x, s):
            report(f"chase{k + 1}")
            print(f"[solve] PHASE {k + 1} FAILED: plate never reached the staging target",
                  flush=True)
            verdict(False)
        step(30)  # settle; the head rises into the plate underside
        report(f"trap{k + 1}")
        if not bool(scene.trapped_now()[0, i]):
            print(f"[solve] PHASE {k + 1} FAILED: p{i} not trapped under the plate", flush=True)
            verdict(False)
        phase_score(f"trap{k + 1}")  # expect >= expect[k]
        assert sc() >= expect[k] - 1e-6, f"trap{k + 1} score {sc():.3f} < {expect[k]}"

    # ================= PHASE 4: wait for the sustained-success counter ======================
    push_off()
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

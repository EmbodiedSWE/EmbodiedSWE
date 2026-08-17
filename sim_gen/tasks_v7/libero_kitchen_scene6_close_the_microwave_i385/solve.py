"""solve — teleport + crank solution for GenevaVaultFeederScene (…_i385).

Scene-level env (robot="null"). The ONLY teleports are free-space transports: each
staged ball is lifted from the ground and released 50 mm above the roof port (a
Franka pick-carry-release). EVERYTHING load-bearing runs through the live plant:

  LOAD    drop the ball through the yellow port collar; gravity lands it in the
          compartment at azimuth +90 deg (latches `in_lane`).
  INDEX   velocity-servo torque on the crank (`scene.crank_torque`, -Z = CW, cap
          2.5 N*m — a wrist-flick effort): exactly ONE driver revolution, tracked by
          incremental unwrap. The Geneva pin engages over driver 135..225 deg and
          indexes the paddle wheel EXACTLY +90 deg (CCW); outside that arc the wheel
          is untouched. Each index moves every loaded ball one station CCW (away from
          the crank shaft): +90 -> 180 (mid latch) -> 270, where the deck drop-hole
          is — the ball falls into the enclosed under-deck vault (vault latch).
  ORDER   k balls need interleaved loads (the port only feeds the +90 compartment):
          load, index, [load, index]..., final index. k=1 -> load,2 indexes;
          k=2 -> load A, index, load B, index (A drops), index (B drops).
  PERSIST after the last drop all buffers are zeroed; success() is current physical
          state; >= 3.5 simulated seconds hands-off before SIM_GEN_SOLVE: SUCCESS.

Prints `SIM_GEN_SCORE <score>` at phase boundaries (asserted non-decreasing).
Hard exit (os._exit) after the verdict; daemon watchdog Timer as backstop.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene6_close_the_microwave_i385.solve --headless [--seed N]
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

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.libero_kitchen_scene6_close_the_microwave_i385 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
_wd = threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                             os._exit(3)))
_wd.daemon = True
_wd.start()

# Crank servo (the honesty argument): torque cap 2.5 N*m at the 0.11 m knob orbit
# = <= 23 N tangential at the red knob — a one-hand crank. KV*dt/Iz = 1.0*(1/120)/0.03
# = 0.28, safely under the wrench-delay stability bound.
TQ_MAX = 2.5  # N*m, crank torque cap
KV = 1.0  # N*m*s/rad velocity-servo gain
KI = 1.5  # N*m/rad integral gain — escalates to the cap through the Geneva's
#          inertial peak (a pure P servo tops out at KV*OMEGA, half the cap)
OMEGA = 1.2  # rad/s cruise target in the disengaged arc
OMEGA_ENG = 0.35  # rad/s cap while the pin is near engagement (abs driver angle within
#                   ENG_HALF of 180 deg): wheel exit velocity ~ ratio*omega -> tiny coast
ENG_HALF = math.radians(70.0)  # engagement slow-zone half-width (pin engages 135..225)
K_APP = 4.0  # approach gain: |w_des| = clamp(K_APP*remaining, 0.15, cap)
END_TOL = 0.02  # rad, revolution end tolerance
PARK_TOL = math.radians(6.0)  # wheel must park within this of a station
DIR = -1.0  # crank CW (-Z): the wheel indexes +90 deg CCW, carrying the loaded
#             ball AWAY from the crank shaft (90 -> 180 -> 270 = drop hole)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.geneva_vault_feeder")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    origin = scene.env_origins[0]

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        phi = math.degrees(float(scene.driver_angle()[0]))
        wa = math.degrees(float(scene.wheel_angle()[0]))
        pos = scene._ball_pos()[0]
        pb = " ".join(
            f"b{i}=({pos[i, 0]:+.3f},{pos[i, 1]:+.3f},{pos[i, 2]:+.3f})"
            for i in range(pos.shape[0]) if bool(scene.present[0, i]))
        print(f"[solve] {tag:12s} crank={phi:7.1f}deg wheel={wa:7.1f}deg {pb} "
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

    # ---- LOAD: free-space transport to 50 mm above the port, then gravity ---------------------
    def load_ball(i: int) -> None:
        nm = c.ball_names[i]
        st = torch.zeros(1, 13, device=device)
        st[0, 0] = 0.0
        st[0, 1] = 0.5 * (c.win_y0 + c.win_y1)  # 0.275 — port centreline
        st[0, 2] = 0.47  # above the collar top (0.42), inside the port footprint
        st[0, 3] = 1.0
        st[0, 0:3] += origin
        scene.balls[nm].write_root_state_to_sim(st, torch.tensor([0], device=device))
        for _ in range(360):  # 3 s budget; ~0.5 s expected
            step(1)
            if bool(scene.in_lane()[0, i]) and bool(scene.settled()[0, i]):
                break
        if not (bool(scene.in_lane()[0, i]) and bool(scene.settled()[0, i])):
            report(f"load{i}-fail")
            print(f"[solve] LOAD FAILED: ball {i} not settled in the lane", flush=True)
            verdict(False)
        report(f"loaded-{i}")

    # ---- INDEX: exactly one crank revolution through the velocity servo -----------------------
    def index(n_th: int) -> None:
        d_prev = float(scene.driver_angle()[0])
        w_prev = float(scene.wheel_angle()[0])
        d_total = 0.0
        w_total = 0.0
        target = 2.0 * math.pi
        done = False
        acc = 0.0  # integral state (N*m)
        for it in range(3600):  # 30 s budget; ~7 s expected
            a = float(scene.driver_angle()[0])
            wa = float(scene.wheel_angle()[0])
            d_total += wrap(a - d_prev)
            w_total += wrap(wa - w_prev)
            d_prev, w_prev = a, wa
            rem = target - DIR * d_total  # signed progress along the cranking direction
            if rem <= END_TOL:
                done = True
                break
            cap = OMEGA_ENG if abs(wrap(a - math.pi)) < ENG_HALF else OMEGA
            w_des = DIR * min(cap, max(0.15, K_APP * rem))
            werr = w_des - float(scene.driver_rate()[0])
            acc = max(-TQ_MAX, min(TQ_MAX, acc + KI * werr * env.dt))
            tau = max(-TQ_MAX, min(TQ_MAX, KV * werr + acc))
            scene.crank_torque[0] = tau
            step(1)
            if it and it % 240 == 0:
                print(f"[solve]   idx{n_th} it={it} d_total={math.degrees(d_total):7.1f}deg "
                      f"rate={float(scene.driver_rate()[0]):+.2f} tau={tau:+.2f} "
                      f"wheel={math.degrees(float(scene.wheel_angle()[0])):+7.1f}deg "
                      f"w_rate={float(scene.wheel_rate()[0]):+.2f}", flush=True)
        # brake: servo to zero rate, then release
        for _ in range(240):
            if abs(float(scene.driver_rate()[0])) < 0.05:
                break
            werr = -float(scene.driver_rate()[0])
            scene.crank_torque[0] = max(-TQ_MAX, min(TQ_MAX, KV * werr))
            step(1)
        scene.crank_torque[0] = 0.0
        step(30)
        a = float(scene.driver_angle()[0])
        wa = float(scene.wheel_angle()[0])
        d_total += wrap(a - d_prev)
        w_total += wrap(wa - w_prev)
        if not done:
            report(f"index{n_th}-fail")
            print(f"[solve] INDEX {n_th} FAILED: servo timed out at {d_total:.2f} rad",
                  flush=True)
            verdict(False)
        park_err = ((wa + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0
        print(f"[solve] index {n_th}: driver {math.degrees(d_total):+.1f}deg, "
              f"wheel {math.degrees(w_total):+.1f}deg, park_err {math.degrees(park_err):+.2f}deg",
              flush=True)
        if abs(w_total + DIR * math.pi / 2.0) > math.radians(12.0) or abs(park_err) > PARK_TOL:
            report(f"index{n_th}-fail")
            print(f"[solve] INDEX {n_th} FAILED: wheel did not index {-DIR * 90:+.0f}deg cleanly",
                  flush=True)
            verdict(False)
        report(f"indexed-{n_th}")

    # ================= reset + settle ==========================================================
    env.reset(seed=args.seed)
    step(60)
    present = [i for i in range(len(c.ball_names)) if bool(scene.present[0, i])]
    print(f"[solve] seed={args.seed} balls={present} "
          f"crank0={math.degrees(float(scene.driver_angle()[0])):.1f}deg "
          f"wheel0={math.degrees(float(scene.wheel_angle()[0])):.1f}deg", flush=True)
    report("reset")
    assert float(scene.crank_torque.abs().max()) == 0.0
    assert float(scene.ball_probe.abs().max()) == 0.0
    phase_score("reset")  # ~0.000

    # ================= LOAD / INDEX program ====================================================
    n_index = 0
    for i in present:
        load_ball(i)
        phase_score(f"loaded-{i}")
        n_index += 1
        index(n_index)
        phase_score(f"indexed-{n_index}")
    n_index += 1
    index(n_index)  # the final index drops the last ball
    phase_score(f"indexed-{n_index}")

    # ================= settle into the vault ===================================================
    scene.crank_torque[0] = 0.0
    ok_settle = False
    for _ in range(60):  # up to 5 s for the last ball to land + still
        step(10)
        if bool(scene.success()[0]):
            ok_settle = True
            break
    if not ok_settle:
        report("vault-fail")
        print("[solve] FAILED: success() not reached after the final index", flush=True)
        verdict(False)
    report("vaulted")
    phase_score("vaulted")  # 1.000

    # ================= persistence (>= 3.5 simulated seconds, hands off) =======================
    assert float(scene.crank_torque.abs().max()) == 0.0, "crank torque must be zero"
    assert float(scene.ball_probe.abs().max()) == 0.0, "ball probes must be zero"
    persist_steps = int(round(3.5 / env.dt))  # 420 physics steps at 1/120 s
    step(persist_steps)
    report("final")
    phase_score("final")
    still_ok = bool(scene.success()[0]) and abs(sc() - 1.0) < 1e-3
    print(f"[solve] persistence: {persist_steps} steps ({persist_steps * env.dt:.2f} s) "
          f"hands-off, success={still_ok}", flush=True)
    verdict(still_ok)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 — never leave a GPU zombie
        print(f"[solve] CRASH: {type(e).__name__}: {e}", flush=True)
        import traceback

        traceback.print_exc()
        os._exit(1)

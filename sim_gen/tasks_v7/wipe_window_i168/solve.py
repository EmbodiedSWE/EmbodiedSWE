"""solve — demonstration solution for FrostScrapeScene (wipe_window_i168).

Scene-level env (robot="null"). NOTHING is teleported — there is no transport in
this task at all: every present chip starts racked on its ledge (the reset
state) and every centimetre of progress flows through contact dynamics. The
solver only writes the scene's `drive_f` buffer — the stand-in for a fingertip
push on the chip's side edge (TASK.md) — and post_step applies it as a force on
the chip. The load-bearing chain is entirely physical: the sideways push must
overcome ledge + glass friction, the chip's own weight must tip it off the
narrow ledge, gravity must slide it down the 75-deg glass face, and the trough
must CATCH and retain it. The solver never touches the chip below the ledge row
— the fall and the capture are pure passive physics.

  Per chip (leftmost first; the task declares no order): a velocity-cascade
  lateral push — F_y = clamp(KV * (v_des - v_y), +-F_CAP), F_CAP = 1.0 N (a
  fingertip on a 30 g chip — comfortable authority; KV*dt/m ~ 0.7 < 1 against
  the one-substep wrench delay). Release the instant the chip starts dropping
  (readback: centre 3 cm below its racked start) or is fully clear of the
  ledge; if the chip hangs up, retry with a slightly faster push (escalate the
  commanded rate, never the force cap). Then WAIT: the chip slides down the
  glass and settles in the trough on its own; the contained latch matures only
  through the scene's slow-gate streak.

Prints `SIM_GEN_SCORE <v>` at phase boundaries (asserted non-decreasing:
latched credit must not evaporate). After success() first holds, keeps
simulating >= 3.5 more simulated seconds with the drive buffer zero (asserted);
only if success() still holds prints exactly `SIM_GEN_SOLVE: SUCCESS`. Hard
exit (os._exit) after the verdict, watchdog Timer as backstop — Kit teardown
hangs otherwise.

Run (forge): python -u -m simgen_tasks.wipe_window_i168.solve --headless [--seed N]
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
    from simgen_tasks.wipe_window_i168 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()

# Fingertip-scale push (chip mass 30 g, dt 1/120: KV*dt/m ~ 0.7 < 1 — the cascade
# stays discretely monotone against the one-substep wrench delay).
KV = 2.5  # N*s/m inner rate gain
V_DES0 = 0.12  # m/s commanded lateral rate (starting value)
V_DES_MAX = 0.25  # escalation ceiling (escalate the RATE on stalls, never the cap)
F_CAP = 1.0  # N hard cap — fingertip authority
DROP_RELEASE = 0.03  # m: chip centre this far below its rack = it is tipping, let go
CLEAR_DY = 0.055  # m: fully clear of the 32 mm ledge under a 70 mm chip
FALL_CONFIRM = 0.15  # m: drop confirming the chip really left the row


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.frost_scrape")().build(num_envs=1, device=device)
    scene = env.scene
    no_action = torch.empty(0, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        p = scene.chip_pos()[0]
        pres = scene.present[0]
        drops = [f"{float(scene.start_z[0, i] - p[i, 2]):+.3f}" if bool(pres[i]) else "  --  "
                 for i in range(4)]
        cont = [int(bool(scene.contained_now()[0, i])) if bool(pres[i]) else "-"
                for i in range(4)]
        print(f"[solve] {tag:12s} drops={drops} contained={cont} "
              f"latch(d)={scene.dislodge_latch[0].tolist()} "
              f"latch(c)={scene.contained_latch[0].tolist()} "
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
        threading.Timer(10.0, lambda: os._exit(code)).start()
        try:
            env.close()
            app.close()
        except Exception:  # noqa: BLE001
            pass
        os._exit(code)

    def push_off(i: int) -> bool:
        """Push chip i in -y with the velocity cascade until it tips off its ledge
        (readback drop) or is fully clear; then wait for the passive fall. Retries
        with an escalated commanded rate if the chip hangs up."""
        v_des = V_DES0
        for attempt in range(3):
            y0 = float(scene.chip_pos()[0, i, 1])
            z_rack = float(scene.start_z[0, i])
            released = False
            last_y, last_ck = y0, 0
            for k in range(900):
                p = scene.chip_pos()[0, i]
                v = scene.chips[i].data.root_lin_vel_w[0]
                drop = z_rack - float(p[2])
                dy = float(p[1]) - y0
                if drop > DROP_RELEASE or dy < -CLEAR_DY:
                    released = True
                    break
                f = KV * (-v_des - float(v[1]))
                scene.drive_f[0, i, 1] = max(-F_CAP, min(F_CAP, f))
                step(1)
                if k - last_ck >= 180:  # stall watch: escalate the RATE, not the cap
                    if abs(float(p[1]) - last_y) < 0.003 and v_des < V_DES_MAX:
                        v_des = min(V_DES_MAX, v_des * 1.3)
                        print(f"[solve] chip{i} push stalled -> v_des={v_des:.3f}", flush=True)
                    last_y, last_ck = float(p[1]), k
            scene.drive_f[0, i, :] = 0.0
            # passive fall: hands off, wait for the chip to leave the row
            for _ in range(48):
                if float(z_rack - scene.chip_pos()[0, i, 2]) > FALL_CONFIRM:
                    break
                step(10)
            if float(z_rack - scene.chip_pos()[0, i, 2]) > FALL_CONFIRM:
                return True
            print(f"[solve] chip{i} attempt {attempt} did not come off "
                  f"(released={released}) — retrying faster", flush=True)
            v_des = min(V_DES_MAX, v_des * 1.4)
        return False

    def wait_capture(i: int) -> bool:
        """Wait (hands off) until chip i is contained and slow, then let the slow-gate
        streak mature."""
        for _ in range(120):
            if bool(scene.contained_now()[0, i]) \
                    and float(scene.chip_speed()[0, i]) < 0.10:
                break
            step(10)
        step(60)  # streak matures; chip settles in the trough
        return bool(scene.contained_latch[0, i])

    # ================= reset + settle ==========================================================
    env.reset(seed=args.seed)
    step(60)
    pres = scene.present[0]
    order = sorted([i for i in range(4) if bool(pres[i])],
                   key=lambda i: float(scene.slot_y[0, i]))
    print(f"[solve] seed={args.seed} present={pres.tolist()} "
          f"slots_y={[round(float(scene.slot_y[0, i]), 3) for i in range(4)]} "
          f"row_s={float(scene.row_s[0]):.3f} order={order}", flush=True)
    report("reset")
    assert float(scene.drive_f.abs().max()) == 0.0
    assert not bool(scene.dislodge_latch[0].any()), "reset must not pre-latch"
    phase_score("reset")  # ~0.000

    # ================= one chip at a time: push off, passive fall, trough capture ==============
    for n_done, i in enumerate(order):
        if not push_off(i):
            report(f"chip{i}-stuck")
            print(f"[solve] PHASE {n_done + 1} FAILED: chip{i} never left its ledge",
                  flush=True)
            verdict(False)
        if not wait_capture(i):
            report(f"chip{i}-lost")
            print(f"[solve] PHASE {n_done + 1} FAILED: chip{i} was not captured by the "
                  f"trough", flush=True)
            verdict(False)
        report(f"chip{i}-in")
        phase_score(f"chip{i}")

    # ================= all chips in: success must hold =========================================
    ok = False
    for _ in range(48):  # up to 4 s for everything to settle
        if bool(scene.success()[0]):
            ok = True
            break
        step(10)
    if not ok:
        report("settle-fail")
        print("[solve] FINAL FAILED: success() not reached after all captures", flush=True)
        verdict(False)
    report("all-in")
    phase_score("all-in")  # 1.000

    # ================= persistence (>= 3.5 simulated seconds, hands off) =======================
    assert float(scene.drive_f.abs().max()) == 0.0, "drive must be zero for persistence"
    persist_steps = int(round(3.5 / env.dt))  # 420 physics steps at 1/120 s
    step(persist_steps)
    report("final")
    phase_score("final")
    still_ok = bool(scene.success()[0]) and abs(sc() - 1.0) < 1e-3
    print(f"[solve] persistence: {persist_steps} steps ({persist_steps * env.dt:.2f} s) "
          f"hands-off, success={still_ok}", flush=True)
    verdict(still_ok)


if __name__ == "__main__":
    main()

"""solve — solution for WardedSpindleScene (beat_the_buzz_i263).

Scene-level env (robot="null"). NO teleports are needed at all: the sleeve is
captive on the post, so the entire solution is force/torque control through the
scene's drive buffers (the stand-in for the Franka's grasp on the yellow fin,
TASK.md), applied by scene.post_step, which also latches the rubric every step.

  For each ward (upper th1, then lower th2 — the order is forced by topology):
    ALIGN:   yaw PD torque about the post axis rotates the fin toward the notch
             while the sleeve rests on the shelf by gravity. The gain escalates
             if the servo stalls against friction (escalate GAIN, not the cap).
             The fin drops through the notch as soon as it is aligned — detected
             by the z readback, not assumed.
    DESCEND: vertical velocity-servo force (fz = m*g + KV*(v_des - vz),
             v_des = -0.12 m/s) lowers the sleeve gently while a soft yaw PD
             holds the fin centred in the notch. If the descent wedges, recover:
             lift 25 mm, re-align, retry (escalating alignment gain).
    SETTLE:  drives zeroed, streak-gated settle (velocity thresholds alone fire
             vacuously at turning points).

  After the second ward the sleeve lands seated on the plinth; drives are zeroed,
  success() is awaited, then >= 3.5 more simulated seconds hands-off (drives
  asserted zero) before `SIM_GEN_SOLVE: SUCCESS`.

Servo sizing (post_step wrenches act one step late -> K*dt/m well under 1):
m = 0.15 kg, KV = 4.0 -> 0.22 at dt = 1/120. Yaw: I_z ~ 2e-4 kg m^2, Kd = 0.010
-> Kd*dt/I ~ 0.4; Kp 0.03 escalating x1.5 to 0.12, torque cap 0.06 N m.

Prints `SIM_GEN_SCORE <v>` at phase boundaries (asserted non-decreasing).
Hard exit (os._exit) after the verdict, watchdog Timer as backstop.

Run (forge): python -u -m simgen_tasks.beat_the_buzz_i263.solve --headless [--seed N]
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
    from simgen_tasks.beat_the_buzz_i263 import scene as scene_mod
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()

wrap_pi = scene_mod.wrap_pi

G = 9.81
KV = 4.0  # N*s/m vertical velocity-servo gain (KV*dt/m = 0.22 — delay-stable)
V_DOWN = -0.12  # m/s controlled descent rate
F_MIN, F_MAX = -2.0, 4.0  # N clamp on the vertical drive (m*g = 1.47 N)
KP0, KP_MAX = 0.03, 0.12  # N*m/rad yaw PD, escalating on stall
KD = 0.010  # N*m*s/rad (KD*dt/I ~ 0.4 — delay-stable)
T_CAP = 0.06  # N*m torque clamp
ALIGN_TOL = math.radians(3.0)
DROP_DZ = 0.012  # m below the rest height = "fell through"


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.warded_spindle")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def z_now() -> float:
        return float(scene.sleeve_z()[0])

    def yaw_now() -> float:
        return float(scene.tab_yaw()[0])

    def wz_now() -> float:
        return float(scene.sleeve.data.root_ang_vel_w[0, 2])

    def vz_now() -> float:
        return float(scene.sleeve.data.root_lin_vel_w[0, 2])

    def report(tag: str) -> None:
        print(f"[solve] {tag:12s} z={z_now():.3f} yaw={math.degrees(yaw_now()):+7.1f}deg "
              f"thr={int(scene.threaded()[0])} up={int(scene.upright()[0])} "
              f"p1={float(scene.p1_latch[0]):.0f} p2={float(scene.p2_latch[0]):.0f} "
              f"seat={float(scene.seat_latch[0]):.0f} score={sc():.3f} "
              f"success={bool(scene.success()[0])}", flush=True)

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

    def yaw_pd(target: float, kp: float) -> None:
        """One-step yaw PD torque toward `target` (shortest way), clamped."""
        err = float(wrap_pi(torch.tensor(target - yaw_now())))
        tau = kp * err - KD * wz_now()
        scene.drive_t[0, 0, 2] = max(-T_CAP, min(T_CAP, tau))

    def hold_z(v_des: float) -> None:
        """One-step vertical velocity servo (gravity feedforward + KV)."""
        f = c.sleeve_mass * G + KV * (v_des - vz_now())
        scene.drive_f[0, 0, 2] = max(F_MIN, min(F_MAX, f))

    def align(target: float, z_rest: float, kp0: float,
              budget: int = 900) -> tuple[str, float]:
        """Rotate the fin toward `target` while resting at `z_rest`. Returns
        ('dropped'|'aligned'|'stuck', final kp). The gain escalates on a plain
        timer — the fin corner-hooks on the notch edge and rattles (|wz| never
        low), so a velocity-gated stall counter never fires; a hard grind does."""
        kp = kp0
        aligned_streak = 0
        esc = 0
        for _ in range(budget):
            if z_now() < z_rest - DROP_DZ:
                scene.drive_t[0, 0, 2] = 0.0
                return "dropped", kp
            err = float(wrap_pi(torch.tensor(target - yaw_now())))
            if abs(err) < ALIGN_TOL and abs(wz_now()) < 0.2:
                aligned_streak += 1
                if aligned_streak >= 240:  # 2 s aligned yet still up: let descend() push
                    scene.drive_t[0, 0, 2] = 0.0
                    return "aligned", kp
            else:
                aligned_streak = 0
            esc += 1
            if esc >= 240 and kp < KP_MAX:  # 2 s without a drop -> escalate the GAIN
                kp = min(kp * 1.5, KP_MAX)
                esc = 0
                print(f"[solve] align escalate -> kp={kp:.3f} "
                      f"(err={math.degrees(err):+.1f}deg)", flush=True)
            yaw_pd(target, kp)
            step(1)
        scene.drive_t[0, 0, 2] = 0.0
        return "stuck", kp

    def descend(target: float, z_stop: float, budget: int = 1200) -> bool:
        """Velocity-servo descent to env-local height z_stop with a soft yaw hold
        at `target`. Returns True when resting at z_stop."""
        stall = 0
        for _ in range(budget):
            z = z_now()
            if abs(z - z_stop) < 0.005 and abs(vz_now()) < 0.05:
                scene.drive_f[0, 0] = 0.0
                scene.drive_t[0, 0] = 0.0
                return True
            hold_z(V_DOWN if z > z_stop + 0.004 else 0.0)
            yaw_pd(target, 0.02)
            # wedge detection: not moving, still well above the stop
            stall = stall + 1 if (abs(vz_now()) < 0.02 and z > z_stop + 0.015) else 0
            if stall >= 240:
                scene.drive_f[0, 0] = 0.0
                scene.drive_t[0, 0] = 0.0
                return False
            step(1)
        scene.drive_f[0, 0] = 0.0
        scene.drive_t[0, 0] = 0.0
        return False

    def settle(budget: int = 600, streak_need: int = 30) -> bool:
        """Drives zero; streak-gated settle."""
        scene.drive_f[0, 0] = 0.0
        scene.drive_t[0, 0] = 0.0
        streak = 0
        for _ in range(budget):
            step(1)
            v = float(scene.sleeve.data.root_lin_vel_w[0].norm())
            w = float(scene.sleeve.data.root_ang_vel_w[0].norm())
            streak = streak + 1 if (v < 0.03 and w < 0.5) else 0
            if streak >= streak_need:
                return True
        return False

    def pass_ward(target: float, z_rest_above: float, z_rest_below: float,
                  tag: str) -> bool:
        """Align to `target` on the shelf at z_rest_above, drop + controlled descent
        to z_rest_below, settle. Retries with a lift-and-re-align on wedge; the
        escalated alignment gain carries ACROSS attempts (the jam does not reset)."""
        kp = KP0
        for attempt in range(4):
            res, kp = align(target, z_rest_above, kp)
            report(f"{tag}-align{attempt}")
            if res == "stuck":
                print(f"[solve] {tag}: align stuck (attempt {attempt})", flush=True)
                continue
            # 'dropped' or 'aligned': drive the descent either way
            if descend(target, z_rest_below):
                report(f"{tag}-down")
                return settle()
            # wedged: lift 25 mm back above the shelf and re-align
            print(f"[solve] {tag}: descent wedged (attempt {attempt}) — lift + retry",
                  flush=True)
            lift_to = z_rest_above + 0.004
            for _ in range(600):
                if z_now() >= lift_to - 0.003:
                    break
                hold_z(0.10)
                yaw_pd(target, 0.02)
                step(1)
            scene.drive_f[0, 0] = 0.0
            scene.drive_t[0, 0] = 0.0
            settle(budget=240)
        return False

    # ================= reset + settle ==========================================================
    torch.manual_seed(args.seed)
    env.reset()
    step(60)
    th1 = float(scene.th1[0])
    th2 = float(scene.th2[0])
    ths = float(scene.th_s0[0])
    try:
        m_s = float(scene.sleeve.root_physx_view.get_masses().reshape(-1)[0])
        print(f"[solve] mass readback: sleeve={m_s:.3f}kg (cfg {c.sleeve_mass})", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[solve] mass readback unavailable ({exc!r})", flush=True)
    print(f"[solve] seed={args.seed} th1={math.degrees(th1):+.1f}deg "
          f"th2={math.degrees(th2):+.1f}deg fin0={math.degrees(ths):+.1f}deg "
          f"d(fin,th1)={math.degrees(float(wrap_pi(torch.tensor(ths - th1)))):+.1f}deg",
          flush=True)
    report("reset")
    assert float(scene.drive_f.abs().max()) == 0.0
    assert float(scene.drive_t.abs().max()) == 0.0
    if not bool(scene.threaded()[0]):
        print("[solve] FAILED: sleeve not threaded after reset", flush=True)
        verdict(False)
    phase_score("reset")  # ~0.000

    # ================= PHASE 1: through the upper (red) ward ===================================
    if not pass_ward(th1, c.z_rest_w1, c.z_rest_w2, "ward1"):
        print("[solve] PHASE 1 FAILED: could not pass the upper ward", flush=True)
        verdict(False)
    report("ward1-done")
    if not (bool(scene.threaded()[0]) and float(scene.p1_latch[0]) == 1.0):
        print("[solve] PHASE 1 FAILED: p1 latch / threaded readback", flush=True)
        verdict(False)
    phase_score("phase1")  # 0.300

    # ================= PHASE 2: through the lower (orange) ward + seat =========================
    if not pass_ward(th2, c.z_rest_w2, c.z_seat, "ward2"):
        print("[solve] PHASE 2 FAILED: could not pass the lower ward / seat", flush=True)
        verdict(False)
    ok = False
    for _ in range(48):  # up to 4 s to co-settle into success
        if bool(scene.success()[0]):
            ok = True
            break
        step(10)
    report("seated")
    if not ok:
        print("[solve] PHASE 2 FAILED: success() not reached after seating", flush=True)
        verdict(False)
    phase_score("phase2")  # 1.000

    # ================= PHASE 3: persistence (>= 3.5 simulated seconds, hands off) ==============
    assert float(scene.drive_f.abs().max()) == 0.0, "drives must be zero for persistence"
    assert float(scene.drive_t.abs().max()) == 0.0, "drives must be zero for persistence"
    persist_steps = int(round(3.5 / env.dt))
    step(persist_steps)
    report("final")
    phase_score("final")
    still_ok = bool(scene.success()[0]) and abs(sc() - 1.0) < 1e-3
    print(f"[solve] persistence: {persist_steps} steps ({persist_steps * env.dt:.2f} s) "
          f"hands-off, success={still_ok}", flush=True)
    verdict(still_ok)


if __name__ == "__main__":
    main()

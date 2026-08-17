"""solve — force-driven solution for TrolleyShuntScene (reach_and_drag_i427).

Scene-level env (robot="null"). NO teleports at all: the trolley is DRIVEN the whole
way through the scene's world-frame deck force/torque buffers (frame-encoded plant-side
in scene.post_step), exactly the push-and-steer interaction a robot hand would apply.

  PHASE 0  reset + settle; read the goal side s from scene.flip (the readback the real
           agent gets by looking at the green/red roof tiles) and mirror the waypoints.
  PHASE 1  drive EAST down the start lane past the divider tip (latches g1, 0.20).
  PHASE 2  U-turn in the open east zone and drive WEST up the far lane to the shed
           approach (latches g2, 0.45).
  PHASE 3  line up east of the ramp toe and roll in over the sill (ramp feed-forward
           3.5 N while on the approach ramp), brake DEAD inside the park band (g3, 0.70).
  PHASE 4  all drive off, wait for the sustained-success counter (1.00).
  PHASE 5  keep simulating >= 3 s more, hands-off; success() must still hold.

Controller (audited): heading P-servo tau = k_th*err - k_w*w_z (k_w*dt/I = 0.44 < 1,
zeta ~ 1.9), forward velocity servo F = k_v*(v_des - v_fwd) along the current heading
(k_v*dt/m = 0.13 < 1), v_des scaled by max(0, cos(err)) so the trolley pivots first.
Caps: 8 N / 1.0 N*m (the scene's plant caps).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (asserted non-decreasing) and
exactly `SIM_GEN_SOLVE: SUCCESS` only if success() holds after the persistence window.
Hard exit (os._exit) with a daemon watchdog Timer backstop.

Run (forge): python -u -m simgen_tasks.reach_and_drag_i427.solve --headless
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
    from simgen_tasks.reach_and_drag_i427 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

G = scene_mod.G

# Watchdog: never leave a GPU zombie if anything below stalls.
_wd = threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                             os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.trolley_shunt")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        rel = scene.deck_rel()[0]
        h = scene.heading_rel()[0]
        v = float(scene.deck.data.root_lin_vel_w[0].norm())
        print(f"[solve] {tag:8s} p=({rel[0]:+.3f},{rel[1]:+.3f},{rel[2]:.3f}) "
              f"th={math.degrees(math.atan2(float(h[1]), float(h[0]))):+7.1f} "
              f"v={v:.3f} g={int(scene._g1[0])}{int(scene._g2[0])}{int(scene._g3[0])} "
              f"hold={int(scene._hold[0])} score={sc():.3f} "
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
        t = threading.Timer(10.0, lambda: os._exit(code))
        t.daemon = True
        t.start()
        try:
            env.close()
            app.close()
        except Exception:  # noqa: BLE001
            pass
        os._exit(code)

    # ----- differential-drive waypoint controller (yard frame) --------------------------------
    K_V, K_TH, K_W = 30.0, 1.2, 0.35
    ez = torch.tensor([0.0, 0.0, 1.0], device=device)

    def control(wp, v_max: float, ramp_ff: bool) -> None:
        """One substep of drive toward waypoint wp = (x, y) in the yard frame."""
        rel = scene.deck_rel()[0]
        h = scene.heading_rel()[0]
        th = math.atan2(float(h[1]), float(h[0]))
        dx, dy = float(wp[0] - rel[0]), float(wp[1] - rel[1])
        err = (math.atan2(dy, dx) - th + math.pi) % (2 * math.pi) - math.pi
        w_z = float(scene.deck.data.root_ang_vel_w[0, 2])
        tau = K_TH * err - K_W * w_z
        scene.push_tau_w[0] = ez * max(-c.tau_cap, min(c.tau_cap, tau))
        head_w = scene._qa(scene.deck.data.root_quat_w, scene._ex)[0]
        v_fwd = float((scene.deck.data.root_lin_vel_w[0] * head_w).sum())
        v_des = v_max * max(0.0, math.cos(err)) if abs(err) < math.pi / 2 else 0.0
        f = K_V * (v_des - v_fwd)
        if ramp_ff and -0.25 < float(rel[0]) < -0.045:
            f += 3.5  # sill approach-ramp feed-forward (m*g*sin(9.5deg) = 3.2 N)
        scene.push_f_w[0] = head_w * max(-c.f_cap, min(c.f_cap, f))

    def drive_to(wp, v_max: float, tol: float, max_steps: int, ramp_ff: bool = False,
                 stop_x: float | None = None) -> bool:
        """Drive to wp; done when within tol (or, if stop_x is set, once deck x <= stop_x)."""
        for _ in range(max_steps):
            control(wp, v_max, ramp_ff)
            step(1)
            rel = scene.deck_rel()[0]
            if stop_x is not None:
                if float(rel[0]) <= stop_x:
                    return True
            elif (float(wp[0] - rel[0]) ** 2 + float(wp[1] - rel[1]) ** 2) < tol * tol:
                return True
        return False

    def brake(max_steps: int = 720) -> None:
        """Active braking to a DEAD stop (wheels roll nearly friction-free: 7 mm/s of
        residual coast walked the trolley out of the park band over the persistence
        window). Hold the velocity-kill servo until v < 2 mm/s for 30 straight steps."""
        scene.push_tau_w[0] = 0.0
        streak = 0
        for _ in range(max_steps):
            v_w = scene.deck.data.root_lin_vel_w[0]
            streak = streak + 1 if float(v_w.norm()) < 0.002 else 0
            if streak >= 30:
                break
            f = -K_V * v_w
            f[2] = 0.0
            n = float(f.norm())
            if n > c.f_cap:
                f = f * (c.f_cap / n)
            scene.push_f_w[0] = f
            step(1)
        scene.push_f_w[0] = 0.0

    def hands_off() -> None:
        scene.push_f_w[0] = 0.0
        scene.push_tau_w[0] = 0.0

    # ================= PHASE 0: reset + settle + read the goal side ==========================
    env.reset(seed=args.seed)
    step(60)
    report("reset")
    phase_score("reset")  # 0.000
    s = float(scene.flip[0])  # goal lane sign (readback of the green roof tile side)
    print(f"[solve] goal side s={s:+.0f} (green shed at y={s * G.SHED_YC:+.3f}, "
          f"start lane y={-s * 0.135:+.3f})", flush=True)

    # ================= PHASE 1: east down the start lane (g1) ================================
    ok = drive_to((0.22, -0.135 * s), 0.15, 0.05, 2400)
    report("east")
    if not (ok and bool(scene._g1[0])):
        print("[solve] PHASE 1 FAILED: never reached the east zone", flush=True)
        verdict(False)
    phase_score("east")  # 0.20

    # ================= PHASE 2: U-turn, west up the far lane (g2) ============================
    ok = (drive_to((0.26, 0.135 * s), 0.15, 0.06, 2400)
          and drive_to((0.05, 0.140 * s), 0.15, 0.05, 2400)
          and drive_to((-0.02, 0.145 * s), 0.12, 0.04, 2400))
    report("farlane")
    if not (ok and bool(scene._g2[0])):
        print("[solve] PHASE 2 FAILED: never reached the far lane approach", flush=True)
        verdict(False)
    phase_score("farlane")  # 0.45

    # ================= PHASE 3: line up, roll in over the sill, brake (g3) ===================
    # line up EAST of the ramp toe (-0.108): a slow stop on the incline would roll back
    ok = drive_to((-0.040, 0.150 * s), 0.08, 0.025, 2400)
    report("mouth")
    if not ok:
        print("[solve] PHASE 3 FAILED: could not line up on the shed mouth", flush=True)
        verdict(False)
    ok = drive_to((-0.34, 0.150 * s), 0.10, 0.02, 2400, ramp_ff=True, stop_x=-0.290)
    brake()
    report("parked")
    if not (ok and bool(scene._g3[0])):
        print("[solve] PHASE 3 FAILED: never crossed the sill into the green shed",
              flush=True)
        verdict(False)
    phase_score("parked")  # 0.70

    # ================= PHASE 4: hands off, wait for the sustained success ====================
    hands_off()
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

    # ================= PHASE 5: persistence (>= 3 simulated seconds, hands off) ==============
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

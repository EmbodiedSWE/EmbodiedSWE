"""Applied-wrench solution for MarbleRouterScene (sim_gen task `scene_b_i305`)
— the task's legitimacy certificate.

NO teleports are used at all: every interaction is an applied force or torque
on a graspable handle, exactly what a gripper does.

  P0  settle + perception: read the target bin t, its half, and the sampled
      fin start angles back from the LIVE state (mass readback via
      get_masses() on ball, fins and gate guards the density-mass trap);
      assert both path fins rest at their WRONG stops, the gate is shut, the
      marble waits behind it; score ~ 0.
  P1  write the routing program: PD torque about each path fin's own board
      normal (body z) throws it across to its CORRECT stop; gravity then
      presses it into the stop hands-off.  The `routed` latch fires (0.20)
      while the gate is still shut and the marble still waits.
  P2  commit: PD force along the gate's prismatic axis (body y) pulls it
      open past the release threshold (`released`, 0.40 cumulative); the
      marble rolls out; force off once it is well clear of the gate slot.
  P3  hands off: the marble rolls junction A (`branched`, 0.55) then
      junction B and comes to rest against the end wall INSIDE the marked
      bin; success() turns True on the live state (score 1.0) and holds
      through a 3.3 s hands-off persistence window before
      `SIM_GEN_SOLVE: SUCCESS`.

The whole episode repeats on a second seed (fresh reset; the SIM_GEN_SCORE
stream is printed for the first seed only and is non-decreasing).

Run (forge): python -u -m simgen_tasks.scene_b_i305.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.marble_router")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    zero = torch.zeros(n, 1, 3, device=device)
    psi_a, psi_b = c.psi_a(), c.psi_b()

    print("[solve] " + "=" * 70, flush=True)
    print(scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    # board normal in world (pitch about +y): n = (sin p, 0, cos p)
    import math

    p = math.radians(c.pitch_deg)
    n_w = torch.tensor([math.sin(p), 0.0, math.cos(p)], device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        b = scene.ball_b()[0]
        print(f"[solve] {tag:12s} | thA={float(scene.fin_angle(scene.fin_a)[0]):+.3f} "
              f"thBL={float(scene.fin_angle(scene.fin_bl)[0]):+.3f} "
              f"thBR={float(scene.fin_angle(scene.fin_br)[0]):+.3f} "
              f"gate={float(scene.gate_y()[0]):+.4f} "
              f"ball=({float(b[0]):+.3f},{float(b[1]):+.3f},{float(b[2]):+.3f}) "
              f"fins_ok={bool(scene.fins_correct()[0])} "
              f"routed={bool(scene._routed_l[0])} released={bool(scene._released_l[0])} "
              f"branched={bool(scene._branched_l[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def throw_fin(fin, tgt: float, label: str) -> None:
        """PD torque about the fin's own hinge axis (body z = board normal)
        until it rests at the target stop; gravity holds it there after."""
        kp, kd, clamp = 0.02, 0.004, 0.008
        for i in range(400):
            th = scene.fin_angle(fin)
            w = (fin.data.root_ang_vel_w * n_w).sum(dim=-1)
            tau = (kp * (torch.full_like(th, tgt) - th) - kd * w).clamp(-clamp, clamp)
            t_b = torch.zeros(n, 1, 3, device=device)
            t_b[:, 0, 2] = tau
            fin.set_external_force_and_torque(zero, t_b)
            env.step(no_action)
            if i > 10 and abs(float(th[0]) - tgt) < 0.030 and abs(float(w[0])) < 0.4:
                break
        fin.set_external_force_and_torque(zero, zero)
        step(30)  # gravity seats it on the stop
        th = float(scene.fin_angle(fin)[0])
        print(f"[solve] fin {label}: target {tgt:+.3f} -> settled {th:+.3f}", flush=True)
        assert abs(th - tgt) < c.fin_tol, f"fin {label} failed to hold its stop: {th:+.3f}"

    def pull_gate() -> None:
        """PD force along the prismatic axis (body y) to the open end; hold
        while the marble rolls out, then let go (gravity-neutral, damped)."""
        kp, kd, clamp, tgt = 30.0, 3.0, 4.0, 0.095
        for i in range(600):
            y = scene.gate_y()
            v = scene.gate.data.root_lin_vel_w[:, 1]
            fy = (kp * (torch.full_like(y, tgt) - y) - kd * v).clamp(-clamp, clamp)
            f_b = torch.zeros(n, 1, 3, device=device)
            f_b[:, 0, 1] = fy
            scene.gate.set_external_force_and_torque(f_b, zero)
            env.step(no_action)
            if i > 20 and float(scene.ball_b()[0, 0]) > 0.30:
                break
        scene.gate.set_external_force_and_torque(zero, zero)
        print(f"[solve] gate released at y={float(scene.gate_y()[0]):+.4f}, "
              f"ball_x={float(scene.ball_b()[0, 0]):+.3f}", flush=True)

    def episode(seed: int, announce: bool) -> bool:
        env.reset(seed=seed)
        s_prev = 0.0

        def print_score(tag: str) -> float:
            nonlocal s_prev
            s = float(scene.score()[0])
            print(f"[solve] phase boundary: {tag}", flush=True)
            if announce:
                print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
            assert s >= s_prev - 1e-6, f"score decreased: {s_prev:.4f} -> {s:.4f}"
            s_prev = s
            return s

        # ---------------- phase 0: settle, perception, mass audit -----------------------
        step(120)
        t = int(float(scene.layout[0, 0]))
        half = float(scene.layout[0, 1])
        sgn_b = 1.0 if t in (0, 2) else -1.0
        print(f"[solve] layout readback (seed {seed}): t={t} half={half:+.0f} "
              f"sgn_b={sgn_b:+.0f} marker_y={float(scene.layout[0, 7]):+.4f}", flush=True)
        # mass readback guards the density-mass trap on every custom-spawned body
        for name, body, want in (("ball", scene.ball, c.ball_mass),
                                 ("fin_a", scene.fin_a, c.fin_mass),
                                 ("fin_bl", scene.fin_bl, c.fin_mass),
                                 ("fin_br", scene.fin_br, c.fin_mass),
                                 ("gate", scene.gate, c.gate_mass)):
            mm = float(body.root_physx_view.get_masses().sum())
            assert abs(mm - want) < 0.005, f"{name} mass readback {mm:.4f} vs {want}"
        report("reset")
        assert bool(scene._finite()[0]), "NaN/inf after settle"
        assert bool(scene.ball_waiting()[0]), "marble must wait behind the gate"
        assert float(scene.gate_y()[0]) < c.gate_shut_y, "gate must start shut"
        # both path fins rest at their WRONG stops (readback of the sampled starts)
        fin_b = scene.fin_bl if half > 0 else scene.fin_br
        th_a = float(scene.fin_angle(scene.fin_a)[0])
        th_b = float(scene.fin_angle(fin_b)[0])
        assert abs(th_a - (-half * psi_a)) < c.fin_tol, f"fin A must start wrong: {th_a:+.3f}"
        assert abs(th_b - (-sgn_b * psi_b)) < c.fin_tol, f"path fin B must start wrong: {th_b:+.3f}"
        assert not bool(scene.fins_correct()[0]) and not bool(scene.success()[0])
        s = print_score("P0 reset+settle (marble waits, path fins wrong)")
        assert s <= 0.05, f"baseline score should be ~0, got {s}"

        # ---------------- phase 1: write the routing program (torque servos) ------------
        throw_fin(scene.fin_a, half * psi_a, "A")
        throw_fin(fin_b, sgn_b * psi_b, "B-path")
        report("fins set")
        assert bool(scene.fins_correct()[0]), "both path fins must sit at their correct stops"
        assert bool(scene.ball_waiting()[0]) and float(scene.gate_y()[0]) < c.gate_shut_y
        s = print_score("P1 routing program written (gate still shut)")
        assert s >= c.w_routed - 1e-6, f"P1 score {s:.3f} below routed credit"

        # ---------------- phase 2: pull the gate, marble rolls out ----------------------
        pull_gate()
        report("gate open")
        assert bool(scene._released_l[0]), "released latch must fire"
        s = print_score("P2 gate pulled open, marble committed")
        assert s >= c.w_routed + c.w_released - 1e-6, f"P2 score {s:.3f}"

        # ---------------- phase 3: hands off — roll the tree, land, success -------------
        won_at = None
        for i in range(1500):
            env.step(no_action)
            if bool(scene.success()[0]):
                won_at = i
                break
        report("hands-off")
        b = scene.ball_b()[0]
        assert won_at is not None and bool(scene.success()[0]), \
            f"success must turn True hands-off (ball {b.tolist()})"
        assert bool(scene._branched_l[0]), "branched latch must have fired en route"
        s = print_score(f"P3 marble at rest in bin {t} ({won_at} steps)")
        assert s >= 1.0 - 1e-6, "success must score 1.0"

        # ---------------- persistence (>= 3 simulated seconds, hands-off) ---------------
        hold = True
        for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
            step(40)
            hold = hold and bool(scene.success()[0])
        report("persist")
        s_final = print_score("P-final persistence 3.3 s")
        ok = hold and bool(scene.success()[0]) and s_final >= 1.0 - 1e-6
        print(f"[solve] episode seed={seed}: {'OK' if ok else 'FAILED'}", flush=True)
        return ok

    ok = episode(args.seed, announce=True)
    ok = episode(args.seed + 1, announce=False) and ok

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
    except Exception as e:  # noqa: BLE001 - fast fail beats a 20-min watchdog hang
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)

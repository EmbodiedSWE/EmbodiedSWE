"""Teleport solution for ButterHatchScene (sim_gen task
`libero_kitchen_scene10_..._i380`) — the task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. CLOSE the drawer (contact/force dynamics — the order-creating act): a gentle
   velocity-servoed external push (F_x = kp*(v_des - vx), kp = 8 N·s/m, v_des =
   +0.06 m/s; K*dt/m = 0.13 << 1, well inside the one-substep wrench-delay bound)
   drives the drawer along its real prismatic slide until it seats against the q = 0
   hard stop. This stands in for the Franka pressing the protruding dark knob bar
   straight in — a planar push on a real sliding mechanism, no pose writes to the
   drawer EVER. The drawer's closure is what ALIGNS the countertop hatch with the
   drawer cavity; before it, any deposit falls past the drawer into the reject cellar.
2. DEPOSIT the butter (transport teleport + gravity contact): one pose write carries
   the yellow butter from its floor spawn slot to a hover ABOVE the orange collar
   mouth — deliberately above the chute scoring band (asserted: nothing latches at the
   write instant). It then FALLS under gravity: through the collar, through the
   100 mm hatch hole in the countertop, and lands inside the closed drawer cavity
   through real contact. The chute-transit latch and the inside latch both fire on
   live pose readbacks during/after this purely ballistic delivery.
3. Success is judged on settled poses: butter at rest inside the cavity (drawer-frame
   box), drawer fully closed and still, white paraffin outside.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still holds.

The intended single-Franka-arm strategy for the same plan (push the knob bar home,
then top-down pinch of the butter's 32 mm faces and release above the collar) lives in
TASK.md as the embodiment argument.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it_i380.solve --headless [--seed N]
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

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.butter_hatch")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def qd() -> float:
        return float(scene.q()[0])

    def report(tag: str) -> None:
        b = (scene.butter.data.root_pos_w - scene.env_origins)[0]
        p = (scene.paraffin.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] {tag:10s} | q={qd():+.4f} butter=({float(b[0]):+.3f},"
              f"{float(b[1]):+.3f},{float(b[2]):.3f}) "
              f"paraffin=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):.3f}) "
              f"|vb|={float(scene.butter.data.root_lin_vel_w[0].norm()):.4f} "
              f"|vd|={float(scene.drawer.data.root_lin_vel_w[0].norm()):.4f} "
              f"closed={bool(scene.drawer_closed()[0])} "
              f"in_cav={bool(scene.in_cavity(scene.butter)[0])} "
              f"in_chute={bool(scene.in_chute(scene.butter)[0])} "
              f"in_cellar={bool(scene.in_cellar(scene.butter)[0])} "
              f"app={float(scene._app_max[0]):.3f} "
              f"L(closed/chute/inside)=({int(scene._closed[0])},{int(scene._chuted[0])},"
              f"{int(scene._inside[0])}) success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def place(body, x: float, y: float, z: float) -> None:
        """One pose write (transport across free space): identity quat, zero velocity."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    b0 = (scene.butter.data.root_pos_w - scene.env_origins)[0]
    p0 = (scene.paraffin.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): q0={qd():+.4f} "
          f"butter=({float(b0[0]):+.3f},{float(b0[1]):+.3f}) "
          f"paraffin=({float(p0[0]):+.3f},{float(p0[1]):+.3f})", flush=True)
    report("reset")
    assert c.q0_min - 0.01 < qd() < c.q0_max + 0.01, \
        f"drawer q0 {qd():+.4f} outside the sampling band — joint or reset broken"
    assert not bool(scene.drawer_closed()[0]), "drawer spawned closed"
    assert not bool(scene.in_cavity(scene.butter)[0]), "butter spawned inside the drawer"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"baseline score not ~0 ({s0:.3f})"

    # ---------------- phase 1: CLOSE the drawer (force dynamics on the real slide) ----------
    # Velocity-servoed push along +x (the knob-press direction). No pose writes to the
    # drawer: the motion is pure joint dynamics driven by a regulated external force.
    kp, v_des = 8.0, 0.06
    q_start = qd()
    closed_reached = False
    for i in range(900):
        vx = scene.drawer.data.root_lin_vel_w[:, 0]
        f = torch.zeros(n, 1, 3, device=device)
        f[:, 0, 0] = kp * (v_des - vx)
        scene.drawer.set_external_force_and_torque(f, zero_wrench, env_ids=all_ids,
                                                   is_global=True)
        env.step(no_action)
        if qd() > -0.004:
            closed_reached = True
            break
    scene.drawer.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    step(60)  # hands off: seat against the q=0 stop, come to rest
    report("closed")
    print(f"[solve] drawer pushed {qd() - q_start:+.4f} m in {i + 1} servo steps", flush=True)
    assert closed_reached, f"drawer never reached the closed stop (q={qd():+.4f})"
    assert bool(scene.drawer_closed()[0]), f"drawer not within close_tol (q={qd():+.4f})"
    assert bool(scene._closed[0]), "closed latch not set (drawer not still?)"
    s1 = print_score("P1 drawer pushed shut (velocity-servoed force on the slide)")
    assert s1 >= s0 - 1e-6, "score decreased across the close"
    assert s1 >= c.w_close - 0.01, f"close credit missing ({s1:.3f})"

    # ---------------- phase 2: DEPOSIT — transport hover, then gravity does the rest --------
    # One pose write to a hover above the collar mouth. Asserted: the endpoint is above
    # the chute scoring band and outside the cavity band, so nothing latches at the
    # write instant. The fall through collar -> hatch hole -> cavity is ballistic + real
    # contact on the drawer floor.
    cx, cy = c.cab_pos
    hover_z = c.chute_z_hi + 0.008
    place(scene.butter, cx + c.port_cx, cy, hover_z)
    assert not bool(scene.in_chute(scene.butter)[0]), \
        "hover endpoint already inside the chute scoring band"
    assert not bool(scene.in_cavity(scene.butter)[0]), \
        "hover endpoint already inside the cavity band"
    step(200)
    report("deposited")
    assert bool(scene._chuted[0]), \
        "chute-transit latch never fired — butter did not pass the hatch while closed"
    assert bool(scene.in_cavity(scene.butter)[0]), "butter did not land inside the cavity"
    assert bool(scene.drawer_closed()[0]), \
        f"deposit impact knocked the drawer open (q={qd():+.4f})"
    assert not bool(scene.in_cavity(scene.paraffin)[0]), "paraffin ended up in the cavity"
    s2 = print_score("P2 butter posted through the hatch (gravity delivery)")
    assert s2 >= s1 - 1e-6, "score decreased across the deposit"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the deposit)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 3: persistence (>= 3 simulated seconds, no intervention) -------
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
    except BaseException as exc:  # noqa: BLE001 — die loudly, never hang until the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(exc).__name__}: {exc})", flush=True)
        os._exit(1)

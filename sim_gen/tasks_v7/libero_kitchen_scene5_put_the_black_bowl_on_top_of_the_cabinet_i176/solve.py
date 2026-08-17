"""Teleport solution for BallastRockerScene (sim_gen task
`libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet_i176`) — the task's
legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, twice): ONE pose write per object moves it across free space
   to a hover ABOVE its goal — touching nothing, inside no scoring band. The ballast
   hovers 3.0 cm above the socket's z band top (tray-local z 0.075 > sock_z_hi 0.045);
   the bowl hovers 2.0 cm above the zone's z band top (tray-local z 0.060 >
   zone_z_hi 0.040). Both hovers are ASSERTED to satisfy no gate and no success.
2. SEATING THE BALLAST (contact dynamics — the interlock stage): the ballast is
   released from its hover and falls into the red-fenced socket under gravity; the
   landing impulse, any fence funnelling, and the tray's response on its REAL hinge
   are pure physics. The solve asserts the tray then rests STILL on its rest stop
   with the ballast seated — the hinge demonstrably carries the counterweight.
3. PLACING THE BOWL (contact dynamics — the payoff the interlock exists for): the
   bowl is released from its hover and falls onto the raised zone end. Without the
   ballast this exact drop tips the tray and dumps the bowl (smoke.py constructs that
   outcome and asserts rejection); here the seated ballast out-torques the bowl 5:1,
   so the tray absorbs the landing nod and stays on its rest stop, physically
   CARRYING the bowl. Landing, nod and settling are pure physics.

The tray itself is NEVER teleported and never pushed: gravity alone holds it against
its rest stop, and the counterweight alone keeps it there when loaded.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet_i176.solve --headless [--seed N]
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

import math
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
    env = ENVS.get("simgen.ballast_rocker")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    # Rest-tilt quat (roll +rest_deg about +x): hover pre-tilted to land flat on the
    # tilted tray.
    half = math.radians(c.rest_deg) / 2
    rest_quat = (math.cos(half), math.sin(half), 0.0, 0.0)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def loc(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        p_bal, p_bowl = loc(scene.ballast), loc(scene.bowl)
        print(f"[solve] {tag:12s} | bal=({float(p_bal[0]):+.3f},{float(p_bal[1]):+.3f},"
              f"{float(p_bal[2]):.3f}) bowl=({float(p_bowl[0]):+.3f},"
              f"{float(p_bowl[1]):+.3f},{float(p_bowl[2]):.3f}) "
              f"tray={float(scene.tray_deg()[0]):+.1f}deg "
              f"insock={bool(scene.ballast_in_socket()[0])} "
              f"seated={bool(scene.ballast_seated()[0])} "
              f"inzone={bool(scene.bowl_in_zone()[0])} "
              f"lat=[cb{float(scene._carry_bal_max[0]):.2f} s{int(scene._seated[0])} "
              f"ck{float(scene._carry_bowl_max[0]):.2f} p{int(scene._placed[0])}] "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def teleport(body, pos, quat) -> None:
        """Transport ONLY: one free-space pose write, zero velocity."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = pos
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def settle(body, max_steps: int = 600, need: int = 30) -> None:
        quiet = 0
        for _ in range(max_steps):
            env.step(no_action)
            ok_now = bool(scene._still(body)[0]) and bool(scene._tray_still()[0])
            quiet = quiet + 1 if ok_now else 0
            if quiet >= need:
                break

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    p_bal, p_bowl = loc(scene.ballast), loc(scene.bowl)
    tray0 = float(scene.tray_deg()[0])
    print(f"[solve] layout readback (seed {args.seed}): bal=({float(p_bal[0]):+.3f},"
          f"{float(p_bal[1]):+.3f}) bowl=({float(p_bowl[0]):+.3f},{float(p_bowl[1]):+.3f}) "
          f"tray={tray0:+.1f}deg sock_goal=({c.sock_goal[0]:.3f},{c.sock_goal[1]:.3f},"
          f"{c.sock_goal[2]:.3f}) zone_goal=({c.zone_goal[0]:.3f},{c.zone_goal[1]:.3f},"
          f"{c.zone_goal[2]:.3f})", flush=True)
    report("reset")
    assert bool(scene.tray_at_rest()[0]), \
        "empty tray did not rest on its +rest_deg stop at reset"
    assert float(p_bal[2]) < 0.05 and float(p_bowl[2]) < 0.03, \
        "ballast/bowl did not settle standing on the floor"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.05, "score not ~0 at reset"

    # ---------------- phase 1: TRANSPORT + SEAT THE BALLAST (interlock, contact) ------------
    # Hover: tray-local (0, sock_y, 0.075) at the rest angle — 3.0 cm above the
    # socket gate's z band (sock_z_hi 0.045), touching nothing (fences top out at
    # tray-local 0.025; the cube's lowest point at the hover is 0.050).
    hover_bal = c.rest_point(c.sock_y, 0.075)
    teleport(scene.ballast, hover_bal, rest_quat)
    report("bal-hover")
    assert not bool(scene.ballast_in_socket()[0]) and not bool(scene.ballast_seated()[0]), \
        "hovering above the socket must not satisfy the socket gate (teleport is transport only)"
    s1a = print_score("P1a transport: ballast hovering above the socket")
    assert s1a >= s0 - 1e-6, "score decreased across ballast transport"
    settle(scene.ballast)
    report("bal-seated")
    assert bool(scene.ballast_in_socket()[0]), \
        "ballast did not land inside the fenced socket"
    assert bool(scene.ballast_seated()[0]), \
        "ballast+tray did not settle with the tray on its rest stop (interlock not established)"
    s1 = print_score("P1 ballast dropped and seated in the socket (contact)")
    assert s1 >= s1a - 1e-6, "score decreased across the ballast seating"
    assert s1 >= 0.30, "seating the ballast did not earn the interlock credit"

    # ---------------- phase 2: TRANSPORT the black bowl (one free-space teleport) -----------
    # Hover: tray-local (0, zone_y, 0.060) at the rest angle — 2.0 cm above the zone
    # gate's z band (zone_z_hi 0.040), pre-tilted to the tray's rest angle.
    hover_bowl = c.rest_point(c.zone_y, 0.060)
    teleport(scene.bowl, hover_bowl, rest_quat)
    report("bowl-hover")
    assert not bool(scene.bowl_in_zone()[0]) and not bool(scene.success()[0]), \
        "hovering above the zone must not satisfy the zone gate (teleport is transport only)"
    s2 = print_score("P2 transport: bowl hovering above the zone")
    assert s2 >= s1 - 1e-6, "score decreased across bowl transport"

    # ---------------- phase 3: PLACEMENT (gravity drop + settle, pure physics) --------------
    # The ballasted tray must absorb the landing nod and keep resting on its stop —
    # the counterweight interlock demonstrably carries the bowl.
    settle(scene.bowl)
    report("placed")
    assert bool(scene.ballast_seated()[0]), \
        "tray did not stay pinned on its rest stop under the bowl (interlock failed)"
    assert bool(scene.bowl_in_zone()[0]), \
        "bowl did not settle upright inside the zone on the ballasted tray"
    s3 = print_score("P3 bowl landed and settled on the zone (contact)")
    assert s3 >= s2 - 1e-6, "score decreased across the placement"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after placement)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3 simulated seconds, no intervention) --------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s4 >= s3 - 1e-6
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
    main()

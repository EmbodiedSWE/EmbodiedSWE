"""Teleport solution for DrawbridgeVaultScene (sim_gen task
`libero_kitchen_scene10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it_i332`)
— the task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): ONE pose write moves the BROWN pudding box across free space
   from its table slot to a hover just above the LOWER-MIDDLE of the lowered
   drawbridge ramp — touching nothing, far outside the deep-inside scoring band (the
   box is still ~25 cm from the chamber). The white decoy box is never touched.
2. ENTRY THROUGH THE DOORWAY (contact dynamics — teleporting the box into the chamber
   would bypass the task and is never done): a regulated force (velocity-servoed at
   0.22 m/s along the slope, capped at 2.0 N) pushes the box UP the ~20 deg ramp,
   over the sill, across the porch and through the doorway; the force is CUT the
   moment the box's centre crosses the deep-inside threshold and the box slides to
   rest on the chamber floor under friction alone. The whole climb is real contact:
   ramp friction (defined 0.75 material), the sill crossing, and the doorway aperture.
3. CLOSING THE BRIDGE (contact/joint dynamics): a regulated torque about the hinge
   axis (velocity-servoed on the hinge rate, capped at 0.8 N*m — barely 3x the static
   gravity torque) rotates the bridge up from the table through ~113 deg; the torque
   is CUT at 93 deg, just past vertical, and gravity alone carries the plate onto its
   +98 deg stop and pins it shut. The solve ASSERTS the bridge started at its lowered
   table rest (~-20 deg) and ended settled shut — the mechanism demonstrably
   travelled its whole arc under dynamics, with the box already inside.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it_i332.solve --headless [--seed N]
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
    env = ENVS.get("simgen.drawbridge_vault")().build(num_envs=args.num_envs, device=device)
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

    def loc(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        p_pud, p_dec = loc(scene.pudding), loc(scene.decoy)
        print(f"[solve] {tag:12s} | pudding=({float(p_pud[0]):+.3f},{float(p_pud[1]):+.3f},"
              f"{float(p_pud[2]):.3f}) decoy=({float(p_dec[0]):+.3f},"
              f"{float(p_dec[1]):+.3f},{float(p_dec[2]):.3f}) "
              f"bridge={float(scene.bridge_deg()[0]):+.1f}deg "
              f"shut={bool(scene.bridge_shut()[0])} "
              f"in={bool(scene.box_inside(scene.pudding)[0])} "
              f"lat=[a{float(scene._appr_max[0]):.2f} i{int(scene._inside[0])} "
              f"c{float(scene._close_max[0]):.2f} s{int(scene._closed[0])}] "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    p_pud, p_dec = loc(scene.pudding), loc(scene.decoy)
    a_rest = float(scene.bridge_deg()[0])
    print(f"[solve] layout readback (seed {args.seed}): pudding=({float(p_pud[0]):+.3f},"
          f"{float(p_pud[1]):+.3f}) decoy=({float(p_dec[0]):+.3f},{float(p_dec[1]):+.3f}) "
          f"bridge_rest={a_rest:.1f}deg mouth=({c.mouth_pt[0]:.3f},{c.mouth_pt[1]:.3f},"
          f"{c.mouth_pt[2]:.3f})", flush=True)
    report("reset")
    assert -24.0 < a_rest < -15.0, "bridge did not rest lowered on the table at reset"
    assert float(p_pud[2]) < 0.05 and float(p_dec[2]) < 0.05, \
        "boxes did not settle standing on the table"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.05, "score not ~0 at reset"

    # ---------------- phase 1: TRANSPORT the pudding box (one free-space teleport) ----------
    # Endpoint: hovering 8 mm above the lower-middle of the lowered ramp, tilted to the
    # ramp's rest angle — free space, touching nothing, ~25 cm outside the deep-inside
    # band. Gravity lands it on the ramp; the defined 0.75 friction holds it there.
    th = math.radians(-a_rest)  # ramp inclination (positive)
    s_down = 0.15  # down-slope distance from the hinge to the drop point
    nx, nz = -math.sin(th), math.cos(th)  # ramp top-face normal
    px = c.hinge_x - s_down * math.cos(th) + (c.edge / 2 + 0.008) * nx
    pz = c.hinge_z - s_down * math.sin(th) + (c.edge / 2 + 0.008) * nz
    half = math.radians(a_rest) / 2
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1], st[:, 2] = px, c.keep_pos[1], pz
    st[:, 3], st[:, 5] = math.cos(half), math.sin(half)
    st[:, 0:3] += scene.env_origins
    scene.pudding.write_root_state_to_sim(st, all_ids)
    quiet = 0
    for _ in range(240):
        env.step(no_action)
        quiet = quiet + 1 if bool(scene._box_still(scene.pudding)[0]) else 0
        if quiet >= 30:
            break
    report("on-ramp")
    assert not bool(scene.box_inside(scene.pudding)[0]) and not bool(scene.success()[0]), \
        "the transport teleport must not enter the deep-inside band (transport only)"
    assert float(loc(scene.pudding)[2]) > 0.035, \
        "box did not rest ON the ramp (slid off or fell through)"
    s1 = print_score("P1 transport: pudding resting on the lowered ramp")
    assert s1 >= s0 - 1e-6, "score decreased across transport"

    # ---------------- phase 2: ENTRY through the doorway (contact dynamics) -----------------
    # Velocity-servoed push: up the slope while on the ramp, horizontal once past the
    # sill; CUT the moment the centre crosses the deep-inside threshold — the last
    # centimetres and the coming-to-rest are pure friction.
    pushed = 0
    for _ in range(1200):
        p = loc(scene.pudding)
        if float(p[0]) >= c.inside_x_lo + 0.010:
            break
        if float(p[0]) < c.hinge_x - 0.02:
            u = torch.tensor([math.cos(th), 0.0, math.sin(th)], device=device)
        else:
            u = torch.tensor([1.0, 0.0, 0.0], device=device)
        v = float((scene.pudding.data.root_lin_vel_w[0] * u).sum())
        f = max(0.0, min(2.0, 8.0 * (0.22 - v)))
        f_w = (f * u).view(1, 1, 3).expand(n, 1, 3)
        scene.pudding.set_external_force_and_torque(f_w.contiguous(), zero_wrench,
                                                    env_ids=all_ids, is_global=True)
        env.step(no_action)
        pushed += 1
    scene.pudding.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    quiet = 0
    for _ in range(600):
        env.step(no_action)
        ok_now = (bool(scene.box_inside(scene.pudding)[0])
                  and bool(scene._box_still(scene.pudding)[0]))
        quiet = quiet + 1 if ok_now else 0
        if quiet >= 30:
            break
    report("inside")
    print(f"[solve] entry: pushed {pushed} steps up the ramp and through the doorway",
          flush=True)
    assert bool(scene.box_inside(scene.pudding)[0]) and \
        bool(scene._box_still(scene.pudding)[0]), \
        "pudding did not settle deep inside the chamber after the push"
    s2 = print_score("P2 pudding pushed up the ramp and through the doorway (contact)")
    assert s2 >= s1 - 1e-6, "score decreased across the entry"

    # ---------------- phase 3: RAISE THE BRIDGE (contact/joint dynamics) --------------------
    # Regulated torque about the hinge axis: ~1.2 rad/s until 70 deg, then a gentle
    # 0.5 rad/s approach; CUT at 93 deg (just past vertical) — the fall onto the
    # +98 deg stop and the pinning are pure gravity + the joint limit.
    raised = 0
    for _ in range(1200):
        a = float(scene.bridge_deg()[0])
        if a >= 93.0:
            break
        w_tgt = 1.2 if a < 70.0 else 0.5
        w = float(scene.bridge.data.root_ang_vel_w[0, 1])
        tq = max(0.0, min(0.8, 2.5 * (w_tgt - w)))
        t_w = torch.tensor([0.0, tq, 0.0], device=device).view(1, 1, 3).expand(n, 1, 3)
        scene.bridge.set_external_force_and_torque(zero_wrench, t_w.contiguous(),
                                                   env_ids=all_ids, is_global=True)
        env.step(no_action)
        raised += 1
    scene.bridge.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    quiet = 0
    for _ in range(600):
        env.step(no_action)
        ok_now = bool(scene.bridge_shut()[0]) and bool(scene._bridge_still()[0])
        quiet = quiet + 1 if ok_now else 0
        if quiet >= 30:
            break
    report("shut")
    print(f"[solve] bridge: torqued {raised} steps from {a_rest:.1f} deg, now "
          f"{float(scene.bridge_deg()[0]):+.1f} deg", flush=True)
    assert bool(scene.bridge_shut()[0]) and bool(scene._bridge_still()[0]), \
        "bridge did not settle shut on its stop after the past-vertical torque"
    assert bool(scene.box_inside(scene.pudding)[0]), \
        "pudding was disturbed out of the chamber by the closing bridge"
    s3 = print_score("P3 bridge driven past vertical and fallen shut (contact)")
    assert s3 >= s2 - 1e-6, "score decreased across the closing"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after closing)", flush=True)
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

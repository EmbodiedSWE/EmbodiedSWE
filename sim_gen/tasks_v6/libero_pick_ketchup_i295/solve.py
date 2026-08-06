"""Teleport solution for PostboxDepositScene (sim_gen task `libero_pick_ketchup_i295`)
— the task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): one pose write carries the red ketchup bottle from its
   randomized floor slot to LYING ON THE LOADING TRAY, cap toward the box — a
   supported free pose, verifiably OUTSIDE the box and with the flap still hanging
   shut (both asserted). This earns exactly the on-tray staging credit that carrying
   the bottle to the tray earns; every other rubric clause remains 0.
2. PUSH THROUGH THE FLAP (contact dynamics): a bounded horizontal force on the bottle
   (the applied-wrench emulation of the arm's fingertips pushing its base) slides it
   along the tray; its cap meets the flap and shoves it inward THROUGH CONTACT — the
   flap's whole trajectory is hinge physics — until the bottle's centre crosses the
   sill. The force is then REMOVED. No pose write touches the bottle after phase 1.
3. GRAVITY DESCENT + FLAP RETURN (contact dynamics): hands off — the bottle tips over
   the sill, slides down the internal low-friction ramp to the box floor, and the
   flap swings shut behind it under gravity + its weak closing spring. success()
   first turns True here, judged on settled physical state (containment below sill
   level, flap hanging shut, decoys outside, everything at rest).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.<task>.solve --headless [--seed N]
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
    env = ENVS.get("simgen.postbox_deposit")().build(num_envs=args.num_envs, device=device)
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

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def kloc() -> torch.Tensor:
        return scene._local(scene.ketchup)[0]

    def report(tag: str) -> None:
        p = kloc()
        print(f"[solve] {tag:12s} | flap={float(scene.flap_open_deg()[0]):6.1f} deg "
              f"ketchup_loc=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):.3f}) "
              f"inside={bool(scene._inside_box(scene.ketchup)[0])} "
              f"tray={bool(scene._tray[0])} eng={bool(scene._eng[0])} "
              f"crossed={bool(scene._crossed[0])} desc={float(scene._desc_max[0]):.3f} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(90)
    k0 = (scene.ketchup.data.root_pos_w - scene.env_origins)[0]
    m0 = (scene.mustard.data.root_pos_w - scene.env_origins)[0]
    y0 = (scene.mayo.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"ketchup=({float(k0[0]):+.3f},{float(k0[1]):+.3f}) "
          f"mustard=({float(m0[0]):+.3f},{float(m0[1]):+.3f}) "
          f"mayo=({float(y0[0]):+.3f},{float(y0[1]):+.3f}) "
          f"flap={float(scene.flap_open_deg()[0]):.1f} deg", flush=True)
    report("reset")
    assert float(scene.flap_open_deg()[0]) <= 2.0, "flap did not hang shut at reset"
    assert not bool(scene._inside_box(scene.ketchup)[0]), "ketchup spawned inside the box"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"baseline score not ~0 ({s0:.3f})"

    # ---------------- phase 1: TRANSPORT to the loading tray (teleport) ---------------------
    # One pose write: ketchup lying on the tray, cap (+z body) toward the box (+x),
    # centred between the guide rails, provably outside the box and not touching the
    # flap (its nose ends ~15 mm short of the front wall).
    dx, dy, _dz = c.depot_pos
    lie_q = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)  # R_y(90): +z -> +x
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = dx - 0.115
    st[:, 1] = dy
    st[:, 2] = c.sill_z + c.ketchup_body_r + 0.003
    st[:, 3], st[:, 5] = lie_q[0], lie_q[2]
    st[:, 0:3] += scene.env_origins
    scene.ketchup.write_root_state_to_sim(st, all_ids)
    step(60)  # settle onto the tray
    report("on-tray")
    assert bool(scene._tray[0]), "on-tray latch did not fire"
    assert not bool(scene._inside_box(scene.ketchup)[0]), \
        "tray pose is already inside the box (teleport must stay outside)"
    assert float(scene.flap_open_deg()[0]) <= 2.0, "flap disturbed by the tray placement"
    s1 = print_score("P1 transport: ketchup lying on the loading tray (still outside)")
    assert s1 >= s0 - 1e-6, "score decreased across transport"

    # ---------------- phase 2: PUSH through the flap (contact dynamics) ---------------------
    # Bounded horizontal force at the bottle's centre (world frame): a bang-bang law
    # that always exceeds tray static friction (~1.0 N for the 0.2 kg bottle) while
    # capping the slide speed; the cap meets the flap and shoves it inward through
    # contact; stop pushing once the bottle's centre is past the sill (local x > 0.045).
    v_cap, f_push, f_coast = 0.15, 4.0, 0.5
    zero3 = torch.zeros(n, 1, 3, device=device)
    pushed = False
    for i in range(900):
        vx = float(scene.ketchup.data.root_lin_vel_w[0, 0])
        vy = float(scene.ketchup.data.root_lin_vel_w[0, 1])
        y = float(kloc()[1])
        f = f_push if vx < v_cap else f_coast
        fy = max(-1.0, min(1.0, 4.0 * (0.0 - y) - 2.0 * vy))  # stay centred in the rails
        force = torch.zeros(n, 1, 3, device=device)
        force[:, 0, 0] = f
        force[:, 0, 1] = fy
        scene.ketchup.set_external_force_and_torque(force, zero3, env_ids=all_ids,
                                                    is_global=True)
        env.step(no_action)
        if i % 100 == 99:
            print(f"[solve]   push i={i + 1}: x={float(kloc()[0]):+.3f} vx={vx:+.3f} "
                  f"f={f:.1f} flap={float(scene.flap_open_deg()[0]):.1f} deg", flush=True)
        if float(kloc()[0]) > 0.045:
            pushed = True
            break
    scene.ketchup.set_external_force_and_torque(zero3, zero3, env_ids=all_ids,
                                                is_global=True)
    report("pushed")
    assert pushed, "push never carried the ketchup centre past the sill"
    assert bool(scene._eng[0]), "flap-engaged latch did not fire during the push"
    assert bool(scene._crossed[0]), "crossed-the-sill latch did not fire"
    s2 = print_score("P2 pushed through the flap: centre past the sill, force removed")
    assert s2 >= s1 - 1e-6, "score decreased across the push"

    # ---------------- phase 3: gravity descent + flap return (hands off) --------------------
    settled = False
    quiet = 0
    for _ in range(700):
        env.step(no_action)
        ok_now = (bool(scene._inside_box(scene.ketchup)[0])
                  and float(scene.ketchup.data.root_lin_vel_w[0].norm()) < 0.04
                  and float(scene.flap_open_deg()[0]) <= c.closed_tol_deg)
        quiet = quiet + 1 if ok_now else 0
        if quiet >= 30:
            settled = True
            break
    report("descended")
    assert settled and bool(scene._inside_box(scene.ketchup)[0]), \
        "ketchup did not settle inside the box with the flap shut"
    s3 = print_score("P3 gravity descent down the ramp; flap swung shut behind it")
    assert s3 >= s2 - 1e-6, "score decreased across the descent"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after descent)", flush=True)
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
    try:
        main()
    except Exception as exc:  # noqa: BLE001 — die loudly, never hang in Kit teardown
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({exc})", flush=True)
        os._exit(1)

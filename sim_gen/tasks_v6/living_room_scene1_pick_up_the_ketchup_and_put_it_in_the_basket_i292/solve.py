"""Teleport solution for RollInGarageScene (sim_gen task
`living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket_i292`) — the task's
legitimacy certificate.

The task's load-bearing interaction and how it is executed:

1. CLEAR THE DOORWAY (transport, teleport): one pose write relocates the orange post
   from the doorway throat to an open floor patch — pure free-space transport of a free
   object (the arm's version is a straightforward pinch-lift of the jaw-sized post).
   No interaction is bypassed: the post's only role is to occupy the aperture.
2. TRANSPORT (teleport): one pose write carries the red ketchup bottle from its spawn
   slot to a staging pose on the approach axis, 14 cm OUTSIDE the doorway plane, lying
   with its rolling axis across the approach (verifiably outside the garage —
   asserted). Hovering/staging satisfies no rubric clause beyond the doorway-approach
   ramp.
3. ROLL IN OVER THE CREST (contact dynamics — the load-bearing interaction, never
   teleported): a velocity-limited horizontal force at the bottle's CoM (the applied-
   wrench emulation of the arm's fingertip push) rolls the bottle across the floor,
   through the doorway, UP the 12 deg ramp. The force is CUT the moment the bottle's
   CoM passes the crest — gravity tips it over the drop-off and it rolls out into the
   landing bay, where the overhanging crest face retains it. From the force cut to the
   verdict nothing touches the bottle; every millimetre of the doorway passage, climb,
   drop and settle is contact physics.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after success()
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
    env = ENVS.get("simgen.roll_in_garage")().build(num_envs=args.num_envs, device=device)
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

    from isaaclab.utils.math import quat_apply, quat_mul

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def k_loc() -> torch.Tensor:
        return scene._garage_local(scene.ketchup)[0]

    def report(tag: str) -> None:
        lo = k_loc()
        print(f"[solve] {tag:12s} | ketchup_loc=({float(lo[0]):+.3f},{float(lo[1]):+.3f},"
              f"{float(lo[2]):.3f}) blocking={bool(scene._post_blocking()[0])} "
              f"cleared={bool(scene._cleared[0])} app={float(scene._app_max[0]):.3f} "
              f"entered={bool(scene._entered[0])} landed={bool(scene._landed[0])} "
              f"in_bay={bool(scene._in_landing_bay(scene.ketchup)[0])} "
              f"mustard_in={bool(scene._in_garage(scene.mustard)[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    g = (scene.garage.data.root_pos_w - scene.env_origins)[0]
    gq = scene.garage.data.root_quat_w[0]
    g_yaw = 2.0 * math.atan2(float(gq[3]), float(gq[0]))
    k0 = (scene.ketchup.data.root_pos_w - scene.env_origins)[0]
    m0 = (scene.mustard.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"garage=({float(g[0]):+.3f},{float(g[1]):+.3f}) yaw={math.degrees(g_yaw):+.1f}deg "
          f"ketchup=({float(k0[0]):+.3f},{float(k0[1]):+.3f}) "
          f"mustard=({float(m0[0]):+.3f},{float(m0[1]):+.3f})", flush=True)
    report("reset")
    assert bool(scene._post_blocking()[0]), "post did not spawn blocking the doorway"
    assert not bool(scene._in_garage(scene.ketchup)[0]), "ketchup spawned inside"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"reset score should be ~0, got {s0}"

    # ---------------- phase 1: CLEAR THE DOORWAY (transport teleport) -----------------------
    # One pose write: post to the open floor patch, standing. Free-space transport.
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1], st[:, 2] = c.park_spot[0], c.park_spot[1], 0.002
    st[:, 3] = 1.0
    st[:, 0:3] += scene.env_origins
    scene.post.write_root_state_to_sim(st, all_ids)
    step(30)
    report("cleared")
    assert bool(scene._cleared[0]), "post relocation did not clear the doorway"
    s1 = print_score("P1 post transported out of the doorway")
    assert s1 >= s0 - 1e-6, "score decreased across clearing"

    # ---------------- phase 2: TRANSPORT to the staging pose (teleport, outside) ------------
    # One pose write: ketchup to 14 cm outside the doorway plane on the garage axis,
    # lying with its long axis ACROSS the approach (so it rolls straight in).
    g_pos = scene.garage.data.root_pos_w
    g_quat = scene.garage.data.root_quat_w
    stage_loc = torch.tensor([-0.14, 0.0, c.body_r + 0.002], device=device).expand(n, 3)
    qx = torch.tensor([math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0.0, 0.0],
                      device=device).expand(n, 4)  # bottle +z -> +y
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = g_pos + quat_apply(g_quat, stage_loc)
    st[:, 3:7] = quat_mul(g_quat, qx)  # axis along the garage's v axis
    scene.ketchup.write_root_state_to_sim(st, all_ids)
    step(30)
    report("staged")
    assert not bool(scene._in_garage(scene.ketchup)[0]), \
        "staging pose is already inside the garage (teleport must stay outside)"
    s2 = print_score("P2 ketchup transported to the staging pose (outside the doorway)")
    assert s2 >= s1 - 1e-6, "score decreased across transport"

    # ---------------- phase 3: ROLL IN OVER THE CREST (contact dynamics) --------------------
    # Velocity-limited CoM force along the garage axis + a small lateral centering term;
    # cut the force the moment the CoM passes the crest — gravity finishes the drop.
    push_dir = quat_apply(g_quat, torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))
    lat_dir = quat_apply(g_quat, torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3))
    f_push, v_des = 1.8, 0.22
    best_u, last_bump = -1.0, 0
    crossed = False
    for i in range(1500):
        lo = k_loc()
        u, v = float(lo[0]), float(lo[1])
        if u > c.crest_u + 0.004:
            crossed = True
            break
        vel = scene.ketchup.data.root_lin_vel_w[0]
        u_vel = float((vel * push_dir[0]).sum())
        v_vel = float((vel * lat_dir[0]).sum())
        f_axis = f_push if u_vel < v_des else 0.0
        f_lat = max(-0.8, min(0.8, -6.0 * v - 1.5 * v_vel))
        f_w = (push_dir * f_axis + lat_dir * f_lat).view(n, 1, 3)
        scene.ketchup.set_external_force_and_torque(f_w.contiguous(), zero_wrench,
                                                    env_ids=all_ids, is_global=True)
        env.step(no_action)
        if u > best_u + 0.003:
            best_u, last_bump = u, i
        elif i - last_bump > 240:  # stalled on the ramp: push harder
            f_push = min(f_push + 0.6, 6.0)
            last_bump = i
            print(f"[solve] stall at u={u:+.3f} -> f_push={f_push:.1f} N", flush=True)
    scene.ketchup.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    report("crest")
    assert crossed, "push never carried the ketchup CoM past the crest"

    # hands-off: drop into the bay and settle (the roller needs a moment to calm down)
    quiet = 0
    for _ in range(900):
        env.step(no_action)
        still = (float(scene.ketchup.data.root_lin_vel_w[0].norm()) < 0.03
                 and float(scene.ketchup.data.root_ang_vel_w[0].norm()) < 0.4)
        quiet = quiet + 1 if (still and bool(scene._in_landing_bay(scene.ketchup)[0])) else 0
        if quiet >= 30:
            break
    report("landed")
    assert bool(scene._in_landing_bay(scene.ketchup)[0]), \
        "ketchup did not settle in the landing bay"
    s3 = print_score("P3 rolled through the doorway and over the crest, settled in the bay")
    assert s3 >= s2 - 1e-6, "score decreased across the roll-in"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after landing)", flush=True)
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, no intervention) ------
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

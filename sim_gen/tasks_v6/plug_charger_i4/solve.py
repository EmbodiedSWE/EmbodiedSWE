"""Teleport solution for BayonetLockScene (sim_gen task `plug_charger_i4`) — the task's
legitimacy certificate.

Teleportation handles TRANSPORT ONLY: one pose write carries the plug across open space
from its spawn to a hover pose centred over the socket mouth, bottom ~4 mm ABOVE the
overhang plates, lugs aligned with the entry slot (exactly the pose an arm would lower
from; the plug is still entirely outside the well, and hovering above the plates is far
outside every scoring gate). Both LOAD-BEARING interactions then go through CONTACT
DYNAMICS:
  1. SEATING — a gentle regulated downforce (plus small lateral centring) lowers the
     plug through the slot; the lugs pass the 54 mm opening, the plug body descends the
     30 mm well and lands on the floor under real contact;
  2. LOCKING — an external torque about the vertical axis (velocity-regulated
     bang-bang, with a small hold-down force) twists the seated plug against floor
     friction until the lugs ride ~68 deg under the plates; the torque is cut and the
     plug settles on real friction. Nothing is ever written into a locked state.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.plug_charger_i4.solve --headless [--seed N]
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
    env = ENVS.get("simgen.plug_bayonet_lock")().build(num_envs=args.num_envs, device=device)
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

    def socket_pose() -> tuple[torch.Tensor, float]:
        sp = (scene.socket.data.root_pos_w - scene.env_origins)[0]
        q = scene.socket.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return sp, yaw

    def report(tag: str) -> None:
        loc = scene._plug_local()[0]
        print(f"[solve] {tag:12s} | plug_local=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
              f"{float(loc[2]):.3f}) slot_dist={float(scene.slot_distance_deg()[0]):5.1f}deg "
              f"seated={bool(scene.seated()[0])} locked={bool(scene.locked()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_wrench() -> None:
        scene.plug.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    sp, syaw = socket_pose()
    plug0 = (scene.plug.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): socket=({float(sp[0]):+.3f},"
          f"{float(sp[1]):+.3f}) yaw={math.degrees(syaw):+.1f}deg "
          f"plug_spawn=({float(plug0[0]):+.3f},{float(plug0[1]):+.3f}) "
          f"extract_limit={c.extract_deg:.1f}deg lock_deg={c.lock_deg:.1f}deg", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: TRANSPORT (teleport across free space only) -----------------
    # Hover pose: centred over the well, plug bottom ~4 mm ABOVE the plate top, lugs
    # aligned with the entry slot (plug yaw = socket yaw). The plug is entirely outside
    # the well; hovering above the plates satisfies no scoring gate, so no required
    # interaction is bypassed.
    hover_z = c.entry_z + 0.004
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = float(sp[0])
    st[:, 1] = float(sp[1])
    st[:, 2] = hover_z
    st[:, 3] = math.cos(syaw / 2)
    st[:, 6] = math.sin(syaw / 2)
    st[:, 0:3] += scene.env_origins
    scene.plug.write_root_state_to_sim(st, all_ids)
    step(1)
    report("staged")
    s1 = print_score("P1 transport to hover over the slot")
    assert s1 >= s0 - 1e-6, "score decreased across transport"

    # ---------------- phase 2: SEAT through the slot (contact dynamics) --------------------
    # Regulated downforce + small lateral centring (rotated into the world frame): the
    # lugs pass through the open strip and the plug descends the 30 mm well onto the
    # floor under real contact. Force is cut once the bottom reaches the floor.
    cy, sy = math.cos(syaw), math.sin(syaw)
    f_down, v_des = 2.5, 0.10
    seated_steps = 0
    for i in range(900):
        loc = scene._plug_local()[0]
        if float(loc[2]) <= c.seated_z + 0.0015:
            break
        vz = float(scene.plug.data.root_lin_vel_w[0, 2])
        fz = -f_down if vz > -v_des else 0.0
        flx = max(-0.6, min(0.6, -15.0 * float(loc[0])))
        fly = max(-0.6, min(0.6, -15.0 * float(loc[1])))
        fx = cy * flx - sy * fly
        fy = sy * flx + cy * fly
        f_w = torch.tensor([fx, fy, fz], device=device).view(1, 1, 3).expand(n, 1, 3)
        scene.plug.set_external_force_and_torque(f_w.contiguous(), zero_wrench,
                                                env_ids=all_ids, is_global=True)
        env.step(no_action)
        seated_steps += 1
    clear_wrench()
    step(30)
    report("seated")
    print(f"[solve] seating complete after {seated_steps} steps", flush=True)
    s2 = print_score("P2 seat through the slot (contact)")
    assert s2 >= s1 - 1e-6, "score decreased across seating"

    # ---------------- phase 3: TWIST to lock (contact dynamics) ----------------------------
    # Velocity-regulated bang-bang torque about world z (plus a small hold-down force)
    # twists the seated plug against floor friction; the lugs ride under the plates.
    # Torque is CUT at ~68 deg — inside the 55-90 deg lock zone with margin on both
    # sides — and the plug settles on real friction.
    tau, w_des = 0.05, 1.5
    best_d, last_bump = -1.0, 0
    twist_steps = 0
    for i in range(3600):
        d = float(scene.slot_distance_deg()[0])
        if d >= 68.0:
            break
        wz = float(scene.plug.data.root_ang_vel_w[0, 2])
        tz = tau if wz < w_des else 0.0
        f_w = torch.tensor([0.0, 0.0, -1.5], device=device).view(1, 1, 3).expand(n, 1, 3)
        t_w = torch.tensor([0.0, 0.0, tz], device=device).view(1, 1, 3).expand(n, 1, 3)
        scene.plug.set_external_force_and_torque(f_w.contiguous(), t_w.contiguous(),
                                                env_ids=all_ids, is_global=True)
        env.step(no_action)
        twist_steps += 1
        if d > best_d + 1.0:
            best_d, last_bump = d, i
        elif i - last_bump > 300:  # stalled: twist harder (friction was underestimated)
            tau = min(tau + 0.05, 0.30)
            last_bump = i
            print(f"[solve] twist stalled at {d:.1f} deg, raising torque to {tau:.2f} N*m",
                  flush=True)
    clear_wrench()
    step(120)
    report("twisted")
    print(f"[solve] twist complete after {twist_steps} steps, torque {tau:.2f} N*m, "
          f"slot_dist={float(scene.slot_distance_deg()[0]):.1f} deg", flush=True)
    s3 = print_score("P3 twist to lock (contact)")
    assert s3 >= s2 - 1e-6, "score decreased across twist"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after twist+settle)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3 simulated seconds, no intervention) -------
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

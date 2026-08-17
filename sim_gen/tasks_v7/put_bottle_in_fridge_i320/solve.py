"""Teleport solution for ChestSwapScene (sim_gen task `put_bottle_in_fridge_i320`) —
the task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. OPEN (contact dynamics, no teleport): the captive sliding lid is driven along the
   chest's +y rail axis by a horizontal external force at its CoM (world frame,
   velocity-regulated bang-bang with stall escalation) until the well is fully
   exposed. The lid never receives a pose write.
2. EVICT (contact-free force lift through the open aperture): a vertical hoist force
   (~1.5 mg, vz-gated, with lateral velocity damping) lifts the red can straight up
   out of the well and clear of the walls — exactly what a grasp-and-lift does, and
   only possible because the lid is open (with the lid shut the can would jam against
   its underside). The `evicted` latch sets here from the real state change.
3. BIN TRANSPORT (teleport, free air to free air): one root-state write carries the
   hovering can across to a release pose ABOVE the discard bin's rim — above the
   bin z-band, so the freshly-written state earns no credit. Gravity drops it in;
   the `binned` latch sets from the settled containment.
4. SEAT TRANSPORT (teleport, free air to free air): one root-state write carries the
   amber bottle from its upright ground spawn to a hover pose with its base 95 mm up,
   ABOVE the well mouth (outside the seat depth band — earns nothing), then gravity
   drops it 85 mm down into the vacated well where it self-centres and settles
   upright. Physically possible ONLY because the can is gone: the 84 mm well cannot
   hold both cylinders (30 mm max centre separation < 63 mm sum of radii).
5. CLOSE (contact dynamics, no teleport): the same bang-bang force drive slides the
   lid back to its closed stop over the seated bottle.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: all credit is
latched), then holds HANDS-OFF for >= 3 simulated seconds after success() first turns
True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.put_bottle_in_fridge_i320.solve --headless [--seed N]
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.chest_swap")().build(num_envs=args.num_envs, device=device)
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

    def chest_yaw() -> float:
        q = scene.chest.data.root_quat_w[0]
        return 2.0 * math.atan2(float(q[3]), float(q[0]))

    def report(tag: str) -> None:
        u = float(scene._lid_u()[0])
        bl = scene._chest_local(scene.bottle)[0]
        print(f"[solve] {tag:12s} | lid_u={u:+.3f} bottle_chest=({float(bl[0]):+.3f},"
              f"{float(bl[1]):+.3f},{float(bl[2]):.3f}) can_in_well={bool(scene._can_in_well()[0])} "
              f"opened={bool(scene._opened[0])} evicted={bool(scene._evicted[0])} "
              f"binned={bool(scene._binned[0])} seated={bool(scene._seated[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_lid_force() -> None:
        scene.lid.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def clear_can_force() -> None:
        scene.can.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def push_lid(target_u: float, direction: float, tag: str, v_des: float = 0.10,
                 max_steps: int = 2400) -> None:
        """Drive the lid along its rail axis (chest +y, direction=+1 open / -1 close)
        with a world-frame horizontal force at its CoM (velocity-regulated bang-bang
        with stall escalation) until its travel u crosses `target_u`. Contact
        dynamics only — no pose write ever touches the lid."""
        cyaw = chest_yaw()
        push_dir = torch.tensor([-math.sin(cyaw) * direction, math.cos(cyaw) * direction,
                                 0.0], device=device)
        f_push = 4.0
        best, last_bump = -1e9, 0
        for i in range(max_steps):
            u = float(scene._lid_u()[0])
            if (u - target_u) * direction >= 0.0:
                break
            v_axis = float((scene.lid.data.root_lin_vel_w[0] * push_dir).sum())
            f_axis = f_push if v_axis < v_des else 0.0
            f_w = (push_dir * f_axis).view(1, 1, 3).expand(n, 1, 3)
            scene.lid.set_external_force_and_torque(f_w.contiguous(), zero_wrench,
                                                    env_ids=all_ids, is_global=True)
            env.step(no_action)
            prog = direction * u
            if prog > best + 0.004:
                best, last_bump = prog, i
            elif i - last_bump > 240:  # stalled: push harder (friction underestimated)
                f_push = min(f_push + 2.0, 12.0)
                last_bump = i
                print(f"[solve] {tag}: lid stalled at u={u:+.3f}, raising force "
                      f"to {f_push:.1f} N", flush=True)
        clear_lid_force()
        step(40)  # coast + settle on real friction

    def hoist_can(z_target: float, tag: str, max_steps: int = 1200) -> None:
        """Lift the can straight up out of the well with a vertical external force
        (~1.5 mg, vz-gated bang-bang, lateral velocity damping, stall escalation)
        until its origin clears `z_target`. Only possible with the lid open."""
        mg = c.can_mass * 9.81
        mult = 1.5
        best, last_bump = -1e9, 0
        for i in range(max_steps):
            z = float((scene.can.data.root_pos_w - scene.env_origins)[0, 2])
            if z >= z_target:
                break
            vz = float(scene.can.data.root_lin_vel_w[0, 2])
            fz = mult * mg if vz < 0.12 else 1.0 * mg
            vxy = scene.can.data.root_lin_vel_w[0, :2]
            f = torch.tensor([-0.8 * float(vxy[0]), -0.8 * float(vxy[1]), fz],
                             device=device).view(1, 1, 3).expand(n, 1, 3)
            scene.can.set_external_force_and_torque(f.contiguous(), zero_wrench,
                                                    env_ids=all_ids, is_global=True)
            env.step(no_action)
            if z > best + 0.004:
                best, last_bump = z, i
            elif i - last_bump > 240:
                mult = min(mult + 0.5, 3.0)
                last_bump = i
                print(f"[solve] {tag}: hoist stalled at z={z:.3f}, raising force "
                      f"to {mult:.1f} mg", flush=True)
        # NOTE: force cleared by the caller right at the transport teleport.

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    cyaw = chest_yaw()
    cp = (scene.chest.data.root_pos_w - scene.env_origins)[0]
    bp = (scene.bottle.data.root_pos_w - scene.env_origins)[0]
    np_ = (scene.bin.data.root_pos_w - scene.env_origins)[0]
    kp = (scene.can.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): chest=({float(cp[0]):+.3f},"
          f"{float(cp[1]):+.3f}) yaw={math.degrees(cyaw):+.1f}deg "
          f"lid_u={float(scene._lid_u()[0]):+.3f} "
          f"bottle=({float(bp[0]):+.3f},{float(bp[1]):+.3f}) "
          f"bin=({float(np_[0]):+.3f},{float(np_[1]):+.3f}) "
          f"can=({float(kp[0]):+.3f},{float(kp[1]):+.3f})", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert bool(scene._can_in_well()[0]), "can must start in the well"
    assert bool(scene._lid_closed_now()[0]), "lid must start closed"

    # ---------------- phase 1: OPEN — slide the lid along its rails ------------------------
    push_lid(0.240, +1.0, "open")
    report("opened")
    assert bool(scene._opened[0]), "lid did not open past the latch threshold"
    assert float(scene._lid_u()[0]) >= 0.20, "lid not far enough open"
    assert not bool(scene.success()[0])
    s1 = print_score("P1 lid slid open (contact dynamics)")
    assert s1 >= s0 - 1e-6, "score decreased across open"

    # ---------------- phase 2: EVICT — hoist the can up out of the well --------------------
    hoist_can(0.30, "evict")
    report("evicted")
    assert bool(scene._evicted[0]), "can did not leave the well"
    assert float((scene.can.data.root_pos_w - scene.env_origins)[0, 2]) >= 0.28
    s2 = print_score("P2 can hoisted out through the open aperture")
    assert s2 >= s1 - 1e-6, "score decreased across evict"

    # ---------------- phase 3: BIN — transport teleport + gravity drop ---------------------
    # One root-state write carries the hovering can (free air above the chest) to free
    # air ABOVE the bin rim — outside the bin z-band, so the written state earns
    # nothing; the drop and settled containment set the latch.
    bin_pos = scene.bin.data.root_pos_w.clone()
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = bin_pos + torch.tensor([0.0, 0.0, 0.28], device=device)
    st[:, 3] = 1.0
    clear_can_force()
    scene.can.write_root_state_to_sim(st, all_ids)
    assert not bool(scene._binned_now()[0]), \
        "transport must not place the can inside the bin band"
    step(150)  # drop ~13 cm, clatter, settle
    report("binned")
    assert bool(scene._binned[0]) and bool(scene._binned_now()[0]), \
        "can did not settle inside the discard bin"
    s3 = print_score("P3 can dropped into the discard bin")
    assert s3 >= s2 - 1e-6, "score decreased across bin"

    # ---------------- phase 4: SEAT — transport teleport + gravity drop --------------------
    # One root-state write carries the bottle to a hover pose over the vacated well,
    # base 95 mm up — ABOVE the seat depth band, so the written state earns nothing;
    # gravity drops it 85 mm into the well where it self-centres and settles upright.
    from isaaclab.utils.math import quat_apply

    chest_pos = scene.chest.data.root_pos_w.clone()
    rel = torch.tensor([0.0, 0.0, 0.095 + c.body_h / 2], device=device).expand(n, 3)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = chest_pos + quat_apply(scene.chest.data.root_quat_w, rel)
    st[:, 3:7] = scene.chest.data.root_quat_w  # upright, chest-aligned (yaw cosmetic)
    scene.bottle.write_root_state_to_sim(st, all_ids)
    assert not bool(scene._seated_now()[0]), \
        "transport must not place the bottle inside the seat band"
    report("hover")
    step(150)  # drop into the well, bed down, settle
    report("seated")
    assert bool(scene._seated[0]) and bool(scene._seated_now()[0]), \
        "bottle did not seat upright in the well"
    assert not bool(scene.success()[0]), "cannot be success while the lid is open"
    s4 = print_score("P4 bottle dropped into the vacated well")
    assert s4 >= s3 - 1e-6, "score decreased across seat"

    # ---------------- phase 5: CLOSE — slide the lid back shut ------------------------------
    push_lid(0.006, -1.0, "close", v_des=0.08)
    # trim: if the lid rebounded off the back fence past tolerance, nudge it back
    for _ in range(3):
        if abs(float(scene._lid_u()[0])) <= c.lid_closed_tol - 0.002:
            break
        push_lid(0.006, -1.0, "close-trim", v_des=0.04, max_steps=240)
    step(60)
    report("closed")
    assert bool(scene._lid_closed_now()[0]), "lid did not close within tolerance"
    s5 = print_score("P5 lid slid shut over the seated bottle")
    assert s5 >= s4 - 1e-6, "score decreased across close"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after closing)", flush=True)
        os._exit(1)

    # ---------------- phase 6: persistence (>= 3 simulated seconds, no intervention) -------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s6 = print_score("P6 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s6 >= s5 - 1e-6
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
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001 - die loudly, don't idle to the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL (crash: {type(e).__name__})", flush=True)
        os._exit(1)

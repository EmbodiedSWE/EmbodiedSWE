"""Teleport solution for DropGateOvenScene (sim_gen task `close_microwave_i4`) —
the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): a single root-state write carries the BLUE can from its
   chamber slot to the free-space point 30 mm above the green pad, upright, zero
   velocity. The path is free space: the doorway is fully open while the gate is
   raised (aperture ~200 x 114 mm vs a 50 mm can), so the transport bypasses no
   contact interaction. The write satisfies no placement clause by itself: the can is
   airborne over the pad.
2. SEATING (gravity + contact): the can FALLS the last 30 mm onto the pad and settles.
   `blue_on_pad` (xy on the pad axis, correct rest height, upright, settled) is
   produced by contact, never written.
3. RELEASE UNDER LOAD (applied force + contact): the yellow prop post carries the
   gate's weight. A horizontal force at the post's CoM (oven-local +x, out of the
   doorway; ramping 2 -> 10 N) drags/topples it out from under the gate through
   friction contact — the exact strut-extraction a gripper pull would perform. The
   post is NEVER teleported.
4. CLOSURE (gravity + contact, hands-off): the instant the post clears, the gate
   free-falls ~115 mm in its slots and seats on the apron between the guides. Every
   closure fact the rubric checks (slot-plane position, seated height, uprightness,
   settled) is produced by ballistics and contact. If the gate wedges in the slots, a
   small escalating downward force at its CoM stands in for a fingertip tap —
   contact-consistent, cleared at once.
5. ORDER: blue can out FIRST, then the release — the task's forced order (a closed
   gate seals the chamber; smoke check 5 proves a 3x-weight shove cannot get a can
   out). The RED can is never touched.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.close_microwave_i4.solve --headless [--seed N]
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
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.drop_gate_oven")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import quat_apply

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def gate_z() -> float:
        return float(scene._oven_local(scene.gate.data.root_pos_w)[0, 2])

    def report(tag: str) -> None:
        gl = scene._oven_local(scene.gate.data.root_pos_w)[0]
        pl = scene._oven_local(scene.post.data.root_pos_w)[0]
        print(f"[solve] {tag:14s} | gate_loc=({float(gl[0]):+.3f},{float(gl[1]):+.3f},"
              f"{float(gl[2]):+.3f}) post_loc=({float(pl[0]):+.3f},{float(pl[1]):+.3f},"
              f"{float(pl[2]):+.3f}) blue_out={bool(scene.blue_outside()[0])} "
              f"on_pad={bool(scene.blue_on_pad()[0])} red_in={bool(scene.red_inside()[0])} "
              f"seated={bool(scene.gate_seated()[0])} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # gate settles onto the post, cans onto the floor
    ov = (scene.oven.data.root_pos_w - scene.env_origins)[0]
    oq = scene.oven.data.root_quat_w[0]
    oyaw = math.degrees(2.0 * math.atan2(float(oq[3]), float(oq[0])))
    pd = (scene.pad.data.root_pos_w - scene.env_origins)[0]
    post_loc = scene._oven_local(scene.post.data.root_pos_w)[0]
    blue_side = "+y" if float(scene.blue_slot[0]) > 0 else "-y"
    print(f"[solve] layout readback (seed {args.seed}): "
          f"oven=({float(ov[0]):+.3f},{float(ov[1]):+.3f}) yaw={oyaw:+.1f}deg "
          f"pad=({float(pd[0]):+.3f},{float(pd[1]):+.3f}) "
          f"post_y={float(post_loc[1]):+.3f} blue_slot={blue_side}", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert gate_z() > c.z_seat + 0.08, f"gate must start RAISED, z_loc={gate_z():.3f}"
    assert not bool(scene.blue_outside()[0]), "blue can must start inside the chamber"
    assert bool(scene.red_inside()[0]), "red can must start inside the chamber"
    s0 = print_score("P0 reset+settle (gate raised on the post, cans inside)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: blue can -> pad (transport, then gravity seats it) ----------
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.pad.data.root_pos_w
    st[:, 2] += c.pad_t / 2 + c.can_h / 2 + 0.030  # 30 mm free fall onto the pad
    st[:, 3] = 1.0
    scene.blue.write_root_state_to_sim(st, all_ids)
    step(150)  # fall + settle, hands-off
    report("blue->pad")
    assert bool(scene.blue_on_pad()[0]), "blue can must be seated upright on the pad"
    assert bool(scene.red_inside()[0]), "red can must remain inside"
    assert not bool(scene.success()[0]), "cannot be success while the gate is raised"
    s1 = print_score("P1 blue can seated on the green pad")
    assert s1 >= s0 - 1e-6 and s1 >= 0.44, f"P1 score {s1} (expect out+placed=0.45)"

    # ---------------- phase 2: pull the post out under load; gravity drops the gate --------
    fdir = quat_apply(scene.oven.data.root_quat_w,
                      torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))
    fmag = 2.0
    gate_nudge = 0.0
    post_clear_at = None
    seated = False
    for i in range(900):
        pl = scene._oven_local(scene.post.data.root_pos_w)[0]
        post_clear = float(pl[0]) > c.x_gate + c.gate_t / 2 + c.post_xy / 2 + 0.005 \
            or float(pl[2]) < c.zf + c.post_h / 2 - 0.03  # slid past or toppled off
        if not post_clear:
            f = (fmag * fdir).view(n, 1, 3)
            scene.post.set_external_force_and_torque(f, zero_wrench, env_ids=all_ids,
                                                     is_global=True)
            if i % 60 == 59:
                fmag = min(fmag + 1.0, 10.0)
                print(f"[solve] post still under the gate (x_loc={float(pl[0]):+.3f}); "
                      f"pull force -> {fmag:.0f} N", flush=True)
        else:
            if post_clear_at is None:
                post_clear_at = i
                scene.post.set_external_force_and_torque(
                    zero_wrench, zero_wrench, env_ids=all_ids)
                print(f"[solve] post clear of the gate at step {i} "
                      f"(x_loc={float(pl[0]):+.3f}, z_loc={float(pl[2]):+.3f}); "
                      f"gate falling", flush=True)
            elif i - post_clear_at >= 180 and (i - post_clear_at) % 90 == 0 \
                    and gate_z() > c.z_seat + c.gate_z_tol:
                gate_nudge = min(gate_nudge + 1.0, 4.0)  # wedged: fingertip-tap stand-in
                print(f"[solve] gate hangs at z_loc={gate_z():.3f}; "
                      f"downward nudge {gate_nudge:.1f} N", flush=True)
            if gate_nudge > 0.0:
                f = torch.zeros(n, 1, 3, device=device)
                f[:, 0, 2] = -gate_nudge
                scene.gate.set_external_force_and_torque(f, zero_wrench, env_ids=all_ids,
                                                         is_global=True)
        env.step(no_action)
        if post_clear and bool(scene.gate_seated()[0]) and bool(scene.settled()[0]):
            seated = True
            break
    scene.post.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    scene.gate.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    step(120)  # full settle, hands-off
    report("gate-drop")
    if not seated or not bool(scene.gate_seated()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (gate never seated)", flush=True)
        os._exit(1)
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the gate seated)", flush=True)
        os._exit(1)
    s2 = print_score("P2 post extracted under load; gate fell and sealed the doorway")
    assert s2 >= s1 - 1e-6, "score decreased across the release"

    # ---------------- phase 3: persistence (>= 3 simulated seconds, hands-off) -------------
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
    main()

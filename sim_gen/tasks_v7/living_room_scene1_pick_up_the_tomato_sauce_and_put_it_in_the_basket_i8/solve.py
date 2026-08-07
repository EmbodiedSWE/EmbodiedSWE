"""Teleport solution for SlideLidHamperScene (sim_gen task
`living_room_scene1_pick_up_the_tomato_sauce_and_put_it_in_the_basket_i8`) — the
task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. OPEN (contact dynamics, no teleport): the lid is captive in its guide channel —
   sliding it open IS the mechanism the task is about, so it is driven by a horizontal
   external force at its CoM (world frame, aligned with the hamper's slide axis,
   velocity-regulated bang-bang with stall escalation). Friction on the support plane,
   the rails, the lips and the end stop act on it the whole way. No pose write ever
   touches the lid.
2. TRANSPORT (teleport, the only pose write on the can): one root-state write carries
   the red can from its ground spawn across open air to a release pose 25 mm ABOVE the
   lid plane, centred over the EXPOSED part of the aperture — fully outside the
   containment volume, so the freshly-teleported state earns no insertion credit and
   cannot satisfy success(). Both endpoints are in free space.
3. INSERT (contact dynamics): gravity drops the can through the real aperture into the
   cavity; it impacts the hamper floor, topples/beds down and settles under physics.
   Had the lid been shut this exact drop would end ON the lid (smoke check #6) — the
   open channel is what the drop passes through.
4. CLOSE (contact dynamics): the same force scheme drives the lid back until it seats
   against the closed stop within `closed_tol`. success() first turns True only here,
   judged on the settled physical state (can inside below the lid plane, lid shut,
   everything at rest).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.living_room_scene1_pick_up_the_tomato_sauce_and_put_it_in_the_basket_i8.solve --headless [--seed N]
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
    env = ENVS.get("simgen.slidelid_hamper")().build(num_envs=args.num_envs, device=device)
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

    def hamper_pose() -> tuple[torch.Tensor, float]:
        hp = (scene.hamper.data.root_pos_w - scene.env_origins)[0]
        q = scene.hamper.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return hp, yaw

    def report(tag: str) -> None:
        x_l = float(scene._lid_travel()[0])
        tl = scene._local(scene.tomato)[0]
        print(f"[solve] {tag:12s} | lid_x={x_l:+.3f} tomato_local=({float(tl[0]):+.3f},"
              f"{float(tl[1]):+.3f},{float(tl[2]):.3f}) open_latch={float(scene._open_max[0]):.3f} "
              f"in={bool(scene._in[0])} close_latch={float(scene._close_max[0]):.3f} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_force() -> None:
        scene.lid.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def slide_lid(target_x: float, direction: float, tag: str,
                  v_des: float = 0.10, max_steps: int = 1800) -> None:
        """Drive the lid along the channel with a world-frame horizontal force at its
        CoM (velocity-regulated bang-bang, stall escalation). `direction` is +1 to
        open, -1 to close; stops once the lid's local x passes `target_x` in that
        direction. Contact dynamics only — no pose write ever touches the lid."""
        hp, hyaw = hamper_pose()
        push_dir = torch.tensor([math.cos(hyaw), math.sin(hyaw), 0.0], device=device) * direction
        f_push = 4.0
        best, last_bump = -1e9, 0
        for i in range(max_steps):
            x_l = float(scene._lid_travel()[0])
            if (x_l - target_x) * direction > 0.0:
                break
            v_axis = float((scene.lid.data.root_lin_vel_w[0] * push_dir).sum())
            f_axis = f_push if v_axis < v_des else 0.0
            f_w = (push_dir * f_axis).view(1, 1, 3).expand(n, 1, 3)
            scene.lid.set_external_force_and_torque(f_w.contiguous(), zero_wrench,
                                                    env_ids=all_ids, is_global=True)
            env.step(no_action)
            prog = x_l * direction
            if prog > best + 0.004:
                best, last_bump = prog, i
            elif i - last_bump > 240:  # stalled: push harder (friction was underestimated)
                f_push = min(f_push + 2.0, 12.0)
                last_bump = i
                print(f"[solve] {tag}: lid stalled at x_local={x_l:+.3f}, raising force "
                      f"to {f_push:.1f} N", flush=True)
        clear_force()
        step(40)  # coast + settle on real friction

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    hp, hyaw = hamper_pose()
    tom0 = (scene.tomato.data.root_pos_w - scene.env_origins)[0]
    dis0 = (scene.distractor.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): hamper=({float(hp[0]):+.3f},"
          f"{float(hp[1]):+.3f}) yaw={math.degrees(hyaw):+.1f}deg "
          f"tomato=({float(tom0[0]):+.3f},{float(tom0[1]):+.3f}) "
          f"distractor=({float(dis0[0]):+.3f},{float(dis0[1]):+.3f}) "
          f"lid_x={float(scene._lid_travel()[0]):+.4f}", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: OPEN the lid through contact dynamics -----------------------
    # The captive lid is dragged open along its rails by a world-frame force at its CoM
    # until its travel exposes the full can-sized aperture (x_local >= open_ref - 5 mm).
    slide_lid(c.open_ref - 0.005, +1.0, "open")
    report("opened")
    assert float(scene._open_max[0]) > 0.9, "lid did not open under the slide force"
    s1 = print_score("P1 contact-dynamics lid OPEN")
    assert s1 >= s0 - 1e-6, "score decreased across open"

    # ---------------- phase 2: TRANSPORT (teleport across free air only) -------------------
    # One pose write carries the red can from the ground to a release pose 25 mm ABOVE
    # the lid plane, over the exposed aperture (hamper-local (-0.02, 0)). It is fully
    # OUTSIDE the containment volume: no insertion credit, success() unreachable here.
    hp, hyaw = hamper_pose()
    drop_local = torch.tensor([-0.02, 0.0], device=device)
    wx = float(hp[0]) + math.cos(hyaw) * float(drop_local[0]) - math.sin(hyaw) * float(drop_local[1])
    wy = float(hp[1]) + math.sin(hyaw) * float(drop_local[0]) + math.cos(hyaw) * float(drop_local[1])
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1] = wx, wy
    st[:, 2] = c.z_top + 0.025 + c.can_h / 2  # base 25 mm above the lid plane — not inside
    st[:, 3] = 1.0
    st[:, 0:3] += scene.env_origins
    scene.tomato.write_root_state_to_sim(st, all_ids)
    report("transported")
    assert not bool(scene._in[0]), "transport must not place the can inside the cavity"
    s2 = print_score("P2 transport to release pose above the open aperture")
    assert s2 >= s1 - 1e-6, "score decreased across transport"

    # ---------------- phase 3: INSERT through gravity + contact -----------------------------
    step(150)  # 1.25 s: fall through the aperture, impact the floor, settle
    report("inserted")
    assert bool(scene._in[0]), "can did not come to rest inside the cavity"
    s3 = print_score("P3 gravity insertion through the open aperture")
    assert s3 >= s2 - 1e-6, "score decreased across insertion"

    # ---------------- phase 4: CLOSE the lid through contact dynamics ----------------------
    slide_lid(0.008, -1.0, "close", v_des=0.08)
    # trim: if the lid rebounded off the closed stop past tolerance, nudge it back
    for _ in range(3):
        if abs(float(scene._lid_travel()[0])) <= c.closed_tol - 0.002:
            break
        slide_lid(0.006, -1.0, "close-trim", v_des=0.04, max_steps=240)
    report("closed")
    s4 = print_score("P4 contact-dynamics lid CLOSE")
    assert s4 >= s3 - 1e-6, "score decreased across close"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after close)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 5: persistence (>= 3 simulated seconds, no intervention) -------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s5 = print_score("P5 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s5 >= s4 - 1e-6
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

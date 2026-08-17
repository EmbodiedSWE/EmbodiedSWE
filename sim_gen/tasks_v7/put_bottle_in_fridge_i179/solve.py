"""Teleport solution for HangChillerScene (sim_gen task `put_bottle_in_fridge_i179`) —
the task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, the only pose write on the bottle): one root-state write
   carries the green bottle, upright, from its ground spawn across open air to a hover
   pose over the rail's loading tongue — neck centred in the slot, cap flange 8 mm
   ABOVE the bar tops. That is above the seated z-band, so the freshly-teleported
   state is NOT "hung" and earns no credit; both endpoints are free space (the neck
   sits in the slot gap with clearance, the body hangs 30 mm below the bars).
2. SEAT (contact dynamics): gravity drops the bottle the last 8 mm; the cap flange
   lands on the two bars and the bottle HANGS — the `hung` latch first sets here,
   from a real resting flange-on-bars contact with the body suspended in free air.
3. SLIDE (contact dynamics, no teleport): the hanging bottle is dragged rearward
   along the slot by a horizontal external force at its CoM (world frame, aligned
   with the cabinet's inward axis, velocity-regulated bang-bang with stall
   escalation) until the cap is >= 0.15 m behind the fascia — through the front
   opening, under the roof, into the cold zone. The flange must physically stay
   seated on the bars for the whole ride: if the bottle dropped off, success() could
   never turn True (a fell-off guard aborts loudly). No pose write ever touches the
   rail or the cap contact.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.put_bottle_in_fridge_i179.solve --headless [--seed N]
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


def _die(code: int) -> None:
    """Hard exit: Kit teardown hangs — arm a timer, try to close, then exit."""
    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
    os._exit(code)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.hang_chiller")().build(num_envs=args.num_envs, device=device)
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

    def cab_yaw() -> float:
        q = scene.cabinet.data.root_quat_w[0]
        return 2.0 * math.atan2(float(q[3]), float(q[0]))

    def cap_rail(idx: int = 0) -> torch.Tensor:
        """Cap-flange centre in the RAIL frame (x along the slot, z=0 at bar tops)."""
        return scene._in_frame(scene._cap_center_w(), scene.rail)[idx]

    def report(tag: str) -> None:
        cl = cap_rail()
        d = float(scene._cap_depth()[0])
        print(f"[solve] {tag:12s} | cap_rail=({float(cl[0]):+.3f},{float(cl[1]):+.3f},"
              f"{float(cl[2]):+.3f}) depth={d:+.3f} hung_now={bool(scene._hung_now()[0])} "
              f"hung={bool(scene._hung[0])} depth_max={float(scene._depth_max[0]):.3f} "
              f"up={float(scene._bottle_up()[0]):.4f} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_force() -> None:
        scene.bottle.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def drag_bottle(target_depth: float, tag: str, v_des: float = 0.06,
                    max_steps: int = 2400) -> None:
        """Drag the hanging bottle rearward along the slot with a world-frame
        horizontal force at its CoM (velocity-regulated bang-bang, stall escalation)
        until the cap depth exceeds `target_depth`. Contact dynamics only — the cap
        flange rides the bars through gravity + friction; no pose write."""
        yaw = cab_yaw()
        # cabinet local -x (inward) expressed in world
        push_dir = torch.tensor([-math.cos(yaw), -math.sin(yaw), 0.0], device=device)
        f_push = 1.5
        best, last_bump = -1e9, 0
        for i in range(max_steps):
            d = float(scene._cap_depth()[0])
            if d >= target_depth:
                break
            if float(cap_rail()[2]) < -0.06:  # fell off the rail: unrecoverable
                clear_force()
                report("FELL-OFF")
                print(f"SIM_GEN_SOLVE: FAIL ({tag}: bottle fell off the rail)", flush=True)
                _die(2)
            v_axis = float((scene.bottle.data.root_lin_vel_w[0] * push_dir).sum())
            f_axis = f_push if v_axis < v_des else 0.0
            f_w = (push_dir * f_axis).view(1, 1, 3).expand(n, 1, 3)
            scene.bottle.set_external_force_and_torque(f_w.contiguous(), zero_wrench,
                                                       env_ids=all_ids, is_global=True)
            env.step(no_action)
            if d > best + 0.004:
                best, last_bump = d, i
            elif i - last_bump > 240:  # stalled: push harder (friction underestimated)
                f_push = min(f_push + 1.0, 5.0)
                last_bump = i
                print(f"[solve] {tag}: stalled at depth={d:+.3f}, raising force to "
                      f"{f_push:.1f} N", flush=True)
        clear_force()
        # let the pendulum swing die out on real damping + friction
        for _ in range(15):  # up to 5 s
            step(40)
            if bool(scene.settled()[0]):
                break

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    yaw = cab_yaw()
    cab = (scene.cabinet.data.root_pos_w - scene.env_origins)[0]
    bot0 = (scene.bottle.data.root_pos_w - scene.env_origins)[0]
    can0 = (scene.can.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): cabinet=({float(cab[0]):+.3f},"
          f"{float(cab[1]):+.3f}) yaw={math.degrees(yaw):+.1f}deg "
          f"rail_y_off={float(scene._y_off[0]):+.3f} "
          f"bottle=({float(bot0[0]):+.3f},{float(bot0[1]):+.3f}) "
          f"can=({float(can0[0]):+.3f},{float(can0[1]):+.3f})", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 < 0.05, f"fresh reset should score ~0, got {s0}"

    # ---------------- phase 1: TRANSPORT (teleport across free air only) -------------------
    # One pose write carries the bottle, upright, to a hover pose over the loading
    # tongue: neck centred in the slot gap, cap flange 8 mm ABOVE the bar tops (cap
    # centre z = +0.015 in the rail frame — above the seated band, so the freshly-
    # written state is NOT hung and earns no credit). Both endpoints are free space.
    from isaaclab.utils.math import quat_apply

    hover_z = 0.008 + c.cap_t / 2 - c.cap_off       # bottle origin z rel rail top
    rel = torch.tensor([0.020, 0.0, hover_z], device=device).expand(n, 3)
    target = scene.rail.data.root_pos_w + quat_apply(scene.rail.data.root_quat_w, rel)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = target
    st[:, 3] = 1.0  # upright, identity yaw (the bottle is a solid of revolution)
    scene.bottle.write_root_state_to_sim(st, all_ids)
    assert not bool(scene._hung_now()[0]), \
        "transport must not place the cap inside the seated band"
    report("transported")
    s1 = print_score("P1 transport to hover over the loading tongue")
    assert s1 >= s0 - 1e-6, "score decreased across transport"
    assert s1 < 0.05, f"hover pose must earn nothing, got {s1}"

    # ---------------- phase 2: SEAT through gravity + contact -------------------------------
    step(120)  # 1 s: drop 8 mm, flange lands on the bars, pendulum settles
    report("seated")
    assert bool(scene._hung[0]), "cap flange did not seat on the rail bars"
    assert bool(scene._hung_now()[0]), "bottle is not hanging after the drop"
    assert not bool(scene.success()[0]), "cannot be success out on the tongue"
    s2 = print_score("P2 gravity seat onto the bars (bottle hangs)")
    assert s2 >= s1 - 1e-6, "score decreased across seating"
    assert s2 >= 0.25, f"hung latch should be worth ~0.30, got {s2}"

    # ---------------- phase 3: SLIDE — drag the hanging bottle into the cold zone ----------
    drag_bottle(0.15, "slide")
    report("chilled")
    assert float(scene._cap_depth()[0]) >= c.chill_depth, \
        "cap did not reach the cold zone"
    assert bool(scene._hung_now()[0]), "bottle not hanging after the slide"
    s3 = print_score("P3 contact-dynamics slide into the cold zone")
    assert s3 >= s2 - 1e-6, "score decreased across the slide"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the slide)", flush=True)
        _die(1)

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
    _die(0 if ok else 1)


if __name__ == "__main__":
    main()

"""Teleport solution for ChuteSwitchScene (sim_gen task
`libero_kitchen_scene1_open_top_drawer_i14`) — the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. SWITCHING (applied force + contact): the gate is never posed. Each reconfiguration
   applies a horizontal force along the bar's own axis (what a fingertip on the
   exposed tail would do); the bar slides through its wall slots and is ARRESTED by
   the end-stop post — contact decides where it parks. At least one mid-task flip is
   mandatory (the two colors' channels share the one gate).
2. TRANSPORT (teleport): each cargo ball is carried by a single root-state write from
   its ring slot to a free-space hover ~50 mm above the OPEN-TOP INLET of the chosen
   channel (in the router's live body frame), zero velocity. The write satisfies no
   rubric clause by itself: the ball is airborne at the very top of the chute, two
   full channel-lengths from any bin.
3. ROUTING (gravity + contact, hands-off): the ball falls onto the inlet floor and
   ROLLS the length of the channel, past the switch station (open side), off the
   outlet edge, through the bin mouth, onto the bin floor — every centimetre of that
   path is decided by the chute's geometry and the gate's parked position, not by the
   solver. Delivery latches fire only for balls settled INSIDE the correct bin.
4. IDENTITY: the BLUE decoy ball is never touched; the two greens go through the
   green-routed channel and the red through the red-routed one, exactly as the
   instruction demands.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: delivery
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene1_open_top_drawer_i14.solve
             --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.chute_switch")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        ig = scene.balls_in_bin("green")[0]
        ir = scene.balls_in_bin("red")[0]
        print(f"[solve] {tag:12s} | gate_side={int(scene.gate_side()[0]):+d} "
              f"gate_y={float(scene.gate_y()[0]):+.3f} "
              f"green_bin={[i for i in range(4) if bool(ig[i])]} "
              f"red_bin={[i for i in range(4) if bool(ir[i])]} "
              f"seated={bool(scene.bins_seated()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def push_dir() -> torch.Tensor:
        """World direction of the router's +y axis (horizontal: pitch is about y)."""
        ey = torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3)
        return quat_apply(scene.router.data.root_quat_w, ey)

    flips = 0

    def set_gate(target: int) -> None:
        """SWITCHING: park the gate on `target` (+1 blocks LEFT, -1 blocks RIGHT) by
        applying a force along the bar axis — the end-stop post arrests it. No-op if
        already parked there."""
        nonlocal flips
        if int(scene.gate_side()[0]) == target:
            return
        flips += 1
        zero = torch.zeros(n, 1, 3, device=device)
        yt = target * (c.gate_park + 0.006)   # aim just past the park, into the post
        ok_frames = 0
        for _ in range(480):
            # PD force along the bar axis (fingertip guiding the bar): ramming the
            # end post at full force wedges the tip into it, so brake on approach.
            axis = push_dir()
            v = (scene.gate.data.root_lin_vel_w * axis).sum(-1)
            f_mag = (150.0 * (yt - scene.gate_y()) - 12.0 * v).clamp(-5.0, 5.0)
            fw = axis * f_mag.unsqueeze(-1)
            # set_external_force_and_torque takes BODY-frame forces: convert each step.
            fb = quat_apply_inverse(scene.gate.data.root_quat_w, fw)
            scene.gate.set_external_force_and_torque(fb.reshape(n, 1, 3), zero)
            env.step(no_action)
            parked = int(scene.gate_side()[0]) == target and abs(float(v[0])) < 0.02
            ok_frames = ok_frames + 1 if parked else 0
            if ok_frames >= 5:
                break
        scene.gate.set_external_force_and_torque(zero, zero)
        step(60)
        assert int(scene.gate_side()[0]) == target, \
            f"gate failed to park at side {target:+d} (y={float(scene.gate_y()[0]):+.3f})"

    def side_of(bin_name: str) -> int:
        """Which channel (+1 LEFT / -1 RIGHT) feeds `bin_name` this episode."""
        swapped = bool(scene.bins_swapped[0])
        green_left = not swapped
        return (1 if green_left else -1) if bin_name == "green" else \
            (-1 if green_left else 1)

    def deliver(ball: str, bin_name: str, idx: int) -> None:
        """TRANSPORT then ROUTING: hover the ball over the open inlet of the channel
        that feeds `bin_name` (gate must already open that channel), release, and
        wait hands-off until it settles inside the bin."""
        side = side_of(bin_name)
        assert int(scene.gate_side()[0]) == -side, "gate must open the target channel"
        local = torch.tensor([-0.24, side * c.ch_off, c.ball_r + 0.050],
                             device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.router.data.root_pos_w + quat_apply(
            scene.router.data.root_quat_w, local)
        st[:, 3] = 1.0
        scene.balls[ball].write_root_state_to_sim(st, torch.arange(n, device=device))
        for _ in range(600):
            env.step(no_action)
            if bool(scene.balls_in_bin(bin_name)[0, idx]) and bool(scene.settled()[0]):
                break
        step(60)
        assert bool(scene.balls_in_bin(bin_name)[0, idx]), \
            f"{ball} did not settle inside the {bin_name} bin"

    # ---------------- phase 0: settle, layout readback, baseline ---------------------------------
    step(120)  # gate settles onto the inclined floor against its post
    rp = (scene.router.data.root_pos_w - scene.env_origins)[0]
    rq = scene.router.data.root_quat_w[0]
    ryaw = math.degrees(2.0 * math.atan2(float(rq[3]), float(rq[0])))
    balls_xy = {nm: (scene.balls[nm].data.root_pos_w - scene.env_origins)[0]
                for nm in scene.balls}
    ball_str = " ".join(f"{nm}=({float(p[0]):+.2f},{float(p[1]):+.2f})"
                        for nm, p in balls_xy.items())
    print(f"[solve] layout readback (seed {args.seed}): "
          f"router=({float(rp[0]):+.3f},{float(rp[1]):+.3f}) yaw~{ryaw:+.1f}deg "
          f"swapped={bool(scene.bins_swapped[0])} "
          f"gate_init_left={bool(scene.gate_init_left[0])} "
          f"perm={scene.slot_perm[0].tolist()} {ball_str}", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert int(scene.gate_side()[0]) != 0, "gate must start parked at a stop"
    assert bool(scene.bins_seated()[0]), "bins must start seated at the outlets"
    s0 = print_score("P0 reset+settle (gate parked, bins mated, balls on the ring)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: switch for green, deliver green_0 ---------------------------------
    set_gate(-side_of("green"))
    deliver("green_0", "green", 0)
    report("green-1")
    s1 = print_score("P1 gate set for the green channel; first green ball delivered")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_route + c.w_del - 1e-6, f"P1 score {s1}"

    # ---------------- phase 2: deliver green_1 (same route) --------------------------------------
    deliver("green_1", "green", 1)
    report("green-2")
    s2 = print_score("P2 second green ball delivered to the green bin")
    assert s2 >= s1 - 1e-6 and s2 >= c.w_route + 2 * c.w_del - 1e-6, f"P2 score {s2}"

    # ---------------- phase 3: MANDATORY flip, deliver red_0 -------------------------------------
    set_gate(-side_of("red"))
    assert flips >= 1, "the shared switch forces at least one reconfiguration"
    deliver("red_0", "red", 2)
    report("red")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (deliveries complete but success() is False)",
              flush=True)
        os._exit(1)
    s3 = print_score("P3 gate flipped; red ball delivered to the red bin")
    assert s3 >= s2 - 1e-6, "score decreased across the final delivery"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, hands-off) -------------------
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
    except Exception as e:  # noqa: BLE001 - fast fail beats a 20-min watchdog hang
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)

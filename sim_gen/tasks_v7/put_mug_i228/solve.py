"""Teleport solution for MugDispenserScene (sim_gen task `put_mug_i228`) — the
task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, the only pose write on the mug): one root-state write
   carries the mug from its ground spawn across open air to the STAGED pose —
   upright on the rig's base plate, centred under the silo bore, handle turned
   along the rig's clear side. Both endpoints are contact-free rests; this is what
   a carry does. The written state cannot satisfy success(): the balls are still
   sealed in the silo, and the pose jump resets the stillness streak anyway.
2. PULL THE GATE (contact dynamics, no teleport): a horizontal external force at
   the gate's CoM (velocity-regulated bang-bang along the rig's channel direction,
   stall escalation) drags the captive plate outward through its channel, under
   the resting ball column, until the bore is uncovered. The balls ride the
   retracting plate, drop through the bore, fall ~7 cm and are CAUGHT by the mug —
   gravity + contact all the way; nothing ever writes a ball pose after reset.
3. SETTLE (hands-off): forces cleared; the balls rattle to rest inside the mug and
   the stillness streak accumulates to success().

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.put_mug_i228.solve --headless [--seed N]
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
import traceback

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)",
                                             flush=True), os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.mug_dispenser")().build(num_envs=args.num_envs,
                                                   device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback
    # so distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    from isaaclab.utils.math import quat_apply

    def report(tag: str) -> None:
        gz = float(scene.gate_open_x()[0])
        bin_ = scene.balls_in()[0]
        bz = [float((b.data.root_pos_w - scene.env_origins)[0, 2])
              for b in scene.balls]
        print(f"[solve] {tag:12s} | mug_z={float(scene._mug_z()[0]):.3f} "
              f"gate_x={gz:.3f} staged={bool(scene.staged_now()[0])} "
              f"balls_in={int(bin_.sum())} ball_z={['%.3f' % v for v in bz]} "
              f"still={int(scene._still[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    last_score = [0.0]

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        assert s >= last_score[0] - 1e-6, f"score decreased at {tag}"
        last_score[0] = s
        return s

    def clear_force() -> None:
        scene.gate.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                 env_ids=all_ids)

    # ---------------- phase 0: reset, settle, baseline --------------------------------------
    step(60)
    m_mug = float(scene.mug.root_physx_view.get_masses().reshape(-1)[0])
    m_gate = float(scene.gate.root_physx_view.get_masses().reshape(-1)[0])
    m_ball = float(scene.balls[0].root_physx_view.get_masses().reshape(-1)[0])
    assert abs(m_mug - c.mug_mass) < 0.02, f"mug mass not applied ({m_mug})"
    assert abs(m_gate - c.gate_mass) < 0.01, f"gate mass not applied ({m_gate})"
    assert abs(m_ball - c.ball_mass) < 0.005, f"ball mass not applied ({m_ball})"
    rp = (scene.rig.data.root_pos_w - scene.env_origins)[0]
    rq = scene.rig.data.root_quat_w[0]
    ryaw = 2.0 * math.atan2(float(rq[3]), float(rq[0]))
    mp = (scene.mug.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): rig=({float(rp[0]):+.3f},"
          f"{float(rp[1]):+.3f}) yaw={math.degrees(ryaw):+.1f}deg "
          f"mug=({float(mp[0]):+.3f},{float(mp[1]):+.3f},{float(mp[2]):.3f}) "
          f"gate_x={float(scene.gate_open_x()[0]):.3f} "
          f"masses mug={m_mug:.3f} gate={m_gate:.3f} ball={m_ball:.3f}", flush=True)
    report("reset")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert abs(float(scene.gate_open_x()[0]) - c.gate_closed_x) < 0.006, \
        "gate did not settle at the closed pose"
    print_score("P0 reset+settle")

    # ---------------- phase 1: TRANSPORT (teleport across free air only) --------------------
    # One pose write: the mug staged upright on the base plate, centred under the
    # bore, handle along rig local +y (the clear side between the leg pairs). The
    # balls stay sealed in the silo — nothing about this state is success.
    rig_p = scene.rig.data.root_pos_w[0]
    stage_l = torch.tensor([0.0, 0.0, c.base_t + c.body_h / 2 + 0.002],
                           device=device)
    pos = rig_p + quat_apply(rq.unsqueeze(0), stage_l.unsqueeze(0))[0]
    myaw = ryaw + math.pi / 2
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = pos.unsqueeze(0)
    st[:, 3] = math.cos(myaw / 2)
    st[:, 6] = math.sin(myaw / 2)
    scene.mug.write_root_state_to_sim(st, all_ids)
    step(60)  # settle onto the base plate; the staged latch fires on real frames
    assert bool(scene.staged_now()[0]), "mug failed to stage under the outlet"
    assert not bool(scene.success()[0]), "staged empty mug must not be success"
    report("staged")
    print_score("P1 transport: mug staged under the silo outlet")

    # ---------------- phase 2: PULL THE GATE (external force, contact) ----------------------
    # Drag the captive plate outward along the channel: horizontal velocity-
    # regulated bang-bang force at the CoM along the rig's +x (world-encoded), with
    # stall escalation. The ball column rides the plate and drops through the bore.
    pull_dir = quat_apply(rq.unsqueeze(0),
                          torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
    pull_dir[2] = 0.0
    pull_dir = pull_dir / pull_dir.norm()
    target_x = 0.058
    f_mag, v_des = 1.5, 0.05
    best, last_bump = float(scene.gate_open_x()[0]), 0
    for i in range(2400):
        gx = float(scene.gate_open_x()[0])
        if gx >= target_x:
            break
        v_along = float((scene.gate.data.root_lin_vel_w[0] * pull_dir).sum())
        f_des = pull_dir * (f_mag if v_along < v_des else 0.0)
        f_w = f_des.view(1, 1, 3).expand(n, 1, 3)
        scene.gate.set_external_force_and_torque(f_w.contiguous(), zero_wrench,
                                                 env_ids=all_ids, is_global=True)
        env.step(no_action)
        if gx > best + 0.004:
            best, last_bump = gx, i
        elif i - last_bump > 240:  # stalled: pull harder
            f_mag = min(f_mag + 1.5, 8.0)
            last_bump = i
            print(f"[solve] pull stalled at gate_x={gx:.3f}, raising force to "
                  f"{f_mag:.1f} N", flush=True)
    clear_force()
    gx = float(scene.gate_open_x()[0])
    assert gx >= target_x - 0.004, f"gate did not open (gate_x={gx:.3f})"
    assert bool(scene._opened[0]), "opened latch did not fire"
    report("opened")
    print_score("P2 gate pulled open through the channel (balls dropping)")

    # ---------------- phase 3: hands-off settle to success ----------------------------------
    ok_settle = False
    for _ in range(40):  # up to 10 s
        step(30)
        if bool(scene.success()[0]):
            ok_settle = True
            break
    report("settled")
    if not ok_settle:
        print("SIM_GEN_SOLVE: FAIL (no success after settle)", flush=True)
        os._exit(1)
    assert int(scene.balls_in()[0].sum()) == c.n_balls, "not all balls in the mug"
    print_score("P3 balls caught by the mug, settled to success")

    # ---------------- phase 4: persistence (>= 3 simulated seconds, no intervention) --------
    hold = True
    for _ in range(10):  # 10 x 42 steps = 420 substeps = 3.5 s at 120 Hz
        step(42)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.5 s")
    ok = hold and bool(scene.success()[0]) and s4 >= 1.0 - 1e-6
    if ok:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)

    code = 0 if ok else 1
    # Hard exit: Kit teardown hangs — watchdog then die.
    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)

"""Teleport solution for RockerLockScene (sim_gen task `libero_pick_ketchup_i104`) —
the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY. Exactly one teleport of one free body:
  (a) the RED ketchup bottle, from its floor spawn to the open LOADING BAY of the
      rocker's tray — laid across the channel against the end wall, zero velocity.
      The loading bay sits at the LOW end of the parked beam under open sky: this is
      precisely what a pick-and-carry delivers. The bin itself is never approached —
      it is sealed, and the closed gate's every aperture undercuts the bottle
      (asserted in the scene config), so no teleport into the bin is ever attempted.
Every load-bearing interaction happens through contact dynamics and applied wrenches
(a proxy for a hand pressing the green paddle):

  press   — a downward force at the paddle (emulated as F at the body plus the
            paddle-lever torque tau = (R_beam @ r_paddle) x F, ramped until the beam
            starts pitching) rotates the see-saw on its pure-contact journal; the
            beam settles onto its press stop, its tip sinks out of the letterbox
            window, and the tray becomes a ramp.
  ferry   — HANDS DO NOT TOUCH THE BOTTLE: gravity rolls it down the tilted channel,
            over the tip and through the open window into the sealed bin. Every
            newton the bottle feels comes from gravity and tray contact.
  release — forces cleared; the beam's authored mass bias re-parks it against the
            rest stop, plugging the window and SEALING the bottle inside.

All wrenches go through `encode_force` with a RUNTIME force-frame probe (some pods
rotate applied wrenches by the body's rotation since a reference orientation): the
press is attempted in mode 1 (pre-encoded against the press-start readback); if the
beam does not pitch the state is rolled back and mode 0 is tried. (The beam only ever
rotates ~15 degrees, so both modes are near-identical here — the probe is belt and
braces, not load-bearing.)

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing, latched by the
scene: 0 -> 0.15 loaded -> 0.75 opened+delivered -> 1.0 sealed), then holds HANDS-OFF
for >= 3 simulated seconds after success() first turns True and prints
`SIM_GEN_SOLVE: SUCCESS` only if it still holds.

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
    from .scene import (BOT_R, FLOOR_TOP, TILT_PRESS, TILT_REST, _qapply, _qmul, _qx,
                        encode_force)
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import (BOT_R, FLOOR_TOP, TILT_PRESS, TILT_REST, _qapply, _qmul, _qx,
                       encode_force)
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

LOAD_X = -0.25  # beam-frame x of the loading bay (bottle rests against the end wall)
PADDLE_GRIP = (0.100, 0.1175, 0.060)  # beam-frame lever arm of the paddle press point
F_START, F_RATE, F_MAX = 4.0, 0.10, 34.0  # press ramp: N, N/step (12 N/s), cap
OPEN_HOLD_DEG = TILT_PRESS - 0.5  # freeze the ramp once tilt <= -6.0 deg


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.rocker_lock")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the statement on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def clear_forces() -> None:
        scene.beam.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                 env_ids=all_ids)

    def report(tag: str) -> None:
        tilt = float(scene.beam_tilt_deg()[0])
        kloc = scene._rig_local(scene.ketchup.data.root_pos_w)[0]
        print(f"[solve] {tag:12s} | tilt={tilt:+.2f}deg "
              f"seated={bool(scene.axle_seated()[0])} "
              f"closed={bool(scene.gate_closed()[0])} "
              f"k_rig=({float(kloc[0]):+.3f},{float(kloc[1]):+.3f},"
              f"{float(kloc[2]):+.3f}) "
              f"tray={bool(scene.in_tray(scene.ketchup)[0])} "
              f"bin={bool(scene.in_bin(scene.ketchup)[0])} "
              f"mus_out={bool(scene.mustard_out()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def wait_settled(bodies, max_steps: int = 600) -> None:
        for _ in range(max_steps // 30):
            step(30)
            if all(bool(scene.settled(b)[0]) for b in bodies):
                break

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    wait_settled([scene.beam, scene.ketchup, scene.mustard], 360)
    rig_p = (scene.rig.data.root_pos_w - scene.env_origins)[0]
    yaw = 2.0 * math.atan2(float(scene.rig.data.root_quat_w[0, 3]),
                           float(scene.rig.data.root_quat_w[0, 0]))
    k_loc = scene._rig_local(scene.ketchup.data.root_pos_w)[0]
    m_loc = scene._rig_local(scene.mustard.data.root_pos_w)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"rig=({float(rig_p[0]):+.3f},{float(rig_p[1]):+.3f}) "
          f"yaw={math.degrees(yaw):+.1f}deg "
          f"tilt={float(scene.beam_tilt_deg()[0]):+.2f}deg "
          f"ketchup=({float(k_loc[0]):+.3f},{float(k_loc[1]):+.3f}) "
          f"mustard=({float(m_loc[0]):+.3f},{float(m_loc[1]):+.3f})", flush=True)
    report("reset")
    tilt0 = float(scene.beam_tilt_deg()[0])
    assert TILT_REST - 1.5 <= tilt0 <= TILT_REST + 1.0, \
        f"beam must park on its rest stop, tilt={tilt0}"
    assert bool(scene.gate_closed()[0]), "gate must start closed"
    assert bool(scene.axle_seated()[0]), "axle must start seated"
    assert not bool(scene.in_tray(scene.ketchup)[0]), "ketchup must start on the floor"
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    s0 = print_score("P0 reset+settle (gate plugged, bottles on the floor)")
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: TRANSPORT ONLY — lay the ketchup into the loading bay -------
    def teleport_load() -> None:
        """Lay the RED bottle across the tray channel at the low loading end (open
        sky above it; a plain pick-and-carry). Zero velocity; gravity seats it."""
        loc = torch.tensor([LOAD_X, 0.0, FLOOR_TOP + BOT_R + 0.006],
                           device=device).expand(n, 3)
        q_beam = scene.beam.data.root_quat_w
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.beam.data.root_pos_w + _qapply(q_beam, loc)
        half_pi = torch.full((n,), math.pi / 2, device=device)
        st[:, 3:7] = _qmul(q_beam, _qx(half_pi))  # cylinder axis across the channel
        scene.ketchup.write_root_state_to_sim(st, all_ids)

    teleport_load()
    wait_settled([scene.ketchup, scene.beam], 300)
    report("loaded")
    assert bool(scene.in_tray(scene.ketchup)[0]), "ketchup must rest in the tray"
    assert bool(scene.gate_closed()[0]), "loading must not open the gate"
    s1 = print_score("P1 ketchup laid into the loading bay (teleport transport, "
                     "zero velocity)")
    assert s1 >= 0.15 - 1e-6 and s1 >= s0 - 1e-6, "loaded credit missing"

    # ---------------- phase 2: PRESS the paddle — the machine ferries the bottle -----------
    r_paddle = torch.tensor(PADDLE_GRIP, device=device).expand(n, 3)

    def press_wrench(mode: int, q_ref: torch.Tensor, force_n: float) -> None:
        """One step of the emulated hand: downward force at the paddle = F at the
        body + the paddle-lever torque about the body frame, world-encoded."""
        q_now = scene.beam.data.root_quat_w
        f_w = torch.zeros(n, 3, device=device)
        f_w[:, 2] = -force_n
        t_w = torch.cross(_qapply(q_now, r_paddle), f_w, dim=-1)
        f_arg = encode_force(mode, q_ref, q_now, f_w)
        t_arg = encode_force(mode, q_ref, q_now, t_w)
        scene.beam.set_external_force_and_torque(
            f_arg.view(n, 1, 3), t_arg.view(n, 1, 3), env_ids=all_ids, is_global=True)

    def press_and_hold(mode: int, q_ref: torch.Tensor, tag: str) -> bool:
        """Ramp the press until the beam pitches onto its press stop, then HOLD while
        gravity rolls the bottle down the ramp and through the window. True once the
        delivery latch fires (bottle inside the bin volume)."""
        force = F_START
        frozen = None
        for i in range(1800):
            tilt = float(scene.beam_tilt_deg()[0])
            if frozen is None:
                force = min(force + F_RATE, F_MAX)
                if tilt <= -OPEN_HOLD_DEG:
                    frozen = force + 1.0  # small reserve so the hold never sags
                    print(f"[solve] {tag} @{i}: open (tilt={tilt:+.2f}deg) — "
                          f"holding at {frozen:.1f} N", flush=True)
            press_wrench(mode, q_ref, force if frozen is None else frozen)
            env.step(no_action)
            scene.score()  # keep the opened/delivered latches current
            if i % 120 == 119:
                kloc = scene._rig_local(scene.ketchup.data.root_pos_w)[0]
                print(f"[solve] {tag} @{i + 1}: tilt={tilt:+.2f}deg F="
                      f"{(force if frozen is None else frozen):.1f}N "
                      f"k_x={float(kloc[0]):+.3f} "
                      f"delivered={bool(scene._delivered[0])}", flush=True)
            # wrong-mode probe: ramp maxed and the beam never pitched 2 degrees
            if i == 299 and frozen is None and tilt > TILT_REST - 2.0:
                clear_forces()
                print(f"[solve] {tag}: beam did not pitch under the ramp — wrong "
                      f"force-frame mode?", flush=True)
                return False
            if not bool(scene.axle_seated()[0]):
                clear_forces()
                print(f"[solve] {tag}: axle left the journal under the press",
                      flush=True)
                return False
            if bool(scene._delivered[0]):
                for _ in range(60):  # keep the gate open while the bottle lands
                    press_wrench(mode, q_ref, frozen if frozen is not None else force)
                    env.step(no_action)
                    scene.score()
                clear_forces()
                return True
        clear_forces()
        print(f"[solve] {tag}: press budget exhausted "
              f"(tilt={float(scene.beam_tilt_deg()[0]):+.2f})", flush=True)
        return False

    snap_loaded = scene.get_state(all_ids)
    q_press = scene.beam.data.root_quat_w.clone()
    delivered = press_and_hold(1, q_press, "press[m1/ref]")
    if not delivered:  # rollback, other force-frame mode
        scene.set_state(snap_loaded, all_ids)
        step(2)
        delivered = press_and_hold(0, q_press, "press[m0/world]")
    assert delivered, "press failed to ferry the bottle in both force-frame modes"
    report("delivered")
    s2 = print_score("P2 paddle pressed and held — beam pitched onto its press stop, "
                     "gravity rolled the bottle through the window into the bin")
    assert s2 >= 0.75 - 1e-6 and s2 >= s1 - 1e-6, "opened/delivered credit missing"

    # ---------------- phase 3: RELEASE — the bias re-parks the beam, sealing the bin -------
    wait_settled([scene.beam, scene.ketchup], 600)
    report("sealed")
    assert bool(scene.gate_closed()[0]), "beam must re-park (gate closed) on release"
    assert bool(scene.axle_seated()[0]), "axle must remain seated"
    assert bool(scene.in_bin(scene.ketchup)[0]), "ketchup must rest inside the bin"
    s3 = print_score("P3 hands off — mass bias re-parked the beam, window plugged, "
                     "ketchup sealed inside")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after release)", flush=True)
        os._exit(1)
    assert s3 >= 1.0 - 1e-6, "success must score 1.0"

    # ---------------- persistence (>= 3 simulated seconds, no intervention) ----------------
    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                print(f"[solve] persist flicker @step {i}: "
                      f"bin={bool(scene.in_bin(scene.ketchup)[0])} "
                      f"k_set={bool(scene.settled(scene.ketchup)[0])} "
                      f"closed={bool(scene.gate_closed()[0])} "
                      f"b_set={bool(scene.settled(scene.beam)[0])} "
                      f"seated={bool(scene.axle_seated()[0])} "
                      f"mus_out={bool(scene.mustard_out()[0])}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s4 = print_score("P-persist persistence 3.3 s (bottle sealed in the bin, gate "
                     "closed)")
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)

"""Solution for RelayTunnelScene (sim_gen task `pick_i269`) — the task's legitimacy
certificate.

Teleports are TRANSPORT ONLY (staging free cubes on the open apron in front of the
mouth — exactly what an arm does by pick-and-place); every load-bearing interaction
is real contact dynamics:

  P1  FINGERTIP PUSH (applied force on RED, no teleport): the red cube, staged on
      the open apron, is pushed through the mouth along the slab by a velocity-servo
      horizontal force at its centre of mass — the wrench of a closed-jaw fingertip
      pressing its trailing face — as deep as a hand could follow (~35 mm past the
      mouth; the well lies at 90+ mm, far beyond). Real sliding friction, real wall
      guidance.
  P2  RELAY PUSH (applied force on BLUE only): the blue cube is staged on the apron
      behind red and pushed in. Every newton that moves RED from here on is
      transmitted through blue->red block contact — the object-as-tool relay the
      task is about — until red slides off the slab edge, tumbles the 16 mm drop,
      and seats in the sunken well. The servo pushes only BLUE; RED is never
      touched again by any force or teleport.
  P3  settle to live success (red seated in the well, settled, finite) -> 1.0.
  P4  HANDS-OFF persistence >= 3.3 simulated seconds, then `SIM_GEN_SOLVE: SUCCESS`
      (the 16 mm step is what holds the seat — nothing is pinned).

No teleport crosses the roof line: both staging poses sit on the OPEN apron
(garage-frame x < 0, in front of the roofed channel); the smoke battery separately
proves the roof rejects top-down delivery and that a blue-first end state fails.

Prints `SIM_GEN_SCORE <s>` at each phase boundary; the scene's credit is latched so
the sequence is non-decreasing: 0.00 -> 0.25 (entered) -> 0.50 (deep) -> 0.75
(seated latch) -> 1.00 (success).

Run (forge): python -u -m simgen_tasks.pick_i269.solve --headless [--seed N]
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

import os
import threading

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from .scene import _qapply, _qinv, _qz
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scene import _qapply, _qinv, _qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.relay_tunnel")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the statement on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def clear_forces() -> None:
        scene.red.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
        scene.blue.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)

    def gq() -> torch.Tensor:
        return scene.garage.data.root_quat_w

    def red_local() -> torch.Tensor:
        return scene._garage_local(scene.red.data.root_pos_w)[0]

    def blue_local() -> torch.Tensor:
        return scene._garage_local(scene.blue.data.root_pos_w)[0]

    def report(tag: str) -> None:
        rl, bl = red_local(), blue_local()
        print(f"[solve] {tag:12s} | red=({float(rl[0]):+.3f},{float(rl[1]):+.3f},"
              f"{float(rl[2]):+.3f}) blue=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):+.3f}) entered={bool(scene.entered(scene.red)[0])} "
              f"deep={bool(scene.deep(scene.red)[0])} in_well={bool(scene.in_well(scene.red)[0])} "
              f"l_ent={bool(scene._l_entered[0])} l_deep={bool(scene._l_deep[0])} "
              f"l_seat={bool(scene._l_seated[0])} settled={bool(scene.settled_red()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    prev_score = [0.0]

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        assert s >= prev_score[0] - 1e-6, f"score decreased {prev_score[0]} -> {s}"
        prev_score[0] = s
        return s

    def stage(body, local_xyz) -> None:
        """TRANSPORT ONLY: set a free cube down on the open apron (garage-frame
        x < 0, outside the roofed channel), squared to the channel axis — what an
        arm does by pick-and-place before the pushes begin."""
        assert local_xyz[0] < -0.004, "staging must stay on the open apron"
        st = torch.zeros(n, 13, device=device)
        loc = torch.tensor([list(local_xyz)], device=device, dtype=torch.float)
        st[:, 0:3] = scene.garage.data.root_pos_w + _qapply(gq(), loc.expand(n, 3))
        st[:, 3:7] = gq()
        body.write_root_state_to_sim(st, all_ids)

    def apply_push(body, f_world: torch.Tensor, mode: str) -> None:
        """Set a one-step external push whose WORLD direction is `f_world`.
        mode "body" (default, validated): pre-encode with the live inverse body
        quat and use the body-frame call — immune to the pod's `is_global` stale-
        reference frame drag (a world push on the blue cube arrived rotated ~100
        deg on seed 1 because its first-ever wrench set happened at its random
        spawn yaw). mode "world": raw `is_global=True` fallback for pods where
        the default call misbehaves instead."""
        f = torch.zeros(n, 1, 3, device=device)
        if mode == "body":
            f[0, 0, :] = _qapply(_qinv(body.data.root_quat_w), f_world.expand(n, 3))[0]
            body.set_external_force_and_torque(f, zero_w, env_ids=all_ids)
        else:
            f[0, 0, :] = f_world
            body.set_external_force_and_torque(f, zero_w, env_ids=all_ids,
                                               is_global=True)

    def servo_push(body, v_des: float, f_cap: float, gain0: float, done, imax: int,
                   tag: str, on_step=None) -> bool:
        """Velocity-servo push along the garage +x axis: horizontal force at the
        pushed cube's CoM (the fingertip / palm wrench), capped hand-scale, with a
        stall watch that escalates the GAIN (not the cap) and, after two fruitless
        windows at max gain, toggles the force-frame encoding (pod frame-drag
        defense). Returns the done-flag."""
        gain = gain0
        mode, fruitless = "body", 0
        u_local = torch.tensor([[1.0, 0.0, 0.0]], device=device)
        x_start = float(scene._garage_local(body.data.root_pos_w)[0, 0])
        win_i, win_x = 0, x_start
        for i in range(imax):
            if done():
                clear_forces()
                return True
            u_w = _qapply(gq(), u_local)[0]
            v_along = float(torch.dot(body.data.root_lin_vel_w[0], u_w))
            f_mag = max(-f_cap, min(f_cap, gain * (v_des - v_along)))
            apply_push(body, u_w * f_mag, mode)
            env.step(no_action)
            if on_step is not None:
                on_step()
            x_now = float(scene._garage_local(body.data.root_pos_w)[0, 0])
            if i - win_i >= 120:  # stall watch: 1 s windows
                if x_now - win_x < 0.006:
                    gain = min(gain * 1.5, 60.0)
                    fruitless += 1
                    print(f"[solve] {tag} stalled at x={x_now:+.4f}; gain -> {gain:.1f} "
                          f"(fruitless {fruitless}, mode {mode})", flush=True)
                    # Encoding-jam signature (vs a real obstruction): the body never
                    # moved AT ALL since the push began. A physically wedged body that
                    # already made progress keeps its (correct) encoding.
                    if fruitless >= 2 and gain >= 60.0 and abs(x_now - x_start) < 0.010:
                        clear_forces()
                        step(60)  # relax any wedge before re-approaching
                        mode = "world" if mode == "body" else "body"
                        gain, fruitless = gain0, 0
                        print(f"[solve] {tag}: encoding toggled -> {mode}", flush=True)
                else:
                    fruitless = 0
                win_i, win_x = i, x_now
        clear_forces()
        return done()

    # ---------------- phase 0: reset, settle, layout readback --------------------------------
    step(240)
    gp = (scene.garage.data.root_pos_w - scene.env_origins)[0]
    q0 = gq()[0]
    import math as _m
    gyaw = _m.degrees(2.0 * _m.atan2(float(q0[3]), float(q0[0])))
    rl0, bl0 = red_local(), blue_local()
    print(f"[solve] layout readback (seed {args.seed}): "
          f"garage=({float(gp[0]):+.3f},{float(gp[1]):+.3f}) yaw={gyaw:+.1f}deg "
          f"red_local=({float(rl0[0]):+.3f},{float(rl0[1]):+.3f}) "
          f"blue_local=({float(bl0[0]):+.3f},{float(bl0[1]):+.3f}) swap={float(scene.swap[0]):+.0f}",
          flush=True)
    report("reset")
    # honesty readbacks: masses really authored, cubes really outside on the ground
    m_red = float(scene.red.root_physx_view.get_masses().sum())
    m_blue = float(scene.blue.root_physx_view.get_masses().sum())
    print(f"[solve] mass readback: red={m_red:.3f} kg blue={m_blue:.3f} kg", flush=True)
    assert abs(m_red - c.cube_mass) < 0.02, f"red mass not authored: {m_red}"
    assert abs(m_blue - c.cube_mass) < 0.02, f"blue mass not authored: {m_blue}"
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert float(rl0[0]) < -c.apron_len and float(bl0[0]) < -c.apron_len, \
        "both cubes must start on the ground in front of the apron"
    assert not bool(scene.entered(scene.red)[0]), "red must start outside"
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    s0 = print_score("P0 reset+settle (cubes on the floor, channel empty)")
    assert s0 <= 0.01, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: stage red, fingertip-push it into the mouth -------------------
    stage(scene.red, (-0.045, 0.0, c.z_on_slab + 0.002))
    step(30)
    rl = red_local()
    assert abs(float(rl[2]) - c.z_on_slab) < 0.006, "red must rest on the apron slab"

    target1 = 0.060  # trailing face 35 mm inside: a fingertip can follow this far

    def done1() -> bool:
        return float(red_local()[0]) >= target1

    ok = servo_push(scene.red, v_des=0.08, f_cap=3.0, gain0=6.0, done=done1,
                    imax=3000, tag="finger push")
    step(30)
    report("finger-push")
    if not ok:
        print("SIM_GEN_SOLVE: FAIL (fingertip push did not converge)", flush=True)
        os._exit(1)
    assert bool(scene._l_entered[0]), "entered latch must be set"
    assert not bool(scene.in_well(scene.red)[0]), "red must not have reached the well yet"
    s1 = print_score("P1 red pushed inside the mouth as far as a fingertip reaches")
    assert s1 >= c.w_entered - 1e-6, f"entered credit missing: {s1}"

    # ---------------- phase 2: stage blue behind, RELAY-push red into the well ---------------
    stage(scene.blue, (-0.030, 0.0, c.z_on_slab + 0.002))
    step(30)
    bl = blue_local()
    print(f"[solve] blue staged readback: ({float(bl[0]):+.4f},{float(bl[1]):+.4f},"
          f"{float(bl[2]):+.4f}) vel={float(scene.blue.data.root_lin_vel_w[0].norm()):.3f}",
          flush=True)
    assert abs(float(bl[0]) + 0.030) < 0.02 and abs(float(bl[2]) - c.z_on_slab) < 0.006, \
        f"blue must rest where staged, got {bl.tolist()}"
    seen = {"deep": False, "seat": False}

    def done2() -> bool:
        return bool(scene.in_well(scene.red)[0]) and float(red_local()[0]) >= 0.116

    def on_step2() -> None:
        if not seen["deep"] and bool(scene._l_deep[0]):
            seen["deep"] = True
            report("deep")
            print_score("P2a red past half-depth (driven only through blue contact)")
        if not seen["seat"] and bool(scene._l_seated[0]):
            seen["seat"] = True
            report("seated")
            print_score("P2b red dropped off the slab edge into the well")

    # Quasi-static approach speed: the red cube must TIP at the lip, not be shot
    # past it — an over-pushed drop starts beyond the lip and narrows the tumble
    # clearance to the back wall (the seed-1 wedge).
    done_relay = servo_push(scene.blue, v_des=0.045, f_cap=4.0, gain0=12.0, done=done2,
                            imax=4000, tag="relay push", on_step=on_step2)
    report("relay-done")
    if not done_relay:
        print("SIM_GEN_SOLVE: FAIL (relay push did not seat the red cube)", flush=True)
        os._exit(1)
    assert seen["deep"] and bool(scene._l_seated[0]), "deep+seated latches must be set"

    # ---------------- phase 3: settle to live success ----------------------------------------
    step(240)
    report("settled")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (final state is not success)", flush=True)
        os._exit(1)
    s3 = print_score("P3 red cube seated in the well, settled (success)")
    assert s3 >= 1.0 - 1e-6, f"success must score 1.0, got {s3}"

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 steps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s hands-off")
    ok = hold and bool(scene.success()[0]) and s4 >= s3 - 1e-6
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
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL (exception: {exc!r})", flush=True)
        os._exit(2)

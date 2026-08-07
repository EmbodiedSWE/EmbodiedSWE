"""Teleport solution for GapFerryScene (sim_gen task
`libero_kitchen_scene9_turn_on_the_stove_i36`) — the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. BRIDGE (transport teleport + gravity seating): one root-state write carries the
   plank from its rack to a free-space hover ~22 mm ABOVE seat height over the chasm,
   long axis aligned with the crossing direction, rails up, zero velocity. The write
   satisfies nothing: `bridged()` demands the plank AT seat height (+/-6 mm) and calm.
   The plank then FALLS into the rebate ledges; the seating — both ends supported,
   seat height, level, calm — is produced by gravity + contact. If a bounce spoils
   it, the drop is retried; the seat itself is always made by physics.
2. EMBARK (transport teleport): one root-state write moves the BALL a few centimetres
   within the start deck, to the channel mouth, aligned with the seated plank's
   centreline. Still on the start deck: no latch, no score change (asserted).
3. FERRY (applied force + rolling contact): the ball is ROLLED across the bridge by a
   horizontal, velocity-regulated force at its CoM (~0.22 m/s, clamp 3.0 N) — the
   wrench of a fingertip push — with a lateral term that steers it down the plank's
   channel. The crossing facts (`on_plank & over_chasm`, then `in_far_dock`) are all
   produced by rolling contact on the bridge the solver built. Pod force-frame quirk:
   some pods rotate an applied wrench by the body's rotation since its reference —
   fatal for a ROLLING ball unless the desired world force is re-encoded with a fresh
   orientation readback EVERY step (`encode_force`, mode probed from actual progress).
   The push releases once the ball is past the far ledge; arrival and rest are
   gravity + wall + damping. If the ball is lost to the chasm, it is teleported back
   onto the start deck (transport) and the roll retried — the crossing itself is
   always rolled.
4. ORDER: bridge-before-crossing is imposed by physics (no bridge, no crossing), not
   by the rubric; there is no other ordering freedom to declare.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge):
python -u -m simgen_tasks.libero_kitchen_scene9_turn_on_the_stove_i36.solve --headless [--seed N]
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
    from .scene import _qapply, _qz, encode_force
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scene import _qapply, _qz, encode_force

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.gap_ferry")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
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
        scene.ball.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                 env_ids=all_ids)

    def report(tag: str) -> None:
        bp = (scene.ball.data.root_pos_w - scene.env_origins)[0]
        pp = (scene.plank.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] {tag:10s} | ball=({float(bp[0]):+.3f},{float(bp[1]):+.3f},"
              f"{float(bp[2]):+.3f}) plank_z={float(pp[2]):+.3f} "
              f"up_z={float(scene.plank_up_z()[0]):+.3f} "
              f"bridged={bool(scene.bridged()[0])} on_plank={bool(scene.on_plank()[0])} "
              f"in_far={bool(scene.in_far_dock()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # Frames + helpers (single-env solve: index 0 everywhere).
    ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)

    def start_axis_w() -> torch.Tensor:
        """(3,) world direction of the start dock's +x (into the chasm)."""
        u = _qapply(scene.start_dock.data.root_quat_w, ex)[0]
        u = u.clone()
        u[2] = 0.0
        return u / u.norm()

    def chasm_centre_w() -> torch.Tensor:
        """(3,) world xy of the chasm centre (z = env origin)."""
        p = scene.start_dock.data.root_pos_w[0] + start_axis_w() * (c.gap / 2)
        return p

    def far_x() -> float:
        return float(scene._dock_local(scene.far_dock,
                                       scene.ball.data.root_pos_w)[0, 0])

    def start_x() -> float:
        return float(scene._dock_local(scene.start_dock,
                                       scene.ball.data.root_pos_w)[0, 0])

    def ball_z() -> float:
        return float(scene.ball.data.root_pos_w[0, 2] - scene.env_origins[0, 2])

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    step(120)
    sd = (scene.start_dock.data.root_pos_w - scene.env_origins)[0]
    fd = (scene.far_dock.data.root_pos_w - scene.env_origins)[0]
    pl = (scene.plank.data.root_pos_w - scene.env_origins)[0]
    bl = (scene.ball.data.root_pos_w - scene.env_origins)[0]
    side = "+y" if float(scene.rack_side[0]) > 0 else "-y"
    print(f"[solve] layout readback (seed {args.seed}): "
          f"start_dock=({float(sd[0]):+.3f},{float(sd[1]):+.3f}) "
          f"far_dock=({float(fd[0]):+.3f},{float(fd[1]):+.3f}) "
          f"plank=({float(pl[0]):+.3f},{float(pl[1]):+.3f},{float(pl[2]):+.3f}) "
          f"[rack {side}] ball=({float(bl[0]):+.3f},{float(bl[1]):+.3f})", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert not bool(scene.bridged()[0]), "plank on the rack must not count as bridged"
    assert not bool(scene.in_far_dock()[0]), "ball must start in the START dock"
    s0 = print_score("P0 reset+settle (plank racked, ball in the start dock)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # Force-frame reference orientation for `encode_force` (readback now, before the
    # ball has rolled anywhere under our push).
    q_ref = scene.ball.data.root_quat_w.clone()
    mode = 0  # probed from actual progress during the ferry

    # ---------------- phase 1: bridge the chasm (gravity seats the plank) ------------------
    seat_z = c.deck_h - c.rebate_drop + c.plank_t / 2
    u = start_axis_w()
    yaw = torch.atan2(u[1], u[0]).view(1)
    bridged = False
    for attempt in range(3):
        st = torch.zeros(n, 13, device=device)
        ctr = chasm_centre_w()
        st[0, 0], st[0, 1] = ctr[0], ctr[1]
        st[0, 2] = scene.env_origins[0, 2] + seat_z + 0.022
        st[:, 3:7] = _qz(yaw.to(device))
        scene.plank.write_root_state_to_sim(st, all_ids)
        step(180)  # free fall into the rebates + settle, hands-off
        bridged = bool(scene.bridged()[0])
        if bridged:
            break
        print(f"[solve] seating attempt {attempt} missed "
              f"(bridged={bridged}); re-dropping", flush=True)
    report("bridge")
    if not bridged:
        print("SIM_GEN_SOLVE: FAIL (plank never seated across the chasm)", flush=True)
        os._exit(1)
    assert not bool(scene.success()[0]), "a bridge alone cannot be success"
    s1 = print_score("P1 plank dropped into the rebates; seat made by gravity")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_bridge - 1e-6, \
        f"P1 score {s1} (expect bridged=0.30)"

    # ---------------- phase 2: ferry the ball across (applied force + rolling) -------------
    def teleport_ball_to_mouth() -> None:
        """TRANSPORT ONLY: a few cm within the start deck, onto the channel axis."""
        ploc = scene._dock_local(scene.start_dock, scene.plank.data.root_pos_w)[0]
        y_off = float(ploc[1].clamp(-0.03, 0.03))
        q_s = scene.start_dock.data.root_quat_w
        local = torch.tensor([-(c.ball_r + 0.030), y_off, 0.0], device=device)
        pos = scene.start_dock.data.root_pos_w[0] + _qapply(q_s, local.expand(n, 3))[0]
        st = torch.zeros(n, 13, device=device)
        st[0, 0], st[0, 1] = pos[0], pos[1]
        st[0, 2] = scene.env_origins[0, 2] + c.deck_h + c.ball_r + 0.002
        st[0, 3] = 1.0
        scene.ball.write_root_state_to_sim(st, all_ids)

    def ferry() -> bool:
        """Velocity-regulated CoM push down the channel. True once the ball is past
        the far ledge (release point); arrival/rest are then hands-off."""
        nonlocal mode
        # crossing direction: the plank's long axis, signed into the chasm
        axis = scene._plank_axis()[0].clone()
        axis[2] = 0.0
        axis = axis / axis.norm()
        if float((axis * start_axis_w()).sum()) < 0.0:
            axis = -axis
        ey = torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3)
        win_i, win_x = 0, start_x()
        for i in range(2400):
            if far_x() < -0.085:  # past the far ledge, onto the deck: release
                clear_forces()
                return True
            if ball_z() < 0.09:  # lost to the chasm
                clear_forces()
                print("[solve] ferry: ball fell into the chasm", flush=True)
                return False
            speed = 0.22 if far_x() > 0.015 else 0.14
            lat = float(scene._plank_local(scene.ball.data.root_pos_w)[0, 1])
            ey_p = _qapply(scene.plank.data.root_quat_w, ey)[0].clone()
            ey_p[2] = 0.0
            ey_p = ey_p / ey_p.norm().clamp(min=1e-6)
            v_des = axis * speed - ey_p * max(-0.05, min(0.05, 2.0 * lat))
            v = scene.ball.data.root_lin_vel_w[0]
            f_xy = c.ball_mass * 8.0 * (v_des[:2] - v[:2])
            fn = float(f_xy.norm())
            if fn > 3.0:
                f_xy = f_xy * (3.0 / fn)
            f_world = torch.zeros(n, 3, device=device)
            f_world[0, :2] = f_xy
            # rolling ball: re-encode with a FRESH orientation readback every step
            f_arg = encode_force(mode, q_ref, scene.ball.data.root_quat_w, f_world)
            scene.ball.set_external_force_and_torque(
                f_arg.view(n, 1, 3), zero_wrench, env_ids=all_ids, is_global=True)
            env.step(no_action)
            # progress probe every 45 steps: wrong force-frame mode rolls us AWAY
            if i - win_i >= 45:
                x = start_x()
                if x < win_x - 0.008:
                    mode = 1 - mode
                    print(f"[solve] ferry: rolling away (x {win_x:+.3f} -> "
                          f"{x:+.3f}); force-frame mode -> {mode}", flush=True)
                elif x < win_x + 0.005:
                    print(f"[solve] ferry: stalled at x {x:+.3f}", flush=True)
                win_i, win_x = i, x
        clear_forces()
        print(f"[solve] ferry: timed out at far_x {far_x():+.3f}", flush=True)
        return False

    ok = False
    for attempt in range(3):
        teleport_ball_to_mouth()
        step(60)  # settle on the deck after the transport write
        assert abs(float(scene.score()[0]) - s1) < 1e-6, \
            "the embark transport write must not change the score"
        if ferry():
            ok = True
            break
        print(f"[solve] ferry attempt {attempt} failed; recovering", flush=True)
        if not bool(scene.bridged()[0]):  # re-seat if the bridge was disturbed
            st = torch.zeros(n, 13, device=device)
            ctr = chasm_centre_w()
            st[0, 0], st[0, 1] = ctr[0], ctr[1]
            st[0, 2] = scene.env_origins[0, 2] + seat_z + 0.022
            st[:, 3:7] = _qz(yaw.to(device))
            scene.plank.write_root_state_to_sim(st, all_ids)
            step(180)
    step(300)  # arrival + full settle, hands-off
    if ok and not bool(scene.success()[0]):
        step(300)  # a slow roll to the back wall can need longer
    report("ferry")
    if not ok or not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (ball not delivered to the far dock)", flush=True)
        os._exit(1)
    s2 = print_score("P2 ball rolled across the bridge into the far dock")
    assert s2 >= s1 - 1e-6, "score decreased across the ferry"
    assert s2 >= 1.0 - 1e-6, f"success must score 1.0, got {s2}"

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
    try:
        main()
    except BaseException:  # noqa: BLE001 - Kit teardown hangs; die loudly and fast
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(2)

"""Teleport solution for GravityFeedRackScene (sim_gen task
`put_groceries_in_cupboard_i222`) — the task's legitimacy certificate.

The task's load-bearing interactions and how they are executed, PER CAN, in the
forced order red -> green -> blue (FIFO: first in ends up frontmost):

1. PICK (contact dynamics — applied wrench): the can standing on its floor slot is
   lifted straight up by a PD wrench (gravity compensation + z hold + xy centring +
   attitude damping — the "hand" carrying the can). The `lifted` latch is earned by
   this force lift, BEFORE any teleport.
2. TRANSPORT (teleport): ONE pose write stages the can hovering centred over the
   loading port, axis CROSS-LANE, 2 cm above the funnel mouth — free air, outside
   the lane volume, on no queue slot. Verifiably credit-free (asserted).
3. FEED (gravity + contact — never teleported): all wrenches are cut. The can falls
   through the port onto the ramp and the rack's own mechanism does the placement:
   it rolls downhill under the roof and queues against the stop wall (or the can
   already stored). Every queue position is produced by gravity, rolling friction
   and can-on-can contact. The solver merely WAITS and verifies the seat readback.

Order is load-bearing: the port is behind the queue and the roof pins it down, so
whatever drops first is frontmost forever. Dropping red, green, blue yields the
goal facing red-green-blue.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

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
    env = ENVS.get("simgen.gravity_feed_rack")().build(num_envs=args.num_envs,
                                                       device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback
    # so distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    from isaaclab.utils.math import quat_apply, quat_mul

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def loc(body) -> torch.Tensor:
        return scene._rack_local(body)[0]

    def report(tag: str) -> None:
        parts = []
        for nm, body, i in (("red", scene.red, 0), ("green", scene.green, 1),
                            ("blue", scene.blue, 2)):
            p = loc(body)
            parts.append(f"{nm}=({float(p[0]):+.3f},{float(p[1]):+.3f},"
                         f"{float(p[2]):.3f}) seat{i}={bool(scene._seated_slot(body, i)[0])}")
        print(f"[solve] {tag:12s} | " + " ".join(parts)
              + f" | lift={bool(scene._lift_ever[0])} enter={bool(scene._enter_ever[0])}"
              f" q1={bool(scene._q1_ever[0])} q2={bool(scene._q2_ever[0])}"
              f" success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # rack pose readback (kinematic: constant through the episode)
    rack_pos = scene.rack.data.root_pos_w[0].clone()
    rack_quat = scene.rack.data.root_quat_w[0].clone()

    def hold_lift(body, z_des: float, xy0: torch.Tensor) -> None:
        """Grasp-emulation wrench: gravity compensation + z PD + xy centring over the
        pick point + attitude damping (a force-limited hand carrying the can)."""
        pos = body.data.root_pos_w[0]
        vel = body.data.root_lin_vel_w[0]
        fz = c.can_mass * 9.81 + 60.0 * (z_des - float(pos[2])) - 10.0 * float(vel[2])
        fz = max(0.0, min(6.0, fz))
        fx = max(-2.0, min(2.0, -20.0 * float(pos[0] - xy0[0]) - 4.0 * float(vel[0])))
        fy = max(-2.0, min(2.0, -20.0 * float(pos[1] - xy0[1]) - 4.0 * float(vel[1])))
        f = torch.tensor([fx, fy, fz], device=device).view(1, 1, 3).expand(n, 1, 3)
        tau = (-0.05 * body.data.root_ang_vel_w[0]).clamp(-0.10, 0.10)
        body.set_external_force_and_torque(
            f.contiguous(), tau.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            env_ids=all_ids, is_global=True)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    print(f"[solve] layout readback (seed {args.seed}): "
          f"rack_pos=({float(rack_pos[0]):+.3f},{float(rack_pos[1]):+.3f}) "
          f"rack_quat_w={float(rack_quat[0]):+.4f} rack_quat_z={float(rack_quat[3]):+.4f}",
          flush=True)
    for nm, body in (("red", scene.red), ("green", scene.green), ("blue", scene.blue)):
        p = (body.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve]   {nm} floor slot = ({float(p[0]):+.3f},{float(p[1]):+.3f})",
              flush=True)
    report("reset")
    assert not bool(scene._in_lane(scene.red)[0]), "red spawned inside the rack"
    s_prev = print_score("P0 reset+settle")
    assert s_prev <= 0.02, f"reset score should be ~0, got {s_prev}"

    # hover pose over the port: centred, axis cross-lane, above the funnel mouth
    hover_local = torch.tensor(
        [(c.ramp_x0 + c.roof_x0) / 2, 0.0, 0.215], device=device)
    hover_w = rack_pos + quat_apply(rack_quat.unsqueeze(0),
                                    hover_local.unsqueeze(0))[0]
    s45 = torch.tensor([0.7071068, 0.7071068, 0.0, 0.0], device=device)  # z -> -y
    hover_q = quat_mul(rack_quat.unsqueeze(0), s45.unsqueeze(0))[0]

    # ---------------- phases 1..3: load red, green, blue in FIFO order ---------------------
    for phase, (nm, body, slot_i, s_exp) in enumerate(
            (("red", scene.red, 0, 0.40), ("green", scene.green, 1, 0.60),
             ("blue", scene.blue, 2, 1.00)), start=1):
        # --- PICK: force-lift the can off its floor slot (contact dynamics) ---
        xy0 = body.data.root_pos_w[0, 0:2].clone()
        z_goal = 0.25
        lifted = False
        for _ in range(300):
            if float(body.data.root_pos_w[0, 2] - scene.env_origins[0, 2]) >= z_goal - 0.01:
                lifted = True
                break
            hold_lift(body, z_goal, xy0)
            env.step(no_action)
        assert lifted, f"{nm}: force lift never reached the carry height"
        if nm == "red":
            assert bool(scene._lift_ever[0]), "lifted latch not earned by the force lift"
        report(f"{nm}-lifted")

        # --- TRANSPORT: one pose write to the staged hover over the port ---
        body.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = hover_w
        st[:, 3:7] = hover_q
        body.write_root_state_to_sim(st, all_ids)
        env.step(no_action)  # refresh buffers; the can starts to fall
        assert not bool(scene._in_lane(body)[0]), \
            f"{nm}: hover already reads as inside the lane (teleport must be credit-free)"
        for j in range(3):
            assert not bool(scene._seated_slot(body, j)[0]), \
                f"{nm}: hover already reads as seated at slot {j}"
        report(f"{nm}-staged")

        # --- FEED: hands off — gravity drops it through the port, it rolls and queues ---
        streak = 0
        for _ in range(1100):
            env.step(no_action)
            ok = bool(scene._seated_slot(body, slot_i)[0]) \
                and bool(scene._settled(body)[0])
            streak = streak + 1 if ok else 0
            if streak >= 40:
                break
        report(f"{nm}-queued")
        assert streak >= 40, \
            f"{nm}: never settled at queue slot {slot_i} after the port drop"
        s_now = print_score(f"P{phase} {nm} can fed through the port -> queue slot {slot_i}")
        assert s_now >= s_prev - 1e-6, f"score decreased across the {nm} feed"
        assert s_now >= s_exp - 1e-4, \
            f"{nm}: expected score >= {s_exp}, got {s_now}"
        s_prev = s_now

    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after all three feeds)", flush=True)
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, no intervention) ------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s4 >= s_prev - 1e-6
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

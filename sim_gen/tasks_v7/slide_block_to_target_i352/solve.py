"""Teleport solution for TunnelRelayScene (sim_gen task `slide_block_to_target_i352`)
— the task's legitimacy certificate.

The task's load-bearing interaction is INDIRECT TRANSMISSION THROUGH A COLUMN: the
red cargo cube sits inside a bore no tool can enter, and only a column of feeder
blocks pushed in through the mouth can move it. How this solution executes it:

1. TRANSPORT (teleport): each feed cycle starts with ONE pose write that stages the
   next feeder block on the open apron behind the mouth, long side leading — the
   applied-wrench emulation of the arm picking the block off the floor and setting
   it down aligned. Staging never touches the cargo, never places anything inside
   the bore, and is asserted OUTSIDE the bore by readback. The CARGO IS NEVER
   TELEPORTED and NEVER DIRECTLY FORCED anywhere in this file: every millimetre it
   moves is transmitted through feeder-to-feeder and feeder-to-cargo contact.
2. INSERTION (contact dynamics): a velocity-capped horizontal force along the bore
   axis (the fingertip push) slides the staged feeder into the mouth; it shunts the
   column ahead of it, which shunts the cargo. Cycles 1-3 stop with the feeder's
   rear flush with the mouth plane (the fingertip never enters the bore); cycle 4
   stops the instant the cargo readback shows it past the cliff line — the tip-over
   and the drop into the pen are hands-off gravity. A stall escalates the force,
   then flips to body-frame wrench encoding (pod-dependent frame drag).
3. Resource arithmetic does the rest: four 5.5 cm feeders inserted to the mouth
   plane reach past the ejection line; three do not (the scene cfg asserts both).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.4 simulated seconds after
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

import math
import os
import threading

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers simgen.tunnel_relay)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tunnel_relay")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)
    fl = c.feeder_size[0]

    # Seed AFTER build (the EnvCfg.build reseed trap); print the scene's own readouts.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    from isaaclab.utils.math import quat_apply_inverse

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def cargo_l() -> torch.Tensor:
        return scene.cargo_local()[0]

    def feeder_l(i: int) -> torch.Tensor:
        return scene.feeder_local(i)[0]

    def rig_dir_w() -> torch.Tensor:
        """World-frame unit vector of the rig-local +x (bore axis, mouth -> cliff)."""
        yaw = float(scene._rig_yaw[0])
        return torch.tensor([math.cos(yaw), math.sin(yaw), 0.0], device=device)

    def report(tag: str) -> None:
        p = cargo_l()
        print(f"[solve] {tag:14s} | cargo_l=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):.3f}) in_pen={bool(scene.in_pen()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"prog={float(scene._prog_max[0]):.3f} "
              f"eject={bool(scene._eject_ever[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    last_score = [0.0]

    def print_score(tag: str) -> float:
        sc = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {sc:.4f}", flush=True)
        assert sc >= last_score[0] - 1e-6, \
            f"score decreased across {tag}: {last_score[0]:.4f} -> {sc:.4f}"
        last_score[0] = sc
        return sc

    def settle_wait(bodies, max_steps: int = 600, quiet_need: int = 25) -> bool:
        quiet = 0
        for _ in range(max_steps):
            env.step(no_action)
            still = all(
                float(b.data.root_lin_vel_w[0].norm()) < 0.04
                and float(b.data.root_ang_vel_w[0].norm()) < 0.45
                for b in bodies)
            quiet = quiet + 1 if still else 0
            if quiet >= quiet_need:
                return True
        return False

    def stage_feeder(k: int) -> None:
        """TRANSPORT ONLY: one pose write sets feeder k on the apron behind the
        mouth, long side leading (the arm's pick-and-place of a graspable 4 cm-wide
        block). Asserted outside the bore, clear of the mouth plane, by readback."""
        yaw = float(scene._rig_yaw[0])
        pos_l = torch.tensor(
            [[-0.110, 0.0, c.deck_top + c.feeder_size[2] / 2 + 0.003]], device=device)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.to_world(pos_l, all_ids) + scene.env_origins
        st[:, 3] = math.cos(yaw / 2)
        st[:, 6] = math.sin(yaw / 2)
        scene.feeders[k].write_root_state_to_sim(st, all_ids)
        settle_wait([scene.feeders[k]], 240)
        p = feeder_l(k)
        rear = float(p[0]) - fl / 2
        print(f"[solve] staged feeder{k}: local=({float(p[0]):+.3f},"
              f"{float(p[1]):+.3f},{float(p[2]):.3f}) rear={rear:+.3f}", flush=True)
        assert rear < c.x_mouth - 0.02, f"staged feeder{k} rear not clear of the mouth"
        assert float(p[2]) > c.deck_top, f"staged feeder{k} fell off the apron"

    def push_feeder(k: int, last: bool) -> bool:
        """Contact push: velocity-capped force along the bore axis on feeder k only.
        Cycles 0-2 stop at the mouth plane; the last cycle stops the instant the
        cargo passes the cliff line (tip-over + drop are hands-off)."""
        feeder = scene.feeders[k]
        d_w = rig_dir_w()
        f_mag, v_cap = 1.5, 0.05
        mode_body = False
        x_mark, mark_i = -1.0, 0
        done = False
        for i in range(3600):
            pf = feeder_l(k)
            rear = float(pf[0]) - fl / 2
            pc = cargo_l()
            if last:
                if float(pc[0]) > c.x_cliff + 0.002 or float(pc[2]) < c.deck_top - 0.01:
                    done = True
                    break
            elif rear >= c.x_mouth - 0.006:
                done = True
                break
            v_along = float(torch.dot(feeder.data.root_lin_vel_w[0], d_w))
            push = f_mag if v_along < v_cap else 0.0
            # gentle lateral centring (the fingertip steering), rig-frame -y*P
            yaw = float(scene._rig_yaw[0])
            lat = max(-0.5, min(0.5, -30.0 * float(pf[1]) * 0.02))
            f_w = torch.tensor(
                [push * math.cos(yaw) - lat * math.sin(yaw),
                 push * math.sin(yaw) + lat * math.cos(yaw), 0.0], device=device)
            f_cmd = f_w.view(1, 1, 3).expand(n, 1, 3)
            if mode_body:
                f_body = quat_apply_inverse(
                    feeder.data.root_quat_w, f_cmd.view(n, 3)).view(n, 1, 3)
                feeder.set_external_force_and_torque(
                    f_body.contiguous(), zero_w, env_ids=all_ids)
            else:
                feeder.set_external_force_and_torque(
                    f_cmd.contiguous(), zero_w, env_ids=all_ids, is_global=True)
            env.step(no_action)
            x_now = float(pf[0])
            if x_now > x_mark + 0.004:
                x_mark, mark_i = x_now, i
            elif i - mark_i > 240:
                if f_mag < 12.0:
                    f_mag += 1.5
                    print(f"[solve] feeder{k}: stall at x={x_now:+.3f} -> "
                          f"F={f_mag:.1f} N", flush=True)
                elif not mode_body:
                    mode_body = True
                    print(f"[solve] feeder{k}: still stalled -> body-frame force "
                          f"encoding", flush=True)
                mark_i = i
        feeder.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
        if not done:
            print(f"[solve] feeder{k}: push never reached its stop condition",
                  flush=True)
            return False
        pf = feeder_l(k)
        print(f"[solve] feeder{k}: push done, rear={float(pf[0]) - fl / 2:+.3f} "
              f"cargo_x={float(cargo_l()[0]):+.3f}", flush=True)
        return True

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    p0 = cargo_l()
    print(f"[solve] layout readback (seed {args.seed}): "
          f"rig_xy=({float(scene._rig_xy[0, 0]):+.3f},{float(scene._rig_xy[0, 1]):+.3f}) "
          f"rig_yaw={math.degrees(float(scene._rig_yaw[0])):+.1f} deg "
          f"cargo_x0={float(scene._x0[0]):.3f} "
          f"cargo_l=({float(p0[0]):+.3f},{float(p0[1]):+.3f},{float(p0[2]):.3f})",
          flush=True)
    for i in range(4):
        pf = feeder_l(i)
        print(f"[solve]   feeder{i} local=({float(pf[0]):+.3f},{float(pf[1]):+.3f},"
              f"{float(pf[2]):.3f})", flush=True)
    report("reset")
    assert abs(float(p0[0]) - float(scene._x0[0])) < 0.01, "cargo not at its spawn depth"
    assert float(p0[2]) > c.deck_top, "cargo not riding the deck top"
    assert not bool(scene.success()[0]), "success at reset (broken)"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"reset score should be ~0, got {s0}"

    # ---------------- phases 1-4: four feed cycles (teleport stage + contact push) ----------
    for k in range(4):
        stage_feeder(k)
        ok = push_feeder(k, last=(k == 3))
        if not ok:
            report("FAIL-state")
            print("SIM_GEN_SOLVE: FAIL (feed cycle wedged)", flush=True)
            os._exit(1)
        settle_wait([scene.feeders[k], scene.cargo], 300)
        report(f"cycle{k}")
        print_score(f"P{k + 1} feeder {k + 1}/4 inserted through the mouth")

    # ---------------- phase 5: hands-off settle + success -----------------------------------
    settle_wait([scene.cargo], 600)
    report("settled")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (cargo not settled in the pen)", flush=True)
        os._exit(1)
    s5 = print_score("P5 cargo dropped into the pen, settled, success verified")

    # ---------------- phase 6: persistence (>= 3.4 simulated seconds, no intervention) ------
    hold = True
    for _ in range(10):  # 10 x 42 steps = 420 substeps = 3.5 s at 120 Hz
        step(42)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s6 = print_score("P6 persistence 3.5 s")
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
    except Exception as exc:  # noqa: BLE001 — die fast, not at the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL (exception: {exc!r})", flush=True)
        os._exit(2)

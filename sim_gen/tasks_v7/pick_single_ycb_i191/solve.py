"""Teleport solution for CradleClampScene (sim_gen task `pick_single_ycb_i191`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY. Both LOAD-BEARING interactions go through
CONTACT DYNAMICS:
  1. BAR TRANSPORT: the blue bar is teleported from the ground to a free-air hover
     straight above the saddle line (fixture frame (0,0,~0.114), axis along the
     saddle, rolled 45 deg to the DIAMOND orientation) — exactly the pose an arm
     reaches after grasping the bar and re-orienting it. The hover is far outside the
     saddle's z band and scores only the transport-legal lift/approach latches.
  2. BAR SEATING (contact): the bar is simply RELEASED. It falls ~55 mm onto the two
     45-degree V faces; the V centers it and it rings down to the flush rest height.
     The seating the rubric judges is established by gravity + contact, never written.
  3. BRACKET TRANSPORT: the bracket is teleported to a free-air hover above the seated
     bar — upright, leg line along the pocket line, legs' feet above the pocket wall
     tops (non-scoring: ~43 mm above the seat band).
  4. BRACKET SEATING (contact): a gravity-assisted, velocity-regulated downward push
     with weak fixture-frame centering springs lowers the bracket; the legs enter the
     two pocket wells under real collision (any misalignment scrapes legs on walls),
     bottom out on the pocket floors, and the wrench is cleared well before judging.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after success()
first holds and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.pick_single_ycb_i191.solve --headless [--seed N]
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
    from isaaclab.utils.math import quat_apply, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cradle_clamp")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def fixture_pose() -> tuple[torch.Tensor, torch.Tensor, float]:
        fp = scene.fixture.data.root_pos_w
        fq = scene.fixture.data.root_quat_w
        yaw = 2.0 * math.atan2(float(fq[0, 3]), float(fq[0, 0]))
        return fp, fq, yaw

    def teleport_fixture_frame(body, local_pos, q_rel) -> None:
        """Write `body` to a fixture-frame pose (free air, non-scoring), zero vel."""
        fp, fq, _ = fixture_pose()
        lp = torch.tensor(local_pos, device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = fp + quat_apply(fq, lp)
        st[:, 3:7] = quat_mul(fq, torch.tensor(q_rel, device=device).expand(n, 4))
        body.write_root_state_to_sim(st, all_ids)

    def report(tag: str) -> None:
        bar_l = scene._fixture_local(scene.bar)[0][0]
        br_l = scene._fixture_local(scene.bracket)[0][0]
        print(f"[solve] {tag:12s} | bar_loc=({float(bar_l[0]):+.3f},{float(bar_l[1]):+.3f},"
              f"{float(bar_l[2]):.3f}) in_saddle={bool(scene.bar_in_saddle()[0])} "
              f"| br_loc=({float(br_l[0]):+.3f},{float(br_l[1]):+.3f},{float(br_l[2]):.3f}) "
              f"seated={bool(scene.bracket_seated()[0])} "
              f"| success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_wrench(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    fp, fq, fyaw = fixture_pose()
    fo = (fp - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): fixture at "
          f"({float(fo[0]):+.3f},{float(fo[1]):+.3f}) yaw={math.degrees(fyaw):+.1f}deg",
          flush=True)
    for nm, body in (("bar_blue", scene.bar), ("bar_red", scene.decoy),
                     ("bracket", scene.bracket)):
        p = (body.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] layout readback (seed {args.seed}): {nm} at "
              f"({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):.3f})", flush=True)
    print(f"[solve] d_ref_bar={float(scene.d_ref_bar[0]):.3f} "
          f"d_ref_br={float(scene.d_ref_br[0]):.3f}", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: bar — transport hover, then gravity-seat in the V -----------
    # Hover: fixture frame (0,0,~0.114) — 55 mm above the flush rest height, DIAMOND
    # orientation: axis along the saddle line (bar local +z -> fixture +y = qx(-90))
    # composed with a 45 deg roll about the bar axis (qz(45)) so both lower faces land
    # flush on the 45-degree V faces.  q_rel = qx(-90) x qz(45).
    a, cq, sq = math.cos(math.pi / 4), math.cos(math.pi / 8), math.sin(math.pi / 8)
    q_diamond = (a * cq, -a * cq, a * sq, a * sq)
    teleport_fixture_frame(scene.bar, (0.0, 0.0, c.bar_rest_z + 0.055), q_diamond)
    step(3)  # airborne hover: latches transport lift/approach, scores nothing else
    report("bar-hover")
    # Release: gravity + the V faces do the seating (contact).
    for _ in range(16):  # up to 4 s
        step(30)
        if bool((scene.bar_in_saddle() & scene.settled(scene.bar))[0]):
            break
    report("bar-seated")
    s1 = print_score("P1 bar seated in V (contact)")
    assert s1 >= s0 - 1e-6, "score decreased across bar seating"
    if not bool(scene.bar_in_saddle()[0]):
        print("SIM_GEN_SOLVE: FAIL (bar did not seat in the V)", flush=True)
        os._exit(1)

    # ---------------- phase 2: bracket — transport hover, then regulated press -------------
    # Hover: upright, leg line along the pocket line, origin 43 mm above the seat band —
    # legs' feet (origin - 0.078) sit above the pocket wall tops (0.052): free air.
    hover_z = c.seat_z + 0.043
    teleport_fixture_frame(scene.bracket, (0.0, 0.0, hover_z), (1.0, 0.0, 0.0, 0.0))
    step(3)
    report("br-hover")

    # Regulated press (contact): gravity + a velocity-capped extra push, weak
    # fixture-frame centering springs; legs thread the pocket wells under collision.
    grav = 9.81 * c.bracket_mass
    f_ax, v_des = 0.4, 0.08
    best_z, last_bump, steps = 10.0, 0, 0
    for i in range(1200):
        loc = scene._fixture_local(scene.bracket)[0][0]
        z_l = float(loc[2])
        if z_l <= c.seat_z + 0.003:
            break
        v_dn = -float(scene.bracket.data.root_lin_vel_w[0, 2])
        fz = -f_ax if v_dn < v_des else 0.0  # extra push only when descending slowly
        fl = torch.tensor([max(-0.5, min(0.5, -12.0 * float(loc[0]))),
                           max(-0.5, min(0.5, -12.0 * float(loc[1]))), 0.0], device=device)
        _, fq_now, _ = fixture_pose()
        f_w = quat_apply(fq_now, fl.expand(n, 3)) + torch.tensor(
            [0.0, 0.0, fz], device=device).expand(n, 3)
        scene.bracket.set_external_force_and_torque(
            f_w.view(n, 1, 3).contiguous(), zero_wrench, env_ids=all_ids, is_global=True)
        env.step(no_action)
        steps += 1
        if z_l < best_z - 0.002:
            best_z, last_bump = z_l, i
        elif i - last_bump > 200:  # stalled on a wall top: push harder
            f_ax = min(f_ax + 0.4, 2.4)
            last_bump = i
            print(f"[solve] bracket press stalled at z={z_l:.3f}, raising push to "
                  f"{f_ax:.1f} N", flush=True)
    clear_wrench(scene.bracket)
    print(f"[solve] bracket pressed home after {steps} steps (push {f_ax:.1f} N)",
          flush=True)
    for _ in range(12):  # up to 3 s hands-off ring-down
        step(30)
        if bool((scene.bracket_seated() & scene.settled(scene.bracket))[0]):
            break
    report("br-seated")
    s2 = print_score("P2 bracket seated (contact)")
    assert s2 >= s1 - 1e-6, "score decreased across bracket seating"

    # ---------------- phase 3: settle to success -------------------------------------------
    for _ in range(12):  # up to 3 s
        if bool(scene.success()[0]):
            break
        step(30)
    report("settled")
    s3 = print_score("P3 all settled")
    assert s3 >= s2 - 1e-6, "score decreased across settling"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after seating+settle)", flush=True)
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, hands-off) -----------
    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                bar_l = scene._fixture_local(scene.bar)[0][0]
                br_l = scene._fixture_local(scene.bracket)[0][0]
                print(f"[solve] persist flicker @step {i}: "
                      f"saddle={bool(scene.bar_in_saddle()[0])} "
                      f"seated={bool(scene.bracket_seated()[0])} "
                      f"bar_v={float(scene.bar.data.root_lin_vel_w[0].norm()):.4f} "
                      f"br_v={float(scene.bracket.data.root_lin_vel_w[0].norm()):.4f} "
                      f"bar_z={float(bar_l[2]):.4f} br_z={float(br_l[2]):.4f}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
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
    main()

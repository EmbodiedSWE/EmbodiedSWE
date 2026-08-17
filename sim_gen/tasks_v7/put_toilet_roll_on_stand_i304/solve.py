"""Teleport solution for RollMagazineScene (sim_gen task
`put_toilet_roll_on_stand_i304`) — the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY. Every load-bearing interaction goes through
contact dynamics, twice (once per roll):
  1. STAGE (transport + gravity): the roll is teleported to a hover 20 mm above the
     open entry apron, axis across the channel — clear of the rails, the roof, and
     the other roll — then RELEASED. Gravity sets it down on the apron. Nothing is
     ever teleported into contact or past the roof line.
  2. FEED (contact dynamics): a regulated CoM push (velocity-capped force along the
     channel + a weak lateral centering PD — the wrench a gripper pushing the roll
     would transmit) drives the roll along the apron and UNDER the low roof edge.
     The 80 mm mouth admits it only because it stays ON the surface — the roof is
     real collision geometry, not a rubric fiction. The moment its center crosses
     the crest the wrench is CUT: the delivery — accelerating down the 13-degree
     ramp, crossing the tunnel, impacting the stop wall (or the queued first roll)
     and coming to rest in the bay — is pure gravity + contact, fully out of reach
     and untouched.

The goal state (both rolls queued in the bay, across the channel, at rest) is
reached with all wrenches cut; success() reads settled poses only. Prints
`SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's credit
is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it persists.

Run (forge): python -u -m simgen_tasks.put_toilet_roll_on_stand_i304.solve --headless [--seed N]
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
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod

OUT_FLAT = scene_mod.OUT_FLAT
OUT_CORNER = scene_mod.OUT_CORNER
Z_AP = scene_mod.Z_AP
X_AP0 = scene_mod.X_AP0
X_CREST = scene_mod.X_CREST
X_WALL = scene_mod.X_WALL
X_BAY = scene_mod.X_BAY

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.roll_magazine")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def wrench(body, f_w: torch.Tensor, tau_w: torch.Tensor | None = None) -> None:
        """Apply a WORLD force/torque (n,3) to `body`, expressed in its CURRENT link
        frame (`is_global=True` silently drops torques on this stack — transform
        manually, the house convention). Re-set every step while pushing."""
        q = body.data.root_link_quat_w
        if tau_w is None:
            tau_w = torch.zeros(n, 3, device=device)
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f_w).unsqueeze(1),
            quat_apply_inverse(q, tau_w).unsqueeze(1),
            env_ids=all_ids)

    zero3 = torch.zeros(n, 3, device=device)

    def cut(body) -> None:
        wrench(body, zero3, zero3)

    def place(body, pos_w: torch.Tensor, quat_w: torch.Tensor) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat_w
        body.write_root_state_to_sim(st, all_ids)

    # ----- garage frame (kinematic; fixed for the whole episode) ---------------------------
    q_g = scene.magazine.data.root_quat_w.clone()
    p_g = scene.magazine.data.root_pos_w.clone()

    def loc_of(body) -> torch.Tensor:
        return quat_apply_inverse(q_g, body.data.root_pos_w - p_g)

    def vel_of(body) -> torch.Tensor:
        return quat_apply_inverse(q_g, body.data.root_lin_vel_w)

    # roll target orientation: tube local +z -> garage +y (axis ACROSS the channel);
    # quat_mul applies the right factor first: qy90 maps z->x, then qz90 maps x->y.
    qy90 = torch.tensor([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0],
                        device=device).expand(n, 4)
    qz90 = torch.tensor([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)],
                        device=device).expand(n, 4)
    q_across = quat_mul(q_g, quat_mul(qz90, qy90))

    def report(tag: str) -> None:
        p, _a = scene._rolls_local()
        st = scene.staged()
        ins = scene.inside()
        pk = scene.parked()
        al = scene.aligned()
        print(f"[solve] {tag:12s} | "
              f"A=({float(p[0, 0, 0]):+.3f},{float(p[0, 0, 1]):+.3f},{float(p[0, 0, 2]):.3f}) "
              f"B=({float(p[0, 1, 0]):+.3f},{float(p[0, 1, 1]):+.3f},{float(p[0, 1, 2]):.3f}) "
              f"staged=({bool(st[0, 0])},{bool(st[0, 1])}) "
              f"inside=({bool(ins[0, 0])},{bool(ins[0, 1])}) "
              f"parked=({bool(pk[0, 0])},{bool(pk[0, 1])}) "
              f"aligned=({bool(al[0, 0])},{bool(al[0, 1])}) "
              f"set={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f} still={int(scene.still_count[0])} "
              f"| vA={float(scene.rolls[0].data.root_lin_vel_w[0].norm()):.4f} "
              f"vB={float(scene.rolls[1].data.root_lin_vel_w[0].norm()):.4f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def stage_roll(k: int) -> None:
        """STAGE: teleport roll k to a hover over the open apron, drop by gravity."""
        hover = torch.tensor([-0.190, 0.0, Z_AP + OUT_CORNER + 0.020],
                             device=device).expand(n, 3)
        place(scene.rolls[k], p_g + quat_apply(q_g, hover), q_across)
        step(90)
        report(f"staged {'AB'[k]}")
        assert bool(scene.staged()[0, k]), f"roll {'AB'[k]} did not stage on the apron"

    def feed_roll(k: int) -> None:
        """FEED: regulated CoM push along the channel until the center crosses the
        crest; then CUT — the delivery down the covered ramp is untouched."""
        body = scene.rolls[k]
        crossed = False
        for _ in range(900):
            p_l = loc_of(body)
            if bool(p_l[0, 0] > X_CREST + 0.030):
                crossed = True
                break
            v_l = vel_of(body)
            f_l = torch.zeros(n, 3, device=device)
            # velocity-capped drive (K*dt/m = 3*(1/120)/0.1 = 0.25 — inside the
            # wrench-delay bound) + weak lateral centering; no vertical component,
            # no torque: gravity and surface friction own the rolling.
            f_l[:, 0] = (3.0 * (0.30 - v_l[:, 0])).clamp(-0.6, 1.2)
            f_l[:, 1] = (4.0 * (0.0 - p_l[:, 1]) - 1.0 * v_l[:, 1]).clamp(-0.6, 0.6)
            wrench(body, quat_apply(q_g, f_l))
            env.step(no_action)
        cut(body)
        assert crossed, f"the push never drove roll {'AB'[k]} past the crest"
        # hands-off delivery + settle: ramp, wall/queue impact, rest in the bay
        for j in range(40):  # up to 10 s
            if bool(scene.parked()[0, k]) and \
                    float(scene.still_count[0]) >= float(c.settle_steps):
                break
            step(30)
            if j % 4 == 3:
                p_l = loc_of(body)
                print(f"[solve] deliver {'AB'[k]} chunk {j}: x={float(p_l[0, 0]):+.3f} "
                      f"z={float(p_l[0, 2]):.3f} "
                      f"v={float(body.data.root_lin_vel_w[0].norm()):.4f} "
                      f"w={float(body.data.root_ang_vel_w[0].norm()):.4f} "
                      f"still={int(scene.still_count[0])}", flush=True)
        report(f"delivered {'AB'[k]}")
        assert bool(scene.inside()[0, k]), f"roll {'AB'[k]} not inside the tunnel"
        assert bool(scene.parked()[0, k]), f"roll {'AB'[k]} stopped short of the bay"
        assert bool(scene.aligned()[0, k]), f"roll {'AB'[k]} arrived crooked"

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    gq = q_g[0]
    gyaw = math.degrees(2.0 * math.atan2(float(gq[3]), float(gq[0])))
    gp = (p_g - scene.env_origins)[0]
    p0, _ = scene._rolls_local()
    print(f"[solve] layout readback (seed {args.seed}): "
          f"magazine=({float(gp[0]):+.3f},{float(gp[1]):+.3f}) yaw={gyaw:+.1f} "
          f"A_loc=({float(p0[0, 0, 0]):+.3f},{float(p0[0, 0, 1]):+.3f}) "
          f"B_loc=({float(p0[0, 1, 0]):+.3f},{float(p0[0, 1, 1]):+.3f})", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, "null credit at reset — rubric leak"

    # ---------------- phase 1: STAGE roll A (transport + gravity drop) ---------------------
    stage_roll(0)
    s1 = print_score("P1 roll A staged on the apron (transport + gravity)")
    assert s1 >= 0.10 - 1e-6 and s1 >= s0 - 1e-6

    # ---------------- phase 2: FEED roll A (contact dynamics, cut at the crest) ------------
    feed_roll(0)
    s2 = print_score("P2 roll A fed under the roof; delivered by gravity (contact)")
    assert s2 >= 0.30 - 1e-6 and s2 >= s1 - 1e-6

    # ---------------- phase 3: STAGE roll B ------------------------------------------------
    stage_roll(1)
    s3 = print_score("P3 roll B staged on the apron (transport + gravity)")
    assert s3 >= 0.40 - 1e-6 and s3 >= s2 - 1e-6

    # ---------------- phase 4: FEED roll B; queue forms; hands-off settle ------------------
    feed_roll(1)
    # everything is already hands-off; wait for the full success gate (both parked,
    # aligned, sustained stillness)
    for j in range(32):  # up to 8 s
        if bool(scene.success()[0]):
            break
        step(30)
        if j % 4 == 3:
            print(f"[solve] settle chunk {j}: "
                  f"vA={float(scene.rolls[0].data.root_lin_vel_w[0].norm()):.4f} "
                  f"vB={float(scene.rolls[1].data.root_lin_vel_w[0].norm()):.4f} "
                  f"wA={float(scene.rolls[0].data.root_ang_vel_w[0].norm()):.4f} "
                  f"wB={float(scene.rolls[1].data.root_ang_vel_w[0].norm()):.4f} "
                  f"still={int(scene.still_count[0])}", flush=True)
    if bool(scene.success()[0]):
        step(240)  # 2 s more hands-off margin before the strict persistence window
    report("loaded")
    s4 = print_score("P4 roll B delivered; magazine loaded (gravity + contact)")
    assert s4 >= s3 - 1e-6, "score decreased across the final delivery"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after both deliveries)", flush=True)
        os._exit(1)

    # ---------------- phase 5: persistence (>= 3.3 simulated seconds, hands off) -----------
    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                pk = scene.parked()
                al = scene.aligned()
                print(f"[solve] persist flicker @step {i}: "
                      f"parked=({bool(pk[0, 0])},{bool(pk[0, 1])}) "
                      f"aligned=({bool(al[0, 0])},{bool(al[0, 1])}) "
                      f"set={bool(scene.settled()[0])} "
                      f"vA={float(scene.rolls[0].data.root_lin_vel_w[0].norm()):.4f} "
                      f"vB={float(scene.rolls[1].data.root_lin_vel_w[0].norm()):.4f}",
                      flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
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
    try:
        main()
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)

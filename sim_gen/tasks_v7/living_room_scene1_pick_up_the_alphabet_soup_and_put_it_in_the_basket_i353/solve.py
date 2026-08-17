"""Teleport solution for PantryRackOrderScene (sim_gen task
`living_room_scene1_pick_up_the_alphabet_soup_and_put_it_in_the_basket_i353`) — the
task's legitimacy certificate.

The task's load-bearing interactions and how they are executed:

1. SEQUENCE DERIVATION (readback): the target permutation is read back from the
   scene (the same information the roof tiles display) and the insertion order is
   its deepest-first order — the plan IS the answer to the puzzle.
2. Per can, TRANSPORT teleport: ONE pose write stands the can upright on the loading
   APRON, on the rack's CURRENT axis (the rack xy/yaw are randomized, so the staging
   pose is computed in the rack's readback frame), fully OUTSIDE the mouth, touching
   nothing but the apron floor. The can is never written to a pose inside the rack.
3. INSERTION is pure contact physics: a horizontal velocity-servo force (with a
   compensating torque that lowers the effective push point to ~8 mm above the
   floor, like a fingertip pushing low on the can) slides the can along the slick
   floor through the mouth. Each later can SHOVES the earlier ones deeper through
   can-to-can contact — exactly the mechanism that makes insertion order equal
   final order. Forces are cleared the moment the can passes its goal depth; the
   coast-out and the settling are hands-off.

Wrong-order pushes would produce an arrangement that can never be repaired (the
roof forbids lifting; the channel forbids passing); this solve derives the correct
order first, so each stage latch (0.30 / 0.60) fires as its prefix completes and
the final live success() (exact depth order, all upright, all still) reaches 1.0.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: stage credit
is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

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
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.pantry_rack_order")().build(num_envs=args.num_envs,
                                                       device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback
    # so distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    from isaaclab.utils.math import quat_apply

    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(n, 1, 3, device=device)
    names = [s[0] for s in c.can_specs]

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        d = scene.depths()[0]
        ins = scene.inside()[0]
        st = scene.still()[0]
        print(f"[solve] {tag:14s} | d=({float(d[0]):+.3f},{float(d[1]):+.3f},"
              f"{float(d[2]):+.3f}) in=({bool(ins[0])},{bool(ins[1])},{bool(ins[2])}) "
              f"still=({bool(st[0])},{bool(st[1])},{bool(st[2])}) "
              f"s1_ever={bool(scene._s1_ever[0])} s2_ever={bool(scene._s2_ever[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, readback -------------------------------------
    step(60)
    rq0 = scene.rack.data.root_quat_w[0]
    r_yaw = math.degrees(2.0 * math.atan2(float(rq0[3]), float(rq0[0])))
    rp0 = (scene.rack.data.root_pos_w - scene.env_origins)[0]
    perm = scene._perm[0].tolist()  # can indices, deepest first (what the tiles show)
    can_xy = [(scene.cans[i].data.root_pos_w - scene.env_origins)[0] for i in range(3)]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"rack=({float(rp0[0]):+.3f},{float(rp0[1]):+.3f}) yaw={r_yaw:+.1f}deg "
          f"perm(deepest-first)={[names[i] for i in perm]} "
          + " ".join(f"{names[i]}=({float(can_xy[i][0]):+.3f},{float(can_xy[i][1]):+.3f})"
                     for i in range(3)), flush=True)
    report("reset")
    assert not bool(scene.inside()[0].any()), "a can spawned inside the rack?!"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"reset score should be ~0, got {s0}"

    # ---------------- phases 1..3: insert deepest-first, each push is physics ---------------
    mu_pair = (c.floor_mu + c.can_mu) / 2  # PhysX pair friction ~ average of the prims
    dz_eff = 0.017  # lower the effective push point to ~8 mm above the floor
    off = torch.tensor([0.0, 0.0, -dz_eff], device=device).expand(n, 3)
    ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
    chain_mass = 0.0
    prev = s0
    stage_names = ("P1 deepest can seated", "P2 middle can seated",
                   "P3 outermost can seated -> live success")
    for rank, ci in enumerate(perm):
        body = scene.cans[ci]
        r = float(scene._radii[ci])
        m_can = float(c.can_specs[ci][3])
        chain_mass += m_can

        # --- transport teleport: stage upright on the apron, on the rack's axis,
        # fully OUTSIDE the mouth (front edge 45 mm short of the mouth plane) ---
        rq = scene.rack.data.root_quat_w
        rp = scene.rack.data.root_pos_w
        loc = torch.tensor([-(r + 0.045), 0.0,
                            c.floor_top + scene._height / 2 + 0.002],
                           device=device).expand(n, 3)
        st_w = torch.zeros(n, 13, device=device)
        st_w[:, 0:3] = rp + quat_apply(rq, loc)
        st_w[:, 3:7] = rq
        body.write_root_state_to_sim(st_w)
        step(12)  # settle onto the apron
        d_stage = float(scene.depths()[0, ci])
        assert d_stage < -r, f"{names[ci]} staged past the mouth (d={d_stage:.3f})"

        # --- push through the mouth: velocity-servo horizontal force + torque that
        # lowers the effective push point; cleared at the goal depth ---
        u = quat_apply(rq, ex)  # rack +x (depth) in world, horizontal
        ff = 1.1 * mu_pair * chain_mass * 9.81  # feedforward: slide the whole chain
        cap = 2.2
        gain = 4.0
        v_des = 0.12
        target_d = r + 0.022
        pushed = False
        last_d = d_stage
        for i in range(1800):
            d = float(scene.depths()[0, ci])
            if d >= target_d:
                pushed = True
                break
            v_along = (body.data.root_lin_vel_w * u).sum(dim=-1)  # (n,)
            f_mag = (ff + gain * (v_des - v_along)).clamp(min=0.0, max=cap)
            fw = (f_mag.unsqueeze(-1) * u).unsqueeze(1)  # (n,1,3)
            tw = torch.cross(off, f_mag.unsqueeze(-1) * u, dim=-1).unsqueeze(1)
            body.set_external_force_and_torque(fw, tw, env_ids=all_ids,
                                               is_global=True)
            env.step(no_action)
            if (i + 1) % 120 == 0:  # stall watch: escalate feedforward, then cap
                if d - last_d < 0.005:
                    ff *= 1.4
                    cap = min(cap * 1.25, 4.0)
                    print(f"[solve] {names[ci]} push stalled at d={d:+.3f} "
                          f"-> ff={ff:.2f}N cap={cap:.2f}N", flush=True)
                last_d = d
        body.set_external_force_and_torque(zero3, zero3, env_ids=all_ids,
                                           is_global=True)
        assert pushed, f"{names[ci]} never reached goal depth {target_d:.3f}"

        # --- hands-off: coast out, settle, let the stage latch fire -----------------------
        latched = False
        for _ in range(480):
            env.step(no_action)
            if rank == 0:
                latched = bool(scene._s1_ever[0])
            elif rank == 1:
                latched = bool(scene._s2_ever[0])
            else:
                latched = bool(scene.success()[0]) and bool(scene.still()[0].all())
            if latched:
                break
        report(f"rank{rank} in")
        assert latched, f"stage {rank} latch did not fire after {names[ci]}"
        assert bool(scene.inside()[0, ci]), f"{names[ci]} not fully inside after push"
        s = print_score(stage_names[rank])
        assert s >= prev - 1e-6, f"score decreased across insertion {rank}"
        prev = s
    assert prev >= 0.99, f"final insertion did not reach live success, score {prev}"

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, hands-off) ------------
    hold = True
    for _ in range(10):  # 10 x 40 = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s4 >= prev - 1e-6
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
    except Exception as e:  # noqa: BLE001 — fail fast, don't idle until the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)

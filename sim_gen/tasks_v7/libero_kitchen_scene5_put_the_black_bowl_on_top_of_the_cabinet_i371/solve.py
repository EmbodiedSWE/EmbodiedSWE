"""Teleport solution for RoofShuttleScene (sim_gen task
`libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet_i371`) — the
task's legitimacy certificate.

Teleports are TRANSPORT ONLY (relocating a body the solver is already holding in
free air). Every load-bearing interaction is contact dynamics or applied force:

1. PERCEPTION: panel start offset, pedestal pose and bowl pose/yaw are read back
   from the episode state — never hard-coded.
2. ROOF SLIDE (applied force): a velocity-servo push (<= 12 N, what a hand on the
   yellow handle bar does) drives the captive roof panel back along its prismatic
   rail until the bowl bay is uncovered, then brakes and lets go. The panel's own
   joint carries its weight; the slide is the real mechanism the task is about.
3. BOWL OUT (applied force): a PD force + gravity feedforward (a firm grasp,
   <= 8 N) lifts the bowl straight up OUT through the bay opening the slide just
   created — the bay walls and the panel's parked position bound the escape
   corridor, so this force-driven exit proves the opening is real. A small
   righting torque stands in for the grasp's orientation constraint.
4. SEAT (gravity): once the bowl is in free air well above the cabinet it is
   TELEPORTED over the panel's top face and RELEASED 2 cm up — gravity seats it.
   The seat clause is judged on the settled contact outcome, not on the carry.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
rubric latches), then holds HANDS-OFF >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet_i371.solve --headless [--seed N]
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
    env = ENVS.get("simgen.roof_shuttle")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so
    # the task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        print(f"[solve] {tag:12s} | off={float(scene.slab_off()[0]):+.4f} "
              f"uncov={float(scene.uncovered()[0]):+.4f} "
              f"in_bay={bool(scene.bowl_in_bay()[0])} "
              f"on_slab={bool(scene.bowl_on_slab()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    zero = torch.zeros(n, 1, 3, device=device)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)

    # ---------------- phase 0: settle, layout readback, baseline ---------------------------------
    step(120)   # bowl seats on its pedestal; panel hangs still on its rail
    origin = scene.env_origins
    ped = (scene.pedestal.data.root_pos_w - origin)[0]
    bwl = (scene.bowl.data.root_pos_w - origin)[0]
    bq = scene.bowl.data.root_quat_w[0]
    yaw = math.degrees(2.0 * math.atan2(float(bq[3]), float(bq[0])))
    print(f"[solve] layout readback (seed {args.seed}): "
          f"slab_off0={float(scene.slab_off0[0]):+.4f} (measured {float(scene.slab_off()[0]):+.4f}) "
          f"ped=({float(ped[0]):+.3f},{float(ped[1]):+.3f}) "
          f"bowl=({float(bwl[0]):+.3f},{float(bwl[1]):+.3f},{float(bwl[2]):+.3f}) "
          f"yaw~{yaw:+.1f}deg", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert bool(scene.bowl_in_bay()[0]), "bowl must start captive in the bay"
    assert float(scene.uncovered()[0]) < c.open_pass / 2, "bay must start roofed"
    assert abs(float(scene.slab_off()[0]) - float(scene.slab_off0[0])) < 0.006, \
        "panel must rest at its sampled start offset"
    assert bool(scene.settled()[0]), "start state must be settled"
    s_prev = print_score("P0 reset+settle (bowl sealed under the roof panel)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s_prev <= 0.05, f"baseline score should be ~0, got {s_prev}"

    # ---------------- phase 1: slide the roof panel open (applied force) -------------------------
    # Velocity servo on the panel along its rail (+y): F = kv * (v_des - v_y),
    # clamped — a steady hand on the handle bar. The prismatic joint is the rail.
    kv, v_des, f_clamp = 40.0, 0.15, 12.0
    for i in range(600):
        if float(scene.slab_off()[0]) >= 0.272:
            break
        v = scene.slab.data.root_lin_vel_w
        f_w = torch.zeros(n, 3, device=device)
        f_w[:, 1] = (kv * (v_des - v[:, 1])).clamp(-f_clamp, f_clamp)
        f_b = quat_apply_inverse(scene.slab.data.root_quat_w, f_w)
        scene.slab.set_external_force_and_torque(f_b.reshape(n, 1, 3), zero)
        env.step(no_action)
    # brake to a stop, then hands off
    for _ in range(80):
        v = scene.slab.data.root_lin_vel_w
        if float(v[0].norm()) < 0.01:
            break
        f_w = torch.zeros(n, 3, device=device)
        f_w[:, 1] = (-kv * v[:, 1]).clamp(-f_clamp, f_clamp)
        f_b = quat_apply_inverse(scene.slab.data.root_quat_w, f_w)
        scene.slab.set_external_force_and_torque(f_b.reshape(n, 1, 3), zero)
        env.step(no_action)
    scene.slab.set_external_force_and_torque(zero, zero)
    step(90)
    report("slide-open")
    assert float(scene.slab_off()[0]) >= 0.26, \
        f"panel did not slide open: off={float(scene.slab_off()[0]):+.4f}"
    assert float(scene.uncovered()[0]) >= c.open_pass, "bay must be uncovered wide"
    assert bool(scene._open_l[0]), "open latch must fire"
    assert bool(scene.bowl_in_bay()[0]), "bowl untouched so far"
    s = print_score("P1 roof panel slid back by the handle (force servo)")
    assert s >= s_prev - 1e-6, "score decreased across P1"
    assert s >= c.w_slide * 0.85 + c.w_open - 1e-6, f"P1 score {s:.3f} below slide+open credit"
    s_prev = s

    # ---------------- phase 2: bowl lifted out through the opening (applied force) ---------------
    # PD force + gravity feedforward, straight up at the bowl's own xy — the exit
    # corridor is the opening P1 created; this force lift proves it is real.
    tgt = scene.bowl.data.root_pos_w.clone()
    tgt[:, 2] = origin[:, 2] + 0.47
    kp, kd, clamp, ku, kw = 60.0, 8.0, 8.0, 0.10, 0.02
    for i in range(480):
        p = scene.bowl.data.root_pos_w
        v = scene.bowl.data.root_lin_vel_w
        w = scene.bowl.data.root_ang_vel_w
        q = scene.bowl.data.root_quat_w
        f_w = c.bowl_mass * 9.81 * ez + kp * (tgt - p) - kd * v
        f_norm = f_w.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        f_w = f_w * (f_norm.clamp(max=clamp) / f_norm)
        f_b = quat_apply_inverse(q, f_w)
        axis = quat_apply(q, ez)
        t_w = ku * torch.cross(axis, ez, dim=-1) - kw * w
        t_b = quat_apply_inverse(q, t_w)
        scene.bowl.set_external_force_and_torque(f_b.reshape(n, 1, 3), t_b.reshape(n, 1, 3))
        env.step(no_action)
        if not bool(scene.bowl_in_bay()[0]) \
                and float((scene.bowl.data.root_pos_w - origin)[0, 2]) > 0.43:
            break
    # keep holding while we check the latch (bowl is in free air, grasp still on)
    assert bool(scene._out_l[0]), "bowl_out must latch during the force lift"
    assert not bool(scene.bowl_in_bay()[0]), "bowl must be clear of the bay"
    print(f"[solve] lift: bowl at z={float((scene.bowl.data.root_pos_w - origin)[0, 2]):+.3f} "
          f"(free air above the cabinet)", flush=True)

    # ---------------- phase 3: transport over the panel, release, gravity seats ------------------
    # TRANSPORT ONLY: the held bowl is relocated in free air to 2 cm above the
    # panel's top face (panel-frame, clear of the handle bar) and released.
    sp = scene.slab.data.root_pos_w
    sq = scene.slab.data.root_quat_w
    local = torch.tensor([0.0, 0.05, c.slab_ht + c.bowl_org + 0.020],
                         device=device).expand(n, 3)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = sp + quat_apply(sq, local)
    st[:, 3] = 1.0
    scene.bowl.write_root_state_to_sim(st, torch.arange(n, device=device))
    scene.bowl.set_external_force_and_torque(zero, zero)
    step(300)   # free fall 2 cm, seat on the panel, everything settles
    report("seat")
    assert bool(scene.bowl_on_slab()[0]), "bowl must rest upright on the panel top"
    assert bool(scene._seat_l[0]), "seat latch must fire once settled"
    assert bool(scene.success()[0]), "success must hold after the seat settles"
    s = print_score("P3 bowl released over the panel; gravity seated it")
    assert s >= s_prev - 1e-6, "score decreased across P3"
    assert s >= 1.0 - 1e-6, "success must score 1.0"
    s_prev = s

    # ---------------- persistence (>= 3 simulated seconds, hands-off) ----------------------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s_final = print_score("P-final persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s_final >= s_prev - 1e-6
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

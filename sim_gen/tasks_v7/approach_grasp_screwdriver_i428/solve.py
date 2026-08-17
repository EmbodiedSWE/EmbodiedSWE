"""Teleport solution for CorbelReachScene (sim_gen task `approach_grasp_screwdriver_i428`)
— the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY, and every write ends in FREE SPACE:
  P1 — the corbel is built bottom-up: each plank (graspable: the jaw pinches its
  60 mm width) is carried to a HOVER pose 3 mm above its course — long axis along x,
  centred on y = 0, far edge stepped past the cliff by scaled harmonic offsets
  [L/8, L/6, L/4, L/2] * (5/6) * s, where s is chosen so the four offsets sum to
  R + 0.005 — velocities zeroed. Each plank FALLS and SETTLES onto the deck / the
  course below through real contact, and is verified settled near its commanded pose
  before the next course is placed. Every partial stack is statically stable (each
  interface at <= ~70% of its critical overhang), so the structure is a rested
  construct found by physics, never a wedged or pre-compressed write.
  P2 — the last 15 mm of reach are earned through CONTACT DYNAMICS: a capped
  velocity-servo force at the top plank's centre (the push of a fingertip) slides it
  forward along the course below until its far edge passes R + 0.018; the force is
  cut and the stack settles. The final, judged reach is the product of friction,
  gravity, and the counterweight courses — not of any write.
  Build quirk handled: on this forge build `is_global=True` is silently ignored
  (deprecated wrench path) and forces apply in the BODY frame; the solve rotates the
  desired world force into the body frame each step (quat_apply_inverse).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is streak-latched), then holds HANDS-OFF for >= 3.3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

Run (forge): python -u -m simgen_tasks.approach_grasp_screwdriver_i428.solve --headless [--seed N]
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

from isaaclab.utils.math import quat_apply_inverse

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

HARMONIC_BASE = (0.025, 1.0 / 30.0, 0.05, 0.10)  # [L/8, L/6, L/4, L/2] * 5/6, L = 0.24
SLIDE = 0.015  # final top-course advance earned through contact (m)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.corbel_reach")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def plank_p(k: int) -> torch.Tensor:
        return (scene.planks[k].data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        cx, cz = scene._plank_extremes()
        vl = [float(pl.data.root_lin_vel_w[0].norm()) for pl in scene.planks]
        va = [float(pl.data.root_ang_vel_w[0].norm()) for pl in scene.planks]
        print(f"[solve] {tag:16s} | reach_now={float(scene.reach_now()[0]):+.4f} "
              f"R={float(scene.reach_R[0]):.4f} corner_x="
              f"{[f'{float(v):+.3f}' for v in cx[0]]} min_z="
              f"{[f'{float(v):.3f}' for v in cz[0]]} "
              f"lin={[f'{v:.3f}' for v in vl]} ang={[f'{v:.2f}' for v in va]} "
              f"latch={float(scene.reach_latch[0]):.3f} "
              f"streak={int(scene._still[0])} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline --------------------------------------
    step(90)
    r_line = float(scene.reach_R[0])
    spawns = [tuple(round(float(v), 3) for v in plank_p(k)[:2]) for k in range(4)]
    print(f"[solve] layout readback (seed {args.seed}): R={r_line:.4f} "
          f"plank_xy={spawns}", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: build the corbel (hover TRANSPORT + contact settle) ----------
    # Offsets scaled so their sum = R + 0.005; the last SLIDE mm come from phase 2.
    s_scale = (r_line + 0.005) / sum(HARMONIC_BASE)
    offs = [b * s_scale for b in HARMONIC_BASE]
    edges = []
    acc = 0.0
    for o in offs:
        acc += o
        edges.append(acc)  # far-edge x of each course, bottom -> top
    # The top course is dropped SLIDE short of its final position.
    edges[3] -= SLIDE
    margins = [offs[k] / crit for k, crit in enumerate((0.03, 0.04, 0.06, 0.12))]
    print(f"[solve] corbel plan: s={s_scale:.3f} offsets="
          f"{[f'{o * 1000:.1f}mm' for o in offs]} (fractions of critical: "
          f"{[f'{m:.2f}' for m in margins]}) edges={[f'{e:.4f}' for e in edges]} "
          f"top after slide -> {edges[3] + SLIDE:.4f} (line {r_line:.4f})", flush=True)
    assert all(m < 0.80 for m in margins), "corbel plan must keep >= 20% static margin"

    prev = s0
    for k in range(4):
        rest_z = c.ped_h + c.plank_t / 2 + k * c.plank_t
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = edges[k] - c.plank_l / 2
        st[:, 2] = rest_z + 0.003  # hover 3 mm above the course: falls + settles
        st[:, 3] = 1.0  # long axis along x (the jaw re-orients a held plank)
        st[:, 0:3] += scene.env_origins
        scene.planks[k].write_root_state_to_sim(st, all_ids)
        step(60)  # fall + settle through real contact (~0.5 s)
        p = plank_p(k)
        ok_pose = (abs(float(p[0]) - (edges[k] - c.plank_l / 2)) < 0.008
                   and abs(float(p[2]) - rest_z) < 0.006
                   and abs(float(p[1])) < 0.010)
        report(f"course-{k}")
        if not ok_pose:
            print(f"SIM_GEN_SOLVE: FAIL (course {k} did not settle in place)", flush=True)
            os._exit(1)
        sk = print_score(f"P1.{k} course {k} dropped + settled")
        assert sk >= prev - 1e-6, "score decreased during construction"
        prev = sk
    step(80)  # let the finished 4-course stack prove itself + mature the latch streak
    report("stack-built")
    s1 = print_score("P1 corbel built (all courses settled)")
    assert s1 >= prev - 1e-6

    # ---------------- phase 2: contact slide of the top course ------------------------------
    # Capped velocity-servo force at the top plank's centre; the wrench is applied in
    # the plank's BODY frame on this build, so rotate the world force each step.
    target_edge = edges[3] + SLIDE
    m_pl = c.plank_mass
    # Feedforward + servo: a pure velocity servo tops out at m*gain*v_des ~ 0.2 N,
    # far below the ~1.6 N static-friction breakaway (mu_s * m * g) — the classic
    # stall. Bias with a feedforward just above kinetic friction and escalate the
    # CAP on stagnation; the servo term brakes once v_des is exceeded.
    gain, cap = 30.0, 2.2
    ff = 1.05 * (c.friction_plank - 0.1) * m_pl * 9.81  # ~1.44 N
    best = -1.0
    stagnant = 0
    slid_ok = False
    top = scene.planks[3]
    for i in range(1200):
        cx, _cz = scene._plank_extremes()
        edge = float(cx[0, 3])
        if edge >= target_edge - 0.002:
            slid_ok = True
            break
        v = top.data.root_lin_vel_w[0]
        p = plank_p(3)
        fx = ff + m_pl * gain * (0.03 - float(v[0]))
        fy = m_pl * gain * (max(-0.02, min(0.02, -2.0 * float(p[1]))) - float(v[1]))
        fx = max(-cap, min(cap, fx))
        fy = max(-1.0, min(1.0, fy))
        f_w = torch.tensor([fx, fy, 0.0], device=device).view(1, 3).expand(n, 3)
        f_b = quat_apply_inverse(top.data.root_quat_w, f_w)
        top.set_external_force_and_torque(f_b.view(n, 1, 3).contiguous(), zero_wrench,
                                          env_ids=all_ids)
        env.step(no_action)
        if edge > best + 0.0005:
            best = edge
            stagnant = 0
        else:
            stagnant += 1
        if stagnant >= 150:
            cap = min(cap * 1.3, 4.0)  # static-friction breakaway headroom
            stagnant = 0
            print(f"[solve] slide stagnant at edge={edge:+.4f}, cap={cap:.2f} N",
                  flush=True)
        if i % 300 == 299:
            report(f"sliding-{i + 1}")
    top.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    if not slid_ok:
        report("SLIDE-TIMEOUT")
        print("SIM_GEN_SOLVE: FAIL (top course never crossed the line)", flush=True)
        os._exit(1)
    step(90)  # settle + mature the latch streak
    report("slid")
    s2 = print_score("P2 top course slid past the line by contact + settle")
    assert s2 >= s1 - 1e-6, "score decreased across the slide"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after slide+settle)", flush=True)
        os._exit(1)

    # ---------------- phase 3: persistence (>= 3.3 simulated seconds, hands-off) ------------
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
    main()

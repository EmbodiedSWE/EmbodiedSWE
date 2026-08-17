"""Teleport solution for ScreedPatchScene (sim_gen task
libero_kitchen_scene2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle_i422)
— the task's legitimacy certificate.

Teleports handle TRANSPORT ONLY; every load-bearing interaction runs through contact
dynamics:

  * every slab is teleported to a HOVER 30 mm above its target seat (outside the 6 mm
    interred band and the 4 mm flush band) and RELEASED: the keystone lands on the pit
    floor, each paver lands on the course below it, all by gravity and contact. No slab is
    ever written into a scoring pose;
  * the paver COUNT is read off the episode (depth readback -> depth-1 pavers), which is
    the actual skill the rubric prices: a fixed count fails half the episodes;
  * the SCREED PASS — the phase the score's motion gate exists for — is a pure force
    interaction: a horizontal bang-bang force on the bar (the body the arm would drag by
    its handle) under a speed cap, plus a small lateral P-D steer toward the pit
    centreline (uneven runner friction walks an unsteered bar sideways), slides it from
    its near-side park, across the finished patch riding at deck level, to the far side,
    then an active reverse-force brake stops it. The pass credit latches only while the bar is genuinely in deck-level motion over
    the completed patch;
  * failed attempts rebuild the START of the phase by transport teleports (slab back to a
    hover, bar back to its park) and retry through physics.

Phases (score latches must be non-decreasing along the run; SIM_GEN_SCORE is printed at
every phase boundary):
  0. reset(seed), settle, layout readback (pit, depth, keystone slot)      -> 0.000
  1. inter the keystone: hover-release onto the pit floor                  -> 0.150
  2. pave: hover-release depth-1 pavers onto the keystone (patch flush)    -> 0.500
  3. screed: force-drag the bar across the patch, brake, park clear        -> 1.000
  4. persistence: >= 3 simulated seconds hands-off, success() must hold    -> 1.000
                                                                             SIM_GEN_SOLVE

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

# Kit teardown hangs leave GPU zombies: global watchdog long before the forge timeout.
threading.Timer(1350.0, lambda: os._exit(4)).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.screed_patch")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    origin = env.iscene.env_origins
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    last_score = [0.0]

    def phase_score(tag: str) -> None:
        s = float(scene.score()[0])
        print(f"[solve] {tag}: interred={bool(scene.interred()[0])} "
              f"covered={bool(scene.covered()[0])} flush={bool(scene.flush()[0])} "
              f"swept={bool(scene.ever_swept[0])} clear={bool(scene.bar_clear()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])}",
              flush=True)
        if s + 1e-6 < last_score[0]:
            print(f"[solve] FATAL: score decreased {last_score[0]:.3f} -> {s:.3f}", flush=True)
            print("SIM_GEN_SOLVE: FAIL", flush=True)
            os._exit(1)
        last_score[0] = s
        print(f"SIM_GEN_SCORE {s:.3f}", flush=True)

    def settle(max_steps: int = 500, poll: int = 10) -> None:
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if bool(scene.settled()[0]):
                return

    def local(body) -> torch.Tensor:
        return (body.data.root_pos_w[0] - origin[0]).clone()

    def write_body(body, x: float, y: float, z: float, yaw: float = 0.0) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = math.cos(yaw / 2)
        st[:, 6] = math.sin(yaw / 2)
        st[:, 0:3] += origin
        body.write_root_state_to_sim(st, all_ids)

    def clear_force() -> None:
        scene.bar.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def bar_force(fx: float, fy: float = 0.0) -> None:
        f_w = torch.zeros(n, 1, 3, device=device)
        f_w[:, 0, 0] = fx
        f_w[:, 0, 1] = fy
        scene.bar.set_external_force_and_torque(f_w, zero_wrench, env_ids=all_ids,
                                                is_global=True)

    def steer_fy(py: float) -> float:
        """Lateral P-D steering toward the pit centreline (the arm would steer its drag
        too): uneven runner friction yaws the bar and walks it sideways, and over the OPEN
        pit a ~30 mm drift drops a runner into the opening."""
        by = float(scene.bar.data.root_pos_w[0, 1] - origin[0, 1])
        vy = float(scene.bar.data.root_lin_vel_w[0, 1])
        return max(-0.8, min(0.8, -6.0 * (by - py) - 1.2 * vy))

    def drop_slab(body, x: float, y: float, seat_z: float) -> None:
        """Transport teleport to a hover 30 mm above the target seat (outside every scoring
        band), release, settle: the slab SEATS by gravity and contact."""
        write_body(body, x, y, seat_z + 0.030)
        settle(max_steps=300)

    # ---------------- phase 0: reset + readback ----------------------------------------------
    env.reset(seed=args.seed)
    step(60)  # settle the racked slabs and parked bar
    pit = scene.pit_xy[0].clone()
    px, py = float(pit[0]), float(pit[1])
    depth = int(round(float(scene.depth_units[0])))
    floor_top = float(scene.floor_top()[0])
    floor_phys = float(scene.pit_floor.data.root_pos_w[0, 2] - origin[0, 2]) + c.floor_t / 2
    print(f"[solve] seed={args.seed} pit=({px:+.3f},{py:+.3f}) depth={depth} courses "
          f"(floor_top={floor_top:.3f}, physical readback {floor_phys:.3f}) "
          f"side={float(scene.depot_side[0]):+.0f} key_slot={int(float(scene.key_slot[0]))} "
          f"bar_park_x={float(scene.bar_park_x[0]):+.3f}", flush=True)
    print(f"[solve] plan: inter keystone at pit floor, pave {depth - 1} paver(s), screed",
          flush=True)
    phase_score("phase0 reset")

    # ---------------- phase 1: inter the keystone ----------------------------------------------
    interred = False
    for attempt, (dx, dy) in enumerate(((0.0, 0.0), (0.004, 0.0), (-0.004, 0.004))):
        drop_slab(scene.keystone, px + dx, py + dy, floor_top + c.slab_t / 2)
        interred = bool(scene.interred()[0])
        k = local(scene.keystone)
        print(f"[solve] keystone drop {attempt}: rest=({float(k[0]):+.3f},"
              f"{float(k[1]):+.3f},{float(k[2]):.4f}) seat_z={floor_top + c.slab_t / 2:.4f} "
              f"interred={interred}", flush=True)
        if interred:
            break
    if not interred:
        print("[solve] FATAL: keystone never interred", flush=True)
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)
    phase_score("phase1 keystone interred")

    # ---------------- phase 2: pave depth-1 courses --------------------------------------------
    for course in range(1, depth):
        paver = scene.pavers[course - 1]
        seat_z = floor_top + (course + 0.5) * c.slab_t
        ok = False
        for attempt, (dx, dy) in enumerate(((0.0, 0.0), (0.004, 0.0), (-0.004, 0.004))):
            drop_slab(paver, px + dx, py + dy, seat_z)
            p = local(paver)
            ok = bool(scene.covered()[0]) and abs(float(p[2]) - seat_z) < 0.006
            print(f"[solve] paver course {course} drop {attempt}: "
                  f"rest_z={float(p[2]):.4f} seat_z={seat_z:.4f} ok={ok}", flush=True)
            if ok:
                break
        if not ok:
            print("[solve] FATAL: paver course never seated", flush=True)
            print("SIM_GEN_SOLVE: FAIL", flush=True)
            os._exit(1)
    settle(max_steps=300)
    top = max(float(local(s)[2]) for s in [scene.keystone] + scene.pavers[:depth - 1]) \
        + c.slab_t / 2
    print(f"[solve] patch top={top:.4f} deck={c.deck_top:.4f} "
          f"(flush band +/-{c.flush_tol * 1000:.0f} mm) flush={bool(scene.flush()[0])}",
          flush=True)
    phase_score("phase2 patch complete")

    # ---------------- phase 3: the screed pass -------------------------------------------------
    v_cap = 0.20
    done = False
    for attempt in range(3):
        f_drag = (1.5, 2.0, 2.5)[attempt]
        # drag: bang-bang +x force under the speed cap until well past the pit
        for _ in range(700):
            b = local(scene.bar)
            if float(b[0]) - px > 0.16:
                break
            vx = float(scene.bar.data.root_lin_vel_w[0, 0])
            bar_force(f_drag if vx < v_cap else 0.0, steer_fy(py))
            env.step(no_action)
        # brake: reverse force until slow
        for _ in range(150):
            v = scene.bar.data.root_lin_vel_w[0, :2]
            sp = float(v.norm())
            if sp < 0.06:
                break
            bar_force(-1.5 if float(v[0]) > 0 else 1.5, steer_fy(py))
            env.step(no_action)
        clear_force()
        settle(max_steps=300)
        b = local(scene.bar)
        done = bool(scene.ever_swept[0]) and bool(scene.bar_clear()[0])
        print(f"[solve] screed attempt {attempt}: f={f_drag} "
              f"bar=({float(b[0]):+.3f},{float(b[1]):+.3f},{float(b[2]):.4f}) "
              f"swept={bool(scene.ever_swept[0])} clear={bool(scene.bar_clear()[0])}",
              flush=True)
        if done:
            break
        # rebuild the phase START (transport only): bar back to its park, at rest
        write_body(scene.bar, float(scene.bar_park_x[0]), py, c.bar_rest_z + 0.001)
        settle(max_steps=200)
    if not done:
        print("[solve] FATAL: screed pass never completed", flush=True)
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)

    settle(max_steps=600)
    ok = bool(scene.success()[0])
    print(f"[solve] goal state reached: success={ok} score={float(scene.score()[0]):.3f}",
          flush=True)
    phase_score("phase3 screed done")

    # ---------------- phase 4: hands-off persistence (>= 3 simulated seconds) ------------------
    persist_ok = ok
    if ok:
        held = 0
        while held < 400:  # 400 steps at 120 Hz = 3.33 s, no further intervention
            step(40)
            held += 40
            if not bool(scene.success()[0]):
                persist_ok = False
                break
        phase_score("persistence")

    if persist_ok and bool(scene.success()[0]) and float(scene.score()[0]) == 1.0:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
        code = 0
    else:
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        code = 1

    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()

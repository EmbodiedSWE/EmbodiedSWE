"""Teleport solution for GimbalServiceScene (sim_gen task
libero_kitchen_scene2_stack_the_middle_black_bowl_on_the_back_black_bowl_i377) — the
task's legitimacy certificate.

Teleports handle TRANSPORT ONLY; every load-bearing interaction runs through contact
dynamics and gravity:

  * the SOCKET bowl is hover-released 22 mm above its ring seat, face-aligned to the
    tilted shelf, with zero velocity. It falls onto the slick slab, slides downhill a
    few mm and is ARRESTED by the ring curb — the curb, not the writer, holds it (the
    bare slab cannot: pair mu 0.03 << tan 16 deg, proven by the smoke battery);
  * the GIMBAL bowl is hover-released WORLD-LEVEL above the socket's mouth. The
    polished ball bottom falls through the mouth, wedges on the polished socket seat,
    and the below-pivot ballast (34 mm) keeps the cup level ON ITS OWN while the
    socket under it stays tilted 16-20 deg with the shelf. Nothing ever writes the
    cup's orientation after release — the self-leveling that the `nested` gate
    demands is pure contact physics (the slick ball/seat pairing lets the righting
    torque slide the wedged contacts), and a rigid ride at shelf pitch could never
    pass the 8 deg gate;
  * the CUBE is hover-released above the (self-leveled) cup and must fall in, land on
    the cup floor and rest world-level by itself. Placed directly in the tilted
    socket instead, it rests face-flat on the tilted floor, contained by the lower
    wall, at 16-20 deg — failing the 10 deg gate by geometry (smoke-proven).

Phases (SIM_GEN_SCORE printed at every boundary, non-decreasing):
  0. reset(seed), settle at the parks, layout readback                    -> 0.000
  1. socket bowl: hover-release into the ring curb                        -> 0.200
  2. gimbal bowl: hover-release through the mouth; cup self-levels        -> 0.500
  3. cube: hover-release into the level cup                               -> 0.800
  4. settle to full success                                               -> 1.000
  5. persistence: >= 3.3 simulated seconds hands-off, success holds       -> SIM_GEN_SOLVE

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

from isaaclab.utils.math import quat_apply

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Kit teardown hangs leave GPU zombies: global watchdog long before the forge timeout.
_wd = threading.Timer(1350.0, lambda: os._exit(4))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.gimbal_service")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    origin = env.iscene.env_origins
    z0 = c.surface_z

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    last_score = [0.0]

    def phase_score(tag: str) -> None:
        s = float(scene.score()[0])
        print(f"[solve] {tag}: ringed={bool(scene.a_in_ring()[0])} "
              f"nested={bool(scene.b_in_a()[0])} served={bool(scene.cube_in_b()[0])} "
              f"ever=({bool(scene.ever_ringed[0])},{bool(scene.ever_nested[0])},"
              f"{bool(scene.ever_served[0])}) settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])}", flush=True)
        if s + 1e-6 < last_score[0]:
            print(f"[solve] FATAL: score decreased {last_score[0]:.3f} -> {s:.3f}",
                  flush=True)
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

    def write_state(body, pos_w: torch.Tensor, quat_w: torch.Tensor) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat_w
        body.write_root_state_to_sim(st, all_ids)

    level_q = torch.zeros(n, 4, device=device)
    level_q[:, 0] = 1.0

    # ---------------- phase 0: reset + settle at the parks + layout readback ----------------
    env.reset(seed=args.seed)
    settle(max_steps=360)
    a_park_w = scene.bowl_a.data.root_pos_w.clone()
    b_park_w = scene.bowl_b.data.root_pos_w.clone()
    cube_park_w = scene.cube.data.root_pos_w.clone()
    pitch = float(scene.station_pitch_deg[0])
    seat = (scene.seat_center_w()[0] - origin[0]).tolist()
    print(f"[solve] seed={args.seed} pitch={pitch:.2f}deg "
          f"yaw={float(scene.station_yaw[0]):.2f}rad "
          f"center={[round(v, 3) for v in (scene.station_pos[0] - origin[0]).tolist()]} "
          f"seat_w={[round(v, 3) for v in seat]} "
          f"a_side={float(scene.a_side[0]):+.0f} cube_side={float(scene.cube_side[0]):+.0f}",
          flush=True)
    if bool(scene.success()[0]) or float(scene.score()[0]) > 0.0:
        print("[solve] FATAL: score nonzero at reset", flush=True)
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)
    phase_score("phase0 reset")

    normal = scene.station_up_w()

    # ---------------- phase 1: socket bowl hover-released into the ring curb ----------------
    # Aligned to the shelf, 22 mm above the seat; it lands, slides downhill on the
    # slick coat and the curb arrests it. Failed drops re-park and retry with a nudge.
    ringed = False
    for attempt in range(4):
        dx, dy = ((0.0, 0.0), (0.004, 0.0), (-0.004, 0.004), (0.0, -0.004))[attempt]
        nudge = quat_apply(scene.station_quat,
                           torch.tensor([dx, dy, 0.022], device=device).expand(n, 3))
        write_state(scene.bowl_a, scene.seat_center_w() + nudge, scene.station_quat)
        waited = 0
        while waited < 600:
            step(10)
            waited += 10
            if bool(scene.a_in_ring()[0]) and bool(scene.settled()[0]):
                break
        settle(max_steps=240)
        ringed = bool(scene.a_in_ring()[0]) and bool(scene.settled()[0])
        loc = scene.st_local(scene.bowl_a.data.root_pos_w)[0].tolist()
        print(f"[solve] socket drop attempt {attempt}: ringed={ringed} "
              f"st_loc={[round(v, 4) for v in loc]}", flush=True)
        if ringed:
            break
        write_state(scene.bowl_a, a_park_w, level_q)
        settle(max_steps=240)
    if not ringed:
        print("[solve] FATAL: socket bowl never seated in the ring", flush=True)
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)
    phase_score("phase1 socket ringed")

    # ---------------- phase 2: gimbal bowl hover-released through the mouth ------------------
    # Released WORLD-LEVEL above the socket mouth; the ball falls through, seats on
    # the socket floor, and the ballast keeps the cup level — never written again.
    nested = False
    for attempt in range(4):
        dx, dy = ((0.0, 0.0), (0.005, 0.0), (-0.005, 0.005), (0.0, -0.005))[attempt]
        above = quat_apply(scene.bowl_a.data.root_quat_w,
                           torch.tensor([dx, dy, 0.115], device=device).expand(n, 3))
        write_state(scene.bowl_b, scene.bowl_a.data.root_pos_w + above, level_q)
        settle(max_steps=500)
        nested = (bool(scene.b_in_a()[0]) and bool(scene.a_in_ring()[0])
                  and bool(scene.settled()[0]))
        up = float(scene._up_z(scene.bowl_b.data.root_quat_w)[0])
        print(f"[solve] gimbal drop attempt {attempt}: nested={nested} "
              f"cup_up_z={up:.4f} (level gate {float(torch.cos(torch.deg2rad(torch.tensor(c.level_max_deg)))):.4f})",
              flush=True)
        if nested:
            break
        write_state(scene.bowl_b, b_park_w, level_q)
        settle(max_steps=240)
    if not nested:
        print("[solve] FATAL: gimbal bowl never nested level in the socket", flush=True)
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)
    phase_score("phase2 gimbal nested")

    # ---------------- phase 3: cube hover-released into the self-leveled cup -----------------
    served = False
    for attempt in range(4):
        dx, dy = ((0.0, 0.0), (0.006, 0.0), (-0.006, 0.006), (0.0, -0.006))[attempt]
        drop = scene.bowl_b.data.root_pos_w + torch.tensor(
            [dx, dy, 0.130], device=device).expand(n, 3)
        write_state(scene.cube, drop, level_q)
        settle(max_steps=400)
        served = (bool(scene.cube_in_b()[0]) and bool(scene.b_in_a()[0])
                  and bool(scene.a_in_ring()[0]))
        print(f"[solve] cube drop attempt {attempt}: served={served} "
              f"in_cup={bool(scene.cube_in_b()[0])}", flush=True)
        if served:
            break
        write_state(scene.cube, cube_park_w, level_q)
        step(30)
    if not served:
        print("[solve] FATAL: cube never rested level inside the gimbal cup", flush=True)
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)
    phase_score("phase3 cube served")

    # ---------------- phase 4: settle to full success ----------------------------------------
    settle(max_steps=600)
    ok = bool(scene.success()[0])
    print(f"[solve] goal state reached: success={ok} "
          f"score={float(scene.score()[0]):.3f}", flush=True)
    if not ok:
        print("[solve] FATAL: success() false after settling", flush=True)
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)
    phase_score("phase4 success")

    # ---------------- phase 5: hands-off persistence (>= 3.3 simulated seconds) --------------
    persist_ok = True
    heldsteps = 0
    while heldsteps < 400:  # 400 steps at 120 Hz = 3.33 s, no further intervention
        step(40)
        heldsteps += 40
        if not bool(scene.success()[0]):
            persist_ok = False
            break
    phase_score("phase5 persistence")

    if persist_ok and bool(scene.success()[0]) and float(scene.score()[0]) == 1.0:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
        code = 0
    else:
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        code = 1

    exit_t = threading.Timer(10.0, lambda: os._exit(code))
    exit_t.daemon = True
    exit_t.start()
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
    except Exception as e:  # noqa: BLE001
        print(f"[solve] FATAL: unhandled {type(e).__name__}: {e}", flush=True)
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)

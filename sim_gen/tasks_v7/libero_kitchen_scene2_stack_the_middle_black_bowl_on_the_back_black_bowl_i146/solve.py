"""Teleport solution for CounterweightVaultScene (sim_gen task
libero_kitchen_scene2_stack_the_middle_black_bowl_on_the_back_black_bowl_i146) — the
task's legitimacy certificate.

Teleports handle TRANSPORT ONLY; both load-bearing interactions run through contact
dynamics and gravity:

  * the GATE is opened by WEIGHT, never written open: the black counterweight bowl is
    teleported to a hover ~18 mm above the brass pan floor (outside the 12 mm pan-seat
    scoring z-band, below the pan wall tops so it cannot perch on a wall) with zero
    velocity and RELEASED. It falls into the pan
    under gravity; the loaded pan side out-torques the flap side and the rotor swings
    ~100 deg open about its tilted axis entirely on its own. The rotor is never teleported
    or forced — if the bowl misses the pan, the gate stays closed (or re-closes) and the
    attempt re-parks the bowl and re-drops;
  * the DELIVERY is a gravity drop through the physically opened mouth: the cube is
    released ~35 mm above the vault rim (far outside its 8 mm scoring z-band) over the
    vault centre and must fall through the open mouth and come to rest inside the white
    bowl by itself. With the gate closed this exact drop is impossible (the flap covers
    the mouth) — the smoke battery proves that side.

Phases (SIM_GEN_SCORE printed at every boundary, non-decreasing):
  0. reset(seed), settle to the gravity-closed rest, layout readback     -> 0.000
  1. counterweight: hover-release into the pan; gate swings open         -> 0.500
     (0.20 loaded + 0.30 held latch once the rotor rests at its stop)
  2. cube: hover-release through the opened mouth into the white bowl    -> 0.800
  3. settle to full success                                              -> 1.000
  4. persistence: >= 3.3 simulated seconds hands-off, success holds      -> SIM_GEN_SOLVE

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

from isaaclab.utils.math import quat_apply_inverse

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
    env = ENVS.get("simgen.counterweight_vault")().build(num_envs=args.num_envs,
                                                         device=device)
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
        print(f"[solve] {tag}: open={float(scene.open_angle_deg()[0]):.1f}deg "
              f"loaded={bool(scene.ever_loaded[0])} held={bool(scene.ever_held[0])} "
              f"delivered={bool(scene.ever_delivered[0])} "
              f"in_pan={bool(scene.bowl_in_pan()[0])} "
              f"cube_in={bool(scene.cube_in_vbowl()[0])} "
              f"home={bool(scene.vbowl_home()[0])} settled={bool(scene.settled()[0])} "
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

    def write_body(body, x: float, y: float, z: float, yaw: float = 0.0) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = math.cos(yaw / 2)
        st[:, 6] = math.sin(yaw / 2)
        st[:, 0:3] += origin
        body.write_root_state_to_sim(st, all_ids)

    # ---------------- phase 0: reset + settle onto the gravity rest + readback -------------
    env.reset(seed=args.seed)
    settle(max_steps=360)
    bowl_p = (scene.cw_bowl.data.root_pos_w[0] - origin[0]).tolist()
    cube_p = (scene.cube.data.root_pos_w[0] - origin[0]).tolist()
    pan_w = (scene.pan_center_w()[0] - origin[0]).tolist()
    ang0 = float(scene.open_angle_deg()[0])
    print(f"[solve] seed={args.seed} rest_open={ang0:.1f}deg "
          f"bowl_mass={float(scene.bowl_mass[0]):.3f}kg "
          f"bowl_park={[round(v, 3) for v in bowl_p]} "
          f"cube_park={[round(v, 3) for v in cube_p]} "
          f"pan_center={[round(v, 3) for v in pan_w]} "
          f"sides=({float(scene.bowl_side[0]):+.0f},{float(scene.cube_side[0]):+.0f})",
          flush=True)
    if ang0 > c.closed_max_deg:
        print(f"[solve] FATAL: gate not closed at rest ({ang0:.1f} deg)", flush=True)
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)
    phase_score("phase0 reset")

    # ---------------- phase 1: counterweight hover-released into the pan --------------------
    # The bowl is dropped from ABOVE the pan walls / outside the seat z-band and must
    # land, seat, and swing the gate open purely under gravity. The rotor is never
    # touched. Failed drops re-park the bowl, let the empty gate re-close, and retry.
    park = (bowl_p[0], bowl_p[1])
    held = False
    for attempt in range(4):
        dx, dy = ((0.0, 0.0), (0.004, 0.0), (-0.004, 0.004), (0.0, -0.004))[attempt]
        pan = scene.pan_center_w()[0] - origin[0]  # current pan pose (gate at rest)
        write_body(scene.cw_bowl, float(pan[0]) + dx, float(pan[1]) + dy,
                   float(pan[2]) + 0.018)
        # wait for the swing: gate open AND rotor at rest AND bowl still in the pan
        waited = 0
        while waited < 900:
            step(10)
            waited += 10
            if (bool(scene.gate_open()[0])
                    and float(scene.rotor_axis_w()[0]) < c.settle_rotor_w
                    and bool(scene.bowl_in_pan()[0])):
                break
        settle(max_steps=300)
        ang = float(scene.open_angle_deg()[0])
        held = (bool(scene.bowl_in_pan()[0]) and bool(scene.gate_open()[0])
                and bool(scene.settled()[0]))
        bw = (scene.cw_bowl.data.root_pos_w[0] - origin[0]).tolist()
        loc = quat_apply_inverse(
            scene.rotor.data.root_quat_w,
            scene.cw_bowl.data.root_pos_w - scene.pan_center_w())[0].tolist()
        print(f"[solve] drop attempt {attempt}: open={ang:.1f}deg "
              f"in_pan={bool(scene.bowl_in_pan()[0])} held={held} "
              f"bowl_w={[round(v, 3) for v in bw]} "
              f"pan_loc={[round(v, 3) for v in loc]}", flush=True)
        if held:
            break
        # re-park the bowl; the emptied gate must re-close by itself before the retry
        write_body(scene.cw_bowl, park[0], park[1], z0 + 0.001)
        waited = 0
        while waited < 600:
            step(20)
            waited += 20
            if (float(scene.open_angle_deg()[0]) < c.closed_max_deg
                    and float(scene.rotor_axis_w()[0]) < c.settle_rotor_w):
                break
        print(f"[solve] re-parked; gate back to "
              f"{float(scene.open_angle_deg()[0]):.1f}deg", flush=True)
    if not held:
        print("[solve] FATAL: counterweight never held the gate open", flush=True)
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)
    phase_score("phase1 gate held open")

    # ---------------- phase 2: cube hover-released through the opened mouth ------------------
    vx, vy = c.vault_xy
    delivered = False
    for attempt in range(4):
        dx, dy = ((0.0, 0.0), (0.006, 0.0), (-0.006, 0.006), (0.0, -0.006))[attempt]
        # aim at the white bowl's actual centre (recess play + fresh yaw per episode)
        vb = scene.v_bowl.data.root_pos_w[0] - origin[0]
        write_body(scene.cube, float(vb[0]) + dx, float(vb[1]) + dy,
                   z0 + c.vault_h + 0.035 + c.cube_size / 2)
        settle(max_steps=400)
        delivered = bool(scene.cube_in_vbowl()[0]) and bool(scene.vbowl_home()[0])
        print(f"[solve] cube drop attempt {attempt}: in={bool(scene.cube_in_vbowl()[0])} "
              f"home={bool(scene.vbowl_home()[0])}", flush=True)
        if delivered:
            break
        # re-park (transport only) and retry with a nudged aim point
        write_body(scene.cube, cube_p[0], cube_p[1], z0 + c.cube_size / 2 + 0.001)
        step(30)
    if not delivered:
        print("[solve] FATAL: cube never rested inside the white bowl", flush=True)
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)
    phase_score("phase2 cube delivered")

    # ---------------- phase 3: settle to full success ----------------------------------------
    settle(max_steps=600)
    ok = bool(scene.success()[0])
    print(f"[solve] goal state reached: success={ok} "
          f"score={float(scene.score()[0]):.3f}", flush=True)
    if not ok:
        print("[solve] FATAL: success() false after settling", flush=True)
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)
    phase_score("phase3 success")

    # ---------------- phase 4: hands-off persistence (>= 3.3 simulated seconds) --------------
    persist_ok = True
    heldsteps = 0
    while heldsteps < 400:  # 400 steps at 120 Hz = 3.33 s, no further intervention
        step(40)
        heldsteps += 40
        if not bool(scene.success()[0]):
            persist_ok = False
            break
    phase_score("phase4 persistence")

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

"""Teleport solution for ServiceCarouselScene (sim_gen task
libero_kitchen_scene2_put_the_black_bowl_in_the_middle_on_the_plate_i78) — the task's
legitimacy certificate.

Teleports handle TRANSPORT of the free bowl ONLY; every load-bearing interaction runs
through contact dynamics:

  * BOTH carousel rotations (fetch and return) are driven by a physical z-torque on the
    dynamic disc through `set_external_force_and_torque` — a velocity-servo standing in for
    a finger pushing the peg handles. The plate (and later the loaded bowl) is carried
    purely by pocket contact + friction; nothing about the plate is ever written.
    (Known forge quirk: external wrenches are rotated by the body's rotation-since-reset;
    a pure z-torque on a body that only yaws is invariant under that rotation, so it is
    unaffected.)
  * the MIDDLE bowl is teleported through free air only: incremental raise off the counter,
    one hop above all geometry (carry z 0.42 > hutch roof top 0.347), incremental descent
    in the open loading-side column, then RELEASED ~28 mm above the plate — it seats on the
    plate under gravity and contact, it is NEVER spawned seated;
  * the loaded return ride is pure physics: friction holds the bowl on the plate while the
    servo parks the pocket on the serve stripe.

Phases (score latches must be non-decreasing along the run; SIM_GEN_SCORE is printed at
every phase boundary):
  0. reset(seed), settle, readback (start azimuth, middle-bowl identity)   -> 0.000
  1. FETCH: torque-servo the carousel until the plate parks on the open
     loading side (azimuth ~180 deg), let it settle                        -> 0.250
  2. LOAD: raise/hop/lower the MIDDLE bowl, release above the plate,
     gravity seating (retry with small offsets if it slides off)           -> 0.600
  3. RETURN: torque-servo the loaded carousel back to the serve stripe
     (azimuth 0), zero torque, settle                                      -> 1.000
  4. persistence: >= 3 simulated seconds hands-off, success() must hold    -> SIM_GEN_SOLVE

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


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.service_carousel")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    origin = env.iscene.env_origins

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    last_score = [0.0]

    def phase_score(tag: str) -> None:
        s = float(scene.score()[0])
        print(f"[solve] {tag}: fetched={bool(scene.ever_fetched[0])} "
              f"loaded={bool(scene.ever_loaded[0])} "
              f"az={math.degrees(float(scene.plate_azimuth()[0])):+.1f}deg "
              f"on_plate={bool(scene.target_on_plate()[0])} "
              f"delivered={bool(scene.plate_delivered()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])}",
              flush=True)
        if s + 1e-6 < last_score[0]:
            print(f"[solve] FATAL: score decreased {last_score[0]:.3f} -> {s:.3f}", flush=True)
            print("SIM_GEN_SOLVE: FAIL", flush=True)
            os._exit(1)
        last_score[0] = s
        print(f"SIM_GEN_SCORE {s:.3f}", flush=True)

    # ---------------- carousel drive: physical z-torque velocity-servo -----------------------
    zero3 = torch.zeros(n, 1, 3, device=device)

    def set_disc_torque(tau: float) -> None:
        t = torch.zeros(n, 1, 3, device=device)
        t[:, 0, 2] = tau
        scene.disc.set_external_force_and_torque(zero3, t)

    def rotate_to(target: float, tag: str, w_max: float = 0.55, tau_max: float = 0.06,
                  max_steps: int = 5000) -> None:
        """Drive the disc yaw to `target` with a clamped torque servo (the finger-on-peg
        surrogate), then cut the torque and coast. Gentle: max tangential accel at the
        pocket radius stays far below the friction budget of the riding plate/bowl."""
        for _ in range(max_steps):
            yaw = float(scene.disc_yaw()[0])
            w = float(scene.disc.data.root_ang_vel_w[0, 2])
            err = wrap(target - yaw)
            if abs(err) < 0.04 and abs(w) < 0.06:
                break
            w_des = max(-w_max, min(w_max, 2.5 * err))
            tau = max(-tau_max, min(tau_max, 0.30 * (w_des - w)))
            set_disc_torque(tau)
            step(1)
        # active braking tail: hold the disc against residual coasting before releasing
        for _ in range(240):
            w = float(scene.disc.data.root_ang_vel_w[0, 2])
            if abs(w) < 0.01:
                break
            set_disc_torque(max(-tau_max, min(tau_max, -0.40 * w)))
            step(1)
        set_disc_torque(0.0)
        step(30)
        print(f"[solve] {tag}: disc yaw={math.degrees(float(scene.disc_yaw()[0])):+.1f}deg "
              f"w={float(scene.disc.data.root_ang_vel_w[0, 2]):+.3f}", flush=True)

    # ---------------- bowl transport (free air only) ------------------------------------------
    def bowl_state(b: int) -> torch.Tensor:
        return scene.bowls[b].data.root_state_w[all_ids].clone()

    def write_bowl(b: int, st: torch.Tensor) -> None:
        st = st.clone()
        st[:, 7:13] = 0.0
        scene.bowls[b].write_root_state_to_sim(st, all_ids)

    def flat_quat(yaw: float) -> tuple:
        return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

    def raise_bowl(b: int, z_to: float) -> None:
        """Incremental vertical lift off the counter: small pose writes with real physics
        steps between (the arm's own free path; nothing above the bowl row)."""
        st = bowl_state(b)
        z = float(st[0, 2])
        while z < z_to:
            z = min(z + 0.006, z_to)
            st = bowl_state(b)
            st[:, 2] = z
            write_bowl(b, st)
            step(2)

    def hop_bowl(b: int, x: float, y: float, z: float, yaw: float) -> None:
        """One free-space transport write ABOVE every obstacle (env-local x, y)."""
        st = bowl_state(b)
        st[:, 0] = x
        st[:, 1] = y
        st[:, 2] = z
        st[:, 3:7] = torch.tensor(flat_quat(yaw), device=device)
        st[:, 0:2] += origin[:, 0:2]
        write_bowl(b, st)
        step(2)

    def lower_bowl(b: int, z_to: float) -> None:
        """Incremental descent in the open column above the loading-side plate (the hutch
        covers only the far serve sector; this column is free air)."""
        st = bowl_state(b)
        z = float(st[0, 2])
        while z > z_to:
            z = max(z - 0.006, z_to)
            st = bowl_state(b)
            st[:, 2] = z
            write_bowl(b, st)
            step(2)

    def settle(max_steps: int = 600, poll: int = 10) -> None:
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if bool(scene.settled()[0]):
                return

    carry_z = 0.42  # above hutch roof top (0.347) and peg tops (0.322)

    def load_bowl(b: int) -> None:
        """Raise the bowl, carry it over, release ~28 mm above the plate; gravity + contact
        seat it. Retry with small lateral offsets if it skids off the rim."""
        plate_top = float(scene.plate.data.root_pos_w[0, 2]) + c.plate_h / 2
        release_z = plate_top + c.bowl_floor_h / 2 + 0.028
        offs = ((0.0, 0.0), (0.005, -0.004), (-0.005, 0.004), (0.004, 0.005))
        for attempt, (ox, oy) in enumerate(offs):
            raise_bowl(b, carry_z)
            pp = scene.plate.data.root_pos_w[0] - origin[0]
            hop_bowl(b, float(pp[0]) + ox, float(pp[1]) + oy, carry_z, 0.4 * attempt)
            lower_bowl(b, release_z)
            settle(max_steps=500)
            if bool(scene.target_on_plate()[0]):
                return
            print(f"[solve] load attempt {attempt} missed; retrying", flush=True)
        print("[solve] FATAL: middle bowl never seated on the plate", flush=True)
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)

    # ---------------- phase 0: reset + readback -----------------------------------------------
    env.reset(seed=args.seed)
    step(60)  # settle the authored layout
    mid = int(scene.mid_idx[0])
    slots = scene.slot_of[0].tolist()
    print(f"[solve] seed={args.seed} yaw0={math.degrees(float(scene.yaw0[0])):+.1f}deg "
          f"mid_bowl=bowl_{mid} slot_of={slots} "
          f"plate={(scene.plate.data.root_pos_w[0] - origin[0])[:2].tolist()} "
          f"disc_yaw={math.degrees(float(scene.disc_yaw()[0])):+.1f}deg", flush=True)
    phase_score("phase0 reset")

    # ---------------- phase 1: fetch — rotate the plate to the open loading side --------------
    rotate_to(math.pi, "phase1 servo", max_steps=5000)
    settle(max_steps=400)
    phase_score("phase1 fetch")

    # ---------------- phase 2: load — seat the MIDDLE bowl on the plate through gravity -------
    load_bowl(mid)
    phase_score("phase2 load")

    # ---------------- phase 3: return — rotate the loaded carousel to the serve stripe --------
    rotate_to(0.0, "phase3 servo", w_max=0.50, max_steps=6000)
    settle(max_steps=600)
    phase_score("phase3 return")

    if not bool(scene.success()[0]):
        settle(max_steps=600)
    ok = bool(scene.success()[0])
    print(f"[solve] goal state reached: success={ok} score={float(scene.score()[0]):.3f}",
          flush=True)
    phase_score("goal")

    # ---------------- phase 4: hands-off persistence (>= 3 simulated seconds) -----------------
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

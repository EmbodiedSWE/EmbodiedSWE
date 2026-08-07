"""Teleport solution for SpindleServeScene (sim_gen task
libero_kitchen_scene2_put_the_black_bowl_in_the_middle_on_the_plate_i2) — the task's
legitimacy certificate.

Teleports handle TRANSPORT ONLY; every load-bearing interaction runs through contact
dynamics:

  * a ring leaves the source dowel by an INCREMENTAL vertical raise (small pose writes with
    real physics steps between — the same free path the arm would take up the shaft, no
    solid is ever crossed), then a single free-space hop ABOVE all geometry;
  * THREADING onto the spare dowel is done by gravity through contact: the ring is released
    from just above the dowel tip with a deliberate lateral offset, so it touches the dowel,
    self-centres on it, slides the full ~110 mm shaft under contact guidance, and lands on
    the base / the previous ring — it is NEVER spawned seated;
  * the middle ring is released ~30 mm above the plate and settles onto it under gravity.

Phases (score latches must be non-decreasing along the run; SIM_GEN_SCORE is printed at
every phase boundary):
  0. reset(seed), settle, readback (stack color order, mirror side)      -> 0.000
  1. TOP ring (stack level 2): raise off source dowel, drop-thread SPARE -> 0.150
  2. MIDDLE ring (level 1, now the exposed top): raise, drop on PLATE    -> 0.450
  3. BOTTOM ring (level 0): raise, drop-thread SPARE, settle             -> 1.000
  4. persistence: >= 3 simulated seconds hands-off, success() must hold  -> SIM_GEN_SOLVE

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
    env = ENVS.get("simgen.spindle_serve")().build(num_envs=args.num_envs, device=device)
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
        thr = scene.threaded_spare()[0].tolist()
        print(f"[solve] {tag}: served={bool(scene.served()[0])} threaded_spare={thr} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])}",
              flush=True)
        if s + 1e-6 < last_score[0]:
            print(f"[solve] FATAL: score decreased {last_score[0]:.3f} -> {s:.3f}", flush=True)
            print("SIM_GEN_SOLVE: FAIL", flush=True)
            os._exit(1)
        last_score[0] = s
        print(f"SIM_GEN_SCORE {s:.3f}", flush=True)

    def ring_state(r: int) -> torch.Tensor:
        return scene.rings[r].data.root_state_w[all_ids].clone()

    def write_ring(r: int, st: torch.Tensor) -> None:
        st = st.clone()
        st[:, 7:13] = 0.0
        scene.rings[r].write_root_state_to_sim(st, all_ids)

    def flat_quat(yaw: float) -> tuple:
        return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

    def spare_tip_z() -> float:
        return float(scene.stand_spare.data.root_pos_w[0, 2]) + c.base_h / 2 + c.dowel_h

    def src_tip_z() -> float:
        return float(scene.stand_src.data.root_pos_w[0, 2]) + c.base_h / 2 + c.dowel_h

    def raise_ring(r: int, z_to: float) -> None:
        """Incremental vertical transport off a dowel: small pose writes with real steps
        between (the arm's own free path up the shaft; nothing is crossed)."""
        st = ring_state(r)
        z = float(st[0, 2])
        while z < z_to:
            z = min(z + 0.006, z_to)
            st = ring_state(r)
            st[:, 2] = z
            write_ring(r, st)
            step(2)

    def hop(r: int, x: float, y: float, z: float, yaw: float) -> None:
        """One free-space transport write, ABOVE every obstacle (both dowel tips)."""
        st = ring_state(r)
        st[:, 0] = x
        st[:, 1] = y
        st[:, 2] = z
        st[:, 3:7] = torch.tensor(flat_quat(yaw), device=device)
        st[:, 0:2] += origin[:, 0:2]
        write_ring(r, st)
        step(2)

    def settle(max_steps: int = 600, poll: int = 10) -> None:
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if bool(scene.settled()[0]):
                return

    def drop_thread(r: int, attempt: int = 0) -> bool:
        """Release the ring just above the SPARE dowel tip with a lateral offset: gravity +
        dowel contact do the threading (hole 32 mm over the 12 mm dowel, contact-guided
        descent down the full shaft)."""
        sp = scene.stand_spare.data.root_pos_w[0] - origin[0]
        off = (0.004, -0.004, 0.0)[attempt % 3]
        yaw = 0.35 * (attempt + 1)
        hop(r, float(sp[0]) + off, float(sp[1]) + off * 0.5,
            spare_tip_z() + c.ring_thick / 2 + 0.030, yaw)
        settle(max_steps=500)
        return bool(scene.threaded_spare()[0, r])

    def drop_on_plate(r: int, attempt: int = 0) -> bool:
        pp = scene.plate.data.root_pos_w[0] - origin[0]
        plate_top = float(pp[2]) + c.plate_h / 2
        off = (0.0, 0.006, -0.006)[attempt % 3]
        hop(r, float(pp[0]) + off, float(pp[1]) + off,
            plate_top + c.ring_thick / 2 + 0.030, 0.6 * attempt)
        settle(max_steps=500)
        return bool(scene.on_plate()[0, r])

    def clear_and_send(r: int, target: str) -> None:
        """Raise ring r off whatever it rests on, then thread/place with retries."""
        carry_z = max(spare_tip_z(), src_tip_z()) + c.ring_thick / 2 + 0.045
        for attempt in range(4):
            raise_ring(r, carry_z)
            ok = drop_thread(r, attempt) if target == "spare" else drop_on_plate(r, attempt)
            if ok:
                return
            print(f"[solve] {target} drop attempt {attempt} missed for ring {r}; retrying",
                  flush=True)
        print(f"[solve] FATAL: ring {r} never landed on {target}", flush=True)
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)

    # ---------------- phase 0: reset + readback ----------------------------------------------
    env.reset(seed=args.seed)
    step(60)  # settle the authored stack
    level_of = scene.level_of[0].tolist()  # ring index -> stack level (0 bottom, 1 mid, 2 top)
    mid = int(scene.mid_idx[0])
    order = sorted(range(c.n_rings), key=lambda r: level_of[r])  # bottom -> top
    names = list(c.ring_names)
    print(f"[solve] seed={args.seed} side={float(scene.side[0]):+.0f} "
          f"stack bottom->top={[names[r] for r in order]} middle={names[mid]} "
          f"src={(scene.stand_src.data.root_pos_w[0] - origin[0])[:2].tolist()} "
          f"spare={(scene.stand_spare.data.root_pos_w[0] - origin[0])[:2].tolist()} "
          f"plate={(scene.plate.data.root_pos_w[0] - origin[0])[:2].tolist()}", flush=True)
    phase_score("phase0 reset")

    top, bottom = order[2], order[0]
    assert order[1] == mid

    # ---------------- phase 1: top ring -> spare dowel (drop-threaded) ------------------------
    clear_and_send(top, "spare")
    phase_score("phase1 top->spare")

    # ---------------- phase 2: middle ring (now exposed) -> plate -----------------------------
    clear_and_send(mid, "plate")
    phase_score("phase2 middle->plate")

    # ---------------- phase 3: bottom ring -> spare dowel --------------------------------------
    clear_and_send(bottom, "spare")
    settle(max_steps=600)
    phase_score("phase3 bottom->spare")

    if not bool(scene.success()[0]):
        settle(max_steps=600)
    ok = bool(scene.success()[0])
    print(f"[solve] goal state reached: success={ok} score={float(scene.score()[0]):.3f}",
          flush=True)
    phase_score("goal")

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

"""Teleport solution for TrestleServiceScene (sim_gen task
libero_kitchen_scene2_stack_the_middle_black_bowl_on_the_back_black_bowl_i7) — the task's
legitimacy certificate.

Teleports handle TRANSPORT ONLY; every load-bearing interaction runs through contact
dynamics:

  * each bowl is teleported to a STAGING spot ~90 mm to the side of its pad (still fully
    OFF the pad, resting on the counter — no gate is anywhere near satisfied), then PUSHED
    onto the pad with a velocity-regulated horizontal force: the seat on the mark is
    reached by sliding on real counter friction, exactly the contact strategy the arm will
    use (the 100 mm bowls are wider than the 80 mm jaw span — pushing is the point of the
    task);
  * the BRIDGE is formed by gravity through contact: the board is released level ~30 mm
    above the two seated bowls' feet, falls, and must land and balance on BOTH feet on its
    own — the two-point support that defines the task is never written into place (the
    release height is far outside the 8 mm scoring z-band);
  * the cube is released ~25 mm above the deck and settles onto the board under gravity
    (again released outside its scoring band), loading the bridge for the persistence
    window.

Phases (score latches must be non-decreasing along the run; SIM_GEN_SCORE is printed at
every phase boundary):
  0. reset(seed), settle, layout readback                                  -> 0.000
  1. bowl A: stage beside its pad, force-push onto the pad                 -> 0.150
  2. bowl B: stage beside the other pad, force-push onto the pad           -> 0.300
  3. board: hover-release across both feet, lands + balances               -> 0.650
  4. cube: hover-release over the deck (over a trestle), settles           -> 1.000
  5. persistence: >= 3 simulated seconds hands-off, success() must hold    -> SIM_GEN_SOLVE

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
    env = ENVS.get("simgen.trestle_service")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    origin = env.iscene.env_origins
    zero_wrench = torch.zeros(n, 1, 3, device=device)
    z0 = c.surface_z

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    last_score = [0.0]

    def phase_score(tag: str) -> None:
        s = float(scene.score()[0])
        sm = scene.seated_matrix()[0].flatten().tolist()
        print(f"[solve] {tag}: seated={sm} pads_seated={bool(scene.pads_seated()[0])} "
              f"bridged={bool(scene.bridged()[0])} topped={bool(scene.topped()[0])} "
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

    def bowl_xy(i: int) -> torch.Tensor:
        return (scene.bowls[i].data.root_pos_w[0, :2] - origin[0, :2]).clone()

    def write_body(body, x: float, y: float, z: float, yaw: float = 0.0) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = math.cos(yaw / 2)
        st[:, 6] = math.sin(yaw / 2)
        st[:, 0:3] += origin
        body.write_root_state_to_sim(st, all_ids)

    def clear_force(i: int) -> None:
        scene.bowls[i].set_external_force_and_torque(zero_wrench, zero_wrench,
                                                     env_ids=all_ids)

    def push_bowl_to_pad(i: int, j: int) -> None:
        """Stage bowl i beside pad j (transport only, fully off the pad), then force-push
        it onto the pad through counter friction. Velocity-regulated bang-bang force with
        closed-loop heading; force is CUT near the pad centre and the bowl coasts to rest."""
        pad = scene.pad_xy[0, j].clone()
        psi = float(scene.bridge_yaw[0])
        perp = torch.tensor([-math.sin(psi), math.cos(psi)], device=device)
        side = 1.0 if float(bowl_xy(i)[1] - pad[1]) >= 0 else -1.0
        stage = pad + 0.09 * side * perp
        write_body(scene.bowls[i], float(stage[0]), float(stage[1]), z0 + 0.001)
        step(20)
        print(f"[solve] bowl{i} staged at ({float(stage[0]):+.3f},{float(stage[1]):+.3f}) "
              f"for pad{j} ({float(pad[0]):+.3f},{float(pad[1]):+.3f}), approach side "
              f"{side:+.0f}", flush=True)

        f_push, v_des = 3.5, 0.08
        best_err, last_bump = 1.0, 0
        for k in range(1500):
            err = pad - bowl_xy(i)
            e = float(err.norm())
            if e < 0.008:
                break
            d = err / (e + 1e-9)
            v_along = float((scene.bowls[i].data.root_lin_vel_w[0, :2] * d).sum())
            f = f_push if v_along < v_des else 0.0
            f_w = torch.zeros(n, 1, 3, device=device)
            f_w[:, 0, 0] = f * float(d[0])
            f_w[:, 0, 1] = f * float(d[1])
            scene.bowls[i].set_external_force_and_torque(f_w, zero_wrench,
                                                         env_ids=all_ids, is_global=True)
            env.step(no_action)
            if e < best_err - 0.004:
                best_err, last_bump = e, k
            elif k - last_bump > 300:  # stalled: friction underestimated, push harder
                f_push = min(f_push + 1.5, 8.0)
                last_bump = k
                print(f"[solve] push stalled at err={e:.3f}, raising force to "
                      f"{f_push:.1f} N", flush=True)
        clear_force(i)
        settle(max_steps=300)
        err = float((pad - bowl_xy(i)).norm())
        seated = bool(scene.seated_matrix()[0, i, j])
        print(f"[solve] bowl{i} push done: err={err * 1000:.1f} mm seated={seated}",
              flush=True)
        if not seated:
            print(f"[solve] FATAL: bowl{i} failed to seat on pad{j}", flush=True)
            print("SIM_GEN_SOLVE: FAIL", flush=True)
            os._exit(1)

    # ---------------- phase 0: reset + readback ----------------------------------------------
    env.reset(seed=args.seed)
    step(60)  # settle the authored layout
    pads = scene.pad_xy[0].tolist()
    print(f"[solve] seed={args.seed} psi={math.degrees(float(scene.bridge_yaw[0])):+.1f}deg "
          f"pads={[[round(v, 3) for v in p] for p in pads]} "
          f"bowl0={[round(float(v), 3) for v in bowl_xy(0)]} "
          f"bowl1={[round(float(v), 3) for v in bowl_xy(1)]} "
          f"swap={float(scene.swap[0]):+.0f} board_side={float(scene.board_side[0]):+.0f} "
          f"cube_side={float(scene.cube_side[0]):+.0f}", flush=True)
    phase_score("phase0 reset")

    # assign each bowl to its nearest pad (distinct)
    d = torch.zeros(2, 2)
    for i in range(2):
        for j in range(2):
            d[i, j] = float((scene.pad_xy[0, j] - bowl_xy(i)).norm())
    if float(d[0, 0] + d[1, 1]) <= float(d[0, 1] + d[1, 0]):
        assign = [(0, 0), (1, 1)]
    else:
        assign = [(0, 1), (1, 0)]

    # ---------------- phase 1+2: push each bowl onto its pad ----------------------------------
    push_bowl_to_pad(*assign[0])
    phase_score("phase1 first bowl seated")
    push_bowl_to_pad(*assign[1])
    phase_score("phase2 both bowls seated")

    # ---------------- phase 3: board hover-released across both feet --------------------------
    def bridge_geometry() -> tuple:
        a = bowl_xy(0)
        b = bowl_xy(1)
        mid = (a + b) / 2
        yaw = math.atan2(float(b[1] - a[1]), float(b[0] - a[0]))
        return mid, yaw

    bridged = False
    for attempt in range(4):
        mid, yaw = bridge_geometry()
        dx = (0.0, 0.004, -0.004, 0.0)[attempt]
        write_body(scene.board, float(mid[0]) + dx, float(mid[1]) + dx * 0.5,
                   z0 + c.bowl_h + c.board_t / 2 + 0.030, yaw)
        settle(max_steps=400)
        bridged = bool(scene.bridged()[0])
        if bridged:
            break
        print(f"[solve] board drop attempt {attempt} missed (bridged=False); retrying",
              flush=True)
    if not bridged:
        print("[solve] FATAL: board never bridged both bowls", flush=True)
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)
    phase_score("phase3 board bridged")

    # ---------------- phase 4: cube hover-released over the deck ------------------------------
    from isaaclab.utils.math import quat_apply

    topped = False
    for attempt in range(4):
        x_l = (0.08, -0.08, 0.06, -0.06)[attempt]  # over a trestle foot: no tipping moment
        bq = scene.board.data.root_quat_w[0:1]
        off = quat_apply(bq, torch.tensor([[x_l, 0.0, 0.0]], device=device))[0]
        bp = scene.board.data.root_pos_w[0] - origin[0]
        drop_z = float(bp[2]) + c.board_t / 2 + c.cube_size / 2 + 0.025
        write_body(scene.cube, float(bp[0] + off[0]), float(bp[1] + off[1]), drop_z)
        settle(max_steps=400)
        topped = bool(scene.topped()[0])
        if topped:
            break
        print(f"[solve] cube drop attempt {attempt} missed (topped=False); retrying",
              flush=True)
    if not topped:
        print("[solve] FATAL: cube never rested on the bridged deck", flush=True)
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)

    settle(max_steps=600)
    ok = bool(scene.success()[0])
    print(f"[solve] goal state reached: success={ok} score={float(scene.score()[0]):.3f}",
          flush=True)
    phase_score("phase4 cube on deck")

    # ---------------- phase 5: hands-off persistence (>= 3 simulated seconds) ------------------
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

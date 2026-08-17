"""Teleport solution for WaitersPullScene (sim_gen task
libero_kitchen_scene2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle_i117)
— the task's legitimacy certificate.

Teleports handle TRANSPORT ONLY; every load-bearing interaction runs through contact
dynamics:

  * the DRAW — the entire point of the task — is a pure force interaction: a horizontal
    external force on the slat (the body the arm would pull by its red tab) accelerates it
    hard along the pull axis under a speed cap, sliding it out from under the loaded bowl on
    real (slick, bound-material) friction. The bowl is NEVER touched: it loses support and
    falls into the pedestal recess by gravity, and the flange-on-rim seat that defines the
    task is never written into place. A deliberately slow draw is the documented failure
    mode (friction walks the bowl off the capture window); the solver demonstrates the fast
    draw;
  * after the slat is clear it is braked with a reverse force (again, no teleport while it
    interacts with anything it could disturb);
  * the STOW is transport + gravity: the slat is teleported to a hover 20 mm over the tray
    floor and released; it lands and settles between the rails on its own (the release
    height is outside the 6 mm scoring z band);
  * if a draw attempt fails, the sandwich is rebuilt by transport teleports (slat re-laid
    across the rim, bowl re-set on the slat — reconstructing the START configuration, never
    the goal) and the draw is attempted again through physics.

Phases (score latches must be non-decreasing along the run; SIM_GEN_SCORE is printed at
every phase boundary):
  0. reset(seed), settle, layout readback                                   -> 0.000
  1. draw: force-yank the slat out from under the bowl; bowl seats          -> 0.450
  2. stow: hover-release the slat into the tray                             -> 0.700
  3. persistence: >= 3 simulated seconds hands-off, success() must hold     -> 1.000
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
    env = ENVS.get("simgen.waiters_pull")().build(num_envs=args.num_envs, device=device)
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
        print(f"[solve] {tag}: seated={bool(scene.seated()[0])} "
              f"stowed={bool(scene.stowed()[0])} clear={bool(scene.slat_clear()[0])} "
              f"home={bool(scene.pedestal_home()[0])} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])}", flush=True)
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

    def local_xy(body) -> torch.Tensor:
        return (body.data.root_pos_w[0, :2] - origin[0, :2]).clone()

    def write_body(body, x: float, y: float, z: float, yaw: float = 0.0) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = math.cos(yaw / 2)
        st[:, 6] = math.sin(yaw / 2)
        st[:, 0:3] += origin
        body.write_root_state_to_sim(st, all_ids)

    def clear_force() -> None:
        scene.slat.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def apply_slat_force(fx: float, fy: float) -> None:
        f_w = torch.zeros(n, 1, 3, device=device)
        f_w[:, 0, 0] = fx
        f_w[:, 0, 1] = fy
        scene.slat.set_external_force_and_torque(f_w, zero_wrench, env_ids=all_ids,
                                                 is_global=True)

    def rebuild_sandwich(theta: float, ped: torch.Tensor) -> None:
        """Transport-only reconstruction of the START stack (never the goal): slat re-laid
        across the rim, bowl re-set riding it, all released to settle."""
        write_body(scene.slat, float(ped[0]) + c.slat_lead * math.cos(theta),
                   float(ped[1]) + c.slat_lead * math.sin(theta),
                   z0 + c.rim_top + c.slat_t / 2 + 0.002, theta)
        step(10)
        write_body(scene.bowl, float(ped[0]), float(ped[1]),
                   z0 + c.rim_top + c.slat_t + 0.003)
        settle(max_steps=300)

    def draw(theta: float, v_cap: float, f_pull: float) -> None:
        """The tablecloth pull: bang-bang force along the pull axis under a speed cap until
        the slat is well clear, then an active reverse-force brake. The bowl is untouched."""
        ux, uy = math.cos(theta), math.sin(theta)
        start = local_xy(scene.slat)
        for _ in range(240):  # 2 s hard bound
            disp = float(((local_xy(scene.slat) - start)
                          * torch.tensor([ux, uy], device=device)).sum())
            if disp > 0.35:
                break
            v = scene.slat.data.root_lin_vel_w[0, :2]
            v_along = float(v[0]) * ux + float(v[1]) * uy
            f = f_pull if v_along < v_cap else 0.0
            apply_slat_force(f * ux, f * uy)
            env.step(no_action)
        # brake: reverse force until slow (the slick slat coasts a long way on mu 0.08)
        for _ in range(200):
            v = scene.slat.data.root_lin_vel_w[0, :2]
            sp = float(v.norm())
            if sp < 0.12:
                break
            apply_slat_force(-3.0 * float(v[0]) / sp, -3.0 * float(v[1]) / sp)
            env.step(no_action)
        clear_force()
        settle(max_steps=400)

    # ---------------- phase 0: reset + readback ----------------------------------------------
    env.reset(seed=args.seed)
    step(60)  # settle the authored sandwich
    theta = float(scene.pull_yaw[0])
    ped = (scene.pedestal_xy[0]).clone()
    print(f"[solve] seed={args.seed} pull_yaw={math.degrees(theta):+.1f}deg "
          f"pedestal={[round(float(v), 3) for v in ped]} "
          f"bowl={[round(float(v), 3) for v in local_xy(scene.bowl)]} "
          f"slat={[round(float(v), 3) for v in local_xy(scene.slat)]} "
          f"tray={[round(float(v), 3) for v in scene.tray_xy[0]]} "
          f"tray_yaw={math.degrees(float(scene.tray_yaw[0])):+.1f}deg "
          f"tray_side={float(scene.tray_side[0]):+.0f}", flush=True)
    riding_dz = float(scene.bowl.data.root_pos_w[0, 2] - scene.pedestal.data.root_pos_w[0, 2])
    print(f"[solve] sandwich check: bowl rides {riding_dz * 1000:.1f} mm above pedestal "
          f"origin (seated band is {c.seat_dz * 1000:.0f}+/-{c.seat_z_tol * 1000:.0f} mm)",
          flush=True)
    phase_score("phase0 reset")

    # ---------------- phase 1: the draw --------------------------------------------------------
    seated = False
    for attempt in range(3):
        v_cap = (2.0, 2.5, 3.0)[attempt]
        f_pull = (6.0, 8.0, 10.0)[attempt]
        draw(theta, v_cap, f_pull)
        err = float((local_xy(scene.bowl) - ped.to(device)).norm())
        dz = float(scene.bowl.data.root_pos_w[0, 2] - scene.pedestal.data.root_pos_w[0, 2])
        seated = bool(scene.seated()[0])
        print(f"[solve] draw attempt {attempt}: v_cap={v_cap} f={f_pull} "
              f"bowl_err={err * 1000:.1f} mm dz={dz * 1000:.1f} mm seated={seated} "
              f"clear={bool(scene.slat_clear()[0])} home={bool(scene.pedestal_home()[0])}",
              flush=True)
        if seated:
            break
        if attempt < 2:
            print("[solve] draw failed; rebuilding the start sandwich (transport only) "
                  "and retrying", flush=True)
            rebuild_sandwich(theta, ped)
    if not seated:
        print("[solve] FATAL: bowl never seated on the pedestal", flush=True)
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)
    phase_score("phase1 bowl seated by the draw")

    # ---------------- phase 2: stow the slat ---------------------------------------------------
    from isaaclab.utils.math import quat_apply

    stowed = False
    for attempt in range(4):
        dx_l = (0.0, -0.01, 0.01, 0.0)[attempt]
        tq = scene.tray.data.root_quat_w[0:1]
        off = quat_apply(tq, torch.tensor([[dx_l, 0.0, 0.0]], device=device))[0]
        txy = scene.tray_xy[0]
        drop_z = z0 + c.tray_floor_t + c.slat_t / 2 + 0.020
        write_body(scene.slat, float(txy[0] + off[0]), float(txy[1] + off[1]), drop_z,
                   float(scene.tray_yaw[0]))
        settle(max_steps=300)
        stowed = bool(scene.stowed()[0])
        if stowed:
            break
        print(f"[solve] stow attempt {attempt} missed (stowed=False); retrying", flush=True)
    if not stowed:
        print("[solve] FATAL: slat never stowed in the tray", flush=True)
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)

    settle(max_steps=600)
    ok = bool(scene.success()[0])
    print(f"[solve] goal state reached: success={ok} score={float(scene.score()[0]):.3f}",
          flush=True)
    phase_score("phase2 slat stowed")

    # ---------------- phase 3: hands-off persistence (>= 3 simulated seconds) ------------------
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

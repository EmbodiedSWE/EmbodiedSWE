"""solve — TELEPORT solution for RockerTwinDrawersScene (…_i424).

Scene-level env (robot="null"). Teleportation handles TRANSPORT ONLY:

  PHASE 1 (open the goal drawer, via the transmission): pure contact/joint
  dynamics — a velocity-servoed world-X force on the TWIN drawer's panel (the
  scene's `drawer_drive` buffer, capped at F_MAX = 10 N, a closed-gripper push on
  the 15.5 x 11.3 cm front) drives the open twin fully IN; the hidden walking beam
  inverts the motion and slides the shut GOAL drawer OUT through its aperture.
  The drive is cut once the twin reaches its inner stop and viscous slide friction
  parks the assembly; the goal drawer now stands ~10.6 cm out, tray exposed.

  PHASE 2 (load): the ONLY teleport of the target — one pose write moves the
  pudding cube from its floor spawn to a hover ~4.9 cm ABOVE the exposed goal
  tray (free space, outside every collider). Gravity lands it on the tray floor;
  nothing else is written — the landing and seating are live contact dynamics.

  PHASE 3 (shut): joint/contact dynamics — the same velocity-servoed force pushes
  the GOAL drawer's panel back IN with the pudding riding inside (held by nothing
  but friction and the tray walls); the twin necessarily pops back out through
  the beam (by design — only the goal drawer is judged). The drive is cut once
  the goal drawer is on its inner stop, flush.

Prints the scene readouts and `SIM_GEN_SCORE <score>` at each phase boundary (the
printed sequence never decreases — asserted). After success() first holds, keeps
simulating >= 3.5 more simulated seconds with every drive buffer zero; only if
success() still holds (live state — a pudding that creeps out or a drawer that
drifts off flush would revert it) prints exactly `SIM_GEN_SOLVE: SUCCESS`. Hard
exit (os._exit) after the verdict, with a watchdog Timer as backstop — Kit
teardown hangs otherwise.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it_i424.solve --headless [--seed N]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_sec", type=float, default=900.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import os  # noqa: E402
import threading  # noqa: E402

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.libero_kitchen_scene10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it_i424 import (  # noqa: F401,E501
        scene as scene_mod,
    )
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()

# Fingertip-scale drives (the honesty argument, shared with smoke.py):
# F_MAX 10 N on a 15.5 x 11.3 cm drawer front — an easy closed-gripper knuckle
# push (it moves ~1.3 kg of coupled sliding mass against 6 N*s/m of viscous slide
# friction and the pin-fork rub). V_CAP 0.10 m/s keeps carriage accelerations tiny.
F_MAX = 10.0  # N force cap on a drawer panel
KP = 60.0  # N*s/m drawer velocity-servo gain
V_CAP = 0.10  # m/s max slide speed
K_APPROACH = 2.0  # v_des ramps down near the stop -> soft landing
CUT_EXT = 0.004  # cut the drive once the driven drawer is this close to its stop
HOVER = 0.049  # teleport hover height above the cargo's tray rest pose


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.rocker_twin_drawers")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        e = scene.ext()[0].tolist()
        loc = scene.cargo_local()[0].tolist()
        print(f"[solve] {tag:12s} ext=(L {e[0]:+.4f}, R {e[1]:+.4f}) "
              f"target_ext={float(scene.target_ext()[0]):+.4f} "
              f"cargo_local=({loc[0]:+.3f},{loc[1]:+.3f},{loc[2]:+.3f}) "
              f"in_tray={bool(scene.in_tray()[0])} score={sc():.3f} "
              f"success={bool(scene.success()[0])}", flush=True)

    last_score = -1.0

    def phase_score(tag: str) -> None:
        nonlocal last_score
        s = sc()
        assert s >= last_score - 1e-6, f"score decreased at {tag}: {last_score} -> {s}"
        last_score = s
        print(f"SIM_GEN_SCORE {s:.3f}", flush=True)

    def verdict(ok: bool) -> None:
        print("SIM_GEN_SOLVE: SUCCESS" if ok else "SIM_GEN_SOLVE: FAIL", flush=True)
        code = 0 if ok else 1
        threading.Timer(10.0, lambda: os._exit(code)).start()
        try:
            env.close()
            app.close()
        except Exception:  # noqa: BLE001
            pass
        os._exit(code)

    def settle_until(pred, max_steps: int = 480, poll: int = 10) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def servo_drawer(idx: int, budget: int = 3000) -> bool:
        """Velocity-servo drawer `idx` (0 = left, 1 = right) INWARD (+x) until its
        extension reaches CUT_EXT (its inner hard stop), then cut the drive.
        Returns True if the stop was reached."""
        body = scene.drawer_l if idx == 0 else scene.drawer_r
        for i in range(budget):
            e = float(scene.ext()[0, idx])
            if e <= CUT_EXT:
                scene.drawer_drive[0, idx] = 0.0
                return True
            v = float(body.data.root_lin_vel_w[0, 0])  # +x closes
            v_des = min(V_CAP, K_APPROACH * e)
            f = KP * (v_des - v)
            scene.drawer_drive[0, idx] = max(-F_MAX, min(F_MAX, f))
            step(1)
            if i and i % 300 == 0:
                report("pushing")
        scene.drawer_drive[0, idx] = 0.0
        return False

    # ================= reset + settle ==========================================================
    env.reset(seed=args.seed)
    step(60)
    side = int(scene.target_side[0])
    goal_idx, twin_idx = side, 1 - side
    y_goal = c.drawer_y if side == 0 else -c.drawer_y
    print(f"[solve] seed={args.seed} goal={'LEFT' if side == 0 else 'RIGHT'} "
          f"cargo=({float(scene.cargo.data.root_pos_w[0, 0] - scene.env_origins[0, 0]):.3f},"
          f"{float(scene.cargo.data.root_pos_w[0, 1] - scene.env_origins[0, 1]):.3f})",
          flush=True)
    report("reset")
    assert float(scene.target_ext()[0]) < c.closed_tol, "goal drawer must start shut"
    assert float(scene.twin_ext()[0]) > c.travel - 0.010, "twin drawer must start out"
    assert float(scene.drawer_drive.abs().max()) == 0.0
    assert float(scene.cargo_drive.abs().max()) == 0.0
    phase_score("reset")  # ~0.000

    # ================= PHASE 1: open the goal drawer via the transmission ======================
    if not servo_drawer(twin_idx):
        print("[solve] PHASE 1 FAILED: twin push timed out short of its stop", flush=True)
        verdict(False)
    if not settle_until(lambda: float(scene.target_ext()[0]) >= c.open_thresh
                        and bool(scene.settled()[0]), max_steps=360):
        report("open-fail")
        print("[solve] PHASE 1 FAILED: goal drawer did not emerge/settle", flush=True)
        verdict(False)
    report("opened")
    phase_score("phase1")  # 0.150 (open latch)

    # ================= PHASE 2: load the pudding (transport + gravity landing) =================
    # Teleport = transport only: hover ~4.9 cm above the exposed tray floor, zero
    # velocity, in free air outside every collider (the goal tray stands proud of
    # the chest face; the hover is below the lintel with 8 mm clearance). Gravity
    # does the landing; the seating is live contact dynamics.
    st = torch.zeros(1, 13, device=device)
    st[0, 0] = c.open_x + c.drop_dx
    st[0, 1] = y_goal
    st[0, 2] = c.floor_z + c.tray_rest_local_z + HOVER
    st[0, 3] = 1.0  # yaw 0: faces square to the tray
    st[0, 0:3] += scene.env_origins[0]
    scene.cargo.write_root_state_to_sim(st, torch.tensor([0], device=device))
    step(2)
    scene.mark_cargo_ref()  # wrench-frame reference (probes only; no push planned)
    if not settle_until(lambda: bool((scene.in_tray()
                                      & (scene.cargo.data.root_lin_vel_w.norm(dim=-1)
                                         < c.settle_cargo))[0]), max_steps=480):
        report("load-fail")
        print("[solve] PHASE 2 FAILED: pudding did not settle in the goal tray", flush=True)
        verdict(False)
    report("loaded")
    phase_score("phase2")  # 0.500 (open + load latches)

    # ================= PHASE 3: shut the goal drawer (payload rides on friction) ===============
    if not servo_drawer(goal_idx):
        print("[solve] PHASE 3 FAILED: goal push timed out short of flush", flush=True)
        verdict(False)
    if not bool(scene.in_tray()[0]):
        report("cargo-lost")
        print("[solve] PHASE 3 FAILED: pudding left the tray during the carry", flush=True)
        verdict(False)
    print(f"[solve] drive cut at target_ext={float(scene.target_ext()[0]):.4f} m "
          f"(twin popped back to {float(scene.twin_ext()[0]):.4f} m — expected)", flush=True)
    if not settle_until(lambda: bool(scene.success()[0]), max_steps=600):
        report("shut-fail")
        print("[solve] PHASE 3 FAILED: success() not reached after the cut", flush=True)
        verdict(False)
    report("shut")
    phase_score("phase3")  # 1.000

    # ================= PHASE 4: persistence (>= 3.5 simulated seconds, hands off) ==============
    assert float(scene.drawer_drive.abs().max()) == 0.0, "drives must be zero for persistence"
    assert float(scene.cargo_drive.abs().max()) == 0.0
    persist_steps = int(round(3.5 / env.dt))  # 420 physics steps at 1/120 s
    step(persist_steps)
    report("final")
    phase_score("final")
    still_ok = bool(scene.success()[0]) and abs(sc() - 1.0) < 1e-3
    print(f"[solve] persistence: {persist_steps} steps ({persist_steps * env.dt:.2f} s) "
          f"hands-off, success={still_ok}", flush=True)
    verdict(still_ok)


if __name__ == "__main__":
    main()

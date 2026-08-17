"""Teleport solution for LeaningChainScene (sim_gen task `screw_nail_i309`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY. Every load-bearing interaction — every leaning
seat in the chain — goes through contact dynamics:

  For each tile, BACK-TO-FRONT (blue -> green -> red, the mechanically forced order):
  1. TRANSPORT: teleport the tile across free space to a STANDING pose in the lane at
     its build spot (foot distance from its support), wide face toward the anvil.
  2. TOPPLE (dynamics): a rate-regulated torque about the lane's cross axis tips the
     standing tile past its balance point, then releases it; gravity swings it down
     and its head lands ON its support (the anvil for blue, the previous tile
     otherwise) purely through contact. The settled leaning rest — the thing the
     rubric judges — is made by gravity + friction + the support's contact force,
     never by spawning a tile mid-lean.
  3. If the settled pose is out of band (fell flat = spacing too wide, too steep =
     too close), the tile is re-transported standing and re-toppled with adjusted
     spacing; a disturbed support is rebuilt the same way (deepest broken stage
     first).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.screw_nail_i309.solve --headless [--seed N]
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

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

TILE_IDX = {"red": 0, "green": 1, "blue": 2}


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.leaning_chain")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def wrench(body, f3: torch.Tensor, t3: torch.Tensor) -> None:
        """World wrench expressed in the body's CURRENT link frame (`is_global=True`
        silently drops torques on this stack — transform manually, the house
        convention). Re-set every step while pushing."""
        from isaaclab.utils.math import quat_apply_inverse

        q = body.data.root_link_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f3.view(1, 3).expand(n, 3)).unsqueeze(1),
            quat_apply_inverse(q, t3.view(1, 3).expand(n, 3)).unsqueeze(1),
            env_ids=all_ids)

    # ---- lane frame -----------------------------------------------------------------------
    def lane_pose() -> tuple[torch.Tensor, float]:
        lp = (scene.lane.data.root_pos_w - scene.env_origins)[0]
        q = scene.lane.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return lp, yaw

    def lane_to_world(loc) -> torch.Tensor:
        lp, yaw = lane_pose()
        cy, sy = math.cos(yaw), math.sin(yaw)
        return torch.tensor([float(lp[0]) + cy * loc[0] - sy * loc[1],
                             float(lp[1]) + sy * loc[0] + cy * loc[1],
                             loc[2]], device=device)

    def geo():
        return scene.tile_geometry()

    def tile_axis_z_world(name: str) -> float:
        from isaaclab.utils.math import quat_apply

        q = scene.tiles[name].data.root_quat_w
        ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
        return abs(float(quat_apply(q, ez)[0, 2]))

    def report(tag: str) -> None:
        g = geo()
        seat = scene.seated_now(g)[0]
        parts = []
        for nm, k in TILE_IDX.items():
            th = math.degrees(math.asin(max(-1.0, min(1.0, float(g["axis"][0, k, 2])))))
            parts.append(f"{nm}: foot=({float(g['foot'][0, k, 0]):+.3f},"
                         f"{float(g['foot'][0, k, 1]):+.3f}) tilt={th:5.1f} "
                         f"headz={float(g['head'][0, k, 2]):.3f} seat={bool(seat[k])}")
        print(f"[solve] {tag:12s} | " + " | ".join(parts)
              + f" | success={bool(scene.success()[0])}"
              f" score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def settle_all(max_steps: int = 300) -> None:
        for _ in range(max_steps // 30):
            step(30)
            g = geo()
            if bool(g["still"][0].all()):
                break

    # ---- build primitives -----------------------------------------------------------------
    def place_standing(name: str, foot_x: float) -> None:
        """TRANSPORT ONLY: teleport the free tile to a standing pose in the lane,
        wide face toward the anvil (body y = thickness along lane +x)."""
        _lp, yaw = lane_pose()
        psi = yaw - math.pi / 2
        pos = lane_to_world((foot_x, 0.0, c.tile_len / 2 + 0.002))
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos + scene.env_origins
        st[:, 3] = math.cos(psi / 2)
        st[:, 6] = math.sin(psi / 2)
        scene.tiles[name].write_root_state_to_sim(st, all_ids)
        step(10)

    def topple(name: str) -> None:
        """DYNAMICS: rate-regulated torque about the lane cross axis tips the
        standing tile past its balance point toward the anvil; released, it falls
        and seats on its support through contact alone."""
        body = scene.tiles[name]
        _lp, yaw = lane_pose()
        axis_w = torch.tensor([-math.sin(yaw), math.cos(yaw), 0.0], device=device)
        t3 = torch.zeros(3, device=device)
        for _ in range(360):
            if tile_axis_z_world(name) < math.cos(math.radians(30.0)):
                break  # past the balance point and falling — release
            w = float(torch.dot(body.data.root_ang_vel_w[0], axis_w))
            tau = max(min(0.010 * (1.5 - w), 0.02), -0.02)
            wrench(body, zero3, tau * axis_w)
            step(1)
        wrench(body, zero3, zero3)
        # let it fall onto its support and ring down
        for _ in range(10):
            step(30)
            g = geo()
            if bool(g["still"][0, TILE_IDX[name]]):
                break

    def stage_foot_target(stage: str, spacing: dict) -> float:
        """Lane-frame foot x for this stage: from the anvil for blue, from the
        SUPPORT tile's actual settled foot otherwise."""
        if stage == "blue":
            return c.anvil_face - spacing["blue"]
        g = geo()
        sup = {"green": "blue", "red": "green"}[stage]
        return float(g["foot"][0, TILE_IDX[sup], 0]) - spacing[stage]

    def build_stage(stage: str, spacing: dict) -> bool:
        """Place standing + topple; adapt spacing from the settled tilt readback."""
        for attempt in range(4):
            place_standing(stage, stage_foot_target(stage, spacing))
            topple(stage)
            g = geo()
            k = TILE_IDX[stage]
            sz = float(g["axis"][0, k, 2])  # sin(tilt)
            seat = scene.seated_now(g)[0]
            report(f"{stage} try{attempt}")
            if bool(seat[k]):
                return True
            if sz < math.sin(math.radians(c.tilt_min_deg)):
                spacing[stage] = max(spacing[stage] - 0.007, 0.020)  # fell flat: closer
            elif sz > math.sin(math.radians(c.tilt_max_deg)):
                spacing[stage] += 0.007  # too steep: farther
            else:
                spacing[stage] -= 0.004  # marginal (support/height): nudge closer
        return False

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    lp, yaw = lane_pose()
    print(f"[solve] layout readback (seed {args.seed}): lane=({float(lp[0]):+.3f},"
          f"{float(lp[1]):+.3f}) yaw={math.degrees(yaw):+.1f} deg", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, "null credit at reset — rubric leak"

    # ---------------- phases 1-3: build the chain back-to-front ----------------------------
    spacing = {"blue": 0.042, "green": 0.052, "red": 0.048}
    stages = ("blue", "green", "red")
    scores = [s0]
    done: set[str] = set()
    for round_i in range(10):
        # deepest broken stage first (a later topple can disturb its support)
        seat = scene.seated_now()[0]
        pending = [st_ for st_ in stages if not bool(seat[TILE_IDX[st_]])]
        if not pending:
            break
        stage = pending[0]
        ok = build_stage(stage, spacing)
        if ok and stage not in done:
            done.add(stage)
            step(40)  # hold still so the streak-gated stage latch engages
            s = print_score(f"P{len(done)} {stage} toppled onto its support (contact dynamics)")
            assert s >= scores[-1] - 1e-6, "latched score decreased"
            scores.append(s)
    assert bool(scene.seated_now()[0].all()), "chain incomplete after build rounds"

    # ---------------- phase 4: hands-off settle to success ---------------------------------
    settle_all()
    report("settled")
    s4 = print_score("P4 all settled")
    assert s4 >= scores[-1] - 1e-6, "score decreased across settling"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after chain build + settle)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 5: persistence (>= 3 simulated seconds, hands-off) ------------
    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                g = geo()
                seat = scene.seated_now(g)[0]
                print(f"[solve] persist flicker @step {i}: seat="
                      f"{[bool(seat[TILE_IDX[nm]]) for nm in TILE_IDX]} "
                      f"still={[bool(g['still'][0, k]) for k in range(3)]}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s5 = print_score("P5 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s5 >= s4 - 1e-6
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)

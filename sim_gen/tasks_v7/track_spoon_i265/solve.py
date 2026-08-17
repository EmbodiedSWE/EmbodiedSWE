"""Teleport solution for SpoonKnifeEdgeScene (sim_gen task `track_spoon_i265`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY, always ending in FREE SPACE or a non-contact
hover; every load-bearing interaction goes through CONTACT DYNAMICS:
  P0 — SETTLE: reset scatters the stand (xy + yaw), the spoon (xy + free yaw) and the
  cube on the floor; everything settles under gravity. Nothing is touched. Score ~0.
  P1 — LOAD: the cube is carried to a hover 25 mm ABOVE the open bowl pocket of the
  floor-lying spoon (yaw-aligned so it drops cleanly through the 40 mm opening) and
  RELEASED. Gravity seats it: the rims retain it and it comes to rest on the pocket
  floor — a real containment produced by contact, not by writing a seated state.
  P2 — PERCH: the cube's true seated pose is READ BACK in the spoon frame and the
  loaded balance point recomputed from the authored masses; the assembly (spoon +
  cube, exact relative pose preserved) is carried to a hover 3 mm above the fin crest
  with the balance point deliberately offset 4 mm from the crest centre — INSIDE the
  10 mm support window, proving the window is physically real — and RELEASED. Gravity
  and the crest's contact patch decide: only a correctly composed CoM placement
  settles level with both ends in free air. The teleport bypasses nothing: the judged
  quantity is the settled equilibrium, which only physics produces.
  P3 — persistence: hands off for >= 3.5 simulated seconds after success() first
  turns True; `SIM_GEN_SOLVE: SUCCESS` only if it still holds.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched).

Run (forge): python -u -m simgen_tasks.track_spoon_i265.solve --headless [--seed N]
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.spoon_knife_edge")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def deg(x: float) -> float:
        return math.degrees(x)

    def report(tag: str) -> None:
        d = scene.cube_in_spoon()[0]
        print(f"[solve] {tag:12s} | v_sp={float(scene.spoon.data.root_lin_vel_w.norm(dim=-1)[0]):.4f} "
              f"w_sp={float(scene.spoon.data.root_ang_vel_w.norm(dim=-1)[0]):.4f} "
              f"v_cu={float(scene.cube.data.root_lin_vel_w.norm(dim=-1)[0]):.4f} "
              f"v_st={float(scene.stand.data.root_lin_vel_w.norm(dim=-1)[0]):.4f}", flush=True)
        print(f"[solve] {tag:12s} | tilt={deg(float(scene.spoon_tilt()[0])):+6.2f}deg "
              f"pocket={bool(scene.in_pocket()[0])} "
              f"cube_rel=({float(d[0]):+.3f},{float(d[1]):+.3f},{float(d[2]):+.3f}) "
              f"perched={bool(scene.perched()[0])} level={bool(scene.level()[0])} "
              f"roll={bool(scene.roll_ok()[0])} air={bool(scene.ends_air()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def write_state(body, pos_w: torch.Tensor, quat_w: torch.Tensor) -> None:
        """Transport-only teleport to a WORLD pose with zero velocities."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat_w
        body.write_root_state_to_sim(st, all_ids)

    # ---------------- phase 0: reset, settle, readback -------------------------------------
    step(240)  # everything comes to rest on the floor (~2 s)
    print(f"[solve] layout readback (seed {args.seed}): "
          f"stand=({float(scene._stand_xy[0, 0]):+.3f},{float(scene._stand_xy[0, 1]):+.3f}) "
          f"yaw={deg(float(scene._stand_yaw[0])):+.1f}deg "
          f"bal_x={c.bal_x * 1000:+.1f}mm com_x={c.com_x * 1000:+.1f}mm "
          f"m_spoon={c.m_spoon * 1000:.2f}g m_cube={c.m_cube * 1000:.2f}g", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    if bool(scene.success()[0]) or s0 > 0.03:
        print("SIM_GEN_SOLVE: FAIL (reset state already scores)", flush=True)
        os._exit(1)

    # ---------------- phase 1: LOAD — drop the cube into the bowl pocket -------------------
    sp_pos = scene.spoon.data.root_pos_w.clone()
    sp_q = scene.spoon.data.root_quat_w.clone()
    hover_local = torch.tensor(
        [c.bowl_cx, 0.0, c.floor_t + c.rim_h + 0.025 + c.cube_s / 2],
        device=device).expand(n, 3)
    write_state(scene.cube, sp_pos + quat_apply(sp_q, hover_local), sp_q)
    step(90)  # free fall through the pocket opening + impact
    for _ in range(10):  # wait for the seated cube to go quiet
        if float(scene.cube.data.root_lin_vel_w.norm(dim=-1)[0]) < 0.02:
            break
        step(30)
    report("load")
    s1 = print_score("P1 cube seated in the pocket")
    assert s1 >= s0 - 1e-6, "score decreased across the load phase"
    if not bool(scene.in_pocket()[0]):
        print("SIM_GEN_SOLVE: FAIL (cube did not seat in the pocket)", flush=True)
        os._exit(1)
    if s1 < 0.20 - 1e-4:
        print("SIM_GEN_SOLVE: FAIL (load credit did not latch)", flush=True)
        os._exit(1)

    # ---------------- phase 2: PERCH — compose the CoM, place, release --------------------
    b_rel = scene.cube_in_spoon()[0]  # the cube's TRUE seated pose (spoon frame)
    bal = (c.m_spoon * c.com_x + c.m_cube * float(b_rel[0])) / (c.m_spoon + c.m_cube)
    offset = 0.004  # deliberately off the crest centre — still inside the window
    print(f"[solve] composed balance point: bal={bal * 1000:+.2f}mm "
          f"(cube_rel_x={float(b_rel[0]) * 1000:+.2f}mm), placing it "
          f"{offset * 1000:+.1f}mm from the crest centre", flush=True)

    st_pos = scene.stand.data.root_pos_w.clone()
    st_q = scene.stand.data.root_quat_w.clone()
    target_local = torch.tensor([-bal + offset, 0.0, c.crest_top + 0.003],
                                device=device).expand(n, 3)
    spoon_pos_t = st_pos + quat_apply(st_q, target_local)
    spoon_q_t = st_q  # spoon axis along the stand x — across the fin
    q_rel = quat_mul(quat_inv(scene.spoon.data.root_quat_w), scene.cube.data.root_quat_w)
    cube_pos_t = spoon_pos_t + quat_apply(spoon_q_t, b_rel.unsqueeze(0).expand(n, 3))
    cube_q_t = quat_mul(spoon_q_t, q_rel)
    # one whole-assembly transport: spoon AND cube written together, then released
    write_state(scene.spoon, spoon_pos_t, spoon_q_t)
    write_state(scene.cube, cube_pos_t, cube_q_t)
    for _ in range(30):  # drop 3 mm, rock on the crest, settle level (~2-4 s)
        step(30)
        if bool(scene.success()[0]):
            break
    report("perch")
    s2 = print_score("P2 loaded spoon balanced on the crest")
    assert s2 >= s1 - 1e-6, "score decreased across the perch phase"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the perch settled)", flush=True)
        os._exit(1)

    # ---------------- phase 3: persistence (>= 3.5 simulated seconds, hands off) ----------
    hold = True
    for i in range(12):  # 12 x 36 steps = 432 substeps = 3.6 s at 120 Hz
        step(36)
        ok_i = bool(scene.success()[0])
        if not ok_i:
            print(f"[solve] persist blip @block {i}: "
                  f"pocket={bool(scene.in_pocket()[0])} perched={bool(scene.perched()[0])} "
                  f"level={bool(scene.level()[0])} roll={bool(scene.roll_ok()[0])} "
                  f"air={bool(scene.ends_air()[0])} settled={bool(scene.settled()[0])} "
                  f"still={bool(scene._still_pos[0])} "
                  f"v_sp={float(scene.spoon.data.root_lin_vel_w.norm(dim=-1)[0]):.4f} "
                  f"w_sp={float(scene.spoon.data.root_ang_vel_w.norm(dim=-1)[0]):.4f} "
                  f"v_cu={float(scene.cube.data.root_lin_vel_w.norm(dim=-1)[0]):.4f}",
                  flush=True)
        hold = hold and ok_i
    report("persist")
    s3 = print_score("P3 persistence 3.6 s")
    ok = hold and bool(scene.success()[0]) and s3 >= s2 - 1e-6
    if ok:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)

    code = 0 if ok else 1
    # Hard exit: Kit teardown hangs — watchdog then die.
    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()

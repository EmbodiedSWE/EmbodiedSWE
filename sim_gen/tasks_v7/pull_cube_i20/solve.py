"""Teleport solution for BeamScaleScene (sim_gen task `pull_cube_i20`) — the task's
legitimacy certificate.

Teleportation handles TRANSPORT ONLY: each red cube is teleported from the floor to a
release pose ABOVE the raised pan (clear of the walls) with zero velocity — exactly
what a pick-and-carry delivers — and then RELEASED. Everything the rubric reads
happens through contact dynamics after the release: the cube free-falls into the pan,
the slick pan floor slides it to the downhill (inner) wall, its weight loads the beam
through pan contact, and when the accumulated moment beats the counterweight's the
beam rotates about its knife-edge pivot and falls onto the opposite stop. The beam,
the stand, and the counterweight are never touched after reset — no wrench is ever
applied to anything; the ONLY intervention is placing cubes above the target pan.

The demonstrated policy is the honest one from describe(): the counterweight's mass
is unknown, so add ONE cube, hold off, watch the beam; if it stays down, add the
next; stop as soon as the beam swings over and rests on the cube side (1, 2 or 3
cubes depending on the sampled counterweight).

Prints `SIM_GEN_SCORE <score>` at each settled phase boundary (non-decreasing:
banked cubes stay banked, the tilt only ever moves toward the cube side), then holds
HANDS-OFF for >= 3 simulated seconds after success() first turns True and prints
`SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.pull_cube_i20.solve --headless [--seed N]
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

_DY_SLOTS = (0.0, 0.056, -0.056)  # per-cube y offset in the pan (avoid stacking)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.beam_scale")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    sin_t = math.sin(math.radians(c.tip_deg))

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def tilt() -> float:
        """Signed tilt progress: cw_side * u (>0 means tipped toward the cube side)."""
        return float((scene.cw_side * scene.beam_tilt())[0])

    def teleport_cube(i: int) -> None:
        """TRANSPORT ONLY: place cube `i` at rest above the raised pan's center (beam
        frame x = -side * pan center, z clear of the walls), aligned with the beam,
        zero velocity — then hands off; gravity and pan contact do the rest."""
        from isaaclab.utils.math import quat_apply

        local = torch.tensor([float(-scene.cw_side[0]) * 0.20, _DY_SLOTS[i], 0.095],
                             device=device).expand(n, 3)
        q = scene.beam.data.root_quat_w
        pos = scene.beam.data.root_pos_w + quat_apply(q, local)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3:7] = q
        scene.cubes[i].write_root_state_to_sim(st, all_ids)

    def wait_settled(max_steps: int = 720) -> None:
        """Hands-off until the beam and all cargo are settled (or the budget runs out)."""
        for _ in range(max_steps // 30):
            step(30)
            ok = bool(scene.beam_settled()[0]) and bool(scene._chosen(scene.settled)[0])
            for b in scene.cubes:
                ok = ok and bool(scene.settled(b)[0])
            if ok:
                break

    def report(tag: str) -> None:
        u = float(scene.beam_tilt()[0])
        lv = float(scene.beam.data.root_lin_vel_w[0].norm())
        av = float(scene.beam.data.root_ang_vel_w[0].norm())
        print(f"[solve] {tag:14s} | side={float(scene.cw_side[0]):+.0f} "
              f"cw_idx={int(scene.cw_idx[0])} u={u:+.3f} s*u={tilt():+.3f} "
              f"(need >= {sin_t:.3f}) | cubes_in={int(scene.n_cubes_target()[0])} "
              f"cw_in={bool(scene.cw_in_place()[0])} beam(lin={lv:.3f},ang={av:.3f}) | "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    # The beam is reset LEVEL; hands-off it falls onto the counterweight side.
    wait_settled(480)
    side = float(scene.cw_side[0])
    print(f"[solve] layout readback (seed {args.seed}): cw_idx={int(scene.cw_idx[0])} "
          f"(mass {c.cw_masses[int(scene.cw_idx[0])] * 1000:.0f} g) side={side:+.0f} "
          f"-> target pan on side {-side:+.0f}", flush=True)
    for i, b in enumerate(scene.cubes):
        p = (b.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] layout readback (seed {args.seed}): cube_{i} at "
              f"({float(p[0]):+.3f},{float(p[1]):+.3f})", flush=True)
    report("reset")
    assert tilt() < -0.15, "beam must rest tilted toward the counterweight after reset"
    s_prev = print_score("P0 reset+settle (beam rests on the counterweight side)")

    # ---------------- phases 1..3: add cubes one at a time, watch the beam ------------------
    dropped = 0
    for i in range(3):
        for attempt in range(3):
            teleport_cube(i)
            wait_settled(600)
            if bool(scene.in_tray(scene.cubes[i], -scene.cw_side)[0]):
                break
            print(f"[solve] cube_{i} missed the pan (attempt {attempt}) — re-dropping",
                  flush=True)
        assert bool(scene.in_tray(scene.cubes[i], -scene.cw_side)[0]), \
            f"cube_{i} failed to land in the target pan"
        dropped += 1
        report(f"P{dropped}-dropped")
        s_now = print_score(f"P{dropped} cube_{i} banked in the target pan "
                            f"(drop + slide by contact)")
        assert s_now >= s_prev - 1e-6, "score decreased across a drop"
        s_prev = s_now
        if tilt() >= sin_t:
            print(f"[solve] beam tipped after {dropped} cube(s) — stopping", flush=True)
            break
        print(f"[solve] beam still on the counterweight side after {dropped} cube(s) "
              f"(s*u={tilt():+.3f}) — adding another", flush=True)

    # ---------------- final settle to success ----------------------------------------------
    for _ in range(16):  # up to 4 s extra hands-off settling
        if bool(scene.success()[0]):
            break
        step(30)
    report("settled")
    s3 = print_score("P-final all settled")
    assert s3 >= s_prev - 1e-6, "score decreased across settling"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after drops+settle)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- persistence (>= 3 simulated seconds, no intervention) ----------------
    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                lv = float(scene.beam.data.root_lin_vel_w[0].norm())
                av = float(scene.beam.data.root_ang_vel_w[0].norm())
                print(f"[solve] persist flicker @step {i}: s*u={tilt():+.3f} "
                      f"beam(lin={lv:.4f},ang={av:.4f}) "
                      f"cubes_in={int(scene.n_cubes_target()[0])} "
                      f"cw_in={bool(scene.cw_in_place()[0])}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s4 = print_score("P-persist persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s4 >= s3 - 1e-6
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

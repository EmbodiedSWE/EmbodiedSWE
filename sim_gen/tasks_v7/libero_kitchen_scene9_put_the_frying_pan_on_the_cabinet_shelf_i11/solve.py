"""Teleport solution for MatchboxDrawerScene (sim_gen task
`libero_kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf_i11`) — the task's
legitimacy certificate.

Teleportation handles TRANSPORT ONLY, in one write ending in FREE SPACE:
  P2 — the RED cube is carried from its counter spawn to a hover pose ~100 mm above the
  OPEN tray's cavity (open air: the exposed cavity sits forward of the sleeve roof), and
  released. The loading itself happens through CONTACT DYNAMICS: the cube falls into the
  cavity and settles on the tray floor under gravity and real contact — exactly the
  release an arm performs when placing an object into an open drawer.
Everything else is DRIVEN, not bypassed:
  P1 — a force governor applies a bounded horizontal force to the tray along the
  cabinet's pull-out axis (read from the scene: the cabinet yaw is randomized) — the
  force a hand pulling the black knob applies — sliding the drawer open against rail
  friction to ~14.5 cm of opening on the porch.
  P3 — the same governor, reversed, pushes the tray back shut; the cube RIDES INSIDE on
  friction, passes under the sleeve roof, and ends ENCLOSED. A brief gentle press seats
  the tray against the back wall, the force is cut, and everything settles on real
  contact. The cube is never teleported into the cavity, the sleeve, or the shut drawer;
  the load latch, the closing latch, and success() are all earned by physics.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf_i11.solve --headless [--seed N]
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


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.matchbox_drawer")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def pull_dir() -> torch.Tensor:
        """World-frame unit vector of the cabinet's pull-out (+x) axis."""
        yaw = float(scene._yaw_of(scene.cabinet)[0])
        return torch.tensor([math.cos(yaw), math.sin(yaw), 0.0], device=device)

    def clear_wrench() -> None:
        scene.tray.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def apply_axial(f: float) -> None:
        fw = (pull_dir() * f).view(1, 1, 3).expand(n, 1, 3).contiguous()
        scene.tray.set_external_force_and_torque(
            fw, zero_wrench, env_ids=all_ids, is_global=True)

    def report(tag: str) -> None:
        o = float(scene.opening()[0])
        pl = scene.cab_local(scene.payload.data.root_pos_w)[0]
        dl = scene.cab_local(scene.decoy.data.root_pos_w)[0]
        print(f"[solve] {tag:12s} | opening={o * 1000:+.1f}mm "
              f"seated={bool(scene.tray_seated()[0])} shut={bool(scene.shut()[0])} "
              f"pay_cab=({float(pl[0]):+.3f},{float(pl[1]):+.3f},{float(pl[2]):.3f}) "
              f"in_cav={bool(scene.in_cavity(scene.payload)[0])} "
              f"encl={bool(scene.enclosed(scene.payload)[0])} "
              f"decoy_cav={bool(scene.in_cavity(scene.decoy)[0])} "
              f"open_l={float(scene.open_latch[0]):.2f} load_l={float(scene.load_latch[0]):.2f} "
              f"shut_l={float(scene.shut_latch[0]):.2f} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)
        _ = dl  # decoy cab-local kept for debugging parity

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def drive_to(target: float, v_tgt: float, cap: float, budget: int,
                 label: str) -> bool:
        """Force-govern the tray along the pull axis until `opening` crosses `target`
        (direction from sign of v_tgt). Returns True if the target was reached."""
        sgn = 1.0 if v_tgt > 0 else -1.0
        for i in range(budget):
            o = float(scene.opening()[0])
            if sgn > 0 and o >= target:
                return True
            if sgn < 0 and o <= target:
                return True
            v = float(torch.dot(scene.tray.data.root_lin_vel_w[0], pull_dir()))
            f = max(-cap, min(cap, 25.0 * (v_tgt - v)))
            apply_axial(f)
            env.step(no_action)
            if i == budget // 2:
                oo = float(scene.opening()[0])
                still_short = (sgn > 0 and oo < target) or (sgn < 0 and oo > target)
                if still_short:
                    v_tgt *= 1.5
                    cap = min(cap * 1.5, 8.0)
                    print(f"[solve] {label}: slow at step {i} (opening "
                          f"{oo * 1000:+.1f}mm) — raising drive", flush=True)
        return False

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    hp = (scene.cabinet.data.root_pos_w - scene.env_origins)[0]
    hyaw = float(scene._yaw_of(scene.cabinet)[0])
    pl0 = scene.cab_local(scene.payload.data.root_pos_w)[0]
    dl0 = scene.cab_local(scene.decoy.data.root_pos_w)[0]
    print(f"[solve] layout readback (seed {args.seed}): cabinet=({float(hp[0]):+.3f},"
          f"{float(hp[1]):+.3f}) yaw={math.degrees(hyaw):+.1f}deg "
          f"opening={float(scene.opening()[0]) * 1000:+.1f}mm "
          f"payload_cab=({float(pl0[0]):+.3f},{float(pl0[1]):+.3f}) "
          f"decoy_cab=({float(dl0[0]):+.3f},{float(dl0[1]):+.3f})", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: DRIVE the drawer open ---------------------------------------
    # Force governor: f = clamp(25 * (0.10 - v_along), +/-4 N) along the pull axis —
    # the force of a hand pulling the 18 mm knob bar. Rail friction is ~0.7 N.
    opened = drive_to(c.open_target, v_tgt=0.10, cap=4.0, budget=2400, label="open")
    clear_wrench()
    step(60)
    report("opened")
    s1 = print_score("P1 drawer driven open")
    assert s1 >= s0 - 1e-6, "score decreased across the opening drive"
    if not opened or float(scene.opening()[0]) < c.open_gate:
        print("SIM_GEN_SOLVE: FAIL (drawer did not open)", flush=True)
        os._exit(1)

    # ---------------- phase 2: cube TRANSPORT (hover over the open cavity) + gravity drop --
    # Hover: ~100 mm above the tray floor, centred on the exposed cavity (which sits
    # forward of the sleeve roof at this opening), then free fall + settle — the release
    # an arm performs when placing the cube into the open drawer.
    from isaaclab.utils.math import quat_apply

    def drop_at(local_x: float) -> None:
        tq = scene.tray.data.root_quat_w[0:1]
        tp = scene.tray.data.root_pos_w[0:1]
        off = torch.tensor([[local_x, 0.0, 0.100]], device=device)
        w = tp + quat_apply(tq, off)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = w
        st[:, 3:7] = tq
        scene.payload.write_root_state_to_sim(st, all_ids)
        step(120)  # fall + settle on the tray floor (1 s)

    drop_at(0.0)
    if not bool(scene.load_latch[0] > 0.5):  # rare: perched on a wall — retry off-centre
        print("[solve] load retry at tray-local x=-0.02", flush=True)
        drop_at(-0.02)
    report("loaded")
    s2 = print_score("P2 cube transport + gravity load into the open cavity")
    assert s2 >= s1 - 1e-6, "score decreased across the cube transport"
    if not bool(scene.load_latch[0] > 0.5):
        print("SIM_GEN_SOLVE: FAIL (cube did not load into the open cavity)", flush=True)
        os._exit(1)

    # ---------------- phase 3: DRIVE the drawer shut ---------------------------------------
    # Reversed governor (v_tgt = -0.08 m/s, cap 4 N): the cube rides the closing tray on
    # friction, passes under the roof, ends enclosed. Then a brief gentle press seats
    # the tray against the back wall, the force is cut, and everything settles.
    drive_to(0.004, v_tgt=-0.08, cap=4.0, budget=2400, label="close")
    for _ in range(60):  # gentle 1.2 N seating press (0.5 s)
        apply_axial(-1.2)
        env.step(no_action)
    clear_wrench()
    step(90)
    report("shut")
    s3 = print_score("P3 drawer driven shut (cube riding inside)")
    assert s3 >= s2 - 1e-6, "score decreased across the closing drive"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the closing drive)", flush=True)
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, no intervention) -----
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
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
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 — die loudly, don't wait for the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL (exception: {exc!r})", flush=True)
        os._exit(2)

"""Teleport solution for CarouselAirlockScene (sim_gen task
`libero_kitchen_scene10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_i9`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY, in one write ending in FREE SPACE:
  P1 — the BLACK ball is carried from its floor spawn to a hover pose 150 mm up, in the
  open air directly ABOVE the roof's LOADING WINDOW (azimuth chosen away from wherever
  the orange vane is parked, read from the scene). The loading itself happens through
  CONTACT DYNAMICS: the ball falls through the open window and comes to rest ON the
  rotor disc inside the covered well under gravity and real contact — exactly the
  release an arm performs when dropping an object through an aperture.
  P2 — the machine is DRIVEN, not bypassed: a torque governor applies a bounded
  external torque about the vertical axis to the rotor (the moment a hand pushing the
  yellow spokes tangentially applies), spinning it counter-clockwise at ~1.2 rad/s.
  The angled ORANGE plow vane sweeps the ball around the covered annulus under real
  contact, expels it through the side DISCHARGE GAP, and the ball rolls down the hooded
  chute into the GREEN basin by gravity. The torque is then cut and everything settles
  on real contact. The ball is never teleported into the well, the corridor, or the
  basin; every scored latch is earned by physics.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_i9.solve --headless [--seed N]
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


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.carousel_airlock")().build(num_envs=args.num_envs, device=device)
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

    def housing_pose() -> tuple[torch.Tensor, float]:
        hp = (scene.housing.data.root_pos_w - scene.env_origins)[0]
        return hp, float(scene._yaw_of(scene.housing)[0])

    def ball_local() -> torch.Tensor:
        return scene._housing_local(scene.ball.data.root_pos_w)[0]

    def report(tag: str) -> None:
        bl = ball_local()
        r = math.hypot(float(bl[0]), float(bl[1]))
        azd = math.degrees(math.atan2(float(bl[1]), float(bl[0])))
        wz = float(scene.rotor.data.root_ang_vel_w[0, 2])
        print(f"[solve] {tag:12s} | ball_local=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) r={r:.3f} az={azd:+.1f}deg "
              f"ann={bool(scene.in_annulus(scene.ball)[0])} "
              f"basin={bool(scene.in_basin(scene.ball)[0])} "
              f"load={float(scene.load_latch[0]):.2f} sweep={float(scene.sweep_latch[0]):.2f} "
              f"transit={float(scene.transit_latch[0]):.2f} rotor_w={wz:+.2f} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_wrench() -> None:
        scene.rotor.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    hp, hyaw = housing_pose()
    ryaw = float(scene._yaw_of(scene.rotor)[0])
    b0 = (scene.ball.data.root_pos_w - scene.env_origins)[0]
    d0v = (scene.decoy.data.root_pos_w - scene.env_origins)[0]
    vane_rel = _wrap(ryaw - hyaw)  # vane azimuth in the housing frame
    print(f"[solve] layout readback (seed {args.seed}): housing=({float(hp[0]):+.3f},"
          f"{float(hp[1]):+.3f}) yaw={math.degrees(hyaw):+.1f}deg "
          f"vane_rel_az={math.degrees(vane_rel):+.1f}deg "
          f"ball_spawn=({float(b0[0]):+.3f},{float(b0[1]):+.3f}) "
          f"decoy_spawn=({float(d0v[0]):+.3f},{float(d0v[1]):+.3f}) "
          f"d0={float(scene.d0[0]):.3f}", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: ball TRANSPORT (hover over the window) + gravity load -------
    # Hover: 150 mm up in the open air above the loading window, at an azimuth inside
    # the window chosen AWAY from wherever the vane is parked (read from the scene) so
    # the ball lands on the bare disc; the loading itself is the free fall through the
    # window onto the disc under real contact.
    def drop_at(phi: float) -> None:
        lx = c.load_r * math.cos(phi)
        ly = c.load_r * math.sin(phi)
        w_xy = scene._housing_world_xy(lx, ly)[0]
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = float(w_xy[0]) - float(scene.env_origins[0, 0])
        st[:, 1] = float(w_xy[1]) - float(scene.env_origins[0, 1])
        st[:, 2] = 0.150
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        scene.ball.write_root_state_to_sim(st, all_ids)
        step(120)  # fall + settle on the disc (1 s)

    candidates = [0.0, math.radians(-28.0), math.radians(28.0)]
    phi = next(p for p in candidates if abs(_wrap(vane_rel - p)) > math.radians(25.0))
    drop_at(phi)
    if not bool(scene.load_latch[0] > 0.5):  # rare: ball perched on the vane — retry
        vane_rel = _wrap(float(scene._yaw_of(scene.rotor)[0]) - housing_pose()[1])
        phi = next(p for p in candidates if abs(_wrap(vane_rel - p)) > math.radians(25.0))
        print(f"[solve] load retry at phi={math.degrees(phi):+.1f}deg", flush=True)
        drop_at(phi)
    report("ball-loaded")
    s1 = print_score("P1 ball transport + gravity load through the window")
    assert s1 >= s0 - 1e-6, "score decreased across the ball transport"
    if not bool(scene.load_latch[0] > 0.5):
        print("SIM_GEN_SOLVE: FAIL (ball did not load into the annulus)", flush=True)
        os._exit(1)

    # ---------------- phase 2: DRIVE the mechanism (bounded torque on the rotor) -----------
    # Torque governor: tau_z = clamp(1.6 * (w_tgt - w_z), +/-0.35 N m) about world z —
    # the moment of a hand pushing the yellow spokes tangentially (~2.3 N at r 0.15).
    # The vane sweeps the ball around the covered annulus and out the discharge gap
    # under real contact; gravity takes it down the chute into the basin.
    w_tgt, tau_cap = 1.2, 0.35
    delivered = False
    for i in range(3000):
        bl = ball_local()
        r_loc = math.hypot(float(bl[0]), float(bl[1]))
        if bool(scene.transit_latch[0] > 0.5) and r_loc > 0.16:
            delivered = True
            break
        if bool(scene.in_basin(scene.ball)[0]):
            delivered = True
            break
        wz = float(scene.rotor.data.root_ang_vel_w[0, 2])
        tau = max(-tau_cap, min(tau_cap, 1.6 * (w_tgt - wz)))
        tq = torch.tensor([0.0, 0.0, tau], device=device).view(1, 1, 3).expand(n, 1, 3)
        scene.rotor.set_external_force_and_torque(
            zero_wrench, tq.contiguous(), env_ids=all_ids, is_global=True)
        env.step(no_action)
        if i == 1800 and not bool(scene.transit_latch[0] > 0.5):
            w_tgt = 1.8
            print("[solve] sweep slow — raising target spin to 1.8 rad/s", flush=True)
    clear_wrench()
    print(f"[solve] sweep loop done (delivered={delivered}, steps={i + 1})", flush=True)
    # gravity delivery: down the chute, across the sill, settle in the basin
    for _ in range(24):
        step(30)
        if bool(scene.success()[0]):
            break
    step(60)
    report("delivered")
    s2 = print_score("P2 mechanism drive + gravity delivery")
    assert s2 >= s1 - 1e-6, "score decreased across the sweep"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after sweep + delivery)", flush=True)
        os._exit(1)

    # ---------------- phase 3: persistence (>= 3.3 simulated seconds, no intervention) -----
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s3 = print_score("P3 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s3 >= s2 - 1e-6
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
    except BaseException:  # noqa: BLE001 - die fast, don't wait for the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(2)

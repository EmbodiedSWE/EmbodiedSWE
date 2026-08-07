"""Teleport solution for GearTrainDialScene (sim_gen task
`libero_kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf_i32`) — the task's
legitimacy certificate.

Teleportation handles TRANSPORT ONLY, in one write ending in FREE SPACE:
  P1 — the blue TOOTHED idler gear is carried from its floor spawn to a hover pose in
  the open air directly ABOVE the empty centre axle (bore over the axle tip, ~10 mm of
  clearance). The installation itself happens through CONTACT DYNAMICS: the gear falls,
  its bore captures the axle, and it drops onto the plate under gravity. If it lands
  tooth-on-tooth on a neighbour, a small alternating z-torque (the wiggle a hand gives
  a part to seat it) walks the teeth into the gaps until `idler_seated()` reads True.
  The SMOOTH decoy is never touched.
  P2 — the machine is DRIVEN, not bypassed: a torque governor applies a bounded pure-z
  torque to the ORANGE driver (the moment of a hand pushing the green crank pin in
  circles, ~3.3 N at r 46 mm), and tooth contact carries the rotation driver -> idler ->
  caged output until the RED marker bar points at the MAGENTA stripe. The crank torque
  is then cut, the driver braked, and everything settles on real contact. The output
  wheel is NEVER written to after reset — every degree it turns is earned through the
  mesh, which is exactly what the scene's gated error account certifies.

All applied wrenches are pure world-z (torque about z, optional -z press), so they are
invariant to the pod-dependent external-force frame-drag quirk.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf_i32.solve --headless [--seed N]
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
    env = ENVS.get("simgen.gear_train_dial")().build(num_envs=args.num_envs, device=device)
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

    def z_wrench(body, fz: float, tz: float) -> None:
        f = torch.tensor([0.0, 0.0, fz], device=device).view(1, 1, 3).expand(n, 1, 3)
        t = torch.tensor([0.0, 0.0, tz], device=device).view(1, 1, 3).expand(n, 1, 3)
        body.set_external_force_and_torque(
            f.contiguous(), t.contiguous(), env_ids=all_ids, is_global=True)

    def clear_wrench(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def rel_delta() -> float:
        """Signed output rotation still needed to put the marker on the stripe (rad)."""
        rel = _wrap(float(scene._yaw_of(scene.output)[0]) - float(scene._yaw_of(scene.frame)[0]))
        return _wrap(c.stripe_az - rel)

    def report(tag: str) -> None:
        ip = (scene.idler.data.root_pos_w - scene.env_origins)[0]
        wd = float(scene.driver.data.root_ang_vel_w[0, 2])
        wo = float(scene.output.data.root_ang_vel_w[0, 2])
        print(f"[solve] {tag:12s} | idler=({float(ip[0]):+.3f},{float(ip[1]):+.3f},"
              f"{float(ip[2]):.3f}) seated={bool(scene.idler_seated()[0])} "
              f"drv_seated={bool(scene.driver_seated()[0])} "
              f"err={math.degrees(float(scene.marker_err()[0])):+.1f}deg "
              f"acc={math.degrees(float(scene.err_acc[0])):+.1f}deg "
              f"delta={math.degrees(rel_delta()):+.1f}deg w_drv={wd:+.2f} w_out={wo:+.2f} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    fp = (scene.frame.data.root_pos_w - scene.env_origins)[0]
    fyaw = float(scene._yaw_of(scene.frame)[0])
    i0 = (scene.idler.data.root_pos_w - scene.env_origins)[0]
    d0v = (scene.decoy.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): frame=({float(fp[0]):+.3f},"
          f"{float(fp[1]):+.3f}) yaw={math.degrees(fyaw):+.1f}deg "
          f"idler_spawn=({float(i0[0]):+.3f},{float(i0[1]):+.3f}) "
          f"decoy_spawn=({float(d0v[0]):+.3f},{float(d0v[1]):+.3f}) "
          f"err0={math.degrees(float(scene.err0[0])):+.1f}deg "
          f"delta={math.degrees(rel_delta()):+.1f}deg d0={float(scene.d0[0]):.3f}",
          flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: idler TRANSPORT (hover over the axle) + gravity seat --------
    # Hover: bore centred over the empty CENTRE axle, hub bottom ~10 mm above the axle
    # tip, in the open air; the installation is the free fall (bore captures the axle)
    # plus, if it lands tooth-on-tooth, a hand-scale wiggle (alternating z-torque, light
    # z press) until the teeth walk into the gaps and the hub sits flush on the plate.
    hover_z = c.plate_top + c.axle_h + 0.010

    def drop_idler(yaw: float) -> None:
        axle_xy = scene._frame_world_xy(0.0, 0.0)[0]
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = float(axle_xy[0])
        st[:, 1] = float(axle_xy[1])
        st[:, 2] = hover_z + float(scene.env_origins[0, 2])
        st[:, 3] = math.cos(yaw / 2)
        st[:, 6] = math.sin(yaw / 2)
        scene.idler.write_root_state_to_sim(st, all_ids)
        step(90)  # fall + first settle (0.75 s)

    def wiggle_to_seat() -> bool:
        # Speed-GOVERNED jiggle (a constant torque on a free hub spins it up and flings
        # it off the axle): alternate the target wiggle speed +/-2 rad/s with a light
        # press, and stop the burst the instant the teeth drop into the gaps.
        for j in range(10):
            if bool(scene.idler_seated()[0]):
                break
            w_tgt = 2.0 if j % 2 == 0 else -2.0
            for _ in range(45):
                wz = float(scene.idler.data.root_ang_vel_w[0, 2])
                tau = max(-0.06, min(0.06, 0.8 * (w_tgt - wz)))
                z_wrench(scene.idler, -1.5, tau)
                env.step(no_action)
                if bool(scene.idler_seated()[0]):
                    break
            clear_wrench(scene.idler)
            step(25)
        clear_wrench(scene.idler)
        step(40)
        ip = (scene.idler.data.root_pos_w - scene.env_origins)[0]
        axle = scene._frame_world_xy(0.0, 0.0)[0]
        d_ax = math.hypot(float(scene.idler.data.root_pos_w[0, 0]) - float(axle[0]),
                          float(scene.idler.data.root_pos_w[0, 1]) - float(axle[1]))
        print(f"[solve] seat readback: idler_z={float(ip[2]):.4f} d_axle={d_ax:.4f} "
              f"seated={bool(scene.idler_seated()[0])}", flush=True)
        return bool(scene.idler_seated()[0])

    seated = False
    for attempt, yaw in enumerate((0.0, math.radians(22.5), math.radians(11.25))):
        drop_idler(fyaw + yaw)
        seated = wiggle_to_seat()
        if seated:
            break
        print(f"[solve] seat attempt {attempt + 1} failed — re-dropping", flush=True)
    report("idler-seated")
    s1 = print_score("P1 idler transport + gravity/wiggle seating")
    assert s1 >= s0 - 1e-6, "score decreased across the idler transport"
    if not seated:
        print("SIM_GEN_SOLVE: FAIL (idler did not seat on the centre axle)", flush=True)
        os._exit(1)

    # ---------------- phase 2: DRIVE the mechanism (bounded torque on the driver) ----------
    # Torque governor on the ORANGE driver: tau_z = clamp(2.0 * (w_tgt - w_z), +/-0.20
    # N m) — the moment of a hand pushing the green crank pin tangentially. Two tooth
    # meshes (driver->idler->output) turn the caged output the SAME direction as the
    # driver at 1:1; we crank at a steady ~1.2 rad/s and cut + brake at the mark.
    # (A slower near-mark speed was tried and stalls: below ~0.5 rad/s the driver
    # chatters inside the ~31 deg backlash without net transmission. Full speed with a
    # per-step stop check lands within ~1 deg of the stop threshold after braking.)
    tau_cap = 0.20

    def crank_until(stop_rad: float, max_steps: int) -> None:
        last_delta = abs(rel_delta())
        last_check = 0
        for i in range(max_steps):
            delta = rel_delta()
            if abs(delta) < stop_rad:
                break
            direction = 1.0 if delta > 0 else -1.0
            speed = 1.2
            wz = float(scene.driver.data.root_ang_vel_w[0, 2])
            tau = max(-tau_cap, min(tau_cap, 2.0 * (direction * speed - wz)))
            z_wrench(scene.driver, 0.0, tau)
            env.step(no_action)
            if i - last_check >= 600:  # stall watch: no progress in 5 s -> brief reversal
                if last_delta - abs(delta) < 0.05:
                    print(f"[solve] crank stalled at delta="
                          f"{math.degrees(delta):+.1f}deg — reversing briefly", flush=True)
                    for _ in range(50):
                        wz = float(scene.driver.data.root_ang_vel_w[0, 2])
                        tau = max(-tau_cap, min(tau_cap, 2.0 * (-direction * 0.6 - wz)))
                        z_wrench(scene.driver, 0.0, tau)
                        env.step(no_action)
                last_delta, last_check = abs(delta), i
        # brake the driver (output stops through the mesh + friction)
        for _ in range(90):
            wz = float(scene.driver.data.root_ang_vel_w[0, 2])
            z_wrench(scene.driver, 0.0, max(-tau_cap, min(tau_cap, -2.0 * wz)))
            env.step(no_action)
        clear_wrench(scene.driver)
        step(60)

    for attempt in range(4):
        crank_until(stop_rad=0.10, max_steps=6000)
        err = float(scene.marker_err()[0])
        print(f"[solve] crank pass {attempt + 1}: err={math.degrees(err):+.1f}deg "
              f"acc={math.degrees(float(scene.err_acc[0])):+.1f}deg", flush=True)
        if err < c.tol * 0.7 and float(scene.err_acc[0]) < c.tol * 0.9:
            break
    # settle until everything is at rest
    for _ in range(20):
        step(30)
        if bool(scene.success()[0]):
            break
    report("aligned")
    s2 = print_score("P2 mechanism drive through the mesh")
    assert s2 >= s1 - 1e-6, "score decreased across the crank"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after crank + settle)", flush=True)
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

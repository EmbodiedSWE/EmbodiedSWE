"""Contact-dynamics solution for BallastHatchScene (sim_gen task `stack_cube_i183`)
— the task's legitimacy certificate.

Teleports are used for TRANSPORT ONLY; every load-bearing interaction is earned through
contact dynamics:
  - Each steel ballast cube is teleported to hover 3 cm ABOVE its hopper CELL's CURRENT
    opening (gate-pose readback), attitude-aligned with the tilted cell, and then
    RELEASED — it falls in under gravity, and its WEIGHT (contact forces on the cage
    floor) is what actuates the bell-crank: the gate angle is never written, no wrench
    ever touches the gate. Four cubes swing the flap to the ~62 deg limit stop.
  - The parcel is teleported only BETWEEN free ground poses in the open approach lane
    (spawn band -> aligned staging spot). The doorway transit itself is a pure
    nonprehensile PUSH: a bounded external force on the parcel body (velocity servo,
    |F| <= 0.7 N — a light fingertip push, and BELOW the m*g ~ 0.78 N quasi-static
    tipping bound so the parcel slides instead of toppling), applied in the parcel's
    BODY frame recomputed each step (immune to the is_global wrench rotation-drag
    quirk). The force is CUT once the parcel is inside; it settles by friction alone.

The push direction and lateral centering are read from the SCENE'S OWN vault frame
(payload_vault_local readback), so the same code works at any randomized apparatus yaw.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.4 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.stack_cube_i183.solve --headless [--seed N]
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
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ballast_hatch")().build(num_envs=args.num_envs, device=device)
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

    def clear_wrench() -> None:
        scene.payload.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def report(tag: str) -> None:
        p = scene.payload_vault_local()[0]
        ang = math.degrees(float(scene.gate_open_angle()[0]))
        print(f"[solve] {tag:14s} | parcel_vault=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):+.3f}) gate={ang:+.1f}deg "
              f"open={bool(scene.lat_open[0])} door={bool(scene.lat_door[0])} "
              f"cham={bool(scene.in_chamber()[0])} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def fail(msg: str) -> None:
        report("FAIL-state")
        print(f"SIM_GEN_SOLVE: FAIL ({msg})", flush=True)
        os._exit(1)

    # ---------------- phase 0: reset, settle, baseline --------------------------------------
    step(60)
    vp = (scene.vault.data.root_pos_w - scene.env_origins)[0]
    qv0 = scene.vault.data.root_quat_w[0]
    yaw = math.atan2(2 * (float(qv0[0]) * float(qv0[3])), 1 - 2 * float(qv0[3]) ** 2)
    print(f"[solve] layout readback (seed {args.seed}): vault=({float(vp[0]):+.3f},"
          f"{float(vp[1]):+.3f}) yaw={math.degrees(yaw):+.1f}deg "
          f"gate={math.degrees(float(scene.gate_open_angle()[0])):+.2f}deg", flush=True)
    report("reset")
    ang0 = float(scene.gate_open_angle()[0])
    if abs(math.degrees(ang0)) > 4.0:
        fail("gate not closed at reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: load the hopper (gravity drops, weight actuation) ------------
    # Teleport each cube to hover above its hopper CELL's CURRENT opening (gate-pose
    # readback — the cells move as the gate opens), attitude-aligned with the tilted
    # cell so it falls straight in, then RELEASE: gravity + contact do the work. Far
    # cell pair first (bigger lever arm -> more early opening moment).
    for i, body in enumerate(scene.ballast):
        mw = scene.cell_mouth_world(i)[0]
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = mw[0]
        st[:, 1] = mw[1]
        st[:, 2] = mw[2] + 0.032
        st[:, 3:7] = scene.cell_drop_quat()  # cell-aligned attitude (transport only)
        body.write_root_state_to_sim(st, all_ids)
        step(300)  # 2.5 s: fall in, cage swings, gate finds its new equilibrium
        ang = math.degrees(float(scene.gate_open_angle()[0]))
        print(f"[solve] ballast {i + 1}/4 dropped -> gate={ang:+.1f}deg "
              f"score={float(scene.score()[0]):.3f}", flush=True)
    step(120)
    ang = math.degrees(float(scene.gate_open_angle()[0]))
    if ang < 52.0:
        fail(f"hopper loaded but gate only opened to {ang:.1f} deg")
    if not bool(scene.lat_open[0]):
        fail("gate-open latch not earned")
    report("hopper-loaded")
    s1 = print_score("P1 hopper loaded, hatch held open by ballast weight")
    assert s1 >= s0 - 1e-6, "score decreased while loading the hopper"

    # ---------------- phase 2: stage the parcel (free-lane transport teleport) --------------
    qv = scene.vault.data.root_quat_w
    stage_local = torch.tensor([0.19, 0.0, c.payload_size / 2 + 0.0005], device=device)
    sw = scene.vault.data.root_pos_w + quat_apply(qv, stage_local.unsqueeze(0).expand(n, 3))
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = sw
    st[:, 3:7] = qv  # face the doorway squarely
    scene.payload.write_root_state_to_sim(st, all_ids)
    step(60)
    report("staged")
    s2 = print_score("P2 parcel staged in the approach lane")
    assert s2 >= s1 - 1e-6, "score decreased while staging the parcel"

    # ---------------- phase 3: nonprehensile push through the doorway -----------------------
    # Velocity servo along the vault's -x axis with lateral centering; bounded force,
    # applied in the parcel's BODY frame recomputed each step.
    # k_v*v_des = 0.60 N at rest > the ~0.35 N ground breakaway friction (a smaller gain
    # stalls below static friction forever); K*dt/m = 5/(120*0.08) ~ 0.52 < 1 (stable)
    v_des, k_v, cap_along = 0.12, 5.0, 0.70
    k_y, k_dy, cap_lat = 6.0, 2.0, 0.40
    entered = False
    for i in range(2000):
        qv = scene.vault.data.root_quat_w
        p = scene.payload_vault_local()[0]
        if float(p[0]) < 0.02:
            entered = True
            break
        v_local = quat_apply_inverse(qv, scene.payload.data.root_lin_vel_w)[0]
        f_along = max(-cap_along, min(cap_along, k_v * (v_des - float(-v_local[0]))))
        f_lat = max(-cap_lat, min(cap_lat, k_y * (0.0 - float(p[1])) - k_dy * float(v_local[1])))
        f_vault = torch.tensor([-f_along, f_lat, 0.0], device=device).unsqueeze(0)
        f_world = quat_apply(qv, f_vault)
        f_body = quat_apply_inverse(scene.payload.data.root_quat_w, f_world)
        scene.payload.set_external_force_and_torque(
            f_body.view(n, 1, 3).contiguous(), zero_wrench, env_ids=all_ids)
        env.step(no_action)
        if float(p[2]) > 0.09:
            fail("parcel left the ground during the push")
    clear_wrench()
    if not entered:
        fail("push did not bring the parcel into the chamber")
    print(f"[solve] parcel crossed into the chamber in {i + 1} push steps; force cut, "
          "friction settles it", flush=True)
    step(240)
    report("delivered")
    if not bool(scene.lat_door[0]):
        fail("doorway latch not earned during the transit")
    if not bool(scene.success()[0]):
        fail("no success after the parcel settled inside")
    s3 = print_score("P3 doorway transit + settle")
    assert s3 >= s2 - 1e-6, "score decreased across the doorway transit"

    # ---------------- phase 4: persistence (>= 3.4 simulated seconds, hands-off) ------------
    hold = True
    for _ in range(10):  # 10 x 41 steps = 410 substeps = 3.42 s at 120 Hz
        step(41)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.4 s")
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
    except BaseException:  # noqa: BLE001 - die fast, don't wait for the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(2)

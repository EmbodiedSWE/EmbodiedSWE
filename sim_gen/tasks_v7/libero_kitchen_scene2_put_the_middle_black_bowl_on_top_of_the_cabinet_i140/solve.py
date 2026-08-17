"""solve — teleport-transport + force-push solution for RampHutchScene
(libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet_i140).

Scene-level env (robot="null"). Teleports are TRANSPORT ONLY; every credit-earning
interaction is applied external force + contact dynamics:

  PHASE T  TRANSPORT — identify the MIDDLE bowl (scene.target_idx) and teleport it
           from the floor row onto the LOWER ramp (fixture x = -0.55, far outside
           every credit band), tilt-matched to the 20.4 deg slope, zero velocity.
           Settle; assert the score is STILL 0.000 — the teleport earned nothing.
  PHASE P1 PUSH (lower->upper ramp) — pulsed ~2.5 N force on the bowl along the
           fixture's uphill tangent (+x, +z), speed-capped at 0.08 m/s, until the
           bowl center passes fixture x > -0.30 (inside the hi-ramp band). Release;
           static friction (mu 0.55 > tan 20.4 deg) parks it on the slope. -> 0.150
  PHASE P2 PUSH (doorway + drop-in) — same pulsed push until the bowl crosses the
           crest, passes THROUGH the doorway cut (passage latch), drops the 3 cm
           step into the roofed tray, and reaches fixture x > -0.12 (seated band).
           Release, settle: success() = seated & doored & no decoy & settled. -> 1.000
  PERSIST  >= 3.5 simulated seconds fully hands-off; success() must still hold
           before `SIM_GEN_SOLVE: SUCCESS` is printed.

The pod's external-force API may interpret wrenches in the body's current frame
(pod/version dependent), so drive() probes force encoding at runtime from measured
progress and toggles between raw-world and quat_apply_inverse(q_now, f).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (latched credit — the printed
sequence never decreases) and exactly `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds after the persistence window. Hard exit (os._exit) with a watchdog Timer.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet_i140.solve --headless
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

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet_i140 import (  # noqa: F401,E501
        scene as scene_mod,
    )
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ramp_hutch")().build(num_envs=1, device=device)
    scene = env.scene
    no_action = torch.empty(0, device=device)
    zero3 = torch.zeros(1, 1, 3, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def clear_forces() -> None:
        for b in scene.bowls:
            b.set_external_force_and_torque(zero3, zero3)

    def push(body, f_world: torch.Tensor) -> None:
        clear_forces()
        body.set_external_force_and_torque(f_world.view(1, 1, 3), zero3)

    def settle_until(pred, max_steps: int = 720, poll: int = 10) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def report(tag: str) -> None:
        loc = scene.target_fix()[0]
        print(f"[solve] {tag:12s} target_idx={int(scene.target_idx[0])} "
              f"tgt_fix=({loc[0]:+.3f},{loc[1]:+.3f},{loc[2]:+.3f}) "
              f"hi={bool(scene._hi_ever[0])} doored={bool(scene._doored_ever[0])} "
              f"seated={bool(scene._seated_ever[0])} "
              f"decoy_in={bool(scene.decoy_in_tray()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"score={sc():.3f} success={bool(scene.success()[0])}", flush=True)

    last_score = -1.0

    def phase_score(tag: str) -> None:
        nonlocal last_score
        s = sc()
        assert s >= last_score - 1e-6, f"score decreased at {tag}: {last_score} -> {s}"
        last_score = s
        print(f"SIM_GEN_SCORE {s:.3f}", flush=True)

    def verdict(ok: bool) -> None:
        if ok:
            print("SIM_GEN_SOLVE: SUCCESS", flush=True)
        else:
            print("SIM_GEN_SOLVE: FAIL", flush=True)
        code = 0 if ok else 1
        threading.Timer(10.0, lambda: os._exit(code)).start()
        try:
            env.close()
            app.close()
        except Exception:  # noqa: BLE001
            pass
        os._exit(code)

    # ================= reset + settle ==========================================================
    env.reset(seed=args.seed)
    step(60)
    report("reset")
    assert not bool(scene._hi_ever[0] | scene._doored_ever[0] | scene._seated_ever[0]), \
        "no latch may fire at reset"
    phase_score("reset")  # ~0.000

    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul  # noqa: E402

    c = scene.cfg
    q_fix = scene.fixture.data.root_quat_w[0:1]
    p_fix = scene.fixture.data.root_pos_w[0:1]
    theta = math.atan(c.ramp_slope)  # 20.39 deg

    def fix_to_world(loc) -> torch.Tensor:
        loc_t = torch.tensor([loc], device=device, dtype=torch.float32)
        return (p_fix + quat_apply(q_fix, loc_t))[0]

    # uphill tangent of the ramp, world frame (+x, +z in the fixture frame)
    up_tan = quat_apply(q_fix, torch.tensor(
        [[math.cos(theta), 0.0, math.sin(theta)]], device=device))[0]

    tgt = scene.bowls[int(scene.target_idx[0])]

    def x_fix() -> float:
        return float(scene.target_fix()[0, 0])

    def v_along(body, axis: torch.Tensor) -> float:
        return float((body.data.root_lin_vel_w[0] * axis).sum())

    def drive(body, axis: torch.Tensor, fmag: float, vmax: float, done,
              max_steps: int, tag: str) -> bool:
        """Pulsed push along world `axis` until `done()`. Probes force encoding from
        measured progress: if headway stalls, toggle between raw-world and
        quat_apply_inverse(q_now, f) (pod may rotate wrenches by the body's frame)."""
        mode = 0
        last_probe = float((body.data.root_pos_w[0] * axis).sum())
        for i in range(max_steps):
            if done():
                clear_forces()
                return True
            f = fmag if v_along(body, axis) < vmax else 0.0
            fw = f * axis
            if mode == 1:
                fw = quat_apply_inverse(body.data.root_quat_w[0:1], fw.view(1, 3))[0]
            push(body, fw)
            step(1)
            if (i + 1) % 40 == 0:
                cur = float((body.data.root_pos_w[0] * axis).sum())
                if cur - last_probe < 0.001:
                    mode ^= 1
                    print(f"[solve] {tag}: progress stalled, force mode -> {mode}",
                          flush=True)
                last_probe = cur
        clear_forces()
        return done()

    # ================= PHASE T: teleport the MIDDLE bowl onto the lower ramp ===================
    # Transport only: fixture x=-0.55 is far below the hi band (x > -0.33), outside the
    # doorway and the tray; the teleport must not move the score. Tilt-matched to the
    # slope, resting height = surface z + (h/2)/cos(theta) + 4 mm drop clearance.
    x_park = -0.55
    z_surf = (x_park - c.ramp_foot_x) * c.ramp_slope
    z_park = z_surf + (c.bowl_h / 2) / math.cos(theta) + 0.004
    # bowl pitched -theta about the fixture's y axis (matches the ramp top face)
    q_pitch = torch.tensor([[math.cos(theta / 2), 0.0, -math.sin(theta / 2), 0.0]],
                           device=device)
    st = torch.zeros(1, 13, device=device)
    st[0, 0:3] = fix_to_world((x_park, 0.0, z_park))
    st[0, 3:7] = quat_mul(q_fix, q_pitch)
    tgt.write_root_state_to_sim(st, torch.tensor([0], device=device))
    ok_t = settle_until(lambda: bool(scene.settled()[0]), max_steps=360)
    report("transport")
    if not ok_t:
        print("[solve] PHASE T FAILED: bowl did not settle on the lower ramp", flush=True)
        verdict(False)
    assert sc() < 1e-6, f"teleport transport earned credit: score={sc():.3f}"
    assert -0.62 < x_fix() < -0.40, f"bowl not parked on the lower ramp: x={x_fix():.3f}"
    phase_score("transport")  # still 0.000

    # ================= PHASE P1: push up the ramp into the hi band =============================
    # mg(sin+mu*cos) ~ 1.27 N resists; 2.5 N pulsed (only below 0.08 m/s) creeps the
    # bowl up the curb channel. Stop past x > -0.30; static friction holds it there.
    drive(tgt, up_tan, 2.5, 0.08, lambda: x_fix() > -0.30, 3000, "climb")
    ok1 = settle_until(lambda: bool(scene.settled()[0]), max_steps=360)
    report("hi_ramp")
    if not (ok1 and bool(scene._hi_ever[0])):
        print("[solve] PHASE P1 FAILED: bowl did not hold the upper ramp", flush=True)
        verdict(False)
    assert not bool(scene._doored_ever[0]), "door latch must not fire below the crest"
    phase_score("phase1")  # 0.150

    # ================= PHASE P2: through the doorway, over the sill, into the tray =============
    # Same pulsed push: through the door cut (passage latch), over the crest/sill,
    # 3 cm drop onto the tray floor, creep to x > -0.12 (inside the seated band).
    drive(tgt, up_tan, 2.5, 0.08, lambda: x_fix() > -0.12, 3600, "doorway")
    ok2 = settle_until(lambda: bool(scene.success()[0]))
    report("seated")
    if not ok2:
        print("[solve] PHASE P2 FAILED: bowl did not seat inside the tray", flush=True)
        verdict(False)
    phase_score("phase2")  # 1.000

    # ================= persistence (>= 3.5 simulated seconds, hands off) =======================
    persist_steps = int(round(3.5 / env.dt))  # 420 physics steps at 1/120 s
    step(persist_steps)
    report("final")
    phase_score("final")
    still_ok = bool(scene.success()[0]) and sc() == 1.0
    print(f"[solve] persistence: {persist_steps} steps ({persist_steps * env.dt:.2f} s) "
          f"hands-off, success={still_ok}", flush=True)
    verdict(still_ok)


if __name__ == "__main__":
    main()

"""solve — force-driven solution for SwitchbackRampScene
(libero_kitchen_scene4_put_the_black_bowl_on_top_of_the_cabinet_i409).

Scene-level env (robot="null"). This solution uses NO teleports at all — the basin's
climb IS the task, so every interaction is applied external force + contact dynamics:

  PHASE 1  FLIGHT A — pulsed horizontal push on the basin along the fixture's +x, up
           the shallow outer flight from the porch onto the turning pad.
  PHASE 2  THE TURN — pulsed push along -y across the turning pad into the inner-lane
           row (the direction of travel reverses here).
  PHASE 3  FLIGHT B — pulsed push along -x, across the pad lip and up the steeper
           inner flight onto the top pad level with the roof.
  PHASE 4  ENTRY — gentle pulsed push along -y, off the top pad (2 mm step DOWN)
           through the roof-rail gap onto the cabinet roof; release, settle.
  PHASE 5  persistence — >= 3.5 simulated seconds completely hands-off; success()
           must still hold before the verdict is printed.

Prints the scene readouts, `SIM_GEN_SCORE <score>` at each phase boundary (latched
credit — the printed sequence never decreases), and exactly `SIM_GEN_SOLVE: SUCCESS`
only if success() still holds after the persistence window. Hard exit (os._exit) after
the verdict, with a watchdog Timer as backstop.

The intended single-Franka-arm strategy for the same plan (fingertip pushes low on the
basin wall from above the open-top lanes) lives in TASK.md as the embodiment argument.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene4_put_the_black_bowl_on_top_of_the_cabinet_i409.solve --headless
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
    from simgen_tasks.libero_kitchen_scene4_put_the_black_bowl_on_top_of_the_cabinet_i409 import (  # noqa: F401,E501
        scene as scene_mod,
    )
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
_wd = threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                             os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.switchback_ramp")().build(num_envs=1, device=device)
    scene = env.scene
    no_action = torch.empty(0, device=device)
    zero3 = torch.zeros(1, 1, 3, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def clear_forces() -> None:
        scene.bowl.set_external_force_and_torque(zero3, zero3)

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

    def bfx() -> torch.Tensor:
        return scene.bowl_fix()[0]

    def report(tag: str) -> None:
        b = bfx()
        print(f"[solve] {tag:8s} bowl_fix=({float(b[0]):+.3f},{float(b[1]):+.3f},"
              f"{float(b[2]):+.3f}) "
              f"A={bool(scene._flightA_ever[0])} P={bool(scene._pad_ever[0])} "
              f"B={bool(scene._flightB_ever[0])} roof={bool(scene.on_roof()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"score={sc():.2f} success={bool(scene.success()[0])}", flush=True)

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
        t = threading.Timer(10.0, lambda: os._exit(code))
        t.daemon = True
        t.start()
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
    assert not bool(scene._flightA_ever[0] | scene._pad_ever[0]
                    | scene._flightB_ever[0]), "no latch may fire at reset"
    assert float(bfx()[2]) < 0.03, "basin must spawn on the ground-level porch"
    phase_score("reset")  # ~0.000

    from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402

    fq = scene.fixture.data.root_quat_w[0:1]
    ex = quat_apply(fq, torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]  # fixture +x
    ey = quat_apply(fq, torch.tensor([[0.0, 1.0, 0.0]], device=device))[0]  # fixture +y

    def v_along(body, axis: torch.Tensor) -> float:
        return float((body.data.root_lin_vel_w[0] * axis).sum())

    def drive(axis: torch.Tensor, fmag: float, vmax: float, done,
              max_steps: int, tag: str) -> bool:
        """Pulsed push on the basin along world `axis` until `done()`. On these pods
        the default external-force call applies the wrench in the body's CURRENT
        frame, so mode 0 pre-encodes per step with quat_apply_inverse(q_now, f)
        (correct for the basin's free spawn yaw); mode 1 is raw world as the
        fallback, toggled from a measured-progress stall probe."""
        body = scene.bowl
        mode = 0
        last_probe = float((body.data.root_pos_w[0] * axis).sum())
        for i in range(max_steps):
            if done():
                clear_forces()
                return True
            f = fmag if v_along(body, axis) < vmax else 0.0
            fw = f * axis
            if mode == 0:
                fw = quat_apply_inverse(body.data.root_quat_w[0:1], fw.view(1, 3))[0]
            body.set_external_force_and_torque(fw.view(1, 1, 3), zero3)
            step(1)
            if (i + 1) % 40 == 0:
                cur = float((body.data.root_pos_w[0] * axis).sum())
                b = bfx()
                print(f"[solve] {tag} i={i + 1} along={cur:+.4f} "
                      f"d={cur - last_probe:+.4f} v={v_along(body, axis):+.3f} "
                      f"fix=({float(b[0]):+.3f},{float(b[1]):+.3f},{float(b[2]):+.3f})",
                      flush=True)
                if cur - last_probe < 0.001:
                    mode ^= 1
                    print(f"[solve] {tag}: progress stalled, force mode -> {mode}",
                          flush=True)
                last_probe = cur
        clear_forces()
        return done()

    # ================= PHASE 1: flight A (+x up the outer lane) ================================
    ok1 = drive(ex, 9.0, 0.10, lambda: float(bfx()[0]) > 0.38, 3600, "flightA")
    settle_until(lambda: bool(scene.settled()[0]))
    report("flightA")
    if not (ok1 and bool(scene._flightA_ever[0])):
        print("[solve] PHASE 1 FAILED: basin did not climb flight A onto the pad", flush=True)
        verdict(False)
    phase_score("phase1")  # 0.150

    # ================= PHASE 2: the turn (-y across the pad) ===================================
    ok2 = drive(-ey, 7.0, 0.10, lambda: float(bfx()[1]) < 0.26, 2400, "turn")
    settle_until(lambda: bool(scene.settled()[0]))
    report("turned")
    if not (ok2 and bool(scene._pad_ever[0])):
        print("[solve] PHASE 2 FAILED: basin did not cross into the inner-lane row", flush=True)
        verdict(False)
    phase_score("phase2")  # 0.350

    # ================= PHASE 3: flight B (-x up the inner lane) ================================
    ok3 = drive(-ex, 13.0, 0.10, lambda: float(bfx()[0]) < -0.10, 3600, "flightB")
    settle_until(lambda: bool(scene.settled()[0]))
    report("flightB")
    if not (ok3 and bool(scene._flightB_ever[0])):
        print("[solve] PHASE 3 FAILED: basin did not climb flight B to the top pad", flush=True)
        verdict(False)
    phase_score("phase3")  # 0.600

    # ================= PHASE 4: roof entry (-y through the rail gap) ===========================
    ok4 = drive(-ey, 7.0, 0.08, lambda: float(bfx()[1]) < 0.06, 2400, "entry")
    ok4s = settle_until(lambda: bool(scene.on_roof()[0]) and bool(scene.settled()[0]))
    report("roof")
    if not (ok4 and ok4s):
        print("[solve] PHASE 4 FAILED: basin did not come to rest on the roof", flush=True)
        verdict(False)
    phase_score("phase4")  # 1.000 (success)

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

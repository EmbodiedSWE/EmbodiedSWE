"""solve — force-driven solution for WedgeLiftScene
(libero_kitchen_scene4_put_the_black_bowl_on_top_of_the_cabinet_i308).

Scene-level env (robot="null"). This solution uses NO teleports at all — the ram's
travel and the bowl's slide ARE the load-bearing mechanism, so every interaction is
applied external force + contact dynamics:

  PHASE 1  RAM — pulsed horizontal push on the wedge ram along the fixture's -x (into
           the tunnel) until the paddle seats against the cabinet lintel. The
           elevator's cylindrical foot rides the 40 deg wedge face and the platform
           (with the bowl aboard) climbs ~13.5 cm to stand ~4 mm proud of the roof;
           the foot then rests on the ram's FLAT crest, so the lift holds with the
           force off. Release, settle.
  PHASE 2  SLIDE — gentle pulsed push on the bowl along -x: it slides off the proud
           platform (a small step DOWN) onto the rooftop landing area behind the well;
           release, settle. The terminal state is fully passive on static roof.
  PHASE 3  persistence — >= 3 simulated seconds completely hands-off; success() must
           still hold before the verdict is printed.

Prints the scene readouts, `SIM_GEN_SCORE <score>` at each phase boundary (latched
credit — the printed sequence never decreases), and exactly `SIM_GEN_SOLVE: SUCCESS`
only if success() still holds after the persistence window. Hard exit (os._exit) after
the verdict, with a watchdog Timer as backstop.

The intended single-Franka-arm strategy for the same plan (palm-push the tall orange
paddle, then reach over the roof and slide the exposed bowl backward with a fingertip)
lives in TASK.md as the embodiment argument.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene4_put_the_black_bowl_on_top_of_the_cabinet_i308.solve --headless
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
    from simgen_tasks.libero_kitchen_scene4_put_the_black_bowl_on_top_of_the_cabinet_i308 import (  # noqa: F401,E501
        scene as scene_mod,
    )
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.wedge_lift")().build(num_envs=1, device=device)
    scene = env.scene
    no_action = torch.empty(0, device=device)
    zero3 = torch.zeros(1, 1, 3, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def clear_forces() -> None:
        scene.ram.set_external_force_and_torque(zero3, zero3)
        scene.bowl.set_external_force_and_torque(zero3, zero3)

    def push(body, f_world: torch.Tensor) -> None:
        scene.ram.set_external_force_and_torque(zero3, zero3)
        scene.bowl.set_external_force_and_torque(zero3, zero3)
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
        rx = float(scene.ram_fix_x()[0])
        ez = float(scene.elev_top_z()[0])
        bf = scene._to_fix(scene.bowl.data.root_pos_w)[0]
        print(f"[solve] {tag:12s} ram_x={rx:+.3f} elev_top={ez:.3f} "
              f"bowl_fix=({bf[0]:+.3f},{bf[1]:+.3f},{bf[2]:+.3f}) "
              f"aboard={bool(scene.bowl_aboard()[0])} raised={bool(scene.raised()[0])} "
              f"landed={bool(scene.landed()[0])} settled={bool(scene.settled()[0])} "
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
    assert not bool(scene._partlift_ever[0] | scene._raised_ever[0]
                    | scene._landed_ever[0]), "no latch may fire at reset"
    assert bool(scene.bowl_aboard()[0]), "bowl must spawn riding the lowered elevator"
    phase_score("reset")  # ~0.000

    from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402

    ex = quat_apply(scene.fixture.data.root_quat_w[0:1],
                    torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]  # fixture +x, world

    def v_along(body, axis: torch.Tensor) -> float:
        return float((body.data.root_lin_vel_w[0] * axis).sum())

    def drive(body, axis: torch.Tensor, fmag: float, vmax: float, done,
              max_steps: int, tag: str) -> bool:
        """Pulsed push along world `axis` until `done()`. On these pods the default
        external-force call applies the wrench in the body's CURRENT frame, so mode 0
        pre-encodes per step with quat_apply_inverse(q_now, f) (correct for any spawn
        yaw — the bowl's is free); mode 1 is raw world as the fallback, toggled from a
        measured-progress stall probe."""
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
            push(body, fw)
            step(1)
            if (i + 1) % 40 == 0:
                cur = float((body.data.root_pos_w[0] * axis).sum())
                print(f"[solve] {tag} i={i + 1} along={cur:+.4f} "
                      f"d={cur - last_probe:+.4f} v={v_along(body, axis):+.3f} "
                      f"elev_top={float(scene.elev_top_z()[0]):.3f}", flush=True)
                if cur - last_probe < 0.001:
                    mode ^= 1
                    print(f"[solve] {tag}: progress stalled, force mode -> {mode}",
                          flush=True)
                last_probe = cur
        clear_forces()
        return done()

    # ================= PHASE 1: ram the wedge home =============================================
    # Pulsed push toward -x (into the tunnel): force on only while the ram is slow, so
    # it creeps up the rail; the elevator foot rides the wedge face (platform climbs at
    # ~tan40 deg of the ram speed) and the cabinet lintel takes a soft seating impact.
    # Done = paddle seated (ram x near the built-in stop) AND platform in the raised band.
    seat_x = scene.cfg.ram_in_x + 0.012
    drive(scene.ram, -ex, 25.0, 0.10,
          lambda: bool(scene.raised()[0]) and float(scene.ram_fix_x()[0]) < seat_x
          and abs(v_along(scene.ram, -ex)) < 0.02,
          3600, "ram")
    ok1 = settle_until(lambda: bool(scene.raised()[0]) and bool(scene.settled()[0]))
    report("rammed")
    if not (ok1 and bool(scene._partlift_ever[0]) and bool(scene._raised_ever[0])):
        print("[solve] PHASE 1 FAILED: platform did not reach the raised band "
              "with the bowl aboard", flush=True)
        verdict(False)
    assert not bool(scene._landed_ever[0]), "landed latch must not fire from the lift"
    phase_score("phase1")  # 0.400

    # ================= PHASE 2: slide the bowl off onto the roof ===============================
    # Gentle pulsed push toward -x: the bowl slides off the proud platform (small step
    # DOWN onto the roof) and keeps creeping until it sits mid-landing-area
    # (fixture x < -0.14, comfortably inside the [-0.30, -0.06] band); friction stops it.
    drive(scene.bowl, -ex, 1.5, 0.08,
          lambda: float(scene._to_fix(scene.bowl.data.root_pos_w)[0, 0]) < -0.14,
          2400, "slide")
    ok2 = settle_until(lambda: bool(scene.landed()[0]) and bool(scene.settled()[0]))
    report("landed")
    if not (ok2 and bool(scene._landed_ever[0])):
        print("[solve] PHASE 2 FAILED: bowl did not come to rest on the roof landing "
              "area", flush=True)
        verdict(False)
    phase_score("phase2")  # 1.000 (success)

    # ================= persistence (>= 3 simulated seconds, hands off) =========================
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

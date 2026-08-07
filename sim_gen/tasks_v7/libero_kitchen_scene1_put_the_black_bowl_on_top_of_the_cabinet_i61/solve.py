"""solve — force-driven solution for CartFerryScene
(libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet_i61).

Scene-level env (robot="null"). This solution uses NO teleports at all — the cart's
travel and the bowl's slide ARE the load-bearing mechanism, so every interaction is
applied external force + contact dynamics:

  PHASE 1  DOCK — pulsed horizontal push on the cart along the fixture's +x (lane)
           axis until its base seats against the dock stop (speed-capped pulses, the
           stop takes the impact); release, let friction settle it.
  PHASE 2  TRANSFER — gentle pulsed push on the bowl along the fixture's -x axis: it
           slides along the ledge, out the alcove's open front, across the 2.5 cm gap
           and 2 mm step down onto the cart's docked counter top; release, settle.
  PHASE 3  FERRY — gentle pulsed pull on the cart along -x (net acceleration kept well
           under mu*g so friction keeps the bowl aboard) until the cart is in the HOME
           band; release, settle, then >= 3 simulated seconds hands-off; success()
           must persist before the verdict is printed.

Prints the scene readouts, `SIM_GEN_SCORE <score>` at each phase boundary (latched
credit — the printed sequence never decreases), and exactly `SIM_GEN_SOLVE: SUCCESS`
only if success() still holds after the persistence window. Hard exit (os._exit) after
the verdict, with a watchdog Timer as backstop.

The intended single-Franka-arm strategy for the same plan (palm-push the cart, reach
into the alcove alongside the too-wide-to-grasp bowl and drag it out by a fingertip
behind its wall, pull the handle bar home) lives in TASK.md as the embodiment argument.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet_i61.solve --headless
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
    from simgen_tasks.libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet_i61 import (  # noqa: F401,E501
        scene as scene_mod,
    )
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cart_ferry")().build(num_envs=1, device=device)
    scene = env.scene
    no_action = torch.empty(0, device=device)
    zero3 = torch.zeros(1, 1, 3, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def clear_forces() -> None:
        scene.cart.set_external_force_and_torque(zero3, zero3)
        scene.bowl.set_external_force_and_torque(zero3, zero3)

    def push(body, f_world: torch.Tensor) -> None:
        scene.cart.set_external_force_and_torque(zero3, zero3)
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
        cxy = scene.cart_fix_xy()[0]
        bf = scene._to_fix(scene.bowl.data.root_pos_w)[0]
        bc = scene._to_cart(scene.bowl.data.root_pos_w)[0]
        print(f"[solve] {tag:14s} cart_fix=({cxy[0]:+.3f},{cxy[1]:+.3f}) "
              f"bowl_fix=({bf[0]:+.3f},{bf[1]:+.3f},{bf[2]:+.3f}) "
              f"bowl_cart=({bc[0]:+.3f},{bc[1]:+.3f},{bc[2]:+.3f}) "
              f"docked={bool(scene.cart_docked()[0])} "
              f"aboard={bool(scene.bowl_on_cart()[0])} home={bool(scene.cart_home()[0])} "
              f"void={bool(scene._voided_ever[0])} settled={bool(scene.settled()[0])} "
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
    assert not bool(scene._docked_ever[0] | scene._transferred_ever[0]
                    | scene._ride_ever[0] | scene._home_ever[0]
                    | scene._voided_ever[0]), "no latch may fire at reset"
    phase_score("reset")  # ~0.000

    from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402

    ex = quat_apply(scene.fixture.data.root_quat_w[0:1],
                    torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]  # lane +x, world

    def v_along(body, axis: torch.Tensor) -> float:
        return float((body.data.root_lin_vel_w[0] * axis).sum())

    def drive(body, axis: torch.Tensor, fmag: float, vmax: float, done,
              max_steps: int, tag: str) -> bool:
        """Pulsed push along world `axis` until `done()`. The pod's external-force API
        may interpret the wrench in the body's CURRENT frame (pod/version dependent),
        so probe from measured progress: if the body stops making headway, toggle
        between raw-world and quat_apply_inverse(q_now, f) encodings."""
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

    # ================= PHASE 1: push the cart to the dock stop =================================
    # Pulsed push: force on only while the cart is slow, so it creeps down the lane and
    # the dock stop takes a soft (<0.15 m/s) seating impact.
    drive(scene.cart, ex, 25.0, 0.15,
          lambda: bool(scene.cart_docked()[0]) and v_along(scene.cart, ex) < 0.02,
          1800, "dock")
    ok1 = settle_until(lambda: bool(scene.cart_docked()[0]) and bool(scene.settled()[0]))
    report("docked")
    if not (ok1 and bool(scene._docked_ever[0])):
        print("[solve] PHASE 1 FAILED: cart did not seat at the dock", flush=True)
        verdict(False)
    assert not bool(scene._transferred_ever[0]), "transfer latch must not fire from docking"
    phase_score("phase1")  # 0.200

    # ================= PHASE 2: slide the bowl out onto the counter top ========================
    # Gentle pulsed push toward -x: the bowl slides under the roof, out the open front,
    # across the gap onto the docked counter top, and keeps creeping until it sits WELL
    # onto the plate (fixture x < -0.08 -> cart-frame x ~ 0.155, ~11 cm of slack to the
    # plate's front edge); friction stops it there.
    drive(scene.bowl, -ex, 1.2, 0.08,
          lambda: float(scene._to_fix(scene.bowl.data.root_pos_w)[0, 0]) < -0.08,
          2400, "slide")
    ok2 = settle_until(lambda: bool(scene.bowl_on_cart()[0]) and bool(scene.settled()[0]))
    report("transferred")
    if not (ok2 and bool(scene._transferred_ever[0]) and not bool(scene._voided_ever[0])):
        print("[solve] PHASE 2 FAILED: bowl did not come to rest on the counter top",
              flush=True)
        verdict(False)
    phase_score("phase2")  # 0.500

    # ================= PHASE 3: ferry the cart home with the bowl riding =======================
    # Gentle pulsed pull toward -x: net acceleration ~2 m/s^2 << mu*g (~3.7), so the
    # bowl rides on friction alone. Stop pulling past the home line; friction stops it.
    drive(scene.cart, -ex, 20.0, 0.08,
          lambda: float(scene.cart_fix_xy()[0, 0]) < scene.cfg.home_x - 0.02,
          2400, "ferry")
    ok3 = settle_until(lambda: bool(scene.success()[0]))
    report("home")
    if not ok3:
        print("[solve] PHASE 3 FAILED: cart+bowl did not settle in the home band",
              flush=True)
        verdict(False)
    phase_score("phase3")  # 1.000

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

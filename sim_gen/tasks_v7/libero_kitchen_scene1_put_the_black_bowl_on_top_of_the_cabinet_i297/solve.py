"""solve — teleport-transport + force solution for BowlAirlockScene
(libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet_i297).

Scene-level env (robot="null"). Teleports are TRANSPORT ONLY: the single teleport moves
the bowl from the pedestal to just above the loading sill (the part a robot would do by
rim-pinch grasp + carry through open space). Every load-bearing interaction — sliding
the gate carriage, pushing the bowl through the front window, the gravity feed down the
slick chamber slope and out the inner window — is applied external force + contact
dynamics.

  PHASE A  GATE->LOAD — pulsed push on the carriage along the fixture's -y until it
           rests at the LOAD end (no-op if it randomized there). Front window opens.
  PHASE B  STAGE — teleport the bowl to 2 cm above the sill center, drop, settle.
  PHASE C  LOAD — gentle pulsed push on the bowl along the fixture's +x: across the
           sill, through the front window, onto the slick 20-deg slope; release — it
           glides down and rests against the closed inner panel.
  PHASE D  DISPENSE — pulsed push on the carriage along +y to the DISPENSE end: the
           inner window opens (the front seals) and the bowl gravity-feeds through,
           drops onto the case deck, and settles upright. No assist force is applied
           after the gate opens — the slope must do it, exactly as for a real robot
           (the chamber is unreachable once the front is sealed).

Prints readouts, `SIM_GEN_SCORE <score>` at phase boundaries (latched, never
decreasing), then holds >= 3.5 simulated seconds fully hands-off and prints exactly
`SIM_GEN_SOLVE: SUCCESS` only if success() still holds. Hard exit via os._exit with a
watchdog Timer backstop.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet_i297.solve --headless
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
    from simgen_tasks.libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet_i297 import (  # noqa: F401,E501
        scene as scene_mod,
    )
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.bowl_airlock")().build(num_envs=1, device=device)
    scene = env.scene
    no_action = torch.empty(0, device=device)
    zero3 = torch.zeros(1, 1, 3, device=device)
    ids0 = torch.tensor([0], device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def clear_forces() -> None:
        scene.carriage.set_external_force_and_torque(zero3, zero3)
        scene.bowl.set_external_force_and_torque(zero3, zero3)

    def push(body, f_world: torch.Tensor) -> None:
        clear_forces()
        body.set_external_force_and_torque(f_world.view(1, 1, 3), zero3)

    def settle_until(pred, max_steps: int = 720, poll: int = 10, min_steps: int = 0) -> bool:
        if min_steps:
            step(min_steps)
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
        bf = scene.bowl_fix()[0]
        s = float(scene.carriage_s()[0])
        print(f"[solve] {tag:12s} bowl_fix=({bf[0]:+.3f},{bf[1]:+.3f},{bf[2]:+.3f}) "
              f"carriage_s={s:+.3f} load={bool(scene.carriage_at_load()[0])} "
              f"disp={bool(scene.carriage_at_dispense()[0])} "
              f"staged={bool(scene._staged_ever[0])} "
              f"chambered={bool(scene._chambered_ever[0])} "
              f"delivered={bool(scene._delivered_ever[0])} "
              f"placed={bool(scene.bowl_placed()[0])} "
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
    assert not bool(scene._staged_ever[0] | scene._chambered_ever[0]
                    | scene._delivered_ever[0]), "no latch may fire at reset"
    phase_score("reset")  # ~0.000

    from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402

    fq0 = scene.fixture.data.root_quat_w[0:1]
    ex = quat_apply(fq0, torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
    ey = quat_apply(fq0, torch.tensor([[0.0, 1.0, 0.0]], device=device))[0]

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

    # ================= PHASE A: slide the gate carriage to the LOAD end ========================
    # No-op if it randomized at LOAD. Pulsed push so the end post takes a soft stop.
    okA = drive(scene.carriage, -ey, 8.0, 0.10,
                lambda: bool(scene.carriage_at_load()[0])
                and abs(v_along(scene.carriage, ey)) < 0.02,
                1800, "gate->load")
    okA = settle_until(lambda: bool(scene.carriage_at_load()[0])
                       and bool(scene.settled()[0])) and okA
    report("gate_load")
    if not okA:
        print("[solve] PHASE A FAILED: carriage did not reach the LOAD end", flush=True)
        verdict(False)
    phase_score("phaseA")  # still ~0.000

    # ================= PHASE B: stage the bowl on the sill (teleport = transport) ==============
    fp0 = scene.fixture.data.root_pos_w[0]
    tgt = quat_apply(fq0, torch.tensor([[-0.255, 0.0, 0.3495]], device=device))[0] + fp0
    st = torch.zeros(1, 13, device=device)
    st[0, 0:3] = tgt
    st[0, 3:7] = fq0[0]
    scene.bowl.write_root_state_to_sim(st, ids0)
    okB = settle_until(lambda: bool(scene.bowl_staged()[0]) and bool(scene.settled()[0]),
                       min_steps=30)
    report("staged")
    if not (okB and bool(scene._staged_ever[0])):
        print("[solve] PHASE B FAILED: bowl did not settle staged on the sill", flush=True)
        verdict(False)
    assert not bool(scene._chambered_ever[0]), "chamber latch must not fire from staging"
    phase_score("phaseB")  # 0.150

    # ================= PHASE C: push the bowl through the front window =========================
    # Gentle pulsed push toward +x: across the grippy sill, through the open front
    # window, onto the slick slope. Release once it is past the wall plane (fixture
    # x > -0.145); gravity feeds it down against the closed inner panel.
    drive(scene.bowl, ex, 2.2, 0.08,
          lambda: float(scene.bowl_fix()[0, 0]) > -0.145,
          2400, "load-push")
    okC = settle_until(lambda: bool(scene.bowl_chambered()[0]) and bool(scene.settled()[0]),
                       max_steps=960, min_steps=30)
    report("chambered")
    if not (okC and bool(scene._chambered_ever[0])):
        print("[solve] PHASE C FAILED: bowl did not come to rest in the chamber", flush=True)
        verdict(False)
    assert not bool(scene._delivered_ever[0]), \
        "delivered latch must not fire while the inner window is closed"
    phase_score("phaseC")  # 0.400

    # ================= PHASE D: slide the carriage to DISPENSE — gravity does the rest =========
    # The inner window opens (the front seals). NO force ever touches the bowl in this
    # phase: it must glide through the window and onto the deck on the slope alone.
    okD = drive(scene.carriage, ey, 8.0, 0.10,
                lambda: bool(scene.carriage_at_dispense()[0])
                and abs(v_along(scene.carriage, ey)) < 0.02,
                1800, "gate->dispense")
    okD = settle_until(lambda: bool(scene.success()[0]),
                       max_steps=1200, min_steps=30) and okD
    report("delivered")
    if not okD:
        print("[solve] PHASE D FAILED: bowl did not dispense onto the case deck", flush=True)
        verdict(False)
    phase_score("phaseD")  # 1.000

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

"""solve — TELEPORT solution for BeamBalanceScene
(libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i187).

Scene-level env (robot="null"). Teleportation is used for TRANSPORT ONLY; every
load-bearing interaction goes through contact dynamics:

  PHASE 1  weigh a LIGHT bowl (the honest measurement the task is about): transport it
           to just ABOVE the pan — the release height is deliberately OUTSIDE the
           geometric "on pan" z band, so no credit can latch at the teleport instant.
           Gravity drops it onto the plate; the beam's response is pure contact physics:
           it STAYS at its pan-up stop, showing this bowl is light.
  PHASE 2  transport the light bowl back to its ground slot (weighing done, verdict
           negative). The weigh credit stays latched — printed scores never decrease.
  PHASE 3  weigh the HEAVY bowl the same way: released above the pan, it lands by
           contact and its real mass — written to PhysX at reset, verified by readback
           here — overcomes the built-in counterweight bias and drives the beam down
           against its lower stop. Nothing is pinned, no wrench is applied: the tipping
           IS the simulator's own verdict on the bowl's mass, which is exactly what
           success() requires.
  PHASE 4  hands off for >= 3 simulated seconds; success() must persist (a light bowl
           faking the tip transiently would relax back up and fail here) before the
           verdict is printed.

Prints the scene readouts, `SIM_GEN_SCORE <score>` at each phase boundary (latched
credit — the printed sequence never decreases), and exactly `SIM_GEN_SOLVE: SUCCESS`
only if success() still holds after the persistence window. Hard exit (os._exit) after
the verdict, with a watchdog Timer as backstop — Kit teardown hangs otherwise.

The intended single-Franka-arm strategy for the same plan (rim pinch-grasp per bowl,
place/observe/remove loop on the pan) lives in TASK.md as the embodiment argument.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i187.solve --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_sec", type=float, default=600.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i187 import (  # noqa: F401,E501
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
    env = ENVS.get("simgen.beam_balance")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(1, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def settle_until(pred, max_steps: int = 600, poll: int = 10) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def sc() -> float:
        return float(scene.score()[0])

    def ang() -> float:
        return math.degrees(float(scene.beam_angle()[0]))

    def report(tag: str) -> None:
        onp = scene.on_pan()[0]
        zs = (scene._bowl_pos_w() - scene.env_origins[:, None, :])[0, :, 2]
        print(f"[solve] {tag:14s} ang={ang():+6.2f} "
              f"on_pan={[bool(v) for v in onp]} "
              f"bowl_z=({zs[0]:.3f},{zs[1]:.3f},{zs[2]:.3f}) "
              f"down={bool(scene.beam_down()[0])} up={bool(scene.beam_up()[0])} "
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

    def beam_local_to_world(loc):
        from isaaclab.utils.math import quat_apply

        p = quat_apply(scene.beam.data.root_quat_w[0:1],
                       torch.tensor([loc], device=device))[0]
        return p + scene.beam.data.root_pos_w[0]

    def teleport(body, pos_w, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        """TRANSPORT: set a root pose with zero velocity. Never used to enter a scoring
        band — release points are chosen outside every credit region."""
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = torch.tensor([float(v) for v in pos_w], device=device)
        st[0, 3:7] = torch.tensor([float(v) for v in quat], device=device)
        body.write_root_state_to_sim(st, all_ids)

    def release_over_pan(bowl) -> None:
        """Transport a bowl to just above the pan (beam-local, tracks the live beam
        tilt), OUTSIDE the on-pan z band, and let gravity do the weighing."""
        rel_z = c.plate_top_z + c.bowl_h / 2 + 0.045
        assert rel_z > c.pan_z_hi, "release point must be outside the on-pan band"
        pos = beam_local_to_world([0.0, c.arm_len, rel_z])
        teleport(bowl, [float(v) for v in pos])

    # ================= reset + settle ==========================================================
    env.reset(seed=args.seed)
    step(90)
    report("reset")

    # mass readback guard (custom spawners silently ignore cfg masses; reset() writes
    # them through the physx view — verify the sim runs the masses the episode sampled)
    for k, b in enumerate(scene.bowls):
        authored = float(scene._masses[0, k])
        actual = float(b.root_physx_view.get_masses().reshape(-1)[0])
        assert abs(actual - authored) < 1e-4, (
            f"bowl_{k} mass readback {actual:.4f} != authored {authored:.4f}")
    heavy = int(scene.heavy_index()[0])
    light = int(scene._masses[0].argmin())
    print(f"[solve] masses={[round(float(v), 3) for v in scene._masses[0]]} "
          f"heavy_idx={heavy} light_idx={light}", flush=True)
    assert bool(scene.beam_up()[0]), "beam must start resting pan-side-up"
    phase_score("reset")  # ~0.000

    # ================= PHASE 1: weigh a LIGHT bowl (beam stays up) =============================
    home = (scene.bowls[light].data.root_pos_w[0] - scene.env_origins[0]).clone()
    release_over_pan(scene.bowls[light])
    step(1)
    assert not bool(scene._weigh_ever[0]), "weigh latch must not fire at the teleport"
    ok1 = settle_until(lambda: bool(scene.on_pan()[0, light]) and bool(scene.settled()[0]),
                       max_steps=600)
    report("weigh-light")
    if not (ok1 and bool(scene._weigh_ever[0])):
        print("[solve] PHASE 1 FAILED: light bowl did not settle on the pan", flush=True)
        verdict(False)
    if not bool(scene.beam_up()[0]):
        print("[solve] PHASE 1 FAILED: beam should NOT tip for a light bowl", flush=True)
        verdict(False)
    print(f"[solve] measurement: light bowl on pan, beam stays at {ang():+.2f} deg "
          "(pan-up) -> this bowl is light", flush=True)
    phase_score("phase1")  # 0.250 (lifted + weighed)

    # ================= PHASE 2: take the light bowl back off ===================================
    teleport(scene.bowls[light], [float(home[0]), float(home[1]),
                                  c.bowl_h / 2 + 0.010])
    ok2 = settle_until(lambda: bool(scene.grounded()[0, light]) and bool(scene.settled()[0])
                       and bool(scene.beam_up()[0]), max_steps=600)
    report("light-off")
    if not ok2:
        print("[solve] PHASE 2 FAILED: scene did not re-settle after removing the light "
              "bowl", flush=True)
        verdict(False)
    phase_score("phase2")  # 0.250 (latched credit does not evaporate)

    # ================= PHASE 3: weigh the HEAVY bowl (beam driven to its stop) =================
    release_over_pan(scene.bowls[heavy])
    step(1)
    report("release-heavy")
    ok3 = settle_until(lambda: bool(scene.success()[0]), max_steps=900)
    report("tipped")
    if not ok3:
        print("[solve] PHASE 3 FAILED: heavy bowl did not drive the beam to its down "
              "stop / goal state not reached", flush=True)
        verdict(False)
    phase_score("phase3")  # 1.000

    # ================= PHASE 4: persistence (>= 3 simulated seconds, hands off) ================
    persist_steps = int(round(3.5 / env.dt))  # 420 physics steps at 1/120 s
    step(persist_steps)
    report("final")
    phase_score("final")
    still_ok = bool(scene.success()[0]) and sc() == 1.0
    print(f"[solve] persistence: {persist_steps} steps ({persist_steps * env.dt:.2f} s) "
          f"hands-off, success={still_ok}", flush=True)
    verdict(still_ok)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(2)

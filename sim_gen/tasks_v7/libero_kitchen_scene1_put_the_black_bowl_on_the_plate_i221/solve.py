"""solve — TELEPORT solution for ClocheServiceScene
(libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i221).

Scene-level env (robot="null"). Teleportation is used for TRANSPORT ONLY; every
load-bearing interaction goes through contact dynamics:

  PHASE 1  UNCOVER: transport the resting cover from the plate to a park spot on the
           floor (released a few mm up; it drops onto its rim and settles standing).
           Lifting a freely resting cover IS transport — no interaction is bypassed.
  PHASE 2  SERVE: transport the cake to just ABOVE the freed plate — the release
           height is asserted OUTSIDE the on-plate z band, so no credit can exist at
           the teleport instant. Gravity drops it; it lands and settles standing on
           the plate top through contact.
  PHASE 3  RE-COVER (the load-bearing finale): transport the cover to just above the
           plate, slightly OFF-center, with its rim asserted ABOVE the seated z band
           and above the cake top. It falls under gravity; the rim must pass AROUND
           the standing cake and land flat on the plate — the seating, the miss of
           the cake, and the final rest pose are pure contact dynamics. Nothing is
           spawned seated, nothing is pinned.
  PHASE 4  hands off for >= 3.5 simulated seconds; success() must persist (a cover
           perched on the cake or still rocking would fail here) before the verdict.

Prints the scene readouts, `SIM_GEN_SCORE <score>` at each phase boundary (latched
credit — the printed sequence never decreases), and exactly `SIM_GEN_SOLVE: SUCCESS`
only if success() still holds after the persistence window. Hard exit (os._exit) after
the verdict, with a daemon watchdog Timer as backstop — Kit teardown hangs otherwise.

The intended single-Franka-arm strategy for the same plan (knob pinch-grasp for the
cover, side pinch for the cake) lives in TASK.md as the embodiment argument.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i221.solve --headless
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

import os  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i221 import (  # noqa: F401,E501
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
    env = ENVS.get("simgen.cloche_service")().build(num_envs=1, device=device)
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

    def report(tag: str) -> None:
        d_cp = float(scene._dxy(scene.cake, scene.plate)[0])
        d_cc = float(scene._dxy(scene.cake, scene.cloche)[0])
        rim_dz = float((scene.cloche_rim_z() - scene.plate_top_z())[0])
        print(f"[solve] {tag:12s} cake_on_plate={bool(scene.cake_on_plate()[0])} "
              f"seated={bool(scene.seated()[0])} enclosed={bool(scene.enclosed()[0])} "
              f"covered={bool(scene.covered()[0])} settled={bool(scene.settled()[0])} "
              f"d(cake,plate)={d_cp:.3f} d(cake,cover)={d_cc:.3f} rim_dz={rim_dz:+.4f} "
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

    def teleport(body, pos_w, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        """TRANSPORT: set a root pose with zero velocity. Release points are chosen
        (and asserted) OUTSIDE every credit band — teleports never enter a scoring
        state; gravity and contact finish every placement."""
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = torch.tensor([float(v) for v in pos_w], device=device)
        st[0, 3:7] = torch.tensor([float(v) for v in quat], device=device)
        body.write_root_state_to_sim(st, all_ids)

    # ================= reset + settle ==========================================================
    env.reset(seed=args.seed)
    step(90)
    report("reset")

    # mass readback guard (custom spawners silently ignore cfg masses — the spawn funcs
    # author MassAPI; verify the sim runs the authored masses)
    for name, body, want in (("plate", scene.plate, c.plate_mass),
                             ("cloche", scene.cloche, c.cloche_mass),
                             ("cake", scene.cake, c.cake_mass)):
        actual = float(body.root_physx_view.get_masses().reshape(-1)[0])
        assert abs(actual - want) < 1e-3, f"{name} mass readback {actual:.4f} != {want:.4f}"
        print(f"[solve] mass readback {name}: {actual:.3f} kg", flush=True)

    assert bool(scene.seated()[0]), "cover must start seated on the plate"
    assert not bool(scene.success()[0]), "reset state must not satisfy the goal"
    phase_score("reset")  # 0.000

    plate_xy = scene.plate.data.root_pos_w[0, :2].clone()
    plate_top = float(scene.plate_top_z()[0])

    # ================= PHASE 1: UNCOVER (transport the resting cover off the plate) ============
    park = (float(plate_xy[0]), float(plate_xy[1]) + 0.32)
    assert ((park[0] - plate_xy[0]) ** 2 + (park[1] - plate_xy[1]) ** 2) ** 0.5 > c.uncover_dist
    teleport(scene.cloche, [park[0], park[1], c.cloche_wall_h / 2 + 0.006])
    step(30)  # settled() is vacuously true at the teleport instant — force a landing window
    ok1 = settle_until(lambda: bool(scene.settled()[0]) and bool(scene.cloche_upright()[0]))
    report("uncovered")
    if not (ok1 and bool(scene._uncover_ever[0])):
        print("[solve] PHASE 1 FAILED: cover did not settle standing at the park spot", flush=True)
        verdict(False)
    phase_score("phase1")  # 0.150

    # ================= PHASE 2: SERVE (drop the cake onto the freed plate) =====================
    rel_z = plate_top + c.cake_h / 2 + 0.050
    assert rel_z - (plate_top + c.cake_h / 2) > c.cake_z_hi + 0.02, (
        "cake release must be outside the on-plate z band")
    teleport(scene.cake, [float(plate_xy[0]), float(plate_xy[1]), rel_z])
    step(1)
    assert not bool(scene.cake_on_plate()[0]), "no on-plate credit at the release instant"
    step(30)
    ok2 = settle_until(lambda: bool(scene.cake_on_plate()[0]) and bool(scene.settled()[0]))
    report("served")
    if not (ok2 and bool(scene._serve_ever[0])):
        print("[solve] PHASE 2 FAILED: cake did not settle standing on the plate", flush=True)
        verdict(False)
    phase_score("phase2")  # 0.400

    # ================= PHASE 3: RE-COVER (rim must fall AROUND the cake onto the plate) ========
    rel_rim_z = plate_top + c.cake_h + 0.020  # rim released above the cake top
    root_z = rel_rim_z + c.cloche_wall_h / 2
    assert rel_rim_z - plate_top > c.seat_z_hi + 0.02, (
        "cover release must be outside the seated z band")
    # deliberately off-center: the funnel of the falling rim must still clear the cake
    teleport(scene.cloche, [float(plate_xy[0]) + 0.010, float(plate_xy[1]) - 0.007, root_z])
    step(1)
    assert not bool(scene.seated()[0]), "no seated credit at the release instant"
    step(30)
    ok3 = settle_until(lambda: bool(scene.success()[0]), max_steps=900)
    report("covered")
    if not ok3:
        print("[solve] PHASE 3 FAILED: cover did not seat over the cake / goal not reached",
              flush=True)
        verdict(False)
    phase_score("phase3")  # 1.000

    # ================= PHASE 4: persistence (>= 3.5 simulated seconds, hands off) ==============
    persist_steps = int(round(3.5 / env.dt))
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

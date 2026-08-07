"""solve — TELEPORT solution for CounterweightShelfScene
(libero_kitchen_scene9_put_the_white_bowl_on_top_of_the_cabinet_i5).

Scene-level env (robot="null"). Teleportation is used for TRANSPORT ONLY; every
load-bearing interaction goes through contact dynamics:

  PHASE 1  transport the counterweight cube from the floor to just ABOVE the rim of the
           shelf's rear socket (release point computed from the LIVE tipped-tray pose,
           deliberately OUTSIDE the geometric "seated" band so no credit can latch at the
           teleport instant). Gravity drops it in; the SEATING is pure contact, and the
           rear torque swings the front-heavy shelf up against its level stop — the
           mechanism does the leveling, nothing is pinned or velocity-clamped.
  PHASE 2  transport the white bowl from the floor to just ABOVE the now-level front
           platform (again outside the "placed" z band). It falls the last ~4 cm, lands
           on its flat bottom and RESTS by contact; success() additionally requires the
           whole scene settled and the shelf level, which only the seated counterweight
           can hold.
  PHASE 3  hands off for >= 3 simulated seconds; success() must persist (rejects
           fly-through or precariously balanced states) before the verdict is printed.

Prints the scene readouts, `SIM_GEN_SCORE <score>` at each phase boundary (latched
credit — the printed sequence never decreases), and exactly `SIM_GEN_SOLVE: SUCCESS`
only if success() still holds after the persistence window. Hard exit (os._exit) after
the verdict, with a watchdog Timer as backstop — Kit teardown hangs otherwise.

The intended single-Franka-arm strategy for the same plan (measured and executed in an
earlier arm run: cube pinch-carry into the socket, rim-pinch bowl place) lives in
TASK.md as the embodiment argument.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene9_put_the_white_bowl_on_top_of_the_cabinet_i5.solve --headless
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

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.libero_kitchen_scene9_put_the_white_bowl_on_top_of_the_cabinet_i5 import (  # noqa: F401,E501
        scene as scene_mod,
    )
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.counterweight_shelf")().build(num_envs=1, device=device)
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
        return math.degrees(float(scene.tray_angle()[0]))

    def report(tag: str) -> None:
        bl = scene._to_tray_frame(scene.block.data.root_pos_w)[0]
        wl = scene._to_tray_frame(scene.bowl.data.root_pos_w)[0]
        print(f"[solve] {tag:16s} ang={ang():+7.2f} "
              f"block_loc=({bl[0]:+.3f},{bl[1]:+.3f},{bl[2]:+.3f}) "
              f"bowl_loc=({wl[0]:+.3f},{wl[1]:+.3f},{wl[2]:+.3f}) "
              f"seated={bool(scene.block_seated()[0])} level={bool(scene.tray_level()[0])} "
              f"on_plat={bool(scene.bowl_on_platform()[0])} settled={bool(scene.settled()[0])} "
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

    def tray_local_to_world(loc):
        from isaaclab.utils.math import quat_apply

        p = quat_apply(scene.tray.data.root_quat_w[0:1],
                       torch.tensor([loc], device=device))[0]
        return p + scene.tray.data.root_pos_w[0]

    def teleport(body, pos_w, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        """TRANSPORT: set a root pose with zero velocity. Never used to enter a scoring
        band — release points are chosen outside every credit region."""
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = torch.tensor([float(v) for v in pos_w], device=device)
        st[0, 3:7] = torch.tensor([float(v) for v in quat], device=device)
        body.write_root_state_to_sim(st, all_ids)

    def tilt_quat():
        th = float(scene.tray_angle()[0])
        return (math.cos(th / 2), 0.0, math.sin(th / 2), 0.0)

    # ================= reset + settle ==========================================================
    env.reset(seed=args.seed)
    step(90)
    report("reset")
    phase_score("reset")  # ~0.000

    # ================= PHASE 1: counterweight -> socket (seating by contact) ==================
    # Release the cube ~12 mm above the SOCKET RIM of the tipped tray: tray-local z =
    # wall_h + block/2 + 12 mm = ~0.063, above the seated band's z_hi (0.055), so the
    # teleport itself latches nothing. The drop, the seating against the downhill wall,
    # and the swing to the level stop are all contact dynamics.
    drop_loc = [c.sock_cx, 0.0, c.sock_wall_h + c.block_size / 2 + 0.012]
    assert drop_loc[2] > c.seat_z_hi, "release point must be outside the seated band"
    pos = tray_local_to_world(drop_loc)
    teleport(scene.block, [float(v) for v in pos], tilt_quat())
    step(1)
    report("release-block")
    assert not bool(scene._seated_ever[0]), "seated latch must not fire at the teleport"
    ok1 = settle_until(lambda: bool(scene.block_seated()[0]) and bool(scene.tray_level()[0])
                       and bool(scene.tray_still()[0]), max_steps=600)
    report("leveled")
    if not (ok1 and bool(scene._seated_ever[0]) and bool(scene._leveled_ever[0])):
        print("[solve] PHASE 1 FAILED: counterweight did not seat / shelf did not level",
              flush=True)
        verdict(False)
    phase_score("phase1")  # 0.300 (seated + leveled)

    # ================= PHASE 2: bowl -> level platform (resting by contact) ====================
    # Release the bowl ~40 mm above the platform: tray-local z = bowl_h/2 + 38 mm =
    # ~0.063, above the placed band's z_hi (0.060) — again nothing latches at the
    # teleport; the bowl falls, lands on its flat bottom and rests by contact.
    place_loc = [-0.07, 0.0, c.bowl_h / 2 + 0.038]
    assert place_loc[2] > c.place_z_hi, "release point must be outside the placed band"
    pos = tray_local_to_world(place_loc)
    teleport(scene.bowl, [float(v) for v in pos])
    step(1)
    report("release-bowl")
    assert not bool(scene._placed_ever[0]), "placed latch must not fire at the teleport"
    ok2 = settle_until(lambda: bool(scene.success()[0]), max_steps=600)
    report("placed")
    if not ok2:
        print("[solve] PHASE 2 FAILED: bowl did not come to rest on the level platform",
              flush=True)
        verdict(False)
    phase_score("phase2")  # 1.000

    # ================= PHASE 3: persistence (>= 3 simulated seconds, hands off) ================
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

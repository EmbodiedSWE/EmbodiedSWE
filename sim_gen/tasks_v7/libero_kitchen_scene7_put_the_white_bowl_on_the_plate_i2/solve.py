"""Teleport solution for BowlDecantScene (sim_gen task
libero_kitchen_scene7_put_the_white_bowl_on_the_plate_i2) — the task's legitimacy
certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, small per-step pose increments): the bowl is "held" by
   writing its root pose each physics step along a smooth path — lift, carry, tilt —
   exactly the poses a gripper holding the rim would impose. The BALLS ARE NEVER
   WRITTEN: while the bowl is carried they ride inside it on real contacts; each
   per-step increment is <= 2 mm / <= 1.5 deg. Holding a pose satisfies no rubric
   clause by itself (balls inside the bowl are excluded from "in the dish" by the
   scene's containment clause).
2. POUR (gravity + contact, the core interaction): with the bowl held over the dish,
   its tilt is ramped to ~135 deg and physics does the transfer — the balls roll
   over the bowl's lip, FALL, and are caught and settled by the dish's raised rim.
   Every "in the dish" fact the rubric checks is produced by ballistics and contact,
   never written. If a ball hangs up inside the tilted bowl, the hold angle is
   wiggled (120..150 deg) — the wrist shake a robot would use — never a ball write.
3. PARK (release + gravity): the emptied bowl is carried inverted over the tray and
   RELEASED ~8 mm above the pad; the final resting contact (rim-down inside the
   tray's lips) is made by gravity. Order is forced by physics: inverting before
   pouring would dump the balls outside the dish.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene7_put_the_white_bowl_on_the_plate_i2.solve --headless [--seed N]
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
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod

_qy = scene_mod._qy

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.bowl_decant")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        pos, vel = scene._ball_tensors()
        dloc = scene._local(scene.dish, pos)[0]
        inb = scene.in_bowl()[0]
        ind = scene.in_dish()[0]
        pres = scene.present[0]
        balls = " ".join(
            f"{nm}[{'P' if bool(pres[i]) else '-'}]"
            f"({float(dloc[i, 0]):+.3f},{float(dloc[i, 1]):+.3f},{float(dloc[i, 2]):+.3f})"
            f"{'B' if bool(inb[i]) else ''}{'D' if bool(ind[i]) else ''}"
            for i, nm in enumerate(scene.BALL_NAMES))
        bz = float((scene.bowl.data.root_pos_w - scene.env_origins)[0, 2])
        up = float(scene.bowl_up()[0, 2])
        print(f"[solve] {tag:12s} | {balls} bowl_z={bz:.3f} up_z={up:+.2f} "
              f"vmax={float(vel[0].max()):.3f} lift={bool(scene._lift[0])} "
              f"m1={bool(scene._m1[0])} mall={bool(scene._mall[0])} "
              f"park={bool(scene._park[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def hold_pose(p_local: torch.Tensor, theta: float) -> None:
        """One held-pose write: bowl root at env-local `p_local`, tipped `theta`
        about +y (local +z leans toward +x), zero velocity. Balls are never written."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = p_local + scene.env_origins
        st[:, 3:7] = _qy(torch.full((n,), theta, device=device))
        scene.bowl.write_root_state_to_sim(st, all_ids)

    def glide(p0: torch.Tensor, p1: torch.Tensor, th0: float, th1: float,
              steps: int) -> None:
        """Carry the held bowl smoothly p0->p1, th0->th1 over `steps` physics steps
        (per-step increments stay small so the contents ride on real contacts)."""
        for s in range(1, steps + 1):
            f = s / steps
            hold_pose(p0 + (p1 - p0) * f, th0 + (th1 - th0) * f)
            env.step(no_action)

    def hold(theta: float, p_local: torch.Tensor, steps: int) -> None:
        for _ in range(steps):
            hold_pose(p_local, theta)
            env.step(no_action)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # balls seat inside the bowl
    pres = scene.present[0]
    k_present = int(pres.sum())
    side = int(scene.dish_side[0])
    dishc = (scene.dish.data.root_pos_w - scene.env_origins)[0].clone()
    trayc = (scene.tray.data.root_pos_w - scene.env_origins)[0].clone()
    bowlp = (scene.bowl.data.root_pos_w - scene.env_origins)[0].clone()
    print(f"[solve] layout readback (seed {args.seed}): dish_side={side:+d} "
          f"dish=({float(dishc[0]):+.3f},{float(dishc[1]):+.3f}) "
          f"tray=({float(trayc[0]):+.3f},{float(trayc[1]):+.3f}) "
          f"bowl=({float(bowlp[0]):+.3f},{float(bowlp[1]):+.3f}) "
          f"balls_present={k_present} mask={pres.tolist()}", flush=True)
    report("reset")
    pos, _v = scene._ball_tensors()
    assert torch.isfinite(pos).all(), "NaN/inf in ball states after settle"
    assert bool((scene.in_bowl()[0] | ~pres).all()), \
        "every present ball must start inside the bowl"
    assert int((scene.in_dish()[0] & pres).sum()) == 0, \
        "no present ball may start in the dish"
    s0 = print_score("P0 reset+settle (loaded bowl on the floor)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: TRANSPORT — lift the loaded bowl, carry it over the dish ----
    p_hold = dishc.clone()
    p_hold[0] -= 0.010
    p_hold[2] = c.pour_hold_z
    b0 = bowlp.clone()
    b_up = b0.clone()
    b_up[2] = c.pour_hold_z
    glide(b0.expand(n, 3), b_up.expand(n, 3), 0.0, 0.0, 80)       # lift straight up
    glide(b_up.expand(n, 3), p_hold.expand(n, 3), 0.0, 0.0, 140)  # carry over the dish
    report("carry")
    assert bool(scene._lift[0]), "lift latch did not set during the carry"
    assert bool((scene.in_bowl()[0] | ~pres).all()), \
        "a ball fell out of the bowl during transport"
    s1 = print_score("P1 loaded bowl held over the dish")
    assert s1 >= s0 - 1e-6 and s1 >= 0.09, f"P1 score {s1} (expect lift=0.10)"

    # ---------------- phase 2: POUR — gravity empties the bowl into the dish ---------------
    ph = p_hold.expand(n, 3)
    tilt = math.radians(135.0)
    glide(ph, ph, 0.0, tilt, 100)  # ramp the tip; balls roll out over the lip
    drained = False
    for cycle in range(6):
        hold(tilt, ph, 90)
        left = int((scene.in_bowl()[0] & pres).sum())
        if left == 0:
            drained = True
            break
        # wrist wiggle: swing the hold angle 150 -> 120 -> back (contact-consistent)
        print(f"[solve] {left} ball(s) hung up in the tilted bowl; wiggling",
              flush=True)
        glide(ph, ph, tilt, math.radians(150.0), 30)
        glide(ph, ph, math.radians(150.0), math.radians(120.0), 40)
        glide(ph, ph, math.radians(120.0), tilt, 30)
    if not drained:
        report("POUR-STUCK")
        print("SIM_GEN_SOLVE: FAIL (bowl never drained)", flush=True)
        os._exit(1)
    # keep holding while the balls settle into the dish; wait for the all-in latch
    ok_all = False
    for _ in range(12):
        hold(tilt, ph, 40)
        if bool(scene._mall[0]):
            ok_all = True
            break
    report("poured")
    if not ok_all:
        print("SIM_GEN_SOLVE: FAIL (poured balls never settled all-in-dish)",
              flush=True)
        os._exit(1)
    assert bool(scene._m1[0]), "first-ball latch did not set"
    s2 = print_score("P2 every present ball settled in the dish")
    assert s2 >= s1 - 1e-6 and s2 >= 0.54, f"P2 score {s2} (expect 0.55)"

    # ---------------- phase 3: PARK — carry inverted over the tray, release, settle --------
    t_over = trayc.clone()
    t_over[2] = 0.170
    glide(ph, t_over.expand(n, 3), tilt, math.pi, 160)  # invert while in transit
    t_rel = trayc.clone()
    t_rel[2] = c.tray_base_t + c.bowl_h + 0.008         # rim 8 mm above the pad
    glide(t_over.expand(n, 3), t_rel.expand(n, 3), math.pi, math.pi, 60)
    step(150)  # RELEASE: hands off — gravity seats the rim on the pad
    report("parked")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after parking the bowl)", flush=True)
        os._exit(1)
    assert bool(scene._park[0]), "park latch did not set"
    s3 = print_score("P3 bowl released rim-down inside the tray")
    assert s3 >= s2 - 1e-6, "score decreased across the park"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, hands-off) -------------
    holds = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        holds = holds and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
    ok = holds and bool(scene.success()[0]) and s4 >= s3 - 1e-6
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
    main()

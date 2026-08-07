"""Teleport solution for BayonetLidScene (sim_gen task
libero_kitchen_scene7_put_the_white_bowl_on_the_plate_i3) — the task's legitimacy
certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, small per-step pose increments): the red-knobbed lid is
   "held" by writing its root pose each physics step along a smooth path — lift,
   carry over the pot, rotate in FREE SPACE so its lugs line up with the collar's
   entry gaps (the pot's heading is read back: perception, then alignment). Holding
   a pose satisfies no rubric clause: the seat band and the locked window are both
   unreachable while hovering.
2. KEYED INSERTION (gravity + contact): the aligned lid is released ~12 mm above
   the rim and FALLS through the three entry gaps onto the internal ledge. The
   seating contact is made by gravity; if the drop bounces out of alignment, the
   lid is picked back up and re-dropped — its pose below the rim is NEVER written.
3. CLOCKWISE TWIST (applied torque + contact, the core interaction): with the lid
   seated, a small speed-governed torque about -z spins it clockwise; the lugs
   slide under the brass tabs until they HIT THE END STOPS — the locked engagement
   angle is produced by the stop contact, never written. The torque is then
   removed and the lid must stay locked at rest. The lid's pose is never written
   after the release: every fact success() checks (seat z, engagement angle,
   settledness) is the outcome of contact dynamics.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene7_put_the_white_bowl_on_the_plate_i3.solve --headless [--seed N]
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

_qz = scene_mod._qz
_yaw_of = scene_mod._yaw_of

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.lid_bayonet")().build(num_envs=args.num_envs, device=device)
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
        xy, z = scene._rel(scene.lid)
        e = float(torch.rad2deg(scene.engagement())[0])
        v = float(scene.lid.data.root_lin_vel_w.norm(dim=-1)[0])
        w = float(scene.lid.data.root_ang_vel_w.norm(dim=-1)[0])
        print(f"[solve] {tag:12s} | lid xy={float(xy[0]) * 1000:6.1f}mm "
              f"z={float(z[0]) * 1000:6.1f}mm delta={e:6.1f}deg v={v:.3f} w={w:.2f} "
              f"seated={bool(scene.seated()[0])} locked={bool(scene.locked()[0])} "
              f"lift={bool(scene._lift[0])} over={bool(scene._over[0])} "
              f"seat={bool(scene._seat[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def hold_pose(p_local: torch.Tensor, yaw: float) -> None:
        """One held-pose write: lid root at env-local `p_local`, upright with `yaw`,
        zero velocity."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = p_local + scene.env_origins
        st[:, 3:7] = _qz(torch.full((n,), yaw, device=device))
        scene.lid.write_root_state_to_sim(st, all_ids)

    def glide(p0: torch.Tensor, p1: torch.Tensor, y0: float, y1: float,
              steps: int) -> None:
        """Carry the held lid smoothly p0->p1, yaw y0->y1 over `steps` physics steps
        (per-step increments stay small: this is transport, not interaction)."""
        for s in range(1, steps + 1):
            f = s / steps
            hold_pose(p0 + (p1 - p0) * f, y0 + (y1 - y0) * f)
            env.step(no_action)

    zero3 = torch.zeros(n, 1, 3, device=device)

    def twist(tau: float, steps: int, w_max: float = 1.5) -> None:
        """Speed-governed clockwise torque about -z for `steps` steps, then clear."""
        for _ in range(steps):
            wz = scene.lid.data.root_ang_vel_w[:, 2]
            t = torch.zeros(n, 1, 3, device=device)
            t[:, 0, 2] = torch.where(wz > -w_max,
                                     torch.full_like(wz, -tau),
                                     torch.zeros_like(wz))
            scene.lid.set_external_force_and_torque(zero3, t, env_ids=all_ids,
                                                    is_global=True)
            env.step(no_action)
        scene.lid.set_external_force_and_torque(zero3, zero3, env_ids=all_ids)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)  # lids settle flat on the ground
    potc = (scene.pot.data.root_pos_w - scene.env_origins)[0].clone()
    pot_yaw = float(_yaw_of(scene.pot.data.root_quat_w)[0])
    lidp = (scene.lid.data.root_pos_w - scene.env_origins)[0].clone()
    lid_yaw0 = float(_yaw_of(scene.lid.data.root_quat_w)[0])
    side = int(scene.lid_side[0])
    print(f"[solve] layout readback (seed {args.seed}): "
          f"pot=({float(potc[0]):+.3f},{float(potc[1]):+.3f}) "
          f"pot_yaw={math.degrees(pot_yaw):+.1f}deg red_lid_side={side:+d} "
          f"lid=({float(lidp[0]):+.3f},{float(lidp[1]):+.3f}) "
          f"lid_yaw={math.degrees(lid_yaw0):+.1f}deg "
          f"delta={float(torch.rad2deg(scene.engagement())[0]):.1f}deg", flush=True)
    report("reset")
    assert torch.isfinite(scene.lid.data.root_pos_w).all(), "NaN/inf after settle"
    s0 = print_score("P0 reset+settle (both lids flat on the ground)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: TRANSPORT — lift the red lid, align, carry over the pot -----
    # Entry alignment: lid yaw such that delta = (lid_yaw - pot_yaw) mod 120° = 90°
    # (gap centers). Pick the 120°-equivalent target nearest the lid's current yaw.
    period = 2 * math.pi / 3
    raw = pot_yaw + math.radians(c.entry_deg)
    k = round((lid_yaw0 - raw) / period)
    yaw_entry = raw + k * period
    p0 = lidp.clone()
    p_up = lidp.clone()
    p_up[2] = 0.22
    hover = potc.clone()
    hover[2] = c.pot_collar_top + c.lid_disc_t + 0.012  # lid bottom 12 mm above rim
    high = hover.clone()
    high[2] = 0.22
    glide(p0.expand(n, 3), p_up.expand(n, 3), lid_yaw0, lid_yaw0, 80)
    glide(p_up.expand(n, 3), high.expand(n, 3), lid_yaw0, yaw_entry, 140)
    report("carry")
    assert bool(scene._lift[0]), "lift latch did not set during the carry"
    s1a = print_score("P1a red lid aligned high over the pot")
    glide(high.expand(n, 3), hover.expand(n, 3), yaw_entry, yaw_entry, 60)
    report("hover")
    assert bool(scene._over[0]), "over-collar latch did not set"
    s1 = print_score("P1 red lid hovering over the collar mouth, lugs on the gaps")
    assert s1 >= s1a - 1e-6 and s1 >= 0.24, f"P1 score {s1} (expect lift+over=0.25)"

    # ---------------- phase 2: KEYED INSERTION — release; gravity drops it through --------
    seated_ok = False
    for attempt in range(4):
        step(150)  # RELEASE: free fall through the gaps, seat on the ledge, settle
        report(f"drop{attempt}")
        if bool(scene.seated()[0]):
            seated_ok = True
            break
        # bounced/misaligned: pick it back up (transport only) and re-drop
        cur = (scene.lid.data.root_pos_w - scene.env_origins)[0].clone()
        cur_yaw = float(_yaw_of(scene.lid.data.root_quat_w)[0])
        k = round((cur_yaw - raw) / period)
        y2 = raw + k * period
        lift_p = cur.clone()
        lift_p[2] = 0.22
        print(f"[solve] drop attempt {attempt} missed the seat; retrying", flush=True)
        glide(cur.expand(n, 3), lift_p.expand(n, 3), cur_yaw, cur_yaw, 60)
        glide(lift_p.expand(n, 3), high.expand(n, 3), cur_yaw, y2, 80)
        glide(high.expand(n, 3), hover.expand(n, 3), y2, y2, 50)
    if not seated_ok:
        report("SEAT-FAIL")
        print("SIM_GEN_SOLVE: FAIL (lid never seated on the ledge)", flush=True)
        os._exit(1)
    assert bool(scene._seat[0]), "seat latch did not set"
    s2 = print_score("P2 red lid seated on the internal ledge (not yet locked)")
    assert s2 >= s1 - 1e-6 and s2 >= 0.74, f"P2 score {s2} (expect 0.75)"
    assert not bool(scene.success()[0]), "seated-but-unlocked must NOT be success"

    # ---------------- phase 3: CLOCKWISE TWIST — torque to the end stops ------------------
    tau = 0.12
    locked_ok = False
    for cycle in range(24):
        twist(tau, 25)
        step(15)  # coast + settle readout
        e = float(torch.rad2deg(scene.engagement())[0])
        if not bool(scene.seated()[0]):
            report("TWIST-UNSEAT")
            print("SIM_GEN_SOLVE: FAIL (lid popped out of the seat while twisting)",
                  flush=True)
            os._exit(1)
        if bool(scene.locked()[0]):
            locked_ok = True
            break
        if cycle in (7, 15):
            tau *= 1.6  # friction margin: escalate if the twist stalls
            print(f"[solve] twist stalled at delta={e:.1f}deg; "
                  f"raising torque to {tau:.2f} N.m", flush=True)
    if not locked_ok:
        report("TWIST-FAIL")
        print("SIM_GEN_SOLVE: FAIL (lugs never reached the locked window)", flush=True)
        os._exit(1)
    for _ in range(2):  # press the lugs firmly against the end stops
        twist(tau, 25)
        step(15)
    step(80)  # torque cleared inside twist(); let everything come to rest
    report("locked")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the twist settled)", flush=True)
        os._exit(1)
    s3 = print_score("P3 lugs against the end stops — lid locked")
    assert s3 >= s2 - 1e-6 and s3 >= 0.99, f"P3 score {s3} (expect 1.0)"

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

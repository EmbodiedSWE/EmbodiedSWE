"""Teleport solution for DrawerBeadPourScene (sim_gen task
`libero_kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_i134`)
— the task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. OPEN THE DRAWER (contact dynamics — never teleported): a bounded horizontal force
   on the drawer body (the applied-wrench emulation of the arm pulling the blue
   handle bar) servoes the drawer out along its real prismatic slide against the
   slide damping; the drive is then REMOVED and the drawer stays out on its own
   (springless slide). The drawer's pose is never written after reset.
2. CARRY (held-object transport): the bowl is moved along a slow smooth path — lift,
   traverse, hover over the exposed drawer mouth — by per-step pose writes with
   path-consistent velocities (the emulation of the arm holding the bowl rim). The
   BEADS are never written: they ride inside the bowl through real contacts, and the
   solve asserts they are all still aboard after the traverse.
3. POUR (contact dynamics): the held bowl is tilted about a horizontal axis over the
   open mouth; the beads roll over the rim, FALL through the mouth under gravity,
   land on the drawer floor and settle through real impacts. Stragglers get a gentle
   translational shake of the held bowl. No bead is ever teleported; nothing writes
   the drawer.
4. PARK (transport + release): the emptied bowl is carried back over the open floor,
   lowered to just above the ground, released (writes stop), and it settles upright
   under gravity. success() first turns True here, judged on settled physical state.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.<task>.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True  # never keep a dead interpreter alive waiting for the timer
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.drawer_bead_pour")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    dt = 1.0 / 120.0

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    all_ids = torch.arange(n, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def open_m() -> float:
        return float(scene.drawer_open()[0])

    def n_present() -> int:
        return int(scene.present[0].sum())

    def n_aboard() -> int:
        return int((scene.beads_in_bowl() & scene.present)[0].sum())

    def n_dep() -> int:
        return int((scene.beads_deposited() & scene.present)[0].sum())

    def report(tag: str) -> None:
        b = (scene.bowl.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] {tag:12s} | open={open_m() * 1000:5.1f}mm "
              f"bowl=({float(b[0]):+.3f},{float(b[1]):+.3f},{float(b[2]):.3f}) "
              f"aboard={n_aboard()}/{n_present()} dep={n_dep()}/{n_present()} "
              f"opened={bool(scene._opened[0])} carry={float(scene._carry_max[0]):.3f} "
              f"depmax={float(scene._dep_max[0]):.3f} park={bool(scene._park[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ----- held-bowl transport helpers (bowl pose written per step; beads never) -----
    def hold_write(pos, quat, lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor(pos, device=device) + scene.env_origins
        st[:, 3:7] = torch.tensor(quat, device=device)
        st[:, 7:10] = torch.tensor(lin_vel, device=device)
        st[:, 10:13] = torch.tensor(ang_vel, device=device)
        scene.bowl.write_root_state_to_sim(st, all_ids)

    def q_tilt(a: float) -> tuple:
        """Rotation about +y by angle a (rad): bowl up-axis tips toward -x for a < 0."""
        return (math.cos(a / 2), 0.0, math.sin(a / 2), 0.0)

    def bowl_pos() -> list:
        return [float(v) for v in (scene.bowl.data.root_pos_w - scene.env_origins)[0]]

    def carry_to(target, steps: int, tilt: float = 0.0) -> None:
        """Move the held bowl along a smoothstep path at constant tilt."""
        p0 = bowl_pos()
        for i in range(steps):
            s = (i + 1) / steps
            w = s * s * (3 - 2 * s)  # smoothstep
            dw = 6 * s * (1 - s) / steps  # d(w)/d(step)
            pos = [p0[j] + (target[j] - p0[j]) * w for j in range(3)]
            vel = [(target[j] - p0[j]) * dw / dt for j in range(3)]
            hold_write(pos, q_tilt(tilt), lin_vel=vel)
            env.step(no_action)

    def tilt_ramp(a0: float, a1: float, steps: int, center) -> None:
        """Tilt the held bowl in place about +y from a0 to a1 (rad)."""
        for i in range(steps):
            s = (i + 1) / steps
            w = s * s * (3 - 2 * s)
            a = a0 + (a1 - a0) * w
            omega = (a1 - a0) * 6 * s * (1 - s) / steps / dt
            hold_write(center, q_tilt(a), ang_vel=(0.0, omega, 0.0))
            env.step(no_action)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(90)  # beads settle into the bowl floor
    b0 = bowl_pos()
    print(f"[solve] layout readback (seed {args.seed}): bowl=({b0[0]:+.3f},{b0[1]:+.3f}) "
          f"beads_present={n_present()} aboard={n_aboard()} drawer_open={open_m() * 1000:.1f}mm",
          flush=True)
    report("reset")
    assert open_m() < 0.005, "drawer did not start shut"
    assert n_aboard() == n_present(), "beads did not settle inside the bowl"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"score not ~0 at reset ({s0:.3f})"

    # ---------------- phase 1: OPEN the drawer through the slide (contact dynamics) ---------
    target = c.travel - 0.006
    for _ in range(700):
        x = open_m()
        v = float(scene.drawer.data.root_lin_vel_w[0, 0])  # +x = closing
        # pull toward -x: force = -kp*(target - x) - kd*(-v)  (drive along +x axis)
        f = -60.0 * (target - x) + 8.0 * (-v)
        scene.drawer_drive[:] = max(-10.0, min(10.0, f))
        env.step(no_action)
        if x >= target - 0.004:
            break
    scene.drawer_drive[:] = 0.0
    step(60)  # drive off: the springless slide stays where it was left
    report("opened")
    assert open_m() >= c.open_min + 0.02, f"drawer did not stay open ({open_m() * 1000:.1f}mm)"
    assert bool(scene._opened[0]), "opened latch did not fire"
    s1 = print_score("P1 drawer pulled open by slide force, resting out")
    assert s1 >= s0 - 1e-6, "score decreased across opening"

    # ---------------- phase 2: CARRY the loaded bowl over the mouth (transport) -------------
    # Slow smoothstep segments; the beads ride inside through real contacts.
    hover = [float(c.front_x) - float(c.travel) + float(c.mouth_anchor_local[0]), 0.0, 0.25]
    carry_to([b0[0], b0[1], 0.20], 180)          # lift straight up
    carry_to([hover[0], hover[1], 0.25], 300)    # traverse to above the exposed mouth
    step_hold = 30
    for _ in range(step_hold):
        hold_write(hover, q_tilt(0.0))
        env.step(no_action)
    report("carried")
    assert n_aboard() == n_present(), "a bead was spilled during the carry"
    assert not bool(scene.bowl_in_drawer()[0]), "hover pose is inside the drawer volume"
    s2 = print_score("P2 loaded bowl carried to hover above the open mouth")
    assert s2 >= s1 - 1e-6, "score decreased across the carry"

    # ---------------- phase 3: POUR (gravity + contact; beads never written) ----------------
    poured = False
    tilt_now = 0.0
    for round_i in range(5):
        a1 = math.radians(-(115 + 5 * round_i))
        tilt_ramp(tilt_now, a1, 300, hover)
        tilt_now = a1
        # hold the tilt; let beads clear the rim and fall
        for _ in range(120):
            hold_write(hover, q_tilt(tilt_now))
            env.step(no_action)
        if n_aboard() == 0:
            poured = True
        else:
            # translational carrot shake along x (the spill direction) at the held tilt
            for cyc in range(4):
                for phase_k in range(40):
                    dx = 0.015 * math.sin(2 * math.pi * phase_k / 40)
                    hold_write([hover[0] + dx, hover[1], hover[2]], q_tilt(tilt_now))
                    env.step(no_action)
                if n_aboard() == 0:
                    break
            poured = n_aboard() == 0
        if poured:
            break
    # settle the fallen beads while still holding the bowl clear
    for _ in range(600):
        hold_write(hover, q_tilt(tilt_now))
        env.step(no_action)
        sp = float((scene._bead_speed() * scene.present.float())[0].max())
        if n_dep() == n_present() and sp < 0.04:
            break
    report("poured")
    assert poured and n_aboard() == 0, f"{n_aboard()} bead(s) still in the bowl after pouring"
    assert n_dep() == n_present(), "not every bead settled inside the drawer"
    s3 = print_score("P3 beads poured through the open mouth, settled on the drawer floor")
    assert s3 >= s2 - 1e-6, "score decreased across the pour"

    # ---------------- phase 4: PARK the empty bowl (transport + release) --------------------
    tilt_ramp(tilt_now, 0.0, 240, hover)  # un-tilt in place, still clear of the drawer
    park_xy = [b0[0] - 0.04, b0[1]]  # back to the (now empty) spawn side
    carry_to([hover[0], hover[1], 0.28], 60)
    carry_to([park_xy[0], park_xy[1], 0.20], 240)
    carry_to([park_xy[0], park_xy[1], c.bowl_h / 2 + 0.006], 180)
    hold_write([park_xy[0], park_xy[1], c.bowl_h / 2 + 0.004], q_tilt(0.0))
    # release: no further writes — the bowl settles under gravity
    step(150)
    report("parked")
    assert bool(scene.bowl_parked()[0]), "bowl did not settle parked upright on the floor"
    s4 = print_score("P4 bowl released, settled upright on the floor")
    assert s4 >= s3 - 1e-6, "score decreased across parking"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after parking)", flush=True)
        os._exit(1)

    # ---------------- phase 5: persistence (>= 3 simulated seconds, no intervention) --------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s5 = print_score("P5 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s5 >= s4 - 1e-6
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
    try:
        main()
    except BaseException:  # noqa: BLE001 - Kit threads would hang the interpreter
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)

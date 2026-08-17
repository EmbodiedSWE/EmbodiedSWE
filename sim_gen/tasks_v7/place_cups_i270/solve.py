"""Teleport solution for CargoShuttleScene (sim_gen task `place_cups_i270`) — the task's
legitimacy certificate.

Teleportation handles TRANSPORT ONLY: each canister is teleported (zero velocity,
shuttle-aligned) to ~20 mm above its color-matched socket floor — computed from the
LIVE shuttle pose readback — and released; gravity and the socket rims seat it through
contact. The shuttle itself is NEVER teleported: the whole delivery leg is executed
through contact dynamics — a velocity-servo'd horizontal external force pushes the
loaded shuttle down the channel (guide walls steer it, floor friction resists it),
under the bay canopy, until it presses against the end stop; the force is then cut and
the dock state is whatever friction and the hard stop hold. Nothing is ever teleported
INTO the rubric state: `seated` demands socket-floor contact height (only gravity puts
it there), `docked` demands the shuttle body physically past the dock line (only the
push puts it there), and `settled` demands the assembly comes to rest on its own.

PLAN (from describe()): the sockets are color-matched (RED rear, GREEN middle, BLUE
front) and the bay roof clears a seated canister by ~18 mm — so load first, ship
second. Seat red, green, blue while the shuttle sits in the open loading zone, then
push it by force (the arm would push the rear handle) until it docks.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.place_cups_i270.solve --headless [--seed N]
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
_wd.daemon = True
_wd.start()

_DT = 1.0 / 120.0


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cargo_shuttle")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sx() -> float:
        return float(scene.shuttle_x()[0])

    def report(tag: str) -> None:
        bits = []
        for nm in c.cup_names:
            loc = scene._cup_local(nm)[0]
            bits.append(f"{nm}=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
                        f"{float(loc[2]):+.3f})s in={bool(scene.seated(nm)[0])}")
        print(f"[solve] {tag:14s} | shuttle_x={sx():+.3f} docked={bool(scene.docked()[0])}"
              f" | " + " | ".join(bits)
              + f" | success={bool(scene.success()[0])} "
                f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def settle(max_steps: int = 600) -> None:
        for _ in range(max_steps // 30):
            step(30)
            if bool(scene.settled()[0]):
                break

    def seat_cup(nm: str) -> None:
        """TRANSPORT: teleport canister `nm` (zero velocity, shuttle-aligned) to 20 mm
        above its color-matched socket floor, computed from the LIVE shuttle pose —
        then hands off: gravity + the socket rims seat it through contact."""
        from isaaclab.utils.math import quat_apply

        i = c.cup_names.index(nm)
        sp = scene.shuttle.data.root_pos_w[0]
        sq = scene.shuttle.data.root_quat_w[0]
        local = torch.tensor([c.socket_xs[i], 0.0, c.seat_rest_z + 0.020], device=device)
        world = sp + quat_apply(sq.unsqueeze(0), local.unsqueeze(0))[0]
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = world
        st[:, 3:7] = sq  # shuttle-aligned: drops straight down the pocket
        scene.cups[nm].write_root_state_to_sim(st, all_ids)
        print(f"[solve] transport {nm} -> {nm} socket (release 20 mm above the seat, "
              f"shuttle at x={sx():+.3f})", flush=True)
        settle()

    def clear_wrench() -> None:
        scene.shuttle.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                    env_ids=all_ids)

    def push_to_dock() -> bool:
        """The delivery leg, pure contact dynamics: a velocity-servo'd horizontal
        force on the shuttle (the arm's stand-in at the rear handle) drives it down
        the channel to the end stop. Gain (not cap) escalates on a stall; the servo
        slows on final approach so the wall impact is gentle."""
        kv = 40.0  # N per (m/s) velocity error
        f_max = 12.0
        target = c.dock_rest_x - 0.004  # just short of the hard-stop rest
        stall_i, last_x = 0, sx()
        for i in range(3600):  # up to 30 s
            x = sx()
            if x >= target:
                break
            v = float(scene.shuttle.data.root_lin_vel_w[0, 0])
            v_des = 0.18 if x < c.dock_x - 0.12 else 0.06
            f = max(0.0, min(f_max, kv * (v_des - v)))
            fw = torch.zeros(n, 1, 3, device=device)
            fw[0, 0, 0] = f
            # shuttle yaw is pinned by the guide walls (quat ~ identity), so world
            # and body frames coincide — no frame ambiguity for this wrench
            scene.shuttle.set_external_force_and_torque(fw, zero_wrench,
                                                        env_ids=all_ids, is_global=True)
            env.step(no_action)
            if i % 120 == 119:
                prog = sx() - last_x
                last_x = sx()
                print(f"[solve] push: x={sx():+.3f} v={v:+.3f} f={f:.1f} "
                      f"prog={prog:+.3f}/s", flush=True)
                if prog < 0.005:
                    stall_i += 1
                    kv *= 1.5  # escalate GAIN, not the cap
                    print(f"[solve] push stall #{stall_i}: gain -> {kv:.0f}", flush=True)
                    if stall_i >= 6:
                        clear_wrench()
                        return False
        # brief light press to seat against the end stop, then hands off
        fw = torch.zeros(n, 1, 3, device=device)
        fw[0, 0, 0] = 3.0
        scene.shuttle.set_external_force_and_torque(fw, zero_wrench,
                                                    env_ids=all_ids, is_global=True)
        step(30)
        clear_wrench()
        settle()
        return True

    # ---------------- phase 0: reset, settle, plan readback --------------------------------
    step(90)
    m = scene.cup_mass[0]
    print(f"[solve] mass readback (seed {args.seed}): "
          + " ".join(f"{nm}={float(m[i]):.4f}" for i, nm in enumerate(c.cup_names)),
          flush=True)
    for nm in c.cup_names:
        p = (scene.cups[nm].data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] layout readback: {nm} at ({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):+.3f})", flush=True)
    print(f"[solve] layout readback: shuttle at x={sx():+.3f} "
          f"(dock threshold {c.dock_x:+.3f}, hard-stop rest {c.dock_rest_x:+.3f})",
          flush=True)
    report("reset")
    assert sx() < c.dock_x - 0.3, "shuttle must start in the open loading zone"
    s0 = print_score("P0 reset+settle")
    assert s0 < 0.05, "score must start ~0"

    # ---------------- phases 1-3: seat each canister in its matched socket -----------------
    scores = [s0]
    for pi, nm in enumerate(c.cup_names, start=1):
        seat_cup(nm)
        report(f"P{pi}-{nm}")
        assert bool(scene.seated(nm)[0]), f"{nm} must seat in its socket after the drop"
        s = print_score(f"P{pi} {nm} seated by gravity (contact)")
        assert s >= scores[-1] - 1e-6, f"score decreased across P{pi}"
        scores.append(s)
    assert bool(scene.all_seated()[0]), "all three must ride their sockets before shipping"

    # ---------------- phase 4: the delivery push (pure contact dynamics) -------------------
    ok_push = push_to_dock()
    report("P4-push")
    if not ok_push or not bool(scene.docked()[0]):
        print("SIM_GEN_SOLVE: FAIL (shuttle never docked)", flush=True)
        os._exit(1)
    assert bool(scene.all_seated()[0]), "cargo must still ride its sockets after the push"
    s4 = print_score("P4 shuttle docked against the end stop (force push)")
    assert s4 >= scores[-1] - 1e-6, "score decreased across P4"

    # wait for success() to hold 120 consecutive steps (settle counter + any ring-down)
    consec = 0
    for _ in range(1200):
        step(1)
        if bool(scene.success()[0]):
            consec += 1
            if consec >= 120:
                break
        else:
            consec = 0
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after dock + settle)", flush=True)
        os._exit(1)
    s5 = print_score("P5 success held 1 s")
    assert s5 >= s4 - 1e-6, "score decreased across P5"

    # ---------------- phase 6: persistence (>= 3 simulated seconds, no intervention) -------
    hold, flickers = True, 0
    for i in range(400):  # 400 steps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:
                sv = float(scene.shuttle.data.root_lin_vel_w[0].norm())
                print(f"[solve] persist flicker @step {i}: x={sx():+.3f} |v|={sv:.4f} "
                      f"seated={bool(scene.all_seated()[0])} "
                      f"docked={bool(scene.docked()[0])} "
                      f"settled={bool(scene.settled()[0])}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s6 = print_score("P6 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s6 >= s5 - 1e-6
    if ok:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)

    code = 0 if ok else 1
    # Hard exit: Kit teardown hangs — watchdog then die.
    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)

"""Teleport solution for ButterCellarScene (sim_gen task
`libero_kitchen_scene10_..._i307`) — the task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): one pose write carries the yellow butter from its floor spawn
   slot across free space to a hover 25 mm ABOVE the tray load point. It then FALLS
   under gravity and settles on the tray through real contact — the teleport endpoint
   is deliberately outside the on-tray scoring band (asserted), so nothing latches at
   the write instant.
2. PRESS (contact dynamics — the core interaction): one pose write carries the iron
   ingot to a hover 20 mm above the plunger's socket. It drops in, and from there
   EVERYTHING is gravity: the ingot's 5.4 N overwhelms the spring's ~1.1 N surplus and
   drives the platform down its real prismatic slide; the descending tray edge CAMS
   the brass pawl down into its wall channel by direct contact (the pawl is never
   pose-written — its max fold angle during the pass is asserted), passes below it,
   and the pawl's return spring re-seats it horizontal. No force is applied to any
   body by this script; the scene's own spring/damper plant is the only non-contact
   force, and it acts identically in every module.
3. RELEASE + LATCH (contact dynamics): one pose write lifts the ingot out of the
   socket and returns it to open floor (transport of a free body out of a resting
   contact — the same thing a pick does). The spring throws the platform back up and
   the tray's edge JAMS against the pawl's underside at the 0-deg joint stop: the
   latch engagement is pure contact. Asserted: the platform rises OFF the bottom stop
   (> 0.12 m) yet is held BELOW the latch ceiling (< 0.152 m) — only the pawl can do
   that — with the butter still riding the tray. success() first turns True here,
   judged on settled poses.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

The intended single-Franka-arm strategy for the same plan (top-down pinch of the 32 mm
butter faces, drop onto the tray basin; pick-place of the 45 mm ingot into / out of
the always-above-rim socket — no finger ever enters the bore) lives in TASK.md as the
embodiment argument.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it_i307.solve --headless [--seed N]
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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.butter_cellar")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def pawl_deg() -> float:
        return math.degrees(float(scene.pawl_angle()[0]))

    def plat_z() -> float:
        return float(scene.plat_z()[0])

    def report(tag: str) -> None:
        b = (scene.butter.data.root_pos_w - scene.env_origins)[0]
        g = (scene.ingot.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] {tag:12s} | butter=({float(b[0]):+.3f},{float(b[1]):+.3f},"
              f"{float(b[2]):.3f}) ingot=({float(g[0]):+.3f},{float(g[1]):+.3f},"
              f"{float(g[2]):.3f}) plat_z={plat_z():.3f} pawl={pawl_deg():+.1f}deg "
              f"|vb|={float(scene.butter.data.root_lin_vel_w[0].norm()):.4f} "
              f"|vp|={float(scene.platform.data.root_lin_vel_w[0].norm()):.4f} "
              f"in_band={bool(scene.in_latch_band()[0])} "
              f"seated={bool(scene.pawl_seated()[0])} "
              f"ing_clr={bool(scene.ingot_clear()[0])} "
              f"lard_in={bool(scene.in_bore(scene.lard)[0])} "
              f"on_tray={bool(scene.on_tray(scene.butter)[0])} "
              f"app={float(scene._app_max[0]):.3f} loaded={bool(scene._loaded[0])} "
              f"min_z={float(scene._min_z[0]):.3f} latched={bool(scene._latched[0])} "
              f"cleared={bool(scene._cleared[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def place(body, x: float, y: float, z: float) -> None:
        """One pose write (transport across free space): identity quat, zero velocity."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    b0 = (scene.butter.data.root_pos_w - scene.env_origins)[0]
    l0 = (scene.lard.data.root_pos_w - scene.env_origins)[0]
    g0 = (scene.ingot.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): butter=({float(b0[0]):+.3f},"
          f"{float(b0[1]):+.3f}) lard=({float(l0[0]):+.3f},{float(l0[1]):+.3f}) "
          f"ingot=({float(g0[0]):+.3f},{float(g0[1]):+.3f}) plat_z={plat_z():.3f} "
          f"pawl={pawl_deg():+.1f}deg", flush=True)
    report("reset")
    assert abs(plat_z() - c.z_top) < 0.010, "platform not resting at its top stop"
    assert abs(pawl_deg()) <= 3.0, "pawl not seated horizontal at reset"
    assert not bool(scene.on_tray(scene.butter)[0]), "butter spawned on the tray"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"baseline score not ~0 ({s0:.3f})"

    # ---------------- phase 1: TRANSPORT butter -> drop onto the raised tray ----------------
    # One pose write to a hover 50 mm above the tray load point (asserted: not yet in
    # the on-tray band), then gravity lands it on the tray through real contact.
    cx, cy = c.cellar_pos
    lp = scene._load_point_world()[0] - scene.env_origins[0]
    place(scene.butter, float(lp[0]), float(lp[1]), float(lp[2]) + 0.050)
    assert not bool(scene.on_tray(scene.butter)[0]), \
        "hover endpoint already inside the on-tray band"
    step(120)
    report("loaded")
    assert bool(scene.on_tray(scene.butter)[0]), "butter did not settle on the tray"
    assert bool(scene._loaded[0]), "loaded latch not set"
    assert abs(plat_z() - c.z_top) < 0.012, \
        "platform sagged under the butter (spring surplus miscalibrated)"
    s1 = print_score("P1 butter dropped onto the tray (contact landing)")
    assert s1 >= s0 - 1e-6, "score decreased across loading"

    # ---------------- phase 2: PRESS — ingot into the socket, gravity drives the slide ------
    # One pose write to a hover 20 mm above the socket floor, then hands off: the drop,
    # the spring compression, the pawl cam-aside (contact on the real hinge) and the
    # ride to the bottom stop are all pure contact dynamics.
    sf = float(scene.platform.data.root_pos_w[0, 2] - scene.env_origins[0, 2]) \
        + c.socket_floor_dz
    place(scene.ingot, cx + c.post_xy[0], cy + c.post_xy[1],
          sf + c.ingot_size[2] / 2 + 0.020)
    max_fold = 0.0
    for i in range(900):
        env.step(no_action)
        max_fold = max(max_fold, -pawl_deg())
        if plat_z() < c.latch_z_lo + 0.012 and \
                float(scene.platform.data.root_lin_vel_w[0].norm()) < 0.02:
            break
    step(30)  # let the pawl's return spring finish re-seating it
    report("pressed")
    print(f"[solve] max pawl fold during the pass: {max_fold:.1f} deg", flush=True)
    assert plat_z() < 0.105, f"platform did not reach the bottom of the stroke ({plat_z():.3f})"
    assert max_fold >= 45.0, \
        "pawl never cammed aside — the tray cannot have passed it by contact"
    assert abs(pawl_deg()) <= c.pawl_closed_deg, "pawl did not re-seat after the pass"
    assert bool(scene.on_tray(scene.butter)[0]), "butter fell off the tray during the press"
    s2 = print_score("P2 ingot press: spring overcome, pawl cammed aside by contact")
    assert s2 >= s1 - 1e-6, "score decreased across the press"

    # ---------------- phase 3: REMOVE the ingot -> spring vs pawl, the ratchet holds --------
    # One pose write lifts the ingot out of the socket and returns it to open floor.
    # Hands off: the spring throws the platform up and the tray edge jams against the
    # pawl underside at its 0-deg stop — the latch is pure contact.
    place(scene.ingot, 0.15, -0.32, c.ingot_size[2] / 2 + 0.002)
    step(240)
    report("latched")
    assert bool(scene.ingot_clear()[0]), "ingot not clear of the silo after removal"
    assert plat_z() > 0.120, \
        f"platform never rose off the bottom stop ({plat_z():.3f}) — spring dead?"
    assert plat_z() < 0.152, \
        f"platform escaped past the pawl ({plat_z():.3f}) — the ratchet failed to hold"
    assert abs(pawl_deg()) <= c.pawl_closed_deg, "pawl not seated under load"
    assert bool(scene.on_tray(scene.butter)[0]), "butter not on the tray after the latch"
    s3 = print_score("P3 ingot removed: spring-vs-pawl latch engaged (contact)")
    assert s3 >= s2 - 1e-6, "score decreased across the latch"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the latch)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3 simulated seconds, no intervention) -------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s4 >= s3 - 1e-6
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
    except BaseException as exc:  # noqa: BLE001 — die loudly, never hang until the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(exc).__name__}: {exc})", flush=True)
        os._exit(1)

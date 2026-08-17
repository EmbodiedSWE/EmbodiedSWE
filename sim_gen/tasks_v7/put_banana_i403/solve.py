"""Teleport solution for FragilePackScene (sim_gen task `put_banana_i403`) — the task's
legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. INSTALL (contact-respecting kinematic carry + transport teleport): the cradle base is
   pose-HELD each step (the kinematic-hold emulation of a fin grasp) and raised straight
   up; its sprung tray is NOT written — it rides the real prismatic suspension the whole
   way. One transport teleport (both bodies of the jointed pair written together,
   consistently) parks the cradle hovering over the crate centre; a pose-hold lower at
   6 cm/s takes it down between the walls to 5 mm above the crate floor, then RELEASE:
   gravity seats it. Seating alone is worth 0.25 latched credit — success still needs
   the intact orb on the tray.
2. PACK (the core interaction — a real compliant catch): the orb (with its concentric
   core: the pair is always written together so the brittle weld is never loaded by the
   transport) is teleported to a hover above the tray, pose-hold lowered to 15 mm above
   its riding height — asserted NOT yet orb_on_tray_now (outside the 12 mm z tolerance)
   — then RELEASED: it free-falls the last stretch onto the tray and the SPRING
   SUSPENSION arrests it through contact + drive dynamics. The same release onto any
   hard surface snaps the weld (the smoke battery pins that outcome); the entire
   fragility margin is physics, not bookkeeping. success() first turns True here.
3. The core weld is never touched: intact_now() stays True throughout, verified at
   every phase boundary.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.put_banana_i403.solve --headless [--seed N]
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

DT = 1.0 / 120.0
G_DT = 9.81 * DT  # gravity-compensated kinematic hold: write vz=+g*dt so net vz ~ 0


def main() -> None:
    from isaaclab.utils.math import quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.fragile_pack")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the statement on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        o = (scene.orb.data.root_pos_w - scene.env_origins)[0]
        k = (scene.cradle.data.root_pos_w - scene.env_origins)[0]
        t = (scene.tray.data.root_pos_w - scene.env_origins)[0]
        kl = scene._local(scene.crate, scene.cradle.data.root_pos_w)[0]
        print(f"[solve] {tag:12s} | orb=({float(o[0]):+.3f},{float(o[1]):+.3f},"
              f"{float(o[2]):.3f}) cradle=({float(k[0]):+.3f},{float(k[1]):+.3f},"
              f"{float(k[2]):.3f}) cradle_loc=({float(kl[0]):+.3f},{float(kl[1]):+.3f}) "
              f"tray_z={float(t[2]):.3f} core_off={float(scene.core_offset()[0]) * 1000:.1f}mm "
              f"intact={bool(scene.intact_now()[0])} "
              f"seated={bool(scene.cradle_seated_now()[0])} "
              f"on_tray={bool(scene.orb_on_tray_now()[0])} "
              f"L(seat={bool(scene._seated[0])},app={float(scene._app_max[0]):.3f},"
              f"pack={bool(scene._packed[0])},brk={bool(scene._broken[0])}) "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def hold(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
        """One kinematic-hold write: pose imposed, vz=+g*dt cancels the gravity kick."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3:7] = quat
        st[:, 9] = G_DT
        body.write_root_state_to_sim(st, all_ids)

    def write_pose(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
        """One zero-velocity pose write (release / transport)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3:7] = quat
        body.write_root_state_to_sim(st, all_ids)

    def tray_offset_w(quat: torch.Tensor) -> torch.Tensor:
        """World offset cradle-origin -> tray-origin at joint rest, for a cradle quat."""
        loc = torch.tensor([c.tray_rest_lx, 0.0, c.tray_rest_lz], device=device)
        return quat_apply(quat, loc.expand(n, 3))

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    step(60)
    cr = (scene.crate.data.root_pos_w - scene.env_origins)[0]
    cq = scene.crate.data.root_quat_w[0]
    cyaw = 2.0 * math.atan2(float(cq[3]), float(cq[0]))
    k0 = (scene.cradle.data.root_pos_w - scene.env_origins)[0]
    o0 = (scene.orb.data.root_pos_w - scene.env_origins)[0]
    m_cradle = float(scene.cradle.root_physx_view.get_masses().reshape(-1)[0])
    m_tray = float(scene.tray.root_physx_view.get_masses().reshape(-1)[0])
    m_orb = float(scene.orb.root_physx_view.get_masses().reshape(-1)[0])
    m_core = float(scene.core.root_physx_view.get_masses().reshape(-1)[0])
    print(f"[solve] layout readback (seed {args.seed}): crate=({float(cr[0]):+.3f},"
          f"{float(cr[1]):+.3f}) yaw={math.degrees(cyaw):+.1f}deg "
          f"cradle=({float(k0[0]):+.3f},{float(k0[1]):+.3f}) "
          f"orb=({float(o0[0]):+.3f},{float(o0[1]):+.3f}) "
          f"masses: cradle={m_cradle:.3f} tray={m_tray:.3f} orb={m_orb:.3f} "
          f"core={m_core:.3f}", flush=True)
    report("reset")
    assert 0.10 < m_cradle < 0.30, "cradle MassAPI not applied (density mass?)"
    assert 0.010 < m_tray < 0.035, "tray MassAPI not applied"
    assert 0.030 < m_orb < 0.075, "orb MassAPI not applied"
    assert 0.010 < m_core < 0.035, "core MassAPI not applied"
    assert bool(scene.intact_now()[0]), "orb must start intact"
    assert not bool(scene.cradle_seated_now()[0]), "cradle must start outside the crate"
    assert not bool(scene.orb_on_tray_now()[0]), "orb must start off the tray"
    s0 = print_score("P0 reset+settle (all loose on open ground, orb intact)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: INSTALL — carry the cradle into the crate -------------------
    # Pose-hold the cradle base (fin-grasp emulation) and raise it straight up at
    # 0.07 m/s; the sprung tray is never written — it rides its real suspension.
    base_pos = scene.cradle.data.root_pos_w.clone()
    base_quat = scene.cradle.data.root_quat_w.clone()
    lift = 0.15
    steps_up = int(lift / 0.07 / DT)
    for i in range(steps_up):
        p = base_pos.clone()
        p[:, 2] += lift * (i + 1) / steps_up
        hold(scene.cradle, p, base_quat)
        env.step(no_action)
    report("lifted")
    assert bool(scene.intact_now()[0]), "orb broke during the cradle lift?!"

    # Transport teleport to a hover over the crate centre: BOTH bodies of the jointed
    # pair written together, consistently (joint at rest), in the same step.
    hover_z = 0.12  # slab bottom 40 mm above the 80 mm wall
    hover = scene.crate.data.root_pos_w.clone()
    hover[:, 2] = hover_z
    write_pose(scene.cradle, hover, base_quat)
    write_pose(scene.tray, hover + tray_offset_w(base_quat), base_quat)
    step(10)
    report("hover")

    # Pose-hold lower between the walls to 5 mm above the crate floor, then release:
    # gravity seats the slab on the ground through real contact.
    lower = hover_z - 0.005
    steps_dn = int(lower / 0.06 / DT)
    for i in range(steps_dn):
        p = hover.clone()
        p[:, 2] -= lower * (i + 1) / steps_dn
        hold(scene.cradle, p, base_quat)
        env.step(no_action)
    p = hover.clone()
    p[:, 2] -= lower
    write_pose(scene.cradle, p, base_quat)
    step(90)  # drop 5 mm, tray suspension rings down, settle
    report("seated")
    assert bool(scene.cradle_seated_now()[0]), "cradle did not seat in the crate"
    assert bool(scene.intact_now()[0]), "orb broke while it was never touched?!"
    s1 = print_score("P1 cradle lowered into the crate and gravity-seated")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_seat - 1e-6, f"P1 score {s1}"

    # ---------------- phase 2: PACK — compliant catch on the sprung tray -------------------
    # Transport teleport: orb + core (concentric pair, written together — the brittle
    # weld sees zero relative motion) to a hover above the tray.
    tq = scene.orb.data.root_quat_w.clone()
    hover_o = scene.tray.data.root_pos_w.clone()
    hover_o[:, 2] += c.plate_t / 2 + c.orb_r + 0.05
    write_pose(scene.orb, hover_o, tq)
    write_pose(scene.core, hover_o, tq)
    step(2)
    sA = print_score("P2a orb staged over the tray (approach latched)")
    assert sA >= s1 - 1e-6, f"P2a score {sA}"

    # Kinematic lower (both bodies held together) to 15 mm above riding height.
    ride_z = scene.tray.data.root_pos_w[:, 2] + c.orb_tray_lz  # tray frame is upright
    release_z = ride_z + 0.015
    drop_to = float((hover_o[:, 2] - release_z)[0])
    steps_dn = int(drop_to / 0.05 / DT)
    for i in range(steps_dn):
        p = hover_o.clone()
        p[:, 2] -= drop_to * (i + 1) / steps_dn
        hold(scene.orb, p, tq)
        hold(scene.core, p, tq)
        env.step(no_action)
    report("pre-release")
    assert not bool(scene.orb_on_tray_now()[0]), \
        "held orb already satisfies the on-tray clause (release band too low)"
    assert bool(scene.intact_now()[0]), "orb broke during the held lower?!"
    # Release: one zero-velocity write at the held pose, then HANDS OFF — the last
    # 15 mm is free fall onto the tray and the spring suspension arrests it.
    p = hover_o.clone()
    p[:, 2] -= drop_to
    write_pose(scene.orb, p, tq)
    write_pose(scene.core, p, tq)
    step(150)  # 1.25 s: catch, suspension rings down, settle
    report("packed")
    assert bool(scene.intact_now()[0]), \
        "the tray catch broke the orb (suspension miscalibrated)"
    assert bool(scene.orb_on_tray_now()[0]), "orb did not settle on the tray"
    s2 = print_score("P2 orb released onto the sprung tray, caught intact")
    assert s2 >= sA - 1e-6, f"P2 score {s2}"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the pack)", flush=True)
        os._exit(1)

    # ---------------- phase 3: persistence (>= 3 simulated seconds, no intervention) -------
    hold_ok = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold_ok = hold_ok and bool(scene.success()[0])
    report("persist")
    s3 = print_score("P3 persistence 3.3 s")
    ok = hold_ok and bool(scene.success()[0]) and s3 >= s2 - 1e-6
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

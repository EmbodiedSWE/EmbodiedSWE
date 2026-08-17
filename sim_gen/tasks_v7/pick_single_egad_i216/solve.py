"""Teleport solution for GaugeSortScene (sim_gen task `pick_single_egad_i216`) —
the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY: each bar is teleported from its floor
scatter spot to a pose a pick-and-carry would deliver — thick bars to just above
the tray floor (then released to settle), fitting bars to a nose-down hover with
the bar's tip a few millimetres ABOVE the roof plates, centered on the slot —
and RELEASED with zero velocity. The load-bearing interaction, the aperture
transit, happens entirely under contact dynamics: the released bar free-falls
into the slot channel, threads the gauge gap (>= 3 mm side clearance at the
tightest class, brushing the plate edges), drops to the bin floor, topples and
settles INSIDE — the bin is closed everywhere else, so the rubric's in-bin
readout can only be reached through the gauge. No bar is ever written below the
sill or inside a destination volume; if a drop bounces astray on the plate edge
the bar is picked back up (teleport above the slot again — exactly a re-grasp)
and re-dropped, once with a small guided downward velocity.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: routed
bars rest in walled destinations and stay put), then holds HANDS-OFF for >= 3
simulated seconds after success() first turns True and prints
`SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.pick_single_egad_i216.solve --headless [--seed N]
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
    import scene as scene_mod

_PL = scene_mod._PL
_BIN_C = scene_mod._BIN_C
_TRAY_C = scene_mod._TRAY_C
_TRAY_FT = scene_mod._TRAY_FT
_PLATE_Z = scene_mod._PLATE_Z
_PLATE_T = scene_mod._PLATE_T

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.gauge_sort")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    env.reset(seed=args.seed)  # seed AFTER build
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def station_pose(local, world_yaw_off: float = 0.0, flat: bool = False):
        """(pos, quat) world pose for a station-frame `local` position; quat =
        station yaw + offset, either upright (long axis vertical) or lying flat."""
        pos = scene.env_origins.clone()
        pos[:, 0:2] += scene.anchor
        pos += scene.station_dir([local[0], local[1], 0.0])
        pos[:, 2] = local[2]
        u = scene.yaw + world_yaw_off
        ch, sh = torch.cos(u / 2), torch.sin(u / 2)
        z = torch.zeros_like(ch)
        if flat:
            c45 = math.sqrt(0.5)
            quat = torch.stack([ch * c45, ch * c45, sh * c45, sh * c45], dim=-1)
        else:
            quat = torch.stack([ch, z, z, sh], dim=-1)
        return pos, quat

    def teleport(body, pos, quat, vel_z: float = 0.0) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3:7] = quat
        st[:, 9] = vel_z
        body.write_root_state_to_sim(st, all_ids)

    def report(tag: str) -> None:
        loc = scene.bar_local()[0]
        f = scene.fits()[0]
        ib, it = scene.in_bin()[0], scene.in_tray()[0]
        rows = " | ".join(
            f"{name}: loc=({float(loc[i, 0]):+.3f},{float(loc[i, 1]):+.3f},"
            f"{float(loc[i, 2]):+.3f}) fit={bool(f[i])} bin={bool(ib[i])} "
            f"tray={bool(it[i])}"
            for i, (name, _t, _w) in enumerate(c.parts))
        print(f"[solve] {tag:12s} | W={float(scene.W[0]) * 1000:.1f}mm | {rows}",
              flush=True)
        print(f"[solve] {tag:12s} | correct={scene.correct()[0].tolist()} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, read the episode -----------------------------
    step(120)
    report("reset")
    wgap = float(scene.W[0])
    fits = scene.fits()[0].tolist()
    print(f"[solve] episode readback (seed {args.seed}): gauge gap {wgap * 1000:.1f} mm, "
          f"fits={fits}, perm={scene.slot_perm[0].tolist()}, "
          f"anchor=({float(scene.anchor[0, 0]):+.3f},{float(scene.anchor[0, 1]):+.3f}), "
          f"yaw={math.degrees(float(scene.yaw[0])):+.1f} deg", flush=True)
    assert any(fits) and not all(fits), "episode must have both classes present"
    s_prev = print_score("P0 reset+settle (all bars loose on the floor)")

    # ---------------- phase 1: thick bars -> reject tray (transport + release) --------------
    tray_offs = [-0.068, 0.0, 0.068]
    n_tray = 0
    for i, (name, t, _w) in enumerate(c.parts):
        if fits[i]:
            continue
        yoff = tray_offs[n_tray]
        n_tray += 1
        # lying flat, length along station x, 15 mm above the tray floor
        pos, quat = station_pose(
            [_TRAY_C[0], _TRAY_C[1] + yoff, _TRAY_FT + t / 2 + 0.015],
            world_yaw_off=math.pi / 2, flat=True)
        teleport(scene.bars[i], pos, quat)
        step(90)
        report(f"tray:{name}")
        assert bool(scene.in_tray()[0, i]), f"{name} must rest in the tray"
        s_now = print_score(f"P1 thick bar {name} laid in the reject tray")
        assert s_now >= s_prev - 1e-6
        s_prev = s_now

    # ---------------- phase 2: fitting bars -> through the gauge slot ------------------------
    drop_xs = [0.06, 0.0, -0.06]  # offsets from the bin center along the channel
    #   so bars don't pile up (inner half-length 0.10, widest fitting bar 56 mm)
    n_bin = 0
    for i, (name, t, _w) in enumerate(c.parts):
        if not fits[i]:
            continue
        xoff = drop_xs[n_bin]
        n_bin += 1
        done = False
        for attempt in range(4):
            # nose-down hover: bar tip 5 mm ABOVE the roof plates, thin side across
            # the gap; released with zero (then slightly guided) velocity — the
            # transit itself is pure contact dynamics through the gauge
            pos, quat = station_pose(
                [_BIN_C[0] + xoff, _BIN_C[1], _PLATE_Z + _PLATE_T / 2 + _PL / 2 + 0.005])
            teleport(scene.bars[i], pos, quat, vel_z=-0.4 if attempt else 0.0)
            for _ in range(8):
                step(30)
                if bool(scene.in_bin()[0, i]) and bool(scene.settled()[0, i]):
                    break
            if bool(scene.in_bin()[0, i]):
                done = True
                break
            print(f"[solve] {name} drop attempt {attempt} did not seat "
                  f"(z={float(scene.bar_local()[0, i, 2]):.3f}) — re-grasping",
                  flush=True)
        report(f"slot:{name}")
        assert done, f"{name} failed to pass the gauge in 4 attempts"
        s_now = print_score(f"P2 fitting bar {name} dropped through the gauge slot")
        assert s_now >= s_prev - 1e-6
        s_prev = s_now

    # ---------------- phase 3: settle, verify, hands-off persistence ------------------------
    for _ in range(12):
        step(30)
        if bool(scene.success()[0]):
            break
    report("sorted")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after sorting+settle)", flush=True)
        os._exit(1)
    s_now = print_score("P3 all four bars settled in their gauge-correct destinations")
    assert s_now >= s_prev - 1e-6
    s_prev = s_now

    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz, no intervention
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:
                loc = scene.bar_local()[0]
                print(f"[solve] persist flicker @step {i}: correct="
                      f"{scene.correct()[0].tolist()} z={[round(float(v), 3) for v in loc[:, 2]]}",
                      flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s4 >= s_prev - 1e-6
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

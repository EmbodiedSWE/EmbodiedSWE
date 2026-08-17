"""Teleport solution for WedgePressScene (sim_gen task `approach_grasp_bowl_i285`) —
the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY, and every teleport of the wedge ends in FREE
SPACE: the wedge is carried from its spawn pad to a hover pose apex-down over the roof
slot — tip flat centred on the seam readback, yaw aligned with the housing, tip bottom
well above the roof plane, velocities zeroed — and then it is simply RELEASED. GRAVITY
drives the whole load-bearing interaction: the tip falls into the seam gap between the
sled noses, the 45-degree faces convert the vertical stroke into symmetric horizontal
thrust through real contact, both sleds are plowed outward at once (the housing's end
caps arrest whatever coasts), and the stroke ends when the tip bottoms out on the slick
floor strip. The judged objects (the sleds) are NEVER teleported, pushed, or touched by
anything except the falling wedge; if a strike stalls short of the seat gate, the wedge
is lifted back to a free-space hover (transport again) and re-dropped, up to 5 times.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.4 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.approach_grasp_bowl_i285.solve --headless [--seed N]
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
    env = ENVS.get("simgen.wedge_press")().build(num_envs=args.num_envs, device=device)
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

    psi = float(scene.rig_yaw[0])

    def sleds_rig() -> torch.Tensor:
        """(2,3) sled centres in the RIG frame."""
        return scene._sled_pos()[0]

    def wedge_rig() -> torch.Tensor:
        """(3,) wedge tip-origin in the RIG frame."""
        return scene._wedge_pos()[0]

    def seam_center() -> float:
        """Rig-frame x midpoint of the two sled INNER nose faces (exact readback)."""
        sp = sleds_rig()
        noses = []
        for i in range(2):
            x = float(sp[i, 0])
            noses.append(x - math.copysign(c.sled_len / 2, x))
        return 0.5 * (noses[0] + noses[1])

    def seated_now() -> bool:
        wp = scene._wedge_pos()
        return bool(scene._in_window(wp, c.seat_z)[0])

    def report(tag: str) -> None:
        sp = sleds_rig()
        wp = wedge_rig()
        sv = scene._sled_vel()[0]
        print(f"[solve] {tag:14s} | sleds_x=[" + ",".join(
            f"{float(sp[i, 0]):+.3f}" for i in range(2)) + "] sleds_z=[" + ",".join(
            f"{float(sp[i, 2]):.3f}" for i in range(2)) + "] sleds_v=[" + ",".join(
            f"{float(sv[i]):.3f}" for i in range(2)) + "] "
            f"wedge=({float(wp[0]):+.3f},{float(wp[1]):+.3f},{float(wp[2]):.3f}) "
            f"eng={float(scene.engaged[0]):.0f} "
            f"del={float(scene.delivered[0].sum()):.0f} "
            f"seat={float(scene.seated[0]):.0f} spoiled={bool(scene.spoiled[0])} "
            f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
            flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ----- transport primitive --------------------------------------------------------------
    def hover_release(lx: float, tip_z: float, settle: int) -> None:
        """TRANSPORT: one teleport of the WEDGE to a free-space hover — apex down, tip
        flat at rig-local (lx, 0, tip_z), yaw aligned with the tunnel, velocities
        zeroed. `tip_z` is above the roof plane, so nothing overlaps. Then hands-off:
        gravity drives the strike through real contact."""
        assert tip_z > c.roof_hi + 0.005, "hover must start in free space"
        cpsi, spsi = math.cos(psi), math.sin(psi)
        wx = float(scene.rig_pos[0, 0]) + lx * cpsi
        wy = float(scene.rig_pos[0, 1]) + lx * spsi
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = wx, wy, tip_z
        st[:, 3], st[:, 6] = math.cos(psi / 2), math.sin(psi / 2)
        st[:, 0:3] += scene.env_origins
        scene.wedge.write_root_state_to_sim(st, all_ids)
        step(settle)

    # ---------------- phase 0: reset, settle, baseline --------------------------------------
    step(90)
    sp0 = sleds_rig()
    wp0 = wedge_rig()
    gap0 = abs(float(sp0[0, 0]) - float(sp0[1, 0])) - c.sled_len
    print(f"[solve] layout readback (seed {args.seed}): "
          f"rig=({float(scene.rig_pos[0, 0]):+.3f},{float(scene.rig_pos[0, 1]):+.3f}) "
          f"yaw={psi:+.2f} pad_side={float(scene.pad_side[0]):+.0f} sleds_x=[" + ",".join(
          f"{float(sp0[i, 0]):+.3f}" for i in range(2)) + f"] gap={gap0 * 1000:.1f}mm "
          f"seam_c={seam_center() * 1000:+.1f}mm "
          f"wedge=({float(wp0[0]):+.3f},{float(wp0[1]):+.3f},{float(wp0[2]):.3f})",
          flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert not bool(scene.spoiled[0]), "spoiled at reset"

    # ---------------- phase 1: the strike — hover over the seam, release --------------------
    # Tip flat (4 mm) centred on the seam readback (gap >= 12 mm: +/-4 mm tolerance by
    # construction); drop from 95 mm — 4 cm of free fall before first face contact.
    hover_release(seam_center(), 0.095, settle=180)
    report("strike-1")
    s1 = print_score("P1 strike (transport hover + gravity drive)")
    assert s1 >= s0 - 1e-6, "score decreased across the strike"

    # ---------------- phase 2: re-strike if the stroke stalled short of the seat ------------
    for attempt in range(5):
        if seated_now():
            break
        print(f"[solve] re-strike {attempt}: wedge z={float(wedge_rig()[2]):.3f} "
              f"(seat gate {c.seat_z:.3f}) — lift to free space and re-drop", flush=True)
        hover_release(seam_center(), 0.150, settle=200)
        report(f"re-strike-{attempt}")
    if not seated_now():
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (wedge never seated)", flush=True)
        os._exit(1)
    step(200)  # sleds slam the end caps, rebound dies (damping), everything settles
    report("settled")
    s2 = print_score("P2 seat + settle")
    assert s2 >= s1 - 1e-6, "score decreased across seating"

    if not bool(scene.success()[0]):
        step(240)  # slow creepers: two more seconds hands-off
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after seat+settle)", flush=True)
        os._exit(1)

    # ---------------- phase 3: persistence (>= 3.4 simulated seconds, hands-off) ------------
    hold = True
    for _ in range(10):  # 10 x 41 = 410 substeps = 3.42 s at 120 Hz
        step(41)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s3 = print_score("P3 persistence 3.4 s")
    ok = hold and bool(scene.success()[0]) and s3 >= s2 - 1e-6
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

"""Teleport solution for GradeBeaconScene (sim_gen task `stack_blocks_i415`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY. The plan is episode-dependent ARITHMETIC:
read the collar grade from the scene spec, solve the exact subset-sum over the
four block heights (integer mm), then build. Every load-bearing interaction goes
through contact dynamics:
  1. CARRY (transport): each chosen block is teleported from its scatter slot to
     a hover pose ~18 mm ABOVE its rest height over the pad column — never into
     contact.
  2. SEAT (dynamics): a velocity-regulated vertical force (gravity feed-forward +
     PD to -0.05 m/s, force at the CoM only — no torque, no orientation pinning)
     lowers the block onto the pad / the pillar so far; an xy PD keeps it
     centered. The wrench is DROPPED at first seat contact (or a contact stall);
     the pillar carries its own weight, blocks are never welded or held.
  3. Repeat for each subset block (tallest first), then CROWN: the beacon is
     hover-teleported above the finished pillar and force-lowered the same way,
     coming to rest with its center level with the collar band.
After the crown, hands off: `SIM_GEN_SCORE` prints at every phase boundary
(non-decreasing — build and place credit are latched), success() must hold
through a >= 3.3 simulated-second persistence window with no intervention, and
only then `SIM_GEN_SOLVE: SUCCESS` is printed.

Run (forge): python -u -m simgen_tasks.stack_blocks_i415.solve --headless [--seed N]
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

BLOCK_HS = scene_mod.BLOCK_HS
subset_for = scene_mod.subset_for

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.grade_beacon")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def wrench(body, f_w: torch.Tensor) -> None:
        """Apply a WORLD force (n,3) to `body`, expressed in its CURRENT link frame
        (`is_global=True` silently drops torques on this stack — transform manually,
        the house convention). Re-set every step while pushing."""
        from isaaclab.utils.math import quat_apply_inverse

        q = body.data.root_link_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f_w).unsqueeze(1),
            torch.zeros(n, 1, 3, device=device),
            env_ids=all_ids)

    def report(tag: str) -> None:
        top = float(scene.pillar_top()[0])
        dz = float(scene.beacon_dz()[0])
        vmax = max(float(b.data.root_lin_vel_w[0].norm()) for b in scene.blocks)
        wmax = max(float(b.data.root_ang_vel_w[0].norm()) for b in scene.blocks)
        print(f"[solve] {tag:16s} | pillar_top={top * 1000:.0f}mm "
              f"grade={float(scene.grade[0]) * 1000:.0f}mm "
              f"beacon_dz={dz * 1000:+.0f}mm in_col={bool(scene.beacon_in_col()[0])} "
              f"supported={bool(scene.supported()[0])} "
              f"hold={float(scene.hold_count[0]):.0f} "
              f"col_still={float(scene.col_still[0]):.0f} "
              f"bv={float(scene.beacon.data.root_lin_vel_w[0].norm()):.3f} "
              f"bw={float(scene.beacon.data.root_ang_vel_w[0].norm()):.3f} "
              f"blkv={vmax:.3f} blkw={wmax:.3f} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def seat(body, mass: float, xy: torch.Tensor, half_h: float,
             rest_bottom: float) -> None:
        """Force-regulated descent: gravity FF + PD to -0.05 m/s at the CoM, xy PD
        to the pad column, until the body's bottom face reaches `rest_bottom`
        (world z) or the descent stalls in contact. Then the wrench is dropped."""
        seated = False
        for k in range(500):
            z = float(body.data.root_pos_w[0, 2])
            gap = (z - half_h) - rest_bottom
            vz = float(body.data.root_lin_vel_w[0, 2])
            if gap < 0.002 or (gap < 0.010 and abs(vz) < 0.005 and k > 40):
                seated = True
                break
            v = body.data.root_lin_vel_w
            f = torch.zeros(n, 3, device=device)
            f[:, 2] = (mass * 9.81 + 5.0 * (-0.05 - v[:, 2])).clamp(0.0, 2 * mass * 9.81)
            f[:, 0:2] = (3.0 * (xy - body.data.root_pos_w[:, 0:2])
                         - 1.5 * v[:, 0:2]).clamp(-0.5, 0.5)
            wrench(body, f)
            env.step(no_action)
        wrench(body, torch.zeros(n, 3, device=device))
        assert seated, "descent never seated"

    # ---------------- phase 0: reset, settle, read the spec --------------------------------
    step(60)
    grade = float(scene.grade[0])
    pad_xy = scene.pad_xy[0:1, :].clone().expand(n, 2).contiguous()
    pad_top = float(scene.env_origins[0, 2]) + c.pad_t
    subset = subset_for(grade)
    order = sorted(subset, key=lambda i: -BLOCK_HS[i])  # tallest first: low CoM
    print(f"[solve] spec: grade={grade * 1000:.0f}mm -> subset "
          f"{[round(BLOCK_HS[i] * 1000) for i in order]}mm "
          f"(pad at {pad_xy[0].tolist()})", flush=True)
    report("reset")
    last = print_score("P0 reset+settle")
    assert last <= 0.02, "null credit at reset — rubric leak"

    # ---------------- build the pillar: per block CARRY (transport) + SEAT (dynamics) ------
    placed = 0.0
    for ph, i in enumerate(order):
        blk, h = scene.blocks[i], BLOCK_HS[i]
        rest_bottom = pad_top + placed
        # CARRY: hover teleport, upright, ~18 mm above the rest height — no contact
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = pad_xy
        st[:, 2] = rest_bottom + h / 2 + 0.018
        st[:, 3] = 1.0
        blk.write_root_state_to_sim(st, all_ids)
        # SEAT: regulated force descent onto the pad / pillar
        seat(blk, c.block_mass, pad_xy, h / 2, rest_bottom)
        step(60)  # hands off: pillar settles, build latch reads a still column
        placed += h
        report(f"seated-{round(h * 1000)}")
        assert bool(scene.block_in_col()[0, i]), "placed block left the column"
        top = float(scene.pillar_top()[0])
        assert abs(top - placed) < 0.008, \
            f"pillar top {top * 1000:.0f}mm != planned {placed * 1000:.0f}mm"
        s = print_score(f"P{ph + 1} block {round(h * 1000)}mm seated (contact dynamics)")
        assert s >= last - 1e-6, "score decreased across a block placement"
        last = s

    # ---------------- CROWN: beacon carried above the pillar, force-lowered ----------------
    rest_bottom = pad_top + placed
    st = torch.zeros(n, 13, device=device)
    st[:, 0:2] = pad_xy
    st[:, 2] = rest_bottom + c.beacon_s / 2 + 0.018
    st[:, 3] = 1.0
    scene.beacon.write_root_state_to_sim(st, all_ids)
    seat(scene.beacon, c.beacon_mass, pad_xy, c.beacon_s / 2, rest_bottom)
    # hands off: the crown settles; wait (bounded) for the sustained hold counter
    for _ in range(600):
        step(1)
        if bool(scene.success()[0]):
            break
    report("crowned")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after crowning)", flush=True)
        os._exit(1)
    s = print_score(f"P{len(order) + 1} beacon crowned at grade (contact dynamics)")
    assert s >= last - 1e-6
    last = s

    # ---------------- persistence: >= 3.3 s hands off --------------------------------------
    hold, flickers = True, 0
    for k in range(400):  # 400 substeps = 3.33 s at 120 Hz, no intervention
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:
                print(f"[solve] persist flicker @step {k}: "
                      f"dz={float(scene.beacon_dz()[0]) * 1000:+.1f}mm "
                      f"sup={bool(scene.supported()[0])}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s = print_score(f"P{len(order) + 2} persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s >= last - 1e-6
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)

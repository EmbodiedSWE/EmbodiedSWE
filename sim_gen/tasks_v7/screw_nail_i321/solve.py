"""Teleport solution for RamDispenserScene (sim_gen task `screw_nail_i321`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY (one move: carrying the free, empty bin across
open floor to the catch point under the muzzle). Every load-bearing interaction goes
through contact dynamics:
  1. STAGE (transport): the bin is teleported from its park slot to the catch point
     and settled. It is empty and free-standing — pure transport across open space.
  2. PUMP CYCLES (dynamics, one per cube; cubes are NEVER teleported): a
     velocity-regulated horizontal force on the ram body
       - PULLS the ram back until its tail bottoms out on the rear stop wall — the
         bottom cube then gravity-feeds out of the sealed tower onto the channel
         floor (readback-verified);
       - PUSHES the ram forward (cruise, then a fast final segment so the cube
         leaves the muzzle lip with enough speed to carry over the bin's near wall)
         until the handle post bottoms out on the end-plate stop — the ram nose
         shoves the cube down the roofed channel, off the lip, and gravity + contact
         land it in the bin.
     Stalls retry with escalated servo GAIN (not just force cap).
  3. SETTLE + PERSIST: hands-off settling to success(), then >= 3 simulated seconds
     with success() still true.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched) and `SIM_GEN_SOLVE: SUCCESS` only if success() persists.

Run (forge): python -u -m simgen_tasks.screw_nail_i321.solve --headless [--seed N]
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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ram_dispenser")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(3, device=device)
    rx, ry = c.rig_pos

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def wrench(body, f3: torch.Tensor) -> None:
        """Apply a WORLD force to `body`, expressed in its CURRENT link frame
        (`is_global=True` silently drops torques on this stack — transform manually,
        the house convention). Re-set every step while pushing."""
        from isaaclab.utils.math import quat_apply_inverse

        q = body.data.root_link_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f3.view(1, 3).expand(n, 3)).unsqueeze(1),
            torch.zeros(n, 1, 3, device=device),
            env_ids=all_ids)

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def tip() -> float:
        return float(scene.ram_tip()[0])

    def report(tag: str) -> None:
        bp = scene.block_pos()[0]
        binp = rel(scene.bin)
        pres = scene.present[0]
        zs = " ".join(f"b{i}={'--' if not bool(pres[i]) else f'{float(bp[i, 2]):.3f}'}"
                      for i in range(3))
        inb = scene.block_in_bin()[0]
        print(f"[solve] {tag:16s} | tip={tip():+.3f} {zs}"
              f" in_bin=({int(inb[0])},{int(inb[1])},{int(inb[2])})"
              f" bin=({float(binp[0]):+.3f},{float(binp[1]):+.3f})"
              f" staged={bool(scene.bin_staged_now()[0])}"
              f" success={bool(scene.success()[0])}"
              f" score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- ram velocity servo -----------------------------------------------------
    def drive(v_des: float, stop_fn, gain: float = 30.0, cap: float = 10.0,
              max_steps: int = 900) -> bool:
        """Velocity-regulated horizontal force on the ram until `stop_fn()` is true.
        gain*dt/m = 30/(120*0.6) = 0.42 < 1 (wrenches act one substep late)."""
        f3 = torch.zeros(3, device=device)
        for _ in range(max_steps):
            if stop_fn():
                wrench(scene.ram, zero3)
                return True
            v = float(scene.ram.data.root_lin_vel_w[0, 0])
            f3[0] = max(min(gain * (v_des - v), cap), -cap)
            wrench(scene.ram, f3)
            env.step(no_action)
        wrench(scene.ram, zero3)
        return stop_fn()

    def retract(gain: float = 30.0) -> bool:
        ok = drive(-0.15, lambda: tip() <= c.tip_ret + 0.004, gain=gain)
        step(30)
        return ok

    def advance(gain: float = 30.0, eject_probe=None) -> bool:
        """Cruise to the muzzle approach, then a fast final segment so the cube
        leaves the lip at speed and carries over the bin's near wall."""
        done = (lambda: False) if eject_probe is None else eject_probe
        okA = drive(+0.12, lambda: tip() >= 0.110 or done(), gain=gain)
        okB = drive(+0.55, lambda: tip() >= c.tip_adv - 0.006 or done(), gain=gain,
                    max_steps=400)
        step(30)
        return okA and (okB or done())

    # ---------------- phase 0: reset, settle, baseline ----------------------------------------
    step(60)
    pres = scene.present[0]
    k = int(pres.sum())
    binp = rel(scene.bin)
    print(f"[solve] layout readback (seed {args.seed}): k={k} "
          f"present=({int(pres[0])},{int(pres[1])},{int(pres[2])}) tip={tip():+.3f} "
          f"bin=({float(binp[0]):+.3f},{float(binp[1]):+.3f})", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, "null credit at reset — rubric leak"

    # ---------------- phase 1: stage the bin (transport only) ---------------------------------
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1], st[:, 2] = rx + c.catch_x, ry, 0.001
    st[:, 3] = 1.0
    st[:, 0:3] += scene.env_origins
    scene.bin.write_root_state_to_sim(st, all_ids)
    step(60)
    report("bin staged")
    s1 = print_score("P1 bin staged under the muzzle (transport)")
    assert s1 >= s0 - 1e-6
    assert bool(scene.bin_staged_now()[0]), "bin not staged at the catch point"

    # ---------------- phase 2..: one pump cycle per cube (dynamics) ---------------------------
    s_prev = s1
    for cyc in range(k):
        # the feed cube = lowest present cube still inside the machine
        cand = [i for i in range(3)
                if bool(pres[i]) and float(scene.eject_latch[0, i]) < 0.5]
        assert cand, "no candidate cube left but cycles remain"
        bi = min(cand, key=lambda i: float(scene.block_pos()[0, i, 2]))
        body = scene.blocks[bi]

        def bz() -> float:
            return float(rel(body)[2])

        def bx() -> float:
            return float(rel(body)[0]) - rx

        def fed() -> bool:
            return bz() < c.chan_floor_z + c.block_edge / 2 + 0.006

        def ejected() -> bool:
            return bz() < c.eject_z

        ok = False
        gain = 30.0
        for attempt in range(4):
            # -- pull back to the rear stop: the cube gravity-feeds --
            retract(gain=gain)
            for _ in range(12):  # up to 3 s for the drop + settle
                if fed():
                    break
                step(30)
            if not fed():
                print(f"[solve] cycle {cyc}: cube {bi} did not feed "
                      f"(z={bz():.3f}, tip={tip():+.3f}) — jiggle", flush=True)
                drive(+0.12, lambda: tip() >= 0.060, gain=gain, max_steps=240)
                gain *= 1.3
                continue
            print(f"[solve] cycle {cyc}: cube {bi} fed (z={bz():.3f} x={bx():+.3f})",
                  flush=True)
            # -- push forward to the front stop: the nose ejects the cube --
            advance(gain=gain, eject_probe=ejected)
            for _ in range(6):
                if ejected():
                    break
                step(30)
            if ejected():
                ok = True
                break
            print(f"[solve] cycle {cyc}: cube {bi} not ejected "
                  f"(z={bz():.3f} x={bx():+.3f} tip={tip():+.3f}) — retry", flush=True)
            gain *= 1.3
        assert ok, f"cycle {cyc}: cube {bi} never ejected after retries"
        step(90)  # free fall + contact settling inside the bin
        report(f"cycle {cyc} done")
        s_c = print_score(f"P2.{cyc + 1} cube {bi} fed and ejected (contact dynamics)")
        assert s_c >= s_prev - 1e-6, "score decreased across a pump cycle"
        s_prev = s_c

    # park the ram mid-stroke, hands off (nose out of the bin airspace)
    drive(-0.15, lambda: tip() <= 0.100, gain=30.0, max_steps=400)
    wrench(scene.ram, zero3)

    # ---------------- phase 3: settle to success ----------------------------------------------
    for _ in range(16):  # up to 4 s of hands-off settling
        if bool(scene.success()[0]):
            break
        step(30)
    report("settled")
    s3 = print_score("P3 all settled")
    assert s3 >= s_prev - 1e-6, "score decreased across settling"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after stage+pump+settle)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3 simulated seconds, no intervention) ----------
    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                inb = scene.block_in_bin()[0]
                print(f"[solve] persist flicker @step {i}: "
                      f"in_bin=({int(inb[0])},{int(inb[1])},{int(inb[2])}) "
                      f"upright={bool(scene.bin_upright()[0])} "
                      f"settled={bool(scene.settled()[0])}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)

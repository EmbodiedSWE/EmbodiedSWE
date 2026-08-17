"""Teleport solution for ShroudPostsScene (sim_gen task `stack_cups_i434`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY. Both load-bearing interactions go through
contact dynamics, one cup at a time, big -> mid -> small (the LIFO order the nest
physically forces):
  1. UNSHROUD (dynamics): the outermost cup is lifted off the nest by a
     velocity-regulated vertical force (gravity feed-forward + PD to +0.10 m/s,
     force at the CoM only — no torque, no orientation pinning) while an xy PD
     holds it on the storage axis so its wall clears the 3.5 mm annulus around the
     next cup; the lift ends with the rim well above everything left on the
     storage station. If captivity were violated (an inner cup entrained), the
     inner cups would move — asserted still after every unshroud.
  2. CARRY (transport): the freed cup is teleported to a hover pose concentric
     above its size-matched post, mouth ~15 mm above the cone tip (nothing in
     contact), with a brief regulated zero-velocity hold.
  3. SEAT (dynamics): a velocity-regulated descent (PD to -0.06 m/s + xy PD to
     the post axis) lowers the cup; the cone tip funnels through the mouth, the
     shaft slides in, and the rim lands on the pad. The wrench is DROPPED at the
     seat (or on a stall in contact); 60 hands-off steps settle it; `capped`
     asserted for that post.
Nothing is ever teleported into contact, never welded, never held at the end.
`SIM_GEN_SCORE` is printed at every phase boundary (non-decreasing — clear and
cap credits are latched), success() must hold through a >= 3.3 simulated-second
hands-off persistence window, and only then `SIM_GEN_SOLVE: SUCCESS` is printed.

Run (forge): python -u -m simgen_tasks.stack_cups_i434.solve --headless [--seed N]
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

SIZES = scene_mod.SIZES

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.shroud_posts")().build(num_envs=args.num_envs, device=device)
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

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def rim_z(i: int) -> float:
        """Cup i mouth-plane height above the ground plane (world z minus origin)."""
        cz = float(rel(scene.cups[i])[2])
        return cz - float(scene.cup_up_z()[0, i]) * float(scene.heights[i]) / 2

    def report(tag: str) -> None:
        cap = scene.capped()[0]
        up = scene.cup_up_z()[0]
        ad = scene.axis_dist()[0]
        rim = scene.rim_height_over()[0]
        cups = " ".join(
            f"{nm}:up_z={float(up[i]):+.2f} rim@post={float(rim[i, i]) * 1000:+.0f}mm "
            f"ax={float(ad[i, i]) * 1000:.0f}mm"
            for i, (nm, _) in enumerate(SIZES))
        caps = " ".join(f"{nm}:capped={bool(cap[i])}" for i, (nm, _) in enumerate(SIZES))
        print(f"[solve] {tag:14s} | {cups} | {caps} | "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    stor = scene.storage_xy[0] + scene.env_origins[0, :2]
    print(f"[solve] layout (seed {args.seed}) storage_xy="
          f"({float(scene.storage_xy[0, 0]):+.3f},{float(scene.storage_xy[0, 1]):+.3f})",
          flush=True)
    for i, (nm, _) in enumerate(SIZES):
        cu, po = rel(scene.cups[i]), rel(scene.posts[i])
        print(f"[solve] layout {nm}: "
              f"cup=({float(cu[0]):+.3f},{float(cu[1]):+.3f},{float(cu[2]):.3f}) "
              f"post=({float(po[0]):+.3f},{float(po[1]):+.3f},{float(po[2]):.3f}) "
              f"rim_z={rim_z(i) * 1000:.0f}mm", flush=True)
    report("reset")
    last = print_score("P0 reset+settle")
    assert last <= 0.02, "null credit at reset — rubric leak"

    # ---------------- per cup, big -> mid -> small (the forced LIFO order) -----------------
    ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    phase = 0
    for i in (2, 1, 0):
        nm = SIZES[i][0]
        cup, post = scene.cups[i], scene.posts[i]
        mass = c.cup_mass[i]

        # snapshot the cups that stay behind — captivity means they must not move
        inner = [j for j in range(3) if j < i]
        inner_pos0 = [scene.cups[j].data.root_pos_w.clone() for j in inner]

        # --- UNSHROUD: velocity-regulated vertical lift off the nest (dynamics) ---
        lift_to = 0.135 + float(scene.heights[i])  # cup-center target above ground
        lifted = False
        for k in range(600):
            cz = float(rel(cup)[2])
            if cz > lift_to:
                lifted = True
                break
            v = cup.data.root_lin_vel_w
            f = torch.zeros(n, 3, device=device)
            f[:, 2] = (mass * 9.81 + 4.0 * mass * (0.10 - v[:, 2])).clamp(
                0.0, 2.5 * mass * 9.81)
            f[:, 0:2] = (3.0 * (stor[None, :] - cup.data.root_pos_w[:, 0:2])
                         - 1.5 * v[:, 0:2]).clamp(-0.8, 0.8)
            wrench(cup, f)
            env.step(no_action)
        assert lifted, f"{nm} cup never lifted clear of the nest"
        # regulated hold a moment so the readback below is quasi-static
        for _ in range(10):
            v = cup.data.root_lin_vel_w
            f = torch.zeros(n, 3, device=device)
            f[:, 2] = (mass * 9.81 + 4.0 * mass * (0.0 - v[:, 2])).clamp(
                0.0, 2.5 * mass * 9.81)
            wrench(cup, f)
            env.step(no_action)
        for j, p0 in zip(inner, inner_pos0):
            d = float((scene.cups[j].data.root_pos_w - p0)[0].norm())
            print(f"[solve] unshroud-{nm}: inner {SIZES[j][0]} moved {d * 1000:.1f}mm",
                  flush=True)
            assert d < 0.015, f"unshroud entrained inner cup {SIZES[j][0]}"
        assert float(scene.cup_up_z()[0, i]) > 0.95, f"{nm} cup toppled during lift"
        report(f"unshroud-{nm}")

        # --- CARRY (transport): hover concentric over the matched post ---
        p_pos = post.data.root_pos_w.clone()
        tip_z = c.pad_h + c.post_h[i]  # cone-tip height above ground
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = p_pos[:, 0:2]
        st[:, 2] = scene.env_origins[:, 2] + tip_z + 0.015 + float(scene.heights[i]) / 2
        st[:, 3:7] = ident
        cup.write_root_state_to_sim(st, all_ids)
        for _ in range(10):
            v = cup.data.root_lin_vel_w
            f = torch.zeros(n, 3, device=device)
            f[:, 2] = (mass * 9.81 + 4.0 * mass * (0.0 - v[:, 2])).clamp(
                0.0, 2.5 * mass * 9.81)
            f[:, 0:2] = (3.0 * (p_pos[:, 0:2] - cup.data.root_pos_w[:, 0:2])
                         - 1.5 * v[:, 0:2]).clamp(-0.8, 0.8)
            wrench(cup, f)
            env.step(no_action)
        report(f"hover-{nm}")
        assert rim_z(i) > tip_z + 0.008, "hover must start with the mouth above the tip"
        phase += 1
        s = print_score(f"P{phase} {nm} cup unshrouded + carried to hover (transport)")
        assert s >= last - 1e-6, "score decreased at carry"
        last = s

        # --- SEAT: velocity-regulated descent until the rim lands on the pad ---
        seated = False
        for k in range(600):
            rim_over = float(scene.rim_height_over()[0, i, i])
            vz = float(cup.data.root_lin_vel_w[0, 2])
            if rim_over < 0.002 or (rim_over < 0.010 and abs(vz) < 0.004 and k > 40):
                seated = True
                break
            v = cup.data.root_lin_vel_w
            f = torch.zeros(n, 3, device=device)
            f[:, 2] = (mass * 9.81 + 4.0 * mass * (-0.06 - v[:, 2])).clamp(
                0.0, 2.5 * mass * 9.81)
            f[:, 0:2] = (3.0 * (p_pos[:, 0:2] - cup.data.root_pos_w[:, 0:2])
                         - 1.5 * v[:, 0:2]).clamp(-0.8, 0.8)
            wrench(cup, f)
            env.step(no_action)
        wrench(cup, torch.zeros(n, 3, device=device))
        assert seated, f"{nm} cup descent never reached the pad"
        step(60)  # hands off: rim-on-pad contact carries the cup; it settles
        report(f"seated-{nm}")
        assert bool(scene.capped()[0, i]), f"post {nm} did not read capped"
        phase += 1
        s = print_score(f"P{phase} {nm} post capped (contact dynamics)")
        assert s >= last - 1e-6, "score decreased across seating"
        last = s

    # ---------------- final: hands off, success + persistence ------------------------------
    step(60)
    report("all-capped")
    s = print_score("P7 all three posts capped, hands off")
    assert s >= last - 1e-6
    last = s
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after capping all posts)", flush=True)
        os._exit(1)

    hold, flickers = True, 0
    for k in range(400):  # 400 substeps = 3.33 s at 120 Hz, no intervention
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:
                cap = scene.capped()[0]
                print(f"[solve] persist flicker @step {k}: capped="
                      f"{[bool(v) for v in cap]}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s = print_score("P8 persistence 3.3 s")
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

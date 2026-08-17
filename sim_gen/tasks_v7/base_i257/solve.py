"""Teleport solution for CargoTramScene (sim_gen task `base_i257`) — the task's
legitimacy certificate.

Teleportation handles TRANSPORT ONLY. Every load-bearing interaction goes through
contact/constraint dynamics:
  1. LOAD (transport + gravity/contact): the red cargo cube is teleported to a point
     in FREE SPACE above the cart's open bucket mouth (computed from the cart's
     CURRENT pose), released with zero velocity, and DROPPED. The insertion is made
     by gravity and bucket-wall contact. Nothing is ever spawned seated.
  2. CLIMB (pure contact dynamics): the loaded cart — never teleported — is pushed
     along the channel by a velocity-regulated horizontal force: across the flat, up
     the 14 deg ramp (too steep and slick to rest on: tan(theta)=0.25 vs pair
     friction 0.12 — stop pushing mid-ramp and it slides back), over the crest,
     where it tips and DROPS 8 mm into the summit pocket under the canopy.
  3. SEAT (pure contact dynamics): a gentle constant push presses the cart against
     the pocket's back wall — a contact-terminated, self-aligning end stop — then
     everything is released and settles.

The force servo uses a HIGH gain with a hard clamp (f = gain*(v_des - v), clamped to
[0, f_max]): the stall force at v = 0 is the clamp (6 N), far above the ~1.1 N the
ramp demands, while the regulated cruise speed stays low — a constant 6 N would slam
the cart into the back wall and bounce the cargo; a low-gain servo would stall at the
ramp foot.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.base_i257.solve --headless [--seed N]
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
    from isaaclab.utils.math import quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cargo_tram")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(3, device=device)

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
        the house convention). Re-set every step while pushing (`control_period == 1`
        for the null robot, so each env.step is one physics substep)."""
        from isaaclab.utils.math import quat_apply_inverse

        q = body.data.root_link_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f3.view(1, 3).expand(n, 3)).unsqueeze(1),
            torch.zeros(n, 1, 3, device=device),
            env_ids=all_ids)

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        cp = rel(scene.cart)
        gp = rel(scene.cargo)
        print(f"[solve] {tag:16s} |"
              f" cart=({float(cp[0]):+.3f},{float(cp[1]):+.3f},{float(cp[2]):.3f})"
              f" cargo=({float(gp[0]):+.3f},{float(gp[1]):+.3f},{float(gp[2]):.3f})"
              f" loaded={bool(scene.cargo_in_cart()[0])}"
              f" decoy_in={bool(scene.decoy_in_cart()[0])}"
              f" parked={bool(scene.cart_parked()[0])}"
              f" settled={bool(scene.settled()[0])}"
              f" success={bool(scene.success()[0])}"
              f" score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- contact primitives ---------------------------------------------------
    def drop_cargo() -> None:
        """TRANSPORT the red cube to free space above the cart's open bucket mouth
        (computed from the cart's CURRENT pose), release with zero velocity, and let
        gravity + bucket-wall contact make the insertion (the cart is on the open
        flat — the canopy is far away and the sky above the mouth is clear)."""
        local = torch.tensor([0.0, 0.0, c.cart_h + 0.05], device=device)
        mouth = scene.cart.data.root_pos_w[0] + quat_apply(
            scene.cart.data.root_quat_w[0], local)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = mouth  # already world (env origin included via cart pos)
        st[:, 3] = 1.0
        scene.cargo.write_root_state_to_sim(st, all_ids)
        # free fall into the bucket + settling (cart may rock on its slick floor)
        for _ in range(12):
            step(30)
            if float(scene.cargo.data.root_lin_vel_w[0].norm()) < 0.05 \
                    and float(scene.cart.data.root_lin_vel_w[0].norm()) < 0.05:
                break
        assert bool(scene.cargo_in_cart()[0]), \
            f"cargo missed the bucket (cargo={rel(scene.cargo).tolist()})"

    def push_cart() -> None:
        """Velocity-regulated +x push on the loaded cart: across the flat, up the
        ramp, over the crest, into the pocket. The cart is never teleported; the
        force is re-set every physics step in the cart's current frame. Ends with a
        gentle constant seat-press against the back wall, then release."""
        f3 = torch.zeros(3, device=device)
        x0 = float(rel(scene.cart)[0])
        hit_climb = False
        for _ in range(2200):
            rx = float(rel(scene.cart)[0])
            if not hit_climb and rx > c.climb_x:
                hit_climb = True  # print between steps — the applied wrench persists
                report("mid-ramp")
                print_score("P2a loaded cart past mid-ramp (climb latch)")
            if rx > 0.8145:  # in the pocket, ~2.5 mm short of the seated rest
                break
            vx = float(scene.cart.data.root_lin_vel_w[0, 0])
            f = 80.0 * (0.10 - vx)  # stall force = clamp (6 N) >> 1.1 N ramp demand
            f3[0] = max(min(f, 6.0), 0.0)
            wrench(scene.cart, f3)
            env.step(no_action)
        rx = float(rel(scene.cart)[0])
        assert rx - x0 > 0.30, f"push did not move the cart (x {x0:.3f}->{rx:.3f})"
        assert rx > 0.8125, f"cart did not reach the pocket (x={rx:.3f})"
        report("pocket entry")
        print_score("P2b loaded cart dropped into the pocket (dock latch)")
        # seat against the back wall: contact-terminated, self-aligning
        f3[0] = 1.5
        for _ in range(90):
            wrench(scene.cart, f3)
            env.step(no_action)
        wrench(scene.cart, zero3)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(90)  # cart and cubes settle the spawn millimeter onto their supports
    cp, gp, dp = rel(scene.cart), rel(scene.cargo), rel(scene.decoy)
    print(f"[solve] layout readback (seed {args.seed}): "
          f"cart=({float(cp[0]):+.3f},{float(cp[1]):+.3f},{float(cp[2]):.3f}) "
          f"cargo=({float(gp[0]):+.3f},{float(gp[1]):+.3f}) "
          f"decoy=({float(dp[0]):+.3f},{float(dp[1]):+.3f})", flush=True)
    report("reset")
    assert float(cp[0]) < c.flat_x[1] - c.cart_len / 2, "cart must start on the flat"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, "null credit at reset — rubric leak"

    # ---------------- phase 1: load the cargo (drop + gravity/contact) ---------------------
    drop_cargo()
    report("cargo loaded")
    s1 = print_score("P1 cargo dropped into the bucket (gravity+contact)")
    assert s1 >= s0 - 1e-6

    # ---------------- phase 2: push the loaded cart up and into the pocket -----------------
    push_cart()
    report("cart seated")
    s2 = print_score("P2 loaded cart pushed up the ramp and seated (contact dynamics)")
    assert s2 >= s1 - 1e-6

    # ---------------- phase 3: settle to success -------------------------------------------
    for _ in range(16):  # up to 4 s of hands-off settling
        if bool(scene.success()[0]):
            break
        step(30)
    report("settled")
    s3 = print_score("P3 all settled")
    assert s3 >= s2 - 1e-6, "score decreased across settling"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after load+climb+seat+settle)",
              flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3 simulated seconds, no intervention) -------
    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                print(f"[solve] persist flicker @step {i}: "
                      f"loaded={bool(scene.cargo_in_cart()[0])} "
                      f"parked={bool(scene.cart_parked()[0])} "
                      f"settled={bool(scene.settled()[0])} "
                      f"cart_v={float(scene.cart.data.root_lin_vel_w[0].norm()):.4f} "
                      f"cargo_v={float(scene.cargo.data.root_lin_vel_w[0].norm()):.4f}",
                      flush=True)
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

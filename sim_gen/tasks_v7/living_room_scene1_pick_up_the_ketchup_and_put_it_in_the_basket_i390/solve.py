"""Teleport solution for PinnedHatchDeliveryScene (sim_gen task
`living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket_i390`) — the task's
legitimacy certificate.

Teleports are used for TRANSPORT ONLY (one pose write staging the ketchup on the
apron push lane). Every load-bearing interaction is contact physics driven by
velocity-regulated external wrenches (re-set every step, zeroed before judging):
the push THROUGH the window aperture, the topple over the sill into the basket, the
axial pin extraction under the sash's resting load, and the sash's gravity slam are
all dynamics. Gains respect the one-substep wrench delay (KV*dt/m << 1); forces are
rotated into the CURRENT body frame every step (the external-force frame-drag trap).

PLAN (read-only, from scene.describe()): the window is a one-shot resource — the
solid sash, once fallen, faces the bore and the pin can never go back — so deliver
FIRST, seal SECOND:
  P0 settle + readback: sash propped on the pin, bottles on their Bernoulli slots;
     score ~0,
  P1 stage the ketchup on the push lane (transport teleport), then force-push it
     along the housing +x axis: it bridges the sash channel on its base, steps DOWN
     onto the sill, topples over the inner edge and drops into the basket (contact),
  P2 pull the pin straight out (escalating axial force, velocity-capped); when the
     shaft clears the sash plate the sash free-falls and slams the window shut,
  P3 hands-off ring-down to success, then persistence >= 3.3 simulated seconds.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then `SIM_GEN_SOLVE: SUCCESS` only if success() still holds
after the hands-off hold.

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

import os
import threading

import torch

import robobench
from robobench.core import ENVS

try:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse
except ImportError:  # older isaaclab names
    from isaaclab.utils.math import quat_rotate as quat_apply
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# Bottle push servo (mass 0.30: KV*dt/m = 4/(120*0.3) ~= 0.11).
P_VDES = 0.25   # push speed (m/s) — carries the CoM past the drop lip (short-landing trap)
P_KV = 4.0      # velocity loop -> force (N s/m)
P_FMAX = 1.6    # axial clamp (N) — below the ~1.9 N tip threshold of the low-CoM bottle
P_KY = 2.0      # lane keeping -> desired lateral velocity (1/s)
P_FYMAX = 0.5   # lateral clamp (N)
# Pin extraction (escalating axial pull, velocity-capped bang-bang).
X_F0, X_FSTEP, X_FMAX = 0.8, 0.6, 6.0
X_VCAP = 0.30   # coast above this pull speed (m/s)
X_TIP_STOP = -0.180  # stop pulling once the tip is well behind the clear gate


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.pinned_hatch_delivery")().build(num_envs=args.num_envs,
                                                           device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def hframe(body) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos, lin vel) of a body in the housing frame."""
        qh = scene.housing.data.root_quat_w
        p = quat_apply_inverse(qh, body.data.root_pos_w - scene.housing.data.root_pos_w)
        v = quat_apply_inverse(qh, body.data.root_lin_vel_w)
        return p, v

    def push_h(body, f_h: torch.Tensor) -> None:
        """Apply a housing-frame force to a body (re-encoded into its CURRENT
        body frame — the frame-drag trap)."""
        f_w = quat_apply(scene.housing.data.root_quat_w, f_h)
        f_b = quat_apply_inverse(body.data.root_quat_w, f_w)
        body.set_external_force_and_torque(f_b.unsqueeze(1), zero_w, env_ids=all_ids)

    def wrenches_off() -> None:
        scene.ketchup.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
        scene.pin.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)

    def report(tag: str) -> None:
        kp, _ = hframe(scene.ketchup)
        print(f"[solve] {tag:12s} | q={float(scene.sash_q()[0]):+.4f} "
              f"tip={float(scene.pin_tip_x()[0]):+.4f} "
              f"k=({float(kp[0, 0]):+.3f},{float(kp[0, 1]):+.3f},{float(kp[0, 2]):+.3f}) "
              f"in={bool(scene._contained(scene.ketchup)[0])} "
              f"sealed={bool(scene.sealed()[0])} "
              f"latch i/s/f={float(scene.in_ever[0]):.0f}/{float(scene.seal_ever[0]):.0f}/"
              f"{float(scene.full_ever[0]):.0f} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def settle_until(cond, max_steps: int, need: int = 30) -> bool:
        got = 0
        for _ in range(max_steps):
            step(1)
            got = got + 1 if bool(cond()) else 0
            if got >= need:
                return True
        return False

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    step(90)
    for nm, want in (("sash", c.sash_mass), ("pin", c.pin_mass),
                     ("ketchup", c.bottle_mass), ("basket", c.basket_mass)):
        got = float(getattr(scene, nm).root_physx_view.get_masses().flatten()[0])
        assert abs(got - want) <= 0.2 * want, f"authored {nm} mass missing (got {got:.3f})"
    q = float(scene.sash_q()[0])
    tip = float(scene.pin_tip_x()[0])
    kp, _ = hframe(scene.ketchup)
    ket_side = float(kp[0, 1])
    p_h = (scene.housing.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] readback (seed {args.seed}): housing=({float(p_h[0]):+.3f},"
          f"{float(p_h[1]):+.3f}) q={q:+.4f} (prop ~{c.q_spawn:+.4f}) tip={tip:+.4f} "
          f"ketchup_y={ket_side:+.3f}", flush=True)
    assert 0.150 <= q <= c.q_spawn + 0.005, "sash must be propped open on the pin"
    assert tip >= -0.130, "pin tip must be engaged in the wall bore at spawn"
    assert abs(abs(ket_side) - c.slot_y) <= c.bot_jitter + 0.006, "ketchup must sit on a slot"
    report("reset")
    s0 = print_score("P0 reset+settle (sash propped on the pin)")
    assert s0 < 0.02, "score must start ~0"

    # ---------------- phase 1: deliver the ketchup through the open window -----------------
    # Transport teleport: stage the ketchup on the push lane (window axis), upright.
    off = torch.zeros(n, 3, device=device)
    off[:, 0], off[:, 2] = -0.26, 0.185
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.housing.data.root_pos_w + quat_apply(
        scene.housing.data.root_quat_w, off)
    st[:, 3:7] = scene.housing.data.root_quat_w
    scene.ketchup.write_root_state_to_sim(st, all_ids)
    step(30)
    kp, _ = hframe(scene.ketchup)
    assert abs(float(kp[0, 1])) < 0.01 and float(kp[0, 2]) < 0.20, "staging must have settled"
    # Force-push along housing +x: bridge the channel, step down, topple in (contact).
    dropped_at = None
    for i in range(900):
        kp, kv = hframe(scene.ketchup)
        f_h = torch.zeros(n, 3, device=device)
        f_h[:, 0] = (P_KV * (P_VDES - kv[:, 0])).clamp(-P_FMAX, P_FMAX)
        vy_des = (P_KY * (0.0 - kp[:, 1])).clamp(-0.10, 0.10)
        f_h[:, 1] = (P_KV * (vy_des - kv[:, 1])).clamp(-P_FYMAX, P_FYMAX)
        push_h(scene.ketchup, f_h)
        env.step(no_action)
        x, z = float(kp[0, 0]), float(kp[0, 2])
        if x > -0.105 or z < 0.155:
            dropped_at = i
            break
        if (i + 1) % 200 == 0:
            print(f"[solve] P1 telemetry @{i + 1}: x={x:+.4f} z={z:+.4f} "
                  f"vx={float(kv[0, 0]):+.3f}", flush=True)
    wrenches_off()
    assert dropped_at is not None, "push never reached the sill inner edge"
    print(f"[solve] P1: over the sill after {dropped_at + 1} push steps", flush=True)
    ok = settle_until(lambda: scene._contained(scene.ketchup)[0]
                      & (scene.still_count[0] >= 20), 600)
    report("P1-delivered")
    assert ok and bool(scene._contained(scene.ketchup)[0]), \
        "ketchup must rest contained in the basket"
    assert float(scene.in_ever[0]) > 0.5, "containment latch must have fired"
    s1 = print_score("P1 ketchup delivered through the window")
    assert s1 >= max(s0, 0.249), "containment credit missing"

    # ---------------- phase 2: pull the pin — the sash slams shut --------------------------
    f_lvl = X_F0
    last_tip = float(scene.pin_tip_x()[0])
    stall = 0
    for i in range(1500):
        tip = float(scene.pin_tip_x()[0])
        if tip <= X_TIP_STOP:
            break
        _, pv = hframe(scene.pin)
        f_h = torch.zeros(n, 3, device=device)
        pulling = pv[:, 0] > -X_VCAP  # velocity-capped bang-bang
        f_h[:, 0] = torch.where(pulling, -f_lvl * torch.ones_like(pv[:, 0]),
                                torch.zeros_like(pv[:, 0]))
        push_h(scene.pin, f_h)
        env.step(no_action)
        stall += 1
        if stall >= 60:
            if last_tip - tip < 0.002:  # no progress: escalate the pull
                f_lvl = min(f_lvl + X_FSTEP, X_FMAX)
                print(f"[solve] P2: escalate pull to {f_lvl:.1f} N @tip={tip:+.4f}",
                      flush=True)
            last_tip, stall = tip, 0
    wrenches_off()
    tip = float(scene.pin_tip_x()[0])
    print(f"[solve] P2: pin pulled to tip={tip:+.4f} in {i + 1} steps "
          f"(final pull {f_lvl:.1f} N)", flush=True)
    assert tip <= X_TIP_STOP + 0.004, "pin must be pulled clear of the sash channel"
    ok = settle_until(lambda: (scene.sash_q()[0] <= c.closed_tol)
                      & (scene.still_count[0] >= 20), 600)
    report("P2-sealed")
    assert ok and bool(scene.sealed()[0]), "sash must have slammed fully shut, pin clear"
    assert float(scene.seal_ever[0]) > 0.5, "seal latch must have fired"
    assert float(scene.full_ever[0]) > 0.5, "delivered+sealed latch must have fired"
    s2 = print_score("P2 pin pulled — sash slammed shut")
    assert s2 >= max(s1, 0.549), "seal credit missing"

    # ---------------- phase 3: ring-down to success, then persistence ----------------------
    consec = 0
    for _ in range(2400):  # up to 20 s
        step(1)
        if bool(scene.success()[0]):
            consec += 1
            if consec >= 120:
                break
        else:
            consec = 0
    report("P3-ringdown")
    s3 = print_score("P3 at rest, hands off")
    assert s3 >= s2 - 1e-6, "score decreased across P3"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after ring-down)", flush=True)
        os._exit(1)

    hold, flickers = True, 0
    for i in range(400):  # 400 steps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:
                print(f"[solve] persist flicker @step {i}: q={float(scene.sash_q()[0]):+.4f} "
                      f"tip={float(scene.pin_tip_x()[0]):+.4f} "
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

"""Teleport solution for JuiceCarouselScene (sim_gen task `libero_pick_orange_juice_i256`)
— the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY — exactly what a free-space carry delivers.
Every load-bearing interaction runs through real contact / applied-wrench dynamics:

  rotate  — a torque servo on the carousel (the crank-knob push a Franka performs,
            rendered as a small vertical-axis torque on the turntable body): an
            outer position loop commands a capped wheel speed toward zero window
            error, an inner loop tracks it with a clamped torque (<= 0.20 N m).
            The disc, the riding cartons, the bearing damping — all real dynamics.
            The lid... the wall never moves; only the disc turns.
  extract — a bang-bang horizontal drag on the juice carton (<= 1.7 N at CoM,
            velocity-regulated to ~0.10 m/s — the low radial pull a fingertip
            performs): the carton SLIDES off the disc, through the window, onto
            the flush porch, by friction contact the whole way. The wall would
            block this at any misaligned angle; the roof forbids lifting inside.
  deliver — the only teleport: the extracted carton is carried (teleport to a
            hover pose above the basket mouth, zero velocity) and RELEASED;
            gravity drops it the last ~10 cm into the basket.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing by
construction: the rubric is latched), then holds HANDS-OFF for >= 3 simulated
seconds after success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS`
only if it still holds.

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

import math
import os
import threading

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from .scene import BASKET_T, ITEM_H, RIM_H, _qapply, _qinv  # noqa: F401
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import BASKET_T, ITEM_H, RIM_H, _qapply, _qinv  # noqa: F401
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

# --- rotation servo (outer position loop -> wheel-speed command -> torque loop) ---
W_CAP = 0.60  # commanded wheel-speed cap (rad/s)
K_POS = 1.5  # ω_des = clamp(-K_POS * az_err, ±W_CAP)
K_VEL = 0.50  # τ = clamp(K_VEL * (ω_des - ω), ±TAU_MAX)  (N m per rad/s)
TAU_MAX = 0.20  # ~6.5 rad/s^2 on I≈0.031 kg m^2 — cartons never slip (μg/r >> that)
ALIGN_EXIT_DEG = 6.0  # servo exit band (inside the 10 deg align latch band)
W_EXIT = 0.08  # ... and wheel nearly still
EXIT_STREAK = 60  # consecutive steps inside the exit band before torque-off

# --- extraction drag (bang-bang velocity-regulated radial pull) ---
DRAG_F = 1.7  # N (slide threshold ~1.03 N; slide-not-tip asserted in the cfg)
DRAG_V = 0.10  # m/s regulated slide speed
EXTRACT_STOP = 0.24  # cut the drag once base-frame radius exceeds this (latch at 0.22)

HOVER_CLEAR = 0.030  # carton bottom above the basket rim at release


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.juice_carousel")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(n, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the statement on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def apply_wrench(body, force_w: torch.Tensor, torque_w: torch.Tensor) -> None:
        """WORLD-frame wrench at CoM. Measured on this forge build with a
        free-flight probe (no contacts, exact |v| = F*4dt/m): the DEFAULT
        set_external_force_and_torque call applies the given vectors in the
        BODY frame (applied_world = R_live * input; direction error tracked the
        body yaw exactly), while is_global=True resolves against a STALE frame
        and misdirects by up to ~170 deg state-dependently. So: pre-rotate the
        desired world wrench by the LIVE inverse root quat every call, and never
        pass is_global. Re-call every step so body rotation between calls does
        not drag the applied direction."""
        q = body.data.root_quat_w
        body.set_external_force_and_torque(
            _qapply(_qinv(q), force_w).view(n, 1, 3),
            _qapply(_qinv(q), torque_w).view(n, 1, 3), env_ids=all_ids)

    def hands_off() -> None:
        apply_wrench(scene.turntable, zero3, zero3)
        apply_wrench(scene.juice, zero3, zero3)

    def polar0() -> tuple[float, float, float]:
        r, az, z = scene.juice_polar()
        return float(r[0]), float(az[0]), float(z[0])

    def report(tag: str) -> None:
        r, az, z = polar0()
        print(f"[solve] {tag:12s} | juice r={r:.3f} az={math.degrees(az):+7.1f}deg "
              f"z={z:.3f} wheel={float(scene.wheel_speed()[0]):.3f}rad/s "
              f"in_basket={bool(scene.juice_in_basket()[0])} "
              f"milk_on={bool(scene.on_disc(scene.milk)[0])} "
              f"soda_on={bool(scene.on_disc(scene.soda)[0])} "
              f"align={bool(scene.align_latch[0])} extract={bool(scene.extract_latch[0])} "
              f"settled={bool(scene.all_settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    step(120)  # warmup window (60) + settle
    b_p = (scene.base.data.root_pos_w - scene.env_origins)[0]
    k_p = (scene.basket.data.root_pos_w - scene.env_origins)[0]
    yaw = 2.0 * math.atan2(float(scene.base.data.root_quat_w[0, 3]),
                           float(scene.base.data.root_quat_w[0, 0]))
    r0, az0, z0 = polar0()
    print(f"[solve] layout readback (seed {args.seed}): "
          f"base=({float(b_p[0]):+.3f},{float(b_p[1]):+.3f}) yaw={math.degrees(yaw):+.0f}deg "
          f"juice az={math.degrees(az0):+.1f}deg r={r0:.3f} "
          f"basket=({float(k_p[0]):+.3f},{float(k_p[1]):+.3f})", flush=True)
    report("reset")
    assert abs(az0) > math.radians(c.off_lo_deg) - math.radians(8.0), \
        f"juice must start misaligned, az={math.degrees(az0):.1f}deg"
    assert bool(scene.on_disc(scene.juice)[0]), "juice must start riding the disc"
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    s0 = print_score("P0 reset+settle (juice shrouded, misaligned)")
    assert s0 <= 0.05, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: rotate — torque servo on the carousel ------------------------
    streak, s_hi = 0, s0
    for i in range(3000):
        _r, az, _z = scene.juice_polar()
        w = scene.turntable.data.root_ang_vel_w[:, 2]
        w_des = (-K_POS * az).clamp(-W_CAP, W_CAP)
        tau = (K_VEL * (w_des - w)).clamp(-TAU_MAX, TAU_MAX)
        tq = torch.zeros(n, 3, device=device)
        tq[:, 2] = tau
        apply_wrench(scene.turntable, zero3, tq)
        step(1)
        if abs(float(az[0])) < math.radians(ALIGN_EXIT_DEG) and abs(float(w[0])) < W_EXIT:
            streak += 1
            if streak >= EXIT_STREAK:
                break
        else:
            streak = 0
        if i % 240 == 0:
            print(f"[solve] servo {i:4d}: az={math.degrees(float(az[0])):+7.1f}deg "
                  f"w={float(w[0]):+.3f}rad/s tau={float(tau[0]):+.3f}Nm", flush=True)
    hands_off()
    step(120)
    report("aligned")
    _r, az1, _z = polar0()
    assert abs(az1) < math.radians(c.align_deg), \
        f"servo failed to align, az={math.degrees(az1):.1f}deg"
    assert bool(scene.align_latch[0]), "align latch must be set after the servo hold"
    assert bool(scene.on_disc(scene.milk)[0]) and bool(scene.on_disc(scene.soda)[0]), \
        "distractors must still ride the disc after rotation"
    s1 = print_score("P1 carousel servoed to the window (crank-push torque, real dynamics)")
    assert s1 >= s0 - 1e-6, "score decreased across rotation"
    assert s1 >= 0.30, f"alignment credit missing, got {s1}"

    # ---------------- phase 2: extract — velocity-regulated radial drag ---------------------
    ex = torch.zeros(n, 3, device=device)
    ex[:, 0] = 1.0
    prog_r, prog_i = polar0()[0], 0
    for i in range(1800):
        u = _qapply(scene.base.data.root_quat_w, ex)
        u[:, 2] = 0.0
        u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        r_now = polar0()[0]
        if r_now > EXTRACT_STOP:
            break
        v_along = (scene.juice.data.root_lin_vel_w * u).sum(dim=-1)
        f = torch.where((v_along < DRAG_V).view(n, 1), DRAG_F * u, torch.zeros_like(u))
        apply_wrench(scene.juice, f, zero3)
        step(1)
        if i - prog_i >= 240:  # 2 s progress guard
            r_chk = polar0()[0]
            if not r_chk > prog_r + 0.010:
                report("drag-stall")
                q = scene.juice.data.root_quat_w[0]
                up = _qapply(scene.juice.data.root_quat_w,
                             torch.tensor([[0.0, 0.0, 1.0]], device=device))[0]
                print(f"[solve] stall diag: quat=({float(q[0]):+.3f},{float(q[1]):+.3f},"
                      f"{float(q[2]):+.3f},{float(q[3]):+.3f}) "
                      f"up_z={float(up[2]):+.3f}", flush=True)
                raise AssertionError(
                    f"extraction stalled at r={r_chk:.3f} (was {prog_r:.3f})")
            prog_r, prog_i = r_chk, i
        if i % 60 == 0:
            _rr, _aa, _zz = polar0()
            print(f"[solve] drag {i:4d}: r={_rr:.3f} az={math.degrees(_aa):+6.1f}deg "
                  f"z={_zz:.3f} v={float(v_along[0]):+.3f}", flush=True)
    hands_off()
    step(180)
    report("extracted")
    r2, _az2, z2 = polar0()
    assert r2 > c.extract_r, f"juice not outside the shroud, r={r2:.3f}"
    assert z2 < 0.30, f"juice should rest low on the porch, z={z2:.3f}"
    assert bool(scene.extract_latch[0]), "extract latch must be set"
    assert bool(scene.on_disc(scene.milk)[0]) and bool(scene.on_disc(scene.soda)[0]), \
        "distractors must still ride the disc after extraction"
    s2 = print_score("P2 juice dragged out through the window onto the porch (friction slide)")
    assert s2 >= s1 - 1e-6, "score decreased across extraction"
    assert s2 >= 0.60, f"extraction credit missing, got {s2}"

    # ---------------- phase 3: deliver — teleport carry + gravity drop ----------------------
    def teleport_hover() -> None:
        """The free-space carry: hover the carton above the basket mouth, yaw-aligned
        with the basket, zero velocity, hands off — gravity does the placement."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = scene.basket.data.root_pos_w[:, 0:2]
        st[:, 2] = scene.basket.data.root_pos_w[:, 2] + BASKET_T + RIM_H \
            + HOVER_CLEAR + ITEM_H / 2
        st[:, 3:7] = scene.basket.data.root_quat_w
        scene.juice.write_root_state_to_sim(st, all_ids)

    for attempt in range(3):
        teleport_hover()
        step(240)  # free fall ~10 cm + seat + settle (2 s)
        if bool(scene.juice_in_basket()[0]):
            break
        print(f"[solve] drop attempt {attempt}: not seated — retrying", flush=True)
    report("delivered")
    assert bool(scene.juice_in_basket()[0]), "juice failed to seat in the basket"
    s3 = print_score("P3 juice released over the basket mouth (transport + gravity)")
    assert s3 >= s2 - 1e-6, "score decreased across delivery"

    # ---------------- phase 4: success + persistence ---------------------------------------
    for _ in range(12):  # up to 2 s extra hands-off settling
        if bool(scene.success()[0]):
            break
        step(20)
    report("settled")
    s4 = print_score("P4 goal state: juice in basket, distractors riding, all still")
    assert s4 >= s3 - 1e-6, "score decreased across final settle"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after deliver)", flush=True)
        os._exit(1)

    hold, flickers = True, 0
    for i in range(420):  # 420 steps = 3.5 s at 120 Hz, hands off
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                print(f"[solve] persist flicker @step {i}: "
                      f"in_basket={bool(scene.juice_in_basket()[0])} "
                      f"milk_on={bool(scene.on_disc(scene.milk)[0])} "
                      f"soda_on={bool(scene.on_disc(scene.soda)[0])} "
                      f"settled={bool(scene.all_settled()[0])}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/420 steps", flush=True)
    report("persist")
    s5 = print_score("P-persist persistence 3.5 s (hands off, live state holds)")
    ok = hold and bool(scene.success()[0]) and s5 >= s4 - 1e-6
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

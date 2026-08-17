"""Smoke / rubric-REJECTION battery for PawlLadderScene (sim_gen task
`libero_kitchen_scene2_open_the_top_drawer_of_the_cabinet_i431`) — NullRobot,
force-driven probes + teleported constructs, RECORDED.

This is NOT a solution (solve.py — k lift-and-release metering cycles on the pawl
T-handle, no cart forces, no teleports — is the acceptance evidence; it passes on
forge seeds 0/1). Every probe here CONSTRUCTS a wrong (or partial, or
end-state-identical-but-illegal) outcome and asserts the rubric REJECTS it, or
proves a mechanism is real, not a prop. Construct teleports move the cart by exact
multiples of the notch pitch (gap-to-gap) or with the pawl held lifted, so no
construct ever interpenetrates the blade. No rejection probe ever reaches
success(); an audit check asserts exactly that, and a separate POSITIVE control
(after the audit) shows the cleanly-constructed goal state IS accepted, so the
battery is not vacuous.

   1. settle/dead-man   — reset settles finite; the DEAD-MAN is real: the cart
                          spawns a few mm uphill of the station-0 rest, glides,
                          and is ARRESTED by the seated pawl (readback q ~
                          q_rest(0)); ball aboard, everything settled;
   2. fresh reset       — score exactly ~0, no success (spawn arrest earns nothing);
   3. randomization A   — READBACK over 6 seeded resets: station yaw and xy jitter
                          are real (spreads asserted);
   4. randomization B   — target k varies (>= 2 distinct values over 6 seeds) AND
                          the GREEN/RED posts stand exactly on the fin-alignment
                          lines of stations k and k+1 recomputed from the station
                          pose readback (<= 3 mm);
   5. null policy       — 240 idle steps: cart still at the station-0 rest,
                          score ~0;
   6. SEED strategy     — the seed family's whole strategy (grab the sliding body
                          and haul on its DOF), done for real: 3.5 N on the cart
                          (3.2x the ramp-gravity load) downhill for 1.25 s, then
                          uphill — the seated two-way detent holds BOTH ways: the
                          cart rattles only within the notch's free play (bounded
                          by gap - blade_t, truncated by the q_lo joint stop; a
                          detent escape would read >= a full 50 mm pitch), no
                          arrival latches past station 0, no foul, re-arrested at
                          the station-0 rest, score ~0, NOT success;
   7. one-cycle near-miss — a REAL metering cycle with the solve's own pawl servo
                          (lift, release early, catch): the cart genuinely advances
                          one full notch to station 1; with k >= 2 this is NOT
                          success and score == 0.10 + 0.50/k exactly (latched);
   8. overshoot foul    — pawl held lifted, the cart free-runs past the target and
                          the FOUL latches (score -> 0); then the cart+ball are
                          put BACK at the exact target rest, pawl released, seated,
                          settled — END-STATE IDENTICAL to success, but the run is
                          illegal: success STILL False, score STILL 0 (flagship);
   9. pawl-held-up      — teleport-held construct: cart pinned at the target rest
                          every step, ball aboard, but the pawl held LIFTED — every
                          clause passes except pawl-seated -> NOT success (a blade
                          not in the notch cannot count), construct then parked
                          harmlessly at station 0;
  10. ball ejected      — cart+ball moved gap-to-gap to the target rest but the
                          ball placed on the floor: seated, still, at the green
                          post, yet NOT success (the rider clause is real);
  11. wrong station     — same clean construct one notch SHORT (station k-1):
                          seated, still, ball aboard — NOT success (the target is
                          the commanded green post, not any notch);
  12. settle gate       — the goal pose written WITH a real downhill velocity (a
                          zero-velocity teleport would leave the stillness counter
                          running — the teleport-vacuous trap): pose in band, pawl
                          seated, but moving -> NOT success; construct removed
                          gap-to-gap before it can settle;
  13. rejection audit   — success() was never True at ANY judged point above;
  14. positive control  — AFTER the audit: the exact goal state constructed
                          cleanly (cart gap-to-gap to the target, ball aboard,
                          pawl seated, settled) IS accepted: success True,
                          score 1.0 — the gates above reject for the right reasons;
  15. final no-NaN     — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene2_open_the_top_drawer_of_the_cabinet_i431.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version and silently rejects RTX -> the
# annotator returns EMPTY frames. Disable the driver check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math
import os
import threading

import numpy as np
import torch

import robobench
from robobench.core import ENVS

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:  # older isaaclab name
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

robobench.discover()
try:
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

G = 9.81
# The solve's own pawl-lift servo (KV*dt/m ~= 0.17).
P_KP, P_VCAP, P_KV = 8.0, 0.15, 3.0
P_FMIN, P_FMAX = -0.5, 6.0
LIFT_REF = 0.022
RELEASE_DELTA = 0.016
SHOVE_N = 3.5  # seed-strategy shove force on the cart (N)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.pawl_ladder_i431")().build(
        num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.95, -0.85, 0.65)) + o),
                                tuple(np.array((0.00, 0.00, 0.14)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video",
              flush=True)

    step_i = 0

    def step(m: int) -> None:
        nonlocal step_i
        for _ in range(m):
            env.step(no_action)
            if (annot is not None and step_i % args.record_every == 0
                    and len(frames) < args.max_frames):
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    ever_success = [False]
    audit_on = [True]

    def judge() -> tuple[float, bool]:
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        if audit_on[0]:
            ever_success[0] = ever_success[0] or ok
        return s, ok

    def q0() -> float:
        return float(scene.cart_q()[0])

    def pq0() -> float:
        return float(scene.pawl_q()[0])

    def kk() -> int:
        return int(scene.k[0])

    def station_yaw() -> float:
        ex = torch.zeros(n, 3, device=device)
        ex[:, 0] = 1.0
        x_w = scene_mod._qapply(scene.station.data.root_quat_w, ex)[0]
        return float(torch.atan2(x_w[1], x_w[0]))

    def report(tag: str) -> None:
        s, ok = judge()
        print(f"[smoke] {tag:16s} | q={q0():+.4f} pawl={pq0():+.4f} k={kk()} "
              f"| foul={float(scene.foul_latch[0]):.0f} "
              f"lift={float(scene.lift_latch[0]):.0f} "
              f"arrive={[int(v) for v in scene.arrive_latch[0].tolist()]} "
              f"| ball={bool(scene.ball_in_tray()[0])} "
              f"settled={bool(scene.settled()[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    # ----- probe actuators / constructors ---------------------------------------------------
    def pawl_lift(z_ref: float) -> None:
        pq = scene.pawl_q()
        v_body = quat_apply_inverse(scene.pawl.data.root_quat_w,
                                    scene.pawl.data.root_lin_vel_w)
        v_des = (P_KP * (z_ref - pq)).clamp(-P_VCAP, P_VCAP)
        fz = (c.pawl_mass * G * math.cos(c.pitch)
              + P_KV * (v_des - v_body[:, 2])).clamp(P_FMIN, P_FMAX)
        f = torch.zeros(n, 1, 3, device=device)
        f[:, 0, 2] = fz
        scene.pawl.set_external_force_and_torque(f, zero_w, env_ids=all_ids)

    def pawl_off() -> None:
        scene.pawl.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)

    def cart_shove(fx: float) -> None:
        """Body-frame x force on the cart (the cart never yaws vs the station)."""
        f = torch.zeros(n, 1, 3, device=device)
        f[:, 0, 0] = fx
        scene.cart.set_external_force_and_torque(f, zero_w, env_ids=all_ids)

    def cart_off() -> None:
        scene.cart.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)

    def place_cart(q: float, vx: float = 0.0, with_ball: bool = True) -> None:
        """Write the cart at joint position q along the station's CURRENT pose,
        optionally with the ball re-seated in its tray. `vx` writes a real
        rail-axis velocity (the teleport-vacuous trap needs moving constructs).
        Callers must keep teleports gap-to-gap or hold the pawl lifted."""
        p_st = scene.station.data.root_pos_w.clone()
        q_st = scene.station.data.root_quat_w.clone()
        ex = torch.zeros(n, 3, device=device)
        ex[:, 0] = 1.0
        v_w = scene_mod._qapply(q_st, ex) * float(vx)
        off = torch.zeros(n, 3, device=device)
        off[:, 0] = c.cart0_x + q
        off[:, 2] = c.axis_z
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = p_st + scene_mod._qapply(q_st, off)
        st[:, 3:7] = q_st
        st[:, 7:10] = v_w
        scene.cart.write_root_state_to_sim(st, all_ids)
        if with_ball:
            off[:, 0] = c.cart0_x + q + c.tray_x
            off[:, 1] = c.tray_y
            off[:, 2] = c.deck_top + c.ball_r + 0.001
            st = torch.zeros(n, 13, device=device)
            st[:, 0:3] = p_st + scene_mod._qapply(q_st, off)
            st[:, 3] = 1.0
            st[:, 7:10] = v_w
            scene.ball.write_root_state_to_sim(st, all_ids)

    def ball_to_floor() -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.env_origins + torch.tensor(
            [0.9, 0.9, c.ball_r + 0.001], device=device)
        st[:, 3] = 1.0
        scene.ball.write_root_state_to_sim(st, all_ids)

    def meter_one_notch(max_attempts: int = 5) -> bool:
        """One REAL metering cycle with the solve's own servo. True if advanced."""
        start = round((q0() - c.rest_off) / c.notch_pitch)
        for _att in range(max_attempts):
            q_start = q0()
            for _ in range(400):
                pawl_lift(LIFT_REF)
                env.step(no_action)
                if pq0() >= c.lift_min + 0.003:
                    break
            for _ in range(600):
                pawl_lift(LIFT_REF)
                env.step(no_action)
                if q0() - q_start >= RELEASE_DELTA:
                    break
            pawl_off()
            done = 0
            for _ in range(700):
                step(1)
                v = float(scene.cart.data.root_lin_vel_w[0].norm())
                done = done + 1 if (pq0() <= c.seat_tol and v < 0.03) else 0
                if done >= 25:
                    break
            now = round((q0() - c.rest_off) / c.notch_pitch)
            if now > start:
                return True
        return False

    bodies = (scene.station, scene.cart, scene.pawl, scene.ball)

    # =========================== 1-2. settle / dead-man / fresh =============================
    env.reset(seed=11)
    step(150)
    report("reset")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("settle: all states finite; the DEAD-MAN is real — the cart spawned "
          f"uphill of the rest, glided, and was ARRESTED by the seated pawl "
          f"(readback q={q0():+.4f} ~ q_rest(0)={c.q_rest_i(0):.4f}, "
          f"pawl={pq0():+.4f} seated), ball aboard, settled",
          fin and abs(q0() - c.q_rest_i(0)) <= 0.004 and pq0() <= c.seat_tol
          and bool(scene.ball_in_tray()[0]) and bool(scene.settled()[0]))
    s, ok = judge()
    check("fresh reset: score ~0 (the spawn arrest at station 0 earns nothing), "
          "no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(2)
        px, py = (float(v) for v in
                  (scene.station.data.root_pos_w[0, :2] - scene.env_origins[0, :2]))
        # expected GREEN/RED post positions recomputed from the station pose readback
        errs = []
        for body, idx in ((scene.post_g, kk()), (scene.post_r, kk() + 1)):
            off = torch.zeros(n, 3, device=device)
            off[:, 0] = c.cart0_x + c.q_rest_i(idx) + c.fin_x
            off[:, 1] = c.post_y
            off[:, 2] = 0.15
            w = scene.station.data.root_pos_w + scene_mod._qapply(
                scene.station.data.root_quat_w, off)
            errs.append(float(
                (body.data.root_pos_w[0, :2] - w[0, :2]).norm()))
        reads.append((station_yaw(), px, py, kk(), max(errs)))
        print(f"[smoke] seed {sd}: yaw={reads[-1][0]:+.2f} station=({px:+.3f},"
              f"{py:+.3f}) k={kk()} post_err={max(errs) * 1000:.1f}mm", flush=True)
    yaws = [r[0] for r in reads]
    xs = [r[1] for r in reads]
    ys = [r[2] for r in reads]
    ks = [r[3] for r in reads]
    perr = max(r[4] for r in reads)
    check("randomization A: station yaw varies (readback spread "
          f"{max(yaws) - min(yaws):.2f} rad) and xy jitter is real (x spread "
          f"{(max(xs) - min(xs)) * 1000:.0f} mm, y spread "
          f"{(max(ys) - min(ys)) * 1000:.0f} mm)",
          max(yaws) - min(yaws) > 1.5 and max(xs) - min(xs) > 0.01
          and max(ys) - min(ys) > 0.01)
    check("randomization B: the commanded station k varies over seeds (readback "
          f"{ks}, >= 2 distinct in {{2,3,4}}) and BOTH marker posts stand on the "
          f"fin-alignment lines recomputed from the station pose (max err "
          f"{perr * 1000:.1f} mm <= 3 mm)",
          len(set(ks)) >= 2 and all(2 <= v <= 4 for v in ks) and perr <= 0.003)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: 240 idle steps — the cart is still detent-held at the "
          f"station-0 rest (q={q0():+.4f}), score ~0, no success",
          abs(q0() - c.q_rest_i(0)) <= 0.004 and s <= 0.02 and not ok)

    # =========================== 6. SEED strategy is blocked ================================
    env.reset(seed=41)
    step(120)
    q_rest0 = q0()
    dev_max = 0.0
    for fx in (SHOVE_N, -SHOVE_N):  # downhill haul, then uphill haul
        for _ in range(150):
            cart_shove(fx)
            env.step(no_action)
            dev_max = max(dev_max, abs(q0() - q_rest0))
    cart_off()
    step(90)
    report("seed-shove")
    s, ok = judge()
    # Honest peak bound: the notch has real free play (gap 30 - blade 16 = 14 mm),
    # truncated uphill by the joint stop at q_lo — so the cart CAN rattle up to
    # (q_rest0 - q_lo) without the detent yielding. Escape would read >= a full
    # notch pitch. The load-bearing claims: never left the notch's capture basin,
    # no foul, no arrival latched beyond station 0, and the cart glided BACK and
    # was re-arrested at the station-0 rest with the pawl still seated.
    play_cap = (q_rest0 - c.q_lo) + 0.002
    check("SEED strategy (grab the slider and haul on its DOF): "
          f"{SHOVE_N:.1f} N on the cart — {SHOVE_N / (c.cart_mass * G * math.sin(c.pitch)):.1f}x "
          f"the ramp-gravity load — for 1.25 s EACH WAY: the seated two-way detent "
          f"holds both directions (peak |q - rest| = {dev_max * 1000:.1f} mm <= "
          f"in-notch free play {play_cap * 1000:.1f} mm << notch pitch "
          f"{c.notch_pitch * 1000:.0f} mm), no foul, no arrival past station 0, "
          f"re-arrested at the rest (q={q0():+.4f}), pawl seated (pq={pq0():+.4f}), "
          "score ~0, NOT success",
          dev_max <= play_cap and float(scene.foul_latch[0]) < 0.5
          and not bool(scene.arrive_latch[0, 1:].any())
          and abs(q0() - q_rest0) <= 0.004
          and pq0() <= c.seat_tol and s <= 0.02 and not ok)

    # =========================== 7. one-cycle near-miss =====================================
    env.reset(seed=51)
    step(120)
    k = kk()
    advanced = meter_one_notch()
    step(120)
    report("near-miss")
    s, ok = judge()
    expected = 0.10 + 0.50 / k
    check("one-cycle near-miss: a REAL metering cycle (the solve's own pawl servo) "
          f"genuinely advanced the cart one notch (q={q0():+.4f} ~ "
          f"q_rest(1)={c.q_rest_i(1):.4f}, pawl re-seated) — but the commanded "
          f"station is k={k}, so NOT success, and the latched score is exactly "
          f"0.10 + 0.50/k = {expected:.3f} (got {s:.3f})",
          advanced and abs(q0() - c.q_rest_i(1)) <= c.q_tol
          and pq0() <= c.seat_tol and k >= 2 and not ok
          and abs(s - expected) <= 0.012)

    # =========================== 8. overshoot foul (flagship) ===============================
    # Same env, continued: hold the pawl lifted and let the cart FREE-RUN past the
    # target — the foul latches. Then rebuild the EXACT success end-state.
    q_target = float(scene.q_target()[0])
    for _ in range(900):
        pawl_lift(LIFT_REF)
        env.step(no_action)
        if q0() > q_target + c.overshoot + 0.015:
            break
    foul_fired = float(scene.foul_latch[0]) > 0.5
    s_foul, _ok = judge()
    # rebuild: cart+ball back at the exact target rest (pawl still lifted -> the
    # sweep is safe), then PIN the cart there (pose+v=0 rewritten every step)
    # while the pawl drops — otherwise the cart gravity-runs ~4.5 mm during the
    # ~50 ms blade fall, exceeds the 4 mm notch play, and the blade rides one
    # notch further (seen on forge: ended at rest k+1, not k)
    place_cart(c.q_rest_i(k))
    pawl_off()
    for _ in range(15):
        place_cart(c.q_rest_i(k))
        env.step(no_action)
    step(180)
    report("foul-rebuilt")
    s, ok = judge()
    check("overshoot foul (FLAGSHIP): pawl held up, the cart free-ran past the "
          f"target (foul latched={foul_fired}, score at foul {s_foul:.2f}); then "
          f"the cart+ball were put BACK at the exact target rest and the pawl "
          f"released and seated (q={q0():+.4f} ~ {c.q_rest_i(k):.4f}, "
          f"pawl={pq0():+.4f}, ball={bool(scene.ball_in_tray()[0])}, "
          f"settled={bool(scene.settled()[0])}) — END-STATE IDENTICAL to success, "
          "but the run is illegal: success STILL False, score STILL 0",
          foul_fired and s_foul <= 0.001
          and abs(q0() - c.q_rest_i(k)) <= c.q_tol and pq0() <= c.seat_tol
          and bool(scene.ball_in_tray()[0]) and bool(scene.settled()[0])
          and not ok and s <= 0.001)

    # =========================== 9. pawl-held-up refusal ====================================
    env.reset(seed=61)
    step(120)
    k = kk()
    # PRE-LIFT the pawl (cart pinned at station 0 so nothing runs): the servo
    # needs ~15 steps to raise the blade, and pinning the cart at the TARGET
    # while the pawl is still seated is literally the genuine success state
    # (seen on forge: the audit correctly caught it) — lift first, then pin.
    lifted = False
    for _ in range(200):
        pawl_lift(LIFT_REF)
        place_cart(c.q_rest_i(0))
        env.step(no_action)
        if pq0() >= c.lift_min + 0.003:
            lifted = True
            break
    assert lifted, f"pawl pre-lift failed (pq={pq0():+.4f})"
    never_ok = True
    for _ in range(60):  # teleport-hold: cart pinned at the target, pawl held UP
        pawl_lift(LIFT_REF)
        place_cart(c.q_rest_i(k))
        env.step(no_action)
        never_ok = never_ok and not judge()[1]
    q_pin = q0()
    in_band = abs(q_pin - c.q_rest_i(k)) <= c.q_tol
    pq_held = pq0()
    ball_ok = bool(scene.ball_in_tray()[0])
    report("pawl-held-up")
    # park at station 0: pin the cart there while the pawl drops and seats
    # (same anti-drift pattern as the foul rebuild above)
    place_cart(c.q_rest_i(0))
    pawl_off()
    for _ in range(15):
        place_cart(c.q_rest_i(0))
        env.step(no_action)
    step(120)
    check("pawl-held-up: cart pinned at the target rest every step "
          f"(q={q_pin:+.4f} in band={in_band}), ball aboard={ball_ok}, but the pawl "
          f"held LIFTED (pq={pq_held:+.4f} > seat_tol {c.seat_tol}) — every clause "
          "passes except pawl-seated: success never fired in 60 held steps",
          in_band and ball_ok and pq_held > c.lift_min - 0.004 and never_ok)

    # =========================== 10. ball ejected ===========================================
    env.reset(seed=71)
    step(120)
    k = kk()
    place_cart(c.q_rest_i(k), with_ball=False)  # gap-to-gap from station 0
    ball_to_floor()
    step(180)
    report("ball-ejected")
    s, ok = judge()
    check("ball ejected: cart moved gap-to-gap to the target rest "
          f"(q={q0():+.4f}, pawl={pq0():+.4f} seated, "
          f"settled={bool(scene.settled()[0])}) but the ball sits on the floor "
          f"(in_tray={bool(scene.ball_in_tray()[0])}) — NOT success (the rider "
          "clause is real), score <= 0.30",
          abs(q0() - c.q_rest_i(k)) <= c.q_tol and pq0() <= c.seat_tol
          and bool(scene.settled()[0]) and not bool(scene.ball_in_tray()[0])
          and not ok and s <= 0.30)

    # =========================== 11. wrong station ==========================================
    env.reset(seed=81)
    step(120)
    k = kk()
    place_cart(c.q_rest_i(k - 1))  # one notch SHORT, gap-to-gap
    step(180)
    report("wrong-station")
    s, ok = judge()
    check("wrong station: the same clean construct one notch SHORT "
          f"(q={q0():+.4f} ~ q_rest({k - 1})={c.q_rest_i(k - 1):.4f}, pawl seated, "
          f"ball aboard, settled) — NOT success (the commanded stop is station "
          f"{k}), score <= 0.30",
          abs(q0() - c.q_rest_i(k - 1)) <= c.q_tol and pq0() <= c.seat_tol
          and bool(scene.ball_in_tray()[0]) and bool(scene.settled()[0])
          and not ok and s <= 0.30)

    # =========================== 12. settle gate ============================================
    env.reset(seed=91)
    step(120)
    k = kk()
    # goal pose written WITH a real downhill velocity: pose in band, pawl in the
    # notch, but MOVING -> the stillness streak is broken
    place_cart(c.q_rest_i(k) - 0.006, vx=0.12)
    step(2)
    q_now, v_now = q0(), float(scene.cart.data.root_lin_vel_w[0].norm())
    settled_now = bool(scene.settled()[0])
    s, ok = judge()
    report("settle-gate")
    # remove the construct gap-to-gap BEFORE it can settle into genuine success
    place_cart(q0() - k * c.notch_pitch)
    step(120)
    check("settle gate: the goal pose written WITH a real downhill velocity "
          f"(q={q_now:+.4f} in band, cart speed readback {v_now:.3f} > settle_lin "
          f"{c.settle_lin}) — pose passes but stillness has not persisted "
          f"(settled={settled_now}) -> NOT success; construct removed before "
          "ring-down",
          abs(q_now - c.q_rest_i(k)) <= c.q_tol and v_now > c.settle_lin
          and not settled_now and not ok)

    # =========================== 13. rejection audit ========================================
    check("rejection audit: success() was never True at any judged point in the "
          "battery above", not ever_success[0])

    # =========================== 14. positive control (after the audit) =====================
    audit_on[0] = False
    env.reset(seed=101)
    step(120)
    k = kk()
    place_cart(c.q_rest_i(k))  # gap-to-gap, ball re-seated, pawl drops in place
    step(240)
    report("positive-ctrl")
    s, ok = judge()
    check("positive control: the exact goal state constructed cleanly (cart "
          f"gap-to-gap to station {k}, ball aboard, pawl seated "
          f"(pq={pq0():+.4f}), settled) IS accepted: success True, score 1.0 — "
          "the rejections above fire for the right reasons, not vacuously",
          ok and s >= 0.999)

    # =========================== 15. final no-NaN ===========================================
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.pawl_ladder_i431")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(okc for _nm, okc in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, okc in checks:
            if not okc:
                print(f"[smoke]   FAILED: {nm}", flush=True)
    code = 0 if all_ok else 1
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
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)

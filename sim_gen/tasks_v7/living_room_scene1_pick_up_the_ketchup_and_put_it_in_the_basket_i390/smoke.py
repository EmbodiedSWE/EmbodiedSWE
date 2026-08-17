"""Smoke / rubric-REJECTION battery for PinnedHatchDeliveryScene (sim_gen task
`living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket_i390`) — NullRobot,
force-driven probes + teleported constructs, RECORDED.

This is NOT a solution (solve.py — push the ketchup through the propped window so it
drops into the basket, then pull the pin so the sash slams shut — is the acceptance
evidence; it passes on forge seeds 0/1/2). Every probe here CONSTRUCTS a wrong (or
partial) outcome and asserts the rubric REJECTS it, or proves a mechanism is real,
not a prop. Force probes use the solve's own regulated controllers (constant-force
probes walk latches) and assert the actuator actually moved (no vacuous rejections).
No probe in this battery ever reaches success(), and a final audit asserts exactly
that.

   1. settle/no-NaN     — reset settles finite; the PROP is real: the sash rests ON
                          the pin shaft at the prop height (readback), pin engaged in
                          the wall bore; score ~0, no success;
   2. randomization A   — READBACK over 6 seeded resets: housing yaw and xy jitter
                          are real (spreads asserted);
   3. randomization B   — bottle slots Bernoulli-swap (both sides seen, bottles
                          ALWAYS opposite), continuous per-bottle jitter, pin
                          insertion-depth jitter (readback spreads);
   4. null policy       — 240 idle steps: the pin holds, the sash stays propped at
                          height (readback), score ~0, no success;
   5. SEED strategy     — the seed's whole plan (put the ketchup in the basket, stop)
                          done FOR REAL with the solve's own push servo: the bottle
                          drops in and rests contained — 0.25 delivery credit only,
                          NOT success (the window is still open);
   6. latch regression  — the delivered ketchup yanked back OUT to the apron:
                          containment is now False but the 0.25 latch persists;
   7. sealed-but-empty  — with the ketchup OUT, the pin pulled for real (the solve's
                          escalating axial pull): the sash slams shut (readback), the
                          seal latch fires but the SIMULTANEOUS latch stays 0 —
                          score 0.40, NOT success (basket empty);
   8. pin-first trap    — fresh episode, pin pulled FIRST: sash slams; then a REAL
                          delivery attempt (same push servo, full budget): the bottle
                          is WALLED by the closed sash (readback: it stalls at the
                          plate, moved >6 cm so the block is non-vacuous), never
                          contained — score <= 0.151, NOT success: the order is
                          physically forced;
   9. no re-insertion   — the pin staged back in the guide bore aimed at the closed
                          channel and pushed axially (escalating, velocity-capped):
                          the tip contacts the sash plate and STALLS (moved >= 4 mm,
                          never past the plate plane, far short of bore engagement),
                          the sash never lifts: the one-shot resource is really
                          one-shot;
  10. wrong bottle      — the BROWN bbq bottle delivered through the open window for
                          real (same servo): it rests contained in the basket — NO
                          credit (score ~0), NOT success;
  11. near-miss sill    — ketchup standing in the WINDOW on the sill, hugging the
                          axis, centimetres from the basket: not contained, no latch;
  12. near-miss roof    — ketchup resting on the cabinet ROOF directly above the
                          basket (xy inside the gates): the z band rejects it;
  13. exclusion + cap   — constructed sealed cabinet with BOTH bottles resting in
                          the basket: all three latches fire, score pinned at the
                          0.55 cap, NEVER success while the bbq is inside;
  14. settle gate       — the goal pose written WHILE MOVING (a real downward
                          velocity on the ketchup — a zero-velocity teleport would
                          leave the stillness counter running, the teleport-vacuous
                          trap): every pose gate passes but stillness has not
                          persisted -> NOT success; construct removed before it can
                          settle;
  15. rejection audit   — success() was never True at ANY judged point;
  16. final no-NaN      — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.<task>.smoke --headless
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

import os
import threading

import numpy as np
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
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# The solve's own controllers (constant-force probes walk latches).
P_VDES, P_KV, P_FMAX = 0.25, 4.0, 1.6   # bottle push servo (below the tip threshold)
P_KY, P_FYMAX = 2.0, 0.5                # lane keeping
X_F0, X_FSTEP, X_FMAX = 0.8, 0.6, 6.0   # pin pull: escalating, velocity-capped
X_VCAP, X_TIP_STOP = 0.30, -0.180


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
    sm = scene_mod

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.95, -0.60, 0.62)) + o),
                                tuple(np.array((-0.05, 0.02, 0.22)) + o),
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

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
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

    def judge() -> tuple[float, bool]:
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return s, ok

    def hframe(body) -> tuple[torch.Tensor, torch.Tensor]:
        qh = scene.housing.data.root_quat_w
        p = quat_apply_inverse(qh, body.data.root_pos_w - scene.housing.data.root_pos_w)
        v = quat_apply_inverse(qh, body.data.root_lin_vel_w)
        return p, v

    def push_h(body, f_h: torch.Tensor) -> None:
        f_w = quat_apply(scene.housing.data.root_quat_w, f_h)
        f_b = quat_apply_inverse(body.data.root_quat_w, f_w)
        body.set_external_force_and_torque(f_b.unsqueeze(1), zero_w, env_ids=all_ids)

    def wrenches_off() -> None:
        for b in (scene.ketchup, scene.bbq, scene.pin):
            b.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)

    def housing_yaw() -> float:
        ex = torch.zeros(n, 3, device=device)
        ex[:, 0] = 1.0
        x_w = quat_apply(scene.housing.data.root_quat_w, ex)[0]
        return float(torch.atan2(x_w[1], x_w[0]))

    def q0() -> float:
        return float(scene.sash_q()[0])

    def tip0() -> float:
        return float(scene.pin_tip_x()[0])

    def report(tag: str) -> None:
        s, ok = judge()
        kp, _ = hframe(scene.ketchup)
        print(f"[smoke] {tag:16s} | q={q0():+.4f} tip={tip0():+.4f} "
              f"k=({float(kp[0, 0]):+.3f},{float(kp[0, 1]):+.3f},{float(kp[0, 2]):+.3f}) "
              f"in={bool(scene._contained(scene.ketchup)[0])} "
              f"sealed={bool(scene.sealed()[0])} "
              f"latch i/s/f={float(scene.in_ever[0]):.0f}/{float(scene.seal_ever[0]):.0f}/"
              f"{float(scene.full_ever[0]):.0f} settled={bool(scene.settled()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    # ----- teleport helper: pose written in the housing's CURRENT frame --------------------
    def place(body, off_xyz, vel_h=(0.0, 0.0, 0.0)) -> None:
        off = torch.zeros(n, 3, device=device)
        off[:, 0], off[:, 1], off[:, 2] = off_xyz
        vh = torch.zeros(n, 3, device=device)
        vh[:, 0], vh[:, 1], vh[:, 2] = vel_h
        qh = scene.housing.data.root_quat_w
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.housing.data.root_pos_w + quat_apply(qh, off)
        st[:, 3:7] = qh
        st[:, 7:10] = quat_apply(qh, vh)
        body.write_root_state_to_sim(st, all_ids)

    # ----- probe actuators: the solve's own controllers ------------------------------------
    def push_bottle(body, max_steps: int, stop_over_sill: bool = True) -> float:
        """Velocity-regulated axial push along housing +x with lane keeping.
        Returns the max housing-local x reached. Wrenches left OFF at return."""
        x_max = -1.0
        for _ in range(max_steps):
            kp, kv = hframe(body)
            f_h = torch.zeros(n, 3, device=device)
            f_h[:, 0] = (P_KV * (P_VDES - kv[:, 0])).clamp(-P_FMAX, P_FMAX)
            vy_des = (P_KY * (0.0 - kp[:, 1])).clamp(-0.10, 0.10)
            f_h[:, 1] = (P_KV * (vy_des - kv[:, 1])).clamp(-P_FYMAX, P_FYMAX)
            push_h(body, f_h)
            env.step(no_action)
            x, z = float(kp[0, 0]), float(kp[0, 2])
            x_max = max(x_max, x)
            if stop_over_sill and (x > -0.105 or z < 0.155):
                break
        wrenches_off()
        return x_max

    def pull_pin(max_steps: int = 1500) -> float:
        """The solve's escalating velocity-capped axial pull. Returns final tip x."""
        f_lvl, last_tip, stall = X_F0, tip0(), 0
        for _ in range(max_steps):
            tip = tip0()
            if tip <= X_TIP_STOP:
                break
            _, pv = hframe(scene.pin)
            f_h = torch.zeros(n, 3, device=device)
            pulling = pv[:, 0] > -X_VCAP
            f_h[:, 0] = torch.where(pulling, -f_lvl * torch.ones_like(pv[:, 0]),
                                    torch.zeros_like(pv[:, 0]))
            push_h(scene.pin, f_h)
            env.step(no_action)
            stall += 1
            if stall >= 60:
                if last_tip - tip < 0.002:
                    f_lvl = min(f_lvl + X_FSTEP, X_FMAX)
                last_tip, stall = tip, 0
        wrenches_off()
        return tip0()

    def settle_until(cond, max_steps: int, need: int = 30) -> bool:
        got = 0
        for _ in range(max_steps):
            step(1)
            got = got + 1 if bool(cond()) else 0
            if got >= need:
                return True
        return False

    # =========================== 1. settle / the prop is real ===============================
    env.reset(seed=11)
    step(150)
    report("reset")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in scene._bodies())
    s, ok = judge()
    check("settle: all states finite; the PROP is real — the sash rests ON the pin "
          f"shaft at the prop height (readback q={q0():+.4f} ~ {c.q_spawn:+.4f}, far "
          f"above closed_tol {c.closed_tol:.3f}), pin engaged in the wall bore "
          f"(tip={tip0():+.4f} >= -0.130), everything settled; score ~0, no success",
          fin and 0.150 <= q0() <= c.q_spawn + 0.005 and tip0() >= -0.130
          and bool(scene.settled()[0]) and s <= 0.02 and not ok)

    # =========================== 2-3. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(2)  # refresh buffers; nothing moves >1 mm in 2 steps from rest
        px, py = (float(v) for v in
                  (scene.housing.data.root_pos_w[0, :2] - scene.env_origins[0, :2]))
        kp, _ = hframe(scene.ketchup)
        bp, _ = hframe(scene.bbq)
        reads.append((housing_yaw(), px, py, float(kp[0, 1]), float(bp[0, 1]),
                      float(kp[0, 0]), tip0()))
        print(f"[smoke] seed {sd}: yaw={reads[-1][0]:+.3f} housing=({px:+.3f},{py:+.3f}) "
              f"ket_y={reads[-1][3]:+.3f} bbq_y={reads[-1][4]:+.3f} "
              f"ket_x={reads[-1][5]:+.3f} tip={reads[-1][6]:+.4f}", flush=True)
    yaws, xs, ys = [r[0] for r in reads], [r[1] for r in reads], [r[2] for r in reads]
    check("randomization A: housing yaw varies (readback spread "
          f"{max(yaws) - min(yaws):.3f} rad) and xy jitter is real (x spread "
          f"{(max(xs) - min(xs)) * 1000:.0f} mm, y spread "
          f"{(max(ys) - min(ys)) * 1000:.0f} mm)",
          max(yaws) - min(yaws) > 0.10 and max(xs) - min(xs) > 0.010
          and max(ys) - min(ys) > 0.010)
    kys, bys = [r[3] for r in reads], [r[4] for r in reads]
    kxs, tips = [r[5] for r in reads], [r[6] for r in reads]
    check("randomization B: bottle slots Bernoulli-swap — ketchup seen on BOTH sides "
          f"(readback y {[f'{v:+.2f}' for v in kys]}), bottles ALWAYS opposite; "
          f"continuous per-bottle jitter (ket x spread "
          f"{(max(kxs) - min(kxs)) * 1000:.0f} mm) and pin depth jitter (tip spread "
          f"{(max(tips) - min(tips)) * 1000:.1f} mm)",
          max(kys) > 0.03 and min(kys) < -0.03
          and all(ky * by < 0 for ky, by in zip(kys, bys))
          and max(kxs) - min(kxs) > 0.004 and max(tips) - min(tips) > 0.0015)

    # =========================== 4. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: 240 idle steps — the pin holds, the sash stays propped at "
          f"height (readback q={q0():+.4f} >= 0.150), score ~0, no success",
          q0() >= 0.150 and s <= 0.02 and not ok)

    # =========================== 5. SEED strategy: delivery alone caps at 0.25 ==============
    # (continues the seed-31 episode: everything still at spawn)
    place(scene.ketchup, (-0.26, 0.0, 0.185))
    step(30)
    x_max = push_bottle(scene.ketchup, 900)
    settle_until(lambda: scene._contained(scene.ketchup)[0]
                 & (scene.still_count[0] >= 20), 600)
    report("seed-strategy")
    s, ok = judge()
    check("SEED strategy (put the ketchup in the basket, stop), done FOR REAL with "
          "the solve's own push servo: the bottle dropped in and rests contained "
          f"(readback), the delivery latch fired — score {s:.3f} ~ 0.25 ONLY, NOT "
          "success (the window is still open, the cabinet unsealed)",
          bool(scene._contained(scene.ketchup)[0])
          and float(scene.in_ever[0]) > 0.5 and float(scene.seal_ever[0]) < 0.5
          and 0.249 <= s <= 0.2501 and not ok)

    # =========================== 6. latch regression ========================================
    place(scene.ketchup, (-0.26, 0.06, 0.185))
    step(90)
    report("yank-out")
    s, ok = judge()
    check("latch regression: the delivered ketchup yanked back OUT to the apron — "
          f"containment is now False but the 0.25 latch persists (score {s:.3f}), "
          "no success",
          not bool(scene._contained(scene.ketchup)[0])
          and float(scene.in_ever[0]) > 0.5 and 0.249 <= s <= 0.2501 and not ok)

    # =========================== 7. sealed-but-empty + simultaneity =========================
    tip = pull_pin()
    settle_until(lambda: (scene.sash_q()[0] <= c.closed_tol)
                 & (scene.still_count[0] >= 20), 600)
    report("sealed-empty")
    s, ok = judge()
    check("sealed-but-empty: with the ketchup OUT, the pin pulled for real "
          f"(tip={tip:+.4f} <= clear gate {c.pin_clear_x:+.3f}) — the sash slammed "
          f"shut (readback q={q0():+.4f} <= {c.closed_tol:.3f}), the seal latch "
          "fired but the SIMULTANEOUS latch stayed 0 (delivery and seal never held "
          f"at once): score {s:.3f} = 0.40, NOT success (basket empty)",
          tip <= c.pin_clear_x and q0() <= c.closed_tol
          and float(scene.seal_ever[0]) > 0.5 and float(scene.full_ever[0]) < 0.5
          and 0.399 <= s <= 0.4001 and not ok)

    # =========================== 8. pin-first: the order is physically forced ===============
    env.reset(seed=41)
    step(60)
    tip = pull_pin()
    settle_until(lambda: (scene.sash_q()[0] <= c.closed_tol)
                 & (scene.still_count[0] >= 20), 600)
    q_shut = q0()
    place(scene.ketchup, (-0.26, 0.0, 0.185))
    step(30)
    x_max = push_bottle(scene.ketchup, 450, stop_over_sill=False)
    step(90)
    report("pin-first")
    s, ok = judge()
    check("pin-first trap: pin pulled FIRST -> sash slammed "
          f"(q={q_shut:+.4f}); then a REAL delivery attempt with the same push "
          f"servo — the bottle moved {(x_max + 0.26) * 1000:.0f} mm (non-vacuous) "
          f"but was WALLED by the closed sash (readback x_max={x_max:+.4f}, stalls "
          "at the plate, never past -0.170), never contained: score <= 0.151, NOT "
          "success — deliver-then-seal is enforced by the physics, not a script",
          q_shut <= c.closed_tol and x_max + 0.26 > 0.060 and x_max <= -0.170
          and not bool(scene._contained(scene.ketchup)[0])
          and float(scene.in_ever[0]) < 0.5 and s <= 0.151 and not ok)

    # =========================== 9. the pin can NEVER go back ===============================
    # Stage the pin back in the guide bore, tip just short of the (now sash-filled)
    # channel, and press it axially inward HELD LEVEL on the bore axis (y/z velocity
    # servo + gravity feed-forward — the staged pin's CoM overhangs the guide
    # entrance, so a free push would tip knob-down; probes press held, never drop).
    z_axis = sm.BORE_Z - sm.BORE_H + sm.SHAFT_R + 0.001
    place(scene.pin, (-0.262, sm.BORE_Y, z_axis))
    step(2)  # refresh buffers so tip_start reads the STAGED pose, not the stale one
    tip_start = tip0()
    tip_max, f_lvl, last_tip, stall = tip_start, X_F0, tip_start, 0
    H_KV, H_KP = 1.0, 2.0  # KV*dt/m = 1/(120*0.06) ~= 0.14 (wrench-delay bound)
    for _ in range(400):
        tip = tip0()
        pp, pv = hframe(scene.pin)
        f_h = torch.zeros(n, 3, device=device)
        pushing = pv[:, 0] < X_VCAP
        f_h[:, 0] = torch.where(pushing, f_lvl * torch.ones_like(pv[:, 0]),
                                torch.zeros_like(pv[:, 0]))
        vy = (H_KP * (sm.BORE_Y - pp[:, 1])).clamp(-0.2, 0.2)
        vz = (H_KP * (z_axis - pp[:, 2])).clamp(-0.2, 0.2)
        f_h[:, 1] = (H_KV * (vy - pv[:, 1])).clamp(-1.5, 1.5)
        f_h[:, 2] = (c.pin_mass * 9.81 + H_KV * (vz - pv[:, 2])).clamp(-1.5, 1.5)
        push_h(scene.pin, f_h)
        env.step(no_action)
        tip_max = max(tip_max, tip0())
        stall += 1
        if stall >= 60:
            if tip0() - last_tip < 0.002:
                f_lvl = min(f_lvl + X_FSTEP, 4.0)
            last_tip, stall = tip0(), 0
    wrenches_off()
    step(60)
    report("re-insert")
    s, ok = judge()
    check("no re-insertion: the pin staged in the guide bore and pushed axially at "
          f"the closed channel (escalated to {f_lvl:.1f} N) — the tip advanced "
          f"{(tip_max - tip_start) * 1000:.0f} mm into contact (>= 4 mm, "
          "non-vacuous) then STALLED on the solid sash plate (readback "
          f"tip_max={tip_max:+.4f}: never past the plate at -0.144, far short of "
          "bore engagement -0.120), and the sash never lifted "
          f"(q={q0():+.4f} <= closed_tol): the one-shot resource is really one-shot",
          tip_max - tip_start >= 0.004 and tip_max <= -0.144
          and q0() <= c.closed_tol and not ok)

    # =========================== 10. wrong bottle earns nothing =============================
    env.reset(seed=51)
    step(60)
    place(scene.bbq, (-0.26, 0.0, 0.185))
    step(30)
    push_bottle(scene.bbq, 900)
    settle_until(lambda: scene._contained(scene.bbq)[0]
                 & (scene.still_count[0] >= 20), 600)
    report("wrong-bottle")
    s, ok = judge()
    check("wrong bottle: the BROWN bbq bottle delivered through the open window for "
          "real (same servo) — it rests contained in the basket (readback) but earns "
          f"NOTHING (score {s:.3f} ~ 0), and success requires it OUT: NOT success",
          bool(scene._contained(scene.bbq)[0])
          and float(scene.in_ever[0]) < 0.5 and s <= 0.02 and not ok)

    # =========================== 11-12. near-misses =========================================
    env.reset(seed=61)
    step(60)
    place(scene.ketchup, (-0.1425, 0.0, sm.APRON_TOP + 0.003))
    step(90)
    report("near-miss-sill")
    s, ok = judge()
    check("near-miss: ketchup standing IN the window on the sill, hugging the axis, "
          "centimetres from the basket — not contained (basket-frame x gate), no "
          f"latch, score {s:.3f} ~ 0",
          not bool(scene._contained(scene.ketchup)[0])
          and float(scene.in_ever[0]) < 0.5 and s <= 0.02 and not ok)
    place(scene.ketchup, (sm.BASKET_X, 0.0, sm.ROOF_Z1 + 0.002))
    step(90)
    report("near-miss-roof")
    s, ok = judge()
    check("near-miss: ketchup resting on the cabinet ROOF directly above the basket "
          "(xy inside the gates) — the z band rejects it, no latch, score ~0",
          not bool(scene._contained(scene.ketchup)[0])
          and float(scene.in_ever[0]) < 0.5 and s <= 0.02 and not ok)

    # =========================== 13. exclusion + the 0.55 cap ===============================
    env.reset(seed=71)
    step(60)
    place(scene.sash, (sm.SASH_CX, sm.SASH_CY, sm.SASH_CZ0 + 0.001))
    place(scene.pin, (-0.29, 0.14, sm.APRON_TOP + 0.017))
    zk = sm.FLOOR_T + 0.002 + sm.BASKET_FT + 0.002
    place(scene.ketchup, (sm.BASKET_X - 0.045, 0.0, zk))
    place(scene.bbq, (sm.BASKET_X + 0.040, 0.0, zk))
    step(150)
    report("exclusion")
    s, ok = judge()
    check("exclusion + cap: constructed sealed cabinet with BOTH bottles resting in "
          "the basket — delivery, seal and simultaneous latches all fired, score "
          f"pinned at the 0.55 cap ({s:.4f}), NEVER success while the bbq is inside",
          bool(scene._contained(scene.ketchup)[0]) and bool(scene._contained(scene.bbq)[0])
          and bool(scene.sealed()[0]) and float(scene.full_ever[0]) > 0.5
          and 0.549 <= s <= 0.551 and not ok)

    # =========================== 14. settle gate ============================================
    env.reset(seed=81)
    step(60)
    place(scene.sash, (sm.SASH_CX, sm.SASH_CY, sm.SASH_CZ0 + 0.001))
    place(scene.pin, (-0.29, 0.14, sm.APRON_TOP + 0.017))
    step(30)
    # write the ketchup INTO the containment band WITH a real downward velocity
    place(scene.ketchup, (sm.BASKET_X, 0.0, zk + 0.016), vel_h=(0.0, 0.0, -0.35))
    step(1)
    in_now = bool(scene._contained(scene.ketchup)[0])
    sealed_now = bool(scene.sealed()[0])
    v_now = float(scene.ketchup.data.root_lin_vel_w[0].norm())
    settled_now = bool(scene.settled()[0])
    s, ok = judge()
    report("settle-gate")
    # remove the construct BEFORE it can settle (the battery must never succeed)
    place(scene.ketchup, (-0.26, 0.06, sm.APRON_TOP + 0.003))
    step(60)
    check("settle gate: the goal pose written WHILE MOVING (ketchup in the band with "
          f"a real {v_now:.2f} m/s downward velocity > settle_lin {c.settle_lin}) — "
          f"every pose gate passes (contained={in_now}, sealed={sealed_now}) but "
          f"stillness has not persisted (settled={settled_now}) -> NOT success; "
          "construct removed before ring-down",
          in_now and sealed_now and v_now > c.settle_lin and not settled_now
          and not ok)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in scene._bodies())
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.pinned_hatch_delivery")
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

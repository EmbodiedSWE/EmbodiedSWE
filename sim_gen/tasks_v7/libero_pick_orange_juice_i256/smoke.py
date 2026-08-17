"""Smoke / rubric-REJECTION battery for JuiceCarouselScene (sim_gen task
`libero_pick_orange_juice_i256`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — servo the carousel, drag the juice out through
the window, deliver to the basket — is the acceptance evidence). Every teleport
here is instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled
state and asserts the rubric REJECTS it, plus physics probes that prove the
mechanism is real: the SAME radial drag that extracts an aligned carton is
physically BLOCKED by the wall when misaligned. A few probes construct the genuine
end state on purpose (the acceptance construct and its restore-flips) — every
other judged point must stay success()=False and a final audit asserts exactly
that.

  1.  settle              — reset settles finite; juice riding its slot, misaligned
                            45-175 deg behind the wall; distractors riding;
  2.  masses              — PhysX mass READBACK matches the authored masses (guards
                            the custom-spawner mass_props-ignored trap: the whole
                            wall-blocks / drag-slides calibration rides on them);
  3.  baseline            — fresh reset scores ~0, no success;
  4-7. randomization      — READBACK over 8 seeded resets: base xy + free yaw vary;
                            the juice's initial window-error varies IN BOTH SIGNS;
                            its slot assignment varies; the basket position varies;
  8.  null policy         — 400 idle steps: score ~0, no success, and the carousel
                            holds its angle (drift < 2 deg — nothing self-aligns);
  9.  wall blocks         — the misaligned juice given the solve's own radial drag
                            (1.7 N, velocity-regulated) for 4 s: it MOVES (probe
                            non-vacuous) but is stopped by the wall, never exits,
                            no extract latch, no align latch, score stays low;
  10. window admits       — the carousel rotated INTO alignment (linkage-consistent
                            construct), then the IDENTICAL drag: the carton exits
                            onto the porch and the extract latch sets — same force,
                            different angle, opposite outcome (9 is real physics);
  11. transient guard     — the juice held HIGH outside the shroud (a mid-carry
                            pose) does not arm the extract latch (low-through-the-
                            window is what counts, not airborne radius);
  12. mid-fall guard      — the juice judged mid-fall above the basket mouth is not
                            in-basket and not success;
  13. wrong item          — the WHITE MILK delivered into the basket instead (and
                            thereby off the disc): rejected, score stays ~0;
  14. acceptance          — the juice dropped into the basket (gravity seats it),
                            distractors riding, everything still -> success TRUE,
                            |score - 1| < 1e-3 (exactness);
  15. distractor clause   — from the accepted state, the SODA knocked off the disc
                            to the floor: success flips FALSE;
  16. restore             — the soda stood back on its slot: success TRUE again
                            (15's rejection was the distractor clause alone);
  17. liveness            — the carousel left SPINNING in the accepted state:
                            success flips FALSE while it turns (score falls to the
                            latched base, not 1.0), TRUE again once damping parks
                            it — success is live state, not a latch;
  18. rejection audit     — success() was never True at any judged point EXCEPT
                            the constructed acceptance probes (12-settle, 16, 17);
  19. final no-NaN        — all task-object states finite at the end;
  20. camera              — >= 20 rgb frames captured -> frames.npz.

Run (forge): python -u -m simgen_tasks.<task>.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=16)
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

robobench.discover()
try:
    from .scene import (  # noqa: F401
        BASKET_T, DISC_Z, ITEM_H, RIM_H, SLOT_R, WALL_RI,
        _qapply, _qinv,
    )
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import (  # noqa: F401
        BASKET_T, DISC_Z, ITEM_H, RIM_H, SLOT_R, WALL_RI,
        _qapply, _qinv,
    )
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

DRAG_F = 1.7  # the solve's drag force (N)
DRAG_V = 0.10  # ... and regulated slide speed (m/s)
EPS = 1e-4  # float32 score-cap boundary epsilon


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.juice_carousel")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.00, -0.85, 0.75)) + o),
                                tuple(np.array((0.05, 0.00, 0.15)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

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

    ever_bad_success = [False]

    def judge() -> tuple[float, bool]:
        """Judge a REJECTION probe: success here is a rubric failure."""
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_bad_success[0] = ever_bad_success[0] or ok
        return s, ok

    def judge_accept() -> tuple[float, bool]:
        """Judge an ACCEPTANCE construct: success here is expected and allowed."""
        return float(scene.score()[0]), bool(scene.success()[0])

    def polar0() -> tuple[float, float, float]:
        r, az, z = scene.juice_polar()
        return float(r[0]), float(az[0]), float(z[0])

    def report(tag: str, s: float, ok: bool) -> None:
        r, az, z = polar0()
        print(f"[smoke] {tag:18s} | juice r={r:.3f} az={math.degrees(az):+7.1f}deg "
              f"z={z:.3f} in_basket={bool(scene.juice_in_basket()[0])} "
              f"milk_on={bool(scene.on_disc(scene.milk)[0])} "
              f"soda_on={bool(scene.on_disc(scene.soda)[0])} "
              f"align={bool(scene.align_latch[0])} "
              f"extract={bool(scene.extract_latch[0])} "
              f"settled={bool(scene.all_settled()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def teleport(body, pos, quat, vel=None, ang=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3:7] = quat
        if vel is not None:
            st[:, 7:10] = torch.tensor(vel, device=device)
        if ang is not None:
            st[:, 10:13] = torch.tensor(ang, device=device)
        body.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def settle_all(max_steps: int = 600) -> None:
        for _ in range(max_steps // 30):
            step(30)
            if bool(scene.all_settled()[0]):
                break

    def clear_wrench(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def apply_wrench(body, force_w, torque_w) -> None:
        """WORLD-frame wrench at CoM. Measured on this forge build with a
        free-flight probe (no contacts, exact |v| = F*4dt/m): the DEFAULT
        set_external_force_and_torque call applies its vectors in the BODY frame
        (direction error tracked the body yaw exactly), while is_global=True
        resolves against a STALE frame (up to ~170 deg off, state-dependent).
        So: pre-rotate the desired world wrench by the LIVE inverse root quat on
        every call, and never pass is_global."""
        q = body.data.root_quat_w
        body.set_external_force_and_torque(
            _qapply(_qinv(q), force_w).view(n, 1, 3),
            _qapply(_qinv(q), torque_w).view(n, 1, 3), env_ids=all_ids)

    def drag_probe(max_steps: int) -> float:
        """The solve's OWN extraction primitive: a bang-bang velocity-regulated
        horizontal drag on the juice along a FIXED direction — the outward radial
        at the azimuth where the probe STARTS (the solve drags along base +x the
        same way). Fixed direction is self-centering (a drifted carton gets a
        restoring tangential component); recomputing the radial live lets drift
        accumulate. Returns the maximum base-frame radius reached. Identical for
        the blocked (misaligned) and admitted (aligned) probes — only the start
        angle differs."""
        u = scene.juice.data.root_pos_w - scene.base.data.root_pos_w
        u[:, 2] = 0.0
        u = (u / u.norm(dim=-1, keepdim=True).clamp(min=1e-6)).clone()
        r_max = 0.0
        for i in range(max_steps):
            r_now = polar0()[0]
            r_max = max(r_max, r_now)
            if r_now > 0.25:
                break
            v_along = (scene.juice.data.root_lin_vel_w * u).sum(dim=-1)
            f = torch.where((v_along < DRAG_V).view(n, 1), DRAG_F * u,
                            torch.zeros_like(u))
            apply_wrench(scene.juice, f, torch.zeros_like(u))
            step(1)
            if i % 90 == 0:
                _rr, _aa, _zz = polar0()
                print(f"[smoke] drag {i:4d}: r={_rr:.3f} "
                      f"az={math.degrees(_aa):+6.1f}deg z={_zz:.3f} "
                      f"v={float(v_along[0]):+.3f} "
                      f"wheel={float(scene.wheel_speed()[0]):+.3f}", flush=True)
        clear_wrench(scene.juice)
        step(90)
        return max(r_max, polar0()[0])

    def servo_align(max_steps: int = 2400) -> bool:
        """The solve's OWN alignment primitive: a torque servo on the turntable
        (|tau| <= 0.20 N*m about the axle — the crank-knob push a Franka performs at
        the 0.125 m knob orbit, <= 1.6 N tangential). Real dynamics only — alignment
        is reached through the same interaction chain the intended embodiment uses,
        with no teleport inside the shroud. Returns True once |az| < 6 deg and the
        wheel is still for 60 consecutive steps."""
        streak = 0
        for _ in range(max_steps):
            _r, az, _z = scene.juice_polar()
            w = scene.turntable.data.root_ang_vel_w[:, 2]
            w_des = (-1.5 * az).clamp(-0.60, 0.60)
            tau = (0.50 * (w_des - w)).clamp(-0.20, 0.20)
            tq = torch.zeros(n, 3, device=device)
            tq[:, 2] = tau
            apply_wrench(scene.turntable, torch.zeros_like(tq), tq)
            step(1)
            if abs(float(az[0])) < math.radians(6.0) and abs(float(w[0])) < 0.08:
                streak += 1
                if streak >= 60:
                    break
            else:
                streak = 0
        clear_wrench(scene.turntable)
        step(90)
        return streak >= 60

    def basket_hover_pose(body_h: float = ITEM_H):
        pos = scene.basket.data.root_pos_w.clone()
        pos[:, 2] += BASKET_T + RIM_H + 0.03 + body_h / 2
        return pos, scene.basket.data.root_quat_w

    def all_finite() -> bool:
        bodies = [scene.base, scene.turntable, scene.juice, scene.milk,
                  scene.soda, scene.basket]
        return all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)

    # =========================== 1-3. settle / masses / baseline ============================
    env.reset(seed=11)
    settle_all(480)
    s, ok = judge()
    report("reset", s, ok)
    r0, az0, z0 = polar0()
    check("settle: all states finite, juice riding its slot "
          f"(r={r0:.3f}~{SLOT_R}, z={z0:.3f}) MISALIGNED "
          f"{math.degrees(abs(az0)):.1f}deg behind the wall (45-175 expected), "
          "both distractors riding, everything settled",
          all_finite() and bool(scene.on_disc(scene.juice)[0])
          and bool(scene.on_disc(scene.milk)[0]) and bool(scene.on_disc(scene.soda)[0])
          and math.radians(37.0) < abs(az0) < math.radians(179.0)
          and bool(scene.all_settled()[0]))
    masses = {nm: float(getattr(scene, nm).root_physx_view.get_masses()[0].sum())
              for nm in ("base", "turntable", "juice", "milk", "soda", "basket")}
    print(f"[smoke] mass readback: {masses}", flush=True)
    check("masses: PhysX readback matches the authored masses (base 25, wheel 1.8, "
          f"cartons 0.35, basket 1.2; read {masses})",
          abs(masses["base"] - 25.0) < 1.0 and abs(masses["turntable"] - 1.8) < 0.1
          and abs(masses["juice"] - 0.35) < 0.02 and abs(masses["milk"] - 0.35) < 0.02
          and abs(masses["soda"] - 0.35) < 0.02 and abs(masses["basket"] - 1.2) < 0.06)
    check(f"baseline: score ~0 and no success on a fresh reset (score={s:.3f})",
          s <= 0.05 and not ok)

    # =========================== 4-7. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(20)
        bp = (scene.base.data.root_pos_w - scene.env_origins)[0]
        exv = _qapply(scene.base.data.root_quat_w,
                      torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))[0]
        yaw = math.atan2(float(exv[1]), float(exv[0]))
        _r, az, _z = polar0()[0], polar0()[1], polar0()[2]
        # juice slot = its angle in the TURNTABLE frame, snapped to the 3 slots
        loc = _qapply(_qinv(scene.turntable.data.root_quat_w),
                      scene.juice.data.root_pos_w - scene.turntable.data.root_pos_w)[0]
        slot = int(round(math.atan2(float(loc[1]), float(loc[0]))
                         / (2 * math.pi / 3))) % 3
        kp = (scene.basket.data.root_pos_w - scene.env_origins)[0]
        on = bool(scene.on_disc(scene.juice)[0])
        reads.append((float(bp[0]), float(bp[1]), yaw, az, slot,
                      float(kp[0]), float(kp[1]), on))
    arr = np.array([r[:7] for r in reads])
    print("[smoke] randomization readback (base_x, base_y, base_yaw, juice_az, "
          f"slot, basket_x, basket_y):\n{arr.round(3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: base pose varies across seeded resets (readback: "
          f"dx={spread[0]:.3f} dy={spread[1]:.3f} dyaw={spread[2]:.2f} rad)",
          spread[0] > 0.02 and spread[1] > 0.02 and spread[2] > 0.8
          and all(r[7] for r in reads))
    azs = arr[:, 3]
    check("randomization: the juice's initial window error varies AND takes both "
          f"signs (readback: [{', '.join(f'{math.degrees(a):+.0f}' for a in azs)}] deg)",
          azs.min() < -0.5 and azs.max() > 0.5)
    slots = {r[4] for r in reads}
    check(f"randomization: the juice's slot assignment varies (readback slots {slots})",
          len(slots) >= 2)
    check("randomization: basket position varies (readback: "
          f"dx={spread[5]:.3f} dy={spread[6]:.3f})",
          max(spread[5], spread[6]) > 0.05)

    # =========================== 8. null policy fails =======================================
    env.reset(seed=31)
    step(70)
    _r, az_a, _z = polar0()
    step(400)
    s, ok = judge()
    report("null-policy", s, ok)
    _r, az_b, _z = polar0()
    drift = abs(az_b - az_a)
    check("null policy: score ~0, no success, and the carousel HOLDS its angle over "
          f"400 idle steps (score={s:.3f}, drift={math.degrees(drift):.2f}deg < 2)",
          s <= 0.05 and not ok and drift < math.radians(2.0))

    # =========================== 9. the wall blocks a misaligned extraction =================
    r_start, az_c, _z = polar0()
    r_max = drag_probe(480)  # 4 s of the solve's own drag, at the WRONG angle
    s, ok = judge()
    report("wall-blocks", s, ok)
    r_end, _az, z_end = polar0()
    check("wall blocks: the misaligned juice under the solve's own radial drag "
          f"MOVED (r {r_start:.3f} -> max {r_max:.3f}, probe non-vacuous) but the "
          f"wall stopped it INSIDE (r_max < {c.extract_r}), it never left low "
          f"(z={z_end:.3f}), no extract latch, no align latch, no success, "
          f"score={s:.3f} stays low",
          r_max > r_start + 0.020 and r_max < c.extract_r and z_end < 0.30
          and not bool(scene.extract_latch[0]) and not bool(scene.align_latch[0])
          and not ok and s <= 0.30)

    # =========================== 10. the window admits an aligned extraction ================
    env.reset(seed=61)
    step(70)

    aligned = servo_align()
    r_start, az_d, _z = polar0()
    r_max = drag_probe(480)  # the IDENTICAL drag, now through the window
    s, ok = judge()
    report("window-admits", s, ok)
    r_end, _az, z_end = polar0()
    check("window admits: the carousel SERVOED into alignment by crank torque "
          f"(real dynamics, aligned={aligned}, az={math.degrees(az_d):+.1f}deg), "
          f"then the IDENTICAL drag takes the carton OUT onto the porch "
          f"(r={r_end:.3f} > {c.extract_r}, z={z_end:.3f} low), both latches SET, "
          f"distractors still riding, not success, score={s:.3f} <= 0.65",
          aligned and abs(az_d) < math.radians(c.align_deg) and r_end > c.extract_r
          and z_end < 0.30 and bool(scene.extract_latch[0])
          and bool(scene.align_latch[0])
          and bool(scene.on_disc(scene.milk)[0]) and bool(scene.on_disc(scene.soda)[0])
          and not ok and s <= 0.65 + EPS)

    # =========================== 11. transient guard (airborne radius) ======================
    env.reset(seed=71)
    step(70)
    high = scene.base.data.root_pos_w + _qapply(
        scene.base.data.root_quat_w,
        torch.tensor([0.25, 0.0, 0.50], device=device).expand(n, 3))
    teleport(scene.juice, high, scene.base.data.root_quat_w, settle_steps=0)
    step(2)  # buffers refresh; still ~0.5 m up
    s, ok = judge()
    _r, _az, z_now = polar0()
    report("high-hover", s, ok)
    check("transient guard: the juice held HIGH outside the shroud "
          f"(r>=0.22, z={z_now:.2f} >= 0.30) does NOT arm the extract latch — "
          "only low-through-the-window counts",
          not bool(scene.extract_latch[0]) and not ok and z_now > 0.30)

    # =========================== 12. mid-fall guard + 14. acceptance ========================
    env.reset(seed=91)
    step(70)
    pos, q = basket_hover_pose()
    pos[:, 2] += 0.12  # judge well above the rim
    teleport(scene.juice, pos, q, settle_steps=0)
    step(2)
    s, ok = judge()
    report("mid-fall", s, ok)
    check("mid-fall guard: the juice judged in flight above the basket mouth is "
          f"NOT in-basket and NOT success (in_basket={bool(scene.juice_in_basket()[0])})",
          not bool(scene.juice_in_basket()[0]) and not ok)
    settle_all(480)
    s, ok = judge_accept()
    report("acceptance", s, ok)
    check("acceptance: the juice seated in the basket by gravity, distractors "
          f"riding, everything still -> success TRUE and |score-1|={abs(s - 1.0):.4f} "
          "< 1e-3 (exactness)", ok and abs(s - 1.0) < 1e-3)

    # =========================== 15-16. distractor clause + restore =========================
    soda_state = scene.soda.data.root_state_w.clone()
    floor = scene.base.data.root_pos_w + _qapply(
        scene.base.data.root_quat_w,
        torch.tensor([-0.45, 0.25, ITEM_H / 2 + 0.003], device=device).expand(n, 3))
    floor[:, 2] = scene.env_origins[:, 2] + ITEM_H / 2 + 0.003
    teleport(scene.soda, floor, scene.base.data.root_quat_w, settle_steps=90)
    s, ok = judge()
    report("soda-off", s, ok)
    check("distractor clause: the SODA knocked off the disc to the floor flips "
          f"success FALSE while the juice is still delivered (score={s:.3f} falls "
          f"to the latched base <= 0.65)",
          not ok and bool(scene.juice_in_basket()[0]) and s <= 0.65 + EPS)
    scene.soda.write_root_state_to_sim(soda_state, all_ids)
    settle_all(360)
    s, ok = judge_accept()
    report("soda-restored", s, ok)
    check("restore: the soda stood back on its slot -> success TRUE again (15's "
          "rejection was the distractor clause and nothing else)", ok)

    # =========================== 17. liveness (spinning carousel) ===========================
    st = scene.turntable.data.root_state_w.clone()
    st[:, 7:10] = 0.0
    st[:, 10:13] = 0.0
    st[:, 12] = 2.0  # 2 rad/s spin, pose unchanged
    scene.turntable.write_root_state_to_sim(st, all_ids)
    step(3)
    s_spin, ok_spin = judge()
    w_spin = float(scene.wheel_speed()[0])
    report("spinning", s_spin, ok_spin)
    settle_all(600)
    s_after, ok_after = judge_accept()
    report("spun-down", s_after, ok_after)
    check("liveness: the carousel left SPINNING in the accepted state "
          f"(w={w_spin:.2f} rad/s) flips success FALSE (score falls to the latched "
          f"{s_spin:.3f} <= 0.65, not 1.0); once damping parks it, success TRUE "
          "again — success is live state",
          w_spin > 0.5 and not ok_spin and s_spin <= 0.65 + EPS and ok_after)

    # =========================== 18-19. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point except the "
          "constructed acceptance probes", not ever_bad_success[0])
    check("final: all task-object states finite (no NaN)", all_finite())

    # =========================== 20. camera + save ==========================================
    check(f"camera: >= 20 rgb frames captured ({len(frames)})", len(frames) >= 20)
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.juice_carousel")
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

"""Smoke / rubric-REJECTION battery for CradleClampScene (sim_gen task
`pick_single_ycb_i191`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — drop the blue bar into the V under gravity, press
the bracket home under contact — is the acceptance evidence that the rubric ACCEPTS a
correct outcome). Every teleport here is instrumentation that CONSTRUCTS a wrong (or
partial) outcome as a settled state and asserts the rubric REJECTS it — plus an
applied-force probe that proves the geometry-enforced ORDER is physically real. No
probe in this battery ever reaches success(), and a final audit asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite and still, score ~0 at rest;
  3-4. randomization      — READBACK over 6 seeded resets: fixture position AND saddle
                            direction vary; the parts' spawns vary;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  SEED strategy       — the seed's plan (lift the object into the air, hold it
                            raised): real upward force on the blue bar until it is
                            well past the seed's 7.5 cm, then release -> never
                            success, latched score <= 0.15;
  7.  wrong object        — the RED decoy physically seated in the V (geometry
                            readback confirms the flush diamond rest) -> no saddle
                            credit, NOT success, score < 0.9;
  8.  face-down near-miss — the BLUE bar dropped into the V WITHOUT the 45-deg roll:
                            it perches on its corner edges ~12 mm proud of the flush
                            seat (readback) -> rejected by the z band;
  9.  crosswise bar       — blue bar over the saddle with its axis ACROSS the valley
                            -> alignment clause rejects it;
  10. perched bracket     — bracket standing with both legs ON the pocket walls
                            (+22 mm, readback) -> outside the seat band, rejected;
  11. yawed bracket       — bracket at the seat turned 90 deg -> rejected;
  12. out-of-order (a)    — bracket seated FIRST in the empty fixture (real gravity
                            seat, geometry readback) -> partial credit only, < 0.9;
  13. out-of-order (b)    — with the bracket seated, the blue bar is DRIVEN at the
                            closed saddle with a real escalating horizontal force: it
                            moves, jams against the fixture/cross-bar, and never
                            enters the saddle window -> the required order is
                            geometry, not rubric fiat;
  14. partial             — bar correctly seated in the V alone (bracket untouched)
                            -> NOT success, score < 0.9;
  15. latched credit      — teleporting the seated bar back to the ground leaves the
                            latched score unchanged (credit does not evaporate);
  16. rejection audit     — success() was never True at ANY judged point;
  17. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.pick_single_ycb_i191.smoke --headless
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

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cradle_clamp")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.05, -0.85, 0.70)) + o),
                                tuple(np.array((0.15, 0.00, 0.08)) + o),
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

    ever_success = [False]

    def judge() -> tuple[float, bool]:
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return s, ok

    def report(tag: str) -> None:
        s, ok = judge()
        bar_l = scene._fixture_local(scene.bar)[0][0]
        br_l = scene._fixture_local(scene.bracket)[0][0]
        print(f"[smoke] {tag:18s} | bar=({float(bar_l[0]):+.3f},{float(bar_l[1]):+.3f},"
              f"{float(bar_l[2]):.3f})s{int(bool(scene.bar_in_saddle()[0]))} "
              f"br=({float(br_l[0]):+.3f},{float(br_l[1]):+.3f},{float(br_l[2]):.3f})"
              f"s{int(bool(scene.bracket_seated()[0]))} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    # --- fixture-frame probe placement (instrumentation, not a solution) ---
    a45 = math.cos(math.pi / 4)
    cq, sq = math.cos(math.pi / 8), math.sin(math.pi / 8)
    Q_DIAMOND = (a45 * cq, -a45 * cq, a45 * sq, a45 * sq)  # axis->saddle + 45 deg roll
    Q_FACEDOWN = (a45, -a45, 0.0, 0.0)  # axis->saddle, faces square to the world
    Q_CROSS = (a45, 0.0, a45, 0.0)  # axis ACROSS the valley (fixture +x)
    Q_IDENT = (1.0, 0.0, 0.0, 0.0)
    Q_YAW90 = (a45, 0.0, 0.0, a45)

    def place_fixture_frame(body, local_pos, q_rel, settle_steps: int = 30) -> None:
        fp = scene.fixture.data.root_pos_w
        fq = scene.fixture.data.root_quat_w
        lp = torch.tensor(local_pos, device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = fp + quat_apply(fq, lp)
        st[:, 3:7] = quat_mul(fq, torch.tensor(q_rel, device=device).expand(n, 4))
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def place_world(body, x, y, z, q, settle_steps: int = 30) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3:7] = torch.tensor(q, device=device).expand(n, 4)
        st[:, 0:3] += env.iscene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def clear_wrench(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    bodies = (scene.fixture, scene.bar, scene.decoy, scene.bracket)
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    part_z = [float((b.data.root_pos_w - scene.env_origins)[0, 2])
              for b in (scene.bar, scene.decoy, scene.bracket)]
    check("settle: all states finite, parts at rest at their ground heights",
          fin and all(z < 0.09 for z in part_z)
          and all(bool(scene.settled(b)[0]) for b in (scene.bar, scene.bracket)))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        fp = (scene.fixture.data.root_pos_w - scene.env_origins)[0]
        fq = scene.fixture.data.root_quat_w[0]
        fyaw = 2.0 * math.atan2(float(fq[3]), float(fq[0]))
        bp = (scene.bar.data.root_pos_w - scene.env_origins)[0]
        kp = (scene.bracket.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(fp[0]), float(fp[1]), fyaw,
                      float(bp[0]), float(bp[1]), float(kp[0]), float(kp[1])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (fix_x, fix_y, fix_yaw, bar_x, bar_y, "
          f"br_x, br_y):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: fixture position AND saddle direction vary across seeded "
          "resets (readback)",
          spread[0] > 0.02 and spread[1] > 0.02 and spread[2] > 0.5)
    check("randomization: bar and bracket spawns vary across seeded resets (readback)",
          spread[3] + spread[4] > 0.04 and spread[5] + spread[6] > 0.04)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. SEED strategy: lift the object ==========================
    # The seed's whole plan is "grasp the object and raise it 7.5 cm" (a pure z-shift
    # checker). Do exactly that with a real force on the blue bar, then keep holding it
    # raised — it must never count and the latched credit must stay small.
    torch.manual_seed(41)
    env.reset()
    step(10)
    f_up = torch.tensor([0.0, 0.0, 9.81 * c.bar_mass + 0.5],
                        device=device).view(1, 1, 3).expand(n, 1, 3).contiguous()
    z_peak = 0.0
    for _ in range(240):
        scene.bar.set_external_force_and_torque(f_up, zero_wrench, env_ids=all_ids,
                                                is_global=True)
        step(1)
        z_now = float((scene.bar.data.root_pos_w - scene.env_origins)[0, 2])
        z_peak = max(z_peak, z_now)
        if z_now > 0.30:
            break
    judge()  # judged while held raised
    clear_wrench(scene.bar)
    step(120)  # release, fall back, settle
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy (lift the bar into the air, hold it raised): it rose well "
          f"past the seed's 7.5 cm (peak z={z_peak:.2f} m) yet latched score stays "
          "<= 0.15, never success", z_peak > 0.15 and s <= 0.15 and not ever_success[0])

    # =========================== 7. wrong object: RED decoy seated in the V =================
    torch.manual_seed(51)
    env.reset()
    step(10)
    place_fixture_frame(scene.decoy, (0.0, 0.0, c.bar_rest_z + 0.03), Q_DIAMOND,
                        settle_steps=60)
    report("decoy-in-saddle")
    decoy_seated = bool(scene._bar_in_saddle(scene.decoy, xy_tol=c.bar_xy_tol,
                                             y_tol=c.bar_y_tol, z_tol=c.bar_z_tol,
                                             align_max_deg=c.bar_align_max_deg)[0])
    s, ok = judge()
    check("wrong object: the RED decoy is PHYSICALLY seated flush in the V (geometry "
          "readback confirms) yet earns no saddle credit, NOT success, score < 0.9",
          decoy_seated and not ok and float(scene.saddle_latch[0]) < 0.5 and s < 0.9)

    # =========================== 8. face-down near-miss: no 45-deg roll =====================
    torch.manual_seed(61)
    env.reset()
    step(10)
    place_fixture_frame(scene.bar, (0.0, 0.0, c.bar_rest_z + 0.03), Q_FACEDOWN,
                        settle_steps=90)
    bar_l = scene._fixture_local(scene.bar)[0][0]
    z_perch = float(bar_l[2])
    report("face-down")
    s, ok = judge()
    check("face-down near-miss: dropped without the diamond roll the bar perches on "
          f"its corner edges, proud of the flush seat (z={z_perch:.3f} vs flush "
          f"{c.bar_rest_z:.3f}) — rejected by the z band, no saddle credit",
          z_perch > c.bar_rest_z + c.bar_z_tol + 0.002
          and not bool(scene.bar_in_saddle()[0])
          and float(scene.saddle_latch[0]) < 0.5 and not ok)

    # =========================== 9. crosswise bar: axis across the valley ===================
    torch.manual_seed(71)
    env.reset()
    step(10)
    place_fixture_frame(scene.bar, (0.0, 0.0, 0.095), Q_CROSS, settle_steps=90)
    from isaaclab.utils.math import quat_apply as _qa
    _, bq = scene._fixture_local(scene.bar)
    axis_f = _qa(bq, torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3))[0]
    report("crosswise")
    s, ok = judge()
    check("crosswise: bar over the saddle with its axis ACROSS the valley "
          f"(|axis.y_fixture|={abs(float(axis_f[1])):.2f}) — alignment clause rejects "
          "it, no saddle credit, NOT success",
          abs(float(axis_f[1])) < 0.5 and not bool(scene.bar_in_saddle()[0])
          and float(scene.saddle_latch[0]) < 0.5 and not ok)

    # =========================== 10. bracket perched on the pocket walls ====================
    torch.manual_seed(81)
    env.reset()
    step(10)
    # Shift +12 mm in fixture y: both legs land ON wall tops (+22 mm above the floors).
    place_fixture_frame(scene.bracket,
                        (0.0, 0.012, c.base_t + c.wall_h + c.leg_len + c.cbar_t / 2 + 0.002),
                        Q_IDENT, settle_steps=60)
    br_l = scene._fixture_local(scene.bracket)[0][0]
    report("perched-on-walls")
    s, ok = judge()
    check("perched bracket: legs standing ON the pocket walls read "
          f"{(float(br_l[2]) - c.seat_z) * 1000:+.0f} mm above the seat — outside the "
          "seat band, NOT seated, NOT success",
          float(br_l[2]) - c.seat_z > c.seat_z_hi + 0.005
          and not bool(scene.bracket_seated()[0]) and not ok)

    # =========================== 11. bracket yawed 90 deg at the seat =======================
    torch.manual_seed(91)
    env.reset()
    step(10)
    place_fixture_frame(scene.bracket, (0.0, 0.0, c.seat_z + 0.043), Q_YAW90,
                        settle_steps=90)
    report("yawed-90")
    s, ok = judge()
    check("yawed bracket: dropped over the fixture turned 90 deg — legs cannot enter "
          "the pockets, NOT seated, NOT success",
          not bool(scene.bracket_seated()[0]) and not ok and s < 0.9)

    # =========================== 12. out-of-order (a): bracket first ========================
    torch.manual_seed(101)
    env.reset()
    step(10)
    place_fixture_frame(scene.bracket, (0.0, 0.0, c.seat_z + 0.043), Q_IDENT,
                        settle_steps=120)
    br_l = scene._fixture_local(scene.bracket)[0][0]
    report("bracket-first")
    s, ok = judge()
    br_seated_empty = bool(scene.bracket_seated()[0])
    check("out-of-order (a): the bracket seats in the EMPTY fixture (real gravity "
          f"seat, z={float(br_l[2]):.3f}) — partial credit only, < 0.9, NOT success",
          br_seated_empty and not ok and s < 0.9)

    # =========================== 13. out-of-order (b): bar at the closed saddle =============
    # With the bracket seated, DRIVE the blue bar at the closed saddle exactly the way
    # an arm would: a gravity-compensated hold at the flush-seat height (diamond,
    # aligned — the correct approach in every respect except the order) plus a
    # velocity-regulated horizontal push toward the valley, with weak centering
    # springs (the same regulated machinery solve.py uses to seat the bracket — no
    # constant-force squeeze that could pump a wedge). The probe must show the
    # actuator moved: the bar advances tens of mm, then jams against the V plate /
    # seated cross-bar and its center never reaches the saddle window.
    place_fixture_frame(scene.bar, (0.13, 0.0, c.bar_rest_z), Q_DIAMOND,
                        settle_steps=0)
    grav_bar = 9.81 * c.bar_mass
    x_start = float(scene._fixture_local(scene.bar)[0][0, 0])
    x_min, entered = x_start, False
    f_ax, v_des = 0.8, 0.06
    last_bump = 0
    for i in range(480):
        loc = scene._fixture_local(scene.bar)[0][0]
        x_now = float(loc[0])
        if x_now < x_min - 0.002:
            x_min, last_bump = x_now, i
        elif i - last_bump > 150:  # jammed: escalate once, keep trying
            f_ax = min(f_ax + 0.8, 2.4)
            last_bump = i
            print(f"[smoke] order probe jammed at x={x_now:+.3f}, push -> "
                  f"{f_ax:.1f} N", flush=True)
        fq = scene.fixture.data.root_quat_w
        v_w = scene.bar.data.root_lin_vel_w[0]
        ex_w = quat_apply(fq, torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))[0]
        v_ax = -float(torch.dot(v_w, ex_w))  # speed toward the valley (-x_fixture)
        fl_x = -f_ax if v_ax < v_des else 0.0
        fl_y = max(-0.4, min(0.4, -8.0 * float(loc[1])))
        fl_z = max(-0.5, min(0.5, -10.0 * (float(loc[2]) - c.bar_rest_z)))
        f_l = torch.tensor([fl_x, fl_y, 0.0], device=device).expand(n, 3)
        f_w = quat_apply(fq, f_l) + torch.tensor(
            [0.0, 0.0, grav_bar + fl_z], device=device).expand(n, 3)
        scene.bar.set_external_force_and_torque(f_w.view(n, 1, 3).contiguous(),
                                                zero_wrench, env_ids=all_ids,
                                                is_global=True)
        step(1)
        x_min = min(x_min, float(scene._fixture_local(scene.bar)[0][0, 0]))
        entered = entered or bool(scene.bar_in_saddle()[0])
        judge()
        if i % 120 == 119:
            print(f"[smoke] order probe: x={x_now:+.3f} (min {x_min:+.3f}) "
                  f"z={float(loc[2]):.3f} push={f_ax:.1f} N", flush=True)
    clear_wrench(scene.bar)
    step(60)
    report("order-probe")
    s, ok = judge()
    check("out-of-order (b): a regulated push (up to 2.4 N axial, 4 s, grav-comp "
          f"hold) drives the bar {(x_start - x_min) * 1000:.0f} mm toward the CLOSED "
          f"saddle, but it jams outside the window (min |x|={x_min:.3f} > "
          f"{c.bar_xy_tol:.3f}) and never seats — the required order is geometry, "
          "not rubric fiat",
          (x_start - x_min) > 0.03 and x_min > c.bar_xy_tol + 0.010
          and not entered and not ever_success[0])

    # =========================== 14. partial: bar seated alone ==============================
    torch.manual_seed(111)
    env.reset()
    step(10)
    place_fixture_frame(scene.bar, (0.0, 0.0, c.bar_rest_z + 0.03), Q_DIAMOND,
                        settle_steps=90)
    report("bar-only")
    s, ok = judge()
    bar_ok = bool((scene.bar_in_saddle() & scene.settled(scene.bar))[0])
    check("partial: blue bar correctly seated in the V but bracket untouched — NOT "
          "success, score < 0.9", bar_ok and not ok and s < 0.9)

    # =========================== 15. latched credit survives moving back ====================
    s_seated, _ = judge()
    place_world(scene.bar, -0.10, -0.25, c.ground_rest_z + 0.003, Q_FACEDOWN,
                settle_steps=40)
    report("moved-back")
    s_back, ok = judge()
    check("latched credit: returning the seated bar to the ground leaves the latched "
          "score unchanged", abs(s_back - s_seated) < 1e-3 and not ok)

    # =========================== 16-17. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.cradle_clamp")
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
    main()

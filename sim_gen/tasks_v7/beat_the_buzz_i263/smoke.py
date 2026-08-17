"""Smoke battery for WardedSpindleScene — REJECTION tests for the rubric,
NullRobot, RECORDED.

This is NOT a solution (the solution is solve.py — align the fin with each ward's
notch in turn and lower through; the Franka strategy is TASK.md's embodiment
argument). Probes use the same servo numbers as solve.py (shared numbers = shared
honesty); teleports are used for TRANSPORT ONLY (carrying the already-lifted
sleeve across free space in the bypass probe) — every load-bearing state is
reached through contact dynamics and released to settle. What is being tested is
that WRONG settled end states are rejected.

One linear run, 12 named checks:
  1. settle    — clean reset: finite state, sleeve resting ON the upper ward
                 (readback height), threaded, upright, score ~0;
  2. readback  — ward yaws equal the sampled th1/th2 (from the ward QUATS, not
                 the cfg), fin yaw equals th_s0, the misalignment bands hold
                 (|th2-th1| and |fin-th1| >= 55 deg), sleeve mass real, tower xy
                 inside its jitter band;
  3. random    — 10 seeds: th1, the signed notch offset th2-th1, the signed fin
                 offset, and the tower xy ALL vary, both offset signs drawn
                 (READBACK from body poses);
  4. null      — 2 s of nothing: sleeve stays resting on ward1, no latch,
                 score < 0.05, no success;
  5. negative  — the SEED-STRATEGY family (careful lowering): with the fin
                 misaligned (the sampled >= 55 deg), the solve's OWN gentle
                 descent servo runs for seconds and the sleeve does not pass the
                 ward (z readback unchanged, p1 latch still 0) — care and
                 clearance are worthless, only alignment opens the ward;
  6. negative  — near miss: fin rotated to ~30 deg off the notch (actuator
                 MOVED — non-vacuous) then pressed down hard (2x weight): still
                 no pass, no latch;
  7. negative  — bypass: the only alternative the topology leaves — sleeve
                 lifted clean off the TOP of the post (real servo lift), carried
                 (transport teleport) and released beside the plinth: settles at
                 seat height but UNTHREADED — zero credit, no success (this is
                 the analog of 'goal geometry without the required path');
  8. partial   — the real phase-1 (solve's own controller) to a rest on ward2:
                 score == 0.30 exactly, no success;
  9. latch     — regression: sleeve lifted back UP through ward1 (above the
                 latch plane — beyond the running-max metric), parked misaligned
                 on ward1: p1 latch survives (0.30), no success;
 10. exactness — the full correct strategy from there -> success() and
                 score == 1.0;
 11. latch     — success is LIVE state: the seated sleeve lifted back off the
                 plinth — success revoked, score falls back to the latched 0.80;
 12. frames    — >= 20 rgb frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.beat_the_buzz_i263.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=10)
parser.add_argument("--max_frames", type=int, default=280)
parser.add_argument("--out", type=str, default="frames.npz")  # CWD — the pipeline fetches it
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe (proven on this render stack): kit mis-decodes the driver version and
# silently rejects RTX -> the annotator returns EMPTY frames.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.beat_the_buzz_i263 import scene as scene_mod
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod

# Watchdog: never leave a GPU zombie.
threading.Timer(1200.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                 os._exit(3))).start()

wrap_pi = scene_mod.wrap_pi

# Same drive numbers as solve.py (shared numbers = shared honesty).
G = 9.81
KV = 4.0
V_DOWN = -0.12
F_MIN, F_MAX = -2.0, 4.0
KP0, KP_MAX = 0.03, 0.12
KD = 0.010
T_CAP = 0.06
ALIGN_TOL = math.radians(3.0)
DROP_DZ = 0.012
PRESS_F = -1.5  # N extra down-press (net ~2x weight) for the hard near-miss probe


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.warded_spindle")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    origin = scene.env_origins[0]

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = origin.detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.75, -0.75, 1.05)) + o),
                                tuple(np.array((0.10, 0.0, 0.55)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (800, 500))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames — check the RTX recipe",
                  flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action, render=annot is not None)
            if annot is not None and step_i % args.record_every == 0 \
                    and len(frames) < args.max_frames:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    def sc() -> float:
        return float(scene.score()[0])

    def z_now() -> float:
        return float(scene.sleeve_z()[0])

    def yaw_now() -> float:
        return float(scene.tab_yaw()[0])

    def wz_now() -> float:
        return float(scene.sleeve.data.root_ang_vel_w[0, 2])

    def vz_now() -> float:
        return float(scene.sleeve.data.root_lin_vel_w[0, 2])

    def body_yaw(body) -> float:
        q = body.data.root_quat_w[0]
        w, x, y, z = (float(v) for v in q)
        return math.atan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))

    def d_ang(a: float, b: float) -> float:
        return float(wrap_pi(torch.tensor(a - b)))

    def report(tag: str) -> None:
        print(f"[smoke] {tag:14s} | z={z_now():.3f} yaw={math.degrees(yaw_now()):+7.1f}deg "
              f"thr={int(scene.threaded()[0])} up={int(scene.upright()[0])} "
              f"p1={float(scene.p1_latch[0]):.0f} p2={float(scene.p2_latch[0]):.0f} "
              f"seat={float(scene.seat_latch[0]):.0f} score={sc():.3f} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    # --- probe helpers (solve.py's controllers, verbatim numbers) ---
    def yaw_pd(target: float, kp: float) -> None:
        err = d_ang(target, yaw_now())
        tau = kp * err - KD * wz_now()
        scene.drive_t[0, 0, 2] = max(-T_CAP, min(T_CAP, tau))

    def hold_z(v_des: float) -> None:
        f = c.sleeve_mass * G + KV * (v_des - vz_now())
        scene.drive_f[0, 0, 2] = max(F_MIN, min(F_MAX, f))

    def zero_drives() -> None:
        scene.drive_f[0, 0] = 0.0
        scene.drive_t[0, 0] = 0.0

    def align(target: float, z_rest: float, kp0: float = KP0,
              budget: int = 900) -> tuple[str, float]:
        """solve.py's align: PD toward `target` on the shelf, timer-escalated gain."""
        kp = kp0
        aligned_streak = 0
        esc = 0
        for _ in range(budget):
            if z_now() < z_rest - DROP_DZ:
                scene.drive_t[0, 0, 2] = 0.0
                return "dropped", kp
            err = d_ang(target, yaw_now())
            if abs(err) < ALIGN_TOL and abs(wz_now()) < 0.2:
                aligned_streak += 1
                if aligned_streak >= 240:
                    scene.drive_t[0, 0, 2] = 0.0
                    return "aligned", kp
            else:
                aligned_streak = 0
            esc += 1
            if esc >= 240 and kp < KP_MAX:
                kp = min(kp * 1.5, KP_MAX)
                esc = 0
            yaw_pd(target, kp)
            step(1)
        scene.drive_t[0, 0, 2] = 0.0
        return "stuck", kp

    def descend(target: float, z_stop: float, budget: int = 1200) -> bool:
        """solve.py's controlled descent with yaw hold. False = wedged/stalled."""
        stall = 0
        for _ in range(budget):
            z = z_now()
            if abs(z - z_stop) < 0.005 and abs(vz_now()) < 0.05:
                zero_drives()
                return True
            hold_z(V_DOWN if z > z_stop + 0.004 else 0.0)
            yaw_pd(target, 0.02)
            stall = stall + 1 if (abs(vz_now()) < 0.02 and z > z_stop + 0.015) else 0
            if stall >= 240:
                zero_drives()
                return False
            step(1)
        zero_drives()
        return False

    def settle(budget: int = 600, streak_need: int = 30) -> bool:
        zero_drives()
        streak = 0
        for _ in range(budget):
            step(1)
            v = float(scene.sleeve.data.root_lin_vel_w[0].norm())
            w = float(scene.sleeve.data.root_ang_vel_w[0].norm())
            streak = streak + 1 if (v < 0.03 and w < 0.5) else 0
            if streak >= streak_need:
                return True
        return False

    def pass_ward(target: float, z_above: float, z_below: float) -> bool:
        """solve.py's pass_ward (align + drop + descend + settle, gain carried)."""
        kp = KP0
        for _ in range(4):
            res, kp = align(target, z_above, kp)
            if res == "stuck":
                continue
            if descend(target, z_below):
                return settle()
            lift_to = z_above + 0.004
            for _ in range(600):
                if z_now() >= lift_to - 0.003:
                    break
                hold_z(0.10)
                yaw_pd(target, 0.02)
                step(1)
            settle(budget=240)
        return False

    def lift_to(z_des: float, yaw_hold: float, budget: int = 900) -> bool:
        """Vertical servo lift with a soft yaw hold (real contact lift, no teleport)."""
        for _ in range(budget):
            if z_now() >= z_des - 0.005 and abs(vz_now()) < 0.08:
                zero_drives()
                return True
            hold_z(0.18 if z_now() < z_des - 0.01 else 0.0)
            yaw_pd(yaw_hold, 0.02)
            step(1)
        zero_drives()
        return False

    def finite_all() -> bool:
        return all(bool(torch.isfinite(b.data.root_state_w).all())
                   for b in scene._bodies().values()) \
            and bool(torch.isfinite(scene.score()).all())

    # ========================= 1. settle / clean-slate ========================================
    torch.manual_seed(3)
    env.reset()
    step(60)
    report("reset")
    th1 = float(scene.th1[0])
    th2 = float(scene.th2[0])
    ths = float(scene.th_s0[0])
    check("settle: clean reset (finite, sleeve resting ON ward1, threaded, upright, score ~0)",
          finite_all()
          and abs(z_now() - c.z_rest_w1) < 0.008
          and bool(scene.threaded()[0]) and bool(scene.upright()[0])
          and sc() < 0.05 and not bool(scene.success()[0]))

    # ========================= 2. pose + mass readback ========================================
    try:
        m_s = float(scene.sleeve.root_physx_view.get_masses().reshape(-1)[0])
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] mass readback unavailable ({exc!r})", flush=True)
        m_s = float("nan")
    w1_yaw = body_yaw(scene.ward1)
    w2_yaw = body_yaw(scene.ward2)
    post_xy = scene.post.data.root_pos_w[0, 0:2] - origin[0:2]
    t0 = torch.tensor(c.tower_xy0, device=device)
    print(f"[smoke] readback: m={m_s:.3f}kg w1={math.degrees(w1_yaw):+.1f} "
          f"(th1={math.degrees(th1):+.1f}) w2={math.degrees(w2_yaw):+.1f} "
          f"(th2={math.degrees(th2):+.1f}) fin={math.degrees(yaw_now()):+.1f} "
          f"(th_s0={math.degrees(ths):+.1f}) "
          f"d21={math.degrees(d_ang(th2, th1)):+.1f} "
          f"ds1={math.degrees(d_ang(ths, th1)):+.1f} "
          f"post_xy=({float(post_xy[0]):+.3f},{float(post_xy[1]):+.3f})", flush=True)
    lo = math.radians(c.misalign_min_deg)
    check("readback: ward quats == sampled th1/th2, fin == th_s0, misalignment bands "
          "hold, sleeve mass real, tower xy in band",
          abs(m_s - c.sleeve_mass) < 0.01
          and abs(d_ang(w1_yaw, th1)) < 0.02 and abs(d_ang(w2_yaw, th2)) < 0.02
          and abs(d_ang(yaw_now(), ths)) < math.radians(10.0)
          and lo - 0.04 <= abs(d_ang(th2, th1)) <= math.pi + 1e-3
          and abs(d_ang(ths, th1)) >= lo - math.radians(12.0)
          and bool(((post_xy - t0).abs() <= c.tower_jitter + 0.002).all()))

    # ========================= 3. randomization across seeds ==================================
    draws = []
    for seed in range(11, 21):
        torch.manual_seed(seed)
        env.reset()
        step(2)
        pxy = scene.post.data.root_pos_w[0, 0:2] - origin[0:2]
        a1 = body_yaw(scene.ward1)
        d21 = d_ang(body_yaw(scene.ward2), a1)
        ds1 = d_ang(yaw_now(), a1)
        draws.append((round(math.degrees(a1), 1), round(math.degrees(d21), 1),
                      round(math.degrees(ds1), 1),
                      round(float(pxy[0]), 3), round(float(pxy[1]), 3)))
    print(f"[smoke] draws (th1, th2-th1, fin-th1, x, y): {draws}", flush=True)
    th1s = {d[0] for d in draws}
    d21s = {d[1] for d in draws}
    ds1s = {d[2] for d in draws}
    xys = {(d[3], d[4]) for d in draws}
    sgn21 = {d[1] > 0 for d in draws}
    sgns1 = {d[2] > 0 for d in draws}
    check("random: th1, notch offset (both signs), fin offset (both signs) and tower xy "
          "all vary over 10 seeds (pose readback)",
          len(th1s) >= 8 and len(d21s) >= 8 and len(ds1s) >= 8 and len(xys) >= 8
          and sgn21 == {True, False} and sgns1 == {True, False})

    # ========================= 4. null policy =================================================
    torch.manual_seed(3)
    env.reset()
    step(240)  # 2 s of nothing
    report("null")
    check("null: 2 s of no action — sleeve still resting on ward1, no latch, score < 0.05",
          abs(z_now() - c.z_rest_w1) < 0.008
          and float(scene.p1_latch[0]) == 0.0 and float(scene.p2_latch[0]) == 0.0
          and sc() < 0.05 and not bool(scene.success()[0]))

    # ========================= 5. seed strategy: careful misaligned lowering ==================
    # The fin is misaligned by the sampled >= 55 deg. Run the solve's OWN gentle descent
    # servo (the exact controller that succeeds when aligned) — however careful, the
    # sleeve cannot pass the ward: alignment, not clearance control, is the content.
    z0 = z_now()
    yaw_hold = yaw_now()
    passed = descend(yaw_hold, c.z_rest_w2, budget=360)  # 3 s of trying
    settle(budget=240)
    report("careful-lower")
    check("negative (seed strategy): the solve's own gentle descent servo, misaligned — "
          "sleeve does NOT pass (z unchanged, no latch)",
          not passed and abs(z_now() - z0) < 0.010
          and float(scene.p1_latch[0]) == 0.0 and sc() < 0.05)

    # ========================= 6. near miss: ~30 deg off + hard press =========================
    # Rotate toward a target 30 deg off the notch ON THE SAME SIDE as the fin start
    # (the path never crosses the notch — no routing through the goal), then press
    # down at ~2x weight. The actuator provably moved (non-vacuous), still no pass.
    miss = float(wrap_pi(torch.tensor(th1 + math.copysign(math.radians(30.0),
                                                          d_ang(ths, th1)))))
    res, _ = align(miss, c.z_rest_w1, budget=600)
    moved = abs(d_ang(yaw_now(), ths))
    z0 = z_now()
    for _ in range(180):  # 1.5 s hard press
        scene.drive_f[0, 0, 2] = PRESS_F
        step(1)
    zero_drives()
    settle(budget=240)
    report("near-miss")
    check("negative (near miss): fin rotated ~30 deg off the notch (actuator moved) + "
          "2x-weight press — no pass, no latch",
          moved > math.radians(15.0)
          and abs(d_ang(yaw_now(), th1)) > math.radians(20.0)
          and abs(z_now() - c.z_rest_w1) < 0.010 and abs(z_now() - z0) < 0.010
          and float(scene.p1_latch[0]) == 0.0 and sc() < 0.05)

    # ========================= 7. bypass: off the top of the post =============================
    # The hub is a closed loop, so the ONLY alternative to the wards is over the post
    # top. Do it for real: servo-lift the sleeve clean off the post, transport-teleport
    # it beside the plinth (already lifted, zero velocity), release from 8 mm, settle.
    # It ends AT seat height, upright, settled — and earns nothing: never threaded
    # below the pass planes.
    clear_z = c.post_top + c.hub_h / 2 + 0.030
    ok_up = lift_to(clear_z, yaw_now(), budget=1200)
    report("off-top")
    tx, ty = float(scene.post.data.root_pos_w[0, 0] - origin[0]), \
        float(scene.post.data.root_pos_w[0, 1] - origin[1])
    st = torch.zeros(1, 13, device=device)
    st[0, 0] = tx + 0.16
    st[0, 1] = ty
    st[0, 2] = c.z0 + c.hub_h / 2 + 0.008
    st[0, 3] = 1.0
    st[0, 0:3] += origin
    scene.sleeve.write_root_state_to_sim(st, torch.tensor([0], device=device))
    scene.drive_f[0, 0, 2] = c.sleeve_mass * G  # gravity hold for the transition step
    step(1)
    ok_dn = settle()
    step(120)
    report("bypassed")
    check("negative (bypass): sleeve taken off the TOP of the post and set down beside "
          "the plinth — rests at seat height but UNTHREADED: zero credit, no success",
          ok_up and ok_dn
          and z_now() < c.z0 + c.hub_h / 2 + 0.02
          and not bool(scene.threaded()[0])
          and float(scene.p1_latch[0]) == 0.0 and float(scene.p2_latch[0]) == 0.0
          and float(scene.seat_latch[0]) == 0.0
          and sc() < 0.05 and not bool(scene.success()[0]))

    # ========================= 8. partial credit: rest on ward2 ===============================
    torch.manual_seed(3)
    env.reset()
    step(60)
    th1 = float(scene.th1[0])
    th2 = float(scene.th2[0])
    ok_p1 = pass_ward(th1, c.z_rest_w1, c.z_rest_w2)
    report("on-ward2")
    check("partial: real phase-1 through the upper ward to a rest on ward2 — "
          "score == 0.30 exactly, no success",
          ok_p1 and abs(z_now() - c.z_rest_w2) < 0.008
          and abs(sc() - 0.30) < 0.005 and not bool(scene.success()[0]))

    # ========================= 9. latch retention under regression ============================
    # Lift back UP through ward1 (fin still near th1 -> the notch is open), park
    # ABOVE the p1 pass plane — beyond the running-max metric — misaligned on ward1.
    ok_up = lift_to(c.z_rest_w1 + 0.025, th1, budget=1200)
    park = float(wrap_pi(torch.tensor(th1 + math.radians(40.0))))
    for _ in range(300):  # hover + rotate away from the notch
        if abs(d_ang(yaw_now(), park)) < ALIGN_TOL:
            break
        hold_z(0.0)
        yaw_pd(park, 0.03)
        step(1)
    ok_park = settle()
    report("regressed")
    check("latch retention: sleeve lifted back above ward1 and parked misaligned on it — "
          "p1 latch survives (score 0.30), no success",
          ok_up and ok_park and abs(z_now() - c.z_rest_w1) < 0.010
          and abs(d_ang(yaw_now(), th1)) > math.radians(20.0)
          and float(scene.p1_latch[0]) == 1.0
          and abs(sc() - 0.30) < 0.005 and not bool(scene.success()[0]))

    # ========================= 10. exactness ==================================================
    ok_p1b = pass_ward(th1, c.z_rest_w1, c.z_rest_w2)
    ok_p2 = pass_ward(th2, c.z_rest_w2, c.z_seat)
    got = False
    for _ in range(48):
        if bool(scene.success()[0]):
            got = True
            break
        step(10)
    report("goal-state")
    check("exactness: full correct strategy -> success() and score == 1.0",
          ok_p1b and ok_p2 and got and abs(sc() - 1.0) < 1e-3)

    # ========================= 11. success is live: unseat revokes it =========================
    # Lift back UP through ward2's notch (fin still near th2), rotate away and park
    # misaligned ON ward2 — releasing inside the aligned slot would just re-seat it.
    ok_up = lift_to(c.z_rest_w2 + 0.025, th2, budget=1200)
    park2 = float(wrap_pi(torch.tensor(th2 + math.radians(40.0))))
    for _ in range(300):
        if abs(d_ang(yaw_now(), park2)) < ALIGN_TOL:
            break
        hold_z(0.0)
        yaw_pd(park2, 0.03)
        step(1)
    ok_park = settle()
    step(120)
    report("unseated")
    check("live success: seated sleeve lifted back off the plinth and parked on ward2 — "
          "success revoked, score falls back to the latched 0.80",
          ok_up and ok_park and z_now() > c.z_seat + 0.030
          and not bool(scene.success()[0]) and abs(sc() - 0.80) < 0.005)

    # ========================= 12. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.warded_spindle")
        print(f"[smoke] saved {arr.shape} -> {os.path.abspath(args.out)}", flush=True)
    check("video frames recorded", len(frames) >= 20)

    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        bad = [nm for nm, ok in checks if not ok]
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)} — failing: {bad}", flush=True)
    # Kit teardown hangs are routine: watchdog then hard exit.
    threading.Timer(10.0, lambda: os._exit(0 if all_ok else 1)).start()
    try:
        env.close()
        app.close()
    finally:
        os._exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

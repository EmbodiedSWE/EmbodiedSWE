"""Smoke battery for DialSetpointScene — REJECTION tests for the rubric, NullRobot,
RECORDED.

This is NOT a solution (the solution is solve.py — per dial, a rate-cascade torque
servo that stops the free-spinning knob inside the +-6 deg band at its sampled flag;
the Franka strategy is TASK.md's embodiment argument). Drives here are the same
fingertip-scale numbers as solve.py (shared numbers = shared honesty); outcomes are
CONSTRUCTED through the live D6 + damping plant, never teleported into place.

One linear run, 12 named checks:
  1. settle    — clean reset: finite state on every body, both knobs physically AT
                 their sampled start angles (readback), score ~0, no success;
  2. readback  — both flags physically stand on their rims at the SAMPLED target
                 angles (world-pose readback vs the cfg formula), separation >= 40 deg;
  3. random    — target angles (sign AND magnitude) and start angles vary across
                 seeds; min separation holds on every draw (READBACK, 8 seeds);
  4. null      — 2 s of nothing: score < 0.05, no success;
  5. negative  — the SEED's plan (press it): an 8 N axial press + an 8 N lateral
                 shove on a knob change its pointer by < 2 deg and earn ~0 — the
                 reading only changes by rotation;
  6. negative  — end stops are physical: a fast servo commanded far past the limit
                 parks at ~+-92 deg, finite, no success;
  7. negative  — fly-through: sweeping the pointer THROUGH the flag at speed and
                 stopping well past it latches almost nothing (slow-gate counter),
                 no success — transient credit must not stick;
  8. negative  — near miss: dial 0 set exactly, dial 1 parked 2.5x tol off its flag
                 — partial credit only, no success (BOTH dials must rest on target);
  9. negative  — crossed flags: dial 0 turned to dial 1's bearing and vice versa —
                 wrong place, no success;
 10. exactness — full correct strategy (both dials, stop-and-release) -> success()
                 and score == 1.0, still true 1 s later;
 11. latch     — turning dial 1 away revokes success (success is live state); the
                 latched share of earned credit remains;
 12. frames    — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.press_switch_i161.smoke --headless
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
    from simgen_tasks.press_switch_i161 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie.
threading.Timer(1200.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                 os._exit(3))).start()

# Same fingertip-scale drives as solve.py (shared numbers = shared honesty).
K_ANG = 4.0
W_CAP = 1.2
KW = 4e-3
TAU_MAX = 0.04
ERR_DONE = math.radians(2.5)
W_DONE = 0.10
PRESS_F = 8.0  # N — the seed-style press/shove probe (>> the knob's 1 N weight)
FLY_W_CAP = 2.4  # rad/s — the fly-through sweep (>> slow_gate)


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.dial_setpoints")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    tol = math.radians(c.tol_deg)

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.55, -0.60, 1.00)) + o),
                                tuple(np.array((0.0, 0.0, 0.46)) + o),
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

    def report(tag: str) -> None:
        yaw = scene.knob_yaw()[0]
        err = scene.align_err()[0]
        print(f"[smoke] {tag:14s} | yaw=({math.degrees(float(yaw[0])):+6.1f}, "
              f"{math.degrees(float(yaw[1])):+6.1f})deg "
              f"err=({math.degrees(float(err[0])):5.1f}, "
              f"{math.degrees(float(err[1])):5.1f})deg "
              f"latch=({float(scene.align_latch[0, 0]):.2f}, "
              f"{float(scene.align_latch[0, 1]):.2f}) "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    # --- drive helpers (solve.py's servo, dial-generic) ---
    def servo_dial(d: int, tgt: float, budget: int = 2400,
                   err_done: float = ERR_DONE, w_cap: float = W_CAP) -> bool:
        """Rate-cascade torque servo to angle `tgt` (rad); releases only a SLOW knob."""
        for _ in range(budget):
            yaw = float(scene.knob_yaw()[0, d])
            w = float(scene.knobs[d].data.root_ang_vel_w[0, 2])
            e = wrap(tgt - yaw)
            if abs(e) < err_done and abs(w) < W_DONE:
                scene.drive_t[0, d] = 0.0
                return True
            w_des = max(-w_cap, min(w_cap, K_ANG * e))
            scene.drive_t[0, d] = max(-TAU_MAX, min(TAU_MAX, KW * (w_des - w)))
            step(1)
        scene.drive_t[0, d] = 0.0
        return False

    def fly_through(d: int, dest: float) -> float:
        """Sweep dial d to `dest` FAST (rate cap >> slow_gate); returns the peak |w|
        observed while the pointer was inside the tol band (proof it CROSSED at speed)."""
        peak = 0.0
        for _ in range(2400):
            yaw = float(scene.knob_yaw()[0, d])
            w = float(scene.knobs[d].data.root_ang_vel_w[0, 2])
            if abs(wrap(float(scene.target[0, d]) - yaw)) < tol:
                peak = max(peak, abs(w))
            e = wrap(dest - yaw)
            if abs(e) < math.radians(3.0) and abs(w) < W_DONE:
                break
            w_des = max(-FLY_W_CAP, min(FLY_W_CAP, 6.0 * e))
            scene.drive_t[0, d] = max(-TAU_MAX, min(TAU_MAX, KW * (w_des - w)))
            step(1)
        scene.drive_t[0, d] = 0.0
        step(120)
        return peak

    def finite_all() -> bool:
        return all(bool(torch.isfinite(b.data.root_state_w).all())
                   for b in scene._bodies().values()) \
            and bool(torch.isfinite(scene.score()).all())

    # --- pick the fixed probe seed: need fly-through room past one target and
    # --- distinct flag bearings for the crossed-flags check (deterministic search) ---
    seed_pick, fly_dial, fly_dest = None, None, 0.0
    for sd in range(3, 21):
        torch.manual_seed(sd)
        env.reset()
        step(2)
        t = [float(scene.target[0, d]) for d in (0, 1)]
        s = [float(scene.start[0, d]) for d in (0, 1)]
        if abs(wrap(t[0] - t[1])) <= 2.5 * tol:
            continue
        for d in (0, 1):
            dest = t[d] + math.copysign(math.radians(25.0), t[d] - s[d])
            if abs(dest) <= math.radians(88.0):
                seed_pick, fly_dial, fly_dest = sd, d, dest
                break
        if seed_pick is not None:
            break
    assert seed_pick is not None, "no probe seed in 3..20 satisfies the preconditions"
    print(f"[smoke] probe seed={seed_pick} fly_dial={fly_dial} "
          f"fly_dest={math.degrees(fly_dest):+.1f}deg", flush=True)

    # ========================= 1. settle / clean-slate ========================================
    torch.manual_seed(seed_pick)
    env.reset()
    step(60)
    report("reset")
    yaw0 = scene.knob_yaw()[0]
    start_err = (yaw0 - scene.start[0]).abs().max()
    check("settle: clean reset (finite, knobs AT their sampled starts, score ~0)",
          finite_all() and float(start_err) < math.radians(2.5)
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 2. flag pose readback ==========================================
    flag_err = 0.0
    for d in (0, 1):
        a = float(scene.target[0, d])
        want = torch.tensor([c.flag_r * math.cos(a),
                             c.dial_y[d] + c.flag_r * math.sin(a),
                             c.flag_z], device=device)
        got = scene.flags[d].data.root_pos_w[0] - scene.env_origins[0]
        flag_err = max(flag_err, float((got - want).norm()))
    sep = [math.degrees(float(scene.err0[0, d])) for d in (0, 1)]
    print(f"[smoke] targets {[round(math.degrees(float(scene.target[0, d])), 1) for d in (0, 1)]}"
          f" deg, flag readback err={flag_err * 1000:.2f}mm, separations={sep} deg", flush=True)
    check("readback: flags physically posed at the sampled targets; separation >= 40 deg",
          flag_err < 0.002 and min(sep) >= 39.5)

    # ========================= 3. randomization across seeds ==================================
    draws = []
    ok_sep = True
    for seed in (11, 12, 13, 14, 15, 16, 17, 18):
        torch.manual_seed(seed)
        env.reset()
        step(2)
        t = [round(math.degrees(float(scene.target[0, d])), 1) for d in (0, 1)]
        s = [round(math.degrees(float(scene.start[0, d])), 1) for d in (0, 1)]
        # READBACK: the knob is physically at the sampled start
        y = [round(math.degrees(float(scene.knob_yaw()[0, d])), 1) for d in (0, 1)]
        ok_sep &= all(abs(wrap(math.radians(s[d] - t[d]))) >= math.radians(39.5)
                      for d in (0, 1))
        ok_sep &= all(abs(y[d] - s[d]) < 3.0 for d in (0, 1))
        draws.append((t[0], t[1], s[0], s[1]))
    print(f"[smoke] draws (t0, t1, s0, s1): {draws}", flush=True)
    allt = [d[0] for d in draws] + [d[1] for d in draws]
    check("randomization is real (targets sign+magnitude / starts vary; separation holds)",
          len({d[0] for d in draws}) >= 5 and len({d[2] for d in draws}) >= 5
          and any(v < 0 for v in allt) and any(v > 0 for v in allt) and ok_sep)

    # ========================= 4. null policy =================================================
    torch.manual_seed(seed_pick)
    env.reset()
    step(240)  # 2 s of nothing
    report("null")
    check("null policy: score < 0.05 and no success",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 5. negative: the SEED's plan (press it) ========================
    torch.manual_seed(seed_pick)
    env.reset()
    step(30)
    y_before = float(scene.knob_yaw()[0, 0])
    for k in range(240):  # 1 s press down + 1 s lateral shove
        scene.drive_f[0, 0, :] = 0.0
        if k < 120:
            scene.drive_f[0, 0, 2] = -PRESS_F
        else:
            scene.drive_f[0, 0, 0] = PRESS_F
        step(1)
    scene.drive_f[0, 0, :] = 0.0
    step(60)
    report("press-probe")
    y_after = float(scene.knob_yaw()[0, 0])
    kz = float((scene.knobs[0].data.root_pos_w[0] - scene.env_origins[0])[2])
    check("negative (seed strategy): pressing/shoving the knob changes nothing",
          abs(wrap(y_after - y_before)) < math.radians(2.0)
          and abs(kz - c.knob_z) < 0.004 and finite_all()
          and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.05)

    # ========================= 6. negative: physical end stops ================================
    torch.manual_seed(seed_pick)
    env.reset()
    step(30)
    es_d = 0
    es_side = -math.copysign(1.0, float(scene.target[0, es_d]) or 1.0)  # opposite the flag
    servo_dial(es_d, es_side * math.radians(150.0), budget=1200, w_cap=FLY_W_CAP)
    step(120)
    report("end-stop")
    y = float(scene.knob_yaw()[0, es_d])
    check("negative (end stop): commanded 150 deg — parks at the ~92 deg stop, no success",
          math.radians(80.0) <= abs(y) <= math.radians(97.0) and y * es_side > 0
          and finite_all() and not bool(scene.success()[0]))

    # ========================= 7. negative: fly-through latches nothing =======================
    torch.manual_seed(seed_pick)
    env.reset()
    step(30)
    peak = fly_through(fly_dial, fly_dest)
    report("fly-through")
    e_now = float(scene.align_err()[0, fly_dial])
    check("negative (fly-through): swept THROUGH the flag at speed, parked past it — "
          "no aligned latch, no success",
          peak > 0.6 and e_now > 1.8 * tol
          and float(scene.align_latch[0, fly_dial]) < 0.95
          and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.55)

    # ========================= 8. negative: near miss on one dial =============================
    torch.manual_seed(seed_pick)
    env.reset()
    step(30)
    t1v = float(scene.target[0, 1])
    off_dest = t1v - math.copysign(2.5 * tol, t1v)  # 15 deg off, toward the centre
    ok0 = servo_dial(0, float(scene.target[0, 0]))
    ok1 = servo_dial(1, off_dest)
    step(120)
    report("near-miss")
    a_now = scene.aligned_now()[0]
    check("negative (near miss): dial 0 on target, dial 1 parked 2.5x tol off — no success",
          ok0 and ok1 and bool(a_now[0]) and not bool(a_now[1])
          and not bool(scene.success()[0])
          and 0.20 < float(scene.score()[0]) < 0.75)

    # ========================= 9. negative: crossed flags =====================================
    torch.manual_seed(seed_pick)
    env.reset()
    step(30)
    t0v, t1v = float(scene.target[0, 0]), float(scene.target[0, 1])
    ok0 = servo_dial(0, t1v)  # each dial turned to the OTHER dial's bearing
    ok1 = servo_dial(1, t0v)
    step(120)
    report("crossed")
    check("negative (crossed flags): each pointer on the OTHER dial's bearing — no success",
          ok0 and ok1 and not bool(scene.success()[0])
          and float(scene.score()[0]) < 0.55)

    # ========================= 10. exactness: success == score 1.0 ============================
    torch.manual_seed(seed_pick)
    env.reset()
    step(30)
    ok0 = servo_dial(0, float(scene.target[0, 0]))
    ok1 = servo_dial(1, float(scene.target[0, 1]))
    step(240)
    report("goal-state")
    good = ok0 and ok1 and bool(scene.success()[0]) \
        and abs(float(scene.score()[0]) - 1.0) < 1e-3
    step(120)  # ...and it persists hands-off
    check("exactness: both dials rested on their flags -> success() and score == 1.0, stable",
          good and bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 1.0) < 1e-3)

    # ========================= 11. achievement latch ==========================================
    ok_away = servo_dial(1, float(scene.target[0, 1])
                         - math.copysign(math.radians(30.0), float(scene.target[0, 1])))
    step(120)
    report("revoked")
    check("achievement latch: turning dial 1 away revokes success; latched credit remains",
          ok_away and not bool(scene.success()[0])
          and 0.38 < float(scene.score()[0]) < 0.55)

    # ========================= 12. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.dial_setpoints")
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

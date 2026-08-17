"""Smoke / rubric-REJECTION battery for DieRollMatchScene (sim_gen task
`libero_kitchen_scene7_put_the_white_bowl_to_the_right_of_the_plate_i369`) —
NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — quarter-roll the die until the reference's
target face is up, then low push-slide it onto the crimson mat — is the acceptance
evidence; it passes on forge seeds). Every teleport here is instrumentation that
CONSTRUCTS a wrong (or partial) outcome and asserts the rubric REJECTS it — plus
physical probes that prove the tip-roll threshold and the face-preserving slide
are working mechanisms, not props. No probe in this battery ever reaches
success(), and a final audit check asserts exactly that.

   1. settle/no-NaN      — reset settles finite: die flat at counter height on its
                           sampled face, reference die displayed on the pedestal;
   2. fresh reset        — score ~0, no success;
  3-4. randomization     — READBACK over 6 seeded resets: the die's top face and
                           the displayed target face both vary, the target is
                           NEVER the initial top, the reference die's up face
                           readback equals the sampled target; start pose
                           (xy jitter + orientation) varies;
   5. null policy        — 240 idle steps -> score ~0, no success;
   6. sub-threshold push — the roll push at 0.40 mg (< the 0.583 mg tipping
                           threshold): the die neither tips nor slides — the
                           quarter-roll threshold is real physics;
   7. tip-roll mechanism — the same push at 0.70 mg genuinely rolls (top-face
                           READBACK changes) — with the pair 6+7 bracketing the
                           threshold; far from the mat, NOT success;
   8. seed-strategy      — the SEED's strategy (pure transport, no reorientation):
                           the solve's own low slide servo delivers the UNROLLED
                           die to the mat center; it arrives settled and centered
                           but the top face is unchanged (slide-preserves-face
                           READBACK) -> rejected, progress-only score;
   9. spin cheat         — yawing the die in place on the mat never changes the
                           up face: still no orientation credit, NOT success;
  10. wrong place        — die placed flat TARGET-up on the grey DECOY mat,
                           settled: every clause but in-pad true -> NOT success;
  11. place near-miss    — target-up, settled, 3 cm OUTSIDE place_tol -> rejected
                           by the placement clause alone;
  12. tilt near-miss     — die in the pad, target face up but tilted ~18 deg
                           (> flat_max_deg 10): flat clause alone rejects; the
                           die is removed before it can ring down flat;
  13. settle gate        — a perfect state (in pad, flat, target-up) judged WHILE
                           SLIDING (vel readback > settle_lin): stillness must
                           persist -> NOT success; removed before ring-down;
  14. wrong object       — the small REFERENCE die teleported onto the crimson
                           mat earns nothing (the rubric reads the big die only);
  15. rejection audit    — success() was never True at ANY judged point;
  16. final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m
  simgen_tasks.libero_kitchen_scene7_put_the_white_bowl_to_the_right_of_the_plate_i369.smoke
  --headless
"""

from __future__ import annotations

import argparse
import math

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
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:  # older isaaclab name
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

robobench.discover()
try:
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod

FACE_NAMES = scene_mod.FACE_NAMES
FACE_UP_QUATS = scene_mod.FACE_UP_QUATS

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

G = 9.81


def qmul(a, b):
    """Hamilton product (w,x,y,z) of two python quats."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.die_roll_match")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)
    mg = c.die_mass * G
    lever_roll = c.roll_push_h - c.die_half
    lever_slide = c.slide_push_h - c.die_half

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.20, -0.95, 0.85)) + o),
                                tuple(np.array((0.00, 0.10, 0.05)) + o),
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

    def top0() -> tuple[int, float]:
        idx, dom = scene.top_face()
        return int(idx[0]), float(dom[0])

    def die_p0() -> tuple[float, float, float]:
        p = scene.die_local()[0]
        return float(p[0]), float(p[1]), float(p[2])

    def report(tag: str) -> None:
        s, ok = judge()
        idx, dom = top0()
        x, y, z = die_p0()
        print(f"[smoke] {tag:16s} | top={FACE_NAMES[idx]}({dom:+.3f}) "
              f"tgt={FACE_NAMES[int(scene.target[0])]} xyz=({x:+.3f},{y:+.3f},{z:+.3f}) "
              f"pad_err={float(scene.pad_err()[0]):.3f} settled={bool(scene.settled()[0])} "
              f"| r/o/p={float(scene.roll_latch[0]):.0f}/{float(scene.orient_latch[0]):.0f}/"
              f"{float(scene.prog_latch[0]):.2f} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def write_die(xy, quat, z: float | None = None, vel=(0.0, 0.0, 0.0),
                  settle_steps: int = 60) -> None:
        """Teleport the big die (instrumentation, not a solution) + REAL physics
        steps (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.env_origins
        st[:, 0] += float(xy[0])
        st[:, 1] += float(xy[1])
        st[:, 2] += float(c.die_half + 0.004 if z is None else z)
        st[:, 3:7] = torch.tensor([float(v) for v in quat], device=device)
        st[:, 7:10] = torch.tensor([float(v) for v in vel], device=device)
        scene.die.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def wrench_off() -> None:
        scene.die.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)

    def push_high(u, fscale: float, max_steps: int) -> None:
        """The solve's own tip-roll push (CoM force + lever torque at roll_push_h,
        world wrench rotated into the body frame every step), cut past the balance
        point; leaves the wrench OFF."""
        orig_top, _ = top0()
        F = fscale * mg
        f_w = torch.zeros(n, 3, device=device)
        t_w = torch.zeros(n, 3, device=device)
        f_w[:, 0], f_w[:, 1] = F * u[0], F * u[1]
        t_w[:, 0], t_w[:, 1] = -lever_roll * F * u[1], lever_roll * F * u[0]
        for _ in range(max_steps):
            q = scene.die.data.root_quat_w
            fb = quat_apply_inverse(q, f_w).unsqueeze(1)
            tb = quat_apply_inverse(q, t_w).unsqueeze(1)
            scene.die.set_external_force_and_torque(fb, tb, env_ids=all_ids)
            env.step(no_action)
            dots = scene._axes_z(scene.die.data.root_quat_w)[0]
            if float(dots[orig_top]) < math.cos(math.radians(50.0)):
                break
        wrench_off()
        step(180)

    def slide_servo(max_steps: int = 3600) -> int:
        """The solve's own low push-slide servo toward the mat center (kinetic
        friction feedforward + velocity regulation + counter-tip lever torque);
        leaves the wrench OFF. Returns servo steps used."""
        f_w = torch.zeros(n, 3, device=device)
        t_w = torch.zeros(n, 3, device=device)
        pad = torch.tensor([c.pad_center[0], c.pad_center[1]], device=device)
        done, i = 0, 0
        for i in range(max_steps):
            p = scene.die_local()
            v = scene.die.data.root_lin_vel_w[:, 0:2]
            d = pad.unsqueeze(0) - p[:, 0:2]
            dist = d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            dirn = d / dist
            v_des = dirn * (1.5 * dist).clamp(max=0.15)
            fh = 30.0 * (v_des - v) + 7.9 * dirn
            fn = fh.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            fh = fh * (fn.clamp(max=14.0) / fn)
            f_w[:, 0:2] = fh
            t_w[:, 0] = -lever_slide * fh[:, 1]
            t_w[:, 1] = lever_slide * fh[:, 0]
            q = scene.die.data.root_quat_w
            fb = quat_apply_inverse(q, f_w).unsqueeze(1)
            tb = quat_apply_inverse(q, t_w).unsqueeze(1)
            scene.die.set_external_force_and_torque(fb, tb, env_ids=all_ids)
            env.step(no_action)
            done = done + 1 if float(scene.pad_err()[0]) <= c.place_tol - 0.02 else 0
            if done >= 3:
                break
        wrench_off()
        step(120)
        return i + 1

    bodies = (scene.die, scene.ref)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    idx, dom = top0()
    x, y, z = die_p0()
    check("settle: all states finite, die flat at counter height "
          f"(top={FACE_NAMES[idx]} dom={dom:+.3f}, z={z:+.4f} ~ {c.die_half:+.4f}) on "
          f"its sampled face (init_top={FACE_NAMES[int(scene.init_top[0])]}), reference "
          f"displayed target-up on the pedestal, settled",
          fin and dom > 0.98 and abs(z - c.die_half) <= 0.006
          and idx == int(scene.init_top[0])
          and int(scene.ref_top_face()[0]) == int(scene.target[0])
          and bool(scene.settled()[0]))
    s, ok = judge()
    check("fresh reset: score ~0, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(20)
        idx, dom = top0()
        tgt = int(scene.target[0])
        ref_top = int(scene.ref_top_face()[0])
        x, y, _z = die_p0()
        qq = tuple(round(float(v), 2) for v in scene.die.data.root_quat_w[0])
        reads.append((idx, tgt, ref_top, x, y, qq))
        print(f"[smoke] seed {sd}: top={FACE_NAMES[idx]}({dom:+.3f}) "
              f"target={FACE_NAMES[tgt]} ref_shows={FACE_NAMES[ref_top]} "
              f"xy=({x:+.3f},{y:+.3f}) q={qq}", flush=True)
    tops = {r[0] for r in reads}
    tgts = {r[1] for r in reads}
    check("randomization: top face and target face both vary by READBACK "
          f"(tops={sorted(tops)}, targets={sorted(tgts)}), the target is NEVER the "
          "initial top, and the reference die DISPLAYS the sampled target on every seed",
          len(tops) >= 2 and len(tgts) >= 2
          and all(r[1] != r[0] for r in reads) and all(r[2] == r[1] for r in reads))
    x_spread = max(r[3] for r in reads) - min(r[3] for r in reads)
    y_spread = max(r[4] for r in reads) - min(r[4] for r in reads)
    quats = {r[5] for r in reads}
    check("randomization: start pose varies by READBACK (xy spread "
          f"({x_spread * 1000:.0f},{y_spread * 1000:.0f}) mm, {len(quats)}/6 distinct "
          "orientations)", x_spread > 0.02 and y_spread > 0.02 and len(quats) >= 4)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 6. sub-threshold push (threshold is real) ==================
    env.reset(seed=41)
    step(60)
    idx0, _ = top0()
    x0, y0, _ = die_p0()
    push_high((-1.0, 0.0), 0.40, 240)  # 0.40 mg < 0.583 mg tipping threshold
    idx, dom = top0()
    x, y, _ = die_p0()
    moved = math.hypot(x - x0, y - y0)
    report("sub-threshold")
    s, ok = judge()
    check("sub-threshold push: 2 s of the roll push at 0.40 mg (below the 0.583 mg "
          f"tipping threshold, below the 0.9 mg slide threshold) neither tips nor "
          f"slides the die (top still {FACE_NAMES[idx]}, dom={dom:+.3f}, moved "
          f"{moved * 1000:.1f} mm) — the quarter-roll threshold is real physics",
          idx == idx0 and dom > 0.98 and moved <= 0.02 and not ok)

    # =========================== 7. tip-roll mechanism ======================================
    push_high((-1.0, 0.0), 0.70, 360)  # the solve's own working push level
    idx, dom = top0()
    report("tip-roll")
    s, ok = judge()
    check("tip-roll mechanism: the same push at 0.70 mg genuinely quarter-rolls the "
          f"die (top-face readback {FACE_NAMES[idx0]} -> {FACE_NAMES[idx]}, "
          f"dom={dom:+.3f}) — 6+7 bracket the threshold; far from the mat, NOT success",
          idx != idx0 and dom > 0.98 and float(scene.pad_err()[0]) > 0.3
          and s < 0.7 and not ok)

    # =========================== 8. seed-strategy: transport without reorienting ============
    env.reset(seed=51)
    step(60)
    idx0, _ = top0()
    x0, y0, _ = die_p0()
    used = slide_servo()
    idx, dom = top0()
    x, y, _ = die_p0()
    moved = math.hypot(x - x0, y - y0)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy (pure transport, no reorientation): the solve's own slide "
          f"servo delivers the UNROLLED die to the mat ({used} steps, moved "
          f"{moved:.2f} m, pad_err={float(scene.pad_err()[0]):.3f}, settled="
          f"{bool(scene.settled()[0])}) and the slide PRESERVES the top face "
          f"({FACE_NAMES[idx0]} == {FACE_NAMES[idx]}) — rejected: progress-only "
          f"score {s:.2f}, NOT success",
          idx == idx0 and dom > 0.98 and moved >= 0.30
          and float(scene.pad_err()[0]) <= c.place_tol and bool(scene.settled()[0])
          and s <= 0.25 and float(scene.orient_latch[0]) < 0.5 and not ok)

    # =========================== 9. spin cheat ==============================================
    x, y, _ = die_p0()
    q_yaw = qmul((math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)),
                 tuple(float(v) for v in scene.die.data.root_quat_w[0].cpu()))
    write_die((x, y), q_yaw, settle_steps=120)
    idx, dom = top0()
    report("spin-cheat")
    s, ok = judge()
    check("spin cheat: yawing the die 90 deg in place on the mat never changes the "
          f"up face (top still {FACE_NAMES[idx]}) — still no orientation credit, "
          "NOT success",
          idx == idx0 and dom > 0.98 and float(scene.orient_latch[0]) < 0.5
          and not ok)

    # =========================== 10. wrong place (decoy mat) ================================
    env.reset(seed=61)
    step(60)
    tgt = int(scene.target[0])
    write_die(c.decoy_center, FACE_UP_QUATS[tgt], settle_steps=180)
    idx, dom = top0()
    report("wrong-place")
    s, ok = judge()
    check("wrong place: die flat TARGET-up and settled on the grey DECOY mat "
          f"(top={FACE_NAMES[idx]} == target, dom={dom:+.3f}, settled="
          f"{bool(scene.settled()[0])}) but pad_err={float(scene.pad_err()[0]):.3f} "
          "> tol — rejected by the placement clause, NOT success",
          idx == tgt and dom > 0.98 and bool(scene.settled()[0])
          and float(scene.pad_err()[0]) > c.place_tol and not ok)

    # =========================== 11. place near-miss ========================================
    miss = c.place_tol + 0.03
    write_die((c.pad_center[0] - miss, c.pad_center[1]), FACE_UP_QUATS[tgt],
              settle_steps=180)
    report("place-miss")
    s, ok = judge()
    check("place near-miss: target-up and settled at pad_err="
          f"{float(scene.pad_err()[0]):.3f} (3 cm outside place_tol {c.place_tol:.2f}) "
          "— rejected by the placement clause alone, NOT success",
          float(scene.pad_err()[0]) > c.place_tol
          and float(scene.pad_err()[0]) < c.place_tol + 0.05
          and bool(scene.settled()[0]) and not ok)

    # =========================== 12. tilt near-miss (removed before ring-down) ==============
    a = math.radians(18.0)
    q_tilt = qmul((math.cos(a / 2), math.sin(a / 2), 0.0, 0.0), FACE_UP_QUATS[tgt])
    write_die(c.pad_center, q_tilt, z=0.093, settle_steps=0)
    step(2)
    idx, dom = top0()
    in_pad = bool(scene.in_pad()[0])
    flat = bool(scene.flat_now()[0])
    s, ok = judge()
    report("tilt-miss")
    # remove BEFORE it rings down flat target-up in the pad (that would be success)
    write_die((c.start_x, c.start_y), FACE_UP_QUATS[scene_mod.opposite_face(tgt)],
              settle_steps=60)
    check("tilt near-miss: die IN the pad with the target face up but tilted 18 deg "
          f"(dom={dom:+.3f} < cos {c.flat_max_deg:.0f} deg, in_pad={in_pad}) — the "
          "flat clause alone rejects it; removed before ring-down "
          "(battery never succeeds)",
          idx == tgt and not flat and in_pad and not ok)

    # =========================== 13. settle gate (removed before ring-down) =================
    write_die(c.pad_center, FACE_UP_QUATS[tgt], z=c.die_half + 0.001,
              vel=(0.25, 0.0, 0.0), settle_steps=0)
    step(2)
    idx, dom = top0()
    in_pad = bool(scene.in_pad()[0])
    lv = float(scene.die.data.root_lin_vel_w[0].norm())
    settled_now = bool(scene.settled()[0])
    s, ok = judge()
    report("settle-gate")
    write_die((c.start_x, c.start_y), FACE_UP_QUATS[scene_mod.opposite_face(tgt)],
              settle_steps=60)
    check("settle gate: a perfect state (in pad, flat, target-up) judged WHILE "
          f"SLIDING (|v|={lv:.3f} > settle_lin {c.settle_lin:.2f}, settled="
          f"{settled_now}) is NOT success — stillness must persist; removed before "
          "ring-down (battery never succeeds)",
          idx == tgt and dom > 0.98 and in_pad and lv > c.settle_lin
          and not settled_now and not ok)

    # =========================== 14. wrong object ===========================================
    env.reset(seed=71)
    step(60)
    tgt = int(scene.target[0])
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.env_origins
    st[:, 0] += c.pad_center[0]
    st[:, 1] += c.pad_center[1]
    st[:, 2] += c.ref_half + 0.001
    st[:, 3:7] = torch.tensor(FACE_UP_QUATS[tgt], device=device)
    scene.ref.write_root_state_to_sim(st, all_ids)
    step(90)
    rp = (scene.ref.data.root_pos_w - scene.env_origins)[0]
    report("wrong-object")
    s, ok = judge()
    check("wrong object: the small REFERENCE die teleported target-up onto the "
          f"crimson mat (at ({float(rp[0]):+.3f},{float(rp[1]):+.3f})) earns nothing "
          f"(score={s:.2f}) — the rubric reads the big die only, NOT success",
          abs(float(rp[0]) - c.pad_center[0]) < 0.02 and s <= 0.02 and not ok)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.die_roll_match")
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

"""Smoke / rubric-REJECTION battery for PileDriverScene (sim_gen task
`libero_pick_milk_i51`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — drop the slug on the red post until flush, park
the slug — is the acceptance evidence). Every teleport here is instrumentation that
CONSTRUCTS a wrong (or partial) outcome as a settled state and asserts the rubric
REJECTS it, plus physics probes that prove the mechanism is real: quasi-static
pushing genuinely CANNOT move a post, a single gravity impact genuinely CAN, and
the clamp holds depth with no creep. A few probes construct the genuine end state
on purpose (the acceptance construct and its restore-flips) — every other judged
point must stay success()=False and a final audit asserts exactly that.

  1-2. settle/no-NaN      — reset settles finite; both posts HELD at their sampled
                            proud heights by their clamps (gravity does not sink
                            them), slug standing in the holster; score ~0, no
                            success;
  3-5. randomization      — READBACK over 8 seeded resets: rig xy + free yaw vary;
                            both posts' initial proud heights vary; holster
                            position varies;
  6.  null policy         — 400 idle steps -> score ~0, no success, and neither
                            post drifts (the clamp holds against gravity);
  7.  statics fail        — a sustained 90 N downward push on the red post head
                            (a full-arm lean, below the ~200 N calibrated grip) for
                            1.5 s advances it < 4 mm: quasi-static force is the
                            WRONG strategy, the seed's "just move it" cannot work;
  8.  impulse works       — ONE gravity drop of the slug from hover advances the
                            red post >= 4 mm: the impact force spike breaks the
                            same clamp that statics cannot — the mechanism is real;
  9.  no creep            — the slug left STANDING on the red post head for 2 s
                            advances it < 2 mm more (resting weight ~12 N is far
                            below the grip): progress comes only from impacts;
  10. latches persist     — carrying the slug far away afterwards does not erase
                            earned score (latched rubric);
  11. wrong post          — the WHITE post driven by the same drops: white_ok goes
                            False, and even constructing the red post flush AND
                            parking the slug afterwards is STILL rejected — the
                            mistake is irreversible;
  12. near miss           — the red post at 22 mm proud (outside the 12 mm flush
                            band) with the slug parked: rejected;
  13. acceptance construct— the red post placed INSIDE the flush band, white
                            untouched, slug parked (as it spawns), settled ->
                            success TRUE, score 1.0;
  14. park clause         — the slug taken out of the holster and left on the
                            floor: success flips FALSE; stood back in: TRUE again
                            (13's acceptance needed the parked slug and nothing
                            else changed);
  15. settle gate         — the slug kicked in the accepted state and judged
                            immediately: NOT success (must be at rest);
  16. white clause        — the white post driven 30 mm down in the accepted
                            state: success flips FALSE (decoy clause);
  17. rejection audit     — success() was never True at any judged point EXCEPT
                            the constructed acceptance probes (13, 14-restore);
  18. final no-NaN        — all task-object states finite at the end.

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
        DECK_TOP, HOL_T, L_POST, SLUG_H, Y_POST, _qapply, _qz,
    )
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import (  # noqa: F401
        DECK_TOP, HOL_T, L_POST, SLUG_H, Y_POST, _qapply, _qz,
    )
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

DROP_H = 0.14  # solve's full-power hover clearance (slug bottom above the head)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.pile_driver")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.05, -0.85, 0.80)) + o),
                                tuple(np.array((0.00, 0.00, 0.18)) + o),
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

    def proud_r() -> float:
        return float(scene.proud(scene.post_red)[0])

    def proud_w() -> float:
        return float(scene.proud(scene.post_white)[0])

    def report(tag: str, s: float, ok: bool) -> None:
        print(f"[smoke] {tag:18s} | proud_red={proud_r() * 1000:+6.1f}mm "
              f"proud_white={proud_w() * 1000:+6.1f}mm "
              f"flush={bool(scene.red_flush()[0])} "
              f"white_ok={bool(scene.white_ok()[0])} "
              f"parked={bool(scene.slug_parked()[0])} "
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

    def post_pose(y_side: float, proud: float):
        """World pose that puts a post at the given proud height in the rig frame."""
        q_rig = scene.rig.data.root_quat_w
        local = torch.tensor([0.0, y_side * Y_POST, DECK_TOP + proud - L_POST / 2],
                             device=device).expand(n, 3)
        return scene.rig.data.root_pos_w + _qapply(q_rig, local), q_rig

    def place_post(y_side: float, proud_target: float, tol: float = 0.004,
                   iters: int = 6) -> None:
        """Construct a post AT a target proud height. Writing a post's root state
        re-forms the pad squeeze, which lifts the post ~13 mm while the clamp
        re-latches (the settle-lift artifact) — so place ITERATIVELY: write,
        settle, measure the landing error, correct the write height by it."""
        read = proud_r if y_side > 0 else proud_w
        body = scene.post_red if y_side > 0 else scene.post_white
        tgt = proud_target
        for _ in range(iters):
            pos, q = post_pose(y_side, tgt)
            teleport(body, pos, q, settle_steps=60)
            err = read() - proud_target
            if abs(err) <= tol:
                return
            tgt -= err
        print(f"[smoke] place_post: residual {read() - proud_target:+.4f}m after "
              f"{iters} iterations (target {proud_target:.3f})", flush=True)

    def slug_hover(y_side: float, drop_h: float = DROP_H):
        """World pose hovering the slug drop_h above the given post's head."""
        body = scene.post_red if y_side > 0 else scene.post_white
        pos = body.data.root_pos_w.clone()
        pos[:, 2] += L_POST / 2 + drop_h + SLUG_H / 2
        return pos, scene.rig.data.root_quat_w

    def slug_park_pose():
        pos = scene.holster.data.root_pos_w.clone()
        pos[:, 2] += HOL_T + SLUG_H / 2 + 0.015
        return pos, scene.holster.data.root_quat_w

    def slug_floor_pose(dx: float, dy: float):
        pos = scene.rig.data.root_pos_w.clone()
        pos[:, 0] += dx
        pos[:, 1] += dy
        pos[:, 2] = scene.env_origins[:, 2] + SLUG_H / 2 + 0.003
        return pos, torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4)

    def settle_all(max_steps: int = 600) -> None:
        for _ in range(max_steps // 30):
            step(30)
            if bool(scene.all_settled()[0]):
                break

    def push_post(body, force_n: float, steps: int) -> None:
        """Sustained straight-down push on a post head. The posts never rotate
        after reset (yaw-only rig, vertical prismatic guide), so a -z force is
        frame-drag-safe in every force-frame convention."""
        f = torch.tensor([0.0, 0.0, -force_n], device=device).expand(n, 3).clone()
        for _ in range(steps):
            try:
                body.set_external_force_and_torque(f.view(n, 1, 3), zero_wrench,
                                                   env_ids=all_ids, is_global=True)
            except TypeError:  # older API without is_global
                body.set_external_force_and_torque(f.view(n, 1, 3), zero_wrench,
                                                   env_ids=all_ids)
            env.step(no_action)
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def all_finite() -> bool:
        bodies = [scene.rig, scene.post_red, scene.post_white, scene.pad_rn,
                  scene.pad_rp, scene.pad_wn, scene.pad_wp, scene.holster, scene.slug]
        return all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    p_red0, p_wht0 = float(scene.proud0_red[0]), float(scene.proud0_white[0])
    settle_all(480)
    s, ok = judge()
    report("reset", s, ok)
    # proud0 re-latches from readback during the 60-step warmup; after settling the
    # posts must sit ON that latched height (clamps hold), which itself must be
    # close to the sampled height (the settle transient is small).
    pr0, pw0 = float(scene.proud0_red[0]), float(scene.proud0_white[0])
    held = (abs(proud_r() - pr0) < 0.002 and abs(proud_w() - pw0) < 0.002
            and abs(pr0 - p_red0) < 0.020 and abs(pw0 - p_wht0) < 0.020)
    check("settle: all states finite, both posts HELD at their (warmup-latched) "
          f"proud heights by their clamps (red sampled {p_red0 * 1000:.0f} -> latched "
          f"{pr0 * 1000:.1f} -> now {proud_r() * 1000:.1f}mm, white "
          f"{p_wht0 * 1000:.0f} -> {pw0 * 1000:.1f} -> {proud_w() * 1000:.1f}mm), "
          "slug in the holster",
          all_finite() and held and bool(scene.slug_parked()[0]))
    check(f"settle: score ~0 and no success on a fresh reset (score={s:.3f})",
          s <= 0.03 and not ok)

    # =========================== 3-5. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(20)
        rp = (scene.rig.data.root_pos_w - scene.env_origins)[0]
        ex = _qapply(scene.rig.data.root_quat_w,
                     torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))[0]
        yaw = math.atan2(float(ex[1]), float(ex[0]))
        hp = (scene.holster.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(rp[0]), float(rp[1]), yaw, proud_r(), proud_w(),
                      float(hp[0]), float(hp[1])))
    arr = np.array(reads)
    print("[smoke] randomization readback (rig_x, rig_y, rig_yaw, proud_red, "
          f"proud_white, hol_x, hol_y):\n{arr.round(3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: rig pose varies across seeded resets (readback: "
          f"dx={spread[0]:.3f} dy={spread[1]:.3f} dyaw={spread[2]:.2f} rad)",
          spread[0] > 0.02 and spread[1] > 0.02 and spread[2] > 0.8)
    check("randomization: both posts' initial proud heights vary (readback: "
          f"dred={spread[3] * 1000:.1f}mm dwhite={spread[4] * 1000:.1f}mm)",
          spread[3] > 0.004 and spread[4] > 0.004)
    check("randomization: holster position varies (readback: "
          f"dx={spread[5]:.3f} dy={spread[6]:.3f})",
          max(spread[5], spread[6]) > 0.05)

    # =========================== 6. null policy fails =======================================
    env.reset(seed=31)
    step(60)
    p0 = proud_r()
    w0 = proud_w()
    step(400)
    s, ok = judge()
    report("null-policy", s, ok)
    drift = max(abs(proud_r() - p0), abs(proud_w() - w0))
    check("null policy: score ~0, no success, and neither post drifts under gravity "
          f"after 400 idle steps (score={s:.3f}, drift={drift * 1000:.2f}mm)",
          s <= 0.03 and not ok and drift < 0.002)

    # =========================== 7. quasi-static pushing fails ==============================
    p0 = proud_r()
    push_post(scene.post_red, 90.0, 360)  # 1.5 s of a 90 N full-arm lean
    step(60)
    s, ok = judge()
    report("static-push", s, ok)
    adv_push = p0 - proud_r()
    check("statics fail: a sustained 90 N downward push on the red post head for "
          f"1.5 s advances it only {adv_push * 1000:.2f}mm (< 4mm) — the clamp's "
          f"~{c.brake_static:.0f} N static grip defeats quasi-static force",
          adv_push < 0.004 and not ok)

    # =========================== 8. one gravity impact works ================================
    p0 = proud_r()
    pos, q = slug_hover(1.0)
    teleport(scene.slug, pos, q, settle_steps=0)
    settle_all(600)
    s, ok = judge()
    report("one-drop", s, ok)
    adv_drop = p0 - proud_r()
    check("impulse works: ONE gravity drop of the slug from hover advances the red "
          f"post {adv_drop * 1000:.1f}mm (>= 4mm) — the impact force spike breaks "
          "the same clamp that statics cannot", adv_drop >= 0.004 and not ok)

    # =========================== 9. no creep under the resting slug =========================
    head_pos = scene.post_red.data.root_pos_w.clone()
    head_pos[:, 2] += L_POST / 2 + SLUG_H / 2 + 0.001
    teleport(scene.slug, head_pos, scene.rig.data.root_quat_w, settle_steps=60)
    p0 = proud_r()
    step(480)  # 2 s standing on the head
    s, ok = judge()
    report("resting-slug", s, ok)
    creep = p0 - proud_r()
    check("no creep: the slug left STANDING on the red post head for 2 s advances it "
          f"only {creep * 1000:.2f}mm more (< 2mm) — resting weight is far below the "
          "grip; progress comes only from impacts", creep < 0.002 and not ok)

    # =========================== 10. latches persist ========================================
    s_before = float(scene.score()[0])
    pos, q = slug_floor_pose(0.55, 0.40)
    teleport(scene.slug, pos, q, settle_steps=120)
    s_after, ok = judge()
    report("latch-persist", s_after, ok)
    check("latches persist: carrying the slug far away does not erase earned score "
          f"({s_before:.3f} -> {s_after:.3f})", s_after >= s_before - 1e-6 and not ok)

    # =========================== 11. wrong post (irreversible) ==============================
    env.reset(seed=41)
    settle_all(300)
    for _ in range(3):  # drive the WHITE post by the same honest drops
        pos, q = slug_hover(-1.0)
        teleport(scene.slug, pos, q, settle_steps=0)
        settle_all(480)
    wht_moved = float(scene.proud0_white[0]) - proud_w()
    s, ok = judge()
    report("wrong-post", s, ok)
    check("wrong post: the WHITE post driven down by the same drops "
          f"({wht_moved * 1000:.1f}mm > tol {c.white_tol * 1000:.0f}mm) -> white_ok "
          "False, no success", wht_moved > c.white_tol and not ok
          and not bool(scene.white_ok()[0]))
    # ... and the mistake is IRREVERSIBLE: constructing the red post flush and
    # parking the slug afterwards is still rejected.
    place_post(1.0, 0.006)
    pos, q = slug_park_pose()
    teleport(scene.slug, pos, q, settle_steps=0)
    settle_all(480)
    s, ok = judge()
    report("wrong-post-then-fix", s, ok)
    check("wrong post is IRREVERSIBLE: red flush constructed AND slug parked after "
          f"the white mistake is STILL rejected (score={s:.3f}, success={ok})",
          not ok and bool(scene.red_flush()[0]) and bool(scene.slug_parked()[0]))

    # =========================== 12. near miss ==============================================
    env.reset(seed=51)
    settle_all(300)
    place_post(1.0, 0.022)  # 22 mm proud: outside the 12 mm flush band
    step(30)
    s, ok = judge()
    report("near-miss", s, ok)
    check("near miss: the red post at 22 mm proud (outside the flush band) with the "
          f"slug parked is rejected (proud={proud_r() * 1000:.1f}mm, score={s:.3f})",
          not ok and not bool(scene.red_flush()[0]) and bool(scene.slug_parked()[0]))

    # =========================== 13. acceptance construct ===================================
    place_post(1.0, 0.006)  # inside the flush band
    settle_all(480)
    s, ok = judge_accept()
    report("accept-construct", s, ok)
    check("acceptance construct: red post INSIDE the flush band "
          f"(proud={proud_r() * 1000:.1f}mm), white untouched, slug parked, settled "
          f"-> success TRUE, score={s:.3f} >= 0.99", ok and s >= 0.99)

    # =========================== 14. park clause flips success ==============================
    pos, q = slug_floor_pose(0.50, -0.35)
    teleport(scene.slug, pos, q, settle_steps=120)
    s, ok = judge()
    report("slug-out", s, ok)
    still_flush = bool(scene.red_flush()[0]) and bool(scene.white_ok()[0])
    check("park clause: the slug taken out of the holster flips success FALSE while "
          f"the red post is STILL flush (success={ok})", still_flush and not ok
          and not bool(scene.slug_parked()[0]))
    pos, q = slug_park_pose()
    teleport(scene.slug, pos, q, settle_steps=0)
    settle_all(480)
    s, ok = judge_accept()
    report("slug-restored", s, ok)
    check("park clause: the slug stood back in the holster -> success TRUE again "
          "(14's rejection was the park clause and nothing else)", ok)

    # =========================== 15. settle gate ============================================
    pos = scene.slug.data.root_pos_w
    q = scene.slug.data.root_quat_w
    teleport(scene.slug, pos, q, vel=[0.0, 0.0, 0.30], ang=[0.0, 0.0, 5.0],
             settle_steps=0)
    step(1)  # refresh buffers only — judge while still moving
    lv = float(scene.slug.data.root_lin_vel_w[0].norm())
    av = float(scene.slug.data.root_ang_vel_w[0].norm())
    s, ok = judge()
    gate_ok = (lv > c.settle_lin or av > c.settle_ang) and not ok
    report("settle-gate", s, ok)
    check("settle gate: the slug kicked (lin={:.2f} m/s, ang={:.1f} rad/s) in the "
          "accepted state and judged immediately is NOT success (must be at rest)"
          .format(lv, av), gate_ok)
    settle_all(480)  # let it re-seat (not judged as acceptance again)

    # =========================== 16. white clause ===========================================
    place_post(-1.0, float(scene.proud0_white[0]) - 0.030)
    step(30)
    s, ok = judge()
    report("white-driven", s, ok)
    check("white clause: the white post driven 30 mm down in the accepted state "
          f"flips success FALSE (white_ok={bool(scene.white_ok()[0])})",
          not ok and not bool(scene.white_ok()[0]))

    # =========================== 17-18. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point except the "
          "constructed acceptance probes", not ever_bad_success[0])
    check("final: all task-object states finite (no NaN)", all_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.pile_driver")
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

"""Smoke / rubric-REJECTION battery for HookedBasketScene (sim_gen task
`living_room_scene2_pick_up_the_tomato_sauce_and_put_it_in_the_basket_i149`) —
NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — gravity insertion through the mouth opening, then a
composite carry staged above the hook arm and a gravity seat — is the acceptance
evidence that the rubric ACCEPTS a correct outcome). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled state and
asserts the rubric REJECTS it. One check (the spill check) deliberately constructs the
TRUE outcome to test latch non-evaporation; that success is sampled outside judge(),
so the rejection audit still asserts that no REJECTION probe ever reached success().

  1-2. settle/no-NaN      — reset layout settles finite: basket upright on the ground,
                            cans upright in their bands, score ~0 at rest;
  3-4. randomization      — READBACK over 6 seeded resets: stand xy + yaw, basket
                            xy + yaw, both can positions all vary; the can y-bands
                            SWAP across seeds (both orders observed); cans never
                            spawn within 10 cm of each other;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  SEED END STATE      — the seed task's goal (red can settled inside the basket,
                            basket standing on the GROUND) is explicitly NOT success
                            here and earns only the containment latch (score <= 0.36);
  7.  wrong object        — full suspension built with the BEIGE can inside the hung
                            basket, red can on the ground -> no success, score ~0
                            (every latch gates on the red can);
  8.  empty hang          — the EMPTY basket properly seated on the hook arm -> no
                            success, score ~0 (hanging earns nothing without the can);
  9.  wrong support       — loaded basket PERCHED on the post top: elevated, upright,
                            can inside, at rest — but the bar is not seated on the
                            arm -> NOT success, score <= 0.51 (no hang credit);
  10. exclusion           — BOTH cans inside the properly hung basket -> hang latch
                            True yet NOT success (beige must stay out), score <= 0.80;
  11. spill + latches     — a genuine success hang is constructed (sampled outside the
                            audit), then the red can is teleported out to the ground:
                            success turns False but the latched score stays 0.80
                            (credit non-evaporation, live success gating);
  12. rejection audit     — success() was never True at ANY judged rejection probe;
  13. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.living_room_scene2_pick_up_the_tomato_sauce_and_put_it_in_the_basket_i149.smoke --headless
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
    from isaaclab.utils.math import quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.hooked_basket")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.35, -1.25, 0.95)) + o),
                                tuple(np.array((0.35, 0.00, 0.25)) + o),
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

    def stand_pose() -> tuple[torch.Tensor, float]:
        sp = (scene.stand.data.root_pos_w - scene.env_origins)[0]
        q = scene.stand.data.root_quat_w[0]
        return sp, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def basket_pose() -> tuple[torch.Tensor, float]:
        bp = (scene.basket.data.root_pos_w - scene.env_origins)[0]
        q = scene.basket.data.root_quat_w[0]
        return bp, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def report(tag: str) -> None:
        bl = scene._bar_stand_local()[0]
        tl = scene._basket_local(scene.tomato)[0]
        bz = float((scene.basket.data.root_pos_w - scene.env_origins)[0, 2])
        s, ok = judge()
        print(f"[smoke] {tag:16s} | bar_stand=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) tomato_local=({float(tl[0]):+.3f},{float(tl[1]):+.3f},"
              f"{float(tl[2]):.3f}) basket_z={bz:.3f} in={bool(scene._in[0])} "
              f"lift={bool(scene._lift[0])} hang={bool(scene._hang[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def put_can_in_basket(can, bp: torch.Tensor, bq: torch.Tensor, y_l: float,
                          settle_steps: int = 0) -> None:
        """Teleport a can to a resting pose inside the basket cavity, basket-local
        lateral offset `y_l` (instrumentation, not a solution)."""
        loc = torch.zeros(n, 3, device=device)
        loc[:, 1] = y_l
        loc[:, 2] = c.floor_t + c.can_h / 2 + 0.004
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = bp + quat_apply(bq, loc)
        st[:, 3:7] = bq
        can.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def hang_basket(cans=(), stage_gap: float = 0.022, settle_steps: int = 480) -> None:
        """Composite-teleport the basket (with the given (can, y_local) pairs placed
        inside) to the solve's staging pose above the hook arm, then let gravity seat
        the bar and the pendulum ring down."""
        _sp, syaw = stand_pose()
        byaw = syaw + math.pi / 2
        bq = torch.zeros(n, 4, device=device)
        bq[:, 0], bq[:, 3] = math.cos(byaw / 2), math.sin(byaw / 2)
        bar_stand = torch.tensor([c.seat_x_mid, 0.0, c.peg_top_z + stage_gap + c.bar_t / 2],
                                 device=device).expand(n, 3)
        bar_w = scene.stand.data.root_pos_w + quat_apply(scene.stand.data.root_quat_w, bar_stand)
        ez = torch.zeros(n, 3, device=device)
        ez[:, 2] = c.bar_z
        bp = bar_w - quat_apply(bq, ez)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3], st[:, 3:7] = bp, bq
        scene.basket.write_root_state_to_sim(st, all_ids)
        for can, y_l in cans:
            put_can_in_basket(can, bp, bq, y_l)
        step(settle_steps)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=100)
    report("reset")
    step(60)
    report("show")
    fin0 = (torch.isfinite(scene.basket.data.root_state_w).all()
            and torch.isfinite(scene.tomato.data.root_state_w).all()
            and torch.isfinite(scene.distractor.data.root_state_w).all()
            and torch.isfinite(scene.stand.data.root_state_w).all())
    bp0, _ = basket_pose()
    tz = float((scene.tomato.data.root_pos_w - scene.env_origins)[0, 2])
    still = (float(scene.basket.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.tomato.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    check("settle: states finite, basket upright on the ground, red can upright on the "
          "ground, everything at rest",
          bool(fin0) and abs(float(bp0[2])) < 0.01 and float(scene._basket_up()[0]) > 0.99
          and abs(tz - c.can_h / 2) < 0.01 and still)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (101, 102, 103, 104, 105, 106):
        env.reset(seed=sd)
        step(5)
        sp, syaw = stand_pose()
        bp, byaw = basket_pose()
        tp = (scene.tomato.data.root_pos_w - scene.env_origins)[0]
        dp = (scene.distractor.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(sp[0]), float(sp[1]), syaw, float(bp[0]), float(bp[1]), byaw,
                      float(tp[1]), float(dp[1]), float((tp[:2] - dp[:2]).norm())))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (stand_x, stand_y, stand_yaw, basket_x, "
          f"basket_y, basket_yaw, tomato_y, distractor_y, can_sep):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: stand xy + yaw and basket xy + yaw vary across seeded resets "
          "(readback)",
          spread[0] > 0.02 and spread[1] > 0.02 and spread[2] > 0.15
          and spread[3] > 0.03 and spread[4] > 0.03 and spread[5] > 0.5)
    order = np.sign(arr[:, 6] - arr[:, 7])  # tomato_y vs distractor_y: band order
    check("randomization: can y-bands vary AND swap across seeds (both orders observed), "
          "cans never spawn within 10 cm of each other (readback)",
          spread[6] > 0.05 and spread[7] > 0.05 and (order > 0).any() and (order < 0).any()
          and float(arr[:, 8].min()) >= 0.10)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=107)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. SEED END STATE: grounded containment ====================
    # The seed task's whole goal — red can settled inside the basket, basket standing on
    # the ground — is this task's demoted intermediate: containment latch only.
    env.reset(seed=108)
    step(30)
    bp = scene.basket.data.root_pos_w.clone()
    bq = scene.basket.data.root_quat_w.clone()
    put_can_in_basket(scene.tomato, bp, bq, 0.047, settle_steps=150)
    report("seed-end-state")
    s, ok = judge()
    bz = float((scene.basket.data.root_pos_w - scene.env_origins)[0, 2])
    check("SEED END STATE: red can settled inside the GROUNDED basket — containment "
          "latched but NOT success, score <= 0.36",
          bool(scene._in[0]) and bool(scene._inside_now(scene.tomato)[0]) and bz < 0.02
          and not ok and s <= 0.36)

    # =========================== 7. wrong object: beige can hung ============================
    env.reset(seed=109)
    step(30)
    hang_basket(cans=[(scene.distractor, 0.047)])
    report("wrong-object")
    s, ok = judge()
    bl = scene._bar_stand_local()[0]
    seated_geom = (c.seat_x0 < float(bl[0]) < c.seat_x1
                   and abs(float(bl[1])) < c.seat_y_tol
                   and c.peg_top_z + c.seat_z_lo < float(bl[2]) < c.peg_top_z + c.seat_z_hi)
    check("wrong object: BEIGE can inside the properly HUNG basket, red can on the "
          "ground — bar seated yet no success, score <= 0.02 (all credit gates on the "
          "red can)",
          seated_geom and bool(scene._inside_now(scene.distractor)[0]) and not ok
          and s <= 0.02)

    # =========================== 8. empty hang ==============================================
    env.reset(seed=110)
    step(30)
    hang_basket(cans=[])
    report("empty-hang")
    s, ok = judge()
    bl = scene._bar_stand_local()[0]
    seated_geom = (c.seat_x0 < float(bl[0]) < c.seat_x1
                   and abs(float(bl[1])) < c.seat_y_tol
                   and c.peg_top_z + c.seat_z_lo < float(bl[2]) < c.peg_top_z + c.seat_z_hi)
    bz = float((scene.basket.data.root_pos_w - scene.env_origins)[0, 2])
    check("empty hang: EMPTY basket properly seated on the hook arm, elevated and at "
          "rest — no success, score <= 0.02 (hanging earns nothing without the can)",
          seated_geom and bz > c.elevated_min and not ok and s <= 0.02)

    # =========================== 9. wrong support: perched on the post top ==================
    # Elevated + upright + can inside + at rest, but supported by the POST TOP, not by
    # the bar on the arm: the stand-frame seat window rejects it.
    env.reset(seed=111)
    step(30)
    sp, syaw = stand_pose()
    bq = torch.zeros(n, 4, device=device)
    bq[:, 0], bq[:, 3] = math.cos(syaw / 2), math.sin(syaw / 2)
    top_local = torch.tensor([0.0, 0.0, c.post_h + 0.004], device=device).expand(n, 3)
    bp = scene.stand.data.root_pos_w + quat_apply(scene.stand.data.root_quat_w, top_local)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3], st[:, 3:7] = bp, bq
    scene.basket.write_root_state_to_sim(st, all_ids)
    put_can_in_basket(scene.tomato, bp, bq, 0.0)  # centred: CoM over the post face
    step(360)
    report("post-perch")
    s, ok = judge()
    bl = scene._bar_stand_local()[0]
    bz = float((scene.basket.data.root_pos_w - scene.env_origins)[0, 2])
    in_window = (c.seat_x0 < float(bl[0]) < c.seat_x1
                 and c.peg_top_z + c.seat_z_lo < float(bl[2]) < c.peg_top_z + c.seat_z_hi)
    check("wrong support: loaded basket PERCHED on the post top — elevated, upright, can "
          "inside, at rest, but the bar is nowhere near the arm's seat window: NOT "
          "success, score <= 0.51 (no hang credit)",
          bool(scene._inside_now(scene.tomato)[0]) and bz > c.elevated_min
          and not in_window and not ok and s <= 0.51)

    # =========================== 10. exclusion: both cans in the hung basket ================
    env.reset(seed=112)
    step(30)
    hang_basket(cans=[(scene.tomato, 0.047), (scene.distractor, -0.047)])
    report("both-cans")
    s, ok = judge()
    check("exclusion: BOTH cans inside the properly hung basket — hang latch True yet "
          "NOT success (the beige can must stay out), score <= 0.80 + eps",
          bool(scene._hang[0]) and bool(scene._inside_now(scene.tomato)[0])
          and bool(scene._inside_now(scene.distractor)[0]) and not ok and s <= 0.8002)

    # =========================== 11. spill: latched credit, live success gating =============
    env.reset(seed=113)
    step(30)
    hang_basket(cans=[(scene.tomato, 0.047)])
    # Deliberate TRUE outcome (sampled OUTSIDE judge(): the rejection audit stays clean).
    reached = False
    streak = 0
    for _ in range(40):
        step(40)
        if bool(scene.success()[0]):
            streak += 1
            if streak >= 3:
                reached = True
                break
        else:
            streak = 0
    print(f"[smoke] spill-check: constructed genuine hang, success reached = {reached}",
          flush=True)
    # Spill: teleport the red can out of the hanging basket onto open ground.
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1], st[:, 2] = -0.20, -0.30, c.can_h / 2 + 0.002
    st[:, 3] = 1.0
    st[:, 0:3] += scene.env_origins
    scene.tomato.write_root_state_to_sim(st, all_ids)
    step(150)
    report("spilled")
    s, ok = judge()
    check("spill: a genuine success hang was constructed, then the red can teleported "
          "out to the ground — success turns False but the latched score stays 0.80 "
          "(credit non-evaporation, live gating)",
          reached and not ok and 0.7998 <= s <= 0.8002)

    # =========================== 12-13. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged rejection probe",
          not ever_success[0])
    fin = (torch.isfinite(scene.basket.data.root_state_w).all()
           and torch.isfinite(scene.stand.data.root_state_w).all()
           and torch.isfinite(scene.tomato.data.root_state_w).all()
           and torch.isfinite(scene.distractor.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.hooked_basket")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, ok in checks:
            if not ok:
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

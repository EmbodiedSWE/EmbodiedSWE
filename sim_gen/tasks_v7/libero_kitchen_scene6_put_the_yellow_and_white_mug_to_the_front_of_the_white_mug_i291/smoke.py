"""Smoke / rubric-REJECTION battery for MugAlcoveScene (sim_gen task
`libero_kitchen_scene6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug_i291`)
— NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — teleport to a doorway staging pose, force-push
through the doorway under the awning, precision stop, hands-off settle — is the
acceptance evidence that the rubric ACCEPTS a correct outcome). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled state and
asserts the rubric REJECTS it; no probe in this battery ever reaches success(), and a
final audit check asserts exactly that.

  1-2. settle/no-NaN   — reset layout settles finite: alcove inside its jitter box,
                         white mug standing in its depth band inside the alcove,
                         yellow mug upright outside, everything at rest, score ~0;
  3-4. randomization   — READBACK over 6 seeded resets: alcove xy + yaw vary; white
                         mug depth/lateral and yellow mug spawn pose + free yaw vary;
                         the white mug is always inside, the yellow always outside;
  5.  null policy      — 240 idle steps -> score ~0, no success;
  6.  seed strategy    — the closest state the seed's top-down place can reach (the
                         awning blocks the goal zone from above): the yellow mug set
                         down just OUTSIDE the sill, aligned with the white mug ->
                         no success, approach credit only;
  7.  short near-miss  — settled inside the doorway but NOT fully past the sill
                         (local x just under `x_in_lo`) -> no success;
  8.  crowding         — settled fully inside but pressed to ~2.5 mm surface gap
                         from the white mug (< gap_min) -> no success;
  9.  off-axis         — settled at goal depth but 7 cm off the white mug's lane
                         (> y_tol) -> no success;
  10. toppled          — lying on its side at goal depth under the awning -> no
                         success (upright gate);
  11. white disturbed  — the white mug displaced 6 cm sideways, the yellow mug then
                         settled PERFECTLY in front of it (placement readback all
                         true) -> still no success (the don't-disturb gate) — this
                         also rejects "solve the relation by moving the WHITE mug";
  12. roof parking     — the yellow mug parked ON TOP of the awning directly above
                         the goal zone -> no success, no entry credit;
  13. fly-through      — the yellow mug teleported DIRECTLY into the exact goal pose
                         (zero velocity) and judged WITHOUT stepping -> NOT success
                         (pose-jump guard + streak);
  14. latch survives   — a few real frames in the goal window then yanked back
                         outside: latched credit persists at the 0.67 cap, but
                         success never fired at any point;
  15. rejection audit  — success() was never True at ANY judged point;
  16. final no-NaN     — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug_i291.smoke --headless
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
# RTX recipe: kit mis-decodes the L20 driver version and silently rejects RTX -> the
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)",
                                             flush=True), os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.mug_alcove")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.55, -1.35, 1.00)) + o),
                                tuple(np.array((0.45, 0.00, 0.10)) + o),
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

    def y_loc() -> torch.Tensor:
        return scene._local(scene.yellow.data.root_pos_w)[0]

    def w_loc() -> torch.Tensor:
        return scene._local(scene.white.data.root_pos_w)[0]

    def report(tag: str) -> None:
        yl, wl = y_loc(), w_loc()
        s, ok = judge()
        print(f"[smoke] {tag:14s} | y_loc=({float(yl[0]):+.3f},{float(yl[1]):+.3f},"
              f"{float(yl[2]):.3f}) w_loc=({float(wl[0]):+.3f},{float(wl[1]):+.3f}) "
              f"placed={bool(scene._placed_now()[0])} "
              f"intact={bool(scene._white_intact()[0])} still={int(scene._still[0])} "
              f"appr={bool(scene._approached[0])} ent={bool(scene._entered[0])} "
              f"ins={bool(scene._inside[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def alcove_yaw() -> float:
        q = scene.alcove.data.root_quat_w[0]
        return 2.0 * math.atan2(float(q[3]), float(q[0]))

    def place(obj, x_l: float, y_l: float, z: float, yaw_rel: float,
              settle_steps: int = 0, extra_rot: torch.Tensor | None = None) -> None:
        """One root-state write at an ALCOVE-LOCAL pose (yaw_rel about z relative to
        the alcove frame; optional extra pre-rotation e.g. for a lying mug)."""
        from isaaclab.utils.math import quat_mul

        aq = scene.alcove.data.root_quat_w
        apos = scene.alcove.data.root_pos_w
        loc = torch.tensor([x_l, y_l, z], device=device)
        pos = apos[0] + quat_apply(aq, loc.unsqueeze(0))[0]
        yaw = alcove_yaw() + yaw_rel
        q = torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
                         device=device)
        if extra_rot is not None:
            q = quat_mul(q.unsqueeze(0), extra_rot.unsqueeze(0))[0]
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos.unsqueeze(0)
        st[:, 3:7] = q.unsqueeze(0)
        obj.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def goal_x() -> float:
        wl = w_loc()
        x_hi = min(c.roof_len - c.body_r, float(wl[0]) - c.gap_min)
        return 0.5 * (c.x_in_lo + x_hi)

    z0 = c.body_h / 2 + 0.002

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    fin0 = (torch.isfinite(scene.alcove.data.root_state_w).all()
            and torch.isfinite(scene.white.data.root_state_w).all()
            and torch.isfinite(scene.yellow.data.root_state_w).all())
    ap = (scene.alcove.data.root_pos_w - scene.env_origins)[0]
    wl, yl = w_loc(), y_loc()
    still = (float(scene.yellow.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.white.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    check("settle: states finite, alcove inside its jitter box, white mug standing "
          "in its depth band inside the alcove, yellow mug upright outside, at rest",
          bool(fin0)
          and abs(float(ap[0]) - c.alcove_pos[0]) < c.alcove_jitter + 0.005
          and abs(float(ap[1]) - c.alcove_pos[1]) < c.alcove_jitter + 0.005
          and c.white_x_range[0] - 0.01 < float(wl[0]) < c.white_x_range[1] + 0.01
          and abs(float(wl[1])) < c.white_y_jitter + 0.01
          and float(yl[0]) < -0.26 and bool(scene._upright(scene.yellow)[0])
          and bool(scene._upright(scene.white)[0]) and still)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        ap = (scene.alcove.data.root_pos_w - scene.env_origins)[0]
        wl, yl = w_loc(), y_loc()
        yq = scene.yellow.data.root_quat_w[0]
        yyaw = 2.0 * math.atan2(float(yq[3]), float(yq[0]))
        reads.append((float(ap[0]), float(ap[1]), alcove_yaw(), float(wl[0]),
                      float(wl[1]), float(yl[0]), float(yl[1]), yyaw))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (alcove_x, alcove_y, alcove_yaw, "
          f"white_xloc, white_yloc, yellow_xloc, yellow_yloc, yellow_yaw):\n{arr}",
          flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: alcove xy + yaw vary across seeded resets (readback)",
          spread[0] > 0.008 and spread[1] > 0.008 and spread[2] > 0.03)
    check("randomization: white mug depth/lateral and yellow spawn pose + free yaw "
          "vary; white always inside its band, yellow always outside the alcove",
          spread[3] > 0.005 and spread[4] > 0.008 and spread[5] > 0.02
          and spread[6] > 0.02 and spread[7] > 0.5
          and float(arr[:, 3].min()) > c.white_x_range[0] - 0.01
          and float(arr[:, 3].max()) < c.white_x_range[1] + 0.01
          and float(arr[:, 5].max()) < -0.26)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 6. seed strategy: top-down place at the doorway ============
    # The seed's whole plan is a top-down pick-and-place "to the front of the white
    # mug". The awning makes the true goal zone unreachable from above; the closest
    # top-down placement is just OUTSIDE the sill, aligned with the white mug. It
    # earns approach credit only — no success.
    torch.manual_seed(41)
    env.reset()
    step(10)
    place(scene.yellow, -0.055, float(w_loc()[1]), z0, math.pi, settle_steps=120)
    report("seed-place")
    s, ok = judge()
    check("seed strategy: yellow mug set down just OUTSIDE the sill, aligned with "
          "the white mug (the closest top-down place) — no success, score <= 0.13",
          float(y_loc()[0]) < 0.005 and not ok and s <= 0.13)

    # =========================== 7. short near-miss inside the doorway ======================
    torch.manual_seed(51)
    env.reset()
    step(10)
    place(scene.yellow, c.x_in_lo - 0.017, float(w_loc()[1]), z0, math.pi,
          settle_steps=120)
    report("short")
    s, ok = judge()
    check("short near-miss: settled in the doorway but NOT fully past the sill "
          "(local x < x_in_lo) — no success",
          float(y_loc()[0]) < c.x_in_lo - 0.005 and not ok and s <= 0.40)

    # =========================== 8. crowding the white mug ==================================
    torch.manual_seed(61)
    env.reset()
    step(10)
    wl = w_loc()
    place(scene.yellow, float(wl[0]) - 0.0825, float(wl[1]), z0, math.pi,
          settle_steps=120)
    report("crowded")
    s, ok = judge()
    gap = float(w_loc()[0]) - float(y_loc()[0])
    check("crowding: settled fully inside but pressed to ~2.5 mm surface gap from "
          "the white mug (centre spacing < gap_min, readback) — no success",
          gap < c.gap_min and float(y_loc()[0]) > c.x_in_lo and not ok)

    # =========================== 9. off the white mug's lane ================================
    torch.manual_seed(71)
    env.reset()
    step(10)
    wl = w_loc()
    y_off = float(wl[1]) - 0.07 if float(wl[1]) > 0 else float(wl[1]) + 0.07
    place(scene.yellow, goal_x(), y_off, z0, math.pi, settle_steps=120)
    report("off-axis")
    s, ok = judge()
    check("off-axis: settled at goal depth but 7 cm off the white mug's lane "
          "(> y_tol) — no success",
          abs(float(y_loc()[1]) - float(w_loc()[1])) > c.y_tol and not ok)

    # =========================== 10. toppled under the awning ===============================
    torch.manual_seed(81)
    env.reset()
    step(10)
    q_roll = torch.tensor([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0],
                          device=device)  # 90 deg about x: mug lies on its side
    place(scene.yellow, goal_x(), float(w_loc()[1]), c.body_r + 0.002, math.pi,
          settle_steps=150, extra_rot=q_roll)
    report("toppled")
    s, ok = judge()
    check("toppled: mug lying on its side at goal depth under the awning — no "
          "success (upright gate), no entry credit",
          not bool(scene._upright(scene.yellow)[0]) and not ok and s <= 0.13)

    # =========================== 11. white mug disturbed ====================================
    # The don't-disturb gate, and the "move the WHITE mug instead" cheat: displace the
    # white mug 6 cm sideways, then settle the yellow mug PERFECTLY in front of its
    # new position. Placement readback is all true — still not success.
    torch.manual_seed(91)
    env.reset()
    step(10)
    wl = w_loc()
    y_new = float(wl[1]) - 0.06 if float(wl[1]) > 0 else float(wl[1]) + 0.06
    place(scene.white, float(wl[0]), y_new, z0, 0.0, settle_steps=60)
    place(scene.yellow, goal_x(), float(w_loc()[1]), z0, math.pi, settle_steps=150)
    report("white-moved")
    s, ok = judge()
    placed = bool(scene._placed_now()[0])
    intact = bool(scene._white_intact()[0])
    check("white disturbed: white mug displaced 6 cm, yellow mug then settled "
          "perfectly in front of it (placement readback TRUE) — the don't-disturb "
          "gate still rejects success",
          placed and not intact and not ok)

    # =========================== 12. parked on the awning roof ==============================
    torch.manual_seed(101)
    env.reset()
    step(10)
    place(scene.yellow, 0.085, float(w_loc()[1]),
          c.roof_z + c.roof_t + c.body_h / 2 + 0.003, math.pi, settle_steps=120)
    report("roof-park")
    s, ok = judge()
    check("roof parking: yellow mug parked ON TOP of the awning above the goal zone "
          "— no success, no entry credit (score <= 0.13)",
          float((scene.yellow.data.root_pos_w - scene.env_origins)[0, 2]) > 0.12
          and not ok and s <= 0.13)

    # =========================== 13-14. fly-through / teleport guard ========================
    torch.manual_seed(111)
    env.reset()
    step(60)  # build an (irrelevant) resting streak first
    place(scene.yellow, goal_x(), float(w_loc()[1]), z0, math.pi, settle_steps=0)
    placed0 = bool(scene._placed_now()[0])
    s0, ok0 = judge()
    report("fly-through")
    check("fly-through: yellow mug teleported DIRECTLY into the exact goal pose "
          "(zero velocity) judged WITHOUT stepping is NOT success",
          placed0 and not ok0)
    step(12)  # a few real frames: latches fire, streak still far from full
    mid_ok = bool(scene.success()[0])
    ever_success[0] = ever_success[0] or mid_ok
    place(scene.yellow, -0.35, 0.0, z0, math.pi, settle_steps=90)  # yank back out
    report("yank-away")
    s, ok = judge()
    check("latch survives: a few real frames in the goal window then yanked back "
          "outside — success never fired, latched credit persists at the 0.67 cap",
          not mid_ok and not ok and 0.66 <= s <= 0.68)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    fin = (torch.isfinite(scene.alcove.data.root_state_w).all()
           and torch.isfinite(scene.white.data.root_state_w).all()
           and torch.isfinite(scene.yellow.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.mug_alcove")
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
    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)

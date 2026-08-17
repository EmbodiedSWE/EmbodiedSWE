"""Smoke / rubric-REJECTION battery for HangChillerScene (sim_gen task
`put_bottle_in_fridge_i179`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — teleport-to-hover over the tongue, gravity seat,
force-slide into the cold zone — is the acceptance evidence that the rubric ACCEPTS a
correct outcome). Every teleport here is instrumentation that CONSTRUCTS a wrong (or
partial) outcome as a settled state and asserts the rubric REJECTS it; no probe in
this battery ever reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: rail at its written lateral
                            offset (readback vs cabinet pose), bottle + can upright
                            on the ground, score ~0;
  3-4. randomization      — READBACK over 6 seeded resets: cabinet yaw + xy and the
                            rail slot offset vary; bottle + can positions vary and
                            never spawn closer than the separation floor;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed strategy A     — the seed's end state (bottle STANDING inside the fridge):
                            stood upright on the interior floor it falls THROUGH the
                            sparse grate into the drain pan (static geometry assert:
                            gap between rungs > bottle diameter) -> score ~0;
  7.  deep, not hanging   — bottle standing in the drain pan DEEPER than chill_depth:
                            cap well behind the fascia but not suspended -> depth
                            credit is gated on hanging, score ~0, no success;
  8.  seed strategy B     — the seed's carry-drop (release above the target): dropped
                            over the cabinet the bottle settles ON the roof, never
                            inside -> score ~0 (no top access);
  9.  wrong object        — RED can set over the slot on the tongue: it RESTS on the
                            bars (56 mm > the 32 mm slot, static assert), never
                            hangs, never falls through -> score ~0;
  10. topology            — TOP-LOAD attempt with the right object: the bottle set
                            upright over the slot rests on its BODY on the bars
                            (60 mm > 32 mm, static assert) — the slot is end-load
                            only; `hung` never latches -> score ~0;
  11. wrong place         — bottle lying across the grate rungs inside (in the cold
                            zone but on the floor, not the rail) -> score ~0;
  12. near-miss depth     — bottle properly HUNG but only 60 mm behind the fascia
                            (< chill_depth 100 mm) -> hung + partial depth credit
                            (~0.60), NOT success;
  13. hung at the tongue  — bottle properly hung but still OUT on the loading tongue
                            -> hung credit only (~0.30), NOT success;
  14. monotonicity        — the same hung bottle constructed deeper latches strictly
                            more depth credit (30 mm -> 70 mm), still no success
                            short of chill_depth;
  15. rejection audit     — success() was never True at ANY judged point;
  16. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.put_bottle_in_fridge_i179.smoke --headless
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
# RTX recipe: kit mis-decodes some driver versions and silently rejects RTX -> the
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.hang_chiller")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.75, -1.15, 0.95)) + o),
                                tuple(np.array((0.40, 0.00, 0.22)) + o),
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

    def cab_pose() -> tuple[torch.Tensor, float]:
        cp = (scene.cabinet.data.root_pos_w - scene.env_origins)[0]
        q = scene.cabinet.data.root_quat_w[0]
        return cp, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def report(tag: str) -> None:
        cl = scene._in_frame(scene._cap_center_w(), scene.rail)[0]
        d = float(scene._cap_depth()[0])
        s, ok = judge()
        print(f"[smoke] {tag:16s} | cap_rail=({float(cl[0]):+.3f},{float(cl[1]):+.3f},"
              f"{float(cl[2]):+.3f}) depth={d:+.3f} hung_now={bool(scene._hung_now()[0])} "
              f"hung={bool(scene._hung[0])} depth_max={float(scene._depth_max[0]):.3f} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def write_pose(obj, wx: float, wy: float, z: float, quat: tuple,
                   settle_steps: int = 60) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = wx, wy, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += env.iscene.env_origins
        obj.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def to_world_cab(x_l: float, y_l: float) -> tuple[float, float, float]:
        cp, cyaw = cab_pose()
        wx = float(cp[0]) + math.cos(cyaw) * x_l - math.sin(cyaw) * y_l
        wy = float(cp[1]) + math.sin(cyaw) * x_l + math.cos(cyaw) * y_l
        return wx, wy, cyaw

    def hover_drop(obj, x_rail: float, origin_dz: float, settle_steps: int = 120) -> None:
        """Place `obj` upright at rail-frame slot position x_rail, its ORIGIN
        `origin_dz` below/above the rail-top plane, and let gravity do the rest."""
        rel = torch.tensor([x_rail, 0.0, origin_dz], device=device).expand(n, 3)
        tgt = (scene.rail.data.root_pos_w
               + quat_apply(scene.rail.data.root_quat_w, rel) - env.iscene.env_origins)[0]
        write_pose(obj, float(tgt[0]), float(tgt[1]), float(tgt[2]),
                   (1.0, 0.0, 0.0, 0.0), settle_steps)

    hover_dz = 0.008 + c.cap_t / 2 - c.cap_off  # bottle origin z: cap flange 8 mm over bars

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    fin0 = (torch.isfinite(scene.rail.data.root_state_w).all()
            and torch.isfinite(scene.bottle.data.root_state_w).all()
            and torch.isfinite(scene.can.data.root_state_w).all())
    cp, cyaw = cab_pose()
    rp = (scene.rail.data.root_pos_w - scene.env_origins)[0]
    y_off = float(scene._y_off[0])
    rx_exp = float(cp[0]) - math.sin(cyaw) * y_off
    ry_exp = float(cp[1]) + math.cos(cyaw) * y_off
    bz = float((scene.bottle.data.root_pos_w - scene.env_origins)[0, 2])
    still = (float(scene.bottle.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.can.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    check("settle: states finite, rail at its written lateral slot offset (readback vs "
          "cabinet pose), bottle upright on the ground, everything at rest",
          bool(fin0) and abs(float(rp[0]) - rx_exp) < 0.002
          and abs(float(rp[1]) - ry_exp) < 0.002
          and abs(float(rp[2]) - c.rail_top) < 0.002
          and abs(bz - c.body_h / 2) < 0.012 and still)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        cp, cyaw = cab_pose()
        bp = (scene.bottle.data.root_pos_w - scene.env_origins)[0]
        kp = (scene.can.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(cp[0]), float(cp[1]), cyaw, float(scene._y_off[0]),
                      float(bp[0]), float(bp[1]), float(kp[0]), float(kp[1]),
                      float((bp[:2] - kp[:2]).norm())))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (cab_x, cab_y, cab_yaw, rail_y_off, "
          f"bottle_x, bottle_y, can_x, can_y, sep):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: cabinet yaw + xy and the rail slot offset vary across seeded "
          "resets (readback)",
          spread[2] > 0.04 and spread[0] > 0.008 and spread[1] > 0.008
          and spread[3] > 0.015)
    check("randomization: bottle + can positions vary and never spawn within min_sep "
          "of each other (readback)",
          spread[4] > 0.02 and spread[5] > 0.08 and spread[7] > 0.08
          and float(arr[:, 8].min()) >= c.min_sep - 0.005)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy A: STAND it inside ========================
    # The seed's end state is the bottle STANDING inside the fridge. Here the interior
    # floor is a sparse grate whose gaps exceed the bottle diameter (static geometry):
    # stood upright between the rungs it falls THROUGH into the drain pan.
    torch.manual_seed(41)
    env.reset()
    step(10)
    gap_between_rungs = (c.grate_xs[0] - c.grate_xs[1]) - c.rung_s
    wx, wy, _ = to_world_cab(-0.09, 0.0)  # gap centre between rungs at -0.05 / -0.13
    write_pose(scene.bottle, wx, wy, c.grate_top + 0.002 + c.body_h / 2,
               (1.0, 0.0, 0.0, 0.0), settle_steps=150)
    report("seed-standing")
    s, ok = judge()
    bz = float((scene.bottle.data.root_pos_w - scene.env_origins)[0, 2])
    check("seed strategy A: bottle stood upright on the interior floor falls THROUGH "
          "the grate into the drain pan (gap between rungs > bottle diameter, static "
          "assert) — no success, score <= 0.02",
          gap_between_rungs > 2 * c.body_r + 0.005 and bz < c.grate_top - 0.010
          and not bool(scene._hung[0]) and not ok and s <= 0.02)

    # =========================== 7. deep placement WITHOUT suspension =======================
    # Depth credit is gated on hanging: a bottle standing in the drain pan with its cap
    # 170 mm behind the fascia (deeper than chill_depth) still earns nothing.
    torch.manual_seed(51)
    env.reset()
    step(10)
    wx, wy, _ = to_world_cab(-0.17, 0.0)  # gap centre between rungs at -0.13 / -0.21
    write_pose(scene.bottle, wx, wy, c.pan_t + 0.002 + c.body_h / 2,
               (1.0, 0.0, 0.0, 0.0), settle_steps=90)
    report("deep-standing")
    s, ok = judge()
    d = float(scene._cap_depth()[0])
    check("deep, not hanging: bottle standing in the drain pan with its cap deeper "
          "than chill_depth earns nothing (depth credit gated on suspension) — no "
          "success, score <= 0.02",
          d > c.chill_depth and not bool(scene._hung[0]) and not ok and s <= 0.02)

    # =========================== 8. seed strategy B: release above the target ===============
    # The seed's carry-drop, aimed at the cabinet: released above it, the bottle lands
    # ON the roof — the interior is roofed; there is no top access.
    torch.manual_seed(61)
    env.reset()
    step(10)
    wx, wy, cyaw = to_world_cab(-0.11, 0.0)
    lie = (math.cos(cyaw / 2) * math.cos(math.pi / 4),
           -math.sin(cyaw / 2) * math.cos(math.pi / 4),
           math.cos(cyaw / 2) * math.cos(math.pi / 4),
           math.sin(cyaw / 2) * math.cos(math.pi / 4))  # qz(yaw) * qy(90): lying
    write_pose(scene.bottle, wx, wy, c.roof_lo + c.roof_t + 0.05 + c.body_r, lie,
               settle_steps=150)
    report("seed-drop")
    s, ok = judge()
    bz = float((scene.bottle.data.root_pos_w - scene.env_origins)[0, 2])
    check("seed strategy B: bottle released above the cabinet settles ON the roof, "
          "never inside (no top access) — no success, score <= 0.02",
          bz > c.roof_lo and not bool(scene._hung[0]) and not ok and s <= 0.02)

    # =========================== 9. wrong object: can over the slot =========================
    torch.manual_seed(71)
    env.reset()
    step(10)
    hover_can = 0.002 + c.can_h / 2  # can base 2 mm over the bar tops
    hover_drop(scene.can, 0.040, hover_can, settle_steps=120)
    report("wrong-object")
    s, ok = judge()
    kz = float((scene.can.data.root_pos_w - scene.env_origins)[0, 2])
    check("wrong object: the RED can set over the slot RESTS on the bars (can diameter "
          "> slot width, static assert) — it can never hang; no success, score <= 0.02",
          2 * c.can_r > 2 * c.slot_hw + 0.010 and kz > c.rail_top - 0.020
          and not ok and s <= 0.02)

    # =========================== 10. topology: top-load attempt =============================
    # The RIGHT object, the WRONG loading direction: set upright over the slot, the
    # bottle's BODY rests on the bars (body diameter > slot width, static assert) and
    # the cap ends far ABOVE the seated band — the slot only loads from its front end.
    torch.manual_seed(81)
    env.reset()
    step(10)
    hover_drop(scene.bottle, 0.040, 0.002 + c.body_h / 2, settle_steps=120)
    report("top-load")
    s, ok = judge()
    bz = float((scene.bottle.data.root_pos_w - scene.env_origins)[0, 2])
    check("topology: top-load attempt — the bottle set upright over the slot rests on "
          "its BODY on the bars (body diameter > slot width, static assert), `hung` "
          "never latches — no success, score <= 0.02",
          2 * c.body_r > 2 * c.slot_hw + 0.010 and bz > c.rail_top - 0.020
          and not bool(scene._hung[0]) and not ok and s <= 0.02)

    # =========================== 11. wrong place: lying across the grate ====================
    torch.manual_seed(91)
    env.reset()
    step(10)
    wx, wy, cyaw = to_world_cab(-0.085, 0.0)
    lie_back = (math.cos(cyaw / 2) * math.cos(math.pi / 4),
                math.sin(cyaw / 2) * math.cos(math.pi / 4),
                -math.cos(cyaw / 2) * math.cos(math.pi / 4),
                math.sin(cyaw / 2) * math.cos(math.pi / 4))  # qz(yaw) * qy(-90): cap to back
    write_pose(scene.bottle, wx, wy, c.grate_top + c.body_r + 0.002, lie_back,
               settle_steps=120)
    report("wrong-place")
    s, ok = judge()
    bz = float((scene.bottle.data.root_pos_w - scene.env_origins)[0, 2])
    check("wrong place: bottle lying across the grate rungs inside the cold zone (on "
          "the floor, not the rail) — no success, score <= 0.02",
          bz < 0.20 and not bool(scene._hung[0]) and not ok and s <= 0.02)

    # =========================== 12. near-miss: hung, 60 mm short ===========================
    torch.manual_seed(101)
    env.reset()
    step(10)
    hover_drop(scene.bottle, -0.06, hover_dz, settle_steps=120)
    report("near-miss")
    s_nm, ok = judge()
    check("near-miss: bottle properly HUNG but only ~60 mm behind the fascia "
          "(< chill_depth) — hung + partial depth credit, NOT success, 0.45 <= score "
          "<= 0.70",
          bool(scene._hung_now()[0]) and not ok and 0.45 <= s_nm <= 0.70)

    # =========================== 13. hung at the tongue =====================================
    torch.manual_seed(111)
    env.reset()
    step(10)
    hover_drop(scene.bottle, 0.020, hover_dz, settle_steps=120)
    report("tongue-hung")
    s_t, ok = judge()
    check("incomplete: bottle properly hung but still OUT on the loading tongue — "
          "hung credit only (~0.30), NOT success",
          bool(scene._hung_now()[0]) and not ok and 0.25 <= s_t <= 0.35)

    # =========================== 14. monotonicity of depth credit ===========================
    # Same episode: reconstruct the hang deeper (never at/past chill_depth — no prefix
    # of this probe satisfies the goal). The latched depth credit must strictly grow.
    hover_drop(scene.bottle, -0.030, hover_dz, settle_steps=100)
    report("hang-30mm")
    s_a, ok_a = judge()
    hover_drop(scene.bottle, -0.070, hover_dz, settle_steps=100)
    report("hang-70mm")
    s_b, ok_b = judge()
    check("monotonicity: the same hung bottle constructed deeper latches strictly more "
          f"depth credit ({s_a:.3f} < {s_b:.3f}), still no success short of chill_depth",
          s_t < s_a < s_b < 1.0 and not ok_a and not ok_b)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.cabinet.data.root_state_w).all()
           and torch.isfinite(scene.rail.data.root_state_w).all()
           and torch.isfinite(scene.bottle.data.root_state_w).all()
           and torch.isfinite(scene.can.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.hang_chiller")
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
    main()

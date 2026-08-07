"""Smoke / rubric-REJECTION battery for ChillRackScene (sim_gen task
`put_bottle_in_fridge_i311`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — teleport-to-hover, gravity bed-down, force-ride to
seat — is the acceptance evidence that the rubric ACCEPTS a correct outcome). Every
teleport here is instrumentation that CONSTRUCTS a wrong (or partial) outcome as a
settled state and asserts the rubric REJECTS it; no probe in this battery ever reaches
success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: rack drawn out at its written
                            travel, bottle + can upright on the ground, score ~0;
  3-4. randomization      — READBACK over 6 seeded resets: locker yaw + xy and rack
                            draw-out travel vary; bottle + can positions vary
                            (including band swaps) and never spawn close;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed strategy A     — the seed's placement (bottle UPRIGHT "in the fridge"):
                            stood upright on the drawn-out rack's cradle it counts
                            NOTHING (the trough demands lying alignment), and upright
                            it can never pass the 105 mm mouth (static geometry
                            assert: bed top + bottle height >> mouth);
  7.  seed strategy B     — the seed's carry-drop (release above the target): dropped
                            over the locker the bottle settles ON the roof, never
                            inside -> score ~0 (the interior is roofed — no door to
                            open, no top access);
  8.  wrong object        — RED can laid in the trough + rack seated inside, bottle
                            left outside -> no success, score ~0;
  9.  near-miss seat      — bottle in the trough but rack settled 45 mm short of the
                            seat tolerance (12 mm) -> NOT success, score <= 0.85;
  10. wrong place         — bottle lying on the interior floor plate (inside the
                            locker but OFF the rack), rack drawn out -> no success,
                            score ~0;
  11. incomplete          — bottle laid in the trough but rack never pushed -> laid
                            credit only (~0.30), NOT success;
  12. monotonicity        — the same loaded rack constructed at deeper travels latches
                            strictly more ride credit (0.10 -> 0.03), still no
                            success outside the seat tolerance;
  13. ride gating         — an EMPTY rack seated inside earns nothing (ride credit is
                            gated on the bottle being aboard), no success;
  14. rejection audit     — success() was never True at ANY judged point;
  15. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.put_bottle_in_fridge_i311.smoke --headless
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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.chill_rack")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.55, -1.30, 1.00)) + o),
                                tuple(np.array((0.30, 0.00, 0.08)) + o),
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

    def locker_pose() -> tuple[torch.Tensor, float]:
        lp = (scene.locker.data.root_pos_w - scene.env_origins)[0]
        q = scene.locker.data.root_quat_w[0]
        return lp, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def report(tag: str) -> None:
        t = float(scene._travel()[0])
        bl = scene._bottle_in_rack()[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | travel={t:+.3f} bottle_rack=({float(bl[0]):+.3f},"
              f"{float(bl[1]):+.3f},{float(bl[2]):.3f}) in={bool(scene._in_trough_now()[0])} "
              f"laid={bool(scene._laid[0])} ride={float(scene._ride_max[0]):.3f} "
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

    def to_world(x_l: float, y_l: float) -> tuple[float, float, float]:
        lp, lyaw = locker_pose()
        wx = float(lp[0]) + math.cos(lyaw) * x_l - math.sin(lyaw) * y_l
        wy = float(lp[1]) + math.sin(lyaw) * x_l + math.cos(lyaw) * y_l
        return wx, wy, lyaw

    def lying_quat(yaw: float) -> tuple:
        cy, sy, c45 = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(math.pi / 4)
        return (cy * c45, -sy * c45, cy * c45, sy * c45)

    def place_rack(t: float, settle_steps: int = 30) -> None:
        """Kinematic probe placement of the rack at travel t in its channel
        (instrumentation, not a solution) + REAL physics steps before judging."""
        wx, wy, lyaw = to_world(t, 0.0)
        write_pose(scene.rack, wx, wy, c.rack_z0,
                   (math.cos(lyaw / 2), 0.0, 0.0, math.sin(lyaw / 2)), settle_steps)

    def place_bottle_trough(settle_steps: int = 60) -> None:
        """Bottle lying along the trough, 15 mm above the CURRENT rack cradle."""
        from isaaclab.utils.math import quat_apply

        rq = scene.rack.data.root_quat_w
        ryaw = 2.0 * math.atan2(float(rq[0, 3]), float(rq[0, 0]))
        rel = torch.tensor([-0.005, 0.0, c.cradle_z + 0.015], device=device).expand(n, 3)
        tgt = (scene.rack.data.root_pos_w + quat_apply(rq, rel) - env.iscene.env_origins)[0]
        write_pose(scene.bottle, float(tgt[0]), float(tgt[1]), float(tgt[2]),
                   lying_quat(ryaw), settle_steps)

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    fin0 = (torch.isfinite(scene.rack.data.root_state_w).all()
            and torch.isfinite(scene.bottle.data.root_state_w).all()
            and torch.isfinite(scene.can.data.root_state_w).all())
    t_r = float(scene._travel()[0])
    t0_written = float(scene._t0[0])
    bz = float((scene.bottle.data.root_pos_w - scene.env_origins)[0, 2])
    still = (float(scene.rack.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.bottle.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    check("settle: states finite, rack drawn out at its written travel (readback), "
          "bottle upright on the ground, everything at rest",
          bool(fin0) and abs(t_r - t0_written) < 0.012
          and abs(bz - c.body_h / 2) < 0.012 and still)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        lp, lyaw = locker_pose()
        t_r = float(scene._travel()[0])
        bp = (scene.bottle.data.root_pos_w - scene.env_origins)[0]
        cp = (scene.can.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(lp[0]), float(lp[1]), lyaw, t_r, float(bp[0]), float(bp[1]),
                      float(cp[0]), float(cp[1]), float((bp[:2] - cp[:2]).norm())))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (locker_x, locker_y, locker_yaw, travel, "
          f"bottle_x, bottle_y, can_x, can_y, sep):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: locker yaw + xy and rack draw-out travel vary across seeded "
          "resets (readback)",
          spread[2] > 0.04 and spread[0] > 0.008 and spread[1] > 0.008
          and spread[3] > 0.008)
    check("randomization: bottle + can positions vary (incl. band swaps) and never "
          "spawn within 30 cm of each other (readback)",
          spread[4] > 0.02 and spread[5] > 0.08 and spread[7] > 0.08
          and float(arr[:, 8].min()) >= 0.30)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy A: UPRIGHT placement ======================
    # The seed places the bottle UPRIGHT inside the fridge. Stood upright on the rack's
    # cradle (the only serviceable "inside-bound" surface) it counts NOTHING — the
    # trough demands lying alignment — and upright it can never pass the mouth: bed top
    # + standing height is more than double the 105 mm opening (static geometry).
    torch.manual_seed(41)
    env.reset()
    step(10)
    from isaaclab.utils.math import quat_apply as _qa

    rq = scene.rack.data.root_quat_w
    rel = torch.tensor([c.trough_cx, 0.0, c.bed_t / 2 + c.body_h / 2 + 0.010],
                       device=device).expand(n, 3)
    tgt = (scene.rack.data.root_pos_w + _qa(rq, rel) - env.iscene.env_origins)[0]
    write_pose(scene.bottle, float(tgt[0]), float(tgt[1]), float(tgt[2]),
               (1.0, 0.0, 0.0, 0.0), settle_steps=120)
    report("seed-upright")
    s, ok = judge()
    standing_top = c.plate_t + c.bed_t + c.body_h + c.neck_h
    check("seed strategy A: bottle stood UPRIGHT on the rack cradle earns nothing "
          "(lying alignment required), no success, score <= 0.02; and an upright "
          "bottle can never pass the mouth (bed top + height >> mouth header)",
          not bool(scene._laid[0]) and not ok and s <= 0.02
          and standing_top > c.roof_z + 0.08)

    # =========================== 7. seed strategy B: release above the target ===============
    # The seed's carry-drop, aimed at the locker: released above it, the bottle lands
    # ON the roof — the interior is roofed and door-less; there is no top access.
    torch.manual_seed(51)
    env.reset()
    step(10)
    wx, wy, lyaw = to_world(0.0, 0.0)
    write_pose(scene.bottle, wx, wy, c.roof_z + c.roof_t + 0.05 + c.body_r,
               lying_quat(lyaw), settle_steps=150)
    report("seed-drop")
    s, ok = judge()
    bl = scene._local(scene.bottle)[0]
    check("seed strategy B: bottle released above the locker settles ON the roof, "
          "never inside (no top access), no success, score <= 0.02",
          float(bl[2]) > c.inside_z_max and not ok and s <= 0.02)

    # =========================== 8. wrong object: can racked ================================
    torch.manual_seed(61)
    env.reset()
    step(10)
    place_rack(0.006, settle_steps=20)
    rq = scene.rack.data.root_quat_w
    ryaw = 2.0 * math.atan2(float(rq[0, 3]), float(rq[0, 0]))
    rel = torch.tensor([0.0, 0.0, c.bed_t / 2 + c.can_r + 0.010], device=device).expand(n, 3)
    tgt = (scene.rack.data.root_pos_w + _qa(rq, rel) - env.iscene.env_origins)[0]
    write_pose(scene.can, float(tgt[0]), float(tgt[1]), float(tgt[2]),
               lying_quat(ryaw), settle_steps=90)
    report("wrong-object")
    s, ok = judge()
    can_loc = scene._local(scene.can)[0]
    check("wrong object: RED can laid in the trough and racked inside, bottle left "
          "outside — no success, score <= 0.02",
          float(can_loc[0]) < c.inside_x_max and not ok and s <= 0.02)

    # =========================== 9. near-miss: rack 45 mm short of seat =====================
    torch.manual_seed(71)
    env.reset()
    step(10)
    place_rack(0.045, settle_steps=20)
    place_bottle_trough(settle_steps=80)
    report("near-miss-seat")
    s_nm, ok = judge()
    t_r = float(scene._travel()[0])
    check("near-miss: bottle in the trough but rack settled ~45 mm short of the seat "
          "tolerance (12 mm) — NOT success, score <= 0.85",
          bool(scene._in_trough_now()[0]) and t_r > c.seat_tol + 0.010 and not ok
          and s_nm <= 0.85)

    # =========================== 10. wrong place: inside but OFF the rack ===================
    torch.manual_seed(81)
    env.reset()
    step(10)
    wx, wy, lyaw = to_world(-0.02, 0.0)
    write_pose(scene.bottle, wx, wy, c.plate_t + c.body_r + 0.005, lying_quat(lyaw),
               settle_steps=90)
    report("wrong-place")
    s, ok = judge()
    bl = scene._local(scene.bottle)[0]
    check("wrong place: bottle lying on the interior floor plate (inside the locker "
          "but OFF the rack) — no success, score <= 0.02",
          float(bl[0]) < c.inside_x_max and float(bl[2]) < c.inside_z_max
          and not bool(scene._laid[0]) and not ok and s <= 0.02)

    # =========================== 11. incomplete: laid but never pushed ======================
    torch.manual_seed(91)
    env.reset()
    step(10)
    place_bottle_trough(settle_steps=80)
    report("laid-only")
    s_laid, ok = judge()
    check("incomplete: bottle laid in the trough but rack never pushed — laid credit "
          "only, NOT success, score <= 0.35",
          bool(scene._laid[0]) and not ok and 0.20 <= s_laid <= 0.35)

    # =========================== 12. monotonicity of ride credit ============================
    place_rack(0.10, settle_steps=10)
    place_bottle_trough(settle_steps=50)
    report("ride-half")
    s_half, _ok = judge()
    r_half = float(scene._ride_max[0])
    place_rack(0.030, settle_steps=10)
    place_bottle_trough(settle_steps=50)
    report("ride-deep")
    s_deep, ok = judge()
    r_deep = float(scene._ride_max[0])
    check("monotonicity: the loaded rack constructed deeper in the channel latches "
          f"strictly more ride credit ({r_half:.3f} < {r_deep:.3f}), higher score, "
          "and still no success outside the seat tolerance",
          r_half + 0.10 < r_deep and s_half < s_deep and not ok)

    # =========================== 13. ride gating: empty rack seated =========================
    torch.manual_seed(101)
    env.reset()
    step(10)
    place_rack(0.006, settle_steps=60)
    report("empty-seated")
    s, ok = judge()
    check("ride gating: an EMPTY rack seated inside earns nothing (ride credit gated "
          "on the bottle being aboard), no success, score <= 0.02",
          s <= 0.02 and not ok)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.rack.data.root_state_w).all()
           and torch.isfinite(scene.locker.data.root_state_w).all()
           and torch.isfinite(scene.bottle.data.root_state_w).all()
           and torch.isfinite(scene.can.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.chill_rack")
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

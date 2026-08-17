"""Smoke / rubric-REJECTION battery for FoldRackScene (sim_gen task
`put_plate_in_colored_dish_rack_i384`) — NullRobot, constructed probe states, RECORDED.

This is NOT a solution (solve.py — hinge-torque deploy servo + force lift + free-air
hover + gravity insertion — is the acceptance evidence that the rubric ACCEPTS a
correct run; it passes on forge seeds 0 and 1). Every teleport here is instrumentation
that CONSTRUCTS a wrong (or partial) outcome as a judged state and asserts the rubric
REJECTS it — plus an order-forcing reality probe proving the seed's strategy (carry
the plate to the color slot and lower it in) physically fails while the comb is
folded: there is no slot, the drop lands on the closed lid. No probe in this battery
ever reaches success(), and a final audit check asserts exactly that.

   1. settle             — reset settles finite; readback: comb FOLDED (hinge angle),
                           plate FLAT on the stand, everything settled;
   2. fresh score        — score ~0 at reset, no success;
   3. rack randomization — READBACK over 8 seeded resets: chassis yaw varies widely
                           (both signs), live quat agrees with the cached sample,
                           xy jitter is real;
   4. comb/stand/plate   — comb start angle varies (cache spread + live agreement),
                           stand xy and plate on-stand xy + free yaw jitter are real;
   5. null policy        — 240 idle steps -> score ~0, no success, comb still folded,
                           plate still flat on its stand;
   6. order forced       — the SEED's strategy: plate dropped ON EDGE exactly where
                           the blue slot WILL be, while the comb is FOLDED — it lands
                           on the closed lid and is denied (no seat, no seat band,
                           score ~0, comb still folded): no slot exists to aim at;
   7. deployed gate      — comb held just BELOW deploy_min (85 deg, geometrically
                           erect) + plate written into a PERFECT blue-gap seat pose:
                           the geometric seat bands hold, yet seated() is False
                           because deployed() is False — the gate is load-bearing;
   8. bistable closed    — comb released at 60 deg (below over-center) falls back
                           CLOSED; no deploy credit;
   9. bistable open      — comb released at 85 deg (past over-center) gravity-parks
                           OPEN at the stop -> deploy latch fires -> EXACTLY +0.35,
                           not success (the plate is still on its stand);
  10. yellow gap partial — plate gravity-dropped into the YELLOW gap seats and
                           latches the seat credit -> score capped at 0.55, NOT
                           success (the goal is the BLUE gap specifically);
  11. latched credit     — removing the seated plate to the floor keeps the latched
                           0.55 while the live seat drops — score never decreases;
  12. settle gate        — plate written into a perfect BLUE-gap seat with the comb
                           open, judged immediately: seated LIVE but NOT settled ->
                           NOT success; dismantled before the stillness latch can
                           complete (audit safety);
  13. rejection audit    — success() was never True at ANY judged point;
  14. final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.put_plate_in_colored_dish_rack_i384.smoke \
    --headless
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
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

BLUE_GAP_SIGN = scene_mod.BLUE_GAP_SIGN

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)",
                                             flush=True), os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    from isaaclab.utils.math import quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.fold_rack")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    blue_cx = BLUE_GAP_SIGN * c.slot_dx
    yellow_cx = -BLUE_GAP_SIGN * c.slot_dx

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.95, -0.45, 0.60)) + o),
                                tuple(np.array((0.00, 0.15, 0.08)) + o),
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

    def plate_loc() -> torch.Tensor:
        return scene._local(scene.plate, scene.chassis)[0]

    def axis_z() -> float:
        ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
        return float(quat_apply(scene.plate.data.root_quat_w, ez)[0, 2].abs())

    def comb_deg() -> float:
        return math.degrees(float(scene.comb_angle()[0]))

    def yaw_of(q: torch.Tensor) -> float:
        w, x, y, z = (float(v) for v in q)
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def report(tag: str) -> None:
        s, ok = judge()
        lp = plate_loc()
        print(f"[smoke] {tag:16s} | comb={comb_deg():+6.1f}deg "
              f"plate_ch=({float(lp[0]):+.3f},{float(lp[1]):+.3f},"
              f"{float(lp[2]):+.3f}) axis_z={axis_z():.2f} | "
              f"dep={bool(scene.deployed()[0])} fold={bool(scene.folded()[0])} "
              f"seatB={bool(scene.seated('blue')[0])} "
              f"seatY={bool(scene.seated('yellow')[0])} "
              f"stl={bool(scene.settled()[0])} | "
              f"latch d={float(scene.deploy_latch[0]):.0f}/s="
              f"{float(scene.seat_latch[0]):.0f} | score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def write_plate(pos_env, quat, settle_steps: int = 60) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL
        physics steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.as_tensor(pos_env, device=device, dtype=torch.float32) \
            + scene.env_origins[0]
        st[:, 3:7] = torch.as_tensor(quat, device=device, dtype=torch.float32)
        scene.plate.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def chassis_pose() -> tuple[torch.Tensor, torch.Tensor]:
        return scene.chassis.data.root_pos_w.clone(), \
            scene.chassis.data.root_quat_w.clone()

    def write_comb(theta_deg: float, settle_steps: int = 0) -> None:
        """Write the comb at hinge angle theta consistently with the LIVE chassis
        pose (the whole-linkage rule; velocities zeroed)."""
        ch_pos, ch_quat = chassis_pose()
        th = torch.full((n,), math.radians(theta_deg), device=device)
        pos, quat = scene.comb_pose_for(ch_pos, ch_quat, th)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3:7] = quat
        scene.comb.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def chassis_point(local_xyz) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos_env, seat_quat) for a chassis-local point, plate standing on edge."""
        ch_pos, ch_quat = chassis_pose()
        local = torch.tensor(local_xyz, device=device).expand(n, 3)
        pos = (ch_pos + quat_apply(ch_quat, local))[0] - scene.env_origins[0]
        return pos, scene._seat_quat(ch_quat)[0]

    def hover_drop(cx: float, settle_steps: int = 300, z: float = 0.245) -> None:
        """Free-air hover over the gap + pure GRAVITY drop (no force) — the same
        insertion mechanism the verified solve uses."""
        pos, q = chassis_point((cx, 0.012, z))
        write_plate(pos, q, settle_steps=settle_steps)

    def park_on_floor(xy, settle_steps: int = 60) -> None:
        """Plate laid FLAT on the open floor, far from rack and stand."""
        write_plate((xy[0], xy[1], c.plate_t / 2 + 0.004),
                    (1.0, 0.0, 0.0, 0.0), settle_steps=settle_steps)

    # =========================== 1-2. settle / fresh score ==================================
    env.reset(seed=11)
    step(90)
    report("reset")
    bodies = (scene.chassis, scene.comb, scene.stand, scene.plate)
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    sp = (scene.plate.data.root_pos_w - scene.stand.data.root_pos_w)[0]
    on_stand = math.hypot(float(sp[0]), float(sp[1])) <= c.stand_r \
        and abs(float(sp[2]) - (c.stand_h / 2 + c.plate_t / 2)) <= 0.01
    check("settle: all states finite, comb FOLDED by hinge readback "
          f"(angle={comb_deg():+.1f} deg <= {c.fold_max_deg:.0f}), plate FLAT on the "
          f"stand (axis_z={axis_z():.2f}, offset "
          f"{math.hypot(float(sp[0]), float(sp[1])) * 1000:.0f} mm), settled",
          fin and bool(scene.folded()[0]) and axis_z() > 0.9 and on_stand
          and bool(scene.settled()[0]))
    s, ok = judge()
    check("fresh: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    rack_reads, comb_reads, stand_reads, plate_reads = [], [], [], []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(2)
        th_live = comb_deg()
        th_cache = float(scene.comb_th0[0])
        step(18)
        yaw_live = yaw_of(scene.chassis.data.root_quat_w[0])
        yaw_cache = float(scene.rack_yaw0[0])
        d = math.degrees(math.atan2(math.sin(yaw_live - yaw_cache),
                                    math.cos(yaw_live - yaw_cache)))
        rp = (scene.chassis.data.root_pos_w - scene.env_origins)[0]
        rack_reads.append((math.degrees(yaw_live), d, float(rp[0]), float(rp[1])))
        comb_reads.append((th_cache, th_live - th_cache))
        stp = (scene.stand.data.root_pos_w - scene.env_origins)[0]
        stand_reads.append((float(stp[0]), float(stp[1])))
        pp = (scene.plate.data.root_pos_w - scene.stand.data.root_pos_w)[0]
        pyaw = yaw_of(scene.plate.data.root_quat_w[0])
        plate_reads.append((float(pp[0]), float(pp[1]), math.degrees(pyaw)))
        print(f"[smoke] seed {sd}: rack yaw={math.degrees(yaw_live):+7.1f} "
              f"(cache delta {d:+.2f}) pos=({float(rp[0]):+.3f},{float(rp[1]):+.3f})"
              f" | comb th0={th_cache:.2f} (live delta {th_live - th_cache:+.2f}) | "
              f"stand=({float(stp[0]):+.3f},{float(stp[1]):+.3f}) | plate jit "
              f"({float(pp[0]) * 1000:+.1f},{float(pp[1]) * 1000:+.1f})mm "
              f"yaw={math.degrees(pyaw):+.0f}", flush=True)
    yaws = [r[0] for r in rack_reads]
    agree = max(abs(r[1]) for r in rack_reads)
    rxs = [r[2] for r in rack_reads]
    rys = [r[3] for r in rack_reads]
    check("randomization: chassis yaw varies across 8 seeded resets (spread "
          f"{max(yaws) - min(yaws):.0f} deg, both signs: "
          f"{len({y > 0 for y in yaws}) == 2}), live quat agrees with the cached "
          f"sample (max delta {agree:.2f} deg), xy jitter real (x spread "
          f"{(max(rxs) - min(rxs)) * 1000:.0f} mm, y spread "
          f"{(max(rys) - min(rys)) * 1000:.0f} mm)",
          (max(yaws) - min(yaws)) > 90.0 and len({y > 0 for y in yaws}) == 2
          and agree < 1.0 and (max(rxs) - min(rxs)) > 0.008
          and (max(rys) - min(rys)) > 0.008)
    th0s = [r[0] for r in comb_reads]
    th_d = max(abs(r[1]) for r in comb_reads)
    sxs = [r[0] for r in stand_reads]
    sys_ = [r[1] for r in stand_reads]
    pxs = [r[0] for r in plate_reads]
    pys = [r[1] for r in plate_reads]
    pws = [r[2] for r in plate_reads]
    lo, hi = c.comb_start_deg
    check("randomization: comb start angle varies inside its range (cache spread "
          f"{max(th0s) - min(th0s):.1f} deg in [{lo:.0f},{hi:.0f}], live agreement "
          f"{th_d:.2f} deg); stand xy real (x {(max(sxs) - min(sxs)) * 1000:.0f} mm, "
          f"y {(max(sys_) - min(sys_)) * 1000:.0f} mm); plate on-stand jitter real "
          f"(x {(max(pxs) - min(pxs)) * 1000:.1f} mm, y "
          f"{(max(pys) - min(pys)) * 1000:.1f} mm, yaw spread "
          f"{max(pws) - min(pws):.0f} deg)",
          (max(th0s) - min(th0s)) > 1.2
          and all(lo - 0.2 <= t <= hi + 0.2 for t in th0s) and th_d < 3.0
          and (max(sxs) - min(sxs)) > 0.006 and (max(sys_) - min(sys_)) > 0.006
          and (max(pxs) - min(pxs)) > 0.004 and (max(pys) - min(pys)) > 0.004
          and (max(pws) - min(pws)) > 90.0)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    sp = (scene.plate.data.root_pos_w - scene.stand.data.root_pos_w)[0]
    check("null policy: score ~0 and no success after 240 idle steps — the comb "
          f"stays gravity-held CLOSED (angle={comb_deg():+.1f} deg) and the plate "
          "stays flat on its stand",
          s <= 0.02 and not ok and bool(scene.folded()[0])
          and math.hypot(float(sp[0]), float(sp[1])) <= c.stand_r and axis_z() > 0.9)

    # =========================== 6. the seed's strategy is denied ===========================
    env.reset(seed=41)
    step(30)
    # The seed-analog shortcut: carry the plate to the blue slot and lower it in —
    # but the comb is FOLDED, so no slot exists anywhere. The on-edge hover is
    # exactly the verified solve's insertion pose; the only difference is that the
    # comb has not been erected. The drop lands on the closed lid and is denied.
    hover_drop(blue_cx, settle_steps=360)
    report("folded-drop")
    s, ok = judge()
    lp = plate_loc()
    fell = float(lp[2]) < 0.16  # left the hover: the drop really happened
    check("order forced by physics: the seed's strategy (drop the plate exactly "
          "where the blue slot WILL be) while the comb is FOLDED lands on the "
          f"closed lid and is DENIED (plate_ch z={float(lp[2]):+.3f}, "
          f"axis_z={axis_z():.2f}): no seat, no geometric seat band, score<=0.02 "
          f"(got {s:.2f}), comb still folded ({comb_deg():+.1f} deg)",
          fell and not bool(scene.seated("blue")[0])
          and not bool(scene.seated("yellow")[0])
          and not bool(scene.seat_bands(blue_cx)[0])
          and bool(scene.folded()[0]) and s <= 0.02 and not ok)

    # =========================== 7. the deployed() gate is load-bearing =====================
    env.reset(seed=51)
    step(30)
    # Comb held just BELOW the deploy threshold (85 < deploy_min = 92: geometrically
    # erect, gap open) + plate written into a PERFECT in-band blue seat pose (y
    # pushed forward, clear of the leaned fins). One real step, judged immediately:
    # the geometric bands hold, seated() must still be False — deployed() gates it.
    write_comb(85.0, settle_steps=0)
    pos, q = chassis_point((blue_cx, 0.030, c.plate_r + 0.004))
    write_plate(pos, q, settle_steps=0)
    step(1)
    report("gate-probe")
    s, ok = judge()
    bands = bool(scene.seat_bands(blue_cx)[0])
    check("deployed gate: with the comb at 85 deg (< deploy_min "
          f"{c.deploy_min_deg:.0f}) and the plate in a perfect blue seat pose, the "
          f"geometric seat bands HOLD (bands={bands}) yet seated() is False because "
          f"deployed() is False (comb={comb_deg():+.1f} deg) -> no credit, no "
          "success",
          bands and not bool(scene.deployed()[0])
          and not bool(scene.seated("blue")[0]) and s <= 0.02 and not ok)
    # Dismantle NOW: 85 deg is past over-center — gravity would erect it and the
    # seated plate would start earning real credit (audit safety).
    park_on_floor((0.65, -0.65), settle_steps=0)
    env.reset(seed=61)
    step(30)

    # =========================== 8-9. over-center bistability ===============================
    # Released BELOW over-center: gravity pulls the comb back CLOSED.
    write_comb(60.0, settle_steps=300)
    report("release-60")
    s, ok = judge()
    check("bistable closed side: the comb released at 60 deg (below over-center "
          f"~{c.overcenter_deg:.0f}) falls back CLOSED (angle={comb_deg():+.1f} deg "
          f"<= {c.fold_max_deg:.0f}) — no deploy credit, score<=0.02 (got {s:.2f})",
          bool(scene.folded()[0]) and float(scene.deploy_latch[0]) < 0.5
          and s <= 0.02 and not ok)
    # Released PAST over-center: gravity carries it open and parks it at the stop.
    write_comb(85.0, settle_steps=300)
    for _ in range(12):
        if float(scene.deploy_latch[0]) > 0.5:
            break
        step(15)
    report("release-85")
    s_dep, ok = judge()
    check("bistable open side: the comb released at 85 deg (past over-center) "
          f"gravity-parks OPEN (angle={comb_deg():+.1f} deg >= "
          f"{c.deploy_min_deg:.0f}) and the deploy latch fires EXACTLY the +0.35 "
          f"credit (got {s_dep:.3f}) — not success (the plate is still on its stand)",
          bool(scene.deployed()[0]) and 0.33 <= s_dep <= 0.36 and not ok)

    # ================= 10-11. yellow gap partial + latched credit ===========================
    hover_drop(yellow_cx, settle_steps=300)
    for _ in range(12):
        if float(scene.seat_latch[0]) > 0.5:
            break
        step(15)
    report("yellow-seat")
    s_half, ok = judge()
    check("yellow gap is not the goal: the plate gravity-seated in the YELLOW gap "
          f"latches the seat credit and the score caps at 0.55 (got {s_half:.3f}), "
          "NOT success (the task names the BLUE gap)",
          bool(scene.seated("yellow")[0]) and 0.53 <= s_half <= 0.5601 and not ok)
    park_on_floor((0.65, -0.65))
    report("latch-remove")
    s_after, ok = judge()
    check("latched credit: removing the seated plate to the floor keeps the latched "
          f"score ({s_half:.3f} -> {s_after:.3f}) while the live seat drops",
          abs(s_after - s_half) < 1e-3 and not bool(scene.seated("yellow")[0])
          and not ok)

    # =========================== 12. settle gate ============================================
    # Plate written into a perfect BLUE seat with the comb open, judged immediately:
    # seated LIVE, but stillness has not persisted -> NOT success.
    pos, q = chassis_point((blue_cx, 0.012, c.plate_r + 0.004))
    write_plate(pos, q, settle_steps=0)
    step(3)
    report("settle-gate")
    s, ok = judge()
    seated_live = bool(scene.seated("blue")[0])
    check("settle gate: plate seated LIVE in the blue gap immediately after "
          f"placement (seated={seated_live}) but stillness has not persisted "
          f"(settled={bool(scene.settled()[0])}) -> NOT success at the judged "
          "moment",
          seated_live and not bool(scene.settled()[0]) and not ok)
    # Dismantle BEFORE the stillness latch can complete (audit safety).
    park_on_floor((0.65, 0.65), settle_steps=30)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.fold_rack")
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

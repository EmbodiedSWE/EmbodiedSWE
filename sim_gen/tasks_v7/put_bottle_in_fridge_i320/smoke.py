"""Smoke / rubric-REJECTION battery for ChestSwapScene (sim_gen task
`put_bottle_in_fridge_i320`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — force-slide the lid, force-hoist the can, two free-
air transport teleports with gravity drops, force-slide shut — is the acceptance
evidence that the rubric ACCEPTS a correct outcome). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled state and
asserts the rubric REJECTS it; no probe in this battery ever reaches success(), and a
final audit check asserts exactly that.

  1-2.  settle/no-NaN     — reset layout settles finite: lid closed (u~0 readback), can
                            standing in the well, bottle upright on the ground, score ~0;
  3-4.  randomization     — READBACK over 6 seeded resets: chest xy + yaw, bin xy + yaw,
                            bottle xy and the can's in-well jitter all vary; the movable
                            pieces never spawn near each other;
  5.   null policy        — 240 idle steps -> score ~0, no success;
  6.   seed strategy A    — the seed's placement ("stand the bottle inside"): stood
                            upright on the DECK (lid open) it earns only the open credit
                            — the seat band demands the bottle's base INSIDE the well
                            (deck base height 0.080 vs band max 0.030, static assert);
  7.   seed strategy B    — the seed's carry-drop with no lid concept: released above
                            the CLOSED chest the bottle lands on the lid and never gets
                            inside -> score ~0 (the closed lid denies all access);
  8.   occupied well      — the bottle dropped straight into the OCCUPIED well cannot
                            seat (it lands on the can and topples off) + static
                            exclusion math: max in-well centre separation < sum of
                            radii, so evict-before-insert is geometric, not decreed;
  9.   no-open shortcut   — can teleported out of the (never-opened) chest into the
                            bin: it physically sits in the bin (readback) but the
                            order-aware chain latches NOTHING -> score ~0;
  10.  reversed swap      — the BOTTLE dropped into the discard bin earns nothing;
  11.  wrong disposal     — lid opened, can evicted but parked on the GROUND beside the
                            bin, bottle seated in the well -> partial credit only
                            (~0.55), NOT success (can not in the bin);
  12.  clear-path close   — from that state, the lid physically force-slides SHUT over
                            the seated bottle (actuator-moved readback: u 0.24 -> <=
                            0.010, lid_closed_now True) and STILL no success — the
                            unbinned can is the only missing clause (non-vacuous ref);
  13.  near-miss lid      — can then binned, lid re-parked 30 mm short of closed ->
                            all-but-lid state scores 0.75, NOT success;
  14.  lid-path knockover — a bottle stood on the deck pokes above the wall top; the
                            closing lid knocks it over (real force drive, lid moves) ->
                            bottle NOT seated, still no success;
  15.  rejection audit    — success() was never True at ANY judged point;
  16.  final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.put_bottle_in_fridge_i320.smoke --headless
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.chest_swap")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.60, -1.15, 1.05)) + o),
                                tuple(np.array((0.30, 0.00, 0.10)) + o),
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

    def chest_pose() -> tuple[torch.Tensor, float]:
        cp = (scene.chest.data.root_pos_w - scene.env_origins)[0]
        q = scene.chest.data.root_quat_w[0]
        return cp, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def report(tag: str) -> None:
        u = float(scene._lid_u()[0])
        bl = scene._chest_local(scene.bottle)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | lid_u={u:+.3f} bottle_chest=({float(bl[0]):+.3f},"
              f"{float(bl[1]):+.3f},{float(bl[2]):.3f}) well={bool(scene._can_in_well()[0])} "
              f"o={bool(scene._opened[0])} e={bool(scene._evicted[0])} "
              f"b={bool(scene._binned[0])} st={bool(scene._seated[0])} "
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
        cp, cyaw = chest_pose()
        wx = float(cp[0]) + math.cos(cyaw) * x_l - math.sin(cyaw) * y_l
        wy = float(cp[1]) + math.sin(cyaw) * x_l + math.cos(cyaw) * y_l
        return wx, wy, cyaw

    def yaw_quat(yaw: float) -> tuple:
        return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

    def lying_quat(yaw: float) -> tuple:
        cy, sy, c45 = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(math.pi / 4)
        return (cy * c45, -sy * c45, cy * c45, sy * c45)

    def place_lid(u: float, settle_steps: int = 30) -> None:
        """Kinematic probe placement of the lid at travel u in its rail channel
        (instrumentation, not a solution) + REAL physics steps before judging."""
        wx, wy, cyaw = to_world(0.0, u)
        write_pose(scene.lid, wx, wy, c.lid_z0 + 0.0005, yaw_quat(cyaw), settle_steps)

    def drop_bottle_well(hover_base: float = 0.095, settle_steps: int = 150) -> None:
        """Bottle upright over the well centre, base `hover_base` up, gravity drop."""
        wx, wy, cyaw = to_world(0.0, 0.0)
        write_pose(scene.bottle, wx, wy, hover_base + c.body_h / 2, yaw_quat(cyaw),
                   settle_steps)

    def push_lid(target_u: float, direction: float, tag: str, v_des: float = 0.08,
                 max_steps: int = 1200) -> float:
        """REAL force drive of the lid along its rail axis (bang-bang, stall
        escalation) — the smoke's actuator-moved reference. Returns the final u."""
        _cp, cyaw = chest_pose()
        push_dir = torch.tensor([-math.sin(cyaw) * direction,
                                 math.cos(cyaw) * direction, 0.0], device=device)
        f_push = 4.0
        best, last_bump = -1e9, 0
        for i in range(max_steps):
            u = float(scene._lid_u()[0])
            if (u - target_u) * direction >= 0.0:
                break
            v_axis = float((scene.lid.data.root_lin_vel_w[0] * push_dir).sum())
            f_axis = f_push if v_axis < v_des else 0.0
            f_w = (push_dir * f_axis).view(1, 1, 3).expand(n, 1, 3)
            scene.lid.set_external_force_and_torque(f_w.contiguous(), zero_wrench,
                                                    env_ids=all_ids, is_global=True)
            env.step(no_action)
            prog = direction * u
            if prog > best + 0.004:
                best, last_bump = prog, i
            elif i - last_bump > 240:
                f_push = min(f_push + 2.0, 12.0)
                last_bump = i
                print(f"[smoke] {tag}: lid stalled at u={u:+.3f}, force -> "
                      f"{f_push:.1f} N", flush=True)
        scene.lid.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
        step(40)
        return float(scene._lid_u()[0])

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    fin0 = (torch.isfinite(scene.lid.data.root_state_w).all()
            and torch.isfinite(scene.bottle.data.root_state_w).all()
            and torch.isfinite(scene.can.data.root_state_w).all())
    u0 = float(scene._lid_u()[0])
    bz = float((scene.bottle.data.root_pos_w - scene.env_origins)[0, 2])
    still = (float(scene.lid.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.bottle.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.can.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    check("settle: states finite, lid closed (u readback ~0), can standing in the "
          "well, bottle upright on the ground, everything at rest",
          bool(fin0) and abs(u0) <= c.lid_closed_tol
          and bool(scene._can_in_well()[0])
          and abs(bz - c.body_h / 2) < 0.012 and still)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        cp, cyaw = chest_pose()
        bp = (scene.bottle.data.root_pos_w - scene.env_origins)[0]
        np2 = (scene.bin.data.root_pos_w - scene.env_origins)[0]
        bq = scene.bin.data.root_quat_w[0]
        byaw = 2.0 * math.atan2(float(bq[3]), float(bq[0]))
        kl = scene._chest_local(scene.can)[0]
        reads.append((float(cp[0]), float(cp[1]), cyaw,
                      float(np2[0]), float(np2[1]), byaw,
                      float(bp[0]), float(bp[1]), float(kl[0]), float(kl[1]),
                      float((bp[:2] - np2[:2]).norm()),
                      float((bp[:2] - cp[:2]).norm())))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (chest_x, chest_y, chest_yaw, bin_x, bin_y, "
          f"bin_yaw, bottle_x, bottle_y, can_dx, can_dy, bottle-bin, bottle-chest):\n"
          f"{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: chest xy + yaw and bin xy + yaw vary across seeded resets "
          "(readback)",
          spread[0] > 0.008 and spread[1] > 0.008 and spread[2] > 0.04
          and spread[3] > 0.02 and spread[4] > 0.02 and spread[5] > 0.08)
    check("randomization: bottle position and the can's in-well jitter vary; bottle "
          "never spawns within 30 cm of the bin or 20 cm of the chest centre",
          spread[6] > 0.02 and spread[7] > 0.04
          and (spread[8] > 0.0015 or spread[9] > 0.0015)
          and float(arr[:, 10].min()) >= 0.30 and float(arr[:, 11].min()) >= 0.20)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy A: stand it "inside" ======================
    # The seed just stands the bottle upright inside the fridge box. Here, stood
    # upright on the DECK (the only "inside" floor besides the well), it earns only
    # the lid-open credit: the seat band demands the bottle's BASE down inside the
    # well (deck base 0.080 vs band max 0.030 — static assert).
    torch.manual_seed(41)
    env.reset()
    step(10)
    place_lid(0.240, settle_steps=30)
    wx, wy, cyaw = to_world(0.076, 0.0)
    write_pose(scene.bottle, wx, wy, c.deck_z + c.body_h / 2 + 0.001, yaw_quat(cyaw),
               settle_steps=90)
    report("seed-upright")
    s, ok = judge()
    check("seed strategy A: bottle stood upright on the DECK (lid open) earns only "
          "the open credit (<= 0.16), no seat latch, no success; deck base height "
          "is statically outside the seat band",
          not bool(scene._seated[0]) and not bool(scene._seated_now()[0])
          and not ok and s <= 0.16 and c.deck_z > c.seat_base_band[1] + 0.02)

    # =========================== 7. seed strategy B: drop on the CLOSED chest ===============
    # The seed's carry-drop has no lid concept: released above this chest, the CLOSED
    # lid denies access — the bottle lands on the lid (or rolls off outside).
    torch.manual_seed(51)
    env.reset()
    step(10)
    wx, wy, cyaw = to_world(0.0, -0.045)
    write_pose(scene.bottle, wx, wy, c.lid_z0 + c.lid_t / 2 + 0.05 + c.body_r,
               lying_quat(cyaw), settle_steps=180)
    report("seed-drop")
    s, ok = judge()
    bl = scene._chest_local(scene.bottle)[0]
    outside = float(bl[:2].norm()) > c.foot_hw + c.body_r
    on_lid = float(bl[2]) > c.wall_top - 0.02
    check("seed strategy B: bottle released above the CLOSED chest never gets inside "
          "(lands on the lid or off the box), no latches, no success, score <= 0.02",
          (on_lid or outside) and not ok and s <= 0.02
          and not bool(scene._seated[0]) and not bool(scene._opened[0]))

    # =========================== 8. occupied well rejects the bottle ========================
    # Insert-before-evict: dropped straight at the OCCUPIED well the bottle lands on
    # the can and topples off — it can never seat. Static exclusion math: the well
    # holds two centres at most ~30 mm apart, but the cylinders need 63 mm.
    torch.manual_seed(61)
    env.reset()
    step(10)
    place_lid(0.240, settle_steps=30)
    drop_bottle_well(hover_base=0.200, settle_steps=180)
    report("occupied-well")
    s, ok = judge()
    max_sep = math.hypot((c.well_hw - c.can_r) + (c.well_hw - c.body_r),
                         (c.well_hw - c.can_r) + (c.well_hw - c.body_r))
    check("occupied well: bottle dropped into the occupied well cannot seat (lands on "
          "the can, topples off), can stays in the well, no seat latch, no success; "
          "static exclusion: max in-well centre separation < sum of radii",
          not bool(scene._seated_now()[0]) and not bool(scene._seated[0])
          and bool(scene._can_in_well()[0]) and not ok and s <= 0.16
          and max_sep < (c.can_r + c.body_r) - 0.02)

    # =========================== 9. no-open shortcut latches nothing ========================
    # Order chain: the can teleported out of the NEVER-OPENED chest into the bin
    # physically sits in the bin (readback: binned_now True) but `evicted`/`binned`
    # never latch — through a closed lid there is no legitimate way out.
    torch.manual_seed(71)
    env.reset()
    step(10)
    np2 = (scene.bin.data.root_pos_w - scene.env_origins)[0]
    write_pose(scene.can, float(np2[0]), float(np2[1]), 0.28, (1.0, 0.0, 0.0, 0.0),
               settle_steps=150)
    report("no-open-cheat")
    s, ok = judge()
    check("no-open shortcut: can teleported into the bin while the lid never opened — "
          "it IS in the bin (readback) yet the order chain latches nothing, "
          "score <= 0.02, no success",
          bool(scene._binned_now()[0]) and not bool(scene._evicted[0])
          and not bool(scene._binned[0]) and not ok and s <= 0.02)

    # =========================== 10. reversed swap: bottle into the bin =====================
    torch.manual_seed(81)
    env.reset()
    step(10)
    np2 = (scene.bin.data.root_pos_w - scene.env_origins)[0]
    write_pose(scene.bottle, float(np2[0]), float(np2[1]), 0.30, (1.0, 0.0, 0.0, 0.0),
               settle_steps=150)
    report("reversed-swap")
    s, ok = judge()
    check("reversed swap: the BOTTLE dropped into the discard bin earns nothing "
          "(score <= 0.02), no success",
          not ok and s <= 0.02 and not bool(scene._seated[0]))

    # =========================== 11. wrong disposal: can beside the bin =====================
    # Proper open + evict + seat, but the can is parked on the GROUND beside the bin:
    # partial credit only, no success. (This episode also hosts checks 12-13.)
    torch.manual_seed(91)
    env.reset()
    step(10)
    place_lid(0.240, settle_steps=30)                      # opened latch (probe)
    np2 = (scene.bin.data.root_pos_w - scene.env_origins)[0]
    write_pose(scene.can, float(np2[0]) + 0.19, float(np2[1]) + 0.13,
               c.can_h / 2 + 0.001, (1.0, 0.0, 0.0, 0.0), settle_steps=60)  # evicted
    drop_bottle_well(settle_steps=160)                     # seated (well is vacant)
    report("wrong-disposal")
    s_wd, ok = judge()
    check("wrong disposal: lid opened, can evicted but parked on the ground BESIDE "
          "the bin, bottle seated in the well — partial credit only (~0.55), no "
          "binned latch, NOT success",
          bool(scene._seated_now()[0]) and bool(scene._evicted[0])
          and not bool(scene._binned[0]) and not ok and 0.50 <= s_wd <= 0.60)

    # =========================== 12. clear-path force close (actuator-moved ref) ============
    # From that state the lid REALLY closes under force over the seated bottle —
    # proving the closing stroke is physically achievable and that the near-miss
    # below is not hiding a jam — and success STILL fails on the unbinned can alone.
    u_before = float(scene._lid_u()[0])
    u_after = push_lid(0.006, -1.0, "force-close")
    report("forced-closed")
    s_fc, ok = judge()
    check("clear-path close: the lid force-slides shut over the seated bottle "
          f"(actuator-moved readback u {u_before:+.3f} -> {u_after:+.3f}, closed "
          "True) and STILL no success — the unbinned can is the only missing clause",
          u_before > 0.20 and bool(scene._lid_closed_now()[0])
          and bool(scene._seated_now()[0]) and not ok and s_fc >= s_wd - 1e-6)

    # =========================== 13. near-miss lid: 30 mm short =============================
    # Reopen the lid a crack FIRST (so success is never constructed), then bin the
    # can: everything else right, lid 30 mm from its stop -> 0.75, NOT success.
    place_lid(0.030, settle_steps=20)
    np2 = (scene.bin.data.root_pos_w - scene.env_origins)[0]
    write_pose(scene.can, float(np2[0]), float(np2[1]), 0.28, (1.0, 0.0, 0.0, 0.0),
               settle_steps=150)                           # binned latch (evicted set)
    report("near-miss-lid")
    s_nm, ok = judge()
    u_nm = float(scene._lid_u()[0])
    check("near-miss lid: bottle seated, can binned, lid parked 30 mm short of "
          "closed — score 0.75 (all latches, capped below 1), NOT success",
          bool(scene._seated_now()[0]) and bool(scene._binned_now()[0])
          and u_nm > c.lid_closed_tol + 0.010 and not ok
          and 0.70 <= s_nm <= 0.85)

    # =========================== 14. lid-path knockover =====================================
    # A bottle stood on the DECK pokes 40 mm above the wall top, into the lid's
    # path: the closing lid (REAL force drive) knocks it over — it cannot end
    # seated, and deck-standing can never be laundered into success by closing.
    torch.manual_seed(101)
    env.reset()
    step(10)
    place_lid(0.240, settle_steps=30)
    wx, wy, cyaw = to_world(0.076, 0.0)
    write_pose(scene.bottle, wx, wy, c.deck_z + c.body_h / 2 + 0.001, yaw_quat(cyaw),
               settle_steps=60)
    u_before = float(scene._lid_u()[0])
    u_after = push_lid(0.006, -1.0, "knockover-close", max_steps=900)
    step(120)  # let the knocked bottle finish falling + settle
    report("knockover")
    s, ok = judge()
    standing_top = c.deck_z + c.body_h + c.neck_h
    check("lid-path knockover: deck-standing bottle pokes above the wall top (static "
          f"assert) and the force-closing lid knocks it over (lid moved "
          f"{u_before:+.3f} -> {u_after:+.3f}) — bottle NOT seated, no success",
          standing_top > c.wall_top + 0.02 and u_before > 0.20
          and u_after < u_before - 0.10
          and not bool(scene._seated_now()[0]) and not bool(scene._seated[0]) and not ok)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.chest.data.root_state_w).all()
           and torch.isfinite(scene.lid.data.root_state_w).all()
           and torch.isfinite(scene.bottle.data.root_state_w).all()
           and torch.isfinite(scene.can.data.root_state_w).all()
           and torch.isfinite(scene.bin.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.chest_swap")
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
    try:
        main()
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001 - die loudly, don't idle to the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL (crash: {type(e).__name__})", flush=True)
        os._exit(1)

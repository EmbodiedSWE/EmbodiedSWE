"""Smoke / rubric-REJECTION battery for LatchVaultScene (sim_gen task
`screw_nail_i59`) — NullRobot, teleported probe states + force probes, RECORDED.

This is NOT a solution (solve.py — slide both latches out by force, lift the freed lid
dynamically, extract the block and drop it into the dish — is the acceptance evidence
that the rubric ACCEPTS a correct outcome; it passes on seeds 0/1). Every teleport here
is instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled state and
asserts the rubric REJECTS it — plus applied-force probes that prove the LOCK is
physically real: the SAME velocity-regulated lift (solve's own P2 controller, <= 6 N)
that frees an unlocked lid cannot free a locked one.
No probe in this battery ever reaches success(), and a final audit asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: lid seated in the pocket,
                            block covered inside the cavity, both latches engaged;
                            score ~0 at rest, no success;
  3-4. randomization      — READBACK over 6 seeded resets: block position in the
                            cavity, latch engagement depth, dish slot + yaw, and the
                            decoy position all vary (block always covered, latches
                            always engaged);
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed-strategy       — the seed family's move (press down and TWIST, the
                            screwdriver motion) applied to the lid achieves nothing:
                            lid stays seated, score ~0;
  7.  lock (both latches) — solve's own regulated lift (velocity servo, <= 6 N cap,
                            6 N ~ 5x lid weight) raises the locked lid < 12 mm and
                            never clears the case: the interlock is contact geometry;
  8.  lock (one latch)    — one latch teleported to its stop along its own joint DOF,
                            the other still engaged: the same regulated lift still
                            cannot free the lid (rise < 22 mm even tilted, never
                            clear), and the engaged latch stays engaged;
  9.  unlocked (accepts)  — BOTH latches out: the very same regulated lift now pops
                            the lid clear (> 15 cm) — the cap in 7-8 was the latches,
                            nothing else; still NOT success;
  10. near-miss           — block settled on the floor 8.5 cm from the dish center
                            (outside the +-3.5 cm window, inside near_r) -> NOT in
                            dish, no success, only partial credit;
  11. rim perch           — block balanced ON the dish wall reads above the height
                            window (dish-frame z > 34 mm) -> rejected, then removed
                            before it can topple in;
  12. decoy in dish       — the ORANGE decoy settled inside the dish counts for
                            NOTHING (identity, not geometry): score ~0, no success;
  13. settle gate         — the green block IN the dish window but still moving is
                            NOT success (velocity gates are real); removed before it
                            can settle;
  14. latched credit      — teleporting the block far from the dish afterwards leaves
                            the latched score unchanged while gem_in_dish() drops;
  15. rejection audit     — success() was never True at ANY judged point;
  16. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.screw_nail_i59.smoke --headless
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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.latch_vault")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(3, device=device)
    cx, cy = c.case_pos

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.35, -1.05, 0.95)) + o),
                                tuple(np.array((0.42, 0.00, 0.05)) + o),
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

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        s, ok = judge()
        ext = scene.bolt_extension()[0]
        lid, gem, dish = rel(scene.lid), rel(scene.gem), rel(scene.dish)
        print(f"[smoke] {tag:16s} | bolts=({float(ext[0]):.3f},{float(ext[1]):.3f})"
              f" lid=({float(lid[0]):+.3f},{float(lid[1]):+.3f},{float(lid[2]):.3f})"
              f" gem=({float(gem[0]):+.3f},{float(gem[1]):+.3f},{float(gem[2]):.3f})"
              f" dish=({float(dish[0]):+.3f},{float(dish[1]):+.3f})"
              f" in_dish={bool(scene.gem_in_dish()[0])}"
              f" score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, xyz, quat=None, vel=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor(xyz, device=device)
        if quat is None:
            st[:, 3] = 1.0
        else:
            st[:, 3:7] = torch.tensor(quat, device=device)
        if vel is not None:
            st[:, 7:10] = torch.tensor(vel, device=device)
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def wrench(body, f3: torch.Tensor, t3: torch.Tensor) -> None:
        """World wrench expressed in the body's current link frame (the house
        convention: is_global=True silently drops the torque on this stack)."""
        from isaaclab.utils.math import quat_apply_inverse

        q = body.data.root_link_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f3.view(1, 3).expand(n, 3)).unsqueeze(1),
            quat_apply_inverse(q, t3.view(1, 3).expand(n, 3)).unsqueeze(1),
            env_ids=all_ids)

    def lid_lift_probe(steps: int = 300) -> float:
        """Solve's OWN lid-lift controller, verbatim: gravity feed-forward plus a
        velocity servo toward 0.12 m/s, clamped to [0, 6] N (6 N ~ 5x lid weight).
        This is the honest lock probe — the exact regulated pull that pops the
        unlocked lid clear in solve.py P2 (and in check 9). A constant 6 N would
        instead hammer the lid at ~50 m/s^2 and can vibration-walk a springless
        slide latch open — a jackhammer, not a lift. Returns the max lid-center
        height reached; releases afterwards."""
        f3 = torch.zeros(3, device=device)
        zmax = 0.0
        for _ in range(steps):
            vz = float(scene.lid.data.root_lin_vel_w[0, 2])
            f3[2] = min(max(c.lid_mass * 9.81 + 8.0 * (0.12 - vz), 0.0), 6.0)
            wrench(scene.lid, f3, zero3)
            step(1)
            zmax = max(zmax, float(rel(scene.lid)[2]))
            if zmax > 0.20:
                break
        wrench(scene.lid, zero3, zero3)
        step(45)
        return zmax

    def set_bolt(name: str, sgn: float, ext: float, settle_steps: int = 30) -> None:
        """Probe-only: reposition a latch ALONG ITS OWN prismatic DOF (the kinematic
        case anchor is world-fixed; a teleport along the joint axis is safe)."""
        place(scene.bolts[name], (cx + sgn * ext, cy, c.bolt_z),
              settle_steps=settle_steps)

    def layout_sane(tag: str) -> bool:
        """Reset honesty: lid seated, block covered in the cavity, latches engaged."""
        lid, gem = rel(scene.lid), rel(scene.gem)
        ext = scene.bolt_extension()[0]
        ok = (abs(float(lid[2]) - c.lid_seat_z) < 0.005
              and abs(float(lid[0]) - cx) < 0.01 and abs(float(lid[1]) - cy) < 0.01
              and float(gem[2]) < c.wall_top - 0.01
              and abs(float(gem[0]) - cx) < c.cavity_inner[0] / 2
              and abs(float(gem[1]) - cy) < c.cavity_inner[1] / 2
              and all(c.bolt_eng - 0.003 <= float(e) <= c.bolt_eng + c.bolt_jitter + 0.003
                      for e in ext))
        if not ok:
            print(f"[smoke] layout sanity VIOLATION at {tag}", flush=True)
        return ok

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    bodies = (scene.lid, scene.dish, scene.gem, scene.decoy, *scene.bolts.values())
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("settle: all states finite; lid seated, block covered, latches engaged",
          fin and layout_sane("reset"))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    sane = True
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(30)
        sane = sane and layout_sane(f"seed {sd}")
        gem, dish, dec = rel(scene.gem), rel(scene.dish), rel(scene.decoy)
        q = scene.dish.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        ext = scene.bolt_extension()[0]
        slots = np.array(c.dish_slots)
        slot = int(np.argmin(((slots - np.array([float(dish[0]), float(dish[1])])) ** 2
                              ).sum(axis=1)))
        reads.append((float(gem[0]), float(gem[1]), float(ext[0]), float(ext[1]),
                      float(dish[0]), float(dish[1]), yaw, slot,
                      float(dec[0]), float(dec[1])))
    arr = np.array(reads)
    print("[smoke] randomization readback (gem_x, gem_y, ext_px, ext_nx, dish_x, "
          f"dish_y, dish_yaw, slot, decoy_x, decoy_y):\n{np.round(arr, 4)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: block position in the cavity and latch engagement vary "
          f"(readback spreads gem=({spread[0]:.3f},{spread[1]:.3f}) "
          f"ext=({spread[2]:.4f},{spread[3]:.4f}))",
          spread[0] > 0.02 and spread[1] > 0.008
          and (spread[2] > 0.0015 or spread[3] > 0.0015))
    slots_used = {int(x) for x in arr[:, 7]}
    check("randomization: dish slot, dish yaw and decoy position vary (readback: "
          f"{len(slots_used)} slots, yaw spread {spread[6]:.2f} rad), and every reset "
          "spawns covered + locked",
          len(slots_used) >= 2 and spread[6] > 0.3 and spread[9] > 0.02 and sane)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. the seed family's move achieves nothing =================
    # rlbench/screw_nail's verb is press-down-and-TWIST (drive a fastener by rotation).
    # Apply exactly that to the lid: 4 N down + 0.3 N m about the vertical axis.
    env.reset(seed=41)
    step(30)
    f3 = torch.zeros(3, device=device)
    t3 = torch.zeros(3, device=device)
    f3[2], t3[2] = -4.0, 0.3
    for _ in range(240):
        wrench(scene.lid, f3, t3)
        step(1)
    wrench(scene.lid, zero3, zero3)
    step(45)
    report("press+twist")
    lid = rel(scene.lid)
    s, ok = judge()
    check("seed-strategy analog: pressing down and twisting the lid (the screwdriver "
          "motion) achieves nothing — lid still seated, score ~0",
          abs(float(lid[2]) - c.lid_seat_z) < 0.008 and abs(float(lid[0]) - cx) < 0.03
          and abs(float(lid[1]) - cy) < 0.03 and s <= 0.02 and not ok)

    # =========================== 7. lock probe: both latches engaged ========================
    env.reset(seed=51)
    step(30)
    ext0 = scene.bolt_extension()[0].clone()
    zmax = lid_lift_probe()
    report("lift-locked")
    lid = rel(scene.lid)
    ext1 = scene.bolt_extension()[0]
    s, ok = judge()
    check("lock (both latches): solve's regulated lift (<= 6 N) raises the locked lid "
          f"only {(zmax - c.lid_seat_z) * 1000:.1f} mm (< 12 mm), never clear of the "
          "case, it reseats on release, and both latches stay engaged "
          f"(ext {float(ext0[0]):.3f},{float(ext0[1]):.3f} -> "
          f"{float(ext1[0]):.3f},{float(ext1[1]):.3f})",
          zmax - c.lid_seat_z < 0.012 and not bool(scene.lid_latch[0] > 0)
          and abs(float(lid[2]) - c.lid_seat_z) < 0.008 and s <= 0.02 and not ok
          and not bool(scene.bolt_retracted()[0].any()))

    # =========================== 8. lock probe: ONE latch retracted =========================
    env.reset(seed=61)
    step(30)
    set_bolt("bolt_px", +1.0, c.bolt_eng + c.bolt_stroke - 0.003)
    assert bool(scene.bolt_retracted()[0, 0]), "probe setup: +x latch not retracted"
    zmax = lid_lift_probe()
    report("lift-one-latch")
    lid = rel(scene.lid)
    ext1 = scene.bolt_extension()[0]
    _s, ok = judge()
    check("lock (one latch): with one latch at its stop the same regulated lift still "
          f"cannot free the lid — max rise {(zmax - c.lid_seat_z) * 1000:.1f} mm "
          "(< 22 mm even tilted), never clear, reseats in the pocket, and the engaged "
          f"latch stays engaged (ext_nx={float(ext1[1]):.3f})",
          zmax - c.lid_seat_z < 0.022 and not bool(scene.lid_latch[0] > 0)
          and abs(float(lid[0]) - cx) < 0.03 and abs(float(lid[1]) - cy) < 0.03
          and float(lid[2]) < 0.09 and not ok
          and not bool(scene.bolt_retracted()[0, 1]))

    # =========================== 9. unlocked: the same force frees it (accepts) =============
    set_bolt("bolt_nx", -1.0, c.bolt_eng + c.bolt_stroke - 0.003)
    assert bool(scene.bolt_retracted()[0].all()), "probe setup: latches not retracted"
    zmax = lid_lift_probe()
    report("lift-unlocked")
    _s, ok = judge()
    check("unlocked: with BOTH latches out the very same regulated lift pops the lid "
          f"clear (zmax={zmax:.3f} > 0.15) — the cap in the locked probes was the "
          "latches; still not success",
          zmax > 0.15 and bool(scene.lid_latch[0] > 0) and not ok)

    # =========================== 10. near-miss: beside the dish =============================
    env.reset(seed=71)
    step(30)
    dish = rel(scene.dish)
    place(scene.gem, (float(dish[0]) + 0.085, float(dish[1]), c.gem_edge / 2 + 0.002),
          settle_steps=60)
    report("near-miss")
    s, ok = judge()
    check("near-miss: block settled 8.5 cm from the dish center (outside the 3.5 cm "
          "window) -> NOT in dish, no success, score <= 0.36",
          not bool(scene.gem_in_dish()[0]) and not ok and s <= 0.36)

    # =========================== 11. rim perch ==============================================
    env.reset(seed=81)
    step(30)
    from isaaclab.utils.math import quat_apply

    dish = rel(scene.dish)
    q = scene.dish.data.root_quat_w[0]
    loc = torch.tensor([c.dish_inner + 0.004, 0.0,
                        c.dish_wall_top + c.gem_edge / 2 + 0.002], device=device)
    off = quat_apply(q.view(1, 4), loc.view(1, 3))[0]
    place(scene.gem, (float(dish[0] + off[0]), float(dish[1] + off[1]),
                      float(dish[2] + off[2])), quat=(float(q[0]), float(q[1]),
                      float(q[2]), float(q[3])), settle_steps=6)
    rim_rel = scene._gem_in_dish_frame()[0]
    report("rim-perch")
    _s, ok = judge()
    rim_ok = float(rim_rel[2]) > c.dish_z_hi and not bool(scene.gem_in_dish()[0]) and not ok
    # remove the perched block BEFORE it can topple into the dish and settle
    place(scene.gem, (0.85, 0.0, c.gem_edge / 2 + 0.002), settle_steps=30)
    check("rim perch: block balanced ON the dish wall reads dish-frame "
          f"z={float(rim_rel[2]) * 1000:.0f} mm > {c.dish_z_hi * 1000:.0f} mm -> "
          "rejected by the height window", rim_ok)

    # =========================== 12. decoy in dish (identity) ===============================
    env.reset(seed=91)
    step(30)
    dish = rel(scene.dish)
    place(scene.decoy, (float(dish[0]), float(dish[1]), 0.03), settle_steps=60)
    report("decoy-in-dish")
    dec_rel = (scene.decoy.data.root_pos_w - scene.dish.data.root_pos_w)[0]
    s, ok = judge()
    check("decoy: the ORANGE block settled inside the dish "
          f"(|rel|={float(dec_rel[:2].norm()) * 100:.1f} cm) counts for NOTHING — "
          "score ~0, no success (identity, not geometry)",
          float(dec_rel[:2].norm()) < c.dish_xy_tol and s <= 0.02 and not ok)

    # =========================== 13. settle gate ============================================
    env.reset(seed=101)
    step(30)
    dish = rel(scene.dish)
    place(scene.gem, (float(dish[0]), float(dish[1]), 0.025),
          vel=(0.35, 0.0, 0.0), settle_steps=2)
    v_now = float(scene.gem.data.root_lin_vel_w[0].norm())
    report("settle-gate")
    s_before, ok = judge()
    gate_ok = v_now > c.settle_lin and not ok
    check("settle gate: the block IN the dish window but moving at "
          f"{v_now:.2f} m/s is NOT success (velocity gates are real)", gate_ok)

    # =========================== 14. latched credit survives moving away ====================
    # remove it BEFORE it can settle in the dish (this battery must never succeed)
    place(scene.gem, (0.85, 0.0, c.gem_edge / 2 + 0.002), settle_steps=40)
    report("moved-away")
    s_after, ok = judge()
    check("latched credit: teleporting the block far from the dish leaves the latched "
          f"score unchanged ({s_before:.2f} -> {s_after:.2f}) while gem_in_dish() drops",
          abs(s_after - s_before) < 1e-3 and not bool(scene.gem_in_dish()[0]) and not ok)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.latch_vault")
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

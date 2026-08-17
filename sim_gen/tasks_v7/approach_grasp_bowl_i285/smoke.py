"""Smoke / rubric-REJECTION battery for WedgePressScene (sim_gen task
`approach_grasp_bowl_i285`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — hover the wedge over the roof slot and let gravity
drive it into the seam — is the acceptance evidence that the rubric ACCEPTS a correct
outcome). Every teleport here is instrumentation that CONSTRUCTS a wrong (or partial)
outcome and asserts the rubric REJECTS it. Because ANY clean wedge drop into a live,
unspoiled seam genuinely solves the task, every construct that ends with delivered
sleds is deliberately SPOILED FIRST — no probe in this battery ever reaches success(),
and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: sleds at box rest height on
                            the slick strip, wedge lying at its pad height; score ~0;
  3-4. randomization      — READBACK over 8 seeded resets: rig xy and FREE yaw vary,
                            the spawn pad flips to both rig sides (sign-correlated
                            with the latched side), seam gap and seam centre vary,
                            the color-end assignment flips both ways;
  5.  null policy         — 300 idle steps -> score ~0, no success, no spoil, and the
                            sleds do not creep along the tunnel;
  6.  SEED strategy       — the seed's whole plan ("grasp the object, lift it, carry
                            it to the goal") executed on both sleds: each is raised
                            above `lift_z`, carried, and set down past the dock line;
                            the wedge is then genuinely dropped and seats. The tableau
                            is EXACTLY the goal picture — both sleds docked, wedge
                            seated, all settled — yet the lift latch REJECTS it: NOT
                            success, score <= 0.20;
  7.  finger-drag         — a sled slid 30 mm outward at floor height with the wedge
                            nowhere near the seam: the unattended-motion provenance
                            latch spoils it (the seed's floor-level variant fails too);
  8.  spoiled-then-perfect— continuing ep. 7: a flawless wedge strike afterwards
                            physically delivers BOTH sleds and seats the wedge — and
                            the episode still scores <= 0.20, NOT success (the spoil
                            is irreversible; ALSO the honest way to show a full strike
                            in smoke without tripping genuine success);
  9.  tableau w/o wedge   — sleds hand-placed at the docks (after a spoiling lift),
                            wedge never presented: the delivered latch stays 0 (it is
                            GATED on engagement) — the goal picture carries no credit;
  10. engaged-only        — the wedge presented in the seam window for 2 substeps
                            (never touching the sleds, readback-verified) then
                            withdrawn to the pad: exactly the 0.10 engagement credit,
                            which PERSISTS after withdrawal; no spoil, no success;
  11. parked on the roof  — the wedge lying on a roof slab (or slid off it): never
                            presented in the window -> no credit;
  12. rejection audit     — success() was never True at ANY judged point;
  13. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.approach_grasp_bowl_i285.smoke --headless
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
    env = ENVS.get("simgen.wedge_press")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((0.72, -0.68, 0.52)) + o),
                                tuple(np.array((0.00, 0.00, 0.05)) + o),
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

    def sleds_rig() -> torch.Tensor:
        return scene._sled_pos()[0]  # (2,3)

    def wedge_rig() -> torch.Tensor:
        return scene._wedge_pos()[0]  # (3,)

    def psi() -> float:
        return float(scene.rig_yaw[0])

    def yaw_quat() -> tuple[float, float, float, float]:
        return (math.cos(psi() / 2), 0.0, 0.0, math.sin(psi() / 2))

    def rig_world(lx: float, ly: float) -> tuple[float, float]:
        cp, sp = math.cos(psi()), math.sin(psi())
        return (float(scene.rig_pos[0, 0]) + lx * cp - ly * sp,
                float(scene.rig_pos[0, 1]) + lx * sp + ly * cp)

    def report(tag: str) -> None:
        sp = sleds_rig()
        wp = wedge_rig()
        s, ok = judge()
        print(f"[smoke] {tag:18s} | sleds_x=[" + ",".join(
            f"{float(sp[i, 0]):+.3f}" for i in range(2)) + "] sleds_z=[" + ",".join(
            f"{float(sp[i, 2]):.3f}" for i in range(2)) + "] "
            f"wedge=({float(wp[0]):+.3f},{float(wp[1]):+.3f},{float(wp[2]):.3f}) "
            f"eng={float(scene.engaged[0]):.0f} "
            f"del={float(scene.delivered[0].sum()):.0f} "
            f"seat={float(scene.seated[0]):.0f} spoiled={bool(scene.spoiled[0])} "
            f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_body(body, wx: float, wy: float, z: float, quat=(1.0, 0.0, 0.0, 0.0),
                   settle_steps: int = 30) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = wx, wy, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def place_rig(body, lx: float, ly: float, z: float, settle_steps: int = 30) -> None:
        wx, wy = rig_world(lx, ly)
        place_body(body, wx, wy, z, quat=yaw_quat(), settle_steps=settle_steps)

    def seam_center() -> float:
        sp = sleds_rig()
        return 0.5 * sum(
            float(sp[i, 0]) - math.copysign(c.sled_len / 2, float(sp[i, 0]))
            for i in range(2))

    dock_x = 0.105  # inside the delivered band, 5 mm clear of the end cap

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    sp = sleds_rig()
    wp = wedge_rig()
    fin0 = bool(torch.isfinite(scene.wedge.data.root_state_w).all()
                and all(torch.isfinite(s.data.root_state_w).all() for s in scene.sleds)
                and torch.isfinite(scene.housing.data.root_state_w).all())
    z_ok = all(abs(float(sp[i, 2]) - c.sled_rest_z) < 0.006 for i in range(2))
    check("settle: states finite; sleds at box rest height on the strip and the wedge "
          "lying at its pad height (readback)",
          fin0 and z_ok and abs(float(wp[2]) - c.wedge_pad_rest_z) < 0.010
          and not bool(scene.spoiled[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(5)
        sp = sleds_rig()
        pad_rig = scene._rig_frame(scene.spawn_pad.data.root_pos_w - scene.env_origins)[0]
        gap = abs(float(sp[0, 0]) - float(sp[1, 0])) - c.sled_len
        reads.append((float(scene.rig_pos[0, 0]), float(scene.rig_pos[0, 1]), psi(),
                      float(scene.pad_side[0]), float(pad_rig[1]),
                      float(sp[0, 0]), gap, seam_center()))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (rig_x, rig_y, yaw, side, pad_rig_y, "
          f"sled0_x, gap, seam_c):\n{arr}", flush=True)
    sides = set(arr[:, 3].tolist())
    pad_follows = all(r[4] * r[3] > 0.15 for r in reads)
    yaw_spread = max(float(np.ptp(np.sin(arr[:, 2]))), float(np.ptp(np.cos(arr[:, 2]))))
    check("randomization: rig xy and FREE yaw vary; the spawn pad flips to both rig "
          "sides and its teleport follows the latched side (readback)",
          float(np.ptp(arr[:, 0])) > 0.03 and float(np.ptp(arr[:, 1])) > 0.03
          and yaw_spread > 0.7 and sides == {1.0, -1.0} and pad_follows)
    swap_signs = set(np.sign(arr[:, 5]).tolist())
    check("randomization: seam gap and seam centre vary within their bands, and the "
          "color-end assignment flips both ways (readback)",
          float(np.ptp(arr[:, 6])) > 0.002 and float(arr[:, 6].min()) > c.gap_lo - 0.002
          and float(arr[:, 6].max()) < c.gap_hi + 0.002
          and float(np.ptp(arr[:, 7])) > 0.003 and swap_signs == {1.0, -1.0})

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(60)
    x0 = sleds_rig()[:, 0].clone()
    step(300)
    report("null-policy")
    drift = float((sleds_rig()[:, 0] - x0).abs().max())
    s, ok = judge()
    check("null policy: score ~0, no success, no spoil after 300 idle steps, and the "
          f"sleds do not creep (max drift {drift * 1000:.1f} mm)",
          s <= 0.02 and not ok and not bool(scene.spoiled[0]) and drift < 0.005)

    # =========================== 6. SEED strategy ============================================
    # The seed's whole plan: grasp the object, LIFT it, carry it to the goal — executed
    # honestly on both sleds (raised above lift_z, set down past the dock line), then
    # the wedge genuinely dropped so it seats. The tableau is exactly the goal picture —
    # and the lift latch must refuse it anyway.
    env.reset(seed=41)
    step(30)
    for i in range(2):
        sgn = math.copysign(1.0, float(sleds_rig()[i, 0]))
        lx, ly = rig_world(sgn * dock_x, 0.0)
        place_body(scene.sleds[i], lx, ly, 0.150, quat=yaw_quat(),
                   settle_steps=3)  # the lift — trips the latch (above lift_z)
        place_rig(scene.sleds[i], sgn * dock_x, 0.0, c.sled_rest_z + 0.001,
                  settle_steps=30)
    wx, wy = rig_world(seam_center(), 0.0)
    place_body(scene.wedge, wx, wy, 0.095, quat=yaw_quat(), settle_steps=160)
    step(60)
    report("seed-strategy")
    s, ok = judge()
    sp = sleds_rig()
    tableau = (all(abs(float(sp[i, 0])) > c.deliver_x for i in range(2))
               and bool(scene._in_window(scene._wedge_pos(), c.seat_z)[0]))
    check("seed strategy (both sleds lifted above lift_z and carried to the docks, "
          "wedge then seated): tableau complete yet spoiled -> NOT success, "
          "score <= 0.20",
          tableau and bool(scene.spoiled[0]) and not ok and s <= 0.20 + 1e-5)

    # =========================== 7. finger-drag surrogate ===================================
    env.reset(seed=51)
    step(30)
    sp = sleds_rig()
    sgn = math.copysign(1.0, float(sp[0, 0]))
    place_rig(scene.sleds[0], float(sp[0, 0]) + sgn * 0.030, 0.0,
              c.sled_rest_z + 0.001, settle_steps=20)
    report("finger-drag")
    s, ok = judge()
    check("finger-drag: a sled slid 30 mm outward at floor height with the wedge on "
          "its pad — the unattended-motion provenance latch spoils it (score <= 0.20, "
          "NOT success)",
          bool(scene.spoiled[0]) and s <= 0.20 + 1e-5 and not ok)

    # =========================== 8. spoiled-then-perfect ====================================
    # Continue ep. 7: now execute a FLAWLESS strike. Physics delivers both sleds and
    # seats the wedge — and the episode still fails: the spoil is irreversible. (This
    # is also the honest way to show a full strike inside smoke: the seam is already
    # spoiled, so genuine success is unreachable by construction.)
    wx, wy = rig_world(seam_center(), 0.0)
    place_body(scene.wedge, wx, wy, 0.095, quat=yaw_quat(), settle_steps=180)
    step(120)
    report("spoiled-perfect")
    s, ok = judge()
    sp = sleds_rig()
    delivered_phys = all(abs(float(sp[i, 0])) > c.deliver_x for i in range(2))
    check("spoiled-then-perfect: a flawless wedge strike afterwards physically "
          "delivers both sleds and seats the wedge — still NOT success, score <= 0.20 "
          "(spoil is irreversible)",
          delivered_phys and bool(scene._in_window(scene._wedge_pos(), c.seat_z)[0])
          and float(scene.delivered[0].sum()) == 2.0
          and bool(scene.spoiled[0]) and not ok and s <= 0.20 + 1e-5)

    # =========================== 9. tableau without the mechanism ===========================
    env.reset(seed=61)
    step(30)
    for i in range(2):
        sgn = math.copysign(1.0, float(sleds_rig()[i, 0]))
        lx, ly = rig_world(sgn * dock_x, 0.0)
        place_body(scene.sleds[i], lx, ly, 0.150, quat=yaw_quat(), settle_steps=3)
        place_rig(scene.sleds[i], sgn * dock_x, 0.0, c.sled_rest_z + 0.001,
                  settle_steps=30)
    step(60)
    report("tableau-no-wedge")
    s, ok = judge()
    check("tableau without the mechanism: sleds hand-placed at the docks, wedge never "
          "presented — the delivered latch stays 0 (gated on engagement), score ~0, "
          "NOT success",
          float(scene.engaged[0]) == 0.0 and float(scene.delivered[0].sum()) == 0.0
          and s <= 0.02 and not ok)

    # =========================== 10. engaged-only + latch persistence =======================
    env.reset(seed=71)
    step(30)
    wx, wy = rig_world(seam_center(), 0.0)
    place_body(scene.wedge, wx, wy, 0.066, quat=yaw_quat(), settle_steps=2)
    z_mid = float(wedge_rig()[2])
    no_touch = z_mid > 0.058  # first possible sled contact needs tip z ~0.054
    eng_mid = float(scene.engaged[0])
    place_body(scene.wedge, *rig_world(0.0, float(scene.pad_side[0]) * c.pad_off),
               0.30, quat=yaw_quat(), settle_steps=90)
    report("engaged-only")
    s, ok = judge()
    check("engaged-only: wedge presented in the seam window for 2 substeps (no sled "
          f"contact possible, tip z={z_mid:.3f} readback) then withdrawn — exactly the "
          "engagement credit (0.08 <= score <= 0.12), latch persists after withdrawal, "
          "no spoil, NOT success",
          eng_mid == 1.0 and no_touch and float(scene.engaged[0]) == 1.0
          and 0.08 <= s <= 0.12 and float(scene.delivered[0].sum()) == 0.0
          and not bool(scene.spoiled[0]) and not ok)

    # =========================== 11. wedge parked on the roof ===============================
    env.reset(seed=81)
    step(30)
    wyaw = psi()
    qw = math.cos(wyaw / 2) * math.sqrt(0.5)
    qx = math.cos(wyaw / 2) * math.sqrt(0.5)
    qy = math.sin(wyaw / 2) * math.sqrt(0.5)
    qz = math.sin(wyaw / 2) * math.sqrt(0.5)
    wx, wy = rig_world(0.126, 0.0)
    place_body(scene.wedge, wx, wy, c.roof_hi + c.wedge_w / 2 + 0.003,
               quat=(qw, qx, qy, qz), settle_steps=120)
    report("roof-parked")
    s, ok = judge()
    check("parked on the roof: the wedge lying on a roof slab (or slid off it) is "
          "never presented in the seam window — no credit (score <= 0.02), no spoil, "
          "NOT success",
          float(scene.engaged[0]) == 0.0 and s <= 0.02
          and not bool(scene.spoiled[0]) and not ok)

    # =========================== 12-13. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.wedge.data.root_state_w).all()
           and torch.isfinite(scene.housing.data.root_state_w).all()
           and torch.isfinite(scene.spawn_pad.data.root_state_w).all()
           and all(torch.isfinite(s.data.root_state_w).all() for s in scene.sleds))
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.wedge_press")
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

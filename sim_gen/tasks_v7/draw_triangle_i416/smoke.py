"""Smoke / rubric-REJECTION battery for TrayPoiseScene (sim_gen task
`draw_triangle_i416`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — loaded tray dropped onto the cap at the computed
CoM point, cubes dropped onto the live balanced tray, hands-off persistence — is the
acceptance evidence that the rubric ACCEPTS a correct outcome). Every teleport here
is instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled state
and asserts the rubric REJECTS it; no probe in this battery ever reaches success(),
and a final audit check asserts exactly that.

  1-2.  settle/no-NaN    — reset layout settles finite: tray flat on its skids on the
                           ground, slugs seated in DISTINCT pockets, cubes staged on
                           the floor, all quasi-still, score ~0, no success;
  3a-b. randomization    — READBACK over 8 seeded resets: pedestal xy + free yaw are
                           real; the slug LOADOUT flips (heavy-slug side changes sign,
                           light-slug pocket varies); tray pose and cube stage jitter
                           are real;
  4.    null policy      — 240 idle steps -> score ~0, no success;
  5.    state roundtrip  — get_state -> 60 disturbed steps -> set_state restores
                           poses, loadout and latches;
  6.    ground-serve     — both cubes placed on the deck of the GROUNDED tray: the
                           in-deck-band geometry reads true, but serve credit is
                           gated on the MOUNTED state -> score stays ~0;
  7.    centre mount     — the naive move: loaded tray dropped with the cap under its
                           geometric CENTRE. The asserted CoM overhang (>= 8 mm past
                           the cap edge) tips it off -> no credit;
  8.    near-miss mount  — cap under x_com + 40 mm (>= 12 mm CoM overhang): tips off
                           -> no credit (the mount point is a real, tight target);
  9.    FLAGSHIP swap    — slugs SWAPPED between their pockets and the tray mounted
                           at the SWAPPED balance point: it stands level, at height,
                           still — physically a perfect balance — yet the rubric
                           rejects it (slugs must sit in their ASSIGNED pockets):
                           no mount credit, no success, score ~0;
  10.   correct mount    — probe-mount at the true x_com: sustained stand earns
                           exactly the 0.35 mount credit; cubes still on the floor ->
                           NOT success (serving is load-bearing);
  11.   cube under tray  — a cube on the GROUND inside the mounted tray's xy
                           footprint: tray-frame z math rejects it as served;
  12.   far-end serve    — a cube dropped near the tray's far end: the shifted CoM
                           overhangs the cap and the whole tray CAPSIZES — serve
                           never latches, but the earlier mount credit SURVIVES
                           (latched, non-evaporating), success never;
  13.   reset clears     — a fresh reset returns the score to ~0;
  14.   rejection audit  — success() was never True at ANY judged point;
  15.   final no-NaN     — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.<task>.smoke --headless
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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tray_poise")().build(num_envs=args.num_envs,
                                               device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.10, -1.15, 0.85)) + o),
                                tuple(np.array((0.03, 0.03, 0.10)) + o),
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

    def report(tag: str) -> None:
        tz = float(scene.tray.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
        cap = scene._tray_local(scene.pedestal.data.root_pos_w)[0]
        s, ok = judge()
        spd = " ".join(
            f"{nm}={float(b.data.root_lin_vel_w[0].norm()):.3f}"
            for nm, b in (("tray", scene.tray), ("H", scene.slug_h),
                          ("L", scene.slug_l), ("ca", scene.cube_a),
                          ("cb", scene.cube_b)))
        print(f"[smoke] {tag:16s} | tray_z={tz:.3f} "
              f"up_z={float(scene._tray_up_z()[0]):+.3f} "
              f"cap=({float(cap[0]):+.3f},{float(cap[1]):+.3f}) "
              f"mounted={bool(scene._mounted()[0])} "
              f"home={bool(scene._slugs_home()[0])} "
              f"served={[bool(v) for v in scene._served()[0]]} "
              f"still={bool(scene._still()[0])} "
              f"m_ever={bool(scene._mounted_ever[0])} "
              f"s_ever={[bool(v) for v in scene._served_ever[0]]} "
              f"score={s:.3f} success={ok} frames={len(frames)} "
              f"| {spd} trayW={float(scene.tray.data.root_ang_vel_w[0].norm()):.3f}",
              flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, pos_w: torch.Tensor, quat_w: torch.Tensor) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat_w
        body.write_root_state_to_sim(st, all_ids)

    def yaw0() -> torch.Tensor:
        q = torch.zeros(n, 4, device=device)
        q[:, 0] = 1.0
        return q

    def mount_probe(x_cap: float, hxp: float, lxp: float,
                    settle: int = 480) -> None:
        """Probe constructor: teleport the loaded tray (slugs at pockets hxp/lxp)
        to hover 10 mm above the cap with the cap axis under tray-frame (x_cap, 0),
        release, settle. What stands or tips is pure physics. Tray yaw is ALIGNED
        to the pedestal yaw (edge-aligned square-on-square cap contact — the same
        geometry the solve demonstrates settles below the stillness gates; a yawed
        edge-on-face contact rings a larger PhysX contact limit cycle)."""
        tq = scene.pedestal.data.root_quat_w.clone()
        off = torch.tensor([x_cap, 0.0, 0.0], device=device).expand(n, 3)
        tp = torch.zeros(n, 3, device=device)
        tp[:, 0:2] = (scene.pedestal.data.root_pos_w[:, 0:2]
                      - quat_apply(tq, off)[:, 0:2])
        tp[:, 2] = scene.env_origins[:, 2] + c.mount_z + 0.010
        place(scene.tray, tp, tq)
        for sx, body, seat_z in ((hxp, scene.slug_h, c.seat_z_h),
                                 (lxp, scene.slug_l, c.seat_z_l)):
            loc = torch.tensor([sx, 0.0, seat_z], device=device).expand(n, 3)
            place(body, tp + quat_apply(tq, loc), tq)
        for _ in range(settle):
            env.step(no_action)
            if bool(scene._mounted_ever[0]):
                break
        step(60)

    def loadout() -> tuple[float, float, float]:
        hx, lx = float(scene._hx[0]), float(scene._lx[0])
        m_tot = c.tray_mass + c.slug_h_mass + c.slug_l_mass
        return hx, lx, (c.slug_h_mass * hx + c.slug_l_mass * lx) / m_tot

    def finite_all() -> bool:
        return bool(all(torch.isfinite(b.data.root_state_w).all()
                        for b in (scene.pedestal, scene.tray, scene.slug_h,
                                  scene.slug_l, scene.cube_a, scene.cube_b)))

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("show")
    tz = float(scene.tray.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    ph = scene._tray_local(scene.slug_h.data.root_pos_w)[0]
    pl = scene._tray_local(scene.slug_l.data.root_pos_w)[0]
    ca = float(scene.cube_a.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    cb = float(scene.cube_b.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    check("settle: states finite, tray flat on its skids on the ground, slugs seated "
          "in DISTINCT pockets (readback), cubes on the floor, all quasi-still",
          finite_all() and abs(tz - c.ground_z) < 0.004
          and float(scene._tray_up_z()[0]) > 0.99
          and bool(scene._slugs_home()[0])
          and abs(float(ph[0]) - float(pl[0])) > 0.05
          and ca < 0.03 and cb < 0.03 and bool(scene._still()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3. randomization is real ===================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        pp = (scene.pedestal.data.root_pos_w - scene.env_origins)[0]
        pq = scene.pedestal.data.root_quat_w[0]
        pyaw = math.degrees(2.0 * math.atan2(float(pq[3]), float(pq[0])))
        tp = (scene.tray.data.root_pos_w - scene.env_origins)[0]
        cp = (scene.cube_a.data.root_pos_w - scene.env_origins)[0]
        hx, lx, xc = loadout()
        # loadout from OBSERVED slug positions must match the assignment
        ph = scene._tray_local(scene.slug_h.data.root_pos_w)[0]
        pl = scene._tray_local(scene.slug_l.data.root_pos_w)[0]
        obs_ok = abs(float(ph[0]) - hx) < 0.01 and abs(float(pl[0]) - lx) < 0.01
        reads.append((float(pp[0]), float(pp[1]), pyaw, hx, lx, xc,
                      float(tp[0]), float(tp[1]), float(cp[0]), float(cp[1]),
                      1.0 if obs_ok else 0.0))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (ped_x, ped_y, ped_yaw_deg, hx, lx, "
          f"x_com, tray_x, tray_y, cube_x, cube_y, obs_ok):\n{arr}", flush=True)
    ped_spread = float((arr[:, 0:2].max(axis=0) - arr[:, 0:2].min(axis=0)).max())
    yaw_spread = float(arr[:, 2].max() - arr[:, 2].min())
    check("randomization: pedestal xy spread (> 8 mm) and free yaw spread (> 20 deg) "
          "are real (readback)", ped_spread > 0.008 and yaw_spread > 20.0)
    both_sides = (arr[:, 3] > 0).any() and (arr[:, 3] < 0).any()
    n_loadouts = len({(round(a, 3), round(b, 3)) for a, b in arr[:, 3:5]})
    tray_spread = float((arr[:, 6:8].max(axis=0) - arr[:, 6:8].min(axis=0)).max())
    cube_spread = float((arr[:, 8:10].max(axis=0) - arr[:, 8:10].min(axis=0)).max())
    check("randomization: slug loadout flips (heavy slug on BOTH sides, >= 2 distinct "
          "loadouts, observed pockets match the assignment) and tray/cube jitter "
          "(> 8 mm) are real (readback)",
          bool(both_sides) and n_loadouts >= 2 and bool(arr[:, 10].all())
          and tray_spread > 0.008 and cube_spread > 0.008)

    # =========================== 4. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 5. state roundtrip =========================================
    torch.manual_seed(41)
    env.reset()
    step(30)
    saved = scene.get_state(all_ids)
    hx_s, lx_s, _ = loadout()
    # disturb: shove the tray away, then restore
    tq = yaw0()
    tp = torch.zeros(n, 3, device=device)
    tp[:, 0], tp[:, 1], tp[:, 2] = -0.30, -0.30, c.ground_z + 0.05
    tp += scene.env_origins
    place(scene.tray, tp, tq)
    step(60)
    scene.set_state(saved, all_ids)
    step(2)
    d_tray = float((scene.tray.data.root_pos_w[0]
                    - saved["tray"][0, 0:3]).norm())
    d_h = float((scene.slug_h.data.root_pos_w[0] - saved["slug_h"][0, 0:3]).norm())
    hx_r, lx_r, _ = loadout()
    s, ok = judge()
    check("state roundtrip: set_state restores tray/slug poses (< 10 mm after 2 "
          "steps), the loadout assignment and the ~0 score",
          d_tray < 0.010 and d_h < 0.010 and hx_r == hx_s and lx_r == lx_s
          and s <= 0.02 and not ok)

    # =========================== 6. ground-serve bypass =====================================
    # Both cubes on the deck of the GROUNDED tray: geometry reads in-band, but serve
    # credit is gated on the MOUNTED state — nothing scores.
    torch.manual_seed(51)
    env.reset()
    step(30)
    tp = scene.tray.data.root_pos_w
    tq = scene.tray.data.root_quat_w
    zc = c.deck_top + c.cube_s / 2 + 0.004
    for body, sy in ((scene.cube_a, c.serve_y), (scene.cube_b, -c.serve_y)):
        loc = torch.tensor([0.0, sy, zc], device=device).expand(n, 3)
        place(body, tp + quat_apply(tq, loc), tq)
    step(120)
    report("ground-serve")
    sv = scene._served()[0]
    s, ok = judge()
    check("ground-serve bypass: both cubes ON the grounded tray's deck read in-band "
          "geometrically, but serve credit requires the MOUNTED state — score ~0, "
          "no serve latch, no success",
          bool(sv.all()) and not bool(scene._served_ever[0].any())
          and s <= 0.02 and not ok)

    # =========================== 7. naive centre mount tips off =============================
    torch.manual_seed(61)
    env.reset()
    step(30)
    hx, lx, x_com = loadout()
    mount_probe(0.0, hx, lx)
    report("centre-mount")
    tz = float(scene.tray.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    fell = tz < c.mount_z - c.mount_z_tol - 0.004 \
        or float(scene._tray_up_z()[0]) < math.cos(math.radians(c.level_tol_deg))
    s, ok = judge()
    check(f"naive centre mount: cap under the tray CENTRE with CoM at "
          f"{x_com * 1000:+.0f} mm — the asserted overhang tips the tray off "
          f"(fell/tilted), no mount credit, score ~0, no success",
          fell and not bool(scene._mounted_ever[0]) and s <= 0.02 and not ok)

    # =========================== 8. near-miss mount tips off ================================
    torch.manual_seed(71)
    env.reset()
    step(30)
    hx, lx, x_com = loadout()
    off = x_com + math.copysign(0.040, x_com)
    mount_probe(off, hx, lx)
    report("near-miss")
    tz = float(scene.tray.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    fell = tz < c.mount_z - c.mount_z_tol - 0.004 \
        or float(scene._tray_up_z()[0]) < math.cos(math.radians(c.level_tol_deg))
    s, ok = judge()
    check("near-miss mount: cap 40 mm past the true CoM point (>= 12 mm CoM "
          "overhang) — tray tips off, no mount credit, score ~0, no success",
          fell and not bool(scene._mounted_ever[0]) and s <= 0.02 and not ok)

    # =========================== 9. FLAGSHIP: swapped slugs, perfect balance ================
    # Slugs SWAPPED between their two pockets and the tray mounted at the SWAPPED
    # balance point: physically a perfect, still, level stand on the cap — but the
    # slugs are not in their ASSIGNED pockets, so the rubric refuses everything.
    torch.manual_seed(81)
    env.reset()
    step(30)
    hx, lx, _ = loadout()
    m_tot = c.tray_mass + c.slug_h_mass + c.slug_l_mass
    x_sw = (c.slug_h_mass * lx + c.slug_l_mass * hx) / m_tot  # swapped balance point
    mount_probe(x_sw, lx, hx)  # heavy slug into the light slug's pocket & v.v.
    report("swapped-slugs")
    standing = (bool(scene._mounted()[0]) and bool(scene._still()[0])
                and float(scene._tray_up_z()[0]) > 0.996)
    s, ok = judge()
    check("FLAGSHIP swapped slugs: tray mounted at the SWAPPED balance point stands "
          "level, at cap height and still (readback proves a genuine equilibrium) — "
          "yet slugs-home is False, NO mount credit, score ~0, no success (the "
          "assigned-pocket identity clause is load-bearing)",
          standing and not bool(scene._slugs_home()[0])
          and not bool(scene._mounted_ever[0]) and s <= 0.02 and not ok)

    # =========================== 10. correct mount: credit but NOT success ==================
    torch.manual_seed(91)
    env.reset()
    step(30)
    hx, lx, x_com = loadout()
    mount_probe(x_com, hx, lx, settle=600)
    report("correct-mount")
    s10, ok = judge()
    check(f"correct mount: probe-mount at the true x_com={x_com * 1000:+.0f} mm "
          f"stands and earns exactly the mount credit (0.35, got {s10:.3f}); cubes "
          f"still on the floor -> NOT success (serving is load-bearing)",
          bool(scene._mounted_ever[0]) and bool(scene._mounted()[0])
          and abs(s10 - c.w_mount) < 0.01 and not ok)

    # =========================== 11. cube on the ground under the tray ======================
    # Placed in the TRAY frame (xy only) so it lands inside the judged footprint
    # for any pedestal yaw, on the ground, clear of the plinth on the cap's far side.
    tq_live = scene.tray.data.root_quat_w
    loc = torch.tensor([-math.copysign(0.060, x_com), 0.0, 0.0],
                       device=device).expand(n, 3)
    cp = torch.zeros(n, 3, device=device)
    cp[:, 0:2] = (scene.tray.data.root_pos_w[:, 0:2]
                  + quat_apply(tq_live, loc)[:, 0:2])
    cp[:, 2] = scene.env_origins[:, 2] + c.cube_s / 2 + 0.002
    place(scene.cube_a, cp, tq_live)  # tray-aligned: corners clear the plinth edge
    step(90)
    report("cube-under")
    sv = scene._served()[0]
    s, ok = judge()
    check("cube under tray: a cube on the GROUND inside the mounted tray's xy "
          "footprint is NOT served (tray-frame z math rejects), score unchanged "
          "(0.35), no success",
          not bool(sv[0]) and abs(s - c.w_mount) < 0.01 and not ok)

    # =========================== 12. far-end serve upsets the balance; credit survives ======
    # A cube dropped near the tray's far end shifts the ensemble CoM past the cap
    # edge: the balance is LIVE, so the tray tips — and then either capsizes or
    # sheds the free cube over the deck end and re-rights without it. Either way a
    # level mounted tray with a far-end cube aboard is impossible, serve never
    # latches, and the earlier mount credit SURVIVES (latched, non-evaporating).
    tp = scene.tray.data.root_pos_w
    tq_live = scene.tray.data.root_quat_w
    loc = torch.tensor([-math.copysign(0.105, x_com), 0.0,
                        c.deck_top + c.cube_s / 2 + 0.020],
                       device=device).expand(n, 3)
    place(scene.cube_b, tp + quat_apply(tq_live, loc), tq_live)
    min_upz = 1.0
    for _ in range(300):
        step(1)
        min_upz = min(min_upz, float(scene._tray_up_z()[0]))
    report("far-serve")
    print(f"[smoke] far-serve transient: min up_z={min_upz:+.4f} "
          f"(level band floor {math.cos(math.radians(c.level_tol_deg)):.4f})",
          flush=True)
    tz = float(scene.tray.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    fell = tz < c.mount_z - c.mount_z_tol - 0.004 \
        or float(scene._tray_up_z()[0]) < math.cos(math.radians(c.level_tol_deg))
    cb_z = float(scene.cube_b.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    shed = cb_z < 0.06  # cube ended on the ground, far below the 0.161 m deck ride
    tipped = min_upz < math.cos(math.radians(c.level_tol_deg))  # probe not vacuous
    s12, ok = judge()
    check("far-end serve: cube dropped near the tray's far end tips the LIVE "
          "balance out of the level band (readback transient) and the tray either "
          "capsizes or sheds the cube to the ground and re-rights — no level "
          "served state exists there; serve never latches; the mount credit "
          f"survives latched (0.35 -> {s12:.3f}), success never",
          tipped and (fell or shed) and not bool(scene._served_ever[0].any())
          and abs(s12 - c.w_mount) < 0.01 and not ok)

    # =========================== 13. reset clears the latches ===============================
    torch.manual_seed(101)
    env.reset()
    step(10)
    s, ok = judge()
    check("reset clears: fresh reset returns the score to ~0 (latches cleared)",
          s <= 0.02 and not ok)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.tray_poise")
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
    except Exception as exc:  # noqa: BLE001 — die fast, Kit teardown hangs
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({exc})", flush=True)
        os._exit(1)

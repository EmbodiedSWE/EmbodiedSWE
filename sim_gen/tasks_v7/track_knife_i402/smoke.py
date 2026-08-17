"""Smoke / rubric-REJECTION battery for KnifeStandScene (sim_gen task
`track_knife_i402`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — press-fit the cross-lap under a held wrench, then
lower the knife onto the live notch line — is the acceptance evidence that the rubric
ACCEPTS a correct outcome). Every teleport here CONSTRUCTS a wrong (or partial)
outcome and asserts the rubric REJECTS it; no probe in this battery ever reaches
success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — base panel standing, cap panel flat on the rack, knife on
                            the pedestal, everything still and finite, score ~0;
  3-4. randomization      — READBACK over seeded resets: base xy + free yaw vary
                            (max-pairwise over 3 seeds); rack xy/yaw, pedestal xy and
                            knife yaw vary; the cap panel rides its rack;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed-strategy analog — the SEED's whole strategy (waypoint-transport of the
                            knife) earns NOTHING here: the knife is pose-carried along
                            a smooth waypoint path at working height and judged at
                            every waypoint, then set down — score stays ~0 because the
                            knife's destination does not exist yet;
  7.  lone-panel perch    — the knife held level at seat height over the LONE base
                            panel's notch (cap still racked): knife_seated False, the
                            mesh-gated latch never fires, score ~0;
  8.  beside-park         — the cap panel upright BESIDE the base (6 cm off): neither
                            engaged nor meshed (centre-coincidence clause);
  9.  half-press          — the cap aligned in the slot but 4 cm proud: engaged latches
                            (0.20, positive control) yet meshed REFUSED (seat-z clause);
  10. build the stand     — settled cross-lap construct: meshed latches (score 0.45);
                            the empty stand alone is NOT success (knife still parked);
  11. rim-perch           — the knife laid across the two UN-notched top edges of the
                            meshed stand (the opposite diagonal, z ~ 0.1225): seated
                            False (band + off the notch line), score stays 0.45;
  12. skew half-perch     — one blade end in a notch, the other on a rim (~10.7 deg
                            tilt): the level clause REFUSES it;
  13. settle gate + band  — the perfect final pose at rim height is refused (z band);
                            the same pose at seat height while SLIDING at 0.4 m/s
                            latches knife credit (0.75) but success is REFUSED by the
                            settle gate;
  14. latched credit      — scattering the knife to the floor and yanking the cap out
                            of the joint does not evaporate the latched 0.75;
  15. rejection audit     — success() never True at ANY judged point;
  16. final no-NaN        — all task-object states finite; frames.npz saved.

Run (forge): python -u -m simgen_tasks.track_knife_i402.smoke --headless
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
import traceback

import numpy as np
import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

_wrap = scene_mod._wrap
_yaw_of = scene_mod._yaw_of

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.knife_stand")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.15, -0.85, 0.75)) + o),
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

    def loc(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        pa, pb, pk = loc(scene.panel_a), loc(scene.panel_b), loc(scene.knife)
        s, ok = judge()
        print(f"[smoke] {tag:16s} | A=({float(pa[0]):+.3f},{float(pa[1]):+.3f},"
              f"{float(pa[2]):.3f}) B=({float(pb[0]):+.3f},{float(pb[1]):+.3f},"
              f"{float(pb[2]):.3f}) K=({float(pk[0]):+.3f},{float(pk[1]):+.3f},"
              f"{float(pk[2]):.3f}) eng={bool(scene.engaged()[0])} "
              f"mesh={bool(scene.meshed()[0])} seat={bool(scene.knife_seated()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, x: float, y: float, z: float, quat=(1.0, 0.0, 0.0, 0.0),
              vel=(0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 7], st[:, 8], st[:, 9] = vel
        st[:, 0:3] += env.iscene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def qz(yaw: float):
        return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

    def qy(yaw: float, pitch: float):
        """Yaw about z, then pitch about the body's y-axis (tilts the x-axis)."""
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        # q = qz(yaw) * qy(pitch)
        return (cy * cp, -sy * sp, cy * sp, sy * cp)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=5)
    report("reset")
    step(90)
    report("settled")
    fin0 = all(bool(torch.isfinite(b.data.root_state_w).all())
               for b in (scene.panel_a, scene.panel_b, scene.knife))
    pa, pb, pk = loc(scene.panel_a), loc(scene.panel_b), loc(scene.knife)
    upa = float(scene_mod._up_z(scene.panel_a.data.root_quat_w)[0])
    check("settle: states finite, base panel STANDING upright, cap panel FLAT on the "
          "rack, knife on the pedestal, all still",
          fin0 and upa > 0.95 and float(pa[2]) < 0.01
          and abs(float(pb[2]) - c.b_lie_z) < 0.012
          and abs(float(pk[2]) - c.knife_lie_z) < 0.012
          and bool(scene._still()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (11, 12, 13):
        env.reset(seed=sd)
        step(2)
        pa = loc(scene.panel_a)
        ayaw = float(_yaw_of(scene.panel_a.data.root_quat_w)[0])
        pb = loc(scene.panel_b)
        ryaw = float(scene._rack_yaw[0])
        bx, by = float(scene._block_xy[0, 0]), float(scene._block_xy[0, 1])
        kyaw = float(_yaw_of(scene.knife.data.root_quat_w)[0])
        reads.append((float(pa[0]), float(pa[1]), ayaw,
                      float(pb[0]), float(pb[1]), ryaw, bx, by, kyaw))
        print(f"[smoke] seed {sd}: A=({reads[-1][0]:+.3f},{reads[-1][1]:+.3f},"
              f"{math.degrees(ayaw):+.0f}deg) B=({reads[-1][3]:+.3f},{reads[-1][4]:+.3f}) "
              f"rackyaw={math.degrees(ryaw):+.1f} block=({bx:+.3f},{by:+.3f}) "
              f"kyaw={math.degrees(kyaw):+.1f}", flush=True)
    arr = np.array(reads)

    def maxpair(col: int) -> float:
        v = arr[:, col]
        return float(max(abs(v[i] - v[j]) for i in range(3) for j in range(i + 1, 3)))

    in_band = all(abs(arr[i, 0] - c.a_pos[0]) <= c.a_jitter + 0.005
                  and abs(arr[i, 1] - c.a_pos[1]) <= c.a_jitter + 0.005 for i in range(3))
    check("randomization: base panel xy varies (max-pairwise > 4 mm) within the band, "
          "free yaw takes different values (> 20 deg spread)",
          (maxpair(0) > 0.004 or maxpair(1) > 0.004) and in_band
          and maxpair(2) > math.radians(20.0))
    check("randomization: rack/cap xy varies, rack yaw varies, pedestal xy varies, "
          "knife yaw varies",
          (maxpair(3) > 0.004 or maxpair(4) > 0.004) and maxpair(5) > 0.01
          and (maxpair(6) > 0.004 or maxpair(7) > 0.004) and maxpair(8) > 0.02)

    # =========================== 5. null policy =============================================
    env.reset(seed=0)
    step(240)
    s, ok = judge()
    report("null policy")
    check("null policy: 240 idle steps -> score ~0, no success", s <= 0.02 and not ok)

    # =========================== 6. seed-strategy analog ====================================
    # The seed grades dense waypoint-tracking of an already-held knife. Reproduce that
    # exact strategy: carry the knife (pose writes) along a smooth elevated path across
    # the workspace and judge at every waypoint; then set it down gently. It earns
    # NOTHING because the knife's destination does not exist until the stand is built.
    pk0 = loc(scene.knife)
    kyaw0 = float(_yaw_of(scene.knife.data.root_quat_w)[0])
    smax = 0.0
    for t in np.linspace(0.0, 1.0, 9):
        x = float(pk0[0]) + (0.30 - float(pk0[0])) * t
        y = float(pk0[1]) + (-0.10 - float(pk0[1])) * t
        z = 0.033 + 0.12 * math.sin(math.pi * float(t))  # arc up to ~0.15 and back
        place(scene.knife, x, y, max(z, 0.033), qz(kyaw0))
        step(2)
        s, ok = judge()
        smax = max(smax, s)
    place(scene.knife, 0.30, -0.10, 0.006, qz(kyaw0))  # set down on open floor
    step(90)
    s, ok = judge()
    report("seed analog")
    check("seed-strategy analog: waypoint-transport of the knife (the seed's whole "
          "reward) earns ~0 at every judged waypoint and after set-down",
          smax <= 0.02 and s <= 0.02 and not ok)

    # =========================== 7. lone-panel perch ========================================
    env.reset(seed=0)
    step(60)
    pa = loc(scene.panel_a)
    ayaw = float(_yaw_of(scene.panel_a.data.root_quat_w)[0])
    ax = math.cos(ayaw), math.sin(ayaw)
    nax, nay = float(pa[0]) + c.notch_r * ax[0], float(pa[1]) + c.notch_r * ax[1]
    # knife LEVEL at seat height, crossing the lone base panel's notch (cap racked)
    place(scene.knife, nax, nay, c.knife_rest_z, qz(ayaw + math.pi / 2), vel=(0, 0, 0.2))
    step(2)
    s_now, ok = judge()
    seated_now = bool(scene.knife_seated()[0])
    step(120)  # it falls off the 8 mm edge / tumbles — never credited
    s, ok2 = judge()
    report("lone-panel")
    check("lone-panel perch: knife level at seat height over the LONE base panel's "
          "notch is NOT seated (cap panel still racked -> live notch line elsewhere); "
          "no latch, score ~0",
          not seated_now and s_now <= 0.02 and s <= 0.02 and not ok and not ok2)

    # =========================== 8. beside-park =============================================
    env.reset(seed=0)
    step(60)
    pa = loc(scene.panel_a)
    ayaw = float(_yaw_of(scene.panel_a.data.root_quat_w)[0])
    px = math.cos(ayaw + math.pi / 2), math.sin(ayaw + math.pi / 2)
    place(scene.panel_b, float(pa[0]) + 0.06 * px[0], float(pa[1]) + 0.06 * px[1],
          0.002, qz(ayaw + math.pi / 2))
    step(2)
    eng, mesh = bool(scene.engaged()[0]), bool(scene.meshed()[0])
    s, ok = judge()
    report("beside-park")
    check("beside-park: cap panel upright 6 cm BESIDE the base, bottom at the floor — "
          "neither engaged nor meshed (centre-coincidence clause), score ~0",
          not eng and not mesh and s <= 0.02 and not ok)

    # =========================== 9. half-press ==============================================
    # (fresh reset: the beside-park cap is mid-topple)
    env.reset(seed=0)
    step(60)
    pa = loc(scene.panel_a)
    ayaw = float(_yaw_of(scene.panel_a.data.root_quat_w)[0])
    place(scene.panel_b, float(pa[0]), float(pa[1]), 0.040, qz(ayaw + math.pi / 2))
    step(2)
    eng, mesh = bool(scene.engaged()[0]), bool(scene.meshed()[0])
    s, ok = judge()
    report("half-press")
    check("half-press: cap aligned in the slot 4 cm proud — engaged latches (0.20 "
          "positive control) but meshed REFUSED (seat-z clause), no success",
          eng and not mesh and abs(s - c.w_engage) < 0.01 and not ok)

    # =========================== 10. build the stand ========================================
    place(scene.panel_b, float(pa[0]), float(pa[1]), 0.0015, qz(ayaw + math.pi / 2))
    step(150)
    mesh = bool(scene.meshed()[0])
    s, ok = judge()
    report("stand built")
    check("built stand: settled cross-lap construct is MESHED, mesh latches "
          "(score 0.45 = engage + mesh), yet the EMPTY stand is not success",
          mesh and abs(s - (c.w_engage + c.w_mesh)) < 0.01 and not ok)

    # =========================== 11. rim-perch ==============================================
    ayaw = float(_yaw_of(scene.panel_a.data.root_quat_w)[0])
    paj = loc(scene.panel_a)
    diag = ayaw + 3 * math.pi / 4  # the UN-notched diagonal (both -x wings)
    place(scene.knife, float(paj[0]), float(paj[1]), c.height + c.blade_t / 2 + 0.001,
          qz(diag))
    step(150)
    seat = bool(scene.knife_seated()[0])
    pk = loc(scene.knife)
    s, ok = judge()
    report("rim-perch")
    check("rim-perch: knife settled across the two UN-notched top edges "
          f"(z={float(pk[2]):.3f} vs band top {c.kz_hi}) — NOT seated, no knife "
          "latch, score stays 0.45",
          not seat and abs(s - (c.w_engage + c.w_mesh)) < 0.01 and not ok)

    # =========================== 12. skew half-perch ========================================
    na, nb = scene.notch_points()
    midx = (float(na[0, 0]) + float(nb[0, 0])) / 2
    midy = (float(na[0, 1]) + float(nb[0, 1])) / 2
    lyaw = math.atan2(float(nb[0, 1] - na[0, 1]), float(nb[0, 0] - na[0, 0]))
    tilt = math.radians(10.7)  # one end in a notch, the other on a rim
    place(scene.knife, midx, midy, (c.knife_rest_z + c.height + c.blade_t / 2) / 2,
          qy(lyaw, tilt), vel=(0, 0, 0.05))
    step(2)
    seat = bool(scene.knife_seated()[0])
    s, ok = judge()
    report("skew-perch")
    check("skew half-perch: blade tilted 10.7 deg (one end at notch depth, one on the "
          "rim) — the level clause (8 deg) REFUSES it",
          not seat and abs(s - (c.w_engage + c.w_mesh)) < 0.01 and not ok)

    # =========================== 13. settle gate + z band ===================================
    # (a) perfect xy/yaw at RIM height, moving: band refuses
    place(scene.knife, midx, midy, c.height + c.blade_t / 2 + 0.003, qz(lyaw),
          vel=(0.4, 0.0, 0.0))
    step(1)
    seat_hi = bool(scene.knife_seated()[0])
    # (b) the same pose AT seat height while SLIDING 0.4 m/s: pose clauses pass
    # (positive control -> knife credit latches), success REFUSED by the settle gate
    place(scene.knife, midx, midy, c.knife_rest_z + 0.001, qz(lyaw), vel=(0.4, 0.0, 0.0))
    step(1)
    seat_lo = bool(scene.knife_seated()[0])
    s, ok = judge()
    report("moving-perfect")
    moving_refused = not ok and seat_lo and not seat_hi
    check("z band + settle gate: rim-height pose refused by the band; the same pose "
          "at seat height passes the pose clauses (latching 0.30 -> score 0.75) but "
          "the SLIDING assembly is refused by the settle gate",
          moving_refused and abs(s - 0.75) < 0.01)

    # =========================== 14. latched credit survives ================================
    place(scene.knife, 0.60, -0.35, 0.033, qz(0.3))  # scatter the knife to the floor
    pa = loc(scene.panel_a)
    ayaw = float(_yaw_of(scene.panel_a.data.root_quat_w)[0])
    px = math.cos(ayaw + math.pi / 2), math.sin(ayaw + math.pi / 2)
    place(scene.panel_b, float(pa[0]) - 0.30 * px[0], float(pa[1]) - 0.30 * px[1],
          c.height / 2 + 0.02, qz(ayaw + math.pi / 2))  # yank the cap out of the joint
    step(150)
    s, ok = judge()
    report("disassembled")
    check("latched credit: knife scattered to the floor and cap yanked out of the "
          "joint — the latched 0.75 does not evaporate, success False",
          abs(s - 0.75) < 0.01 and not ok)

    # =========================== 15-16. audit + finite ======================================
    check("rejection audit: success() never True at ANY judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all())
              for b in (scene.panel_a, scene.panel_b, scene.knife))
    check("final: all task-object states finite (no NaN/Inf)", fin)

    # --- verdict ---
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.knife_stand")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(1 for _, okc in checks if okc)
    if n_pass == len(checks):
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
        code = 0
    else:
        for name, okc in checks:
            if not okc:
                print(f"[smoke] FAILED CHECK: {name}", flush=True)
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        code = 1
    threading.Timer(10.0, lambda: os._exit(code)).start()
    os._exit(code)


try:
    main()
except BaseException:  # noqa: BLE001 - report, then hard-exit (Kit teardown hangs)
    traceback.print_exc()
    print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
    threading.Timer(10.0, lambda: os._exit(2)).start()
    os._exit(2)

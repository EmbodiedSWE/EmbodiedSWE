"""Smoke / rubric-REJECTION battery for SwingArrestScene (sim_gen task
`draw_svg_i49`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — arrest each swinging pendulum with an arrestor
block, then take the blocks away — is the acceptance evidence that the rubric
ACCEPTS a correct outcome). Every teleport here is instrumentation that CONSTRUCTS
a wrong (or partial) settled outcome and asserts the rubric REJECTS it; no probe
in this battery ever reaches success(), and a final audit asserts exactly that.

Seed-strategy check: N/A — the seed (maniskill/draw_svg) drags a free marker cube
along a prescribed path; here the red cube is a pendulum BOB pinned to a 1-DoF
rod, there is no path, and "dragging" is inexpressible. In its place the battery
attacks this task's own cheat surface:

  1-2. settle/premise    — states finite; both pendulums VERIFIABLY SWINGING at
                           reset (amplitude readback), blocks staged clear,
                           gantries home; score ~0, no success.
  3-4. randomization     — READBACK over 10 seeded resets: release angle
                           magnitudes AND signs vary per pendulum, the long/short
                           gantries swap sides, gantry xy jitters, the staging
                           permutation and block yaws vary; invariants hold
                           (releases inside the 50-75 deg band, blocks clear).
  5.  null policy        — 360 idle steps: the pendulums are STILL SWINGING at
                           full amplitude, still() never opened, score ~0.
  6.  turning points     — free swing sampled every step: the rods' angular speed
                           dips below the stillness gate at swing extremes, yet
                           the sustained counter never matures, still() never
                           fires, no arrest latches ("judged only at sustained
                           rest" is real).
  7.  wedged, not plumb  — a block placed so the falling bob comes to rest
                           leaning on the post at ~15 deg: the rod is sustained-
                           STILL but NOT plumb -> no arrest latch, NOT success
                           (stopping the pendulum anywhere does not count).
  8.  clause C (parked)  — both rods plumb and still, but one block parked
                           INSIDE the exclusion radius (touching nothing):
                           NOT success, score capped.
  9.  clause C (propped) — the solve's own arrest end-state LEFT IN PLACE: bob
                           resting against the post face at ~1 deg, plumb and
                           still — the block inside the radius voids it, NOT
                           success (the blocks must actually be taken away).
  10. clause D (dragged) — a gantry (with its rod, plumb and still) teleported
                           0.12 m from its spawn pose, blocks clear: NOT success
                           (moving the gantry to cheat the geometry is rejected).
  11. latched credit     — pendulum 0 brought to rest (blocks clear) latches
                           arrest+clean credit; re-exciting it to a 45 deg swing
                           leaves the latched score unchanged; never success
                           (pendulum 1 swings throughout).
  12. rejection audit    — success() was never True at ANY judged point.
  13. final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.draw_svg_i49.smoke --headless
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

PARK_XY = ((-0.45, -0.35), (0.45, -0.35))  # bench corners far outside the exclusion


def main() -> None:
    from isaaclab.utils.math import quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.swing_arrest")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    ey = torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.05, -1.05, 0.95)) + o),
                                tuple(np.array((0.0, 0.0, 0.30)) + o),
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

    def report(tag: str) -> None:
        s, ok = judge()
        a = scene.angle()[0]
        print(f"[smoke] {tag:18s} | th=({math.degrees(float(a[0])):+7.2f},"
              f"{math.degrees(float(a[1])):+7.2f}) deg "
              f"arrest={scene.arrest_latch[0].tolist()} clean={scene.clean_latch[0].tolist()} "
              f"clear={scene.blocks_clear()[0].tolist()} home={bool(scene.gantries_home()[0])} "
              f"still={bool(scene.still()[0])} success={ok} score={s:.3f} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    # ----- probe placement (instrumentation, not a solution) --------------------------------
    def write_rod(i: int, theta_deg: float) -> None:
        """Teleport rod i to a signed angle about its LIVE pivot, zero velocity."""
        g = scene.gantries[i]
        gq, gp = g.data.root_quat_w, g.data.root_pos_w
        hp = c.pend_specs[i][1]
        piv = gp + quat_apply(gq, torch.tensor([0.0, 0.0, hp], device=device).expand(n, 3))
        th = torch.full((n,), math.radians(theta_deg), device=device)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = piv
        st[:, 3:7] = scene_mod._qmul(gq, scene_mod._qx(th))
        scene.rods[i].write_root_state_to_sim(st, all_ids)

    def place_block(j: int, i: int, center_y_local: float) -> None:
        """Stand block j on the bench at a gantry-i-local +y offset, front face
        (local -y) toward the gantry."""
        g = scene.gantries[i]
        gq = g.data.root_quat_w
        ydir = quat_apply(gq, ey)
        xy = g.data.root_pos_w[:, :2] + center_y_local * ydir[:, :2]
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = xy
        st[:, 2] = scene.env_origins[:, 2] + c.bench_top + 0.003
        st[:, 3:7] = gq
        scene.blocks[j].write_root_state_to_sim(st, all_ids)

    def park_block(j: int, xy=None) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = torch.tensor(xy or PARK_XY[j], device=device).expand(n, 2)
        st[:, 2] = c.bench_top + 0.003
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        scene.blocks[j].write_root_state_to_sim(st, all_ids)

    def rod_settled(i: int, max_rounds: int = 20) -> bool:
        """REAL physics steps until rod i is sustained-still (or budget ends)."""
        for _ in range(max_rounds):
            step(30)
            if int(scene.rod_still_count[0, i]) >= c.still_steps:
                return True
        return False

    def settle_all(max_rounds: int = 24) -> bool:
        for _ in range(max_rounds):
            step(30)
            if bool(scene.still()[0]):
                return True
        return False

    def swing_amp(steps: int = 160) -> torch.Tensor:
        """Max |angle| per pendulum over `steps` real steps (swing readback)."""
        amp = torch.zeros(n, 2, device=device)
        for _ in range(steps):
            step(1)
            amp = torch.maximum(amp, scene.angle().abs())
        return amp

    def gantry_xy(i: int) -> tuple[float, float]:
        p = (scene.gantries[i].data.root_pos_w - scene.env_origins)[0]
        return float(p[0]), float(p[1])

    def block_yaw(j: int) -> float:
        w, x, y, z = (float(v) for v in scene.blocks[j].data.root_quat_w[0])
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def finite_all() -> bool:
        ok = True
        for b in scene.gantries + scene.rods + scene.blocks:
            ok = ok and bool(torch.isfinite(b.data.root_state_w).all())
        return ok

    # =========================== 1-2. settle / premise / no-NaN =============================
    env.reset(seed=11)
    a0 = scene.angle()[0].abs()
    amp = swing_amp(180)  # ~1.5 periods
    report("reset-swinging")
    s, ok = judge()
    check("premise: states finite; BOTH pendulums verifiably swinging at reset "
          f"(release readback {math.degrees(float(a0[0])):.0f}/"
          f"{math.degrees(float(a0[1])):.0f} deg, observed amplitudes "
          f"{math.degrees(float(amp[0, 0])):.0f}/{math.degrees(float(amp[0, 1])):.0f} deg); "
          "blocks staged clear and gantries home",
          finite_all() and bool((amp[0] > math.radians(25.0)).all())
          and bool(scene.blocks_clear()[0].all()) and bool(scene.gantries_home()[0]))
    check("reset: score ~0 (<= 0.02), no success, still() closed",
          s <= 0.02 and not ok and not bool(scene.still()[0]))

    # =========================== 3-4. randomization is real =================================
    reads = []
    inv_ok = True
    for sd in (21, 22, 23, 24, 25, 26, 27, 28, 29, 30):
        env.reset(seed=sd)
        step(2)
        th = scene.angle()[0]
        gx0, gy0 = gantry_xy(0)
        b0x = float((scene.blocks[0].data.root_pos_w - scene.env_origins)[0, 0])
        reads.append((float(th[0]), float(th[1]), gx0, gy0, b0x, block_yaw(0)))
        inv_ok = inv_ok and bool(scene.blocks_clear()[0].all())
        inv_ok = inv_ok and bool(
            (th.abs() > math.radians(c.rel_min_deg - 5.0)).all()
            and (th.abs() < math.radians(c.rel_max_deg + 5.0)).all())
    arr = np.array(reads)
    print(f"[smoke] randomization readback (th0, th1, gantry0_x, gantry0_y, "
          f"block0_x, block0_yaw):\n{arr.round(3)}", flush=True)
    check("randomization: release angles vary in magnitude AND sign for both "
          "pendulums, and the long gantry appears on both bench sides (side swap)",
          arr[:, 0].ptp() > 0.15 and arr[:, 1].ptp() > 0.15
          and len(set(np.sign(arr[:, 0]))) == 2 and len(set(np.sign(arr[:, 1]))) == 2
          and len(set(np.sign(arr[:, 2]))) == 2)
    check("randomization: gantry xy jitters, the staging permutation and block "
          "yaws vary; invariants hold (releases inside the band, blocks clear "
          "of both gantries)",
          (np.abs(arr[:, 2]) - c.frame_x).ptp() > 0.008 and arr[:, 3].ptp() > 0.008
          and len(set(np.sign(arr[:, 4]))) == 2 and arr[:, 5].ptp() > 0.5 and inv_ok)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(200)
    amp = swing_amp(160)
    report("null-policy")
    s, ok = judge()
    check("null policy: after 360 idle steps both pendulums are STILL SWINGING "
          f"(amplitudes {math.degrees(float(amp[0, 0])):.0f}/"
          f"{math.degrees(float(amp[0, 1])):.0f} deg), still() never opened, "
          "score ~0, no success",
          bool((amp[0] > math.radians(25.0)).all()) and s <= 0.02 and not ok
          and not bool(scene.still()[0]))

    # =========================== 6. turning points do not count =============================
    # A swinging rod's angular speed dips below the stillness gate for an instant
    # at every swing extreme. Sample EVERY step: sub-gate instants must be
    # witnessed, yet the sustained counter must never mature and nothing latches.
    env.reset(seed=32)
    slow_hits, max_cnt, any_still = 0, 0, False
    for _ in range(300):
        step(1)
        w = scene.rod_avel()[0]
        slow_hits += int(bool((w < c.rod_still_avel).any()))
        max_cnt = max(max_cnt, int(scene.rod_still_count[0].max()))
        any_still = any_still or bool(scene.still()[0])
        judge()
    report("turning-points")
    check("turning points: instantaneous sub-gate rod speed witnessed "
          f"({slow_hits} sampled instants) while swinging, yet the sustained "
          f"counter peaked at {max_cnt} < {c.still_steps} steps — still() never "
          "fired and no arrest latched (judged only at SUSTAINED rest)",
          slow_hits > 0 and max_cnt < c.still_steps and not any_still
          and float(scene.arrest_latch[0].sum()) == 0.0)

    # =========================== 7. wedged still, but not plumb =============================
    # Stop the long pendulum ~15 deg from vertical: block placed so the post's
    # far face catches the falling bob's inner face. Rod becomes sustained-still
    # yet NOT plumb: stopping the swing anywhere must earn nothing.
    env.reset(seed=33)
    park_block(1)
    ln = c.pend_specs[0][2]
    th_rest = math.radians(15.0)
    y_face = ln * math.sin(th_rest) - (c.bob_size / 2) * math.cos(th_rest)
    place_block(0, 0, y_face + (c.slab_y / 2 - c.block_post_w))  # post far face at y_face
    step(10)
    write_rod(0, 25.0)  # falls ~10 deg onto the post's far face, e=0
    ok_rod = rod_settled(0)
    ang = math.degrees(abs(float(scene.angle()[0, 0])))
    report("wedged-not-plumb")
    s, ok = judge()
    check("wedged: the long rod is sustained-STILL leaning on the post at "
          f"{ang:.1f} deg (> plumb tolerance {c.plumb_tol_deg:.0f} deg) — "
          "plumb() False, NO arrest latch for it, NOT success",
          ok_rod and ang > c.plumb_tol_deg + 2.0
          and not bool(scene.plumb()[0, 0])
          and float(scene.arrest_latch[0, 0]) == 0.0 and not ok)

    # =========================== 8. clause C: block parked inside the radius ================
    env.reset(seed=34)
    write_rod(0, 0.0)
    write_rod(1, 0.0)
    park_block(1)
    g0 = scene.gantries[0].data.root_pos_w[0]
    park_block(0, (float(g0[0] - scene.env_origins[0, 0]),
                   float(g0[1] - scene.env_origins[0, 1]) - 0.20))  # 0.20 m < exclusion_r
    settled = settle_all()
    d0 = float((scene.blocks[0].data.root_pos_w[0, :2] - g0[:2]).norm())
    report("parked-inside")
    s, ok = judge()
    check("clause C (parked): both rods plumb and everything sustained-still, but "
          f"one block rests {d0:.2f} m (< {c.exclusion_r} m) from a gantry — "
          "NOT success, score capped at 0.70",
          settled and bool(scene.plumb()[0].all()) and d0 < c.exclusion_r
          and not bool(scene.blocks_clear()[0].all()) and not ok and s <= 0.70)

    # =========================== 9. clause C: bob left propped at plumb =====================
    # The solve's own arrest end-state, NOT followed by extraction: block at the
    # strike gap, bob resting against the post face ~1 deg from vertical. Plumb,
    # still — and worthless while the block stays.
    env.reset(seed=35)
    park_block(1)
    write_rod(1, 0.0)
    place_block(0, 0, c.bob_size / 2 + c.strike_gap + c.slab_y / 2)
    step(10)
    write_rod(0, -3.0)  # swings through plumb into the post face, e=0 stop
    settled = settle_all()
    report("propped-at-plumb")
    s, ok = judge()
    check("clause C (propped): bob resting against the post face — plumb, "
          "sustained-still, gantries home — yet the block inside the exclusion "
          "radius voids it: NOT success, score capped",
          settled and bool(scene.plumb()[0].all()) and bool(scene.gantries_home()[0])
          and not bool(scene.blocks_clear()[0].all()) and not ok and s <= 0.70)

    # =========================== 10. clause D: gantry dragged from home =====================
    env.reset(seed=36)
    park_block(0)
    park_block(1)
    write_rod(0, 0.0)
    write_rod(1, 0.0)
    step(60)
    # drag gantry 0 (with its rod, keeping the pair) 0.12 m toward the bench front
    off = torch.tensor([0.0, -0.12, 0.0], device=device).expand(n, 3)
    for body in (scene.gantries[0], scene.rods[0]):
        st = body.data.root_state_w.clone()
        st[:, 0:3] += off
        st[:, 7:13] = 0.0
        body.write_root_state_to_sim(st, all_ids)
    settled = settle_all()
    report("gantry-dragged")
    s, ok = judge()
    check("clause D (dragged): both rods plumb and still, blocks parked clear — "
          "but a gantry sits 0.12 m (> 0.05 m tol) from its spawn pose: "
          "gantries_home False, NOT success",
          settled and bool(scene.plumb()[0].all())
          and bool(scene.blocks_clear()[0].all())
          and not bool(scene.gantries_home()[0]) and not ok and s <= 0.70)

    # =========================== 11. latched credit survives regression =====================
    env.reset(seed=37)
    park_block(0)
    park_block(1)
    write_rod(0, 0.0)  # pendulum 0 at rest, blocks clear -> arrest+clean latch
    for _ in range(20):
        step(30)
        if float(scene.clean_latch[0, 0]) > 0.0:
            break
    s1, _ = judge()
    write_rod(0, 45.0)  # re-excite: a fresh 45 deg swing
    step(150)
    s2, ok = judge()
    amp = swing_amp(120)
    report("latch-regressed")
    check("latched credit: resting pendulum 0 with blocks clear latches "
          f"arrest+clean (score {s1:.2f} >= 0.30); re-exciting it to a "
          f"{math.degrees(float(amp[0, 0])):.0f} deg swing leaves the latched "
          f"score unchanged ({s2:.2f}); never success (pendulum 1 swings "
          "throughout)",
          s1 >= 0.30 and abs(s2 - s1) < 1e-6
          and float(amp[0, 0]) > math.radians(25.0) and not ok and not ever_success[0])

    # =========================== 12-13. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.swing_arrest")
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
    main()

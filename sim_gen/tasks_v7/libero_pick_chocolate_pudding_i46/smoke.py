"""Smoke / rubric-REJECTION battery for PuddingSiftScene (sim_gen task
`libero_pick_chocolate_pudding_i46`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — carry the sealed bin over the grate, tilt-pour
the mixed batch through the port, let the holes sort it, deliver the retained
pudding ball, park the bin — is the acceptance evidence that the rubric ACCEPTS a
correct outcome). Every teleport here is instrumentation that CONSTRUCTS a wrong
(or partial) outcome and asserts the rubric REJECTS it; no probe in this battery
ever reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: the pudding ball and every
                            PRESENT pearl sealed INSIDE the bin (readback), the bin
                            parked, absent pearls in the depot; score ~0, no success;
  3-4. randomization      — READBACK over 6 seeded resets: hopper xy + yaw move and
                            the dish side takes BOTH signs; bin xy + free yaw, dish
                            xy, and the PRESENT pearl count all move;
  5.  null policy         — 240 idle steps -> score ~0, no success, and the bin
                            keeps its contents sealed (nothing leaks while level);
  6.  SEED strategy       — the seed's whole plan ("pick the chocolate pudding, put
                            it in the open goal vessel, ignore the rest") = the
                            pudding ball alone delivered to the dish: pearls still
                            sealed in the bin -> NOT success, score <= 0.32;
  7.  filter honesty A    — the pudding ball dropped from above DIRECTLY OVER an
                            open grate hole is RETAINED on the grate (42 mm ball vs
                            26 mm hole — readback above the grate), NOT success;
  8.  filter honesty B    — a pearl dropped over the same hole PASSES into the
                            basin (18 mm pearl, 8 mm slack — readback below the
                            grate): the filtration route is real, still NOT success;
  9.  dump cheat          — the whole batch dumped into the DISH (pudding + every
                            pearl): pearls in the dish are contamination, not
                            delivery -> NOT success, score <= 0.32;
  10. near miss           — the full sift done EXCEPT delivery: every present pearl
                            genuinely through the holes, the pudding retained on the
                            grate, the bin parked — score ~0.60, NOT success (a
                            pudding ball left on the grate is not done);
  11. bin not parked      — pearls sifted + pudding delivered to the dish but the
                            bin set down ON THE GRATE (readback z at grate height):
                            the bin_parked clause alone rejects success;
  12. latched credit      — a delivered pearl teleported back OUT of the basin onto
                            the floor: the latched sift credit survives unchanged
                            (and still no success);
  13. monotonicity        — sifting a second pearl through a hole latches strictly
                            more credit than the first alone;
  14. rejection audit     — success() was never True at ANY judged point;
  15. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.libero_pick_chocolate_pudding_i46.smoke --headless
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

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.pudding_sift")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.55, -1.15, 0.95)) + o),
                                tuple(np.array((0.55, 0.00, 0.15)) + o),
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

    def present0() -> torch.Tensor:
        return scene.present[0]

    def basin_count() -> int:
        return int((scene.pearls_in_basin()[0] & present0()).sum())

    def bin_count() -> int:
        return int((scene.pearls_in_bin()[0] & present0()).sum())

    def n_present() -> int:
        return int(present0().sum())

    def report(tag: str) -> None:
        s, ok = judge()
        print(f"[smoke] {tag:16s} | pearls {basin_count()}/{n_present()} in basin, "
              f"{bin_count()} in bin | pud_in_bin={bool(scene.pud_in_bin()[0])} "
              f"on_grate={bool(scene.pud_on_grate()[0])} "
              f"in_dish={bool(scene.pud_in_dish()[0])} "
              f"parked={bool(scene.bin_parked()[0])} settled={bool(scene.settled()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_world(body, x: float, y: float, z: float, settle_steps: int = 60) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL
        physics steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def hopper_world(lx: float, ly: float, lz: float) -> tuple[float, float, float]:
        """Hopper-local point -> env-local world (the hopper carries yaw)."""
        loc = torch.tensor([lx, ly, lz], device=device).expand(1, 3)
        w = scene.hopper.data.root_pos_w[0] + quat_apply(
            scene.hopper.data.root_quat_w[0:1], loc)[0] - scene.env_origins[0]
        return float(w[0]), float(w[1]), float(w[2])

    def dish_top(dx: float = 0.0, dy: float = 0.0) -> tuple[float, float, float]:
        w = (scene.dish.data.root_pos_w - scene.env_origins)[0]
        return float(w[0]) + dx, float(w[1]) + dy, float(w[2]) + 0.08

    # Distinct open holes (hopper-local xy) for multi-pearl sifting probes.
    hole_xy = [(c.hole_centers[i % 5], c.hole_centers[(i // 5) % 5]) for i in
               (0, 2, 4, 10, 12, 14, 20, 22, 24)]

    def sift_pearls(idx: list[int], rounds: int = 4) -> None:
        """Drop the listed pearls from free space directly over open holes; the
        GRATE does the passing (same honest mechanism as the solution's sift
        assist). Re-drop bouncers for a few rounds."""
        for r in range(rounds):
            stray = [j for j in idx if not bool(scene.pearls_in_basin()[0][j])]
            if not stray:
                return
            for k, j in enumerate(stray):
                hx, hy = hole_xy[k % len(hole_xy)]
                wx, wy, wz = hopper_world(hx, hy, c.grate_z1 + c.pearl_r + 0.015)
                st = torch.zeros(n, 13, device=device)
                st[:, 0], st[:, 1], st[:, 2] = wx, wy, wz
                st[:, 3] = 1.0
                st[:, 0:3] += scene.env_origins
                scene.pearls[j].write_root_state_to_sim(st, all_ids)
                step(45)
            step(60)

    def fin_all() -> bool:
        ok = (torch.isfinite(scene.hopper.data.root_state_w).all()
              and torch.isfinite(scene.bin.data.root_state_w).all()
              and torch.isfinite(scene.dish.data.root_state_w).all()
              and torch.isfinite(scene.pudding.data.root_state_w).all())
        for b in scene.pearls:
            ok = ok and torch.isfinite(b.data.root_state_w).all()
        return bool(ok)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    check("settle: states finite; the pudding ball and every present pearl sealed "
          "INSIDE the bin (readback), the bin parked, everything settled",
          fin_all() and bool(scene.pud_in_bin()[0]) and bin_count() == n_present()
          and bool(scene.bin_parked()[0]) and bool(scene.settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.03), no success", s <= 0.03 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(5)
        fp = (scene.hopper.data.root_pos_w - scene.env_origins)[0]
        fq = scene.hopper.data.root_quat_w[0]
        bp = (scene.bin.data.root_pos_w - scene.env_origins)[0]
        bq = scene.bin.data.root_quat_w[0]
        dp = (scene.dish.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(fp[0]), float(fp[1]),
                      2.0 * float(torch.atan2(fq[3], fq[0])),
                      float(scene.side[0]),
                      float(bp[0]), float(bp[1]),
                      2.0 * float(torch.atan2(bq[3], bq[0])),
                      float(dp[0]), float(dp[1]),
                      float(n_present())))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (fix_x, fix_y, fix_yaw, side, bin_x, "
          f"bin_y, bin_yaw, dish_x, dish_y, n_present):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: hopper xy + yaw vary AND the dish side takes both signs "
          "across seeded resets (readback)",
          spread[0] > 0.005 and spread[1] > 0.005 and spread[2] > 0.03
          and arr[:, 3].min() < -0.5 and arr[:, 3].max() > 0.5)
    check("randomization: bin xy + free yaw, dish xy, and the present pearl count "
          "vary across seeded resets (readback)",
          spread[4] > 0.005 and spread[5] > 0.01 and spread[6] > 0.5
          and spread[7] > 0.005 and spread[8] > 0.01 and spread[9] >= 1.0)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0, no success after 240 idle steps, and the level bin "
          "keeps its contents sealed (readback)",
          s <= 0.03 and not ok and bool(scene.pud_in_bin()[0])
          and bin_count() == n_present())

    # =========================== 6. SEED strategy ===========================================
    # The seed's whole plan is "pick the chocolate pudding and put it in the open
    # goal vessel, ignore the rest". Here that end state — the pudding ball alone
    # delivered to the dish — must be worth <= 0.32: the judged set is the WHOLE
    # batch and the pearls are still sealed in the bin.
    env.reset(seed=41)
    step(30)
    dx, dy, dz = dish_top()
    place_world(scene.pudding, dx, dy, dz, settle_steps=120)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy (pudding alone delivered to the dish): pudding in_dish by "
          "readback, pearls still sealed in the bin -> NOT success, score <= 0.32",
          bool(scene.pud_in_dish()[0]) and bin_count() == n_present()
          and not ok and s <= 0.32)

    # =========================== 7-8. the filter is honest ==================================
    # A: the 42 mm pudding ball dropped DIRECTLY OVER an open 26 mm hole must be
    # retained ON the grate. B: an 18 mm pearl dropped over the same hole must PASS
    # into the basin. Same drop, opposite outcomes — that is the whole task.
    env.reset(seed=51)
    step(30)
    hx, hy = c.hole_centers[2], c.hole_centers[2]  # centre hole
    wx, wy, wz = hopper_world(hx, hy, c.grate_z1 + c.pud_r + 0.020)
    place_world(scene.pudding, wx, wy, wz, settle_steps=120)
    report("pud-on-hole")
    pud_hl_z = float(scene._local(scene.hopper, scene.pudding.data.root_pos_w)[0][2])
    s, ok = judge()
    check("filter honesty A: the pudding ball dropped directly over an open hole is "
          "RETAINED above the grate (readback z), pud_on_grate True, NOT success",
          pud_hl_z > c.grate_z1 and bool(scene.pud_on_grate()[0]) and not ok)
    j0 = int(present0().nonzero().flatten()[0])
    wx, wy, wz = hopper_world(hx + 2 * (c.hole_w + c.bar_w), hy,
                              c.grate_z1 + c.pearl_r + 0.015)
    place_world(scene.pearls[j0], wx, wy, wz, settle_steps=120)
    report("pearl-thru-hole")
    pearl_hl_z = float(scene._local(scene.hopper,
                                    scene.pearls[j0].data.root_pos_w)[0][2])
    s, ok = judge()
    check("filter honesty B: a pearl dropped over a hole PASSES through into the "
          "basin (readback z below the grate underside, in_basin True), NOT success",
          pearl_hl_z < c.grate_z0 and bool(scene.pearls_in_basin()[0][j0]) and not ok)

    # =========================== 9. dump cheat ==============================================
    # Dump the WHOLE batch into the dish ("empty the bin into the goal vessel and
    # call it done"): pearls in the dish are contamination, not delivery.
    env.reset(seed=61)
    step(30)
    dx, dy, dz = dish_top()
    place_world(scene.pudding, dx, dy, dz, settle_steps=90)
    offs = [(-0.03, -0.03), (0.03, -0.03), (-0.03, 0.03), (0.03, 0.03),
            (0.0, -0.04), (0.0, 0.04)]
    for k, j in enumerate(present0().nonzero().flatten().tolist()):
        ox, oy = offs[k % len(offs)]
        dx, dy, dz = dish_top(ox, oy)
        place_world(scene.pearls[j], dx, dy, dz, settle_steps=45)
    step(90)
    report("dump-cheat")
    s, ok = judge()
    check("dump cheat (whole batch dumped into the dish): pudding delivered but "
          "every pearl is contamination (0 in basin by readback) -> NOT success, "
          "score <= 0.32",
          bool(scene.pud_in_dish()[0]) and basin_count() == 0 and not ok and s <= 0.32)

    # =========================== 10. near miss ==============================================
    # Everything EXCEPT delivery: every present pearl genuinely dropped through the
    # holes, the pudding retained on the grate, the bin parked where it spawned.
    env.reset(seed=71)
    step(30)
    sift_pearls(present0().nonzero().flatten().tolist())
    wx, wy, wz = hopper_world(0.0, 0.0, c.grate_z1 + c.pud_r + 0.020)
    place_world(scene.pudding, wx, wy, wz, settle_steps=120)
    report("near-miss")
    s_near, ok = judge()
    check("near miss (full sift, no delivery): all pearls through the holes + "
          "pudding retained on the grate + bin parked -> score ~0.60, NOT success",
          basin_count() == n_present() and bool(scene.pud_on_grate()[0])
          and bool(scene.bin_parked()[0]) and 0.55 <= s_near <= 0.65 and not ok)

    # =========================== 11. bin not parked =========================================
    # Complete the delivery too — but leave the BIN on the grate. The bin_parked
    # clause alone must reject success.
    dx, dy, dz = dish_top()
    place_world(scene.pudding, dx, dy, dz, settle_steps=120)
    wx, wy, wz = hopper_world(0.0, 0.0, c.grate_z1 + 0.020)
    place_world(scene.bin, wx, wy, wz, settle_steps=150)
    report("bin-on-grate")
    bin_z = float((scene.bin.data.root_pos_w - scene.env_origins)[0][2])
    s_full, ok = judge()
    check("bin not parked: pearls sifted + pudding in the dish but the bin set down "
          "ON THE GRATE (readback z at grate height) — bin_parked False alone "
          "rejects success",
          basin_count() == n_present() and bool(scene.pud_in_dish()[0])
          and bin_z > c.grate_z0 - 0.02 and not bool(scene.bin_parked()[0]) and not ok)

    # =========================== 12. latched credit survives regression =====================
    j1 = int(present0().nonzero().flatten()[0])
    place_world(scene.pearls[j1], 0.90, 0.90, c.pearl_r + 0.002, settle_steps=60)
    report("pearl-removed")
    s_out, ok = judge()
    check("latched credit: teleporting a delivered pearl back OUT of the basin onto "
          "the floor leaves the latched sift + delivery score unchanged (and still "
          "no success)",
          abs(s_out - s_full) < 0.02 and not ok)

    # =========================== 13. monotonicity ===========================================
    env.reset(seed=81)
    step(10)
    idx = present0().nonzero().flatten().tolist()
    sift_pearls([idx[0]])
    s_one = float(scene.score()[0])
    sift_pearls([idx[1]])
    s_two = float(scene.score()[0])
    check("monotonicity: sifting a second pearl through the grate latches strictly "
          f"more credit than one alone ({s_one:.3f} < {s_two:.3f})",
          s_one + 0.05 < s_two)
    judge()

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", fin_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.pudding_sift")
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
    except BaseException:  # noqa: BLE001 — a hung Kit teardown eats the traceback otherwise
        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(2)

"""Smoke / rubric-REJECTION battery for BeamHoistScene (sim_gen task
`lift_numbered_block_i153`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — seat the numbered block, load both counterweights,
let the hinge hoist — is the acceptance evidence that the rubric ACCEPTS a correct
outcome; it passes on seeds 0/1/2). Every teleport here is instrumentation that
CONSTRUCTS a wrong (or partial) outcome as a physically-stepped state and asserts the
rubric REJECTS it. No probe in this battery ever reaches success(), and a final audit
asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: beam pressed cradle-down on
                            its stop, blocks resting on their floor slots, the
                            target's placard mounted, the other two parked; score ~0;
  3-4. randomization      — READBACK over 14 seeded resets: all THREE target numbers
                            occur and the mounted placard tracks the target while the
                            others park; the block scatter (permuted slots + jitter)
                            varies per seed;
  5.  null policy         — 240 idle steps -> score ~0, beam stays down, no success;
  6.  seed strategy fails — rlbench/lift_numbered_block's verb is "grasp the numbered
                            block and LIFT it". Its end-state analog — the arm alone
                            delivers the target to the goal elevation point, i.e. the
                            target SEATED in the cradle with the mechanism unloaded —
                            is constructed and REJECTED: the beam stays down, only the
                            seat latch (0.15), no success. Elevation must come from
                            the loaded beam;
  7.  one counterweight   — + ONE distractor dropped into the pan (lands, non-vacuous)
                            still cannot tip the loaded beam: theta stays at the lower
                            stop, score 0.30, no success — the counterweight premise
                            is physically real;
  8.  wrong-block hoist   — a DISTRACTOR seated in the cradle with the target + the
                            other distractor as pan weights DOES tip the beam
                            (non-vacuous: theta >= raise threshold) yet is rejected
                            purely by identity: no seat credit, no success;
  9.  all-in-pan hoist    — all three blocks in the pan (three abreast) tips the
                            beam with the cradle EMPTY -> rejected;
  10. on-the-bar rest     — the target resting on the bare bar against the cradle's
                            outer wall (the "almost seated" miss) -> rejected by the
                            cradle x window, no seat latch;
  11. wall-top perch      — the target balanced ON a cradle wall top reads above the
                            seated z window -> rejected, removed before it topples;
  12. settle gate         — the full success GEOMETRY observed mid-tip (all three
                            blocks in their trays, beam swinging) is NOT success
                            (settle gates are real); a pan block is removed before
                            the beam can settle;
  13. latched credit+cap  — after that removal the beam falls back down, yet all four
                            latches survive: score == 0.60 cap (float32 + eps), NOT
                            success — full credit short of success is impossible
                            without the held hoist;
  14. rejection audit     — success() was never True at ANY judged point;
  15. final no-NaN        — all task-object states finite at the end;
  16. camera              — >= 20 rgb frames captured -> frames.npz.

Run (forge): python -u -m simgen_tasks.lift_numbered_block_i153.smoke --headless
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
    env = ENVS.get("simgen.beam_hoist")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    s_blk = c.block_s

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.05, -1.05, 0.85)) + o),
                                tuple(np.array((0.0, -0.05, 0.18)) + o),
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

    def theta_deg() -> float:
        return math.degrees(float(scene.beam_theta()[0]))

    def report(tag: str) -> None:
        s, ok = judge()
        tgt, d1, d2 = scene._role_flags()
        print(f"[smoke] {tag:16s} | target=block_{int(scene.target_idx[0])}"
              f" theta={theta_deg():+.1f}deg"
              f" seated={bool(tgt[0])} panA={bool(d1[0])} panB={bool(d2[0])}"
              f" latches=({float(scene.seat_latch[0]):.0f},"
              f"{float(scene.panA_latch[0]):.0f},{float(scene.panB_latch[0]):.0f},"
              f"{float(scene.up_latch[0]):.0f})"
              f" score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def drop_beam_local(idx: int, local_xy: tuple, hover: float = 0.045,
                        settle_steps: int = 150) -> None:
        """Probe placement: hover the block a few cm above a BEAM-FRAME tray point
        (beam-aligned orientation, zero velocity), release, REAL physics steps before
        judging (the zero-step trap). Reads the live beam pose, so probes track the
        current tilt automatically."""
        local = torch.tensor([[local_xy[0], local_xy[1], c.seat_z_b + hover]],
                             device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.beam_to_world(local)
        st[:, 3:7] = scene.beam.data.root_quat_w
        scene.blocks[idx].write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def place_rel(idx: int, xyz, settle_steps: int = 45) -> None:
        """Env-relative world placement (for parking a block on the open floor)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor(xyz, device=device)
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        scene.blocks[idx].write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def roles() -> tuple[int, int, int]:
        return (int(scene.target_idx[0]), int(scene.dist_idx[0, 0]),
                int(scene.dist_idx[0, 1]))

    def layout_sane(tag: str) -> bool:
        """Reset honesty: base at the env origin, beam pressed cradle-down, blocks
        resting on their scatter slots, the target's placard mounted (others parked),
        nothing in any tray."""
        bp = (scene.base.data.root_pos_w - scene.env_origins)[0]
        ok = float(bp[0]) ** 2 + float(bp[1]) ** 2 < 0.01 ** 2
        ok = ok and theta_deg() <= -12.0
        xs_lo = min(c.slot_xs) - c.scatter_jit_x - 0.03
        xs_hi = max(c.slot_xs) + c.scatter_jit_x + 0.03
        for b in scene.blocks:
            p = (b.data.root_pos_w - scene.env_origins)[0]
            ok = ok and xs_lo <= float(p[0]) <= xs_hi \
                and abs(float(p[1]) - c.slot_y) <= c.scatter_jit_y + 0.03 \
                and abs(float(p[2]) - s_blk / 2) < 0.02
        tgt = int(scene.target_idx[0])
        mount = torch.tensor(c.placard_mount, device=device)
        for k, pl in enumerate(scene.placards):
            p = (pl.data.root_pos_w - scene.env_origins)[0]
            if k == tgt:
                ok = ok and float((p - mount).norm()) < 0.02
            else:
                ok = ok and abs(float(p[0]) - c.park_xy[0]) < 0.05
        ok = ok and not bool(scene.in_cradle()[0].any()) \
            and not bool(scene.in_pan()[0].any())
        if not ok:
            print(f"[smoke] layout sanity VIOLATION at {tag}: theta={theta_deg():+.1f}",
                  flush=True)
        return ok

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    bodies = [scene.base, scene.beam] + scene.blocks + scene.placards
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("settle: all states finite; beam pressed cradle-down on its stop, blocks "
          "resting on their floor slots, target placard mounted, others parked",
          fin and layout_sane("reset"))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    sane = True
    for sd in (21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34):
        env.reset(seed=sd)
        step(15)
        sane = sane and layout_sane(f"seed {sd}")
        row = [float(scene.target_idx[0])]
        for b in scene.blocks:
            p = (b.data.root_pos_w - scene.env_origins)[0]
            row += [float(p[0]), float(p[1])]
        reads.append(row)
    arr = np.array(reads)
    print("[smoke] randomization readback (target, b0_x, b0_y, b1_x, b1_y, "
          f"b2_x, b2_y):\n{np.round(arr, 3)}", flush=True)
    targets = {int(t) for t in arr[:, 0]}
    check("randomization: all THREE target numbers occur over 14 seeded resets "
          f"(seen {sorted(targets)}) and the mounted placard tracks the target "
          "while the others park", targets == {0, 1, 2} and sane)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: the block scatter (permuted slots + jitter) varies per "
          f"seed (x spreads {spread[1]:.3f}/{spread[3]:.3f}/{spread[5]:.3f}, "
          f"y spreads {spread[2]:.3f}/{spread[4]:.3f}/{spread[6]:.3f})",
          min(spread[1], spread[3], spread[5]) > 0.05
          and min(spread[2], spread[4], spread[6]) > 0.01)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=41)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0, beam still down, no success after 240 idle steps",
          s <= 0.02 and theta_deg() <= -12.0 and not ok)

    # =========================== 6. seed strategy (arm-lift alone) fails ====================
    # rlbench/lift_numbered_block: identify + grasp + LIFT. The arm alone delivering
    # the target to the goal point = target seated in the cradle, mechanism unloaded.
    env.reset(seed=51)
    step(30)
    tgt_i, d1_i, d2_i = roles()
    drop_beam_local(tgt_i, (c.cradle_cx_b, 0.0), settle_steps=180)
    report("arm-lift-only")
    tgt, _d1, _d2 = scene._role_flags()
    s, ok = judge()
    check("seed strategy fails: the target SEATED in the cradle by the arm alone "
          f"(mechanism unloaded) leaves the beam DOWN (theta={theta_deg():+.1f} deg "
          "<= -12) — only the seat latch (0.15), no success: elevation must come "
          "from the loaded beam",
          bool(tgt[0]) and theta_deg() <= -12.0
          and 0.15 - 1e-6 <= s <= 0.15 + 1e-5 and not ok)

    # =========================== 7. one counterweight is insufficient =======================
    drop_beam_local(d1_i, (c.pan_cx_b, 0.045), settle_steps=300)
    report("one-weight")
    _tgt, d1, _d2 = scene._role_flags()
    s, ok = judge()
    check("one counterweight: the first distractor LANDS in the pan (non-vacuous) "
          f"yet cannot tip the loaded beam — theta={theta_deg():+.1f} deg stays at "
          "the lower stop, score 0.30, no success",
          bool(d1[0]) and theta_deg() <= -8.0
          and 0.30 - 1e-6 <= s <= 0.30 + 1e-5 and not ok)

    # =========================== 8. wrong-block hoist (identity) ============================
    env.reset(seed=61)
    step(30)
    tgt_i, d1_i, d2_i = roles()
    drop_beam_local(d1_i, (c.cradle_cx_b, 0.0), settle_steps=150)  # WRONG block seated
    drop_beam_local(tgt_i, (c.pan_cx_b, 0.045), settle_steps=150)  # target as weight
    drop_beam_local(d2_i, (c.pan_cx_b, -0.045), settle_steps=600)  # tips the beam
    report("wrong-block")
    tgt, _d1, _d2 = scene._role_flags()
    s, ok = judge()
    check("wrong-block hoist: a DISTRACTOR seated in the cradle with the target as a "
          f"pan weight DOES tip the beam (theta={theta_deg():+.1f} deg >= "
          f"{c.theta_up_min_deg:.0f}, non-vacuous) yet is rejected purely by "
          "identity — no seat credit, no success",
          theta_deg() >= c.theta_up_min_deg and not bool(tgt[0])
          and float(scene.seat_latch[0]) == 0.0 and s <= 0.30 + 1e-5 and not ok)

    # =========================== 9. all-in-pan hoist (empty cradle) =========================
    env.reset(seed=71)
    step(30)
    tgt_i, d1_i, d2_i = roles()
    # Three abreast: pan inner half-width 0.052 admits centers at 0/+-0.045 and the
    # y window (0.058) accepts all three; the second drop already tips the beam, the
    # third tracks the live tilted pose via drop_beam_local.
    drop_beam_local(d1_i, (c.pan_cx_b, 0.045), settle_steps=120)
    drop_beam_local(tgt_i, (c.pan_cx_b, 0.0), settle_steps=420)  # beam tips
    drop_beam_local(d2_i, (c.pan_cx_b, -0.045), settle_steps=300)
    report("all-in-pan")
    ic, ip = scene.in_cradle()[0], scene.in_pan()[0]
    s, ok = judge()
    check("all-in-pan hoist: all three blocks in the pan (three abreast, "
          f"all in_pan={bool(ip.all())}) tips the beam (theta={theta_deg():+.1f} "
          "deg) with the cradle EMPTY -> no seat credit, no success",
          bool(ip.all()) and theta_deg() >= c.theta_up_min_deg
          and not bool(ic.any()) and float(scene.seat_latch[0]) == 0.0 and not ok)

    # =========================== 10. on-the-bar rest ========================================
    env.reset(seed=81)
    step(30)
    tgt_i, d1_i, d2_i = roles()
    bar_x = c.cradle_cx_b - c.cradle_out / 2 - s_blk / 2 - 0.002
    drop_beam_local(tgt_i, (bar_x, 0.0), hover=-(c.seat_z_b - c.bar_t / 2
                                                 - s_blk / 2 - 0.004),
                    settle_steps=150)
    report("bar-rest")
    p = scene.blocks_beam()[0, tgt_i]
    tgt, _d1, _d2 = scene._role_flags()
    s, ok = judge()
    check("on-the-bar rest: the target resting on the bare bar against the cradle's "
          f"outer wall (beam-frame x offset {float(p[0]) - c.cradle_cx_b:+.3f} m) "
          "is outside the cradle x window -> no seat latch, no success",
          not bool(tgt[0]) and abs(float(p[0]) - c.cradle_cx_b) > c.cradle_win_xy
          and float(scene.seat_latch[0]) == 0.0 and s <= 0.02 and not ok)

    # =========================== 11. wall-top perch =========================================
    env.reset(seed=91)
    step(30)
    tgt_i, d1_i, d2_i = roles()
    wall_x = c.cradle_cx_b + c.cradle_out / 2 - c.wall_t / 2
    perch_z = c.floor_top_b + c.wall_h + s_blk / 2 + 0.001
    local = torch.tensor([[wall_x, 0.0, perch_z]], device=device).expand(n, 3)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.beam_to_world(local)
    st[:, 3:7] = scene.beam.data.root_quat_w
    scene.blocks[tgt_i].write_root_state_to_sim(st, all_ids)
    step(3)  # judge before it topples off the 8 mm wall
    p = scene.blocks_beam()[0, tgt_i]
    report("wall-perch")
    tgt, _d1, _d2 = scene._role_flags()
    _s, ok = judge()
    perch_ok = (not bool(tgt[0]) and float(p[2]) > c.seat_z_b + c.seat_z_tol
                and not ok and float(scene.seat_latch[0]) == 0.0)
    place_rel(tgt_i, (0.0, -0.5, s_blk / 2 + 0.002), settle_steps=30)
    check("wall-top perch: the target balanced ON a cradle wall top reads beam-frame "
          f"z={float(p[2]) * 1000:.0f} mm above the seated window "
          f"({(c.seat_z_b + c.seat_z_tol) * 1000:.0f} mm) -> rejected; removed "
          "before it topples", perch_ok)

    # =========================== 12-13. settle gate + latched credit / cap ==================
    env.reset(seed=101)
    step(30)
    tgt_i, d1_i, d2_i = roles()
    drop_beam_local(tgt_i, (c.cradle_cx_b, 0.0), settle_steps=150)
    drop_beam_local(d1_i, (c.pan_cx_b, 0.045), settle_steps=150)
    drop_beam_local(d2_i, (c.pan_cx_b, -0.045), settle_steps=0)  # now watch the tip
    gate_seen = False
    removed = False
    for i in range(480):
        step(1)
        tgt, d1, d2 = scene._role_flags()
        w_beam = float(scene.beam.data.root_ang_vel_w[0].norm())
        _s, ok = judge()
        if bool(tgt[0]) and bool(d1[0]) and bool(d2[0]) \
                and w_beam > c.beam_settle_ang and not ok:
            gate_seen = True
        if theta_deg() >= c.theta_up_min_deg + 0.5:
            # all four latches are in — remove a pan block BEFORE the beam settles
            st = torch.zeros(n, 13, device=device)
            st[:, 0:3] = torch.tensor([0.3, -0.55, s_blk / 2 + 0.002], device=device)
            st[:, 3] = 1.0
            st[:, 0:3] += scene.env_origins
            scene.blocks[d2_i].write_root_state_to_sim(st, all_ids)
            removed = True
            print(f"[smoke] gate probe: pan block removed mid-tip @step {i}, "
                  f"theta={theta_deg():+.1f} w={w_beam:.2f}", flush=True)
            break
    check("settle gate: the full success GEOMETRY observed mid-tip (all three in "
          "their trays, beam swinging past the threshold) is NOT success — the "
          f"settle gates are real (gate_seen={gate_seen}, removed={removed})",
          gate_seen and removed)
    step(300)
    report("cap-after-removal")
    s, ok = judge()
    lat = (float(scene.seat_latch[0]), float(scene.panA_latch[0]),
           float(scene.panB_latch[0]), float(scene.up_latch[0]))
    check("latched credit + cap: with a counterweight yanked mid-hoist the beam "
          f"falls back down (theta={theta_deg():+.1f} deg <= -8) yet all four "
          f"latches survive (latches={lat}): score == 0.60 cap ({s:.4f}, float32 + "
          "eps), NOT success — full credit short of success is impossible without "
          "the held hoist",
          lat == (1.0, 1.0, 1.0, 1.0) and theta_deg() <= -8.0
          and 0.595 <= s <= 0.60 + 1e-5 and not ok)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== 16. camera + save + verdict ================================
    check(f"camera: >= 20 rgb frames captured ({len(frames)})", len(frames) >= 20)
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.beam_hoist")
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

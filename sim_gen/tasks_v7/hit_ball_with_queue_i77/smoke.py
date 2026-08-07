"""Smoke / rubric-REJECTION battery for SkywayBridgeScene (sim_gen task
`hit_ball_with_queue_i77`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — bridge dropped onto the tabs under gravity, ball
nudged over the ridge, hands-off gravity descent into the basin — is the acceptance
evidence that the rubric ACCEPTS a correct outcome). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled state and
asserts the rubric REJECTS it; no probe in this battery ever reaches success(), and a
final audit check asserts exactly that.

  1-2. settle/no-NaN     — reset layout settles finite: ball on the deck behind the
                           ridge, both spans flat on the ground, all still, score ~0;
  3-4. randomization     — READBACK over 8 seeded resets: whole-assembly yaw + xy
                           offset are real; bridge/decoy ground-slot assignment flips
                           (Bernoulli swap); per-span jitter and ball deck jitter real;
  5.  null policy        — 240 idle steps -> score ~0, no success;
  6.  seed strategy      — the end state the SEED's plan produces here (STRIKE the
                           ball): fired at 1.5 m/s from the deck with no bridge, the
                           ball hops the ridge, races down slope 1 and falls THROUGH
                           the gap to the floor (it cannot jump the 18 cm break) ->
                           unrecoverable, score ~0, no success;
  7.  out-of-order       — gentle release (the CORRECT ball move) but BEFORE bridging:
                           ball rolled off the deck with no bridge falls through the
                           gap to the floor -> score ~0, no success (order is forced);
  8.  near-miss seat u   — bridge dropped 30 mm off-centre longitudinally (just
                           outside the 20 mm u-tolerance): its uphill end misses the
                           tab and it tips into the gap -> NOT seated, score ~0;
  9.  wrong object       — the DECOY dropped centred over the gap: 90 mm span vs the
                           110 mm tab-tip opening — it bears on NEITHER tab and falls
                           through -> NOT seated, score ~0 (length identification is
                           load-bearing);
  10. near-miss skew     — bridge dropped YAWED 30 deg: its diagonal cannot enter the
                           190 mm guide slot, it rests high/canted -> NOT seated
                           (z/alignment gates), score ~0;
  11. cross gate         — ball placed over the gap AT TRACK HEIGHT with NO seated
                           bridge: the crossed latch (which requires the bridge seated
                           UNDER the ball) never fires; ball falls through -> score ~0;
  12. beside-wall        — ball settled on the GROUND beside the basin at an in-range
                           u: assembly-frame v/z math rejects (floor level is not
                           basin level) -> no success;
  13. latched credit     — bridge seated by a probe drop -> score 0.30; bridge then
                           teleported back to the floor -> latched credit survives
                           (still 0.30), NOT success, non-success cap holds;
  14. rejection audit    — success() was never True at ANY judged point;
  15. final no-NaN       — all task-object states finite at the end.

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
    env = ENVS.get("simgen.skyway_bridge")().build(num_envs=args.num_envs,
                                                   device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply, quat_mul

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.45, -1.30, 1.05)) + o),
                                tuple(np.array((0.05, 0.00, 0.15)) + o),
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
        bl = scene._local(scene.ball)[0]
        rl = scene._local(scene.bridge)[0]
        dl = scene._local(scene.decoy)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | ball=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) bridge=({float(rl[0]):+.3f},{float(rl[1]):+.3f},"
              f"{float(rl[2]):.3f}) decoy=({float(dl[0]):+.3f},{float(dl[1]):+.3f},"
              f"{float(dl[2]):.3f}) seated={bool(scene._bridge_seated()[0])} "
              f"seated_ever={bool(scene._seated_ever[0])} "
              f"crossed_ever={bool(scene._crossed_ever[0])} "
              f"in_basin={bool(scene._in_basin()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_local(body, u: float, v: float, z: float, *, vel_u: float = 0.0,
                    yaw: float = 0.0) -> None:
        """Teleport `body` to an assembly-local point of the skyway's CURRENT pose
        (probe constructor: builds hover/over-gap/beside relations directly),
        optionally with an initial velocity along the downhill axis (the seed's
        strike) and/or a yaw relative to the track."""
        s_pos = scene.skyway.data.root_pos_w
        s_quat = scene.skyway.data.root_quat_w
        loc = torch.tensor([u, v, z], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = s_pos + quat_apply(s_quat, loc)
        if yaw:
            h = yaw / 2
            qy = torch.tensor([math.cos(h), 0.0, 0.0, math.sin(h)],
                              device=device).expand(n, 4)
            st[:, 3:7] = quat_mul(s_quat, qy)
        else:
            st[:, 3:7] = s_quat
        if vel_u:
            axis = quat_apply(s_quat, torch.tensor([1.0, 0.0, 0.0],
                                                   device=device).expand(n, 3))
            st[:, 7:10] = axis * vel_u
        body.write_root_state_to_sim(st, all_ids)

    def finite_all() -> bool:
        return bool(torch.isfinite(scene.skyway.data.root_state_w).all()
                    and torch.isfinite(scene.bridge.data.root_state_w).all()
                    and torch.isfinite(scene.decoy.data.root_state_w).all()
                    and torch.isfinite(scene.ball.data.root_state_w).all())

    def skyway_pose() -> tuple[float, float, float]:
        p = (scene.skyway.data.root_pos_w - scene.env_origins)[0]
        q = scene.skyway.data.root_quat_w[0]
        return (float(p[0]), float(p[1]),
                math.degrees(2.0 * math.atan2(float(q[3]), float(q[0]))))

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("show")
    bl = scene._local(scene.ball)[0]
    rl = scene._local(scene.bridge)[0]
    dl = scene._local(scene.decoy)[0]
    still = (float(scene.ball.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.bridge.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.decoy.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    check("settle: states finite, ball ON the deck behind the ridge, both spans flat "
          "on the ground, all still",
          finite_all() and float(bl[2]) > c.deck_top
          and float(bl[0]) < c.ridge_u and float(rl[2]) < 0.05
          and float(dl[2]) < 0.05 and still)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        sx, sy, syaw = skyway_pose()
        rl = scene._local(scene.bridge)[0]
        bl = scene._local(scene.ball)[0]
        # bridge on the DOWNHILL ground slot (slot_a u=+0.02) vs UPHILL (slot_b -0.32)
        on_a = 1.0 if float(rl[0]) > -0.15 else 0.0
        reads.append((sx, sy, syaw, float(rl[0]), float(rl[1]), on_a,
                      float(bl[0]), float(bl[1])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (skyway_x, skyway_y, skyway_yaw_deg, "
          f"bridge_u, bridge_v, bridge_on_slot_a, ball_u, ball_v):\n{arr}", flush=True)
    yaw_spread = float(arr[:, 2].max() - arr[:, 2].min())
    xy_spread = float((arr[:, 0:2].max(axis=0) - arr[:, 0:2].min(axis=0)).max())
    check("randomization: whole-assembly yaw spread (> 2 deg) and xy offset spread "
          "(> 4 mm) are real (readback)", yaw_spread > 2.0 and xy_spread > 0.004)
    flags = arr[:, 5]
    jit = 0.0
    for flag in (0.0, 1.0):
        grp = arr[flags == flag]
        if len(grp) >= 2:
            jit = max(jit, float((grp[:, 3:5].max(axis=0)
                                  - grp[:, 3:5].min(axis=0)).max()))
    ball_spread = float((arr[:, 6:8].max(axis=0) - arr[:, 6:8].min(axis=0)).max())
    check("randomization: bridge/decoy slot assignment flips across seeded resets, "
          "per-span jitter (> 4 mm) and ball deck jitter (> 4 mm) are real (readback)",
          0.0 < flags.mean() < 1.0 and jit > 0.004 and ball_spread > 0.004)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 6. seed strategy: STRIKE the ball ==========================
    # The seed's plan — impulse the ball toward the goal — executed here: fired at
    # 1.5 m/s from the deck with NO bridge. It hops the ridge, races down slope 1,
    # cannot jump the 18 cm break (that needs > 6 m/s) and falls THROUGH the gap to
    # the floor: unrecoverable, nothing scores.
    torch.manual_seed(41)
    env.reset()
    step(30)
    place_local(scene.ball, c.ball_slot_u, 0.0, c.deck_top + c.ball_r + 0.002,
                vel_u=1.5)
    step(300)
    report("seed-strategy")
    bl = scene._local(scene.ball)[0]
    s, ok = judge()
    check("seed strategy: ball STRUCK at 1.5 m/s with no bridge falls through the "
          "gap to the floor (z < 0.10), not in the basin — score ~0 (<= 0.02), no "
          "success (striking is the losing move)",
          float(bl[2]) < 0.10 and not bool(scene._in_basin()[0])
          and not bool(scene._crossed_ever[0]) and s <= 0.02 and not ok)

    # =========================== 7. out-of-order: release before bridging ===================
    # The CORRECT ball move (gentle release) executed BEFORE the bridge is seated:
    # the ball rolls off the deck edge and falls through the gap to the floor.
    torch.manual_seed(51)
    env.reset()
    step(30)
    place_local(scene.ball, c.deck_u1 - 0.005, 0.0, c.deck_top + c.ball_r + 0.002,
                vel_u=0.05)
    step(300)
    report("out-of-order")
    bl = scene._local(scene.ball)[0]
    s, ok = judge()
    check("out-of-order: ball gently released with NO bridge seated falls through "
          "the gap to the floor (z < 0.10) — unrecoverable, score ~0, no success "
          "(execution order is geometry-forced)",
          float(bl[2]) < 0.10 and not bool(scene._in_basin()[0])
          and not bool(scene._crossed_ever[0]) and s <= 0.02 and not ok)

    # =========================== 8. near-miss seat: longitudinal ============================
    # Bridge dropped 30 mm off-centre in u (just outside the 20 mm tolerance): its
    # uphill end (at u=-0.05) misses the uphill tab (tip at -0.055) entirely — it
    # tips into the gap and never reads seated.
    torch.manual_seed(61)
    env.reset()
    step(30)
    place_local(scene.bridge, 0.030, 0.004, c.seat_z + 0.030)
    step(300)
    report("near-miss-u")
    rl = scene._local(scene.bridge)[0]
    s, ok = judge()
    check("near-miss seat (longitudinal): bridge dropped 30 mm off-centre — uphill "
          "end misses its tab, span tips into the gap: NOT seated, no seating "
          "credit (score <= 0.02), no success",
          not bool(scene._bridge_seated(require_still=False)[0])
          and not bool(scene._seated_ever[0]) and s <= 0.02 and not ok)

    # =========================== 9. wrong object: the decoy =================================
    torch.manual_seed(71)
    env.reset()
    step(30)
    place_local(scene.decoy, 0.0, 0.0, c.seat_z + 0.030)
    step(300)
    report("wrong-object")
    dl = scene._local(scene.decoy)[0]
    s, ok = judge()
    check("wrong object: DECOY (90 mm < the 110 mm tab-tip opening) dropped centred "
          "over the gap bears on NEITHER tab and falls through (ends low, z < 0.15) "
          "— no seating credit, score ~0, no success (length identification is "
          "load-bearing)",
          float(dl[2]) < 0.15 and not bool(scene._seated_ever[0])
          and s <= 0.02 and not ok)

    # =========================== 10. near-miss seat: skew ===================================
    # Bridge dropped YAWED 30 deg: the diagonal (~0.23 m) cannot enter the 0.19 m
    # guide slot — it rests high on the guide walls / canted, failing the z and
    # alignment gates.
    torch.manual_seed(81)
    env.reset()
    step(30)
    place_local(scene.bridge, 0.0, 0.0, c.seat_z + 0.050, yaw=math.radians(30.0))
    step(300)
    report("near-miss-skew")
    rl = scene._local(scene.bridge)[0]
    s, ok = judge()
    check("near-miss seat (skew): bridge dropped yawed 30 deg cannot enter the guide "
          "slot — rests high/canted: NOT seated, no seating credit, no success",
          not bool(scene._bridge_seated(require_still=False)[0])
          and not bool(scene._seated_ever[0]) and s <= 0.02 and not ok)

    # =========================== 11. cross gate needs the bridge UNDER the ball =============
    # Ball constructed over the gap AT TRACK HEIGHT with no seated bridge: _over_gap
    # is momentarily true but the crossed latch requires the bridge seated UNDER the
    # ball — it must never fire; the ball just falls through.
    torch.manual_seed(91)
    env.reset()
    step(30)
    place_local(scene.ball, 0.0, 0.0, 0.27)
    step(240)
    report("cross-gate")
    bl = scene._local(scene.ball)[0]
    s, ok = judge()
    check("cross gate: ball placed over the gap at track height with NO seated "
          "bridge — the crossed latch (bridge-under-ball required) never fires, the "
          "ball falls through (z < 0.10): score ~0, no success",
          not bool(scene._crossed_ever[0]) and float(bl[2]) < 0.10
          and s <= 0.02 and not ok)

    # =========================== 12. beside-wall (frame math) ===============================
    torch.manual_seed(101)
    env.reset()
    step(30)
    place_local(scene.ball, 0.50, c.basin_hw + 0.10, c.ball_r + 0.002)
    step(180)
    report("beside-wall")
    bl = scene._local(scene.ball)[0]
    s, ok = judge()
    check("beside-wall: ball settled on the GROUND beside the basin at an in-range "
          "u — assembly-frame v/z math rejects (floor level is not basin level): "
          "NOT in basin, no success",
          not bool(scene._in_basin()[0]) and float(bl[2]) < 0.10 and not ok)

    # =========================== 13. latched credit survives regression =====================
    torch.manual_seed(111)
    env.reset()
    step(30)
    place_local(scene.bridge, 0.0, 0.004, c.seat_z + 0.030)
    seated = False
    for _ in range(480):
        env.step(no_action)
        if bool(scene._bridge_seated()[0]):
            seated = True
            break
    report("probe-seated")
    s13a, ok = judge()
    got_credit = seated and abs(s13a - c.w_seat) < 0.01 and not ok
    # regression: bridge teleported back to the floor
    place_local(scene.bridge, c.slot_a[0], c.slot_a[1], c.span_floor_t / 2 + 0.002)
    step(60)
    report("regressed")
    s13b, ok = judge()
    check("latched credit: probe-seated bridge earns exactly the seating credit "
          f"(0.30, got {s13a:.3f}) with NO success; teleporting it back to the floor "
          f"leaves the latched score unchanged ({s13a:.3f} -> {s13b:.3f}), still no "
          "success (non-success cap holds)",
          got_credit and abs(s13b - s13a) < 1e-3
          and not bool(scene._bridge_seated(require_still=False)[0]) and not ok)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.skyway_bridge")
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

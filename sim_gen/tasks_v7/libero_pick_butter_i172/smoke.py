"""Smoke / rubric-REJECTION battery for ButterSwitchyardScene (sim_gen task
`libero_pick_butter_i172`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — zero teleports, all four legs contact-dynamic
pushes through the T-channel — is the acceptance evidence that the rubric ACCEPTS a
correct outcome). Every teleport here is instrumentation that CONSTRUCTS a wrong
(or partial) outcome as a settled state and asserts the rubric REJECTS it; no probe
in this battery ever reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: both blocks seated in the
                            trunk (butter behind the brick), basket upright on the
                            ground under a mouth, everything still, score ~0;
  3-4. randomization      — READBACK over 8 seeded resets: the basket's mouth side
                            flips; deck yaw + xy, brick/butter slot x, basket yaw
                            all vary;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed strategy       — the end state the seed's plan (grasp the butter, carry
                            it, drop it into the basket) produces: butter dropped
                            straight down into the basket from above -> geometric
                            containment verified TRUE, yet rejected (the junction
                            route credential was never earned: delivered never
                            sets), score ~0;
  7.  crossbar shortcut   — butter set INTO the crossbar at the basket-side mouth
                            lip (over the wall, skipping the junction) with outward
                            velocity, so it genuinely FALLS into the basket ->
                            containment TRUE, junction False, delivered False,
                            rejected, score ~0;
  8.  wrong mouth         — butter at rest on the bare ground outboard of the AWAY
                            mouth (the disposal exit) -> no credit, no success;
  9.  contamination       — junction credit earned first (>= 0.20), then the BRICK
                            dropped into the basket -> `dirty` latches and caps the
                            score at 0.10, BELOW the credit already earned;
  10. dirty irreversible  — the brick then removed to far ground and settled (which
                            also latches `cleared` + full route progress) -> dirty
                            still True, score still capped at 0.10, no success;
  11. brick clause        — butter delivery constructed legitimately (junction
                            latched, then a real FALL into the basket: delivered
                            verified SET, containment TRUE) but the brick still in
                            the trunk -> the brick-disposal clauses ALONE reject
                            (score 0.55, capped < 1);
  12. latched credit      — junction credit earned, then the butter yanked back to
                            the trunk rear -> the latched score holds, no success;
  13. rejection audit     — success() was never True at ANY judged point;
  14. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.libero_pick_butter_i172.smoke --headless
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
# DAEMON so a crashed main thread doesn't idle the process until the timer fires.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    from isaaclab.utils.math import quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.butter_switchyard")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.65, -1.30, 0.95)) + o),
                                tuple(np.array((0.35, 0.00, 0.15)) + o),
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

    def deck_yaw() -> float:
        q = scene.deck.data.root_quat_w[0]
        return 2.0 * math.atan2(float(q[3]), float(q[0]))

    def report(tag: str) -> None:
        bl = scene._deck_local(scene.butter)[0]
        kl = scene._deck_local(scene.brick)[0]
        bb = scene._basket_local(scene.butter)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | butter_d=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) brick_d=({float(kl[0]):+.3f},{float(kl[1]):+.3f},"
              f"{float(kl[2]):.3f}) butter_b=({float(bb[0]):+.3f},{float(bb[1]):+.3f},"
              f"{float(bb[2]):.3f}) prog={float(scene._prog_max[0]):.2f} "
              f"cleared={bool(scene._cleared[0])} junction={bool(scene._junction[0])} "
              f"delivered={bool(scene._delivered[0])} dirty={bool(scene._dirty[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_world(body, x: float, y: float, z: float,
                    quat=(1.0, 0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 7], st[:, 8], st[:, 9] = vel
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def place_deck(body, x_l: float, y_l: float, z_l: float,
                   v_l=(0.0, 0.0, 0.0)) -> None:
        """Place `body` at a pose (and optional velocity) given in the DECK's body
        frame, yawed to the deck — probe constructor robust to the deck's
        per-episode yaw/xy jitter (the deck origin sits ON the ground)."""
        dp = (scene.deck.data.root_pos_w - scene.env_origins)[0]
        yw = deck_yaw()
        cy, sy = math.cos(yw), math.sin(yw)
        wx = float(dp[0]) + cy * x_l - sy * y_l
        wy = float(dp[1]) + sy * x_l + cy * y_l
        vx = cy * v_l[0] - sy * v_l[1]
        vy = sy * v_l[0] + cy * v_l[1]
        place_world(body, wx, wy, float(dp[2]) + z_l,
                    (math.cos(yw / 2), 0.0, 0.0, math.sin(yw / 2)),
                    (vx, vy, v_l[2]))

    def drop_into_basket(body) -> None:
        """Drop `body` from straight above the basket centre so it genuinely FALLS
        into the interior (this is how a probe constructs real containment)."""
        bp = (scene.basket.data.root_pos_w - scene.env_origins)[0]
        q = scene.basket.data.root_quat_w[0]
        byaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        place_world(body, float(bp[0]), float(bp[1]), 0.32,
                    (math.cos(byaw / 2), 0.0, 0.0, math.sin(byaw / 2)))

    def states_finite() -> bool:
        return bool(torch.isfinite(scene.butter.data.root_state_w).all()
                    and torch.isfinite(scene.brick.data.root_state_w).all()
                    and torch.isfinite(scene.basket.data.root_state_w).all()
                    and torch.isfinite(scene.deck.data.root_state_w).all())

    def all_still() -> bool:
        return (float(scene.butter.data.root_lin_vel_w[0].norm()) < c.settle_speed
                and float(scene.brick.data.root_lin_vel_w[0].norm()) < c.settle_speed
                and float(scene.basket.data.root_lin_vel_w[0].norm()) < c.settle_speed)

    def in_trunk(obj) -> bool:
        loc = scene._deck_local(obj)[0]
        return (c.trunk_x0 - 0.01 < float(loc[0]) < 0.08
                and abs(float(loc[1])) < c.chan_w / 2 + 0.01
                and bool(scene._on_deck(obj)[0]))

    z_chan = c.deck_top + c.block_h / 2 + 0.003  # block on the channel floor
    z_gnd = c.block_h / 2 + 0.002  # block on the ground

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    side = float(scene._side[0])
    bskl = scene._deck_local(scene.basket)[0]
    check("settle: states finite, both blocks seated in the trunk (butter behind the "
          "brick), basket upright on the ground under the sampled mouth, everything "
          "still",
          states_finite() and in_trunk(scene.butter) and in_trunk(scene.brick)
          and float(scene._deck_local(scene.butter)[0][0])
          < float(scene._deck_local(scene.brick)[0][0])
          and bool(scene._basket_upright()[0])
          and abs(float(bskl[1]) - side * c.catch_y) < 0.05 and all_still())
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(2)
        dp = (scene.deck.data.root_pos_w - scene.env_origins)[0]
        bq = scene.basket.data.root_quat_w[0]
        byaw = math.degrees(2.0 * math.atan2(float(bq[3]), float(bq[0])))
        reads.append((float(dp[0]), float(dp[1]), math.degrees(deck_yaw()),
                      float(scene._brick_x0[0]),
                      float(scene._deck_local(scene.butter)[0][0]),
                      1.0 if float(scene._side[0]) > 0 else 0.0,
                      byaw))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (deck_x, deck_y, deck_yaw_deg, brick_x0, "
          f"butter_x, side_pos, basket_yaw_deg):\n{arr}", flush=True)
    check("randomization: the basket's mouth side flips across seeded resets (readback)",
          0.0 < arr[:, 5].mean() < 1.0)
    deck_var = (float(arr[:, 2].max() - arr[:, 2].min()) > 2.0
                and float((arr[:, 0:2].max(axis=0) - arr[:, 0:2].min(axis=0)).max()) > 0.008)
    obj_var = (float(arr[:, 3].max() - arr[:, 3].min()) > 0.01
               and float(arr[:, 4].max() - arr[:, 4].min()) > 0.006
               and float(arr[:, 6].max() - arr[:, 6].min()) > 30.0)
    check("randomization: deck yaw + xy, brick slot x, butter slot x, basket yaw all "
          "vary (readback)", deck_var and obj_var)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy: butter dropped into the basket ===========
    # The seed's whole plan — grasp the butter, carry it, release it over the basket —
    # produces containment via a vertical drop from free space. Geometric containment
    # is verified TRUE; the junction route credential rejects it anyway.
    torch.manual_seed(41)
    env.reset()
    step(10)
    drop_into_basket(scene.butter)
    step(100)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy: butter dropped into the basket from above (grasp-and-carry "
          "end state) — geometric containment TRUE, junction never earned, delivered "
          "never sets, no success, score <= 0.05",
          bool(scene._in_basket(scene.butter)[0]) and not bool(scene._junction[0])
          and not bool(scene._delivered[0]) and not ok and s <= 0.05)

    # =========================== 7. crossbar shortcut (junction skipped) ====================
    # Butter lifted over the wall INTO the crossbar at the basket-side mouth lip and
    # nudged outward: it genuinely FALLS off the mouth into the basket — the exact
    # exit success uses — but the junction was never transited.
    torch.manual_seed(51)
    env.reset()
    step(10)
    side = float(scene._side[0])
    # 5 mm inside the lip with 0.35 m/s outward: friction (~2.4 m/s^2) cannot stop it
    # before the edge, and it lands ~50 mm outboard — well inside the basket.
    place_deck(scene.butter, 0.0, side * (c.cross_half - 0.005), z_chan,
               v_l=(0.0, side * 0.35, 0.0))
    step(120)
    report("crossbar-cheat")
    s, ok = judge()
    check("crossbar shortcut: butter set into the crossbar at the mouth lip (over the "
          "wall, skipping the junction) and genuinely fallen into the basket — "
          "containment TRUE, junction False, delivered False, no success, "
          "score <= 0.05",
          bool(scene._in_basket(scene.butter)[0]) and not bool(scene._junction[0])
          and not bool(scene._delivered[0]) and not ok and s <= 0.05)

    # =========================== 8. wrong mouth =============================================
    torch.manual_seed(61)
    env.reset()
    step(10)
    side = float(scene._side[0])
    place_deck(scene.butter, 0.0, -side * (c.cross_half + 0.06), z_gnd)
    step(60)
    report("wrong-mouth")
    s, ok = judge()
    check("wrong mouth: butter at rest on the bare ground outboard of the AWAY mouth "
          "(disposal exit) — not in the basket, no delivery credit, no success, "
          "score <= 0.05",
          not bool(scene._in_basket(scene.butter)[0]) and not bool(scene._delivered[0])
          and not ok and s <= 0.05)

    # =========================== 9. contamination caps below earned credit ==================
    torch.manual_seed(71)
    env.reset()
    step(10)
    place_deck(scene.butter, 0.0, 0.0, z_chan)  # junction credit (latch on presence)
    step(30)
    s_a, _ = judge()
    drop_into_basket(scene.brick)
    step(100)
    report("contaminated")
    s_b, ok = judge()
    check("contamination: junction credit earned (>= 0.20), then the BRICK dropped "
          "into the basket — dirty latches and the score caps at 0.10, BELOW the "
          "credit already earned, no success",
          s_a >= 0.199 and bool(scene._junction[0]) and bool(scene._dirty[0])
          and s_b <= 0.101 and not ok)

    # =========================== 10. contamination is irreversible ==========================
    place_world(scene.brick, -0.60, -0.80, z_gnd)  # "fish the brick back out"
    step(60)
    report("fished-out")
    s, ok = judge()
    check("dirty irreversible: the brick removed from the basket to far ground and "
          "settled (cleared + route progress now latch TRUE) — dirty still True, "
          "score still capped at 0.10, no success",
          bool(scene._cleared[0]) and bool(scene._dirty[0]) and s <= 0.101 and not ok)

    # =========================== 11. brick clause isolation =================================
    # Butter delivery constructed LEGITIMATELY (junction latched by transit, then a
    # real fall into the basket sets `delivered`), but the brick never left the
    # trunk: the brick-disposal clauses alone must reject.
    torch.manual_seed(81)
    env.reset()
    step(10)
    place_deck(scene.butter, 0.0, 0.0, z_chan)
    step(30)
    drop_into_basket(scene.butter)
    step(100)
    report("brick-clause")
    s, ok = judge()
    check("brick clause: junction latched + butter genuinely FELL into the basket "
          "(delivered verified SET, containment TRUE) but the brick still sits in "
          "the trunk — brick clauses alone reject, no success, 0.50 <= score <= 0.85",
          bool(scene._junction[0]) and bool(scene._delivered[0])
          and bool(scene._in_basket(scene.butter)[0]) and in_trunk(scene.brick)
          and not bool(scene._cleared[0]) and not ok and 0.50 <= s <= 0.85)

    # =========================== 12. latched credit survives regression =====================
    torch.manual_seed(91)
    env.reset()
    step(10)
    place_deck(scene.butter, 0.0, 0.0, z_chan)
    step(30)
    s_a, ok_a = judge()
    place_deck(scene.butter, -0.38, 0.0, z_chan)  # yank it all the way back
    step(60)
    report("regressed")
    s_b, ok_b = judge()
    check("latched credit: junction credit earned then the butter yanked back to the "
          f"trunk rear — score holds ({s_a:.3f} -> {s_b:.3f}), >= 0.199, still no "
          "success",
          bool(scene._junction[0]) and abs(s_a - s_b) < 1e-3 and s_a >= 0.199
          and not ok_a and not ok_b)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", states_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.butter_switchyard")
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
    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001 — any crash must exit NOW, not idle to the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)

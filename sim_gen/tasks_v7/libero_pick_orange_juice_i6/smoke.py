"""Smoke / rubric-REJECTION battery for LedgeCatchScene (sim_gen task
`libero_pick_orange_juice_i6`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — basket staged by transport teleport, then the
ungraspable ORANGE carton slid off the slab edge by contact forces and caught in
free fall — is the acceptance evidence that the rubric ACCEPTS a correct outcome).
Every teleport here is instrumentation that CONSTRUCTS a wrong (or partial) outcome
as a settled state and asserts the rubric REJECTS it; no probe in this battery ever
reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: both cartons standing on
                            the slab, basket upright on open ground far from the
                            catch point, everything still, score ~0;
  3-4. randomization      — READBACK over 8 seeded resets: the target's slab side
                            flips AND the basket's ground side flips; shelf yaw +
                            xy, target slot x + relative yaw, basket x all vary;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed strategy       — the end state the seed's plan (pick the item, drop it
                            into the basket WHERE IT STANDS) produces: carton
                            dropped into the never-moved basket at its spawn ->
                            geometric containment verified TRUE, yet rejected (the
                            basket is not at the catch station; caught never sets);
  7.  staging gate        — basket dumped UPSIDE-DOWN at the catch point (carton
                            still shelved) -> the staged latch refuses, score
                            stays at approach-only;
  8.  near-miss drop      — basket staged correctly, then the carton grounded just
                            OUTBOARD of the basket (an overshot push) -> rejected,
                            credit stays at staged-level;
  9.  capping (order)     — the out-of-order recovery: carton on the ground AT the
                            catch point, basket lowered down OVER it -> geometric
                            containment verified TRUE, the fell-in gate alone
                            rejects (caught False), score approach-only;
  10. capped after stage  — staged first (latch legitimately set), then the carton
                            placed at rest into the basket interior (what a
                            missed-drop + re-cap recovery produces) -> containment
                            TRUE, staged TRUE, still rejected: it never FELL in;
  11. wrong object        — WHITE decoy dropped into the staged basket, ORANGE
                            still shelved -> rejected (wrong carton in the basket,
                            decoy off the shelf);
  12. decoy clause        — decoy knocked to the ground, then the TARGET genuinely
                            dropped into the staged basket (fell-in latch verified
                            SET) -> every target clause holds, the decoy-on-shelf
                            clause ALONE rejects (score capped 0.85);
  13. latched credit      — staging credit earned, then the basket yanked away ->
                            the latched score holds (credit does not evaporate),
                            still no success;
  14. rejection audit     — success() was never True at ANY judged point;
  15. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.libero_pick_orange_juice_i6.smoke --headless
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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ledge_catch")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.70, -1.15, 0.92)) + o),
                                tuple(np.array((0.55, 0.00, 0.14)) + o),
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

    def shelf_pose() -> tuple[torch.Tensor, float]:
        sp = (scene.shelf.data.root_pos_w - scene.env_origins)[0]
        q = scene.shelf.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return sp, yaw

    def report(tag: str) -> None:
        tl = scene._shelf_local(scene.target)[0]
        bl = scene._shelf_local(scene.basket)[0]
        tb = scene._basket_local(scene.target)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | target_s=({float(tl[0]):+.3f},{float(tl[1]):+.3f},"
              f"{float(tl[2]):.3f}) basket_s=({float(bl[0]):+.3f},{float(bl[1]):+.3f}) "
              f"target_b=({float(tb[0]):+.3f},{float(tb[1]):+.3f},{float(tb[2]):.3f}) "
              f"on_shelf(t/d)={bool(scene._on_shelf(scene.target)[0])}/"
              f"{bool(scene._on_shelf(scene.decoy)[0])} "
              f"upright={bool(scene._basket_upright()[0])} "
              f"inside={bool(scene._target_inside_now()[0])} "
              f"staged={bool(scene._staged[0])} caught={bool(scene._caught[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_world(body, x: float, y: float, z: float,
                    quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def place_local(body, x_l: float, y_l: float, z: float,
                    roll_deg: float = 0.0) -> None:
        """Place `body` at a pose given in the SHELF's frame (xy) + world z, yawed to
        the shelf (+ optional roll about local x) — probe constructor robust to the
        shelf's per-episode yaw/xy jitter."""
        sp, syaw = shelf_pose()
        wx = float(sp[0]) + math.cos(syaw) * x_l - math.sin(syaw) * y_l
        wy = float(sp[1]) + math.sin(syaw) * x_l + math.cos(syaw) * y_l
        cy, sy = math.cos(syaw / 2), math.sin(syaw / 2)
        cr, sr = math.cos(math.radians(roll_deg) / 2), math.sin(math.radians(roll_deg) / 2)
        # q = qz(yaw) * qx(roll)
        place_world(body, wx, wy, z, (cy * cr, cy * sr, sy * sr, sy * cr))

    def catch_xy() -> tuple[float, float, float]:
        cl = scene._catch_local[0]
        return float(cl[0]), float(cl[1]), float(scene._side[0])

    def stage_basket() -> None:
        """Construct the basket correctly staged: upright on the ground, centred at
        the catch point, aligned to the shelf yaw."""
        cx, cy_, _side = catch_xy()
        place_local(scene.basket, cx, cy_, 0.002)

    def drop_into_basket(body) -> None:
        """Drop `body` from just above the basket mouth so it genuinely FALLS into
        the interior (this is how a probe constructs real containment)."""
        bp = (scene.basket.data.root_pos_w - scene.env_origins)[0]
        q = scene.basket.data.root_quat_w[0]
        byaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        place_world(body, float(bp[0]), float(bp[1]), 0.30,
                    (math.cos(byaw / 2), 0.0, 0.0, math.sin(byaw / 2)))

    def states_finite() -> bool:
        return bool(torch.isfinite(scene.target.data.root_state_w).all()
                    and torch.isfinite(scene.decoy.data.root_state_w).all()
                    and torch.isfinite(scene.basket.data.root_state_w).all()
                    and torch.isfinite(scene.shelf.data.root_state_w).all())

    def all_still() -> bool:
        return (float(scene.target.data.root_lin_vel_w[0].norm()) < c.settle_speed
                and float(scene.decoy.data.root_lin_vel_w[0].norm()) < c.settle_speed
                and float(scene.basket.data.root_lin_vel_w[0].norm()) < c.settle_speed)

    def geom_inside() -> bool:
        """Geometric containment ONLY (xy + z band + upright, catch station ignored)
        — used to verify a probe really constructed the state it claims before
        asserting the rubric rejects it."""
        loc = scene._basket_local(scene.target)[0]
        return (abs(float(loc[0])) < c.inside_xy_max
                and abs(float(loc[1])) < c.inside_xy_max
                and c.inside_z_min < float(loc[2]) < c.inside_z_max
                and bool(scene._basket_upright()[0]))

    ground_z = c.carton_h / 2 + 0.002  # carton standing on the ground
    in_basket_z = c.basket_wall_t + c.carton_h / 2 + 0.002  # carton at rest on the floor slab

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    bl = scene._shelf_local(scene.basket)[0]
    d_catch = float((bl[:2] - scene._catch_local[0]).norm())
    check("settle: states finite, both cartons standing on the slab, basket upright on "
          "open ground far from the catch point, everything still",
          states_finite() and bool(scene._on_shelf(scene.target)[0])
          and bool(scene._on_shelf(scene.decoy)[0])
          and bool(scene._basket_upright()[0]) and d_catch > 0.25 and all_still())
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(2)
        sp, syaw = shelf_pose()
        tl = scene._shelf_local(scene.target)[0]
        bp = (scene.basket.data.root_pos_w - scene.env_origins)[0]
        q = scene.target.data.root_quat_w[0]
        tyaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        rel_yaw = math.degrees(math.atan2(math.sin(tyaw - syaw), math.cos(tyaw - syaw)))
        side_pos = 1.0 if float(scene._side[0]) > 0 else 0.0
        bside_pos = 1.0 if float(bp[1]) > 0 else 0.0
        reads.append((float(sp[0]), float(sp[1]), math.degrees(syaw), float(tl[0]),
                      rel_yaw, side_pos, float(bp[0]), bside_pos))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (shelf_x, shelf_y, shelf_yaw_deg, "
          f"target_slot_x, target_rel_yaw_deg, target_side_pos_y, basket_x, "
          f"basket_side_pos_y):\n{arr}", flush=True)
    check("randomization: target slab side flips AND basket ground side flips across "
          "seeded resets (readback)",
          0.0 < arr[:, 5].mean() < 1.0 and 0.0 < arr[:, 7].mean() < 1.0)
    shelf_var = (float(arr[:, 2].max() - arr[:, 2].min()) > 2.0
                 and float((arr[:, 0:2].max(axis=0) - arr[:, 0:2].min(axis=0)).max()) > 0.008)
    obj_var = (float(arr[:, 3].max() - arr[:, 3].min()) > 0.01
               and float(arr[:, 4].max() - arr[:, 4].min()) > 20.0
               and float(arr[:, 6].max() - arr[:, 6].min()) > 0.01)
    check("randomization: shelf yaw + xy, target slot x + relative yaw, basket x all "
          "vary (readback)", shelf_var and obj_var)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy: item into the spawned basket =============
    # The seed's plan — pick the item and place it in the basket where it stands —
    # produces containment in a basket that never moved. Geometric containment is
    # verified TRUE; the catch-station clause + fell-in gate reject it anyway.
    torch.manual_seed(41)
    env.reset()
    step(10)
    drop_into_basket(scene.target)
    step(80)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy: carton dropped into the never-moved basket at its spawn — "
          "geometric containment TRUE yet not 'inside' (basket not at the catch "
          "station), caught never sets, no success, score <= 0.05",
          geom_inside() and not bool(scene._target_inside_now()[0])
          and not bool(scene._caught[0]) and not ok and s <= 0.05)

    # =========================== 7. staging gate: upside-down basket ========================
    torch.manual_seed(51)
    env.reset()
    step(10)
    cx, cy_, side = catch_xy()
    place_local(scene.basket, cx, cy_, 0.13, roll_deg=180.0)
    step(60)
    report("flipped-basket")
    s, ok = judge()
    check("staging gate: basket dumped UPSIDE-DOWN at the catch point (carton still "
          "shelved) — upright clause refuses the staged latch, no success, "
          "score <= 0.20 (approach only)",
          not bool(scene._basket_upright()[0]) and not bool(scene._staged[0])
          and not ok and s <= 0.20)

    # =========================== 8. near-miss: overshot the basket ==========================
    torch.manual_seed(61)
    env.reset()
    step(10)
    stage_basket()
    step(30)
    cx, cy_, side = catch_xy()
    place_local(scene.target, cx, cy_ + side * 0.22, ground_z)
    step(60)
    report("overshot")
    s, ok = judge()
    tz = float((scene.target.data.root_pos_w - scene.env_origins)[0, 2])
    check("near-miss drop: basket staged, carton grounded just OUTBOARD of the basket "
          "(overshot push) — not inside, no success, score <= 0.45 (staged credit "
          "only)", tz < 0.10 and bool(scene._staged[0])
          and not bool(scene._target_inside_now()[0]) and not ok and s <= 0.45)

    # =========================== 9. capping: basket lowered over the grounded carton ========
    # Out-of-order recovery: the carton is already on the ground at the catch point
    # (pushed off with no basket below), then the basket is lowered down over it.
    # Geometric containment at the catch station becomes TRUE — only the fell-in
    # gate stands between this and success, and it must reject.
    torch.manual_seed(71)
    env.reset()
    step(10)
    cx, cy_, side = catch_xy()
    place_local(scene.target, cx, cy_, ground_z)
    step(40)
    place_local(scene.basket, cx, cy_, 0.002)
    step(60)
    report("capped")
    s, ok = judge()
    check("capping: basket lowered OVER the carton grounded at the catch point — "
          "containment now geometrically TRUE at the catch station, but the carton "
          "never FELL in: caught False, no success, score <= 0.20",
          bool(scene._target_inside_now()[0]) and not bool(scene._staged[0])
          and not bool(scene._caught[0]) and not ok and s <= 0.20)

    # =========================== 10. capped after a legitimate stage ========================
    # Staged first (latch legitimately set with the carton still shelved), then the
    # carton appears AT REST in the basket interior — the end state of a missed drop
    # followed by lift-and-recap. Staged TRUE + containment TRUE, still rejected.
    torch.manual_seed(81)
    env.reset()
    step(10)
    stage_basket()
    step(30)
    cx, cy_, side = catch_xy()
    place_local(scene.target, cx, cy_, in_basket_z)
    step(60)
    report("recapped")
    s, ok = judge()
    check("capped after stage: staged latch set, then the carton placed AT REST in "
          "the interior (missed-drop + re-cap recovery) — containment TRUE, staged "
          "TRUE, caught still False, no success, score <= 0.45",
          bool(scene._target_inside_now()[0]) and bool(scene._staged[0])
          and not bool(scene._caught[0]) and not ok and s <= 0.45)

    # =========================== 11. wrong object into the basket ===========================
    torch.manual_seed(91)
    env.reset()
    step(10)
    stage_basket()
    step(30)
    drop_into_basket(scene.decoy)
    step(80)
    report("wrong-carton")
    s, ok = judge()
    dl = scene._basket_local(scene.decoy)[0]
    decoy_in = (abs(float(dl[0])) < c.inside_xy_max and abs(float(dl[1])) < c.inside_xy_max
                and float(dl[2]) < c.inside_z_max)
    check("wrong object: WHITE decoy dropped into the staged basket, ORANGE still "
          "shelved — decoy verified in the basket, target not inside, decoy off the "
          "shelf, no success, score <= 0.45",
          decoy_in and not bool(scene._target_inside_now()[0])
          and not bool(scene._on_shelf(scene.decoy)[0])
          and bool(scene._on_shelf(scene.target)[0]) and not ok and s <= 0.45)

    # =========================== 12. decoy clause isolation =================================
    # Decoy knocked to the ground, then the TARGET genuinely dropped into the staged
    # basket (the fell-in latch verifiably SETS): every target clause holds and the
    # decoy-on-shelf clause ALONE rejects.
    torch.manual_seed(101)
    env.reset()
    step(10)
    stage_basket()
    step(30)
    place_world(scene.decoy, 0.15, -0.55, ground_z)
    step(30)
    drop_into_basket(scene.target)
    step(100)
    report("decoy-down")
    s, ok = judge()
    check("decoy clause: target genuinely FELL into the staged basket (caught latch "
          "verified set, containment TRUE) but the WHITE carton is off the shelf — "
          "that clause alone rejects, no success, score <= 0.85",
          bool(scene._caught[0]) and bool(scene._target_inside_now()[0])
          and not bool(scene._on_shelf(scene.decoy)[0]) and not ok and s <= 0.85)

    # =========================== 13. latched credit survives regression =====================
    torch.manual_seed(111)
    env.reset()
    step(10)
    stage_basket()
    step(30)
    s_a, ok_a = judge()
    place_world(scene.basket, 0.10, 0.55, 0.002)  # yank the basket away again
    step(60)
    report("regressed")
    s_b, ok_b = judge()
    check("latched credit: staging credit earned then the basket yanked away — score "
          f"holds ({s_a:.3f} -> {s_b:.3f}), >= 0.35, still no success",
          bool(scene._staged[0]) and abs(s_a - s_b) < 1e-3 and s_a >= 0.35
          and not ok_a and not ok_b)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", states_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.ledge_catch")
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

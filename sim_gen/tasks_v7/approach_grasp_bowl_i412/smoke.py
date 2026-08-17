"""Smoke / rubric-REJECTION battery for ChimneyCatchScene (sim_gen task
`approach_grasp_bowl_i412`) — NullRobot, constructed probe states, RECORDED.

This is NOT a solution (solve.py — cover the live slot with the bowl, push the gate
knob, catch the dropping ball, deliver the loaded bowl to the dock — is the acceptance
evidence that the rubric ACCEPTS a correct outcome). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome and asserts the rubric
REJECTS it; the two force probes assert the actuator actually moved (no vacuous physical
checks); no probe in this battery ever reaches success(), and a final audit check
asserts exactly that.

  1.  settle/no-NaN       — reset settles finite: ball resting on the LIVE gate blade
                            inside the chute, bowl at deck rest height; score ~0;
  2-3. randomization      — READBACK over 8 seeded resets: live side flips both ways
                            and the ball follows the live tower; dock disc (the physical
                            body) teleports to its internal target and varies; bowl spawn
                            xy + yaw vary; latches zeroed;
  4.  null policy         — 300 idle steps -> score ~0, no success, no creep (bowl,
                            ball, blade all hold station);
  5.  trigger-first       — the IRREVERSIBLE branch, forced PHYSICALLY: the live blade
                            is pushed open by force (assert it moved) with the slot
                            uncovered -> the ball falls through the deck into the sealed
                            plenum; `lost` latches; score ~0;
  6.  lost-then-docked    — continue: docking the bowl afterwards is refused — docked
                            reads True but NOT success and the lost cap holds (<= 0.25);
  7.  dead-gate no-op     — the DEAD tower's blade opened fully: nothing falls, the ball
                            stays on the live blade, score ~0;
  8.  wrong-slot cover    — bowl covering the DEAD slot: the covered latch (live-slot
                            specific) stays 0;
  9.  caught-not-docked   — bowl covers the live slot (covered fires), live gate opened,
                            ball drops in (caught fires): score == 0.60 cap, NOT success;
  10. caught latched      — removing the caught ball onto the deck leaves the latched
                            0.60 and success stays gone (ball on deck never reads lost);
  11. lost cap dominates  — then dumping the ball into the plenum drops the score to the
                            0.25 lost cap despite both latches;
  12. docked-empty        — the SEED strategy end state (bowl carried to the goal disc,
                            no ball): docked True, score ~0, NOT success;
  13. ball-beside-bowl    — ball on the deck leaning distance from the docked bowl: NOT
                            in_bowl, NOT success;
  14. dock near-miss      — loaded bowl parked just OUTSIDE dock_tol: in_bowl True but
                            docked False, NOT success;
  15. inboard hard stop   — the parked blade pushed INBOARD by force (assert it moved):
                            the knob-vs-backwall stop arrests it, the aperture stays
                            sealed, the ball stays on the blade and is never lost;
  16. rejection audit + final no-NaN.

Run (forge): python -u -m simgen_tasks.approach_grasp_bowl_i412.smoke --headless
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
    env = ENVS.get("simgen.chimney_catch")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero1 = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.30, -1.30, 1.05)) + o),
                                tuple(np.array((0.00, 0.05, 0.14)) + o),
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

    def bowl_p() -> torch.Tensor:
        return scene._bowl_pos()[0]

    def ball_p() -> torch.Tensor:
        return scene._ball_pos()[0]

    def blade_y(key: str) -> float:
        """Live-frame blade-centre y in the chute frame (0.020 = closed)."""
        return float((scene.blades[key].data.root_pos_w
                      - scene.env_origins)[0, 1]) - c.slot_y

    def side() -> float:
        return float(scene.side[0])

    def live() -> str:
        return "p" if side() > 0 else "n"

    def report(tag: str) -> None:
        wp, bp = bowl_p(), ball_p()
        s, ok = judge()
        print(f"[smoke] {tag:18s} | bowl=({float(wp[0]):+.3f},{float(wp[1]):+.3f},"
              f"{float(wp[2]):.3f}) ball=({float(bp[0]):+.3f},{float(bp[1]):+.3f},"
              f"{float(bp[2]):.3f}) blade_y[{live()}]={blade_y(live()) * 1000:.1f}mm "
              f"cov={float(scene.covered[0]):.0f} cau={float(scene.caught[0]):.0f} "
              f"lost={bool(scene.lost[0])} in_bowl={bool(scene._in_bowl()[0])} "
              f"docked={bool(scene._docked()[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_body(body, x: float, y: float, z: float, quat=(1.0, 0.0, 0.0, 0.0),
                   vel=(0.0, 0.0, 0.0), settle_steps: int = 30) -> None:
        """Probe placement (instrumentation, not a solution): pose AND velocity written
        together, then REAL physics steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 7], st[:, 8], st[:, 9] = vel
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def push_blade_open(key: str, max_steps: int = 600) -> bool:
        """Force-push a gate blade outboard with the solve's controller. Returns True
        once the aperture is fully open. Clears the wrench afterwards."""
        blade = scene.blades[key]
        opened = False
        for _ in range(max_steps):
            by = blade_y(key)
            if by >= c.blade_open_y + 0.003:
                opened = True
                break
            vy = blade.data.root_lin_vel_w[:, 1]
            fy = (0.35 + c.blade_mass * 80.0 * (0.06 - vy)).clamp(min=-0.4, max=1.5)
            f = torch.zeros(n, 3, device=device)
            f[:, 1] = fy
            blade.set_external_force_and_torque(
                f.unsqueeze(1), zero1, env_ids=all_ids, is_global=True)
            env.step(no_action)
        blade.set_external_force_and_torque(zero1, zero1, env_ids=all_ids)
        return opened

    bowl_rest_z = c.deck_h            # bowl origin = base BOTTOM -> rests at deck top
    ball_deck_z = c.deck_h + c.ball_r  # ball resting on the deck
    ball_in_bowl_z = c.deck_h + c.base_t + c.ball_r  # ball resting on the bowl floor
    blade_open_park = c.slot_y + 0.098  # open blade centre y: aperture open, tip clear
    #                                     of the end stop (tip 0.153 < stop face 0.155)

    # =========================== 1. settle / no-NaN ==========================================
    env.reset(seed=11)
    step(90)
    report("reset-settled")
    wp, bp = bowl_p(), ball_p()
    fin0 = bool(torch.isfinite(scene.bowl.data.root_state_w).all()
                and torch.isfinite(scene.ball.data.root_state_w).all()
                and all(torch.isfinite(b.data.root_state_w).all()
                        for b in scene.blades.values()))
    s, ok = judge()
    check("settle: states finite; ball resting ON the live gate blade inside the chute "
          "and bowl at deck rest height (readback); score ~0, no success",
          fin0 and abs(float(bp[2]) - c.ball_blade_z) < 0.008
          and abs(float(bp[0]) - side() * c.slot_x) < 0.012
          and abs(float(wp[2]) - bowl_rest_z) < 0.008
          and s <= 0.02 and not ok)

    # =========================== 2-3. randomization is real ==================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(5)
        bp = ball_p()
        wp = bowl_p()
        q = scene.bowl.data.root_quat_w[0]
        yaw = float(torch.atan2(2 * (q[0] * q[3] + q[1] * q[2]),
                                1 - 2 * (q[2] * q[2] + q[3] * q[3])))
        dock_body = (scene.dock.data.root_pos_w - scene.env_origins)[0]
        latches0 = (float(scene.covered[0]) == 0.0 and float(scene.caught[0]) == 0.0
                    and not bool(scene.lost[0]))
        reads.append((side(), float(bp[0]), float(wp[0]), float(wp[1]), yaw,
                      float(scene.dock_xy[0, 0]), float(scene.dock_xy[0, 1]),
                      float(dock_body[0]), float(dock_body[1]), float(latches0)))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (side, ball_x, bowl_x, bowl_y, yaw, "
          f"dock_tx, dock_ty, dock_bx, dock_by, latches0):\n{arr}", flush=True)
    sides = set(arr[:, 0].tolist())
    ball_follows = all(abs(r[1] - r[0] * c.slot_x) < c.ball_jitter + 0.006 for r in reads)
    dock_body_ok = all(abs(r[7] - r[5]) < 0.002 and abs(r[8] - r[6]) < 0.002
                       for r in reads)
    dock_in_zone = all(abs(r[5]) <= c.dock_x + 1e-6
                       and c.dock_y_lo - 1e-6 <= r[6] <= c.dock_y_hi + 1e-6
                       for r in reads)
    check("randomization: live side flips BOTH ways across seeds, the ball follows the "
          "live tower, and the physical dock disc teleports to its sampled target "
          "inside the front zone (readback)",
          sides == {1.0, -1.0} and ball_follows and dock_body_ok and dock_in_zone)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: dock xy, bowl spawn xy and bowl yaw all vary across seeds; "
          "latches zeroed on every reset (readback)",
          spread[5] > 0.05 and spread[6] > 0.015 and spread[2] > 0.02
          and spread[3] > 0.015 and spread[4] > 0.5 and bool(arr[:, 9].all()))

    # =========================== 4. null policy fails ========================================
    env.reset(seed=31)
    step(60)
    wp0, bp0 = bowl_p()[:2].clone(), ball_p()[:2].clone()
    by0 = blade_y(live())
    step(300)
    report("null-policy")
    drift_bowl = float((bowl_p()[:2] - wp0).norm())
    drift_ball = float((ball_p()[:2] - bp0).norm())
    drift_blade = abs(blade_y(live()) - by0)
    s, ok = judge()
    check("null policy: score ~0 and no success after 300 idle steps; bowl, ball and "
          f"blade all hold station (drift {drift_bowl * 1000:.1f}/"
          f"{drift_ball * 1000:.1f}/{drift_blade * 1000:.1f} mm)",
          s <= 0.02 and not ok and drift_bowl < 0.01 and drift_ball < 0.03
          and drift_blade < 0.01)

    # =========================== 5. trigger-first -> ball lost (physical) ====================
    env.reset(seed=41)
    step(60)
    lv = live()
    by0 = blade_y(lv)
    opened = push_blade_open(lv)
    step(240)  # ball falls through the uncovered slot into the sealed plenum
    report("trigger-first")
    bp = ball_p()
    s, ok = judge()
    check("trigger-first: live gate pushed open BY FORCE with the slot uncovered "
          f"(blade moved {(blade_y(lv) - by0) * 1000:.0f} mm — non-vacuous) -> ball "
          "falls through the deck into the sealed plenum: lost latches, score ~0",
          opened and blade_y(lv) - by0 > 0.050 and float(bp[2]) < c.lost_z
          and bool(scene.lost[0]) and s <= 0.02 and not ok)

    # =========================== 6. lost-then-docked refused =================================
    dock_xy = scene.dock_xy[0]
    place_body(scene.bowl, float(dock_xy[0]), float(dock_xy[1]), bowl_rest_z + 0.002,
               settle_steps=90)
    report("lost-then-docked")
    s, ok = judge()
    check("lost-then-docked: docking the bowl AFTER the ball is sealed in the plenum "
          "is refused — docked reads True yet NOT success, score capped (<= 0.25)",
          bool(scene._docked()[0]) and bool(scene.lost[0]) and not ok
          and s <= 0.25 + 1e-5)

    # =========================== 7-8. dead gate + wrong-slot cover ==========================
    env.reset(seed=51)
    step(60)
    lv, dead = live(), ("n" if side() > 0 else "p")
    place_body(scene.blades[dead], -side() * c.slot_x, blade_open_park, c.blade_z,
               settle_steps=120)
    report("dead-gate")
    bp = ball_p()
    s, ok = judge()
    check("dead-gate no-op: the DEAD tower's blade opened fully — nothing falls, the "
          "ball stays on the live blade, score ~0",
          abs(float(bp[2]) - c.ball_blade_z) < 0.008 and not bool(scene.lost[0])
          and s <= 0.02 and not ok)
    place_body(scene.bowl, -side() * c.slot_x, c.slot_y, bowl_rest_z + 0.002,
               settle_steps=90)
    report("wrong-slot-cover")
    s, ok = judge()
    check("wrong-slot cover: bowl parked exactly over the DEAD slot — the covered "
          "latch is live-slot specific and stays 0 (score ~0)",
          float(scene.covered[0]) == 0.0 and s <= 0.02 and not ok)

    # =========================== 9. caught-not-docked (score cap) ===========================
    env.reset(seed=61)
    step(60)
    lv = live()
    place_body(scene.bowl, side() * c.slot_x, c.slot_y, bowl_rest_z + 0.002,
               settle_steps=90)  # covered latch fires here
    place_body(scene.blades[lv], side() * c.slot_x, blade_open_park, c.blade_z,
               settle_steps=300)  # ball drops through the chute into the bowl
    report("caught-not-docked")
    s9, ok = judge()
    check("caught-not-docked: covered + caught both latched (ball landed in the "
          "covering bowl) -> score == 0.60 cap (float32), NOT success",
          bool(scene._in_bowl()[0]) and float(scene.covered[0]) == 1.0
          and float(scene.caught[0]) == 1.0 and 0.599 <= s9 <= 0.60 + 1e-6 and not ok)

    # =========================== 10. caught credit is latched ================================
    place_body(scene.ball, -side() * 0.05, -0.20, ball_deck_z + 0.002, settle_steps=90)
    report("ball-removed")
    s10, ok = judge()
    bp = ball_p()
    check("caught latched: removing the caught ball onto the open deck keeps the "
          f"latched credit ({s9:.2f} -> {s10:.2f}), ball on deck never reads lost, "
          "success stays gone",
          abs(s10 - s9) < 0.005 and not bool(scene._in_bowl()[0])
          and not bool(scene.lost[0]) and float(bp[2]) > c.lost_z and not ok)

    # =========================== 11. lost cap dominates latched credit =======================
    place_body(scene.ball, 0.0, 0.0, 0.030, settle_steps=30)  # into the sealed plenum
    report("lost-cap")
    s11, ok = judge()
    check("lost cap dominates: dumping the ball into the plenum afterwards drops the "
          f"score from the latched 0.60 to the 0.25 lost cap ({s11:.2f})",
          bool(scene.lost[0]) and s11 <= 0.25 + 1e-5 and not ok)

    # =========================== 12. docked-empty (the seed strategy) ========================
    env.reset(seed=71)
    step(60)
    dock_xy = scene.dock_xy[0]
    place_body(scene.bowl, float(dock_xy[0]), float(dock_xy[1]), bowl_rest_z + 0.002,
               settle_steps=90)
    report("docked-empty")
    s, ok = judge()
    check("docked-empty: the seed's strategy end state (bowl carried to the goal disc, "
          "no ball) — docked True yet score ~0, NOT success",
          bool(scene._docked()[0]) and s <= 0.02 and not ok)

    # =========================== 13. ball beside the docked bowl =============================
    bx = float(dock_xy[0])
    off = -0.13 if bx > 0 else 0.13  # towards the deck centre, clear of the curb
    place_body(scene.ball, bx + off, float(dock_xy[1]), ball_deck_z + 0.002,
               settle_steps=90)
    report("ball-beside-bowl")
    s, ok = judge()
    bp, wp = ball_p(), bowl_p()
    d_xy = float((bp[:2] - wp[:2]).norm())
    check("ball-beside-bowl: ball settled on the deck beside the docked bowl "
          f"(d_xy={d_xy * 1000:.0f} mm) — NOT in_bowl, NOT success",
          not bool(scene._in_bowl()[0]) and d_xy > c.in_bowl_r + 0.02 and not ok)

    # =========================== 14. dock near-miss ==========================================
    env.reset(seed=81)
    step(60)
    dock_xy = scene.dock_xy[0]
    bx = float(dock_xy[0])
    miss = c.dock_tol + 0.012
    off = -miss if bx > 0 else miss  # towards the deck centre, clear of the curb
    place_body(scene.bowl, bx + off, float(dock_xy[1]), bowl_rest_z + 0.002,
               settle_steps=60)
    place_body(scene.ball, bx + off, float(dock_xy[1]), ball_in_bowl_z + 0.010,
               settle_steps=180)  # dropped into the bowl interior; caught latches
    report("dock-near-miss")
    s, ok = judge()
    wp = bowl_p()
    d_dock = float((wp[:2] - dock_xy).norm())
    check("dock near-miss: loaded bowl parked just OUTSIDE dock_tol "
          f"(d={d_dock * 1000:.0f} mm) — in_bowl True but docked False, NOT success",
          bool(scene._in_bowl()[0]) and not bool(scene._docked()[0])
          and d_dock > c.dock_tol and not ok and s <= 0.36)

    # =========================== 15. inboard hard stop (physical) ============================
    env.reset(seed=91)
    step(60)
    lv = live()
    blade = scene.blades[lv]
    by0 = blade_y(lv)
    f = torch.zeros(n, 3, device=device)
    f[:, 1] = -0.8  # inboard shove, ~13x the blade's sliding friction
    blade.set_external_force_and_torque(f.unsqueeze(1), zero1, env_ids=all_ids,
                                        is_global=True)
    step(240)
    blade.set_external_force_and_torque(zero1, zero1, env_ids=all_ids)
    step(60)
    report("inboard-stop")
    by1 = blade_y(lv)
    bp = ball_p()
    s, ok = judge()
    check("inboard hard stop: the live blade shoved INBOARD moves "
          f"({(by0 - by1) * 1000:.1f} mm — non-vacuous) but the knob-vs-backwall stop "
          "arrests it with the aperture still sealed: ball stays on the blade, never "
          "lost",
          by0 - by1 > 0.004 and by0 - by1 < 0.020
          and abs(float(bp[2]) - c.ball_blade_z) < 0.010
          and not bool(scene.lost[0]) and s <= 0.02 and not ok)

    # =========================== 16. audit + no-NaN ==========================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.bowl.data.root_state_w).all()
           and torch.isfinite(scene.ball.data.root_state_w).all()
           and torch.isfinite(scene.dock.data.root_state_w).all()
           and all(torch.isfinite(b.data.root_state_w).all()
                   for b in scene.blades.values()))
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict ==============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.chimney_catch")
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

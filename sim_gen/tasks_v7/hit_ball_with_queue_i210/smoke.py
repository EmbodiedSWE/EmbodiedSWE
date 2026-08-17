"""Smoke / rubric-REJECTION battery for BallCorralScene (sim_gen task
`hit_ball_with_queue_i210`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — cage dropped over the ball under gravity, the
caged assembly force-slid to the goal disc — is the acceptance evidence that the
rubric ACCEPTS a correct outcome). Every teleport here is instrumentation that
CONSTRUCTS a wrong (or partial) outcome as a settled state and asserts the rubric
REJECTS it; no probe in this battery ever reaches success(), and a final audit
check asserts exactly that.

  1-2. settle/no-NaN     — reset layout settles finite: both balls at cradle rest
                           height, cage parked flush, all still, score ~0;
  3-4. randomization     — READBACK over 8 seeded resets: plateau yaw + xy + TILT
                           spreads are real; white/decoy cradle swap flips
                           (Bernoulli); goal zone moves and the marker disc tracks
                           the stored zone; cage parking jitter + ball in-cradle
                           jitter are real;
  5.  null policy        — 240 idle steps -> score ~0, no success;
  6.  seed strategy      — the end state the SEED's plan produces here (STRIKE the
                           ball): white fired downhill at 1.4 m/s pops its cradle
                           lip, rolls off the open edge and falls to the floor ->
                           unrecoverable, score ~0 (striking is the losing move);
  7.  bare ball at goal  — even the CORRECT destination cannot hold a bare ball:
                           white gently placed at rest ON the goal zone rolls
                           downhill and off the edge -> no rest, score ~0 (only a
                           contained arrival can ever be at rest there);
  8.  empty container    — the cage alone delivered flush onto the goal zone (real
                           flush + in-zone readback) with the white still in its
                           cradle -> no caged, no delivered, score ~0;
  9.  wrong object       — the DECOY trapped under the cage flush ON the goal zone
                           (real flush + inside(decoy) + in-zone readback) ->
                           caged() demands the WHITE ball: score ~0, no success;
  10. near-miss capture  — cage dropped over the white 75 mm off-axis (beyond the
                           dome-funnel forgiveness of the seating): the wall meets
                           the ball's dome past its apex and sheds off the shoulder
                           away from it -> never caged;
  11. roof gate          — white dropped ON TOP of the closed cage: xy near the
                           axis but never at deck height inside -> the z-band gate
                           holds, never caged;
  12. inverted container — cage placed MOUTH-UP at the goal and the white dropped
                           into the upturned bowl: ball held BY the cage at the
                           zone, but the flush up-axis gate rejects the inverted
                           cage -> never caged, no success (container orientation
                           is load-bearing);
  13. latched credit     — cage probe-dropped over the white IN ITS CRADLE (far
                           from the zone): capture credit is exactly 0.30, no
                           delivery credit at 22+ cm from the goal; cage then
                           teleported away -> latched 0.30 survives, never success
                           (non-success cap holds);
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
    env = ENVS.get("simgen.ball_corral")().build(num_envs=args.num_envs,
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
        env.sim.set_camera_view(tuple(np.array((1.30, -1.20, 1.05)) + o),
                                tuple(np.array((0.00, 0.00, 0.24)) + o),
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

    def zone_dist() -> float:
        g = scene._local(scene.cage)[0]
        return float((g[0:2] - scene._zone[0]).norm())

    def report(tag: str) -> None:
        wl = scene._local(scene.white)[0]
        dl = scene._local(scene.decoy)[0]
        gl = scene._local(scene.cage)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | white=({float(wl[0]):+.3f},{float(wl[1]):+.3f},"
              f"{float(wl[2]):+.3f}) decoy=({float(dl[0]):+.3f},{float(dl[1]):+.3f},"
              f"{float(dl[2]):+.3f}) cage=({float(gl[0]):+.3f},{float(gl[1]):+.3f},"
              f"{float(gl[2]):+.3f}) zone_d={zone_dist():.3f} "
              f"flush={bool(scene._cage_flush()[0])} caged={bool(scene.caged()[0])} "
              f"caged_ever={bool(scene._caged_ever[0])} "
              f"delivered_ever={bool(scene._delivered_ever[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_local(body, x: float, y: float, z: float, *, vel_x: float = 0.0,
                    roll: float = 0.0) -> None:
        """Teleport `body` to a plateau-local point of the plateau's CURRENT pose
        (probe constructor: builds hover/at-goal/on-roof relations directly),
        optionally with an initial velocity along local +x (downhill — the seed's
        strike) and/or a roll about local x (the inverted-cage probe)."""
        p_pos = scene.plateau.data.root_pos_w
        p_quat = scene.plateau.data.root_quat_w
        loc = torch.tensor([x, y, z], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = p_pos + quat_apply(p_quat, loc)
        if roll:
            h = roll / 2
            qr = torch.tensor([math.cos(h), math.sin(h), 0.0, 0.0],
                              device=device).expand(n, 4)
            st[:, 3:7] = quat_mul(p_quat, qr)
        else:
            st[:, 3:7] = p_quat
        if vel_x:
            axis = quat_apply(p_quat, torch.tensor([1.0, 0.0, 0.0],
                                                   device=device).expand(n, 3))
            st[:, 7:10] = axis * vel_x
        body.write_root_state_to_sim(st, all_ids)

    def world_z(body) -> float:
        """World height above the env origin — the robust 'fell to the floor' probe
        (a ball far downrange sits HIGH in the tilted plateau-local frame: local z
        gains x*sin(tilt), ~+90 mm at 2 m, so a local-z floor test is unsound)."""
        return float((body.data.root_pos_w - scene.env_origins)[0, 2])

    def finite_all() -> bool:
        return bool(torch.isfinite(scene.plateau.data.root_state_w).all()
                    and torch.isfinite(scene.cage.data.root_state_w).all()
                    and torch.isfinite(scene.white.data.root_state_w).all()
                    and torch.isfinite(scene.decoy.data.root_state_w).all())

    def plateau_pose() -> tuple[float, float, float, float]:
        p = (scene.plateau.data.root_pos_w - scene.env_origins)[0]
        q = scene.plateau.data.root_quat_w
        ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
        up = quat_apply(q, ez)[0]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(up[2])))))
        yaw = math.degrees(2.0 * math.atan2(float(q[0, 3]), float(q[0, 0])))
        return (float(p[0]), float(p[1]), yaw, tilt)

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("show")
    wl = scene._local(scene.white)[0]
    dl = scene._local(scene.decoy)[0]
    gl = scene._local(scene.cage)[0]
    still = (float(scene.white.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.decoy.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.cage.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    check("settle: states finite, both balls at cradle rest height (the tilt cannot "
          "dislodge them over the 4 mm lip), cage parked flush on the deck, all still",
          finite_all()
          and abs(float(wl[2]) - c.ball_rest_z) < 0.008
          and abs(float(dl[2]) - c.ball_rest_z) < 0.008
          and abs(float(gl[2])) < 0.010 and still)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        px, py, pyaw, ptilt = plateau_pose()
        wl = scene._local(scene.white)[0]
        gl = scene._local(scene.cage)[0]
        ml = scene._local(scene.marker)[0]
        zn = scene._zone[0]
        marker_match = float((ml[0:2] - zn).norm())
        reads.append((px, py, pyaw, ptilt, float(wl[0]), float(wl[1]),
                      1.0 if float(wl[1]) > 0 else 0.0, float(gl[0]), float(gl[1]),
                      float(zn[0]), float(zn[1]), marker_match))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (px, py, yaw_deg, tilt_deg, white_x, "
          f"white_y, white_in_+y_cradle, cage_x, cage_y, zone_x, zone_y, "
          f"marker_zone_err):\n{np.array2string(arr, precision=4)}", flush=True)
    yaw_spread = float(arr[:, 2].max() - arr[:, 2].min())
    xy_spread = float((arr[:, 0:2].max(axis=0) - arr[:, 0:2].min(axis=0)).max())
    tilt_spread = float(arr[:, 3].max() - arr[:, 3].min())
    check("randomization: plateau yaw spread (> 2 deg), xy offset spread (> 4 mm) "
          "and TILT spread (> 0.15 deg) are real (readback)",
          yaw_spread > 2.0 and xy_spread > 0.004 and tilt_spread > 0.15)
    flags = arr[:, 6]
    ball_jit = float(arr[:, 4].max() - arr[:, 4].min())
    cage_jit = float((arr[:, 7:9].max(axis=0) - arr[:, 7:9].min(axis=0)).max())
    zone_spread = float((arr[:, 9:11].max(axis=0) - arr[:, 9:11].min(axis=0)).max())
    check("randomization: white/decoy cradle swap flips across seeded resets, goal "
          "zone moves (> 10 mm) with the marker disc tracking the stored zone "
          "(< 5 mm err), cage jitter (> 4 mm) and in-cradle ball jitter (> 4 mm) "
          "are real (readback)",
          0.0 < flags.mean() < 1.0 and zone_spread > 0.010
          and float(arr[:, 11].max()) < 0.005 and cage_jit > 0.004
          and ball_jit > 0.004)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 6. seed strategy: STRIKE the ball ==========================
    # The seed's plan — impulse the ball toward the goal — executed here: the white
    # fired downhill at 1.4 m/s. It pops the 4 mm cradle lip, accelerates down the
    # tilted deck, rolls off the open edge and falls 24 cm to the floor:
    # unrecoverable (85 mm ball > 80 mm jaw span; the plateau face is a cliff).
    torch.manual_seed(41)
    env.reset()
    step(30)
    wl = scene._local(scene.white)[0]
    place_local(scene.white, float(wl[0]), float(wl[1]), c.ball_rest_z + 0.001,
                vel_x=1.4)
    step(360)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy: white STRUCK downhill at 1.4 m/s pops the cradle lip and "
          "leaves the plateau (world height < 0.10 m — on the floor; deck rest is "
          "~0.28 m) — score ~0 (<= 0.02), no success (striking is the losing move)",
          world_z(scene.white) < 0.10 and not bool(scene._caged_ever[0])
          and s <= 0.02 and not ok)

    # =========================== 7. bare ball at the goal ===================================
    # Even the CORRECT destination cannot hold an uncontained ball: the white gently
    # placed AT REST on the goal half rolls downhill (between the cradle bands) and
    # off the open edge. Only a contained arrival can ever rest in the zone.
    torch.manual_seed(51)
    env.reset()
    step(30)
    place_local(scene.white, -0.20, 0.0, c.ball_r + 0.002)
    step(420)
    report("bare-at-goal")
    s, ok = judge()
    check("bare ball at goal: white placed at rest ON the goal half (bare tilted "
          "deck) rolls downhill and off the edge (world height < 0.10 m — on the "
          "floor) — a bare ball can NEVER rest at the goal: score ~0, no success",
          world_z(scene.white) < 0.10 and not bool(scene._caged_ever[0])
          and s <= 0.02 and not ok)

    # =========================== 8. empty container at the goal =============================
    # The cage alone delivered flush onto the goal zone — a REAL flush + in-zone
    # readback — while the white stays in its cradle: delivery credit demands the
    # ball enclosed, so nothing scores.
    torch.manual_seed(61)
    env.reset()
    step(30)
    zn = scene._zone[0]
    place_local(scene.cage, float(zn[0]), float(zn[1]), 0.030)
    step(300)
    report("empty-cage")
    s, ok = judge()
    check("empty container: cage settled flush ON the goal zone (real flush + "
          "in-zone readback) with the white still in its cradle — no capture, no "
          "delivery credit, score ~0, no success",
          bool(scene._cage_flush()[0]) and bool(scene.in_zone()[0])
          and not bool(scene.caged()[0]) and not bool(scene._caged_ever[0])
          and s <= 0.02 and not ok)

    # =========================== 9. wrong object: the decoy =================================
    # The DECOY trapped under the flush cage ON the goal zone — the full delivered
    # GEOMETRY with the wrong ball: caged() names the white, so nothing scores.
    torch.manual_seed(71)
    env.reset()
    step(30)
    zn = scene._zone[0]
    place_local(scene.decoy, float(zn[0]), float(zn[1]), c.ball_r + 0.002)
    place_local(scene.cage, float(zn[0]) + 0.004, float(zn[1]), 0.045)
    step(360)
    report("wrong-object")
    s, ok = judge()
    check("wrong object: DECOY trapped under the flush cage ON the goal zone (real "
          "flush + inside(decoy) + in-zone readback) — caged() demands the WHITE "
          "ball: score ~0, no success (identity is load-bearing)",
          bool(scene._cage_flush()[0]) and bool(scene._inside(scene.decoy)[0])
          and bool(scene.in_zone()[0]) and not bool(scene.caged()[0])
          and not bool(scene._caged_ever[0]) and s <= 0.02 and not ok)

    # =========================== 10. near-miss capture (lateral) ============================
    # Cage dropped over the white 75 mm off-axis. NOTE the honest geometry: for
    # moderate offsets (up to roughly the 60 mm wall apothem) the descending rim
    # lands on the NEAR slope of the ball's dome and gravity FUNNELS the cage into
    # a centred capture — the seating is deliberately tolerant of realistic
    # placement error. At 75 mm the wall meets the dome PAST its apex: the cage
    # sheds off the ball's shoulder away from it and seats beside the ball (or
    # knocks it loose) — it never reads caged.
    torch.manual_seed(81)
    env.reset()
    step(30)
    wl = scene._local(scene.white)[0]
    place_local(scene.cage, float(wl[0]) + 0.075, float(wl[1]), 0.045)
    step(360)
    report("near-miss-75mm")
    s, ok = judge()
    check("near-miss capture: cage dropped 75 mm off-axis (beyond the dome-funnel "
          "forgiveness) sheds off the ball's shoulder — never caged, no capture "
          "credit, score ~0, no success",
          not bool(scene._caged_ever[0]) and s <= 0.02 and not ok)

    # =========================== 11. roof gate (z-band honesty) =============================
    # White dropped ON TOP of the closed cage at its parking spot: its xy passes near
    # the cage axis but it is never AT DECK HEIGHT inside the walls — the z-band
    # gate must hold (a ball on the roof is not an enclosed ball).
    torch.manual_seed(91)
    env.reset()
    step(30)
    gl = scene._local(scene.cage)[0]
    place_local(scene.white, float(gl[0]), float(gl[1]), 0.24)
    step(300)
    report("roof-gate")
    s, ok = judge()
    check("roof gate: white dropped ON TOP of the closed cage (xy near the axis, "
          "z far above the deck band) — never reads caged: the z-band gate holds, "
          "score ~0, no success",
          not bool(scene._caged_ever[0]) and s <= 0.02 and not ok)

    # =========================== 12. inverted container =====================================
    # Cage placed MOUTH-UP at the goal, white dropped into the upturned bowl: the
    # cage HOLDS the ball at the zone — but the flush gate demands the cage's +z
    # aligned with the deck normal, so an inverted 'basket' can never read caged.
    torch.manual_seed(101)
    env.reset()
    step(30)
    zn = scene._zone[0]
    place_local(scene.cage, float(zn[0]), float(zn[1]), 0.150, roll=math.pi)
    step(120)
    gl = scene._local(scene.cage)[0]
    place_local(scene.white, float(gl[0]), float(gl[1]), 0.30)
    step(360)
    report("inverted-cage")
    wl = scene._local(scene.white)[0]
    gl = scene._local(scene.cage)[0]
    # Inverted, the cage's barred ROOF is the bowl floor: the ball rests on the bars
    # with its centre ~64 mm up — clearly above the 42.5 mm bare-deck rest, i.e.
    # held by the cage, not the deck.
    held_by_cage = (float(wl[2]) > c.ball_r + 0.015
                    and float((wl[0:2] - gl[0:2]).norm()) < 0.20)
    s, ok = judge()
    check("inverted container: white rests INSIDE the mouth-up cage bowl at the "
          "goal (held above the deck band by the cage — real readback) — the flush "
          "up-axis gate rejects the inverted cage: never caged, score ~0, no "
          "success (container orientation is load-bearing)",
          held_by_cage and not bool(scene._cage_flush()[0])
          and not bool(scene._caged_ever[0]) and s <= 0.02 and not ok)

    # =========================== 13. latched credit + regression ============================
    # Cage probe-dropped CENTRED over the white in its cradle (22+ cm from the goal):
    # capture credit latches at exactly 0.30, delivery never fires; the cage is then
    # teleported back to parking (the freed ball stays safe in its cradle) and the
    # latched 0.30 survives — never success (non-success cap holds).
    torch.manual_seed(111)
    env.reset()
    step(30)
    wl = scene._local(scene.white)[0]
    place_local(scene.cage, float(wl[0]) + 0.005, float(wl[1]) + 0.004, 0.045)
    caged = False
    for _ in range(480):
        env.step(no_action)
        if bool(scene.caged()[0]) and bool(scene._still()[0]):
            caged = True
            break
    report("probe-caged")
    s13a, ok = judge()
    got_credit = (caged and abs(s13a - c.w_cage) < 0.01 and not ok
                  and not bool(scene._delivered_ever[0]) and zone_dist() > 0.10)
    place_local(scene.cage, c.parking[0], c.parking[1], 0.003)
    step(90)
    report("regressed")
    s13b, ok = judge()
    check("latched credit: probe-caged white in its cradle earns exactly the "
          f"capture credit (0.30, got {s13a:.3f}) with NO delivery at 10+ cm from "
          f"the goal and NO success; teleporting the cage away leaves the latched "
          f"score unchanged ({s13a:.3f} -> {s13b:.3f}), still no success "
          "(non-success cap holds)",
          got_credit and abs(s13b - s13a) < 1e-3
          and not bool(scene.caged()[0]) and not ok)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.ball_corral")
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

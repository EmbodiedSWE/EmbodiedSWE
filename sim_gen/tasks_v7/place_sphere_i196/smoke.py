"""Smoke / rubric-REJECTION battery for DrainPlugScene (sim_gen task
`place_sphere_i196`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — ball hover-released upslope so gravity rolls it
into the drain seat, then each marble hover-released and rolled to rest against the
plug — is the acceptance evidence that the rubric ACCEPTS a correct outcome). Every
teleport here is instrumentation that CONSTRUCTS a wrong (or partial) outcome as a
settled state and asserts the rubric REJECTS it; no probe in this battery ever
reaches success(), and a final audit check asserts exactly that.

   1-2. settle/no-NaN     — reset layout settles finite: ball + K marbles at rest
                            in the tray, absent marbles parked in the depot,
                            everything still, score ~0, no success;
   3-4. randomization     — READBACK over 8 seeded resets: hopper yaw spans a wide
                            arc + xy jitter real, tray side flips, K varies;
   5.  null policy        — 240 idle steps -> score ~0, no success;
   6.  OPEN DRAIN         — the order violation, proved physically: a marble
                            released onto the floor with NO plug rolls down the
                            fall line and DRAINS into the sealed vault (ends ~15 cm
                            below the floor plane), never latches containment
                            (streak gate: its transit through the containment
                            window while rolling earns nothing), score stays 0;
   7.  funnel (2nd entry) — same from a different release point (near the HIGH
                            corner): the tilt routes every entry into the drain;
   8.  control (non-vac.) — the SAME release with the plug seated first (seated by
                            genuine rolling, latch verified): the marble now STAYS
                            contained — the drain, not the probe, was the cause;
                            partial score only (other marbles still in the tray);
   9.  seed strategy      — the end state of the seed's own plan (sphere placed
                            at rest on/in a shallow bin): ball dropped back into
                            the TRAY -> score 0, nothing latches;
  10.  wrong place        — ball settled on the open ground beside the hopper and
                            a marble grounded beside the tray -> nothing latches;
  11.  unsunk ball        — ball resting ON the false floor held against the
                            SEATED-state plan window is impossible physically
                            (it rolls in), so construct the near-miss directly:
                            ball at rest ON the plates just upslope of the seat
                            (in plan window, NOT sunk) -> seat z-ceiling rejects;
  12.  lost marble        — plug seated (genuine), one marble constructed at rest
                            in the VAULT (the irreversible loss state), the rest
                            genuinely rolled in -> success False, score capped at
                            plug + (K-1)/K beads;
  13.  latched credit     — plug + one bead latched, then that marble yanked to
                            the depot -> the latched score holds, no success;
  14.  rejection audit    — success() was never True at ANY judged point;
  15.  final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.place_sphere_i196.smoke --headless
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
    env = ENVS.get("simgen.drain_plug")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.35, -0.95, 0.85)) + o),
                                tuple(np.array((0.45, 0.05, 0.16)) + o),
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

    def hopper_pose() -> tuple[torch.Tensor, float]:
        hp = (scene.hopper.data.root_pos_w - scene.env_origins)[0]
        q = scene.hopper.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return hp, yaw

    def report(tag: str) -> None:
        bl = scene._fixture_local(scene.ball)[0]
        k = int(scene._k[0])
        beads = "".join(
            "L" if bool(scene._bead_latch[0, i]) else ("i" if bool(scene._bead_in(i)[0]) else ".")
            for i in range(k))
        s, ok = judge()
        print(f"[smoke] {tag:16s} | ball_l=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.4f}) plug_now={bool(scene._plug_now()[0])} "
              f"plug_latch={bool(scene._plug_latch[0])} beads[{k}]={beads} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_local(body, lx: float, ly: float, z: float) -> None:
        """Zero-velocity root-state write at hopper-fixture (lx, ly), world z."""
        hp, hyaw = hopper_pose()
        wx = float(hp[0]) + math.cos(hyaw) * lx - math.sin(hyaw) * ly
        wy = float(hp[1]) + math.sin(hyaw) * lx + math.cos(hyaw) * ly
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = wx, wy, z
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def place_world(body, x: float, y: float, z: float) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def plane_z(lx: float, ly: float) -> float:
        return c.z_mid + c.slope_s * (lx + ly)

    def hover_drop(body, lx: float, ly: float, r: float) -> None:
        """Release `body` just above the floor plane at fixture (lx, ly) — the
        probe constructor for GENUINE rolled-in states."""
        place_local(body, lx, ly, plane_z(lx, ly) + r + 0.006)

    def seat_plug() -> bool:
        """Construct the plug by GENUINE physics: hover-release upslope, let it
        roll in, return whether the latch set."""
        hover_drop(scene.ball, -c.h + 0.12, -c.h + 0.12, c.ball_r)
        for _ in range(25):
            step(20)
            if bool(scene._plug_latch[0]):
                return True
        return False

    def marble_l(i: int) -> torch.Tensor:
        return scene._fixture_local(scene.marbles[i])[0]

    def in_vault(i: int) -> bool:
        """Marble i inside the hopper footprint but UNDER the false floor."""
        loc = marble_l(i)
        below = float(loc[2]) < plane_z(float(loc[0]), float(loc[1])) - 0.05
        return (abs(float(loc[0])) < c.h and abs(float(loc[1])) < c.h and below)

    def states_finite() -> bool:
        ok = bool(torch.isfinite(scene.ball.data.root_state_w).all()
                  and torch.isfinite(scene.hopper.data.root_state_w).all()
                  and torch.isfinite(scene.tray.data.root_state_w).all())
        for mb in scene.marbles:
            ok = ok and bool(torch.isfinite(mb.data.root_state_w).all())
        return ok

    def all_still() -> bool:
        ok = float(scene.ball.data.root_lin_vel_w[0].norm()) < c.settle_speed
        for mb in scene.marbles:
            ok = ok and float(mb.data.root_lin_vel_w[0].norm()) < c.settle_speed
        return ok

    def tray_local(body) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        rel = body.data.root_pos_w - scene.tray.data.root_pos_w
        return quat_apply_inverse(scene.tray.data.root_quat_w, rel)[0]

    def in_tray(body, r: float) -> bool:
        loc = tray_local(body)
        return (abs(float(loc[0])) < c.tray_half_x and abs(float(loc[1])) < c.tray_half_y
                and abs(float(loc[2]) - (c.tray_floor_t + r)) < 0.02)

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("show")
    k = int(scene._k[0])
    live_in_tray = all(in_tray(scene.marbles[i], c.marble_r) for i in range(k))
    absent_parked = True
    for i in range(k, c.n_marbles):
        dp = (scene.marbles[i].data.root_pos_w - scene.env_origins)[0]
        absent_parked = absent_parked and (
            math.hypot(float(dp[0]) - c.depot_pos[0], float(dp[1]) - c.depot_pos[1]) < 0.15)
    check("settle: states finite, ball + K marbles at rest in the tray, absent marbles "
          "parked in the depot, everything still",
          states_finite() and in_tray(scene.ball, c.ball_r) and live_in_tray
          and absent_parked and all_still())
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(2)
        hp, hyaw = hopper_pose()
        tp = (scene.tray.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(hp[0]), float(hp[1]), math.degrees(hyaw),
                      float(tp[1]) > 0.0, int(scene._k[0])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (hopper_x, hopper_y, hopper_yaw_deg, "
          f"tray_side_pos_y, K):\n{arr}", flush=True)
    # yaw span: compare unit vectors (degrees wrap at +/-180)
    yaws = np.radians(arr[:, 2])
    yaw_spread = float(np.ptp(np.cos(yaws)) + np.ptp(np.sin(yaws)))
    check("randomization: hopper yaw spans a wide arc, xy jitter real, tray side "
          "flips (readback)",
          yaw_spread > 0.8
          and float((arr[:, 0:2].max(axis=0) - arr[:, 0:2].min(axis=0)).max()) > 0.01
          and 0.0 < arr[:, 3].mean() < 1.0)
    check("randomization: K (live marble count) varies across seeded resets",
          int(arr[:, 4].max()) > int(arr[:, 4].min()))

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. OPEN DRAIN: marble before plug is lost ==================
    # The core order violation, proved physically: with NO plug, a marble released
    # onto the floor rolls down the fall line and drains into the sealed vault. Its
    # transit through the containment window latches nothing (streak + order gate).
    torch.manual_seed(41)
    env.reset()
    step(10)
    hover_drop(scene.marbles[0], -c.h + 0.12, -c.h + 0.12, c.marble_r)
    step(300)
    report("open-drain")
    s, ok = judge()
    m0 = marble_l(0)
    check("open drain: marble released mid-floor with NO plug ends in the VAULT "
          f"(fixture z={float(m0[2]):.3f}, ~0.15 below the floor plane), containment "
          "never latches, score 0, no success",
          in_vault(0) and not bool(scene._bead_latch[0, 0])
          and not bool(scene._bead_in(0)[0]) and s <= 0.02 and not ok)

    # =========================== 7. funnel: a second entry point also drains ================
    torch.manual_seed(51)
    env.reset()
    step(10)
    hover_drop(scene.marbles[1], c.h - 0.05, c.h - 0.05, c.marble_r)  # near the HIGH corner
    step(360)
    report("funnel")
    s, ok = judge()
    check("funnel: marble released near the HIGH corner also drains into the vault "
          "(the tilt routes every entry to the drain), nothing latches",
          in_vault(1) and not bool(scene._bead_latch[0, 1]) and s <= 0.02 and not ok)

    # =========================== 8. control: with the plug seated, the same drop stays ======
    # Non-vacuousness of 6/7: seat the plug by GENUINE rolling first, then repeat
    # the identical marble release — it must now stay contained on the floor.
    torch.manual_seed(61)
    env.reset()
    step(10)
    check("control setup: ball hover-released upslope rolls in and the plug latch "
          "sets by genuine physics", seat_plug())
    bl = scene._fixture_local(scene.ball)[0]
    print(f"[smoke] seat readback: fixture z={float(bl[2]):.4f} (expected ~{c.z_seat:.4f})",
          flush=True)
    hover_drop(scene.marbles[0], -c.h + 0.12, -c.h + 0.12, c.marble_r)
    step(300)
    report("control")
    s, ok = judge()
    check("control: the SAME release WITH the plug seated stays contained (bead latch "
          "sets, marble rests above the floor plane against the ball) — the open "
          "drain, not the probe, caused the loss; success still False (marbles "
          "remain in the tray), score = plug + 1/K beads",
          bool(scene._bead_latch[0, 0]) and bool(scene._bead_in(0)[0]) and not ok
          and s <= c.w_plug + c.w_beads / max(int(scene._k[0]), 1) + 0.005)

    # =========================== 9. seed strategy: sphere into the shallow bin ==============
    # The seed's own success geometry — the red sphere placed at rest in/on a
    # shallow bin — is the TRAY here, and scores nothing.
    torch.manual_seed(71)
    env.reset()
    step(10)
    tp = (scene.tray.data.root_pos_w - scene.env_origins)[0]
    place_world(scene.ball, float(tp[0]), float(tp[1]),
                c.tray_floor_t + c.ball_r + 0.02)
    step(90)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy: ball dropped back into the shallow TRAY (the seed's "
          "sphere-on-bin end state) settles there — nothing latches, score 0",
          in_tray(scene.ball, c.ball_r) and not bool(scene._plug_latch[0])
          and s <= 0.02 and not ok)

    # =========================== 10. wrong place ============================================
    torch.manual_seed(81)
    env.reset()
    step(10)
    place_world(scene.ball, 0.30, -0.30, c.ball_r + 0.002)  # open ground, beside everything
    place_world(scene.marbles[0], 0.25, 0.30, c.marble_r + 0.002)
    step(90)
    report("wrong-place")
    s, ok = judge()
    check("wrong place: ball settled on open ground and a marble grounded outside "
          "the hopper — nothing latches, score 0, no success",
          not bool(scene._plug_latch[0]) and not bool(scene._bead_latch[0, 0])
          and s <= 0.02 and not ok)

    # =========================== 11. unsunk ball near-miss ==================================
    # A ball INSIDE the seat's plan window but resting ON the plate (never sunk
    # into the gap) must be rejected by the z ceiling alone. Constructed on plate
    # A near the ny wall: fixture (-h+0.060, -h+0.032) is 0.068 from the corner
    # (< seat_r 0.075) yet fully plate-supported (contact point 8 mm inside the
    # plate edge, ball 2 mm clear of the wall), so its rest z ~ plane + r*nrm ~
    # 0.199 > seat_z_hi. Judge the first frames (witnessing the state), then let
    # it roll the last centimetres in and confirm the latch sets only once sunk.
    torch.manual_seed(91)
    env.reset()
    step(10)
    nrm = math.sqrt(1.0 + 2.0 * c.slope_s ** 2)
    lx, ly = -c.h + 0.060, -c.h + 0.032
    place_local(scene.ball, lx, ly, plane_z(lx, ly) + c.ball_r * nrm + 0.001)
    saw_nearmiss, false_read = False, False
    for _ in range(8):  # judge the first frames: ball on-plate, in plan window, unsunk
        env.step(no_action)
        bl = scene._fixture_local(scene.ball)[0]
        in_plan = math.hypot(float(bl[0]) + c.h, float(bl[1]) + c.h) < c.seat_r
        unsunk = float(bl[2]) > c.seat_z_hi
        judge()
        if in_plan and unsunk:
            saw_nearmiss = True
            if bool(scene._plug_now()[0]) or bool(scene._plug_latch[0]):
                false_read = True
    check("unsunk near-miss: ball resting ON the plate INSIDE the seat plan window "
          "(state witnessed) is rejected by the seat z-ceiling alone",
          saw_nearmiss and not false_read)
    step(300)  # it now rolls the last centimetres in and seats — the window is honest
    report("rolls-in")
    check("unsunk near-miss control: the same ball then rolls in and the latch sets "
          "only once genuinely sunk + settled", bool(scene._plug_latch[0]))

    # =========================== 12. lost marble: success impossible, capped ================
    torch.manual_seed(101)
    env.reset()
    step(10)
    k = int(scene._k[0])
    check("lost-marble setup: plug seats by genuine rolling", seat_plug())
    # one marble constructed at rest in the vault (what checks 6/7 proved a
    # pre-plug release produces), the OTHERS genuinely rolled in
    place_local(scene.marbles[0], -c.h + 0.10, -c.h + 0.04, c.marble_r + 0.001)
    step(60)
    offsets = (0.0, 0.04, -0.04, 0.08)
    for i in range(1, k):
        d = offsets[i]
        hover_drop(scene.marbles[i], -c.h + 0.14 + d * 0.7071, -c.h + 0.14 - d * 0.7071,
                   c.marble_r)
        for _ in range(25):
            step(20)
            if bool(scene._bead_latch[0, i]):
                break
    report("lost-marble")
    s, ok = judge()
    others_in = all(bool(scene._bead_latch[0, i]) for i in range(1, k))
    cap = c.w_plug + c.w_beads * (k - 1) / k
    check("lost marble: one marble at rest in the sealed vault, plug seated, every "
          f"other marble genuinely contained — success False, score <= {cap:.3f} "
          "+ eps (the lost marble's share is unrecoverable)",
          in_vault(0) and others_in and not ok and s <= cap + 0.005)

    # =========================== 13. latched credit survives regression =====================
    torch.manual_seed(111)
    env.reset()
    step(10)
    check("latched-credit setup: plug seats by genuine rolling", seat_plug())
    hover_drop(scene.marbles[0], -c.h + 0.12, -c.h + 0.12, c.marble_r)
    for _ in range(25):
        step(20)
        if bool(scene._bead_latch[0, 0]):
            break
    s_a, ok_a = judge()
    place_world(scene.marbles[0], c.depot_pos[0], c.depot_pos[1], 0.10)  # yank it away
    step(60)
    report("regressed")
    s_b, ok_b = judge()
    check("latched credit: bead credit earned then the marble yanked away — the "
          f"latched score holds ({s_a:.3f} -> {s_b:.3f}), no success either side",
          bool(scene._bead_latch[0, 0]) and abs(s_a - s_b) < 1e-3
          and s_a >= c.w_plug + c.w_beads / 4 - 0.005 and not ok_a and not ok_b)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", states_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.drain_plug")
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

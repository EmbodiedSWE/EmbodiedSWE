"""Smoke / rubric-REJECTION battery for CageCaptureScene (sim_gen task `hockey_i294`)
— NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — cage staged, capture slide under external
wrench, chock set down by gravity — is the acceptance evidence that the rubric
ACCEPTS a correct outcome). Every teleport here is instrumentation that
CONSTRUCTS a wrong (or partial) outcome as a settled state and asserts the
rubric REJECTS it; no probe in this battery ever reaches success(), and a final
audit check asserts exactly that.

  1.  settle/no-NaN     — reset layout settles finite and still; MASS READBACK
                          (root_physx_view.get_masses) proves the authored masses
                          took (custom spawners silently ignore cfg mass_props);
  2.  score ~0 at reset — no credit for existing;
  3-4. randomization    — READBACK over 8 seeded resets: white-ball xy band,
                          decoy Bernoulli side flips, cage pose + yaw spreads,
                          chock parked on the flank OPPOSITE the decoy w/ jitter;
  5.  null policy       — 300 idle steps -> score ~0, no success, and the ball
                          stays ON ITS ANCHOR (sphere-creep guard);
  6.  seed strategy     — the end state the SEED's plan produces here (propel
                          the ball at the goal): the white ball is FIRED at the
                          open cage mouth; it rolls off its anchor (and typically
                          into the cage) — the anchor gate forfeits EVERYTHING:
                          score ~0, no latch, no success;
  7.  approach-only     — cage parked with the mouth near the anchored ball but
                          the ball still outside -> exactly the 0.15 band;
  8.  partial-only      — ball past the mouth plane but SHORT of the capture
                          margin -> 0.35 band, no capture credit, no success;
  9.  capture-no-chock  — ball fully inside, mouth open -> 0.60 band, NOT
                          success (the mouth must be barred);
  10. chock misplaced   — bar dropped beside the side wall instead of across
                          the mouth -> still 0.60, no chock credit, no success;
  11. chock-first       — bar seated squarely across the mouth of the EMPTY cage
                          (ball still out in the field) -> the chock latch is
                          gated on capture: score ~0, no success;
  12. wrong object      — cage slid over the BLACK DECOY + chock seated -> the
                          rubric tracks the white ball: score ~0, no success;
  13. decoy exclusion   — white ball captured + chock seated + decoy TELEPORTED
                          inside too: every other clause passes, the decoy
                          clause alone rejects -> latched 0.70 cap, NOT success;
  14. latched credit    — white ball then teleported back OUT: the latched
                          credit is unchanged (does not evaporate), no success;
  15. rejection audit   — success() was never True at ANY judged point;
  16. final no-NaN      — all task-object states finite at the end.

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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cage_capture")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.35, -1.05, 0.80)) + o),
                                tuple(np.array((0.45, 0.00, 0.08)) + o),
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

    def anchor_err() -> float:
        d = scene.ball.data.root_pos_w[0, 0:2] - scene._anchor[0]
        return float(d.norm())

    def cage_yaw_deg() -> float:
        q = scene.cage.data.root_quat_w[0]
        return math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))

    def report(tag: str) -> None:
        bl = scene._local(scene.ball)[0]
        dl = scene._local(scene.decoy)[0]
        kl = scene._local(scene.chock)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | ball_cage=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):+.3f}) anchor_err={anchor_err() * 1000:.1f}mm "
              f"decoy_cage=({float(dl[0]):+.3f},{float(dl[1]):+.3f}) "
              f"chock_cage=({float(kl[0]):+.3f},{float(kl[1]):+.3f},{float(kl[2]):+.3f}) "
              f"appr={bool(scene._appr_ever[0])} part={bool(scene._part_ever[0])} "
              f"capt={bool(scene._capt_ever[0])} chk={bool(scene._chk_ever[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def wxy(body) -> tuple[float, float]:
        p = (body.data.root_pos_w - scene.env_origins)[0]
        return float(p[0]), float(p[1])

    def place_cage(x: float, y: float) -> None:
        """Teleport the cage (probe constructor) to world (x, y) at yaw 0 —
        mouth (local +x) facing world +x."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, 0.002
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        scene.cage.write_root_state_to_sim(st, all_ids)

    def place_chock_cage_local(x: float, y: float) -> None:
        """Teleport the chock 4 mm above the floor at a cage-frame (x, y), bar
        long axis (local y) along the cage's mouth line."""
        cp = scene.cage.data.root_pos_w
        cq = scene.cage.data.root_quat_w
        loc = torch.tensor([x, y, 0.0], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = cp + quat_apply(cq, loc)
        st[:, 2] = scene.env_origins[:, 2] + c.chock_side / 2 + 0.004
        st[:, 3:7] = cq
        scene.chock.write_root_state_to_sim(st, all_ids)

    def place_body_world(body, x: float, y: float, z: float,
                         vel: tuple[float, float] = (0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = 1.0
        st[:, 7], st[:, 8] = vel[0], vel[1]
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def finite_all() -> bool:
        return bool(torch.isfinite(scene.cage.data.root_state_w).all()
                    and torch.isfinite(scene.chock.data.root_state_w).all()
                    and torch.isfinite(scene.ball.data.root_state_w).all()
                    and torch.isfinite(scene.decoy.data.root_state_w).all())

    # =========================== 1-2. settle / no-NaN / masses ==============================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(120)
    report("show")
    masses_ok = True
    try:
        mc = float(scene.cage.root_physx_view.get_masses()[0, 0])
        mb = float(scene.ball.root_physx_view.get_masses()[0, 0])
        mk = float(scene.chock.root_physx_view.get_masses()[0, 0])
        md = float(scene.decoy.root_physx_view.get_masses()[0, 0])
        print(f"[smoke] mass readback: cage={mc:.2f} ball={mb:.3f} chock={mk:.3f} "
              f"decoy={md:.3f}", flush=True)
        masses_ok = (abs(mc - c.cage_mass) < 0.05 * c.cage_mass
                     and abs(mb - c.ball_mass) < 0.05 * c.ball_mass
                     and abs(mk - c.chock_mass) < 0.05 * c.chock_mass
                     and abs(md - c.ball_mass) < 0.05 * c.ball_mass)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] mass readback FAILED ({exc!r})", flush=True)
        masses_ok = False
    still = (float(scene.ball.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.cage.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    check("settle: states finite, cage seated upright, ball on its anchor, all "
          "still; authored masses verified by readback",
          finite_all() and bool(scene.cage_seated()[0]) and anchor_err() < 0.010
          and still and masses_ok)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        bx, by = wxy(scene.ball)
        dx, dy = wxy(scene.decoy)
        gx, gy = wxy(scene.cage)
        kx, ky = wxy(scene.chock)
        side = 1.0 if dy > by else -1.0
        reads.append((bx, by, side, gx, gy, cage_yaw_deg(), kx, ky))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (ball_x, ball_y, decoy_side, cage_x, "
          f"cage_y, cage_yaw_deg, chock_x, chock_y):\n{np.round(arr, 3)}", flush=True)
    sides = arr[:, 2]
    ball_spread = float((arr[:, 0:2].max(axis=0) - arr[:, 0:2].min(axis=0)).max())
    check("randomization A: white-ball xy band is real (spread > 8 mm) and the "
          "decoy side flips across seeded resets (Bernoulli readback)",
          ball_spread > 0.008 and 0.0 < (sides > 0).mean() < 1.0)
    cage_spread = float((arr[:, 3:5].max(axis=0) - arr[:, 3:5].min(axis=0)).max())
    yaw_spread = float(arr[:, 5].max() - arr[:, 5].min())
    chock_opposite = bool(np.all(np.sign(arr[:, 7]) == -sides))
    chock_spread = float(arr[:, 6].max() - arr[:, 6].min())
    check("randomization B: cage pose spread (> 8 mm) + yaw spread (> 5 deg), "
          "chock parked on the flank OPPOSITE the decoy every seed, chock x "
          "jitter (> 4 mm) — all by readback",
          cage_spread > 0.008 and yaw_spread > 5.0 and chock_opposite
          and chock_spread > 0.004)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(300)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0, no success, and the ball stays on its anchor "
          "(< 10 mm drift; sphere-creep guard) after 300 idle steps",
          s <= 0.02 and not ok and anchor_err() < 0.010)

    # =========================== 6. seed strategy: propel the ball at the goal ==============
    # The seed's plan — strike the ball toward the open goal mouth — executed
    # here: the white ball is FIRED at the cage mouth. It leaves its anchor long
    # before it gets near the mouth (>= 16 cm of travel vs the 4.5 cm anchor),
    # so the anchor gate forfeits every credit clause no matter where it lands
    # (typically INSIDE the cage — physically "in the goal", worth nothing).
    torch.manual_seed(41)
    env.reset()
    step(30)
    bx, by = wxy(scene.ball)
    mc = (scene.mouth_center_w() - scene.env_origins)[0]
    d = np.array([float(mc[0]) - bx, float(mc[1]) - by])
    d = d / (np.linalg.norm(d) + 1e-9)
    place_body_world(scene.ball, bx, by, c.ball_r + 0.002,
                     vel=(1.6 * float(d[0]), 1.6 * float(d[1])))
    step(300)
    report("seed-strategy")
    bl = scene._local(scene.ball)[0]
    s, ok = judge()
    check("seed strategy: white ball FIRED at the cage mouth travels off its "
          f"anchor (err={anchor_err() * 1000:.0f} mm > 45 mm) — the anchor gate "
          "forfeits everything: score ~0 (<= 0.02), no latch, no success "
          f"(ball ended at cage-frame x={float(bl[0]):+.3f})",
          anchor_err() > c.anchor_tol and s <= 0.02 and not ok
          and not bool(scene._capt_ever[0]))

    # =========================== 7. approach-only ===========================================
    torch.manual_seed(51)
    env.reset()
    step(30)
    bx, by = wxy(scene.ball)
    place_cage(bx - (c.hx + 0.05), by)  # mouth centre 5 cm short of the ball
    step(60)
    report("approach-only")
    s, ok = judge()
    check("approach-only: mouth parked near the anchored ball, ball still "
          "OUTSIDE the mouth plane -> exactly the 0.15 approach band, no "
          "partial/capture credit, no success",
          0.14 <= s <= 0.16 and bool(scene._appr_ever[0])
          and not bool(scene._part_ever[0]) and not ok)

    # =========================== 8. partial-only (short of the margin) ======================
    torch.manual_seed(61)
    env.reset()
    step(30)
    bx, by = wxy(scene.ball)
    place_cage(bx - (c.hx - 0.02), by)  # ball 2 cm past the mouth plane
    step(60)
    report("partial-only")
    bl = scene._local(scene.ball)[0]
    s, ok = judge()
    check("partial near-miss: ball past the mouth plane but SHORT of the 4 cm "
          f"capture margin (x_loc={float(bl[0]):+.3f}) -> 0.35 band, no capture "
          "credit, no success",
          0.34 <= s <= 0.36 and bool(scene._part_ever[0])
          and not bool(scene._capt_ever[0]) and not ok)

    # =========================== 9. capture but mouth left open =============================
    torch.manual_seed(71)
    env.reset()
    step(30)
    bx, by = wxy(scene.ball)
    place_cage(bx - 0.050, by)  # ball fully inside (x_loc = 0.050)
    step(60)
    report("capture-open")
    s9, ok = judge()
    check("capture-no-chock: ball fully inside the cage on its anchor but the "
          "mouth is OPEN -> 0.60 band, NOT success (the mouth must be barred)",
          0.59 <= s9 <= 0.61 and bool(scene._capt_ever[0]) and not ok)

    # =========================== 10. chock misplaced (beside the side wall) =================
    place_chock_cage_local(c.hx + c.chock_seat_dx, 0.20)
    step(90)
    report("chock-beside")
    s, ok = judge()
    check("chock misplaced: bar dropped BESIDE the side wall (cage-frame y = "
          "+0.20, outside the 4 cm window) -> no chock credit, still 0.60, "
          "no success",
          abs(s - s9) < 0.005 and not bool(scene._chk_ever[0])
          and not bool(scene._chock_seated()[0]) and not ok)

    # =========================== 11. chock-first on the EMPTY cage ==========================
    torch.manual_seed(81)
    env.reset()
    step(30)
    place_chock_cage_local(c.hx + c.chock_seat_dx, 0.0)
    step(90)
    report("chock-first")
    s, ok = judge()
    check("chock-first: bar seated squarely across the mouth of the EMPTY cage "
          "(physically seated, readback) but the ball is still in the field -> "
          "the chock latch is capture-gated: score ~0 (<= 0.02), no success",
          bool(scene._chock_seated()[0]) and not bool(scene._chk_ever[0])
          and s <= 0.02 and not ok)

    # =========================== 12. wrong object: capture the DECOY ========================
    torch.manual_seed(91)
    env.reset()
    step(30)
    dx, dy = wxy(scene.decoy)
    place_cage(dx - 0.050, dy)  # decoy fully "captured"
    step(30)
    place_chock_cage_local(c.hx + c.chock_seat_dx, 0.0)
    step(90)
    report("wrong-object")
    s, ok = judge()
    check("wrong object: cage over the BLACK DECOY + chock seated across the "
          "mouth — the rubric tracks the WHITE ball (far outside): score ~0 "
          "(<= 0.02), no success",
          bool(scene._decoy_inside()[0]) and not bool(scene._capt_ever[0])
          and s <= 0.02 and not ok)

    # =========================== 13. decoy exclusion clause =================================
    # White ball captured + chock seated + decoy inside too. Construct order is
    # goal-safe: the decoy goes in FIRST, so no prefix of the construction ever
    # satisfies the goal (the decoy clause fails throughout).
    torch.manual_seed(101)
    env.reset()
    step(30)
    bx, by = wxy(scene.ball)
    place_cage(bx - 0.050, by)
    step(30)
    gx, gy = wxy(scene.cage)
    place_body_world(scene.decoy, gx - 0.045, gy + 0.045, c.ball_r + 0.002)
    step(30)
    place_chock_cage_local(c.hx + c.chock_seat_dx, 0.0)
    step(120)
    report("decoy-inside")
    s13, ok = judge()
    check("decoy exclusion: white ball captured on its anchor + chock seated + "
          "decoy TELEPORTED inside too — every other clause passes, the decoy "
          "clause alone rejects: latched 0.70 cap, NOT success",
          bool(scene._capt_ever[0]) and bool(scene._chk_ever[0])
          and bool(scene._decoy_inside()[0]) and 0.69 <= s13 <= 0.701 and not ok)

    # =========================== 14. latched credit survives regression =====================
    place_body_world(scene.ball, 0.90, 0.45, c.ball_r + 0.002)
    step(60)
    report("ball-removed")
    s14, ok = judge()
    check("latched credit: after the white ball is teleported back OUT the "
          f"latched score is unchanged ({s13:.3f} -> {s14:.3f}, <= 0.70 cap), "
          "still no success",
          abs(s14 - s13) < 1e-3 and s14 <= 0.701
          and not bool(scene._captured()[0]) and not ok)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.cage_capture")
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
    except Exception as e:  # noqa: BLE001 - fast fail beats a 20-min watchdog hang
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)

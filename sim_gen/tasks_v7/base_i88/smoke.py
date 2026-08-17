"""Smoke / rubric-REJECTION battery for BallastLiftScene (sim_gen task `base_i88`) —
NullRobot, teleported probe states + force probes, RECORDED.

This is NOT a solution (solve.py — drop three cubes into the hopper, let the see-saw
tip, push the ball into the dock — is the acceptance evidence that the rubric ACCEPTS
a correct outcome; it passes on seeds 0/1). Every teleport here is instrumentation
that CONSTRUCTS a wrong (or partial) outcome as a settled state and asserts the rubric
REJECTS it — plus applied-force probes that prove the MECHANISM is physically real:
the SAME velocity-regulated push (solve's own P2 controller, <= 4 N) that docks the
ball from the raised tray only dumps it on the floor when the beam is down, and one
ballast cube demonstrably cannot tip the beam while three demonstrably do.
No probe in this battery ever reaches success(), and a final audit asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: beam pinned cargo-end-down at
                            the stop, ball at rest in the tray, cubes loose on the
                            floor, hopper empty; score ~0, no success;
  3-4. randomization      — READBACK over 6 seeded resets: cube ground slots are
                            permuted (>= 2 distinct assignments), cube xy jitter and
                            free yaw vary, the ball's hinge-axis spot varies; every
                            reset spawns sane (beam down, ball in tray);
  5.  null policy         — 240 idle steps -> score ~0, no success, beam still down;
  6.  seed-strategy       — the seed family's move (transport the cargo straight to
      (gate probe)          the goal) is physically closed: solve's OWN regulated push
                            (<= 4 N) applied with the beam DOWN cannot get the ball
                            anywhere near the shelf — the tilted lip edge rolls it
                            into the lip/fence corner where it wedges (and even if a
                            seed let it over, it would only fall to the floor): the
                            ball never rises toward the shelf top, never crosses, no
                            credit, no success. Non-vacuous: the probe measurably
                            moved the ball (peak speed + lip climb are asserted);
  7.  1 cube never tips   — one cube dropped into the hopper (gravity+contact, solve's
                            own drop primitive) + long settle: the beam stays pinned
                            DOWN (the cfg-asserted moment ledger, demonstrated), no
                            raise credit, no success;
  8.  3 cubes always tip  — the remaining two cubes dropped in: the same hinge now
                            tips to the raised stop (mechanism accept side) — and
                            still NOT success (the ball is merely in the raised tray);
  9.  raised-tray reject  — the ball's raised-tray rest pose sits INSIDE the dock z
                            window but is rejected by the x window: riding the tray up
                            is not docking;
  10. floor at pedestal   — ball placed on the floor at the pedestal base (the fallen
                            outcome of any lift-less strategy): rejected (x and z);
  11. cube in dock        — a BLUE cube settled inside the dock counts for NOTHING
                            (identity, not geometry): success stays False;
  12. latched credit      — teleporting all cubes out of the hopper afterwards leaves
                            the latched score unchanged while cubes_in_hopper() drops
                            to zero (credit never evaporates);
  13. settle gate         — the ball IN the dock window but still moving fast is NOT
                            success (velocity gates are real); removed before it can
                            settle (this battery must never succeed);
  14. rejection audit     — success() was never True at ANY judged point;
  15. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.base_i88.smoke --headless
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
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ballast_lift")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(3, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.50, -1.15, 1.00)) + o),
                                tuple(np.array((0.38, -0.02, 0.15)) + o),
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

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        s, ok = judge()
        b = rel(scene.ball)
        hop = scene.cubes_in_hopper()[0]
        print(f"[smoke] {tag:16s} | tilt={float(scene.beam_tilt()[0]):+.1f}deg"
              f" in_hopper=({bool(hop[0])},{bool(hop[1])},{bool(hop[2])})"
              f" raised={bool(scene.raised()[0])}"
              f" ball=({float(b[0]):+.3f},{float(b[1]):+.3f},{float(b[2]):.3f})"
              f" in_dock={bool(scene.ball_in_dock()[0])}"
              f" score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, xyz, quat=None, vel=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor(xyz, device=device)
        if quat is None:
            st[:, 3] = 1.0
        else:
            st[:, 3:7] = torch.tensor(quat, device=device)
        if vel is not None:
            st[:, 7:10] = torch.tensor(vel, device=device)
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def wrench(body, f3: torch.Tensor) -> None:
        """World force expressed in the body's current link frame (the house
        convention: is_global=True silently drops torques on this stack)."""
        q = body.data.root_link_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f3.view(1, 3).expand(n, 3)).unsqueeze(1),
            torch.zeros(n, 1, 3, device=device),
            env_ids=all_ids)

    def push_ball_probe(steps: int = 900) -> tuple[float, float, float]:
        """Solve's OWN P2 push controller, verbatim: +x velocity servo toward
        0.11 m/s, clamped to [0, 4] N (stall force 4 N — crosses a 12 mm lip that
        demands ~2.3 N). This is the honest gate probe. Returns (total displacement
        of the ball, max ball height reached, peak +x speed reached — the
        non-vacuity evidence that the force really drove the ball); releases
        afterwards."""
        f3 = torch.zeros(3, device=device)
        p0 = rel(scene.ball).clone()
        zmax, vmax = 0.0, 0.0
        for _ in range(steps):
            rx = float(rel(scene.ball)[0])
            if rx > 0.585:
                break
            vx = float(scene.ball.data.root_lin_vel_w[0, 0])
            vmax = max(vmax, vx)
            f3[0] = max(min(60.0 * (0.11 - vx), 4.0), 0.0)
            wrench(scene.ball, f3)
            env.step(no_action)
            zmax = max(zmax, float(rel(scene.ball)[2]))
        wrench(scene.ball, zero3)
        step(60)
        disp = float((rel(scene.ball) - p0).norm())
        return disp, zmax, vmax

    def drop_cube(i: int) -> None:
        """Solve's OWN drop primitive: transport cube `i` to free space just above
        the hopper mouth (from the beam's CURRENT pose), release, let gravity +
        chimney contact insert it, and wait out any beam swing."""
        body = scene.cubes[i]
        hop_mid_y = (c.hop_y[0] + c.hop_y[1]) / 2
        local = torch.tensor([0.0, hop_mid_y, c.hop_top + 0.055], device=device)
        mouth = scene.beam.data.root_pos_w[0] + quat_apply(
            scene.beam.data.root_quat_w[0], local)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = mouth
        st[:, 3] = 1.0
        body.write_root_state_to_sim(st, all_ids)
        for _ in range(20):
            step(30)
            if float(body.data.root_lin_vel_w[0].norm()) < 0.05 \
                    and float(scene.beam.data.root_ang_vel_w[0].norm()) < 0.05:
                break
        assert bool(scene.cubes_in_hopper()[0, i]), f"probe setup: cube {i} missed the hopper"

    def ball_beam_frame() -> torch.Tensor:
        d = (scene.ball.data.root_pos_w - scene.beam.data.root_pos_w)[0]
        return quat_apply_inverse(scene.beam.data.root_quat_w[0].view(1, 4),
                                  d.view(1, 3))[0]

    def layout_sane(tag: str) -> bool:
        """Reset honesty: beam pinned at the lower stop, ball at rest in the tray,
        cubes loose on the floor, hopper empty."""
        tilt = float(scene.beam_tilt()[0])
        bf = ball_beam_frame()
        cubes_low = all(float(rel(b)[2]) < 0.05 for b in scene.cubes)
        ok = (tilt <= -c.raise_deg
              and abs(float(bf[0])) < 0.055
              and c.tray_y[0] < float(bf[1]) < c.tray_y[1]
              and 0.040 < float(bf[2]) < 0.085
              and cubes_low
              and not bool(scene.cubes_in_hopper()[0].any()))
        if not ok:
            print(f"[smoke] layout sanity VIOLATION at {tag}: tilt={tilt:+.1f} "
                  f"ball_bf={bf.tolist()} cubes_low={cubes_low}", flush=True)
        return ok

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    bodies = (scene.beam, scene.ball, *scene.cubes)
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("settle: all states finite; beam pinned cargo-end-down, ball in the tray, "
          "cubes on the floor, hopper empty", fin and layout_sane("reset"))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads, perms, yaws = [], set(), []
    sane = True
    slots = np.array(c.cube_slots)
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(30)
        sane = sane and layout_sane(f"seed {sd}")
        row = [float(rel(scene.ball)[0])]
        assign = []
        for b in scene.cubes:
            p = rel(b)
            row += [float(p[0]), float(p[1])]
            assign.append(int(np.argmin(((slots - np.array([float(p[0]), float(p[1])]))
                                         ** 2).sum(axis=1))))
            q = b.data.root_quat_w[0]
            yaws.append(2.0 * math.atan2(float(q[3]), float(q[0])))
        perms.add(tuple(assign))
        reads.append(row)
    arr = np.array(reads)
    print("[smoke] randomization readback (ball_x, cube0_xy, cube1_xy, cube2_xy):\n"
          f"{np.round(arr, 4)}\n[smoke] slot assignments: {sorted(perms)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    yaw_spread = max(yaws) - min(yaws)
    check("randomization: cube slots are permuted across seeds "
          f"({len(perms)} distinct assignments) and cube xy jitters "
          f"(spreads x={spread[1]:.3f},{spread[3]:.3f},{spread[5]:.3f})",
          len(perms) >= 2 and sane
          and all(spread[k] > 0.015 for k in (1, 2, 3, 4, 5, 6)))
    check("randomization: cube yaw is free (readback spread "
          f"{yaw_spread:.2f} rad) and the ball's hinge-axis spot varies "
          f"(x spread {spread[0]:.3f})",
          yaw_spread > 0.5 and spread[0] > 0.008)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0, no success, beam still down after 240 idle steps",
          s <= 0.02 and not ok and bool(scene.lowered()[0]))

    # =========================== 6. gate probe: push without the lift =======================
    # The seed family's move — transport the cargo straight to the goal — is
    # physically closed here: with the beam DOWN the lip edge is TILTED (it runs
    # along the beam's y axis), so the pushed ball rolls along the edge into the
    # lip/outer-fence corner and wedges there (force closure); even a seed that let
    # it over the lip would only drop it to the floor far below the shelf.
    env.reset(seed=41)
    step(60)
    z0 = float(rel(scene.ball)[2])
    disp, zmax, vmax = push_ball_probe()
    report("push-no-lift")
    b = rel(scene.ball)
    s, ok = judge()
    check("gate probe (seed-strategy analog): solve's own <= 4 N regulated push with "
          f"the beam DOWN measurably drives the ball (peak vx {vmax:.2f} m/s, "
          f"displacement {disp * 100:.1f} cm, climbs {(zmax - z0) * 1000:.0f} mm up "
          "the lip — non-vacuous) yet cannot take it anywhere near the shelf: it "
          f"never rises above {zmax:.3f} (shelf top {c.shelf_top:.3f}), never "
          f"crosses (x={float(b[0]):.3f} < {c.dock_x[0]}), no dock, no credit, no "
          "success",
          vmax > 0.05 and disp > 0.005 and zmax - z0 > 0.003 and zmax < 0.20
          and float(b[0]) < c.dock_x[0] and float(b[2]) < 0.20
          and not bool(scene.ball_in_dock()[0]) and s <= 0.02 and not ok)

    # =========================== 7. one cube can NEVER tip the beam =========================
    env.reset(seed=51)
    step(60)
    drop_cube(0)
    step(300)  # long settle: give a marginal mechanism every chance to creep
    report("one-cube")
    s, ok = judge()
    check("mechanism (reject side): ONE cube in the hopper cannot tip the beam — it "
          f"stays pinned down (tilt={float(scene.beam_tilt()[0]):+.1f} deg), no raise "
          "credit, no success",
          bool(scene.lowered()[0]) and not bool(scene.raised()[0])
          and s <= 0.12 and not ok)

    # =========================== 8. three cubes ALWAYS tip it ===============================
    drop_cube(1)
    drop_cube(2)
    step(120)
    report("three-cubes")
    s, ok = judge()
    check("mechanism (accept side): with all three cubes dropped in, the same hinge "
          f"tips to the raised stop (tilt={float(scene.beam_tilt()[0]):+.1f} deg) — "
          "and still NOT success (the ball merely rode the tray up)",
          bool(scene.raised()[0]) and 0.50 <= s <= 0.66 and not ok)

    # =========================== 9. raised-tray rest pose is rejected =======================
    b = rel(scene.ball)
    in_z_window = c.dock_z[0] <= float(b[2]) <= c.dock_z[1]
    check("raised-tray reject: the ball riding the raised tray sits INSIDE the dock z "
          f"window (z={float(b[2]):.3f}) but the x window rejects it "
          f"(x={float(b[0]):.3f} < {c.dock_x[0]}): riding up is not docking",
          in_z_window and float(b[0]) < c.dock_x[0]
          and not bool(scene.ball_in_dock()[0]) and not bool(scene.success()[0]))

    # =========================== 10. floor at the pedestal base =============================
    place(scene.ball, (0.47, c.shelf_xy[1], c.ball_r + 0.001), settle_steps=60)
    report("floor-at-base")
    b = rel(scene.ball)
    _s, ok = judge()
    check("floor reject: the ball resting on the floor at the pedestal base "
          f"(z={float(b[2]):.3f}) is rejected by the z window — there is no path "
          "onto the shelf from below", float(b[2]) < 0.10
          and not bool(scene.ball_in_dock()[0]) and not ok)

    # =========================== 11. cube in dock (identity) ================================
    place(scene.cubes[0], (0.60, c.shelf_xy[1], 0.26), settle_steps=60)
    report("cube-in-dock")
    cb = rel(scene.cubes[0])
    _s, ok = judge()
    check("identity: a BLUE cube settled inside the dock "
          f"(({float(cb[0]):.3f},{float(cb[1]):.3f},{float(cb[2]):.3f})) counts for "
          "NOTHING — only the red ball can succeed",
          0.545 < float(cb[0]) < 0.655 and 0.11 < float(cb[1]) < 0.23
          and 0.235 < float(cb[2]) < 0.27 and not ok)

    # =========================== 12. latched credit survives removal ========================
    s_before, _ok = judge()
    for i, b_ in enumerate(scene.cubes):
        place(b_, (0.90, -0.30 + 0.12 * i, c.cube_edge / 2 + 0.002), settle_steps=10)
    step(45)
    report("cubes-removed")
    s_after, ok = judge()
    check("latched credit: teleporting every cube out of the hopper leaves the "
          f"latched score unchanged ({s_before:.2f} -> {s_after:.2f}) while "
          "cubes_in_hopper() drops to zero",
          abs(s_after - s_before) < 1e-3
          and not bool(scene.cubes_in_hopper()[0].any()) and not ok)

    # =========================== 13. settle gate ============================================
    env.reset(seed=101)
    step(30)
    place(scene.ball, (0.575, c.shelf_xy[1], c.shelf_top + c.ball_r + 0.001),
          vel=(0.5, 0.0, 0.0), settle_steps=2)
    v_now = float(scene.ball.data.root_lin_vel_w[0].norm())
    in_window = bool(scene.ball_in_dock()[0])
    report("settle-gate")
    _s, ok = judge()
    gate_ok = in_window and v_now > c.settle_lin and not ok
    # remove the ball BEFORE it can settle in the dock (this battery must never succeed)
    place(scene.ball, (0.05, 0.45, c.ball_r + 0.001), settle_steps=30)
    check("settle gate: the ball IN the dock window but moving at "
          f"{v_now:.2f} m/s is NOT success (velocity gates are real)", gate_ok)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.ballast_lift")
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

"""Smoke / rubric-REJECTION battery for ChairTuckScene (sim_gen task
`stack_chairs_i110`) — NullRobot, teleported probe states + real force probes, RECORDED.

This is NOT a solution (solve.py — stage each chair before its matched bay, force-slide
it in through the opening until the backrest docks — is the acceptance evidence that
the rubric ACCEPTS a correct outcome; it passes on seeds 0/1/2). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome and asserts the rubric
REJECTS it; every force probe asserts the chair actually MOVED before claiming physics
blocked it (no vacuous physical checks). No probe reaches success(); a final audit
asserts exactly that.

  1-2. settle/no-NaN    — reset layout settles finite and sane: 6 desk pieces standing
                          in a row, 3 upright chairs on the floor IN FRONT; score ~0;
  3-4. randomization    — READBACK over 8 seeded resets from PHYSICAL wall-piece
                          positions: the left-to-right bay-width ORDER permutes, the
                          desk pose/yaw and the chair scatter vary; every reset sane;
  5.  null policy       — 240 idle steps -> score ~0, no success;
  6.  seed strategy     — the rlbench/stack_chairs outcome: the three chairs STACKED
                          into a vertical chair-on-chair tower (real, settled, top
                          chair z read back) -> nothing tucked, score ~0;
  7.  dead-end          — green chair tucked in the WIDE bay + blue in the MID bay
                          (both genuinely covered) -> the red chair now fits NO empty
                          bay: success False, score = 2/3 exactly, never 1;
  8.  misfit jam        — the 100 mm red chair force-pushed at the 90 mm bay: it
                          MOVES, reaches the mouth, and is physically stopped short of
                          the depth gate by the walls -> bay not covered;
  9.  backward blocked  — red chair backrest-first at the WIDE bay, force-pushed: the
                          120 mm backrest hits the desk edge (clearance 80 mm) well in
                          front of the opening -> depth pinned outside, facing < 0,
                          not tucked (the "can't insert backward" claim is physical);
  10. near-miss depth   — green chair settled only 12 mm behind the edge
                          (depth_min = 20 mm) -> not covered;
  11. on top of desk    — red chair standing ON the slab above its bay: depth/lateral/
                          upright/facing all read fine, the HEIGHT BAND alone rejects;
  12. settle gate       — a genuinely covered chair with velocity injected is NOT
                          covered while moving (sustained stillness required);
  13. latched credit    — teleporting that chair away leaves the latched tuck credit
                          (0.4/3) while the covered credit correctly drops;
  14. rejection audit   — success() was never True at ANY judged point;
  15. final no-NaN      — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.stack_chairs_i110.smoke --headless
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

CHAIRS = scene_mod.CHAIRS

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.chair_tuck")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.25, -0.90, 0.70)) + o),
                                tuple(np.array((0.38, 0.00, 0.04)) + o),
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
        cov = scene.covered()[0]
        dep = scene.chair_depth()[0]
        up = scene.chair_up_z()[0]
        parts = " ".join(
            f"{nm}:dep={float(dep[i]) * 1000:+.0f} up={float(up[i]):+.2f}"
            for i, (nm, _) in enumerate(CHAIRS))
        print(f"[smoke] {tag:16s} | {parts} | covered={[bool(v) for v in cov]} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    # ----- desk-frame helpers (mirror the solve's transport math) -----------------------
    def desk_axes() -> tuple[torch.Tensor, torch.Tensor]:
        cy, sy = torch.cos(scene.desk_yaw), torch.sin(scene.desk_yaw)
        return torch.stack([cy, sy], dim=1), torch.stack([-sy, cy], dim=1)

    def desk_to_world(x_t: float, y_t: torch.Tensor) -> torch.Tensor:
        fwd, lat = desk_axes()
        return (scene.desk_xy + fwd * x_t + lat * y_t.unsqueeze(1)
                + scene.env_origins[:, 0:2])

    def place(body, xyz, quat=None, vel=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor([float(v) for v in xyz], device=device)
        if quat is None:
            st[:, 3] = 1.0
        else:
            st[:, 3:7] = torch.tensor([float(v) for v in quat], device=device)
        if vel is not None:
            st[:, 7:10] = torch.tensor([float(v) for v in vel], device=device)
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
        return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

    def place_desk_frame(body, x_t: float, y_t: float, z: float,
                         yaw_off: float = 0.0, settle_steps: int = 45) -> None:
        """Place `body` at a DESK-frame pose (yaw = desk yaw + yaw_off)."""
        xy = desk_to_world(x_t, torch.full((n,), y_t, device=device))
        yw = float(scene.desk_yaw[0]) + yaw_off
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = xy
        st[:, 2] = scene.env_origins[:, 2] + z
        st[:, 3:7] = torch.tensor(yaw_quat(yw), device=device)
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    frame_mode = [0]  # pod-dependent wrench frame; probed on the first push

    def push(chair_i: int, y_bay_t: float, steps: int, align: bool) -> float:
        """Force-push chair `chair_i` toward the desk along the tuck direction with
        lateral centering on `y_bay_t` (velocity-servo, the solve's verified
        controller). Returns the total horizontal distance the chair moved — callers
        assert on it so no physical rejection check can pass vacuously."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        chair = scene.chairs[chair_i]
        p0 = chair.data.root_pos_w[0, 0:2].clone()
        probed = False
        for k in range(steps):
            fwd, lat = desk_axes()
            if not probed:
                disp = chair.data.root_pos_w[0, 0:2] - p0
                d = float(disp.norm())
                if d > 0.004:
                    if float((disp / d * fwd[0]).sum()) < 0.5:
                        frame_mode[0] = 1 - frame_mode[0]
                        print(f"[smoke] push frame probe: flipped to mode "
                              f"{frame_mode[0]}", flush=True)
                    probed = True
            v = chair.data.root_lin_vel_w[:, 0:2]
            v_along = (v * fwd).sum(dim=1)
            v_lat = (v * lat).sum(dim=1)
            _x_t, y_t = scene.chair_desk_xy()
            f_along = (0.65 + 4.0 * (0.055 - v_along)).clamp(0.0, 1.5)
            f_lat = (3.0 * (y_bay_t - y_t[:, chair_i]) - 1.5 * v_lat).clamp(-0.5, 0.5)
            f = torch.zeros(n, 3, device=device)
            f[:, 0:2] = fwd * f_along.unsqueeze(1) + lat * f_lat.unsqueeze(1)
            t = torch.zeros(n, 3, device=device)
            if align:  # steer only when the chair goes in nose-first
                ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
                fh = quat_apply(chair.data.root_quat_w, ex)[:, 0:2]
                fh = fh / fh.norm(dim=1, keepdim=True).clamp(min=1e-9)
                yaw_err = torch.atan2(fh[:, 0] * fwd[:, 1] - fh[:, 1] * fwd[:, 0],
                                      (fh * fwd).sum(dim=1))
                t[:, 2] = (0.06 * f_lat + 0.010 * yaw_err
                           - 0.0025 * chair.data.root_ang_vel_w[:, 2]).clamp(-0.02, 0.02)
            q = chair.data.root_link_quat_w
            if frame_mode[0] == 0:
                chair.set_external_force_and_torque(
                    quat_apply_inverse(q, f).unsqueeze(1),
                    quat_apply_inverse(q, t).unsqueeze(1), env_ids=all_ids)
            else:
                chair.set_external_force_and_torque(f.unsqueeze(1), t.unsqueeze(1),
                                                    env_ids=all_ids)
            env.step(no_action)
        chair.set_external_force_and_torque(torch.zeros(n, 1, 3, device=device),
                                            torch.zeros(n, 1, 3, device=device),
                                            env_ids=all_ids)
        step(45)
        return float((chair.data.root_pos_w[0, 0:2] - p0).norm())

    def stage_x() -> float:
        return -c.desk_depth / 2 - 0.080 - c.seat_depth / 2

    def bay_of_width(w: float) -> int:
        return int((scene.bay_w[0] - w).abs().argmin())

    def tuck_pose_x() -> float:
        return -c.desk_depth / 2 + 0.034  # x_t of a docked seat center (dep = +34 mm)

    def layout_sane(tag: str) -> bool:
        """Reset honesty: slab up at clear_h, 4 walls + backwall standing, all chairs
        upright ON THE FLOOR strictly in front of the desk's front edge."""
        slab_z = float(rel(scene.pieces["slab"])[2])
        wall_z = [float(rel(scene.pieces[f"wall_{k}"])[2]) for k in range(4)]
        dep = scene.chair_depth()[0]
        up = scene.chair_up_z()[0]
        cz = [float(rel(b)[2]) for b in scene.chairs]
        ok = (abs(slab_z - (c.clear_h + c.top_t / 2)) < 0.005
              and all(abs(z - c.clear_h / 2) < 0.005 for z in wall_z)
              and all(float(v) < -0.05 for v in dep)
              and all(float(v) > 0.9 for v in up)
              and all(abs(z - c.origin_h) < 0.012 for z in cz))
        if not ok:
            print(f"[smoke] layout sanity VIOLATION at {tag}: slab_z={slab_z:.3f} "
                  f"wall_z={wall_z} dep={dep.tolist()} up={up.tolist()} cz={cz}",
                  flush=True)
        return ok

    bodies = scene.chairs + list(scene.pieces.values())

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("settle: all states finite; desk assembled (slab on walls), all three "
          "chairs upright on the floor in front of the desk", fin and layout_sane("reset"))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    # READBACK from the PHYSICAL kinematic wall pieces (not the scene's bookkeeping):
    # adjacent-wall gaps give the left-to-right bay widths; the wall row direction
    # gives the desk yaw; wall row center gives the desk position.
    orders, yaws, cxs, chair_ys, chair_yaws = [], [], [], [], []
    sane = True
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(30)
        sane = sane and layout_sane(f"seed {sd}")
        wp = [rel(scene.pieces[f"wall_{k}"])[0:2] for k in range(4)]
        gaps = [float((wp[k + 1] - wp[k]).norm()) for k in range(3)]  # wall_t + bay_w
        orders.append(tuple(int(v) for v in np.argsort(gaps)))
        row = (wp[3] - wp[0])
        yaws.append(math.degrees(math.atan2(-float(row[0]), float(row[1]))))
        cxs.append(float((wp[0][0] + wp[3][0]) / 2))
        chair_ys.append([float(rel(b)[1]) for b in scene.chairs])
        from isaaclab.utils.math import quat_apply
        ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(1, 3)
        chair_yaws.append([
            math.degrees(math.atan2(
                float(quat_apply(b.data.root_quat_w[:1], ex)[0, 1]),
                float(quat_apply(b.data.root_quat_w[:1], ex)[0, 0])))
            for b in scene.chairs])
        gaps_mm = [f"{(g - c.wall_t) * 1000:.0f}" for g in gaps]
        print(f"[smoke] seed {sd}: bays L->R {gaps_mm} mm order={orders[-1]} "
              f"yaw={yaws[-1]:+.1f}deg", flush=True)
    check("randomization: the left-to-right bay-width ORDER (read from physical "
          f"wall-piece gaps) permutes across seeds ({len(set(orders))} distinct "
          "orders in 8 resets), every reset sane",
          len(set(orders)) >= 3 and sane)
    yaw_spread = max(yaws) - min(yaws)
    cx_spread = max(cxs) - min(cxs)
    cy_spread = max(max(ys) for ys in chair_ys) - min(min(ys) for ys in chair_ys)
    cyaw_spread = max(max(ys) for ys in chair_yaws) - min(min(ys) for ys in chair_yaws)
    check("randomization: desk yaw spread "
          f"{yaw_spread:.1f}deg > 8, desk x spread {cx_spread * 1000:.0f}mm > 10, "
          f"chair scatter y spread {cy_spread * 1000:.0f}mm > 100, chair yaw spread "
          f"{cyaw_spread:.0f}deg > 60 — all from physical readback",
          yaw_spread > 8.0 and cx_spread > 0.010 and cy_spread > 0.10
          and cyaw_spread > 60.0)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 6. seed strategy: the chair TOWER ==========================
    # rlbench/stack_chairs succeeds by stacking the chairs on one another. Build that
    # outcome for real — green's legs on red's seat, blue's legs on green's seat — let
    # it settle, and read the top chair's height back so the tower is provably real.
    env.reset(seed=41)
    step(30)
    from isaaclab.utils.math import quat_apply

    ex1 = torch.tensor([1.0, 0.0, 0.0], device=device).expand(1, 3)

    def chair_pose(i: int) -> tuple[float, float, float, float]:
        p = rel(scene.chairs[i])
        f = quat_apply(scene.chairs[i].data.root_quat_w[:1], ex1)[0]
        return (float(p[0]), float(p[1]), float(p[2]),
                math.atan2(float(f[1]), float(f[0])))

    # origin-to-origin spacing: seat-top height above origin (5 mm) + leg length
    # below origin (47 mm) + a small drop gap. Build SEQUENTIALLY, matching each
    # settled chair's actual pose. Green stacks aligned on red; the blue chair goes
    # on top CROSSWISE (90 deg) — an aligned third level is geometrically impossible
    # (its backrest bottom would interpenetrate green's backrest top), crosswise its
    # backrest hangs over green's open side and the legs land on the seat.
    lift = (c.seat_top - c.origin_h) + (c.origin_h - 0.003)
    for upper, lower, dyaw_up, gap in ((1, 0, 0.0, 0.002), (2, 1, math.pi / 2, 0.001)):
        x0, y0, z0, yw = chair_pose(lower)
        place(scene.chairs[upper], (x0, y0, z0 + lift + gap),
              quat=yaw_quat(yw + dyaw_up), settle_steps=80)
    step(60)
    report("chair-tower")
    s, ok = judge()
    z_top = float(rel(scene.chairs[2])[2])
    z_mid = float(rel(scene.chairs[1])[2])
    check("seed strategy: chairs STACKED into a chair-on-chair tower (settled, top "
          f"origin z={z_top * 1000:.0f}mm, middle z={z_mid * 1000:.0f}mm — the tower "
          "is real) -> nothing tucked, score ~0: piling chairs counts for NOTHING",
          z_top > 0.13 and z_mid > 0.09 and s <= 0.02 and not ok
          and not bool(scene.covered()[0].any()))

    # =========================== 7. dead-end assignment =====================================
    # Tuck the GREEN chair in the WIDE bay and the BLUE chair in the MID bay (teleport-
    # constructed, penetration-free: the docked backrest stays in front of the slab).
    # Both bays genuinely covered — but now the RED (100 mm) chair fits neither the
    # narrow (62 mm) nor any occupied bay: the episode is dead-ended at 2/3 forever.
    env.reset(seed=51)
    step(30)
    b_wide = bay_of_width(c.bay_widths[0])
    b_mid = bay_of_width(c.bay_widths[1])
    place_desk_frame(scene.chairs[1], tuck_pose_x(), float(scene.bay_y[0, b_wide]),
                     c.origin_h + 0.002, settle_steps=60)
    place_desk_frame(scene.chairs[2], tuck_pose_x(), float(scene.bay_y[0, b_mid]),
                     c.origin_h + 0.002, settle_steps=60)
    report("dead-end")
    s, ok = judge()
    cov = scene.covered()[0]
    n_cov = int(cov.sum())
    check("dead-end: green tucked in the WIDE bay + blue in the MID bay (both "
          f"genuinely covered: {n_cov}/3 bays) -> the red chair has no bay left that "
          f"fits it; success False, score = 2/3 ({s:.3f}), never 1",
          n_cov == 2 and not ok and abs(s - 2.0 / 3.0) < 0.02)

    # =========================== 8. misfit jam (physical) ===================================
    # Stage the 100 mm RED chair before the 90 mm MID bay and push with the solve's
    # own verified controller: the chair must MOVE (assert travel), reach the mouth,
    # and be stopped by the partition walls short of the depth gate.
    env.reset(seed=61)
    step(30)
    y_mid = float(scene.bay_y[0, b_mid := bay_of_width(c.bay_widths[1])])
    place_desk_frame(scene.chairs[0], stage_x(), y_mid, c.origin_h + 0.004,
                     settle_steps=20)
    dep0 = float(scene.chair_depth()[0, 0])
    moved = push(0, y_mid, steps=380, align=True)
    report("misfit-jam")
    s, ok = judge()
    dep1 = float(scene.chair_depth()[0, 0])
    check("misfit jam: the 100 mm red chair pushed at the 90 mm bay MOVED "
          f"{moved * 1000:.0f}mm (> 50), advanced from dep={dep0 * 1000:+.0f}mm to "
          f"{dep1 * 1000:+.0f}mm, and the walls stopped it short of the "
          f"{c.depth_min * 1000:.0f}mm gate -> bay not covered",
          moved > 0.050 and dep1 > dep0 + 0.030 and dep1 < c.depth_min
          and not bool(scene.covered()[0, b_mid]) and not ok)

    # =========================== 9. backward approach blocked (physical) ====================
    # Backrest-first at the WIDE bay: the 120 mm backrest cannot pass under the 80 mm
    # clearance — the chair is stopped with its seat center still OUTSIDE the desk.
    env.reset(seed=71)
    step(30)
    y_wide = float(scene.bay_y[0, b_wide := bay_of_width(c.bay_widths[0])])
    place_desk_frame(scene.chairs[0], stage_x(), y_wide, c.origin_h + 0.004,
                     yaw_off=math.pi, settle_steps=20)
    dep0 = float(scene.chair_depth()[0, 0])
    moved = push(0, y_wide, steps=380, align=False)
    report("backward")
    s, ok = judge()
    dep1 = float(scene.chair_depth()[0, 0])
    face = float(scene.chair_facing()[0, 0])
    check("backward blocked: red chair backrest-first at the wide bay MOVED "
          f"{moved * 1000:.0f}mm (> 25) and the desk edge stopped it at dep="
          f"{dep1 * 1000:+.0f}mm < -20 (backrest taller than the opening), facing="
          f"{face:+.2f} < 0 -> not tucked, not covered",
          moved > 0.025 and dep1 > dep0 + 0.020 and dep1 < -0.020 and face < 0.0
          and not bool(scene.covered()[0, b_wide]) and not ok)

    # =========================== 10. near-miss: not deep enough =============================
    env.reset(seed=81)
    step(30)
    b_g = bay_of_width(c.bay_widths[1])
    place_desk_frame(scene.chairs[1], -c.desk_depth / 2 + 0.012,
                     float(scene.bay_y[0, b_g]), c.origin_h + 0.002, settle_steps=60)
    report("near-miss")
    s, ok = judge()
    d_g = float(scene.chair_depth()[0, 1])
    check("near-miss: green chair settled upright in its bay's mouth but only "
          f"{d_g * 1000:.0f}mm behind the edge (< depth_min "
          f"{c.depth_min * 1000:.0f}mm) -> not covered, no success",
          0.0 < d_g < c.depth_min and not bool(scene.covered()[0, b_g]) and not ok)

    # =========================== 11. chair ON TOP of the desk ===============================
    # Standing on the slab directly above the wide bay: depth, lateral, upright and
    # facing ALL read fine — the height band alone must reject it.
    env.reset(seed=91)
    step(30)
    b_w = bay_of_width(c.bay_widths[0])
    place_desk_frame(scene.chairs[0], 0.0, float(scene.bay_y[0, b_w]),
                     c.clear_h + c.top_t + c.origin_h + 0.003, settle_steps=60)
    report("on-top")
    s, ok = judge()
    d_r = float(scene.chair_depth()[0, 0])
    up_r = float(scene.chair_up_z()[0, 0])
    z_r = float(rel(scene.chairs[0])[2])
    check("on top: red chair standing ON the desk top above the wide bay (dep="
          f"{d_r * 1000:+.0f}mm >= gate, upright {up_r:+.2f}, aligned) -> the height "
          f"band alone rejects it (z={z_r * 1000:.0f}mm outside "
          f"[{c.z_lo * 1000:.0f},{c.z_hi * 1000:.0f}]mm), not covered",
          d_r > c.depth_min and up_r > 0.95 and z_r > c.z_hi
          and not bool(scene.covered()[0, b_w]) and not ok)

    # =========================== 12. settle gate ============================================
    # Construct ONE genuinely covered bay, then inject velocity: while the chair
    # moves, the sustained-stillness counter is zero and covered() must read False.
    env.reset(seed=101)
    step(30)
    b_n = bay_of_width(c.bay_widths[2])
    place_desk_frame(scene.chairs[2], tuck_pose_x(), float(scene.bay_y[0, b_n]),
                     c.origin_h + 0.002, settle_steps=60)
    report("covered-anchor")
    was_cov = bool(scene.covered()[0, b_n])
    ch = scene.chairs[2]
    fwd, _lat = desk_axes()
    st = ch.data.root_state_w.clone()
    st[:, 7:9] = -0.30 * fwd  # kick OUT of the bay, along the open front
    ch.write_root_state_to_sim(st, all_ids)
    step(2)
    report("kicked")
    v_now = float(ch.data.root_lin_vel_w[0].norm())
    moving_cov = bool(scene.covered()[0, b_n])
    s, ok = judge()
    check("settle gate: a genuinely covered bay (rubric anchor: covered read True "
          f"once settled) with velocity injected ({v_now:.2f} m/s) is NOT covered "
          "while the chair moves — sustained stillness is required",
          was_cov and not moving_cov and not ok)

    # =========================== 13. latched tuck credit ====================================
    place(scene.chairs[2], (0.10, -0.35, c.origin_h + 0.003), settle_steps=60)
    report("moved-away")
    s_after, ok = judge()
    check("latched credit: teleporting the blue chair back OUT leaves the latched "
          f"tuck credit (score={s_after:.3f} ~= {c.tuck_credit / 3:.3f}) while the "
          "covered credit correctly drops with the physical state",
          abs(s_after - c.tuck_credit / 3) < 0.02
          and not bool(scene.covered()[0, b_n]) and not ok)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.chair_tuck")
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

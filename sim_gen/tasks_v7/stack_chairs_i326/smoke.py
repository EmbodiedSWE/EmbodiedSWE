"""Smoke / rubric-REJECTION battery for ChairFoldawayScene (sim_gen task
`stack_chairs_i326`) — NullRobot, teleported probe states + real force probes, RECORDED.

This is NOT a solution (solve.py — fold each seat with a hinge torque servo, then
lower the folded chair down into a slot — is the acceptance evidence that the rubric
ACCEPTS a correct outcome; it passes on forge seeds). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome and asserts the rubric
REJECTS it; every physical claim is backed by readback (the probe MOVED / the chair
really rests where claimed — no vacuous physical checks). No probe reaches success();
a final audit asserts exactly that.

  1-2. settle/no-NaN    — reset layout settles finite and sane: rack assembled (plate,
                          walls, 4 fins read back at pose), 3 DEPLOYED chairs upright
                          on the floor in front; score ~0;
  3-4. randomization    — READBACK over 8 seeded resets: rack yaw/x from the PHYSICAL
                          fin pieces, chair scatter slot-order permutes, chair yaws
                          spread; every reset sane;
  5.  null policy       — 240 idle steps -> score ~0, no success;
  6.  seed strategy     — the rlbench/stack_chairs outcome: DEPLOYED chairs stacked
                          into a real settled chair-on-chair pile, as tall as they
                          free-stand (heights read back) -> nothing racked, score ~0;
  7.  deployed drop     — a DEPLOYED chair released over a slot falls IN (drop read
                          back: its 95 mm foot passes the gap) but the fin CATCHES
                          the seat and props it well below the fold gate -> not
                          covered: the aperture physically forces the fold;
  8.  deployed press    — the fin-propped deployed chair pressed DOWN at 3x its
                          weight for 2 s: the prop is geometric — the seat never
                          passes the fold gate, the slot never covers (contrast: the
                          same lowering controller racks a FOLDED chair in check 11);
  9.  fold-gate miss    — a chair standing IN a slot with its seat PROPPED on the fin
                          top (~45-60 deg, a real settled contact state): position,
                          height and uprightness all read fine — the FOLD gate alone
                          rejects;
  10. height band       — a FOLDED chair standing ON the fin tops bridging two fins:
                          folded, upright, over a slot center — the height band alone
                          rejects;
  11. occupied slot     — a folded chair genuinely LOWERED into a slot (velocity-servo
                          wrench, the solve's controller) covers it; a second folded
                          chair PRESSED down onto that slot (held upright by the probe
                          wrench — no tumble) rides the first chair's folded seat far
                          above the band -> the slot counts EXACTLY ONE, no success;
  12. settle gate       — the covered chair kicked to 0.25 m/s is NOT covered while
                          moving (sustained stillness required), and re-covers once
                          it settles again;
  13. latched credit    — teleporting the covered chair back OUT drops the covered
                          credit with the physical state while the latched fold
                          credits (2 chairs ever-folded) survive;
  14. rejection audit   — success() was never True at ANY judged point;
  15. final no-NaN      — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.stack_chairs_i326.smoke --headless
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
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.chair_foldaway")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.30, -0.95, 0.80)) + o),
                                tuple(np.array((0.35, 0.00, 0.10)) + o),
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
        fd = scene.fold_deg()[0]
        up = scene.chair_up_z()[0]
        parts = " ".join(
            f"{nm}:fold={float(fd[i]):+.0f} up={float(up[i]):+.2f} "
            f"z={float(rel(scene.frames[i])[2]) * 1000:.0f}"
            for i, (nm, _) in enumerate(CHAIRS))
        print(f"[smoke] {tag:16s} | {parts} | covered={[bool(v) for v in cov]} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    # ----- rack-frame helpers (mirror the solve's transport math) -----------------------
    def rack_axes() -> tuple[torch.Tensor, torch.Tensor]:
        cy, sy = torch.cos(scene.rack_yaw), torch.sin(scene.rack_yaw)
        return torch.stack([cy, sy], dim=1), torch.stack([-sy, cy], dim=1)

    def slot_xy(s: int) -> torch.Tensor:
        _ex, ey = rack_axes()
        return scene.rack_xy + ey * float(c.slot_y[s])

    def yaw_quat_t(yaw: torch.Tensor) -> torch.Tensor:
        h = yaw / 2
        z = torch.zeros_like(yaw)
        return torch.stack([torch.cos(h), z, z, torch.sin(h)], dim=1)

    hinge_v = torch.tensor(c.hinge, device=device).expand(n, 3)

    def place_chair(i: int, xy, z: float, yaw, fold_deg=None,
                    settle_steps: int = 45) -> None:
        """Kinematic probe placement of the WHOLE chair linkage (instrumentation, not
        a solution) + REAL physics steps before judging. `fold_deg` None keeps the
        chair's CURRENT relative fold pose; a number writes that hinge angle."""
        frame, seat = scene.frames[i], scene.seats[i]
        if fold_deg is None:
            from isaaclab.utils.math import quat_inv

            fq, sq = frame.data.root_quat_w, seat.data.root_quat_w
            q_rel = quat_mul(quat_inv(fq), sq)
        else:
            phi = math.radians(float(fold_deg))
            q_rel = torch.tensor([math.cos(phi / 2), 0.0, -math.sin(phi / 2), 0.0],
                                 device=device).expand(n, 4)
        if not torch.is_tensor(yaw):
            yaw = torch.full((n,), float(yaw), device=device)
        if not torch.is_tensor(xy):
            xy = torch.tensor([float(v) for v in xy], device=device).expand(n, 2)
        tq = yaw_quat_t(yaw)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = xy
        st[:, 2] = z
        st[:, 3:7] = tq
        st[:, 0:3] += scene.env_origins
        frame.write_root_state_to_sim(st, all_ids)
        ss = torch.zeros(n, 13, device=device)
        ss[:, 0:3] = st[:, 0:3] + quat_apply(tq, hinge_v)
        ss[:, 3:7] = quat_mul(tq, q_rel)
        seat.write_root_state_to_sim(ss, all_ids)
        step(settle_steps)

    # ----- wrench helpers (pod-dependent frame, probed with a HORIZONTAL push) ----------
    frame_mode = [0]
    zero3 = torch.zeros(n, 3, device=device)
    W = (c.frame_mass + c.seat_mass) * 9.81
    ez_w = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
    ex_b = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)

    def wrench(body, f_w: torch.Tensor, t_w: torch.Tensor | None = None) -> None:
        if t_w is None:
            t_w = zero3
        if frame_mode[0] == 0:
            q = body.data.root_link_quat_w
            f, t = quat_apply_inverse(q, f_w), quat_apply_inverse(q, t_w)
        else:
            f, t = f_w, t_w
        body.set_external_force_and_torque(f.unsqueeze(1), t.unsqueeze(1),
                                           env_ids=all_ids)

    def clear(body) -> None:
        body.set_external_force_and_torque(zero3.unsqueeze(1), zero3.unsqueeze(1),
                                           env_ids=all_ids)

    def probe_wrench_frame(i: int) -> float:
        """Push chair i's frame +world-x on the floor and read the displacement: fixes
        `frame_mode` AND proves applied wrenches move bodies (probe non-vacuity)."""
        frame = scene.frames[i]
        p0 = frame.data.root_pos_w[0, 0:2].clone()
        f = torch.zeros(n, 3, device=device)
        f[:, 0] = 3.2  # > mu*W = 2.75 N, far under the ~5.5 N tip bound
        for _ in range(25):
            wrench(frame, f)
            env.step(no_action)
        disp = frame.data.root_pos_w[0, 0:2] - p0
        d = float(disp.norm())
        if d > 0.002 and float(disp[0]) < 0.5 * d:
            frame_mode[0] = 1 - frame_mode[0]
            print(f"[smoke] wrench frame probe: flipped to mode {frame_mode[0]}",
                  flush=True)
        clear(frame)
        step(30)
        return d

    def press_down(i: int, s: int, force: float,
                   steps: int) -> tuple[float, float, bool]:
        """Press chair i straight DOWN over slot s with an xy PD holding it on the
        slot center. Returns (min env-local z, MAX fold angle seen, covered-ever) —
        the fold max is sampled EVERY substep so a transient fold-through would be
        caught, not masked by the post-release settle."""
        frame = scene.frames[i]
        tgt = slot_xy(s) + scene.env_origins[:, 0:2]
        z_min, fd_max, cov_any = 10.0, -180.0, False
        for _ in range(steps):
            p = frame.data.root_pos_w
            z_loc = float((p[:, 2] - scene.env_origins[:, 2])[0])
            z_min = min(z_min, z_loc)
            fd_max = max(fd_max, float(scene.fold_deg()[0, i]))
            cov_any = cov_any or bool(scene.covered()[0, s])
            v = frame.data.root_lin_vel_w
            f = torch.zeros(n, 3, device=device)
            f[:, 0:2] = (10.0 * (tgt - p[:, 0:2]) - 4.0 * v[:, 0:2]).clamp(-1.2, 1.2)
            f[:, 2] = -float(force)
            wrench(frame, f)
            env.step(no_action)
        clear(frame)
        step(45)
        fd_max = max(fd_max, float(scene.fold_deg()[0, i]))
        cov_any = cov_any or bool(scene.covered()[0, s])
        return z_min, fd_max, cov_any

    def lower_into_slot(i: int, s: int) -> bool:
        """The solve's verified insertion controller: hover the FOLDED chair over slot
        s, then velocity-servo it down with xy/yaw/uprighting PDs; drop the wrench
        12 mm above the rack floor; settle. Returns covered(s)."""
        frame = scene.frames[i]
        hover_z = c.fin_top + 0.005 + c.origin_h
        yaw_tgt = scene.rack_yaw + math.pi / 2
        place_chair(i, slot_xy(s), hover_z, yaw_tgt, settle_steps=0)
        release_z = c.rack_floor_z + 0.012
        entry_z = c.fin_top + c.origin_h + 0.002
        tgt = slot_xy(s) + scene.env_origins[:, 0:2]
        for _ in range(700):
            p = frame.data.root_pos_w
            z_loc = float((p[:, 2] - scene.env_origins[:, 2])[0])
            if z_loc <= release_z:
                break
            v = frame.data.root_lin_vel_w
            w = frame.data.root_ang_vel_w
            err_xy = tgt - p[:, 0:2]
            aligned = float(err_xy[0].norm()) < 0.010
            v_des = -0.22 if (aligned or z_loc < entry_z) else 0.0
            f = torch.zeros(n, 3, device=device)
            f[:, 0:2] = (12.0 * err_xy - 5.0 * v[:, 0:2]).clamp(-1.5, 1.5)
            f[:, 2] = (W + 6.0 * (v_des - v[:, 2])).clamp(0.0, 2.0 * W)
            q = frame.data.root_quat_w
            z_b = quat_apply(q, ez_w)
            t = (0.40 * torch.cross(z_b, ez_w, dim=1) - 0.020 * w).clamp(-0.15, 0.15)
            fx_w = quat_apply(q, ex_b)
            yaw_now = torch.atan2(fx_w[:, 1], fx_w[:, 0])
            yerr = torch.atan2(torch.sin(yaw_tgt - yaw_now),
                               torch.cos(yaw_tgt - yaw_now))
            t[:, 2] = (0.060 * yerr - 0.012 * w[:, 2]).clamp(-0.06, 0.06)
            wrench(frame, f, t)
            env.step(no_action)
        clear(frame)
        step(80)
        return bool(scene.covered()[0, s])

    def layout_sane(tag: str) -> bool:
        """Reset honesty from PHYSICAL readback: plate/walls/fins at their kinematic
        poses (fin row consistent with the scene's bookkeeping yaw), all chairs
        DEPLOYED and upright on the floor."""
        plate_z = float(rel(scene.pieces["plate"])[2])
        fin_z = [float(rel(scene.pieces[f"fin_{k}"])[2]) for k in range(4)]
        row = rel(scene.pieces["fin_3"])[0:2] - rel(scene.pieces["fin_0"])[0:2]
        yaw_read = math.atan2(-float(row[0]), float(row[1]))
        yaw_err = abs(math.atan2(math.sin(yaw_read - float(scene.rack_yaw[0])),
                                 math.cos(yaw_read - float(scene.rack_yaw[0]))))
        fd = scene.fold_deg()[0]
        up = scene.chair_up_z()[0]
        cz = [float(rel(b)[2]) for b in scene.frames]
        wall_z = c.plate_size[2] + c.fin_h / 2
        ok = (abs(plate_z - c.plate_size[2] / 2) < 0.005
              and all(abs(z - wall_z) < 0.005 for z in fin_z)
              and yaw_err < 0.02
              and all(abs(float(v)) < 6.0 for v in fd)
              and all(float(v) > 0.9 for v in up)
              and all(abs(z - c.origin_h) < 0.012 for z in cz))
        if not ok:
            print(f"[smoke] layout sanity VIOLATION at {tag}: plate_z={plate_z:.3f} "
                  f"fin_z={fin_z} yaw_err={yaw_err:.3f} fold={fd.tolist()} "
                  f"up={up.tolist()} cz={cz}", flush=True)
        return ok

    bodies = scene.frames + scene.seats + list(scene.pieces.values())

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("settle: all states finite; rack assembled (plate + walls + 4 fins read "
          "back at pose), all three chairs DEPLOYED and upright on the floor",
          fin and layout_sane("reset"))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    yaws, rxs, orders, cyaws = [], [], [], []
    sane = True
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(30)
        sane = sane and layout_sane(f"seed {sd}")
        row = rel(scene.pieces["fin_3"])[0:2] - rel(scene.pieces["fin_0"])[0:2]
        yaws.append(math.degrees(math.atan2(-float(row[0]), float(row[1]))))
        mid = (rel(scene.pieces["fin_3"])[0:2] + rel(scene.pieces["fin_0"])[0:2]) / 2
        rxs.append(float(mid[0]))
        cy = [float(rel(b)[1]) for b in scene.frames]
        orders.append(tuple(int(v) for v in np.argsort(cy)))
        ex1 = torch.tensor([1.0, 0.0, 0.0], device=device).expand(1, 3)
        cyaws.extend(
            math.degrees(math.atan2(
                float(quat_apply(b.data.root_quat_w[:1], ex1)[0, 1]),
                float(quat_apply(b.data.root_quat_w[:1], ex1)[0, 0])))
            for b in scene.frames)
        print(f"[smoke] seed {sd}: rack yaw={yaws[-1]:+.1f}deg x={rxs[-1]:+.3f} "
              f"chair y-order={orders[-1]}", flush=True)
    yaw_spread = max(yaws) - min(yaws)
    rx_spread = max(rxs) - min(rxs)
    check("randomization: rack pose read from the PHYSICAL fin pieces — yaw spread "
          f"{yaw_spread:.1f}deg > 8, x spread {rx_spread * 1000:.0f}mm > 15; every "
          "reset sane", yaw_spread > 8.0 and rx_spread > 0.015 and sane)
    cyaw_spread = max(cyaws) - min(cyaws)
    check("randomization: chair scatter — left-to-right chair ORDER permutes "
          f"({len(set(orders))} distinct orders in 8 resets, >= 3) and chair yaws "
          f"spread {cyaw_spread:.0f}deg > 60",
          len(set(orders)) >= 3 and cyaw_spread > 60.0)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 6. seed strategy: the chair PILE ===========================
    # rlbench/stack_chairs succeeds by piling chairs on one another. Build that outcome
    # for real, as far as these top-heavy folding chairs physically stand: green
    # stacked on red's DEPLOYED seat (a genuine settled 2-high chair-on-chair stack,
    # 0.5 mm set-down, heights read back; a 3rd free-standing level is statically
    # impossible — the pile CoM walks past the 50 mm foot edge). The stack counts
    # NOTHING: no seat folded, nothing in a slot.
    env.reset(seed=41)
    step(30)
    ex1 = torch.tensor([1.0, 0.0, 0.0], device=device).expand(1, 3)
    p0 = rel(scene.frames[0])
    f0 = quat_apply(scene.frames[0].data.root_quat_w[:1], ex1)[0]
    yw = math.atan2(float(f0[1]), float(f0[0]))
    # green's foot 0.5 mm above red's deployed seat top; green yawed +90 deg
    # (crosswise, the stable way piles are built) so its seat mass hangs SIDEWAYS,
    # keeping the combined CoM ~9 mm inside red's front foot edge
    seat_top = c.hinge[2] + c.seat_com[2] + c.seat_size[2] / 2  # rel frame origin
    foot_bot = c.foot_center[2] - c.foot_size[2] / 2
    lift = seat_top - foot_bot  # upper origin height above lower origin
    place_chair(1, (float(p0[0]) + 0.072 * math.cos(yw),
                    float(p0[1]) + 0.072 * math.sin(yw)),
                float(p0[2]) + lift + 0.0005, yw + math.pi / 2, fold_deg=0.0,
                settle_steps=150)
    step(60)
    report("chair-pile")
    s, ok = judge()
    z_mid = float(rel(scene.frames[1])[2])
    up_lo = float(scene.chair_up_z()[0, 0])
    folds_ok = bool((scene.fold_deg()[0].abs() < 25.0).all())
    check("seed strategy: green stacked on red's seat — a real settled chair-on-chair "
          f"stack (upper origin z={z_mid * 1000:.0f}mm > 120, lower still upright "
          f"{up_lo:+.2f}, all seats deployed) -> nothing racked, score ~0: piling "
          "counts for NOTHING", z_mid > 0.12 and up_lo > 0.9 and folds_ok
          and s <= 0.02 and not ok and not bool(scene.covered()[0].any()))

    # =========================== 7-8. deployed chair: drop + press ==========================
    env.reset(seed=51)
    step(30)
    d_probe = probe_wrench_frame(2)  # blue, on the floor — also proves wrenches move bodies
    print(f"[smoke] wrench probe moved blue {d_probe * 1000:.0f}mm "
          f"(mode {frame_mode[0]})", flush=True)
    # release the DEPLOYED red chair with its seat panel 5 mm above the fin tops, over
    # the middle slot, in the solve's insertion yaw
    seat_under_z = c.hinge[2] + c.seat_com[2] - c.seat_size[2] / 2  # seat underside
    drop_z = c.fin_top - seat_under_z + 0.005
    place_chair(0, slot_xy(1), drop_z, scene.rack_yaw + math.pi / 2, fold_deg=0.0,
                settle_steps=90)
    report("deployed-drop")
    s, ok = judge()
    z_end = float(rel(scene.frames[0])[2])
    fd_r = float(scene.fold_deg()[0, 0])
    dropped = drop_z - z_end
    check("deployed drop: the DEPLOYED red chair released over a slot fell "
          f"{dropped * 1000:.0f}mm INTO it (> 30, its 95mm foot passes the gap) but "
          f"the fin CAUGHT the seat and props it at {fd_r:+.0f}deg — geometrically "
          f"capped >= 10deg below fold_min {c.fold_min_deg:.0f} — so even standing "
          f"near the rack floor (z={z_end * 1000:.0f}mm) the slot is NOT covered: "
          "the aperture forces the fold",
          dropped > 0.030 and z_end < c.z_hi + 0.010
          and 30.0 < fd_r < c.fold_min_deg - 10.0
          and not bool(scene.covered()[0, 1]) and not ok)
    z_min, fd_max, cov_any = press_down(0, 1, 3.0 * W, 240)
    report("deployed-press")
    s, ok = judge()
    check("deployed press: pressed straight DOWN at 3x its weight for 2 s the fin "
          f"prop NEVER folds the seat past the gate (max fold {fd_max:+.0f}deg < "
          f"{c.fold_min_deg - 5.0:.0f}, sampled every substep) and the slot never "
          "read covered: ramming cannot substitute for the fold",
          fd_max < c.fold_min_deg - 5.0 and not cov_any
          and not bool(scene.covered()[0, 1]) and not ok)

    # =========================== 9. fold-gate near-miss =====================================
    # A chair STANDING IN a slot at the correct depth/height/uprightness whose seat is
    # propped on the fin top (a real settled contact state ~45-60 deg): every gate
    # except the fold reads fine — the fold gate alone must reject.
    env.reset(seed=61)
    step(30)
    place_chair(0, slot_xy(0), c.rack_floor_z + 0.002, scene.rack_yaw + math.pi / 2,
                fold_deg=55.0, settle_steps=90)
    report("propped-seat")
    s, ok = judge()
    fd_p = float(scene.fold_deg()[0, 0])
    z_p = float(rel(scene.frames[0])[2])
    up_p = float(scene.chair_up_z()[0, 0])
    x_r, y_r = scene.chair_rack_xy()
    in_band = (abs(float(x_r[0, 0])) < c.band_x
               and abs(float(y_r[0, 0]) - c.slot_y[0]) < c.band_y)
    check("fold-gate near-miss: chair standing IN slot 0 (z="
          f"{z_p * 1000:.0f}mm in band, up={up_p:+.2f}, xy in band) with its seat "
          f"PROPPED on the fin top at {fd_p:+.0f}deg (a settled contact state, "
          f"< fold_min {c.fold_min_deg:.0f}deg) -> the fold gate alone rejects",
          c.z_lo < z_p < c.z_hi and up_p > 0.95 and in_band
          and 25.0 < fd_p < c.fold_min_deg - 5.0
          and not bool(scene.covered()[0].any()) and not ok)

    # =========================== 10. height band: folded ON the fin tops ====================
    env.reset(seed=71)
    step(30)
    place_chair(0, slot_xy(0), c.fin_top + c.origin_h + 0.003, scene.rack_yaw,
                fold_deg=float(c.fold_stop_deg), settle_steps=90)
    report("on-fin-tops")
    s, ok = judge()
    z_t = float(rel(scene.frames[0])[2])
    fd_t = float(scene.fold_deg()[0, 0])
    up_t = float(scene.chair_up_z()[0, 0])
    check("height band: a FOLDED chair standing ON the fin tops bridging two fins "
          f"(folded {fd_t:+.0f}deg, upright {up_t:+.2f}, over slot 0) is rejected by "
          f"the height band alone (z={z_t * 1000:.0f}mm > "
          f"{c.z_hi * 1000:.0f}mm) -> not covered",
          fd_t > c.fold_min_deg and up_t > 0.95 and z_t > c.z_hi + 0.010
          and not bool(scene.covered()[0].any()) and not ok)

    # =========================== 11. occupied slot ==========================================
    # Anchor: fold red by TELEPORT (instrumentation) and lower it into slot 1 with the
    # solve's verified wrench controller — the slot genuinely covers (this is also the
    # folded-vs-deployed CONTRAST for check 8). Then drop folded green onto the SAME
    # occupied slot: it cannot reach the band; the slot still counts exactly one.
    env.reset(seed=81)
    step(30)
    place_chair(0, (0.05, -0.30), c.origin_h + 0.003, 0.0,
                fold_deg=float(c.fold_stop_deg), settle_steps=45)
    anchor = lower_into_slot(0, 1)
    report("covered-anchor")
    assert_cov = anchor and bool(scene.covered()[0, 1])
    # folded green PRESSED down onto the occupied slot: spawned 6 mm above racked
    # red's highest point (the folded seat's free-edge tip — above the backrest top,
    # no interpenetration), then driven down at 1.5x its weight for 1.5 s while the
    # probe wrench holds it upright + centered (so it cannot tumble off the
    # knife-edge seat tip and bat red's bistable seat open). Its foot rides red's
    # folded seat >100 mm above the band — 2 x 95 mm > the 115 mm gap. The intruder
    # is then removed by teleport (transport only), never dropped.
    seat_tip = (c.hinge[2]
                + (c.seat_com[0] + c.seat_size[0] / 2)
                * math.sin(math.radians(c.fold_stop_deg))
                + c.seat_size[2] / 2)
    gz0 = c.rack_floor_z + seat_tip + 0.0005 + c.origin_h
    yaw_tgt = scene.rack_yaw + math.pi / 2
    place_chair(1, slot_xy(1), gz0, yaw_tgt, fold_deg=float(c.fold_stop_deg),
                settle_steps=0)
    fd_g0 = float(scene.fold_deg()[0, 1])  # arrives folded
    gframe = scene.frames[1]
    tgt = slot_xy(1) + scene.env_origins[:, 0:2]
    gz_min, n_rack_max = 10.0, 0
    for _ in range(180):
        p = gframe.data.root_pos_w
        gz_min = min(gz_min, float((p[:, 2] - scene.env_origins[:, 2])[0]))
        v = gframe.data.root_lin_vel_w
        w = gframe.data.root_ang_vel_w
        f = torch.zeros(n, 3, device=device)
        f[:, 0:2] = (12.0 * (tgt - p[:, 0:2]) - 5.0 * v[:, 0:2]).clamp(-1.5, 1.5)
        f[:, 2] = -1.5 * W
        q = gframe.data.root_quat_w
        z_b = quat_apply(q, ez_w)
        t = (0.40 * torch.cross(z_b, ez_w, dim=1) - 0.020 * w).clamp(-0.15, 0.15)
        fx_w = quat_apply(q, ex_b)
        yaw_now = torch.atan2(fx_w[:, 1], fx_w[:, 0])
        yerr = torch.atan2(torch.sin(yaw_tgt - yaw_now),
                           torch.cos(yaw_tgt - yaw_now))
        t[:, 2] = (0.060 * yerr - 0.012 * w[:, 2]).clamp(-0.06, 0.06)
        wrench(gframe, f, t)
        env.step(no_action)
        gz_min = min(gz_min, float((gframe.data.root_pos_w[0, 2]
                                    - scene.env_origins[0, 2])))
        n_rack_max = max(n_rack_max, int(
            (scene.racked() & scene.settled()[:, :, None])[0, :, 1].sum()))
    report("second-chair")
    s, ok = judge()
    clear(gframe)
    # remove the intruder by teleport — transport only, no tumble
    place_chair(1, (0.05, 0.30), c.origin_h + 0.003, 0.0, settle_steps=60)
    report("intruder-out")
    _s2, ok2 = judge()
    fd_red = float(scene.fold_deg()[0, 0])
    check("occupied slot: red genuinely LOWERED into slot 1 by the solve's wrench "
          "controller covers it (folded insertion works — the check-8 contrast); a "
          f"second chair arriving FOLDED ({fd_g0:+.0f}deg) pressed onto the SAME "
          f"slot at 1.5x its weight fell {(gz0 - gz_min) * 1000:.0f}mm to contact "
          f"(> 3, read back) then stalled riding red's folded seat at min "
          f"z={gz_min * 1000:.0f}mm, >100mm above the band — the slot NEVER counts "
          f"more than one chair (max {n_rack_max}) and red is covered + folded "
          f"({fd_red:+.0f}deg) once the intruder is removed",
          assert_cov and fd_g0 >= c.fold_min_deg and gz0 - gz_min > 0.003
          and gz_min > c.z_hi + 0.10 and n_rack_max <= 1
          and bool(scene.covered()[0, 1]) and fd_red >= c.fold_min_deg
          and not ok and not ok2)

    # =========================== 12. settle gate ============================================
    step(30)
    was_cov = bool(scene.covered()[0, 1])
    ex_r, _ey_r = rack_axes()
    for b in (scene.frames[0], scene.seats[0]):
        st = b.data.root_state_w.clone()
        st[:, 7:9] += 0.25 * ex_r
        b.write_root_state_to_sim(st, all_ids)
    step(2)
    v_now = float(scene.frames[0].data.root_lin_vel_w[0].norm())
    moving_cov = bool(scene.covered()[0, 1])
    judge()
    step(90)
    re_cov = bool(scene.covered()[0, 1])
    report("kicked")
    check("settle gate: the covered chair kicked to "
          f"{v_now:.2f} m/s is NOT covered while moving (sustained stillness "
          f"required), and re-covers once settled again (re_cov={re_cov})",
          was_cov and v_now > 0.10 and not moving_cov and re_cov)

    # =========================== 13. latched fold credit ====================================
    place_chair(0, (0.05, -0.30), c.origin_h + 0.003, 0.0, settle_steps=60)
    report("moved-away")
    s_after, ok = judge()
    expect = 2.0 * c.fold_credit / 3.0  # red + green were ever-folded; blue never
    check("latched credit: teleporting the racked chair back OUT drops the covered "
          f"credit with the physical state while the two latched fold credits "
          f"survive (score={s_after:.3f} ~= {expect:.3f})",
          abs(s_after - expect) < 0.02 and not bool(scene.covered()[0].any())
          and not ok)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.chair_foldaway")
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

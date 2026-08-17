"""Smoke / rubric-REJECTION battery for FloodlightEjectScene (sim_gen task
`light_bulb_out_i204`) — NullRobot, teleported probe states + force probes, RECORDED.

This is NOT a solution (solve.py — insert the push rod through the rear service port
and push the bulb over the retention ridge so gravity ejects it into the bin — is the
acceptance evidence that the rubric ACCEPTS a correct outcome; it passes on seeds
0/1). Every teleport here is instrumentation that CONSTRUCTS a wrong (or partial)
outcome as a settled state and asserts the rubric REJECTS it — plus applied-force
probes that prove the front-door DENIAL and the ridge RETENTION are physically real
geometry. No probe in this battery ever reaches success(), and a final audit asserts
exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: bulb SEATED in the pocket
                            behind the ridge (housing frame), rod flat on the floor,
                            bin empty; score ~0 at rest, no success;
  3-4. randomization      — READBACK over 8 seeded resets: housing xy + yaw vary and
                            the seated bulb + bin TRACK the housing frame; the rod's
                            spawn side / xy vary;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  front door denied   — seed-strategy family: reach in through the MOUTH and
                            work the bulb out. The straight-line version — a
                            regulated push (<= 1.5 N ~ 3x bulb weight) on the bulb
                            toward the mouth-facing tool's only pressing direction
                            (INTO the shroud) — moves the bulb measurably (>= 2 mm,
                            non-vacuous) and stalls it against the seat back by real
                            collision: it stays pocketed, score ~0;
  7.  ridge retention     — quasi-static approach (velocity-servo creep to ridge
                            contact), then a constant 0.25 N forward push — BELOW
                            the ~0.43 N quasi-static threshold — held 2.5 s: the
                            bulb never crosses the ridge, settles back pocketed,
                            score ~0 (the retention is a real force threshold, not
                            a scripted flag);
  8.  ejected-but-missed  — bulb constructed settled on the FLOOR beside the bin:
                            not in the bin, no transit credit, score ~0;
  9.  rim perch           — bulb balanced ON the bin's rim wall reads far above the
                            bin-frame height window -> rejected, removed before it
                            can topple;
  10. settle gate         — bulb INSIDE the bin but still moving is NOT success
                            (velocity gates are real); removed before it can settle;
  11. wrong object        — the ROD dumped into the bin with the bulb still seated
                            counts for nothing;
  12. rod-in-port         — rod HELD inserted through the service port (per-step
                            state writes — pure instrumentation): the probe latch
                            fires and score == 0.15 + eps ONLY (no unseat/eject
                            credit from mere insertion);
  13. cap                 — all three latches constructed (rod held in port; bulb
                            placed on the ramp -> rolls through the mouth window),
                            bulb TELEPORTED AWAY mid-flight before it can land:
                            score == 0.65 cap (float32 + eps), NOT success — full
                            credit short of success is impossible without the bulb
                            resting in the bin;
  14. latched credit      — 40 further steps with the bulb far away leave the
                            latched 0.65 unchanged while in_bin() stays False;
  15. rejection audit     — success() was never True at ANY judged point;
  16. final no-NaN        — all task-object states finite at the end;
  17. camera              — >= 20 rgb frames captured -> frames.npz.

Run (forge): python -u -m simgen_tasks.light_bulb_out_i204.smoke --headless
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
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.floodlight_eject")().build(num_envs=args.num_envs,
                                                      device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(3, device=device)
    c45 = math.cos(math.pi / 4)
    lie_quat = (c45, 0.0, c45, 0.0)  # rod local +z -> world +x

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.90, -1.40, 1.05)) + o),
                                tuple(np.array((0.70, 0.00, 0.20)) + o),
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

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        s, ok = judge()
        b = scene.bulb_local()[0]
        print(f"[smoke] {tag:16s} |"
              f" bulb_loc=({float(b[0]):+.3f},{float(b[1]):+.3f},{float(b[2]):+.3f})"
              f" latches=({float(scene.probe_latch[0]):.0f},"
              f"{float(scene.unseat_latch[0]):.0f},{float(scene.eject_latch[0]):.0f})"
              f" in_bin={bool(scene.in_bin()[0])}"
              f" score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, xyz, quat=None, vel=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap). `xyz` is env-relative."""
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

    def wrench(body, f3: torch.Tensor, t3: torch.Tensor) -> None:
        """World wrench expressed in the body's current link frame (the house
        convention: is_global=True silently drops the torque on this stack)."""
        q = body.data.root_link_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f3.view(1, 3).expand(n, 3)).unsqueeze(1),
            quat_apply_inverse(q, t3.view(1, 3).expand(n, 3)).unsqueeze(1),
            env_ids=all_ids)

    def h_local(off_xyz) -> tuple[float, float, float]:
        """Env-relative world position of a housing-frame offset."""
        off = torch.tensor(off_xyz, device=device).view(1, 3)
        p = scene.housing.data.root_pos_w[:1] + quat_apply(
            scene.housing.data.root_quat_w[:1], off)
        p = (p - scene.env_origins[:1])[0]
        return (float(p[0]), float(p[1]), float(p[2]))

    def bin_pos(off_xyz) -> tuple[float, float, float]:
        """Env-relative world position of a bin-frame offset."""
        off = torch.tensor(off_xyz, device=device).view(1, 3)
        p = scene.bin.data.root_pos_w[:1] + quat_apply(
            scene.bin.data.root_quat_w[:1], off)
        p = (p - scene.env_origins[:1])[0]
        return (float(p[0]), float(p[1]), float(p[2]))

    def h_axis_w(sign: float) -> torch.Tensor:
        """World unit vector along the housing's +-x axis."""
        ex = torch.zeros(1, 3, device=device)
        ex[:, 0] = sign
        return quat_apply(scene.housing.data.root_quat_w[:1], ex)[0]

    def layout_sane(tag: str) -> bool:
        """Reset honesty: bulb seated in the pocket behind the ridge (housing
        frame), rod flat on the floor outside the port, bin empty."""
        b = scene.bulb_local()[0]
        rodp = rel(scene.rod)
        ok = (abs(float(b[0]) - c.seat_x) < c.seat_jitter + 0.005
              and abs(float(b[1])) < 0.012
              and abs(float(b[2]) - (c.seat_floor_z + c.bulb_r)) < 0.010
              and float(rodp[2]) < 0.03
              and not bool(scene.rod_in_port()[0])
              and not bool(scene.in_bin()[0]))
        if not ok:
            print(f"[smoke] layout sanity VIOLATION at {tag}: "
                  f"bulb_loc=({float(b[0]):+.3f},{float(b[1]):+.3f},"
                  f"{float(b[2]):+.3f}) rod_z={float(rodp[2]):.3f}", flush=True)
        return ok

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    bodies = (scene.housing, scene.bin, scene.bulb, scene.rod)
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("settle: all states finite; bulb seated behind the ridge, rod on the "
          "floor, bin empty", fin and layout_sane("reset"))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    sane, tracks = True, True
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(30)
        sane = sane and layout_sane(f"seed {sd}")
        hp = rel(scene.housing)
        hq = scene.housing.data.root_quat_w[0]
        hyaw = 2.0 * math.atan2(float(hq[3]), float(hq[0]))
        rodp = rel(scene.rod)
        boff = quat_apply_inverse(
            scene.housing.data.root_quat_w[:1],
            scene.bin.data.root_pos_w[:1] - scene.housing.data.root_pos_w[:1])[0]
        tracks = tracks and (abs(float(boff[0]) - c.bin_off_x) < 0.003
                             and abs(float(boff[1])) < c.bin_jitter + 0.003)
        reads.append((float(hp[0]), float(hp[1]), hyaw, float(rodp[0]),
                      float(rodp[1]), float(scene.bulb_local()[0][0])))
    arr = np.array(reads)
    print("[smoke] randomization readback (hx, hy, hyaw, rod_x, rod_y, "
          f"bulb_seat_x):\n{np.round(arr, 4)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: housing xy + yaw vary and the seated bulb + bin TRACK "
          f"the housing frame (readback spread x={spread[0]:.3f} y={spread[1]:.3f} "
          f"yaw={math.degrees(spread[2]):.1f} deg; all seated, bin tracks)",
          spread[0] > 0.01 and spread[1] > 0.02 and spread[2] > 0.08
          and sane and tracks)
    sides = {float(np.sign(v)) for v in arr[:, 4]}
    check("randomization: rod spawn side / xy vary (readback: "
          f"{len(sides)} sides, x spread {spread[3]:.3f}, y spread {spread[4]:.3f})",
          spread[3] > 0.03 and (len(sides) >= 2 or spread[4] > 0.04))

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 6. front door denied (seed-strategy family) ================
    # rlbench/light_bulb_out's verb is reach-the-bulb-and-take-it-out. The mouth denies
    # the jaw by asserted geometry; a thin tool through the mouth can only PRESS the
    # bulb deeper. Regulated press (velocity servo, <= 1.5 N ~ 3x bulb weight) along
    # the housing -x axis: the bulb moves (non-vacuous) and stalls on the seat back.
    env.reset(seed=41)
    step(30)
    x0 = float(scene.bulb_local()[0][0])
    xmin, xmax = x0, x0
    d_in = h_axis_w(-1.0)
    f3 = torch.zeros(3, device=device)
    for _ in range(180):
        v_along = float((scene.bulb.data.root_lin_vel_w[0] * d_in).sum())
        f3 = d_in * min(max(2.0 * (0.10 - v_along), -1.5), 1.5)
        wrench(scene.bulb, f3, zero3)
        step(1)
        bx = float(scene.bulb_local()[0][0])
        xmin, xmax = min(xmin, bx), max(xmax, bx)
    wrench(scene.bulb, zero3, zero3)
    step(45)
    report("front-press")
    bx_end = float(scene.bulb_local()[0][0])
    s, ok = judge()
    check("front door denied: the regulated press through the mouth (<= 1.5 N) "
          f"moves the bulb {(x0 - xmin) * 1000:.1f} mm deeper (>= 2, non-vacuous) "
          f"and stalls it on the seat back by real collision — it never advances "
          f"toward the mouth (max x {xmax:+.3f} <= start + 2 mm) and settles still "
          f"pocketed (x {bx_end:+.3f}), score ~0",
          x0 - xmin >= 0.002 and xmax <= x0 + 0.002 and bx_end < -0.110
          and s <= 0.02 and not ok)

    # =========================== 7. ridge retention is a real force threshold ===============
    # Quasi-static approach: creep the bulb to ridge contact with a weak velocity
    # servo (<= 0.12 N), THEN hold a constant 0.25 N forward — BELOW the ~0.43 N
    # quasi-static threshold — for 2.5 s. A fast slew would vault the ridge with
    # kinetic energy; from rest in contact, sub-threshold force cannot start the
    # pivot at all.
    env.reset(seed=51)
    step(30)
    x0 = float(scene.bulb_local()[0][0])
    d_fwd = h_axis_w(+1.0)
    for _ in range(300):
        v_along = float((scene.bulb.data.root_lin_vel_w[0] * d_fwd).sum())
        f3 = d_fwd * min(max(2.0 * (0.04 - v_along), -0.12), 0.12)
        wrench(scene.bulb, f3, zero3)
        step(1)
    x_contact = float(scene.bulb_local()[0][0])
    xmax = x_contact
    wrench(scene.bulb, d_fwd * 0.25, zero3)
    for _ in range(300):
        step(1)
        xmax = max(xmax, float(scene.bulb_local()[0][0]))
    wrench(scene.bulb, zero3, zero3)
    step(60)
    report("ridge-push")
    bx_end = float(scene.bulb_local()[0][0])
    s, ok = judge()
    check("ridge retention: quasi-static creep advances the bulb "
          f"{(x_contact - x0) * 1000:.1f} mm to ridge contact (>= 2, non-vacuous); "
          f"a constant 0.25 N push (< 0.43 N threshold) held 2.5 s never carries it "
          f"past the ridge (max x {xmax:+.3f} < {c.unseat_x:+.3f}), it settles back "
          f"pocketed (x {bx_end:+.3f}), no unseat credit, score ~0",
          x_contact - x0 >= 0.002 and xmax < c.unseat_x
          and float(scene.unseat_latch[0]) < 0.5 and bx_end < c.ridge_x + 0.005
          and s <= 0.02 and not ok)

    # =========================== 8. ejected but MISSED the bin ==============================
    env.reset(seed=61)
    step(30)
    beside = bin_pos((0.0, c.bin_inner_y + 0.012 + c.bulb_r + 0.012, 0.0))
    place(scene.bulb, (beside[0], beside[1], c.bulb_r + 0.002), settle_steps=60)
    report("missed-bin")
    bl = scene.bin_local()[0]
    s, ok = judge()
    check("ejected-but-missed: bulb settled on the FLOOR beside the bin (bin-frame "
          f"|y|={abs(float(bl[1])) * 1000:.0f} mm > {c.bin_xy_tol_y * 1000:.0f} mm) "
          "-> not in the bin, no transit credit, score ~0",
          not bool(scene.in_bin()[0]) and float(scene.eject_latch[0]) < 0.5
          and s <= 0.02 and not ok)

    # =========================== 9. rim perch ===============================================
    env.reset(seed=63)
    step(30)
    perch = bin_pos((0.0, 0.094, c.bin_rim_z + c.bulb_r + 0.001))
    place(scene.bulb, perch, settle_steps=4)
    bl = scene.bin_local()[0]
    report("rim-perch")
    _s, ok = judge()
    perch_ok = (float(bl[2]) > c.bin_z_hi and not bool(scene.in_bin()[0]) and not ok)
    # remove the perched bulb BEFORE it can topple anywhere interesting
    place(scene.bulb, (-0.30, 0.35, c.bulb_r + 0.002), settle_steps=30)
    check("rim perch: bulb balanced ON the bin rim reads bin-frame "
          f"z={float(bl[2]) * 1000:.0f} mm > {c.bin_z_hi * 1000:.0f} mm -> rejected "
          "by the height window", perch_ok)

    # =========================== 10. settle gate ============================================
    env.reset(seed=65)
    step(30)
    inside = bin_pos((0.0, 0.0, c.bin_floor_top + c.bulb_r + 0.001))
    place(scene.bulb, inside, vel=(0.40, 0.0, 0.0), settle_steps=2)
    v_now = float(scene.bulb.data.root_lin_vel_w[0].norm())
    report("settle-gate")
    _s, ok = judge()
    gate_ok = (v_now > c.settle_lin and bool(scene.in_bin()[0]) and not ok)
    # remove it BEFORE it can settle in the bin (this battery must never succeed)
    place(scene.bulb, (-0.30, -0.35, c.bulb_r + 0.002), settle_steps=30)
    check("settle gate: bulb INSIDE the bin but moving at "
          f"{v_now:.2f} m/s is NOT success (velocity gates are real)", gate_ok)

    # =========================== 11. wrong object in the bin ================================
    env.reset(seed=71)
    step(30)
    drop = bin_pos((0.0, 0.0, 0.09))
    bq = scene.bin.data.root_quat_w[:1]
    qy90 = torch.zeros(1, 4, device=device)
    qy90[:, 0], qy90[:, 2] = c45, c45
    rq = quat_mul(bq, qy90)[0]
    place(scene.rod, drop,
          quat=(float(rq[0]), float(rq[1]), float(rq[2]), float(rq[3])),
          settle_steps=90)
    report("wrong-object")
    rl = quat_apply_inverse(scene.bin.data.root_quat_w[:1],
                            scene.rod.data.root_pos_w[:1]
                            - scene.bin.data.root_pos_w[:1])[0]
    b = scene.bulb_local()[0]
    s, ok = judge()
    check("wrong object: the ROD dumped into the bin (bin-frame "
          f"({float(rl[0]):+.3f},{float(rl[1]):+.3f},{float(rl[2]):.3f})) with the "
          f"bulb still seated (x {float(b[0]):+.3f}) counts for nothing — score ~0",
          abs(float(rl[0])) < 0.13 and abs(float(rl[1])) < 0.10
          and 0.015 < float(rl[2]) < 0.20
          and abs(float(b[0]) - c.seat_x) < c.seat_jitter + 0.006
          and s <= 0.02 and not ok)

    # =========================== 12. rod held in the port -> probe credit only ==============
    env.reset(seed=81)
    step(30)
    hq = scene.housing.data.root_quat_w[:1]
    qy90n = torch.zeros(1, 4, device=device)
    qy90n[:, 0], qy90n[:, 2] = c45, c45
    rod_hold = torch.zeros(n, 13, device=device)
    hold_p = h_local((-0.310, 0.0, c.axis_z))
    rod_hold[:, 0:3] = torch.tensor(hold_p, device=device) + scene.env_origins
    rod_hold[:, 3:7] = quat_mul(hq, qy90n)
    for _ in range(30):  # held insertion: tip at housing x ~ -0.160, before the bulb
        scene.rod.write_root_state_to_sim(rod_hold, all_ids)
        step(1)
        scene.rod.write_root_state_to_sim(rod_hold, all_ids)
    report("rod-in-port")
    b = scene.bulb_local()[0]
    s_port, ok = judge()
    check("rod-in-port: rod HELD inserted through the service port fires the probe "
          f"latch and NOTHING else — score {s_port:.4f} == 0.15 + eps, bulb "
          f"undisturbed (x {float(b[0]):+.3f}), not success",
          float(scene.probe_latch[0]) > 0.5 and float(scene.unseat_latch[0]) < 0.5
          and float(scene.eject_latch[0]) < 0.5 and 0.149 <= s_port <= 0.15 + 1e-5
          and abs(float(b[0]) - c.seat_x) < c.seat_jitter + 0.006 and not ok)
    # park the rod back on the floor; the probe credit must stay latched
    place(scene.rod, (0.35, 0.30, c.rod_r + 0.003), quat=lie_quat, settle_steps=30)

    # =========================== 13. cap: all latches, bulb yanked mid-flight ===============
    # Construct unseat + eject on the same episode (probe already latched): bulb
    # placed on the RAMP inside the unseat window; it rolls out through the mouth
    # window by gravity; TELEPORT it away mid-flight, before it can land in the bin.
    ramp_z = 0.220 - (0.032 / 0.088) * (-0.05 + 0.088)
    place(scene.bulb, h_local((-0.05, 0.0, ramp_z + c.bulb_r + 0.001)),
          settle_steps=0)
    caught = False
    for _ in range(300):
        step(1)
        bx = float(scene.bulb_local()[0][0])
        if float(scene.eject_latch[0]) > 0.5 and bx > c.mouth_win[1] + 0.004:
            place(scene.bulb, (-0.30, -0.40, c.bulb_r + 0.002), settle_steps=0)
            caught = True
            break
    step(45)
    report("cap")
    s_cap, ok = judge()
    check("cap: probe + unseat + eject all latched with the bulb yanked away "
          f"mid-flight -> score {s_cap:.4f} == 0.65 cap (float32 + eps), NOT "
          "success — full credit short of success is impossible without the bulb "
          "resting in the bin",
          caught and float(scene.unseat_latch[0]) > 0.5
          and float(scene.eject_latch[0]) > 0.5
          and 0.649 <= s_cap <= 0.65 + 1e-5
          and not bool(scene.in_bin()[0]) and not ok)

    # =========================== 14. latched credit survives ================================
    step(40)
    report("latched")
    s_after, ok = judge()
    check("latched credit: 40 further steps with the bulb far away leave the "
          f"latched score unchanged ({s_cap:.2f} -> {s_after:.2f}) while in_bin() "
          "stays False",
          abs(s_after - s_cap) < 1e-3 and not bool(scene.in_bin()[0]) and not ok)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== 17. camera + save + verdict ================================
    check(f"camera: >= 20 rgb frames captured ({len(frames)})", len(frames) >= 20)
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.floodlight_eject")
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

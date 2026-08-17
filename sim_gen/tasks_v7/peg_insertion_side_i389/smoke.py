"""Smoke / rubric-REJECTION battery for LineBoreScene (sim_gen task
`peg_insertion_side_i389`) — NullRobot, teleported probe states + force probes, RECORDED.

This is NOT a solution (solve.py — drop-seat the silver coupler, then held-carry
PD-thread the headed shaft through window -> channel -> window to flush — is the
acceptance evidence that the rubric ACCEPTS a correct outcome). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled state and
asserts the rubric REJECTS it — plus an applied-force probe that proves the decoy's
refusal is physically real. No probe in this battery ever reaches success(), and a
final audit asserts exactly that.

  1-2.  settle/no-NaN     — reset layout settles finite: fixture at the workspace,
                            both blocks upright at their (possibly swapped) slots,
                            shaft lying at its slot; score ~0 at rest, no success;
  3-4.  randomization     — READBACK over 8 seeded resets: BOTH coupler/decoy slot
                            assignments occur, fixture yaw/xy jitter is physically
                            posed and varies; object slot jitter is real and the
                            blocks track the fixture pose;
  5.    null policy       — 240 idle steps -> score ~0, no success;
  6.    seed strategy      — the seed's verb (insert the peg into the PRE-EXISTING
        (partial)           hole) built literally: shaft tipped into the near window
                            only, coupler never touched -> score ~0, no success;
  7.    seed strategy      — FLAGSHIP: shaft threaded through BOTH windows to a
        (full, flagship)    flush, spanning rest on the sills — geometrically a
                            perfect seed-style insertion — with the coupler still at
                            its slot: installed() reads True yet seated() and
                            encircle are False -> score ~0, no success. The passage
                            must be BUILT before threading counts;
  8.    decoy refusal     — decoy dropped into the pocket (same funnel), then a
                            regulated force probe (velocity-servo push + bore
                            centering, same effort class as solve) drives the shaft
                            into the decoy's 16 mm channel: it MOVES >= 8 mm into
                            the window (non-vacuous) then stalls at the decoy face,
                            never passes, score ~0 — the wrong block dead-ends;
  9.    seat near-miss    — coupler standing against the pylon OUTER face (touching
                            the fixture, 30 mm from the pocket) -> seated() False,
                            score ~0;
  10.   seat-only credit  — coupler dropped + keyed into the pocket -> score ==
                            0.30 band exactly, no success;
  11.   depth near-miss   — shaft threaded through all three gates but stopped
                            ~30 mm short of flush: spanning True yet flush False ->
                            no success, score in the partial band (~0.71);
  12.   latched credit    — teleporting that shaft off to the floor leaves the
                            latched score unchanged, still no success;
  13.   roof route        — shaft laid across the pylon TOPS (over the windows) with
                            the coupler seated -> bore z-band rejects, score stays
                            at the 0.30 seat band;
  14.   settle gate       — full assembled geometry with injected axial velocity is
                            NOT success at the judged instant (settle gates real);
                            shaft removed before it can settle;
  15.   rejection audit   — success() was never True at ANY judged point;
  16.   final no-NaN      — all task-object states finite at the end;
  17.   camera            — >= 20 rgb frames captured -> frames.npz.

Run (forge): python -u -m simgen_tasks.peg_insertion_side_i389.smoke --headless
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.line_bore")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    x_out = c.gap / 2 + c.pyl_t
    half = c.shank / 2
    seat_z = c.deck[2] + c.block_hz / 2
    win_lo = c.win_z - c.win / 2
    flush_cx = -x_out + half  # shaft center x at flush (entry -x, tip +x)
    start_cx = -x_out - 0.006 - half

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.10, -1.10, 0.85)) + o),
                                tuple(np.array((0.0, 0.0, 0.10)) + o),
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

    def report(tag: str) -> None:
        s, ok = judge()
        m = scene.shaft_metrics()
        p = m["p"][0]
        b = scene.to_canon(scene.coupler.data.root_pos_w)[0]
        print(f"[smoke] {tag:16s} | slot={float(scene.slot_sign[0]):+.0f}"
              f" shaft=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f})"
              f" coupler=({float(b[0]):+.3f},{float(b[1]):+.3f},{float(b[2]):+.3f})"
              f" seated={bool(scene.seated()[0])}"
              f" inst={bool(scene.installed()[0])} enc={bool(m['encircle'][0])}"
              f" latches=({float(scene.seat_latch[0]):.0f},"
              f"{float(scene.engage_latch[0]):.0f},{float(scene.depth_latch[0]):.2f})"
              f" score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, canon_xyz, vel=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement at a fixture-CANONICAL point, orientation
        fixture-aligned (instrumentation, not a solution) + REAL physics steps
        before judging (the zero-step trap). Tracks the sampled fixture pose."""
        canon = torch.tensor(canon_xyz, device=device).view(1, 3).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.canon_to_world(canon.clone())
        st[:, 3:7] = scene.fixture.data.root_quat_w
        if vel is not None:
            st[:, 7:10] = torch.tensor(vel, device=device)
        body.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def park(body, xyz, settle_steps: int = 30) -> None:
        """Env-relative world placement on the open floor (removal)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor(xyz, device=device)
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def wrench(f_world: torch.Tensor, t_world: torch.Tensor) -> None:
        """World wrench on the shaft in its CURRENT body frame (house convention)."""
        from isaaclab.utils.math import quat_apply_inverse

        q = scene.shaft.data.root_link_quat_w
        scene.shaft.set_external_force_and_torque(
            quat_apply_inverse(q, f_world).unsqueeze(1),
            quat_apply_inverse(q, t_world).unsqueeze(1), env_ids=all_ids)

    def clear_wrench() -> None:
        z = torch.zeros(n, 1, 3, device=device)
        scene.shaft.set_external_force_and_torque(z, z, env_ids=all_ids)

    def in_pocket(body) -> bool:
        """Generic (any-block) seat readback in the fixture canonical frame."""
        p = scene.to_canon(body.data.root_pos_w)[0]
        up = scene._axis_canon(body, (0.0, 0.0, 1.0))[0]
        return (abs(float(p[0])) <= c.seat_x_tol and abs(float(p[1])) <= c.seat_y_tol
                and abs(float(p[2]) - seat_z) <= c.seat_z_tol
                and float(up[2]) >= c.seat_up_min)

    def seat_block(body) -> bool:
        """Drop-seat a block into the pocket (the same contact funnel solve uses)."""
        for _attempt in range(4):
            place(body, (0.0, 0.0, seat_z + 0.015), settle_steps=150)
            if in_pocket(body):
                return True
        return False

    def layout_sane(tag: str) -> bool:
        """Reset honesty: fixture within jitter of the workspace, coupler and decoy
        upright at their slot_sign-swapped slots, shaft lying at its slot."""
        ss = float(scene.slot_sign[0])
        fp = (scene.fixture.data.root_pos_w - scene.env_origins)[0]
        pc = scene.to_canon(scene.coupler.data.root_pos_w)[0]
        pd = scene.to_canon(scene.decoy.data.root_pos_w)[0]
        ps = scene.to_canon(scene.shaft.data.root_pos_w)[0]
        upc = scene._axis_canon(scene.coupler, (0.0, 0.0, 1.0))[0]
        upd = scene._axis_canon(scene.decoy, (0.0, 0.0, 1.0))[0]
        tol = c.slot_jitter + 0.04
        ok = (abs(float(fp[0])) <= c.pos_jitter + 0.01
              and abs(float(fp[1])) <= c.pos_jitter + 0.01
              and abs(float(pc[0]) - c.block_slot[0]) <= tol
              and abs(float(pc[1]) - ss * c.block_slot[1]) <= tol
              and abs(float(pc[2]) - c.block_hz / 2) <= 0.012
              and float(upc[2]) >= 0.95
              and abs(float(pd[0]) - c.block_slot[0]) <= tol
              and abs(float(pd[1]) + ss * c.block_slot[1]) <= tol
              and float(upd[2]) >= 0.95
              and abs(float(ps[0]) - c.shaft_slot[0]) <= tol + 0.04
              and abs(float(ps[1]) - c.shaft_slot[1]) <= tol + 0.04
              and 0.004 <= float(ps[2]) <= 0.040
              and not bool(scene.seated()[0]))
        if not ok:
            print(f"[smoke] layout sanity VIOLATION at {tag}: fp={fp} pc={pc} "
                  f"pd={pd} ps={ps}", flush=True)
        return ok

    bodies = None

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    bodies = (scene.fixture, scene.coupler, scene.decoy, scene.shaft)
    report("reset")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("settle: all states finite; fixture at the workspace, both blocks upright "
          "at their slots, shaft lying at its slot", fin and layout_sane("reset"))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    sane = True
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(30)
        sane = sane and layout_sane(f"seed {sd}")
        q = scene.fixture.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        fp = (scene.fixture.data.root_pos_w - scene.env_origins)[0]
        pc = scene.to_canon(scene.coupler.data.root_pos_w)[0]
        ps = scene.to_canon(scene.shaft.data.root_pos_w)[0]
        reads.append((float(scene.slot_sign[0]), yaw, float(fp[0]), float(fp[1]),
                      float(pc[0]), float(pc[1]), float(ps[0]), float(ps[1])))
    arr = np.array(reads)
    print("[smoke] randomization readback (slot, yaw, fx, fy, coup_cx, coup_cy, "
          f"shaft_cx, shaft_cy):\n{np.round(arr, 4)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    slots = {float(x) for x in arr[:, 0]}
    check("randomization: BOTH coupler/decoy slot assignments occur and the fixture "
          f"yaw/xy jitter is physically posed and varies (readback: {len(slots)} "
          f"slot signs, yaw spread {math.degrees(spread[1]):.1f} deg, xy spread "
          f"({spread[2]:.3f},{spread[3]:.3f}))",
          len(slots) == 2 and spread[1] > 0.03
          and (spread[2] > 0.008 or spread[3] > 0.008) and sane)
    check("randomization: object slot jitter is real — coupler canonical spread "
          f"({spread[4]:.3f},{spread[5]:.3f} — y includes the slot swap), shaft "
          f"({spread[6]:.3f},{spread[7]:.3f})",
          (spread[4] > 0.008 or spread[5] > 0.008)
          and (spread[6] > 0.008 or spread[7] > 0.008))

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 6. seed strategy, partial ==================================
    # The seed's verb — insert the peg into a PRE-EXISTING hole — built literally:
    # shaft tipped into the near window only, coupler never touched.
    env.reset(seed=41)
    step(30)
    place(scene.shaft, (-x_out + 0.005 - half, 0.0, c.win_z), settle_steps=90)
    report("window-only")
    s, ok = judge()
    check("seed strategy (partial): shaft tipped into the near window only, coupler "
          f"never touched -> score ~0 ({s:.3f}), no success", s <= 0.02 and not ok)

    # =========================== 7. seed strategy, full (FLAGSHIP) ==========================
    # A geometrically PERFECT seed-style insertion: shaft threaded through BOTH
    # windows, spanning + flush, resting on the sills — but the coupler was never
    # seated. installed() reads True; seated() and encircle reject it.
    env.reset(seed=51)
    step(30)
    place(scene.shaft, (flush_cx, 0.0, win_lo + c.shaft_r), settle_steps=90)
    report("bypass")
    m = scene.shaft_metrics()
    inst = bool(scene.installed()[0])
    s, ok = judge()
    check("FLAGSHIP bypass: shaft threaded through BOTH windows to a flush, "
          f"spanning rest on the sills (installed={inst}, spanning="
          f"{bool(m['spanning'][0])}, flush={bool(m['flush'][0])}) with the coupler "
          f"still at its slot -> seated False, encircle False, score ~0 ({s:.3f}), "
          "no success — threading before building the passage is worthless",
          inst and not bool(scene.seated()[0]) and not bool(m["encircle"][0])
          and s <= 0.02 and not ok)

    # =========================== 8. decoy refusal (force probe) =============================
    env.reset(seed=61)
    step(30)
    seated_decoy = seat_block(scene.decoy)
    # settle_steps=0: the held-carry servo starts the instant the hover teleport
    # lands (solve's regime) — letting the shaft fall first tips it and the
    # alignment torque then fights a large axis error on a tiny inertia
    place(scene.shaft, (start_cx, 0.0, c.win_z), settle_steps=0)
    from isaaclab.utils.math import quat_apply

    ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
    fx_axis = quat_apply(scene.fixture.data.root_quat_w, ex)
    tip0 = float(scene.shaft_metrics()["tip"][0, 0])
    tip_max = tip0
    mg = c.shaft_mass * 9.81
    # press ALONG THE SILL (carrot z at the sill-rest height, not levitating at the
    # bore center): the sill contact + friction damp the jam so the sustained press
    # cannot ring up a free-floating limit cycle
    probe_z = win_lo + c.shaft_r
    last_gain = 0
    for i in range(700):
        p_c = scene.to_canon(scene.shaft.data.root_pos_w)
        v_w = scene.shaft.data.root_lin_vel_w
        # axial: velocity servo (kv*dt/m = 10/(120*0.15) = 0.56 < 1), stall drive
        # ~2 N (~2.7x sliding friction) — same effort class as solve's servo
        vx = float((v_w[0] * fx_axis[0]).sum())
        f_ax = min(max(10.0 * (0.20 - vx), -1.0), 3.0)
        # lateral: PD centering onto the bore line (canonical y=0, z=sill rest)
        carrot = torch.tensor([float(p_c[0, 0]), 0.0, probe_z],
                              device=device).view(1, 3)
        tgt_w = scene.canon_to_world(carrot.expand(n, 3).clone())
        f = 40.0 * (tgt_w - scene.shaft.data.root_pos_w) - 6.0 * v_w
        f = f - (f * fx_axis).sum(dim=-1, keepdim=True) * fx_axis  # lateral only
        fn = f.norm(dim=-1, keepdim=True)
        f = f * (fn.clamp(max=5.0) / fn.clamp_min(1e-9))
        f = f + f_ax * fx_axis
        f[:, 2] += mg
        a_cur = quat_apply(scene.shaft.data.root_quat_w, ex)
        tq = 0.4 * torch.cross(a_cur, fx_axis, dim=-1) \
            - 0.010 * scene.shaft.data.root_ang_vel_w
        wrench(f, tq)
        step(1)
        tip_now = float(scene.shaft_metrics()["tip"][0, 0])
        if not math.isfinite(tip_now):
            print(f"[smoke] decoy probe went non-finite @step {i}", flush=True)
            break
        if tip_now > tip_max + 0.0005:
            tip_max, last_gain = tip_now, i
        # stall confirmed: pressed for 1 s with no further gain — stop grinding
        # (a sustained wedged press is what excites edge-contact limit cycles)
        if i - last_gain > 120 and tip_max - tip0 >= 0.008:
            print(f"[smoke] decoy stall confirmed @step {i} "
                  f"(tip_max={tip_max:+.4f})", flush=True)
            break
    clear_wrench()
    step(45)
    report("decoy-stall")
    m = scene.shaft_metrics()
    s, ok = judge()
    decoy_face = -c.block_lx / 2  # decoy near face (canonical x) when seated
    check("decoy refusal (force probe): decoy dropped + keyed into the pocket "
          f"(in_pocket={seated_decoy}), regulated push drives the shaft "
          f"{(tip_max - tip0) * 1000:.0f} mm forward into the window (>= 8, "
          f"non-vacuous) but the tip stalls at the decoy face (max tip x "
          f"{tip_max:+.3f} < {decoy_face + 0.006:+.3f}) — the 16 mm channel refuses "
          f"the 20 mm shank; never spanning, score ~0 ({s:.3f}), no success",
          seated_decoy and tip_max - tip0 >= 0.008
          and tip_max < decoy_face + 0.006 and not bool(m["spanning"][0])
          and s <= 0.02 and not ok)

    # =========================== 9. seat near-miss ==========================================
    env.reset(seed=71)
    step(30)
    place(scene.coupler, (x_out + c.block_lx / 2 + 0.004, 0.0, seat_z + 0.003),
          settle_steps=90)
    report("seat-miss")
    p = scene.to_canon(scene.coupler.data.root_pos_w)[0]
    s, ok = judge()
    check("seat near-miss: coupler standing against the pylon OUTER face (canonical "
          f"x={float(p[0]):+.3f}, {abs(float(p[0])) * 1000:.0f} mm from the pocket "
          f"center) -> seated False, score ~0 ({s:.3f})",
          not bool(scene.seated()[0]) and abs(float(p[0])) > c.seat_x_tol
          and s <= 0.02 and not ok)

    # =========================== 10-12. seat credit, depth near-miss, latched credit ========
    env.reset(seed=81)
    step(30)
    seated_c = seat_block(scene.coupler)
    report("seat-only")
    s_seat, ok = judge()
    check("seat-only credit: coupler dropped + keyed into the pocket "
          f"(seated={seated_c}) -> score == 0.30 band ({s_seat:.4f}), no success",
          seated_c and bool(scene.seated()[0]) and 0.295 <= s_seat <= 0.305
          and not ok)

    place(scene.shaft, (flush_cx - 0.030, 0.0, c.win_z), settle_steps=90)
    report("short-thread")
    m = scene.shaft_metrics()
    rem = float(m["remaining"][0])
    s_short, ok = judge()
    check("depth near-miss: shaft threaded through all three gates but stopped "
          f"{rem * 1000:.0f} mm short of flush — spanning={bool(m['spanning'][0])} "
          f"yet flush={bool(m['flush'][0])} -> no success, score in the partial "
          f"band ({s_short:.3f})",
          bool(m["spanning"][0]) and not bool(m["flush"][0])
          and c.flush_tol + 0.005 <= rem <= 0.045
          and 0.55 <= s_short <= 0.95 and not ok)

    park(scene.shaft, (-0.9, -0.9, c.head_r + 0.002), settle_steps=40)
    report("moved-away")
    s_after, ok = judge()
    check("latched credit: teleporting that shaft off to the floor leaves the "
          f"latched score unchanged ({s_short:.3f} -> {s_after:.3f}), still no "
          "success", abs(s_after - s_short) < 1e-3 and not ok)

    # =========================== 13. roof route =============================================
    env.reset(seed=91)
    step(30)
    seated_c = seat_block(scene.coupler)
    place(scene.shaft, (0.0, 0.0, c.deck[2] + c.pyl_h + c.shaft_r + 0.002),
          settle_steps=60)
    report("roof")
    m = scene.shaft_metrics()
    p = m["p"][0]
    s, ok = judge()
    check("roof route: shaft laid across the pylon TOPS (canonical z="
          f"{float(p[2]) * 1000:.0f} mm > bore band {c.bore_z_hi * 1000:.0f} mm) "
          f"with the coupler seated ({seated_c}) -> in_band False, score stays at "
          f"the seat band ({s:.3f})",
          seated_c and float(p[2]) > c.bore_z_hi + 0.02
          and not bool(m["in_band"][0]) and 0.295 <= s <= 0.305 and not ok)

    # =========================== 14. settle gate ============================================
    env.reset(seed=101)
    step(30)
    seated_c = seat_block(scene.coupler)
    # assembled geometry, but MOVING axially (backing out) — judge immediately
    place(scene.shaft, (flush_cx, 0.0, win_lo + c.shaft_r), vel=(-0.4, 0.0, 0.0),
          settle_steps=1)
    m = scene.shaft_metrics()
    v_now = float(scene.shaft.data.root_lin_vel_w[0].norm())
    inst_now = bool(scene.installed()[0])
    settled_now = bool(scene.settled()[0])
    _s, ok = judge()
    gate_ok = (seated_c and inst_now and v_now > c.settle_lin
               and not settled_now and not ok)
    # remove it BEFORE it can settle assembled (this battery must never succeed)
    park(scene.shaft, (-0.9, 0.9, c.head_r + 0.002), settle_steps=30)
    report("settle-gate")
    check("settle gate: fully assembled geometry (installed="
          f"{inst_now}) but MOVING (|v|={v_now:.2f} m/s) is NOT success at the "
          "judged instant; shaft removed before it can settle", gate_ok)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== 17. camera + save + verdict ================================
    check(f"camera: >= 20 rgb frames captured ({len(frames)})", len(frames) >= 20)
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.line_bore")
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

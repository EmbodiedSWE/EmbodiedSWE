"""Smoke / rubric-REJECTION battery for WedgeGateCabinetScene (sim_gen task
`slide_cabinet_open_and_place_cups_i195`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — park cups, drive the wedge into the slit, slide
both cups through the propped gap, extract the wedge, gate falls closed — is the
acceptance evidence; it passes on forge seeds 0/1/2). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome and asserts the rubric
REJECTS it — plus physical probes that prove the gravity-return gate and the ramp
wedge are a working mechanism, not a prop. No probe in this battery ever reaches
success(), and a final audit check asserts exactly that.

   1. settle/no-NaN      — reset settles finite: gate on its lower stop, cups upright
                           on the floor, wedge flat on the ground, everything settled;
   2. fresh reset        — score ~0, no success;
  3-4. randomization     — READBACK over 6 seeded resets: the cups' slot assignment
                           varies (never sharing a slot) with real xy jitter; the
                           wedge's spawn xy and free yaw vary;
   5. null policy        — 240 idle steps -> score ~0, no success;
   6. gravity return     — the gate CAN be forced open directly (regulated vertical
                           force, lift readback past full prop height) but the moment
                           the force is zeroed it falls back CLOSED on its own: there
                           is no state where the cabinet stays open unpropped;
   7. sealed doorway     — the solve's own tip-safe floor push aimed straight at the
                           closed gate is DENIED: the cup travels to the gate face
                           (displacement readback), stalls outside, stays upright,
                           and the gate does not budge;
   8. flush face         — the wedge driven TAIL-FIRST (re-aimed 180 deg) at the same
                           slit stalls against the gate's flush face (displacement
                           readback) and produces NO lift: tool orientation is
                           load-bearing, only the thin tip fits the slit;
   9. seed-analog        — the seed family's end state ("door open, cups placed"):
                           wedge physically driven in (solve's servo), both cups
                           inside — the gate READS not-closed (resting on the plateau,
                           height readback) -> NOT success, score <= latched 0.65;
  10. tip-in-slit miss   — from 9, the wedge withdrawn until the gate is back on its
                           stop but the tool is NOT yet clear: gate closed AND cups in
                           yet wedge_clear False -> NOT success;
  11. lying cup          — a cup on its side inside: the z band PASSES (readback) but
                           uprightness rejects it;
  12. z-band rejections  — a cup STACKED on a contained cup, and a cup on the ROOF
                           over the interior: both rejected by the z band;
  13. gate-hugging cup   — a cup physically inside but pressed against the closed
                           gate's back face: rejected by the y band (must be CLEAR of
                           the gate);
  14. wedge left inside  — both cups in, gate closed, but the wedge parked ON the
                           cabinet floor inside -> NOT success (tool must end clear);
  15. latched credit     — removing a contained cup leaves the latched score unchanged
                           while both_inside() drops;
  16. settle gate        — the full success geometry judged while the last cup is
                           still FALLING is NOT success; the cup is removed before
                           ring-down completes (the battery never succeeds);
  17. rejection audit    — success() was never True at ANY judged point;
  18. final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.slide_cabinet_open_and_place_cups_i195.smoke --headless
"""

from __future__ import annotations

import argparse
import math

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

DT = 1.0 / 120.0
UP = (1.0, 0.0, 0.0, 0.0)
FLIP = (0.0, 0.0, 0.0, 1.0)          # yaw 180 deg: tail toward the cabinet
LYING = (0.70710678, 0.0, 0.70710678, 0.0)  # 90 deg about y — cup axis along x


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.wedge_gate_cabinet")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import quat_apply_inverse

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.95, 0.95, 0.65)) + o),
                                tuple(np.array((0.00, 0.02, 0.06)) + o),
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

    def gate_j() -> float:
        return float(scene.gate_j()[0])

    def wedge_y() -> float:
        return float(scene.wedge_local()[0, 1])

    def cup_loc(nm: str) -> tuple[float, float, float]:
        p = scene._cup_local(nm)[0]
        return float(p[0]), float(p[1]), float(p[2])

    def report(tag: str) -> None:
        s, ok = judge()
        bits = []
        for nm in c.cup_names:
            x, y, z = cup_loc(nm)
            bits.append(f"{nm}=({x:+.3f},{y:+.3f},{z:+.3f}) in={bool(scene.cup_inside(nm)[0])}")
        print(f"[smoke] {tag:16s} | j={gate_j():+.4f} wy={wedge_y():+.3f} | "
              + " ".join(bits) + f" score={s:.3f} success={ok} frames={len(frames)}",
              flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def write_body(body, xyz, quat=UP, settle_steps: int = 60) -> None:
        """Kinematic probe placement in the ENV (== cabinet) frame + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = float(xyz[0])
        st[:, 1] = float(xyz[1])
        st[:, 2] = float(xyz[2])
        st[:, 0:3] += scene.env_origins
        st[:, 3:7] = torch.tensor(quat, device=device)
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def push_y(body, y_target: float, *, v_cap: float, kv: float, ki: float,
               f_max: float, max_steps: int) -> tuple[float, float]:
        """The solve's velocity-regulated PI floor push along world y (world->body
        every step). Returns (final y, max gate_j observed during the push)."""
        acc = torch.zeros(n, device=device)
        max_j = gate_j()
        done = 0
        for _ in range(max_steps):
            y = (body.data.root_pos_w - scene.env_origins)[:, 1]
            v = body.data.root_lin_vel_w[:, 1]
            err = y_target - y
            v_des = (3.0 * err).clamp(-v_cap, v_cap)
            near = err.abs() <= 0.005
            acc = (acc + ki * (v_des - v) * DT).clamp(-f_max, f_max)
            acc = torch.where(near, torch.zeros_like(acc), acc)
            fy = (acc + kv * (v_des - v)).clamp(-f_max, f_max)
            slow = v.abs() < 0.02
            fy = torch.where(near & slow, torch.zeros_like(fy), fy)
            f_world = torch.zeros(n, 3, device=device)
            f_world[:, 1] = fy
            f_body = quat_apply_inverse(body.data.root_quat_w, f_world)
            body.set_external_force_and_torque(f_body.unsqueeze(1), zero_w,
                                               env_ids=all_ids)
            env.step(no_action)
            max_j = max(max_j, gate_j())
            if bool(near[0]) and bool(slow[0]):
                done += 1
                if done >= 12:
                    break
            else:
                done = 0
        body.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
        step(30)
        return float((body.data.root_pos_w - scene.env_origins)[0, 1]), max_j

    bodies = (scene.cabinet, scene.gate, scene.wedge, *scene.cups.values())
    REST = c.cup_h / 2 + 0.003

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    z_ok = all(abs(cup_loc(nm)[2] - c.cup_h / 2) < 0.012 for nm in c.cup_names)
    wz = float((scene.wedge.data.root_pos_w - scene.env_origins)[0, 2])
    check("settle: all states finite, gate on its lower stop "
          f"(j={gate_j():+.4f} ~ {c.gate_j_lo}), cups upright on the floor, wedge on "
          f"the ground (z={wz:+.3f}), everything settled",
          fin and z_ok and abs(gate_j() - c.gate_j_lo) <= 0.004
          and abs(wz) < 0.02 and bool(scene.settled()[0]))
    s, ok = judge()
    check("fresh reset: score ~0, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(20)
        slot = tuple(int(np.argmin([abs(cup_loc(nm)[0] - sx) for sx in c.slot_xs]))
                     for nm in c.cup_names)
        q = scene.wedge.data.root_quat_w[0]
        w, qx, qy, qz = (float(v) for v in q)
        yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        wx, wyy = (float(v) for v in (scene.wedge.data.root_pos_w - scene.env_origins)[0, :2])
        reads.append((slot, cup_loc(c.cup_names[0])[0], wx, wyy, yaw))
        print(f"[smoke] seed {sd}: slots={slot} cup_a_x={cup_loc(c.cup_names[0])[0]:+.3f} "
              f"wedge=({wx:+.3f},{wyy:+.3f}) yaw={math.degrees(yaw):+.0f}deg", flush=True)
    slots = {r[0] for r in reads}
    x_spread = max(r[1] for r in reads) - min(r[1] for r in reads)
    check("randomization: cup slot assignment varies (readback: "
          f"{len(slots)} distinct / 6, never sharing a slot) and xy jitter is real "
          f"(cup_a x spread {x_spread * 1000:.0f} mm)",
          len(slots) >= 3 and all(r[0][0] != r[0][1] for r in reads) and x_spread > 0.01)
    wx_spread = max(r[2] for r in reads) - min(r[2] for r in reads)
    yaws = [r[4] for r in reads]
    yaw_spread = max(math.degrees(math.acos(max(-1.0, min(1.0,
                     math.cos(a - b))))) for a in yaws for b in yaws)
    check(f"randomization: wedge spawn varies — x spread {wx_spread * 1000:.0f} mm, "
          f"free yaw spread {yaw_spread:.0f} deg (must be re-aimed before it fits the slit)",
          wx_spread > 0.01 and yaw_spread > 30.0)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 6. gravity return: forced open, falls back closed =========
    env.reset(seed=41)
    step(30)
    f = torch.zeros(n, 1, 3, device=device)
    max_j = gate_j()
    mg = 9.81 * c.gate_mass  # feedforward: a pure P velocity loop's stall force
    for _ in range(360):     # (KV*V_CAP = 0.6 N) is far below the 2.94 N gate weight
        j = scene.gate_j()   # (body frame == world: a prismatic plate never rotates)
        vz = scene.gate.data.root_lin_vel_w[:, 2]
        v_des = (3.0 * (0.070 - j)).clamp(-0.10, 0.10)
        f[:, 0, 2] = (mg + 6.0 * (v_des - vz)).clamp(0.0, 8.0)
        scene.gate.set_external_force_and_torque(f, zero_w, env_ids=all_ids)
        step(1)
        max_j = max(max_j, gate_j())
    scene.gate.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
    step(240)
    report("gravity-return")
    s, ok = judge()
    check("gravity return: a regulated vertical force lifts the gate past full prop "
          f"height (max j={max_j:+.4f} >= {c.deep_j_min}) but the instant it is zeroed "
          f"the gate falls back CLOSED on its own (j={gate_j():+.4f} <= "
          f"{c.closed_j_max}) — no state stays open unpropped; only the lift latches "
          f"score ({s:.2f})",
          max_j >= c.deep_j_min + 0.004 and gate_j() <= c.closed_j_max
          and 0.28 <= s <= 0.31 and not ok)

    # =========================== 7. sealed doorway physically denies a cup ==================
    # The solve's own tip-safe push, aimed straight at the CLOSED gate.
    write_body(scene.cups[c.cup_names[0]], (0.0, 0.150, REST), UP, 30)
    y_end, max_j7 = push_y(scene.cups[c.cup_names[0]], -0.130, v_cap=0.05, kv=3.0,
                           ki=25.0, f_max=0.50, max_steps=600)
    report("sealed-push")
    s, ok = judge()
    check("sealed doorway: the tip-safe floor push aimed at the closed gate is DENIED "
          f"— the cup travelled to the gate face and stalled OUTSIDE (y={y_end:+.3f}, "
          f"gate front {c.gate_y + c.gate_t / 2:+.3f}), upright, and the gate never "
          f"lifted (max j={max_j7:+.4f})",
          0.015 <= y_end <= 0.060 and bool(scene.cup_upright(c.cup_names[0])[0])
          and not bool(scene.cup_inside(c.cup_names[0])[0])
          and max_j7 <= c.closed_j_max + 0.004 and not ok)

    # =========================== 8. flush face: tail-first wedge cannot open ================
    env.reset(seed=51)
    step(30)
    write_body(scene.wedge, (c.wedge_insert_x, 0.170, 0.0035), FLIP, 30)
    y_end, max_j8 = push_y(scene.wedge, -0.075, v_cap=0.05, kv=8.0, ki=60.0,
                           f_max=6.0, max_steps=700)
    report("tail-first")
    s, ok = judge()
    check("flush face: the wedge driven TAIL-FIRST at the slit travels to the gate "
          f"and STALLS against its flush face (y={y_end:+.3f}, moved "
          f"{(0.170 - y_end) * 1000:.0f} mm, expected stall ~+0.095) with NO lift "
          f"(max j={max_j8:+.4f} < lift_j_min {c.lift_j_min}) — only the thin tip "
          "fits the slit, orientation is load-bearing",
          0.080 <= y_end <= 0.115 and max_j8 < c.lift_j_min and s <= 0.02 and not ok)

    # =========================== 9. seed-analog: door open, cups placed, walk away =========
    env.reset(seed=61)
    step(30)
    write_body(scene.wedge, (c.wedge_insert_x, 0.170, 0.0035), UP, 30)
    y_in, _ = push_y(scene.wedge, -0.075, v_cap=0.05, kv=8.0, ki=60.0,
                     f_max=12.0, max_steps=3000)  # the solve's REAL insertion
    step(60)
    j_prop = gate_j()
    write_body(scene.cups[c.cup_names[0]], (-0.010, -0.130, c.cup_rest_z + 0.010), UP, 60)
    write_body(scene.cups[c.cup_names[1]], (+0.070, -0.130, c.cup_rest_z + 0.010), UP, 90)
    report("seed-analog")
    s, ok = judge()
    check("seed-analog (open the door, place the cups, walk away): wedge physically "
          f"driven in (y={y_in:+.3f}), gate propped at j={j_prop:+.4f}, both cups "
          "inside — but the gate resting on the plateau is NOT closed and the tool is "
          f"NOT clear -> NOT success, score <= latched 0.65 ({s:.2f})",
          j_prop >= c.deep_j_min and bool(scene.both_inside()[0])
          and not bool(scene.gate_closed()[0]) and not bool(scene.wedge_clear()[0])
          and not ok and 0.63 <= s <= 0.66)

    # =========================== 10. tip-in-slit near-miss ==================================
    # Withdraw only until the gate is back on its stop — the tip still under/at the
    # doorway (root +0.085 << clear_y_min 0.115).
    y_mid, _ = push_y(scene.wedge, +0.085, v_cap=0.06, kv=8.0, ki=60.0,
                      f_max=12.0, max_steps=1500)
    step(120)
    report("tip-in-slit")
    s, ok = judge()
    check("tip-in-slit near-miss: wedge withdrawn just short of clear "
          f"(y={y_mid:+.3f} < {c.clear_y_min}) — the gate is back on its stop "
          f"(j={gate_j():+.4f}, closed={bool(scene.gate_closed()[0])}) and both cups "
          "are in, yet wedge_clear is False -> NOT success",
          bool(scene.gate_closed()[0]) and bool(scene.both_inside()[0])
          and not bool(scene.wedge_clear()[0]) and not ok)

    # =========================== 11. lying cup: z band passes, uprightness rejects =========
    env.reset(seed=71)
    step(30)
    write_body(scene.cups[c.cup_names[0]], (0.0, -0.150, c.cup_r + 0.006), LYING, 120)
    report("lying")
    _x, _y, zl = cup_loc(c.cup_names[0])
    z_in_band = abs(zl - c.cup_rest_z) <= c.inside_z_tol
    s, ok = judge()
    check("lying cup inside: the z band PASSES (readback "
          f"z={zl:+.3f}, band +/-{c.inside_z_tol:.3f} about {c.cup_rest_z:+.3f}) yet "
          "uprightness rejects it -> not contained",
          z_in_band and not bool(scene.cup_upright(c.cup_names[0])[0])
          and not bool(scene.cup_inside(c.cup_names[0])[0]) and s <= 0.02 and not ok)

    # =========================== 12. z-band rejections: stacked + on the roof ==============
    write_body(scene.cups[c.cup_names[0]], (-0.010, -0.130, c.cup_rest_z + 0.010), UP, 60)
    write_body(scene.cups[c.cup_names[1]],
               (-0.010, -0.130, c.cup_rest_z + c.cup_h + 0.008), UP, 120)
    _x, _y, z_stk = cup_loc(c.cup_names[1])
    stacked_rej = (not bool(scene.cup_inside(c.cup_names[1])[0])
                   and z_stk > c.cup_rest_z + 0.04)
    write_body(scene.cups[c.cup_names[1]],
               (0.0, -0.150, c.inner_h + c.roof_t + c.cup_h / 2 + 0.004), UP, 90)
    _x, _y, z_roof = cup_loc(c.cup_names[1])
    report("z-band")
    s, ok = judge()
    check("z band: a cup STACKED on a contained cup reads "
          f"z={z_stk:+.3f} (floor rest {c.cup_rest_z:+.3f}) -> rejected; a cup on the "
          f"ROOF over the interior (z={z_roof:+.3f}) -> rejected",
          stacked_rej and not bool(scene.cup_inside(c.cup_names[1])[0])
          and z_roof > c.cup_rest_z + 0.10 and not ok)

    # =========================== 13. gate-hugging cup =======================================
    env.reset(seed=81)
    step(30)
    write_body(scene.cups[c.cup_names[0]], (0.0, -0.044, REST), UP, 90)
    report("gate-hug")
    _x, yh, _z = cup_loc(c.cup_names[0])
    s, ok = judge()
    check("gate-hugging cup: physically inside but pressed against the closed gate's "
          f"back face (y={yh:+.3f} > band edge {c.inside_y_hi}) -> rejected by the y "
          "band (containment demands CLEAR of the gate)",
          yh > c.inside_y_hi and bool(scene.cup_upright(c.cup_names[0])[0])
          and not bool(scene.cup_inside(c.cup_names[0])[0]) and s <= 0.02 and not ok)

    # =========================== 14. wedge left inside the cabinet ==========================
    env.reset(seed=91)
    step(30)
    # Wedge FIRST: writing the cups in while the wedge still sits at its faraway home
    # would make a construction prefix satisfy both_inside AND wedge_clear AND closed,
    # firing the +0.10 restore latch and inflating the "no restore credit" reading.
    write_body(scene.wedge, (+0.095, -0.150, 0.003), UP, 60)
    write_body(scene.cups[c.cup_names[0]], (-0.100, -0.130, c.cup_rest_z + 0.010), UP, 60)
    write_body(scene.cups[c.cup_names[1]], (-0.020, -0.130, c.cup_rest_z + 0.010), UP, 120)
    report("wedge-inside")
    s, ok = judge()
    check("wedge left inside: both cups in and the gate closed, but the wedge parked "
          f"ON the cabinet floor inside (y={wedge_y():+.3f}) -> wedge_clear False, "
          f"NOT success, score <= 0.36 ({s:.2f}: containment only, no restore credit)",
          bool(scene.both_inside()[0]) and bool(scene.gate_closed()[0])
          and not bool(scene.wedge_clear()[0]) and not ok and s <= 0.36)

    # =========================== 15. latched credit survives removal ========================
    s_before, _ = judge()
    write_body(scene.cups[c.cup_names[0]], (-0.30, 0.30, REST), UP, 90)
    report("removed")
    s_after, ok = judge()
    check("latched credit: removing a contained cup leaves the latched score "
          f"unchanged ({s_before:.2f} -> {s_after:.2f}) while both_inside() drops",
          abs(s_after - s_before) < 1e-3 and not bool(scene.both_inside()[0]) and not ok)

    # =========================== 16. settle gate ============================================
    env.reset(seed=101)
    step(30)
    write_body(scene.cups[c.cup_names[0]], (-0.010, -0.130, c.cup_rest_z + 0.010), UP, 60)
    # the final cup — judged while STILL FALLING inside (below the roof, above the floor)
    write_body(scene.cups[c.cup_names[1]], (+0.070, -0.130, 0.100), UP, 6)
    _x, _y, z_fall = cup_loc(c.cup_names[1])
    s, ok = judge()
    falling_reject = not ok
    step(8)
    falling_reject = falling_reject and not bool(scene.success()[0])
    report("settle-gate")
    # remove it BEFORE stillness can accrue (the battery must never reach success)
    write_body(scene.cups[c.cup_names[1]], (+0.30, 0.30, REST), UP, 60)
    check("settle gate: the full success geometry judged while the last cup is still "
          f"falling (z={z_fall:+.3f}, floor rest {c.cup_rest_z:+.3f}) is NOT success; "
          "the cup was removed before ring-down (battery never succeeds)",
          falling_reject and z_fall > c.cup_rest_z + c.inside_z_tol
          and not bool(scene.success()[0]))

    # =========================== 17-18. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.wedge_gate_cabinet")
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

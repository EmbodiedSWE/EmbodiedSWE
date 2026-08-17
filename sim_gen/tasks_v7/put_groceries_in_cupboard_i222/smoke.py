"""Smoke / rubric-REJECTION battery for GravityFeedRackScene (sim_gen task
`put_groceries_in_cupboard_i222`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — force-lift each can, stage it over the loading
port, and let gravity feed it down the ramp into the queue, in FIFO order red ->
green -> blue — is the acceptance evidence that the rubric ACCEPTS a correct
outcome; it is replicated here once as a positive control). Every other teleport is
instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled state and
asserts the rubric REJECTS it; an audit check asserts no rejection probe ever
reached success().

  1.  settle/no-NaN      — reset layout settles finite: cans standing at their floor
                           slots, still, none inside the rack, score ~0;
  2.  null policy        — 240 idle steps -> score ~0, no success (the queue does
                           not stock itself);
  3-4. randomization     — READBACK over 8 seeded resets: rack xy+yaw jitter is
                           real, can-to-slot assignment is a PERMUTATION and varies,
                           per-slot xy jitter is real;
  5.  seed strategy      — the end state the SEED's plan produces here (set the
                           grocery down on the cupboard from above): the red can
                           released over the rack just RESTS ON THE ROOF — never
                           entered, never seated, no success;
  6.  viewing-slot press — a REAL 6 N press (with a carry hold) drives the red can
                           into the 24 mm front slot: the probe demonstrably MOVES,
                           then the wall arrests it OUTSIDE — never entered;
  7.  wrong orientation  — the blue can dropped over the port lying ALONG the slope
                           (90 mm footprint vs 82.5 mm port) never becomes a queued
                           can: not seated at any slot, no credit, no success;
  8.  wrong order        — REAL port drops in the order green, red, blue: gravity
                           builds the queue green-red-blue — every can stored, but
                           NOT success and score <= 0.25 (order is load-bearing and
                           irreversible);
  9.  prefix only        — real drops red then green (correct prefix): latched
                           credit reaches the 0.60 cap exactly, NOT success;
  10. latched credit     — teleporting the red can back out afterwards leaves the
                           latched score unchanged, still no success;
  11. standing in port   — red+green queued (restored) and the blue can left
                           STANDING in the port shaft (stable there — asserted by
                           readback): everything stored but blue is not a queue
                           member -> NOT success;
  12. axis near-miss     — red can lying ALONG the slope mid-lane (inside, under the
                           roof): in-lane credit only, not seated (cross-lane axis
                           clause rejects), no success;
  13. positive control   — full FIFO drops red, green, blue -> success() True and
                           score 1.0 (the rubric accepts the mechanism's outcome);
                           then blue teleported out -> success drops, score falls
                           back to the latched 0.60 (success is judged LIVE);
  14. rejection audit    — success() was never True in ANY rejection probe (1-12);
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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.gravity_feed_rack")().build(num_envs=args.num_envs,
                                                       device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.30, -0.95, 0.80)) + o),
                                tuple(np.array((0.44, 0.00, 0.15)) + o),
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

    def report(tag: str) -> None:
        s, ok = judge()
        parts = []
        for nm, body, i in (("red", scene.red, 0), ("green", scene.green, 1),
                            ("blue", scene.blue, 2)):
            p = scene._rack_local(body)[0]
            parts.append(f"{nm}=({float(p[0]):+.3f},{float(p[1]):+.3f},"
                         f"{float(p[2]):.3f})s{i}={bool(scene._seated_slot(body, i)[0])}")
        print(f"[smoke] {tag:16s} | " + " ".join(parts)
              + f" | lift={bool(scene._lift_ever[0])} enter={bool(scene._enter_ever[0])}"
              f" q1={bool(scene._q1_ever[0])} q2={bool(scene._q2_ever[0])}"
              f" score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def rack_pose() -> tuple[torch.Tensor, torch.Tensor]:
        return scene.rack.data.root_pos_w[0], scene.rack.data.root_quat_w[0]

    def place_local(body, x: float, y: float, z: float, local_quat=None) -> None:
        """Teleport `body` to a rack-local pose (probe constructor: builds in-lane /
        on-roof / in-port relations directly, walls notwithstanding)."""
        rp, rq = rack_pose()
        off = quat_apply(rq.unsqueeze(0),
                         torch.tensor([x, y, z], device=device).unsqueeze(0))[0]
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = rp + off  # rack pos is world (env origin included)
        if local_quat is None:
            st[:, 3:7] = rq
        else:
            lq = torch.tensor(local_quat, device=device)
            st[:, 3:7] = quat_mul(rq.unsqueeze(0), lq.unsqueeze(0))[0]
        body.write_root_state_to_sim(st, all_ids)

    def place_world(body, x: float, y: float, z: float) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    Q_CROSS = (0.7071068, 0.7071068, 0.0, 0.0)  # axis z -> -y (cross-lane)
    Q_ALONG = (0.7071068, 0.0, 0.7071068, 0.0)  # axis z -> +x (along the slope)
    HOVER = ((c.ramp_x0 + c.roof_x0) / 2, 0.0, 0.215)

    def drop_port(body, local_quat=Q_CROSS, wait: int = 1100) -> bool:
        """The real feed, used as a probe constructor: stage over the port, hands off,
        wait for gravity + rolling to settle the can inside the rack."""
        place_local(body, *HOVER, local_quat=local_quat)
        streak = 0
        for _ in range(wait):
            env.step(no_action)
            ok = bool(scene._settled(body)[0]) and bool(scene._in_lane(body)[0])
            streak = streak + 1 if ok else 0
            if streak >= 40:
                return True
        return False

    def axis_world_z(body) -> float:
        ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
        return float(quat_apply(body.data.root_quat_w, ez)[0, 2])

    def seated_none(body) -> bool:
        return not any(bool(scene._seated_slot(body, j)[0]) for j in range(3))

    def finite_all() -> bool:
        return bool(torch.isfinite(scene.rack.data.root_state_w).all()
                    and torch.isfinite(scene.red.data.root_state_w).all()
                    and torch.isfinite(scene.green.data.root_state_w).all()
                    and torch.isfinite(scene.blue.data.root_state_w).all())

    # =========================== 1. settle / no-NaN =========================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("show")
    standing = all(
        abs(float((b.data.root_pos_w - scene.env_origins)[0, 2]) - c.can_l / 2) < 0.012
        and bool(scene._settled(b)[0]) for b in (scene.red, scene.green, scene.blue))
    s, ok = judge()
    check("settle: states finite, all three cans standing at their floor slots, "
          "still, none inside the rack, score ~0 (<= 0.02), no success",
          finite_all() and standing
          and not any(bool(scene._in_lane(b)[0]) for b in (scene.red, scene.green, scene.blue))
          and s <= 0.02 and not ok)

    # =========================== 2. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps (the rack does "
          "not stock itself)", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    slots = np.array(c.slots)
    reads, racks = [], []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        row = []
        for body in (scene.red, scene.green, scene.blue):
            p = (body.data.root_pos_w - scene.env_origins)[0]
            x, y = float(p[0]), float(p[1])
            d = np.linalg.norm(slots - np.array([x, y]), axis=1)
            row += [x, y, int(np.argmin(d))]
        reads.append(row)
        rp, rq = rack_pose()
        rpo = rp - scene.env_origins[0]
        yaw = math.degrees(2.0 * math.atan2(float(rq[3]), float(rq[0])))
        racks.append([float(rpo[0]), float(rpo[1]), yaw % 360.0])
    arr = np.array(reads)
    rk = np.array(racks)
    print(f"[smoke] can readback (x,y,slot per can):\n{arr}", flush=True)
    print(f"[smoke] rack readback (x,y,yaw_deg):\n{rk}", flush=True)
    assigns = arr[:, [2, 5, 8]].astype(int)
    check("randomization: can-to-slot assignment is a PERMUTATION every seed and the "
          "red can's slot varies across seeded resets (readback)",
          all(sorted(a.tolist()) == [0, 1, 2] for a in assigns)
          and len(set(assigns[:, 0].tolist())) >= 2)
    jit = 0.0
    for slot_id in (0, 1, 2):
        grp = arr[assigns[:, 0] == slot_id]
        if len(grp) >= 2:
            jit = max(jit, float((grp[:, 0:2].max(axis=0) - grp[:, 0:2].min(axis=0)).max()))
    rack_xy_spread = float((rk[:, 0:2].max(axis=0) - rk[:, 0:2].min(axis=0)).max())
    rack_yaw_spread = float(rk[:, 2].max() - rk[:, 2].min())
    yaw_in_band = all(abs(y - c.cab_yaw_deg) <= c.cab_yaw_jitter_deg + 0.5 for y in rk[:, 2])
    check("randomization: rack xy jitter (> 8 mm), rack yaw jitter (> 3 deg, within "
          "the +/-15 deg band) and per-slot can xy jitter (> 4 mm) are real (readback)",
          rack_xy_spread > 0.008 and rack_yaw_spread > 3.0 and yaw_in_band
          and jit > 0.004)

    # =========================== 5. seed strategy: set it down on the cupboard ==============
    # The seed's plan — lower the grocery onto the cupboard and let go — executed
    # here: the red can released over the rack. There is no shelf; it rests ON THE
    # ROOF (the walled roof tray) and never enters.
    torch.manual_seed(41)
    env.reset()
    step(30)
    place_local(scene.red, 0.06, 0.0, 0.225)
    step(240)
    report("seed-strategy")
    zloc = float(scene._rack_local(scene.red)[0, 2])
    s, ok = judge()
    check("seed strategy: red can set down on the cupboard from above RESTS ON THE "
          "ROOF (local z > 0.19) — never entered, never seated, lift latch only "
          "(score <= 0.101), no success",
          zloc > 0.19 and not bool(scene._enter_ever[0])
          and seated_none(scene.red) and s <= 0.101 and not ok)

    # =========================== 6. viewing-slot press (real force, must move) ==============
    # A hand presses the red can straight at the 24 mm viewing slot with 6 N while
    # carrying its weight. The probe must MOVE (approach the wall) and then be
    # ARRESTED by the slotted front wall — outside, never entered.
    torch.manual_seed(51)
    env.reset()
    step(30)
    # Clear the press corridor first: the middle floor slot (0.20, 0) sits exactly
    # on the approach line, and a teleported probe start must FIT (depenetration
    # would shove the probe sideways). Park the two non-probe cans far aside.
    place_world(scene.green, 1.2, 0.7, c.can_l / 2 + 0.002)
    place_world(scene.blue, 1.2, -0.7, c.can_l / 2 + 0.002)
    env.step(no_action)
    x_start = c.ramp_x1 + c.wall_t + c.can_r + 0.045
    z_press = (c.slot_z0 + c.slot_z1) / 2
    place_local(scene.red, x_start, 0.0, z_press, local_quat=Q_CROSS)
    arrest_x = c.ramp_x1 + c.wall_t + c.can_r - 0.008
    min_x = 1e9
    # Quasi-static velocity-servo push at the slot (v_des 0.12 m/s, stall press
    # ~2.4 N, clamp 6 N; K*dt/m = 0.67 < 1). min_x is sampled only while the can
    # is ENGAGED in the press corridor; the loop breaks the moment it pops out,
    # so the wrench can never launch an escaped can.
    for it in range(300):
        rl = scene._rack_local(scene.red)[0]
        if abs(float(rl[1])) > 0.08 or abs(float(rl[2]) - z_press) > 0.05 \
                or float(rl[0]) < arrest_x - 0.03:
            print(f"[smoke]   press break at it={it}: rl=({float(rl[0]):+.4f},"
                  f"{float(rl[1]):+.4f},{float(rl[2]):.4f})", flush=True)
            break  # escaped the corridor (or tunneled) — stop pushing
        min_x = min(min_x, float(rl[0]))
        _, rq = rack_pose()
        vel = scene.red.data.root_lin_vel_w[0]
        v_loc = quat_apply_inverse(rq.unsqueeze(0), vel.unsqueeze(0))[0]
        f_loc = torch.tensor([
            max(-6.0, min(6.0, 12.0 * (-0.10 - float(v_loc[0])))),
            max(-2.0, min(2.0, -10.0 * float(rl[1]) - 3.0 * float(v_loc[1]))),
            0.0], device=device)
        f_w = quat_apply(rq.unsqueeze(0), f_loc.unsqueeze(0))[0]
        pos = scene.red.data.root_pos_w[0]
        rp, _ = rack_pose()
        z_err = (float(rp[2]) + z_press) - float(pos[2])
        fz = c.can_mass * 9.81 + 15.0 * z_err - 4.0 * float(vel[2])
        f_w = f_w + torch.tensor([0.0, 0.0, max(0.0, min(6.0, fz))], device=device)
        # The wrench is applied in the BODY frame (the Q_CROSS 90-degree can makes
        # a world-frame command scramble y and z): rotate the desired world wrench
        # into the can's current body frame and pass is_global=False.
        cq = scene.red.data.root_quat_w[0]
        f_b = quat_apply_inverse(cq.unsqueeze(0), f_w.unsqueeze(0))[0]
        tau_w = (-0.05 * scene.red.data.root_ang_vel_w[0]).clamp(-0.10, 0.10)
        tau_b = quat_apply_inverse(cq.unsqueeze(0), tau_w.unsqueeze(0))[0]
        if it < 6 or it % 40 == 0:
            print(f"[smoke]   press it={it:3d} rl=({float(rl[0]):+.4f},"
                  f"{float(rl[1]):+.4f},{float(rl[2]):.4f}) "
                  f"v_loc=({float(v_loc[0]):+.3f},{float(v_loc[1]):+.3f},"
                  f"{float(v_loc[2]):+.3f}) f_loc=({float(f_loc[0]):+.2f},"
                  f"{float(f_loc[1]):+.2f}) fz={max(0.0, min(6.0, fz)):+.2f}",
                  flush=True)
        scene.red.set_external_force_and_torque(
            f_b.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            tau_b.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            env_ids=all_ids, is_global=False)
        env.step(no_action)
    scene.red.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
    env.step(no_action)
    report("slot-press")
    moved = x_start - min_x
    s, ok = judge()
    check("viewing-slot press: quasi-static carried press at the 24 mm slot MOVED "
          f"the can (advance {moved * 1000:.0f} mm > 15 mm), REACHED the wall "
          f"(min x {min_x:.3f} < {arrest_x + 0.020:.3f}) and was ARRESTED outside "
          f"(min x > wall face {arrest_x:.3f}) — never entered, no success",
          moved > 0.015 and arrest_x < min_x < arrest_x + 0.020
          and not bool(scene._enter_ever[0]) and not ok)

    # =========================== 7. wrong orientation at the port ===========================
    # The blue can dropped over the port lying ALONG the slope: its 90 mm footprint
    # exceeds the 82.5 mm port span, so gravity alone cannot make it a queued can.
    torch.manual_seed(61)
    env.reset()
    step(30)
    place_local(scene.blue, *HOVER, local_quat=Q_ALONG)
    step(420)
    report("wrong-axis-port")
    s, ok = judge()
    check("wrong orientation: blue can dropped over the port lying ALONG the slope "
          "(90 mm vs 82.5 mm port) never becomes a queued can — not seated at any "
          "slot, no red credit (score <= 0.02), no success",
          seated_none(scene.blue) and s <= 0.02 and not ok)

    # =========================== 8. wrong order (real drops) ================================
    # The mechanism run honestly, in the WRONG order: green, red, blue. Gravity
    # stores every can — the queue reads green-red-blue and can never be reordered.
    torch.manual_seed(71)
    env.reset()
    step(30)
    ok_g = drop_port(scene.green)
    ok_r = drop_port(scene.red)
    ok_b = drop_port(scene.blue)
    step(60)
    report("wrong-order")
    s8, ok = judge()
    check("wrong order: real port drops green,red,blue all stored (green at the "
          "window, red mid, blue rear — readback) but NOT success and score <= 0.25 "
          "(FIFO order is load-bearing and irreversible)",
          ok_g and ok_r and ok_b
          and bool(scene._seated_slot(scene.green, 0)[0])
          and bool(scene._seated_slot(scene.red, 1)[0])
          and bool(scene._seated_slot(scene.blue, 2)[0])
          and s8 <= 0.25 and not ok)

    # =========================== 9. prefix only (correct order, incomplete) =================
    torch.manual_seed(81)
    env.reset()
    step(30)
    ok_r = drop_port(scene.red)
    ok_g = drop_port(scene.green)
    step(60)
    report("prefix")
    s9, ok = judge()
    check("prefix only: real drops red then green — q1+q2 latched, score at the "
          "0.60 non-success cap exactly (0.599 <= s <= 0.601), NOT success (blue "
          "still on the floor)",
          ok_r and ok_g and bool(scene._q1_ever[0]) and bool(scene._q2_ever[0])
          and 0.599 <= s9 <= 0.601 and not ok)
    prefix_state = scene.get_state(all_ids)

    # =========================== 10. latched credit survives regression =====================
    place_world(scene.red, c.slots[0][0], c.slots[0][1], c.can_l / 2 + 0.002)
    step(80)
    report("regressed")
    s10, ok = judge()
    check("latched credit: teleporting the red can back OUT of the rack leaves the "
          f"latched score unchanged ({s9:.3f} -> {s10:.3f}), still no success",
          abs(s10 - s9) < 1e-3 and not bool(scene._in_lane(scene.red)[0]) and not ok)

    # =========================== 11. standing in the port shaft =============================
    # Red+green restored in the queue; the blue can left STANDING in the port shaft
    # (14 deg < the tipping angle and mu > tan(14) — genuinely stable, asserted by
    # readback). Everything is "in the rack", but blue is not a queue member.
    scene.set_state(prefix_state, all_ids)
    step(10)
    place_local(scene.blue, (c.ramp_x0 + c.roof_x0) / 2, 0.0,
                c.zsurf((c.ramp_x0 + c.roof_x0) / 2) + c.can_l / 2 / math.cos(
                    math.radians(c.incline_deg)) + 0.004)
    step(240)
    report("standing-port")
    up = axis_world_z(scene.blue)
    blue_x = float(scene._rack_local(scene.blue)[0, 0])
    s11, ok = judge()
    check("standing in port: blue can left standing in the port shaft is STABLE "
          f"(axis up {up:.2f} > 0.9, still, x {blue_x:+.3f} in the shaft) with "
          "red+green queued — not a queue member: NOT success, score <= 0.601",
          up > 0.9 and bool(scene._settled(scene.blue)[0])
          and c.ramp_x0 - 0.01 <= blue_x <= c.roof_x0 + 0.01
          and seated_none(scene.blue) and s11 <= 0.601 and not ok)

    # =========================== 12. axis near-miss inside the lane =========================
    # Red can lying ALONG the slope mid-lane (fits under the roof, cannot roll):
    # inside the rack but the cross-lane axis clause rejects it at every slot.
    torch.manual_seed(91)
    env.reset()
    step(30)
    place_local(scene.red, 0.05, 0.0, c.zsurf(0.05) + c.can_r + 0.004,
                local_quat=Q_ALONG)
    step(240)
    report("axis-near-miss")
    s12, ok = judge()
    check("axis near-miss: red can lying ALONG the slope mid-lane — in-lane credit "
          "only (score <= 0.11), not seated anywhere (cross-lane clause), no success",
          bool(scene._enter_ever[0]) and seated_none(scene.red)
          and s12 <= 0.11 and not ok)

    # =========================== 13. positive control + live success ========================
    ever_before_positive = ever_success[0]
    torch.manual_seed(101)
    env.reset()
    step(30)
    ok_r = drop_port(scene.red)
    ok_g = drop_port(scene.green)
    ok_b = drop_port(scene.blue)
    step(60)
    report("positive")
    s13a, ok13 = judge()
    place_world(scene.blue, c.slots[2][0], c.slots[2][1], c.can_l / 2 + 0.002)
    step(60)
    report("blue-removed")
    s13b, ok13b = judge()
    check("positive control: full FIFO drops red,green,blue -> success() True and "
          "score 1.0; teleporting blue out drops success and the score falls back "
          "to the latched 0.60 (success is judged LIVE)",
          ok_r and ok_g and ok_b and ok13 and s13a >= 0.999
          and not ok13b and 0.599 <= s13b <= 0.601)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True in any rejection probe (checks "
          "1-12); it first turned True only in the positive control",
          not ever_before_positive and ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.gravity_feed_rack")
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

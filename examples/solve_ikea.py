"""solve_four — ALL FOUR ikea legs seated, robot-only, deterministic layout (no jitter).

ARCHITECTURE (v4): no mid-air handoff (v1-v3's fragile link — the left close knocked legs out of
the right's line-contact pads). The RIGHT arm does every leg at ONE proven station; the LEFT arm
only rotates the slab between legs and braces it during threading.

Per leg (right arm, all skills proven in solve_leg/solve_pick or validated in four_v1-v3):
  pick the lying leg (convergence-gated side pinch) -> slow 90-deg reorient -> carry to P1 (a bench
  spot near the pick zone) -> PLANT the bottom on the bench -> ERECT about the planted anchor
  (30 deg/s wrist ratchet; four_v3: 44 deg droop -> 3-4 deg) -> release: the leg FREE-STANDS on the
  bench (flat-on-flat, tips only past ~8.4 deg) -> re-cage TOP-DOWN (the threading grip) -> carry
  to the STATION stud = near-south (-0.20,-0.25), the exact solve_leg-v4-proven threading pose ->
  spin-insertion place (perch-retry) -> straighten -> prewound pinch-wrench thread -> auto-weld.

Between legs: rotate the slab +90 deg so the next EMPTY stud arrives at the station. Handles =
welded standing legs (side cage low on the shaft) and empty studs (top cage); greedy choice of
(arm, handle) by in-reach arc headroom; rate-limited pure-pursuit tangential push.

Deterministic: torch.manual_seed(0), reset_pos_jitter=0, explicit leg_init_xy:
  leg0 (-0.05,-0.03)  leg1 (-0.13,+0.10)  leg2 (+0.01,+0.18)  leg3 (+0.13,+0.18)

Run from the repo root:
    .venv/bin/python experiments/ikea_bimanual_franka_fable/solve_four.py --headless
    .venv/bin/python experiments/ikea_bimanual_franka_fable/solve_four.py --livestream 2
"""

from __future__ import annotations

import argparse
import math
import os
import threading
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--max_sec", type=float, default=1800.0, help="total sim-time budget (s)")
parser.add_argument("--task_sec", type=float, default=300.0, help="per-task sim-time budget (s)")
parser.add_argument("--pinch_n", type=float, default=3.0)
parser.add_argument("--pinch_wind_n", type=float, default=25.0)
parser.add_argument("--lean", type=float, default=0.003)
parser.add_argument("--wind_rate", type=float, default=2.0)
parser.add_argument("--rewind_rate", type=float, default=3.0)
parser.add_argument("--sweep_deg", type=float, default=120.0)
parser.add_argument("--seek_deg", type=float, default=30.0)
parser.add_argument("--palm_gap", type=float, default=0.06)
parser.add_argument("--cage_slack", type=float, default=0.010)
parser.add_argument("--pad_local", type=float, default=0.093)
parser.add_argument("--grip_local", type=float, default=0.12)
parser.add_argument("--brace_lean", type=float, default=0.006)
parser.add_argument("--drag_z", type=float, default=1.16)
parser.add_argument("--drag_lead_deg", type=float, default=3.0, help="pursuit lead (deg); 6 stored too much stick-slip energy at 1.0kg")
parser.add_argument("--drag_rate", type=float, default=0.20)
parser.add_argument("--slab_mass", type=float, default=1.0, help="runtime slab mass override (kg). STOCK 1.0 is the tested config — the guarded kp350 pushes rotate it cleanly, and heavier slabs break the place compliance (2.5/1.5 wedge the bore-drop; 4.0 exceeds a single arm's push ceiling). No scene change needed.")
parser.add_argument("--restore", default="", help="snapshot .pt to restore (jump straight to the task after the one that saved it — Haoxiang's set_state debug accelerator)")
parser.add_argument("--snap_dir", default="", help="where to save per-task snapshots (default: this experiment's logs/)")
parser.add_argument("--prefer_right", action="store_true",
                    help="deliveries target the RIGHT disc only: rotate PAST the left-disc window "
                         "(+~50deg more CCW) so the fully proven right-arm cycle threads — no "
                         "left-thread/handover machinery on the critical path")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
livestream_on = args.livestream > 0

app = AppLauncher(args).app

import torch  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    axis_angle_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_from_matrix,
    quat_mul,
)

import robobench  # noqa: E402
from robobench.core import EnvCfg  # noqa: E402
from robobench.robots.franka import FrankaRobot, FrankaRobotCfg  # noqa: E402
from robobench.robots.multi import BimanualFrankaCfg  # noqa: E402
from robobench.suites.assembly.scenes import IkeaTableAssemblySceneCfg  # noqa: E402

OPEN = 0.04
GRIP_KP = 8000.0
SHAFT_TOP = 0.290
NUT_BOTTOM = -0.010
DT = 1.0 / 120.0
# Legs 1-3 lie in a NORTH row OUTSIDE the slab's corner-swept disc (r 0.389 + drift ~0.06 -> keep
# lying legs > 0.46 m from the slab centre): four_v7-v9's rotation stalled at +41 deg because the
# swept corner smashed into the old row (leg1 sat 0.34 m out). leg0 (picked before any rotation)
# may stay close. All grips remain in the right arm's proven 0.31-0.43 m pick band.
LEG_XY = ((-0.05, -0.03), (0.01, 0.18), (0.13, 0.18), (0.25, 0.18))
P1 = (-0.02, -0.26)        # bench plant/erect spot (near the pick zone; low poses in right's reach)
STATION = (-0.20, -0.25)   # the proven threading stud position (solve_leg v4)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    robobench.discover()
    FrankaRobot.TORQUE_CONTROL_DT = DT
    scene_cfg = IkeaTableAssemblySceneCfg(reset_pos_jitter=0.0, leg_init_xy=LEG_XY)
    robot_cfg = BimanualFrankaCfg(robots={
        "left": ("franka", FrankaRobotCfg(
            base_pos=(-0.95, 0.0, 0.994), nullspace_dof_pos=(), gripper_stiffness=GRIP_KP)),
        "right": ("franka", FrankaRobotCfg(
            base_pos=(0.25, -0.25, 0.994), base_rot=(0.0, 0.0, 0.0, 1.0),
            nullspace_dof_pos=(), gripper_stiffness=GRIP_KP)),
    })
    env = EnvCfg(
        scene="ikea_table", scene_cfg=scene_cfg, robot="bimanual_franka", robot_cfg=robot_cfg,
        control_mode="osc", env_spacing=3,
    ).build(num_envs=1, device=device)
    scene, robot = env.scene, env.robot
    arms = {n: robot[n] for n in ("left", "right")}
    oscs = {}
    for n, arm in arms.items():
        osc = arm.controller.controllers[0]
        osc._kp = torch.tensor([150.0, 150.0, 150.0, 600.0, 600.0, 600.0], device=device)
        osc._kd = 2.0 * osc._kp.sqrt()
        osc.cfg.rot_scale = 0.15
        oscs[n] = osc
    arts = {n: arm.articulation for n, arm in arms.items()}
    ee_idx = {n: arts[n].body_names.index("panda_hand") for n in arms}
    fj1 = {n: arts[n].find_joints(["panda_finger_joint1"])[0] for n in arms}
    slices = robot.action_slices
    render = (not args.headless) or livestream_on
    n_act = robot.action_dim
    CTRL_HZ = 1.0 / (env.dt * robot.control_period)
    c = scene.cfg
    slab_top = c.surface_z + c.table_thickness
    bench = c.surface_z
    PAD = torch.tensor([0.0, 0.0, args.pad_local], device=device)
    print(f"[cfg] dt=1/{round(1 / env.dt)} ctrl={CTRL_HZ:.0f}Hz seed=0 jitter=0 station={STATION} P1={P1}", flush=True)
    env.reset()
    if args.slab_mass > 0:  # experiment-only scene tweak (runtime): a realistic slab mass — the
        m = scene.table.root_physx_view.get_masses().clone()  # stock 1.0kg drifts/snaps under pushes
        scene.table.root_physx_view.set_masses(torch.full_like(m, args.slab_mass), torch.arange(m.shape[0]))
        print(f"[cfg] slab mass {float(m.flatten()[0]):.1f} -> {args.slab_mass:.1f} kg (runtime override)", flush=True)

    # ----- helpers ------------------------------------------------------------------------------
    def ee_pose(n):
        return arts[n].data.body_pos_w[0, ee_idx[n]], arts[n].data.body_quat_w[0, ee_idx[n]]

    def servo(n, goal_pos, goal_quat, grip, a):
        p, q = ee_pose(n)
        s = slices[n]
        a[0, s.start:s.start + 3] = ((goal_pos - p) / oscs[n].cfg.pos_scale).clamp(-1.0, 1.0)
        qe = quat_mul(goal_quat.unsqueeze(0), quat_conjugate(q.unsqueeze(0)))
        a[0, s.start + 3:s.start + 6] = (axis_angle_from_quat(qe)[0] / oscs[n].cfg.rot_scale).clamp(-1.0, 1.0)
        a[0, s.start + 6:s.start + 8] = grip

    def servo_pad(n, pad_target, goal_quat, grip, a):
        servo(n, pad_target - quat_apply(goal_quat.unsqueeze(0), PAD.unsqueeze(0))[0], goal_quat, grip, a)

    def table_pq():
        return scene.table.data.root_pos_w[0], scene.table.data.root_quat_w[0]

    def stud_world(slot):
        tp, tq = table_pq()
        s = torch.tensor([c.slots[slot][0], c.slots[slot][1], 0.0], device=device)
        return tp + quat_apply(tq.unsqueeze(0), s.unsqueeze(0))[0]

    def leg_pq(k):
        return scene.legs[k].data.root_pos_w[0], scene.legs[k].data.root_quat_w[0]

    def leg_state(k, slot):
        tp, tq = table_pq()
        off = quat_apply_inverse(tq.unsqueeze(0), (scene.legs[k].data.root_pos_w[0] - tp).unsqueeze(0))[0]
        lat = (off[:2] - torch.tensor(c.slots[slot], device=device)).norm()
        return off[2].item(), lat.item()

    def leg_tilt(k):
        ax = quat_apply(leg_pq(k)[1].unsqueeze(0), torch.tensor([[0.0, 0.0, 1.0]], device=device))[0]
        return math.degrees(math.acos(max(-1.0, min(1.0, ax[2].item())))), ax

    def yaw_of(q):
        ex = quat_apply(q.unsqueeze(0), torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
        return math.degrees(math.atan2(ex[1].item(), ex[0].item()))

    def yaw_about_z(qbase, ang):
        yz = quat_from_euler_xyz(*(torch.tensor([v], device=device) for v in (0.0, 0.0, ang)))
        return quat_mul(yz, qbase.unsqueeze(0))[0]

    def width(n):
        return 2.0 * arts[n].data.joint_pos[0, fj1[n]].item()

    V3 = lambda x, y, z: torch.tensor([float(x), float(y), float(z)], device=device)

    def radial_cage_quat(center_xy, handle_xy, topdown, arm=None, ret_err=False):
        """Cage orientation at a rotation handle. The jaw axis is 180deg-symmetric: pick the
        tangent branch whose yaw stays nearest the arm's home yaw — the fixed branch ran the wrist
        into its yaw limit ~+41deg into every drag (four_v7/v8's deterministic stall)."""
        r = handle_xy - center_xy
        r = r / r.norm()
        zh = torch.tensor([0.0, 0.0, -1.0], device=device) if topdown else torch.tensor([-r[0], -r[1], 0.0], device=device)
        best_q, best_err = None, 1e9
        for sgn in (1.0, -1.0):
            yh = sgn * torch.tensor([-r[1], r[0], 0.0], device=device)
            xh = torch.cross(yh, zh, dim=0)
            q = quat_from_matrix(torch.stack([xh, yh, zh], dim=1).unsqueeze(0))[0]
            if arm is None:
                return q
            err = abs((yaw_of(q) - yaw_of(down[arm]) + 180.0) % 360.0 - 180.0)
            if err < best_err:
                best_q, best_err = q, err
        return (best_q, best_err) if ret_err else best_q

    down = {}
    for n in arms:
        _, q = ee_pose(n)
        down[n] = quat_from_euler_xyz(*(torch.tensor([v], device=device) for v in (math.pi, 0.0, math.radians(yaw_of(q)))))[0]

    PARKS = {"left": V3(-0.60, 0.05, 1.35), "right": V3(0.10, -0.32, 1.30)}

    def park(n, a, grip=0.0):
        # altitude-first, ALWAYS: a straight-line park transit at z1.30 sweeps fingertips (~1.20)
        # through standing welded legs (tops 1.356) — v29's restore dragged the slab 0.5m this way
        high_travel(n, PARKS[n], down[n], grip, a)

    TRAVEL_Z = 1.52  # transit altitude: fingertips (~103mm below the hand) clear standing welded
                     # legs (tops ~1.346) with ~6cm margin — at 1.46 the 11mm margin was eaten by
                     # OSC z-sag during fast transits (v30's post-restore park grazed the welded
                     # leg and dragged the slab 177mm)

    def high_travel(n, tgt, q, grip, a):
        p, _ = ee_pose(n)
        xy_err = (p[:2] - tgt[:2]).norm().item()
        if xy_err > 0.03 and p[2].item() < TRAVEL_Z - 0.04:
            servo(n, V3(p[0], p[1], TRAVEL_Z), q, grip, a)
        elif xy_err > 0.03:
            servo(n, V3(tgt[0], tgt[1], TRAVEL_Z), q, grip, a)
        else:
            servo(n, tgt, q, grip, a)

    SEC = lambda s: max(1, round(s * CTRL_HZ))
    SWEEP = math.radians(args.sweep_deg)
    D_WIND, D_REWIND = args.wind_rate / CTRL_HZ, args.rewind_rate / CTRL_HZ
    W_LAND = SWEEP / 2.0
    PREWIND = math.radians(126.0)
    SEEK_CAP = PREWIND + math.radians(args.seek_deg)

    # ----- mission ------------------------------------------------------------------------------
    yaw_acc = {"slab_prev": yaw_of(table_pq()[1]), "slab": 0.0}

    def slab_yaw_step():
        y = yaw_of(table_pq()[1])
        yaw_acc["slab"] += (y - yaw_acc["slab_prev"] + 180.0) % 360.0 - 180.0
        yaw_acc["slab_prev"] = y

    results = {}
    assigned = {}

    def station_slot():
        """The best EMPTY stud for the right arm to thread: nearest to the centre of its threading
        disc (0.30-0.53 m from its base). The slab drifts under rotation drags — chasing a fixed
        station point is unnecessary since every downstream skill tracks live poses; what matters
        is reachability."""
        rb = torch.tensor([0.25, -0.25], device=device)
        used = set(assigned.values())
        cand = []
        for s in range(4):
            if s in used:
                continue
            d = float((stud_world(s)[:2] - rb).norm())
            cand.append((abs(d - 0.42), d, s))  # 0.42 = the sweet spot (v4 threaded at 0.45)
        cand.sort()
        return cand[0][2], cand[0][1]

    def stud_threadable(d):
        return 0.30 <= d <= 0.53

    def station_slot2():
        """HANDOVER VARIANT: best empty stud over BOTH arms' threading discs. At the -157 hole the
        right delivery needs -26deg (CW, 20 tasks stuck) but the LEFT disc is only +15deg CCW away
        — a direction never attempted when only the right arm could thread."""
        used = set(assigned.values())
        best = None
        arms_s = ("right",) if args.prefer_right else ("right", "left")
        for arm_s in arms_s:
            for s in range(4):
                if s in used:
                    continue
                d = float((stud_world(s)[:2] - BASES[arm_s]).norm())
                key = (abs(d - 0.42), d, s, arm_s)
                if best is None or key < best:
                    best = key
        return best[2], best[1], best[3]

    def delivery_delta(force_sgn=None):
        """Signed smallest rotation (deg) about the slab's centre that puts an EMPTY stud in the
        right arm's threading disc. Deliveries repeat every 90deg, so the nearest one is often
        CLOCKWISE — the direction the recenter pulls drift anyway (v57-v60 fought it uphill).
        force_sgn: return only that direction's nearest delivery (None if none exists)."""
        rb = torch.tensor([0.25, -0.25], device=device)
        ctr = table_pq()[0][:2]
        used = set(assigned.values())
        vs = [stud_world(k)[:2] - ctr for k in range(4) if k not in used]
        near = {-1.0: None, 1.0: None}
        for mag in range(0, 181, 2):
            for sgn in (-1.0, 1.0):
                if near[sgn] is not None:
                    continue
                th = math.radians(sgn * mag)
                c_, s_ = math.cos(th), math.sin(th)
                for v in vs:
                    p = ctr + torch.tensor([v[0] * c_ - v[1] * s_, v[0] * s_ + v[1] * c_], device=device)
                    if stud_threadable(float((p - rb).norm())) or \
                            (not args.prefer_right and stud_threadable(float((p - BASES["left"]).norm()))):
                        near[sgn] = sgn * max(mag, 8)
                        break
        if force_sgn is not None:
            return near[float(force_sgn)]
        # CW handicap is WELD-COUNT-DEPENDENT (B19): 45deg below 3 welds — the proven early-game
        # behavior (v57-v60 measured CW cycles gaining ~55deg/cycle; a flat 5deg handicap changed
        # task-1's direction from scratch and B18's new lineage knocked leg1 onto the slab) —
        # but only 5deg at 3 welds, where the 45 made the endgame re-choose the dead CW -26 over
        # the never-tried CCW +16 (and 10 still tied: the 2deg scan returns +16, 26 <= 16+10).
        hcap = 5 if int(scene.welded[0].sum()) >= 3 else 45
        if near[-1.0] is not None and (near[1.0] is None or abs(near[-1.0]) <= abs(near[1.0]) + hcap):
            return near[-1.0]
        return near[1.0] if near[1.0] is not None else 120.0

    plan = ["leg", "rot", "leg", "rot", "leg", "rot", "leg"]
    # MISSION-level approach bans: pose-infeasibility is geometry-bound, not task-bound —
    # per-task bans reset on every extension task and mint4 burned 4 tasks retrying the same
    # fold-infeasible pair. Cleared on reversal and whenever a drag actually moves the yaw.
    app_bl = set()
    plan_i = -1
    task = None
    legs_used = 0
    step_i = 0
    ids0 = torch.tensor([0], device=device)
    snap_dir = args.snap_dir or str(Path(__file__).resolve().parent / "logs")

    def save_snapshot():
        """After each task: the full restorable sim state + the mission bookkeeping."""
        payload = {
            "states": env.get_states(ids0),
            "mission": {"plan_i": plan_i, "assigned": dict(assigned), "legs_used": legs_used,
                        "slab_yaw": yaw_acc["slab"], "slab_prev": yaw_acc["slab_prev"],
                        "results": dict(results)},
        }
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
        p = f"{snap_dir}/snap_after_task{plan_i}.pt"
        torch.save(payload, p)
        print(f"[snap] saved {p}", flush=True)

    def restore_snapshot(path):
        nonlocal plan_i, legs_used
        payload = torch.load(path, map_location=device, weights_only=False)
        env.set_states(payload["states"], ids0)
        m = payload["mission"]
        plan_i = m["plan_i"]
        assigned.clear()
        assigned.update(m["assigned"])
        legs_used = m["legs_used"]
        yaw_acc["slab"], yaw_acc["slab_prev"] = m["slab_yaw"], m["slab_prev"]
        results.update(m["results"])
        print(f"[snap] restored {path}: resuming after task {plan_i} "
              f"(welds={scene.welded[0].int().tolist()}, slabYaw={yaw_acc['slab']:.0f})", flush=True)

    def next_task():
        nonlocal plan_i, task, legs_used
        if plan_i >= 0:
            save_snapshot()
        # ONE mission-level retry for a failed leg task: legs are consumable (4 legs, 4 slots),
        # so a single bimodal failure (verify run: ERECT_FAIL 11.8deg) must not forfeit a slot.
        # The pick re-acquires the leg from wherever it now lies/stands.
        if (task is not None and task["kind"] == "leg"
                and not bool(scene.welded[0, task["leg"]]) and not task.get("retried")):
            lp_r = leg_pq(task["leg"])[0]
            slot_d = float((stud_world(task["slot"])[:2] - BASES[task.get("arm", "right")]).norm())
            if lp_r[2].item() > 0.90 and stud_threadable(slot_d):  # still on the bench somewhere
                # (and the slot still deliverable — a knocked slab can carry it off-disc,
                # and retrying then repeats the cross-assembly carry that caused the knock)
                task = dict(task, retried=True, t0=step_i)
                print(f"[task {plan_i}] RETRY leg{task['leg']} (leg z {lp_r[2].item():.2f})", flush=True)
                return
        if task is not None and task["kind"] == "leg" and not bool(scene.welded[0, task["leg"]]):
            # the leg task FAILED for good: release its slot so the delivery scan can re-serve
            # it to a later attempt (a consumed-but-empty slot blocked re-delivery forever)
            assigned.pop(task["leg"], None)
        plan_i += 1
        # DYNAMIC PLAN: the fixed 7-task list ends the mission on any single leg failure
        # (verify6: 1/4 with 2 legs still on the bench). Keep alternating rot/leg while
        # unwelded on-bench legs remain, up to a hard task cap.
        while plan_i >= len(plan) and plan_i < 30:
            # fill UP TO the current index — a restored mission can resume past the base plan's
            # end and needs more than one appended slot (the single append fell off the end)
            if all(scene.welded[0].tolist()):
                break
            spare = any(not bool(scene.welded[0, kk]) and leg_pq(kk)[0][2].item() > 0.90
                        for kk in range(4))
            if not spare:
                break
            stn_d2 = station_slot2()[1]
            plan.append("leg" if stud_threadable(stn_d2) else "rot")
            print(f"[plan] extended with '{plan[-1]}' (task {len(plan) - 1})", flush=True)
        if plan_i >= len(plan):
            task = None
            return
        if plan[plan_i] == "leg" and not stud_threadable(station_slot2()[1]):
            # base-plan leg entries assumed the preceding rotation delivered; after a TIMEOUT
            # rotation the nearest slot can sit far off-disc, and the leg cycle then carries the
            # hanging leg ACROSS the welded assembly (record_3of4_scratch: the carried leg struck
            # the table and slid the slab off the bench). No stud in the disc -> rotate instead.
            print(f"[plan] task {plan_i}: no stud in the disc "
                  f"({station_slot2()[1] * 1e3:.0f}mm) — rotating instead of a leg cycle", flush=True)
            plan[plan_i] = "rot"
        if plan[plan_i] == "leg":
            # next never-assigned leg first; else any unwelded leg still on the bench
            # (extended-plan retries reuse dropped-but-recoverable legs)
            k = None
            for kk in range(4):
                if kk not in assigned and not bool(scene.welded[0, kk]):
                    k = kk
                    break
            if k is None:
                for kk in range(4):
                    if not bool(scene.welded[0, kk]) and leg_pq(kk)[0][2].item() > 0.90:
                        k = kk
                        break
            if k is None:  # nothing workable left
                task = None
                return
            legs_used += 1
            slot, d, s_arm = station_slot2()
            assigned[k] = slot
            task = {"kind": "leg", "leg": k, "slot": slot, "arm": s_arm, "t0": step_i}
            print(f"[task {plan_i}] leg{k} -> slot {slot} at {[round(v, 2) for v in stud_world(slot)[:2].tolist()]} "
                  f"({d * 1e3:.0f}mm from the {s_arm} base — {s_arm} threads)", flush=True)
        else:
            dd = delivery_delta()
            dirn = 1.0 if dd >= 0 else -1.0
            task = {"kind": "rot", "target": yaw_acc["slab"] + dd + dirn * 10.0, "dir": dirn, "t0": step_i}
            print(f"[task {plan_i}] rotate the slab until an empty stud reaches the station "
                  f"(from yaw {yaw_acc['slab']:.0f}, nearest delivery {dd:+.0f}deg -> going "
                  f"{'CCW' if dirn > 0 else 'CW'})", flush=True)

    st = {}
    phase, ph_t = "park", 0

    def enter(ph):
        nonlocal phase, ph_t
        phase, ph_t = ph, -1

    def fresh_state():
        for osc_ in oscs.values():  # restore thread-tuned gains if a push phase was interrupted
            osc_._kp[:3] = 150.0
            osc_._kd = 2.0 * osc_._kp.sqrt()
        st.clear()
        st.update({
            "p_flat_pick": None, "p_flat_top": None, "grip_off": None,
            "pick_q": None, "pick_pad": None, "reor_axis": None, "reor_q0": None, "reor_q1": None,
            "carry_from": None, "carry_to": None, "carry_T": 1, "ground": bench,
            "lower_z": None, "leg_hist": [], "place_try": 0, "pick_try": 0, "cage_try": 0,
            "t_state": "seek", "wound": 0.0, "t": 0, "dz_seek0": None, "stroke": 0,
            "ee0": None, "stall_t": 0, "w_start": 0.0, "stall_run": 0, "re_straighten": 0,
            "brace_z": None, "brace_hist": [], "brace_touch": None, "brace_on": False, "brace_xy": None,
            "ee_u": 0.0, "ee_prev": None,
        })

    def ee_unwrap(n="right"):  # threading-arm hand yaw (handover variant: either arm)
        y = yaw_of(ee_pose(n)[1])
        if st.get("ee_prev") is not None:
            st["ee_u"] += (y - st["ee_prev"] + 180.0) % 360.0 - 180.0
        st["ee_prev"] = y
        return st["ee_u"]

    def pinch_top(newtons):
        return st["p_flat_top"] - newtons / GRIP_KP

    def standing_legs_xy():
        return [leg_pq(k)[0][:2] for k in assigned if bool(scene.welded[0, int(k)])]

    def left_idle(a, barm="left"):
        """Brace arm (default left): park, or brace the slab while the other arm threads."""
        if not st.get("brace_on"):
            park(barm, a, 0.0)
            return
        if st["brace_xy"] is None:  # pick a brace point clear of standing legs + the station stud
            tp, _ = table_pq()
            stn = stud_world(task["slot"])[:2] if task and task["kind"] == "leg" else torch.tensor(STATION, device=device)
            best, score = None, -1.0
            rels = ((-0.17, 0.12), (-0.17, -0.12), (-0.25, 0.0), (-0.05, 0.18))
            if barm == "right":  # mirrored candidates: the east side is the right arm's reach
                rels = tuple((-x_r, y_r) for x_r, y_r in rels)
            for rel in rels:
                cd = tp[:2] + torch.tensor(rel, device=device)
                if float((cd - BASES[barm]).norm()) > 0.55:
                    continue
                d_leg = min([float((cd - s).norm()) for s in standing_legs_xy()], default=1.0)
                sc = min(d_leg, float((cd - stn).norm()))
                if sc > score:
                    best, score = cd, sc
            st["brace_xy"] = best if best is not None else tp[:2] + torch.tensor((-0.2, 0.0), device=device)
        bx = st["brace_xy"]
        if st["brace_touch"] is None:
            if st["brace_z"] is None:
                st["brace_z"] = ee_pose("left")[0][2].item()
            st["brace_z"] = max(st["brace_z"] - 0.02 / CTRL_HZ, slab_top + 0.095)
            lz = ee_pose("left")[0][2].item()
            st["brace_hist"].append(lz)
            st["brace_hist"] = st["brace_hist"][-SEC(0.4):]
            quiet = len(st["brace_hist"]) == SEC(0.4) and (max(st["brace_hist"]) - min(st["brace_hist"])) < 2e-4
            if (quiet and st["brace_z"] < lz - 0.003) or st["brace_z"] <= slab_top + 0.0951:
                st["brace_touch"] = lz
            servo(barm, V3(bx[0], bx[1], st["brace_z"]), down[barm], 0.0, a)
        else:
            servo(barm, V3(bx[0], bx[1], st["brace_touch"] - args.brace_lean), down[barm], 0.0, a)

    def center_pin(parm, a):
        """CENTRE PIN: one arm's fingertips press the slab's centre while the OTHER drags a handle.
        A centre press resists translation (added friction, zero lever for rotation) — the robot's
        own fixture pin. This is what finally stops the drag drift (v10-v41's compounding failure;
        v76's UNPINNED left drag flung the slab 645mm — every drag gets a pin now)."""
        ctr = table_pq()[0][:2]
        if st.get("pin_arm") != parm:
            # the pin ARM changed (drag arms swapped): the stored touch height belongs to the
            # OTHER arm — v78's left pin slammed to the right arm's stale height and shoved the
            # slab 528mm. Restart the gentle descent from scratch.
            st.pop("pin_touch", None)
            st.pop("pin_z", None)
            st.pop("pin_hist", None)
        st["pin_arm"] = parm
        if st.get("pin_touch") is None:
            if st.get("pin_z") is None:
                st["pin_z"] = ee_pose(parm)[0][2].item()
            st["pin_z"] = max(st["pin_z"] - 0.02 / CTRL_HZ, slab_top + 0.095)
            lz = ee_pose(parm)[0][2].item()
            st.setdefault("pin_hist", []).append(lz)
            st["pin_hist"] = st["pin_hist"][-SEC(0.4):]
            quiet = len(st["pin_hist"]) == SEC(0.4) and (max(st["pin_hist"]) - min(st["pin_hist"])) < 2e-4
            if (quiet and st["pin_z"] < lz - 0.003) or st["pin_z"] <= slab_top + 0.0951:
                st["pin_touch"] = lz
                print(f"[rotate] centre PIN down ({parm})", flush=True)
            servo(parm, V3(ctr[0], ctr[1], st["pin_z"]), down[parm], 0.0, a)
        else:  # LIGHT pin (2mm lean ~3-4N): 6mm doubled the slab friction and stalled the drags (v42)
            servo(parm, V3(ctr[0], ctr[1], st["pin_touch"] - 0.002), down[parm], 0.0, a)

    def left_pin(a):
        center_pin("left", a)

    def anchor_hold(a):
        """ANCHORED rotation (Haoxiang): the off-arm holds a LOOSE cage around one welded leg —
        a geometric translation lock (the shaft cannot leave the jaws; ±6mm slack) — while the
        drag arm arcs another welded leg about it. Unlike the friction pin (~3N, slips 3mm/deg),
        this cannot be dragged: rotation about the anchor is translation-free by construction."""
        parm, nm = st["a_arm"], st["a_name"]
        kA = int(nm[3])
        lpA, lqA = leg_pq(kA)
        axA = quat_apply(lqA.unsqueeze(0), V3(0, 0, 1).unsqueeze(0))[0]
        topA = lpA + SHAFT_TOP * axA
        if st.get("anc_state") != "hold":
            # B5: yaw-DIVERSE approach (the recage rescue): the cage is yaw-free around a
            # vertical shaft, but each wrist yaw is a different arm fold — right/leg2's 0-yaw
            # fold never landed in 20s across every B-run while other folds were never tried.
            yaws_a = (0.0, 45.0, -45.0, 90.0) if int(scene.welded[0].sum()) >= 3 else (0.0,)
            i_ya = min(int(st.get("anc_t", 0) / SEC(6.0)), len(yaws_a) - 1)
            st["anc_t"] = st.get("anc_t", 0) + 1
            q_anc = yaw_about_z(down[parm], math.radians(yaws_a[i_ya]))
            goal = V3(topA[0].item(), topA[1].item(), topA[2].item() + args.palm_gap)
            high_travel(parm, goal, q_anc, 0.058, a)
            if float((ee_pose(parm)[0] - goal).norm()) < 0.02:
                st["anc_state"] = "hold"
                st["anc_goal"] = goal  # FIXED from here on: the hold is the lock
                st["anc_q"] = q_anc
                print(f"[rotate] anchor cage set ({parm}/{nm}, yaw {yaws_a[i_ya]:.0f})", flush=True)
        else:
            servo(parm, st["anc_goal"], st.get("anc_q", down[parm]), 0.052, a)

    def assist_push(oarm, a, dh_v, drag_hxy):
        """RC-COUPLE (B6, Haoxiang's two-arm idea): during a 3-weld recenter pull the off-arm
        PUSHES a welded shaft toward the same target instead of parking — each mechanism alone
        sits just under the 3-weld breakaway (~12N: the pulls gained 0mm across B1-B5, the
        shaft-pushes all ended NO-PROGRESS); pulling and pushing the same translation together
        is the first move that sums them. Fingertip press at the e_shaft push height."""
        best_as = None
        for kk_a in assigned:
            if not bool(scene.welded[0, int(kk_a)]):
                continue
            w_l = leg_pq(int(kk_a))[0][:2]
            d_a = float((w_l - BASES[oarm]).norm())
            # [0.15, 0.68]: a fingertip PRESS tolerates far more than a cage — B3/B5's left
            # anchor cage landed at leg1's 0.664, and the press pose is easier than the cage.
            # (The original [0.26,0.62] left the LEFT arm with no shaft at all, so the only
            # coupled pull — right/leg2 — ran solo and gained 0mm.)
            if not (0.15 <= d_a <= 0.68):
                continue
            if float((w_l - drag_hxy).norm()) < 0.20:  # never crowd the drag hand
                continue
            if best_as is None or abs(d_a - 0.45) < best_as[0]:  # prefer mid-range poses
                best_as = (abs(d_a - 0.45), kk_a)
        if best_as is None:
            park(oarm, a, 0.0)
            return
        kk_a = int(best_as[1])
        w_a = leg_pq(kk_a)[0][:2]  # LIVE shaft xy
        key = (oarm, kk_a)
        if st.get("as_key") != key:
            st["as_key"], st["as_t"] = key, 0
            st.pop("as_ph", None)
            print(f"[rotate] RC-COUPLE: {oarm} assist-pushes leg{kk_a} toward the pull target", flush=True)
        st["as_t"] += 1
        AS_Z = 1.21  # e_shaft height: fingertips (~-0.10) land ON the shaft, above the nut
        app_a = V3(w_a[0] - dh_v[0] * 0.06, w_a[1] - dh_v[1] * 0.06, AS_Z)
        if st.get("as_ph") != "press":
            # approach from behind the shaft (opposite the push direction), CONVERGE-gated —
            # the fixed 3s cutover pressed from wherever the hand happened to be
            high_travel(oarm, app_a, down[oarm], 0.0, a)
            if float((ee_pose(oarm)[0] - app_a).norm()) < 0.04 or st["as_t"] >= SEC(6.0):
                st["as_ph"] = "press"
        else:  # press THROUGH the shaft along the pull direction (3cm lean = force at the kp)
            tgt_a = V3(w_a[0] + dh_v[0] * 0.03, w_a[1] + dh_v[1] * 0.03, AS_Z)
            servo(oarm, tgt_a, down[oarm], 0.0, a)

    def find_ho_spot(slot_h):
        """Both-arms HANDOVER spot: inside both carry envelopes, off the slab (frame-aware
        edge margin), clear of welded legs and of the target stud. Returns (x, y) or None."""
        tp_h, tq_h = table_pq()
        best_h, best_s = None, None
        for xi_h in range(-70, -5, 2):
            for yi_h in range(-30, 49, 2):
                cnd_h = (xi_h / 100.0, yi_h / 100.0)
                c_h = torch.tensor(cnd_h, device=device)
                d_r = float((c_h - BASES["right"]).norm())
                d_l = float((c_h - BASES["left"]).norm())
                if d_r > 0.70 or d_l > 0.68:
                    continue
                if abs(cnd_h[1]) > 0.32:
                    continue  # bench-edge margin: B11's relay drift near y -0.46 rolled the
                    # toppled leg clean off the bench
                loc_h = quat_apply_inverse(tq_h.unsqueeze(0),
                    (V3(cnd_h[0], cnd_h[1], tp_h[2].item()) - tp_h).unsqueeze(0))[0]
                if max(abs(loc_h[0].item()), abs(loc_h[1].item())) < 0.275 + 0.045:
                    continue  # on (or hugging) the slab
                if any(float((c_h - leg_pq(int(kk_h))[0][:2]).norm()) < 0.15
                       for kk_h in range(4) if bool(scene.welded[0, kk_h])):
                    continue
                if float((c_h - stud_world(slot_h)[:2]).norm()) < 0.07:
                    continue
                # score = RIGHT extension (B12: at dR 0.69 the wrist sags 12-13deg and the firm
                # pinch tilts the leg with it — every relay plant failed the gate). The right
                # arm does the delicate plant; the left only needs its loose top cage, which
                # proved itself at 0.664 (the anchor hold), so cap dL at 0.66 and MINIMIZE dR.
                if d_l > 0.66:
                    continue
                s_h = d_r
                if best_s is None or s_h < best_s:
                    best_h, best_s = cnd_h, s_h
        return best_h

    # rotation-drag helpers
    CORNERS = [(0.275, 0.275), (-0.275, 0.275), (-0.275, -0.275), (0.275, -0.275)]

    def handles():
        # TOP-DOWN CAGE handles only — the smooth, never-explosive drag mechanism. Welded-leg
        # shafts + the slab's own CORNERS (a cage straddles the 90-deg corner like a shaft; four of
        # them = one arrives in an arm's zone every 90 deg). Studs pop out (v6-v8); rigid edge
        # pushes stick-slip-explode the 1.0kg slab (v31-v35).
        out = []
        welded_xy = []
        for k in assigned:
            if bool(scene.welded[0, int(k)]):
                wxy = leg_pq(int(k))[0][:2]
                welded_xy.append(wxy)
                out.append((f"leg{k}", wxy, True))
        tp, tq = table_pq()
        for kc, cl in enumerate(CORNERS):
            v = torch.tensor([cl[0], cl[1], 0.0], device=device)
            cxy_w = (tp + quat_apply(tq.unsqueeze(0), v.unsqueeze(0))[0])[:2]
            if min([float((cxy_w - w_).norm()) for w_ in welded_xy], default=1.0) < 0.12:
                continue  # a welded leg stands 35mm from this corner: the palm would hit it
            out.append((f"cor{kc}", cxy_w, True))
        return out

    BASES = {"left": torch.tensor([-0.95, 0.0], device=device), "right": torch.tensor([0.25, -0.25], device=device)}
    # The rotation is commanded about the FIXED spawn centre, not the live one: circling the live
    # centre followed the drag-induced translation instead of correcting it (four_v10: the slab
    # drifted 25 cm to (-0.33,+0.21) and jammed at the bench edge — THE +41 deg wall).
    C0 = torch.tensor([-0.45, 0.0], device=device)
    R0 = 0.354  # stud/leg handle radius in the slab frame

    def drag_headroom(arm, hxy, dirn=1.0, center=None):
        # sweep about the LIVE centre (or an explicit rotation center, e.g. an anchored leg):
        # the old C0-centred sweep mis-modelled every handle's arc by the slab's offset — at the
        # 3/4 corner (26cm off) all headrooms read ~0 and the mission stalled on zero-net pulls
        ctr_h = center if center is not None else table_pq()[0][:2]
        th0 = math.atan2(float(hxy[1] - ctr_h[1]), float(hxy[0] - ctr_h[0]))
        r_h = float((hxy - ctr_h).norm())  # handle radius (legs 0.354, corners 0.389)
        b = BASES[arm]

        def ok(th):
            p = ctr_h + r_h * torch.tensor([math.cos(th), math.sin(th)], device=device)
            d = float((p - b).norm())
            # ceiling 0.66 (was 0.62): at 3 welds the left arm's only CW handles sit at
            # 0.63-0.65 — its pulls couple at 0-4mm pad error there, so drags can work too
            return 0.28 <= d <= 0.66

        if not ok(th0):
            return 0.0
        dth = 0.0
        while dth < math.radians(70) and ok(th0 + dirn * (dth + math.radians(2))):
            dth += math.radians(2)
        return math.degrees(dth)

    def pick_drag(blacklist, dirn=1.0):
        best, best_score = None, -1e9
        tpc_l = table_pq()[0][:2]
        off_l = C0 - tpc_l
        for arm in ("right", "left"):  # right drags pinned; LEFT drags (unpinned) when the right
            # arm's cage pose is unreachable — v71's wall: at yaw -45 the right arm hovers 550mm
            # from leg2 (ori 156deg) while the left couples at 0-4mm
            for name, hxy, topdown in handles():
                if (arm, name) in blacklist:
                    continue
                if (arm, name) in app_bl:
                    # approach-infeasible pairs persist for the whole task (bearing-dependent
                    # pose limit, NOT position-dependent): v79 cleared them on slab movement,
                    # retried the flailing approach, and the arm threw the table 742mm
                    continue
                hr = drag_headroom(arm, hxy, dirn)
                if hr < 12.0:
                    continue
                # (v82 tried a qerr>70 pre-filter here: it also rejected the PROVEN left/leg0
                # -94 sweep — branch yaw error is not a reliable infeasibility signal. The
                # persistent app_bl + hover/d_in aborts contain the flail class instead.)
                score = hr + (100.0 if name.startswith("leg") else 0.0) \
                    + (15.0 if arm == "right" else 0.0)  # pinned right drags preferred
                if float(off_l.norm()) > 0.06:
                    # RESTORING bias (Haoxiang: the fall was an edge-margin problem) — prefer
                    # handles whose tangential push also nudges the slab back toward the spawn spot
                    rv_l = hxy - tpc_l
                    t_l = torch.tensor([-rv_l[1], rv_l[0]], device=device)
                    t_l = t_l / t_l.norm()
                    score += 40.0 * float((t_l * (off_l / off_l.norm())).sum())
                if score > best_score:
                    best, best_score = (hr, arm, name, topdown), score
        return best

    # --- EDGE-PUSH rotation (v14): closed fingertips shove the slab's SIDE FACE tangentially.
    # Cage-drags topped out (+43/cycle: the one leg handle exits the zone; stud cages pop off).
    # The edge push has flat-face contact at z~1.02 (no pop-off, no tipping), palm-down orientation
    # (always reachable), and free choice of contact point along the boundary.
    EDGE_LOCAL = [(0.275, y) for y in (-0.18, -0.09, 0.0, 0.09, 0.18)] + \
                 [(-0.275, y) for y in (-0.18, -0.09, 0.0, 0.09, 0.18)] + \
                 [(x, 0.275) for x in (-0.18, -0.09, 0.0, 0.09, 0.18)] + \
                 [(x, -0.275) for x in (-0.18, -0.09, 0.0, 0.09, 0.18)]

    def edge_world(pt_local):
        tp, tq = table_pq()
        v = torch.tensor([pt_local[0], pt_local[1], 0.0], device=device)
        return (tp + quat_apply(tq.unsqueeze(0), v.unsqueeze(0))[0])[:2]

    def clear_of_uprights(wxy, margin=0.13):
        for k in assigned:
            if bool(scene.welded[0, int(k)]) and float((wxy - leg_pq(int(k))[0][:2]).norm()) < margin:
                return False
        used = set(assigned.values())
        for s in range(4):
            if s not in used and float((wxy - stud_world(s)[:2]).norm()) < margin:
                return False
        return True

    def pick_push(blacklist):
        """Greedy (arm, local edge point) with the most CCW headroom: how far the point can travel
        (rotating about the live centre) while staying in the arm's [0.30, 0.55] disc."""
        tp, _ = table_pq()
        cxy_l = tp[:2]
        best, best_hr = None, 0.0
        for arm in ("left", "right"):
            b = BASES[arm]
            for i, pl_ in enumerate(EDGE_LOCAL):
                if (arm, i) in blacklist:
                    continue
                w = edge_world(pl_)
                if not (0.30 <= float((w - b).norm()) <= 0.55) or not clear_of_uprights(w):
                    continue
                rv = w - cxy_l
                r = float(rv.norm())
                th0 = math.atan2(float(rv[1]), float(rv[0]))
                dth = 0.0
                while dth < math.radians(60):
                    th = th0 + dth + math.radians(3)
                    p = cxy_l + r * torch.tensor([math.cos(th), math.sin(th)], device=device)
                    if not (0.30 <= float((p - b).norm()) <= 0.55) or not clear_of_uprights(p):
                        break
                    dth += math.radians(3)
                if math.degrees(dth) > best_hr:
                    best, best_hr = (arm, i), math.degrees(dth)
        return (best[0], best[1], best_hr) if best and best_hr >= 10.0 else None

    def opposite_pt(i):
        pi = EDGE_LOCAL[i]
        best, bd = None, 1e9
        for j, pj in enumerate(EDGE_LOCAL):
            d = abs(pj[0] + pi[0]) + abs(pj[1] + pi[1])
            if d < bd:
                best, bd = j, d
        return best

    def pt_headroom(arm, i, cxy_l):
        """CCW deg the LOCAL point i can travel about cxy_l while staying in `arm`'s disc + clear."""
        b = BASES[arm]
        w = edge_world(EDGE_LOCAL[i])
        if not (0.30 <= float((w - b).norm()) <= 0.55) or not clear_of_uprights(w):
            return 0.0
        rv = w - cxy_l
        r = float(rv.norm())
        th0 = math.atan2(float(rv[1]), float(rv[0]))
        dth = 0.0
        while dth < math.radians(60):
            th = th0 + dth + math.radians(3)
            p = cxy_l + r * torch.tensor([math.cos(th), math.sin(th)], device=device)
            if not (0.30 <= float((p - b).norm()) <= 0.55) or not clear_of_uprights(p):
                break
            dth += math.radians(3)
        return math.degrees(dth)

    def pick_couple(blacklist):
        """TWO-ARM push: right on point i, left on point j, both CCW tangent — torque adds, net
        force partially cancels (strictly-opposite pairs are unreachable with these bases: the west
        edge is inside the left base's dead zone). Score = min headroom − a net-force penalty.
        Returns (i, j, headroom) or None."""
        tp, _ = table_pq()
        cxy_l = tp[:2]

        def t_hat_of(i):
            w = edge_world(EDGE_LOCAL[i])
            rv = w - cxy_l
            t = torch.tensor([-rv[1], rv[0]], device=device)
            return t / t.norm()

        best, best_score, best_hr = None, -1e9, 0.0
        for i in range(len(EDGE_LOCAL)):
            hri = pt_headroom("right", i, cxy_l)
            if hri < 10.0:
                continue
            for j in range(len(EDGE_LOCAL)):
                if i == j or ("pair", i, j) in blacklist:
                    continue
                hrj = pt_headroom("left", j, cxy_l)
                if hrj < 10.0:
                    continue
                hr = min(hri, hrj)
                net = float((t_hat_of(i) + t_hat_of(j)).norm())  # 0 = perfect couple, 2 = parallel
                score = hr - 25.0 * net
                if score > best_score:
                    best, best_score, best_hr = (i, j), score, hr
        return (best[0], best[1], best_hr) if best else None

    def handle_xy(name):
        if name.startswith("leg"):
            return leg_pq(int(name[3]))[0][:2]
        if name.startswith("cor"):
            tp, tq = table_pq()
            cl = CORNERS[int(name[3])]
            v = torch.tensor([cl[0], cl[1], 0.0], device=device)
            return (tp + quat_apply(tq.unsqueeze(0), v.unsqueeze(0))[0])[:2]
        return stud_world(int(name[4]))[:2]

    # ----- main loop ------------------------------------------------------------------------------
    if args.restore:
        restore_snapshot(args.restore)
        for _ in range(SEC(1.0)):  # settle: hold in place so the restored state relaxes cleanly
            a0 = torch.zeros(1, n_act, device=device)
            for n_ in arms:
                p_, q_ = ee_pose(n_)
                servo(n_, p_, q_, 0.0, a0)
            env.step(a0, render=render)
    next_task()
    fresh_state()
    enter("park")
    LOG_EVERY = SEC(1.0)

    for step_i in range(1, SEC(args.max_sec) + 1):
        a = torch.zeros(1, n_act, device=device)
        ph_t += 1
        slab_yaw_step()
        if task is None:
            break
        if step_i - task["t0"] > SEC(args.task_sec):
            print(f"[task {plan_i}] TIMEOUT — skipping", flush=True)
            results[plan_i] = results.get(plan_i, "TIMEOUT")
            next_task()
            fresh_state()
            enter("park")
            continue

        # ======================= LEG CYCLE (right arm only) ======================================
        if task["kind"] == "leg":
            K, SLOT = task["leg"], task["slot"]
            # HANDOVER VARIANT: TARM = the arm that recages/carries/threads (station arm).
            # The pick half (park..stand_release) is always RIGHT (the lying legs are in its
            # zone and the pick/erect skills are 5/5-proven there); for TARM=="left" the plant
            # spot becomes a both-arms-reachable HANDOVER spot and the acting arm switches to
            # left when the cage half begins (recage entry).
            TARM = task.get("arm", "right")
            ACT = st.get("act_arm", "right")
            OTH = "left" if ACT == "right" else "right"
            dz, lat = leg_state(K, SLOT)
            lp, lq = leg_pq(K)
            pr, qr = ee_pose(ACT)
            ee_u = ee_unwrap(ACT)
            welded = bool(scene.welded[0, K])

            if phase == "park":
                park(ACT, a, OPEN)
                park(OTH, a, 0.0)
                if ph_t >= SEC(1.0):
                    tilt0, _ = leg_tilt(K)
                    if tilt0 < 10.0 and not welded:
                        # the leg is ALREADY STANDING (a retry after a failed recage): the pick
                        # is for LYING legs — re-picking a standing leg knocked it off the bench
                        # (verify4: LEG_FELL top z 0.02). Cage it where it stands.
                        print(f"[task {plan_i}] leg{K} already standing — recage directly", flush=True)
                        st["cage_try"] = 0
                        d_t3 = float((lp[:2] - BASES[TARM]).norm())
                        if TARM == "left" and d_t3 > 0.65:
                            d_r3 = float((lp[:2] - BASES["right"]).norm())
                            if 0.24 <= d_r3 <= 0.66:
                                # RELAY: the leg stands outside the thread arm's reach (plant sag
                                # at the right arm's reach edge, or a knock) — RIGHT re-cages it
                                # and ferries it to the handover spot before the left takes over
                                st["act_arm"], st["ee_prev"] = "right", None
                                st["relay"] = True
                                print(f"[task {plan_i}] leg{K} out of left reach ({d_t3 * 1e3:.0f}mm) "
                                      f"— RIGHT relays it to the handover spot", flush=True)
                            else:
                                print(f"[task {plan_i}] leg{K} unreachable by BOTH arms "
                                      f"(L {d_t3 * 1e3:.0f} R {d_r3 * 1e3:.0f}mm)", flush=True)
                                results[plan_i] = "LEG_UNREACHABLE"
                                enter("abandon")
                                continue
                        else:
                            st["act_arm"], st["ee_prev"] = TARM, None
                            st.pop("relay", None)
                        enter("recage")
                    else:
                        enter("pre_pick")
            elif phase in ("pre_pick", "descend_pick", "close_pick"):
                axis = quat_apply(lq.unsqueeze(0), V3(0, 0, 1).unsqueeze(0))[0]
                alpha = math.atan2(axis[1].item(), axis[0].item())
                home_yaw = math.radians(yaw_of(down[ACT]))
                psi = alpha
                while psi - home_yaw > math.pi / 2:
                    psi -= math.pi
                while psi - home_yaw < -math.pi / 2:
                    psi += math.pi
                if st.get("psi_flip"):
                    # 180deg-symmetric jaw: the folded psi can sit AT the wrist-fold boundary,
                    # where OSC convergence is bimodal across runs (v85: 4.6mm, v86: 817mm from
                    # the same snapshot) — retries alternate the branch
                    psi += math.pi
                q_pick = quat_from_euler_xyz(*(torch.tensor([v], device=device) for v in (math.pi, 0.0, psi)))[0]
                if st.get("g_pick") is None:
                    # FIRST attempt = the PROVEN grip_local (verify run 3: choosing by band even
                    # on attempt 1 changed the reorient lever and broke the 5/5 task-0 erect).
                    # RETRIES walk band-sorted alternates along the shaft — that's what rescued
                    # leg3 (its spawn settled +10cm east, grip_local out of the 0.31-0.43 band).
                    rb_p = torch.tensor([0.25, -0.25], device=device)
                    cands_g = []
                    for g_ in (args.grip_local - 0.05, args.grip_local + 0.05,
                               args.grip_local - 0.10, args.grip_local + 0.10):
                        g_ = min(max(g_, 0.05), 0.27)  # stay ON the shaft (off-shaft spots
                        # closed on the NUT / air in verify run 2)
                        p_ = lp + quat_apply(lq.unsqueeze(0), V3(0, 0, g_).unsqueeze(0))[0]
                        cands_g.append((abs(float((p_[:2] - rb_p).norm()) - 0.40), g_))
                    cands_g.sort()
                    st["g_cands"] = [args.grip_local] + [g_ for _, g_ in cands_g]
                    st["g_pick"] = args.grip_local
                gp = lp + quat_apply(lq.unsqueeze(0), V3(0, 0, st["g_pick"]).unsqueeze(0))[0]
                pad_now = pr + quat_apply(qr.unsqueeze(0), PAD.unsqueeze(0))[0]
                yaw_err = abs((yaw_of(qr) - math.degrees(psi) + 180.0) % 360.0 - 180.0)
                park(OTH, a, 0.0)
                if phase == "pre_pick":
                    hand_goal = (gp + V3(0, 0, 0.12)) - quat_apply(q_pick.unsqueeze(0), PAD.unsqueeze(0))[0]
                    high_travel(ACT, hand_goal, q_pick, OPEN, a)
                    pad_err = (pad_now - (gp + V3(0, 0, 0.12))).norm().item()
                    if (ph_t >= SEC(2.0) and pad_err < 0.005 and yaw_err < 4.0) or ph_t >= SEC(8.0):
                        enter("descend_pick")
                elif phase == "descend_pick":
                    servo_pad(ACT, gp, q_pick, OPEN, a)
                    pad_err = (pad_now - gp).norm().item()
                    if (ph_t >= SEC(1.2) and pad_err < 0.004 and yaw_err < 4.0) or ph_t >= SEC(5.0):
                        print(f"[pick{K}] at grip pose err {pad_err * 1e3:.1f}mm/{yaw_err:.1f}deg", flush=True)
                        enter("close_pick")
                else:
                    servo_pad(ACT, gp, q_pick, 0.0, a)
                    if ph_t >= SEC(0.6):
                        w = width(ACT)
                        if not 0.030 < w < 0.043:
                            st["pick_try"] += 1
                            if st["pick_try"] > 7:
                                print(f"[task {plan_i}] PICK_MISS x{st['pick_try']}", flush=True)
                                results[plan_i] = "PICK_MISS"
                                next_task()
                                fresh_state()
                                enter("park")
                                continue
                            st["psi_flip"] = not st.get("psi_flip", False)  # alternate wrist branch
                            gcs = st.get("g_cands")
                            if gcs:  # walk the grip-spot candidates every second retry
                                st["g_pick"] = gcs[min(st["pick_try"] // 2, len(gcs) - 1)]
                            print(f"[pick{K}] MISS (w {w * 1e3:.1f}mm) retry {st['pick_try']} "
                                  f"(grip spot {st.get('g_pick', 0) * 1e3:.0f}mm)", flush=True)
                            enter("pre_pick")
                        else:
                            st["p_flat_pick"] = w / 2.0
                            st["pick_q"] = qr.clone()
                            st["pick_pad"] = pr + quat_apply(qr.unsqueeze(0), PAD.unsqueeze(0))[0]
                            yhat = quat_apply(qr.unsqueeze(0), V3(0, 1, 0).unsqueeze(0))[0]
                            axis_w = quat_apply(lq.unsqueeze(0), V3(0, 0, 1).unsqueeze(0))[0]
                            n_axis = yhat / yhat.norm()
                            cr = torch.cross(n_axis, axis_w, dim=0)
                            st["reor_axis"] = n_axis if cr[2] > 0 else -n_axis
                            print(f"[pick{K}] width {w * 1e3:.1f}mm", flush=True)
                            enter("lift")
            elif phase == "lift":
                st["pick_pad"][2] = min(st["pick_pad"][2] + 0.11 / CTRL_HZ, 1.26)
                servo_pad(ACT, st["pick_pad"], st["pick_q"], st["p_flat_pick"] - args.pinch_wind_n / GRIP_KP, a)
                park(OTH, a, 0.0)
                if ph_t >= SEC(1.6):
                    st["reor_q0"] = st["pick_q"].clone()
                    enter("reorient")
            elif phase == "reorient":
                s_frac = min(1.0, ph_t / SEC(3.2))
                qd = quat_from_angle_axis(torch.tensor([math.pi / 2 * s_frac], device=device), st["reor_axis"].unsqueeze(0))[0]
                q_t = quat_mul(qd.unsqueeze(0), st["reor_q0"].unsqueeze(0))[0]
                servo_pad(ACT, st["pick_pad"], q_t, st["p_flat_pick"] - args.pinch_wind_n / GRIP_KP, a)
                park(OTH, a, 0.0)
                if ph_t >= SEC(3.8):
                    st["reor_q1"] = q_t.clone()
                    st["carry_from"] = st["pick_pad"].clone()
                    st["ground"] = bench
                    # BAND-AWARE plant spot: classic P1 sits 0.27m from the right base — under
                    # the arm's 0.31-0.43 working band — and at task 6's configuration every
                    # recage approach to it is fold-infeasible (v89/v90: stuck 630-1016mm at all
                    # wrist yaws). Pick the first candidate in-band and clear of slab + welds.
                    rb_c = torch.tensor([0.25, -0.25], device=device)
                    tpc_c = table_pq()[0][:2]
                    if TARM == "left":
                        # HANDOVER spot: right plants/erects here, LEFT recages. Live grid scan:
                        # inside both carry envelopes, off the slab (frame-aware edge margin),
                        # clear of welded legs and of the target stud. The lens is narrow — at the
                        # -157 endgame only a strip just east of the delivered stud survives.
                        best_h = find_ho_spot(SLOT)
                        if best_h is None:
                            print(f"[task {plan_i}] HANDOVER_NO_SPOT — no both-arms plant spot", flush=True)
                            results[plan_i] = "HANDOVER_NO_SPOT"
                            enter("abandon")
                            continue
                        p1_eff = best_h
                        st["ho_spot"] = p1_eff
                        # (B14's aim-overshoot REVERTED: aiming past the spot deepened the sag and
                        # the stand-release blades flung the leg off the bench. Land short and let
                        # the HOP-relay walk it in — its 7cm hops planted dead-vertical.)
                        print(f"[plant{K}] HANDOVER spot {p1_eff}", flush=True)
                        st["carry_to"] = V3(p1_eff[0], p1_eff[1], bench + 0.01 - NUT_BOTTOM + args.grip_local + 0.06)
                        st["carry_T"] = SEC(2.6)
                        enter("carry_p1")
                        continue
                    # PROVEN P1 FIRST (it welded legs 0-2 all lineage; banding it out broke the
                    # from-scratch task 0 — verify run 2); alternates only when P1 is cluttered.
                    p1_eff = None
                    for i_c, cnd in enumerate((P1, (-0.06, -0.35), (0.06, -0.38), (-0.14, -0.38))):
                        c_t = torch.tensor(cnd, device=device)
                        if i_c > 0 and not (0.30 <= float((c_t - rb_c).norm()) <= 0.43):
                            continue
                        if float((c_t - tpc_c).norm()) < 0.42:
                            continue
                        if any(float((c_t - leg_pq(int(kk))[0][:2]).norm()) < 0.18
                               for kk in assigned if bool(scene.welded[0, int(kk)])):
                            continue
                        p1_eff = cnd
                        break
                    if p1_eff is None:
                        p1_eff = P1
                    if p1_eff != P1:
                        print(f"[plant{K}] band-aware plant spot {p1_eff}", flush=True)
                    st["carry_to"] = V3(p1_eff[0], p1_eff[1], bench + 0.01 - NUT_BOTTOM + args.grip_local + 0.06)
                    st["carry_T"] = SEC(2.0)
                    enter("carry_p1")
            elif phase == "carry_p1":
                s_frac = min(1.0, ph_t / st["carry_T"])
                tgt = st["carry_from"] + (st["carry_to"] - st["carry_from"]) * s_frac
                servo_pad(ACT, tgt, st["reor_q1"], st["p_flat_pick"] - args.pinch_wind_n / GRIP_KP, a)
                park(OTH, a, 0.0)
                if ph_t >= st["carry_T"] + SEC(1.0):
                    st.pop("plant_z", None)
                    enter("plant")
            elif phase == "plant":  # lower the hanging leg until its bottom rests on the bench
                if "plant_z" not in st:
                    st["plant_z"] = (pr + quat_apply(qr.unsqueeze(0), PAD.unsqueeze(0))[0])[2].item()
                    st["plant_hist"] = []
                st["plant_z"] = max(st["plant_z"] - 0.05 / CTRL_HZ, st["ground"] + 0.075)
                st["plant_hist"].append(lp[2].item())
                st["plant_hist"] = st["plant_hist"][-SEC(0.3):]
                quiet = len(st["plant_hist"]) == SEC(0.3) and (max(st["plant_hist"]) - min(st["plant_hist"])) < 2e-4
                servo_pad(ACT, V3(st["carry_to"][0], st["carry_to"][1], st["plant_z"]),
                          st["reor_q1"], st["p_flat_pick"] - args.pinch_wind_n / GRIP_KP, a)
                park(OTH, a, 0.0)
                if (quiet and ph_t > SEC(0.6)) or st["plant_z"] <= st["ground"] + 0.076:
                    _, axw = leg_tilt(K)
                    anchor = lp + NUT_BOTTOM * axw
                    st["anchor"] = anchor[:2].clone()
                    print(f"[plant{K}] bottom down (leg z {lp[2].item():.3f})", flush=True)
                    if st.get("relay"):
                        # RELAY plant: erect can't run (top cage — its side-grip pad target would
                        # press ~20cm below the cage and shove the leg over). Instead GATE on
                        # tilt: the cage slack lets the leg pick up lean during the ferry, and
                        # releasing a tilted plant topples it (B11 cycle 2: 90deg). Re-lift and
                        # re-carry — hanging from the cage self-straightens the shaft.
                        tilt_rp, _ = leg_tilt(K)
                        sp_rp = st.get("hop_tgt") or st.get("ho_spot", (lp[0].item(), lp[1].item()))
                        d_sp = math.hypot(lp[0].item() - sp_rp[0], lp[1].item() - sp_rp[1])
                        st.pop("hop_tgt", None)  # every plant ends the hop; the next cycle
                        # computes a fresh one from wherever the leg actually is
                        if tilt_rp >= 8.0 or d_sp > 0.10:
                            # gate on tilt AND position vs the HOP target: releasing a tilted
                            # plant topples it (B11: rolled off the bench). Gate at 8.0: the
                            # free leg TIPS only past 8.4deg and self-rights below it — B15's
                            # plants were pinned at 7.7deg (deterministic reach-sag angle) and
                            # a 5deg gate rejected every one of them.
                            st["relay_try"] = st.get("relay_try", 0) + 1
                            if st["relay_try"] > 10:
                                print(f"[task {plan_i}] RELAY_TILT x{st['relay_try']} — abandoning", flush=True)
                                results[plan_i] = "RELAY_TILT"
                                enter("abandon")
                                continue
                            print(f"[relay{K}] plant off (tilt {tilt_rp:.1f}deg, {d_sp * 1e3:.0f}mm "
                                  f"from the hop target) — re-lift retry {st['relay_try']}", flush=True)
                            st["carry_from"] = None
                            enter("carry_st")
                        else:
                            enter("stand_release")
                    else:
                        enter("erect")
            elif phase == "erect":  # stand the pole about the planted anchor
                tilt_now, axis_w = leg_tilt(K)
                pad_tgt = V3(st["anchor"][0].item(), st["anchor"][1].item(),
                             st["ground"] + 0.01 - NUT_BOTTOM + args.grip_local - 0.003)
                if tilt_now >= 4.0:
                    e = torch.cross(axis_w, V3(0, 0, 1), dim=0)
                    e = e / max(e.norm().item(), 1e-6)
                    dq = quat_from_angle_axis(torch.tensor([math.radians(30.0) / CTRL_HZ], device=device), e.unsqueeze(0))[0]
                    st["reor_q1"] = quat_mul(dq.unsqueeze(0), qr.unsqueeze(0))[0]
                    st["erect_settle"] = 0
                else:
                    st["erect_settle"] = st.get("erect_settle", 0) + 1
                servo_pad(ACT, pad_tgt, st["reor_q1"], st["p_flat_pick"] - args.pinch_wind_n / GRIP_KP, a)
                park(OTH, a, 0.0)
                if st.get("erect_settle", 0) >= SEC(0.6):
                    print(f"[erect{K}] vertical (tilt {tilt_now:.1f}deg)", flush=True)
                    enter("stand_release")
                elif ph_t >= SEC(10.0):
                    print(f"[task {plan_i}] ERECT_FAIL (tilt {tilt_now:.1f}deg)", flush=True)
                    results[plan_i] = "ERECT_FAIL"
                    enter("abandon")
            elif phase == "stand_release":  # open and rise straight up: the leg FREE-STANDS.
                # KEEP the erect orientation for the whole rise — four_v4 switched yaw at 0.4s and
                # the rotating open blades flung the leg off the bench. Reorient only above the top.
                servo(ACT, V3(pr[0], pr[1], min(pr[2].item() + 0.12 / CTRL_HZ, 1.44)),
                      st["reor_q1"] if pr[2].item() < 1.40 else down[ACT], OPEN, a)
                park(OTH, a, 0.0)
                if ph_t >= SEC(3.0):
                    tilt_now, _ = leg_tilt(K)
                    print(f"[stand{K}] free-standing tilt {tilt_now:.1f}deg at "
                          f"{[round(v, 2) for v in lp[:2].tolist()]}", flush=True)
                    d_t3 = float((lp[:2] - BASES[TARM]).norm())
                    if TARM == "left" and d_t3 > 0.65:
                        d_r3 = float((lp[:2] - BASES["right"]).norm())
                        if 0.24 <= d_r3 <= 0.66:
                            # RELAY: the leg stands outside the thread arm's reach (plant sag at
                            # the right arm's reach edge, or a knock) — RIGHT re-cages it and
                            # ferries it to the handover spot before the left takes over
                            st["act_arm"], st["ee_prev"] = "right", None
                            st["relay"] = True
                            print(f"[task {plan_i}] leg{K} out of left reach ({d_t3 * 1e3:.0f}mm) "
                                  f"— RIGHT relays it to the handover spot", flush=True)
                        else:
                            print(f"[task {plan_i}] leg{K} unreachable by BOTH arms "
                                  f"(L {d_t3 * 1e3:.0f} R {d_r3 * 1e3:.0f}mm)", flush=True)
                            results[plan_i] = "LEG_UNREACHABLE"
                            enter("abandon")
                            continue
                    else:
                        st["act_arm"], st["ee_prev"] = TARM, None
                        st.pop("relay", None)
                    enter("recage")
            elif phase == "recage":  # top-down threading cage over the free-standing leg
                _, axis_w = leg_tilt(K)
                top = lp + SHAFT_TOP * axis_w
                cage_z = top[2].item() + args.palm_gap
                hand_goal = V3(top[0].item(), top[1].item(), cage_z)  # palm frame = pad-free servo
                if top[2] < 0.90:
                    # the leg is OFF the bench (verify run: the goal tracked a fallen leg down
                    # to z=0.08) — nothing to cage
                    print(f"[task {plan_i}] LEG_FELL (top z {top[2]:.2f})", flush=True)
                    results[plan_i] = "LEG_FELL"
                    enter("abandon")
                    continue
                # the cage is yaw-free around a vertical shaft: re-approaches rotate the wrist
                # yaw to change the whole arm fold — at 0.26m from the base the 0-yaw fold is
                # infeasible in most rollouts (v89: stuck 964-1016mm out)
                q_rc = yaw_about_z(down[ACT], math.radians(st.get("rc_yaw_off", 0.0)))
                st["rc_q"] = q_rc
                yaw_gap = abs((yaw_of(qr) - yaw_of(q_rc) + 180.0) % 360.0 - 180.0)
                near_shaft = (float((pr[:2] - top[:2]).norm()) < 0.10 and pr[2] < top[2] + 0.12
                              and yaw_gap > 15.0)
                herr = (pr - hand_goal).norm().item()
                if near_shaft and herr >= 0.05:
                    # NEVER rotate the wrist with the jaws around the shaft: the verify-1 yaw-
                    # changing re-approach swept the standing leg 40cm off the bench. Rise first,
                    # keeping the CURRENT yaw. (yaw_gap condition: without it this branch fought
                    # every normal descent — verify5 hovered at top+0.12/0.20 forever, 9/9 miss)
                    servo(ACT, V3(pr[0].item(), pr[1].item(), top[2].item() + 0.20), qr, OPEN, a)
                elif herr < 0.05:
                    # nearly there (verify: 16mm miss then a wasteful full re-approach) —
                    # finish with a direct fine servo, no rise
                    servo(ACT, hand_goal, q_rc, OPEN, a)
                else:
                    high_travel(ACT, hand_goal, q_rc, OPEN, a)
                if float((ee_pose(OTH)[0] - PARKS[OTH]).norm()) > 0.15:
                    # rise-then-travel retreat: B10's FLAT park path from the handover release
                    # crossed the standing leg (park z 1.30 vs leg top 1.31) and dragged it
                    # 0.45m east into the right base
                    high_travel(OTH, PARKS[OTH], down[OTH], 0.0, a)
                else:
                    park(OTH, a, 0.0)
                # gate thresholds match the CAGE's real tolerance (~±20mm on a Ø40 shaft with
                # wide-open jaws): the 5mm gate rejected proven-good approaches (verify1's 16mm
                # "miss" would have caged fine) and burned every retry (verify4: RECAGE_MISS x9
                # at the 5/5-proven task-0 configuration)
                if ph_t >= SEC(2.0) and herr < 0.018:
                    enter("reclose")
                elif ph_t >= SEC(7.0):
                    if herr < 0.025:
                        enter("reclose")
                    else:
                        # NEVER close unconverged: the blind close beside the shaft grazes the
                        # standing leg and the fallback plant then knocks it flat (v85: 74.8deg,
                        # v88: 83.3deg). Re-approach instead — no contact, no knock.
                        st["cage_try"] += 1
                        if st["cage_try"] > 8:
                            print(f"[task {plan_i}] RECAGE_MISS x{st['cage_try']}", flush=True)
                            results[plan_i] = "RECAGE_MISS"
                            enter("abandon")
                            continue
                        if herr >= 0.15:
                            # far off: a different wrist yaw = a different arm fold
                            st["rc_yaw_off"] = (0.0, 45.0, -45.0, 90.0)[st["cage_try"] % 4]
                        # else: CONVERGING (v91 hit 66mm as the budget expired) — keep the yaw
                        print(f"[recage{K}] approach unconverged ({herr * 1e3:.0f}mm) — "
                              f"re-approach {st['cage_try']} (yaw {st.get('rc_yaw_off', 0.0):.0f}) "
                              f"hand=({pr[0]:.2f},{pr[1]:.2f},{pr[2]:.2f}) "
                              f"goal=({hand_goal[0]:.2f},{hand_goal[1]:.2f},{hand_goal[2]:.2f})", flush=True)
                        enter("recage")
            elif phase == "reclose":
                _, axis_w = leg_tilt(K)
                top = lp + SHAFT_TOP * axis_w
                servo(ACT, V3(top[0].item(), top[1].item(), top[2].item() + args.palm_gap),
                      st.get("rc_q", down[ACT]), 0.0, a)
                if float((ee_pose(OTH)[0] - PARKS[OTH]).norm()) > 0.15:
                    high_travel(OTH, PARKS[OTH], down[OTH], 0.0, a)  # safe retreat (see recage)
                else:
                    park(OTH, a, 0.0)
                if ph_t >= SEC(0.6):
                    w = width(ACT)
                    if not 0.030 < w < 0.043:
                        st["cage_try"] += 1
                        if st["cage_try"] > 3:
                            print(f"[task {plan_i}] RECAGE_MISS x{st['cage_try']}", flush=True)
                            results[plan_i] = "RECAGE_MISS"
                            enter("abandon")
                            continue
                        print(f"[recage{K}] MISS (w {w * 1e3:.1f}mm) retry {st['cage_try']}", flush=True)
                        tilt_rc, _ = leg_tilt(K)
                        if tilt_rc < 5.0:
                            # the leg is STILL STANDING (jaws just closed beside it) — re-approach
                            # the cage; v85 re-entered plant here, which lowered the EMPTY hand
                            # onto the standing leg and knocked it to 74.8deg
                            enter("recage")
                        else:
                            st.pop("plant_z", None)
                            st["carry_to"] = V3(lp[0].item(), lp[1].item(), lp[2].item() + args.grip_local + 0.05)
                            enter("plant")  # the leg tipped: plant+erect it again where it is
                    else:
                        st["p_flat_top"] = w / 2.0
                        st["grip_off"] = pr[2].item() - lp[2].item()
                        print(f"[recage{K}] width {w * 1e3:.1f}mm grip_off {st['grip_off'] * 1e3:.0f}mm", flush=True)
                        st["carry_from"] = None
                        enter("carry_st")
            elif phase == "carry_st":  # carry the hanging leg to the station stud
                stud_live = stud_world(SLOT)
                if st.get("relay"):
                    # RELAY: destination is a capped HOP toward the handover spot, not the spot
                    # itself — the short (7cm) relay hop planted dead-vertical every time while
                    # full-distance hops tilted 12-23deg (extension change per hop is what tilts)
                    spot_r = st.get("ho_spot") or find_ho_spot(SLOT)
                    if spot_r is None:
                        print(f"[task {plan_i}] HANDOVER_NO_SPOT — no both-arms plant spot", flush=True)
                        results[plan_i] = "HANDOVER_NO_SPOT"
                        enter("abandon")
                        continue
                    st["ho_spot"] = spot_r
                    if "hop_tgt" not in st:
                        v_hop = torch.tensor([spot_r[0] - lp[0].item(), spot_r[1] - lp[1].item()], device=device)
                        d_hop = float(v_hop.norm())
                        step_h = min(0.08, d_hop)
                        h_t = (lp[0].item() + float(v_hop[0]) / max(d_hop, 1e-9) * step_h,
                               lp[1].item() + float(v_hop[1]) / max(d_hop, 1e-9) * step_h)
                        st["hop_tgt"] = h_t
                        print(f"[relay{K}] hop {step_h * 1e3:.0f}mm toward the spot -> "
                              f"({h_t[0]:.2f},{h_t[1]:.2f})", flush=True)
                    stud_live = V3(st["hop_tgt"][0], st["hop_tgt"][1], 0.0)
                if st["carry_from"] is None:
                    st["carry_from"] = pr.clone()
                    leg_z_goal = (bench + 0.025 - NUT_BOTTOM) if st.get("relay") \
                        else (slab_top + 0.035 - NUT_BOTTOM + 0.012)
                    st["carry_to"] = V3(stud_live[0], stud_live[1], leg_z_goal + st["grip_off"])
                    st["rise_T"], st["carry_T"] = SEC(0.9), SEC(2.2)
                if ph_t <= st["rise_T"]:
                    zf = st["carry_from"][2] + (st["carry_to"][2] + 0.02 - st["carry_from"][2]) * (ph_t / st["rise_T"])
                    tgt = V3(st["carry_from"][0], st["carry_from"][1], zf)
                    if ph_t == st["rise_T"]:  # the leg hangs free + vertical now: true offset
                        st["grip_off"] = pr[2].item() - lp[2].item()
                        st["carry_to"][2] = ((bench + 0.025 - NUT_BOTTOM) if st.get("relay")
                                            else (slab_top + 0.035 - NUT_BOTTOM + 0.012)) + st["grip_off"]
                else:
                    s_frac = min(1.0, (ph_t - st["rise_T"]) / st["carry_T"])
                    tgt = st["carry_from"] + (st["carry_to"] - st["carry_from"]) * s_frac
                    tgt = tgt.clone()
                    tgt[2] = st["carry_to"][2] + 0.02 * (1.0 - s_frac)
                    if s_frac >= 1.0:
                        tgt[0] = stud_live[0] + (pr[0] - lp[0])
                        tgt[1] = stud_live[1] + (pr[1] - lp[1])
                        tgt[2] = st["carry_to"][2]
                servo(ACT, tgt, down[ACT], pinch_top(args.pinch_wind_n), a)
                park(OTH, a, 0.0)
                if ph_t >= st["rise_T"] + st["carry_T"] + SEC(1.2):
                    if st.get("relay"):
                        # re-plant the ferried leg at the handover spot (top cage: keep the
                        # hold quat; skip the side-grip erect — the leg hangs vertical already)
                        st.pop("plant_z", None)
                        st["reor_q1"] = down[ACT]
                        st["ground"] = bench
                        enter("plant")
                    else:
                        st["lower_z"] = pr[2].item()
                        st["leg_hist"] = []
                        enter("lower")
            elif phase == "lower":  # spin-insertion lower onto the stud (dither only at tip contact)
                stud_live = stud_world(SLOT)
                st["lower_z"] -= (0.012 if dz < 0.055 else 0.02) / CTRL_HZ
                st["leg_hist"].append(lp[2].item())
                st["leg_hist"] = st["leg_hist"][-SEC(0.35):]
                quiet = len(st["leg_hist"]) == SEC(0.35) and (max(st["leg_hist"]) - min(st["leg_hist"])) < 2e-4
                near_tip = dz < 0.048
                dith = math.radians(15.0) * math.sin(2 * math.pi * 1.5 * ph_t / CTRL_HZ) if near_tip else 0.0
                # NO xy spiral: the 1.5mm orbit spoiled the arrival alignment and CAUSED the perch
                # it was meant to break (v19-v22 all first-perched at dz~25/lat~2.3; the spiral-free
                # v5-v15 runs landed with the yaw dither alone)
                xy = V3(stud_live[0] + (pr[0] - lp[0]).item(), stud_live[1] + (pr[1] - lp[1]).item(), st["lower_z"])
                # FIRM pinch: the 8N "compliant" lower let the leg lean into the 30deg lat-5mm wedge
                # (v26); a rigid vertical grip cannot tilt, alignment comes from the leg-origin
                # steering, and the contact force stays OSC-limited (pick_v2 placed at lat 0.1 firm)
                servo(ACT, xy, yaw_about_z(down[ACT], PREWIND + dith), pinch_top(args.pinch_wind_n), a)
                park(OTH, a, 0.0)
                if quiet and st["lower_z"] < pr[2].item() - 0.004:
                    if dz > 0.0258:
                        st["place_try"] += 1
                        if st["place_try"] > 5:
                            print(f"[task {plan_i}] PLACE_PERCH x{st['place_try']}", flush=True)
                            results[plan_i] = "PLACE_PERCH"
                            enter("abandon")
                            continue
                        print(f"[place{K}] PERCH dz {dz * 1e3:.1f} lat {lat * 1e3:.1f} retry {st['place_try']}", flush=True)
                        # retry on the existing grip; grip_off re-measured after the lift (measuring
                        # at the tilted perch bakes the tilt in -> crash-landing, v19-v24)
                        st["carry_from"] = None
                        enter("carry_st")
                    elif dz > 0.0245:
                        # CREST-REST (rest + ~one 1.86mm pitch, crest-on-crest): go DIRECTLY to the
                        # firm-grip wind — the first stroke's rotation drops it into the groove. Any
                        # soft phase here (straighten's 3N) lets the crest-balanced leg topple into
                        # the 30deg lat-5mm wedge (v26/v27's deterministic failure).
                        st["grip_off"] = pr[2].item() - lp[2].item()
                        st["dz_seek0"] = dz
                        st["wound"] = PREWIND
                        st["t_state"], st["ee0"] = "wind", None  # no soft seek either: firm all the way
                        st["brace_on"], st["brace_xy"] = True, None
                        st["brace_z"], st["brace_hist"], st["brace_touch"] = None, [], None
                        print(f"[place{K}] CREST-rest dz {dz * 1e3:.1f} lat {lat * 1e3:.1f} — winding in firm", flush=True)
                        enter("thread")
                    else:
                        print(f"[place{K}] rest dz {dz * 1e3:.1f} lat {lat * 1e3:.1f}", flush=True)
                        st["brace_on"], st["brace_xy"] = True, None
                        st["brace_z"], st["brace_hist"], st["brace_touch"] = None, [], None
                        enter("straighten")
            elif phase == "regrip_st":  # re-take the top cage on the (caged) perched leg, then retry
                _, axis_w = leg_tilt(K)
                top = lp + SHAFT_TOP * axis_w
                if ph_t < SEC(0.5):  # open in place (jaws still cage the shaft)
                    servo(ACT, pr, down[ACT], st["p_flat_top"] + args.cage_slack, a)
                elif ph_t < SEC(1.8):  # ride the open cage up to the proper grip height
                    servo(ACT, V3(top[0].item(), top[1].item(), top[2].item() + args.palm_gap),
                          down[ACT], st["p_flat_top"] + args.cage_slack, a)
                else:
                    servo(ACT, V3(top[0].item(), top[1].item(), top[2].item() + args.palm_gap),
                          down[ACT], 0.0, a)
                park(OTH, a, 0.0)
                if ph_t >= SEC(2.4):
                    w = width(ACT)
                    if 0.030 < w < 0.043:
                        st["p_flat_top"] = w / 2.0
                        st["grip_off"] = pr[2].item() - lp[2].item()
                        print(f"[regrip{K}] width {w * 1e3:.1f}mm grip_off {st['grip_off'] * 1e3:.0f}mm", flush=True)
                        st["carry_from"] = None
                        enter("carry_st")
                    else:
                        print(f"[task {plan_i}] REGRIP_MISS (w {w * 1e3:.1f}mm)", flush=True)
                        results[plan_i] = "REGRIP_MISS"
                        enter("abandon")
                        continue
            elif phase == "straighten":
                stud_live = stud_world(SLOT)
                tilt_now, _ = leg_tilt(K)
                zt = lp[2].item() + SHAFT_TOP + args.palm_gap
                # OPEN for the first second ONLY on the thread-watchdog path (the wrist unwinds a
                # large yaw there; swinging while pinched dragged the engaged leg to a 30deg wedge,
                # v16). On the normal place->straighten entry the wrist is already at PREWIND —
                # opening there just hands the metastable resting leg to B1b (v19/v20 perch loops).
                open_first = st.get("str_open_first", False) and ph_t < SEC(1.0)
                g = (st["p_flat_top"] + args.cage_slack) if open_first else pinch_top(args.pinch_n)
                servo(ACT, V3(stud_live[0], stud_live[1], zt), yaw_about_z(down[ACT], PREWIND), g, a)
                left_idle(a, OTH)
                if (ph_t >= SEC(1.2) and tilt_now < 3.0) or ph_t >= SEC(4.0):
                    st["str_open_first"] = False  # one-shot: only the watchdog path re-arms it
                    if tilt_now >= 3.0 or dz > 0.0258:
                        st["place_try"] += 1
                        if st["place_try"] > 5:
                            print(f"[task {plan_i}] PLACE_PERCH — abandoning leg{K}", flush=True)
                            results[plan_i] = "PLACE_PERCH"
                            enter("abandon")
                            continue
                        if tilt_now >= 5.0:
                            if st["place_try"] >= 2 and dz < 0.0258:
                                # PERSISTENT wedge AT rest: the deterministic lift-and-replace
                                # re-lands the identical wedge (v87: 30.3deg x4 at dz 23.0
                                # lat 4.6) — go FIRM-DIRECT into the wind instead (the crest
                                # recovery): the thread pulls the shaft vertical as it engages
                                st["grip_off"] = pr[2].item() - lp[2].item()
                                st["dz_seek0"] = dz
                                st["wound"] = PREWIND
                                st["t_state"], st["ee0"] = "wind", None
                                st["brace_on"], st["brace_xy"] = True, None
                                st["brace_z"], st["brace_hist"], st["brace_touch"] = None, [], None
                                print(f"[straighten{K}] wedge persists at rest (tilt {tilt_now:.1f}) "
                                      f"— FIRM-direct wind", flush=True)
                                enter("thread")
                                continue
                            # WEDGE (v65 leg2: 11deg lean, nut on the tip, bottom against the slab;
                            # dz/lat read as perfect rest) — re-lower can't right it, it just
                            # presses along the leaning shaft. LIFT-AND-REPLACE instead: hanging
                            # from the pinch self-straightens the shaft (the perch-retry path).
                            print(f"[straighten{K}] WEDGE (tilt {tilt_now:.1f}) — lift and re-place "
                                  f"retry {st['place_try']}", flush=True)
                            st["carry_from"] = None
                            enter("carry_st")
                            continue
                        print(f"[straighten{K}] not ready (tilt {tilt_now:.1f} dz {dz * 1e3:.1f}) re-lower", flush=True)
                        st["lower_z"] = pr[2].item()
                        st["leg_hist"] = []
                        enter("lower")
                    else:
                        st["grip_off"] = pr[2].item() - lp[2].item()
                        st["dz_seek0"] = dz
                        st["wound"] = PREWIND
                        st["t_state"], st["ee0"] = "seek", None
                        print(f"[straighten{K}] tilt {tilt_now:.1f}deg — threading", flush=True)
                        enter("thread")
            elif phase == "thread":
                ts = st["t_state"]
                grip = pinch_top(args.pinch_n if ts == "seek" else args.pinch_wind_n)
                lean = 0.0
                if ts == "seek":
                    st["wound"] += D_WIND
                    lean = args.lean
                    if st["dz_seek0"] - dz > 4e-4 or st["wound"] >= SEEK_CAP:
                        st["t_state"] = "wind"
                elif ts == "wind":
                    if st["ee0"] is None:
                        st["ee0"], st["stall_t"], st["w_start"] = ee_u, 0, st["wound"]
                        st["stall_run"] = 0
                    lean = args.lean
                    swept = math.radians(st["ee0"] - ee_u)
                    lag = st["w_start"] - st["wound"] - swept
                    st["stall_run"] = st["stall_run"] + 1 if lag >= 0.35 else 0
                    w_end = -W_LAND if st["stroke"] == 0 else st["w_start"] - SWEEP
                    if st["stall_run"] >= SEC(2.5):
                        st["t_state"], st["t"], st["ee0"] = "open", 0, None
                    elif st["wound"] > w_end:
                        if lag < 0.35:
                            st["wound"] = max(w_end, st["wound"] - D_WIND)
                    else:
                        st["stall_t"] += 1
                        if lag < 0.26 or st["stall_t"] >= SEC(1.5):
                            st["t_state"], st["t"], st["ee0"] = "open", 0, None
                    if st["t_state"] == "open":
                        st["stroke"] += 1
                elif ts == "open":
                    grip, st["t"] = st["p_flat_top"] + args.cage_slack, st["t"] + 1
                    if st["t"] >= SEC(0.4):
                        st["t_state"] = "rewind"
                elif ts == "rewind":
                    grip = st["p_flat_top"] + args.cage_slack
                    st["wound"] = min(W_LAND, st["wound"] + D_REWIND)
                    if st["wound"] >= W_LAND:
                        st["t_state"], st["t"] = "reclose", 0
                elif ts == "reclose":
                    grip, st["t"] = pinch_top(args.pinch_n), st["t"] + 1
                    if st["t"] >= SEC(0.5):
                        st["t_state"] = "wind"
                tilt_now, _ = leg_tilt(K)
                if tilt_now > 8.0 and st["re_straighten"] < 4:
                    st["re_straighten"] += 1
                    print(f"[thread{K}] tilt {tilt_now:.1f} — re-straighten {st['re_straighten']}", flush=True)
                    st["wound"], st["t_state"], st["ee0"] = 0.0, "seek", None
                    st["str_open_first"] = True  # the wrist will unwind a large yaw: open the jaws for it
                    enter("straighten")
                zt = lp[2].item() + st["grip_off"] - lean
                stud_live = stud_world(SLOT)
                servo(ACT, V3(stud_live[0], stud_live[1], zt), yaw_about_z(down[ACT], st["wound"]), grip, a)
                left_idle(a, OTH)
                if welded or dz <= 0.0115:
                    print(f"[done{K}] welded={int(welded)} dz={dz * 1e3:.1f} strokes={st['stroke']}", flush=True)
                    results[plan_i] = "SEATED" if welded else "THREAD_END_UNWELDED"
                    enter("retreat")
            elif phase == "retreat":
                zt = min(pr[2].item() + 0.05 / CTRL_HZ, slab_top + 0.45)
                servo(ACT, V3(pr[0], pr[1], zt), down[ACT], OPEN, a)
                park(OTH, a, 0.0)
                if ph_t >= SEC(2.5):
                    next_task()
                    fresh_state()
                    enter("park")
                    continue
            elif phase == "abandon":
                servo(ACT, V3(pr[0], pr[1], 1.34), qr if ph_t < SEC(0.3) else down[ACT], OPEN, a)
                park(OTH, a, 0.0)
                if ph_t >= SEC(2.0):
                    next_task()
                    fresh_state()
                    enter("park")
                    continue

            if phase in ("carry_st", "lower", "straighten", "thread") and lp[2].item() < bench - 0.05:
                print(f"[task {plan_i}] LEG_DROPPED", flush=True)
                results[plan_i] = "LEG_DROPPED"
                next_task()
                fresh_state()
                enter("park")
                continue

        # ======================= SLAB ROTATION (+90 per rot task) ================================
        else:
            done_yaw = yaw_acc["slab"]
            # gain schedule: pinned cage-drags get 250 (the pin raises friction; the cage slack
            # bounds stored energy so no stick-slip explosion); pushes 180; everything else 150
            hi = (350.0 if st.get("d_mode") == "recenter" else 250.0) if phase == "d_drag" \
                else (380.0 if st.get("e_shaft") else 180.0) if phase in ("c_push", "e_push") \
                else 150.0
            # shaft pushes at 180/3deg-lead delivered ~3N — under the 1.36kg assembly's ~12N
            # breakaway (mint8 NO-PROGRESS); 380 + the 6deg lead below restores v17's boost
            for osc_ in oscs.values():
                osc_._kp[:3] = hi
                osc_._kd = 2.0 * osc_._kp.sqrt()
            if phase == "park":
                # hold the pin until the DRAG hand is fully parked — the release kick (+40deg and
                # the v43 table-off-the-bench whip) happens while the drag arm is still near the
                # handle; the pin damps it (Haoxiang: the fall is partly edge-margin, so keep the
                # slab restrained whenever anything moves near it)
                if st.get("anc_state") is not None:
                    # release the anchor cage: straight up first (never move laterally with the
                    # jaws around the shaft), then park
                    if "anc_rel" not in st:
                        p_now = ee_pose(st["a_arm"])[0]
                        st["anc_rel"] = V3(p_now[0].item(), p_now[1].item(), p_now[2].item() + 0.22)
                    servo(st["a_arm"], st["anc_rel"], down[st["a_arm"]], OPEN, a)
                    park("right" if st["a_arm"] == "left" else "left", a, 0.0)
                    if ph_t >= SEC(1.4):
                        st.pop("anc_state", None)
                        st.pop("anc_goal", None)
                        st.pop("anc_rel", None)
                parm_h = st.get("pin_arm", "left")
                darm_h = "right" if parm_h == "left" else "left"
                drag_parked = float((ee_pose(darm_h)[0] - PARKS[darm_h]).norm()) < 0.06
                if st.get("anc_state") is not None:
                    pass  # anchor release in progress (handled above)
                elif st.get("pin_touch") is not None and not (drag_parked or ph_t >= SEC(6.0)):
                    center_pin(parm_h, a)
                else:
                    st.pop("pin_touch", None)
                    st.pop("pin_z", None)
                    st.pop("pin_hist", None)
                    park(parm_h, a, 0.0)
                if st.get("anc_state") is None:
                    park(darm_h, a, 0.0)
                # B9 SETTLE GATE: after a drag's release kick the slab can still be SLIDING —
                # B8 selected the next drag against a moving slab (ctr jumped 16cm in the same
                # second), the approach flailed into the moving assembly and whipped it off the
                # bench (yaw -155 -> -259 in ~1s). Select mechanisms only against a QUIET slab.
                if ph_t == 0:
                    st["settle_hist"] = []
                sh = st.setdefault("settle_hist", [])
                tp_sg = table_pq()[0][:2]
                sh.append((float(tp_sg[0]), float(tp_sg[1]), done_yaw))
                del sh[:-int(SEC(0.6))]
                slab_quiet = len(sh) == int(SEC(0.6)) and \
                    max(abs(h_[0] - sh[0][0]) + abs(h_[1] - sh[0][1]) for h_ in sh) < 0.005 and \
                    max(abs(h_[2] - sh[0][2]) for h_ in sh) < 1.0
                if int(scene.welded[0].sum()) < 3:
                    # B20: the gate exists for 3-weld release kicks; below 3 welds it subtly
                    # changes rotation stop-yaws and pushed task 2 into an unproven geometry
                    # (B18+B19 both deterministically ERECT_FAIL/RECAGE_MISS there)
                    slab_quiet = True
                if ph_t >= SEC(1.5) and st.get("anc_state") is None and slab_quiet:
                    # DELIVERY criterion: done once an empty stud sits in EITHER arm's threading
                    # disc (live poses; no fixed station point to chase — the slab may drift freely)
                    _, stn_d, stn_arm = station_slot2()
                    t_dir = task.get("dir", 1.0)
                    if not stud_threadable(stn_d) and (done_yaw - task["target"]) * t_dir >= -5.0:
                        # yaw target reached but the stud ISN'T in the disc (the window moves with
                        # the drifting centre — v83 declared "done" at 696mm and doomed task 6):
                        # RETARGET from the live geometry instead of declaring success
                        dd_x = delivery_delta()
                        task["dir"] = 1.0 if dd_x >= 0 else -1.0
                        task["target"] = done_yaw + dd_x + task["dir"] * 10.0
                        print(f"[rotate] yaw target hit, stud {stn_d * 1e3:.0f}mm — retarget "
                              f"{dd_x:+.0f}deg ({'CCW' if dd_x >= 0 else 'CW'})", flush=True)
                    if stud_threadable(stn_d):
                        print(f"[rotate] done at {done_yaw:.0f}deg (stud {stn_d * 1e3:.0f}mm from the {stn_arm} base)", flush=True)
                        results[plan_i] = "ROTATED"
                        next_task()
                        fresh_state()
                        enter("park")
                        continue
                    bl = st.setdefault("blacklist", set())
                    # RECENTER first when the slab has wandered: accumulated snap-back translation
                    # put the south edge over the bench lip in v28 (ctr (-0.32,-0.28)) and every
                    # push there tipped-and-snapped. A linear push walks it back toward C0.
                    tpc = table_pq()[0][:2]
                    off_c = C0 - tpc
                    # RE-ENABLED (v47): with TWO welded legs the pin no longer keeps drags
                    # drift-free (13.6cm/drag at 9.5cm overhang — the slab pivots on the bench lip
                    # and slid off in v46, the same fall Haoxiang saw in the task3 snap)
                    dsel = pick_drag(st.setdefault("blacklist", set()), task.get("dir", 1.0))
                    # recenter when the slab has wandered OR as a REPOSITIONING move when no drag
                    # handle is in anyone's zone. The pull target is C0 in the classic case; when
                    # centred-but-stuck (v62: STUCK at 35 with off<30mm) it is a DISPLACEMENT that
                    # walks some handle back to the 0.42m sweet radius of an arm.
                    rc_tgt = None
                    if not st.get("no_recenter"):
                        if float(off_c.norm()) > 0.16:
                            # trigger raised 0.10->0.16 (v58): every recenter round back-rotates
                            # the slab 12-23deg, so fewer rounds per drag wins; the mid-drag edge
                            # guard (0.26) still protects the bench edge
                            rc_tgt = C0.clone()
                        elif dsel is None:
                            best_disp, why = None, "refresh a drag handle"
                            # FIRST choice: translate the nearest empty stud straight into the
                            # threading disc — the endgame v64 missed by 18mm (548 vs 530)
                            s_del2, s_d2, s_arm2 = station_slot2()
                            if not stud_threadable(s_d2):
                                rb2 = BASES[s_arm2]
                                v_s = rb2 - stud_world(s_del2)[:2]
                                d_s = v_s / v_s.norm() * (float(v_s.norm()) - 0.42)
                                if float(d_s.norm()) <= 0.16 and float((tpc + d_s - C0).norm()) <= 0.26:
                                    best_disp, why = d_s, f"DELIVER stud {s_del2} by translation"
                            if best_disp is None:
                                for arm_r in ("right", "left"):
                                    for nm_r, h_r, _ in handles():
                                        v_r = BASES[arm_r] - h_r
                                        need = float(v_r.norm()) - 0.42
                                        if abs(need) < 0.01:
                                            continue
                                        d_r = v_r / v_r.norm() * need
                                        # cap 0.24 (was 0.16): bringing a far leg into the LEFT
                                        # arm's drag zone takes ~0.23; the 0.22-from-C0 bench cap
                                        # and ~40mm/pull gains keep each step gentle anyway
                                        if float(d_r.norm()) > 0.24 or float((tpc + d_r - C0).norm()) > 0.22:
                                            continue
                                        if best_disp is None or float(d_r.norm()) < float(best_disp.norm()):
                                            best_disp = d_r
                            if best_disp is not None:
                                rc_tgt = tpc + best_disp
                                print(f"[rotate] REPOSITION: pull the slab {float(best_disp.norm()) * 1e3:.0f}mm "
                                      f"to {why}", flush=True)
                    if rc_tgt is not None:
                        # CAGE-RECENTER: drag a cage handle linearly toward the spawn centre.
                        # Handle choice matters (v52): a handle on the far side of the centre
                        # (pull line passing THROUGH the centre) transmits pure translation
                        # (leg0: 50mm/pull); a sideways handle converts the pull to yaw and the
                        # shaft slips out of the jaws (leg1: 6mm then escape).
                        st["rc_n"] = st.get("rc_n", 0) + 1
                        if st.get("rc_yaw_mark") is None:
                            st["rc_yaw_mark"] = done_yaw
                        elif (done_yaw - st["rc_yaw_mark"]) * task.get("dir", 1.0) >= 5.0:
                            # the pulls themselves rotate the slab ~5deg/pull in the CW direction —
                            # that IS mission progress (v69 latched mid-endgame while advancing)
                            st["rc_n"], st["rc_yaw_mark"] = 1, done_yaw
                            # approach-failure bans expire on progress too: they're intermittent
                            # (v53 coupled on the 4th try; v70 STUCK with every pair banned)
                            st.pop("rc_bl", None)
                        if st["rc_n"] > 14:
                            st["no_recenter"] = True
                        best_r, best_al = None, -1e9
                        off_r = rc_tgt - tpc
                        dhp = off_r / off_r.norm()
                        # score alignment about the WELDED-MASS centroid, not the geometric centre:
                        # each welded leg (0.12kg vs slab 1.0) shifts the friction centroid ~3.5cm;
                        # pulls aimed "through the centre" still torque about the true centroid
                        w_tot, cen = 1.0, tpc.clone()
                        for k_c in assigned:
                            if bool(scene.welded[0, int(k_c)]):
                                cen = cen + 0.12 * leg_pq(int(k_c))[0][:2]
                                w_tot += 0.12
                        cen = cen / w_tot
                        # two-pass annulus: strict, then relaxed — v73 STUCK with every leg just
                        # outside [0.26,0.60] (left base to the far legs: 0.62-0.65)
                        for lo_a, hi_a in ((0.26, 0.60), (0.20, 0.70)):
                            for arm_c in ("right", "left"):  # BOTH arms scored — v66's right-arm-
                                # first break left a workspace-infeasible right pull retrying 14x
                                # while the left arm sat idle (pad_err 455mm: unreachable pose)
                                for nm_c, hxy_c, _ in handles():
                                    if not nm_c.startswith("leg"):
                                        # PULLS ARE LEG-ONLY: a corner's shallow 8mm cage with the
                                        # pull-quat is a rigid edge push — v67's left/cor1 "pull"
                                        # stick-slip-yanked the slab 25cm+(-27deg) in one burst
                                        continue
                                    if (arm_c, nm_c) in st.get("rc_bl", set()):  # never coupled
                                        continue
                                    if (arm_c, nm_c) in app_bl:
                                        # drag-approach bans apply to pulls too — same approach
                                        # mechanics, same flail (v81 wasted pulls re-trying them)
                                        continue
                                    if not (lo_a <= float((hxy_c - BASES[arm_c]).norm()) <= hi_a):
                                        continue
                                    rv_c = hxy_c - cen
                                    # PURE alignment (pull line through the CENTROID = clean
                                    # translation). The old torque bonus picked sideways handles
                                    # whose "pulls" yanked the slab (v68: left/leg1 221->389mm);
                                    # rotation comes from drags, pulls must only translate.
                                    al = float((rv_c * dhp).sum()) \
                                        + (0.02 if arm_c == "right" else 0.0)
                                    if al > best_al:
                                        best_r, best_al = (arm_c, nm_c), al
                            if best_r is not None:
                                break
                        if st.get("no_recenter"):
                            best_r = None
                        if best_r is not None:
                            st["d_arm"], st["d_name"], st["d_top"] = best_r[0], best_r[1], True
                            st["d_mode"] = "recenter"
                            st["rc_tgt_v"] = rc_tgt
                            st.pop("d_th", None); st.pop("d_flip", None)
                            print(f"[rotate] CAGE-RECENTER: {best_r[0]} drags {best_r[1]} (slab off by "
                                  f"{float(off_r.norm()) * 1e3:.0f}mm)", flush=True)
                            enter("d_hover")
                            continue
                    # ANCHORED two-arm rotation first (Haoxiang's strategy): translation-free by
                    # construction, and the anchor is scored so the slab-centre's orbit doubles
                    # as the recentring move. Needs >= 2 welded legs and a feasible pair.
                    asel = None
                    if sum(1 for kk in assigned if bool(scene.welded[0, int(kk)])) >= 2:
                        best_s = -1e9
                        t_dir0 = task.get("dir", 1.0)
                        for a_arm2 in ("left", "right"):
                            d_arm2 = "right" if a_arm2 == "left" else "left"
                            for nmA, hA, _ in handles():
                                if not nmA.startswith("leg"):
                                    continue
                                if not (0.28 <= float((hA - BASES[a_arm2]).norm()) <= 0.68):
                                    continue
                                # ceiling 0.64 -> 0.68 (B3): the anchor is a STATIC loose cage —
                                # reach-edge is fine (no arc to draw); at the -157 corner left/leg1
                                # sits at 0.664 and is the only alternative to the failing
                                # right/leg2 anchor
                                # NOTE: plain (arm, name) app_bl entries are NOT checked for the
                                # anchor — those bans come from DRAG-pose approaches (tangential
                                # quat); the anchor uses a plain top-down cage, a different pose
                                # class (the -157 corner's only unlock is anchoring the drag-banned
                                # leg2). Anchor-FAIL bans use the distinct ("anc", ...) key:
                                if ("anc", a_arm2, nmA) in app_bl:
                                    continue
                                for nmB, hB, _ in handles():
                                    # drag handle: legs OR corners (corners drag fine; only
                                    # PULLS are legs-only) — mint4's endgame had no leg pair
                                    # and never considered the reachable corners
                                    if nmB == nmA:
                                        continue
                                    if (d_arm2, nmB) in app_bl \
                                            or (d_arm2, nmB) in st.get("blacklist", set()):
                                        continue
                                    hrB = drag_headroom(d_arm2, hB, t_dir0, center=hA)
                                    if hrB < 10.0:
                                        continue
                                    rvA = tpc - hA
                                    orb = t_dir0 * torch.tensor([-rvA[1], rvA[0]], device=device)
                                    sc = hrB + 50.0 * float((orb / (orb.norm() + 1e-9) * off_c).sum())
                                    if sc > best_s:
                                        best_s, asel = sc, (a_arm2, nmA, d_arm2, nmB, hrB)
                    if asel is not None:
                        a_arm2, nmA, d_arm2, nmB, hrB = asel
                        st["d_mode"] = "anchored"
                        st["a_arm"], st["a_name"] = a_arm2, nmA
                        st["d_arm"], st["d_name"], st["d_top"] = d_arm2, nmB, True
                        t_dir0 = task.get("dir", 1.0)
                        st["d_goal"] = done_yaw + t_dir0 * (min(hrB, 30.0) - 4.0)
                        if (st["d_goal"] - task["target"]) * t_dir0 > 0:
                            st["d_goal"] = task["target"]
                        st["d_dir"] = t_dir0
                        st["d_start_yaw"] = done_yaw
                        st["anc_state"] = None
                        st["anc_t"] = 0  # restart the yaw-ladder for the new anchor
                        st.pop("d_th", None); st.pop("d_flip", None)
                        print(f"[rotate] ANCHORED: {a_arm2} cages {nmA}, {d_arm2} arcs {nmB} "
                              f"(headroom {hrB:.0f}) -> {st['d_goal']:.0f}deg", flush=True)
                        enter("d_hover")
                        continue
                    # CAGE DRAGS ONLY (welded legs + slab corners): the smooth mechanism. Rigid
                    # edge pushes stick-slip-explode the 1.0kg slab (v31-v35) and are retired.
                    n_weld3 = int(scene.welded[0].sum())
                    drag_block = 0.20 if n_weld3 >= 3 else 0.26
                    # B3: at >= 3 welds a plain drag is pure downside — B2's left/cor2 drag
                    # started at 0.245 off, stick-slipped the WRONG way (-8deg) while translating
                    # 90mm SW, and the release kick put the slab over the lip. STUCK > fall.
                    if dsel is not None and float(off_c.norm()) > drag_block:
                        # NEVER start a drag beyond the safety radius (v67 dragged from 390mm
                        # off; verify8 re-tested a 0.32 hysteresis and the permitted drag whipped
                        # the slab to yaw 275 and off the bench — a STUCK keeps a 3/4 table, a
                        # fall destroys it; the block stays strict at the in-drag guard radius)
                        print(f"[rotate] slab too far out ({float(off_c.norm()) * 1e3:.0f}mm) and "
                              f"recenter unavailable — STUCK at {done_yaw:.0f}deg", flush=True)
                        dsel = None
                    if dsel is None:
                        # LAST MECHANISM — SHAFT-PUSH (Haoxiang): fingertips push a welded leg's
                        # shaft tangentially; the push pose is nearly unconstrained in wrist yaw,
                        # so it works at bearings where every cage pose is fold-infeasible
                        psel, best_hp = None, 0.0
                        t_dir0 = task.get("dir", 1.0)
                        for arm_p in ("right", "left"):
                            for nmP, hP, _ in handles():
                                if not nmP.startswith("leg"):
                                    continue
                                if (arm_p, "push" + nmP) in st.get("blacklist", set()):
                                    continue
                                if not (0.28 <= float((hP - BASES[arm_p]).norm()) <= 0.55):
                                    continue
                                rvP = hP - tpc
                                rP = float(rvP.norm())
                                th0P = math.atan2(float(rvP[1]), float(rvP[0]))
                                dthP = 0.0
                                while dthP < math.radians(50):
                                    thP = th0P + t_dir0 * (dthP + math.radians(3))
                                    pP = tpc + rP * torch.tensor([math.cos(thP), math.sin(thP)], device=device)
                                    if not (0.28 <= float((pP - BASES[arm_p]).norm()) <= 0.55):
                                        break
                                    dthP += math.radians(3)
                                hrP = math.degrees(dthP)
                                if hrP > best_hp:
                                    best_hp, psel = hrP, (arm_p, nmP, hrP)
                        if psel is not None and best_hp >= 8.0:
                            arm_p, nmP, hrP = psel
                            st["e_arm"], st["e_idx"] = arm_p, 0
                            st["e_shaft"] = nmP
                            st["e_dir"] = t_dir0
                            st["push_mode"] = None
                            st["d_goal"] = done_yaw + t_dir0 * min(hrP, 22.0)
                            st["d_start_yaw"] = done_yaw
                            st.pop("p_th", None)
                            st.pop("p_cmd", None)
                            st.pop("braking", None)
                            print(f"[rotate] SHAFT-PUSH: {arm_p} pushes {nmP} "
                                  f"(headroom {hrP:.0f}) -> {st['d_goal']:.0f}deg", flush=True)
                            enter("e_hover")
                            continue
                        # AMNESTY before surrender: mission-level bans persist across tasks, but
                        # they were earned at OLDER geometry — clear them once per task and
                        # re-scan before burning the task (mint5b: 15 instant-STUCK tasks while
                        # a valid anchored pair existed behind a stale ban)
                        if app_bl and not st.get("amnesty"):
                            st["amnesty"] = True
                            app_bl.clear()
                            print("[rotate] ban AMNESTY — re-scanning", flush=True)
                            continue
                        # REVERSAL before surrender: deliveries repeat every 90deg, and the OTHER
                        # direction may have fresh handles even when it is FARTHER (v81: CW
                        # nearer but fully exhausted -> STUCK; CCW had headroom). Flip whenever
                        # the current direction dead-ends. Max 2 flips per task.
                        if st.get("rev_n", 0) < 2:
                            dirn_r = -task.get("dir", 1.0)
                            dd_r = delivery_delta(force_sgn=dirn_r)
                            if dd_r is not None:
                                st["rev_n"] = st.get("rev_n", 0) + 1
                                task["dir"] = dirn_r
                                task["target"] = done_yaw + dd_r + dirn_r * 10.0
                                st.pop("blacklist", None)
                                st.pop("rc_bl", None)
                                app_bl.clear()  # bearings change wholesale on reversal
                                st.pop("no_recenter", None)
                                st["rc_n"], st["rc_yaw_mark"] = 0, done_yaw
                                print(f"[rotate] REVERSING to {'CCW' if dirn_r > 0 else 'CW'} "
                                      f"(delivery {dd_r:+.0f}deg, reversal {st['rev_n']})", flush=True)
                                continue
                        print(f"[rotate] STUCK at {done_yaw:.0f}deg", flush=True)
                        results[plan_i] = f"STUCK@{done_yaw:.0f}"
                        next_task()
                        fresh_state()
                        enter("park")
                        continue
                    hr, arm, name, topdown = dsel
                    st["d_arm"], st["d_name"], st["d_top"] = arm, name, topdown
                    st["d_mode"] = "rot"
                    # arc capped at 45deg: with the MID-DRAG edge guard (ends any drag at 260mm
                    # off-centre) the drift risk is bounded per-step, and every recenter round
                    # costs 12-20deg of back-rotation — longer arcs win (v59: 30deg cap = +7 net)
                    t_dir = task.get("dir", 1.0)
                    arc = min(hr, 45.0) - 4.0
                    st["d_goal"] = done_yaw + t_dir * arc
                    if (st["d_goal"] - task["target"]) * t_dir > 0:
                        st["d_goal"] = task["target"]
                    st["d_dir"] = t_dir
                    st["d_start_yaw"] = done_yaw
                    st.pop("d_th", None); st.pop("d_flip", None)
                    print(f"[rotate] {arm} drags {name} (headroom {hr:.0f}deg) -> {st['d_goal']:.0f}deg", flush=True)
                    enter("d_hover")
            elif phase in ("c_hover", "c_down", "c_push", "c_out"):
                # COUPLE push: right on pt i, left on the opposite pt j — both CCW tangent, net
                # force ~zero, pure torque. Per-arm live pursuit with a 6deg lead.
                done = True
                for arm, idx in (("right", st["c_i"]), ("left", st["c_j"])):
                    pl_ = EDGE_LOCAL[idx]
                    w = edge_world(pl_)
                    tp, tq = table_pq()
                    cxy_l = tp[:2]
                    rv = w - cxy_l
                    t_hat = torch.tensor([-rv[1], rv[0]], device=device)
                    t_hat = t_hat / t_hat.norm()
                    nl = (1.0, 0.0) if pl_[0] > 0.2 else (-1.0, 0.0) if pl_[0] < -0.2 else (0.0, 1.0) if pl_[1] > 0.2 else (0.0, -1.0)
                    nv = quat_apply(tq.unsqueeze(0), torch.tensor([[nl[0], nl[1], 0.0]], device=device))[0][:2]
                    PUSH_Z = 1.125
                    if phase == "c_hover":
                        tgt = V3(w[0] + nv[0] * 0.06 - t_hat[0] * 0.04, w[1] + nv[1] * 0.06 - t_hat[1] * 0.04, PUSH_Z)
                        high_travel(arm, tgt, down[arm], 0.0, a)
                        done = done and ((ee_pose(arm)[0] - tgt).norm().item() < 0.012)
                    elif phase == "c_down":
                        tgt = V3(w[0] + nv[0] * 0.02 - t_hat[0] * 0.03, w[1] + nv[1] * 0.02 - t_hat[1] * 0.03, PUSH_Z)
                        servo(arm, tgt, down[arm], 0.0, a)
                    elif phase == "c_push":
                        tgt = V3(w[0] + t_hat[0] * 0.025 - nv[0] * 0.006, w[1] + t_hat[1] * 0.025 - nv[1] * 0.006, PUSH_Z)
                        servo(arm, tgt, down[arm], 0.0, a)
                    else:  # c_out
                        tgt = V3(w[0] + nv[0] * 0.08, w[1] + nv[1] * 0.08, PUSH_Z + 0.12)
                        servo(arm, tgt, down[arm], 0.0, a)
                if phase == "c_hover" and ((ph_t >= SEC(2.5) and done) or ph_t >= SEC(9.0)):
                    enter("c_down")
                elif phase == "c_down" and ph_t >= SEC(1.4):
                    enter("c_push")
                elif phase == "c_push":
                    if done_yaw < st["d_start_yaw"] - 3.0:
                        st.setdefault("blacklist", set()).add(("pair", st["c_i"], st["c_j"]))
                        print(f"[rotate] couple ROLLBACK to {done_yaw:.0f}deg — abort pair", flush=True)
                        enter("c_out")
                    elif done_yaw >= st["d_goal"] or ph_t >= SEC(25.0):
                        gained = done_yaw - st["d_start_yaw"]
                        if gained < 3.0:
                            st.setdefault("blacklist", set()).add(("pair", st["c_i"], st["c_j"]))
                            print(f"[rotate] couple end: {done_yaw:.0f}deg (gained {gained:.0f} — blacklist pair)", flush=True)
                        else:
                            st.setdefault("blacklist", set()).clear()
                            print(f"[rotate] couple end: {done_yaw:.0f}deg (goal {st['d_goal']:.0f})", flush=True)
                        enter("c_out")
                elif phase == "c_out" and ph_t >= SEC(1.5):
                    enter("park")
            elif phase in ("e_hover", "e_down", "e_push", "e_out"):
                arm, idx = st["e_arm"], st["e_idx"]
                other = "left" if arm == "right" else "right"
                park(other, a, 0.0)
                tp, tq = table_pq()
                cxy_l = tp[:2]
                dd_p = st.get("e_dir", 1.0)
                if st.get("e_shaft"):
                    # SHAFT-PUSH (Haoxiang): fingertips push a welded LEG SHAFT tangentially —
                    # a push pose has none of the drag cage's wrist-fold constraints, so it
                    # works where the cage pose is infeasible (the -157 dead zone)
                    w = handle_xy(st["e_shaft"])
                    rv = w - cxy_l
                    nv = rv / rv.norm()  # radial outward = the face to stay behind
                    t_hat = dd_p * torch.tensor([-rv[1], rv[0]], device=device)
                    t_hat = t_hat / t_hat.norm()
                    PUSH_Z = 1.21  # fingertips (~PUSH_Z-0.10) land at ~1.11: on the SHAFT, above
                    # the slab top 1.044 (1.10 put them at ~1.00 — pressing the slab edge/air)
                else:
                    pl_ = EDGE_LOCAL[idx]
                    w = edge_world(pl_)
                    rv = w - cxy_l
                    t_hat = torch.tensor([-rv[1], rv[0]], device=device)
                    t_hat = t_hat / t_hat.norm()
                    nl = (1.0, 0.0) if pl_[0] > 0.2 else (-1.0, 0.0) if pl_[0] < -0.2 else (0.0, 1.0) if pl_[1] > 0.2 else (0.0, -1.0)
                    nv = quat_apply(tq.unsqueeze(0), torch.tensor([[nl[0], nl[1], 0.0]], device=device))[0][:2]
                    PUSH_Z = 1.125  # hand z: closed fingertips at ~1.022, mid slab side-face (0.994-1.044)
                if phase == "e_hover":
                    tgt = V3(w[0] + nv[0] * 0.06 - t_hat[0] * 0.04, w[1] + nv[1] * 0.06 - t_hat[1] * 0.04, PUSH_Z)
                    high_travel(arm, tgt, down[arm], 0.0, a)
                    if (ph_t >= SEC(2.0) and (ee_pose(arm)[0] - tgt).norm().item() < 0.01) or ph_t >= SEC(8.0):
                        enter("e_down")
                elif phase == "e_down":  # settle just outside the face, trailing the contact point
                    tgt = V3(w[0] + nv[0] * 0.02 - t_hat[0] * 0.03, w[1] + nv[1] * 0.02 - t_hat[1] * 0.03, PUSH_Z)
                    servo(arm, tgt, down[arm], 0.0, a)
                    if ph_t >= SEC(1.2):
                        enter("e_push")
                elif phase == "e_push" and st.get("push_mode") == "recenter":
                    tpc2 = table_pq()[0][:2]  # linear push: walk the slab back toward the spawn centre
                    offc = C0 - tpc2
                    if "rc_best" not in st:
                        st["rc_best"] = float(offc.norm())
                    st["rc_best"] = min(st["rc_best"], float(offc.norm()))
                    if float(offc.norm()) > st["rc_best"] + 0.03:  # making it WORSE: bail
                        print(f"[rotate] recenter WORSENING ({float(offc.norm()) * 1e3:.0f}mm) — abort", flush=True)
                        st.pop("rc_best", None)
                        enter("e_out")
                    elif float(offc.norm()) < 0.04 or ph_t >= SEC(8.0):
                        print(f"[rotate] recenter end: slab off by {float(offc.norm()) * 1e3:.0f}mm", flush=True)
                        st.pop("rc_best", None)
                        enter("e_out")
                    else:
                        dh = offc / offc.norm()
                        tgt = V3(w[0] + dh[0] * 0.025, w[1] + dh[1] * 0.025, PUSH_Z)
                        servo(arm, tgt, down[arm], 0.0, a)
                elif phase == "e_push":  # shove tangentially, biased slightly into the face
                    if st.get("braking"):
                        # BRAKE: freeze the cage as a wall so the coasting slab bumps to a stop
                        # (at 1.0kg the slab COASTS past the goal — v31 spun +143deg and slid 859mm)
                        servo(arm, st["brake_tgt"], down[arm], 0.0, a)
                        st["brake_t"] = st.get("brake_t", 0) + 1
                        if st["brake_t"] >= SEC(1.2):
                            print(f"[rotate] braked at {done_yaw:.0f}deg", flush=True)
                            st.pop("braking", None)
                            st.pop("brake_t", None)
                            enter("e_out")
                    else:
                        # SOFT-START + RATE-LIMITED pursuit (the cage drags' recipe — the edge push
                        # applied its full lead instantly and chased the accelerating slab, dumping
                        # stick-slip energy: v31-v33 snapped 50deg per step)
                        th_raw2 = math.atan2(float(rv[1]), float(rv[0]))
                        if "p_th" not in st:
                            st["p_th"], st["p_cmd"] = th_raw2, th_raw2
                        st["p_th"] += (th_raw2 - st["p_th"] + math.pi) % (2 * math.pi) - math.pi
                        st["p_cmd"] += dd_p * 0.12 / CTRL_HZ  # command angle creeps at ~7 deg/s
                        lead_deg = 6.0 if st.get("e_shaft") else args.drag_lead_deg
                        lead_r = math.radians(lead_deg) * min(1.0, ph_t / SEC(2.0))
                        th_led = st["p_th"] + dd_p * lead_r
                        th_c = min(th_led, st["p_cmd"]) if dd_p > 0 else max(th_led, st["p_cmd"])
                        pxy2 = cxy_l + float(rv.norm()) * torch.tensor([math.cos(th_c), math.sin(th_c)], device=device)
                        tgt = V3(pxy2[0] - nv[0] * 0.004, pxy2[1] - nv[1] * 0.004, PUSH_Z)
                        servo(arm, tgt, down[arm], 0.0, a)
                        if (done_yaw - st["d_goal"]) * dd_p >= -8.0:
                            st["braking"] = True
                            st["brake_tgt"] = V3(w[0].item(), w[1].item(), PUSH_Z)
                            st.pop("p_th", None)
                            st.pop("p_cmd", None)
                        if float((C0 - cxy_l).norm()) > 0.30:  # drifting toward an edge: bail
                            st["braking"] = True
                            st["brake_tgt"] = V3(w[0].item(), w[1].item(), PUSH_Z)
                        if ph_t >= SEC(14.0) and (done_yaw - st["d_start_yaw"]) * dd_p < 2.0:
                            # no progress: not contacting / slipping — bail and blacklist
                            bl_key = (arm, "push" + st["e_shaft"]) if st.get("e_shaft") else (arm, idx)
                            st.setdefault("blacklist", set()).add(bl_key)
                            print(f"[rotate] push NO-PROGRESS — abort {bl_key}", flush=True)
                            st.pop("p_th", None)
                            st.pop("p_cmd", None)
                            enter("e_out")
                    if (done_yaw - st["d_start_yaw"]) * dd_p < -3.0:  # rollback: hooked — bail now
                        bl_key = (arm, "push" + st["e_shaft"]) if st.get("e_shaft") else (arm, idx)
                        st.setdefault("blacklist", set()).add(bl_key)
                        print(f"[rotate] push ROLLBACK to {done_yaw:.0f}deg — abort + blacklist {bl_key}", flush=True)
                        enter("e_out")
                    elif done_yaw >= st["d_goal"] + 30.0 or (ph_t >= SEC(25.0) and not st.get("braking")):
                        gained = done_yaw - st["d_start_yaw"]
                        if gained < 3.0:
                            st.setdefault("blacklist", set()).add((arm, idx))
                            print(f"[rotate] push end: {done_yaw:.0f}deg (gained {gained:.0f} — blacklist {arm}/pt{idx})", flush=True)
                        else:
                            st.setdefault("blacklist", set()).clear()
                            print(f"[rotate] push end: {done_yaw:.0f}deg (goal {st['d_goal']:.0f})", flush=True)
                        enter("e_out")
                else:  # e_out: retreat outward then up
                    tgt = V3(w[0] + nv[0] * 0.08, w[1] + nv[1] * 0.08, PUSH_Z + 0.12)
                    servo(arm, tgt, down[arm], 0.0, a)
                    if ph_t >= SEC(1.5):
                        enter("park")
            elif phase in ("d_hover", "d_in", "d_drag", "d_out"):
                arm, name, topdown = st["d_arm"], st["d_name"], st["d_top"]
                oarm = "left" if arm == "right" else "right"
                if st.get("d_mode") == "anchored":
                    anchor_hold(a)  # the off-arm IS the anchor cage
                elif st.get("d_mode") != "recenter":
                    center_pin(oarm, a)  # EVERY drag gets a pin (v76: an unpinned left drag
                    # stick-slip-flung the slab 645mm)
                else:
                    # pin UP during recenter — v38's "the pull can't move the slab" was the left
                    # arm pressing the slab down (~3N) while the right tried to SLIDE it.
                    # B6: at >= 3 welds the off-arm ASSIST-PUSHES a welded shaft toward the same
                    # target instead (RC-COUPLE) — pull alone is under the 3-weld breakaway.
                    if int(scene.welded[0].sum()) >= 3:
                        tp_as = table_pq()[0][:2]
                        off_as = st.get("rc_tgt_v", C0) - tp_as
                        dh_as = off_as / (off_as.norm() + 1e-9)
                        assist_push(oarm, a, dh_as, handle_xy(name))
                    else:
                        park(oarm, a, 0.0)
                # BLENDED command centre: the live circle shifted toward the spawn centre by AT MOST
                # the cage slack (~6mm). Pure-live follows the drag drift (v10: slab walked 25cm off
                # the bench); any radial demand beyond the slack JAMS the drag outright (v11 at 35cm,
                # v12 at 3cm both gained ~0). The correction must ride inside the slack, continuously.
                live_ctr = table_pq()[0][:2]
                if st.get("d_mode") == "anchored":
                    cxy = handle_xy(st["a_name"])  # arc about the anchored leg, live
                else:
                    corr = C0 - live_ctr
                    cn = corr.norm().item()
                    if cn > 0.006:
                        corr = corr * (0.006 / cn)
                    cxy = live_ctr + corr
                hxy = handle_xy(name)
                rvec = hxy - cxy
                rlen = rvec.norm()
                rhat = rvec / rlen
                if name.startswith("leg"):  # welded leg: pad band at the shaft's top ~40mm
                    gz = leg_pq(int(name[3]))[0][2].item() + SHAFT_TOP - 0.025
                    cage_w = 0.026  # top band only: a top-down cage cannot reach mid-shaft
                    # (v49: the palm parks on the shaft top, pad_err 118mm)
                elif name.startswith("cor"):  # slab corner: pad band on the upper side faces
                    gz = slab_top - 0.010
                    cage_w = 0.008
                else:  # bare stud (legacy; studs pop out — kept for completeness)
                    gz = slab_top + 0.112
                    cage_w = 0.014
                if st.get("d_mode") == "recenter" and float((st.get("rc_tgt_v", C0) - live_ctr).norm()) > 1e-6:
                    # radial_cage_quat(v -> handle) puts the PAD FACES PERPENDICULAR to v (rot
                    # drags: v radial, pads press tangential). To make the pads face the pull,
                    # pass the pull's PERPENDICULAR — v38/v48-v51 passed the pull itself, so the
                    # pads faced sideways and the shaft rolled out along them (pad->handle 39mm/s)
                    tgt_rc = st.get("rc_tgt_v", C0)
                    dh0 = (tgt_rc - live_ctr) / (tgt_rc - live_ctr).norm()
                    perp0 = torch.tensor([-dh0[1], dh0[0]], device=device)
                    qg = radial_cage_quat(hxy - perp0, hxy, True, arm)
                else:
                    qg = radial_cage_quat(cxy, hxy, True, arm)
                if st.get("d_flip"):
                    # 180deg jaw flip = the OTHER wrist-fold branch (the pick phases' psi_flip;
                    # v85/v86: feasibility is bimodal across the flip). Set by the one-shot
                    # anchored-hover retry below.
                    qg = yaw_about_z(qg, math.pi)
                if phase == "d_hover":
                    tgt = V3(hxy[0], hxy[1], gz + 0.10)
                    hand_goal = tgt - quat_apply(qg.unsqueeze(0), PAD.unsqueeze(0))[0]
                    high_travel(arm, hand_goal, qg, OPEN, a)
                    perr = ((ee_pose(arm)[0] + quat_apply(ee_pose(arm)[1].unsqueeze(0), PAD.unsqueeze(0))[0]) - tgt).norm().item()
                    anc_ready = st.get("d_mode") != "anchored" or st.get("anc_state") == "hold"
                    if st.get("d_mode") == "anchored" and not anc_ready and \
                            ph_t >= SEC(28.0 if int(scene.welded[0].sum()) >= 3 else 20.0):
                        # 20 -> 28s (B5): room for the anchor approach's 4-step yaw ladder
                        # (>= 3 welds only — keep the proven 20s early-game timing)
                        # the ANCHOR approach never landed: ban the anchor pair, retreat.
                        # DISTINCT key ("anc", ...): plain (arm, name) bans are drag-pose bans,
                        # which the anchor selector deliberately ignores — writing the failure
                        # there was invisible and B1 looped the same failing anchor for 900s.
                        print(f"[rotate] anchor failed — abort anchored ({st['a_arm']}/{st['a_name']})", flush=True)
                        app_bl.add(("anc", st["a_arm"], st["a_name"]))
                        st.pop("d_th", None); st.pop("d_flip", None)
                        enter("d_out")
                    elif ph_t >= SEC(8.0) and perr > 0.10 and st.get("d_mode") != "recenter":
                        # hover never converged: the pose is infeasible and the arm is flailing
                        # mid-workspace (v79: it swept the assembly and threw the slab 742mm
                        # BEFORE the d_in check could fire) — ban and retreat immediately
                        if st.get("d_mode") == "anchored" and not st.get("d_flip"):
                            # B4: try the OTHER wrist-fold branch first — the hover is contact-
                            # free, so a second high approach is safe, and the flip is the same
                            # trick that rescues bimodal picks
                            print(f"[rotate] hover failed ({perr * 1e3:.0f}mm) — JAW-FLIP retry "
                                  f"({arm}/{name})", flush=True)
                            st["d_flip"] = True
                            enter("d_hover")
                        else:
                            print(f"[rotate] hover failed ({perr * 1e3:.0f}mm) — abort drag ({arm}/{name})", flush=True)
                            app_bl.add((arm, name))
                            st.pop("d_th", None); st.pop("d_flip", None)
                            st.pop("d_flip", None)
                            enter("d_out")
                    elif ((ph_t >= SEC(2.5) and perr < 0.01) or ph_t >= SEC(8.0)) and anc_ready:
                        enter("d_in")
                elif phase == "d_in":
                    tgt = V3(hxy[0], hxy[1], gz)
                    servo_pad(arm, tgt, qg, OPEN if ph_t < SEC(0.8) else cage_w, a)
                    if ph_t >= SEC(1.4):
                        p_now, q_now = ee_pose(arm)
                        pad_now = p_now + quat_apply(q_now.unsqueeze(0), PAD.unsqueeze(0))[0]
                        qe = quat_mul(qg.unsqueeze(0), quat_conjugate(q_now.unsqueeze(0)))
                        perr_in = (pad_now - tgt).norm().item()
                        print(f"[rotate]   in: pad_err {perr_in * 1e3:.0f}mm "
                              f"ori_err {math.degrees(float(axis_angle_from_quat(qe)[0].norm())):.0f}deg", flush=True)
                        if perr_in > 0.06 and st.get("d_mode") != "recenter":
                            # NEVER drag with an uncoupled cage: v77's flailing right arm (pose
                            # unreachable, pad 550mm out) caught the assembly mid-servo and
                            # whipped the slab +117deg. Ban the pair (PERSISTENT) and retreat.
                            print(f"[rotate] approach failed — abort drag ({arm}/{name})", flush=True)
                            app_bl.add((arm, name))
                            st.pop("d_th", None); st.pop("d_flip", None)
                            enter("d_out")
                        else:
                            enter("d_drag")
                elif phase == "d_drag" and st.get("d_mode") == "recenter":
                    tpc2 = table_pq()[0][:2]
                    offc = st.get("rc_tgt_v", C0) - tpc2
                    if "rc_best" not in st:
                        st["rc_best"] = st["rc_start"] = float(offc.norm())
                    st["rc_best"] = min(st["rc_best"], float(offc.norm()))
                    # also end when the pull WALKS the handle out of the arm's reach annulus
                    # (pulling toward C0 can move the handle AWAY from the base) — re-select fresh.
                    # B8: RELATIVE, not absolute — the absolute 0.60 killed the left/leg1 pull at
                    # birth (leg1 starts at 0.664 yet the left arm coupled to it at pad_err 0mm;
                    # it was the ONLY converging pull at the -157 corner and never got to run)
                    d_hf = float((handle_xy(name) - BASES[arm]).norm())
                    if "hf0" not in st:
                        st["hf0"] = d_hf
                    handle_far = d_hf > max(0.70, st["hf0"] + 0.03)
                    # ESCAPE detector: the slab's own yaw during the pull swings the shaft out of
                    # the jaws' open sides (~50mm/pull ceiling in v52) — end instantly, re-grab
                    p_esc, q_esc = ee_pose(arm)
                    pad_esc = p_esc + quat_apply(q_esc.unsqueeze(0), PAD.unsqueeze(0))[0]
                    ph_now = float((pad_esc[:2] - hxy).norm())
                    # coupling is judged in 3D vs the grip point — v66's right arm hovered 455mm
                    # ABOVE the handle with a clean XY reading (20mm) and retried 14x
                    ph3_now = float((pad_esc - V3(hxy[0], hxy[1], gz)).norm())
                    st["rc_min_ph"] = min(st.get("rc_min_ph", 9.0), ph3_now)
                    escaped = ph_now > 0.035
                    if float(offc.norm()) < 0.05 or ph_t >= SEC(20.0) or float(offc.norm()) > st["rc_best"] + 0.04 or handle_far or escaped:
                        mp3 = st.pop("rc_min_ph", 9.0)
                        if mp3 > 0.06:
                            # this ARM's approach NEVER coupled to this handle (blocked path /
                            # workspace edge) — ban the (arm, handle) PAIR so the other arm gets it
                            st.setdefault("rc_bl", set()).add((arm, name))
                        if abs(st["rc_start"] - float(offc.norm())) < 0.03 and not escaped and mp3 <= 0.06:
                            # truly IMMOBILE: ban THIS pair, not the whole mechanism — verify9's
                            # corner had TWO in-range pairs (leg0-L 0.54, leg1-R 0.57) and the
                            # global latch refused both after one weak reach-edge pull. The rc_n
                            # budget still bounds futile cycling.
                            st.setdefault("rc_bl", set()).add((arm, name))
                        elif st["rc_start"] - float(offc.norm()) >= 0.03:
                            # the slab MOVED: dead drag pairs may have re-entered their zones
                            st.setdefault("blacklist", set()).clear()
                            st.pop("rc_bl", None)
                        print(f"[rotate] cage-recenter end: slab off by {float(offc.norm()) * 1e3:.0f}mm", flush=True)
                        st.pop("rc_best", None)
                        st.pop("rc_start", None)
                        st.pop("hf0", None)
                        enter("d_out")
                    else:
                        dh = offc / offc.norm()
                        # 5cm lead at kp350 ~ 17N: the 2cm/kp250 pull (~5N) sat below the
                        # slab+two-legs breakaway friction and gained 1-2mm (v47/v48)
                        tgt2 = V3(hxy[0] + dh[0] * 0.05, hxy[1] + dh[1] * 0.05, gz)
                        # keep the PULL-oriented jaws through the pull — v47 recomputed the
                        # tangential quat here, so the wall rotated away mid-drag and the pull
                        # gained 2mm/241 (the same "pulled air" as v38, one stage later)
                        servo_pad(arm, tgt2, radial_cage_quat(hxy - dh, hxy, True, arm), cage_w, a)
                        if step_i % 120 == 0:  # WHO fails: the arm (pad short of tgt) or the cage (pad at tgt, slab still)?
                            p_dbg, q_dbg = ee_pose(arm)
                            pad_dbg = p_dbg + quat_apply(q_dbg.unsqueeze(0), PAD.unsqueeze(0))[0]
                            print(f"[rc] off={float(offc.norm()) * 1e3:.0f}mm pad->tgt="
                                  f"{float((pad_dbg - tgt2).norm()) * 1e3:.0f}mm "
                                  f"pad->handle={float((pad_dbg[:2] - hxy).norm()) * 1e3:.0f}mm", flush=True)
                elif phase == "d_drag" and st.get("d_mode") == "anchored" and st.get("anc_state") != "hold":
                    # hold caged until the ANCHOR is locked
                    servo_pad(arm, V3(hxy[0], hxy[1], gz), qg, cage_w, a)
                elif phase == "d_drag" and st.get("d_mode") == "rot" and st.get("pin_touch") is None:
                    # hold caged until the centre pin is down — dragging an unpinned slab drifts it
                    servo_pad(arm, V3(hxy[0], hxy[1], gz), qg, cage_w, a)
                elif phase == "d_drag":
                    th_raw = math.atan2(float(rvec[1]), float(rvec[0]))
                    if "d_th" not in st:
                        st["d_th"], st["d_cmd"] = th_raw, th_raw
                    st["d_th"] += (th_raw - st["d_th"] + math.pi) % (2 * math.pi) - math.pi
                    rate = args.drag_rate if name.startswith("leg") else 0.6 * args.drag_rate
                    lead = 6.0  # the cage drag's proven lead (v6-v13); the global arg was lowered
                    # to 3.0 for the (retired) edge pushes and starved the drag in v36
                    dd_ = st.get("d_dir", 1.0)
                    if "d_relax" not in st:
                        st["d_cmd"] += dd_ * rate / CTRL_HZ
                        th_led = st["d_th"] + dd_ * math.radians(lead)
                        th = min(th_led, st["d_cmd"]) if dd_ > 0 else max(th_led, st["d_cmd"])
                    else:
                        # B9 RELAX: hold at the LIVE handle angle (zero lead) so the cage spring
                        # bleeds against the closed jaws before they open — B8's raw release
                        # kicked the 3-weld slab 16cm/1s and the follow-up whipped it off the bench
                        th = st["d_th"]
                    pxy = cxy + rlen * torch.tensor([math.cos(th), math.sin(th)], device=device)
                    servo_pad(arm, V3(pxy[0], pxy[1], gz), radial_cage_quat(cxy, pxy, True, arm), cage_w, a)
                    # mid-drag edge guard: end the drag the moment the slab strays too far — the
                    # park loop recenters before the next arc (v55 fell exactly here)
                    drifted = float((C0 - live_ctr).norm()) > 0.26
                    end_c = (done_yaw - st["d_goal"]) * dd_ >= 0 or ph_t >= SEC(50.0) or drifted
                    if end_c and "d_relax" not in st:
                        # relaxed release only at >= 3 welds (its reason to exist); below that it
                        # changed the proven early-game stop dynamics (B18/B19 task-2 regression)
                        st["d_relax"] = 0 if int(scene.welded[0].sum()) >= 3 else SEC(0.8)
                    if "d_relax" in st:
                        st["d_relax"] += 1
                    if st.get("d_relax", 0) >= SEC(0.8):
                        st.pop("d_relax", None)
                        gained = (done_yaw - st["d_start_yaw"]) * dd_
                        if gained < 5.0:  # this (arm, handle) pair delivers nothing: stop retrying it
                            st.setdefault("blacklist", set()).add((arm, name))
                            print(f"[rotate] drag end: {done_yaw:.0f}deg (gained {gained:.0f} — blacklist {arm}/{name})", flush=True)
                        else:  # geometry changed: previously-dead pairs may work again
                            st.setdefault("blacklist", set()).clear()
                            app_bl.clear()  # bearings moved: banned approaches may be feasible now
                            st["rc_n"] = 0  # real rotation progress: refresh the recenter budget
                            # (the cap guards against futile loops, not against long missions —
                            # v56: pulls back-rotate ~15deg/round, so ~+11deg net/cycle needs many rounds)
                            print(f"[rotate] drag end: {done_yaw:.0f}deg (goal {st['d_goal']:.0f})", flush=True)
                        st.pop("d_th", None); st.pop("d_flip", None)
                        enter("d_out")
                else:  # d_out — FROZEN target: live-tracking the handle during the retreat let the
                    # open cage CHASE the moving shaft in a feedback spiral (v43: the slab spun
                    # 3 revolutions at ~300deg/s and flew off the bench)
                    if "out_tgt" not in st:
                        # rise STRAIGHT UP from the hand's own position: freezing at the HANDLE xy
                        # made the (leading) hand swipe back through the shaft (+60deg kick, v44)
                        p_now, q_now = ee_pose(arm)
                        pad_now = p_now + quat_apply(q_now.unsqueeze(0), PAD.unsqueeze(0))[0]
                        st["out_tgt"] = (V3(pad_now[0].item(), pad_now[1].item(), gz + 0.15), q_now.clone())
                    servo_pad(arm, st["out_tgt"][0], st["out_tgt"][1], OPEN, a)
                    if ph_t >= SEC(1.5):
                        st.pop("out_tgt", None)
                        enter("park")

        env.step(a, render=render)

        if step_i % LOG_EVERY == 0:
            tag = f"T{plan_i}/{task['kind']}/{phase}"
            extra = ""
            if task["kind"] == "leg":
                dz, lat = leg_state(task["leg"], task["slot"])
                extra = (f"dz={dz * 1e3:6.1f} lat={lat * 1e3:5.1f} wound={math.degrees(st.get('wound', 0)):+6.0f} "
                         f"stroke={st.get('stroke', 0)}")
            tpc = table_pq()[0]
            print(f"  {step_i:6d} [{tag:>24s}] slabYaw={yaw_acc['slab']:+7.1f} "
                  f"ctr=({tpc[0].item():+.2f},{tpc[1].item():+.2f}) "
                  f"welds={scene.welded[0].int().tolist()} {extra}", flush=True)
            if tpc[2].item() < slab_top - 0.15:  # unrecoverable: don't servo a fallen slab for minutes
                print("[abort] SLAB FELL OFF THE BENCH — ending run", flush=True)
                break

    seated = scene.seated()[0].int().tolist()
    welds = scene.welded[0].int().tolist()
    print(f"SOLVE_FOUR[fable] | seated={seated} welded={welds} n={sum(welds)}/4 "
          f"| slabYaw={yaw_acc['slab']:.0f}deg | results={results}", flush=True)
    threading.Timer(10.0, lambda: os._exit(0)).start()
    env.close()


if __name__ == "__main__":
    main()
    app.close()
    os._exit(0)

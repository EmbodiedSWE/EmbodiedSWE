"""solve — Fable's scripted solver for assembly.nut_thread.franka.osc at the NATURAL layout (no sink).

SOLVED 2026-07-18 with the default args below: `SOLVE[fable] | seated=1 | dz=5.0mm lat=0.2mm |
strokes=34` — base at the tabletop plane (`surface_z=0`, like a real bench robot), no platform, no
riser. Lineage: the original 2026-07-01 solve ran at `surface_z=0.20` (base buried 20 cm below the
work; preserved as `solve_sink.py`); an Opus session forked it arg-driven for the flat attempt; the
2026-07-17/18 Fable session added the five fixes that made flat work (BUILD_LOG "FLAT-layout
campaign"): wrist-roll re-centred via `down_yaw=-0.5`, bolt at radius 0.50 (elbow mid-range),
persistent-target xy-servo integrators for pick/carry/thread (`--nut_center`, default ON — the flat
posture has a 2-16mm run-variable task-space bias), seek skipped when the touch-settle already
proves engagement, and a width-gated 8mm jaw lift for the rewind swing with a measured re-descend.
Layout stays arg-driven; `--reach_only` runs the pick/place approach and reports reach (grasp width,
threading-pose joint angles, OSC tracking error) then exits, so layout iteration is fast.

Approach phases keep the env truths: forward nullspace, OSC rot-gain, HAND_OFFSET=0.058, nut_friction
=0.4, dt=1/480 (the scene's 1/120 tunnels a pressed M16). The THREADING strategy is unchanged:

  Instead of a rigid full-force grip (which welds the nut level so it can't settle onto the helix
  start), the jaws hold a LIGHT PINCH (~3 N/finger, position-PD overshoot): torque still transmits
  geometrically (a wrench), while the nut keeps friction-limited vertical/tilt compliance — the same
  micro-freedom that let the screw_drive socket cup thread its bolt. The press is a small z-lean whose
  transmitted force saturates at pad friction (~3 N), so it cannot cross-thread-jam like a rigid press.
  Strokes are 120 deg (hex symmetry: after each rewind the jaws re-land flat-on-flat), rewinds happen
  with the jaws opened past the hex corners, and the hand z closed-loop follows the nut's real height.

Run from the repo root:

    .venv/bin/python experiments/nut_thread_franka_osc_fable/solve.py --headless
    .venv/bin/python experiments/nut_thread_franka_osc_fable/solve.py --livestream 2
"""

from __future__ import annotations

import argparse
import math
import os
import threading

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--max_sec", type=float, default=300.0, help="sim-time budget (s; the r9 flat solve reached the 5mm stop at ~291)")
parser.add_argument("--pinch_n", type=float, default=3.0, help="SOFT pinch per finger for engagement/seek (N) — keeps settle compliance")
parser.add_argument("--pinch_wind_n", type=float, default=25.0, help="FIRM pinch per finger for wind strokes (N) — run 1: 3 N cams over the hex corners once the thread is engaged (needs ~0.15 N*m)")
parser.add_argument("--lean", type=float, default=0.003, help="z press-lean below the follow height while winding (m)")
parser.add_argument("--wind_rate", type=float, default=2.0, help="tighten rate (rad/s)")
parser.add_argument("--rewind_rate", type=float, default=3.0, help="unwind rate (rad/s)")
parser.add_argument("--torque_dt", type=float, default=-1.0, help="OSC target-latch dt; -1 = latch every physics step (max servo authority), 0 = robot default 1/15")
parser.add_argument("--sweep_deg", type=float, default=120.0, help="stroke sweep (deg; 120 = hex-symmetric)")
parser.add_argument("--seek_deg", type=float, default=50.0, help="max back-rotation seeking the thread start (deg; capped so a fruitless seek can't eat the wrist-roll range — and skipped entirely when the touch-settle already dropped the nut into the thread)")
parser.add_argument("--stop_dz", type=float, default=0.005, help="stop threading once nut-above-bolt <= this (m; bottom-out is ~0.001, pads graze the bolt head below ~0.004)")
parser.add_argument("--dt", type=float, default=0.00208333, help="sim dt (probe v1: the scene's 1/120 TUNNELS on a pressed M16; 1/480 per screw_drive)")
# --- natural-layout knobs (world = workbench_pos (0.5,0) + slot; base at origin) ------------------
parser.add_argument("--surface_z", type=float, default=0.0, help="table-top height (m); 0.0 = lab_table natural (no sink). Fable used 0.20.")
parser.add_argument("--base_z", type=float, default=0.0, help="Franka base height (m); >0 mounts the arm on a riser above the table (unfolds the elbow for down-press authority) WITHOUT sinking the table")
parser.add_argument("--down_yaw", type=float, default=-0.5, help="gripper-down yaw (rad); centres the wrist-roll DOF (q7) so the wind strokes have range (the sink-era 1.83 parks q7 0.6 rad from its limit at the flat posture -> strokes bind at a fixed angle)")
parser.add_argument("--nut_center", action=argparse.BooleanOptionalAction, default=True, help="persistent-target xy-servo integrators: centre the jaws on the nut for the pick and the carried nut on the bolt axis for the place/thread. REQUIRED at the flat layout (2-16mm posture bias); --no-nut_center replicates the raw sink-era servo")
parser.add_argument("--bolt_slot_x", type=float, default=0.0, help="bolt slot x (table-rel.); world x = 0.5 + this. 0.0 -> radius 0.50, elbow mid-range (0.37 folds it to its limit — zero stroke authority)")
parser.add_argument("--nut_x", type=float, default=-0.12, help="nut spawn x (table-rel.); world x = 0.5 + this. -0.12 -> 0.38 m ahead")
parser.add_argument("--nut_y", type=float, default=0.0, help="nut spawn y (table-rel.)")
parser.add_argument("--reach_only", action="store_true", help="run the pick/place approach, report reach at threading entry, then exit (fast layout probe)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
livestream_on = args.livestream > 0

app = AppLauncher(args).app

import torch  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    axis_angle_from_quat,
    quat_apply,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
)

import robobench  # noqa: E402
from robobench.core import EnvCfg  # noqa: E402
from robobench.robots.franka import FrankaRobotCfg  # noqa: E402
from robobench.suites.assembly.scenes import NutThreadAssemblySceneCfg  # noqa: E402

POS_SCALE, ROT_SCALE = 0.02, 0.097  # controller action scaling (servo divides by these)
OPEN, CLOSE = 0.04, 0.0             # finger position targets (m)
HAND_OFFSET = 0.058                 # panda_hand -> fingertip body, down the approach axis (measured by Opus)
# gripper-down yaw (rad). 1.83 kept joint-1 ~0 at base_z=0; on a riser it drives q7 to its limit, so
# it is retunable via --down_yaw to re-centre the wrist-roll DOF the wind strokes turn.
GRIP_KP = 8000.0                    # gripper stiffness we build the robot with (pinch force = KP * overshoot)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    if args.torque_dt != 0:  # latch the OSC pose target faster than the 15 Hz default (torque law already runs per step)
        from robobench.robots.franka import FrankaRobot

        FrankaRobot.TORQUE_CONTROL_DT = args.dt if args.torque_dt < 0 else args.torque_dt
    scene_cfg = NutThreadAssemblySceneCfg(
        surface_z=args.surface_z,
        bolt_slots=((args.bolt_slot_x, 0.0),),
        nut_init_xy=((args.nut_x, args.nut_y),),
        nut_friction=0.4,
    )
    robot_cfg = FrankaRobotCfg(base_pos=(0.0, 0.0, args.base_z), nullspace_dof_pos=(), gripper_stiffness=GRIP_KP)
    env = EnvCfg(
        scene="nut_thread", scene_cfg=scene_cfg, robot="franka", robot_cfg=robot_cfg, control_mode="osc", env_spacing=2,
        sim_overrides={"dt": args.dt},
    ).build(num_envs=1, device=device)
    scene, robot = env.scene, env.robot
    osc = robot.controller.controllers[0]
    # Rot authority: with per-step latching the pose error is capped at rot_scale, so usable torque ~
    # kp_rot * rot_scale. Run 1 stalled turning the engaged nut -> raise both (250*0.097 -> 600*0.15).
    osc._kp = torch.tensor([150.0, 150.0, 150.0, 600.0, 600.0, 600.0], device=device)
    osc._kd = 2.0 * osc._kp.sqrt()
    osc.cfg.rot_scale = 0.15
    art = robot.articulation
    ee_idx = art.body_names.index("panda_hand")
    lf_idx = art.body_names.index("panda_leftfinger")
    rf_idx = art.body_names.index("panda_rightfinger")
    fj1 = art.find_joints(["panda_finger_joint1"])[0]
    render = (not args.headless) or livestream_on
    down = quat_from_euler_xyz(*(torch.tensor([v], device=device) for v in (3.14159, 0.0, args.down_yaw)))[0]
    n, dim = env.num_envs, robot.action_dim
    DECIM = robot.control_period
    CTRL_HZ = 1.0 / (env.dt * DECIM)
    print(f"[cfg] control_period={DECIM} -> control {CTRL_HZ:.1f} Hz | pinch={args.pinch_n}N "
          f"lean={args.lean * 1e3:.0f}mm wind={args.wind_rate}rad/s sweep={args.sweep_deg}deg", flush=True)
    env.reset()

    def ee_pose():
        return art.data.body_pos_w[0, ee_idx], art.data.body_quat_w[0, ee_idx]

    def servo(goal_pos, goal_quat, grip):
        p, q = ee_pose()
        a = torch.zeros(1, dim, device=device)
        a[0, 0:3] = ((goal_pos - p) / osc.cfg.pos_scale).clamp(-1.0, 1.0)  # cfg, not the module constants:
        qe = quat_mul(goal_quat.unsqueeze(0), quat_conjugate(q.unsqueeze(0)))
        a[0, 3:6] = (axis_angle_from_quat(qe)[0] / osc.cfg.rot_scale).clamp(-1.0, 1.0)  # rot_scale is patched above
        a[0, 6:8] = grip
        return a

    def nut():
        return scene.nuts[0].data.root_pos_w[0]

    def grasp_center():  # xy midpoint of the two fingertip bodies — the true jaw axis (the hand-origin
        return 0.5 * (art.data.body_pos_w[0, lf_idx] + art.data.body_pos_w[0, rf_idx])  # xy lies once tilted)

    def bolt():
        return scene.bolts[0].data.root_pos_w[0]

    def yaw_of(q):  # world yaw of a frame's x-axis (deg)
        ex = quat_apply(q.unsqueeze(0), torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
        return math.degrees(math.atan2(ex[1].item(), ex[0].item()))

    def yaw_about_z(qbase, ang):
        yz = quat_from_euler_xyz(*(torch.tensor([v], device=device) for v in (0.0, 0.0, ang)))
        return quat_mul(yz, qbase.unsqueeze(0))[0]

    Z = lambda z: torch.tensor([0.0, 0.0, z], device=device)

    # ---- phase schedule (seconds -> control steps, rate-independent) ----------------------------
    SEC = lambda s: max(1, round(s * CTRL_HZ))
    # hover/descend/lift/over lengthened (Fable 1.7/1.5/1.7/1.3) for the flat-table layout: from the
    # home pose the pick/place poses are a bigger posture swing at table level (mirrors the bulb re-solve).
    T_SETTLE, T_HOVER, T_DESCEND, T_GRASP, T_LIFT, T_OVER = (SEC(s) for s in (0.5, 3.0, 3.0, 1.0, 2.5, 1.8))
    T_PINCH, T_OPEN, T_RECLOSE, T_QUIET = SEC(0.5), SEC(0.4), SEC(0.5), SEC(0.5)
    LOWER_MPS = 0.015  # touch-detect descent rate (m/s)
    SWEEP, SEEK_MAX = math.radians(args.sweep_deg), math.radians(args.seek_deg)
    D_WIND, D_REWIND = args.wind_rate / CTRL_HZ, args.rewind_rate / CTRL_HZ

    # run state --------------------------------------------------------------------------------
    phase, ph_t = "settle", 0            # top-level phase + steps spent in it
    st = {                               # measured calibration + threading state
        "p_flat": None,                  # finger joint pos gripping the flats (m)
        "grip_off": None,                # hand_z - nut_z at grasp (follow offset)
        "z_contact": None,               # hand z when the nut touched down on the bolt
        "lower_z": None,                 # descending target during touch-detect lower
        "nut_hist": [],                  # recent nut z (touch detect)
        "t_state": "pinch", "wound": 0.0, "t": 0, "dz_seek0": None, "stroke": 0,
        "caught": False,
        "ee0": None,                     # unwrapped EE yaw (deg) at wind-stroke start, for lag pacing
        "stall_t": 0,                    # steps spent waiting for the EE to catch up at stroke end
        "w_start": 0.0,                  # wound at wind-stroke start (strokes sweep w_start - SWEEP)
        "w_land": 0.0,                   # rewind landing angle (hex-aligned to the nut, not always 0)
        "align": None,                   # (wound_deg - nut_yaw_deg) mod 60 measured while flat-on-flat
        "flat_ok": False,                # wind-entry gate: jaws confirmed flat-on-flat before clamping firm
        "stall_run": 0,                  # consecutive steps at max pacing lag (bind escape)
        "xy_tgt": None,                  # nut-centring integrator state: the PERSISTENT xy servo target.
        "xy_tgt_g": None,                # same integrator for the PICK (hover/descend/grasp), jaw-axis obs.
        "dz_touch": None,                # dz at the touch event; the settle drop below it = pre-engagement
        "rc_at_band": False,             # reclose descend-around done: pads measured back at grip height
        # r2 bug: integrating off the MEASURED hand pos (tgt = p + corr) re-injects the posture bias
        # every step; with |corr| capped below the ~4mm bias the loop runs AWAY from the bolt (landed
        # 13mm off). A true integrator accumulates on its own target and only rests when nut == axis.
    }
    # unwrapped yaw trackers (logging/coupling)
    yaw_acc = {"ee": 0.0, "nut": 0.0, "ee_prev": None, "nut_prev": None}

    def unwrap(key, val):
        prev = yaw_acc[key + "_prev"]
        if prev is not None:
            d = (val - prev + 180.0) % 360.0 - 180.0
            yaw_acc[key] += d
        yaw_acc[key + "_prev"] = val
        return yaw_acc[key]

    def pinch_target(newtons):
        return st["p_flat"] - newtons / GRIP_KP  # position overshoot -> ~N per finger

    def wide_target():
        return st["p_flat"] + 0.005  # past the hex corner protrusion (~1.9mm) + margin

    LOG_EVERY = SEC(0.8)

    for i in range(1, SEC(args.max_sec) + 1):
        nx, bx = nut(), bolt()
        dz = (nx - bx)[2].item()
        p, q = ee_pose()
        ee_u = unwrap("ee", yaw_of(q))  # per-step so wind pacing and logs share one accumulator
        nut_u = unwrap("nut", yaw_of(scene.nuts[0].data.root_quat_w[0]))

        if phase == "settle":
            action = servo(p, down, OPEN)
            if ph_t >= T_SETTLE:
                phase, ph_t = "hover", 0
        elif phase == "hover":
            tgt = nx + Z(HAND_OFFSET + 0.05)
            if args.nut_center:  # pick centering: r4 showed the flat-posture xy bias reaches ~16mm at the
                # extended pick pose — enough to close the jaws BESIDE the nut. Integrate the servo target
                # until the fingertip midpoint sits over the nut (same integrator as the carry, jaw-axis
                # observable). Frozen through the grasp so jaw contact can't excite it.
                if st["xy_tgt_g"] is None:
                    st["xy_tgt_g"] = nx[:2].clone()
                st["xy_tgt_g"] += (0.01 * (nx[:2] - grasp_center()[:2])).clamp(-3e-4, 3e-4)
                st["xy_tgt_g"] = nx[:2] + (st["xy_tgt_g"] - nx[:2]).clamp(-0.03, 0.03)
                tgt[0:2] = st["xy_tgt_g"]
            action = servo(tgt, down, OPEN)
            if ph_t >= T_HOVER:
                phase, ph_t = "descend", 0
        elif phase == "descend":
            tgt = nx + Z(HAND_OFFSET)
            if args.nut_center:
                st["xy_tgt_g"] += (0.01 * (nx[:2] - grasp_center()[:2])).clamp(-3e-4, 3e-4)
                st["xy_tgt_g"] = nx[:2] + (st["xy_tgt_g"] - nx[:2]).clamp(-0.03, 0.03)
                tgt[0:2] = st["xy_tgt_g"]
            action = servo(tgt, down, OPEN)
            if ph_t >= T_DESCEND:
                st["err_descend"] = (p - (nx + Z(HAND_OFFSET)))  # reach: how far the pick pose is unreached
                gc_err = (nx[:2] - grasp_center()[:2]).norm().item()
                print(f"[centre] pick: jaw-axis-to-nut err={gc_err * 1e3:.1f}mm", flush=True)
                phase, ph_t = "grasp", 0
        elif phase == "grasp":
            tgt = nx + Z(HAND_OFFSET)
            if args.nut_center and st["xy_tgt_g"] is not None:
                tgt[0:2] = st["xy_tgt_g"]  # frozen — no integration while the jaws touch the nut
            action = servo(tgt, down, CLOSE)
            if ph_t >= T_GRASP:
                st["p_flat"] = art.data.joint_pos[0, fj1].item()
                st["grip_off"] = p[2].item() - nx[2].item()
                ed = st["err_descend"]
                print(f"[cal] across-flats grip: finger={st['p_flat'] * 1e3:.2f}mm (width {2 * st['p_flat'] * 1e3:.1f}mm) "
                      f"grip_off={st['grip_off'] * 1e3:.1f}mm | descend reach-err "
                      f"({ed[0] * 1e3:+.1f},{ed[1] * 1e3:+.1f},{ed[2] * 1e3:+.1f})mm", flush=True)
                phase, ph_t = "lift", 0
        elif phase == "lift":
            action = servo(bx + Z(HAND_OFFSET + 0.16), down, CLOSE)
            if ph_t >= T_LIFT:
                phase, ph_t = "over", 0
        elif phase == "over":
            tgt = bx + Z(HAND_OFFSET + 0.16)
            if args.nut_center:  # carry centering: the nut is RIGID in the jaws here, so easing the NUT
                # onto the bolt axis cancels the flat-posture hand error before touchdown (r1: raw
                # bolt-xy targets landed the nut 4.1mm off-axis, cocked on the crest -> stroke 0 eject).
                if st["xy_tgt"] is None:
                    st["xy_tgt"] = bx[:2].clone()
                st["xy_tgt"] += (0.01 * (bx[:2] - nx[:2])).clamp(-3e-4, 3e-4)  # gain 4.8/s < OSC bandwidth
                st["xy_tgt"] = bx[:2] + (st["xy_tgt"] - bx[:2]).clamp(-0.025, 0.025)  # anti-windup
                tgt[0:2] = st["xy_tgt"]
            action = servo(tgt, down, CLOSE)
            if ph_t >= T_OVER:
                phase, ph_t = "lower", 0
                st["lower_z"] = p[2].item()
        elif phase == "lower":
            # touch-detect: sink the target ~15 mm/s; contact = nut z quiet while the target keeps going
            st["lower_z"] -= LOWER_MPS / CTRL_HZ
            st["nut_hist"].append(nx[2].item())
            st["nut_hist"] = st["nut_hist"][-T_QUIET:]
            tgt = torch.tensor([bx[0], bx[1], st["lower_z"]], device=device)
            if args.nut_center:  # keep the gripped nut centred on the axis through the touchdown
                st["xy_tgt"] += (0.01 * (bx[:2] - nx[:2])).clamp(-3e-4, 3e-4)
                st["xy_tgt"] = bx[:2] + (st["xy_tgt"] - bx[:2]).clamp(-0.025, 0.025)
                tgt[0:2] = st["xy_tgt"]
            action = servo(tgt, down, CLOSE)
            quiet = len(st["nut_hist"]) == T_QUIET and (max(st["nut_hist"]) - min(st["nut_hist"])) < 2e-4
            if (quiet and st["lower_z"] < p[2].item() - 0.002 and dz < 0.035) or dz < 0.0245:
                st["z_contact"] = p[2].item()
                st["dz_touch"] = dz
                print(f"[touch] nut down on bolt: dz={dz * 1e3:.1f}mm hand_z={p[2].item():.4f} "
                      f"(lower_z target {st['lower_z']:.4f})", flush=True)
                if st["xy_tgt"] is not None:  # learned posture bias (target offset that centres the nut)
                    bias = st["xy_tgt"] - bx[:2]
                    print(f"[centre] learned xy bias=({bias[0] * 1e3:+.1f},{bias[1] * 1e3:+.1f})mm "
                          f"nut_lat={(nx[:2] - bx[:2]).norm() * 1e3:.1f}mm", flush=True)
                # Reach report at the threading pose: joint angles (q7 is the wind DOF; must keep margin
                # to +-2.90 rad through +-SWEEP), and the placement geometry. Cheap layout go/no-go.
                jp = art.data.joint_pos[0].tolist()
                jl = art.data.joint_pos_limits[0]  # (n,2) lower/upper
                arm = jp[:7]
                margins = [min(jp[k] - jl[k, 0].item(), jl[k, 1].item() - jp[k]) for k in range(7)]
                print("[reach] threading-pose arm q(rad)=[" + ", ".join(f"{v:+.2f}" for v in arm) + "]", flush=True)
                print("[reach] joint-limit margins(rad)=[" + ", ".join(f"{v:+.2f}" for v in margins) + "]"
                      f" | min={min(margins):+.2f} | bolt=({bx[0]:.3f},{bx[1]:.3f},{bx[2]:.3f}) "
                      f"nut=({nx[0]:.3f},{nx[1]:.3f})", flush=True)
                if args.reach_only:
                    print(f"REACH_ONLY | grip_width={2 * st['p_flat'] * 1e3:.1f}mm grip_off={st['grip_off'] * 1e3:.1f}mm "
                          f"z_contact={st['z_contact']:.4f} dz={dz * 1e3:.1f}mm q7={arm[6]:+.2f} min_margin={min(margins):+.2f}",
                          flush=True)
                    break
                phase, ph_t = "thread", 0
        elif phase == "thread":
            ts = st["t_state"]
            # Two-stage pinch: SOFT while engaging (the nut needs settle compliance to catch the helix),
            # FIRM while winding (an engaged nut takes ~0.15 N*m; a soft wrench cams over the corners).
            grip = pinch_target(args.pinch_n if ts in ("pinch", "seek") else args.pinch_wind_n)
            lean = 0.0
            if ts == "pinch":        # soften from the carry grip to the wrench pinch
                st["t"] += 1
                if st["t"] >= T_PINCH:
                    drop = (st["dz_touch"] - dz) if st["dz_touch"] is not None else 0.0
                    if drop > 8e-4:
                        # The touch-settle already dropped the nut into the thread/chamfer — seeking
                        # would only unscrew it (r5: the +100deg blind back-seek drove q7 into its
                        # +2.90 limit, backed the nut off a quarter turn, and poisoned stroke 0).
                        print(f"[seek] pre-engaged at touchdown (settle drop {drop * 1e3:.2f}mm) — "
                              f"winding directly", flush=True)
                        st["caught"] = True
                        st["t_state"] = "wind"
                    else:
                        st["t_state"], st["t"] = "seek", 0
                        st["dz_seek0"] = dz
            elif ts == "seek":       # back-rotate under light lean: the start 'clicks' = dz drops
                st["wound"] += D_WIND
                lean = args.lean
                if st["dz_seek0"] - dz > 4e-4:
                    st["caught"] = True
                    print(f"[seek] caught the thread start after {math.degrees(st['wound']):.0f}deg "
                          f"(dropped {(st['dz_seek0'] - dz) * 1e3:.2f}mm)", flush=True)
                    st["t_state"] = "wind"
                elif dz - st["dz_seek0"] > 5e-4:
                    # Fable's seek flaw: if the touch-settle had ALREADY caught, further back-rotation
                    # just UNSCREWS the nut (dz rises) — it then backs out and the first wide open loses
                    # it (platform runs r7/r9: dz 23.9->24.2, wound to +89, then fly-off). Abort on rise
                    # and go straight to wind: the nut is already engaged, drive it DOWN.
                    print(f"[seek] nut rising ({(dz - st['dz_seek0']) * 1e3:.2f}mm) after "
                          f"{math.degrees(st['wound']):.0f}deg — already engaged, winding", flush=True)
                    st["caught"] = True
                    st["t_state"] = "wind"
                elif st["wound"] >= SEEK_MAX:
                    st["t_state"] = "wind"
            elif ts == "wind":       # tighten stroke, PACED by the real EE so commanded sweep = real turn
                if st["ee0"] is None:
                    st["ee0"], st["stall_t"], st["w_start"] = ee_u, 0, st["wound"]
                    st["flat_ok"], st["stall_run"], st["t"] = False, 0, 0
                gpos_now = art.data.joint_pos[0, fj1].item()
                if not st["flat_ok"]:
                    # Flat-contact gate (run 4: clamping 25 N onto the corner ramps BINDS — the wrist
                    # then crawls). Stay soft so the nut cams into the flat basin, then clamp firm.
                    grip, st["t"] = pinch_target(args.pinch_n), st["t"] + 1
                    if gpos_now < st["p_flat"] + 5e-4 or st["t"] >= SEC(1.5):
                        if gpos_now >= st["p_flat"] + 5e-4:
                            print(f"[wind] stroke {st['stroke']}: no flat contact after 1.5s "
                                  f"(grip {gpos_now * 1e3:.2f}mm) — clamping anyway", flush=True)
                        st["flat_ok"] = True
                        st["ee0"] = ee_u  # restart pacing from the true stroke start
                else:
                    swept = math.radians(st["ee0"] - ee_u)  # actual CW turn of the jaws this stroke (rad)
                    lag = st["w_start"] - st["wound"] - swept
                    lean = args.lean
                    st["stall_run"] = st["stall_run"] + 1 if lag >= 0.35 else 0
                    w_end = st["w_start"] - SWEEP
                    if st["stall_run"] >= SEC(2.5):  # bound/binding: end the stroke, let the aligned reclose reset
                        print(f"[wind] stroke {st['stroke']} BOUND (lag {math.degrees(lag):.0f}deg for 2.5s) "
                              f"at wound {math.degrees(st['wound']):.0f}deg — recycling grip", flush=True)
                        st["t_state"], st["t"], st["ee0"] = "open", 0, None
                        st["stroke"] += 1
                    elif st["wound"] > w_end:
                        if lag < 0.35:  # advance the command only while the EE keeps up (run 2: it lagged 66deg)
                            st["wound"] = max(w_end, st["wound"] - D_WIND)
                    else:               # commanded sweep done -> dwell until the EE catches up (or timeout)
                        st["stall_t"] += 1
                        if lag < 0.26 or st["stall_t"] >= SEC(1.5):
                            if lag >= 0.26:
                                print(f"[wind] stroke {st['stroke']} ended {math.degrees(lag):.0f}deg short", flush=True)
                            st["t_state"], st["t"], st["ee0"] = "open", 0, None
                            st["stroke"] += 1
            elif ts == "open":       # release to past-corners width; stop pressing
                grip, st["t"] = wide_target(), st["t"] + 1
                if st["t"] >= T_OPEN:
                    # Land at +45deg: gives the aligning reclose room to search tighten-ward (up to
                    # ~-65deg) while keeping the following stroke's end >= ~-140 (wrist-safe).
                    # (Estimate-based hex alignment failed: under load the wrist tilts and biases the
                    # yaw projection by 15-30deg — runs 5/6. Align by the WIDTH observable instead.)
                    st["w_land"] = math.radians(45.0)
                    st["t_state"] = "rewind"
            elif ts == "rewind":     # unwind the wrist while clear of the hex
                grip = wide_target()
                st["wound"] = min(st["w_land"], st["wound"] + D_REWIND)
                if st["wound"] >= st["w_land"]:
                    st["t_state"], st["t"], st["rc_at_band"] = "reclose", 0, False
            elif ts == "reclose":    # width-feedback aligning reclose: descend around the hex WIDE, then
                st["t"] += 1         # soft-land + rotate tighten-ward until the width reads FLATS, clamp
                gpos_now = art.data.joint_pos[0, fj1].item()
                if not st["rc_at_band"]:
                    # Descend around the hex OPEN until the pads MEASURE back at the grip band (r8: the
                    # timed 0.5s descend lost the race after the 8mm lift — the close began ~3mm high,
                    # the pads landed ON the nut's top face, and every stroke spun air at dz pinned).
                    grip = wide_target()
                    if (p[2].item() - (nx[2].item() + st["grip_off"])) < 1.5e-3 or st["t"] >= SEC(2.0):
                        st["rc_at_band"], st["t"] = True, 0
                elif st["t"] <= T_RECLOSE:              # soft-close + settle
                    grip = pinch_target(args.pinch_n)
                elif gpos_now > st["p_flat"] + 3e-4 and st["t"] < T_RECLOSE + SEC(2.5):
                    grip = pinch_target(args.pinch_n)   # on the corner ramps: keep soft, dither on
                    st["wound"] -= 0.6 / CTRL_HZ
                else:                                   # flat found (or search timeout): firm clamp -> stroke
                    st["t_state"] = "wind"
            # closed-loop z: follow the nut down at the measured grip offset (minus the active lean)
            zt = nx[2].item() + st["grip_off"] - lean
            if ts in ("open", "rewind") and art.data.joint_pos[0, fj1].item() > st["p_flat"] + 3.5e-3:
                # Lift the opened jaws clear of the nut for the rewind swing: the pads grip only the top
                # ~4mm band, so +8mm cannot touch it (r6: the swing at ~2mm lateral clearance flicked the
                # 1-thread-deep nut off the bolt). Gated on the MEASURED width: lifting while the fingers
                # are still travelling to wide drags the clamped nut up the thread (r7: unseated it).
                zt += 0.008
            zt = max(zt, st["z_contact"] - 0.021)  # hard floor: full travel is ~24mm; below this the pads graze the bolt head
            # xy target. Default (Fable): servo the hand to the bolt axis — proven at the in-band
            # geometry. Optional nut-centring drives the NUT onto the axis (for marginal layouts that
            # orbit); its correction is tiny per step (a big clamp at 480 Hz slams the hand sideways and
            # ejects the nut — r6), so it eases the nut over rather than snapping.
            if args.nut_center and st["xy_tgt"] is not None:
                err = bx[:2] - nx[:2]
                if err.norm() > 5e-4:  # deadband: trickle-track only the slow posture-bias drift (~1mm/s
                    st["xy_tgt"] += (0.01 * err).clamp(-2e-6, 2e-6)  # max) so hex cam wiggle isn't chased
                tgt_xy = (st["xy_tgt"][0].item(), st["xy_tgt"][1].item())
            else:
                tgt_xy = (bx[0].item(), bx[1].item())
            action = servo(torch.tensor([tgt_xy[0], tgt_xy[1], zt], device=device), yaw_about_z(down, st["wound"]), grip)
            if dz <= args.stop_dz:  # drive well past the 20mm seat threshold; verdict prints seated()
                print(f"[done] dz={dz * 1e3:.1f}mm after {st['stroke']} strokes", flush=True)
                phase, ph_t = "finish", 0
        elif phase == "finish":      # open, rise (~30 mm/s), settle
            grip = OPEN
            zt = min(p[2].item() + 0.03 / CTRL_HZ, st["z_contact"] + 0.08)
            action = servo(torch.tensor([bx[0], bx[1], zt], device=device), down, grip)
            if ph_t >= SEC(2.0):
                break

        env.step(action, render=render)
        ph_t += 1

        if i % LOG_EVERY == 0:
            off = nut() - bolt()
            gpos = art.data.joint_pos[0, fj1].item()
            wz = scene.nuts[0].data.root_ang_vel_w[0, 2].item()
            tag = f"{phase}" + (f"/{st['t_state']}#{st['stroke']}" if phase == "thread" else "")
            print(f"  {i:4d} [{tag:>14s}] dz={off[2] * 1e3:6.2f}mm lat={off[:2].norm() * 1e3:4.1f}mm "
                  f"| wound={math.degrees(st['wound']):+6.1f} eeYaw={ee_u:+7.1f} nutYaw={nut_u:+7.1f} wz={wz:+5.2f} "
                  f"| grip={gpos * 1e3:5.2f}mm seated={int(scene.seated()[0, 0])}", flush=True)

    off = nut() - bolt()
    seated = int(scene.seated()[0, 0])
    print(f"SOLVE[fable] | seated={seated} | dz={off[2] * 1e3:.1f}mm lat={off[:2].norm() * 1e3:.1f}mm "
          f"| strokes={st['stroke']} caught={st['caught']}", flush=True)
    threading.Timer(10.0, lambda: os._exit(0)).start()  # Isaac teardown hangs; free the GPU regardless
    env.close()


if __name__ == "__main__":
    main()
    app.close()
    os._exit(0)

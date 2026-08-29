"""What can the G1's three-finger hand actually DO in the clear_organic_objects cell?

The franka binding is a solved parallel jaw: one number (80 mm aperture) predicts whether a
fruit is graspable. The G1 hand is a 7-DOF pinch — index and middle (two 45.8 mm phalanges
each) opposing a 3-joint thumb — and NOTHING about it is known here: no G1 solve exists in the
repo. Before writing one, measure the four things a pick-and-place plan needs:

  info     — body/joint names, action layout, and where the shoulder lands relative to the work.
  aperture — sweep the right hand from open to closed and report the opposition GAP (thumb pad
             to index/middle pads) in the wrist frame, so the graspable size band is a
             measurement instead of a guess.
  reach    — servo the right wrist to a hover over every scatter slot and over the bin rim and
             report the residual. Decides whether the layout dials in
             `_clear_organic_objects_g1_cfg` fit inside the arm's annulus (the franka lesson:
             reach and contact fail identically from the outside, so measure them apart).
  grasp    — the whole pick on ONE item, sweeping wrist tilt / approach depth / thumb
             opposition, reporting the lift achieved. The only phase that tests the hand PD and
             the produce friction together.

CONTROL MODE: `joint`. The G1 also registers `pink_ik`, but `pinocchio` is not installed in
this venv, so that mode cannot build here — and no solution in the repo uses it. This probe
(like the repo's joint-mode solves) drives the arm with its OWN damped-least-squares IK over
Isaac's `DifferentialIKController` on the PhysX Jacobian, and holds the waist and left arm at
home.

Run (per phase, so a failing phase is cheap to iterate on):
    OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 \
      .venv/bin/python -u scripts/probe_g1_organics.py --phase info --headless
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--phase", type=str, default="info",
                    help="info | aperture | reach | grasp | all")
parser.add_argument("--item", type=str, default="lemon_01", help="grasp-phase target item")
parser.add_argument("--env_name", type=str, default="packing.clear_organic_objects.g1.joint")
AppLauncher.add_app_launcher_args(parser)
args, _unknown = parser.parse_known_args()
app = AppLauncher(args).app

import torch  # noqa: E402
import isaaclab.utils.math as mu  # noqa: E402
from isaaclab.controllers.differential_ik import DifferentialIKController  # noqa: E402
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

ARM = ("right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
       "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
       "right_wrist_yaw_joint")
TIP_BODIES = ("right_hand_index_1_link", "right_hand_middle_1_link", "right_hand_thumb_2_link")
WRIST = "right_wrist_yaw_link"
# Distal phalanx length beyond the link origin (URDF: index/middle link 2 and the thumb distal
# are all 45.8 mm) — the link ORIGIN is the joint, the pad sits out along the link's local +x.
TIP_OUT = 0.0458
MAX_DQ = 0.03      # per-control-step joint target increment (the joint-solve convention)
LEAD_MAX = 0.25    # how far the target may lead the measured posture

# Centre of the pinch zone in the RIGHT WRIST frame, from the aperture sweep (2026-08-28):
# closing sweeps the index/middle pads from (211, -5) to (114, 68) mm in wrist (x, y) while the
# thumb pad travels (110, 62) -> (110, 17) mm, so the two sets CROSS and the object they trap
# ends up around x 130-170, y 30-60, z ~0 (the fingers straddle z = +26 / -31 mm). The wrist
# goal for a grasp at point P is therefore P - R_wrist @ GRASP_CENTRE, not "0.16 m back".
GRASP_CENTRE = (0.145, 0.040, 0.000)


def main() -> None:  # noqa: C901, PLR0915
    robobench.discover()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get(args.env_name)().build(num_envs=1, device=dev)
    scene, robot = env.scene, env.robot
    art = robot.articulation
    c = scene.cfg
    origin = env.iscene.env_origins[0]
    hz = 1.0 / (env.dt * robot.control_period)

    subs = robot.controller.controllers
    arm_c, hand_c = subs[0], subs[1]
    arm_names = [art.joint_names[j] for j in arm_c.joint_ids]
    hand_names = [art.joint_names[j] for j in hand_c.joint_ids]
    # slot of each driven joint within its sub-controller's action block
    a_slot = {n: i for i, n in enumerate(arm_names)}
    h_slot = {n: i for i, n in enumerate(hand_names)}
    n_arm = arm_c.action_dim
    arm_ids = [arm_c.joint_ids[a_slot[n]] for n in ARM]  # articulation ids, in ARM order

    wrist_i = art.body_names.index(WRIST)
    tip_i = [art.body_names.index(b) for b in TIP_BODIES]
    jac_i = wrist_i - 1 if art.is_fixed_base else wrist_i
    ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False,
                                    ik_method="dls", ik_params={"lambda_val": 0.05}),
        num_envs=1, device=dev)
    limits = art.data.joint_pos_limits[:, arm_ids, :]

    env.reset()
    home = art.data.joint_pos[0].clone()
    q_ref = art.data.joint_pos[:, arm_ids].clone()

    def V3(x, y, z):
        return torch.tensor([float(x), float(y), float(z)], device=dev)

    def wrist_pose_w():
        return (art.data.body_pos_w[:, wrist_i], art.data.body_quat_w[:, wrist_i])

    def wrist_env():
        return art.data.body_pos_w[0, wrist_i] - origin, art.data.body_quat_w[0, wrist_i]

    def tips_local():
        """Estimated pad positions of (index, middle, thumb) in the right-wrist frame."""
        wp, wq = wrist_env()
        Rw = mu.matrix_from_quat(wq.unsqueeze(0))[0]
        out = []
        for bi in tip_i:
            bp = art.data.body_pos_w[0, bi] - origin
            R = mu.matrix_from_quat(art.data.body_quat_w[0, bi].unsqueeze(0))[0]
            out.append(Rw.T @ ((bp + TIP_OUT * R[:, 0]) - wp))
        return out

    def hand_cmd(close: float, thumb0: float = 0.0, thumb1: float = 0.0) -> torch.Tensor:
        """Right-hand joint targets for a close fraction; the left hand stays at home (zeros).

        Right-hand signs from the URDF limits: index/middle 0..+1.571 (0 = straight),
        thumb_2 -1.745..0, thumb_1 -1.047..+0.724, thumb_0 +/-1.047 (opposition/abduction).
        """
        q = torch.zeros(1, hand_c.action_dim, device=dev)
        q[0, h_slot["right_hand_index_0_joint"]] = 1.00 * close
        q[0, h_slot["right_hand_index_1_joint"]] = 1.30 * close
        q[0, h_slot["right_hand_middle_0_joint"]] = 1.00 * close
        q[0, h_slot["right_hand_middle_1_joint"]] = 1.30 * close
        q[0, h_slot["right_hand_thumb_0_joint"]] = thumb0
        q[0, h_slot["right_hand_thumb_1_joint"]] = thumb1
        q[0, h_slot["right_hand_thumb_2_joint"]] = -1.40 * close
        return q

    def arm_hold() -> torch.Tensor:
        """Arm+waist block holding the home posture (the left arm and waist never move here)."""
        return home[arm_c.joint_ids].unsqueeze(0).clone()

    def arm_ik(goal_p, goal_q) -> torch.Tensor:
        """One DLS IK step of the right arm toward a world-frame wrist pose; other arm+waist
        joints stay at home."""
        nonlocal q_ref
        root_p, root_q = art.data.root_pos_w, art.data.root_quat_w
        tp, tq = mu.subtract_frame_transforms(root_p, root_q,
                                              goal_p.unsqueeze(0), goal_q.unsqueeze(0))
        ee_p, ee_q = mu.subtract_frame_transforms(root_p, root_q, *wrist_pose_w())
        ik.set_command(torch.cat([tp, tq], dim=-1), ee_p, ee_q)
        J = art.root_physx_view.get_jacobians()[:, jac_i][:, :, arm_ids]
        R = mu.matrix_from_quat(mu.quat_inv(root_q))
        J = torch.cat([torch.bmm(R, J[:, :3, :]), torch.bmm(R, J[:, 3:, :])], dim=1)
        q_arm = art.data.joint_pos[:, arm_ids]
        q_des = ik.compute(ee_p, ee_q, J, q_arm)
        q_new = (q_ref + (q_des - q_ref).clamp(-MAX_DQ, MAX_DQ)).clamp(limits[..., 0],
                                                                      limits[..., 1])
        q_ref = q_new.clamp(q_arm - LEAD_MAX, q_arm + LEAD_MAX)
        block = arm_hold()
        for k, n in enumerate(ARM):
            block[0, a_slot[n]] = q_ref[0, k]
        return block

    def step(arm_block, hand) -> None:
        env.step(torch.cat([arm_block, hand], dim=-1))

    def drive(goal_p, goal_q, secs, hand=None):
        hand = hand_cmd(0.0) if hand is None else hand
        for _ in range(max(1, round(secs * hz))):
            step(arm_ik(goal_p, goal_q), hand)

    def settle(secs, hand=None):
        hand = hand_cmd(0.0) if hand is None else hand
        for _ in range(max(1, round(secs * hz))):
            step(arm_hold(), hand)

    def rehome():
        nonlocal q_ref
        robot.reset(torch.tensor([0], device=dev, dtype=torch.long))
        q_ref = art.data.joint_pos[:, arm_ids].clone()
        ik.reset()

    def jaw_quat(azimuth: float, tilt: float) -> torch.Tensor:
        """Wrist orientation for a pinch. Local +x = the finger direction, local +y = the
        opposition axis (the thumb sits at +y and the fingers curl toward it), so a top-down
        pinch points +x down and keeps +y horizontal.

        `tilt` leans the finger direction back from straight-down (0 = straight down), which is
        what lets the pads meet a fruit near its equator instead of driving into the table.
        `azimuth` rotates the opposition axis in the horizontal plane.
        """
        ca, sa = math.cos(azimuth), math.sin(azimuth)
        y_l = V3(ca, sa, 0.0)
        down = V3(0.0, 0.0, -1.0)
        horiz = torch.cross(y_l, down, dim=0)
        x_l = math.cos(tilt) * down + math.sin(tilt) * horiz
        x_l = x_l / x_l.norm()
        z_l = torch.cross(x_l, y_l, dim=0)
        z_l = z_l / z_l.norm()
        y_l = torch.cross(z_l, x_l, dim=0)
        return mu.quat_from_matrix(torch.stack([x_l, y_l, z_l], dim=1).unsqueeze(0))[0]

    settle(1.0)  # let the scatter settle

    # ---- info ---------------------------------------------------------------------------------
    if args.phase in ("info", "all"):
        print(f"[g1] env={args.env_name} action_dim={robot.action_dim} "
              f"(arm+waist {n_arm} + hand {hand_c.action_dim})  fixed_base={art.is_fixed_base}",
              flush=True)
        print(f"[g1] arm block  = {arm_names}", flush=True)
        print(f"[g1] hand block = {hand_names}", flush=True)
        print(f"[g1] surface_z={c.surface_z} bin_pos={c.bin_pos} bin_scale={c.bin_scale}",
              flush=True)
        wp, _ = wrist_env()
        shp = art.data.body_pos_w[0, art.body_names.index("right_shoulder_pitch_link")] - origin
        idx, mid, thb = tips_local()
        print(f"[g1] right shoulder @ ({float(shp[0]):.3f},{float(shp[1]):.3f},{float(shp[2]):.3f})"
              f"  home wrist @ ({float(wp[0]):.3f},{float(wp[1]):.3f},{float(wp[2]):.3f})",
              flush=True)
        print(f"[g1] shoulder sits {float(shp[2]) - c.surface_z:.3f} m above the surface -> "
              f"usable horizontal radius <= "
              f"{math.sqrt(max(0.51**2 - (float(shp[2]) - c.surface_z)**2, 0)):.3f} m "
              f"(0.51 m arm+hand)", flush=True)
        print(f"[g1] open-hand pads in wrist frame: index {tuple(round(float(v), 3) for v in idx)}"
              f" middle {tuple(round(float(v), 3) for v in mid)}"
              f" thumb {tuple(round(float(v), 3) for v in thb)}", flush=True)
        for nm in scene.names:
            p = scene.items[nm].data.root_pos_w[0] - origin
            print(f"[g1]   item {nm:24s} @ ({float(p[0]):6.3f},{float(p[1]):6.3f},"
                  f"{float(p[2]):6.3f})  d_shoulder_xy="
                  f"{float(((p[:2] - shp[:2])**2).sum().sqrt()):.3f}", flush=True)

    # ---- aperture ------------------------------------------------------------------------------
    if args.phase in ("aperture", "all"):
        print(f"[g1] {'close':>6s} {'thumb0':>7s} {'thumb1':>7s} "
              f"{'gap_idx':>8s} {'gap_mid':>8s} {'idx_y':>7s} {'thb_y':>7s} {'reach_x':>8s}",
              flush=True)
        for thumb0 in (0.0, 0.5, 1.0, -0.5):
            for thumb1 in (0.0, 0.7):
                for close in (0.0, 0.3, 0.6, 0.9, 1.0):
                    hand = hand_cmd(close, thumb0, thumb1)
                    settle(0.5, hand)
                    idx, mid, thb = tips_local()
                    print(f"[g1] {close:6.2f} {thumb0:7.2f} {thumb1:7.2f} "
                          f"{float((idx - thb).norm()) * 1000:7.0f}mm "
                          f"{float((mid - thb).norm()) * 1000:7.0f}mm "
                          f"{float(idx[1]) * 1000:6.0f}mm {float(thb[1]) * 1000:6.0f}mm "
                          f"{float(max(idx[0], thb[0])) * 1000:7.0f}mm", flush=True)
        settle(0.3, hand_cmd(0.0))

    # ---- reach ---------------------------------------------------------------------------------
    if args.phase in ("reach", "all"):
        wx, wy = c.workbench_pos
        slots = [scene._slot_xy(i) for i in range(len(c.manifest))]
        targets = [(f"slot{i}", wx + sx, wy + sy, c.surface_z + 0.12)
                   for i, (sx, sy) in enumerate(slots)]
        rim = c.surface_z + c.bin_bbox[2] * c.bin_scale[2] + 0.10
        targets.append(("bin_rim", wx + c.bin_pos[0], wy + c.bin_pos[1], rim))
        print(f"[g1] {'target':>9s} {'wrist goal':>24s} {'resid':>7s} {'grasp_err':>10s}",
              flush=True)
        gq = jaw_quat(0.0, math.radians(30))
        Rg = mu.matrix_from_quat(gq.unsqueeze(0))[0]
        off = Rg @ V3(*GRASP_CENTRE)
        for nm, tx, ty, tz in targets:
            rehome()
            tgt = V3(tx, ty, tz)
            goal = tgt - off
            drive(goal, gq, 4.0)
            wp, wq = wrist_env()
            Rw = mu.matrix_from_quat(wq.unsqueeze(0))[0]
            grasp_p = wp + Rw @ V3(*GRASP_CENTRE)
            res = float(((wp - goal) ** 2).sum().sqrt())
            gerr = float(((grasp_p - tgt) ** 2).sum().sqrt())
            print(f"[g1] {nm:>9s} ({float(goal[0]):6.3f},{float(goal[1]):6.3f},"
                  f"{float(goal[2]):6.3f}) {res * 1000:6.0f}mm {gerr * 1000:9.0f}mm", flush=True)

    # ---- grasp ---------------------------------------------------------------------------------
    if args.phase in ("grasp", "all"):
        item = scene.items[args.item]
        # A controlled sweep needs the SAME pose every trial, and the item must be present:
        # `subset_sample` may leave it in the depot and the jitter/shuffle would confound the
        # parameters with the pose. Probe-local, not a scene change.
        scene.cfg.subset_sample = False
        scene.cfg.shuffle_slots = False
        scene.cfg.reset_pos_jitter = 0.0
        scene.cfg.reset_yaw_deg = 0.0
        print(f"[g1] grasp target {args.item} (fixed pose, no jitter/shuffle/subset)", flush=True)
        # The 48-trial sweep showed azimuth is a BINARY: 45/90 deg lifted 8/8 and 30/60 deg
        # failed 8/8, at every depth and grip. Span along the opposition axis does not explain
        # it (a 76 x 50 mm lemon spans 89 mm at 45 deg but only 81 mm at 60 deg, so the WORSE
        # span is the one that works), so resolve azimuth finely and log what the wrist actually
        # ACHIEVED: if the bad azimuths are orientations this arm cannot hold at this position,
        # the hand arrives rotated and the open fingers plough through the fruit — a kinematic
        # story the solve must handle by choosing feasible azimuths, not narrower ones.
        # Azimuth resolution (2026-08-28) said it is NOT orientation error — rot_err was 7-11 deg
        # on successes AND failures — but exposed the real defect: pos_res 36-55 mm, i.e. the
        # wrist stalls 5 cm short of the commanded grip pose at EVERY azimuth. Geometry explains
        # it: the open fingertips sit 211 mm out from the wrist while the pinch zone sits at
        # 145 mm, so commanding the pinch zone onto a fruit centre 25 mm above the table drives
        # the fingertips ~40 mm INTO the tabletop and the descent stops on table contact.
        # Leaning the hand back fixes it analytically — the fingertip rides
        # (0.211 - 0.145) * cos(tilt) below the pinch zone, which clears a 25 mm fruit centre
        # only past tilt ~68 deg — and tilt >= 40 was never validly tested (those rows were the
        # ones corrupted by the missing scene reset). So sweep the LEAN.
        print(f"[g1] {'tilt':>5s} {'dz':>6s} {'grip':>5s} {'az':>4s} {'pos_res':>8s} "
              f"{'rot_err':>8s} {'lift':>7s} {'pre_push':>9s} {'held':>5s}", flush=True)
        for tilt_deg in (30, 45, 60, 75):
            for dz in (0.012,):
                for grip in (1.15,):
                    for az_deg in (45, 75, 90, 135):
                        # RESET THE SCENE, not just the arm: the first sweep knocked the lemon
                        # onto the floor and every later trial then aimed at a target under the
                        # table and reported a clean 0 mm, which reads as "no effect" rather
                        # than "no target". Reset per trial or the sweep silently self-corrupts.
                        scene.reset(torch.tensor([0], device=dev, dtype=torch.long))
                        rehome()
                        settle(1.5)
                        ip = (item.data.root_pos_w[0] - origin).clone()
                        gq = jaw_quat(math.radians(az_deg), math.radians(tilt_deg))
                        Rg = mu.matrix_from_quat(gq.unsqueeze(0))[0]
                        off = Rg @ V3(*GRASP_CENTRE)
                        open_h = hand_cmd(0.0)
                        tgt = ip + V3(0, 0, dz)
                        drive(tgt + V3(0, 0, 0.16) - off, gq, 2.5, open_h)
                        drive(tgt - off, gq, 3.0, open_h)
                        # how far the OPEN hand shoved the fruit on the way in — separates
                        # "knocked away during the descent" from "slipped during the lift"
                        ipd = item.data.root_pos_w[0] - origin
                        pre = float(((ipd[:2] - ip[:2]) ** 2).sum().sqrt())
                        # what the wrist ACHIEVED vs what was commanded, at the grip pose
                        wp, wq = wrist_env()
                        pos_res = float((((wp - (tgt - off)) ** 2).sum().sqrt()))
                        dq = mu.quat_mul(gq.unsqueeze(0), mu.quat_conjugate(wq.unsqueeze(0)))
                        rot_err = math.degrees(float(mu.axis_angle_from_quat(dq)[0].norm()))
                        for close in (0.35, 0.7, grip):  # cage, then squeeze
                            drive(tgt - off, gq, 0.7, hand_cmd(close))
                        drive(tgt + V3(0, 0, 0.20) - off, gq, 3.0, hand_cmd(grip))
                        ip2 = item.data.root_pos_w[0] - origin
                        lift = float(ip2[2] - ip[2])
                        print(f"[g1] {tilt_deg:5d} {dz:6.3f} {grip:5.2f} {az_deg:4d} "
                              f"{pos_res * 1000:7.0f}mm {rot_err:7.1f}d "
                              f"{lift * 1000:6.0f}mm {pre * 1000:8.0f}mm "
                              f"{str(lift > 0.15):>5s}", flush=True)

    # ---- calib ---------------------------------------------------------------------------------
    # GRASP_CENTRE was ESTIMATED from pad link origins plus a 45.8 mm phalanx guess, and the
    # estimate is not good enough: with it, a solve whose hover residual was 19 mm still left the
    # fruit 65 mm from the nominal pinch zone, and the closing fingers then batted a pumpkin 0.9 m
    # across the table. So measure the real thing — run a grasp configuration KNOWN to lift
    # (tilt 45 / az 90 from the tilt sweep) and report where the fruit actually ends up in the
    # wrist frame, both just before the close and while genuinely held after the lift.
    if args.phase in ("calib", "all"):
        scene.cfg.subset_sample = False
        scene.cfg.shuffle_slots = False
        scene.cfg.reset_pos_jitter = 0.0
        scene.cfg.reset_yaw_deg = 0.0
        print(f"[g1] {'item':>14s} {'az':>4s} {'tilt':>5s} {'pre_close (wrist frame)':>26s} "
              f"{'held (wrist frame)':>26s} {'lift':>7s}", flush=True)
        for nm in ("lemon_01", "lemon_02", "pumpkinsmall", "lime01"):
            if nm not in scene.items:
                continue
            item = scene.items[nm]
            # NATURAL palm-facing-left orientations (az ~180; local +y is the palm normal, and the
            # right hand's palm should face the robot's left). The first calibration pass used
            # az 80-90 with a positive lean, which is a backhand — the numbers it produced are not
            # valid for a natural grasp, so they are re-measured here.
            for az_deg, tilt_deg in ((180, -45), (165, -45), (195, -35), (180, -55)):
                scene.reset(torch.tensor([0], device=dev, dtype=torch.long))
                rehome()
                settle(1.5)
                ip = (item.data.root_pos_w[0] - origin).clone()
                gq = jaw_quat(math.radians(az_deg), math.radians(tilt_deg))
                off = mu.matrix_from_quat(gq.unsqueeze(0))[0] @ V3(*GRASP_CENTRE)
                tgt = ip + V3(0, 0, 0.012)
                open_h = hand_cmd(0.0)
                drive(tgt + V3(0, 0, 0.16) - off, gq, 2.5, open_h)
                drive(tgt - off, gq, 3.0, open_h)

                def in_wrist():
                    wp, wq = wrist_env()
                    R = mu.matrix_from_quat(wq.unsqueeze(0))[0]
                    return R.T @ ((item.data.root_pos_w[0] - origin) - wp)

                pre_v = in_wrist()
                for close in (0.35, 0.7, 1.15):
                    drive(tgt - off, gq, 0.7, hand_cmd(close))
                drive(tgt + V3(0, 0, 0.20) - off, gq, 3.0, hand_cmd(1.15))
                ip2 = item.data.root_pos_w[0] - origin
                lift = float(ip2[2] - ip[2])
                held_v = in_wrist()
                print(f"[g1] {nm:>14s} {az_deg:4d} {tilt_deg:5d} "
                      f"({float(pre_v[0]):6.3f},{float(pre_v[1]):6.3f},{float(pre_v[2]):6.3f}) "
                      f"({float(held_v[0]):6.3f},{float(held_v[1]):6.3f},{float(held_v[2]):6.3f})"
                      f" {lift * 1000:6.0f}mm {'HELD' if lift > 0.15 else ''}", flush=True)

    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
    import os
    import threading
    threading.Timer(8.0, lambda: os._exit(0)).start()
    app.close()
    os._exit(0)

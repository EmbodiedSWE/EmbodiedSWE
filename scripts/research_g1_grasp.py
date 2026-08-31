"""Automated study of what the G1 can actually pick, and from where.

Two hypotheses to test independently, so each gets its own answer instead of one confounded
number:

  A. PALM AZIMUTH IS A RANGE, NOT A POINT. The solve currently pins the palm to "facing left"
     (az 180 +/- 45). The claim under test is that anything short of a full 180 deg reversal is
     fine — i.e. the usable window is much wider than the solve assumes, and pinning it that
     tightly is throwing away azimuths that would have fitted an item's narrow axis.
  B. SPACING IS THE REAL LIMIT. The claim is that the produce sits too close to the bin and too
     close to its neighbours, so the failures are about crowding and clearance rather than about
     grasping.

Design. The scene is stripped to ONE lime and the bin — no other produce, no clutter — which
removes neighbour interference entirely. If a lone lime picks reliably, crowding is confirmed as a
cause; whatever still fails with a lone lime is not about crowding at all.

  --phase az   fix the lime at a reference spot, sweep the palm azimuth over the FULL circle in
               20 deg steps. Answers A directly: it maps the usable window instead of assuming it.
  --phase pos  fix the azimuth at the best from the az phase, sweep the lime over a grid, and
               report each trial against BOTH distances that could matter — to the shoulder and
               to the bin's near wall. Answers B.
  --phase both run one then the other.

This is a PROBE, so it may place objects directly (the oracle-smoke convention); a solution may
not. Constants come from the solution module itself, so the study cannot silently drift from the
thing it is meant to inform.

Run:
    OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 \
      .venv/bin/python -u scripts/research_g1_grasp.py --phase az --headless
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import math
import os

from isaaclab.app import AppLauncher

SOLVE_PY = os.path.expanduser(
    "~/CoSiGen_Solutions/packing/clear_organic_objects/g1/joint/solve.py")

parser = argparse.ArgumentParser()
parser.add_argument("--phase", type=str, default="az", help="az | pos | both")
parser.add_argument("--item", type=str, default="lime01")
parser.add_argument("--az_fixed", type=float, default=180.0,
                    help="azimuth (deg) used by the pos phase")
parser.add_argument("--tilt", type=float, default=45.0, help="lean magnitude (deg)")
parser.add_argument("--aim_x", type=float, default=-1.0,
                    help="wrist-frame aim depth for the az phase; <0 = the solve default. Aim "
                         "depth and tilt INTERACT: at a near-horizontal tilt a shallow "
                         "(fingertip) aim throws the wrist far out and nothing is reachable, "
                         "while a deep (palm) aim keeps it close.")
parser.add_argument("--solve_py", type=str, default=SOLVE_PY)
AppLauncher.add_app_launcher_args(parser)
args, _unknown = parser.parse_known_args()
app = AppLauncher(args).app

import torch  # noqa: E402
import isaaclab.utils.math as mu  # noqa: E402
from isaaclab.controllers.differential_ik import DifferentialIKController  # noqa: E402
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402


def _load_solve_consts(path):
    """Import the solution module (it is import-safe by contract) purely for its constants."""
    spec = importlib.util.spec_from_file_location("g1_solve", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> None:  # noqa: C901, PLR0915
    S = _load_solve_consts(args.solve_py)
    robobench.discover()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ---- env: the g1 binding, stripped to ONE organic and no clutter ----------------------
    cfg = ENVS.get("packing.clear_organic_objects.g1.joint")()
    keep = args.item
    drop = tuple(n for n, _k, _o, _s, _m in cfg.scene_cfg.MANIFEST if n != keep)
    # dataclasses.replace re-runs __post_init__, which is what re-derives `manifest`
    cfg.scene_cfg = dataclasses.replace(cfg.scene_cfg, exclude=drop, subset_sample=False,
                                        shuffle_slots=False, reset_pos_jitter=0.0,
                                        reset_yaw_deg=0.0)
    env = cfg.build(num_envs=1, device=dev)
    scene, robot = env.scene, env.robot
    art = robot.articulation
    c = scene.cfg
    o = env.iscene.env_origins[0]
    hz = 1.0 / (env.dt * robot.control_period)
    print(f"[res] items in scene: {scene.names}", flush=True)

    arm_c, hand_c = robot.controller.controllers[0], robot.controller.controllers[1]
    arm_names = [art.joint_names[j] for j in arm_c.joint_ids]
    hand_names = [art.joint_names[j] for j in hand_c.joint_ids]
    a_slot = {n: i for i, n in enumerate(arm_names)}
    h_slot = {n: i for i, n in enumerate(hand_names)}
    arm_ids = [arm_c.joint_ids[a_slot[n]] for n in S.ARM]
    wrist_i = art.body_names.index(S.WRIST)
    sh_i = art.body_names.index(S.SHOULDER)
    jac_i = wrist_i - 1 if art.is_fixed_base else wrist_i
    ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False,
                                    ik_method="dls", ik_params={"lambda_val": 0.05}),
        num_envs=1, device=dev)
    limits = art.data.joint_pos_limits[:, arm_ids, :]

    env.reset()
    hold_arm = art.data.joint_pos[0, arm_c.joint_ids].clone()
    for nm, v in S.LEFT_TUCK.items():          # same left-arm tuck the solve uses
        hold_arm[a_slot[nm]] = v
    q_ref = art.data.joint_pos[:, arm_ids].clone()
    item = scene.items[args.item]

    def V3(x, y, z):
        return torch.tensor([float(x), float(y), float(z)], device=dev)

    def wrist_pq():
        return art.data.body_pos_w[:, wrist_i], art.data.body_quat_w[:, wrist_i]

    def wrist_env():
        return art.data.body_pos_w[0, wrist_i] - o, art.data.body_quat_w[0, wrist_i]

    def shoulder_xy():
        return (art.data.body_pos_w[0, sh_i] - o)[:2]

    def item_p():
        return item.data.root_pos_w[0] - o

    def hand_cmd(close, thumb1=0.0):
        q = torch.zeros(1, hand_c.action_dim, device=dev)
        q[0, h_slot["right_hand_index_0_joint"]] = 1.00 * close
        q[0, h_slot["right_hand_index_1_joint"]] = 1.30 * close
        q[0, h_slot["right_hand_middle_0_joint"]] = 1.00 * close
        q[0, h_slot["right_hand_middle_1_joint"]] = 1.30 * close
        q[0, h_slot["right_hand_thumb_1_joint"]] = thumb1
        q[0, h_slot["right_hand_thumb_2_joint"]] = -1.40 * close
        return q

    OPEN, SHUT = (0.0, S.THUMB_PARK), (S.GRIP, 0.0)

    def one_step(goal_p, goal_q, hand):
        nonlocal q_ref
        root_p, root_q = art.data.root_pos_w, art.data.root_quat_w
        tp, tq = mu.subtract_frame_transforms(root_p, root_q,
                                              (goal_p + o).unsqueeze(0), goal_q.unsqueeze(0))
        ee_p, ee_q = mu.subtract_frame_transforms(root_p, root_q, *wrist_pq())
        ik.set_command(torch.cat([tp, tq], dim=-1), ee_p, ee_q)
        J = art.root_physx_view.get_jacobians()[:, jac_i][:, :, arm_ids]
        R = mu.matrix_from_quat(mu.quat_inv(root_q))
        J = torch.cat([torch.bmm(R, J[:, :3, :]), torch.bmm(R, J[:, 3:, :])], dim=1)
        q_arm = art.data.joint_pos[:, arm_ids]
        q_des = ik.compute(ee_p, ee_q, J, q_arm)
        if not bool(torch.isfinite(q_des).all()):
            q_des = q_ref
        q_new = (q_ref + (q_des - q_ref).clamp(-S.MAX_DQ, S.MAX_DQ)).clamp(limits[..., 0],
                                                                          limits[..., 1])
        q_ref = q_new.clamp(q_arm - S.LEAD_MAX, q_arm + S.LEAD_MAX)
        block = hold_arm.unsqueeze(0).clone()
        for k, nm in enumerate(S.ARM):
            block[0, a_slot[nm]] = q_ref[0, k]
        env.step(torch.cat([block, hand_cmd(*hand)], dim=-1))

    def jaw_quat(az, tilt):
        y_l = V3(math.cos(az), math.sin(az), 0.0)
        down = V3(0.0, 0.0, -1.0)
        horiz = torch.cross(y_l, down, dim=0)
        x_l = math.cos(tilt) * down + math.sin(tilt) * horiz
        x_l = x_l / x_l.norm()
        z_l = torch.cross(x_l, y_l, dim=0)
        z_l = z_l / z_l.norm()
        y_l = torch.cross(z_l, x_l, dim=0)
        return mu.quat_from_matrix(torch.stack([x_l, y_l, z_l], dim=1).unsqueeze(0))[0]

    def pinch_now(ref=None):
        ref = S.SEAT_REF if ref is None else ref
        wp, wq = wrist_env()
        return wp + mu.matrix_from_quat(wq.unsqueeze(0))[0] @ V3(*ref)

    def item_in_wrist():
        wp, wq = wrist_env()
        return mu.matrix_from_quat(wq.unsqueeze(0))[0].T @ (item_p() - wp)

    def servo(point, gq, hand, secs, tol=None, ref=None):
        ref = S.SEAT_REF if ref is None else ref
        R = mu.matrix_from_quat(gq.unsqueeze(0))[0]
        goal = point - R @ V3(*ref)
        for _ in range(max(1, round(secs * hz))):
            one_step(goal, gq, hand)
            if tol is not None and float((pinch_now(ref) - point).norm()) < tol:
                break
        return float((pinch_now(ref) - point).norm())

    def hold(secs, hand):
        wp, wq = wrist_env()
        for _ in range(max(1, round(secs * hz))):
            one_step(wp, wq, hand)

    def lean_sign(az, tilt, aim_xy):
        best, best_d = 1.0, None
        for s in (1.0, -1.0):
            R = mu.matrix_from_quat(jaw_quat(az, s * tilt).unsqueeze(0))[0]
            w = aim_xy - (R @ V3(*S.PALM_REF))[:2]
            d = float((w - shoulder_xy()).norm())
            if best_d is None or d < best_d:
                best, best_d = s, d
        return best

    def ready(secs=3.0):
        sx, sy = shoulder_xy()
        sh_z = float((art.data.body_pos_w[0, sh_i] - o)[2])
        aim = V3(float(sx) + 0.08, float(sy) + 0.26, 0.0)
        az = math.radians(180.0)
        gq = jaw_quat(az, lean_sign(az, math.radians(45), aim[:2]) * math.radians(45))
        R = mu.matrix_from_quat(gq.unsqueeze(0))[0]
        return servo(V3(float(sx) + 0.08, float(sy) + 0.26, sh_z - 0.06)
                     + R @ V3(*S.GRASP_CENTRE), gq, OPEN, secs, tol=0.04)

    q_ready: list = []

    def goto_ready(secs=2.2):
        """Joint-space return to the captured staging posture.

        MANDATORY between trials, not tidiness. Without it the first trial that completes a carry
        leaves the arm far from the staging pose, `ready()` alone cannot recover, and every later
        trial reports a huge hover residual — measured 115-390 mm on rows that followed a
        successful lift, while rows before it read 18-73 mm. That silently invalidates the second
        half of any sweep, which is exactly how an automated study lies to you.
        """
        nonlocal q_ref
        if not q_ready:
            return
        tgt = torch.stack(q_ready).unsqueeze(0)
        for _ in range(max(1, round(secs * hz))):
            q_arm = art.data.joint_pos[:, arm_ids]
            q_ref = (q_ref + (tgt - q_ref).clamp(-S.MAX_DQ, S.MAX_DQ)).clamp(limits[..., 0],
                                                                            limits[..., 1])
            block = hold_arm.unsqueeze(0).clone()
            for k, nm in enumerate(S.ARM):
                block[0, a_slot[nm]] = q_ref[0, k]
            env.step(torch.cat([block, hand_cmd(*OPEN)], dim=-1))
        q_ref = art.data.joint_pos[:, arm_ids].clone()
        ik.reset()

    def bin_xy():
        return (scene.bin.data.root_pos_w[0] - o)[:2]

    def rim_world():
        return float((scene.bin.data.root_pos_w[0] - o)[2]) + c.bin_bbox[2] * c.bin_scale[2]

    def bin_wall_x():
        """World x of the bin's near (-x) outer wall — the edge the produce must stay clear of."""
        return float(bin_xy()[0]) - c.bin_bbox[0] * c.bin_scale[0] / 2.0

    def place_item(px, py):
        """Teleport the lime to (px, py) and let it settle. Probe-only."""
        st = torch.zeros(1, 13, device=dev)
        st[0, 0], st[0, 1], st[0, 2] = px, py, c.surface_z + 0.02
        st[0, 3] = 1.0
        st[0, 0:3] += o
        item.write_root_state_to_sim(st, torch.tensor([0], device=dev, dtype=torch.long))
        hold(1.5, OPEN)

    def trial(px, py, az_deg, tilt_deg, aim=None, grip=None):
        """One full pick-and-place attempt. Returns a metrics dict.

        `aim` overrides the wrist-frame point that gets driven onto the fruit, and `grip` the final
        close fraction. Both exist for the PALM-GRASP study: the calibrated references are
        fingertip positions (SEAT_REF x = 0.167 is out at the pads), and a grasp that holds the
        fruit against the PALM instead needs it much deeper in the hand — around x 0.09, where the
        finger roots are. A fingertip pinch that over-closes also self-penetrates: with the fruit
        out at the pads the fingers keep travelling into the palm, jam, and then cannot reopen,
        which is why lifts succeeded but the fruit was never released.
        """
        aim = S.SEAT_REF if aim is None else aim
        grip = S.GRIP if grip is None else grip
        place_item(px, py)
        goto_ready()      # repeatable conditioning, so trial N is comparable to trial 0
        ready()
        p0 = item_p().clone()
        az = math.radians(az_deg)
        tmag = math.radians(tilt_deg)
        tilt = lean_sign(az, tmag, p0[:2]) * tmag
        gq = jaw_quat(az, tilt)
        servo(V3(float(p0[0]), float(p0[1]), c.surface_z + S.TRAVERSE), gq, OPEN, 3.5,
              tol=0.03, ref=aim)
        r_hov = servo(p0 + V3(0, 0, S.HOVER), gq, OPEN, 4.0, tol=0.02, ref=aim)
        p1 = item_p().clone()
        # Same fingertip-clearance clamp the solve uses: the open fingertips ride 66 mm beyond the
        # pinch zone, so without it the descent is commanded through the tabletop and stalls.
        # Clamp measured from the ACTUAL aim depth, not a hardcoded one. The fingertips sit
        # (FINGER_OUT - aim_x) beyond whatever point is being driven onto the fruit, so a deep
        # palm aim (x 0.09) puts them 131 mm out rather than 66 mm — and the old constant
        # under-clamped by 65 mm, driving the fingers into the table and stalling the descent.
        floor_z = (c.surface_z + 0.004
                   + (S.FINGER_OUT - aim[0]) * math.cos(tilt))
        grip_pt = V3(float(p1[0]), float(p1[1]), max(float(p1[2]) + 0.012, floor_z))
        r_des = servo(grip_pt, gq, OPEN, 4.5, tol=0.015, ref=aim)
        servo(grip_pt, gq, (S.CAGE, 0.0), 0.6, ref=aim)
        iw = item_in_wrist()
        seat = float((iw - V3(*aim)).norm())
        for close in [x for x in (0.55, 0.75, 0.95, grip) if x <= grip]:
            servo(grip_pt, gq, (close, 0.0), 0.5, ref=aim)
        servo(grip_pt + V3(0, 0, 0.20), gq, (grip, 0.0), 3.0, ref=aim)
        lift = float(item_p()[2] - p1[2])
        cleared = False
        place = {}
        if lift > 0.10:      # only bother carrying if it actually came up
            bx, by = bin_xy()
            gq_c = jaw_quat(math.radians(180.0),
                            lean_sign(math.radians(180.0), tmag,
                                      V3(float(bx), float(by), 0.0)[:2]) * tmag)
            held_at = pinch_now()
            for s in (0.5, 1.0):
                servo(held_at, gq_c, (grip, 0.0), 0.8)
            pin, pit = pinch_now(), item_p()
            dest = (V3(float(bx), float(by), rim_world() + S.CARRY_OVER_RIM)
                    + (pin - pit) * V3(1, 1, 0))
            start = pin.clone()
            r_carry = 0.0
            for leg in (0.34, 0.67, 1.0):
                r_carry = servo(start + (dest - start) * leg, gq_c, (grip, 0.0), 2.0, tol=0.02)
            pin, pit = pinch_now(), item_p()
            held_after_carry = float((item_p() - pinch_now()).norm())
            low = (V3(float(bx), float(by), rim_world() + S.PLACE_OVER_RIM)
                   + (pin - pit) * V3(1, 1, 0))
            r_low = servo(low, gq_c, (grip, 0.0), 3.0, tol=0.02)
            place = {"r_carry": r_carry, "r_low": r_low, "held": held_after_carry}
            gq_t = jaw_quat(math.radians(180.0),
                            lean_sign(math.radians(180.0), math.radians(S.TIP_DEG),
                                      V3(float(bx), float(by), 0.0)[:2]) * math.radians(S.TIP_DEG))
            for close in (0.85, 0.5, 0.15, 0.0):
                servo(low, gq_t, (close, S.THUMB_PARK), 0.45)
            servo(low + V3(0, 0, 0.18), gq_t, OPEN, 2.0)
            hold(0.8, OPEN)
            cleared = bool(scene.cleared()[0, 0].item())
            rel = item_p()
            place.update({"in_bin": bool(scene._in_bin()[0, 0].item()),
                          "rel": (float(rel[0]), float(rel[1]), float(rel[2]))})
            print(f"[place] r_carry={place['r_carry'] * 1000:4.0f}mm "
                  f"held_after_carry={place['held'] * 1000:4.0f}mm "
                  f"r_low={place['r_low'] * 1000:4.0f}mm "
                  f"released=({rel[0]:6.3f},{rel[1]:6.3f},{rel[2]:6.3f}) "
                  f"in_bin={place['in_bin']} cleared={cleared}", flush=True)
        sx, sy = shoulder_xy()
        return {
            "d_sh": float(((p0[:2] - torch.stack([sx, sy]))**2).sum().sqrt()),
            "d_bin": float(p0[0]) - bin_wall_x(),
            "r_hov": r_hov, "r_des": r_des, "seat": seat, "lift": lift, "cleared": cleared,
        }

    HDR = (f"[res] {'x':>6s} {'y':>6s} {'az':>5s} {'d_sh':>5s} {'d_bin':>6s} "
           f"{'r_hov':>6s} {'r_des':>6s} {'seat':>5s} {'lift':>6s} {'cleared':>7s}")

    def report(px, py, az_deg, m):
        print(f"[res] {px:6.3f} {py:6.3f} {az_deg:5.0f} {m['d_sh']:5.3f} {m['d_bin']:6.3f} "
              f"{m['r_hov'] * 1000:5.0f}m {m['r_des'] * 1000:5.0f}m {m['seat'] * 1000:4.0f}m "
              f"{m['lift'] * 1000:5.0f}m {str(m['cleared']):>7s}", flush=True)

    # A reference spot for the azimuth sweep: mid-annulus from the shoulder and well clear of the
    # bin wall, so neither distance is the thing under test.
    sx0, sy0 = (float(v) for v in shoulder_xy())
    REF = (sx0 - 0.10, sy0 + 0.26)
    print(f"[res] shoulder=({sx0:.3f},{sy0:.3f}) bin_wall_x={bin_wall_x():.3f} "
          f"rim={rim_world():.3f} surface={c.surface_z}", flush=True)
    print(f"[res] azimuth-sweep reference spot = ({REF[0]:.3f},{REF[1]:.3f})", flush=True)
    r_rdy = ready(secs=4.5)                       # capture the staging posture once, from clean
    q_ready.extend(list(art.data.joint_pos[0, arm_ids].clone()))
    print(f"[res] staging posture captured (residual {r_rdy * 1000:.0f}mm)", flush=True)

    if args.phase in ("az", "both"):
        print("[res] === PHASE A: palm azimuth over the full circle, one lime, fixed spot ===",
              flush=True)
        print(HDR, flush=True)
        _aim = None if args.aim_x < 0 else (args.aim_x, 0.035, -0.010)
        for az_deg in range(0, 360, 30):
            m = trial(REF[0], REF[1], float(az_deg), args.tilt, aim=_aim)
            report(REF[0], REF[1], float(az_deg), m)

    if args.phase == "palm":
        # PALM GRASP vs FINGERTIP PINCH. The failure being chased: lifts succeed but the fruit is
        # never released, because a fingertip pinch that over-closes drives the fingers into the
        # palm, self-penetrates, jams, and cannot reopen. The fix under test is to hold the fruit
        # against the PALM instead — i.e. aim a point much deeper in the hand than the calibrated
        # fingertip references (SEAT_REF x = 0.167) — and to stop over-driving the close.
        # `held_end` is the metric that matters: distance from the hand at the end of the trial.
        # A big number means it genuinely let go; a small one means it is still stuck in the hand.
        print("[res] === PHASE PALM: aim depth x grip, one item, fixed spot ===", flush=True)
        print(f"[res] {'aim_x':>6s} {'tilt':>6s} {'grip':>5s} {'seat':>5s} {'lift':>6s} "
              f"{'held_end':>8s} {'cleared':>7s}", flush=True)
        for tilt_deg in (70.0, 85.0):
          for aim_x in (0.090, 0.110):
            for grip in (0.85, 1.05):
                aim = (aim_x, 0.035, -0.010)
                m = trial(REF[0], REF[1], 180.0, tilt_deg, aim=aim, grip=grip)
                he = float((item_p() - pinch_now(aim)).norm())
                print(f"[res] {aim_x:6.3f} {tilt_deg:6.0f} {grip:5.2f} {m['seat'] * 1000:4.0f}m "
                      f"{m['lift'] * 1000:5.0f}m {he * 1000:7.0f}m {str(m['cleared']):>7s}",
                      flush=True)

    if args.phase == "one":
        # Repeat the single best-measured cell a few times. Exists to be WATCHED: it is the
        # shortest recording that shows the grasp close up, and repetition shows how consistent it
        # is rather than whether it can happen once.
        print(f"[res] === PHASE ONE: az={args.az_fixed:.0f} tilt={args.tilt:.0f}, x3 ===",
              flush=True)
        print(HDR, flush=True)
        for _rep in range(3):
            m = trial(REF[0], REF[1], args.az_fixed, args.tilt)
            report(REF[0], REF[1], args.az_fixed, m)

    if args.phase in ("grid", "both"):
        # TILT x AZIMUTH. Tilt is the axis that decides whether the pinch envelops the fruit or
        # grazes its top, because it sets how far the open fingertips hang below the pinch zone
        # (0.066 * cos(tilt)) and therefore whether the table-clearance clamp fires at all:
        #   tilt 45 -> fingertips 47 mm below, clamp lifts the aim to surface+0.051, i.e. ~26 mm
        #              ABOVE a lemon's centre — the pinch lands on the fruit's top surface;
        #   tilt 65 -> fingertips 28 mm below, clamp inactive, the aim sits at the fruit's centre.
        # A steep-tilt set was rejected early on, but that test predates the palm-direction fix,
        # the re-calibrated seat references and the left-arm tuck, so it is worth re-running.
        print("[res] === PHASE C: tilt x azimuth, one item, fixed spot ===", flush=True)
        print(HDR + f"{'tilt':>5s}", flush=True)
        for tilt_deg in (40.0, 45.0, 50.0):
            for az_deg in (150.0, 165.0, 180.0, 195.0, 210.0):
                m = trial(REF[0], REF[1], az_deg, tilt_deg)
                report(REF[0], REF[1], az_deg, m)
                print(f"[res]   ^ tilt={tilt_deg:.0f}", flush=True)

    if args.phase in ("pos", "both"):
        print(f"[res] === PHASE B: position sweep at az={args.az_fixed:.0f} ===", flush=True)
        print(HDR, flush=True)
        # vary distance from the shoulder (rows) and lateral position toward the bin (cols), so
        # `d_sh` and `d_bin` separate in the output
        for dy in (0.18, 0.24, 0.30, 0.36):
            for dx in (-0.20, -0.12, -0.04, 0.04, 0.12):
                px, py = sx0 + dx, sy0 + dy
                m = trial(px, py, args.az_fixed, args.tilt)
                report(px, py, args.az_fixed, m)

    print("RESEARCH_DONE", flush=True)


if __name__ == "__main__":
    main()
    import threading
    threading.Timer(8.0, lambda: os._exit(0)).start()
    app.close()
    os._exit(0)

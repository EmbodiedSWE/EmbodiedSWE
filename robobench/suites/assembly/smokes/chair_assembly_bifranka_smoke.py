"""Bi-Franka scripted solution for ChairAssemblyScene — two arms, OSC, RECORDED.

The robot solution the NullRobot oracle previews, on `assembly.chair.multi.osc`. Layout
(reach-audited): one Franka at each END of the double-width packing table's long axis,
facing each other; BOTH chair components sit between them, split across the WIDTH and
facing each other — the body on the back half (yaw 180, studs pointing across the
table) and the backrest lying face-up on the front half, holes toward the studs. The
assembly reads directly off the layout: carry the back across the lane onto the studs,
then thread a nut from each side. Choreography:

  1. settle    — deterministic reset (no jitters, fixed yaw/slots);
  2. dual grasp— the panel lies on its riser with a SIDE band facing each arm; each
                 arm pinches its band head-on at the measured 60-66 mm zone (the
                 finger-vertical roll is about the approach axis = joint 7, the one
                 rotation this controller always tracks);
  3. lift+erect+carry — both arms servo the panel's LIVE pose toward seated: from
                 this spawn yaw the whole reorientation is a pure world-x rotation
                 (the controller's good axis), then the carry is pure translation;
  4. insert    — both arms servo the panel's pose error (targets from the live chair
                 base each step — the chair scoots; track it) and both holes slide
                 onto both studs; open to release; the scene welds (base,back) on
                 seat;
  5. nut runs  — EACH arm fetches ITS nut (top-down pinch across the hex flats),
                 reorients it axis-horizontal (all stud-frame math composed through the
                 live base quat), presents it to its stud tip until the measured-rotation
                 screw joint engages, then threads it home with wrist-roll REGRIP CYCLES
                 (roll -2.4 rad, open, roll back, re-pinch, ...); the mechanic holds the
                 nut on the stud between grips (sag-proof) and converts only REAL
                 rotation into advance, so every degree is earned;
  6. verdict   — success() (all 3 pairs welded), zero ordering violations, video.

Run (GPU node with the isaaclab env):
    python -m robobench.suites.assembly.smokes.chair_assembly_bifranka_smoke --headless
"""

from __future__ import annotations

try:  # pink_ik eigenpy converters want pinocchio before AppLauncher; harmless for osc
    import pinocchio  # noqa: F401
except Exception as _exc:  # noqa: BLE001
    print(f"[bifranka] pinocchio unavailable ({_exc!r}) — continuing (osc mode)", flush=True)

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--out", type=str, default="chair_bifranka_frames.npz")
parser.add_argument("--max_cycles", type=int, default=34,
                    help="regrip spin cycles per nut before giving up")
parser.add_argument("--wrist-test", action="store_true",
                    help="measure which side-band grasp orientations the RIGHT arm can "
                         "actually reach (4 candidates), print a table, and exit")
parser.add_argument("--pick-only", action="store_true",
                    help="end the run (and save the recording) right after the dual "
                         "pick + carry — the fast evidence loop")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math

import numpy as np
import torch

import robobench
from robobench.core import ENVS
from robobench.suites.assembly.scenes import ChairAssemblySceneCfg

MAX_STEP = 0.008  # carrier step (m) — the binding smoke's proven marching pace
ROT_STEP = 0.9  # rot action clamp, units of rot_scale (0.097 rad)
OPEN, CLOSED = 0.04, 0.0  # per-finger joint targets
SPIN_PER_CYCLE = 2.4  # wrist roll per regrip cycle (rad), inside panda_joint7's +/-2.9


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    # determinism only — the LAYOUT (rotated chair, standing back, per-arm nuts) comes
    # from the registered multi binding's scene cfg
    from robobench.suites.assembly.configs.envs import _chair_multi_cfg

    scene_cfg = _chair_multi_cfg()
    scene_cfg.reset_pos_jitter = 0.0
    scene_cfg.reset_yaw_deg = 0.0
    scene_cfg.seat_jitter = 0.0
    scene_cfg.seat_yaw_deg = 0.0
    scene_cfg.shuffle_slots = False
    env = ENVS.get("assembly.chair.multi.osc")().build(
        num_envs=1, device=device, scene_cfg=scene_cfg)
    scene = env.scene
    c = scene.cfg
    no_op = torch.zeros(1, env.robot.action_dim, device=device)
    slices = env.robot.action_slices  # {"left": slice, "right": slice}
    arms = {name: env.robot.robots[name] for name in ("left", "right")}

    from isaaclab.utils.math import (quat_apply, quat_apply_inverse, quat_conjugate,
                                     quat_mul)

    # --- per-arm bookkeeping -------------------------------------------------------------
    ee_idx = {s: arms[s].articulation.find_bodies(arms[s].EE_BODY)[0][0] for s in arms}
    pos_scale = {s: float(getattr(
        env.robot.robots[s].controller.controllers[0].cfg, "pos_scale", 0.02)) for s in arms}
    rot_scale = {s: float(getattr(
        env.robot.robots[s].controller.controllers[0].cfg, "rot_scale", 0.097)) for s in arms}
    grip_now = {"left": OPEN, "right": OPEN}
    latch: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}  # per-arm hold target

    origins = env.iscene.env_origins

    def ee_pose(s: str) -> tuple[torch.Tensor, torch.Tensor]:
        st = arms[s].articulation.data.body_link_state_w[0, ee_idx[s]]
        return st[0:3] - origins[0], st[3:7]

    def body_pose(name: str) -> tuple[torch.Tensor, torch.Tensor]:
        obj = env.iscene[name]
        return obj.data.root_pos_w[0] - origins[0], obj.data.root_quat_w[0]

    # --- recording -----------------------------------------------------------------------
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        o = origins[0].detach().cpu().numpy().astype(float)
        # FULL rendering: the default pipeline desyncs rendered transforms from physics
        # for bodies whose root state is written directly (the panel spawn) — the last
        # recording showed the panel frozen at rest while physics telemetry had it
        # carried 38 cm up. Footage must show the truth.
        env.sim.set_render_mode(env.sim.RenderMode.FULL_RENDERING)
        env.sim.set_camera_view(tuple(np.array((2.15, -2.35, 1.50)) + o),
                                tuple(np.array((0.0, -0.05, c.surface_z + 0.30)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        print(f"[bifranka] camera ready {np.asarray(annot.get_data()).shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[bifranka] camera setup FAILED ({exc!r}) — continuing", flush=True)

    step_i = 0

    def build_actions(drives: dict | None = None) -> torch.Tensor:
        """16-dim action: each driven arm gets its (pos_act, rot_act) from `drives`;
        every other arm holds its latched pose (or zero action when unlatched). Grips
        always from grip_now."""
        act = no_op.clone()
        drives = drives or {}
        for s in arms:
            sl = slices[s]
            if s in drives:
                pos_act, rot_act = drives[s]
                a = torch.cat([pos_act, rot_act if rot_act is not None
                               else torch.zeros(3, device=device)])
            elif s in latch:
                lp, lq = latch[s]
                p, q = ee_pose(s)
                pa = ((lp - p) / pos_scale[s]).clamp(-1.0, 1.0)
                qe = quat_mul(lq.unsqueeze(0), quat_conjugate(q.unsqueeze(0)))[0]
                ra = (_axis_angle(qe) / rot_scale[s]).clamp(-ROT_STEP, ROT_STEP)
                a = torch.cat([pa, ra])
            else:
                a = torch.zeros(6, device=device)
            act[0, sl] = torch.cat([a, torch.full((2,), grip_now[s], device=device)])
        return act

    def _axis_angle(q: torch.Tensor) -> torch.Tensor:
        """wxyz quat -> axis*angle (3,), the OSC rot-action parameterization."""
        q = q / q.norm().clamp_min(1e-9)
        w = float(q[0].clamp(-1.0, 1.0))
        ang = 2.0 * math.acos(abs(w))
        if ang < 1e-6:
            return torch.zeros(3, device=q.device)
        v = q[1:4] * (1.0 if w >= 0 else -1.0)
        return v / v.norm().clamp_min(1e-9) * ang

    def step(k: int = 1, drive=None, pos_act=None, rot_act=None, drives=None) -> None:
        nonlocal step_i
        if drives is None and drive is not None:
            drives = {drive: (pos_act, rot_act)}
        for _ in range(k):
            # render only on capture steps (explicit renders below): per-step rendering
            # made a ~5000-step run take 12 wall minutes; physics alone is ~4x faster
            env.step(build_actions(drives), render=False)
            if annot is not None and step_i % args.record_every == 0:
                for _f in range(3):
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    def march(s: str, pos_tgt, quat_tgt=None, steps=600, tol=0.012) -> bool:
        """Virtual-carrier march of arm `s` to a pose (env-origin-local). Returns whether
        the EE converged. The other arm holds its latch throughout."""
        pos_tgt = torch.as_tensor(pos_tgt, dtype=torch.float32, device=device)
        cur = ee_pose(s)[0].clone()
        for _i in range(steps):
            p, q = ee_pose(s)
            d = pos_tgt - cur
            cur = cur + d * (MAX_STEP / d.norm().clamp_min(1e-9)).clamp(max=1.0)
            pa = ((cur - p) / pos_scale[s]).clamp(-1.0, 1.0)
            ra = torch.zeros(3, device=device)
            if quat_tgt is not None:
                qe = quat_mul(quat_tgt.unsqueeze(0), quat_conjugate(q.unsqueeze(0)))[0]
                ra = (_axis_angle(qe) / rot_scale[s]).clamp(-ROT_STEP, ROT_STEP)
            step(1, drive=s, pos_act=pa, rot_act=ra)
            if (ee_pose(s)[0] - pos_tgt).norm() < tol and (
                    quat_tgt is None or _quat_err(ee_pose(s)[1], quat_tgt) < 0.12):
                return True
        return (ee_pose(s)[0] - pos_tgt).norm() < tol * 2

    def _quat_err(q, qt) -> float:
        qe = quat_mul(qt.unsqueeze(0), quat_conjugate(q.unsqueeze(0)))[0]
        return float(_axis_angle(qe).norm())

    def set_grip(s: str, val: float, settle: int = 30) -> None:
        grip_now[s] = val
        step(settle)

    def hold_here(s: str) -> None:
        p, q = ee_pose(s)
        latch[s] = (p.clone(), q.clone())

    def unlatch(s: str) -> None:
        latch.pop(s, None)

    checks: list[tuple[str, bool]] = []

    def check(name: str, ok: bool) -> None:
        checks.append((name, ok))
        print(f"[bifranka] {'PASS' if ok else 'FAIL'}: {name}", flush=True)

    def report(tag: str) -> None:
        held = scene.grasp_held[0].int().tolist() if hasattr(scene, "grasp_held") else "n/a"
        print(f"[bifranka] {tag:14s} welded={scene.welded[0].int().tolist()} "
              f"score={int(scene.score()[0])} held={held} "
              f"eng={scene._scr_eng[0].int().tolist()} frames={len(frames)}", flush=True)

    # ============================ phase 1: settle ===========================================
    env.reset()
    step(90)
    report("settle")
    check("settle: nothing welded", not bool(scene.welded[0].any()))

    down = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)  # EE z-axis down (w,x,y,z)
    ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    def back_axes() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Live back pose + its local axes in world: (pos, quat, x_w, y_w, z_w)."""
        bp_l, bq_l = body_pose("back")
        ax = [quat_apply(bq_l.unsqueeze(0), torch.tensor([[float(i == j) for j in range(3)]],
                                                         device=device))[0] for i in range(3)]
        return bp_l, bq_l, ax[0], ax[1], ax[2]

    def stage_diag(tag: str) -> None:
        bp_s, bq_s, _bx, _by, bz = back_axes()
        j7 = {s: float(arms[s].articulation.data.joint_pos[0, 6]) for s in arms}
        sp = scene.base.data.root_pos_w[0] - origins[0]
        sq = scene.base.data.root_quat_w[0]
        syaw = math.degrees(math.atan2(
            2 * (float(sq[0]) * float(sq[3]) + float(sq[1]) * float(sq[2])),
            1 - 2 * (float(sq[2]) ** 2 + float(sq[3]) ** 2)))
        print(f"[bifranka]   {tag}: back=({bp_s[0]:+.3f},{bp_s[1]:+.3f},{bp_s[2]:+.3f}) "
              f"crown_up={float(bz[2]):+.2f} seat=({sp[0]:+.3f},{sp[1]:+.3f},{sp[2]:+.3f},"
              f"yaw{syaw:+.0f}) held={scene.grasp_held[0].int().tolist()} "
              f"j7=(L{j7['left']:+.2f},R{j7['right']:+.2f})",
              flush=True)

    # ============================ phase 2: side-band grasp (RIGHT) ==========================
    # The back lies face-up behind the right robot (its only stable free rest — standing
    # tips over, measured). The right hand pinches the near SIDE band where the panel
    # lies (jaws across the shell depth, which lying face-up runs near-vertical).
    def _quat_from_cols(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """wxyz quat from an orthonormal frame given as world-frame column axes."""
        m00, m01, m02 = float(x[0]), float(y[0]), float(z[0])
        m10, m11, m12 = float(x[1]), float(y[1]), float(z[1])
        m20, m21, m22 = float(x[2]), float(y[2]), float(z[2])
        t = m00 + m11 + m22
        if t > 0:
            r = math.sqrt(1 + t)
            w = 0.5 * r
            xq = (m21 - m12) / (2 * r)
            yq = (m02 - m20) / (2 * r)
            zq = (m10 - m01) / (2 * r)
        elif m00 >= m11 and m00 >= m22:
            r = math.sqrt(1 + m00 - m11 - m22)
            xq = 0.5 * r
            w = (m21 - m12) / (2 * r)
            yq = (m01 + m10) / (2 * r)
            zq = (m02 + m20) / (2 * r)
        elif m11 >= m22:
            r = math.sqrt(1 - m00 + m11 - m22)
            yq = 0.5 * r
            w = (m02 - m20) / (2 * r)
            xq = (m01 + m10) / (2 * r)
            zq = (m12 + m21) / (2 * r)
        else:
            r = math.sqrt(1 - m00 - m11 + m22)
            zq = 0.5 * r
            w = (m10 - m01) / (2 * r)
            xq = (m02 + m20) / (2 * r)
            yq = (m12 + m21) / (2 * r)
        return torch.tensor([w, xq, yq, zq], device=device)

    def side_quat(sign: float, up: float = +1.0) -> torch.Tensor:
        """EE frame for a side-band pinch, valid for ANY panel orientation: approach
        (+z_ee) points toward the panel along -sign*bx_w; the fingers travel (+/-y_ee)
        along the shell depth by_w; y_ee's sign is chosen so the hand's flat axis x_ee
        points up-ish (up=+1) or down-ish (up=-1) — the two mirrored wrists grip
        identically and which one the arm can REACH depends on the base geometry."""
        _bp, _bq, bx_w, by_w, _bz = back_axes()
        z_ee = -sign * bx_w
        y_ee = by_w.clone()
        x_ee = torch.linalg.cross(y_ee, z_ee)
        if float(x_ee[2]) * up < 0:
            y_ee, x_ee = -y_ee, -x_ee
        return _quat_from_cols(x_ee, y_ee, z_ee)

    def grasp_side(s: str, sign: float, site: int, band_z: float,
                   approach: float | None = None, up: float = +1.0) -> bool:
        """Pinch the panel's local +x (sign=+1) or -x (sign=-1) side band at height
        `band_z` up the panel (part-local z along bz_w). `approach` picks WHICH side
        the hand comes from along +/-bx (default: same as `sign`) — on a lying panel
        both bands take a vertical-jaw pinch from either side, and the wrist only
        converges when the hand points back toward its own base (measured: the
        away-facing wrist stalled 143 deg from target). Refuses (loudly) if the band
        is outside the arm's reach — marching at an unreachable target plows the arm
        through whatever stands in between (measured: shoved the panel off the table)."""
        if approach is None:
            approach = sign
        bp_g, _bq, bx_g, by_g, bz_g = back_axes()
        edge = bp_g + bx_g * (0.19 * sign) + by_g * 0.055 + bz_g * band_z
        base_p = arms[s].articulation.data.root_pos_w[0] - origins[0]
        shoulder = base_p + torch.tensor([0.0, 0.0, 0.333], device=device)
        d_reach = float((edge - shoulder).norm())
        if d_reach > 0.62:
            print(f"[bifranka]   grasp_side({s},{sign:+.0f}) REFUSED: band at "
                  f"({edge[0]:+.3f},{edge[1]:+.3f},{edge[2]:+.3f}) is {d_reach:.2f} m "
                  f"from the shoulder (>0.62)", flush=True)
            return False
        q_side = side_quat(approach, up=up)
        set_grip(s, OPEN, settle=10)
        march(s, edge + bx_g * (0.20 * approach), q_side, steps=500, tol=0.012)

        # RE-AIM at the LIVE band through the advance: the panel drifts when a
        # fingertip grazes it (measured 4 cm during one approach — the fingers then
        # close beside the band, 70 mm in an 80 mm jaw leaves 5 mm of aim margin).
        # One retry with fresh geometry catches a shove at close time.
        def _live_edge() -> tuple[torch.Tensor, torch.Tensor]:
            bp2, _b2, bx2, by2, bz2 = back_axes()
            return bp2 + bx2 * (0.19 * sign) + by2 * 0.055 + bz2 * band_z, bx2
        for _try in range(2):
            edge, bx_l = _live_edge()
            march(s, edge + bx_l * ((0.1034 + 0.02) * approach), q_side, steps=200, tol=0.008)
            edge, bx_l = _live_edge()
            march(s, edge + bx_l * ((0.1034 - 0.004) * approach), q_side, steps=150, tol=0.005)
            set_grip(s, CLOSED, settle=40)
            if bool(scene.grasp_held[0, :, site].any()):
                break
            set_grip(s, OPEN, settle=15)
            ep_r, _ = ee_pose(s)
            march(s, ep_r + bx_l * (0.06 * approach), q_side, steps=120, tol=0.01)
        got_site = bool(scene.grasp_held[0, :, site].any())
        if not got_site:
            ep_d, eq_d = ee_pose(s)
            print(f"[bifranka]   grasp_side({s},{sign:+.0f}) MISS: ee="
                  f"({ep_d[0]:+.3f},{ep_d[1]:+.3f},{ep_d[2]:+.3f}) "
                  f"edge=({edge[0]:+.3f},{edge[1]:+.3f},{edge[2]:+.3f}) "
                  f"qerr={_quat_err(eq_d, q_side):.2f}", flush=True)
        return got_site

    # lying at yaw -90 (crown +x) the panel's local -x band is the NEAR one (world
    # y -0.19); after the carry's re-orientation to seated (body yaw 180) that band
    # maps to world +x — the right hand ends the insert on its own side. Pinch at the
    # band's CROWN end (band_z 0.50): the crown points at the right robot, so that end
    # sits 0.49 m from its shoulder (the bottom end is 0.72 m — out of reach). The
    # left joins mid-air after the carry.
    def grasp_bottom(s: str) -> bool:
        """Pinch the panel's bottom-edge lip (measured 54 mm across, the thinnest
        pinch on the part): approach along +bz (from beyond the bottom edge toward
        the crown), jaws across the shell depth."""
        bp_g, _bq, _bx_g, by_g, bz_g = back_axes()
        band = bp_g + by_g * 0.0505 + bz_g * 0.025
        base_p = arms[s].articulation.data.root_pos_w[0] - origins[0]
        shoulder = base_p + torch.tensor([0.0, 0.0, 0.333], device=device)
        d_reach = float((band - shoulder).norm())
        if d_reach > 0.62:
            print(f"[bifranka]   grasp_bottom({s}) REFUSED: lip at "
                  f"({band[0]:+.3f},{band[1]:+.3f},{band[2]:+.3f}) is {d_reach:.2f} m "
                  f"from the shoulder (>0.62)", flush=True)
            return False
        z_ee = bz_g.clone()
        y_ee = by_g.clone()
        x_ee = torch.linalg.cross(y_ee, z_ee)
        if float(x_ee[2]) < 0:
            y_ee, x_ee = -y_ee, -x_ee
        q_b = _quat_from_cols(x_ee, y_ee, z_ee)
        set_grip(s, OPEN, settle=10)
        march(s, band - bz_g * 0.20, q_b, steps=500, tol=0.012)
        march(s, band - bz_g * (0.1034 - 0.004), q_b, steps=300, tol=0.006)
        set_grip(s, CLOSED, settle=40)
        got_b = bool(scene.grasp_held[0, :, 0].any())
        if not got_b:
            ep_d, eq_d = ee_pose(s)
            print(f"[bifranka]   grasp_bottom({s}) MISS: ee="
                  f"({ep_d[0]:+.3f},{ep_d[1]:+.3f},{ep_d[2]:+.3f}) "
                  f"band=({band[0]:+.3f},{band[1]:+.3f},{band[2]:+.3f}) "
                  f"qerr={_quat_err(eq_d, q_b):.2f}", flush=True)
        return got_b

    if args.wrist_test:
        from isaaclab.utils.math import quat_from_angle_axis

        def rot_probe(s: str, q_goal: torch.Tensor, steps: int, tag: str,
                      hold_pos: bool = True) -> float:
            """Rotate toward q_goal (holding position unless hold_pos=False — the
            free variant discriminates controller failure from joint-space pose
            unreachability), printing ori err + joints periodically; returns the
            final ori err (deg)."""
            p_hold = ee_pose(s)[0].clone()
            for i in range(steps):
                p_c, q_c = ee_pose(s)
                pa = (((p_hold - p_c) / pos_scale[s]).clamp(-1.0, 1.0) if hold_pos
                      else torch.zeros(3, device=device))
                qe_c = quat_mul(q_goal.unsqueeze(0), quat_conjugate(q_c.unsqueeze(0)))[0]
                ra = (_axis_angle(qe_c) / rot_scale[s]).clamp(-ROT_STEP, ROT_STEP)
                step(1, drive=s, pos_act=pa, rot_act=ra)
                if i % 100 == 0 or i == steps - 1:
                    jp_c = arms[s].articulation.data.joint_pos[0, :7]
                    print(f"[rot] {tag} i={i}: err="
                          f"{math.degrees(_quat_err(ee_pose(s)[1], q_goal)):.0f}deg "
                          f"joints={[f'{float(v):+.2f}' for v in jp_c]}", flush=True)
            return math.degrees(_quat_err(ee_pose(s)[1], q_goal))

        # discriminator: the z-90 rotation with position HELD stalls at 74 deg.
        # Command the SAME rotation with position FREE: completion (with drift)
        # means the pose PAIR was joint-unreachable, not the torque law.
        p_home, q_home = ee_pose("left")
        axz = torch.tensor([0.0, 0.0, 1.0], device=device)
        q_goal_z = quat_mul(quat_from_angle_axis(
            torch.tensor([math.pi / 2], device=device), axz.unsqueeze(0)),
            q_home.unsqueeze(0))[0]
        e_free = rot_probe("left", q_goal_z, 300, "z-FREE", hold_pos=False)
        p_drift = ee_pose("left")[0]
        print(f"[wrist] pure-rot z FREE-POSITION: final={e_free:.0f}deg "
              f"drift={float((p_drift - p_home).norm()) * 1000:.0f}mm", flush=True)
        rot_probe("left", q_home, 250, "z-back", hold_pos=False)
        march("left", p_home, q_home, steps=250, tol=0.02)

        # measured grasp probe for the RIGHT arm at the CURRENT staging: for each
        # knuckle branch, march to the pre-pose position first (orientation free),
        # rotate in place (the decomposition that beats coupled stalls), then
        # advance + close and report the WELD. The winner gets baked.
        for up_t in (+1.0, -1.0):
            bp_t, _bq_t, bx_t, by_t, bz_t = back_axes()
            edge_t = bp_t + bx_t * (0.19 * -1.0) + by_t * 0.055 + bz_t * 0.28
            q_t = side_quat(-1.0, up=up_t)
            pre_t = edge_t + bx_t * (0.20 * -1.0)
            set_grip("right", OPEN, settle=10)
            march("right", pre_t, None, steps=300, tol=0.015)
            oe = rot_probe("right", q_t, 350, f"right-up{up_t:+.0f}")
            print(f"[wrist] right east-band up={up_t:+.0f}: ori_err={oe:.0f}deg", flush=True)
            if oe < 25.0:
                march("right", edge_t + bx_t * ((0.1034 - 0.004) * -1.0), q_t,
                      steps=300, tol=0.006)
                set_grip("right", CLOSED, settle=40)
                got_t = bool(scene.grasp_held[0, :, 2].any())
                print(f"[wrist] right pinch up={up_t:+.0f}: WELD={got_t}", flush=True)
                if got_t:
                    break
                set_grip("right", OPEN, settle=15)
            ep_t, _ = ee_pose("right")
            march("right", ep_t + torch.tensor([0.15, 0.05, 0.10], device=device),
                  None, steps=200, tol=0.02)
        if frames:
            arr = np.stack(frames, axis=0)
            np.savez_compressed(args.out, frames=arr, env="assembly.chair.multi.osc")
            print(f"[bifranka] saved {arr.shape} -> {args.out}", flush=True)
        print("WRIST_TEST_DONE", flush=True)
        return  # no env.close(): it hangs this build's teardown; _hard_exit handles it

    # TWO-ARM head-on pick (the user's layout): the panel lies crown toward the front
    # rim, so its side bands face the arms — each approach is the arm's natural
    # facing, and the finger-vertical roll is about the approach axis (joint 7, the
    # rotation this controller always tracks). Pinches at band_z 0.28 — near the
    # Left takes the west band (site 1), right the east (site 2).
    # AT THE RACK only the LEFT can grasp: the right's east-band orientation there is
    # kinematically unreachable from its base — measured, both knuckle branches, free
    # hand, position-then-rotate (60 and 145 deg residuals). The right joins later at
    # the table centre, where its identical grasp PASSED in an earlier run.
    # band_z 0.28 = the CoM band, the ONE left-graspable height on this shell: the
    # low band (0.10) sits in the shell's bottom curl, whose edge tangent twists
    # the demanded wrist roll past the left's branch (qerr 2.19, measured MISS);
    # 0.32's 72 mm edge stalls exactly at the release window top.
    ok_l = grasp_side("left", +1.0, 1, band_z=0.28)
    stage_diag("left-grasp")
    check("grasp: left hand pinches the west side band", ok_l)
    ok = ok_l

    # ============================ phase 3: solo lift + carry (RIGHT) ========================
    # The weld is rigid, so one pinch carries honestly. The servo drives EVERY hand
    # currently holding a panel band with the panel's LIVE pose error each step; targets
    # are computed from the LIVE chair base (it scoots; track it, don't brace it).
    def base_frame_target(off_xyz, yaw_q=None):
        bp_b = scene.base.data.root_pos_w[0] - origins[0]
        bq_b = scene.base.data.root_quat_w[0]
        p = bp_b + quat_apply(bq_b.unsqueeze(0), torch.tensor([off_xyz], device=device))[0]
        q = quat_mul(bq_b.unsqueeze(0), (yaw_q if yaw_q is not None else ident).unsqueeze(0))[0]
        return p, q

    def panel_servo(off_xyz, steps=600, tol=0.005, ori_tol=0.06, q_abs=None) -> bool:
        """Drive every hand welded to a panel band with a RIGID-BODY-consistent step:
        shared translation PLUS each hand's own rotation arc (dtheta x lever). The
        earlier same-translation-for-both version made the position-locked hands
        structurally fight any commanded rotation — measured: the erect stalled at
        16 of 90 deg with both welds holding."""
        for _i in range(steps):
            p_tgt, q_tgt = base_frame_target(off_xyz)
            if q_abs is not None:
                q_tgt = q_abs
            bp_l, bq_l = body_pose("back")
            err_p = p_tgt - bp_l
            qe = quat_mul(q_tgt.unsqueeze(0), quat_conjugate(bq_l.unsqueeze(0)))[0]
            err_r = _axis_angle(qe)
            if float(err_p.norm()) < tol and float(err_r.norm()) < ori_tol:
                return True
            step_p = err_p.clamp(-MAX_STEP, MAX_STEP)
            # UPWARD lead may run ahead of the live error clamp: the task-space
            # spring is F = kp * gap (measured at kp 100 and 500), so a lead capped
            # at MAX_STEP=8 mm tops out at ~4 N — structurally unable to lift the
            # 2.5 kg panel (measured: 900 steps, zero climb). 0.06 m of upward lead
            # buys ~30 N at kp=500; descent keeps the tight clamp.
            step_p[2] = float(err_p[2].clamp(-MAX_STEP, 0.06))
            mag_r = float(err_r.norm())
            dth = err_r * min(1.0, (ROT_STEP * 0.097) / mag_r) if mag_r > 1e-6 \
                else torch.zeros(3, device=device)  # the panel's per-step rotation vector
            drives = {}
            # hands are ordered (left=0, right=1); panel side bands are sites 1..2
            for s, hand_i in (("left", 0), ("right", 1)):
                if scene.grasp_held[0, hand_i, 1:3].any():
                    r_h = ee_pose(s)[0] - bp_l
                    arc = torch.linalg.cross(dth, r_h)
                    drives[s] = (((step_p + arc) / pos_scale[s]).clamp(-1.0, 1.0),
                                 (err_r / rot_scale[s]).clamp(-ROT_STEP, ROT_STEP))
            if not drives:
                return False
            step(1, drives=drives)
        p_tgt, _ = base_frame_target(off_xyz)
        return float((p_tgt - body_pose("back")[0]).norm()) < tol * 2

    # ERECT with a BRAKE: swing the band north-up about the friction-pinned base at
    # ~0.5 deg/step (quasi-static), watching the crown EVERY step and stopping the
    # instant it points up. The general-purpose servo already erected the panel
    # through vertical twice — its arc term is geometrically right — but nothing
    # told it to stop: it kept pushing past vertical and gravity dumped the panel
    # north onto the seat (crown 0.87 -> 0.18, both measured runs). An erect is an
    # ORIENTATION goal; it needs an orientation stop, not a position tolerance.
    def erect_servo(steps: int = 900) -> bool:
        z_up_l = torch.tensor([0.0, 0.0, 1.0], device=device)
        for _i in range(steps):
            bp_e, _bq_e = body_pose("back")
            crown_e = back_axes()[4]
            if float(crown_e[2]) > 0.97:
                return True
            ax_e = torch.linalg.cross(crown_e, z_up_l)
            n_e = float(ax_e.norm())
            if n_e < 1e-6:
                return True
            ax_e = ax_e / n_e
            ang_e = float(torch.arccos(crown_e[2].clamp(-1.0, 1.0)))
            dth_e = ax_e * min(ang_e, 0.009)  # ~0.5 deg/step: friction damps, no momentum
            drives_e = {}
            for s_e, hand_e in (("left", 0), ("right", 1)):
                if scene.grasp_held[0, hand_e, 1:3].any():
                    r_e = ee_pose(s_e)[0] - bp_e
                    arc_e = torch.linalg.cross(dth_e, r_e)
                    # leads x12 rot / x4 arc: at x3 the wrist's command lead topped
                    # out at 2.4 Nm against the 30-deg lean's 3.4 Nm gravity moment
                    # — 900 steps, zero rotation, measured. The larger leads put
                    # ~8 Nm behind the swing; the per-step crown brake still stops
                    # it (velocity damping caps the run-on at a degree or two).
                    drives_e[s_e] = ((arc_e * 4.0 / pos_scale[s_e]).clamp(-1.0, 1.0),
                                     (dth_e * 12.0 / rot_scale[s_e]).clamp(-ROT_STEP, ROT_STEP))
            if not drives_e:
                return False
            step(1, drives=drives_e)
        return float(back_axes()[4][2]) > 0.9
    # The staging sits directly SOUTH OF THE SEAT, so the erect's base skid (the
    # base slides north under the swing — friction alone cannot pin it, measured
    # 25 cm at the far staging) runs INTO the seat's south face, which anchors the
    # pivot and self-aligns the erected panel at the studs. No drag phase: the
    # panel erects in the work zone, which is also the right arm's proven grasp
    # zone.
    ok = erect_servo() and ok
    stage_diag("erect")
    check("erect: panel upright at the seat face", float(back_axes()[4][2]) > 0.9)
    bq_now = body_pose("back")[1]
    crown_w = back_axes()[4]
    z_up = torch.tensor([0.0, 0.0, 1.0], device=device)
    pitch_axis = torch.linalg.cross(crown_w, z_up)
    pitch_axis = pitch_axis / pitch_axis.norm().clamp_min(1e-9)
    pitch_ang = float(torch.arccos(crown_w[2].clamp(-1.0, 1.0)))
    q_pitch = torch.cat([torch.tensor([math.cos(pitch_ang / 2)], device=device),
                         pitch_axis * math.sin(pitch_ang / 2)])
    q_upright = quat_mul(q_pitch.unsqueeze(0), bq_now.unsqueeze(0))[0]
    print(f"[bifranka]   erect plan: pitch {math.degrees(pitch_ang):.0f}deg about "
          f"({pitch_axis[0]:+.2f},{pitch_axis[1]:+.2f},{pitch_axis[2]:+.2f})", flush=True)
    if float(crown_w[2]) > 0.9:
        # upright and ANCHORED at the seat face — SOLO WALL-SLIDE INSERT: press
        # the panel gently north into the seat/stud plane and lift; the wall
        # supplies the bracing moment a lone side grip cannot (a free solo carry
        # toppled 88 deg, measured — the seat is the second hand). The z target
        # overshoots by the arm's measured ~5 cm payload sag at kp=500 so the
        # realized height reaches the studs; x is live-corrected by the servo.
        print("[bifranka]   erect: upright at the seat — wall-slide insert", flush=True)
        p_dbg, _q_dbg = base_frame_target((0.0, c.back_seat_pos[1], c.back_seat_pos[2]))
        print(f"[bifranka]   seated target (world): "
              f"({p_dbg[0]:+.3f},{p_dbg[1]:+.3f},{p_dbg[2]:+.3f})", flush=True)
        # SECOND HAND before the climb: the erected panel stands at the seat
        # front — the one zone where the right's east-band pinch has a measured
        # PASS. Every solo-climb failure is unbraced rotation about the single
        # grip (peel-off at 37 deg, drift, j7 windup — measured three times);
        # two position-locked hands on opposite edges are the brace.
        got_r = grasp_side("right", -1.0, 2, band_z=0.28)
        stage_diag("right-join")
        check("join: right hand takes the east band at the seat", got_r)
        ok = ok and got_r
        # RIDE THE STUD TIPS. The seated y lives at SLAB height — at table level
        # that same y is deep inside the leg footprint, so pressing toward it
        # plowed the whole 5 kg chair (19 cm, then clear across the table,
        # measured twice). The real rail: the two stud tips poke 9 cm south of
        # the slab face (world y ~-0.01, z 0.647), exactly where the erect
        # naturally parks the panel. Climb pressed LIGHTLY against the tips —
        # a two-point rail that braces pitch and yaw — until the holes reach
        # stud height, then drive the holes 7.6 cm north along the shanks.
        panel_servo((0.0, c.back_seat_pos[1] + 0.074, c.back_seat_pos[2]),
                    steps=600, tol=0.010, ori_tol=0.30)
        stage_diag("tip-climb")
        panel_servo((0.0, c.back_seat_pos[1] + 0.0015, c.back_seat_pos[2] + 0.002),
                    steps=400, tol=0.003, ori_tol=0.30)
        stage_diag("stud-ride")
        set_grip("left", OPEN, settle=25)
        for _ in range(12):
            if bool(scene.welded[0, 0]):
                break
            step(15)
        stage_diag("insert-settle")
        check("insert: backrest welds to the base", bool(scene.welded[0, 0]))
        ok = ok and bool(scene.welded[0, 0])
        report("insert")
    else:
        # No arm-powered erect exists: every welded-wrist rotation beyond ~25 deg
        # stalls against this controller (measured across 8 variants). If the drag
        # didn't deliver the panel upright, the run is over — fail fast instead of
        # burning the budget on a recovery that cannot work.
        check("drag: panel arrives upright (no recovery path)", False)
        ok = False
    if args.pick_only:
        if frames:
            arr = np.stack(frames, axis=0)
            np.savez_compressed(args.out, frames=arr, env="assembly.chair.multi.osc")
            print(f"[bifranka] saved {arr.shape} -> {args.out}", flush=True)
        n_ok0 = sum(okc for _n, okc in checks)
        print(f"[bifranka] PICK-ONLY RESULT: {n_ok0}/{len(checks)} checks", flush=True)
        print("CHAIR_BIFRANKA_SMOKE_DONE", flush=True)
        return
    ok = panel_servo((0.0, c.back_seat_pos[1] + 0.06, c.back_seat_pos[2] + 0.002),
                     steps=500, tol=0.006) and ok
    panel_servo((0.0, c.back_seat_pos[1] + 0.0015, c.back_seat_pos[2] + 0.002),
                steps=500, tol=0.003)
    bp_i, _ = body_pose("back")
    p_tgt_i, _ = base_frame_target((0.0, c.back_seat_pos[1], c.back_seat_pos[2]))
    print(f"[bifranka]   insert diag: back=({bp_i[0]:+.4f},{bp_i[1]:+.4f},{bp_i[2]:+.4f}) "
          f"target=({p_tgt_i[0]:+.4f},{p_tgt_i[1]:+.4f},{p_tgt_i[2]:+.4f}) "
          f"err={[f'{float(bp_i[j] - p_tgt_i[j]) * 1000:+.1f}' for j in range(3)]}mm "
          f"back_seated={bool(scene._back_seated()[0])}", flush=True)
    set_grip("right", OPEN, settle=25)
    set_grip("left", OPEN, settle=25)
    for _ in range(12):
        if bool(scene.welded[0, 0]):
            break
        step(15)
    report("insert")
    check("insert: backrest welds to the base", bool(scene.welded[0, 0]))
    for s, dx in (("right", 0.20), ("left", -0.20)):  # retreat toward own base, up+back
        ep_s, _ = ee_pose(s)
        march(s, ep_s + torch.tensor([dx, 0.05, 0.10], device=device), steps=250, tol=0.02)

    # ============================ phase 5: nut runs =========================================
    def stud_frame() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Live chair base pose + the stud AXIS in world (base-frame +y)."""
        bq_b = scene.base.data.root_quat_w[0]
        bp_b = scene.base.data.root_pos_w[0] - origins[0]
        axis_w = quat_apply(bq_b.unsqueeze(0),
                            torch.tensor([[0.0, 1.0, 0.0]], device=device))[0]
        return bp_b, bq_b, axis_w

    def thread_nut(s: str, k: int, sx: float) -> None:
        """Arm `s` fetches nut k (scene index) and threads it onto the stud at base-frame
        x=`sx` (the mechanic engages the nearest stud; welds are order-independent). All
        stud-relative geometry is composed through the LIVE base frame."""
        nut_name = f"nut_{k}"
        np_, nq = body_pose(nut_name)
        hover = torch.tensor([0.0, 0.0, 0.22], device=device)
        pinch = torch.tensor([0.0, 0.0, 0.1034 + 0.0165 - 0.004], device=device)
        march(s, np_ + hover, down)
        march(s, np_ + pinch, down, steps=300, tol=0.005)
        set_grip(s, CLOSED, settle=50)
        got = bool(scene.grasp_held[0, :, 3 + k].any())  # nut sites follow the 3 back bands
        check(f"nut {nut_name}: {s} hand welds it", got)
        if not got:
            return
        # rel transform hand<->nut, then present the nut to its stud tip
        ep2, eq2 = ee_pose(s)
        np2, nq2 = body_pose(nut_name)
        r_p = quat_apply_inverse(nq2.unsqueeze(0), (ep2 - np2).unsqueeze(0))[0]
        r_q = quat_mul(quat_conjugate(nq2.unsqueeze(0)), eq2.unsqueeze(0))[0]
        qx = torch.tensor([math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0.0, 0.0],
                          device=device)  # nut local +z -> base +y (the stud axis)

        def ee_for_nut(nut_p, nut_q):
            p = torch.as_tensor(nut_p, dtype=torch.float32, device=device)
            q = torch.as_tensor(nut_q, dtype=torch.float32, device=device)
            return (p + quat_apply(q.unsqueeze(0), r_p.unsqueeze(0))[0],
                    quat_mul(q.unsqueeze(0), r_q.unsqueeze(0))[0])

        def tip_pose(back_along: float) -> tuple[torch.Tensor, torch.Tensor]:
            """Nut target `back_along` metres BEFORE the thread tip, axis-aligned —
            position AND orientation composed from the live base."""
            p, q = base_frame_target((sx, c.thread_y1 + back_along, c.stud_z), yaw_q=qx)
            return p, q

        np2u = np2 + torch.tensor([0.0, 0.0, 0.20], device=device)
        up_p, up_q = ee_for_nut(np2u, nq2)
        march(s, up_p, up_q, steps=350, tol=0.02)
        t_p, t_q = tip_pose(0.05)
        app_p, app_q = ee_for_nut(t_p, t_q)
        march(s, app_p, app_q, steps=700, tol=0.008)
        t_p, t_q = tip_pose(0.001)
        eng_p, eng_q = ee_for_nut(t_p, t_q)
        march(s, eng_p, eng_q, steps=300, tol=0.005)
        for _ in range(10):
            if bool(scene._scr_eng[0, k]):
                break
            step(10)
        check(f"nut {nut_name}: screw joint engages", bool(scene._scr_eng[0, k]))
        if not bool(scene._scr_eng[0, k]):
            return  # no engagement -> the 34-cycle budget is pure waste (measured: hours)
        # regrip spin cycles: roll about the LIVE stud axis while pinched, release, unwind
        for cyc in range(args.max_cycles):
            if bool(scene.welded[0, 1 + k]):
                break
            axis_w = stud_frame()[2]
            p0, q0 = ee_pose(s)
            n_sub = int(SPIN_PER_CYCLE / (rot_scale[s] * ROT_STEP)) + 6
            half = -SPIN_PER_CYCLE / (2 * n_sub)  # screw-in sense
            dq = torch.cat([torch.tensor([math.cos(half)], device=device),
                            axis_w * math.sin(half)])
            qt = q0.clone()
            for _j in range(n_sub):
                qt = quat_mul(dq.unsqueeze(0), qt.unsqueeze(0))[0]
                pa = ((p0 - ee_pose(s)[0]) / pos_scale[s]).clamp(-1.0, 1.0)
                qe = quat_mul(qt.unsqueeze(0), quat_conjugate(ee_pose(s)[1].unsqueeze(0)))[0]
                ra = (_axis_angle(qe) / rot_scale[s]).clamp(-ROT_STEP, ROT_STEP)
                step(1, drive=s, pos_act=pa, rot_act=ra)
            set_grip(s, OPEN, settle=20)  # the mechanic holds the nut; unwind the wrist
            p1, q1 = ee_pose(s)
            back_off = p1 + quat_apply(q1.unsqueeze(0), torch.tensor(
                [[0.0, 0.0, -0.03]], device=device))[0]
            march(s, back_off, q0, steps=200, tol=0.01)
            # re-approach the (advanced) nut and re-pinch along the stud axis
            np3, nq3 = body_pose(nut_name)
            re_p, re_q = ee_for_nut(np3, nq3)
            march(s, re_p + stud_frame()[2] * 0.012, re_q, steps=200, tol=0.006)
            march(s, re_p, re_q, steps=150, tol=0.004)
            set_grip(s, CLOSED, settle=30)
            if not bool(scene.grasp_held[0, :, 3 + k].any()):
                print(f"[bifranka]   cycle {cyc}: re-pinch missed, retrying", flush=True)
                set_grip(s, OPEN, settle=15)
                march(s, re_p, re_q, steps=120, tol=0.003)
                set_grip(s, CLOSED, settle=30)
            if cyc % 4 == 0:
                loc = scene._nut_rel_base()[0, k]
                print(f"[bifranka]   {nut_name} cycle {cyc}: depth="
                      f"{(c.thread_y1 - float(loc[1])) * 1000:+.1f}mm "
                      f"turn={float(scene._scr_turn[0, k]):.1f}rad", flush=True)
        set_grip(s, OPEN, settle=20)
        ep3, _ = ee_pose(s)
        march(s, ep3 + stud_frame()[2] * 0.15
              + torch.tensor([0.0, 0.0, 0.12], device=device))
        check(f"nut {nut_name}: threads home and welds", bool(scene.welded[0, 1 + k]))

    # body at yaw 180: base-frame -x maps to world +x, so the RIGHT arm's stud (world
    # +x, fed by nut_0 on its side) is base-frame -stud_x; mirrored for the left
    thread_nut("right", k=0, sx=-c.stud_x)
    thread_nut("left", k=1, sx=c.stud_x)

    # ============================ verdict ===================================================
    report("final")
    check("success: all 3 pairs assembled", bool(scene.success()[0]))
    check("zero ordering violations", int(scene.order_violations[0]) == 0)
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="assembly.chair.multi.osc")
        print(f"[bifranka] saved {arr.shape} -> {args.out}", flush=True)
    n_ok = sum(okc for _n, okc in checks)
    print(f"[bifranka] RESULT: {'ALL PASS' if n_ok == len(checks) else 'FAIL'} "
          f"({n_ok}/{len(checks)})", flush=True)
    print("CHAIR_BIFRANKA_SMOKE_DONE", flush=True)
    env.close()


def _hard_exit_teardown() -> None:
    import os as _os
    import threading as _threading

    watchdog = _threading.Timer(10.0, lambda: _os._exit(0))
    watchdog.daemon = True
    watchdog.start()
    app.close()
    _os._exit(0)


if __name__ == "__main__":
    main()
    _hard_exit_teardown()

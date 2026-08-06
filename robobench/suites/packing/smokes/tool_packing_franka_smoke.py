"""Franka-arm solution smoke for ToolPackingScene — a REAL manipulation solve, RECORDED.

Where `tool_packing_smoke.py` is the NullRobot oracle (joint writes + kinematic carries
validating the scene rules), this drives the `packing.tool_packing.franka.osc` binding
through the whole pack with the ARM ONLY — no teleports, no joint writes:

  1. crack the RIGHT door: the two closed door edges meet at box centre with a 3 mm gap
     and the handle sits flush (nothing to hook), so the closed fingertips press the
     handle's END FACE and sweep toward the hinge — the tangential push swings the door
     out ~30 deg — then pinch the now-proud free edge and arc-pull it to ~110 deg;
  2. the LEFT door's edge is exposed once the right door is away: pinch + arc-pull;
  3. per drawer top->mid->bot: pinch the tray's front-wall top edge, pull the slide out,
     top-down pick the drawer's item off the table (pinch across its short axis), carry it
     over the EXPOSED tray span, lower, release, then push the drawer shut by its face;
  4. close both doors by pushing their outer faces near the free edge, following the arc;
  5. assert the score staircase 30/60/90 -> 100 and scene success().

Waypoint control: each control step sends the OSC action `clamp((target-ee)/pos_scale)` +
the axis-angle orientation error over `rot_scale` (the microwave-franka skeleton). All
world targets address the PINCH POINT (hand origin + 0.1034 m along hand z).

    python -m robobench.suites.packing.smokes.tool_packing_franka_smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--record_every", type=int, default=4)
parser.add_argument("--only_doors", action="store_true",
                    help="debug clip: run ONLY the two door-open skills, record densely, exit")
parser.add_argument("--out", type=str, default="tool_packing_franka_frames.npz")
parser.add_argument("--hdfs_dir", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import json
import math
import os

import numpy as np
import torch

import robobench
from robobench.core import ENVS

FAILS: list[str] = []
PINCH = 0.1034
GRIP_OPEN, GRIP_SHUT = 0.04, 0.0

# ----- box-local geometry, measured from the vendored baked asset -------------------------------
# doors: hinge pins and panel spans in the BOX body frame (z up, -y = front face)
DOOR_L_HINGE = (-0.2229, -0.1292)  # left door link origin = its hinge (lp1 = 0)
DOOR_R_HINGE = (0.2287, -0.1348)  # right hinge = link origin + lp1 x offset 0.2572
DOOR_LEN = 0.22  # hinge -> free edge
DOOR_Z_MID = 0.13
# right-door handle: bar x in [0, 0.053] door-local from the free-edge end, z 0.1125..0.1321
HANDLE_END_BOX = (-0.030, -0.140, 0.122)  # the bar's centre-gap END FACE, box frame
# drawers: front-wall top-edge pinch point, drawer-local (frame y=-0.0993 from box centre)
TRAY_LIP_LOCAL = (0.0, -0.0015, 0.040)  # x-centre, wall mid-thickness, just under the rim
TRAY_RIM_Z = 0.0471
TRAY_FLOOR_Z = -0.0154
DROP_LOCAL_Y = 0.09  # drop over the exposed tray span (the null-smoke value)
# items: top-down pinch points (item-local) + hand yaw so the gap spans the SHORT axis,
# and the long axis lands ACROSS the tray at the place pose
ITEM_GRASP = {  # name: (local grasp point, gap-axis ('x'|'y'), long-axis half-length)
    "stapler": ((0.0, 0.0, 0.034), "y", 0.083),
    "scissors": ((0.0, -0.02, 0.007), "x", 0.111),
    "knife": ((0.0, -0.02, 0.010), "x", 0.097),
}


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[tp-franka] {'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def quat_from_axes(z: torch.Tensor, x_hint: torch.Tensor) -> torch.Tensor:
    """wxyz quat with local +z = `z`, local +x = `x_hint` orthogonalised (Shepperd)."""
    z = z / z.norm()
    x = x_hint - (x_hint @ z) * z
    x = x / x.norm()
    y = torch.cross(z, x, dim=-1)
    m = [[float(v) for v in row] for row in torch.stack([x, y, z], dim=-1)]
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = (s / 4, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    elif m[0][0] >= m[1][1] and m[0][0] >= m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        q = ((m[2][1] - m[1][2]) / s, s / 4, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
    elif m[1][1] >= m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        q = ((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, s / 4, (m[1][2] + m[2][1]) / s)
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        q = ((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, s / 4)
    return torch.tensor(q, dtype=torch.float32)


def main() -> None:  # noqa: PLR0915
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("packing.tool_packing.franka.osc")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    art = env.robot.articulation
    ee_i = art.find_bodies(env.robot.EE_BODY)[0][0]
    origin = env.iscene.env_origins[0]
    osc = env.robot.controller.controllers[0]
    pos_scale, rot_scale = osc.cfg.pos_scale, osc.cfg.rot_scale
    grip_ids = env.robot.controller.controllers[-1].joint_ids
    from isaaclab.utils.math import (
        axis_angle_from_quat,
        quat_apply,
        quat_apply_inverse,
        quat_conjugate,
        quat_mul,
    )

    # --- recording ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = origin.detach().cpu().numpy().astype(float)
        wx, wy = c.workbench_pos
        look = np.array((wx + c.box_pos[0] - 0.18, wy + c.box_pos[1], c.surface_z + 0.16))
        env.sim.set_camera_view(tuple(np.array((wx - 0.85, wy - 0.65, c.surface_z + 0.65)) + o),
                                tuple(look + o), camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        print(f"[tp-franka] camera ready shape={np.asarray(annot.get_data()).shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[tp-franka] camera setup FAILED ({exc!r})", flush=True)

    step_i = 0
    phases: list[tuple[int, str]] = []

    def phase(label: str) -> None:
        phases.append((step_i, label))
        print(f"[tp-franka] PHASE @{step_i}: {label}", flush=True)

    def hand_pose() -> tuple[torch.Tensor, torch.Tensor]:
        st = art.data.body_link_state_w[0, ee_i]
        return st[0:3].clone(), st[3:7].clone()

    def pinch_pos(hp: torch.Tensor, hq: torch.Tensor) -> torch.Tensor:
        off = torch.tensor([0.0, 0.0, PINCH], device=device)
        return hp + quat_apply(hq.unsqueeze(0), off.unsqueeze(0))[0]

    def step_action(dpos: torch.Tensor, drot: torch.Tensor, grip: float) -> None:
        nonlocal step_i
        a = torch.cat([dpos, drot, torch.full((2,), grip, device=device)]).unsqueeze(0)
        env.step(a, render=True)
        if annot is not None and step_i % args.record_every == 0:
            for _ in range(3):
                env.sim.render()
            arr = np.asarray(annot.get_data())
            if arr.size:
                frames.append(arr[..., :3].astype(np.uint8).copy())
        step_i += 1

    def goto(target_w: torch.Tensor, tquat: torch.Tensor | None, grip: float,
             tol: float = 0.012, max_steps: int = 220, settle: int = 0) -> bool:
        tq = None if tquat is None else tquat.to(device)
        rerr = 0.0
        for _ in range(max_steps):
            hp, hq = hand_pose()
            err = target_w.to(device) - pinch_pos(hp, hq)
            drot = torch.zeros(3, device=device)
            if tq is not None:
                q_t = torch.where((tq * hq).sum() >= 0, tq, -tq)
                aa = axis_angle_from_quat(quat_mul(q_t.unsqueeze(0),
                                                   quat_conjugate(hq.unsqueeze(0))))[0]
                rerr = float(aa.norm())
                drot = (aa / rot_scale).clamp(-1.0, 1.0)
            if float(err.norm()) < tol and rerr < 0.08:
                break
            step_action((err / pos_scale).clamp(-1.0, 1.0), drot, grip)
        for _ in range(settle):
            hp, hq = hand_pose()
            err = target_w.to(device) - pinch_pos(hp, hq)
            step_action((err / pos_scale).clamp(-1.0, 1.0), torch.zeros(3, device=device), grip)
        hp, hq = hand_pose()
        resid = float((target_w.to(device) - pinch_pos(hp, hq)).norm())
        ok = resid < tol * 1.6
        if not ok:
            print(f"[tp-franka]   goto MISSED: resid={resid * 1000:.1f}mm rot_err={rerr:.3f} "
                  f"target={[round(float(v), 3) for v in target_w.tolist()]}", flush=True)
        return ok

    def hold(grip: float, k: int = 8) -> None:
        for _ in range(k):
            step_action(torch.zeros(3, device=device), torch.zeros(3, device=device), grip)

    def aperture() -> float:
        return float(art.data.joint_pos[0, grip_ids].sum())

    # --- frames + anchors --------------------------------------------------------------------
    def box_pose() -> tuple[torch.Tensor, torch.Tensor]:
        b = scene.box.body_names.index(c.box_body)
        return scene.box.data.body_pos_w[0, b].clone(), scene.box.data.body_quat_w[0, b].clone()

    def BW(x: float, y: float, z: float) -> torch.Tensor:
        """Box-local -> world."""
        bp, bq = box_pose()
        v = torch.tensor([x, y, z], device=device)
        return bp + quat_apply(bq.unsqueeze(0), v.unsqueeze(0))[0]

    def box_axes() -> tuple[torch.Tensor, torch.Tensor]:
        """(front_dir, right_dir) of the box in world: front = -y_local, right = +x_local."""
        _bp, bq = box_pose()
        f = quat_apply(bq.unsqueeze(0), torch.tensor([[0.0, -1.0, 0.0]], device=device))[0]
        r = quat_apply(bq.unsqueeze(0), torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
        return f, r

    ez = torch.tensor([0.0, 0.0, 1.0])

    def _nearest_grip_quat(approach: torch.Tensor, gap: torch.Tensor) -> torch.Tensor:
        """The grip is symmetric under gap sign AND a pi roll: build all four equivalent
        quats and return the one nearest the CURRENT hand orientation. Kills every
        avoidable wrist fight (runs 1-3: rot_err pinned at 1-3 rad on poses whose mirror
        twin was trivially reachable)."""
        a = (approach / approach.norm()).cpu()
        g0 = (gap / gap.norm()).cpu()
        g0 = g0 - (g0 @ a) * a
        g0 = g0 / g0.norm()
        _hp, hq = hand_pose()
        hqc = hq.cpu()
        best, best_d = None, 1e9
        for g in (g0, -g0):
            x_hint = torch.cross(g, a, dim=-1)
            q = quat_from_axes(a, x_hint)
            d = 1.0 - abs(float((q * hqc).sum()))
            if d < best_d:
                best, best_d = q, d
        return best

    def q_topdown(gap_world: torch.Tensor) -> torch.Tensor:
        """Top-down hand pose whose FINGER GAP spans `gap_world` (gap = hand local y)."""
        return _nearest_grip_quat(-ez, gap_world.cpu())

    def q_oblique(gap_world: torch.Tensor, tilt_deg: float = 32.0) -> torch.Tensor:
        """Approach tilted `tilt_deg` off vertical toward +x (the reach direction): at the
        box face (0.45-0.60 m out) the wrist cannot hold a pure top-down pose, but the
        microwave-proven tilted family tracks fine. Gap = hand local y spans `gap_world`
        (projected off the approach axis)."""
        t = math.radians(tilt_deg)
        a = torch.tensor([math.sin(t), 0.0, -math.cos(t)])
        return _nearest_grip_quat(a, gap_world.cpu())

    TIP = 0.022  # pinch centre -> fingertip, along the approach axis

    def tip_lift(tilt_deg: float = 32.0) -> float:
        """Raise a pinch target by this so the fingerTIPS (not the pads' centre) land on a
        feature: run 3's crack sweep ran 7 mm BELOW the handle on flat glass."""
        return TIP * math.cos(math.radians(tilt_deg))

    def door_angles() -> tuple[float, float]:
        dp = scene.door_pos()[0]
        return float(dp[0]), float(dp[1])

    def drawer_val(d: int) -> float:
        return float(scene.drawer_pos()[0, d])

    # --- skills --------------------------------------------------------------------------------
    def crack_right_door() -> bool:
        """Press the flush handle's centre-gap END FACE with the closed fingertips (hand
        top-down, fingers hanging in front of the face) and sweep toward the hinge: the
        tangential push swings the door out. Nothing is pinchable while both doors are
        shut (3 mm edge gap, handle flush)."""
        phase("crack the right door (fingertip sweep on the handle end)")
        front, right = box_axes()
        q = q_oblique(front)  # pads face along the front axis; sideways sweep is stiff
        # Two z-passes bracket the handle band; SLOW deep-pressed waypoints — runs 4-8 swept
        # with 28-step waypoints that never converged, so the tips never even touched the
        # glass (40-90 mm residuals all through the "sweep").
        for dz in (0.0, 0.008):
            end = BW(*HANDLE_END_BOX) + front * 0.006 + ez.to(device) * (tip_lift() + dz)
            goto(end + ez.to(device) * 0.10, q, GRIP_SHUT, tol=0.02, max_steps=90)
            goto(end - right * 0.020, q, GRIP_SHUT, tol=0.006, max_steps=120, settle=4)
            for k in range(1, 8):
                tgt = end + right * (0.022 * k) - front * 0.025
                goto(tgt, q, GRIP_SHUT, tol=0.012, max_steps=90)
                if door_angles()[1] > math.radians(18.0):
                    break
            if door_angles()[1] > math.radians(18.0):
                break
        hp, hq = hand_pose()
        goto(pinch_pos(hp, hq) + front * 0.10 + ez.to(device) * 0.10, q, GRIP_SHUT,
             tol=0.03, max_steps=50)
        ang = math.degrees(door_angles()[1])
        print(f"[tp-franka]   right door cracked to {ang:.1f} deg", flush=True)
        return ang > 15.0

    def pull_door_open(side: str, target_deg: float = 88.0) -> bool:
        # 88 deg, not more: past ~95 deg the panel plane leans INTO the drawer channel
        # (the hinges sit only ~2 cm outside it) and the drawers would jam on the glass.
        """Pinch the door's free edge and arc-pull around its hinge. `side` in {left,right}."""
        phase(f"pull the {side} door open to ~{target_deg:.0f} deg")
        sgn = 1.0 if side == "right" else -1.0
        hinge = DOOR_R_HINGE if side == "right" else DOOR_L_HINGE
        jcol = 1 if side == "right" else 0
        bp, bq = box_pose()
        hinge_w = BW(hinge[0], hinge[1], DOOR_Z_MID)
        ang0 = door_angles()[jcol]
        front, right = box_axes()
        # free-edge point, world (door rotated ang0 about its hinge, z axis)
        edge_r = DOOR_LEN - 0.012

        def edge_at(ang: float) -> torch.Tensor:
            # door-local edge dir at angle: rotate the box-frame closed-pose edge direction.
            # BOTH doors' edges swing toward the FRONT when opening (run 7's left waypoints
            # walked backward into the box).
            base = -right if side == "right" else right  # from hinge toward centre
            ca, sa = math.cos(ang), math.sin(ang)
            dirw = base * ca + front * sa
            return hinge_w + dirw * edge_r

        # TOP-DOWN edge pinch: fingers descend straddling the plate near its free edge
        # (one pad in front of the panel, one behind — 17 mm of clearance to the drawer
        # faces inside), gap spanning the door NORMAL at its current swing angle.
        def door_normal(ang: float) -> torch.Tensor:
            base = -right if side == "right" else right
            ca, sa = math.cos(ang), math.sin(ang)
            dirw = base * ca + front * sa
            n = torch.cross(dirw.cpu(), ez, dim=-1).to(device)
            return n if side == "right" else -n  # outward (robot-side) face normal

        pinch_z = 0.13  # MID-panel: far from the roof rail, whose 9 mm lip sits inside the
        # door-thickness aperture band — run 7 pinched the rail 5 cm above a missed descent
        # and pulled the shell
        a0 = abs(ang0)

        def edge_pinch_pt(ang: float) -> torch.Tensor:
            # centre the pads on the plate MID-THICKNESS (the hinge plane is the OUTER
            # face; pads centred there close beside the 7 mm plate — run 3's 0.2 mm
            # aperture), and lift for the tip offset
            e = edge_at(ang) + door_normal(ang) * -0.006
            e[2] = float(BW(0.0, 0.0, pinch_z)[2]) + tip_lift()
            return e

        # approach ladder (the microwave-franka pattern): the feasible wrist cone is
        # narrow at this extension, but a 7 mm plate under 40 mm pads tolerates +-25 deg
        # of gap misalignment — search tilt x gap-yaw until BOTH errors converge.
        grasped = False
        nrm0 = door_normal(a0)
        for tilt in (25.0, 40.0):
            for gyaw in (0.0, 0.35, -0.35):
                cg, sg = math.cos(gyaw), math.sin(gyaw)
                up = ez.to(device)
                tang = torch.cross(nrm0.cpu(), ez, dim=-1).to(device)
                g_try = nrm0 * cg + tang * sg
                q_pinch = q_oblique(g_try, tilt)
                goto(edge_pinch_pt(a0) + ez.to(device) * 0.10, q_pinch, GRIP_OPEN,
                     tol=0.02, max_steps=110)
                ok_at = goto(edge_pinch_pt(a0), q_pinch, GRIP_OPEN, tol=0.007,
                             max_steps=140, settle=4)
                hold(GRIP_SHUT, 14)
                ap = aperture() * 1000
                # WIGGLE TEST — the only decisive grasp check: a 15 mm pull must rotate the
                # DOOR. Aperture alone cannot tell the door plate from the roof rail (both
                # 7-12 mm; run 7 pulled the shell by its rail).
                moved = False
                if 3.0 <= ap <= 16.0:
                    a_before = abs(door_angles()[jcol])
                    hp, hq = hand_pose()
                    goto(pinch_pos(hp, hq) + front * 0.015, None, GRIP_SHUT,
                         tol=0.006, max_steps=30)
                    moved = abs(door_angles()[jcol]) - a_before > math.radians(2.0)
                print(f"[tp-franka]   {side} pinch tilt={tilt:.0f} gyaw={gyaw:+.2f}: "
                      f"at={ok_at} aperture={ap:.1f}mm door_moved={moved}", flush=True)
                if moved:
                    grasped = True
                    break
                hold(GRIP_OPEN, 6)
            if grasped:
                break
        if not grasped:
            print(f"[tp-franka]   {side} pinch ladder exhausted — abort", flush=True)
            return False
        for deg in range(int(math.degrees(a0)) + 8, int(target_deg) + 8, 6):
            a = math.radians(deg)
            q = q_oblique(door_normal(a))
            goto(edge_pinch_pt(a), q, GRIP_SHUT, tol=0.02, max_steps=36)
            if math.degrees(abs(door_angles()[jcol])) < deg - 26:
                print(f"[tp-franka]   {side} door not following at {deg} deg "
                      f"(door={math.degrees(abs(door_angles()[jcol])):.1f})", flush=True)
                break
        hold(GRIP_OPEN, 10)
        hp, hq = hand_pose()
        goto(pinch_pos(hp, hq) + ez.to(device) * 0.14, None, GRIP_OPEN,
             tol=0.04, max_steps=60)
        got = math.degrees(abs(door_angles()[jcol]))
        print(f"[tp-franka]   {side} door now {got:.1f} deg", flush=True)
        return got >= 78.0

    def open_left_from_inside(target_deg: float = 88.0) -> bool:
        """With the right door open, push the LEFT door's INNER face near its free edge and
        follow the opening arc. No pinch anywhere: the closed door cannot be straddled (the
        roof rail blocks the inner finger's descent corridor), but its inner face is fully
        reachable through the opened right half."""
        phase("push the left door open from inside the right half")
        front, right = box_axes()
        hinge_w = BW(DOOR_L_HINGE[0], DOOR_L_HINGE[1], DOOR_Z_MID)
        q = q_oblique(right)  # gap across the push direction; fingertips do the pushing
        z_w = float(BW(0.0, 0.0, 0.13)[2]) + tip_lift()

        def push_pt(ang: float) -> torch.Tensor:
            """Just past the door's edge/inner surface at swing angle `ang`: at small
            angles this presses the exposed 7 mm EDGE FACE (facing the opened right bay),
            at larger angles the INNER face — both torque the door open."""
            ca, sa = math.cos(ang), math.sin(ang)
            dirw = right * ca + front * sa  # hinge -> edge
            n_out = -torch.cross(dirw.cpu(), ez, dim=-1).to(device)  # robot-side face
            p = hinge_w + dirw * (DOOR_LEN - 0.025) - n_out * 0.004
            p[2] = z_w
            return p

        # stage in the opened right doorway, then slide onto the edge face
        stage_pt = BW(0.06, -0.155, 0.13) + ez.to(device) * tip_lift()
        goto(stage_pt + ez.to(device) * 0.16, q, GRIP_SHUT, tol=0.02, max_steps=110)
        goto(stage_pt, q, GRIP_SHUT, tol=0.012, max_steps=120, settle=2)
        for deg in range(4, int(target_deg) + 6, 6):
            goto(push_pt(math.radians(deg)), q, GRIP_SHUT, tol=0.014, max_steps=70)
            if math.degrees(abs(door_angles()[0])) > target_deg - 8:
                break
        hp, hq = hand_pose()
        goto(pinch_pos(hp, hq) + ez.to(device) * 0.16, None, GRIP_SHUT, tol=0.04, max_steps=60)
        got = math.degrees(abs(door_angles()[0]))
        print(f"[tp-franka]   left door now {got:.1f} deg", flush=True)
        return got >= 78.0

    def pull_drawer(d: int, out: float = 0.20) -> bool:
        """Pinch the tray's front-wall top edge and pull the slide out by `out`."""
        phase(f"pull drawer {c.drawers[d]} out {out:.2f} m")
        front, _right = box_axes()
        b = scene._drawer_b[d]

        def lip_world() -> torch.Tensor:
            dp = scene.box.data.body_pos_w[0, b]
            dq = scene.box.data.body_quat_w[0, b]
            v = torch.tensor(TRAY_LIP_LOCAL, device=device)
            return dp + quat_apply(dq.unsqueeze(0), v.unsqueeze(0))[0]

        lip = lip_world()
        q = q_oblique(front)  # gap spans the pull axis (across the 8 mm wall)
        goto(lip + ez.to(device) * 0.10, q, GRIP_OPEN, settle=2)
        ok_at = goto(lip + ez.to(device) * 0.002, q, GRIP_OPEN, tol=0.006, settle=6)
        hold(GRIP_SHUT, 14)
        ap = aperture() * 1000
        print(f"[tp-franka]   drawer lip pinch: at={ok_at} aperture={ap:.1f}mm (wall=8)",
              flush=True)
        if ap < 3.0:
            hold(GRIP_OPEN, 6)
            return False
        start = drawer_val(d)
        for k in range(1, 11):
            tgt = lip + front * min(out + 0.02 - (start + out) * 0, 0.02 * k + 0.02)
            goto(tgt, q, GRIP_SHUT, tol=0.015, max_steps=30)
            if drawer_val(d) <= -(out - 0.01):
                break
            if k >= 3 and drawer_val(d) > start - 0.02:
                print(f"[tp-franka]   drawer not following (joint={drawer_val(d):.3f})",
                      flush=True)
                hold(GRIP_OPEN, 6)
                return False
        hold(GRIP_OPEN, 8)
        hp, hq = hand_pose()
        goto(pinch_pos(hp, hq) + ez.to(device) * 0.12, None, GRIP_OPEN, tol=0.03, max_steps=50)
        print(f"[tp-franka]   drawer {c.drawers[d]} at {drawer_val(d):.3f}", flush=True)
        return drawer_val(d) <= -(out - 0.03)

    def pick_item(name: str) -> bool:
        """Top-down pinch across the item's short axis, lift high."""
        phase(f"pick the {name}")
        item = scene.items[name]
        gp_local, gap_axis, _half = ITEM_GRASP[name]
        ip = item.data.root_pos_w[0].clone()
        iq = item.data.root_quat_w[0].clone()
        gp = ip + quat_apply(iq.unsqueeze(0),
                             torch.tensor([gp_local], device=device))[0]
        ax = torch.tensor([1.0, 0.0, 0.0] if gap_axis == "x" else [0.0, 1.0, 0.0],
                          device=device)
        gap_w = quat_apply(iq.unsqueeze(0), ax.unsqueeze(0))[0]
        gap_w[2] = 0.0
        q = q_topdown(gap_w)
        goto(gp + ez.to(device) * 0.12, q, GRIP_OPEN, settle=2)
        goto(gp, q, GRIP_OPEN, tol=0.006, settle=6)
        hold(GRIP_SHUT, 16)
        lift = gp + ez.to(device) * 0.22
        goto(lift, q, GRIP_SHUT, tol=0.02, settle=2)
        dz = float(item.data.root_pos_w[0, 2] - ip[2])
        print(f"[tp-franka]   {name} lift dz={dz * 100:.1f}cm aperture={aperture() * 1000:.1f}mm",
              flush=True)
        return dz > 0.10

    def place_item(name: str, d: int) -> bool:
        """Carry over the EXPOSED tray span of (pulled) drawer d, long axis across, lower,
        release, retreat straight up."""
        phase(f"place the {name} in drawer {c.drawers[d]}")
        b = scene._drawer_b[d]
        dp = scene.box.data.body_pos_w[0, b]
        dq = scene.box.data.body_quat_w[0, b]

        def tray_pt(local_y: float, z: float) -> torch.Tensor:
            v = torch.tensor([0.0, local_y, z], device=device)
            return dp + quat_apply(dq.unsqueeze(0), v.unsqueeze(0))[0]

        _gp, gap_axis, half = ITEM_GRASP[name]
        # hand gap must end up along the PULL axis so the item's long axis lies ACROSS
        front, _right = box_axes()
        q = q_oblique(front, 30.0)
        item = scene.items[name]
        high = tray_pt(DROP_LOCAL_Y, TRAY_RIM_Z + 0.20)
        goto(high, q, GRIP_SHUT, tol=0.02, settle=2)
        low = tray_pt(DROP_LOCAL_Y, TRAY_RIM_Z + 0.05)
        goto(low, q, GRIP_SHUT, tol=0.010, settle=4)
        hold(GRIP_OPEN, 12)
        hp, hq = hand_pose()
        goto(pinch_pos(hp, hq) + ez.to(device) * 0.18, None, GRIP_OPEN, tol=0.03, max_steps=60)
        stowed = bool(scene.stowed()[0, [n for n, *_ in c.manifest].index(name)])
        loc = quat_apply_inverse(dq.unsqueeze(0),
                                 (item.data.root_pos_w[0] - dp).unsqueeze(0))[0]
        print(f"[tp-franka]   {name} stowed={stowed} "
              f"drawer-frame={[round(float(v), 3) for v in loc.tolist()]}", flush=True)
        return stowed

    def push_drawer(d: int) -> bool:
        """Push the tray front wall straight in until the slide reads shut."""
        phase(f"push drawer {c.drawers[d]} shut")
        front, _right = box_axes()
        b = scene._drawer_b[d]
        dp = scene.box.data.body_pos_w[0, b]
        dq = scene.box.data.body_quat_w[0, b]
        v = torch.tensor([0.0, -0.012, 0.036], device=device)  # just outside the wall, high
        face = dp + quat_apply(dq.unsqueeze(0), v.unsqueeze(0))[0]
        q = q_oblique(front)  # fingertips hang in front of the wall; the push is shear
        goto(face + front * 0.06 + ez.to(device) * 0.10, q, GRIP_SHUT, settle=2)
        goto(face + front * 0.05, q, GRIP_SHUT, tol=0.015, max_steps=50)
        for k in range(1, 10):
            tgt = face - front * (0.025 * k)
            goto(tgt, q, GRIP_SHUT, tol=0.015, max_steps=26)
            if drawer_val(d) > -c.drawer_closed_tol:
                break
        hp, hq = hand_pose()
        goto(pinch_pos(hp, hq) + front * 0.10 + ez.to(device) * 0.08, None, GRIP_SHUT,
             tol=0.04, max_steps=50)
        print(f"[tp-franka]   drawer {c.drawers[d]} at {drawer_val(d):.3f}", flush=True)
        return drawer_val(d) > -c.drawer_closed_tol

    def close_door(side: str) -> bool:
        """Push the door's outer face near the free edge along the closing arc."""
        phase(f"push the {side} door shut")
        sgn = 1.0 if side == "right" else -1.0
        hinge = DOOR_R_HINGE if side == "right" else DOOR_L_HINGE
        jcol = 1 if side == "right" else 0
        hinge_w = BW(hinge[0], hinge[1], DOOR_Z_MID)
        front, right = box_axes()
        base = -right if side == "right" else right

        def face_pt(ang: float) -> tuple[torch.Tensor, torch.Tensor]:
            ca, sa = math.cos(ang), math.sin(ang)
            dirw = base * ca + front * sa  # both doors swing frontward (run 7 sign fix)
            nrm = torch.cross(dirw.cpu(), ez, dim=-1).to(device)
            if side != "right":
                nrm = -nrm
            p = hinge_w + dirw * (DOOR_LEN - 0.02) + nrm * 0.015
            p[2] = float(BW(0.0, 0.0, 0.13)[2]) + tip_lift()  # mid-panel, tip-compensated
            return p, nrm

        start = math.degrees(abs(door_angles()[jcol]))
        p0, n0 = face_pt(math.radians(start))
        q = q_oblique(n0)
        goto(p0 + n0 * 0.06 + ez.to(device) * 0.10, q, GRIP_SHUT, tol=0.03, max_steps=80)
        goto(p0 + n0 * 0.05, q, GRIP_SHUT, tol=0.02, max_steps=60)
        for deg in list(np.arange(start, -8.0, -7.0)):
            a = math.radians(max(float(deg), 0.0))
            p, n = face_pt(a)
            goto(p, q_oblique(n), GRIP_SHUT, tol=0.02, max_steps=30)
            if abs(math.degrees(door_angles()[jcol])) < 4.0:
                break
        hp, hq = hand_pose()
        goto(pinch_pos(hp, hq) + front * 0.08 + ez.to(device) * 0.10, None, GRIP_SHUT,
             tol=0.04, max_steps=50)
        got = abs(math.degrees(door_angles()[jcol]))
        print(f"[tp-franka]   {side} door at {got:.1f} deg", flush=True)
        return got <= c.door_closed_deg

    # --- the solve -------------------------------------------------------------------------
    env.reset()
    hold(GRIP_OPEN, 20)
    bp, _ = box_pose()
    print(f"[tp-franka] box at {[round(float(v), 3) for v in (bp - origin).tolist()]} "
          f"doors={door_angles()} drawers={[round(drawer_val(k), 3) for k in range(3)]}",
          flush=True)
    # controller sanity: rotation must converge in free space, or nothing else matters
    # (BASE-relative: the base rides on the bench top at surface height, not at origin z)
    base_w = art.data.root_pos_w[0].clone()
    free = base_w + torch.tensor([0.35, 0.0, 0.35], device=device)
    ok_free = goto(free, q_topdown(torch.tensor([1.0, 0.0, 0.0], device=device)),
                   GRIP_OPEN, tol=0.015, max_steps=160)
    hp, hq = hand_pose()
    print(f"[tp-franka] free-space rot self-test: ok={ok_free}", flush=True)

    if args.only_doors:
        ok1 = crack_right_door()
        ok2 = pull_door_open("right")
        ok3 = open_left_from_inside()
        print(f"[tp-franka] only_doors: crack={ok1} right={ok2} left={ok3}", flush=True)
    else:
        check("crack-right-door", crack_right_door())
        check("open-right-door", pull_door_open("right"))
        check("open-left-door", open_left_from_inside())
        items = [name for name, *_ in c.manifest]
        want = 0
        for i, name in enumerate(items):
            d = scene._assigned[i]
            check(f"{name}-drawer-out", pull_drawer(d))
            check(f"{name}-picked", pick_item(name))
            check(f"{name}-stowed", place_item(name, d))
            check(f"{name}-drawer-shut", push_drawer(d))
            want += 30
            got = int(scene.score()[0])
            check(f"score-{want}", got == want, f"score={got}")
        check("close-left-door", close_door("left"))
        check("close-right-door", close_door("right"))
        hold(GRIP_OPEN, 30)
        check("score-100", int(scene.score()[0]) == 100, f"score={int(scene.score()[0])}")
        check("success", bool(scene.success()[0]))
        print(f"[tp-franka] RESULT: {'ALL PASS' if not FAILS else 'FAILED: ' + ','.join(FAILS)}",
              flush=True)

    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="packing.tool_packing.franka.osc",
                            phases=json.dumps(phases))
        print(f"[tp-franka] saved {arr.shape} -> {args.out}", flush=True)
        if args.hdfs_dir:
            rc = os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir} 2>/dev/null; hdfs dfs -put -f "
                           f"{args.out} {args.hdfs_dir}/{os.path.basename(args.out)}")
            print(f"[tp-franka] hdfs upload rc={rc}", flush=True)
    print("TOOL_PACKING_FRANKA_SMOKE_DONE", flush=True)


def _hard_exit_teardown() -> None:
    """Kit teardown regularly hangs in app.close(); the repo's standard hard-exit."""
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

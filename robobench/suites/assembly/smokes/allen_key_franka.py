"""Franka smoke for AllenBoltAssemblyScene — the arm picks the allen key off the table, stands it
tip-down in the staged bolt's hex socket, and ratchet-screws the bolt down the platform's real SDF
threads until it seats.

Grasping follows the benchmark weld-on-closure contract (cf. the pc_* franka smokes): a
normally-disabled FixedJoint hand<->key is enabled when the gripper is verifiably closed around
the key's hex arm (closure verified geometrically: pads flanking the arm at the grip band,
fingers at a real-grip width) and released when it opens. Everything else is live physics —
key<->socket hex contact, bolt<->platform thread contact, and the key standing unheld in the
socket during the one mid-task regrasp — so a missed grasp, a jammed insertion, or a dropped key
fails honestly.

The choreography works around two hard constraints:
  * A flat-spawned key can only be erected tip-down by pitching the hand 90 deg with it — the
    hand that erects the key ends HORIZONTAL, and no single grasp of the lying key yields the
    top-down grip that screwing needs. So the arm picks the lying key by its working arm, erects
    it about the HANDLE axis (the handle stays put; the arm sweeps tip-down), inserts the tip
    into the socket with the horizontal hand, releases, and re-grasps the standing arm TOP-DOWN —
    from there the wrist roll maps 1:1 onto the key's screw axis.
  * The wrist's +-166 deg joint-7 range cannot turn the ~10 revolutions a full seat needs, so the
    smoke ratchets WITHOUT ever letting go: press + twist a stroke until joint 7 nears its stop,
    bleed the torsional wind-up, lift the tip just clear of the socket, rewind the wrist by a
    multiple of 60 deg (the hex-symmetry step — clocking is preserved exactly), drop back in and
    stroke again. The key never leaves the hand between the regrasp and the final release.

The bolt itself is staged HAND-STARTED, the way a person finger-spins a bolt two turns before
reaching for the key: teleported upright over the hole, dropped to nest on the thread crests
(which measures a valid depth<->yaw helix registration), then advanced along its own helix to
2.5 turns deep — a thread-true pose by construction (a wrench-driven start strips crests under
any misalignment) that must then HOLD unaided through a settle. A merely crest-nested bolt is
pried loose by any insertion tap; the captured one holds against them. The robot's job — and
the verdict's measure — is everything from the key pick onward.

Phases: show -> stage(drop/nest -> helix-set -> hold) -> pick(hover/down/close) -> lift -> erect -> carry ->
insert(descend, peck-retry) -> handoff(release in socket) -> regrasp(hover/down/close) ->
[stroke -> unload -> lift -> rewind -> reinsert]* -> release -> retreat -> settle.
Verdict: seated count, depth the key drove vs the ratchet's revolutions (expected ~2.0 mm/rev,
the M16 pitch), key->bolt slip, cycles, picks, drops.

.venv/bin/python -m robobench.suites.assembly.smokes.allen_key_franka --headless
python -m robobench.suites.assembly.smokes.allen_key_franka --livestream 2
python -m robobench.suites.assembly.smokes.allen_key_franka \
    --headless --enable_cameras --video robobench/suites/assembly/videos/allen_key_franka.mp4
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--video", type=str, default="", help="save an mp4 here (needs --headless --enable_cameras)")
parser.add_argument("--cap", type=int, default=1, help="with --video: capture one frame every N control steps (1 at 15 Hz control -> 2x slow-mo at 30 fps)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
livestream_on = args.livestream > 0

app = AppLauncher(args).app

from typing import TYPE_CHECKING  # noqa: E402

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402

import robobench  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    axis_angle_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
)
from robobench.core import ENVS  # noqa: E402
from robobench.suites.assembly.smokes import close_and_exit  # noqa: E402

if TYPE_CHECKING:
    from robobench.suites.assembly.scenes import AllenBoltAssemblyScene

DT = 1.0 / 240.0  # sim timestep (matches the registered env's dt override)

# Key geometry baked into the committed key USD (informs every grip/clearance constant below):
# tip at origin, 50 mm working arm up +z, 120 mm handle along +x off the elbow; both arms hex
# 12.6 mm across flats / 14.4 mm across corners, corners authored at k*60 deg; the handle spans
# 42.8-57.2 mm above the tip.
# ----- bolt-local geometry (origin = thread tip, +z up through the head) -------------------------
SOCKET_FLOOR_Z = 0.0355  # hex recess floor
SOCKET_MOUTH_Z = 0.0428  # head top = recess mouth (7.3 mm deep socket, 0.75 mm/side clearance)
PITCH_MM = 2.0           # M16 coarse pitch, the expected descent per revolution
STOP_DEPTH = 0.0235      # stop screwing at this tip depth (m) — just before the head bottoms at 24.8 mm
STAGE_GAP = 0.0015       # staging gap (m) between bolt tip and plate top, just above the thread entry
STAGE_DEPTH = 0.005      # stage the bolt at this tip depth (~2.5 turns in: thread-captured)

# ----- Franka OSC action semantics (6 EE pose deltas + 2 finger targets at 15 Hz) -----------------
POS_SCALE, ROT_SCALE = 0.02, 0.097
ROT_SAT = 8.0              # rotation lead cap (units): gravity droop stalls a 1-unit lead
OPEN_W = 0.04
FINGER_TO_PAD = 0.045      # panda_finger body origin -> finger-pad centre, along the approach
                           # (the finger TIP ends 8.8 mm past the pad centre)

# ----- grips --------------------------------------------------------------------------------------
# Pick: top-down pinch of the LYING key's working arm (fingers close across the arm; the lying
# hex presents its corners sideways, so the pads land across corners, 14.4 mm).
PICK_GRIP_D = 0.030        # grip point: this far up the lying arm from the tip. HIGH on purpose:
                           # the tip must reach the socket floor with a HORIZONTAL hand, whose
                           # workspace bottoms out around z ~0.096 m here — every mm of grip
                           # height is a mm of tip depth. The 18 mm pads then span 21-39 mm of
                           # the 50 mm arm, still clear of the handle root (flank at ~43.7 mm).
PICK_PAD_LIFT = 0.0045     # pad-centre height above the lying arm's axis (fingertips ~2 mm off
                           # the table; the 18 mm pad still spans the whole 12.6 mm hex)
PICK_W = 0.0069            # per-finger closed width: across-corners half-width minus a kiss
# Screw grip: top-down pinch of the STANDING arm, fingers across the flats (the hand x axis runs
# along the handle, so the finger bodies clear it; the pads' upper edges stay under the handle).
GRIP_UP = 0.030            # grip point: this far above the key tip (pad top 39 mm < handle 42.8)
SCREW_W = 0.0057           # per-finger closed width: across-flats half-width minus a kiss
STRADDLE_W = 0.015         # per-finger width while descending AROUND the arm
CLOSED_MIN, CLOSED_MAX = 0.009, 0.017  # closure window (finger-joint sum, m): hex 12.6-14.4 mm

# ----- choreography -------------------------------------------------------------------------------
HOVER_CLEAR = 0.05         # pad hover height above the grip point before a descent
CARRY_TIP_Z = 0.16         # key-tip height (above the table top) for the lift/erect/carry legs
INSERT_HOVER = 0.012       # tip hover above the socket mouth before the first insertion
PRESS_LEAD = 0.004         # stroke press: command the tip this far below the live socket floor
                           # (the press is what keeps the hex from camming out under torque)
LIFT_CLEAR = 0.004         # recock lift: tip this far above the socket mouth (out of the hex)
J7_GUARD = 2.6             # end a stroke when joint 7 exceeds this (limit 2.897 rad)
J7_START = -2.1            # rewind aims joint 7 back here (leaves ~270 deg of stroke)
STROKE_W = 1.2             # commanded stroke yaw rate (rad/s), under the 1.455 rad/s action cap
REWIND_W = 1.8             # rewind yaw rate (rad/s), free air
SLIP_ABORT = math.radians(45.0)  # end a stroke early if the key slips this far over the hex
MAX_CYCLES = 30            # ratchet cycle budget (a clean run needs ~13)
PICK_RETRIES = 3
INSERT_RETRIES = 8         # peck re-tries per insertion (rise, re-trim, drop again)
DROP_BUDGET = 3

# Waypoint tolerances and per-phase step budgets, in CONTROL steps (15 Hz -> 16 substeps each).
TOL_P, TOL_R = 0.004, 0.06
SHOW_END, STAGE_SETTLE = 20, 25
WP_TIMEOUT, CLOSE_STEPS, SETTLE_STEPS = 75, 18, 45
LIFT_STEPS, ERECT_STEPS, CARRY_STEPS, INSERT_STEPS, RETREAT_STEPS = 55, 60, 90, 45, 50
STROKE_TIMEOUT, REWIND_TIMEOUT, REINSERT_TIMEOUT = 90, 60, 60
HARD_CAP = 6000
LOG_EVERY = 45


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_of(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_wxyz.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def up_axis_of(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """World direction of the body's local +z, shape (n, 3)."""
    w, x, y, z = quat_wxyz.unbind(-1)
    return torch.stack((2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)), dim=-1)


def smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def main() -> None:
    device = getattr(args, "device", None) or ("cuda:0" if torch.cuda.is_available() else "cpu")
    robobench.discover()

    env = ENVS.get("assembly.allen_bolt.franka.osc")().build(num_envs=args.num_envs, device=device)
    sc: AllenBoltAssemblyScene = env.scene  # type: ignore[assignment]
    n = env.num_envs
    dev = device
    ids = torch.arange(n, device=dev)
    bolt, key, plat = sc.bolts[0], sc.keys[0], sc.platforms[0]
    plate_top = sc.cfg.plate_top
    art = env.robot.articulation
    hand_idx = art.body_names.index("panda_hand")
    j7 = art.joint_names.index("panda_joint7")
    fingers = art.find_joints(["panda_finger_joint.*"])[0]
    render = (not args.headless) or livestream_on
    ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
    ex = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3)

    # ----- hand<->key welds (the grasp contract): pre-authored, disabled FixedJoints --------------
    # PhysX latches a joint's local frames when it is FIRST enabled; frame rewrites on a re-enabled
    # joint are ignored. Each grasp therefore consumes a fresh joint from the pool: author the live
    # relative pose while it is still disabled, enable it once, and on release disable it for good.
    from pxr import Gf, UsdPhysics

    stage = env.stage
    WELD_POOL = 16  # pick + regrasp + retries' slack (strokes never release)
    weld_paths: list[list[str]] = []
    for e in range(n):
        base = f"/World/envs/env_{e}"
        assert stage.GetPrimAtPath(f"{base}/Key_0/allen_key").HasAPI(UsdPhysics.RigidBodyAPI)
        row = []
        for k_ in range(WELD_POOL):
            j = UsdPhysics.FixedJoint.Define(stage, f"{base}/hand_key_weld_{k_}")
            j.CreateBody0Rel().SetTargets([f"{base}/Robot/panda_hand"])
            j.CreateBody1Rel().SetTargets([f"{base}/Key_0/allen_key"])
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateJointEnabledAttr(False)
            j.CreateExcludeFromArticulationAttr(True)  # a maximal-coordinate weld, not an arm DOF
            row.append(f"{base}/hand_key_weld_{k_}")
        weld_paths.append(row)

    cam = writer = None
    if args.video:
        import imageio.v2 as imageio
        from isaaclab.sensors import Camera, CameraCfg
        cam = Camera(CameraCfg(prim_path="/World/cam", update_period=0.0, height=720, width=1280, data_types=["rgb"],
                               spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.01, 100.0))))
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args.video, fps=30)

    env.sim.reset()  # re-parse physics so the pre-authored welds (and camera) are picked up
    env.reset()

    # Open-env OSC retune (the pc_* franka smokes' values): the controller carries no gravity
    # compensation, so the stock gains leave a configuration-dependent pose sag; the stiffer gains
    # bring it into the learned bias' budget, slightly overdamped so transits do not ring.
    osc = env.robot.controller.controllers[0]
    osc._kp[0:3] = 400.0
    osc._kp[3:6] = 450.0
    osc._kd = 2.2 * osc._kp.sqrt()

    hole_xy = plat.data.root_pos_w[:, :2].clone()  # insert bore axis == platform origin (bore-centred)
    plat_z = plat.data.root_pos_w[:, 2].clone()
    table_z = plat_z.clone()  # platform origin sits ON the table top

    # ----- camera: a two-anchor shot blended by the key's own trip toward the bolt ---------------
    cam_pose = None
    if cam is not None:
        p0 = plat.data.root_pos_w[0]
        k0 = key.data.root_pos_w[0]
        pick_eye = torch.tensor([float(k0[0]) - 0.28, float(k0[1]) - 0.38, float(p0[2]) + 0.40], device=dev)
        pick_tgt = torch.tensor([float(k0[0]), float(k0[1]), float(p0[2]) + 0.05], device=dev)
        ins_eye = torch.tensor([float(p0[0]) + 0.16, float(p0[1]) - 0.30, float(p0[2]) + 0.22], device=dev)
        ins_tgt = torch.tensor([float(p0[0]), float(p0[1]), float(p0[2]) + 0.07], device=dev)
        trip = float((k0[0:2] - p0[0:2]).norm())
        cam_s = 0.0

        def cam_pose() -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal cam_s
            u = 1.0 - float((key.data.root_pos_w[0, 0:2] - p0[0:2]).norm()) / max(trip, 1e-6)
            s = smoothstep(u)
            cam_s += 0.06 * (s - cam_s)
            eye = pick_eye + (ins_eye - pick_eye) * cam_s
            tgt = pick_tgt + (ins_tgt - pick_tgt) * cam_s
            return eye.unsqueeze(0), tgt.unsqueeze(0)

    print(env.describe(), flush=True)

    # Measure hand-frame -> finger-pad-centre once from the live articulation (robust to asset edits).
    lf = art.body_names.index("panda_leftfinger")
    _hq0 = art.data.body_quat_w[:, hand_idx]
    _rel = quat_apply_inverse(_hq0, art.data.body_pos_w[:, lf] - art.data.body_pos_w[:, hand_idx])
    hand_to_pad = float(_rel[0, 2]) + FINGER_TO_PAD
    print(f"hand->pad centre: {hand_to_pad:.4f} m", flush=True)

    # ----- weld toggles + grasp transform ---------------------------------------------------------
    rel_p = torch.zeros(n, 3, device=dev)  # key pose in the hand frame, captured at weld time
    rel_q = torch.zeros(n, 4, device=dev)
    welded = torch.zeros(n, dtype=torch.bool, device=dev)
    weld_k = 0  # next fresh joint in the pool (all envs grasp in lockstep)

    def hand_pose() -> tuple[torch.Tensor, torch.Tensor]:
        return art.data.body_pos_w[:, hand_idx], art.data.body_quat_w[:, hand_idx]

    def weld_on() -> None:
        nonlocal weld_k
        assert weld_k < WELD_POOL, "weld pool exhausted"
        hp, hq = hand_pose()
        rel_p[:] = quat_apply_inverse(hq, key.data.root_pos_w - hp)
        rel_q[:] = quat_mul(quat_conjugate(hq), key.data.root_quat_w)
        for e in range(n):
            j = UsdPhysics.FixedJoint.Get(stage, weld_paths[e][weld_k])
            p, q = rel_p[e].tolist(), rel_q[e].tolist()
            j.GetLocalPos0Attr().Set(Gf.Vec3f(p[0], p[1], p[2]))
            j.GetLocalRot0Attr().Set(Gf.Quatf(q[0], Gf.Vec3f(q[1], q[2], q[3])))
            j.GetJointEnabledAttr().Set(True)
        welded[:] = True

    def weld_off() -> None:
        nonlocal weld_k
        for e in range(n):
            UsdPhysics.FixedJoint.Get(stage, weld_paths[e][weld_k]).GetJointEnabledAttr().Set(False)
        weld_k += 1
        welded[:] = False

    # ----- servo layer: waypoints -> OSC action ---------------------------------------------------
    def servo(tp: torch.Tensor, tq: torch.Tensor, grip) -> torch.Tensor:
        """8-D action toward a hand waypoint: clamped EE pose delta + finger targets. `grip` is a
        float (both fingers) or an (n, 2) tensor of explicit finger targets."""
        hp, hq = hand_pose()
        a = torch.zeros(n, 8, device=dev)
        v = (tp - hp) / POS_SCALE
        vn = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        a[:, 0:3] = v * (vn.clamp(max=1.0) / vn)  # norm-clamp: keep the DIRECTION, or a long
        # diagonal move finishes its short axis first (the hand drops early, then skims low)
        qe = quat_mul(tq, quat_conjugate(hq))
        qe = torch.where(qe[:, :1] >= 0, qe, -qe)
        rot = axis_angle_from_quat(qe) / ROT_SCALE
        nrm = rot.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        a[:, 3:6] = rot * (nrm.clamp(max=ROT_SAT) / nrm)
        a[:, 6:8] = grip
        return a

    def at(tp: torch.Tensor, tq: torch.Tensor) -> torch.Tensor:
        hp, hq = hand_pose()
        qe = quat_mul(tq, quat_conjugate(hq))
        ang = 2.0 * torch.arccos(qe[:, 0].abs().clamp(max=1.0))
        return ((tp - hp).norm(dim=-1) < TOL_P) & (ang < TOL_R)

    def hand_for_key(kp: torch.Tensor, kq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Hand waypoint that puts the WELDED key at pose (kp, kq), via the captured grasp frame."""
        hq = quat_mul(kq, quat_conjugate(rel_q))
        return kp - quat_apply(hq, rel_p), hq

    def q_down(yaw: torch.Tensor) -> torch.Tensor:
        """Top-down hand orientation (approach -z) with the given world yaw."""
        flip = torch.zeros(n, 4, device=dev)
        flip[:, 1] = 1.0  # 180 deg about x: hand +z -> world -z
        return quat_mul(quat_from_angle_axis(yaw, ez), flip)

    def j7_after(cand: torch.Tensor) -> torch.Tensor:
        """Predicted joint-7 angle after the servo's shortest-path yaw to `cand` (valid near a
        top-down hand, where yaw routes through the wrist roll; the flipped hand maps a positive
        world yaw to a NEGATIVE joint-7 move)."""
        hx = quat_apply(hand_pose()[1], ex)
        cur = torch.atan2(hx[:, 1], hx[:, 0])
        return art.data.joint_pos[:, j7] - _wrap(cand - cur)

    def nearest_parity(psi: torch.Tensor) -> torch.Tensor:
        """The gripper is 180-deg symmetric: return psi or psi-pi, whichever leaves the wrist roll
        nearer its centre."""
        alt = _wrap(psi - math.pi)
        return torch.where(j7_after(psi).abs() <= j7_after(alt).abs(), psi, alt)

    def pad_centre() -> torch.Tensor:
        """World centre of the closed finger pads, along the hand's approach axis."""
        hp, hq = hand_pose()
        return hp + quat_apply(hq, ez) * hand_to_pad

    # ----- task frames -----------------------------------------------------------------------------
    def depth() -> torch.Tensor:  # bolt tip depth below the plate top (m), per env
        return plat_z + plate_top - bolt.data.root_pos_w[:, 2]

    def tip_axial() -> torch.Tensor:
        """Key tip height above the bolt origin along the bolt axis (m): SOCKET_MOUTH_Z at the
        recess mouth, SOCKET_FLOOR_Z seated on the floor."""
        ub = up_axis_of(bolt.data.root_quat_w)
        return ((key.data.root_pos_w - bolt.data.root_pos_w) * ub).sum(-1)

    def tip_lateral() -> torch.Tensor:
        ub = up_axis_of(bolt.data.root_quat_w)
        rel = key.data.root_pos_w - bolt.data.root_pos_w
        return (rel - (rel * ub).sum(-1, keepdim=True) * ub).norm(dim=-1)

    def clock_err() -> torch.Tensor:
        """Key hex yaw error to the socket's nearest hex sector, wrapped to [-30, 30) deg."""
        dp = (yaw_of(bolt.data.root_quat_w) - yaw_of(key.data.root_quat_w)) % (math.pi / 3)
        return torch.where(dp > math.pi / 6, dp - math.pi / 3, dp)

    def handle_heading() -> torch.Tensor:
        hv = quat_apply(key.data.root_quat_w, ex)
        return torch.atan2(hv[:, 1], hv[:, 0])

    def key_up() -> torch.Tensor:
        return up_axis_of(key.data.root_quat_w)[:, 2]

    def in_socket(margin: float = 0.002) -> torch.Tensor:
        return (tip_axial() < SOCKET_MOUTH_Z - margin) & (tip_lateral() < 0.004)

    def bolt_ok(up_min: float = 0.98) -> torch.Tensor:
        """The bolt is still where the choreography needs it: on the bore axis, threaded in, near
        upright (ejection shows in the axis/depth terms)."""
        axis_err = (bolt.data.root_pos_w[:, 0:2] - hole_xy).norm(dim=-1)
        return (axis_err < 0.004) & (depth() > 0.002) & (up_axis_of(bolt.data.root_quat_w)[:, 2] > up_min)

    def bolt_state() -> str:
        tilt = torch.rad2deg(torch.acos(up_axis_of(bolt.data.root_quat_w)[:, 2].clamp(-1, 1)))
        axis_err = (bolt.data.root_pos_w[:, 0:2] - hole_xy).norm(dim=-1)
        return (f"bolt: depth {float(depth().min()) * 1e3:+.1f}mm, axis err "
                f"{float(axis_err.max()) * 1e3:.1f}mm, tilt {float(tilt.max()):.1f}deg")

    def upright_cmd(psi: torch.Tensor) -> torch.Tensor:
        """Commanded key orientation: upright at yaw `psi`, pre-rotated by the learned tilt bias so
        the ACHIEVED pose stands vertical (yaw is the caller's command, never biased)."""
        ang = tilt_bias.norm(dim=-1).clamp_min(1e-9)
        axis = torch.cat([tilt_bias / ang.unsqueeze(-1), torch.zeros(n, 1, device=dev)], dim=-1)
        return quat_mul(quat_from_angle_axis(ang, axis), quat_from_angle_axis(psi, ez))

    def learn_tilt(gain: float = 0.2) -> None:
        """Integrate the key's residual LEAN (free air only): the command already carries the bias,
        so the residual drives it until the achieved arm axis is vertical."""
        r = torch.cross(up_axis_of(key.data.root_quat_w), ez, dim=-1)[:, 0:2]  # rights up_k onto ez
        tilt_bias[:] = (tilt_bias + gain * r).clamp(-0.15, 0.15)

    # ----- staging: teleport the bolt over the hole, then hand-start it with a wrench -------------
    def stage_drop() -> None:
        st = torch.zeros(n, 13, device=dev)
        st[:, 0:2] = hole_xy
        st[:, 2] = plat_z + plate_top + STAGE_GAP
        st[:, 3] = 1.0
        bolt.write_root_state_to_sim(st, ids)

    # ----- step + capture -------------------------------------------------------------------------
    step_i = 0

    def step(action: torch.Tensor) -> None:
        nonlocal step_i
        step_i += 1
        capture = writer is not None and step_i % args.cap == 0
        if capture:
            cam.set_world_poses_from_view(*cam_pose())
        env.step(action, render=capture or render)
        if capture:
            import numpy as np
            cam.update(env.dt)
            img = cam.data.output["rgb"][0].detach().cpu().numpy()
            if img.dtype != np.uint8:
                img = (img.clip(0, 1) * 255).astype(np.uint8)
            writer.append_data(img[..., :3])

    # ----- run ------------------------------------------------------------------------------------
    home_p, home_q = None, None
    bolt_turn = torch.zeros(n, device=dev)  # cumulative screw-in rotation (rad, +ve = descending)
    key_turn = torch.zeros(n, device=dev)
    prev_bolt_yaw = yaw_of(bolt.data.root_quat_w)
    prev_key_yaw = yaw_of(key.data.root_quat_w)
    handoff_depth = None       # depth when the ratchet takes over (gain baseline)
    nest_depth = torch.zeros(n, device=dev)
    staged_depth = torch.zeros(n, device=dev)
    cycles, picks, drops, insert_pecks = 0, 0, 0, 0
    pos_off = torch.zeros(n, 3, device=dev)   # INTEGRATED bias (desired - achieved, free air): the
    # OSC has no gravity compensation, so its realized pose sags configuration-dependently by
    # several mm — commands near contact add this learned offset so the achieved pose lands true
    tilt_bias = torch.zeros(n, 2, device=dev)  # the same for the key's LEAN (axis-angle xy)
    grip_freeze = torch.zeros(n, 2, device=dev)  # finger targets latched at release: freezing the
    # PD at the MEASURED positions decays the squeeze before the pads separate
    close_ok = torch.zeros(n, device=dev)     # consecutive ticks the closure has verified
    seat_ok = torch.zeros(n, device=dev)      # consecutive ticks the inserted key has read seated
    grip_pt = torch.zeros(n, 3, device=dev)
    grip_yaw = torch.zeros(n, device=dev)
    wp_p = torch.zeros(n, 3, device=dev)
    wp_q = torch.zeros(n, 4, device=dev)
    glide_from_p = torch.zeros(n, 3, device=dev)
    glide_from_z = torch.zeros(n, device=dev)
    glide_yaw0 = torch.zeros(n, device=dev)
    glide_q0 = torch.zeros(n, 4, device=dev)
    glide_aa = torch.zeros(n, 3, device=dev)
    psi_cmd = torch.zeros(n, device=dev)      # commanded key yaw through insert/stroke/rewind
    rewind_tgt = torch.zeros(n, device=dev)
    rw_arrived = torch.zeros(n, dtype=torch.bool, device=dev)
    stroke_slip0 = torch.zeros(n, device=dev)
    slip_armed = torch.zeros(n, dtype=torch.bool, device=dev)
    drive_slip = torch.zeros(n, device=dev)  # summed key->bolt slip WITHIN strokes (hex fidelity)
    back_axis = torch.zeros(n, 3, device=dev)
    release_p = torch.zeros(n, 3, device=dev)
    release_q = torch.zeros(n, 4, device=dev)
    peck_tries = 0
    regrasp_tries = 0

    def start_insert_glide() -> None:
        """Latch the descent glide state (shared by insert and reinsert)."""
        glide_from_z[:] = tip_axial()
        psi_cmd[:] = yaw_of(key.data.root_quat_w) + clock_err()

    phase, marker = "show", 0
    i = 0
    while True:
        i += 1
        t_in = i - marker
        hp, hq = hand_pose()
        if home_p is None:
            home_p, home_q = hp.clone(), hq.clone()
        act = servo(home_p, home_q, OPEN_W)  # default: hold home, fingers open

        if phase == "show":
            if t_in >= SHOW_END:
                stage_drop()
                prev_bolt_yaw = yaw_of(bolt.data.root_quat_w)
                phase, marker = "stage_drop", i
        elif phase == "stage_drop":  # hands off: the bolt falls the 1.5 mm gap and nests on the
            # crests, which MEASURES a valid (depth, yaw) helix registration — then advance it
            # ALONG ITS OWN HELIX to the stage depth, a thread-true pose by construction (the
            # sibling smoke's staging teleport, extended down the helix; a wrench-driven start
            # strips crests under any misalignment — a 3 N press walked the bolt clean through).
            if t_in >= STAGE_SETTLE:
                nest_depth[:] = depth()
                dyaw = -2 * math.pi * (STAGE_DEPTH - nest_depth) / (PITCH_MM * 1e-3)
                st = torch.zeros(n, 13, device=dev)
                st[:, 0:2] = hole_xy
                st[:, 2] = plat_z + plate_top - STAGE_DEPTH
                st[:, 3:7] = quat_from_angle_axis(yaw_of(bolt.data.root_quat_w) + dyaw, ez)
                bolt.write_root_state_to_sim(st, ids)
                phase, marker = "stage_hold", i
        elif phase == "stage_hold":  # settle: the staged pose must HOLD unaided (self-locking
            # thread at bolt_friction 0.3) before the robot is allowed near it
            if t_in >= STAGE_SETTLE:
                staged_depth[:] = depth()
                print(f"  staged: bolt set {float(staged_depth.mean()) * 1e3:+.2f} mm down the "
                      f"helix (nested at {float(nest_depth.mean()) * 1e3:+.2f} mm) and holding", flush=True)
                picks += 1
                phase, marker = "pick_hover", i
        elif phase == "pick_hover":  # glide to a top-down hover over the lying key's working arm
            if t_in == 1:
                pos_off.zero_()
                glide_from_p[:] = hp
                arm = up_axis_of(key.data.root_quat_w)  # lying: the arm's horizontal direction
                grip_yaw[:] = nearest_parity(torch.atan2(arm[:, 1], arm[:, 0]))  # hand x along the
                # arm -> the fingers close across it
                hx = quat_apply(hq, ex)
                glide_yaw0[:] = torch.atan2(hx[:, 1], hx[:, 0])
            arm = up_axis_of(key.data.root_quat_w)
            grip_pt[:] = key.data.root_pos_w + arm * PICK_GRIP_D
            grip_pt[:, 2] = key.data.root_pos_w[:, 2] + PICK_PAD_LIFT
            wp_p[:] = grip_pt
            wp_p[:, 2] = grip_pt[:, 2] + hand_to_pad + HOVER_CLEAR
            s = smoothstep(t_in / WP_TIMEOUT)
            wp_q[:] = q_down(glide_yaw0 + _wrap(grip_yaw - glide_yaw0) * s)
            if s >= 1.0:  # arrived, free air: learn the pad-centre bias for the descent
                want = grip_pt.clone()
                want[:, 2] += HOVER_CLEAR
                pos_off[:] = (pos_off + 0.3 * (want - pad_centre())).clamp(-0.12, 0.12)
            goal = wp_p + pos_off
            act = servo(glide_from_p + (goal - glide_from_p) * s, wp_q, STRADDLE_W)
            pad_err = (pad_centre()[:, 0:2] - grip_pt[:, 0:2]).norm(dim=-1)
            if (t_in >= WP_TIMEOUT + 10 and bool((pad_err < 0.004).all()) and bool(at(goal, wp_q).all())) \
                    or t_in >= 3 * WP_TIMEOUT:
                phase, marker = "pick_down", i
        elif phase == "pick_down":  # descend AROUND the arm: open fingers pass it on both sides
            s = smoothstep(t_in / 40.0)
            wp_p[:] = grip_pt
            wp_p[:, 2] = grip_pt[:, 2] + hand_to_pad + HOVER_CLEAR * (1.0 - s)
            if s >= 1.0:
                want = grip_pt.clone()
                pos_off[:] = (pos_off + 0.25 * (want - pad_centre())).clamp(-0.12, 0.12)
            act = servo(wp_p + pos_off, wp_q, STRADDLE_W)
            pad_on = (pad_centre() - grip_pt).norm(dim=-1) < 0.003
            if (t_in >= 48 and bool(pad_on.all())) or t_in >= 2 * WP_TIMEOUT:
                close_ok.zero_()
                phase, marker = "pick_close", i
        elif phase == "pick_close":  # close to the arm's width; verify geometrically, then weld
            s = smoothstep(t_in / CLOSE_STEPS)
            width = STRADDLE_W + (PICK_W - STRADDLE_W) * s
            act = servo(wp_p + pos_off, wp_q, width)
            gap = art.data.joint_pos[:, fingers].sum(dim=-1)
            arm = up_axis_of(key.data.root_quat_w)  # the closing pads may nudge the key: verify
            grip_pt[:] = key.data.root_pos_w + arm * PICK_GRIP_D  # closure against its LIVE pose
            grip_pt[:, 2] = key.data.root_pos_w[:, 2] + PICK_PAD_LIFT
            near = (pad_centre() - grip_pt).norm(dim=-1) < 0.004
            ok = near & (gap > CLOSED_MIN) & (gap < CLOSED_MAX)
            close_ok[:] = torch.where(ok, close_ok + 1, torch.zeros_like(close_ok))
            if t_in >= CLOSE_STEPS + 6 and bool((close_ok >= 4).all()):
                weld_on()
                print(f"  picked the key: pads across the lying arm, finger gap "
                      f"{[f'{float(g) * 1e3:.1f}' for g in gap]} mm (hex 12.6/14.4)", flush=True)
                phase, marker = "lift", i
            elif t_in >= CLOSE_STEPS + 30:
                if picks < PICK_RETRIES:
                    print(f"  pick missed (gap {[f'{float(g) * 1e3:.1f}' for g in gap]} mm), retrying", flush=True)
                    picks += 1
                    phase, marker = "pick_hover", i
                else:
                    print("  ABORT: pick failed", flush=True)
                    phase, marker = "retreat", i
        elif phase == "lift":  # straight up to carry height, holding the as-picked orientation
            if t_in == 1:
                glide_from_p[:] = key.data.root_pos_w
                q0 = key.data.root_quat_w
                glide_q0[:] = torch.where(q0[:, :1] >= 0, q0, -q0)
            s = smoothstep(t_in / LIFT_STEPS)
            kp = glide_from_p.clone()
            kp[:, 2] = glide_from_p[:, 2] + s * (table_z + CARRY_TIP_Z - glide_from_p[:, 2])
            tp, tq = hand_for_key(kp, glide_q0)
            act = servo(tp, tq, PICK_W)
            if (t_in >= LIFT_STEPS and bool(((table_z + CARRY_TIP_Z - key.data.root_pos_w[:, 2]).abs() < 0.015).all())) \
                    or t_in >= LIFT_STEPS + 40:
                phase, marker = "erect", i
        elif phase == "erect":  # pitch the welded key tip-down ABOUT ITS HANDLE AXIS (the handle
            # stays put, the arm sweeps down); the target yaw is the bolt's hex sector nearest the
            # current handle heading, so the erection IS the clocking and everything after it only
            # translates. The rotation glides — a step-jump goal would swing the gravity-
            # uncompensated arm wide.
            if t_in == 1:
                h0 = handle_heading()
                dp = (yaw_of(bolt.data.root_quat_w) - h0) % (math.pi / 3)
                dp = torch.where(dp > math.pi / 6, dp - math.pi / 3, dp)
                psi_cmd[:] = h0 + dp
                q_tgt = quat_from_angle_axis(psi_cmd, ez)
                q0 = key.data.root_quat_w
                glide_q0[:] = torch.where(q0[:, :1] >= 0, q0, -q0)
                qe = quat_mul(q_tgt, quat_conjugate(glide_q0))
                qe = torch.where(qe[:, :1] >= 0, qe, -qe)
                glide_aa[:] = axis_angle_from_quat(qe)
                glide_from_p[:] = key.data.root_pos_w
            s = smoothstep(t_in / ERECT_STEPS)
            ang = glide_aa.norm(dim=-1).clamp_min(1e-9)
            q_cmd = quat_mul(quat_from_angle_axis(ang * s, glide_aa / ang.unsqueeze(-1)), glide_q0)
            tp, tq = hand_for_key(glide_from_p, q_cmd)  # tip holds its spot; only the quat sweeps
            act = servo(tp, tq, PICK_W)
            upright = key_up() > 0.995
            if t_in >= ERECT_STEPS + 10 and bool(upright.all()):
                phase, marker = "carry", i
            elif t_in >= ERECT_STEPS + 2 * WP_TIMEOUT:
                print(f"  ABORT: erection stalled (key up_z {float(key_up().min()):+.2f})", flush=True)
                phase, marker = "retreat", i
        elif phase == "carry":  # translate the upright key to the hover over the bore, learning
            # the tip's pose bias in free air on arrival (the tip IS the body origin, so the
            # learned offset absorbs the lean-induced tip shift too)
            if t_in == 1:
                glide_from_p[:] = key.data.root_pos_w
                pos_off.zero_()
            goal = torch.zeros(n, 3, device=dev)
            goal[:, 0:2] = bolt.data.root_pos_w[:, 0:2]
            goal[:, 2] = bolt.data.root_pos_w[:, 2] + SOCKET_MOUTH_Z + INSERT_HOVER
            s = smoothstep(t_in / CARRY_STEPS)
            kp = glide_from_p + (goal - glide_from_p) * s
            if s >= 1.0:
                pos_off[:] = (pos_off + 0.25 * (goal - key.data.root_pos_w)).clamp(-0.12, 0.12)
                learn_tilt()
            psi_cmd[:] = yaw_of(key.data.root_quat_w) + clock_err()  # live trim: the wrench-staged
            # bolt's yaw is whatever the helix left; the trim is <= 30 deg by construction
            tp, tq = hand_for_key(kp + pos_off, upright_cmd(psi_cmd))
            act = servo(tp, tq, PICK_W)
            settled = (goal - key.data.root_pos_w).norm(dim=-1) < 0.0008
            clocked = clock_err().abs() < math.radians(3.0)
            if (t_in >= CARRY_STEPS + 15 and bool((settled & clocked).all())) or t_in >= CARRY_STEPS + 3 * WP_TIMEOUT:
                if not bool(bolt_ok().all()):
                    print(f"  ABORT: bolt left its stage before insertion ({bolt_state()})", flush=True)
                    phase, marker = "retreat", i
                else:
                    peck_tries = 0
                    seat_ok.zero_()
                    start_insert_glide()
                    phase, marker = "insert", i
        elif phase == "insert":  # straight-down descent through the socket mouth to the floor; a
            # rim-stall (sharp hex on sharp mouth, 0.75 mm/side) rises and re-drops with a fresh
            # free-air bias — the descent that lines up drops in and presses to the floor
            s = smoothstep(t_in / INSERT_STEPS)
            kp = torch.zeros(n, 3, device=dev)
            kp[:, 0:2] = bolt.data.root_pos_w[:, 0:2]
            kp[:, 2] = bolt.data.root_pos_w[:, 2] + glide_from_z + s * (SOCKET_FLOOR_Z - glide_from_z)
            tp, tq = hand_for_key(kp + pos_off, upright_cmd(psi_cmd))
            act = servo(tp, tq, PICK_W)
            # "in far enough": ~5 mm of hex engagement stands through the handoff — the LOW
            # horizontal hand may saturate a hair short of the floor, and the first top-down
            # stroke presses the last bit home anyway
            seated = (tip_axial() < SOCKET_FLOOR_Z + 0.0022) & (tip_lateral() < 0.002) & (key_up() > 0.99)
            seat_ok[:] = torch.where(seated & bolt_ok(), seat_ok + 1, torch.zeros_like(seat_ok))
            stalled = t_in >= INSERT_STEPS + 12 and bool((tip_axial() > SOCKET_MOUTH_Z - 0.0015).any())
            if not bool(bolt_ok().all()):
                print(f"  ABORT: bolt knocked out of its stage during insertion ({bolt_state()})", flush=True)
                phase, marker = "retreat", i
            elif bool((seat_ok >= 5).all()):
                print(f"  inserted: tip {float((tip_axial() - SOCKET_FLOOR_Z).mean()) * 1e3:+.2f} mm "
                      f"off the floor after {peck_tries} pecks, lateral "
                      f"{float(tip_lateral().max()) * 1e3:.2f} mm", flush=True)
                phase, marker = "handoff", i
            elif stalled or t_in >= INSERT_STEPS + 45:
                peck_tries += 1
                insert_pecks += 1
                if peck_tries > INSERT_RETRIES:
                    drops += 1
                    print("  DROP: insertion never entered the socket", flush=True)
                    phase, marker = "retreat", i
                else:  # rise clear, let the hover re-learn the bias, drop again
                    phase, marker = "insert_retry", i
        elif phase == "insert_retry":  # back to the hover: free air, re-learn, re-trim the clock
            goal = torch.zeros(n, 3, device=dev)
            goal[:, 0:2] = bolt.data.root_pos_w[:, 0:2]
            goal[:, 2] = bolt.data.root_pos_w[:, 2] + SOCKET_MOUTH_Z + 0.006
            settled = (goal - key.data.root_pos_w).norm(dim=-1) < 0.0008
            if t_in > 10:
                pos_off[:] = (pos_off + 0.25 * (goal - key.data.root_pos_w)).clamp(-0.12, 0.12)
                learn_tilt(0.1)
            psi_cmd[:] = yaw_of(key.data.root_quat_w) + clock_err()
            tp, tq = hand_for_key(goal + pos_off, upright_cmd(psi_cmd))
            act = servo(tp, tq, PICK_W)
            if (t_in >= 20 and bool(settled.all())) or t_in >= WP_TIMEOUT:
                seat_ok.zero_()
                start_insert_glide()
                phase, marker = "insert", i
        elif phase == "handoff":  # the key stands in the socket — let go WITHOUT knocking it over.
            # The unheld key MUST lean toward its handle (off-axis weight) until the hex binds
            # (~12-18 deg, a stable tip+wall+rim tripod), and the pads flank the arm exactly
            # along that lean line (fixed at the pick, rigid through the weld) — so the key
            # settles ONTO a pad. The release therefore: unloads the press + torsion while still
            # welded, then GLIDES the fingers open so the pad lowers the leaning key gently onto
            # its bind instead of whipping it past (a step-open ejected it), dwells, and only
            # then backs the hand straight out of the grip corridor and rises.
            if t_in == 1:
                grip_freeze[:] = art.data.joint_pos[:, fingers]
                back_axis[:] = -quat_apply(hq, ez)  # -approach: straight back out of the grip
                psi_cmd[:] = psi_cmd + math.radians(2.0)  # bleed the clock-servo's wind-up
            if t_in <= 10:  # still welded: lift the press off the floor, unload the torsion
                ub = up_axis_of(bolt.data.root_quat_w)
                kp = bolt.data.root_pos_w + ub * (SOCKET_FLOOR_Z + 0.0005)
                tp, tq = hand_for_key(kp + pos_off, upright_cmd(psi_cmd))
                act = servo(tp, tq, grip_freeze)
                if t_in == 10:
                    release_p[:] = hp
                    release_q[:] = hq
            else:
                if welded.any():
                    weld_off()
                w = grip_freeze + (OPEN_W - grip_freeze) * smoothstep((t_in - 10) / 20.0)
                wp2 = release_p.clone()
                if t_in > 50:  # fingers spread + the lean settled before the hand moves
                    s2 = smoothstep((t_in - 50) / 25.0)
                    wp2 += back_axis * (0.06 * s2)
                if t_in > 75:
                    s3 = smoothstep((t_in - 75) / 25.0)
                    wp2[:, 2] = release_p[:, 2] + s3 * 0.10
                act = servo(wp2, release_q, w)
            if t_in % 8 == 0:
                print(f"    [handoff {t_in:3d}] tip {float((tip_axial() - SOCKET_FLOOR_Z).mean()) * 1e3:+5.2f}mm "
                      f"| lat {float(tip_lateral().max()) * 1e3:4.2f}mm | up {float(key_up().min()):+.3f} "
                      f"| gap {float((art.data.joint_pos[:, fingers].sum(dim=-1)).mean()) * 1e3:4.1f}mm", flush=True)
            if t_in >= 105:
                if bool(in_socket(0.0015).all()) and bool((key_up() > 0.90).all()):
                    if handoff_depth is None:
                        handoff_depth = depth().clone()
                    pos_off.zero_()  # learned in the horizontal-hand configuration — stale for
                    # the top-down regrasp; better to relearn from zero
                    regrasp_tries = 0
                    phase, marker = "regrasp_hover", i
                else:
                    drops += 1
                    if drops > DROP_BUDGET:
                        print("  ABORT: drop budget exhausted", flush=True)
                        phase, marker = "retreat", i
                    elif bool((key_up() < 0.5).all()):  # fell flat: pick it up wherever it lies
                        print("  DROP: key left the socket during the handoff — re-picking", flush=True)
                        picks += 1
                        phase, marker = "pick_hover", i
                    else:
                        print("  DROP: key adrift after the handoff", flush=True)
                        phase, marker = "retreat", i
        elif phase == "regrasp_hover":  # top-down hover over the STANDING arm: hand x runs along
            # the handle (the only headings whose finger bodies clear it), pads will land across
            # the hex flats; the free-air dwell learns the pad bias before the descent. The hand
            # arrives HORIZONTAL from the handoff, so the reorientation glides as one axis-angle
            # sweep (the top-down yaw heuristics are degenerate here) and the 180-deg grip parity
            # is picked by the smaller total hand rotation.
            if t_in == 1:
                glide_from_p[:] = hp
                psi_h = handle_heading()
                alt = _wrap(psi_h - math.pi)
                ang_a = 2 * torch.arccos(quat_mul(q_down(psi_h), quat_conjugate(hq))[:, 0].abs().clamp(max=1.0))
                ang_b = 2 * torch.arccos(quat_mul(q_down(alt), quat_conjugate(hq))[:, 0].abs().clamp(max=1.0))
                grip_yaw[:] = torch.where(ang_a <= ang_b, psi_h, alt)
                glide_q0[:] = torch.where(hq[:, :1] >= 0, hq, -hq)
                qe = quat_mul(q_down(grip_yaw), quat_conjugate(glide_q0))
                qe = torch.where(qe[:, :1] >= 0, qe, -qe)
                glide_aa[:] = axis_angle_from_quat(qe)
            up_k = up_axis_of(key.data.root_quat_w)
            grip_pt[:] = key.data.root_pos_w + up_k * GRIP_UP
            wp_p[:] = grip_pt
            wp_p[:, 2] = grip_pt[:, 2] + hand_to_pad + HOVER_CLEAR
            s = smoothstep(t_in / WP_TIMEOUT)
            ang = glide_aa.norm(dim=-1).clamp_min(1e-9)
            wp_q[:] = quat_mul(quat_from_angle_axis(ang * s, glide_aa / ang.unsqueeze(-1)), glide_q0)
            if s >= 1.0:
                want = grip_pt.clone()
                want[:, 2] += HOVER_CLEAR
                pos_off[:] = (pos_off + 0.3 * (want - pad_centre())).clamp(-0.12, 0.12)
            goal = wp_p + pos_off
            act = servo(glide_from_p + (goal - glide_from_p) * s, wp_q, STRADDLE_W)
            pad_err = (pad_centre()[:, 0:2] - grip_pt[:, 0:2]).norm(dim=-1)
            if (t_in >= WP_TIMEOUT + 10 and bool((pad_err < 0.004).all()) and bool(at(goal, wp_q).all())) \
                    or t_in >= 3 * WP_TIMEOUT:
                phase, marker = "regrasp_down", i
        elif phase == "regrasp_down":  # descend around the standing arm to the grip band
            s = smoothstep(t_in / 40.0)
            up_k = up_axis_of(key.data.root_quat_w)
            grip_pt[:] = key.data.root_pos_w + up_k * GRIP_UP
            wp_p[:] = grip_pt
            wp_p[:, 2] = grip_pt[:, 2] + hand_to_pad + HOVER_CLEAR * (1.0 - s)
            if s >= 1.0:
                pos_off[:] = (pos_off + 0.25 * (grip_pt - pad_centre())).clamp(-0.12, 0.12)
            act = servo(wp_p + pos_off, wp_q, STRADDLE_W)
            pad_on = (pad_centre() - grip_pt).norm(dim=-1) < 0.003
            if (t_in >= 48 and bool(pad_on.all())) or t_in >= 2 * WP_TIMEOUT:
                close_ok.zero_()
                phase, marker = "regrasp_close", i
        elif phase == "regrasp_close":
            s = smoothstep(t_in / CLOSE_STEPS)
            width = STRADDLE_W + (SCREW_W - STRADDLE_W) * s
            act = servo(wp_p + pos_off, wp_q, width)
            gap = art.data.joint_pos[:, fingers].sum(dim=-1)
            up_k = up_axis_of(key.data.root_quat_w)  # closure verified against the LIVE key pose
            grip_pt[:] = key.data.root_pos_w + up_k * GRIP_UP
            near = (pad_centre() - grip_pt).norm(dim=-1) < 0.004
            ok = near & (gap > CLOSED_MIN) & (gap < CLOSED_MAX)
            close_ok[:] = torch.where(ok, close_ok + 1, torch.zeros_like(close_ok))
            if t_in >= CLOSE_STEPS + 6 and bool((close_ok >= 4).all()):
                weld_on()
                cycles = 0
                bolt_turn.zero_()  # the ratchet metrics start here: forget the staging's turns
                key_turn.zero_()
                prev_bolt_yaw = yaw_of(bolt.data.root_quat_w)
                prev_key_yaw = yaw_of(key.data.root_quat_w)
                print(f"  re-grasped the standing key top-down (gap "
                      f"{[f'{float(g) * 1e3:.1f}' for g in gap]} mm) — ratcheting", flush=True)
                phase, marker = "stroke", i
            elif t_in >= CLOSE_STEPS + 30:
                regrasp_tries += 1
                if regrasp_tries < 4 and bool(in_socket(0.0015).all()) and bool((key_up() > 0.85).all()):
                    print(f"  regrasp missed (gap {[f'{float(g) * 1e3:.1f}' for g in gap]} mm), retrying", flush=True)
                    phase, marker = "regrasp_hover", i
                elif bool((key_up() < 0.5).all()):
                    drops += 1
                    picks += 1
                    if drops > DROP_BUDGET:
                        print("  ABORT: drop budget exhausted", flush=True)
                        phase, marker = "retreat", i
                    else:
                        print("  DROP: key fell during the regrasp — re-picking", flush=True)
                        phase, marker = "pick_hover", i
                else:
                    print("  ABORT: regrasp failed", flush=True)
                    phase, marker = "retreat", i
        elif phase == "stroke":  # press + twist: the key drives the bolt down the threads until
            # the wrist nears its stop. The tip is commanded BELOW the live socket floor (the
            # press feed rides the descending bolt); the yaw sweeps at a fixed rate so the torque
            # comes from a small, bounded tracking lag.
            if t_in == 1:
                psi_cmd[:] = yaw_of(key.data.root_quat_w)
                stroke_slip0[:] = key_turn - bolt_turn
                slip_armed[:] = False
                peck_tries = 0
                cycles += 1
            # arm the slip watch only once the key reads upright: righting a leaned key changes
            # its READ yaw without any true hex slip, which tripped the guard as a phantom
            newly_up = (key_up() > 0.995) & ~slip_armed
            stroke_slip0[:] = torch.where(newly_up, key_turn - bolt_turn, stroke_slip0)
            slip_armed |= newly_up
            room = J7_GUARD - art.data.joint_pos[:, j7]
            sweep = torch.full_like(room, STROKE_W / 15.0)
            psi_cmd[:] = psi_cmd - torch.where(room > 0.15, sweep, torch.zeros_like(sweep))
            ub = up_axis_of(bolt.data.root_quat_w)
            kp = bolt.data.root_pos_w + ub * (SOCKET_FLOOR_Z - PRESS_LEAD)
            tp, tq = hand_for_key(kp + pos_off, quat_from_angle_axis(psi_cmd, ez))
            act = servo(tp, tq, SCREW_W)
            # screw-in sweeps NEGATIVE yaw, so a camming key runs AHEAD of the bolt in the
            # negative direction — watch the magnitude, not one sign
            slip = ((key_turn - bolt_turn) - stroke_slip0).abs() * slip_armed
            done = bool((depth() >= STOP_DEPTH).all())
            wound = bool((room <= 0.15).all())
            slipped = bool((slip > SLIP_ABORT).any())
            popped = bool((tip_axial() > SOCKET_MOUTH_Z).any())
            if not bool(bolt_ok(up_min=0.95).all()):
                print(f"  ABORT: bolt left the hole mid-stroke ({bolt_state()})", flush=True)
                phase, marker = "retreat", i
            elif done or wound or slipped or popped or t_in >= STROKE_TIMEOUT:
                # per-stroke hex-drive fidelity: the deliberate rewinds are excluded by design
                drive_slip += ((key_turn - bolt_turn) - stroke_slip0) * slip_armed
                if slipped:
                    print(f"  stroke {cycles}: hex slipped {float(torch.rad2deg(slip.max())):.0f} deg — recocking", flush=True)
                phase, marker = "unload", i
        elif phase == "unload":  # bleed the torsional wind-up BEFORE lifting: back-rotate a few
            # degrees while still pressed — the stroke leaves the hex corners jammed against the
            # socket walls, and lifting under that stored torsion kicks the key as it disengages
            if t_in == 1:
                psi_cmd[:] = psi_cmd + math.radians(4.0)
            ub = up_axis_of(bolt.data.root_quat_w)
            kp = bolt.data.root_pos_w + ub * (SOCKET_FLOOR_Z - PRESS_LEAD)
            tp, tq = hand_for_key(kp + pos_off, quat_from_angle_axis(psi_cmd, ez))
            act = servo(tp, tq, SCREW_W)
            if t_in >= 6:
                if bool((depth() >= STOP_DEPTH).all()) or cycles >= MAX_CYCLES:
                    if cycles >= MAX_CYCLES and not bool((depth() >= STOP_DEPTH).all()):
                        print("  cycle budget exhausted", flush=True)
                    phase, marker = "release", i
                else:
                    phase, marker = "recock_lift", i
        elif phase == "recock_lift":  # lift the tip just clear of the socket, yaw held; the glide
            # starts from the PRESSED command depth so the stored press bleeds smoothly
            s = smoothstep(t_in / 20.0)
            zt = (SOCKET_FLOOR_Z - PRESS_LEAD) + s * (SOCKET_MOUTH_Z + LIFT_CLEAR - SOCKET_FLOOR_Z + PRESS_LEAD)
            ub = up_axis_of(bolt.data.root_quat_w)
            kp = bolt.data.root_pos_w + ub * zt
            tp, tq = hand_for_key(kp + pos_off, quat_from_angle_axis(psi_cmd, ez))
            act = servo(tp, tq, SCREW_W)
            cleared = bool((tip_axial() > SOCKET_MOUTH_Z + 0.002).all())
            if t_in >= 24 and cleared:
                # rewind by a MULTIPLE OF 60 DEG (hex symmetry: clocking is preserved exactly),
                # sized to re-arm the wrist near J7_START; the flipped top-down hand maps a
                # positive key yaw to a negative joint-7 move
                steps60 = torch.round((art.data.joint_pos[:, j7] - J7_START) / (math.pi / 3))
                rewind_tgt[:] = psi_cmd + steps60 * (math.pi / 3)
                phase, marker = "recock_rewind", i
            elif t_in >= 2 * WP_TIMEOUT:  # NEVER rewind while the hex is engaged (it would just
                # unscrew the bolt): a tip that cannot disengage is an honest failure
                print("  ABORT: key would not lift out of the socket", flush=True)
                phase, marker = "retreat", i
        elif phase == "recock_rewind":  # swept, monotone yaw glide the LONG way round — a plain
            # quat target would take the short path and wind joint 7 THROUGH its stop. Arrival is
            # LATCHED; the hex-slip correction is a ONE-SHOT at arrival plus a gentle trickle —
            # a fat per-step trim servo limit-cycles around the +-30 deg sector boundary.
            if t_in == 1:
                rw_arrived[:] = False
            newly_arr = ((rewind_tgt - psi_cmd) < 1e-6) & ~rw_arrived
            rw_arrived |= newly_arr
            psi_cmd[:] = torch.where(newly_arr, psi_cmd + clock_err(),
                                     torch.where(rw_arrived, psi_cmd + 0.1 * clock_err(),
                                                 torch.minimum(psi_cmd + REWIND_W / 15.0, rewind_tgt)))
            ub = up_axis_of(bolt.data.root_quat_w)
            kp = bolt.data.root_pos_w + ub * (SOCKET_MOUTH_Z + LIFT_CLEAR)
            if bool(rw_arrived.all()) and t_in % 2 == 0:  # settled free air: re-learn tip bias
                pos_off[:] = (pos_off + 0.2 * ((kp - key.data.root_pos_w))).clamp(-0.12, 0.12)
                learn_tilt(0.1)
            tp, tq = hand_for_key(kp + pos_off, upright_cmd(psi_cmd))
            act = servo(tp, tq, SCREW_W)
            # the reinsert only gets a sub-clearance start: xy on the bore axis to half the
            # 0.75 mm/side play, tip height settled, hex clocked
            xy_err = (key.data.root_pos_w[:, 0:2] - bolt.data.root_pos_w[:, 0:2]).norm(dim=-1)
            settled = (xy_err < 0.0005) & ((kp[:, 2] - key.data.root_pos_w[:, 2]).abs() < 0.0012)
            clocked = clock_err().abs() < math.radians(2.5)
            if bool(rw_arrived.all()) and t_in >= 10 and bool((settled & clocked).all()):
                seat_ok.zero_()
                start_insert_glide()
                phase, marker = "reinsert", i
            elif t_in >= 2 * (REWIND_TIMEOUT + int(4.8 / (REWIND_W / 15.0))):
                print(f"  ABORT: rewind never settled (xy {float(xy_err.max()) * 1e3:.2f}mm, clock "
                      f"{float(torch.rad2deg(clock_err().abs().max())):.1f}deg)", flush=True)
                phase, marker = "retreat", i
        elif phase == "reinsert":  # drop the tip back into the socket and press to the floor
            s = smoothstep(t_in / 25.0)
            zt = glide_from_z + s * ((SOCKET_FLOOR_Z - PRESS_LEAD) - glide_from_z)
            ub = up_axis_of(bolt.data.root_quat_w)
            kp = bolt.data.root_pos_w + ub * zt
            tp, tq = hand_for_key(kp + pos_off, upright_cmd(psi_cmd))
            act = servo(tp, tq, SCREW_W)
            entered = (tip_axial() < SOCKET_FLOOR_Z + 0.0015) & (tip_lateral() < 0.002)
            seat_ok[:] = torch.where(entered & bolt_ok(), seat_ok + 1, torch.zeros_like(seat_ok))
            stalled = t_in >= 35 and bool((tip_axial() > SOCKET_MOUTH_Z - 0.0015).any())
            if not bool(bolt_ok().all()):
                print(f"  ABORT: bolt left the hole during a reinsert ({bolt_state()})", flush=True)
                phase, marker = "retreat", i
            elif bool((seat_ok >= 4).all()):
                phase, marker = "stroke", i
            elif stalled or t_in >= REINSERT_TIMEOUT:
                peck_tries += 1
                insert_pecks += 1
                if peck_tries > INSERT_RETRIES:
                    print("  ABORT: reinsert never re-entered the socket", flush=True)
                    phase, marker = "retreat", i
                else:  # rise a few mm and re-run the rewind's settle (fresh bias + clock trim)
                    rewind_tgt[:] = psi_cmd  # no further rewind: just re-settle and drop again
                    phase, marker = "recock_rewind", i
        elif phase == "release":  # seated (or budget spent): bleed, let go, rise straight off the
            # standing key — the top-down fingers open past the handle's flanks, the palm is
            # already above its top
            if t_in == 1:
                grip_freeze[:] = art.data.joint_pos[:, fingers]
                release_p[:] = hp
                release_q[:] = hq
            if t_in <= 8:
                act = servo(release_p, release_q, grip_freeze)
            else:
                if welded.any():
                    weld_off()
                wp2 = release_p.clone()
                if t_in > 22:  # open FIRST, then rise: rising frozen pads would drag the key up
                    s2 = smoothstep((t_in - 22) / 30.0)
                    wp2[:, 2] = release_p[:, 2] + s2 * 0.12
                act = servo(wp2, release_q, STRADDLE_W if t_in > 10 else grip_freeze)
            if t_in >= 60:
                phase, marker = "retreat", i
        elif phase == "retreat":  # glide home — position AND orientation
            if t_in == 1:
                if welded.any():
                    weld_off()
                glide_from_p[:] = hp
                glide_q0[:] = hq
                qe = quat_mul(home_q, quat_conjugate(hq))
                qe = torch.where(qe[:, :1] >= 0, qe, -qe)
                glide_aa[:] = axis_angle_from_quat(qe)
            s = smoothstep(t_in / RETREAT_STEPS)
            ang = glide_aa.norm(dim=-1).clamp_min(1e-9)
            q_cmd = quat_mul(quat_from_angle_axis(ang * s, glide_aa / ang.unsqueeze(-1)), glide_q0)
            act = servo(glide_from_p + (home_p - glide_from_p) * s, q_cmd, OPEN_W)
            if t_in >= RETREAT_STEPS + 10:
                phase, marker = "settle", i
        else:  # settle: hands off — the seated bolt and the self-locking thread hold on their own
            act = servo(home_p, home_q, OPEN_W)
            if t_in >= SETTLE_STEPS:
                break

        step(act)

        # Track yaws EVERY step (a stale prev across the staging spin reads as a +-pi jump), but
        # accumulate the turn counters only once the robot phase is live.
        cur = yaw_of(bolt.data.root_quat_w)
        kcur = yaw_of(key.data.root_quat_w)
        if phase not in ("show", "stage_drop", "stage_hold"):
            bolt_turn = bolt_turn - _wrap(cur - prev_bolt_yaw)
            key_turn = key_turn - _wrap(kcur - prev_key_yaw)
        prev_bolt_yaw, prev_key_yaw = cur, kcur

        if not torch.isfinite(bolt.data.root_pos_w).all() or not torch.isfinite(key.data.root_pos_w).all():
            print("  ABORT: state went non-finite", flush=True)
            break
        if i >= HARD_CAP:  # every phase carries its own timeout; this only guards a livelock
            print("  ABORT: global step budget exhausted", flush=True)
            break
        if i % LOG_EVERY == 0:
            d = depth() * 1e3
            slip = torch.rad2deg(key_turn - bolt_turn)
            q7d = torch.rad2deg(art.data.joint_pos[:, j7])
            gap = art.data.joint_pos[:, fingers].sum(dim=-1) * 1e3
            print(f"  ctrl {i:5d} [{phase:13s}] | cyc {cycles:2d} | tip depth {float(d.mean()):+6.2f}mm | bolt "
                  f"{float(torch.rad2deg(bolt_turn).mean()):+7.0f}deg | slip {float(slip.mean()):+6.1f}deg | "
                  f"q7 {float(q7d.mean()):+6.0f}deg | key z {float(key.data.root_pos_w[:, 2].mean()):.3f} "
                  f"up {float(key_up().mean()):+.2f} | hand z {float(hp[:, 2].mean()):.3f} | "
                  f"gap {float(gap.mean()):4.1f}mm", flush=True)

    if writer is not None:
        writer.close()
        print("MP4:", args.video, flush=True)

    seated = sc.seated()
    d = depth() * 1e3
    gain = (d - handoff_depth * 1e3) if handoff_depth is not None else d - staged_depth * 1e3
    revs = bolt_turn / (2 * math.pi)
    turned = revs > 0.5
    mm_per_rev = float((gain[turned] / revs[turned]).median()) if turned.any() else float("nan")
    slip = torch.rad2deg(drive_slip)  # within-stroke only: the rewinds slip on purpose
    print(f"ALLEN-KEY-FRANKA | seated {int(seated.all(dim=1).sum())}/{n} | staged "
          f"{float(staged_depth.mean()) * 1e3:.1f}mm, key drove {float(gain.mean()):+.1f}mm over "
          f"{float(revs.mean()):.1f} revs = {mm_per_rev:.2f} mm/rev (pitch {PITCH_MM:.1f}) | "
          f"stroke slip {float(slip.mean()):+.1f}deg | {cycles} cycles, {picks} picks, "
          f"{insert_pecks} pecks, {drops} drops | tip depth mm: min={float(d.min()):+.1f} "
          f"mean={float(d.mean()):+.1f} max={float(d.max()):+.1f} "
          f"(seat >= {sc.cfg.seat_depth * 1e3:.0f})", flush=True)
    close_and_exit(env, app)


if __name__ == "__main__":
    main()

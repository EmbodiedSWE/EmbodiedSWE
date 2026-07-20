"""Franka smoke for AllenBoltAssemblyScene — the arm picks the allen key off the table, seats it
in the staged bolt's hex socket, and ratchet-screws the bolt down the platform's real SDF threads.

Grasping follows the benchmark weld-on-closure contract (cf. the pouring suite / so101 fastening):
a normally-disabled FixedJoint hand<->key is enabled when the gripper is verifiably closed around
the key and released when it opens. Everything else is live physics — key<->socket hex contact,
bolt<->platform thread contact, and the key standing unheld in the socket between grips — so a
missed grasp, a jammed insertion, or a dropped key fails honestly.

The wrist's ±166 deg joint-7 range cannot turn the ~12 revolutions a full seat needs, so the smoke
ratchets like a hand on a stubby key: press + twist a stroke, release, lift clear of the handle,
unwind the wrist to a fresh hex-aligned grip on the short arm (targets recomputed from the key's
live pose), re-close, repeat. The bolt itself is teleport-staged upright over the hole exactly as
in the force-driven sibling smoke — the robot's job here is the key.

Phases: show -> stage -> pick -> lift -> unwind -> carry -> erect -> aim -> clock ->
insert(peck lattice) -> handoff -> regrasp -> ratchet(stroke/rewind cycles, with drop recovery
via re-grasp + re-insert) -> retreat -> settle. Verdict: seated count, depth gained per rev vs
the 2.0 mm pitch, key->bolt slip, ratchet cycles, and drops.

The ratchet drives at the thread's exact pitch (2.00 mm/rev measured); full seating needs
~19 clean cycles and the release windows carry a run-to-run extraction risk the recovery
path absorbs, so headline depth varies run to run — the verdict prints the honest numbers.

cosigen/bin/python -m robobench.suites.assembly.smokes.allen_key_franka_smoke --headless
python -m robobench.suites.assembly.smokes.allen_key_franka_smoke --livestream 2
python -m robobench.suites.assembly.smokes.allen_key_franka_smoke \
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

# Key geometry baked into the committed key USD (tip at origin, 50 mm working arm up +z, 120 mm
# handle +x, hex 12.6 mm across flats / 14.4 mm across corners):
ARM_LEN = 0.050
HEX_AF = 0.0126
# Bolt-local geometry (bolt origin = thread TIP, +z up): socket recess floor and mouth (head top).
SOCKET_FLOOR_Z = 0.0355
SOCKET_MOUTH_Z = 0.0428
STAGE_GAP = 0.0015       # staging gap (m) between bolt tip and plate top, just above the thread entry
STOP_DEPTH = 0.0235      # stop twisting at this tip depth (m) — just before the head bottoms at 24.8 mm
PITCH_MM = 2.0           # M16 coarse pitch, the expected descent per revolution
DT = 1.0 / 240.0         # sim timestep (matches the registered env's dt override)

# Franka OSC action semantics (FrankaRobot: 6 EE pose deltas + 2 finger position targets at 15 Hz;
# deltas latch as target = current EE pose + action * scale, torque recomputed every substep).
POS_SCALE, ROT_SCALE = 0.02, 0.097
# The OSC carries no gravity compensation, and a 1-unit rotation lead (0.097 rad) yields less
# torque than the hand+key gravity moment — a 90-deg pitch stalls at its equilibrium angle, worst
# as the hand reaches horizontal. Let the servo lead rotations by up to this many units (the lead
# equals the remaining error once inside ~45 deg, so it tapers naturally).
ROT_SAT = 8.0
OPEN_W, CLOSE_W = 0.04, 0.0
# panda_finger body origin -> finger-pad centre, along the hand's approach axis (the hand-frame ->
# pad distance itself is measured live at reset: hand -> finger base + this).
FINGER_TO_PAD = 0.045
# Gripper closure window on the key (sum of the two finger joints, m): the hex is 12.6 mm across
# flats / 14.4 mm across corners, so a real grip stalls in [5, 17] mm; near 0 means grabbed air.
CLOSED_MIN, CLOSED_MAX = 0.005, 0.017

# Choreography (all waypoints are recomputed from live poses; distances in m, angles in rad):
PICK_GRIP_D = 0.028      # pick grip point: this far up the lying working arm from the tip
STROKE_GRIP_UP = 0.030   # ratchet grip point: this far above the key tip (5 mm below the elbow underside)
CARRY_Z = 0.24           # table-frame height the key tip is carried/erected/aimed at — the 120 mm
                         # handle sweeps a full arc during aim and rides several cm below command
                         # (rotation transient + horizontal-grip droop), and must clear the bolt head
INSERT_HOVER = 0.035     # tip hover above the socket mouth while clocking: the transit into the
                         # hover overshoots vertically by up to ~8 mm, so it must clear the head
INSERT_DEPTH = 0.006     # tip this deep into the recess (below the mouth) triggers the handoff
PRESS_DZ = 0.004         # stroke press: command the tip this far below the live socket floor
# Insertion relies on the self-centring hex clocking alone — a yaw wiggle just drags the free
# bolt around by rim friction (the slick bolt spins at a touch), so the relative phase never sweeps.
WIGGLE_DEG, WIGGLE_HZ = 0.0, 1.5
STROKE_RAD = math.radians(180.0)   # nominal ratchet stroke sweep (screw-in = negative yaw)
STROKE_W = 1.2           # commanded stroke yaw rate (rad/s), under the 1.455 rad/s action cap
J7_GUARD = 2.6           # end the stroke when |panda_joint7| exceeds this (limit 2.897 rad)
J7_START = math.radians(-105.0)    # preferred joint-7 angle at stroke start: the screw-in sweep
                                   # (key yaw negative) WINDS j7 positive on the flipped hand, so
                                   # start low and end near +75 deg
REWIND_LIFT = 0.045      # lift between strokes so the open fingers clear the swept handle
MAX_CYCLES = 40          # ratchet cycle budget
PICK_RETRIES = 3

# Waypoint tolerances and per-phase timeouts, in CONTROL steps (15 Hz -> 16 substeps each at 1/240).
TOL_P, TOL_R = 0.004, 0.06
SHOW_END, STAGE_SETTLE = 20, 40
WP_TIMEOUT, CLOSE_STEPS, STROKE_TIMEOUT, SETTLE_STEPS = 75, 12, 90, 45
INSERT_TIMEOUT = 260  # descent from the raised hover + a full 19-point peck lattice
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

    # ----- hand<->key weld (the grasp contract): a pool of pre-authored, disabled FixedJoints ----
    # PhysX latches a joint's local frames when it is FIRST enabled; frame rewrites on a re-enabled
    # joint are ignored. Each grasp therefore consumes a fresh joint from the pool: author the live
    # relative pose while it is still disabled, enable it once, and on release disable it for good.
    from pxr import Gf, UsdPhysics

    stage = env.stage
    WELD_POOL = 3 * MAX_CYCLES  # strokes + drop recoveries (a recovery consumes two extra welds)
    weld_paths = []
    for e in range(n):
        base = f"/World/envs/env_{e}"
        row = []
        for k in range(WELD_POOL):
            j = UsdPhysics.FixedJoint.Define(stage, f"{base}/hand_key_weld_{k}")
            j.CreateBody0Rel().SetTargets([f"{base}/Robot/panda_hand"])
            j.CreateBody1Rel().SetTargets([f"{base}/Key_0/allen_key"])
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateJointEnabledAttr(False)
            j.CreateExcludeFromArticulationAttr(True)  # a maximal-coordinate weld, not a new arm DOF
            row.append(f"{base}/hand_key_weld_{k}")
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

    # Stiffen the OSC's ROTATIONAL gains (open-env retune): the controller has no gravity
    # compensation, so at the stock kp_rot=30 the hand+key gravity moment leaves a ~24 deg pitch
    # droop with the hand horizontal — the erection can never finish. kp_rot=150 puts the static
    # droop under the 5.7 deg uprightness gate; damping stays critical.
    osc = env.robot.controller.controllers[0]
    osc._kp[3:6] = 150.0
    osc._kd[3:6] = 2.0 * math.sqrt(150.0)

    hole_xy = plat.data.root_pos_w[:, :2].clone()  # insert bore axis == platform origin (bore-centred)
    plat_z = plat.data.root_pos_w[:, 2].clone()

    if cam is not None:
        p0 = plat.data.root_pos_w[0]
        eye = torch.tensor([[p0[0] + 0.05, p0[1] - 0.50, p0[2] + 0.35]], dtype=torch.float32, device=dev)
        tgt = torch.tensor([[p0[0], p0[1], p0[2] + 0.08]], dtype=torch.float32, device=dev)
        cam.set_world_poses_from_view(eye, tgt)
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

    def weld_drift() -> torch.Tensor:
        """|key position - where the active weld says it should be| (m); ~0 while a weld holds."""
        hp, hq = hand_pose()
        return (key.data.root_pos_w - (hp + quat_apply(hq, rel_p))).norm(dim=-1)

    # ----- staging (same teleport as the force-driven sibling: the robot's job is the key) --------
    def stage_parts() -> None:
        st = torch.zeros(n, 13, device=dev)
        st[:, 0:2] = hole_xy
        st[:, 2] = plat_z + plate_top + STAGE_GAP
        st[:, 3] = 1.0
        bolt.write_root_state_to_sim(st, ids)

    def depth() -> torch.Tensor:  # bolt tip depth below the plate top (m), per env
        return plat_z + plate_top - bolt.data.root_pos_w[:, 2]

    # ----- servo layer: waypoints -> OSC action ---------------------------------------------------
    def servo(tp: torch.Tensor, tq: torch.Tensor, grip: float) -> torch.Tensor:
        """8-D action toward a hand waypoint: clamped EE pose delta + finger targets."""
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
        top-down hand, where yaw routes through the wrist roll). The hand is FLIPPED (approach -z),
        so a positive world yaw turns joint 7 NEGATIVE."""
        ex = torch.zeros(n, 3, device=dev)
        ex[:, 0] = 1.0
        hx = quat_apply(hand_pose()[1], ex)
        cur = torch.atan2(hx[:, 1], hx[:, 0])
        return art.data.joint_pos[:, j7] - _wrap(cand - cur)

    def nearest_parity(psi: torch.Tensor, prev: torch.Tensor | None = None) -> torch.Tensor:
        """The gripper is 180-deg symmetric: return psi or psi-pi, whichever leaves the wrist roll
        nearer its centre (keeps the servo from winding joint 7 toward a stop). With `prev`, keep
        the previous choice unless the alternative is clearly better (hysteresis for per-tick use —
        the prediction only becomes reliable once the hand is actually top-down)."""
        alt = _wrap(psi - math.pi)
        a, b = j7_after(psi).abs(), j7_after(alt).abs()
        if prev is None:
            return torch.where(a <= b, psi, alt)
        keep_psi = _wrap(prev - psi).abs() < _wrap(prev - alt).abs()  # which candidate matches prev
        margin = math.radians(10.0)
        pick_psi = torch.where(keep_psi, a <= b + margin, a + margin < b)
        return torch.where(pick_psi, psi, alt)

    def hex_rep_for_wrist(psi: torch.Tensor) -> torch.Tensor:
        """Among the six hex-equivalent yaws psi + k*60 deg, the one predicted to leave joint 7
        nearest its centre — the key's clocking only matters mod 60 deg."""
        best, best_score = None, None
        for k in range(6):
            cand = _wrap(psi + k * (math.pi / 3.0))
            score = j7_after(cand).abs()
            if best is None:
                best, best_score = cand, score
            else:
                best = torch.where(score < best_score, cand, best)
                best_score = torch.minimum(score, best_score)
        return best

    def hex_grip_yaw() -> torch.Tensor:
        """Hand yaw for a flat-aligned top-down grip on the upright key's arm: aim the wrist roll at
        `J7_START` (so the negative screw-in stroke has room before the -166 deg stop), and steer
        away from putting a descending finger in line with the horizontal handle."""
        ex = torch.zeros(n, 3, device=dev)
        ex[:, 0] = 1.0
        hh = quat_apply(key.data.root_quat_w, ex)  # handle heading (world)
        psi_h = torch.atan2(hh[:, 1], hh[:, 0])
        ky = yaw_of(key.data.root_quat_w)
        best, best_score = None, None
        for k in range(6):
            cand = _wrap(ky + k * (math.pi / 3.0))
            fin = torch.minimum((_wrap(psi_h - cand - math.pi / 2)).abs(), (_wrap(psi_h - cand + math.pi / 2)).abs())
            score = (j7_after(cand) - J7_START).abs() + torch.where(fin < math.radians(20.0), 10.0, 0.0)
            score = score + torch.where(j7_after(cand).abs() > math.radians(160.0), 100.0, 0.0)  # never target a stop
            if best is None:
                best, best_score = cand, score
            else:
                best = torch.where(score < best_score, cand, best)
                best_score = torch.minimum(score, best_score)
        return best

    def key_tip_axial() -> torch.Tensor:
        """Key tip height above the bolt origin along the bolt axis (m): SOCKET_MOUTH_Z at the recess
        mouth, SOCKET_FLOOR_Z when fully seated on the floor."""
        ub = up_axis_of(bolt.data.root_quat_w)
        return ((key.data.root_pos_w - bolt.data.root_pos_w) * ub).sum(-1)

    def key_lateral() -> torch.Tensor:
        ub = up_axis_of(bolt.data.root_quat_w)
        rel = key.data.root_pos_w - bolt.data.root_pos_w
        return (rel - (rel * ub).sum(-1, keepdim=True) * ub).norm(dim=-1)

    def bolt_ok(say: bool = False, up_min: float = 0.99) -> torch.Tensor:
        """The bolt is still where the choreography needs it: in the hole, near upright. A phase in
        contact with the bolt passes a looser `up_min` — a pressed key rights a tilting bolt, so a
        transient tilt is recoverable there; ejection shows in the axis/depth terms."""
        axis_err = (bolt.data.root_pos_w[:, 0:2] - hole_xy).norm(dim=-1)
        ok = (axis_err < 0.004) & (depth() > 0.001) & (up_axis_of(bolt.data.root_quat_w)[:, 2] > up_min)
        if say and not bool(ok.all()):
            tilt = torch.rad2deg(torch.acos(up_axis_of(bolt.data.root_quat_w)[:, 2].clamp(-1, 1)))
            print(f"    bolt state: axis err {float(axis_err.max()) * 1e3:.1f}mm | depth "
                  f"{float(depth().min()) * 1e3:+.1f}mm | tilt {float(tilt.max()):.1f}deg", flush=True)
        return ok

    # ----- step + capture -------------------------------------------------------------------------
    step_i = 0

    def step(action: torch.Tensor) -> None:
        nonlocal step_i
        step_i += 1
        capture = writer is not None and step_i % args.cap == 0
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
    handoff_depth = None
    cycles, drops, picks, regrasp_tries, aim_tries = 0, 0, 0, 0, 0
    grip_freeze = torch.zeros(n, 2, device=dev)  # finger targets latched at release: freezing the
    # PD at the MEASURED positions decays the squeeze to zero before the pads separate — pads
    # sliding apart under load rub a friction COUPLE into the free key and spin it out of the hex
    seat_ok = torch.zeros(n, device=dev)      # consecutive ticks the inserted key has read settled
    pos_off = torch.zeros(n, 3, device=dev)   # 3-D integral offset (desired - achieved key pos,
    # EMA-learned in FREE AIR only): the OSC has no gravity compensation, so its realized pose
    # carries a configuration-dependent bias of several mm in all axes — commands near the bolt
    # add this offset so the achieved pose lands on the desired one
    uw_legs = torch.zeros(n, 2, device=dev)   # latched unwind yaw waypoints (two 90-deg legs — a
    # single 180-deg yaw error is directionally ambiguous and can push INTO the joint stop)
    uw_leg = torch.zeros(n, dtype=torch.long, device=dev)
    psi_star = torch.zeros(n, device=dev)     # latched hex-aligned yaw target for clock/insert —
    # latched ONCE at phase entry: re-picking the mod-60 sector every tick flaps ±60 deg at the
    # sector boundary and winds the wrist
    stroke_psi = torch.zeros(n, device=dev)   # commanded stroke sweep so far (rad, >= 0)
    stroke_q0 = torch.zeros(n, 4, device=dev)  # key orientation at stroke start
    grip_pt = torch.zeros(n, 3, device=dev)
    grip_yaw = torch.zeros(n, device=dev)
    wp_p = torch.zeros(n, 3, device=dev)
    wp_q = torch.zeros(n, 4, device=dev)

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
                stage_parts()
                prev_bolt_yaw = yaw_of(bolt.data.root_quat_w)
                phase, marker = "stage", i
        elif phase == "stage":  # hands off: the bolt drops the 1.5 mm gap and nests on the crests,
            # which MEASURES a valid (depth, yaw) helix registration — then teleport it deeper
            # along that helix so the threads genuinely capture it. A merely crest-nested bolt is
            # pried loose by ANY insertion contact; a captured one holds against the key's taps.
            if t_in == STAGE_SETTLE // 2:
                d2 = 0.0055
                dyaw = -2 * math.pi * (d2 - depth()) / (PITCH_MM * 1e-3)
                st = torch.zeros(n, 13, device=dev)
                st[:, 0:2] = hole_xy
                st[:, 2] = plat_z + plate_top - d2
                st[:, 3:7] = quat_from_angle_axis(yaw_of(bolt.data.root_quat_w) + dyaw, ez)
                bolt.write_root_state_to_sim(st, ids)
            if t_in >= STAGE_SETTLE:
                print(f"  staged: bolt tip depth {float(depth().mean()) * 1e3:+.2f}mm (thread-captured)", flush=True)
                phase, marker = "pick_hover", i
                picks += 1
        elif phase == "pick_hover":  # top-down over the lying key's working arm, fingers open
            ka = up_axis_of(key.data.root_quat_w)  # arm direction (lying: horizontal)
            grip_pt[:] = key.data.root_pos_w + ka * PICK_GRIP_D
            raw_yaw = torch.atan2(ka[:, 1], ka[:, 0])  # hand x along the arm -> fingers across it
            grip_yaw[:] = nearest_parity(raw_yaw, prev=grip_yaw if t_in > 1 else None)
            wp_p[:] = grip_pt
            wp_p[:, 2] = grip_pt[:, 2] + hand_to_pad + 0.06
            wp_q[:] = q_down(grip_yaw)
            act = servo(wp_p, wp_q, OPEN_W)
            if bool(at(wp_p, wp_q).all()) or t_in >= WP_TIMEOUT:
                phase, marker = "pick_down", i
        elif phase == "pick_down":  # descend until the pads straddle the arm's upper half
            wp_p[:, 2] = grip_pt[:, 2] + 0.006 + hand_to_pad  # pad centre 6 mm above the arm axis: fingertips clear the table
            act = servo(wp_p, wp_q, OPEN_W)
            if bool(at(wp_p, wp_q).all()) or t_in >= WP_TIMEOUT:
                phase, marker = "pick_close", i
        elif phase == "pick_close":
            act = servo(wp_p, wp_q, CLOSE_W)
            if t_in >= CLOSE_STEPS:
                gap = art.data.joint_pos[:, fingers].sum(dim=-1)
                near = (hp[:, :2] - grip_pt[:, :2]).norm(dim=-1) < 0.02
                ok = (gap > CLOSED_MIN) & (gap < CLOSED_MAX) & near
                if bool(ok.all()):
                    weld_on()
                    phase, marker = "lift", i
                elif picks < PICK_RETRIES:
                    print(f"  pick {picks} missed (gap {gap.tolist()}), retrying", flush=True)
                    phase, marker = "pick_hover", i
                    picks += 1
                else:
                    print("  ABORT: pick failed", flush=True)
                    phase, marker = "retreat", i
        elif phase == "lift":
            wp_p[:, 2] = grip_pt[:, 2] + hand_to_pad + 0.12
            act = servo(wp_p, wp_q, CLOSE_W)
            if bool(at(wp_p, wp_q).all()) or t_in >= WP_TIMEOUT:
                phase, marker = "unwind", i
        elif phase == "unwind":  # the pick routinely parks joint 7 at a stop; flip the top-down
            # hand's 180-deg grip parity here (welded, high, clear air) so the erection has wrist
            # mobility — at a stop the 90-deg erect rotation stalls and the key stays flat
            q7v = art.data.joint_pos[:, j7]
            ex = torch.zeros(n, 3, device=dev)
            ex[:, 0] = 1.0
            hx = quat_apply(hq, ex)
            cur_yaw = torch.atan2(hx[:, 1], hx[:, 0])
            if t_in == 1:
                # flipped hand: world yaw +delta -> j7 -delta, so unwinding +q7 needs +yaw legs
                s = torch.sign(q7v) * (q7v.abs() > 2.1)
                uw_legs[:, 0] = _wrap(cur_yaw + s * (math.pi / 2))
                uw_legs[:, 1] = _wrap(cur_yaw + s * math.pi)
                uw_leg.zero_()
            uw_leg[( _wrap(cur_yaw - uw_legs[:, 0]).abs() < 0.2) & (uw_leg == 0)] = 1
            tgt_yaw = torch.where(uw_leg == 0, uw_legs[:, 0], uw_legs[:, 1])
            act = servo(wp_p, q_down(tgt_yaw), CLOSE_W)
            if bool((q7v.abs() < 1.6).all()) or t_in >= 3 * WP_TIMEOUT:
                phase, marker = "carry", i
        elif phase == "carry":  # key (still lying flat in the grip) to the erection spot — OFF-AXIS
            # and high: during the erection the welded key sweeps a wide arc about the HAND, so the
            # whole manoeuvre stays displaced from the staged bolt and well above the plate
            kp = torch.zeros(n, 3, device=dev)
            kp[:, 0] = hole_xy[:, 0]
            kp[:, 1] = hole_xy[:, 1] + 0.10
            kp[:, 2] = plat_z + CARRY_Z
            tp, tq = hand_for_key(kp, key.data.root_quat_w)  # translate only; keep orientation
            act = servo(tp, tq, CLOSE_W)
            if bool(((kp - key.data.root_pos_w).norm(dim=-1) < 0.02).all()) or t_in >= WP_TIMEOUT:
                phase, marker = "erect", i
        elif phase == "erect":  # rotate the welded key arm tip-down — ROTATION-ONLY (the position
            # target rides the live hand pose so nothing fights the 90-deg pitch), toward the ONE
            # latched upright orientation: of the six hex-aligned yaws, the one the hand can reach
            # by the SMALLEST rotation from where it is (the final key yaw is free — deriving it
            # per-tick from the swinging key drifts, and an arbitrary yaw can demand a hand roll
            # that runs a wrist joint into its stop)
            if t_in == 1:
                by = yaw_of(bolt.data.root_quat_w)
                best_q, best_ang = None, None
                for k in range(6):
                    psi = _wrap(by + k * (math.pi / 3.0))
                    kq_c = quat_from_angle_axis(psi, ez)
                    hq_c = quat_mul(kq_c, quat_conjugate(rel_q))
                    qe_c = quat_mul(hq_c, quat_conjugate(hq))
                    ang_c = 2 * torch.arccos(qe_c[:, 0].abs().clamp(max=1.0))
                    if best_q is None:
                        best_q, best_ang, best_psi = hq_c, ang_c, psi
                    else:
                        pick = ang_c < best_ang
                        best_q = torch.where(pick.unsqueeze(-1), hq_c, best_q)
                        best_psi = torch.where(pick, psi, best_psi)
                        best_ang = torch.minimum(ang_c, best_ang)
                wp_q[:] = best_q
                psi_star[:] = best_psi
                wp_p[:] = hp  # hold this spot while rotating: a free position sinks under gravity
            act = servo(wp_p, wp_q, CLOSE_W)
            upright = (up_axis_of(key.data.root_quat_w)[:, 2] > 0.995)
            if bool(upright.all()):
                phase, marker = "aim", i
            elif t_in >= 3 * WP_TIMEOUT:  # a flat key must never proceed toward the bolt
                print(f"  ABORT: erection stalled (key up_z {float(up_axis_of(key.data.root_quat_w)[:, 2].min()):+.2f})", flush=True)
                phase, marker = "retreat", i
        elif phase == "aim":  # settle at the erect-latched hex-aligned yaw — STILL at the off-axis
            # spot, so any residual rotation happens in clear air; every phase near the bolt then
            # only translates (psi_star was chosen by erect; re-picking here could demand another
            # large rotation)
            kq = quat_from_angle_axis(psi_star, ez)
            kp = torch.zeros(n, 3, device=dev)
            kp[:, 0] = hole_xy[:, 0]
            kp[:, 1] = hole_xy[:, 1] + 0.10
            kp[:, 2] = plat_z + CARRY_Z
            tp, tq = hand_for_key(kp, kq)
            act = servo(tp, tq, CLOSE_W)
            yaw_err = _wrap(yaw_of(key.data.root_quat_w) - psi_star).abs()
            still_up = up_axis_of(key.data.root_quat_w)[:, 2] > 0.99
            if bool((yaw_err < math.radians(2.5)).all()) and bool(still_up.all()):
                phase, marker = "clock", i
            elif t_in >= 2 * WP_TIMEOUT:
                print(f"  ABORT: aim stalled (yaw err {float(torch.rad2deg(yaw_err).max()):.1f}deg, "
                      f"up {float(up_axis_of(key.data.root_quat_w)[:, 2].min()):+.2f})", flush=True)
                phase, marker = "retreat", i
        elif phase == "clock":  # translate-only to the hover over the BORE axis, yaw held at the
            # latched psi_star — no rotation commands anywhere near the bolt
            kq = quat_from_angle_axis(psi_star, ez)
            kp = torch.zeros(n, 3, device=dev)
            kp[:, 0:2] = bolt.data.root_pos_w[:, 0:2]
            kp[:, 2] = bolt.data.root_pos_w[:, 2] + SOCKET_MOUTH_Z + INSERT_HOVER
            if t_in > 15:  # contact-free here: learn the pose bias for the insert phase
                pos_off[:] = (0.7 * pos_off + 0.3 * (kp - key.data.root_pos_w)).clamp(-0.02, 0.02)
            tp, tq = hand_for_key(kp + pos_off, kq)
            act = servo(tp, tq, CLOSE_W)
            yaw_err = _wrap(yaw_of(key.data.root_quat_w) - psi_star).abs()
            centred = ((key.data.root_pos_w[:, 0:2] - bolt.data.root_pos_w[:, 0:2]).norm(dim=-1) < 0.001) \
                & (yaw_err < math.radians(2.5))
            if not bool(bolt_ok(say=True).all()):
                print("  ABORT: bolt left its nest before insertion", flush=True)
                phase, marker = "retreat", i
            elif bool(centred.all()) or t_in >= 2 * WP_TIMEOUT:
                phase, marker = "insert", i
        elif phase == "insert":  # hover-peck probe: the hex clearance is 0.7 mm/side and a pressed
            # tip is stiction-locked on the rim (a sliding spiral cannot move it) — so probe a
            # lattice of lateral offsets, repositioning in FREE AIR between straight-down pecks;
            # the peck that lines up drops in, and a caught tip is driven on to the floor
            period = 12  # ctrl steps per peck (~0.8 s: hover-reposition, then press)
            if t_in % period == 1:  # re-latch the clocking each peck: the taps walk the bolt's yaw
                ky = yaw_of(key.data.root_quat_w)
                dpsi = _wrap((yaw_of(bolt.data.root_quat_w) - ky) % (math.pi / 3.0))
                dpsi = torch.where(dpsi > math.pi / 6.0, dpsi - math.pi / 3.0, dpsi)
                psi_star[:] = ky + dpsi
            kq = quat_from_angle_axis(psi_star, ez)
            probe = (t_in // period) % 19
            if probe == 0:
                ox, oy = 0.0, 0.0
            elif probe <= 6:
                ox = 0.00055 * math.cos((probe - 1) * math.pi / 3)
                oy = 0.00055 * math.sin((probe - 1) * math.pi / 3)
            else:
                ox = 0.00095 * math.cos((probe - 7) * math.pi / 6)
                oy = 0.00095 * math.sin((probe - 7) * math.pi / 6)
            mouth_z = bolt.data.root_pos_w[:, 2] + SOCKET_MOUTH_Z
            floor_z = bolt.data.root_pos_w[:, 2] + SOCKET_FLOOR_Z
            hover = (t_in % period) < 6
            z_cmd = mouth_z + (0.002 if hover else -0.0012)
            tip_in = key_tip_axial() < SOCKET_MOUTH_Z - 0.001  # caught: drive to the floor
            if t_in == 1:
                seat_ok.zero_()
            kp = torch.zeros(n, 3, device=dev)
            # probe around the BOLT's live axis — the socket rides the bolt, which sits/walks
            # within its thread play off the hole axis; a CAUGHT tip recentres (no probe offset)
            # so the wedged-in-tilted key straightens before the handoff lets go of it
            kp[:, 0] = bolt.data.root_pos_w[:, 0] + torch.where(tip_in, torch.zeros_like(depth()), torch.full_like(depth(), ox))
            kp[:, 1] = bolt.data.root_pos_w[:, 1] + torch.where(tip_in, torch.zeros_like(depth()), torch.full_like(depth(), oy))
            kp[:, 2] = torch.where(tip_in, floor_z, z_cmd.clamp(min=floor_z))
            if hover and (t_in % period) >= 4 and not bool(tip_in.any()):  # settled free air: learn
                pos_off[:] = (0.7 * pos_off + 0.3 * (kp - key.data.root_pos_w)).clamp(-0.02, 0.02)
            tp, tq = hand_for_key(kp + pos_off, kq)
            act = servo(tp, tq, CLOSE_W)
            if t_in % period in (5, period - 1):  # end of hover (free air) and end of press
                leg = "hover" if t_in % period == 5 else "press"
                dp = _wrap((yaw_of(bolt.data.root_quat_w) - yaw_of(key.data.root_quat_w)) % (math.pi / 3.0))
                dp = torch.where(dp > math.pi / 6.0, dp - math.pi / 3.0, dp)
                lat = (key.data.root_pos_w[:, 0:2] - bolt.data.root_pos_w[:, 0:2]).norm(dim=-1)
                over = key_tip_axial() - SOCKET_MOUTH_Z
                print(f"    [probe {probe:2d} {leg}] lat {float(lat.max()) * 1e3:4.2f}mm | dpsi "
                      f"{float(torch.rad2deg(dp.abs().max())):4.1f}deg | tip over mouth {float(over.min()) * 1e3:+5.2f}mm", flush=True)
            settled = (key_tip_axial() < SOCKET_FLOOR_Z + 0.002) & (key_tip_axial() > SOCKET_FLOOR_Z - 0.002) \
                & (key_lateral() < 0.0015) & (up_axis_of(key.data.root_quat_w)[:, 2] > 0.99) & bolt_ok()
            seat_ok[:] = torch.where(settled, seat_ok + 1, torch.zeros_like(seat_ok))
            if not bool(bolt_ok(say=True).all()):
                print("  ABORT: bolt knocked out of its nest during insertion", flush=True)
                phase, marker = "retreat", i
            elif bool((seat_ok >= 5).all()):  # flush on the floor, upright, centred — and STAYING so
                aim_tries = 0
                phase, marker = "handoff", i
            elif t_in >= INSERT_TIMEOUT:
                aim_tries += 1
                if aim_tries >= 5:  # a wedged configuration can re-aim forever — fail honestly
                    drops += 1
                    print("  DROP: insertion never landed after repeated re-aims", flush=True)
                    phase, marker = "retreat", i
                else:
                    print("  insert timed out, re-aiming", flush=True)
                    phase, marker = "aim", i
        elif phase == "handoff":  # bleed the stored press off through the weld, release, open, then
            # retreat DOWN-WIND of the handle. Two flip hazards at release: (1) the impedance still
            # carries the insertion press — cutting the weld pops the arm back with the fingers
            # friction-gripping the key; (2) rising open fingers sweep the horizontal handle's plane.
            if t_in == 1:
                ez_h = torch.zeros(n, 3, device=dev)
                ez_h[:, 2] = 1.0
                grip_pt[:] = quat_apply(hq, ez_h)  # latched approach axis: back out along -this
                wp_p[:] = hp
                wp_q[:] = hq
            if t_in == 1:
                grip_freeze[:] = art.data.joint_pos[:, fingers]
                wp_q[:] = quat_mul(quat_from_angle_axis(torch.full((n,), math.radians(3.0), device=dev), ez), hq)
            if t_in <= 8:  # relax + torque-release (see rewind_open), still welded and gripped
                act = servo(hp, wp_q, grip_freeze)
            else:
                if welded.any():
                    weld_off()
                if t_in > 14:  # back out along the approach axis (space the hand just traversed —
                    # the horizontal palm sweeps a wide azimuth if it rises in place), then rise
                    back = 0.0025 * min(t_in - 14, 28)
                    wp_p2 = wp_p - grip_pt * back
                    if t_in > 45:
                        wp_p2[:, 2] = plat_z + CARRY_Z
                    act = servo(wp_p2, wp_q, OPEN_W)
                else:
                    act = servo(wp_p, wp_q, OPEN_W)
            if t_in >= 60:  # after the bleed + back-out + rise have all played out
                in_socket = (key_tip_axial() < SOCKET_MOUTH_Z - 0.002) & (key_lateral() < 0.004) & bolt_ok()
                if not bool(in_socket.all()):
                    drops += 1
                    print("  DROP: key left the socket during the handoff", flush=True)
                    phase, marker = "retreat", i
                else:
                    if handoff_depth is None:  # keep the FIRST handoff as the gain baseline
                        handoff_depth = depth().clone()
                    pos_off.zero_()  # learned in the horizontal-hand configuration — wrong for the
                    # top-down regrasp; better to relearn from zero than aim with a stale bias
                    phase, marker = "regrasp_reset", i
        elif phase == "regrasp_reset":  # hand is empty: go HOME first. The handoff leaves the arm
            # horizontal in an unpredictable IK branch (sometimes wedged at a wrist stop, unable to
            # descend); from home, the top-down approach is exactly the proven pick geometry.
            act = servo(home_p, home_q, OPEN_W)
            if (bool(((home_p - hp).norm(dim=-1) < 0.05).all()) and bool((art.data.joint_pos[:, j7].abs() < 2.0).all())) \
                    or t_in >= 2 * WP_TIMEOUT:
                phase, marker = "regrasp_high", i
        elif phase == "regrasp_high":  # high staging waypoint first: the handoff back-out can leave
            # the arm in an IK branch whose lowest reachable z near the bolt is ~0.6 m (the OSC
            # cannot cross the singularity to descend) — passing through this comfortable pose
            # re-branches the arm before the vertical descent
            wp_p[:, 0:2] = key.data.root_pos_w[:, 0:2]
            wp_p[:, 2] = plat_z + 0.40
            wp_q[:] = q_down(hex_grip_yaw())
            act = servo(wp_p, wp_q, OPEN_W)
            hp_err = (wp_p - hp).norm(dim=-1)
            if bool((hp_err < 0.03).all()) or t_in >= WP_TIMEOUT:
                phase, marker = "regrasp_hover", i
        elif phase == "regrasp_hover":  # top-down above the upright arm, hex-flat aligned; the
            # top-down configuration has its own pose bias — learn it here (free air) so the
            # descend and close land on the arm, not beside it
            grip_yaw[:] = hex_grip_yaw()
            grip_pt[:] = key.data.root_pos_w + up_axis_of(key.data.root_quat_w) * STROKE_GRIP_UP
            wp_p[:] = grip_pt
            wp_p[:, 2] = grip_pt[:, 2] + hand_to_pad + REWIND_LIFT
            wp_q[:] = q_down(grip_yaw)
            if t_in > 10:  # settled free air: pad-centre target vs achieved pad centre
                pad_c = hp + quat_apply(hq, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)) * hand_to_pad
                want = grip_pt.clone()
                want[:, 2] += REWIND_LIFT
                pos_off[:] = (0.7 * pos_off + 0.3 * (want - pad_c)).clamp(-0.02, 0.02)
            act = servo(wp_p + pos_off, wp_q, OPEN_W)
            if (t_in >= 25 and bool(at(wp_p + pos_off, wp_q).all())) or t_in >= WP_TIMEOUT:
                phase, marker = "regrasp_down", i  # let the offset EMA converge before descending
        elif phase == "regrasp_down":
            wp_p[:, 2] = grip_pt[:, 2] + hand_to_pad
            act = servo(wp_p + pos_off, wp_q, OPEN_W)
            if bool(at(wp_p + pos_off, wp_q).all()) or t_in >= WP_TIMEOUT:
                phase, marker = "regrasp_close", i
        elif phase == "regrasp_close":
            act = servo(wp_p + pos_off, wp_q, CLOSE_W)
            if t_in >= CLOSE_STEPS:
                gap = art.data.joint_pos[:, fingers].sum(dim=-1)
                ok = (gap > CLOSED_MIN) & (gap < CLOSED_MAX)
                if bool(ok.all()):
                    weld_on()
                    regrasp_tries = 0
                    if not bool((key_tip_axial() < SOCKET_MOUTH_Z - 0.002).all()):
                        # grasped a key that is OUT of the socket (rewind-drop recovery):
                        # re-run the insertion pipeline with this top-down grip
                        print("  key in hand but out of the socket — re-inserting", flush=True)
                        phase, marker = "aim", i
                    else:
                        stroke_psi.zero_()
                        stroke_q0[:] = key.data.root_quat_w
                        cycles += 1
                        if cycles == 1:  # metrics start with the ratchet: forget pre-ratchet handling
                            bolt_turn.zero_()
                            key_turn.zero_()
                            prev_bolt_yaw = yaw_of(bolt.data.root_quat_w)
                            prev_key_yaw = yaw_of(key.data.root_quat_w)
                        phase, marker = "stroke", i
                else:
                    regrasp_tries += 1
                    if regrasp_tries >= 4 or not bool(((key_tip_axial() < SOCKET_MOUTH_Z - 0.002)
                                                       & (up_axis_of(key.data.root_quat_w)[:, 2] > 0.9)).all()):
                        drops += 1
                        print(f"  DROP: regrasp failed (gap {gap.tolist()}, key up "
                              f"{float(up_axis_of(key.data.root_quat_w)[:, 2].min()):+.2f})", flush=True)
                        phase, marker = "retreat", i
                    else:
                        print(f"  regrasp missed (gap {gap.tolist()}), retrying", flush=True)
                        phase, marker = "regrasp_high", i
        elif phase == "stroke":  # press + twist: the key drives the bolt down the threads
            stroke_psi += STROKE_W / 15.0
            stroke_psi.clamp_(max=STROKE_RAD)
            # pure-yaw target: sweeping the CAPTURED orientation would preserve (and accumulate)
            # the key's lean cycle over cycle — commanding upright rights it through the weld
            kq = quat_from_angle_axis(yaw_of(stroke_q0) - stroke_psi, ez)
            ub = up_axis_of(bolt.data.root_quat_w)
            kp = bolt.data.root_pos_w + ub * (SOCKET_FLOOR_Z - PRESS_DZ)
            tp, tq = hand_for_key(kp, kq)
            act = servo(tp, tq, CLOSE_W)
            done = depth() >= STOP_DEPTH
            swept = stroke_psi >= STROKE_RAD - 1e-6
            guard = art.data.joint_pos[:, j7].abs() > J7_GUARD
            if bool(done.all()):
                phase, marker = "retreat", i
            elif bool((swept | guard).all()) or t_in >= STROKE_TIMEOUT:
                phase, marker = "rewind_open", i
        elif phase == "rewind_open":  # bleed the stored press through the weld; open AT the live
            # pose (any position correction while the pads still touch can drag the key up out of
            # the socket); only then lift clear of the handle plane
            if t_in <= 6:
                if t_in == 1:
                    grip_freeze[:] = art.data.joint_pos[:, fingers]
                    wp_q[:] = quat_mul(quat_from_angle_axis(torch.full((n,), math.radians(3.0), device=dev), ez), hq)
                    wp_p[:] = hp
                # torque-release: back-rotate ~3 deg while welded — the stroke leaves the hex
                # corners JAMMED against the socket walls, and cutting the weld under that stored
                # torsion kicks the free key up and out
                act = servo(wp_p, wp_q, grip_freeze)
            elif t_in <= 14:
                if welded.any():
                    weld_off()
                act = servo(hp, hq, OPEN_W)
            else:
                if t_in == 15:  # latch: lift STRAIGHT UP from wherever the hand is — any lateral
                    # re-aim while the fingers pass the arm/handle can strike the free key
                    wp_p[:] = hp
                    wp_p[:, 2] = key.data.root_pos_w[:, 2] + STROKE_GRIP_UP + hand_to_pad + REWIND_LIFT
                act = servo(wp_p, wp_q, OPEN_W)
            ax = key_tip_axial()
            if bool((ax > SOCKET_MOUTH_Z - 0.001).any()) or bool((key_lateral() > 0.008).any()):
                drops += 1
                if bool((up_axis_of(key.data.root_quat_w)[:, 2] > 0.8).all()):
                    print("  DROP: key left the socket while unheld — still upright, re-grasping to re-insert", flush=True)
                    phase, marker = "regrasp_high", i
                else:
                    print("  DROP: key left the socket while unheld", flush=True)
                    phase, marker = "retreat", i
            elif bool(at(wp_p, wp_q).all()) or t_in >= WP_TIMEOUT // 2:
                if cycles >= MAX_CYCLES:
                    print("  cycle budget exhausted", flush=True)
                    phase, marker = "retreat", i
                else:
                    phase, marker = "regrasp_high", i
        elif phase == "retreat":
            if welded.any():
                weld_off()
            wp_p[:] = hp
            wp_p[:, 2] = plat_z + 0.30
            act = servo(wp_p, hq, OPEN_W)
            if t_in >= WP_TIMEOUT // 2:
                phase, marker = "settle", i
        else:  # settle
            act = servo(home_p, home_q, OPEN_W)
            if t_in >= SETTLE_STEPS:
                break

        prev_depth = depth().clone()
        step(act)
        dd = (depth() - prev_depth).abs()
        if phase in ("pick_hover", "pick_down", "pick_close", "lift", "carry", "erect", "aim", "clock", "insert") and float(dd.max()) > 0.0015:
            kt = key.data.root_pos_w
            print(f"    [watchdog] ctrl {i} {phase}: bolt depth jumped {float(dd.max()) * 1e3:+.1f}mm | "
                  f"key ({kt[0, 0]:.3f},{kt[0, 1]:.3f},{kt[0, 2]:.3f}) | hand z {hand_pose()[0][0, 2]:.3f}", flush=True)
        if phase not in ("show", "stage"):
            cur = yaw_of(bolt.data.root_quat_w)
            bolt_turn = bolt_turn - _wrap(cur - prev_bolt_yaw)
            prev_bolt_yaw = cur
            kcur = yaw_of(key.data.root_quat_w)
            key_turn = key_turn - _wrap(kcur - prev_key_yaw)
            prev_key_yaw = kcur

        if i % LOG_EVERY == 0:
            d = depth() * 1e3
            slip = torch.rad2deg(key_turn - bolt_turn)
            q7 = torch.rad2deg(art.data.joint_pos[:, j7])
            gap = art.data.joint_pos[:, fingers].sum(dim=-1) * 1e3
            wd = float(weld_drift().max() * 1e3) if bool(welded.any()) else float("nan")
            kup = float(up_axis_of(key.data.root_quat_w)[:, 2].mean())
            print(f"  ctrl {i:5d} [{phase:13s}] | cyc {cycles:2d} | tip depth {d.mean():+6.2f}mm | bolt "
                  f"{torch.rad2deg(bolt_turn).mean():+7.0f}deg | slip {slip.mean():+6.1f}deg | "
                  f"q7 {q7.mean():+6.0f}deg | key z {key.data.root_pos_w[:, 2].mean():.3f} "
                  f"up {kup:+.2f} | hand z {hp[:, 2].mean():.3f} | gap {gap.mean():4.1f}mm | weld {wd:4.1f}mm", flush=True)
            if not torch.isfinite(bolt.data.root_pos_w).all() or float((bolt.data.root_pos_w[:, 0:2] - hole_xy).norm(dim=-1).max()) > 0.05:
                print("  ABORT: bolt left the hole region (ejected or blew up)", flush=True)
                break

    if writer is not None:
        writer.close()
        print("MP4:", args.video, flush=True)

    seated = sc.seated()
    d = depth() * 1e3
    gain = (d - handoff_depth * 1e3) if handoff_depth is not None else d
    revs = bolt_turn / (2 * math.pi)
    turned = revs > 0.5
    mm_per_rev = float((gain[turned] / revs[turned]).median()) if turned.any() else float("nan")
    slip = torch.rad2deg(key_turn - bolt_turn)
    print(f"ALLEN-KEY-FRANKA | seated {int(seated.all(dim=1).sum())}/{n} | drove {float(gain.mean()):+.1f}mm "
          f"over {float(revs.mean()):.1f} revs = {mm_per_rev:.2f} mm/rev (pitch {PITCH_MM:.1f}) | "
          f"slip {float(slip.mean()):+.1f}deg | {cycles} cycles, {picks} picks, {drops} drops | tip depth mm: "
          f"min={d.min():+.1f} mean={d.mean():+.1f} max={d.max():+.1f} (seat>= {sc.cfg.seat_depth * 1e3:.0f})", flush=True)
    close_and_exit(env, app)


if __name__ == "__main__":
    main()

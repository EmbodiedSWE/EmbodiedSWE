"""Franka smoke for PcMotherboardAssemblyScene — the arm picks the long allen key off the table,
stands it in the first staged bolt's hex socket, and fastens ALL the motherboard's case-mount
bolts with it, ratcheting each one down and carrying the key hole-to-hole without ever picking
it up again.

The threading is the scene smoke's kinematic screw-joint mechanic (cf. pc_motherboard_smoke):
each bolt is made kinematic and follows the key's MEASURED rotation through the hex lash —
one-way, like a frictional thread — descending on the 1 mm-pitch helix to a hard stop; the
thread inserts' collision is disabled (the screw joint IS the thread). The key<->socket hex
contact is LIVE: the key must genuinely stand in each socket, stay clocked, and turn — a missed
insertion or a popped key stalls the joint honestly. Bolts stage pre-engaged (the run is the
final tightening of an already-started board).

Grasping follows the benchmark weld-on-closure contract (cf. the pc_* franka smokes): a
normally-disabled FixedJoint hand<->key is enabled when the gripper is verifiably closed around
one of the key's hex arms (closure verified geometrically) and released when it opens.

The long-series key IS the tool for this case: its 210 mm working arm inserts, so the 120 mm
handle cranks just ABOVE the case's 195 mm walls and every grip rides high — the one
horizontal-hand insertion grips the arm 185 mm above the tip (the wrist skims the wall top; the
case walls are visual-only, but the choreography respects them), and the ratchet re-grasps the
HANDLE top-down, orbiting a 22 mm crank circle while the wrist roll winds 1:1 with the screw.
The choreography works around the same two constraints as the allen-key task:
  * No single grasp of the lying key yields a top-down screwing grip, so the arm picks the lying
    key by its working arm, erects it about the HANDLE axis, lowers the tip into the FIRST
    socket with the horizontal hand, releases (slow-open, onto the key's stable lean), and
    re-grasps the handle top-down — once. Every later hole is a top-down carry: lift the key
    out, glide it over the standing heads, drop it into the next socket.
  * The wrist cannot turn the ~6 revolutions a hole needs, so each hole ratchets WITHOUT
    letting go: press + twist to the wrist stop, unload, lift 4 mm out of the hex, rewind by a
    multiple of 60 deg (clocking preserved exactly), drop back in, repeat.

Phases: show -> stage -> pick(hover/down/close) -> lift -> erect -> carry(raise/glide/drop) ->
insert(descend, peck-retry) -> handoff -> regrasp(hover/down/close) ->
[[stroke -> unload]* -> extract -> travel(glide + rewind) -> reinsert]* per hole ->
release -> retreat -> settle.
Verdict: per-hole seated flags and depth gained vs the joint's revolutions (1.0 mm/rev by
construction), within-stroke key->bolt slip, cycles, picks, drops.

.venv/bin/python -m robobench.suites.assembly.smokes.pc_motherboard_franka --headless
python -m robobench.suites.assembly.smokes.pc_motherboard_franka --headless --holes 2
python -m robobench.suites.assembly.smokes.pc_motherboard_franka --livestream 2
python -m robobench.suites.assembly.smokes.pc_motherboard_franka \
    --headless --enable_cameras --video robobench/suites/assembly/videos/pc_motherboard_franka.mp4 --cap 2
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--holes", type=int, default=0, help="fasten only the first N holes of the drive order (0 = all)")
parser.add_argument("--video", type=str, default="", help="save an mp4 here (needs --headless --enable_cameras)")
parser.add_argument("--cap", type=int, default=2, help="with --video: capture one frame every N control steps")
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
    from robobench.suites.assembly.scenes import PcMotherboardAssemblyScene

DT = 1.0 / 240.0  # sim timestep (matches the registered env's dt override)

# Key geometry baked into the committed long-key USD (informs every grip constant below): tip at
# the body origin, 210 mm working arm up local +z, 120 mm handle along local +x off the elbow at
# z = 210 mm; both arms hex 6.2 mm across flats / 7.2 mm across corners, corners at k*60 deg.
ARM_LEN = 0.210
HANDLE_LO_Z = 0.2064       # handle underside above the tip (the graspable arm band ends here)
PICK_GRIP_D = 0.185        # pick grip: this far up the arm from the tip. HIGH on purpose: the
                           # horizontal-hand insertion then rides at the case wall's TOP instead
                           # of inside the box; the pads stay under the handle root.
CRANK_GRIP_LOCAL = (0.022, 0.0, 0.210)  # screw grip: on the handle, 22 mm out from the shaft
# ----- bolt-local geometry (origin = thread tip, +z up through the head) -------------------------
SOCKET_FLOOR_Z = 0.01775   # hex recess floor
SOCKET_MOUTH_Z = 0.0214    # head top = recess mouth (3.6 mm deep socket, ~0.4 mm/side clearance)
THREAD_LEN = 0.0124        # head bottom above the tip: depth at which the head bottoms out
# ----- the kinematic screw joint (the scene smoke's thread mechanic) ------------------------------
PITCH = 0.001              # helix pitch (m per revolution)
PITCH_MM = PITCH * 1e3
STAGE_DEPTH = 0.006        # bolts stage pre-engaged at this tip depth below the board face
STAGE_YAW = math.pi        # bolt yaw at stage (a k*60 deg hex clocking)
SEAT_MARGIN = 0.0001       # hard stop: the head held this far above the board (never preloads it)
STOP_DEPTH = 0.0118        # stop cranking at this tip depth — just before the head bottoms at 12.4
LASH_HALF = math.radians(8.0)  # key rotation before the hex flats engage (per re-entry)
# ----- case geometry (case origin = board-face centre, z = 0 ON the face) ------------------------
WALL_TOP_Z = 0.195         # the case's perimeter walls end here above the board (visual-only
                           # physics, but the choreography stays out of them)
TRAVEL_TIP_Z = 0.032       # key-tip height above the board for in-case hole-to-hole travel
                           # (clears the standing 21.4 mm heads)
CROSS_TIP_Z = 0.215        # key-tip height for crossing the case wall from outside

# ----- Franka OSC action semantics (6 EE pose deltas + 2 finger targets at 15 Hz) -----------------
POS_SCALE, ROT_SCALE = 0.02, 0.097
ROT_SAT = 8.0              # rotation lead cap (units): gravity droop stalls a 1-unit lead
OPEN_W = 0.04
FINGER_TO_PAD = 0.045      # panda_finger body origin -> finger-pad centre, along the approach
                           # (the finger TIP ends 8.8 mm past the pad centre)

# ----- grips --------------------------------------------------------------------------------------
# Pick: top-down pinch of the LYING working arm (the lying hex presents its corners sideways, so
# the pads land across corners, 7.2 mm). The arm is only 6.2 mm tall on the table, so the pads
# grip by their LOWER band: pad centre 8 mm above the arm axis keeps the fingertips 2 mm off the
# table while the 18 mm pad still covers the corner ridge.
PICK_PAD_LIFT = 0.008
PICK_W = 0.0033            # per-finger closed width: across-corners half-width minus a kiss
# Screw grip: top-down pinch of the horizontal HANDLE (the crank), 22 mm out from the shaft —
# pads across its FLATS, the inner finger clear of the shaft, nothing above the grip.
SCREW_W = 0.0027           # per-finger closed width: across-flats half-width minus a kiss
STRADDLE_W = 0.012         # per-finger width while descending AROUND an arm
CLOSED_MIN, CLOSED_MAX = 0.004, 0.010  # closure window (finger-joint sum, m): hex 6.2-7.2 mm

# ----- choreography -------------------------------------------------------------------------------
HOVER_CLEAR = 0.05         # pad hover height above the grip point before a descent
ERECT_TIP_Z = 0.05         # key-tip height (above the table) while erecting: the body sweeps UP
                           # from flat around the held tip, south of the case wall
INSERT_HOVER = 0.008       # tip hover above the socket mouth before a descent
PRESS_LEAD = 0.003         # stroke press: command the tip this far below the live socket floor
                           # (the press is what keeps the hex from camming out under torque)
J7_GUARD = 2.6             # end a stroke when joint 7 exceeds this (limit 2.897 rad)
J7_START = -2.1            # rewind aims joint 7 back here (leaves ~270 deg of stroke)
STROKE_W = 1.0             # commanded stroke spin rate (rad/s), under the 1.455 rad/s action cap
REWIND_W = 1.4             # rewind spin rate (rad/s), free air
SLIP_ABORT = math.radians(45.0)  # end a stroke early if the key slips this far over the hex
MAX_CYCLES_HOLE = 14       # ratchet cycle budget per hole (a clean hole needs ~9)
PICK_RETRIES = 3
INSERT_RETRIES = 8         # peck re-tries per insertion (rise, re-trim, drop again)
DROP_BUDGET = 3
# Drive order over the scene's authored holes: mid_right FIRST (the wall-freest spot for the one
# horizontal-hand insertion), then a shortest-walk serpentine over the rest.
DRIVE_ORDER = (2, 1, 0, 3, 4, 5, 6)

# Waypoint tolerances and per-phase step budgets, in CONTROL steps (15 Hz -> 16 substeps each).
# Every long move GLIDES its commanded target (smoothstep); budgets are deliberately UNHURRIED —
# a rushed glide saturates the norm-clamped servo, the gravity-uncompensated arm swings wide, and
# an insertion that arrives with residual error pecks where a settled one drops straight in.
TOL_P, TOL_R = 0.004, 0.06
SHOW_END, STAGE_SETTLE = 20, 15
WP_TIMEOUT, CLOSE_STEPS, SETTLE_STEPS = 100, 24, 45
LIFT_STEPS, ERECT_STEPS, RAISE_STEPS, GLIDE_STEPS, DROP_STEPS, INSERT_STEPS, RETREAT_STEPS = 80, 130, 70, 110, 90, 70, 70
STROKE_TIMEOUT, TRAVEL_TIMEOUT, REINSERT_TIMEOUT = 110, 80, 75
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

    env = ENVS.get("assembly.pc_motherboard.franka.osc")().build(num_envs=args.num_envs, device=device)
    sc: PcMotherboardAssemblyScene = env.scene  # type: ignore[assignment]
    n = env.num_envs
    dev = device
    ids = torch.arange(n, device=dev)
    bolts, key, case = sc.bolts, sc.key, sc.case
    B = len(DRIVE_ORDER) if args.holes <= 0 else min(args.holes, len(DRIVE_ORDER))
    order = DRIVE_ORDER[:B]
    art = env.robot.articulation
    hand_idx = art.body_names.index("panda_hand")
    j7 = art.joint_names.index("panda_joint7")
    fingers = art.find_joints(["panda_finger_joint.*"])[0]
    render = (not args.headless) or livestream_on
    ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
    ex = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3)
    crank_grip_local = torch.tensor(CRANK_GRIP_LOCAL, device=dev).expand(n, 3)
    HARD_CAP = 3000 + 3200 * B  # per-phase timeouts backstop; this only guards a livelock

    # ----- USD edits before the physics re-parse ---------------------------------------------------
    # (1) hand<->key welds (the grasp contract): pre-authored, disabled FixedJoints. PhysX latches
    # a joint's local frames when it is FIRST enabled, so each grasp consumes a fresh joint from
    # the pool. (2) The scene smoke's screw-joint mechanic: every bolt kinematic (a scripted screw
    # DOF the dynamic key pushes against) and the thread inserts' collision off — the joint IS the
    # thread; the bolt's SOCKET walls stay live for the key.
    from pxr import Gf, UsdPhysics

    stage = env.stage
    WELD_POOL = 16  # pick + regrasp + retries' slack (the ratchet never releases)
    weld_paths: list[list[str]] = []
    for e in range(n):
        base = f"/World/envs/env_{e}"
        assert stage.GetPrimAtPath(f"{base}/Key/allen_key").HasAPI(UsdPhysics.RigidBodyAPI)
        row = []
        for k_ in range(WELD_POOL):
            j = UsdPhysics.FixedJoint.Define(stage, f"{base}/hand_key_weld_{k_}")
            j.CreateBody0Rel().SetTargets([f"{base}/Robot/panda_hand"])
            j.CreateBody1Rel().SetTargets([f"{base}/Key/allen_key"])
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateJointEnabledAttr(False)
            j.CreateExcludeFromArticulationAttr(True)  # a maximal-coordinate weld, not an arm DOF
            row.append(f"{base}/hand_key_weld_{k_}")
        weld_paths.append(row)
        for i in range(sc.cfg.num_holes):
            prim = stage.GetPrimAtPath(f"{base}/Bolt_{i}/allen_bolt")
            assert prim.IsValid(), f"missing bolt body prim: bolt {i}, env {e}"
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
            prim = stage.GetPrimAtPath(f"{base}/Case/case/hole_{i}/thread_insert")
            assert prim.IsValid(), f"thread_insert prim missing: hole {i}, env {e}"
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)

    cam = writer = None
    if args.video:
        import imageio.v2 as imageio
        from isaaclab.sensors import Camera, CameraCfg
        cam = Camera(CameraCfg(prim_path="/World/cam", update_period=0.0, height=720, width=1280, data_types=["rgb"],
                               spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.01, 100.0))))
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args.video, fps=30)

    env.sim.reset()  # re-parse physics so the USD edits (and camera) are picked up
    env.reset()

    # Open-env OSC retune (the pc_* franka smokes' values): the controller carries no gravity
    # compensation, so the stock gains leave a configuration-dependent pose sag; the stiffer gains
    # bring it into the learned bias' budget, slightly overdamped so transits do not ring.
    osc = env.robot.controller.controllers[0]
    osc._kp[0:3] = 400.0
    osc._kp[3:6] = 450.0
    osc._kd = 2.2 * osc._kp.sqrt()

    case_pos = case.data.root_pos_w.clone()  # (n, 3): origin ON the board face, at its centre
    board_z = case_pos[:, 2].clone()
    holes_w = case_pos[:, None, :2] + torch.tensor(sc.cfg.hole_xy, device=dev)[None]  # (n, 7, 2)
    table_z = board_z - sc.cfg.case_lift

    # ----- camera: a two-anchor shot blended by the key's trip toward the case. The insert
    # anchor hangs HIGH and DUE SOUTH, looking steeply down into the open box: the case's tall
    # rear section hides the east holes from any east viewpoint (cf. the scene smoke's camera
    # note) and the west arm body looms from the west — from south-high every hole, the key,
    # and the crank stay in sight, with the hand the nearest arm segment to the camera.
    cam_pose = None
    if cam is not None:
        p0 = case_pos[0]
        k0 = key.data.root_pos_w[0]
        pick_eye = torch.tensor([float(k0[0]) + 0.36, float(k0[1]) - 0.34, float(p0[2]) + 0.40], device=dev)
        pick_tgt = torch.tensor([float(k0[0]), float(k0[1]) + 0.02, float(p0[2]) + 0.06], device=dev)
        ins_eye = torch.tensor([float(p0[0]) + 0.02, float(p0[1]) - 0.54, float(p0[2]) + 0.82], device=dev)
        ins_tgt = torch.tensor([float(p0[0]), float(p0[1]) + 0.02, float(p0[2]) + 0.04], device=dev)
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
        """Hand waypoint that puts the WELDED key at pose (kp, kq), via the captured grasp frame.
        The key's origin IS its inserting tip."""
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

    # ----- the kinematic screw joints --------------------------------------------------------------
    z0 = board_z - STAGE_DEPTH                          # bolt z at stage (n,)
    turn_max = (THREAD_LEN - SEAT_MARGIN - STAGE_DEPTH) * 2 * math.pi / PITCH
    bolt_turn = torch.zeros(n, sc.cfg.num_holes, device=dev)  # per-hole screw-in rotation (rad)
    coupled_turn = torch.zeros(n, device=dev)  # key rotation fed to the ACTIVE joint: re-latched
    # at each stroke (the hex lash re-charges every re-entry) and accumulated only while the key
    # is genuinely hex-engaged

    def stage_parts() -> None:
        """Teleport every bolt pre-engaged in its hole (upright, tip STAGE_DEPTH below the board
        face, a k*60 clocking). The key stays where it lies — the robot fetches it."""
        for b in range(sc.cfg.num_holes):
            st = torch.zeros(n, 13, device=dev)
            st[:, 0:2] = holes_w[:, b]
            st[:, 2] = z0
            st[:, 3] = math.cos(STAGE_YAW / 2)
            st[:, 6] = math.sin(STAGE_YAW / 2)
            bolts[b].write_root_state_to_sim(st, ids)

    def project_bolts(active_b: int | None) -> None:
        """Advance the ACTIVE hole's screw joint — it follows the key's engaged rotation through
        the hex lash, one-way, and descends on the helix to a hard stop — then write every bolt's
        kinematic pose (parked bolts hold theirs)."""
        if active_b is not None:
            follow = torch.clamp(coupled_turn - LASH_HALF, max=turn_max)
            bolt_turn[:, active_b] = torch.maximum(bolt_turn[:, active_b], follow)
        for b in range(sc.cfg.num_holes):
            yaw = STAGE_YAW - bolt_turn[:, b]
            st = torch.zeros(n, 7, device=dev)
            st[:, 0:2] = holes_w[:, b]
            st[:, 2] = z0 - PITCH * bolt_turn[:, b] / (2 * math.pi)
            st[:, 3] = torch.cos(yaw / 2)
            st[:, 6] = torch.sin(yaw / 2)
            bolts[b].write_root_pose_to_sim(st, ids)

    # ----- task frames -----------------------------------------------------------------------------
    def depth(b: int) -> torch.Tensor:  # bolt b's tip depth below the board face (m), per env
        return board_z - bolts[b].data.root_pos_w[:, 2]

    def tip_axial(b: int) -> torch.Tensor:
        """Key-tip height above bolt b's origin (m): SOCKET_MOUTH_Z at the recess mouth,
        SOCKET_FLOOR_Z seated on the floor."""
        return key.data.root_pos_w[:, 2] - bolts[b].data.root_pos_w[:, 2]

    def tip_lateral(b: int) -> torch.Tensor:
        return (key.data.root_pos_w[:, 0:2] - holes_w[:, b]).norm(dim=-1)

    def clock_err(b: int) -> torch.Tensor:
        """Key hex spin error to hole b's socket sector, wrapped to [-30, 30) deg."""
        dp = (yaw_of(bolts[b].data.root_quat_w) - yaw_of(key.data.root_quat_w)) % (math.pi / 3)
        return torch.where(dp > math.pi / 6, dp - math.pi / 3, dp)

    def handle_heading() -> torch.Tensor:
        hv = quat_apply(key.data.root_quat_w, ex)
        return torch.atan2(hv[:, 1], hv[:, 0])

    def shaft_up() -> torch.Tensor:
        """z-component of the working arm's up direction: 1 = standing tip-down."""
        return up_axis_of(key.data.root_quat_w)[:, 2]

    def in_socket(b: int, margin: float = 0.0012) -> torch.Tensor:
        return (tip_axial(b) < SOCKET_MOUTH_Z - margin) & (tip_lateral(b) < 0.0025)

    def flip_cmd(psi: torch.Tensor) -> torch.Tensor:
        """Commanded key orientation: standing at spin `psi`, pre-rotated by the learned tilt
        bias so the ACHIEVED arm stands vertical (psi is the caller's command, never biased)."""
        ang = tilt_bias.norm(dim=-1).clamp_min(1e-9)
        axis = torch.cat([tilt_bias / ang.unsqueeze(-1), torch.zeros(n, 1, device=dev)], dim=-1)
        return quat_mul(quat_from_angle_axis(ang, axis), quat_from_angle_axis(psi, ez))

    def learn_tilt(gain: float = 0.05) -> None:
        """Integrate the arm's residual LEAN (free air only): the command already carries the
        bias, so the residual drives it until the achieved arm is vertical."""
        r = torch.cross(up_axis_of(key.data.root_quat_w), ez, dim=-1)[:, 0:2]
        tilt_bias[:] = (tilt_bias + gain * r).clamp(-0.15, 0.15)

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
    key_turn = torch.zeros(n, device=dev)  # cumulative key screw-in rotation (rad, +ve = down)
    prev_key_yaw = yaw_of(key.data.root_quat_w)
    seq = 0                     # index into `order`: the hole being fastened
    active = order[0]           # the scene hole index under the key
    cycles_hole, picks, drops, insert_pecks = 0, 1, 0, 0
    advance = False
    hole_cycles = [0] * sc.cfg.num_holes
    pos_off = torch.zeros(n, 3, device=dev)   # INTEGRATED bias (desired - achieved, free air): the
    # OSC has no gravity compensation, so its realized pose sags configuration-dependently by
    # several mm — commands near contact add this learned offset so the achieved pose lands true.
    # Every learning gain stays <= 0.12/step: the servo answers an offset ~5 ticks late, and a
    # hotter integrator limit-cycles against that lag.
    tilt_bias = torch.zeros(n, 2, device=dev)  # the same for the arm's LEAN (axis-angle xy)
    grip_freeze = torch.zeros(n, 2, device=dev)  # finger targets latched at release: freezing the
    # PD at the MEASURED positions decays the squeeze before the pads separate
    close_ok = torch.zeros(n, device=dev)     # consecutive ticks the closure has verified
    seat_ok = torch.zeros(n, device=dev)      # consecutive ticks a settle/seat gate has held
    grip_pt = torch.zeros(n, 3, device=dev)
    grip_yaw = torch.zeros(n, device=dev)
    wp_p = torch.zeros(n, 3, device=dev)
    wp_q = torch.zeros(n, 4, device=dev)
    glide_from_p = torch.zeros(n, 3, device=dev)
    glide_from_z = torch.zeros(n, device=dev)
    glide_yaw0 = torch.zeros(n, device=dev)
    glide_q0 = torch.zeros(n, 4, device=dev)
    glide_aa = torch.zeros(n, 3, device=dev)
    psi_cmd = torch.zeros(n, device=dev)      # commanded key spin through insert/stroke/travel
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

    def hover_tip_z(b: int) -> torch.Tensor:
        return bolts[b].data.root_pos_w[:, 2] + SOCKET_MOUTH_Z + INSERT_HOVER

    def start_insert_glide(b: int) -> None:
        """Latch the descent glide state (shared by insert and reinsert)."""
        glide_from_z[:] = key.data.root_pos_w[:, 2] - bolts[b].data.root_pos_w[:, 2]
        psi_cmd[:] = yaw_of(key.data.root_quat_w) + clock_err(b)

    phase, marker = "show", 0
    i = 0
    while True:
        i += 1
        t_in = i - marker
        phase_at_act = phase  # the phase whose block computes THIS step's action: the post-step
        # coupling must key off it, not off a phase reassigned mid-iteration — the transition
        # step into `stroke` otherwise feeds the PREVIOUS hole's accumulated turn into the fresh
        # joint before the stroke's own latch runs
        hp, hq = hand_pose()
        if home_p is None:
            home_p, home_q = hp.clone(), hq.clone()
        act = servo(home_p, home_q, OPEN_W)  # default: hold home, fingers open

        if phase == "show":
            if t_in >= SHOW_END:
                stage_parts()
                phase, marker = "stage_hold", i
        elif phase == "stage_hold":  # let the writes settle one beat, then report
            if t_in >= STAGE_SETTLE:
                print(f"  staged: {sc.cfg.num_holes} bolts pre-engaged at "
                      f"{STAGE_DEPTH * 1e3:.0f} mm (kinematic screw joints), fastening {B} "
                      f"in order {list(order)}", flush=True)
                phase, marker = "pick_hover", i
        elif phase == "pick_hover":  # glide to a top-down hover over the lying key's working arm
            if t_in == 1:
                pos_off.zero_()
                glide_from_p[:] = hp
                arm = up_axis_of(key.data.root_quat_w)  # lying: the arm's horizontal direction
                grip_yaw[:] = nearest_parity(torch.atan2(arm[:, 1], arm[:, 0]))  # hand x along
                # the arm -> the fingers close across it
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
                pos_off[:] = (pos_off + 0.12 * (want - pad_centre())).clamp(-0.12, 0.12)
            goal = wp_p + pos_off
            act = servo(glide_from_p + (goal - glide_from_p) * s, wp_q, STRADDLE_W)
            pad_err = (pad_centre()[:, 0:2] - grip_pt[:, 0:2]).norm(dim=-1)
            if (t_in >= WP_TIMEOUT + 10 and bool((pad_err < 0.004).all()) and bool(at(goal, wp_q).all())) \
                    or t_in >= 3 * WP_TIMEOUT:
                phase, marker = "pick_down", i
        elif phase == "pick_down":  # descend AROUND the arm: open fingers pass it on both sides
            s = smoothstep(t_in / 60.0)
            wp_p[:] = grip_pt
            wp_p[:, 2] = grip_pt[:, 2] + hand_to_pad + HOVER_CLEAR * (1.0 - s)
            if s >= 1.0:
                want = grip_pt.clone()
                pos_off[:] = (pos_off + 0.12 * (want - pad_centre())).clamp(-0.12, 0.12)
            act = servo(wp_p + pos_off, wp_q, STRADDLE_W)
            pad_on = (pad_centre() - grip_pt).norm(dim=-1) < 0.003
            if (t_in >= 68 and bool(pad_on.all())) or t_in >= 2 * WP_TIMEOUT:
                close_ok.zero_()
                phase, marker = "pick_close", i
        elif phase == "pick_close":  # close to the arm's width; verify geometrically, then weld
            s = smoothstep(t_in / CLOSE_STEPS)
            width = STRADDLE_W + (PICK_W - STRADDLE_W) * s
            act = servo(wp_p + pos_off, wp_q, width)
            gap = art.data.joint_pos[:, fingers].sum(dim=-1)
            arm = up_axis_of(key.data.root_quat_w)  # closure verified against the LIVE key pose
            grip_pt[:] = key.data.root_pos_w + arm * PICK_GRIP_D
            grip_pt[:, 2] = key.data.root_pos_w[:, 2] + PICK_PAD_LIFT
            near = (pad_centre() - grip_pt).norm(dim=-1) < 0.004
            ok = near & (gap > CLOSED_MIN) & (gap < CLOSED_MAX)
            close_ok[:] = torch.where(ok, close_ok + 1, torch.zeros_like(close_ok))
            if t_in >= CLOSE_STEPS + 6 and bool((close_ok >= 4).all()):
                weld_on()
                print(f"  picked the key: pads across the lying arm, finger gap "
                      f"{[f'{float(g) * 1e3:.1f}' for g in gap]} mm (hex 6.2/7.2)", flush=True)
                phase, marker = "lift", i
            elif t_in >= CLOSE_STEPS + 30:
                if picks < PICK_RETRIES:
                    print(f"  pick missed (gap {[f'{float(g) * 1e3:.1f}' for g in gap]} mm), retrying", flush=True)
                    picks += 1
                    phase, marker = "pick_hover", i
                else:
                    print("  ABORT: pick failed", flush=True)
                    phase, marker = "retreat", i
        elif phase == "lift":  # straight up to the erect height, holding the as-picked orientation
            if t_in == 1:
                glide_from_p[:] = key.data.root_pos_w
                q0 = key.data.root_quat_w
                glide_q0[:] = torch.where(q0[:, :1] >= 0, q0, -q0)
            s = smoothstep(t_in / LIFT_STEPS)
            kp = glide_from_p.clone()
            kp[:, 2] = glide_from_p[:, 2] + s * (table_z + ERECT_TIP_Z - glide_from_p[:, 2])
            tp, tq = hand_for_key(kp, glide_q0)
            act = servo(tp, tq, PICK_W)
            if (t_in >= LIFT_STEPS and bool(((table_z + ERECT_TIP_Z - key.data.root_pos_w[:, 2]).abs() < 0.015).all())) \
                    or t_in >= LIFT_STEPS + 40:
                phase, marker = "erect", i
        elif phase == "erect":  # pitch the welded key tip-down ABOUT THE HANDLE AXIS: the whole
            # body sweeps UP around the held tip, south of the case wall; the target spin is the
            # first socket's hex sector nearest the current one, so the erection IS the clocking.
            # The rotation glides — a step-jump goal would swing the arm wide.
            if t_in == 1:
                h0 = handle_heading()
                dp = (yaw_of(bolts[active].data.root_quat_w) - h0) % (math.pi / 3)
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
            tp, tq = hand_for_key(glide_from_p, q_cmd)  # tip holds; only the quat sweeps
            act = servo(tp, tq, PICK_W)
            upright = shaft_up() > 0.995
            if t_in >= ERECT_STEPS + 10 and bool(upright.all()):
                phase, marker = "carry_raise", i
            elif t_in >= ERECT_STEPS + 2 * WP_TIMEOUT:
                print(f"  ABORT: erection stalled (arm up_z {float(shaft_up().min()):+.2f})", flush=True)
                phase, marker = "retreat", i
        elif phase == "carry_raise":  # tip straight up to the wall-crossing height
            if t_in == 1:
                glide_from_p[:] = key.data.root_pos_w
            s = smoothstep(t_in / RAISE_STEPS)
            kp = glide_from_p.clone()
            kp[:, 2] = glide_from_p[:, 2] + s * (board_z + CROSS_TIP_Z - glide_from_p[:, 2])
            tp, tq = hand_for_key(kp, flip_cmd(psi_cmd))
            act = servo(tp, tq, PICK_W)
            if (t_in >= RAISE_STEPS and bool(((board_z + CROSS_TIP_Z - key.data.root_pos_w[:, 2]).abs() < 0.02).all())) \
                    or t_in >= RAISE_STEPS + 40:
                phase, marker = "carry_glide", i
        elif phase == "carry_glide":  # translate over the case wall to above the first hole; the
            # key hangs tip-down ABOVE the wall top the whole way
            if t_in == 1:
                glide_from_p[:] = key.data.root_pos_w
                pos_off.zero_()
            goal = torch.zeros(n, 3, device=dev)
            goal[:, 0:2] = holes_w[:, active]
            goal[:, 2] = board_z + CROSS_TIP_Z
            s = smoothstep(t_in / GLIDE_STEPS)
            kp = glide_from_p + (goal - glide_from_p) * s
            if s >= 1.0:
                pos_off[:] = (pos_off + 0.1 * (goal - key.data.root_pos_w)).clamp(-0.12, 0.12)
                learn_tilt()
            psi_cmd[:] = yaw_of(key.data.root_quat_w) + clock_err(active)
            tp, tq = hand_for_key(kp + pos_off, flip_cmd(psi_cmd))
            act = servo(tp, tq, PICK_W)
            arrived = (goal[:, 0:2] - key.data.root_pos_w[:, 0:2]).norm(dim=-1) < 0.003
            if (t_in >= GLIDE_STEPS + 10 and bool(arrived.all())) or t_in >= GLIDE_STEPS + 2 * WP_TIMEOUT:
                phase, marker = "carry_drop", i
        elif phase == "carry_drop":  # descend inside the case to the insert hover, settling to a
            # sub-clearance start: xy within the 0.4 mm/side play, clocked, and STAYING so
            if t_in == 1:
                glide_from_z[:] = key.data.root_pos_w[:, 2]
                seat_ok.zero_()
            goal = torch.zeros(n, 3, device=dev)
            goal[:, 0:2] = holes_w[:, active]
            goal[:, 2] = hover_tip_z(active)
            s = smoothstep(t_in / DROP_STEPS)
            kp = goal.clone()
            kp[:, 2] = glide_from_z + s * (goal[:, 2] - glide_from_z)
            if s >= 1.0:
                pos_off[:] = (pos_off + 0.1 * (goal - key.data.root_pos_w)).clamp(-0.12, 0.12)
                learn_tilt()
            psi_cmd[:] = yaw_of(key.data.root_quat_w) + clock_err(active)
            tp, tq = hand_for_key(kp + pos_off, flip_cmd(psi_cmd))
            act = servo(tp, tq, PICK_W)
            settled = ((goal - key.data.root_pos_w)[:, 0:2].norm(dim=-1) < 0.0004) \
                & ((goal - key.data.root_pos_w)[:, 2].abs() < 0.0012) \
                & (clock_err(active).abs() < math.radians(2.5))
            seat_ok[:] = torch.where(settled, seat_ok + 1, torch.zeros_like(seat_ok))
            if (t_in >= DROP_STEPS + 15 and bool((seat_ok >= 8).all())) or t_in >= DROP_STEPS + 3 * WP_TIMEOUT:
                peck_tries = 0
                seat_ok.zero_()
                start_insert_glide(active)
                phase, marker = "insert", i
        elif phase == "insert":  # straight-down descent through the socket mouth to the floor; a
            # rim-stall rises and re-drops with a fresh free-air bias
            s = smoothstep(t_in / INSERT_STEPS)
            kp = torch.zeros(n, 3, device=dev)
            kp[:, 0:2] = holes_w[:, active]
            kp[:, 2] = bolts[active].data.root_pos_w[:, 2] + glide_from_z + s * (SOCKET_FLOOR_Z - glide_from_z)
            free = bool((tip_axial(active) > SOCKET_MOUTH_Z + 0.0015).all())
            if free and t_in % 2 == 0:  # still contact-free: keep trimming on the way down
                pos_off[:, 0:2] = (pos_off[:, 0:2]
                                   + 0.08 * (kp[:, 0:2] - key.data.root_pos_w[:, 0:2])).clamp(-0.12, 0.12)
                learn_tilt(0.03)
            tp, tq = hand_for_key(kp + pos_off, flip_cmd(psi_cmd))
            act = servo(tp, tq, PICK_W)
            seated = (tip_axial(active) < SOCKET_FLOOR_Z + 0.0012) & (tip_lateral(active) < 0.0012) \
                & (shaft_up() > 0.99)
            seat_ok[:] = torch.where(seated, seat_ok + 1, torch.zeros_like(seat_ok))
            stalled = t_in >= INSERT_STEPS + 15 and bool((tip_axial(active) > SOCKET_MOUTH_Z - 0.001).any())
            if bool((seat_ok >= 5).all()):
                print(f"  inserted at hole {active}: tip "
                      f"{float((tip_axial(active) - SOCKET_FLOOR_Z).mean()) * 1e3:+.2f} mm off the floor "
                      f"after {peck_tries} pecks, lateral {float(tip_lateral(active).max()) * 1e3:.2f} mm", flush=True)
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
            goal[:, 0:2] = holes_w[:, active]
            goal[:, 2] = bolts[active].data.root_pos_w[:, 2] + SOCKET_MOUTH_Z + 0.005
            settled = (goal - key.data.root_pos_w).norm(dim=-1) < 0.0006
            if t_in > 10:
                pos_off[:] = (pos_off + 0.1 * (goal - key.data.root_pos_w)).clamp(-0.12, 0.12)
                learn_tilt(0.05)
            psi_cmd[:] = yaw_of(key.data.root_quat_w) + clock_err(active)
            tp, tq = hand_for_key(goal + pos_off, flip_cmd(psi_cmd))
            act = servo(tp, tq, PICK_W)
            if (t_in >= 20 and bool(settled.all())) or t_in >= WP_TIMEOUT:
                seat_ok.zero_()
                start_insert_glide(active)
                phase, marker = "insert", i
        elif phase == "handoff":  # the key stands in the first socket — let go WITHOUT knocking
            # it over. The unheld key leans toward its handle until the hex binds (a stable
            # tip+wall+rim tripod), and the pads flank the arm exactly along that lean line — so
            # the key settles ONTO a pad. Unload the press + torsion while still welded, GLIDE
            # the fingers open so the pad lowers the leaning key gently onto its bind, dwell,
            # then back the hand straight out of the grip corridor and rise.
            if t_in == 1:
                grip_freeze[:] = art.data.joint_pos[:, fingers]
                back_axis[:] = -quat_apply(hq, ez)  # -approach: straight back out of the grip
                psi_cmd[:] = psi_cmd + math.radians(2.0)  # bleed the clock-servo's wind-up
            if t_in <= 10:  # still welded: lift the press off the floor, unload the torsion
                kp = torch.zeros(n, 3, device=dev)
                kp[:, 0:2] = holes_w[:, active]
                kp[:, 2] = bolts[active].data.root_pos_w[:, 2] + SOCKET_FLOOR_Z + 0.0004
                tp, tq = hand_for_key(kp + pos_off, flip_cmd(psi_cmd))
                act = servo(tp, tq, grip_freeze)
                if t_in == 10:
                    release_p[:] = hp
                    release_q[:] = hq
            else:
                if welded.any():
                    weld_off()
                w = grip_freeze + (OPEN_W - grip_freeze) * smoothstep((t_in - 10) / 25.0)
                wp2 = release_p.clone()
                if t_in > 60:  # the fingers had 1.7 s to spread clear before the hand moves
                    s2 = smoothstep((t_in - 60) / 35.0)
                    wp2 += back_axis * (0.06 * s2)
                if t_in > 95:
                    s3 = smoothstep((t_in - 95) / 35.0)
                    wp2[:, 2] = release_p[:, 2] + s3 * 0.12
                act = servo(wp2, release_q, w)
            if t_in >= 135:
                if bool(in_socket(active, 0.0008).all()) and bool((shaft_up() > 0.90).all()):
                    pos_off.zero_()  # learned in the horizontal-hand configuration — stale for
                    # the top-down regrasp; better to relearn from zero
                    regrasp_tries = 0
                    phase, marker = "regrasp_hover", i
                else:
                    drops += 1
                    if drops > DROP_BUDGET:
                        print("  ABORT: drop budget exhausted", flush=True)
                        phase, marker = "retreat", i
                    elif bool((shaft_up() < 0.5).all()):  # fell flat: pick it up wherever it lies
                        print("  DROP: key left the socket during the handoff — re-picking", flush=True)
                        picks += 1
                        phase, marker = "pick_hover", i
                    else:
                        print("  DROP: key adrift after the handoff", flush=True)
                        phase, marker = "retreat", i
        elif phase == "regrasp_hover":  # top-down hover over the horizontal HANDLE (the crank,
            # riding just above the case's wall top with nothing near it): hand x runs along the
            # crank so the fingers close across its FLATS; the inner finger stays clear of the
            # shaft. The hand arrives HORIZONTAL from the handoff, so the reorientation glides as
            # one axis-angle sweep and the 180-deg grip parity is picked by the smaller rotation.
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
            grip_pt[:] = key.data.root_pos_w + quat_apply(key.data.root_quat_w, crank_grip_local)
            wp_p[:] = grip_pt
            wp_p[:, 2] = grip_pt[:, 2] + hand_to_pad + HOVER_CLEAR
            s = smoothstep(t_in / WP_TIMEOUT)
            ang = glide_aa.norm(dim=-1).clamp_min(1e-9)
            wp_q[:] = quat_mul(quat_from_angle_axis(ang * s, glide_aa / ang.unsqueeze(-1)), glide_q0)
            if s >= 1.0:
                want = grip_pt.clone()
                want[:, 2] += HOVER_CLEAR
                pos_off[:] = (pos_off + 0.12 * (want - pad_centre())).clamp(-0.12, 0.12)
            goal = wp_p + pos_off
            act = servo(glide_from_p + (goal - glide_from_p) * s, wp_q, STRADDLE_W)
            pad_err = (pad_centre()[:, 0:2] - grip_pt[:, 0:2]).norm(dim=-1)
            if (t_in >= WP_TIMEOUT + 10 and bool((pad_err < 0.004).all()) and bool(at(goal, wp_q).all())) \
                    or t_in >= 3 * WP_TIMEOUT:
                phase, marker = "regrasp_down", i
        elif phase == "regrasp_down":  # descend around the crank to its axis height
            s = smoothstep(t_in / 60.0)
            grip_pt[:] = key.data.root_pos_w + quat_apply(key.data.root_quat_w, crank_grip_local)
            wp_p[:] = grip_pt
            wp_p[:, 2] = grip_pt[:, 2] + hand_to_pad + HOVER_CLEAR * (1.0 - s)
            if s >= 1.0:
                pos_off[:] = (pos_off + 0.12 * (grip_pt - pad_centre())).clamp(-0.12, 0.12)
            act = servo(wp_p + pos_off, wp_q, STRADDLE_W)
            pad_on = (pad_centre() - grip_pt).norm(dim=-1) < 0.003
            if (t_in >= 68 and bool(pad_on.all())) or t_in >= 2 * WP_TIMEOUT:
                close_ok.zero_()
                phase, marker = "regrasp_close", i
        elif phase == "regrasp_close":
            s = smoothstep(t_in / CLOSE_STEPS)
            width = STRADDLE_W + (SCREW_W - STRADDLE_W) * s
            act = servo(wp_p + pos_off, wp_q, width)
            gap = art.data.joint_pos[:, fingers].sum(dim=-1)
            grip_pt[:] = key.data.root_pos_w + quat_apply(key.data.root_quat_w, crank_grip_local)
            near = (pad_centre() - grip_pt).norm(dim=-1) < 0.004  # closure vs the LIVE key pose
            ok = near & (gap > CLOSED_MIN) & (gap < CLOSED_MAX)
            close_ok[:] = torch.where(ok, close_ok + 1, torch.zeros_like(close_ok))
            if t_in >= CLOSE_STEPS + 6 and bool((close_ok >= 4).all()):
                weld_on()
                cycles_hole = 0
                print(f"  re-grasped the crank top-down (gap "
                      f"{[f'{float(g) * 1e3:.1f}' for g in gap]} mm) — fastening hole {active}", flush=True)
                phase, marker = "stroke", i
            elif t_in >= CLOSE_STEPS + 30:
                regrasp_tries += 1
                if regrasp_tries < 4 and bool(in_socket(active, 0.0008).all()) and bool((shaft_up() > 0.85).all()):
                    print(f"  regrasp missed (gap {[f'{float(g) * 1e3:.1f}' for g in gap]} mm), retrying", flush=True)
                    phase, marker = "regrasp_hover", i
                elif bool((shaft_up() < 0.5).all()):
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
        elif phase == "stroke":  # press + twist: the engaged key feeds the hole's screw joint,
            # which descends 1 mm/rev until the wrist nears its stop. The tip is commanded BELOW
            # the live socket floor (the press rides the descending bolt); the spin sweeps at a
            # fixed rate; the hand orbits the 22 mm crank circle while its roll winds 1:1.
            if t_in == 1:
                psi_cmd[:] = yaw_of(key.data.root_quat_w)
                coupled_turn[:] = bolt_turn[:, active]  # the hex lash re-charges every re-entry
                stroke_slip0[:] = key_turn - bolt_turn[:, active]
                slip_armed[:] = False
                peck_tries = 0
                cycles_hole += 1
                hole_cycles[active] += 1
            # arm the slip watch only once the arm reads vertical: righting a leaned key changes
            # its READ spin without any true hex slip
            newly_up = (shaft_up() > 0.995) & ~slip_armed
            stroke_slip0[:] = torch.where(newly_up, key_turn - bolt_turn[:, active], stroke_slip0)
            slip_armed |= newly_up
            room = J7_GUARD - art.data.joint_pos[:, j7]
            sweep = torch.full_like(room, STROKE_W / 15.0)
            psi_cmd[:] = psi_cmd - torch.where(room > 0.15, sweep, torch.zeros_like(sweep))
            kp = torch.zeros(n, 3, device=dev)
            kp[:, 0:2] = holes_w[:, active]
            kp[:, 2] = bolts[active].data.root_pos_w[:, 2] + SOCKET_FLOOR_Z - PRESS_LEAD
            tp, tq = hand_for_key(kp + pos_off, quat_from_angle_axis(psi_cmd, ez))
            act = servo(tp, tq, SCREW_W)
            # screw-in sweeps NEGATIVE spin, so a camming key runs AHEAD of the joint in the
            # negative direction — watch the magnitude, not one sign
            slip = ((key_turn - bolt_turn[:, active]) - stroke_slip0).abs() * slip_armed
            done = bool((depth(active) >= STOP_DEPTH).all())
            wound = bool((room <= 0.15).all())
            slipped = bool((slip > SLIP_ABORT).any())
            popped = bool((tip_axial(active) > SOCKET_MOUTH_Z).any())
            if done or wound or slipped or popped or t_in >= STROKE_TIMEOUT:
                # per-stroke hex-drive fidelity: the deliberate rewinds are excluded by design
                drive_slip += ((key_turn - bolt_turn[:, active]) - stroke_slip0) * slip_armed
                if slipped:
                    print(f"  hole {active} stroke {cycles_hole}: hex slipped "
                          f"{float(torch.rad2deg(slip.max())):.0f} deg — recocking", flush=True)
                phase, marker = "unload", i
        elif phase == "unload":  # bleed the torsional wind-up BEFORE lifting: back-rotate a few
            # degrees while still pressed — lifting under stored torsion kicks the key
            if t_in == 1:
                psi_cmd[:] = psi_cmd + math.radians(4.0)
            kp = torch.zeros(n, 3, device=dev)
            kp[:, 0:2] = holes_w[:, active]
            kp[:, 2] = bolts[active].data.root_pos_w[:, 2] + SOCKET_FLOOR_Z - PRESS_LEAD
            tp, tq = hand_for_key(kp + pos_off, quat_from_angle_axis(psi_cmd, ez))
            act = servo(tp, tq, SCREW_W)
            if t_in >= 6:
                hole_done = bool((depth(active) >= STOP_DEPTH).all()) or cycles_hole >= MAX_CYCLES_HOLE
                if hole_done and not bool((depth(active) >= STOP_DEPTH).all()):
                    print(f"  hole {active}: cycle budget exhausted at "
                          f"{float(depth(active).mean()) * 1e3:+.2f} mm", flush=True)
                if hole_done:
                    print(f"  hole {active} fastened: depth {float(depth(active).mean()) * 1e3:+.2f} mm "
                          f"in {cycles_hole} cycles", flush=True)
                    if seq + 1 >= B:
                        phase, marker = "release", i
                    else:
                        advance = True
                        phase, marker = "extract", i
                else:
                    advance = False
                    phase, marker = "extract", i
        elif phase == "extract":  # lift the tip out of the CURRENT socket to the in-case travel
            # height, spin held, xy anchored on the hole it is leaving
            if t_in == 1:
                glide_from_z[:] = key.data.root_pos_w[:, 2]
            s = smoothstep(t_in / 40.0)
            kp = torch.zeros(n, 3, device=dev)
            kp[:, 0:2] = holes_w[:, active]
            kp[:, 2] = glide_from_z + s * (board_z + TRAVEL_TIP_Z - glide_from_z)
            tp, tq = hand_for_key(kp + pos_off, quat_from_angle_axis(psi_cmd, ez))
            act = servo(tp, tq, SCREW_W)
            cleared = bool((key.data.root_pos_w[:, 2] > board_z + TRAVEL_TIP_Z - 0.004).all())
            if t_in >= 44 and cleared:
                if advance:  # only now does the key leave this hole for the next
                    seq += 1
                    active = order[seq]
                    advance = False
                    cycles_hole = 0
                # rewind by a MULTIPLE OF 60 DEG (hex symmetry: clocking preserved exactly),
                # sized to re-arm the wrist near J7_START
                steps60 = torch.round((art.data.joint_pos[:, j7] - J7_START) / (math.pi / 3))
                rewind_tgt[:] = psi_cmd + steps60 * (math.pi / 3)
                phase, marker = "travel", i
            elif t_in >= 2 * WP_TIMEOUT:
                print("  ABORT: key would not lift out of the socket", flush=True)
                phase, marker = "retreat", i
        elif phase == "travel":  # glide to the (same or next) hole at travel height while
            # rewinding the wrist — a swept, monotone spin glide the LONG way round; arrival is
            # LATCHED and the hex-slip correction is a one-shot plus a gentle trickle
            if t_in == 1:
                rw_arrived[:] = False
                glide_from_p[:] = key.data.root_pos_w
            newly_arr = ((rewind_tgt - psi_cmd) < 1e-6) & ~rw_arrived
            rw_arrived |= newly_arr
            psi_cmd[:] = torch.where(newly_arr, psi_cmd + clock_err(active),
                                     torch.where(rw_arrived, psi_cmd + 0.1 * clock_err(active),
                                                 torch.minimum(psi_cmd + REWIND_W / 15.0, rewind_tgt)))
            goal = torch.zeros(n, 3, device=dev)
            goal[:, 0:2] = holes_w[:, active]
            goal[:, 2] = board_z + TRAVEL_TIP_Z
            s = smoothstep(t_in / 70.0)
            kp = glide_from_p + (goal - glide_from_p) * s
            if bool(rw_arrived.all()) and s >= 1.0 and t_in % 2 == 0:  # settled free air
                pos_off[:] = (pos_off + 0.1 * (goal - key.data.root_pos_w)).clamp(-0.12, 0.12)
                learn_tilt()
            tp, tq = hand_for_key(kp + pos_off, flip_cmd(psi_cmd))
            act = servo(tp, tq, SCREW_W)
            xy_err = (key.data.root_pos_w[:, 0:2] - holes_w[:, active]).norm(dim=-1)
            settled = (xy_err < 0.0004) & ((goal[:, 2] - key.data.root_pos_w[:, 2]).abs() < 0.0015)
            clocked = clock_err(active).abs() < math.radians(2.5)
            if bool(rw_arrived.all()) and t_in >= 80 and bool((settled & clocked).all()):
                seat_ok.zero_()
                phase, marker = "re_hover", i
            elif t_in >= 2 * (TRAVEL_TIMEOUT + int(4.8 / (REWIND_W / 15.0))):
                print(f"  ABORT: travel never settled (xy {float(xy_err.max()) * 1e3:.2f}mm, clock "
                      f"{float(torch.rad2deg(clock_err(active).abs().max())):.1f}deg)", flush=True)
                phase, marker = "retreat", i
        elif phase == "re_hover":  # descend from travel height to the insert hover and SETTLE
            # there: the plunge only gets a sub-clearance, streak-held start — a straight drop
            # from travel height drifts more than the 0.4 mm/side play at the close-in holes
            if t_in == 1:
                glide_from_z[:] = key.data.root_pos_w[:, 2]
                seat_ok.zero_()
            goal = torch.zeros(n, 3, device=dev)
            goal[:, 0:2] = holes_w[:, active]
            goal[:, 2] = hover_tip_z(active)
            s = smoothstep(t_in / 50.0)
            kp = goal.clone()
            kp[:, 2] = glide_from_z + s * (goal[:, 2] - glide_from_z)
            if s >= 1.0:
                pos_off[:] = (pos_off + 0.1 * (goal - key.data.root_pos_w)).clamp(-0.12, 0.12)
                learn_tilt()
            psi_cmd[:] = psi_cmd + 0.1 * clock_err(active)
            tp, tq = hand_for_key(kp + pos_off, flip_cmd(psi_cmd))
            act = servo(tp, tq, SCREW_W)
            settled = ((goal - key.data.root_pos_w)[:, 0:2].norm(dim=-1) < 0.0004) \
                & ((goal - key.data.root_pos_w)[:, 2].abs() < 0.0012) \
                & (clock_err(active).abs() < math.radians(2.5))
            seat_ok[:] = torch.where(settled, seat_ok + 1, torch.zeros_like(seat_ok))
            if (t_in >= 55 and bool((seat_ok >= 8).all())) or t_in >= 3 * WP_TIMEOUT:
                seat_ok.zero_()
                start_insert_glide(active)
                phase, marker = "reinsert", i
        elif phase == "reinsert":  # drop the tip from the settled hover into the socket and press
            s = smoothstep(t_in / 45.0)
            zt = glide_from_z + s * ((SOCKET_FLOOR_Z - PRESS_LEAD) - glide_from_z)
            kp = torch.zeros(n, 3, device=dev)
            kp[:, 0:2] = holes_w[:, active]
            kp[:, 2] = bolts[active].data.root_pos_w[:, 2] + zt
            free = bool((tip_axial(active) > SOCKET_MOUTH_Z + 0.0015).all())
            if free and t_in % 2 == 0:
                pos_off[:, 0:2] = (pos_off[:, 0:2]
                                   + 0.08 * (kp[:, 0:2] - key.data.root_pos_w[:, 0:2])).clamp(-0.12, 0.12)
                learn_tilt(0.03)
            tp, tq = hand_for_key(kp + pos_off, flip_cmd(psi_cmd))
            act = servo(tp, tq, SCREW_W)
            entered = (tip_axial(active) < SOCKET_FLOOR_Z + 0.0012) & (tip_lateral(active) < 0.0012)
            seat_ok[:] = torch.where(entered, seat_ok + 1, torch.zeros_like(seat_ok))
            stalled = t_in >= 55 and bool((tip_axial(active) > SOCKET_MOUTH_Z - 0.001).any())
            if bool((seat_ok >= 4).all()):
                phase, marker = "stroke", i
            elif stalled or t_in >= REINSERT_TIMEOUT + 45:
                peck_tries += 1
                insert_pecks += 1
                if peck_tries > INSERT_RETRIES:
                    print("  ABORT: reinsert never re-entered the socket", flush=True)
                    phase, marker = "retreat", i
                else:  # rise back to the hover and re-settle before dropping again
                    phase, marker = "re_hover", i
        elif phase == "release":  # every hole fastened: bleed, let go, rise straight off the
            # crank — nothing sits above it, so the open fingers exit clean
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
                if t_in > 26:  # open FIRST, then rise: rising frozen pads would drag the key up
                    s2 = smoothstep((t_in - 26) / 40.0)
                    wp2[:, 2] = release_p[:, 2] + s2 * 0.12
                act = servo(wp2, release_q, STRADDLE_W if t_in > 10 else grip_freeze)
            if t_in >= 75:
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
        else:  # settle: hands off — the fastened board holds on its own
            act = servo(home_p, home_q, OPEN_W)
            if t_in >= SETTLE_STEPS:
                break

        step(act)

        # Advance the active screw joint from the key's ENGAGED rotation, then write every bolt's
        # kinematic pose. Track the key's spin every step (a stale prev reads as a +-pi jump).
        kcur = yaw_of(key.data.root_quat_w)
        if phase_at_act not in ("show", "stage_hold"):
            dspin = -_wrap(kcur - prev_key_yaw)
            key_turn = key_turn + dspin
            if phase_at_act in ("stroke", "unload"):
                engaged = tip_axial(active) < SOCKET_MOUTH_Z - 0.0008
                coupled_turn += torch.where(engaged, dspin, torch.zeros_like(dspin))
                project_bolts(active)
            else:
                project_bolts(None)
        prev_key_yaw = kcur

        if not torch.isfinite(key.data.root_pos_w).all():
            print("  ABORT: state went non-finite", flush=True)
            break
        if i >= HARD_CAP:  # every phase carries its own timeout; this only guards a livelock
            print("  ABORT: global step budget exhausted", flush=True)
            break
        if i % LOG_EVERY == 0:
            d = depth(active) * 1e3
            q7d = torch.rad2deg(art.data.joint_pos[:, j7])
            gap = art.data.joint_pos[:, fingers].sum(dim=-1) * 1e3
            print(f"  ctrl {i:5d} [{phase:13s}] hole {active} ({seq + 1}/{B}) | cyc {cycles_hole:2d} | "
                  f"depth {float(d.mean()):+6.2f}mm | joint "
                  f"{float(torch.rad2deg(bolt_turn[:, active]).mean()):+7.0f}deg | "
                  f"q7 {float(q7d.mean()):+6.0f}deg | key z {float(key.data.root_pos_w[:, 2].mean()):.3f} "
                  f"up {float(shaft_up().mean()):+.2f} | hand z {float(hp[:, 2].mean()):.3f} | "
                  f"gap {float(gap.mean()):4.1f}mm", flush=True)

    if writer is not None:
        writer.close()
        print("MP4:", args.video, flush=True)

    seated = sc.seated()  # (n, num_holes)
    revs = bolt_turn / (2 * math.pi)
    gain_mm = torch.clamp(bolt_turn, min=0.0) * PITCH / (2 * math.pi) * 1e3
    fastened = [order[k] for k in range(seq + 1)]
    per_hole = " ".join(f"h{b}:{float(gain_mm[:, b].mean()):+.1f}mm/{float(revs[:, b].mean()):.1f}rev"
                        for b in fastened)
    slip = torch.rad2deg(drive_slip)
    print(f"PC-MOTHERBOARD-FRANKA | seated {int(seated[:, list(order)].all(dim=1).sum())}/{n} envs "
          f"({int(seated[:, list(order)].sum())}/{B * n} bolts of {B} driven) | "
          f"{PITCH_MM:.1f} mm/rev by construction | stroke slip {float(slip.mean()):+.1f}deg | "
          f"{sum(hole_cycles)} cycles, {picks} picks, {insert_pecks} pecks, {drops} drops | "
          f"{per_hole}", flush=True)
    close_and_exit(env, app)


if __name__ == "__main__":
    main()

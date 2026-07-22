"""ALOHA smoke for AllenBoltAssemblyScene — the RIGHT WXAI arm picks the allen key off the table
by its long handle, seats the key's tip in the staged bolt's hex socket, and CRANKS the bolt down
the platform's real SDF threads by orbiting the handle about the bolt axis, like a hand on a key.

Grasping follows the benchmark weld-on-closure contract (cf. the pouring suite / so101 fastening):
a normally-disabled FixedJoint gripper<->key is enabled when the jaws are verifiably closed around
the handle and released when they open. Everything else is live physics — key<->socket hex contact,
bolt<->platform thread contact, and the key standing unheld in the socket between grips — so a
missed grasp, a jammed insertion, or a dropped key fails honestly.

The WXAI's big parallel jaws (70 x 60 mm pads) suit the 120 mm handle, and the handle grip is BOTH
the insertion grip and the crank grip — the first stroke starts without ever letting go. Strokes
sweep <= 180 deg (bounded by arm-joint margins), then the jaws open, lift, and re-grip the handle
at its new heading; drops that leave the key upright are recovered by re-grip + re-insert and
counted honestly. The arm runs in its native "joint" mode (stiff Trossen-tuned PD): waypoints go
through a damped-least-squares differential IK in this script, so there is no gravity droop to
compensate. The LEFT arm parks folded across the table. The bolt is teleport-staged thread-captured
over the hole exactly as the Franka smoke did — the robot's job here is the key.

Phases: show -> stage -> pick -> lift -> carry -> erect -> clock -> insert(peck lattice) ->
crank(stroke/rewind/regrip cycles, with drop recovery via re-grip + re-insert) -> retreat ->
settle. Verdict: seated count, depth gained per rev vs the 2.0 mm pitch, key->bolt slip, crank
cycles, and drops.

cosigen/bin/python -m robobench.suites.assembly.smokes.allen_key_aloha_smoke --headless
python -m robobench.suites.assembly.smokes.allen_key_aloha_smoke --livestream 2
python -m robobench.suites.assembly.smokes.allen_key_aloha_smoke \
    --headless --enable_cameras --video robobench/suites/assembly/videos/allen_key_aloha.mp4
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--video", type=str, default="", help="save an mp4 here (needs --headless --enable_cameras)")
parser.add_argument("--cap", type=int, default=5, help="with --video: capture one frame every N control steps (5 at 50 Hz control -> ~3x speed at 30 fps)")
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
    quat_from_matrix,
    quat_mul,
)
from robobench.core import ENVS  # noqa: E402
from robobench.suites.assembly.smokes import close_and_exit  # noqa: E402

if TYPE_CHECKING:
    from robobench.suites.assembly.scenes import AllenBoltAssemblyScene

# Key geometry baked into the committed key USD (tip at origin, 50 mm working arm up +z, 120 mm
# handle +x, hex 12.6 mm across flats / 14.4 mm across corners):
ARM_LEN = 0.050
HANDLE_LEN = 0.120
# Bolt-local geometry (bolt origin = thread TIP, +z up): socket recess floor and mouth (head top).
SOCKET_FLOOR_Z = 0.0355
SOCKET_MOUTH_Z = 0.0428
STAGE_GAP = 0.0015       # staging gap (m) between bolt tip and plate top, just above the thread entry
STOP_DEPTH = 0.0235      # stop cranking at this tip depth (m) — just before the head bottoms at 24.8 mm
PITCH_MM = 2.0           # M16 coarse pitch, the expected descent per revolution
DT = 1.0 / 240.0         # sim timestep (matches the registered env's dt override)

# WXAI joint-mode action semantics per arm: [6 arm joint position targets | 1 gripper carriage
# target] at ~50 Hz; ALOHA concatenates [left | right]. Both carriage joints are driven from the
# one gripper action (mirrored in the robot class; the assembly env uses the MIMIC-FREE asset
# variant — the PhysX mimic freezes the gripper subtree's constraint anchors on this GPU build).
#
# GRASPING IS REAL CONTACT via welded PAD PROXIES: the claws' own collision shapes freeze at
# their spawn pose on this pipeline (asset-specific parser bug, still present on IsaacSim 6.0),
# so each claw carries a free rigid PAD — a box matching the claw's tip block — welded to its
# carriage at reset. Pads are ordinary rigid bodies: their shapes track, the drive's squeeze
# force flows carriage -> weld -> pad -> handle, and a real grasp STALLS the carriage pair at
# the hex width. The weld-on-verified-closure contract then holds the key (benchmark standard),
# with the verification physical again:
#   pad inner faces sit at |y| = carriage - 18 mm  =>  empty close -> pair-mid = 18 mm target;
#   12.6 mm-across-flats handle -> stall at pair-mid ~ 24.3 mm (+ contact offsets).
OPEN_C, CLOSE_C = 0.044, 0.018
# BIMANUAL PINCH: one sphere fingertip per arm, FixedJoint-welded to that arm's link_6 — the
# only runtime attachment this GPU pipeline executes faithfully (key-weld precedent; carriage-
# anchored joints and runtime prismatic joints both fail, see the WIP notes). Grasp force is
# ARM-level: the two arms press their fingertips on the handle from opposite sides; the stall
# gate reads the true tip-center separation across the 12.6 mm hex (hex + 2r = 32.6 mm).
TIP_R = 0.010            # fingertip sphere radius (m)
TIP_ALONG = 0.155        # fingertip center along the tool axis from link_6 (m)
PINCH_MIN, PINCH_MAX = 0.0285, 0.0365  # accept window on tip separation across the handle
PRESS_PAST = 0.002       # command each tip this far past the handle flank (bounded press)
GRASP_DRIFT = 0.008      # the close must not shove the key (m) — a swept-aside key is a miss
# Pad geometry (carriage frame; from the claw collision-mesh tip block): x 50..69.6 mm along the
# finger, walls at |y| 18..23 mm, +-8.6 mm along-handle.
# gripper_left finger-body origin -> claw hook pocket, along the tool (+x of link_6): the finger
# is 70 mm long; the concave hook pocket spans x ~ 52-64 mm (measured from the collision mesh).
# The finger-base offset itself is measured live at reset.
FINGER_TO_GRIP = 0.061
FINGER_TO_TIP = 0.070

# Differential IK (damped least squares) — the servo core replacing the Franka smoke's OSC deltas.
IK_LAMBDA = 0.05
POS_STEP = 0.008         # max Cartesian step per control tick (m; ~0.4 m/s at 50 Hz)
ROT_STEP = 0.06          # max rotation step per control tick (rad; ~3 rad/s)
DQ_STEP = 0.12           # per-joint step clamp (rad per tick)

# Choreography (all waypoints recomputed from live poses; distances in m, angles in rad):
HANDLE_GRIP_D = 0.070    # grip the handle this far out from the elbow (crank radius ~= this)
CARRY_Z = 0.20           # table-frame height the key rides at while carried / erected off-axis
INSERT_HOVER = 0.030     # tip hover above the socket mouth while clocking
PRESS_DZ = 0.0010        # crank press: command the tip this far below the live socket floor
# (the stiff joint-PD press is FAR harder than the franka's compliant OSC — 1.5 mm of
# commanded interpenetration hammered the staged bolt off its helix capture: it then spun
# crest-nested, +76 deg with zero descent, and the key cammed out on the next stroke)
STROKE_RAD = math.radians(120.0)   # nominal crank sweep per cycle (screw-in = negative yaw)
STROKE_W = 1.0           # commanded crank rate (rad/s)
JOINT_MARGIN = 0.12      # end the stroke when any arm joint is this close to a limit (rad)
REWIND_LIFT = 0.045      # lift between strokes so the open jaws clear the handle sweep
MAX_CYCLES = 40          # crank cycle budget (full seat ~ 11 revs at <= 120 deg per stroke)
PICK_RETRIES = 3

# Timeouts in CONTROL steps (~50 Hz -> ~5 substeps each at 1/240).
SHOW_END, STAGE_SETTLE = 60, 120
WP_TIMEOUT, CLOSE_STEPS, STROKE_TIMEOUT, SETTLE_STEPS = 250, 100, 400, 150  # the carriage needs ~2 s to travel its 44 mm
PECK_PERIOD = 40         # ctrl steps per peck (hover-reposition, then press)
INSERT_TIMEOUT = 850     # descent from the hover + a full 19-point peck lattice
LOG_EVERY = 150


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

    env = ENVS.get("assembly.allen_bolt.aloha.joint")().build(num_envs=args.num_envs, device=device)
    sc: AllenBoltAssemblyScene = env.scene  # type: ignore[assignment]
    n = env.num_envs
    dev = device
    ids = torch.arange(n, device=dev)
    bolt, key, plat = sc.bolts[0], sc.keys[0], sc.platforms[0]
    plate_top = sc.cfg.plate_top
    render = (not args.headless) or livestream_on
    ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
    ex1 = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3)

    # The worker is the RIGHT arm; the LEFT arm holds its folded home for the whole run.
    artR = env.robot["right"].articulation
    artL = env.robot["left"].articulation
    slices = env.robot.action_slices
    r_s, l_s = slices["right"], slices["left"]
    arm_ids = artR.find_joints(["joint_[0-5]"])[0]
    grip_id = artR.find_joints(["left_carriage_joint"])[0][0]
    grip_id_r = artR.find_joints(["right_carriage_joint"])[0][0]
    arm_ids_L = artL.find_joints(["joint_[0-5]"])[0]
    grip_id_L = artL.find_joints(["left_carriage_joint"])[0][0]
    ee_idx = artR.body_names.index("link_6")
    act_dim = env.robot.action_dim
    lo_lim = artR.data.joint_pos_limits[:, arm_ids, 0]
    hi_lim = artR.data.joint_pos_limits[:, arm_ids, 1]

    # ----- gripper<->key weld (the grasp contract): a pool of pre-authored, disabled FixedJoints --
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
            j = UsdPhysics.FixedJoint.Define(stage, f"{base}/grip_key_weld_{k}")
            j.CreateBody0Rel().SetTargets([f"{base}/Right/link_6"])
            j.CreateBody1Rel().SetTargets([f"{base}/Key_0/allen_key"])
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateJointEnabledAttr(False)
            j.CreateExcludeFromArticulationAttr(True)  # a maximal-coordinate weld, not a new arm DOF
            row.append(f"{base}/grip_key_weld_{k}")
        weld_paths.append(row)
    weld_paths_L = []
    for e in range(n):
        base = f"/World/envs/env_{e}"
        rowl = []
        for k in range(WELD_POOL):
            j = UsdPhysics.FixedJoint.Define(stage, f"{base}/hold_key_weld_{k}")
            j.CreateBody0Rel().SetTargets([f"{base}/Left/link_6"])
            j.CreateBody1Rel().SetTargets([f"{base}/Key_0/allen_key"])
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateJointEnabledAttr(False)
            j.CreateExcludeFromArticulationAttr(True)
            rowl.append(f"{base}/hold_key_weld_{k}")
        weld_paths_L.append(rowl)

    # ----- contact pads: the REAL grasp surfaces (see the constants note) -------------------------
    # One sphere FINGERTIP per arm — the physical contact surfaces of the bimanual pinch. Each is
    # FixedJoint-welded (runtime latch, the key-weld mechanism) to its arm's LINK_6, hanging
    # TIP_ALONG down the tool axis. Pinch force comes from the ARMS pressing the tips together;
    # the articulation claws stay visual. Welds are authored disabled and latched post-reset.
    from isaaclab.assets import RigidObject, RigidObjectCfg

    tips: dict[str, RigidObject] = {}
    tip_weld_paths: dict[str, list[str]] = {}
    for ai, arm_name in enumerate(("Left", "Right")):
        tips[arm_name] = RigidObject(RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/Tip_{arm_name}",
            spawn=sim_utils.SphereCfg(
                radius=TIP_R,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=5.0, disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.18)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5 + 0.1 * ai, 0.5, 0.6)),
        ))
        row = []
        for e in range(n):
            base = f"/World/envs/env_{e}"
            jp = UsdPhysics.FixedJoint.Define(stage, f"{base}/tip_weld_{arm_name}")
            jp.CreateBody0Rel().SetTargets([f"{base}/{arm_name}/link_6"])
            jp.CreateBody1Rel().SetTargets([f"{base}/Tip_{arm_name}"])
            jp.CreateJointEnabledAttr(False)
            jp.CreateExcludeFromArticulationAttr(True)
            row.append(f"{base}/tip_weld_{arm_name}")
        tip_weld_paths[arm_name] = row

    cam = writer = None
    if args.video:
        import imageio.v2 as imageio
        from isaaclab.sensors import Camera, CameraCfg
        cam = Camera(CameraCfg(prim_path="/World/cam", update_period=0.0, height=720, width=1280, data_types=["rgb"],
                               spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.01, 100.0))))
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args.video, fps=30)

    import os
    from pxr import PhysxSchema

    # A settled key/bolt goes to SLEEP, and sweeping articulation claws do NOT wake a sleeping
    # rigid (they pass through with zero interaction — even its table contact stops reporting).
    # sleepThreshold=0 keeps both parts permanently awake; they are the two bodies the whole
    # task manipulates, so the cost is nil.
    for e in range(n):
        for pth in (f"/World/envs/env_{e}/Key_0/allen_key", f"/World/envs/env_{e}/Bolt_0/allen_bolt"):
            prim = stage.GetPrimAtPath(pth)
            if prim:
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateSleepThresholdAttr(0.0)
            else:
                print(f"[warn] no prim at {pth} (sleepThreshold not set)", flush=True)

    contact_debug = bool(os.environ.get("CONTACT_DEBUG"))
    if contact_debug:  # ground truth for grasp debugging: report every contact pair on the key
        for e in range(n):
            api = PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(f"/World/envs/env_{e}/Key_0/allen_key"))
            api.CreateThresholdAttr(0.0)
        print("[debug] contact report armed on the key", flush=True)
    contact_pairs: set = set()

    def poll_contacts() -> None:
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools
        hdrs, _ = get_physx_simulation_interface().get_contact_report()
        for h in hdrs:
            a = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
            b = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
            if "allen_key" in a or "allen_key" in b:
                contact_pairs.add((a.rsplit("/", 1)[-1], b.rsplit("/", 1)[-1]))

    env.sim.reset()  # re-parse physics so the pre-authored welds (and camera) are picked up
    env.reset()

    hole_xy = plat.data.root_pos_w[:, :2].clone()  # insert bore axis == platform origin (bore-centred)
    plat_z = plat.data.root_pos_w[:, 2].clone()

    import os
    if os.environ.get("KEY_XY"):  # debug: pin the key spawn to reproduce a specific geometry
        _kx, _ky = (float(v) for v in os.environ["KEY_XY"].split(","))
        _st = key.data.root_state_w.clone()
        _st[:, 0], _st[:, 1] = _kx, _ky
        _st[:, 7:] = 0.0
        key.write_root_state_to_sim(_st, ids)
        print(f"[debug] key teleported to ({_kx:.3f},{_ky:.3f})", flush=True)

    if cam is not None:
        p0 = plat.data.root_pos_w[0]
        eye = torch.tensor([[p0[0] + 0.05, p0[1] - 0.45, p0[2] + 0.30]], dtype=torch.float32, device=dev)
        tgt = torch.tensor([[p0[0], p0[1], p0[2] + 0.07]], dtype=torch.float32, device=dev)
        cam.set_world_poses_from_view(eye, tgt)
    print(env.describe(), flush=True)

    # Measure tool-frame (link_6) -> finger-pad offsets once from the live articulation.
    gl = artR.body_names.index("gripper_left")
    _q6 = artR.data.body_quat_w[:, ee_idx]
    _rel = quat_apply_inverse(_q6, artR.data.body_pos_w[:, gl] - artR.data.body_pos_w[:, ee_idx])
    tool_to_grip = float(_rel[0, 0]) + FINGER_TO_GRIP  # along the tool (+x of link_6)
    tool_to_tip = float(_rel[0, 0]) + FINGER_TO_TIP
    print(f"tool->grip point: {tool_to_grip:.4f} m | tool->pad tip: {tool_to_tip:.4f} m", flush=True)

    # LEFT arm action: hold the folded home (its own default joint state) for the whole run.
    left_cmd = torch.zeros(n, l_s.stop - l_s.start, device=dev)
    left_cmd[:, 0:6] = artL.data.default_joint_pos[:, arm_ids_L]
    left_cmd[:, 6] = artL.data.default_joint_pos[:, grip_id_L]

    # ----- weld toggles + grasp transform ---------------------------------------------------------
    rel_p = torch.zeros(n, 3, device=dev)  # key pose in the TOOL (link_6) frame, captured at weld
    rel_q = torch.zeros(n, 4, device=dev)
    welded = torch.zeros(n, dtype=torch.bool, device=dev)
    weld_k = 0  # next fresh joint in the pool (all envs grasp in lockstep)

    def tool_pose() -> tuple[torch.Tensor, torch.Tensor]:
        return artR.data.body_pos_w[:, ee_idx], artR.data.body_quat_w[:, ee_idx]

    def weld_on() -> None:
        nonlocal weld_k
        assert weld_k < WELD_POOL, "weld pool exhausted"
        tp, tq = tool_pose()
        rel_p[:] = quat_apply_inverse(tq, key.data.root_pos_w - tp)
        rel_q[:] = quat_mul(quat_conjugate(tq), key.data.root_quat_w)
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
        tp, tq = tool_pose()
        return (key.data.root_pos_w - (tp + quat_apply(tq, rel_p))).norm(dim=-1)

    # ----- staging (same helix-captured teleport as the Franka smoke) -----------------------------
    def stage_parts() -> None:
        st = torch.zeros(n, 13, device=dev)
        st[:, 0:2] = hole_xy
        st[:, 2] = plat_z + plate_top + STAGE_GAP
        st[:, 3] = 1.0
        bolt.write_root_state_to_sim(st, ids)

    def depth() -> torch.Tensor:  # bolt tip depth below the plate top (m), per env
        return plat_top_depth()

    def plat_top_depth() -> torch.Tensor:
        return plat_z + plate_top - bolt.data.root_pos_w[:, 2]

    # ----- servo core: damped-least-squares differential IK -> joint position targets -------------
    eyeM = torch.eye(6, device=dev).expand(n, 6, 6)

    def ik_arm(tp: torch.Tensor, tq: torch.Tensor) -> torch.Tensor:
        """One DLS step of the RIGHT arm's 6 joints toward the tool waypoint (world frame)."""
        p6, q6 = tool_pose()
        e_p = tp - p6
        pn = e_p.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        e_p = e_p * (pn.clamp(max=POS_STEP) / pn)
        qe = quat_mul(tq, quat_conjugate(q6))
        qe = torch.where(qe[:, :1] >= 0, qe, -qe)
        e_r = axis_angle_from_quat(qe)
        rn = e_r.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        e_r = e_r * (rn.clamp(max=ROT_STEP) / rn)
        e = torch.cat((e_p, 2.0 * e_r), dim=-1)  # (n, 6); rotation weighted up so the DLS doesn't
        # let the position term dominate and park the tool tilted
        jac = artR.root_physx_view.get_jacobians()[:, ee_idx - 1, 0:6, :][:, :, arm_ids]  # (n, 6, 6)
        JJt = jac @ jac.transpose(1, 2) + (IK_LAMBDA**2) * eyeM
        dq = (jac.transpose(1, 2) @ torch.linalg.solve(JJt, e.unsqueeze(-1))).squeeze(-1)
        dq = dq.clamp(-DQ_STEP, DQ_STEP)
        q_tgt = artR.data.joint_pos[:, arm_ids] + dq
        return q_tgt.clamp(lo_lim + 0.02, hi_lim - 0.02)

    def act_of(tp: torch.Tensor, tq: torch.Tensor, grip: float | torch.Tensor) -> torch.Tensor:
        a = torch.zeros(n, act_dim, device=dev)
        a[:, l_s] = left_cmd
        a[:, r_s.start : r_s.start + 6] = ik_arm(tp, tq)
        a[:, r_s.start + 6] = grip
        return a

    def hold_act(grip: float | torch.Tensor) -> torch.Tensor:
        """Freeze the arm at its measured joints (no IK) — the release/bleed posture."""
        a = torch.zeros(n, act_dim, device=dev)
        a[:, l_s] = left_cmd
        a[:, r_s.start : r_s.start + 6] = artR.data.joint_pos[:, arm_ids]
        a[:, r_s.start + 6] = grip
        return a

    def at(tp: torch.Tensor, tq: torch.Tensor, tol_p: float = 0.004, tol_r: float = 0.06) -> torch.Tensor:
        p6, q6 = tool_pose()
        qe = quat_mul(tq, quat_conjugate(q6))
        ang = 2.0 * torch.arccos(qe[:, 0].abs().clamp(max=1.0))
        return ((tp - p6).norm(dim=-1) < tol_p) & (ang < tol_r)

    # ----- LEFT-arm mirrors (the vice hand: press moves + key holds, never cranks) ----------------
    eeL_idx = artL.body_names.index("link_6")
    lo_lim_L = artL.data.joint_pos_limits[:, arm_ids_L, 0]
    hi_lim_L = artL.data.joint_pos_limits[:, arm_ids_L, 1]

    def toolL_pose() -> tuple[torch.Tensor, torch.Tensor]:
        return artL.data.body_pos_w[:, eeL_idx], artL.data.body_quat_w[:, eeL_idx]

    def ik_left(tp: torch.Tensor, tq: torch.Tensor) -> torch.Tensor:
        p6, q6 = toolL_pose()
        e_p = tp - p6
        pn = e_p.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        e_p = e_p * (pn.clamp(max=POS_STEP) / pn)
        qe = quat_mul(tq, quat_conjugate(q6))
        qe = torch.where(qe[:, :1] >= 0, qe, -qe)
        e_r = axis_angle_from_quat(qe)
        rn = e_r.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        e_r = e_r * (rn.clamp(max=ROT_STEP) / rn)
        e = torch.cat((e_p, 2.0 * e_r), dim=-1)
        jac = artL.root_physx_view.get_jacobians()[:, eeL_idx - 1, 0:6, :][:, :, arm_ids_L]
        JJt = jac @ jac.transpose(1, 2) + (IK_LAMBDA**2) * eyeM
        dq = (jac.transpose(1, 2) @ torch.linalg.solve(JJt, e.unsqueeze(-1))).squeeze(-1)
        dq = dq.clamp(-DQ_STEP, DQ_STEP)
        q_tgt = artL.data.joint_pos[:, arm_ids_L] + dq
        return q_tgt.clamp(lo_lim_L + 0.02, hi_lim_L - 0.02)

    def left_to(tp: torch.Tensor, tq: torch.Tensor, grip: float = OPEN_C) -> None:
        """Aim the LEFT arm's next action at a tool waypoint (one DLS step; call every tick)."""
        left_cmd[:, 0:6] = ik_left(tp, tq)
        left_cmd[:, 6] = grip

    def left_hold() -> None:
        left_cmd[:, 0:6] = artL.data.joint_pos[:, arm_ids_L]

    left_home_q = artL.data.default_joint_pos[:, arm_ids_L].clone()

    def left_park() -> None:
        left_cmd[:, 0:6] = left_home_q

    def atL(tp: torch.Tensor, tq: torch.Tensor, tol_p: float = 0.004, tol_r: float = 0.06) -> torch.Tensor:
        p6, q6 = toolL_pose()
        qe = quat_mul(tq, quat_conjugate(q6))
        ang = 2.0 * torch.arccos(qe[:, 0].abs().clamp(max=1.0))
        return ((tp - p6).norm(dim=-1) < tol_p) & (ang < tol_r)

    welded_L = torch.zeros(n, dtype=torch.bool, device=dev)
    weld_k_L = 0

    def weld_on_L() -> None:
        nonlocal weld_k_L
        assert weld_k_L < WELD_POOL, "left weld pool exhausted"
        tpl, tql = toolL_pose()
        for e in range(n):
            j = UsdPhysics.FixedJoint.Get(stage, weld_paths_L[e][weld_k_L])
            relp = quat_apply_inverse(tql, key.data.root_pos_w - tpl)[e].tolist()
            relq = quat_mul(quat_conjugate(tql), key.data.root_quat_w)[e].tolist()
            j.GetLocalPos0Attr().Set(Gf.Vec3f(relp[0], relp[1], relp[2]))
            j.GetLocalRot0Attr().Set(Gf.Quatf(relq[0], Gf.Vec3f(relq[1], relq[2], relq[3])))
            j.GetJointEnabledAttr().Set(True)
        welded_L[:] = True

    def weld_off_L() -> None:
        nonlocal weld_k_L
        for e in range(n):
            UsdPhysics.FixedJoint.Get(stage, weld_paths_L[e][weld_k_L]).GetJointEnabledAttr().Set(False)
        weld_k_L += 1
        welded_L[:] = False

    # ----- bimanual pinch geometry helpers --------------------------------------------------------
    base_R_xy = torch.tensor([[0.20, 0.0]], device=dev).expand(n, 2)
    base_L_xy = torch.tensor([[0.80, 0.0]], device=dev).expand(n, 2)

    def pinch_q(h: torch.Tensor, press: torch.Tensor) -> torch.Tensor:
        """Tool orientation for a fingertip press: tool axis 30 deg inward from vertical along
        `press` (unit, horizontal, perpendicular to the handle `h`), handle along tool z."""
        x = press * 0.5 - ez * 0.8660254
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        y = torch.linalg.cross(h, x)
        return quat_from_matrix(torch.stack((x, y, h), dim=-1))

    def tip_perp(arm_name: str) -> torch.Tensor:
        """Perpendicular distance (m) of the arm's fingertip center to the HANDLE line."""
        hd3 = handle_dir()
        hd3 = hd3 / hd3.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        elbow = key.data.root_pos_w + up_axis_of(key.data.root_quat_w) * ARM_LEN
        rel = tip_pos(arm_name) - elbow
        return (rel - (rel * hd3).sum(-1, keepdim=True) * hd3).norm(dim=-1)

    def tool_for_key(kp: torch.Tensor, kq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Tool waypoint that puts the WELDED key at pose (kp, kq), via the captured grasp frame."""
        tq = quat_mul(kq, quat_conjugate(rel_q))
        return kp - quat_apply(tq, rel_p), tq

    def q_down(yaw: torch.Tensor) -> torch.Tensor:
        """Tool-down orientation: link_6 +x (the tool axis) -> world -z, jaw axis at yaw+90 deg."""
        ey = torch.zeros(n, 3, device=dev)
        ey[:, 1] = 1.0
        pitch = quat_from_angle_axis(torch.full((n,), math.pi / 2, device=dev), ey)  # x^ -> -z^
        return quat_mul(quat_from_angle_axis(yaw, ez), pitch)

    def grasp_frame(h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(quat, approach) for gripping a possibly TILTED handle direction `h`: the handle lies
        along the tool z (as in q_down), the jaws close across it, and the tool axis points along
        the most-vertical direction perpendicular to the handle — for a horizontal handle this
        degenerates exactly to q_down. `approach` is that (unit, upward) offset direction."""
        h = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        nv = ez - (ez * h).sum(-1, keepdim=True) * h
        nv = nv / nv.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        x = -nv
        y = torch.linalg.cross(h, x)
        return quat_from_matrix(torch.stack((x, y, h), dim=-1)), nv

    def joint_cost(tq_a: torch.Tensor, tp: torch.Tensor) -> torch.Tensor:
        """|dq| of one virtual DLS step toward (tp, tq_a) — a cheap reachability/comfort score."""
        return (ik_arm(tp, tq_a) - artR.data.joint_pos[:, arm_ids]).norm(dim=-1)

    def key_tip_axial() -> torch.Tensor:
        """Key tip height above the bolt origin along the bolt axis (m): SOCKET_MOUTH_Z at the
        recess mouth, SOCKET_FLOOR_Z when fully seated on the floor."""
        ub = up_axis_of(bolt.data.root_quat_w)
        return ((key.data.root_pos_w - bolt.data.root_pos_w) * ub).sum(-1)

    def key_lateral() -> torch.Tensor:
        ub = up_axis_of(bolt.data.root_quat_w)
        rel = key.data.root_pos_w - bolt.data.root_pos_w
        return (rel - (rel * ub).sum(-1, keepdim=True) * ub).norm(dim=-1)

    def handle_dir() -> torch.Tensor:
        """World direction of the key's handle (local +x), shape (n, 3)."""
        return quat_apply(key.data.root_quat_w, ex1)

    def bolt_ok(say: bool = False, up_min: float = 0.99) -> torch.Tensor:
        """The bolt is still where the choreography needs it: in the hole, near upright."""
        axis_err = (bolt.data.root_pos_w[:, 0:2] - hole_xy).norm(dim=-1)
        ok = (axis_err < 0.004) & (depth() > 0.001) & (up_axis_of(bolt.data.root_quat_w)[:, 2] > up_min)
        if say and not bool(ok.all()):
            tilt = torch.rad2deg(torch.acos(up_axis_of(bolt.data.root_quat_w)[:, 2].clamp(-1, 1)))
            print(f"    bolt state: axis err {float(axis_err.max()) * 1e3:.1f}mm | depth "
                  f"{float(depth().min()) * 1e3:+.1f}mm | tilt {float(tilt.max()):.1f}deg", flush=True)
        return ok

    def arm_near_limit() -> torch.Tensor:
        """True where any arm joint is near a limit AND still moving INTO it. The insert leaves
        several joints parked near (but not crossing) limits; ending every stroke on mere
        proximity cut sweeps to ~10-40 deg of the commanded 120."""
        q = artR.data.joint_pos[:, arm_ids]
        qd = artR.data.joint_vel[:, arm_ids]
        lo_hit = (q - lo_lim < JOINT_MARGIN) & (qd < -0.02)
        hi_hit = (hi_lim - q < JOINT_MARGIN) & (qd > 0.02)
        return (lo_hit | hi_hit).any(dim=-1)

    # ----- step + capture -------------------------------------------------------------------------
    step_i = 0

    grip_id_Lr = artL.find_joints(["right_carriage_joint"])[0][0]

    def step(action: torch.Tensor) -> None:
        nonlocal step_i
        step_i += 1
        # mirror each arm's 1-dof gripper action onto its RIGHT carriage target explicitly
        # (position targets persist across steps; harmless if the controller already served it)
        artL.set_joint_position_target(action[:, l_s.start + 6].unsqueeze(-1), joint_ids=[grip_id_Lr])
        artR.set_joint_position_target(action[:, r_s.start + 6].unsqueeze(-1), joint_ids=[grip_id_r])
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
    bolt_turn = torch.zeros(n, device=dev)  # cumulative screw-in rotation (rad, +ve = descending)
    key_turn = torch.zeros(n, device=dev)
    prev_bolt_yaw = yaw_of(bolt.data.root_quat_w)
    prev_key_yaw = yaw_of(key.data.root_quat_w)
    insert_depth0 = None
    cycles, drops, picks, regrip_tries, aim_tries = 0, 0, 0, 0, 0
    grip_freeze = torch.zeros(n, device=dev)   # carriage target latched at release (bleed the squeeze)
    seat_ok = torch.zeros(n, device=dev)       # consecutive ticks the inserted key has read settled
    psi_star = torch.zeros(n, device=dev)      # latched hex-aligned key yaw for clock/insert
    stroke_psi = torch.zeros(n, device=dev)    # commanded crank sweep so far (rad, >= 0)
    stroke_y0 = torch.zeros(n, device=dev)     # key yaw at stroke start
    grip_pt = torch.zeros(n, 3, device=dev)
    pinch_n = torch.zeros(n, 3, device=dev)  # unit, horizontal: grip -> the RIGHT arm's side
    # Carriage command while HOLDING the welded key. Pressing the full CLOSE_C past the stall
    # turns the hook undersides into a downward wedge that pins the key to the table (the lift
    # then can't raise the arm at all) — so on weld, relax to the measured stall + 0.5 mm.
    grip_c: float | torch.Tensor = CLOSE_C
    wp_p = torch.zeros(n, 3, device=dev)
    wp_q = torch.zeros(n, 4, device=dev)

    # ----- place the fingertips at the tool tips and latch their welds ---------------------------
    for arm_name, art in (("Left", artL), ("Right", artR)):
        eei = art.body_names.index("link_6")
        q6a = art.data.body_quat_w[:, eei]
        off = torch.tensor([[TIP_ALONG, 0.0, 0.0]], device=dev).expand(n, 3)
        st = torch.zeros(n, 13, device=dev)
        st[:, 0:3] = art.data.body_pos_w[:, eei] + quat_apply(q6a, off)
        st[:, 3:7] = q6a
        tips[arm_name].write_root_state_to_sim(st, ids)
    for _ in range(2):
        step(hold_act(OPEN_C))
    for arm_name in tip_weld_paths:
        for e in range(n):
            UsdPhysics.FixedJoint.Get(stage, tip_weld_paths[arm_name][e]).GetJointEnabledAttr().Set(True)
    for _ in range(5):
        step(hold_act(OPEN_C))
    print("[tips] 2 fingertips welded to the wrists (bimanual pinch hardware)", flush=True)

    def tip_pos(arm_name: str) -> torch.Tensor:
        tips[arm_name].update(0.0)
        return tips[arm_name].data.root_pos_w

    def pinch_sep(nvec: torch.Tensor) -> torch.Tensor:
        """Tip-center separation projected on the pinch axis (m)."""
        return ((tip_pos("Right") - tip_pos("Left")) * nvec).sum(-1).abs()

    phase, marker = "show", 0
    i = 0
    while True:
        i += 1
        t_in = i - marker
        tp_now, tq_now = tool_pose()
        act = hold_act(OPEN_C)  # default: hold still, jaws open

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
                phase, marker = "pinch_high", i
                picks += 1
        elif phase == "pinch_high":  # both arms to their sides of the handle, high standoff.
            # The pinch axis n is horizontal, perpendicular to the handle; the RIGHT arm takes
            # the side facing its own base, the LEFT mirrors — each presses from home turf.
            hd = handle_dir()
            hd2 = hd.clone(); hd2[:, 2] = 0.0
            hd2 = hd2 / hd2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            elbow = key.data.root_pos_w + up_axis_of(key.data.root_quat_w) * ARM_LEN
            grip_pt[:] = elbow + hd2 * HANDLE_GRIP_D
            nv = torch.stack((-hd2[:, 1], hd2[:, 0], torch.zeros(n, device=dev)), dim=-1)
            sgn = torch.sign(((base_R_xy - grip_pt[:, :2]) * nv[:, :2]).sum(-1)).unsqueeze(-1)
            sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
            pinch_n[:] = nv * sgn
            tqR = pinch_q(hd2, -pinch_n)
            tqL = pinch_q(-hd2, pinch_n)
            tipR = grip_pt + pinch_n * 0.06 + ez * 0.06
            tipL = grip_pt - pinch_n * 0.06 + ez * 0.06
            wp_p[:] = tipR - quat_apply(tqR, ex1) * TIP_ALONG
            wp_q[:] = tqR
            left_to(tipL - quat_apply(tqL, ex1) * TIP_ALONG, tqL)
            act = act_of(wp_p, wp_q, OPEN_C)
            if bool((at(wp_p, wp_q, 0.006, 0.10)).all()) or t_in >= WP_TIMEOUT:
                phase, marker = "pinch_in", i
        elif phase == "pinch_in":  # descend to handle height, still outside the flanks
            hd = handle_dir()
            hd2 = hd.clone(); hd2[:, 2] = 0.0
            hd2 = hd2 / hd2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            elbow = key.data.root_pos_w + up_axis_of(key.data.root_quat_w) * ARM_LEN
            grip_pt[:] = elbow + hd2 * HANDLE_GRIP_D
            tqR = pinch_q(hd2, -pinch_n)
            tqL = pinch_q(-hd2, pinch_n)
            tipR = grip_pt + pinch_n * 0.045
            tipL = grip_pt - pinch_n * 0.045
            tipR[:, 2] = 0.008
            tipL[:, 2] = 0.008
            wp_p[:] = tipR - quat_apply(tqR, ex1) * TIP_ALONG
            wp_q[:] = tqR
            left_to(tipL - quat_apply(tqL, ex1) * TIP_ALONG, tqL)
            act = act_of(wp_p, wp_q, OPEN_C)
            if bool((at(wp_p, wp_q, 0.005, 0.08)).all()) or t_in >= WP_TIMEOUT:
                phase, marker = "pinch_close", i
        elif phase == "pinch_close":  # both arms press inward; REAL bilateral stall on the hex
            if t_in == 1:
                close_key0 = key.data.root_pos_w[:, :2].clone()
                for _an in ("Left", "Right"):
                    pp = tip_pos(_an)[0]
                    print(f"    [pinch] tip {_an} ({float(pp[0]):+.3f},{float(pp[1]):+.3f},{float(pp[2]):+.4f})"
                          f" | grip ({float(grip_pt[0, 0]):+.3f},{float(grip_pt[0, 1]):+.3f})", flush=True)
            hd = handle_dir()
            hd2 = hd.clone(); hd2[:, 2] = 0.0
            hd2 = hd2 / hd2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            tqR = pinch_q(hd2, -pinch_n)
            tqL = pinch_q(-hd2, pinch_n)
            reach = 0.0063 + TIP_R - PRESS_PAST
            tipR = grip_pt + pinch_n * reach
            tipL = grip_pt - pinch_n * reach
            tipR[:, 2] = 0.008
            tipL[:, 2] = 0.008
            wp_p[:] = tipR - quat_apply(tqR, ex1) * TIP_ALONG
            wp_q[:] = tqR
            left_to(tipL - quat_apply(tqL, ex1) * TIP_ALONG, tqL)
            act = act_of(wp_p, wp_q, OPEN_C)
            if t_in % 20 == 0:
                sep = pinch_sep(pinch_n)
                kp0 = key.data.root_pos_w
                print(f"    [pinch t{t_in:3d}] sep {float(sep[0]) * 1e3:5.1f}mm"
                      f" | key ({float(kp0[0, 0]):+.3f},{float(kp0[0, 1]):+.3f},{float(kp0[0, 2]):+.4f})", flush=True)
            if t_in >= CLOSE_STEPS:
                sep = pinch_sep(pinch_n)
                mid = 0.5 * (tip_pos("Right") + tip_pos("Left"))
                near = (mid[:, :2] - grip_pt[:, :2]).norm(dim=-1) < 0.02
                drift = (key.data.root_pos_w[:, :2] - close_key0).norm(dim=-1)
                ok = (sep > PINCH_MIN) & (sep < PINCH_MAX) & near & (drift < GRASP_DRIFT)
                if bool(ok.all()):
                    weld_on()
                    phase, marker = "pinch_off", i
                elif picks < PICK_RETRIES:
                    print(f"  pinch {picks} missed (sep {sep.tolist()}, drift {drift.tolist()}), retrying", flush=True)
                    phase, marker = "pinch_high", i
                    picks += 1
                else:
                    print("  ABORT: pinch failed", flush=True)
                    phase, marker = "retreat", i
        elif phase == "pinch_off":  # grasp welded to the RIGHT wrist: the left backs out
            tpl, tql = toolL_pose()
            left_to(tpl - pinch_n * 0.002 + ez * 0.002, tql)
            act = hold_act(grip_c)
            if t_in >= 50:
                left_park()
                phase, marker = "lift", i
        elif phase == "lift":  # raise the welded key straight up (weld-relative, pose-agnostic)
            if t_in == 1:
                lift_kp = key.data.root_pos_w.clone()
                lift_kp[:, 2] += 0.10
                lift_kq = key.data.root_quat_w.clone()
            tp, tq = tool_for_key(lift_kp, lift_kq)
            act = act_of(tp, tq, grip_c)
            if bool(((lift_kp - key.data.root_pos_w).norm(dim=-1) < 0.02).all()) or t_in >= WP_TIMEOUT:
                phase, marker = "carry", i
        elif phase == "carry":  # key (still flat in the grip) to the erection spot — off-axis and
            # high, so the erection swing happens in clear air away from the staged bolt
            kp = torch.zeros(n, 3, device=dev)
            kp[:, 0] = hole_xy[:, 0]
            kp[:, 1] = hole_xy[:, 1] + 0.10
            kp[:, 2] = plat_z + CARRY_Z
            tp, tq = tool_for_key(kp, key.data.root_quat_w)  # translate only; keep orientation
            act = act_of(tp, tq, grip_c)
            if bool(((kp - key.data.root_pos_w).norm(dim=-1) < 0.02).all()) or t_in >= WP_TIMEOUT:
                phase, marker = "erect", i
        elif phase == "erect":  # rotate the welded key arm tip-down: the target key orientation is
            # pure-yaw (arm up, tip down). The handle azimuth is FREE (the clock phase re-yaws to
            # the nearest hex representative anyway), and which azimuth the wrist can actually
            # roll to is hard to predict — so try them in predicted-cost order and SWITCH to the
            # next candidate whenever the roll plateaus short of vertical
            kp = torch.zeros(n, 3, device=dev)
            kp[:, 0] = hole_xy[:, 0]
            kp[:, 1] = hole_xy[:, 1] + 0.10
            kp[:, 2] = plat_z + CARRY_Z
            up_now = up_axis_of(key.data.root_quat_w)[:, 2]
            if t_in == 1:
                hd = handle_dir()
                psi0 = torch.atan2(hd[:, 1], hd[:, 0])
                # score candidates at BOTH poses this azimuth must serve: the erect spot AND the
                # subsequent clock hover over the bolt (an azimuth that erects but leaves the
                # wrist unable to reach the bolt strands the insert 60+ mm short at joint limits)
                kp_clock = bolt.data.root_pos_w + ez * (SOCKET_MOUTH_Z + INSERT_HOVER)
                cands, costs = [], []
                for dpsi in (0.0, math.pi / 2, -math.pi / 2, math.pi):
                    cand = _wrap(psi0 + dpsi)
                    kq_c = quat_from_angle_axis(cand, ez)
                    tp_c, tq_c = tool_for_key(kp, kq_c)
                    tp_k, tq_k = tool_for_key(kp_clock, kq_c)
                    cands.append(cand)
                    costs.append(joint_cost(tq_c, tp_c) + joint_cost(tq_k, tp_k))
                erect_order = torch.stack(costs, dim=-1).argsort(dim=-1)  # (n, 4) best-first
                erect_cands = torch.stack(cands, dim=-1)  # (n, 4)
                erect_i = 0
                erect_best, erect_watch = -1.0, i
                psi_star[:] = erect_cands.gather(-1, erect_order[:, 0:1]).squeeze(-1)
            if float(up_now.min()) > erect_best + 0.01:
                erect_best, erect_watch = float(up_now.min()), i
            elif i - erect_watch > 80 and erect_i < 3:  # plateaued short of vertical: next azimuth
                erect_i += 1
                psi_star[:] = erect_cands.gather(-1, erect_order[:, erect_i:erect_i + 1]).squeeze(-1)
                erect_best, erect_watch = -1.0, i
                print(f"    [erect] plateau at up_z {float(up_now.min()):+.2f} — azimuth candidate {erect_i + 1}/4", flush=True)
            kq = quat_from_angle_axis(psi_star, ez)
            tp, tq = tool_for_key(kp, kq)
            act = act_of(tp, tq, grip_c)
            upright = (up_now > 0.995)
            if bool(upright.all()):
                phase, marker = "clock", i
            elif t_in >= 4 * WP_TIMEOUT:
                print(f"  ABORT: erection stalled (key up_z {float(up_now.min()):+.2f})", flush=True)
                phase, marker = "retreat", i
        elif phase == "clock":  # translate to the hover over the BORE axis, hex-clocked to the bolt
            # (nearest mod-60 representative to the key's current yaw — minimal rotation)
            if t_in == 1:
                ky = yaw_of(key.data.root_quat_w)
                dpsi = _wrap((yaw_of(bolt.data.root_quat_w) - ky) % (math.pi / 3.0))
                dpsi = torch.where(dpsi > math.pi / 6.0, dpsi - math.pi / 3.0, dpsi)
                psi_star[:] = ky + dpsi
            kq = quat_from_angle_axis(psi_star, ez)
            kp = torch.zeros(n, 3, device=dev)
            kp[:, 0:2] = bolt.data.root_pos_w[:, 0:2]
            kp[:, 2] = bolt.data.root_pos_w[:, 2] + SOCKET_MOUTH_Z + INSERT_HOVER
            tp, tq = tool_for_key(kp, kq)
            act = act_of(tp, tq, grip_c)
            yaw_err = _wrap(yaw_of(key.data.root_quat_w) - psi_star).abs()
            centred = ((key.data.root_pos_w[:, 0:2] - bolt.data.root_pos_w[:, 0:2]).norm(dim=-1) < 0.001) \
                & (yaw_err < math.radians(2.5))
            if not bool(bolt_ok(say=True).all()):
                print("  ABORT: bolt left its nest before insertion", flush=True)
                phase, marker = "retreat", i
            elif bool(centred.all()) or t_in >= 2 * WP_TIMEOUT:
                phase, marker = "insert", i
        elif phase == "insert":  # hover-peck probe (the hex clearance is 0.7 mm/side): reposition
            # in FREE AIR between straight-down pecks over a small lateral lattice around the
            # bolt's live axis; the peck that lines up drops in, and a caught tip is driven to the
            # floor. The stiff joint PD tracks mm-level, so most runs catch on the first pecks.
            if t_in == 1:
                seat_ok.zero_()
            if t_in % PECK_PERIOD == 1:  # re-latch the clocking each peck: taps walk the bolt's yaw
                ky = yaw_of(key.data.root_quat_w)
                dpsi = _wrap((yaw_of(bolt.data.root_quat_w) - ky) % (math.pi / 3.0))
                dpsi = torch.where(dpsi > math.pi / 6.0, dpsi - math.pi / 3.0, dpsi)
                psi_star[:] = ky + dpsi
            kq = quat_from_angle_axis(psi_star, ez)
            probe = (t_in // PECK_PERIOD) % 19
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
            hover = (t_in % PECK_PERIOD) < PECK_PERIOD // 2
            z_cmd = mouth_z + (0.002 if hover else -0.0012)
            tip_in = key_tip_axial() < SOCKET_MOUTH_Z - 0.001  # caught: drive to the floor
            kp = torch.zeros(n, 3, device=dev)
            kp[:, 0] = bolt.data.root_pos_w[:, 0] + torch.where(tip_in, torch.zeros_like(depth()), torch.full_like(depth(), ox))
            kp[:, 1] = bolt.data.root_pos_w[:, 1] + torch.where(tip_in, torch.zeros_like(depth()), torch.full_like(depth(), oy))
            kp[:, 2] = torch.where(tip_in, floor_z, z_cmd.clamp(min=floor_z))
            tp, tq = tool_for_key(kp, kq)
            act = act_of(tp, tq, grip_c)
            if t_in % PECK_PERIOD == PECK_PERIOD - 1:  # end of each press: where did it land?
                dp = _wrap((yaw_of(bolt.data.root_quat_w) - yaw_of(key.data.root_quat_w)) % (math.pi / 3.0))
                dp = torch.where(dp > math.pi / 6.0, dp - math.pi / 3.0, dp)
                lat = (key.data.root_pos_w[:, 0:2] - bolt.data.root_pos_w[:, 0:2]).norm(dim=-1)
                over = key_tip_axial() - SOCKET_MOUTH_Z
                print(f"    [probe {probe:2d}] lat {float(lat.max()) * 1e3:4.2f}mm | dpsi "
                      f"{float(torch.rad2deg(dp.abs().max())):4.1f}deg | tip over mouth {float(over.min()) * 1e3:+5.2f}mm", flush=True)
            settled = (key_tip_axial() < SOCKET_FLOOR_Z + 0.002) & (key_tip_axial() > SOCKET_FLOOR_Z - 0.002) \
                & (key_lateral() < 0.0015) & (up_axis_of(key.data.root_quat_w)[:, 2] > 0.99) & bolt_ok()
            seat_ok[:] = torch.where(settled, seat_ok + 1, torch.zeros_like(seat_ok))
            if not bool(bolt_ok(say=True).all()):
                print("  ABORT: bolt knocked out of its nest during insertion", flush=True)
                phase, marker = "retreat", i
            elif bool((seat_ok >= 15).all()):  # flush on the floor, upright, centred — and STAYING so
                aim_tries = 0
                if insert_depth0 is None:
                    insert_depth0 = depth().clone()
                stroke_psi.zero_()
                stroke_y0[:] = yaw_of(key.data.root_quat_w)
                cycles += 1
                if cycles == 1:  # metrics start with the crank: forget pre-crank handling
                    bolt_turn.zero_()
                    key_turn.zero_()
                    prev_bolt_yaw = yaw_of(bolt.data.root_quat_w)
                    prev_key_yaw = yaw_of(key.data.root_quat_w)
                phase, marker = "stroke", i  # the handle grip IS the crank grip: no handoff
            elif t_in >= INSERT_TIMEOUT:
                aim_tries += 1
                if aim_tries >= 5:
                    drops += 1
                    print("  DROP: insertion never landed after repeated re-clocks", flush=True)
                    phase, marker = "retreat", i
                else:
                    print("  insert timed out, re-clocking", flush=True)
                    phase, marker = "clock", i
        elif phase == "stroke":  # crank: orbit the handle about the bolt axis while pressing the
            # tip to the socket floor — the welded key's pose target fully determines the tool's
            # orbiting waypoint, and the stiff joint PD tracks it (torque comes from the lever)
            if t_in == 1:
                stroke_slip0 = key_turn - bolt_turn
            stroke_psi += STROKE_W / 50.0
            stroke_psi.clamp_(max=STROKE_RAD)
            kq = quat_from_angle_axis(stroke_y0 - stroke_psi, ez)
            ub = up_axis_of(bolt.data.root_quat_w)
            kp = bolt.data.root_pos_w + ub * (SOCKET_FLOOR_Z - PRESS_DZ)
            tp, tq = tool_for_key(kp, kq)
            act = act_of(tp, tq, grip_c)
            done = depth() >= STOP_DEPTH
            swept = stroke_psi >= STROKE_RAD - 1e-6
            # cam-out guard: hex slip within THIS stroke means the key is climbing out of the
            # socket — end the stroke early; the regrip re-clocks the hex before the next one
            slipping = ((key_turn - bolt_turn) - stroke_slip0).abs() > math.radians(8.0)
            if bool(done.all()):
                phase, marker = "retreat", i
            elif bool((swept | arm_near_limit() | slipping).all()) or t_in >= STROKE_TIMEOUT:
                phase, marker = "hold_press", i
        elif phase == "hold_press":  # VICE HAND: the LEFT arm presses its fingertip on the
            # handle (the key is still welded to the RIGHT, so the press verifies real contact
            # against a rigid target), then takes over the hold so the right can unwind freely.
            hd = handle_dir()
            hd2 = hd.clone(); hd2[:, 2] = 0.0
            hd2 = hd2 / hd2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            elbow = key.data.root_pos_w + up_axis_of(key.data.root_quat_w) * ARM_LEN
            hp = elbow + hd2 * 0.045
            nv = torch.stack((-hd2[:, 1], hd2[:, 0], torch.zeros(n, device=dev)), dim=-1)
            sgnL = torch.sign(((base_L_xy - hp[:, :2]) * nv[:, :2]).sum(-1)).unsqueeze(-1)
            sgnL = torch.where(sgnL == 0, torch.ones_like(sgnL), sgnL)
            press = -(nv * sgnL)
            tqL = pinch_q(-hd2, press)
            tip_tgt = hp + nv * sgnL * (0.0063 + TIP_R - PRESS_PAST)
            approach = tip_tgt - quat_apply(tqL, ex1) * TIP_ALONG
            if t_in < 60:  # standoff first, then press in
                left_to(approach + nv * sgnL * 0.04, tqL)
            else:
                left_to(approach, tqL)
            act = hold_act(grip_c)
            if t_in >= 120 and t_in % 20 == 0:
                dperp = tip_perp("Left")
                okp = (dperp > TIP_R + 0.0043) & (dperp < TIP_R + 0.0093)
                print(f"    [hold t{t_in:3d}] L-tip perp {float(dperp[0]) * 1e3:5.1f}mm", flush=True)
                if bool(okp.all()):
                    weld_on_L()
                    weld_off()
                    print("  vice handoff: LEFT holds, RIGHT unwinding", flush=True)
                    phase, marker = "unwind", i
            if t_in >= 3 * WP_TIMEOUT and phase == "hold_press":
                print("  ABORT: vice press never verified", flush=True)
                phase, marker = "retreat", i
        elif phase == "unwind":  # the right arm returns to its home posture, key safe in the vice
            a = torch.zeros(n, act_dim, device=dev)
            a[:, l_s] = left_cmd
            a[:, r_s.start : r_s.start + 6] = artR.data.default_joint_pos[:, arm_ids]
            a[:, r_s.start + 6] = OPEN_C
            act = a
            left_hold()
            qerr = (artR.data.joint_pos[:, arm_ids] - artR.data.default_joint_pos[:, arm_ids]).abs().max()
            if float(qerr) < 0.10 or t_in >= WP_TIMEOUT:
                phase, marker = "re_press", i
        elif phase == "re_press":  # the RIGHT presses from ITS side and takes the weld back
            hd = handle_dir()
            hd2 = hd.clone(); hd2[:, 2] = 0.0
            hd2 = hd2 / hd2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            elbow = key.data.root_pos_w + up_axis_of(key.data.root_quat_w) * ARM_LEN
            hp = elbow + hd2 * HANDLE_GRIP_D
            nv = torch.stack((-hd2[:, 1], hd2[:, 0], torch.zeros(n, device=dev)), dim=-1)
            sgnR = torch.sign(((base_R_xy - hp[:, :2]) * nv[:, :2]).sum(-1)).unsqueeze(-1)
            sgnR = torch.where(sgnR == 0, torch.ones_like(sgnR), sgnR)
            press = -(nv * sgnR)
            tqR2 = pinch_q(hd2, press)
            tip_tgt = hp + nv * sgnR * (0.0063 + TIP_R - PRESS_PAST)
            approach = tip_tgt - quat_apply(tqR2, ex1) * TIP_ALONG
            wp_q[:] = tqR2
            wp_p[:] = approach + (nv * sgnR * 0.04 if t_in < 60 else 0.0)
            act = act_of(wp_p, wp_q, grip_c)
            left_hold()
            if t_in >= 120 and t_in % 20 == 0:
                dperp = tip_perp("Right")
                okp = (dperp > TIP_R + 0.0043) & (dperp < TIP_R + 0.0093)
                print(f"    [re-press t{t_in:3d}] R-tip perp {float(dperp[0]) * 1e3:5.1f}mm", flush=True)
                if bool(okp.all()):
                    weld_on()
                    weld_off_L()
                    stroke_psi.zero_()
                    stroke_y0[:] = yaw_of(key.data.root_quat_w)
                    cycles += 1
                    print("  vice handoff back: RIGHT holds, cranking", flush=True)
                    phase, marker = "hand_clear", i
            if t_in >= 3 * WP_TIMEOUT and phase == "re_press":
                print("  ABORT: right re-press never verified", flush=True)
                phase, marker = "retreat", i
        elif phase == "hand_clear":  # the vice hand backs off before the orbit sweeps
            hd = handle_dir()
            hd2 = hd.clone(); hd2[:, 2] = 0.0
            hd2 = hd2 / hd2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            nv = torch.stack((-hd2[:, 1], hd2[:, 0], torch.zeros(n, device=dev)), dim=-1)
            sgnL = torch.sign(((base_L_xy - key.data.root_pos_w[:, :2]) * nv[:, :2]).sum(-1)).unsqueeze(-1)
            tpl, tql = toolL_pose()
            back = tpl + nv * sgnL * 0.002 + ez * 0.0015
            left_to(back, tql)
            act = hold_act(grip_c)
            if t_in >= 60:
                left_park()
                phase, marker = "stroke", i
        elif phase == "retreat":
            if welded.any():
                weld_off()
            if t_in == 1:
                wp_p[:] = tp_now
                wp_p[:, 2] = plat_z + 0.25
                wp_q[:] = tq_now
            act = act_of(wp_p, wp_q, OPEN_C)
            if t_in >= WP_TIMEOUT // 2:
                phase, marker = "settle", i
        else:  # settle
            act = hold_act(OPEN_C)
            if t_in >= SETTLE_STEPS:
                break

        prev_depth = depth().clone()
        step(act)
        dd = (depth() - prev_depth).abs()
        if phase in ("pinch_in", "pinch_close", "lift", "carry", "erect", "clock", "insert") and float(dd.max()) > 0.0015:
            kt = key.data.root_pos_w
            print(f"    [watchdog] ctrl {i} {phase}: bolt depth jumped {float(dd.max()) * 1e3:+.1f}mm | "
                  f"key ({kt[0, 0]:.3f},{kt[0, 1]:.3f},{kt[0, 2]:.3f}) | tool z {tool_pose()[0][0, 2]:.3f}", flush=True)
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
            gap = artR.data.joint_pos[:, grip_id] * 1e3
            wd = float(weld_drift().max() * 1e3) if bool(welded.any()) else float("nan")
            kup = float(up_axis_of(key.data.root_quat_w)[:, 2].mean())
            near = "LIM" if bool(arm_near_limit().any()) else "ok"
            print(f"  ctrl {i:5d} [{phase:13s}] | cyc {cycles:2d} | tip depth {d.mean():+6.2f}mm | bolt "
                  f"{torch.rad2deg(bolt_turn).mean():+7.0f}deg | slip {slip.mean():+6.1f}deg | joints {near} | "
                  f"key z {key.data.root_pos_w[:, 2].mean():.3f} up {kup:+.2f} | tool z {tp_now[:, 2].mean():.3f} | "
                  f"carriage {gap.mean():4.1f}mm | weld {wd:4.1f}mm", flush=True)
            if not torch.isfinite(bolt.data.root_pos_w).all() or float((bolt.data.root_pos_w[:, 0:2] - hole_xy).norm(dim=-1).max()) > 0.05:
                print("  ABORT: bolt left the hole region (ejected or blew up)", flush=True)
                break

    if writer is not None:
        writer.close()
        print("MP4:", args.video, flush=True)

    seated = sc.seated()
    d = depth() * 1e3
    gain = (d - insert_depth0 * 1e3) if insert_depth0 is not None else d
    revs = bolt_turn / (2 * math.pi)
    turned = revs > 0.5
    mm_per_rev = float((gain[turned] / revs[turned]).median()) if turned.any() else float("nan")
    slip = torch.rad2deg(key_turn - bolt_turn)
    print(f"ALLEN-KEY-ALOHA | seated {int(seated.all(dim=1).sum())}/{n} | drove {float(gain.mean()):+.1f}mm "
          f"over {float(revs.mean()):.1f} revs = {mm_per_rev:.2f} mm/rev (pitch {PITCH_MM:.1f}) | "
          f"slip {float(slip.mean()):+.1f}deg | {cycles} cycles, {picks} picks, {drops} drops | tip depth mm: "
          f"min={d.min():+.1f} mean={d.mean():+.1f} max={d.max():+.1f} (seat>= {sc.cfg.seat_depth * 1e3:.0f})", flush=True)
    close_and_exit(env, app)


if __name__ == "__main__":
    main()

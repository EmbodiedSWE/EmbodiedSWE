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
parser.add_argument("--demo-insert", action="store_true", help="end after the verified insert (pick->carry->erect->insert), key left seated — the motion-quality demo")
parser.add_argument("--demo-crank", action="store_true", help="STAGE 2: after the verified insert, the LEFT converts to a passive steady-rest CAGE and the RIGHT cranks the short arm until --demo-revs of bolt rotation")
parser.add_argument("--demo-revs", type=float, default=1.5, help="with --demo-crank: end after this many bolt revolutions")
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
SHORT_GRIP_D = 0.016     # FLIPPED pick: grip the SHORT arm 16mm from its tip — max clearance from
                         # the LYING HANDLE (20mm gap at 30mm put finger plates on top of it: run 127)
HEX_TRIM = 0.0           # long-arm hex phase vs the crank azimuth (mod-60; try pi/6 if pecks never catch)
HANDOFF_DY = -0.06       # handoff spot: bore + (0, HANDOFF_DY) — r~0.30 from BOTH bases (proven band)
HANDOFF_ROOT_Z = 0.150   # crank-tip height at handoff: the long tip PLANTS on the plate — a
                         # free-hanging post is a compliant beam the closing pocket just pushes
                         # away (runs 132-137: transient stall, then the member escapes the
                         # squeeze); planted + held-at-top it is backed at both ends
PINCH_Z = 0.120          # LEFT's post-pinch height: just below the ELBOW (the held end, the
                         # stiffest point on the post) — mid-post pinches deflected the free
                         # beam out of the pocket
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
COMFORT_R = 0.26         # wrist parks this far (horizontal) from its own base: the reach annulus sweet spot
# Per-arm fingertip levers + tool tilts. The RIGHT (near arm, ~150mm from the key spawn) uses a
# SHORT lever at near-vertical tilt — the old contract-gate pick's proven deep regime (link_6
# descended to z~0.062; run 50 showed it cannot hold z 0.17 at 60deg-down). The LEFT (far arm)
# uses a LONG lever at 30deg — reach extension it converges with at 0.1mm (runs 49/50). The old
# 150mm blade-reach estimate was wrong (jaws sit 57-70mm below link_6): both levers CLEAR or
# BETWEEN the jaws, and the run-45 jam was the identity-frame snap, not blade contact.
TILT = {"Right": math.radians(20.0), "Left": math.radians(30.0)}  # mutated by the attempt ladder
# Per-attempt tilt ladders: each missed pinch advances the arm to its next candidate — the
# sim is the reachability oracle (runs 48-51: the wrist's reachable depth depends strongly
# on the frame family, and near-vertical tilts kill the comfort-azimuth freedom entirely).
R_TILTS = tuple(math.radians(a) for a in (30.0, 45.0, 20.0, 6.0))
L_TILTS = tuple(math.radians(a) for a in (30.0, 45.0, 20.0, 12.0))
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
POS_STEP = 0.0009        # max Cartesian step per control tick (m) — ~8x slower than the fast era
ROT_STEP = 0.008         # max rotation step per control tick (rad)
DQ_STEP = 0.03           # per-joint step clamp (rad per tick)
FILT_BETA = 0.02         # folding-style target-filter pole: the servo never sees step targets
GRIP_BETA = 0.025        # grip command ramp (~0.8s open<->close instead of an instant snap)

# Choreography (all waypoints recomputed from live poses; distances in m, angles in rad):
HANDLE_GRIP_D = 0.045    # grip the handle this far out from the elbow (crank radius ~= this)
L_GRIP_D = 0.095         # the LEFT presses HERE (along the handle) during the pinch: staggering
                         # the two stations 50mm apart keeps the claw BODIES (70mm long, above
                         # the tips) from meeting nose-to-nose over the key — run 58's stall
                         # frame showed the two grippers physically tangled, resting on each
                         # other; every 'unreachable' station was arm-vs-arm collision
CARRY_Z = 0.16           # carry/erect height — FLIPPED grip rides 120mm of key + 37mm of grasp
                         # offset above the root: 0.20 pushed the clock tool to z~0.29, past the
                         # little arms' dome (run 130: yaw converged, position frozen 352mm out)
INSERT_HOVER = 0.018     # tip hover above the socket mouth while clocking (kept low: dome budget)
PRESS_DZ = 0.0035        # crank press: command the tip this far below the live socket floor (cam resistance scales with seating force — the torqued crank needs a firm seat)
# (the stiff joint-PD press is FAR harder than the franka's compliant OSC — 1.5 mm of
# commanded interpenetration hammered the staged bolt off its helix capture: it then spun
# crest-nested, +76 deg with zero descent, and the key cammed out on the next stroke)
STROKE_RAD = math.radians(300.0)   # sweep cap per grab — the JOINT-LIMIT GUARD is the real
                                   # limiter; the R's good-grab arc is narrow (~±40deg of the
                                   # nearest azimuth: run 151 cycle 2 pinned the shoulder at
                                   # home-52deg), so each grab must harvest as much sweep as
                                   # the arm allows before regripping near home
CAGE_C = 0.0060          # STAGE 2 steady-rest: L carriage command for the passive cage (hex spins
                         # inside, tip bounded); the closed-bite pocket already wraps the post
CAGE_SLIDE = 0.035       # cage slides this far DOWN the post so the orbiting crank clears the claw
CRANK_GRIP_D = 0.012     # R's stroke grip: 12mm inboard of the crank tip (end-on grab)
POST_HOLD_Z_R = 0.060    # the R's steady-hand station (below the L's pocket: both fit the post)
POST_HOLD_Z_L = 0.095    # the L's steady-hand station
CRANK_W = 0.0025         # crank rate (rad/tick): 120deg in ~840 ticks — watchable
MAX_CRANK_CYCLES = 10    # stroke budget for the demo (release honestly with measured revs)
STROKE_W = 1.0           # commanded crank rate (rad/s)
JOINT_MARGIN = 0.12      # end the stroke when any arm joint is this close to a limit (rad)
REWIND_LIFT = 0.045      # lift between strokes so the open jaws clear the handle sweep
MAX_CYCLES = 40          # fast-ratchet budget: in-place regrips make cycles ~30s
PICK_RETRIES = 3

# Timeouts in CONTROL steps (~50 Hz -> ~5 substeps each at 1/240).
SHOW_END, STAGE_SETTLE = 60, 120
WP_TIMEOUT, CLOSE_STEPS, STROKE_TIMEOUT, SETTLE_STEPS = 1200, 260, 1400, 150  # budgets scaled for the 8x-slowed motion
PECK_PERIOD = 40         # ctrl steps per peck (hover-reposition, then press)
INSERT_TIMEOUT = 2400    # descent + the 19-point lattice + floor drive (8x-slowed handovers eat lead time)
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

    # ----- grasping is done by the REAL articulation jaws (URDF-imported asset) --------------
    # The imported asset's carriage collision TRACKS on GPU (c5b9985): grasp verification is
    # the carriage-pair STALL at ~hex width — the probe's cube-squeeze gate on the real task.

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
    # URDF-imported asset: the finger meshes are merged INTO the carriage links (the old
    # asset's gripper_left hung off carriage_left with an IDENTITY fixed joint, so the
    # carriage origin is the same frame).
    gl = artR.body_names.index("carriage_left")
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
    eye3 = torch.eye(3, device=dev).expand(n, 3, 3)

    def skew(v: torch.Tensor) -> torch.Tensor:
        z = torch.zeros(n, device=dev)
        return torch.stack((
            torch.stack((z, -v[:, 2], v[:, 1]), dim=-1),
            torch.stack((v[:, 2], z, -v[:, 0]), dim=-1),
            torch.stack((-v[:, 1], v[:, 0], z), dim=-1),
        ), dim=1)
    wrist_mask = torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0, 1.0]], device=dev)  # null pull: base joint free, rest toward home (the folding recipe)

    def ik_arm(tp: torch.Tensor, tq: torch.Tensor, rot_w: float = 2.0) -> torch.Tensor:
        """One DLS step of the RIGHT arm's 6 joints toward the tool waypoint (world frame).
        rot_w flips the position/rotation priority: both error channels are rate-capped
        BEFORE weighting (8mm vs 0.06rad), so at the default 2.0 a pending rotation
        outweighs position ~15:1 and the DLS parks on orientation manifolds that never
        reach the target position (runs 48-54). Press phases pass rot_w<1: the sphere
        tips are orientation-agnostic and position is the physical objective."""
        p6, q6 = tool_pose()
        e_p = tp - p6
        pn = e_p.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        e_p = e_p * (pn.clamp(max=POS_STEP) / pn)
        qe = quat_mul(tq, quat_conjugate(q6))
        qe = torch.where(qe[:, :1] >= 0, qe, -qe)
        e_r = axis_angle_from_quat(qe)
        rn = e_r.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        e_r = e_r * (rn.clamp(max=ROT_STEP) / rn)
        e = torch.cat((e_p, rot_w * e_r), dim=-1)
        jac = artR.root_physx_view.get_jacobians()[:, ee_idx - 1, 0:6, :][:, :, arm_ids]  # (n, 6, 6)
        JJt = jac @ jac.transpose(1, 2) + (IK_LAMBDA**2) * eyeM
        dq = (jac.transpose(1, 2) @ torch.linalg.solve(JJt, e.unsqueeze(-1))).squeeze(-1)
        JpJ = jac.transpose(1, 2) @ torch.linalg.solve(JJt, jac)
        home_pull = (artR.data.default_joint_pos[:, arm_ids] - artR.data.joint_pos[:, arm_ids]).clamp(-0.5, 0.5)
        home_pull = home_pull * wrist_mask
        dq = dq + 0.04 * ((eyeM - JpJ) @ home_pull.unsqueeze(-1)).squeeze(-1)
        dq = dq.clamp(-DQ_STEP, DQ_STEP)
        q_tgt = artR.data.joint_pos[:, arm_ids] + dq
        return q_tgt.clamp(lo_lim + 0.02, hi_lim - 0.02)

    def act_of(tp: torch.Tensor, tq: torch.Tensor, grip: float | torch.Tensor, rot_w: float = 2.0,
               raw: bool = False) -> torch.Tensor:
        # raw=True: precision micro-motion (the insert's peck square-wave) — the smoothing
        # filter would attenuate exactly that signal (run 117: pecks smoothed into mush)
        sp, sq = (tp, tq) if raw else _shape_pose(_fRp, tool_pose, tp, tq)
        a = torch.zeros(n, act_dim, device=dev)
        a[:, l_s] = left_cmd
        a[:, r_s.start : r_s.start + 6] = ik_arm(sp, sq, rot_w)
        a[:, r_s.start + 6] = _shape_grip("R", grip)
        return a

    def hold_act(grip: float | torch.Tensor) -> torch.Tensor:
        """Freeze the arm at its measured joints (no IK) — the release/bleed posture."""
        a = torch.zeros(n, act_dim, device=dev)
        a[:, l_s] = left_cmd
        a[:, r_s.start : r_s.start + 6] = artR.data.joint_pos[:, arm_ids]
        a[:, r_s.start + 6] = grip
        return a

    def act_R_hold(gr_R: float | torch.Tensor) -> torch.Tensor:
        """LEFT drives (left_cmd); RIGHT holds its LATCHED steady-hand posture (hold_qR)."""
        a = torch.zeros(n, act_dim, device=dev)
        a[:, l_s] = left_cmd
        a[:, r_s.start : r_s.start + 6] = hold_qR
        a[:, r_s.start + 6] = gr_R
        return a

    def act_L_park(gr: float | torch.Tensor = OPEN_C) -> torch.Tensor:
        """LEFT drives (left_cmd), RIGHT ramps gently toward its home posture."""
        a = torch.zeros(n, act_dim, device=dev)
        a[:, l_s] = left_cmd
        rh = artR.data.joint_pos[:, arm_ids]
        a[:, r_s.start : r_s.start + 6] = rh + (artR.data.default_joint_pos[:, arm_ids] - rh).clamp(-DQ_STEP, DQ_STEP)
        a[:, r_s.start + 6] = gr
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

    def ik_left(tp: torch.Tensor, tq: torch.Tensor, rot_w: float = 2.0) -> torch.Tensor:
        p6, q6 = toolL_pose()
        e_p = tp - p6
        pn = e_p.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        e_p = e_p * (pn.clamp(max=POS_STEP) / pn)
        qe = quat_mul(tq, quat_conjugate(q6))
        qe = torch.where(qe[:, :1] >= 0, qe, -qe)
        e_r = axis_angle_from_quat(qe)
        rn = e_r.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        e_r = e_r * (rn.clamp(max=ROT_STEP) / rn)
        e = torch.cat((e_p, rot_w * e_r), dim=-1)
        jac = artL.root_physx_view.get_jacobians()[:, eeL_idx - 1, 0:6, :][:, :, arm_ids_L]
        JJt = jac @ jac.transpose(1, 2) + (IK_LAMBDA**2) * eyeM
        dq = (jac.transpose(1, 2) @ torch.linalg.solve(JJt, e.unsqueeze(-1))).squeeze(-1)
        JpJ = jac.transpose(1, 2) @ torch.linalg.solve(JJt, jac)
        home_pull = (artL.data.default_joint_pos[:, arm_ids_L] - artL.data.joint_pos[:, arm_ids_L]).clamp(-0.5, 0.5)
        home_pull = home_pull * wrist_mask
        dq = dq + 0.04 * ((eyeM - JpJ) @ home_pull.unsqueeze(-1)).squeeze(-1)
        dq = dq.clamp(-DQ_STEP, DQ_STEP)
        q_tgt = artL.data.joint_pos[:, arm_ids_L] + dq
        return q_tgt.clamp(lo_lim_L + 0.02, hi_lim_L - 0.02)

    def left_to(tp: torch.Tensor, tq: torch.Tensor, grip: float = OPEN_C, rot_w: float = 2.0,
                raw: bool = False) -> None:
        """Aim the LEFT arm's next action at a tool waypoint (one DLS step; call every tick)."""
        sp, sq = (tp, tq) if raw else _shape_pose(_fLp, toolL_pose, tp, tq)
        left_cmd[:, 0:6] = ik_left(sp, sq, rot_w)
        left_cmd[:, 6] = _shape_grip("L", grip)

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
        rel_p_L[:] = quat_apply_inverse(tql, key.data.root_pos_w - tpl)
        rel_q_L[:] = quat_mul(quat_conjugate(tql), key.data.root_quat_w)
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

    def pinch_q(u: torch.Tensor, spin: float, tilt: float) -> torch.Tensor:
        """Reachable fingertip frame: tool axis `tilt` rad from vertical, tilted along the
        horizontal azimuth `u`; spin sets tool-z = +-horizontal (mirrored robots prefer
        mirrored spins). Sphere tips are orientation-agnostic, so ALL free rotational
        DOF are spent on reachability — the handle heading constrains nothing."""
        x = u * math.sin(tilt) - ez * math.cos(tilt)
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        z = spin * torch.linalg.cross(ez.expand_as(x), x)
        z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        y = torch.linalg.cross(z, x)
        return quat_from_matrix(torch.stack((x, y, z), dim=-1))

    def tilt_azimuth(base_xy: torch.Tensor, tip_tgt: torch.Tensor, lever: float, tilt: float, away_xy=None) -> torch.Tensor:
        """Tilt azimuth that parks the wrist at COMFORT_R from the arm's own base. The wrist
        sits lever*sin(tilt) horizontally from the tip along -u; solve the azimuth (law of
        cosines) so that standoff lands the wrist on the comfort annulus — near stations
        tilt sideways, far stations pull straight back. Near-vertical tilts have almost no
        standoff; the solve degrades gracefully to u=fwd."""
        v = tip_tgt[:, :2] - base_xy
        d = v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        fwd = v / d
        s = max(lever * math.sin(tilt), 1e-4)
        cosf = ((d * d + s * s - COMFORT_R * COMFORT_R) / (2.0 * d * s)).clamp(-1.0, 1.0)
        sinf = (1.0 - cosf * cosf).clamp_min(0.0).sqrt()
        perp = torch.stack((-fwd[:, 1], fwd[:, 0]), dim=-1)
        if away_xy is not None:
            flip = torch.sign((perp * away_xy).sum(-1, keepdim=True))
            flip = torch.where(flip == 0, torch.ones_like(flip), flip)
            perp = perp * flip
        u2 = cosf * fwd + sinf * perp
        return torch.cat((u2, torch.zeros(n, 1, device=dev)), dim=-1)

    def joint_cost_L(tq_a: torch.Tensor, tp: torch.Tensor) -> torch.Tensor:
        return (ik_left(tp, tq_a) - artL.data.joint_pos[:, arm_ids_L]).norm(dim=-1)

    def post_frames(pinch_pt: torch.Tensor) -> list:
        """Ranked (u, spin, tilt_rad) candidates for the L post pinch, cheapest virtual-IK
        step first. Run 134 proved the hardcoded 78-deg comfort frame is wrist-infeasible in
        BOTH spins at some stations (spin -1 pins j3, spin +1 pins j4) — so score a ladder of
        azimuth modes x tilts x spins and walk it on retries."""
        u78 = tilt_azimuth(base_L_xy, pinch_pt, tool_to_grip, math.radians(78.0))
        u58 = tilt_azimuth(base_L_xy, pinch_pt, tool_to_grip, math.radians(58.0))
        v = pinch_pt[:, :2] - base_L_xy
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        u_fwd = torch.cat((v, torch.zeros(n, 1, device=dev)), dim=-1)
        scored = []
        for u, s, t in ((u78, -1.0, 78.0), (u78, 1.0, 78.0), (u58, -1.0, 58.0),
                        (u58, 1.0, 58.0), (u_fwd, -1.0, 65.0), (u_fwd, 1.0, 65.0)):
            tr = math.radians(t)
            q = pinch_q(u, s, tr)
            tp = pinch_pt - quat_apply(q, ex1 * tool_to_grip)
            scored.append((float(joint_cost_L(q, tp).max()), u, s, tr))
        scored.sort(key=lambda e: e[0])
        return scored

    def crank_frames(cp: torch.Tensor, u_in: torch.Tensor) -> list:
        """Ranked frames for the R's END-ON crank grab (horizontal member: approach along
        the crank axis, slot across it) — post_frames' mirror, scored on the RIGHT arm."""
        u65 = tilt_azimuth(base_R_xy, cp, tool_to_grip, math.radians(65.0))
        scored = []
        for u, s, t in ((u_in, -1.0, 78.0), (u_in, 1.0, 78.0), (u_in, -1.0, 65.0),
                        (u_in, 1.0, 65.0), (u65, -1.0, 65.0), (u65, 1.0, 65.0)):
            tr = math.radians(t)
            q = pinch_q(u, s, tr)
            tp = cp - quat_apply(q, ex1 * tool_to_grip)
            scored.append((float(joint_cost(q, tp).max()), u, s, tr))
        scored.sort(key=lambda e: e[0])
        return scored

    def crank_frames_L(cp: torch.Tensor, u_in: torch.Tensor) -> list:
        """The LEFT's crank-grab ladder (far-arc azimuths are the L's near arc)."""
        u65 = tilt_azimuth(base_L_xy, cp, tool_to_grip, math.radians(65.0))
        scored = []
        for u, s, t in ((u_in, -1.0, 78.0), (u_in, 1.0, 78.0), (u_in, -1.0, 65.0),
                        (u_in, 1.0, 65.0), (u65, -1.0, 65.0), (u65, 1.0, 65.0)):
            tr = math.radians(t)
            q = pinch_q(u, s, tr)
            tp = cp - quat_apply(q, ex1 * tool_to_grip)
            scored.append((float(joint_cost_L(q, tp).max()), u, s, tr))
        scored.sort(key=lambda e: e[0])
        return scored

    def post_frames_R(pinch_pt: torch.Tensor) -> list:
        """The RIGHT's post-hold ladder (role swap: R becomes the steady hand)."""
        u78 = tilt_azimuth(base_R_xy, pinch_pt, tool_to_grip, math.radians(78.0))
        u58 = tilt_azimuth(base_R_xy, pinch_pt, tool_to_grip, math.radians(58.0))
        v = pinch_pt[:, :2] - base_R_xy
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        u_fwd = torch.cat((v, torch.zeros(n, 1, device=dev)), dim=-1)
        scored = []
        for u, s, t in ((u78, -1.0, 78.0), (u78, 1.0, 78.0), (u58, -1.0, 58.0),
                        (u58, 1.0, 58.0), (u_fwd, -1.0, 65.0), (u_fwd, 1.0, 65.0)):
            tr = math.radians(t)
            q = pinch_q(u, s, tr)
            tp = pinch_pt - quat_apply(q, ex1 * tool_to_grip)
            scored.append((float(joint_cost(q, tp).max()), u, s, tr))
        scored.sort(key=lambda e: e[0])
        return scored

    def crank_perp(arm_name: str) -> torch.Tensor:
        """Distance of the arm's pocket to the CRANK line (root along the short arm)."""
        sd3 = up_axis_of(key.data.root_quat_w)
        sd3 = sd3 / sd3.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        rel = tip_pos(arm_name) - key.data.root_pos_w
        return (rel - (rel * sd3).sum(-1, keepdim=True) * sd3).norm(dim=-1)

    def arm_frame(arm: str, tip_tgt: torch.Tensor, spin: float, away=None) -> tuple[torch.Tensor, torch.Tensor]:
        """The committed frame + wrist target that put `arm`'s fingertip at tip_tgt.
        `away` (n,2): lean the claw body away from this direction's opposite — pass the
        other arm's station so the two claw bodies diverge instead of crossing."""
        base = base_R_xy if arm == "Right" else base_L_xy
        u = tilt_azimuth(base, tip_tgt, tool_to_grip, TILT[arm], away)
        q = pinch_q(u, spin, TILT[arm])
        return q, tip_tgt - quat_apply(q, ex1) * tool_to_grip

    def best_spin_arm(arm: str, tip_tgt: torch.Tensor, away=None) -> float:
        """Score both frame spins by one-step IK cost (the erect-azimuth trick) and commit
        the cheaper one — the preferred spin is config- and azimuth-dependent, so measuring
        beats guessing (run 49: R needed the flip)."""
        costf = joint_cost if arm == "Right" else joint_cost_L
        cs = []
        for s in (+1.0, -1.0):
            q, wp = arm_frame(arm, tip_tgt, s, away)
            cs.append(float(costf(q, wp).max()))
        return +1.0 if cs[0] <= cs[1] else -1.0

    def _ik_tip(art, ids_a, eei, lever, lo, hi, tt: torch.Tensor) -> torch.Tensor:
        """One DLS step of a 3-dof POINT task on the fingertip: all six joints serve the
        tip position; orientation lives entirely in the null-space home bias. (A wrist-
        position servo assumes the lever is orientation-constant — with soft orientation
        the lever swing cancels wrist translation and the tip chases its tail.)"""
        p6 = art.data.body_pos_w[:, eei]
        q6 = art.data.body_quat_w[:, eei]
        lever_w = quat_apply(q6, ex1 * lever)
        e = tt - (p6 + lever_w)
        en = e.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        e = e * (en.clamp(max=POS_STEP) / en)
        jac = art.root_physx_view.get_jacobians()[:, eei - 1, 0:6, :][:, :, ids_a]
        Jt = jac[:, 0:3] - skew(lever_w) @ jac[:, 3:6]
        JJt = Jt @ Jt.transpose(1, 2) + (IK_LAMBDA**2) * eye3
        dq = (Jt.transpose(1, 2) @ torch.linalg.solve(JJt, e.unsqueeze(-1))).squeeze(-1)
        JpJ = Jt.transpose(1, 2) @ torch.linalg.solve(JJt, Jt)
        home_pull = (art.data.default_joint_pos[:, ids_a] - art.data.joint_pos[:, ids_a]).clamp(-0.5, 0.5)
        dq = dq + 0.05 * ((eyeM - JpJ) @ home_pull.unsqueeze(-1)).squeeze(-1)
        dq = dq.clamp(-DQ_STEP, DQ_STEP)
        return (art.data.joint_pos[:, ids_a] + dq).clamp(lo + 0.02, hi - 0.02)

    def act_of_tip(tt: torch.Tensor, grip: float | torch.Tensor) -> torch.Tensor:
        st = _shape_pt(_fRt, lambda: tip_pos("Right"), tt)
        a = torch.zeros(n, act_dim, device=dev)
        a[:, l_s] = left_cmd
        a[:, r_s.start : r_s.start + 6] = _ik_tip(artR, arm_ids, ee_idx, tool_to_grip, lo_lim, hi_lim, st)
        a[:, r_s.start + 6] = _shape_grip("R", grip)
        return a

    def left_to_tip(tt: torch.Tensor, grip: float = OPEN_C) -> None:
        st = _shape_pt(_fLt, lambda: tip_pos("Left"), tt)
        left_cmd[:, 0:6] = _ik_tip(artL, arm_ids_L, eeL_idx, tool_to_grip, lo_lim_L, hi_lim_L, st)
        left_cmd[:, 6] = _shape_grip("L", grip)

    # ----- folding-style command shaping ---------------------------------------------------------
    # The folding suite's smoothness recipe: every Cartesian goal is EXPONENTIALLY eased
    # (filt += (goal - filt) * beta) and the filter re-seeds from the LIVE pose whenever its
    # channel wasn't driven last tick — so phase switches never step the servo input. Four
    # independent channels (R/L x pose-path/tip-path) because the two paths command different
    # semantic points (wrist vs TCP). Grip commands ramp through their own pole.
    _fRp = {"p": None, "q": None, "t": -9}
    _fRt = {"p": None, "t": -9}
    _fLp = {"p": None, "q": None, "t": -9}
    _fLt = {"p": None, "t": -9}
    _fg = {"R": None, "L": None}

    def _nlerp(qf: torch.Tensor, qt: torch.Tensor, alpha: float) -> torch.Tensor:
        qt = torch.where((qf * qt).sum(-1, keepdim=True) < 0.0, -qt, qt)
        q = qf + (qt - qf) * alpha
        return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)

    def _shape_pose(f: dict, live, gp: torch.Tensor, gq: torch.Tensor):
        if f["t"] != step_i - 1 or f["p"] is None:
            lp, lq = live()
            f["p"], f["q"] = lp.clone(), lq.clone()
        f["p"] = f["p"] + (gp - f["p"]) * FILT_BETA
        f["q"] = _nlerp(f["q"], gq, FILT_BETA)
        f["t"] = step_i
        return f["p"], f["q"]

    def _shape_pt(f: dict, live, gp: torch.Tensor) -> torch.Tensor:
        if f["t"] != step_i - 1 or f["p"] is None:
            f["p"] = live().clone()
        f["p"] = f["p"] + (gp - f["p"]) * FILT_BETA
        f["t"] = step_i
        return f["p"]

    def _shape_grip(arm: str, g) -> torch.Tensor:
        gt = torch.as_tensor(g, device=dev, dtype=torch.float32).expand(n).clone() \
            if not torch.is_tensor(g) else g.expand(n).clone()
        if _fg[arm] is None:
            _fg[arm] = gt.clone()
        _fg[arm] = _fg[arm] + (gt - _fg[arm]) * GRIP_BETA
        return _fg[arm]

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

    def tool_for_key_L(kp: torch.Tensor, kq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """LEFT-tool waypoint that puts the L-WELDED key at pose (kp, kq)."""
        tq = quat_mul(kq, quat_conjugate(rel_q_L))
        return kp - quat_apply(tq, rel_p_L), tq

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

    def handle_dir() -> torch.Tensor:
        """World direction of the key's handle — the LONG arm, local +x, shape (n, 3)."""
        return quat_apply(key.data.root_quat_w, ex1)

    # FLIPPED INSERTION: the LONG arm goes into the socket (no bend to bottom out on the mouth,
    # so the full ~7mm recess engages), the SHORT arm becomes the top crank. The exposed
    # ~110mm of vertical hex is the steady-rest post; the crank orbit radius shrinks to ~30mm.
    _tip_local = torch.zeros(1, 3, device=dev)
    _tip_local[0, 0] = HANDLE_LEN
    _tip_local[0, 2] = ARM_LEN

    def key_tip_pos() -> torch.Tensor:
        """World position of the LONG arm's tip — the inserted end (root -> elbow -> tip)."""
        return key.data.root_pos_w + quat_apply(key.data.root_quat_w, _tip_local.expand(n, 3))

    def key_tip_axial() -> torch.Tensor:
        """LONG-tip height above the bolt origin along the bolt axis (m): SOCKET_MOUTH_Z at
        the recess mouth, SOCKET_FLOOR_Z when bottomed on the floor."""
        ub = up_axis_of(bolt.data.root_quat_w)
        return ((key_tip_pos() - bolt.data.root_pos_w) * ub).sum(-1)

    def key_lateral() -> torch.Tensor:
        ub = up_axis_of(bolt.data.root_quat_w)
        rel = key_tip_pos() - bolt.data.root_pos_w
        return (rel - (rel * ub).sum(-1, keepdim=True) * ub).norm(dim=-1)

    def crank_azim() -> torch.Tensor:
        """Azimuth of the SHORT arm pointing OUT from the post (elbow -> short tip): THE 1-dof
        rotation coordinate of the flipped key about the bore axis (yaw_of is ill-defined on a
        90deg-pitched quat)."""
        sa = -up_axis_of(key.data.root_quat_w)
        return torch.atan2(sa[:, 1], sa[:, 0])

    _ey1 = torch.zeros(1, 3, device=dev)
    _ey1[0, 1] = 1.0
    _q_flip = quat_from_angle_axis(torch.full((1,), math.pi / 2, device=dev), _ey1).expand(n, 4)

    def kq_flip(psi: torch.Tensor) -> torch.Tensor:
        """Key target quat: LONG arm vertical DOWN, short-arm crank at azimuth psi."""
        return quat_mul(quat_from_angle_axis(psi + math.pi, ez), _q_flip)

    def root_for_tip(tip_tgt: torch.Tensor, kq: torch.Tensor) -> torch.Tensor:
        """Key ROOT target that puts the LONG tip at tip_tgt under orientation kq."""
        return tip_tgt - quat_apply(kq, _tip_local.expand(n, 3))

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
        # Guard only the joints the crank WINDS toward hard stops (waist j0 + wrist-roll
        # j5, the orbit axes). j1-j4 oscillate near their bounds during ordinary orbit
        # tracking — run 95: full-joint guarding ended strokes at 33 of 120 deg.
        crank_j = [0, 5]
        q = artR.data.joint_pos[:, arm_ids][:, crank_j]
        qd = artR.data.joint_vel[:, arm_ids][:, crank_j]
        lo_hit = (q - lo_lim[:, crank_j] < JOINT_MARGIN) & (qd < -0.02)
        hi_hit = (hi_lim[:, crank_j] - q < JOINT_MARGIN) & (qd > 0.02)
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
    prev_key_yaw = crank_azim()
    insert_depth0 = None
    cycles, drops, picks, regrip_tries, aim_tries, repress_tries, hold_tries, rr_tries = 0, 0, 0, 0, 0, 0, 0, 0
    grip_freeze = torch.zeros(n, device=dev)   # carriage target latched at release (bleed the squeeze)
    seat_ok = torch.zeros(n, device=dev)       # consecutive ticks the inserted key has read settled
    psi_star = torch.zeros(n, device=dev)      # latched hex-aligned key yaw for clock/insert
    stroke_psi = torch.zeros(n, device=dev)    # commanded crank sweep so far (rad, >= 0)
    stroke_y0 = torch.zeros(n, device=dev)     # key yaw at stroke start
    grip_pt = torch.zeros(n, 3, device=dev)
    pinch_n = torch.zeros(n, 3, device=dev)  # unit, horizontal: grip -> the RIGHT arm's side
    spin_R, spin_L = 1.0, -1.0    # per-arm frame spins, committed at pinch_high entry
    hold_qR = torch.zeros(n, 6, device=dev)   # RIGID hold targets, captured at phase entry —
    hold_qL = torch.zeros(n, 6, device=dev)   # re-targeting measured joints makes a pushover arm
    vice_sL = torch.full((n, 1), 0.05, device=dev)  # where along the handle each vice press lands
    vice_sR = torch.full((n, 1), 0.05, device=dev)
    vice_sgnL = torch.ones(n, 1, device=dev)  # the side the vice hand held from
    hold_perp_ref, repress_perp_ref = 1.0, 1.0  # perp at the extension checkpoint
    right_hold_s = torch.full((n, 1), HANDLE_GRIP_D, device=dev)  # where the RIGHT's weld grips the handle
    vice_gripL: float = CLOSE_C   # the LEFT's carriage hold while it owns the key (stall-based)
    unwind_up = torch.zeros(n, 3, device=dev)
    vspin_R, vspin_L = 1.0, -1.0  # ditto for the vice-hand presses
    # Carriage command while HOLDING the welded key. Pressing the full CLOSE_C past the stall
    # turns the hook undersides into a downward wedge that pins the key to the table (the lift
    # then can't raise the arm at all) — so on weld, relax to the measured stall + 0.5 mm.
    grip_c: float | torch.Tensor = CLOSE_C
    wp_p = torch.zeros(n, 3, device=dev)
    wp_q = torch.zeros(n, 4, device=dev)
    wpL_p = torch.zeros(n, 3, device=dev)
    wpL0_p = torch.zeros(n, 3, device=dev)
    wpL_q = torch.zeros(n, 4, device=dev)
    rel_p_L = torch.zeros(n, 3, device=dev)   # L grasp frame, captured at weld_on_L
    rel_q_L = torch.zeros(n, 4, device=dev)
    pinch_lock = torch.zeros(n, 3, device=dev)  # post-pinch target, LATCHED at phase entry
    uL_lock = torch.zeros(n, 3, device=dev)     # latched approach azimuth
    pf_spin, pf_tilt = -1.0, math.radians(78.0)  # latched pinch frame (spin, tilt)
    crank_lock = torch.zeros(n, 3, device=dev)  # STAGE 2: latched crank grab point
    uR_lock = torch.zeros(n, 3, device=dev)
    cf_spin, cf_tilt = 1.0, math.radians(65.0)
    wpR0_p = torch.zeros(n, 3, device=dev)      # R release/dwell pose
    wpR0_q = torch.zeros(n, 4, device=dev)
    cage_p0 = torch.zeros(n, 3, device=dev)     # L cage pose at conversion
    cage_q0 = torch.zeros(n, 4, device=dev)
    pairR_hold = torch.zeros(n, device=dev)     # stability sample for the crank-grab gate
    crank_tries, crank_cycles, crank_bounce = 0, 0, 0
    grip_hold_R: float | torch.Tensor = CLOSE_C  # the R's steady-hand bite command
    holder = "L"  # which arm is the steady hand (admire/retreat must not yank it)
    crank_sign, crank_dir_checked = 1.0, False
    grip_c_L: float | torch.Tensor = CLOSE_C  # L's hold at its measured post stall

    # ----- jaw plumbing + boot diagnostics --------------------------------------------------
    jaw_ids_R = [artR.joint_names.index(j) for j in ("left_carriage_joint", "right_carriage_joint")]
    jaw_ids_L = [artL.joint_names.index(j) for j in ("left_carriage_joint", "right_carriage_joint")]
    _lo = artR.data.joint_pos_limits[0, arm_ids, 0].tolist()
    _hi = artR.data.joint_pos_limits[0, arm_ids, 1].tolist()
    print("[jaws] grasping with the articulation claws | R arm limits lo "
          + " ".join(f"{v:+.2f}" for v in _lo) + " | hi " + " ".join(f"{v:+.2f}" for v in _hi), flush=True)

    def jaw_pair(art, jids) -> torch.Tensor:
        """Carriage-pair opening (m): the physical stall reads ~hex width; an empty close
        reaches ~2x the closed command."""
        return art.data.joint_pos[:, jids].sum(-1)

    def tip_pos(arm_name: str) -> torch.Tensor:
        """The TCP = the claw grip pocket, FK from link_6 (the spheres are gone)."""
        art = artL if arm_name == "Left" else artR
        eei = eeL_idx if arm_name == "Left" else ee_idx
        return art.data.body_pos_w[:, eei] + quat_apply(art.data.body_quat_w[:, eei], ex1 * tool_to_grip)

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
            if t_in == STAGE_SETTLE // 2 + 20:
                # Normalize the KEY's settle: root to (0.34, 0.10) with the SHORT arm heading
                # +x — the member the jaws capture (the handle sticks off along +-y),
                # so the grip point lands ~(0.50, 0.10) — equidistant from both arm bases. A
                # random settle heading swings the grip +-110mm along the handle and strands
                # one arm out of range (run 59: grip at x=0.578 = 0.37m from the right base).
                # Yaw about z at rest is contact-invariant; same baked-preset philosophy as
                # the scene-init presets (5b8f5a8).
                sd0 = up_axis_of(key.data.root_quat_w)  # SHORT arm (root -> elbow), flat = horizontal
                psi0 = torch.atan2(sd0[:, 1], sd0[:, 0])
                stk = torch.zeros(n, 13, device=dev)
                stk[:, 0] = 0.34
                stk[:, 1] = 0.10
                stk[:, 2] = key.data.root_pos_w[:, 2]
                stk[:, 3:7] = quat_mul(quat_from_angle_axis(-psi0, ez), key.data.root_quat_w)
                key.write_root_state_to_sim(stk, ids)
            if t_in >= STAGE_SETTLE:
                print(f"  staged: bolt tip depth {float(depth().mean()) * 1e3:+.2f}mm (thread-captured)", flush=True)
                phase, marker = "pinch_high", i
                picks += 1
        elif phase == "pinch_high":  # the RIGHT stages its fingertip high above the grip point.
            # SINGLE-ARM REAL-CONTACT PICK: the table is the opposing surface — the tip presses
            # DOWN on the handle and the stall against it is the physical grasp verification.
            # (Runs 45-64: both arms at table level at r>=0.28 is the dome edge — some wrist or
            # elbow joint saturates in every frame family. The bimanual show is the VICE CRANK
            # LOOP at the platform, where the key is held at z~0.07 and the left converges
            # effortlessly.) The LEFT parks clear until the vice stage.
            sd = up_axis_of(key.data.root_quat_w)  # SHORT-ARM PICK (flipped-insertion scheme)
            sd2 = sd.clone(); sd2[:, 2] = 0.0
            sd2 = sd2 / sd2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            grip_pt[:] = key.data.root_pos_w + sd2 * SHORT_GRIP_D
            nv = torch.stack((-sd2[:, 1], sd2[:, 0], torch.zeros(n, device=dev)), dim=-1)
            sgn = torch.sign(((base_R_xy - grip_pt[:, :2]) * nv[:, :2]).sum(-1)).unsqueeze(-1)
            sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
            pinch_n[:] = nv * sgn
            tipR = grip_pt + ez * 0.10
            if t_in == 1:  # commit the press frame for this attempt (no flapping)
                rung = max(picks - 1, 0)  # picks counts attempts STARTED (staging pre-increments)
                TILT["Right"] = R_TILTS[min(rung, len(R_TILTS) - 1)]
                spin_R = best_spin_arm("Right", tipR)
                print(f"    [frames] attempt {picks}: R tilt {math.degrees(TILT['Right']):.0f}deg"
                      f" spin {spin_R:+.0f} (press-down pick)", flush=True)
            tR_err = tipR - tip_pos("Right")
            left_park()
            act = act_of_tip(tipR, OPEN_C)
            if t_in == WP_TIMEOUT:
                print(f"    [pinch_high timeout] tip err R {float(tR_err.norm(dim=-1).max()) * 1e3:.1f}mm", flush=True)
            if bool((tR_err.norm(dim=-1) < 0.012).all()) or t_in >= WP_TIMEOUT:
                phase, marker = "pinch_in", i
        elif phase == "pinch_in":  # descend to a low hover directly above the handle
            sd = up_axis_of(key.data.root_quat_w)
            sd2 = sd.clone(); sd2[:, 2] = 0.0
            sd2 = sd2 / sd2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            grip_pt[:] = key.data.root_pos_w + sd2 * SHORT_GRIP_D
            tipR = grip_pt + ez * 0.05
            tR_err = tipR - tip_pos("Right")
            left_park()
            act = act_of_tip(tipR, OPEN_C)
            if t_in == WP_TIMEOUT:
                print(f"    [pinch_in timeout] tip err R {float(tR_err.norm(dim=-1).max()) * 1e3:.1f}mm", flush=True)
                mR = torch.minimum(artR.data.joint_pos[:, arm_ids] - lo_lim, hi_lim - artR.data.joint_pos[:, arm_ids])
                print("    [stall] R margins " + " ".join(f"{float(v):+.2f}" for v in mR[0]), flush=True)
                print(f"    [stall] R tip {[round(v, 3) for v in tip_pos('Right')[0].tolist()]}"
                      f" -> {[round(v, 3) for v in tipR[0].tolist()]}", flush=True)
                if cam is not None:
                    import imageio.v2 as _iio
                    _iio.imwrite(str(Path(args.video).parent / f"stall_a{picks}.png"),
                                 cam.data.output["rgb"][0].cpu().numpy())
                    print("    [stall] frame dumped", flush=True)
            if bool((tR_err.norm(dim=-1) < 0.006).all()) or t_in >= WP_TIMEOUT:
                phase, marker = "pinch_close", i
        elif phase == "pinch_close":  # REAL JAW PINCH: straddle the handle, close the claws,
            # and verify the grasp by the carriage-pair STALL at ~hex width — an empty close
            # reaches ~8mm pair; the 12.6mm hex stops it in the stall window. Key drift and
            # centering complete the gate; then the weld (the benchmark's grasp contract).
            if t_in == 1:
                close_key0 = key.data.root_pos_w[:, :2].clone()
            sd = up_axis_of(key.data.root_quat_w)
            sd2 = sd.clone(); sd2[:, 2] = 0.0
            sd2 = sd2 / sd2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            grip_pt[:] = key.data.root_pos_w + sd2 * SHORT_GRIP_D
            tcp_tgt = grip_pt.clone()
            tcp_tgt[:, 2] = 0.012  # pocket brackets the handle's upper half; claw tips clear the table
            grip_cmd = OPEN_C if t_in < 240 else 0.004
            left_park()
            act = act_of_tip(tcp_tgt, grip_cmd)
            if t_in % 20 == 0:
                pr = jaw_pair(artR, jaw_ids_R)
                kp0 = key.data.root_pos_w
                tz0 = tip_pos("Right")
                print(f"    [jaw t{t_in:3d}] pair {float(pr[0]) * 1e3:5.1f}mm"
                      f" | key ({float(kp0[0, 0]):+.3f},{float(kp0[0, 1]):+.3f},{float(kp0[0, 2]):+.4f})"
                      f" | tcp ({float(tz0[0, 0]):+.3f},{float(tz0[0, 1]):+.3f},{float(tz0[0, 2]):+.4f})"
                      f" -> ({float(tcp_tgt[0, 0]):+.3f},{float(tcp_tgt[0, 1]):+.3f})", flush=True)
            if t_in >= 240 + CLOSE_STEPS:
                pr = jaw_pair(artR, jaw_ids_R)
                tzz = tip_pos("Right")
                # GRASP CONTRACT for the free-end grip: the pocket sits ON the member at the
                # intended station, in 3D. The claw legitimately SCOOPS the free short arm
                # ~2cm up while squeezing into the pocket (run 128: pair 15.0 with the key
                # hoisted to z=23mm — the old table-flatness gate false-failed a physically
                # perfect capture; the handle end just pivots on the table).
                sd_live = up_axis_of(key.data.root_quat_w)
                grip_now = key.data.root_pos_w + sd_live * SHORT_GRIP_D
                near = (tzz - grip_now).norm(dim=-1) < 0.020
                ok = (pr > 0.0105) & (pr < 0.0165) & near
                if bool(ok.all()):
                    grip_c = float(pr[0]) * 0.5 + 0.0005  # hold at the measured stall + margin
                    weld_on()
                    phase, marker = "pinch_off", i
                elif picks < PICK_RETRIES:
                    print(f"  pinch {picks} missed (pair {[round(float(v) * 1e3, 1) for v in pr]}mm), retrying", flush=True)
                    phase, marker = "pinch_high", i
                    picks += 1
                else:
                    print("  ABORT: jaw pinch failed", flush=True)
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
            up_now = -handle_dir()[:, 2]  # 1.0 when the LONG arm hangs straight DOWN (flipped)
            if t_in == 1:
                psi0 = crank_azim()
                # score candidates at BOTH poses this azimuth must serve: the erect spot AND the
                # subsequent clock hover over the bolt (an azimuth that erects but leaves the
                # wrist unable to reach the bolt strands the insert 60+ mm short at joint limits)
                kp_hand = torch.zeros(n, 3, device=dev)
                kp_hand[:, 0] = hole_xy[:, 0]
                kp_hand[:, 1] = hole_xy[:, 1] + HANDOFF_DY
                kp_hand[:, 2] = HANDOFF_ROOT_Z
                azim_L = torch.atan2(base_L_xy[:, 1] - kp_hand[:, 1], base_L_xy[:, 0] - kp_hand[:, 0])
                cands, costs = [], []
                for dpsi in (0.0, math.pi / 2, -math.pi / 2, math.pi):
                    cand = _wrap(psi0 + dpsi)
                    kq_c = kq_flip(cand)
                    tp_c, tq_c = tool_for_key(kp, kq_c)
                    tp_k, tq_k = tool_for_key(kp_hand, kq_c)
                    # keep the short-arm crank OUT of the LEFT's post-pinch corridor
                    block = torch.cos(cand - azim_L).clamp_min(0.0) * 2.0
                    cands.append(cand)
                    costs.append(joint_cost(tq_c, tp_c) + joint_cost(tq_k, tp_k) + block)
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
            kq = kq_flip(psi_star)
            tp, tq = tool_for_key(kp, kq)
            act = act_of(tp, tq, grip_c)
            upright = (up_now > 0.995)
            if bool(upright.all()):
                phase, marker = "handoff_carry", i
            elif t_in >= 4 * WP_TIMEOUT:
                print(f"  ABORT: erection stalled (key up_z {float(up_now.min()):+.2f})", flush=True)
                phase, marker = "retreat", i
        elif phase == "re_home":  # config reset: lift the welded key clear, then drive the
            # RIGHT arm home in joint space (the key swings along on the weld) — plateau
            # families only change the TARGET yaw; a tangled global config needs a restart
            # (run 87: lat walked 65->163mm through four families)
            if t_in < 50:
                kp = key.data.root_pos_w.clone()
                kp[:, 2] = kp[:, 2] + 0.10
                tp, tq = tool_for_key(kp, key.data.root_quat_w)
                act = act_of(tp, tq, grip_c)
            else:
                a = torch.zeros(n, act_dim, device=dev)
                a[:, l_s] = left_cmd
                a[:, r_s.start : r_s.start + 6] = artR.data.default_joint_pos[:, arm_ids]
                a[:, r_s.start + 6] = grip_c
                act = a
            if t_in >= 300:
                phase, marker = "carry", i  # re-approach from clear air: carry -> erect -> clock
        elif phase == "handoff_carry":  # R carries the flip-erect key to the HANDOFF spot.
            # WHY A HANDOFF AT ALL: the R's crank grip rides 157mm above the long tip — putting
            # the tip at the mouth demands tool z ~0.20+ at the bore radius, past the little
            # arm's dome (runs 130/131: yaw converged, position frozen 350-500mm out). The LEFT
            # takes the POST low instead (tool z ~0.13) and inserts from there; its post grip
            # then IS the Stage-2 steady-rest.
            kp = torch.zeros(n, 3, device=dev)
            kp[:, 0] = hole_xy[:, 0]
            kp[:, 1] = hole_xy[:, 1] + HANDOFF_DY
            xy_far = (kp[:, 0:2] - key.data.root_pos_w[:, 0:2]).norm(dim=-1) > 0.015
            # carry HIGH, plant only at the station: a planted-tip transit drags the tip
            # across the plate and through the BOLT (run 145: bolt flat at depth -7mm,
            # tilt 73deg, after the retry carries plowed it)
            kp[:, 2] = HANDOFF_ROOT_Z + torch.where(xy_far, torch.full_like(kp[:, 2], 0.030),
                                                    torch.zeros_like(kp[:, 2]))
            kq = kq_flip(psi_star)
            tp, tq = tool_for_key(kp, kq)
            elbow = key.data.root_pos_w + up_axis_of(key.data.root_quat_w) * ARM_LEN
            hdn = handle_dir()
            s_p = (elbow[:, 2:3] - PINCH_Z).clamp(0.02, 0.11)
            pinch_pt = elbow + hdn * s_p
            if t_in == 1:
                wpL0_p[:] = toolL_pose()[0]  # dwell spot for the open-in-place discipline
            _c, uL, sL, tL = post_frames(pinch_pt)[min(hold_tries, 5)]
            qL = pinch_q(uL, sL, tL)
            # after a MISSED pinch the L jaws are still around the post: moving while closing
            # drags the welded train and can knock the bolt out (run 143) — open IN PLACE
            # for 150 ticks, then go to the staging hover
            if t_in < 150:
                left_to(wpL0_p, toolL_pose()[1], OPEN_C, rot_w=1.2)
            else:
                left_to(pinch_pt + uL * 0.07 - quat_apply(qL, ex1 * tool_to_grip), qL, OPEN_C, rot_w=1.2)
            hold_qR[:] = artR.data.joint_pos[:, arm_ids]  # continuously refreshed while R OWNS the move
            act = act_of(tp, tq, grip_c)
            if bool(((kp - key.data.root_pos_w).norm(dim=-1) < 0.012).all()) or t_in >= 2 * WP_TIMEOUT:
                phase, marker = "post_pinch", i
        elif phase == "post_pinch":  # LEFT pinches the vertical POST at mid-height (lateral
            # member-pinch frames, proven in the vice era); verified by the L carriage stall
            # + the pocket sitting ON the post line.
            # LATCH EVERYTHING AT ENTRY: live-derived targets + per-tick frame re-scoring
            # created a chase loop (run 135: the open claw nudged the key, the target
            # followed, the frame flipped mid-flight, and the pair towed the key 100mm) —
            # and hold_act's follow-the-measured-joints made the R a pushover (the vice-era
            # hold_qR law).
            elbow = key.data.root_pos_w + up_axis_of(key.data.root_quat_w) * ARM_LEN
            hdn = handle_dir()
            s_p = (elbow[:, 2:3] - PINCH_Z).clamp(0.02, 0.11)
            pinch_live = elbow + hdn * s_p
            if t_in == 1:
                pinch_lock[:] = pinch_live
                hold_qR[:] = artR.data.joint_pos[:, arm_ids]
                _c, _u, _s, _t = post_frames(pinch_lock)[min(hold_tries, 5)]
                uL_lock[:] = _u
                pf_spin, pf_tilt = _s, _t
                print(f"    [post] frame locked: cost {_c:.2f} spin {pf_spin:+.0f} tilt "
                      f"{math.degrees(pf_tilt):.0f}deg", flush=True)
            qL = pinch_q(uL_lock, pf_spin, pf_tilt)
            close_now = t_in > 500
            if t_in == 480:
                # ONE re-latch just before the close: the arrival ringing nudges the hanging
                # key a couple of cm; center the close on the LIVE post (run 136: fingers
                # straddling the post, pocket 20mm off the stale latch)
                pinch_lock[:] = pinch_live
            # during the close, press 6mm INTO the post (the R pick's press-down analogue)
            vt = pinch_lock - uL_lock * 0.006 if close_now else pinch_lock
            tpL = vt - quat_apply(qL, ex1 * tool_to_grip)
            left_to(tpL, qL, 0.004 if close_now else OPEN_C, rot_w=1.2)
            a = torch.zeros(n, act_dim, device=dev)
            a[:, l_s] = left_cmd
            a[:, r_s.start : r_s.start + 6] = hold_qR
            a[:, r_s.start + 6] = grip_c
            act = a
            drift_now = (pinch_live - pinch_lock).norm(dim=-1)
            if bool((drift_now > 0.030).any()) and not close_now:
                print(f"    [post] key drifted {float(drift_now.max()) * 1e3:.0f}mm from the latch — re-staging", flush=True)
                phase, marker = "handoff_carry", i
            if t_in % 100 == 0:
                prL = jaw_pair(artL, jaw_ids_L)
                tl = tip_pos("Left")
                mL = torch.minimum(artL.data.joint_pos[:, arm_ids_L] - lo_lim_L,
                                   hi_lim_L - artL.data.joint_pos[:, arm_ids_L])
                print(f"    [post t{t_in:4d}] L pair {float(prL[0]) * 1e3:5.1f}mm | perp "
                      f"{float(tip_perp('Left').max()) * 1e3:5.1f}mm | Ltip "
                      f"({float(tl[0, 0]):+.3f},{float(tl[0, 1]):+.3f},{float(tl[0, 2]):+.3f}) -> "
                      f"({float(pinch_lock[0, 0]):+.3f},{float(pinch_lock[0, 1]):+.3f},{float(pinch_lock[0, 2]):+.3f})"
                      f" | Lmargin " + " ".join(f"{float(v):+.2f}" for v in mL[0]), flush=True)
            if cam is not None and t_in in (400, 800, 1150):
                import imageio.v2 as _iio
                _iio.imwrite(str(Path(args.video).parent / f"pinch_{picks}_{t_in}.png"),
                             cam.data.output["rgb"][0].cpu().numpy())
                print(f"    [post] frame dumped at t{t_in}", flush=True)
            if t_in == 800:
                pairL_800 = jaw_pair(artL, jaw_ids_L).clone()
            if t_in >= 900:
                prL = jaw_pair(artL, jaw_ids_L)
                perp_now = tip_perp("Left")
                # TWO REAL CAPTURE MODES (run 138): pocket-seated hex stalls ~15mm with the
                # post ON the pocket line; a FLARE catch stalls ~28mm with the post held
                # 12-18mm forward of the pocket — a stable, weldable grip the pocket-only
                # window kept rejecting. Accept either, and demand the stall held steady
                # over the last 100 ticks (grazes slip within ~50).
                stable = (prL - pairL_800).abs() < 0.002
                # HANDOFF CONTRACT (superset): the physical squeeze is NOT the load-bearing
                # verification here — the weld is, and end-to-end honesty rests on the insert
                # gate. The lateral grab keeps finding new stable contact modes (pocket 15mm,
                # flare 28mm, post+corner span 46mm — runs 132-139); accept ANY stable,
                # substantially-closed configuration with the pocket near the post line.
                # CLOSED BITE ONLY: wide-span grabs (28-50mm pair) weld the tool at a lever
                # where the clock's lateral translate is kinematically DEGENERATE — the L
                # descends but cannot move sideways (run 142: lat frozen 21mm through the
                # whole insert, margins healthy). The 8-16mm closed bite (runs 140/141) is
                # the mode whose clock works; retry until the claw gets it (pre-commit, the
                # R still holds the key).
                okL = (prL < 0.020) & (perp_now < 0.018) & stable
                if bool(okL.all()):
                    grip_c_L = float(prL[0]) * 0.5 + 0.0005
                    weld_on_L()
                    print(f"  POST HANDOFF: L stall {float(prL[0]) * 1e3:.1f}mm — the left owns the key", flush=True)
                    phase, marker = "handoff", i
                elif hold_tries < 4:
                    hold_tries += 1
                    print(f"  post pinch missed (pair {float(prL[0]) * 1e3:.1f}mm, perp "
                          f"{float(tip_perp('Left').max()) * 1e3:.1f}mm), retrying frame {hold_tries + 1}", flush=True)
                    phase, marker = "handoff_carry", i
                else:
                    print("  ABORT: post pinch failed", flush=True)
                    phase, marker = "retreat", i
        elif phase == "handoff":  # weld transfer: L latched at the true pose; R releases the
            # crank and backs toward home
            if t_in == 1:
                weld_off()
            left_hold()
            act = act_L_park(_shape_grip("R", OPEN_C))
            if t_in >= 240:
                phase, marker = "l_clock", i
        elif phase == "l_clock":  # LEFT translates the LONG tip over the bore, hex-clocked
            if t_in == 1:
                ky = crank_azim()
                dpsi = _wrap((yaw_of(bolt.data.root_quat_w) + HEX_TRIM - ky) % (math.pi / 3.0))
                dpsi = torch.where(dpsi > math.pi / 6.0, dpsi - math.pi / 3.0, dpsi)
                psi_star[:] = ky + dpsi
            kq = kq_flip(psi_star)
            if t_in == 1:
                wpL0_p[:] = key.data.root_pos_w
                wpL0_p[:, 2] = wpL0_p[:, 2] + 0.055
            if t_in < 220:
                # UN-PLANT PRE-LIFT (the R clock's un-wedge, 8x-scaled): rise straight OFF the
                # plate before any lateral travel — the filtered lift+translate blend cuts the
                # corner and drags the low tip across the bolt head (run 144: bolt knocked out)
                tpL2, tqL2 = tool_for_key_L(wpL0_p, kq)
            else:
                tip_tgt = torch.zeros(n, 3, device=dev)
                tip_tgt[:, 0:2] = bolt.data.root_pos_w[:, 0:2]
                tip_tgt[:, 2] = bolt.data.root_pos_w[:, 2] + SOCKET_MOUTH_Z + INSERT_HOVER
                lat_now = key_lateral()
                tip_tgt[:, 2] = tip_tgt[:, 2] + torch.where(lat_now > 0.020,
                                                            torch.full_like(lat_now, 0.020), torch.zeros_like(lat_now))
                tpL2, tqL2 = tool_for_key_L(root_for_tip(tip_tgt, kq), kq)
            left_to(tpL2, tqL2, grip_c_L, rot_w=1.2)  # rot-priority parked the L 20mm short (run 140)
            act = act_L_park()
            if t_in % 150 == 0:
                mL = torch.minimum(artL.data.joint_pos[:, arm_ids_L] - lo_lim_L,
                                   hi_lim_L - artL.data.joint_pos[:, arm_ids_L])
                tlp = toolL_pose()[0]
                print(f"    [l_clock t{t_in:4d}] lat {float(key_lateral().max()) * 1e3:5.1f}mm | Ltool "
                      f"({float(tlp[0, 0]):+.3f},{float(tlp[0, 1]):+.3f},{float(tlp[0, 2]):+.3f}) -> "
                      f"({float(tpL2[0, 0]):+.3f},{float(tpL2[0, 1]):+.3f},{float(tpL2[0, 2]):+.3f})"
                      f" | Lmargin " + " ".join(f"{float(v):+.2f}" for v in mL[0]), flush=True)
            centred = (key_lateral() < 0.001) & (_wrap(crank_azim() - psi_star).abs() < math.radians(2.5))
            if not bool(bolt_ok(say=True).all()):
                print("  ABORT: bolt left its nest before insertion", flush=True)
                phase, marker = "retreat", i
            elif bool(centred.all()) or t_in >= 3 * WP_TIMEOUT:
                phase, marker = "l_insert", i
        elif phase == "l_insert":  # hover-peck probe, LEFT-driven (same lattice/breaker/dither
            # contract as the R-era insert; the L post grip keeps the tool at z~0.13)
            if t_in == 1:
                seat_ok.zero_()
                ins_last_ax = 9.9
                ins_stuck = 0
                ins_unwedge_until = 0
            if t_in % PECK_PERIOD == 1:
                ky = crank_azim()
                dpsi = _wrap((yaw_of(bolt.data.root_quat_w) + HEX_TRIM - ky) % (math.pi / 3.0))
                dpsi = torch.where(dpsi > math.pi / 6.0, dpsi - math.pi / 3.0, dpsi)
                psi_star[:] = ky + dpsi
            kq = kq_flip(psi_star)
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
            tip_in = (key_tip_axial() < SOCKET_MOUTH_Z - 0.0005) & (key_lateral() < 0.003)
            tip_tgt = torch.zeros(n, 3, device=dev)
            tip_tgt[:, 0] = bolt.data.root_pos_w[:, 0] + torch.where(tip_in, torch.zeros_like(depth()), torch.full_like(depth(), ox))
            tip_tgt[:, 1] = bolt.data.root_pos_w[:, 1] + torch.where(tip_in, torch.zeros_like(depth()), torch.full_like(depth(), oy))
            tip_tgt[:, 2] = torch.where(tip_in, floor_z, z_cmd.clamp(min=floor_z))
            _axn = float(key_tip_axial().mean())
            if bool(tip_in.all()) and abs(ins_last_ax - _axn) < 2e-4:
                ins_stuck += 1
            else:
                ins_stuck = 0
            ins_last_ax = _axn
            if bool(tip_in.all()) and 20 <= ins_stuck:
                _ph = (t_in // 10) % 4
                tip_tgt[:, 0] = tip_tgt[:, 0] + (0.0005, 0.0, -0.0005, 0.0)[_ph]
                tip_tgt[:, 1] = tip_tgt[:, 1] + (0.0, 0.0005, 0.0, -0.0005)[_ph]
            if ins_stuck > 100:
                ins_stuck = 0
                ins_unwedge_until = t_in + 30
                print(f"    [l_insert] wedged catch at {_axn * 1e3:+.2f}mm — lifting to re-peck", flush=True)
            if t_in < ins_unwedge_until:
                tip_tgt[:, 2] = mouth_z + 0.002
            tpL2, tqL2 = tool_for_key_L(root_for_tip(tip_tgt, kq), kq)
            left_to(tpL2, tqL2, grip_c_L, rot_w=0.6, raw=True)
            act = act_L_park()
            if t_in % PECK_PERIOD == PECK_PERIOD - 1:
                dp = _wrap((yaw_of(bolt.data.root_quat_w) + HEX_TRIM - crank_azim()) % (math.pi / 3.0))
                dp = torch.where(dp > math.pi / 6.0, dp - math.pi / 3.0, dp)
                over = key_tip_axial() - SOCKET_MOUTH_Z
                print(f"    [l probe {probe:2d}] lat {float(key_lateral().max()) * 1e3:4.2f}mm | dpsi "
                      f"{float(torch.rad2deg(dp.abs().max())):4.1f}deg | tip over mouth {float(over.min()) * 1e3:+5.2f}mm", flush=True)
            settled = (key_tip_axial() < SOCKET_FLOOR_Z + 0.0025) & (key_tip_axial() > SOCKET_FLOOR_Z - 0.003) \
                & (key_lateral() < 0.0015) & (-handle_dir()[:, 2] > 0.99) & bolt_ok()
            seat_ok[:] = torch.where(settled, seat_ok + 1, torch.zeros_like(seat_ok))
            if not bool(bolt_ok(say=True).all()):
                print("  ABORT: bolt knocked out of its nest during insertion", flush=True)
                phase, marker = "retreat", i
            elif bool((seat_ok >= 15).all()):
                aim_tries = 0
                if insert_depth0 is None:
                    insert_depth0 = depth().clone()
                cycles += 1
                if cycles == 1:
                    bolt_turn.zero_()
                    key_turn.zero_()
                    prev_bolt_yaw = yaw_of(bolt.data.root_quat_w)
                    prev_key_yaw = crank_azim()
                if args.demo_crank:
                    print("  insert verified — LEFT converts to the steady-rest cage, RIGHT cranks", flush=True)
                    phase, marker = "cage_convert", i
                else:
                    print("  [demo] insert verified — the left holds the seated key (steady-rest preview)", flush=True)
                    phase, marker = "admire", i
            elif t_in >= INSERT_TIMEOUT:
                aim_tries += 1
                if aim_tries >= 3:
                    drops += 1
                    print("  DROP: l_insert never landed after repeated re-clocks", flush=True)
                    phase, marker = "retreat", i
                else:
                    print("  l_insert timed out, re-clocking", flush=True)
                    phase, marker = "l_clock", i
        elif phase == "clock":  # translate to the hover over the BORE axis, hex-clocked to the bolt
            # (nearest mod-60 representative to the key's current yaw — minimal rotation)
            if t_in == 1:
                ky = crank_azim()
                dpsi = _wrap((yaw_of(bolt.data.root_quat_w) + HEX_TRIM - ky) % (math.pi / 3.0))
                dpsi = torch.where(dpsi > math.pi / 6.0, dpsi - math.pi / 3.0, dpsi)
                # bias by one hex family per failed insert round: the nearest-family latch
                # would otherwise snap straight back to a family whose peck-band descent is
                # clamp-forbidden (URDF j1/j2 zero lower bounds; run 93 hovered +6.2mm forever)
                psi_star[:] = ky + dpsi + (math.pi / 3.0) * min(aim_tries, 5)
                clock_lat_prev = 9.9
                clock_switches = 0
            kq = kq_flip(psi_star)
            kp = torch.zeros(n, 3, device=dev)
            if t_in < 50:
                # UN-WEDGE: lift straight up from wherever the key actually is before any
                # lateral approach — with the whole train colliding, erect/travel can wedge
                # the key tip against the table or plate edge (runs 80/82: frozen at the
                # same geometry, tip at table level 67mm off-axis, servo unable to back out)
                kp[:] = key.data.root_pos_w
                kp[:, 2] = kp[:, 2] + 0.08
                tp, tq = tool_for_key(kp, kq)
                act = act_of(tp, tq, grip_c)
                continue_clock = False
            else:
                continue_clock = True
            tip_tgt = torch.zeros(n, 3, device=dev)
            tip_tgt[:, 0:2] = bolt.data.root_pos_w[:, 0:2]
            tip_tgt[:, 2] = bolt.data.root_pos_w[:, 2] + SOCKET_MOUTH_Z + INSERT_HOVER
            # HIGH APPROACH: with the full assembly colliding (key + fingertip + live claw),
            # a lateral translate at hover height drags the low-hanging train across the
            # platform edge and it snags (run 80: frozen 67mm off-axis, tip 33mm below the
            # mouth). Stay 60mm up until laterally centered, then descend on-axis.
            lat_now = key_lateral()  # TIP-space: the LONG tip must center over the bore
            # high-approach margin: the flip has NOTHING hanging below the tip (the claw is at
            # the TOP of the train), so 25mm clears the platform edge — the old 60mm was for the
            # low-hanging claw and busts the dome budget with the tall flipped train
            tip_tgt[:, 2] = tip_tgt[:, 2] + torch.where(lat_now > 0.020,
                                                        torch.full_like(lat_now, 0.025), torch.zeros_like(lat_now))
            if continue_clock:
                kp = root_for_tip(tip_tgt, kq)
                tp, tq = tool_for_key(kp, kq)
                act = act_of(tp, tq, grip_c)
                # PLATEAU-SWITCH the hex azimuth: every mod-60 representative is physically
                # equivalent (insert re-latches the clocking each peck), but they demand
                # DIFFERENT wrist configs — on the imported asset one family can hold the
                # erect orientation yet never reach the bore (run 83: parked 67mm off-axis,
                # weld 0.0, up +1.00). No lateral progress in 100 ticks -> next candidate.
                # 8x-slow reality check: a 60deg re-aim alone needs ~130 ticks at ROT_STEP,
                # and act_of's default rot-priority parks translation while yawing — 100-tick
                # windows churned families forever, swinging the key 0.5m out (run 129).
                if t_in >= 400 and t_in % 250 == 0:
                    lat_chk = float(lat_now.max())
                    if lat_chk > 0.020 and lat_chk > clock_lat_prev - 0.010:
                        clock_switches += 1
                        if clock_switches > 6:  # full turn tried: this CONFIG is tangled, not the azimuth
                            print(f"    [clock] no approach after {clock_switches - 1} family switches — config reset", flush=True)
                            phase, marker = "re_home", i
                        else:
                            psi_star[:] = psi_star + math.pi / 3.0
                            print(f"    [clock] azimuth plateau at lat {lat_chk * 1e3:.0f}mm -> +60deg", flush=True)
                    clock_lat_prev = lat_chk
            yaw_err = _wrap(crank_azim() - psi_star).abs()
            centred = (key_lateral() < 0.001) & (yaw_err < math.radians(2.5))
            if not bool(bolt_ok(say=True).all()):
                print("  ABORT: bolt left its nest before insertion", flush=True)
                phase, marker = "retreat", i
            elif bool(centred.all()) or t_in >= 3 * WP_TIMEOUT:
                phase, marker = "insert", i
        elif phase == "insert":  # hover-peck probe (the hex clearance is 0.7 mm/side): reposition
            # in FREE AIR between straight-down pecks over a small lateral lattice around the
            # bolt's live axis; the peck that lines up drops in, and a caught tip is driven to the
            # floor. The stiff joint PD tracks mm-level, so most runs catch on the first pecks.
            if t_in == 1:
                seat_ok.zero_()
                ins_last_ax = 9.9
                ins_stuck = 0
                ins_unwedge_until = 0
            if t_in % PECK_PERIOD == 1:  # re-latch the clocking each peck: taps walk the bolt's yaw
                ky = crank_azim()
                dpsi = _wrap((yaw_of(bolt.data.root_quat_w) + HEX_TRIM - ky) % (math.pi / 3.0))
                dpsi = torch.where(dpsi > math.pi / 6.0, dpsi - math.pi / 3.0, dpsi)
                psi_star[:] = ky + dpsi
            kq = kq_flip(psi_star)
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
            tip_in = (key_tip_axial() < SOCKET_MOUTH_Z - 0.0005) & (key_lateral() < 0.003)  # caught = below the mouth AND over the bore — axial alone reads true with the key beside the platform on the table (run 123: 'caught' at lat 185mm)
            tip_tgt = torch.zeros(n, 3, device=dev)
            tip_tgt[:, 0] = bolt.data.root_pos_w[:, 0] + torch.where(tip_in, torch.zeros_like(depth()), torch.full_like(depth(), ox))
            tip_tgt[:, 1] = bolt.data.root_pos_w[:, 1] + torch.where(tip_in, torch.zeros_like(depth()), torch.full_like(depth(), oy))
            tip_tgt[:, 2] = torch.where(tip_in, floor_z, z_cmd.clamp(min=floor_z))
            # WEDGED-CATCH BREAKER: a catch at lat > the socket's 0.7mm clearance jams on the
            # wall and the permanent straight-down drive can never advance or free it (run
            # 119: frozen at -2.28mm, lat 1.27). Caught + axially stagnant ~100 ticks -> lift
            # 2mm above the mouth for 30 ticks and let the lattice re-peck fresh.
            _axn = float(key_tip_axial().mean())
            if bool(tip_in.all()) and abs(ins_last_ax - _axn) < 2e-4:
                ins_stuck += 1
            else:
                ins_stuck = 0
            ins_last_ax = _axn
            # SEATING DITHER: the old full-speed pecks seated by tap MOMENTUM rattling the
            # hex through the tight zone; the slowed quasi-static press just leans on the
            # wall (runs 117-120: deterministic wedge at -4.2mm). Caught + stagnating ->
            # rotate a 0.5mm lateral micro-wiggle while pressing.
            if bool(tip_in.all()) and 20 <= ins_stuck:
                _ph = (t_in // 10) % 4
                tip_tgt[:, 0] = tip_tgt[:, 0] + (0.0005, 0.0, -0.0005, 0.0)[_ph]
                tip_tgt[:, 1] = tip_tgt[:, 1] + (0.0, 0.0005, 0.0, -0.0005)[_ph]
            if ins_stuck > 100:
                ins_stuck = 0
                ins_unwedge_until = t_in + 30
                print(f"    [insert] wedged catch at {_axn * 1e3:+.2f}mm — lifting to re-peck", flush=True)
                if cam is not None:
                    import imageio.v2 as _iio
                    _iio.imwrite(str(Path(args.video).parent / f"wedge_{t_in}.png"),
                                 cam.data.output["rgb"][0].cpu().numpy())
            if t_in < ins_unwedge_until:
                tip_tgt[:, 2] = mouth_z + 0.002
            kp = root_for_tip(tip_tgt, kq)
            tp, tq = tool_for_key(kp, kq)
            # position-primary during the peck descent: the jaw grip's wrist pose can make
            # the last few mm of descent fight the held orientation (run 92: hovering +5.6mm
            # with dpsi 0.0 forever). The clocking re-latches every peck, so a few degrees of
            # transient drift are self-correcting.
            act = act_of(tp, tq, grip_c, rot_w=0.6, raw=True)
            if t_in % PECK_PERIOD == PECK_PERIOD - 1:  # end of each press: where did it land?
                dp = _wrap((yaw_of(bolt.data.root_quat_w) + HEX_TRIM - crank_azim()) % (math.pi / 3.0))
                dp = torch.where(dp > math.pi / 6.0, dp - math.pi / 3.0, dp)
                lat = key_lateral()
                over = key_tip_axial() - SOCKET_MOUTH_Z
                print(f"    [probe {probe:2d}] lat {float(lat.max()) * 1e3:4.2f}mm | dpsi "
                      f"{float(torch.rad2deg(dp.abs().max())):4.1f}deg | tip over mouth {float(over.min()) * 1e3:+5.2f}mm", flush=True)
            # Seated (FLIPPED) = the LONG tip near the socket FLOOR — no bend to bottom out on
            # the mouth, so the full ~7mm recess engages; accept within 2.5mm of the floor
            # (wall friction can hold the quasi-static press just shy of bottom).
            settled = (key_tip_axial() < SOCKET_FLOOR_Z + 0.0025) & (key_tip_axial() > SOCKET_FLOOR_Z - 0.003) \
                & (key_lateral() < 0.0015) & (-handle_dir()[:, 2] > 0.99) & bolt_ok()
            seat_ok[:] = torch.where(settled, seat_ok + 1, torch.zeros_like(seat_ok))
            if not bool(bolt_ok(say=True).all()):
                print("  ABORT: bolt knocked out of its nest during insertion", flush=True)
                phase, marker = "retreat", i
            elif bool((seat_ok >= 15).all()):  # flush on the floor, upright, centred — and STAYING so
                aim_tries = 0
                if insert_depth0 is None:
                    insert_depth0 = depth().clone()
                stroke_psi.zero_()
                stroke_y0[:] = crank_azim()
                cycles += 1
                if cycles == 1:  # metrics start with the crank: forget pre-crank handling
                    bolt_turn.zero_()
                    key_turn.zero_()
                    prev_bolt_yaw = yaw_of(bolt.data.root_quat_w)
                    prev_key_yaw = crank_azim()
                if args.demo_insert:
                    print("  [demo] insert verified — holding the pose, then a gentle release", flush=True)
                    phase, marker = "admire", i
                else:
                    phase, marker = "stroke", i  # the handle grip IS the crank grip: no handoff
            elif t_in >= INSERT_TIMEOUT:
                aim_tries += 1
                if aim_tries >= 5 and cycles > 0:  # mid-loop re-seat exhausted: the key is at
                    # the HOVER (out of the socket) — stroking would spin air. End honestly
                    # with the cycles achieved.
                    print(f"  re-seat never landed after {cycles} cycles; ending honestly", flush=True)
                    phase, marker = "retreat", i
                elif aim_tries >= 5:
                    drops += 1
                    print("  DROP: insertion never landed after repeated re-clocks", flush=True)
                    phase, marker = "retreat", i
                elif aim_tries >= 2:
                    print("  insert timed out; RESETTING the arm config before re-clocking", flush=True)
                    phase, marker = "re_home", i
                else:
                    print("  insert timed out, re-clocking", flush=True)
                    phase, marker = "clock", i
        elif phase == "cage_convert":  # ALWAYS-HELD protocol: the L's insert grip becomes the
            # STEADY HAND — weld off (the key must SPIN) but the jaws STAY CLOSED: the pocket
            # cavity is larger than the hex (pair ~8mm = full closure AROUND the member), so
            # the post rotates freely inside while the pocket ROOF blocks lift-out and the
            # walls block tipping. The key is never unheld again until the demo ends.
            if t_in == 1:
                weld_off_L()
                hold_qL[:] = artL.data.joint_pos[:, arm_ids_L]  # RIGID latch (no pushover drift)
            left_cmd[:, 0:6] = hold_qL
            left_cmd[:, 6] = grip_c_L
            act = hold_act(OPEN_C)
            if t_in >= 150:
                ok_seat = (key_tip_axial() < SOCKET_MOUTH_Z - 0.004) & (key_lateral() < 0.004)
                if bool(ok_seat.all()):
                    holder = "L"
                    print("  STEADY HAND set (L, weld off, bite closed) — key held; R cranks", flush=True)
                    phase, marker = "crank_approach", i
                else:
                    print("  hold convert lost the seat; ending with the insert result", flush=True)
                    phase, marker = "admire", i
        elif phase == "crank_approach":  # RIGHT grabs the CRANK end-on (approach along the
            # member, slot across it) with its own scored frame ladder; the caged, seated key
            # is backed at both ends so the close cannot push it away
            sdh = up_axis_of(key.data.root_quat_w).clone()
            sdh[:, 2] = 0.0
            sdh = sdh / sdh.norm(dim=-1, keepdim=True).clamp_min(1e-6)  # tip -> elbow, horizontal
            crank_live = key.data.root_pos_w + up_axis_of(key.data.root_quat_w) * CRANK_GRIP_D
            if t_in == 1:
                wpR0_p[:], wpR0_q[:] = tool_pose()
                crank_lock[:] = crank_live
                crank_near = False
                _c, _u, _s, _t = crank_frames(crank_lock, sdh)[min(crank_tries, 5)]
                uR_lock[:] = _u
                cf_spin, cf_tilt = _s, _t
                print(f"    [crank] frame locked: cost {_c:.2f} spin {cf_spin:+.0f} tilt "
                      f"{math.degrees(cf_tilt):.0f}deg (cycle {crank_cycles + 1})", flush=True)
            qR = pinch_q(uR_lock, cf_spin, cf_tilt)
            if t_in < 200:
                # open-in-place dwell (matters on retries, harmless on entry)
                act = act_of(wpR0_p, wpR0_q, OPEN_C, rot_w=1.2)
                left_cmd[:, 0:6] = hold_qL
                left_cmd[:, 6] = grip_c_L
            else:
                if t_in == 1130:
                    crank_lock[:] = crank_live  # one pre-close re-latch
                    crank_near = bool((crank_perp("Right") < 0.025).all())
                    if not crank_near:
                        print(f"    [crank] approach never converged (perp "
                              f"{float(crank_perp('Right').max()) * 1e3:.0f}mm) — NOT closing (a far"
                              f" press-in topples the standing key)", flush=True)
                close_now = t_in > 1150 and crank_near
                # the FIRST approach starts from the R's park (~0.25m out): long hover leg
                off = -0.06 if t_in < 800 else (0.006 if close_now else 0.0)
                vt = crank_lock + uR_lock * off
                tpR = vt - quat_apply(qR, ex1 * tool_to_grip)
                act = act_of(tpR, qR, 0.004 if close_now else OPEN_C, rot_w=1.2)
                left_cmd[:, 0:6] = hold_qL
                left_cmd[:, 6] = grip_c_L
            if t_in % 150 == 0:
                prR = jaw_pair(artR, jaw_ids_R)
                mR = torch.minimum(artR.data.joint_pos[:, arm_ids] - lo_lim,
                                   hi_lim - artR.data.joint_pos[:, arm_ids])
                print(f"    [crank t{t_in:4d}] R pair {float(prR[0]) * 1e3:5.1f}mm | perp "
                      f"{float(crank_perp('Right').max()) * 1e3:5.1f}mm | Rmargin "
                      + " ".join(f"{float(v):+.2f}" for v in mR[0]), flush=True)
            if t_in == 1500:
                pairR_hold[:] = jaw_pair(artR, jaw_ids_R)
            if t_in >= 1600:
                prR = jaw_pair(artR, jaw_ids_R)
                stableR = (prR - pairR_hold).abs() < 0.002
                okR = (prR < 0.020) & (crank_perp("Right") < 0.018) & stableR
                if bool(okR.all()):
                    grip_c = float(prR[0]) * 0.5 + 0.0005
                    weld_on()
                    crank_tries = 0
                    crank_bounce = 0
                    print(f"  CRANK GRAB: R stall {float(prR[0]) * 1e3:.1f}mm — stroking", flush=True)
                    phase, marker = "crank_stroke", i
                elif crank_tries < 4:
                    crank_tries += 1
                    print(f"  crank grab missed (pair {float(prR[0]) * 1e3:.1f}mm, perp "
                          f"{float(crank_perp('Right').max()) * 1e3:.1f}mm), frame {crank_tries + 1}", flush=True)
                    marker = i
                elif crank_bounce < 1:
                    crank_bounce += 1
                    crank_tries = 0
                    print("  R grab arc exhausted — ROLE SWAP (R holds, L cranks)", flush=True)
                    phase, marker = "role_swap_R_holder", i
                else:
                    print("  crank grab exhausted; releasing with the progress", flush=True)
                    phase, marker = "crank_release", i
        elif phase == "crank_stroke":  # the small-orbit screw stroke: rotate the key about the
            # bore axis (helical floor press), the L cage holding the post vertical
            if t_in == 1:
                stroke_psi.zero_()
                stroke_y0[:] = crank_azim()
                crank_press_hold = 0
                if bool((key_tip_axial() > SOCKET_MOUTH_Z + 0.004).any()) or bool((key.data.root_pos_w[:, 2] < 0.10).any()):
                    print("  [crank] key NOT seated at stroke start (fell during a swap) — releasing honestly", flush=True)
                    phase, marker = "crank_release", i
            popped = key_tip_axial() > SOCKET_MOUTH_Z - 0.002
            if crank_press_hold == 0 and bool(popped.any()):
                # the orbit start can YANK the tip above the mouth (run 148 stroke 2): DO NOT
                # release a precarious key — freeze the orbit and RE-PRESS while still welded
                crank_press_hold = 300
                print("  [crank] tip above the mouth — freezing the orbit to re-press", flush=True)
            if crank_press_hold > 0:
                crank_press_hold -= 1
                if not bool(popped.any()):
                    crank_press_hold = 0  # re-seated: resume the sweep
                elif crank_press_hold == 0:
                    print("  [crank] re-press failed — regripping", flush=True)
                    phase, marker = "crank_regrip", i
            else:
                stroke_psi += CRANK_W * min(t_in / 200.0, 1.0)  # rate ramp: no start jerk
                stroke_psi.clamp_(max=STROKE_RAD)
            kq = kq_flip(stroke_y0 - crank_sign * stroke_psi)
            tip_tgt = torch.zeros(n, 3, device=dev)
            tip_tgt[:, 0:2] = bolt.data.root_pos_w[:, 0:2]
            tip_tgt[:, 2] = bolt.data.root_pos_w[:, 2] + SOCKET_FLOOR_Z - 0.0025
            kp = root_for_tip(tip_tgt, kq)
            tp, tq = tool_for_key(kp, kq)
            act = act_of(tp, tq, grip_c, rot_w=0.8)  # press-primary: the orbit must not win over the seat
            left_cmd[:, 0:6] = hold_qL
            left_cmd[:, 6] = grip_c_L
            if not crank_dir_checked and t_in == 500:
                crank_dir_checked = True
                if float(torch.rad2deg(bolt_turn).mean()) < -3.0:
                    crank_sign = -1.0  # run-level lock: flip ONCE (backing-out response)
                    print("  [crank] direction flip: the bolt was backing out — locked reversed", flush=True)
                    marker = i
            if not bool(bolt_ok(say=True).all()):
                print("  ABORT: bolt left its nest mid-stroke", flush=True)
                phase, marker = "retreat", i
            elif bool((-handle_dir()[:, 2] < 0.97).any()) and float(stroke_psi.max()) > math.radians(35):
                crank_cycles += 1
                print(f"  [stroke end] tilt-guard at {math.degrees(float(stroke_psi.max())):.0f}deg "
                      f"| bolt {float(torch.rad2deg(bolt_turn).mean()):+.0f}deg — swapping hands", flush=True)
                phase, marker = "crank_regrip", i
            elif bool(arm_near_limit().any()) and t_in > 100:
                crank_cycles += 1
                print(f"  [stroke end] joint-limit at {math.degrees(float(stroke_psi.max())):.0f}deg "
                      f"| bolt {float(torch.rad2deg(bolt_turn).mean()):+.0f}deg", flush=True)
                phase, marker = "crank_regrip", i
            elif bool((stroke_psi >= STROKE_RAD - 1e-6).all()) or t_in > 2600:
                crank_cycles += 1
                slip_now = float(torch.rad2deg(key_turn - bolt_turn).mean())
                print(f"  [stroke {crank_cycles}] swept {math.degrees(float(stroke_psi.max())):.0f}deg | "
                      f"bolt {float(torch.rad2deg(bolt_turn).mean()):+.0f}deg | slip {slip_now:+.0f}deg | "
                      f"depth {float(depth().mean()) * 1e3:+.2f}mm", flush=True)
                if float(bolt_turn.mean()) >= 2.0 * math.pi * args.demo_revs:
                    print(f"  [demo] {args.demo_revs} revolutions driven — releasing", flush=True)
                    phase, marker = "crank_release", i
                elif crank_cycles >= MAX_CRANK_CYCLES:
                    print("  stroke budget spent — releasing with the measured revs", flush=True)
                    phase, marker = "crank_release", i
                else:
                    phase, marker = "crank_regrip", i
        elif phase == "crank_regrip":  # release discipline, then re-approach the advanced crank.
            # EXIT ALONG THE CRANK AXIS, not vertically: the hooks scoop UNDER the horizontal
            # member — a rising exit lifts the key out of the socket and the falling key
            # unscrews the bolt (runs 148/150: +54deg wound back to +21deg). Slide the open
            # pocket OFF the member's end instead (the way it came in).
            if t_in == 1:
                wpR0_p[:], wpR0_q[:] = tool_pose()
                weld_off()
            act = act_of(wpR0_p - (uR_lock * 0.08 if t_in >= 250 else uR_lock * 0.0), wpR0_q, OPEN_C, rot_w=1.2)
            left_cmd[:, 0:6] = hold_qL
            left_cmd[:, 6] = grip_c_L
            if t_in >= 600:
                crank_tries = 0
                # ADAPTIVE HAND: whichever arm's best frame is cheaper takes the next stroke
                # (strides differ, so the crank isn't alternately in each arc — run 153: the
                # L swept 127deg and left the crank still deep in its own territory)
                sdh2 = up_axis_of(key.data.root_quat_w).clone()
                sdh2[:, 2] = 0.0
                sdh2 = sdh2 / sdh2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                cl2 = key.data.root_pos_w + up_axis_of(key.data.root_quat_w) * CRANK_GRIP_D
                cR2 = crank_frames(cl2, sdh2)[0][0]
                cL2 = crank_frames_L(cl2, sdh2)[0][0]
                phase = "crank_approach" if cR2 <= cL2 else "role_swap_R_holder"
                print(f"    [swap] frame costs R {cR2:.2f} / L {cL2:.2f} -> "
                      f"{'RIGHT re-grabs' if cR2 <= cL2 else 'ROLE SWAP (R holds, L cranks)'}", flush=True)
                marker = i
        elif phase == "crank_approach_L":  # the LEFT grabs the advanced crank (far arc)
            sdh = up_axis_of(key.data.root_quat_w).clone()
            sdh[:, 2] = 0.0
            sdh = sdh / sdh.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            crank_live = key.data.root_pos_w + up_axis_of(key.data.root_quat_w) * CRANK_GRIP_D
            if t_in == 1:
                wpL0_p[:] = toolL_pose()[0]
                wpL_q[:] = toolL_pose()[1]
                crank_lock[:] = crank_live
                crank_near = False
                _c, _u, _s, _t = crank_frames_L(crank_lock, sdh)[min(crank_tries, 5)]
                uL_lock[:] = _u
                pf_spin, pf_tilt = _s, _t
                print(f"    [crankL] frame locked: cost {_c:.2f} spin {pf_spin:+.0f} tilt "
                      f"{math.degrees(pf_tilt):.0f}deg (cycle {crank_cycles + 1})", flush=True)
            qL = pinch_q(uL_lock, pf_spin, pf_tilt)
            if t_in < 200:
                left_to(wpL0_p, wpL_q, OPEN_C, rot_w=1.2)
            else:
                if t_in == 1130:
                    crank_lock[:] = crank_live
                    crank_near = bool((crank_perp("Left") < 0.025).all())
                    if not crank_near:
                        print(f"    [crankL] approach never converged (perp "
                              f"{float(crank_perp('Left').max()) * 1e3:.0f}mm) — NOT closing", flush=True)
                close_now = t_in > 1150 and crank_near
                off = -0.06 if t_in < 800 else (0.006 if close_now else 0.0)
                vt = crank_lock + uL_lock * off
                left_to(vt - quat_apply(qL, ex1 * tool_to_grip), qL, 0.004 if close_now else OPEN_C, rot_w=1.2)
            act = act_R_hold(grip_hold_R)
            if t_in % 300 == 0:
                prL = jaw_pair(artL, jaw_ids_L)
                print(f"    [crankL t{t_in:4d}] L pair {float(prL[0]) * 1e3:5.1f}mm | perp "
                      f"{float(crank_perp('Left').max()) * 1e3:5.1f}mm", flush=True)
            if t_in == 1500:
                pairR_hold[:] = jaw_pair(artL, jaw_ids_L)
            if t_in >= 1600:
                prL = jaw_pair(artL, jaw_ids_L)
                stableL = (prL - pairR_hold).abs() < 0.002
                okL2 = (prL < 0.020) & (crank_perp("Left") < 0.018) & stableL
                if bool(okL2.all()):
                    grip_c_L = float(prL[0]) * 0.5 + 0.0005
                    weld_on_L()
                    crank_tries = 0
                    crank_bounce = 0
                    print(f"  CRANK GRAB (L): stall {float(prL[0]) * 1e3:.1f}mm — stroking", flush=True)
                    phase, marker = "crank_stroke_L", i
                elif crank_tries < 4:
                    crank_tries += 1
                    print(f"  L crank grab missed (pair {float(prL[0]) * 1e3:.1f}mm, perp "
                          f"{float(crank_perp('Left').max()) * 1e3:.1f}mm), frame {crank_tries + 1}", flush=True)
                    marker = i
                elif crank_bounce < 1:
                    crank_bounce += 1
                    crank_tries = 0
                    print("  L grab arc exhausted — ROLE SWAP (L holds, R cranks)", flush=True)
                    phase, marker = "role_swap_L_holder", i
                else:
                    print("  L crank grab exhausted; releasing with the progress", flush=True)
                    phase, marker = "crank_release_L", i
        elif phase == "crank_stroke_L":  # the LEFT's sweep through the far arc
            if t_in == 1:
                stroke_psi.zero_()
                stroke_y0[:] = crank_azim()
                crank_press_hold = 0
                if bool((key_tip_axial() > SOCKET_MOUTH_Z + 0.004).any()) or bool((key.data.root_pos_w[:, 2] < 0.10).any()):
                    print("  [crankL] key NOT seated at stroke start (fell during a swap) — releasing honestly", flush=True)
                    phase, marker = "crank_release_L", i
            popped = key_tip_axial() > SOCKET_MOUTH_Z - 0.002
            if crank_press_hold == 0 and bool(popped.any()):
                crank_press_hold = 300
                print("  [crankL] tip above the mouth — freezing to re-press", flush=True)
            if crank_press_hold > 0:
                crank_press_hold -= 1
                if not bool(popped.any()):
                    crank_press_hold = 0
                elif crank_press_hold == 0:
                    print("  [crankL] re-press failed — swapping back", flush=True)
                    phase, marker = "crank_regrip_L", i
            else:
                stroke_psi += CRANK_W * min(t_in / 200.0, 1.0)
                stroke_psi.clamp_(max=STROKE_RAD)
            kq = kq_flip(stroke_y0 - crank_sign * stroke_psi)
            tip_tgt = torch.zeros(n, 3, device=dev)
            tip_tgt[:, 0:2] = bolt.data.root_pos_w[:, 0:2]
            tip_tgt[:, 2] = bolt.data.root_pos_w[:, 2] + SOCKET_FLOOR_Z - 0.0025
            tpL2, tqL2 = tool_for_key_L(root_for_tip(tip_tgt, kq), kq)
            left_to(tpL2, tqL2, grip_c_L, rot_w=0.8)
            act = act_R_hold(grip_hold_R)
            if not bool(bolt_ok(say=True).all()):
                print("  ABORT: bolt left its nest mid-stroke (L)", flush=True)
                phase, marker = "retreat", i
            elif bool((-handle_dir()[:, 2] < 0.97).any()) and float(stroke_psi.max()) > math.radians(35):
                crank_cycles += 1
                print(f"  [strokeL end] tilt-guard at {math.degrees(float(stroke_psi.max())):.0f}deg "
                      f"| bolt {float(torch.rad2deg(bolt_turn).mean()):+.0f}deg — swapping hands", flush=True)
                phase, marker = "crank_regrip_L", i
            elif bool((stroke_psi >= STROKE_RAD - 1e-6).all()) or t_in > 2600:
                crank_cycles += 1
                print(f"  [strokeL {crank_cycles}] swept {math.degrees(float(stroke_psi.max())):.0f}deg | "
                      f"bolt {float(torch.rad2deg(bolt_turn).mean()):+.0f}deg | "
                      f"depth {float(depth().mean()) * 1e3:+.2f}mm", flush=True)
                if float(bolt_turn.mean()) >= 2.0 * math.pi * args.demo_revs:
                    print(f"  [demo] {args.demo_revs} revolutions driven — releasing", flush=True)
                    phase, marker = "crank_release_L", i
                elif crank_cycles >= MAX_CRANK_CYCLES:
                    print("  stroke budget spent — releasing with the measured revs", flush=True)
                    phase, marker = "crank_release_L", i
                else:
                    phase, marker = "crank_regrip_L", i
        elif phase == "crank_regrip_L":  # L releases along the member axis, hand back to the R
            if t_in == 1:
                wpL0_p[:] = toolL_pose()[0]
                wpL_q[:] = toolL_pose()[1]
                if bool(welded_L.any()):
                    weld_off_L()
            left_to(wpL0_p - (uL_lock * 0.08 if t_in >= 250 else uL_lock * 0.0), wpL_q, OPEN_C, rot_w=1.2)
            act = act_R_hold(grip_hold_R)
            if t_in >= 600:
                crank_tries = 0
                if crank_cycles >= MAX_CRANK_CYCLES or float(bolt_turn.mean()) >= 2.0 * math.pi * args.demo_revs:
                    phase, marker = "admire", i
                else:
                    sdh2 = up_axis_of(key.data.root_quat_w).clone()
                    sdh2[:, 2] = 0.0
                    sdh2 = sdh2 / sdh2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    cl2 = key.data.root_pos_w + up_axis_of(key.data.root_quat_w) * CRANK_GRIP_D
                    cR2 = crank_frames(cl2, sdh2)[0][0]
                    cL2 = crank_frames_L(cl2, sdh2)[0][0]
                    phase = "crank_approach_L" if cL2 <= cR2 else "role_swap_L_holder"
                    print(f"    [swap] frame costs R {cR2:.2f} / L {cL2:.2f} -> "
                          f"{'LEFT re-grabs' if cL2 <= cR2 else 'ROLE SWAP (L holds, R cranks)'}", flush=True)
                    marker = i
        elif phase == "crank_release_L":  # demo end from an L-held stroke
            if t_in == 1:
                wpL0_p[:] = toolL_pose()[0]
                wpL_q[:] = toolL_pose()[1]
                if bool(welded_L.any()):
                    weld_off_L()
            left_to(wpL0_p - (uL_lock * 0.08 if t_in >= 250 else uL_lock * 0.0), wpL_q, OPEN_C, rot_w=1.2)
            act = act_R_hold(grip_hold_R)
            if t_in >= 600:
                phase, marker = "admire", i
        elif phase == "role_swap_R_holder":  # the R takes the steady-hand role: grab the post
            # LOW (below the L's pocket — both fit the 110mm post) while the L STILL holds:
            # the key is never free for a single tick.
            elbow = key.data.root_pos_w + up_axis_of(key.data.root_quat_w) * ARM_LEN
            hdn = handle_dir()
            s_p = (elbow[:, 2:3] - POST_HOLD_Z_R).clamp(0.02, 0.11)
            post_pt = elbow + hdn * s_p
            if t_in == 1:
                wpR0_p[:], wpR0_q[:] = tool_pose()
                crank_lock[:] = post_pt
                crank_near = False
                _c, _u, _s, _t = post_frames_R(crank_lock)[min(crank_tries, 5)]
                uR_lock[:] = _u
                cf_spin, cf_tilt = _s, _t
                print(f"    [roleswap->R] post frame: cost {_c:.2f} spin {_s:+.0f} tilt "
                      f"{math.degrees(_t):.0f}deg", flush=True)
            qR = pinch_q(uR_lock, cf_spin, cf_tilt)
            if t_in < 200:
                act = act_of(wpR0_p, wpR0_q, OPEN_C, rot_w=1.2)
            else:
                if t_in == 1130:
                    crank_lock[:] = post_pt
                    crank_near = bool((tip_perp("Right") < 0.030).all())
                    if not crank_near:
                        print(f"    [roleswap->R] approach never converged (perp "
                              f"{float(tip_perp('Right').max()) * 1e3:.0f}mm) — NOT closing", flush=True)
                close_now = t_in > 1150 and crank_near
                off = -0.06 if t_in < 800 else (0.006 if close_now else 0.0)
                vt = crank_lock + uR_lock * off
                act = act_of(vt - quat_apply(qR, ex1 * tool_to_grip), qR,
                             0.004 if close_now else OPEN_C, rot_w=1.2)
            left_cmd[:, 0:6] = hold_qL
            left_cmd[:, 6] = grip_c_L
            if t_in == 1500:
                pairR_hold[:] = jaw_pair(artR, jaw_ids_R)
            if t_in >= 1600:
                prR = jaw_pair(artR, jaw_ids_R)
                stH = (prR - pairR_hold).abs() < 0.002
                okH = (prR < 0.033) & (tip_perp("Right") < 0.018) & stH
                if bool(okH.all()):
                    grip_hold_R = float(prR[0]) * 0.5 + 0.001
                    hold_qR[:] = artR.data.joint_pos[:, arm_ids]
                    crank_tries = 0
                    holder = "R"
                    print(f"  ROLE SWAP: R steady hand set (stall {float(prR[0]) * 1e3:.1f}mm) — L releases", flush=True)
                    phase, marker = "role_swap_L_release", i
                elif crank_tries < 4:
                    crank_tries += 1
                    print(f"  R post-hold missed (pair {float(prR[0]) * 1e3:.1f}mm, perp "
                          f"{float(tip_perp('Right').max()) * 1e3:.1f}mm), frame {crank_tries + 1}", flush=True)
                    marker = i
                else:
                    print("  role swap failed; releasing with the progress", flush=True)
                    phase, marker = "crank_release", i
        elif phase == "role_swap_L_release":  # the L hands over: open + VERTICAL exit (post)
            if t_in == 1:
                wpL0_p[:] = toolL_pose()[0]
                wpL_q[:] = toolL_pose()[1]
            left_to(wpL0_p + (ez * 0.06 if t_in >= 250 else ez * 0.0), wpL_q, OPEN_C, rot_w=1.2)
            act = act_R_hold(grip_hold_R)
            if t_in >= 600:
                crank_tries = 0
                phase, marker = "crank_approach_L", i
        elif phase == "role_swap_L_holder":  # the L takes the steady-hand role back
            elbow = key.data.root_pos_w + up_axis_of(key.data.root_quat_w) * ARM_LEN
            hdn = handle_dir()
            s_p = (elbow[:, 2:3] - POST_HOLD_Z_L).clamp(0.02, 0.11)
            post_pt = elbow + hdn * s_p
            if t_in == 1:
                wpL0_p[:] = toolL_pose()[0]
                wpL_q[:] = toolL_pose()[1]
                pinch_lock[:] = post_pt
                crank_near = False
                _c, _u, _s, _t = post_frames(pinch_lock)[min(crank_tries, 5)]
                uL_lock[:] = _u
                pf_spin, pf_tilt = _s, _t
                print(f"    [roleswap->L] post frame: cost {_c:.2f} spin {_s:+.0f} tilt "
                      f"{math.degrees(_t):.0f}deg", flush=True)
            qL = pinch_q(uL_lock, pf_spin, pf_tilt)
            if t_in < 200:
                left_to(wpL0_p, wpL_q, OPEN_C, rot_w=1.2)
            else:
                if t_in == 1130:
                    pinch_lock[:] = post_pt
                    crank_near = bool((tip_perp("Left") < 0.030).all())
                    if not crank_near:
                        print(f"    [roleswap->L] approach never converged (perp "
                              f"{float(tip_perp('Left').max()) * 1e3:.0f}mm) — NOT closing", flush=True)
                close_now = t_in > 1150 and crank_near
                off = -0.06 if t_in < 800 else (0.006 if close_now else 0.0)
                vt = pinch_lock + uL_lock * off
                left_to(vt - quat_apply(qL, ex1 * tool_to_grip), qL,
                        0.004 if close_now else OPEN_C, rot_w=1.2)
            act = act_R_hold(grip_hold_R)
            if t_in == 1500:
                pairR_hold[:] = jaw_pair(artL, jaw_ids_L)
            if t_in >= 1600:
                prL = jaw_pair(artL, jaw_ids_L)
                stH = (prL - pairR_hold).abs() < 0.002
                okH = (prL < 0.033) & (tip_perp("Left") < 0.018) & stH
                if bool(okH.all()):
                    grip_c_L = float(prL[0]) * 0.5 + 0.001
                    hold_qL[:] = artL.data.joint_pos[:, arm_ids_L]
                    crank_tries = 0
                    holder = "L"
                    print(f"  ROLE SWAP: L steady hand set (stall {float(prL[0]) * 1e3:.1f}mm) — R releases", flush=True)
                    phase, marker = "role_swap_R_release", i
                elif crank_tries < 4:
                    crank_tries += 1
                    print(f"  L post-hold missed (pair {float(prL[0]) * 1e3:.1f}mm, perp "
                          f"{float(tip_perp('Left').max()) * 1e3:.1f}mm), frame {crank_tries + 1}", flush=True)
                    marker = i
                else:
                    print("  role swap failed; releasing with the progress", flush=True)
                    phase, marker = "crank_release_L", i
        elif phase == "role_swap_R_release":  # the R hands over: open + VERTICAL exit (post)
            if t_in == 1:
                wpR0_p[:], wpR0_q[:] = tool_pose()
            act = act_of(wpR0_p + (ez * 0.06 if t_in >= 250 else ez * 0.0), wpR0_q, OPEN_C, rot_w=1.2)
            left_cmd[:, 0:6] = hold_qL
            left_cmd[:, 6] = grip_c_L
            if t_in >= 600:
                crank_tries = 0
                phase, marker = "crank_approach", i
        elif phase == "crank_release":  # demo end: open in place, exit along the crank axis
            if t_in == 1:
                wpR0_p[:], wpR0_q[:] = tool_pose()
                if bool(welded.any()):
                    weld_off()
            act = act_of(wpR0_p - (uR_lock * 0.08 if t_in >= 250 else uR_lock * 0.0), wpR0_q, OPEN_C, rot_w=1.2)
            left_cmd[:, 0:6] = hold_qL
            left_cmd[:, 6] = grip_c_L
            if t_in >= 600:
                phase, marker = "admire", i
        elif phase == "admire":  # demo ending: hold the seated key for the camera. The
            # STEADY HAND keeps its bite (holder-aware — never yank the holding arm home).
            a = torch.zeros(n, act_dim, device=dev)
            if holder == "L":
                a[:, l_s.start : l_s.start + 6] = hold_qL
                a[:, l_s.start + 6] = grip_c_L
                rh2 = artR.data.joint_pos[:, arm_ids]
                a[:, r_s.start : r_s.start + 6] = rh2 + (artR.data.default_joint_pos[:, arm_ids] - rh2).clamp(-DQ_STEP, DQ_STEP)
                a[:, r_s.start + 6] = OPEN_C
            else:
                a[:, r_s.start : r_s.start + 6] = hold_qR
                a[:, r_s.start + 6] = grip_hold_R
                a[:, l_s.start : l_s.start + 6] = artL.data.joint_pos[:, arm_ids_L]
                a[:, l_s.start + 6] = OPEN_C
            act = a
            if t_in >= 180:
                phase, marker = "retreat", i
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
            # cam-out guard: hex slip within THIS stroke means the key is riding the corners —
            # end the stroke; the periodic re-seat (every 3rd cycle) restores registration+depth
            slipping = ((key_turn - bolt_turn) - stroke_slip0).abs() > math.radians(20.0)
            if bool(done.all()):
                phase, marker = "retreat", i
            elif bool((swept | arm_near_limit() | slipping).all()) or t_in >= STROKE_TIMEOUT:
                why = ("swept" if bool(swept.all()) else
                       "limit" if bool(arm_near_limit().all()) else
                       "slip" if bool(slipping.all()) else "timeout")
                print(f"    [stroke end] {why} at psi {float(stroke_psi.max()) * 57.3:.0f}deg"
                      f" (t {t_in})", flush=True)
                # the present/handoff/re-seat cadence is the PROVEN loop (run 102: +197deg net
                # with a full bimanual jaw handoff). release_regrip (the fast in-place ratchet)
                # stays as future work: its re-pinch positioning at arbitrary post-stroke
                # azimuths is reach-roulette (residuals 8-130mm across attempts).
                phase, marker = "present", i
        elif phase == "release_regrip":  # FAST RATCHET: the key STANDS in the socket (5mm
            # hex engagement + self-locking thread) while the right releases, unwinds, and
            # re-pinches the handle IN PLACE — no lift, no re-insert: ~10x the screwing
            # cadence of the present/handoff cycle. Grasp re-verified by the pair stall.
            hd = handle_dir()
            hd2 = hd.clone(); hd2[:, 2] = 0.0
            hd2 = hd2 / hd2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            elbow = key.data.root_pos_w + up_axis_of(key.data.root_quat_w) * ARM_LEN
            if t_in == 1:
                weld_off()
                unwind_up[:] = tip_pos("Right")
                unwind_up[:, 2] = unwind_up[:, 2] + 0.07
                sRr = ((base_R_xy - elbow[:, :2]) * hd2[:, :2]).sum(-1, keepdim=True).clamp(0.030, 0.095)
                vice_sR[:] = (sRr + (0.0, 0.018, -0.018)[min(rr_tries, 2)]).clamp(0.030, 0.095)
            left_park()
            if t_in < 45:
                act = act_of_tip(unwind_up, OPEN_C)
            elif t_in < 140:
                a = torch.zeros(n, act_dim, device=dev)
                a[:, l_s] = left_cmd
                a[:, r_s.start : r_s.start + 6] = artR.data.default_joint_pos[:, arm_ids]
                a[:, r_s.start + 6] = OPEN_C
                act = a
            else:
                t2 = t_in - 140
                hp = elbow + hd2 * vice_sR
                tcp_tgt = hp.clone()
                tcp_tgt[:, 2] = hp[:, 2] + 0.0024
                nvp = torch.stack((-hd2[:, 1], hd2[:, 0], torch.zeros(n, device=dev)), dim=-1)
                sgnRr = torch.sign(((base_R_xy - hp[:, :2]) * nvp[:, :2]).sum(-1)).unsqueeze(-1)
                sgnRr = torch.where(sgnRr == 0, torch.ones_like(sgnRr), sgnRr)
                tqRr = pinch_q(nvp * sgnRr, +1.0, math.radians(25.0))
                if t2 < 50:
                    vt = tcp_tgt.clone()
                    vt[:, 2] = vt[:, 2] + 0.06
                    gcmd = OPEN_C
                elif t2 < 110:
                    vt = tcp_tgt
                    gcmd = OPEN_C
                else:
                    vt = tcp_tgt
                    # close ONLY while the straddle is actually over the press point — a
                    # timer-blind close grabs the key's corner 26mm off (run 104: pair 48.2
                    # jammed on the handle-member junction with cmd/targets perfect)
                    _pos_ok = (tip_pos("Right")[:, 0:2] - hp[:, 0:2]).norm(dim=-1) < 0.008
                    gcmd = 0.004 if bool(_pos_ok.all()) else OPEN_C
                wp_q[:] = tqRr
                wp_p[:] = vt - quat_apply(tqRr, ex1) * tool_to_grip
                act = act_of(wp_p, wp_q, gcmd, rot_w=1.2)
                if t2 >= 110 + CLOSE_STEPS and (t2 - 110 - CLOSE_STEPS) % 20 == 0:
                    prr = jaw_pair(artR, jaw_ids_R)
                    dxy = (tip_pos("Right")[:, 0:2] - hp[:, 0:2]).norm(dim=-1)
                    xy_ok = dxy < 0.015
                    okp = (prr > 0.0105) & (prr < 0.0165) & xy_ok
                    jt = artR.data.joint_pos_target[0, jaw_ids_R]
                    print(f"    [regrip t{t2:3d}] pair {float(prr[0]) * 1e3:5.1f}mm"
                          f" | cmd {gcmd * 1e3:.1f} | tgt ({float(jt[0]) * 1e3:.1f},{float(jt[1]) * 1e3:.1f})"
                          f" | dxy {float(dxy[0]) * 1e3:5.1f}mm", flush=True)
                    if bool(okp.all()):
                        rr_tries = 0
                        grip_c = float(prr[0]) * 0.5 + 0.0005
                        weld_on()
                        stroke_psi.zero_()
                        stroke_y0[:] = crank_azim()
                        cycles += 1
                        print(f"  ratchet regrip {cycles}: cranking", flush=True)
                        phase, marker = "stroke", i
            if t_in >= 3 * WP_TIMEOUT and phase == "release_regrip":
                if rr_tries < 2:
                    rr_tries += 1
                    print(f"  regrip stuck; retrying another spot ({rr_tries}/2)", flush=True)
                    marker = i - 140  # restart at the pinch stages (t_in jumps past unwind)
                else:
                    print("  ABORT: ratchet regrip never verified", flush=True)
                    phase, marker = "retreat", i
        elif phase == "present":  # lift the key out and PRESENT the handle at +y — the one
            # fixed, symmetric azimuth — before any handoff. The hex re-clocks at insert.
            hd0 = handle_dir()
            psi_h = torch.atan2(hd0[:, 1], hd0[:, 0])
            psi_want = torch.atan2(base_R_xy[:, 1] - bolt.data.root_pos_w[:, 1],
                                   base_R_xy[:, 0] - bolt.data.root_pos_w[:, 0])
            dpsi_h = _wrap(psi_want - psi_h)
            kq = quat_mul(quat_from_angle_axis(dpsi_h, ez), key.data.root_quat_w)
            kp = torch.zeros(n, 3, device=dev)
            kp[:, 0:2] = bolt.data.root_pos_w[:, 0:2]
            kp[:, 2] = bolt.data.root_pos_w[:, 2] + SOCKET_MOUTH_Z + INSERT_HOVER + 0.025
            tp, tq = tool_for_key(kp, kq)
            act = act_of(tp, tq, grip_c)
            left_park()
            lifted = key_tip_axial() > SOCKET_MOUTH_Z + 0.015
            aligned = dpsi_h.abs() < math.radians(8.0)
            if bool((lifted & aligned).all()) or t_in >= 2 * WP_TIMEOUT:
                phase, marker = "hold_press", i
        elif phase == "hold_press":  # VICE: the LEFT jaw-pinches the key's VERTICAL MEMBER
            # at the presented hover — fully exposed below the handle, azimuth-agnostic, and
            # geometrically disjoint from the RIGHT's handle grip (no stagger needed). The
            # 75deg tilt makes pinch_q close the jaws horizontally ACROSS the vertical hex;
            # tilt_azimuth's comfort solve picks a reachable wrist on the lever circle.
            if t_in == 1:
                hold_qR[:] = artR.data.joint_pos[:, arm_ids]
            up_k = up_axis_of(key.data.root_quat_w)
            vp = key.data.root_pos_w + up_k * (ARM_LEN - 0.048)
            uLv = tilt_azimuth(base_L_xy, vp, tool_to_grip, math.radians(75.0))
            spinLv = -1.0 if hold_tries % 2 == 0 else 1.0
            tqLv = pinch_q(uLv, spinLv, math.radians(75.0))
            if t_in < 60:
                vt = vp + uLv * 0.05  # stage 50mm before the member along the approach
                gcmd = OPEN_C
            elif t_in < 110:
                vt = vp
                gcmd = OPEN_C
            else:
                vt = vp
                gcmd = 0.004
            left_to(vt - quat_apply(tqLv, ex1) * tool_to_grip, tqLv, grip=gcmd, rot_w=1.2)
            a = torch.zeros(n, act_dim, device=dev)
            a[:, l_s] = left_cmd
            a[:, r_s.start : r_s.start + 6] = hold_qR
            a[:, r_s.start + 6] = grip_c
            act = a
            if t_in >= 110 + CLOSE_STEPS and (t_in - 110 - CLOSE_STEPS) % 20 == 0:
                pl = jaw_pair(artL, jaw_ids_L)
                tzz = tip_pos("Left")
                xy_ok = (tzz - vp).norm(dim=-1) < 0.015
                okp = (pl > 0.0105) & (pl < 0.0165) & xy_ok
                print(f"    [hold t{t_in:3d}] L-pair {float(pl[0]) * 1e3:5.1f}mm"
                      f" | d {float((tzz - vp).norm(dim=-1)[0]) * 1e3:4.1f}mm", flush=True)
                if bool(okp.all()):
                    hold_tries = 0
                    vice_gripL = float(pl[0]) * 0.5 + 0.0005
                    weld_on_L()
                    weld_off()
                    print("  vice handoff: LEFT jaw-holds the member, RIGHT unwinding", flush=True)
                    phase, marker = "unwind", i
            if t_in >= 3 * WP_TIMEOUT and phase == "hold_press":
                if hold_tries < 2:
                    hold_tries += 1
                    print(f"  member pinch stuck; retrying ({hold_tries}/2)", flush=True)
                    left_park()
                    marker = i
                else:
                    hold_tries = 0
                    print("  vice pinch failed; re-seating and cranking WITHOUT a handoff", flush=True)
                    left_park()
                    phase, marker = "clock", i  # the key is at the HOVER: re-seat, then stroke
        elif phase == "unwind":  # the right OPENS, lifts its claw straight off the handle,
            # then returns to home posture — the key is safe in the LEFT's jaw+weld vice
            if t_in == 1:
                hold_qL[:] = artL.data.joint_pos[:, arm_ids_L]
                unwind_up[:] = tip_pos("Right")
                unwind_up[:, 2] = unwind_up[:, 2] + 0.07
            left_cmd[:, 0:6] = hold_qL
            left_cmd[:, 6] = vice_gripL
            if t_in < 45:
                act = act_of_tip(unwind_up, OPEN_C)
            else:
                a = torch.zeros(n, act_dim, device=dev)
                a[:, l_s] = left_cmd
                a[:, r_s.start : r_s.start + 6] = artR.data.default_joint_pos[:, arm_ids]
                a[:, r_s.start + 6] = OPEN_C
                act = a
            qerr = (artR.data.joint_pos[:, arm_ids] - artR.data.default_joint_pos[:, arm_ids]).abs().max()
            if (t_in > 45 and float(qerr) < 0.10) or t_in >= WP_TIMEOUT:
                phase, marker = "re_press", i
        elif phase == "re_press":  # the RIGHT re-pinches the handle TOP-DOWN (jaws) at a spot
            # staggered >=55mm from the LEFT's grip and takes the weld back
            hd = handle_dir()
            hd2 = hd.clone(); hd2[:, 2] = 0.0
            hd2 = hd2 / hd2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            elbow = key.data.root_pos_w + up_axis_of(key.data.root_quat_w) * ARM_LEN
            if t_in == 1:
                sR = ((base_R_xy - elbow[:, :2]) * hd2[:, :2]).sum(-1, keepdim=True).clamp(0.020, 0.095)
                # the LEFT holds the vertical MEMBER — the whole handle is free for the right
                vice_sR[:] = (sR + (0.0, 0.018, -0.018)[min(repress_tries, 2)]).clamp(0.020, 0.095)
            hp = elbow + hd2 * vice_sR
            tcp_tgt = hp.clone()
            tcp_tgt[:, 2] = hp[:, 2] + 0.0024
            nvp = torch.stack((-hd2[:, 1], hd2[:, 0], torch.zeros(n, device=dev)), dim=-1)
            sgnRp = torch.sign(((base_R_xy - hp[:, :2]) * nvp[:, :2]).sum(-1)).unsqueeze(-1)
            sgnRp = torch.where(sgnRp == 0, torch.ones_like(sgnRp), sgnRp)
            tqRv = pinch_q(nvp * sgnRp, +1.0, math.radians(25.0))
            if t_in < 50:
                vt = tcp_tgt.clone()
                vt[:, 2] = vt[:, 2] + 0.06
                gcmd = OPEN_C
            elif t_in < 110:
                vt = tcp_tgt
                gcmd = OPEN_C
            else:
                vt = tcp_tgt
                gcmd = 0.004
            wp_q[:] = tqRv
            wp_p[:] = vt - quat_apply(tqRv, ex1) * tool_to_grip
            act = act_of(wp_p, wp_q, gcmd, rot_w=1.2)
            left_cmd[:, 0:6] = hold_qL
            left_cmd[:, 6] = vice_gripL
            if t_in >= 110 + CLOSE_STEPS and (t_in - 110 - CLOSE_STEPS) % 20 == 0:
                pr = jaw_pair(artR, jaw_ids_R)
                tzz = tip_pos("Right")
                xy_ok = (tzz[:, 0:2] - hp[:, 0:2]).norm(dim=-1) < 0.015
                okp = (pr > 0.0105) & (pr < 0.0165) & xy_ok
                print(f"    [re-press t{t_in:3d}] R-pair {float(pr[0]) * 1e3:5.1f}mm"
                      f" | xy {float((tzz[:, 0:2] - hp[:, 0:2]).norm(dim=-1)[0]) * 1e3:4.1f}mm", flush=True)
                if bool(okp.all()):
                    repress_tries = 0
                    right_hold_s[:] = vice_sR
                    grip_c = float(pr[0]) * 0.5 + 0.0005
                    weld_on()
                    weld_off_L()
                    stroke_psi.zero_()
                    stroke_y0[:] = crank_azim()
                    cycles += 1
                    print("  vice handoff back: RIGHT holds (jaw pinch), cranking", flush=True)
                    phase, marker = "hand_clear", i
            if t_in >= 3 * WP_TIMEOUT and phase == "re_press":
                if repress_tries < 2:
                    repress_tries += 1
                    print(f"  re-pinch missed; retrying via a fresh unwind ({repress_tries}/2)", flush=True)
                    phase, marker = "unwind", i
                else:
                    print("  ABORT: right re-pinch never verified", flush=True)
                    phase, marker = "retreat", i
        elif phase == "hand_clear":  # the vice hand backs off before the orbit sweeps
            hd = handle_dir()
            hd2 = hd.clone(); hd2[:, 2] = 0.0
            hd2 = hd2 / hd2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            tpl, tql = toolL_pose()
            back = tpl + ez * 0.0012  # OPEN and rise straight off the handle
            left_to(back, tql, grip=OPEN_C)
            act = hold_act(grip_c)
            if t_in >= 60:
                left_park()
                phase, marker = "clock", i  # the handoff happened at the hover: re-seat always
        elif phase == "retreat":
            if welded.any():
                weld_off()
            if welded_L.any():
                weld_off_L()
            if t_in == 1:
                wp_p[:] = tp_now
                wp_p[:, 2] = plat_z + 0.25
                wp_q[:] = tq_now
                tplr, tqlr = toolL_pose()
                wpL0_p[:] = tplr  # dwell spot: open the jaws IN PLACE first
                wpL_p[:] = tplr
                wpL_p[:, 0:2] = wpL_p[:, 0:2] + (base_L_xy - tplr[:, 0:2]) * 0.35
                wpL_p[:, 2] = wpL_p[:, 2] + 0.05
                wpL_q[:] = tqlr
            # release discipline: open fully IN PLACE (300 ticks), then exit VERTICALLY —
            # the horizontal slot cannot drag the post sideways on a pure +z path (runs
            # 141/146: lateral-first withdrawal left the seated key leaning 15-17deg)
            if t_in < 300:
                left_to(wpL0_p, wpL_q, OPEN_C)
            else:
                left_to(wpL0_p + ez * 0.05, wpL_q, OPEN_C)
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
        if phase in ("pinch_in", "pinch_close", "lift", "carry", "erect", "handoff_carry", "post_pinch",
                     "handoff", "l_clock", "l_insert", "cage_convert", "crank_approach",
                     "crank_regrip", "crank_approach_L", "crank_regrip_L", "role_swap_R_holder",
                     "role_swap_L_holder", "role_swap_L_release", "role_swap_R_release",
                     "clock", "insert") and float(dd.max()) > 0.0015:
            kt = key.data.root_pos_w
            print(f"    [watchdog] ctrl {i} {phase}: bolt depth jumped {float(dd.max()) * 1e3:+.1f}mm | "
                  f"key ({kt[0, 0]:.3f},{kt[0, 1]:.3f},{kt[0, 2]:.3f}) | tool z {tool_pose()[0][0, 2]:.3f}", flush=True)
        if phase not in ("show", "stage"):
            cur = yaw_of(bolt.data.root_quat_w)
            bolt_turn = bolt_turn - _wrap(cur - prev_bolt_yaw)
            prev_bolt_yaw = cur
            kcur = crank_azim()
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

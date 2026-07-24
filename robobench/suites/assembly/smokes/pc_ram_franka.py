"""Franka smoke for PcRamAssemblyScene — the arm installs BOTH TridentZ sticks, one after the
other, into the motherboard's dual-channel DIMM pair, the way the force-driven `pc_ram_smoke`
proved a build goes: stand the stick over its slot, line its edge connector up, and press it
straight down until it bottoms out.

Grasping follows the benchmark weld-on-closure contract (cf. `pc_gpu_franka`): a normally-disabled
FixedJoint hand<->stick is enabled when the gripper is verifiably closed around the stick's body
slab and released when it opens. The first stick is pressed to FULL DEPTH while still pinched.
The second cannot be: the slots sit 19 mm apart and a flanking fingertip is 14.3 mm deep against
the 11.6 mm inter-stick gap, so ANY pad-on-face pinch collides with the seated neighbour over the
last ~13 mm of travel. It is installed the way a person actually seats a DIMM beside another one:
pinched by its TOP EDGE (see GRIP_DEPTH_HI) and lowered, still gripped, until the slot mouth has
captured the blade — then the hand lets go, re-forms, and presses the stick home with both
FINGERTIPS on the top edge (real contact force; see TIP_W). Stick<->channel physics and each
seated stick holding its own after the hand opens are real — a missed grasp or a stalled press
fails honestly.

The `assembly.pc_ram.franka.*` env stages both sticks UPRIGHT in the scene's foam holders, already
in the seated orientation (lying flat, their only sub-80 mm dimension points up — no parallel-jaw
pinch exists; see the env registration). The far slot is inserted first, the near slot second, so
the camera never watches an insertion behind an already-standing stick — and so the top-edge,
neighbour-constrained install is only ever needed once.

Phases, per stick: pick (hover/down/close, geometry-verified) -> lift -> carry (over the 195 mm
case rim) -> drop (to the align hover over the slot) -> align (0.4 mm gate: the blade must beat
the funnel AND the 0.5 mm end-stop play) -> press (straight down, gripped; full depth for the
first stick, mouth-capture for the second) -> [second stick only: reform -> seatpress (fingertip
press to full depth), with re-tries] -> release (bleed, let go, rise) -> clear; then the next
stick, then retreat -> settle.
Verdict: per-stick seated count, blade depth vs the 4.44 mm stroke, and the residual errors after
the final settle.

The recorded video is one continuous MOVING shot per trip: it frames the active stick's holder for
the grasp, cranes across the case as the stick is carried (the blend is driven by the stick's own
progress along its holder->seat line), settles into the DIMM-cluster close-up for the drop + press,
and dollies back for the second pick.

.venv/bin/python -m robobench.suites.assembly.smokes.pc_ram_franka --headless
python -m robobench.suites.assembly.smokes.pc_ram_franka --livestream 2
python -m robobench.suites.assembly.smokes.pc_ram_franka \
    --headless --enable_cameras --video robobench/suites/assembly/videos/pc_ram_franka.mp4
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
    from robobench.suites.assembly.scenes import PcRamAssemblyScene

DT = 1.0 / 240.0  # sim timestep (matches the registered env's dt override)

# Stick-local grasp geometry (from ram_tridentz.usd `/ram/collision`; origin = blade bottom centre,
# axes = the seated/case axes; the body collider matches the visual shell). The grip is a top-down
# PINCH of the body slab's upper faces — the slab is 7.3 mm thick across x, so the fingers close
# along WORLD X and the hand's x axis runs along the stick's length (world y). On this stack the
# flat finger faces and the slab faces generate no contacts (only fingertip/chamfer features do),
# so closure is verified GEOMETRICALLY (pads flanking the slab at the grip band, fingers at the
# commanded width) and the weld contract carries the stick, as in `pc_gpu_franka`.
GRIP_TOP_ZC = 0.0401     # body-slab TOP face above the stick origin (= the visual top edge)
GRIP_XC = -0.00005       # body-slab mid-plane (faces at x -0.0037 / +0.0036)
GRIP_DEPTH = 0.005       # pad-centre depth below the slab top at the pinch (pads wrap the faces)
HOVER_CLEAR = 0.05       # pad hover height above the slab top before the descent: clears the
                         # arm's unbiased gravity sag so the tips cannot snag the stick early
# panda_finger body origin -> finger-pad centre, along the hand's approach axis (the hand-frame ->
# pad distance itself is measured live at reset: hand -> finger base + this); the finger TIP ends
# 8.8 mm past the pad centre — the pusher geometry below builds on it.
FINGER_TO_PAD = 0.045
PAD_TO_TIP = 0.0088
OPEN_W = 0.04
STRADDLE_W = 0.011       # per-finger width while descending AROUND the 7.3 mm slab (the holder
                         # rails' outer faces sit at +-12.9 mm — the open fingers stay above them)
GRIP_W = 0.0035          # per-finger closed width: the slab's half-width minus a 0.15 mm kiss,
                         # so the pads land exactly ON the stick's faces
GRIP_DEPTH_HI = -0.00575  # SECOND-stick pad-centre height: 5.75 mm ABOVE the slab top — a
                         # top-edge pinch, ~3 mm of pad on each face. The slots sit 18.96 mm
                         # apart and a fingertip is 14.3 mm deep outward with its pad face
                         # running to its very bottom, so at ANY pad-on-face grip the
                         # neighbour-side tip overhangs the seated neighbour's slab (near face
                         # 15.26 mm out) as soon as the tip is below the neighbour's top, and
                         # lands ON it (tip-bottom-on-top-face fires; pushing a pad >1 mm INTO
                         # a face fires too — measured, not the GPU card's dead pair). With the
                         # top-edge pinch the tips ride 1 mm below the gripped slab's own top,
                         # keeping the gripped press legal down to blade ~1 mm — deeper than
                         # the mouth's capture, so the pinch can hand over to the fingertips.
CAPTURE_Z = 0.0042       # gripped-press handoff height: the blade is captured by the slot mouth
                         # (proven at ~4.3 mm) and the stick stands on its own once the pinch opens
TIP_W = 0.0005           # per-finger width for the fingertip seat-press: the closed tips' bottoms
                         # overlap the stick's top edge ~3.1 mm/side, and the outer tip edge
                         # (14.8 mm) clears the seated neighbour's near face (15.26 mm)
REFORM_STEPS, SEAT_STEPS, SEAT_MAX = 42, 40, 120
# Closure window (sum of the two finger joints, m): both fingers at GRIP_W means nothing snagged
# them on the way in; the pad-position check pins the stick between them.
CLOSED_MIN, CLOSED_MAX = 0.005, 0.010

# Franka OSC action semantics (6 EE pose deltas + 2 finger position targets at 15 Hz).
POS_SCALE, ROT_SCALE = 0.02, 0.097
ROT_SAT = 8.0            # rotation lead cap (units): gravity droop stalls a 1-unit lead

# Insertion flight plan (m; heights are the stick-origin/blade-bottom z in the CASE frame).
# ORDER pairs stick k with slot k; slot 1 (far from the camera, nearer the base) goes first.
ORDER = (1, 0)
CROSS_Z = 0.240          # origin height while crossing the case rim (clears the 195 mm walls)
# Insertion heights (blade-bottom z, case frame). The stick ALIGNS at 17 mm — fine xy servoing
# needs the hand up there; lower, the arm loses its last ~1.5 mm of lateral authority, wider
# than the 1.2 mm funnel — then presses straight down GRIPPED to full depth: the stick stays
# held until it is seated, like a hand would keep hold of it.
ALIGN_Z = 0.017
PRESS_TGT = -0.002       # commanded blade z below the seated origin (standing lead — the blade
                         # needs a sustained push through the 0.15 mm/side grip band; the
                         # channel floor takes the surplus)
RESEAT_Z = 0.010         # press re-tries rise back to this blade height, still gripped
PRESS_DONE = 0.0042      # blade depth below the slot mouth to call a stick seated (stroke 4.44)
MAX_RETRIES = 2          # press re-tries per stick (rise, re-settle, press again)

# Waypoint tolerances and per-phase budgets, in CONTROL steps (15 Hz -> 16 substeps each at 1/240).
TOL_P, TOL_R = 0.004, 0.06
ALIGN_TOL = 0.0004       # stick-origin xy gate at the release hover (must beat the 1.2 mm/side
                         # funnel AND the 0.5 mm end-stop play)
ALIGN_ROT_TOL = math.radians(1.0)
SHOW_END = 20
WP_TIMEOUT, SETTLE_STEPS = 75, 45
HOVER_STEPS, DOWN_STEPS, CLOSE_STEPS = 110, 40, 18  # pick glide lengths (approach / descend / close)
LIFT_STEPS, CARRY_STEPS, RETREAT_STEPS = 60, 100, 50
DROP_STEPS, PRESS_STEPS, PRESS_MAX = 70, 60, 180
# Every long move GLIDES its commanded target (smoothstep from the phase-entry pose; the approach
# glides the wrist yaw too): a step-jump goal saturates the norm-clamped servo and the gravity-
# uncompensated arm swings wide and rings. Budgets are deliberately unhurried.
PICK_RETRIES = 3
LOG_EVERY = 45


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi


def up_axis_of(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """World direction of the body's local +z, shape (n, 3)."""
    w, x, y, z = quat_wxyz.unbind(-1)
    return torch.stack((2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)), dim=-1)


def main() -> None:
    device = getattr(args, "device", None) or ("cuda:0" if torch.cuda.is_available() else "cpu")
    robobench.discover()

    env = ENVS.get("assembly.pc_ram.franka.osc")().build(num_envs=args.num_envs, device=device)
    sc: PcRamAssemblyScene = env.scene  # type: ignore[assignment]
    n = env.num_envs
    dev = device
    case = sc.case
    art = env.robot.articulation
    hand_idx = art.body_names.index("panda_hand")
    j7 = art.joint_names.index("panda_joint7")
    fingers = art.find_joints(["panda_finger_joint.*"])[0]
    render = (not args.headless) or livestream_on
    ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)

    # ----- hand<->stick welds (the grasp contract): pre-authored, disabled FixedJoints ------------
    # PhysX latches a joint's local frames when it is FIRST enabled; frame rewrites on a re-enabled
    # joint are ignored. Each grasp therefore consumes a fresh joint from its stick's pool: author
    # the live relative pose while it is still disabled, enable it once, and on release disable it
    # for good.
    from pxr import Gf, UsdPhysics

    stage = env.stage
    WELD_POOL = 4  # per stick: one pick + retries' slack
    weld_paths: list[list[list[str]]] = []
    for e in range(n):
        base = f"/World/envs/env_{e}"
        rows = []
        for k in range(sc.cfg.num_slots):
            assert stage.GetPrimAtPath(f"{base}/Ram_{k}").HasAPI(UsdPhysics.RigidBodyAPI)
            row = []
            for j_ in range(WELD_POOL):
                j = UsdPhysics.FixedJoint.Define(stage, f"{base}/hand_ram{k}_weld_{j_}")
                j.CreateBody0Rel().SetTargets([f"{base}/Robot/panda_hand"])
                j.CreateBody1Rel().SetTargets([f"{base}/Ram_{k}"])
                j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateJointEnabledAttr(False)
                j.CreateExcludeFromArticulationAttr(True)  # a maximal-coordinate weld
                row.append(f"{base}/hand_ram{k}_weld_{j_}")
            rows.append(row)
        weld_paths.append(rows)

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

    # Open-env OSC retune: the controller carries no gravity compensation, so the stock gains
    # leave a configuration-dependent pose sag — the DIMM hover parks the hand at z ~0.17, a
    # lower/heavier configuration than pc_gpu's, so the bias budget runs +-120 mm / +-0.2 rad.
    # Damping runs slightly overdamped so transits do not ring.
    osc = env.robot.controller.controllers[0]
    osc._kp[0:3] = 400.0
    osc._kp[3:6] = 450.0
    osc._kd = 2.2 * osc._kp.sqrt()

    case_pos = case.data.root_pos_w.clone()  # (n, 3): origin ON the board face, at its centre
    board_z = case_pos[:, 2].clone()
    seats_w = [case_pos + torch.tensor(p, device=dev) for p in sc.cfg.seat_pos]  # per-slot, world

    seq = 0                 # index into ORDER; k = ORDER[seq] is the active stick/slot
    k = ORDER[seq]

    def ram():  # the active stick's asset
        return sc.rams[k]

    # ----- moving camera: one continuous shot per trip --------------------------------------------
    # Two anchor framings, blended by the ACTIVE stick's own progress along its holder->seat line:
    # a 3/4 view on its holder for the pick, craning across the case into the DIMM-cluster
    # close-up (floating inside the case-opening footprint — an outside eye low enough to face the
    # sticks is blocked by the 195 mm walls). Per-frame easing keeps the shot smooth, and eases
    # BACK to the second holder after the first insertion; a lift-follow tilt keeps the rising
    # stick in frame (pure-z motion never advances the xy-driven blend).
    cam_pose = None
    if cam is not None:
        p0 = case_pos[0]
        holders_xy = [r.data.root_pos_w[0, 0:2].clone() for r in sc.rams]  # spawn xy = the holders
        ins_eye = torch.tensor([float(p0[0]) - 0.31, float(p0[1]) - 0.18, float(p0[2]) + 0.245], device=dev)
        ins_tgt = torch.tensor([float(p0[0]) - 0.08, float(p0[1]) - 0.06, float(p0[2]) + 0.02], device=dev)
        cam_s = 0.0  # eased blend state

        def cam_pose() -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal cam_s
            hxy = holders_xy[k]
            path_v = seats_w[k][0, 0:2] - hxy
            path_len = float(path_v.norm())
            u = float((ram().data.root_pos_w[0, 0:2] - hxy) @ (path_v / path_len)) / path_len
            s = smoothstep(u)
            cam_s += 0.08 * (s - cam_s)  # ease both ways: dollies back for the second pick
            arc = math.sin(math.pi * cam_s)
            lift_h = max(0.0, float(ram().data.root_pos_w[0, 2]) - 0.15) * (1.0 - cam_s)
            pick_eye = torch.tensor([float(hxy[0]) - 0.30, float(hxy[1]) - 0.38, 0.38], device=dev)
            pick_tgt = torch.tensor([float(hxy[0]), float(hxy[1]), 0.06], device=dev)
            eye = pick_eye + (ins_eye - pick_eye) * cam_s
            tgt = pick_tgt + (ins_tgt - pick_tgt) * cam_s
            eye = eye + torch.tensor([0.0, 0.0, 0.10 * arc + 0.9 * lift_h], device=dev)
            tgt = tgt + torch.tensor([0.0, 0.0, 0.14 * arc + 1.6 * lift_h], device=dev)
            return eye.unsqueeze(0), tgt.unsqueeze(0)

    print(env.describe(), flush=True)

    # Measure hand-frame -> finger-pad-centre once from the live articulation (robust to asset edits).
    lf = art.body_names.index("panda_leftfinger")
    _hq0 = art.data.body_quat_w[:, hand_idx]
    _rel = quat_apply_inverse(_hq0, art.data.body_pos_w[:, lf] - art.data.body_pos_w[:, hand_idx])
    hand_to_pad = float(_rel[0, 2]) + FINGER_TO_PAD
    hand_to_tip = hand_to_pad + PAD_TO_TIP
    print(f"hand->pad centre: {hand_to_pad:.4f} m", flush=True)

    # ----- weld toggles + grasp transform ---------------------------------------------------------
    rel_p = torch.zeros(n, 3, device=dev)  # stick pose in the hand frame, captured at weld time
    rel_q = torch.zeros(n, 4, device=dev)
    welded = torch.zeros(n, dtype=torch.bool, device=dev)
    weld_k = [0 for _ in range(sc.cfg.num_slots)]  # next fresh joint per stick's pool

    def hand_pose() -> tuple[torch.Tensor, torch.Tensor]:
        return art.data.body_pos_w[:, hand_idx], art.data.body_quat_w[:, hand_idx]

    def weld_on() -> None:
        assert weld_k[k] < WELD_POOL, "weld pool exhausted"
        hp, hq = hand_pose()
        rel_p[:] = quat_apply_inverse(hq, ram().data.root_pos_w - hp)
        rel_q[:] = quat_mul(quat_conjugate(hq), ram().data.root_quat_w)
        for e in range(n):
            j = UsdPhysics.FixedJoint.Get(stage, weld_paths[e][k][weld_k[k]])
            p, q = rel_p[e].tolist(), rel_q[e].tolist()
            j.GetLocalPos0Attr().Set(Gf.Vec3f(p[0], p[1], p[2]))
            j.GetLocalRot0Attr().Set(Gf.Quatf(q[0], Gf.Vec3f(q[1], q[2], q[3])))
            j.GetJointEnabledAttr().Set(True)
        welded[:] = True

    def weld_off() -> None:
        for e in range(n):
            UsdPhysics.FixedJoint.Get(stage, weld_paths[e][k][weld_k[k]]).GetJointEnabledAttr().Set(False)
        weld_k[k] += 1
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

    def hand_for_stick(kp: torch.Tensor, kq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Hand waypoint that puts the WELDED stick at pose (kp, kq), via the captured grasp frame."""
        hq = quat_mul(kq, quat_conjugate(rel_q))
        return kp - quat_apply(hq, rel_p), hq

    def q_down(yaw: torch.Tensor) -> torch.Tensor:
        """Top-down hand orientation (approach -z) with the given world yaw."""
        flip = torch.zeros(n, 4, device=dev)
        flip[:, 1] = 1.0  # 180 deg about x: hand +z -> world -z
        return quat_mul(quat_from_angle_axis(yaw, ez), flip)

    def j7_after(cand: torch.Tensor) -> torch.Tensor:
        """Predicted joint-7 angle after the servo's shortest-path yaw to `cand` (valid near a
        top-down hand, where yaw routes through the wrist roll)."""
        ex = torch.zeros(n, 3, device=dev)
        ex[:, 0] = 1.0
        hx = quat_apply(hand_pose()[1], ex)
        cur = torch.atan2(hx[:, 1], hx[:, 0])
        return art.data.joint_pos[:, j7] - _wrap(cand - cur)

    def nearest_parity(psi: torch.Tensor) -> torch.Tensor:
        """The gripper is 180-deg symmetric: return psi or psi-pi, whichever leaves the wrist roll
        nearer its centre."""
        alt = _wrap(psi - math.pi)
        return torch.where(j7_after(psi).abs() <= j7_after(alt).abs(), psi, alt)

    def grip_point() -> torch.Tensor:
        """World grip point on the ACTIVE stick: the body slab's TOP face centre."""
        off = torch.tensor([GRIP_XC, 0.0, GRIP_TOP_ZC], device=dev).expand(n, 3)
        return ram().data.root_pos_w + quat_apply(ram().data.root_quat_w, off)

    def pad_centre() -> torch.Tensor:
        """World centre of the closed finger pads, along the hand's approach axis."""
        hp, hq = hand_pose()
        return hp + quat_apply(hq, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)) * hand_to_pad

    def depth() -> torch.Tensor:  # blade depth below the ACTIVE slot's mouth (m), per env
        return sc.engaged()[:, k]

    def xy_err() -> torch.Tensor:
        return (ram().data.root_pos_w[:, 0:2] - seats_w[k][:, 0:2]).norm(dim=-1)

    def rot_err() -> torch.Tensor:
        """Axis-angle error norm from the ACTIVE stick's orientation to seated (world identity)."""
        return axis_angle_from_quat(quat_conjugate(ram().data.root_quat_w)).norm(dim=-1)

    def stick_upright() -> torch.Tensor:
        return up_axis_of(ram().data.root_quat_w)[:, 2]

    # ----- step + capture -------------------------------------------------------------------------
    step_i = 0

    def step(action: torch.Tensor) -> None:
        nonlocal step_i
        step_i += 1
        capture = writer is not None and step_i % args.cap == 0
        if capture:
            cam.set_world_poses_from_view(*cam_pose())  # move the shot BEFORE this step's render
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
    picks = 0
    retries = 0
    pos_off = torch.zeros(n, 3, device=dev)   # INTEGRATED bias (desired - achieved, free air): the
    # OSC has no gravity compensation, so its realized pose sags configuration-dependently by
    # several mm — commands near contact add this learned offset so the achieved pose lands true
    rot_bias = torch.zeros(n, 3, device=dev)  # the same for ORIENTATION (axis-angle, integrated
    # in free air): the wrist's gravity moment leaves a standing stick tilt the funnel cannot eat
    grip_freeze = torch.zeros(n, 2, device=dev)  # finger targets latched at release: freezing the
    # PD at the MEASURED positions decays the squeeze before the pads separate
    grip_pt = torch.zeros(n, 3, device=dev)
    grip_yaw = torch.zeros(n, device=dev)
    close_ok = torch.zeros(n, device=dev)  # consecutive ticks the closure has verified
    wp_p = torch.zeros(n, 3, device=dev)
    wp_q = torch.zeros(n, 4, device=dev)
    hover_from = torch.zeros(n, 3, device=dev)
    hover_yaw0 = torch.zeros(n, device=dev)
    carry_from = torch.zeros(n, 3, device=dev)
    retreat_from = torch.zeros(n, 3, device=dev)
    retreat_q0 = torch.zeros(n, 4, device=dev)
    retreat_aa = torch.zeros(n, 3, device=dev)
    lift_xy = torch.zeros(n, 2, device=dev)
    lift_from = torch.zeros(n, device=dev)
    drop_from = torch.zeros(n, device=dev)
    release_p = torch.zeros(n, 3, device=dev)
    release_q = torch.zeros(n, 4, device=dev)
    press_from = torch.zeros(n, device=dev)
    reform_p = torch.zeros(n, 3, device=dev)
    reform_q = torch.zeros(n, 4, device=dev)
    seat_from = torch.zeros(n, device=dev)

    def smoothstep(t: float) -> float:
        t = min(max(t, 0.0), 1.0)
        return t * t * (3 - 2 * t)

    def upright_cmd() -> torch.Tensor:
        """Commanded stick orientation: upright (the seated identity), pre-rotated by the learned
        droop bias so the ACHIEVED orientation lands upright."""
        ang = rot_bias.norm(dim=-1).clamp_min(1e-9)
        return quat_from_angle_axis(ang, rot_bias / ang.unsqueeze(-1))

    def learn_rot(gain: float = 0.2) -> None:
        """Integrate the stick's residual orientation error (free air only): the command already
        carries the bias, so the residual drives it until the achieved pose is upright."""
        r = axis_angle_from_quat(quat_conjugate(ram().data.root_quat_w))
        rot_bias[:] = (rot_bias + gain * r).clamp(-0.2, 0.2)

    phase, marker = "show", 0
    i = 0
    while True:
        i += 1
        t_in = i - marker
        hp, hq = hand_pose()
        if home_p is None:
            home_p, home_q = hp.clone(), hq.clone()
        act = servo(home_p, home_q, OPEN_W)  # default: hold home, fingers open
        gd = GRIP_DEPTH if seq == 0 else GRIP_DEPTH_HI  # second stick: top-edge pinch


        if phase == "show":
            if t_in >= SHOW_END:
                phase, marker = "pick_hover", i
                picks += 1
        elif phase == "pick_hover":  # glide to a top-down hover over the active standing stick,
            # fingers already open wider than the slab, wrist yaw glided along with the position
            if t_in == 1:
                pos_off.zero_()  # fresh site, fresh bias
                hover_from[:] = hp
                ky = quat_apply(ram().data.root_quat_w, torch.tensor([0.0, 1.0, 0.0], device=dev).expand(n, 3))
                ex = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3)
                hx = quat_apply(hq, ex)
                hover_yaw0[:] = torch.atan2(hx[:, 1], hx[:, 0])
                grip_yaw[:] = nearest_parity(torch.atan2(ky[:, 1], ky[:, 0]))  # hand x along the
                # stick's LENGTH (world y) -> the fingers close across the 7.3 mm slab
            grip_pt[:] = grip_point()
            wp_p[:] = grip_pt
            wp_p[:, 2] = grip_pt[:, 2] + hand_to_pad + HOVER_CLEAR  # clear of the top face even
            # before the sag bias is learned
            s = smoothstep(t_in / HOVER_STEPS)
            wp_q[:] = q_down(hover_yaw0 + _wrap(grip_yaw - hover_yaw0) * s)
            if s >= 1.0:  # arrived, free air: learn the pad-centre bias for the descent
                want = grip_pt.clone()
                want[:, 2] += HOVER_CLEAR
                pos_off[:] = (pos_off + 0.3 * (want - pad_centre())).clamp(-0.12, 0.12)
            goal = wp_p + pos_off
            act = servo(hover_from + (goal - hover_from) * s, wp_q, STRADDLE_W)
            # descend only once the PAD is measured on-target: the straddle clearance is
            # ~7 mm/side to the slab and the rails sit just outside the open fingers
            pad_err = (pad_centre()[:, 0:2] - grip_pt[:, 0:2]).norm(dim=-1)
            if (t_in >= HOVER_STEPS + 10 and bool((pad_err < 0.004).all()) and bool(at(goal, wp_q).all())) \
                    or t_in >= HOVER_STEPS + 2 * WP_TIMEOUT:
                phase, marker = "pick_down", i
        elif phase == "pick_down":  # descend AROUND the stick: the open fingers pass the top edge
            # on both sides until the pads flank the slab's upper faces — free air all the way,
            # so the bias keeps learning once the glide's target goes stationary
            s = smoothstep(t_in / DOWN_STEPS)
            wp_p[:] = grip_pt
            wp_p[:, 2] = grip_pt[:, 2] + hand_to_pad + HOVER_CLEAR - s * (HOVER_CLEAR + gd)
            if s >= 1.0:
                want = grip_pt.clone()
                want[:, 2] -= gd
                pos_off[:] = (pos_off + 0.25 * (want - pad_centre())).clamp(-0.12, 0.12)
            act = servo(wp_p + pos_off, wp_q, STRADDLE_W)
            want = grip_pt.clone()
            want[:, 2] -= gd
            pad_on = (pad_centre() - want).norm(dim=-1) < 0.003
            if (t_in >= DOWN_STEPS + 8 and bool(pad_on.all())) or t_in >= 2 * WP_TIMEOUT:
                close_ok.zero_()
                phase, marker = "pick_close", i
        elif phase == "pick_close":  # close to the stick's visual width: the pads land ON the
            # faces and the finger PD holds them there; closure is verified geometrically (see
            # the grasp note above) and the weld contract then carries the stick
            s = smoothstep(t_in / CLOSE_STEPS)
            width = STRADDLE_W + (GRIP_W - STRADDLE_W) * s
            act = servo(wp_p + pos_off, wp_q, width)
            gap = art.data.joint_pos[:, fingers].sum(dim=-1)
            want = grip_pt.clone()
            want[:, 2] -= gd
            near = (pad_centre() - want).norm(dim=-1) < 0.004
            ok = near & (gap > CLOSED_MIN) & (gap < CLOSED_MAX)
            close_ok[:] = torch.where(ok, close_ok + 1, torch.zeros_like(close_ok))
            if t_in >= CLOSE_STEPS + 6 and bool((close_ok >= 4).all()):
                weld_on()
                print(f"  grasped stick {k}: pads on the slab faces, finger gap "
                      f"{[f'{float(g)*1e3:.1f}' for g in gap]} mm (slab 7.3), pad err "
                      f"{float((pad_centre() - want).norm(dim=-1).max()) * 1e3:.1f} mm", flush=True)
                phase, marker = "lift", i
            elif t_in >= CLOSE_STEPS + 30:
                if picks < PICK_RETRIES * len(ORDER):
                    print(f"  pick of stick {k} missed (gap {[f'{float(g)*1e3:.1f}' for g in gap]} mm), "
                          f"retrying", flush=True)
                    phase, marker = "pick_hover", i
                    picks += 1
                else:
                    print("  ABORT: pick failed", flush=True)
                    phase, marker = "retreat", i
        elif phase == "lift":  # straight up out of the holder to rim-crossing height (glided)
            if t_in == 1:
                lift_xy[:] = ram().data.root_pos_w[:, 0:2]  # rise only: latched, not live
                lift_from[:] = ram().data.root_pos_w[:, 2]
            s = smoothstep(t_in / LIFT_STEPS)
            kp = ram().data.root_pos_w.clone()
            kp[:, 0:2] = lift_xy
            kp[:, 2] = lift_from + s * (board_z + CROSS_Z - lift_from)
            tp, tq = hand_for_stick(kp, upright_cmd())
            act = servo(tp, tq, GRIP_W)
            if (t_in >= LIFT_STEPS and bool(((board_z + CROSS_Z - ram().data.root_pos_w[:, 2]).abs() < 0.01).all())) \
                    or t_in >= LIFT_STEPS + 45:
                pos_off.zero_()  # pick-spot bias is stale here; re-learn on the carry
                phase, marker = "carry", i
        elif phase == "carry":  # translate to above the active slot, at crossing height (glided)
            if t_in == 1:
                carry_from[:] = ram().data.root_pos_w
            goal = seats_w[k].clone()
            goal[:, 2] = board_z + CROSS_Z
            s = smoothstep(t_in / CARRY_STEPS)
            if s >= 1.0:  # arrived, free air near the case: learn the stick-frame biases
                pos_off[:] = (pos_off + 0.25 * (goal - ram().data.root_pos_w)).clamp(-0.12, 0.12)
                learn_rot()
            kp = carry_from + (goal - carry_from) * s
            tp, tq = hand_for_stick(kp + pos_off, upright_cmd())
            act = servo(tp, tq, GRIP_W)
            arrived = (ram().data.root_pos_w[:, 0:2] - seats_w[k][:, 0:2]).norm(dim=-1) < 0.003
            if (t_in >= CARRY_STEPS + 5 and bool(arrived.all())) or t_in >= CARRY_STEPS + WP_TIMEOUT:
                drop_from[:] = ram().data.root_pos_w[:, 2]
                phase, marker = "drop", i
        elif phase == "drop":  # descend over the slot to this sequence's release hover
            s = smoothstep(t_in / DROP_STEPS)
            kp = seats_w[k].clone()
            kp[:, 2] = drop_from + s * (board_z + ALIGN_Z - drop_from)
            if s >= 1.0:  # learn only once the target is stationary (mid-glide error is mostly
                # tracking lag, which would poison the integrator)
                pos_off[:] = (pos_off + 0.2 * (kp - ram().data.root_pos_w)).clamp(-0.12, 0.12)
                learn_rot()
            tp, tq = hand_for_stick(kp + pos_off, upright_cmd())
            act = servo(tp, tq, GRIP_W)
            if t_in >= DROP_STEPS + 15:
                phase, marker = "align", i
        elif phase == "align":  # settle at the align hover: the 0.4 mm gate beats the funnel
            # AND the end-stop play, so the lowered blade enters clean
            kp = seats_w[k].clone()
            kp[:, 2] = board_z + ALIGN_Z
            pos_off[:] = (pos_off + 0.1 * (kp - ram().data.root_pos_w)).clamp(-0.12, 0.12)
            learn_rot(0.05)  # gentle: rot is converged by now, and each correction sways the stick
            tp, tq = hand_for_stick(kp + pos_off, upright_cmd())
            act = servo(tp, tq, GRIP_W)
            still = ram().data.root_lin_vel_w.norm(dim=-1) < 0.01
            z_ok = (ram().data.root_pos_w[:, 2] - (board_z + ALIGN_Z)).abs() < 0.001
            ok = (xy_err() < ALIGN_TOL) & z_ok & (rot_err() < ALIGN_ROT_TOL) & still
            if bool(ok.all()) or t_in >= 6 * WP_TIMEOUT:
                print(f"  aligned stick {k}: xy err {float(xy_err().max()) * 1e3:.2f} mm, rot "
                      f"{float(torch.rad2deg(rot_err()).max()):.2f} deg", flush=True)
                press_from[:] = ram().data.root_pos_w[:, 2]
                phase, marker = "press", i
        elif phase == "press":  # straight down, STILL GRIPPED: the channel funnel guides the
            # blade while the weld carries the press force. The first stick presses to FULL
            # depth pinched; the second stops as soon as the mouth has captured the blade
            # (CAPTURE_Z) — beside the seated neighbour no pad-on-face grip may go deeper (see
            # GRIP_DEPTH_HI), so the fingertips take over from there ("reform"/"seatpress").
            s = smoothstep(t_in / PRESS_STEPS)
            z_end = board_z + sc.cfg.seat_pos[k][2] + (PRESS_TGT if seq == 0 else CAPTURE_Z - 0.0005)
            kp = seats_w[k].clone()
            kp[:, 2] = press_from + s * (z_end - press_from)
            if t_in % 3 == 0 and s < 0.6:  # free air until the blade meets the mouth: keep the
                # xy trim live so the blade arrives centred on the funnel
                pos_off[:, 0:2] = (pos_off[:, 0:2]
                                   + 0.1 * (kp[:, 0:2] - ram().data.root_pos_w[:, 0:2])).clamp(-0.12, 0.12)
            tp, tq = hand_for_stick(kp + pos_off, upright_cmd())
            act = servo(tp, tq, GRIP_W)
            bz = ram().data.root_pos_w[:, 2] - (board_z + sc.cfg.seat_pos[k][2])
            if seq == 0 and bool((depth() >= PRESS_DONE).all()):
                phase, marker = "release", i
            elif seq > 0 and t_in >= 10 and bool((bz <= CAPTURE_Z + 0.0002).all()):
                print(f"  stick {k} captured: blade {float(bz.mean() * 1e3):.2f} mm in the mouth, "
                      f"handing over to the fingertips", flush=True)
                phase, marker = "reform", i
            elif t_in >= PRESS_MAX:
                if retries < MAX_RETRIES:
                    retries += 1
                    print(f"  WARN: press on stick {k} stalled at {float(depth().mean() * 1e3):+.2f} mm "
                          f"— retry {retries}/{MAX_RETRIES}", flush=True)
                    phase, marker = "reseat", i
                else:
                    print(f"  WARN: press on stick {k} exhausted its retries — releasing as-is", flush=True)
                    phase, marker = "release", i
        elif phase == "reseat":  # rise back to just above the mouth, still gripped, and re-press
            kp = seats_w[k].clone()
            kp[:, 2] = board_z + RESEAT_Z
            tp, tq = hand_for_stick(kp + pos_off, upright_cmd())
            act = servo(tp, tq, GRIP_W)
            if t_in >= WP_TIMEOUT // 2:
                press_from[:] = ram().data.root_pos_w[:, 2]
                phase, marker = "press", i
        elif phase == "reform":  # the stick stands captured in the slot mouth: bleed the press,
            # let go, lift a few mm, and re-form the hand into a two-fingertip press hovering
            # over the stick's TOP EDGE — the way a person seats a DIMM beside another one.
            if t_in == 1:
                grip_freeze[:] = art.data.joint_pos[:, fingers]
                reform_p[:] = hp
                reform_q[:] = hq
            if t_in <= 6:
                act = servo(reform_p, reform_q, grip_freeze)
            elif t_in <= 24:  # let go and rise off the pinch, fingers opening clear of the stick
                if welded.any():
                    weld_off()
                s = smoothstep((t_in - 6) / 14.0)
                wp_p[:] = reform_p
                wp_p[:, 2] = reform_p[:, 2] + s * 0.008
                act = servo(wp_p, reform_q, STRADDLE_W)
            else:  # close the tips over the top edge, hovering 1.5 mm above it
                if t_in == 25:
                    hover_from[:] = wp_p
                tip_p = grip_point()
                tip_p[:, 2] = tip_p[:, 2] + 0.0015 + hand_to_tip
                s = smoothstep((t_in - 24) / (REFORM_STEPS - 24.0))
                wp_p[:] = hover_from + (tip_p - hover_from) * s
                if s >= 1.0 and t_in % 2 == 0:  # trim the hand's own droop onto the edge
                    pos_off[:, 0:2] = (pos_off[:, 0:2] + 0.15 * (tip_p[:, 0:2] - hp[:, 0:2])).clamp(-0.12, 0.12)
                w = STRADDLE_W + (TIP_W - STRADDLE_W) * s
                act = servo(wp_p + pos_off, reform_q, w)
                if (t_in >= REFORM_STEPS + 10
                        and bool(((tip_p[:, 0:2] - hp[:, 0:2]).norm(dim=-1) < 0.0012).all()))                         or t_in >= 2 * WP_TIMEOUT:
                    phase, marker = "seatpress", i
        elif phase == "seatpress":  # drive the fingertips down on the top edge until the blade
            # bottoms out: REAL contact seats the stick (tip-bottom-on-top-face is a live pair)
            # while the channel keeps the blade centred; the tips' outer edges clear the
            # neighbour all the way (see TIP_W)
            if t_in == 1:
                seat_from[:] = hp[:, 2]
            z_end = board_z + sc.cfg.seat_pos[k][2] + GRIP_TOP_ZC + PRESS_TGT + hand_to_tip
            s = smoothstep(t_in / SEAT_STEPS)
            wp_p[:] = seats_w[k]
            wp_p[:, 2] = seat_from + s * (z_end - seat_from)
            if t_in % 2 == 0:
                pos_off[:, 0:2] = (pos_off[:, 0:2] + 0.1 * (wp_p[:, 0:2] - hp[:, 0:2])).clamp(-0.12, 0.12)
            cmd = wp_p.clone()
            cmd[:, 0:2] += pos_off[:, 0:2]
            act = servo(cmd, reform_q, TIP_W)
            if bool((depth() >= PRESS_DONE).all()):
                phase, marker = "release", i
            elif t_in >= SEAT_MAX:
                if retries < MAX_RETRIES:
                    retries += 1
                    print(f"  WARN: seat-press on stick {k} stalled at {float(depth().mean() * 1e3):+.2f} mm "
                          f"— retry {retries}/{MAX_RETRIES}", flush=True)
                    phase, marker = "reform", i
                else:
                    print(f"  WARN: seat-press on stick {k} exhausted its retries — releasing as-is", flush=True)
                    phase, marker = "release", i
        elif phase == "release":  # the stick is seated: bleed the stored press through the weld,
            # let go, rise clear. Cutting the weld while the servo still presses pops the arm
            # back with the fingers dragging the stick — so first re-target the LIVE pose with
            # the finger PD frozen at its measured positions, then disable the weld, open, rise.
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
                if t_in > 14:  # rise first (glided): the fingers back off the seated top edge
                    s2 = smoothstep((t_in - 14) / 16.0)
                    wp2[:, 2] = release_p[:, 2] + s2 * 0.045
                if t_in > 26:  # then peel WEST while still rising — transverse to the insert
                    # view's axis, so the hand visibly exits sideways off the seated stick
                    s3 = smoothstep((t_in - 26) / 14.0)
                    wp2[:, 0] = release_p[:, 0] - s3 * 0.035
                # spread only once the hand is a few mm up: the fingertip sweep then passes
                # ABOVE both seated sticks instead of across the neighbour's top edge
                act = servo(wp2, release_q, STRADDLE_W if t_in > 20 else grip_freeze)
            if t_in >= 44:
                phase, marker = "clear", i
        elif phase == "clear":  # rise straight off the seated stick, then the next stick/retreat
            if t_in == 1:
                drop_from[:] = hp[:, 2]
                wp_p[:] = hp  # hold the peeled-away xy; rise only
            s = smoothstep(t_in / 30.0)
            kp = wp_p.clone()
            kp[:, 2] = drop_from + s * (board_z + CROSS_Z + hand_to_tip - drop_from)
            act = servo(kp, release_q, OPEN_W)
            if t_in >= 34:
                print(f"  stick {k} pressed: depth {float(depth().mean() * 1e3):+.2f} mm, xy "
                      f"{float(xy_err().mean() * 1e3):.2f} mm", flush=True)
                seq += 1
                if seq < len(ORDER):
                    k = ORDER[seq]
                    retries = 0
                    picks += 1
                    phase, marker = "pick_hover", i
                else:
                    phase, marker = "retreat", i
        elif phase == "retreat":  # glide home — position AND orientation (a step-jump
            # orientation target would snap the wrist)
            if t_in == 1:
                if welded.any():
                    weld_off()
                retreat_from[:] = hp
                retreat_q0[:] = hq
                qe = quat_mul(home_q, quat_conjugate(hq))
                qe = torch.where(qe[:, :1] >= 0, qe, -qe)
                retreat_aa[:] = axis_angle_from_quat(qe)
            s = smoothstep(t_in / RETREAT_STEPS)
            ang = retreat_aa.norm(dim=-1).clamp_min(1e-9)
            q_cmd = quat_mul(quat_from_angle_axis(ang * s, retreat_aa / ang.unsqueeze(-1)), retreat_q0)
            act = servo(retreat_from + (home_p - retreat_from) * s, q_cmd, OPEN_W)
            if t_in >= RETREAT_STEPS + 10:
                phase, marker = "settle", i
        else:  # settle: hands off — both seated sticks must hold on their own
            act = servo(home_p, home_q, OPEN_W)
            if t_in >= SETTLE_STEPS:
                break

        step(act)

        bad = not torch.isfinite(ram().data.root_pos_w).all()
        if bad:
            print("  ABORT: stick state went non-finite", flush=True)
            break
        if i % LOG_EVERY == 0:
            gap = art.data.joint_pos[:, fingers].sum(dim=-1) * 1e3
            print(f"  ctrl {i:4d} [{phase:9s} stick {k}] | depth {float(depth().mean() * 1e3):+6.2f}mm | "
                  f"xy err {float(xy_err().mean() * 1e3):6.2f}mm | rot {float(torch.rad2deg(rot_err()).max()):5.2f}deg | "
                  f"stick z {float(ram().data.root_pos_w[:, 2].mean()):.3f} | hand z {float(hp[:, 2].mean()):.3f} | "
                  f"gap {float(gap.mean()):4.1f}mm | bias ({float(pos_off[0, 0]) * 1e3:+.0f},"
                  f"{float(pos_off[0, 1]) * 1e3:+.0f},{float(pos_off[0, 2]) * 1e3:+.0f})mm", flush=True)

    if writer is not None:
        writer.close()
        print("MP4:", args.video, flush=True)

    seated = sc.seated()  # (n, S) — re-checked for BOTH sticks after the final settle
    all_ok = seated.all(dim=1)
    depths = sc.engaged()
    per_slot = " | ".join(
        f"slot{j}: depth {float(depths[:, j].mean() * 1e3):+.2f} mm, xy "
        f"{float((sc.rams[j].data.root_pos_w[:, 0:2] - seats_w[j][:, 0:2]).norm(dim=-1).mean() * 1e3):.2f} mm"
        for j in range(sc.cfg.num_slots)
    )
    print(f"PC-RAM-FRANKA | seated {int(all_ok.sum())}/{n} envs ({int(seated.sum())}/{n * sc.cfg.num_slots} "
          f"sticks) | stroke 4.44, seat >= {sc.cfg.seat_depth * 1e3:.1f} | {per_slot} | "
          f"{picks} picks, {retries} press retries", flush=True)
    close_and_exit(env, app)


if __name__ == "__main__":
    main()

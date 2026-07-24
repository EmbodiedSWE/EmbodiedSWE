"""Franka smoke for PcGpuRamAssemblyScene — the arm builds the PC: it installs the RTX 2060 in
the PCIe x16 slot FIRST, then presses both TridentZ sticks into the motherboard's dual-channel
DIMM pair, each part the way its single-task smoke proved a build goes.

The card (cf. `pc_gpu_franka`): picked out of its foam holder by a top-down pinch of its body
slab, carried over the case rim, lowered inside the case with the I/O bracket just forward of the
rear panel, slid rearward until the bracket and ports pass through the expansion-slot cutout,
then pressed straight down to seat. The sticks (cf. `pc_ram_franka`): each picked out of its foam
holder, carried across, aligned over its slot and pressed straight down. The first stick presses
to full depth while pinched; the second stands beside its seated neighbour, where no pad-on-face
pinch may follow it home (the slots sit 19 mm apart and a fingertip is 14.3 mm deep against the
11.6 mm inter-stick gap), so it is installed the way a person seats a DIMM beside another one:
pinched by its TOP EDGE, lowered gripped until the slot mouth captures the blade, then let go,
the hand re-formed, and pressed home with both FINGERTIPS on the top edge.

Grasping follows the benchmark weld-on-closure contract: a normally-disabled FixedJoint
hand<->part is enabled when the gripper is verifiably closed around the part's body slab
(closure verified GEOMETRICALLY: pads flanking the slab at the grip band, fingers at the
commanded width) and released when it opens. Everything else is live physics — part<->channel,
card<->rear-panel frame, and every released part holding its seat under gravity — so a missed
grasp, a jammed slide, or a stalled press fails honestly.

The `assembly.pc_gpu_ram.franka.*` env stages all three parts UPRIGHT in foam holders, already
in their seated orientations (lying flat, each part's only sub-80 mm dimension points up — no
parallel-jaw pinch exists; see the env registration). The card goes first, so its holder (north
of the stick holders, dead-ahead of the base) is empty before any stick flies; the far DIMM slot
is filled before the near one, so the camera never watches an insertion behind an
already-standing stick.

Phases, per part: pick (hover/down/close, geometry-verified) -> lift -> carry (over the 195 mm
case rim) -> drop -> align -> [card only: slide (rearward through the cutout)] -> press
(straight down, gripped; full depth for the card and the first stick, mouth-capture for the
second stick) -> [second stick only: reform -> seatpress (fingertip press to full depth)] ->
release (bleed, let go, rise) -> clear; then the next part, then retreat -> settle.
Verdict: per-part seated flags, insertion depths vs their strokes (card 5.0 mm, sticks 4.44 mm),
and the residual errors after the final settle.

The recorded video is one continuous MOVING shot per trip: it frames the active part's holder
for the grasp, cranes across the case as the part is carried (the blend is driven by the part's
own progress along its holder->seat line), settles into that part's slot close-up for the
insertion, and dollies back for the next pick.

.venv/bin/python -m robobench.suites.assembly.smokes.pc_gpu_ram_franka --headless
python -m robobench.suites.assembly.smokes.pc_gpu_ram_franka --livestream 2
python -m robobench.suites.assembly.smokes.pc_gpu_ram_franka \
    --headless --enable_cameras --video robobench/suites/assembly/videos/pc_gpu_ram_franka.mp4
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
    from robobench.suites.assembly.scenes import PcGpuRamAssemblyScene

DT = 1.0 / 240.0  # sim timestep (matches the registered env's dt override)

# ----- card grasp + flight plan (pc_gpu_franka's proven values; case-frame heights) --------------
# The card grip is a top-down PINCH of its 34.8 mm body slab's upper faces, centred above the PCB
# tab — the fingers close along WORLD Y and the hand's x axis runs along the card's length.
CARD_GRIP_TOP_ZC = 0.1155  # body-slab TOP face above the card origin (= the shroud top edge)
CARD_GRIP_YC = 0.0154      # body-slab mid-plane (faces at y -0.002 / +0.0328)
CARD_GRIP_DEPTH = 0.015    # pad-centre depth below the slab top at the pinch
CARD_STRADDLE_W = 0.022    # per-finger width while descending AROUND the slab
CARD_GRIP_W = 0.0172       # per-finger closed width: the slab's visual half-width minus a kiss
CARD_CLOSED_MIN, CARD_CLOSED_MAX = 0.030, 0.039  # closure window (finger-joint sum, m)
SLIDE_OFF = 0.028          # forward (-x) placement offset: bracket clear of the rear panel
SLIDE_Z = 0.0172           # placement/slide height: tab 1.7 mm above the channel walls, bracket
                           # 1.1 mm under the cutout top
CARD_PRESS_DONE = 0.0048   # tab depth below the PCIe mouth to call the card seated (stroke 5 mm)
CARD_PLACE_TOL = 0.0008    # card-origin xy gate at the placement point (funnel absorbs 1.2 mm/side)
CARD_HOVER, CARD_DOWN, CARD_CLOSE = 140, 45, 20
CARD_CARRY, SLIDE_STEPS = 120, 100
CARD_PRESS_STEPS, CARD_PRESS_MAX = 75, 200

# ----- stick grasp + flight plan (pc_ram_franka's proven values) ---------------------------------
# A stick grip is a top-down PINCH of its 7.3 mm body slab's upper faces — the fingers close
# along WORLD X and the hand's x axis runs along the stick's length. The pads stop at a light
# kiss of the faces; closure is verified geometrically and the weld contract carries the part.
RAM_GRIP_TOP_ZC = 0.0401   # body-slab TOP face above the stick origin (= the visual top edge)
RAM_GRIP_XC = -0.00005     # body-slab mid-plane (faces at x -0.0037 / +0.0036)
RAM_GRIP_DEPTH = 0.005     # FIRST-stick pad-centre depth below the slab top at the pinch
RAM_GRIP_DEPTH_HI = -0.00575  # SECOND-stick pad-centre height: 5.75 mm ABOVE the slab top — a
                           # top-edge pinch with ~3 mm of pad on each face. The slots sit
                           # 18.96 mm apart and a fingertip reaches 14.3 mm outward with its pad
                           # face running to its very bottom, so any pad-on-face grip overhangs
                           # the seated neighbour's slab (near face 15.26 mm out) once the tips
                           # are below the neighbour's top. With the top-edge pinch the tips ride
                           # just 1 mm below the gripped slab's own top, keeping the gripped
                           # press clear of the neighbour down to blade ~1 mm — deeper than the
                           # mouth's capture height, so the pinch can hand the stick to the
                           # fingertips.
RAM_STRADDLE_W = 0.011     # per-finger width while descending AROUND the slab (the holder
                           # rails' outer faces sit at +-12.9 mm — the open fingers stay above)
RAM_GRIP_W = 0.0035        # per-finger closed width: the slab's half-width minus a 0.15 mm kiss
RAM_CLOSED_MIN, RAM_CLOSED_MAX = 0.005, 0.010  # closure window (finger-joint sum, m)
ALIGN_Z = 0.017            # stick align hover (blade-bottom z, case frame): fine xy servoing
                           # needs the hand up here — lower, the arm loses its last ~1.5 mm of
                           # lateral authority, wider than the 1.2 mm funnel
CAPTURE_Z = 0.0042         # gripped-press handoff height: the slot mouth has captured the blade,
                           # so the stick stands on its own once the pinch opens
TIP_W = 0.0005             # per-finger width for the fingertip seat-press: the closed tips'
                           # bottoms overlap the stick's top edge ~3.1 mm/side, and the outer tip
                           # edge (14.8 mm) clears the seated neighbour's near face (15.26 mm)
RAM_PRESS_DONE = 0.0042    # blade depth below a DIMM mouth to call a stick seated (stroke 4.44)
RESEAT_Z = 0.010           # stick press re-tries rise back to this blade height, still gripped
RAM_ALIGN_TOL = 0.0004     # stick-origin xy gate at the align hover (beats the 1.2 mm/side
                           # funnel AND the 0.5 mm end-stop play)
ALIGN_ROT_TOL = math.radians(1.0)
RAM_HOVER, RAM_DOWN, RAM_CLOSE = 110, 40, 18
RAM_CARRY = 100
RAM_PRESS_STEPS, RAM_PRESS_MAX = 60, 180
REFORM_STEPS, SEAT_STEPS, SEAT_MAX = 42, 40, 120  # re-form glide / seat-press glide / press budget
RAM_ORDER = (1, 0)         # far slot first, so the near insertion is never behind a stick

# ----- shared -------------------------------------------------------------------------------------
HOVER_CLEAR = 0.05         # pad hover height above a part's top before the descent: clears the
                           # arm's unbiased gravity sag so the tips cannot snag the part early
FINGER_TO_PAD = 0.045      # panda_finger body origin -> finger-pad centre, along the approach
PAD_TO_TIP = 0.0088        # the finger TIP ends this far past the pad centre
OPEN_W = 0.04
POS_SCALE, ROT_SCALE = 0.02, 0.097  # Franka OSC action semantics (6 EE deltas + 2 fingers, 15 Hz)
ROT_SAT = 8.0              # rotation lead cap (units): gravity droop stalls a 1-unit lead
CROSS_Z = 0.240            # part-origin height while crossing the case rim (clears 195 mm walls)
PRESS_TGT = -0.002         # commanded origin z below the seated point during a press (standing
                           # lead — the soft OSC servo needs one; the channel floor takes it)
MAX_RETRIES = 2            # press re-tries per part (rise, re-settle, press again)
TOL_P, TOL_R = 0.004, 0.06
SHOW_END = 20
WP_TIMEOUT, SETTLE_STEPS = 75, 45
LIFT_STEPS, DROP_STEPS, RETREAT_STEPS = 70, 70, 50
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

    env = ENVS.get("assembly.pc_gpu_ram.franka.osc")().build(num_envs=args.num_envs, device=device)
    sc: PcGpuRamAssemblyScene = env.scene  # type: ignore[assignment]
    n = env.num_envs
    dev = device
    case, card = sc.case, sc.card
    art = env.robot.articulation
    hand_idx = art.body_names.index("panda_hand")
    j7 = art.joint_names.index("panda_joint7")
    fingers = art.find_joints(["panda_finger_joint.*"])[0]
    render = (not args.headless) or livestream_on
    ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)

    # ----- hand<->part welds (the grasp contract): pre-authored, disabled FixedJoints -------------
    # PhysX latches a joint's local frames when it is FIRST enabled; frame rewrites on a re-enabled
    # joint are ignored. Each grasp therefore consumes a fresh joint from its part's pool: author
    # the live relative pose while it is still disabled, enable it once, and on release disable it
    # for good.
    from pxr import Gf, UsdPhysics

    stage = env.stage
    WELD_POOL = 4  # per part: one pick + retries' slack
    BODIES = ["Card"] + [f"Ram_{k}" for k in range(sc.cfg.num_slots)]
    weld_paths: list[dict[str, list[str]]] = []
    for e in range(n):
        base = f"/World/envs/env_{e}"
        pools: dict[str, list[str]] = {}
        for body in BODIES:
            assert stage.GetPrimAtPath(f"{base}/{body}").HasAPI(UsdPhysics.RigidBodyAPI)
            row = []
            for j_ in range(WELD_POOL):
                j = UsdPhysics.FixedJoint.Define(stage, f"{base}/hand_{body.lower()}_weld_{j_}")
                j.CreateBody0Rel().SetTargets([f"{base}/Robot/panda_hand"])
                j.CreateBody1Rel().SetTargets([f"{base}/{body}"])
                j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateJointEnabledAttr(False)
                j.CreateExcludeFromArticulationAttr(True)  # a maximal-coordinate weld
                row.append(f"{base}/hand_{body.lower()}_weld_{j_}")
            pools[body] = row
        weld_paths.append(pools)

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
    # leave a configuration-dependent pose sag; the stiffer gains bring it into the learned bias'
    # budget, and damping runs slightly overdamped so transits do not ring.
    osc = env.robot.controller.controllers[0]
    osc._kp[0:3] = 400.0
    osc._kp[3:6] = 450.0
    osc._kd = 2.2 * osc._kp.sqrt()

    case_pos = case.data.root_pos_w.clone()  # (n, 3): origin ON the board face, at its centre
    board_z = case_pos[:, 2].clone()
    gpu_seat_w = case_pos + torch.tensor(sc.cfg.gpu_seat_pos, device=dev)  # seated card origin
    place_w = gpu_seat_w.clone()  # card placement point: bracket forward of the rear panel
    place_w[:, 0] -= SLIDE_OFF
    seats_w = [case_pos + torch.tensor(p, device=dev) for p in sc.cfg.ram_seat_pos]  # per-slot

    # Stage sequencing: seq 0 installs the card, seq 1/2 the sticks in RAM_ORDER.
    seq = 0
    is_card = True
    k = -1  # active DIMM slot (ram stages only)
    ram_i = -1  # 0-based stick sequence (ram stages only; 1 = beside the seated neighbour)

    def part():  # the active part's asset
        return card if is_card else sc.rams[k]

    def seat_p() -> torch.Tensor:  # the active part's seated origin, world
        return gpu_seat_w if is_card else seats_w[k]

    # ----- moving camera: one continuous shot per trip --------------------------------------------
    # Two anchor framings per trip, blended by the ACTIVE part's own progress along its
    # holder->seat line: a 3/4 view on its holder for the pick, craning across the case into that
    # part's slot close-up (the PCIe view from the east, the DIMM view floating inside the
    # case-opening footprint). Per-frame easing keeps the shot smooth and eases BACK to the next
    # holder after each insertion; a lift-follow tilt keeps a rising part in frame.
    cam_pose = None
    if cam is not None:
        p0 = case_pos[0]
        holders_xy = [card.data.root_pos_w[0, 0:2].clone()] + [
            sc.rams[j].data.root_pos_w[0, 0:2].clone() for j in RAM_ORDER
        ]  # spawn xy = the holders, in stage order (card, first stick, second stick)
        gpu_eye = torch.tensor([float(p0[0]) + 0.13, float(p0[1]) - 0.26, float(p0[2]) + 0.40], device=dev)
        gpu_tgt = torch.tensor([float(p0[0]) + 0.03, float(p0[1]) + 0.02, float(p0[2]) + 0.075], device=dev)
        ram_eye = torch.tensor([float(p0[0]) - 0.31, float(p0[1]) - 0.18, float(p0[2]) + 0.245], device=dev)
        ram_tgt = torch.tensor([float(p0[0]) - 0.08, float(p0[1]) - 0.06, float(p0[2]) + 0.02], device=dev)
        cam_s = 0.0  # eased blend state

        def cam_pose() -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal cam_s
            sq = min(seq, 2)
            hxy = holders_xy[sq]
            spx = (gpu_seat_w if sq == 0 else seats_w[RAM_ORDER[sq - 1]])[0, 0:2]
            path_v = spx - hxy
            path_len = float(path_v.norm())
            u = float((part().data.root_pos_w[0, 0:2] - hxy) @ (path_v / path_len)) / path_len
            s = smoothstep(u)
            cam_s += 0.08 * (s - cam_s)  # ease both ways: dollies back for the next pick
            arc = math.sin(math.pi * cam_s)
            lift_h = max(0.0, float(part().data.root_pos_w[0, 2]) - 0.15) * (1.0 - cam_s)
            if sq == 0:
                pick_eye = torch.tensor([float(hxy[0]) - 0.35, float(hxy[1]) - 0.42, 0.40], device=dev)
                pick_tgt = torch.tensor([float(hxy[0]), float(hxy[1]), 0.12], device=dev)
                ins_eye, ins_tgt, arc_e, arc_t = gpu_eye, gpu_tgt, 0.12, 0.16
            else:
                pick_eye = torch.tensor([float(hxy[0]) - 0.30, float(hxy[1]) - 0.38, 0.38], device=dev)
                pick_tgt = torch.tensor([float(hxy[0]), float(hxy[1]), 0.06], device=dev)
                ins_eye, ins_tgt, arc_e, arc_t = ram_eye, ram_tgt, 0.10, 0.14
            eye = pick_eye + (ins_eye - pick_eye) * cam_s
            tgt = pick_tgt + (ins_tgt - pick_tgt) * cam_s
            eye = eye + torch.tensor([0.0, 0.0, arc_e * arc + 0.9 * lift_h], device=dev)
            tgt = tgt + torch.tensor([0.0, 0.0, arc_t * arc + 1.6 * lift_h], device=dev)
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
    rel_p = torch.zeros(n, 3, device=dev)  # part pose in the hand frame, captured at weld time
    rel_q = torch.zeros(n, 4, device=dev)
    welded = torch.zeros(n, dtype=torch.bool, device=dev)
    weld_k = {body: 0 for body in BODIES}  # next fresh joint per part's pool

    def active_body() -> str:
        return "Card" if is_card else f"Ram_{k}"

    def hand_pose() -> tuple[torch.Tensor, torch.Tensor]:
        return art.data.body_pos_w[:, hand_idx], art.data.body_quat_w[:, hand_idx]

    def weld_on() -> None:
        body = active_body()
        assert weld_k[body] < WELD_POOL, "weld pool exhausted"
        hp, hq = hand_pose()
        rel_p[:] = quat_apply_inverse(hq, part().data.root_pos_w - hp)
        rel_q[:] = quat_mul(quat_conjugate(hq), part().data.root_quat_w)
        for e in range(n):
            j = UsdPhysics.FixedJoint.Get(stage, weld_paths[e][body][weld_k[body]])
            p, q = rel_p[e].tolist(), rel_q[e].tolist()
            j.GetLocalPos0Attr().Set(Gf.Vec3f(p[0], p[1], p[2]))
            j.GetLocalRot0Attr().Set(Gf.Quatf(q[0], Gf.Vec3f(q[1], q[2], q[3])))
            j.GetJointEnabledAttr().Set(True)
        welded[:] = True

    def weld_off() -> None:
        body = active_body()
        for e in range(n):
            UsdPhysics.FixedJoint.Get(stage, weld_paths[e][body][weld_k[body]]).GetJointEnabledAttr().Set(False)
        weld_k[body] += 1
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

    def hand_for_part(kp: torch.Tensor, kq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Hand waypoint that puts the WELDED part at pose (kp, kq), via the captured grasp frame."""
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
        """World grip point on the ACTIVE part: its body slab's TOP face centre."""
        if is_card:
            off = torch.tensor([0.0, CARD_GRIP_YC, CARD_GRIP_TOP_ZC], device=dev).expand(n, 3)
        else:
            off = torch.tensor([RAM_GRIP_XC, 0.0, RAM_GRIP_TOP_ZC], device=dev).expand(n, 3)
        return part().data.root_pos_w + quat_apply(part().data.root_quat_w, off)

    def pad_centre() -> torch.Tensor:
        """World centre of the closed finger pads, along the hand's approach axis."""
        hp, hq = hand_pose()
        return hp + quat_apply(hq, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)) * hand_to_pad

    def depth() -> torch.Tensor:  # the ACTIVE part's depth below its slot mouth (m), per env
        return sc.gpu_engaged() if is_card else sc.ram_engaged()[:, k]

    def xy_err(ref: torch.Tensor | None = None) -> torch.Tensor:
        ref = seat_p() if ref is None else ref
        return (part().data.root_pos_w[:, 0:2] - ref[:, 0:2]).norm(dim=-1)

    def rot_err() -> torch.Tensor:
        """Axis-angle error norm from the ACTIVE part's orientation to seated (world identity)."""
        return axis_angle_from_quat(quat_conjugate(part().data.root_quat_w)).norm(dim=-1)

    def part_upright() -> torch.Tensor:
        return up_axis_of(part().data.root_quat_w)[:, 2]

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
    stage_picks = 0
    retries = 0
    total_retries = 0
    pos_off = torch.zeros(n, 3, device=dev)   # INTEGRATED bias (desired - achieved, free air): the
    # OSC has no gravity compensation, so its realized pose sags configuration-dependently by
    # several mm — commands near contact add this learned offset so the achieved pose lands true
    rot_bias = torch.zeros(n, 3, device=dev)  # the same for ORIENTATION (axis-angle, integrated
    # in free air): the wrist's gravity moment leaves a standing part tilt the funnel cannot eat
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
        """Commanded part orientation: upright (the seated identity), pre-rotated by the learned
        droop bias so the ACHIEVED orientation lands upright."""
        ang = rot_bias.norm(dim=-1).clamp_min(1e-9)
        return quat_from_angle_axis(ang, rot_bias / ang.unsqueeze(-1))

    def learn_rot(gain: float = 0.2) -> None:
        """Integrate the part's residual orientation error (free air only): the command already
        carries the bias, so the residual drives it until the achieved pose is upright."""
        r = axis_angle_from_quat(quat_conjugate(part().data.root_quat_w))
        rot_bias[:] = (rot_bias + gain * r).clamp(-0.2, 0.2)

    def label() -> str:
        return "card" if is_card else f"stick {k}"

    phase, marker = "show", 0
    i = 0
    while True:
        i += 1
        t_in = i - marker
        hp, hq = hand_pose()
        if home_p is None:
            home_p, home_q = hp.clone(), hq.clone()
        act = servo(home_p, home_q, OPEN_W)  # default: hold home, fingers open

        # Active part's grasp/flight parameters (the second stick uses the top-edge pinch).
        if is_card:
            gd, w_strad, w_grip = CARD_GRIP_DEPTH, CARD_STRADDLE_W, CARD_GRIP_W
            c_min, c_max, slab = CARD_CLOSED_MIN, CARD_CLOSED_MAX, 34.8
            hover_steps, down_steps, close_steps, carry_steps = CARD_HOVER, CARD_DOWN, CARD_CLOSE, CARD_CARRY
            press_steps, press_max, press_done = CARD_PRESS_STEPS, CARD_PRESS_MAX, CARD_PRESS_DONE
        else:
            gd = RAM_GRIP_DEPTH if ram_i == 0 else RAM_GRIP_DEPTH_HI
            w_strad, w_grip = RAM_STRADDLE_W, RAM_GRIP_W
            c_min, c_max, slab = RAM_CLOSED_MIN, RAM_CLOSED_MAX, 7.3
            hover_steps, down_steps, close_steps, carry_steps = RAM_HOVER, RAM_DOWN, RAM_CLOSE, RAM_CARRY
            press_steps, press_max, press_done = RAM_PRESS_STEPS, RAM_PRESS_MAX, RAM_PRESS_DONE

        if phase == "show":
            if t_in >= SHOW_END:
                phase, marker = "pick_hover", i
                picks += 1
                stage_picks += 1
        elif phase == "pick_hover":  # glide to a top-down hover over the active standing part,
            # fingers already open wider than its slab, wrist yaw glided along with the position
            if t_in == 1:
                pos_off.zero_()  # fresh site, fresh bias
                hover_from[:] = hp
                ax = torch.zeros(n, 3, device=dev)
                ax[:, 0 if is_card else 1] = 1.0  # the part's length axis (card x / stick y)
                kv = quat_apply(part().data.root_quat_w, ax)
                ex = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3)
                hx = quat_apply(hq, ex)
                hover_yaw0[:] = torch.atan2(hx[:, 1], hx[:, 0])
                grip_yaw[:] = nearest_parity(torch.atan2(kv[:, 1], kv[:, 0]))  # hand x along the
                # part's LENGTH -> the fingers close across its body slab
            grip_pt[:] = grip_point()
            wp_p[:] = grip_pt
            wp_p[:, 2] = grip_pt[:, 2] + hand_to_pad + HOVER_CLEAR  # clear of the top face even
            # before the sag bias is learned
            s = smoothstep(t_in / hover_steps)
            wp_q[:] = q_down(hover_yaw0 + _wrap(grip_yaw - hover_yaw0) * s)
            if s >= 1.0:  # arrived, free air: learn the pad-centre bias for the descent
                want = grip_pt.clone()
                want[:, 2] += HOVER_CLEAR
                pos_off[:] = (pos_off + 0.3 * (want - pad_centre())).clamp(-0.12, 0.12)
            goal = wp_p + pos_off
            act = servo(hover_from + (goal - hover_from) * s, wp_q, w_strad)
            # descend only once the PAD is measured on-target: the straddle clearance per side is
            # only a few mm and the holder rails sit just outside the open fingers
            pad_err = (pad_centre()[:, 0:2] - grip_pt[:, 0:2]).norm(dim=-1)
            if (t_in >= hover_steps + 10 and bool((pad_err < 0.004).all()) and bool(at(goal, wp_q).all())) \
                    or t_in >= hover_steps + 2 * WP_TIMEOUT:
                phase, marker = "pick_down", i
        elif phase == "pick_down":  # descend AROUND the part: the open fingers pass the top edge
            # on both sides until the pads flank the slab's upper faces — free air all the way,
            # so the bias keeps learning once the glide's target goes stationary
            s = smoothstep(t_in / down_steps)
            wp_p[:] = grip_pt
            wp_p[:, 2] = grip_pt[:, 2] + hand_to_pad + HOVER_CLEAR - s * (HOVER_CLEAR + gd)
            if s >= 1.0:
                want = grip_pt.clone()
                want[:, 2] -= gd
                pos_off[:] = (pos_off + 0.25 * (want - pad_centre())).clamp(-0.12, 0.12)
            act = servo(wp_p + pos_off, wp_q, w_strad)
            want = grip_pt.clone()
            want[:, 2] -= gd
            pad_on = (pad_centre() - want).norm(dim=-1) < 0.003
            if (t_in >= down_steps + 8 and bool(pad_on.all())) or t_in >= 2 * WP_TIMEOUT:
                close_ok.zero_()
                phase, marker = "pick_close", i
        elif phase == "pick_close":  # close to the part's visual width: the pads land ON the
            # faces and the finger PD holds them there; closure is verified geometrically and the
            # weld contract then carries the part
            s = smoothstep(t_in / close_steps)
            width = w_strad + (w_grip - w_strad) * s
            act = servo(wp_p + pos_off, wp_q, width)
            gap = art.data.joint_pos[:, fingers].sum(dim=-1)
            want = grip_pt.clone()
            want[:, 2] -= gd
            near = (pad_centre() - want).norm(dim=-1) < 0.004
            ok = near & (gap > c_min) & (gap < c_max)
            close_ok[:] = torch.where(ok, close_ok + 1, torch.zeros_like(close_ok))
            if t_in >= close_steps + 6 and bool((close_ok >= 4).all()):
                weld_on()
                print(f"  grasped the {label()}: pads on the slab faces, finger gap "
                      f"{[f'{float(g)*1e3:.1f}' for g in gap]} mm (slab {slab}), pad err "
                      f"{float((pad_centre() - want).norm(dim=-1).max()) * 1e3:.1f} mm", flush=True)
                phase, marker = "lift", i
            elif t_in >= close_steps + 30:
                if stage_picks < PICK_RETRIES:
                    print(f"  pick of the {label()} missed (gap "
                          f"{[f'{float(g)*1e3:.1f}' for g in gap]} mm), retrying", flush=True)
                    phase, marker = "pick_hover", i
                    picks += 1
                    stage_picks += 1
                else:
                    print("  ABORT: pick failed", flush=True)
                    phase, marker = "retreat", i
        elif phase == "lift":  # straight up out of the holder to rim-crossing height (glided)
            if t_in == 1:
                lift_xy[:] = part().data.root_pos_w[:, 0:2]  # rise only: latched, not live (a
                # live xy target rides the achieved pose and drifts through the holder rails)
                lift_from[:] = part().data.root_pos_w[:, 2]
            s = smoothstep(t_in / LIFT_STEPS)
            kp = part().data.root_pos_w.clone()
            kp[:, 0:2] = lift_xy
            kp[:, 2] = lift_from + s * (board_z + CROSS_Z - lift_from)
            tp, tq = hand_for_part(kp, upright_cmd())
            act = servo(tp, tq, w_grip)
            if (t_in >= LIFT_STEPS and bool(((board_z + CROSS_Z - part().data.root_pos_w[:, 2]).abs() < 0.01).all())) \
                    or t_in >= LIFT_STEPS + 45:
                pos_off.zero_()  # pick-spot bias is stale here; re-learn on the carry
                phase, marker = "carry", i
        elif phase == "carry":  # translate to above the work point, at crossing height (glided):
            # the card heads for its PLACEMENT point (bracket forward of the rear panel), a stick
            # for its slot
            if t_in == 1:
                carry_from[:] = part().data.root_pos_w
            goal = (place_w if is_card else seats_w[k]).clone()
            goal[:, 2] = board_z + CROSS_Z
            s = smoothstep(t_in / carry_steps)
            if s >= 1.0:  # arrived, free air near the case: learn the part-frame biases
                pos_off[:] = (pos_off + 0.25 * (goal - part().data.root_pos_w)).clamp(-0.12, 0.12)
                learn_rot()
            kp = carry_from + (goal - carry_from) * s
            tp, tq = hand_for_part(kp + pos_off, upright_cmd())
            act = servo(tp, tq, w_grip)
            ref = place_w if is_card else seats_w[k]
            arrived = (part().data.root_pos_w[:, 0:2] - ref[:, 0:2]).norm(dim=-1) < 0.003
            if (t_in >= carry_steps + 5 and bool(arrived.all())) or t_in >= carry_steps + WP_TIMEOUT:
                drop_from[:] = part().data.root_pos_w[:, 2]
                phase, marker = "drop", i
        elif phase == "drop":  # descend to the work height: the card INSIDE the case at slide
            # height (bracket forward of the rear panel), a stick to its align hover
            zt = board_z + (SLIDE_Z if is_card else ALIGN_Z)
            s = smoothstep(t_in / DROP_STEPS)
            kp = (place_w if is_card else seats_w[k]).clone()
            kp[:, 2] = drop_from + s * (zt - drop_from)
            if s >= 1.0:  # learn only once the target is stationary (mid-glide error is mostly
                # tracking lag, which would poison the integrator)
                pos_off[:] = (pos_off + 0.2 * (kp - part().data.root_pos_w)).clamp(-0.12, 0.12)
                learn_rot()
            tp, tq = hand_for_part(kp + pos_off, upright_cmd())
            act = servo(tp, tq, w_grip)
            if t_in >= DROP_STEPS + 15:
                phase, marker = "align", i
        elif phase == "align" and is_card:  # settle at the placement point before the rearward
            # slide: the bracket must enter the cutout square-on
            kp = place_w.clone()
            kp[:, 2] = board_z + SLIDE_Z
            pos_off[:] = (pos_off + 0.1 * (kp - part().data.root_pos_w)).clamp(-0.12, 0.12)
            learn_rot(0.05)  # gentle: rot is converged by now, and each correction sways the
            # card (0.24 m below the hand) sideways
            tp, tq = hand_for_part(kp + pos_off, upright_cmd())
            act = servo(tp, tq, w_grip)
            still = part().data.root_lin_vel_w.norm(dim=-1) < 0.01
            z_ok = (part().data.root_pos_w[:, 2] - (board_z + SLIDE_Z)).abs() < 0.0005
            ok = (xy_err(place_w) < CARD_PLACE_TOL) & z_ok \
                & (part_upright() > math.cos(math.radians(1.5))) & still
            if bool(ok.all()) or t_in >= 4 * WP_TIMEOUT:
                print(f"  placed the card: xy err {float(xy_err(place_w).max()) * 1e3:.2f} mm, "
                      f"tilt {float(torch.rad2deg(torch.acos(part_upright().clamp(-1, 1))).max()):.2f} deg",
                      flush=True)
                phase, marker = "slide", i
        elif phase == "align":  # (stick) settle at the align hover: the 0.4 mm gate beats the
            # funnel AND the end-stop play, so the lowered blade enters clean
            kp = seats_w[k].clone()
            kp[:, 2] = board_z + ALIGN_Z
            pos_off[:] = (pos_off + 0.1 * (kp - part().data.root_pos_w)).clamp(-0.12, 0.12)
            learn_rot(0.05)
            tp, tq = hand_for_part(kp + pos_off, upright_cmd())
            act = servo(tp, tq, w_grip)
            still = part().data.root_lin_vel_w.norm(dim=-1) < 0.01
            z_ok = (part().data.root_pos_w[:, 2] - (board_z + ALIGN_Z)).abs() < 0.001
            ok = (xy_err() < RAM_ALIGN_TOL) & z_ok & (rot_err() < ALIGN_ROT_TOL) & still
            if bool(ok.all()) or t_in >= 6 * WP_TIMEOUT:
                print(f"  aligned stick {k}: xy err {float(xy_err().max()) * 1e3:.2f} mm, rot "
                      f"{float(torch.rad2deg(rot_err()).max()):.2f} deg", flush=True)
                press_from[:] = part().data.root_pos_w[:, 2]
                phase, marker = "press", i
        elif phase == "slide":  # (card) rearward: the bracket/ports pass through the I/O cutout
            s = smoothstep(t_in / SLIDE_STEPS)
            kp = place_w.clone()
            kp[:, 0] = place_w[:, 0] + s * SLIDE_OFF
            kp[:, 2] = board_z + SLIDE_Z  # y/z biases frozen from align: contact is possible there
            if s >= 1.0:  # x stays contact-free through the whole slide (the end stops act below
                # the wall tops), so keep integrating its bias — the sag changes along the travel
                pos_off[:, 0] = (pos_off[:, 0]
                                 + 0.15 * (kp[:, 0] - part().data.root_pos_w[:, 0])).clamp(-0.12, 0.12)
            tp, tq = hand_for_part(kp + pos_off, upright_cmd())
            act = servo(tp, tq, w_grip)
            still = part().data.root_lin_vel_w.norm(dim=-1) < 0.01
            done = (xy_err() < 0.001) & still  # the funnel mouth absorbs 1.2 mm/side
            if (t_in >= SLIDE_STEPS and bool(done.all())) or t_in >= 2 * SLIDE_STEPS:
                print(f"  slid the card: xy err {float(xy_err().max()) * 1e3:.2f} mm", flush=True)
                press_from[:] = part().data.root_pos_w[:, 2]
                phase, marker = "press", i
        elif phase == "press":  # straight down, STILL GRIPPED: the channel funnel guides the
            # part's edge while the weld carries the press force. The card and the first stick
            # press to FULL depth pinched; the second stick stops as soon as the mouth has
            # captured its blade (CAPTURE_Z) — beside the seated neighbour no pad-on-face grip
            # may go deeper (see RAM_GRIP_DEPTH_HI), so the fingertips take over from there.
            s = smoothstep(t_in / press_steps)
            handoff = (not is_card) and ram_i > 0
            z_end = board_z + sc.cfg.ram_seat_pos[k][2] + (CAPTURE_Z - 0.0005) if handoff \
                else seat_p()[:, 2] + PRESS_TGT
            kp = seat_p().clone()
            kp[:, 2] = press_from + s * (z_end - press_from)
            if (not is_card) and t_in % 3 == 0 and s < 0.6:  # free air until the blade meets the
                # mouth: keep the xy trim live so the blade arrives centred on the funnel
                pos_off[:, 0:2] = (pos_off[:, 0:2]
                                   + 0.1 * (kp[:, 0:2] - part().data.root_pos_w[:, 0:2])).clamp(-0.12, 0.12)
            tp, tq = hand_for_part(kp + pos_off, upright_cmd())
            act = servo(tp, tq, w_grip)
            bz = part().data.root_pos_w[:, 2] - seat_p()[:, 2]
            if not handoff and bool((depth() >= press_done).all()):
                phase, marker = "release", i
            elif handoff and t_in >= 10 and bool((bz <= CAPTURE_Z + 0.0002).all()):
                print(f"  stick {k} captured: blade {float(bz.mean() * 1e3):.2f} mm in the mouth, "
                      f"handing over to the fingertips", flush=True)
                phase, marker = "reform", i
            elif t_in >= press_max:
                if retries < MAX_RETRIES:
                    retries += 1
                    total_retries += 1
                    print(f"  WARN: press on the {label()} stalled at "
                          f"{float(depth().mean() * 1e3):+.2f} mm — retry {retries}/{MAX_RETRIES}", flush=True)
                    phase, marker = "reseat", i
                else:
                    print(f"  WARN: press on the {label()} exhausted its retries — releasing as-is",
                          flush=True)
                    phase, marker = "release", i
        elif phase == "reseat":  # rise back, still gripped, and re-press: the card only to slide
            # height (its bracket sits in the cutout — higher would jam on the panel above), a
            # stick to just above its slot mouth
            kp = seat_p().clone()
            kp[:, 2] = board_z + (SLIDE_Z if is_card else RESEAT_Z)
            tp, tq = hand_for_part(kp + pos_off, upright_cmd())
            act = servo(tp, tq, w_grip)
            if t_in >= WP_TIMEOUT // 2:
                press_from[:] = part().data.root_pos_w[:, 2]
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
                act = servo(wp_p, reform_q, RAM_STRADDLE_W)
            else:  # close the tips over the top edge, hovering 1.5 mm above it
                if t_in == 25:
                    hover_from[:] = wp_p
                tip_p = grip_point()
                tip_p[:, 2] = tip_p[:, 2] + 0.0015 + hand_to_tip
                s = smoothstep((t_in - 24) / (REFORM_STEPS - 24.0))
                wp_p[:] = hover_from + (tip_p - hover_from) * s
                if s >= 1.0 and t_in % 2 == 0:  # trim the hand's own droop onto the edge
                    pos_off[:, 0:2] = (pos_off[:, 0:2] + 0.15 * (tip_p[:, 0:2] - hp[:, 0:2])).clamp(-0.12, 0.12)
                w = RAM_STRADDLE_W + (TIP_W - RAM_STRADDLE_W) * s
                act = servo(wp_p + pos_off, reform_q, w)
                on_edge = bool(((tip_p[:, 0:2] - hp[:, 0:2]).norm(dim=-1) < 0.0012).all())
                if (t_in >= REFORM_STEPS + 10 and on_edge) or t_in >= 2 * WP_TIMEOUT:
                    phase, marker = "seatpress", i
        elif phase == "seatpress":  # drive the fingertips down on the top edge until the blade
            # bottoms out: REAL contact seats the stick (tip-bottom-on-top-face is a live pair)
            # while the channel keeps the blade centred; the tips' outer edges clear the
            # neighbour all the way (see TIP_W)
            if t_in == 1:
                seat_from[:] = hp[:, 2]
            z_end = seat_p()[:, 2] + RAM_GRIP_TOP_ZC + PRESS_TGT + hand_to_tip
            s = smoothstep(t_in / SEAT_STEPS)
            wp_p[:] = seats_w[k]
            wp_p[:, 2] = seat_from + s * (z_end - seat_from)
            if t_in % 2 == 0:
                pos_off[:, 0:2] = (pos_off[:, 0:2] + 0.1 * (wp_p[:, 0:2] - hp[:, 0:2])).clamp(-0.12, 0.12)
            cmd = wp_p.clone()
            cmd[:, 0:2] += pos_off[:, 0:2]
            act = servo(cmd, reform_q, TIP_W)
            if bool((depth() >= press_done).all()):
                phase, marker = "release", i
            elif t_in >= SEAT_MAX:
                if retries < MAX_RETRIES:
                    retries += 1
                    total_retries += 1
                    print(f"  WARN: seat-press on stick {k} stalled at "
                          f"{float(depth().mean() * 1e3):+.2f} mm — retry {retries}/{MAX_RETRIES}", flush=True)
                    phase, marker = "reform", i
                else:
                    print(f"  WARN: seat-press on stick {k} exhausted its retries — releasing as-is",
                          flush=True)
                    phase, marker = "release", i
        elif phase == "release":  # the part is seated: bleed the stored press through the weld,
            # let go, rise clear. Cutting the weld while the servo still presses pops the arm
            # back with the fingers dragging the part — so first re-target the LIVE pose with
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
                if is_card:  # straight up (glided): the open fingers back off the card's top edge
                    if t_in > 14:
                        s2 = smoothstep((t_in - 14) / 35.0)
                        wp2[:, 2] = release_p[:, 2] + s2 * (board_z + CROSS_Z - release_p[:, 2])
                    act = servo(wp2, release_q, OPEN_W)
                else:
                    if t_in > 14:  # rise first (glided): the fingers back off the seated top edge
                        s2 = smoothstep((t_in - 14) / 16.0)
                        wp2[:, 2] = release_p[:, 2] + s2 * 0.045
                    if t_in > 26:  # then peel WEST while still rising — transverse to the insert
                        # view's axis, so the hand visibly exits sideways off the seated stick
                        s3 = smoothstep((t_in - 26) / 14.0)
                        wp2[:, 0] = release_p[:, 0] - s3 * 0.035
                    # spread only once the hand is a few mm up: the fingertip sweep then passes
                    # ABOVE both seated sticks instead of across the neighbour's top edge
                    act = servo(wp2, release_q, RAM_STRADDLE_W if t_in > 20 else grip_freeze)
            if t_in >= (60 if is_card else 44):
                phase, marker = "clear", i
        elif phase == "clear":  # rise straight off the seated part, then the next part/retreat
            if t_in == 1:
                drop_from[:] = hp[:, 2]
                wp_p[:] = hp  # hold the exit xy; rise only
            s = smoothstep(t_in / 30.0)
            kp = wp_p.clone()
            kp[:, 2] = drop_from + s * (board_z + CROSS_Z + hand_to_tip - drop_from)
            act = servo(kp, release_q, OPEN_W)
            if t_in >= 34:
                print(f"  {label()} pressed: depth {float(depth().mean() * 1e3):+.2f} mm, xy "
                      f"{float(xy_err().mean() * 1e3):.2f} mm", flush=True)
                seq += 1
                if seq <= sc.cfg.num_slots:
                    is_card = False
                    ram_i = seq - 1
                    k = RAM_ORDER[ram_i]
                    retries = 0
                    stage_picks = 1
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
        else:  # settle: hands off — every seated part must hold on its own
            act = servo(home_p, home_q, OPEN_W)
            if t_in >= SETTLE_STEPS:
                break

        step(act)

        if not torch.isfinite(part().data.root_pos_w).all():
            print(f"  ABORT: {label()} state went non-finite", flush=True)
            break
        if i % LOG_EVERY == 0:
            gap = art.data.joint_pos[:, fingers].sum(dim=-1) * 1e3
            print(f"  ctrl {i:4d} [{phase:10s} {label():7s}] | depth {float(depth().mean() * 1e3):+6.2f}mm | "
                  f"xy err {float(xy_err().mean() * 1e3):6.2f}mm | rot {float(torch.rad2deg(rot_err()).max()):5.2f}deg | "
                  f"part z {float(part().data.root_pos_w[:, 2].mean()):.3f} | hand z {float(hp[:, 2].mean()):.3f} | "
                  f"gap {float(gap.mean()):4.1f}mm", flush=True)

    if writer is not None:
        writer.close()
        print("MP4:", args.video, flush=True)

    seated = sc.seated()  # (n, 1 + S) — re-checked for ALL parts after the final settle
    all_ok = seated.all(dim=1)
    gpu_d = sc.gpu_engaged()
    ram_d = sc.ram_engaged()
    per_part = f"gpu: depth {float(gpu_d.mean() * 1e3):+.2f} mm (stroke 5.0), xy " \
               f"{float((card.data.root_pos_w[:, 0:2] - gpu_seat_w[:, 0:2]).norm(dim=-1).mean() * 1e3):.2f} mm"
    per_part += " | " + " | ".join(
        f"slot{j}: depth {float(ram_d[:, j].mean() * 1e3):+.2f} mm, xy "
        f"{float((sc.rams[j].data.root_pos_w[:, 0:2] - seats_w[j][:, 0:2]).norm(dim=-1).mean() * 1e3):.2f} mm"
        for j in range(sc.cfg.num_slots)
    )
    print(f"PC-GPU-RAM-FRANKA | complete {int(all_ok.sum())}/{n} envs "
          f"({int(seated.sum())}/{n * (1 + sc.cfg.num_slots)} parts) | {per_part} | "
          f"{picks} picks, {total_retries} press retries", flush=True)
    close_and_exit(env, app)


if __name__ == "__main__":
    main()

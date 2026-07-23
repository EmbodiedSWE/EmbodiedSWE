"""Franka smoke for PcGpuAssemblyScene — the arm picks the RTX 2060 out of its foam holder and
installs it in the case's PCIe x16 slot through the rear I/O cutout, the way the force-driven
`pc_gpu_smoke` proved a build goes: place the card inside the case at a forward offset, slide it
rearward so the bracket/ports pass through the expansion-slot opening, then press it down to seat.

Grasping follows the benchmark weld-on-closure contract (cf. `allen_key_franka_smoke` / the
pouring suite): a normally-disabled FixedJoint hand<->card is enabled when the gripper is
verifiably closed around the card's body slab and released when it opens. Everything else is live
physics — card<->slot-channel, card<->rear-panel-frame, and the released card holding its seat
under gravity — so a missed grasp, a jammed slide, or a popped seat fails honestly.

The `assembly.pc_gpu.franka.*` env stages the card UPRIGHT in the scene's foam holder, already in
the seated orientation (lying flat, its only sub-80 mm dimension points up — no parallel-jaw pinch
exists; see the env registration). One top-down FINGERTIP grip on the card's top edge, taken above
the PCB tab, then serves the whole task: pre-narrowed fingers press onto the body slab's top face
and their tip chamfers cam outward to clamp the two top corners (a real, stall-verified closure —
see the grasp-geometry note below for why a mid-slab parallel pinch is not verifiable on this
stack). The card never rotates, and the press force line runs straight down the grip into the slot.

Phases: show -> pick (hover/wedge, stall-verified) -> lift -> carry (over the 195 mm
case rim) -> drop (inside the case, bracket forward of the rear panel) -> align -> slide (rearward
through the cutout) -> press (with reseat re-tries that never rise above slide height — the
bracket sits in the cutout) -> release -> retreat -> settle. Verdict: seated count, tab depth vs
the 5 mm stroke, residual errors after release, picks and press retries.

The recorded video is one continuous MOVING shot: it opens framing the foam holder for the
grasp, cranes across the case as the card is carried (the blend is driven by the card's own
progress along its holder->seat line), and settles into the proven slot close-up for the
slide + press + release.

.venv/bin/python -m robobench.suites.assembly.smokes.pc_gpu_franka --headless
python -m robobench.suites.assembly.smokes.pc_gpu_franka --livestream 2
python -m robobench.suites.assembly.smokes.pc_gpu_franka \
    --headless --enable_cameras --video robobench/suites/assembly/videos/pc_gpu_franka.mp4
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
    from robobench.suites.assembly.scenes import PcGpuAssemblyScene

DT = 1.0 / 240.0  # sim timestep (matches the registered env's dt override)

# Card-local grasp geometry (from gpu_rtx2060.usd `/gpu/collision`; origin = PCB-tab bottom centre,
# axes = the seated/case axes). The grip is a FINGERTIP WEDGE on the body slab's TOP-EDGE corners,
# centred ABOVE the tab so the press force line runs straight down into the slot: the fingers
# pre-narrow to less than the 36 mm slab, press down onto the top face, and the tip chamfers cam
# outward until they clamp the two top corners. Probe-measured ground truth behind this choice:
# on this stack the finger<->card pair generates contacts at the fingerTIP/chamfer features (a
# 24 mm-narrowed descent stalls the arm and forces the finger joints apart at a ~24 mm sum) but
# NOT between the long flat finger faces and the slab's side faces — a mid-slab parallel pinch
# closes clean through the card at any speed, offset, or card representation (convex/box/SDF),
# so the classic 36 mm-stall grip cannot be verified. The wedge grip gives a real, force-verified
# closure; the weld contract (below) then carries the load, as in the sibling franka smokes.
GRIP_TOP_ZC = 0.1155     # body-slab TOP face above the card origin (the collider is trimmed to
                         # the VISUAL shroud top edge, so the wedging tips touch what they grip)
GRIP_YC = 0.0154         # body-slab mid-plane (faces at y -0.002 / +0.0328)
WEDGE_BELOW = 0.015      # commanded pad-centre depth below the slab top during the wedge press
                         # (the corners stall it after a few mm — the command keeps a bite force)
HOVER_CLEAR = 0.05       # pad-centre hover height above the slab top before the wedge: must
                         # out-clear the unbiased gravity sag (~15 mm standing, ~35 mm at the
                         # approach's low transient) or the tips snag the card while learning
# panda_finger body origin -> finger-pad centre, along the hand's approach axis (the hand-frame ->
# pad distance itself is measured live at reset: hand -> finger base + this).
FINGER_TO_PAD = 0.045
OPEN_W, NARROW_W = 0.04, 0.012
# Gripper closure window on the card (sum of the two finger joints, m): the wedged tips clamp the
# top corners near the narrow target (measured 0.0236); well below it means the tips slid past the
# corners into the slab (no contact), well above means they never straddled the top edge.
CLOSED_MIN, CLOSED_MAX = 0.018, 0.034

# Franka OSC action semantics (6 EE pose deltas + 2 finger position targets at 15 Hz).
POS_SCALE, ROT_SCALE = 0.02, 0.097
ROT_SAT = 8.0            # rotation lead cap (units): gravity droop stalls a 1-unit lead

# Insertion flight plan (m, case frame — the force-driven smoke's validated numbers).
CROSS_Z = 0.240          # card-origin height while crossing the case rim (foot clears the 195 mm walls)
SLIDE_OFF = 0.028        # forward (-x) placement offset: bracket clear of the rear panel
SLIDE_Z = 0.0172         # placement/slide height: tab 1.7 mm above the channel walls, bracket
                         # 1.1 mm under the cutout top. The window is [wall tops 0.0155 + z-servo
                         # slack, cutout top 0.128 - bracket 0.1097]: the arm places z to ~0.5 mm
                         # (gated below), so the tab must start with >1 mm wall clearance or its
                         # front corner beaches on the wall tops during the slide
PRESS_TGT = -0.002       # commanded card-origin z below the seat during the press (the soft OSC
                         # servo needs a standing lead; the channel floor takes the surplus)
PRESS_DONE = 0.0048      # tab depth below the slot mouth to call the press finished (stroke 5 mm)
MAX_RETRIES = 2          # press re-tries (rise back to SLIDE_Z ONLY — the bracket sits in the
                         # rear cutout and would jam on the panel above it — then press again)

# Waypoint tolerances and per-phase budgets, in CONTROL steps (15 Hz -> 16 substeps each at 1/240).
TOL_P, TOL_R = 0.004, 0.06
PLACE_TOL = 0.0008       # card-origin xy gate at the placement point / after the slide (the
                         # channel funnel absorbs 1.2 mm/side; the scripted smoke used the same)
SHOW_END = 20
WP_TIMEOUT, SETTLE_STEPS = 75, 45
HOVER_STEPS, LIFT_STEPS, CARRY_STEPS, RETREAT_STEPS = 40, 30, 45, 25  # transit glide lengths:
# every long move GLIDES its commanded target (smoothstep) from the phase-entry pose — a
# step-jump goal saturates the norm-clamped servo and the gravity-uncompensated arm swings
# 30-40 mm wide, then rings while the bias integrator unwinds the veer (reads as hovering)
DROP_STEPS, SLIDE_STEPS, PRESS_STEPS, PRESS_MAX = 45, 75, 60, 180
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

    env = ENVS.get("assembly.pc_gpu.franka.osc")().build(num_envs=args.num_envs, device=device)
    sc: PcGpuAssemblyScene = env.scene  # type: ignore[assignment]
    n = env.num_envs
    dev = device
    card, case = sc.card, sc.case
    art = env.robot.articulation
    hand_idx = art.body_names.index("panda_hand")
    j7 = art.joint_names.index("panda_joint7")
    fingers = art.find_joints(["panda_finger_joint.*"])[0]
    render = (not args.headless) or livestream_on
    ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)

    # ----- hand<->card weld (the grasp contract): pre-authored, disabled FixedJoints --------------
    # PhysX latches a joint's local frames when it is FIRST enabled; frame rewrites on a re-enabled
    # joint are ignored. Each grasp therefore consumes a fresh joint from the pool: author the live
    # relative pose while it is still disabled, enable it once, and on release disable it for good.
    from pxr import Gf, UsdPhysics

    stage = env.stage
    WELD_POOL = 4  # one pick + slack (press re-tries keep the grip; there is no regrasp path)
    weld_paths = []
    for e in range(n):
        base = f"/World/envs/env_{e}"
        assert stage.GetPrimAtPath(f"{base}/Card").HasAPI(UsdPhysics.RigidBodyAPI)
        row = []
        for k in range(WELD_POOL):
            j = UsdPhysics.FixedJoint.Define(stage, f"{base}/hand_card_weld_{k}")
            j.CreateBody0Rel().SetTargets([f"{base}/Robot/panda_hand"])
            j.CreateBody1Rel().SetTargets([f"{base}/Card"])
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateJointEnabledAttr(False)
            j.CreateExcludeFromArticulationAttr(True)  # a maximal-coordinate weld, not a new arm DOF
            row.append(f"{base}/hand_card_weld_{k}")
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

    # Open-env OSC retune (cf. allen_key_franka_smoke): the controller carries no gravity
    # compensation, so the stock gains leave a configuration-dependent sag — measured ~30 mm at
    # the 0.48 m pick reach at the stock kp. kp_pos=400 halves it into the learned-bias budget
    # (the +-80 mm bias clamp covers the rest); kp_rot=450 cuts the wrist's gravity-moment tilt
    # droop to ~1-2 deg so the learned rotation bias can finish the job (seating gate is 3 deg).
    # Damping stays critical.
    osc = env.robot.controller.controllers[0]
    osc._kp[0:3] = 400.0
    osc._kp[3:6] = 450.0
    osc._kd = 2.2 * osc._kp.sqrt()  # slightly overdamped: transits must not ring

    case_pos = case.data.root_pos_w.clone()  # (n, 3): origin ON the board face, at its centre
    board_z = case_pos[:, 2].clone()
    seat_w = case_pos + torch.tensor(sc.cfg.seat_pos, device=dev)  # (n, 3) seated card origin, world
    place_w = seat_w.clone()  # placement point INSIDE the case: bracket forward of the rear panel
    place_w[:, 0] -= SLIDE_OFF

    # ----- moving camera: one continuous shot that shows the grasp AND the insertion --------------
    # Two anchor framings, blended by the CARD'S OWN PROGRESS along its holder->seat line so the
    # camera always follows the action with no phase plumbing: a 3/4 view from the holder's open
    # (-x, +y) quadrant for the pick (from the slot view's side the 195 mm case rim occludes the
    # holder), craning across the case into the force-driven smoke's proven slot close-up — the
    # only angle that shows the gold edge connector over the slot. A sin(pi*s) altitude bump keeps
    # the elevated carry in frame mid-transit; per-frame easing plus a monotonic latch keep the
    # shot smooth through pick retries and hold the final framing once the card is seated.
    cam_pose = None
    if cam is not None:
        p0 = case_pos[0]
        holder_xy = card.data.root_pos_w[0, 0:2].clone()  # card spawn xy = the foam holder
        path_v = seat_w[0, 0:2] - holder_xy
        path_len = float(path_v.norm())
        path_dir = path_v / path_len
        pick_eye = torch.tensor([float(holder_xy[0]) - 0.38, float(holder_xy[1]) + 0.39, 0.40], device=dev)
        pick_tgt = torch.tensor([float(holder_xy[0]), float(holder_xy[1]), 0.12], device=dev)
        ins_eye = torch.tensor([float(p0[0]) + 0.13, float(p0[1]) - 0.26, float(p0[2]) + 0.40], device=dev)
        ins_tgt = torch.tensor([float(p0[0]) + 0.03, float(p0[1]) + 0.02, float(p0[2]) + 0.075], device=dev)
        cam_s = 0.0  # eased, latched blend state

        def cam_pose() -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal cam_s
            u = float((card.data.root_pos_w[0, 0:2] - holder_xy) @ path_dir) / path_len
            s = smoothstep(u)
            cam_s = max(cam_s, cam_s + 0.12 * (s - cam_s))  # ease toward s, never retreat
            arc = math.sin(math.pi * cam_s)
            eye = pick_eye + (ins_eye - pick_eye) * cam_s
            tgt = pick_tgt + (ins_tgt - pick_tgt) * cam_s
            eye = eye + torch.tensor([0.0, 0.0, 0.12 * arc], device=dev)
            tgt = tgt + torch.tensor([0.0, 0.0, 0.16 * arc], device=dev)
            return eye.unsqueeze(0), tgt.unsqueeze(0)

    print(env.describe(), flush=True)

    # Measure hand-frame -> finger-pad-centre once from the live articulation (robust to asset edits).
    lf = art.body_names.index("panda_leftfinger")
    _hq0 = art.data.body_quat_w[:, hand_idx]
    _rel = quat_apply_inverse(_hq0, art.data.body_pos_w[:, lf] - art.data.body_pos_w[:, hand_idx])
    hand_to_pad = float(_rel[0, 2]) + FINGER_TO_PAD
    print(f"hand->pad centre: {hand_to_pad:.4f} m", flush=True)

    # ----- weld toggles + grasp transform ---------------------------------------------------------
    rel_p = torch.zeros(n, 3, device=dev)  # card pose in the hand frame, captured at weld time
    rel_q = torch.zeros(n, 4, device=dev)
    welded = torch.zeros(n, dtype=torch.bool, device=dev)
    weld_k = 0  # next fresh joint in the pool (all envs grasp in lockstep)

    def hand_pose() -> tuple[torch.Tensor, torch.Tensor]:
        return art.data.body_pos_w[:, hand_idx], art.data.body_quat_w[:, hand_idx]

    def weld_on() -> None:
        nonlocal weld_k
        assert weld_k < WELD_POOL, "weld pool exhausted"
        hp, hq = hand_pose()
        rel_p[:] = quat_apply_inverse(hq, card.data.root_pos_w - hp)
        rel_q[:] = quat_mul(quat_conjugate(hq), card.data.root_quat_w)
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

    def hand_for_card(kp: torch.Tensor, kq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Hand waypoint that puts the WELDED card at pose (kp, kq), via the captured grasp frame."""
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

    def nearest_parity(psi: torch.Tensor, prev: torch.Tensor | None = None) -> torch.Tensor:
        """The gripper is 180-deg symmetric: return psi or psi-pi, whichever leaves the wrist roll
        nearer its centre. With `prev`, keep the previous choice unless clearly better (hysteresis)."""
        alt = _wrap(psi - math.pi)
        a, b = j7_after(psi).abs(), j7_after(alt).abs()
        if prev is None:
            return torch.where(a <= b, psi, alt)
        keep_psi = _wrap(prev - psi).abs() < _wrap(prev - alt).abs()
        margin = math.radians(10.0)
        pick_psi = torch.where(keep_psi, a <= b + margin, a + margin < b)
        return torch.where(pick_psi, psi, alt)

    def grip_point() -> torch.Tensor:
        """World wedge point on the live card: the body slab's TOP face centre, above the tab."""
        off = torch.tensor([0.0, GRIP_YC, GRIP_TOP_ZC], device=dev).expand(n, 3)
        return card.data.root_pos_w + quat_apply(card.data.root_quat_w, off)

    def pad_centre() -> torch.Tensor:
        """World centre of the closed finger pads, along the hand's approach axis."""
        hp, hq = hand_pose()
        return hp + quat_apply(hq, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)) * hand_to_pad

    def depth() -> torch.Tensor:  # tab depth below the slot mouth (m), per env
        return sc.engaged()

    def xy_err(ref: torch.Tensor) -> torch.Tensor:
        return (card.data.root_pos_w[:, 0:2] - ref[:, 0:2]).norm(dim=-1)

    def card_upright() -> torch.Tensor:
        return up_axis_of(card.data.root_quat_w)[:, 2]

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
    picks, retries = 0, 0
    pos_off = torch.zeros(n, 3, device=dev)   # INTEGRATED bias (desired - achieved, free air): the
    # OSC has no gravity compensation, so its realized pose sags configuration-dependently by
    # several mm — commands near contact add this learned offset so the achieved pose lands true
    rot_bias = torch.zeros(n, 3, device=dev)  # the same for ORIENTATION (axis-angle, integrated
    # in free air): the wrist's gravity moment leaves a standing card tilt the funnel cannot eat
    grip_freeze = torch.zeros(n, 2, device=dev)  # finger targets latched at release: freezing the
    # PD at the MEASURED positions decays the squeeze before the pads separate
    grip_pt = torch.zeros(n, 3, device=dev)
    grip_yaw = torch.zeros(n, device=dev)
    wedge_ok = torch.zeros(n, device=dev)  # consecutive ticks the wedge has read stalled+near
    wp_p = torch.zeros(n, 3, device=dev)
    wp_q = torch.zeros(n, 4, device=dev)
    hover_from = torch.zeros(n, 3, device=dev)
    carry_from = torch.zeros(n, 3, device=dev)
    retreat_from = torch.zeros(n, 3, device=dev)
    lift_xy = torch.zeros(n, 2, device=dev)
    lift_from = torch.zeros(n, device=dev)
    drop_from = torch.zeros(n, device=dev)
    press_from = torch.zeros(n, device=dev)
    release_p = torch.zeros(n, 3, device=dev)
    release_q = torch.zeros(n, 4, device=dev)

    def smoothstep(t: float) -> float:
        t = min(max(t, 0.0), 1.0)
        return t * t * (3 - 2 * t)

    def upright_cmd() -> torch.Tensor:
        """Commanded card orientation: upright (the seated identity), pre-rotated by the learned
        droop bias so the ACHIEVED orientation lands upright."""
        ang = rot_bias.norm(dim=-1).clamp_min(1e-9)
        return quat_from_angle_axis(ang, rot_bias / ang.unsqueeze(-1))

    def learn_rot() -> None:
        """Integrate the card's residual orientation error (free air only): the command already
        carries the bias, so the residual drives it until the achieved pose is upright."""
        r = axis_angle_from_quat(quat_conjugate(card.data.root_quat_w))
        rot_bias[:] = (rot_bias + 0.2 * r).clamp(-0.2, 0.2)

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
                phase, marker = "pick_hover", i
                picks += 1
        elif phase == "pick_hover":  # top-down over the standing card's top edge, fingers
            # pre-narrowed to less than the slab so the descending tips land ON the top face;
            # the approach GLIDES from wherever the hand is (see HOVER_STEPS note)
            if t_in == 1:
                hover_from[:] = hp
            grip_pt[:] = grip_point()
            kx = quat_apply(card.data.root_quat_w, torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3))
            raw_yaw = torch.atan2(kx[:, 1], kx[:, 0])  # hand x along the card length -> fingers across the slab
            grip_yaw[:] = nearest_parity(raw_yaw, prev=grip_yaw if t_in > 1 else None)
            wp_p[:] = grip_pt
            wp_p[:, 2] = grip_pt[:, 2] + hand_to_pad + HOVER_CLEAR  # tips well above the top
            # face even UNBIASED: the raw gravity sag is ~15 mm, and tips that dip into the top
            # edge before the bias has learned knock the card around the holder
            wp_q[:] = q_down(grip_yaw)
            s = smoothstep(t_in / HOVER_STEPS)
            if s >= 1.0:  # arrived, free air: learn the pad-centre bias for the wedge press
                want = grip_pt.clone()
                want[:, 2] += HOVER_CLEAR
                pos_off[:] = (pos_off + 0.3 * (want - pad_centre())).clamp(-0.08, 0.08)
            goal = wp_p + pos_off
            act = servo(hover_from + (goal - hover_from) * s, wp_q, NARROW_W)
            # Enter the wedge only once the PAD is measured on-target in xy — a wedge started
            # off-centre just grinds its budget away re-centring.
            pad_err = (pad_centre()[:, 0:2] - grip_pt[:, 0:2]).norm(dim=-1)
            if (t_in >= HOVER_STEPS + 8 and bool((pad_err < 0.006).all()) and bool(at(goal, wp_q).all())) \
                    or t_in >= 2 * WP_TIMEOUT:
                wedge_ok.zero_()
                phase, marker = "pick_wedge", i
        elif phase == "pick_wedge":  # press the narrowed tips down onto the slab top: the tip
            # chamfers cam onto the two top-edge corners and the ARM STALLS several mm above its
            # command — the closure signature is that sustained stall (NOT a finger-gap stall,
            # and NOT a squeeze: driving the fingers to 0 cams the tips past the corners and the
            # grip collapses — the fingers HOLD the narrow target through the weld and beyond).
            # grip_pt stays latched from the hover (the press nudges the card a little; chasing
            # it would wander). The descent is smoothstepped — a standing 35 mm z error starves
            # the xy axes through the action's direction norm-clamp and the hand veers off the
            # grip point — and the XY bias keeps learning against the live pad position (the tips
            # may grind along the top face back over the tab; only z is contact-held).
            s = smoothstep(t_in / 20.0)
            wp_p[:] = grip_pt
            wp_p[:, 2] = grip_pt[:, 2] + HOVER_CLEAR + hand_to_pad - s * (HOVER_CLEAR + WEDGE_BELOW)
            pos_off[:, 0:2] = (0.8 * pos_off[:, 0:2]
                               + 0.2 * (grip_pt[:, 0:2] - pad_centre()[:, 0:2])).clamp(-0.08, 0.08)
            act = servo(wp_p + pos_off, wp_q, NARROW_W)
            gap = art.data.joint_pos[:, fingers].sum(dim=-1)
            stalled = hp[:, 2] - (wp_p[:, 2] + pos_off[:, 2]) > 0.008
            near = (pad_centre()[:, 0:2] - grip_pt[:, 0:2]).norm(dim=-1) < 0.02
            ok = stalled & near & (gap > CLOSED_MIN) & (gap < CLOSED_MAX)
            wedge_ok[:] = torch.where(ok, wedge_ok + 1, torch.zeros_like(wedge_ok))
            if bool((wedge_ok >= 6).all()):
                weld_on()
                print(f"  grasped: tip wedge on the top corners, finger gap "
                      f"{[f'{float(g)*1e3:.1f}' for g in gap]} mm, "
                      f"stall {float((hp[:, 2] - wp_p[:, 2] - pos_off[:, 2]).mean()) * 1e3:.1f} mm", flush=True)
                phase, marker = "lift", i
            elif t_in >= 40:
                if picks < PICK_RETRIES:
                    print(f"  pick {picks} missed (gap {[f'{float(g)*1e3:.1f}' for g in gap]} mm, "
                          f"stall {float((hp[:, 2] - wp_p[:, 2] - pos_off[:, 2]).mean()) * 1e3:.1f} mm), retrying", flush=True)
                    phase, marker = "pick_hover", i
                    picks += 1
                else:
                    print("  ABORT: pick failed", flush=True)
                    phase, marker = "retreat", i
        elif phase == "lift":  # straight up out of the holder to rim-crossing height (glided)
            if t_in == 1:
                lift_xy[:] = card.data.root_pos_w[:, 0:2]  # rise only: latched, not live (a live
                # xy target rides the achieved pose and drifts sideways through the holder rails)
                lift_from[:] = card.data.root_pos_w[:, 2]
            s = smoothstep(t_in / LIFT_STEPS)
            kp = card.data.root_pos_w.clone()
            kp[:, 0:2] = lift_xy
            kp[:, 2] = lift_from + s * (board_z + CROSS_Z - lift_from)
            tp, tq = hand_for_card(kp, upright_cmd())
            act = servo(tp, tq, NARROW_W)
            if (t_in >= LIFT_STEPS and bool(((board_z + CROSS_Z - card.data.root_pos_w[:, 2]).abs() < 0.01).all())) \
                    or t_in >= WP_TIMEOUT:
                pos_off.zero_()  # pick-spot bias is stale here; re-learn on the carry
                phase, marker = "carry", i
        elif phase == "carry":  # translate to above the PLACEMENT point, at crossing height,
            # gliding the card target across
            if t_in == 1:
                carry_from[:] = card.data.root_pos_w
            goal = place_w.clone()
            goal[:, 2] = board_z + CROSS_Z
            s = smoothstep(t_in / CARRY_STEPS)
            if s >= 1.0:  # arrived, free air near the case: learn the card-frame biases
                pos_off[:] = (pos_off + 0.25 * (goal - card.data.root_pos_w)).clamp(-0.08, 0.08)
                learn_rot()
            kp = carry_from + (goal - carry_from) * s
            tp, tq = hand_for_card(kp + pos_off, upright_cmd())
            act = servo(tp, tq, NARROW_W)
            arrived = (card.data.root_pos_w[:, 0:2] - place_w[:, 0:2]).norm(dim=-1) < 0.003
            if (t_in >= CARRY_STEPS + 5 and bool(arrived.all())) or t_in >= 2 * WP_TIMEOUT:
                drop_from[:] = card.data.root_pos_w[:, 2]
                phase, marker = "drop", i
        elif phase == "drop":  # descend INSIDE the case, bracket forward of the rear panel
            s = smoothstep(t_in / DROP_STEPS)
            kp = place_w.clone()
            kp[:, 2] = drop_from + s * (board_z + SLIDE_Z - drop_from)
            if s >= 1.0:  # learn only once the target is stationary — mid-descent the error is
                # mostly tracking lag, and integrating lag poisons the bias align must then unwind
                pos_off[:] = (pos_off + 0.2 * (kp - card.data.root_pos_w)).clamp(-0.08, 0.08)
                learn_rot()
            tp, tq = hand_for_card(kp + pos_off, upright_cmd())
            act = servo(tp, tq, NARROW_W)
            if t_in >= DROP_STEPS + 15:
                phase, marker = "align", i
        elif phase == "align":  # settle at the placement point before the rearward slide
            kp = place_w.clone()
            kp[:, 2] = board_z + SLIDE_Z
            pos_off[:] = (pos_off + 0.1 * (kp - card.data.root_pos_w)).clamp(-0.08, 0.08)
            r = axis_angle_from_quat(quat_conjugate(card.data.root_quat_w))
            rot_bias[:] = (rot_bias + 0.05 * r).clamp(-0.2, 0.2)  # gentle: rot is converged by now,
            # and each correction sways the card (0.24 m below the hand) sideways
            tp, tq = hand_for_card(kp + pos_off, upright_cmd())
            act = servo(tp, tq, NARROW_W)
            still = card.data.root_lin_vel_w.norm(dim=-1) < 0.01
            z_ok = (card.data.root_pos_w[:, 2] - (board_z + SLIDE_Z)).abs() < 0.0005
            ok = (xy_err(place_w) < PLACE_TOL) & z_ok & (card_upright() > math.cos(math.radians(1.5))) & still
            if bool(ok.all()) or t_in >= 4 * WP_TIMEOUT:
                print(f"  placed: xy err {float(xy_err(place_w).max()) * 1e3:.2f} mm, z err "
                      f"{float((card.data.root_pos_w[:, 2] - board_z - SLIDE_Z).abs().max()) * 1e3:.2f} mm, "
                      f"tilt {float(torch.rad2deg(torch.acos(card_upright().clamp(-1, 1))).max()):.2f} deg", flush=True)
                phase, marker = "slide", i
        elif phase == "slide":  # rearward: the bracket/ports pass through the I/O panel cutout
            s = smoothstep(t_in / SLIDE_STEPS)
            kp = place_w.clone()
            kp[:, 0] = place_w[:, 0] + s * SLIDE_OFF
            kp[:, 2] = board_z + SLIDE_Z  # y/z biases frozen from align: contact is possible there.
            if s >= 1.0:  # x, though, stays contact-free through the whole slide (the end stops
                # act below the wall tops) and its droop grows ~5 mm over the 28 mm of travel —
                # keep integrating it once the target is stationary, or the press starts with the
                # tab's front corner over the channel's end stop
                pos_off[:, 0] = (pos_off[:, 0]
                                 + 0.15 * (kp[:, 0] - card.data.root_pos_w[:, 0])).clamp(-0.08, 0.08)
            tp, tq = hand_for_card(kp + pos_off, upright_cmd())
            act = servo(tp, tq, NARROW_W)
            still = card.data.root_lin_vel_w.norm(dim=-1) < 0.01
            done = (xy_err(seat_w) < 0.001) & still  # the funnel mouth eats 1.2 mm/side — idling
            # a timeout to shave the last 0.1 mm buys nothing
            if (t_in >= SLIDE_STEPS and bool(done.all())) or t_in >= 2 * SLIDE_STEPS:
                print(f"  slid: xy err {float(xy_err(seat_w).max()) * 1e3:.2f} mm", flush=True)
                press_from[:] = card.data.root_pos_w[:, 2]
                phase, marker = "press", i
        elif phase == "press":  # straight down; the channel funnel guides the last 5 mm
            s = smoothstep(t_in / PRESS_STEPS)
            kp = seat_w.clone()
            kp[:, 2] = press_from + s * (seat_w[:, 2] + PRESS_TGT - press_from)
            tp, tq = hand_for_card(kp + pos_off, upright_cmd())
            act = servo(tp, tq, NARROW_W)
            if bool((depth() >= PRESS_DONE).all()):
                phase, marker = "release", i
            elif t_in >= PRESS_MAX:
                if retries < MAX_RETRIES:
                    retries += 1
                    print(f"  WARN: press stalled at {float(depth().mean() * 1e3):+.2f} mm — "
                          f"retry {retries}/{MAX_RETRIES}", flush=True)
                    phase, marker = "reseat", i
                else:
                    print("  WARN: press exhausted its retries — releasing as-is", flush=True)
                    phase, marker = "release", i
        elif phase == "reseat":  # back up to slide height ONLY (the bracket sits in the cutout —
            kp = seat_w.clone()  # rising higher would jam it on the panel above the opening)
            kp[:, 2] = board_z + SLIDE_Z
            tp, tq = hand_for_card(kp + pos_off, upright_cmd())
            act = servo(tp, tq, NARROW_W)
            if t_in >= WP_TIMEOUT // 2:
                press_from[:] = card.data.root_pos_w[:, 2]
                phase, marker = "press", i
        elif phase == "release":  # bleed the stored press through the weld, let go, rise clear.
            # Cutting the weld while the servo still presses pops the arm back with the fingers
            # friction-gripping the card — so first re-target the LIVE pose with the finger PD
            # frozen at its measured squeeze, then disable the weld, open, and only then rise.
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
                if t_in > 14:  # straight up (glided): the open fingers back off the card's top edge
                    s2 = smoothstep((t_in - 14) / 20.0)
                    wp2[:, 2] = release_p[:, 2] + s2 * (board_z + CROSS_Z - release_p[:, 2])
                act = servo(wp2, release_q, OPEN_W)
            if t_in >= 45:
                phase, marker = "retreat", i
        elif phase == "retreat":  # glide home
            if t_in == 1:
                retreat_from[:] = hp
            s = smoothstep(t_in / RETREAT_STEPS)
            act = servo(retreat_from + (home_p - retreat_from) * s, home_q, OPEN_W)
            if t_in >= RETREAT_STEPS + 10:
                phase, marker = "settle", i
        else:  # settle: hands off — the seated card must hold on its own
            act = servo(home_p, home_q, OPEN_W)
            if t_in >= SETTLE_STEPS:
                break

        step(act)

        if not torch.isfinite(card.data.root_pos_w).all():
            print("  ABORT: card state went non-finite", flush=True)
            break
        if i % LOG_EVERY == 0:
            gap = art.data.joint_pos[:, fingers].sum(dim=-1) * 1e3
            tilt = torch.rad2deg(torch.acos(card_upright().clamp(-1, 1)))
            print(f"  ctrl {i:4d} [{phase:10s}] | depth {float(depth().mean() * 1e3):+6.2f}mm | "
                  f"xy err {float(xy_err(seat_w).mean() * 1e3):5.2f}mm | tilt {float(tilt.max()):4.2f}deg | "
                  f"card z {float(card.data.root_pos_w[:, 2].mean()):.3f} | hand z {float(hp[:, 2].mean()):.3f} | "
                  f"gap {float(gap.mean()):4.1f}mm", flush=True)

    if writer is not None:
        writer.close()
        print("MP4:", args.video, flush=True)

    seated = sc.seated()  # (n,)
    tilt = torch.rad2deg(torch.acos(card_upright().clamp(-1, 1)))
    print(f"PC-GPU-FRANKA | seated {int(seated.sum())}/{n} envs | tab depth "
          f"{float(depth().mean() * 1e3):+.2f} mm (stroke 5.0, seat >= {sc.cfg.seat_depth * 1e3:.0f}) | "
          f"residual xy {float(xy_err(seat_w).mean() * 1e3):.2f} mm, tilt {float(tilt.mean()):.2f} deg | "
          f"{picks} picks, {retries} press retries", flush=True)
    close_and_exit(env, app)


if __name__ == "__main__":
    main()

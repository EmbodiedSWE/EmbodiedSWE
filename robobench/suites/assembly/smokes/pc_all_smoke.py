"""Physics smoke test for PcAllAssemblyScene — the COMPLETE PC install, in assembly order:
motherboard first, then memory, then the graphics card. No robot: every part is driven purely by
forces, each leg exactly the way its single-task smoke proved.

Leg 1 — motherboard (cf. `pc_motherboard_smoke`): all 7 bolts stage pre-engaged (the scene's
`screw_mechanic` kinematic screw joints); the ALLEN KEY starts where it lies on the table and
never teleports — it is flown to each hole under force-only PD (gravity-free key; the soft PD
wrenches stand in for the steadying hand), aligned to the socket's hex clocking, dropped in, and
driven: a ramped press along the bolt's axis + a torque-capped velocity-servo twist. The
key<->socket hex contact is LIVE; the scene's joint follows the key's measured rotation through
the lash and descends 1 mm/rev to a hard stop. Unlike the single-task smoke, the key's first
approach and its final park-out cross the case's 195 mm wall line HIGH — inside the case it
travels just over the standing heads. After the last hole the key parks back at its spawn spot,
PD-held, out of the later legs' way.

Leg 2 — memory (cf. `pc_ram_smoke`): each TridentZ stick is lifted off the table by a PD "hand"
(weight feedforward — gravity stays ON, so every hold-check is a real retention test), righted,
carried over the case rim, hovered over its DIMM slot and pressed straight down; the invisible
channel guides the last 4.4 mm. Far slot first, then the near one.

Leg 3 — graphics card (cf. `pc_gpu_smoke`): the RTX 2060 is lifted, righted, carried inside the
case at a 28 mm forward offset (bracket clear of the rear panel), slid rearward through the I/O
cutout, and pressed straight down into the PCIe x16 slot.

Phases: show -> [k_lift -> k_glide -> k_descend -> k_align -> k_insert -> k_drive] x 7 ->
k_exit -> k_park -> k_park_drop -> [r_lift -> r_cross -> r_drop -> r_align -> r_press
(re-tries) -> r_release] x 2 -> [g_lift -> g_cross -> g_drop -> g_align -> g_slide -> g_press
(re-tries)] -> settle. Verdict: the scene's combined seated() over all 10 parts (7 bolts + 2
sticks + 1 card), with per-family depths and residuals.

python -m robobench.suites.assembly.smokes.pc_all_smoke --livestream 2
python -m robobench.suites.assembly.smokes.pc_all_smoke \
    --headless --enable_cameras --video robobench/suites/assembly/videos/pc_all.mp4
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--holes", type=int, default=0, help="fasten only the first N holes (0 = all)")
parser.add_argument("--video", type=str, default="", help="save an mp4 here (needs --headless --enable_cameras)")
parser.add_argument("--cap", type=int, default=8, help="with --video: capture one frame every N steps (8 at dt=1/240 -> real-time at 30 fps)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
livestream_on = args.livestream > 0

app = AppLauncher(args).app

from typing import TYPE_CHECKING  # noqa: E402

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402

import robobench  # noqa: E402
from isaaclab.utils.math import axis_angle_from_quat, quat_apply_inverse  # noqa: E402
from robobench.core import EnvCfg  # noqa: E402
from robobench.suites.assembly.scenes import PcAllAssemblySceneCfg  # noqa: E402
from robobench.suites.assembly.smokes import close_and_exit  # noqa: E402

if TYPE_CHECKING:
    from robobench.suites.assembly.scenes import PcAllAssemblyScene

DT = 1.0 / 240.0          # sim timestep
GRAV = 9.81

# ----- leg 1: the key + the scene's screw joints (pc_motherboard_smoke's proven values) ----------
# Bolt-local geometry baked into the committed bolt USD (bolt origin = thread TIP, +z up):
THREAD_LEN = 0.0124       # head bottom above the tip: tip depth at which the head bottoms out
SOCKET_FLOOR_Z = 0.01775  # hex recess floor
KEY_TIP_HOVER = 0.0001    # key tip rides this far above the socket floor
STAGE_DEPTH = 0.006       # bolt tip depth below the board face at stage (m)
STAGE_YAW = math.pi       # bolt (and key) yaw at stage (a k*60 deg hex clocking)
SEAT_MARGIN = 0.0001      # screw-joint hard stop: head held this far above the board
STOP_DEPTH = 0.0118       # stop twisting at this tip depth (m) — just before the head bottoms
TRAVEL_Z = 0.030          # key TIP height above the board while hopping (clears the standing heads)
KEY_CROSS_Z = 0.215       # key TIP height for crossing the case's 195 mm wall line (in and out)
PITCH = 0.001             # screw-joint thread pitch (m per revolution; M8-scaled)
PITCH_MM = PITCH * 1e3
KW = 0.05                 # twist servo gain (N m s/rad)
DRIVE_CTL = (1.0, 2.5, 0.03)  # (press N, spin target rad/s, twist torque cap N m)
LASH_HALF = math.radians(8.0)  # key rotation before the hex flats engage
RAMP_STEPS = 120          # press/twist authority ramp-in at each drive start (steps at dt=1/240)
# Soft hand-steadying PD wrenches on the key:
K_KP_XY, K_KD_XY = 100.0, 4.0     # N/m, N s/m — recenters the key tip on the socket axis
K_KP_TILT, K_KD_TILT = 0.5, 0.02  # N m/rad, N m s/rad — rights the key (spin free)
K_KP_Z, K_KD_Z = 60.0, 5.0        # N/m, N s/m — the hop's vertical hold
K_KP_ZK, K_KD_ZK = 300.0, 8.0     # N/m, N s/m — keeps the tip riding the descending socket floor
K_KP_YAW, K_KD_YAW = 0.01, 0.002  # N m/rad, N m s/rad — aligns the hex to the socket before entry
# Key mass properties authored at start-up — diagonal inertia, COM on the working-arm axis:
SYM_INERTIA = (5.0e-4, 5.0e-4, 1.0e-4, 0.08)  # (Ixx, Iyy, Izz, com_z); 210 mm arm
DRIVE_MAX = 6000
K_LIFT_STEPS, K_GLIDE_STEPS, K_ALIGN_STEPS, K_INSERT_STEPS = 240, 240, 192, 144
K_DESCEND_STEPS, K_PARK_STEPS = 168, 300
ALIGN_TOL = math.radians(2.0)  # yaw error under which the key may descend into the socket
PARK_TIP_Z = 0.004        # parked key: tip this far off the table, standing at its spawn spot

# ----- legs 2 + 3: the stick and card PD "hand" (pc_ram_smoke / pc_gpu_smoke's proven values) ----
CROSS_Z = 0.240           # part-origin height while crossing the case rim (clears 195 mm walls)
RAM_ALIGN_Z = 0.011       # stick hover height over the slot: blade 2 mm above the 9 mm stop tops
PRESS_TGT = -0.0005       # press z target below the seated origin (sustained push until bottomed)
RAM_PRESS_DONE = 0.0042   # blade depth below the slot mouth to call a stick pressed (stroke 4.44)
GPU_SLIDE_OFF = 0.028     # forward (-x) offset while placing the card inside (bracket clear)
GPU_SLIDE_Z = 0.0165      # placement + rearward slide height: tab 1 mm above the channel walls
GPU_PRESS_DONE = 0.0048   # tab depth below the slot mouth to call the card pressed (stroke 5 mm)
MAX_RETRIES = 2           # press re-tries per part
RAM_ORDER = (1, 0)        # slot insertion order: far slot 1 first, then 0
P_KP_XY, P_KD_XY = 600.0, 50.0    # N/m, N s/m — holds a part origin on the carry/slot axis
P_KP_Z, P_KD_Z = 200.0, 30.0      # N/m, N s/m — vertical carry/press servo
P_KP_ROT, P_KD_ROT = 4.0, 0.4     # N m/rad, N m s/rad — rights a part to the seated orientation
F_XY_CAP, F_Z_CAP = 30.0, 20.0    # N — force authority around the weight feedforward
T_CAP = 3.0                       # N m — torque authority
P_LIFT_STEPS, P_CROSS_STEPS, P_DROP_STEPS, P_ALIGN_STEPS = 600, 420, 420, 240
SLIDE_STEPS = 360
PRESS_STEPS, PRESS_MAX = 480, 1440
RAM_ALIGN_XY_TOL = 0.0004  # m — stick gate at the hover (beats the 0.5 mm end-stop play)
GPU_ALIGN_XY_TOL = 0.0008  # m — card gate at the placement point / end of slide
P_ALIGN_ROT_TOL = math.radians(1.0)

SHOW_END, SETTLE_STEPS = 150, 300


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

    # Gravity-free key: the soft PD wrenches stand in for the steadying hand. Everything else at
    # the scene's lying defaults.
    env = EnvCfg(scene="pc_all", scene_cfg=PcAllAssemblySceneCfg(key_disable_gravity=True),
                 robot="null", sim_overrides={"dt": DT}).build(num_envs=args.num_envs, device=device)
    sc: PcAllAssemblyScene = env.scene  # type: ignore[assignment]
    n = env.num_envs
    B = sc.cfg.num_holes if args.holes <= 0 else min(args.holes, sc.cfg.num_holes)
    no_action = torch.empty(n, 0, device=device)
    bolts, key, case, card = sc.bolts, sc.key, sc.case, sc.card
    zero3 = torch.zeros(n, 1, 3, device=device)
    render = (not args.headless) or livestream_on
    ram_weight = sc.cfg.ram_mass * GRAV
    card_weight = sc.cfg.card_mass * GRAV

    # The bolts' kinematics, the inserts' disabled collision, and the screw joints are the
    # SCENE's (`screw_mechanic`) — this smoke asserts its planning constants match the scene's.
    assert sc.cfg.screw_mechanic, "this smoke exercises the scene's screw mechanic"
    assert abs(sc.cfg.stage_depth - STAGE_DEPTH) < 1e-9 and abs(sc.cfg.stage_yaw - STAGE_YAW) < 1e-9
    assert abs(sc.SCREW_PITCH - PITCH) < 1e-9 and abs(sc.SCREW_LASH_HALF - LASH_HALF) < 1e-9
    assert abs(sc.cfg.thread_len - THREAD_LEN) < 1e-9 and abs(sc.SCREW_SEAT_MARGIN - SEAT_MARGIN) < 1e-9
    from pxr import Gf, UsdPhysics
    stage = env.stage
    for e in range(n):
        prim = stage.GetPrimAtPath(f"/World/envs/env_{e}/Key/allen_key")
        assert prim.IsValid(), f"missing key body prim in env {e}"
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateDiagonalInertiaAttr(tuple(SYM_INERTIA[:3]))
        mass_api.CreateCenterOfMassAttr((0.0, 0.0, SYM_INERTIA[3]))
        mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    cam = writer = None
    if args.video:
        import imageio.v2 as imageio
        from isaaclab.sensors import Camera, CameraCfg
        cam = Camera(CameraCfg(prim_path="/World/cam", update_period=0.0, height=720, width=1280, data_types=["rgb"],
                               spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.01, 100.0))))
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args.video, fps=30)

    env.sim.reset()  # re-parse physics so the USD edits (and camera) are picked up
    env.reset()

    case_pos = case.data.root_pos_w.clone()  # (n, 3): origin ON the board face, at its centre
    board_z = case_pos[:, 2].clone()
    table_z = board_z - sc.cfg.case_lift
    holes_w = case_pos[:, None, :2] + torch.tensor(sc.cfg.hole_xy, device=device)[None, :B]  # (n, B, 2)
    ram_seats_w = [case_pos + torch.tensor(p, device=device) for p in sc.cfg.ram_seat_pos]
    gpu_seat_w = case_pos + torch.tensor(sc.cfg.gpu_seat_pos, device=device)
    gpu_place_w = gpu_seat_w.clone()  # placement point INSIDE the case: bracket forward of the panel
    gpu_place_w[:, 0] -= GPU_SLIDE_OFF
    park_xy = key.data.root_pos_w[:, 0:2].clone()  # the key parks back at its spawn spot

    # ----- camera: one anchor per leg, eased between legs (a slow dolly during transitions) ------
    cam_pose = None
    if cam is not None:
        p0 = case_pos[0]
        anchors = {  # (eye, tgt) per leg, each the single-task smoke's proven framing
            "mb": (torch.tensor([p0[0] - 0.34, p0[1] - 0.36, p0[2] + 0.80], device=device),
                   torch.tensor([p0[0] + 0.02, p0[1], p0[2] + 0.05], device=device)),
            "ram": (torch.tensor([p0[0] - 0.31, p0[1] - 0.18, p0[2] + 0.245], device=device),
                    torch.tensor([p0[0] - 0.080, p0[1] - 0.060, p0[2] + 0.02], device=device)),
            "gpu": (torch.tensor([p0[0] + 0.13, p0[1] - 0.26, p0[2] + 0.40], device=device),
                    torch.tensor([p0[0] + 0.03, p0[1] + 0.02, p0[2] + 0.075], device=device)),
        }
        cam_eye = anchors["mb"][0].clone()
        cam_tgt = anchors["mb"][1].clone()

        def cam_pose() -> tuple[torch.Tensor, torch.Tensor]:
            leg = "mb" if phase.startswith(("show", "k_")) else ("ram" if phase.startswith("r_") else "gpu")
            te, tt = anchors[leg]
            cam_eye.add_(0.025 * (te - cam_eye))  # eased dolly toward the active leg's anchor
            cam_tgt.add_(0.025 * (tt - cam_tgt))
            return cam_eye.unsqueeze(0), cam_tgt.unsqueeze(0)

    print(env.describe(), flush=True)

    # ----- leg-1 helpers: the force-only key hand -------------------------------------------------
    def bolt_depth(b: int) -> torch.Tensor:  # bolt b's tip depth below the board face (m), per env
        return board_z - bolts[b].data.root_pos_w[:, 2]

    def key_wrench(f: torch.Tensor, t: torch.Tensor) -> None:
        """Apply a WORLD wrench to the key in its CURRENT link frame."""
        qk = key.data.root_link_quat_w
        key.set_external_force_and_torque(quat_apply_inverse(qk, f).unsqueeze(1),
                                          quat_apply_inverse(qk, t).unsqueeze(1))

    def drive_key(b: int, ramp: float) -> None:
        """One force-only 'hand' update on the key, everything in bolt `b`'s SOCKET frame: ramped
        press along the bolt's axis + torque-capped velocity-servo twist + soft PD pulling the key
        tip onto the socket axis and the key's axis onto the bolt's."""
        press, w_tgt, t_cap = DRIVE_CTL
        bolt = bolts[b]
        up_b = up_axis_of(bolt.data.root_quat_w)
        f = torch.zeros(n, 3, device=device)
        t = torch.zeros(n, 3, device=device)
        f[:, :] = -press * ramp * up_b
        wz = key.data.root_ang_vel_w[:, 2]
        tgt = torch.where(bolt_depth(b) < STOP_DEPTH, -w_tgt * torch.ones_like(wz), torch.zeros_like(wz))
        t[:, 2] = torch.clamp(KW * (tgt - wz), -t_cap * ramp, t_cap * ramp)
        pos = key.data.root_link_pos_w
        vel = key.data.root_link_lin_vel_w
        f[:, 0:2] += -K_KP_XY * (pos[:, 0:2] - holes_w[:, b]) - K_KD_XY * vel[:, 0:2]
        floor_z = bolt.data.root_pos_w[:, 2] + SOCKET_FLOOR_Z + KEY_TIP_HOVER
        f[:, 2] += -K_KP_ZK * (pos[:, 2] - floor_z) - K_KD_ZK * vel[:, 2]
        up_k = up_axis_of(key.data.root_quat_w)
        t[:, 0:2] += K_KP_TILT * torch.cross(up_k, up_b, dim=-1)[:, 0:2] - K_KD_TILT * key.data.root_ang_vel_w[:, 0:2]
        key_wrench(f, t)

    def yaw_err_to(b: int) -> torch.Tensor:
        """Smallest key yaw change that matches bolt `b`'s hex clocking (mod 60 deg), (n,)."""
        d = (STAGE_YAW - sc.screw_turn[:, b] - yaw_of(key.data.root_quat_w)) % (math.pi / 3)
        return torch.where(d > math.pi / 6, d - math.pi / 3, d)

    def hop_key(tgt_xy: torch.Tensor, tgt_z: torch.Tensor, align_to: int | None = None) -> None:
        """Force-only travel update: PD the key tip toward a world target (no press, tilt righted
        to vertical). The spin is braked, or yaw-servoed onto bolt `align_to`'s hex clocking."""
        pos = key.data.root_link_pos_w
        vel = key.data.root_link_lin_vel_w
        f = torch.zeros(n, 3, device=device)
        t = torch.zeros(n, 3, device=device)
        f[:, 0:2] = -K_KP_XY * (pos[:, 0:2] - tgt_xy) - K_KD_XY * vel[:, 0:2]
        f[:, 2] = -K_KP_Z * (pos[:, 2] - tgt_z) - K_KD_Z * vel[:, 2]
        wz = key.data.root_ang_vel_w[:, 2]
        if align_to is None:
            t[:, 2] = torch.clamp(KW * (0.0 - wz), -DRIVE_CTL[2], DRIVE_CTL[2])
        else:
            t[:, 2] = torch.clamp(K_KP_YAW * yaw_err_to(align_to) - K_KD_YAW * wz, -DRIVE_CTL[2], DRIVE_CTL[2])
        up_k = up_axis_of(key.data.root_quat_w)
        ez = torch.zeros_like(up_k)
        ez[:, 2] = 1.0
        t[:, 0:2] += K_KP_TILT * torch.cross(up_k, ez, dim=-1)[:, 0:2] - K_KD_TILT * key.data.root_ang_vel_w[:, 0:2]
        key_wrench(f, t)

    def park_key() -> None:
        """Hold the parked key standing at its spawn spot (it is gravity-free — the PD is its
        hand for the rest of the run)."""
        hop_key(park_xy, table_z + PARK_TIP_Z)

    # ----- leg-2/3 helpers: the force-only part hand ----------------------------------------------
    ram_k = RAM_ORDER[0]  # active DIMM slot during leg 2

    def part_asset(name: str):
        return card if name == "gpu" else sc.rams[ram_k]

    def part_weight(name: str) -> float:
        return card_weight if name == "gpu" else ram_weight

    def rot_err(name: str) -> torch.Tensor:
        """Axis-angle error (n, 3) from the part's current orientation to the seated one (world
        identity — the case spawns unrotated), expressed in the WORLD frame."""
        return axis_angle_from_quat(part_asset(name).data.root_quat_w)

    def part_hand(name: str, tgt_pos: torch.Tensor, right: bool = True) -> None:
        """One force-only 'hand' update on the active part: clamped PD toward a world position
        target around the weight feedforward, plus (optionally) a righting PD."""
        p = part_asset(name)
        pos = p.data.root_link_pos_w
        vel = p.data.root_link_lin_vel_w
        f = -P_KP_XY * (pos - tgt_pos) - P_KD_XY * vel
        f[:, 2] = -P_KP_Z * (pos[:, 2] - tgt_pos[:, 2]) - P_KD_Z * vel[:, 2]
        f[:, 0:2] = f[:, 0:2].clamp(-F_XY_CAP, F_XY_CAP)
        f[:, 2] = f[:, 2].clamp(-F_Z_CAP, F_Z_CAP) + part_weight(name)
        t = torch.zeros_like(f)
        if right:
            t = (-P_KP_ROT * rot_err(name) - P_KD_ROT * p.data.root_ang_vel_w).clamp(-T_CAP, T_CAP)
        qc = p.data.root_link_quat_w
        p.set_external_force_and_torque(quat_apply_inverse(qc, f).unsqueeze(1),
                                        quat_apply_inverse(qc, t).unsqueeze(1))

    def ram_depth() -> torch.Tensor:  # active stick's blade depth below its slot mouth (m)
        return sc.ram_engaged()[:, ram_k]

    def gpu_depth() -> torch.Tensor:  # card tab depth below the PCIe slot mouth (m)
        return sc.gpu_engaged()

    def part_xy_err(name: str, ref: torch.Tensor) -> torch.Tensor:
        return (part_asset(name).data.root_link_pos_w[:, 0:2] - ref[:, 0:2]).norm(dim=-1)

    def step(i: int) -> None:
        capture = writer is not None and i % args.cap == 0
        if capture:
            cam.set_world_poses_from_view(*cam_pose())
        env.step(no_action, render=capture or render)
        if capture:
            import numpy as np
            cam.update(env.dt)
            img = cam.data.output["rgb"][0].detach().cpu().numpy()
            if img.dtype != np.uint8:
                img = (img.clip(0, 1) * 255).astype(np.uint8)
            writer.append_data(img[..., :3])

    # Scale phase STEP budgets by (1/240)/dt so the sim TIME per phase stays constant.
    ts = (1.0 / 240.0) / DT
    show_end, settle_steps = int(SHOW_END * ts), int(SETTLE_STEPS * ts)
    drive_max = int(DRIVE_MAX * ts)
    k_lift_steps, k_glide_steps = int(K_LIFT_STEPS * ts), int(K_GLIDE_STEPS * ts)
    k_align_steps, k_insert_steps = int(K_ALIGN_STEPS * ts), int(K_INSERT_STEPS * ts)
    k_descend_steps, k_park_steps = int(K_DESCEND_STEPS * ts), int(K_PARK_STEPS * ts)
    ramp_steps = int(RAMP_STEPS * ts)
    p_lift_steps, p_cross_steps = int(P_LIFT_STEPS * ts), int(P_CROSS_STEPS * ts)
    p_drop_steps, p_align_steps = int(P_DROP_STEPS * ts), int(P_ALIGN_STEPS * ts)
    slide_steps = int(SLIDE_STEPS * ts)
    press_steps, press_max = int(PRESS_STEPS * ts), int(PRESS_MAX * ts)
    log_every = max(1, int(300 * ts))

    # ----- run ------------------------------------------------------------------------------------
    key_turn = torch.zeros(n, device=device)  # cumulative key rotation over the active hole
    prev_key_yaw = torch.zeros(n, device=device)
    hole_gain = torch.zeros(n, B, device=device)
    slip_last = torch.zeros(n, device=device)
    glide_from = torch.zeros(n, 2, device=device)
    lift_from3 = torch.zeros(n, 3, device=device)
    cross_from = torch.zeros(n, 2, device=device)
    drop_from = torch.zeros(n, device=device)
    press_from = torch.zeros(n, device=device)
    active = 0        # the hole the key is at (or flying toward)
    ram_seq = 0       # index into RAM_ORDER
    retries = 0
    retries_total = 0
    phase, i, marker = "show", 0, 0
    while True:
        i += 1
        ramp = min(1.0, (i - marker) / ramp_steps) if ramp_steps > 0 else 1.0
        key_parked = phase.startswith(("r_", "g_")) or phase == "settle"

        if phase == "show":
            if i >= show_end:  # the bolts spawned staged; the key flies over from the table
                prev_key_yaw = yaw_of(key.data.root_quat_w)
                phase, marker = "k_lift", i
        # ---- leg 1: the motherboard -------------------------------------------------------------
        elif phase == "k_lift":  # rise (and right itself), rate-limited; the FIRST approach goes
            # to wall-crossing height (the key starts outside the case), later hops stay low
            pos = key.data.root_link_pos_w
            lift_z = board_z + (KEY_CROSS_Z if active == 0 else TRAVEL_Z)
            tgt_z = torch.minimum(pos[:, 2] + 0.03, lift_z)
            hop_key(pos[:, 0:2], tgt_z)
            up_here = (lift_z - pos[:, 2]) < 0.004
            upright = up_axis_of(key.data.root_quat_w)[:, 2] > math.cos(math.radians(5.0))
            if bool((up_here & upright).all()) or i - marker >= (2 if active == 0 else 1) * k_lift_steps:
                glide_from = key.data.root_link_pos_w[:, 0:2].clone()
                phase, marker = "k_glide", i
        elif phase == "k_glide":  # ease over to the target hole (high on the first, wall-crossing
            # approach; just over the standing heads between holes)
            s = smoothstep((i - marker) / k_glide_steps)
            tgt = glide_from + s * (holes_w[:, active] - glide_from)
            hop_key(tgt, board_z + (KEY_CROSS_Z if active == 0 else TRAVEL_Z))
            arrived = (key.data.root_link_pos_w[:, 0:2] - holes_w[:, active]).norm(dim=-1) < 0.003
            if (i - marker >= k_glide_steps and bool(arrived.all())) or i - marker >= 2 * k_glide_steps:
                phase, marker = ("k_descend" if active == 0 else "k_align"), i
                drop_from = key.data.root_link_pos_w[:, 2].clone()
        elif phase == "k_descend":  # (first hole only) sink from crossing height to travel height
            s = smoothstep((i - marker) / k_descend_steps)
            tgt_z = drop_from + s * (board_z + TRAVEL_Z - drop_from)
            hop_key(holes_w[:, active], tgt_z)
            if i - marker >= k_descend_steps:
                phase, marker = "k_align", i
        elif phase == "k_align":  # hover over the socket, servo the hex clocking into register
            hop_key(holes_w[:, active], board_z + TRAVEL_Z, align_to=active)
            centred = (key.data.root_link_pos_w[:, 0:2] - holes_w[:, active]).norm(dim=-1) < 0.0015
            aligned = (yaw_err_to(active).abs() < ALIGN_TOL) & (key.data.root_ang_vel_w[:, 2].abs() < 0.5)
            if bool((aligned & centred).all()) or i - marker >= k_align_steps:
                lift_from3[:, 2] = key.data.root_link_pos_w[:, 2]
                phase, marker = "k_insert", i
        elif phase == "k_insert":  # descend into the open hex, clocking held
            s = smoothstep((i - marker) / k_insert_steps)
            floor_z = bolts[active].data.root_pos_w[:, 2] + SOCKET_FLOOR_Z + KEY_TIP_HOVER
            hop_key(holes_w[:, active], lift_from3[:, 2] + s * (floor_z - lift_from3[:, 2]), align_to=active)
            kgap = key.data.root_link_pos_w[:, 2] - floor_z
            centred = (key.data.root_link_pos_w[:, 0:2] - holes_w[:, active]).norm(dim=-1) < 0.0015
            if bool(((kgap.abs() < 0.0005) & centred).all()):
                prev_key_yaw = yaw_of(key.data.root_quat_w)
                key_turn = torch.zeros(n, device=device)
                phase, marker = "k_drive", i
            elif i - marker >= 2 * k_insert_steps:  # fallback, keeps the smoke robust (logged)
                print(f"  WARN: insertion at hole {active} timed out — reseating by teleport", flush=True)
                yaw = STAGE_YAW - sc.screw_turn[:, active]
                kt = torch.zeros(n, 13, device=device)
                kt[:, 0:2] = holes_w[:, active]
                kt[:, 2] = bolts[active].data.root_pos_w[:, 2] + SOCKET_FLOOR_Z + KEY_TIP_HOVER
                kt[:, 3] = torch.cos(yaw / 2)
                kt[:, 6] = torch.sin(yaw / 2)
                key.write_root_state_to_sim(kt, torch.arange(n, device=device))
                prev_key_yaw = yaw_of(key.data.root_quat_w)
                key_turn = torch.zeros(n, device=device)
                phase, marker = "k_drive", i
        elif phase == "k_drive":
            drive_key(active, ramp)
            if bool((bolt_depth(active) >= STOP_DEPTH).all()) or i - marker >= drive_max:
                key.set_external_force_and_torque(zero3, zero3)  # applied wrench persists — zero it
                hole_gain[:, active] = bolt_depth(active) - STAGE_DEPTH
                slip_last = torch.rad2deg(key_turn - sc.screw_turn[:, active])
                if active + 1 < B:
                    active += 1
                    phase, marker = "k_lift", i
                else:
                    phase, marker = "k_exit", i
        elif phase == "k_exit":  # rise out of the last socket to wall-crossing height
            pos = key.data.root_link_pos_w
            tgt_z = torch.minimum(pos[:, 2] + 0.03, board_z + KEY_CROSS_Z)
            hop_key(holes_w[:, active], tgt_z)
            if bool(((board_z + KEY_CROSS_Z - pos[:, 2]) < 0.004).all()) or i - marker >= 2 * k_lift_steps:
                glide_from = key.data.root_link_pos_w[:, 0:2].clone()
                phase, marker = "k_park", i
        elif phase == "k_park":  # carry the key back over the wall line to its spawn spot
            s = smoothstep((i - marker) / k_park_steps)
            tgt = glide_from + s * (park_xy - glide_from)
            hop_key(tgt, board_z + KEY_CROSS_Z)
            arrived = (key.data.root_link_pos_w[:, 0:2] - park_xy).norm(dim=-1) < 0.005
            if (i - marker >= k_park_steps and bool(arrived.all())) or i - marker >= 2 * k_park_steps:
                drop_from = key.data.root_link_pos_w[:, 2].clone()
                phase, marker = "k_park_drop", i
        elif phase == "k_park_drop":  # set the key down standing at its spawn spot (PD-held)
            s = smoothstep((i - marker) / k_descend_steps)
            hop_key(park_xy, drop_from + s * (table_z + PARK_TIP_Z - drop_from))
            if i - marker >= k_descend_steps + 30:
                print(f"  board fastened ({B} bolts) — key parked; starting the memory", flush=True)
                lift_from3 = sc.rams[ram_k].data.root_link_pos_w.clone()
                phase, marker = "r_lift", i
        # ---- leg 2: the memory --------------------------------------------------------------------
        elif phase == "r_lift":  # rise off the table to the rim-crossing height while righting
            s = smoothstep((i - marker) / p_lift_steps)
            tgt = lift_from3.clone()
            tgt[:, 2] = lift_from3[:, 2] + s * (board_z + CROSS_Z - lift_from3[:, 2])
            part_hand("ram", tgt)
            up_here = (board_z + CROSS_Z - sc.rams[ram_k].data.root_link_pos_w[:, 2]).abs() < 0.005
            upright = rot_err("ram").norm(dim=-1) < math.radians(5.0)
            if bool((up_here & upright).all()) or i - marker >= 2 * p_lift_steps:
                cross_from = sc.rams[ram_k].data.root_link_pos_w[:, 0:2].clone()
                phase, marker = "r_cross", i
        elif phase == "r_cross":  # glide over the rim to above the slot, at crossing height
            s = smoothstep((i - marker) / p_cross_steps)
            tgt = torch.zeros(n, 3, device=device)
            tgt[:, 0:2] = cross_from + s * (ram_seats_w[ram_k][:, 0:2] - cross_from)
            tgt[:, 2] = board_z + CROSS_Z
            part_hand("ram", tgt)
            arrived = part_xy_err("ram", ram_seats_w[ram_k]) < 0.003
            if (i - marker >= p_cross_steps and bool(arrived.all())) or i - marker >= 2 * p_cross_steps:
                drop_from = sc.rams[ram_k].data.root_link_pos_w[:, 2].clone()
                phase, marker = "r_drop", i
        elif phase == "r_drop":  # descend over the slot to the hover above the end stops
            s = smoothstep((i - marker) / p_drop_steps)
            tgt = ram_seats_w[ram_k].clone()
            tgt[:, 2] = drop_from + s * (board_z + RAM_ALIGN_Z - drop_from)
            part_hand("ram", tgt)
            if i - marker >= p_drop_steps:
                phase, marker = "r_align", i
        elif phase == "r_align":  # settle at the hover: the 0.4 mm gate beats the funnel AND the
            tgt = ram_seats_w[ram_k].clone()  # 0.5 mm end-stop play, so the blade enters clean
            tgt[:, 2] = board_z + RAM_ALIGN_Z
            part_hand("ram", tgt)
            perr = part_xy_err("ram", ram_seats_w[ram_k])
            still = sc.rams[ram_k].data.root_link_lin_vel_w.norm(dim=-1) < 0.01
            ok = (perr < RAM_ALIGN_XY_TOL) & (rot_err("ram").norm(dim=-1) < P_ALIGN_ROT_TOL) & still
            if bool(ok.all()) or i - marker >= 2 * p_align_steps:
                press_from = sc.rams[ram_k].data.root_link_pos_w[:, 2].clone()
                phase, marker = "r_press", i
        elif phase == "r_press":  # straight down; the channel funnel guides the last 4.4 mm
            s = smoothstep((i - marker) / press_steps)
            tgt = ram_seats_w[ram_k].clone()
            tgt[:, 2] = press_from + s * (ram_seats_w[ram_k][:, 2] + PRESS_TGT - press_from)
            part_hand("ram", tgt)
            if bool((ram_depth() >= RAM_PRESS_DONE).all()):
                sc.rams[ram_k].set_external_force_and_torque(zero3, zero3)  # wrench persists — zero it
                phase, marker = "r_handoff", i
            elif i - marker >= press_max:
                if retries < MAX_RETRIES:
                    retries += 1
                    retries_total += 1
                    print(f"  WARN: press on slot {ram_k} stalled at "
                          f"{float(ram_depth().mean() * 1e3):+.2f} mm — retry {retries}/{MAX_RETRIES}", flush=True)
                    phase, marker = "r_reseat", i
                else:
                    print(f"  WARN: press on slot {ram_k} exhausted its retries — releasing as-is", flush=True)
                    sc.rams[ram_k].set_external_force_and_torque(zero3, zero3)
                    phase, marker = "r_handoff", i
        elif phase == "r_reseat":  # back up to the hover (open air above the slot) and re-align
            tgt = ram_seats_w[ram_k].clone()
            tgt[:, 2] = board_z + RAM_ALIGN_Z
            part_hand("ram", tgt)
            if i - marker >= p_align_steps:
                press_from = sc.rams[ram_k].data.root_link_pos_w[:, 2].clone()
                phase, marker = "r_press", i
        elif phase == "r_handoff":  # hands off this stick; next stick, or the card if both seated
            ram_seq += 1
            if ram_seq < len(RAM_ORDER):
                ram_k = RAM_ORDER[ram_seq]
                retries = 0
                lift_from3 = sc.rams[ram_k].data.root_link_pos_w.clone()
                phase, marker = "r_lift", i
            else:
                print("  memory seated — starting the graphics card", flush=True)
                retries = 0
                lift_from3 = card.data.root_link_pos_w.clone()
                phase, marker = "g_lift", i
        # ---- leg 3: the graphics card -------------------------------------------------------------
        elif phase == "g_lift":  # rise off the table to the rim-crossing height while righting
            s = smoothstep((i - marker) / p_lift_steps)
            tgt = lift_from3.clone()
            tgt[:, 2] = lift_from3[:, 2] + s * (board_z + CROSS_Z - lift_from3[:, 2])
            part_hand("gpu", tgt)
            up_here = (board_z + CROSS_Z - card.data.root_link_pos_w[:, 2]).abs() < 0.005
            upright = rot_err("gpu").norm(dim=-1) < math.radians(5.0)
            if bool((up_here & upright).all()) or i - marker >= 2 * p_lift_steps:
                cross_from = card.data.root_link_pos_w[:, 0:2].clone()
                phase, marker = "g_cross", i
        elif phase == "g_cross":  # glide over the rim to the PLACEMENT point, at crossing height
            s = smoothstep((i - marker) / p_cross_steps)
            tgt = torch.zeros(n, 3, device=device)
            tgt[:, 0:2] = cross_from + s * (gpu_place_w[:, 0:2] - cross_from)
            tgt[:, 2] = board_z + CROSS_Z
            part_hand("gpu", tgt)
            arrived = part_xy_err("gpu", gpu_place_w) < 0.003
            if (i - marker >= p_cross_steps and bool(arrived.all())) or i - marker >= 2 * p_cross_steps:
                drop_from = card.data.root_link_pos_w[:, 2].clone()
                phase, marker = "g_drop", i
        elif phase == "g_drop":  # place the card INSIDE the case, bracket forward of the rear panel
            s = smoothstep((i - marker) / p_drop_steps)
            tgt = gpu_place_w.clone()
            tgt[:, 2] = drop_from + s * (board_z + GPU_SLIDE_Z - drop_from)
            part_hand("gpu", tgt)
            if i - marker >= p_drop_steps:
                phase, marker = "g_align", i
        elif phase == "g_align":  # settle at the placement point before the rearward slide
            tgt = gpu_place_w.clone()
            tgt[:, 2] = board_z + GPU_SLIDE_Z
            part_hand("gpu", tgt)
            perr = part_xy_err("gpu", gpu_place_w)
            still = card.data.root_link_lin_vel_w.norm(dim=-1) < 0.01
            ok = (perr < GPU_ALIGN_XY_TOL) & (rot_err("gpu").norm(dim=-1) < P_ALIGN_ROT_TOL) & still
            if bool(ok.all()) or i - marker >= 2 * p_align_steps:
                phase, marker = "g_slide", i
        elif phase == "g_slide":  # rearward: the bracket/ports pass through the I/O panel cutout
            s = smoothstep((i - marker) / slide_steps)
            tgt = gpu_place_w.clone()
            tgt[:, 0] = gpu_place_w[:, 0] + s * GPU_SLIDE_OFF
            tgt[:, 2] = board_z + GPU_SLIDE_Z
            part_hand("gpu", tgt)
            still = card.data.root_link_lin_vel_w.norm(dim=-1) < 0.01
            done = (part_xy_err("gpu", gpu_seat_w) < GPU_ALIGN_XY_TOL) & still
            if (i - marker >= slide_steps and bool(done.all())) or i - marker >= 2 * slide_steps:
                press_from = card.data.root_link_pos_w[:, 2].clone()
                phase, marker = "g_press", i
        elif phase == "g_press":  # straight down; the channel funnel guides the last 5 mm
            s = smoothstep((i - marker) / press_steps)
            tgt = gpu_seat_w.clone()
            tgt[:, 2] = press_from + s * (gpu_seat_w[:, 2] + PRESS_TGT - press_from)
            part_hand("gpu", tgt)
            if bool((gpu_depth() >= GPU_PRESS_DONE).all()):
                card.set_external_force_and_torque(zero3, zero3)  # wrench persists — zero it
                phase, marker = "settle", i
            elif i - marker >= press_max:
                if retries < MAX_RETRIES:
                    retries += 1
                    retries_total += 1
                    print(f"  WARN: card press stalled at {float(gpu_depth().mean() * 1e3):+.2f} mm — "
                          f"retry {retries}/{MAX_RETRIES}", flush=True)
                    phase, marker = "g_reseat", i
                else:
                    print("  WARN: card press exhausted its retries — releasing as-is", flush=True)
                    card.set_external_force_and_torque(zero3, zero3)
                    phase, marker = "settle", i
        elif phase == "g_reseat":  # back up to slide height ONLY (the bracket sits in the cutout —
            tgt = gpu_seat_w.clone()  # rising higher would jam it on the panel above the opening)
            tgt[:, 2] = board_z + GPU_SLIDE_Z
            part_hand("gpu", tgt)
            if i - marker >= p_align_steps:
                press_from = card.data.root_link_pos_w[:, 2].clone()
                phase, marker = "g_press", i
        else:  # settle: hands off everything (the key stays PD-parked) — seats must hold
            if i - marker >= settle_steps:
                break

        if key_parked:
            park_key()  # the gravity-free key needs its hand for the rest of the run
        step(i)
        if phase != "show":  # the scene's mechanic advances the joints; track our own key spin
            kcur = yaw_of(key.data.root_quat_w)
            key_turn = key_turn - _wrap(kcur - prev_key_yaw)
            prev_key_yaw = kcur

        if i % log_every == 0:
            if phase.startswith(("show", "k_")):
                d = bolt_depth(active) * 1e3
                slip = torch.rad2deg(key_turn - sc.screw_turn[:, active])
                print(f"  step {i:5d} [{phase:11s}] hole {active} | tip depth {d.mean():+6.2f}mm | bolt "
                      f"{torch.rad2deg(sc.screw_turn[:, active]).mean():+7.0f}deg | key-bolt slip "
                      f"{slip.mean():+6.1f}deg", flush=True)
            elif phase.startswith("r_"):
                e = rot_err("ram")
                print(f"  step {i:5d} [{phase:11s}] slot {ram_k} | depth {float(ram_depth().mean() * 1e3):+6.2f}mm | "
                      f"xy err {float(part_xy_err('ram', ram_seats_w[ram_k]).mean() * 1e3):5.2f}mm | "
                      f"rot err {float(torch.rad2deg(e.norm(dim=-1)).mean()):5.2f}deg", flush=True)
            else:
                e = rot_err("gpu")
                print(f"  step {i:5d} [{phase:11s}] card | depth {float(gpu_depth().mean() * 1e3):+6.2f}mm | "
                      f"xy err {float(part_xy_err('gpu', gpu_seat_w).mean() * 1e3):5.2f}mm | "
                      f"rot err {float(torch.rad2deg(e.norm(dim=-1)).mean()):5.2f}deg", flush=True)

    if writer is not None:
        writer.close()
        print("MP4:", args.video, flush=True)

    seated = sc.seated()  # (n, 7 + 2 + 1) — bolts, sticks, card, re-checked after the final settle
    all_ok = seated.all(dim=1)
    nb, ns = sc.cfg.num_holes, sc.cfg.num_slots
    revs = sc.screw_turn[:, :B] / (2 * math.pi)
    gain_mm = hole_gain * 1e3
    turned = revs > 0.5
    mm_per_rev = float((gain_mm[turned] / revs[turned]).median()) if turned.any() else float("nan")
    ram_d = sc.ram_engaged()
    per_hole = " ".join(f"h{b}:{float(gain_mm[:, b].mean()):+.1f}mm/{float(revs[:, b].mean()):.1f}rev"
                        for b in range(B))
    per_slot = " | ".join(
        f"slot{j}: depth {float(ram_d[:, j].mean() * 1e3):+.2f} mm, xy "
        f"{float((sc.rams[j].data.root_pos_w[:, 0:2] - ram_seats_w[j][:, 0:2]).norm(dim=-1).mean() * 1e3):.2f} mm"
        for j in range(ns)
    )
    print(f"PC-ALL | complete {int(all_ok.sum())}/{n} envs ({int(seated.sum())}/{(nb + ns + 1) * n} parts: "
          f"{int(seated[:, :nb].sum())}/{nb * n} bolts, {int(seated[:, nb:nb + ns].sum())}/{ns * n} sticks, "
          f"{int(seated[:, -1].sum())}/{n} card) | bolts {mm_per_rev:.2f} mm/rev (pitch {PITCH_MM:.1f}), "
          f"last slip {float(slip_last.mean()):+.1f}deg | {per_hole} | {per_slot} | "
          f"card depth {float(sc.gpu_engaged().mean() * 1e3):+.2f} mm, xy "
          f"{float((card.data.root_pos_w[:, 0:2] - gpu_seat_w[:, 0:2]).norm(dim=-1).mean() * 1e3):.2f} mm | "
          f"press retries {retries_total}", flush=True)
    close_and_exit(env, app)


if __name__ == "__main__":
    main()

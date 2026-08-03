"""Physics smoke test for PcMotherboardAssemblyScene — one ALLEN KEY fastens all 7 motherboard
bolts. The key<->socket contact is LIVE and the key is driven purely by forces. Each bolt is a
kinematic screw joint — the SCENE's thread mechanic (`screw_mechanic`, on by default): it
follows the key's measured rotation through the hex lash (one-way, like a frictional thread)
and descends on the 1 mm-pitch helix, with hard stops; the insert's collision is disabled and
bolts not being driven hold their pose. This smoke only flies the key — staging, kinematics,
and the joints all belong to the scene, and the constants below are asserted against it.

All bolts stage pre-engaged (`STAGE_DEPTH` deep — the run is the final tightening of an
already-started board); the key starts where it lies on the table and never teleports: it is
flown to each hole under force-only PD (lift clear of the standing heads, glide over, align its
yaw to the socket's nearest hex clocking, descend into the socket) and then driven like a hand
would — a ramped press along the bolt's axis, a torque-capped velocity-servo twist, and soft
xy-centering / tilt-righting PD wrenches (gravity-free key).

Phases: show -> stage -> [lift -> glide -> align -> insert -> drive] x 7 -> settle. Verdict:
seated count, per-hole depth gained per rev vs the 1.0 mm pitch, and key->bolt slip angle.

python -m robobench.suites.assembly.smokes.pc_motherboard_smoke --livestream 2
python -m robobench.suites.assembly.smokes.pc_motherboard_smoke \
    --headless --enable_cameras --video robobench/suites/assembly/videos/pc_motherboard.mp4
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
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402
from robobench.core import EnvCfg  # noqa: E402
from robobench.suites.assembly.scenes import PcMotherboardAssemblySceneCfg  # noqa: E402
from robobench.suites.assembly.smokes import close_and_exit  # noqa: E402

if TYPE_CHECKING:
    from robobench.suites.assembly.scenes import PcMotherboardAssemblyScene

# Bolt-local geometry baked into the committed bolt USD (bolt origin = thread TIP, +z up):
THREAD_LEN = 0.0124       # head bottom above the tip: tip depth at which the head bottoms out
SOCKET_FLOOR_Z = 0.01775  # hex recess floor
KEY_TIP_HOVER = 0.0001    # key tip staged this far above the socket floor
STAGE_DEPTH = 0.006       # bolt tip depth below the board face at stage (m)
STAGE_YAW = math.pi       # bolt (and key) yaw at stage (a k*60 deg hex clocking)
SEAT_MARGIN = 0.0001      # screw-joint hard stop: head held this far above the board (never preloads it)
STOP_DEPTH = 0.0118       # stop twisting at this tip depth (m) — just before the head bottoms at 12.4 mm
TRAVEL_Z = 0.030          # key TIP height above the board while hopping (clears the standing heads)
# Drive parameters (module constants, like the sibling smokes). The twist is a torque-capped
# velocity servo: tau = clamp(KW * (w_tgt - wz), -cap, +cap).
DT = 1.0 / 240.0          # sim timestep
PITCH = 0.001             # screw-joint thread pitch (m per revolution; M8-scaled)
PITCH_MM = PITCH * 1e3
KW = 0.05                 # twist servo gain (N m s/rad)
DRIVE_CTL = (1.0, 2.5, 0.03)  # (press N, spin target rad/s, twist torque cap N m)
LASH_HALF = math.radians(8.0)  # key rotation before the hex flats engage (half the 15.9 deg lash)
RAMP_STEPS = 120          # press/twist authority ramp-in at each drive start (steps at dt=1/240)
# Soft hand-steadying PD wrenches on the key:
KP_XY, KD_XY = 100.0, 4.0     # N/m, N s/m — recenters the key tip on the socket axis
KP_TILT, KD_TILT = 0.5, 0.02  # N m/rad, N m s/rad — rights the key onto the bolt axis (spin free)
KP_Z, KD_Z = 60.0, 5.0        # N/m, N s/m — the hop's vertical hold (no press while travelling)
KP_ZK, KD_ZK = 300.0, 8.0     # N/m, N s/m — keeps the key tip riding the descending socket floor
KP_YAW, KD_YAW = 0.01, 0.002  # N m/rad, N m s/rad — aligns the key's hex to the socket before entry
# Key mass properties authored at start-up — diagonal inertia, COM on the working-arm axis:
SYM_INERTIA = (5.0e-4, 5.0e-4, 1.0e-4, 0.08)  # (Ixx, Iyy, Izz, com_z); 210 mm arm
# Phase step budgets at dt=1/240 (rescaled at run time so sim TIME per phase is constant).
SHOW_END, DRIVE_MAX, SETTLE_STEPS = 150, 6000, 300
LIFT_STEPS, GLIDE_STEPS, ALIGN_STEPS, INSERT_STEPS = 240, 240, 192, 144
ALIGN_TOL = math.radians(2.0)  # yaw error under which the key may descend into the socket


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

    # Gravity-free key: the soft PD wrenches stand in for the steadying hand.
    env = EnvCfg(scene="pc_motherboard", scene_cfg=PcMotherboardAssemblySceneCfg(key_disable_gravity=True),
                 robot="null", sim_overrides={"dt": DT}).build(num_envs=args.num_envs, device=device)
    sc: PcMotherboardAssemblyScene = env.scene  # type: ignore[assignment]
    n = env.num_envs
    B = sc.cfg.num_holes if args.holes <= 0 else min(args.holes, sc.cfg.num_holes)
    ids = torch.arange(n, device=device)
    no_action = torch.empty(n, 0, device=device)
    bolts, key, case = sc.bolts, sc.key, sc.case
    zero3 = torch.zeros(n, 1, 3, device=device)
    render = (not args.headless) or livestream_on

    # Author the key's mass properties (`SYM_INERTIA`). The bolts' kinematics, the inserts'
    # disabled collision, and the screw joints are the SCENE's (`screw_mechanic`) — this smoke
    # asserts its planning constants match the scene's and otherwise just flies the key.
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
    holes_w = case_pos[:, None, :2] + torch.tensor(sc.cfg.hole_xy, device=device)[None, :B]  # (n, B, 2)

    if cam is not None:
        p0 = case_pos[0]
        # View from the case-FRONT side (local -x): the rear wall stands only ~16 mm past the
        # nearest holes and hides them from any +x viewpoint. Aimed high to keep the key in frame.
        eye = torch.tensor([[p0[0] - 0.34, p0[1] - 0.36, p0[2] + 0.80]], dtype=torch.float32, device=device)
        tgt = torch.tensor([[p0[0] + 0.02, p0[1], p0[2] + 0.05]], dtype=torch.float32, device=device)
        cam.set_world_poses_from_view(eye, tgt)
    print(env.describe(), flush=True)

    # The screw joints are the scene's (`sc.screw_turn` is their per-hole rotation state);
    # the bolts spawned already staged in their holes at reset.

    def depth(b: int) -> torch.Tensor:  # bolt b's tip depth below the board face (m), per env
        return board_z - bolts[b].data.root_pos_w[:, 2]

    def reseat_key(b: int) -> None:
        """FALLBACK ONLY (logged): teleport the key into bolt `b`'s socket if a PD insertion ever
        times out — tip 0.1 mm off the floor on the bolt's axis, hex clocking matched."""
        yaw = STAGE_YAW - sc.screw_turn[:, b]
        kt = torch.zeros(n, 13, device=device)
        kt[:, 0:2] = holes_w[:, b]
        kt[:, 2] = bolts[b].data.root_pos_w[:, 2] + SOCKET_FLOOR_Z + KEY_TIP_HOVER
        kt[:, 3] = torch.cos(yaw / 2)
        kt[:, 6] = torch.sin(yaw / 2)
        key.write_root_state_to_sim(kt, ids)

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
        # servo the spin toward -w_tgt (screw-in), authority t_cap; target 0 once travel is done
        wz = key.data.root_ang_vel_w[:, 2]
        tgt = torch.where(depth(b) < STOP_DEPTH, -w_tgt * torch.ones_like(wz), torch.zeros_like(wz))
        t[:, 2] = torch.clamp(KW * (tgt - wz), -t_cap * ramp, t_cap * ramp)
        # soft PD pulling the key tip onto the socket axis and riding the descending floor
        pos = key.data.root_link_pos_w
        vel = key.data.root_link_lin_vel_w
        f[:, 0:2] += -KP_XY * (pos[:, 0:2] - holes_w[:, b]) - KD_XY * vel[:, 0:2]
        floor_z = bolt.data.root_pos_w[:, 2] + SOCKET_FLOOR_Z + KEY_TIP_HOVER
        f[:, 2] += -KP_ZK * (pos[:, 2] - floor_z) - KD_ZK * vel[:, 2]
        # soft tilt righting toward the BOLT's axis; damping only on wx/wy — yaw + spin stay free
        up_k = up_axis_of(key.data.root_quat_w)
        t[:, 0:2] += KP_TILT * torch.cross(up_k, up_b, dim=-1)[:, 0:2] - KD_TILT * key.data.root_ang_vel_w[:, 0:2]
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
        f[:, 0:2] = -KP_XY * (pos[:, 0:2] - tgt_xy) - KD_XY * vel[:, 0:2]
        f[:, 2] = -KP_Z * (pos[:, 2] - tgt_z) - KD_Z * vel[:, 2]
        wz = key.data.root_ang_vel_w[:, 2]
        if align_to is None:
            t[:, 2] = torch.clamp(KW * (0.0 - wz), -DRIVE_CTL[2], DRIVE_CTL[2])
        else:
            t[:, 2] = torch.clamp(KP_YAW * yaw_err_to(align_to) - KD_YAW * wz, -DRIVE_CTL[2], DRIVE_CTL[2])
        up_k = up_axis_of(key.data.root_quat_w)
        ez = torch.zeros_like(up_k)
        ez[:, 2] = 1.0
        t[:, 0:2] += KP_TILT * torch.cross(up_k, ez, dim=-1)[:, 0:2] - KD_TILT * key.data.root_ang_vel_w[:, 0:2]
        key_wrench(f, t)

    def step(i: int) -> None:
        capture = writer is not None and i % args.cap == 0
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
    show_end, drive_max, settle_steps = int(SHOW_END * ts), int(DRIVE_MAX * ts), int(SETTLE_STEPS * ts)
    lift_steps, glide_steps = int(LIFT_STEPS * ts), int(GLIDE_STEPS * ts)
    align_steps, insert_steps = int(ALIGN_STEPS * ts), int(INSERT_STEPS * ts)
    ramp_steps = int(RAMP_STEPS * ts)
    log_every = max(1, int(300 * ts))

    key_turn = torch.zeros(n, device=device)  # cumulative key rotation over the active hole
    prev_key_yaw = torch.zeros(n, device=device)
    hole_gain = torch.zeros(n, B, device=device)
    slip_last = torch.zeros(n, device=device)
    glide_from = torch.zeros(n, 2, device=device)
    active = 0  # the hole the key is at (or flying toward)
    phase, i, marker = "show", 0, 0
    while True:
        i += 1
        ramp = min(1.0, (i - marker) / ramp_steps) if ramp_steps > 0 else 1.0
        if phase == "show":
            if i >= show_end:  # the bolts spawned staged; the key flies over from the table
                prev_key_yaw = yaw_of(key.data.root_quat_w)
                phase, marker = "lift", i
        elif phase == "lift":  # rise (and right itself) to the travel height, rate-limited
            pos = key.data.root_link_pos_w
            tgt_z = torch.minimum(pos[:, 2] + 0.03, board_z + TRAVEL_Z)
            hop_key(pos[:, 0:2], tgt_z)
            up_here = (board_z + TRAVEL_Z - pos[:, 2]) < 0.004
            upright = up_axis_of(key.data.root_quat_w)[:, 2] > math.cos(math.radians(5.0))
            if bool((up_here & upright).all()) or i - marker >= lift_steps:
                glide_from = key.data.root_link_pos_w[:, 0:2].clone()
                phase, marker = "glide", i
        elif phase == "glide":  # ease over to the target hole at travel height
            s = smoothstep((i - marker) / glide_steps)
            tgt = glide_from + s * (holes_w[:, active] - glide_from)
            hop_key(tgt, board_z + TRAVEL_Z)
            arrived = (key.data.root_link_pos_w[:, 0:2] - holes_w[:, active]).norm(dim=-1) < 0.003
            if (i - marker >= glide_steps and bool(arrived.all())) or i - marker >= 2 * glide_steps:
                phase, marker = "align", i
        elif phase == "align":  # hover over the socket, servo the hex clocking into register
            hop_key(holes_w[:, active], board_z + TRAVEL_Z, align_to=active)
            centred = (key.data.root_link_pos_w[:, 0:2] - holes_w[:, active]).norm(dim=-1) < 0.0015
            aligned = (yaw_err_to(active).abs() < ALIGN_TOL) & (key.data.root_ang_vel_w[:, 2].abs() < 0.5)
            if bool((aligned & centred).all()) or i - marker >= align_steps:
                insert_from = key.data.root_link_pos_w[:, 2].clone()
                phase, marker = "insert", i
        elif phase == "insert":  # descend into the open hex, clocking held
            s = smoothstep((i - marker) / insert_steps)
            floor_z = bolts[active].data.root_pos_w[:, 2] + SOCKET_FLOOR_Z + KEY_TIP_HOVER
            hop_key(holes_w[:, active], insert_from + s * (floor_z - insert_from), align_to=active)
            kgap = key.data.root_link_pos_w[:, 2] - floor_z
            centred = (key.data.root_link_pos_w[:, 0:2] - holes_w[:, active]).norm(dim=-1) < 0.0015
            if bool(((kgap.abs() < 0.0005) & centred).all()):
                prev_key_yaw = yaw_of(key.data.root_quat_w)
                key_turn = torch.zeros(n, device=device)
                phase, marker = "drive", i
            elif i - marker >= 2 * insert_steps:  # fallback, keeps the smoke robust
                print(f"  WARN: insertion at hole {active} timed out — reseating by teleport", flush=True)
                reseat_key(active)
                prev_key_yaw = yaw_of(key.data.root_quat_w)
                key_turn = torch.zeros(n, device=device)
                phase, marker = "drive", i
        elif phase == "drive":
            drive_key(active, ramp)
            if bool((depth(active) >= STOP_DEPTH).all()) or i - marker >= drive_max:
                key.set_external_force_and_torque(zero3, zero3)  # applied wrench persists — zero it
                hole_gain[:, active] = depth(active) - STAGE_DEPTH
                slip_last = torch.rad2deg(key_turn - sc.screw_turn[:, active])
                if active + 1 < B:
                    active += 1
                    phase, marker = "lift", i
                else:
                    phase, marker = "settle", i
        else:
            if i - marker >= settle_steps:
                break

        step(i)
        if phase != "show":  # the scene's mechanic advances the joints; track our own key spin
            kcur = yaw_of(key.data.root_quat_w)
            key_turn = key_turn - _wrap(kcur - prev_key_yaw)
            prev_key_yaw = kcur

        if i % log_every == 0:
            d = depth(active) * 1e3
            slip = torch.rad2deg(key_turn - sc.screw_turn[:, active])
            kgap = (key.data.root_link_pos_w[:, 2] - bolts[active].data.root_pos_w[:, 2] - SOCKET_FLOOR_Z) * 1e3
            print(f"  step {i:5d} [{phase:6s}] hole {active} | tip depth {d.mean():+6.2f}mm | bolt "
                  f"{torch.rad2deg(sc.screw_turn[:, active]).mean():+7.0f}deg | key-bolt slip {slip.mean():+6.1f}deg | "
                  f"key-floor {kgap.mean():+5.2f}mm", flush=True)

    if writer is not None:
        writer.close()
        print("MP4:", args.video, flush=True)

    seated = sc.seated()  # (n, num_holes)
    revs = sc.screw_turn[:, :B] / (2 * math.pi)  # (n, B)
    gain_mm = hole_gain * 1e3
    turned = revs > 0.5
    mm_per_rev = float((gain_mm[turned] / revs[turned]).median()) if turned.any() else float("nan")
    per_hole = " ".join(f"h{b}:{float(gain_mm[:, b].mean()):+.1f}mm/{float(revs[:, b].mean()):.1f}rev"
                        for b in range(B))
    print(f"PC-MOTHERBOARD | seated {int(seated.all(dim=1).sum())}/{n} envs "
          f"({int(seated.sum())}/{sc.cfg.num_holes * n} bolts) | {mm_per_rev:.2f} mm/rev "
          f"(pitch {PITCH_MM:.1f}) | last-hole key-bolt slip {float(slip_last.mean()):+.1f}deg | "
          f"depth gains (seat >= {sc.cfg.seat_depth * 1e3:.0f}mm): {per_hole}", flush=True)
    close_and_exit(env, app)


if __name__ == "__main__":
    main()

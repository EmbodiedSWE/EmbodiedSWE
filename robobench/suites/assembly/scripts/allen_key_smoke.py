"""Physics smoke test for AllenBoltAssemblyScene — the allen KEY drives the bolt down REAL SDF
threads. Every body is fully dynamic and the key is driven purely by forces; the bolt<->platform
thread contact and the key<->socket contact are both live.

The bolt is staged upright with its tip just above the hole, so the threads self-engage under
press + twist (staging deeper cannot match the thread phase). The key is driven like a hand
would: a ramped press along the bolt's axis, a torque-capped velocity-servo twist, and soft
xy-centering / tilt-righting PD wrenches (gravity-free key). Two workarounds keep the force
drive stable:

  * WRENCH FRAME: this Isaac Lab checkout converts `is_global=True` external wrenches with link
    poses cached at the FIRST call, so a "world" wrench silently rotates with the body — any xy
    component on a spinning key becomes a rotating force, i.e. an energy pump. The wrench is
    therefore rotated into the key's CURRENT link frame here and applied with is_global=False.
  * DRIVER-SYMMETRIC KEY INERTIA (`SYM_INERTIA`): a free L-key is torque-drive unstable — the
    handle's products of inertia couple the drive torque into tumble faster than the soft
    steadying PD can arrest at this dt. Authored diagonal inertia + on-axis COM make the key
    behave like a stubby T-handle driver, while its collision shapes and visuals stay the L-key.

The drive comes up in two stages because the bolt stands tip-on-crest with no lateral captivity
until the first thread captures: a low-authority ENGAGE (light press, slow spin) until the tip is
~1 turn in, then the full DRIVE. Phases: show -> stage (teleport + short hands-off settle, key
re-seated on the bolt as nested) -> engage -> drive -> settle. Verdict: seated count, depth per
rev vs the 2.0 mm pitch, and key->bolt slip (steady slip within the 15.9 deg hex lash = clean
form-closure drive). Verified 2026-07-09: seated 1/1, 1.92 mm/rev, slip +7.1 deg.

python -m robobench.suites.assembly.scripts.allen_key_smoke --livestream 2
python -m robobench.suites.assembly.scripts.allen_key_smoke \
    --headless --enable_cameras --video robobench/suites/assembly/videos/allen_key.mp4
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
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
from robobench.suites.assembly.scenes import AllenBoltAssemblySceneCfg  # noqa: E402

if TYPE_CHECKING:
    from robobench.suites.assembly.scenes import AllenBoltAssemblyScene

# Bolt-local geometry baked into the committed bolt USD (bolt origin = thread TIP, +z up):
SOCKET_FLOOR_Z = 0.0355  # hex recess floor
KEY_TIP_HOVER = 0.0001   # key tip staged this far above the socket floor
STAGE_GAP = 0.0015       # staging gap (m) between bolt tip and plate top, just above the thread entry
# Drive parameters (module constants, like the sibling smokes). The twist is a torque-capped
# velocity SERVO (tau = clamp(KW*(w_tgt - wz))), not bang-bang: a raw N m-scale torque step slews
# the key several rad/s in ONE 240 Hz step and hammers the socket corners — before the threads
# bite, that lash impact simply knocks the free-standing bolt off the hole.
DT = 1.0 / 240.0         # sim timestep
PITCH_MM = 2.0           # M16 coarse pitch, the expected descent per revolution
STOP_DEPTH = 0.0235      # stop twisting at this tip depth (m) — just before the head bottoms at 24.8 mm
ENGAGE_DEPTH = 0.002     # depth gained past the nested handoff depth (m) to call the thread captured
KW = 0.05                # twist servo gain (N m s/rad)
# (press N, spin target rad/s, twist torque cap N m) per stage — gentle until captured:
ENGAGE_CTL = (1.0, 1.5, 0.04)
DRIVE_CTL = (3.0, 2.5, 0.12)
RAMP_STEPS = 120         # press/twist authority ramp-in at each stage start (steps at dt=1/240)
# Soft hand-steadying PD wrenches on the key:
KP_XY, KD_XY = 100.0, 4.0     # N/m, N s/m — recenters the key tip on the socket axis
KP_TILT, KD_TILT = 0.5, 0.02  # N m/rad, N m s/rad — rights the key onto the bolt axis (spin free)
# Driver-symmetric key mass properties (see the module docstring): (Ixx, Iyy, Izz, com_z).
SYM_INERTIA = (1.5e-4, 1.5e-4, 1.0e-4, 0.025)
# Phase step budgets at dt=1/240 (rescaled at run time so sim TIME per phase is constant).
SHOW_END, STAGE_SETTLE, ENGAGE_MAX, DRIVE_MAX, SETTLE_STEPS = 150, 60, 3600, 12000, 300


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

    # Gravity-free key: the soft PD stands in for the steadying hand, and the L-handle's gravity
    # torque would otherwise have to be fought by that same PD.
    env = EnvCfg(scene="allen_bolt", scene_cfg=AllenBoltAssemblySceneCfg(key_disable_gravity=True),
                 robot="null", sim_overrides={"dt": DT}).build(num_envs=args.num_envs, device=device)
    sc: AllenBoltAssemblyScene = env.scene  # type: ignore[assignment]
    n = env.num_envs
    ids = torch.arange(n, device=device)
    no_action = torch.empty(n, 0, device=device)
    bolt, key, plat = sc.bolts[0], sc.keys[0], sc.platforms[0]
    plate_top = sc.cfg.plate_top
    zero3 = torch.zeros(n, 1, 3, device=device)
    render = (not args.headless) or livestream_on

    # Author the key's driver-symmetric mass properties. Must happen BEFORE the explicit sim
    # reset so the re-parse honours it.
    from pxr import Gf, UsdPhysics
    stage = env.stage
    for e in range(n):
        prim = stage.GetPrimAtPath(f"/World/envs/env_{e}/Key_0/allen_key")
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

    env.sim.reset()  # re-parse physics so the mass edit (and camera) are picked up
    env.reset()

    hole_xy = plat.data.root_pos_w[:, :2].clone()  # insert bore axis == platform origin (bore-centred)
    plat_z = plat.data.root_pos_w[:, 2].clone()

    if cam is not None:
        p0 = plat.data.root_pos_w[0]
        eye = torch.tensor([[p0[0] + 0.35, p0[1] - 0.35, p0[2] + 0.28]], dtype=torch.float32, device=device)
        tgt = torch.tensor([[p0[0], p0[1], p0[2] + 0.07]], dtype=torch.float32, device=device)
        cam.set_world_poses_from_view(eye, tgt)
    print(env.describe(), flush=True)

    def stage_parts() -> None:
        """Teleport the bolt upright with its tip `STAGE_GAP` above the plate top (threads will
        self-engage) and seat the key tip in its socket at the same yaw."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = hole_xy
        st[:, 2] = plat_z + plate_top + STAGE_GAP
        st[:, 3] = 1.0
        bolt.write_root_state_to_sim(st, ids)
        kt = torch.zeros(n, 13, device=device)
        kt[:, 0:2] = hole_xy
        kt[:, 2] = st[:, 2] + SOCKET_FLOOR_Z + KEY_TIP_HOVER
        kt[:, 3] = 1.0  # same hex clocking as the socket (both author corners at k*60 deg)
        key.write_root_state_to_sim(kt, ids)

    def depth() -> torch.Tensor:  # bolt tip depth below the plate top (m), per env
        return plat_z + plate_top - bolt.data.root_pos_w[:, 2]

    def reseat_key() -> None:
        """Teleport the key into the socket of the bolt AS NESTED — tip 0.1 mm off the floor along
        the bolt's own (slightly tilted) axis, hex clocking matched — so the drive starts with zero
        contact preload. Staging against the WORLD axis instead loads the tilted socket's rim, and
        the first press turns that wedge into a depenetration blast."""
        up_b = up_axis_of(bolt.data.root_quat_w)
        kt = torch.zeros(n, 13, device=device)
        kt[:, 0:3] = bolt.data.root_pos_w + up_b * (SOCKET_FLOOR_Z + KEY_TIP_HOVER)
        kt[:, 3:7] = bolt.data.root_quat_w
        key.write_root_state_to_sim(kt, ids)

    def drive_key(ctl: tuple[float, float, float], ramp: float) -> None:
        """One force-only 'hand' update on the key, everything in the SOCKET frame: ramped press
        along the bolt's axis + torque-capped velocity-servo twist + soft PD pulling the key tip
        onto the socket axis and the key's axis onto the bolt's. Forces act at the key's COM
        (measured behaviour of the wrench path; with `SYM_INERTIA` the COM sits on the arm axis,
        so the press is axial). `ctl` = (press N, spin target rad/s, twist cap N m)."""
        press, w_tgt, t_cap = ctl
        up_b = up_axis_of(bolt.data.root_quat_w)
        f = torch.zeros(n, 1, 3, device=device)
        t = torch.zeros(n, 1, 3, device=device)
        f[:, 0, :] = -press * ramp * up_b
        # servo the spin toward -w_tgt (screw-in), authority t_cap; target 0 once travel is done —
        # the servo then BRAKES the key instead of camming it over the hex lobes at the hard stop
        wz = key.data.root_ang_vel_w[:, 2]
        tgt = torch.where(depth() < STOP_DEPTH, -w_tgt * torch.ones_like(wz), torch.zeros_like(wz))
        t[:, 0, 2] = torch.clamp(KW * (tgt - wz), -t_cap * ramp, t_cap * ramp)
        # soft PD pulling the key tip onto the socket axis (at the tip's height); tip (link-frame)
        # position/velocity, which with the on-axis COM are spin-invariant — no pump channel
        pos = key.data.root_link_pos_w
        vel = key.data.root_link_lin_vel_w
        rel = pos - bolt.data.root_pos_w
        axis_pt = bolt.data.root_pos_w + (rel * up_b).sum(-1, keepdim=True) * up_b  # tip proj. on axis
        f[:, 0, 0:2] += -KP_XY * (pos[:, 0:2] - axis_pt[:, 0:2]) - KD_XY * vel[:, 0:2]
        # soft tilt righting toward the BOLT's axis; damping only on wx/wy — yaw + spin stay free
        up_k = up_axis_of(key.data.root_quat_w)
        t[:, 0, 0:2] += KP_TILT * torch.cross(up_k, up_b, dim=-1)[:, 0:2] - KD_TILT * key.data.root_ang_vel_w[:, 0:2]
        # Rotate the WORLD wrench into the key's CURRENT link frame and apply it as a LOCAL wrench
        # (see the module docstring: is_global=True is frame-stale in this Isaac Lab checkout).
        qk = key.data.root_link_quat_w
        key.set_external_force_and_torque(quat_apply_inverse(qk, f[:, 0]).unsqueeze(1),
                                          quat_apply_inverse(qk, t[:, 0]).unsqueeze(1))

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
    show_end, stage_settle = int(SHOW_END * ts), int(STAGE_SETTLE * ts)
    engage_max, drive_max, settle_steps = int(ENGAGE_MAX * ts), int(DRIVE_MAX * ts), int(SETTLE_STEPS * ts)
    ramp_steps = int(RAMP_STEPS * ts)
    log_every = max(1, int(300 * ts))

    def on_axis() -> torch.Tensor:  # a fallen/ejected bolt can read "deep"; require it in the hole
        return (bolt.data.root_pos_w[:, 0:2] - hole_xy).norm(dim=-1) < 0.004

    bolt_turn = torch.zeros(n, device=device)  # cumulative screw-in rotation (rad, +ve = descending)
    key_turn = torch.zeros(n, device=device)
    prev_bolt_yaw = yaw_of(bolt.data.root_quat_w)
    prev_key_yaw = yaw_of(key.data.root_quat_w)
    handoff_depth = None
    phase, i, marker = "show", 0, 0
    while True:
        i += 1
        ramp = min(1.0, (i - marker) / ramp_steps) if ramp_steps > 0 else 1.0
        if phase == "show":
            if i >= show_end:
                stage_parts()
                prev_bolt_yaw = yaw_of(bolt.data.root_quat_w)
                prev_key_yaw = yaw_of(key.data.root_quat_w)
                phase, marker = "stage", i
        elif phase == "stage":  # hands off: let the bolt drop the 1.5 mm gap and nest on the crests
            if i - marker >= stage_settle:
                reseat_key()  # re-seat on the bolt AS NESTED (tilted) — zero preload at handoff
                prev_key_yaw = yaw_of(key.data.root_quat_w)
                handoff_depth = depth().clone()
                phase, marker = "engage", i
        elif phase == "engage":  # gentle press + slow servo spin until the thread captures
            drive_key(ENGAGE_CTL, ramp)
            captured = depth() >= handoff_depth + ENGAGE_DEPTH  # ~1 turn past the nest depth
            if bool((captured & on_axis()).all()) or i - marker >= engage_max:
                phase, marker = "drive", i
        elif phase == "drive":
            drive_key(DRIVE_CTL, ramp)
            if bool(((depth() >= STOP_DEPTH) & on_axis()).all()) or i - marker >= drive_max:
                key.set_external_force_and_torque(zero3, zero3)  # applied wrench persists — zero it
                phase, marker = "settle", i
        else:
            if i - marker >= settle_steps:
                break

        step(i)
        if phase != "show":
            cur = yaw_of(bolt.data.root_quat_w)
            bolt_turn = bolt_turn - _wrap(cur - prev_bolt_yaw)
            prev_bolt_yaw = cur
            kcur = yaw_of(key.data.root_quat_w)
            key_turn = key_turn - _wrap(kcur - prev_key_yaw)
            prev_key_yaw = kcur

        if i % log_every == 0:
            d = depth() * 1e3
            slip = torch.rad2deg(key_turn - bolt_turn)
            tilt = torch.rad2deg(torch.acos(up_axis_of(bolt.data.root_quat_w)[:, 2].clamp(-1, 1)))
            xy = (bolt.data.root_pos_w[:, 0:2] - hole_xy).norm(dim=-1) * 1e3
            print(f"  step {i:5d} [{phase:6s}] | tip depth {d.mean():+6.2f}mm | bolt "
                  f"{torch.rad2deg(bolt_turn).mean():+7.0f}deg | key-bolt slip {slip.mean():+6.1f}deg | "
                  f"tilt {tilt.max():4.1f}deg | xy {xy.max():4.2f}mm", flush=True)
            if phase != "show" and (not torch.isfinite(bolt.data.root_pos_w).all() or float(xy.max()) > 50.0):
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
    print(f"ALLEN-KEY | seated {int(seated.all(dim=1).sum())}/{n} | key drove the bolt "
          f"{float(gain.mean()):+.1f}mm over {float(revs.mean()):.1f} revs = {mm_per_rev:.2f} mm/rev "
          f"(pitch {PITCH_MM:.1f}) | key-bolt slip {float(slip.mean()):+.1f}deg | tip depth mm: "
          f"min={d.min():+.1f} mean={d.mean():+.1f} max={d.max():+.1f} (seat>= {sc.cfg.seat_depth * 1e3:.0f})", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()

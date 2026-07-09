"""Physics smoke test for AllenBoltAssemblyScene — the allen KEY drives the bolt down its thread.

Real SDF thread contact can't survive a second rigid contact pair driving the threaded body, and the
key-in-socket form closure is exactly such a pair. So the bolt<->platform thread is idealized as a
per-step helical projection (bolt z slaved to its accumulated turn at the real pitch, xy pinned, tilt
zeroed, spin free, with Coulomb thread friction and hard stops) and the platform insert's collision is
disabled — while the key<->socket contact stays LIVE, so all drive reaches the bolt through the hex walls.

The key is railed: xy pinned to the hole axis, tip riding the socket floor, tilt zeroed; only its yaw
and spin rate stay free, driven by a capped bang-bang torque.

Phases: show -> stage (bolt pre-engaged, key tip seated) -> drive -> settle. Verdict: bolt revs, depth
gained, and key->bolt slip angle.

python -m robobench.suites.assembly.scripts.allen_key_smoke --livestream 2
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
livestream_on = args.livestream > 0

app = AppLauncher(args).app

from typing import TYPE_CHECKING  # noqa: E402

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import EnvCfg  # noqa: E402
from robobench.suites.assembly.scenes import AllenBoltAssemblySceneCfg  # noqa: E402

if TYPE_CHECKING:
    from robobench.suites.assembly.scenes import AllenBoltAssemblyScene

# Bolt-local geometry baked into the committed bolt USD (bolt origin = thread TIP, +z up):
THREAD_LEN = 0.0248      # head bottom above the tip: tip depth at which the head bottoms out
SOCKET_FLOOR_Z = 0.0355  # hex recess floor
KEY_TIP_HOVER = 0.0001   # key tip held this far above the socket floor (rides the bolt down)
SEAT_MARGIN = 0.0002     # screw-joint hard stop: head held this far above the plate (never preloads it)
EXIT_MARGIN = 0.0005     # end the drive phase once within this of the hard stop
# Screw-joint drive parameters (hardcoded to keep the args simple, like the other smoke files):
DT = 1.0 / 240.0         # sim timestep
TWIST = 0.3              # screw-in torque on the key about z (N m)
TWIST_CAP = 3.0          # cap on the key spin rate while driving (rad/s)
PITCH = 0.002            # screw-joint thread pitch (m per revolution)
THREAD_FRICTION = 0.005  # Coulomb-style thread-friction torque on the bolt (N m); the key's torque must exceed it
START_DEPTH = 0.006      # bolt tip depth below the plate top at stage (m); the joint needs no thread-phase match
# Phase step budgets at dt=1/240 (rescaled at run time so sim TIME per phase is constant).
SHOW_END, DRIVE_MAX, SETTLE_STEPS = 150, 8000, 300


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_of(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_wxyz.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main() -> None:
    device = getattr(args, "device", None) or ("cuda:0" if torch.cuda.is_available() else "cpu")
    robobench.discover()

    # Gravity-free key: its z is railed to follow the bolt, so gravity would only add socket-floor load.
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

    # Idealize the thread: disable ONLY the insert's collision — the key<->socket pair stays live.
    # Must happen BEFORE the explicit sim reset so the re-parse honours it.
    from pxr import UsdPhysics
    stage = env.stage
    for e in range(n):
        prim = stage.GetPrimAtPath(f"/World/envs/env_{e}/Platform_0/platform/thread_insert")
        assert prim.IsValid(), f"thread_insert prim missing in env {e}"
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)

    env.sim.reset()  # re-parse physics so the collision edit is picked up
    env.reset()

    hole_xy = plat.data.root_pos_w[:, :2].clone()  # insert bore axis == platform origin (bore-centred)
    plat_z = plat.data.root_pos_w[:, 2].clone()
    print(env.describe(), flush=True)

    # Screw-joint state; anchors are set at stage time.
    bolt_turn = torch.zeros(n, device=device)  # cumulative screw-in rotation (rad, +ve = descending)
    z0 = torch.zeros(n, device=device)         # bolt z at stage (depth = START_DEPTH there)
    depth_max = THREAD_LEN - SEAT_MARGIN
    turn_max = torch.full((n,), (depth_max - START_DEPTH) * 2 * math.pi / PITCH, device=device)
    turn_min = torch.full((n,), -(START_DEPTH - 0.001) * 2 * math.pi / PITCH, device=device)

    def stage_parts() -> torch.Tensor:
        """Teleport the bolt pre-engaged (upright, tip `START_DEPTH` below the plate top) and seat
        the key tip in its socket at the same yaw."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = hole_xy
        st[:, 2] = plat_z + plate_top - START_DEPTH
        st[:, 3] = 1.0
        bolt.write_root_state_to_sim(st, ids)
        kt = torch.zeros(n, 13, device=device)
        kt[:, 0:2] = hole_xy
        kt[:, 2] = st[:, 2] + SOCKET_FLOOR_Z + KEY_TIP_HOVER
        kt[:, 3] = 1.0  # same hex clocking as the socket (both author corners at k*60 deg)
        key.write_root_state_to_sim(kt, ids)
        return st[:, 2].clone()

    def key_twist(twist: float) -> None:
        """Capped bang-bang screw-in torque about z (CW from above), gated per env on BOTH the spin
        cap and remaining travel — torquing a bolt at its hard stop cams the key over the hex lobes."""
        t = torch.zeros(n, 1, 3, device=device)
        if twist != 0.0:
            wz = key.data.root_ang_vel_w[:, 2]
            drive = (wz > -TWIST_CAP) & (depth() < depth_max - EXIT_MARGIN)
            t[drive, 0, 2] = -twist
        key.set_external_force_and_torque(zero3, t, is_global=True)

    def apply_thread_friction() -> None:
        """Coulomb-style thread friction on the bolt: always resists spin, never drives it."""
        tf = torch.zeros(n, 1, 3, device=device)
        wz = bolt.data.root_ang_vel_w[:, 2]
        tf[:, 0, 2] = -THREAD_FRICTION * torch.tanh(wz / 0.05)
        bolt.set_external_force_and_torque(zero3, tf)

    def project_helix(cur_yaw: torch.Tensor) -> None:
        """Enforce the screw joint on the BOLT: z slaved to the accumulated turn at the pitch with
        hard stops, xy pinned, tilt removed. Spin about z stays FREE — only the key's contact drives it."""
        nonlocal bolt_turn
        bolt_turn = torch.clamp(bolt_turn, min=turn_min, max=turn_max)
        wz = bolt.data.root_ang_vel_w[:, 2].clone()
        wz = torch.where((bolt_turn >= turn_max) & (wz < 0), torch.zeros_like(wz), wz)  # seated stop
        wz = torch.where((bolt_turn <= turn_min) & (wz > 0), torch.zeros_like(wz), wz)  # exit stop
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = hole_xy
        st[:, 2] = z0 - PITCH * bolt_turn / (2 * math.pi)
        st[:, 3] = torch.cos(cur_yaw / 2)
        st[:, 6] = torch.sin(cur_yaw / 2)
        st[:, 9] = PITCH * wz / (2 * math.pi)  # v_z on the helix (wz < 0 -> descending)
        st[:, 12] = wz
        bolt.write_root_state_to_sim(st, ids)

    def project_key() -> None:
        """Rail the key: xy on the hole axis, tip riding the socket floor, tilt zero; yaw + spin FREE.
        Written at the LINK origin (not the 13-dim COM-state path): the L-key's COM is ~42 mm off the
        working-arm axis, so a zero COM-velocity write would sweep the arm laterally every step."""
        yaw = yaw_of(key.data.root_quat_w)
        wz = key.data.root_ang_vel_w[:, 2]
        pose = torch.zeros(n, 7, device=device)
        pose[:, 0:2] = hole_xy
        pose[:, 2] = bolt.data.root_pos_w[:, 2] + SOCKET_FLOOR_Z + KEY_TIP_HOVER
        pose[:, 3] = torch.cos(yaw / 2)
        pose[:, 6] = torch.sin(yaw / 2)
        key.write_root_link_pose_to_sim(pose, ids)
        vel = torch.zeros(n, 6, device=device)
        vel[:, 5] = wz
        key.write_root_link_velocity_to_sim(vel, ids)

    def depth() -> torch.Tensor:  # bolt tip depth below the plate top (m), per env
        return plat_z + plate_top - bolt.data.root_pos_w[:, 2]

    # Scale phase STEP budgets by (1/240)/dt so the sim TIME per phase stays constant.
    ts = (1.0 / 240.0) / DT
    show_end, drive_max, settle_steps = int(SHOW_END * ts), int(DRIVE_MAX * ts), int(SETTLE_STEPS * ts)
    log_every = max(1, int(300 * ts))

    prev_bolt_yaw = yaw_of(bolt.data.root_quat_w)
    key_turn = torch.zeros(n, device=device)
    prev_key_yaw = yaw_of(key.data.root_quat_w)
    handoff_depth = None
    phase, i, marker = "show", 0, 0
    while True:
        i += 1
        if phase == "show":
            tw = 0.0
            if i >= show_end:
                z0 = stage_parts()
                prev_bolt_yaw = yaw_of(bolt.data.root_quat_w)
                prev_key_yaw = yaw_of(key.data.root_quat_w)
                handoff_depth = depth().clone()
                phase, marker = "drive", i
        elif phase == "drive":
            tw = TWIST
            if float(depth().min()) >= depth_max - EXIT_MARGIN or i - marker >= drive_max:
                phase, marker = "settle", i
        else:
            tw = 0.0
            if i - marker >= settle_steps:
                break

        if phase != "show":
            key_twist(tw)
            apply_thread_friction()
        env.step(no_action, render=render)
        if phase != "show":
            cur = yaw_of(bolt.data.root_quat_w)
            bolt_turn = bolt_turn - _wrap(cur - prev_bolt_yaw)
            prev_bolt_yaw = cur
            kcur = yaw_of(key.data.root_quat_w)
            key_turn = key_turn - _wrap(kcur - prev_key_yaw)
            prev_key_yaw = kcur
            project_helix(cur)
            project_key()

        if i % log_every == 0:
            d = depth() * 1e3
            slip = torch.rad2deg(key_turn - bolt_turn)
            print(f"  step {i:5d} [{phase:6s}] | tip depth {d.mean():+6.2f}mm | bolt "
                  f"{torch.rad2deg(bolt_turn).mean():+7.0f}deg | key-bolt slip {slip.mean():+6.1f}deg", flush=True)

    seated = sc.seated()
    d = depth() * 1e3
    gain = (d - handoff_depth * 1e3) if handoff_depth is not None else d
    revs = bolt_turn / (2 * math.pi)
    turned = revs > 0.5
    mm_per_rev = float((gain[turned] / revs[turned]).median()) if turned.any() else float("nan")
    slip = torch.rad2deg(key_turn - bolt_turn)
    print(f"ALLEN-KEY [SCREW-JOINT] | seated {int(seated.all(dim=1).sum())}/{n} | key drove the bolt "
          f"{float(gain.mean()):+.1f}mm over {float(revs.mean()):.1f} revs = {mm_per_rev:.2f} mm/rev "
          f"(pitch {PITCH * 1e3:.1f}) | key-bolt slip {float(slip.mean()):+.1f}deg | tip depth mm: "
          f"min={d.min():+.1f} mean={d.mean():+.1f} max={d.max():+.1f} (seat>= {sc.cfg.seat_depth * 1e3:.0f})", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()

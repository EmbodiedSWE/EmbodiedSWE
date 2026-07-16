"""Smoke test for BulbAssemblyScene — visual or headless, no real robot.

One linear run with a NullRobot. It can't pick-and-place, so it cheats the staging by teleport:
  1. show   — leave the reset layout as is (bulbs lying on their side beside the fixed sockets);
  2. stage  — by direct sim write, lift each bulb up, stand it upright (cap/thread down — bulb.usd is
              already authored cap-down at identity) and centre it just above its socket's bore;
  3. screw  — press down + hold on the socket axis + a capped bang-bang twist, screwing each bulb into
              its socket until it seats;
  4. settle — let the bulb(s) come to rest, then read out seating.

Bodies are placed / driven straight through the scene handles (env.scene.bulbs / .sockets); the
NullRobot applies nothing. The drive is a simple open-loop press + xy-hold + capped twist (in the
GLOBAL frame). Watch with --livestream; --headless runs the same motion non-visually.

It also prints the bulb-origin height above the socket origin across the run — use the resting-on-top
vs fully-seated spread to calibrate `BulbAssemblySceneCfg.seat_z`.

python -m robobench.suites.assembly.smokes.bulb_assembly_smoke --livestream 2
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
livestream_on = args.livestream > 0

app = AppLauncher(args).app

from typing import TYPE_CHECKING

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402
from robobench.suites.assembly.smokes import close_and_exit  # noqa: E402

if TYPE_CHECKING:
    from robobench.suites.assembly.scenes import BulbAssemblyScene

# Assemble: press down (-z) + hold the bulb on the socket axis (xy PD) + a capped bang-bang twist about the
# socket axis. With the scene's small dt (1/240) the thread doesn't tunnel under gravity, so this simple
# open-loop drive threads the bulb cleanly down to its seat.
PRESS, TWIST, TARGET_WZ = -2.5, -0.15, -3.0
KP_XY, KD_XY = 30.0, 3.0
# Staging slack: the bulb origin IS its lowest point (measured bbox — the thread collider starts 4 mm up),
# so this just drops the bulb a few mm of clearance above the socket bore mouth, ready to engage.
BULB_FREE_END = 0.004
GAP = 0.001
# Phase boundaries, cumulative sim steps: show -> stage(teleport) -> screw -> settle.
SHOW_END, ASSEMBLE_END, END = 300, 8000, 8300  # more steps: the scene runs at dt=1/240


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()  # populate the registries (robots + suites) before resolving by name
    env = ENVS.get("assembly.bulb")().build(num_envs=args.num_envs, device=device)  # real gravity
    scene: BulbAssemblyScene = env.scene  # type: ignore[assignment]
    render = (not args.headless) or livestream_on
    n = env.num_envs
    no_action = torch.empty(0, device=device)  # NullRobot ignores it
    all_ids = torch.arange(n, device=device)
    stage_gap = scene.cfg.socket_opening_z + GAP - BULB_FREE_END  # bulb-origin height above the socket origin at staging

    env.reset()

    for i in range(1, END + 1):
        if i == SHOW_END:  # stage each bulb upright (cap-down) just above its socket's bore mouth
            for k, bulb in enumerate(scene.bulbs):
                socket_pos = scene.sockets[k].data.root_pos_w  # (n, 3), already world (incl. env origin)
                st = torch.zeros(n, 13, device=device)
                st[:, 0:3] = socket_pos
                st[:, 2] += stage_gap
                st[:, 3] = 1.0  # identity quat -> cap/thread down, screw axis up, aligned with the socket
                bulb.write_root_state_to_sim(st, all_ids)
        elif SHOW_END < i < ASSEMBLE_END:  # press + hold-on-axis + capped twist to screw each bulb in
            for k, bulb in enumerate(scene.bulbs):
                socket_xy = scene.sockets[k].data.root_pos_w[:, :2]
                pos, lin, w = bulb.data.root_pos_w, bulb.data.root_lin_vel_w, bulb.data.root_ang_vel_w
                f = torch.zeros(n, 3, device=device)
                f[:, :2] = KP_XY * (socket_xy - pos[:, :2]) - KD_XY * lin[:, :2]  # hold the bulb on the socket axis
                f[:, 2] = PRESS
                t = torch.zeros(n, 3, device=device)
                t[w[:, 2] > TARGET_WZ, 2] = TWIST  # capped bang-bang twist about z
                bulb.set_external_force_and_torque(f.unsqueeze(1), t.unsqueeze(1), is_global=True)

        env.step(no_action, render=render)  # NullRobot ignores the action

        if i % 150 == 0:  # progress: bulb0 height above its socket origin across ALL envs (min/mean/max, mm)
            h = (scene.bulbs[0].data.root_pos_w - scene.sockets[0].data.root_pos_w)[:, 2] * 1e3
            print(f"  step {i:4d} | height mm: min={h.min():+.1f} mean={h.mean():+.1f} max={h.max():+.1f}", flush=True)

    # Verdict across ALL envs: how many fully seated, and the spread of final bulb0 heights above the
    # socket origin (consistent threading -> all near the seated depth; flaky -> a wide spread / some high).
    seated = scene.seated()  # (num_envs, num_pairs)
    h0 = (scene.bulbs[0].data.root_pos_w - scene.sockets[0].data.root_pos_w)[:, 2] * 1e3
    print(f"BULB-ASSEMBLY | seated {int(seated.all(dim=1).sum())}/{n} envs | seat_z={scene.cfg.seat_z * 1e3:.0f}mm "
          f"| bulb0 height mm: min={h0.min():+.1f} mean={h0.mean():+.1f} max={h0.max():+.1f}", flush=True)
    close_and_exit(env, app)


if __name__ == "__main__":
    main()

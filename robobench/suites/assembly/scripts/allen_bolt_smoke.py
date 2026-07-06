"""Smoke test for AllenBoltAssemblyScene — visual or headless, no real robot.

Validates the core mechanic: the allen bolt threads INTO the platform's hole by real SDF thread
contact, driven by smooth press + twist applied directly to the bolt (the one regime SDF threads
survive) — no tool, no gripper.

One linear run with a NullRobot, staging by teleport:
  1. show   — leave the reset layout as is;
  2. stage  — stand each bolt upright, tip just above its platform's hole;
  3. screw  — press + twist each bolt down until it seats;
  4. settle — let the bolt(s) come to rest, then read out seating.

Verdict: descent-per-revolution over the screw phase must track the M16 pitch (~2.0 mm/rev).

python -m robobench.suites.assembly.scripts.allen_bolt_smoke --livestream 2
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

if TYPE_CHECKING:
    from robobench.suites.assembly.scenes import AllenBoltAssemblyScene

# Press down (-z) + twist clockwise seen from above (= negative world yaw) so the right-hand
# thread drives IN; TARGET_WZ caps the spin rate (twist only while spinning slower than this).
PRESS, TWIST, TARGET_WZ = -2.5, -0.15, -3.0
PITCH_MM = 2.0  # M16 coarse pitch, the expected descent per revolution
STOP_DEPTH = 0.0235  # stop twisting at this tip depth (m) — just before the head bottoms at 24.8 mm
STAGE_GAP = 0.0015  # staging gap (m) between bolt tip and plate top, just above the thread entry
# Cumulative step boundaries: show -> stage(teleport) -> screw -> settle (full seating ~13 turns).
SHOW_END, ASSEMBLE_END, END = 150, 5400, 5550


def yaw_of(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """World yaw (rad) of wxyz quaternions, shape (n,)."""
    w, x, y, z = quat_wxyz.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("assembly.allen_bolt")().build(num_envs=args.num_envs, device=device)
    scene: AllenBoltAssemblyScene = env.scene  # type: ignore[assignment]
    render = (not args.headless) or livestream_on
    n = env.num_envs
    no_action = torch.empty(0, device=device)  # NullRobot ignores it
    all_ids = torch.arange(n, device=device)

    env.reset()

    # Accumulated screw-phase turn of bolt0 (rad) and its depth at engagement, for the mm/rev verdict.
    turn = torch.zeros(n, device=device)
    prev_yaw = torch.zeros(n, device=device)
    depth0 = None

    for i in range(1, END + 1):
        if i == SHOW_END:  # stage each bolt upright, tip just above its platform's thread entry
            for k, bolt in enumerate(scene.bolts):
                plat_pos = scene.platforms[k].data.root_pos_w  # (n, 3), world (incl. env origin)
                st = torch.zeros(n, 13, device=device)
                st[:, 0:3] = plat_pos
                st[:, 2] += scene.cfg.plate_top + STAGE_GAP  # bolt origin is its TIP
                st[:, 3] = 1.0  # identity quat -> head up, thread down, aligned with the hole
                bolt.write_root_state_to_sim(st, all_ids)
                if k == 0:
                    print(f"  staged | platform0 w={plat_pos[0].tolist()} bolt0 w={st[0, 0:3].tolist()}", flush=True)
            prev_yaw = yaw_of(scene.bolts[0].data.root_quat_w)
        elif SHOW_END < i < ASSEMBLE_END:  # press + twist (capped spin) to screw each bolt down
            f = torch.zeros(n, 3, device=device)
            f[:, 2] = PRESS
            engaged = scene.engaged()  # (n, num_pairs)
            for k, bolt in enumerate(scene.bolts):
                t = torch.zeros(n, 3, device=device)
                # Gate the twist short of the head stop: driving against the seated head preloads
                # the threads through a second contact pair, the one regime SDF threads don't survive.
                drive = (bolt.data.root_ang_vel_w[:, 2] > TARGET_WZ) & (engaged[:, k] < STOP_DEPTH)
                t[drive, 2] = TWIST
                bolt.set_external_force_and_torque(f.unsqueeze(1), t.unsqueeze(1))
            yaw = yaw_of(scene.bolts[0].data.root_quat_w)
            d = yaw - prev_yaw
            turn += torch.abs(d - 2 * torch.pi * torch.round(d / (2 * torch.pi)))  # wrapped |delta|
            prev_yaw = yaw
            if depth0 is None:
                depth0 = scene.engaged()[:, 0].clone()
        elif i == ASSEMBLE_END:  # applied force/torque persists across steps — zero it for the settle
            zero = torch.zeros(n, 1, 3, device=device)
            for bolt in scene.bolts:
                bolt.set_external_force_and_torque(zero, zero)

        env.step(no_action, render=render)

        if i % 300 == 0:
            d = scene.engaged()[:, 0] * 1e3
            print(f"  step {i:4d} | tip depth mm: min={d.min():+.1f} mean={d.mean():+.1f} "
                  f"max={d.max():+.1f} | turns={float(turn.mean()) / (2 * torch.pi):.1f}", flush=True)

    seated = scene.seated()  # (num_envs, num_pairs)
    d = scene.engaged()[:, 0] * 1e3
    revs = turn / (2 * torch.pi)
    travel = d - (depth0 if depth0 is not None else 0.0) * 1e3
    turned = revs > 0.5  # only rate envs that actually screwed; a jammed bolt would blow up the ratio
    mm_per_rev = (travel[turned] / revs[turned]).median() if turned.any() else float("nan")
    print(f"ALLEN-BOLT | seated {int(seated.all(dim=1).sum())}/{n} envs "
          f"| seat_depth={scene.cfg.seat_depth * 1e3:.0f}mm "
          f"| tip depth mm: min={d.min():+.1f} mean={d.mean():+.1f} max={d.max():+.1f} "
          f"| bolt0 {float(revs.mean()):.1f} revs, {float(mm_per_rev):.2f} mm/rev median "
          f"(pitch {PITCH_MM:.1f}, {int(turned.sum())}/{n} turned)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()

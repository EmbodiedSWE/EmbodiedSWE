"""Smoke / firmness test for IkeaTableAssemblyScene — visual or headless, no real robot.

One linear run with a NullRobot. It can't pick-and-place, so it cheats the staging by teleport:
  1. show   — leave the reset layout as is (legs laid flat beside the off-centre table) so you can see it;
  2. stage  — restore the pre-screwable pose by direct sim write: centre the tabletop (undo the reset's
              `table_offset` slide) and stand each leg upright on its stud;
  3. screw  — press + twist each leg down its stud until it seats and the scene auto-welds it;
  4. settle — let the welded assembly come to rest;
  5. lift    — PD-lift the table off the bench by force on the table body;
  6. flip    — roll it over about world-x to check the welds hold.

Bodies are placed / driven straight through the scene handles (env.scene.legs / .table); the
NullRobot applies nothing. Watch with --livestream; --headless runs the same motion non-visually.

python -m robobench.suites.assembly.scripts.ikea_table_assembly_smoke --livestream 2
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

from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

if TYPE_CHECKING:
    from robobench.suites.assembly.scenes import IkeaTableAssemblyScene

# Assemble: press down + twist clockwise to thread a leg onto its stud.
PRESS, TWIST, TARGET_WZ = -2.5, -0.15, -3.0
# Firmness: lift the welded assembly off the bench, then roll it over.
G, LIFT = 9.81, 0.5        # gravity; how high to lift the assembly (m)
KP, KD = 60.0, 12.0        # PD on the table's lift height (force)
KLVL, KDLVL = 4.0, 0.6     # PD that keeps the table level while lifting (torque)
KROLL, ROLL_WZ = 1.5, 3.0  # P on the roll rate about world-x during the flip (target rad/s)
# Phase boundaries, cumulative sim steps: show -> stage(teleport) -> screw -> settle -> lift -> flip.
SHOW_END, ASSEMBLE_END, SETTLE_END, RAISE_END, END = 150, 650, 750, 1000, 1200


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Build from the suite's REGISTERED config (scene + NullRobot + the scene's own sim) — the same
    # binding `--env assembly.ikea_table` loads. This script then drives the SCENE's mechanic directly
    # (teleport legs onto studs, screw, check auto-weld) — what generic random actions can't exercise.
    robobench.discover()  # populate the registries (robots + suites) before resolving by name
    env = ENVS.get("assembly.ikea_table")().build(num_envs=args.num_envs, device=device)
    scene: IkeaTableAssemblyScene = env.scene  # type: ignore[assignment]
    render = (not args.headless) or livestream_on
    n, nlegs = env.num_envs, scene.cfg.num_legs
    no_action = torch.empty(0, device=device)  # NullRobot ignores it
    ez = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(n, 3)
    assembly_m = 1.0 + nlegs * scene.cfg.leg_mass  # table + welded legs: the mass the lift carries

    def leg_offsets() -> torch.Tensor:
        # Each leg's position in the table frame, (num_envs, nlegs, 3). Constant while a leg stays
        # rigidly welded; growing == the leg is slipping off (firmness failing).
        tp, tq = scene.table.data.root_pos_w, scene.table.data.root_quat_w
        return torch.stack([quat_apply_inverse(tq, leg.data.root_pos_w - tp) for leg in scene.legs], dim=1)

    env.reset()
    base_pos: torch.Tensor | None = None  # table rest pose, captured after the settle (hold target)
    ref_off = leg_offsets()  # leg offsets in the table frame; re-captured once welded (below)

    # Pre-screwable pose: tabletop centred on the workbench + each leg upright over its stud (env-local).
    # The reset slides the tabletop off-centre (table_offset) and lays the legs flat beside it; at
    # SHOW_END the test restores this centred, ready-to-screw config (the screw mechanic threads each leg
    # straight down its stud, so the legs must sit on the studs — which only line up with a centred table).
    all_ids = torch.arange(n, device=device)
    slab_top = scene.cfg.surface_z + scene.cfg.table_thickness
    wx, wy = scene.cfg.workbench_pos
    table_center = torch.tensor((wx, wy, slab_top), device=device)  # centre the tabletop (undo table_offset)
    stud_xyz = [torch.tensor((wx + sx, wy + sy, slab_top + scene.cfg.leg_start_z), device=device)
                for sx, sy in scene.cfg.slots[:nlegs]]

    for i in range(1, END + 1):
        leg_wrench = [(torch.zeros(n, 3, device=device), torch.zeros(n, 3, device=device)) for _ in range(nlegs)]
        table_f = torch.zeros(n, 3, device=device)
        table_t = torch.zeros(n, 3, device=device)

        if i <= SHOW_END:  # leave the laid-out parts as they are (visualise the reset layout) ...
            if i == SHOW_END:  # ... then restore the pre-screwable pose: centre the tabletop, then
                tbl = torch.zeros(n, 13, device=device)  # stand each leg upright on its (now centred) stud
                tbl[:, 0:3] = env.iscene.env_origins + table_center
                tbl[:, 3] = 1.0
                scene.table.write_root_state_to_sim(tbl, all_ids)
                for k, leg in enumerate(scene.legs):
                    st = torch.zeros(n, 13, device=device)
                    st[:, 0:3] = env.iscene.env_origins + stud_xyz[k]
                    st[:, 3] = 1.0
                    leg.write_root_state_to_sim(st, all_ids)
        elif i < ASSEMBLE_END:  # press + twist each leg onto its stud (the scene auto-welds on seat)
            for k, leg in enumerate(scene.legs):
                f, t = leg_wrench[k]
                f[:, 2] = PRESS
                spin = leg.data.root_ang_vel_w[:, 2] > TARGET_WZ
                t[spin, 2] = TWIST
        elif i == ASSEMBLE_END:
            ref_off = leg_offsets().clone()  # reference offsets once assembled + welded, before the lift
        elif i == SETTLE_END:  # capture the rest pose, then hold its xy through the lift + flip
            base_pos = scene.table.data.root_pos_w.clone()
        elif base_pos is not None:  # i > SETTLE_END: hold xy, ramp z up, then roll it over in place
            ramp = min(1.0, (i - SETTLE_END) / (RAISE_END - SETTLE_END))
            target = base_pos.clone()
            target[:, 2] = base_pos[:, 2] + LIFT * ramp
            pos = scene.table.data.root_pos_w
            vel = scene.table.data.root_lin_vel_w
            w = scene.table.data.root_ang_vel_w
            table_f = KP * (target - pos) - KD * vel  # PD-hold position so it can't drift / fly off
            table_f[:, 2] += assembly_m * G           # + feed-forward gravity on the z axis
            if i <= RAISE_END:  # lift straight up, kept level
                up = quat_apply(scene.table.data.root_quat_w, ez)
                table_t = KLVL * torch.cross(up, ez, dim=-1) - KDLVL * w
            else:  # roll about world-x to turn it over; damp the other two axes so it rolls cleanly
                table_t = -KDLVL * w
                table_t[:, 0] = KROLL * (ROLL_WZ - w[:, 0])

        for k, leg in enumerate(scene.legs):
            f, t = leg_wrench[k]
            leg.set_external_force_and_torque(f.unsqueeze(1), t.unsqueeze(1))
        scene.table.set_external_force_and_torque(table_f.unsqueeze(1), table_t.unsqueeze(1))

        env.step(no_action, render=render)  # NullRobot ignores the action; scene.post_step() auto-welds

    # Verdict (env 0): every leg should still be welded + seated, and barely slipped in the table
    # frame, after the flip. (print is swallowed in --headless — watch with --livestream.)
    slip_mm = ((leg_offsets() - ref_off).norm(dim=-1)[0] * 1e3).tolist()
    print(f"FIRMNESS env0 | welded={scene.welded[0].int().tolist()} | seated={scene.seated()[0].int().tolist()} "
          f"| leg-slip(mm)={[round(v, 1) for v in slip_mm]}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()

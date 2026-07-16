"""Physics smoke test for PcGpuAssemblyScene — a loose RTX 2060 is installed into the PCIe slot
through the case's REAR I/O CUTOUT, the way a real build goes: place the card inside the case
(bracket pulled forward of the rear panel), slide it rearward so the bracket/ports pass through
the expansion-slot opening, then press it down into the slot.

The card <-> fixture contact is LIVE (slot channel + the rear-panel cutout frame — a straight
vertical drop is physically blocked by the panel above the opening) and the card is driven
purely by forces: a PD "hand" (with a weight feedforward — gravity stays ON, so the final
hold-check is a real retention test) lifts the lying card off the table, rights it, carries it
over the case rim (the walls stand 195 mm above the board face), descends INSIDE the case at a
28 mm forward offset, slides rearward through the cutout at 16.5 mm height (tab 1 mm above the
channel walls, bracket 1.8 mm under the opening top), and presses straight down. The invisible
channel (a 1.2 mm/side funnel mouth narrowing to a 0.15 mm/side grip on the 4 mm PCB tab) guides
the last 5 mm. Then the hand lets go: the card must hold its seat on its own — the snug grip
caps its gravity roll (COM ~16 mm on the fan side of the PCB plane) at ~2 deg.

Phases: show -> lift -> cross -> drop (inside, at the forward offset) -> align -> slide (rear-
ward through the cutout) -> press (with logged re-tries) -> release -> settle. Verdict: seated
count, tab depth below the slot mouth vs the 5 mm stroke, and the residual errors after release.

python -m robobench.suites.assembly.smokes.pc_gpu_smoke --livestream 2
python -m robobench.suites.assembly.smokes.pc_gpu_smoke \
    --headless --enable_cameras --video robobench/suites/assembly/videos/pc_gpu.mp4
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
from isaaclab.utils.math import axis_angle_from_quat, quat_apply_inverse  # noqa: E402
from robobench.core import EnvCfg  # noqa: E402
from robobench.suites.assembly.smokes import close_and_exit  # noqa: E402

if TYPE_CHECKING:
    from robobench.suites.assembly.scenes import PcGpuAssemblyScene

DT = 1.0 / 240.0        # sim timestep
GRAV = 9.81
# Flight geometry (m, relative to the board face unless said otherwise). The card origin is its
# PCB-tab bottom centre; upright, the card body rises 123 mm above it.
CROSS_Z = 0.240         # origin height while crossing the case rim (bottom clears the 195 mm walls)
SLIDE_OFF = 0.028       # forward (-x) offset while placing the card inside (bracket clear of panel)
SLIDE_Z = 0.0165        # origin height for the placement + rearward slide: tab 1 mm above the
                        # channel wall tops (15.5), bracket 1.8 mm under the cutout top (18.3 max)
PRESS_TGT = -0.0005     # press z target below the seated origin (sustained push until bottomed)
PRESS_DONE = 0.0048     # tab depth below the slot mouth to call the press finished (stroke 5 mm)
MAX_RETRIES = 2         # press re-tries (raise back to SLIDE_Z — NOT higher: the bracket sits in
                        # the rear cutout and would jam on the panel above it — then press again)
# PD "hand" gains for the 1 kg card (its rigid props carry 2.0 linear/angular damping):
KP_XY, KD_XY = 600.0, 50.0    # N/m, N s/m — holds the origin on the carry/slot axis
KP_Z, KD_Z = 200.0, 30.0      # N/m, N s/m — vertical carry/press servo
KP_ROT, KD_ROT = 4.0, 0.4     # N m/rad, N m s/rad — rights the card to the seated orientation
F_XY_CAP, F_Z_CAP = 30.0, 20.0  # N — force authority around the weight feedforward
T_CAP = 3.0                   # N m — torque authority
# Phase step budgets at dt=1/240 (rescaled at run time so sim TIME per phase is constant).
SHOW_END, SETTLE_STEPS = 150, 300
LIFT_STEPS, CROSS_STEPS, DROP_STEPS, ALIGN_STEPS = 600, 420, 420, 240
SLIDE_STEPS = 360
PRESS_STEPS, PRESS_MAX = 480, 1440
ALIGN_XY_TOL = 0.0008   # m — position gate at the placement point and at the end of the slide
ALIGN_ROT_TOL = math.radians(1.0)


def smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def main() -> None:
    device = getattr(args, "device", None) or ("cuda:0" if torch.cuda.is_available() else "cpu")
    robobench.discover()

    env = EnvCfg(scene="pc_gpu", robot="null", sim_overrides={"dt": DT}).build(
        num_envs=args.num_envs, device=device
    )
    sc: PcGpuAssemblyScene = env.scene  # type: ignore[assignment]
    n = env.num_envs
    ids = torch.arange(n, device=device)
    no_action = torch.empty(n, 0, device=device)
    card, case = sc.card, sc.case
    zero3 = torch.zeros(n, 1, 3, device=device)
    render = (not args.headless) or livestream_on
    weight = sc.cfg.card_mass * GRAV

    cam = writer = None
    if args.video:
        import imageio.v2 as imageio
        from isaaclab.sensors import Camera, CameraCfg
        cam = Camera(CameraCfg(prim_path="/World/cam", update_period=0.0, height=720, width=1280, data_types=["rgb"],
                               spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.01, 100.0))))
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args.video, fps=30)

    env.sim.reset()  # re-parse physics so the camera (if any) is picked up
    env.reset()

    case_pos = case.data.root_pos_w.clone()  # (n, 3): origin ON the board face, at its centre
    board_z = case_pos[:, 2].clone()
    seat_w = case_pos + torch.tensor(sc.cfg.seat_pos, device=device)  # (n, 3) seated origin, world
    place_w = seat_w.clone()  # placement point INSIDE the case: bracket forward of the rear panel
    place_w[:, 0] -= SLIDE_OFF
    if cam is not None:
        p0 = case_pos[0]
        # Close-in 3/4 view onto the card's BACKPLATE, floating just inside the case opening:
        # only this side shows the gold edge connector over the slot (the cooler shroud hides it
        # from the fan side), an outside camera low enough to face the vertical card is blocked
        # by the 195 mm wall rim, and from high above the card foreshortens to a sliver.
        eye = torch.tensor([[p0[0] + 0.13, p0[1] - 0.26, p0[2] + 0.40]], dtype=torch.float32, device=device)
        tgt = torch.tensor([[p0[0] + 0.03, p0[1] + 0.02, p0[2] + 0.075]], dtype=torch.float32, device=device)
        cam.set_world_poses_from_view(eye, tgt)
    print(env.describe(), flush=True)

    def rot_err() -> torch.Tensor:
        """Axis-angle error (n, 3) from the card's current orientation to the seated one (world
        identity — the case spawns unrotated), expressed in the WORLD frame."""
        return axis_angle_from_quat(card.data.root_quat_w)

    def card_wrench(f: torch.Tensor, t: torch.Tensor) -> None:
        """Apply a WORLD wrench to the card in its CURRENT link frame."""
        qc = card.data.root_link_quat_w
        card.set_external_force_and_torque(quat_apply_inverse(qc, f).unsqueeze(1),
                                           quat_apply_inverse(qc, t).unsqueeze(1))

    def hand(tgt_pos: torch.Tensor, right: bool = True) -> None:
        """One force-only 'hand' update: clamped PD toward a world position target around the
        weight feedforward, plus (optionally) a PD righting the card to the seated orientation."""
        pos = card.data.root_link_pos_w
        vel = card.data.root_link_lin_vel_w
        f = -KP_XY * (pos - tgt_pos) - KD_XY * vel
        f[:, 2] = -KP_Z * (pos[:, 2] - tgt_pos[:, 2]) - KD_Z * vel[:, 2]
        f[:, 0:2] = f[:, 0:2].clamp(-F_XY_CAP, F_XY_CAP)
        f[:, 2] = f[:, 2].clamp(-F_Z_CAP, F_Z_CAP) + weight
        t = torch.zeros_like(f)
        if right:
            t = (-KP_ROT * rot_err() - KD_ROT * card.data.root_ang_vel_w).clamp(-T_CAP, T_CAP)
        card_wrench(f, t)

    def depth() -> torch.Tensor:  # tab depth below the slot mouth (m), per env
        return sc.engaged()

    def xy_err() -> torch.Tensor:
        return (card.data.root_link_pos_w[:, 0:2] - seat_w[:, 0:2]).norm(dim=-1)

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
    show_end, settle_steps = int(SHOW_END * ts), int(SETTLE_STEPS * ts)
    lift_steps, cross_steps = int(LIFT_STEPS * ts), int(CROSS_STEPS * ts)
    drop_steps, align_steps = int(DROP_STEPS * ts), int(ALIGN_STEPS * ts)
    slide_steps = int(SLIDE_STEPS * ts)
    press_steps, press_max = int(PRESS_STEPS * ts), int(PRESS_MAX * ts)
    log_every = max(1, int(300 * ts))

    lift_from = torch.zeros(n, 3, device=device)
    cross_from = torch.zeros(n, 2, device=device)
    drop_from = torch.zeros(n, device=device)
    press_from = torch.zeros(n, device=device)
    retries = 0
    phase, i, marker = "show", 0, 0
    while True:
        i += 1
        if phase == "show":
            if i >= show_end:
                lift_from = card.data.root_link_pos_w.clone()
                phase, marker = "lift", i
        elif phase == "lift":  # rise off the table to the rim-crossing height while righting
            s = smoothstep((i - marker) / lift_steps)
            tgt = lift_from.clone()
            tgt[:, 2] = lift_from[:, 2] + s * (board_z + CROSS_Z - lift_from[:, 2])
            hand(tgt)
            up_here = (board_z + CROSS_Z - card.data.root_link_pos_w[:, 2]).abs() < 0.005
            upright = rot_err().norm(dim=-1) < math.radians(5.0)
            if bool((up_here & upright).all()) or i - marker >= 2 * lift_steps:
                cross_from = card.data.root_link_pos_w[:, 0:2].clone()
                phase, marker = "cross", i
        elif phase == "cross":  # glide over the rim to the PLACEMENT point, at crossing height
            s = smoothstep((i - marker) / cross_steps)
            tgt = torch.zeros(n, 3, device=device)
            tgt[:, 0:2] = cross_from + s * (place_w[:, 0:2] - cross_from)
            tgt[:, 2] = board_z + CROSS_Z
            hand(tgt)
            arrived = (card.data.root_link_pos_w[:, 0:2] - place_w[:, 0:2]).norm(dim=-1) < 0.003
            if (i - marker >= cross_steps and bool(arrived.all())) or i - marker >= 2 * cross_steps:
                drop_from = card.data.root_link_pos_w[:, 2].clone()
                phase, marker = "drop", i
        elif phase == "drop":  # place the card INSIDE the case, bracket forward of the rear panel
            s = smoothstep((i - marker) / drop_steps)
            tgt = place_w.clone()
            tgt[:, 2] = drop_from + s * (board_z + SLIDE_Z - drop_from)
            hand(tgt)
            if i - marker >= drop_steps:
                phase, marker = "align", i
        elif phase == "align":  # settle at the placement point before the rearward slide
            tgt = place_w.clone()
            tgt[:, 2] = board_z + SLIDE_Z
            hand(tgt)
            perr = (card.data.root_link_pos_w[:, 0:2] - place_w[:, 0:2]).norm(dim=-1)
            still = card.data.root_link_lin_vel_w.norm(dim=-1) < 0.01
            ok = (perr < ALIGN_XY_TOL) & (rot_err().norm(dim=-1) < ALIGN_ROT_TOL) & still
            if bool(ok.all()) or i - marker >= 2 * align_steps:
                phase, marker = "slide", i
        elif phase == "slide":  # rearward: the bracket/ports pass through the I/O panel cutout
            s = smoothstep((i - marker) / slide_steps)
            tgt = place_w.clone()
            tgt[:, 0] = place_w[:, 0] + s * SLIDE_OFF
            tgt[:, 2] = board_z + SLIDE_Z
            hand(tgt)
            still = card.data.root_link_lin_vel_w.norm(dim=-1) < 0.01
            if (i - marker >= slide_steps and bool(((xy_err() < ALIGN_XY_TOL) & still).all())) \
                    or i - marker >= 2 * slide_steps:
                press_from = card.data.root_link_pos_w[:, 2].clone()
                phase, marker = "press", i
        elif phase == "press":  # straight down; the channel funnel guides the last 5 mm
            s = smoothstep((i - marker) / press_steps)
            tgt = seat_w.clone()
            tgt[:, 2] = press_from + s * (seat_w[:, 2] + PRESS_TGT - press_from)
            hand(tgt)
            if bool((depth() >= PRESS_DONE).all()):
                card.set_external_force_and_torque(zero3, zero3)  # applied wrench persists — zero it
                phase, marker = "settle", i
            elif i - marker >= press_max:
                if retries < MAX_RETRIES:
                    retries += 1
                    print(f"  WARN: press stalled at {float(depth().mean() * 1e3):+.2f} mm — "
                          f"retry {retries}/{MAX_RETRIES}", flush=True)
                    phase, marker = "reseat", i
                else:
                    print("  WARN: press exhausted its retries — releasing as-is", flush=True)
                    card.set_external_force_and_torque(zero3, zero3)
                    phase, marker = "settle", i
        elif phase == "reseat":  # back up to slide height ONLY (the bracket sits in the cutout —
            tgt = seat_w.clone()  # rising higher would jam it on the panel above the opening)
            tgt[:, 2] = board_z + SLIDE_Z
            hand(tgt)
            if i - marker >= align_steps:
                press_from = card.data.root_link_pos_w[:, 2].clone()
                phase, marker = "press", i
        else:  # settle: hands off — the seated card must hold on its own
            if i - marker >= settle_steps:
                break

        step(i)
        if i % log_every == 0:
            e = rot_err()
            print(f"  step {i:5d} [{phase:6s}] | depth {float(depth().mean() * 1e3):+6.2f}mm | "
                  f"xy err {float(xy_err().mean() * 1e3):5.2f}mm | "
                  f"rot err {float(torch.rad2deg(e.norm(dim=-1)).mean()):5.2f}deg", flush=True)

    if writer is not None:
        writer.close()
        print("MP4:", args.video, flush=True)

    seated = sc.seated()  # (n,)
    e = rot_err()
    print(f"PC-GPU | seated {int(seated.sum())}/{n} envs | tab depth "
          f"{float(depth().mean() * 1e3):+.2f} mm (stroke 5.0, seat >= {sc.cfg.seat_depth * 1e3:.0f}) | "
          f"residual xy {float(xy_err().mean() * 1e3):.2f} mm, "
          f"rot {float(torch.rad2deg(e.norm(dim=-1)).mean()):.2f} deg | retries {retries}", flush=True)
    close_and_exit(env, app)


if __name__ == "__main__":
    main()

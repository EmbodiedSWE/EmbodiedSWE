"""Debug smoke for the SO101 elbow-HORN fastening — starts FULLY ELBOW-ASSEMBLED.

The full smoke (so101_smoke.py, phases A-J) proves the tab-screw story; this one skips it and
exercises ONLY the new mechanic — attaching the distal half's forearm fork to the seated elbow
servo and locking it with the M3 horn screw. The setup hand-builds the post-[G] state directly:
arm clamped at the lying working pose (horn side up), servo welded in its pocket, all four M2
tab screws welded at their seats. Once verified, these phases merge into the full smoke.

Drives ALL EIGHT horn-line M3 screws — the real SO101 fastening, four on each side of the
fork: the NEAR four go down the peripheral access channels in the fork's outer skin, seat
their heads on the fork's inner plate and thread into the servo horn's metal holes; the FAR
four go through the far plate into the servo's case-back holes. The far side faces down in
the lying pose, so the smoke re-fixtures to the flipped pose between the two rounds (the same
re-clamp stand-in as the far M2 tabs).

Phases:
  S) SETUP   — teleport + clamp the arm; seat + weld the motor and the four tab screws; settle.
  B) ATTACH  — a hand (kinematic, the same stand-in as the clamp) SLIDES the fork onto the
     servo mouth-first along its long axis, from the front where the forearm extends into open
     space (the clevis path — in the kit the closed fork print-flexes over the bosses; here the
     swept locating features carry trimmed collision instead), then presses it down onto the
     horn. The hand keeps HOLDING the fork at the seat until the first screw bites (a real
     hand steadies the part while screwing).
  Then, for each of the eight horn holes (near 0-3 lying, far 4-7 flipped; the flip re-grab is
  WELD-SAFE — welds off, arm teleported, the welded chain re-seated, welds back on):
  D) PICKUP  — the drill picks a loose M3 up with its magnetic bit.
  E) CARRY   — carries it over that hole and lowers it in, tip-first.
  F) DRIVE   — trigger -> gate -> the M3 drives to its seat -> weld (the lower_arm<->motor
     weld comes on with the first); the hand releases the fork.
  G) RETREAT — the drill lifts off gently; parks aside after the last screw.
  I) STRESS  — back in the lying pose: knock a driven screw, wrench the distal; nothing may
     come apart.
  J) FINALE  — the clamp lifts the whole robot (both halves now), shakes it, rotates 180 deg,
     sets it down and RELEASES it: the robot rests assembled on the workbench.

  python -m robobench.suites.assembly.scripts.so101_horn_smoke --livestream 2
  python -m robobench.suites.assembly.scripts.so101_horn_smoke --headless
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
from robobench.core import ENVS  # noqa: E402

if TYPE_CHECKING:
    from robobench.suites.assembly.scenes.so101_assembly import SO101AssemblyScene

JOINT2_DEG = -90.0  # shoulder_lift pose for the fasten phases (the arm swings flat)
FINGER = 0.08       # squeeze torque at the drill hinge (N*m)
BIT_STOP = 0.001    # how far above the seat the bit stops driving (m)
RETREAT_S = 1.5     # how long the drill lifts off after fastening (s)
HOVER = 0.10        # bit-tip standoff above the seat before the drill descends (m)
PICK_STOP = 0.002   # bit-tip standoff above the lying screw where the magnet takes it (m)
DRIVE_START = 0.002  # carried-screw head height above the seat where the driving begins (m)
STATE_NAMES = ("free", "driving", "fastened")

HOLD = True
HOLD_HEIGHT = 0.08                              # base clamp height, lying pose (m)
HOLD_HEIGHT_FAR = 0.12                          # flipped pose: high enough to clear the table top
HOLD_QUAT = (0.7071068, 0.0, 0.7071068, 0.0)    # Ry(90): arm on its side, horn side facing up
FLIP_QUAT = (0.0, 1.0, 0.0, 0.0)                # Rx(180) pre-rotation: far side facing up
NUM_NEAR_M3 = 4  # horn holes 0-3 = near (horn) side; 4-7 = far (case-back) side
DRILL_QUAT = (0.7071068, -0.7071068, 0.0, 0.0)  # drill working orientation: bit pointing down
DRILL_PARK = (0.70, -0.2, 0.145)                # where the drill parks after the screw is in:
# on the workbench top (east end, past the screw rows) and 0.73 m from the base, clear of the
# finale's 180-deg sweep of the assembled robot (~0.55 m reach). The old (0.45, -0.3) spot
# left the drill body overhanging the table's -y edge.
# The fork ATTACH is a SLIDE-ON (the clevis path): the fork approaches from the servo's front
# — where the forearm naturally extends, open space — mouth-first along the servo's long axis,
# lifted slightly off the engagement bosses, then presses down onto the horn. In the kit this
# is the print-flex snap; rigid bodies get the swept features' collision trimmed instead.
SLIDE_IN = 0.028     # slide travel along the fork's mouth axis (m): mouth starts past the nose
SLIDE_S = 1.2        # how long the slide takes (s)
FORK_LIFT = 0.0010   # lift off the seat during the slide, along the out-axis (m)
PRESS_S = 0.75       # how long the final press takes (s)


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("assembly.so101")().build(num_envs=args.num_envs, device=device)
    scene: SO101AssemblyScene = env.scene  # type: ignore[assignment]
    cfg = scene.cfg
    render = (not args.headless) or livestream_on
    n = env.num_envs
    dev = env.device
    no_action = torch.empty(0, device=dev)
    env.reset()

    sps = round(1.0 / env.dt)
    ne = cfg.num_elbow_screws
    n_m3 = cfg.num_horn_screws  # M3 screws are scene.screws[ne + k]; near holes are horn 0..3
    print(f"[horn] dt={env.dt:.5f} ({sps} steps/s) | {ne} tab screws pre-welded, "
          f"{n_m3} horn screws to drive", flush=True)

    # env-frame anchor ON the working surface: every choreography height (clamp, park, hover)
    # is surface-relative, so the whole smoke rides the scene's workbench preset
    origin = env.iscene.env_origins + torch.tensor((0.0, 0.0, cfg.surface_z), device=dev)
    q_hold = torch.tensor(HOLD_QUAT, device=dev)
    q_far = quat_mul(torch.tensor(FLIP_QUAT, device=dev).unsqueeze(0), q_hold.unsqueeze(0))[0]
    q_drill = torch.tensor(DRILL_QUAT, device=dev).expand(n, 4)
    finger = torch.zeros(n, 1, device=dev)
    standoff = torch.full((n,), HOVER, device=dev)
    i_j2 = scene.proximal.find_joints("shoulder_lift")[0][0]
    j2_target = torch.zeros(n, scene.proximal.num_joints, device=dev)
    j2_target[:, i_j2] = math.radians(JOINT2_DEG)
    distal_target = torch.zeros(n, scene.distal.num_joints, device=dev)

    hold_arm = True          # the base clamp (the smoke's vise) — on for the whole run until J
    hold_fork = False        # the hand holding the fork at its seat (offset by fork_off)
    fork_off = {"slide": 0.0, "lift": 0.0}  # hand offset in the SEAT frame: slide = out along
    # the mouth axis (toward the servo's front), lift = off the seat along the out-axis
    track_drill = False
    pick: dict = {"anchor": None}
    arm_target_pos = (origin + torch.tensor((0.0, 0.0, HOLD_HEIGHT), device=dev)).contiguous()
    arm_target_quat = q_hold.expand(n, 4).clone()  # clone: at n=1, expand().contiguous()
    # returns an ALIAS of q_hold and the clamp-target writes would corrupt it (see so101_smoke)
    stepno = 0

    import logging as _logging
    _logging.getLogger("isaaclab").setLevel(_logging.ERROR)

    cur_hole = 0  # the near horn hole being addressed (0..3): drill track + readout follow it

    def horn_seat_axis(hole: int | None = None):
        s, a = scene.elbow_horn_screw_seats_w()
        h = cur_hole if hole is None else hole
        return s[:, h], a[:, h]

    def la_err() -> torch.Tensor:
        exp_p, _ = scene.lower_arm_seat_w()
        return (scene.distal.data.root_pos_w - exp_p).norm(dim=-1)

    def motor_err() -> torch.Tensor:
        ap, _ = scene.upper_arm_pose()
        return (scene.motor.data.root_pos_w - ap).norm(dim=-1)

    def horn_screw_errs() -> torch.Tensor:
        """Max distance of any fastened M3 from its own hole's seat (m)."""
        seats, _ = scene.elbow_horn_screw_seats_w()
        arange = torch.arange(n, device=dev)
        errs = []
        for k in range(n_m3):
            h = (scene.fastened[:, ne + k] - cfg.num_elbow_holes).clamp_min(0)
            err = (scene.screws[ne + k].data.root_pos_w - seats[arange, h]).norm(dim=-1)
            errs.append(torch.where(scene.fastened[:, ne + k] >= 0, err, torch.zeros_like(err)))
        return torch.stack(errs, dim=1).max(dim=1).values

    def step(k: int, phase: str) -> None:
        nonlocal stepno
        for _ in range(k):
            if hold_arm:  # the vise
                st = torch.zeros(n, 13, device=dev)
                st[:, 0:3] = arm_target_pos
                st[:, 3:7] = arm_target_quat
                scene.proximal.write_root_state_to_sim(st, None)
            if hold_fork:  # the hand: the fork rides its live seat, offset by fork_off
                seat_p, seat_q = scene.lower_arm_seat_w()
                off = torch.tensor((-fork_off["slide"], 0.0, -fork_off["lift"]), device=dev)
                st = torch.zeros(n, 13, device=dev)
                st[:, 0:3] = seat_p + quat_apply(seat_q, off.expand(n, 3))
                st[:, 3:7] = seat_q
                scene.distal.write_root_state_to_sim(st, None)
                zj = torch.zeros(n, scene.distal.num_joints, device=dev)
                scene.distal.write_joint_state_to_sim(zj, zj)
            if pick["anchor"] is not None:  # pickup: the bit descends onto the lying screw
                st = torch.zeros(n, 13, device=dev)
                tip_target = pick["anchor"] + torch.tensor(
                    (0.0, 0.0, 1.0), device=dev) * standoff.unsqueeze(-1)
                st[:, 0:3] = tip_target - quat_apply(q_drill, torch.tensor(
                    cfg.bit_tip, device=dev).expand(n, 3))
                st[:, 3:7] = q_drill
                scene.drill.write_root_state_to_sim(st, None)
            if track_drill:
                s, a = horn_seat_axis()
                st = torch.zeros(n, 13, device=dev)
                tip_target = s + standoff.unsqueeze(-1) * a
                st[:, 0:3] = tip_target - quat_apply(q_drill, torch.tensor(
                    cfg.bit_tip, device=dev).expand(n, 3))
                st[:, 3:7] = q_drill
                scene.drill.write_root_state_to_sim(st, None)
            scene.drill.set_joint_effort_target(finger, joint_ids=[scene.i_trig])
            scene.proximal.set_joint_position_target(j2_target)
            scene.distal.set_joint_position_target(distal_target)
            env.step(no_action, render=render)
            stepno += 1
            if stepno % (sps // 4) == 0:
                s, a = horn_seat_axis()
                t_a = ((scene.screws[ne + cur_hole].data.root_pos_w - s) * a).sum(-1)[0].item()
                bv = scene.drill.data.joint_vel[0, scene.i_bit].item()
                print(f"  step {stepno:5d} [{phase:9s}] {STATE_NAMES[scene.state[0]]:8s} "
                      f"M3 hole{cur_hole} depth-above-seat {t_a * 1000:+7.2f} mm | fork err "
                      f"{la_err()[0] * 1000:5.2f} mm | motor err {motor_err()[0] * 1000:5.2f} mm "
                      f"| bit {bv:+6.2f} rad/s | fastened "
                      f"{int((scene.fastened[0] >= 0).sum())}/{cfg.num_screws}", flush=True)

    def grab(base_quat: torch.Tensor, height: float) -> None:
        """Teleport the arm to a working pose and clamp the base there — WELD-SAFE: enabled
        welds must never see the teleport as an error (they would yank the welded chain across
        the workspace), so the welds are dropped, the arm is teleported and stepped under the
        clamp (the link FK is stale until stepped), the welds re-enabled and the whole chain
        re-seated by the scene's own reconciler against the fresh pose, then a full-second
        settle rings the flipped arm down."""
        nonlocal hold_arm
        fastened = scene.fastened.clone()
        for i in range(n):
            for s in range(cfg.num_screws):
                if int(fastened[i, s]) >= 0:
                    scene._set_weld(i, s, 0, False)
        st = torch.zeros(n, 13, device=dev)
        st[:, 0:3] = origin + torch.tensor((0.0, 0.0, height), device=dev)
        st[:, 3:7] = base_quat
        scene.proximal.write_root_state_to_sim(st, None)
        jq = torch.zeros(n, scene.proximal.num_joints, device=dev)
        jq[:, i_j2] = math.radians(JOINT2_DEG)
        scene.proximal.write_joint_state_to_sim(jq, torch.zeros_like(jq))
        arm_target_pos[:] = origin + torch.tensor((0.0, 0.0, height), device=dev)
        arm_target_quat[:] = base_quat
        hold_arm = HOLD
        step(2, "re-grab")  # FK refresh under the clamp
        ap, aq = scene.upper_arm_pose()
        for i in range(n):
            for s in range(cfg.num_screws):
                h = int(fastened[i, s])
                if h >= 0:
                    scene._set_weld(i, s, h, True)
        scene._reconcile_fastened(arm_pose=(ap, aq), motor_pose=(ap, aq))
        step(sps, "re-grab")  # ring-down settle at the new facing

    # ============ S) SETUP: clamp the arm, weld the motor + all four tab screws ================
    print("[S] hand-building the post-[G] state: arm clamped flat, servo + 4 tab screws welded",
          flush=True)
    st = torch.zeros(n, 13, device=dev)
    st[:, 0:3] = arm_target_pos
    st[:, 3:7] = arm_target_quat
    scene.proximal.write_root_state_to_sim(st, None)
    jq = torch.zeros(n, scene.proximal.num_joints, device=dev)
    jq[:, i_j2] = math.radians(JOINT2_DEG)
    scene.proximal.write_joint_state_to_sim(jq, torch.zeros_like(jq))
    step(2, "S clamp")  # FK refresh under the clamp
    ap, aq = scene.upper_arm_pose()
    st = torch.zeros(n, 13, device=dev)  # the servo seats exactly on the upper_arm link frame
    st[:, 0:3] = ap
    st[:, 3:7] = aq
    scene.motor.write_root_state_to_sim(st, None)
    seats, _ = scene.elbow_screw_seats_w()
    for s in range(ne):
        st = torch.zeros(n, 13, device=dev)
        st[:, 0:3] = seats[:, s]
        st[:, 3:7] = quat_mul(aq, scene._elbow_screw_seat_quats[s].expand(n, 4))
        scene.screws[s].write_root_state_to_sim(st, None)
    for i in range(n):
        for s in range(ne):
            scene._set_weld(i, s, s, True)
    step(sps // 2, "S settle")
    print(f"[S] assembled: motor err {motor_err().max() * 1000:.2f} mm", flush=True)

    # ============ B) ATTACH: slide the fork on along the servo axis, then press it down ========
    print("[B] the hand slides the forearm fork onto the servo, mouth-first along its axis "
          "(the clevis path), then presses it down onto the horn", flush=True)
    hold_fork = True
    fork_off.update(slide=SLIDE_IN, lift=FORK_LIFT)
    step(sps // 2, "B hover")
    for j in range(int(SLIDE_S * sps)):
        fork_off["slide"] = SLIDE_IN * (1.0 - (j + 1) / (SLIDE_S * sps))
        step(1, "B slide")
    for j in range(int(PRESS_S * sps)):
        fork_off["lift"] = FORK_LIFT * (1.0 - (j + 1) / (PRESS_S * sps))
        step(1, "B press")
    step(sps // 4, "B seated")
    print(f"[B] fork slid on and pressed onto the horn: fork err {la_err().max() * 1000:.2f} mm "
          f"(hand keeps holding until the screw bites)", flush=True)

    # ============ D-G) per near hole: pick an M3 up, carry it in, drive, retreat ===============
    picked = torch.zeros(n_m3, dtype=torch.bool)
    fasten_steps: list[int] = []
    for k in range(n_m3):
        if k == NUM_NEAR_M3:  # the far holes face down in the lying pose — re-fixture flipped
            print("[re-fixture] flip: far side up (weld-safe re-grab)", flush=True)
            grab(q_far, HOLD_HEIGHT_FAR)
        cur_hole = k
        i_s = ne + k
        print(f"[D] hole {k}: the drill picks M3 screw {k} up with its magnetic bit", flush=True)
        pick["anchor"] = scene.screws[i_s].data.root_pos_w.clone()
        track_drill = False
        standoff[:] = HOVER
        step(sps // 8, "D hover")
        for j in range(sps):
            standoff[:] = HOVER + (PICK_STOP - HOVER) * (j + 1) / sps
            step(1, "D pickup")
            if bool(scene.attached[:, i_s].all()):
                break
        picked[k] = bool(scene.attached[:, i_s].all())
        print(f"[D] M3 {k} {'is on the bit' if picked[k] else 'MISSED the pickup'}", flush=True)
        z_now = float(standoff[0])
        for j in range(sps // 2):
            standoff[:] = z_now + (HOVER - z_now) * (j + 1) / (sps // 2)
            step(1, "D lift")
        pick["anchor"] = None

        print(f"[E] the drill carries it over hole {k}'s access channel and lowers it in",
              flush=True)
        standoff[:] = HOVER
        track_drill = True
        step(sps // 8, "E hover")
        for j in range(sps):
            standoff[:] = HOVER + (DRIVE_START - HOVER) * (j + 1) / sps
            step(1, "E descend")

        print("[F] trigger on — driving", flush=True)
        finger[:] = -FINGER
        fasten_step = -1
        for _ in range(int(1.5 * sps)):
            s_, a_ = horn_seat_axis()
            t_live = ((scene.screws[i_s].data.root_pos_w - s_) * a_).sum(-1)
            standoff[:] = torch.maximum(t_live + 0.0004, torch.full_like(t_live, BIT_STOP))
            step(1, "F drive")
            if fasten_step < 0 and bool((scene.fastened[:, i_s] >= 0).all()):
                fasten_step = stepno
                hold_fork = False  # the first screw holds the fork now — the hand lets go
            if fasten_step > 0 and stepno - fasten_step > sps // 4:
                break
        fasten_steps.append(fasten_step)
        print(f"[F] M3 {k} fastened at step {fasten_step}", flush=True)

        print("[G] trigger off, drill lifts off gently", flush=True)
        finger[:] = 0.0
        z_now = float(standoff[0])
        for j in range(int(RETREAT_S * sps)):
            ease = 0.5 - 0.5 * math.cos(math.pi * (j + 1) / (RETREAT_S * sps))
            standoff[:] = z_now + (HOVER - z_now) * ease
            step(1, "G retreat")

    print("[G] all horn screws driven — the drill parks aside", flush=True)
    st = torch.zeros(n, 13, device=dev)
    st[:, 0:3] = origin + torch.tensor(DRILL_PARK, device=dev)
    st[:, 3] = 1.0
    scene.drill.write_root_state_to_sim(st, None)
    track_drill = False
    step(sps // 4, "G park")

    # ============ I) STRESS: back to the lying pose, knock a screw, wrench the distal ==========
    print("[I] back to the lying pose; knock a driven M3, wrench the distal — nothing may "
          "come apart", flush=True)
    grab(q_hold, HOLD_HEIGHT)
    knock = torch.tensor((0.25, 0.0, 0.35, 0.0, 0.0, 0.0), device=dev).expand(n, 6).contiguous()
    scene.screws[ne].write_root_velocity_to_sim(knock, None)
    step(sps // 2, "I knock")
    err_knock = horn_screw_errs()
    wrench = torch.tensor((0.0, 0.4, 0.5, 0.0, 0.0, 0.0), device=dev).expand(n, 6).contiguous()
    scene.distal.write_root_velocity_to_sim(wrench, None)
    step(sps // 2, "I wrench")
    err_la_stress = la_err()

    # ============ J) FINALE: lift the WHOLE robot, shake, rotate, set down, release ============
    print("[J] the clamp lifts the whole robot +15 cm, shakes, rotates 180 deg, sets it down, "
          "then RELEASES it", flush=True)
    anchor = arm_target_pos.clone()
    lie_q = arm_target_quat.clone()

    def move_arm(dz: float, dx: float = 0.0, dy: float = 0.0, yaw: float = 0.0) -> None:
        arm_target_pos[:] = anchor + torch.tensor((dx, dy, dz), device=dev)
        half = 0.5 * yaw
        qz = torch.tensor((math.cos(half), 0.0, 0.0, math.sin(half)), device=dev).expand(n, 4)
        arm_target_quat[:] = quat_mul(qz, lie_q)

    for j in range(sps):
        move_arm(0.15 * (j + 1) / sps)
        step(1, "J lift")
    for j in range(int(1.5 * sps)):
        ph = 2.0 * math.pi * 3.0 * (j + 1) / sps
        move_arm(0.15, dx=0.015 * math.sin(ph), dy=0.010 * math.sin(0.7 * ph))
        step(1, "J shake")
    # the payoff: the joint the screws just closed is a REAL driven servo joint
    print("[J] the assembled elbow ARTICULATES — driven by the servo it was screwed onto",
          flush=True)
    step(sps // 2, "J settle")  # let the shake ring down before the joint sweep
    elbow_max = 0.0
    for j in range(int(3.0 * sps)):
        scene.set_elbow_target(math.radians(30.0)
                               * math.sin(2.0 * math.pi * (j + 1) / (3.0 * sps)))
        step(1, "J elbow")
        _, sq_ = scene.lower_arm_seat_w()
        qz_, _ = scene._elbow_angle_split(scene.distal.data.root_quat_w, sq_)
        elbow_max = max(elbow_max, abs(math.degrees(
            2.0 * math.atan2(qz_[0, 3].item(), qz_[0, 0].item()))))
    scene.set_elbow_target(0.0)
    step(sps // 2, "J elbow0")
    print(f"[J] elbow articulated to {elbow_max:.1f} deg and back", flush=True)
    for j in range(int(1.5 * sps)):
        move_arm(0.15, yaw=math.pi * (j + 1) / (1.5 * sps))
        step(1, "J rotate")
    for j in range(sps):
        move_arm(0.15 - 0.13 * (j + 1) / sps, yaw=math.pi)
        step(1, "J lower")
    hold_arm = False
    print("[J] clamp RELEASED — the free assembled robot must hold together", flush=True)
    step(int(1.2 * sps), "J free")
    err_free, err_la_free = horn_screw_errs(), la_err()
    err_motor_free = motor_err()
    print(f"[J] resting free: max M3-in-seat err {err_free.max() * 1000:.2f} mm, fork-on-horn "
          f"err {err_la_free.max() * 1000:.2f} mm, servo-in-pocket err "
          f"{err_motor_free.max() * 1000:.2f} mm", flush=True)

    # ============ verdict ======================================================================
    all_fastened = bool((scene.fastened[:, ne:] >= 0).all())
    ok = (all_fastened and bool(picked.all()) and all(f > 0 for f in fasten_steps)
          and elbow_max > 20.0
          and float(err_knock.max()) < 0.0015 and float(err_la_stress.max()) < 0.0015
          and float(err_free.max()) < 0.0015 and float(err_la_free.max()) < 0.0015
          and float(err_motor_free.max()) < 0.0015)
    print(f"SO101-HORN-SMOKE | picked {int(picked.sum())}/{n_m3} | fastened at steps "
          f"{fasten_steps} | elbow swung {elbow_max:.1f} deg | errs (mm): "
          f"knock {err_knock.max() * 1000:.2f}, wrench {err_la_stress.max() * 1000:.2f}, "
          f"finale M3 {err_free.max() * 1000:.2f} / fork {err_la_free.max() * 1000:.2f} / "
          f"motor {err_motor_free.max() * 1000:.2f} | {'PASS' if ok else 'FAIL'}", flush=True)
    _close_and_exit(env)


def _close_and_exit(env) -> None:
    """Kit teardown regularly hangs INSIDE env.close()/app.close() — everything is printed by
    the time we get here, so arm the force-exit BEFORE closing."""
    import os
    import threading

    watchdog = threading.Timer(10.0, lambda: os._exit(0))
    watchdog.daemon = True
    watchdog.start()
    env.close()
    app.close()
    os._exit(0)


if __name__ == "__main__":
    main()

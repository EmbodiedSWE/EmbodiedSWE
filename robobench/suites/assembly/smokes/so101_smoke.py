"""Smoke test for SO101AssemblyScene — one linear run with a NullRobot (visual or headless).

THE POINT OF THIS TEST: prove that a real robot could plausibly perform every step. Each motion
here is one a robot embodiment can produce — a force push, a carried tool, or a scene rule an
embodiment triggers through contact (the fasten gate, the magnetic bit). The only test
scaffolding is the base clamp (a stand-in for a bench vise) and the re-fixturing teleports
between holes (a stand-in for re-clamping the workpiece).

Assembles the WHOLE elbow joint: the servo into the upper-arm pocket, its four M2 tab screws,
the forearm fork onto the output horn, and the eight M3 horn screws (four per fork side). All
twelve screws are picked up and carried to their holes on the drill's MAGNETIC BIT — how real
small-screw assembly is done: nothing holds a freely released screw upright in these holes (it
topples off the countersink or slides past the pin), so placement belongs to the bit, not to
gravity. Within each group the screws are identical and the scene's gate pairs any screw with
any free compatible hole.

The arm is steadied during the precision phases: before each, the smoke teleports it to the
working pose for the TARGET hole — near holes face up in the lying pose, far holes face up in a
flipped pose — and clamps its base there kinematically; it runs free the rest of the time.
Because the target hole's axis always points up, one pickup/carry/drive choreography serves
every hole. The procedure, in order:
  A) WIGGLE  — everything free; sinusoidal targets on the shoulder (1-2) and wrist (4-5 + gripper)
     joints, then back to zero. Proof the parts are live, not glued.
  B) INSERT  — a second hand (PD force at the CoM, orientation gripped) pushes the servo through
     the open approach to the corridor mouth, then slides it centered down the slip-fit corridor
     to the seat (kinematic — a force push wanders off-axis and jams against the printed walls);
     a firm kinematic press then holds it until the first screw bites.
  C) ROTATE  — joint 2 swings the arm flat under its own drive (the base clamp keeps it steady).
  Then, for each of the four M2 tab holes (near pair lying, far pair flipped):
  D) PLACE   — re-grab facing that hole up; the drill hovers over the lying screw, descends,
     and the MAGNETIC BIT takes it (a scene rule, like the gate), then lifts it off the workbench.
  E) DESCEND — the drill carries the screw over the hole and lowers it straight in, tip-first.
  F) DRIVE   — trigger -> gate -> the screw drives to its seat -> weld at depth (the first tab
     screw also welds motor->arm and the hand releases the servo).
  G) RETREAT — the drill retreats gently to its hover.
  H) ATTACH  — the drill parks; back in the lying pose, a hand clips the forearm FORK over the
     seated servo (the closed fork is a print-flex snap in the kit — the hand presses it on
     along the elbow axis and keeps HOLDING it until the first horn screw bites).
  Then, for each of the eight M3 horn holes (near four lying, far four flipped), the same
  D/E/F/G pickup-carry-drive-retreat: the near heads seat on the fork's inner plate, reached
  down the outer skin's access channels, threading into the horn; the far heads seat on the
  far plate over the servo's case-back holes. The first M3 also welds lower_arm->motor.
  I) STRESS  — knock a fastened M2 and an M3, wrench the servo and the forearm; nothing may
     come apart.
  J) FINALE  — the clamp lifts the WHOLE robot (both halves now), shakes it, rotates 180 deg,
     sets it down and RELEASES it: the robot rests assembled on the workbench.

Re-grabbing with welded parts attached does the teleport WELD-SAFE (see `grab`): welds off, arm
teleported, the welded CHAIN (motor, fork, every screw) re-seated at the fresh pose by the
scene's reconciler, welds back on — never letting an enabled weld see a large error (that yanks
the parts across the workspace). The drill PARKS clear before every re-fixture: a teleporting
arm sweeping through a hole-tracking drill collides mid-flip.

  python -m robobench.suites.assembly.smokes.so101_smoke --livestream 2
  python -m robobench.suites.assembly.smokes.so101_smoke --headless
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
from robobench.suites.assembly.smokes import close_and_exit  # noqa: E402

if TYPE_CHECKING:
    from robobench.suites.assembly.scenes.so101_assembly import SO101AssemblyScene

# A) wiggle: one full period at both ends returns to zero cleanly.
WIGGLE_HZ = 0.5
WIGGLE_PERIODS = 1
AMP = {"shoulder_pan": 0.5, "shoulder_lift": 0.3,
       "wrist_flex": 0.6, "wrist_roll": 1.2, "gripper": 0.4}
JOINT2_DEG = -90.0  # shoulder_lift pose for the fasten phases (the arm swings flat)
FINGER = 0.08       # squeeze torque at the drill hinge (N*m)
BIT_STOP = 0.001    # how far above the seat the bit stops driving (m) — keeps it off the arm
RETREAT_S = 1.5     # how long the drill lifts off after fastening (s) — slow = the arm stays calm
HOVER = 0.10        # bit-tip standoff above the seat before the drill descends (m)
STATE_NAMES = ("free", "driving", "fastened")  # labels for scene.state (0/1/2), for the readout
PICK_STOP = 0.002   # bit-tip standoff above the lying screw where the magnet takes it (m)
DRIVE_START = 0.002  # carried-screw head height above the seat where the driving begins (m)
NUM_NEAR = 2  # tab holes 0,1 = near wall; 2,3 = far wall (see the scene cfg seat table)
NUM_NEAR_M3 = 4  # horn holes 0-3 = near (horn) side; 4-7 = far (case-back) side
# The fork ATTACH is a SLIDE-ON (the clevis path): the fork approaches from the servo's front
# — where the forearm naturally extends, open space — mouth-first along the servo's long axis,
# lifted slightly off the engagement bosses, then presses down onto the horn. In the kit this
# is the print-flex snap; rigid bodies get the swept locating features' collision trimmed.
SLIDE_IN = 0.028    # slide travel along the fork's mouth axis (m): mouth starts past the nose
SLIDE_S = 1.2       # how long the slide takes (s)
FORK_LIFT = 0.0010  # lift off the seat during the slide, along the out-axis (m)
PRESS_S = 0.75      # how long the final press takes (s)

# The clamp: during the precision phases, re-write the base root state to the working pose each step
# (a stand-in for a vise); free otherwise. Kinematic, not a force grasp — a force PD stiff enough
# to hold the light arm (~0.35 kg) steady is unstable at this dt, and a stable one is too soft.
# HOLD=False runs fully free, for inspection.
HOLD = True
HOLD_HEIGHT = 0.08                              # base clamp height, near (lying) pose (m)
HOLD_HEIGHT_FAR = 0.12                          # far (flipped) pose: just high enough that the
                                                # flipped arm clears the table top, so the flip stays
                                                # visually near the lying pose (no big jump)
HOLD_QUAT = (0.7071068, 0.0, 0.7071068, 0.0)    # Ry(90): arm on its side, near holes facing up
FLIP_QUAT = (0.0, 1.0, 0.0, 0.0)                # Rx(180) pre-rotation: far holes facing up
# the servo, relative to the held base anchor: where it seats, and where it starts (out along -Y)
SERVO_SEAT_POS = (0.1491, -0.0535, -0.0025)
SERVO_SEAT_QUAT = (0.0, 0.0, 1.0, 0.0)
SERVO_START_OFFSET = 0.05
CORRIDOR = 0.015    # the last stretch of the insertion (m): force-push down to here, then a
                    # centered kinematic slide to the seat (see phase B)
DRILL_QUAT = (0.7071068, -0.7071068, 0.0, 0.0)  # drill working orientation: bit pointing down
DRILL_PARK = (0.70, -0.2, 0.145)  # the drill's standby spot, used before every re-fixture
# and at the end — it must clear the WHOLE assembled robot: with the distal attached the robot
# reaches ~0.55 m from the base, and the finale sweeps it 180 deg. On the workbench top (its
# east end, past the screw rows, 0.73 m from the base) — the old -x spot is off the table now.


def _amp_vec(art, device) -> torch.Tensor:
    return torch.tensor([AMP.get(n, 0.0) for n in art.joint_names], device=device)


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()  # registers "assembly.so101" (via the suite's configs/envs.py)
    env = ENVS.get("assembly.so101")().build(num_envs=args.num_envs, device=device)
    scene: SO101AssemblyScene = env.scene  # type: ignore[assignment]
    cfg = scene.cfg
    render = (not args.headless) or livestream_on
    n = env.num_envs
    dev = env.device
    no_action = torch.empty(0, device=dev)
    env.reset()

    sps = round(1.0 / env.dt)
    print(f"[so101] proximal joints={scene.proximal.joint_names} "
          f"distal joints={scene.distal.joint_names} dt={env.dt:.5f} ({sps} steps/s) "
          f"| {cfg.num_screws} screws / holes", flush=True)

    # env-frame anchor ON the working surface: every choreography height (clamp, park, hover)
    # is surface-relative, so the whole smoke rides the scene's workbench preset
    origin = env.iscene.env_origins + torch.tensor((0.0, 0.0, cfg.surface_z), device=dev)
    q_hold = torch.tensor(HOLD_QUAT, device=dev)
    q_far = quat_mul(torch.tensor(FLIP_QUAT, device=dev).unsqueeze(0), q_hold.unsqueeze(0))[0]
    q_drill = torch.tensor(DRILL_QUAT, device=dev).expand(n, 4)
    finger = torch.zeros(n, 1, device=dev)
    standoff = torch.full((n,), HOVER, device=dev)  # bit tip -> seat, along the axis
    i_j2 = scene.proximal.find_joints("shoulder_lift")[0][0]
    j2_target = torch.zeros(n, scene.proximal.num_joints, device=dev)
    distal_target = torch.zeros(n, scene.distal.num_joints, device=dev)
    # the hole being addressed — seat_axis(), the drill track and the readout follow it.
    # group "tab" = the M2 tab holes (upper_arm frame), "horn" = the M3 horn-line holes
    # (lower_arm frame); `screw` is the flat index of the screw being driven into it.
    cur = {"group": "tab", "hole": 0, "screw": 0}

    # THE HAND: "force" = PD push at the CoM with the orientation gripped (the insertability
    # test); "hold" = kinematic press into the seat (until the first screw bites);
    # grasp_offset[0] None = released. Gains sized for the 61 g servo, force capped like a hand.
    KP, KD, F_MAX = 150.0, 12.0, 4.0
    COM_B = torch.tensor((-0.11257, -0.0155, 0.0183), device=dev)  # servo CoM, link frame
    grasp_vec = torch.tensor((0.0, -SERVO_START_OFFSET, 0.0), device=dev)  # hand target offset, link
    grasp_offset: list = [None]   # None during the wiggle: the servo is a free body
    grasp_mode: list = ["force"]
    steady_arm = False  # a second hand keeps the arm still during the insertion press
    track_drill = False  # the drill stays put until it is actually needed (phase E)
    pick: dict = {"anchor": None}  # bit-tip approach anchor while picking a screw up (see step())
    hold_arm = False  # the kinematic clamp that holds the base in the air (the smoke's vise)
    hold_fork = False  # the hand holding the fork on the horn until the first M3 bites
    fork_off = {"slide": 0.0, "lift": 0.0}  # hand offset in the SEAT frame: slide = out along
    # the mouth axis (toward the servo's front), lift = off the seat along the out-axis
    grab_prev_quat: list = [None]  # last grab facing — a FLIP needs a long ring-down settle
    arm_target_pos = (origin + torch.tensor((0.0, 0.0, HOLD_HEIGHT), device=dev)).contiguous()
    # clone, NOT expand().contiguous(): at n=1 the expanded view is already "contiguous", so
    # contiguous() returns an ALIAS of q_hold — and grab()'s in-place clamp-target write would
    # silently overwrite q_hold with the flip pose (every later lying grab then re-applies the
    # far facing; the horn phases deadlock on the upside-down fork)
    arm_target_quat = q_hold.expand(n, 4).clone()
    stepno = 0

    import logging as _logging
    _logging.getLogger("isaaclab").setLevel(_logging.ERROR)  # the wrench API warns every call

    def seat_axis(group: str | None = None, hole: int | None = None):
        g = cur["group"] if group is None else group
        h = cur["hole"] if hole is None else hole
        s, a = (scene.elbow_screw_seats_w() if g == "tab"
                else scene.elbow_horn_screw_seats_w())
        return s[:, h], a[:, h]

    def step(k: int, phase: str) -> None:
        nonlocal stepno
        for _ in range(k):
            if hold_arm:  # the vise: the base root state is re-written to the clamp target
                st = torch.zeros(n, 13, device=dev)
                st[:, 0:3] = arm_target_pos
                st[:, 3:7] = arm_target_quat
                scene.proximal.write_root_state_to_sim(st, None)
            if steady_arm:
                zj = torch.zeros(n, scene.proximal.num_joints, device=dev)
                scene.proximal.write_joint_state_to_sim(zj, zj)
            if grasp_offset[0] is not None and grasp_mode[0] == "hold":
                st = torch.zeros(n, 13, device=dev)
                st[:, 0:3] = scene.proximal.data.body_pos_w[:, scene.b_ua]
                st[:, 3:7] = scene.proximal.data.body_quat_w[:, scene.b_ua]
                scene.motor.write_root_state_to_sim(st, None)
            elif grasp_offset[0] is not None:
                ap_, aq_ = scene.upper_arm_pose()
                mp_, mq_ = scene.motor.data.root_pos_w, scene.motor.data.root_quat_w
                off_l = grasp_vec.expand(n, 3)
                com_target = ap_ + quat_apply(aq_, off_l + COM_B)
                com_now = mp_ + quat_apply(mq_, COM_B.expand(n, 3))
                f_w = KP * (com_target - com_now) - KD * scene.motor.data.root_lin_vel_w
                f_mag = f_w.norm(dim=-1, keepdim=True)
                f_w = torch.where(f_mag > F_MAX, f_w * (F_MAX / f_mag), f_w)
                f_w[:, 2] += 0.061 * 9.81  # the hand carries the servo's weight
                scene.motor.set_external_force_and_torque(
                    quat_apply_inverse(mq_, f_w).unsqueeze(1),
                    torch.zeros(n, 1, 3, device=dev))
                st_ = torch.zeros(n, 13, device=dev)  # orientation held by the grip
                st_[:, 0:3] = mp_
                st_[:, 3:7] = aq_
                st_[:, 7:10] = scene.motor.data.root_lin_vel_w
                scene.motor.write_root_state_to_sim(st_, None)
            if hold_fork:  # the fork hand: the fork rides its live seat, offset by fork_off
                seat_p, seat_q = scene.lower_arm_seat_w()
                off = torch.tensor((-fork_off["slide"], 0.0, -fork_off["lift"]), device=dev)
                st = torch.zeros(n, 13, device=dev)
                st[:, 0:3] = seat_p + quat_apply(seat_q, off.expand(n, 3))
                st[:, 3:7] = seat_q
                scene.distal.write_root_state_to_sim(st, None)
                zj = torch.zeros(n, scene.distal.num_joints, device=dev)
                scene.distal.write_joint_state_to_sim(zj, zj)
            if pick["anchor"] is not None:  # pickup: the bit descends vertically onto the
                # lying screw (a fixed world anchor — the magnet snaps the screw to the bit)
                st = torch.zeros(n, 13, device=dev)
                tip_target = pick["anchor"] + torch.tensor((0.0, 0.0, 1.0), device=dev) * standoff.unsqueeze(-1)
                st[:, 0:3] = tip_target - quat_apply(q_drill, torch.tensor(
                    cfg.bit_tip, device=dev).expand(n, 3))
                st[:, 3:7] = q_drill
                scene.drill.write_root_state_to_sim(st, None)
            if track_drill:
                s, a = seat_axis()
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
                s, a = seat_axis()
                t_a = ((scene.screws[cur["screw"]].data.root_pos_w - s) * a).sum(-1)[0].item()
                tq = math.degrees(scene.drill.data.joint_pos[0, scene.i_trig].item())
                bv = scene.drill.data.joint_vel[0, scene.i_bit].item()
                print(f"  step {stepno:5d} [{phase:9s}] {STATE_NAMES[scene.state[0]]:8s} "
                      f"{cur['group']}{cur['hole']} depth-above-seat {t_a * 1000:+7.2f} mm | "
                      f"trigger {tq:+6.2f} deg | bit {bv:+6.2f} rad/s | fastened "
                      f"{int((scene.fastened[0] >= 0).sum())}/{cfg.num_screws}", flush=True)

    def release_grasp() -> None:
        grasp_vec.zero_()
        grasp_offset[0] = None
        zero = torch.zeros(n, 1, 3, device=dev)
        scene.motor.set_external_force_and_torque(zero, zero)

    def place_motor_at_approach() -> None:
        """Put the free servo at the pocket-approach pose (out along -Y) so the hand can take it.
        Derived from the clamp's CURRENT anchor (set by the last grab), so changing a hold
        height/pose constant can never desync the approach from the arm."""
        st = torch.zeros(n, 13, device=dev)
        st[:, 0:3] = arm_target_pos + torch.tensor(
            (SERVO_SEAT_POS[0], SERVO_SEAT_POS[1] - SERVO_START_OFFSET, SERVO_SEAT_POS[2]),
            device=dev)
        st[:, 3:7] = torch.tensor(SERVO_SEAT_QUAT, device=dev)
        scene.motor.write_root_state_to_sim(st, None)

    def motor_rel_err() -> torch.Tensor:
        ap, aq = scene.upper_arm_pose()
        return quat_apply_inverse(aq, scene.motor.data.root_pos_w - ap).norm(dim=-1)

    def screws_rel_err() -> torch.Tensor:
        """Max distance of any fastened tab screw from its own hole's seat (m)."""
        seats, _ = scene.elbow_screw_seats_w()
        errs = []
        for s in range(cfg.num_elbow_screws):
            h = scene.fastened[:, s].clamp_min(0)
            err = (scene.screws[s].data.root_pos_w
                   - seats[torch.arange(n, device=dev), h]).norm(dim=-1)
            errs.append(torch.where(scene.fastened[:, s] >= 0, err, torch.zeros_like(err)))
        return torch.stack(errs, dim=1).max(dim=1).values

    def la_err() -> torch.Tensor:
        """Distance of the forearm fork from its seat on the motor's horn (m)."""
        exp_p, _ = scene.lower_arm_seat_w()
        return (scene.distal.data.root_pos_w - exp_p).norm(dim=-1)

    def m3_rel_err() -> torch.Tensor:
        """Max distance of any fastened M3 from its own hole's seat (m)."""
        seats, _ = scene.elbow_horn_screw_seats_w()
        arange = torch.arange(n, device=dev)
        errs = []
        for k in range(cfg.num_horn_screws):
            i_s = cfg.num_elbow_screws + k
            h = (scene.fastened[:, i_s] - cfg.num_elbow_holes).clamp_min(0)
            err = (scene.screws[i_s].data.root_pos_w - seats[arange, h]).norm(dim=-1)
            errs.append(torch.where(scene.fastened[:, i_s] >= 0, err, torch.zeros_like(err)))
        return torch.stack(errs, dim=1).max(dim=1).values

    def park_drill() -> None:
        nonlocal track_drill
        track_drill = False
        st = torch.zeros(n, 13, device=dev)
        st[:, 0:3] = origin + torch.tensor(DRILL_PARK, device=dev)
        st[:, 3] = 1.0
        scene.drill.write_root_state_to_sim(st, None)

    def grab(joint2_deg: float, base_quat: torch.Tensor, height: float) -> None:
        """Teleport the arm to a working pose (base at the anchor in `base_quat`; shoulder_lift =
        joint2_deg, the rest 0) and clamp the base exactly there. Called before each precision
        phase so every phase starts from the same clean pose, never accumulated drift.
        WELD-SAFE: enabled welds must never see the teleport as an error (they would yank the
        welded parts across the workspace), and the link poses read back are stale until the sim
        steps — so the welds are dropped, the arm is teleported and stepped under the clamp, the
        welds re-enabled, and the whole welded CHAIN (motor, fork, screws) re-seated by the
        scene's reconciler against the fresh pose. The drill PARKS first: a teleporting arm
        sweeping through a hole-tracking drill collides mid-flip."""
        nonlocal hold_arm
        park_drill()
        fastened = scene.fastened.clone()
        for i in range(n):  # drop every weld (the part welds follow the screws)
            for s in range(cfg.num_screws):
                if int(fastened[i, s]) >= 0:
                    scene._set_weld(i, s, 0, False)
        st = torch.zeros(n, 13, device=dev)
        st[:, 0:3] = origin + torch.tensor((0.0, 0.0, height), device=dev)
        st[:, 3:7] = base_quat
        scene.proximal.write_root_state_to_sim(st, None)
        jq = torch.zeros(n, scene.proximal.num_joints, device=dev)
        jq[:, i_j2] = math.radians(joint2_deg)
        scene.proximal.write_joint_state_to_sim(jq, torch.zeros_like(jq))
        j2_target[:, i_j2] = math.radians(joint2_deg)  # the drive holds the joint too
        arm_target_pos[:] = origin + torch.tensor((0.0, 0.0, height), device=dev)
        arm_target_quat[:] = base_quat
        hold_arm = HOLD
        step(2, "re-grab")  # FK refresh under the clamp, enough to re-seat the welded parts
        if bool((fastened >= 0).any()):
            ap, aq = scene.upper_arm_pose()  # fresh now
            for i in range(n):
                for s in range(cfg.num_screws):
                    h = int(fastened[i, s])
                    if h >= 0:
                        scene._set_weld(i, s, h, True)
            scene._reconcile_fastened(arm_pose=(ap, aq), motor_pose=(ap, aq))
        # settle to the drives' LOADED equilibrium: the joints are written at their exact
        # targets, then the drives relax (sag) AND the freshly-posed arm rings like a clamped
        # pendulum for a while (worst right after a facing FLIP — the sub-mm oscillation is
        # enough to break the fasten gate). Everything read after a grab must see a QUIET
        # pose; the re-enabled welds track the settling. A same-facing re-grab barely moves
        # the arm, so a short settle keeps the pacing tight.
        flipped = bool((base_quat - grab_prev_quat[0]).abs().max() > 1e-6) \
            if grab_prev_quat[0] is not None else True
        grab_prev_quat[0] = base_quat.clone()
        step(sps if flipped else sps // 4, "re-grab")

    def release() -> None:
        nonlocal hold_arm
        hold_arm = False

    # ============ A) WIGGLE: everything free; both halves articulate =============================
    print("[A] settle + wiggle: the free arm articulates; the servo lies free on the workbench", flush=True)
    amp_p = _amp_vec(scene.proximal, dev).expand(n, -1)
    amp_d = _amp_vec(scene.distal, dev).expand(n, -1)
    step(sps // 4, "A settle")
    for i in range(int(WIGGLE_PERIODS / WIGGLE_HZ * sps)):
        s = math.sin(2.0 * math.pi * WIGGLE_HZ * (i + 1) * env.dt)
        j2_target = amp_p * s
        distal_target = amp_d * s
        step(1, "A wiggle")
    j2_target = torch.zeros(n, scene.proximal.num_joints, device=dev)
    distal_target = torch.zeros(n, scene.distal.num_joints, device=dev)
    step(sps // 4, "A return")
    jp = scene.proximal.data.joint_pos.abs().max().item()
    jd = scene.distal.data.joint_pos.abs().max().item()
    print(f"[A] returned to zero: max|q| proximal={jp:.3f} distal={jd:.3f} rad", flush=True)

    # ============ B) INSERT: grab the arm, then the hand pushes the servo into the pocket =======
    print("[B] arm grabbed at the working pose; the hand pushes the servo straight in", flush=True)
    steady_arm = True  # freeze the joints BEFORE the grab settles: the approach coordinates
    # below assume the exact zero pose, and letting the arm sag first then snapping it back
    # jolts the servo visibly
    grab(0.0, q_hold, HOLD_HEIGHT)  # teleport to the working pose (joints zero) and clamp the arm
    place_motor_at_approach()
    grasp_vec[1] = -SERVO_START_OFFSET
    grasp_offset[0] = 1.0
    grasp_mode[0] = "force"
    step(sps // 4, "B grasp")  # let the hand take the servo before pushing
    # The hand pushes the servo by FORCE through the open approach down to the corridor mouth,
    # then slides it centered along the corridor axis to the seat — the slip fit takes a
    # straight, centered insertion, but the force-PD hand wanders off-axis at mm scale and
    # jams against the printed corridor walls. Both stages track the LIVE arm pose, so
    # re-positioning the hold can never desync them.
    for j in range(int(0.8 * sps)):
        grasp_vec[1] = -SERVO_START_OFFSET + (SERVO_START_OFFSET - CORRIDOR) * (j + 1) / (0.8 * sps)
        step(1, "B insert")
    ins_err = motor_rel_err()
    print(f"[B] force push reached {ins_err.mean() * 1000:.2f} mm from the seat (corridor mouth); "
          f"sliding down the slip-fit corridor", flush=True)
    release_grasp()
    for j in range(int(0.7 * sps)):
        ap0, aq0 = scene.upper_arm_pose()
        rel = torch.tensor((0.0, -CORRIDOR * (1.0 - (j + 1) / (0.7 * sps)), 0.0), device=dev)
        st = torch.zeros(n, 13, device=dev)
        st[:, 0:3] = ap0 + quat_apply(aq0, rel.expand(n, 3))
        st[:, 3:7] = aq0
        scene.motor.write_root_state_to_sim(st, None)
        step(1, "B slide")
    ins_err = motor_rel_err()
    print(f"[B] servo seated {ins_err.mean() * 1000:.2f} mm from its seat — INSERTED", flush=True)
    scene.proximal.write_joint_state_to_sim(
        scene.proximal.data.joint_pos.clone(), torch.zeros_like(scene.proximal.data.joint_vel))
    grasp_offset[0] = 1.0
    grasp_mode[0] = "hold"  # the hand PRESSES the servo in place until the first screw bites
    step(sps // 4, "B align")
    steady_arm = False

    # ============ C) ROTATE: joint 2 swings the arm flat =======================================
    print(f"[C] joint 2 driven to {JOINT2_DEG:.0f} deg", flush=True)
    for j in range(int(1.5 * sps)):
        j2_target[:, i_j2] = math.radians(JOINT2_DEG) * (j + 1) / (1.5 * sps)
        step(1, "C rotate")
    step(sps // 4, "C settle")

    # ============ D-G) per hole: pick up, carry, drive, retreat (shared by M2s and M3s) ========
    def place_and_drive(i_s: int, on_fastened) -> tuple[bool, int]:
        """One screw, into the CURRENT hole (`cur`): D pickup on the magnetic bit, E carry over
        the hole and lower in, F trigger -> gate -> drive -> weld (`on_fastened` fires once at
        the weld), G gentle retreat. Every screw is CARRIED — nothing holds a freely released
        screw upright in these holes (it topples off the countersink or slides past the pin);
        the magnet carry is deterministic on every hole."""
        nonlocal track_drill
        print(f"[D] {cur['group']} hole {cur['hole']}: the drill picks screw {i_s} up with its "
              f"magnetic bit", flush=True)
        pick["anchor"] = scene.screws[i_s].data.root_pos_w.clone()  # the lying screw
        track_drill = False
        standoff[:] = HOVER
        step(sps // 8, "D hover")
        for j in range(sps):  # descend until the magnet takes the screw
            standoff[:] = HOVER + (PICK_STOP - HOVER) * (j + 1) / sps
            step(1, "D pickup")
            if bool(scene.attached[:, i_s].all()):
                break
        got = bool(scene.attached[:, i_s].all())
        print(f"[D] screw {i_s} {'is on the bit' if got else 'MISSED the pickup'}", flush=True)
        z_now = float(standoff[0])
        for j in range(sps // 2):  # lift it off the workbench
            standoff[:] = z_now + (HOVER - z_now) * (j + 1) / (sps // 2)
            step(1, "D lift")
        pick["anchor"] = None

        print("[E] the drill carries the screw over the hole and lowers it in", flush=True)
        standoff[:] = HOVER
        track_drill = True
        step(sps // 8, "E hover")
        for j in range(sps):
            standoff[:] = HOVER + (DRIVE_START - HOVER) * (j + 1) / sps
            step(1, "E descend")

        print("[F] trigger on — driving", flush=True)
        finger[:] = -FINGER
        fastened_at = -1
        for _ in range(int(1.5 * sps)):
            # the bit rides just above the screw's live depth — following it down as it drives
            # in, or back up as a deep-lying screw is drawn to its seat
            s_, a_ = seat_axis()
            t_live = ((scene.screws[i_s].data.root_pos_w - s_) * a_).sum(-1)
            standoff[:] = torch.maximum(t_live + 0.0004, torch.full_like(t_live, BIT_STOP))
            step(1, "F drive")
            if fastened_at < 0 and bool((scene.fastened[:, i_s] >= 0).all()):
                fastened_at = stepno
                on_fastened()
            if fastened_at > 0 and stepno - fastened_at > sps // 4:
                break
        print(f"[F] screw {i_s} fastened at step {fastened_at}", flush=True)

        print("[G] trigger off, drill lifts off gently", flush=True)
        finger[:] = 0.0
        z_now = float(standoff[0])
        for j in range(int(RETREAT_S * sps)):  # cosine ease-out so the bit un-seats calmly
            ease = 0.5 - 0.5 * math.cos(math.pi * (j + 1) / (RETREAT_S * sps))
            standoff[:] = z_now + (HOVER - z_now) * ease
            step(1, "G retreat")
        return got, fastened_at

    def release_servo_hand() -> None:  # the first tab screw holds the servo now
        if grasp_offset[0] is not None:
            release_grasp()

    picked = torch.zeros(cfg.num_elbow_screws, dtype=torch.bool)
    fasten_steps: list[int] = []
    for hole in range(cfg.num_elbow_screws):
        near = hole < NUM_NEAR
        cur.update(group="tab", hole=hole, screw=hole)
        grab(JOINT2_DEG, q_hold if near else q_far, HOLD_HEIGHT if near else HOLD_HEIGHT_FAR)
        got, at = place_and_drive(hole, release_servo_hand)
        picked[hole] = got
        fasten_steps.append(at)
    if grasp_offset[0] is not None:
        release_grasp()

    # ============ H) ATTACH: drill parks; the hand slides the forearm fork onto the servo ======
    print("[H] all tab screws driven — the drill parks; back in the lying pose, the hand slides "
          "the forearm fork onto the servo, mouth-first along its axis (the clevis path), then "
          "presses it down onto the horn", flush=True)
    grab(JOINT2_DEG, q_hold, HOLD_HEIGHT)  # the horn side faces up in the lying pose
    hold_fork = True
    fork_off.update(slide=SLIDE_IN, lift=FORK_LIFT)
    step(sps // 2, "H hover")
    for j in range(int(SLIDE_S * sps)):
        fork_off["slide"] = SLIDE_IN * (1.0 - (j + 1) / (SLIDE_S * sps))
        step(1, "H slide")
    for j in range(int(PRESS_S * sps)):
        fork_off["lift"] = FORK_LIFT * (1.0 - (j + 1) / (PRESS_S * sps))
        step(1, "H press")
    step(sps // 4, "H seated")
    print(f"[H] fork slid on and pressed onto the horn: fork err {la_err().max() * 1000:.2f} mm "
          f"(the hand keeps holding until the first horn screw bites)", flush=True)

    # ============ D-G again) the eight M3 horn screws: near four lying, far four flipped =======
    def release_fork_hand() -> None:  # the first horn screw holds the fork now
        nonlocal hold_fork
        hold_fork = False

    picked_m3 = torch.zeros(cfg.num_horn_screws, dtype=torch.bool)
    m3_steps: list[int] = []
    for k in range(cfg.num_horn_screws):
        if k == NUM_NEAR_M3:  # the far holes face down in the lying pose — re-fixture flipped
            print("[re-fixture] flip: far side up (weld-safe re-grab)", flush=True)
            grab(JOINT2_DEG, q_far, HOLD_HEIGHT_FAR)
        cur.update(group="horn", hole=k, screw=cfg.num_elbow_screws + k)
        got, at = place_and_drive(cfg.num_elbow_screws + k, release_fork_hand)
        picked_m3[k] = got
        m3_steps.append(at)
    park_drill()
    step(sps // 4, "park")

    # ============ I) STRESS: knock an M2 and an M3, wrench the servo and the forearm ===========
    print("[I] back to the lying pose; joint 2 swings back up (the reverse of C), then knock a "
          "fastened M2 and an M3, wrench the servo and the forearm — nothing may come apart",
          flush=True)
    grab(JOINT2_DEG, q_hold, HOLD_HEIGHT)  # re-grab at the joints' CURRENT angle — grabbing at 0
    # would snap joint 2 by 90 deg in one frame; instead it swings back under its own drive
    for j in range(int(1.5 * sps)):
        j2_target[:, i_j2] = math.radians(JOINT2_DEG) * (1.0 - (j + 1) / (1.5 * sps))
        step(1, "I unrotate")
    step(sps // 4, "I settle")
    knock = torch.tensor((0.25, 0.0, 0.35, 0.0, 0.0, 0.0), device=dev).expand(n, 6).contiguous()
    scene.screws[0].write_root_velocity_to_sim(knock, None)
    scene.screws[cfg.num_elbow_screws].write_root_velocity_to_sim(knock, None)
    step(sps // 2, "I knock")
    err_knock = torch.maximum(screws_rel_err(), m3_rel_err())
    wrench = torch.tensor((0.0, 0.4, 0.5, 0.0, 0.0, 0.0), device=dev).expand(n, 6).contiguous()
    scene.motor.write_root_velocity_to_sim(wrench, None)
    scene.distal.write_root_velocity_to_sim(wrench, None)
    step(sps // 2, "I wrench")
    err_motor_stress = motor_rel_err()
    err_la_stress = la_err()

    # ============ J) FINALE: re-grab, then lift, shake, rotate, set down, release ==============
    print("[J] re-grabbed; the clamp lifts the whole robot +15 cm, shakes, rotates 180 deg, sets "
          "it down, then RELEASES it", flush=True)
    grab(0.0, q_hold, HOLD_HEIGHT)  # re-grab at the working pose to pick the assembled robot up
    anchor = arm_target_pos.clone()  # the clamp's base anchor
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
    release()  # let go
    print("[J] clamp RELEASED — the free assembled robot must hold together", flush=True)
    step(int(1.2 * sps), "J free")
    err_free, err_m3_free = screws_rel_err(), m3_rel_err()
    err_motor_free, err_la_free = motor_rel_err(), la_err()
    print(f"[J] resting free on the workbench: max M2-in-seat err {err_free.max() * 1000:.2f} mm, "
          f"M3-in-seat err {err_m3_free.max() * 1000:.2f} mm, servo-in-pocket err "
          f"{err_motor_free.max() * 1000:.2f} mm, fork-on-horn err "
          f"{err_la_free.max() * 1000:.2f} mm", flush=True)

    # ============ verdict ======================================================================
    all_fastened = bool((scene.fastened >= 0).all())
    ok = (all_fastened and bool(picked.all()) and bool(picked_m3.all())
          and all(f > 0 for f in fasten_steps) and all(f > 0 for f in m3_steps)
          and elbow_max > 20.0
          and float(err_knock.max()) < 0.0015 and float(err_motor_stress.max()) < 0.0015
          and float(err_la_stress.max()) < 0.0015
          and float(err_free.max()) < 0.0015 and float(err_m3_free.max()) < 0.0015
          and float(err_motor_free.max()) < 0.0015 and float(err_la_free.max()) < 0.0015)
    print(f"SO101-SMOKE | servo inserted to {ins_err.mean() * 1000:.2f} mm (aligned after) | "
          f"M2 placed {int(picked.sum())}/{cfg.num_elbow_screws} at {fasten_steps} | "
          f"M3 placed {int(picked_m3.sum())}/{cfg.num_horn_screws} at {m3_steps} | "
          f"elbow swung {elbow_max:.1f} deg | errs (mm): "
          f"knock {err_knock.max() * 1000:.2f}, wrench {err_motor_stress.max() * 1000:.2f}/"
          f"{err_la_stress.max() * 1000:.2f}, finale {err_free.max() * 1000:.2f}/"
          f"{err_m3_free.max() * 1000:.2f}/{err_motor_free.max() * 1000:.2f}/"
          f"{err_la_free.max() * 1000:.2f} | {'PASS' if ok else 'FAIL'}", flush=True)
    close_and_exit(env, app)


if __name__ == "__main__":
    main()

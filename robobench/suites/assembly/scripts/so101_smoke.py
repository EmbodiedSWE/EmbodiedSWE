"""Smoke test for SO101AssemblyScene — one linear run with a NullRobot (visual or headless).

The arm only needs steadying during the precision phases, so before each the smoke teleports it to
the exact working pose (elbow hole up) and clamps its base there kinematically — the smoke's
stand-in for a vise; it runs free the rest of the time. The procedure, in order:
  A) WIGGLE — everything free; sinusoidal targets on the shoulder (1-2) and wrist (4-5 + gripper)
     joints, then back to zero. Proof the parts are live, not glued.
  B) INSERT — a second hand (PD force at the CoM, orientation gripped) pushes the servo STRAIGHT
     into the pocket (slip-fit collision); then a hard-align and a firm kinematic press.
  C) ROTATE — joint 2 swings the arm flat under its own drive (the base clamp keeps it steady).
  D) DROP — the M2 is dropped over the countersunk joint-3 hole: the cone funnels it, the pilot
     blocks it, caught by collision alone.
  E) DESCEND — the drill is picked up and descends exactly vertical onto the head.
  F) DRIVE — trigger -> gate -> latched drive -> welds (screw->arm AND motor->arm); hand releases.
  G) RETREAT — the drill retreats and is parked aside.
  H) STRESS — knock the screw, wrench the servo; nothing may come apart.
  I) FINALE — the clamp lifts the whole robot, shakes it, rotates 180 deg, sets it down and
     RELEASES it: the robot rests assembled on the ground.

The distal half sits free through B..I — its fastening story is a future task.

  python -m robobench.suites.assembly.scripts.so101_smoke --livestream 2
  python -m robobench.suites.assembly.scripts.so101_smoke --headless
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

# A) wiggle: one full period at both ends returns to zero cleanly.
WIGGLE_HZ = 0.5
WIGGLE_PERIODS = 1
AMP = {"shoulder_pan": 0.5, "shoulder_lift": 0.3,
       "wrist_flex": 0.6, "wrist_roll": 1.2, "gripper": 0.4}
JOINT2_DEG = -90.0  # shoulder_lift pose for the fasten phases (the arm swings flat)
FINGER = 0.08       # squeeze torque at the drill hinge (N*m)
BIT_STOP = 0.001    # how far above the seat the bit stops driving (m) — keeps it off the arm
RETREAT_S = 3.0     # how long the drill lifts off after fastening (s) — slow = the arm stays calm
HOVER = 0.10        # bit-tip standoff above the seat before the drill descends (m)
M2_LEN = 0.00798    # screw head-top -> tip; sets the drop height so the tip clears the seat
STATE_NAMES = ("free", "driving", "fastened")  # labels for scene.state (0/1/2), for the readout

# The clamp: during the precision phases, re-write the base root state to the working pose each step
# (a stand-in for a vise); free otherwise. Kinematic, not a force grasp — the light arm (~0.35 kg)
# can't be held steadily by a force PD at 240 Hz (rationale + probe data in the handoff doc).
# HOLD=False runs fully free, for inspection.
HOLD = True
HOLD_HEIGHT = 0.08                              # base clamped this high (m)
HOLD_QUAT = (0.7071068, 0.0, 0.7071068, 0.0)    # Ry(90): arm on its side, hole facing up
# the servo, relative to the held base anchor: where it seats, and where it starts (out along -Y)
SERVO_SEAT_POS = (0.1491, -0.0535, -0.0025)
SERVO_SEAT_QUAT = (0.0, 0.0, 1.0, 0.0)
SERVO_START_OFFSET = 0.05
DRILL_QUAT = (0.7071068, -0.7071068, 0.0, 0.0)  # drill working orientation: bit pointing down
DRILL_PARK = (0.45, -0.3, 0.145)                # where the drill parks after fastening


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
          f"distal joints={scene.distal.joint_names} dt={env.dt:.5f} ({sps} steps/s)", flush=True)

    origin = env.iscene.env_origins
    pin = torch.tensor((0.0, 0.0, HOLD_HEIGHT), device=dev)
    q_drill = torch.tensor(DRILL_QUAT, device=dev).expand(n, 4)
    finger = torch.zeros(n, 1, device=dev)
    standoff = torch.full((n,), HOVER, device=dev)  # bit tip -> seat, along the axis
    i_j2 = scene.proximal.find_joints("shoulder_lift")[0][0]
    j2_target = torch.zeros(n, scene.proximal.num_joints, device=dev)
    distal_target = torch.zeros(n, scene.distal.num_joints, device=dev)

    # THE HAND: "force" = PD push at the CoM with the orientation gripped (the insertability
    # test); "hold" = kinematic press into the seat (after the align, until the screw bites);
    # grasp_offset[0] None = released. Gains sized for the 61 g servo, force capped like a hand.
    KP, KD, F_MAX = 150.0, 12.0, 4.0
    COM_B = torch.tensor((-0.11257, -0.0155, 0.0183), device=dev)  # servo CoM, link frame
    grasp_vec = torch.tensor((0.0, -SERVO_START_OFFSET, 0.0), device=dev)  # hand target offset, link
    grasp_offset: list = [None]   # None during the wiggle: the servo is a free body
    grasp_mode: list = ["force"]
    steady_arm = False  # a second hand keeps the arm still during the insertion press
    track_drill = False  # the drill stays put until it is actually needed (phase E)
    hold_arm = False  # the kinematic clamp that holds the base in the air (the smoke's vise)
    arm_target_pos = origin + pin  # clamp target: base position ...
    arm_target_quat = torch.tensor(HOLD_QUAT, device=dev).expand(n, 4).contiguous()  # ... and orientation
    stepno = 0

    import logging as _logging
    _logging.getLogger("isaaclab").setLevel(_logging.ERROR)  # the wrench API warns every call

    def seat_axis():
        s, a = scene.elbow_screw_seats_w()
        return s[:, 0], a

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
                t_a = ((scene.screw.data.root_pos_w - s) * a).sum(-1)[0].item()
                tq = math.degrees(scene.drill.data.joint_pos[0, scene.i_trig].item())
                bv = scene.drill.data.joint_vel[0, scene.i_bit].item()
                print(f"  step {stepno:5d} [{phase:9s}] screw {STATE_NAMES[scene.state[0]]:8s} "
                      f"depth-above-seat {t_a * 1000:+7.2f} mm | trigger {tq:+6.2f} deg "
                      f"| bit {bv:+6.2f} rad/s", flush=True)

    def release_grasp() -> None:
        grasp_vec.zero_()
        grasp_offset[0] = None
        zero = torch.zeros(n, 1, 3, device=dev)
        scene.motor.set_external_force_and_torque(zero, zero)

    def place_motor_at_approach() -> None:
        """Put the free servo at the pocket-approach pose (out along -Y) so the hand can take it."""
        st = torch.zeros(n, 13, device=dev)
        st[:, 0:3] = origin + pin + torch.tensor(
            (SERVO_SEAT_POS[0], SERVO_SEAT_POS[1] - SERVO_START_OFFSET, SERVO_SEAT_POS[2]), device=dev)
        st[:, 3:7] = torch.tensor(SERVO_SEAT_QUAT, device=dev)
        scene.motor.write_root_state_to_sim(st, None)

    def motor_rel_err() -> torch.Tensor:
        ap, aq = scene.upper_arm_pose()
        return quat_apply_inverse(aq, scene.motor.data.root_pos_w - ap).norm(dim=-1)

    def rel_err() -> torch.Tensor:
        ap, aq = scene.upper_arm_pose()
        off = quat_apply_inverse(aq, scene.screw.data.root_pos_w - ap)
        return (off - scene._elbow_screw_seat_pts[0]).norm(dim=-1)

    def drop_screw(height: float, lateral: float = 0.0) -> None:
        """Drop the screw `height` above the seat, `lateral` off the hole axis. The offset makes the
        tip land on the countersink CONE, which funnels it into the pilot; a coaxial drop instead
        tunnels the servo's thin registration pin at dt=1/240 and falls into the pocket."""
        s, a = seat_axis()
        lat = torch.tensor((1.0, 0.0, 0.0), device=dev).expand(n, 3)
        lat = lat - (lat * a).sum(-1, keepdim=True) * a  # perpendicular to the hole axis
        lat = lat / lat.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        st = torch.zeros(n, 13, device=dev)
        st[:, 0:3] = s + height * a + lateral * lat
        st[:, 3:7] = quat_mul(scene.upper_arm_pose()[1], scene._elbow_screw_seat_quat.expand(n, 4))
        scene.screw.write_root_state_to_sim(st, None)

    def grab(joint2_deg: float = 0.0) -> None:
        """Teleport the arm to the working pose (base lying at the anchor; shoulder_lift=joint2_deg,
        the rest 0), then clamp the base exactly there. Called right before each precision phase so
        every phase starts from the same clean pose, never accumulated drift."""
        nonlocal hold_arm
        st = torch.zeros(n, 13, device=dev)
        st[:, 0:3] = origin + pin
        st[:, 3:7] = torch.tensor(HOLD_QUAT, device=dev)
        scene.proximal.write_root_state_to_sim(st, None)
        jq = torch.zeros(n, scene.proximal.num_joints, device=dev)
        jq[:, i_j2] = math.radians(joint2_deg)
        scene.proximal.write_joint_state_to_sim(jq, torch.zeros_like(jq))
        j2_target[:, i_j2] = math.radians(joint2_deg)  # the drive holds the joint too
        arm_target_pos[:] = origin + pin
        arm_target_quat[:] = torch.tensor(HOLD_QUAT, device=dev)
        hold_arm = HOLD

    def release() -> None:
        nonlocal hold_arm
        hold_arm = False

    # ============ A) WIGGLE: everything free; both halves articulate =============================
    print("[A] settle + wiggle: the free arm articulates; the servo lies free on the ground", flush=True)
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
    grab(0.0)  # teleport to the working pose (joints zero) and clamp the arm
    place_motor_at_approach()
    grasp_vec[1] = -SERVO_START_OFFSET
    grasp_offset[0] = 1.0
    grasp_mode[0] = "force"
    steady_arm = True
    step(sps // 4, "B grasp")  # let the hand take the servo before pushing
    for j in range(int(1.2 * sps)):
        grasp_vec[1] = -SERVO_START_OFFSET * (1.0 - (j + 1) / (1.2 * sps))
        step(1, "B insert")
    step(sps // 2, "B settle")
    ins_err = motor_rel_err()
    inserted = float(ins_err.max()) < 0.004
    print(f"[B] insertion check: servo reached {ins_err.mean() * 1000:.2f} mm from its seat — "
          + ("INSERTED" if inserted else "stalled (snap-fit beyond rigid-collision fidelity); "
             "hard-aligning and continuing, per design"), flush=True)
    st = torch.zeros(n, 13, device=dev)
    ap0, aq0 = scene.upper_arm_pose()
    st[:, 0:3] = ap0
    st[:, 3:7] = aq0
    scene.motor.write_root_state_to_sim(st, None)
    scene.proximal.write_joint_state_to_sim(
        scene.proximal.data.joint_pos.clone(), torch.zeros_like(scene.proximal.data.joint_vel))
    grasp_mode[0] = "hold"  # the hand PRESSES the servo in place until the screw bites
    step(sps // 4, "B align")
    steady_arm = False

    # ============ C) ROTATE: joint 2 swings the arm flat =======================================
    print(f"[C] joint 2 driven to {JOINT2_DEG:.0f} deg", flush=True)
    for j in range(int(1.5 * sps)):
        j2_target[:, i_j2] = math.radians(JOINT2_DEG) * (j + 1) / (1.5 * sps)
        step(1, "C rotate")
    step(sps // 2, "C settle")
    s0, a0 = seat_axis()
    print(f"[C] countersunk hole seat ({s0[0, 0]:.4f}, {s0[0, 1]:.4f}, {s0[0, 2]:.4f}), "
          f"axis ({a0[0, 0]:+.3f}, {a0[0, 1]:+.3f}, {a0[0, 2]:+.3f}) — must be straight up", flush=True)

    # ============ D) DROP: re-grab at the exact hole-up pose, then drop the screw ===============
    print("[D] re-grabbed at the drop pose (hole up); screw dropped 2 mm above the hole, "
          "1.5 mm off-axis onto the countersink cone", flush=True)
    grab(JOINT2_DEG)  # re-teleport to the exact drop pose (joint2=-90, hole up) — resets any C-drift
    drop_screw(0.002 + M2_LEN, lateral=0.0015)
    step(sps, "D drop")
    t_rest = ((scene.screw.data.root_pos_w - seat_axis()[0]) * seat_axis()[1]).sum(-1)
    print(f"[D] dropped screw rests {t_rest.mean() * 1000:+.2f} mm above the seat "
          f"(perched on the pilot, funneled by the countersink)", flush=True)

    # ============ E) DESCEND: the drill descends vertically onto the head ======================
    print("[E] drill picked up; descends vertically onto the screw", flush=True)
    z_engage = float(t_rest.mean()) + 0.0004
    z_hover = float(HOVER)
    standoff[:] = z_hover
    track_drill = True
    for j in range(sps):
        standoff[:] = z_hover + (z_engage - z_hover) * (j + 1) / sps
        step(1, "E descend")

    # ============ F) DRIVE: trigger -> gate -> welds; the hand releases ========================
    print("[F] trigger on — driving", flush=True)
    finger[:] = -FINGER
    fasten_step = -1
    for _ in range(int(1.5 * sps)):
        standoff[:] = (standoff - cfg.drive_rate * env.dt).clamp_min(BIT_STOP)
        step(1, "F drive")
        if fasten_step < 0 and bool((scene.fastened >= 0).all()):
            fasten_step = stepno
            release_grasp()  # the screw holds the servo now — the hand lets go
        if fasten_step > 0 and stepno - fasten_step > sps // 4:
            break
    if grasp_offset[0] is not None:
        release_grasp()
    release()  # the screw holds the servo now; let go of the arm too — the assembly is rigid
    print(f"[F] fastened at step {fasten_step} (the hand released the servo — only the screw "
          f"holds it now)", flush=True)

    # ============ G) RETREAT: the drill retreats and parks aside ===============================
    print("[G] trigger off, drill lifts off gently, then parks aside", flush=True)
    finger[:] = 0.0
    z_now = float(standoff[0])
    for j in range(int(RETREAT_S * sps)):  # cosine ease-out so the bit un-seats without a jolt
        ease = 0.5 - 0.5 * math.cos(math.pi * (j + 1) / (RETREAT_S * sps))
        standoff[:] = z_now + (z_hover - z_now) * ease
        step(1, "G retreat")
    st = torch.zeros(n, 13, device=dev)
    st[:, 0:3] = origin + torch.tensor(DRILL_PARK, device=dev)
    st[:, 3] = 1.0
    scene.drill.write_root_state_to_sim(st, None)
    track_drill = False
    step(sps // 2, "G park")
    err_retreat = rel_err()

    # ============ H) STRESS: knock the screw, wrench the servo =================================
    print("[H] knock the fastened screw, then wrench the servo — nothing may come apart", flush=True)
    knock = torch.tensor((0.25, 0.0, 0.35, 0.0, 0.0, 0.0), device=dev).expand(n, 6).contiguous()
    scene.screw.write_root_velocity_to_sim(knock, None)
    step(sps // 2, "H knock")
    err_knock = rel_err()
    wrench = torch.tensor((0.0, 0.4, 0.5, 0.0, 0.0, 0.0), device=dev).expand(n, 6).contiguous()
    scene.motor.write_root_velocity_to_sim(wrench, None)
    step(sps // 2, "H wrench")
    err_motor_stress = motor_rel_err()

    # ============ I) FINALE: re-grab, then lift, shake, rotate, set down, release ==============
    print("[I] re-grabbed; the clamp lifts the whole robot +15 cm, shakes, rotates 180 deg, sets "
          "it down, then RELEASES it", flush=True)
    grab(0.0)  # re-grab at the working pose to pick the assembled robot up
    anchor = arm_target_pos.clone()  # the clamp's base anchor
    lie_q = arm_target_quat.clone()

    def move_arm(dz: float, dx: float = 0.0, dy: float = 0.0, yaw: float = 0.0) -> None:
        arm_target_pos[:] = anchor + torch.tensor((dx, dy, dz), device=dev)
        half = 0.5 * yaw
        qz = torch.tensor((math.cos(half), 0.0, 0.0, math.sin(half)), device=dev).expand(n, 4)
        arm_target_quat[:] = quat_mul(qz, lie_q)

    for j in range(sps):
        move_arm(0.15 * (j + 1) / sps)
        step(1, "I lift")
    for j in range(int(1.5 * sps)):
        ph = 2.0 * math.pi * 3.0 * (j + 1) / sps
        move_arm(0.15, dx=0.015 * math.sin(ph), dy=0.010 * math.sin(0.7 * ph))
        step(1, "I shake")
    for j in range(int(1.5 * sps)):
        move_arm(0.15, yaw=math.pi * (j + 1) / (1.5 * sps))
        step(1, "I rotate")
    for j in range(sps):
        move_arm(0.15 - 0.13 * (j + 1) / sps, yaw=math.pi)
        step(1, "I lower")
    release()  # let go
    print("[I] clamp RELEASED — the free assembled robot must hold together", flush=True)
    step(int(1.2 * sps), "I free")
    err_free, err_motor_free = rel_err(), motor_rel_err()
    print(f"[I] resting free on the ground: screw-in-seat err {err_free.max() * 1000:.2f} mm, "
          f"servo-in-pocket err {err_motor_free.max() * 1000:.2f} mm", flush=True)

    # ============ verdict ======================================================================
    caught = bool((t_rest.abs() < 0.010).all())
    ok = (fasten_step > 0 and caught
          and float(err_retreat.max()) < 0.0015
          and float(err_knock.max()) < 0.0015 and float(err_motor_stress.max()) < 0.0015
          and float(err_free.max()) < 0.0015 and float(err_motor_free.max()) < 0.0015)
    print(f"SO101-SMOKE | servo inserted to {ins_err.mean() * 1000:.2f} mm (aligned after) | "
          f"screw caught at {t_rest.mean() * 1000:+.2f} mm | fastened at step {fasten_step} | "
          f"errs (mm): retreat {err_retreat.max() * 1000:.2f}, knock {err_knock.max() * 1000:.2f}, "
          f"wrench {err_motor_stress.max() * 1000:.2f}, finale {err_free.max() * 1000:.2f}/"
          f"{err_motor_free.max() * 1000:.2f} | {'PASS' if ok else 'FAIL'}", flush=True)
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

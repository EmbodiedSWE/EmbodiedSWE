"""Franka-arm solution smoke for MicrowaveMealScene — a REAL manipulation solve, RECORDED.

Where `microwave_meal_smoke.py` is the NullRobot oracle (drive tensors + teleports validating
the appliance rules), this drives the `articulated.microwave.franka.osc` binding through the
full two-cycle meal with the ARM ONLY — no teleports, no scene drive tensors:

  1. pull the door open by its handle bar (grasp, arc-pull following the hinge circle);
  2. pick a bowl off the counter by its rim (tilted pinch — a top-down carry cannot enter
     the cavity: the wrist would exceed the 0.32 m ceiling) and set it centred on the plate;
  3. push the door shut by its face until the latch catches;
  4. key the sampled program on the panel with the closed fingertips (TIME x requested,
     then START) and WAIT out the cycle, hands clear;
  5. reopen, retrieve the hot bowl to the serving mat, repeat for bowl two.

Waypoint control: each control step sends the OSC action `clamp((target-ee)/pos_scale)` +
the axis-angle orientation error over `rot_scale` — a saturating P-law in action units (the
controller latches targets at ~15 Hz and EMA-smooths, so sustained errors are the signal).
All world targets address the PINCH POINT (hand origin + 0.1034 m along hand z).

Between EVERY pair of subtasks the arm is re-homed (`rehome()`): back to the boot pinch
pose, dwelling until the MEASURED joint-space debt vs the boot configuration clears —
contact work twists joint 7 / coils the elbow, and a subtask launched from a tangled arm
reliably misses (runs 12-13, 28, 32). The debt is printed per rehome, so the log proves
each subtask started from a normal posture.

    python -m robobench.suites.articulated.smokes.microwave_meal_franka_smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--record_every", type=int, default=4)
parser.add_argument("--only_door", action="store_true",
                    help="debug clip: run ONLY the door-open skill, record every step, exit")
parser.add_argument("--probe_grasp", action="store_true",
                    help="debug clip: top-down rim weld + mouth orientation ladder, exit")
parser.add_argument("--out", type=str, default="microwave_franka_frames.npz")
parser.add_argument("--hdfs_dir", type=str, default="",
                    help="optional HDFS dir to upload the frames npz to ('' = no upload)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import json
import math
import os
import threading

import numpy as np
import torch

import robobench
from robobench.core import ENVS

FAILS: list[str] = []
PINCH = 0.1034  # hand origin -> finger-pad centre, along hand z (panda)
GRIP_OPEN, GRIP_SHUT = 0.04, 0.0
# Weld-contract close: a PARTIAL target (12 mm aperture) that stalls INSIDE every
# grip site's closure window by construction — the scene's weld-on-closure contract
# then locks the part to the hand (finger-pad contact with the parts is unreliable
# on this GPU stack; the contract replaces it, pc_motherboard-style).
GRIP_WELD = 0.006


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[mw-franka] {'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def quat_from_axes(z: torch.Tensor, x_hint: torch.Tensor) -> torch.Tensor:
    """wxyz quat whose local +z is `z` and local +x is `x_hint` projected orthogonal.
    (The pinch approach axis is hand +z; the finger gap spans hand +y.) Shepperd's
    largest-diagonal method — the fingers-down pose is a 180 deg rotation (trace -1),
    where the naive w-first formula degenerates (crashed franka run 1)."""
    z = z / z.norm()
    x = x_hint - (x_hint @ z) * z
    x = x / x.norm()
    y = torch.cross(z, x, dim=-1)
    m = [[float(v) for v in row] for row in torch.stack([x, y, z], dim=-1)]
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = (s / 4, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    elif m[0][0] >= m[1][1] and m[0][0] >= m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        q = ((m[2][1] - m[1][2]) / s, s / 4, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
    elif m[1][1] >= m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        q = ((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, s / 4, (m[1][2] + m[2][1]) / s)
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        q = ((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, s / 4)
    return torch.tensor(q, dtype=torch.float32)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("articulated.microwave.franka.osc")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    art = env.robot.articulation
    ee_i = art.find_bodies(env.robot.EE_BODY)[0][0]
    origin = env.iscene.env_origins[0]
    osc = env.robot.controller.controllers[0]
    pos_scale, rot_scale = osc.cfg.pos_scale, osc.cfg.rot_scale

    # --- recording ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = origin.detach().cpu().numpy().astype(float)
        o[2] += c.surface_z  # the work rides the table top
        env.sim.set_camera_view(tuple(np.array((1.30, -1.72, 1.15)) + o),
                                tuple(np.array((0.0, -0.10, 0.22)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        print(f"[mw-franka] camera ready shape={np.asarray(annot.get_data()).shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[mw-franka] camera setup FAILED ({exc!r})", flush=True)

    step_i = 0
    phases: list[tuple[int, str]] = []

    def phase(label: str) -> None:
        phases.append((step_i, label))
        print(f"[mw-franka] PHASE @{step_i}: {label}", flush=True)

    # Partial-footage saver: the frames buffer only hit disk at the END of a run,
    # so every crashed/killed run (34-38) left ZERO footage to debug from. Every
    # 500 recorded frames a daemon thread snapshots the buffer to *_part.npz
    # (atomic replace); the final save still writes args.out as before.
    _part_saving = threading.Event()
    _part_base = args.out[:-4] if args.out.endswith(".npz") else args.out

    def _save_partial(snapshot: list) -> None:
        try:
            arr = np.stack(snapshot, axis=0)
            tmp = _part_base + "_part_tmp.npz"
            np.savez_compressed(tmp, frames=arr, phases=json.dumps(phases))
            os.replace(tmp, _part_base + "_part.npz")
            print(f"[mw-franka] partial footage saved: {arr.shape[0]} frames -> "
                  f"{_part_base}_part.npz", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[mw-franka] partial footage save FAILED: {exc!r}", flush=True)
        finally:
            _part_saving.clear()

    # --- pinch-point state + the saturating P action --------------------------------------
    def hand_pose() -> tuple[torch.Tensor, torch.Tensor]:
        st = art.data.body_link_state_w[0, ee_i]
        return st[0:3].clone(), st[3:7].clone()

    def pinch_pos(hp: torch.Tensor, hq: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        off = torch.tensor([0.0, 0.0, PINCH], device=device)
        return hp + quat_apply(hq.unsqueeze(0), off.unsqueeze(0))[0]

    def step_action(dpos: torch.Tensor, drot: torch.Tensor, grip: float) -> None:
        nonlocal step_i
        a = torch.cat([dpos, drot, torch.full((2,), grip, device=device)]).unsqueeze(0)
        env.step(a, render=True)
        if annot is not None and step_i % args.record_every == 0:
            for _ in range(3):
                env.sim.render()
            arr = np.asarray(annot.get_data())
            if arr.size:
                frames.append(arr[..., :3].astype(np.uint8).copy())
                if len(frames) % 500 == 0 and not _part_saving.is_set():
                    _part_saving.set()
                    threading.Thread(target=_save_partial, args=(list(frames),),
                                     daemon=True).start()
        step_i += 1

    def goto(target_w: torch.Tensor, tquat: torch.Tensor | None, grip: float,
             tol: float = 0.012, max_steps: int = 220, settle: int = 0,
             rot_tol: float = 0.12, guard: bool = True) -> bool:
        """March the PINCH POINT to `target_w` (world) and the hand to `tquat`.
        ROTATION-FIRST: the OSC's rotation gains (30) are 3x weaker than position (100),
        so commanding both at once trades orientation away and can leave the hand at the
        right pinch point pointing the wrong way (franka runs 2-6: every grasp closed on
        air with position 'converged'). Align the hand while holding position, THEN
        translate; success requires BOTH residuals."""
        from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul

        tq = None if tquat is None else tquat.to(device)

        def errors() -> tuple[torch.Tensor, torch.Tensor, float]:
            hp, hq = hand_pose()
            perr = target_w.to(device) - pinch_pos(hp, hq)
            if tq is None:
                return perr, torch.zeros(3, device=device), 0.0
            q_t = torch.where((tq * hq).sum() >= 0, tq, -tq)
            aa = axis_angle_from_quat(quat_mul(q_t.unsqueeze(0),
                                               quat_conjugate(hq.unsqueeze(0))))[0]
            return perr, aa, float(aa.norm())

        used = 0
        if tq is not None:  # phase 1: orientation (position held softly), CAPPED —
            # a twisted wrist can be unrecoverable (run 33) and must not eat the budget
            for _ in range(min(60, max_steps)):
                perr, aa, rerr = errors()
                if rerr < rot_tol:
                    break
                step_action((perr / pos_scale).clamp(-0.3, 0.3),
                            (aa / rot_scale).clamp(-1.0, 1.0), grip)
                used += 1
        for _ in range(max(1, max_steps - used)):  # phase 2: position (+rot hold)
            perr, aa, rerr = errors()
            if float(perr.norm()) < tol:
                break
            # TWIST GOVERNOR: translating while the wrist is lost RATCHETS the twist
            # (run 35: rot_err 1.5 -> 3.0 rad across one carry, ending near the pi
            # flip where the shortest-path command chatters and cannot recover).
            # Pause translation until the wrist is back under the guard; the rot
            # channel gets the whole step meanwhile. Success stays position-only.
            # Guard at 1.2 rad: free transits swing the wrist 0.7-0.8 rad routinely
            # (position outruns the 3x-weaker rot channel — run 36 approach hovers
            # stalled 350+ mm short on a 0.7 guard); only a real tangle trips it.
            # `guard=False` for WELDED mat-carry legs: loaded flights sit pinned
            # at ~1.0-1.2 rad and the guard strangles them to a timeout (run 46:
            # every leg 30 cm short), while a twisted welded carry is harmless —
            # the drop target is recomputed from the live bowl offset (run 35
            # landed both drops at 3 rad).
            if tq is not None and guard and rerr > 1.2:
                step_action(torch.zeros(3, device=device),
                            (aa / rot_scale).clamp(-1.0, 1.0), grip)
                continue
            step_action((perr / pos_scale).clamp(-1.0, 1.0),
                        (aa / rot_scale).clamp(-1.0, 1.0), grip)
        for _ in range(settle):
            perr, aa, _ = errors()
            step_action((perr / pos_scale).clamp(-1.0, 1.0),
                        (aa / rot_scale).clamp(-1.0, 1.0), grip)
        perr, _aa, rerr = errors()
        resid = float(perr.norm())
        # Success is POSITION-ONLY. Orientation stays commanded (a bias the OSC works
        # toward) but is never gated on: a twisted joint-7 accumulates through contact
        # work and the rotation channel cannot undo 60-90 deg (run 33: descents ARRIVED
        # at 4.5 mm and were vetoed for rot_err ~0.8 — while every grasp surface here
        # (vertical bar band, rim circle) is twist-invariant about the approach axis).
        ok = resid < tol * 1.6
        if not ok:
            print(f"[mw-franka]   goto MISSED: resid={resid * 1000:.1f}mm rot_err={rerr:.3f}rad "
                  f"target={[round(float(v), 3) for v in target_w.tolist()]}", flush=True)
        return ok

    def hold(grip: float, k: int = 8) -> None:
        for _ in range(k):
            step_action(torch.zeros(3, device=device), torch.zeros(3, device=device), grip)

    # --- world anchors (model frame + mw_pos + env origin) ---------------------------------
    def W(x: float, y: float, z: float, *, mw: bool = True) -> torch.Tensor:
        cx, cy = (c.mw_pos if mw else (0.0, 0.0))
        return origin + torch.tensor([cx + x, cy + y, c.surface_z + z], device=device)

    # FRAME CONVENTION, measured from the vendored panda USD and mis-read twice (runs
    # 2-4 all closed on air): panda_finger_joint1/2 declare axis "X" with localRot0 =
    # (w=.7071, z=+/-.7071) — a +/-90 deg rotation ABOUT Z, so joint-x maps to hand-local
    # +/-Y: THE FINGER GAP SPANS HAND-LOCAL Y = z cross x. `x_hint` below is therefore
    # chosen so that (z cross x_hint) is the world direction the fingers must close along.
    ez = torch.tensor([0.0, 0.0, 1.0])
    ex = torch.tensor([1.0, 0.0, 0.0])
    Q_DOWN = quat_from_axes(-ez, ex)  # top-down (park/retreat only; gap irrelevant)
    # approach +y; local y = (0,1,0) x (0,0,1) = world x: closes ACROSS the vertical bar
    Q_FWD = quat_from_axes(torch.tensor([0.0, 1.0, 0.0]), ez)
    # Tilted carry: approach 30 deg below horizontal, pointing +y (into the cavity); the
    # wrist trails LOW behind the pinch, clearing the 0.32 m ceiling. local y =
    # (0,c,-s) x (1,0,0) = -(0,s,c): the oblique-radial axis straddling the -y rim wall
    # (a gap spans both signs).
    tilt = math.radians(30.0)
    Q_TILT = quat_from_axes(torch.tensor([0.0, math.cos(tilt), -math.sin(tilt)]), ex)
    # Top-down bar grasp: hand at its HOME-like orientation (no wrist gymnastics — the
    # forward-facing rolls sit near a wrist limit, runs 3-7), fingers descending AROUND
    # the vertical bar's upper section; gap = (-ez) x (ey) = +ex, across the bar.
    ey = torch.tensor([0.0, 1.0, 0.0])
    Q_TOPBAR = quat_from_axes(-ez, ey)
    grip_ids = env.robot.controller.controllers[-1].joint_ids

    def aperture() -> float:
        return float(art.data.joint_pos[0, grip_ids].sum())

    hinge_w = W(c.door_hinge[0], c.door_hinge[1], 0.0)[:2]
    BAR_LOCAL = torch.tensor([0.203, -0.301, 0.19])  # handle bar centre, DOOR body frame

    def door_ang() -> float:
        return float(scene.door_angle_deg()[0])

    def bar_now() -> torch.Tensor:
        """The handle bar's LIVE world position — the door swings when bumped, and a
        grasp at the spawn-time coordinates closes on air (debug clip 2026-08-05: an
        early approach shoved the door 37 deg open and every later grasp missed)."""
        from isaaclab.utils.math import quat_apply

        dp = scene.door.data.root_pos_w[0]
        dq = scene.door.data.root_quat_w[0]
        return dp + quat_apply(dq.unsqueeze(0), BAR_LOCAL.to(device).unsqueeze(0))[0]

    # --- skills ----------------------------------------------------------------------------
    def open_door(target_deg: float = 60.0) -> bool:
        # 60 (was 72): the weld-pull's release momentum let the door COAST to
        # 105-109 deg (runs 40/49), and a door that wide puts every closing push
        # point beyond the arm's reachable azimuth (run 49: 0.9 m misses, door
        # never closed). 60 clears the bowl easily (the task's own open
        # threshold is 50) and the coast stays closable.
        """Up to two full pull+push attempts: the pull's top-out angle varies across
        boots (51.4 deg thrice, then 24.4 deg in run 22 — contact nondeterminism), and
        a retry from a partially open door regains the rest."""
        for attempt in range(2):
            if _open_door_attempt(target_deg, attempt):
                return True
        print(f"[mw-franka]   door open FAILED after retries ({door_ang():.1f} deg)",
              flush=True)
        return False

    def _open_door_attempt(target_deg: float, attempt: int) -> bool:
        phase(f"open door to ~{target_deg:.0f} deg (weld-pull + fist-push, "
              f"attempt {attempt + 1})")
        # start from a VERIFIED clean home posture: approaches launched from a tangled
        # arm (post-carry configs) reliably miss and produce garbage mid-air welds
        # (runs 12-13, 28). LOW-FORWARD first — it un-folds a coiled elbow (run 32:
        # the arm wrapped over the appliance top and every descent missed) — then home.
        goto(W(0.30, -0.44, 0.16, mw=False), Q_HOME, GRIP_OPEN, tol=0.04, max_steps=120)
        for _ in range(2):
            if goto(W(0.05, -0.42, 0.45), Q_HOME, GRIP_OPEN, tol=0.05, max_steps=160):
                break
        # Descend on the LIVE bar pose (the door swings when bumped — a grasp at
        # spawn-time coordinates closes on air; debug clip 2026-08-05) at Q_HOME, the
        # one orientation the wrist tracks (probe run 25): the pads straddle the bar
        # diagonally and the weld contract catches the in-window stall. Two descents:
        # nominal, then 2 cm lower (a nudged door shifts the bar).
        variants = (
            {"q": Q_HOME, "pre": (0.0, 0.0, 0.215), "at": (0.0, 0.0, 0.125)},
            {"q": Q_HOME, "pre": (0.0, 0.0, 0.215), "at": (0.0, 0.0, 0.105)},
        )
        grasped = False
        q_grab = Q_HOME
        grasp_anchor = bar_now()
        grasp_z = float(grasp_anchor[2])
        home = W(0.05, -0.42, 0.45)
        for v in variants:
            anchor = bar_now()  # re-read: earlier attempts may have nudged the door
            pre = anchor + torch.tensor(v["pre"], device=device)
            at = anchor + torch.tensor(v["at"], device=device)
            ok_pre = goto(pre, v["q"], GRIP_OPEN, settle=4)
            ok_at = goto(at, v["q"], GRIP_OPEN, tol=0.008, settle=4)
            # A stall-brush during a missed descent can weld on its own — a weld is a
            # weld, the pull re-reads the live geometry. Otherwise only close FROM a
            # reached approach: closing from a miss produced garbage mid-air welds
            # with doomed pulls (run 28: welded at 22.9 mm, pull died at 1.6 deg).
            if ok_at and not bool(scene.grasp_held[0, 0]):
                # Close to the WELD aperture (12 mm, inside the door site's closure
                # window) while holding position at the bar: the fingers stall
                # in-window on the bar's grip band and the scene welds.
                for _ in range(24):
                    hp, hq = hand_pose()
                    perr = at - pinch_pos(hp, hq)
                    step_action((perr / pos_scale).clamp(-0.4, 0.4),
                                torch.zeros(3, device=device), GRIP_WELD)
                    if bool(scene.grasp_held[0, 0]):
                        break
            held = bool(scene.grasp_held[0, 0])
            ap = aperture() * 1000
            print(f"[mw-franka]   door grasp at={[round(float(x), 3) for x in (at - origin).tolist()]}: "
                  f"pre={ok_pre} reached={ok_at} welded={held} aperture={ap:.1f}mm "
                  f"door={door_ang():.1f}deg", flush=True)
            if held:
                grasped = True
                q_grab = v["q"]
                grasp_anchor = bar_now()
                grasp_z = float(at[2])
                break
            hold(GRIP_OPEN, 6)  # reopen, untangle to home, try the next variant
            goto(home, Q_HOME, GRIP_OPEN, tol=0.04, max_steps=140)
        if not grasped:
            print("[mw-franka]   no door grasp — aborting the pull", flush=True)
            return False
        from isaaclab.utils.math import quat_mul

        r_vec = grasp_anchor[:2] - hinge_w
        r = float(r_vec.norm())
        th0 = math.atan2(float(r_vec[1]), float(r_vec[0]))
        # Chase the MEASURED door angle: command the bar's arc point 12 deg ahead of
        # where the door actually is (a fixed ladder outruns the arm — run 16 aborted
        # at 31 deg with the weld still holding), yaw following the door. The weld
        # cannot be lost, so the only exits are target reached or progress stalled.
        d0 = door_ang()
        stall = 0
        last = d0
        for it in range(40):
            d = door_ang()
            if d >= target_deg:
                break
            # first waypoints lead harder: the latch detent (0.84 N*m breakout) needs
            # a sustained pull before the door moves at all; the lead TAPERS near
            # the target so the release momentum cannot coast the door 30+ deg
            # past it (runs 40/49)
            lead = min(d + (18.0 if it < 3 and d < 5.0 else 12.0), target_deg + 8.0)
            if d > target_deg - 12.0:
                lead = min(lead, target_deg + 3.0)
            th = th0 - math.radians(lead - d0)  # opening swings NEGATIVE about z
            tgt = torch.tensor([float(hinge_w[0]) + r * math.cos(th),
                                float(hinge_w[1]) + r * math.sin(th),
                                grasp_z], device=device)
            half = math.radians(-(d - d0)) / 2
            qz = torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)], device=device)
            tq = quat_mul(qz.unsqueeze(0), q_grab.unsqueeze(0))[0]
            goto(tgt, tq, GRIP_WELD, tol=0.02, max_steps=30)
            if not bool(scene.grasp_held[0, 0]):
                print(f"[mw-franka]   weld lost at door={door_ang():.1f} deg", flush=True)
                break
            stall = stall + 1 if door_ang() < last + 0.5 else 0
            last = door_ang()
            if stall >= 6:
                print(f"[mw-franka]   pull stalled at door={last:.1f} deg", flush=True)
                break
        hold(GRIP_OPEN, 10)  # release the bar (the weld cuts on the wide-open aperture)
        # retreat radially outward from the door edge
        hp, hq = hand_pose()
        pp = pinch_pos(hp, hq)
        out = pp + torch.tensor([0.10, -0.14, 0.05], device=device)
        goto(out, Q_HOME, GRIP_OPEN, tol=0.03, max_steps=90)
        # FIST-PUSH FINISH: the single weld-pull tops out ~51 deg (the forearm ends
        # draped across the appliance at full span — run 19 footage). Push the slab's
        # INNER face tangentially with the closed fist: position-only (no yaw tracking,
        # the OSC's strong channel) and hand<->door contact is the proven pair.
        # Only meaningful on a PARTIALLY OPEN door: from ~closed there is no wedge and
        # the staging points lie INSIDE the appliance (run 28 burned ~500 steps
        # commanding them); a near-closed door needs another PULL, not a push.
        if 15.0 < door_ang() < target_deg - 4.0:
            phase("fist-push the door the rest of the way")
            push_z = float(origin[2]) + c.surface_z + 0.20
            th_closed = math.radians(-4.9)  # slab azimuth from the hinge at door=0
            # zero rotation demand: whatever orientation the arm has after the retreat
            # is the push orientation (fingers-down at 0.6 m left is unconvergeable —
            # rot_err 1.16 rad, run 20; a fist push cares only about position)
            q_push = hand_pose()[1].clone()
            z_over = float(origin[2]) + c.surface_z + c.outer[2] + 0.06  # above the slab top
            for it in range(10):
                d = door_ang()
                if d >= target_deg:
                    break
                ph_now = th_closed - math.radians(d)
                ph_lead = th_closed - math.radians(min(d + 16.0, target_deg + 6.0))
                u_now = torch.tensor([math.cos(ph_now), math.sin(ph_now), 0.0], device=device)
                u_lead = torch.tensor([math.cos(ph_lead), math.sin(ph_lead), 0.0], device=device)
                n_in = torch.tensor([-math.sin(ph_now), math.cos(ph_now), 0.0], device=device)
                hw3 = torch.tensor([float(hinge_w[0]), float(hinge_w[1]), push_z], device=device)
                stage_p = hw3 + 0.38 * u_now + 0.07 * n_in  # inner side, off the face
                press_p = hw3 + 0.38 * u_lead - 0.02 * n_in  # through the face, 16 deg ahead
                if it == 0:
                    # ENTER THE WEDGE FROM ABOVE: a straight transit to the staging
                    # point crosses the slab when the wedge is narrow — at door 24 deg
                    # the fist pushed the door SHUT on its way in (run 22)
                    over = stage_p.clone()
                    over[2] = z_over
                    goto(over, q_push, GRIP_SHUT, tol=0.03, max_steps=60)
                goto(stage_p, q_push, GRIP_SHUT, tol=0.025, max_steps=40)
                goto(press_p, q_push, GRIP_SHUT, tol=0.02, max_steps=25)
            hp, hq = hand_pose()
            goto(pinch_pos(hp, hq) + torch.tensor([0.12, -0.10, 0.08], device=device),
                 q_push, GRIP_OPEN, tol=0.03, max_steps=90)
        ok = door_ang() >= c.door_open_deg
        print(f"[mw-franka]   door now {door_ang():.1f} deg", flush=True)
        return ok

    def bowl_pos(b: int) -> torch.Tensor:
        return scene.bowls[b].data.root_pos_w[0].clone()

    # Rim-straddle at Q_HOME everywhere. Probe run 25 measured the controller's law:
    # Q_HOME tracks to 0.001 rad even far-left at the mouth, while EVERY yawed/tilted
    # variant (Q_NAT 0.53, Q_FWD 0.60, Q_TILT 2.10 rad) fails — the wrist holds exactly
    # one orientation. Position converges everywhere except low+far stretches, which
    # the near bowl slots avoid. The straddle azimuth is MEASURED from the live pad
    # positions at boot (run 26 closed on air at an ASSUMED 52.5-deg gap diagonal);
    # RIM_DIR/RIM_R are filled in by measure_gap_axis() in the solve prologue.
    RIM_R = c.bowl_outer_r - 0.004
    RIM_DIR = torch.tensor([1.0, 0.0, 0.0], device=device)  # overwritten at boot

    def measure_gap_axis(toward=None) -> None:
        """Set RIM_DIR from the MEASURED finger-gap axis at the home pose: the rim
        wall must lie ACROSS the pads, i.e. the straddle point sits along the gap
        axis from the bowl centre. Of the two ends, take the one along `toward`
        (callers pass base-minus-bowl: the BASE-PROXIMAL rim — the reachability
        intent, resolved properly). The old '-y end' rule was a coin flip for a
        gap lying east-west (the deciding y-component was +/-0.002 of noise), and
        a west-rim flip put the cavity entry line at the far-west azimuth where
        joint 1 saturates (run 49: 455 mm miss, bowl released at the mouth).
        `toward=None` keeps the -y rule (boot-time call, no bowl in play)."""
        pl, pr = pads()
        g = (pl - pr)[:2]
        g = g / g.norm().clamp_min(1e-9)
        if toward is None:
            if float(g[1]) > 0.0:
                g = -g
        elif float((g * toward.to(g.device)).sum()) < 0.0:
            g = -g
        RIM_DIR[0], RIM_DIR[1] = float(g[0]), float(g[1])
        print(f"[mw-franka] gap axis measured: dir=({float(g[0]):+.3f},"
              f"{float(g[1]):+.3f}) az={math.degrees(math.atan2(float(g[1]), float(g[0]))):.1f}deg",
              flush=True)

    def rim_point(b: int) -> torch.Tensor:
        """The straddle point on bowl b's rim (world): top of the wall along the
        measured gap axis — any azimuth is legal for the circle grip site."""
        return bowl_pos(b) + RIM_R * RIM_DIR + torch.tensor(
            [0.0, 0.0, c.bowl_h / 2 - 0.008], device=device)

    def weld_close(site: int, steps: int = 24) -> bool:
        """Close to the weld aperture while HOLDING the entry pose in BOTH channels:
        hold()'s zero-action steps re-latch whatever pose the hand drifts to, so
        rim-contact torques ratchet the wrist freely during the close (run 36:
        bowl 1 spun to 2.0 rad between weld and lift and the lift stalled)."""
        from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul

        hp0, hq0 = hand_pose()
        p0, q0 = pinch_pos(hp0, hq0).clone(), hq0.clone()
        for _ in range(steps):
            hp, hq = hand_pose()
            perr = p0 - pinch_pos(hp, hq)
            q_t = torch.where((q0 * hq).sum() >= 0, q0, -q0)
            aa = axis_angle_from_quat(quat_mul(q_t.unsqueeze(0),
                                               quat_conjugate(hq.unsqueeze(0))))[0]
            step_action((perr / pos_scale).clamp(-0.4, 0.4),
                        (aa / rot_scale).clamp(-1.0, 1.0), GRIP_WELD)
            if bool(scene.grasp_held[0, site]):
                return True
        return False

    def grasp_bowl_rim(b: int, toward=None, grasp_q=None) -> bool:
        """TOP-DOWN rim straddle at Q_HOME, held via the weld contract (site 1 + b).
        The gap axis is RE-MEASURED live first: joint-7 twist accumulates through
        contact work and rotates the pads — the straddle azimuth must follow.
        `toward` picks WHICH rim end (default: the bowl's base-proximal side);
        the CAVITY trip passes base-minus-axis instead — the pinch rides the
        grasped rim through the mouth, so a west-rim grasp forces the west entry
        line at the far-west azimuth where joint 1 saturates (runs 49/50: 0.4 to
        0.5 m entry misses, bowl released half a metre short).
        The approach TRANSITS HIGH then descends vertically (user route,
        2026-08-06): the old direct low transit from the boot pose is the longest
        leg in the task (~0.8 m) and timed out ~35 cm short on EVERY run; bowl 0's
        descent absorbed the shortfall (10 mm — inside the 22 mm weld engage),
        bowl 1's landed ~50 mm off the rim and closed on air (runs 36/38/42)."""
        if toward is None:
            toward = BASE_XY - bowl_pos(b)[:2]
        if grasp_q is None:
            grasp_q = Q_HOME
        measure_gap_axis(toward=toward)
        bp = bowl_pos(b)
        rim = rim_point(b)
        # LOW DIRECT approach — the proven route (10 straight bowl-0 grasps
        # before the high-route experiments). Bowl targets sit at azimuth ~180
        # deg from the base, within 9-17 deg of joint-1's hard stop: westward
        # flights AT ALTITUDE approach the envelope edge and wrap the wrist
        # 2.3-2.9 rad (runs 43/47/48, with and without obstacles nearby — the
        # high route belongs to the short EASTWARD mat reaches only). The old
        # route's one defect was budget: the hover leg timed out ~35 cm short
        # (every run) and bowl 1's descent could not absorb it — so the hover
        # gets max_steps 400 (was 220) and the descent 300.
        # a NON-home grasp orientation (the tilted cavity pinch) needs the
        # null-space pull out of the way: run 25 measured Q_TILT tracking fail
        # at 2.10 rad under the full home-posture pull — approach with the pull
        # weakened and re-targeted to the live configuration
        kp_g, tgt_g = osc.cfg.kp_null, osc._q_default.clone()
        if grasp_q is not Q_HOME:
            osc.cfg.kp_null = 2.0
            osc._q_default = art.data.joint_pos[0, ARM_IDS].clone()
        try:
            # fly the approach at Q_HOME — the ONLY orientation that survives
            # westward flights (8/8 staging grasps; the tilted approach wound
            # 1.5 rad in flight and missed by half a metre, run 57) — then
            # rotate to the grasp orientation IN PLACE at the hover
            goto(rim + torch.tensor([0.0, 0.0, 0.14], device=device), Q_HOME,
                 GRIP_OPEN, tol=0.015, settle=2, max_steps=400)
            if grasp_q is not Q_HOME:
                hp_t = pinch_pos(*hand_pose()).clone()
                goto(hp_t, grasp_q, GRIP_OPEN, tol=0.02, settle=6, max_steps=140)
            # the approach may have untwisted the wrist a little
            measure_gap_axis(toward=toward)
            rim = rim_point(b)
            goto(rim, grasp_q, GRIP_OPEN, tol=0.006, settle=6, max_steps=300)
            weld_close(1 + b)
            if not bool(scene.grasp_held[0, 1 + b]):
                # PRESS-TO-ENGAGE retry: an oblique (tilted) descent stalls on
                # the rim's flared lip just outside the 22 mm weld envelope
                # (run 54: 17 mm short, closed on air) — reopen, advance 12 mm
                # along the hand's own approach axis, close again
                from isaaclab.utils.math import quat_apply
                hold(GRIP_OPEN, 6)
                hp_p, hq_p = hand_pose()
                press = pinch_pos(hp_p, hq_p) + quat_apply(
                    hq_p.unsqueeze(0),
                    torch.tensor([[0.0, 0.0, 0.012]], device=device))[0]
                goto(press, grasp_q, GRIP_OPEN, tol=0.005, settle=4, max_steps=120)
                weld_close(1 + b)
        finally:
            osc.cfg.kp_null = kp_g
            osc._q_default = tgt_g
        # Lift with the wrist AS IS: fighting a contact-set twist against a
        # counter-borne bowl stalls the lift on the governor (run 36: dz 0.7 cm at
        # rot_err 2.0); the wrist is restored mid-air once the bowl is clear.
        q_lift = hand_pose()[1].clone()
        lifted = rim + torch.tensor([0.0, 0.0, 0.08], device=device)
        goto(lifted, q_lift, GRIP_WELD, tol=0.012, settle=2)
        dz = float(bowl_pos(b)[2] - bp[2])
        print(f"[mw-franka]   bowl{b} lift dz={dz * 100:.1f}cm "
              f"welded={bool(scene.grasp_held[0, 1 + b])} "
              f"aperture={aperture() * 1000:.1f}mm (rim=8.5)", flush=True)
        return dz > 0.03

    # Cavity geometry for the Q_HOME line entry (all measured, run 25): pinch rides at
    # plate-top + bowl_h + 2 mm; the wrist column above it tops out ~0.145 m over the
    # pinch — under the 0.345 m cavity ceiling with ~4 cm to spare.
    def mouth_line_z(carry: bool) -> float:
        base = c.surface_z + float(origin[2]) + c.tt_top_z + c.bowl_h
        return base + (0.002 if carry else -0.008)

    MOUTH_Y = float(W(0.0, -0.36, 0.0)[1])

    def place_bowl_on_plate(b: int) -> bool:
        """Carry the welded bowl through the mouth on ONE constant-height straight
        line (+y) to the plate axis, open (8 mm drop), straight back out."""
        phase(f"place bowl {b} on the plate")
        ax = W(c.tt_off_x, c.tt_off_y, 0.0)
        # STAGE-THEN-INSERT (runs 50-56: the long westward loaded carry arrives
        # at the mouth with the wrist wound 1.2-2.7 rad under EVERY guard and
        # posture combination, and a twisted arrival kills the entry). The long
        # carry and the precise insert have incompatible needs, so they are
        # SEPARATE: (1) plain grasp + sloppy carry to a REST SPOT in front of
        # the mouth (the live-offset drop lands at any twist — run 35 proved
        # 3 rad), (2) release + the bulletproof bare-arm rehome (<=14 deg every
        # run), (3) TILTED re-grasp at the rest spot and a 25 cm fresh-wrist
        # insert — the original design's entry, finally started clean.
        if not grasp_bowl_rim(b):
            check(f"bowl{b}-grasp", False, "lift failed")
            return False
        drop_r = bowl_drop_pinch(0.0, -0.30, b)
        goto(torch.tensor([float(drop_r[0]), float(drop_r[1]),
                           float(drop_r[2]) + 0.10], device=device), Q_HOME,
             GRIP_WELD, tol=0.02, settle=2, max_steps=300, guard=False)
        drop_r = bowl_drop_pinch(0.0, -0.30, b)
        goto(drop_r, Q_HOME, GRIP_WELD, tol=0.010, settle=6, max_steps=200,
             guard=False)
        hold(GRIP_OPEN, 10)
        hp_e, hq_e = hand_pose()
        goto(pinch_pos(hp_e, hq_e) + torch.tensor([0.10, -0.10, 0.10], device=device),
             None, GRIP_OPEN, tol=0.03, max_steps=100)
        rehome(f"pre-insert{b}")
        # TILTED SOUTH-RIM grasp at the rest spot — wrist-LOW under the lintel,
        # pinch trailing the bowl on the centreline (the file's original design)
        if not grasp_bowl_rim(b, toward=torch.tensor([0.0, -1.0], device=device),
                              grasp_q=Q_TILT.to(device)):
            check(f"bowl{b}-grasp", False, "insert re-grasp failed")
            return False
        q_carry = hand_pose()[1].clone()  # insert AS GRASPED (tilted)
        z_in = mouth_line_z(carry=True)
        px, py = float(ax[0] + RIM_R * RIM_DIR[0]), float(ax[1] + RIM_R * RIM_DIR[1])
        # SPLIT the mat->mouth carry with a close-in south rest: the single 0.56 m
        # westward loaded sweep crosses the joint-1-saturation sector and flips
        # the wrist mid-flight (run 51: 2.4 rad, 419 mm miss) — two ~0.3 m legs
        # with settles let the rotation channel recover at the pause
        # WEAK HOLD-CURRENT POSTURE for the whole carry+entry: the home-posture
        # pull (a) drags the elbow toward the box during deep entries (run 52,
        # on camera: forearm laid on the cabinet edge, 21 cm stall) and (b)
        # winds the tilted wrist away during the westward carry (run 55: 0.17
        # rad at grasp -> 1.73 at the mouth, entry dead at the face plane).
        # Carry legs run UNGUARDED (welded, open counter); the tilt is restored
        # STATIONARY at the staging point before the guarded entry.
        q_saved, kp_saved = osc._q_default.clone(), osc.cfg.kp_null
        osc._q_default = art.data.joint_pos[0, ARM_IDS].clone()
        osc.cfg.kp_null = 2.0
        try:
            # the insert starts at the rest spot 10 cm south of the mouth — no
            # long carry, no mid-rest; lift to the mouth line and go straight in
            goto(torch.tensor([px, MOUTH_Y, z_in], device=device), q_carry,
                 GRIP_WELD, tol=0.012, settle=8, max_steps=200)
            ok_in = goto(torch.tensor([px, py, z_in], device=device), q_carry,
                         GRIP_WELD, tol=0.010, settle=6, max_steps=280)
        finally:
            osc._q_default = q_saved
            osc.cfg.kp_null = kp_saved
        hold(GRIP_OPEN, 10)
        goto(torch.tensor([px, MOUTH_Y, z_in + 0.01], device=device), q_carry,
             GRIP_OPEN, tol=0.02, max_steps=200)
        centred = bool(scene.bowl_centred()[0, b])
        off = float(scene.bowl_offsets()[0, b]) * 100
        print(f"[mw-franka]   bowl{b} centred={centred} offset={off:.1f}cm", flush=True)
        return centred

    def bowl_drop_pinch(slot_x: float, slot_y: float, b: int) -> torch.Tensor:
        """Pinch target that sets the CARRIED bowl's centre onto (slot_x, slot_y),
        from the LIVE welded offset (bowl centre minus pinch, measured now) — no
        grasp-time assumption survives a wrist that twists mid-carry."""
        off = bowl_pos(b)[:2] - pinch_pos(*hand_pose())[:2]
        tgt = W(slot_x, slot_y, 0.0, mw=False)
        return torch.tensor([float(tgt[0] - off[0]), float(tgt[1] - off[1]),
                             float(origin[2]) + c.surface_z + c.bowl_h - 0.004],
                            device=device)

    def stage_bowl(b: int, slot_x: float) -> bool:
        """Move bowl b from its spawn slot (inside the door's swing arc — the task
        hazard) to a mat slot on the right, BEFORE the door ever opens."""
        phase(f"stage bowl {b} out of the door arc")
        if not grasp_bowl_rim(b):
            check(f"stage{b}-grasp", False, "lift failed")
            return False
        restore_wrist(GRIP_WELD)  # un-twist mid-air over the spawn slot, pre-carry
        # HIGH ROUTE to the mat (user route, 2026-08-06): rise over the pickup,
        # one high diagonal, descend at the drop. Replaces the southern corridor,
        # whose long LOW loaded sweeps were where the twist re-accumulated
        # (measured 27 cm misses at the guard, run 40). At Z_HI the hanging
        # bowl's bottom clears the appliance top by 10 cm and the arm's own
        # shoulder tower by 12 cm; the farthest point is 0.40 m of reach.
        # guard=False on every welded leg (run 46: the guard strangled the loaded
        # flight to a timeout); the drop target is recomputed AFTER the flight so
        # whatever twist the carry accumulated is measured into it, not fought.
        # NO mid-carry restore_wrist (run 38: a 2.0 rad restore at carry height
        # swept the welded bowl into the counter).
        drop0 = bowl_drop_pinch(slot_x, c.mat_pos[1], b)
        hp_now = pinch_pos(*hand_pose())
        goto(torch.tensor([float(hp_now[0]), float(hp_now[1]), Z_HI], device=device),
             Q_HOME, GRIP_WELD, tol=0.03, settle=2, max_steps=160, guard=False)
        goto(torch.tensor([float(drop0[0]), float(drop0[1]), Z_HI], device=device),
             Q_HOME, GRIP_WELD, tol=0.02, settle=2, max_steps=300, guard=False)
        drop = bowl_drop_pinch(slot_x, c.mat_pos[1], b)  # LIVE offset, post-flight
        goto(drop + torch.tensor([0.0, 0.0, 0.10], device=device), Q_HOME,
             GRIP_WELD, tol=0.015, settle=2, max_steps=160, guard=False)
        ok_drop = goto(drop, Q_HOME, GRIP_WELD, tol=0.008, settle=6, max_steps=200,
                       guard=False)
        if not ok_drop:  # one fresh try from straight above (a long carry can tangle)
            drop = bowl_drop_pinch(slot_x, c.mat_pos[1], b)
            goto(drop + torch.tensor([0.0, 0.0, 0.08], device=device), Q_HOME,
                 GRIP_WELD, tol=0.015, settle=2, max_steps=120, guard=False)
            goto(drop, Q_HOME, GRIP_WELD, tol=0.008, settle=6, max_steps=160,
                 guard=False)
        hold(GRIP_OPEN, 10)
        # UNCOIL low-forward: the close-in mat drop folds the elbow up, and a straight
        # pull to a high home point drags it over the appliance top into a coil that
        # every later approach inherits (run 32 footage — the arm ended wrapped above
        # the microwave and all four door descents missed from the coil)
        goto(W(0.30, -0.44, 0.16, mw=False), Q_HOME, GRIP_OPEN, tol=0.04, max_steps=120)
        staged = bool(scene.bowl_on_mat()[0, b])
        print(f"[mw-franka]   bowl{b} staged on mat: {staged}", flush=True)
        return staged

    def close_door() -> bool:
        phase("push the door shut")
        # push against the door's OUTER face NEAR THE HINGE, following the arc inward.
        # r=0.26 (was 0.42): the free-edge push points on an open door live in the
        # far-south-west azimuth where joint 1 saturates (run 49: 0.6-0.9 m misses
        # at every arc point, door never moved) — the near-hinge band stays inside
        # the proven reachable azimuth (the bowl-0 grasp zone). Shorter lever means
        # more force per torque, which only matters at the latch re-engage, where
        # the detent assists.
        # th0 = the arc angle of the push radius at door CLOSED, recovered from the LIVE
        # bar direction rotated back by the live door angle (opening swings negative).
        r = 0.26
        bnow = bar_now()
        th0 = math.atan2(float(bnow[1]) - float(hinge_w[1]),
                         float(bnow[0]) - float(hinge_w[0])) + math.radians(door_ang())
        start = max(door_ang() - 4.0, 8.0)
        push_z = float(origin[2]) + c.surface_z + 0.20
        for deg in [start] + list(np.arange(start - 8, -6, -8.0)):
            th = th0 - math.radians(max(float(deg), 0.0))
            tgt = torch.tensor([float(hinge_w[0]) + (r + 0.05) * math.cos(th),
                                float(hinge_w[1]) + (r + 0.05) * math.sin(th), push_z],
                               device=device)
            goto(tgt, Q_HOME, GRIP_SHUT, tol=0.02, max_steps=50)
        for _ in range(90):  # settle onto the latch
            if bool(scene.door_closed()[0]):
                break
            hold(GRIP_SHUT, 1)
        hp, hq = hand_pose()
        goto(pinch_pos(hp, hq) + torch.tensor([0.0, -0.12, 0.06], device=device), Q_HOME,
             GRIP_SHUT, tol=0.03, max_steps=80)
        print(f"[mw-franka]   door {door_ang():.1f} deg closed={bool(scene.door_closed()[0])}",
              flush=True)
        return bool(scene.door_closed()[0])

    def press_key(k: int) -> bool:
        """Poke key k (0=TIME, 1=START) with one pad tip at full open, hand at Q_HOME
        (the only orientation the wrist tracks). The pinch sits one gap-half-width
        along the MEASURED gap axis (RIM_DIR points to the -y half), so the pad on the
        +y side lands on the key while the hand-base column stays ~3 cm in FRONT of
        the panel (keys sit flush with the solid panel collider — a centred fist would
        jam its base on the panel before any key travel)."""
        boff = c.btn_time_off if k == 0 else c.btn_start_off
        face = W(boff[0], -0.2894, boff[2])
        pad = 0.04 * RIM_DIR
        pre = face + pad + torch.tensor([0.0, -0.045, 0.0], device=device)
        goto(pre, Q_HOME, GRIP_OPEN, tol=0.006, settle=6)
        pressed = False
        tgt = face + pad + torch.tensor([0.0, 0.0075, 0.0], device=device)
        for _ in range(90):
            hp, hq = hand_pose()
            err = tgt - pinch_pos(hp, hq)
            step_action((err / pos_scale).clamp(-1.0, 1.0), torch.zeros(3, device=device),
                        GRIP_OPEN)
            if bool(scene._btn_pressed[0, k]):
                pressed = True
                break
        goto(pre, Q_HOME, GRIP_OPEN, tol=0.01, max_steps=60)  # release (re-arm)
        for _ in range(30):
            if float(scene.button_depth()[0, k]) < 0.0005:
                break
            hold(GRIP_OPEN, 1)
        return pressed

    def key_program(requested: int) -> bool:
        phase(f"key program: TIME x{requested} + START")
        ok = True
        for i in range(requested):
            got = press_key(0)
            print(f"[mw-franka]   TIME press {i + 1}/{requested}: {got} "
                  f"entry={int(scene._entry[0])}", flush=True)
            ok &= got
        got = press_key(1)
        print(f"[mw-franka]   START press: {got} running={bool(scene._running[0])}", flush=True)
        return ok and got

    def wait_cycle() -> bool:
        phase("WAIT out the cycle (hands clear)")
        park = W(0.10, -0.30, 0.40, mw=False)  # forward-left of the base, clear of everything
        goto(park, Q_HOME, GRIP_OPEN, tol=0.03, max_steps=120)
        spun = False
        for _ in range(int(scene._timer[0]) // env.robot.control_period + 220):
            if not bool(scene._running[0]):
                break
            hold(GRIP_OPEN, 1)
            spun = spun or abs(float(scene.turntable_rate_dps()[0])) > 0.5 * c.spin_rate_dps
        return spun and not bool(scene._running[0])

    def serve_bowl(b: int, slot_x: float) -> bool:
        """Retrieve the (heated) bowl from the plate on the same constant-height line:
        enter 26 mm above the rim (pads clear the near wall), descend onto the rim
        straddle, weld, lift 20 mm, straight back out, carry to the mat slot."""
        phase(f"retrieve bowl {b} to the mat")
        bp = bowl_pos(b)
        rim = rim_point(b)
        z_entry = float(rim[2]) + 0.026
        goto(torch.tensor([float(rim[0]), MOUTH_Y, z_entry], device=device), Q_HOME,
             GRIP_OPEN, tol=0.012, settle=2)
        goto(torch.tensor([float(rim[0]), float(rim[1]), z_entry], device=device),
             Q_HOME, GRIP_OPEN, tol=0.008, settle=6, max_steps=280)
        goto(rim, Q_HOME, GRIP_OPEN, tol=0.006, settle=6)
        weld_close(1 + b)
        held = bool(scene.grasp_held[0, 1 + b])
        z_out = float(rim[2]) + 0.020
        goto(torch.tensor([float(rim[0]), float(rim[1]), z_out], device=device),
             Q_HOME, GRIP_WELD, tol=0.010, settle=2)
        goto(torch.tensor([float(rim[0]), MOUTH_Y, z_out], device=device), Q_HOME,
             GRIP_WELD, tol=0.015, max_steps=280)
        restore_wrist(GRIP_WELD)  # un-twist outside the mouth, before the mat carry
        # same HIGH ROUTE as stage_bowl (user route, 2026-08-06): rise outside the
        # mouth, one high diagonal (the open door's slab swings far LEFT — the
        # right-side crossing stays clear), descend at the drop; guard=False on
        # welded legs and the drop recomputed post-flight (see stage_bowl)
        drop0 = bowl_drop_pinch(slot_x, c.mat_pos[1], b)
        hp_now = pinch_pos(*hand_pose())
        goto(torch.tensor([float(hp_now[0]), float(hp_now[1]), Z_HI], device=device),
             Q_HOME, GRIP_WELD, tol=0.03, settle=2, max_steps=160, guard=False)
        goto(torch.tensor([float(drop0[0]), float(drop0[1]), Z_HI], device=device),
             Q_HOME, GRIP_WELD, tol=0.02, settle=2, max_steps=300, guard=False)
        drop = bowl_drop_pinch(slot_x, c.mat_pos[1], b)  # LIVE offset, post-flight
        goto(drop + torch.tensor([0.0, 0.0, 0.10], device=device), Q_HOME, GRIP_WELD,
             tol=0.02, settle=2, max_steps=160, guard=False)
        goto(drop, Q_HOME, GRIP_WELD, tol=0.008, settle=6, guard=False)
        hold(GRIP_OPEN, 10)
        hp, hq = hand_pose()
        goto(pinch_pos(hp, hq) + torch.tensor([0.0, -0.10, 0.12], device=device), Q_HOME,
             GRIP_OPEN, tol=0.03, max_steps=80)
        served = bool((scene.bowl_on_mat() & scene.bowls_settled())[0, b])
        print(f"[mw-franka]   bowl{b} welded_in_cavity={held} "
              f"on_mat={bool(scene.bowl_on_mat()[0, b])}", flush=True)
        return served

    # --- the solve ---------------------------------------------------------------------------
    env.reset()
    hold(GRIP_OPEN, 20)
    # The hand's HOME orientation (top-down, gap on the ready pose's 45-deg diagonal):
    # grasps commanded AT this orientation need zero rotation — the OSC's rotation
    # channel converges too slowly for 45-deg yaws (rot_err 0.3-0.8 rad across runs
    # 5-13, hands arriving visibly tilted), so the bar is straddled DIAGONALLY across
    # its 34x38 square section (stall ~50 mm) instead of fighting for a square-on gap.
    _hp0, _hq0 = hand_pose()
    Q_HOME = _hq0.clone()
    # anchors for the azimuth-adaptive natural orientation (see q_nat above)
    BASE_XY = art.data.root_pos_w[0, :2].clone()
    _hp_boot = pinch_pos(_hp0, _hq0)
    HOME_AZ = math.atan2(float(_hp_boot[1] - BASE_XY[1]), float(_hp_boot[0] - BASE_XY[0]))
    fid = [art.body_names.index(nm) for nm in ("panda_leftfinger", "panda_rightfinger")]
    # Joint-space home reference for the between-subtask re-home (user directive
    # 2026-08-05: restore the arm to a normal angle after each subtask so the twist
    # does not compound). Captured at the settled boot pose = the OSC's null-space
    # posture target.
    ARM_IDS = list(osc.joint_ids)
    Q_ARM_HOME = art.data.joint_pos[0, ARM_IDS].clone()
    P_BOOT = _hp_boot.clone()
    # Transit altitude for the HIGH ROUTE (user route, 2026-08-06): all long
    # transits fly at Z_HI and end in short vertical descents — appliance top
    # + 0.17 m, so a rim-carried bowl (hangs ~0.07 m below the pinch) clears the
    # box by 10 cm and the arm's own shoulder tower by 12 cm.
    Z_HI = float(origin[2]) + c.surface_z + c.outer[2] + 0.17
    # (A separate launch-safe PARK pose was tried on runs 44-45 and REVERTED:
    # parking over the base poisoned cross-body launches worse than the boot
    # pose's box proximity — the rise-first legs in every approach are the fix.)
    # articulation order != joint-number order is possible; print the actual mapping
    # once so the per-joint debt labels are trustworthy (run 39 printed a debt that
    # looked impossible for its labeled joint)
    print(f"[mw-franka] arm joint order: "
          f"{[art.joint_names[j] for j in ARM_IDS]} home="
          f"{[round(float(v), 3) for v in Q_ARM_HOME]}", flush=True)

    def arm_debt_vec() -> torch.Tensor:
        """Per-joint (boot pose − arm joint), UNWRAPPED: panda joints are
        limited-range, so the plain difference IS the true error. (A [-pi,pi]
        wrap mirrored >180 deg tangles into wrong small/opposite values and hid
        the reversed-pull controller bug found in run 38.)"""
        return Q_ARM_HOME - art.data.joint_pos[0, ARM_IDS]

    def arm_debt() -> float:
        return float(arm_debt_vec().abs().max())

    REHOME_TOL = 0.25  # rad (~14 deg) max joint debt that counts as "arm is normal again"

    def restore_wrist(grip: float, max_leg: float = 0.8) -> None:
        """Rotate the hand back to Q_HOME IN PLACE (position held at the live
        pinch) — call it only in OPEN AIR: welded cargo swings with the hand.
        A near-pi error is walked back in <=max_leg arc legs because the
        one-shot shortest-path command is ill-conditioned there (the axis flips
        sign step to step — run 35's rehome made the debt worse with it)."""
        from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul

        hq0 = hand_pose()[1].clone()
        q_t = torch.where((Q_HOME * hq0).sum() >= 0, Q_HOME, -Q_HOME)
        aa_tot = axis_angle_from_quat(quat_mul(q_t.unsqueeze(0),
                                               quat_conjugate(hq0.unsqueeze(0))))[0]
        ang = float(aa_tot.norm())
        if ang < 0.3:
            return
        if ang > 0.9 and grip != GRIP_OPEN:
            # NEVER ladder a big twist with welded cargo: the swing sweeps the
            # load into the counter and winds the wrist through pi (run 38:
            # 2.0 -> 3.1 rad, bowl released 60 cm off the mat)
            print(f"[mw-franka]   restore_wrist REFUSED with cargo ({ang:.2f} rad)",
                  flush=True)
            return
        hold_p = pinch_pos(*hand_pose()).clone()
        if ang > 0.9:
            axis = aa_tot / ang
            n_leg = int(ang / max_leg) + 1
            for i in range(1, n_leg + 1):
                half = 0.5 * ang * (i / n_leg)
                dq = torch.cat([torch.tensor([math.cos(half)], device=device),
                                axis * math.sin(half)])
                q_way = quat_mul(dq.unsqueeze(0), hq0.unsqueeze(0))[0]
                goto(hold_p, q_way / q_way.norm(), grip, tol=0.05, max_steps=90)
        goto(hold_p, Q_HOME, grip, tol=0.05, max_steps=60)

    def rehome(tag: str, cycles: int = 2, dwell: int = 220) -> float:
        """Restore the arm to its normal (boot) configuration between subtasks.

        The task-space channel alone leaves joint-7/elbow debt that compounds
        across subtasks (runs 12-13, 28, 32: every approach launched from a
        tangled arm missed), and the debt is MEASURED in joint space
        before/after, not assumed. Three stages, each earned by a measured
        failure:
          (a) position-only transit to the low-forward point (open air), THEN
              restore_wrist() there — the arc-leg ladder that a near-pi flip
              needs (one-shot shortest-path is ill-conditioned there; run 35's
              plain rehome made the debt WORSE, 147 -> 165 deg), rotated in the
              clear because run 36 laddered beside the appliance and shoved the
              EE into it;
          (b) low-forward at Q_HOME (un-folds a coiled elbow, the proven
              door-approach prologue), then the BOOT pinch pose, where the task
              target and the null-space posture target agree exactly;
          (c) STRONG-POSTURE DWELL — hold the boot pose while the controller's
              null-space gain is raised (kp_null 10 -> 60, critically damped,
              restored right after): the stock pull measurably cannot unwind a
              coiled elbow (run 35), and the dynamically-consistent projector
              keeps the EE pinned while the posture drains, so the task is
              untouched."""
        from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul

        d0 = arm_debt()
        if d0 < REHOME_TOL:  # already clean: do NOTHING — running the extension
            # gymnastics on a clean arm TANGLED it to 71 deg (run 44's boot
            # rehome regression, from a perfect 0-debt spawn), and the PARK spot
            # it then moved to poisoned the next launch (run 45: cross-body
            # flight from over-the-base twisted the wrist 0.97 rad)
            print(f"[mw-franka]   rehome({tag}): debt {math.degrees(d0):.0f} deg, "
                  f"already clean", flush=True)
            return d0
        kp0, kd0 = osc.cfg.kp_null, osc.cfg.kd_null
        for _cyc in range(cycles):
            # position-only transit to open air FIRST (no rot demand, no governor):
            # run 36 laddered in place next to the appliance top, where rotating a
            # tangled wrist shoves the EE into the box — rotate only in the clear
            goto(W(0.30, -0.44, 0.16, mw=False), None, GRIP_OPEN, tol=0.05, max_steps=140)
            restore_wrist(GRIP_OPEN)  # (a) the wrist ladder, in the clear
            goto(W(0.30, -0.44, 0.16, mw=False), Q_HOME, GRIP_OPEN, tol=0.04, max_steps=100)
            # (b2) FAR EXTENSION along the home azimuth, then boot: the boot-pose
            # dwell has TWO stable equilibria on the elbow's redundancy circle
            # (home = elbow down-forward; attractor = elbow up-right, measured
            # identically on every run-40/41 rehome: +70,-38,-85,... — a tucked
            # waypoint did NOT flip it, run 41). A NEARLY-EXTENDED reach leaves no
            # elbow-swivel freedom — the configuration is forced unique with the
            # elbow down — and retracting from there arrives at the boot pose on
            # the home branch by continuity (measured: 124 -> 14 deg, three runs).
            # The far point's GEOMETRY is proven — do not move it (run 44: putting
            # it at Z_HI shrank the retract arc and tangled a clean arm to 71 deg).
            # Only the APPROACH flies high: rise, level ABOVE the far point,
            # vertical descent (the old direct leg squeezed along the box flank
            # with 5 cm and scraped it from failed states — run 43).
            far = torch.tensor(
                [float(BASE_XY[0]) + 0.60 * math.cos(HOME_AZ),
                 float(BASE_XY[1]) + 0.60 * math.sin(HOME_AZ),
                 float(origin[2]) + c.surface_z + 0.25], device=device)
            hp_r = pinch_pos(*hand_pose())
            goto(torch.tensor([float(hp_r[0]), float(hp_r[1]), Z_HI], device=device),
                 None, GRIP_OPEN, tol=0.04, max_steps=140)
            goto(torch.tensor([float(far[0]), float(far[1]), Z_HI], device=device),
                 None, GRIP_OPEN, tol=0.04, max_steps=160)
            goto(far, None, GRIP_OPEN, tol=0.04, max_steps=160)
            goto(P_BOOT, Q_HOME, GRIP_OPEN, tol=0.02, max_steps=160)
            osc.cfg.kp_null, osc.cfg.kd_null = 60.0, 15.5  # (c) 2*sqrt(60) damping
            try:
                for _ in range(dwell):
                    if arm_debt() < REHOME_TOL:
                        break
                    hp, hq = hand_pose()
                    perr = P_BOOT - pinch_pos(hp, hq)
                    q_t = torch.where((Q_HOME * hq).sum() >= 0, Q_HOME, -Q_HOME)
                    aa = axis_angle_from_quat(quat_mul(q_t.unsqueeze(0),
                                                       quat_conjugate(hq.unsqueeze(0))))[0]
                    step_action((perr / pos_scale).clamp(-1.0, 1.0),
                                (aa / rot_scale).clamp(-1.0, 1.0), GRIP_OPEN)
            finally:
                osc.cfg.kp_null, osc.cfg.kd_null = kp0, kd0
            if arm_debt() < REHOME_TOL:
                break
        d1 = arm_debt()
        per_j = ",".join(f"{math.degrees(float(v)):+.0f}" for v in arm_debt_vec())
        print(f"[mw-franka]   rehome({tag}): debt {math.degrees(d0):.0f} -> "
              f"{math.degrees(d1):.0f} deg (j1..j7: {per_j}) "
              f"{'ok' if d1 < REHOME_TOL else 'STILL TANGLED'}", flush=True)
        return d1

    def pads() -> tuple:
        bs = art.data.body_link_state_w[0]
        return bs[fid[0], 0:3] - origin, bs[fid[1], 0:3] - origin

    measure_gap_axis()

    requested = int(scene._requested[0])
    dp = (scene.door.data.root_pos_w[0] - origin).tolist()
    print(f"[mw-franka] program: TIME x{requested}; door={door_ang():.1f} deg "
          f"door_body=({dp[0]:.3f},{dp[1]:.3f},{dp[2]:.3f}) expected=({c.mw_pos[0]:.3f},"
          f"{c.mw_pos[1]:.3f},{c.surface_z:.3f}); "
          f"bar_now={[round(float(v), 3) for v in (bar_now() - origin).tolist()]}",
          flush=True)

    if args.probe_grasp:
        phase("DEBUG: front half of cycle 1 (stage, open, place, retrieve)")
        args.record_every = 2
        r_stage = stage_bowl(0, c.mat_pos[0] - 0.06)
        rehome("probe-post-stage")
        r_open = open_door()
        rehome("probe-post-open")
        r_place = place_bowl_on_plate(0)
        rehome("probe-post-place")
        r_serve = serve_bowl(0, c.mat_pos[0] - 0.06)
        print(f"[mw-franka] probe: stage={r_stage} open={r_open} place={r_place} "
              f"retrieve={r_serve}", flush=True)
        if frames:
            arr = np.stack(frames, axis=0)
            np.savez_compressed(args.out, frames=arr, phases=json.dumps(phases))
            print(f"[mw-franka] saved {arr.shape} -> {args.out}", flush=True)
        print("MICROWAVE_FRANKA_SMOKE_DONE", flush=True)
        env.close()
        return

    if args.only_door:
        phase("DEBUG: door-open only")
        args.record_every = 1
        ok = open_door()
        print(f"[mw-franka] only_door result: {ok}", flush=True)
        if frames:
            arr = np.stack(frames, axis=0)
            np.savez_compressed(args.out, frames=arr, phases=json.dumps(phases))
            print(f"[mw-franka] saved {arr.shape} -> {args.out}", flush=True)
        print("MICROWAVE_FRANKA_SMOKE_DONE", flush=True)
        env.close()
        return

    # Serve slots on the mat (world x): bowl 0 left half, bowl 1 right half.
    SLOT_X = (c.mat_pos[0] - 0.06, c.mat_pos[0] + 0.06)
    # The spawn slots sit INSIDE the door's swing arc (the task's intended hazard):
    # clear both bowls to the mat BEFORE the first door-open.
    # A rehome() follows EVERY subtask (user directive 2026-08-05) so no subtask
    # inherits the previous one's twist. The boot rehome parks the arm LAUNCH-SAFE
    # before the very first grasp (the spawn pose sits beside the box corner).
    rehome("boot")
    for b in (0, 1):
        check(f"stage{b}-on-mat", stage_bowl(b, SLOT_X[b]))
        rehome(f"post-stage{b}")

    for b in range(2):
        phase(f"===== CYCLE {b + 1}: bowl {b} =====")
        check(f"cycle{b}-door-open", open_door())
        rehome(f"c{b}-post-open")
        check(f"cycle{b}-bowl-on-plate", place_bowl_on_plate(b))
        rehome(f"c{b}-post-place")
        check(f"cycle{b}-door-closed", close_door())
        rehome(f"c{b}-post-close")
        check(f"cycle{b}-keyed", key_program(requested))
        check(f"cycle{b}-started", bool(scene._running[0]),
              f"entry={int(scene._entry[0])} timer={int(scene._timer[0])}")
        check(f"cycle{b}-completed", wait_cycle() and bool(scene._heated[0, b]),
              f"heated={bool(scene._heated[0, b])}")
        rehome(f"c{b}-post-wait")
        check(f"cycle{b}-door-reopen", open_door())
        rehome(f"c{b}-post-reopen")
        check(f"cycle{b}-served", serve_bowl(b, SLOT_X[b]))
        rehome(f"c{b}-post-serve")
        check(f"cycle{b}-door-reclose", close_door())
        rehome(f"c{b}-post-reclose")

    flags = int(scene.stage_flags()[0].long().sum())
    check("stage-flags-12", flags == 12, f"{flags}/12: {scene._flags[0].tolist()}")
    check("score-100", int(scene.score()[0]) == 100, f"score={int(scene.score()[0])}")
    check("success", bool(scene.success()[0]))
    print(f"[mw-franka] RESULT: {'ALL PASS' if not FAILS else 'FAILED: ' + ','.join(FAILS)}",
          flush=True)

    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="articulated.microwave.franka.osc",
                            phases=json.dumps(phases), requested=requested)
        print(f"[mw-franka] saved {arr.shape} -> {args.out}", flush=True)
        if args.hdfs_dir:
            rc = os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir} 2>/dev/null; hdfs dfs -put -f "
                           f"{args.out} {args.hdfs_dir}/{os.path.basename(args.out)}")
            print(f"[mw-franka] hdfs upload rc={rc}", flush=True)
    print("MICROWAVE_FRANKA_SMOKE_DONE", flush=True)
    env.close()


def _hard_exit_teardown() -> None:
    """Kit teardown regularly hangs inside env.close()/app.close() — the repo's standard
    hard-exit: a watchdog guarantees the process ends."""
    import os as _os
    import threading as _threading

    watchdog = _threading.Timer(10.0, lambda: _os._exit(0))
    watchdog.daemon = True
    watchdog.start()
    app.close()
    _os._exit(0)


if __name__ == "__main__":
    main()
    _hard_exit_teardown()

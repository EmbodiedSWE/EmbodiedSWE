"""Smoke / oracle test for SpatulaFlipServeScene — NullRobot, kinematic tool drive, RECORDED.

One linear run (pen-holder smoke skeleton):
  1. show       — settle the reset layout (patty on the board raw-side-up, plate beside,
                  spatula at rest) so you can see it; score must read 0;
  2. oracle     — the FULL v2 chain (goal `flip_serve`): pick the spatula up (gravity-
                  compensated kinematic hold, the pen-holder trick), WEDGE the flat blade under
                  the patty (a genuine slide at board level — it rides up the patty chamfer),
                  FLIP it (lift a few cm, roll the blade past the commit point, let physics
                  land it browned-side-up), re-wedge, level CARRY to the plate, and TIP it
                  off — score climbing 10 -> 25 -> 50 -> 70 -> 100 with every transition
                  checked, then success(). Flip retries are counted (the brief's
                  flip-attempts metric); max carry tilt is reported (the finesse number).
  3. variants   — goal `serve` (v0: 10/30/60/100) and goal `flip` (v1: 10/30/100) each
                  driven to success — the goal knob only changes judging, so it is switched
                  at runtime; plus one size-sampled episode (sample_size=True) served to
                  success, proving the rubric across the patty-size axis;
  4. negative A — a patty TELEPORTED onto the plate (never on the blade) must never serve:
                  the arrival-on-blade clause pins the score at 0;
  5. negative B — the blade held BESIDE the patty must not count as wedged, and a patty
                  standing ON EDGE (90 deg) must not count as flipped or resting;
  6. negative C — rolling the blade far past the tilt budget mid-carry over the bare bench
                  must SPILL the payload: spill counter increments, serve never latches;
  7. sweep      — calibration: carry the loaded blade at 0/8/16/24 deg (2 seeds each)
                  through an 18 cm traverse; publishes the per-tilt stay-aboard rates
                  and the TILT BUDGET knee — the number that defines the
                  difficulty. Statistical assertions only (level carry reliable, some
                  nonzero tilt reliable) — raw friction physics is measured, not asserted.
  8. repeat     — the serve chain 3x from fresh resets (the determinism gate, scaled down
                  from the brief's 20/20 spike for smoke runtime; the knob for the full
                  spike is --repeat).
  --demo runs ONLY show + the v2 oracle chain and saves the deliverable video.

Flip/carry maneuver knobs are CLI args (--flip_height/--flip_roll_deg/--flip_steps/
--tip_deg) so GPU tuning iterates without edits. ALWAYS records video via the viewport
rgb annotator (same recipe as crate_packing_smoke: RTX driver-version override, 3-render
ghost flush, npz -> HDFS). Bodies are driven straight through scene handles; the
NullRobot applies nothing.

Run (on a GPU node with the isaaclab env):
    python -m robobench.suites.tool_use.smokes.spatula_flip_serve_smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--demo", action="store_true", default=False,
                    help="record ONE clean v2 chain only (no variants, no negative "
                         "controls, no sweep) — the user-facing deliverable video")
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--out", type=str, default="/tmp/spatula_frames.npz")
parser.add_argument("--hdfs_dir", type=str, default="",
                    help="optional HDFS dir to upload the frames npz to ('' = no upload)")
# maneuver tuning knobs (GPU iteration without edits). The flip retry LADDER starts at
# these values (attempt 1 = slow topple-against-the-board) and escalates to snappier
# ballistic rolls on retries (see FLIP_LADDER below).
parser.add_argument("--flip_height", type=float, default=0.02,
                    help="blade height above the board top when the flip roll starts (m)")
parser.add_argument("--flip_roll_deg", type=float, default=150.0,
                    help="how far past level the blade rolls during the flip")
parser.add_argument("--flip_steps", type=int, default=90,
                    help="steps for the flip roll (fewer = snappier wrist)")
parser.add_argument("--tip_deg", type=float, default=45.0,
                    help="blade pitch that slides the patty off onto the plate")
parser.add_argument("--repeat", type=int, default=3,
                    help="serve-chain repeat count for the determinism phase")
parser.add_argument("--drive_mass", type=float, default=0.75,
                    help="effective held-tool mass used by the NullRobot fixture (kg)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe (same as the render server): kit mis-decodes the L20 driver version and
# silently rejects RTX -> annotator returns EMPTY frames. Disable the check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math
import os
import shutil

import numpy as np
import torch

import robobench
from robobench.core import ENVS
from robobench.suites.tool_use.scenes import SpatulaFlipServeSceneCfg


# ---- float quat helpers (w, x, y, z tuples; env-0 staging math) ----
def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def qx(deg):
    h = math.radians(deg) / 2
    return (math.cos(h), math.sin(h), 0.0, 0.0)


def qy(deg):
    h = math.radians(deg) / 2
    return (math.cos(h), 0.0, math.sin(h), 0.0)


def qnlerp(a, b, f):
    if sum(u * v for u, v in zip(a, b)) < 0.0:
        b = tuple(-v for v in b)
    q = tuple(u + (v - u) * f for u, v in zip(a, b))
    n = math.sqrt(sum(v * v for v in q)) or 1.0
    return tuple(v / n for v in q)


def qang_vel(a, b, dt: float):
    """World angular velocity (rad/s) that takes orientation `a` to `b` in one `dt`
    (small-angle: omega = 2 * vec(b * conj(a)) / dt)."""
    d = qmul(b, (a[0], -a[1], -a[2], -a[3]))
    if d[0] < 0.0:
        d = tuple(-v for v in d)
    return (2.0 * d[1] / dt, 2.0 * d[2] / dt, 2.0 * d[3] / dt)


def ease(f: float) -> float:
    """Cosine ease 0->1: gentle accelerations so a friction-only payload stays aboard."""
    return 0.5 - 0.5 * math.cos(math.pi * f)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    if args.drive_mass <= 0.0:
        raise ValueError(f"--drive_mass must be positive, got {args.drive_mass}")
    # Size sampling OFF for the deterministic main phases (mid patty); one variant phase
    # toggles it on. Goal starts at the full v2 chain.
    # A robot-held 150 g spatula does not respond to payload contact like a free 150 g
    # rigid body: the arm contributes reflected inertia at the blade. The smoke drives
    # root velocity directly, so model that fixture inertia explicitly. At the scene's
    # free-tool mass, a 60 g patty can stop the blade within a step; the final pose pin
    # then advances the blade alone and phases it through the payload. This override is
    # smoke-only: embodied environments retain the physical 0.15 kg tool mass.
    env = ENVS.get("tool_use.spatula")().build(
        num_envs=args.num_envs, device=device,
        scene_cfg=SpatulaFlipServeSceneCfg(
            sample_size=False, goal="flip_serve", spatula_mass=args.drive_mass))
    scene = env.scene
    c = scene.cfg
    print(f"[smoke] NullRobot fixture effective spatula mass={c.spatula_mass:.2f}kg",
          flush=True)
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    names = [name for name, _r in c.families]

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.25, -1.25, 0.95)) + o),
                                tuple(np.array((0.0, 0.0, c.surface_z + 0.10)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames — check RTX recipe "
                  "(driver-version override, NVIDIA_DRIVER_CAPABILITIES)", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0
    # While pinned, the spatula is "held": its root state is rewritten before every
    # physics step — a kinematic hold through the scene handle, no robot. THREE velocity
    # lessons baked in (one per GPU round — this block is the hard-won core):
    #  - pen-holder: every write carries +g*dt of UPWARD velocity so PhysX's gravity
    #    integration cancels to exactly zero (a naive re-pin leaves the platform
    #    free-falling g*dt each step and the PAYLOAD inherits it);
    #  - round 1: a zero-velocity teleport is a blade PhysX believes is stationary —
    #    contacts feel no motion, friction cannot drag cargo;
    #  - round 4: writing the ALREADY-DISPLACED pose (even with the right velocity)
    #    TELEPORTS the 2 mm blade into/through its payload — a vertical lift penetrated
    #    past the thin plate's midplane in one step, the contact normal flipped, and the
    #    solver squeezed the patty out the bottom: every lift shed its payload with
    #    bf x,y unchanged. So the pre-step write carries the OLD pose + the transport
    #    velocity and lets INTEGRATION carry the tool to the target — contacts stay
    #    continuous, the payload is pushed, never phased through.
    # The transport write is single-step-scoped (subsequent steps pin at the commanded
    # pose, zero velocity + g*dt), and after each step() call the tool is re-pinned at
    # ZERO velocity + update(0.0) before judging.
    hold_write: torch.Tensor | None = None  # next pre-step write (may carry transport)
    hold_pin: torch.Tensor | None = None  # zero-velocity pin at the commanded pose
    g_dt = 9.81 * env.dt

    def step(k: int, render: bool = True) -> None:
        nonlocal step_i, hold_write
        for _ in range(k):
            if hold_pin is not None:
                pre = hold_write.clone()
                pre[:, 9] += g_dt  # cancel the gravity kick on top of the transport vel
                scene.spatula.write_root_state_to_sim(pre, all_ids)
                hold_write = hold_pin.clone()  # transport used once; then hold station
            env.step(no_action, render=render)
            if annot is not None and step_i % args.record_every == 0:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                data = annot.get_data()
                arr = np.asarray(data)
                if step_i == 0:
                    print(f"[smoke] first capture: dtype={arr.dtype} shape={arr.shape}",
                          flush=True)
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1
        if hold_pin is not None:
            scene.spatula.write_root_state_to_sim(hold_pin, all_ids)
            env.iscene.update(0.0)

    def make_state(pos, quat=(1.0, 0.0, 0.0, 0.0)) -> torch.Tensor:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = env.iscene.env_origins + torch.tensor(pos, device=device)
        st[:, 3:7] = torch.tensor(quat, device=device)
        return st

    # current commanded hold pose (env-local), maintained by hold()/move_to()
    cur = {"pos": None, "quat": None}

    def hold(pos, quat) -> None:
        """Command the held tool to (pos, quat): the next pre-step write is the OLD pose
        + the transport velocity that reaches the target by integration (see the block
        comment above; zero-velocity pin-in-place on the first grab after a release)."""
        nonlocal hold_write, hold_pin
        pos = tuple(float(v) for v in pos)
        quat = tuple(float(v) for v in quat)
        if cur["pos"] is None:
            hold_write = make_state(pos, quat)
        else:
            hold_write = make_state(cur["pos"], cur["quat"])
            hold_write[:, 7] = (pos[0] - cur["pos"][0]) / env.dt
            hold_write[:, 8] = (pos[1] - cur["pos"][1]) / env.dt
            hold_write[:, 9] = (pos[2] - cur["pos"][2]) / env.dt
            hold_write[:, 10:13] = torch.tensor(qang_vel(cur["quat"], quat, env.dt),
                                                device=device)
        hold_pin = make_state(pos, quat)
        cur["pos"], cur["quat"] = pos, quat

    def move_to(pos, quat, steps: int) -> None:
        """Ease the held tool from its current commanded pose to (pos, quat)."""
        p0, q0 = cur["pos"], cur["quat"]
        for t in range(steps):
            f = ease((t + 1) / steps)
            p = tuple(p0[i] + (pos[i] - p0[i]) * f for i in range(3))
            hold(p, qnlerp(q0, quat, f))
            step(1)
        # write the exact target once more (nlerp path ends there anyway)
        hold(pos, quat)
        step(1)

    def release() -> None:
        nonlocal hold_write, hold_pin
        hold_write = None
        hold_pin = None
        cur["pos"], cur["quat"] = None, None

    # --- env-0 readbacks (local coords) ---
    def pi() -> int:
        return int(scene._present[0].int().argmax())

    def patty_pos():
        b = scene.patties[names[pi()]]
        return (b.data.root_pos_w[0] - env.iscene.env_origins[0]).tolist()

    def plate_pos():
        return (scene.plate.data.root_pos_w[0] - env.iscene.env_origins[0]).tolist()

    def spatula_pose():
        st = scene.spatula.data.root_state_w[0]
        pos = (st[0:3] - env.iscene.env_origins[0]).tolist()
        return pos, tuple(st[3:7].tolist())

    def report(tag: str) -> None:
        i = pi()
        flags = {nm: bool(getattr(scene, "_" + nm)[0])
                 for nm in ("lifted", "wedged", "flipped", "loaded", "served")}
        print(f"[smoke] {tag:12s} | patty={names[i]} {flags} "
              f"score={int(scene.score()[0])} success={bool(scene.success()[0])} "
              f"tilt_max={float(scene._max_carry_tilt[0]):.1f}deg "
              f"spills={int(scene._spills[0])} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, cond))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def diagnose(tag: str) -> None:
        """Blade-frame + rubric-term breakdown — printed on a failed check so the log
        names the guilty term instead of leaving it to the video."""
        i = pi()
        loc = scene._patty_in_blade_frame()[0, i]
        up_z = float(scene._patty_up()[0, i, 2])
        print(f"[smoke]   {tag} {names[i]}: blade-frame x={float(loc[0]) * 1000:.0f}mm "
              f"y={float(loc[1]) * 1000:.0f}mm z={float(loc[2]) * 1000:.0f}mm up_z={up_z:+.2f} "
              f"on_blade={bool(scene.on_blade()[0, i])} on_board={bool(scene.on_board()[0, i])} "
              f"on_plate={bool(scene.on_plate()[0, i])} settled={bool(scene.settled()[0, i])} "
              f"since_loaded={int(scene._since_loaded[0])}", flush=True)

    def trace(tag: str) -> None:
        """Compact one-line probe for maneuver waypoints — world pos, blade-frame pos
        (bf z = the patty BOTTOM, the on-blade band variable), orientation and the live
        predicates: enough to reconstruct WHERE a payload was lost without the video."""
        i = pi()
        loc = scene._patty_in_blade_frame()[0, i]
        up_z = float(scene._patty_up()[0, i, 2])
        pw = (scene.patties[names[i]].data.root_pos_w[0]
              - env.iscene.env_origins[0]).tolist()
        print(f"[smoke]   . {tag:14s} pw=({pw[0]:+.3f},{pw[1]:+.3f},{pw[2]:+.3f}) "
              f"bf=({float(loc[0]) * 1000:+4.0f},{float(loc[1]) * 1000:+4.0f},"
              f"{(float(loc[2]) - c.patty_h / 2) * 1000:+4.0f})mm up_z={up_z:+.2f} "
              f"blade={int(scene.on_blade()[0, i])} board={int(scene.on_board()[0, i])} "
              f"plate={int(scene.on_plate()[0, i])} loaded={int(scene.loaded_now()[0])} "
              f"score={int(scene.score()[0])}", flush=True)

    def settle_until(pred, max_steps: int = 360, poll: int = 15) -> bool:
        """Step in `poll`-sized chunks until `pred()` is true or the budget runs out (the
        pen-holder lesson: judging honest settling on a fixed clock tick calls it a miss)."""
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    # --- maneuvers (all through the kinematic hold; blade frame: +x = tip, origin =
    # blade-bottom centre) ---
    LEVEL = (1.0, 0.0, 0.0, 0.0)
    # the wedge sets the tool's working heading: `dir` = blade +x in the world, `yaw` its
    # angle; every later maneuver composes body-frame rolls/pitches on top of base_q()
    tool = {"yaw": 0.0, "dir": (1.0, 0.0)}

    def base_q():
        h = tool["yaw"] / 2
        return (math.cos(h), 0.0, 0.0, math.sin(h))

    def pickup() -> None:
        """Take hold of the spatula at its resting pose, then raise it to a staging hover."""
        pos, quat = spatula_pose()
        hold(pos, quat)
        step(2)
        move_to((pos[0], pos[1], c.surface_z + 0.15), LEVEL, 60)

    def wedge() -> bool:
        """JAB the blade under the present patty. Round-2 GPU lesson: a quasi-static slide
        always BULLDOZES — blade-side friction sticks to the payload before board friction
        yields, so the patty is shoved across the board instead of climbing. The anchor
        that forces the wedge to slip UNDER is the payload's INERTIA, so the entry is a
        jab, escalating in speed over up to three tries (too slow lets the payload run
        away; too fast is a slap shot — 10-16 step jabs launched it 30-60 cm). The
        approach direction points from the patty toward the board centre whenever the
        patty has drifted, so whatever bulldozing does happen pushes it back INWARD; the
        wedge PLANE comes from the patty's ACTUAL resting height, not the board's (round
        3: a patty knocked onto the bench sat 15 mm below the hardcoded board plane and
        the blade slid clean over it). Returns on-blade (riding, not slid-under)."""
        p = patty_pos()
        dx, dy = c.board_pos[0] - p[0], c.board_pos[1] - p[1]
        nrm = math.hypot(dx, dy)
        dirv = (dx / nrm, dy / nrm) if nrm > 0.05 else (1.0, 0.0)
        tool["dir"], tool["yaw"] = dirv, math.atan2(dirv[1], dirv[0])
        qb = base_q()
        r_p = c.families[pi()][1]
        back = c.blade_l / 2 + r_p + 0.02  # stage the tip 2 cm short of the patty rim
        move_to((p[0] - dirv[0] * back, p[1] - dirv[1] * back, c.surface_z + 0.06), qb, 50)
        for jab_steps in (60, 45, 30):  # bounded escalation (~0.4 / 0.55 / 0.8 m/s peak)
            p = patty_pos()
            z = max(p[2] - c.patty_h / 2 + 0.0005, 0.0005)  # blade sole on the PATTY's plane
            move_to((p[0] - dirv[0] * back, p[1] - dirv[1] * back, z), qb, 30)
            # end 1 cm PAST the patty centre: any bulldozing drift during the climb still
            # leaves the payload well inside the blade footprint
            move_to((p[0] + dirv[0] * 0.01, p[1] + dirv[1] * 0.01, z), qb, jab_steps)
            step(20)
            # CENTERING pass (round-5 lesson): the jab shoves the payload forward a
            # seed-dependent few cm before it climbs, often leaving it riding the TIP
            # half-overhung (bf x up to +59 observed) — such a load survives the wedge
            # but PIVOTS OVER THE TIP EDGE on the first lift (the round-5 lift losses;
            # the two lifts that worked both carried a centred payload). If the patty is
            # at blade-top height, slide the blade forward under it — a gentle relative
            # slide on the slippery plate; the payload keeps its world spot — until the
            # origin sits under its centre, then judge.
            i = pi()
            loc = scene._patty_in_blade_frame()[0, i]
            bottom = float(loc[2]) - c.patty_h / 2
            if (0.001 < bottom < 0.015
                    and abs(float(loc[1])) < c.blade_w / 2 + 0.01
                    and float(loc[0]) < c.blade_l / 2 + c.families[i][1]):
                p = patty_pos()
                move_to((p[0], p[1], z), qb, 40)
                step(20)
            trace(f"jab{jab_steps}")
            if bool((scene.on_blade() & scene._present)[0].any()):
                return True
            p = patty_pos()
            # A failed jab may have moved the payload far enough that the original +x
            # heading now points off the board. Re-aim the next approach inward before
            # retreating; this turns a miss into a recoverable retry instead of a launch.
            dx, dy = c.board_pos[0] - p[0], c.board_pos[1] - p[1]
            nrm = math.hypot(dx, dy)
            if nrm > 0.05:
                dirv = (dx / nrm, dy / nrm)
                tool["dir"], tool["yaw"] = dirv, math.atan2(dirv[1], dirv[0])
                qb = base_q()
            move_to((p[0] - dirv[0] * back, p[1] - dirv[1] * back,
                     max(p[2] - c.patty_h / 2 + 0.0005, 0.0005)), qb, 40)
        return False

    def lift(dz: float, steps: int = 90) -> None:
        # slow by default: ~1.9 mm/step peak — a fast lift is the tip-over lever arm's
        # best friend when the payload sits anywhere forward of centre
        move_to((cur["pos"][0], cur["pos"][1], cur["pos"][2] + dz), cur["quat"], steps)

    # Flip retry ladder: attempt 1 is the slow TOPPLE-AGAINST-THE-BOARD turn (low hover,
    # gentle roll — the patty peels off, catches its rim on the board and falls over its
    # edge); later attempts are progressively snappier BALLISTIC rolls (the wrist-snap:
    # at ~30 rad/s the departing patty inherits enough spin to complete ~170 deg during
    # a ~5 cm fall). Which regime wins is exactly the brief's feasibility spike — the
    # ladder measures it (flip_attempts is the published metric).
    FLIP_LADDER = ((args.flip_height, args.flip_roll_deg, args.flip_steps),
                   (0.05, 160.0, 22),
                   (0.065, 170.0, 12))

    def flip_once(height: float, roll_deg: float, steps: int) -> None:
        """One commit move: carry the loaded blade OVER THE BOARD CENTRE (round-3
        lesson: `flipped` honestly demands a landing ON the board — attempt 1 landed a
        genuinely inverted patty on the bench and earned nothing, so the flip must
        happen where the landing zone is the board), hover at `height`, roll the blade
        about its own long axis to `roll_deg` in `steps`; the patty departs over the low
        edge and physics decides. The blade is biased sideways so the landing centres,
        and the hold height tracks the swinging low edge (rolling about the blade-bottom
        origin would otherwise drive that edge into the board)."""
        qb = base_q()
        perp = (-tool["dir"][1], tool["dir"][0])  # blade-frame +y in the world
        bx, by = c.board_pos[0] + perp[0] * 0.02, c.board_pos[1] + perp[1] * 0.02
        move_to((cur["pos"][0], cur["pos"][1], c.board_top + 0.06), qb, 40)
        move_to((bx, by, c.board_top + 0.06), qb, 110)
        move_to((bx, by, c.board_top + height), qb, 40)
        px, py = cur["pos"][0], cur["pos"][1]
        trace(f"pre-roll h={height:.3f}")
        marks = {steps // 3, (2 * steps) // 3, steps - 1}  # mid-roll departure probes
        for t in range(steps):
            f = ease((t + 1) / steps)
            ang = roll_deg * f
            edge_drop = (c.blade_w / 2) * math.sin(math.radians(min(ang, 179.0)))
            z = c.board_top + max(height, 0.008 + edge_drop)
            hold((px, py, z), qmul(qb, qx(ang)))
            step(1)
            if t in marks:
                trace(f"roll@{ang:.0f}deg")
        step(25)
        trace("post-roll")
        # retreat UP at the rolled orientation first (un-rolling in place would sweep the
        # blade through the landing zone), then away and back to level
        move_to((px, py, c.board_top + 0.14), cur["quat"], 30)
        move_to((px - tool["dir"][0] * 0.18, py - tool["dir"][1] * 0.18,
                 c.board_top + 0.10), qb, 40)

    def carry_and_tip() -> None:
        """Level carry to the plate, then pitch the blade to slide the patty off. The
        traverse is SLOW (the round-3 friction budget: at a combined mu ~0.29 the max
        no-slip horizontal acceleration is ~2.8 m/s^2, so the old 130-step profile was
        at the limit), and the tip-off happens low so the payload drops, not falls."""
        pl = plate_pos()
        dirv, qb = tool["dir"], base_q()
        tx, ty = pl[0] - dirv[0] * 0.03, pl[1] - dirv[1] * 0.03
        move_to((cur["pos"][0], cur["pos"][1], c.surface_z + 0.14), qb, 40)
        move_to((tx, ty, c.surface_z + 0.14), qb, 190)
        trace("carry-hi")
        move_to((tx, ty, c.plate_top + 0.008), qb, 60)
        trace("carry-low")
        for t in range(140):
            f = ease((t + 1) / 140)
            ang = args.tip_deg * f
            # pitching about the blade-bottom origin swings the tip DOWN by sin(ang)*l/2;
            # raise the hold in step so the tip skims ~6 mm over the plate the whole way
            z = c.plate_top + 0.006 + math.sin(math.radians(ang)) * (c.blade_l / 2)
            hold((tx, ty, z), qmul(qb, qy(ang)))
            step(1)
        step(30)
        trace("tipped")
        move_to((pl[0] - dirv[0] * 0.22, pl[1] - dirv[1] * 0.22, c.surface_z + 0.12), qb, 60)

    def set_down() -> None:
        move_to((c.spatula_pos[0], c.spatula_pos[1], c.surface_z + 0.004), LEVEL, 70)
        release()
        step(30)

    # =========================== 1. show ====================================================
    env.reset()
    report("reset")
    step(60)
    report("show")
    # Predicates must read a clean slate — if anything scores at reset, the rubric is broken
    # and everything downstream is meaningless. Fail loudly.
    assert int(scene.score()[0]) == 0, \
        f"score={int(scene.score()[0])} at reset — rubric predicates broken"

    # =========================== 2. oracle: the full v2 chain ===============================
    flip_attempts = 0

    pickup()
    check("v2 rubric transition 10 (tool lifted)",
          settle_until(lambda: int(scene.score()[0]) == 10))

    ok_w = wedge()
    check("v2 wedge: patty rides the blade", ok_w)
    check("v2 rubric transition 25 (blade under)",
          settle_until(lambda: int(scene.score()[0]) == 25))
    if int(scene.score()[0]) != 25:
        diagnose("post-wedge")

    # the flip, escalating through the ladder with re-wedge retries (the brief's
    # flip-attempts metric)
    for attempt, (fh, fr, fs) in enumerate(FLIP_LADDER):
        flip_attempts += 1
        flip_once(fh, fr, fs)
        if settle_until(lambda: int(scene.score()[0]) == 50, max_steps=420):
            break
        diagnose(f"flip-attempt-{attempt}")
        if attempt < len(FLIP_LADDER) - 1:
            print(f"[smoke]   flip retry: not flipped-at-rest, re-wedging for "
                  f"h={FLIP_LADDER[attempt + 1][0]} roll={FLIP_LADDER[attempt + 1][1]} "
                  f"steps={FLIP_LADDER[attempt + 1][2]}", flush=True)
            wedge()
    report("flipped")
    check("v2 rubric transition 50 (flipped, at rest on the board)",
          int(scene.score()[0]) == 50)

    ok_w2 = wedge()
    check("v2 re-wedge for the serve leg", ok_w2)
    lift(0.12)
    trace("lifted-v2")
    check("v2 rubric transition 70 (loaded after the flip)",
          settle_until(lambda: int(scene.score()[0]) == 70))
    check("v2 loaded-but-not-served is NOT success", not bool(scene.success()[0]))

    carry_and_tip()
    check("v2 rubric transition 100 (served)",
          settle_until(lambda: int(scene.score()[0]) == 100, max_steps=420))
    if int(scene.score()[0]) != 100:
        diagnose("post-serve")
    set_down()
    ok_success = bool(scene.success()[0])
    check("v2 oracle chain reaches success()", ok_success)
    report("v2-done")
    print(f"[smoke] RESULT: {'FLIP+SERVE — SUCCESS' if ok_success else 'CHAIN FAIL'} "
          f"(flip attempts={flip_attempts}, max carry tilt="
          f"{float(scene._max_carry_tilt[0]):.1f} deg, spills={int(scene._spills[0])})",
          flush=True)
    check("v2 chain spills nothing", int(scene._spills[0]) == 0)

    if args.demo:  # deliverable video = the one clean chain above; stop here
        if frames:
            arr = np.stack(frames, axis=0)
            np.savez_compressed(args.out, frames=arr, env="tool_use.spatula")
            print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
            if shutil.which("hdfs"):  # optional archive channel; absent on RunPod
                os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir} 2>/dev/null; "
                          f"hdfs dfs -put -f {args.out} "
                          f"{args.hdfs_dir}/{os.path.basename(args.out)}")
        print("SPATULA_SMOKE_DONE", flush=True)
        env.close()
        return

    # =========================== 3. goal variants ===========================================
    # The goal knob only changes JUDGING (tables + face gate), so it is switched at runtime.
    def serve_chain() -> bool:
        pickup()
        if not wedge():
            wedge()  # one retry — raw wedge physics is measured in the sweep, not here
        lift(0.12)
        trace("lifted")
        ok60 = settle_until(lambda: int(scene.score()[0]) == 60)
        if not ok60:
            diagnose("serve-no-60")
        carry_and_tip()
        ok100 = settle_until(lambda: int(scene.score()[0]) == 100, max_steps=420)
        if not ok100:
            diagnose("serve-no-100")
        set_down()
        return ok60 and ok100 and bool(scene.success()[0])

    scene.cfg.goal = "serve"
    torch.manual_seed(3)
    env.reset()
    step(40)
    ok = serve_chain()
    report("v0-serve")
    if not ok:
        diagnose("v0-serve")
    check("v0 serve chain: 60 -> 100 -> success", ok)

    scene.cfg.goal = "flip"
    torch.manual_seed(4)
    env.reset()
    step(40)
    pickup()
    ok10_30 = settle_until(lambda: int(scene.score()[0]) == 10) and wedge() and \
        settle_until(lambda: int(scene.score()[0]) == 30)
    if not ok10_30:
        diagnose("v1-no-30")
    for attempt, (fh, fr, fs) in enumerate(FLIP_LADDER):
        flip_once(fh, fr, fs)
        if settle_until(lambda: int(scene.score()[0]) == 100, max_steps=420):
            break
        diagnose(f"v1-flip-{attempt}")
        wedge()
    set_down()
    report("v1-flip")
    check("v1 flip chain: 10 -> 30 -> 100 -> success",
          ok10_30 and int(scene.score()[0]) == 100 and bool(scene.success()[0]))

    # size-sampling episode: prove the rubric judges the SAMPLED patty (any of 3 sizes)
    scene.cfg.goal = "serve"
    scene.cfg.sample_size = True
    torch.manual_seed(7)
    env.reset()
    step(40)
    print(f"[smoke] size-sampled episode: present patty = {names[pi()]}", flush=True)
    ok = serve_chain()
    report("size-sample")
    check(f"size-sampled serve ({names[pi()]}) reaches success", ok)
    scene.cfg.sample_size = False

    # =========================== 4. negative control A (teleport-serve) =====================
    env.reset()
    step(40)
    pl = plate_pos()
    i = pi()
    st = make_state((pl[0], pl[1], c.plate_top + c.patty_h / 2 + 0.001))
    scene.patties[names[i]].write_root_state_to_sim(st, all_ids)
    env.iscene.update(0.0)
    step(80)
    report("teleport")
    check("teleported patty: on the plate but NOT served (arrival-on-blade clause)",
          bool(scene.on_plate()[0, i]) and not bool(scene._served[0]))
    check("teleported patty: score pinned at 0", int(scene.score()[0]) == 0)
    check("teleported patty: success rejected", not bool(scene.success()[0]))

    # =========================== 5. negative control B (beside / on edge) ===================
    env.reset()
    step(40)
    pickup()
    p = patty_pos()
    # blade held BESIDE the patty at board level: near, but never under
    beside_y = p[1] + c.blade_w / 2 + c.families[pi()][1] + 0.03
    move_to((p[0], beside_y, c.board_top + 0.001), LEVEL, 60)
    step(40)
    check("blade beside the patty: wedged NOT latched", not bool(scene._wedged[0]))
    set_down()
    # patty ON EDGE (90 deg about x, resting on its rim): neither flipped nor flat
    i = pi()
    r_i = c.families[i][1]
    st = make_state((p[0], p[1], c.board_top + r_i + 0.001), qx(90.0))
    scene.patties[names[i]].write_root_state_to_sim(st, all_ids)
    env.iscene.update(0.0)
    check("patty on edge: flipped_now rejects (under-rotation)",
          not bool(scene.flipped_now()[0, i]))
    check("patty on edge: not 'resting flat' on the board",
          not bool(scene.on_board()[0, i]))
    step(80)  # let it topple wherever physics likes; nothing asserted about the aftermath

    # =========================== 6. negative control C (over-tilt spill) ====================
    env.reset()
    step(40)
    scene.cfg.goal = "serve"
    pickup()
    wedge()
    lift(0.15, steps=110)
    trace("spill-lifted")
    # carry off the board over the bare bench, then roll far past any plausible budget
    move_to((0.02, -0.22, c.surface_z + 0.16), base_q(), 130)
    trace("spill-carry")
    for t in range(50):
        f = ease((t + 1) / 50)
        hold(cur["pos"], qmul(base_q(), qx(65.0 * f)))
        step(1)
    trace("spill-rolled")
    settle_until(lambda: bool(scene._spills[0] > 0), max_steps=300)
    set_down()
    report("spill")
    check("over-tilt carry: payload spill counted", int(scene._spills[0]) >= 1)
    check("over-tilt carry: never served", not bool(scene._served[0]))
    check("over-tilt carry: success rejected", not bool(scene.success()[0]))

    # =========================== 7. calibration sweep (the tilt budget) =====================
    # The brief's load-bearing number: how much blade tilt the friction carry tolerates.
    # Wedge (its own 3-jab ladder), lift, roll to a fixed tilt, traverse 18 cm. COARSE
    # 4x2 grid for smoke runtime (expected knee ~ atan(mu 0.33), around 18 deg, so
    # 8-deg spacing brackets it); the fine grid belongs to the full 20/20
    # feasibility spike. Reported as rates; the knee (last all-seeds tilt) is PUBLISHED.
    SWEEP_TILTS, SWEEP_SEEDS = (0.0, 8.0, 16.0, 24.0), 2
    print(f"[smoke] CALIBRATION SWEEP (carry tilt -> stay-aboard rate, "
          f"{SWEEP_SEEDS} seeds each)", flush=True)
    results: dict[float, int] = {}
    for tilt_deg in SWEEP_TILTS:
        aboard = 0
        for seed in range(SWEEP_SEEDS):
            release()  # a failed prior trial may still be holding the tool
            torch.manual_seed(seed)
            env.reset()
            step(20)
            pickup()
            if not wedge():  # the jab traces above name the guilty term per attempt
                print(f"[smoke]   tilt={tilt_deg:.0f} seed={seed}: WEDGE failed (see jab "
                      f"traces) — counted as lost", flush=True)
                continue
            lift(0.10, steps=80)
            for t in range(40):
                f = ease((t + 1) / 40)
                hold(cur["pos"], qmul(base_q(), qx(tilt_deg * f)))
                step(1)
            # tilted traverse: 18 cm along the blade heading at ~0.14 m/s
            p0, dv = cur["pos"], tool["dir"]
            for t in range(150):
                f = ease((t + 1) / 150)
                hold((p0[0] + dv[0] * 0.18 * f, p0[1] + dv[1] * 0.18 * f, p0[2]),
                     cur["quat"])
                step(1)
            step(20)
            trace(f"sweep t={tilt_deg:.0f} s={seed}")
            stay = bool((scene.on_blade() & scene._present)[0].any())
            aboard += int(stay)
            print(f"[smoke]   tilt={tilt_deg:.0f}deg seed={seed}: aboard={stay}", flush=True)
            release()
        results[tilt_deg] = aboard
    knee = max((k for k, v in results.items() if v == SWEEP_SEEDS), default=0.0)
    print("[smoke] SWEEP RESULT: " +
          " | ".join(f"{k:.0f}deg: {v}/{SWEEP_SEEDS}" for k, v in results.items()) +
          f"  -> carry tilt budget (last all-seeds) = {knee:.0f} deg", flush=True)
    # Statistical assertions only (raw friction physics; the pen-holder sweep precedent): a level
    # carry must be reliable and SOME nonzero tilt must hold — the exact knee is measured.
    check("sweep: level carry (0 deg) holds every seed",
          results.get(0.0, 0) == SWEEP_SEEDS)
    check("sweep: some nonzero tilt holds every seed",
          any(v == SWEEP_SEEDS for k, v in results.items() if k > 0))

    # =========================== 8. determinism (repeat serve chains) =======================
    wins = 0
    for seed in range(args.repeat):
        torch.manual_seed(100 + seed)
        env.reset()
        step(30)
        if serve_chain():
            wins += 1
        print(f"[smoke]   repeat seed={seed}: success={bool(scene.success()[0])}", flush=True)
    check(f"repeat: serve chain {args.repeat}/{args.repeat}", wins == args.repeat)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="tool_use.spatula")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
        if shutil.which("hdfs"):  # optional archive channel; absent on RunPod
            rc = args.hdfs_dir and os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir} 2>/dev/null; "
                           f"hdfs dfs -put -f {args.out} {args.hdfs_dir}/"
                           f"{os.path.basename(args.out)}")
            print(f"[smoke] hdfs upload rc={rc} -> "
                  f"{args.hdfs_dir}/{os.path.basename(args.out)}", flush=True)
    all_ok = all(ok for _name, ok in checks)
    print(f"[smoke] RESULT: {'ALL PASS' if all_ok else 'FAIL'} "
          f"({sum(ok for _n, ok in checks)}/{len(checks)} checks)", flush=True)
    print("SPATULA_SMOKE_DONE", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()

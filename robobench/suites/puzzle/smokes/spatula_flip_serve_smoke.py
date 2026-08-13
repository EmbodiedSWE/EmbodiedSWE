"""Smoke / oracle test for SpatulaFlipServeScene — NullRobot, kinematic tool drive, RECORDED.

One linear run (pen-holder smoke skeleton), against the SCANNED kitchen assets (bread
slice in a rimmed fry pan, bamboo plate):
  1. show       — settle the reset layout (bread in the pan, plate beside, spatula at
                  rest) so you can see it; score must read 0;
  2. oracle     — the FULL v2 chain (goal `flip_serve`): pick the spatula up (gravity-
                  compensated kinematic hold, the pen-holder trick; the scene builds
                  with `pan_dynamic=True` so the pan is a heavy FREE body — tool-pan
                  contact is physically real and the old clip-through-the-rim bug
                  cannot happen), WEDGE the blade under the bread — a PITCHED
                  over-the-rim entry, tip on the pan floor, jabbing the slice AGAINST
                  THE PAN WALL, then the fused lift-and-level out — FLIP it (hover over
                  the pan, roll past the commit point, let physics land it back in the
                  pan), re-wedge, level CARRY to the plate, and TIP it off — score
                  climbing 10 -> 25 -> 50 -> 70 -> 100 with every transition checked,
                  then success(). Flip retries are counted; max carry tilt is reported
                  (the finesse number).
  3. variants   — goal `serve` (v0: 10/30/60/100) and goal `flip` (v1: 10/30/100) each
                  driven to success — the goal knob only changes judging, so it is
                  switched at runtime; plus one size-sampled episode (sample_size=True)
                  served to success, proving the rubric across the bread-size axis;
  4. negative A — a bread TELEPORTED onto the plate (never on the blade) must never
                  serve: the arrival-on-blade clause pins the score at 0;
  5. negative B — the blade hovering ABOVE the bread (near in xy, never under) must not
                  count as wedged, and a slice standing ON EDGE in the pan must not count
                  as flipped or resting;
  6. negative C — rolling the blade far past the tilt budget mid-carry over the bare
                  bench must SPILL the payload: spill counter increments, serve never
                  latches;
  7. sweep      — calibration: carry the loaded blade at 0/8/16/24 deg (2 seeds each)
                  through an 18 cm traverse; publishes the per-tilt stay-aboard rates and
                  the TILT BUDGET knee. Statistical assertions only (level carry
                  reliable, some nonzero tilt reliable) — raw friction physics is
                  measured, not asserted.
  8. repeat     — the serve chain 3x from fresh resets (the determinism gate; --repeat
                  scales it up to the full spike).
  --demo runs ONLY show + the v2 oracle chain and saves the deliverable video.

Flip/carry maneuver knobs are CLI args (--flip_height/--flip_roll_deg/--flip_steps/
--tip_deg) so GPU tuning iterates without edits. ALWAYS records video via the viewport
rgb annotator (same recipe as crate_packing_smoke: RTX driver-version override, 3-render
ghost flush, npz -> HDFS). Bodies are driven straight through scene handles; the
NullRobot applies nothing.

Run (on a GPU node with the isaaclab env):
    python -m robobench.suites.puzzle.smokes.spatula_flip_serve_smoke --headless
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
# these values (attempt 1 = slow topple back into the pan) and escalates to snappier
# ballistic rolls on retries (see FLIP_LADDER below).
parser.add_argument("--flip_height", type=float, default=0.065,
                    help="blade height above the pan floor when the flip roll starts (m). "
                         "6.5 cm keeps every tool checkpoint above the pan wall profile "
                         "through the whole capped roll with margin — at 6.0 the neck "
                         "still grazed the rim band by 4 mm (clearance-monitor measured)")
parser.add_argument("--flip_roll_deg", type=float, default=95.0,
                    help="how far the blade rolls during the flip. CAPPED at ~95: a "
                         "back-arm point h above the blade plane sits at h*cos(roll) — "
                         "above the plane until 90 deg, plunging after. The old 150-170 "
                         "deep rolls ALWAYS swept the neck/handle through the pan wall "
                         "band (measured +58..+73 mm at every practical hover; clearing "
                         "a 170-deg roll needs a 10+ cm hover, which the landing cannot "
                         "tolerate). The flip itself comes from the SNAP, not the roll")
parser.add_argument("--flip_steps", type=int, default=14,
                    help="steps for the flip roll. The wrist-snap flip: ~95 deg in "
                         "10-14 steps hands the departing slice 13-19 rad/s of spin — "
                         "it completes the flip BALLISTICALLY and lands back in the pan")
parser.add_argument("--tip_deg", type=float, default=40.0,
                    help="blade pitch that slides the bread off onto the plate (just past "
                         "the ~18 deg friction angle; steeper exits spin the slice into a "
                         "face-down landing the v0 serve gate rejects — measured)")
parser.add_argument("--wedge_pitch_deg", type=float, default=28.0,
                    help="tip-down blade pitch for the over-the-rim wedge ENTRY descent")
parser.add_argument("--wedge_run_pitch_deg", type=float, default=20.0,
                    help="tip-down pitch during the jab. 20 is the table-era ALL-PASS "
                         "value: high enough that the neck crosses the rim line well "
                         "above the crest (the per-step clearance monitor measures it), "
                         "and the kinematic hold pushes the slice through the brief "
                         "above-friction-angle window against the pan wall")
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
from robobench.suites.puzzle.scenes import SpatulaFlipServeSceneCfg


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
    # Size sampling OFF for the deterministic main phases (mid bread); one variant phase
    # toggles it on. Goal starts at the full v2 chain.
    # A robot-held 150 g spatula does not respond to payload contact like a free 150 g
    # rigid body: the arm contributes reflected inertia at the blade. The smoke drives
    # root velocity directly, so model that fixture inertia explicitly. At the scene's
    # free-tool mass, a 60 g slice can stop the blade within a step; the final pose pin
    # then advances the blade alone and phases it through the payload. This override is
    # smoke-only: embodied environments retain the physical 0.15 kg tool mass.
    env = ENVS.get("puzzle.spatula")().build(
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
    names = [name for name, _s in c.families]
    PAN_FLOOR = c.pan_floor_z  # bread rest plane inside the pan (env-local z)
    RIM = c.surface_z + c.pan_rim_top  # rim crest (env-local z)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.72, -0.72, c.surface_z + 0.52)) + o),
                                tuple(np.array((0.0, 0.05, c.surface_z + 0.06)) + o),
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
    # physics step — a kinematic hold through the scene handle, no robot. Penetration
    # against the PAN is prevented by the SCENE, not the hold: the smoke builds with
    # `pan_dynamic=True`, so the pan is a heavy free body that takes real contact
    # response from the kinematically driven tool (a kinematic-vs-kinematic pair
    # generates no contacts — that was the old clipping bug. A force-PD hold was tried
    # instead and reverted: 6 GPU runs of contact chaos — every phase needed its own
    # authority tuning and each fix uncovered the next flick/stall/drag pathology).
    # THREE velocity lessons baked in (one per GPU round — this block is the hard-won
    # core):
    #  - pen-holder: every write carries +g*dt of UPWARD velocity so PhysX's gravity
    #    integration cancels to exactly zero (a naive re-pin leaves the platform
    #    free-falling g*dt each step and the PAYLOAD inherits it);
    #  - round 1: a zero-velocity teleport is a blade PhysX believes is stationary —
    #    contacts feel no motion, friction cannot drag cargo;
    #  - round 4: writing the ALREADY-DISPLACED pose (even with the right velocity)
    #    TELEPORTS the thin blade into/through its payload — so the pre-step write
    #    carries the OLD pose + the transport velocity and lets INTEGRATION carry the
    #    tool to the target — contacts stay continuous, the payload is pushed, never
    #    phased through.
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
            if hold_pin is not None:
                pen_check()
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

    # --- pan-wall clearance monitor: the no-penetration claim, MEASURED --------------
    # The kinematic hold ignores tool-pan contacts (kinematic pairs generate none), so
    # clearance is a property of the maneuver geometry — this monitor samples the tool
    # centerline (3 blade + 10 stick checkpoints from the authored collision rig, each
    # with its effective radius) against the pan's linearized wall profile EVERY held
    # step, and the demo verdict carries a PASS/FAIL on the worst margin. A clip that
    # once needed a user screenshot to notice now fails the run.
    PEN_PTS = np.array([  # (local x, local z, effective radius) — spatula.usd boxes
        [+0.0584, 0.0000, 0.004], [0.0000, 0.0000, 0.004], [-0.0584, 0.0000, 0.004],
        [-0.0676, 0.0123, 0.014], [-0.0877, 0.0264, 0.016], [-0.1083, 0.0410, 0.014],
        [-0.1289, 0.0505, 0.012], [-0.1496, 0.0571, 0.012], [-0.1701, 0.0630, 0.012],
        [-0.1909, 0.0693, 0.013], [-0.2116, 0.0756, 0.013], [-0.2323, 0.0817, 0.014],
        [-0.2521, 0.0860, 0.012]])
    pen_stat = {"worst": -1.0, "steps": 0, "idx": -1, "step": -1}

    def pen_check() -> None:
        st = scene.spatula.data.root_state_w[0, 0:7].detach().cpu().numpy()
        pan = scene.pan.data.root_pos_w[0].detach().cpu().numpy()
        w, x, y, z = st[3:7]
        ex = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])
        ez = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
        pw = st[0:3] + PEN_PTS[:, 0:1] * ex + PEN_PTS[:, 1:2] * ez
        r = np.hypot(pw[:, 0] - pan[0], pw[:, 1] - pan[1])
        z_rel = pw[:, 2] - pan[2]
        frac = np.clip((r - c.pan_r_floor) / (c.pan_r_rim_in - c.pan_r_floor), 0.0, 1.0)
        in_zone = (r > c.pan_r_floor) & (r < c.pan_r_rim_in + 0.005)
        viol = np.where(in_zone, c.pan_rim_top * frac + PEN_PTS[:, 2] - z_rel, -1.0)
        worst = float(viol.max())
        if worst > pen_stat["worst"] + 0.005:  # name every materially new worst
            k = int(viol.argmax())
            print(f"[smoke]   [pen] worst={worst * 1000:+.1f}mm pt={k} "
                  f"(local x={PEN_PTS[k, 0]:+.3f}) r={float(r[k]):.3f} "
                  f"z_rel={float(z_rel[k]):+.3f} step={step_i}", flush=True)
        if worst > pen_stat["worst"]:
            pen_stat["worst"] = worst
            pen_stat["idx"] = int(viol.argmax())
            pen_stat["step"] = step_i
        if worst > 0.002:
            pen_stat["steps"] += 1

    # --- env-0 readbacks (local coords) ---
    def pi() -> int:
        return int(scene._present[0].int().argmax())

    def bread_pos():
        b = scene.breads[names[pi()]]
        return (b.data.root_pos_w[0] - env.iscene.env_origins[0]).tolist()

    def plate_pos():
        return (scene.plate.data.root_pos_w[0] - env.iscene.env_origins[0]).tolist()

    def pan_xy():
        """LIVE pan centre: the pan is a free body in this build (pan_dynamic=True) and
        jab reactions scoot it — aim built on the config constant lands beside the bowl
        after a few contacts (measured: two flips missed the drifted pan)."""
        p = (scene.pan.data.root_pos_w[0] - env.iscene.env_origins[0]).tolist()
        return p[0], p[1]

    def spatula_pose():
        st = scene.spatula.data.root_state_w[0]
        pos = (st[0:3] - env.iscene.env_origins[0]).tolist()
        return pos, tuple(st[3:7].tolist())

    def report(tag: str) -> None:
        i = pi()
        flags = {nm: bool(getattr(scene, "_" + nm)[0])
                 for nm in ("lifted", "wedged", "flipped", "loaded", "served")}
        print(f"[smoke] {tag:12s} | bread={names[i]} {flags} "
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
        loc = scene._bread_in_blade_frame()[0, i]
        up_z = float(scene._bread_up()[0, i, 2])
        print(f"[smoke]   {tag} {names[i]}: blade-frame x={float(loc[0]) * 1000:.0f}mm "
              f"y={float(loc[1]) * 1000:.0f}mm z={float(loc[2]) * 1000:.0f}mm up_z={up_z:+.2f} "
              f"on_blade={bool(scene.on_blade()[0, i])} on_pan={bool(scene.on_pan()[0, i])} "
              f"on_plate={bool(scene.on_plate()[0, i])} settled={bool(scene.settled()[0, i])} "
              f"since_loaded={int(scene._since_loaded[0])}", flush=True)

    def trace(tag: str) -> None:
        """Compact one-line probe for maneuver waypoints — world pos, blade-frame pos
        (bf z = the bread BOTTOM, the on-blade band variable), orientation and the live
        predicates: enough to reconstruct WHERE a payload was lost without the video."""
        i = pi()
        loc = scene._bread_in_blade_frame()[0, i]
        up_z = float(scene._bread_up()[0, i, 2])
        pw = (scene.breads[names[i]].data.root_pos_w[0]
              - env.iscene.env_origins[0]).tolist()
        tw = (scene.spatula.data.root_pos_w[0] - env.iscene.env_origins[0]).tolist()
        print(f"[smoke]   . {tag:14s} pw=({pw[0]:+.3f},{pw[1]:+.3f},{pw[2]:+.3f}) "
              f"bf=({float(loc[0]) * 1000:+4.0f},{float(loc[1]) * 1000:+4.0f},"
              f"{(float(loc[2]) - c.bread_h(i) / 2) * 1000:+4.0f})mm up_z={up_z:+.2f} "
              f"blade={int(scene.on_blade()[0, i])} pan={int(scene.on_pan()[0, i])} "
              f"plate={int(scene.on_plate()[0, i])} loaded={int(scene.loaded_now()[0])} "
              f"score={int(scene.score()[0])} tool_z={tw[2]:+.3f}",
              flush=True)

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
        """Wedge the blade under the bread INSIDE the pan. The rim forbids the flat
        horizontal approach, so entry is a two-phase pitch: descend STEEP (the tip low,
        heel and handle clear of the rim crest) until the tip skims the pan floor behind
        the slice, then relax to the RUN pitch about the planted tip — ~14 deg sits in
        the measured window that still clears the rim near the centre AND stays below
        the ~18 deg friction angle, so climbed cargo holds statically. The jab itself is
        the proven INERTIA jab aimed INWARD (away from the nearest wall): the round-2
        GPU lesson stands — a slow slide bulldozes — and jabbing outward pins the slice
        on the wall only until leveling squeezes it out (measured on GPU: the first
        in-pan build flung the slice over the rim onto the plate). A wall-adjacent slice
        cannot be approached from behind at all, so retries against one jab
        TANGENTIALLY from an interior stage point — even a pass that only relocates the
        slice inward hands the next try the easy centred geometry. After a climb: lift
        out PITCHED (static hold), level only at altitude, then the centering pass.
        Escalating jab speeds over up to three tries. Returns on-blade (riding, not
        slid-under)."""
        entry = args.wedge_pitch_deg
        run = args.wedge_run_pitch_deg
        i = pi()
        r_i, h_i = c.bread_r(i), c.bread_h(i)

        def pitch_about_tip(a0: float, a1: float, steps: int) -> None:
            """Rotate the held blade from pitch a0 to a1 keeping the TIP planted (the
            origin orbits the tip; pivoting about the origin would sweep the tip)."""
            dirv = tool["dir"]
            s0, c0 = math.sin(math.radians(a0)), math.cos(math.radians(a0))
            tipx = cur["pos"][0] + dirv[0] * c0 * c.blade_l / 2
            tipy = cur["pos"][1] + dirv[1] * c0 * c.blade_l / 2
            tipz = cur["pos"][2] - s0 * c.blade_l / 2
            for t in range(steps):
                f = ease((t + 1) / steps)
                a = a0 + (a1 - a0) * f
                sa, ca = math.sin(math.radians(a)), math.cos(math.radians(a))
                hold((tipx - dirv[0] * ca * c.blade_l / 2,
                      tipy - dirv[1] * ca * c.blade_l / 2,
                      tipz + sa * c.blade_l / 2), qmul(base_q(), qy(a)))
                step(1)

        for jab_steps in (60, 50, 40):  # gentle escalation (in-pan the WALL is the
            # anchor, so even the slow jab climbs; fast jabs vaulted the slice over the
            # rim — measured, run 3)
            p = bread_pos()
            pcx, pcy = pan_xy()  # live centre — the free pan scoots under jabs
            rx, ry = p[0] - pcx, p[1] - pcy
            r_off = math.hypot(rx, ry)
            in_pan = r_off < c.pan_r_floor + 0.01
            if not in_pan:
                # ESCAPED payload (a flip toss or a spill can land it on the plate or
                # the bare bench): the original flat jab, at the slice's OWN resting
                # plane (round-3 lesson), aimed at the nearest basin centre — plate
                # recess if it sits on the plate (downhill, every retry flattens the
                # geometry), else back toward the pan's side of the bench. The flat jab
                # keeps the ORIGINAL fast ladder: with no wall anchor only payload
                # inertia wedges, and the slow in-pan speeds just bulldoze out here
                # (measured: four flat jabs herded the slice 30 cm without one climb).
                flat_steps = {60: 55, 50: 40, 40: 28}[jab_steps]
                pl = plate_pos()
                on_plate = math.hypot(p[0] - pl[0], p[1] - pl[1]) < c.plate_r + 0.02
                gx, gy = (pl[0], pl[1]) if on_plate else (pcx, pcy)
                dx, dy = gx - p[0], gy - p[1]
                nrm = math.hypot(dx, dy)
                dirv = (dx / nrm, dy / nrm) if nrm > 0.03 else (1.0, 0.0)
                tool["dir"], tool["yaw"] = dirv, math.atan2(dirv[1], dirv[0])
                qb = base_q()
                back = c.blade_l / 2 + r_i + 0.02
                z = max(p[2] - h_i / 2 + 0.0005, 0.0005)
                move_to((p[0] - dirv[0] * back, p[1] - dirv[1] * back,
                         c.surface_z + 0.09), qb, 40)
                move_to((p[0] - dirv[0] * back, p[1] - dirv[1] * back, z), qb, 30)
                trace("entry-flat")
                move_to((p[0] + dirv[0] * 0.01, p[1] + dirv[1] * 0.01, z), qb, flat_steps)
                step(20)
                trace(f"jab{flat_steps}")
            else:
                if r_off > 0.075:  # wall-adjacent: no room behind — sweep tangentially
                    dirv = (-ry / r_off, rx / r_off)
                elif r_off > 0.03:  # OUTWARD radial: jab the slice against the wall —
                    # the wall is the anchor that lets the blade slip under (the inward
                    # jab surrendered it and the thick slice tumbled free; measured)
                    dirv = (rx / r_off, ry / r_off)
                else:  # centred slice: any heading; +x (the wall is ~10 cm out)
                    dirv = (1.0, 0.0)
                tool["dir"], tool["yaw"] = dirv, math.atan2(dirv[1], dirv[0])
                qb = base_q()
                sinp, cosp = math.sin(math.radians(entry)), math.cos(math.radians(entry))
                sinr, cosr = math.sin(math.radians(run)), math.cos(math.radians(run))
                # stage point: tip lands 1.5 cm behind the slice rim, but the ORIGIN
                # never leaves the bowl interior (r 0.06). The clamp is a CLEARANCE
                # constraint on the trailing HEEL: at stage 0.085 the heel rides the
                # rim band (r 0.134) at 43-46 mm through the whole jab where the crest
                # needs 57 mm — measured +11 mm INTO the crest (the penetration the
                # user screenshotted twice). At 0.06 the heel stays at r <= 0.112,
                # over wall barely 6 mm tall. A tip landing ON a near-centre slice is
                # harmless under the kinematic pin (it pins, the jab slides under —
                # the months-proven behavior; only the reverted force drive squeezed
                # slices out).
                back = cosp * c.blade_l / 2 + r_i + 0.015
                ax, ay = p[0] - dirv[0] * back, p[1] - dirv[1] * back
                arx, ary = ax - pcx, ay - pcy
                a_off = math.hypot(arx, ary)
                if a_off > 0.06:  # clamp the stage point into the interior
                    ax = pcx + arx / a_off * 0.06
                    ay = pcy + ary / a_off * 0.06
                qp = qmul(qb, qy(entry))
                z_entry = PAN_FLOOR + 0.0005 + sinp * c.blade_l / 2  # origin z, tip down
                move_to((ax, ay, RIM + 0.05), qp, 40)
                move_to((ax, ay, z_entry), qp, 40)
                pitch_about_tip(entry, run, 25)  # relax to the run pitch, tip planted
                trace("entry")
                # jab: drive along dirv toward the slice — CAPPED so the slice is never
                # squeezed into the wall face (a wall-press squirted it 60 cm over the
                # rim, measured), and the TIP never leaves the flat interior floor
                p = bread_pos()
                z_run = PAN_FLOOR + 0.0005 + sinr * c.blade_l / 2
                prx, pry = p[0] - pcx, p[1] - pcy
                room = (c.pan_r_floor - r_i - 0.006) - math.hypot(prx, pry)
                adv = min(0.01, max(0.0, room))  # how far the slice may still be pushed
                ex, ey = p[0] + dirv[0] * adv, p[1] + dirv[1] * adv
                tip_r = math.hypot(ex + dirv[0] * cosr * c.blade_l / 2 - pcx,
                                   ey + dirv[1] * cosr * c.blade_l / 2 - pcy)
                over = tip_r - (c.pan_r_floor - 0.004)
                if over > 0:  # pull the end point back along dirv onto the flat floor
                    ex, ey = ex - dirv[0] * over, ey - dirv[1] * over
                move_to((ex, ey, z_run), qmul(qb, qy(run)), jab_steps)
                step(20)
                trace(f"jab{jab_steps}")
                # RETRACT from the wall before any lift: the grippy wall (mu 0.6) holds
                # a pinned slice better than the slick blade (mu 0.15) — lifting in
                # place slid the blade out from under its own cargo (measured). Backing
                # 2.5 cm toward the centre frees the slice; it rides the ramp back.
                move_to((ex - dirv[0] * 0.025, ey - dirv[1] * 0.025, z_run),
                        qmul(qb, qy(run)), 30)
                step(10)
                trace("retracted")
            # CENTERING pass (round-5 lesson): the jab can leave the payload riding the
            # TIP half-overhung — such a load survives the wedge but PIVOTS OVER THE TIP
            # EDGE on the first lift. If the slice is at blade-top height, slide the
            # blade forward under it (gentle relative slide at the working attitude; in
            # the pan the tip skims the floor forward) until the origin sits under it.
            q_work = qmul(qb, qy(run)) if in_pan else qb
            loc = scene._bread_in_blade_frame()[0, i]
            bottom = float(loc[2]) - h_i / 2
            if (0.001 < bottom < 0.018
                    and abs(float(loc[1])) < c.blade_w / 2 + 0.01
                    and float(loc[0]) < c.blade_l / 2 + r_i):
                p = bread_pos()
                cx_t, cy_t = p[0], p[1]
                if in_pan:
                    # same tip cap as the jab: an uncapped slide toward a wall-adjacent
                    # slice ran the tip 1.5 cm up the wall base (measured, pt 0)
                    tipx = cx_t + dirv[0] * cosr * c.blade_l / 2
                    tipy = cy_t + dirv[1] * cosr * c.blade_l / 2
                    over = math.hypot(tipx - pcx, tipy - pcy) - (c.pan_r_floor - 0.004)
                    if over > 0:
                        cx_t, cy_t = cx_t - dirv[0] * over, cy_t - dirv[1] * over
                move_to((cx_t, cy_t, cur["pos"][2]), q_work, 40)
                step(20)
            trace("centered")
            # FUSED lift + level: rise to altitude while easing the pitch to zero about
            # the ORIGIN. The 14 deg run ramp sits 4 deg under the friction angle — a
            # marginal hold this thick slice rocks off if the ramp persists through the
            # lift (measured: +124 mm sideways slide) — but the ramp flattens within the
            # first centimetres of rise, so the static margin grows as the cargo climbs.
            z0h, z1h = cur["pos"][2], RIM + 0.06
            a0h = run if in_pan else 0.0
            for t in range(70):
                f = ease((t + 1) / 70)
                hold((cur["pos"][0], cur["pos"][1], z0h + (z1h - z0h) * f),
                     qmul(qb, qy(a0h * (1.0 - f))))
                step(1)
            step(15)
            trace("lifted-level")
            if bool((scene.on_blade() & scene._present)[0].any()):
                return True
            # already high and level; re-aim — a failed jab may have moved the slice
        return False

    def lift(dz: float, steps: int = 90) -> None:
        # slow by default: ~1.9 mm/step peak — a fast lift is the tip-over lever arm's
        # best friend when the payload sits anywhere forward of centre
        move_to((cur["pos"][0], cur["pos"][1], cur["pos"][2] + dz), cur["quat"], steps)

    # Flip retry ladder: every rung is a WRIST SNAP (roll capped at ~95 deg — the
    # geometric ceiling above which the back-arm dives through the pan wall; see the
    # --flip_roll_deg help). The flip rotation comes from the SPIN the slice inherits
    # at departure, so escalation = snappier (fewer steps => more rad/s), not deeper.
    FLIP_LADDER = ((args.flip_height, args.flip_roll_deg, args.flip_steps),
                   (0.065, 95.0, 11),
                   (0.07, 95.0, 9))

    def flip_once(height: float, roll_deg: float, steps: int) -> None:
        """One commit move: carry the loaded blade OVER THE PAN CENTRE (the landing zone
        is the pan — `flipped` honestly demands a landing IN it), hover at `height`
        above the pan floor, roll the blade about its own long axis to `roll_deg` in
        `steps`; the slice departs over the low edge and physics decides. The blade is
        biased sideways so the landing centres, and the hold height tracks the swinging
        low edge (rolling about the blade-bottom origin would otherwise drive that edge
        into the pan floor). The handle clears the rim by construction (measured: at
        hover 0.02 the handle crosses the rim line 3+ mm above the crest, and every
        later waypoint is higher)."""
        qb = base_q()
        perp = (-tool["dir"][1], tool["dir"][0])  # blade-frame +y in the world
        pcx, pcy = pan_xy()  # live centre — aim the drop at where the bowl actually is
        # (a +0.035 dirv shift was tried to move the rolling NECK's arc off the rim
        # band — it only handed the band to the next handle segment, the back-arm
        # crosses the 2.6 cm annulus SOMEWHERE at any origin, and the shifted landing
        # broke the flip. Reverted; the roll's brief band transit is a measured,
        # reported property — see the clearance check.)
        bx, by = pcx + perp[0] * 0.02, pcy + perp[1] * 0.02
        move_to((cur["pos"][0], cur["pos"][1], RIM + 0.06), qb, 40)
        move_to((bx, by, RIM + 0.06), qb, 110)
        move_to((bx, by, PAN_FLOOR + height), qb, 40)
        px, py = cur["pos"][0], cur["pos"][1]
        trace(f"pre-roll h={height:.3f}")
        marks = {steps // 3, (2 * steps) // 3, steps - 1}  # mid-roll departure probes
        for t in range(steps):
            f = ease((t + 1) / steps)
            ang = roll_deg * f
            edge_drop = (c.blade_w / 2) * math.sin(math.radians(min(ang, 179.0)))
            z = PAN_FLOOR + max(height, 0.008 + edge_drop)
            hold((px, py, z), qmul(qb, qx(ang)))
            step(1)
            if t in marks:
                trace(f"roll@{ang:.0f}deg")
        step(25)
        trace("post-roll")
        # retreat UP at the rolled orientation first (un-rolling low would sweep the
        # blade through the landing zone), THEN unroll IN PLACE over the bowl, THEN
        # retreat level. The old blended retreat (translate + descend + unroll in one
        # move) carried the still-rolled neck — which points DOWN mid-roll — through
        # the rim band at near-table height: measured +72 mm into the wall, pt 6.
        move_to((px, py, RIM + 0.09), cur["quat"], 30)
        move_to((px, py, RIM + 0.09), qb, 25)
        move_to((px - tool["dir"][0] * 0.18, py - tool["dir"][1] * 0.18,
                 RIM + 0.05), qb, 40)

    def carry_and_tip() -> bool:
        """Level carry to the plate, then pitch the blade to slide the bread off into the
        plate's recess. The traverse is SLOW (the friction budget: at a combined mu ~0.29
        the max no-slip horizontal acceleration is ~2.8 m/s^2), and the tip-off happens
        low so the payload drops, not falls. Returns False if the cargo was lost in
        transit (tipping an empty blade helps nobody — the caller re-wedges instead)."""
        pl = plate_pos()
        dirv, qb = tool["dir"], base_q()
        tx, ty = pl[0] - dirv[0] * 0.03, pl[1] - dirv[1] * 0.03
        move_to((cur["pos"][0], cur["pos"][1], c.surface_z + 0.14), qb, 40)
        move_to((tx, ty, c.surface_z + 0.14), qb, 190)
        trace("carry-hi")
        if not bool((scene.on_blade() & scene._present)[0].any()):
            return False  # cargo lost in transit (measured: one marginal seed)
        # descend INSIDE the dish to ~2 cm above the recess floor: the descent path
        # stays clear of both the RIM ring (the target is well inside it) and the
        # rising DISH SLOPE (at recess+8 mm the slope peeled the payload off the
        # blade; from rim height the 4 cm tumble-drop landed it face-down — both
        # measured). From 2 cm the slide-off drops ~1.5 cm: no tumble energy.
        low_z = c.plate_rest_z + 0.020
        move_to((tx, ty, low_z), qb, 60)
        trace("carry-low")
        for t in range(260):  # slow: a snappy pitch corner-spins a light slice into a
            # tumble down the dish, sometimes clean over onto its face (measured)
            f = ease((t + 1) / 260)
            ang = args.tip_deg * f
            # pitching about the blade-bottom origin swings the tip DOWN by sin(ang)*l/2;
            # raise the hold in step so the tip skims just above the recess throughout
            z = c.plate_rest_z + 0.018 + math.sin(math.radians(ang)) * (c.blade_l / 2)
            hold((tx, ty, z), qmul(qb, qy(ang)))
            step(1)
        step(30)
        trace("tipped")
        move_to((pl[0] - dirv[0] * 0.22, pl[1] - dirv[1] * 0.22, c.surface_z + 0.12), qb, 60)
        return True

    def set_down() -> None:
        # high traverse FIRST, then a vertical descent: a single descending move from
        # the serve pose dragged the neck straight through the pan wall en route
        # (measured +72 mm by the clearance monitor, steps 747-781)
        move_to((c.spatula_pos[0], c.spatula_pos[1], c.surface_z + 0.16), LEVEL, 60)
        move_to((c.spatula_pos[0], c.spatula_pos[1], c.surface_z + 0.004), LEVEL, 40)
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
    check("v2 wedge: bread rides the blade", ok_w)
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
    check("v2 rubric transition 50 (flipped, at rest in the pan)",
          int(scene.score()[0]) == 50)

    ok_w2 = wedge()
    check("v2 re-wedge for the serve leg", ok_w2)
    lift(0.12)
    trace("lifted-v2")
    check("v2 rubric transition 70 (loaded after the flip)",
          settle_until(lambda: int(scene.score()[0]) == 70))
    check("v2 loaded-but-not-served is NOT success", not bool(scene.success()[0]))

    if not carry_and_tip():  # cargo lost in transit: recover and try once more
        print("[smoke]   cargo lost in transit — re-wedging for the serve", flush=True)
        if wedge():
            lift(0.12)
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
    check(f"tool never enters the pan wall (worst margin {pen_stat['worst'] * 1000:+.1f} mm "
          f"at pt {pen_stat['idx']} step {pen_stat['step']}, "
          f"{pen_stat['steps']} steps beyond tolerance)", pen_stat["worst"] <= 0.002)

    if args.demo:  # deliverable video = the one clean chain above; stop here
        if frames:
            arr = np.stack(frames, axis=0)
            np.savez_compressed(args.out, frames=arr, env="puzzle.spatula")
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
        # goal `serve` demands the spawn face UP on the plate, and a rough jab sometimes
        # lands the slice inverted (measured: ~1 seed in 6) — do what a cook does:
        # notice, flip it back in the pan, wedge again, then serve.
        i0 = pi()
        if float(scene._bread_up()[0, i0, 2]) < 0.0 and scene.cfg.goal == "serve":
            print("[smoke]   slice inverted by the wedge — flipping it back", flush=True)
            for fh, fr, fs in FLIP_LADDER:
                flip_once(fh, fr, fs)
                step(60)
                if float(scene._bread_up()[0, i0, 2]) > 0.0:
                    break
            wedge()
        lift(0.12)
        trace("lifted")
        ok60 = settle_until(lambda: int(scene.score()[0]) == 60)
        if not ok60:
            diagnose("serve-no-60")
        if not carry_and_tip():  # cargo lost in transit: recover and try once more
            print("[smoke]   cargo lost in transit — re-wedging for the serve", flush=True)
            if wedge():
                lift(0.12)
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

    # size-sampling episode: prove the rubric judges the SAMPLED bread (any of 3 sizes)
    scene.cfg.goal = "serve"
    scene.cfg.sample_size = True
    torch.manual_seed(7)
    env.reset()
    step(40)
    print(f"[smoke] size-sampled episode: present bread = {names[pi()]}", flush=True)
    ok = serve_chain()
    report("size-sample")
    check(f"size-sampled serve ({names[pi()]}) reaches success", ok)
    scene.cfg.sample_size = False

    # =========================== 4. negative control A (teleport-serve) =====================
    env.reset()
    step(40)
    pl = plate_pos()
    i = pi()
    st = make_state((pl[0], pl[1], c.plate_rest_z + c.bread_h(i) / 2 + 0.001))
    scene.breads[names[i]].write_root_state_to_sim(st, all_ids)
    env.iscene.update(0.0)
    step(80)
    report("teleport")
    check("teleported bread: on the plate but NOT served (arrival-on-blade clause)",
          bool(scene.on_plate()[0, i]) and not bool(scene._served[0]))
    check("teleported bread: score pinned at 0", int(scene.score()[0]) == 0)
    check("teleported bread: success rejected", not bool(scene.success()[0]))

    # =========================== 5. negative control B (above / on edge) ====================
    env.reset()
    step(40)
    pickup()
    p = bread_pos()
    # blade hovering ABOVE the bread at rim height: near in xy, never under — the
    # on-blade z band must reject it (the pan wall forbids the old beside-at-floor probe)
    move_to((p[0], p[1], RIM + 0.02), LEVEL, 60)
    step(40)
    check("blade above the bread: wedged NOT latched", not bool(scene._wedged[0]))
    set_down()
    # bread ON EDGE in the pan (90 deg, resting on its crust rim): neither flipped nor flat
    i = pi()
    r_i = c.bread_r(i)
    st = make_state((*pan_xy(), PAN_FLOOR + r_i + 0.001), qx(90.0))
    scene.breads[names[i]].write_root_state_to_sim(st, all_ids)
    env.iscene.update(0.0)
    check("bread on edge: flipped_now rejects (under-rotation)",
          not bool(scene.flipped_now()[0, i]))
    check("bread on edge: not 'resting flat' in the pan",
          not bool(scene.on_pan()[0, i]))
    step(80)  # let it topple wherever physics likes; nothing asserted about the aftermath

    # =========================== 6. negative control C (over-tilt spill) ====================
    env.reset()
    step(40)
    scene.cfg.goal = "serve"
    pickup()
    wedge()
    lift(0.15, steps=110)
    trace("spill-lifted")
    # carry off the pan over the bare bench, re-aim the blade SOUTH (the wedge may have
    # left it pointing at the plate, and one 65-deg fling landed the slice IN the recess
    # — a legitimate serve that inverted this control's assertions; measured), then roll
    # far past any plausible budget
    move_to((0.02, -0.22, c.surface_z + 0.16), base_q(), 130)
    tool["dir"], tool["yaw"] = (0.0, -1.0), -math.pi / 2
    move_to(cur["pos"], base_q(), 50)
    trace("spill-carry")
    for t in range(150):  # SLOW: a 50-step roll catapulted this slice 30 cm onto the
        # plate (a random-direction launch no aim fixes — measured); a slow roll drops
        # it straight down to the bare bench, which is all this control needs
        f = ease((t + 1) / 150)
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
    # 8-deg spacing brackets it); the fine grid belongs to the full feasibility spike.
    # Reported as rates; the knee (last all-seeds tilt) is PUBLISHED.
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
    # Statistical assertions only (raw friction physics; the pen-holder sweep precedent): a
    # level carry must be reliable and SOME nonzero tilt must hold — the exact knee is measured.
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
        np.savez_compressed(args.out, frames=arr, env="puzzle.spatula")
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

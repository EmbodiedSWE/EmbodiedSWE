"""Smoke / rubric-REJECTION battery for MicrowaveBallastDoorScene (sim_gen task
`libero_kitchen_scene6_close_the_microwave_i114`) — NullRobot, teleported probe
states + a push-torque probe, RECORDED.

This is NOT a solution (solve.py — transport-only mug teleports, the door closed by
pure ballast mechanism physics — is the acceptance evidence that the rubric ACCEPTS
a correct outcome; it passes on seeds 0/1). Every teleport here is instrumentation
that CONSTRUCTS a wrong (or partial) outcome and asserts the rubric REJECTS it.
Probes that pose the door move the WHOLE linkage (door + hanging tray + any tray
cargo) in one batched write before stepping — teleporting one body of a linkage gets
depenetrated back. No probe in this battery ever reaches success() at a judged
point, and a final audit asserts exactly that.

  1-2. settle/no-NaN     — reset layout settles finite: door at the ~85 deg open
                           stop, both mugs on the floor, tray empty; score ~0, no
                           success;
  3-4. randomization     — READBACK over 6 seeded resets: mug floor spots jitter,
                           the mugs SWAP SIDES (both signs of y seen), yaws vary;
                           every reset spawns door-open and tray-empty;
  5.  null policy        — 240 idle steps -> score ~0, no success;
  6.  anti-push probe    — the SEED task's strategy (push the door shut by hand),
                           physically refuted: a 2.2 N m hand-torque really slams
                           the EMPTY door flush (< 6 deg — the actuator is
                           non-vacuous) yet earns NOTHING while held (both mug
                           clauses fail), and on release the counterweight swings
                           the door back to the open stop on its own;
  7.  one mug            — a single mug transported into the tray: the door STAYS at
                           the open stop (statics condition B live), first-mug
                           milestone only (score ~0.20), no success;
  8.  latched credit     — lifting that mug back OUT leaves the latched 0.20 while
                           the tray is empty again; no success;
  9.  mugs in chamber    — both mugs set INSIDE the microwave chamber instead of the
                           tray: not ballast — door stays open, score ~0;
  10. mugs under tray    — both mugs on the GROUND directly below the hanging tray
                           (xy inside the footprint, z far below): the 3-D
                           containment box rejects them, score ~0;
  11. gates              — the full linkage CONSTRUCTED at 10 deg with both mugs in
                           the tray, judged mid-motion: outside the closed window
                           and not settled -> NOT success (window + settle gates
                           real);
  12. empty reopen       — the ballast then removed mid-swing: the empty door
                           returns to the open stop from near-flush (the
                           push-and-release cheat fails from ANY angle); the swing
                           credit stays latched (0.70) but success never fires;
  13. rejection audit    — success() was never True at ANY judged point;
  14. final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene6_close_the_microwave_i114.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version and silently rejects RTX -> the
# annotator returns EMPTY frames. Disable the driver check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math
import os
import threading

import numpy as np
import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

PUSH_TAU = 2.2       # N m — the "hand push" on the door (beats the counterweight)
MUG_SLOT_Y = 0.048   # pan-frame y of each mug slot
HOVER_LZ = -0.050    # pan-frame hover release height (solve's transport pose)
REST_LZ = -0.0605    # pan-frame z of a mug standing on the tray floor


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.microwave_ballast_door")().build(num_envs=args.num_envs,
                                                            device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        # In front of and beside the microwave: the raised door, the hanging tray,
        # and both floor mugs are all in frame.
        env.sim.set_camera_view(tuple(np.array((-0.85, -0.95, 0.85)) + o),
                                tuple(np.array((0.42, 0.00, 0.32)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action)
            if (annot is not None and step_i % args.record_every == 0
                    and len(frames) < args.max_frames):
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    ever_success = [False]

    def judge() -> tuple[float, bool]:
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return s, ok

    def door_deg() -> float:
        return math.degrees(float(scene.door_angle()[0]))

    def report(tag: str) -> None:
        s, ok = judge()
        la = scene._mug_local(scene.mug_a)[0]
        lb = scene._mug_local(scene.mug_b)[0]
        print(f"[smoke] {tag:16s} | door={door_deg():+7.1f}deg "
              f"in_pan={int(scene.mugs_in_pan()[0])} "
              f"a=({float(la[0]):+.3f},{float(la[1]):+.3f},{float(la[2]):+.3f}) "
              f"b=({float(lb[0]):+.3f},{float(lb[1]):+.3f},{float(lb[2]):+.3f}) "
              f"settled={bool(scene.settled()[0])} "
              f"L=({float(scene.latch_one[0]):.0f},{float(scene.latch_two[0]):.0f},"
              f"{float(scene.latch_swing[0]):.0f}) "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, xyz, quat=None, ang_vel=None) -> None:
        """One-body kinematic write (NO stepping here — batch linkage writes first)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor(xyz, device=device)
        st[:, 3:7] = torch.tensor(quat if quat is not None else (1.0, 0.0, 0.0, 0.0),
                                  device=device)
        if ang_vel is not None:
            st[:, 10:13] = torch.tensor(ang_vel, device=device)
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def mug_to_tray(body, side: float, lz: float) -> None:
        """Write one mug at a tray slot in the LIVE pan frame (solve's transport)."""
        from isaaclab.utils.math import quat_apply

        local = torch.tensor([0.0, side * MUG_SLOT_Y, lz], device=device).expand(n, 3)
        world = scene.pan.data.root_pos_w + quat_apply(scene.pan.data.root_quat_w,
                                                       local)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = world
        st[:, 3] = 1.0
        body.write_root_state_to_sim(st, all_ids)

    def pose_linkage(phi_deg: float, mugs_in: bool) -> None:
        """Probe-only: write the WHOLE door+tray(+cargo) linkage at door angle
        `phi_deg` in one batch before any stepping (one-body teleports of a linkage
        get depenetrated back)."""
        phi = math.radians(phi_deg)
        sx, sy = c.shell_pos
        hinge = (sx + c.front_x, sy, c.hinge_z)
        place(scene.door, hinge,
              quat=(math.cos(phi / 2), 0.0, math.sin(phi / 2), 0.0))
        px, pz = c.pin_local[0], c.pin_local[2]
        pin = (hinge[0] + px * math.cos(phi) + pz * math.sin(phi), sy,
               hinge[2] - px * math.sin(phi) + pz * math.cos(phi))
        place(scene.pan, pin)
        if mugs_in:
            place(scene.mug_a, (pin[0], pin[1] + MUG_SLOT_Y, pin[2] + REST_LZ))
            place(scene.mug_b, (pin[0], pin[1] - MUG_SLOT_Y, pin[2] + REST_LZ))

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    bodies = (scene.shell, scene.door, scene.pan, scene.mug_a, scene.mug_b)
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("settle: all states finite; door at the open stop "
          f"({door_deg():.1f} deg >= 78), tray empty, both mugs on the floor",
          fin and door_deg() >= 78.0 and int(scene.mugs_in_pan()[0]) == 0)
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    sane = True
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(30)
        pa = scene.mug_a.data.root_pos_w[0] - scene.env_origins[0]
        qa = scene.mug_a.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(qa[3]), float(qa[0]))
        sane = sane and door_deg() >= 78.0 and int(scene.mugs_in_pan()[0]) == 0
        reads.append((float(pa[0]), float(pa[1]), yaw))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (mugA x, y, yaw):\n{np.round(arr, 3)}",
          flush=True)
    check("randomization: the mugs swap sides across seeds (both signs of mug-A y "
          f"seen: {np.round(arr[:, 1], 2).tolist()})",
          bool((arr[:, 1] > 0.1).any()) and bool((arr[:, 1] < -0.1).any()))
    check("randomization: floor jitter and yaw vary "
          f"(x spread {arr[:, 0].ptp():.3f} m, yaw spread {arr[:, 2].ptp():.2f} rad), "
          "and every reset spawns door-open with an empty tray",
          arr[:, 0].ptp() > 0.02 and arr[:, 2].ptp() > 1.0 and sane)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps (mugs on the "
          "floor, door resting at the open stop)", s <= 0.02 and not ok)

    # =========================== 6. anti-push probe (the seed's strategy) ===================
    # The seed task closes the door by pushing it with the hand. Push the EMPTY door
    # flush with a hand-scale external torque, hold, judge, release: the probe must
    # really close the door (non-vacuous), earn nothing, and the counterweight must
    # reopen it on its own.
    t3 = torch.zeros(n, 1, 3, device=device)
    z3 = torch.zeros(n, 1, 3, device=device)
    for _ in range(300):
        t3[:, 0, 1] = -PUSH_TAU  # closing torque about the hinge axis (y)
        scene.door.set_external_force_and_torque(z3, t3, env_ids=all_ids)
        step(1)
    held_deg = door_deg()
    report("push-held")
    s_held, ok_held = judge()
    scene.door.set_external_force_and_torque(z3, z3, env_ids=all_ids)
    step(600)
    report("push-released")
    s, ok = judge()
    check("anti-push: a 2.2 N m hand-torque really pushed the empty door flush "
          f"({held_deg:+.1f} deg < 6 — the probe is non-vacuous) yet earned NOTHING "
          "while held; released, the counterweight reopened the door to "
          f"{door_deg():.1f} deg (>= 78) — the seed's push strategy is physically "
          "refuted",
          held_deg < 6.0 and s_held <= 0.02 and not ok_held
          and door_deg() >= 78.0 and s <= 0.02 and not ok)

    # =========================== 7. one mug is not enough ===================================
    env.reset(seed=41)
    step(60)
    mug_to_tray(scene.mug_a, +1.0, HOVER_LZ)
    step(420)  # land + one-mug statics: the counterweight must hold the open stop
    report("one-mug")
    s_one, ok = judge()
    check("one mug: a single mug riding the tray does NOT close the door "
          f"({door_deg():.1f} deg >= 70, statics condition B live); first-mug "
          f"milestone only (score {s_one:.2f} in [0.19, 0.21]), no success",
          door_deg() >= 70.0 and int(scene.mugs_in_pan()[0]) == 1
          and 0.19 <= s_one <= 0.21 and not ok)

    # =========================== 8. latched credit survives =================================
    place(scene.mug_a, (0.30, -0.55, c.mug_h / 2 + 0.002))
    step(240)
    report("mug-removed")
    s, ok = judge()
    check("latched credit: lifting the mug back OUT leaves the latched score "
          f"unchanged ({s_one:.2f} -> {s:.2f}) with the tray empty again; no success",
          abs(s - s_one) < 1e-3 and int(scene.mugs_in_pan()[0]) == 0 and not ok)

    # =========================== 9. mugs in the chamber are not ballast =====================
    env.reset(seed=51)
    step(60)
    sx, sy = c.shell_pos
    zc = c.floor_top_z + 0.012 + c.mug_h / 2 + 0.002  # standing on the plate
    place(scene.mug_a, (sx - 0.06, sy + 0.06, zc))
    place(scene.mug_b, (sx - 0.06, sy - 0.06, zc))
    step(240)
    report("mugs-in-chamber")
    s, ok = judge()
    check("mugs in chamber: both mugs set INSIDE the microwave instead of the tray "
          f"are not ballast — door stays at {door_deg():.1f} deg (>= 78), "
          "score ~0, no success",
          door_deg() >= 78.0 and int(scene.mugs_in_pan()[0]) == 0
          and s <= 0.02 and not ok)

    # =========================== 10. mugs under the tray (3-D containment) ==================
    env.reset(seed=61)
    step(60)
    pin_w = scene.pan.data.root_pos_w[0] - scene.env_origins[0]
    ux, uy = float(pin_w[0]), float(pin_w[1])
    place(scene.mug_a, (ux, uy + 0.06, c.mug_h / 2 + 0.002))
    place(scene.mug_b, (ux, uy - 0.06, c.mug_h / 2 + 0.002))
    step(240)
    report("mugs-under-tray")
    la = scene._mug_local(scene.mug_a)[0]
    s, ok = judge()
    check("mugs under tray: both mugs on the GROUND directly below the hanging tray "
          f"(mug-A pan-frame z {float(la[2]):+.3f} m, far below the containment "
          "floor) are rejected by the 3-D box — score ~0, no success",
          float(la[2]) < c.pan_lz[0] - 0.1 and int(scene.mugs_in_pan()[0]) == 0
          and s <= 0.02 and not ok)

    # =========================== 11-12. window + settle gates; empty reopen =================
    env.reset(seed=71)
    step(60)
    pose_linkage(10.0, mugs_in=True)
    step(2)
    report("built-at-10deg")
    s, ok = judge()
    gate_ok = (door_deg() > c.door_closed_deg and int(scene.mugs_in_pan()[0]) == 2
               and not bool(scene.settled()[0]) and not ok)
    check("gates: the linkage CONSTRUCTED at 10 deg with both mugs in the tray, "
          f"judged mid-motion ({door_deg():.1f} deg, settled="
          f"{bool(scene.settled()[0])}) is NOT success — the closed window and the "
          "settle streak both gate", gate_ok)
    # remove the ballast BEFORE the two-mug mechanism can finish a real close
    step(4)
    place(scene.mug_a, (0.30, -0.55, c.mug_h / 2 + 0.002))
    place(scene.mug_b, (0.30, +0.55, c.mug_h / 2 + 0.002))
    step(600)
    report("ballast-removed")
    s, ok = judge()
    check("empty reopen: with the ballast removed near flush, the counterweight "
          f"reopened the door to {door_deg():.1f} deg (>= 78) — push-and-release "
          f"fails from ANY angle; the swing credit stays latched ({s:.2f} = 0.70) "
          "but success never fired",
          door_deg() >= 78.0 and 0.69 <= s <= 0.71 and not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.microwave_ballast_door")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(okc for _nm, okc in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, okc in checks:
            if not okc:
                print(f"[smoke]   FAILED: {nm}", flush=True)
    code = 0 if all_ok else 1
    # Hard exit: Kit teardown hangs — watchdog then die.
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)

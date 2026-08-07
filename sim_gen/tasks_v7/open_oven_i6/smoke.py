"""Smoke / rubric-REJECTION battery for DiceTumbleScene (sim_gen task
`open_oven_i6`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — tip each die with a ramped high push until the pad's
color faces up, then slide it low onto the pad — is the acceptance evidence that the
rubric ACCEPTS a correct outcome; it passes on seeds 0/1/2). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled state and
asserts the rubric REJECTS it — plus applied-force probes that prove the two verbs of
the mechanism are physically real (a LOW push slides the die with its face preserved; a
HIGH push tips it over an edge and changes the face). No probe in this battery ever
reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite and still on the floor, score
                            ~0 at rest, and NEITHER die spawns with EITHER pad color up;
  3-4. randomization      — READBACK over 6 seeded resets: the sampled pad-color pair
                            varies; pad positions jitter; the dice's positions and
                            spawn up-faces vary (every reset also re-verified to spawn
                            no pad color face-up);
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  naive transport     — the seed family's move (carry the object to the goal
                            WITHOUT reorienting it): die_a teleported to the pad center
                            keeping its spawn orientation -> wrong face up, scores ~0;
  7.  wrong face          — die_a centered on pad0 but showing pad1's color up ->
                            pad0 NOT served, only stray show credit;
  8.  off-center          — right face up but 9 cm from the pad center (outside the
                            5.5 cm per-axis window, inside near_r) -> NOT served,
                            only show+near credit;
  9.  stacked die         — die_b correctly on pad0, die_a stacked ON TOP with the
                            same color up: the top die reads ~100 mm too high and its
                            on_pad is rejected by the height window;
  10. partial (accepts)   — ONE pad properly served by a settled teleport-constructed
                            die -> served(0) True (the predicate can accept) but NOT
                            success, score < 0.9;
  11. settle gate         — the same served die given 0.4 m/s -> while moving it is
                            NOT served (velocity gates are real);
  12. latched credit      — teleporting the served die far off the pad leaves the
                            latched score unchanged (credit does not evaporate) while
                            served() correctly drops;
  13. mechanism: slide    — a LOW velocity-regulated push (effective height ~20 mm)
                            moves the die > 8 cm with its up-face PRESERVED;
  14. mechanism: tip      — a HIGH ramped push (edge-height couple) changes the
                            up-face: the quarter-turn tumble is contact physics, not
                            scripting;
  15. rejection audit     — success() was never True at ANY judged point;
  16. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.open_oven_i6.smoke --headless
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
# RTX recipe: kit mis-decodes the L20 driver version and silently rejects RTX -> the
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


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.dice_tumble")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(3, device=device)
    face_names = [nm for nm, _col in c.palette]

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.10, -0.90, 0.80)) + o),
                                tuple(np.array((0.20, 0.00, 0.05)) + o),
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

    def up_face(nm: str) -> int:
        return int(scene._face_dots(nm)[0].argmax())

    def die_xy(nm: str) -> torch.Tensor:
        return (scene.dice[nm].data.root_pos_w - scene.env_origins)[0, :2]

    def pad_center(p: int) -> torch.Tensor:
        pp, _pq = scene._pad_state(p)
        return (pp - scene.env_origins)[0]

    def pad_col(p: int) -> int:
        return int(scene.pad_color[0, p])

    def report(tag: str) -> None:
        s, ok = judge()
        bits = []
        for nm in scene.die_names:
            xy = die_xy(nm)
            bits.append(f"{nm}=({float(xy[0]):+.3f},{float(xy[1]):+.3f})"
                        f"up={face_names[up_face(nm)]}")
        bits.append(f"pads=({face_names[pad_col(0)]},{face_names[pad_col(1)]})")
        print(f"[smoke] {tag:16s} | " + " ".join(bits)
              + f" score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    # base orientation putting face f up, as in scene.reset (palette order +x..-z)
    r = math.sqrt(0.5)
    q_face = torch.tensor([
        [r, 0.0, -r, 0.0], [r, 0.0, r, 0.0], [r, r, 0.0, 0.0],
        [r, -r, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
    ], device=device)

    def place_die(nm: str, x: float, y: float, face: int | None, yaw: float = 0.0,
                  z: float | None = None, vel=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap). `face=None` keeps the die's
        CURRENT orientation (the naive-transport construction)."""
        from isaaclab.utils.math import quat_mul

        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1] = x, y
        st[:, 2] = (c.die_size / 2 + 0.003) if z is None else z
        if face is None:
            st[:, 3:7] = scene.dice[nm].data.root_quat_w
        else:
            q_yaw = torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]],
                                 device=device)
            st[:, 3:7] = quat_mul(q_yaw, q_face[face].view(1, 4))
        if vel is not None:
            st[:, 7:10] = torch.tensor(vel, device=device)
        st[:, 0:3] += scene.env_origins
        scene.dice[nm].write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def wrench(nm: str, f3: torch.Tensor, t3: torch.Tensor) -> None:
        """World wrench expressed in the die's current link frame (the solve/pc_ram
        convention: is_global=True silently drops the torque on this stack)."""
        from isaaclab.utils.math import quat_apply_inverse

        body = scene.dice[nm]
        q = body.data.root_link_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f3.view(1, 3).expand(n, 3)).unsqueeze(1),
            quat_apply_inverse(q, t3.view(1, 3).expand(n, 3)).unsqueeze(1),
            env_ids=all_ids)

    def spawn_honest(tag: str) -> bool:
        """Neither die shows either pad color face-up (the reset guarantee)."""
        ok = all(up_face(nm) not in (pad_col(0), pad_col(1)) for nm in scene.die_names)
        if not ok:
            print(f"[smoke] spawn honesty VIOLATION at {tag}", flush=True)
        return ok

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all())
              for b in (*scene.dice.values(), *scene.pads.values()))
    z_ok = all(abs(float(scene.dice[nm].data.root_pos_w[0, 2]) - c.die_size / 2) < 0.01
               for nm in scene.die_names)
    check("settle: all states finite, dice at rest flat on the floor",
          fin and z_ok and all(bool(scene.settled(nm)[0]) for nm in scene.die_names))
    s, ok = judge()
    check("settle: score ~0 at reset, no success, and neither die spawns with either "
          "pad color up", s <= 0.02 and not ok and spawn_honest("reset"))

    # =========================== 3-4. randomization is real =================================
    reads = []
    honest = True
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(20)
        honest = honest and spawn_honest(f"seed {sd}")
        p0 = pad_center(0)
        axy = die_xy("die_a")
        reads.append((pad_col(0), pad_col(1), float(p0[0]), float(p0[1]),
                      float(axy[0]), float(axy[1]), up_face("die_a"), up_face("die_b")))
    arr = np.array(reads)
    print("[smoke] randomization readback (pc0, pc1, pad0_x, pad0_y, die_a_x, die_a_y, "
          f"upA, upB):\n{arr}", flush=True)
    pairs = {(int(a), int(b)) for a, b, *_ in reads}
    check("randomization: the sampled pad-color pair varies across seeded resets "
          f"(readback: {len(pairs)} distinct pairs / 6)", len(pairs) >= 3)
    spread = arr.max(axis=0) - arr.min(axis=0)
    ufaces = {int(x) for x in arr[:, 6]} | {int(x) for x in arr[:, 7]}
    check("randomization: pad position jitters, die position and spawn up-face vary "
          "(readback), and every reset spawns with no pad color face-up",
          spread[2] > 0.01 and spread[4] > 0.01 and len(ufaces) >= 3 and honest)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. naive transport (the seed family's move) ================
    # The seed family's plan is pick-and-place: carry the object to the goal WITHOUT
    # reorienting it. Teleport die_a to pad0's center KEEPING its spawn orientation:
    # the up-face (by construction not a pad color) is wrong -> nothing counts.
    env.reset(seed=41)
    step(30)
    p0 = pad_center(0)
    place_die("die_a", float(p0[0]), float(p0[1]), face=None, settle_steps=45)
    report("naive-transport")
    s, ok = judge()
    check("naive transport (seed's move, no reorientation): die centered on the pad "
          "with its spawn face up -> NOT served, score ~0",
          not bool(scene.served(0)[0]) and not ok and s <= 0.02)

    # =========================== 7. wrong face on the pad ===================================
    env.reset(seed=51)
    step(30)
    p0 = pad_center(0)
    place_die("die_a", float(p0[0]), float(p0[1]), face=pad_col(1), yaw=0.35)
    report("wrong-face")
    s, ok = judge()
    check("wrong face: die centered on pad0 showing pad1's color -> pad0 NOT served, "
          "no placed credit (score <= 0.15)",
          not bool(scene.served(0)[0]) and not ok and s <= 0.15)

    # =========================== 8. off-center ==============================================
    env.reset(seed=61)
    step(30)
    p0 = pad_center(0)
    place_die("die_a", float(p0[0]) + 0.09, float(p0[1]), face=pad_col(0), yaw=0.2)
    report("off-center")
    s, ok = judge()
    check("off-center: right face up but 9 cm from the pad center -> NOT served "
          "(window is 5.5 cm per axis), only show+near credit (score <= 0.25)",
          not bool(scene.served(0)[0]) and not ok and s <= 0.25)

    # =========================== 9. stacked die =============================================
    env.reset(seed=71)
    step(30)
    p0 = pad_center(0)
    place_die("die_b", float(p0[0]), float(p0[1]), face=pad_col(0), settle_steps=30)
    place_die("die_a", float(p0[0]), float(p0[1]), face=pad_col(0),
              z=c.die_size * 1.5 + 0.006, settle_steps=90)
    report("stacked")
    top_z = float(scene.dice["die_a"].data.root_pos_w[0, 2])
    s, ok = judge()
    check("stacked: die_a rests ON die_b over the pad with the right color up "
          f"(z={top_z:.3f}) yet its on_pad is rejected by the height window",
          top_z > 0.12 and not bool(scene.on_pad("die_a", 0)[0]) and not ok)

    # =========================== 10. partial: one pad served (accepts) ======================
    env.reset(seed=81)
    step(30)
    p0 = pad_center(0)
    place_die("die_a", float(p0[0]), float(p0[1]), face=pad_col(0), yaw=0.3,
              settle_steps=60)
    report("partial")
    s, ok = judge()
    served0 = bool(scene.served(0)[0])
    check("partial: ONE pad properly served (predicate accepts a correct single "
          "placement) but success needs both -> NOT success, score < 0.9",
          served0 and not ok and 0.30 <= s < 0.9)

    # =========================== 11. settle gate: moving die not served =====================
    xy = die_xy("die_a")
    place_die("die_a", float(xy[0]), float(xy[1]), face=pad_col(0),
              vel=(0.4, 0.0, 0.0), settle_steps=2)
    v_now = float(scene.dice["die_a"].data.root_lin_vel_w[0].norm())
    moving_not_served = v_now > c.settle_lin and not bool(scene.served(0)[0])
    _s, _ok = judge()
    step(60)  # let it come to rest again
    report("settle-gate")
    check("settle gate: the same die moving at "
          f"{v_now:.2f} m/s on the pad is NOT served (velocity gates are real)",
          moving_not_served)

    # =========================== 12. latched credit survives moving away ====================
    s_before, _ = judge()
    off = next(f for f in range(6) if f not in (pad_col(0), pad_col(1)))  # no pad color
    place_die("die_a", -0.25, 0.30, face=off, settle_steps=40)
    report("moved-away")
    s_after, ok = judge()
    check("latched credit: teleporting the served die far off the pad leaves the "
          "latched score unchanged while served() drops",
          abs(s_after - s_before) < 1e-3 and not bool(scene.served(0)[0]) and not ok)

    # =========================== 13. mechanism: LOW push slides =============================
    env.reset(seed=91)
    step(30)
    face0 = up_face("die_a")
    x0 = die_xy("die_a").clone()
    f3 = torch.zeros(3, device=device)
    t3 = torch.zeros(3, device=device)
    for _ in range(150):  # velocity-regulated push toward -x, effective height ~20 mm
        v = scene.dice["die_a"].data.root_lin_vel_w[0, :2]
        f2 = 30.0 * (torch.tensor([-0.20, 0.0], device=device) - v)
        fn = float(f2.norm())
        if fn > 4.5:
            f2 = f2 * (4.5 / fn)
        elif fn < 3.4 and float(v.norm()) < 0.02:
            f2 = f2 * (3.4 / max(fn, 1e-6))
        f3[0], f3[1] = f2[0], f2[1]
        t3[0], t3[1] = 0.030 * f3[1], -0.030 * f3[0]  # tau = (-0.030 z) x F
        wrench("die_a", f3, t3)
        env.step(no_action)
    wrench("die_a", zero3, zero3)
    step(60)
    dx = float((die_xy("die_a") - x0)[0])
    report("mech-slide")
    check("mechanism (slide): a LOW push moved the die "
          f"{abs(dx) * 100:.0f} cm with its up-face PRESERVED "
          f"({face_names[face0]})", dx < -0.08 and up_face("die_a") == face0)

    # =========================== 14. mechanism: HIGH push tips ==============================
    face0 = up_face("die_a")
    for i in range(360):  # solve's ramp: force + edge-height couple, cut on face change
        f = 1.8 + 1.1 * min(i / 240.0, 1.0)
        f3[0], f3[1] = f, 0.0
        t3[0], t3[1] = 0.0, 0.045 * f
        wrench("die_a", f3, t3)
        env.step(no_action)
        if up_face("die_a") != face0:
            break
    wrench("die_a", zero3, zero3)
    step(90)
    report("mech-tip")
    check("mechanism (tip): the HIGH ramped push tumbled the die — up-face changed "
          f"({face_names[face0]} -> {face_names[up_face('die_a')]}) by contact "
          "physics, not scripting", up_face("die_a") != face0)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all())
              for b in (*scene.dice.values(), *scene.pads.values()))
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.dice_tumble")
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

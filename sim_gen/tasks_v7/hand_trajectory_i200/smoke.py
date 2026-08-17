"""smoke — REJECTION battery for the ChannelRunScene rubric (NullRobot, RECORDED).

This module is NOT a solution (solve.py — the contact-only push run — already proves the
rubric ACCEPTS the correct outcome). Every check here CONSTRUCTS a wrong strategy or a
near-miss (teleports are instrumentation) and asserts the rubric REJECTS it. success()
is audited at every judged point and must NEVER be True anywhere in this battery:

  1. settle/no-NaN     — reset settles finite, puck standing in the start bay, score 0;
  2. randomization     — READBACK: track position/yaw and puck spawn differ across seeds;
  3. mirror            — READBACK: t_mir takes both values across seeds AND the physical
                         pieces are actually mirrored (north-leg floor flips side);
  4. null policy       — 240 idle steps -> score 0, nothing latched;
  5. seed strategy     — the seed's plan (CARRY the object through the air along the exact
                         waypoint route): the puck is pinned at every checkpoint xy in
                         perfect route order but ABOVE the walls -> zero latches, score 0
                         (the probe verifies its own altitude so the z-gate is really hit);
  6. air-drop          — dropping the puck from above the pocket cannot enter it (the roof
                         is in the way): never in_well, score 0;
  7. pocket shortcut   — the puck placed INSIDE the pocket with no route latched: in_well
                         reads True (positive control of the geometric gate) but delivered
                         never latches, score 0, no success;
  8. order enforcement — floor placements at cp3, cp5 latch NOTHING without predecessors;
                         cp1 then latches alone (0.09); cp3 still refuses (cp2 missing);
  9. partial shortcut  — with only cp1 latched, the puck put in the pocket still delivers
                         nothing (score stays 0.09);
 10. near-miss         — the full route latched on the floor, puck parked at the roof face
                         but never dropped: score 0.45, no success;
 11. latched credit    — teleporting the puck back to the start bay keeps the 0.45;
 12. settle gate       — the delivered configuration MOVING at 0.4 m/s is refused
                         (success False while |v| is high); the puck is yanked out before
                         it can settle, so delivered latches 0.70 but success never fires;
 13. audit             — success() was never True at any judged point, max score 0.70;
 14. final no-NaN; frames.npz saved.

Run (forge): python -u -m simgen_tasks.hand_trajectory_i200.smoke --headless
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the L20/4090 driver version and silently rejects RTX -> the
# annotator returns EMPTY frames. Disable the driver check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.hand_trajectory_i200 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.channel_run")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    ids = torch.zeros(1, dtype=torch.long, device=device)

    # --- recording (viewport rgb annotator, the proven forge mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.42, -1.15, 0.85)) + o),
                                tuple(np.array((0.42, 0.0, 0.05)) + o),
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

    never_success = [True]
    max_score = [0.0]

    def judge() -> tuple[float, bool]:
        """Sample the rubric at a judged point and feed the global audit."""
        s = float(scene.score()[0])
        ok = bool(scene.success()[0])
        never_success[0] &= not ok
        max_score[0] = max(max_score[0], s)
        return s, ok

    def report(tag: str) -> None:
        p = scene.puck_local()[0]
        s, ok = judge()
        print(f"[smoke] {tag:18s} local=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):.3f}) cp={''.join('1' if bool(b) else '0' for b in scene.cp_latch[0])} "
              f"delivered={bool(scene.delivered[0])} in_well={bool(scene.in_well()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def puck_state(x: float, y: float, z: float, vel: tuple = (0.0, 0.0, 0.0)) -> torch.Tensor:
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = scene.local_to_world(
            torch.tensor([[x, y, z]], device=device), ids)
        st[0, 3] = 1.0
        st[0, 7:10] = torch.tensor(vel, device=device)
        return st

    def put_puck(x: float, y: float, z: float | None = None, settle: int = 60,
                 vel: tuple = (0.0, 0.0, 0.0)) -> None:
        """Teleport the puck to a canonical track-local pose (instrumentation only)."""
        if z is None:
            z = c.floor_t + c.puck_h / 2 + 0.002  # standing on the channel floor
        scene.puck.write_root_state_to_sim(puck_state(x, y, z, vel), ids)
        step(settle)

    def pin_puck(x: float, y: float, z: float, n: int) -> None:
        """Hold the puck at a pose for n substeps (rewrite every step, zero velocity)."""
        for _ in range(n):
            scene.puck.write_root_state_to_sim(puck_state(x, y, z), ids)
            step(1)

    z_stand = c.floor_t + c.puck_h / 2  # 0.055: standing on the channel floor
    pocket = (-0.170, c.checkpoints[4][1])  # canonical xy deep inside the pocket

    # =========================== 1. settle / no-NaN =========================================
    env.reset(seed=11)
    step(90)
    report("reset")
    p = scene.puck_local()[0]
    check("settle: finite state, puck standing on the start-leg floor, still",
          bool(torch.isfinite(scene.puck.data.root_state_w).all())
          and abs(float(p[2]) - z_stand) < 0.006
          and c.start_x[0] - 0.02 < float(p[0]) < c.start_x[1] + 0.02
          and bool(scene.settled()[0]))
    check("settle: score 0 and nothing latched at reset",
          float(scene.score()[0]) <= 1e-6 and not bool(scene.cp_latch.any()))

    # =========================== 2./3. randomization + mirror (readback) ====================
    reads, mirs, flip_ok = [], [], []
    for s in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=s)
        step(5)
        p = scene.puck_local()[0]
        reads.append((float(scene.t_pos[0, 0]), float(scene.t_pos[0, 1]),
                      float(scene.t_yaw[0]), float(p[0]), float(p[1])))
        mirs.append(float(scene.t_mir[0]))
        # physical mirror readback: un-rotate (but do NOT un-mirror) the north-leg floor
        # piece's world offset; its side flips with t_mir if the pieces really moved.
        fc = (scene.track["floor_c"].data.root_pos_w - scene.env_origins)[0]
        dx = float(fc[0]) - float(scene.t_pos[0, 0])
        dy = float(fc[1]) - float(scene.t_pos[0, 1])
        cy, sy = math.cos(float(scene.t_yaw[0])), math.sin(float(scene.t_yaw[0]))
        y_norot = -sy * dx + cy * dy
        flip_ok.append(y_norot * float(scene.t_mir[0]) > 0.05)
    arr = np.array(reads)
    spread = arr.max(axis=0) - arr.min(axis=0)
    print(f"[smoke] randomization readback (tx, ty, yaw, puck_x, puck_y):\n{arr}", flush=True)
    print(f"[smoke] mirrors={mirs} physical_flip_consistent={flip_ok}", flush=True)
    check("randomization: track pose and puck spawn vary across seeded resets (readback)",
          spread[0] > 0.01 and spread[1] > 0.01 and spread[2] > 0.05 and spread[3] > 0.02)
    check("mirror: both hands appear across seeds AND the pieces physically flip sides",
          (min(mirs) < 0 < max(mirs)) and all(flip_ok))

    # =========================== 4. null policy =============================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    check("null policy: 240 idle steps -> score 0, nothing latched, no success",
          float(scene.score()[0]) <= 1e-6 and not bool(scene.cp_latch.any())
          and not bool(scene.success()[0]))

    # =========================== 5. seed strategy: aerial carry =============================
    # The seed's whole plan — carry the object through free space along the exact waypoint
    # route, in perfect order. The puck visits every checkpoint xy but 65+ mm above the
    # gate; the probe asserts its own altitude so the rejection cannot be vacuous.
    env.reset(seed=41)
    step(60)
    z_fly = 0.135  # above the wall tops (0.067) and far above cp_zmax (0.070)
    alt_ok = True
    for (cx, cy_) in c.checkpoints:
        pin_puck(cx, cy_, z_fly, 25)
        pl = scene.puck_local()[0]
        alt_ok &= float(pl[2]) > c.cp_zmax + 0.02
    report("aerial-carry")
    check("seed strategy: flying the puck along the exact route latches NOTHING "
          "(z-gate; probe altitude verified)",
          alt_ok and not bool(scene.cp_latch.any()) and float(scene.score()[0]) <= 1e-6)

    # =========================== 6. air-drop over the pocket ================================
    env.reset(seed=51)
    step(60)
    put_puck(pocket[0], pocket[1], z=0.25, settle=300)
    report("air-drop")
    p = scene.puck_local()[0]
    check("air-drop: the pocket cannot be entered from above (roof in the way) — "
          "never in_well, score 0",
          not bool(scene.in_well()[0]) and float(p[2]) > c.well_zmax
          and float(scene.score()[0]) <= 1e-6 and not bool(scene.success()[0]))

    # =========================== 7. pocket shortcut, no route ===============================
    env.reset(seed=61)
    step(60)
    put_puck(pocket[0], pocket[1], z=0.030, settle=120)  # standing on the pocket floor
    report("pocket-shortcut")
    check("pocket shortcut: puck IN the pocket (in_well True — positive control) with no "
          "route latched -> no delivery, score 0, no success",
          bool(scene.in_well()[0]) and not bool(scene.delivered[0])
          and float(scene.score()[0]) <= 1e-6 and not bool(scene.success()[0]))

    # =========================== 8./9. order enforcement + partial shortcut =================
    env.reset(seed=71)
    step(60)
    put_puck(*c.checkpoints[2], settle=30)  # cp3 first: predecessor missing
    put_puck(*c.checkpoints[4], settle=30)  # cp5: same
    s_bad = float(scene.score()[0])
    cp_bad = bool(scene.cp_latch.any())
    put_puck(*c.checkpoints[0], settle=30)  # cp1: the only legal first latch
    s_cp1 = float(scene.score()[0])
    put_puck(*c.checkpoints[2], settle=30)  # cp3 again: cp2 still missing
    s_cp3 = float(scene.score()[0])
    report("order-probe")
    check("order: cp3/cp5 visits latch nothing without predecessors; cp1 alone latches "
          "exactly one; cp3 still refuses",
          not cp_bad and s_bad <= 1e-6 and abs(s_cp1 - 0.09) < 1e-3
          and abs(s_cp3 - 0.09) < 1e-3
          and bool(scene.cp_latch[0, 0]) and not bool(scene.cp_latch[0, 2]))
    put_puck(pocket[0], pocket[1], z=0.030, settle=120)
    report("partial-shortcut")
    check("partial shortcut: pocket entry with only cp1 latched delivers nothing "
          "(score stays 0.09)",
          bool(scene.in_well()[0]) and not bool(scene.delivered[0])
          and abs(float(scene.score()[0]) - 0.09) < 1e-3 and not bool(scene.success()[0]))

    # =========================== 10./11. full-route near-miss + latched credit ==============
    env.reset(seed=81)
    step(60)
    for cp in c.checkpoints:  # the legitimate route order, ON the floor
        put_puck(*cp, settle=30)
    put_puck(-0.10, c.checkpoints[4][1], settle=120)  # parked at the roof face, never dropped
    report("route-no-drop")
    check("near-miss: full route latched but the puck never dropped into the pocket -> "
          "score 0.45, no success",
          bool(scene.cp_latch.all()) and not bool(scene.delivered[0])
          and abs(float(scene.score()[0]) - 0.45) < 1e-3 and not bool(scene.success()[0]))
    put_puck(-0.14, -0.12, settle=60)  # back to the start bay
    report("regressed")
    check("latched credit: pushing the puck back to the start bay keeps the 0.45",
          abs(float(scene.score()[0]) - 0.45) < 1e-3 and not bool(scene.success()[0]))

    # =========================== 12. settle gate ============================================
    # The DELIVERED configuration, moving: full route already latched; the puck is written
    # into the pocket sliding at 0.4 m/s. success() must refuse while it moves; the puck is
    # yanked back out before it can settle, so success never fires (delivered latches 0.70
    # — latched credit — but the live conjunction is never satisfied).
    scene.puck.write_root_state_to_sim(
        puck_state(pocket[0], pocket[1], 0.030, vel=(0.0, 0.35, 0.0)), ids)
    step(1)
    v_now = float(scene.puck.data.root_lin_vel_w[0].norm())
    s_move, ok_move = judge()
    print(f"[smoke] settle-gate probe: |v|={v_now:.2f} in_well={bool(scene.in_well()[0])} "
          f"score={s_move:.3f} success={ok_move}", flush=True)
    check("settle gate: the delivered configuration moving at speed is refused "
          "(delivered latches 0.70, success stays False)",
          v_now > c.settle_speed and not ok_move
          and bool(scene.delivered[0]) and abs(s_move - 0.70) < 1e-3)
    put_puck(-0.14, -0.12, settle=90)  # yank it out before it can settle in the pocket
    report("yanked-out")
    check("no live success after the yank: score capped at 0.70 without the live "
          "in-pocket settled state",
          abs(float(scene.score()[0]) - 0.70) < 1e-3 and not bool(scene.success()[0]))

    # =========================== 13./14. audit + final no-NaN ===============================
    check("audit: success() was never True at any judged point; max score 0.70",
          never_success[0] and max_score[0] <= 0.70 + 1e-3)
    check("final: all puck states finite",
          bool(torch.isfinite(scene.puck.data.root_state_w).all()))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.channel_run")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, ok in checks:
            if not ok:
                print(f"[smoke]   FAILED: {nm}", flush=True)
    code = 0 if all_ok else 1
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()

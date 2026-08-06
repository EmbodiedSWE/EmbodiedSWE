"""Smoke / rubric-REJECTION battery for LetterboxDepositScene (sim_gen task
`put_item_in_drawer_i296`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — the kinematic-held contact-dynamics push through the
one-way flap — is the acceptance evidence that the rubric ACCEPTS a correct outcome).
Every teleport here is instrumentation that CONSTRUCTS a wrong (or partial) outcome as
a settled (or, for the flap clause, deliberately transient) state and asserts the
rubric REJECTS it; no probe in this battery ever reaches success(), and a final audit
check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: flap shut, both parcels
                            outside on the floor, everything still, score ~0 at rest;
  3-4. randomization      — READBACK over 8 seeded resets: green/white spawn-slot
                            assignment flips; per-slot xy jitter, spawn yaw and the
                            initial flap ajar angle all vary;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed strategy       — the end state the seed's plan (open the receptacle, drop
                            the item in from above) produces on a sealed box: the green
                            parcel settled ON TOP of the closed letterbox -> rejected;
  7.  near-miss front     — green parcel settled on the ground against the front face
                            just below the slot -> rejected, score stays low;
  8.  shallow insertion   — green parcel perched nose-on-lip with its CoM outside the
                            support: gravity tips it back OUT to the ground (the
                            under-pushed deposit physically fails) -> rejected;
  9.  wrong object        — the WHITE decoy settled on the cavity floor, green parcel
                            still outside -> rejected, score stays low (color
                            identification is load-bearing);
  10. both inside         — green AND white settled on the cavity floor -> the
                            decoy-exclusion clause rejects (score capped 0.85);
  11. flap-open clause    — green parcel at rest on the cavity floor but the flap held
                            open (transient probe, judged before the flap can fall
                            shut) -> rejected; the parcel is removed again before the
                            flap closes so the battery never constructs success;
  12. flap-open aftermath — the flap falls shut by itself, parcel back outside ->
                            still no success;
  13. latched credit      — after transit+deposit latches are earned, yanking the
                            parcel back OUT leaves the latched score unchanged (credit
                            does not evaporate), still no success;
  14. rejection audit     — success() was never True at ANY judged point;
  15. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.put_item_in_drawer_i296.smoke --headless
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
    env = ENVS.get("simgen.letterbox_deposit")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.05, -1.30, 0.85)) + o),
                                tuple(np.array((0.55, 0.00, 0.15)) + o),
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

    def flap_deg() -> float:
        return math.degrees(float(scene.flap_open()[0]))

    def report(tag: str) -> None:
        p = (scene.parcel.data.root_pos_w - scene.env_origins)[0]
        d = (scene.decoy.data.root_pos_w - scene.env_origins)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | parcel=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):.3f}) decoy=({float(d[0]):+.3f},{float(d[1]):+.3f},"
              f"{float(d[2]):.3f}) flap={flap_deg():+.1f}deg "
              f"p_in={bool(scene._contained(scene.parcel)[0])} "
              f"d_in={bool(scene._contained(scene.decoy)[0])} "
              f"app={float(scene._app_max[0]):.3f} pushed={bool(scene._flap_pushed[0])} "
              f"transit={bool(scene._in[0])} dep={bool(scene._dep[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, x: float, y: float, z: float, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += env.iscene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def write_flap(open_deg: float) -> None:
        """Follower-only re-pose of the flap about its unchanged hinge (probe
        constructor for the flap-open clause)."""
        half = -math.radians(open_deg) / 2
        hw = scene._hinge_world()[0]
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = float(hw[0]), float(hw[1]), float(hw[2])
        st[:, 3] = math.cos(half)
        st[:, 5] = math.sin(half)
        st[:, 0:3] += env.iscene.env_origins
        scene.flap.write_root_state_to_sim(st, all_ids)

    def obj_xy(body) -> tuple[float, float]:
        p = (body.data.root_pos_w - scene.env_origins)[0]
        return float(p[0]), float(p[1])

    bx, by = c.box_pos
    floor_rest_z = c.box_z + c.parcel_h / 2 + 0.002  # parcel resting on the cavity floor

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    fin0 = (torch.isfinite(scene.parcel.data.root_state_w).all()
            and torch.isfinite(scene.decoy.data.root_state_w).all()
            and torch.isfinite(scene.flap.data.root_state_w).all())
    still = (float(scene.parcel.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.decoy.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    check("settle: states finite, flap fallen shut, both parcels outside on the floor, "
          "everything still",
          bool(fin0) and abs(flap_deg()) <= c.flap_closed_deg and still
          and not bool(scene._contained(scene.parcel)[0])
          and not bool(scene._contained(scene.decoy)[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        ajar0 = flap_deg()
        step(2)
        px, py = obj_xy(scene.parcel)
        q = scene.parcel.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        da = math.hypot(px - c.spawn_a[0], py - c.spawn_a[1])
        db = math.hypot(px - c.spawn_b[0], py - c.spawn_b[1])
        reads.append((px, py, math.degrees(yaw), ajar0, 1.0 if da < db else 0.0))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (parcel_x, parcel_y, yaw_deg, flap_ajar_deg, "
          f"green_in_slot_a):\n{arr}", flush=True)
    flags = arr[:, 4]
    check("randomization: green/white spawn-slot assignment flips across seeded resets "
          "(readback)", 0.0 < flags.mean() < 1.0)
    jit = 0.0
    for flag in (0.0, 1.0):
        grp = arr[flags == flag]
        if len(grp) >= 2:
            jit = max(jit, float((grp[:, 0:2].max(axis=0) - grp[:, 0:2].min(axis=0)).max()))
    yaw_spread = float(arr[:, 2].max() - arr[:, 2].min())
    ajar_spread = float(arr[:, 3].max() - arr[:, 3].min())
    check("randomization: per-slot xy jitter, spawn yaw and initial flap ajar angle all "
          "vary (readback)", jit > 0.004 and yaw_spread > 5.0 and ajar_spread > 1.5)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy: drop onto the receptacle =================
    # The seed's plan — expose the receptacle's open volume and drop the item in from
    # above — lands the item ON TOP of this sealed box. Constructed settled: rejected.
    torch.manual_seed(41)
    env.reset()
    step(10)
    place(scene.parcel, bx, by, c.box_z + c.in_h + c.t + c.parcel_h / 2 + 0.003)
    step(60)
    report("seed-strategy")
    s, ok = judge()
    pz = float((scene.parcel.data.root_pos_w - scene.env_origins)[0, 2])
    check("seed strategy: green parcel settled ON TOP of the closed letterbox (the "
          "drop-from-above end state) — no success, score < 0.5",
          pz > c.in_h and not bool(scene._contained(scene.parcel)[0])
          and not ok and s < 0.5)

    # =========================== 7. near-miss: against the front face =======================
    torch.manual_seed(51)
    env.reset()
    step(10)
    front_x = bx - c.in_d / 2 - c.t
    place(scene.parcel, front_x - c.parcel_d / 2 - 0.012, by, c.parcel_h / 2 + 0.002)
    step(60)
    report("near-miss-front")
    s, ok = judge()
    check("near-miss: green parcel settled on the ground against the front face just "
          "below the slot — no success, score <= 0.3",
          not bool(scene._contained(scene.parcel)[0]) and not ok and s <= 0.3)

    # =========================== 8. shallow insertion falls back out ========================
    # Nose over the lip but CoM outside the support: the under-pushed deposit tips back
    # out under gravity and lands in front of the box — physically rejected.
    torch.manual_seed(61)
    env.reset()
    step(10)
    lip_z = c.box_z + c.slot_z0 + c.parcel_h / 2 + 0.004
    place(scene.parcel, front_x - c.parcel_d / 2 + 0.008, by, lip_z)
    step(150)
    report("shallow-insert")
    s, ok = judge()
    pz = float((scene.parcel.data.root_pos_w - scene.env_origins)[0, 2])
    px = float((scene.parcel.data.root_pos_w - scene.env_origins)[0, 0])
    check("shallow insertion: parcel perched nose-on-lip with CoM outside tips back OUT "
          "to the ground — no success, parcel not contained",
          not bool(scene._contained(scene.parcel)[0]) and not ok
          and pz < 0.12 and px < front_x)

    # =========================== 9. wrong object: the white decoy ===========================
    torch.manual_seed(71)
    env.reset()
    step(10)
    place(scene.decoy, bx + 0.03, by - 0.04, floor_rest_z)
    step(60)
    report("wrong-object")
    s, ok = judge()
    check("wrong object: WHITE decoy settled on the cavity floor, green parcel outside — "
          "no success, score <= 0.3 (color identification is load-bearing)",
          bool(scene._contained(scene.decoy)[0])
          and not bool(scene._contained(scene.parcel)[0]) and not ok and s <= 0.3)

    # =========================== 10. both inside ============================================
    torch.manual_seed(81)
    env.reset()
    step(10)
    place(scene.parcel, bx - 0.04, by + 0.05, floor_rest_z)
    place(scene.decoy, bx + 0.05, by - 0.05, floor_rest_z)
    step(60)
    report("both-inside")
    s, ok = judge()
    check("both inside: green AND white settled on the cavity floor — the decoy-exclusion "
          "clause rejects, no success, score <= 0.85",
          bool(scene._contained(scene.parcel)[0]) and bool(scene._contained(scene.decoy)[0])
          and not ok and s <= 0.85)

    # =========================== 11. flap-open clause (transient probe) =====================
    # Green parcel at rest on the cavity floor but the flap held open: judged one step
    # after construction (a 70 deg pendulum cannot fall shut in 8 ms) — the flap clause
    # must reject. The parcel is then removed BEFORE the flap can close, so the battery
    # never constructs a success state.
    torch.manual_seed(91)
    env.reset()
    step(10)
    place(scene.parcel, bx - 0.02, by, floor_rest_z)
    write_flap(70.0)
    step(1)
    report("flap-open")
    s, ok = judge()
    open_now = flap_deg()
    check("flap-open clause: parcel at rest on the cavity floor but the flap still open "
          f"({open_now:.0f} deg) — no success (the flap must be left re-closed)",
          bool(scene._contained(scene.parcel)[0]) and open_now > c.flap_open_min_deg
          and not ok)
    place(scene.parcel, 0.25, 0.30, c.parcel_h / 2 + 0.002)  # remove before the flap shuts
    step(120)
    report("flap-reclosed")
    s, ok = judge()
    check("flap-open clause aftermath: flap falls shut by itself, parcel back outside — "
          "still no success", abs(flap_deg()) <= c.flap_closed_deg and not ok)

    # =========================== 12. latched credit survives regression =====================
    torch.manual_seed(101)
    env.reset()
    step(10)
    place(scene.parcel, bx, by, 0.10)  # mid-air inside the cavity: transit+deposit latch
    step(3)
    place(scene.parcel, 0.30, -0.30, c.parcel_h / 2 + 0.002)  # yank it back out
    step(60)
    report("regressed")
    s_a, ok_a = judge()
    step(60)
    report("regressed2")
    s_b, ok_b = judge()
    check("latched credit: transit/deposit latches earned then the parcel yanked back "
          f"outside — score holds ({s_a:.3f} -> {s_b:.3f}), still no success",
          bool(scene._in[0]) and abs(s_a - s_b) < 1e-3 and s_a >= 0.5
          and not ok_a and not ok_b)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.parcel.data.root_state_w).all()
           and torch.isfinite(scene.decoy.data.root_state_w).all()
           and torch.isfinite(scene.flap.data.root_state_w).all()
           and torch.isfinite(scene.box.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.letterbox_deposit")
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
    # Hard exit: Kit teardown hangs — watchdog then die.
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()

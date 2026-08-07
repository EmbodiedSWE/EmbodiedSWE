"""Smoke / rubric-REJECTION battery for RollInGarageScene (sim_gen task
`living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket_i292`) — NullRobot,
teleported probe states, RECORDED.

This is NOT a solution (solve.py — post transported clear, ketchup force-rolled
through the doorway and over the crest — is the acceptance evidence that the rubric
ACCEPTS a correct outcome). Every teleport here is instrumentation that CONSTRUCTS a
wrong (or partial) outcome as a settled state and asserts the rubric REJECTS it; no
probe in this battery ever reaches success(), and a final audit check asserts exactly
that.

  1-2. settle/no-NaN     — reset layout settles finite: post standing in the doorway
                           (blocking), bottles lying at their slots, all still, score ~0;
  3-4. randomization     — READBACK over 8 seeded resets: ketchup/mustard slot
                           assignment flips (Bernoulli side swap), per-slot xy jitter
                           and garage yaw spread are real;
  5.  null policy        — 240 idle steps -> score ~0, no success;
  6.  seed strategy      — the end state the seed's plan produces here (carry the
                           bottle over the receptacle and release): ketchup settled ON
                           THE ROOF -> score ~0, no success (the roof is the interlock);
  7.  near-miss crest    — post cleared, ketchup abandoned on the ramp slope short of
                           the crest: the ramp rejects it back out of the doorway ->
                           NOT success, score <= 0.55 (entered credit only — the crest
                           is the load-bearing tolerance);
  8.  near-miss doorway  — post cleared, ketchup settled centred just OUTSIDE the
                           doorway plane -> NOT success, score < 0.6;
  9.  wrong object       — post still blocking, YELLOW mustard constructed in the
                           landing bay, ketchup untouched -> score ~0, no success
                           (color identification is load-bearing);
  10. exclusion clause   — post cleared, ketchup AND mustard both settled in the bay:
                           every ketchup gate passes, the mustard clause alone rejects
                           -> NOT success, score <= 0.85;
  11. latched credit     — teleporting the ketchup back OUT afterwards leaves the
                           latched score unchanged (credit does not evaporate), still
                           no success;
  12. beside-wall        — ketchup settled against the garage's OUTSIDE wall at an
                           in-range depth (u inside the bay band but v outside) ->
                           NOT success, score < 0.5 (frame math, not world-z, judges
                           containment);
  13. rejection audit    — success() was never True at ANY judged point;
  14. final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.<task>.smoke --headless
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
    env = ENVS.get("simgen.roll_in_garage")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply, quat_mul

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.30, -0.95, 0.75)) + o),
                                tuple(np.array((0.45, 0.00, 0.06)) + o),
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

    def report(tag: str) -> None:
        lo = scene._garage_local(scene.ketchup)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | ketchup_loc=({float(lo[0]):+.3f},{float(lo[1]):+.3f},"
              f"{float(lo[2]):.3f}) blocking={bool(scene._post_blocking()[0])} "
              f"cleared={bool(scene._cleared[0])} app={float(scene._app_max[0]):.3f} "
              f"entered={bool(scene._entered[0])} landed={bool(scene._landed[0])} "
              f"in_bay={bool(scene._in_landing_bay(scene.ketchup)[0])} "
              f"mustard_in={bool(scene._in_garage(scene.mustard)[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_local(body, u: float, v: float, z: float, axis: str = "v") -> None:
        """Teleport `body` to a garage-local point of the garage's CURRENT pose (probe
        constructor: builds inside/on-top relations directly, walls notwithstanding).
        `axis`: lying long-axis direction in the garage frame ("v" or "u")."""
        g_pos = scene.garage.data.root_pos_w
        g_quat = scene.garage.data.root_quat_w
        loc = torch.tensor([u, v, z], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = g_pos + quat_apply(g_quat, loc)
        if axis == "v":  # body +z -> garage +v
            q_rel = torch.tensor([math.cos(-math.pi / 4), math.sin(-math.pi / 4),
                                  0.0, 0.0], device=device).expand(n, 4)
        else:  # body +z -> garage +u
            q_rel = torch.tensor([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4),
                                  0.0], device=device).expand(n, 4)
        st[:, 3:7] = quat_mul(g_quat, q_rel)
        body.write_root_state_to_sim(st, all_ids)

    def place_world(body, x: float, y: float, z: float) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def clear_post() -> None:
        place_world(scene.post, c.park_spot[0], c.park_spot[1], 0.002)

    def obj_xy(body) -> tuple[float, float]:
        p = (body.data.root_pos_w - scene.env_origins)[0]
        return float(p[0]), float(p[1])

    def garage_yaw() -> float:
        q = scene.garage.data.root_quat_w[0]
        return math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))

    def finite_all() -> bool:
        return bool(torch.isfinite(scene.garage.data.root_state_w).all()
                    and torch.isfinite(scene.ketchup.data.root_state_w).all()
                    and torch.isfinite(scene.mustard.data.root_state_w).all()
                    and torch.isfinite(scene.post.data.root_state_w).all())

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("show")
    kz = float((scene.ketchup.data.root_pos_w - scene.env_origins)[0, 2])
    still = (float(scene.ketchup.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.post.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    check("settle: states finite, post standing and BLOCKING the doorway, ketchup lying "
          "at its slot, all still",
          finite_all() and bool(scene._post_blocking()[0])
          and abs(kz - c.body_r) < 0.012 and still
          and not bool(scene._in_garage(scene.ketchup)[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        kx, ky = obj_xy(scene.ketchup)
        mx, my = obj_xy(scene.mustard)
        reads.append((kx, ky, mx, my, 1.0 if ky > 0 else 0.0, garage_yaw()))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (ketchup_x, ketchup_y, mustard_x, mustard_y, "
          f"ketchup_on_left, garage_yaw_deg):\n{arr}", flush=True)
    flags = arr[:, 4]
    check("randomization: ketchup/mustard slot assignment flips across seeded resets AND "
          "the two bottles always take opposite slots (readback)",
          0.0 < flags.mean() < 1.0
          and all((r[1] > 0) != (r[3] > 0) for r in reads))
    jit = 0.0
    for flag in (0.0, 1.0):
        grp = arr[flags == flag]
        if len(grp) >= 2:
            jit = max(jit, float((grp[:, 0:2].max(axis=0) - grp[:, 0:2].min(axis=0)).max()))
    yaw_spread = float(arr[:, 5].max() - arr[:, 5].min())
    check("randomization: per-slot spawn jitter (> 4 mm) and garage yaw spread (> 2 deg) "
          "are real (readback)", jit > 0.004 and yaw_spread > 2.0)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy: release above the receptacle =============
    # The seed's plan — carry the bottle over the receptacle and let go — ends HERE with
    # the bottle resting ON THE ROOF (the roof is the interlock). Post untouched.
    torch.manual_seed(41)
    env.reset()
    step(30)
    roof_top = c.roof_z + c.roof_t
    place_local(scene.ketchup, 0.12, 0.0, roof_top + c.body_r + 0.004)
    step(150)
    report("seed-strategy")
    kz = float(scene._garage_local(scene.ketchup)[0, 2])
    s, ok = judge()
    check("seed strategy: ketchup released over the garage rests ON THE ROOF — score ~0 "
          "(<= 0.05), no success (the roof interlock rejects the seed's plan)",
          kz > c.roof_z and not bool(scene._in_garage(scene.ketchup)[0])
          and not ok and s <= 0.05)

    # =========================== 7. near-miss: abandoned on the ramp slope ==================
    # Post cleared, ketchup left on the ramp SHORT of the crest: the slope rejects it
    # back out of the doorway. Entered credit latches; landing/success must not.
    torch.manual_seed(51)
    env.reset()
    step(30)
    clear_post()
    step(20)
    place_local(scene.ketchup, 0.045, 0.0, 0.042)
    step(300)
    report("near-miss-crest")
    lo = scene._garage_local(scene.ketchup)[0]
    s, ok = judge()
    check("near-miss crest: ketchup abandoned on the ramp short of the crest rolls back "
          "out (ends before the crest), entered-only credit, NOT success, score <= 0.55",
          bool(scene._entered[0]) and float(lo[0]) < c.crest_u
          and not bool(scene._landed[0]) and not ok and s <= 0.55)

    # =========================== 8. near-miss: centred just outside the doorway =============
    torch.manual_seed(61)
    env.reset()
    step(30)
    clear_post()
    step(20)
    place_local(scene.ketchup, -0.045, 0.0, c.body_r + 0.002)
    step(120)
    report("near-miss-door")
    s, ok = judge()
    check("near-miss doorway: ketchup settled centred just OUTSIDE the doorway plane — "
          "NOT success, score < 0.6",
          not bool(scene._in_landing_bay(scene.ketchup)[0]) and not ok and s < 0.6)

    # =========================== 9. wrong object: mustard garaged ===========================
    torch.manual_seed(71)
    env.reset()
    step(30)
    place_local(scene.mustard, 0.15, 0.0, c.body_r + 0.004)
    step(150)
    report("wrong-object")
    s, ok = judge()
    check("wrong object: YELLOW mustard in the landing bay, ketchup untouched, post "
          "still blocking — score ~0 (<= 0.02), no success (color identification is "
          "load-bearing)",
          bool(scene._in_garage(scene.mustard)[0]) and not ok and s <= 0.02)

    # =========================== 10. exclusion clause =======================================
    # Post cleared, ketchup AND mustard both settled in the bay: every ketchup gate
    # passes; the mustard clause alone must reject.
    torch.manual_seed(81)
    env.reset()
    step(30)
    clear_post()
    step(20)
    place_local(scene.mustard, 0.195, 0.0, c.body_r + 0.004)
    place_local(scene.ketchup, 0.115, 0.0, c.body_r + 0.004)
    step(240)
    report("both-garaged")
    s10, ok = judge()
    k_still = float(scene.ketchup.data.root_lin_vel_w[0].norm()) < c.settle_speed
    check("exclusion clause: ketchup AND mustard both settled in the bay — all ketchup "
          "gates pass, the mustard clause alone rejects: NOT success, score <= 0.85",
          bool(scene._in_landing_bay(scene.ketchup)[0])
          and bool(scene._in_garage(scene.mustard)[0]) and k_still
          and not ok and s10 <= 0.85)

    # =========================== 11. latched credit survives regression =====================
    place_world(scene.ketchup, c.slot_a[0], c.slot_a[1], c.body_r + 0.002)
    step(60)
    report("regressed")
    s11, ok = judge()
    check("latched credit: teleporting the ketchup back OUT of the bay leaves the "
          f"latched score unchanged ({s10:.3f} -> {s11:.3f}), still no success",
          abs(s11 - s10) < 1e-3 and not bool(scene._in_landing_bay(scene.ketchup)[0])
          and not ok)

    # =========================== 12. beside-wall (frame math) ===============================
    torch.manual_seed(91)
    env.reset()
    step(30)
    clear_post()
    step(20)
    place_local(scene.ketchup, 0.12, 0.155, c.body_r + 0.002, axis="u")
    step(120)
    report("beside-wall")
    lo = scene._garage_local(scene.ketchup)[0]
    s, ok = judge()
    check("beside-wall: ketchup settled against the garage's OUTSIDE wall at an "
          "in-range depth — v gate rejects: NOT success, score < 0.5",
          abs(float(lo[1])) > c.in_w / 2 and not bool(scene._in_garage(scene.ketchup)[0])
          and not ok and s < 0.5)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.roll_in_garage")
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

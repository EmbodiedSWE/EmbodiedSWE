"""Smoke / rubric-REJECTION battery for MugTipoutScene (sim_gen task `put_banana_i6`) —
NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — the kinematic-held contact-dynamics pour + gravity
set-down — is the acceptance evidence that the rubric ACCEPTS a correct outcome). Every
teleport here is instrumentation that CONSTRUCTS a wrong (or partial) outcome as a
settled state and asserts the rubric REJECTS it; no probe in this battery ever reaches
success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: orange seated inside the mug
                            bore, everything still, score ~0 at rest;
  3-4. randomization      — READBACK over 8 seeded resets: mug spawn + home pad move;
                            green/white dish SLOT ASSIGNMENT flips and per-slot jitter
                            is real;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed strategy       — the seed's terminal relation (fruit INSIDE the mug), mug
                            parked upright on the home pad: every credit gate stays
                            locked -> score ~0, no success;
  7.  container cheat     — the loaded mug (orange still inside) parked standing IN the
                            green dish: the in-mug clause rejects the containment ->
                            score ~0, no success;
  8.  near-miss           — orange settled on the ground just OUTSIDE the green dish
                            wall: NOT success, score < 0.9;
  9.  wrong place         — orange settled inside the WHITE decoy dish: NOT success,
                            score < 0.9 (color identification is load-bearing);
  10. mug off home        — orange in the green dish but the mug upright on the ground
                            12 cm from the pad (tol 4.5 cm): NOT success;
  11. mug tilted          — orange in the green dish but the mug LYING ON ITS SIDE over
                            the pad: the upright + base-height gates reject -> NOT
                            success;
  12. latched credit      — re-seating the orange back INSIDE the mug after the in-dish
                            probes leaves the latched score unchanged (credit does not
                            evaporate), still no success;
  13. rejection audit     — success() was never True at ANY judged point in this
                            battery (rejection-only contract);
  14. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.put_banana_i6.smoke --headless
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
    env = ENVS.get("simgen.mug_tipout_rehome")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.35, -1.05, 0.95)) + o),
                                tuple(np.array((0.47, 0.00, 0.04)) + o),
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
        b = (scene.ball.data.root_pos_w - scene.env_origins)[0]
        m_ = (scene.mug.data.root_pos_w - scene.env_origins)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | ball=({float(b[0]):+.3f},{float(b[1]):+.3f},"
              f"{float(b[2]):.3f}) mug=({float(m_[0]):+.3f},{float(m_[1]):+.3f},"
              f"{float(m_[2]):.3f}) in_mug={bool(scene._in_mug()[0])} "
              f"in_dish={bool(scene._in_dish_now(scene.dish_t)[0])} "
              f"out={bool(scene._out[0])} app={float(scene._app_max[0]):.3f} "
              f"in={bool(scene._in[0])} home={float(scene._home_max[0]):.3f} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_mug(x: float, y: float, z: float, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += env.iscene.env_origins
        scene.mug.write_root_state_to_sim(st, all_ids)

    def place_ball(x: float, y: float, z: float) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = 1.0
        st[:, 0:3] += env.iscene.env_origins
        scene.ball.write_root_state_to_sim(st, all_ids)

    def seat_ball_in_mug() -> None:
        """Teleport the orange to the bore-bottom seat of the mug's CURRENT pose
        (probe constructor: builds the fruit-in-container relation)."""
        from isaaclab.utils.math import quat_apply

        seat = torch.tensor([0.0, 0.0, -c.mug_h / 2 + c.ball_seat_z], device=device)
        pos = scene.mug.data.root_pos_w + quat_apply(
            scene.mug.data.root_quat_w, seat.expand(n, 3))
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3] = 1.0
        scene.ball.write_root_state_to_sim(st, all_ids)

    def obj_xy(body) -> tuple[float, float]:
        p = (body.data.root_pos_w - scene.env_origins)[0]
        return float(p[0]), float(p[1])

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    fin0 = (torch.isfinite(scene.mug.data.root_state_w).all()
            and torch.isfinite(scene.ball.data.root_state_w).all())
    ball_z = float((scene.ball.data.root_pos_w - scene.env_origins)[0, 2])
    still = (float(scene.ball.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.mug.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    check("settle: states finite, orange seated inside the mug bore, everything still",
          bool(fin0) and bool(scene._in_mug()[0]) and ball_z < 0.05 and still)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        mx, my = obj_xy(scene.mug)
        px, py = obj_xy(scene.pad)
        gx, gy = obj_xy(scene.dish_t)
        da = math.hypot(gx - c.dish_slot_a[0], gy - c.dish_slot_a[1])
        db = math.hypot(gx - c.dish_slot_b[0], gy - c.dish_slot_b[1])
        reads.append((mx, my, px, py, gx, gy, 1.0 if da < db else 0.0))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (mug_x, mug_y, pad_x, pad_y, green_x, green_y, "
          f"green_in_slot_a):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: mug spawn xy + home pad xy vary across seeded resets (readback)",
          spread[0] > 0.015 and spread[1] > 0.015 and spread[2] > 0.01 and spread[3] > 0.01)
    slot_flags = arr[:, 6]
    jit = 0.0
    for flag in (0.0, 1.0):
        grp = arr[slot_flags == flag]
        if len(grp) >= 2:
            jit = max(jit, float((grp[:, 4:6].max(axis=0) - grp[:, 4:6].min(axis=0)).max()))
    check("randomization: green/white slot assignment flips across seeds AND per-slot dish "
          "jitter is real (readback)",
          0.0 < slot_flags.mean() < 1.0 and jit > 0.004)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy: fruit in container =======================
    # The seed's terminal relation — the fruit INSIDE the mug — with the mug even parked
    # perfectly on its home pad: every credit term is gated on the orange leaving the
    # bore, so this is worth ~nothing and is not success.
    torch.manual_seed(41)
    env.reset()
    step(10)
    px, py = obj_xy(scene.pad)
    place_mug(px, py, c.pad_h + c.mug_h / 2 + 0.002)
    seat_ball_in_mug()
    step(60)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy: fruit inside the mug, mug parked upright on the home pad — "
          "score ~0 (<= 0.05), no success",
          bool(scene._in_mug()[0]) and bool(scene._mug_home_now()[0])
          and not ok and s <= 0.05)

    # =========================== 7. container cheat: loaded mug in the dish =================
    torch.manual_seed(51)
    env.reset()
    step(10)
    gx, gy = obj_xy(scene.dish_t)
    place_mug(gx, gy, c.dish_floor_t + c.mug_h / 2 + 0.002)
    seat_ball_in_mug()
    step(60)
    report("container-cheat")
    s, ok = judge()
    d_xy = float((scene.ball.data.root_pos_w[0, :2]
                  - scene.dish_t.data.root_pos_w[0, :2]).norm())
    check("container cheat: loaded mug parked standing IN the green dish — orange xy is "
          "over the dish but the in-mug clause rejects it: score ~0 (<= 0.05), no success",
          bool(scene._in_mug()[0]) and d_xy < c.dish_capture_r
          and not bool(scene._in_dish_now(scene.dish_t)[0]) and not ok and s <= 0.05)

    # =========================== 8. near-miss: outside the dish wall ========================
    torch.manual_seed(61)
    env.reset()
    step(10)
    gx, gy = obj_xy(scene.dish_t)
    off = c.dish_inner_r + c.dish_wall_t + c.ball_r + 0.009  # just past the wall
    place_ball(gx + off, gy, c.ball_r + 0.002)
    step(60)
    report("near-miss")
    s, ok = judge()
    check("near-miss: orange settled on the ground just OUTSIDE the green dish wall — "
          "NOT success, score < 0.9", not ok and s < 0.9)

    # =========================== 9. wrong place: the white decoy dish =======================
    torch.manual_seed(71)
    env.reset()
    step(10)
    wx, wy = obj_xy(scene.dish_d)
    place_ball(wx, wy, c.dish_floor_t + c.ball_r + 0.003)
    step(60)
    report("wrong-place")
    s, ok = judge()
    d_dec = float((scene.ball.data.root_pos_w[0, :2]
                   - scene.dish_d.data.root_pos_w[0, :2]).norm())
    check("wrong place: orange settled inside the WHITE decoy dish — NOT success, "
          "score < 0.9 (color identification is load-bearing)",
          d_dec < c.dish_capture_r and not ok and s < 0.9)

    # =========================== 10. mug off home ===========================================
    torch.manual_seed(81)
    env.reset()
    step(10)
    gx, gy = obj_xy(scene.dish_t)
    place_ball(gx, gy, c.dish_floor_t + c.ball_r + 0.003)
    px, py = obj_xy(scene.pad)
    place_mug(px + 0.12, py, c.mug_h / 2 + 0.002)  # upright, on the ground, off the pad
    step(60)
    report("mug-off-home")
    s10, ok = judge()
    check("mug off home: orange IS in the green dish but the mug stands 12 cm from the "
          "pad (tol 4.5 cm) — NOT success",
          bool(scene._in_dish_now(scene.dish_t)[0]) and not ok)

    # =========================== 11. mug lying on its side over the pad =====================
    # (same episode: the orange stays where it is, in the green dish)
    place_mug(px, py, c.mug_inner_r + c.mug_wall_t + 0.002,
              quat=(math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0))
    step(60)
    report("mug-tilted")
    s11, ok = judge()
    from isaaclab.utils.math import quat_apply
    up = quat_apply(scene.mug.data.root_quat_w,
                    torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(n, 3))
    check("mug tilted: orange in the green dish but the mug LYING ON ITS SIDE over the "
          "pad — upright gate rejects, NOT success",
          abs(float(up[0, 2])) < 0.5 and not ok)

    # =========================== 12. latched credit survives regression =====================
    # Re-stand the mug on open ground and put the orange BACK inside the bore: the
    # latched credit must not evaporate, and the regressed state is of course no success.
    place_mug(0.55, 0.12, c.mug_h / 2 + 0.002)
    step(20)
    seat_ball_in_mug()
    step(40)
    report("regressed")
    s12, ok = judge()
    check("latched credit: re-seating the orange inside the mug leaves the latched score "
          f"unchanged ({s11:.3f} -> {s12:.3f}), still no success",
          abs(s12 - s11) < 1e-3 and bool(scene._in_mug()[0]) and not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.mug.data.root_state_w).all()
           and torch.isfinite(scene.ball.data.root_state_w).all()
           and torch.isfinite(scene.dish_t.data.root_state_w).all()
           and torch.isfinite(scene.dish_d.data.root_state_w).all()
           and torch.isfinite(scene.pad.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.mug_tipout_rehome")
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

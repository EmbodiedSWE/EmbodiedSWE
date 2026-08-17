"""Smoke / rubric-REJECTION battery for GriddleSpatulaScene (sim_gen task
`meat_off_grill_i355`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — the wedge/carry/tilt-pour executed through
contact dynamics — is the acceptance evidence that the rubric ACCEPTS a correct
outcome). Every teleport here is instrumentation that CONSTRUCTS a wrong (or
partial) outcome as a settled state and asserts the rubric REJECTS it; no probe in
this battery ever reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN     — reset layout settles finite: patty flat ON the griddle,
                           spatula on the ground, all still, score ~0, no success;
  3-4. randomization     — READBACK over 8 seeded resets: patty griddle-local xy and
                           fixture yaw spreads are real; the dish side FLIPS;
  5.  null policy        — 240 idle steps -> score ~0, no success;
  6.  rim retention      — the patty shoved quasi-statically (velocity-limited CoM
                           force) into the +x rim wall STAYS on the griddle (the
                           slide/plow-off exit does not exist -> the tool is
                           mandatory), score <= 0.16 (engagement only), no success;
  7.  seed strategy      — the end state the SEED's plan produces (lift the meat
                           off, set it beside): patty settled on the ground beside
                           the griddle -> score ~0, no success (a teleported exit
                           earns no engagement credit; only the dish counts);
  8.  near-miss ground   — patty settled on the ground BESIDE the dish -> rejected;
  9.  near-miss wall     — patty straddling the dish wall (not flat on the floor,
                           CoM out of the inside band) -> rejected;
  10. tool-near          — the spatula lying NEXT to the dish (10 cm) while the
                           patty is perfectly served -> the tool-away clause alone
                           rejects (constructed tool-first so no prefix succeeds);
  11. not released       — the loaded spatula RESTING ON the dish (patty still on
                           the blade over the dish) -> rejected;
  12. latched credit     — a scooped state constructed on the ground fires the scoop
                           latch (score ~0.30 < 0.70); removing the patty afterwards
                           leaves the latched score unchanged, still no success;
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
    env = ENVS.get("simgen.griddle_spatula")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.40, -0.85, 0.80)) + o),
                                tuple(np.array((0.42, 0.00, 0.08)) + o),
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
        gp = scene._local(scene.griddle, scene.patty)[0]
        dp = scene._local(scene.dish, scene.patty)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | patty_griddle=({float(gp[0]):+.3f},{float(gp[1]):+.3f},"
              f"{float(gp[2]):+.3f}) patty_dish=({float(dp[0]):+.3f},{float(dp[1]):+.3f},"
              f"{float(dp[2]):+.3f}) on_griddle={bool(scene._on_griddle()[0])} "
              f"scooped={bool(scene._scooped()[0])} in_dish={bool(scene._in_dish()[0])} "
              f"tool_away={bool(scene._tool_away()[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_ref(body, ref, x: float, y: float, z: float, align: bool = True) -> None:
        """Teleport `body` to a point in `ref`'s CURRENT body frame (probe constructor)."""
        loc = torch.tensor([x, y, z], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = ref.data.root_pos_w + quat_apply(ref.data.root_quat_w, loc)
        if align:
            st[:, 3:7] = ref.data.root_quat_w
        else:
            st[:, 3] = 1.0
        body.write_root_state_to_sim(st, all_ids)

    def place_world(body, x: float, y: float, z: float) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def fix_yaw() -> float:
        q = scene.griddle.data.root_quat_w[0]
        return math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))

    def all_bodies():
        return [scene.griddle, scene.dish, scene.patty, scene.spatula]

    def finite_all() -> bool:
        return bool(all(torch.isfinite(b.data.root_state_w).all() for b in all_bodies()))

    def serve_patty() -> None:
        """Probe constructor: drop the patty flat into the dish and settle."""
        place_ref(scene.patty, scene.dish, 0.0, 0.0, 0.045)
        step(120)

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("show")
    still = (float(scene.patty.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.spatula.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    spat_low = float(scene.spatula.data.root_pos_w[0, 2] - scene.env_origins[0, 2]) < 0.05
    check("settle: states finite, patty flat ON the griddle, spatula on the ground, "
          "all still", finite_all() and bool(scene._on_griddle()[0]) and spat_low and still)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        gp = scene._local(scene.griddle, scene.patty)[0]
        reads.append((float(gp[0]), float(gp[1]), fix_yaw(), float(scene._dish_side[0])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (patty_x, patty_y, yaw_deg, dish_side):\n"
          f"{arr}", flush=True)
    x_spread = float(arr[:, 0].max() - arr[:, 0].min())
    y_spread = float(arr[:, 1].max() - arr[:, 1].min())
    yaw_spread = float(arr[:, 2].max() - arr[:, 2].min())
    check("randomization: patty start spreads (x > 15 mm, y > 40 mm) and fixture yaw "
          "spread (> 4 deg) are real (readback)",
          x_spread > 0.015 and y_spread > 0.040 and yaw_spread > 4.0)
    sides = {int(r[3]) for r in reads}
    check("randomization: the dish side FLIPS across seeded resets (both +y and -y "
          "seen, readback)", sides == {-1, 1})

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. rim retention (slide-off exit closed) ===================
    # The scene's core claim: the patty cannot be slid or plowed off the griddle. A
    # quasi-static, velocity-limited CoM force (~10x the sliding-friction need) shoves
    # it into the +x rim wall for 5 s; force off; it must still be ON the griddle.
    torch.manual_seed(31)
    env.reset()
    step(30)
    x_before = float(scene._local(scene.griddle, scene.patty)[0, 0])
    push_dir = quat_apply(scene.griddle.data.root_quat_w,
                          torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))
    for _ in range(600):
        v = float((scene.patty.data.root_lin_vel_w[0] * push_dir[0]).sum())
        f = 1.5 if v < 0.06 else 0.0
        scene.patty.set_external_force_and_torque(
            (push_dir * f).view(n, 1, 3).contiguous(), zero_wrench,
            env_ids=all_ids, is_global=True)
        env.step(no_action)
    scene.patty.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    step(120)
    report("rim-push")
    x_after = float(scene._local(scene.griddle, scene.patty)[0, 0])
    s, ok = judge()
    check("rim retention: patty shoved quasi-statically INTO the +x rim wall moved "
          f"({x_before:+.3f} -> {x_after:+.3f}) yet STAYS on the griddle (no "
          "slide-off exit; the tool is mandatory) — score <= 0.16 (engagement only), "
          "no success",
          x_after > x_before + 0.03 and x_after > 0.10
          and bool(scene._on_griddle()[0]) and s <= 0.16 and not ok)

    # =========================== 7. seed strategy: lift off, set beside =====================
    # The seed's plan — grasp the meat, lift it off the grill, set it down beside —
    # constructed as its end state: the patty settled on the ground beside the
    # griddle. Only the dish counts; a teleported exit earns no engagement credit.
    torch.manual_seed(41)
    env.reset()
    step(30)
    place_ref(scene.patty, scene.griddle, 0.0, -0.30 * float(scene._dish_side[0]),
              c.patty_thick / 2 + 0.004)  # beside the griddle, OPPOSITE the dish
    step(150)
    report("seed-strategy")
    s, ok = judge()
    pz = float(scene.patty.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    check("seed strategy: patty 'lifted off and set beside' the griddle on the "
          "ground — score ~0 (<= 0.05), no success (only the dish counts)",
          not bool(scene._on_griddle()[0]) and pz < 0.02 and s <= 0.05 and not ok)

    # =========================== 8. near-miss: on the ground beside the dish ================
    torch.manual_seed(51)
    env.reset()
    step(30)
    place_ref(scene.patty, scene.dish, 0.16, 0.0, c.patty_thick / 2 + 0.004)
    step(150)
    report("beside-dish")
    dp = scene._local(scene.dish, scene.patty)[0]
    s, ok = judge()
    check("near-miss: patty settled on the ground BESIDE the dish (missed it by a "
          "few cm) — NOT success",
          abs(float(dp[0])) < 0.30 and not bool(scene._in_dish()[0]) and not ok)

    # =========================== 9. near-miss: straddling the dish wall =====================
    torch.manual_seed(61)
    env.reset()
    step(30)
    place_ref(scene.patty, scene.dish, c.dish_inner_half + c.dish_wall_t / 2, 0.0, 0.045)
    step(150)
    report("on-wall")
    dp = scene._local(scene.dish, scene.patty)[0]
    s, ok = judge()
    check("near-miss: patty landed straddling the dish wall (not flat on the dish "
          "floor inside the walls) — NOT success",
          abs(float(dp[0])) < 0.20 and not bool(scene._in_dish()[0]) and not ok)

    # =========================== 10. tool-near: spatula left at the dish ====================
    # Constructed TOOL-FIRST so no prefix state satisfies the goal: the spatula is
    # laid on the ground 10 cm from the dish, THEN the patty is served perfectly.
    torch.manual_seed(71)
    env.reset()
    step(30)
    place_ref(scene.spatula, scene.dish, c.dish_inner_half + 0.10, 0.0, 0.020)
    step(60)
    serve_patty()
    report("tool-near")
    d_tool = float((scene.spatula.data.root_pos_w[0, :2]
                    - scene.dish.data.root_pos_w[0, :2]).norm())
    s, ok = judge()
    check("tool-near: patty perfectly served but the spatula lies "
          f"{d_tool * 100:.0f} cm from the dish (< 20 cm) — the tool-away clause "
          "alone rejects: NOT success",
          bool(scene._in_dish()[0]) and bool(scene._still(scene.patty)[0])
          and d_tool < c.tool_away_r and not ok)

    # =========================== 11. delivered but never released ===========================
    # The loaded spatula parked ON the dish: the patty is at the dish but still on
    # the blade — not resting flat on the dish floor, and the tool is at the dish.
    torch.manual_seed(81)
    env.reset()
    step(30)
    place_ref(scene.spatula, scene.dish, -0.02, 0.0, 0.045)
    step(90)
    place_ref(scene.patty, scene.spatula, 0.0, 0.0,
              c.blade_t / 2 + c.patty_thick / 2 + 0.003)
    step(150)
    report("not-released")
    s, ok = judge()
    check("not released: the loaded spatula rests ON the dish with the patty still "
          "on the blade — NOT success (patty is not flat on the dish floor and the "
          "tool is at the dish)",
          bool(scene._scooped()[0]) and not bool(scene._in_dish()[0]) and not ok)

    # =========================== 12. latched credit =========================================
    # A scooped state constructed on open ground fires the scoop latch; removing the
    # patty afterwards must not reduce the latched score.
    torch.manual_seed(91)
    env.reset()
    step(30)
    place_world(scene.spatula, 1.00, 0.55, 0.020)
    step(60)
    place_ref(scene.patty, scene.spatula, 0.0, 0.0,
              c.blade_t / 2 + c.patty_thick / 2 + 0.003)
    step(90)
    report("scoop-latch")
    s12a, ok = judge()
    scoop_latched = bool(scene._scoop_ever[0])
    place_world(scene.patty, 1.30, -0.60, c.patty_thick / 2 + 0.004)
    step(90)
    report("regressed")
    s12b, ok = judge()
    check("latched credit: a constructed scooped state fires the scoop latch "
          f"(score {s12a:.2f}, within [0.25, 0.40] < 0.70 cap); removing the patty "
          f"leaves the latched score unchanged ({s12a:.3f} -> {s12b:.3f}), still no "
          "success",
          scoop_latched and 0.25 <= s12a <= 0.40 and abs(s12b - s12a) < 1e-3 and not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.griddle_spatula")
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
    try:
        main()
    except BaseException:  # noqa: BLE001 — die loudly instead of idling to the watchdog
        import traceback
        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)

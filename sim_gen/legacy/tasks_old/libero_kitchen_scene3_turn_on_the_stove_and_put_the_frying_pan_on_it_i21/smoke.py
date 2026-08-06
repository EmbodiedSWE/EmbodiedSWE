"""Smoke / oracle test for ShortOrderScene (sim_gen task
`libero_kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it_i21`) —
NullRobot, teleport-oracle, RECORDED.

Battery (pen_holder / tunnel_shuffle smoke skeleton):
  1. settle/no-NaN      — reset layout settles finite, score ~0 at rest;
  2. randomization      — READBACK: stove, plate and patty poses all move across seeded
                          resets (including the y-side layout);
  3. subset readback    — the present patty count varies (2 vs 3) across seeded resets;
  4. null-policy-fails  — 300 idle steps -> score ~0, no success;
  5-7. oracle x3 seeds  — the shuttle oracle (cook each patty to the latch, remove in
                          time, serve) reaches success() and score exactly 1.0;
  8. subset episode     — oracle succeeds when judged on the sampled subset;
  9. monotonicity       — milestone scores along the correct plan strictly increase
                          to 1.0 (reset -> seared -> cooked -> served, per patty);
 10. single-slot A      — of two patties beside the burner only the centred one accrues
                          heat; a tangent symmetric pair accrues none (geometric
                          one-patty capacity);
 11. single-slot B      — a patty stacked on top of the cooking one accrues nothing;
 12. negative A (SEED)  — the seed task's own terminal state, "put it on the hot stove
                          and stop": the patty burns, its credit collapses to the burnt
                          cap, and success is impossible even after serving everything;
 13. negative B         — skipping the stove entirely (raw patties straight onto the
                          plate) scores ~0;
 14. near-miss          — undercooked (60% of the cook window) earns sear credit only,
                          no cooked credit, no success;
 15. calibration probe  — measured substep-per-step rate and cook->burn latch timings
                          consistent with the configured window;
 16. irreversibility    — the burnt latch survives removal from the stove.

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
    from .scene import ShortOrderSceneCfg  # noqa: F401  (import registers the scene)
except ImportError:  # bare-script fallback (forge runs the package module form)
    from scene import ShortOrderSceneCfg  # type: ignore


def _env_of(scene_or_env):
    return scene_or_env if hasattr(scene_or_env, "iscene") else scene_or_env.env


def oracle_solution(scene_or_env, verbose: bool = True, step_fn=None) -> bool:
    """Teleport-oracle: solve the CURRENT episode. For each present patty in turn:
    set it down on the glowing burner disc (a kinematic place — the resting, the heat
    accrual and the settling are then real stepped physics), wait for the cooked latch,
    IMMEDIATELY lift it off to its serving-plate slot, wait for delivery. The dwell
    window is generous (cook 1.5 s, burn 4.0 s) but the oracle still must sequence:
    one patty at a time through the single-slot burner. Returns True iff
    scene.success() holds."""
    env = _env_of(scene_or_env)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=env.device)
    all_ids = torch.arange(env.num_envs, device=env.device)

    def _step(k: int) -> None:
        if step_fn is not None:
            step_fn(k)
        else:
            for _ in range(k):
                env.step(no_action)

    def _wait(pred, max_steps: int, poll: int = 5) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            _step(poll)
            waited += poll
            if pred():
                return True
        return False

    def _place(i: int, x: float, y: float, z: float) -> None:
        st = torch.zeros(env.num_envs, 13, device=env.device)
        st[:, 0] = x
        st[:, 1] = y
        st[:, 2] = z
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        scene.patties[c.patty_names[i]].write_root_state_to_sim(st, all_ids)

    for i in range(len(c.patty_names)):
        if not bool(scene.present[0, i]):
            continue
        bx, by = (float(v) for v in scene.stove_xy[0])
        _place(i, bx, by, c.burner_top + c.patty_h / 2 + 0.003)
        _step(15)
        ok = _wait(lambda i=i: bool(scene.cooked[0, i]), max_steps=8 * c.cook_min_steps)
        if verbose:
            print(f"[oracle] patty_{i}: cooked={bool(scene.cooked[0, i])} "
                  f"steps={int(scene.cook_steps[0, i])} "
                  f"burnt={bool(scene.burnt[0, i])} score={float(scene.score()[0]):.3f}",
                  flush=True)
        if not ok or bool(scene.burnt[0, i]):
            return False
        px, py = (float(v) for v in scene.plate_xy[0])
        ang = math.radians(90.0 + 120.0 * i)
        _place(i, px + 0.038 * math.cos(ang), py + 0.038 * math.sin(ang),
               c.plate_top + c.patty_h / 2 + 0.003)
        _step(15)
        _wait(lambda i=i: bool(scene.delivered()[0, i]), max_steps=200)
    _wait(lambda: bool(scene.success()[0]), max_steps=200)
    return bool(scene.success()[0])


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.short_order")().build(
        num_envs=args.num_envs, device=device,
        scene_cfg=ShortOrderSceneCfg(subset_sample=False))
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    P = len(c.patty_names)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.85, -1.10, 0.95)) + o),
                                tuple(np.array((-0.02, 0.0, 0.10)) + o),
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

    def wait_until(pred, max_steps: int = 400, poll: int = 5) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def place_patty(i: int, x: float, y: float, z: float, settle_steps: int = 20) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = x
        st[:, 1] = y
        st[:, 2] = z
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        scene.patties[c.patty_names[i]].write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def burner_xy() -> tuple:
        return (float(scene.stove_xy[0, 0]), float(scene.stove_xy[0, 1]))

    def plate_slot(i: int) -> tuple:
        px, py = (float(v) for v in scene.plate_xy[0])
        ang = math.radians(90.0 + 120.0 * i)
        return (px + 0.038 * math.cos(ang), py + 0.038 * math.sin(ang))

    def to_burner(i: int, settle_steps: int = 15) -> None:
        bx, by = burner_xy()
        place_patty(i, bx, by, c.burner_top + c.patty_h / 2 + 0.003, settle_steps)

    def to_plate(i: int, settle_steps: int = 25) -> None:
        sx, sy = plate_slot(i)
        place_patty(i, sx, sy, c.plate_top + c.patty_h / 2 + 0.003, settle_steps)

    def cook_and_serve(i: int) -> bool:
        to_burner(i)
        if not wait_until(lambda: bool(scene.cooked[0, i]),
                          max_steps=8 * c.cook_min_steps):
            return False
        to_plate(i)
        return wait_until(lambda: bool(scene.delivered()[0, i]), max_steps=200)

    def report(tag: str) -> None:
        print(f"[smoke] {tag:14s} | present={scene.present[0].int().tolist()} "
              f"steps={scene.cook_steps[0].tolist()} "
              f"seared={scene.seared[0].int().tolist()} "
              f"cooked={scene.cooked[0].int().tolist()} "
              f"burnt={scene.burnt[0].int().tolist()} "
              f"delivered={scene.delivered()[0].int().tolist()} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    # =========================== 1. settle / no-NaN =========================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    pos = torch.stack([b.data.root_state_w for b in scene.patties.values()])
    onboard = all(
        abs(float((b.data.root_pos_w - scene.env_origins)[0, 2])
            - (c.board_top + c.patty_h / 2)) < 0.01
        for b in scene.patties.values())
    check("settle: patty states finite and at rest on the prep board",
          bool(torch.isfinite(pos).all()) and onboard)
    check("settle: score ~0 at reset (<= 0.005)", float(scene.score()[0]) <= 0.005)

    # =========================== 2-3. randomization is real =================================
    reads, counts = [], []
    scene.cfg.subset_sample = True
    for s in (21, 22, 23, 24, 25, 26, 27, 28, 29, 30):
        torch.manual_seed(s)
        env.reset()
        step(3)
        p0 = (scene.patties[c.patty_names[0]].data.root_pos_w - scene.env_origins)[0]
        reads.append((float(scene.stove_xy[0, 0]), float(scene.stove_xy[0, 1]),
                      float(scene.plate_xy[0, 0]), float(scene.plate_xy[0, 1]),
                      float(p0[0]), float(p0[1])))
        counts.append(int(scene.present[0].sum()))
    scene.cfg.subset_sample = False
    arr = np.array(reads)
    spread = arr.max(axis=0) - arr.min(axis=0)
    print(f"[smoke] randomization readback (stove_x, stove_y, plate_x, plate_y, "
          f"p0_x, p0_y):\n{arr}\n[smoke] spreads={spread} counts={counts}", flush=True)
    check("randomization: stove, plate and patty poses all move on reset (readback)",
          spread[0] > 0.02 and spread[1] > 0.03 and spread[3] > 0.03
          and (spread[4] > 0.008 or spread[5] > 0.008))
    check("randomization: present patty count varies across resets (2 vs 3)",
          min(counts) == 2 and max(counts) == 3)

    # =========================== 4. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(300)
    report("null-policy")
    check("null policy: score ~0 and no success after 300 idle steps",
          float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 5-7. oracle on 3 seeds =====================================
    for s in (0, 1, 2):
        torch.manual_seed(s)
        env.reset()
        step(20)
        ok = oracle_solution(env, step_fn=step)
        report(f"oracle-seed{s}")
        check(f"oracle reaches success() on seed {s} (score exactly 1.0)",
              ok and bool(scene.success()[0]) and float(scene.score()[0]) == 1.0)

    # =========================== 8. subset episode ==========================================
    scene.cfg.subset_sample = True
    n_present = P
    for s in range(60, 70):
        torch.manual_seed(s)
        env.reset()
        n_present = int(scene.present[0].sum())
        if n_present == 2:
            break
    step(20)
    ok = oracle_solution(env, step_fn=step)
    absent = [i for i in range(P) if not bool(scene.present[0, i])]
    parked_ok = all(
        float((scene.patties[c.patty_names[i]].data.root_pos_w
               - scene.env_origins)[0, 0]) > 0.6 for i in absent)
    report("subset")
    check(f"subset episode ({n_present}/{P} present): oracle success, "
          f"absent patties stay parked",
          ok and bool(scene.success()[0]) and parked_ok)
    scene.cfg.subset_sample = False

    # =========================== 9. rubric monotonicity =====================================
    torch.manual_seed(100)
    env.reset()
    step(40)
    ms: list[float] = [float(scene.score()[0])]

    def milestone(tag: str) -> None:
        ms.append(float(scene.score()[0]))
        print(f"[smoke]   milestone {tag}: score={ms[-1]:.3f}", flush=True)

    to_burner(0)
    wait_until(lambda: bool(scene.seared[0, 0]), max_steps=200)
    milestone("patty0 seared")
    wait_until(lambda: bool(scene.cooked[0, 0]), max_steps=8 * c.cook_min_steps)
    milestone("patty0 cooked")
    to_plate(0)
    wait_until(lambda: bool(scene.delivered()[0, 0]), max_steps=200)
    milestone("patty0 served")
    cook_and_serve(1)
    # judge on BOTH served patties settled (patty1's landing can transiently unsettle
    # patty0 — wait for the rubric state, don't assert on a clock tick)
    wait_until(lambda: bool(scene.delivered()[0, :2].all()), max_steps=200)
    milestone("patty1 served")
    cook_and_serve(2)
    wait_until(lambda: bool(scene.success()[0]), max_steps=200)
    milestone("patty2 served")
    report("monotonic")
    check("monotonicity: milestone scores strictly increase along the correct plan",
          all(b > a + 1e-4 for a, b in zip(ms, ms[1:])))
    check("monotonicity: final milestone is success with score exactly 1.0",
          bool(scene.success()[0]) and ms[-1] == 1.0)

    # =========================== 10. single-slot A (side by side) ===========================
    torch.manual_seed(110)
    env.reset()
    step(20)
    bx, by = burner_xy()
    to_burner(0, settle_steps=0)
    # beside: flat on the stove body next to the disc (12 mm below the disc top)
    place_patty(1, bx + 0.080, by, c.burner_top + c.patty_h / 2 + 0.003, settle_steps=0)
    step(80)
    beside = (int(scene.cook_steps[0, 0]), int(scene.cook_steps[0, 1]))
    torch.manual_seed(111)
    env.reset()
    step(20)
    bx, by = burner_xy()
    # the closest STABLE non-overlapping pair on the disc: both 34 mm off-axis (any pair
    # both inside the 28 mm cook radius would have to interpenetrate — centres <= 56 mm
    # apart for 60 mm discs)
    place_patty(0, bx + 0.034, by, c.burner_top + c.patty_h / 2 + 0.003, settle_steps=0)
    place_patty(1, bx - 0.034, by, c.burner_top + c.patty_h / 2 + 0.003, settle_steps=0)
    step(80)
    tangent = (int(scene.cook_steps[0, 0]), int(scene.cook_steps[0, 1]))
    print(f"[smoke] single-slot: centred+beside accrual={beside}, "
          f"symmetric tangent pair accrual={tangent}", flush=True)
    check("single-slot: only the centred patty accrues; a tangent pair accrues none",
          beside[0] > 30 and beside[1] == 0 and tangent[0] == 0 and tangent[1] == 0)

    # =========================== 11. single-slot B (stacked) ================================
    torch.manual_seed(112)
    env.reset()
    step(20)
    bx, by = burner_xy()
    to_burner(0, settle_steps=0)
    place_patty(1, bx, by, c.burner_top + 1.5 * c.patty_h + 0.004, settle_steps=0)
    step(80)
    print(f"[smoke] stacked accrual: bottom={int(scene.cook_steps[0, 0])} "
          f"top={int(scene.cook_steps[0, 1])}", flush=True)
    check("single-slot: a patty stacked on the cooking one accrues nothing",
          int(scene.cook_steps[0, 0]) > 30 and int(scene.cook_steps[0, 1]) == 0)

    # =========================== 12. negative A: the seed's own strategy ====================
    # The seed's terminal state — the object resting on the powered stove — left in place:
    # the patty burns, its credit collapses to the burnt cap, and success is impossible
    # even after everything (including the burnt patty) is later put on the plate.
    torch.manual_seed(200)
    env.reset()
    step(20)
    to_burner(0)
    burned = wait_until(lambda: bool(scene.burnt[0, 0]), max_steps=12 * c.burn_steps)
    report("seed-strategy")
    score_after_burn = float(scene.score()[0])
    # Clear the slot FIRST (run 1: teleporting patty1 into the still-occupied burner
    # interpenetrated and ejected it), then cook and serve the remaining two properly.
    to_plate(0)
    ok1 = cook_and_serve(1)
    ok2 = cook_and_serve(2)
    wait_until(lambda: bool(scene.delivered()[0, 1] and scene.delivered()[0, 2]),
               max_steps=200)
    report("seed-aftermath")
    check("negative A (seed strategy): parking on the hot stove burns the patty "
          "and collapses its credit",
          burned and score_after_burn <= 0.06 and bool(scene.cooked[0, 0]))
    check("negative A: success impossible after burning (score capped < 0.8)",
          ok1 and ok2 and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.80)

    # =========================== 13. negative B: skip the stove =============================
    torch.manual_seed(210)
    env.reset()
    step(20)
    for i in range(P):
        to_plate(i, settle_steps=15)
    step(60)
    report("raw-delivery")
    check("negative B: raw patties straight onto the plate score ~0, no success",
          float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 14. near-miss: undercooked =================================
    torch.manual_seed(220)
    env.reset()
    step(20)
    to_burner(0)
    target = int(0.6 * c.cook_min_steps)
    wait_until(lambda: int(scene.cook_steps[0, 0]) >= target,
               max_steps=8 * c.cook_min_steps, poll=3)
    to_plate(0)
    step(40)
    report("near-miss")
    check("near-miss: undercooked patty on the plate — sear credit only, not cooked, "
          "no success",
          bool(scene.seared[0, 0]) and not bool(scene.cooked[0, 0])
          and not bool(scene.delivered()[0, 0])
          and float(scene.score()[0]) <= 0.10 and not bool(scene.success()[0]))

    # =========================== 15. calibration probe ======================================
    # Measure the substeps-per-env-step rate from the accrual counter, then verify both
    # latch timings against the configured window using that measured rate.
    torch.manual_seed(230)
    env.reset()
    step(20)
    to_burner(0)
    c0 = int(scene.cook_steps[0, 0])
    step(50)
    rate = (int(scene.cook_steps[0, 0]) - c0) / 50.0
    k_cook = 0
    while not bool(scene.cooked[0, 0]) and k_cook < 12 * c.cook_min_steps:
        step(5)
        k_cook += 5
    k_burn = 0
    while not bool(scene.burnt[0, 0]) and k_burn < 12 * c.burn_steps:
        step(5)
        k_burn += 5
    exp_burn = (c.burn_steps - c.cook_min_steps) / max(rate, 1e-6)
    print(f"[smoke] calibration: accrual rate={rate:.2f} substeps/env-step, "
          f"cooked after +{k_cook} env-steps, burnt after +{k_burn} more "
          f"(expected burn gap ~{exp_burn:.0f})", flush=True)
    check("calibration: accrual rate sane and latch timings match the configured window",
          0.5 <= rate <= 8.0 and bool(scene.cooked[0, 0]) and bool(scene.burnt[0, 0])
          and 0.6 * exp_burn <= k_burn <= 1.6 * exp_burn + 10)

    # =========================== 16. burnt latch irreversible ===============================
    steps_at_burn = int(scene.cook_steps[0, 0])
    place_patty(0, c.board_x, 0.0, c.board_top + c.patty_h / 2 + 0.003, settle_steps=120)
    report("post-burn")
    check("irreversibility: burnt latch and heat counter survive removal from the stove",
          bool(scene.burnt[0, 0]) and int(scene.cook_steps[0, 0]) == steps_at_burn
          and float(scene.score()[0]) <= 0.05)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.short_order")
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

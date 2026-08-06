"""Smoke / oracle test for ServeIngredientScene — NullRobot, teleport-oracle, RECORDED.

One linear run (pen_holder-smoke skeleton):
  1. show       — settle the reset layout; rubric must read 0; all states finite; then 200
                  idle steps prove the null policy scores ~0 (nothing acts -> nothing counts);
  2. random     — up to 8 seeded resets, READBACK: dish/cup layouts actually move, >= 2
                  distinct source cups get sampled, and the ball physically sits in the
                  sampled cup every time;
  3. oracle     — 3 seeds: kinematically carry the loaded cup over the dish (gravity-
                  compensated hold), POUR — a genuine roll out of the tilted mouth and fall
                  onto the dish floor — then set the cup back down away from the dish;
                  success() required each seed; seed 0 additionally pins the rubric
                  milestones (0 -> escape >= 0.25 -> served-but-cup-overhead 0.85 -> 1.0);
  4. negative A — THE SEED'S OWN STRATEGY: carry the loaded cup (ball riding inside, moved
                  as one rigid assembly) onto the dish. Must NOT succeed and must score ~0
                  (the ball never left its cup — the plan itself is wrong here);
  5. negative B — near miss: ball placed on the table just outside the dish -> exactly the
                  0.25 escape credit, no success;
  6. negative C — dish not cleared: ball served + the empty cup parked inside the dish caps
                  the score at 0.85; removing the cup completes to success (tolerance twin:
                  a served ball 35 mm off-axis counts);
  7. sweep      — calibration: drop the ball from 50 mm above the rim at growing xy offsets
                  (2 seeds each); publishes per-offset capture rates and the capture limit
                  (geometric funnel = dish_inner_r - ball_r = 60 mm). Statistical assertions.

ALWAYS records video via the viewport rgb annotator (RTX driver-version override, 3-render
ghost flush) and saves `frames.npz` in the CURRENT WORKING DIRECTORY. Bodies are driven
straight through scene handles; the NullRobot applies nothing.

Run (on the forge):
    python -u -m simgen_tasks.<task>.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=10)
parser.add_argument("--max_frames", type=int, default=900)
parser.add_argument("--out", type=str, default="frames.npz")  # CWD — the pipeline fetches it
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the L20 driver version and silently rejects RTX -> annotator
# returns EMPTY frames. Disable the check.
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

from .scene import ServeIngredientSceneCfg  # noqa: E402 — also registers scene + env


# ----- the exported teleport-oracle --------------------------------------------------------------
def oracle_solution(env, on_step=None):
    """Solve env 0 by POURING: kinematic gravity-compensated hold of the loaded cup, carry it
    above the dish, tilt to ~130 deg so the ball rolls out of the mouth and falls onto the
    dish floor (real physics — the ball is never pose-set unless a pour retry is needed),
    then set the cup back down at its original slot and release.

    Returns {"success": bool, "retries": int, "milestones": {reset, escape, landed, final,
    landed_success}} — the milestone scores pin the rubric's monotone ladder.
    """
    scene = env.scene
    c = scene.cfg
    dev = env.device
    n = env.num_envs
    all_ids = torch.arange(n, device=dev)
    no_action = torch.empty(0, device=dev)
    g_dt = 9.81 * env.dt
    origin = env.iscene.env_origins[0]

    src = int(scene.src[0])
    cup = scene.cups[src]
    hold = {"st": None}

    def make_state(pos, quat=(1.0, 0.0, 0.0, 0.0)):
        st = torch.zeros(n, 13, device=dev)
        st[:, 0:3] = env.iscene.env_origins + torch.tensor(pos, device=dev)
        st[:, 3:7] = torch.tensor(quat, device=dev)
        return st

    def step(k):
        # While hold["st"] is set, re-pin the cup pre-step with +g*dt of upward velocity so
        # PhysX's gravity integration cancels exactly (the pen_holder held-platform lesson: a
        # zero-velocity re-pin leaves the floor free-falling g*dt every step and the BALL
        # riding it inherits that velocity), then re-pin at zero vel before judging.
        for _ in range(k):
            if hold["st"] is not None:
                pre = hold["st"].clone()
                pre[:, 9] += g_dt
                cup.write_root_state_to_sim(pre, all_ids)
            env.step(no_action, render=False)
            if on_step is not None:
                on_step()
        if hold["st"] is not None:
            cup.write_root_state_to_sim(hold["st"], all_ids)
            env.iscene.update(0.0)

    def settle_until(pred, max_steps=480, poll=10):
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    ms = {"reset": float(scene.score()[0])}
    retries = 0

    p0 = (cup.data.root_pos_w[0] - origin).tolist()
    park = (p0[0], p0[1], c.surface_z + c.cup_h / 2 + 0.003)
    dxy = (scene.dish.data.root_pos_w[0] - origin)[:2].tolist()
    rim_top = c.surface_z + 0.002 + c.dish_h
    hold_z = rim_top + 0.055 + c.cup_h / 2 * 0.0  # cup CENTRE height during the pour
    tilt_max = math.radians(130.0)
    # Tilt about world +y -> the mouth swings toward +x; lean the cup centre back so the
    # mouth (where the ball exits, ~110 deg) ends up over the dish centre.
    lean = (c.cup_h / 2) * math.sin(math.radians(110.0))
    hold_pos = (dxy[0] - lean, dxy[1], hold_z)

    # --- phase A: lift the cup from its slot to the pour station, upright, slowly (the ball
    # rides inside on contact alone — fast holds pop it out) ---
    KA = 180
    for t in range(KA):
        f = (t + 1) / KA
        pos = tuple(p0[i] + (hold_pos[i] - p0[i]) * f for i in range(3))
        hold["st"] = make_state(pos)
        step(1)

    # --- phase B: tilt 0 -> 130 deg; the ball rolls down the wall and out of the mouth ---
    KB = 240
    for t in range(KB):
        th = tilt_max * (t + 1) / KB
        q = (math.cos(th / 2), 0.0, math.sin(th / 2), 0.0)
        hold["st"] = make_state(hold_pos, q)
        step(1)
        if "escape" not in ms and bool(scene.escaped[0]):
            ms["escape"] = float(scene.score()[0])

    # --- phase C: wait for the served ball to settle on the dish floor; teleport-drop retry
    # if the pour misjudged (the oracle may place kinematically; the JUDGE never does) ---
    served = lambda: bool((scene.ball_on_dish() & scene.settled())[0])  # noqa: E731
    if not settle_until(served, 480):
        retries += 1
        print(f"[smoke]   pour retry: teleport-drop above the dish centre", flush=True)
        st = make_state((dxy[0], dxy[1], rim_top + 0.045))
        scene.ball.write_root_state_to_sim(st, all_ids)
        step(150)
        settle_until(served, 300)
    ms.setdefault("escape", float(scene.score()[0]))
    ms["landed"] = float(scene.score()[0])
    ms["landed_success"] = bool(scene.success()[0])

    # --- phase D: set the cup back down at its original slot and release. Split into
    # untilt-in-place -> translate HIGH -> lower at the slot: a straight tilted drag from the
    # pour station would sweep the cup's ~46 mm corner extent through the dish rim. ---
    lift_z = rim_top + 0.090
    for t in range(80):  # D1: untilt + rise, xy fixed over the dish
        f = (t + 1) / 80
        th = tilt_max * (1 - f)
        q = (math.cos(th / 2), 0.0, math.sin(th / 2), 0.0)
        hold["st"] = make_state((hold_pos[0], hold_pos[1],
                                 hold_z + (lift_z - hold_z) * f), q)
        step(1)
    for t in range(120):  # D2: carry upright, high, back over the park slot
        f = (t + 1) / 120
        hold["st"] = make_state((hold_pos[0] + (park[0] - hold_pos[0]) * f,
                                 hold_pos[1] + (park[1] - hold_pos[1]) * f, lift_z))
        step(1)
    for t in range(80):  # D3: lower to the surface
        f = (t + 1) / 80
        hold["st"] = make_state((park[0], park[1], lift_z + (park[2] - lift_z) * f))
        step(1)
    hold["st"] = None
    step(60)
    settle_until(lambda: bool(scene.success()[0]), 300)
    ms["final"] = float(scene.score()[0])
    return {"success": bool(scene.success()[0]), "retries": retries, "milestones": ms}


# ----- the battery --------------------------------------------------------------------------------
def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.serve_ingredient")().build(
        num_envs=args.num_envs, device=device, scene_cfg=ServeIngredientSceneCfg())
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.15, -1.15, 0.85)) + o),
                                tuple(np.array((0.05, 0.0, c.surface_z + 0.10)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames — check RTX recipe", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    tick = {"i": 0}

    def rec() -> None:
        if annot is not None and tick["i"] % args.record_every == 0 \
                and len(frames) < args.max_frames:
            for _ in range(3):  # flush accumulated history (ghosting fix)
                env.sim.render()
            arr = np.asarray(annot.get_data())
            if arr.size:
                frames.append(arr[..., :3].astype(np.uint8).copy())
        tick["i"] += 1

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action, render=False)
            rec()

    def settle_until(pred, max_steps=300, poll=10) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def make_state(pos, quat=(1.0, 0.0, 0.0, 0.0)) -> torch.Tensor:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = env.iscene.env_origins + torch.tensor(pos, device=device)
        st[:, 3:7] = torch.tensor(quat, device=device)
        return st

    def report(tag: str) -> None:
        print(f"[smoke] {tag:12s} | src={int(scene.src[0])} escaped={bool(scene.escaped[0])} "
              f"on_dish={bool(scene.ball_on_dish()[0])} clear={bool(scene.dish_clear()[0])} "
              f"score={float(scene.score()[0]):.2f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def all_states() -> torch.Tensor:
        bodies = [scene.dish, scene.ball, *scene.cups]
        return torch.cat([b.data.root_state_w for b in bodies], dim=-1)

    # =========================== 1. show + null policy ==========================================
    torch.manual_seed(0)
    env.reset()
    step(40)
    report("show")
    check("settle: reset states finite (no NaN)", bool(torch.isfinite(all_states()).all()))
    check("rubric clean at reset (score 0)", float(scene.score()[0]) == 0.0)
    step(200)
    report("null-policy")
    check("null policy: score ~0, no success after 200 idle steps",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # =========================== 2. randomization is real (readback) ============================
    layouts = []
    srcs = set()
    ball_in_cup_ok = True
    for s in range(8):
        torch.manual_seed(100 + s)
        env.reset()
        step(25)
        dish_xy = (scene.dish.data.root_pos_w[0, :2] - env.iscene.env_origins[0, :2])
        cup0_xy = (scene.cups[0].data.root_pos_w[0, :2] - env.iscene.env_origins[0, :2])
        layouts.append(torch.cat([dish_xy, cup0_xy]).clone())
        srcs.add(int(scene.src[0]))
        # the ball must PHYSICALLY sit in the sampled cup: nearest cup == src and in-cavity
        bxy = scene.ball.data.root_pos_w[0, :2]
        dists = [float((cup.data.root_pos_w[0, :2] - bxy).norm()) for cup in scene.cups]
        nearest = int(np.argmin(dists))
        inside = bool(scene._ball_in_source_cup()[0])
        if nearest != int(scene.src[0]) or not inside:
            ball_in_cup_ok = False
            print(f"[smoke]   seed {100 + s}: src={int(scene.src[0])} nearest={nearest} "
                  f"inside={inside} — MISMATCH", flush=True)
        if len(srcs) >= 2 and s >= 2:
            break
    deltas = [float((a - b).abs().max()) for a in layouts for b in layouts if a is not b]
    check("randomization: dish + cup layouts differ across resets (readback)",
          len(layouts) >= 3 and max(deltas) > 0.008)
    check("randomization: >=2 distinct source cups sampled (readback)", len(srcs) >= 2)
    check("randomization: ball sits in the sampled source cup (readback)", ball_in_cup_ok)

    # =========================== 3. oracle x 3 seeds ============================================
    for seed in range(3):
        torch.manual_seed(seed)
        env.reset()
        step(30)
        res = oracle_solution(env, on_step=rec)
        report(f"oracle-{seed}")
        check(f"oracle seed {seed}: pour reaches success()", res["success"])
        print(f"[smoke]   oracle seed {seed}: retries={res['retries']} "
              f"milestones={res['milestones']}", flush=True)
        if seed == 0:
            ms = res["milestones"]
            check("oracle milestone: escape latched at >=0.25 (partial credit)",
                  0.25 - 1e-6 <= ms["escape"] < 0.85)
            check("oracle milestone: ball served, cup overhead -> 0.85, not success",
                  abs(ms["landed"] - 0.85) < 0.01 and not ms["landed_success"])
            check("rubric monotonicity along the oracle (0 < escape < 0.85 < 1.0)",
                  ms["reset"] == 0.0 and ms["reset"] < ms["escape"] < ms["landed"]
                  and abs(ms["final"] - 1.0) < 1e-6)

    # =========================== 4. negative A: the seed's own strategy =========================
    # "Put the (loaded) bowl on the plate": carry the source cup — ball riding inside, moved
    # as ONE RIGID ASSEMBLY (teleporting only the cup would leave the ball impaled through its
    # wall) — onto the dish and set it down. The plan that solved the seed must FAIL here.
    torch.manual_seed(11)
    env.reset()
    step(30)
    src = int(scene.src[0])
    cup = scene.cups[src]
    hp = cup.data.root_pos_w[0].clone()
    hq = cup.data.root_quat_w[0:1].clone()
    b_loc = quat_apply_inverse(hq, (scene.ball.data.root_pos_w[0] - hp).unsqueeze(0))[0]
    dish_p = scene.dish.data.root_pos_w[0] - env.iscene.env_origins[0]
    cup_target = (float(dish_p[0]), float(dish_p[1]),
                  float(dish_p[2]) - c.dish_h / 2 + c.dish_bot_t + c.cup_h / 2 + 0.003)
    cup_st = make_state(cup_target, tuple(hq[0].tolist()))
    ball_st = torch.zeros(n, 13, device=device)
    ball_st[:, 0:3] = cup_st[:, 0:3] + quat_apply(cup_st[:, 3:7], b_loc.unsqueeze(0))
    ball_st[:, 3] = 1.0
    cup.write_root_state_to_sim(cup_st, all_ids)
    scene.ball.write_root_state_to_sim(ball_st, all_ids)
    env.iscene.update(0.0)
    step(120)
    report("seed-strat")
    check("negative (seed strategy): loaded cup onto dish -> no success",
          not bool(scene.success()[0]))
    check("negative (seed strategy): score pinned ~0 (ball never left the cup)",
          float(scene.score()[0]) < 0.05)

    # =========================== 5. negative B: near miss =======================================
    torch.manual_seed(12)
    env.reset()
    step(30)
    dish_p = scene.dish.data.root_pos_w[0] - env.iscene.env_origins[0]
    beside = (float(dish_p[0]) + c.dish_outer_r + c.ball_r + 0.020, float(dish_p[1]),
              c.surface_z + c.ball_r + 0.003)
    scene.ball.write_root_state_to_sim(make_state(beside), all_ids)
    env.iscene.update(0.0)
    step(100)
    report("near-miss")
    check("negative (near miss): ball beside the dish -> exactly 0.25, no success",
          abs(float(scene.score()[0]) - 0.25) < 0.01 and not bool(scene.success()[0]))

    # =========================== 6. negative C: dish not cleared ================================
    # Ball served 35 mm off-axis (inside tolerance — the positive twin) + the EMPTY source cup
    # parked inside the dish on the far side: score must cap at 0.85; removing the cup must
    # complete to success.
    torch.manual_seed(13)
    env.reset()
    step(30)
    src = int(scene.src[0])
    cup = scene.cups[src]
    park_back = (cup.data.root_pos_w[0] - env.iscene.env_origins[0]).tolist()
    dish_p = scene.dish.data.root_pos_w[0] - env.iscene.env_origins[0]
    rim_top = float(dish_p[2]) + c.dish_h / 2
    scene.ball.write_root_state_to_sim(
        make_state((float(dish_p[0]) + 0.035, float(dish_p[1]), rim_top + 0.030)), all_ids)
    cup_in = (float(dish_p[0]) - 0.035, float(dish_p[1]),
              float(dish_p[2]) - c.dish_h / 2 + c.dish_bot_t + c.cup_h / 2 + 0.003)
    cup.write_root_state_to_sim(make_state(cup_in), all_ids)
    env.iscene.update(0.0)
    ok85 = settle_until(lambda: abs(float(scene.score()[0]) - 0.85) < 0.01, 300)
    report("not-cleared")
    check("negative (dish not cleared): cup parked in dish caps score at 0.85",
          ok85 and not bool(scene.success()[0]))
    cup.write_root_state_to_sim(
        make_state((park_back[0], park_back[1], c.surface_z + c.cup_h / 2 + 0.003)), all_ids)
    env.iscene.update(0.0)
    ok_done = settle_until(lambda: bool(scene.success()[0]), 300)
    report("cleared")
    check("clearing the dish completes: success after the cup is removed (35mm off-axis ok)",
          ok_done and abs(float(scene.score()[0]) - 1.0) < 1e-6)

    # =========================== 7. calibration sweep ===========================================
    # Drop the ball from 50 mm above the rim at growing xy offsets, 2 seeds each. Geometric
    # funnel = dish_inner_r - ball_r = 60 mm (corner nestle up to ~65 mm). Raw drop outcomes
    # are stochastic — asserted statistically, per-offset rates are published.
    print("[smoke] CALIBRATION SWEEP (drop offset -> capture rate, 2 seeds each)", flush=True)
    results: dict[float, int] = {}
    for off_mm in (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0):
        hits = 0
        for seed in range(2):
            torch.manual_seed(1000 + seed)
            env.reset()
            step(20)
            dish_p = scene.dish.data.root_pos_w[0] - env.iscene.env_origins[0]
            rim_top = float(dish_p[2]) + c.dish_h / 2
            ang = math.pi / 4 + seed * (2 * math.pi / 3)
            drop = (float(dish_p[0]) + off_mm / 1000.0 * math.cos(ang),
                    float(dish_p[1]) + off_mm / 1000.0 * math.sin(ang),
                    rim_top + 0.050)
            scene.ball.write_root_state_to_sim(make_state(drop), all_ids)
            env.iscene.update(0.0)
            step(60)  # the ball is written at zero velocity — settled() would read true
            # BEFORE the drop even starts; step through the fall first, then wait it out
            settle_until(lambda: bool(scene.settled()[0]), 200)
            hit = bool(scene.ball_on_dish()[0])
            hits += int(hit)
            print(f"[smoke]   off={off_mm:.0f}mm seed={seed}: captured={hit}", flush=True)
        results[off_mm] = hits
    in_funnel = {k: v for k, v in results.items() if k <= 50.0}
    rate = sum(in_funnel.values()) / (2 * len(in_funnel))
    band = [k for k, v in results.items() if v == 2]
    limit = max(band, default=0.0)
    print("[smoke] SWEEP RESULT: " +
          " | ".join(f"{k:.0f}mm: {v}/2" for k, v in results.items()) +
          f"  -> capture limit (last 2/2) = {limit:.0f}mm, within-funnel rate = {rate:.0%}",
          flush=True)
    check("sweep: within-funnel (<=50mm) capture rate >= 70%", rate >= 0.70)
    check("sweep: some offset fully reliable (2/2)", len(band) > 0)

    # =========================== save + verdict =================================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.serve_ingredient")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    for nm, ok in checks:
        if not ok:
            print(f"[smoke] FAILED CHECK: {nm}", flush=True)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {len(checks)}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)

    # Hard exit: Kit teardown hangs otherwise. Watchdog first, then a best-effort close.
    rc = 0 if all_ok else 1
    threading.Timer(10.0, lambda: os._exit(rc)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(rc)


if __name__ == "__main__":
    main()

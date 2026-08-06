"""Smoke / oracle test for SearServeScene — NullRobot, teleport oracle, RECORDED.

One linear run (pen_holder-smoke skeleton):
  1. show        — settle the reset layout (raw patties on the board); score must be 0;
  2. random      — randomization-is-real by READBACK across 3 seeds (grill/plate move, yaw
                   changes) + patty-count subset sampling varies across resets;
  3. null        — null policy: 240 idle steps must score ~0, cook nothing, never succeed;
  4. oracle x3   — the cook-and-serve schedule: teleport each present patty onto the sear
                   pad, let REAL contact time accumulate past cook_min (well short of
                   burn_time), teleport it flat onto the plate, settle -> success() on
                   seeds 0/1/2, with per-seed milestone scores strictly increasing;
  5. pipeline    — a guaranteed TWO-patty episode: the full sequential schedule (pad is
                   one-slot, so A must come off before B goes on) walks 6 milestones up
                   to 1.0;
  6. negative A  — the SEED's own strategy: transport the meat straight to the target
                   (raw patties onto the plate, perfectly settled — sanity-asserted via
                   the geometric on_plate) -> score exactly 0, success rejected;
  7. negative B  — overcook: leave a patty on the pad past burn_time, then plate it. The
                   spoil latch must hold (served=False, ~0 score) and must be TERMINAL;
  8. near-miss   — the seed's "grab it off the grill promptly": pulled off at 0.6*cook_min
                   -> not cooked, no credit beyond the pad latch;
  9. one-slot    — two patties can never BOTH be on the pad: dry config arithmetic + a
                   physical stacking attempt;
 10. calibration — dwell sweep on the cook clock: rate ~1 s/s against real pad time and
                   the raw / cooked / burned window boundaries land where configured.

ALWAYS records video via the viewport rgb annotator (RTX driver-version override, 3-render
ghost flush) and saves frames.npz in the CURRENT WORKING DIRECTORY. Bodies are driven
straight through scene handles; the NullRobot applies nothing.

Run (forge): python -m simgen_tasks.<task>.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=10)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the L20 driver version and silently rejects RTX -> the
# annotator returns EMPTY frames. Disable the check.
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

try:
    from .scene import SearServeSceneCfg  # noqa: E402  (import registers the scene + env)
except ImportError:  # standalone execution fallback
    from scene import SearServeSceneCfg  # noqa: E402


# ----- driver: stepping + recording ---------------------------------------------------------------
class Driver:
    """Steps the env with the empty NullRobot action, records viewport frames, and provides
    predicate-polled settling (settling time is physics, not what the checks are about)."""

    def __init__(self, env) -> None:
        self.env = env
        self.scene = env.scene
        self.no_action = torch.empty(0, device=env.device)
        self.all_ids = torch.arange(env.num_envs, device=env.device)
        self.step_i = 0
        self.frames: list[np.ndarray] = []
        self.annot = None

    def make_state(self, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> torch.Tensor:
        st = torch.zeros(self.env.num_envs, 13, device=self.env.device)
        st[:, 0:3] = self.env.iscene.env_origins + torch.tensor(
            [float(v) for v in pos], device=self.env.device)
        st[:, 3:7] = torch.tensor([float(v) for v in quat], device=self.env.device)
        return st

    def step(self, k: int) -> None:
        for _ in range(k):
            self.env.step(self.no_action, render=True)
            if self.annot is not None and self.step_i % args.record_every == 0:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    self.env.sim.render()
                arr = np.asarray(self.annot.get_data())
                if arr.size:
                    self.frames.append(arr[..., :3].astype(np.uint8).copy())
            self.step_i += 1

    def settle_until(self, pred, max_steps: int = 300, poll: int = 12) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            self.step(poll)
            waited += poll
            if pred():
                return True
        return False

    def put_patty(self, name: str, pos) -> None:
        """Teleport a patty (zero velocity, flat) and refresh buffers so the authored pose
        is what the very next predicates / physics step see."""
        self.scene.patties[name].write_root_state_to_sim(self.make_state(pos), self.all_ids)
        self.env.iscene.update(0.0)


# ----- the teleport-oracle schedule ----------------------------------------------------------------
def oracle_solution(scene_or_env, drv: Driver | None = None) -> dict:
    """Solve the CURRENT episode from its reset state: for each present patty in turn,
    place it flat on the sear pad, let REAL accumulated contact cross cook_min (removing it
    long before burn_time), then place it flat on the plate. Returns metrics: success,
    milestone scores (recorded only after each milestone predicate physically holds, so
    strict rubric increase is meaningful), and the cook clock at each removal."""
    env = scene_or_env if hasattr(scene_or_env, "iscene") else scene_or_env.env
    scene = env.scene
    drv = drv or Driver(env)
    c = scene.cfg
    origin = env.iscene.env_origins[0]

    drv.step(50)  # settle the fresh layout
    present = scene.present[0].clone()
    names = list(scene.patties)
    grill_p = (scene.grill.data.root_pos_w[0] - origin).tolist()
    plate_p = (scene.plate.data.root_pos_w[0] - origin).tolist()
    pad_top = grill_p[2] + c.pad_top_local
    plate_top = plate_p[2] + c.plate_t / 2

    milestones = [float(scene.score()[0])]
    cook_at_removal: list[float] = []
    cook_budget = int(c.cook_min * 120) + 360
    slot = 0
    ok_stages = True
    for i, name in enumerate(names):
        if not bool(present[i]):
            continue
        h = c.patties[i][2]
        # stage 1: raw patty onto the (empty — one-slot) sear pad
        drv.put_patty(name, (grill_p[0], grill_p[1], pad_top + h / 2 + 0.001))
        ok_stages &= drv.settle_until(lambda i=i: bool(scene.ever_on_pad[0, i]), max_steps=120)
        milestones.append(float(scene.score()[0]))
        # stage 2: dwell — the real cook clock must cross cook_min
        ok_stages &= drv.settle_until(lambda i=i: bool(scene.cooked()[0, i]),
                                      max_steps=cook_budget)
        milestones.append(float(scene.score()[0]))
        cook_at_removal.append(float(scene.cook_t[0, i]))
        # stage 3: off the grill, flat onto its serving slot on the plate
        sx, sy = c.plate_slots[slot]
        slot += 1
        drv.put_patty(name, (plate_p[0] + sx, plate_p[1] + sy, plate_top + h / 2 + 0.002))
        ok_stages &= drv.settle_until(lambda i=i: bool(scene.served()[0, i]), max_steps=300)
        milestones.append(float(scene.score()[0]))

    ok = drv.settle_until(lambda: bool(scene.success()[0]), max_steps=240)
    increasing = all(b > a + 1e-6 for a, b in zip(milestones, milestones[1:]))
    return {"success": ok and ok_stages, "milestones": milestones, "increasing": increasing,
            "cook_at_removal": cook_at_removal, "n_present": int(present.sum()),
            "burned_any": bool(scene.burned[0, present].any()),
            "final_score": float(scene.score()[0])}


# ----- main battery ---------------------------------------------------------------------------------
def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.sear_and_serve")().build(
        num_envs=args.num_envs, device=device, scene_cfg=SearServeSceneCfg())
    scene = env.scene
    c = scene.cfg
    drv = Driver(env)
    origin0 = env.iscene.env_origins[0]
    dt = env.dt

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = origin0.detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.15, -1.05, 0.90)) + o),
                                tuple(np.array((0.00, 0.08, 0.06)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        drv.annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        drv.annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(drv.annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)
        drv.annot = None

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def report(tag: str) -> None:
        print(f"[smoke] {tag:12s} | present={scene.present[0].int().tolist()} "
              f"cook_t={[f'{v:.2f}' for v in scene.cook_t[0].tolist()]} "
              f"cooked={scene.cooked()[0].int().tolist()} "
              f"burned={scene.burned[0].int().tolist()} "
              f"on_pad={scene.on_pad()[0].int().tolist()} "
              f"on_plate={scene.on_plate()[0].int().tolist()} "
              f"served={scene.served()[0].int().tolist()} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(drv.frames)}", flush=True)

    def first_present() -> int:
        return int(torch.nonzero(scene.present[0])[0])

    def pad_target(i: int) -> tuple:
        grill_p = (scene.grill.data.root_pos_w[0] - origin0).tolist()
        h = c.patties[i][2]
        return (grill_p[0], grill_p[1], grill_p[2] + c.pad_top_local + h / 2 + 0.001)

    def plate_target(i: int, slot: int = 0) -> tuple:
        plate_p = (scene.plate.data.root_pos_w[0] - origin0).tolist()
        sx, sy = c.plate_slots[slot]
        h = c.patties[i][2]
        return (plate_p[0] + sx, plate_p[1] + sy, plate_p[2] + c.plate_t / 2 + h / 2 + 0.002)

    # =========================== 1. show =======================================================
    torch.manual_seed(0)
    env.reset()
    report("reset")
    drv.step(60)
    report("show")
    finite = all(torch.isfinite(b.data.root_state_w).all()
                 for b in [scene.board, scene.grill, scene.plate, *scene.patties.values()])
    check("reset settles finite with score 0",
          finite and float(scene.score()[0]) == 0.0 and not bool(scene.success()[0]))

    # =========================== 2. randomization by readback ==================================
    obs = []
    for s in (201, 202, 203):
        torch.manual_seed(s)
        env.reset()
        drv.step(5)
        obs.append((scene.grill.data.root_pos_w[0, :2] - origin0[:2],
                    scene.plate.data.root_pos_w[0, :2] - origin0[:2],
                    scene.grill.data.root_quat_w[0].clone()))
    grill_moves = max(float((a[0] - b[0]).norm()) for a in obs for b in obs)
    plate_moves = max(float((a[1] - b[1]).norm()) for a in obs for b in obs)
    yaw_moves = min(float((a[2] - b[2]).norm()) for i, a in enumerate(obs)
                    for j, b in enumerate(obs) if i < j)
    print(f"[smoke] randomization: grill dxy={grill_moves * 1000:.1f}mm "
          f"plate dxy={plate_moves * 1000:.1f}mm grill dq={yaw_moves:.3f}", flush=True)
    check("randomization is real (grill+plate move on readback)",
          grill_moves > 0.005 and plate_moves > 0.005 and yaw_moves > 0.01)
    counts = set()
    for s in range(210, 222):
        torch.manual_seed(s)
        env.reset()
        counts.add(int(scene.present[0].sum()))
    print(f"[smoke] subset-sampled patty counts over 12 resets: {sorted(counts)}", flush=True)
    check("patty-count subset sampling varies", counts == {1, 2})

    # =========================== 3. null policy ================================================
    torch.manual_seed(42)
    env.reset()
    drv.step(240)
    report("null")
    check("null policy scores ~0, cooks nothing, no success",
          float(scene.score()[0]) <= 0.05 and float(scene.cook_t.max()) == 0.0
          and not bool(scene.success()[0]))

    # =========================== 4. oracle on 3 seeds ==========================================
    runs = []
    for s in (0, 1, 2):
        torch.manual_seed(s)
        env.reset()
        m = oracle_solution(env, drv)
        report(f"oracle-s{s}")
        runs.append(m)
        print(f"[smoke]   oracle seed {s}: n_present={m['n_present']} "
              f"milestones={[f'{v:.3f}' for v in m['milestones']]} "
              f"cook_at_removal={[f'{v:.2f}' for v in m['cook_at_removal']]} "
              f"burned_any={m['burned_any']} final={m['final_score']:.2f}", flush=True)
        check(f"oracle solve reaches success() on seed {s}",
              m["success"] and not m["burned_any"] and abs(m["final_score"] - 1.0) < 1e-6)
    check("oracle milestone scores strictly increase on every seed",
          all(m["increasing"] for m in runs))
    check("oracle always unloads the grill well before burn_time",
          all(v <= c.burn_time - 1.0 for m in runs for v in m["cook_at_removal"]))

    # =========================== 5. guaranteed two-patty pipeline ==============================
    seed2 = None
    for s in range(30, 80):
        torch.manual_seed(s)
        env.reset()
        if int(scene.present[0].sum()) == 2:
            seed2 = s
            break
    m2 = oracle_solution(env, drv)
    report("pipeline")
    print(f"[smoke]   two-patty pipeline (seed {seed2}): "
          f"milestones={[f'{v:.3f}' for v in m2['milestones']]}", flush=True)
    check("two-patty episode: sequential schedule walks 6 milestones to 1.0",
          m2["n_present"] == 2 and len(m2["milestones"]) == 7 and m2["increasing"]
          and m2["success"])

    # =========================== 6. negative A: the seed's own strategy =========================
    # meat_off_grill's plan is pure transport: pick the meat up and put it at the target.
    # Execute it perfectly — every present patty straight from the board onto the plate,
    # settled flat (sanity-asserted via the geometric on_plate) — and it must earn NOTHING.
    torch.manual_seed(11)
    env.reset()
    drv.step(50)
    slot = 0
    for i, name in enumerate(scene.patties):
        if not bool(scene.present[0, i]):
            continue
        drv.put_patty(name, plate_target(i, slot))
        slot += 1
    drv.step(90)
    report("seed-strat")
    plated = bool((scene.on_plate()[0] | ~scene.present[0]).all())
    check("negative A: seed strategy (raw meat straight to the plate) scores 0",
          plated and float(scene.score()[0]) <= 1e-6 and not bool(scene.success()[0]))

    # =========================== 7. negative B: overcook ========================================
    torch.manual_seed(12)
    env.reset()
    drv.step(50)
    i0 = first_present()
    name0 = list(scene.patties)[i0]
    drv.put_patty(name0, pad_target(i0))
    drv.step(int(round((c.burn_time + 0.5) / dt)))  # burn_time + 0.5 s on the pad
    burned_now = bool(scene.burned[0, i0])
    cooked_too = bool(scene.cooked()[0, i0])
    drv.put_patty(name0, plate_target(i0, 0))
    drv.settle_until(lambda: bool(scene.on_plate()[0, i0]) and bool(scene.settled()[0, i0]),
                     max_steps=240)
    report("overcook")
    check("negative B: overcooked patty is spoiled — plating it earns ~nothing",
          burned_now and cooked_too and not bool(scene.served()[0, i0])
          and float(scene.score()[0]) <= 0.15 and not bool(scene.success()[0]))
    drv.step(120)
    check("negative B: the burn latch is terminal",
          bool(scene.burned[0, i0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.15)

    # =========================== 8. near-miss: pulled off too early =============================
    torch.manual_seed(13)
    env.reset()
    drv.step(50)
    i0 = first_present()
    name0 = list(scene.patties)[i0]
    drv.put_patty(name0, pad_target(i0))
    drv.step(int(round(0.6 * c.cook_min / dt)))  # the seed's prompt grab-off: 0.9 s < cook_min
    early_ct = float(scene.cook_t[0, i0])
    drv.put_patty(name0, plate_target(i0, 0))
    drv.settle_until(lambda: bool(scene.on_plate()[0, i0]) and bool(scene.settled()[0, i0]),
                     max_steps=240)
    report("near-miss")
    print(f"[smoke]   near-miss cook_t at removal: {early_ct:.2f}s "
          f"(cook_min {c.cook_min:.2f}s)", flush=True)
    check("near-miss: patty pulled off the grill too early is not cooked, no serve credit",
          not bool(scene.cooked()[0, i0]) and not bool(scene.served()[0, i0])
          and float(scene.score()[0]) <= 0.21 and not bool(scene.success()[0]))

    # =========================== 9. one-slot pad ================================================
    r0, r1 = c.patties[0][1], c.patties[1][1]
    slack_diag = math.hypot((c.pad_half - r0) + (c.pad_half - r1),
                            (c.pad_half - r0) + (c.pad_half - r1))
    print(f"[smoke] one-slot dry math: max both-inside separation {slack_diag * 1000:.1f}mm "
          f"< touching distance {(r0 + r1) * 1000:.1f}mm", flush=True)
    seed_both = None
    for s in range(80, 130):
        torch.manual_seed(s)
        env.reset()
        if int(scene.present[0].sum()) == 2:
            seed_both = s
            break
    drv.step(40)
    names = list(scene.patties)
    drv.put_patty(names[0], pad_target(0))
    drv.step(30)
    gp = pad_target(0)
    drv.put_patty(names[1], (gp[0], gp[1],
                             gp[2] + c.patties[0][2] / 2 + c.patties[1][2] / 2 + 0.001))
    drv.step(60)
    report("stacked")
    both_on = bool(scene.on_pad()[0, 0]) and bool(scene.on_pad()[0, 1])
    check("one-slot pad: two patties can never both be on the pad (dry math + stacking probe)",
          slack_diag < r0 + r1 - 0.01 and not both_on and seed_both is not None)

    # =========================== 10. calibration: the cook clock ================================
    outcomes = []
    for dwell in (0.5 * c.cook_min, 1.2 * c.cook_min, c.burn_time + 0.5):
        torch.manual_seed(14)
        env.reset()
        drv.step(40)
        i0 = first_present()
        drv.put_patty(list(scene.patties)[i0], pad_target(i0))
        drv.step(int(round(dwell / dt)))
        outcomes.append((dwell, float(scene.cook_t[0, i0]),
                         bool(scene.cooked()[0, i0]), bool(scene.burned[0, i0])))
    for dwell, ct, ck, bn in outcomes:
        print(f"[smoke] CALIBRATION dwell={dwell:.2f}s -> cook_t={ct:.2f}s "
              f"cooked={ck} burned={bn}", flush=True)
    rate_ok = all(0.80 <= ct / dwell <= 1.05 for dwell, ct, _ck, _bn in outcomes)
    check("calibration: cook clock tracks real pad time (rate ~1 s/s)", rate_ok)
    check("calibration: window boundaries behave (raw / cooked / burned)",
          outcomes[0][2] is False and outcomes[0][3] is False
          and outcomes[1][2] is True and outcomes[1][3] is False
          and outcomes[2][3] is True)

    # =========================== save + verdict ================================================
    arr = (np.stack(drv.frames, axis=0) if drv.frames
           else np.zeros((0, 600, 960, 3), np.uint8))
    np.savez_compressed(args.out, frames=arr, env="simgen.sear_and_serve")
    print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)

    n_ok = sum(ok for _n, ok in checks)
    if n_ok == len(checks):
        print(f"SIM_GEN_SMOKE: ALL PASS {n_ok}/{len(checks)}", flush=True)
    else:
        for name, ok in checks:
            if not ok:
                print(f"[smoke] FAILED CHECK: {name}", flush=True)
        print(f"SIM_GEN_SMOKE: FAIL {n_ok}/{len(checks)}", flush=True)

    # Kit teardown hangs are routine — hard-exit behind a watchdog.
    threading.Timer(10.0, lambda: os._exit(0)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(0)


if __name__ == "__main__":
    main()

"""Smoke / oracle test for TeeUpScene — NullRobot, teleport-oracle, RECORDED.

Battery (linear run, pen_holder-smoke skeleton):
  1. settle/no-NaN      — reset layout settles finite, ball contained, score 0;
  2. randomization      — two seeded resets, READBACK basket/tee/ball poses differ;
  3. null-policy        — 240 idle steps, score stays ~0;
  4. oracle (3 seeds)   — kinematic carry: lift ball out of the basket, set it down on
                          the ground (score 0.20), carry to a hover over the tee (0.45),
                          descend to 6 mm and RELEASE — the seat is a genuine physical
                          drop-and-settle; success() on every seed, rubric monotone;
  5. negative controls  — (a) the SEED'S OWN STRATEGY: dump from 0.25 m above the tee
                          (the basketball_in_hoop release) bounces off, no success;
                          (b) near-miss: ball resting at the tee base (xy-near, z-low);
                          (c) re-containing the ball in the basket scores <= 0.20;
                          (d) tolerance: gentle release 35 mm off-axis rolls off;
  6. calibration probe  — release-height sweep at fixed 12 mm offset, 3 seeds each:
                          publishes seat-rate vs drop energy (the gentleness envelope);
                          gentle (<= 30 mm) must mostly seat, high (>= 200 mm) mostly not.

Records video (viewport rgb annotator, RTX driver-version override) during the show +
oracle-seed-0 + dump-control phases and saves `frames.npz` in the CWD. Prints exactly
`SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes, then hard-exits (Kit
teardown hangs otherwise).

Run: python smoke.py --headless
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
# RTX recipe: kit mis-decodes the driver version on some GPUs and silently rejects RTX
# -> the annotator returns EMPTY frames. Disable the check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math
import os
import sys
import threading

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import robobench
from robobench.core import ENVS

robobench.discover()
import scene as task_scene  # noqa: F401  (registers SCENES["tee_up"] + ENVS["sim_gen.tee_up"])

# ----- module state wired up in main() ---------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}


def _step(k: int = 1) -> None:
    """Advance k control steps (null action), capturing frames while recording is on."""
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        if rec and _REC["i"] % args.record_every == 0:
            for _f in range(3):  # flush accumulated history (ghosting fix)
                env.sim.render()
            arr = np.asarray(_REC["annot"].get_data())
            if arr.size:
                _REC["frames"].append(arr[..., :3].astype(np.uint8).copy())
        _REC["i"] += 1


def _refresh() -> None:
    _ENV.iscene.update(0.0)


def _all_ids():
    return torch.arange(_ENV.num_envs, device=_ENV.device)


def _write_ball(pos_w: torch.Tensor) -> None:
    """Teleport the ball (world pos (N,3)), zero velocities."""
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    st[:, 3] = 1.0
    _ENV.scene.ball.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _carry(targets: list[torch.Tensor], speed: float = 0.006) -> None:
    """Kinematically carry the ball through straight-line waypoints ((N,3) world),
    rewriting its state before every physics step (max `speed` m per 120 Hz step)."""
    scene = _ENV.scene
    cur = scene.ball.data.root_pos_w.clone()
    for tgt in targets:
        while True:
            delta = tgt - cur
            dist = delta.norm(dim=-1, keepdim=True)
            if float(dist.max()) < 1e-4:
                break
            cur = cur + delta / dist.clamp(min=1e-9) * dist.clamp(max=speed)
            _write_ball(cur)
            _step(1)


def _settle_until(pred, max_steps: int = 300, poll: int = 15) -> bool:
    if pred():
        return True
    waited = 0
    while waited < max_steps:
        _step(poll)
        waited += poll
        if pred():
            return True
    return False


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    bp = scene.ball.data.root_pos_w[0]
    print(f"[smoke] {tag:16s} | ball=({float(bp[0]):+.3f},{float(bp[1]):+.3f},"
          f"{float(bp[2]):.3f}) score={float(scene.score()[0]):.2f} "
          f"seated={bool(scene.seated()[0])} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- the exported oracle ---------------------------------------------------------------------
def oracle_solution(scene_or_env, on_stage=None) -> bool:
    """Teleport-oracle: extract the ball from the basket (set it down on the ground —
    a real release), then carry it over the tee and RELEASE it 6 mm above the perch;
    seating is honest drop-and-settle physics. Returns success() for env 0."""
    env = scene_or_env if hasattr(scene_or_env, "iscene") else scene_or_env.env
    scene = env.scene
    c = scene.cfg
    _refresh()
    kb = scene.basket.data.root_pos_w.clone()
    tp = scene.tee.data.root_pos_w.clone()
    bp = scene.ball.data.root_pos_w.clone()
    dev = env.device
    n = env.num_envs

    def pt(x, y, z):
        p = torch.zeros(n, 3, device=dev)
        p[:, 0], p[:, 1], p[:, 2] = x, y, z
        return p

    lift_z = kb[:, 2] + c.basket_h + c.ball_r + 0.08
    # set-down point 60% of the way to the tee — clear of both footprints
    mx = kb[:, 0] + 0.6 * (tp[:, 0] - kb[:, 0])
    my = kb[:, 1] + 0.6 * (tp[:, 1] - kb[:, 1])

    # stage 1: extract — lift out of the basket, set down on the ground, release
    _carry([pt(bp[:, 0], bp[:, 1], lift_z), pt(mx, my, lift_z),
            pt(mx, my, kb[:, 2] + c.ball_r + 0.004)])
    _settle_until(lambda: bool(scene.settled()[0]), max_steps=180)
    if on_stage:
        on_stage("extracted")

    # stage 2: seat — hover over the cradle, descend, release 6 mm above the perch
    for attempt in range(2):
        seat_z = tp[:, 2] + c.seat_z_local
        _carry([pt(mx, my, lift_z), pt(tp[:, 0], tp[:, 1], seat_z + 0.10)])
        if on_stage and attempt == 0:
            on_stage("hover")
        _carry([pt(tp[:, 0], tp[:, 1], seat_z + 0.006)], speed=0.003)
        # release: stop writing, let it drop the last 6 mm and settle
        ok = _settle_until(lambda: bool(scene.success()[0]), max_steps=300)
        if ok:
            break
        print(f"[smoke]   oracle retry: seat attempt {attempt} missed", flush=True)
        mx, my = tp[:, 0], tp[:, 1] - 0.10  # re-grab from wherever it fell, via a hover
        _refresh()
        b2 = scene.ball.data.root_pos_w
        mx, my = b2[:, 0], b2[:, 1]
    if on_stage:
        on_stage("seated")
    return bool(scene.success()[0])


# ----- main ------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("sim_gen.tee_up")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.85, -0.85, 0.60)) + o),
                                tuple(np.array((0.0, 0.0, 0.12)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        _REC["annot"] = annot if warm.size else None
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def seat_point(dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> torch.Tensor:
        _refresh()
        tp = scene.tee.data.root_pos_w.clone()
        p = tp.clone()
        p[:, 0] += dx
        p[:, 1] += dy
        p[:, 2] += c.seat_z_local + dz
        return p

    # ================= 1. settle / no-NaN ================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(60)
    _report("show")
    states = torch.cat([scene.ball.data.root_state_w, scene.basket.data.root_state_w,
                        scene.tee.data.root_state_w], dim=-1)
    bp = scene.ball.data.root_pos_w[0]
    kp = scene.basket.data.root_pos_w[0]
    contained = (float((bp[:2] - kp[:2]).norm()) < c.basket_inner_r
                 and 0.02 < float(bp[2] - kp[2]) < c.basket_h)
    check("settle/no-NaN: layout settles finite, ball contained in basket, score 0",
          bool(torch.isfinite(states).all()) and contained
          and float(scene.score()[0]) == 0.0)
    _REC["on"] = False

    # ================= 2. randomization is real (readback) ===============================
    def readback():
        _refresh()
        return (scene.basket.data.root_pos_w[0, :2].clone(),
                scene.tee.data.root_pos_w[0, :2].clone(),
                scene.ball.data.root_pos_w[0, :2].clone())

    torch.manual_seed(101)
    env.reset()
    a_basket, a_tee, a_ball = readback()
    torch.manual_seed(202)
    env.reset()
    b_basket, b_tee, b_ball = readback()
    d_basket = float((a_basket - b_basket).norm())
    d_tee = float((a_tee - b_tee).norm())
    d_rel = float(((a_ball - a_basket) - (b_ball - b_basket)).norm())
    print(f"[smoke] randomization deltas: basket={d_basket * 1000:.1f}mm "
          f"tee={d_tee * 1000:.1f}mm ball-in-basket={d_rel * 1000:.1f}mm", flush=True)
    check("randomization-is-real: basket/tee/ball readback differs across resets",
          d_basket > 0.005 and d_tee > 0.005 and d_rel > 0.005)

    # ================= 3. null policy fails ==============================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 4. oracle on 3 seeds + rubric monotonicity ========================
    stage_scores: list[float] = []

    def on_stage(tag: str) -> None:
        _refresh()
        stage_scores.append(float(scene.score()[0]))
        _report(f"oracle-{tag}")

    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        env.reset()
        _step(30)
        _REC["on"] = seed == 0
        stage_scores.clear()
        stage_scores.append(float(scene.score()[0]))
        ok = oracle_solution(env, on_stage=on_stage if seed == 0 else None)
        _report(f"oracle-seed-{seed}")
        check(f"oracle reaches success() (seed {seed})", ok)
        if seed == 0:
            # stage_scores = [reset, extracted, hover, seated]
            check("rubric: extraction stage scores 0.20",
                  abs(stage_scores[1] - 0.20) < 1e-6)
            check("rubric: hover-over-tee stage scores 0.45",
                  abs(stage_scores[2] - 0.45) < 1e-6)
            check("rubric: monotone 0 -> 0.20 -> 0.45 -> 1.0",
                  stage_scores == sorted(stage_scores) and stage_scores[0] == 0.0
                  and stage_scores[-1] == 1.0)
        _REC["on"] = False

    # ================= 5a. negative: the seed's own strategy (dump from height) ==========
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _write_ball(seat_point(dx=0.012, dz=0.25))
    _step(360)
    _report("seed-dump")
    check("negative (seed strategy): 0.25 m dump bounces off — not seated, no success",
          not bool(scene.seated()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) < 1.0)
    _REC["on"] = False

    # ================= 5b. negative: near-miss at the tee base ===========================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _refresh()
    tp = scene.tee.data.root_pos_w.clone()
    p = tp.clone()
    p[:, 0] += c.post_r + c.ball_r + 0.002
    p[:, 2] += c.tee_base_t + c.ball_r + 0.002
    _write_ball(p)
    _step(180)
    _report("near-miss-base")
    check("negative (near-miss): ball at the tee base does not count",
          not bool(scene.seated()[0]) and not bool(scene.success()[0]))

    # ================= 5c. negative: re-containing the ball (the seed's goal) ============
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _refresh()
    kb = scene.basket.data.root_pos_w.clone()
    b0 = scene.ball.data.root_pos_w.clone()
    up = kb.clone()
    up[:, 2] += c.basket_h + c.ball_r + 0.06  # clears the rim -> extraction latch fires
    _carry([torch.stack([b0[:, 0], b0[:, 1], up[:, 2]], dim=-1), up])
    _write_ball(up)  # release above the basket mouth: drops straight back in
    _settle_until(lambda: bool(scene.settled()[0]), max_steps=240)
    _report("re-basket")
    check("negative (re-contain): ball back in the basket scores <= 0.20 (latched only)",
          float(scene.score()[0]) <= 0.20 + 1e-6 and not bool(scene.success()[0]))

    # ================= 5d. negative: tolerance (35 mm off-axis) ==========================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _write_ball(seat_point(dx=0.035, dz=0.010))
    _step(240)
    _report("off-axis")
    check("negative (tolerance): gentle release 35 mm off-axis does not seat",
          not bool(scene.seated()[0]) and not bool(scene.success()[0]))

    # ================= 6. calibration probe: release-height sweep ========================
    print("[smoke] CALIBRATION SWEEP (release height above perch @ 12 mm offset "
          "-> seat rate, 3 seeds each)", flush=True)
    heights = (0.010, 0.030, 0.060, 0.120, 0.200, 0.280)
    rates: dict[float, int] = {}
    for h in heights:
        hits = 0
        for seed in (10, 11, 12):
            torch.manual_seed(seed)
            env.reset()
            _step(20)
            ang = 2 * math.pi * (seed - 10) / 3.0
            _write_ball(seat_point(dx=0.012 * math.cos(ang), dy=0.012 * math.sin(ang),
                                   dz=h))
            _settle_until(lambda: bool(scene.success()[0]), max_steps=300)
            hit = bool(scene.success()[0])
            hits += int(hit)
            print(f"[smoke]   h={h * 1000:.0f}mm seed={seed}: seated={hit}", flush=True)
        rates[h] = hits
    gentle = sum(v for k, v in rates.items() if k <= 0.030)
    high = sum(v for k, v in rates.items() if k >= 0.200)
    print("[smoke] SWEEP RESULT: " +
          " | ".join(f"{k * 1000:.0f}mm: {v}/3" for k, v in rates.items()) +
          f"  -> gentle(<=30mm)={gentle}/6, high(>=200mm)={high}/6", flush=True)
    check("calibration: gentle releases (<=30 mm) seat >= 5/6", gentle >= 5)
    check("calibration: high dumps (>=200 mm) seat <= 2/6", high <= 2)

    # ================= save + verdict ====================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="sim_gen.tee_up")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)

    n_pass = sum(ok for _n, ok in checks)
    n_tot = len(checks)
    if n_pass == n_tot:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{n_tot}", flush=True)
        code = 0
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{n_tot}", flush=True)
        code = 1

    # hard exit — Kit teardown hangs otherwise
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()

"""Smoke / oracle test for LidTrayStowScene — NullRobot, teleport-oracle, RECORDED.

Battery (linear run, tee_up/pen_holder smoke skeleton):
  1. settle/no-NaN      — reset layout settles finite, cargo resting ON the lid, score 0;
  2. randomization      — two seeded resets, READBACK box/lid poses AND yaws differ;
  3. subset sampling    — present-count varies across 10 resets (bottle always present);
  4. null-policy        — 240 idle steps, score stays ~0;
  5. oracle (3 seeds)   — kinematic carry: stow the bottle FLAT + each present cube into
                          the open box (real release-and-settle each), then fetch the
                          freed lid, align it over the mouth and RELEASE 8 mm above the
                          seat — the flush capping is genuine drop-and-settle physics;
                          success() on every seed (seed 1 runs with subset sampling ON —
                          proves success is judged on the sampled subset), rubric stow
                          transitions strictly increase and the lid-over latch fires;
  6. negative controls  — (a) the SEED'S OWN STRATEGY: cap the box immediately with the
                          loaded lid (cargo riding on top) — the lid seats but NOTHING is
                          stowed: score <= 0.25, no success;
                          (b) near-miss: lid released 40 mm off-axis tips into the box;
                          (c) incomplete stow: one cube left outside, lid seated flush —
                          no success;
                          (d) upright bottle: inside-but-standing earns no stow credit,
                          and the lid physically CANNOT seat on it (130 mm > 100 mm);
                          (e) tolerance: a 45 deg twisted lid at the exact seat pose is
                          rejected by the yaw gate (aperture corners exposed);
  7. calibration probe  — lid-release xy-offset sweep (box-frame x), 3 seeds each:
                          publishes seat-rate vs offset; small offsets (<= 10 mm) must be
                          reliable, large (>= 20 mm) must fail.

Records video (viewport rgb annotator, RTX driver-version override) during show +
oracle-seed-0 + the seed-strategy and upright-bottle controls, and saves `frames.npz`
in the CWD. Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes,
then hard-exits (Kit teardown hangs otherwise).

Run: python smoke.py --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=12)
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
import scene as task_scene  # noqa: F401  (registers SCENES["lid_tray_stow"] + the env)

# ----- module state wired up in main() ---------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}


def _step(k: int = 1) -> None:
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


def _write_body(body, pos_w: torch.Tensor, quat: torch.Tensor | None = None) -> None:
    """Teleport a body (world pos (N,3), optional quat (N,4)), zero velocities."""
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _carry(body, targets: list[torch.Tensor], quat: torch.Tensor,
           speed: float = 0.006) -> None:
    """Kinematically carry a body through straight-line waypoints ((N,3) world) at fixed
    orientation, rewriting its state before every physics step."""
    _refresh()
    cur = body.data.root_pos_w.clone()
    for tgt in targets:
        while True:
            delta = tgt - cur
            dist = delta.norm(dim=-1, keepdim=True)
            if float(dist.max()) < 1e-4:
                break
            cur = cur + delta / dist.clamp(min=1e-9) * dist.clamp(max=speed)
            _write_body(body, cur, quat)
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
    print(f"[smoke] {tag:16s} | stowed={scene.stowed()[0].int().tolist()} "
          f"present={scene.present[0].int().tolist()} "
          f"lid_seated={bool(scene.lid_seated()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


def _quats():
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul
    return quat_apply, quat_apply_inverse, quat_mul


def _seat_pose():
    """(pos (N,3), quat (N,4)) of the flush-seated lid in world frame."""
    scene = _ENV.scene
    _refresh()
    bp = scene.box.data.root_pos_w.clone()
    bq = scene.box.data.root_quat_w.clone()
    p = bp.clone()
    p[:, 2] += scene.cfg.seat_z_local
    return p, bq


def _stow_target(i: int):
    """(release pos (N,3), quat (N,4)) for item i inside the box (box-frame slot)."""
    quat_apply, _qai, quat_mul = _quats()
    scene = _ENV.scene
    c = scene.cfg
    _refresh()
    bp = scene.box.data.root_pos_w.clone()
    bq = scene.box.data.root_quat_w.clone()
    n = _ENV.num_envs
    sx, sy = c.stow_slots[i]
    name, kind, r, _length, _m, _rgb = c.items[i]
    loc = torch.zeros(n, 3, device=_ENV.device)
    loc[:, 0], loc[:, 1] = sx, sy
    loc[:, 2] = c.bot_t + r + 0.012  # release 12 mm above resting contact
    pos = bp + quat_apply(bq, loc)
    if kind == "cyl":  # lying flat along the box x-axis: q = bq * qy(90)
        qy = torch.zeros(n, 4, device=_ENV.device)
        c45 = math.cos(math.pi / 4)
        qy[:, 0], qy[:, 2] = c45, c45
        quat = quat_mul(bq, qy)
    else:
        quat = bq.clone()
    return pos, quat


# ----- the exported oracle ---------------------------------------------------------------------
def oracle_solution(scene_or_env, on_stage=None, speed: float = 0.006) -> bool:
    """Teleport-oracle: carry the bottle (laid FLAT) and each present cube into the open
    box (real release-and-settle each), then fetch the freed lid, align it over the mouth
    and release 8 mm above the seat — capping is honest drop-and-settle physics. Returns
    success() for env 0."""
    env = scene_or_env if hasattr(scene_or_env, "iscene") else scene_or_env.env
    scene = env.scene
    c = scene.cfg
    _refresh()

    def over(p: torch.Tensor, z: float) -> torch.Tensor:
        q = p.clone()
        q[:, 2] = z
        return q

    kk = int(scene.present[0].sum())
    done = 0
    for i, (name, _kind, _r, _length, _m, _rgb) in enumerate(c.items):
        if not bool(scene.present[0, i]):
            continue
        body = scene.items[name]
        tgt, tq = _stow_target(i)
        _refresh()
        p0 = body.data.root_pos_w.clone()
        _carry(body, [over(p0, 0.28), over(tgt, 0.28), tgt], tq, speed=speed)
        done += 1
        want = 0.55 * done / kk - 0.01
        ok = _settle_until(
            lambda i=i, want=want: bool(scene.stowed()[0, i])
            and float(scene.score()[0]) >= want, max_steps=360)
        if not ok:
            print(f"[smoke]   oracle: stow of {name} did not settle in", flush=True)
        if on_stage:
            on_stage(f"stow-{name}")

    # lid: fetch, align over the mouth, release 8 mm above the seat
    seat_p, seat_q = _seat_pose()
    for attempt in range(2):
        _refresh()
        lp = scene.lid.data.root_pos_w.clone()
        _carry(scene.lid, [over(lp, 0.30), over(seat_p, 0.30)], seat_q, speed=speed)
        if on_stage and attempt == 0:
            on_stage("lid-hover")
        drop = seat_p.clone()
        drop[:, 2] += 0.008
        _carry(scene.lid, [drop], seat_q, speed=max(speed / 2, 0.003))
        ok = _settle_until(lambda: bool(scene.success()[0]), max_steps=300)
        if ok:
            break
        print(f"[smoke]   oracle retry: lid seat attempt {attempt} missed", flush=True)
    if on_stage:
        on_stage("capped")
    return bool(scene.success()[0])


# ----- main ------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("sim_gen.lid_tray_stow")().build(
        num_envs=args.num_envs, device=device,
        scene_cfg=task_scene.LidTrayStowSceneCfg(subset_sample=False))
    _ENV = env
    scene = env.scene
    c = scene.cfg
    quat_apply, quat_apply_inverse, quat_mul = _quats()

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.85, -0.85, 0.65)) + o),
                                tuple(np.array((0.04, 0.0, 0.08)) + o),
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

    def yaw_of(quat_row: torch.Tensor) -> float:
        return math.degrees(2.0 * math.atan2(float(quat_row[3]), float(quat_row[0])))

    def dyaw(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    def stow_teleport(indices) -> None:
        """Place the given items directly at their in-box slots (kinematic teleport,
        used by the negative controls to reach a stowed state quickly)."""
        for i in indices:
            name = c.items[i][0]
            tgt, tq = _stow_target(i)
            tgt = tgt.clone()
            tgt[:, 2] -= 0.008  # nearly resting
            _write_body(scene.items[name], tgt, tq)
        _step(60)

    # ================= 1. settle / no-NaN ================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(60)
    _report("show")
    _REC["on"] = False
    states = torch.cat([scene.lid.data.root_state_w, scene.box.data.root_state_w]
                       + [b.data.root_state_w for b in scene.items.values()], dim=-1)
    lp = scene.lid.data.root_pos_w[0]
    on_lid = True
    for i, (name, *_r) in enumerate(c.items):
        ip = scene.items[name].data.root_pos_w[0]
        on_lid &= (float((ip[:2] - lp[:2]).norm()) < 0.16 and float(ip[2]) > 0.022)
    check("settle/no-NaN: layout settles finite, cargo resting ON the lid, score 0",
          bool(torch.isfinite(states).all()) and on_lid
          and float(scene.score()[0]) == 0.0)

    # ================= 2. randomization is real (readback) ===============================
    def readback():
        _refresh()
        return (scene.box.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.box.data.root_quat_w[0]),
                scene.lid.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.lid.data.root_quat_w[0]))

    torch.manual_seed(101)
    env.reset()
    a_bp, a_byaw, a_lp, a_lyaw = readback()
    torch.manual_seed(202)
    env.reset()
    b_bp, b_byaw, b_lp, b_lyaw = readback()
    d_box, d_lid = float((a_bp - b_bp).norm()), float((a_lp - b_lp).norm())
    d_byaw, d_lyaw = dyaw(a_byaw, b_byaw), dyaw(a_lyaw, b_lyaw)
    print(f"[smoke] randomization deltas: box={d_box * 1000:.1f}mm/{d_byaw:.1f}deg "
          f"lid={d_lid * 1000:.1f}mm/{d_lyaw:.1f}deg", flush=True)
    check("randomization-is-real: box/lid position AND yaw readback differ across resets",
          d_box > 0.005 and d_lid > 0.005 and d_byaw > 3.0 and d_lyaw > 3.0)

    # ================= 3. subset sampling is real ========================================
    scene.cfg.subset_sample = True
    counts = set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        counts.add(int(scene.present[0].sum()))
    scene.cfg.subset_sample = False
    print(f"[smoke] present-count values over 10 resets: {sorted(counts)}", flush=True)
    check("subset sampling: present count varies across resets (bottle always in)",
          len(counts) >= 2 and min(counts) >= 2)

    # ================= 4. null policy fails ==============================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. oracle on 3 seeds + rubric transitions =========================
    stage_scores: list[float] = []

    def on_stage(tag: str) -> None:
        _refresh()
        stage_scores.append(float(scene.score()[0]))
        _report(f"oracle-{tag}")

    for seed in (0, 1, 2):
        subset = seed == 1  # seed 1 proves success is judged on the sampled subset
        scene.cfg.subset_sample = subset
        torch.manual_seed(seed)
        env.reset()
        _step(30)
        _REC["on"] = seed == 0
        stage_scores.clear()
        stage_scores.append(float(scene.score()[0]))
        kk = int(scene.present[0].sum())
        ok = oracle_solution(env, on_stage=on_stage if seed == 0 else None,
                             speed=0.006 if seed == 0 else 0.012)
        _report(f"oracle-seed-{seed}")
        check(f"oracle reaches success() (seed {seed}"
              + (f", subset {kk}/3 present" if subset else "") + ")", ok)
        _REC["on"] = False
        scene.cfg.subset_sample = False
        if seed == 0:
            # stage_scores = [reset, stow x3, lid-hover, capped]
            expect = [0.55 * k / 3 for k in (1, 2, 3)]
            stows_ok = all(abs(stage_scores[1 + j] - expect[j]) < 0.02 for j in range(3))
            check("rubric: stow transitions hit 0.55*k/K "
                  f"({[f'{v:.2f}' for v in stage_scores[1:4]]})", stows_ok)
            check("rubric: lid-over-box latch lifts score to ~0.75",
                  abs(stage_scores[4] - 0.75) < 0.02)
            mono = all(stage_scores[j] < stage_scores[j + 1] - 1e-6
                       for j in range(len(stage_scores) - 1))
            check("rubric: monotone 0 -> stows -> 0.75 -> 1.0",
                  mono and stage_scores[0] == 0.0 and stage_scores[-1] == 1.0)

    # ================= 6a. negative: the seed's own strategy (cap immediately) ===========
    # close_box's whole plan is "push the lid shut". Executed here: cap the box with the
    # loaded lid, cargo riding on top — teleport the WHOLE tray assembly (lid + items at
    # their exact lid-relative poses) onto the rim. The lid seats; nothing is inside.
    torch.manual_seed(100)
    env.reset()
    _step(40)
    _REC["on"] = True
    _refresh()
    lp = scene.lid.data.root_pos_w.clone()
    lq = scene.lid.data.root_quat_w.clone()
    seat_p, seat_q = _seat_pose()
    lq_conj = lq.clone()
    lq_conj[:, 1:] = -lq_conj[:, 1:]
    item_states = {}
    for name, b in scene.items.items():
        p_loc = quat_apply_inverse(lq, b.data.root_pos_w - lp)
        q_loc = quat_mul(lq_conj, b.data.root_quat_w)
        item_states[name] = (seat_p + quat_apply(seat_q, p_loc), quat_mul(seat_q, q_loc))
    _write_body(scene.lid, seat_p, seat_q)
    for name, (p, q) in item_states.items():
        _write_body(scene.items[name], p, q)
    _step(180)
    _report("seed-strategy")
    check("negative (seed strategy): capping with cargo riding — lid seats, ZERO stowed, "
          "score <= 0.25, no success",
          bool(scene.lid_seated()[0]) and int(scene.stowed()[0].sum()) == 0
          and float(scene.score()[0]) <= 0.25 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 6b. negative: near-miss (lid 40 mm off-axis) ======================
    torch.manual_seed(100)
    env.reset()
    _step(40)
    stow_teleport(range(len(c.items)))
    seat_p, seat_q = _seat_pose()
    off = torch.zeros(env.num_envs, 3, device=env.device)
    off[:, 0] = 0.040
    p = seat_p + quat_apply(seat_q, off)
    p[:, 2] += 0.008
    _write_body(scene.lid, p, seat_q)
    _step(240)
    _report("near-miss-40mm")
    check("negative (near-miss): lid released 40 mm off-axis does not seat, score "
          "pinned ~0.75, no success",
          not bool(scene.lid_seated()[0]) and not bool(scene.success()[0])
          and 0.50 <= float(scene.score()[0]) <= 0.76)

    # ================= 6c. negative: incomplete stow =====================================
    torch.manual_seed(100)
    env.reset()
    _step(40)
    stow_teleport([0, 1])  # bottle + red cube in; green cube left on the ground outside
    _refresh()
    bp = scene.box.data.root_pos_w.clone()
    out_p = bp.clone()
    out_p[:, 1] -= c.outer_half + 0.12
    out_p[:, 2] = c.items[2][2] + 0.003
    _write_body(scene.items[c.items[2][0]], out_p)
    seat_p, seat_q = _seat_pose()
    drop = seat_p.clone()
    drop[:, 2] += 0.008
    _write_body(scene.lid, drop, seat_q)
    _step(200)
    _report("incomplete")
    check("negative (incomplete): lid seated flush but one cube left outside — no success",
          bool(scene.lid_seated()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.80)

    # ================= 6d. negative: upright bottle ======================================
    torch.manual_seed(100)
    env.reset()
    _step(40)
    _REC["on"] = True
    stow_teleport([1, 2])  # cubes in properly
    _refresh()
    bp = scene.box.data.root_pos_w.clone()
    bq = scene.box.data.root_quat_w.clone()
    up_loc = torch.zeros(env.num_envs, 3, device=env.device)
    up_loc[:, 1] = -0.030
    up_loc[:, 2] = c.bot_t + c.items[0][3] / 2 + 0.003
    _write_body(scene.items["bottle"], bp + quat_apply(bq, up_loc), None)  # standing
    _step(30)
    _refresh()
    check("negative (upright bottle): standing inside the box earns NO stow credit",
          not bool(scene.stowed_geo()[0, 0]) and bool(scene.stowed()[0, 1])
          and bool(scene.stowed()[0, 2]))
    seat_p, seat_q = _seat_pose()
    drop = seat_p.clone()
    # release ABOVE the protruding bottle top (140 mm local + lid half + clearance):
    # the bottle sticks ~30 mm past the rim, so this drop is what "closing" costs here
    drop[:, 2] += (c.bot_t + c.items[0][3]) - c.seat_z_local + c.lid_t / 2 + 0.010
    _write_body(scene.lid, drop, seat_q)
    _step(300)
    _report("upright-bottle")
    check("negative (upright bottle): the lid physically cannot seat over it — no success",
          not bool(scene.lid_seated()[0]) and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 6e. negative: tolerance (45 deg twisted lid) ======================
    torch.manual_seed(100)
    env.reset()
    _step(40)
    stow_teleport(range(len(c.items)))
    seat_p, seat_q = _seat_pose()
    half = math.radians(45.0) / 2
    qz45 = torch.zeros(env.num_envs, 4, device=env.device)
    qz45[:, 0], qz45[:, 3] = math.cos(half), math.sin(half)
    _write_body(scene.lid, seat_p, quat_mul(seat_q, qz45))
    _refresh()
    check("negative (tolerance): 45 deg twisted lid at the exact seat pose is rejected "
          "(aperture corners exposed)",
          not bool(scene.lid_seated()[0]) and not bool(scene.success()[0]))

    # ================= 7. calibration probe: lid-release offset sweep ====================
    print("[smoke] CALIBRATION SWEEP (lid release offset along box-x -> seat rate, "
          "3 seeds each)", flush=True)
    offsets = (0.0, 0.005, 0.010, 0.020, 0.035)
    rates: dict[float, int] = {}
    for off_m in offsets:
        hits = 0
        for seed in (10, 11, 12):
            torch.manual_seed(seed)
            env.reset()
            _step(20)
            seat_p, seat_q = _seat_pose()
            off = torch.zeros(env.num_envs, 3, device=env.device)
            off[:, 0] = off_m if seed % 2 == 0 else -off_m
            p = seat_p + quat_apply(seat_q, off)
            p[:, 2] += 0.008
            _write_body(scene.lid, p, seat_q)
            hit = _settle_until(
                lambda: bool(scene.lid_seated()[0]) and bool(scene.lid_still()[0]),
                max_steps=240)
            hits += int(hit)
            print(f"[smoke]   off={off_m * 1000:.0f}mm seed={seed}: seated={hit}", flush=True)
        rates[off_m] = hits
    small = sum(v for k, v in rates.items() if k <= 0.010)
    large = sum(v for k, v in rates.items() if k >= 0.020)
    print("[smoke] SWEEP RESULT: " +
          " | ".join(f"{k * 1000:.0f}mm: {v}/3" for k, v in rates.items()) +
          f"  -> small(<=10mm)={small}/9, large(>=20mm)={large}/6", flush=True)
    check("calibration: small offsets (<=10 mm) seat >= 8/9", small >= 8)
    check("calibration: large offsets (>=20 mm) seat <= 1/6", large <= 1)

    # ================= save + verdict ====================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="sim_gen.lid_tray_stow")
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

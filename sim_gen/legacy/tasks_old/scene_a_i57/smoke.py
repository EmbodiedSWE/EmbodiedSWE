"""Smoke / oracle test for TableclothPullScene (sim_gen task `scene_a_i57`) —
NullRobot, teleport-oracle, RECORDED.

Battery (compass_crate / pen_holder smoke skeleton):
  1. settle/no-NaN      — reset layout settles finite: trophy standing ON the runner,
                          at rest, score ~0, nothing latched;
  2. randomization      — READBACK: runner heading + position, the trophy's seat on the
                          runner, and the pad position all move across seeded resets;
  3. null-policy-fails  — 240 idle steps -> score ~0, no success, no latches;
  4. oracle x3 seeds    — read the runner heading, YANK it out (12 mm/substep kinematic
                          drive with matched velocity — real friction acts on the trophy),
                          verify the trophy stood clear, carry the runner to the pad,
                          reach success() and score 1.0;
  5. monotonicity       — 0 (reset) -> partial withdrawal -> extracted -> stowed: score
                          strictly increases, partials < 1.0, final is success;
  6. negative A (seed)  — the seed's own strategy (pick the block up / relocate it) trips
                          the permanent `disturbed` latch; then finishing everything and
                          putting the trophy back EXACTLY at its anchor yields perfect
                          final geometry yet stays capped at 0.05 (latch permanence /
                          anti-teleport: `extracted` never latched while disturbed);
  7. negative B (slow)  — the SAME pull done quasi-statically (0.5 mm/substep): friction
                          drags the trophy along past the latch radius -> capped, no
                          success. The manner is the task;
  8. near-miss          — yank stopped with the runner still under the trophy: no
                          extraction latch, partial credit < 0.5; finishing recovers;
  9. negative C (tipped)— extracted and stowed but the trophy left lying on its side:
                          no success (preserved gate), score capped at the milestones;
 10. calibration probe  — pull-speed sweep 0.5/2/5/12 mm-per-substep on identical resets,
                          published trophy-displacement table: the quasi-static end drags
                          past 60 mm (latch), the yank end stays under 30 mm.

Run (forge): python -u -m simgen_tasks.scene_a_i57.smoke --headless
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
    from simgen_tasks.scene_a_i57 import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

YANK_SPEED = 0.012  # m per substep (1.44 m/s at 120 Hz) — the oracle's yank
CREEP_SPEED = 0.0005  # m per substep (0.06 m/s) — the failing quasi-static control


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _env_of(scene_or_env):
    return scene_or_env if hasattr(scene_or_env, "iscene") else scene_or_env.env


def _heading(quat_row: torch.Tensor) -> float:
    """World yaw (rad) of a body's +x axis, from its root quaternion (1, 4) or (4,)."""
    from isaaclab.utils.math import quat_apply

    q = quat_row.reshape(1, 4)
    ex = torch.tensor([[1.0, 0.0, 0.0]], device=q.device)
    f = quat_apply(q, ex)[0]
    return math.atan2(float(f[1]), float(f[0]))


def _pull_plan(scene) -> tuple[tuple[float, float], float, float, float]:
    """(unit pull dir d_xy, extraction travel, sign, trophy local x) for env 0. The pull
    runs along the runner's long axis, on the side pointing AWAY from the stow pad, far
    enough that the trailing edge passes the trophy by half-diagonal + margin + 60 mm."""
    from isaaclab.utils.math import quat_apply_inverse

    c = scene.cfg
    yaw = _heading(scene.runner.data.root_quat_w[0])
    ax = (math.cos(yaw), math.sin(yaw))
    rel = (scene.trophy.data.root_pos_w - scene.runner.data.root_pos_w)[0:1]
    loc = quat_apply_inverse(scene.runner.data.root_quat_w[0:1], rel)[0]
    t_x = float(loc[0])
    r_xy = (scene.runner.data.root_pos_w - scene.env_origins)[0, :2]
    pad_vec = scene.pad_xy[0] - r_xy
    s = -1.0 if (ax[0] * float(pad_vec[0]) + ax[1] * float(pad_vec[1])) > 0 else 1.0
    d = (s * ax[0], s * ax[1])
    half_diag = math.hypot(c.trophy_size[0], c.trophy_size[1]) / 2
    travel = c.runner_size[0] / 2 + s * t_x + half_diag + c.clear_margin + 0.06
    return d, travel, s, t_x


def _drive_runner(scene, step_fn, d_xy, dist: float, speed: float, stop_pred=None) -> float:
    """Kinematic per-substep drive of the runner along `d_xy` for `dist` meters at
    `speed` m/substep, with MATCHED linear velocity written each substep (PhysX friction
    sees the true sliding speed — the trophy's drag is real physics) and +g*dt on vz so
    gravity integration cancels. Ends with a zero-velocity catch. Returns the distance
    actually driven (stop_pred can end the pull early)."""
    env = scene.env
    all_ids = torch.arange(env.num_envs, device=env.device)
    dt = env.dt
    g_dt = 9.81 * dt
    st0 = scene.runner.data.root_state_w[all_ids].clone()
    k = max(1, int(math.ceil(dist / speed)))
    adv = 0.0
    for i in range(1, k + 1):
        adv = min(i * speed, dist)
        st = st0.clone()
        st[:, 0] = st0[:, 0] + d_xy[0] * adv
        st[:, 1] = st0[:, 1] + d_xy[1] * adv
        st[:, 7] = d_xy[0] * speed / dt
        st[:, 8] = d_xy[1] * speed / dt
        st[:, 9] = g_dt
        st[:, 10:13] = 0.0
        scene.runner.write_root_state_to_sim(st, all_ids)
        step_fn(1)
        if stop_pred is not None and stop_pred():
            break
    st = st0.clone()
    st[:, 0] = st0[:, 0] + d_xy[0] * adv
    st[:, 1] = st0[:, 1] + d_xy[1] * adv
    st[:, 7:13] = 0.0
    scene.runner.write_root_state_to_sim(st, all_ids)
    env.iscene.update(0.0)
    return adv


def _carry_runner_to_pad(scene, step_fn) -> None:
    """Kinematic paced carry of the (empty) runner: lift to 0.13 m, translate to the pad
    center while turning to the pad heading (mod 180), lower onto the pad, release."""
    env = scene.env
    all_ids = torch.arange(env.num_envs, device=env.device)
    dt = env.dt
    g_dt = 9.81 * dt
    c = scene.cfg
    p0 = (scene.runner.data.root_pos_w - scene.env_origins)[0].clone()
    yaw0 = _heading(scene.runner.data.root_quat_w[0])
    pad_p = scene.pad_xy[0]
    pad_yaw = _heading(scene.pad.data.root_quat_w[0])
    dyaw = _wrap(pad_yaw - yaw0)
    if abs(dyaw) > math.pi / 2:  # mod-180 alignment: turn to the nearer end
        dyaw = _wrap(dyaw - math.copysign(math.pi, dyaw))
    z_fly = 0.13
    z_rest = c.pad_size[2] + c.runner_size[2] / 2 + 0.004
    dist = math.hypot(float(pad_p[0] - p0[0]), float(pad_p[1] - p0[1]))

    def write(x: float, y: float, z: float, yaw: float, vz: float = 0.0) -> None:
        st = torch.zeros(env.num_envs, 13, device=env.device)
        st[:, 0] = x
        st[:, 1] = y
        st[:, 2] = z
        st[:, 3] = math.cos(yaw / 2)
        st[:, 6] = math.sin(yaw / 2)
        st[:, 0:3] += env.iscene.env_origins
        st[:, 9] = vz + g_dt
        scene.runner.write_root_state_to_sim(st, all_ids)

    for i in range(1, 21):  # lift
        z = float(p0[2]) + (z_fly - float(p0[2])) * i / 20
        write(float(p0[0]), float(p0[1]), z, yaw0)
        step_fn(1)
    k = max(1, int(math.ceil(dist / 0.006)))
    for i in range(1, k + 1):  # translate + turn
        f = i / k
        write(float(p0[0]) + (float(pad_p[0]) - float(p0[0])) * f,
              float(p0[1]) + (float(pad_p[1]) - float(p0[1])) * f,
              z_fly, yaw0 + dyaw * f)
        step_fn(1)
    for i in range(1, 19):  # lower
        z = z_fly + (z_rest - z_fly) * i / 18
        write(float(pad_p[0]), float(pad_p[1]), z, yaw0 + dyaw)
        step_fn(1)
    st = torch.zeros(env.num_envs, 13, device=env.device)  # release at rest
    st[:, 0] = float(pad_p[0])
    st[:, 1] = float(pad_p[1])
    st[:, 2] = z_rest
    st[:, 3] = math.cos((yaw0 + dyaw) / 2)
    st[:, 6] = math.sin((yaw0 + dyaw) / 2)
    st[:, 0:3] += env.iscene.env_origins
    scene.runner.write_root_state_to_sim(st, all_ids)
    env.iscene.update(0.0)


def oracle_solution(scene_or_env, step_fn=None, verbose: bool = True) -> bool:
    """Teleport-oracle: solve the CURRENT episode. Reads the runner heading and the
    trophy's seat, YANKS the runner out along its axis away from the pad (12 mm/substep,
    matched velocity — the trophy's ~5 mm drag is real friction physics), waits for the
    extraction latch, then carries the empty runner onto the stow pad and releases.
    Returns True iff scene.success()."""
    env = _env_of(scene_or_env)
    scene = env.scene
    no_action = torch.empty(0, device=env.device)

    def _step(k: int) -> None:
        if step_fn is not None:
            step_fn(k)
        else:
            for _ in range(k):
                env.step(no_action)

    def wait(pred, max_steps: int = 240) -> bool:
        waited = 0
        while waited < max_steps and not pred():
            _step(10)
            waited += 10
        return bool(pred())

    d, travel, _s, _tx = _pull_plan(scene)
    _drive_runner(scene, _step, d, travel, YANK_SPEED)
    ok = wait(lambda: bool(scene.extracted[0]) and bool(scene.trophy_preserved()[0]))
    if verbose:
        disp = float((scene._local(scene.trophy) - scene.trophy_anchor)[0].norm())
        print(f"[oracle] yank done: trophy disp={disp * 1000:.1f}mm "
              f"extracted={bool(scene.extracted[0])} disturbed={bool(scene.disturbed[0])}",
              flush=True)
    if not ok:
        return False
    _carry_runner_to_pad(scene, _step)
    wait(lambda: bool(scene.success()[0]))
    return bool(scene.success()[0])


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tablecloth_pull")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.45, -1.45, 1.05)) + o),
                                tuple(np.array((0.0, 0.0, 0.05)) + o),
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

    def settle_until(pred, max_steps: int = 300, poll: int = 10) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def trophy_disp() -> float:
        return float((scene._local(scene.trophy) - scene.trophy_anchor)[0].norm())

    def report(tag: str) -> None:
        bot = float(scene._local(scene.trophy)[0, 2]) - c.trophy_size[2] / 2
        print(f"[smoke] {tag:14s} | tdisp={trophy_disp() * 1000:6.1f}mm "
              f"bot={bot * 1000:5.1f}mm "
              f"disturbed={bool(scene.disturbed[0])} extracted={bool(scene.extracted[0])} "
              f"preserved={bool(scene.trophy_preserved()[0])} "
              f"stowed={bool(scene.runner_stowed()[0])} "
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
    st_r = scene.runner.data.root_state_w
    st_t = scene.trophy.data.root_state_w
    check("settle: states finite; trophy standing ON the runner, in place, at rest",
          bool(torch.isfinite(st_r).all()) and bool(torch.isfinite(st_t).all())
          and bool(scene.trophy_upright()[0]) and bool(scene.trophy_in_place()[0])
          and bool(scene.trophy_settled()[0]) and not bool(scene.clear_of_runner()[0]))
    check("settle: score ~0 at reset, no success, nothing latched",
          float(scene.score()[0]) <= 0.005 and not bool(scene.success()[0])
          and not bool(scene.disturbed[0]) and not bool(scene.extracted[0]))

    # =========================== 2. randomization is real ===================================
    reads = []
    for s in (21, 22, 23, 24):
        torch.manual_seed(s)
        env.reset()
        step(4)
        _d, _tv, _sgn, t_x = _pull_plan(scene)
        yaw = _heading(scene.runner.data.root_quat_w[0])
        r = (scene.runner.data.root_pos_w - scene.env_origins)[0]
        p = scene.pad_xy[0]
        reads.append((math.cos(yaw), math.sin(yaw), float(r[0]), float(r[1]),
                      t_x, float(p[0]), float(p[1])))
    arr = np.array(reads)
    print("[smoke] randomization readback (cos_yaw, sin_yaw, rx, ry, trophy_seat_x, "
          f"pad_x, pad_y):\n{np.round(arr, 3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: runner heading, runner xy, trophy seat and pad position all "
          "move (readback)",
          (spread[0] > 0.2 or spread[1] > 0.2) and (spread[2] > 0.02 or spread[3] > 0.02)
          and spread[4] > 0.01 and (spread[5] > 0.10 or spread[6] > 0.10))

    # =========================== 3. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    check("null policy: score ~0, no success, no latches after 240 idle steps",
          float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0])
          and not bool(scene.disturbed[0]) and not bool(scene.extracted[0]))

    # =========================== 4. oracle on 3 seeds =======================================
    for s in (0, 1, 2):
        torch.manual_seed(s)
        env.reset()
        step(20)
        ok = oracle_solution(env, step_fn=step)
        report(f"oracle-seed{s}")
        check(f"oracle reaches success() on seed {s} (score 1.0)",
              ok and bool(scene.success()[0]) and float(scene.score()[0]) == 1.0)

    # =========================== 5. rubric monotonicity =====================================
    torch.manual_seed(41)
    env.reset()
    step(20)
    s0 = float(scene.score()[0])
    d, travel, _sgn, _tx = _pull_plan(scene)
    _drive_runner(scene, step, d, 0.06, YANK_SPEED)  # partial withdrawal, trophy aboard
    step(20)
    s1 = float(scene.score()[0])
    report("partial")
    d, travel, _sgn, _tx = _pull_plan(scene)  # re-plan from the current poses
    _drive_runner(scene, step, d, travel, YANK_SPEED)
    settle_until(lambda: bool(scene.extracted[0]) and bool(scene.trophy_preserved()[0]))
    s2 = float(scene.score()[0])
    report("extracted")
    _carry_runner_to_pad(scene, step)
    settle_until(lambda: bool(scene.success()[0]))
    s3 = float(scene.score()[0])
    report("stowed")
    print(f"[smoke] monotonicity ladder: {s0:.3f} -> {s1:.3f} -> {s2:.3f} -> {s3:.3f}",
          flush=True)
    check("monotonicity: score strictly increases along partial -> extracted -> stowed",
          s0 < s1 < s2 < s3)
    check("monotonicity: partials < 1.0; final is success with score 1.0",
          s2 < 1.0 and s3 == 1.0 and bool(scene.success()[0]))

    # =========================== 6. negative A: the seed's own strategy =====================
    # CALVIN's plan skeleton is grasp-block -> transport -> release. Lifting the trophy
    # trips the permanent latch; then even doing everything else perfectly AND putting the
    # trophy back exactly at its anchor leaves perfect final geometry — still capped.
    torch.manual_seed(51)
    env.reset()
    step(20)
    anchor = scene.trophy_anchor[0].clone()
    st = scene.trophy.data.root_state_w[all_ids].clone()
    st[:, 2] += 0.15
    st[:, 7:13] = 0.0
    scene.trophy.write_root_state_to_sim(st, all_ids)
    step(3)
    report("lifted")
    check("negative A (seed strategy): lifting the trophy trips the disturbed latch",
          bool(scene.disturbed[0]))
    # park the trophy aside (perpendicular to the runner, away from the pad), clear the
    # runner to the pad, then put the trophy back EXACTLY at its anchor
    d, _tv, _sgn, _tx = _pull_plan(scene)
    perp = (-d[1], d[0])
    r_xy = (scene.runner.data.root_pos_w - scene.env_origins)[0, :2]
    pad_vec = scene.pad_xy[0] - r_xy
    if perp[0] * float(pad_vec[0]) + perp[1] * float(pad_vec[1]) > 0:
        perp = (-perp[0], -perp[1])
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = float(anchor[0]) + perp[0] * 0.25
    st[:, 1] = float(anchor[1]) + perp[1] * 0.25
    st[:, 2] = c.trophy_size[2] / 2 + 0.002
    st[:, 3] = 1.0
    st[:, 0:3] += env.iscene.env_origins
    scene.trophy.write_root_state_to_sim(st, all_ids)
    step(30)
    _drive_runner(scene, step, d, 0.30, YANK_SPEED)
    step(20)
    _carry_runner_to_pad(scene, step)
    settle_until(lambda: bool(scene.runner_stowed()[0]))
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = float(anchor[0])
    st[:, 1] = float(anchor[1])
    st[:, 2] = c.trophy_size[2] / 2 + 0.001
    st[:, 3] = 1.0
    st[:, 0:3] += env.iscene.env_origins
    scene.trophy.write_root_state_to_sim(st, all_ids)
    settle_until(lambda: bool(scene.trophy_preserved()[0]), max_steps=120)
    report("geo-perfect")
    check("negative A: perfect final geometry (trophy home + runner stowed) stays capped "
          "<= 0.05, no success, extraction never latched",
          bool(scene.trophy_preserved()[0]) and bool(scene.runner_stowed()[0])
          and not bool(scene.extracted[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.0505)  # float32 clamp(max=0.05) reads 0.05000000074

    # =========================== 7. negative B: quasi-static pull ===========================
    # The SAME trajectory at 0.5 mm/substep: there is no other horizontal force on the
    # trophy, so it rides the runner past the 60 mm latch radius long before extraction.
    torch.manual_seed(61)
    env.reset()
    step(20)
    d, travel, _sgn, _tx = _pull_plan(scene)
    _drive_runner(scene, step, d, travel, CREEP_SPEED,
                  stop_pred=lambda: bool(scene.disturbed[0]))
    step(30)
    report("slow-pull")
    check("negative B (quasi-static pull): the trophy rides past the latch radius — "
          "capped <= 0.05, no success",
          bool(scene.disturbed[0]) and trophy_disp() > c.stray_tol
          and float(scene.score()[0]) <= 0.0505 and not bool(scene.success()[0]))

    # =========================== 8. near-miss + recovery ====================================
    # Yank stopped early: the runner's trailing edge still 30 mm under the trophy — no
    # extraction latch, shaping credit only; finishing the yank + stow recovers success.
    torch.manual_seed(81)
    env.reset()
    step(20)
    d, travel, sgn, t_x = _pull_plan(scene)
    short = sgn * t_x + c.runner_size[0] / 2 - 0.03  # trophy ends 30 mm inside the edge
    _drive_runner(scene, step, d, short, YANK_SPEED)
    step(30)
    report("near-miss")
    check("near-miss: runner stopped still under the trophy -> no extraction, partial "
          "credit < 0.5, no success",
          not bool(scene.extracted[0]) and not bool(scene.clear_of_runner()[0])
          and 0.0 < float(scene.score()[0]) < 0.5 and not bool(scene.success()[0]))
    ok = oracle_solution(env, step_fn=step)
    report("recovered")
    check("near-miss recovery: finishing the yank + stow reaches success", ok)

    # =========================== 9. negative C: trophy tipped after extraction ==============
    # A clean yank, then the trophy is knocked over IN PLACE (inside the stray sphere) and
    # the runner stowed: milestones only — the preserved gate kills success.
    torch.manual_seed(71)
    env.reset()
    step(20)
    d, travel, _sgn, _tx = _pull_plan(scene)
    _drive_runner(scene, step, d, travel, YANK_SPEED)
    settle_until(lambda: bool(scene.extracted[0]) and bool(scene.trophy_preserved()[0]))
    anchor = scene.trophy_anchor[0].clone()
    c45 = math.cos(math.pi / 4)
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = float(anchor[0])
    st[:, 1] = float(anchor[1])
    st[:, 2] = c.trophy_size[1] / 2 + 0.002  # lying on its side
    st[:, 3] = c45
    st[:, 4] = c45
    st[:, 0:3] += env.iscene.env_origins
    scene.trophy.write_root_state_to_sim(st, all_ids)
    step(30)
    _carry_runner_to_pad(scene, step)
    settle_until(lambda: bool(scene.runner_stowed()[0]))
    report("tipped")
    check("negative C: extracted + stowed but the trophy lies tipped -> no success, "
          "score <= 0.85, latch NOT tripped (it fell in place)",
          bool(scene.extracted[0]) and bool(scene.runner_stowed()[0])
          and not bool(scene.disturbed[0]) and not bool(scene.trophy_upright()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.85)

    # =========================== 10. calibration probe ======================================
    # The physics the task is made of: identical resets, one pull each at growing speed;
    # published trophy displacement. Quasi-static drags the trophy the full distance to
    # the runner edge (geometry, friction-independent); the yank leaves millimeters.
    print("[smoke] CALIBRATION: pull speed (mm/substep) -> trophy displacement", flush=True)
    cal = []
    for spd in (0.0005, 0.002, 0.005, 0.012):
        torch.manual_seed(91)
        env.reset()
        step(10)
        d, travel, _sgn, _tx = _pull_plan(scene)
        _drive_runner(scene, step, d, travel, spd)
        step(40)
        disp = trophy_disp()
        cal.append((spd, disp, bool(scene.disturbed[0]), bool(scene.extracted[0])))
        print(f"[smoke]   speed={spd * 1000:5.1f}mm/substep ({spd * 120:4.2f} m/s): "
              f"trophy disp={disp * 1000:6.1f}mm disturbed={bool(scene.disturbed[0])} "
              f"extracted={bool(scene.extracted[0])}", flush=True)
    check("calibration: quasi-static pull (0.5 mm/substep) drags the trophy past the "
          "60 mm latch", cal[0][1] > c.stray_tol and cal[0][2])
    check("calibration: yank (12 mm/substep) leaves the trophy within 30 mm and "
          "extracts cleanly", cal[-1][1] < 0.03 and cal[-1][3] and not cal[-1][2])

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.tablecloth_pull")
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

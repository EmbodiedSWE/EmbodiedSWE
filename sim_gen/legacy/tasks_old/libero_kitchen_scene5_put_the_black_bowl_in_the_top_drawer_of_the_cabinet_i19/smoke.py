"""Smoke / oracle test for BarredDrawerScene — NullRobot, teleport-oracle, RECORDED.

One linear run (the pen_holder / pressure_plate smoke skeleton):
  1. settle    — reset layout settles clean (no NaN, drawer shut, bar barred, score 0);
  2. random    — randomization is REAL (readback poses differ across seeded resets);
  3. null      — null policy: 2 s of nothing scores ~0, no success;
  4-6. oracle  — 3 seeds: carry the bar out of its cradles (kinematic carry), pull the
                 drawer open with an honest 3 N force, drop the bowl into the basin,
                 push the drawer shut with the payload riding inside -> success + 1.0;
  7. rubric    — monotone staged milestones from the seed-0 oracle:
                 0 -> 0.15 (bar cleared) -> 0.35 (opened) -> 0.60 (stowed) -> 1.0 (shut);
  8. negative A — THE SEED'S FIRST MOVE: pulling the BARRED drawer with the oracle's own
                 force moves it < 6 cm (measured contact block) and scores ~0;
  9. negative A2 — remove the bar, apply the SAME pull: the drawer opens past open_min
                 (proves the bar is the only blocker AND validates the force convention);
  10. negative B — THE SEED'S GOAL STATE: bowl stowed in the OPEN drawer is NOT success
                 (score pinned at 0.60 — the drawer must end shut);
  11. negative C — bowl upside-down inside the shut drawer: in_basin but never success,
                 score pinned below the 0.85 tier (upright clause);
  12. near-miss D — drawer ajar (7 cm > closed_tol 3 cm) with the bowl stowed upright:
                 not success (tolerance control);
  13. near-miss E — bowl on the counter directly ABOVE the shut basin: right xy, wrong
                 side of the counter — no credit;
  14-15. probe — calibration cliff: pull the drawer while the bar is HELD at +2 / +5 /
                 +8 cm lift; 2 cm must still bar, 8 cm must clear (the 5 cm midpoint is
                 published, not asserted).

Bodies are driven straight through scene handles; the NullRobot applies nothing. Frames
are saved to ./frames.npz (CWD) for the pipeline. Prints the acceptance marker
`SIM_GEN_SMOKE: ALL PASS n/n` and hard-exits (watchdog) — Kit teardown hangs otherwise.

Run (on a GPU node with the isaaclab env):
    python smoke.py --headless
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=10)
parser.add_argument("--max_frames", type=int, default=520)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the L20 driver version and silently rejects RTX ->
# annotator returns EMPTY frames. Disable the check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math
import threading

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import robobench
from robobench.core import ENVS

robobench.discover()

try:  # forge runs `python -m simgen_tasks.<task>.smoke`; local runs `python smoke.py`
    from . import scene as scene_mod  # noqa: F401  (registers "sim_gen.barred_drawer")
except ImportError:
    import scene as scene_mod  # noqa: F401


# ----- shared helpers (module level so the exported oracle is standalone) -----------------------
def _settle_until(pred, step_fn, max_steps: int = 450, poll: int = 15) -> bool:
    """Step in poll-sized chunks until pred() or budget out (settling time is physics,
    not what the checks are about)."""
    if pred():
        return True
    waited = 0
    while waited < max_steps:
        step_fn(poll)
        waited += poll
        if pred():
            return True
    return False


def _write_body(env, body, world_pos, quat) -> None:
    """Teleport `body` to WORLD `world_pos` (zero velocity) in every env."""
    n = env.num_envs
    st = torch.zeros(n, 13, device=env.device)
    st[:, 0:3] = torch.as_tensor(world_pos, device=env.device, dtype=torch.float32)
    st[:, 3:7] = torch.as_tensor(quat, device=env.device, dtype=torch.float32)
    body.write_root_state_to_sim(st, torch.arange(n, device=env.device))


def _assembly_world(scene, local_xyz) -> torch.Tensor:
    """Carcass-frame point -> world (env 0)."""
    from isaaclab.utils.math import quat_apply

    p = torch.as_tensor(local_xyz, device=scene.env.device, dtype=torch.float32)
    return scene.carcass.data.root_pos_w[0] + quat_apply(
        scene.carcass.data.root_quat_w[0:1], p.unsqueeze(0))[0]


def _pull(scene, newton: float) -> None:
    """Constant force on the drawer along its OWN slide axis (body-frame +x; the
    IsaacLab external wrench is applied in the body's local frame). Persists in the
    wrench buffer every step until overwritten with zero."""
    n = scene.env.num_envs
    dev = scene.env.device
    f = torch.zeros(n, 1, 3, device=dev)
    f[:, 0, 0] = newton
    scene.drawer.set_external_force_and_torque(f, torch.zeros(n, 1, 3, device=dev))


def _carry_bar(scene, step_fn, waypoints, steps_per_leg) -> None:
    """Kinematic carry of the bar along carcass-frame waypoints (gravity-compensated
    pre-step pin — the pen_holder held-body pattern), then release at the last point."""
    g_dt = 9.81 * scene.env.dt
    n = scene.env.num_envs
    dev = scene.env.device
    ids = torch.arange(n, device=dev)
    quat = scene.carcass.data.root_quat_w[0].clone()
    prev = scene.bar.data.root_pos_w[0].clone()
    for target, k in zip(waypoints, steps_per_leg):
        tgt = _assembly_world(scene, target)
        for t in range(k):
            f = (t + 1) / k
            pos = prev + (tgt - prev) * f
            st = torch.zeros(n, 13, device=dev)
            st[:, 0:3] = pos
            st[:, 3:7] = quat
            st[:, 9] = g_dt  # cancel the gravity kick -> truly static hold
            scene.bar.write_root_state_to_sim(st, ids)
            step_fn(1)
        prev = tgt
    # release: re-pin at zero velocity, let it fall free
    st = torch.zeros(n, 13, device=dev)
    st[:, 0:3] = prev
    st[:, 3:7] = quat
    scene.bar.write_root_state_to_sim(st, ids)
    step_fn(30)


def _drop_bowl(scene, env, step_fn, xy_off=(0.0, 0.0), drop_h: float = 0.05) -> bool:
    """Release the bowl `drop_h` above the LIVE basin floor, centred + xy_off in the
    drawer frame — a genuine drop into the (wherever-it-is) basin. One retry."""
    from isaaclab.utils.math import quat_apply

    c = scene.cfg
    for attempt, off in enumerate((xy_off, (0.012, -0.010))):
        local = torch.tensor([off[0], off[1], c.bowl_h / 2 + drop_h],
                             device=env.device, dtype=torch.float32)
        pos = scene.drawer.data.root_pos_w[0] + quat_apply(
            scene.drawer.data.root_quat_w[0:1], local.unsqueeze(0))[0]
        _write_body(env, scene.bowl, pos, scene.drawer.data.root_quat_w[0])
        ok = _settle_until(
            lambda: bool(scene.in_basin()[0])
            and float(scene.bowl.data.root_lin_vel_w[0].norm()) < c.settle_speed,
            step_fn, max_steps=240)
        if ok:
            return True
        if attempt == 0:
            print("[smoke]   drop retry: bowl missed the basin", flush=True)
    return False


def oracle_solution(scene_or_env, step_fn=None, milestones=None,
                    stop_before_close: bool = False) -> bool:
    """Teleport-oracle solve of the CURRENT episode: carry the security bar out of its
    cradles and set it on the ground clear of the slide, pull the drawer open with an
    honest body-frame force, drop the bowl into the basin, then push the drawer shut
    with the payload inside. Returns success()[0]."""
    env = scene_or_env if hasattr(scene_or_env, "iscene") else scene_or_env.env
    scene = env.scene
    c = scene.cfg
    if step_fn is None:
        no_action = torch.empty(0, device=env.device)

        def step_fn(k):
            for _ in range(k):
                env.step(no_action)

    def mark():
        if milestones is not None:
            step_fn(2)
            milestones.append(float(scene.score()[0]))

    # stage 1: unbar — straight up out of the cradle slot, over, down, release.
    bh = c.bar_home
    _carry_bar(scene, step_fn,
               waypoints=[(bh[0], bh[1], bh[2] + 0.16),
                          (0.45, -0.32, bh[2] + 0.16),
                          (0.45, -0.32, 0.06)],
               steps_per_leg=[45, 55, 30])
    mark()  # bar cleared -> 0.15

    # stage 2: slide the drawer open (honest force pull to the rail stop).
    _pull(scene, c.pull_force)
    opened = _settle_until(lambda: float(scene.drawer_disp()[0]) > c.travel - 0.02,
                           step_fn, max_steps=360)
    _pull(scene, 0.0)
    _settle_until(lambda: float(scene.drawer.data.root_lin_vel_w[0].norm()) < c.settle_speed,
                  step_fn, max_steps=120)
    if not opened:
        print(f"[smoke]   oracle: drawer only reached "
              f"{float(scene.drawer_disp()[0]) * 1000:.0f}mm", flush=True)
    mark()  # opened -> 0.35

    # stage 3: stow the bowl (genuine drop into the exposed basin).
    _drop_bowl(scene, env, step_fn)
    mark()  # stowed -> 0.60

    if stop_before_close:
        return bool(scene.success()[0])

    # stage 4: push the drawer shut with the bowl riding inside.
    _pull(scene, -c.close_force)
    _settle_until(lambda: bool(scene.shut()[0]), step_fn, max_steps=360)
    _pull(scene, 0.0)
    ok = _settle_until(lambda: bool(scene.success()[0]), step_fn, max_steps=300)
    mark()  # success -> 1.0
    return ok


# ----- main -------------------------------------------------------------------------------------
def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("sim_gen.barred_drawer")().build(
        num_envs=args.num_envs, device=device,
        scene_cfg=scene_mod.BarredDrawerSceneCfg())
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
        env.sim.set_camera_view(tuple(np.array((1.15, -1.05, 0.75)) + o),
                                tuple(np.array((0.10, 0.0, 0.10)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (800, 500))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames — check RTX recipe",
                  flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0
    holds: list = []  # [(body, state13)] pinned pre-step (calibration bar hold)
    g_dt = 9.81 * env.dt

    def step(k: int, render: bool = True) -> None:
        nonlocal step_i
        for _ in range(k):
            for body, st in holds:
                pre = st.clone()
                pre[:, 9] += g_dt
                body.write_root_state_to_sim(pre, all_ids)
            env.step(no_action, render=render)
            if annot is not None and step_i % args.record_every == 0 \
                    and len(frames) < args.max_frames:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1
        if holds:
            for body, st in holds:
                body.write_root_state_to_sim(st, all_ids)
            env.iscene.update(0.0)

    def report(tag: str) -> None:
        print(f"[smoke] {tag:12s} | disp={float(scene.drawer_disp()[0]) * 1000:6.1f}mm "
              f"barred={bool(scene.bar_blocking()[0])} "
              f"in_basin={bool(scene.in_basin()[0])} up={bool(scene.bowl_upright()[0])} "
              f"shut={bool(scene.shut()[0])} cleared={bool(scene._bar_cleared[0])} "
              f"opened={bool(scene._opened[0])} score={float(scene.score()[0]):.2f} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    # =========================== 1. settle / no-NaN =========================================
    torch.manual_seed(3)
    env.reset()
    step(60)
    report("settle")
    states = [b.data.root_state_w for b in (scene.carcass, scene.drawer, scene.bar, scene.bowl)]
    no_nan = not any(bool(torch.isnan(s).any()) for s in states)
    check("settle: no NaN, drawer shut, bar barred, score 0",
          no_nan and abs(float(scene.drawer_disp()[0])) < 0.010
          and bool(scene.bar_blocking()[0]) and float(scene.score()[0]) < 0.01)

    # =========================== 2. randomization is real ===================================
    def snapshot() -> torch.Tensor:
        step(2)
        return torch.cat([scene.carcass.data.root_pos_w[0, :2],
                          scene.carcass.data.root_quat_w[0],
                          scene.bar.data.root_pos_w[0, :2],
                          scene.bowl.data.root_pos_w[0, :2]]).clone()

    torch.manual_seed(101)
    env.reset()
    snap_a = snapshot()
    torch.manual_seed(202)
    env.reset()
    snap_b = snapshot()
    delta = float((snap_a - snap_b).abs().max())
    print(f"[smoke] randomization readback max delta = {delta:.4f}", flush=True)
    check("randomization is real (readback differs across seeded resets)", delta > 0.01)

    # =========================== 3. null policy =============================================
    torch.manual_seed(5)
    env.reset()
    step(240)
    report("null")
    check("null policy: score ~0, no success",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # =========================== 4-6. oracle, 3 seeds (+ milestones on seed 0) ==============
    milestones: list[float] = []
    for s in range(3):
        torch.manual_seed(s)
        env.reset()
        step(30)
        ms = milestones if s == 0 else None
        if ms is not None:
            step(2)
            ms.append(float(scene.score()[0]))
        ok = oracle_solution(env, step_fn=step, milestones=ms)
        report(f"oracle-s{s}")
        check(f"oracle seed {s}: success + score 1.0",
              ok and float(scene.score()[0]) >= 0.999)

    # =========================== 7. rubric monotonicity =====================================
    print(f"[smoke] rubric milestones: "
          + " -> ".join(f"{v:.2f}" for v in milestones), flush=True)
    mono = all(b >= a for a, b in zip(milestones, milestones[1:]))
    check("rubric monotone staged milestones 0 -> 0.15 -> 0.35 -> 0.60 -> 1.0",
          len(milestones) == 5 and mono
          and milestones[0] < 0.05 and abs(milestones[1] - 0.15) < 0.011
          and abs(milestones[2] - 0.35) < 0.011 and abs(milestones[3] - 0.60) < 0.011
          and milestones[4] >= 0.999)

    # =========================== 8. negative A: pull the BARRED drawer ======================
    # The seed's first move — slide the drawer open — executed with the oracle's own
    # force while the bar is in place. The bar (a free body, pure contact) must stop the
    # drawer inside 6 cm, nothing latches, score stays ~0.
    torch.manual_seed(11)
    env.reset()
    step(30)
    _pull(scene, c.pull_force)
    step(240)
    d_barred = float(scene.drawer_disp()[0])
    _pull(scene, 0.0)
    step(30)
    report("barred-pull")
    print(f"[smoke]   barred pull displacement = {d_barred * 1000:.1f}mm "
          f"(block is measured contact, not script)", flush=True)
    check("seed strategy (pull the barred drawer) blocked < 60mm, score ~0",
          d_barred < 0.060 and not bool(scene._opened[0])
          and float(scene.score()[0]) < 0.05)

    # =========================== 9. negative A2: same pull, bar removed =====================
    bar_q = scene.carcass.data.root_quat_w[0]
    _write_body(env, scene.bar, _assembly_world(scene, (0.50, -0.40, 0.015)), bar_q)
    step(10)
    _pull(scene, c.pull_force)
    opened = _settle_until(lambda: float(scene.drawer_disp()[0]) > c.open_min, step,
                           max_steps=360)
    _pull(scene, 0.0)
    step(30)
    report("unbarred")
    check("same pull with the bar removed opens the drawer (bar is the only blocker)",
          opened and bool(scene._opened[0]))

    # =========================== 10. negative B: the seed's GOAL state ======================
    # Full seed plan (after unbarring): bowl into the OPEN drawer, walk away. Here that
    # is only 0.60 — the drawer must end shut.
    torch.manual_seed(13)
    env.reset()
    step(30)
    oracle_solution(env, step_fn=step, stop_before_close=True)
    step(20)
    report("open-stow")
    check("seed goal state (bowl in the OPEN drawer): NOT success, score pinned at 0.60",
          bool(scene.in_basin()[0]) and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.60) < 0.011)

    # =========================== 11. negative C: flipped bowl in the shut drawer ============
    torch.manual_seed(17)
    env.reset()
    step(30)
    from isaaclab.utils.math import quat_apply, quat_mul

    dq = scene.drawer.data.root_quat_w[0]
    flip = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)  # 180 deg about x
    local = torch.tensor([0.0, 0.0, c.bowl_h / 2 + 0.004], device=device)
    pos = scene.drawer.data.root_pos_w[0] + quat_apply(dq.unsqueeze(0), local.unsqueeze(0))[0]
    _write_body(env, scene.bowl, pos, quat_mul(dq.unsqueeze(0), flip.unsqueeze(0))[0])
    step(60)
    report("flipped")
    check("upside-down bowl in the shut drawer: in_basin but NOT success, score <= 0.61",
          bool(scene.in_basin()[0]) and not bool(scene.bowl_upright()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.61)

    # =========================== 12. near-miss D: drawer ajar ================================
    torch.manual_seed(19)
    env.reset()
    step(30)
    _write_body(env, scene.bar, _assembly_world(scene, (0.50, -0.40, 0.015)), bar_q)
    dq = scene.drawer.data.root_quat_w[0]
    _write_body(env, scene.drawer,
                _assembly_world(scene, (c.drawer_x0 + 0.070, 0.0, c.drawer_z0)), dq)
    local = torch.tensor([0.0, 0.0, c.bowl_h / 2 + 0.004], device=device)
    pos = scene.drawer.data.root_pos_w[0] + quat_apply(dq.unsqueeze(0), local.unsqueeze(0))[0]
    _write_body(env, scene.bowl, pos, dq)
    step(60)
    report("ajar")
    check("drawer ajar (70mm) with the bowl stowed upright: NOT success (shut gate)",
          bool(scene.in_basin()[0]) and bool(scene.bowl_upright()[0])
          and not bool(scene.shut()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) < 0.999)

    # =========================== 13. near-miss E: bowl ON the counter above the basin =======
    torch.manual_seed(23)
    env.reset()
    step(30)
    _write_body(env, scene.bowl,
                _assembly_world(scene, (0.0, 0.0, c.counter_top_z + c.bowl_h / 2 + 0.003)),
                scene.carcass.data.root_quat_w[0])
    step(60)
    report("on-counter")
    check("bowl on the counter directly above the shut basin: no credit",
          not bool(scene.in_basin()[0]) and float(scene.score()[0]) < 0.05)

    # =========================== 14-15. calibration probe: bar-lift cliff ===================
    # Hold the bar kinematically at growing lift above its cradle rest pose and pull the
    # drawer: the cliff sits between the cradle-wall top (+33mm) and the face-plate top
    # (+54mm). 20mm must still bar; 80mm must clear; the 50mm midpoint is published.
    print("[smoke] CALIBRATION SWEEP (bar lift -> pulled displacement)", flush=True)
    lift_disp: dict[float, float] = {}
    for i, lift in enumerate((0.020, 0.050, 0.080)):
        torch.manual_seed(29 + i)
        env.reset()
        step(20)
        bh = c.bar_home
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = _assembly_world(scene, (bh[0], bh[1], bh[2] + lift))
        st[:, 3:7] = scene.carcass.data.root_quat_w[0]
        holds.append((scene.bar, st))
        step(5)
        _pull(scene, c.pull_force)
        step(240)
        lift_disp[lift] = float(scene.drawer_disp()[0])
        _pull(scene, 0.0)
        step(10)
        holds.clear()
        print(f"[smoke]   lift={lift * 1000:.0f}mm -> disp={lift_disp[lift] * 1000:.1f}mm",
              flush=True)
    check("probe: bar lifted 20mm still bars the drawer (< 60mm)",
          lift_disp[0.020] < 0.060)
    check("probe: bar lifted 80mm clears the drawer (>= open_min)",
          lift_disp[0.080] >= c.open_min)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="sim_gen.barred_drawer")
        print(f"[smoke] saved {arr.shape} -> {os.path.abspath(args.out)}", flush=True)
    n_pass = sum(ok for _n, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {len(checks)}/{len(checks)}", flush=True)
    else:
        for name, ok in checks:
            if not ok:
                print(f"[smoke] FAILED CHECK: {name}", flush=True)
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)

    # Hard exit: Kit teardown hangs; the watchdog guarantees the process dies.
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

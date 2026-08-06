"""Smoke / oracle battery for DrawerStashScene — NullRobot, teleport-oracle, RECORDED.

One linear run, 16 named checks:
  1. settle      — clean reset: finite state, drawer closed on its spring home, score ~0;
  2. random      — cabinet yaw + bottle/prop poses differ across seeds (by READBACK);
  3. null        — 2 s of nothing: score < 0.05, no success;
  4. spring      — calibration a: the empty drawer released from 3 pull depths always
                   glides itself CLOSED (self-closing runner is real; times published);
  5. prop        — calibration b: the stop bar at 3 lateral positions in the floor gap
                   holds the released drawer open (rest opens published);
  6. opened latch — pulling past open_min latches 0.15 and it persists after re-close;
  7. monotone    — staged oracle scores never decrease;
  8. loaded      — bottle laid flat in the propped drawer: 0.45 <= score < 1, no success;
  9. near miss   — prop STILL IN (drawer can't close): loaded but success rejected;
 10. oracle A    — staged oracle run reaches success() with score exactly 1.0;
 11/12. oracle B/C — `oracle_solution` succeeds on 2 more fresh seeds;
 13. negative A  — the SEED's plan: open the drawer, let go, then deliver the bottle to
                   where the open basin was — the drawer has sprung shut; the bottle
                   lands outside; score stays at the opened latch, no success;
 14. negative B  — anti-teleport: bottle written straight INSIDE the closed drawer —
                   containment is true but the loaded latch never set: no success,
                   score ~0;
 15. upright    — a bottle standing in the propped basin is NOT 'inside' (below-rim
                   containment, geometric); prop-removal aftermath published, not asserted;
 16. frames     — video frames recorded; saved as frames.npz in the CWD.

Bodies are driven straight through scene handles (kinematic drags/pins — the physics
verdicts come from the spring, the prop compression, and real settling). The NullRobot
applies nothing.

Run (forge): python -u -m simgen_tasks.<task>.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=10)
parser.add_argument("--max_frames", type=int, default=300)
parser.add_argument("--out", type=str, default="frames.npz")  # CWD — the pipeline fetches it
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe (proven): kit mis-decodes the L20 driver version and silently rejects RTX
# -> annotator returns EMPTY frames. Disable the check.
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
    from . import scene as scene_mod  # registers "drawer_stash" + "simgen.drawer_stash"
except ImportError:  # forge fallback: cwd on sys.path
    import scene as scene_mod

assert scene_mod.DrawerStashScene is not None  # keep the import explicit


# ----- shared drive helpers ---------------------------------------------------------------------
def _pin_drawer(env, open_d: float) -> None:
    """Kinematic pin: drawer root at `open_d` along the (negative) slide axis, zero vel."""
    scene = env.scene
    n = env.num_envs
    st = torch.zeros(n, 13, device=env.device)
    st[:, 0:3] = scene._home_w - scene.axis_w() * open_d
    st[:, 3:7] = scene._base_quat
    scene.drawer.write_root_state_to_sim(st, torch.arange(n, device=env.device))


def _place_prop(env, y_local: float, x_back_gap: float = 0.002) -> tuple:
    """Prop-bar target pose in the floor gap (cabinet-local), pressed toward the plinth."""
    c = env.scene.cfg
    x = c.front_x - c.bar_size[0] / 2 - x_back_gap
    return (x, y_local, c.bar_size[2] / 2 + 0.001)


def _write_prop(env, p_local) -> None:
    scene = env.scene
    n = env.num_envs
    st = torch.zeros(n, 13, device=env.device)
    st[:, 0:3] = scene.to_world(p_local)
    st[:, 3:7] = scene._base_quat
    scene.prop.write_root_state_to_sim(st, torch.arange(n, device=env.device))


def _lying_quat(env) -> torch.Tensor:
    """Bottle lying flat along the drawer-local y axis: q_drawer * q_x(90 deg)."""
    from isaaclab.utils.math import quat_mul

    n = env.num_envs
    c45 = math.cos(math.pi / 4)
    qx = torch.tensor([c45, c45, 0.0, 0.0], device=env.device).expand(n, 4)
    return quat_mul(env.scene.drawer.data.root_quat_w, qx)


def _write_bottle_rel_drawer(env, offset_local, quat) -> None:
    """Bottle root at drawer-frame `offset_local` from the drawer origin, zero vel."""
    from isaaclab.utils.math import quat_apply

    scene = env.scene
    n = env.num_envs
    off = torch.as_tensor(offset_local, dtype=torch.float, device=env.device).expand(n, 3)
    st = torch.zeros(n, 13, device=env.device)
    st[:, 0:3] = scene.drawer.data.root_pos_w + quat_apply(
        scene.drawer.data.root_quat_w, off)
    st[:, 3:7] = quat
    scene.bottle.write_root_state_to_sim(st, torch.arange(n, device=env.device))


def oracle_solution(scene_or_env, step_fn=None, staged_cb=None) -> bool:
    """Teleport-oracle: drag the drawer open (kinematic pin each substep), slide the stop
    bar into the floor gap from the flank, release the drawer onto the prop, lay the
    bottle flat into the exposed basin (a real 25 mm drop), slide the prop back out, and
    let the SPRING close the loaded drawer. Every verdict is physical: prop compression,
    drop capture, spring glide, settled closed containment. Returns env-0 success()."""
    env = scene_or_env if hasattr(scene_or_env, "iscene") else scene_or_env.env
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=env.device)

    def step(k: int) -> None:
        if step_fn is not None:
            step_fn(k)
        else:
            for _ in range(k):
                env.step(no_action)

    def stage(name: str) -> None:
        if staged_cb is not None:
            staged_cb(name)

    # 1. pull the drawer open (2.7 mm/substep) and hold
    for t in range(48):
        _pin_drawer(env, c.travel * (t + 1) / 48)
        step(1)
    for _ in range(10):
        _pin_drawer(env, c.travel)
        step(1)
    env.iscene.update(0.0)
    stage("opened")

    # 2. slide the stop bar in from the flank OPPOSITE the bottle, drawer still held
    d = -float(scene._bottle_side[0])
    tgt = _place_prop(env, 0.0)
    for t in range(20):
        y = d * 0.34 * (1 - (t + 1) / 20)
        _write_prop(env, (tgt[0], y, tgt[2]))
        _pin_drawer(env, c.travel)
        step(1)

    # 3. release the drawer onto the prop (the spring presses it against the bar)
    step(60)
    env.iscene.update(0.0)
    stage("propped")

    # 4. lay the bottle flat into the exposed basin: 25 mm real drop, then settle
    _write_bottle_rel_drawer(env, (-0.055, 0.0, c.body_r + 0.026), _lying_quat(env))
    step(20)
    waited = 0
    while waited < 360 and not bool((scene.inside() & scene.settled())[0]):
        step(15)
        waited += 15
    env.iscene.update(0.0)
    stage("loaded")

    # 5. extract the prop: slide it back out the way it came, then park it away
    for t in range(24):
        y = d * 0.34 * (t + 1) / 24
        _write_prop(env, (tgt[0], y, tgt[2]))
        step(1)
    _write_prop(env, (-0.32, d * 0.50, c.bar_size[2] / 2 + 0.002))

    # 6. the spring closes the loaded drawer; wait for the settled success state
    waited = 0
    while waited < 600 and not bool(scene.success()[0]):
        step(15)
        waited += 15
    stage("done")
    return bool(scene.success()[0])


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.drawer_stash")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.85, -0.70, 0.60)) + o),
                                tuple(np.array((0.05, 0.0, 0.10)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (800, 500))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames — check the RTX recipe",
                  flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action, render=annot is not None)
            if annot is not None and step_i % args.record_every == 0 \
                    and len(frames) < args.max_frames:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    def report(tag: str) -> None:
        print(f"[smoke] {tag:12s} | open={float(scene.open_dist()[0]) * 1000:6.1f}mm "
              f"inside={bool(scene.inside()[0])} opened={bool(scene._opened[0])} "
              f"loaded={bool(scene._loaded[0])} closed={bool(scene.closed()[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def settle_until(pred, max_steps: int = 360, poll: int = 15) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def drag_open(open_d: float, steps: int = 36, hold: int = 8) -> None:
        for t in range(steps):
            _pin_drawer(env, open_d * (t + 1) / steps)
            step(1)
        for _ in range(hold):
            _pin_drawer(env, open_d)
            step(1)
        env.iscene.update(0.0)

    # ========================= 1. settle / clean slate ========================================
    torch.manual_seed(3)
    env.reset()
    report("reset")
    step(60)
    report("settled")
    finite = all(bool(torch.isfinite(b.data.root_state_w).all())
                 for b in (scene.drawer, scene.bottle, scene.prop)) \
        and bool(torch.isfinite(scene.score()).all())
    check("settle: clean reset (finite, drawer closed, score ~0)",
          finite and bool(scene.closed()[0]) and float(scene.score()[0]) < 0.05)

    # ========================= 2. randomization by readback ===================================
    draws = []
    for seed in (11, 12, 13, 14, 15, 16, 17, 18):
        torch.manual_seed(seed)
        env.reset()
        step(5)
        bq = scene.bottle.data.root_quat_w[0]
        byaw = math.degrees(2 * math.atan2(float(bq[3]), float(bq[0])))
        bp = (scene.bottle.data.root_pos_w[0] - scene.env_origins[0]).tolist()
        pp = (scene.prop.data.root_pos_w[0] - scene.env_origins[0]).tolist()
        draws.append((int(scene._bottle_side[0]), round(byaw, 1),
                      round(bp[0], 3), round(bp[1], 3), round(pp[0], 3), round(pp[1], 3)))
    print(f"[smoke] draws (flank, bottle_yaw, bottle_xy, prop_xy) across seeds: {draws}",
          flush=True)
    flanks = {d[0] for d in draws}
    check("randomization is real (flank swap + item poses differ across seeds, readback)",
          len({str(d) for d in draws}) >= 4 and flanks == {1, -1})

    # ========================= 3. null policy =================================================
    torch.manual_seed(3)
    env.reset()
    step(240)
    report("null")
    check("null policy: score < 0.05 and no success",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 4. calibration a: the spring closes ============================
    torch.manual_seed(4)
    env.reset()
    step(30)
    close_times = {}
    for pull in (0.05, 0.10, 0.16):
        drag_open(pull)
        waited = 0
        while waited < 300 and not bool(scene.closed()[0]):
            step(10)
            waited += 10
        close_times[pull] = (waited, bool(scene.closed()[0]),
                             round(float(scene.open_dist()[0]) * 1000, 1))
    print(f"[smoke] SPRING CALIBRATION pull -> (steps, closed, rest_mm): {close_times}",
          flush=True)
    check("calibration: released empty drawer self-closes from every pull depth",
          all(v[1] for v in close_times.values()))

    # ========================= 5. calibration b: the prop holds ===============================
    prop_rest = {}
    for y in (-0.06, 0.0, 0.06):
        torch.manual_seed(5)
        env.reset()
        step(20)
        drag_open(c.travel)
        d5 = -float(scene._bottle_side[0])  # slide in from the flank opposite the bottle
        tgt = _place_prop(env, y)
        for t in range(16):
            f = (t + 1) / 16
            _write_prop(env, (tgt[0], d5 * 0.34 * (1 - f) + y * f, tgt[2]))
            _pin_drawer(env, c.travel)
            step(1)
        step(200)  # released: spring presses the drawer onto the bar
        prop_rest[y] = round(float(scene.open_dist()[0]) * 1000, 1)
    print(f"[smoke] PROP CALIBRATION bar_y -> rest open (mm): {prop_rest}", flush=True)
    check("calibration: stop bar props the drawer open at every lateral position",
          all(90.0 < v < 145.0 for v in prop_rest.values()))

    # ========================= 6-10. staged oracle run ========================================
    torch.manual_seed(101)
    env.reset()
    step(30)
    stage_scores: dict[str, float] = {"reset": float(scene.score()[0])}
    stage_flags: dict[str, tuple] = {}

    def staged_cb(name: str) -> None:
        stage_scores[name] = float(scene.score()[0])
        stage_flags[name] = (bool(scene.inside()[0]), bool(scene.closed()[0]),
                             bool(scene.success()[0]),
                             round(float(scene.open_dist()[0]) * 1000, 1))
        report(f"stage:{name}")

    ok_a = oracle_solution(env, step_fn=step, staged_cb=staged_cb)
    print(f"[smoke] stage scores: {stage_scores} | flags: {stage_flags}", flush=True)

    check("opened latch: pulling the drawer open latches 0.15 and persists",
          abs(stage_scores["opened"] - 0.15) < 0.01
          and stage_scores["propped"] >= 0.15 - 1e-6)
    seq = [stage_scores[k] for k in ("reset", "opened", "propped", "loaded", "done")]
    check("rubric monotonicity: staged scores never decrease",
          all(b >= a - 1e-6 for a, b in zip(seq, seq[1:])))
    check("loaded partial credit: bottle in the propped drawer -> 0.45 <= score < 1",
          0.45 - 1e-6 <= stage_scores["loaded"] < 1.0
          and not stage_flags["loaded"][2])
    check("near miss: prop still in -> drawer cannot close, success rejected",
          stage_flags["loaded"][0] and not stage_flags["loaded"][1]
          and not stage_flags["loaded"][2] and stage_scores["loaded"] < 0.9)
    check("oracle A: staged run reaches success() with score exactly 1.0",
          ok_a and abs(float(scene.score()[0]) - 1.0) < 1e-3)

    # ========================= 11+12. oracle on 2 more fresh seeds ============================
    for tag, seed in (("B", 202), ("C", 303)):
        torch.manual_seed(seed)
        env.reset()
        step(30)
        got = oracle_solution(env, step_fn=step)
        report(f"oracle-{tag}")
        check(f"oracle {tag}: fresh seed {seed} reaches success()", got)

    # ========================= 13. negative A: the seed's plan ================================
    # Open the drawer, LET GO (the seed's drawer stays open; this one doesn't), then
    # deliver the bottle to where the open basin used to be.
    torch.manual_seed(7)
    env.reset()
    step(30)
    drag_open(0.14, steps=40, hold=10)
    step(150)  # released: the drawer springs shut while the "robot" fetches the bottle
    was_closed = bool(scene.closed()[0])
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.to_world((-0.175, 0.0, 0.16))  # the former mouth of the open basin
    st[:, 3:7] = _lying_quat(env)
    scene.bottle.write_root_state_to_sim(st, all_ids)
    step(30)
    settle_until(lambda: bool(scene.settled()[0]), 300)
    report("seed-plan")
    check("negative A (seed strategy): drawer sprung shut, bottle lands outside, no success",
          was_closed and not bool(scene.inside()[0])
          and float(scene.score()[0]) <= 0.2 and not bool(scene.success()[0]))

    # ========================= 14. negative B: anti-teleport ==================================
    # Write the bottle straight INSIDE the closed drawer: containment true, but the
    # loaded latch (inside while OPEN, no same-substep teleport) never set.
    torch.manual_seed(8)
    env.reset()
    step(30)
    _write_bottle_rel_drawer(env, (0.0, 0.0, c.body_r + 0.001), _lying_quat(env))
    env.iscene.update(0.0)
    inside_now = bool(scene.inside()[0])
    step(90)
    report("teleport-in")
    check("negative B (anti-teleport): bottle written into the closed drawer never succeeds",
          inside_now and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.1)

    # ========================= 15. upright probe ==============================================
    # A bottle standing in the propped basin: geometric containment must reject it (top
    # endpoint far above the rim). The prop-removal aftermath (lintel jam, or the closing
    # drawer knocking it flat) is real physics — published, not asserted.
    torch.manual_seed(9)
    env.reset()
    step(20)
    drag_open(c.travel)
    d9 = -float(scene._bottle_side[0])
    tgt = _place_prop(env, 0.0)
    for t in range(16):
        _write_prop(env, (tgt[0], d9 * 0.34 * (1 - (t + 1) / 16), tgt[2]))
        _pin_drawer(env, c.travel)
        step(1)
    step(80)
    _write_bottle_rel_drawer(env, (-0.055, 0.0, c.bottle_len / 2 + 0.003),
                             scene._base_quat.clone())  # standing upright in the basin
    env.iscene.update(0.0)
    upright_inside = bool(scene.inside()[0])
    step(40)
    upright_loaded = bool(scene._loaded[0])
    report("upright")
    check("upright bottle in the open drawer is rejected by below-rim containment",
          not upright_inside and not upright_loaded
          and float(scene.score()[0]) <= 0.2)
    for t in range(24):  # remove the prop and watch (published only)
        _write_prop(env, (tgt[0], d9 * 0.34 * (t + 1) / 24, tgt[2]))
        step(1)
    _write_prop(env, (-0.32, d9 * 0.50, c.bar_size[2] / 2 + 0.002))
    step(400)
    print(f"[smoke]   upright aftermath (observed, not asserted): "
          f"open={float(scene.open_dist()[0]) * 1000:.1f}mm "
          f"inside={bool(scene.inside()[0])} success={bool(scene.success()[0])}", flush=True)

    # ========================= 16. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.drawer_stash")
        print(f"[smoke] saved {arr.shape} -> {os.path.abspath(args.out)}", flush=True)
    check("video frames recorded", len(frames) >= 20)

    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        bad = [nm for nm, ok in checks if not ok]
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)} — failing: {bad}", flush=True)
    # Kit teardown hangs are routine: watchdog then hard exit.
    threading.Timer(10.0, lambda: os._exit(0 if all_ok else 1)).start()
    try:
        env.close()
        app.close()
    finally:
        os._exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

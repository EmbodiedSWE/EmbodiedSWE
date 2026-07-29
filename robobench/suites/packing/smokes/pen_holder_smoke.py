"""Smoke / oracle test for PenHolderScene — NullRobot, drop staging, RECORDED.

One linear run (crate/stacking-smoke skeleton):
  1. show     — settle the reset layout (pens scattered flat, holder standing) so you can
                see it;
  2. oracle   — the source's HELD-holder fill: pick the holder up (gravity-compensated
                kinematic hold, slight tilt), drop each pen tip-up into the held mouth (a
                GENUINE drop through the opening, not a pose-set; released ~20 mm off-axis
                so it rubs down the wall and sheds speed), score climbing
                10 -> 25 -> 40 -> 90 with every transition checked, then LOWER the loaded
                holder onto the surface and release -> 100 + success() (the top-heavy
                set-down is real physics);
  3. subset   — re-reset with subset sampling ON, fill the STANDING holder with only the
                PRESENT pens, and require success() (proves the judged-on-subset mask AND
                the single-arm standing-fill path);
  4. negative A — one pen dropped tip-DOWN must never count: score pins at 40 with the
                other three in correctly;
  5. negative B — a pen lying ACROSS the mouth (on the rim) must not count as inserted;
  6. negative C — tipping the loaded holder past 45 deg must zero the score (holder-up
                gate) and kill success;
  7. sweep    — calibration: drop with increasing xy offset (+15 deg yaw error), 3 seeds
                each; publishes per-offset insert rates, the capture limit and the
                within-funnel rate (geometric funnel = inner_r - pen_r ~= 34 mm). Raw drop
                outcomes are stochastic — asserted statistically, not per-offset.
  --demo runs ONLY show + oracle (all four pens, subset sampling off) and saves the
  deliverable video.

ALWAYS records video via the viewport rgb annotator (same recipe as crate_packing_smoke:
RTX driver-version override, 3-render ghost flush, npz -> HDFS). Bodies are driven
straight through scene handles; the NullRobot applies nothing.

Run (on a GPU node with the isaaclab env):
    python -m robobench.suites.packing.smokes.pen_holder_smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--demo", action="store_true", default=False,
                    help="record ONE clean successful run only (no subset phase, no "
                         "negative controls, no sweep) — the user-facing deliverable video")
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--out", type=str, default="pen_holder_frames.npz")
parser.add_argument("--hdfs_dir", type=str, default="",
                    help="optional HDFS dir to upload the frames npz to ('' = no upload)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe (same as the render server): kit mis-decodes the L20 driver version and
# silently rejects RTX -> annotator returns EMPTY frames. Disable the check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math
import os
import shutil

import numpy as np
import torch

import robobench
from robobench.core import ENVS
from robobench.suites.packing.scenes import PenHolderSceneCfg


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    # Subset sampling OFF for the deterministic main phases; the subset phase toggles it on.
    env = ENVS.get("packing.pen_holder")().build(
        num_envs=args.num_envs, device=device,
        scene_cfg=PenHolderSceneCfg(subset_sample=False))
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    idx = {name: i for i, (name, _f, _r, _l) in enumerate(c.manifest)}
    names = [name for name, _f, _r, _l in c.manifest]
    half_l = {name: l / 2 for name, _f, _r, l in c.manifest}

    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.1, -1.1, 0.9)) + o),
                                tuple(np.array((0.0, 0.0, c.surface_z + 0.15)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames — check RTX recipe "
                  "(driver-version override, NVIDIA_DRIVER_CAPABILITIES)", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0
    # While non-None, the holder is "held": its root state is rewritten before every physics
    # step — a kinematic hold through the scene handle, no robot. The pre-step write carries
    # +g*dt of UPWARD velocity so PhysX's gravity integration cancels to exactly zero: a
    # naive zero-velocity re-pin leaves the "floor" free-falling g*dt every step (0.082 m/s
    # at 120 Hz), and the PENS riding on it inherit that velocity and fail their 0.05 m/s
    # settle gate — run 2 failed every held-phase transition on exactly this (the pens were
    # in the cup the whole time; 4/4 counted the moment the holder was released).
    hold_state: torch.Tensor | None = None
    g_dt = 9.81 * env.dt  # one-step gravity velocity, the hold compensation term

    def step(k: int, render: bool = True) -> None:
        nonlocal step_i
        for _ in range(k):
            if hold_state is not None:
                pre = hold_state.clone()
                pre[:, 9] += g_dt  # cancel the gravity kick -> truly static platform
                scene.holder.write_root_state_to_sim(pre, all_ids)
            env.step(no_action, render=render)
            if annot is not None and step_i % args.record_every == 0:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                data = annot.get_data()
                arr = np.asarray(data)
                if step_i == 0:
                    print(f"[smoke] first capture: dtype={arr.dtype} shape={arr.shape}", flush=True)
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1
        if hold_state is not None:
            # Re-pin at ZERO velocity before judging (the compensated pre-step write would
            # otherwise leave +g*dt in the holder's judged buffer).
            scene.holder.write_root_state_to_sim(hold_state, all_ids)
            env.iscene.update(0.0)

    def report(tag: str) -> None:
        pres = scene.present[0].int().tolist()
        cnt = scene.counted()[0].int().tolist()
        print(f"[smoke] {tag:10s} | present={pres} counted={cnt} "
              f"score={int(scene.score()[0])} placed={bool(scene.holder_placed()[0])} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, cond))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def diagnose(tag: str) -> None:
        """Per-pen rubric-term breakdown — printed on a failed check so the log names the
        guilty term (r_xy / depth / tilt / settled) instead of leaving it to the video."""
        b_loc, t_loc = scene._pen_ends_local()
        ins, stl = scene.inserted()[0], scene.settled()[0]
        for i2, nm in enumerate(names):
            bl, ax = b_loc[0, i2], (t_loc - b_loc)[0, i2]
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(ax[2] / ax.norm())))))
            print(f"[smoke]   {tag} {nm}: r_xy={float(bl[:2].norm()) * 1000:.0f}mm "
                  f"depth={(c.holder_h / 2 - float(bl[2])) * 1000:.0f}mm tilt={tilt:.0f}deg "
                  f"inserted={bool(ins[i2])} settled={bool(stl[i2])} "
                  f"present={bool(scene.present[0, i2])}", flush=True)

    # --- staging helpers ------------------------------------------------------------------
    def make_state(pos, quat=(1.0, 0.0, 0.0, 0.0)) -> torch.Tensor:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = env.iscene.env_origins + torch.tensor(pos, device=device)
        st[:, 3:7] = torch.tensor(quat, device=device)
        return st

    def teleport_pen(name: str, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        scene.pens[name].write_root_state_to_sim(make_state(pos, quat), all_ids)

    def holder_axis_and_mouth() -> tuple[torch.Tensor, torch.Tensor]:
        """Holder up-axis (3,) and mouth-centre world position (3,) for env 0 (local coords)."""
        q = scene.holder.data.root_quat_w[0:1]
        axis = quat_apply(q, torch.tensor([[0.0, 0.0, 1.0]], device=device))[0]
        mouth = (scene.holder.data.root_pos_w[0] - env.iscene.env_origins[0]
                 + axis * c.holder_h / 2)
        return axis, mouth

    drops = 0  # bounce-out retries — the throughput metric the brief asks for
    SLIDE_OFF = 0.020  # wall-slide release offset (m): rub the wall, shed speed on the way down

    def settle_until(pred, max_steps: int = 300, poll: int = 15) -> bool:
        """Step in `poll`-sized chunks until `pred()` is true or the budget runs out.
        Settling time is PHYSICS, not what the checks are about: a pen creeping into place
        in a crowded cup can need a few hundred extra steps (run 3 failed the last rubric
        transition on exactly this — the 4th pen lands on a pile of three and slides down
        slowly), and judging it on a fixed clock tick calls honest settling a miss."""
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def drop_pen(name: str, xy_off=None, yaw_deg: float = 0.0,
                 tip_down: bool = False, drop_h: float = 0.015, settle: int = 70,
                 retry: bool = True) -> bool:
        """Release `name` from `drop_h` above the holder mouth, axis aligned with the
        holder's (tip-up unless `tip_down`), let it fall in, settle, report counted.
        `xy_off=None` -> the WALL-SLIDE default: release ~20 mm off-axis (angle spread per
        pen) so the pen rubs down the wall and lands leaning instead of slamming the floor
        end-on (belt-and-braces on top of the scene's depenetration cap). Pass an explicit
        `xy_off` (the sweep does) to measure the raw funnel.
        A pen that is geometrically `inserted` but not yet settled is WAITED for, not
        retried — a retry would teleport a perfectly good pen back out of the cup."""
        nonlocal drops
        i = idx[name]
        base_ang = math.radians(45.0 + 90.0 * i)  # spread pens around the cup
        if xy_off is None:
            xy_off = (SLIDE_OFF * math.cos(base_ang), SLIDE_OFF * math.sin(base_ang))
        hq = scene.holder.data.root_quat_w[0].tolist()
        if tip_down:  # flip 180 deg about the holder-local x: q_h * (0,1,0,0)
            w, x, y, z = hq
            hq = [-x, w, z, -y]
        half = math.radians(yaw_deg) / 2
        # yaw error about world z: q_z(yaw) * q_h
        cw, sw = math.cos(half), math.sin(half)
        w, x, y, z = hq
        q = (cw * w - sw * z, cw * x - sw * y, cw * y + sw * x, cw * z + sw * w)
        for attempt in range(2 if retry else 1):
            axis, mouth = holder_axis_and_mouth()
            center = mouth + axis * (drop_h + half_l[name])
            teleport_pen(name, (float(center[0]) + xy_off[0],
                                float(center[1]) + xy_off[1], float(center[2])), q)
            step(settle)
            # wait out slow settling while the pen is IN; bail early once it is truly out
            if settle_until(lambda: bool(scene.counted()[0, i]) or
                            not bool(scene.inserted()[0, i])) and \
                    bool(scene.counted()[0, i]):
                return True
            if attempt == 0 and retry:
                drops += 1
                print(f"[smoke]   drop retry: {name} missed the holder", flush=True)
                # retry on the wall-slide band at a rotated angle (NOT dead-centre — that
                # is the flaky spot)
                a2 = base_ang + math.radians(150.0)
                xy_off = (SLIDE_OFF * math.cos(a2), SLIDE_OFF * math.sin(a2))
        return False

    # =========================== 1. show ====================================================
    env.reset()
    report("reset")
    step(60)
    report("show")
    # Predicates must read a clean slate — if anything scores at reset, the rubric is broken
    # and everything downstream is meaningless. Fail loudly.
    assert int(scene.score()[0]) == 0, \
        f"score={int(scene.score()[0])} at reset — rubric predicates broken"

    # =========================== 2. oracle: held-holder fill ================================
    # Pick the holder up: kinematic hold 0.22 m above the surface with a 3 deg tilt (inside
    # the 45 deg gate — the rubric must judge the HELD holder exactly like a standing one;
    # steeper tilts slide every pen bottom downhill into one cluster and the 4th pen lands
    # on the pile — run 4's one failure).
    hold_z = c.surface_z + c.holder_h / 2 + 0.22
    a = math.radians(3.0)
    hold_quat = (math.cos(a / 2), math.sin(a / 2), 0.0, 0.0)
    hold_pos = (c.holder_pos[0], c.holder_pos[1], hold_z)
    hold_state = make_state(hold_pos, hold_quat)
    step(30)
    report("held")
    check("held holder (3 deg) still passes holder_up", bool(scene.holder_up()[0]))

    # Fat pens (oil, r=12 mm) first, thin last: the final pen threads the leftover gap, so
    # give it the thinnest body (the packing-order trick; order is free under the rubric).
    fill_order = sorted(names, key=lambda nm: -c.manifest[idx[nm]][2])
    expect = [10, 25, 40, 90]
    for t, name in enumerate(fill_order):
        drop_pen(name)
        # patient judge: this drop may have jostled an earlier pen back above the settle
        # gate for a moment — wait for the rubric state, don't assert it on a clock tick
        ok = settle_until(lambda t=t: int(scene.score()[0]) == expect[t])
        report(f"pen-{name}")
        check(f"rubric transition {expect[t]} after '{name}'", ok)
        if not ok:
            diagnose(f"after-{name}")
    check("all-in but still held: NOT success (set-down pending)",
          not bool(scene.success()[0]))

    # Set-down: lower the loaded holder to the surface while easing the tilt to zero (pens
    # ride inside), then release and let the top-heavy cup prove it stands.
    place_z = c.surface_z + c.holder_h / 2 + 0.002
    K = 240
    for t in range(K):
        f = (t + 1) / K
        z = hold_z + (place_z - hold_z) * f
        ang = a * (1 - f)
        hold_state = make_state((hold_pos[0], hold_pos[1], z),
                                (math.cos(ang / 2), math.sin(ang / 2), 0.0, 0.0))
        step(1)
    hold_state = None  # release
    step(40)
    report("set-down")
    check("set-down: score 100", settle_until(lambda: int(scene.score()[0]) == 100))
    ok_success = bool(scene.success()[0])
    check("oracle solve reaches success()", ok_success)
    print(f"[smoke] RESULT: {'FILLED — SUCCESS' if ok_success else 'NOT FILLED — FAIL'} "
          f"(bounce-out retries={drops})", flush=True)

    if args.demo:  # deliverable video = the one clean run above; stop here
        if frames:
            arr = np.stack(frames, axis=0)
            np.savez_compressed(args.out, frames=arr, env="packing.pen_holder")
            print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
            if shutil.which("hdfs"):  # optional archive channel; absent on RunPod
                args.hdfs_dir and os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir} 2>/dev/null; "
                          f"hdfs dfs -put -f {args.out} "
                          f"{args.hdfs_dir}/{os.path.basename(args.out)}")
        print("PEN_HOLDER_SMOKE_DONE", flush=True)
        env.close()
        return

    # =========================== 3. subset phase ============================================
    # Prove success is judged on the SAMPLED subset — and validate the single-arm path: fill
    # the STANDING holder (never held) with only the present pens.
    scene.cfg.subset_sample = True
    torch.manual_seed(7)
    env.reset()
    step(40)
    report("subset")
    n_present = int(scene.present[0].sum())
    for name in fill_order:
        if bool(scene.present[0, idx[name]]):
            drop_pen(name)
    ok = settle_until(lambda: bool(scene.success()[0]))
    report("subset-in")
    check(f"subset success with {n_present}/4 present (standing fill)", ok)
    scene.cfg.subset_sample = False

    # =========================== 4. negative control A (tip-down) ===========================
    env.reset()
    step(40)
    drop_pen("pen_0", tip_down=True, retry=False)
    check("tip-down: pen_0 never counts", not bool(scene.counted()[0, idx["pen_0"]]))
    for name in [nm for nm in fill_order if nm != "pen_0"]:
        drop_pen(name)
    # wait for the three GOOD pens to count, THEN assert the score is pinned (waiting for
    # the pinned value itself would be circular)
    good = [idx[n] for n in names[1:]]
    settle_until(lambda: int(scene.counted()[0, good].sum()) == 3)
    report("tip-down")
    check("tip-down: score pinned at 40 with the other three in",
          int(scene.score()[0]) == 40)
    check("tip-down: success rejected", not bool(scene.success()[0]))

    # =========================== 5. negative control B (across the rim) =====================
    # A pen lying horizontally across the mouth: on-axis in xy but shallow and sideways —
    # must fail `inserted` (depth AND tip-up terms). Judged at the authored pose BEFORE
    # physics (zero velocities, so `settled` passes and the rejection pins on geometry).
    env.reset()
    step(40)
    axis, mouth = holder_axis_and_mouth()
    hq = scene.holder.data.root_quat_w[0].tolist()  # pen axis -> holder-local x: q_h * q_y(90)
    w, x, y, z = hq
    c45 = math.cos(math.pi / 4)
    q_across = (c45 * (w - y), c45 * (x - z), c45 * (y + w), c45 * (z + x))
    r0 = c.manifest[idx["pen_0"]][2]
    teleport_pen("pen_0", (float(mouth[0]), float(mouth[1]), float(mouth[2]) + r0), q_across)
    env.iscene.update(0.0)  # refresh data buffers from the written state (no physics step)
    check("across-rim: pen on the mouth is NOT inserted",
          not bool(scene.inserted()[0, idx["pen_0"]]))
    step(80)
    report("across-rim")
    print(f"[smoke]   across-rim aftermath (observed, not asserted): "
          f"counted={bool(scene.counted()[0, idx['pen_0']])}", flush=True)

    # =========================== 6. negative control C (tipped holder) ======================
    # Fill the standing holder, then knock it past the 45 deg gate: the whole score must
    # collapse to 0 (holder-up gate) and success must die.
    env.reset()
    step(40)
    for name in fill_order:
        drop_pen(name)
    check("pre-tip: standing fill reaches 100",
          settle_until(lambda: int(scene.score()[0]) == 100))
    # Tip the LOADED holder as ONE RIGID ASSEMBLY: teleporting only the cup rotates it out
    # from under the pens and leaves them impaled through its wall (the 0:14 artifact in
    # run 5's video) — an impossible interpenetrating state. Re-express every pen in the
    # holder frame, move the holder onto its side, and restore the pens at the SAME
    # holder-relative pose: nothing changed in the judging frame, so the score collapse is
    # pinned on the holder_up gate ALONE — and the pens then spill under real physics.
    b = math.radians(80.0)  # on its side, well past the gate
    hp = scene.holder.data.root_pos_w.clone()
    hq = scene.holder.data.root_quat_w.clone()
    side_state = make_state(
        (c.holder_pos[0], c.holder_pos[1], c.surface_z + c.holder_outer_r + 0.006),
        (math.cos(b / 2), math.sin(b / 2), 0.0, 0.0))
    hq_conj = hq.clone()
    hq_conj[:, 1:] = -hq_conj[:, 1:]
    pen_states = {}
    for name in names:
        p_loc = quat_apply_inverse(hq, scene.pens[name].data.root_pos_w - hp)
        q_loc = quat_mul(hq_conj, scene.pens[name].data.root_quat_w)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = side_state[:, 0:3] + quat_apply(side_state[:, 3:7], p_loc)
        st[:, 3:7] = quat_mul(side_state[:, 3:7], q_loc)
        pen_states[name] = st
    scene.holder.write_root_state_to_sim(side_state, all_ids)
    for name, st in pen_states.items():
        scene.pens[name].write_root_state_to_sim(st, all_ids)
    env.iscene.update(0.0)
    check("tipped: holder_up gate rejects", not bool(scene.holder_up()[0]))
    check("tipped: score collapses to 0", int(scene.score()[0]) == 0)
    check("tipped: success rejected", not bool(scene.success()[0]))
    step(80)
    report("tipped")

    # =========================== 7. calibration sweep =======================================
    # The honest funnel numbers: drop-release into the STANDING holder with growing xy offset
    # (+15 deg yaw error), 3 seeds each, raw (no retry). Geometric funnel = inner_r - pen_r
    # ~= 34 mm. This is a MEASUREMENT with a stochastic outcome (3-seed drop physics: runs 1
    # and 2 both landed ~70-75% inside the funnel but disagreed on WHICH offsets were 3/3),
    # so the assertions are statistical — per-offset rates are published, not asserted. The
    # per-body max-depenetration cap (see the scene spawners) was added after run 2 to tame
    # the end-on solver pop that drove the variance.
    print("[smoke] CALIBRATION SWEEP (drop offset -> insert rate, 3 seeds each)", flush=True)
    results: dict[float, int] = {}
    for off_mm in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0):
        hits = 0
        for seed in range(3):
            torch.manual_seed(seed)
            env.reset()
            step(20)
            ang = 2 * math.pi * (seed / 3.0)
            off = (off_mm / 1000.0 * math.cos(ang), off_mm / 1000.0 * math.sin(ang))
            # single attempt, no retry: the sweep measures the raw funnel, not persistence
            hit = drop_pen("pen_0", xy_off=off, yaw_deg=15.0, retry=False)
            hits += int(hit)
            print(f"[smoke]   off={off_mm:.0f}mm seed={seed}: inserted={hit}", flush=True)
        results[off_mm] = hits
    limit = max((k for k, v in results.items() if v == 3), default=0.0)
    band = [k for k, v in results.items() if v == 3]
    in_funnel = {k: v for k, v in results.items() if k <= 30.0}
    rate = sum(in_funnel.values()) / (3 * len(in_funnel))
    print("[smoke] SWEEP RESULT: " +
          " | ".join(f"{k:.0f}mm: {v}/3" for k, v in results.items()) +
          f"  -> capture limit (last 3/3) = {limit:.0f}mm, 3/3 offsets = "
          f"{[f'{b:.0f}' for b in band]}mm, within-funnel rate = {rate:.0%}", flush=True)
    # No single offset is asserted (per-offset 3-seed outcomes are stochastic — measured);
    # what must hold: raw drops mostly work inside the funnel, and some offset is fully
    # reliable. The ORACLE's own reliability is asserted where it matters — the transition
    # checks above, which drop with retries like any competent executor would.
    check("sweep: within-funnel (<=30mm) raw insert rate >= 60%", rate >= 0.60)
    check("sweep: some offset is fully reliable (3/3)", len(band) > 0)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="packing.pen_holder")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
        if shutil.which("hdfs"):  # optional archive channel; absent on RunPod
            rc = args.hdfs_dir and os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir} 2>/dev/null; "
                           f"hdfs dfs -put -f {args.out} {args.hdfs_dir}/"
                           f"{os.path.basename(args.out)}")
            print(f"[smoke] hdfs upload rc={rc} -> "
                  f"{args.hdfs_dir}/{os.path.basename(args.out)}", flush=True)
    all_ok = all(ok for _name, ok in checks)
    print(f"[smoke] RESULT: {'ALL PASS' if all_ok else 'FAIL'} "
          f"({sum(ok for _n, ok in checks)}/{len(checks)} checks, drops={drops})", flush=True)
    print("PEN_HOLDER_SMOKE_DONE", flush=True)
    env.close()


def _hard_exit_teardown() -> None:
    """Kit teardown regularly hangs inside env.close()/app.close() (100% CPU spin),
    wedging headless runs after everything is printed — the repo's standard hard-exit
    (see robobench/scripts/smoke.py): a watchdog guarantees the process ends."""
    import os as _os
    import threading as _threading

    watchdog = _threading.Timer(10.0, lambda: _os._exit(0))
    watchdog.daemon = True
    watchdog.start()
    app.close()
    _os._exit(0)


if __name__ == "__main__":
    main()
    _hard_exit_teardown()

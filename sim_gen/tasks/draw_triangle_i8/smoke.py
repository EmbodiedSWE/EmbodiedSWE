"""Smoke — REJECTION battery for BarTriangleScene's rubric (sim_gen task draw_triangle_i8).

This is NOT a solution (solve.py, the real Franka run, proves the rubric ACCEPTS correct
outcomes). Every probe here CONSTRUCTS a settled wrong outcome by teleporting bars
(instrumentation only), lets physics settle it, and asserts the rubric REJECTS it. The
battery never constructs a success() state: the most any probe reaches is 2 closed corners
plus gate-rejected geometry.

Battery:
   1. settle/no-NaN     — reset layout settles finite, bars at rest in the staging spots;
   2. reset-zero        — score ~0 on the fresh scene, nothing valid, no corners;
   3. randomization     — READBACK across seeds: mat centre, bar spots (permutation +
                          jitters) and bar yaws all move; bars clear of the mat every seed;
   4. null policy       — 240 idle steps -> score ~0, no success;
   5. state roundtrip   — set_state(get_state) restores poses (readback);
   6. near-miss gaps    — the ideal frame ON the mat but every corner gap at
                          joint_tol + 15 mm -> 0 corners, no success (gap knee is real);
   7. off-mat           — the gap-closed frame built beside the mat -> 0 corners, no
                          success, score ~0 (the seed-shape without the required site);
   8. bundle            — bars laid side by side on the mat (ends staggered): fakes <= 2
                          corners, never all three -> no success;
   9. hub fan           — three bars radiating from one hub: ALL THREE pair gaps close,
                          all bars valid, but each bar re-uses ONE end -> cycle gate (and
                          the collapsed corner area) reject it — no success;
  10. woven/propped     — frame with one bar's end resting ON a neighbour: the propped bar
                          fails the flat/ground-height gates -> its corners don't count,
                          no success;
  11-12. calibration    — two-bar corner: gap 28 mm counts, gap 55 mm does not (knee sits
                          at joint_tol); the third bar stays in staging so success never
                          fires;
  13. latch             — credit latches (placing then REMOVING a bar keeps the latched
                          partial — correct behaviour's credit doesn't evaporate) yet
                          success stays False;
  14. latch reset       — a fresh reset() clears the latch back to ~0;
  15. frames            — video frames recorded and saved to frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.draw_triangle_i8.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the L20 driver version and silently rejects RTX -> empty
# frames. Disable the driver check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.draw_triangle_i8 import scene as scene_mod
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod

frame_layout = scene_mod.frame_layout


def main() -> None:  # noqa: PLR0915
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.bar_triangle")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(1, device=device)
    origin = scene.env_origins[0]

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = origin.detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.95, -0.95, 0.75)) + o),
                                tuple(np.array((0.0, 0.0, 0.02)) + o),
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

    def sc() -> float:
        return float(scene.score()[0])

    def bar_xy(i: int):
        p = scene.bars[i].data.root_pos_w[0] - origin
        return float(p[0]), float(p[1])

    def bar_yaw(i: int) -> float:
        from isaaclab.utils.math import quat_apply

        q = scene.bars[i].data.root_quat_w[0]
        ax = quat_apply(q.unsqueeze(0), torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
        return math.atan2(float(ax[1]), float(ax[0]))

    def put(i: int, x: float, y: float, yaw: float, z: float | None = None) -> None:
        """Teleport bar i to an env-local pose with zero velocity (probe instrumentation)."""
        st = torch.zeros(1, 13, device=device)
        st[0, 0], st[0, 1] = x, y
        st[0, 2] = (c.surface_z + c.bar_w / 2 + 0.002) if z is None else z
        st[0, 3] = math.cos(yaw / 2)
        st[0, 6] = math.sin(yaw / 2)
        st[0, 0:3] += origin
        scene.bars[i].write_root_state_to_sim(st, all_ids)

    def place_frame(center, rot, corner_gap=0.024, which=(0, 1, 2)) -> tuple:
        """Teleport the chosen bars into the ideal frame layout; settle; return the layout."""
        centers, yaws, verts, gaps = frame_layout(c, center, rot, corner_gap=corner_gap)
        for i in which:
            put(i, centers[i][0], centers[i][1], yaws[i])
        step(50)
        return centers, yaws, verts, gaps

    def mat_xy():
        p = scene.pad.data.root_pos_w[0] - origin
        return float(p[0]), float(p[1])

    def report(tag: str) -> None:
        print(f"[smoke] {tag:16s} | {scene.corner_report()}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def cnt_sum() -> int:
        return int(scene.corners()[1].sum())

    # =========================== 1-2. settle / no-NaN / reset-zero ==========================
    torch.manual_seed(11)
    env.reset()
    step(60)
    report("settle")
    finite = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in scene.bars)
    speeds = [float(b.data.root_lin_vel_w[0].norm()) for b in scene.bars]
    check("settle: states finite, all bars at rest after 60 steps",
          finite and max(speeds) < 0.02)
    check("reset-zero: score ~0, no valid bars, no corners, no success on the fresh scene",
          sc() <= 0.005 and int(scene.bar_valid().sum()) == 0 and cnt_sum() == 0
          and not bool(scene.success()[0]))

    # =========================== 3. randomization is real ===================================
    reads = []
    for s in (21, 22, 23, 24):
        torch.manual_seed(s)
        env.reset()
        step(4)
        mx, my = mat_xy()
        row = [mx, my]
        for i in range(3):
            x, y = bar_xy(i)
            row += [x, y, bar_yaw(i)]
        reads.append(row)
        # every bar clear of the mat (centre outside pad half + half bar length)
        clear = all(max(abs(bar_xy(i)[0] - mx), abs(bar_xy(i)[1] - my))
                    > c.pad_size / 2 + 0.02 for i in range(3))
        reads[-1].append(1.0 if clear else 0.0)
    arr = np.array(reads)
    print(f"[smoke] randomization readback (mat_xy, 3x bar_xy+yaw, clear):\n"
          f"{np.round(arr, 3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: mat centre, every bar position and yaw move across seeds "
          "(readback); bars spawn clear of the mat every seed",
          spread[0] > 0.01 and spread[1] > 0.01
          and all(max(spread[2 + 3 * i], spread[3 + 3 * i]) > 0.03 for i in range(3))
          and all(spread[4 + 3 * i] > 0.05 for i in range(3))
          and bool((arr[:, -1] == 1.0).all()))

    # =========================== 4. null policy ==============================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    check("null policy: 240 idle steps -> score ~0, no corners, no success",
          sc() <= 0.02 and cnt_sum() == 0 and not bool(scene.success()[0]))

    # =========================== 5. state roundtrip ==========================================
    torch.manual_seed(41)
    env.reset()
    step(30)
    saved = scene.get_state(all_ids)
    p_before = [bar_xy(i) for i in range(3)]
    put(0, 0.0, 0.0, 0.5)
    put(1, 0.05, 0.05, 1.0)
    step(5)
    scene.set_state(saved, all_ids)
    step(1)
    p_after = [bar_xy(i) for i in range(3)]
    err = max(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(p_before, p_after))
    check("state roundtrip: set_state(get_state) restores bar poses (max drift < 5 mm)",
          err < 0.005)

    # =========================== 6. near-miss gaps ===========================================
    torch.manual_seed(51)
    env.reset()
    step(20)
    mx, my = mat_xy()
    _cn, _yw, _v, gaps = place_frame((mx, my), 0.35, corner_gap=c.joint_tol + 0.015)
    report("near-miss")
    check("near-miss: ideal frame ON the mat with every corner gap at joint_tol + 15 mm -> "
          "all bars valid yet 0 corners, no success, score <= 0.16",
          int(scene.bar_valid().sum()) == 3 and cnt_sum() == 0
          and not bool(scene.success()[0]) and sc() <= 0.16)

    # =========================== 7. off-mat ==================================================
    torch.manual_seed(61)
    env.reset()
    step(20)
    mx, my = mat_xy()
    place_frame((mx + 0.30, my + 0.28), -0.4, corner_gap=0.024)
    report("off-mat")
    check("off-mat: the gap-closed frame built beside the mat -> 0 corners, no success, "
          "score ~0 (site requirement is load-bearing)",
          cnt_sum() == 0 and not bool(scene.success()[0]) and sc() <= 0.02)

    # =========================== 8. bundle ===================================================
    torch.manual_seed(71)
    env.reset()
    step(20)
    mx, my = mat_xy()
    for i, (dy, dx) in enumerate(((0.0, 0.0), (0.025, 0.004), (0.050, 0.008))):
        put(i, mx + dx, my + dy - 0.025, 0.0)
    step(50)
    report("bundle")
    check("bundle: bars laid side by side on the mat fake at most 2 corners, never success",
          1 <= cnt_sum() <= 2 and not bool(scene.success()[0]))

    # =========================== 9. hub fan (cycle + area gates) =============================
    torch.manual_seed(81)
    env.reset()
    step(20)
    mx, my = mat_xy()
    hub = (mx + 0.13, my + 0.13)
    r0 = 0.026
    for i, ang in enumerate((190.0, 235.0, 280.0)):
        u = (math.cos(math.radians(ang)), math.sin(math.radians(ang)))
        d = r0 + c.bar_lengths[i] / 2
        put(i, hub[0] + d * u[0], hub[1] + d * u[1], math.radians(ang))
    step(50)
    report("hub-fan")
    gaps_now, cnt_now, _xy, use_now = scene.corners()
    check("hub fan: all three pair gaps close (bars valid) BUT each bar re-uses one end -> "
          "cycle/area gates reject, no success",
          int(cnt_now.sum()) == 3 and not bool(scene.success()[0])
          and not bool(scene._cycle_ok(use_now)[0]))

    # =========================== 10. woven / propped =========================================
    torch.manual_seed(91)
    env.reset()
    step(20)
    mx, my = mat_xy()
    centers, yaws, _v, _g = place_frame((mx, my), 0.2, which=(0, 2))
    # bar 1 crossing bar 0: centre over bar 0's centre, rotated 60 deg off it, one
    # bar-width up — it settles propped across bar 0 (a woven corner)
    put(1, centers[0][0], centers[0][1], yaws[0] + math.radians(60.0),
        z=c.surface_z + c.bar_w * 1.5 + 0.003)
    step(80)
    report("woven")
    valid = scene.bar_valid()[0]
    check("woven/propped: a bar resting across another settles tilted/raised -> it is not "
          "'flat at ground level', its corners don't count, no success",
          not bool(valid[1]) and not bool(scene.success()[0]))

    # =========================== 11-12. calibration knee =====================================
    cal = []
    for seed, g in ((101, 0.028), (102, c.joint_tol + 0.015)):
        torch.manual_seed(seed)
        env.reset()
        step(20)
        mx, my = mat_xy()
        place_frame((mx, my), 0.1, corner_gap=g, which=(0, 1))
        gaps_now, cnt_now, _xy, _use = scene.corners()
        cal.append((g, float(gaps_now[0, 0]), bool(cnt_now[0, 0]), bool(scene.success()[0])))
        report(f"cal gap={g * 1000:.0f}mm")
    print(f"[smoke] calibration (planned_gap, measured_gap, counted): "
          f"{[(round(g * 1000), round(m * 1000, 1), cn) for g, m, cn, _s in cal]}", flush=True)
    check("calibration: a 28 mm two-bar corner COUNTS (knee below joint_tol); success still "
          "False with the third bar in staging",
          cal[0][2] and not cal[0][3])
    check("calibration: a 55 mm two-bar corner does NOT count (knee above joint_tol)",
          not cal[1][2] and not cal[1][3])

    # =========================== 13-14. latch behaviour ======================================
    torch.manual_seed(111)
    env.reset()
    step(20)
    mx, my = mat_xy()
    centers, yaws, _v, _g = frame_layout(c, (mx, my), 0.25)
    put(0, centers[0][0], centers[0][1], yaws[0])
    step(40)
    s1 = sc()
    put(1, centers[1][0], centers[1][1], yaws[1])
    step(40)
    s2 = sc()
    report("latch: 2 bars")
    put(1, mx + 0.32, my - 0.30, 0.3)  # remove bar 1 again (off the mat)
    step(40)
    s3 = sc()
    report("latch: removed")
    check("latch: 1 bar ~0.05, +second bar closes a corner (>= 0.25); removing it keeps "
          "the latched credit (no evaporation) and success stays False",
          0.04 <= s1 <= 0.11 and s2 >= 0.25 and s3 >= s2 - 1e-6
          and not bool(scene.success()[0]))
    torch.manual_seed(112)
    env.reset()
    step(10)
    check("latch: a fresh reset clears the latch back to ~0", sc() <= 0.005)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.bar_triangle")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    check("frames: video frames recorded and saved to frames.npz", len(frames) > 20)

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
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()

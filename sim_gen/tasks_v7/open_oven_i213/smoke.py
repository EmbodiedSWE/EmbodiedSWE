"""Smoke battery for BatchScaleScene — REJECTION tests for the rubric, NullRobot, RECORDED.

This is NOT a solution (the solution is solve.py — transport-only teleports drop the
exact counterweight combination into the tray and the spring plant does the measuring;
the Franka pick-carry-release strategy is TASK.md's embodiment argument).
Teleported weight states here are rubric INSTRUMENTATION: construct each wrong outcome
as a settled state under the scene's live spring plant, then assert the rubric REJECTS
it. Force probes write the scene's `ext_force`/`ext_torque` buffers (their designed
single writer) and are sampled WHILE the force is on — the spring restores the shoved
state the instant the force stops (measure-at-force-off lesson).

One linear run, 18 named checks:
  1. settle      — clean reset: finite state, tray resting at the zero tick, weights
                   resting on the floor, score ~0;
  2. random      — u* draws AND floor layouts differ across seeds (by READBACK);
  3. indicator   — the green tab physically sits at the sampled target tick (readback);
  4. null        — 2 s of nothing: score < 0.05, no success;
  5. negative A  — the SEED's plan (grasp + pull through an arc): 25 N pull + 2.5 N*m
                   pry on the tray — it rides its rail to the stop, opens nothing,
                   springs back when released; no credit;
  6. negative B  — press cheat: a constant downward force parks the pointer EXACTLY in
                   the band — in_band true, but unaccounted (no resting load): no
                   success, no latched credit, and the tray springs back;
  7. negative C  — hover cheat: a weight HELD inside the tray airspace counts as load
                   (on_tray true — the trap is armed) but the deflection doesn't match:
                   unaccounted, streak 0, no credit;
  8. negative D  — under-load: u*-1 units resting and accounted — out of band, rejected;
  9. negative E  — over-load: u*+1 units resting and accounted — out of band, rejected
                   (too much fails exactly like too little);
 10. negative F  — wrong place: the correct total mass settled on the FLOOR beside the
                   scale earns nothing;
 11. fly-through — all five weights dumped from high above: the pointer SWEEPS through
                   the target band on the way down (observed) but success() never fires
                   — streak + consistency gates hold;
 12. monotone    — loading the exact combo one weight at a time: the printed score
                   sequence never decreases and climbs;
 13. partial     — first weight resting: partial credit, no success;
 14. exactness   — exact load resting: success() and score == 1.0;
 15. latch       — knock the load out of the tray: success revoked, score 0.60 (latched
                   progress does not evaporate; the success bonus does);
 16. equivalence — a DIFFERENT exact decomposition of u* (small-first) also succeeds:
                   any combination with the right total counts;
 17. finite      — final no-NaN audit over every body and the rubric;
 18. frames      — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.open_oven_i213.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=10)
parser.add_argument("--max_frames", type=int, default=280)
parser.add_argument("--out", type=str, default="frames.npz")  # CWD — the pipeline fetches it
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe (proven on this render stack): kit mis-decodes the driver version and
# silently rejects RTX -> the annotator returns EMPTY frames.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import os  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

try:
    from simgen_tasks.open_oven_i213 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

G = 9.81
# Exact decompositions (2x big = 2 units, 3x small = 1 unit) for every load 1..7.
COMBOS = {
    1: ("small_0",),
    2: ("big_0",),
    3: ("big_0", "small_0"),
    4: ("big_0", "big_1"),
    5: ("big_0", "big_1", "small_0"),
    6: ("big_0", "big_1", "small_0", "small_1"),
    7: ("big_0", "big_1", "small_0", "small_1", "small_2"),
}
# Alternative (small-first) decompositions — different multisets from COMBOS.
ALT_COMBOS = {
    2: ("small_0", "small_1"),
    3: ("small_0", "small_1", "small_2"),
    4: ("big_0", "small_0", "small_1"),
    5: ("big_0", "small_0", "small_1", "small_2"),
}
SPOTS = ((-0.042, -0.042), (0.042, -0.042), (-0.042, 0.042), (0.042, 0.042), (0.0, 0.0))


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.batch_scale")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    sizes = {name: size for name, size, _m, _u in c.manifest}
    units_of = {name: u for name, _s, _m, u in c.manifest}

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.20, 0.45, 0.70)) + o),
                                tuple(np.array((0.50, -0.12, 0.15)) + o),
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
        print(f"[smoke] {tag:14s} | defl={float(scene.deflection()[0]) * 1000:6.1f}mm "
              f"tgt={float(scene.target_deflection()[0]) * 1000:5.1f}mm "
              f"units_on={float(scene.units_on()[0]):.2f} acc={bool(scene.accounted()[0])} "
              f"streak={int(scene.acc_streak[0])} score={float(scene.score()[0]):.3f} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def settle_until(pred, max_steps: int = 900, poll: int = 15) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def plate_top() -> float:
        return (float(scene.tray.data.root_pos_w[0, 2])
                - float(scene.env_origins[0, 2]) + c.plate_size[2] / 2)

    def place_batch(names, drop: float = 0.010) -> None:
        """TRANSPORT teleport (instrumentation): hover each named weight over its own
        tray quadrant, zero velocity, all in the SAME step — they land together, so no
        settled intermediate load ever exists (no probe routes through the goal)."""
        top = plate_top()
        for k, name in enumerate(names):
            st = torch.zeros(n, 13, device=device)
            st[:, 0] = c.scale_xy[0] + SPOTS[k][0]
            st[:, 1] = c.scale_xy[1] + SPOTS[k][1]
            st[:, 2] = top + sizes[name] / 2 + drop
            st[:, 3] = 1.0
            st[:, 0:3] += scene.env_origins
            scene.weights[name].write_root_state_to_sim(st, all_ids)

    def park_floor(names, x0: float = 0.95) -> None:
        """Teleport weights to open floor far from the scale, resting poses."""
        for k, name in enumerate(names):
            st = torch.zeros(n, 13, device=device)
            st[:, 0] = x0 + 0.09 * k
            st[:, 1] = -0.45
            st[:, 2] = sizes[name] / 2 + 0.002
            st[:, 3] = 1.0
            st[:, 0:3] += scene.env_origins
            scene.weights[name].write_root_state_to_sim(st, all_ids)

    def loaded(units: int) -> bool:
        return (abs(float(scene.units_on()[0]) - units) < 0.25
                and bool(scene.accounted()[0])
                and int(scene.acc_streak[0]) >= c.latch_streak)

    def all_finite() -> bool:
        bodies = [scene.tray, scene.marker, *scene.weights.values()]
        return (all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
                and bool(torch.isfinite(scene.score()).all())
                and bool(torch.isfinite(scene.deflection()).all()))

    def reset_with_u(pred, seeds) -> int:
        """Reset until the sampled u* satisfies pred; returns u*."""
        for s in seeds:
            torch.manual_seed(s)
            env.reset()
            u = int(scene.target_units[0])
            if pred(u):
                return u
        raise AssertionError("no seed in range produced the wanted u*")

    # ========================= 1. settle / clean-slate ========================================
    torch.manual_seed(3)
    env.reset()
    report("reset")
    step(90)
    report("settled")
    wmax = max(float(b.data.root_lin_vel_w.norm(dim=-1).max()) for b in scene.weights.values())
    check("settle: clean reset (finite, tray at zero tick, weights resting, score ~0)",
          all_finite() and abs(float(scene.deflection()[0])) < 0.004
          and wmax < 0.08 and float(scene.score()[0]) < 0.05
          and not bool(scene.success()[0]))

    # ========================= 2+3. randomization + goal indicator ============================
    draws, layouts = [], []
    tab_err_max = 0.0
    for seed in (11, 12, 13, 14, 15, 16):
        torch.manual_seed(seed)
        env.reset()
        step(5)
        u = int(scene.target_units[0])
        draws.append(u)
        xy = scene.weights["small_0"].data.root_pos_w[0, :2] - scene.env_origins[0, :2]
        layouts.append((round(float(xy[0]), 2), round(float(xy[1]), 2)))
        want = torch.tensor(c.marker_pos(u), device=device)
        got = scene.marker.data.root_pos_w[0] - scene.env_origins[0]
        tab_err_max = max(tab_err_max, float((got - want).norm()))
    print(f"[smoke] u* draws: {draws} | small_0 layouts: {layouts} | "
          f"tab readback err={tab_err_max * 1000:.1f}mm", flush=True)
    check("randomization is real (u* and floor layout differ across seeds)",
          len(set(draws)) >= 2 and len(set(layouts)) >= 2)
    check("goal indicator: green tab parked at the sampled tick (readback)",
          tab_err_max < 0.003)

    # ========================= 4. null policy =================================================
    torch.manual_seed(3)
    env.reset()
    step(240)  # 2 s of nothing
    report("null")
    check("null policy: score < 0.05 and no success",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 5. negative A: the seed's plan =================================
    # open_oven's strategy — grasp and PULL through an arc. Here: 25 N upward pull +
    # 2.5 N*m pry on the tray. It rides its prismatic rail to the 4 mm upper stop, opens
    # nothing, and springs straight back when the hand lets go. Sampled DURING the pull.
    torch.manual_seed(7)
    env.reset()
    step(60)
    xy0 = (scene.tray.data.root_pos_w[0, :2] - scene.env_origins[0, :2]).clone()
    scene.ext_force[:, 2] = 25.0
    scene.ext_torque[:, 0] = 2.5
    step(240)
    z_pull = float(scene.tray.data.root_pos_w[0, 2]) - float(scene.env_origins[0, 2])
    xy_drift = float((scene.tray.data.root_pos_w[0, :2]
                      - scene.env_origins[0, :2] - xy0).norm())
    on_rail = z_pull <= c.z_nat + c.joint_up + 0.005 and xy_drift < 0.005
    report("pull-on")
    scene.ext_force[:] = 0.0
    scene.ext_torque[:] = 0.0
    step(180)
    report("pull-off")
    print(f"[smoke]   pull: rode to z={z_pull:.4f} (stop {c.z_nat + c.joint_up:.4f}), "
          f"xy_drift={xy_drift * 1000:.1f}mm, back to "
          f"defl={float(scene.deflection()[0]) * 1000:.1f}mm", flush=True)
    check("negative (seed strategy): pulling/prying the tray opens nothing — it rides "
          "its rail and springs back, no credit",
          on_rail and abs(float(scene.deflection()[0])) < 0.004
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 6. negative B: press cheat =====================================
    # A constant downward force parks the pointer EXACTLY in the band (in_band true —
    # sampled WHILE pressing) but nothing rests in the tray: unaccounted, streak 0.
    torch.manual_seed(9)
    env.reset()
    step(60)
    d_star = float(scene.target_deflection()[0])
    scene.ext_force[:, 2] = -c.k_spring * d_star
    step(300)
    in_band_pressed = bool(scene.in_band()[0])
    acc_pressed = bool(scene.accounted()[0])
    streak_pressed = int(scene.acc_streak[0])
    sc_pressed = float(scene.score()[0])
    report("press-on")
    scene.ext_force[:] = 0.0
    step(180)
    report("press-off")
    check("negative (press cheat): pointer held in the band with no resting load — "
          "unaccounted, no success, no latched credit",
          in_band_pressed and not acc_pressed and streak_pressed == 0
          and sc_pressed < 0.05 and abs(float(scene.deflection()[0])) < 0.004
          and float(scene.score()[0]) < 0.05)

    # ========================= 7. negative C: hover cheat =====================================
    # Hold small_0 statically inside the tray airspace (re-written every step with a
    # +g*dt/2 upward velocity bias so the within-step speed stays under the rest gate —
    # a genuinely HELD weight, not a falling one). on_tray sees it (the trap is armed)
    # but the deflection stays ~0: unaccounted, streak 0, nothing latches.
    torch.manual_seed(10)
    env.reset()
    step(60)
    hover = torch.zeros(n, 13, device=device)
    hover[:, 0] = c.scale_xy[0]
    hover[:, 1] = c.scale_xy[1]
    hover[:, 2] = plate_top() + 0.05
    hover[:, 3] = 1.0
    hover[:, 9] = 0.5 * G * float(env.dt)  # cancels gravity over the step: |v| < gate
    hover[:, 0:3] += scene.env_origins
    on_tray_mid = acc_mid = False
    streak_mid = 0
    for i in range(180):
        scene.weights["small_0"].write_root_state_to_sim(hover, all_ids)
        step(1)
        if i == 120:
            on_tray_mid = bool(scene.on_tray()[0, 0])
            acc_mid = bool(scene.accounted()[0])
            streak_mid = int(scene.acc_streak[0])
    park_floor(("small_0",))
    step(120)
    report("hover-done")
    check("negative (hover cheat): a held weight counts as load (on_tray TRUE) but is "
          "unaccounted — streak 0, no credit",
          on_tray_mid and not acc_mid and streak_mid == 0
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 8. negative D: under-load ======================================
    u = reset_with_u(lambda x: True, range(20, 40))
    step(60)
    place_batch(COMBOS[u - 1])
    ok = settle_until(lambda: loaded(u - 1))
    report("under-load")
    check("negative (under-load): u*-1 units resting and accounted — out of band, "
          "no success, score well below top",
          ok and not bool(scene.in_band()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) < 0.75)

    # ========================= 9. negative E: over-load =======================================
    u = reset_with_u(lambda x: True, range(40, 60))
    step(60)
    place_batch(COMBOS[u + 1])
    ok = settle_until(lambda: loaded(u + 1))
    report("over-load")
    check("negative (over-load): u*+1 units resting and accounted — too much fails "
          "exactly like too little",
          ok and not bool(scene.in_band()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) < 0.75)

    # ========================= 10. negative F: wrong place ====================================
    torch.manual_seed(61)
    env.reset()
    step(60)
    u = int(scene.target_units[0])
    park_floor(COMBOS[u])  # the correct total mass — settled on the FLOOR
    step(180)
    report("wrong-place")
    check("negative (wrong place): the correct total settled on the floor earns nothing",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 11. anti fly-through ===========================================
    # Dump ALL FIVE weights (7 units) from 15 cm up: the pointer sweeps down THROUGH the
    # target band (observed — the probe is not vacuous) but success() never fires: the
    # transit is unsettled/unaccounted and the streak gate holds.
    torch.manual_seed(62)
    env.reset()
    step(60)
    place_batch(COMBOS[7], drop=0.15)
    swept = fired = False
    for _ in range(600):
        step(1)
        swept = swept or bool(scene.in_band()[0])
        fired = fired or bool(scene.success()[0])
    report("fly-through")
    check("anti fly-through: the band sweep is observed but success() never fires",
          swept and not fired and not bool(scene.success()[0]))

    # ========================= 12-15. monotone / partial / exactness / latch ==================
    u = reset_with_u(lambda x: x >= 3, range(70, 100))  # >= 2 drops for a real climb
    step(60)
    combo = COMBOS[u]
    scores = [float(scene.score()[0])]
    expect = 0
    partial_sc = None
    for k, name in enumerate(combo):
        expect += units_of[name]
        top = plate_top()
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = c.scale_xy[0] + SPOTS[k][0]
        st[:, 1] = c.scale_xy[1] + SPOTS[k][1]
        st[:, 2] = top + sizes[name] / 2 + 0.010
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        scene.weights[name].write_root_state_to_sim(st, all_ids)
        settle_until(lambda e=expect: loaded(e))
        scores.append(float(scene.score()[0]))
        if k == 0:
            partial_sc = scores[-1]
        report(f"loaded-{expect}u")
    mono = all(b >= a - 1e-4 for a, b in zip(scores, scores[1:]))
    print(f"[smoke] climb scores (u*={u}): " + " ".join(f"{s:.3f}" for s in scores),
          flush=True)
    check("rubric monotonicity: score never decreases along the loading sequence and climbs",
          mono and scores[-1] > scores[0] + 0.5)
    check("partial credit: first weight resting -> partial score, no success",
          partial_sc is not None and 0.20 <= partial_sc <= 0.70 and partial_sc < scores[-1])
    ok = settle_until(lambda: bool(scene.success()[0]))
    report("exact-load")
    check("exactness: exact load resting -> success() and score == 1.0",
          ok and abs(float(scene.score()[0]) - 1.0) < 1e-3)
    park_floor(combo)  # knock the whole load out of the tray
    step(240)
    report("knock-off")
    check("achievement latch: knock-off revokes success, latched 0.60 remains",
          not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.60) < 0.02)

    # ========================= 16. equivalence: another exact combination =====================
    u = reset_with_u(lambda x: x in ALT_COMBOS, range(100, 140))
    step(60)
    place_batch(ALT_COMBOS[u])
    ok = settle_until(lambda: bool(scene.success()[0]))
    report("alt-combo")
    check("equivalence: a different exact decomposition of u* also succeeds",
          ok and abs(float(scene.score()[0]) - 1.0) < 1e-3)

    # ========================= 17. final finite audit =========================================
    check("final state finite (no NaN anywhere)", all_finite())

    # ========================= 18. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.batch_scale")
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

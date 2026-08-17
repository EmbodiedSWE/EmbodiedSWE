"""Smoke battery for StoveDialInterlockScene — REJECTION tests for the rubric, NullRobot,
RECORDED.

This is NOT a solution (the solution is solve.py — torque-servo each dial's notch
under its fence foot and let the fence gravity-drop; the Franka strategy is TASK.md's
embodiment argument). Drives here are the same hand-scale numbers as solve.py (shared
numbers = shared honesty); every asserted outcome is CONSTRUCTED through the live
revolute/prismatic joints + contact plant. The only teleport is the CHEAT probe
(check 9), whose asserted outcome is precisely that physics rejects it.

One linear run, 12 named checks:
  1. settle   — clean reset: finite everywhere, fence physically riding the walls
                (readback), every dial physically AT its sampled yaw (readback vs
                yaw0), score ~0, no success;
  2. random   — across 8 seeds: all dials spawn outside the exclusion band (>= 40 deg
                from aligned, readback), per-dial yaws vary, no two seeds share a
                triple;
  3. null     — 2 s of nothing: score < 0.05, no success;
  4. captive  — +5 N pull on the fence LIFTS it to the ~12 mm joint stop (probe force
                path proven live, fence cannot be lifted out) and it falls back to
                ride when released;
  5. press    — with every dial unaligned, a sustained 25 N press (>20x fence weight)
                cannot push the fence down: peak drop < 8 mm (< drop_thresh margin),
                returns to ride, no success — you cannot force the interlock;
  6. negative — the SEED's strategy (turn ONE dial): middle dial cranked into
                alignment — score ~0.2, fence still riding, NO success;
  7. negative — two dials aligned, third far: score ~0.4, fence still riding
                (ONE wall is enough support), NO success;
  8. negative — near miss: two aligned, third cranked to ~33 deg (outside the
                +-22.8 deg physical window AND the 30 deg hard window) + the 25 N
                press again: fence never drops, NO success;
  9. cheat    — fence TELEPORTED to the dropped pose with dials unaligned: the feet
                end up buried in the rim walls — the rubric refuses the fake (no
                success, no credit, hard-window conjunct) AND the illegal state
                cannot be legalized: a dial cranked toward alignment jams on the
                buried foot far outside the hard window;
 10. exact    — all three dials servo-aligned -> the fence drops ~22 mm BY GRAVITY
                ALONE -> success() and score == 1.0, still true 1 s later hands-off;
 11. pinned   — with the fence dropped, an adversarial quasi-static crank (solve's
                full hand-scale torque) cannot turn a dial out: it jams on the
                sunken foot inside the hard window, the fence stays down, success
                survives (the physical latch is real);
 12. frames   — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene9_turn_on_the_stove_i338.smoke --headless
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

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.libero_kitchen_scene9_turn_on_the_stove_i338 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie (daemon Timer).
_wd = threading.Timer(1200.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

# Same hand-scale dial servo as solve.py (shared numbers = shared honesty).
K_A = 4.0
W_CAP = 1.2
KW = 0.08
TAU_MAX = 0.30
E_DONE = math.radians(2.0)
W_DONE = 0.06

PROBE_SEED = 5
PRESS_N = -25.0  # press probe (>20x fence weight)
PULL_N = 5.0  # captivity probe (~4x fence weight)


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.stove_dial_interlock")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    ids0 = torch.tensor([0], device=device)

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.62, -0.55, 0.92)) + o),
                                tuple(np.array((0.02, 0.0, 0.46)) + o),
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
        e = scene.align_err_deg()[0]
        al = scene.aligned()[0]
        print(f"[smoke] {tag:14s} | err=({float(e[0]):6.1f}, {float(e[1]):6.1f}, "
              f"{float(e[2]):6.1f})deg aligned=({int(al[0])},{int(al[1])},{int(al[2])}) "
              f"drop={float(scene.fence_drop()[0]) * 1000:+6.1f}mm "
              f"dropped={bool(scene.dropped()[0])} settled={bool(scene.settled()[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def finite_all() -> bool:
        return all(bool(torch.isfinite(b.data.root_state_w).all())
                   for b in scene._bodies().values()) \
            and bool(torch.isfinite(scene.score()).all())

    def drop_mm() -> float:
        return float(scene.fence_drop()[0]) * 1000

    # --- drive helpers (solve.py's servo, verbatim numbers, arbitrary target) ---
    def crank(i: int, tgt: float, budget: int = 1800) -> tuple[bool, float]:
        """Velocity-cascade torque servo on dial i toward yaw `tgt`; releases only
        close AND slow. Returns (reached, max |err from 0| seen in deg)."""
        max_err0 = 0.0
        for _ in range(budget):
            yaw = float(scene.yaws()[0, i])
            max_err0 = max(max_err0, abs(math.degrees(wrap(yaw))))
            w = float(scene.rotors[i].data.root_ang_vel_w[0, 2])
            e = wrap(tgt - yaw)
            if abs(e) < E_DONE and abs(w) < W_DONE:
                scene.rotor_tau[0, i] = 0.0
                return True, max_err0
            w_des = max(-W_CAP, min(W_CAP, K_A * e))
            scene.rotor_tau[0, i] = max(-TAU_MAX, min(TAU_MAX, KW * (w_des - w)))
            step(1)
        scene.rotor_tau[0, i] = 0.0
        return False, max_err0

    def push_fence(force: float, steps: int = 240) -> tuple[float, float]:
        """Hold `force` (N, +up) on the fence; returns (peak drop, peak rise) in mm
        sampled DURING the hold (back-driven probes measure at force-on peak)."""
        peak_down, peak_up = -1e9, 1e9
        for _ in range(steps):
            scene.fence_f[0] = force
            step(1)
            d = drop_mm()
            peak_down, peak_up = max(peak_down, d), min(peak_up, d)
        scene.fence_f[0] = 0.0
        step(180)  # release + settle
        return peak_down, peak_up

    def reset(seed: int) -> None:
        env.reset(seed=seed)
        step(30)

    # ========================= 1. settle / clean-slate ========================================
    reset(PROBE_SEED)
    report("reset")
    yaw_rb = float(torch.rad2deg(
        (scene.yaws()[0] - scene.yaw0[0] + math.pi) % (2 * math.pi) - math.pi).abs().max())
    print(f"[smoke] seed={PROBE_SEED} yaw readback err={yaw_rb:.2f}deg "
          f"drop={drop_mm():+.1f}mm", flush=True)
    check("settle: clean reset (finite, fence riding, dials AT their sampled yaws, score ~0)",
          finite_all() and abs(drop_mm()) < 3.0 and yaw_rb < 2.0
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 2. randomization across seeds ==================================
    draws = []
    ok_rb, min_err = True, 1e9
    for sd in (11, 12, 13, 14, 15, 16, 17, 18):
        reset(sd)
        yy = [round(math.degrees(float(v)) % 360.0, 1) for v in scene.yaws()[0]]
        rb = float(torch.rad2deg(
            (scene.yaws()[0] - scene.yaw0[0] + math.pi) % (2 * math.pi) - math.pi).abs().max())
        ok_rb &= rb < 2.0 and abs(drop_mm()) < 3.0
        min_err = min(min_err, float(scene.align_err_deg()[0].min()))
        draws.append(tuple(yy))
    print(f"[smoke] draws (yaw_l, yaw_m, yaw_r deg): {draws} min_err={min_err:.1f}deg",
          flush=True)
    per_dial_variety = all(len({d[i] for d in draws}) >= 5 for i in range(3))
    check("randomization is real (8 seeds: all spawns >= 40 deg from aligned, per-dial "
          "variety, no repeated triple, all readback)",
          ok_rb and min_err >= 40.0 and per_dial_variety and len(set(draws)) == 8)

    # ========================= 3. null policy =================================================
    reset(PROBE_SEED)
    step(240)  # 2 s of nothing
    report("null")
    check("null policy: score < 0.05 and no success",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 4. captive fence + live force path =============================
    reset(PROBE_SEED)
    _pd, peak_up = push_fence(PULL_N)
    report("pulled")
    check("captive: +5 N pull LIFTS the fence to the ~12 mm stop (force path live, "
          "cannot be lifted out) and it returns to ride",
          -20.0 < peak_up < -6.0 and abs(drop_mm()) < 3.0
          and not bool(scene.success()[0]) and finite_all())

    # ========================= 5. press probe: the interlock cannot be forced =================
    peak_down, _pu = push_fence(PRESS_N)
    report("pressed")
    check("press: 25 N on the unaligned interlock — peak drop < 8 mm, returns to ride, "
          "no success (walls carry the load)",
          peak_down < 8.0 and abs(drop_mm()) < 3.0
          and not bool(scene.success()[0]) and finite_all())

    # ========================= 6. negative: the SEED's strategy (turn ONE dial) ===============
    reset(PROBE_SEED)
    ok1, _ = crank(1, 0.0)
    step(60)
    report("one-dial")
    check("negative (seed strategy): ONE dial aligned — score ~0.2, fence still riding, "
          "no success",
          ok1 and bool(scene.aligned()[0, 1]) and abs(drop_mm()) < 3.0
          and not bool(scene.dropped()[0]) and not bool(scene.success()[0])
          and 0.15 < float(scene.score()[0]) < 0.25)

    # ========================= 7. negative: two dials aligned =================================
    ok0, _ = crank(0, 0.0)
    step(60)
    report("two-dials")
    check("negative: TWO dials aligned, third far — score ~0.4, one wall still carries "
          "the fence, no success",
          ok0 and abs(drop_mm()) < 3.0 and not bool(scene.dropped()[0])
          and not bool(scene.success()[0])
          and 0.35 < float(scene.score()[0]) < 0.45
          and float(scene.align_err_deg()[0, 2]) > c.align_hard_deg)

    # ========================= 8. negative: near miss at ~33 deg + press ======================
    ok2, _ = crank(2, math.radians(33.0))
    step(60)
    report("near-miss")
    e2 = float(scene.align_err_deg()[0, 2])
    near_pose = ok2 and 30.5 < e2 < 36.0
    peak_down, _pu = push_fence(PRESS_N)
    report("near-press")
    check("negative (near miss): third dial ~33 deg off (outside the +-22.8 deg physical "
          "window) — fence never drops even under the 25 N press, no success",
          near_pose and peak_down < 8.0 and abs(drop_mm()) < 3.0
          and not bool(scene.success()[0])
          and 0.35 < float(scene.score()[0]) < 0.45)

    # ========================= 9. cheat: teleported-down fence is rejected ====================
    reset(PROBE_SEED)
    st = torch.zeros(1, 13, device=device)
    st[0, 0] = c.row_x + c.fence_x_off
    st[0, 2] = c.fence_ride_z - c.wall_h  # the dropped pose — but the dials are NOT aligned
    st[0, 3] = 1.0
    st[0, 0:3] += scene.env_origins[0]
    scene.fence.write_root_state_to_sim(st, ids0)
    step(360)  # 3 s: settle (the feet end up buried in the walls; the prismatic joint
    # blocks the horizontal minimal-translation depenetration, so the fence stays put)
    report("cheat")
    fake_refused = (not bool(scene.success()[0])) and float(scene.score()[0]) < 0.05 \
        and finite_all()
    # ...and the illegal state cannot be LEGALIZED: the buried foot is a pin through
    # the rim wall, so a dial cranked toward alignment jams far outside the hard window.
    ok_legal, _ = crank(1, 0.0, budget=600)
    step(120)
    report("cheat-crank")
    e1 = float(scene.align_err_deg()[0, 1])
    check("cheat: fence teleported to the dropped pose with dials unaligned — rubric "
          "refuses the fake (no success, no credit) and a dial cranked toward alignment "
          "jams on the buried foot far outside the hard window",
          fake_refused and not bool(scene.success()[0])
          and ((not ok_legal and e1 > c.align_hard_deg) or not bool(scene.dropped()[0]))
          and finite_all())

    # ========================= 10. exactness: align all three -> gravity drop ================
    reset(PROBE_SEED)
    okA = all(crank(i, 0.0)[0] for i in (0, 1, 2))
    settled = False
    for _ in range(60):
        if bool(scene.success()[0]):
            settled = True
            break
        step(10)
    report("goal-state")
    good = okA and settled and bool(scene.dropped()[0]) and drop_mm() > 15.0 \
        and abs(float(scene.score()[0]) - 1.0) < 1e-3
    step(120)  # ...and it persists hands-off (1 s)
    report("goal-persist")
    check("exactness: all three aligned -> fence gravity-drops ~22 mm -> success() and "
          "score == 1.0, stable hands-off",
          good and bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 1.0) < 1e-3)

    # ========================= 11. pinned: success survives an adversarial crank =============
    ok_esc, max_err0 = crank(1, math.radians(60.0), budget=600)
    step(120)
    report("pinned")
    check("pinned: full hand-scale crank cannot turn a dial out from under the dropped "
          "fence (jams inside the hard window; it DID move) — success survives",
          (not ok_esc) and 5.0 < max_err0 < c.align_hard_deg
          and bool(scene.dropped()[0]) and bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 1.0) < 1e-3 and finite_all())

    # ========================= 12. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.stove_dial_interlock")
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
    t = threading.Timer(10.0, lambda: os._exit(0 if all_ok else 1))
    t.daemon = True
    t.start()
    try:
        env.close()
        app.close()
    finally:
        os._exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

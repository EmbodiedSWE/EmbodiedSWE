"""Smoke battery for EggShelfFridgeScene — REJECTION tests for the rubric, NullRobot,
RECORDED.

This is NOT a solution (the solution is solve.py — gravity-seat the egg in the target
cup, then a 0.35 rad/s velocity-servo carry-close; the Franka strategy is TASK.md's
embodiment argument). Teleported states here are rubric INSTRUMENTATION: construct an
outcome as a settled state under the scene's live door plant, then assert the rubric's
verdict on it.

One linear run, 14 named checks:
  1. settle      — clean reset: finite state, egg on the counter, door at its authored
                   angle (readback), score ~0;
  2. random      — door angle / egg xy / target cup draws differ across seeds (READBACK);
  3. indicator   — the target-colored beacon physically sits on the pedestal, the other
                   parks off-view (readback);
  4. null        — 2 s of nothing: score < 0.05, no success;
  5. calibration — gentle carry-close (the solve strategy) keeps the egg seated: the
                   retention physics ACCEPTS careful motion (context for check 6);
  6. negative A  — the SEED's plan (one uncontrolled push until the door is shut): a
                   4 N*m shove slams the loaded door into its stop at > 2 rad/s and the
                   egg physically vaults the rim — ejection asserted for BOTH cups on
                   the SETTLED aftermath, end state (door closed, egg out) rejected;
  7. negative B  — near miss: egg seated, door servo-LANDED just outside
                   `closed_tol_deg` (~7-9 deg ajar, ~zero rate so it stays) — rejected;
  8. negative C  — egg seated in the WRONG cup, door closed gently — rejected, ~0 score
                   (goal assignment matters);
  9. negative D  — egg loose on the cabinet floor, door closed — rejected;
 10. negative E  — egg on the door shelf BETWEEN the cups, door closed gently — an egg
                   riding the door but not seated in the target cup is rejected;
 11. seal        — order forcing is physical: with the door closed, an egg dropped at
                   the doorway bounces off the closed door and never enters (loading
                   after closing is impossible, for the arm as for gravity);
 12. exactness   — constructed full goal state -> success() and score == 1.0;
 13. latch       — reopening the door revokes success; score falls to the latched 0.60
                   (earned credit does not evaporate; the live 0.40 does);
 14. frames      — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.close_fridge_i89.smoke --headless
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
# RTX recipe (proven on this render stack): kit mis-decodes the L20 driver version and
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
    from simgen_tasks.close_fridge_i89 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie.
threading.Timer(1200.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                 os._exit(3))).start()

# Same fingertip-scale gentle drive as solve.py (shared numbers = shared honesty).
TAU_MAX = 1.2
KV = 3.0
OMEGA_CAP = 0.35
K_APPROACH = 1.2
RELEASE_RAD = math.radians(1.5)
SLAM_TAU = 4.0  # the seed-style uncontrolled shove (~10 N at the handle — an easy arm push)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.egg_shelf_fridge")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.55, -0.70, 0.95)) + o),
                                tuple(np.array((0.35, 0.05, 0.35)) + o),
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
        th = math.degrees(float(scene.door_angle()[0]))
        loc = scene.egg_local()[0].tolist()
        print(f"[smoke] {tag:12s} | door={th:6.1f}deg "
              f"egg_local=({loc[0]:+.3f},{loc[1]:+.3f},{loc[2]:+.3f}) "
              f"in_cup={bool(scene.egg_in_target()[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def settle_until(pred, max_steps: int = 480, poll: int = 10) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    from isaaclab.utils.math import quat_apply

    def teleport_egg_local(local_xyz: tuple) -> None:
        """Transport-only: egg to a DOOR-frame point with zero velocity (drop follows)."""
        loc = torch.tensor([list(local_xyz)], device=device)
        w = scene.door.data.root_pos_w + quat_apply(scene.door.data.root_quat_w, loc)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = w
        st[:, 3] = 1.0
        scene.egg.write_root_state_to_sim(st, all_ids)

    def teleport_egg_world(pos: tuple) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.env_origins + torch.tensor(pos, device=device)
        st[:, 3] = 1.0
        scene.egg.write_root_state_to_sim(st, all_ids)

    def seat_egg(which: int) -> bool:
        """Hover 24 mm above cup `which`'s rim, release, let gravity seat it (contact)."""
        teleport_egg_local((c.cup_x, c.cup_y[which], c.egg_rest_local + 0.039))
        w = torch.full((n,), which, dtype=torch.long, device=device)
        return settle_until(
            lambda: bool((scene.egg_in_cup(w)
                          & (scene.egg.data.root_lin_vel_w.norm(dim=-1) < c.settle_egg))[0]))

    def gentle_close(stop_rad: float = RELEASE_RAD, budget: int = 2400) -> bool:
        """The solve-style velocity-servo close; returns True when theta <= stop_rad."""
        for _ in range(budget):
            theta = float(scene.door_angle()[0])
            if theta <= stop_rad:
                scene.door_drive[0] = 0.0
                return True
            wv = float(scene.door_rate()[0])
            w_des = -min(OMEGA_CAP, K_APPROACH * theta)
            scene.door_drive[0] = max(-TAU_MAX, min(TAU_MAX, KV * (w_des - wv)))
            step(1)
        scene.door_drive[0] = 0.0
        return False

    def land_at(theta_target: float, budget: int = 2400) -> bool:
        """Servo-LAND the door at `theta_target` with ~zero rate, then release. With
        viscous-only hinge friction the parked door stays put; releasing while still
        moving would coast ~omega*I/c (~9 deg from the 0.17 rad/s cap) to the stop."""
        for _ in range(budget):
            theta = float(scene.door_angle()[0])
            wv = float(scene.door_rate()[0])
            if theta - theta_target <= math.radians(0.5) and abs(wv) <= 0.02:
                scene.door_drive[0] = 0.0
                return True
            w_des = -min(OMEGA_CAP, K_APPROACH * max(theta - theta_target, 0.0))
            scene.door_drive[0] = max(-TAU_MAX, min(TAU_MAX, KV * (w_des - wv)))
            step(1)
        scene.door_drive[0] = 0.0
        return False

    # ========================= 1. settle / clean-slate ========================================
    torch.manual_seed(3)
    env.reset()
    report("reset")
    step(60)
    report("settled")
    finite = all(bool(torch.isfinite(b.data.root_state_w).all())
                 for b in (scene.door, scene.egg, *scene.beacons.values())) \
        and bool(torch.isfinite(scene.score()).all())
    egg_p = (scene.egg.data.root_pos_w[0] - scene.env_origins[0]).tolist()
    on_counter = abs(egg_p[2] - (c.counter_top_z + c.egg_r)) < 0.01 \
        and abs(egg_p[0] - c.counter_pos[0]) < c.egg_jitter + 0.02
    door_ok = abs(float(scene.door_angle()[0]) - float(scene.theta0[0])) < math.radians(2.0)
    check("settle: clean reset (finite, egg on counter, door at authored angle, score ~0)",
          finite and on_counter and door_ok and float(scene.score()[0]) < 0.05)

    # ========================= 2+3. randomization + goal indicator ============================
    draws = []
    beacon_err = 0.0
    depot_min = 1e9
    for seed in (11, 12, 13):
        torch.manual_seed(seed)
        env.reset()
        step(10)
        tc = int(scene.target_cup[0])
        egg_xy = (scene.egg.data.root_pos_w[0, :2] - scene.env_origins[0, :2]).tolist()
        draws.append((tc, round(math.degrees(float(scene.theta0[0])), 1),
                      round(egg_xy[0], 3), round(egg_xy[1], 3)))
        on_name = "blue" if tc == 0 else "orange"
        off_name = "orange" if tc == 0 else "blue"
        want = torch.tensor(c.beacon_on_pedestal, device=device)
        got = scene.beacons[on_name].data.root_pos_w[0] - scene.env_origins[0]
        beacon_err = max(beacon_err, float((got - want).norm()))
        off_p = scene.beacons[off_name].data.root_pos_w[0] - scene.env_origins[0]
        depot_min = min(depot_min, float(off_p[1]))
    print(f"[smoke] draws (cup, theta0, egg xy): {draws} | beacon readback "
          f"err={beacon_err * 1000:.1f}mm, off-beacon min y={depot_min:.2f}", flush=True)
    check("randomization is real (cup/door-angle/egg draws differ across seeds)",
          len({str(d) for d in draws}) >= 2
          and len({d[0] for d in draws} | {1 - draws[0][0]}) == 2
          and len({d[1] for d in draws}) >= 2)
    check("goal indicator: matching beacon on the pedestal, other parked off-view",
          beacon_err < 0.005 and depot_min > 1.0)

    # ========================= 4. null policy =================================================
    torch.manual_seed(3)
    env.reset()
    step(240)  # 2 s of nothing
    report("null")
    check("null policy: score < 0.05 and no success",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 5. calibration: gentle carry keeps the egg =====================
    # The retention physics must ACCEPT careful motion (else check 6 would reject noise,
    # not slamming). Solve's strategy, re-run here as the accept side of the boundary.
    torch.manual_seed(21)
    env.reset()
    step(30)
    tgt = int(scene.target_cup[0])
    ok_seat = seat_egg(tgt)
    ok_close = gentle_close()
    settle_until(lambda: bool(scene.success()[0]))
    report("gentle")
    check("calibration: gentle carry-close keeps the egg seated (accept side)",
          ok_seat and ok_close and bool(scene.egg_in_target()[0]))

    # ========================= 6. negative A: the SEED's plan (slam) ==========================
    # close_fridge's strategy — ONE uncontrolled push until the door is shut, nothing
    # riding on the manner of the motion. Here: 4 N*m of constant shove on the loaded
    # door. The door slaps its stop; the egg keeps its tangential speed and the arrest
    # impulse at the 8 mm rim corner vaults it out — the plan is physically
    # self-defeating, not rubric-vetoed. Run for BOTH cups (near-hinge has the shorter
    # lever arm, so it is the harder ejection); judge the SETTLED aftermath, not a
    # bounce apex mid-flight.
    slam_results = []
    tgt = 0
    for kind in ("other", "target"):  # end on the TARGET episode -> feeds the end-state check
        torch.manual_seed(31)
        env.reset()
        step(30)
        tgt = int(scene.target_cup[0])
        which = tgt if kind == "target" else 1 - tgt
        ok_seat = seat_egg(which)
        report(f"pre-slam-{which}")
        peak_rate = 0.0
        for _ in range(600):  # shove until the door is at (or bouncing off) its stop
            scene.door_drive[0] = -SLAM_TAU
            step(1)
            peak_rate = max(peak_rate, abs(float(scene.door_rate()[0])))
            if float(scene.door_angle()[0]) <= math.radians(0.5):
                break
        for _ in range(60):  # keep pressing briefly through the bounce, then release
            scene.door_drive[0] = -SLAM_TAU
            step(1)
        scene.door_drive[0] = 0.0
        step(360)  # settle: only a PERSISTENT seat counts as retained
        wt = torch.full((n,), which, dtype=torch.long, device=device)
        ejected = not bool(scene.egg_in_cup(wt)[0])
        slam_results.append((ok_seat, ejected, peak_rate))
        report(f"post-slam-{which}")
        print(f"[smoke]   slam cup {which} ({'near-hinge' if which == 0 else 'near-edge'}): "
              f"peak rate={peak_rate:.2f} rad/s "
              f"(cup speed ~{peak_rate * c.cup_radius(which):.2f} m/s) ejected={ejected}",
              flush=True)
    # End state on the seed's plan (the target episode): door shut, egg gone. Park the
    # ejected egg's chaos (transport-only) on the cabinet floor, close up, and judge.
    teleport_egg_world((0.60, 0.10, c.cab_floor_pos[2] + c.cab_floor_size[2] / 2
                        + c.egg_r + 0.002))
    gentle_close()
    settle_until(lambda: bool(scene.door_closed()[0] & scene.settled()[0]))
    report("slam-end")
    check("negative (seed strategy): slam ejects the egg from BOTH cups — physically enforced",
          all(s and e and p > 2.0 for s, e, p in slam_results))
    check("negative (seed strategy): door-shut-egg-out end state rejected, credit capped",
          bool(scene.door_closed()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.62)

    # ========================= 7. negative B: near miss (door ajar) ===========================
    torch.manual_seed(41)
    env.reset()
    step(30)
    tgt = int(scene.target_cup[0])
    ok_seat = seat_egg(tgt)
    ok_land = land_at(math.radians(9.0))
    step(240)
    th_ajar = math.degrees(float(scene.door_angle()[0]))
    report("ajar")
    check("negative (near miss): egg seated but door left ajar rejected",
          ok_seat and ok_land and th_ajar > c.closed_tol_deg + 1.0
          and bool(scene.egg_in_target()[0]) and not bool(scene.door_closed()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.90)

    # ========================= 8. negative C: wrong cup =======================================
    torch.manual_seed(51)
    env.reset()
    step(30)
    tgt = int(scene.target_cup[0])
    wrong = 1 - tgt
    ok_seat = seat_egg(wrong)
    gentle_close()
    settle_until(lambda: bool(scene.door_closed()[0] & scene.settled()[0]))
    wtens = torch.full((n,), wrong, dtype=torch.long, device=device)
    still_wrong = bool(scene.egg_in_cup(wtens)[0])
    report("wrong-cup")
    check("negative (wrong cup): egg in the unmarked cup, door closed — rejected, ~0",
          ok_seat and still_wrong and not bool(scene.success()[0])
          and float(scene.score()[0]) < 0.10)

    # ========================= 9. negative D: egg loose in the cabinet ========================
    torch.manual_seed(61)
    env.reset()
    step(30)
    teleport_egg_world((0.62, -0.05, c.cab_floor_pos[2] + c.cab_floor_size[2] / 2
                        + c.egg_r + 0.002))
    step(120)
    gentle_close()
    settle_until(lambda: bool(scene.door_closed()[0] & scene.settled()[0]))
    report("floor-egg")
    check("negative (wrong place): egg on the cabinet floor, door closed — rejected",
          bool(scene.door_closed()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) < 0.10)

    # ========================= 10. negative E: on the shelf, not in a cup =====================
    torch.manual_seed(71)
    env.reset()
    step(30)
    teleport_egg_local((c.cup_x, (c.cup_y[0] + c.cup_y[1]) / 2, c.egg_rest_local + 0.030))
    step(180)  # lands on the bare shelf strip between the two cup frames
    on_shelf = float(scene.egg_local()[0, 2]) < -0.08
    gentle_close()
    settle_until(lambda: bool(scene.door_closed()[0] & scene.settled()[0]), max_steps=600)
    report("shelf-loose")
    check("negative (on shelf, not in cup): rides the door unseated — rejected",
          on_shelf and not bool(scene.egg_in_target()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) < 0.10)

    # ========================= 11. seal: loading after closing is impossible ==================
    torch.manual_seed(81)
    env.reset()
    step(30)
    gentle_close()
    settle_until(lambda: bool(scene.door_closed()[0] & scene.settled()[0]))
    # Drop the egg right at the doorway, at shelf height: it must bounce off the closed
    # door — the cups are sealed inside; execution order is forced by geometry.
    teleport_egg_world((0.30, -0.095, 0.33))
    step(300)
    egg_x = float((scene.egg.data.root_pos_w[0] - scene.env_origins[0])[0])
    report("seal-drop")
    check("seal: with the door closed an egg cannot enter (order physically forced)",
          egg_x < 0.33 and not bool(scene.egg_in_target()[0])
          and not bool(scene.success()[0]))

    # ========================= 12. exactness: success == score 1.0 ============================
    torch.manual_seed(91)
    env.reset()
    step(30)
    tgt = int(scene.target_cup[0])
    ok_seat = seat_egg(tgt)
    gentle_close()
    ok = settle_until(lambda: bool(scene.success()[0]))
    report("goal-state")
    check("exactness: full goal state -> success() and score == 1.0",
          ok_seat and ok and abs(float(scene.score()[0]) - 1.0) < 1e-3)

    # ========================= 13. achievement latch ==========================================
    reopened = False
    for _ in range(1200):
        scene.door_drive[0] = min(TAU_MAX, 2.0)
        step(1)
        if float(scene.door_angle()[0]) > math.radians(35.0):
            reopened = True
            break
    scene.door_drive[0] = 0.0
    step(180)
    report("reopened")
    check("achievement latch: reopening revokes success; latched 0.60 remains",
          reopened and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.60) < 0.02)

    # ========================= 14. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.egg_shelf_fridge")
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

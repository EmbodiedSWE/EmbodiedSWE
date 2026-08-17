"""Smoke battery for GenevaVaultFeederScene — REJECTION tests for the rubric, NullRobot,
RECORDED.

This is NOT a solution (the solution is solve.py — drop each ball through the roof port,
then crank the Geneva driver one full revolution per quarter-turn index until the ball's
compartment crosses the under-deck drop hole; the Franka strategy is TASK.md's embodiment
argument). Teleported states here are rubric INSTRUMENTATION: construct an outcome as a
settled state under the scene's live plant, then assert the rubric's verdict on it.

One linear run, 12 named checks:
  1. settle    — clean reset: finite state, present balls resting staged on the ground
                 east of the rig, wheel parked at a station (readback), score ~0;
  2. random    — crank phase / ball staging xy / wheel park / ball count draws across
                 3 seeds (max-pairwise on the continuous draws, READBACK where physical);
  3. null      — 2 s of nothing: balls stay staged, score < 0.05, no success;
  4. detent    — the transmission is intermittent: a +/-15 deg crank rock inside the
                 disengaged arc (the crank VERIFIABLY swings) leaves the wheel parked
                 (< 2 deg) — the wheel only moves while the pin is engaged;
  5. load-only — a ball dropped through the port (solve's LOAD) without any cranking:
                 in_lane, score caps at the load credit, no success;
  6. sealed    — the loaded compartment is sealed: a sustained horizontal probe push
                 (the ball VERIFIABLY moves >= 15 mm) presses it against a vane but
                 cannot carry it to the mid station, the drop hole, or the vault;
  7. one-short — ONE full crank revolution indexes the wheel EXACTLY one quarter turn
                 (parked within 3 deg) and carries the ball to the mid station only:
                 mid credit, ball NOT in the vault, no success;
  8. exactness — constructed goal state (every present ball settled in the vault) ->
                 success() and score == 1.0;
  9. latch     — removing a ball from the vault (instrumentation teleport) revokes
                 success; the score falls to the latched 0.90, not 1.0;
 10. slit      — the vault is escape-proof: a vault ball pushed hard at the east pin
                 slit (VERIFIABLY pressed against the wall) stays inside the skirts
                 (the slit floor sits above the ball's top);
 11. roof      — the roof is closed everywhere but the port: a ball dropped directly
                 above the DROP HOLE lands ON the roof and never reaches lane or vault;
 12. frames    — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene6_close_the_microwave_i385.smoke --headless
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
    from simgen_tasks.libero_kitchen_scene6_close_the_microwave_i385 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie.
_wd = threading.Timer(1200.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

# Same wrist-scale crank drive as solve.py (shared numbers = shared honesty).
TQ_MAX = 2.5
KV = 1.0
KI = 1.5
OMEGA = 1.2
OMEGA_ENG = 0.35
ENG_HALF = math.radians(70.0)
K_APP = 4.0
END_TOL = 0.02
DIR = -1.0
F_PROBE = 0.3  # N, ball probe pushes (5 g's on a 60 g ball — a firm fingertip flick)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.geneva_vault_feeder")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    origin = scene.env_origins[0]

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = origin.detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.95, -0.85, 1.05)) + o),
                                tuple(np.array((0.05, 0.0, 0.28)) + o),
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

    def sc() -> float:
        return float(scene.score()[0])

    def npres() -> int:
        return int(scene.present[0].sum())

    def ball_p(i: int) -> tuple:
        p = scene._ball_pos()[0, i]
        return float(p[0]), float(p[1]), float(p[2])

    def az_deg(i: int) -> float:
        x, y, _z = ball_p(i)
        return math.degrees(math.atan2(y, x))

    def report(tag: str) -> None:
        phi = math.degrees(float(scene.driver_angle()[0]))
        wa = math.degrees(float(scene.wheel_angle()[0]))
        pb = " ".join(f"b{i}=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})"
                      for i in range(len(c.ball_names))
                      for p in [ball_p(i)] if bool(scene.present[0, i]))
        print(f"[smoke] {tag:12s} | crank={phi:7.1f}deg wheel={wa:7.1f}deg {pb} "
              f"score={sc():.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    # ----- shared probe motions (solve.py's drives) --------------------------------------------
    def load_ball(i: int) -> bool:
        """Solve's LOAD: free-space transport to above the port, then gravity."""
        nm = c.ball_names[i]
        st = torch.zeros(1, 13, device=device)
        st[0, 1] = 0.5 * (c.win_y0 + c.win_y1)
        st[0, 2] = 0.47
        st[0, 3] = 1.0
        st[0, 0:3] += origin
        scene.balls[nm].write_root_state_to_sim(st, torch.tensor([0], device=device))
        for _ in range(360):
            step(1)
            if bool(scene.in_lane()[0, i]) and bool(scene.settled()[0, i]):
                return True
        return False

    def index_once() -> tuple:
        """Solve's INDEX: one full crank revolution through the PI velocity servo.
        Returns (done, d_total, w_total)."""
        d_prev = float(scene.driver_angle()[0])
        w_prev = float(scene.wheel_angle()[0])
        d_total = w_total = 0.0
        acc = 0.0
        done = False
        for _ in range(3600):
            a = float(scene.driver_angle()[0])
            wa = float(scene.wheel_angle()[0])
            d_total += wrap(a - d_prev)
            w_total += wrap(wa - w_prev)
            d_prev, w_prev = a, wa
            rem = 2.0 * math.pi - DIR * d_total
            if rem <= END_TOL:
                done = True
                break
            cap = OMEGA_ENG if abs(wrap(a - math.pi)) < ENG_HALF else OMEGA
            w_des = DIR * min(cap, max(0.15, K_APP * rem))
            werr = w_des - float(scene.driver_rate()[0])
            acc = max(-TQ_MAX, min(TQ_MAX, acc + KI * werr * env.dt))
            scene.crank_torque[0] = max(-TQ_MAX, min(TQ_MAX, KV * werr + acc))
            step(1)
        for _ in range(240):  # brake + release
            if abs(float(scene.driver_rate()[0])) < 0.05:
                break
            scene.crank_torque[0] = max(-TQ_MAX, min(TQ_MAX,
                                                     -KV * float(scene.driver_rate()[0])))
            step(1)
        scene.crank_torque[0] = 0.0
        step(30)
        a = float(scene.driver_angle()[0])
        wa = float(scene.wheel_angle()[0])
        d_total += wrap(a - d_prev)
        w_total += wrap(wa - w_prev)
        return done, d_total, w_total

    def teleport_ball(i: int, pos: tuple) -> None:
        st = torch.zeros(1, 13, device=device)
        st[0, 0], st[0, 1], st[0, 2] = pos
        st[0, 3] = 1.0
        st[0, 0:3] += origin
        scene.balls[c.ball_names[i]].write_root_state_to_sim(
            st, torch.tensor([0], device=device))

    # ========================= 1. settle / clean-slate ========================================
    env.reset(seed=3)
    step(60)
    report("reset")
    finite = all(bool(torch.isfinite(b.data.root_state_w).all())
                 for b in (scene.wheel, scene.driver, *scene.balls.values())) \
        and bool(torch.isfinite(scene.score()).all())
    staged_ok = all(
        (lambda p: p[2] < 0.06 and p[0] > 0.45 and abs(p[1]) < 0.40)(ball_p(i))
        for i in range(len(c.ball_names)) if bool(scene.present[0, i]))
    wheel_park = math.degrees(abs(wrap(float(scene.wheel_angle()[0]))))
    wheel_park = min(wheel_park % 90.0, 90.0 - wheel_park % 90.0)
    check("clean reset: finite, balls staged on the ground, wheel parked at a station, "
          "score ~0",
          finite and staged_ok and wheel_park < c.park_jitter_deg + 1.0
          and sc() < 0.05 and not bool(scene.success()[0]))

    # ========================= 2. randomization by readback ===================================
    draws = []
    for s in (10, 11, 12):
        env.reset(seed=s)
        step(30)
        d = (math.degrees(float(scene.driver_angle()[0])),
             ball_p(0)[0], ball_p(0)[1],
             math.degrees(float(scene.wheel_angle()[0])), npres())
        draws.append(d)
        print(f"[smoke] seed {s}: crank={d[0]:+.1f}deg ball0=({d[1]:.3f},{d[2]:.3f}) "
              f"wheel={d[3]:+.2f}deg k={d[4]}", flush=True)
    d_cr = max(abs(a[0] - b[0]) for a in draws for b in draws)
    d_xy = max(math.hypot(a[1] - b[1], a[2] - b[2]) for a in draws for b in draws)
    d_wp = max(abs(a[3] - b[3]) for a in draws for b in draws)
    check("randomization: crank phase, ball staging xy and wheel park differ across "
          "3 seeds (readback)",
          d_cr > 5.0 and d_xy > 0.005 and d_wp > 0.2)

    # ========================= 3. null policy =================================================
    env.reset(seed=4)
    step(30)
    p0 = [ball_p(i) for i in range(len(c.ball_names))]
    step(240)  # 2 s of nothing
    drift = max(math.hypot(ball_p(i)[0] - p0[i][0], ball_p(i)[1] - p0[i][1])
                for i in range(len(c.ball_names)) if bool(scene.present[0, i]))
    report("null")
    check("null policy: balls stay staged, score < 0.05, no success",
          drift < 0.02 and sc() < 0.05 and not bool(scene.success()[0]))

    # ========================= 4. detent: intermittent motion =================================
    env.reset(seed=5)
    step(30)
    phi0 = float(scene.driver_angle()[0])
    w0 = float(scene.wheel_angle()[0])
    rock = -math.copysign(math.radians(15.0), phi0)  # rock AWAY from the 180 engagement
    swung = 0.0
    for tgt in (phi0 + rock, phi0, phi0 + rock, phi0):
        for _ in range(240):
            a = float(scene.driver_angle()[0])
            err = wrap(tgt - a)
            swung = max(swung, abs(wrap(a - phi0)))
            if abs(err) < math.radians(1.0) and abs(float(scene.driver_rate()[0])) < 0.1:
                break
            tq = 3.0 * err - 1.2 * float(scene.driver_rate()[0])
            scene.crank_torque[0] = max(-TQ_MAX, min(TQ_MAX, tq))
            step(1)
    scene.crank_torque[0] = 0.0
    step(30)
    w_moved = math.degrees(abs(wrap(float(scene.wheel_angle()[0]) - w0)))
    report("detent")
    check("intermittent motion: a +/-15 deg crank rock (crank verifiably swung) leaves "
          "the parked wheel untouched (< 2 deg)",
          math.degrees(swung) > 10.0 and w_moved < 2.0)

    # ========================= 5. load-only cap ===============================================
    env.reset(seed=6)
    step(30)
    k5 = npres()
    ok_load = load_ball(0)
    step(60)
    report("load-only")
    exp5 = 0.9 * 0.28 / k5
    check("load without cranking: in_lane, score caps at the load credit, no success",
          ok_load and abs(sc() - exp5) < 0.03 and not bool(scene.success()[0]))

    # ========================= 6. sealed compartment ==========================================
    # Push the loaded ball hard toward the leading vane (tangentially, toward the drop
    # hole's side). The vane arrests it inside the compartment: no mid, no vault.
    az_start = az_deg(0)
    x0, y0, _ = ball_p(0)
    w6_0 = float(scene.wheel_angle()[0])
    moved = 0.0
    # Sustained tangential push (+x at the port station). The ball slides freely along
    # its compartment, then presses on the leading vane; the vane arrests it (the ball
    # never crosses into the next compartment). The parked wheel is not rigidly locked
    # during dwell (a Geneva locks only through the crank), so a sustained press can
    # slowly back-drive it against damping — cap the probe at 8 deg of wheel yield,
    # far short of the 60 deg the ball would need to ride to a mid station.
    for _ in range(360):
        scene.ball_probe[0, 0, 0] = F_PROBE
        step(1)
        x1, y1, _ = ball_p(0)
        moved = max(moved, math.hypot(x1 - x0, y1 - y0))
        if math.degrees(abs(wrap(float(scene.wheel_angle()[0]) - w6_0))) > 8.0:
            break
    scene.ball_probe[0] = 0.0
    step(60)
    # compartment-relative azimuth: vanes flank the port compartment at 45/135 deg
    rel = az_deg(0) - math.degrees(float(scene.wheel_angle()[0]))
    report("sealed-push")
    check("sealed compartment: probed ball moved >= 15 mm but stays captive between its "
          "two vanes (relative azimuth inside 40..140) — no mid credit, no vault",
          moved >= 0.015 and bool(scene.in_lane()[0, 0]) and 35.0 < az_deg(0) < 145.0
          and 40.0 < rel < 140.0
          and float(scene.mid_latch[0, 0]) == 0.0 and not bool(scene.in_vault()[0, 0]))

    # ========================= 7. one index is one quarter-turn, and is not enough ============
    env.reset(seed=7)  # fresh episode: a centred load (not check 6's vane-pressed ball)
    step(30)
    k7 = npres()
    ok_load7 = load_ball(0)
    done, _d_tot, w_tot = index_once()
    step(60)
    report("one-short")
    park = math.degrees(abs(float(scene.wheel_angle()[0])))
    park_err = min(park % 90.0, 90.0 - park % 90.0)
    exp7 = 0.9 * (0.28 + 0.22) / k7
    check("one crank revolution = EXACTLY one quarter-turn (parked); ball only reaches "
          "the mid station: mid credit, not in vault, no success",
          ok_load7 and done and abs(math.degrees(w_tot) - 90.0) < 12.0 and park_err < 3.0
          and float(scene.mid_latch[0, 0]) == 1.0 and bool(scene.in_lane()[0, 0])
          and float(scene.vault_latch[0, 0]) == 0.0
          and not bool(scene.in_vault()[0, 0])
          and abs(sc() - exp7) < 0.03 and not bool(scene.success()[0]))

    # ========================= 8. exactness: constructed goal state ===========================
    env.reset(seed=13)
    step(30)
    k8 = npres()
    spots = ((-0.15, -0.15, 0.045), (-0.15, 0.15, 0.045))
    for i in range(k8):
        teleport_ball(i, spots[i])
    step(120)  # settle on the vault floor
    report("goal-state")
    ok8 = bool(scene.success()[0])
    check("exactness: every present ball settled in the vault -> success() and "
          "score == 1.0",
          ok8 and abs(sc() - 1.0) < 1e-3)

    # ========================= 9. achievement latch ===========================================
    teleport_ball(0, (0.60, 0.30, 0.03))  # instrumentation: yank one ball back out
    step(120)
    report("revoked")
    check("removing a vault ball revokes success; score falls to the latched 0.90",
          not bool(scene.success()[0]) and abs(sc() - 0.90) < 0.02)

    # ========================= 10. the vault is escape-proof (pin slit) =======================
    env.reset(seed=14)
    step(30)
    teleport_ball(0, (0.20, 0.0, 0.045))  # in the vault, east of centre
    step(60)
    x_max = 0.0
    for _ in range(360):  # 3 s shove straight at the pin slit
        scene.ball_probe[0, 0, 0] = 2.0 * F_PROBE
        step(1)
        x_max = max(x_max, ball_p(0)[0])
    scene.ball_probe[0] = 0.0
    step(60)
    xf, yf, zf = ball_p(0)
    report("slit-push")
    check("vault is escape-proof: ball pressed against the east skirt (verifiable "
          "contact) stays inside — the pin slit floor is above the ball",
          x_max > c.ext_half - c.ball_radius - 0.012 and bool(scene.in_vault()[0, 0])
          and abs(xf) < c.ext_half and abs(yf) < c.ext_half)

    # ========================= 11. roof closed above the drop hole ============================
    env.reset(seed=15)
    step(30)
    teleport_ball(0, (0.0, -0.28, 0.47))  # directly above the DROP HOLE — but on the roof
    step(240)
    xf, yf, zf = ball_p(0)
    report("roof-drop")
    check("no shortcut: a ball dropped above the drop hole lands ON the roof — never "
          "in the lane or the vault",
          not bool(scene.in_lane()[0, 0]) and not bool(scene.in_vault()[0, 0])
          and (zf > c.roof_hi - 0.01 or math.hypot(xf, yf) > 0.37)
          and sc() < 0.05 and not bool(scene.success()[0]))

    # ========================= 12. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.geneva_vault_feeder")
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
    try:
        main()
    except Exception as e:  # noqa: BLE001 — never leave a GPU zombie
        print(f"[smoke] CRASH: {type(e).__name__}: {e}", flush=True)
        import traceback

        traceback.print_exc()
        os._exit(1)

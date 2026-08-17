"""Smoke battery for SwingTopBottleScene — REJECTION tests for the rubric, NullRobot,
RECORDED.

This is NOT a solution (the solution is solve.py — torque-snap the bail through its
over-center apex, pull the freed stopper, stand it on the coaster; the Franka strategy
is TASK.md's embodiment argument). Teleported states here are rubric INSTRUMENTATION:
construct an outcome as a settled state under the scene's live wrench plant, then
assert the rubric's verdict on it.

One linear run, 14 named checks:
  1. settle      — clean reset: finite state, both bails at the closed limit, both
                   stoppers seated, both bottles upright, score ~0;
  2. random      — slot swap / bottle xy / yaw / coaster xy draws differ across seeds
                   (READBACK), and BOTH slot assignments occur;
  3. null        — 2 s of nothing: score < 0.05, no success;
  4. interlock A — the SEED's plan (grip the cap, pull it straight off) run with the
                   bail CLOSED: the solve's own pull, then 3x it, on the target
                   stopper. The cap rises the few mm to the crossbar and JAMS (the
                   over-center geometry converts the pull into closing torque) — the
                   probe asserts the actuator MOVED the stopper (no vacuous pass), that
                   the plug never came near clear, and that it re-seats on release;
  5. interlock B — after those pull attempts: zero credit (no latches, score < 0.05) —
                   the seed strategy earns nothing against this mechanism;
  6. accept A    — calibration, solve strategy: a finger-scale hinge torque, cut past
                   the apex, snaps the bail to REST past the open threshold (accept
                   side of the boundary that check 4 rejects);
  7. accept B    — the SAME pull that jammed in check 4 clears the bore once the bail
                   is open — the gate is the bail state, not the force budget;
  8. negative A  — near miss: freed stopper stood on the FLOOR beside the coaster —
                   rejected, score is the latched 0.60;
  9. exactness   — stopper stood on the coaster: success() and score == 1.0;
 10. revocation  — driving the target bail back shut revokes success; the latched
                   0.70 cap remains (earned credit survives, the live 0.30 does not);
 11. negative B  — success restored (bail re-opened), then the DECOY's stopper set
                   out on the floor: decoy unsealed -> rejected, capped at 0.70;
 12. negative C  — decoy re-sealed (success again), then the decoy's bail driven
                   open: decoy bail past its closed limit -> rejected;
 13. negative D  — decoy bail re-closed (success again), then the whole decoy
                   linkage laid on its side: bottle not upright -> rejected;
 14. frames     — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.open_wine_bottle_i99.smoke --headless
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
    from simgen_tasks.open_wine_bottle_i99 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie.
threading.Timer(1200.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                 os._exit(3))).start()

# Same finger-scale drive numbers as solve.py (shared numbers = shared honesty).
TAU = 0.009        # hinge torque that snaps the bail (N*m), ~2x static gravity torque
PULL = 0.55        # vertical stopper pull (N), ~1.9x stopper weight
CLOSE_TAU = 0.012  # re-closing torque (> max gravity torque 0.0057 N*m)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.swing_top_bottle")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.45, -0.85, 0.80)) + o),
                                tuple(np.array((0.35, 0.00, 0.18)) + o),
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
        ang = scene.bail_angle_deg()[0]
        loc = scene.stopper_local()[0]
        print(f"[smoke] {tag:12s} | bail(t/d)=({float(ang[0]):+7.1f},{float(ang[1]):+7.1f})deg "
              f"stop_z(t/d)=({float(loc[0, 2]):.3f},{float(loc[1, 2]):.3f}) "
              f"on_coaster={bool(scene.on_coaster()[0])} "
              f"up={bool(scene.bottle_up()[0].all())} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f} "
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

    # --- drive helpers: the solve's own strategy, reused as instrumentation ---
    def open_bail(col: int) -> bool:
        tau = TAU
        for _attempt in range(3):
            scene.bail_drive[:, col] = tau
            for _ in range(300):
                step(1)
                if float(scene.bail_angle_deg()[0, col]) > 65.0:
                    break
            scene.bail_drive[:, col] = 0.0
            ok = settle_until(
                lambda: float(scene.bail_angle_deg()[0, col]) > c.open_min_deg
                and float(scene.bails[col].data.root_ang_vel_w.norm(dim=-1)[0]) < 0.3)
            if ok:
                return True
            tau *= 1.6
        return False

    def close_bail(col: int) -> bool:
        scene.bail_drive[:, col] = -CLOSE_TAU
        for _ in range(400):
            step(1)
            if float(scene.bail_angle_deg()[0, col]) < 25.0:
                break
        scene.bail_drive[:, col] = 0.0
        return settle_until(
            lambda: float(scene.bail_angle_deg()[0, col]) < 5.0
            and float(scene.bails[col].data.root_ang_vel_w.norm(dim=-1)[0]) < 0.5)

    def pull_out(col: int) -> bool:
        pull = PULL
        for _attempt in range(3):
            scene.stopper_pull[:, col, :] = 0.0
            scene.stopper_pull[:, col, 2] = pull
            freed = False
            for _ in range(360):
                step(1)
                if bool(scene.plug_clear()[0, col]):
                    freed = True
                    break
            scene.stopper_pull[:, col, :] = 0.0
            if freed:
                return True
            pull *= 1.5
            step(120)
        return False

    def drop_stopper_at(col: int, xy_w: torch.Tensor, z_top: float) -> None:
        """Transport-only: hover the stopper 20 mm above (xy, z_top), upright, zero
        velocity; the landing and rest are gravity + contact."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = xy_w
        st[:, 2] = z_top + 0.020
        st[:, 3] = 1.0
        scene.stoppers[col].write_root_state_to_sim(st, all_ids)
        settle_until(
            lambda: float(scene.stoppers[col].data.root_lin_vel_w.norm(dim=-1)[0])
            < c.settle_speed, max_steps=300)
        step(60)

    def seat_stopper(col: int) -> bool:
        """Transport-only: write the seated pose (bottle frame) and let it settle."""
        qb = scene.bottles[col].data.root_quat_w
        off = torch.tensor([0.0, 0.0, c.seat_z], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.bottles[col].data.root_pos_w + quat_apply(qb, off)
        st[:, 3:7] = qb
        scene.stoppers[col].write_root_state_to_sim(st, all_ids)
        step(120)
        return bool(scene.seated()[0, col])

    # ========================= 1. settle / clean-slate ========================================
    env.reset(seed=3)
    step(120)
    report("settled")
    ang = scene.bail_angle_deg()[0]
    check("settle: clean reset (finite, bails closed, stoppers seated, upright, score ~0)",
          bool(scene._finite()[0]) and float(ang.abs().max()) < 6.0
          and bool(scene.seated()[0].all()) and bool(scene.bottle_up()[0].all())
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 2. randomization readback ======================================
    draws = []
    for seed in (0, 1, 2, 3, 4, 5):
        env.reset(seed=seed)
        step(10)
        o = scene.env_origins[0]
        t_xy = (scene.bottles[0].data.root_pos_w[0, :2] - o[:2]).tolist()
        co_xy = (scene.coaster.data.root_pos_w[0, :2] - o[:2]).tolist()
        q = scene._qref_bail[0, 0]
        yaw = round(math.degrees(2.0 * math.atan2(float(q[3]), float(q[0]))), 1)
        draws.append((bool(scene.swap[0]), round(t_xy[0], 3), round(t_xy[1], 3), yaw,
                      round(co_xy[0], 3), round(co_xy[1], 3)))
    print(f"[smoke] draws (swap, target xy, yaw, coaster xy): {draws}", flush=True)
    check("randomization is real (swap/xy/yaw/coaster draws differ; both slots seen)",
          len({str(d) for d in draws}) >= 5
          and len({d[0] for d in draws}) == 2
          and len({d[3] for d in draws}) >= 5)

    # ========================= 3. null policy =================================================
    env.reset(seed=3)
    step(240)  # 2 s of nothing
    report("null")
    check("null policy: score < 0.05 and no success",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 4+5. interlock: the SEED's plan jams ===========================
    # rlbench/open_wine_bottle's plan — grip the cap and pull it straight off. Here the
    # bail is CLOSED, so the rising cap meets the crossbar a few mm up and the
    # over-center geometry converts the pull into CLOSING torque: pulling harder only
    # jams harder. Physical rejection, with a vacuity guard: the probe must show the
    # stopper actually rose to the jam before we credit the rejection.
    env.reset(seed=7)
    step(120)
    seat0 = float(scene.stopper_local()[0, 0, 2])
    max_rise, ever_clear = 0.0, False
    for pull in (PULL, 3.0 * PULL):
        scene.stopper_pull[:, 0, :] = 0.0
        scene.stopper_pull[:, 0, 2] = pull
        for _ in range(360):
            step(1)
            max_rise = max(max_rise, float(scene.stopper_local()[0, 0, 2]) - seat0)
            ever_clear = ever_clear or bool(scene.plug_clear()[0, 0])
    scene.stopper_pull[:, 0, :] = 0.0
    step(240)
    report("jam")
    print(f"[smoke]   jam probe: max rise {max_rise * 1000:.1f} mm "
          f"(clear needs {(c.rim_top_z + c.plug_clear_dz - c.seat_z) * 1000:.0f} mm), "
          f"ever_clear={ever_clear}", flush=True)
    check("interlock (seed strategy): closed-bail pull rises to the crossbar and JAMS",
          0.002 <= max_rise <= 0.016 and not ever_clear and bool(scene.seated()[0, 0])
          and float(scene.bail_angle_deg()[0, 0]) < c.closed_max_deg)
    check("interlock (seed strategy): the pull attempts earn zero credit",
          not bool(scene._moved[0]) and not bool(scene._out[0])
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 6. accept A: the solve's snap-open =============================
    # The boundary has an accept side: the finger-scale hinge torque, cut past the
    # apex, leaves the bail AT REST past the open threshold (else check 4 would be
    # rejecting force in general, not the wrong strategy).
    env.reset(seed=21)
    step(120)
    ok_open = open_bail(0)
    report("snap-open")
    check("accept: finger-scale hinge torque snaps the bail to rest past the open limit",
          ok_open and bool(scene._open[0])
          and float(scene.bail_angle_deg()[0, 0]) > c.open_min_deg)

    # ========================= 7. accept B: gated extraction ==================================
    # The SAME pull that jammed in check 4 clears the bore now that the bail is open:
    # the gate is the mechanism's state, not the force budget.
    ok_pull = pull_out(0)
    report("extract")
    check("accept: the identical pull clears the bore once the bail is open",
          ok_pull and bool(scene._out[0]))

    # ========================= 8. negative A: beside the coaster ==============================
    cp = scene.coaster.data.root_pos_w
    beside = cp[:, 0:2].clone()
    beside[:, 0] += 0.10  # > coaster_xy_tol away: on the floor next to it
    drop_stopper_at(0, beside, 0.0)
    report("floor-drop")
    check("negative (near miss): stopper stood on the floor beside the coaster — "
          "rejected, latched 0.60",
          not bool(scene.on_coaster()[0]) and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.60) < 0.005)

    # ========================= 9. exactness: success == score 1.0 =============================
    placed = False
    for _attempt in range(3):
        cp = scene.coaster.data.root_pos_w
        drop_stopper_at(0, cp[:, 0:2], float(cp[0, 2]) + c.coaster_h / 2)
        if bool(scene.on_coaster()[0]) and bool(scene.success()[0]):
            placed = True
            break
    report("goal-state")
    check("exactness: full goal state -> success() and score == 1.0",
          placed and float(scene.score()[0]) >= 0.999)

    # ========================= 10. revocation + latched cap ===================================
    ok_close = close_bail(0)
    step(120)
    report("re-closed")
    check("revocation: closing the target bail revokes success; latched 0.70 cap remains",
          ok_close and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.70) < 0.005)

    # ========================= 11. negative B: decoy unsealed =================================
    ok_reopen = open_bail(0)
    step(60)
    back = bool(scene.success()[0])  # success is live: re-opening restores it
    o = scene.env_origins[0]
    away = torch.zeros(n, 2, device=device)
    away[:, 0] = o[0] + 0.05
    away[:, 1] = scene.bottles[1].data.root_pos_w[:, 1][0]
    drop_stopper_at(1, away, 0.0)  # decoy stopper stands on the floor
    report("decoy-out")
    check("negative (decoy unsealed): decoy stopper out of its bottle — rejected, <= 0.70",
          ok_reopen and back and not bool(scene.seated()[0, 1])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.705)

    # ========================= 12. negative C: decoy bail opened ==============================
    ok_seat = seat_stopper(1)
    step(60)
    back = bool(scene.success()[0])
    ok_dopen = open_bail(1)
    step(60)
    report("decoy-open")
    check("negative (decoy opened): decoy bail past its closed limit — rejected",
          ok_seat and back and ok_dopen
          and float(scene.bail_angle_deg()[0, 1]) > c.closed_max_deg
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.705)

    # ========================= 13. negative D: tipped bottle ==================================
    ok_dclose = close_bail(1)
    step(60)
    back = bool(scene.success()[0])
    # lay the WHOLE decoy linkage on its side (one consistent write per body: teleport
    # one body of a linkage and PhysX depenetrates it right back)
    qb = scene.bottles[1].data.root_quat_w
    half = math.radians(-90.0) / 2
    qt = torch.zeros(n, 4, device=device)
    qt[:, 0], qt[:, 2] = math.cos(half), math.sin(half)
    q = scene_mod._qmul(qt, qb)  # tilt about world y: bottle axis -> world -x
    base = scene.bottles[1].data.root_pos_w.clone()
    base[:, 2] = c.body_r + 0.004
    for body, h in ((scene.bottles[1], 0.0), (scene.bails[1], c.hinge_z),
                    (scene.stoppers[1], c.seat_z)):
        off = torch.tensor([0.0, 0.0, h], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = base + quat_apply(q, off)
        st[:, 3:7] = q
        body.write_root_state_to_sim(st, all_ids)
    step(360)
    report("tipped")
    check("negative (tipped bottle): decoy laid on its side — not upright, rejected",
          ok_dclose and back and not bool(scene.bottle_up()[0, 1])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.705)

    # ========================= 14. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.swing_top_bottle")
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

"""Smoke battery for WeaveCloseCrateScene — REJECTION tests for the rubric, NullRobot,
RECORDED.

This is NOT the solution (solve.py is — it certifies the rubric ACCEPTS the correct
outcome). Teleported states here are rubric INSTRUMENTATION: construct a wrong outcome
as a settled state (or drive it through the SAME plant buffers the solve uses), then
assert the rubric REJECTS it.

One linear run, 14 named checks:
  1. settle       — clean reset: finite state, crate upright at rest, both flaps at
                    their open rests, both items standing on the floor, score ~0;
  2. random       — crate xy + yaw, item radius and item side assignment differ
                    across seeds (READBACK, not cfg);
  3. reach        — every seed: crate + both items within the documented Franka
                    envelope, and both flaps settled at the open rest;
  4. null         — 2.5 s of nothing: score ~0, flaps stay open, no crate creep;
  5. negative     — sub-lift torque: 61% of the tuck flap's lift-off moment held
                    1.5 s through the same plant buffer -> the flap never leaves the
                    open rest (flipping a flap is a real physical threshold);
  6. near-miss A  — can settled on the floor AGAINST the crate's outside wall ->
                    no packing credit;
  7. near-miss B  — candle stacked ON TOP of the in-crate can, poking above the rim
                    (settled) -> rejected (containment judged below the aperture);
  8. negative     — the SEED's full plan (pack both items, lid untouched): both items
                    settled inside, flaps still open -> success False, score exactly
                    the packing credit 0.40 — packing alone is NOT the task;
  9. negative     — WRONG ORDER: main flap closed first (same servo/buffer; it parks
                    in the closed band on its dip stop — and that alone is not
                    success); then the tuck flap driven shut with the FULL working
                    torque, ending with a hard 2 s press -> the tuck moved but rides
                    the main plate, cocked outside the closed band, weave INVERTED
                    (delta < 0), no lid credit — the weave order is geometrically
                    enforced;
 10. oracle       — fresh episode, the full correct sequence through the SAME
                    buffers: drop can, drop candle, tuck first, main on top ->
                    success() and score == 1.0 exactly (ladder 0 -> 0.2 -> 0.4 ->
                    0.6 -> 1.0);
 11. monotone     — the score never decreases along the oracle run;
 12. persist      — success holds 2 further seconds with no flicker;
 13. latch        — the main flap driven back open: success revoked, score falls to
                    exactly 0.80 (latched credit does not evaporate);
 14. frames       — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.box_task_replay_i318.smoke --headless
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

try:
    from .scene import WeaveCloseCrateScene  # registers "weave_close_crate" + env
except ImportError:  # direct-file fallback
    from scene import WeaveCloseCrateScene

assert WeaveCloseCrateScene is not None


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.weave_close_crate")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    one = torch.arange(1, device=device)

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.80, -0.80, 0.65)) + o),
                                tuple(np.array((0.0, 0.0, 0.07)) + o),
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
            if annot is not None and step_i % args.record_every == 0 and len(frames) < args.max_frames:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    def report(tag: str) -> None:
        p = scene.crate_pos()[0].tolist()
        loc = scene.items_local()[0]
        print(f"[smoke] {tag:12s} | crate=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
              f"tuck={math.degrees(float(scene.tuck_open()[0])):+7.1f} "
              f"main={math.degrees(float(scene.main_open()[0])):+7.1f} "
              f"weave={float(scene.weave_delta()[0]):+.4f} "
              f"can=({float(loc[0, 0]):+.3f},{float(loc[0, 1]):+.3f},{float(loc[0, 2]):+.3f}) "
              f"cnd=({float(loc[1, 0]):+.3f},{float(loc[1, 1]):+.3f},{float(loc[1, 2]):+.3f}) "
              f"Pc={bool(scene.packed_can[0])} Pd={bool(scene.packed_candle[0])} "
              f"T={bool(scene.tuck_set[0])} L={bool(scene.lid_set[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def item_world(body):
        return body.data.root_pos_w[0] - scene.env_origins[0]

    def teleport_local(body, dx: float, dy: float, z_world: float) -> None:
        """Place a body at a crate-frame xy offset, world z, crate-yaw aligned."""
        from isaaclab.utils.math import quat_apply

        q = scene.crate.data.root_quat_w[0:1]
        off = quat_apply(q, torch.tensor([[dx, dy, 0.0]], device=device))[0]
        cp = scene.crate.data.root_pos_w[0]
        st = torch.zeros(1, 13, device=device)
        st[0, 0] = cp[0] + off[0]
        st[0, 1] = cp[1] + off[1]
        st[0, 2] = scene.env_origins[0, 2] + z_world
        st[0, 3:7] = q[0]
        body.write_root_state_to_sim(st, one)
        env.iscene.update(0.0)

    def drop_in(body, dx: float, h: float) -> None:
        """Transport an item to hover above the mouth (crate-frame x = dx), release."""
        from isaaclab.utils.math import quat_apply

        q = scene.crate.data.root_quat_w[0:1]
        off = torch.tensor([[dx, 0.0, c.rim_z_local + c.drop_clear + h / 2]], device=device)
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = scene.crate.data.root_pos_w[0] + quat_apply(q, off)[0]
        st[0, 3:7] = q[0]
        body.write_root_state_to_sim(st, one)

    def drive_flap(buf: torch.Tensor, angle_fn, col: int, mgd: float, k: float,
                   w_des: float, cap: float, max_steps: int,
                   trace: list[float] | None = None) -> bool:
        """The solve's closing servo, through the SAME plant buffer: gravity-ff rate
        servo, torque cut at 8 deg. Returns True iff the closed approach was reached."""
        cut = math.radians(8.0)
        best_a, best_i = float("inf"), 0
        done = False
        for i in range(max_steps):
            a = float(angle_fn()[0])
            if a < cut:
                done = True
                break
            if a < best_a - 0.01:
                best_a, best_i = a, i
            if i - best_i > 480:  # 4 s without progress: stalled (expected when blocked)
                break
            w_close = -float(scene.flap_rate[0, col])
            w_eff = w_des * min(1.0, max(0.15, a / 0.7))
            buf[:] = max(-cap, min(cap, -mgd * math.cos(a) + k * (w_eff - w_close)))
            step(1)
            if trace is not None and i % 20 == 0:
                trace.append(float(scene.score()[0]))
        buf[:] = 0.0
        return done

    base2 = torch.tensor([c.base_pos[0], c.base_pos[1]], device=device)
    deg = math.degrees

    # ========================= 1. settle / clean-slate ========================================
    env.reset(seed=101)
    step(90)
    report("reset")
    finite = all(bool(torch.isfinite(b.data.root_state_w).all())
                 for b in (scene.crate, scene.tuck, scene.main, scene.can, scene.candle)) \
        and bool(torch.isfinite(scene.score()).all())
    can_z = float(item_world(scene.can)[2])
    cnd_z = float(item_world(scene.candle)[2])
    check("settle: clean reset (finite, crate upright at rest, flaps at open rest, "
          "items standing, score ~0)",
          finite and bool(scene.crate_upright()[0])
          and deg(float(scene.tuck_open()[0])) > 150.0
          and deg(float(scene.main_open()[0])) > 150.0
          and abs(can_z - c.can_h / 2) < 0.02 and abs(cnd_z - c.candle_h / 2) < 0.02
          and float(scene.score()[0]) < 0.02)

    # ========================= 2+3. randomization + reach =====================================
    draws = []
    sides = set()
    max_d, min_open = 0.0, 1e9
    for seed in (11, 12, 13, 14, 15):
        env.reset(seed=seed)
        step(30)
        p = scene.crate_pos()[0]
        q = scene.crate.data.root_quat_w[0]
        yaw = float(2.0 * torch.atan2(q[3], q[0]))
        loc = scene.items_local()[0]
        r_can = float(loc[0, :2].norm())
        sides.add(1 if float(loc[0, 1]) > 0 else -1)
        draws.append((round(float(p[0]), 3), round(float(p[1]), 3), round(yaw, 2),
                      round(r_can, 3)))
        for body in (scene.can, scene.candle):
            max_d = max(max_d, float((item_world(body)[:2] - base2).norm()))
        max_d = max(max_d, float((p[:2] - base2).norm()))
        min_open = min(min_open, deg(float(scene.tuck_open()[0])),
                       deg(float(scene.main_open()[0])))
    print(f"[smoke] draws (crate_xy, yaw, r_can) across seeds: {draws} | sides={sides} | "
          f"max dist(base)={max_d:.3f} m | min flap open={min_open:.1f} deg", flush=True)
    check("randomization is real (crate xy + yaw / item radius / item side vary across seeds)",
          len({d[:2] for d in draws}) >= 3 and len({d[2] for d in draws}) >= 3
          and len({d[3] for d in draws}) >= 3 and len(sides) == 2)
    check("reach: everything within 0.75 m of the documented base; flaps settle at the "
          "open rest, every seed",
          max_d < 0.75 and min_open > 150.0)

    # ========================= 4. null policy =================================================
    env.reset(seed=7)
    step(30)
    p0 = scene.crate_pos()[0, :2].clone()
    step(300)
    report("null")
    check("null policy: 2.5 s of nothing -> score ~0, flaps stay open, no crate creep",
          float(scene.score()[0]) < 0.02 and not bool(scene.success()[0])
          and deg(float(scene.tuck_open()[0])) > 150.0
          and deg(float(scene.main_open()[0])) > 150.0
          and float((scene.crate_pos()[0, :2] - p0).norm()) < 0.01)

    # ========================= 5. negative: sub-lift torque ====================================
    lift_need = 0.985 * c.tuck_mgd  # moment to lift the tuck off the -170 open stop
    scene.tuck_tau[:] = 0.61 * lift_need
    step(180)  # held 1.5 s
    a_held = deg(float(scene.tuck_open()[0]))
    scene.tuck_tau[:] = 0.0
    step(60)
    report("sub-lift")
    check("sub-lift torque: 61% of the tuck's lift-off moment held 1.5 s -> the flap "
          "never leaves the open rest (flipping is a real threshold)",
          a_held > 150.0 and deg(float(scene.tuck_open()[0])) > 150.0
          and float(scene.score()[0]) < 0.02)

    # ========================= 6. near-miss A: against the outside wall ========================
    teleport_local(scene.can, 0.0, c.outer[1] / 2 + c.can_r + 0.002, c.can_h / 2 + 0.002)
    step(150)
    report("beside")
    loc = scene.items_local()[0]
    beside = float(loc[0, 1].abs()) > c.contain_y and float(item_world(scene.can)[2]) < 0.06
    check("near-miss: can settled on the floor AGAINST the crate's outside wall -> "
          "no packing credit, score ~0",
          beside and not bool(scene.packed_can[0]) and float(scene.score()[0]) < 0.02)

    # ========================= drop the can in (sets up 7+8) ==================================
    drop_in(scene.can, c.drop_dx[0], c.can_h)
    ok_can = False
    for _ in range(480):
        step(1)
        if bool(scene.packed_can[0]) and bool(scene.contained()[0, 0]):
            ok_can = True
            break
    step(30)
    s_can = float(scene.score()[0])
    report("can-in")

    # ========================= 7. near-miss B: stacked, poking above the rim ==================
    can_w = item_world(scene.can)
    from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402

    can_loc = scene.items_local()[0, 0]
    teleport_local(scene.candle, float(can_loc[0]), float(can_loc[1]),
                   float(can_w[2]) + c.can_h / 2 + c.candle_h / 2 + 0.003)
    step(180)
    report("stacked")
    loc = scene.items_local()[0]
    stacked = float(loc[1, 2]) > c.contain_z[1] and bool(scene.items_settled()[0, 1])
    check("near-miss: candle stacked ON the in-crate can, poking above the rim -> "
          "rejected (containment judged below the aperture), score stays 0.20",
          ok_can and abs(s_can - 0.20) < 0.02 and stacked
          and not bool(scene.packed_candle[0]) and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.20) < 0.02)

    # ========================= 8. negative: the seed's full plan (lid untouched) ==============
    drop_in(scene.candle, c.drop_dx[1], c.candle_h)
    ok_cnd = False
    for _ in range(480):
        step(1)
        if bool(scene.packed_candle[0]) and bool(scene.contained()[0, 1]):
            ok_cnd = True
            break
    step(30)
    report("seed-plan")
    check("negative (seed strategy): BOTH items packed into the open crate, lid never "
          "touched -> success False, score exactly the packing credit 0.40",
          ok_cnd and deg(float(scene.tuck_open()[0])) > 150.0
          and deg(float(scene.main_open()[0])) > 150.0
          and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.40) < 0.02)

    # ========================= 9. negative: WRONG ORDER (main first) ==========================
    # Close the MAIN flap first through the same servo/buffer: it parks on its dip
    # stop, inside the closed band — and that alone must NOT be success. Then drive
    # the TUCK with the full working torque, ending with a hard 2 s press at the cap.
    main_first = drive_flap(scene.main_tau, scene.main_open, 1,
                            c.main_mgd, 0.10, 2.0, c.main_tau_max, 1200)
    step(240)
    a_main_solo = float(scene.main_open()[0])
    solo_ok = (main_first and c.closed_tol[0] < a_main_solo < c.closed_tol[1]
               and not bool(scene.success()[0]))
    report("main-first")
    drive_flap(scene.tuck_tau, scene.tuck_open, 0,
               c.tuck_mgd, 0.004, 2.5, c.tuck_tau_max, 900)  # expected to stall
    scene.tuck_tau[:] = c.tuck_tau_max  # hard press at the full working cap
    step(240)
    a_pressed = float(scene.tuck_open()[0])
    scene.tuck_tau[:] = 0.0
    step(240)
    report("wrong-order")
    a_tuck = float(scene.tuck_open()[0])
    moved = a_tuck < math.radians(35.0)  # the probe genuinely drove the flap down
    cocked = a_tuck > c.closed_tol[1] and a_pressed > c.closed_tol[1]
    check("negative (wrong order): main closed first parks in-band but is NOT success; "
          "tuck then driven at the full working torque rides ON the main, cocked out "
          "of band, weave INVERTED -> no lid credit",
          solo_ok and moved and cocked
          and float(scene.weave_delta()[0]) < 0.0
          and not bool(scene.tuck_set[0]) and not bool(scene.lid_set[0])
          and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.40) < 0.02)

    # ========================= 10+11. oracle: full correct sequence ===========================
    env.reset(seed=5)
    step(90)
    trace: list[float] = [float(scene.score()[0])]
    s_start = trace[0]
    drop_in(scene.can, c.drop_dx[0], c.can_h)
    for _ in range(480):
        step(1)
        if bool(scene.packed_can[0]):
            break
    trace.append(float(scene.score()[0]))
    drop_in(scene.candle, c.drop_dx[1], c.candle_h)
    for _ in range(480):
        step(1)
        if bool(scene.packed_candle[0]):
            break
    step(30)
    trace.append(s_pack := float(scene.score()[0]))
    ok_tuck = drive_flap(scene.tuck_tau, scene.tuck_open, 0,
                         c.tuck_mgd, 0.004, 2.5, c.tuck_tau_max, 1800, trace)
    for _ in range(900):
        step(1)
        if bool(scene.tuck_set[0]):
            break
    step(30)
    trace.append(s_tuck := float(scene.score()[0]))
    report("oracle-tuck")
    ok_main = drive_flap(scene.main_tau, scene.main_open, 1,
                         c.main_mgd, 0.10, 2.0, c.main_tau_max, 1800, trace)
    for _ in range(900):
        step(1)
        if bool(scene.lid_set[0]) and bool(scene.success()[0]):
            break
    step(60)
    trace.append(s_fin := float(scene.score()[0]))
    report("oracle-done")
    ladder = [s_start, s_pack, s_tuck, s_fin]
    print(f"[smoke] ladder: {' -> '.join(f'{s:.3f}' for s in ladder)}", flush=True)
    check("oracle: same-buffer full sequence (pack, tuck first, main on top) -> "
          "success() and score == 1.0 exactly (ladder 0 -> 0.4 -> 0.6 -> 1.0)",
          ok_tuck and ok_main and bool(scene.success()[0]) and abs(s_fin - 1.0) < 1e-3
          and abs(s_pack - 0.40) < 0.02 and abs(s_tuck - 0.60) < 0.02
          and float(scene.weave_delta()[0]) > c.weave_min)
    mono = all(b >= a - 1e-4 for a, b in zip(trace, trace[1:]))
    print(f"[smoke] score trace: "
          f"{' '.join(f'{s:.3f}' for s in trace[::max(1, len(trace) // 14)])}", flush=True)
    check("rubric monotonicity: score never decreases along the oracle run", mono)

    # ========================= 12. persistence ================================================
    flicker = 0
    for _ in range(12):
        step(20)
        if not bool(scene.success()[0]):
            flicker += 1
    report("persist")
    check("persistence: success holds 2 further seconds with no flicker", flicker == 0)

    # ========================= 13. achievement latch ==========================================
    # Drive the main flap back open (constant opening torque, cut past vertical).
    for _ in range(600):
        a = float(scene.main_open()[0])
        if a > math.radians(100.0):
            break
        scene.main_tau[:] = -0.45
        step(1)
    scene.main_tau[:] = 0.0
    step(300)
    report("reopened")
    check("achievement latch: main flap driven back open -> success revoked, latched "
          "0.80 remains",
          deg(float(scene.main_open()[0])) > 150.0
          and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.80) < 0.02)

    # ========================= 14. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.weave_close_crate")
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

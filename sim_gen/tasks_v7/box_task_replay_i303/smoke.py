"""Smoke battery for CartonFlipPackScene — REJECTION tests for the rubric, NullRobot,
RECORDED.

This is NOT the solution (solve.py is — it certifies the rubric ACCEPTS the correct
outcome). Teleported states here are rubric INSTRUMENTATION: construct a wrong outcome
as a settled state, then assert the rubric REJECTS it.

One linear run, 15 named checks:
  1. settle      — clean reset: finite state, carton MOUTH-DOWN at rest, both items
                   standing on the floor, score ~0;
  2. random      — carton xy + yaw, item radii and item side assignment differ across
                   seeds (READBACK, not cfg);
  3. reach       — every seed: carton + both items within the documented Franka
                   envelope, AND both roll corridors clear of the items (|item body-x|
                   beyond the swept half-width);
  4. null        — 2.5 s of nothing: score ~0, the carton stays mouth-down, no creep;
  5. negative A  — the SEED's plan shape without the flip (items placed ON TOP of the
                   overturned carton's upturned base, settled there) -> rejected ~0;
  6. negative B  — item trapped UNDER the mouth-down carton (standing in the cavity,
                   settled) -> rejected ~0 (the upright gate);
  7. negative C  — sub-threshold torque: 88% of the tipping moment held 1.5 s through
                   the same plant buffer, then released -> the carton must NOT right
                   (still mouth-down, no credit);
  8. oracle      — the solve's rate-servo roll through the SAME buffer: two edge
                   pivots, carton settles upright, `righted` latches -> score 0.30,
                   and the carton translated a plausible rolling distance (pivoted,
                   not scooted);
  9. monotone    — the score never decreases along the roll;
 10. near-miss A — can settled on the FLOOR leaning against the upright carton's
                   outside wall -> no packing credit, score stays 0.30;
 11. near-miss B — candle stacked ON TOP of the in-carton can, poking out of the
                   mouth (settled) -> rejected (containment is judged BELOW the
                   aperture), score stays 0.55;
 12. exactness   — candle physically dropped in beside the can -> success() and
                   score == 1.0 exactly (ladder 0 -> 0.30 -> 0.55 -> 1.00);
 13. persist     — success holds 2 further seconds with no flicker;
 14. latch       — the can lifted back out to open floor: success revoked, score
                   falls to exactly 0.80 (latched credit does not evaporate);
 15. frames      — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.box_task_replay_i303.smoke --headless
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

import os  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

try:
    from .scene import CartonFlipPackScene  # registers "carton_flip_pack" + env
except ImportError:  # direct-file fallback
    from scene import CartonFlipPackScene

assert CartonFlipPackScene is not None


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.carton_flip_pack")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.85, -0.85, 0.70)) + o),
                                tuple(np.array((0.0, -0.05, 0.06)) + o),
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
        p = scene.carton_pos()[0].tolist()
        loc = scene.items_local()[0]
        print(f"[smoke] {tag:12s} | carton=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
              f"up_z={float(scene.up_z()[0]):+.3f} "
              f"can_loc=({float(loc[0, 0]):+.3f},{float(loc[0, 1]):+.3f},{float(loc[0, 2]):+.3f}) "
              f"cnd_loc=({float(loc[1, 0]):+.3f},{float(loc[1, 1]):+.3f},{float(loc[1, 2]):+.3f}) "
              f"R={bool(scene.righted[0])} Pc={bool(scene.packed_can[0])} "
              f"Pd={bool(scene.packed_candle[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def teleport(body, x: float, y: float, z: float) -> None:
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = scene.env_origins[0] + torch.tensor([x, y, z], device=device)
        st[0, 3] = 1.0
        body.write_root_state_to_sim(st, one)
        env.iscene.update(0.0)

    def item_world(body):
        return (body.data.root_pos_w[0] - scene.env_origins[0])

    def drive_roll(max_steps: int, trace: list[float]) -> bool:
        """The oracle's righting loop (same servo law as solve.py): smooth one-sided
        rate servo torque about the carton's ridge axis through the scene's
        `roll_tau` buffer; gravity is the brake; cut past the second balance point."""
        a = scene.roll_axis_w()[0]
        axy = a[:2] / a[:2].norm().clamp_min(1e-6)
        perp = torch.tensor([-float(axy[1]), float(axy[0])], device=device)
        c_xy = scene.carton_pos()[0, :2]
        item_xy = torch.stack([item_world(scene.can)[:2], item_world(scene.candle)[:2]])

        def clear(sgn: float) -> float:
            t = -sgn * perp
            rel = item_xy - c_xy
            lam = (rel @ t).clamp(0.0, 0.30)
            return float((rel - lam.unsqueeze(-1) * t).norm(dim=-1).min())

        s = 1.0 if clear(1.0) >= clear(-1.0) else -1.0
        done = False
        for i in range(max_steps):
            if float(scene.up_z()[0]) >= 0.70:
                done = True
                break
            w = s * float(scene.roll_rate()[0])
            scene.roll_tau[:] = s * min(0.42, max(0.0, 0.20 + 0.25 * (1.0 - w)))
            step(1)
            if i % 20 == 0:
                trace.append(float(scene.score()[0]))
        scene.roll_tau[:] = 0.0
        trace.append(float(scene.score()[0]))
        return done

    def drop_in(body, dx: float, h: float) -> None:
        """Transport an item to hover above the mouth (body-x offset dx), release."""
        from isaaclab.utils.math import quat_apply

        q = scene.carton.data.root_quat_w[0:1]
        off = quat_apply(q, torch.tensor([[dx, 0.0, 0.0]], device=device))[0]
        cp = scene.carton.data.root_pos_w[0]
        st = torch.zeros(1, 13, device=device)
        st[0, 0] = cp[0] + off[0]
        st[0, 1] = cp[1] + off[1]
        st[0, 2] = cp[2] + c.rim_z_local + c.drop_clear + h / 2
        st[0, 3] = 1.0
        body.write_root_state_to_sim(st, one)

    base2 = torch.tensor([c.base_pos[0], c.base_pos[1]], device=device)

    # ========================= 1. settle / clean-slate ========================================
    env.reset(seed=101)
    step(60)
    report("reset")
    finite = all(bool(torch.isfinite(b.data.root_state_w).all())
                 for b in (scene.carton, scene.can, scene.candle)) \
        and bool(torch.isfinite(scene.score()).all())
    can_z = float(item_world(scene.can)[2])
    cnd_z = float(item_world(scene.candle)[2])
    check("settle: clean reset (finite, carton mouth-down at rest, items standing, score ~0)",
          finite and float(scene.up_z()[0]) < -0.95
          and abs(float(scene.carton_pos()[0, 2]) - c.rest_z_down) < 0.01
          and abs(can_z - c.can_h / 2) < 0.02 and abs(cnd_z - c.candle_h / 2) < 0.02
          and float(scene.score()[0]) < 0.02)

    # ========================= 2+3. randomization + reach + corridors =========================
    draws = []
    sides = set()
    max_d, min_corridor = 0.0, 1e9
    for seed in (11, 12, 13, 14, 15):
        env.reset(seed=seed)
        step(10)
        p = scene.carton_pos()[0]
        q = scene.carton.data.root_quat_w[0]
        # mouth-down quat (0, cos y/2, sin y/2, 0): yaw = 2*atan2(qx... recover from q
        yaw = float(2.0 * torch.atan2(q[2], q[1]))
        loc = scene.items_local()[0]
        r_can = float(loc[0, :2].norm())
        sides.add(1 if float(loc[0, 0]) > 0 else -1)
        draws.append((round(float(p[0]), 3), round(float(p[1]), 3), round(yaw, 2),
                      round(r_can, 3)))
        for body in (scene.can, scene.candle):
            max_d = max(max_d, float((item_world(body)[:2] - base2).norm()))
        max_d = max(max_d, float((p[:2] - base2).norm()))
        # both roll corridors sweep +-(outer_x/2) about the ridge line: items must
        # sit beyond that in |body x| (plus their radius)
        min_corridor = min(min_corridor,
                           float(loc[:, 0].abs().min()) - c.outer[0] / 2 - c.can_r)
    print(f"[smoke] draws (carton_xy, yaw, r_can) across seeds: {draws} | sides={sides} | "
          f"max dist(base)={max_d:.3f} m | min corridor margin={min_corridor:.3f} m",
          flush=True)
    check("randomization is real (carton xy + yaw / item radius / item side vary across seeds)",
          len({d[:2] for d in draws}) >= 3 and len({d[2] for d in draws}) >= 3
          and len({d[3] for d in draws}) >= 3 and len(sides) == 2)
    check("reach + corridors: everything within 0.75 m of the documented base; both roll "
          "corridors clear of the items, every seed",
          max_d < 0.75 and min_corridor > 0.0)

    # ========================= 4. null policy =================================================
    env.reset(seed=7)
    p0 = scene.carton_pos()[0, :2].clone()
    step(300)
    report("null")
    check("null policy: 2.5 s of nothing -> score ~0, carton stays mouth-down, no creep",
          float(scene.score()[0]) < 0.02 and not bool(scene.success()[0])
          and float(scene.up_z()[0]) < -0.95
          and float((scene.carton_pos()[0, :2] - p0).norm()) < 0.01)

    # ========================= 5. negative A: the seed's plan without the flip =================
    # box_task_replay's plan shape: put the items onto/into the ready box. With the
    # carton overturned, the closest literal execution is items ON the upturned base.
    cp = scene.carton_pos()[0]
    teleport(scene.can, float(cp[0]) + 0.03, float(cp[1]), c.outer[2] + c.can_h / 2 + 0.003)
    teleport(scene.candle, float(cp[0]) - 0.04, float(cp[1]), c.outer[2] + c.candle_h / 2 + 0.003)
    step(150)
    report("on-top")
    on_top = (float(item_world(scene.can)[2]) > 0.09
              and float(item_world(scene.candle)[2]) > 0.09
              and float((item_world(scene.can)[:2] - cp[:2]).norm()) < 0.09
              and float((item_world(scene.candle)[:2] - cp[:2]).norm()) < 0.09)
    check("negative (seed strategy): items settled ON TOP of the overturned carton -> "
          "rejected, score ~0",
          on_top and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.02)

    # ========================= 6. negative B: trapped under the carton =========================
    env.reset(seed=8)
    step(30)
    cp = scene.carton_pos()[0]
    teleport(scene.can, float(cp[0]), float(cp[1]), c.can_h / 2 + 0.002)  # inside the cavity
    step(150)
    report("trapped")
    d_xy = float((item_world(scene.can)[:2] - scene.carton_pos()[0, :2]).norm())
    check("negative: can standing INSIDE the cavity under the mouth-down carton -> "
          "rejected, score ~0 (upright gate)",
          d_xy < 0.05 and abs(float(item_world(scene.can)[2]) - c.can_h / 2) < 0.02
          and float(scene.up_z()[0]) < -0.95
          and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.02)

    # ========================= 7. negative C: sub-threshold torque =============================
    env.reset(seed=3)
    step(30)
    p0 = scene.carton_pos()[0, :2].clone()
    scene.roll_tau[:] = 0.88 * c.tau_tip1  # 88% of the mouth-down tipping moment
    step(180)  # held 1.5 s
    uz_held = float(scene.up_z()[0])
    scene.roll_tau[:] = 0.0
    step(90)
    report("sub-thresh")
    check("sub-threshold: 88% of the tipping moment held 1.5 s -> the carton never "
          "rights (tipping is a real physical threshold)",
          uz_held < -0.98 and float(scene.up_z()[0]) < -0.98
          and float((scene.carton_pos()[0, :2] - p0).norm()) < 0.03
          and float(scene.score()[0]) < 0.02)

    # ========================= 8. oracle: the two-pivot roll ===================================
    trace: list[float] = [float(scene.score()[0])]
    s_start = trace[0]
    roll_start = scene.carton_pos()[0, :2].clone()
    rolled = drive_roll(1800, trace)
    ok_r = False
    for _ in range(900):
        step(1)
        if bool(scene.righted[0]) and bool(scene.carton_upright()[0]) \
                and bool(scene.carton_settled()[0]):
            ok_r = True
            break
    step(30)
    trace.append(float(scene.score()[0]))
    report("righted")
    s_right = float(scene.score()[0])
    travel = float((scene.carton_pos()[0, :2] - roll_start).norm())
    check("oracle roll: same-buffer rate servo -> two edge pivots, carton settles upright, "
          "righted latches at 0.30, travel is a rolling distance (pivoted, not scooted)",
          rolled and ok_r and abs(s_right - 0.30) < 0.02 and 0.10 < travel < 0.45)

    # ========================= 9. monotonicity =================================================
    mono = all(b >= a - 1e-4 for a, b in zip(trace, trace[1:]))
    print(f"[smoke] score trace: "
          f"{' '.join(f'{s:.3f}' for s in trace[::max(1, len(trace) // 14)])}", flush=True)
    check("rubric monotonicity: score never decreases along the roll", mono)

    # ========================= 10. near-miss A: leaning against the outside ===================
    from isaaclab.utils.math import quat_apply

    off_w = quat_apply(scene.carton.data.root_quat_w[0:1], torch.tensor(
        [[0.0, c.outer[1] / 2 + c.can_r + 0.002, 0.0]], device=device))[0]
    cp = scene.carton_pos()[0]
    teleport(scene.can, float(cp[0] + off_w[0]), float(cp[1] + off_w[1]), c.can_h / 2 + 0.002)
    step(150)
    report("beside")
    loc = scene.items_local()[0]
    beside = float(loc[0, 1].abs()) > c.contain_y and float(item_world(scene.can)[2]) < 0.06
    check("near-miss: can settled on the floor AGAINST the upright carton's outside wall "
          "-> no packing credit, score stays 0.30",
          beside and not bool(scene.packed_can[0])
          and abs(float(scene.score()[0]) - 0.30) < 0.02)

    # ========================= 11. near-miss B: stacked, poking out of the mouth ===============
    drop_in(scene.can, c.drop_dx[0], c.can_h)  # the real can drop first
    ok_can = False
    for _ in range(480):
        step(1)
        if bool(scene.packed_can[0]) and bool(scene.contained()[0, 0]):
            ok_can = True
            break
    step(30)
    s_can = float(scene.score()[0])
    report("can-in")
    # candle ON TOP of the in-carton can: rests settled but pokes out of the mouth
    can_w = item_world(scene.can)
    teleport(scene.candle, float(can_w[0]), float(can_w[1]),
             float(can_w[2]) + c.can_h / 2 + c.candle_h / 2 + 0.003)
    step(180)
    report("stacked")
    loc = scene.items_local()[0]
    stacked = float(loc[1, 2]) > c.contain_z[1] and bool(scene.items_settled()[0, 1])
    check("near-miss: candle stacked ON the in-carton can, poking out of the mouth -> "
          "rejected (containment judged below the aperture), score stays 0.55",
          ok_can and abs(s_can - 0.55) < 0.02 and stacked
          and not bool(scene.packed_candle[0]) and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.55) < 0.02)

    # ========================= 12. exactness: drop the candle in beside the can ================
    drop_in(scene.candle, c.drop_dx[1], c.candle_h)
    for _ in range(480):
        step(1)
        if bool(scene.success()[0]):
            break
    step(60)
    report("packed")
    s_fin = float(scene.score()[0])
    ladder = [s_start, s_right, s_can, s_fin]
    print(f"[smoke] ladder: {' -> '.join(f'{s:.3f}' for s in ladder)}", flush=True)
    check("exactness: candle dropped in beside the can -> success() and score == 1.0 "
          "exactly (ladder 0 -> 0.30 -> 0.55 -> 1.00, monotone)",
          bool(scene.success()[0]) and abs(s_fin - 1.0) < 1e-3
          and all(b >= a - 1e-6 for a, b in zip(ladder, ladder[1:])))

    # ========================= 13. persistence =================================================
    flicker = 0
    for _ in range(12):
        step(20)
        if not bool(scene.success()[0]):
            flicker += 1
    report("persist")
    check("persistence: success holds 2 further seconds with no flicker", flicker == 0)

    # ========================= 14. achievement latch ===========================================
    cp = scene.carton_pos()[0]
    teleport(scene.can, float(cp[0]) + 0.25, float(cp[1]) + 0.20, c.can_h / 2 + 0.002)
    step(90)
    report("knock-out")
    check("achievement latch: can lifted back out to open floor -> success revoked, "
          "latched 0.80 remains",
          not bool(scene.success()[0]) and abs(float(scene.score()[0]) - 0.80) < 0.02)

    # ========================= 15. save + verdict ==============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.carton_flip_pack")
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

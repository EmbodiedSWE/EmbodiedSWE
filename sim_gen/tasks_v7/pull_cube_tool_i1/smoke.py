"""Smoke battery for CarouselFerryScene — REJECTION tests for the rubric, NullRobot,
RECORDED.

This is NOT the solution (solve.py is — it certifies the rubric ACCEPTS the correct
outcome). Teleported states here are rubric INSTRUMENTATION: construct a wrong outcome
as a settled state, then assert the rubric REJECTS it.

One linear run, 15 named checks:
  1. settle      — clean reset: finite state, cube riding the far rim, score ~0;
  2. random      — cube spawn / platter yaw / tray centre differ across seeds (READBACK);
  3. reach       — the cube spawns > 0.90 m from the documented base, every seed;
  4. null        — 2 s of nothing: score ~0, no success;
  5. negative A  — the SEED's plan end state (cube dragged straight inward, parked near
                   the base on open floor): no ferry, no tray -> score ~0, rejected;
  6. negative B  — teleport straight into the tray: verified physically inside and
                   settled, but the ferried latch never fired -> rejected, score ~0;
  7. negative C  — rocking the platter back and forth (real riding physics, no net
                   transport): signed sweep cancels -> no latch, no meaningful credit;
  8. oracle      — real torque-driven ferry: cube carried around by friction, ferried
                   latches, score ~0.30;
  9. monotone    — the score never decreases along the ferry;
 10. anchored    — the axle holds: platter centre moves < 5 mm during the drive;
 11. near-miss   — after a real ferry, cube settled just OUTSIDE the tray wall: ferry
                   credit only, no placement, no success;
 12. exactness   — ferry + physical drop into the tray -> success() and score == 1.0
                   exactly (full ladder 0 -> 0.30 -> 1.00, non-decreasing);
 13. persist     — success holds 2 further seconds with no flicker;
 14. latch       — knock the cube back out: success revoked, score falls to exactly
                   0.50 (latched ferry + placement credit does not evaporate);
 15. frames      — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.pull_cube_tool_i1.smoke --headless
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
    from .scene import CarouselFerryScene  # registers "carousel_ferry" + env
except ImportError:  # direct-file fallback
    from scene import CarouselFerryScene

assert CarouselFerryScene is not None


def _wrap(x: float) -> float:
    return (x + math.pi) % (2 * math.pi) - math.pi


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.carousel_ferry")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.55, -0.95, 1.05)) + o),
                                tuple(np.array((0.50, -0.05, 0.08)) + o),
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
        cube = (scene.cube.data.root_pos_w[0] - scene.env_origins[0]).tolist()
        print(f"[smoke] {tag:12s} | cube=({cube[0]:.3f},{cube[1]:.3f},{cube[2]:.3f}) "
              f"az={math.degrees(float(scene.cube_azimuth()[0])):7.1f} "
              f"sweep={math.degrees(float(scene.net_sweep[0])):7.1f} "
              f"ferried={bool(scene.ferried[0])} placed={bool(scene.placed[0])} "
              f"in_tray={bool(scene.in_tray()[0])} score={float(scene.score()[0]):.3f} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def teleport_cube(x: float, y: float, z: float) -> None:
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = scene.env_origins[0] + torch.tensor([x, y, z], device=device)
        st[0, 3] = 1.0
        scene.cube.write_root_state_to_sim(st, one)
        env.iscene.update(0.0)

    def drive_ferry(max_steps: int = 2400) -> tuple[bool, list[float]]:
        """The oracle's ferry loop (same law as solve.py): closed-loop axle torque until
        the cube azimuth reaches the near sector; returns (converged, score trace)."""
        kp, kd, tau_max = 2.0, 1.5, 1.2
        trace = [float(scene.score()[0])]
        done = False
        for i in range(max_steps):
            az = float(scene.cube_azimuth()[0])
            om = float(scene.platter_rate()[0])
            e = _wrap(math.pi - az)
            if abs(e) < math.radians(4.0) and abs(om) < 0.05:
                done = True
                break
            scene.platter_drive[:] = max(-tau_max, min(tau_max, kp * e - kd * om))
            step(1)
            if i % 20 == 0:
                trace.append(float(scene.score()[0]))
        scene.platter_drive[:] = 0.0
        step(90)
        trace.append(float(scene.score()[0]))
        return done, trace

    base = torch.tensor(c.base_pos, device=device)

    # ========================= 1. settle / clean-slate ========================================
    env.reset(seed=101)
    step(60)
    report("reset")
    finite = all(bool(torch.isfinite(b.data.root_state_w).all())
                 for b in (scene.platter, scene.cube, scene.mat, *scene.pegs, *scene.walls)) \
        and bool(torch.isfinite(scene.score()).all())
    check("settle: clean reset (finite state, cube riding the far rim, score ~0)",
          finite and bool(scene.riding()[0]) and float(scene.score()[0]) < 0.02)

    # ========================= 2+3. randomization + out-of-reach ==============================
    draws = []
    min_dist = 1e9
    for seed in (11, 12, 13):
        env.reset(seed=seed)
        step(10)
        cube = (scene.cube.data.root_pos_w[0, :2] - scene.env_origins[0, :2])
        yaw = float(scene.platter_yaw()[0])
        tray = scene.tray_xy()[0]
        draws.append((round(float(cube[0]), 3), round(float(cube[1]), 3),
                      round(math.degrees(yaw), 1),
                      round(float(tray[0]), 3), round(float(tray[1]), 3)))
        min_dist = min(min_dist, float((cube - base).norm()))
    print(f"[smoke] draws (cube_xy, platter_yaw, tray_xy) across seeds: {draws} | "
          f"min dist(cube, base)={min_dist:.3f} m", flush=True)
    check("randomization is real (cube spawn / platter yaw / tray centre differ across seeds)",
          len({d[:2] for d in draws}) >= 2 and len({d[2] for d in draws}) >= 2
          and len({d[3:] for d in draws}) >= 2)
    check("spawn out of reach (cube > 0.90 m from the documented base, every seed)",
          min_dist > 0.90)

    # ========================= 4. null policy =================================================
    env.reset(seed=7)
    step(240)
    report("null")
    check("null policy: 2 s of nothing -> score ~0, no success",
          float(scene.score()[0]) < 0.02 and not bool(scene.success()[0]))

    # ========================= 5. negative A: the seed's plan end state ========================
    # pull_cube_tool's goal: the cube parked within a proximity disc of the base. Here that
    # end state (cube on open floor near the base, never ferried, not in the tray) earns
    # nothing.
    env.reset(seed=7)
    step(30)
    teleport_cube(0.25, 0.0, c.cube_size / 2 + 0.003)
    step(120)
    report("seed-plan")
    near_base = float((scene.cube.data.root_pos_w[0, :2] - scene.env_origins[0, :2]
                       - base).norm()) < 0.45
    check("negative (seed strategy): cube dragged straight to the base -> rejected",
          near_base and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.02)

    # ========================= 6. negative B: teleport into the tray ===========================
    env.reset(seed=8)
    step(30)
    tray = scene.tray_xy()[0]
    teleport_cube(float(tray[0]), float(tray[1]),
                  c.wall_top + c.cube_size / 2 + 0.025)
    step(180)
    report("tp-tray")
    d = float((scene.cube.data.root_pos_w[0, :2] - scene.env_origins[0, :2]
               - scene.tray_xy()[0]).norm())
    z = float(scene.cube.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    physically_in = d < 0.05 and abs(z - c.rest_z) < c.rest_z_tol \
        and float(scene.cube.data.root_lin_vel_w[0].norm()) < c.settle_v
    check("anti-cheat: teleport straight into the tray (settled inside) -> rejected, score ~0",
          physically_in and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.02)

    # ========================= 7. negative C: rocking farms nothing ============================
    env.reset(seed=9)
    step(30)
    for phase in range(8):
        scene.platter_drive[:] = 0.9 if phase % 2 == 0 else -0.9
        step(40)
    scene.platter_drive[:] = 0.0
    step(60)
    report("rocked")
    check("anti-cheat: rocking the platter back and forth earns no ferry credit",
          not bool(scene.ferried[0]) and float(scene.score()[0]) < 0.15
          and not bool(scene.success()[0]))

    # ========================= 8-10. oracle ferry ==============================================
    env.reset(seed=3)
    step(30)
    p0 = (scene.platter.data.root_pos_w[0, :2] - scene.env_origins[0, :2]).clone()
    s_start = float(scene.score()[0])
    done, trace = drive_ferry()
    report("ferried")
    p1 = scene.platter.data.root_pos_w[0, :2] - scene.env_origins[0, :2]
    az = math.degrees(abs(_wrap(float(scene.cube_azimuth()[0]) - math.pi)))
    sc = float(scene.score()[0])
    check("oracle ferry: real carried rotation latches ferried, score ~0.30",
          done and bool(scene.ferried[0]) and bool(scene.riding()[0])
          and az <= c.sector_deg and 0.28 <= sc <= 0.32)
    mono = all(b >= a - 1e-4 for a, b in zip(trace, trace[1:]))
    print(f"[smoke] ferry score trace: {' '.join(f'{s:.3f}' for s in trace[::max(1, len(trace)//12)])}",
          flush=True)
    check("rubric monotonicity: score never decreases along the ferry", mono)
    check("axle anchored: platter centre moves < 5 mm during the drive",
          float((p1 - p0).norm()) < 0.005)
    ck_ferry = env.get_states()

    # ========================= 11. near-miss beside the tray ===================================
    tray = scene.tray_xy()[0]
    off = c.tray_inner + c.wall_t + c.cube_size / 2 + 0.012
    teleport_cube(float(tray[0]), float(tray[1]) + off, c.cube_size / 2 + 0.003)
    step(120)
    report("near-miss")
    check("near-miss: settled just outside the tray wall -> ferry credit only, no success",
          not bool(scene.success()[0]) and not bool(scene.placed[0])
          and 0.28 <= float(scene.score()[0]) <= 0.32)

    # ========================= 12. exactness: ferry + physical drop ============================
    env.set_states(ck_ferry)
    step(5)
    tray = scene.tray_xy()[0]
    st = torch.zeros(1, 13, device=device)
    st[0, 0:3] = scene.env_origins[0] + torch.tensor(
        [float(tray[0]), float(tray[1]), c.wall_top + c.cube_size / 2 + 0.025], device=device)
    st[0, 3:7] = scene.cube.data.root_quat_w[0]
    scene.cube.write_root_state_to_sim(st, one)
    step(240)
    report("delivered")
    ladder = [s_start, sc, float(scene.score()[0])]
    print(f"[smoke] ladder: {' -> '.join(f'{s:.3f}' for s in ladder)}", flush=True)
    check("exactness: ferry + drop -> success() and score == 1.0 exactly (ladder monotone)",
          bool(scene.success()[0]) and abs(float(scene.score()[0]) - 1.0) < 1e-3
          and ladder[0] <= ladder[1] <= ladder[2])

    # ========================= 13. persistence =================================================
    flicker = 0
    for _ in range(12):
        step(20)
        if not bool(scene.success()[0]):
            flicker += 1
    report("persist")
    check("persistence: success holds 2 further seconds with no flicker", flicker == 0)

    # ========================= 14. achievement latch ===========================================
    teleport_cube(0.55, 0.45, c.cube_size / 2 + 0.003)  # open floor, clear of platter + tray
    step(90)
    report("knock-out")
    check("achievement latch: knock-out revokes success, latched 0.50 remains",
          not bool(scene.success()[0]) and abs(float(scene.score()[0]) - 0.5) < 0.02)

    # ========================= 15. save + verdict ==============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.carousel_ferry")
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

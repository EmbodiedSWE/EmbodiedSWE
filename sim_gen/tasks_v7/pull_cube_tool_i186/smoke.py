"""Smoke battery for ShelfDropDispatchScene — REJECTION tests for the rubric, NullRobot,
RECORDED.

This is NOT the solution (solve.py is — it certifies the rubric ACCEPTS the correct
outcome). Teleported states here are rubric INSTRUMENTATION: construct a wrong outcome
as a settled state, then assert the rubric REJECTS it.

One linear run, 16 named checks:
  1. settle      — clean reset: finite state, cube on the level shelf, score ~0;
  2. random      — cube spawn xy+yaw / column channel y / pedestal xy differ across
                   seeds (READBACK, not cfg);
  3. reach       — the cube spawns > 0.86 m (3D) from the documented base, every seed;
  4. null        — 2.5 s of nothing: score ~0, shelf stays level, the loaded column
                   does not creep;
  5. negative A  — the SEED's plan end state (cube dragged straight inward, parked on
                   open floor near the base): no release, no pedestal -> score ~0;
  6. negative B  — teleport the cube straight onto the pedestal top: verified
                   physically resting there, settled — but the shelf was never
                   released -> rejected, score ~0 (execution order is enforced);
  7. negative C  — LOAD-BEARING column: pull the column partway (still under the
                   shelf's width) and let go — the shelf must NOT drop, no credit;
  8. oracle      — finish the extraction (real force through the buffer, column slides
                   out under load): the shelf swings down, released latches;
  9. delivery    — entirely passive: the cube slides down the released ramp into the
                   walled pit and settles there -> delivered, score ~0.55;
 10. monotone    — the score never decreases along extraction + delivery;
 11. near-miss A — cube settled on the FLOOR beside the pedestal: delivery credit
                   only, no success;
 12. near-miss B — cube resting ON the pedestal top but off-centre (beyond
                   ped_xy_tol): rejected;
 13. exactness   — centred physical drop onto the pedestal -> success() and
                   score == 1.0 exactly (ladder 0 -> 0.55 -> 1.00, non-decreasing);
 14. persist     — success holds 2 further seconds with no flicker;
 15. latch       — knock the cube off to open floor: success revoked, score falls to
                   exactly 0.55 (latched release+delivery credit does not evaporate);
 16. frames      — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.pull_cube_tool_i186.smoke --headless
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
    from .scene import ShelfDropDispatchScene  # registers "shelf_drop_dispatch" + env
except ImportError:  # direct-file fallback
    from scene import ShelfDropDispatchScene

assert ShelfDropDispatchScene is not None


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.shelf_drop_dispatch")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.60, -1.05, 0.95)) + o),
                                tuple(np.array((0.60, -0.05, 0.12)) + o),
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
        cube = scene.cube_pos()[0].tolist()
        print(f"[smoke] {tag:12s} | cube=({cube[0]:.3f},{cube[1]:.3f},{cube[2]:.3f}) "
              f"pitch={math.degrees(float(scene.shelf_pitch()[0])):6.1f}deg "
              f"prop_y={float(scene.prop_y()[0]):6.3f} "
              f"released={bool(scene.released[0])} delivered={bool(scene.delivered[0])} "
              f"in_pit={bool(scene.in_pit()[0])} on_ped={bool(scene.on_pedestal()[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

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

    def pull_prop(y_goal: float, max_steps: int, trace: list[float]) -> bool:
        """The oracle's extraction loop (same servo law as solve.py): bounded
        closed-loop y force on the column CoM through the scene's `prop_force`
        buffer, until the column reaches y_goal."""
        kv, f_max, v_des = 40.0, 12.0, -0.10
        done = False
        for i in range(max_steps):
            if float(scene.prop_y()[0]) <= y_goal:
                done = True
                break
            vy = float(scene.prop.data.root_lin_vel_w[0, 1])
            scene.prop_force[:, 1] = max(-f_max, min(f_max, kv * (v_des - vy)))
            step(1)
            if i % 20 == 0:
                trace.append(float(scene.score()[0]))
        scene.prop_force[:] = 0.0
        trace.append(float(scene.score()[0]))
        return done

    base3 = torch.tensor([c.base_pos[0], c.base_pos[1], 0.0], device=device)
    rest_on_shelf = c.shelf_top + c.cube_size / 2

    # ========================= 1. settle / clean-slate ========================================
    env.reset(seed=101)
    step(60)
    report("reset")
    finite = all(bool(torch.isfinite(b.data.root_state_w).all())
                 for b in (scene.shelf, scene.prop, scene.cube, scene.ped)) \
        and bool(torch.isfinite(scene.score()).all())
    cube0 = scene.cube_pos()[0]
    check("settle: clean reset (finite state, cube on the level shelf, score ~0)",
          finite and abs(float(cube0[2]) - rest_on_shelf) < 0.02
          and abs(math.degrees(float(scene.shelf_pitch()[0]))) < 2.0
          and float(scene.score()[0]) < 0.02)

    # ========================= 2+3. randomization + out-of-reach ==============================
    draws = []
    min_dist = 1e9
    for seed in (11, 12, 13):
        env.reset(seed=seed)
        step(10)
        p = scene.cube_pos()[0]
        q = scene.cube.data.root_quat_w[0]
        yaw = math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))
        ped = scene.ped_xy()[0]
        draws.append((round(float(p[0]), 3), round(float(p[1]), 3), round(yaw, 1),
                      round(float(scene.prop_y()[0]), 4),
                      round(float(ped[0]), 3), round(float(ped[1]), 3)))
        min_dist = min(min_dist, float((scene.cube_pos()[0] - base3).norm()))
    print(f"[smoke] draws (cube_xy, cube_yaw, prop_y, ped_xy) across seeds: {draws} | "
          f"min 3D dist(cube, base)={min_dist:.3f} m", flush=True)
    check("randomization is real (cube xy+yaw / column y / pedestal xy differ across seeds)",
          len({d[:2] for d in draws}) >= 2 and len({d[2] for d in draws}) >= 2
          and len({d[3] for d in draws}) >= 2 and len({d[4:] for d in draws}) >= 2)
    check("spawn out of reach (cube > 0.86 m from the documented base, every seed)",
          min_dist > 0.86)

    # ========================= 4. null policy =================================================
    env.reset(seed=7)
    py0 = float(scene.prop_y()[0])
    step(300)
    report("null")
    check("null policy: 2.5 s of nothing -> score ~0, shelf level, loaded column does not creep",
          float(scene.score()[0]) < 0.02 and not bool(scene.success()[0])
          and math.degrees(float(scene.shelf_pitch()[0])) > -2.0
          and abs(float(scene.prop_y()[0]) - py0) < 0.008)

    # ========================= 5. negative A: the seed's plan end state ========================
    # pull_cube_tool's goal: the cube dragged inward to a proximity disc around the base.
    # Here that end state (cube on open floor near the base — no release, no pedestal)
    # earns nothing.
    env.reset(seed=7)
    step(30)
    teleport_cube(0.18, 0.05, c.cube_size / 2 + 0.003)
    step(120)
    report("seed-plan")
    near_base = float((scene.cube_pos()[0] - base3).norm()) < 0.45
    check("negative (seed strategy): cube dragged straight to the base -> rejected",
          near_base and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.02)

    # ========================= 6. negative B: teleport onto the pedestal =======================
    env.reset(seed=8)
    step(30)
    ped = scene.ped_xy()[0]
    teleport_cube(float(ped[0]), float(ped[1]), c.ped_rest_z + 0.025)
    step(180)
    report("tp-pedestal")
    p = scene.cube_pos()[0]
    d = (p[:2] - scene.ped_xy()[0]).abs()
    physically_on = bool((d < 0.03).all()) \
        and abs(float(p[2]) - c.ped_rest_z) < c.rest_z_tol \
        and float(scene.cube.data.root_lin_vel_w[0].norm()) < c.settle_v
    check("anti-cheat: cube teleported onto the pedestal (resting, settled) -> rejected, "
          "score ~0 (the shelf was never released)",
          physically_on and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.02)

    # ========================= 7. negative C: partial pull -> the column still bars =============
    env.reset(seed=3)
    step(30)
    trace: list[float] = [float(scene.score()[0])]
    s_start = trace[0]
    part = pull_prop(-0.05, 600, trace)  # still under the shelf's width (half = 0.07)
    step(120)  # let go: the shelf must NOT drop
    report("partial")
    check("load-bearing: column pulled partway (still under the shelf) -> shelf stays up, "
          "no credit",
          part and math.degrees(float(scene.shelf_pitch()[0])) > -5.0
          and not bool(scene.released[0]) and float(scene.score()[0]) < 0.02)

    # ========================= 8. oracle: finish the extraction ================================
    full = pull_prop(-(c.prop_clear_y + 0.03), 900, trace)
    # The shelf swings down under gravity (damped hinge); wait for the release latch.
    rel = False
    for _ in range(50):
        step(12)
        trace.append(float(scene.score()[0]))
        if bool(scene.released[0]):
            rel = True
            break
    report("released")
    check("oracle extraction: column slid out under load -> the shelf swings down, "
          "released latches",
          full and rel and math.degrees(float(scene.shelf_pitch()[0])) < -float(c.release_deg))

    # ========================= 9. passive delivery =============================================
    ok_del = False
    for i in range(1200):
        step(1)
        if i % 20 == 0:
            trace.append(float(scene.score()[0]))
        if (bool(scene.delivered[0]) and bool(scene.in_pit()[0])
                and float(scene.cube.data.root_lin_vel_w[0].norm()) < 0.03):
            ok_del = True
            break
    step(60)
    trace.append(float(scene.score()[0]))
    report("delivered")
    s_del = float(scene.score()[0])
    check("passive delivery: cube slides down the released ramp into the pit "
          "(untouched) -> delivered, score ~0.55",
          ok_del and bool(scene.in_pit()[0]) and abs(s_del - 0.55) < 0.02)

    # ========================= 10. monotonicity ================================================
    mono = all(b >= a - 1e-4 for a, b in zip(trace, trace[1:]))
    print(f"[smoke] score trace: "
          f"{' '.join(f'{s:.3f}' for s in trace[::max(1, len(trace) // 14)])}", flush=True)
    check("rubric monotonicity: score never decreases along extraction + delivery", mono)
    ck_del = env.get_states()

    # ========================= 11. near-miss A: floor beside the pedestal ======================
    ped = scene.ped_xy()[0]
    off = c.ped_size[1] / 2 + c.cube_size / 2 + 0.05
    teleport_cube(float(ped[0]), float(ped[1]) - off, c.cube_size / 2 + 0.003)
    step(120)
    report("near-miss-A")
    check("near-miss: cube settled on the FLOOR beside the pedestal -> delivery credit only",
          not bool(scene.success()[0]) and abs(float(scene.score()[0]) - 0.55) < 0.02)

    # ========================= 12. near-miss B: on top but off-centre ==========================
    env.set_states(ck_del)
    step(5)
    ped = scene.ped_xy()[0]
    teleport_cube(float(ped[0]) + c.ped_xy_tol + 0.015, float(ped[1]), c.ped_rest_z + 0.02)
    step(150)
    report("near-miss-B")
    p = scene.cube_pos()[0]
    on_top = abs(float(p[2]) - c.ped_rest_z) < c.rest_z_tol \
        and float(scene.cube.data.root_lin_vel_w[0].norm()) < c.settle_v
    dx = abs(float(p[0]) - float(scene.ped_xy()[0][0]))
    check("near-miss: cube resting ON the pedestal top but off-centre -> rejected",
          on_top and dx > c.ped_xy_tol
          and not bool(scene.success()[0]) and abs(float(scene.score()[0]) - 0.55) < 0.02)

    # ========================= 13. exactness: centred physical drop ============================
    env.set_states(ck_del)
    step(5)
    ped = scene.ped_xy()[0]
    teleport_cube(float(ped[0]), float(ped[1]), c.ped_rest_z + 0.03)
    step(240)
    report("placed")
    s_fin = float(scene.score()[0])
    ladder = [s_start, s_del, s_fin]
    print(f"[smoke] ladder: {' -> '.join(f'{s:.3f}' for s in ladder)}", flush=True)
    check("exactness: centred drop -> success() and score == 1.0 exactly (ladder monotone)",
          bool(scene.success()[0]) and abs(s_fin - 1.0) < 1e-3
          and ladder[0] <= ladder[1] <= ladder[2])

    # ========================= 14. persistence =================================================
    flicker = 0
    for _ in range(12):
        step(20)
        if not bool(scene.success()[0]):
            flicker += 1
    report("persist")
    check("persistence: success holds 2 further seconds with no flicker", flicker == 0)

    # ========================= 15. achievement latch ===========================================
    teleport_cube(0.08, 0.35, c.cube_size / 2 + 0.003)  # open floor: not pit, not pedestal
    step(90)
    report("knock-out")
    check("achievement latch: knock-out revokes success, latched 0.55 remains",
          not bool(scene.success()[0]) and abs(float(scene.score()[0]) - 0.55) < 0.02)

    # ========================= 16. save + verdict ==============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.shelf_drop_dispatch")
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

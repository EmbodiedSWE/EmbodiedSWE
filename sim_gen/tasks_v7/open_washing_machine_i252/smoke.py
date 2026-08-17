"""Smoke battery for DetergentDrawerScene — REJECTION tests for the rubric, NullRobot,
RECORDED.

This is NOT a solution (the solution is solve.py — pull the drawer with a velocity-
cascade force servo, gravity-drop the pod into the blue-marked cell, push the drawer
home; the Franka strategy is TASK.md's embodiment argument). Drives here are the same
hand-scale numbers as solve.py (shared numbers = shared honesty); every outcome that
is asserted is CONSTRUCTED through the live prismatic joint + contact plant —
teleports are transport-only (free-air pod placement, exactly like solve.py).

One linear run, 12 named checks:
  1. settle    — clean reset: finite everywhere, drawer physically AT its sampled
                 crack (readback), pod physically AT its sampled table pose
                 (readback), score ~0, no success;
  2. markers   — the BLUE tile physically sits over the sampled target cell and the
                 WHITE tile over the other (world readback vs the cfg formula);
  3. random    — across 8 seeds: both target sides drawn, crack varies, pod start
                 varies; every draw readback-verified (opening == crack, pod at start);
  4. null      — 2 s of nothing: score < 0.05, no success;
  5. end stop  — the rail is physical: servo commanded 50 mm PAST the stroke parks
                 at the ~130 mm joint stop, finite, no success;
  6. negative  — the SEED's end state (open it, done): drawer pulled open and left
                 open — opened latch only, score ~0.20, NO success;
  7. negative  — ordering is GEOMETRY: with the drawer shut, the pod dropped from
                 above the target cell lands ON the hood — never in the cell,
                 deposit latch stays 0, score ~0 (you cannot dose a shut drawer);
  8. negative  — wrong compartment: pod deposited in the WHITE-marked cell, drawer
                 pushed shut — pod physically rests on the wrong side, no deposit
                 latch, score ~0.20, NO success;
  9. negative  — near miss: correct deposit but the drawer parked ~30 mm ajar —
                 score ~0.65 (latched credit), NO success (shut means SHUT);
 10. exactness — full correct sequence (open -> blue-cell drop -> shut) -> success()
                 and score == 1.0, still true 1 s later hands-off;
 11. latch     — pulling the drawer back open REVOKES success (success is live
                 state); the latched credit (~0.65, pod still riding in the cell)
                 remains — it never evaporates;
 12. frames    — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.open_washing_machine_i252.smoke --headless
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

robobench.discover()
try:
    from simgen_tasks.open_washing_machine_i252 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie.
threading.Timer(1200.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                 os._exit(3))).start()

# Same hand-scale drawer servo as solve.py (shared numbers = shared honesty).
KX = 3.0
V_OPEN = 0.15
V_SHUT = 0.10
KV = 15.0
F_MAX = 6.0
G_DONE = 0.005
V_DONE = 0.02

PROBE_SEED = 5


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.detergent_drawer")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((0.62, -0.55, 0.95)) + o),
                                tuple(np.array((0.02, 0.0, 0.48)) + o),
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
        g = float(scene.opening()[0])
        p = scene.pod_local()[0]
        print(f"[smoke] {tag:14s} | opening={g * 1000:6.1f}mm "
              f"pod_local=({float(p[0]) * 1000:+6.1f}, {float(p[1]) * 1000:+6.1f}, "
              f"{float(p[2]) * 1000:+6.1f})mm in_cell={bool(scene.in_cell()[0])} "
              f"shut={bool(scene.shut_now()[0])} "
              f"latch=(o={float(scene.opened_latch[0]):.0f}, "
              f"d={float(scene.deposit_latch[0]):.0f}) "
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

    # --- drive helpers (solve.py's servo, verbatim numbers) ---
    def slide_to(g_tgt: float, v_cap: float, budget: int = 1800) -> bool:
        """Velocity-cascade force servo on the drawer's pull axis (solve.py's plant
        interface); releases only close AND slow."""
        for _ in range(budget):
            g = float(scene.opening()[0])
            v = float(scene.drawer.data.root_lin_vel_w[0, 0])
            e = g_tgt - g
            if abs(e) < G_DONE and abs(v) < V_DONE:
                scene.drive_f[0] = 0.0
                return True
            v_des = max(-v_cap, min(v_cap, KX * e))
            scene.drive_f[0] = max(-F_MAX, min(F_MAX, KV * (v_des - v)))
            step(1)
        scene.drive_f[0] = 0.0
        return False

    def drop_pod(side_sign: float, dz: float | None = None) -> None:
        """Transport-only teleport: pod to FREE AIR above the cell on `side_sign`,
        zero velocity; gravity does the deposit. Same recipe as solve.py."""
        origin = scene.env_origins[0]
        drawer_x = float(scene.drawer.data.root_pos_w[0, 0])
        drop = torch.zeros(1, 13, device=device)
        drop[0, 0] = drawer_x + 0.015
        drop[0, 1] = float(origin[1]) + side_sign * c.cell_y_c
        drop[0, 2] = c.rim_z + 0.031 if dz is None else dz
        drop[0, 3] = 1.0
        scene.pod.write_root_state_to_sim(drop, ids0)
        for _ in range(40):
            step(10)
            if bool(scene.settled()[0]):
                break

    def reset(seed: int) -> None:
        torch.manual_seed(seed)
        env.reset()
        step(30)

    # ========================= 1. settle / clean-slate ========================================
    reset(PROBE_SEED)
    report("reset")
    side = float(scene.side[0])
    crack = float(scene.crack0[0])
    open_rb = abs(float(scene.opening()[0]) - crack)
    pod_rb = float((scene.pod.data.root_pos_w[0] - scene.env_origins[0]
                    - scene.pod_start[0]).norm())
    print(f"[smoke] seed={PROBE_SEED} side={side:+.0f} crack={crack * 1000:.1f}mm "
          f"opening_rb_err={open_rb * 1000:.2f}mm pod_rb_err={pod_rb * 1000:.2f}mm", flush=True)
    check("settle: clean reset (finite, drawer AT its crack, pod AT its start, score ~0)",
          finite_all() and open_rb < 0.003 and pod_rb < 0.005
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 2. marker tiles readback =======================================
    tile_err = 0.0
    for name, s in (("blue", side), ("white", -side)):
        want = torch.tensor([c.front_face_x + c.marker_size[0] / 2,
                             s * c.cell_y_c, c.marker_z], device=device)
        got = scene.markers[name].data.root_pos_w[0] - scene.env_origins[0]
        tile_err = max(tile_err, float((got - want).norm()))
    print(f"[smoke] marker readback err={tile_err * 1000:.2f}mm "
          f"(blue over side {side:+.0f})", flush=True)
    check("markers: BLUE tile physically over the sampled target cell, WHITE over the other",
          tile_err < 0.002)

    # ========================= 3. randomization across seeds ==================================
    draws = []
    ok_rb = True
    for sd in (11, 12, 13, 14, 15, 16, 17, 18):
        reset(sd)
        s = float(scene.side[0])
        cr = round(float(scene.crack0[0]) * 1000, 1)
        px = round(float(scene.pod_start[0, 0]) * 1000, 1)
        py = round(float(scene.pod_start[0, 1]) * 1000, 1)
        ok_rb &= abs(float(scene.opening()[0]) - float(scene.crack0[0])) < 0.003
        ok_rb &= float((scene.pod.data.root_pos_w[0] - scene.env_origins[0]
                        - scene.pod_start[0]).norm()) < 0.005
        # the blue tile physically follows the sampled side
        by = float((scene.markers["blue"].data.root_pos_w[0] - scene.env_origins[0])[1])
        ok_rb &= abs(by - s * c.cell_y_c) < 0.002
        draws.append((int(s), cr, px, py))
    print(f"[smoke] draws (side, crack_mm, pod_x_mm, pod_y_mm): {draws}", flush=True)
    sides = {d[0] for d in draws}
    check("randomization is real (both sides drawn; crack + pod pose vary; all readback)",
          ok_rb and sides == {-1, 1}
          and len({d[1] for d in draws}) >= 5 and len({d[2] for d in draws}) >= 5)

    # ========================= 4. null policy =================================================
    reset(PROBE_SEED)
    step(240)  # 2 s of nothing
    report("null")
    check("null policy: score < 0.05 and no success",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 5. the rail's end stop is physical =============================
    reset(PROBE_SEED)
    slide_to(c.stroke + 0.05, V_OPEN, budget=500)  # commanded 50 mm PAST the stroke
    step(120)
    report("end-stop")
    g = float(scene.opening()[0])
    check("end stop: commanded past the stroke — parks at the ~130 mm joint stop, finite",
          0.115 <= g <= 0.138 and finite_all() and not bool(scene.success()[0]))

    # ========================= 6. negative: the SEED's end state (open it, done) ==============
    reset(PROBE_SEED)
    ok_open = slide_to(c.stroke - 0.008, V_OPEN)
    step(120)
    report("left-open")
    check("negative (seed strategy): drawer opened and LEFT OPEN — opened latch only, "
          "score ~0.20, no success",
          ok_open and float(scene.opened_latch[0]) == 1.0
          and not bool(scene.success()[0])
          and 0.15 < float(scene.score()[0]) < 0.30)

    # ========================= 7. negative: hood blocks dosing a shut drawer ==================
    reset(PROBE_SEED)
    # Drop from ABOVE the hood, straight over where the target cell sits (shut).
    drop_pod(side, dz=c.hood_top_z + 0.04)
    report("hood-block")
    pod_z = float((scene.pod.data.root_pos_w[0] - scene.env_origins[0])[2])
    check("negative (ordering is geometry): pod dropped over the SHUT drawer lands on the "
          "hood — never in the cell, no deposit latch, score ~0",
          not bool(scene.in_cell()[0]) and float(scene.deposit_latch[0]) == 0.0
          and pod_z > c.rim_z + 0.02  # rests on/above the hood, not inside the tray
          and float(scene.score()[0]) < 0.05 and finite_all()
          and not bool(scene.success()[0]))

    # ========================= 8. negative: wrong compartment =================================
    reset(PROBE_SEED)
    ok_open = slide_to(c.stroke - 0.008, V_OPEN)
    drop_pod(-side)  # the WHITE-marked softener cell
    ok_shut = slide_to(0.0, V_SHUT)
    step(120)
    report("wrong-cell")
    y_s = float(scene.pod_local()[0, 1]) * side  # < 0 = physically on the wrong side
    check("negative (wrong compartment): pod rests in the WHITE cell, drawer shut — "
          "no deposit latch, score ~0.20, no success",
          ok_open and (ok_shut or bool(scene.shut_now()[0]))
          and y_s < -c.wall_t / 2 and float(scene.deposit_latch[0]) == 0.0
          and not bool(scene.success()[0])
          and 0.15 < float(scene.score()[0]) < 0.30)

    # ========================= 9. negative: near miss — drawer left ajar ======================
    reset(PROBE_SEED)
    ok_open = slide_to(c.stroke - 0.008, V_OPEN)
    drop_pod(side)  # correct cell
    ok_ajar = slide_to(0.030, V_SHUT)  # parked 30 mm ajar — 5x shut_tol
    step(120)
    report("ajar")
    g = float(scene.opening()[0])
    check("negative (near miss): correct deposit but drawer ~30 mm ajar — latched credit "
          "~0.65, no success",
          ok_open and ok_ajar and c.shut_tol + 0.010 < g < 0.050
          and bool(scene.in_cell()[0]) and float(scene.deposit_latch[0]) == 1.0
          and not bool(scene.success()[0])
          and 0.55 < float(scene.score()[0]) < 0.75)

    # ========================= 10. exactness: success == score 1.0 ============================
    reset(PROBE_SEED)
    ok_open = slide_to(c.stroke - 0.008, V_OPEN)
    drop_pod(side)
    ok_shut = slide_to(0.0, V_SHUT)
    step(120)
    report("goal-state")
    good = ok_open and (ok_shut or bool(scene.shut_now()[0])) \
        and bool(scene.success()[0]) and abs(float(scene.score()[0]) - 1.0) < 1e-3
    step(120)  # ...and it persists hands-off (1 s)
    report("goal-persist")
    check("exactness: open -> blue-cell drop -> shut -> success() and score == 1.0, stable",
          good and bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 1.0) < 1e-3)

    # ========================= 11. achievement latch ==========================================
    ok_reopen = slide_to(0.110, V_OPEN)
    step(120)
    report("reopened")
    check("achievement latch: reopening REVOKES success; latched credit (~0.65) remains",
          ok_reopen and not bool(scene.success()[0])
          and 0.55 < float(scene.score()[0]) < 0.75)

    # ========================= 12. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.detergent_drawer")
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

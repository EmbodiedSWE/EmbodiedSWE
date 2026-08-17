"""Smoke battery for PortDropboxScene — REJECTION tests for the rubric, NullRobot,
RECORDED.

This is NOT a solution (the solution is solve.py — stage the red can lying on the
tray, push it end-on through the port with a quasi-static force, gravity drops it
inside; the Franka strategy is TASK.md's embodiment argument). Wrong outcomes are
CONSTRUCTED as settled states (teleport, settle, judge) or driven with the same
fingertip-scale forces as solve.py; every probe must be REJECTED — the audit check
asserts success() never fires at ANY step of this battery.

One linear run, 13 named checks:
  1. settle    — clean reset: finite state, both cans physically upright AT their
                 sampled ground spots (readback), score ~0, no success;
  2. random    — 8 seeds: box yaw varies (both signs), the red/beige SIDES swap,
                 positions vary, cans >= 9 cm apart and clear of the tray (READBACK);
  3. null      — 2 s of nothing: score < 0.05, no success;
  4. negative  — the SEED's plan (carry over the container and let go): a can released
                 above the box settles ON THE ROOF, never inside — no success, ~0;
  5. negative  — sideways at the port: a can lying ACROSS the channel, pushed at the
                 port with solve.py's force scheme, ADVANCES then jams outside (the
                 110 mm length cannot pass the 80 mm aperture) — no credit, no success;
  6. negative  — incomplete: the reorientation stage alone (can rested on the tray,
                 aligned) earns exactly the staged share, no success;
  7. negative  — near miss: can straddling the sill, nose cantilevered past the inner
                 face, CoM outside — settled, NOT inside, partial credit only;
  8. latch     — pulling the straddling can back to the ground leaves the latched
                 credit UNCHANGED (it never evaporates), still no success;
  9. negative  — wrong object: the BEIGE can constructed inside, red outside — no
                 success, ~0;
 10. negative  — constraint: red constructed inside TOO (beige still in) — red is
                 physically inside, yet no success (the beige-outside clause bites);
 11. audit     — success() was never True at any step of this battery;
 12. finite    — no NaN anywhere at the end;
 13. frames    — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.living_room_scene1_pick_up_the_tomato_sauce_and_put_it_in_the_basket_i203.smoke --headless
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
    from simgen_tasks.living_room_scene1_pick_up_the_tomato_sauce_and_put_it_in_the_basket_i203 \
        import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie.
wd = threading.Timer(1200.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                      os._exit(3)))
wd.daemon = True
wd.start()

PROBE_SEED = 5  # fixed probe seed (geometry is deterministic; only poses are sampled)
PUSH_F = 5.0  # N — the sideways-jam probe force (same scale as solve.py's 3.5..9 N)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.port_dropbox")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    e0 = torch.tensor([0], device=device)

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.45, -0.95, 0.85)) + o),
                                tuple(np.array((0.25, 0.0, 0.18)) + o),
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
    success_seen = False

    def step(k: int) -> None:
        nonlocal step_i, success_seen
        for _ in range(k):
            env.step(no_action, render=annot is not None)
            success_seen = success_seen or bool(scene.success()[0])
            if annot is not None and step_i % args.record_every == 0 \
                    and len(frames) < args.max_frames:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    def local_red() -> torch.Tensor:
        return scene.bin_local(scene.red.data.root_pos_w)[0]

    def report(tag: str) -> None:
        p = local_red()
        print(f"[smoke] {tag:14s} | red_local=({float(p[0]):+.3f}, {float(p[1]):+.3f}, "
              f"{float(p[2]):.3f}) staged={float(scene.staged_latch[0]):.2f} "
              f"insert={float(scene.insert_latch[0]):.2f} "
              f"in(red)={bool(scene.inside(scene.red)[0])} "
              f"in(beige)={bool(scene.inside(scene.beige)[0])} "
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

    def teleport(body, lx: float, ly: float, lz: float, quat: tuple) -> None:
        """Pose a can at BIN-LOCAL (lx, ly, lz) with world quat (w, x, y, z), zero vel."""
        yaw = float(scene.bin_yaw[0])
        cy, sy = math.cos(yaw), math.sin(yaw)
        bx, by = (float(v) for v in scene.bin_xy[0])
        origin = scene.env_origins[0]
        st = torch.zeros(1, 13, device=device)
        st[0, 0] = origin[0] + bx + lx * cy - ly * sy
        st[0, 1] = origin[1] + by + lx * sy + ly * cy
        st[0, 2] = origin[2] + lz
        st[0, 3:7] = torch.tensor(quat, device=device)
        body.write_root_state_to_sim(st, e0)

    def lying_quat(extra_yaw: float = 0.0) -> tuple:
        """World quat for a can lying with its axis at bin yaw + extra_yaw."""
        yaw = float(scene.bin_yaw[0]) + extra_yaw
        half, c45 = yaw / 2, math.cos(math.pi / 4)
        return (math.cos(half) * c45, -math.sin(half) * c45,
                math.cos(half) * c45, math.sin(half) * c45)

    # ========================= 1. settle / clean-slate ========================================
    torch.manual_seed(PROBE_SEED)
    env.reset()
    step(60)
    report("reset")
    origin = scene.env_origins[0]
    red_err = float((scene.red.data.root_pos_w[0, :2] - origin[:2] - scene.red0[0]).norm())
    beige_err = float((scene.beige.data.root_pos_w[0, :2] - origin[:2] - scene.beige0[0]).norm())
    up_z = float(scene.can_axis(scene.red)[0, 2])
    check("settle: clean reset (finite, cans upright AT their sampled spots, score ~0)",
          finite_all() and red_err < 0.006 and beige_err < 0.006 and up_z > 0.95
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 2. randomization across seeds ==================================
    draws = []
    ok_geom = True
    for seed in (11, 12, 13, 14, 15, 16, 17, 18):
        torch.manual_seed(seed)
        env.reset()
        step(2)
        yaw_d = round(math.degrees(float(scene.bin_yaw[0])), 1)
        r = scene.red.data.root_pos_w[0, :2] - origin[:2]
        b = scene.beige.data.root_pos_w[0, :2] - origin[:2]
        sep = float((r - b).norm())
        ok_geom &= sep >= 0.09  # disjoint side bands
        ok_geom &= float(r[0]) < 0.075 and float(b[0]) < 0.075  # clear of the tray
        draws.append((yaw_d, round(float(r[0]), 3), round(float(r[1]), 3),
                      round(float(b[1]), 3)))
    print(f"[smoke] draws (yaw, red_x, red_y, beige_y): {draws}", flush=True)
    red_ys = [d[2] for d in draws]
    yaws = [d[0] for d in draws]
    check("randomization is real (box yaw both signs, red/beige sides swap, "
          "positions vary, cans separated, clear of the tray — READBACK)",
          len(set(yaws)) >= 5 and any(v < 0 for v in yaws) and any(v > 0 for v in yaws)
          and any(v < 0 for v in red_ys) and any(v > 0 for v in red_ys)
          and len({d[1] for d in draws}) >= 5 and ok_geom)

    # ========================= 3. null policy =================================================
    torch.manual_seed(PROBE_SEED)
    env.reset()
    step(240)  # 2 s of nothing
    report("null")
    check("null policy: score < 0.05 and no success",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 4. negative: the SEED's plan (top drop) ========================
    torch.manual_seed(PROBE_SEED)
    env.reset()
    step(30)
    # Carry the can over the container and let go — released 4 mm above the roof.
    teleport(scene.red, 0.0, 0.0, c.wall_top + c.roof_t + c.can_l / 2 + 0.004,
             (1.0, 0.0, 0.0, 0.0))
    step(240)
    report("roof-drop")
    pz = float(local_red()[2])
    check("negative (seed strategy): can released above the box settles ON THE ROOF — "
          "never inside, no success",
          c.wall_top < pz < 0.42 and not bool(scene.inside(scene.red)[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.05
          and finite_all())

    # ========================= 5. negative: sideways at the port ==============================
    torch.manual_seed(PROBE_SEED)
    env.reset()
    step(30)
    rail_rest = c.tray_top + c.rail_h + c.can_r  # lying across, resting on both rails
    teleport(scene.red, -0.21, 0.0, rail_rest + 0.004, lying_quat(math.pi / 2))
    step(90)
    x_before = float(local_red()[0])
    yaw = float(scene.bin_yaw[0])
    dirx, diry = math.cos(yaw), math.sin(yaw)
    for _ in range(360):  # 3 s of pushing at the port
        scene.push_f[0, 0, 0] = PUSH_F * dirx
        scene.push_f[0, 0, 1] = PUSH_F * diry
        step(1)
    scene.push_f[0, 0, :] = 0.0
    step(120)
    report("sideways-jam")
    x_after = float(local_red()[0])
    check("negative (sideways): can lying ACROSS the channel advances then JAMS at the "
          "port — moved but stayed outside, no credit, no success",
          x_after - x_before > 0.015 and -0.21 < x_after < -0.150
          and not bool(scene.inside(scene.red)[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) < 0.05 and finite_all())

    # ========================= 6. negative: incomplete (staged only) ==========================
    torch.manual_seed(PROBE_SEED)
    env.reset()
    step(30)
    teleport(scene.red, c.tray_cx, 0.0, c.can_stage_z + 0.003, lying_quat())
    step(120)
    report("staged-only")
    check("negative (incomplete): reorientation stage alone — staged latch full, "
          "score == the staged share only, no success",
          float(scene.staged_latch[0]) > 0.99
          and 0.24 <= float(scene.score()[0]) <= 0.275
          and not bool(scene.success()[0]))

    # ========================= 7. negative: near miss (straddling the sill) ===================
    torch.manual_seed(PROBE_SEED)
    env.reset()
    step(30)
    teleport(scene.red, -0.14, 0.0, c.can_stage_z + 0.003, lying_quat())
    step(150)
    report("straddle")
    p = local_red()
    straddle_score = float(scene.score()[0])
    check("negative (near miss): can straddling the sill, nose past the inner face, "
          "CoM outside — settled, NOT inside, partial credit only",
          abs(float(p[0]) + 0.14) < 0.02 and not bool(scene.inside(scene.red)[0])
          and not bool(scene.success()[0]) and 0.10 < straddle_score < 0.36)

    # ========================= 8. latched credit never evaporates =============================
    # Pull the straddling can clear of the box — back to open ground on the robot side.
    origin = scene.env_origins[0]
    st = torch.zeros(1, 13, device=device)
    st[0, 0] = origin[0] - 0.15
    st[0, 1] = origin[1] + 0.30
    st[0, 2] = origin[2] + c.can_l / 2 + 0.003
    st[0, 3] = 1.0
    scene.red.write_root_state_to_sim(st, e0)
    step(90)
    report("pulled-back")
    check("latch: pulling the straddling can back out leaves the latched credit "
          "unchanged, still no success",
          abs(float(scene.score()[0]) - straddle_score) < 1e-3
          and not bool(scene.success()[0]))

    # ========================= 9. negative: wrong object inside ===============================
    torch.manual_seed(PROBE_SEED)
    env.reset()
    step(30)
    teleport(scene.beige, 0.05, 0.04, 0.20, (1.0, 0.0, 0.0, 0.0))  # into the interior air
    step(240)
    report("wrong-object")
    check("negative (wrong object): BEIGE can inside, red outside — no success, ~0",
          bool(scene.inside(scene.beige)[0]) and not bool(scene.inside(scene.red)[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.05)

    # ========================= 10. negative: beige-outside clause =============================
    teleport(scene.red, -0.05, -0.04, 0.20, (1.0, 0.0, 0.0, 0.0))
    step(240)
    report("both-inside")
    check("negative (constraint): red inside TOO but beige still in — red is physically "
          "contained yet success is refused",
          bool(scene.inside(scene.red)[0]) and bool(scene.inside(scene.beige)[0])
          and not bool(scene.success()[0]))

    # ========================= 11. rejection audit ============================================
    check("audit: success() never fired at any step of this battery", not success_seen)

    # ========================= 12. finite =====================================================
    check("finite: no NaN anywhere at the end", finite_all())

    # ========================= 13. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.port_dropbox")
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

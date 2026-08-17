"""Smoke battery for BoomCorralScene — REJECTION tests for the rubric, NullRobot,
RECORDED.

This is NOT the solution (solve.py is — it certifies the rubric ACCEPTS the honest
strategy). Teleported states here are rubric INSTRUMENTATION: construct a wrong outcome
as a settled state, then assert the rubric REJECTS it.

One linear run, 16 named checks:
  1. settle      — clean reset: finite state, cube resting on the ground inside the
                   sweep annulus, blade behind it, score ~0;
  2. random      — cube spawn / boom start yaw / pen pose differ across seeds (READBACK);
  3. side flip   — the pen lands on BOTH sides of the cube across a seed scan (the
                   required sweep direction really flips per episode);
  4. reach       — cube AND pen mouth spawn > 1.0 m from the documented base (far beyond
                   the 0.855 m Franka envelope) while the handle starts within 0.75 m;
  5. null        — 2 s of nothing: score ~0, no success;
  6. negative A  — the SEED's plan end state (cube parked near the base on open floor,
                   inside pull_cube_tool's proximity disc): rejected, score ~0;
  7. negative B  — teleport straight into the pen: verified physically inside and
                   settled, but swept/entered never latched -> rejected, score ~0
                   (execution order is enforced, not just declared);
  8. negative C  — exercising the boom back and forth AWAY from the cube (real drive,
                   >= 15 deg of real excursion, no transport): no latch, no credit;
  9. oracle      — the solve's own servo law herds the cube through the mouth: swept +
                   entered latch and success() goes true after the blade parks;
 10. exactness   — score == 1.0 exactly at success; ladder 0 -> 0.5 -> 1.0 monotone;
 11. monotone    — the score never decreases along the oracle herd;
 12. anchored    — the boom stays on its pivot (< 5 mm drift) and the HANDLE stays
                   within the documented 0.75 m envelope for the whole drive;
 13. near-miss   — a real herd stopped ~10 cm short of the mouth, blade parked: real
                   partial credit but NO swept latch, no entry, no success;
 14. persist     — success holds 2 further seconds with no flicker;
 15. latch       — knock the cube back out of the pen: success revoked, score falls to
                   exactly 0.50 (latched sweep + entry credit does not evaporate);
 16. frames      — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.pull_cube_tool_i348.smoke --headless
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

try:
    from .scene import BoomCorralScene  # registers "boom_corral" + env
except ImportError:  # direct-file fallback
    from scene import BoomCorralScene

assert BoomCorralScene is not None


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.boom_corral")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    one = torch.arange(1, device=device)
    dt = 1.0 / 120.0

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.60, -0.80, 1.20)) + o),
                                tuple(np.array((0.60, 0.00, 0.05)) + o),
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
        uw = scene.cube_pen_frame()[0].tolist()
        print(f"[smoke] {tag:12s} | cube=({cube[0]:.3f},{cube[1]:.3f},{cube[2]:.3f}) "
              f"az={math.degrees(float(scene.cube_azimuth()[0])):6.1f} "
              f"boom={math.degrees(float(scene.boom_yaw()[0])):6.1f} "
              f"arc={math.degrees(float(scene.arc_done[0])):6.1f}/"
              f"{math.degrees(float(scene.arc_req[0])):.1f} side={float(scene.side[0]):+.0f} "
              f"uw=({uw[0]:+.3f},{uw[1]:+.3f}) swept={bool(scene.swept[0])} "
              f"entered={bool(scene.entered[0])} in_pen={bool(scene.in_pen()[0])} "
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

    base = torch.tensor(c.base_pos, device=device)

    def handle_xy() -> torch.Tensor:
        return scene.boom["handle"].data.root_pos_w[0, :2] - scene.env_origins[0, :2]

    def drive_herd(stop_u: float, max_steps: int = 3000):
        """The oracle's herd loop — the same servo law as solve.py (FD yaw rate,
        escalating bias, honest drive cap), then the same park. Returns
        (done, score trace, max pivot drift, max handle distance from the base)."""
        side = float(scene.side[0])
        om_des = side * 0.25
        k_om, bias = 4.0, 0.0
        yaw_prev = float(scene.boom_yaw()[0])
        trace = [float(scene.score()[0])]
        piv = torch.tensor(c.pivot, device=device)
        max_drift, max_hand = 0.0, float((handle_xy() - base).norm())
        done = False
        for i in range(max_steps):
            uw = scene.cube_pen_frame()[0]
            if float(uw[0]) >= stop_u and abs(float(uw[1])) < c.in_pen_w:
                done = True
                break
            yaw = float(scene.boom_yaw()[0])
            om_fd = ((yaw - yaw_prev + math.pi) % (2 * math.pi) - math.pi) / dt
            yaw_prev = yaw
            if side * om_fd < 0.06:
                bias = min(bias + 0.01, 1.0)
            elif side * om_fd > 0.125:
                bias = max(bias - 0.05, 0.0)
            tau = k_om * (om_des - om_fd) + side * bias
            scene.boom_drive[:] = max(-c.drive_cap, min(c.drive_cap, tau))
            step(1)
            if i % 20 == 0:
                trace.append(float(scene.score()[0]))
                spar = scene.boom["spar"]
                yw = float(scene.boom_yaw()[0])
                est = spar.data.root_pos_w[0, :2] - scene.env_origins[0, :2] \
                    - c.spar_off[0] * torch.tensor([math.cos(yw), math.sin(yw)], device=device)
                max_drift = max(max_drift, float((est - piv).norm()))
                max_hand = max(max_hand, float((handle_xy() - base).norm()))
        # park: back the blade out, zero the drive, settle
        yaw_prev = float(scene.boom_yaw()[0])
        for _ in range(110):
            yaw = float(scene.boom_yaw()[0])
            om_fd = ((yaw - yaw_prev + math.pi) % (2 * math.pi) - math.pi) / dt
            yaw_prev = yaw
            scene.boom_drive[:] = max(-c.drive_cap,
                                      min(c.drive_cap, k_om * (-side * 0.22 - om_fd)))
            step(1)
        scene.boom_drive[:] = 0.0
        step(180)
        trace.append(float(scene.score()[0]))
        return done, trace, max_drift, max_hand

    # ========================= 1. settle / clean-slate ========================================
    env.reset(seed=101)
    step(60)
    report("reset")
    bodies = list(scene._bodies().values())
    finite = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies) \
        and bool(torch.isfinite(scene.score()).all())
    r0 = float(scene.cube_rel()[0].norm())
    z0 = float(scene.cube.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    check("settle: clean reset (finite state, cube resting in the sweep annulus, score ~0)",
          finite and c.band_r_min < r0 < c.band_r_max and abs(z0 - c.rest_z) < 0.01
          and float(scene.score()[0]) < 0.02)

    # ========================= 2-4. randomization + side flip + reach ==========================
    draws = []
    min_cube, min_pen = 1e9, 1e9
    max_hand0 = 0.0
    for seed in (11, 12, 13):
        env.reset(seed=seed)
        step(10)
        cube = scene.cube.data.root_pos_w[0, :2] - scene.env_origins[0, :2]
        yaw = math.degrees(float(scene.boom_yaw()[0]))
        pen = scene.pen_org[0]
        draws.append((round(float(cube[0]), 3), round(float(cube[1]), 3),
                      round(yaw, 1), round(float(pen[0]), 3), round(float(pen[1]), 3)))
        min_cube = min(min_cube, float((cube - base).norm()))
        min_pen = min(min_pen, float((pen - base).norm()))
        max_hand0 = max(max_hand0, float((handle_xy() - base).norm()))
    print(f"[smoke] draws (cube_xy, boom_yaw, pen_xy): {draws} | min d(cube,base)="
          f"{min_cube:.3f} min d(pen,base)={min_pen:.3f} max d(handle,base)={max_hand0:.3f}",
          flush=True)
    check("randomization is real (cube spawn / boom yaw / pen pose differ across seeds)",
          len({d[:2] for d in draws}) >= 2 and len({d[2] for d in draws}) >= 2
          and len({d[3:] for d in draws}) >= 2)
    sides = []
    for seed in range(11, 27):
        env.reset(seed=seed)
        sides.append(int(scene.side[0].item()))
        if 1 in sides and -1 in sides:
            break
    print(f"[smoke] side draws across seeds 11..: {sides}", flush=True)
    check("side flip: the pen lands on BOTH sides of the cube across seeds",
          1 in sides and -1 in sides)
    check("reach: cube and pen mouth spawn > 1.0 m from the base; handle starts < 0.75 m",
          min_cube > 1.0 and min_pen > 1.0 and max_hand0 < 0.75)

    # ========================= 5. null policy =================================================
    env.reset(seed=7)
    step(240)
    report("null")
    check("null policy: 2 s of nothing -> score ~0, no success",
          float(scene.score()[0]) < 0.02 and not bool(scene.success()[0]))

    # ========================= 6. negative A: the seed's plan end state ========================
    # pull_cube_tool's goal: the cube parked inside a proximity disc around the base. Here
    # that end state (cube on open floor near the base — never herded, not in the pen)
    # earns nothing.
    env.reset(seed=7)
    step(30)
    teleport_cube(0.25, 0.0, c.rest_z + 0.003)
    step(120)
    report("seed-plan")
    near_base = float((scene.cube.data.root_pos_w[0, :2] - scene.env_origins[0, :2]
                       - base).norm()) < 0.45
    check("negative (seed strategy): cube parked near the base -> rejected, score ~0",
          near_base and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.02)

    # ========================= 7. negative B: teleport into the pen ============================
    env.reset(seed=8)
    step(30)
    pyaw = float(scene.pen_yaw[0])
    pen = scene.pen_org[0]
    cx = float(pen[0]) + 0.095 * math.cos(pyaw)
    cy = float(pen[1]) + 0.095 * math.sin(pyaw)
    teleport_cube(cx, cy, c.rest_z + 0.003)
    step(120)
    report("tp-pen")
    check("anti-cheat: teleport straight into the pen (settled inside) -> rejected, score ~0",
          bool(scene.in_pen()[0]) and not bool(scene.entered[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.02)

    # ========================= 8. negative C: boom exercise without transport ==================
    env.reset(seed=9)
    step(30)
    side = float(scene.side[0])
    yaw_start = float(scene.boom_yaw()[0])
    lo = hi = yaw_start
    yaw_prev = yaw_start
    # Real bidirectional drive, always on the anti-cube side of yaw_start: long AWAY
    # legs (with the solve's stiction-bias escalation so the excursion really lands
    # > 15 deg), and return legs that BREAK 5 deg before yaw_start so the blade can
    # never swing past start and touch the cube.
    for phase in range(4):
        away = phase % 2 == 0
        sgn = -side if away else side
        om_des = sgn * 0.30
        bias = 0.0
        for _ in range(170 if away else 300):
            yaw = float(scene.boom_yaw()[0])
            if not away and side * (yaw_start - yaw) < math.radians(5.0):
                break  # back near start — stop before the cube side
            om_fd = ((yaw - yaw_prev + math.pi) % (2 * math.pi) - math.pi) / dt
            yaw_prev = yaw
            if sgn * om_fd < 0.06:
                bias = min(bias + 0.01, 1.0)
            elif sgn * om_fd > 0.15:
                bias = max(bias - 0.05, 0.0)
            tau = 4.0 * (om_des - om_fd) + sgn * bias
            scene.boom_drive[:] = max(-c.drive_cap, min(c.drive_cap, tau))
            step(1)
            lo, hi = min(lo, yaw), max(hi, yaw)
    scene.boom_drive[:] = 0.0
    step(60)
    report("exercised")
    excur = math.degrees(hi - lo)
    print(f"[smoke] boom excursion during exercise: {excur:.1f} deg", flush=True)
    check("anti-cheat: exercising the boom away from the cube (>= 15 deg real excursion) "
          "earns no credit",
          excur >= 15.0 and not bool(scene.swept[0]) and not bool(scene.entered[0])
          and float(scene.score()[0]) < 0.10 and not bool(scene.success()[0]))

    # ========================= 9-12. oracle herd ===============================================
    env.reset(seed=3)
    step(30)
    s_start = float(scene.score()[0])
    done, trace, drift, max_hand = drive_herd(stop_u=0.070)
    report("herded")
    s_end = float(scene.score()[0])
    check("oracle: the solve's servo law herds the cube through the mouth (swept + entered "
          "+ success)",
          done and bool(scene.swept[0]) and bool(scene.entered[0])
          and bool(scene.success()[0]))
    print(f"[smoke] ladder: {s_start:.3f} -> {trace[len(trace) // 2]:.3f} -> {s_end:.3f}",
          flush=True)
    check("exactness: score == 1.0 exactly at success",
          abs(s_end - 1.0) < 1e-3 and s_start < 0.02)
    mono = all(b >= a - 1e-4 for a, b in zip(trace, trace[1:]))
    print(f"[smoke] herd score trace: "
          f"{' '.join(f'{s:.3f}' for s in trace[::max(1, len(trace) // 12)])}", flush=True)
    check("rubric monotonicity: score never decreases along the herd", mono)
    print(f"[smoke] pivot drift={drift * 1000:.1f} mm, max d(handle,base)={max_hand:.3f} m",
          flush=True)
    check("anchored: pivot drift < 5 mm and the handle stays < 0.75 m from the base "
          "for the whole drive",
          drift < 0.005 and max_hand < 0.75)

    # ========================= 13. near-miss: stop 10 cm short =================================
    env.reset(seed=4)
    step(30)
    done_nm, _tr, _dr, _mh = drive_herd(stop_u=-0.10)
    report("near-miss")
    arc_deg = math.degrees(float(scene.arc_done[0]))
    check("near-miss: real herd stopped ~10 cm short of the mouth -> partial credit only, "
          "no swept latch, no success",
          done_nm and arc_deg > 10.0 and not bool(scene.swept[0])
          and not bool(scene.entered[0]) and not bool(scene.success()[0])
          and 0.03 < float(scene.score()[0]) < 0.29)

    # ========================= 14. persistence (back on the oracle's success) =================
    env.reset(seed=3)
    step(30)
    done2, _tr2, _dr2, _mh2 = drive_herd(stop_u=0.070)
    ok_start = done2 and bool(scene.success()[0])
    flicker = 0
    for _ in range(12):
        step(20)
        if not bool(scene.success()[0]):
            flicker += 1
    report("persist")
    check("persistence: success holds 2 further seconds with no flicker",
          ok_start and flicker == 0)

    # ========================= 15. achievement latch ===========================================
    teleport_cube(0.62, 0.30 * (-float(scene.side[0])), c.rest_z + 0.003)
    step(90)
    report("knock-out")
    check("achievement latch: knock-out revokes success, latched 0.50 remains",
          not bool(scene.success()[0]) and abs(float(scene.score()[0]) - 0.5) < 0.02)

    # ========================= 16. save + verdict ==============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.boom_corral")
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

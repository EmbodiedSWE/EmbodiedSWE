"""Smoke battery for FlywheelInterlockScene — REJECTION tests for the rubric,
NullRobot, RECORDED.

This is NOT the solution (solve.py is — it certifies the rubric ACCEPTS the honest
brake/align/post/restart trajectory). Teleports and direct parcel wrenches here are
rubric INSTRUMENTATION: construct a wrong outcome as a settled state, then assert the
rubric REJECTS it. Every force-driven negative also asserts the actuator/probe REALLY
moved (no vacuous probes).

One linear run, 15 named checks:
  1. settle      — clean reset: finite state, parcel resting on the deck, the wheel
                   verifiably LIVE (FD rate matches the episode draw), score ~0;
  2. random      — across seeds: spin sign flips, magnitude/yaw/parcel spawn vary,
                   wheel readback (rate AND yaw) matches the sampled draw;
  3. null        — 3 s of nothing: the wheel is still coasting fast (it never
                   self-calms), score ~0, no success;
  4. negative A  — the SEED's plan (transport the object to the goal directly):
                   teleport the parcel into the vault while the machine still runs —
                   the success end state, photographically -> rejected, score ~0;
  5. negative B  — honest ARREST first (real brake, calm latch fires), THEN teleport
                   the parcel into the vault: the transit credential is disarmed by
                   the jump and never re-arms inside -> deposited never latches,
                   score stays at the calm credit;
  6. negative C  — the declared SAFETY INTERLOCK: the parcel is force-pushed (it
                   really travels the deck) into the blade sweep while the wheel is
                   running -> permanent foul, score capped at 0.15;
  7. irreversible— from the fouled episode, brake the wheel for real: the calm latch
                   fires (earned) yet the score stays at the foul cap and success
                   stays False — the strike can never be made good;
  8. roof-deny   — parcel dropped from above the vault: it lands ON the roof and
                   stays out — the mouth is the only way in;
  9. closed-gate — wheel honestly calmed but parked with the open sector AWAY from
                   the mouth: the same real push that succeeds in solve.py is
                   physically refused by the blade ring (the parcel really moved,
                   never passes the wall), deposited never latches;
 10. stopped-no  — oracle phases 1-3 for real (brake, align, stage + push through the
                   mouth): deposited latches at 0.50, but a STOPPED machine with the
                   parcel inside is NOT success;
 11. oracle      — real spin-up: respun latches, success() goes live, score == 1.0
                   exactly (ladder 0 -> 0.15 -> 0.50 -> 1.00);
 12. monotone    — the score never decreases along the whole oracle trace;
 13. persist     — success holds 2 further seconds with no flicker;
 14. live-hold   — re-brake the running wheel: success is revoked while the machine
                   is stopped (latched 0.70 floor holds); a real re-spin restores it
                   — the "machine RUNNING" clause is live, not sticky;
 15. frames      — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.approach_grasp_i405.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=20)
parser.add_argument("--max_frames", type=int, default=400)
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
    from .scene import FlywheelInterlockScene  # registers "flywheel_interlock" + env
except ImportError:  # direct-file fallback
    from scene import FlywheelInterlockScene

assert FlywheelInterlockScene is not None


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.flywheel_interlock")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    from isaaclab.utils.math import quat_apply_inverse  # noqa: PLC0415

    no_action = torch.empty(0, device=device)
    one = torch.arange(1, device=device)
    zero3 = torch.zeros(1, 1, 3, device=device)

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.62, -0.70, 0.58)) + o),
                                tuple(np.array((0.00, -0.02, 0.16)) + o),
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

    def wrap(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    def report(tag: str) -> None:
        p = scene._local(scene.parcel)[0].tolist()
        print(f"[smoke] {tag:12s} | th={math.degrees(wrap(float(scene.wheel_yaw()[0]))):+7.1f} "
              f"w={float(scene.wheel_w[0]):+.3f} "
              f"parcel=({p[0]:+.3f},{p[1]:+.3f},{p[2]:.3f}) "
              f"calm={bool(scene.calmed[0])} dep={bool(scene.deposited[0])} "
              f"respun={bool(scene.respun[0])} foul={bool(scene.fouled[0])} "
              f"vault={bool(scene.in_vault()[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def teleport_parcel(x: float, y: float, z: float) -> None:
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = scene.env_origins[0] + torch.tensor([x, y, z], device=device)
        st[0, 3] = 1.0
        scene.parcel.write_root_state_to_sim(st, one)
        env.iscene.update(0.0)

    def parcel_wrench(fx: float, fy: float, fz: float) -> None:
        q = scene.parcel.data.root_quat_w
        f_b = quat_apply_inverse(q, torch.tensor([[fx, fy, fz]], device=device))
        scene.parcel.set_external_force_and_torque(f_b.reshape(1, 1, 3), zero3)

    def brake(max_steps: int = 1500, trace: list | None = None) -> bool:
        """REAL arrest: the solve's brake servo written into the plant buffer."""
        for i in range(max_steps):
            w = float(scene.wheel_w[0])
            scene.wheel_drive[:] = max(-0.27, min(0.27, -0.20 * w))
            step(1)
            if trace is not None and i % 20 == 0:
                trace.append(float(scene.score()[0]))
            if bool(scene.calmed[0]):
                scene.wheel_drive[:] = 0.0
                return True
        scene.wheel_drive[:] = 0.0
        return False

    def park(th_t: float, max_steps: int = 2000, tol: float = 0.06) -> bool:
        """REAL align: jog the stopped wheel until its yaw parks at th_t."""
        ok_n = 0
        for _ in range(max_steps):
            err = wrap(float(scene.wheel_yaw()[0]) - th_t)
            w = float(scene.wheel_w[0])
            scene.wheel_drive[:] = max(-0.22, min(0.22, -0.6 * err - 0.12 * w))
            step(1)
            err = wrap(float(scene.wheel_yaw()[0]) - th_t)
            if abs(err) < tol and abs(float(scene.wheel_w[0])) < 0.10:
                ok_n += 1
                if ok_n >= 30:
                    return True
            else:
                ok_n = 0
        return False

    def hold_at(th_t: float) -> None:
        err = wrap(float(scene.wheel_yaw()[0]) - th_t)
        w = float(scene.wheel_w[0])
        scene.wheel_drive[:] = max(-0.22, min(0.22, -0.6 * err - 0.12 * w))

    def push(max_steps: int, stop, hold_th: float | None = 0.0) -> float:
        """REAL post: the solve's regulated fingertip push. Returns |y travel|."""
        y0 = float(scene._local(scene.parcel)[0, 1])
        for _ in range(max_steps):
            if stop():
                break
            if hold_th is not None:
                hold_at(hold_th)
            p = scene._local(scene.parcel)[0]
            vy = float(scene.parcel.data.root_lin_vel_w[0, 1])
            fy = max(0.0, min(0.55, 6.0 * (0.08 - vy)))
            fx = max(-0.20, min(0.20, -8.0 * float(p[0])))
            parcel_wrench(fx, fy, 0.0)
            step(1)
        parcel_wrench(0.0, 0.0, 0.0)
        scene.wheel_drive[:] = 0.0
        return float(scene._local(scene.parcel)[0, 1]) - y0

    def spin_up(w_t: float = 1.6, max_steps: int = 1500,
                trace: list | None = None) -> bool:
        """REAL restart: the solve's spin-up servo, then hands off."""
        for i in range(max_steps):
            w = float(scene.wheel_w[0])
            scene.wheel_drive[:] = max(-0.27, min(0.27, 0.5 * (w_t - w)))
            step(1)
            if trace is not None and i % 20 == 0:
                trace.append(float(scene.score()[0]))
            if bool(scene.respun[0]) and abs(w) >= c.w_hold:
                break
        scene.wheel_drive[:] = 0.0
        step(30)
        return bool(scene.respun[0])

    # ========================= 1. settle / clean-slate ========================================
    env.reset(seed=101)
    step(30)
    report("reset")
    finite = all(bool(torch.isfinite(b.data.root_state_w).all())
                 for b in (scene.rig, scene.wheel, scene.parcel)) \
        and bool(torch.isfinite(scene.score()).all())
    p = scene._local(scene.parcel)[0]
    w0, w_fd = float(scene.w0[0]), float(scene.wheel_w[0])
    on_deck = (abs(float(p[2]) - c.deck_rest_z) < 0.006
               and c.py_lo - 0.01 < float(p[1]) < c.py_hi + 0.01
               and abs(float(p[0])) < c.px_jit + 0.01)
    live = abs(w_fd - w0) < 0.15 * abs(w0) + 0.10 and abs(w_fd) >= 0.8 * c.w0_lo
    check("settle: clean reset (finite state, parcel resting on the deck, wheel LIVE "
          "at the drawn rate, score ~0, latches clean)",
          finite and on_deck and live and float(scene.score()[0]) < 0.05
          and not bool(scene.calmed[0]) and not bool(scene.fouled[0]))

    # ========================= 2. randomization (spin + yaw + spawn readback) =================
    draws = []
    signs = set()
    readback_ok = True
    for seed in (11, 12, 13, 14, 15, 16, 17, 18):
        env.reset(seed=seed)
        step(6)
        w0 = float(scene.w0[0])
        th0 = float(scene.th0[0])
        w_fd = float(scene.wheel_w[0])
        th_pred = th0 + w0 * 6 / 120.0
        yaw_err = abs(wrap(float(scene.wheel_yaw()[0]) - th_pred))
        p = scene._local(scene.parcel)[0]
        readback_ok = readback_ok and abs(w_fd - w0) < 0.15 * abs(w0) + 0.15 \
            and yaw_err < 0.12
        signs.add(w0 > 0)
        draws.append((round(w0, 2), round(math.degrees(th0), 1),
                      round(float(p[0]), 3), round(float(p[1]), 3)))
    print(f"[smoke] draws (w0, th0_deg, px, py) across seeds: {draws}", flush=True)
    check("randomization is real (spin sign flips, magnitude/yaw/parcel spawn vary, "
          "wheel rate AND yaw readback match the draw)",
          len(signs) == 2 and len(set(draws)) >= 6 and readback_ok)

    # ========================= 3. null policy =================================================
    env.reset(seed=7)
    step(360)
    report("null")
    check("null policy: 3 s of nothing -> the wheel is still coasting fast (never "
          "self-calms), score ~0, no success",
          abs(float(scene.wheel_w[0])) > 1.4 and not bool(scene.calmed[0])
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 4. negative A: the seed's plan (direct transport) ==============
    # approach_grasp's plan: acquire the object and take it to the goal directly,
    # ignoring the machine. Teleport the parcel into the vault (the success photo)
    # while the wheel still runs.
    env.reset(seed=21)
    step(30)
    teleport_parcel(0.0, 0.062, c.vault_rest_z + 0.003)
    step(240)
    report("tp-vault")
    check("negative (seed strategy): parcel teleported into the vault while the "
          "machine still runs -> settled in the vault, rubric rejects it, score ~0",
          bool(scene.in_vault()[0])
          and float(scene.parcel.data.root_lin_vel_w[0].norm()) < c.settle_v
          and abs(float(scene.wheel_w[0])) > 1.4
          and not bool(scene.deposited[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) < 0.05)

    # ========================= 5. negative B: honest calm, then teleport ======================
    env.reset(seed=8)
    step(30)
    calm_ok = brake()
    step(30)
    s_calm = float(scene.score()[0])
    teleport_parcel(0.0, 0.062, c.vault_rest_z + 0.003)
    step(240)
    report("calm+tp")
    check("anti-cheat (transit credential): wheel honestly calmed (latch earned, "
          "score 0.15) then parcel teleported into the vault -> deposited never "
          "latches, score stays at the calm credit",
          calm_ok and abs(s_calm - c.w_calmed) < 1e-3 and bool(scene.in_vault()[0])
          and bool(scene.calmed[0]) and not bool(scene.deposited[0])
          and not bool(scene.success()[0])
          and float(scene.score()[0]) < c.w_calmed + 1e-3)

    # ========================= 6. negative C: the declared safety interlock ===================
    env.reset(seed=9)
    step(30)
    y_start = float(scene._local(scene.parcel)[0, 1])
    travel = push(900, lambda: bool(scene.fouled[0]), hold_th=None)
    step(60)
    report("blade-strike")
    check("negative (interlock): parcel force-pushed across the deck into the blade "
          "sweep while the wheel runs (it really travelled) -> permanent foul, score "
          "capped at 0.15",
          travel > 0.04 and y_start < -0.10 and bool(scene.fouled[0])
          and not bool(scene.success()[0])
          and float(scene.score()[0]) <= c.foul_cap + 1e-3)

    # ========================= 7. the foul is irreversible ====================================
    calm_ok2 = brake()
    step(60)
    report("too-late")
    check("irreversibility: after the strike, a REAL brake earns the calm latch, yet "
          "the score stays at the foul cap and success stays False",
          calm_ok2 and bool(scene.calmed[0]) and bool(scene.fouled[0])
          and not bool(scene.success()[0])
          and float(scene.score()[0]) <= c.foul_cap + 1e-3)

    # ========================= 8. the roof denies entry from above ============================
    env.reset(seed=31)
    step(30)
    roof_top = c.roof_z0 + c.roof_t
    teleport_parcel(0.0, 0.062, roof_top + c.cube / 2 + 0.030)
    step(240)
    report("roof-deny")
    p = scene._local(scene.parcel)[0]
    check("containment: parcel dropped from above the vault lands ON the roof and "
          "stays out — the mouth is the only way in",
          abs(float(p[2]) - (roof_top + c.cube / 2)) < 0.012
          and not bool(scene.in_vault()[0]) and not bool(scene.deposited[0])
          and float(scene.score()[0]) < 0.05)

    # ========================= 9. a closed gate refuses the post ==============================
    # Honest calm, but park the open sector AWAY from the mouth (plates wall it off),
    # then run the SAME push that succeeds in solve.py.
    env.reset(seed=5)
    step(30)
    # (park at +90 deg: the ring covers the mouth to +-32 deg, and the yellow handle
    # peg swings to the side, clear of the deck lane — the refusal is the BLADE ring.)
    calm_ok3 = brake()
    park_ok = park(math.pi / 2)
    teleport_parcel(0.0, -0.075, c.deck_rest_z + 0.004)  # legal transport staging
    step(30)
    travel = push(900, lambda: float(scene._local(scene.parcel)[0, 1]) > 0.0,
                  hold_th=math.pi / 2)
    step(120)
    report("closed-gate")
    p = scene._local(scene.parcel)[0]
    check("order-forcing: with the wheel calmed but parked open-sector-AWAY, the "
          "same real push is refused by the blade ring (parcel really moved, never "
          "passes the wall), deposited never latches, no foul (wheel stopped)",
          calm_ok3 and park_ok and travel > 0.025 and float(p[1]) < 0.0
          and not bool(scene.deposited[0]) and not bool(scene.fouled[0])
          and float(scene.score()[0]) < c.w_calmed + 1e-3)

    # ========================= 10-12. oracle: full honest run =================================
    env.reset(seed=3)
    step(30)
    trace: list[float] = [float(scene.score()[0])]
    s_start = trace[0]
    ok_calm = brake(trace=trace)
    s_calm = float(scene.score()[0])
    ok_park = park(0.0)
    trace.append(float(scene.score()[0]))
    teleport_parcel(0.0, -0.075, c.deck_rest_z + 0.004)
    step(30)
    push(1500, lambda: (float(scene._local(scene.parcel)[0, 1]) > 0.050
                        or float(scene._local(scene.parcel)[0, 2])
                        < c.deck_rest_z - 0.020))
    for _ in range(240):
        hold_at(0.0)
        step(1)
    scene.wheel_drive[:] = 0.0
    report("posted")
    s_dep = float(scene.score()[0])
    trace.append(s_dep)
    check("kinetic goal: deposited latches at 0.50 after the real post, but a "
          "STOPPED machine with the parcel inside is NOT success",
          ok_calm and ok_park and bool(scene.deposited[0])
          and abs(s_dep - (c.w_calmed + c.w_dep)) < 1e-3
          and abs(float(scene.wheel_w[0])) < c.w_calm
          and not bool(scene.success()[0]))

    ok_spin = spin_up(trace=trace)
    report("restarted")
    s_end = float(scene.score()[0])
    trace.append(s_end)
    ladder = [s_start, s_calm, s_dep, s_end]
    print(f"[smoke] ladder: {' -> '.join(f'{s:.3f}' for s in ladder)}", flush=True)
    check("oracle: real brake/align/post/restart earns every latch, success() goes "
          "live, score == 1.0 exactly",
          ok_spin and bool(scene.success()[0]) and abs(s_end - 1.0) < 1e-3
          and not bool(scene.fouled[0]))
    mono = all(b >= a - 1e-4 for a, b in zip(trace, trace[1:]))
    print(f"[smoke] oracle score trace: "
          f"{' '.join(f'{s:.3f}' for s in trace[::max(1, len(trace) // 12)])}", flush=True)
    check("rubric monotonicity: score never decreases along the oracle run (ladder "
          "0 -> 0.15 -> 0.50 -> 1.00)",
          mono and all(b >= a - 1e-4 for a, b in zip(ladder, ladder[1:])))

    # ========================= 13. persistence ================================================
    flicker = 0
    for _ in range(12):
        step(20)
        if not bool(scene.success()[0]):
            flicker += 1
    report("persist")
    check("persistence: success holds 2 further hands-off seconds with no flicker",
          flicker == 0 and bool(scene.success()[0]))

    # ========================= 14. the RUNNING clause is live =================================
    revoked = False
    floor_held = True
    for _ in range(1200):
        w = float(scene.wheel_w[0])
        scene.wheel_drive[:] = max(-0.27, min(0.27, -0.20 * w))
        step(1)
        s_now = float(scene.score()[0])
        floor_held = floor_held and s_now >= c.w_calmed + c.w_dep + c.w_respun - 1e-3
        if abs(float(scene.wheel_w[0])) < 0.5 * c.w_hold and not bool(scene.success()[0]):
            revoked = True
        if abs(float(scene.wheel_w[0])) < 0.05:
            break
    scene.wheel_drive[:] = 0.0
    step(60)
    report("re-braked")
    stopped_no = not bool(scene.success()[0])
    ok_respin = spin_up()
    report("re-spun")
    check("live RUNNING clause: braking the wheel after success revokes it (latched "
          "0.70 floor holds); a real re-spin restores success at 1.0",
          revoked and floor_held and stopped_no and ok_respin
          and bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 1.0) < 1e-3)

    # ========================= 15. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.flywheel_interlock")
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

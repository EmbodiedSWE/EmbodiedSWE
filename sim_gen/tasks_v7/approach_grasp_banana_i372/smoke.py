"""Smoke battery for CrankEjectorScene — REJECTION tests for the rubric, NullRobot,
RECORDED.

This is NOT the solution (solve.py is — it certifies the rubric ACCEPTS the honest
crank-driven ejection). Teleports and direct cube wrenches here are rubric
INSTRUMENTATION: construct a wrong outcome as a settled state, then assert the rubric
REJECTS it. Every force-driven negative also asserts the actuator/probe REALLY moved
(no vacuous probes).

One linear run, 15 named checks:
  1. settle      — clean reset: finite state, both cubes resting deep in the tunnel,
                   score ~0;
  2. random      — cargo side flips across seeds, crank angle / cube gaps vary; cube
                   readback matches the sampled side;
  3. null        — 2 s of nothing: score ~0, no success;
  4. negative A  — the SEED's plan (transport the object to the goal directly): the
                   cargo is force-dragged out of the mouth and drops into its pocket
                   WITHOUT the crank ever turning — physically delivered, but the
                   crank-path latches never fired -> rejected, score ~0;
  5. negative B  — teleport the cargo straight into the goal pocket (the success end
                   state, photographically) -> rejected, score ~0;
  6. negative C  — REAL crank in the WRONG direction (crank verifiably turned): the
                   grey decoy is ejected -> permanent foul, score capped at 0.20;
  7. irreversible— from the fouled episode, crank the CORRECT way for real: the cargo
                   is genuinely delivered (seated latch fires) yet success stays False
                   and the score stays at the foul cap — the wrong stroke can never be
                   made good;
  8. slot-deny   — a bounded lift wrench raises the cargo (it really moves) but the
                   roof shelves keep it in the tunnel: the cube cannot leave upward;
  9. oracle      — real servo crank toward the cargo side: ejected + seated latch,
                   score 0.70 while the crank is still moving;
 10. monotone    — the score never decreases along the oracle drive;
 11. exactness   — release the crank: everything settles -> success() and
                   score == 1.0 exactly (ladder 0 -> 0.45 -> 0.70 -> 1.00);
 12. persist     — success holds 2 further seconds with no flicker;
 13. latch       — re-spin the crank (real reverse servo): success revoked while the
                   mechanism moves, latched 0.70 floor holds; brake + settle: success
                   returns (live, not sticky);
 14. z-band      — cargo teleported to pocket-wall-top height: success drops (the seat
                   z band rejects a perch), falls back to the pocket floor: success
                   returns;
 15. frames      — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.approach_grasp_banana_i372.smoke --headless
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
    from .scene import CrankEjectorScene  # registers "crank_ejector" + env
except ImportError:  # direct-file fallback
    from scene import CrankEjectorScene

assert CrankEjectorScene is not None


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.crank_ejector")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
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
        env.sim.set_camera_view(tuple(np.array((0.60, 0.75, 0.58)) + o),
                                tuple(np.array((0.00, 0.00, 0.10)) + o),
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
        pc = scene._local(scene.cargo)[0].tolist()
        pd = scene._local(scene.decoy)[0].tolist()
        print(f"[smoke] {tag:12s} | th={math.degrees(float(scene.crank_theta()[0])):6.1f} "
              f"ram_x={float(scene.ram_x()[0]):+.3f} side={float(scene.side[0]):+.0f} "
              f"cargo=({pc[0]:+.3f},{pc[1]:+.3f},{pc[2]:.3f}) "
              f"decoy=({pd[0]:+.3f},{pd[2]:.3f}) prog={float(scene.prog[0]):.2f} "
              f"ej={bool(scene.ejected[0])} seat={bool(scene.seated[0])} "
              f"foul={bool(scene.fouled[0])} goal={bool(scene.in_goal_pocket()[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def teleport_cargo(x: float, y: float, z: float) -> None:
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = scene.env_origins[0] + torch.tensor([x, y, z], device=device)
        st[0, 3] = 1.0
        scene.cargo.write_root_state_to_sim(st, one)
        env.iscene.update(0.0)

    def cargo_wrench(fx: float, fz: float) -> None:
        f = torch.zeros(1, 1, 3, device=device)
        f[0, 0, 0] = fx
        f[0, 0, 2] = fz
        scene.cargo.set_external_force_and_torque(f, zero3)

    def crank(direction: float, stop, max_steps: int, trace: list | None = None,
              w_des: float = 1.2):
        """Real crank drive: the solve's stall-proof velocity servo written into the
        scene's plant buffer. Returns (stopped, max_|dtheta|_deg)."""
        k, ff = 0.25, 0.06
        cap = 0.9 * c.drive_max
        th0 = float(scene.crank_theta()[0])
        th_mark, i_mark = th0, 0
        sweep = 0.0
        done = False
        for i in range(max_steps):
            if stop():
                done = True
                break
            w = float(scene.crank_w[0])
            tau = direction * ff + k * (direction * w_des - w)
            if abs(w) > 2.5 * w_des:
                tau = 0.0
            scene.crank_drive[:] = max(-cap, min(cap, tau))
            step(1)
            sweep = max(sweep, abs(math.degrees(float(scene.crank_theta()[0]) - th0)))
            if trace is not None and i % 20 == 0:
                trace.append(float(scene.score()[0]))
            if i - i_mark >= 240:  # stall escalation (identical to solve)
                th = float(scene.crank_theta()[0])
                if direction * (th - th_mark) < math.radians(2.0):
                    ff = min(ff * 1.6, 0.40)
                    k = min(k * 1.3, 0.60)
                th_mark, i_mark = th, i
        scene.crank_drive[:] = 0.0
        return done, sweep

    def brake(max_steps: int = 240) -> None:
        """Actively kill the crank rate (it coasts otherwise), then zero the drive."""
        for _ in range(max_steps):
            w = float(scene.crank_w[0])
            if abs(w) < 0.05:
                break
            scene.crank_drive[:] = max(-0.3, min(0.3, 0.3 * (0.0 - w)))
            step(1)
        scene.crank_drive[:] = 0.0

    # ========================= 1. settle / clean-slate ========================================
    env.reset(seed=101)
    step(60)
    report("reset")
    finite = all(bool(torch.isfinite(b.data.root_state_w).all())
                 for b in (scene.rig, scene.ram, scene.crank, scene.cargo, scene.decoy)) \
        and bool(torch.isfinite(scene.score()).all())
    pc = scene._local(scene.cargo)[0]
    pd = scene._local(scene.decoy)[0]
    in_tunnel = (abs(float(pc[0])) < c.tunnel_hl - 0.02
                 and abs(float(pc[2]) - c.cube_rest_z) < 0.008
                 and abs(float(pd[0])) < c.tunnel_hl - 0.02
                 and abs(float(pd[2]) - c.cube_rest_z) < 0.008)
    check("settle: clean reset (finite state, both cubes resting deep in the tunnel, "
          "score ~0)", finite and in_tunnel and float(scene.score()[0]) < 0.05)

    # ========================= 2. randomization (side + pose, cube readback) ==================
    draws = []
    sides = set()
    readback_ok = True
    for seed in (11, 12, 13, 14, 15, 16, 17, 18):
        env.reset(seed=seed)
        step(10)
        sd = float(scene.side[0])
        sides.add(sd)
        th = round(math.degrees(float(scene.crank_theta()[0])), 1)
        cx = round(float(scene._local(scene.cargo)[0, 0]), 3)
        dx = round(float(scene._local(scene.decoy)[0, 0]), 3)
        readback_ok = readback_ok and (sd * cx > 0) and (sd * dx < 0)
        draws.append((int(sd), th, cx, dx))
    print(f"[smoke] draws (side, theta0_deg, cargo_x, decoy_x) across seeds: {draws}",
          flush=True)
    check("randomization is real (cargo side flips, crank angle/cube gaps vary, cube "
          "positions match the sampled side)",
          len(sides) == 2 and len({d[1:] for d in draws}) >= 4 and readback_ok)

    # ========================= 3. null policy =================================================
    env.reset(seed=7)
    step(240)
    report("null")
    check("null policy: 2 s of nothing -> score ~0, no success",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 4. negative A: the seed's plan (direct transport) ==============
    # approach_grasp_banana's plan: take the object to the goal directly. Here the cargo
    # is force-dragged (regulated wrench, ~0.25 m/s) out of its mouth and over the drop
    # lip — it REALLY lands in the goal pocket, but the crank never turned.
    env.reset(seed=21)
    step(30)
    sd = float(scene.side[0])
    x_start = float(scene._local(scene.cargo)[0, 0])
    th_a = float(scene.crank_theta()[0])
    for _ in range(1200):
        p = scene._local(scene.cargo)[0]
        if sd * float(p[0]) > c.tunnel_hl + 0.008:  # half over the lip: let gravity finish
            break
        vx = float(scene.cargo.data.root_lin_vel_w[0, 0])
        cargo_wrench(max(-0.6, min(0.6, 2.0 * (sd * 0.25 - vx))), 0.0)
        step(1)
    cargo_wrench(0.0, 0.0)
    step(360)
    report("drag-out")
    moved = sd * (float(scene._local(scene.cargo)[0, 0]) - x_start) > 0.05
    crank_still = abs(math.degrees(float(scene.crank_theta()[0]) - th_a)) < 2.0
    check("negative (seed strategy): cargo force-dragged into the goal pocket without "
          "the crank turning -> physically delivered, rubric rejects it, score ~0",
          moved and crank_still and bool(scene.in_goal_pocket()[0])
          and float(scene.cargo.data.root_lin_vel_w[0].norm()) < c.settle_v
          and not bool(scene.ejected[0]) and not bool(scene.seated[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.05)

    # ========================= 5. negative B: teleport into the pocket ========================
    env.reset(seed=8)
    step(30)
    sd = float(scene.side[0])
    teleport_cargo(sd * (c.tunnel_hl + 0.055), 0.0, c.pocket_rest_z + 0.003)
    step(240)
    report("tp-pocket")
    check("anti-cheat: cargo teleported straight into the goal pocket (settled) -> "
          "rejected, score ~0",
          bool(scene.in_goal_pocket()[0])
          and float(scene.cargo.data.root_lin_vel_w[0].norm()) < c.settle_v
          and not bool(scene.seated[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) < 0.05)

    # ========================= 6. negative C: REAL crank the WRONG way ========================
    env.reset(seed=9)
    step(30)
    sd = float(scene.side[0])
    ok_w, sweep_w = crank(-sd, lambda: bool(scene.fouled[0]), 2400)
    brake()
    step(240)
    report("wrong-way")
    pd = scene._local(scene.decoy)[0]
    decoy_out = (-sd * float(pd[0]) > c.eject_x) or (float(pd[2]) < c.eject_z)
    check("negative (wrong direction): crank verifiably turned the other way, decoy "
          "ejected -> permanent foul, score capped at 0.20",
          ok_w and sweep_w > 30.0 and decoy_out and bool(scene.fouled[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= c.foul_cap + 1e-3)

    # ========================= 7. the foul is irreversible ====================================
    ok_c, _sw = crank(sd, lambda: bool(scene.seated[0]), 3600)
    brake()
    step(300)
    report("too-late")
    check("irreversibility: after the foul, a REAL correct stroke delivers the cargo "
          "(seated latch fires) yet success stays False and the score stays capped",
          ok_c and bool(scene.seated[0]) and bool(scene.in_goal_pocket()[0])
          and bool(scene.fouled[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= c.foul_cap + 1e-3)

    # ========================= 8. the roof slot denies a lift-out =============================
    env.reset(seed=31)
    step(30)
    z0 = float(scene._local(scene.cargo)[0, 2])
    z_max = z0
    for _ in range(150):
        cargo_wrench(0.0, 1.0)  # ~2x the cube's weight, straight up
        step(1)
        z_max = max(z_max, float(scene._local(scene.cargo)[0, 2]))
    cargo_wrench(0.0, 0.0)
    step(180)
    report("slot-deny")
    pc = scene._local(scene.cargo)[0]
    check("containment: a 2x-weight lift wrench raises the cube (it really moved) but "
          "the roof keeps it inside the tunnel — no upward exit",
          z_max - z0 > 0.003 and z_max < c.roof_lo - c.cube / 2 + 0.006
          and abs(float(pc[0])) < c.tunnel_hl and abs(float(pc[2]) - c.cube_rest_z) < 0.01
          and float(scene.score()[0]) < 0.05)

    # ========================= 9-10. oracle crank + monotone trace ============================
    env.reset(seed=3)
    step(30)
    sd = float(scene.side[0])
    s_start = float(scene.score()[0])
    trace: list[float] = [s_start]
    ok_e, sweep_o = crank(sd, lambda: bool(scene.ejected[0]), 2400, trace)
    s_ej = float(scene.score()[0])
    ok_s, _sw2 = crank(sd, lambda: bool(scene.seated[0]), 1200, trace)
    report("oracle-seat")
    s_seat = float(scene.score()[0])
    check("oracle: real servo crank toward the cargo side latches ejected + seated, "
          "score 0.70 while the crank is still live",
          ok_e and ok_s and sweep_o > 20.0 and bool(scene.ejected[0])
          and bool(scene.seated[0]) and not bool(scene.fouled[0])
          and 0.44 <= s_ej <= 0.46 and 0.695 <= s_seat <= 0.705)
    trace.append(s_seat)
    mono = all(b >= a - 1e-4 for a, b in zip(trace, trace[1:]))
    print(f"[smoke] oracle score trace: "
          f"{' '.join(f'{s:.3f}' for s in trace[::max(1, len(trace) // 12)])}", flush=True)
    check("rubric monotonicity: score never decreases along the oracle drive", mono)

    # ========================= 11. exactness on release =======================================
    step(300)  # release: crank settles, cargo rests, decoy never moved
    report("released")
    ladder = [s_start, s_ej, s_seat, float(scene.score()[0])]
    print(f"[smoke] ladder: {' -> '.join(f'{s:.3f}' for s in ladder)}", flush=True)
    check("exactness: release -> success() and score == 1.0 exactly (ladder monotone)",
          bool(scene.success()[0]) and abs(float(scene.score()[0]) - 1.0) < 1e-3
          and all(b >= a - 1e-4 for a, b in zip(ladder, ladder[1:])))

    # ========================= 12. persistence ================================================
    flicker = 0
    for _ in range(12):
        step(20)
        if not bool(scene.success()[0]):
            flicker += 1
    report("persist")
    check("persistence: success holds 2 further seconds with no flicker", flicker == 0)

    # ========================= 13. achievement latch (disturb + recover) ======================
    # Re-spin the crank a few degrees (REAL reverse servo, small angle — the ram backs
    # off ~2 mm, nowhere near the decoy): while the mechanism moves success must be
    # revoked and the latched 0.70 floor must hold; brake + settle: success returns.
    th_stop = float(scene.crank_theta()[0])
    revoked = False
    floor_held = True
    for _ in range(600):
        if sd * math.degrees(th_stop - float(scene.crank_theta()[0])) > 10.0:
            break
        w = float(scene.crank_w[0])
        scene.crank_drive[:] = max(-0.5, min(0.5, -sd * 0.06 + 0.25 * (-sd * 1.2 - w)))
        step(1)
        s_now = float(scene.score()[0])
        floor_held = floor_held and s_now >= 0.695
        if abs(float(scene.crank_w[0])) > c.settle_w and not bool(scene.success()[0]):
            revoked = True
    scene.crank_drive[:] = 0.0
    report("disturbed")
    brake()
    step(300)
    report("recovered")
    check("achievement latch: a moving crank revokes success (latched 0.70 floor "
          "holds); braking + settling restores it (success is live)",
          revoked and floor_held and bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 1.0) < 1e-3)

    # ========================= 14. seat z band rejects a perch ================================
    p_seat = scene._local(scene.cargo)[0].tolist()
    teleport_cargo(sd * (c.tunnel_hl + 0.055), 0.0, c.pk_wall_top + c.cube / 2 + 0.002)
    perch_denied = (not bool(scene.in_goal_pocket()[0])) and (not bool(scene.success()[0])) \
        and float(scene.score()[0]) >= 0.695
    step(240)  # it falls back onto the pocket floor and settles
    report("z-band")
    check("seat z band: cargo raised to pocket-wall-top height is NOT in the goal "
          "(success revoked, floor 0.70 holds); falling back restores success",
          perch_denied and bool(scene.in_goal_pocket()[0]) and bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 1.0) < 1e-3
          and abs(p_seat[2] - c.pocket_rest_z) < 0.01)

    # ========================= 15. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.crank_ejector")
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

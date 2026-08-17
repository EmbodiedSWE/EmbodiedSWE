"""Smoke battery for TwistlockUnplugScene — REJECTION tests for the rubric, NullRobot,
RECORDED.

This is NOT a solution (the solution is solve.py — twist the LAMP plug to the sampled
slot angle with a rate-cascade torque servo, then a delay-stable velocity-servo pull;
the Franka strategy is TASK.md's embodiment argument). Drives here are the same
fingertip-scale numbers as solve.py (shared numbers = shared honesty); teleports are
never used to construct outcomes — every probe flows through the live D6 + collision
plant.

One linear run, 14 named checks:
  1. settle    — clean reset: finite state on every body, both plugs seated and
                 untwisted, score ~0;
  2. readback  — the faceplate boxes physically sit at the SAMPLED slot angles
                 (world-pose readback vs the cfg formula, both sockets);
  3. readback  — identification channel: the red cord's riser sits under the LAMP's
                 socket, the blue cord's under the fan's; cords start at their
                 appliance; lamp and fan on opposite y bands;
  4. random    — socket assignment, slot angles (magnitude AND sign) and appliance
                 positions vary across seeds (READBACK, 8 seeds);
  5. null      — 2 s of nothing: score < 0.05, no success;
  6. negative  — the interlock is physical: an UNTWISTED 1.5 N yank moves the plug
                 through the 3 mm free travel (probe moved — non-vacuous) and JAMS on
                 the plate; settled ext < ext_pass, pass_latch still 0, score ~0;
  7. negative  — alignment must be RIGHT: twisted 25 deg OFF the slot, the same yank
                 still jams (rotation alone is not the key — the ANGLE is);
  8. negative  — the SEED's plan (act on the lamp itself): a 15 N shove slides the
                 lamp (probe moved — non-vacuous) and earns ~0 — the goal lives at
                 the outlet, not on the lamp;
  9. negative  — near miss: correctly aligned pull LANDED at ~30 mm (< ext_goal) —
                 partial credit only, no success;
 10. negative  — wrong plug: the full twist+pull strategy executed on the FAN's plug
                 — decoy unseated, ~0 score, no success;
 11. negative  — keep-alive: SAME episode, lamp plug then ALSO pulled correctly —
                 lamp is out, but the fan was unplugged: rejected, credit quartered;
 12. exactness — full correct strategy -> success() and score == 1.0;
 13. latch     — pushing the lamp plug back IN revokes success (success is live
                 state); the latched 0.70 of earned credit remains;
 14. frames    — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.lamp_off_i101.smoke --headless
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
    from simgen_tasks.lamp_off_i101 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie.
threading.Timer(1200.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                 os._exit(3))).start()

# Same fingertip-scale drives as solve.py (shared numbers = shared honesty).
K_ANG = 4.0
W_CAP_TW = 1.0
KW = 3e-4
TAU_MAX = 0.02
ALIGN_DONE = math.radians(3.0)
W_DONE = 0.3
KV = 2.0
V_CAP = 0.06
K_APP = 3.0
F_MAX = 6.0
V_DONE = 0.02
YANK_F = 1.5  # N — the untwisted/mis-twisted pull probe (~3x the servo's working force)
SHOVE_F = 15.0  # N — the seed-style lateral shove on the lamp (must beat ~9.4 N friction)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.twistlock_unplug")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.62, -0.55, 0.95)) + o),
                                tuple(np.array((-0.22, 0.0, 0.48)) + o),
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
        ls = int(scene.lamp_socket[0])
        tw = math.degrees(float(scene.plug_twist()[0, ls]))
        print(f"[smoke] {tag:14s} | twist={tw:+6.1f}deg "
              f"ext={float(scene.lamp_ext()[0]) * 1000:6.1f}mm "
              f"decoy={float(scene.decoy_ext()[0]) * 1000:5.1f}mm "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    # --- drive helpers (solve.py's servos, socket-generic) ---
    def servo_twist(s: int, alpha: float, budget: int = 1200) -> bool:
        for _ in range(budget):
            tw = float(scene.plug_twist()[0, s])
            w = float(scene.plugs[s].data.root_ang_vel_w[0, 0])
            if abs(alpha - tw) < ALIGN_DONE and abs(w) < W_DONE:
                scene.drive_t[0, s] = 0.0
                return True
            w_des = max(-W_CAP_TW, min(W_CAP_TW, K_ANG * (alpha - tw)))
            scene.drive_t[0, s] = max(-TAU_MAX, min(TAU_MAX, KW * (w_des - w)))
            step(1)
        scene.drive_t[0, s] = 0.0
        return False

    def servo_to(s: int, x_des: float, alpha: float, budget: int = 2400,
                 tol: float = 0.004) -> bool:
        """Velocity-servo the plug to extraction `x_des`, twist held at `alpha`;
        releases only a SLOW plug. Works both directions (pull out / push back)."""
        ok = False
        for _ in range(budget):
            ext = float(scene.plug_ext()[0, s])
            v = float(scene.plugs[s].data.root_lin_vel_w[0, 0])
            if abs(ext - x_des) <= tol and abs(v) < V_DONE:
                ok = True
                break
            v_des = max(-V_CAP, min(V_CAP, K_APP * (x_des - ext)))
            scene.drive_f[0, s] = max(-F_MAX, min(F_MAX, KV * (v_des - v)))
            scene.drive_t[0, s] = twist_hold(s, alpha)
            step(1)
        scene.drive_f[0, s] = 0.0
        scene.drive_t[0, s] = 0.0
        return ok

    def twist_hold(s: int, alpha: float) -> float:
        tw = float(scene.plug_twist()[0, s])
        w = float(scene.plugs[s].data.root_ang_vel_w[0, 0])
        w_des = max(-W_CAP_TW, min(W_CAP_TW, K_ANG * (alpha - tw)))
        return max(-TAU_MAX, min(TAU_MAX, KW * (w_des - w)))

    def yank(s: int, secs: float = 2.0) -> float:
        """Constant outward YANK_F on plug `s` (twist untouched); returns max ext seen."""
        mx = 0.0
        for _ in range(int(secs * 120)):
            scene.drive_f[0, s] = YANK_F
            step(1)
            mx = max(mx, float(scene.plug_ext()[0, s]))
        scene.drive_f[0, s] = 0.0
        step(120)  # judge the settled aftermath, not the impact bounce
        return mx

    def finite_all() -> bool:
        return all(bool(torch.isfinite(b.data.root_state_w).all())
                   for b in scene._bodies().values()) \
            and bool(torch.isfinite(scene.score()).all())

    # ========================= 1. settle / clean-slate ========================================
    torch.manual_seed(3)
    env.reset()
    step(60)
    report("reset")
    ext0 = scene.plug_ext()[0]
    tw0 = scene.plug_twist()[0]
    check("settle: clean reset (finite, both plugs seated & untwisted, score ~0)",
          finite_all()
          and float(ext0.abs().max()) < 0.002 and float(tw0.abs().max()) < math.radians(2.0)
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 2. faceplate pose readback =====================================
    plate_err = 0.0
    for s in (0, 1):
        a = math.radians(float(scene.slot_deg[0, s]))
        ca, sa = math.cos(a), math.sin(a)
        for k, (loc, _size) in enumerate(c.plate_box_locals()):
            want = torch.tensor([c.plate_cx + loc[0],
                                 c.socket_y[s] + ca * loc[1] - sa * loc[2],
                                 c.socket_z + sa * loc[1] + ca * loc[2]], device=device)
            got = scene.plates[s][k].data.root_pos_w[0] - scene.env_origins[0]
            plate_err = max(plate_err, float((got - want).norm()))
    print(f"[smoke] slot angles {[round(float(x), 1) for x in scene.slot_deg[0]]} deg, "
          f"plate readback err={plate_err * 1000:.2f}mm", flush=True)
    check("readback: faceplates physically posed at the sampled slot angles",
          plate_err < 0.0015)

    # ========================= 3. cord routing + appliance bands ==============================
    ls = int(scene.lamp_socket[0])
    lamp_y = float((scene.lamp.data.root_pos_w[0] - scene.env_origins[0])[1])
    fan_y = float((scene.fan.data.root_pos_w[0] - scene.env_origins[0])[1])
    cord_ok = True
    for nm, body, sock in (("cord_lamp", scene.lamp, ls), ("cord_fan", scene.fan, 1 - ls)):
        riser = scene.cords[nm][c.n_cord_segs].data.root_pos_w[0] - scene.env_origins[0]
        first = scene.cords[nm][0].data.root_pos_w[0] - scene.env_origins[0]
        base = body.data.root_pos_w[0] - scene.env_origins[0]
        cord_ok &= abs(float(riser[1]) - c.socket_y[sock]) < 0.002
        cord_ok &= float((first[:2] - base[:2]).norm()) < 0.15
    check("readback: cords route lamp->its socket / fan->the other; bands opposite",
          cord_ok and lamp_y * fan_y < 0
          and 0.10 <= abs(lamp_y) <= 0.34 and 0.10 <= abs(fan_y) <= 0.34)

    # ========================= 4. randomization across seeds ==================================
    draws = []
    for seed in (11, 12, 13, 14, 15, 16, 17, 18):
        torch.manual_seed(seed)
        env.reset()
        step(2)
        lp = scene.lamp.data.root_pos_w[0] - scene.env_origins[0]
        draws.append((int(scene.lamp_socket[0]),
                      round(float(scene.slot_deg[0, 0]), 1),
                      round(float(scene.slot_deg[0, 1]), 1),
                      round(float(lp[0]), 3), round(float(lp[1]), 3)))
    print(f"[smoke] draws (socket, slot0, slot1, lamp xy): {draws}", flush=True)
    slots = [d[1] for d in draws] + [d[2] for d in draws]
    check("randomization is real (socket / slot sign+magnitude / positions vary)",
          {d[0] for d in draws} == {0, 1}
          and len({d[1] for d in draws}) >= 4
          and any(v < 0 for v in slots) and any(v > 0 for v in slots)
          and len({(d[3], d[4]) for d in draws}) >= 4)

    # ========================= 5. null policy =================================================
    torch.manual_seed(3)
    env.reset()
    step(240)  # 2 s of nothing
    report("null")
    check("null policy: score < 0.05 and no success",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 6. negative: locked pull jams (physically) =====================
    torch.manual_seed(3)
    env.reset()
    step(30)
    ls = int(scene.lamp_socket[0])
    mx = yank(ls)
    report("locked-yank")
    ext_now = float(scene.lamp_ext()[0])
    check("negative (locked pull): untwisted yank moves 3 mm then JAMS on the plate",
          mx >= 0.0015 and ext_now < c.ext_pass and mx < c.ext_pass + 0.004
          and float(scene.pass_latch[0]) == 0.0
          and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.10)

    # ========================= 7. negative: 25 deg off the slot still jams ====================
    torch.manual_seed(3)
    env.reset()
    step(30)
    ls = int(scene.lamp_socket[0])
    alpha = math.radians(float(scene.slot_deg[0, ls]))
    off = alpha - math.copysign(math.radians(25.0), alpha)  # 25 deg misaligned, within limits
    ok_off = servo_twist(ls, off)
    mx = yank(ls)
    report("mis-twist")
    check("negative (mis-twist): rotated but 25 deg OFF the slot — still jams, no pass",
          ok_off and mx < c.ext_pass + 0.004 and float(scene.lamp_ext()[0]) < c.ext_pass
          and float(scene.pass_latch[0]) == 0.0
          and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.35)

    # ========================= 8. negative: the SEED's plan (act on the lamp) =================
    torch.manual_seed(3)
    env.reset()
    step(30)
    p0 = (scene.lamp.data.root_pos_w[0] - scene.env_origins[0]).clone()
    push_dir = -1.0 if float(p0[1]) > 0 else 1.0
    f = torch.zeros(n, 1, 3, device=device)
    f[:, 0, 1] = push_dir * SHOVE_F
    for _ in range(60):  # 0.5 s shove
        scene.lamp.set_external_force_and_torque(f, torch.zeros_like(f))
        step(1)
    scene.lamp.set_external_force_and_torque(torch.zeros_like(f), torch.zeros_like(f))
    step(180)
    disp = float(((scene.lamp.data.root_pos_w[0] - scene.env_origins[0]) - p0)[:2].norm())
    report("lamp-shove")
    check("negative (seed strategy): shoving the LAMP moves it but earns nothing",
          disp > 0.010 and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.05)

    # ========================= 9. negative: near miss (30 mm < ext_goal) ======================
    torch.manual_seed(3)
    env.reset()
    step(30)
    ls = int(scene.lamp_socket[0])
    alpha = math.radians(float(scene.slot_deg[0, ls]))
    ok_tw = servo_twist(ls, alpha)
    ok_pl = servo_to(ls, 0.030, alpha)
    step(240)
    report("near-miss")
    e = float(scene.lamp_ext()[0])
    check("negative (near miss): aligned pull landed at ~30 mm — partial credit, no success",
          ok_tw and ok_pl and 0.024 < e < c.ext_goal
          and not bool(scene.success()[0])
          and 0.30 < float(scene.score()[0]) < 0.90)

    # ========================= 10+11. wrong plug, then keep-alive =============================
    torch.manual_seed(3)
    env.reset()
    step(30)
    ls = int(scene.lamp_socket[0])
    ds = 1 - ls
    alpha_d = math.radians(float(scene.slot_deg[0, ds]))
    ok_tw = servo_twist(ds, alpha_d)
    ok_pl = servo_to(ds, c.ext_goal + 0.008, alpha_d)
    step(240)
    report("wrong-plug")
    check("negative (wrong plug): full strategy on the FAN's plug — ~0 score, no success",
          ok_tw and ok_pl and not bool(scene.decoy_seated()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.05)
    # ...and in the SAME episode also do the lamp plug CORRECTLY: keep-alive must bite.
    alpha = math.radians(float(scene.slot_deg[0, ls]))
    ok_tw2 = servo_twist(ls, alpha)
    ok_pl2 = servo_to(ls, c.ext_goal + 0.008, alpha)
    step(240)
    report("both-out")
    check("negative (keep-alive): lamp out but fan unplugged too — rejected, quartered",
          ok_tw2 and ok_pl2 and float(scene.lamp_ext()[0]) >= c.ext_goal
          and not bool(scene.decoy_seated()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.30)

    # ========================= 12. exactness: success == score 1.0 ============================
    torch.manual_seed(3)
    env.reset()
    step(30)
    ls = int(scene.lamp_socket[0])
    alpha = math.radians(float(scene.slot_deg[0, ls]))
    ok_tw = servo_twist(ls, alpha)
    ok_pl = servo_to(ls, c.ext_goal + 0.008, alpha)
    step(240)
    report("goal-state")
    check("exactness: full correct strategy -> success() and score == 1.0",
          ok_tw and ok_pl and bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 1.0) < 1e-3)

    # ========================= 13. achievement latch ==========================================
    ok_back = servo_to(ls, 0.001, alpha, tol=0.003)
    step(240)
    report("reseated")
    check("achievement latch: reseating revokes success; latched 0.70 remains",
          ok_back and float(scene.lamp_ext()[0]) < 0.006
          and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.70) < 0.02)

    # ========================= 14. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.twistlock_unplug")
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

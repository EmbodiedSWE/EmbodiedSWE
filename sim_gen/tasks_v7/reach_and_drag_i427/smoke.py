"""smoke — REJECTION battery for the TrolleyShuntScene rubric (NullRobot, RECORDED).

This module is NOT a solution (solve.py already proves the rubric ACCEPTS the driven
route on two seeds). Every check here CONSTRUCTS a wrong outcome as a settled state
(teleports are instrumentation; probe forces go through the scene's frame-encoded
world-wrench buffers) and asserts the rubric REJECTS it:

  1. settle/no-NaN   — reset settles finite: trolley upright at rolling height in the
                       red lane, no latches, score ~0;
  2. randomization   — READBACK: yard xy + yaw vary across seeded resets; BOTH goal
                       sides drawn; the GREEN flag sits on the flip-side shed roof and
                       the RED flag on the other every reset; the trolley starts in the
                       red lane every reset;
  3. null policy     — 600 idle steps -> trolley stays at its start (< 2 cm), score 0;
  4. nonholonomy     — the same 1.5 N x 0.5 s shove moves the trolley >= 3x farther
                       along its heading than across it (the wheels are real: it must
                       be steered, not dragged sideways like the seed's cube);
  5. red-shed park   — the seed strategy ("shove it onto the nearest target"): trolley
                       parked perfectly inside the RED decoy shed behind the start ->
                       score 0 (no route latch ever fires), no success;
  6. overshoot       — full route latched, then the trolley left settled TOO DEEP in
                       the green shed (8 mm past the band): g3 banked (0.70) but the
                       park band rejects it, no success;
  7. short stop      — full lane approach latched, trolley stopped 16+ cm SHORT of the
                       sill, perfectly aligned and still: g3 never fires, score 0.45,
                       no success;
  8. fly-over        — g1+g2 banked, then the trolley CARRIED (held pose, 120 steps)
                       directly above the green park spot: the rolling-band z-gate
                       keeps g3 off throughout — a lifted trolley earns nothing, 0.45,
                       no success;
  9. sanctioned park — from the far-lane approach the trolley is force-DRIVEN over the
                       sill ramp and braked in the band (same servo as solve): genuine
                       accept -> success, score 1.0;
 10. sill retention  — the wheels are first SEATED dead against the inner 12 mm step
                       (1.5 N creep, zero run-up), then 3 N of sustained outward shove
                       for 1 s never climbs the step (static exit needs ~11 N > the
                       8 N cap); g3 stays banked;
 11. recovery        — after the failed extraction the sanctioned servo re-parks the
                       trolley in the band: success and score 1.0 return;
 12. no accidental success — success() never fired outside the sanctioned segment;
 13. final no-NaN    — every body finite at the end.

Records video frames -> frames.npz in the CWD.
Run (forge): python -u -m simgen_tasks.reach_and_drag_i427.smoke --headless
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--record_every", type=int, default=12)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--max_sec", type=float, default=1200.0)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version and silently rejects RTX -> the
# annotator returns EMPTY frames. Disable the driver check.
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
    from simgen_tasks.reach_and_drag_i427 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

G = scene_mod.G

_wd = threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                             os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.trolley_shunt")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    ids = torch.zeros(1, dtype=torch.long, device=device)

    # --- recording (viewport rgb annotator, the proven forge mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.85, -1.05, 0.85)) + o),
                                tuple(np.array((-0.05, 0.0, 0.05)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0
    saw_success = False
    allow_success = False

    def step(k: int) -> None:
        nonlocal step_i, saw_success
        for _ in range(k):
            env.step(no_action)
            if not allow_success:
                saw_success = saw_success or bool(scene.success()[0])
            if (annot is not None and step_i % args.record_every == 0
                    and len(frames) < args.max_frames):
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        rel = scene.deck_rel()[0]
        h = scene.heading_rel()[0]
        print(f"[smoke] {tag:14s} p=({rel[0]:+.3f},{rel[1]:+.3f},{rel[2]:.3f}) "
              f"th={math.degrees(math.atan2(float(h[1]), float(h[0]))):+7.1f} "
              f"up={float(scene.up_z()[0]):+.2f} "
              f"g={int(scene._g1[0])}{int(scene._g2[0])}{int(scene._g3[0])} "
              f"hold={int(scene._hold[0])} score={sc():.3f} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def push_off() -> None:
        scene.push_f_w[0] = 0.0
        scene.push_tau_w[0] = 0.0

    # trolley body offsets (trolley frame): deck + wheels + caster
    OFFS = (
        (scene.deck, (0.0, 0.0, 0.0)),
        (scene.wheels[0], (G.WHEEL_X, G.WHEEL_Y, G.WHEEL_Z)),
        (scene.wheels[1], (G.WHEEL_X, -G.WHEEL_Y, G.WHEEL_Z)),
        (scene.caster, (G.CAST_X, 0.0, G.CAST_Z)),
    )

    def place_trolley(x: float, y: float, yaw_deg: float, z: float = G.Z0 + 0.002) -> None:
        """Teleport the whole trolley cluster to a yard-frame pose (zero velocity)."""
        py = scene.yard.data.root_pos_w[0]
        qy = scene.yard.data.root_quat_w[0].view(1, 4)
        yaw = torch.tensor([math.radians(yaw_deg)], device=device)
        zero = torch.zeros(1, device=device)
        ql = quat_from_euler_xyz(zero, zero, yaw)  # Rz(yaw) in the yard frame
        qb = quat_mul(qy, ql)
        base = torch.tensor([[x, y, z]], device=device)
        for body, off in OFFS:
            ov = torch.tensor([off], device=device)
            pos_yard = base + quat_apply(ql, ov)
            st = torch.zeros(1, 13, device=device)
            st[0, 0:3] = py + quat_apply(qy, pos_yard)[0]
            st[0, 3:7] = qb[0]
            body.write_root_state_to_sim(st, ids)

    def x_hat_w() -> torch.Tensor:
        ex = torch.zeros(1, 3, device=device)
        ex[0, 0] = 1.0
        return quat_apply(scene.yard.data.root_quat_w, ex)[0]

    def head_w() -> torch.Tensor:
        return quat_apply(scene.deck.data.root_quat_w, scene._ex)[0]

    def finite() -> bool:
        return all(bool(torch.isfinite(b.data.root_state_w).all())
                   for b in (scene.yard, scene.deck, *scene.wheels, scene.caster,
                             scene.flags["g"], scene.flags["r"]))

    def upright_at(x: float, y_s: float, tol: float = 0.05) -> bool:
        rel = scene.deck_rel()[0]
        s = float(scene.flip[0])
        return (abs(float(rel[0]) - x) < tol and abs(s * float(rel[1]) - y_s) < tol
                and 0.03 < float(rel[2]) < 0.075 and float(scene.up_z()[0]) > 0.9)

    def drive_in_and_park() -> bool:
        """Solve's audited servo: drive west from the far-lane approach over the sill,
        brake in the band. k_v*dt/m = 0.13 < 1."""
        s = float(scene.flip[0])
        ez = torch.tensor([0.0, 0.0, 1.0], device=device)
        for _ in range(2400):
            rel = scene.deck_rel()[0]
            if float(rel[0]) <= -0.290:
                break
            h = scene.heading_rel()[0]
            th = math.atan2(float(h[1]), float(h[0]))
            err = (math.atan2(s * 0.150 - float(rel[1]), -0.34 - float(rel[0]))
                   - th + math.pi) % (2 * math.pi) - math.pi
            w_z = float(scene.deck.data.root_ang_vel_w[0, 2])
            scene.push_tau_w[0] = ez * max(-1.0, min(1.0, 1.2 * err - 0.35 * w_z))
            hw = head_w()
            v_fwd = float((scene.deck.data.root_lin_vel_w[0] * hw).sum())
            v_des = 0.10 * max(0.0, math.cos(err))
            f = 30.0 * (v_des - v_fwd)
            if -0.25 < float(rel[0]) < -0.045:
                f += 3.5  # ramp feed-forward (m*g*sin(9.5deg) = 3.2 N)
            scene.push_f_w[0] = hw * max(-8.0, min(8.0, f))
            step(1)
        else:
            push_off()
            return False
        # brake to a DEAD stop (residual coast on the free wheels walks it out of the
        # band during the hold otherwise): v < 2 mm/s for 30 straight steps
        scene.push_tau_w[0] = 0.0
        streak = 0
        for _ in range(720):
            v_w = scene.deck.data.root_lin_vel_w[0]
            streak = streak + 1 if float(v_w.norm()) < 0.002 else 0
            if streak >= 30:
                break
            f = -30.0 * v_w
            f[2] = 0.0
            n = float(f.norm())
            if n > 8.0:
                f = f * (8.0 / n)
            scene.push_f_w[0] = f
            step(1)
        push_off()
        return True

    # =========================== 1. settle / no-NaN =========================================
    env.reset(seed=11)
    step(90)
    report("reset")
    s = float(scene.flip[0])
    rel = scene.deck_rel()[0]
    check("settle: finite, trolley upright at rolling height in the red (start) lane, "
          "no latches, score ~0",
          finite() and float(scene.up_z()[0]) > 0.98
          and 0.035 < float(rel[2]) < 0.065 and s * float(rel[1]) < -0.06
          and not (bool(scene._g1[0]) or bool(scene._g2[0]) or bool(scene._g3[0]))
          and sc() <= 0.005 and not bool(scene.success()[0]))

    # =========================== 2. randomization is real ===================================
    reads, flips = [], set()
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(30)
        kp = (scene.yard.data.root_pos_w - scene.env_origins)[0]
        yawv = float(scene.yard_yaw()[0])
        sv = float(scene.flip[0])
        flips.add(sv)
        # flag readback in the yard frame
        fg = quat_apply_inverse(scene.yard.data.root_quat_w,
                                scene.flags["g"].data.root_pos_w
                                - scene.yard.data.root_pos_w)[0]
        fr = quat_apply_inverse(scene.yard.data.root_quat_w,
                                scene.flags["r"].data.root_pos_w
                                - scene.yard.data.root_pos_w)[0]
        relv = scene.deck_rel()[0]
        ok = (abs(float(fg[1]) - sv * G.SHED_YC) < 0.01
              and abs(float(fr[1]) + sv * G.SHED_YC) < 0.01
              and abs(float(fg[0]) - G.FLAG_X) < 0.01
              and float(fg[2]) > G.ROOF_Z1 - 0.01
              and sv * float(relv[1]) < -0.06)  # trolley starts in the red lane
        reads.append((float(kp[0]), float(kp[1]), yawv, ok))
        print(f"[smoke] seed {sd}: yard=({kp[0]:+.3f},{kp[1]:+.3f}) yaw={yawv:+.2f} "
              f"s={sv:+.0f} flags_ok={ok}", flush=True)
    arr = np.array([r[:3] for r in reads])
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: yard xy + yaw vary (readback), BOTH goal sides drawn, green "
          "flag on the flip-side shed roof + red opposite + trolley in the red lane "
          "every reset",
          spread[0] > 0.04 and spread[1] > 0.04 and spread[2] > 1.0
          and len(flips) == 2 and all(r[3] for r in reads))

    # =========================== 3. null policy =============================================
    env.reset(seed=31)
    step(30)
    p0 = scene.deck_rel()[0].clone()
    step(600)
    report("null-policy")
    d = float((scene.deck_rel()[0][:2] - p0[:2]).norm())
    check("null policy: 600 idle steps -> trolley stays at its start (< 2 cm), score 0, "
          "no success",
          d < 0.02 and sc() <= 0.005 and not bool(scene.success()[0]))

    # =========================== 4. nonholonomy (steer, don't drag) =========================
    env.reset(seed=41)
    step(60)
    h0 = head_w().clone()
    left0 = torch.tensor([-float(h0[1]), float(h0[0]), 0.0], device=device)
    p0w = scene.deck.data.root_pos_w[0].clone()
    for _ in range(60):
        scene.push_f_w[0] = left0 * 1.5  # sideways shove, 0.5 s
        step(1)
    push_off()
    step(30)
    d_lat = abs(float(((scene.deck.data.root_pos_w[0] - p0w) * left0).sum()))
    env.reset(seed=41)
    step(60)
    h0 = head_w().clone()
    p0w = scene.deck.data.root_pos_w[0].clone()
    for _ in range(60):
        scene.push_f_w[0] = h0 * 1.5  # the same shove along the heading
        step(1)
    push_off()
    step(30)
    d_fwd = float(((scene.deck.data.root_pos_w[0] - p0w) * h0).sum())
    print(f"[smoke] nonholonomy: d_fwd={d_fwd:.3f} d_lat={d_lat:.3f}", flush=True)
    check("nonholonomy: the same 1.5 N x 0.5 s shove rolls the trolley >= 3x farther "
          "along its heading (>= 4 cm) than it skids sideways",
          d_fwd >= 0.04 and d_fwd >= 3.0 * d_lat)

    # =========================== 5. seed strategy: park in the RED decoy ====================
    env.reset(seed=51)
    step(30)
    s = float(scene.flip[0])
    place_trolley(-0.295, -s * 0.150, 180.0 if s > 0 else 180.0)
    step(300)
    report("red-shed")
    rel = scene.deck_rel()[0]
    check("red-shed park (seed strategy: nearest target): trolley settled fully inside "
          "the RED decoy shed -> no route latch, score 0, no success",
          abs(float(rel[0]) + 0.295) < 0.03 and s * float(rel[1]) < -0.10
          and not (bool(scene._g1[0]) or bool(scene._g2[0]) or bool(scene._g3[0]))
          and sc() <= 0.005 and not bool(scene.success()[0]))

    # =========================== 6. straddle the green shed mouth ===========================
    env.reset(seed=61)
    step(30)
    s = float(scene.flip[0])
    place_trolley(0.20, -s * 0.135, 0.0)   # earn g1 (east zone, rolling height)
    step(5)
    place_trolley(-0.02, s * 0.140, 180.0)  # earn g2 (far lane)
    step(5)
    place_trolley(-0.313, s * 0.150, 180.0)  # overshot: too deep, 8 mm past the band
    step(300)
    report("overshoot")
    rel = scene.deck_rel()[0]
    check("overshoot: route fully latched (g3 = 0.70 banked crossing the sill line) but "
          "the trolley rests jammed too deep, outside the park band -> no success for "
          "2.5 s",
          bool(scene._g1[0]) and bool(scene._g2[0]) and bool(scene._g3[0])
          and abs(float(rel[0]) + 0.313) < 0.02
          and 0.695 <= sc() <= 0.705 and not bool(scene.success()[0]))

    # =========================== 7. short stop outside the mouth ============================
    env.reset(seed=71)
    step(30)
    s = float(scene.flip[0])
    place_trolley(0.20, -s * 0.135, 0.0)
    step(5)
    place_trolley(-0.02, s * 0.140, 180.0)
    step(5)
    place_trolley(-0.030, s * 0.150, 180.0)  # aligned, still, caster 18 mm east of the
    step(300)                                # ramp toe (-0.108): all gear on flat floor
    report("short-stop")
    check("short stop: perfect far-lane approach (g2 = 0.45) but stopped 16+ cm short "
          "of the sill, aligned and still -> g3 never fires, no success",
          bool(scene._g2[0]) and not bool(scene._g3[0])
          and upright_at(-0.030, 0.150)
          and 0.445 <= sc() <= 0.455 and not bool(scene.success()[0]))

    # =========================== 8. fly-over: set down ON the shed roof =====================
    env.reset(seed=81)
    step(30)
    s = float(scene.flip[0])
    place_trolley(0.20, -s * 0.135, 0.0)
    step(5)
    place_trolley(-0.02, s * 0.140, 180.0)
    step(5)
    # HELD carry: pose-written every step (a free wheeled body rolls off the flat roof).
    g3_stayed_off = True
    z_seen = 0.0
    for _ in range(120):
        place_trolley(-0.29, s * 0.150, 180.0, z=0.220)  # hovering over the park spot
        step(1)
        g3_stayed_off = g3_stayed_off and not bool(scene._g3[0])
        z_seen = max(z_seen, float(scene.deck_rel()[0, 2]))
    report("fly-over")
    check("fly-over: trolley CARRIED (held pose, 120 steps) directly above the green "
          "park spot (g1+g2 banked) -> the rolling-band z-gate keeps g3 off the whole "
          "time, score 0.45, no success",
          bool(scene._g2[0]) and g3_stayed_off and z_seen > 0.12
          and 0.445 <= sc() <= 0.455 and not bool(scene.success()[0]))

    # =========================== 9. sanctioned force-driven park ============================
    env.reset(seed=91)
    step(30)
    s = float(scene.flip[0])
    place_trolley(0.20, -s * 0.135, 0.0)
    step(5)
    place_trolley(-0.02, s * 0.140, 180.0)
    step(30)
    allow_success = True  # sanctioned segment: genuine accept
    drove = drive_in_and_park()
    got = False
    for _ in range(100):
        step(10)
        if bool(scene.success()[0]):
            got = True
            break
    report("driven-park")
    check("sanctioned park: force-driven over the sill ramp and braked in the band "
          "(solve's servo) -> success, score 1.0",
          drove and got and sc() == 1.0)

    # =========================== 10. sill retention =========================================
    # Zero-run-up probe: first SEAT the wheels against the inner step with a gentle
    # 1.5 N creep (a run-up start let 3 N pop the old 6 mm step dynamically), then hold
    # the full 3 N outward for 1 s from dead contact. Static exit needs ~11 N > 8 N cap.
    xw = x_hat_w()
    for _ in range(360):
        scene.push_f_w[0] = xw * 1.5  # gentle east creep to the step (contact seat)
        step(1)
    x_contact = float(scene.deck_rel()[0, 0])  # expected ~ -0.273 (rim on step corner)
    x_max = x_contact
    for _ in range(120):
        scene.push_f_w[0] = xw * 3.0  # full sustained outward shove, 1 s, no run-up
        step(1)
        x_max = max(x_max, float(scene.deck_rel()[0, 0]))
    push_off()
    step(300)
    report("retention")
    check("sill retention: seated dead against the inner step (deck x ~ -0.273), a "
          "sustained 3 N outward shove never climbs the 12 mm step (x stays >= 4.5 cm "
          "behind the sill line); g3 stays banked",
          -0.283 < x_contact < -0.250 and x_max < G.SILL_X0 - 0.045
          and bool(scene._g3[0]) and float(scene.deck_rel()[0, 0]) < G.SILL_X0 - 0.045)

    # =========================== 11. recovery re-park =======================================
    # The shove walked the trolley out of the park band; the sanctioned drive can
    # recover it: re-park and success must return.
    drove2 = drive_in_and_park()
    got2 = False
    for _ in range(100):
        step(10)
        if bool(scene.success()[0]):
            got2 = True
            break
    report("re-park")
    check("recovery: after the failed extraction the sanctioned servo re-parks the "
          "trolley in the band -> success and score 1.0 again",
          drove2 and got2 and sc() == 1.0)
    allow_success = False

    # =========================== 12./13. global =============================================
    check("no accidental success: success() never fired outside the sanctioned segment",
          not saw_success)
    check("final: every body state finite", finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.trolley_shunt")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, ok in checks:
            if not ok:
                print(f"[smoke]   FAILED: {nm}", flush=True)
    code = 0 if all_ok else 1
    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL", flush=True)
        os._exit(1)

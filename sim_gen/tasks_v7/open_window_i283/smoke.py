"""Smoke / rubric-REJECTION battery for SashVentScene (sim_gen task
`open_window_i283`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — lift the sash past the pawl, let the tab
gravity-return, set the sash down on it, then push the parcel through the propped
gap onto the tray — is the acceptance evidence; it passes on forge seeds 0/1/2).
Every teleport here is instrumentation that CONSTRUCTS a wrong (or partial) outcome
and asserts the rubric REJECTS it — plus physical probes that prove the dead-man
sash and the one-way pawl are working mechanisms, not props. No probe in this
battery ever reaches success(), and a final audit check asserts exactly that.

   1. settle/no-NaN      — reset settles finite: sash closed on its stop, tab on its
                           horizontal stop, cubes upright on the floor, all settled;
   2. fresh reset        — score ~0, no success;
  3-4. randomization     — READBACK over 6 seeded resets: the cube slot assignment
                           varies (and xy jitter is real); the free yaw varies;
   5. null policy        — 240 idle steps -> score ~0, no success;
   6. dead-man sash      — the sash teleported to q=0.07 (BELOW the pawl's reach,
                           clear of the tab) falls closed ON ITS OWN (q readback
                           0.07 -> ~0); the open latch never fires: merely being
                           put open earns nothing un-propped;
   7. one-way pawl       — a hinge-axis torque swings the tab up to the -85 deg hard
                           stop (clamped, readback), and with the torque OFF gravity
                           returns it to the horizontal stop — the ratchet is real
                           and self-resetting;
   8. seed-analog        — the seed family's end state ("the window is open"): the
                           sash lowered onto the pawl rests propped at q_prop and
                           HOLDS hands-off 240 steps (a genuine mechanism state,
                           credit 0.30) — but it is NOT success and score <= 0.35:
                           opening the window is only a means here;
   9. closed-window denial — the solve's own push servo, window CLOSED: the parcel
                           is physically STOPPED by the sash (x readback jams at the
                           panel face), never transits, score ~0 — the closed sash
                           seals the aperture;
  10. over-the-wall cheat — window propped, parcel teleported DIRECTLY onto the tray
                           (the "carried over/around the wall" analog): in_tray reads
                           True but the through-latch never fired -> tray credit
                           denied, NOT success, score stays 0.30;
  11. near-miss          — parcel settled 7 mm short of the tray x band (past the
                           transit slab, before the band): not in_tray, NOT success;
  12. wrong object       — the BLUE distractor placed on the tray earns nothing:
                           score stays 0.30, NOT success;
  13. stacked z band     — the parcel stacked ON the distractor in the tray (z
                           readback ~ rest + 60 mm) is rejected by the z band;
  14. ground z band      — the parcel on the GROUND beyond the tray (z ~ 0.03) is
                           rejected by the z band and the x band;
  15. settle gate        — the parcel genuinely pushed into the tray but judged
                           WHILE STILL MOVING: in_tray & through & open all read
                           True yet success is False (stillness must persist); the
                           parcel is removed before it settles (the battery never
                           succeeds);
  16. rejection audit    — success() was never True at ANY judged point;
  17. final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.open_window_i283.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version and silently rejects RTX -> the
# annotator returns EMPTY frames. Disable the driver check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import os
import threading

import numpy as np
import torch

import robobench
from robobench.core import ENVS

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:  # older isaaclab name
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.sash_vent")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.15, -0.80, 0.75)) + o),
                                tuple(np.array((-0.05, 0.00, 0.22)) + o),
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

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action)
            if (annot is not None and step_i % args.record_every == 0
                    and len(frames) < args.max_frames):
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    ever_success = [False]

    def judge() -> tuple[float, bool]:
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return s, ok

    def q0() -> float:
        return float(scene.sash_q()[0])

    def tab0() -> float:
        return float(scene.tab_angle()[0])

    def loc(body) -> tuple[float, float, float]:
        p = (body.data.root_pos_w - scene.env_origins)[0]
        return float(p[0]), float(p[1]), float(p[2])

    def report(tag: str) -> None:
        s, ok = judge()
        px, py, pz = loc(scene.parcel)
        print(f"[smoke] {tag:16s} | q={q0():+.4f} tab={tab0():+.3f} "
              f"open={bool(scene.sash_open()[0])} | parcel=({px:+.3f},{py:+.3f},{pz:+.3f}) "
              f"thr={bool(scene.parcel_through_now()[0])} tray={bool(scene.parcel_in_tray()[0])} "
              f"| o/t/t={float(scene.open_latch[0]):.0f}/{float(scene.through_latch[0]):.0f}/"
              f"{float(scene.tray_latch[0]):.0f} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    UP = (1.0, 0.0, 0.0, 0.0)

    def write_body(body, local_xyz, quat=UP, settle_steps: int = 60) -> None:
        """Kinematic probe placement in the FRAME frame (instrumentation, not a
        solution) + REAL physics steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.env_origins
        st[:, 0] += float(local_xyz[0])
        st[:, 1] += float(local_xyz[1])
        st[:, 2] += float(local_xyz[2])
        st[:, 3:7] = torch.tensor(quat, device=device)
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def sash_to(q: float, settle_steps: int = 60) -> None:
        """Teleport the sash ALONG ITS TRACK to joint position q (instrumentation)."""
        write_body(scene.sash, (c.sash_center0[0], c.sash_center0[1],
                                c.sash_center0[2] + q), UP, settle_steps)

    def prop_sash(settle_steps: int = 240) -> None:
        """Construct the propped state the honest way: release the sash just ABOVE
        the pawl (clear of the tab) and let it LAND on the tab."""
        sash_to(c.q_prop + 0.020, settle_steps)

    def push_parcel(x_tgt: float, max_steps: int, x_break: float) -> int:
        """The solve's own push servo (world-frame force rotated into the body
        frame each step). Returns servo steps used; leaves the wrench OFF."""
        done, i = 0, 0
        for i in range(max_steps):
            x = scene.parcel_local()[:, 0]
            v = scene.parcel.data.root_lin_vel_w[:, 0]
            v_des = (3.0 * (x_tgt - x)).clamp(-0.15, 0.15)
            fx = 8.0 * (v_des - v)
            fx = fx + torch.where(v.abs() < 0.02, 0.60 * v_des.sign(),
                                  torch.zeros_like(fx))
            fx = fx.clamp(-2.0, 2.0)
            f_world = torch.zeros(n, 3, device=device)
            f_world[:, 0] = fx
            f_body = quat_apply_inverse(scene.parcel.data.root_quat_w, f_world)
            scene.parcel.set_external_force_and_torque(
                f_body.unsqueeze(1), zero_w, env_ids=all_ids)
            env.step(no_action)
            done = done + 1 if float(x[0]) <= x_break else 0
            if done >= 3:
                break
        scene.parcel.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
        return i + 1

    bodies = (scene.frame, scene.sash, scene.tab, scene.parcel, scene.distractor)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    cubes_ok = all(abs(loc(b)[2] - c.cube / 2) < 0.012
                   for b in (scene.parcel, scene.distractor))
    check("settle: all states finite, sash closed on its stop "
          f"(q={q0():+.4f}), tab on its horizontal stop (ang={tab0():+.3f}), cubes "
          "upright on the floor, everything settled",
          fin and abs(q0()) <= 0.008 and abs(tab0()) <= 0.05 and cubes_ok
          and bool(scene.settled()[0]))
    s, ok = judge()
    check("fresh reset: score ~0, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(20)
        px, py, _ = loc(scene.parcel)
        dx, dy, _ = loc(scene.distractor)
        qp = scene.parcel.data.root_quat_w[0]
        yaw = float(2.0 * torch.atan2(qp[3], qp[0]))
        slot = (int(np.argmin([abs(py - sy) for sy in c.slot_ys])),
                int(np.argmin([abs(dy - sy) for sy in c.slot_ys])))
        reads.append((slot, px, py, yaw))
        print(f"[smoke] seed {sd}: slots={slot} parcel=({px:+.3f},{py:+.3f}) "
              f"yaw={yaw:+.2f}", flush=True)
    slots = {r[0] for r in reads}
    x_spread = max(r[1] for r in reads) - min(r[1] for r in reads)
    check("randomization: cube slot assignment varies (readback: "
          f"{len(slots)} distinct / 6 resets) and xy jitter is real "
          f"(parcel x spread {x_spread * 1000:.0f} mm)", len(slots) >= 3 and x_spread > 0.01)
    yaws = [r[3] for r in reads]
    yaw_spread = max(yaws) - min(yaws)
    check(f"randomization: free yaw varies (readback spread {yaw_spread:.2f} rad)",
          yaw_spread > 0.5)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 6. dead-man sash ===========================================
    # q=0.07: below the pawl's reach (lip top 0.222 < tab bottom 0.232 — no contact)
    # and below q_open_min: the un-propped sash must fall closed on its own.
    sash_to(0.07, settle_steps=0)
    step(2)
    q_mid = q0()
    step(180)
    report("dead-man")
    s, ok = judge()
    check("dead-man: the sash teleported open to q=0.070 falls closed ON ITS OWN "
          f"(readback q {q_mid:+.4f} 2 steps in -> {q0():+.4f}); the open latch "
          "never fires — un-propped openness earns nothing",
          q_mid < 0.075 and q0() <= 0.008 and float(scene.open_latch[0]) < 0.5
          and s <= 0.02 and not ok)

    # =========================== 7. one-way pawl mechanism ==================================
    # Hinge-axis torque in the BODY frame (the tab only rotates about its own x).
    tq = torch.zeros(n, 1, 3, device=device)
    tq[:, 0, 0] = -0.08  # swing-up direction; ~4x the tab's gravity torque
    min_ang = 0.0
    for _ in range(180):
        scene.tab.set_external_force_and_torque(zero_w, tq, env_ids=all_ids)
        env.step(no_action)
        min_ang = min(min_ang, tab0())
    swung = tab0()
    scene.tab.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
    step(300)
    report("pawl-cycle")
    check("one-way pawl: a hinge torque swings the tab up to the hard stop "
          f"(readback {swung:+.3f} rad, never past {min_ang:+.3f} >= -1.55) and with "
          f"the torque OFF gravity returns it to horizontal ({tab0():+.3f}) — the "
          "ratchet is real and self-resetting",
          swung < -1.2 and min_ang > -1.55 and abs(tab0()) < 0.06)

    # =========================== 8. seed-analog: "the window is open" =======================
    env.reset(seed=41)
    step(30)
    prop_sash(settle_steps=240)
    q_propped = q0()
    step(240)  # hands-off hold — the prop is a real mechanism state
    report("seed-analog")
    s, ok = judge()
    check("seed-analog (the seed's whole goal — window open): the sash released "
          f"just above the pawl LANDS on it and holds hands-off (q {q_propped:+.4f} "
          f"-> {q0():+.4f} ~ q_prop {c.q_prop:+.4f}); genuine open credit 0.30, but "
          "NOT success and score <= 0.35 — opening is only a means here",
          abs(q0() - c.q_prop) <= 0.010 and abs(q_propped - c.q_prop) <= 0.010
          and float(scene.open_latch[0]) > 0.5 and 0.28 <= s <= 0.35 and not ok)

    # =========================== 9. closed-window denial ====================================
    env.reset(seed=51)
    step(30)
    write_body(scene.parcel, (0.09, 0.0, c.sill_top + c.cube / 2 + 0.003), UP, 30)
    used = push_parcel(-0.135, 500, -0.095)
    step(60)
    report("closed-denial")
    px, _, pz = loc(scene.parcel)
    s, ok = judge()
    check("closed-window denial: the solve's own push servo against the CLOSED sash "
          f"— the parcel moved ({used} servo steps) but jams at the panel face "
          f"(x={px:+.4f} > +0.02, z={pz:+.3f} still on the sill), never transits "
          "(through latch 0), score ~0",
          used >= 400 and px > 0.02 and abs(pz - (c.sill_top + c.cube / 2)) < 0.02
          and float(scene.through_latch[0]) < 0.5 and s <= 0.02 and not ok)

    # =========================== 10. over-the-wall cheat ====================================
    env.reset(seed=61)
    step(30)
    prop_sash()
    write_body(scene.parcel, (-0.105, 0.0, c.tray_rest_z + 0.002), UP, 90)
    report("cheat-lob")
    s, ok = judge()
    check("over-the-wall cheat: window propped, parcel teleported DIRECTLY onto the "
          f"tray (in_tray reads {bool(scene.parcel_in_tray()[0])}) — but it never "
          "crossed the transit slab, so the through latch is 0, tray credit is "
          "denied, NOT success, score stays 0.30",
          bool(scene.parcel_in_tray()[0]) and float(scene.through_latch[0]) < 0.5
          and float(scene.tray_latch[0]) < 0.5 and 0.28 <= s <= 0.35 and not ok)

    # =========================== 11. near-miss: short of the tray band ======================
    write_body(scene.parcel, (-0.048, 0.0, c.tray_rest_z + 0.002), UP, 90)
    report("near-miss")
    s, ok = judge()
    px, _, _ = loc(scene.parcel)
    check("near-miss: parcel settled 7 mm short of the tray x band "
          f"(x={px:+.4f}, band starts {c.tray_x_hi:+.3f}) -> not in_tray, NOT success",
          not bool(scene.parcel_in_tray()[0]) and not ok and s <= 0.35)

    # =========================== 12. wrong object ===========================================
    write_body(scene.parcel, (0.45, 0.25, c.cube / 2 + 0.003), UP, 30)  # parcel home
    write_body(scene.distractor, (-0.105, 0.0, c.tray_rest_z + 0.002), UP, 90)
    report("wrong-object")
    s, ok = judge()
    check("wrong object: the BLUE distractor on the tray earns nothing — score stays "
          f"{s:.2f} (open credit only), NOT success",
          not bool(scene.parcel_in_tray()[0]) and 0.28 <= s <= 0.35 and not ok)

    # =========================== 13. stacked z band =========================================
    write_body(scene.parcel, (-0.105, 0.0, c.tray_rest_z + c.cube + 0.004), UP, 60)
    px, _, pz = loc(scene.parcel)
    report("stacked")
    s, ok = judge()
    check("z band: the parcel STACKED on the distractor in the tray reads "
          f"z={pz:+.4f} (rest {c.tray_rest_z:+.4f}) -> rejected by the z band, "
          "NOT success",
          pz > c.tray_rest_z + 0.04 and not bool(scene.parcel_in_tray()[0]) and not ok)

    # =========================== 14. ground z band ==========================================
    write_body(scene.parcel, (-0.30, 0.0, c.cube / 2 + 0.003), UP, 60)
    _, _, pz = loc(scene.parcel)
    report("ground")
    s, ok = judge()
    check("z band: the parcel on the GROUND beyond the tray reads "
          f"z={pz:+.4f} (rest {c.tray_rest_z:+.4f}) -> rejected (x and z bands), "
          "NOT success",
          pz < 0.05 and not bool(scene.parcel_in_tray()[0]) and not ok)

    # =========================== 15. settle gate ============================================
    env.reset(seed=71)
    step(30)
    prop_sash()
    # stage INSIDE, then genuinely push through the slab into the tray — but judge
    # while the parcel is still moving.
    write_body(scene.parcel, (0.09, 0.0, c.sill_top + c.cube / 2 + 0.003), UP, 30)
    push_parcel(-0.135, 900, -0.085)
    s, ok = judge()  # parcel just crossed; still sliding
    in_tray_now = bool(scene.parcel_in_tray()[0])
    thr = float(scene.through_latch[0]) > 0.5
    settled_now = bool(scene.settled()[0])
    report("settle-gate")
    # remove it BEFORE it settles (the battery must never reach success)
    write_body(scene.parcel, (0.45, 0.25, c.cube / 2 + 0.003), UP, 30)
    check("settle gate: the parcel genuinely pushed through into the tray but judged "
          f"while still moving (in_tray={in_tray_now}, through={thr}, "
          f"settled={settled_now}) is NOT success; parcel removed before ring-down "
          "(battery never succeeds)",
          thr and not settled_now and not ok and not bool(scene.success()[0]))

    # =========================== 16-17. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.sash_vent")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(okc for _nm, okc in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, okc in checks:
            if not okc:
                print(f"[smoke]   FAILED: {nm}", flush=True)
    code = 0 if all_ok else 1
    # Hard exit: Kit teardown hangs — watchdog then die.
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)

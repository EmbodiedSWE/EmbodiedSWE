"""Smoke / rubric-REJECTION battery for LampTurnstileScene (sim_gen task
`light_bulb_in_i120`) — NullRobot, teleported probe states + force probes, RECORDED.

This is NOT a solution (solve.py — torque the turnstile out, drop the bulb into the
exposed cradle, torque the loaded turnstile back in — is the acceptance evidence that
the rubric ACCEPTS a correct outcome; it passes on seeds 0/1). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled state and
asserts the rubric REJECTS it — plus applied-force probes that prove the SEAL is
physically real geometry. No probe in this battery ever reaches success(), and a
final audit asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: turnstile closed near the
                            lit index, bulb resting in the tray, cradle inside;
                            score ~0 at rest, no success;
  3-4. randomization      — READBACK over 6 seeded resets: turnstile start yaw is
                            physically posed and varies; tray slot / xy / yaw vary
                            and the bulb tracks its tray;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seal (vane)         — seed-strategy family: bring the bulb TO the lamp and
                            push it straight at the closed window (turnstile held
                            closed by per-step state writes — pure instrumentation).
                            The regulated push (<= 1.5 N ~ 3x bulb weight) moves the
                            bulb >= 15 mm and then stalls it on the vane by real
                            collision: the bulb never crosses the window plane,
                            score ~0. Non-vacuous: the probe asserts the bulb MOVED;
  7.  seal (roof)         — the seed's direct move — deliver the bulb from above
                            onto the socket — is physically unavailable: dropped
                            over the cradle's inside position it lands on the ROOF
                            and never reaches the cradle, score ~0;
  8.  loose inside        — bulb constructed loose on the cabinet floor (inside!)
                            counts for nothing: not in the cradle, score ~0;
  9.  near-miss           — turnstile at service (posed along its own DOF), bulb
                            settled on the platter BESIDE the cradle wall -> not in
                            the cradle, no success, only the out-index credit;
  10. wall perch          — bulb balanced ON a cradle wall top reads above the
                            carousel-frame height window -> rejected, then removed
                            before it can topple;
  11. part-way + cap      — out + seated + carried all latched, then the loaded
                            cradle parked 45 deg short of lit: score == 0.70 cap
                            (float32 + eps), NOT success — full credit short of
                            success is impossible without the lit index;
  12. settle gate         — bulb IN the cradle at the lit index but still moving is
                            NOT success (velocity gates are real); removed before it
                            can settle;
  13. latched credit      — teleporting the bulb far away afterwards leaves the
                            latched score unchanged while in_ring() drops;
  14. rejection audit     — success() was never True at ANY judged point;
  15. final no-NaN        — all task-object states finite at the end;
  16. camera              — >= 20 rgb frames captured -> frames.npz.

Run (forge): python -u -m simgen_tasks.light_bulb_in_i120.smoke --headless
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

import math
import os
import threading

import numpy as np
import torch

import robobench
from robobench.core import ENVS

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
    env = ENVS.get("simgen.lamp_turnstile")().build(num_envs=args.num_envs,
                                                    device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(3, device=device)
    ax, ay = c.axle_pos
    seat_z_loc = c.plat_t / 2 + c.bulb_r  # carousel-frame seated bulb center

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.60, -1.20, 1.05)) + o),
                                tuple(np.array((0.42, 0.00, 0.15)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video",
              flush=True)

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

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        s, ok = judge()
        loc = scene.bulb_local()[0]
        print(f"[smoke] {tag:16s} |"
              f" yaw={math.degrees(float(scene.yaw()[0])):+7.1f}deg"
              f" bulb_loc=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
              f"{float(loc[2]):+.3f})"
              f" in_ring={bool(scene.in_ring()[0])}"
              f" inside={bool(scene.bulb_inside()[0])}"
              f" score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, xyz, quat=None, vel=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap). `xyz` is env-relative."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor(xyz, device=device)
        if quat is None:
            st[:, 3] = 1.0
        else:
            st[:, 3:7] = torch.tensor(quat, device=device)
        if vel is not None:
            st[:, 7:10] = torch.tensor(vel, device=device)
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def wrench(body, f3: torch.Tensor, t3: torch.Tensor) -> None:
        """World wrench expressed in the body's current link frame (the house
        convention: is_global=True silently drops the torque on this stack)."""
        from isaaclab.utils.math import quat_apply_inverse

        q = body.data.root_link_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f3.view(1, 3).expand(n, 3)).unsqueeze(1),
            quat_apply_inverse(q, t3.view(1, 3).expand(n, 3)).unsqueeze(1),
            env_ids=all_ids)

    def set_carousel(theta: float) -> None:
        """Probe-only: pose the turnstile ALONG ITS OWN revolute DOF (the kinematic
        housing anchor is world-fixed; a teleport along the joint axis is safe)."""
        half = theta / 2
        place(scene.carousel, (ax, ay, c.plat_mid_z),
              quat=(math.cos(half), 0.0, 0.0, math.sin(half)), settle_steps=8)

    def car_local(off_xyz) -> tuple[float, float, float]:
        """Env-relative world position of a carousel-frame offset."""
        from isaaclab.utils.math import quat_apply

        off = torch.tensor(off_xyz, device=device).view(1, 3)
        p = scene.carousel.data.root_pos_w[:1] + quat_apply(
            scene.carousel.data.root_quat_w[:1], off)
        p = (p - scene.env_origins[:1])[0]
        return (float(p[0]), float(p[1]), float(p[2]))

    def layout_sane(tag: str) -> bool:
        """Reset honesty: turnstile closed near lit (cradle inside), bulb in the
        tray on the open side, far from the cradle."""
        yaw = math.degrees(abs(float(scene.yaw()[0])))
        bulb, tray = rel(scene.bulb), rel(scene.tray)
        ok = (yaw <= c.yaw_jitter_deg + 1.0
              and bool(scene.lit_indexed()[0])
              and abs(float(bulb[0]) - float(tray[0])) < 0.012
              and abs(float(bulb[1]) - float(tray[1])) < 0.012
              and float(bulb[2]) < c.tray_base_h + c.bulb_r + 0.01
              and float(bulb[0]) < ax - c.plat_r
              and not bool(scene.in_ring()[0]))
        if not ok:
            print(f"[smoke] layout sanity VIOLATION at {tag}", flush=True)
        return ok

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    bodies = (scene.carousel, scene.bulb, scene.tray, scene.housing)
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("settle: all states finite; turnstile closed near lit, bulb in the tray, "
          "cradle inside", fin and layout_sane("reset"))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    sane = True
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(30)
        sane = sane and layout_sane(f"seed {sd}")
        yaw = float(scene.yaw()[0])
        tray, bulb = rel(scene.tray), rel(scene.bulb)
        tq = scene.tray.data.root_quat_w[0]
        tyaw = 2.0 * math.atan2(float(tq[3]), float(tq[0]))
        slots = np.array(c.tray_slots)
        slot = int(np.argmin(((slots - np.array([float(tray[0]), float(tray[1])]))
                              ** 2).sum(axis=1)))
        reads.append((yaw, float(tray[0]), float(tray[1]), tyaw, slot,
                      float(bulb[0]), float(bulb[1])))
    arr = np.array(reads)
    print("[smoke] randomization readback (yaw, tray_x, tray_y, tray_yaw, slot, "
          f"bulb_x, bulb_y):\n{np.round(arr, 4)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: turnstile start yaw is physically posed and varies "
          f"(readback spread {math.degrees(spread[0]):.1f} deg, all within "
          "jitter of lit)",
          spread[0] > 0.05 and sane)
    slots_used = {int(x) for x in arr[:, 4]}
    check("randomization: tray slot / xy / yaw vary and the bulb tracks its tray "
          f"(readback: {len(slots_used)} slots, xy spread "
          f"({spread[1]:.3f},{spread[2]:.3f}), yaw spread {spread[3]:.2f} rad)",
          len(slots_used) >= 2 and (spread[1] > 0.01 or spread[2] > 0.01)
          and spread[3] > 0.4)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 6. seal probe: push the bulb at the closed window ==========
    # rlbench/light_bulb_in's verb is carry-the-bulb-to-the-lamp-and-insert. Apply the
    # straight-line version: bulb on the turnstile's outer half, regulated push toward
    # the cradle. The turnstile is HELD closed by per-step state writes so the probe
    # measures the VANE's geometry (a free turnstile would legitimately rotate — that
    # IS the mechanism; here we prove the closed vane itself cannot pass the bulb).
    env.reset(seed=41)
    step(30)
    car_hold = scene.carousel.data.root_state_w.clone()
    car_hold[:, 7:13] = 0.0
    place(scene.bulb, car_local((-c.ring_c, 0.0, seat_z_loc + 0.002)),
          settle_steps=10)
    x0 = float(rel(scene.bulb)[0])
    xmax = x0
    f3 = torch.zeros(3, device=device)
    for _ in range(300):
        scene.carousel.write_root_state_to_sim(car_hold, all_ids)  # hold closed
        vx = float(scene.bulb.data.root_lin_vel_w[0, 0])
        f3[0] = min(max(2.0 * (0.15 - vx), -1.2), 1.2)  # velocity servo, <= 1.2 N
        wrench(scene.bulb, f3, zero3)
        step(1)
        # re-clamp AFTER the step too: a single pre-step write lets the vane yield
        # a hair each step and the pressed bulb ratchet forward on the artifact
        scene.carousel.write_root_state_to_sim(car_hold, all_ids)
        xmax = max(xmax, float(rel(scene.bulb)[0]))
    wrench(scene.bulb, zero3, zero3)
    # settle with the vane still held closed (same clamp as the probe itself —
    # an unclamped settle lets the free turnstile rotate under the resting bulb,
    # which is the mechanism working, not the seal failing)
    for _ in range(45):
        scene.carousel.write_root_state_to_sim(car_hold, all_ids)
        step(1)
        scene.carousel.write_root_state_to_sim(car_hold, all_ids)
    report("vane-push")
    x_loc_end = float(scene.bulb_local()[0][0])
    s, ok = judge()
    check("seal (vane): the regulated straight push (<= 1.2 N ~ 2.4x bulb weight) "
          f"moves the bulb {(xmax - x0) * 1000:.0f} mm (>= 15, non-vacuous) then "
          f"stalls it on the closed vane by real collision — its center never "
          f"reaches the window plane (max x {xmax:.3f} < {ax - 0.005:.3f}) and it "
          f"settles on the OUTER half (local x {x_loc_end:+.3f} < 0), never "
          f"inside; never in the cradle, score ~0",
          xmax - x0 >= 0.015 and xmax < ax - 0.005 and x_loc_end < -0.005
          and not bool(scene.bulb_inside()[0])
          and not bool(scene.in_ring()[0]) and s <= 0.02 and not ok)

    # =========================== 7. seal probe: the roof =====================================
    env.reset(seed=51)
    step(30)
    drop_xy = car_local((c.ring_c, 0.0, 0.0))  # above the cradle's INSIDE position
    place(scene.bulb, (drop_xy[0], drop_xy[1], 0.45), settle_steps=300)
    report("roof-drop")
    bulb = rel(scene.bulb)
    s, ok = judge()
    check("seal (roof): the bulb dropped from above the inside cradle position "
          f"lands on the roof / rolls away (z={float(bulb[2]):.3f}), never reaches "
          "the cradle, score ~0",
          float(bulb[2]) < 0.40 and not bool(scene.in_ring()[0])
          and s <= 0.02 and not ok)

    # =========================== 8. loose bulb INSIDE the cabinet ===========================
    env.reset(seed=61)
    step(30)
    place(scene.bulb, (ax + 0.20, 0.06, c.bulb_r + 0.002), settle_steps=60)
    report("loose-inside")
    s, ok = judge()
    check("loose inside: bulb constructed loose on the cabinet floor (inside!) "
          "counts for nothing — not in the cradle, score ~0",
          bool(scene.bulb_inside()[0]) and not bool(scene.in_ring()[0])
          and s <= 0.02 and not ok)

    # =========================== 9. near-miss: beside the cradle at service =================
    env.reset(seed=71)
    step(30)
    set_carousel(math.pi)
    assert bool(scene.service_indexed()[0]), "probe setup: not at service"
    beside_y = c.ring_inner + c.ring_wall_t + c.bulb_r + 0.006
    place(scene.bulb, car_local((c.ring_c, beside_y, seat_z_loc + 0.002)),
          settle_steps=60)
    report("near-miss")
    loc = scene.bulb_local()[0]
    s, ok = judge()
    check("near-miss: bulb settled on the platter BESIDE the cradle wall "
          f"(|local y|={abs(float(loc[1])) * 1000:.0f} mm > "
          f"{c.ring_xy_tol * 1000:.0f} mm) -> not in the cradle, no success, only "
          "the out-index credit (score <= 0.20 + eps)",
          not bool(scene.in_ring()[0]) and not ok and s <= 0.20 + 1e-5)

    # =========================== 10. wall perch =============================================
    env.reset(seed=81)
    step(30)
    set_carousel(math.pi)
    perch_z_loc = (c.ring_wall_top - c.plat_mid_z) + c.bulb_r + 0.002
    place(scene.bulb, car_local((c.ring_c + c.ring_inner + c.ring_wall_t / 2, 0.0,
                                 perch_z_loc)), settle_steps=5)
    loc = scene.bulb_local()[0]
    report("wall-perch")
    _s, ok = judge()
    perch_ok = (float(loc[2]) > c.ring_z_hi and not bool(scene.in_ring()[0])
                and not ok)
    # remove the perched bulb BEFORE it can topple anywhere interesting
    place(scene.bulb, (-0.30, 0.30, c.bulb_r + 0.002), settle_steps=30)
    check("wall perch: bulb balanced ON the cradle wall top reads carousel-frame "
          f"z={float(loc[2]) * 1000:.0f} mm > {c.ring_z_hi * 1000:.0f} mm -> "
          "rejected by the height window", perch_ok)

    # =========================== 11. part-way round + the 0.70 cap ==========================
    env.reset(seed=91)
    step(30)
    set_carousel(math.pi)  # latches the out index
    place(scene.bulb, car_local((c.ring_c, 0.0, seat_z_loc + 0.004)),
          settle_steps=60)  # latches seated
    assert bool(scene.in_ring()[0]), "probe setup: bulb not seated at service"
    # carry the LOADED cradle back but park it 45 deg short of lit: pose carousel
    # along its own DOF and the riding bulb with it (one consistent transport)
    half = math.radians(45.0) / 2
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1], st[:, 2] = ax, ay, c.plat_mid_z
    st[:, 3], st[:, 6] = math.cos(half), math.sin(half)
    st[:, 0:3] += scene.env_origins
    scene.carousel.write_root_state_to_sim(st, all_ids)
    place(scene.bulb, car_local((c.ring_c, 0.0, seat_z_loc + 0.002)),
          settle_steps=90)
    report("part-way")
    loc = scene.bulb_local()[0]
    s, ok = judge()
    check("part-way + cap: out + seated + carried all latched with the loaded "
          f"cradle parked 45 deg short of lit -> score == 0.70 cap ({s:.4f}, "
          "float32 + eps), NOT success — full credit short of success is "
          "impossible without the lit index",
          bool(scene.in_ring()[0]) and not bool(scene.lit_indexed()[0])
          and 0.695 <= s <= 0.70 + 1e-5 and not ok)

    # =========================== 12. settle gate ============================================
    env.reset(seed=101)
    step(30)  # turnstile already at lit (cradle inside) after reset
    place(scene.bulb, car_local((c.ring_c, 0.0, seat_z_loc + 0.002)),
          vel=(0.0, 0.25, 0.0), settle_steps=2)
    v_now = float(scene.bulb.data.root_lin_vel_w[0].norm())
    report("settle-gate")
    s_before, ok = judge()
    gate_ok = (v_now > c.settle_lin and bool(scene.in_ring()[0])
               and bool(scene.lit_indexed()[0]) and not ok)
    check("settle gate: bulb IN the cradle at the lit index but moving at "
          f"{v_now:.2f} m/s is NOT success (velocity gates are real)", gate_ok)

    # =========================== 13. latched credit survives moving away ====================
    # remove it BEFORE it can settle in the cradle (this battery must never succeed)
    place(scene.bulb, (-0.30, -0.30, c.bulb_r + 0.002), settle_steps=40)
    report("moved-away")
    s_after, ok = judge()
    check("latched credit: teleporting the bulb far away leaves the latched score "
          f"unchanged ({s_before:.2f} -> {s_after:.2f}) while in_ring() drops",
          abs(s_after - s_before) < 1e-3 and not bool(scene.in_ring()[0]) and not ok)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== 16. camera + save + verdict ================================
    check(f"camera: >= 20 rgb frames captured ({len(frames)})", len(frames) >= 20)
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.lamp_turnstile")
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

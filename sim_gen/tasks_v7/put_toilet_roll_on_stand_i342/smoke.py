"""Smoke / rubric-REJECTION battery for RailRingScene (sim_gen task
`put_toilet_roll_on_stand_i342`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — push the captive ring along the rail through both
corners and over the crest, then let gravity drop it down the post onto the plate —
is the acceptance evidence that the rubric ACCEPTS a correct outcome). Every teleport
here is instrumentation that CONSTRUCTS a wrong (or partial, or physically
unreachable) state and asserts the rubric REJECTS it. No probe in this battery ever
reaches success(), and a final audit asserts exactly that.

  1-2.  settle/no-NaN   — reset layout settles finite: the ring HANGING on the first
                          run inside the progress gate, the rail near its nominal;
                          score ~0, no success;
  3-4.  randomization   — READBACK over 6 seeded resets: rail position + yaw and the
                          ring's start station all vary; every reset hangs sane;
  5.    null policy     — 240 idle steps -> score ~0, no success;
  6.    mid-course hang — the ring hung a third of the way along the first run earns
                          PARTIAL latched credit only (0 << score << cap), no success;
  7.    near-crest hang — the ring hung near the end of the second run (a near-miss
                          of the whole delivery) earns graded credit strictly below
                          the cap, is NOT delivered, no success;
  8.    beside the post — the ring FLAT ON THE PLATE, upright, at the delivered
                          height, but with the post NOT through its bore (48 mm away)
                          is rejected by the encircling clause alone: the xy tolerance
                          (24 mm) accepts every around-the-post rest (max offset
                          ~18 mm) and no beside-the-post rest (min offset ~42 mm);
  9.    leaning on post — the ring tilted against the post (rim on the plate) is
                          rejected (tilt and/or off-axis), no success;
  10.   captivity       — PHYSICAL probe: the ring is dragged back toward the red
                          ball cap with a regulated velocity-capped pull — it travels
                          ~100 mm (the probe is not vacuous), then the cap BLOCKS it;
                          it never leaves the rail and backwards travel earns ~0;
  11.   settle gate     — the EXACT goal state built with an injected 4 rad/s spin is
                          delivered() but NOT success while moving (sustained
                          stillness is load-bearing; sampled every substep, removed
                          before it can settle);
  12.   latched credit  — teleporting the ring away after the goal-pose probe keeps
                          the latched course credit (the cap, 0.85) while delivered()
                          drops — credit never evaporates, and an abandoned state
                          never becomes success;
  13.   rejection audit — success() was never True at ANY judged point;
  14.   final no-NaN    — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.put_toilet_roll_on_stand_i342.smoke --headless
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
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod

RAIL_H = scene_mod.RAIL_H
HANG_DROP = scene_mod.HANG_DROP
LAND_Z = scene_mod.LAND_Z
POST_XY = scene_mod.POST_XY
RING_HALF = scene_mod.RING_HALF
CAP_R = scene_mod.CAP_R
BORE_R = scene_mod.BORE_R
PATH_LOCAL = scene_mod.PATH_LOCAL
S_AT = scene_mod.S_AT
S_TOTAL = scene_mod.S_TOTAL
path_s_dist = scene_mod.path_s_dist
path_point = scene_mod.path_point
path_tangent = scene_mod.path_tangent

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.rail_ring")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.50, -1.30, 1.00)) + o),
                                tuple(np.array((0.35, 0.05, 0.12)) + o),
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

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def ring_read() -> tuple[float, float, float, float, float]:
        """(s, dist, z_local, dpost, tilt_cos) of the ring in the rail frame."""
        p, a = scene.ring_in_rail()
        s, d = path_s_dist(p)
        post = torch.tensor([POST_XY[0], POST_XY[1]], device=device)
        return (float(s[0]), float(d[0]), float(p[0, 2]),
                float((p[0, 0:2] - post).norm()), float(a[0, 2].abs()))

    def report(tag: str) -> None:
        s, ok = judge()
        sv, dv, zv, dp, tc = ring_read()
        print(f"[smoke] {tag:16s} | s={sv:.3f}/{S_TOTAL:.3f} dist={dv:.4f} "
              f"z={zv:.3f} dpost={dp:.3f} tilt_cos={tc:.3f} "
              f"del={bool(scene.delivered()[0])} set={bool(scene.settled()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, xyz, quat=None, vel=None, angvel=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = (xyz if torch.is_tensor(xyz)
                      else torch.tensor([float(v) for v in xyz], device=device))
        if quat is None:
            st[:, 3] = 1.0
        elif torch.is_tensor(quat):
            st[:, 3:7] = quat
        else:
            st[:, 3:7] = torch.tensor([float(v) for v in quat], device=device)
        if vel is not None:
            st[:, 7:10] = torch.tensor([float(v) for v in vel], device=device)
        if angvel is not None:
            st[:, 10:13] = torch.tensor([float(v) for v in angvel], device=device)
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def wrench(body, f_w: torch.Tensor) -> None:
        """WORLD force at the CoM, expressed in the body's link frame (the house
        convention — is_global drops torques on this stack)."""
        from isaaclab.utils.math import quat_apply_inverse

        q = body.data.root_link_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f_w).unsqueeze(1),
            torch.zeros(n, 1, 3, device=device), env_ids=all_ids)

    def rail_frame() -> tuple[torch.Tensor, torch.Tensor, float]:
        p = rel(scene.rail)
        q = scene.rail.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return p, q, yaw

    def rail_local(loc) -> torch.Tensor:
        """Env-rel world position of a rail-local point."""
        p, q, _ = rail_frame()
        return p + quat_apply(q.view(1, 4), torch.tensor([loc], device=device))[0]

    def hang_at(s_val: float, settle_steps: int = 45) -> None:
        """Hang the ring concentric on the course at station s (axis on the local
        tangent, it falls HANG_DROP into the hang)."""
        _p, q, _y = rail_frame()
        s_t = torch.full((1,), float(s_val), device=device)
        pt = path_point(s_t, device)[0]
        tg = path_tangent(s_t, device)[0]
        qzt = torch.tensor(scene_mod._quat_z_to([float(v) for v in tg]), device=device)
        quat = quat_mul(q.view(1, 4), qzt.view(1, 4))[0]
        place(scene.ring, rail_local([float(v) for v in pt]), quat=quat,
              settle_steps=settle_steps)

    def layout_sane(tag: str) -> bool:
        """Reset honesty: rail near nominal (jitter band), the ring HANGING on the
        first run inside the progress gate at its sampled station."""
        rp, _q, yaw = rail_frame()
        sv, dv, zv, _dp, tc = ring_read()
        jf = c.rail_jitter + 0.006
        ok = (abs(float(rp[0]) - c.rail_pos[0]) < jf
              and abs(float(rp[1]) - c.rail_pos[1]) < jf
              and abs(yaw) < math.radians(c.rail_yaw_deg) + 0.03
              and dv < c.prog_gate_dist
              and c.start_min - 0.025 < sv < c.start_max + 0.025
              and abs(zv - (RAIL_H - HANG_DROP)) < 0.012
              and tc < 0.35)
        if not ok:
            print(f"[smoke] layout sanity VIOLATION at {tag}: "
                  f"rail=({float(rp[0]):+.3f},{float(rp[1]):+.3f}) yaw={yaw:+.2f} "
                  f"s={sv:.3f} dist={dv:.4f} z={zv:.3f} tilt_cos={tc:.2f}", flush=True)
        return ok

    bodies = (scene.rail, scene.ring)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("settle: all states finite; the ring hangs on the first run inside the "
          "progress gate, rail near nominal", fin and layout_sane("reset"))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    sane = True
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(30)
        sane = sane and layout_sane(f"seed {sd}")
        rp, _q, yaw = rail_frame()
        sv, dv, _zv, _dp, _tc = ring_read()
        reads.append((float(rp[0]), float(rp[1]), yaw, sv, dv))
    arr = np.array(reads)
    print("[smoke] randomization readback (rail x, y, yaw | ring station s, dist):\n"
          f"{np.round(arr, 4)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: the rail fixture pose varies (spreads "
          f"x={spread[0]:.3f} y={spread[1]:.3f} yaw={spread[2]:.2f} rad)",
          spread[0] > 0.02 and spread[1] > 0.02 and spread[2] > 0.15)
    check("randomization: the ring start station varies (spread "
          f"{spread[3]:.3f} m within [{c.start_min:.3f},{c.start_max:.3f}]) and "
          "every reset hangs sane", spread[3] > 0.02 and sane)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. mid-course hang: partial credit only ====================
    env.reset(seed=41)
    step(30)
    hang_at(0.210, settle_steps=60)
    report("mid-course")
    s, ok = judge()
    sv, dv, _zv, _dp, _tc = ring_read()
    check("mid-course hang: the ring a third of the way along the course earns "
          f"PARTIAL latched credit only (score={s:.3f}, on-rail dist={dv * 1000:.0f} mm)"
          " — no success, far from the cap",
          0.02 < s < 0.50 and dv < c.prog_gate_dist
          and not bool(scene.delivered()[0]) and not ok)

    # =========================== 7. near-crest hang: graded, not delivered ==================
    env.reset(seed=51)
    step(30)
    hang_at(0.410, settle_steps=60)
    report("near-crest")
    s, ok = judge()
    _sv, dv, _zv, dp, _tc = ring_read()
    check("near-crest hang: the ring near the end of the second run (the delivery "
          f"near-miss) reads graded credit {s:.3f} in (0.30, cap) — still "
          f"{dp * 1000:.0f} mm from the post axis, NOT delivered, no success",
          0.30 <= s <= c.prog_cap + 0.001 and dp > 0.05
          and not bool(scene.delivered()[0]) and not ok)

    # =========================== 8. flat on the plate BESIDE the post =======================
    # Upright, at the delivered height, ON the plate — but the post is not through
    # the bore. Only the encircling xy clause rejects it: every around-the-post rest
    # reads <= ~18 mm, every beside-the-post rest reads >= ~42 mm.
    env.reset(seed=61)
    step(30)
    _p, q_r, _y = rail_frame()
    place(scene.ring, rail_local([POST_XY[0] + 0.048, POST_XY[1], LAND_Z + 0.002]),
          quat=q_r, settle_steps=60)
    report("beside-post")
    s, ok = judge()
    _sv, _dv, zv, dp, tc = ring_read()
    check("beside the post: the ring flat ON the plate at the right height "
          f"(z={zv * 1000:.0f} mm, tilt_cos={tc:.2f}) but {dp * 1000:.0f} mm off the "
          f"post axis (tol {c.deliver_xy_tol * 1000:.0f} mm) is NOT delivered — the "
          "post-through-the-bore clause is load-bearing",
          dp > c.deliver_xy_tol + 0.005 and abs(zv - LAND_Z) < 0.008 and tc > 0.9
          and not bool(scene.delivered()[0]) and not ok)

    # =========================== 9. leaning against the post ================================
    env.reset(seed=71)
    step(30)
    _p, q_r, _y = rail_frame()
    th = math.radians(40.0)
    q_tilt = torch.tensor([math.cos(th / 2), 0.0, math.sin(th / 2), 0.0], device=device)
    q_lean = quat_mul(q_r.view(1, 4), q_tilt.view(1, 4))[0]
    place(scene.ring, rail_local([POST_XY[0] + 0.034, POST_XY[1], 0.055]),
          quat=q_lean, settle_steps=90)
    report("leaning")
    s, ok = judge()
    _sv, _dv, zv, dp, tc = ring_read()
    check("leaning on the post: the tilted ring dropped against the post settles "
          f"NOT delivered (dpost={dp * 1000:.0f} mm, tilt_cos={tc:.2f}, "
          f"z={zv * 1000:.0f} mm) — no success, and it stayed in the probe area",
          not bool(scene.delivered()[0]) and not ok and dp < 0.10)

    # =========================== 10. captivity: the cap blocks the pull =====================
    # PHYSICAL probe: hang the ring at station 0.16 and DRAG it back toward the red
    # cap with a regulated velocity-capped pull. It must really travel (~100 mm),
    # then the cap blocks it; it never leaves the rail; backwards travel earns ~0.
    env.reset(seed=81)
    step(30)
    hang_at(0.160, settle_steps=45)
    s_start = ring_read()[0]
    # The hang itself may sit slightly AHEAD of this seed's sampled start station
    # and latch a sliver of forward credit (that is check 6's subject, and it is
    # honest). What captivity must show is that the backward pull earns NOTHING
    # on top of it: record the pre-pull score and assert it never grows.
    score_pre = float(scene.score()[0])
    _p, q_r, _y = rail_frame()
    s_min = s_start
    for _i in range(600):
        p_l, _a = scene.ring_in_rail()
        s_t, _d = path_s_dist(p_l)
        t_w = quat_apply(q_r.view(1, 4), path_tangent(s_t, device))
        v = scene.ring.data.root_lin_vel_w
        v_along = (v * t_w).sum(-1, keepdim=True)
        fmag = (3.0 * (-0.10 - v_along)).clamp(-0.6, 0.3)
        wrench(scene.ring, t_w * fmag)
        env.step(no_action)
        s_min = min(s_min, ring_read()[0])
    wrench(scene.ring, torch.zeros(n, 3, device=device))
    step(30)
    report("cap-blocked")
    s, ok = judge()
    sv, dv, _zv, _dp, _tc = ring_read()
    check("captivity: the regulated pull dragged the ring "
          f"{(s_start - s_min) * 1000:.0f} mm back toward the cap (not vacuous), the "
          f"cap BLOCKED it at s={s_min * 1000:.0f} mm > 0 (it never left the rail, "
          f"dist={dv * 1000:.0f} mm), and backwards travel earned NOTHING "
          f"(score {score_pre:.3f} -> {s:.3f}, small)",
          (s_start - s_min) > 0.050 and s_min > 0.012 and dv < c.prog_gate_dist
          and s <= score_pre + 1e-3 and s <= 0.16 and not ok)

    # =========================== 11-12. settle gate + latched credit ========================
    # The TRUE goal state, built with an injected 4 rad/s spin: delivered() but NOT
    # success while moving. Removed before it can settle (this battery must never
    # reach success) — the latched course credit survives, delivered() drops.
    env.reset(seed=91)
    step(30)
    _p, q_r, _y = rail_frame()
    place(scene.ring, rail_local([POST_XY[0], POST_XY[1], LAND_Z + 0.002]),
          quat=q_r, angvel=(0.0, 0.0, 4.0), settle_steps=0)
    gate_ok, w_peak = False, 0.0
    for _ in range(8):
        step(1)
        w_now = float(scene.ring.data.root_ang_vel_w[0].norm())
        w_peak = max(w_peak, w_now)
        _s_b, ok = judge()
        if (bool(scene.delivered()[0]) and w_now > c.settle_ang
                and not bool(scene.settled()[0]) and not ok):
            gate_ok = True
    report("goal-spinning")
    s_before, ok = judge()
    gate_ok = gate_ok and not ok and not bool(scene.settled()[0])
    check("settle gate: the exact goal state still SPINNING (peak "
          f"{w_peak:.1f} rad/s > settle_ang) is delivered() but NOT success — "
          "sustained stillness is load-bearing", gate_ok)
    place(scene.ring, (0.90, 0.60, RING_HALF + 0.002), quat=None, settle_steps=40)
    report("moved-away")
    s_after, ok = judge()
    check("latched credit: teleporting the ring away keeps the latched course "
          f"credit ({s_before:.2f} -> {s_after:.2f}, cap {c.prog_cap:.2f}) while "
          "delivered() drops — and an abandoned goal state never succeeds",
          s_after >= c.prog_cap - 0.011 and s_after <= c.prog_cap + 0.001
          and s_after >= s_before - 1e-4
          and not bool(scene.delivered()[0]) and not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.rail_ring")
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

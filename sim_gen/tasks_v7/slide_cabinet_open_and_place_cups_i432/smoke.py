"""Smoke / rubric-REJECTION battery for FerryDoorCabinetScene (sim_gen task
`slide_cabinet_open_and_place_cups_i432`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — open, drop cup_a through the port, closing stroke
rams it into the bay, re-open, drop cup_b, final chain-ram stroke, ring-down — is the
acceptance evidence; it passes on forge seeds 0/1/2). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome and asserts the rubric
REJECTS it — plus physical probes that prove the ferry door is a working captive-piston
mechanism, not a prop. No probe in this battery ever reaches success(), and a final
audit check asserts exactly that.

   1. settle/no-NaN      — reset settles finite, cups upright on the ground, door on
                           its track, everything reads settled;
   2. fresh reset        — score ~0, no success;
  3-4. randomization     — READBACK over 6 seeded resets: the door's start position
                           varies across the stroke (and stays on the track); the cup
                           slot permutation varies and the xy jitter is real;
   5. null policy        — 240 idle steps -> score ~0, no success;
   6. sealed port        — the solve's own transport move over a port the door COVERS:
                           the cup lands ON the door top (height readback), never
                           reaches the floor -> not staged, not in the bay, score ~0
                           (the door physically denies the only entrance);
   7. roof denial        — a cup rested on the ROOF directly over the bay's y window:
                           inside the footprint, z band rejects it (the bay is only
                           reachable through the shell, never from above);
   8. hard stops         — the solve's velocity-regulated push aimed 36 mm BEYOND the
                           +stop drives the door across its full track (travel
                           readback) and the stop CLAMPS it (never past stroke+4 mm);
   9. no spring          — the door parked MID-track drifts < 2 mm over 240 hands-off
                           steps (it stays where it is left — openness is free to
                           maintain, unlike a gravity gate, but useless by itself);
  10. seed-analog        — the seed family's plan (open the door once, place the cups
                           in the revealed volume, walk away): both cups on the
                           corridor floor under the open port -> staged credit only,
                           nothing in the bay, door open -> NOT success, score <= 0.22;
  11. door-open near-miss— both cups correctly in the bay but the door parked ~28 mm
                           short of the closed band -> NOT success, latched score ~0.5;
  12. latched credit     — removing a bay cup leaves the latched score unchanged
                           while both_seated() correctly drops;
  13. doorway loiterer   — a cup at rest between the bay band and the staged band
                           (y=-0.030) is in NEITHER (membership windows are tight);
  14. lying cup          — a cup on its side in the bay: the z band PASSES (readback
                           — floor height) and in_bay() reads True, yet uprightness
                           rejects seating (the uprightness clause is load-bearing);
  15. stacked cups       — under the open port, a cup stacked on a staged cup reads
                           ~60 mm high -> rejected by the z band (the base cup stays
                           staged);
  16. settle gate        — both cups in the bay and the door driven into the closed
                           band, judged the INSTANT door_closed first reads True while
                           the door is still MOVING -> NOT success (stillness must
                           persist); a cup is removed before ring-down (the battery
                           never succeeds);
  17. rejection audit    — success() was never True at ANY judged point;
  18. final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.slide_cabinet_open_and_place_cups_i432.smoke --headless
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
    env = ENVS.get("simgen.ferry_door_cabinet")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((0.85, -0.60, 0.55)) + o),
                                tuple(np.array((0.00, 0.05, 0.05)) + o),
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

    def door_d() -> float:
        return float(scene.door_d()[0])

    def cup_loc(nm: str) -> tuple[float, float, float]:
        p = scene._cup_local(nm)[0]
        return float(p[0]), float(p[1]), float(p[2])

    def report(tag: str) -> None:
        s, ok = judge()
        bits = []
        for nm in c.cup_names:
            x, y, z = cup_loc(nm)
            bits.append(f"{nm}=({x:+.3f},{y:+.3f},{z:+.3f})c "
                        f"stg={bool(scene.staged(nm)[0])} bay={bool(scene.seated(nm)[0])}")
        print(f"[smoke] {tag:16s} | d={door_d():+.4f} cl={bool(scene.door_closed()[0])} | "
              + " ".join(bits) + f" score={s:.3f} success={ok} frames={len(frames)}",
              flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    UP = (1.0, 0.0, 0.0, 0.0)

    def write_cup(nm: str, local_xyz, quat=UP, settle_steps: int = 60) -> None:
        """Probe placement in the CABINET frame (instrumentation, not a solution) +
        REAL physics steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.case_origin
        st[:, 0] += float(local_xyz[0])
        st[:, 1] += float(local_xyz[1])
        st[:, 2] += float(local_xyz[2])
        st[:, 3:7] = torch.tensor(quat, device=device)
        scene.cups[nm].write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def drop_cup(nm: str, y: float, settle_steps: int = 210) -> None:
        """The solve's transport move: release upright ABOVE the roof over cabinet-
        frame y; gravity decides what admits it."""
        write_cup(nm, (0.0, y, 0.125), UP, settle_steps)

    def to_ground(nm: str, x: float, y: float, settle_steps: int = 60) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = x
        st[:, 1] = y
        st[:, 2] = c.cup_h / 2 + 0.003
        st[:, 0:3] += scene.env_origins
        st[:, 3] = 1.0
        scene.cups[nm].write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def door_to(d: float, settle_steps: int = 30) -> None:
        """Teleport the door ALONG ITS TRACK (within limits — instrumentation)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.case_origin
        st[:, 1] += c.track_y0 + float(d)
        st[:, 2] += c.track_z
        st[:, 3] = 1.0
        scene.door.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    bodies = (scene.case, scene.door, *scene.cups.values())
    port_mid = (c.port_y0 + c.port_y1) / 2

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    z_ok = all(abs(float(scene.cups[nm].data.root_pos_w[0, 2]) - c.cup_h / 2) < 0.012
               for nm in c.cup_names)
    on_track = abs(door_d()) <= c.half_stroke + 0.003
    check("settle: all states finite, cups upright on the ground, door on its track "
          f"(d={door_d():+.4f}), everything settled",
          fin and z_ok and on_track and bool(scene.settled()[0]))
    s, ok = judge()
    check("fresh reset: score ~0, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(20)
        d0 = door_d()
        ys = [cup_loc(nm)[1] for nm in c.cup_names]
        xs = [cup_loc(nm)[0] for nm in c.cup_names]
        slot = tuple(int(np.argmin([abs(y - sy) for sy in c.slot_ys])) for y in ys)
        reads.append((d0, slot, xs[0], ys[0]))
        print(f"[smoke] seed {sd}: door d0={d0:+.4f} slots={slot} "
              f"cup_a_xy=({xs[0]:+.3f},{ys[0]:+.3f})", flush=True)
    d0s = [r[0] for r in reads]
    d0_spread = max(d0s) - min(d0s)
    check("randomization: door start position varies across seeded resets "
          f"(READBACK spread {d0_spread * 1000:.0f} mm) and stays on the track",
          d0_spread > 0.015 and all(abs(v) <= c.d0_range + 0.004 for v in d0s))
    slots = {r[1] for r in reads}
    x_spread = max(r[2] for r in reads) - min(r[2] for r in reads)
    check("randomization: cup slot permutation varies (readback: "
          f"{len(slots)} distinct / 6), slots never shared, and xy jitter is real "
          f"(x spread {x_spread * 1000:.0f} mm)",
          len(slots) >= 3 and all(r[1][0] != r[1][1] for r in reads) and x_spread > 0.01)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 6. the covered port physically denies entry ================
    env.reset(seed=41)
    step(30)
    door_to(-0.052, settle_steps=30)  # door spans y [-0.006, +0.114] -> port covered
    drop_cup("cup_a", port_mid, settle_steps=210)
    report("sealed-port")
    _xa, _ya, za = cup_loc("cup_a")
    s, ok = judge()
    door_top = c.track_z + c.door_h / 2  # +0.076; cup ON it rests ~+0.106
    check("sealed port: the solve's transport over a COVERED port lands the cup ON "
          f"the door top (z={za:+.3f}c vs floor rest {c.cup_rest_z:+.3f}, door top "
          f"{door_top:+.3f}) — not staged, not in the bay, score ~0",
          za > door_top + 0.015 and not bool(scene.staged("cup_a")[0])
          and not bool(scene.in_bay("cup_a")[0]) and s <= 0.02 and not ok)

    # =========================== 7. roof denial over the bay ================================
    roof_top = c.floor_t + c.inner_h + c.roof_t
    write_cup("cup_b", (0.0, -0.100, roof_top + c.cup_h / 2 + 0.004), UP,
              settle_steps=120)
    report("roof-top")
    _xb, yb, zb = cup_loc("cup_b")
    s, ok = judge()
    check("roof denial: a cup rested on the ROOF directly over the bay's y window "
          f"(y={yb:+.3f}, z={zb:+.3f}c vs floor rest {c.cup_rest_z:+.3f}) is inside "
          "the footprint but rejected by the z band — the bay is never top-loadable",
          zb > c.cup_rest_z + 0.05 and c.bay_y_lo <= yb <= c.bay_y_hi
          and not bool(scene.in_bay("cup_b")[0]) and s <= 0.02 and not ok)

    # =========================== 8. mechanism reality: track + hard stop ====================
    env.reset(seed=51)
    step(30)
    door_to(-0.052, settle_steps=30)
    d_start = door_d()
    f = torch.zeros(n, 1, 3, device=device)
    max_d = d_start
    for _ in range(700):  # velocity-regulated push aimed 36 mm BEYOND the +stop
        d = scene.door_d()
        v = scene.door.data.root_lin_vel_w[:, 1]
        v_des = (3.0 * (0.100 - d)).clamp(-0.10, 0.10)
        f[:, 0, 1] = (12.0 * (v_des - v) + 0.9).clamp(-4.0, 4.0)
        scene.door.set_external_force_and_torque(f, zero_w, env_ids=all_ids)
        step(1)
        max_d = max(max_d, door_d())
    scene.door.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
    step(60)
    d_end = door_d()
    report("stop-clamp")
    _s, _ok = judge()
    check("hard stops: a regulated push aimed 36 mm BEYOND the +stop drives the door "
          f"{(d_end - d_start) * 1000:.0f} mm along its track ({d_start:+.4f} -> "
          f"{d_end:+.4f}) and the stop CLAMPS it (max {max_d:+.4f} <= stroke+4 mm)",
          (d_end - d_start) > 0.10 and max_d <= c.half_stroke + 0.004
          and d_end >= c.half_stroke - 0.007)

    # =========================== 9. no spring: the door stays where it is left ==============
    door_to(+0.010, settle_steps=10)
    d_ref = door_d()
    step(240)
    d_drift = abs(door_d() - d_ref)
    report("no-spring")
    check("no spring: the door parked MID-track (d="
          f"{d_ref:+.4f}) drifts only {d_drift * 1000:.1f} mm over 240 hands-off "
          "steps — it stays where it is left",
          d_drift < 0.002)

    # =========================== 10. seed-analog: open once, place, walk away ===============
    env.reset(seed=61)
    step(30)
    door_to(+c.half_stroke - 0.002, settle_steps=30)  # "the door is open"
    write_cup("cup_a", (0.0, 0.002, c.cup_rest_z + 0.006), UP, settle_steps=90)
    write_cup("cup_b", (0.0, 0.062, c.cup_rest_z + 0.006), UP, settle_steps=240)
    report("seed-analog")
    s, ok = judge()
    check("seed-analog (open the door, place the cups in the revealed volume, walk "
          "away): both cups on the corridor floor under the open port -> staged "
          f"credit only (score {s:.2f} <= 0.22), NOTHING in the bay, door open -> "
          "NOT success",
          not bool(scene.in_bay("cup_a")[0]) and not bool(scene.in_bay("cup_b")[0])
          and bool(scene.staged("cup_a")[0]) and bool(scene.staged("cup_b")[0])
          and s <= 0.22 and not ok)

    # =========================== 11. door-open near-miss ====================================
    env.reset(seed=71)
    step(30)
    write_cup("cup_a", (0.0, -0.102, c.cup_rest_z + 0.006), UP, settle_steps=60)
    write_cup("cup_b", (0.0, -0.046, c.cup_rest_z + 0.006), UP, settle_steps=60)
    door_to(-0.030, settle_steps=90)  # ~28 mm short of the closed band
    report("open-miss")
    s, ok = judge()
    both = bool(scene.both_seated()[0])
    check("door-open near-miss: both cups correctly in the bay but the door parked "
          f"~28 mm short of the closed band (d={door_d():+.4f} > -0.058) -> NOT "
          f"success, latched score {s:.2f}",
          both and not bool(scene.door_closed()[0]) and not ok and 0.45 <= s <= 0.55)

    # =========================== 12. latched credit survives removal ========================
    s_before, _ = judge()
    to_ground("cup_b", 0.30, 0.30, settle_steps=90)
    report("removed")
    s_after, ok = judge()
    check("latched credit: removing a bay cup leaves the latched score unchanged "
          f"({s_before:.2f} -> {s_after:.2f}) while both_seated() drops",
          abs(s_after - s_before) < 1e-3 and not bool(scene.both_seated()[0]) and not ok)

    # =========================== 13. doorway loiterer =======================================
    write_cup("cup_b", (0.0, -0.030, c.cup_rest_z + 0.006), UP, settle_steps=90)
    report("loiterer")
    _xl, yl, _zl = cup_loc("cup_b")
    s, ok = judge()
    check("doorway loiterer: a cup at rest between the bay band and the staged band "
          f"(y={yl:+.3f}, bands end at {c.bay_y_hi:+.3f} / start at "
          f"{c.stage_y_lo:+.3f}) is in NEITHER window",
          not bool(scene.in_bay("cup_b")[0]) and not bool(scene.staged("cup_b")[0])
          and c.bay_y_hi < yl < c.stage_y_lo and not ok)

    # =========================== 14. lying cup: z band passes, uprightness rejects ==========
    env.reset(seed=81)
    step(30)
    LYING = (0.70710678, 0.70710678, 0.0, 0.0)  # 90 deg about x — axis along y
    write_cup("cup_a", (0.0, -0.100, c.floor_t + c.cup_r + 0.004), LYING,
              settle_steps=120)
    report("lying")
    _x, _y, zl = cup_loc("cup_a")
    z_in_band = abs(zl - c.cup_rest_z) <= c.z_tol
    s, ok = judge()
    check("lying cup in the bay: the z band PASSES (readback "
          f"z={zl:+.3f}, band +/-{c.z_tol:.3f} about {c.cup_rest_z:+.3f}) and "
          f"in_bay reads {bool(scene.in_bay('cup_a')[0])}, yet uprightness rejects "
          "seating — the uprightness clause is load-bearing",
          z_in_band and bool(scene.in_bay("cup_a")[0])
          and not bool(scene.cup_upright("cup_a")[0])
          and not bool(scene.seated("cup_a")[0]) and not ok and s <= 0.02)

    # =========================== 15. stacked cups under the port ============================
    door_to(+c.half_stroke - 0.002, settle_steps=20)  # port exposed
    write_cup("cup_b", (0.0, port_mid, c.cup_rest_z + 0.006), UP, settle_steps=60)
    write_cup("cup_a", (0.0, port_mid, c.cup_rest_z + c.cup_h + 0.008), UP,
              settle_steps=150)
    report("stacked")
    _x, _y, z_stk = cup_loc("cup_a")
    s, ok = judge()
    check("stacked: under the open port, a cup stacked on a staged cup reads "
          f"z={z_stk:+.3f} (floor rest {c.cup_rest_z:+.3f}) -> rejected by the z "
          "band while the base cup stays staged",
          z_stk > c.cup_rest_z + 0.045 and not bool(scene.staged("cup_a")[0])
          and bool(scene.staged("cup_b")[0]) and not ok)

    # =========================== 16. settle gate ============================================
    env.reset(seed=91)
    step(30)
    door_to(+0.020, settle_steps=30)
    write_cup("cup_a", (0.0, -0.102, c.cup_rest_z + 0.006), UP, settle_steps=60)
    write_cup("cup_b", (0.0, -0.046, c.cup_rest_z + 0.006), UP, settle_steps=60)
    # Drive the door into the closed band and judge the INSTANT door_closed first
    # reads True — the door is still MOVING, so stillness must reject success.
    f = torch.zeros(n, 1, 3, device=device)
    crossed, v_at_cross, gate_reject = False, 0.0, False
    for _ in range(500):
        f[:, 0, 1] = -0.8
        scene.door.set_external_force_and_torque(f, zero_w, env_ids=all_ids)
        step(1)
        if bool(scene.door_closed()[0]):
            crossed = True
            v_at_cross = abs(float(scene.door.data.root_lin_vel_w[0, 1]))
            _s, ok_now = judge()
            gate_reject = not ok_now
            break
    scene.door.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
    report("settle-gate")
    # remove a cup BEFORE ring-down completes (the battery must never succeed)
    to_ground("cup_b", 0.30, -0.30, settle_steps=60)
    check("settle gate: the completed arrangement judged the instant door_closed "
          f"first reads True (door still moving, |v|={v_at_cross:.3f} > "
          f"{c.settle_door}) is NOT success; a cup was removed before ring-down "
          "(battery never succeeds)",
          crossed and v_at_cross > c.settle_door and gate_reject
          and not bool(scene.success()[0]))

    # =========================== 17-18. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.ferry_door_cabinet")
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

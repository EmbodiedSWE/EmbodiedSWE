"""Smoke / rubric-REJECTION battery for ShutterCabinetScene (sim_gen task
`slide_cabinet_open_and_place_cups_i112`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — shutter to the -stop, blue cup through the exposed
blue opening, shutter to the +stop, both red cups through the exposed red opening,
ring-down — is the acceptance evidence; it passes on forge seeds 0/1/2). Every teleport
here is instrumentation that CONSTRUCTS a wrong (or partial) outcome and asserts the
rubric REJECTS it — plus physical probes that prove the bypass shutter is a working
mutual-exclusion mechanism, not a prop. No probe in this battery ever reaches
success(), and a final audit check asserts exactly that.

   1. settle/no-NaN      — reset settles finite, cups upright on the floor, shutter on
                           its track, everything reads settled;
   2. fresh reset        — score ~0, no success;
  3-4. randomization     — READBACK over 6 seeded resets: the shutter's start position
                           varies across the stroke (and stays on the track); the cup
                           slot permutation varies and the xy jitter is real;
   5. null policy        — 240 idle steps -> score ~0, no success;
   6. coverage gating    — EMPTY cabinet with the shutter teleported over the blue
                           opening: covers_blue() reads True yet score stays ~0 (the
                           coverage credit is gated on all-seated — a shutter that
                           merely starts there earns nothing);
   7. sealed opening     — a cup dropped over the COVERED blue opening lands ON the
                           shutter plate (height readback), never reaches any bay ->
                           not seated, score ~0 (the cover physically denies access);
   8. mutual exclusion   — shutter parked MID-track, one cup dropped over the most-
                           open strip of EACH opening: a wedging drop may SHOVE the
                           free shutter aside and fall in (readback: the admission
                           coincides with a large shutter displacement), but doing so
                           SEALS the other opening — the second drop is denied and
                           rides the shutter. At most ONE side ever admits;
   9. mechanism reality  — the same velocity-regulated push the solve uses, aimed
                           BEYOND the +stop, drives the shutter across its full track
                           (travel readback) and the hard stop CLAMPS it (never past
                           stroke + 4 mm); released, it parks covering blue;
  10. seed-analog        — the seed family's plan (open one door, put every cup in the
                           one revealed volume): all three cups through the exposed
                           RED opening -> reds seat but the blue cup is in the wrong
                           bay -> NOT success, score <= 0.35;
  11. lying cup          — the blue cup on its side on the blue pad: the z band PASSES
                           (readback — floor height) but uprightness rejects it;
  12. z-band rejections  — a red cup STACKED on the other red (readback ~65 mm high)
                           and then rested on the solid top plate inside the bay's y
                           window: both rejected by the z band;
  13. parking near-miss  — all three cups correctly seated but the shutter parked
                           18 mm short of the cover band -> covers_blue False, NOT
                           success, score reads the latched 0.60;
  14. latched credit     — removing the blue cup leaves the latched score unchanged
                           while all_seated() correctly drops;
  15. settle gate        — the full success state judged while the last cup is still
                           FALLING (mid-drop) is NOT success; the cup is removed
                           before ring-down completes (the battery never succeeds);
  16. rejection audit    — success() was never True at ANY judged point;
  17. final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.slide_cabinet_open_and_place_cups_i112.smoke --headless
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
    env = ENVS.get("simgen.shutter_cabinet")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.05, -0.85, 0.72)) + o),
                                tuple(np.array((0.08, 0.00, 0.08)) + o),
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

    def shutter_c() -> float:
        return float(scene.shutter_c()[0])

    def cup_loc(nm: str) -> tuple[float, float, float]:
        p = scene._cup_local(nm)[0]
        return float(p[0]), float(p[1]), float(p[2])

    def report(tag: str) -> None:
        s, ok = judge()
        bits = []
        for nm in c.cup_names:
            x, y, z = cup_loc(nm)
            bits.append(f"{nm}=({x:+.3f},{y:+.3f},{z:+.3f})c st={bool(scene.seated(nm)[0])}")
        print(f"[smoke] {tag:16s} | c={shutter_c():+.4f} "
              f"cov={bool(scene.covers_blue()[0])} | " + " ".join(bits)
              + f" score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def write_cup(nm: str, local_xyz, quat, settle_steps: int = 60) -> None:
        """Kinematic probe placement in the CABINET frame (instrumentation, not a
        solution) + REAL physics steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.case_origin
        st[:, 0] += float(local_xyz[0])
        st[:, 1] += float(local_xyz[1])
        st[:, 2] += float(local_xyz[2])
        st[:, 3:7] = torch.tensor(quat, device=device)
        scene.cups[nm].write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    UP = (1.0, 0.0, 0.0, 0.0)

    def drop_cup(nm: str, x: float, y: float, z: float = 0.06,
                 settle_steps: int = 210) -> None:
        """The solve's transport move: release upright ABOVE the top plate; gravity
        decides whether the opening admits it."""
        write_cup(nm, (x, y, z), UP, settle_steps)

    def seat_cup(nm: str, x: float, y: float, settle_steps: int = 60) -> None:
        """Direct settled placement on a bay floor (rubric-probe construction)."""
        write_cup(nm, (x, y, c.cup_rest_z + 0.010), UP, settle_steps)

    def to_ground(nm: str, x: float, y: float, settle_steps: int = 60) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = x
        st[:, 1] = y
        st[:, 2] = c.cup_h / 2 + 0.003
        st[:, 0:3] += scene.env_origins
        st[:, 3] = 1.0
        scene.cups[nm].write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def shutter_to(cpos: float, settle_steps: int = 30) -> None:
        """Teleport the shutter ALONG ITS TRACK (within limits — instrumentation)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.case_origin
        st[:, 1] += float(cpos)
        st[:, 2] += c.shutter_hover
        st[:, 3] = 1.0
        scene.shutter.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    bodies = (scene.case, scene.shutter, *scene.cups.values())

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    z_ok = all(abs(float(scene.cups[nm].data.root_pos_w[0, 2]) - c.cup_h / 2) < 0.012
               for nm in c.cup_names)
    on_track = abs(shutter_c()) <= c.stroke + 0.003
    check("settle: all states finite, cups upright on the floor, shutter on its track "
          f"(c={shutter_c():+.4f}), everything settled",
          fin and z_ok and on_track and bool(scene.settled()[0]))
    s, ok = judge()
    check("fresh reset: score ~0, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(20)
        c0 = shutter_c()
        ys = [cup_loc(nm)[1] for nm in c.cup_names]
        xs = [cup_loc(nm)[0] for nm in c.cup_names]
        slot = tuple(int(np.argmin([abs(y - sy) for sy in c.slot_ys])) for y in ys)
        reads.append((c0, slot, xs[0], ys[0]))
        print(f"[smoke] seed {sd}: shutter c0={c0:+.4f} slots={slot} "
              f"red_a_xy=({xs[0]:+.3f},{ys[0]:+.3f})", flush=True)
    c0s = [r[0] for r in reads]
    c0_spread = max(c0s) - min(c0s)
    check("randomization: shutter start position varies across seeded resets "
          f"(READBACK spread {c0_spread * 1000:.0f} mm) and stays on the track",
          c0_spread > 0.015 and all(abs(v) <= c.c0_range + 0.004 for v in c0s))
    slots = {r[1] for r in reads}
    x_spread = max(r[2] for r in reads) - min(r[2] for r in reads)
    check("randomization: cup slot permutation varies (readback: "
          f"{len(slots)} distinct / 6) and xy jitter is real "
          f"(x spread {x_spread * 1000:.0f} mm)", len(slots) >= 3 and x_spread > 0.01)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 6. coverage credit is gated on seats =======================
    env.reset(seed=41)
    step(30)
    shutter_to(+0.052, settle_steps=60)
    report("covered-empty")
    s, ok = judge()
    check("coverage gating: EMPTY cabinet with the shutter parked over blue — "
          f"covers_blue reads {bool(scene.covers_blue()[0])} yet score ~0, no success "
          "(coverage credit demands all cups seated first)",
          bool(scene.covers_blue()[0]) and s <= 0.02 and not ok)

    # =========================== 7. a sealed opening physically denies access ==============
    # Same transport the solve uses, but over the COVERED blue opening.
    drop_cup("red_a", 0.0, +0.128, settle_steps=210)
    report("sealed-drop")
    xa, ya, za = cup_loc("red_a")
    s, ok = judge()
    on_shutter = za > 0.015  # shutter top is at +0.0135; bay floor rest is -0.0795
    check("sealed opening: a cup dropped over the covered blue opening rides ON the "
          f"shutter plate (z={za:+.3f} cabinet frame, floor rest {c.cup_rest_z:+.3f}) "
          "— not seated anywhere, score ~0",
          on_shutter and not bool(scene.seated("red_a")[0])
          and not bool(scene.in_bay("red_a", +1.0)[0])
          and not bool(scene.in_bay("red_a", -1.0)[0]) and s <= 0.02 and not ok)

    # =========================== 8. mutual exclusion at MID-track ===========================
    env.reset(seed=51)
    step(30)
    shutter_to(0.0, settle_steps=30)
    # Most-open strip of each opening at c=0: |y| in [0.130, 0.180] -> 50 mm < cup 56 mm.
    # The shutter is FREE: a cup wedging into the partial gap can shove it aside and
    # fall in — but the displaced shutter then SEALS the other opening. The invariant
    # under full dynamics is: at most ONE side ever admits.
    drop_cup("blue", 0.0, +0.155, settle_steps=240)
    drop_cup("red_a", 0.0, -0.155, settle_steps=240)
    report("mid-track")
    _s, ok = judge()
    c_end = shutter_c()
    admitted, denied_high = [], []
    for nm in ("blue", "red_a"):
        x, y, z = cup_loc(nm)
        in_any = bool(scene.in_bay(nm, +1.0)[0]) or bool(scene.in_bay(nm, -1.0)[0])
        inside_below = (abs(x) < c.half_x and abs(y) < c.half_y and z < -0.03)
        (admitted if (in_any or inside_below) else denied_high).append((nm, z))
        print(f"[smoke] mid-track {nm}: rest=({x:+.3f},{y:+.3f},{z:+.3f})c "
              f"admitted={in_any or inside_below}", flush=True)
    check("mutual exclusion under free-shutter dynamics: dropped over the most-open "
          "strip of EACH opening, at most ONE cup got in — any admission coincides "
          f"with a large shutter displacement (c: 0 -> {c_end:+.4f}) that seals the "
          "other opening, whose cup is denied and rides high "
          f"(admitted={[a[0] for a in admitted]}, denied z="
          f"{[f'{d[1]:+.3f}' for d in denied_high]})",
          len(admitted) <= 1 and all(z > 0.015 for _nm, z in denied_high)
          and (not admitted or abs(c_end) >= 0.040) and not ok)

    # =========================== 9. mechanism reality: track + hard stop ====================
    env.reset(seed=61)
    step(30)
    shutter_to(-0.052, settle_steps=60)
    c_start = shutter_c()
    f = torch.zeros(n, 1, 3, device=device)
    max_c = c_start
    for _ in range(700):  # velocity-regulated push aimed BEYOND the +stop
        cpos = scene.shutter_c()
        v = scene.shutter.data.root_lin_vel_w[:, 1]
        v_des = (3.0 * (0.10 - cpos)).clamp(-0.12, 0.12)
        f[:, 0, 1] = (6.0 * (v_des - v)).clamp(-3.0, 3.0)
        scene.shutter.set_external_force_and_torque(f, zero_w, env_ids=all_ids)
        step(1)
        max_c = max(max_c, shutter_c())
    scene.shutter.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
    step(60)
    c_end = shutter_c()
    report("stop-clamp")
    _s, _ok = judge()
    check("mechanism reality: a regulated push aimed 45 mm BEYOND the +stop drives "
          f"the shutter {(c_end - c_start) * 1000:.0f} mm along its track "
          f"({c_start:+.4f} -> {c_end:+.4f}) and the hard stop CLAMPS it "
          f"(max {max_c:+.4f} <= stroke+4 mm); released, it parks covering blue",
          (c_end - c_start) > 0.08 and max_c <= c.stroke + 0.004
          and bool(scene.covers_blue()[0]))

    # =========================== 10. seed-analog: everything into the one open bay =========
    env.reset(seed=71)
    step(30)
    shutter_to(+0.052, settle_steps=30)  # red opening exposed — "the door is open"
    drop_cup("red_a", -0.031, -0.105, settle_steps=180)
    drop_cup("red_b", +0.031, -0.105, settle_steps=180)
    drop_cup("blue", 0.0, -0.155, settle_steps=240)
    report("seed-analog")
    s, ok = judge()
    blue_in_red = bool(scene.in_bay("blue", -1.0)[0])
    check("seed-analog (open one door, place every cup in the revealed volume): all "
          "three through the exposed RED opening — the reds seat but the blue cup is "
          f"in the WRONG bay (in_red={blue_in_red}, seated={bool(scene.seated('blue')[0])}) "
          "-> NOT success, score <= 0.35",
          not bool(scene.seated("blue")[0]) and not ok and s <= 0.35)

    # =========================== 11. lying cup: z band passes, uprightness rejects =========
    env.reset(seed=81)
    step(30)
    LYING = (0.70710678, 0.70710678, 0.0, 0.0)  # 90 deg about x — axis along y
    write_cup("blue", (0.0, +c.ap_c, c.floor_z + c.cup_r + 0.006), LYING,
              settle_steps=120)
    report("lying")
    _x, _y, zl = cup_loc("blue")
    z_in_band = abs(zl - c.cup_rest_z) <= c.bay_z_tol
    s, ok = judge()
    check("lying cup on the blue pad: the z band PASSES (readback "
          f"z={zl:+.3f}, band +/-{c.bay_z_tol:.3f} about {c.cup_rest_z:+.3f}) yet "
          "uprightness rejects it -> not seated",
          z_in_band and not bool(scene.cup_upright("blue")[0])
          and not bool(scene.seated("blue")[0]) and not ok and s <= 0.02)

    # =========================== 12. z-band rejections: stacked + on the top plate =========
    seat_cup("red_a", -0.031, -c.ap_c, settle_steps=60)
    write_cup("red_b", (-0.031, -c.ap_c, c.cup_rest_z + c.cup_h + 0.008), UP,
              settle_steps=120)
    _x, _y, z_stk = cup_loc("red_b")
    stacked_rej = (not bool(scene.seated("red_b")[0])) and z_stk > c.cup_rest_z + 0.04
    write_cup("red_b", (0.0, -0.042, c.cup_h / 2 + 0.004), UP, settle_steps=90)
    _x, _y, z_top = cup_loc("red_b")
    report("z-band")
    s, ok = judge()
    check("z band: a red cup STACKED on the seated red reads "
          f"z={z_stk:+.3f} (floor rest {c.cup_rest_z:+.3f}) -> rejected; the same cup "
          f"rested on the SOLID top plate inside the bay's y window (z={z_top:+.3f}) "
          "-> rejected",
          stacked_rej and not bool(scene.seated("red_b")[0])
          and z_top > c.cup_rest_z + 0.04 and not ok)

    # =========================== 13. parking near-miss ======================================
    env.reset(seed=91)
    step(30)
    shutter_to(+0.030, settle_steps=30)  # 18 mm short of the cover band
    seat_cup("blue", 0.0, +c.ap_c)
    seat_cup("red_a", -0.031, -c.ap_c)
    seat_cup("red_b", +0.031, -c.ap_c, settle_steps=150)
    report("park-miss")
    s, ok = judge()
    all_seated = bool(scene.all_seated()[0])
    check("parking near-miss: ALL cups correctly seated but the shutter parked 18 mm "
          f"short of the cover band (c={shutter_c():+.4f} < {c.cover_c_min}) -> "
          f"covers_blue False, NOT success, latched score {s:.2f}",
          all_seated and not bool(scene.covers_blue()[0]) and not ok
          and 0.55 <= s <= 0.66)

    # =========================== 14. latched credit survives removal ========================
    s_before, _ = judge()
    to_ground("blue", 0.32, 0.30, settle_steps=90)
    report("removed")
    s_after, ok = judge()
    check("latched credit: removing the blue cup leaves the latched score unchanged "
          f"({s_before:.2f} -> {s_after:.2f}) while all_seated() drops",
          abs(s_after - s_before) < 1e-3 and not bool(scene.all_seated()[0]) and not ok)

    # =========================== 15. settle gate ============================================
    env.reset(seed=101)
    step(30)
    shutter_to(+0.052, settle_steps=30)
    seat_cup("blue", 0.0, +c.ap_c)
    seat_cup("red_a", -0.031, -c.ap_c)
    # the final cup — judged while STILL FALLING through the open red opening
    drop_cup("red_b", +0.031, -0.128, settle_steps=8)
    _x, _y, z_fall = cup_loc("red_b")
    s, ok = judge()
    falling_reject = not ok
    report("settle-gate")
    # remove it BEFORE ring-down completes (the battery must never reach success)
    to_ground("red_b", 0.32, -0.30, settle_steps=60)
    check("settle gate: the completed arrangement judged while the last cup is still "
          f"falling (z={z_fall:+.3f}, floor rest {c.cup_rest_z:+.3f}) is NOT success; "
          "the cup was removed before ring-down (battery never succeeds)",
          falling_reject and z_fall > c.cup_rest_z + c.bay_z_tol
          and not bool(scene.success()[0]))

    # =========================== 16-17. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.shutter_cabinet")
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

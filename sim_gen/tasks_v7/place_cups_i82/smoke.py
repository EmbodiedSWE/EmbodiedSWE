"""Smoke / rubric-REJECTION battery for BalanceScaleScene (sim_gen task
`place_cups_i82`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — drop big alone into one pan, mid+small side by side
into the other, let the beam ring down level — is the acceptance evidence that the
rubric ACCEPTS a correct outcome; it passes on seeds 0/1/2). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled state and
asserts the rubric REJECTS it — plus a mechanism-reality probe that proves the balance
is a working instrument (a one-sided load heels it far past level; removing the load
lets the pendulum restore level). No probe in this battery ever reaches success(), and
a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite, cups upright on the floor,
                            EMPTY BEAM RESTS LEVEL, score ~0, no success;
  3-4. randomization      — READBACK over 6 seeded resets: the two sampled masses vary
                            AND m_big = m_small + m_mid holds to solver precision every
                            time (through the PhysX view); the staging-slot permutation
                            and xy jitter vary;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed-analog         — the seed family's end state (one object per dedicated
                            target): one weight in EACH pan, third left on the floor ->
                            loaded() False, NOT success;
  7.  wrong partition     — {big, small} vs {mid}: all three riding pans but the beam
                            heels PAST tilt_max (readback) -> level() False, NOT
                            success, score <= 0.5 (physics is the judge);
  8.  all in one pan      — all three weights dumped into ONE pan -> beam hard over,
                            NOT success;
  9.  on the beam, level  — a weight rested on the crossbar at the pivot: the beam
                            stays LEVEL yet in_pan is False (level alone earns
                            nothing), NOT success;
  10. stacked, level      — correct mass split but small STACKED ON mid: the beam is
                            genuinely LEVEL (masses balance!) yet the stacked weight
                            reads ~70 mm too high and is rejected by the pan-floor z
                            window -> NOT success;
  11. settle gate         — the full correct load judged while the beam still swings
                            (angular-velocity readback above the gate) is NOT success;
                            the probe weight is then removed BEFORE ring-down completes
                            (the battery never reaches success);
  12. latched credit      — removing that weight leaves the latched score unchanged
                            while loaded() correctly drops;
  13. mechanism reality   — big alone heels the beam far past tilt_max (readback), and
                            removing it lets the pendulum RESTORE level: the balance is
                            a working instrument, not a static prop;
  14. rejection audit     — success() was never True at ANY judged point;
  15. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.place_cups_i82.smoke --headless
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
# RTX recipe: kit mis-decodes the L20 driver version and silently rejects RTX -> the
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
    env = ENVS.get("simgen.balance_scale")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.15, -0.95, 0.75)) + o),
                                tuple(np.array((0.05, 0.00, 0.12)) + o),
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

    def tilt_deg() -> float:
        return math.degrees(float(scene.tilt()[0]))

    def beam_w() -> float:
        return float(scene.beam.data.root_ang_vel_w[0].norm())

    def report(tag: str) -> None:
        s, ok = judge()
        bits = []
        for nm in c.cup_names:
            loc = scene._cup_local(nm)[0]
            bits.append(f"{nm}=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
                        f"{float(loc[2]):+.3f})b in={bool(scene.in_pan(nm)[0])}")
        print(f"[smoke] {tag:16s} | tilt={tilt_deg():+.2f} w={beam_w():.3f} | "
              + " ".join(bits) + f" score={s:.3f} success={ok} frames={len(frames)}",
              flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def write_cup(nm: str, world: torch.Tensor, quat: torch.Tensor,
                  settle_steps: int = 60) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = world
        st[:, 3:7] = quat
        scene.cups[nm].write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def drop_cup(nm: str, side: float, x_off: float, z_extra: float = 0.025,
                 settle_steps: int = 210) -> None:
        """Teleport `nm` above pan `side`'s floor (LIVE beam pose) and let contact
        seat it — the same transport-only move solve.py uses."""
        from isaaclab.utils.math import quat_apply

        bp = scene.beam.data.root_pos_w[0]
        bq = scene.beam.data.root_quat_w[0]
        local = torch.tensor([x_off, side * c.pan_y, c.cup_rest_z + z_extra],
                             device=device)
        world = bp + quat_apply(bq.unsqueeze(0), local.unsqueeze(0))[0]
        write_cup(nm, world, bq, settle_steps)

    def to_floor(nm: str, x: float, y: float, settle_steps: int = 60) -> None:
        world = torch.tensor([x, y, c.cup_h / 2 + 0.003], device=device) \
            + scene.env_origins[0]
        q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        write_cup(nm, world, q, settle_steps)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    bodies = (scene.beam, scene.post, *scene.cups.values())
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    z_ok = all(abs(float(scene.cups[nm].data.root_pos_w[0, 2]) - c.cup_h / 2) < 0.01
               for nm in c.cup_names)
    check("settle: all states finite, cups upright on the floor, EMPTY beam rests "
          f"LEVEL (tilt={tilt_deg():+.2f} deg)",
          fin and z_ok and abs(tilt_deg()) < 1.0 and bool(scene.settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(20)
        m = scene.cup_mass[0]
        # readback mass through the view AGAIN (independent of the reset-path cache)
        m_view = [float(scene.cups[nm].root_physx_view.get_masses().reshape(-1)[0])
                  for nm in c.cup_names]
        ys = [float((scene.cups[nm].data.root_pos_w - scene.env_origins)[0, 1])
              for nm in c.cup_names]
        xs = [float((scene.cups[nm].data.root_pos_w - scene.env_origins)[0, 0])
              for nm in c.cup_names]
        slot = tuple(int(np.argmin([abs(y - sy) for sy in c.stage_ys])) for y in ys)
        reads.append((m_view[0], m_view[1], m_view[2], slot, xs[0], ys[0],
                      abs(float(m[2] - m[0] - m[1])),
                      max(abs(float(m[i]) - m_view[i]) for i in range(3))))
        print(f"[smoke] seed {sd}: masses=({m_view[0]:.4f},{m_view[1]:.4f},"
              f"{m_view[2]:.4f}) slots={slot} small_xy=({xs[0]:+.3f},{ys[0]:+.3f}) "
              f"|big-(s+m)|={reads[-1][6]:.2e}", flush=True)
    ms_spread = max(r[0] for r in reads) - min(r[0] for r in reads)
    mm_spread = max(r[1] for r in reads) - min(r[1] for r in reads)
    sum_ok = all(r[6] < 1e-4 for r in reads)
    cache_ok = all(r[7] < 1e-5 for r in reads)
    check("randomization: sampled masses vary across seeded resets (VIEW readback: "
          f"spread small={ms_spread * 1000:.1f} g, mid={mm_spread * 1000:.1f} g) and "
          "m_big = m_small + m_mid holds every reset",
          ms_spread > 0.005 and mm_spread > 0.005 and sum_ok and cache_ok)
    slots = {r[3] for r in reads}
    x_spread = max(r[4] for r in reads) - min(r[4] for r in reads)
    check("randomization: staging-slot permutation varies (readback: "
          f"{len(slots)} distinct / 6) and xy jitter is real "
          f"(x spread {x_spread * 1000:.0f} mm)", len(slots) >= 3 and x_spread > 0.01)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps (the empty beam "
          "IS level — level alone earns nothing)", s <= 0.02 and not ok)

    # =========================== 6. seed-analog: one weight per location ====================
    # The seed family's end state: each object on its OWN dedicated target, one each.
    # Nearest analog: one weight in EACH pan, the third left on the floor.
    env.reset(seed=41)
    step(30)
    drop_cup("big", +1.0, 0.0)
    drop_cup("mid", -1.0, 0.0)
    report("seed-analog")
    s, ok = judge()
    check("seed-analog (one weight per location, third on the floor): loaded() False "
          "-> NOT success, score <= 0.25",
          not bool(scene.loaded()[0]) and not ok and s <= 0.25)

    # =========================== 7. wrong partition: {big,small} vs {mid} ===================
    env.reset(seed=51)
    step(30)
    drop_cup("big", +1.0, -0.018)
    drop_cup("mid", -1.0, 0.0)
    drop_cup("small", +1.0, +0.033, settle_steps=330)
    report("wrong-split")
    t = tilt_deg()
    s, ok = judge()
    check("wrong partition {big,small}|{mid}: all three riding pans, but the beam "
          f"heels PAST tilt_max (tilt={t:+.2f} deg vs {c.tilt_max_deg:.0f}) -> "
          "level() False, NOT success, score <= 0.5",
          abs(t) > c.tilt_max_deg and not bool(scene.level()[0]) and not ok and s <= 0.5)

    # =========================== 8. all three in ONE pan ====================================
    env.reset(seed=61)
    step(30)
    drop_cup("big", +1.0, -0.018)
    drop_cup("mid", +1.0, +0.028)
    drop_cup("small", +1.0, +0.030, z_extra=0.10, settle_steps=330)
    report("one-pan")
    t = tilt_deg()
    s, ok = judge()
    check("all three in ONE pan: beam hard over "
          f"(tilt={t:+.2f} deg) -> NOT success", abs(t) > c.tilt_max_deg and not ok)

    # =========================== 9. on the beam, level ======================================
    env.reset(seed=71)
    step(30)
    bp = scene.beam.data.root_pos_w[0]
    q0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    write_cup("small", bp + torch.tensor([0.0, 0.0, 0.008 + c.cup_h / 2 + 0.005],
                                         device=device), q0, settle_steps=120)
    report("on-beam")
    s, ok = judge()
    loc = scene._cup_local("small")[0]
    check("weight rested ON the crossbar at the pivot: beam stays LEVEL "
          f"(tilt={tilt_deg():+.2f}) yet in_pan False (y={float(loc[1]):+.3f}) -> "
          "NOT success, score ~0",
          abs(tilt_deg()) < c.tilt_max_deg and not bool(scene.in_pan("small")[0])
          and not ok and s <= 0.02)

    # =========================== 10. stacked, level =========================================
    env.reset(seed=81)
    step(30)
    drop_cup("big", +1.0, 0.0)
    drop_cup("mid", -1.0, 0.0)
    # small ON TOP of mid: correct masses -> the beam DOES settle level ...
    drop_cup("small", -1.0, 0.0, z_extra=c.cup_h + 0.02, settle_steps=420)
    report("stacked")
    t = tilt_deg()
    z_small = float(scene._cup_local("small")[0, 2])
    s, ok = judge()
    check("stacked: correct mass split, small ON TOP of mid, beam genuinely LEVEL "
          f"(tilt={t:+.2f} deg) — yet the stacked weight reads z={z_small:+.3f} in "
          f"the beam frame (window [{c.pan_z_lo:.3f},{c.pan_z_hi:.3f}]) -> in_pan "
          "False, NOT success",
          abs(t) <= c.tilt_max_deg and z_small > c.pan_z_hi
          and not bool(scene.in_pan("small")[0]) and not ok)

    # =========================== 11. settle gate ============================================
    env.reset(seed=91)
    step(30)
    drop_cup("big", +1.0, 0.0)
    drop_cup("mid", -1.0, -0.026)
    # the correct final load — but judged IMMEDIATELY, while the beam still swings
    drop_cup("small", -1.0, +0.030, settle_steps=18)
    w_now = beam_w()
    s_mid, ok = judge()
    loaded_now = bool(scene.loaded()[0])
    report("settle-gate")
    check("settle gate: the full correct load judged while the beam still swings "
          f"(|w|={w_now:.2f} rad/s > {c.settle_beam}) is NOT success",
          (w_now > c.settle_beam or abs(tilt_deg()) > c.tilt_max_deg) and not ok)

    # =========================== 12. latched credit survives removal ========================
    # remove small BEFORE ring-down completes (the battery must never reach success)
    s_before, _ = judge()
    to_floor("small", -0.08, 0.0, settle_steps=90)
    report("removed")
    s_after, ok = judge()
    check("latched credit: removing the just-landed weight leaves the latched score "
          f"unchanged ({s_before:.2f} -> {s_after:.2f}) while loaded() drops",
          abs(s_after - s_before) < 1e-3 and not bool(scene.loaded()[0]) and not ok
          and (not loaded_now or s_after >= 0.5 - 1e-3))

    # =========================== 13. mechanism reality ======================================
    env.reset(seed=101)
    step(30)
    drop_cup("big", +1.0, 0.0, settle_steps=330)
    t_loaded = tilt_deg()
    report("mech-heel")
    _s, _ok = judge()
    to_floor("big", -0.08, 0.0, settle_steps=420)
    t_free = tilt_deg()
    report("mech-restore")
    _s, _ok = judge()
    check("mechanism reality: big alone heels the beam far past tilt_max "
          f"(tilt={t_loaded:+.2f} deg), removing it restores LEVEL "
          f"(tilt={t_free:+.2f} deg) — a working instrument",
          abs(t_loaded) > 8.0 and abs(t_free) < 1.5)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.balance_scale")
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

"""Smoke / rubric-REJECTION battery for CellarRollStowScene (sim_gen task
`libero_kitchen_scene4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet_i250`)
— NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — unbar, lay, gravity roll-in, gravity drop-seat — is
the acceptance evidence that the rubric ACCEPTS a correct outcome). Every teleport here
is instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled state and
asserts the rubric REJECTS it; no probe in this battery ever reaches success(), and a
final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: bar seated in the target
                            notches, bottle standing on the open floor, all still,
                            score ~0;
  3-4. randomization      — READBACK over 8 seeded resets: the barred (=target) side
                            flips (Bernoulli) and the bar really sits on that side;
                            bottle spawn xy and bar seat y jitter are real;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  bar blocks the roll — the bar is LOAD-BEARING physics, not decoration: with the
                            bar seated, the bottle laid on the target ramp and released
                            rolls down, strikes the bar, and CANNOT enter the bay
                            (gaps under/over the bar < bottle) — not stowed, bar still
                            seated, score ~0;
  7.  wrong-bay stow      — bottle rolled into the open DECOY bay settles inside IT ->
                            no target-bay credit, no success, score ~0 (mechanism-state
                            identification is load-bearing);
  8.  on-roof cheat       — bottle lying on the target bay's roof -> not contained,
                            score ~0, no success;
  9.  no re-bar           — bar parked aside, bottle genuinely rolled in and stowed:
                            partial credit only, NOT success, score <= 0.85 (the
                            re-bar clause is load-bearing);
  10. wrong notches       — same episode: bar drop-seated into the DECOY bay's notches
                            -> still NOT success (the SAME-notch clause rejects);
  11. near-seat miss      — same episode: bar dropped on the target ramp short of the
                            posts (outside the seat x-window) -> not seated, NOT
                            success, score <= 0.85;
  12. latched credit      — teleporting the bottle back OUT afterwards leaves the
                            latched score unchanged (credit does not evaporate), still
                            no success;
  13. rejection audit     — success() was never True at ANY judged point;
  14. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.<task>.smoke --headless
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cellar_roll_stow")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.30, -0.95, 0.85)) + o),
                                tuple(np.array((0.42, 0.00, 0.05)) + o),
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

    def report(tag: str) -> None:
        w = rel(scene.wine)
        b = rel(scene.bar)
        s, ok = judge()
        print(f"[smoke] {tag:16s} | wine=({float(w[0]):+.3f},{float(w[1]):+.3f},{float(w[2]):.3f}) "
              f"bar=({float(b[0]):+.3f},{float(b[1]):+.3f},{float(b[2]):.3f}) "
              f"side={float(scene.target_sign[0]):+.0f} "
              f"in_bay={bool(scene.bottle_in_bay()[0])} lying={bool(scene.bottle_lying()[0])} "
              f"seated={bool(scene.bar_seated()[0])} unbar={bool(scene._unbarred[0])} "
              f"app={float(scene._app_max[0]):.3f} stow={bool(scene._stowed[0])} "
              f"rebar={float(scene._rebar_max[0]):.3f} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, x: float, y: float, z: float, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    LYING = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)  # local +z -> world -y

    def ramp_z(x: float) -> float:
        th = math.atan2(c.ramp_z0 - c.ramp_z1, c.ramp_x1 - c.ramp_x0)
        return c.ramp_z0 - (x - c.ramp_x0) * math.tan(th)

    def lay_on_ramp(sign: float, x: float = 0.315) -> None:
        """Teleport the bottle to the solve's lay pose on bay side `sign`'s ramp
        (lying, axis along y, 4 mm hover, outside the containment box)."""
        place(scene.wine, x, sign * c.bay_cy, ramp_z(x) + c.body_r + 0.004, LYING)

    def sgn() -> float:
        return float(scene.target_sign[0])

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("show")
    fin0 = (torch.isfinite(scene.cellar.data.root_state_w).all()
            and torch.isfinite(scene.wine.data.root_state_w).all()
            and torch.isfinite(scene.bar.data.root_state_w).all())
    wz = float(rel(scene.wine)[2])
    still = (float(scene.wine.data.root_lin_vel_w[0].norm()) < c.settle_lin
             and float(scene.bar.data.root_lin_vel_w[0].norm()) < c.settle_lin)
    check("settle: states finite, bar seated in the target notches, bottle standing "
          "on the open floor, all still",
          bool(fin0) and bool(scene.bar_seated()[0]) and abs(wz - c.body_h / 2) < 0.01
          and not bool(scene.bottle_lying()[0]) and not bool(scene.bottle_in_bay()[0])
          and still)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        w = rel(scene.wine)
        b = rel(scene.bar)
        reads.append((float(w[0]), float(w[1]), float(b[1]), sgn()))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (wine_x, wine_y, bar_y, target_sign):\n{arr}",
          flush=True)
    sides = arr[:, 3]
    check("randomization: the barred (=target) side flips across seeded resets AND the "
          "bar readback really sits on the target side each time",
          0.0 < (sides > 0).mean() < 1.0
          and all(abs(r[2] - r[3] * c.bay_cy) < 0.02 for r in reads))
    wine_jit = float((arr[:, 0:2].max(axis=0) - arr[:, 0:2].min(axis=0)).max())
    bar_jit = 0.0
    for flag in (1.0, -1.0):
        grp = arr[sides == flag]
        if len(grp) >= 2:
            bar_jit = max(bar_jit, float(grp[:, 2].max() - grp[:, 2].min()))
    check("randomization: bottle spawn jitter is real (readback spread > 4 mm) and the "
          "bar's seat y jitter is real (spread > 1.5 mm within a side group)",
          wine_jit > 0.004 and bar_jit > 0.0015)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. the seated bar BLOCKS the roll ==========================
    # The mechanism is load-bearing physics: with the bar seated, the bottle laid on the
    # target ramp (the solve's own lay pose) and RELEASED must be stopped by the bar —
    # the gap under it (20 mm) and over it (40 mm) are both smaller than the 60 mm
    # bottle. Rejects any notion that the bar is decorative.
    torch.manual_seed(41)
    env.reset()
    step(30)
    lay_on_ramp(sgn())
    step(400)
    report("bar-blocks")
    s, ok = judge()
    check("bar blocks the roll: bottle released on the barred ramp is STOPPED by the "
          "seated bar — never enters the bay, bar still seated, score ~0 (<= 0.05), "
          "no success",
          not bool(scene.bottle_in_bay()[0]) and not bool(scene._stowed[0])
          and bool(scene.bar_seated()[0]) and not ok and s <= 0.05)

    # =========================== 7. wrong-bay stow (decoy) ==================================
    # Same physics, wrong bay: rolled into the open DECOY the bottle settles INSIDE the
    # decoy — and earns nothing (identification by mechanism state is load-bearing).
    torch.manual_seed(51)
    env.reset()
    step(30)
    decoy_sign = -sgn()
    lay_on_ramp(decoy_sign)
    step(400)
    report("wrong-bay")
    s, ok = judge()
    in_decoy = bool(scene.bottle_in_bay(scene.target_sign * -1.0)[0])
    check("wrong bay: bottle rolled into the open DECOY bay settles inside it — no "
          "target credit, score ~0 (<= 0.05), no success",
          in_decoy and not bool(scene.bottle_in_bay()[0]) and not ok and s <= 0.05)

    # =========================== 8. on-roof cheat ===========================================
    torch.manual_seed(61)
    env.reset()
    step(30)
    place(scene.wine, 0.585, sgn() * c.bay_cy, 0.125 + c.body_r + 0.002, LYING)
    step(90)
    report("on-roof")
    s, ok = judge()
    check("on-roof cheat: bottle lying on the target bay's roof — not contained, "
          "score ~0 (<= 0.05), no success",
          not bool(scene.bottle_in_bay()[0]) and not ok and s <= 0.05)

    # =========================== 9. stowed but NOT re-barred ================================
    # A genuine partial run: bar parked aside (transport), bottle laid and ROLLED in by
    # gravity. Every bottle clause passes; the re-bar clause alone must reject.
    torch.manual_seed(71)
    env.reset()
    step(30)
    place(scene.bar, 0.08, sgn() * 0.35, c.bar_size[2] / 2 + 0.002)
    step(30)
    lay_on_ramp(sgn())
    step(500)
    report("no-rebar")
    s9, ok = judge()
    check("no re-bar: bottle genuinely rolled in and stowed, bar parked aside — "
          "partial credit only, NOT success, score <= 0.85 (re-bar clause is "
          "load-bearing)",
          bool(scene.bottle_in_bay()[0]) and bool(scene.bottle_lying()[0])
          and bool(scene._stowed[0]) and not bool(scene.bar_seated()[0])
          and not ok and 0.30 <= s9 <= 0.85)

    # =========================== 10. bar in the WRONG notches ===============================
    # Same episode: drop-seat the bar into the DECOY bay's notches. It seats there —
    # and the SAME-notch clause still rejects.
    place(scene.bar, c.post_x, -sgn() * c.bay_cy, 0.070)
    step(150)
    report("wrong-notches")
    s10, ok = judge()
    decoy_seated = bool(scene.bar_seated(scene.target_sign * -1.0)[0])
    check("wrong notches: bar drop-seated in the DECOY bay's notches — target seat "
          "clause rejects: NOT success, score <= 0.85",
          decoy_seated and not bool(scene.bar_seated()[0]) and not ok and s10 <= 0.85)

    # =========================== 11. near-seat miss =========================================
    # Same episode: bar dropped on the target ramp just SHORT of the posts — resting
    # level and near, but outside the seat x-window (|dx| = 42 mm vs 12).
    place(scene.bar, 0.430, sgn() * c.bay_cy, ramp_z(0.430) + c.bar_size[2] / 2 + 0.002)
    step(150)
    report("near-seat")
    s11, ok = judge()
    bar_x = float(rel(scene.bar)[0])
    check("near-seat miss: bar resting on the ramp short of the posts (outside the "
          "seat x-window) — not seated, NOT success, score <= 0.85",
          abs(bar_x - c.post_x) > c.seat_dx and not bool(scene.bar_seated()[0])
          and not ok and s11 <= 0.85)

    # =========================== 12. latched credit survives regression =====================
    place(scene.wine, 0.20, -sgn() * 0.10, c.body_h / 2 + 0.002)
    step(60)
    report("regressed")
    s12, ok = judge()
    check("latched credit: teleporting the bottle back OUT of the bay leaves the "
          f"latched score unchanged ({s11:.3f} -> {s12:.3f}), still no success",
          abs(s12 - s11) < 1e-3 and not bool(scene.bottle_in_bay()[0]) and not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.cellar.data.root_state_w).all()
           and torch.isfinite(scene.wine.data.root_state_w).all()
           and torch.isfinite(scene.bar.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.cellar_roll_stow")
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
    # Hard exit: Kit teardown hangs — watchdog then die.
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
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)

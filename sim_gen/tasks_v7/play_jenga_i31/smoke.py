"""Smoke / rubric-REJECTION battery for CompassDialsScene (sim_gen task
`play_jenga_i31`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — spin each captive arrow about its pivot pin with
pulsed contact torques — is the acceptance evidence that the rubric ACCEPTS a correct
outcome). Every teleport here is instrumentation that CONSTRUCTS a wrong (or partial)
outcome and asserts the rubric REJECTS it; no probe in this battery ever reaches
success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: every arrow seated on its pin,
                            still; score ~0 at rest, no success;
  3-4. randomization      — READBACK over 6 seeded resets: post target azimuths, arrow
                            spawn headings and pin xy all move; invariants hold (posts on
                            their ring, spawn error >= init_sep);
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  SEED strategy       — the seed's whole plan ("slide the bar out and carry it
                            somewhere") = every arrow lifted OFF its pin and laid flat
                            on the bench POINTING CORRECTLY at its own post: heading
                            perfect, yet on_pin is False for all -> NOT success,
                            score ~0 (off-pin arrows are gated out of every credit);
  7.  near-miss           — all three arrows ON their pins at target + 14 deg
                            (align_tol is 10 deg), settled -> no dial aligned, NOT
                            success, only partial progress credit;
  8.  wrong color         — each arrow aimed at the NEXT dial's post as seen from its
                            own pin (color binding violated) -> no dial aligned;
  9.  decoy               — the green arrow aimed at the YELLOW decoy post -> not
                            aligned (the decoy matches no arrow);
  10. flipped 180 deg     — arrows on their pins with the TAIL toward the post
                            (heading off by pi) -> no dial aligned, NOT success;
  11. latched credit      — aligning the red dial then spinning it away leaves its
                            latched credit unchanged (and success never appears);
  12. monotonicity        — swinging a dial closer to its target latches strictly
                            more progress credit than a farther heading;
  13. rejection audit     — success() was never True at ANY judged point;
  14. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.play_jenga_i31.smoke --headless
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
    env = ENVS.get("simgen.compass_dials")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    d = c.n_dials
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    tol = math.radians(c.align_tol_deg)
    z_bar = c.bench_top + c.bar_t / 2 + 0.002

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.55, -0.85, 0.72)) + o),
                                tuple(np.array((0.0, 0.0, 0.11)) + o),
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

    def report(tag: str) -> None:
        err = scene.heading_err()[0]
        on = scene.on_pin()[0]
        al = scene.aligned()[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | err_deg=("
              + ",".join(f"{math.degrees(float(e)):6.1f}" for e in err)
              + ") on_pin=(" + ",".join(str(bool(v))[0] for v in on)
              + ") aligned=(" + ",".join(str(bool(v))[0] for v in al)
              + f") score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def pin_xy(k: int) -> tuple[float, float]:
        p = (scene.pins[k].data.root_pos_w - scene.env_origins)[0]
        return float(p[0]), float(p[1])

    def place_arrow(k: int, yaw: float, x: float, y: float, settle_steps: int = 40) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z_bar
        st[:, 3] = math.cos(yaw / 2)
        st[:, 6] = math.sin(yaw / 2)
        st[:, 0:3] += scene.env_origins
        scene.arrows[k].write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def on_pin_at(k: int, yaw: float, settle_steps: int = 40) -> None:
        px, py = pin_xy(k)
        place_arrow(k, yaw, px, py, settle_steps)

    def targets() -> list[float]:
        return [float(scene.target_az[0, k]) for k in range(d)]

    def finite_all() -> bool:
        ok = True
        for b in scene.arrows + scene.pins + scene.posts + [scene.decoy]:
            ok = ok and bool(torch.isfinite(b.data.root_state_w).all())
        return ok

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    check("settle: states finite; every arrow seated ON its pin (on_pin readback), "
          "everything still",
          finite_all() and bool(scene.on_pin()[0].all()) and bool(scene.settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    inv_ok = True
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(5)
        az = targets()
        yaws = [float(scene.headings()[0, k]) for k in range(d)]
        pxy = [v for k in range(d) for v in pin_xy(k)]
        reads.append(az + yaws + pxy)
        # invariants: every post on its ring, every spawn heading far from its target
        for k in range(d):
            px, py = pin_xy(k)
            q = (scene.posts[k].data.root_pos_w - scene.env_origins)[0]
            ring = math.hypot(float(q[0]) - px, float(q[1]) - py)
            inv_ok = inv_ok and abs(ring - c.post_ring_r) < 0.01
        err = scene.heading_err()[0]
        inv_ok = inv_ok and bool((err >= math.radians(c.init_sep_deg) - 0.1).all())
    arr = np.array(reads)
    print(f"[smoke] randomization readback (az_r, az_g, az_b, yaw_r, yaw_g, yaw_b, "
          f"pin xy x6):\n{arr.round(3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: post target azimuths and arrow spawn headings vary across "
          "seeded resets (readback)",
          bool((spread[0:3] > 0.2).all()) and bool((spread[3:6] > 0.2).all()))
    check("randomization: pin xy jitter varies across seeded resets; invariants hold "
          "(posts on their ring, spawn error >= init_sep)",
          bool((spread[6:12] > 0.003).all()) and inv_ok)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. SEED strategy ===========================================
    # The seed's whole plan is "slide the bar out of the tower and carry it somewhere".
    # Construct exactly that end state for ALL THREE dials: each arrow lifted OFF its
    # pin and laid flat on the bench nearby, POINTING PERFECTLY at its own post
    # (heading = target azimuth). The rubric must refuse all credit: heading is right
    # but the arrow is no longer captive on its pivot.
    env.reset(seed=41)
    step(10)
    az = targets()
    for k in range(d):
        px, py = pin_xy(k)
        # lateral offset perpendicular to the pointing direction: clear of the pin,
        # clear of the neighbouring dials (proven margins in TASK.md)
        ox = -math.sin(az[k]) * 0.045
        oy = math.cos(az[k]) * 0.045
        place_arrow(k, az[k], px + ox, py + oy, settle_steps=10)
    step(40)
    report("seed-strategy")
    s, ok = judge()
    err = scene.heading_err()[0]
    on = scene.on_pin()[0]
    check("seed strategy (arrows OFF their pins, laid pointing correctly at their own "
          "posts): heading within tol for all, yet on_pin False for all -> no dial "
          "aligned, NOT success, score <= 0.02",
          bool((err <= tol).all()) and not bool(on.any())
          and not bool(scene.aligned()[0].any()) and not ok and s <= 0.02)

    # =========================== 7. near-miss ===============================================
    env.reset(seed=51)
    step(10)
    az = targets()
    for k in range(d):
        on_pin_at(k, az[k] + math.radians(14.0))  # align_tol is 10 deg
    report("near-miss")
    s, ok = judge()
    err = scene.heading_err()[0]
    check("near-miss: all three arrows ON their pins at target + 14 deg — on-pin "
          "verified, no dial aligned, NOT success, no alignment credit (score <= 0.35)",
          bool(scene.on_pin()[0].all()) and bool((err > tol).all())
          and not bool(scene.aligned()[0].any()) and not ok and s <= 0.35
          and float(scene.align_latch.max()) == 0.0)

    # =========================== 8. wrong color =============================================
    env.reset(seed=61)
    step(10)
    for k in range(d):
        px, py = pin_xy(k)
        q = (scene.posts[(k + 1) % d].data.root_pos_w - scene.env_origins)[0]
        yaw_wrong = math.atan2(float(q[1]) - py, float(q[0]) - px)
        on_pin_at(k, yaw_wrong)
    report("wrong-color")
    s, ok = judge()
    err = scene.heading_err()[0]
    check("wrong color: every arrow aimed at the NEXT dial's post from its own pin — "
          "on-pin, yet no dial aligned (identity is judged), NOT success",
          bool(scene.on_pin()[0].all()) and bool((err > tol).all())
          and not bool(scene.aligned()[0].any()) and not ok)

    # =========================== 9. decoy ===================================================
    qd = (scene.decoy.data.root_pos_w - scene.env_origins)[0]
    px, py = pin_xy(1)
    yaw_decoy = math.atan2(float(qd[1]) - py, float(qd[0]) - px)
    on_pin_at(1, yaw_decoy)
    report("decoy-aim")
    s, ok = judge()
    check("decoy: the green arrow aimed at the YELLOW decoy post is NOT aligned "
          "(the decoy matches no arrow), NOT success",
          bool(scene.on_pin()[0, 1]) and float(scene.heading_err()[0, 1]) > tol
          and not bool(scene.aligned()[0, 1]) and not ok)

    # =========================== 10. flipped 180 deg ========================================
    env.reset(seed=71)
    step(10)
    az = targets()
    for k in range(d):
        on_pin_at(k, az[k] + math.pi)  # tail toward the post
    report("flipped-180")
    s, ok = judge()
    err = scene.heading_err()[0]
    check("flipped: arrows on their pins with the TAIL toward the post (heading off "
          "by ~180 deg) — no dial aligned, NOT success",
          bool(scene.on_pin()[0].all()) and bool((err > math.radians(90.0)).all())
          and not bool(scene.aligned()[0].any()) and not ok)

    # =========================== 11. latched credit survives regression =====================
    env.reset(seed=81)
    step(10)
    az = targets()
    on_pin_at(0, az[0], settle_steps=80)  # red dial aligned and settled -> latch
    report("red-aligned")
    s_in, _ = judge()
    on_pin_at(0, az[0] + math.pi / 2, settle_steps=60)  # spin it away again
    report("red-regressed")
    s_out, ok = judge()
    check("latched credit: aligning the red dial then spinning it away leaves its "
          "latched credit unchanged (score >= 0.28 both sides), success never appears",
          s_in >= 0.28 and abs(s_out - s_in) < 0.02 and not ok
          and not bool(scene.aligned()[0, 0]))

    # =========================== 12. progress monotonicity ==================================
    env.reset(seed=91)
    step(5)
    az2 = float(scene.target_az[0, 2])
    e0 = float(scene.err0[0, 2])
    sgn = 1.0 if float(scene._wrap(scene.headings()[:, 2] - scene.target_az[:, 2])[0]) > 0 else -1.0
    on_pin_at(2, az2 + sgn * 0.55 * e0, settle_steps=5)
    p_half = float(scene.prog_latch[0, 2])
    on_pin_at(2, az2 + sgn * 0.20 * e0, settle_steps=5)
    p_near = float(scene.prog_latch[0, 2])
    check("monotonicity: swinging the blue dial closer to its target latches strictly "
          f"more progress credit ({p_half:.3f} < {p_near:.3f})", p_half + 0.10 < p_near)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.compass_dials")
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
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()

"""Smoke / rubric-REJECTION battery for CupShellsScene (sim_gen task
`stack_cups_i92`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — flip each cup inverted above its matching die,
lower it through contact until the rim seals on the floor, release — is the
acceptance evidence that the rubric ACCEPTS a correct outcome; it passes on seeds
0/1/2). Every teleport here is instrumentation that CONSTRUCTS a wrong (or partial)
outcome and asserts the rubric REJECTS it. No probe in this battery ever reaches
success(), and a final audit asserts exactly that.

  1-2. settle/no-NaN    — reset layout settles finite: cups rim-UP in one row, dice
                          in the other, three distinct slots each; score ~0;
  3-4. randomization    — READBACK over 8 seeded resets: which row holds cups vs
                          dice flips, and the per-row color-slot permutations vary;
                          every reset lies sane;
  5.  null policy       — 240 idle steps -> score ~0, no success;
  6.  seed-strategy A   — the seed family's outcome expressed on the targets: dice
                          DROPPED INSIDE the rim-UP cups (containment the nesting
                          way) -> zero pairs covered (the cup must be upside-down),
                          score ~0;
  7.  seed-strategy B   — cups STACKED on one another (the literal stack_cups end
                          state; these cups cannot nest, so they pile bottom-on-rim)
                          -> nothing covered, score ~0;
  8.  wrong color       — every die genuinely ENCLOSED under an inverted, sealed cup
                          of the WRONG color (full derangement) -> zero pairs
                          covered (identity is pairwise), success False; score
                          equals the latched flip credit exactly (0.40) — nothing
                          more;
  9.  near-miss outside — inverted cup sealed on the floor BESIDE its die (die
                          against the outer wall) -> die_radial >> xy_tol, rejected;
  10. perched cocked    — inverted cup lowered OFF-CENTER onto its die so it rests
                          cocked on the die instead of sealing -> rejected (rim
                          seal / radial);
  11. die on top        — die set ON TOP of its sealed upside-down cup -> die_height
                          >> die_low_max, rejected (the die must be ON THE FLOOR
                          under the cup);
  12. settle gate       — a genuinely covered pair with velocity injected (the
                          sustained-stillness counter resets) is NOT covered while
                          moving;
  13. latched credit    — teleporting that cup away leaves the FLIP credit (0.40/3)
                          latched while the covered credit correctly drops;
  14. rejection audit   — success() was never True at ANY judged point;
  15. final no-NaN      — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.stack_cups_i92.smoke --headless
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

COLORS = scene_mod.COLORS

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cup_shells")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.30, -1.05, 0.85)) + o),
                                tuple(np.array((0.20, 0.00, 0.03)) + o),
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
        s, ok = judge()
        cov = scene.covered()[0]
        rad = scene.die_radial()[0]
        mh = scene.mouth_height()[0]
        up = scene.cup_up_z()[0]
        dh = scene.die_height()[0]
        pairs = " ".join(
            f"{nm}:cov={bool(cov[i])} rad={float(rad[i]) * 1000:.0f} "
            f"mo={float(mh[i]) * 1000:.0f} up={float(up[i]):+.2f} "
            f"dh={float(dh[i]) * 1000:.0f}"
            for i, (nm, _) in enumerate(COLORS))
        print(f"[smoke] {tag:16s} | {pairs} | score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, xyz, quat=None, vel=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor([float(v) for v in xyz], device=device)
        if quat is None:
            st[:, 3] = 1.0
        else:
            st[:, 3:7] = torch.tensor([float(v) for v in quat], device=device)
        if vel is not None:
            st[:, 7:10] = torch.tensor([float(v) for v in vel], device=device)
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    Q_INV = (0.0, 1.0, 0.0, 0.0)  # 180 deg about x: rim-DOWN

    def cap_over(cup, xy, drop: float = 0.020, settle_steps: int = 90) -> None:
        """Construct an inverted cup sealed at `xy`: hover rim-down `drop` above the
        floor-level obstruction and let gravity + rim-floor contact seal it."""
        place(cup, (xy[0], xy[1], c.die_s + drop + c.height / 2), quat=Q_INV,
              settle_steps=settle_steps)

    def layout_sane(tag: str) -> bool:
        """Reset honesty: cups rim-UP in one row, dice in the other, three distinct
        slots per row."""
        cx = [float(rel(b)[0]) for b in scene.cups]
        cy = [float(rel(b)[1]) for b in scene.cups]
        cz = [float(rel(b)[2]) for b in scene.cups]
        dx = [float(rel(b)[0]) for b in scene.dice]
        dy = [float(rel(b)[1]) for b in scene.dice]
        up = scene.cup_up_z()[0]
        ok = (max(cx) - min(cx) < 2 * c.jitter + 0.02
              and max(dx) - min(dx) < 2 * c.jitter + 0.02
              and abs(sum(cx) / 3 - sum(dx) / 3) > 0.10
              and all(v > 0.9 for v in up.tolist())
              and all(abs(v - c.height / 2) < 0.015 for v in cz)
              and max(cy) - min(cy) > 0.25 and max(dy) - min(dy) > 0.25)
        if not ok:
            print(f"[smoke] layout sanity VIOLATION at {tag}", flush=True)
        return ok

    bodies = scene.cups + scene.dice

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("settle: all states finite; cups rim-up in one row, dice in the other, "
          "three distinct slots each", fin and layout_sane("reset"))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    sane = True
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(30)
        sane = sane and layout_sane(f"seed {sd}")
        cup_x = sum(float(rel(b)[0]) for b in scene.cups) / 3
        reads.append([1.0 if cup_x > (c.row_near_x + c.row_far_x) / 2 else -1.0,
                      float(rel(scene.cups[0])[1]), float(rel(scene.cups[1])[1]),
                      float(rel(scene.dice[0])[1]), float(rel(scene.dice[1])[1])])
    arr = np.array(reads)
    print("[smoke] randomization readback (cup_side, cup_red_y, cup_green_y, "
          f"die_red_y, die_green_y):\n{np.round(arr, 4)}", flush=True)
    sides = {float(v) for v in arr[:, 0]}
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: WHICH row holds the cups flips across seeds "
          f"({len(sides)} sides seen)", len(sides) == 2)
    check("randomization: color-slot permutations vary in BOTH rows (y spreads "
          f"cup_red={spread[1]:.3f} cup_green={spread[2]:.3f} die_red={spread[3]:.3f} "
          f"die_green={spread[4]:.3f}), every reset sane",
          spread[1] > 0.10 and spread[3] > 0.10 and sane)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 6. seed-strategy A: dice INSIDE rim-up cups ================
    # rlbench/stack_cups succeeds by rim-UP containment (things nested inside a
    # standing cup). Express that outcome on our targets: drop every die INSIDE its
    # matching rim-up cup. Geometrically contained — but the cup is not inverted and
    # no rim is sealed: zero pairs covered, no flip ever latched.
    env.reset(seed=41)
    step(30)
    for i in range(3):
        cu = rel(scene.cups[i])
        place(scene.dice[i], (float(cu[0]), float(cu[1]), c.height + 0.03),
              settle_steps=30)
    step(60)
    report("dice-in-cups")
    s, ok = judge()
    rad = scene.die_radial()[0]
    check("seed-strategy A: all three dice dropped INSIDE their rim-UP cups "
          f"(radials {[f'{float(v) * 1000:.0f}' for v in rad]} mm — contained the "
          "nesting way) score ~0, zero pairs covered: the cup must be upside-down",
          s <= 0.02 and not ok and not bool(scene.covered()[0].any()))

    # =========================== 7. seed-strategy B: cups stacked ===========================
    # The literal stack_cups end state: cups piled on one another (ours cannot nest
    # — outer_r > inner_r — so they pile bottom-on-rim). Nothing covered.
    env.reset(seed=51)
    step(30)
    base = rel(scene.cups[0])
    for k, i in enumerate((1, 2)):
        place(scene.cups[i],
              (float(base[0]), float(base[1]),
               c.height / 2 + (k + 1) * (c.height + 0.004)),
              settle_steps=40)
    step(90)
    report("cup-tower")
    s, ok = judge()
    up = scene.cup_up_z()[0]
    check("seed-strategy B: cups STACKED into a tower (bottom-on-rim; nesting is "
          f"impossible by construction) -> up_z={[f'{float(v):+.2f}' for v in up]}, "
          "nothing covered, score ~0",
          s <= 0.02 and not ok and not bool(scene.covered()[0].any()))

    # =========================== 8. wrong color (identity) ==================================
    # Every die genuinely ENCLOSED under an inverted sealed cup — of the WRONG color
    # (full derangement red->green->blue->red). The enclosure geometry is real, yet
    # zero pairs are covered: identity is pairwise. Score reads EXACTLY the latched
    # flip credit (all three cups have been rim-down), nothing more.
    env.reset(seed=61)
    step(30)
    die_xy = [(float(rel(d)[0]), float(rel(d)[1])) for d in scene.dice]
    for cup_i, die_i in ((0, 1), (1, 2), (2, 0)):
        cap_over(scene.cups[cup_i], die_xy[die_i])
    step(60)
    report("wrong-color")
    s, ok = judge()
    mh = scene.mouth_height()[0]
    enclosed = all(float(v) < c.rim_z_max for v in mh)
    check("wrong color: full derangement — every die sealed under an inverted cup "
          f"of the WRONG color (mouth heights {[f'{float(v) * 1000:.0f}' for v in mh]}"
          " mm: enclosures are real) -> zero pairs covered, no success, score = "
          f"latched flip credit {c.flip_credit:.2f} exactly ({s:.3f})",
          enclosed and not ok and not bool(scene.covered()[0].any())
          and abs(s - c.flip_credit) < 0.02)

    # =========================== 9. near-miss: die just OUTSIDE =============================
    env.reset(seed=71)
    step(30)
    d0 = rel(scene.dice[0])
    miss = c.outer_r + c.die_s / 2 + 0.004
    cap_over(scene.cups[0], (float(d0[0]) + miss, float(d0[1])))
    step(30)
    report("near-miss")
    s, ok = judge()
    r0 = float(scene.die_radial()[0, 0])
    check("near-miss: red cup inverted + sealed BESIDE its die (die against the "
          f"outer wall, radial={r0 * 1000:.0f} mm > xy_tol={c.xy_tol * 1000:.0f} mm) "
          "-> not covered, no success",
          r0 > c.xy_tol and not bool(scene.covered()[0, 0]) and not ok)

    # =========================== 10. perched cocked on the die ==============================
    env.reset(seed=81)
    step(30)
    d0 = rel(scene.dice[0])
    cap_over(scene.cups[0], (float(d0[0]) + 0.030, float(d0[1])), settle_steps=120)
    report("perched")
    s, ok = judge()
    r0 = float(scene.die_radial()[0, 0])
    m0 = float(scene.mouth_height()[0, 0])
    check("perched: cup lowered OFF-CENTER onto its die rests cocked (mouth="
          f"{m0 * 1000:.0f} mm, radial={r0 * 1000:.0f} mm) -> rim seal / radial "
          "reject it, not covered",
          not bool(scene.covered()[0, 0]) and not ok
          and (m0 > c.rim_z_max or r0 > c.xy_tol))

    # =========================== 11. die ON TOP of the inverted cup =========================
    env.reset(seed=91)
    step(30)
    spot = (0.55, 0.35)
    cap_over(scene.cups[0], spot, settle_steps=60)
    cu = rel(scene.cups[0])
    place(scene.dice[0], (float(cu[0]), float(cu[1]), c.height + c.die_s / 2 + 0.01),
          settle_steps=60)
    report("die-on-top")
    s, ok = judge()
    r0 = float(scene.die_radial()[0, 0])
    h0 = float(scene.die_height()[0, 0])
    check("die on top: die set on the upturned bottom of its sealed inverted cup "
          f"(radial={r0 * 1000:.0f} mm — near the axis! — but die_h={h0 * 1000:.0f} mm"
          f" > {c.die_low_max * 1000:.0f} mm) -> the floor gate rejects it",
          h0 > c.die_low_max and not bool(scene.covered()[0, 0]) and not ok)

    # =========================== 12. settle gate ============================================
    # Construct ONE genuinely covered pair, then inject velocity: while the pair
    # moves, the sustained-stillness counter is zero and covered() must read False.
    # (One covered pair never makes success — the battery still never succeeds.)
    env.reset(seed=101)
    step(30)
    d0 = rel(scene.dice[0])
    cap_over(scene.cups[0], (float(d0[0]), float(d0[1])), settle_steps=90)
    report("covered-anchor")
    was_cov = bool(scene.covered()[0, 0])
    cu = scene.cups[0]
    st = cu.data.root_state_w.clone()
    st[:, 7] = 0.25  # lateral kick
    cu.write_root_state_to_sim(st, all_ids)
    step(2)
    report("kicked")
    v_now = float(cu.data.root_lin_vel_w[0].norm())
    moving_cov = bool(scene.covered()[0, 0])
    s, ok = judge()
    check("settle gate: a genuinely covered pair (rubric anchor: covered read True "
          f"once settled) with velocity injected ({v_now:.2f} m/s) is NOT covered "
          "while moving — sustained stillness is required",
          was_cov and not moving_cov and not ok)

    # =========================== 13. latched flip credit ====================================
    place(scene.cups[0], (0.65, -0.45, c.height / 2 + 0.003), settle_steps=60)
    report("moved-away")
    s_after, ok = judge()
    check("latched credit: teleporting the red cup away leaves the FLIP credit "
          f"latched (score={s_after:.3f} ~= {c.flip_credit / 3:.3f}) while the "
          "covered credit correctly drops with the physical state",
          abs(s_after - c.flip_credit / 3) < 0.01
          and not bool(scene.covered()[0, 0]) and not ok)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.cup_shells")
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

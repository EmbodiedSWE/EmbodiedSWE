"""Smoke / rubric-REJECTION battery for DieTipPadScene (sim_gen task
`approach_grasp_spoon_i12`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — stage the die, then tip it end-over-end through
contact torques until blue is up on the disk — is the acceptance evidence that the
rubric ACCEPTS a correct outcome). Every teleport here is instrumentation that
CONSTRUCTS a wrong (or partial) outcome and asserts the rubric REJECTS it; no probe
in this battery ever reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: die flat on the floor at cube
                            half-height, decoy on the floor; score ~0 at rest, no success;
  3-4. randomization      — READBACK over 6 seeded resets: disk xy, die xy, die
                            orientation (up face AND full quat), decoy xy and the
                            approach baseline d0 all move; the BLUE face is NEVER the
                            reset up face (at least one tip is always required);
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  SEED strategy       — the seed's whole plan ("grasp the graspable object, set it
                            on the target") = the RED decoy settled at the disk centre,
                            die untouched -> NOT success, score ~0 (identity matters:
                            only the die counts);
  7.  wrong face up       — die flat at the disk centre with GREEN up (blue sideways),
                            settled -> on_pad but NOT success, score <= 0.70 (< 1.0);
  8.  position near-miss  — die flat, BLUE UP, settled just OUTSIDE pad_margin
                            (margin + 40 mm) -> NOT on_pad, NOT success, score <= 0.55;
  9.  stacked loophole    — die BLUE UP centred over the disk but resting ON TOP of the
                            decoy: xy-near and blue-up yet centre height ~95 mm, so the
                            floor-rest clause rejects it -> NOT success;
  10. carried loophole    — die held BLUE UP in the air over the disk centre (judged
                            transiently): NOT success, and the blue-up-while-LOW latch
                            stays 0 — airborne reorientation earns no orient credit;
  11. latched credit      — boarding the disk (wrong face) then removing the die leaves
                            the latched approach + tip + board score unchanged (and
                            success stays gone);
  12. monotonicity        — moving the die closer to the disk latches strictly more
                            approach credit than a farther placement;
  13. rejection audit     — success() was never True at ANY judged point;
  14. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.approach_grasp_spoon_i12.smoke --headless
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

BLUE_IDX = scene_mod.BLUE_IDX
FACE_NAMES = scene_mod.FACE_NAMES

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.die_tip_pad")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.05, -0.95, 0.75)) + o),
                                tuple(np.array((0.06, 0.02, 0.05)) + o),
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

    def pad_xy() -> torch.Tensor:
        return (scene.pad.data.root_pos_w - scene.env_origins)[0, :2]

    def die_xy() -> torch.Tensor:
        return (scene.die.data.root_pos_w - scene.env_origins)[0, :2]

    def die_z() -> float:
        return float(scene._die_z()[0])

    def report(tag: str) -> None:
        nz = scene._face_nz()[0]
        up = int(nz.argmax())
        s, ok = judge()
        print(f"[smoke] {tag:16s} | d_pad={float((die_xy() - pad_xy()).norm()):.3f} "
              f"z={die_z():.3f} up={FACE_NAMES[up]}({float(nz[up]):.3f}) "
              f"blue_nz={float(nz[BLUE_IDX]):+.3f} on_pad={bool(scene.on_pad()[0])} "
              f"appr={float(scene.approach_latch[0]):.3f} "
              f"tip={float(scene.tip_latch[0]):.0f} ori={float(scene.orient_latch[0]):.0f} "
              f"brd={float(scene.board_latch[0]):.0f} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def face_quat(k: int, yaw: float = 0.0) -> tuple[float, float, float, float]:
        """wxyz quat putting body face k up, composed with a world yaw."""
        from isaaclab.utils.math import quat_mul

        qf = scene._face_up_quat(torch.tensor([k], device=device))
        qz = torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]], device=device)
        q = quat_mul(qz, qf)[0]
        return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    def place_body(body, x: float, y: float, z: float, quat=(1.0, 0.0, 0.0, 0.0),
                   settle_steps: int = 30) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    dec_z = float((scene.decoy.data.root_pos_w - scene.env_origins)[0, 2])
    fin0 = bool(torch.isfinite(scene.die.data.root_state_w).all()
                and torch.isfinite(scene.pad.data.root_state_w).all()
                and torch.isfinite(scene.decoy.data.root_state_w).all())
    check("settle: states finite; die flat on the floor at half-height and decoy on "
          "the floor (readback heights)",
          fin0 and abs(die_z() - c.half) < 0.008 and abs(dec_z - c.decoy_a / 2) < 0.008
          and bool(scene.settled()[0]) and bool(scene.flat()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads, quats, upfaces = [], [], []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(5)
        p = pad_xy()
        d = die_xy()
        dec = (scene.decoy.data.root_pos_w - scene.env_origins)[0, :2]
        reads.append((float(p[0]), float(p[1]), float(d[0]), float(d[1]),
                      float(dec[0]), float(dec[1]), float(scene.d0[0])))
        quats.append(scene.die.data.root_quat_w[0].detach().cpu().numpy().copy())
        upfaces.append(int(scene._face_nz()[0].argmax()))
    arr = np.array(reads)
    qarr = np.abs(np.array(quats))  # |q|: quat double-cover safe
    print(f"[smoke] randomization readback (pad_x, pad_y, die_x, die_y, dec_x, dec_y, "
          f"d0):\n{arr}", flush=True)
    print(f"[smoke] up faces across seeds: {[FACE_NAMES[u] for u in upfaces]}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: disk xy and die xy vary across seeded resets (readback)",
          spread[0] > 0.01 and spread[1] > 0.01 and spread[2] > 0.02 and spread[3] > 0.02)
    qspread = float((qarr.max(axis=0) - qarr.min(axis=0)).max())
    check("randomization: die up-face varies and is NEVER blue, full orientation and "
          "decoy xy and d0 vary (readback)",
          len(set(upfaces)) >= 2 and all(u != BLUE_IDX for u in upfaces)
          and qspread > 0.05 and spread[4] > 0.02 and spread[5] > 0.02 and spread[6] > 0.005)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. SEED strategy ============================================
    # The seed's whole plan is "grasp the graspable object and set it on the target".
    # Here that is exactly the RED decoy settled at the disk centre with the die
    # untouched. The rubric must refuse it entirely: only the DIE counts.
    env.reset(seed=41)
    step(10)
    p = pad_xy()
    place_body(scene.decoy, float(p[0]), float(p[1]), c.decoy_a / 2 + 0.003,
               settle_steps=90)
    report("seed-strategy")
    s, ok = judge()
    dec = (scene.decoy.data.root_pos_w - scene.env_origins)[0, :2]
    dec_on = float((dec - p).norm()) < c.pad_margin
    check("seed strategy (graspable RED decoy settled at the disk centre, die "
          "untouched): decoy verified on the disk, yet NOT success, score <= 0.02",
          dec_on and not ok and s <= 0.02)

    # =========================== 7. wrong face up ============================================
    env.reset(seed=51)
    step(10)
    p = pad_xy()
    place_body(scene.die, float(p[0]), float(p[1]), c.half + 0.004,
               quat=face_quat(0), settle_steps=120)  # GREEN up, blue sideways
    report("wrong-face")
    s, ok = judge()
    check("wrong face: die flat and settled AT the disk centre but GREEN up — on_pad "
          "yet NOT success, score <= 0.70",
          bool(scene.on_pad()[0]) and not bool(scene.blue_up()[0]) and not ok and s <= 0.70)

    # =========================== 8. position near-miss ======================================
    env.reset(seed=61)
    step(10)
    p = pad_xy()
    off = c.pad_margin + 0.040
    place_body(scene.die, float(p[0]) + off, float(p[1]), c.half + 0.004,
               quat=face_quat(BLUE_IDX), settle_steps=120)  # BLUE up, just outside
    report("near-miss-pos")
    s, ok = judge()
    d_pad = float((die_xy() - pad_xy()).norm())
    check("position near-miss: die BLUE UP, flat, settled, but 40 mm outside "
          "pad_margin — NOT on_pad, NOT success, score <= 0.55",
          d_pad > c.pad_margin and bool(scene.blue_up()[0]) and not bool(scene.on_pad()[0])
          and not ok and s <= 0.55)

    # =========================== 9. stacked-on-decoy loophole ================================
    # BLUE UP and xy-centred over the disk — but resting ON TOP of the decoy, centre at
    # ~95 mm instead of 50 mm. An xy+orientation-only rubric would accept this; the
    # floor-rest clause must reject it.
    env.reset(seed=71)
    step(10)
    p = pad_xy()
    place_body(scene.decoy, float(p[0]), float(p[1]), c.decoy_a / 2 + 0.002,
               settle_steps=60)
    place_body(scene.die, float(p[0]), float(p[1]), c.decoy_a + c.half + 0.004,
               quat=face_quat(BLUE_IDX), settle_steps=120)
    report("stacked")
    s, ok = judge()
    d_pad = float((die_xy() - pad_xy()).norm())
    check("stacked loophole: die BLUE UP and xy-near the disk centre but resting ON "
          "the decoy (centre ~95 mm high) — floor-rest clause rejects: NOT success",
          d_pad < c.pad_margin and bool(scene.blue_up()[0]) and die_z() > 0.080 and not ok)

    # =========================== 10. carried (airborne) loophole =============================
    # BLUE UP directly over the disk centre but held in the AIR (judged transiently,
    # then removed before it can land): not success, and the blue-up-while-LOW latch
    # must stay 0 — reorienting the die while carrying it earns no orient credit.
    env.reset(seed=81)
    step(5)
    p = pad_xy()
    place_body(scene.die, float(p[0]), float(p[1]), 0.30,
               quat=face_quat(BLUE_IDX), settle_steps=2)
    report("carried-high")
    s, ok = judge()
    z_high = die_z()
    ori_high = float(scene.orient_latch[0])
    # remove it (blue NOT up) before it can land blue-up on the disk
    place_body(scene.die, float(p[0]) + 0.45, float(p[1]), c.half + 0.003,
               quat=face_quat(0), settle_steps=10)
    check("carried loophole: die BLUE UP in the air over the disk — NOT success while "
          "airborne, and the orient latch stays 0 above low_z",
          z_high > c.low_z and not ok and ori_high == 0.0)

    # =========================== 11. latched credit survives regression =====================
    env.reset(seed=91)
    step(10)
    p = pad_xy()
    place_body(scene.die, float(p[0]), float(p[1]), c.half + 0.004,
               quat=face_quat(0), settle_steps=90)  # boarded, wrong face
    report("boarded-wrong")
    s_in, _ = judge()
    place_body(scene.die, float(p[0]) + 0.45, float(p[1]) - 0.10, c.half + 0.003,
               quat=face_quat(0), settle_steps=60)
    report("die-removed")
    s_out, ok = judge()
    check("latched credit: removing the boarded die leaves the latched score "
          f"unchanged ({s_in:.3f} -> {s_out:.3f}) and success stays gone",
          s_in >= 0.40 and abs(s_out - s_in) < 0.02 and not ok
          and not bool(scene.on_pad()[0]))

    # =========================== 12. approach monotonicity ==================================
    env.reset(seed=101)
    step(5)
    p = pad_xy()
    d0_xy = die_xy().clone()
    q0 = scene.die.data.root_quat_w[0]
    qt = (float(q0[0]), float(q0[1]), float(q0[2]), float(q0[3]))
    dx, dy = float(p[0] - d0_xy[0]), float(p[1] - d0_xy[1])
    place_body(scene.die, float(d0_xy[0]) + 0.5 * dx, float(d0_xy[1]) + 0.5 * dy,
               c.half + 0.003, quat=qt, settle_steps=3)
    a_half = float(scene.approach_latch[0])
    place_body(scene.die, float(d0_xy[0]) + 0.85 * dx, float(d0_xy[1]) + 0.85 * dy,
               c.half + 0.003, quat=qt, settle_steps=3)
    a_near = float(scene.approach_latch[0])
    check("monotonicity: moving the die closer to the disk latches strictly more "
          f"approach credit ({a_half:.3f} < {a_near:.3f})", a_half + 0.10 < a_near)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.pad.data.root_state_w).all()
           and torch.isfinite(scene.die.data.root_state_w).all()
           and torch.isfinite(scene.decoy.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.die_tip_pad")
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

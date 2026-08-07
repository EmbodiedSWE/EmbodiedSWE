"""Smoke / rubric-REJECTION battery for SlabEaselScene (sim_gen task
`libero_kitchen_scene9_put_the_frying_pan_on_top_of_the_cabinet_i53`) — NullRobot,
teleported probe states, RECORDED.

This is NOT a solution (solve.py — stage each colored slab flat at its bay and HINGE it
up with a bounded torque through real foot-edge contact until it tips onto its panel —
is the acceptance evidence that the rubric ACCEPTS a correct outcome; verified
SIM_GEN_SOLVE: SUCCESS on seeds 0 and 7). Every teleport here is instrumentation that
CONSTRUCTS a wrong (or partial) outcome and asserts the rubric REJECTS it; no probe in
this battery ever reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: four slabs flat at work-plane
                            height, blank flat; score ~0 at rest, no success;
  3-4. randomization      — READBACK over 8 seeded resets: studio xy + free yaw vary;
                            slab rack-frame xy + yaw vary and the y-slot PERMUTATION
                            actually permutes (the red slab occupies different slots);
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  SEED strategy       — the seed's whole plan ("carry the object to the target
                            region and set it down") = red slab laid FLAT at its bay's
                            staging point: the staging latch may arm (score <= 0.10)
                            but NOT success — flat placement is worthless alone;
  7.  seated-pose inject  — red slab WRITTEN directly into the seated lean pose in its
                            own bay (physically consistent — it stays leaning):
                            seated_now verified True, yet the rising-tilt account is
                            empty -> NOT success, score ~0 (the anti-teleport gate);
  8.  wrong bay           — red slab injected seated in the GREEN bay: rejected;
  9.  lateral near miss   — red slab injected seated BETWEEN bays (0.18 m off):
                            rejected;
  10. wrong object        — the BLANK slab injected seated in the RED bay: the
                            distractor-flat constraint breaks -> NOT success;
  11. latched credit      — staging the red slab (honest flat put-down) then removing
                            it leaves the latched staging credit unchanged (success
                            still False);
  12. out-of-bay sweep    — the red slab rotated 0 -> 80 degrees by INCREMENTAL WRITES
                            (plausible per-substep deltas!) at its spawn spot, OUT of
                            the bay: the erect account stays empty — tilt gained
                            outside the bay is worthless, so prop-it-up-elsewhere-and-
                            carry-it-in cannot work;
  13. rejection audit     — success() was never True at ANY judged point;
  14. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene9_put_the_frying_pan_on_top_of_the_cabinet_i53.smoke --headless
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
    from isaaclab.utils.math import quat_apply, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.slab_easel")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.30, -1.05, 0.95)) + o),
                                tuple(np.array((-0.10, 0.0, 0.15)) + o),
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

    def rack_to_world(lx: float, ly: float, lz: float) -> torch.Tensor:
        off = torch.tensor([[lx, ly, lz]], device=device)
        return (scene.studio.data.root_pos_w[0:1]
                + quat_apply(scene.studio.data.root_quat_w[0:1], off))[0]

    def write_slab(k: int, lx: float, ly: float, lz: float, q_local,
                   settle_steps: int = 40) -> None:
        """Kinematic probe placement in the RACK frame (instrumentation, not a
        solution) + REAL physics steps before judging (the zero-step trap)."""
        ql = torch.tensor([list(q_local)], device=device)
        qw = quat_mul(scene.studio.data.root_quat_w[0:1], ql)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = rack_to_world(lx, ly, lz)
        st[:, 3:7] = qw
        scene.slabs[k].write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    QFLAT = (1.0, 0.0, 0.0, 0.0)
    a80 = math.radians(90.0 - c.panel_beta_deg)  # rotation that lays ez onto the panel normal
    QSEAT = (math.cos(a80 / 2), 0.0, math.sin(a80 / 2), 0.0)
    seat_lz = c.seat_z_exp - c.z_work  # seated centre height above the work plane... local z

    def seat_pose_write(k: int, ly: float, settle_steps: int = 50) -> None:
        write_slab(k, c.seat_x_exp, ly, c.z_work + seat_lz + 0.004, QSEAT,
                   settle_steps=settle_steps)

    def rack_xy(k: int) -> tuple[float, float]:
        loc = scene.rack_local(scene.slabs[k].data.root_pos_w)[0]
        return float(loc[0]), float(loc[1])

    def tilt_deg(k: int) -> float:
        return math.degrees(float(scene.tilts()[0, k]))

    def report(tag: str) -> None:
        s, ok = judge()
        tl = "/".join(f"{tilt_deg(k):.0f}" for k in range(4))
        acc = [round(math.degrees(float(a)), 1) for a in scene.erect_acc[0]]
        print(f"[smoke] {tag:16s} | tilt(deg)={tl} "
              f"seated={[bool(scene.seated_now(k)[0]) for k in range(3)]} "
              f"staged={scene.staged_l[0].tolist()} acc(deg)={acc} "
              f"blank_flat={bool(scene.distractor_flat()[0])} "
              f"still={int(scene.still_cnt[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    fin_all = lambda: bool(  # noqa: E731
        torch.isfinite(scene.studio.data.root_state_w).all()
        and all(torch.isfinite(b.data.root_state_w).all() for b in scene.slabs))

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    zs = [rack_xy(k) for k in range(4)]
    z_ok = all(abs(float(scene.rack_local(scene.slabs[k].data.root_pos_w)[0, 2])
                   - c.z_work - c.slab_size[2] / 2) < 0.010 for k in range(4))
    tilt_ok = all(tilt_deg(k) < 5.0 for k in range(4))
    check("settle: states finite; all four slabs flat at work-plane height (readback); "
          "blank flat; everything pose-still",
          fin_all() and z_ok and tilt_ok and bool(scene.distractor_flat()[0])
          and bool(scene.settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)
    _ = zs

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(5)
        sp = (scene.studio.data.root_pos_w - scene.env_origins)[0]
        sy = float(scene._yaw_of(scene.studio)[0])
        rx, ry = rack_xy(0)
        bx, by = rack_xy(2)
        ryaw = float(scene._yaw_of(scene.slabs[0])[0])
        reads.append((float(sp[0]), float(sp[1]), sy, rx, ry, bx, by, ryaw))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (studio_x, studio_y, studio_yaw, red_x, "
          f"red_y, blue_x, blue_y, red_yaw):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: studio xy + free yaw vary across seeded resets (readback)",
          spread[0] > 0.01 and spread[1] > 0.01 and spread[2] > 0.5)
    red_slots = {min(range(4), key=lambda i, v=v: abs(v - c.slot_ys[i]))
                 for v in arr[:, 4]}
    check("randomization: slab rack-frame xy + yaw vary and the y-slot permutation "
          "actually permutes (the red slab occupies multiple slots)",
          spread[3] > 0.01 and spread[4] > 0.10 and spread[5] > 0.01
          and spread[6] > 0.10 and spread[7] > 0.5 and len(red_slots) >= 2)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. SEED strategy ===========================================
    # The seed's whole plan is "carry the object to the target region and set it down".
    # Here that is the red slab laid FLAT at its bay's staging point. The staging latch
    # may arm — flat placement earns its 0.06 — but never success.
    env.reset(seed=41)
    step(10)
    write_slab(0, c.x_bp + c.stage_dx, c.bay_ys[0],
               c.z_work + c.slab_size[2] / 2 + 0.004, QFLAT, settle_steps=60)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy (red slab set down FLAT in its bay): staging latch may arm "
          "but NOT seated, NOT success, score <= 0.10",
          not bool(scene.seated_now(0)[0]) and tilt_deg(0) < 10.0
          and not ok and s <= 0.10)

    # =========================== 7. seated-pose teleport inject =============================
    # The anti-teleport account: the red slab WRITTEN directly into the seated lean
    # pose in its own bay (a physically consistent pose — it stays leaning), with NO
    # tilt ever swept through the band. seated_now reads True; the empty rising-tilt
    # account must reject it.
    env.reset(seed=51)
    step(10)
    seat_pose_write(0, c.bay_ys[0])
    report("seated-inject")
    s, ok = judge()
    check("seated-pose inject: seated_now verified True (pose held by the panel), yet "
          "the rising-tilt account is empty -> NOT success, score ~0",
          bool(scene.seated_now(0)[0]) and tilt_deg(0) > 60.0
          and float(scene.erect_acc[0, 0]) < math.radians(3.0) and not ok and s <= 0.05)

    # =========================== 8. wrong bay ===============================================
    env.reset(seed=61)
    step(10)
    seat_pose_write(0, c.bay_ys[1])  # red slab leaning in the GREEN bay
    report("wrong-bay")
    s, ok = judge()
    check("wrong bay: red slab leaning in the GREEN bay (verified leaning) — not "
          "seated for RED, NOT success, score ~0",
          tilt_deg(0) > 60.0 and not bool(scene.seated_now(0)[0]) and not ok
          and s <= 0.05)

    # =========================== 9. lateral near miss =======================================
    env.reset(seed=71)
    step(10)
    seat_pose_write(0, c.bay_ys[0] + 0.18)  # between the red and green bays
    report("near-miss")
    s, ok = judge()
    check("lateral near miss: red slab leaning 0.18 m off its bay centre (between "
          "bays) — rejected",
          tilt_deg(0) > 60.0 and not bool(scene.seated_now(0)[0]) and not ok
          and s <= 0.05)

    # =========================== 10. wrong object ===========================================
    env.reset(seed=81)
    step(10)
    seat_pose_write(3, c.bay_ys[0])  # the BLANK slab leaning in the RED bay
    report("wrong-object")
    s, ok = judge()
    check("wrong object: the BLANK slab leaning in the RED bay — the distractor-flat "
          "constraint breaks (verified), NOT success",
          tilt_deg(3) > 60.0 and not bool(scene.distractor_flat()[0]) and not ok)

    # =========================== 11. latched credit survives regression =====================
    env.reset(seed=91)
    step(10)
    write_slab(0, c.x_bp + c.stage_dx, c.bay_ys[0],
               c.z_work + c.slab_size[2] / 2 + 0.004, QFLAT, settle_steps=60)
    s_in, _ = judge()
    write_slab(0, 0.33, 0.45, c.slab_size[2] / 2 + 0.004, QFLAT, settle_steps=60)
    report("staged-removed")
    s_out, ok = judge()
    check("latched credit: staging the red slab latches its credit; removing it "
          "leaves the score unchanged (and no success)",
          s_in >= 0.05 and abs(s_out - s_in) < 0.01 and not ok
          and not bool(scene.staged_now(0)[0]))

    # =========================== 12. out-of-bay incremental-write sweep =====================
    # Rotate the red slab 0 -> 80 degrees by INCREMENTAL WRITES with per-substep deltas
    # inside the plausibility gates — but at its SPAWN spot, outside the bay. The erect
    # account must stay empty: tilt gained outside the bay is worthless (so propping
    # the slab up elsewhere and carrying it in upright cannot earn the band).
    env.reset(seed=101)
    step(10)
    sx0, sy0 = rack_xy(0)
    length, _w, thick = c.slab_size
    acc_before = float(scene.erect_acc[0, 0])
    t = 0.0
    while t < math.radians(80.0):
        t += 0.008
        lz = c.z_work + 0.5 * (length * math.sin(t) + thick * math.cos(t)) + 0.002
        ql = torch.tensor([[math.cos(t / 2), 0.0, math.sin(t / 2), 0.0]], device=device)
        qw = quat_mul(scene.studio.data.root_quat_w[0:1], ql)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = rack_to_world(sx0, sy0, lz)
        st[:, 3:7] = qw
        scene.slabs[0].write_root_state_to_sim(st, all_ids)
        env.step(no_action)
    acc_after = float(scene.erect_acc[0, 0])
    report("oob-sweep")
    s, ok = judge()
    check("out-of-bay sweep: red slab rotated 0->80 deg by incremental writes with "
          "plausible per-substep deltas at its spawn spot — the erect account stays "
          f"empty ({math.degrees(acc_after - acc_before):.2f} deg accrued)",
          acc_after - acc_before < math.radians(2.0) and tilt_deg(0) > 60.0
          and not ok and s <= 0.05)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", fin_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.slab_easel")
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
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 — die loudly, don't wait for the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL (exception: {exc!r})", flush=True)
        os._exit(2)

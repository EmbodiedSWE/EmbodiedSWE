"""Smoke / rubric-REJECTION battery for CounterweighBasketScene (sim_gen task
`living_room_scene1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket_i80`) —
NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — metal counterweight dropped onto the tray, box
dropped into the basket, beam settles level hands-off — is the acceptance evidence that
the rubric ACCEPTS a correct outcome). Every teleport here is instrumentation that
CONSTRUCTS a wrong (or partial) outcome as a settled state and asserts the rubric
REJECTS it; no probe in this battery ever reaches success(), and a final audit check
asserts exactly that.

  1-2. settle/no-NaN   — reset layout settles finite: EMPTY beam self-levelled
                         (|pitch| <= 3 deg — the keel works), objects at their floor
                         slots, all still, score ~0;
  3-4. randomization   — READBACK over 8 seeded resets: metal/foam slot assignment
                         flips (Bernoulli swap) with the two blocks always opposite,
                         per-object xy jitter and stand yaw spread are real;
  5.  null policy      — 240 idle steps -> score ~0, no success;
  6.  seed strategy    — the end state the seed's ENTIRE plan produces here (put the
                         box in the basket, nothing else): beam tips to ~-16 deg, box
                         still inside -> NOT success, score <= 0.35 (the balance is
                         the interlock: containment alone is not the goal);
  7.  latched credit   — teleporting the box back OUT afterwards leaves the latched
                         score unchanged (credit does not evaporate), still no success;
  8.  wrong block      — WHITE FOAM decoy on the tray + box in the basket: beam still
                         ~-14 deg -> NOT success, no counterweight credit, score
                         <= 0.35 (material identification is load-bearing);
  9.  wrong place      — metal dumped INTO THE BASKET on top of the box (both loads
                         one end): beam pinned at the -20 deg stop -> NOT success,
                         score <= 0.35 (the counterweight must go on the OPPOSITE end);
  10. mirrored         — metal in the basket, box on the tray: the beam settles LEVEL
                         and STILL (matched masses!) yet -> NOT success, score ~0 (the
                         goal predicate demands the BOX inside the BASKET, not merely
                         a balanced beam);
  11. near-miss on-bar — metal correctly on the tray, box resting on the BAR just
                         outside the basket's inner wall -> NOT success, score <= 0.25
                         (beam-frame containment window, not "on the beam somewhere");
  12. mid-swing        — the correct outcome constructed, then JUDGED DURING the
                         counterswing: the beam passes through the level band with
                         real angular velocity and success never fires (the
                         consecutive-still counter is the anti-turning-point gate);
                         the box is removed before the swing decays;
  13. rejection audit  — success() was never True at ANY judged point;
  14. final no-NaN     — all task-object states finite at the end.

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
# RTX recipe: kit mis-decodes the pod driver version and silently rejects RTX -> the
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
    env = ENVS.get("simgen.counterweigh_basket")().build(num_envs=args.num_envs,
                                                         device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.45, -1.00, 0.85)) + o),
                                tuple(np.array((0.42, 0.00, 0.18)) + o),
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

    def pitch() -> float:
        return float(scene.beam_pitch_deg()[0])

    def report(tag: str) -> None:
        bl = scene._beam_local(scene.box)[0]
        ml = scene._beam_local(scene.metal)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | pitch={pitch():+6.2f}deg "
              f"box_loc=({float(bl[0]):+.3f},{float(bl[1]):+.3f},{float(bl[2]):+.3f}) "
              f"metal_loc=({float(ml[0]):+.3f},{float(ml[1]):+.3f},{float(ml[2]):+.3f}) "
              f"in_basket={bool(scene._in_basket(scene.box)[0])} "
              f"on_tray={bool(scene._on_tray(scene.metal)[0])} "
              f"cw={bool(scene._cw_latch[0])} bx={bool(scene._box_latch[0])} "
              f"still_n={int(scene._still_n[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_beam_local(body, x: float, y: float, z: float) -> None:
        """Teleport `body` to a beam-local point of the beam's CURRENT pose, oriented
        WITH the beam (probe constructor). A small -z probe velocity hard-resets the
        stillness counter — a zero-velocity teleport reads vacuously still before any
        physics has run."""
        b_pos = scene.beam.data.root_pos_w
        b_quat = scene.beam.data.root_quat_w
        loc = torch.tensor([x, y, z], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = b_pos + quat_apply(b_quat, loc)
        st[:, 3:7] = b_quat
        st[:, 9] = -0.10
        body.write_root_state_to_sim(st, all_ids)

    def place_world(body, x: float, y: float, z: float) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = 1.0
        st[:, 9] = -0.10
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def settle(budget: int, min_steps: int = 60) -> int:
        """Step until the scene's own stillness latch fires (always at least
        `min_steps` — never consult a stillness predicate before physics has run)."""
        for i in range(budget):
            step(1)
            if i + 1 >= min_steps and bool(scene.still()[0]):
                return i + 1
        return budget

    # resting heights (beam frame)
    blk_rest = c.floor_z + 0.004 + c.block_size / 2  # block CoM on the tray/basket floor
    box_rest = c.floor_z + 0.004 + c.box_dims[2] / 2  # box CoM on the basket floor
    rim_top = c.floor_z + 0.004 + c.rim_h  # tray rim top plane

    def obj_xy(body) -> tuple[float, float]:
        p = (body.data.root_pos_w - scene.env_origins)[0]
        return float(p[0]), float(p[1])

    def stand_yaw() -> float:
        q = scene.stand.data.root_quat_w[0]
        return math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))

    def finite_all() -> bool:
        return bool(torch.isfinite(scene.stand.data.root_state_w).all()
                    and torch.isfinite(scene.beam.data.root_state_w).all()
                    and torch.isfinite(scene.box.data.root_state_w).all()
                    and torch.isfinite(scene.metal.data.root_state_w).all()
                    and torch.isfinite(scene.foam.data.root_state_w).all())

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("show")
    bz = float((scene.box.data.root_pos_w - scene.env_origins)[0, 2])
    mz = float((scene.metal.data.root_pos_w - scene.env_origins)[0, 2])
    check("settle: states finite, EMPTY beam self-levelled (|pitch| <= 3 deg — the keel "
          "CoM works), box and blocks lying at their floor slots, all still",
          finite_all() and abs(pitch()) <= 3.0
          and abs(bz - c.box_dims[2] / 2) < 0.012 and abs(mz - c.block_size / 2) < 0.012
          and bool(scene.still()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        mx, my = obj_xy(scene.metal)
        fx, fy = obj_xy(scene.foam)
        bx_, by_ = obj_xy(scene.box)
        reads.append((mx, my, fx, fy, bx_, by_, 1.0 if my > 0 else 0.0, stand_yaw()))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (metal_x, metal_y, foam_x, foam_y, box_x, "
          f"box_y, metal_on_left, stand_yaw_deg):\n{arr}", flush=True)
    flags = arr[:, 6]
    check("randomization: metal/foam slot assignment flips across seeded resets AND the "
          "two blocks always take opposite slots (readback)",
          0.0 < flags.mean() < 1.0
          and all((r[1] > 0) != (r[3] > 0) for r in reads))
    jit = 0.0
    for flag in (0.0, 1.0):
        grp = arr[flags == flag]
        if len(grp) >= 2:
            jit = max(jit, float((grp[:, 0:2].max(axis=0) - grp[:, 0:2].min(axis=0)).max()))
    box_jit = float((arr[:, 4:6].max(axis=0) - arr[:, 4:6].min(axis=0)).max())
    yaw_spread = float(arr[:, 7].max() - arr[:, 7].min())
    check("randomization: per-slot block jitter (> 4 mm), box jitter (> 4 mm) and stand "
          "yaw spread (> 2 deg) are real (readback)",
          jit > 0.004 and box_jit > 0.004 and yaw_spread > 2.0)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy: box in basket, nothing else ==============
    # The seed's ENTIRE plan — pick up the cream cheese box, put it in the basket — ends
    # HERE with the box inside but the beam tipped to its loaded equilibrium (~-16 deg).
    torch.manual_seed(41)
    env.reset()
    step(30)
    place_beam_local(scene.box, c.arm, 0.0, box_rest + 0.020)
    k6 = settle(1800)
    report(f"seed-strat({k6})")
    s6, ok = judge()
    check("seed strategy: box in the basket with NO counterweight — the beam tips to "
          "<= -12 deg, box still inside: NOT success, score <= 0.35 (containment alone "
          "is not the goal; the balance is the interlock)",
          bool(scene._in_basket(scene.box)[0]) and pitch() <= -12.0
          and not ok and s6 <= c.w_box + 1e-4)

    # =========================== 7. latched credit survives regression ======================
    place_world(scene.box, 0.12, 0.32, c.box_dims[2] / 2 + 0.002)
    settle(900)
    report("regressed")
    s7, ok = judge()
    check("latched credit: teleporting the box back OUT of the basket leaves the latched "
          f"score unchanged ({s6:.3f} -> {s7:.3f}), still no success",
          abs(s7 - s6) < 1e-3 and not bool(scene._in_basket(scene.box)[0]) and not ok)

    # =========================== 8. wrong block: the foam decoy =============================
    # White foam (0.03 kg) on the tray + box in the basket: net moment still tips the
    # beam ~-14 deg. Material identification is load-bearing.
    torch.manual_seed(51)
    env.reset()
    step(30)
    place_beam_local(scene.foam, -c.arm, 0.0, blk_rest + 0.030)
    settle(900)
    place_beam_local(scene.box, c.arm, 0.0, box_rest + 0.020)
    k8 = settle(1800)
    report(f"foam-decoy({k8})")
    s8, ok = judge()
    check("wrong block: FOAM decoy on the tray + box in the basket — beam still tipped "
          "(<= -10 deg): NOT success, no counterweight credit, score <= 0.35",
          bool(scene._in_basket(scene.box)[0]) and pitch() <= -10.0
          and not bool(scene._cw_latch[0]) and not ok and s8 <= c.w_box + 1e-4)

    # =========================== 9. wrong place: metal dumped in the basket =================
    # Both loads on ONE end: the beam pins at the -20 deg stop. The counterweight must
    # go on the OPPOSITE end — dumping everything into the basket (the seed's receptacle
    # instinct) is rejected.
    torch.manual_seed(61)
    env.reset()
    step(30)
    place_beam_local(scene.box, c.arm, 0.0, box_rest + 0.020)
    settle(1200)
    place_beam_local(scene.metal, c.arm, 0.0, box_rest + c.box_dims[2] / 2
                     + c.block_size / 2 + 0.020)
    k9 = settle(1800)
    report(f"stacked({k9})")
    s9, ok = judge()
    check("wrong place: metal dumped INTO the basket on top of the box — beam pinned "
          "near the -20 deg stop (<= -17 deg): NOT success, no counterweight credit, "
          "score <= 0.35",
          pitch() <= -17.0 and not bool(scene._cw_latch[0])
          and not ok and s9 <= c.w_box + 1e-4)

    # =========================== 10. mirrored: level and still, yet rejected ================
    # Metal in the BASKET, box on the TRAY: matched 0.20 kg masses — the beam settles
    # LEVEL and STILL. Only the box-in-basket clause rejects: the goal is not "balance
    # the beam somehow".
    torch.manual_seed(71)
    env.reset()
    step(30)
    place_beam_local(scene.box, -0.21, 0.0, rim_top + c.box_dims[2] / 2 + 0.010)
    settle(1200)
    place_beam_local(scene.metal, c.arm, 0.0, blk_rest + 0.020)
    k10 = settle(2400)
    report(f"mirrored({k10})")
    s10, ok = judge()
    check("mirrored: metal in the basket, box on the tray — the beam settles LEVEL "
          "(|pitch| <= 8 deg) and STILL, yet NOT success and score ~0 (the goal "
          "predicate demands the BOX inside the BASKET, not merely a balanced beam)",
          abs(pitch()) <= c.level_band_deg and bool(scene.still()[0])
          and not bool(scene._in_basket(scene.box)[0])
          and not ok and s10 <= 0.02)

    # =========================== 11. near-miss: box on the bar ==============================
    # Metal correctly on the tray; box resting on the BAR just outside the basket's
    # inner wall. Beam-frame containment window rejects "on the beam somewhere".
    torch.manual_seed(81)
    env.reset()
    step(30)
    place_beam_local(scene.metal, -c.arm, 0.0, blk_rest + 0.030)
    settle(1200)
    bar_top = -0.060 + 0.008  # bar centre z + half thickness
    place_beam_local(scene.box, 0.10, 0.0, bar_top + c.box_dims[2] / 2 + 0.010)
    k11 = settle(1800)
    report(f"on-bar({k11})")
    bl = scene._beam_local(scene.box)[0]
    s11, ok = judge()
    check("near-miss on-bar: metal on the tray, box on the BAR outside the basket's "
          "inner wall — NOT success, counterweight credit only (score <= 0.25)",
          not bool(scene._in_basket(scene.box)[0]) and float(bl[0]) < 0.16
          and not ok and s11 <= c.w_cw + 1e-4)

    # =========================== 12. mid-swing: the turning-point gate ======================
    # Construct the CORRECT outcome, then judge DURING the counterswing: the beam passes
    # through the level band carrying real angular velocity; the consecutive-still
    # counter must keep success False. The box is removed before the swing decays, so
    # this probe never reaches a settled success (audit stays honest).
    torch.manual_seed(91)
    env.reset()
    step(30)
    place_beam_local(scene.metal, -c.arm, 0.0, blk_rest + 0.030)
    settle(1200)
    place_beam_local(scene.box, c.arm, 0.0, box_rest + 0.020)
    max_omega, level_moving, mid_ok, mid_still = 0.0, False, False, False
    for _ in range(90):
        step(1)
        w = float(scene.beam.data.root_ang_vel_w[0].norm())
        max_omega = max(max_omega, w)
        if abs(pitch()) <= c.level_band_deg and w > c.still_omega:
            level_moving = True
        _s, _ok = judge()
        mid_ok = mid_ok or _ok
        mid_still = mid_still or bool(scene.still()[0])
    report("mid-swing")
    place_world(scene.box, 0.12, 0.32, c.box_dims[2] / 2 + 0.002)
    settle(900)
    report("swing-cleanup")
    check("mid-swing: during the counterswing the beam crossed the level band with real "
          f"angular velocity (max |omega|={max_omega:.2f} rad/s) and success NEVER fired "
          "— the consecutive-still counter rejects turning-point stillness",
          max_omega > c.still_omega and level_moving and not mid_ok and not mid_still)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.counterweigh_basket")
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

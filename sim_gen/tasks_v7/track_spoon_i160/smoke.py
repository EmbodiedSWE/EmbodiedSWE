"""Smoke / rubric-REJECTION battery for UtensilBalanceScene (sim_gen task
`track_spoon_i160`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — probe the empty pan with a wrong utensil, read the
pinned beam, swap to the matching utensil, watch the beam settle level — is the
acceptance evidence that the rubric ACCEPTS a correct outcome). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome and asserts the rubric
REJECTS it; no probe in this battery ever reaches success(), and a final audit check
asserts exactly that. (post_step latches credit every substep, so every near-success
construct deliberately carries a PERMANENT violation — a utensil perched on the base
slab, or a spare weight parked too close.)

  1-2. settle/no-NaN      — the reset state is a visibly LOADED scale: the hidden
                            counterweight pins the beam at its stop TOWARD its own pan
                            (|pitch| >= 20 deg, sign matches the drawn side), utensils
                            flat on the floor; score ~0, no success;
  3.  authored masses     — get_masses() readback: fork/spoon/ladle = 25/45/75 g and
                            the three identical-LOOKING weights mirror them exactly
                            (custom spawners author MassAPI mass — verified, since cfg
                            mass_props are ignored on this stack);
  4-5. randomization      — READBACK over 8 seeded resets: stand xy + yaw and the
                            utensil floor slots all move; the hidden match identity
                            takes >= 2 values, the weight's pan takes BOTH sides, and
                            each episode's active weight really rests in its declared
                            pan (in_pan readback);
  6.  null policy         — 240 idle steps -> score ~0, no success;
  7.  SEED strategy       — the seed's whole plan ("carry the spoon and set it down at
                            a goal") = the spoon transported and SET DOWN next to the
                            scale on its base slab: mechanism never engaged -> score
                            ~0, NOT success;
  8.  wrong utensil       — a non-matching utensil dropped into the free pan: the beam
                            answers by PINNING at its stop (mass readout readback),
                            engagement latches its 0.20 — and nothing more, NOT
                            success;
  9.  level readout       — positive control of the mechanism WITH a permanent
                            violation: one wrong utensil parked ON the base slab, the
                            MATCH dropped into the free pan -> the beam settles LEVEL
                            (readback) and `balanced` latches (score 0.50), yet the
                            floor clause alone rejects success;
  10. same-pan stuffing   — the match utensil dropped into the SAME pan as the weight:
                            beam stays pinned, engagement does NOT latch (the utensil
                            must sit OPPOSITE the weight), score ~0, NOT success;
  11. no weight           — the active weight removed from its pan: the keel returns
                            the empty beam LEVEL (readback — level alone is not the
                            task), arrangement False, score ~0, NOT success;
  12. parked-weight clause— otherwise-PERFECT terminal state (match opposite the
                            weight, level, rejects on the floor) but a spare weight
                            moved to 0.30 m from the stand: the parked_far clause
                            alone rejects success (score capped at the latched 0.50);
  13. settle gate         — mid-swing after the match lands, settled() reads False
                            (beam angular-velocity readback) and success is False at
                            that judged instant; settled() True once the swing dies;
  14. latched credit      — the match utensil teleported back OUT of its pan: the
                            latched 0.50 survives unchanged, still NOT success;
  15. rejection audit     — success() was never True at ANY judged point;
  16. final no-NaN        — all task-object states finite; frames.npz saved.

Run (forge): python -u -m simgen_tasks.track_spoon_i160.smoke --headless
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
    env = ENVS.get("simgen.utensil_balance")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.45, -1.05, 0.85)) + o),
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

    def deg(x: float) -> float:
        return math.degrees(x)

    def stand_pose() -> tuple[float, float, float]:
        sp = (scene.stand.data.root_pos_w - scene.env_origins)[0]
        q = scene.stand.data.root_quat_w[0]
        return float(sp[0]), float(sp[1]), 2.0 * math.atan2(float(q[3]), float(q[0]))

    def sframe(lx: float, ly: float) -> tuple[float, float]:
        sx, sy, yaw = stand_pose()
        cy, si = math.cos(yaw), math.sin(yaw)
        return (sx + cy * lx - si * ly, sy + si * lx + cy * ly)

    def stand_quat() -> tuple[float, float, float, float]:
        _sx, _sy, yaw = stand_pose()
        return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

    def report(tag: str) -> None:
        wl, wr = scene.wt_in_pans()
        ml, mr = scene.match_in_pans()
        s, ok = judge()
        print(f"[smoke] {tag:16s} | pitch={deg(float(scene.beam_pitch()[0])):+6.2f}deg "
              f"wt_in=({bool(wl[0])},{bool(wr[0])}) match_in=({bool(ml[0])},{bool(mr[0])}) "
              f"arr={bool(scene.arrangement()[0])} level={bool(scene.level()[0])} "
              f"floor={bool(scene.others_on_floor()[0])} parked={bool(scene.parked_far()[0])} "
              f"settled={bool(scene.settled()[0])} eng={bool(scene._engaged[0])} "
              f"bal={bool(scene._balanced[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, wx: float, wy: float, z: float,
              quat=(1.0, 0.0, 0.0, 0.0), settle_steps: int = 30) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = wx, wy, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def drop_into_pan(ute, pan_k: int, oy: float = 0.0, settle_steps: int = 60) -> None:
        """Hover the utensil over pan k's tray (live pose, optional stand-local y
        offset), 30 mm up, aligned with the stand yaw, then let gravity load the pan."""
        pp = (scene.pans[pan_k].data.root_pos_w - scene.env_origins)[0]
        _sx, _sy, yaw = stand_pose()
        wx = float(pp[0]) - math.sin(yaw) * oy
        wy = float(pp[1]) + math.cos(yaw) * oy
        place(ute, wx, wy, float(pp[2]) - c.hang_depth + 0.030, stand_quat(), settle_steps)

    def on_slab(ute) -> None:
        """Park a utensil ON the stand's base slab (stand-local (0, 0.075)) — a
        PERMANENT floor-clause violation (z above floor_z_max, inside clear_r)."""
        wx, wy = sframe(0.0, 0.075)
        place(ute, wx, wy, c.base_t + 0.006, stand_quat(), settle_steps=60)

    def fin_all() -> bool:
        ok = True
        for b in (scene.stand, scene.beam, *scene.pans, *scene.utes, *scene.wts):
            ok = ok and bool(torch.isfinite(b.data.root_state_w).all())
        return ok

    def ute_z(j: int) -> float:
        return float((scene.utes[j].data.root_pos_w - scene.env_origins)[0][2])

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(600)  # the counterweight tips the beam to its hard stop (~5 s)
    report("reset-settled")
    side = float(scene._side[0])
    pitch = float(scene.beam_pitch()[0])
    utes_floor = all(ute_z(j) <= c.floor_z_max for j in range(3))
    check("settle: states finite; the hidden counterweight PINS the beam at its stop "
          "toward its own pan (|pitch| >= 20 deg, sign matches the drawn side — the "
          "reset state is a visibly loaded scale); utensils flat on the floor",
          fin_all() and abs(deg(pitch)) >= 20.0 and pitch * side > 0
          and utes_floor and bool(scene.settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.03), no success", s <= 0.03 and not ok)

    # =========================== 3. authored masses readback ================================
    um = [float(scene.utes[j].root_physx_view.get_masses().flatten()[0]) for j in range(3)]
    wm = [float(scene.wts[j].root_physx_view.get_masses().flatten()[0]) for j in range(3)]
    print(f"[smoke] mass readback: utensils={um} weights={wm} "
          f"declared={list(c.ute_masses)}", flush=True)
    check("authored masses: fork/spoon/ladle read back 25/45/75 g and the three "
          "identical-looking weights mirror them (get_masses readback — custom "
          "spawners must author MassAPI themselves)",
          all(abs(um[j] - c.ute_masses[j]) < 1e-3 for j in range(3))
          and all(abs(wm[j] - c.ute_masses[j]) < 1e-3 for j in range(3)))

    # =========================== 4-5. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(5)
        sx, sy, syaw = stand_pose()
        match = int(scene._match[0])
        sd_side = float(scene._side[0])
        pan_k = 1 if sd_side > 0 else 0
        wt_in = bool(scene.in_pan(scene.wts[match], pan_k)[0])
        u0 = (scene.utes[0].data.root_pos_w - scene.env_origins)[0]
        reads.append((sx, sy, syaw, float(u0[0]), float(u0[1]),
                      float(match), sd_side, float(wt_in)))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (stand_x, stand_y, stand_yaw, ute0_x, "
          f"ute0_y, match, side, wt_in_declared_pan):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: stand xy + yaw and the utensil floor slots all vary across "
          "seeded resets (readback)",
          spread[0] > 0.005 and spread[1] > 0.005 and spread[2] > 0.05
          and spread[3] > 0.005 and spread[4] > 0.03)
    check("randomization: the hidden match identity takes >= 2 values, the weight's "
          "pan takes BOTH sides, and the active weight rests in its declared pan "
          "every episode (in_pan readback)",
          len(set(arr[:, 5])) >= 2 and arr[:, 6].min() < -0.5 and arr[:, 6].max() > 0.5
          and bool(arr[:, 7].min() > 0.5))

    # =========================== 6. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.03 and not ok)

    # =========================== 7. SEED strategy ===========================================
    # The seed's whole plan is "carry the (already-grasped) spoon along a path and set
    # it down at the goal". Here that end state — the spoon transported and SET DOWN
    # next to the scale, on its base slab — must be worthless: the mechanism is never
    # engaged and the beam stays pinned by the counterweight.
    env.reset(seed=41)
    step(120)
    on_slab(scene.utes[1])  # the spoon, the seed's protagonist
    step(60)
    report("seed-strategy")
    s, ok = judge()
    spoon_z = ute_z(1)
    check("seed strategy (the spoon carried and SET DOWN by the scale, on its base "
          "slab): mechanism never engaged, beam still pinned -> score <= 0.03, NOT "
          "success",
          spoon_z > c.floor_z_max and abs(deg(float(scene.beam_pitch()[0]))) >= 20.0
          and s <= 0.03 and not ok)

    # =========================== 8. wrong utensil: the beam answers =========================
    env.reset(seed=51)
    step(600)
    match = int(scene._match[0])
    side = float(scene._side[0])
    free_pan = 0 if side > 0 else 1
    wrong = (match + 1) % 3
    drop_into_pan(scene.utes[wrong], free_pan, settle_steps=60)
    step(840)  # wrong mass -> the beam pins at the OTHER stop (~7 s to settle)
    report("wrong-utensil")
    pitch = float(scene.beam_pitch()[0])
    s, ok = judge()
    check("wrong utensil in the free pan: the beam PINS at its stop (mass readout — "
          "level would need < 3 g of error, the smallest wrong gap is 20 g), "
          "engagement latches 0.20 and nothing more, NOT success",
          abs(deg(pitch)) >= c.level_deg + 2.0 and bool(scene._engaged[0])
          and bool(scene.in_pan(scene.utes[wrong], free_pan)[0])
          and 0.15 <= s <= 0.25 and not ok)

    # =========================== 9. level readout + floor-clause violation ==================
    # Positive control of the mechanism: the MATCH really levels the beam. The
    # construct carries a PERMANENT violation (a wrong utensil parked ON the base
    # slab) so the latching rubric can never reach success while we watch it level.
    env.reset(seed=61)
    step(600)
    match = int(scene._match[0])
    side = float(scene._side[0])
    free_pan = 0 if side > 0 else 1
    wrong = (match + 1) % 3
    on_slab(scene.utes[wrong])  # permanent floor-clause violation, placed FIRST
    drop_into_pan(scene.utes[match], free_pan, settle_steps=60)
    for _ in range(30):
        step(60)
        if bool(scene.level()[0]) and bool(scene.settled()[0]):
            break
    report("match-level")
    pitch = float(scene.beam_pitch()[0])
    s_level, ok = judge()
    check("level readout: the MATCH utensil levels the beam (|pitch| <= level_deg by "
          "readback) and `balanced` latches its 0.30 (score 0.50 with engagement) — "
          "but a wrong utensil parked ON the base slab violates the floor clause: "
          "NOT success",
          abs(deg(pitch)) <= c.level_deg and bool(scene._balanced[0])
          and not bool(scene.others_on_floor()[0])
          and 0.45 <= s_level <= 0.55 and not ok)

    # =========================== 10. same-pan stuffing ======================================
    env.reset(seed=71)
    step(600)
    match = int(scene._match[0])
    side = float(scene._side[0])
    wt_pan = 1 if side > 0 else 0
    drop_into_pan(scene.utes[match], wt_pan, oy=0.040, settle_steps=90)
    step(240)
    report("same-pan")
    pitch = float(scene.beam_pitch()[0])
    s, ok = judge()
    check("same-pan stuffing: the match dropped into the pan WITH the weight — beam "
          "stays pinned toward that side, engagement does NOT latch (the utensil "
          "must sit OPPOSITE the weight), score <= 0.03, NOT success",
          bool(scene.in_pan(scene.utes[match], wt_pan)[0])
          and abs(deg(pitch)) >= 20.0 and pitch * side > 0
          and not bool(scene._engaged[0]) and s <= 0.03 and not ok)

    # =========================== 11. level without the weight ===============================
    env.reset(seed=81)
    step(120)
    match = int(scene._match[0])
    place(scene.wts[match], 1.5, 1.2, 0.001, settle_steps=60)
    step(900)  # the keel returns the empty beam to level (~overdamped, a few s)
    report("no-weight")
    s, ok = judge()
    check("no weight: with the counterweight removed the keel returns the empty beam "
          "LEVEL (readback) — but arrangement is False (level alone is not the "
          "task): score <= 0.03, NOT success",
          bool(scene.level()[0]) and not bool(scene.arrangement()[0])
          and s <= 0.03 and not ok)

    # =========================== 12-13. parked-weight clause + settle gate ==================
    # Otherwise-PERFECT terminal state: match opposite the weight, beam level, both
    # rejects on the floor — but a spare weight moved to 0.30 m (< parked_r = 0.50)
    # is a PERMANENT violation, so the latching rubric can never reach success.
    env.reset(seed=91)
    step(600)
    match = int(scene._match[0])
    side = float(scene._side[0])
    free_pan = 0 if side > 0 else 1
    spare = (match + 1) % 3
    sx, sy, _yaw = stand_pose()
    place(scene.wts[spare], sx, sy + 0.30, 0.001, settle_steps=30)  # violation FIRST
    drop_into_pan(scene.utes[match], free_pan, settle_steps=60)
    step(20)
    settled_mid = bool(scene.settled()[0])
    omega_mid = float(scene.beam.data.root_ang_vel_w[0].norm())
    s_mid, ok_mid = judge()
    for _ in range(30):
        step(60)
        if bool(scene.level()[0]) and bool(scene.settled()[0]):
            break
    report("parked-violation")
    s_park, ok = judge()
    check("parked-weight clause: otherwise-perfect terminal state (arrangement + "
          "level + rejects on the floor + settled, all True by readback) but a spare "
          "weight at 0.30 m — parked_far alone rejects success, score stays at the "
          "latched 0.50",
          bool(scene.arrangement()[0]) and bool(scene.level()[0])
          and bool(scene.others_on_floor()[0]) and bool(scene.settled()[0])
          and not bool(scene.parked_far()[0])
          and 0.45 <= s_park <= 0.55 and not ok)
    check("settle gate: mid-swing after the match lands, settled() reads False (beam "
          f"angular velocity {omega_mid:.3f} rad/s > {c.settle_omega} by readback) "
          "and success is False at that judged instant; settled() True once the "
          "swing dies",
          not settled_mid and omega_mid > c.settle_omega and not ok_mid
          and bool(scene.settled()[0]))

    # =========================== 14. latched credit survives regression =====================
    place(scene.utes[match], -0.10, 0.30, 0.020, settle_steps=90)
    report("match-removed")
    s_out, ok = judge()
    check("latched credit: teleporting the match back OUT of its pan leaves the "
          "latched 0.50 unchanged (and still no success)",
          abs(s_out - s_park) < 0.02 and s_out >= 0.45 and not ok)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", fin_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.utensil_balance")
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

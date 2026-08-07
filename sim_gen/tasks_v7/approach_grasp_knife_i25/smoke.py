"""Smoke / rubric-REJECTION battery for UnderpinSwapScene (sim_gen task
`approach_grasp_knife_i25`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — slide the blue column under the deck and drag the
loaded red column out under contact dynamics — is the acceptance evidence that the
rubric ACCEPTS a correct outcome). Every teleport here is instrumentation that
CONSTRUCTS a wrong (or partial) outcome as a settled state and asserts the rubric
REJECTS it; no probe in this battery ever reaches success(), and a final audit check
asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite; deck level at ride height on
                            pedestal + red column (readback), score ~0 at rest;
  3-4. randomization      — READBACK over 6 seeded resets: fixture xy + yaw, blue /
                            decoy / pad xy all move;
  5.  null policy         — 360 idle steps -> score ~0, no success, no spoil;
  6.  SEED strategy       — the seed's plan applied to the judged object: grasp and
                            HOIST the deck -> spoil latch fires; putting the deck back
                            perfectly does NOT un-fail it (score stays capped);
  7.  order violation     — red column removed BEFORE the blue is seated -> the deck
                            collapses (physics), spoiled, and NO extraction credit
                            (the ordering gate held it at ~0);
  8.  rebuilt end state   — after the collapse, the full perfect-looking final
                            arrangement is rebuilt by teleport -> every geometric
                            predicate holds, still NOT success (irreversible latch);
  9.  wrong object        — the short YELLOW decoy seated instead of the blue -> the
                            deck sags past the spoil threshold: physics itself rejects
                            the decoy, and seat credit stays 0 (identity matters);
  10. near-miss           — blue column under the deck but OUTSIDE the slot-x
                            tolerance (deck still safely supported, nothing spoiled)
                            -> NOT success, only approach credit;
  11. wrong park          — full correct swap but the red column left standing on open
                            ground instead of the green pad -> NOT success, score < 1;
  12. rejection audit     — success() was never True at ANY judged point;
  13. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.approach_grasp_knife_i25.smoke --headless
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
    env = ENVS.get("simgen.underpin_swap")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((0.90, -0.90, 0.75)) + o),
                                tuple(np.array((0.05, 0.00, 0.10)) + o),
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

    def fix_pose() -> tuple[torch.Tensor, float]:
        ap = (scene.fixture.data.root_pos_w - scene.env_origins)[0]
        q = scene.fixture.data.root_quat_w[0]
        return ap, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def report(tag: str) -> None:
        bl = scene.blue_local()[0]
        rl = scene.red_local()[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | blue_loc=({float(bl[0]):+.3f},{float(bl[1]):+.3f}) "
              f"red_loc=({float(rl[0]):+.3f},{float(rl[1]):+.3f}) "
              f"end_z={float(scene.deck_end_z()[0]):.4f} "
              f"tilt={float(scene.deck_tilt_deg()[0]):.2f} "
              f"seat_l={float(scene.seat_latch[0]):.2f} "
              f"ext_l={float(scene.extract_latch[0]):.2f} "
              f"spoiled={bool(scene.spoiled[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_body(body, x: float, y: float, z: float, quat=(1.0, 0.0, 0.0, 0.0),
                   settle_steps: int = 30) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += env.iscene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def place_local(body, x_l: float, y_l: float, z: float, yaw_off: float = 0.0,
                    settle_steps: int = 60) -> None:
        """Place a body at fixture-local (x_l, y_l), world z, yaw = fixture yaw + off."""
        ap, fyaw = fix_pose()
        ca, sa = math.cos(fyaw), math.sin(fyaw)
        wx = float(ap[0]) + ca * x_l - sa * y_l
        wy = float(ap[1]) + sa * x_l + ca * y_l
        half = (fyaw + yaw_off) / 2
        place_body(body, wx, wy, z, quat=(math.cos(half), 0.0, 0.0, math.sin(half)),
                   settle_steps=settle_steps)

    def park_red_on_pad(settle_steps: int = 90) -> None:
        pad = (scene.pad.data.root_pos_w - scene.env_origins)[0]
        _ap, fyaw = fix_pose()
        half = (fyaw + math.pi) / 2
        place_body(scene.red, float(pad[0]), float(pad[1]), c.pad_t + 0.002,
                   quat=(math.cos(half), 0.0, 0.0, math.sin(half)),
                   settle_steps=settle_steps)

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("show")
    rl = scene.red_local()[0]
    deck_z = float((scene.deck.data.root_pos_w - scene.env_origins)[0, 2])
    ez = float(scene.deck_end_z()[0])
    fin0 = bool(torch.isfinite(scene.deck.data.root_state_w).all()
                and torch.isfinite(scene.red.data.root_state_w).all()
                and torch.isfinite(scene.blue.data.root_state_w).all())
    check("settle: states finite; deck level at ride height on pedestal + red column; "
          "red at its bearing station (readback)",
          fin0 and abs(deck_z - c.deck_center_z) < 0.004
          and abs(ez - c.h_red) < 0.004
          and abs(float(rl[0]) - c.red_x) < 0.010 and abs(float(rl[1])) < 0.010
          and float(scene.deck_tilt_deg()[0]) < 1.0 and bool(scene.settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success, no spoil",
          s <= 0.02 and not ok and not bool(scene.spoiled[0]))

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        ap, fyaw = fix_pose()
        bl = (scene.blue.data.root_pos_w - scene.env_origins)[0]
        dc = (scene.decoy.data.root_pos_w - scene.env_origins)[0]
        pd = (scene.pad.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(ap[0]), float(ap[1]), fyaw, float(bl[0]), float(bl[1]),
                      float(dc[0]), float(dc[1]), float(pd[0]), float(pd[1])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (fix_x, fix_y, fix_yaw, blue_xy, decoy_xy, "
          f"pad_xy):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: fixture xy + yaw vary across seeded resets (readback — deck, "
          "red column, slot and band all ride the fixture frame)",
          spread[0] > 0.010 and spread[1] > 0.010 and spread[2] > 0.20)
    check("randomization: blue / decoy / pad xy vary across seeded resets (readback)",
          spread[3] > 0.010 and spread[4] > 0.010 and spread[5] > 0.010
          and spread[6] > 0.010 and spread[7] > 0.010 and spread[8] > 0.010)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(360)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0, no success, no spoil after 360 idle steps",
          s <= 0.05 and not ok and not bool(scene.spoiled[0]))

    # =========================== 6. SEED strategy: grasp + hoist the deck ===================
    # The seed's whole plan is approach-grasp-LIFT the judged object. Applied here: the
    # deck hoisted 12 cm. The spoil latch must fire, and putting the deck back exactly
    # where it was must NOT un-fail the episode.
    torch.manual_seed(41)
    env.reset()
    step(30)
    deck_saved = scene.deck.data.root_state_w.clone()
    st = deck_saved.clone()
    st[:, 2] += 0.12
    st[:, 7:13] = 0.0
    scene.deck.write_root_state_to_sim(st, all_ids)
    step(20)
    report("deck-hoisted")
    s_lift, ok_lift = judge()
    spoiled_lift = bool(scene.spoiled[0])
    st = deck_saved.clone()
    st[:, 7:13] = 0.0
    scene.deck.write_root_state_to_sim(st, all_ids)
    step(90)
    report("deck-restored")
    s, ok = judge()
    check("seed strategy (grasp + hoist the deck): spoil latch fires, score capped",
          spoiled_lift and not ok_lift and s_lift <= 0.10 + 1e-6)
    check("seed strategy: restoring the deck perfectly does NOT un-fail the episode "
          "(irreversible latch)",
          bool(scene.spoiled[0]) and not ok and s <= 0.10 + 1e-6)

    # =========================== 7. order violation: remove red FIRST =======================
    # No replacement seated -> the deck loses its far support and collapses. The spoil
    # latch fires AND the ordering gate keeps extraction credit at ~0.
    torch.manual_seed(51)
    env.reset()
    step(30)
    place_body(scene.red, 0.60, 0.45, 0.0005, settle_steps=150)
    report("red-first")
    s, ok = judge()
    ez = float(scene.deck_end_z()[0])
    check("order violation (red removed before blue seated): deck collapses "
          f"(end_z={ez:.3f} < {c.fall_end_z}), spoiled, NO extraction credit, score <= 0.10",
          bool(scene.spoiled[0]) and ez < c.fall_end_z and not ok
          and float(scene.extract_latch[0]) < 0.05 and s <= 0.10 + 1e-6)

    # =========================== 8. rebuilt perfect end state still rejected ================
    # Continue from the collapse: rebuild the EXACT goal arrangement by teleport (blue
    # seated, deck level in its rails, red on the pad). Every geometric predicate now
    # holds — the latched spoil must still reject it.
    # The fallen deck lies over the slot, so blue and deck poses are written in the
    # SAME frame (no steps in between) — level deck above, blue upright beneath it.
    ap, fyaw = fix_pose()
    ca, sa = math.cos(fyaw), math.sin(fyaw)
    half = fyaw / 2
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = float(ap[0]) + ca * c.slot_x
    st[:, 1] = float(ap[1]) + sa * c.slot_x
    st[:, 2] = 0.0005
    st[:, 3], st[:, 6] = math.cos(half), math.sin(half)
    st[:, 0:3] += env.iscene.env_origins
    scene.blue.write_root_state_to_sim(st, all_ids)
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1], st[:, 2] = float(ap[0]), float(ap[1]), c.deck_center_z + 0.001
    st[:, 3], st[:, 6] = math.cos(half), math.sin(half)
    st[:, 0:3] += env.iscene.env_origins
    scene.deck.write_root_state_to_sim(st, all_ids)
    step(60)
    park_red_on_pad(settle_steps=90)
    report("rebuilt")
    s, ok = judge()
    geom = (bool(scene.blue_seated()[0]), bool(scene.deck_ok()[0]),
            bool(scene.red_clear()[0]), bool(scene.red_on_pad()[0]),
            bool(scene.settled()[0]))
    print(f"[smoke] rebuilt predicates (seated, deck_ok, clear, on_pad, settled)={geom}",
          flush=True)
    check("rebuilt end state after the collapse: all geometric predicates hold, yet "
          "NOT success and score stays capped (failure is permanent)",
          all(geom) and not ok and s <= 0.10 + 1e-6)

    # =========================== 9. wrong object: the short decoy ===========================
    # Seat the YELLOW decoy in the slot instead of the blue, then remove red. The deck
    # sags 18 mm at the slot -> the free end drops past the spoil threshold: physics
    # itself rejects the decoy. Identity also matters: no seat credit is granted.
    torch.manual_seed(61)
    env.reset()
    step(30)
    place_local(scene.decoy, c.slot_x, 0.0, 0.0005, yaw_off=0.0, settle_steps=30)
    park_red_on_pad(settle_steps=150)
    report("decoy-seated")
    s, ok = judge()
    ez = float(scene.deck_end_z()[0])
    check("wrong object (short YELLOW decoy seated, red removed): deck sags past the "
          f"spoil threshold (end_z={ez:.3f} < {c.fall_end_z}) — physics rejects the "
          "decoy; no seat credit, no success, score <= 0.10",
          bool(scene.spoiled[0]) and ez < c.fall_end_z and not ok
          and float(scene.seat_latch[0]) < 0.5 and s <= 0.10 + 1e-6)

    # =========================== 10. near-miss: blue outside the slot tolerance =============
    # Blue column under the deck but 45 mm inboard of the slot centre (outside
    # slot_x_tol) — tall enough that the deck stays safely supported (nothing spoils),
    # yet the seat predicate must refuse it and the ordering gate must give no
    # extraction credit.
    torch.manual_seed(71)
    env.reset()
    step(30)
    place_local(scene.blue, 0.025, 0.0, 0.0005, yaw_off=0.0, settle_steps=30)
    park_red_on_pad(settle_steps=150)
    report("near-miss")
    s, ok = judge()
    ez = float(scene.deck_end_z()[0])
    check("near-miss (blue under the deck but outside the slot-x tolerance): deck still "
          "safely supported (no spoil), NOT seated, NOT success, score <= 0.25",
          not bool(scene.spoiled[0]) and ez > c.fall_end_z
          and not bool(scene.blue_seated()[0]) and not ok and s <= 0.25)

    # =========================== 11. wrong park: red on open ground =========================
    # Full correct swap, but the red column is left standing on open ground instead of
    # the green pad: partial credit only, no success.
    torch.manual_seed(81)
    env.reset()
    step(30)
    place_local(scene.blue, c.slot_x, 0.0, 0.0005, yaw_off=0.0, settle_steps=30)
    place_body(scene.red, 0.60, 0.45, 0.0005, settle_steps=150)
    report("wrong-park")
    s, ok = judge()
    check("wrong park (swap complete but red on open ground, not the pad): NOT success, "
          "score in [0.80, 0.99]",
          bool(scene.blue_seated()[0]) and bool(scene.deck_ok()[0])
          and bool(scene.red_clear()[0]) and not bool(scene.red_on_pad()[0])
          and not ok and 0.80 <= s <= 0.99)

    # =========================== 12-13. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.deck.data.root_state_w).all()
           and torch.isfinite(scene.red.data.root_state_w).all()
           and torch.isfinite(scene.blue.data.root_state_w).all()
           and torch.isfinite(scene.decoy.data.root_state_w).all()
           and torch.isfinite(scene.fixture.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.underpin_swap")
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

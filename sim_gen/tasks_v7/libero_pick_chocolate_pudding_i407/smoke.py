"""Smoke / rubric-REJECTION battery for CleatPierScene (sim_gen task
`libero_pick_chocolate_pudding_i407`) — NullRobot, teleported probe states,
RECORDED.

This is NOT a solution (solve.py — slide the plank under the cleat with contact
forces until it cantilevers, then gravity-drop the payload onto the nose — is the
acceptance evidence that the rubric ACCEPTS a correct outcome). Every teleport
here is instrumentation that CONSTRUCTS a wrong (or partial) outcome and asserts
the rubric REJECTS it; no probe in this battery ever reaches success(), and a
final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: plank flat in the lane
                            behind the cleat, payload/decoy on OPPOSITE side
                            strips; score ~0 at rest, no success;
  3-4. randomization      — READBACK over 6 seeded resets: fixture xy + yaw move,
                            plank depth and item scatter move, sides stay opposite;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  SEED strategy       — the seed's whole plan ("carry the object to the goal
                            and set it down") = payload teleported to the exact
                            goal pose over the void with NO plank there
                            (constructed by teleport as pure instrumentation):
                            there is nothing to rest on — it falls 400 mm into the
                            pit (readback), NOT success, score ~0 (the boarding
                            latch requires the plank under the payload);
  7.  plank-alone partial — plank teleported to the anchored cantilever pose, no
                            payload: ovh+anchor credit exactly 0.50, NOT success;
  8.  near-miss           — payload dropped onto the anchored plank but only
                            60 mm past the edge (short of the 100 mm threshold):
                            rests, settled, boarding credit -> exactly 0.70, NOT
                            success;
  9.  no-anchor collapse  — ORDER FORCER: plank teleported flat with its tail 10
                            mm CLEAR of the cleat (unanchored, self-stable), then
                            the payload dropped on the nose: the plank SEE-SAWS
                            off the cliff, both end in the pit (readback), NOT
                            success;
  10. wrong object        — the light DECOY butter box dropped at the goal pose on
                            an anchored plank (physically stable): NOT success by
                            identity, no boarding credit for the decoy;
  11. z-band stack cheat  — payload dropped on TOP of the decoy standing on the
                            plank nose: payload x/y in the goal band but 55 mm too
                            HIGH (readback): NOT success;
  12. latched credit      — full partial progress (0.70), then the payload removed
                            AND the plank slid back to its spawn depth: the
                            latched 0.70 survives unchanged, still no success;
  13. rejection audit     — success() was never True at ANY judged point;
  14. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.libero_pick_chocolate_pudding_i407.smoke --headless
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
    env = ENVS.get("simgen.cleat_pier")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((0.85, -0.95, 0.95)) + o),
                                tuple(np.array((-0.25, 0.00, 0.30)) + o),
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
        fp = (scene.fixture.data.root_pos_w - scene.env_origins)[0]
        q = scene.fixture.data.root_quat_w[0]
        return fp, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def to_world(lx: float, ly: float) -> tuple[float, float]:
        fp, yaw = fix_pose()
        cy, sy = math.cos(yaw), math.sin(yaw)
        return (float(fp[0]) + lx * cy - ly * sy, float(fp[1]) + lx * sy + ly * cy)

    def loc(body) -> tuple[float, float, float]:
        p = scene._fix_local(body.data.root_pos_w)[0]
        return float(p[0]), float(p[1]), float(p[2])

    def report(tag: str) -> None:
        nose, tail = scene.plank_ends()
        px, py, pz = loc(scene.payload)
        s, ok = judge()
        print(f"[smoke] {tag:18s} | plank nose={float(nose[0]):+.3f} "
              f"tail={float(tail[0]):+.3f} flat={bool(scene.plank_flat()[0])} "
              f"payload=({px:+.3f},{py:+.3f},{pz:+.3f}) "
              f"ovh={float(scene.ovh_latch[0]):.0f} anc={float(scene.anchor_latch[0]):.0f} "
              f"brd={float(scene.board_latch[0]):.0f} "
              f"settled={bool(scene.settled()[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_local(body, lx: float, ly: float, z_local: float, yaw_local: float = 0.0,
                    settle_steps: int = 30) -> None:
        """Kinematic probe placement in FIXTURE-LOCAL coords (instrumentation, not
        a solution) + REAL physics steps before judging (the zero-step trap)."""
        wx, wy = to_world(lx, ly)
        _fp, fyaw = fix_pose()
        tot = fyaw + yaw_local
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = wx, wy, c.deck_h + z_local
        st[:, 3], st[:, 6] = math.cos(tot / 2), math.sin(tot / 2)
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def place_plank_tail(tail: float, settle_steps: int = 40) -> None:
        """Instrumentation: lay the plank flat in the lane with its tail at `tail`."""
        place_local(scene.plank, tail + c.plank_l / 2, 0.0, c.plank_t / 2 + 0.002,
                    settle_steps=settle_steps)

    def fin_all() -> bool:
        return bool(torch.isfinite(scene.fixture.data.root_state_w).all()
                    and torch.isfinite(scene.plank.data.root_state_w).all()
                    and torch.isfinite(scene.payload.data.root_state_w).all()
                    and torch.isfinite(scene.decoy.data.root_state_w).all())

    anchored_tail = c.anchor_tail - 0.020  # the solve's stop: comfortably anchored

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    nose, tail = scene.plank_ends()
    px, py, pz = loc(scene.payload)
    dx, dy, dz = loc(scene.decoy)
    t0, t1 = c.plank_tail_range
    check("settle: states finite; plank flat in the lane behind the cleat (tail in the "
          "spawn window), payload/decoy resting on OPPOSITE side strips (readback)",
          fin_all() and bool(scene.plank_flat()[0])
          and t0 - 0.01 <= float(tail[0]) <= t1 + 0.01
          and abs(pz - c.payload_size / 2) < 0.008 and abs(dz - c.decoy_size / 2) < 0.008
          and py * dy < 0 and 0.10 < abs(py) < 0.17 and 0.10 < abs(dy) < 0.17
          and bool(scene.settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(5)
        fp, fyaw = fix_pose()
        _nose, tl = scene.plank_ends()
        ppx, ppy, _ = loc(scene.payload)
        _ddx, ddy, _ = loc(scene.decoy)
        reads.append((float(fp[0]), float(fp[1]), fyaw, float(tl[0]), ppx, ppy, ddy))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (fix_x, fix_y, fix_yaw, plank_tail, "
          f"payload_x, payload_y, decoy_y):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: fixture xy + yaw vary across seeded resets (readback)",
          spread[0] > 0.02 and spread[1] > 0.015 and spread[2] > 0.05)
    check("randomization: plank depth and payload scatter vary, payload and decoy "
          "always on OPPOSITE strips (readback)",
          spread[3] > 0.012 and spread[4] > 0.03
          and all(r[5] * r[6] < 0 for r in reads))

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. SEED strategy ===========================================
    # The seed's whole plan is "grasp the object, carry it to the goal, set it down".
    # Here the goal region is EMPTY AIR: the payload teleported to the exact goal pose
    # with no plank there (instrumentation) has nothing to rest on and falls 400 mm
    # into the pit.
    env.reset(seed=41)
    step(10)
    place_local(scene.payload, c.min_ovh + 0.03, 0.0, c.z_goal, settle_steps=150)
    report("seed-strategy")
    px, py, pz = loc(scene.payload)
    s, ok = judge()
    check("seed strategy (payload SET DOWN at the exact goal pose over the void, no "
          f"plank): it falls into the pit (readback z={pz:+.3f} << deck level), NOT "
          "success, score ~0 — the boarding latch requires the plank under it",
          pz < -0.15 and not ok and s <= 0.02)

    # =========================== 7. plank-alone partial credit ==============================
    env.reset(seed=51)
    step(10)
    place_plank_tail(anchored_tail, settle_steps=60)
    report("plank-alone")
    nose, tail = scene.plank_ends()
    s, ok = judge()
    check("plank-alone: anchored cantilever built but nothing placed on it: "
          "ovh+anchor credit exactly 0.50, NOT success",
          bool(scene.plank_flat()[0]) and float(nose[0]) >= c.ovh_lat_x
          and float(tail[0]) <= c.anchor_tail and abs(s - 0.50) < 0.005 and not ok)

    # =========================== 8. near-miss (short overhang) ==============================
    place_local(scene.payload, 0.06, 0.0, c.z_goal + 0.040, settle_steps=150)
    report("near-miss")
    px, py, pz = loc(scene.payload)
    s, ok = judge()
    check("near-miss: payload rests on the anchored plank only 60 mm past the edge "
          f"(readback x={px:+.3f} < {c.min_ovh:.2f} threshold, z in band, settled): "
          "boarding credit -> exactly 0.70, NOT success",
          0.03 < px < c.min_ovh - 0.005 and abs(pz - c.z_goal) < c.z_tol
          and bool(scene.settled()[0]) and abs(s - 0.70) < 0.005 and not ok)

    # =========================== 9. no-anchor collapse (ORDER FORCER) =======================
    # Plank laid flat with its tail 10 mm CLEAR of the cleat: self-stable alone (CoM
    # 25 mm behind the edge) but UNANCHORED. Loading the nose must see-saw it off the
    # cliff — anchor-before-load is enforced by physics, not by the rubric.
    env.reset(seed=61)
    step(10)
    place_plank_tail(c.cleat_front + 0.005, settle_steps=60)
    nose, tail = scene.plank_ends()
    plank_stable = bool(scene.plank_flat()[0])
    place_local(scene.payload, 0.12, 0.0, c.z_goal + 0.040, settle_steps=200)
    report("no-anchor")
    px, py, pz = loc(scene.payload)
    _plx, _ply, plz = loc(scene.plank)
    s, ok = judge()
    check("no-anchor collapse: unanchored plank (tail 10 mm clear of the cleat) was "
          f"stable alone (readback flat={plank_stable}), but loading the nose "
          f"see-saws it off the cliff: payload z={pz:+.3f}, plank z={plz:+.3f}, both "
          "in the pit, NOT success",
          plank_stable and pz < -0.15 and plz < -0.05
          and not bool(scene.plank_flat()[0]) and not ok)

    # =========================== 10. wrong object ===========================================
    env.reset(seed=71)
    step(10)
    place_plank_tail(anchored_tail, settle_steps=60)
    place_local(scene.decoy, 0.13, 0.0, c.plank_t + c.decoy_size / 2 + 0.030,
                settle_steps=150)
    report("wrong-object")
    ddx, ddy, ddz = loc(scene.decoy)
    s, ok = judge()
    check("wrong object: the DECOY butter box rests at the goal pose on the anchored "
          f"plank (readback x={ddx:+.3f}, z={ddz:+.3f}, settled): NOT success by "
          "identity, no boarding credit (score stays 0.50)",
          ddx >= c.min_ovh and abs(ddy) <= c.y_tol
          and abs(ddz - (c.plank_t + c.decoy_size / 2)) < 0.02
          and bool(scene.settled()[0]) and abs(s - 0.50) < 0.005 and not ok)

    # =========================== 11. z-band stack cheat =====================================
    # Payload dropped on TOP of the decoy standing on the nose: x/y in the goal band
    # but 55 mm too high — the payload must rest ON the plank, not on a pedestal.
    place_local(scene.payload, 0.13, 0.0, c.plank_t + c.decoy_size + c.payload_size / 2 + 0.020,
                settle_steps=150)
    report("stack-cheat")
    px, py, pz = loc(scene.payload)
    s, ok = judge()
    check("z-band stack cheat: payload rests on the decoy on the plank nose "
          f"(readback x={px:+.3f} in band, z={pz:+.3f} > {c.z_goal + c.z_tol:.3f} "
          "band top): NOT success",
          px >= c.min_ovh - 0.02 and pz > c.z_goal + c.z_tol and not ok)

    # =========================== 12. latched credit survives regression =====================
    env.reset(seed=81)
    step(10)
    place_plank_tail(anchored_tail, settle_steps=60)
    place_local(scene.payload, 0.06, 0.0, c.z_goal + 0.040, settle_steps=120)
    s_in, _ = judge()
    place_local(scene.payload, -0.40, c.strip_y, c.payload_size / 2 + 0.002,
                settle_steps=30)
    place_plank_tail((t0 + t1) / 2, settle_steps=40)
    report("regressed")
    s_out, ok = judge()
    check("latched credit: payload removed from the plank AND the plank slid back to "
          f"its spawn depth — the latched credit survives unchanged ({s_in:.3f} -> "
          f"{s_out:.3f}), still no success",
          abs(s_in - 0.70) < 0.005 and abs(s_out - s_in) < 0.005 and not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", fin_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.cleat_pier")
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
    except BaseException as exc:  # noqa: BLE001 — die NOW, Kit teardown hangs
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({type(exc).__name__}: {exc})", flush=True)
        os._exit(1)

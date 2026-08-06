"""Smoke / rubric-REJECTION battery for FacingLaneScene (sim_gen task
`put_groceries_in_cupboard_i329`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — red tin pressed horizontally against the sprung
pusher to open the cavity, lowered behind the sill, released into the spring clamp —
is the acceptance evidence that the rubric ACCEPTS a correct outcome). Every teleport
here is instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled
state and asserts the rubric REJECTS it; no probe in this battery ever reaches
success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN     — reset layout settles finite: plate at home (lane closed),
                           tins at their floor slots, all still, score ~0;
  3-4. randomization     — READBACK over 8 seeded resets: tin-to-slot PERMUTATION
                           varies, per-slot xy jitter and yaw jitter are real;
  5.  null policy        — 240 idle steps -> score ~0, no success;
  6.  seed strategy      — the end state the SEED's plan produces here (lower the
                           grocery onto the shelf from above): the red tin released
                           over the lane just RESTS ON THE ROOF -> score ~0, no
                           success (there is no passive shelf to set it on);
  7.  strip drop         — red tin dropped over the 25 mm open front strip cannot fit
                           through (55 mm tin) and ends perched/outside -> NOT
                           success, never seated or clamped;
  8.  hand-press only    — the plate pressed fully back by a bare force and released:
                           the spring returns it home; press credit only (score
                           <= 0.16), no success — pressing without stocking is empty;
  9.  behind-the-pusher  — red tin settled on the lane floor BEHIND the resting
                           plate (in the lane, not in the facing slot): seated but
                           NOT clamped -> NOT success (the spring clamp is the
                           load-bearing tolerance);
  10. side-lying clamp   — red tin clamped by the spring while LYING ON ITS SIDE ->
                           NOT success (upright clause rejects);
  11. latched credit     — teleporting the red tin back out afterwards leaves the
                           latched score unchanged (credit does not evaporate),
                           still no success;
  12. wrong tin          — the GREEN decoy spring-clamped in the lane, red untouched
                           -> NOT success, score <= 0.16 (color identification is
                           load-bearing);
  13. approach only      — red tin resting against the plinth front on the ground ->
                           score ~0, no success;
  14. rejection audit    — success() was never True at ANY judged point;
  15. final no-NaN       — all task-object states finite at the end.

The both-tins-in-the-lane exclusion state is geometrically unconstructible as a
settled state (single-file lane: a clamped tin leaves < 31 mm of floor behind the
plate, less than any tin footprint) — the decoy-exclusion clause in success() is
defense in depth; the wrong-tin case (#12) is what identity actually protects.

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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.facing_lane")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)
    bay = torch.tensor(c.bay_pos, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.25, -0.95, 0.75)) + o),
                                tuple(np.array((0.42, 0.00, 0.15)) + o),
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

    def comp() -> float:
        return float(scene.compression()[0])

    def report(tag: str) -> None:
        rl = scene._bay_local(scene.red)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | red_loc=({float(rl[0]):+.3f},{float(rl[1]):+.3f},"
              f"{float(rl[2]):.3f}) comp={comp() * 1000:.1f}mm "
              f"entered={bool(scene._entered(scene.red)[0])} "
              f"seated={bool(scene._seated_now(scene.red)[0])} "
              f"clamped={bool(scene._clamped_now()[0])} "
              f"green_in={bool(scene._in_lane(scene.green)[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_local(body, u: float, v: float, z: float, quat=None) -> None:
        """Teleport `body` to a bay-local point (probe constructor: builds in-lane /
        on-top relations directly, walls notwithstanding)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = bay[0] + u
        st[:, 1] = bay[1] + v
        st[:, 2] = bay[2] + z
        if quat is None:
            st[:, 3] = 1.0
        else:
            st[:, 3:7] = torch.tensor(quat, device=device).expand(n, 4)
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def place_world(body, x: float, y: float, z: float) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def write_plate(compression: float) -> None:
        """Probe constructor: follower-only pose write of the plate along its slide
        (what solve.py earns by pressing through the tin, constructed here to build
        clamped states around probe tins). The bay is never moved."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = bay[0] + c.plate_home_u + compression
        st[:, 1] = bay[1]
        st[:, 2] = bay[2] + c.plate_gap + c.plate_h / 2
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        scene.plate.write_root_state_to_sim(st, all_ids)

    def obj_xy(body) -> tuple[float, float]:
        p = (body.data.root_pos_w - scene.env_origins)[0]
        return float(p[0]), float(p[1])

    def red_yaw() -> float:
        q = scene.red.data.root_quat_w[0]
        return math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))

    def finite_all() -> bool:
        return bool(torch.isfinite(scene.plate.data.root_state_w).all()
                    and torch.isfinite(scene.red.data.root_state_w).all()
                    and torch.isfinite(scene.green.data.root_state_w).all()
                    and torch.isfinite(scene.blue.data.root_state_w).all())

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("show")
    rz = float((scene.red.data.root_pos_w - scene.env_origins)[0, 2])
    still = (float(scene.red.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.plate.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    check("settle: states finite, plate at home (lane closed, comp < 5 mm), red tin "
          "at its floor slot, all still, not in the lane",
          finite_all() and comp() < 0.005 and abs(rz - c.tin_h / 2) < 0.012 and still
          and not bool(scene._entered(scene.red)[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    slots = np.array(c.slots)
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        row = []
        for body in (scene.red, scene.green, scene.blue):
            x, y = obj_xy(body)
            d = np.linalg.norm(slots - np.array([x, y]), axis=1)
            row += [x, y, int(np.argmin(d))]
        row.append(red_yaw())
        reads.append(row)
    arr = np.array(reads)
    print(f"[smoke] randomization readback (x,y,slot per tin + red yaw):\n{arr}",
          flush=True)
    assigns = arr[:, [2, 5, 8]].astype(int)
    check("randomization: tin-to-slot assignment is a PERMUTATION every seed and the "
          "red tin's slot varies across seeded resets (readback)",
          all(sorted(a.tolist()) == [0, 1, 2] for a in assigns)
          and len(set(assigns[:, 0].tolist())) >= 2)
    jit = 0.0
    for slot_id in (0, 1, 2):
        grp = arr[assigns[:, 0] == slot_id]
        if len(grp) >= 2:
            jit = max(jit, float((grp[:, 0:2].max(axis=0) - grp[:, 0:2].min(axis=0)).max()))
    yaw_spread = float(arr[:, 9].max() - arr[:, 9].min())
    check("randomization: per-slot xy jitter (> 4 mm) and red yaw spread (> 5 deg) "
          "are real (readback)", jit > 0.004 and yaw_spread > 5.0)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps (the lane stays "
          "closed by itself)", s <= 0.02 and not ok)

    # =========================== 6. seed strategy: lower onto the shelf from above ==========
    # The seed's plan — set the grocery down on the cupboard shelf — executed here:
    # the red tin is released over the lane. There is no shelf; it rests ON THE ROOF.
    torch.manual_seed(41)
    env.reset()
    step(30)
    place_local(scene.red, 0.085, 0.0, c.wall_h + c.tin_h / 2 + 0.006)
    step(240)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy: red tin lowered onto the bay from above just RESTS ON THE "
          "ROOF — never entered, never seated, score ~0 (<= 0.02), no success",
          not bool(scene._entered(scene.red)[0])
          and not bool(scene._seated_now(scene.red)[0]) and not ok and s <= 0.02)

    # =========================== 7. drop into the open front strip ==========================
    # The only roofless patch over the lane is the 25 mm strip between sill and roof
    # edge — narrower than the 55 mm tin. A tin dropped there cannot pass.
    torch.manual_seed(51)
    env.reset()
    step(30)
    place_local(scene.red, 0.032, 0.0, 0.25)
    step(300)
    report("strip-drop")
    s, ok = judge()
    check("strip drop: red tin dropped over the 25 mm open strip cannot fit through "
          "(55 mm tin) — never seated, never clamped, NOT success",
          not bool(scene._seated_now(scene.red)[0]) and not bool(scene._clamped_now()[0])
          and not ok)

    # =========================== 8. hand-press only =========================================
    # Pressing the plate fully back with a bare force and letting go: the spring
    # returns it home. Press credit only; pressing without stocking is empty.
    torch.manual_seed(61)
    env.reset()
    step(30)
    u_hat = torch.tensor([1.0, 0.0, 0.0], device=device).view(1, 1, 3)
    comp_seen = 0.0
    for _ in range(180):
        scene.plate.set_external_force_and_torque(
            (u_hat * 4.0).expand(n, 1, 3).contiguous(), zero_w, env_ids=all_ids,
            is_global=True)
        env.step(no_action)
        comp_seen = max(comp_seen, comp())
    scene.plate.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
    step(150)
    report("hand-press")
    s, ok = judge()
    check("hand-press only: plate pressed fully back by a bare force (comp > 45 mm "
          "observed) and released — the spring returns it home (comp < 8 mm), press "
          "credit only (score <= 0.16), no success",
          comp_seen > 0.045 and comp() < 0.008 and s <= 0.16 and not ok)

    # =========================== 9. behind-the-pusher near-miss =============================
    # Red tin settled on the lane floor BEHIND the resting plate: in the lane, upright,
    # seated — but the facing slot is empty and the spring is not held. Not clamped.
    torch.manual_seed(71)
    env.reset()
    step(30)
    place_local(scene.red, 0.085, 0.0, c.tin_h / 2 + 0.003)
    step(150)
    report("behind-pusher")
    s9, ok = judge()
    check("behind-the-pusher: red tin settled in the lane BEHIND the resting plate — "
          "seated but NOT clamped (comp ~0, front gap huge): NOT success, "
          "score <= 0.50 (the spring clamp is the load-bearing tolerance)",
          bool(scene._seated_now(scene.red)[0]) and not bool(scene._clamped_now()[0])
          and comp() < 0.010 and not ok and s9 <= 0.50)

    # =========================== 10. side-lying clamp =======================================
    # Plate held back (follower-only write) and the red tin laid ON ITS SIDE in the
    # gap; released, the spring clamps the lying tin against the sill. Everything
    # about the clamp reads back — but the tin is not upright.
    torch.manual_seed(81)
    env.reset()
    step(30)
    write_plate(0.075)
    h = math.sqrt(0.5)
    place_local(scene.red, 0.0585, 0.0, c.tin_w / 2 + 0.003, quat=(h, 0.0, h, 0.0))
    step(200)
    report("side-lying")
    s10, ok = judge()
    check("side-lying clamp: red tin spring-clamped LYING ON ITS SIDE — the upright "
          "clause alone rejects: not seated, NOT success, score <= 0.60",
          comp() > 0.030 and not bool(scene._seated_now(scene.red)[0]) and not ok
          and s10 <= 0.60)

    # =========================== 11. latched credit survives regression =====================
    place_world(scene.red, c.slots[0][0], c.slots[0][1], c.tin_h / 2 + 0.002)
    step(80)
    report("regressed")
    s11, ok = judge()
    check("latched credit: teleporting the red tin back out of the lane leaves the "
          f"latched score unchanged ({s10:.3f} -> {s11:.3f}), still no success",
          abs(s11 - s10) < 1e-3 and not bool(scene._entered(scene.red)[0]) and not ok)

    # =========================== 12. wrong tin ==============================================
    # The GREEN decoy stocked perfectly (constructed): every geometric gate a correct
    # stock would pass — but it is not the red tin.
    torch.manual_seed(91)
    env.reset()
    step(30)
    write_plate(0.058)
    place_local(scene.green, 0.0475, 0.0, c.tin_h / 2 + 0.003)
    step(200)
    report("wrong-tin")
    s, ok = judge()
    check("wrong tin: GREEN decoy spring-clamped in the lane (in-lane readback), red "
          "untouched — NOT success, score <= 0.16 (color identification is "
          "load-bearing)",
          bool(scene._in_lane(scene.green)[0]) and comp() > 0.030 and not ok
          and s <= 0.16)

    # =========================== 13. approach only ==========================================
    torch.manual_seed(101)
    env.reset()
    step(30)
    place_world(scene.red, c.bay_pos[0] - c.tin_w / 2 - 0.004, 0.0, c.tin_h / 2 + 0.002)
    step(90)
    report("approach-only")
    s, ok = judge()
    check("approach only: red tin resting against the plinth front on the ground — "
          "score ~0 (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.facing_lane")
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

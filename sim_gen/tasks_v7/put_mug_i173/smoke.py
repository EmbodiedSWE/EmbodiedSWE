"""Smoke / rubric-REJECTION battery for MugHookScene (sim_gen task `put_mug_i173`)
— NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — teleport to a threaded hover off the peg tip,
gravity catch, force-slide home, hands-off swing decay — is the acceptance evidence
that the rubric ACCEPTS a correct outcome). Every teleport here is instrumentation
that CONSTRUCTS a wrong (or partial) outcome as a settled state and asserts the
rubric REJECTS it; no probe in this battery ever reaches success(), and a final
audit check asserts exactly that.

  1-2. settle/no-NaN   — reset layout settles finite: stand inside its jitter box,
                         mug upright on the ground, everything at rest, score ~0;
  3-4. randomization   — READBACK over 6 seeded resets: stand xy + yaw vary; mug
                         xy + free yaw and pad position vary, the mug/pad y bands
                         SWAP across seeds, and mug and pad never spawn close;
  5.  null policy      — 240 idle steps -> score ~0, no success;
  6.  seed strategy    — the seed's outcome (mug set down on the marked target):
                         stood upright ON the blue pad it counts NOTHING;
  7.  wrong place      — mug parked upright on the STAND'S BASE plate -> nothing;
  8.  wrong peg        — mug hung BY ITS HANDLE on the lower GREY decoy peg, swung
                         to a genuine settled hang (readback: threaded-on-grey,
                         suspended, full stillness streak) -> NOT success, score
                         <= 0.20 (only the lift latch);
  9.  wrong grip       — mug hooked on the RED peg by its RIM (peg through the cup
                         cavity, not the handle window), settled hanging -> NOT
                         threaded (interior radius 34 mm < window x_min 42 mm can
                         never present a window crossing), no success;
  10. fly-through      — the mug teleported DIRECTLY into the threaded hover (the
                         exact success topology, zero velocity) and judged WITHOUT
                         stepping -> NOT success (pose-jump guard + streak);
  11. fly-through B    — stepped a few frames mid-catch then yanked away: the
                         latched partial credit persists (<= 0.85) but success
                         never fired at any point;
  12. latch survives   — lift credit earned in mid-air does NOT evaporate when the
                         mug is returned to the ground (score stays ~0.15, still
                         no success);
  13. rejection audit  — success() was never True at ANY judged point;
  14. final no-NaN     — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.put_mug_i173.smoke --headless
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)",
                                             flush=True), os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.mug_hook")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply, quat_mul

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.75, -1.40, 1.10)) + o),
                                tuple(np.array((0.40, 0.00, 0.30)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video",
              flush=True)

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
        z = float(scene._mug_z()[0])
        v = float(scene.mug.data.root_lin_vel_w[0].norm())
        s, ok = judge()
        print(f"[smoke] {tag:16s} | mug_z={z:.3f} |v|={v:.3f} "
              f"thr_red={bool(scene._threaded_now('red')[0])} "
              f"thr_grey={bool(scene._threaded_now('grey')[0])} "
              f"still={int(scene._still[0])} lift={bool(scene._lifted[0])} "
              f"appr={bool(scene._approached[0])} "
              f"thr_ever={bool(scene._threaded_ever[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def write_state(obj, pos_w: torch.Tensor, quat: torch.Tensor,
                    settle_steps: int = 0) -> None:
        """One root-state write at a WORLD pose (already env-origin absolute)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w.unsqueeze(0)
        st[:, 3:7] = quat.unsqueeze(0)
        obj.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def hover_pose(which: str) -> tuple[torch.Tensor, torch.Tensor]:
        """The solve-P1 threaded hover for a peg: mug upright, local +y toward the
        peg root, the peg axis crossing the OPEN window 20 mm from the tip, 8 mm
        above the peg line (27 mm free fall to the catch). World frame."""
        root_w, tip_w = scene._peg_world(which)
        root_w, tip_w = root_w[0], tip_w[0]
        ax = tip_w - root_w
        ax = ax / ax.norm()
        u = ax.clone()
        u[2] = 0.0
        u = u / u.norm()
        psi = math.atan2(float(u[0]), -float(u[1]))
        qz = torch.tensor([math.cos(psi / 2), 0.0, 0.0, math.sin(psi / 2)],
                          device=device)
        off = torch.tensor([c.wc[0], 0.0, -0.008], device=device)
        pos = (tip_w - ax * 0.020) - quat_apply(qz.unsqueeze(0), off.unsqueeze(0))[0]
        return pos, qz

    def settle_until_still(max_chunks: int = 20, chunk: int = 60) -> None:
        for _ in range(max_chunks):
            step(chunk)
            if int(scene._still[0]) >= c.still_steps:
                break

    def mug_yaw() -> float:
        q = scene.mug.data.root_quat_w[0]
        return 2.0 * math.atan2(float(q[3]), float(q[0]))

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    fin0 = (torch.isfinite(scene.stand.data.root_state_w).all()
            and torch.isfinite(scene.mug.data.root_state_w).all()
            and torch.isfinite(scene.pad.data.root_state_w).all())
    sp = (scene.stand.data.root_pos_w - scene.env_origins)[0]
    mp = (scene.mug.data.root_pos_w - scene.env_origins)[0]
    still = (float(scene.mug.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    check("settle: states finite, stand inside its jitter box, mug upright on the "
          "ground at cup height (readback), at rest",
          bool(fin0)
          and abs(float(sp[0]) - c.stand_pos[0]) < c.stand_jitter + 0.005
          and abs(float(sp[1]) - c.stand_pos[1]) < c.stand_jitter + 0.005
          and abs(float(mp[2]) - c.body_h / 2) < 0.012 and still)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        sp = (scene.stand.data.root_pos_w - scene.env_origins)[0]
        sq = scene.stand.data.root_quat_w[0]
        syaw = 2.0 * math.atan2(float(sq[3]), float(sq[0]))
        mp = (scene.mug.data.root_pos_w - scene.env_origins)[0]
        pp = (scene.pad.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(sp[0]), float(sp[1]), syaw, float(mp[0]), float(mp[1]),
                      mug_yaw(), float(pp[0]), float(pp[1]),
                      float((mp[:2] - pp[:2]).norm())))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (stand_x, stand_y, stand_yaw, mug_x, "
          f"mug_y, mug_yaw, pad_x, pad_y, sep):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: stand xy + yaw vary across seeded resets (readback)",
          spread[0] > 0.008 and spread[1] > 0.008 and spread[2] > 0.05)
    check("randomization: mug xy + free yaw and pad position vary, the mug/pad y "
          "bands SWAP across seeds, and mug and pad never spawn within 15 cm",
          spread[3] > 0.02 and spread[5] > 0.3
          and float(arr[:, 4].min()) < -0.05 and float(arr[:, 4].max()) > 0.05
          and float(arr[:, 8].min()) >= 0.15)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 6. seed strategy: set down on the target pad ===============
    # The seed's whole plan — carry the mug and set it down inside a marked region —
    # executed here: mug stood upright ON the blue pad. It counts NOTHING.
    torch.manual_seed(41)
    env.reset()
    step(10)
    pp = scene.pad.data.root_pos_w[0].clone()
    pos = pp + torch.tensor([0.0, 0.0, c.pad_t + c.body_h / 2 + 0.003], device=device)
    write_state(scene.mug, pos, torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
                settle_steps=120)
    report("seed-pad")
    s, ok = judge()
    mz = float(scene._mug_z()[0])
    check("seed strategy: mug set down upright ON the blue display pad (the seed's "
          "put-on-target outcome) earns nothing — no success, score <= 0.02",
          mz < 0.10 and not ok and s <= 0.02)

    # =========================== 7. wrong place: parked on the stand base ===================
    torch.manual_seed(51)
    env.reset()
    step(10)
    sq = scene.stand.data.root_quat_w[0]
    rel = torch.tensor([0.06, 0.09, c.base_t + c.body_h / 2 + 0.003], device=device)
    pos = scene.stand.data.root_pos_w[0] + quat_apply(sq.unsqueeze(0),
                                                      rel.unsqueeze(0))[0]
    write_state(scene.mug, pos, torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
                settle_steps=120)
    report("stand-base")
    s, ok = judge()
    mz = float(scene._mug_z()[0])
    check("wrong place: mug parked upright on the STAND'S BASE plate — not "
          "suspended, no success, score <= 0.02",
          mz < c.suspend_z_min and not ok and s <= 0.02)

    # =========================== 8. wrong peg: a real settled hang on the GREY decoy ========
    # Constructed exactly like the solution but on the LOWER GREY peg, and allowed to
    # swing to a genuine settled hang (readback proves it). Still not success.
    torch.manual_seed(61)
    env.reset()
    step(10)
    pos, qz = hover_pose("grey")
    write_state(scene.mug, pos, qz, settle_steps=0)
    assert bool(scene._threaded_now("grey")[0]), "grey hover must thread the window"
    settle_until_still()
    report("grey-hang")
    s, ok = judge()
    hung_grey = (bool(scene._threaded_now("grey")[0])
                 and float(scene._mug_z()[0]) > c.suspend_z_min
                 and int(scene._still[0]) >= c.still_steps)
    check("wrong peg: mug hung by its handle on the GREY decoy peg, swung to a "
          "genuine settled hang (threaded-on-grey + suspended + full stillness "
          "streak, readback) — NOT success, score <= 0.20 (lift latch only)",
          hung_grey and not ok and s <= 0.20
          and not bool(scene._threaded_ever[0]))

    # =========================== 9. wrong grip: rim-hooked on the RED peg ===================
    # Mug pitched mouth-toward-post so the RED peg enters the CUP CAVITY (not the
    # handle window) and the mug hangs by its rim/inner wall. The interior radius
    # (34 mm) is strictly inside the window's x-span (42..76 mm), so a cavity hang
    # can never present a window crossing — geometrically not threaded.
    torch.manual_seed(71)
    env.reset()
    step(10)
    root_w, tip_w = scene._peg_world("red")
    root_w, tip_w = root_w[0], tip_w[0]
    ax = tip_w - root_w
    ax = ax / ax.norm()
    d = -ax.clone()
    d[2] = 0.0
    d = d / d.norm()  # horizontal, tip -> root
    yaw_d = math.atan2(float(d[1]), float(d[0]))
    q_yaw = torch.tensor([math.cos(yaw_d / 2), 0.0, 0.0, math.sin(yaw_d / 2)],
                         device=device)
    q_pitch = torch.tensor([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0],
                           device=device)  # ez -> ex, handle (local +x) -> down
    q_rim = quat_mul(q_yaw.unsqueeze(0), q_pitch.unsqueeze(0))[0]
    pos = tip_w - ax * 0.045 - torch.tensor([0.0, 0.0, 0.015], device=device)
    write_state(scene.mug, pos, q_rim, settle_steps=0)
    settle_until_still(max_chunks=10)
    report("rim-hang")
    s, ok = judge()
    check("wrong grip: mug hooked on the RED peg by its rim/cavity, hanging settled "
          "— NOT threaded (no window crossing), no success, score <= 0.36",
          float(scene._mug_z()[0]) > c.suspend_z_min
          and not bool(scene._threaded_now("red")[0])
          and not bool(scene._threaded_ever[0]) and not ok and s <= 0.36)

    # =========================== 10-11. fly-through / teleport guard ========================
    # The exact success topology teleported in cold: threaded, suspended, zero
    # velocity — judged WITHOUT stepping it must NOT be success (pose-jump guard);
    # then a few real frames and a yank-away: latched credit persists, success never.
    torch.manual_seed(81)
    env.reset()
    step(60)  # build an (irrelevant) on-ground stillness streak first
    pos, qz = hover_pose("red")
    write_state(scene.mug, pos, qz, settle_steps=0)
    thr0 = bool(scene._threaded_now("red")[0])
    susp0 = float(scene._mug_z()[0]) > c.suspend_z_min
    s0, ok0 = judge()
    report("fly-through")
    check("fly-through: mug teleported DIRECTLY into the threaded suspended hover "
          "(zero velocity) judged WITHOUT stepping is NOT success",
          thr0 and susp0 and not ok0)
    step(12)  # catch begins; latches update on real frames
    mid_ok = bool(scene.success()[0])
    ever_success[0] = ever_success[0] or mid_ok
    ground = torch.tensor([0.05, -0.45, c.body_h / 2 + 0.002], device=device) \
        + scene.env_origins[0]
    write_state(scene.mug, ground,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=device), settle_steps=90)
    report("yank-away")
    s, ok = judge()
    check("fly-through B: a few real frames mid-catch then yanked to the ground — "
          "success never fired, latched partial credit persists but caps <= 0.85",
          not mid_ok and not ok and 0.30 <= s <= 0.85)

    # =========================== 12. lift latch survives ====================================
    torch.manual_seed(91)
    env.reset()
    step(10)
    mp = scene.mug.data.root_pos_w[0].clone()
    high = mp + torch.tensor([0.0, 0.0, 0.30], device=device)
    write_state(scene.mug, high, torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
                settle_steps=12)  # latch `lifted` on real frames while airborne
    lifted_mid = bool(scene._lifted[0])
    step(228)  # fall back + settle on the ground
    report("lift-return")
    s, ok = judge()
    check("latch survives: lift credit earned in mid-air does not evaporate when "
          "the mug returns to the ground (score ~0.15, still no success)",
          lifted_mid and 0.13 <= s <= 0.17 and not ok
          and float(scene._mug_z()[0]) < 0.10)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    fin = (torch.isfinite(scene.stand.data.root_state_w).all()
           and torch.isfinite(scene.mug.data.root_state_w).all()
           and torch.isfinite(scene.pad.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.mug_hook")
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
    except BaseException:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)

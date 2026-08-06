"""Smoke / rubric-REJECTION battery for BayonetLockScene (sim_gen task
`plug_charger_i4`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — seat through the slot + twist to lock, both under
contact dynamics — is the acceptance evidence that the rubric ACCEPTS a correct
outcome). Every teleport here is instrumentation that CONSTRUCTS a wrong (or partial)
outcome as a settled state and asserts the rubric REJECTS it; no probe in this battery
ever reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite and still, score ~0 at rest;
  3-4. randomization      — READBACK over 6 seeded resets: plug spawn xy+yaw, socket
                            xy+yaw all move;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  SEED strategy       — the seed's plan end state: plug pushed straight in to full
                            depth, never twisted (lugs still on the slot axis) ->
                            NOT success, score capped at ~0.5;
  7.  misaligned drop     — plug lowered onto the mouth with lugs ACROSS the slot ->
                            rests ON the plates, no seating credit, no success;
  8.  near-miss rotation  — seated but twisted only to lock_deg - 15 deg (already past
                            the geometric extraction limit, still short of the lock
                            zone) -> NOT success, score < 0.9;
  9.  monotonicity        — a deeper twist probe latched strictly more rotation credit
                            than a shallower one;
  10. wrong object        — the red lug-less distractor seated in the well (it fits):
                            identity matters, score ~0, no success;
  11. latched credit      — pulling the plug back OUT after the near-miss twist leaves
                            the latched score unchanged (credit does not evaporate);
  12. rejection audit     — success() was never True at ANY judged point;
  13. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.plug_charger_i4.smoke --headless
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


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.plug_bayonet_lock")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((0.95, -0.95, 0.75)) + o),
                                tuple(np.array((0.00, 0.02, 0.05)) + o),
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

    def plug_local():
        return scene._plug_local()[0]

    def socket_pose() -> tuple[torch.Tensor, float]:
        sp = (scene.socket.data.root_pos_w - scene.env_origins)[0]
        q = scene.socket.data.root_quat_w[0]
        return sp, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def report(tag: str) -> None:
        loc = plug_local()
        s, ok = judge()
        print(f"[smoke] {tag:16s} | plug_local=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
              f"{float(loc[2]):.3f}) slot_dist={float(scene.slot_distance_deg()[0]):5.1f}deg "
              f"depth_latch={float(scene.depth_latch[0]):.3f} "
              f"rot_latch={float(scene.rot_latch[0]):.3f} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_body(body, x: float, y: float, z: float, yaw: float = 0.0,
                   settle_steps: int = 30) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = math.cos(yaw / 2)
        st[:, 6] = math.sin(yaw / 2)
        st[:, 0:3] += env.iscene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def place_plug_local(x_l: float, y_l: float, z: float, yaw_rel_deg: float,
                         settle_steps: int = 60) -> None:
        """Place the plug at socket-local (x_l, y_l, z) with the lug axis rotated
        `yaw_rel_deg` from the entry slot."""
        sp, syaw = socket_pose()
        cy, sy = math.cos(syaw), math.sin(syaw)
        wx = float(sp[0]) + cy * x_l - sy * y_l
        wy = float(sp[1]) + sy * x_l + cy * y_l
        place_body(scene.plug, wx, wy, z, yaw=syaw + math.radians(yaw_rel_deg),
                   settle_steps=settle_steps)

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    st0 = scene.plug.data.root_state_w
    base_z = float((scene.plug.data.root_pos_w - scene.env_origins)[0, 2]) - c.plug_h / 2
    check("settle: plug state finite, at rest upright on the table",
          bool(torch.isfinite(st0).all()) and bool(scene.settled()[0])
          and abs(base_z) < 0.01 and float(scene._plug_up_z()[0]) > 0.98)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        p = (scene.plug.data.root_pos_w - scene.env_origins)[0]
        pq = scene.plug.data.root_quat_w[0]
        pyaw = 2.0 * math.atan2(float(pq[3]), float(pq[0]))
        sp, syaw = socket_pose()
        reads.append((float(p[0]), float(p[1]), pyaw, float(sp[0]), float(sp[1]), syaw))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (plug_x, plug_y, plug_yaw, socket_x, socket_y, "
          f"socket_yaw):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: plug spawn xy + yaw vary across seeded resets (readback)",
          spread[0] > 0.005 and spread[1] > 0.005 and spread[2] > 0.2)
    check("randomization: socket xy + yaw vary across seeded resets (readback)",
          spread[3] > 0.005 and spread[4] > 0.005 and spread[5] > 0.03)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. SEED strategy: straight insertion, no twist =============
    # The seed's whole plan is translational: align and push the charger straight in.
    # Constructed here: plug seated at full depth, lugs still on the slot axis. The
    # bayonet is not engaged — the plug lifts straight back out — and the rubric must
    # refuse it: no success, score capped at the seating credit (~0.5).
    torch.manual_seed(41)
    env.reset()
    step(10)
    place_plug_local(0.0, 0.0, c.seated_z + 0.0005, 0.0, settle_steps=90)
    report("seed-strategy")
    s, ok = judge()
    d = float(scene.slot_distance_deg()[0])
    check("seed strategy (full-depth insertion, never twisted): seated but NOT success, "
          "slot_dist ~0, score <= 0.6",
          bool(scene.seated()[0]) and not ok and d < 8.0 and 0.30 <= s <= 0.60)

    # =========================== 7. misaligned drop rests ON the plates =====================
    # Lowered with the lugs ACROSS the slot, the plug cannot enter: the lugs land on the
    # orange plates and it hangs above the well — no seating credit, no success.
    torch.manual_seed(51)
    env.reset()
    step(10)
    place_plug_local(0.0, 0.0, c.entry_z + 0.004, 90.0, settle_steps=120)
    report("misaligned-drop")
    s, ok = judge()
    loc = plug_local()
    check("misaligned drop: lugs across the slot rest ON the plates — plug centre stays "
          "high, no success, score <= 0.35",
          float(loc[2]) > c.seated_z + 0.015 and not ok
          and float(scene.rot_latch[0]) <= 0.01 and s <= 0.35)

    # =========================== 8. near-miss rotation (short of the lock zone) =============
    near_deg = c.lock_deg - 15.0
    torch.manual_seed(61)
    env.reset()
    step(10)
    place_plug_local(0.0, 0.0, c.seated_z + 0.0005, near_deg, settle_steps=90)
    report("near-miss")
    s_near, ok = judge()
    rot_near = float(scene.rot_latch[0])
    check(f"near-miss: seated + twisted only {near_deg:.0f} deg (< lock_deg "
          f"{c.lock_deg:.0f}) — NOT success, score < 0.9",
          bool(scene.seated()[0]) and not bool(scene.locked()[0]) and not ok
          and 0.4 < s_near < 0.9)

    # =========================== 9. monotonicity ============================================
    torch.manual_seed(71)
    env.reset()
    step(10)
    place_plug_local(0.0, 0.0, c.seated_z + 0.0005, 20.0, settle_steps=90)
    report("shallow-twist")
    judge()
    rot_shallow = float(scene.rot_latch[0])
    check("monotonicity: deeper twist latched strictly more rotation credit "
          f"({rot_shallow:.3f} < {rot_near:.3f})", rot_shallow + 0.05 < rot_near)

    # =========================== 10. wrong object ===========================================
    torch.manual_seed(81)
    env.reset()
    step(10)
    sp, syaw = socket_pose()
    place_body(scene.distractor, float(sp[0]), float(sp[1]),
               c.floor_t + c.distractor_h / 2 + 0.001, settle_steps=60)
    report("wrong-object")
    s, ok = judge()
    dis_rel = (scene.distractor.data.root_pos_w - scene.socket.data.root_pos_w)[0, :2].norm()
    check("wrong object: the red lug-less distractor seated in the well scores ~0, "
          "no success",
          float(dis_rel) < 0.03 and s <= 0.05 and not ok)

    # =========================== 11. latched credit survives pull-out =======================
    torch.manual_seed(91)
    env.reset()
    step(10)
    place_plug_local(0.0, 0.0, c.seated_z + 0.0005, near_deg, settle_steps=90)
    s_in, _ = judge()
    place_plug_local(0.0, -0.22, c.plug_h / 2 + 0.002, 0.0, settle_steps=60)
    report("pulled-out")
    s_out, ok = judge()
    check("latched credit: pulling the plug back out leaves the latched score unchanged",
          abs(s_out - s_in) < 1e-3 and not ok)

    # =========================== 12-13. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.plug.data.root_state_w).all()
           and torch.isfinite(scene.distractor.data.root_state_w).all()
           and torch.isfinite(scene.socket.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.plug_bayonet_lock")
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

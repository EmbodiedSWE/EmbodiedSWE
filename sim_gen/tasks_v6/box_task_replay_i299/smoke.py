"""Smoke / rubric-REJECTION battery for FlapPostboxScene (sim_gen task
`box_task_replay_i299`) — NullRobot, teleported/force-constructed probe states, RECORDED.

This is NOT a solution (solve.py — the force-pushed flap passages — is the acceptance
evidence that the rubric ACCEPTS a correct outcome). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled state and
asserts the rubric REJECTS it; no probe in this battery ever reaches success(), and a
final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: parcels standing on the porch,
                            flap hanging shut, score ~0 at rest;
  3-4. randomization      — READBACK over 8 seeded resets: parcel spawn slots swap and
                            jitter; the decoy crate's side flips and its jitter is real;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed strategy       — the seed's plan (place both items into the OPEN container):
                            both parcels settled inside the open gray crate -> score ~0,
                            no success;
  7.  near-miss           — can settled on the porch pressed against the CLOSED flap
                            (delivery attempted, never pushed through) -> not inside,
                            no success, score small;
  8.  half-through        — block force-pushed until it rests PROTRUDING through the
                            doorway, propping the flap open (the incomplete delivery)
                            -> flap not shut, parcel not inside, no success;
  9.  flap-ajar clause    — both parcels teleported inside WHILE the flap is pose-held
                            ajar (14 deg): containment alone is not success — the
                            shut-flap clause is load-bearing;
  10. latched credit      — parcels removed to the porch, flap released: the latched
                            delivered credit does not evaporate (score unchanged),
                            still no success;
  11. roof                — can settled on the postbox ROOF (the "put it on the box"
                            cheat) -> not inside, no success;
  12. one delivered       — can alone inside the shut box, block still on the porch ->
                            no success (BOTH parcels are required);
  13. rejection audit     — success() was never True at ANY judged point;
  14. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.box_task_replay_i299.smoke --headless
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
    env = ENVS.get("simgen.flap_postbox")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)
    bx, by = c.box_pos

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.50, -1.15, 0.90)) + o),
                                tuple(np.array((0.45, 0.00, 0.10)) + o),
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

    def report(tag: str) -> None:
        p_can = (scene.can.data.root_pos_w - scene.env_origins)[0]
        p_blk = (scene.block.data.root_pos_w - scene.env_origins)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | can=({float(p_can[0]):+.3f},{float(p_can[1]):+.3f},"
              f"{float(p_can[2]):.3f}) block=({float(p_blk[0]):+.3f},"
              f"{float(p_blk[1]):+.3f},{float(p_blk[2]):.3f}) "
              f"flap={math.degrees(float(scene.flap_angle()[0])):+.1f}deg "
              f"in_can={bool(scene._inside(scene.can)[0])} "
              f"in_blk={bool(scene._inside(scene.block)[0])} "
              f"lat_in={scene._in[0].tolist()} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, x: float, y: float, z: float, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += env.iscene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def hold_flap(theta_deg: float) -> None:
        """One kinematic-hold write of the flap on its hinge arc (negative = inward)."""
        h = math.radians(theta_deg) / 2
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = bx, by, c.hinge_z
        st[:, 3], st[:, 5] = math.cos(h), math.sin(h)
        st[:, 0:3] += env.iscene.env_origins
        scene.flap.write_root_state_to_sim(st, all_ids)

    def obj_xy(body) -> tuple[float, float]:
        p = (body.data.root_pos_w - scene.env_origins)[0]
        return float(p[0]), float(p[1])

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    fin0 = (torch.isfinite(scene.can.data.root_state_w).all()
            and torch.isfinite(scene.block.data.root_state_w).all()
            and torch.isfinite(scene.flap.data.root_state_w).all())
    z_can = float((scene.can.data.root_pos_w - scene.env_origins)[0, 2])
    z_blk = float((scene.block.data.root_pos_w - scene.env_origins)[0, 2])
    still = bool(scene._parcels_still()[0])
    check("settle: states finite, both parcels standing on the porch, flap hanging "
          "shut, everything still",
          bool(fin0) and z_can > c.sill_h - 0.01 and z_blk > c.sill_h - 0.01
          and bool(scene.flap_shut()[0]) and still)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        cx, cy = obj_xy(scene.can)
        kx, ky = obj_xy(scene.crate)
        reads.append((cx, cy, kx, ky, 1.0 if cy > 0 else 0.0, 1.0 if ky > 0 else 0.0))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (can_x, can_y, crate_x, crate_y, "
          f"can_in_slot_a, crate_left):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: parcel spawn slots swap and xy jitter is real across seeded "
          "resets (readback)",
          0.0 < arr[:, 4].mean() < 1.0 and spread[0] > 0.015 and spread[1] > 0.05)
    jit = 0.0
    for flag in (0.0, 1.0):
        grp = arr[arr[:, 5] == flag]
        if len(grp) >= 2:
            jit = max(jit, float((grp[:, 2:4].max(axis=0) - grp[:, 2:4].min(axis=0)).max()))
    check("randomization: decoy crate side flips across seeds AND per-side jitter is "
          "real (readback)", 0.0 < arr[:, 5].mean() < 1.0 and jit > 0.004)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy: items into the OPEN container ============
    # The seed's plan — pick each item up and place it down into an open box — executed
    # verbatim into the only open container in the scene, the gray decoy crate.
    torch.manual_seed(41)
    env.reset()
    step(10)
    kx, ky = obj_xy(scene.crate)
    place(scene.can, kx - 0.045, ky, c.crate_floor_t + c.can_h / 2 + 0.003)
    place(scene.block, kx + 0.045, ky, c.crate_floor_t + c.block_h / 2 + 0.003)
    step(60)
    report("seed-strategy")
    s, ok = judge()
    d_can = math.hypot(*(a - b for a, b in zip(obj_xy(scene.can), (kx, ky))))
    check("seed strategy: both parcels settled inside the OPEN gray crate — score ~0 "
          "(<= 0.05), no success (open-container placement is the decoy)",
          d_can < c.crate_out_w / 2 and not ok and s <= 0.05)

    # =========================== 7. near-miss: against the closed flap ======================
    torch.manual_seed(51)
    env.reset()
    step(10)
    place(scene.can, bx - c.can_r - c.flap_t - 0.004, by,
          c.sill_h + c.can_h / 2 + 0.002)
    step(60)
    report("near-miss")
    s, ok = judge()
    check("near-miss: can settled on the porch pressed against the CLOSED flap — not "
          "inside, flap still shut, NOT success, score <= 0.15",
          not bool(scene._inside(scene.can)[0]) and bool(scene.flap_shut()[0])
          and not ok and s <= 0.15)

    # =========================== 8. half-through: propping the flap =========================
    # Force-push the block toward the doorway and CUT the push early, so it comes to
    # rest protruding through the doorway with the flap resting on it — the incomplete
    # delivery. (Constructed by force, not teleport: a teleport into the flap's hanging
    # volume would interpenetrate.)
    torch.manual_seed(61)
    env.reset()
    step(10)
    place(scene.block, bx - 0.16, by, c.sill_h + c.block_h / 2 + 0.003,
          quat=(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0))  # lying, long axis x
    step(20)
    for _ in range(600):
        p = (scene.block.data.root_pos_w - scene.env_origins)[0]
        if float(p[0]) > bx - 0.018:
            break
        v = scene.block.data.root_lin_vel_w[0]
        fx = max(-0.6, min(1.2, 5.0 * (0.12 - float(v[0]))))
        fy = max(-0.5, min(0.5, 4.0 * (by - float(p[1])) - 1.0 * float(v[1])))
        f_w = torch.tensor([fx, fy, 0.0], device=device).view(1, 1, 3).expand(n, 1, 3)
        scene.block.set_external_force_and_torque(f_w.contiguous(), zero_wrench,
                                                  env_ids=all_ids, is_global=True)
        env.step(no_action)
    scene.block.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    step(80)
    report("half-through")
    s, ok = judge()
    p_blk = (scene.block.data.root_pos_w - scene.env_origins)[0]
    propped = (not bool(scene.flap_shut()[0])) and not bool(scene._inside(scene.block)[0])
    if not propped:
        # The push carried it fully in (or it slid back): either way this state must
        # still not be judged success (one parcel at most, and if in, flap may be shut
        # — but the can is still outside). Report honestly and assert no-success.
        print(f"[smoke] half-through construct ended at x={float(p_blk[0]):+.3f} "
              f"(not propped) — asserting no-success anyway", flush=True)
    check("half-through: block resting in the doorway props the flap open (or at "
          "minimum the incomplete delivery is not success) — NOT success",
          not ok and (propped or float(p_blk[0]) > bx - 0.05))

    # =========================== 9. flap-ajar clause is load-bearing ========================
    # Both parcels ARE inside — but the flap is pose-held ajar the whole time, so the
    # shut-flap clause must reject. (The hold starts BEFORE the parcels appear inside
    # and lasts until they are removed again, so success is never crossed.)
    torch.manual_seed(71)
    env.reset()
    step(10)
    hold_flap(-14.0)  # negative = inward swing (bottom edge moves into the box)
    place(scene.can, bx + 0.10, by - 0.05, c.can_h / 2 + 0.003)
    place(scene.block, bx + 0.16, by + 0.05, c.block_h / 2 + 0.003)
    for _ in range(80):
        hold_flap(-14.0)
        env.step(no_action)
    report("flap-ajar")
    s9, ok = judge()
    both_in = bool(scene._inside(scene.can)[0]) and bool(scene._inside(scene.block)[0])
    check("flap-ajar: BOTH parcels inside but the flap held ajar (14 deg > 6 deg tol) "
          "— containment alone is not success (shut-flap clause is load-bearing)",
          both_in and not bool(scene.flap_shut()[0]) and not ok)

    # =========================== 10. latched credit survives regression =====================
    # Remove the parcels to the porch BEFORE releasing the flap (rejection-only audit),
    # let the flap fall shut: the latched delivered credit must not evaporate.
    place(scene.can, bx - 0.22, 0.10, c.sill_h + c.can_h / 2 + 0.003)
    place(scene.block, bx - 0.22, -0.10, c.sill_h + c.block_h / 2 + 0.003)
    hold_flap(-14.0)
    step(1)
    step(60)  # flap released: falls shut over an empty doorway
    report("regressed")
    s10, ok = judge()
    check("latched credit: parcels removed and the flap fallen shut — the latched "
          f"delivered score is unchanged ({s9:.3f} -> {s10:.3f}), still no success",
          abs(s10 - s9) < 1e-3 and s10 >= 2 * c.w_in - 0.01 and not ok)

    # =========================== 11. roof cheat =============================================
    torch.manual_seed(81)
    env.reset()
    step(10)
    place(scene.can, bx + 0.16, by, c.in_h + c.wall_t + c.can_h / 2 + 0.003)
    step(60)
    report("roof")
    s, ok = judge()
    p_can = (scene.can.data.root_pos_w - scene.env_origins)[0]
    check("roof cheat: can settled ON the postbox roof — xy over the interior but the "
          "z clause rejects it: not inside, NOT success",
          float(p_can[2]) > c.in_h - 0.02 and not bool(scene._inside(scene.can)[0])
          and not ok)

    # =========================== 12. one delivered is not enough ============================
    torch.manual_seed(91)
    env.reset()
    step(10)
    place(scene.can, bx + 0.12, by, c.can_h / 2 + 0.003)
    step(60)
    report("one-delivered")
    s, ok = judge()
    check("one delivered: can inside the shut box but the block still on the porch — "
          "NOT success (both parcels required), score < 0.85",
          bool(scene._inside(scene.can)[0]) and bool(scene.flap_shut()[0])
          and not bool(scene._inside(scene.block)[0]) and not ok and s < 0.85)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.can.data.root_state_w).all()
           and torch.isfinite(scene.block.data.root_state_w).all()
           and torch.isfinite(scene.flap.data.root_state_w).all()
           and torch.isfinite(scene.crate.data.root_state_w).all()
           and torch.isfinite(scene.postbox.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.flap_postbox")
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

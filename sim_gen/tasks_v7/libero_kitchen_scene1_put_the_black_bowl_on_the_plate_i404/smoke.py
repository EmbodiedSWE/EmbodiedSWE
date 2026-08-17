"""Smoke / rubric-rejection battery for CaliberVaultScene (sim_gen task
`libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i404`) — NullRobot, teleported
probe states (instrumentation, NOT a solution), RECORDED.

solve.py already proves the rubric ACCEPTS the correct outcome (big marble delivered by
the rail ride, small marble by the lane-and-port); this battery proves it REJECTS the
wrong ones:

  1. settle/no-NaN      — reset settles finite: both marbles on the open floor, score
                          0, no latches, no success;
  2. mass readback      — the custom kinematic rig and both marbles run their authored
                          MassAPI masses (cfg mass paths are silently ignored);
  3. randomization      — READBACK across seeds: rig xy and YAW move, and the staging
                          slot assignment (big in A vs B) actually swaps;
  4. null-policy-fails  — 240 idle steps -> score ~0, no success;
  5. rails refuse small — the small marble released over the rails at the big marble's
                          own drop point falls STRAIGHT THROUGH the gap (never-pass
                          interference) and ends low and OUTSIDE the vault;
  6. port refuses big   — the big marble force-pressed down the lane corridor cannot
                          pass the port: it visibly ADVANCES (non-vacuous probe) and is
                          stopped outside the wall plane, vault latch never fires;
  7. seed-style negative— the seed's own strategy (carry the payloads to the goal and
                          set them down from above) parks both marbles ON the canopy
                          roof: nothing latches, no success;
  8. latch persistence  — an honest rail delivery latches 0.50; teleporting the big
                          marble back OUT of the vault leaves the latched score
                          unchanged (credit does not evaporate under regression);
  9. near-miss (lane)   — big delivered + small resting IN the lane short of the port
                          -> 0.65, no success (doorway is not delivery);
 10. near-miss (alone)  — only the small marble inside the vault -> partial credit
                          only, no success;
 11. positive boundary  — both marbles settled inside the vault -> success() True and
                          score exactly 1.0 (the rubric judges the physical state).

Records video frames throughout and saves frames.npz in the CURRENT WORKING DIRECTORY.
Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i404.smoke --headless
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

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i404 import (  # noqa: F401,E501
        scene as scene_mod,
    )
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
_wd = threading.Timer(1200.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.caliber_vault")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.55, -0.85, 0.62)) + o),
                                tuple(np.array((0.15, 0.05, 0.08)) + o),
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

    def settle_until(pred, max_steps: int = 600, poll: int = 10) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def sc() -> float:
        return float(scene.score()[0])

    def loc(body) -> torch.Tensor:
        return scene.local(body)[0]

    def report(tag: str) -> None:
        lb, ls = loc(scene.big), loc(scene.small)
        print(f"[smoke] {tag:16s} big_loc=({lb[0]:+.3f},{lb[1]:+.3f},{lb[2]:+.3f}) "
              f"small_loc=({ls[0]:+.3f},{ls[1]:+.3f},{ls[2]:+.3f}) "
              f"big_vault={bool(scene.in_vault(scene.big)[0])} "
              f"small_vault={bool(scene.in_vault(scene.small)[0])} "
              f"settled={bool(scene.settled()[0])} score={sc():.3f} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def rig_to_world(p_local) -> torch.Tensor:
        p = torch.tensor([float(v) for v in p_local], device=device).unsqueeze(0)
        return (scene.rig.data.root_pos_w + quat_apply(scene.rig.data.root_quat_w, p))[0]

    def tp(body, p_local) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = rig_to_world(p_local)
        st[:, 3] = 1.0
        body.write_root_state_to_sim(st, all_ids)

    def push_body(body, dir_local, f_mag: float, steps: int,
                  stop=None) -> None:
        """Per-step contact-scale force push in a rig-frame direction (body-frame
        re-expression every step: the rolling body's frame spins)."""
        d = torch.tensor([dir_local], device=device)
        for _ in range(steps):
            if stop is not None and stop():
                break
            dir_w = quat_apply(scene.rig.data.root_quat_w, d)[0]
            f_body = quat_apply_inverse(body.data.root_quat_w,
                                        (f_mag * dir_w).unsqueeze(0))
            zt = torch.zeros(n, 1, 3, device=device)
            body.set_external_force_and_torque(f_body.unsqueeze(1), zt, env_ids=all_ids)
            env.step(no_action)
        z = torch.zeros(n, 1, 3, device=device)
        body.set_external_force_and_torque(z, z, env_ids=all_ids)
        step(2)

    def honest_rail_delivery() -> bool:
        """Release the big marble at its drop point and let the rails deliver it."""
        tp(scene.big, (c.drop_x, 0.0, c.rail_z + c.ride_h0 + c.big_r + 0.012))
        step(30)
        return settle_until(
            lambda: bool(scene.in_vault(scene.big)[0]) and bool(scene.settled()[0]),
            max_steps=1500)

    lane_cx = (c.port_x0 + c.port_x1) / 2

    # =========================== 1. settle / no-NaN ==============================================
    torch.manual_seed(11)
    env.reset()
    step(90)
    report("reset")
    st0 = torch.cat([scene.rig.data.root_state_w, scene.big.data.root_state_w,
                     scene.small.data.root_state_w], dim=-1)
    no_latch = not (bool(scene._ride_ever[0]) or bool(scene._canopy_ever[0])
                    or bool(scene._big_vault_ever[0]) or bool(scene._lane_ever[0])
                    or bool(scene._small_vault_ever[0]))
    check("settle: states finite, both marbles resting on the open floor, settled, "
          "score 0, no latches, no success",
          bool(torch.isfinite(st0).all())
          and abs(float(loc(scene.big)[2]) - c.big_r) < 0.02
          and abs(float(loc(scene.small)[2]) - c.small_r) < 0.02
          and bool(scene.settled()[0]) and sc() <= 0.005 and no_latch
          and not bool(scene.success()[0]))

    # =========================== 2. mass readback ================================================
    reads = {name: float(b.root_physx_view.get_masses().reshape(-1)[0])
             for name, b in (("rig", scene.rig), ("big", scene.big),
                             ("small", scene.small))}
    print(f"[smoke] mass readback={ {k: round(v, 4) for k, v in reads.items()} }", flush=True)
    check("mass readback: kinematic rig and both marbles run their authored masses",
          abs(reads["rig"] - c.rig_mass) < 1e-2
          and abs(reads["big"] - c.big_mass) < 1e-3
          and abs(reads["small"] - c.small_mass) < 1e-3)

    # =========================== 3. randomization is real ========================================
    rows = []
    for s in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(s)
        env.reset()
        step(4)
        rp = scene.rig.data.root_pos_w[0] - scene.env_origins[0]
        ex = quat_apply(scene.rig.data.root_quat_w[0:1],
                        torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
        yaw = math.atan2(float(ex[1]), float(ex[0]))
        # slot assignment by READBACK: which staging slot is the big marble near?
        lb = loc(scene.big)
        da = math.hypot(float(lb[0]) - c.slot_a[0], float(lb[1]) - c.slot_a[1])
        db = math.hypot(float(lb[0]) - c.slot_b[0], float(lb[1]) - c.slot_b[1])
        rows.append((float(rp[0]), float(rp[1]), yaw, 1.0 if da < db else 0.0))
    arr = np.array(rows)
    print(f"[smoke] randomization readback (rig_x, rig_y, rig_yaw, big_in_A):\n"
          f"{np.round(arr, 3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: rig xy moves, rig yaw sweeps, and the staging assignment "
          "(big in slot A vs B) swaps across seeds",
          spread[0] > 0.015 and spread[1] > 0.015 and spread[2] > 0.8
          and len(set(arr[:, 3])) == 2)

    # =========================== 4. null policy fails ============================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    check("null policy: 240 idle steps -> score ~0, no success",
          sc() <= 0.02 and not bool(scene.success()[0]))

    # =========================== 5. rails refuse the small marble ================================
    torch.manual_seed(41)
    env.reset()
    step(60)
    tp(scene.small, (c.drop_x, 0.0, c.rail_z + c.ride_h0 + c.big_r + 0.012))
    step(30)
    settle_until(lambda: bool(scene.settled()[0]), max_steps=800)
    step(60)
    ls = loc(scene.small)
    report("small-on-rails")
    check("rails refuse the small marble: released at the big marble's own drop point "
          "it falls straight through the gap, ends low and OUTSIDE the vault",
          float(ls[2]) < 0.06 and not bool(scene.in_vault(scene.small)[0])
          and not bool(scene._small_vault_ever[0]) and not bool(scene.success()[0]))

    # =========================== 6. port refuses the big marble ==================================
    torch.manual_seed(51)
    env.reset()
    step(60)
    tp(scene.big, (lane_cx, 0.220, 0.110))
    step(40)  # lands straddling the lane guide walls (it does not fit inside the lane)
    y_start = float(loc(scene.big)[1])
    push_body(scene.big, (0.0, -1.0, 0.0), 0.40, 480)
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    lb = loc(scene.big)
    report("big-at-port")
    check("port refuses the big marble: force-pressed down the lane corridor it "
          "ADVANCES (non-vacuous) but is stopped outside the wall plane, no vault "
          "latch, no success",
          y_start - float(lb[1]) > 0.02 and float(lb[1]) > c.side_half_w_in + 0.02
          and not bool(scene._big_vault_ever[0]) and not bool(scene.success()[0]))

    # =========================== 7. seed-style negative: set down at the goal ====================
    torch.manual_seed(61)
    env.reset()
    step(60)
    tp(scene.big, (0.340, 0.030, 0.300))
    step(10)
    tp(scene.small, (0.410, -0.030, 0.300))
    step(40)
    settle_until(lambda: bool(scene.settled()[0]), max_steps=400)
    step(60)
    lb, ls = loc(scene.big), loc(scene.small)
    report("roof-perch")
    check("seed-style negative: carrying both marbles to the goal and setting them "
          "down from above parks them ON the canopy roof — nothing latches, no "
          "success, score 0",
          float(lb[2]) > c.canopy_z and float(ls[2]) > c.canopy_z
          and not bool(scene.in_vault(scene.big)[0])
          and not bool(scene.in_vault(scene.small)[0])
          and sc() <= 0.005 and not bool(scene.success()[0]))

    # =========================== 8. latch persistence (no evaporation) ===========================
    torch.manual_seed(71)
    env.reset()
    step(60)
    ok_deliver = honest_rail_delivery()
    s_mid = sc()
    report("rail-delivered")
    tp(scene.big, (-0.20, -0.30, c.big_r + 0.010))  # transport back OUT to the open floor
    step(40)
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    report("regressed")
    check("latch persistence: an honest rail delivery latches 0.50, and pulling the "
          "big marble back out of the vault leaves the latched score unchanged",
          ok_deliver and abs(s_mid - 0.50) < 1e-4 and abs(sc() - s_mid) < 1e-6
          and not bool(scene.in_vault(scene.big)[0]) and not bool(scene.success()[0]))

    # =========================== 9. near-miss: small stalled in the lane =========================
    torch.manual_seed(81)
    env.reset()
    step(60)
    ok_deliver = honest_rail_delivery()
    tp(scene.small, (lane_cx, 0.180, c.lane_floor_top + c.small_r + 0.012))
    step(40)
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    step(60)
    report("lane-stall")
    check("near-miss (doorway is not delivery): big delivered + small resting in the "
          "lane short of the port -> 0.65, no success",
          ok_deliver and bool(scene._lane_ever[0])
          and not bool(scene.in_vault(scene.small)[0])
          and abs(sc() - 0.65) < 1e-4 and not bool(scene.success()[0]))

    # =========================== 10-11. lone small, then the positive boundary ===================
    torch.manual_seed(91)
    env.reset()
    step(60)
    tp(scene.small, (0.410, -0.030, 0.080))  # instrumentation: construct the end state
    step(40)
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    step(60)
    report("small-alone")
    check("near-miss (half a delivery): only the small marble inside the vault -> "
          "partial credit only, below the 0.90 cap, no success",
          bool(scene.in_vault(scene.small)[0]) and sc() <= 0.90 + 1e-6
          and sc() < 0.50 and not bool(scene.success()[0]))
    tp(scene.big, (0.340, 0.030, 0.090))
    step(40)
    settle_until(lambda: bool(scene.success()[0]), max_steps=400)
    step(60)
    report("both-in")
    check("positive boundary: both marbles settled inside the vault -> success() True "
          "and score exactly 1.0 (the rubric judges the physical state)",
          bool(scene.in_vault(scene.big)[0]) and bool(scene.in_vault(scene.small)[0])
          and bool(scene.success()[0]) and sc() >= 1.0 - 1e-6)

    # =========================== save + verdict ==================================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.caliber_vault")
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
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL exception", flush=True)
        os._exit(2)

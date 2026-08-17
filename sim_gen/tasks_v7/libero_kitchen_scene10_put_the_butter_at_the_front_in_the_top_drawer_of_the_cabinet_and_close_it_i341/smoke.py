"""Smoke / rubric-REJECTION battery for ShuttleVaultScene (sim_gen task
`libero_kitchen_scene10_..._i341`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — butter laid on the porch by contact drop, the
shuttle velocity-servo draw that carries it under the hood, arrests it at the stop
bar and withdraws the floor, then the return stroke to the closed stop — is the
acceptance evidence that the rubric ACCEPTS a correct outcome). Every teleport here
is instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled state
and asserts the rubric REJECTS it; no probe in this battery ever reaches success(),
and a final audit check asserts exactly that.

  1-2.  settle/no-NaN     — reset layout settles finite: shuttle parked ajar in its
                            randomized range, both blocks outside on the floor,
                            everything still, score ~0 at rest;
  3-5.  randomization     — READBACK over 8 seeded resets: butter/brick slot
                            assignment varies; per-slot xy jitter and spawn yaw vary;
                            the shuttle's initial opening op0 varies;
  6.   null policy        — 240 idle steps -> score ~0, no success;
  7.   seed strategy      — the end state the seed's plan (lower the item into the
                            receptacle from above, then push it shut) produces here:
                            butter dropped from the sky over the hatch lands ON THE
                            HOOD ROOF, shuttle pushed closed -> rejected;
  8.   porch camp         — butter laid on the porch and left there: boarded +
                            approach credit only, score <= 0.27, no success;
  9.   under-hood camp    — butter parked on the plate under the hood (at the arrest
                            position, hatch still covered): + conveyed credit, score
                            <= 0.36, no success (the cavity is the goal);
  10.  vault left open    — butter settled INSIDE the cavity but the shuttle at full
                            open -> the closure clause rejects;
  11.  brick smuggled     — butter in the cavity AND shuttle closed, but the RED
                            brick inside too -> the brick-out clause rejects at the
                            0.85 cap;
  12.  shuttle ajar       — butter in the cavity but the shuttle parked 25 mm short
                            of its stop (band is 10 mm) -> the closed-band clause
                            rejects;
  13.  latched credit     — from state 12 the butter is yanked back to the floor:
                            the latched score holds (credit does not evaporate),
                            still no success;
  14.  sealed plate       — order/route forced by geometry: with the shuttle CLOSED
                            the butter is parked over the covered hatch and PRESSED
                            DOWN at 2x its weight for 300 steps — it stays at plate
                            height the whole press (min-z tracked DURING the press,
                            not after), never enters the cavity, and the plate holds
                            its stop: a closed shuttle admits nothing, so deposit
                            strictly requires the open stroke and only the return
                            stroke seals -> no success;
  15.  rejection audit    — success() was never True at ANY judged point;
  16.  final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it_i341.smoke --headless
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

YAW90 = (0.7071068, 0.0, 0.0, 0.7071068)  # long axis across the rail


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.shuttle_vault")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.80, -0.85, 0.78)) + o),
                                tuple(np.array((-0.02, 0.00, 0.14)) + o),
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

    def butter_rel() -> tuple[float, float, float]:
        r = scene._rel(scene.butter)[0]
        return float(r[0]), float(r[1]), float(r[2])

    def report(tag: str) -> None:
        bx, by, bz = butter_rel()
        op = float(scene.opening()[0])
        s, ok = judge()
        print(f"[smoke] {tag:16s} | butter=({bx:+.3f},{by:+.3f},{bz:.3f}) "
              f"op={op:+.4f} on_plate={bool(scene.on_plate(scene.butter)[0])} "
              f"conv={bool(scene.conveyed_now()[0])} "
              f"in_vault={bool(scene.in_vault(scene.butter)[0])} "
              f"closed={bool(scene.closed()[0])} "
              f"brick_out={bool(scene.brick_out()[0])} "
              f"boarded={bool(scene._boarded[0])} conveyed={bool(scene._conveyed[0])} "
              f"vaulted={bool(scene._vaulted[0])} resealed={bool(scene._resealed[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

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

    def write_shuttle(op: float) -> None:
        """Follower-only re-pose of the shuttle along its unchanged rail (probe
        constructor; the vault is never moved)."""
        place(scene.shuttle, c.vault_pos[0] + c.shuttle_x_auth + op, c.vault_pos[1],
              c.shuttle_z0)

    vx, vy = c.vault_pos
    bh = c.block_size[2] / 2  # 0.0225
    cav_rest_z = c.base_t + bh + 0.003  # flat rest on the cavity floor (+3 mm)
    plate_rest_z = c.ceil_top + c.plate_t + bh  # flat rest on the plate top

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    fin0 = (torch.isfinite(scene.butter.data.root_state_w).all()
            and torch.isfinite(scene.brick.data.root_state_w).all()
            and torch.isfinite(scene.shuttle.data.root_state_w).all())
    op = float(scene.opening()[0])
    still = (float(scene.butter.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.brick.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.shuttle.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    check("settle: states finite, shuttle parked ajar in its randomized range, both "
          "blocks outside on the floor, everything still",
          bool(fin0) and c.open_init_range[0] - 0.005 < op < c.open_init_range[1] + 0.005
          and still and not bool(scene.in_vault(scene.butter)[0])
          and not bool(scene.in_vault(scene.brick)[0])
          and not bool(scene.on_plate(scene.butter)[0])
          and butter_rel()[2] < 0.05)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-5. randomization is real =================================
    slots = np.array(c.spawn_slots)
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(2)
        row = []
        for body in (scene.butter, scene.brick):
            p = scene._rel(body)[0]
            px, py = float(p[0]), float(p[1])
            d = np.hypot(slots[:, 0] - px, slots[:, 1] - py)
            row += [px, py, int(d.argmin())]
        q = scene.butter.data.root_quat_w[0]
        row.append(math.degrees(2.0 * math.atan2(float(q[3]), float(q[0]))))
        row.append(float(scene.opening()[0]))
        reads.append(row)
    arr = np.array(reads)
    print(f"[smoke] randomization readback (bx, by, b_slot, rx, ry, r_slot, "
          f"b_yaw_deg, op0):\n{arr}", flush=True)
    sigs = {tuple(r[[2, 5]].astype(int)) for r in arr}
    check("randomization: the butter/brick slot assignment varies across seeded "
          f"resets (readback: {len(sigs)} distinct assignments)", len(sigs) >= 2)
    jit = 0.0
    for col, slot_col in ((0, 2), (3, 5)):
        for s_id in range(2):
            grp = arr[arr[:, slot_col] == s_id]
            if len(grp) >= 2:
                jit = max(jit, float((grp[:, col:col + 2].max(axis=0)
                                      - grp[:, col:col + 2].min(axis=0)).max()))
    yaw_spread = float(arr[:, 6].max() - arr[:, 6].min())
    check("randomization: per-slot xy jitter and spawn yaw vary (readback)",
          jit > 0.004 and yaw_spread > 20.0)
    op_spread = float(arr[:, 7].max() - arr[:, 7].min())
    check("randomization: the shuttle's initial opening op0 varies (readback: "
          f"{op_spread * 1000:.1f} mm over a 20 mm range)", op_spread > 0.004)

    # =========================== 6. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 7. seed strategy: place from above, close ==================
    # The seed's plan — open, lower the butter in from above, push shut — maps here to
    # dropping the butter from the sky over the hatch and pushing the shuttle closed.
    # The hatch lies UNDER the fixed hood: the butter lands ON THE HOOD ROOF and never
    # enters. Constructed settled: rejected.
    torch.manual_seed(41)
    env.reset()
    step(10)
    place(scene.butter, vx + 0.185, vy, c.roof_top + bh + 0.004, YAW90)
    write_shuttle(0.0)
    step(90)
    report("seed-strategy")
    s, ok = judge()
    bz = butter_rel()[2]
    check("seed strategy: butter dropped from above over the hatch lands ON THE HOOD "
          f"ROOF (z={bz:.3f}), shuttle pushed closed — never inside, no success, "
          "score <= 0.15",
          bz > 0.25 and not bool(scene.in_vault(scene.butter)[0])
          and bool(scene.closed()[0]) and not ok and s <= 0.15)

    # =========================== 8-9. short-of-the-cavity near misses =======================
    torch.manual_seed(51)
    env.reset()
    step(10)
    op0 = float(scene.opening()[0])
    place(scene.butter, vx + 0.075 + op0 / 2, vy, plate_rest_z + 0.002, YAW90)
    step(90)
    report("porch-camp")
    s, ok = judge()
    check("porch camp: butter laid on the porch and left there — boarded + approach "
          "credit only, no success, score <= 0.27",
          bool(scene.on_plate(scene.butter)[0]) and not bool(scene.conveyed_now()[0])
          and bool(scene._boarded[0]) and not ok and s <= 0.27)

    place(scene.butter, vx + 0.19, vy, plate_rest_z + 0.001, YAW90)  # arrest position
    step(90)
    report("under-hood-camp")
    s, ok = judge()
    check("under-hood camp: butter parked on the plate at the arrest position (hatch "
          "still covered) — + conveyed credit but not in the cavity, no success, "
          "score <= 0.36",
          bool(scene.conveyed_now()[0]) and not bool(scene.in_vault(scene.butter)[0])
          and not ok and s <= 0.36)

    # =========================== 10. vault left open ========================================
    # Butter settled INSIDE the cavity but the shuttle at full open: the closure
    # clause rejects (the goal is a SEALED vault, not a stocked one).
    torch.manual_seed(61)
    env.reset()
    step(10)
    write_shuttle(c.stroke)
    place(scene.butter, vx + 0.17, vy, cav_rest_z, YAW90)
    step(120)
    report("vault-open")
    s, ok = judge()
    check("vault left open: butter settled inside the cavity but the shuttle at full "
          "open — the closure clause rejects, no success, score <= 0.60",
          bool(scene.in_vault(scene.butter)[0]) and not bool(scene.closed()[0])
          and bool(scene._vaulted[0]) and not ok and s <= 0.60)

    # =========================== 11. brick smuggled in ======================================
    # Everything else right — butter settled in the cavity, shuttle at its closed
    # stop — but the RED brick inside too: the brick-out clause rejects at the cap.
    torch.manual_seed(71)
    env.reset()
    step(10)
    place(scene.butter, vx + 0.14, vy, cav_rest_z, YAW90)
    place(scene.brick, vx + 0.22, vy, cav_rest_z, YAW90)
    write_shuttle(0.0)
    step(120)
    report("brick-smuggled")
    s, ok = judge()
    check("brick smuggled: butter in the cavity AND shuttle closed but the RED brick "
          "inside too — no success, score <= 0.85 (color identification and the "
          "brick-out clause are load-bearing)",
          bool(scene.in_vault(scene.butter)[0]) and bool(scene.closed()[0])
          and not bool(scene.brick_out()[0]) and not ok and s <= 0.85)

    # =========================== 12. shuttle ajar ===========================================
    torch.manual_seed(81)
    env.reset()
    step(10)
    place(scene.butter, vx + 0.17, vy, cav_rest_z, YAW90)
    write_shuttle(0.025)  # 25 mm short of the closed stop (band is 10 mm)
    step(120)
    report("shuttle-ajar")
    s, ok = judge()
    check("shuttle ajar: butter in the cavity but the shuttle parked 25 mm short of "
          "its stop — the closed-band clause rejects, no success, score <= 0.60",
          bool(scene.in_vault(scene.butter)[0]) and not bool(scene.closed()[0])
          and not ok and s <= 0.60)

    # =========================== 13. latched credit survives regression =====================
    place(scene.butter, vx - 0.15, vy + 0.30, bh + 0.002)  # yank it back to the floor
    step(60)
    report("regressed")
    s_a, ok_a = judge()
    step(60)
    report("regressed2")
    s_b, ok_b = judge()
    check("latched credit: the vaulted latch earned then the butter yanked back to "
          f"the floor — score holds ({s_a:.3f} -> {s_b:.3f}, >= 0.35), still no "
          "success",
          bool(scene._vaulted[0]) and abs(s_a - s_b) < 1e-3 and s_a >= 0.35
          and not ok_a and not ok_b)

    # =========================== 14. sealed plate: closed shuttle admits nothing ============
    # Route/order forced by geometry, probed with force, not fiat: shuttle CLOSED,
    # butter parked on the plate directly over the covered hatch (where the open-
    # stroke deposit would drop it), then PRESSED DOWN at 2x its weight for 300
    # steps. Non-vacuous asserts: the press is ON while the butter's minimum z is
    # tracked — it never leaves plate height, never enters the cavity, and the plate
    # holds its closed stop under the press. Deposit strictly requires the open
    # stroke; only the return stroke seals.
    torch.manual_seed(91)
    env.reset()
    step(10)
    write_shuttle(0.0)
    place(scene.butter, vx + 0.19, vy, plate_rest_z + 0.002, YAW90)
    step(30)
    report("sealed-loaded")
    z_before = butter_rel()[2]
    fw = torch.zeros(n, 1, 3, device=device)
    fw[0, 0, 2] = -2.0 * c.butter_mass * 9.81  # 2x weight, straight down
    z_min = z_before
    for _ in range(300):
        scene.butter.set_external_force_and_torque(fw, zero_w, env_ids=all_ids,
                                                   is_global=True)
        step(1)
        z_min = min(z_min, butter_rel()[2])
    scene.butter.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids,
                                               is_global=True)
    step(60)
    report("sealed-pressed")
    s, ok = judge()
    op = float(scene.opening()[0])
    check("sealed plate: with the shuttle CLOSED the butter pressed down at 2x its "
          f"weight over the covered hatch stays at plate height the whole press "
          f"(min z={z_min:.4f} > 0.185, tracked DURING the press), never enters the "
          f"cavity, and the plate holds its stop (op={op:+.4f}) — a closed shuttle "
          "admits nothing, no success, score <= 0.40",
          z_before > c.plate_z_lo and z_min > 0.185
          and not bool(scene.in_vault(scene.butter)[0])
          and not bool(scene._vaulted[0])
          and op <= c.closed_tol + 0.004
          and not ok and s <= 0.40)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.butter.data.root_state_w).all()
           and torch.isfinite(scene.brick.data.root_state_w).all()
           and torch.isfinite(scene.shuttle.data.root_state_w).all()
           and torch.isfinite(scene.vault.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.shuttle_vault")
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
    except BaseException as exc:  # noqa: BLE001 — die loudly, never hang until the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({type(exc).__name__}: {exc})", flush=True)
        os._exit(1)

"""Smoke / rubric-REJECTION battery for BottleHangRailScene (sim_gen task
`stack_wine_i202`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — servo each carafe up through a port, slide the neck
down the narrow slot, release to hang — is the acceptance evidence that the rubric
ACCEPTS a correct outcome). Every teleport here is instrumentation that CONSTRUCTS a
wrong (or partial) outcome and asserts the rubric REJECTS it; no probe in this battery
ever reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: carafes standing on the floor
                            at CoM height, settled; score ~0 at rest, no success;
  3.  randomization       — READBACK over 6 seeded resets: rail xy + yaw vary, carafe
                            spawn xy and x-ORDER (slot permutation) vary;
  4.  hanging principle   — PhysX MASS readback matches the authored mass, and a
                            carafe teleport-CONSTRUCTED into a mid-band hang STAYS
                            hanging for 2 s (the collar physically cannot pass the
                            narrow slot — retention is contact dynamics, not
                            bookkeeping); one hung of three is still NOT success;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  SEED strategy       — the seed's plan transplanted verbatim ("lay the bottle ON
                            the rack"): all three carafes laid on TOP of the plate,
                            settled — support-from-below is exactly what this task
                            rejects: NOT success, score <= 0.20;
  7.  released-at-port    — a carafe released while still at the wide port: physics
                            rejects it — the collar falls straight back through, the
                            carafe ends below the plate, NOT hung, and the 24-substep
                            hang latch stays 0 (the keyhole airspace transit earns at
                            most the small ported shaping credit);
  8.  shallow-hang miss   — a carafe physically hanging by its collar but only ~8 mm
                            down the slot (< hang_y_min = 15 mm, still in the lift-out
                            region): collar verified resting in the seat band, yet NOT
                            hung, NOT success (depth is what makes the hang captive);
  9.  floor-below miss    — a carafe standing on the FLOOR directly under its keyhole,
                            x aligned and slot-depth aligned: only the seat-band clause
                            rejects it -> NOT hung, NOT success (alignment without
                            insertion is worthless);
  10. two-in-one-keyhole  — with a carafe correctly hung, a SECOND carafe dropped onto
                            the same keyhole from above the plate: it can never hang
                            there (the hang band is shorter than a collar diameter),
                            the first stays captive, NOT success;
  11. partial completion  — exactly one carafe hung (constructed), other two on the
                            floor: NOT success, score ~0.25 (one carafe's full stage
                            credit, no more);
  12. latched credit      — removing the one hung carafe leaves the latched score
                            unchanged and success stays gone (credit is an event
                            latch, not a live state);
  13. rejection audit     — success() was never True at ANY judged point;
  14. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.stack_wine_i202.smoke --headless
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


def quat_mul_f(a, b):
    """Hamilton product of two (w,x,y,z) float tuples."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + bw * ax + ay * bz - az * by,
            aw * by + bw * ay + az * bx - ax * bz,
            aw * bz + bw * az + ax * by - ay * bx)


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.bottle_hang_rail")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((0.85, -0.85, 0.70)) + o),
                                tuple(np.array((0.0, 0.0, 0.20)) + o),
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

    def finite() -> bool:
        return bool(torch.isfinite(scene.rail.data.root_state_w).all()
                    and all(torch.isfinite(b.data.root_state_w).all() for b in scene.bottles))

    def report(tag: str) -> None:
        s, ok = judge()
        hung, cell = scene.hang_cells()
        print(f"[smoke] {tag:18s} | hung={[bool(v) for v in hung[0]]} "
              f"cell={[int(v) for v in cell[0]]} "
              f"settled={[bool(v) for v in scene.settled()[0]]} "
              f"lift={[f'{float(v):.0f}' for v in scene.lifted_latch[0]]} "
              f"port={[f'{float(v):.0f}' for v in scene.ported_latch[0]]} "
              f"hang={[f'{float(v):.0f}' for v in scene.hung_latch[0]]} "
              f"maxctr={scene.max_hang_ctr[0].tolist()} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def rail_world(p_loc) -> torch.Tensor:
        rp = scene.rail.data.root_pos_w[0]
        rq = scene.rail.data.root_quat_w[0]
        v = torch.tensor([float(p_loc[0]), float(p_loc[1]), float(p_loc[2])], device=device)
        return rp + quat_apply(rq.unsqueeze(0), v.unsqueeze(0))[0]

    def rail_yaw() -> float:
        rq = scene.rail.data.root_quat_w[0]
        return 2.0 * float(torch.atan2(rq[3], rq[0]))

    def collar_local(i: int) -> torch.Tensor:
        b = scene.bottles[i]
        ez = torch.tensor([0.0, 0.0, 1.0], device=device)
        axis = quat_apply(b.data.root_quat_w[0].unsqueeze(0), ez.unsqueeze(0))[0]
        collar = b.data.root_pos_w[0] + axis * c.collar_c
        rp = scene.rail.data.root_pos_w[0]
        rq = scene.rail.data.root_quat_w[0]
        return quat_apply_inverse(rq.unsqueeze(0), (collar - rp).unsqueeze(0))[0]

    def place_world(i: int, pos_w: torch.Tensor, quat=(1.0, 0.0, 0.0, 0.0),
                    settle_steps: int = 30) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap). `pos_w` is the ROOT position."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = pos_w[0], pos_w[1], pos_w[2]
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        scene.bottles[i].write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def place_floor(i: int, x: float, y: float, settle_steps: int = 30) -> None:
        pos = scene.env_origins[0] + torch.tensor([x, y, c.com_z + 0.002], device=device)
        place_world(i, pos, settle_steps=settle_steps)

    def construct_hang(i: int, k: int, along: float, settle_steps: int = 180) -> None:
        """Teleport carafe i into a HANGING pose in keyhole k at slot depth `along`
        (upright, collar 0.5 mm above the seat, neck in the channel), then settle."""
        tgt = rail_world((c.cells_x[k], c.chan_y0 + along, c.seat_z_loc + 0.0005))
        root = tgt - torch.tensor([0.0, 0.0, c.collar_c], device=device)
        place_world(i, root, settle_steps=settle_steps)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    heights = [float((scene.bottles[i].data.root_pos_w[0] - scene.env_origins[0])[2])
               for i in range(3)]
    check("settle: states finite; all three carafes standing on the floor at CoM "
          f"height (readback z={[f'{h:.3f}' for h in heights]}) and settled",
          finite() and all(abs(h - c.com_z) < 0.006 for h in heights)
          and bool(scene.settled()[0].all()))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3. randomization is real ===================================
    rail_reads, spawn_reads, orders = [], [], set()
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(5)
        rp = (scene.rail.data.root_pos_w[0] - scene.env_origins[0])
        rail_reads.append([float(rp[0]), float(rp[1]), math.degrees(rail_yaw())])
        xy = [(float((b.data.root_pos_w[0] - scene.env_origins[0])[0]),
               float((b.data.root_pos_w[0] - scene.env_origins[0])[1]))
              for b in scene.bottles]
        spawn_reads.append([v for p in xy for v in p])
        orders.add(tuple(np.argsort([p[0] for p in xy]).tolist()))
    rarr = np.array(rail_reads)
    sarr = np.array(spawn_reads)
    print(f"[smoke] rail readback (x, y, yaw_deg) across seeds:\n{rarr}", flush=True)
    print(f"[smoke] carafe spawn xy across seeds:\n{sarr}", flush=True)
    print(f"[smoke] carafe x-orderings across seeds: {orders}", flush=True)
    check("randomization: rail xy + yaw vary across seeded resets (readback), and "
          "carafe spawn xy and x-order (slot permutation) vary",
          (rarr.max(0) - rarr.min(0))[0] > 0.015 and (rarr.max(0) - rarr.min(0))[1] > 0.010
          and (rarr.max(0) - rarr.min(0))[2] > 2.0
          and float((sarr.max(0) - sarr.min(0)).max()) > 0.02 and len(orders) >= 2)

    # =========================== 4. the hanging principle is physical =======================
    env.reset(seed=31)
    step(10)
    masses = [float(b.root_physx_view.get_masses().cpu().view(-1)[0]) for b in scene.bottles]
    print(f"[smoke] PhysX mass readback: {masses}", flush=True)
    construct_hang(0, 1, along=0.024, settle_steps=240)  # 2 s of real contact
    report("retention")
    s, ok = judge()
    loc = collar_local(0)
    hung, cell = scene.hang_cells()
    check("hanging principle: authored mass readback matches "
          f"(m={masses[0]:.3f} kg), and a carafe CONSTRUCTED into a mid-band hang "
          f"stays hanging for 2 s (collar z_loc={float(loc[2]):+.4f} in the seat "
          "band — the collar cannot pass the slot), one hung of three -> NOT success",
          all(abs(m - c.mass) < 1e-3 for m in masses)
          and c.collar_z_lo <= float(loc[2]) <= c.collar_z_hi
          and bool(hung[0, 0]) and int(cell[0, 0]) == 1 and not ok)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=41)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. SEED strategy (lay it ON the rack) ======================
    # The seed's whole plan is "grasp the bottle, LAY it on the rack" — object resting
    # ON TOP of the fixture. Transplant it verbatim: all three carafes lying on top of
    # the plate (on the solid strips between/beside the keyholes), settled.
    env.reset(seed=51)
    step(10)
    yaw = rail_yaw()
    q_yaw = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    q_x90 = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)  # +z -> -y_local
    q_lie = quat_mul_f(q_yaw, q_x90)
    for i, xk in enumerate((-c.cell_half, c.cell_half, c.cells_x[-1] + c.cell_half + c.cap_hw)):
        root = rail_world((xk, -0.018, c.plate_t / 2 + c.body_r + 0.0008))
        place_world(i, root, quat=q_lie, settle_steps=90)
    step(240)
    report("seed-on-top")
    s, ok = judge()
    tops = [float((b.data.root_pos_w[0] - scene.env_origins[0])[2]) for b in scene.bottles]
    on_top = all(abs(z - (c.shelf_top_z + c.body_r)) < 0.012 for z in tops)
    check("SEED strategy: all three carafes laid ON TOP of the rail plate (readback "
          f"z={[f'{z:.3f}' for z in tops]}), settled — support-from-below is NOT "
          f"hanging: NOT success, score <= 0.20 (got {s:.3f})",
          on_top and bool(scene.settled()[0].all()) and not ok and s <= 0.20)

    # =========================== 7. released at the wide port ===============================
    env.reset(seed=61)
    step(10)
    tgt = rail_world((c.cells_x[1], c.port_cy, 0.013))  # collar just above the port
    place_world(1, tgt - torch.tensor([0.0, 0.0, c.collar_c], device=device),
                settle_steps=300)
    report("released-at-port")
    s, ok = judge()
    loc = collar_local(1)
    hung, _cell = scene.hang_cells()
    z_env = float((scene.bottles[1].data.root_pos_w[0] - scene.env_origins[0])[2])
    check("released-at-port: a carafe released while still at the wide port falls "
          f"straight back through (collar z_loc={float(loc[2]):+.3f}, root env "
          f"z={z_env:.3f}) — NOT hung, NOT success, and the 24-substep hang latch "
          "stays 0",
          float(loc[2]) < -0.05 and z_env < 0.20 and not bool(hung[0, 1]) and not ok
          and float(scene.hung_latch[0, 1]) == 0.0
          and int(scene.max_hang_ctr[0, 1]) < c.hang_latch_steps)

    # =========================== 8. shallow-hang near-miss ==================================
    env.reset(seed=71)
    step(10)
    construct_hang(2, 0, along=0.008, settle_steps=240)  # in the lift-out region
    report("shallow-hang")
    s, ok = judge()
    loc = collar_local(2)
    hung, _cell = scene.hang_cells()
    along_rb = float(loc[1]) - c.chan_y0
    check("shallow-hang: a carafe physically hanging by its collar (seat-band z "
          f"readback {float(loc[2]):+.4f}) but only {along_rb * 1000:.0f} mm down the "
          "slot (< hang_y_min, still liftable back out through the port) — NOT hung, "
          "NOT success",
          c.collar_z_lo <= float(loc[2]) <= c.collar_z_hi
          and along_rb < c.hang_y_min and not bool(hung[0, 2]) and not ok)

    # =========================== 9. floor-below near-miss ===================================
    env.reset(seed=81)
    step(10)
    under = rail_world((c.cells_x[2], c.chan_y0 + 0.024, 0.0))
    place_floor(0, float(under[0] - scene.env_origins[0][0]),
                float(under[1] - scene.env_origins[0][1]), settle_steps=120)
    report("floor-below")
    s, ok = judge()
    loc = collar_local(0)
    hung, _cell = scene.hang_cells()
    xerr = abs(float(loc[0]) - c.cells_x[2])
    along_rb = float(loc[1]) - c.chan_y0
    check("floor-below: a carafe standing on the floor directly under its keyhole — "
          f"x aligned (|xerr|={xerr * 1000:.1f} mm) and slot-depth aligned "
          f"(along={along_rb * 1000:.0f} mm) yet collar z_loc={float(loc[2]):+.3f} "
          "is far below the seat band — NOT hung, NOT success",
          xerr < c.x_tol and c.hang_y_min <= along_rb <= c.hang_y_max
          and float(loc[2]) < c.collar_z_lo and not bool(hung[0, 0]) and not ok)

    # =========================== 10. two-in-one-keyhole =====================================
    env.reset(seed=91)
    step(10)
    construct_hang(0, 1, along=0.024, settle_steps=180)  # first carafe: genuine hang
    drop = rail_world((c.cells_x[1], c.chan_y0 + 0.015, 0.10))  # second: from above
    place_world(1, drop - torch.tensor([0.0, 0.0, c.collar_c], device=device),
                settle_steps=360)
    report("two-in-one")
    s, ok = judge()
    hung, cell = scene.hang_cells()
    check("two-in-one-keyhole: a second carafe dropped onto an occupied keyhole from "
          "above the plate never hangs there (the hang band is shorter than a collar "
          "diameter), the first stays captive — NOT success",
          bool(hung[0, 0]) and int(cell[0, 0]) == 1 and not bool(hung[0, 1]) and not ok)

    # =========================== 11-12. partial completion + latched credit =================
    env.reset(seed=101)
    step(10)
    construct_hang(0, 2, along=0.024, settle_steps=240)
    report("one-hung")
    s_in, ok = judge()
    check("partial completion: exactly one carafe hung, two still on the floor — NOT "
          f"success, one carafe's stage credit only (score {s_in:.3f} in [0.20, 0.30])",
          0.20 <= s_in <= 0.30 and not ok)
    place_floor(0, 0.30, -0.40, settle_steps=60)
    report("hung-removed")
    s_out, ok = judge()
    hung, _cell = scene.hang_cells()
    check("latched credit: removing the one hung carafe leaves the latched score "
          f"unchanged ({s_in:.3f} -> {s_out:.3f}), live hang gone, success stays gone",
          abs(s_out - s_in) < 0.02 and not bool(hung[0].any()) and not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.bottle_hang_rail")
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
    except BaseException:  # noqa: BLE001 — die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)

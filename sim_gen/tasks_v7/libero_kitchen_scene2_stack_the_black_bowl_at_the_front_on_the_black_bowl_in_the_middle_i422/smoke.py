"""Smoke / rubric-REJECTION battery for ScreedPatchScene (sim_gen task
libero_kitchen_scene2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle_i422)
— NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — hover-release paving + the force-dragged screed pass —
is the acceptance evidence that the rubric ACCEPTS a correct outcome). Every teleport here
is instrumentation that CONSTRUCTS a wrong (or partially-right) outcome as a settled state
and asserts the rubric's verdict on it. No probe below constructs full success.

  1-2.  settle/no-NaN    — racked slabs + parked bar settle finite, nothing in the pit,
                           score 0, no success;
  3.    determinism      — the same seed twice -> identical layout readback;
  4-6.  randomization    — READBACK over 10 seeded resets: pit depth takes BOTH values and
                           the physical pit-floor z matches it; pit centre jitters; depot
                           side takes both values, keystone slot and bar park vary;
  7.    null policy      — 240 idle steps -> score ~0, no success;
  8.    wrong order      — the seed's "stack A on B" plan: pavers laid first, RED keystone
                           stacked ON TOP. The patch is geometrically FLUSH, yet interred
                           and covered are False -> score 0.00, no success;
  9.    underfill        — keystone alone (one course short) -> interred latches 0.15, but
                           not flush, not covered -> score 0.15, no success;
  10.   overfill         — keystone + depth pavers burst-released as one column (never
                           passing through a complete patch): the proud slab is over the
                           pit footprint -> not flush -> patch credit never latches, 0.15;
  11.   flush w/o burial — depth gray pavers, keystone left racked -> flush TRUE, score 0;
  12-15. screed-pass gates (after building a legit patch, credit 0.50):
                           bar TELEPORTED clear to the far side (zero written velocity) ->
                           pass never latches, success False, score pinned 0.50;
                           bar flown OVER the pit with written velocity 60 mm up -> no
                           latch (z band); bar teleport-DROPPED onto the patch (vertical
                           impact transient) -> no latch (vz veto + streak);
  16-17. premature sweep — bar genuinely force-dragged across the EMPTY pit (it crosses,
                           riding its skid runners) -> no latch; completing the patch
                           afterwards still leaves success False (the pass must come
                           after the patch);
  18.   tolerance twins  — keystone dropped 15 mm off-centre still interred (walls
                           self-centre it); a paver dropped 12 mm off still covers;
  19.   latched credit   — the 0.50 patch credit survives the top paver being removed;
  20.   finite at the end.

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
# RTX recipe: kit mis-decodes the driver version and silently rejects RTX -> the annotator
# returns EMPTY frames. Disable the driver check.
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

# Kit teardown hangs leave GPU zombies: global watchdog long before the forge timeout.
threading.Timer(1350.0, lambda: os._exit(4)).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.screed_patch")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    origin = env.iscene.env_origins

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = origin[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.05, -1.05, 0.95)) + o),
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

    def settle(max_steps: int = 500, poll: int = 10) -> None:
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if bool(scene.settled()[0]):
                return

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def report(tag: str) -> None:
        print(f"[smoke] {tag:18s} | interred={bool(scene.interred()[0])} "
              f"covered={bool(scene.covered()[0])} flush={bool(scene.flush()[0])} "
              f"swept={bool(scene.ever_swept[0])} clear={bool(scene.bar_clear()[0])} "
              f"settled={bool(scene.settled()[0])} score={float(scene.score()[0]):.3f} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    def place(body, x: float, y: float, z: float, quat: tuple | None = None,
              vel: tuple | None = None, settle_steps: int = 30) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics steps
        before judging."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3:7] = torch.tensor(quat or (1.0, 0.0, 0.0, 0.0), device=device)
        if vel is not None:
            st[:, 7:10] = torch.tensor(vel, device=device)
        st[:, 0:3] += origin
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def pit_info() -> tuple:
        px = float(scene.pit_xy[0, 0])
        py = float(scene.pit_xy[0, 1])
        d = int(round(float(scene.depth_units[0])))
        ft = float(scene.floor_top()[0])
        side = float(scene.depot_side[0])
        return px, py, d, ft, side

    def local_z(body) -> float:
        return float(body.data.root_pos_w[0, 2] - origin[0, 2])

    def drop_at(body, course: int, dx: float = 0.0, dy: float = 0.0) -> None:
        px, py, _d, ft, _s = pit_info()
        place(body, px + dx, py + dy, ft + (course + 0.5) * c.slab_t + 0.030,
              settle_steps=20)
        settle(max_steps=300)

    def park_on_deck(body, k: int = 0) -> None:
        px, py, _d, _ft, side = pit_info()
        place(body, px - 0.35 - 0.02 * k, py - side * 0.22,
              c.deck_top + c.slab_t / 2 + 0.002, settle_steps=20)

    def build_patch() -> None:
        """Legit patch: keystone hover-dropped onto the floor, then depth-1 pavers."""
        _px, _py, d, _ft, _s = pit_info()
        drop_at(scene.keystone, 0)
        for course in range(1, d):
            drop_at(scene.pavers[course - 1], course)
        settle(max_steps=300)

    def drag_bar(f_drag: float = 1.6, v_cap: float = 0.20) -> None:
        """Genuine force-dragged pass (probe instrumentation mirroring the solver),
        including the solver's lateral P-D steer toward the pit centreline — uneven runner
        friction walks an unsteered bar sideways, and over the OPEN pit a ~30 mm drift
        drops a runner into the opening."""
        px, py, _d, _ft, _s = pit_info()
        zero_wrench = torch.zeros(n, 1, 3, device=device)

        def bar_force(fx: float, fy: float) -> None:
            f_w = torch.zeros(n, 1, 3, device=device)
            f_w[:, 0, 0] = fx
            f_w[:, 0, 1] = fy
            scene.bar.set_external_force_and_torque(f_w, zero_wrench, env_ids=all_ids,
                                                    is_global=True)

        def steer_fy() -> float:
            by = float(scene.bar.data.root_pos_w[0, 1] - origin[0, 1])
            vy = float(scene.bar.data.root_lin_vel_w[0, 1])
            return max(-0.8, min(0.8, -6.0 * (by - py) - 1.2 * vy))

        for i in range(600):
            b = scene.bar.data.root_pos_w[0] - origin[0]
            if float(b[0]) - px > 0.16:
                break
            if i % 100 == 0:
                print(f"[smoke]   drag step {i}: bar=({float(b[0]):+.3f},"
                      f"{float(b[1]):+.3f},{float(b[2]):.4f})", flush=True)
            vx = float(scene.bar.data.root_lin_vel_w[0, 0])
            bar_force(f_drag if vx < v_cap else 0.0, steer_fy())
            env.step(no_action)
        for _ in range(150):
            v = scene.bar.data.root_lin_vel_w[0, :2]
            if float(v.norm()) < 0.06:
                break
            bar_force(-1.5 if float(v[0]) > 0 else 1.5, steer_fy())
            env.step(no_action)
        scene.bar.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
        settle(max_steps=300)
        b = scene.bar.data.root_pos_w[0] - origin[0]
        print(f"[smoke]   drag done: bar=({float(b[0]):+.3f},{float(b[1]):+.3f},"
              f"{float(b[2]):.4f})", flush=True)

    def layout_readback() -> tuple:
        floor_phys = local_z(scene.pit_floor) + c.floor_t / 2
        return (float(scene.pit_xy[0, 0]), float(scene.pit_xy[0, 1]),
                float(scene.depth_units[0]), float(scene.depot_side[0]),
                float(scene.key_slot[0]), float(scene.bar_park_x[0]), floor_phys)

    def all_finite() -> bool:
        bodies = [scene.keystone, *scene.pavers, scene.bar, scene.pit_floor]
        return all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)

    def score() -> float:
        return float(scene.score()[0])

    # =========================== 1-2. settle / no-NaN / clean rubric =========================
    env.reset(seed=11)
    settle()
    report("reset")
    check("settle: racked slabs + parked bar settle finite; keystone still on the rack "
          "above deck level; nothing interred/covered/flush; bar not clear",
          all_finite() and bool(scene.settled()[0])
          and local_z(scene.keystone) > c.deck_top + 0.01
          and not bool(scene.interred()[0]) and not bool(scene.covered()[0])
          and not bool(scene.flush()[0]) and not bool(scene.bar_clear()[0]))
    check("rubric clean at reset: score 0, no success",
          score() <= 0.005 and not bool(scene.success()[0]))

    # =========================== 3. determinism =============================================
    env.reset(seed=777)
    step(3)
    read_a = layout_readback()
    env.reset(seed=777)
    step(3)
    read_b = layout_readback()
    print(f"[smoke] determinism readback: {read_a} vs {read_b}", flush=True)
    check("determinism: same seed -> identical layout readback",
          all(abs(a - b) < 1e-5 for a, b in zip(read_a, read_b)))

    # =========================== 4-6. randomization is real ==================================
    reads = []
    for s in (21, 22, 23, 24, 25, 26, 27, 28, 29, 30):
        env.reset(seed=s)
        step(3)
        reads.append(layout_readback())
    arr = np.array(reads)
    print(f"[smoke] randomization readback (pit_x, pit_y, depth, side, slot, bar_x, "
          f"floor_phys):\n{arr}", flush=True)
    depth_ok = arr[:, 2].max() - arr[:, 2].min() > 0.5
    floor_match = np.all(np.abs(arr[:, 6] - (c.deck_top - arr[:, 2] * c.slab_t)) < 0.002)
    check("randomization: pit depth takes both values AND the physical pit-floor z "
          "readback matches the sampled depth", bool(depth_ok and floor_match))
    check("randomization: pit centre jitters across seeds (readback)",
          (arr[:, 0].max() - arr[:, 0].min()) > 0.010
          and (arr[:, 1].max() - arr[:, 1].min()) > 0.012)
    check("randomization: depot side takes both values, keystone slot varies, bar park "
          "jitters (readback)",
          arr[:, 3].max() - arr[:, 3].min() > 1.0
          and arr[:, 4].max() - arr[:, 4].min() > 0.5
          and (arr[:, 5] - arr[:, 0]).max() - (arr[:, 5] - arr[:, 0]).min() > 0.004)

    # =========================== 7. null policy fails ========================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    check("null policy: score ~0 and no success after 240 idle steps",
          score() <= 0.02 and not bool(scene.success()[0]))

    # =========================== 8. wrong order (the seed strategy) ==========================
    # The seed's whole plan: STACK A on B in the open. Constructed here inside the pit but
    # in the wrong order — gray pavers first, RED keystone on top. The patch is
    # geometrically FLUSH (this also proves the flush gate is satisfiable), yet the
    # keystone is a full course off the floor seat and covered by nothing: score 0.
    env.reset(seed=41)
    settle()
    _px, _py, d, _ft, _s = pit_info()
    for course in range(d - 1):
        drop_at(scene.pavers[course], course)
    drop_at(scene.keystone, d - 1)
    report("wrong-order")
    check("wrong order (pavers first, keystone stacked ON TOP, patch flush): flush TRUE "
          "but interred/covered FALSE -> score 0.00, no success",
          bool(scene.flush()[0]) and not bool(scene.interred()[0])
          and not bool(scene.covered()[0]) and score() <= 0.005
          and not bool(scene.success()[0]))

    # =========================== 9. underfill ================================================
    env.reset(seed=51)
    settle()
    drop_at(scene.keystone, 0)
    report("underfill")
    check("underfill (keystone alone, one course short): interred latches 0.15 but not "
          "covered, not flush -> score 0.15, no success",
          bool(scene.interred()[0]) and not bool(scene.covered()[0])
          and not bool(scene.flush()[0]) and abs(score() - 0.15) < 1e-3
          and not bool(scene.success()[0]))

    # =========================== 10. overfill ================================================
    # Burst construction: the whole depth+1 column is hover-released in ONE burst (graduated
    # hover heights, no settling between releases), so the extra paver is over the pit
    # footprint from the very first step and the run never passes through a legitimate
    # complete-patch rest state. Footprint-membership flush must price the proud slab.
    env.reset(seed=61)
    settle()
    px, py, d, ft, _s = pit_info()
    place(scene.keystone, px, py, ft + 0.5 * c.slab_t + 0.030, settle_steps=0)
    for course in range(1, d + 1):
        place(scene.pavers[course - 1], px, py,
              ft + (course + 0.5) * c.slab_t + 0.030 + 0.010 * course, settle_steps=0)
    step(20)
    settle(max_steps=400)
    report("overfill")
    check("overfill (keystone + depth pavers, one course proud): covered TRUE but flush "
          "FALSE (the proud slab is over the pit footprint) -> patch credit never latches, "
          "score 0.15, no success",
          bool(scene.interred()[0]) and bool(scene.covered()[0])
          and not bool(scene.flush()[0]) and abs(score() - 0.15) < 1e-3
          and not bool(scene.success()[0]))

    # =========================== 11. flush without the keystone ==============================
    env.reset(seed=71)
    settle()
    _px, _py, d, _ft, _s = pit_info()
    for course in range(d):
        drop_at(scene.pavers[course], course)
    report("no-keystone")
    check("flush without burial (depth gray pavers, keystone left racked): flush TRUE but "
          "keystone gates FALSE -> score 0.00, no success",
          bool(scene.flush()[0]) and not bool(scene.interred()[0])
          and not bool(scene.covered()[0]) and score() <= 0.005
          and not bool(scene.success()[0]))

    # =========================== 12-15. screed-pass gates ====================================
    env.reset(seed=81)
    settle()
    px, py, d, ft, side = pit_info()
    build_patch()
    report("patch-built")
    check("legit patch built by hover-drops: interred+covered+flush at rest -> patch "
          "credit 0.50 latched (no pass yet)",
          bool(scene.interred()[0]) and bool(scene.covered()[0])
          and bool(scene.flush()[0]) and abs(score() - 0.50) < 1e-3)
    # 13: bar TELEPORTED clear to the far side (zero written velocity)
    place(scene.bar, px + 0.20, py, c.bar_rest_z + 0.001, settle_steps=20)
    settle()
    report("bar-teleported")
    check("teleported bar: parked clear on the far side with zero written velocity -> "
          "pass never latched, success False, score pinned at 0.50",
          bool(scene.bar_clear()[0]) and not bool(scene.ever_swept[0])
          and not bool(scene.success()[0]) and abs(score() - 0.50) < 1e-3)
    # 14: bar flown OVER the pit, 60 mm up, WITH written velocity
    place(scene.bar, px - 0.03, py, c.bar_rest_z + 0.060, vel=(0.30, 0.0, 0.0),
          settle_steps=3)
    swept_air = bool(scene.ever_swept[0])
    place(scene.bar, float(scene.bar_park_x[0]), py, c.bar_rest_z + 0.001, settle_steps=20)
    check("air carry: bar flown over the pit 60 mm up with written velocity -> no pass "
          "latch (deck-level z band)", not swept_air)
    # 15: bar teleport-DROPPED onto the patch (vertical impact transient, no written vx)
    place(scene.bar, px, py, c.bar_rest_z + 0.004, settle_steps=25)
    settle()
    swept_drop = bool(scene.ever_swept[0])
    place(scene.bar, float(scene.bar_park_x[0]), py, c.bar_rest_z + 0.001, settle_steps=20)
    settle()
    report("bar-dropped-on")
    check("teleport-drop onto the patch: vertical impact transient over the pit -> no "
          "pass latch (vz veto + streak), score still 0.50",
          not swept_drop and abs(score() - 0.50) < 1e-3
          and not bool(scene.success()[0]))

    # =========================== 16-17. premature sweep ======================================
    env.reset(seed=91)
    settle()
    px, py, d, ft, side = pit_info()
    drag_bar()
    bx = float(scene.bar.data.root_pos_w[0, 0] - origin[0, 0])
    report("premature-sweep")
    check("premature sweep: bar genuinely force-dragged across the EMPTY pit (it crosses, "
          "riding its skid runners) -> no pass latch, score 0",
          bx - px > 0.10 and not bool(scene.ever_swept[0]) and score() <= 0.005)
    build_patch()
    settle()
    report("patch-after-sweep")
    check("order forcing: completing the patch AFTER the sweep leaves success False "
          "(the pass must run over the finished patch), score 0.50",
          bool(scene.flush()[0]) and not bool(scene.ever_swept[0])
          and not bool(scene.success()[0]) and abs(score() - 0.50) < 1e-3)

    # =========================== 18. tolerance twins =========================================
    env.reset(seed=101)
    settle()
    px, py, d, ft, side = pit_info()
    drop_at(scene.keystone, 0, dx=0.015, dy=-0.010)
    ok_key = bool(scene.interred()[0])
    drop_at(scene.pavers[0], 1, dx=-0.012, dy=0.012)
    ok_cov = bool(scene.covered()[0])
    report("twins")
    check("tolerance twins: keystone dropped 15 mm off-centre still interred (pit walls "
          "self-centre it); a paver dropped 12 mm off still covers", ok_key and ok_cov)

    # =========================== 19. latched credit survives =================================
    for course in range(2, d):
        drop_at(scene.pavers[course - 1], course)
    settle(max_steps=300)
    got_patch = abs(score() - 0.50) < 1e-3
    top_paver = scene.pavers[d - 2]
    park_on_deck(top_paver, k=1)
    settle()
    report("paver-removed")
    check("latched credit: the 0.50 patch credit survives the top paver being removed "
          "(covered now False, success False)",
          got_patch and abs(score() - 0.50) < 1e-3 and not bool(scene.covered()[0])
          and not bool(scene.success()[0]))

    # =========================== 20. finite at the end ========================================
    check("finite: all states finite at the end", all_finite())

    # =========================== save + verdict ==============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.screed_patch")
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

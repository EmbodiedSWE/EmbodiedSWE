"""Smoke / rubric-REJECTION battery for SpindleServeScene (sim_gen task
libero_kitchen_scene2_put_the_black_bowl_in_the_middle_on_the_plate_i2) — NullRobot,
teleported probe states, RECORDED.

This is NOT a solution (solve.py — the teleport-transport + contact-dynamics run — is the
acceptance evidence that the rubric ACCEPTS a correct outcome). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partially-right) outcome as a settled state and
asserts the rubric's verdict on it. No probe below constructs full success.

  1. settle/no-NaN      — the authored stack settles finite, all three rings threaded on
                          the SOURCE dowel, score 0;
  2. determinism        — the same seed twice -> identical layout readback;
  3. randomization      — READBACK over 6 seeded resets: the middle ring's color varies,
                          the stand mirror side varies, positions jitter;
  4. null policy        — 240 idle steps -> score ~0, no success;
  5. seed strategy      — the seed's plan (grab the one accessible ring, put it on the
                          plate): TOP ring dropped on the plate, all else untouched ->
                          success False, score 0;
  6. incomplete         — middle ring served but the other two left on the SOURCE dowel ->
                          success False, score pinned at the 0.30 served milestone;
  7. gate honesty       — a 20 mm off-centre set-down counts as on_plate (tolerance twin),
                          a settled 50 mm off-centre set-down (physically resting,
                          overhanging) does NOT;
  8. edge stand         — the middle ring standing on its EDGE face on the plate (a stable
                          pose) -> not served, score 0;
  9. beside dowel       — non-middle rings resting on the counter beside the spare stand ->
                          not threaded, success False;
 10. wrong ring         — TOP ring on the plate with middle+bottom threaded on the spare ->
                          success False (score 0.15: one legitimate non-middle threading);
 11. latched credit     — one non-middle ring physically drop-threaded on the spare latches
                          0.15 and the credit survives removing the ring again;
 12. dumped pile        — all three rings piled on the plate -> success False;
 13. finite at the end.

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

# Kit teardown hangs leave GPU zombies: global watchdog long before the forge timeout.
threading.Timer(1350.0, lambda: os._exit(4)).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.spindle_serve")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.30, -1.30, 0.95)) + o),
                                tuple(np.array((0.06, 0.0, 0.24)) + o),
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
        thr_sp = scene.threaded_spare()[0].tolist()
        thr_src = scene.threaded_src()[0].tolist()
        print(f"[smoke] {tag:18s} | mid={int(scene.mid_idx[0])} "
              f"side={float(scene.side[0]):+.0f} src={thr_src} spare={thr_sp} "
              f"served={bool(scene.served()[0])} settled={bool(scene.settled()[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    def flat_quat(yaw: float) -> tuple:
        return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

    def place_ring(r: int, x: float, y: float, z: float, quat: tuple | None = None,
                   settle_steps: int = 30) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics steps
        before judging."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3:7] = torch.tensor(quat or (1.0, 0.0, 0.0, 0.0), device=device)
        st[:, 0:3] += origin
        scene.rings[r].write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def plate_xy() -> tuple:
        p = scene.plate.data.root_pos_w[0] - origin[0]
        return float(p[0]), float(p[1])

    def plate_top() -> float:
        return float(scene.plate.data.root_pos_w[0, 2]) + c.plate_h / 2

    def spare_xy() -> tuple:
        p = scene.stand_spare.data.root_pos_w[0] - origin[0]
        return float(p[0]), float(p[1])

    def spare_tip() -> float:
        return float(scene.stand_spare.data.root_pos_w[0, 2]) + c.base_h / 2 + c.dowel_h

    def drop_on_plate(r: int, dx: float = 0.0, dy: float = 0.0) -> None:
        px, py = plate_xy()
        place_ring(r, px + dx, py + dy, plate_top() + c.ring_thick / 2 + 0.025,
                   flat_quat(0.3), settle_steps=10)
        settle()

    def drop_thread_spare(r: int, off: float = 0.003) -> None:
        sx, sy = spare_xy()
        place_ring(r, sx + off, sy, spare_tip() + c.ring_thick / 2 + 0.030,
                   flat_quat(0.5), settle_steps=10)
        settle()

    def order_ids() -> tuple:
        lev = scene.level_of[0].tolist()
        order = sorted(range(c.n_rings), key=lambda r: lev[r])
        return order[0], order[1], order[2]  # bottom, middle, top

    def layout_readback() -> tuple:
        px, py = plate_xy()
        sx, sy = spare_xy()
        return (int(scene.mid_idx[0]), float(scene.side[0]), px, py, sx, sy)

    # =========================== 1-2. settle / no-NaN / clean rubric =========================
    env.reset(seed=11)
    settle()
    report("reset")
    finite = all(torch.isfinite(r.data.root_state_w).all() for r in scene.rings) \
        and bool(torch.isfinite(scene.plate.data.root_state_w).all())
    check("settle: authored stack finite, settled, all three rings threaded on SOURCE dowel",
          finite and bool(scene.settled()[0])
          and bool(scene.threaded_src()[0].all()) and not bool(scene.threaded_spare()[0].any()))
    check("rubric clean at reset: score 0, no success",
          float(scene.score()[0]) <= 0.005 and not bool(scene.success()[0]))

    # =========================== 3. determinism =============================================
    env.reset(seed=777)
    step(3)
    read_a = layout_readback()
    env.reset(seed=777)
    step(3)
    read_b = layout_readback()
    print(f"[smoke] determinism readback: {read_a} vs {read_b}", flush=True)
    check("determinism: same seed -> identical layout readback",
          read_a[0] == read_b[0] and all(abs(a - b) < 1e-5
                                         for a, b in zip(read_a[1:], read_b[1:])))

    # =========================== 4. randomization is real ====================================
    reads = []
    for s in (21, 22, 23, 24, 25, 26):
        env.reset(seed=s)
        step(3)
        reads.append(layout_readback())
    arr = np.array(reads)
    print(f"[smoke] randomization readback (mid, side, plate_x, plate_y, spare_x, spare_y):\n"
          f"{arr}", flush=True)
    check("randomization: the MIDDLE ring's color varies across seeds (readback)",
          len({int(v) for v in arr[:, 0]}) >= 2)
    check("randomization: the stand mirror side varies across seeds",
          arr[:, 1].max() - arr[:, 1].min() > 1.0)
    check("randomization: plate position jitters across seeds (readback)",
          (arr[:, 2].max() - arr[:, 2].min()) > 0.005
          or (arr[:, 3].max() - arr[:, 3].min()) > 0.005)

    # =========================== 5. null policy fails ========================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    check("null policy: score ~0 and no success after 240 idle steps",
          float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 6. seed strategy control =====================================
    # The seed's whole plan: pick the one accessible item, set it on the plate. Here that is
    # the TOP ring — and it leaves the middle ring buried and the spare dowel empty.
    env.reset(seed=41)
    settle()
    _b, _m, top = order_ids()
    drop_on_plate(top)
    report("seed-strategy")
    check("seed strategy (top ring set on the plate, all else untouched): no success, score 0",
          not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.02
          and bool(scene.on_plate()[0, top]))

    # =========================== 7. incomplete: served but not re-threaded ====================
    env.reset(seed=51)
    settle()
    _b, mid, _t = order_ids()
    drop_on_plate(mid)  # probe constructor: middle out (instrumentation), others left on SOURCE
    settle()
    report("served-only")
    check("incomplete (middle served, other rings still on the SOURCE dowel): no success, "
          "score pinned at 0.30",
          not bool(scene.success()[0]) and bool(scene.served()[0])
          and abs(float(scene.score()[0]) - 0.30) < 1e-3)

    # =========================== 8. gate honesty: twin + near miss ============================
    px, py = plate_xy()
    place_ring(mid, px + 0.020, py, plate_top() + c.ring_thick / 2 + 0.020,
               flat_quat(0.2), settle_steps=10)
    settle()
    report("twin-20mm")
    check("tolerance twin: a 20 mm off-centre set-down counts as on_plate",
          bool(scene.on_plate()[0, mid]) and not bool(scene.success()[0]))
    place_ring(mid, px + 0.050, py, plate_top() + c.ring_thick / 2 + 0.020,
               flat_quat(0.2), settle_steps=10)
    settle()
    report("near-miss-50mm")
    ring_z = float(scene.rings[mid].data.root_pos_w[0, 2])
    check("near miss: a settled 50 mm off-centre set-down (still resting on the plate, "
          "overhanging) is NOT on_plate",
          not bool(scene.on_plate()[0, mid]) and bool(scene.settled()[0])
          and ring_z > plate_top())  # genuinely resting on the plate, not fallen off

    # =========================== 9. edge stand on the plate ===================================
    env.reset(seed=61)
    settle()
    _b, mid, _t = order_ids()
    px, py = plate_xy()
    r45 = math.sqrt(0.5)
    place_ring(mid, px, py, plate_top() + c.ring_outer / 2 + 0.005,
               (r45, r45, 0.0, 0.0), settle_steps=10)  # rolled 90 deg: standing on edge face
    settle()
    report("edge-stand")
    up_z = abs(float(scene._up_z(scene.rings[mid].data.root_quat_w)[0]))
    check("edge stand: the middle ring standing on its edge face on the plate is NOT served "
          "(flat gate), score 0",
          up_z < 0.5 and not bool(scene.served()[0]) and float(scene.score()[0]) <= 0.02
          and not bool(scene.success()[0]))

    # =========================== 10. beside the dowel is not threaded =========================
    env.reset(seed=71)
    settle()
    bot, mid, top = order_ids()
    drop_on_plate(mid)  # middle correctly served...
    sx, sy = spare_xy()
    place_ring(top, sx + 0.095, sy, c.surface_z + c.ring_thick / 2 + 0.003,
               flat_quat(0.1), settle_steps=20)
    place_ring(bot, sx - 0.095, sy, c.surface_z + c.ring_thick / 2 + 0.003,
               flat_quat(0.7), settle_steps=20)
    settle()
    report("beside-dowel")
    check("beside dowel: non-middle rings resting next to the spare stand are NOT threaded "
          "-> no success, score 0.30",
          not bool(scene.threaded_spare()[0].any()) and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.30) < 1e-3)

    # =========================== 11. wrong ring on the plate ==================================
    env.reset(seed=81)
    settle()
    bot, mid, top = order_ids()
    drop_thread_spare(mid)   # the middle ring wrongly parked on the spare dowel
    drop_thread_spare(bot)   # one legitimate non-middle threading
    drop_on_plate(top)       # the WRONG ring served
    report("wrong-ring")
    check("wrong ring: top ring on the plate (middle threaded on spare instead): no success, "
          "score 0.15 (only the legitimate non-middle threading latched)",
          not bool(scene.success()[0]) and not bool(scene.served()[0])
          and bool(scene.on_plate()[0, top])
          and abs(float(scene.score()[0]) - 0.15) < 1e-3)

    # =========================== 12. latched credit ==========================================
    env.reset(seed=91)
    settle()
    bot, mid, top = order_ids()
    drop_thread_spare(top)  # physical drop-threading (contact-guided descent)
    report("latch-fired")
    s_thread = float(scene.score()[0])
    ok_fired = bool(scene.threaded_spare()[0, top]) and abs(s_thread - 0.15) < 1e-3
    check("latched credit: a non-middle ring drop-threaded on the spare scores 0.15",
          ok_fired)
    place_ring(top, 0.30, 0.0, c.surface_z + c.ring_thick / 2 + 0.003,
               flat_quat(0.0), settle_steps=30)
    settle()
    report("latch-survives")
    check("latched credit survives removing the ring from the spare dowel (still 0.15)",
          not bool(scene.threaded_spare()[0, top])
          and abs(float(scene.score()[0]) - 0.15) < 1e-3)

    # =========================== 13. dumped pile on the plate =================================
    env.reset(seed=101)
    settle()
    bot, mid, top = order_ids()
    drop_on_plate(mid)
    drop_on_plate(top, dx=0.004)
    drop_on_plate(bot, dx=-0.004)
    settle()
    report("dumped-pile")
    check("dumped pile: all three rings piled on the plate -> no success (others must be "
          "on the spare dowel)",
          not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.30 + 1e-3)

    # =========================== 14. finite at the end ========================================
    finite = all(torch.isfinite(r.data.root_state_w).all() for r in scene.rings) \
        and bool(torch.isfinite(scene.plate.data.root_state_w).all())
    check("finite: all states finite at the end", finite)

    # =========================== save + verdict ==============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.spindle_serve")
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

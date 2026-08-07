"""Smoke / rubric-REJECTION battery for ServiceCarouselScene (sim_gen task
libero_kitchen_scene2_put_the_black_bowl_in_the_middle_on_the_plate_i78) — NullRobot,
teleported probe states, RECORDED.

This is NOT a solution (solve.py — torque-driven carousel + gravity bowl seating — is the
acceptance evidence that the rubric ACCEPTS a correct outcome). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partially-right) outcome as a settled state and
asserts the rubric's verdict on it. No probe below constructs full success.

  1. settle/no-NaN      — the authored layout settles finite; plate seated in its pocket
                          under the hutch, OFF the serve mark; rubric clean (score 0);
  2. determinism        — the same seed twice -> identical layout readback;
  3. randomization      — READBACK over 6 seeded resets: the middle-slot bowl identity
                          varies, the plate start azimuth varies, bowl positions jitter;
  4. null policy        — 240 idle steps -> score ~0, no success;
  5. seed strategy      — the seed's plan (walk up and set the bowl on the plate WHERE THE
                          PLATE IS): middle bowl seated on the plate at its START azimuth,
                          nothing rotated -> on_plate True, success False, score 0;
  6. azimuth near miss  — plate + middle bowl parked at 20 deg (outside serve_tol 15) ->
                          no delivery, no success; EMPTY plate parked at 10 deg -> delivered
                          True but success False (tolerance twin, never full success);
  7. wrong bowl         — an OUTER bowl delivered on the plate at the serve mark ->
                          on_plate for that bowl, success False;
  8. inverted bowl      — the middle bowl upside-down on the plate (load side) -> upright
                          gate rejects (not on_plate), LOADED never latches;
  9. gate honesty       — 25 mm off-centre set-down counts as on_plate (tolerance twin);
                          a settled 55 mm off-centre set-down (resting, bridging the pocket
                          wall) does NOT;
 10. latched credit     — the 0.60 fetch+load credit survives removing the bowl back to
                          the counter (latches are latches, success is not);
 11. velocity gate      — plate swept through the loading window on a FAST-spinning disc
                          (|w| > latch_disc_w) latches NOTHING; the latch fires only once
                          the disc is slow;
 12. out of pocket      — the plate lying on the counter at azimuth 0 (right direction,
                          not in the pocket) -> not delivered;
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

# Kit teardown hangs leave GPU zombies: global watchdog long before the forge timeout.
threading.Timer(1350.0, lambda: os._exit(4)).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.service_carousel")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    origin = env.iscene.env_origins
    z0 = c.surface_z
    hx, hy = c.hub_xy

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = origin[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.30, -1.30, 1.00)) + o),
                                tuple(np.array((0.08, 0.0, 0.26)) + o),
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
        print(f"[smoke] {tag:18s} | mid={int(scene.mid_idx[0])} "
              f"az={math.degrees(float(scene.plate_azimuth()[0])):+.1f}deg "
              f"in_pocket={bool(scene.plate_in_pocket()[0])} "
              f"on_plate={bool(scene.target_on_plate()[0])} "
              f"delivered={bool(scene.plate_delivered()[0])} "
              f"fetched={bool(scene.ever_fetched[0])} loaded={bool(scene.ever_loaded[0])} "
              f"settled={bool(scene.settled()[0])} score={float(scene.score()[0]):.3f} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    def flat_quat(yaw: float) -> tuple:
        return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

    def set_disc(yaw: float, w: float = 0.0) -> None:
        """Probe constructor: park (or spin) the carousel at a given yaw."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = hx, hy, z0 + c.disc_z_off
        st[:, 3] = math.cos(yaw / 2)
        st[:, 6] = math.sin(yaw / 2)
        st[:, 12] = w
        st[:, 0:3] += origin
        scene.disc.write_root_state_to_sim(st, all_ids)

    def seat_plate(yaw: float, settle_steps: int = 20, w: float = 0.0) -> None:
        """Probe constructor: disc at `yaw`, plate seated in its pocket there."""
        set_disc(yaw, w=w)
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = hx + c.pocket_mount_r * math.cos(yaw)
        st[:, 1] = hy + c.pocket_mount_r * math.sin(yaw)
        st[:, 2] = z0 + c.disc_z_off + c.disc_h / 2 + c.plate_h / 2 + 0.002
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        scene.plate.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def place_bowl(b: int, x: float, y: float, z: float, quat: tuple | None = None,
                   settle_steps: int = 10) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3:7] = torch.tensor(quat or (1.0, 0.0, 0.0, 0.0), device=device)
        st[:, 0:3] += origin
        scene.bowls[b].write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def plate_xy() -> tuple:
        p = scene.plate.data.root_pos_w[0] - origin[0]
        return float(p[0]), float(p[1])

    def plate_top() -> float:
        return float(scene.plate.data.root_pos_w[0, 2]) + c.plate_h / 2

    def seat_bowl(b: int, dx: float = 0.0, dy: float = 0.0, yaw: float = 0.2) -> None:
        px, py = plate_xy()
        place_bowl(b, px + dx, py + dy, plate_top() + c.bowl_floor_h / 2 + 0.006,
                   flat_quat(yaw), settle_steps=10)
        settle()

    def layout_readback() -> tuple:
        px, py = plate_xy()
        b0 = scene.bowls[0].data.root_pos_w[0] - origin[0]
        b1 = scene.bowls[1].data.root_pos_w[0] - origin[0]
        return (int(scene.mid_idx[0]), float(scene.yaw0[0]), px, py,
                float(b0[0]), float(b0[1]), float(b1[0]), float(b1[1]))

    # =========================== 1-2. settle / no-NaN / clean rubric =========================
    env.reset(seed=11)
    settle()
    report("reset")
    finite = all(torch.isfinite(b.data.root_state_w).all() for b in scene.bowls) \
        and bool(torch.isfinite(scene.plate.data.root_state_w).all()) \
        and bool(torch.isfinite(scene.disc.data.root_state_w).all())
    check("settle: authored layout finite + settled; plate seated in pocket, OFF the serve "
          "mark and OUTSIDE the loading window",
          finite and bool(scene.settled()[0]) and bool(scene.plate_in_pocket()[0])
          and not bool(scene.plate_delivered()[0])
          and not bool(scene.plate_in_load_window()[0]))
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
    print(f"[smoke] randomization readback (mid, yaw0, plate_xy, bowl0_xy, bowl1_xy):\n"
          f"{arr}", flush=True)
    check("randomization: the middle-slot bowl identity varies across seeds (readback)",
          len({int(v) for v in arr[:, 0]}) >= 2)
    check("randomization: the plate start azimuth varies across seeds (readback)",
          math.degrees(arr[:, 1].max() - arr[:, 1].min()) > 2.0)
    check("randomization: bowl positions vary across seeds (permutation + jitter readback)",
          (arr[:, 4].max() - arr[:, 4].min()) > 0.004
          or (arr[:, 5].max() - arr[:, 5].min()) > 0.004)

    # =========================== 5. null policy fails ========================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    check("null policy: score ~0 and no success after 240 idle steps",
          float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 6. seed strategy control =====================================
    # The seed's whole plan: set the middle bowl on the plate WHERE THE PLATE IS. Here the
    # plate starts parked under the hutch, off the mark — nothing was rotated.
    env.reset(seed=41)
    settle()
    mid = int(scene.mid_idx[0])
    seat_bowl(mid)
    report("seed-strategy")
    check("seed strategy (middle bowl seated on the un-fetched plate at its start azimuth): "
          "on_plate True but no success, score 0 (latches need the loading window)",
          bool(scene.target_on_plate()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.02)

    # =========================== 7. azimuth near miss + empty twin ============================
    env.reset(seed=51)
    settle()
    mid = int(scene.mid_idx[0])
    seat_plate(math.radians(20.0))
    seat_bowl(mid)
    report("az-20deg")
    check("azimuth near miss: loaded plate parked at 20 deg (serve_tol 15) -> not delivered, "
          "no success",
          bool(scene.target_on_plate()[0]) and not bool(scene.plate_delivered()[0])
          and not bool(scene.success()[0]))
    env.reset(seed=55)
    settle()
    seat_plate(math.radians(10.0))
    settle()
    report("az-10deg-empty")
    check("tolerance twin: EMPTY plate parked at 10 deg IS delivered — but no bowl, "
          "no success",
          bool(scene.plate_delivered()[0]) and not bool(scene.success()[0]))

    # =========================== 8. wrong bowl delivered ======================================
    env.reset(seed=61)
    settle()
    mid = int(scene.mid_idx[0])
    wrong = (mid + 1) % c.n_bowls
    seat_plate(0.0)
    seat_bowl(wrong)
    report("wrong-bowl")
    check("wrong bowl: an OUTER bowl delivered on the plate at the serve mark -> that bowl "
          "is on_plate, success False",
          bool(scene.bowls_on_plate()[0, wrong]) and not bool(scene.target_on_plate()[0])
          and bool(scene.plate_delivered()[0]) and not bool(scene.success()[0]))

    # =========================== 9. inverted bowl on the load side ============================
    env.reset(seed=71)
    settle()
    mid = int(scene.mid_idx[0])
    seat_plate(math.pi)
    px, py = plate_xy()
    place_bowl(mid, px, py,
               plate_top() + c.bowl_floor_h + c.bowl_wall_h + 0.004,
               (0.0, 1.0, 0.0, 0.0), settle_steps=10)  # upside-down
    settle()
    report("inverted")
    check("inverted bowl: middle bowl upside-down on the plate -> upright gate rejects "
          "(not on_plate), LOADED never latches",
          not bool(scene.target_on_plate()[0]) and not bool(scene.ever_loaded[0])
          and not bool(scene.success()[0]))

    # =========================== 10. gate honesty: twin + near miss ===========================
    # (still at seed 71, plate parked on the loading side)
    seat_bowl(mid, dx=0.025)
    report("twin-25mm")
    check("tolerance twin: a 25 mm off-centre set-down counts as on_plate",
          bool(scene.target_on_plate()[0]) and not bool(scene.success()[0]))
    s_loaded = float(scene.score()[0])
    seat_bowl(mid, dx=0.055)
    report("near-miss-55mm")
    bowl_z = float(scene.bowls[mid].data.root_pos_w[0, 2])
    check("near miss: a settled 55 mm off-centre set-down (resting, bridging the pocket "
          "wall) is NOT on_plate",
          not bool(scene.target_on_plate()[0]) and bool(scene.settled()[0])
          and bowl_z > plate_top() - 0.006)  # genuinely resting at plate level, not fallen

    # =========================== 11. latched credit survives ==================================
    check("latched credit: fetch+load latched 0.60 while the bowl sat on the plate in the "
          "loading window", abs(s_loaded - 0.60) < 1e-3)
    place_bowl(mid, -0.50, -0.40, z0 + c.bowl_floor_h / 2 + 0.003,
               flat_quat(0.0), settle_steps=20)
    settle()
    report("latch-survives")
    check("latched credit survives removing the bowl back to the counter (still 0.60, "
          "no success)",
          abs(float(scene.score()[0]) - 0.60) < 1e-3 and not bool(scene.success()[0])
          and not bool(scene.target_on_plate()[0]))

    # =========================== 12. velocity gate on the FETCH latch =========================
    env.reset(seed=81)
    settle()
    seat_plate(math.pi, settle_steps=0, w=2.0)  # swept through the window on a FAST disc
    step(8)
    report("fast-sweep")
    fast_blocked = not bool(scene.ever_fetched[0])
    check("velocity gate: the plate riding a fast-spinning disc (|w|=2.0 > latch 0.5) "
          "through the loading window latches NOTHING", fast_blocked)
    seat_plate(math.pi, settle_steps=20)  # same pose, disc parked
    settle()
    report("slow-park")
    check("velocity gate: the same pose with the disc parked DOES latch the fetch (0.25)",
          bool(scene.ever_fetched[0]) and abs(float(scene.score()[0]) - 0.25) < 1e-3)

    # =========================== 13. right direction, out of the pocket =======================
    env.reset(seed=91)
    settle()
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1] = hx + 0.32, hy  # azimuth 0 from the hub, but flat on the counter
    st[:, 2] = z0 + c.plate_h / 2 + 0.002
    st[:, 3] = 1.0
    st[:, 0:3] += origin
    scene.plate.write_root_state_to_sim(st, all_ids)
    step(20)
    settle()
    report("out-of-pocket")
    check("out of pocket: the plate lying on the counter at azimuth 0 is NOT delivered "
          "(pocket radial/height gates)",
          not bool(scene.plate_in_pocket()[0]) and not bool(scene.plate_delivered()[0])
          and not bool(scene.success()[0]))

    # =========================== 14. finite at the end ========================================
    finite = all(torch.isfinite(b.data.root_state_w).all() for b in scene.bowls) \
        and bool(torch.isfinite(scene.plate.data.root_state_w).all()) \
        and bool(torch.isfinite(scene.disc.data.root_state_w).all())
    check("finite: all states finite at the end", finite)

    # =========================== save + verdict ==============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.service_carousel")
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

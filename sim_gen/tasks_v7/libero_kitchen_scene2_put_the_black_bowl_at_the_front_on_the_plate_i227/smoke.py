"""Smoke / rubric-REJECTION battery for ShutterVaultScene (sim_gen task
libero_kitchen_scene2_put_the_black_bowl_at_the_front_on_the_plate_i227) — NullRobot,
teleported probe states, RECORDED.

This is NOT a solution (solve.py — force-driven tile slides + gravity bowl seating — is the
acceptance evidence that the rubric ACCEPTS a correct outcome). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partially-right) outcome as a settled state and
asserts the rubric's verdict on it. No probe below constructs full success.

  1-2. settle/no-NaN     — the authored layout settles finite; RED tile covers the well,
                           BLUE tile in the middle cell, plate seated at the well bottom;
                           rubric clean (score 0, no success);
  3.  determinism        — the same seed twice -> identical layout readback;
  4-6. randomization     — READBACK over 6 seeded resets: the front-slot bowl identity
                           varies, the vault pose (xy + yaw) varies, bowl positions vary;
  7.  null policy        — 240 idle steps -> score ~0, no success;
  8.  seed strategy      — the seed's plan (carry the front bowl straight to where the
                           plate is): the bowl set down over the well ON THE CLOSED SHUTTER
                           (beside the knob, the closest physically constructible state) ->
                           rests 82 mm above the well floor, SERVED never latches, score 0;
  9-10. interlock        — the SAME calibrated push recipe on both tiles: RED pushed toward
                           B while BLUE occupies B barely moves (blocked by the tile, not by
                           a vacuous probe — the twin proves the recipe moves a free tile
                           its full 150 mm and fires the UNLOCK latch);
  11. latch persistence  — a constructed OPEN state latches 0.50; closing the tiles back
                           over the empty well keeps the latched 0.50 (latches are latches,
                           success is not);
  12. wrong bowl         — a REAR bowl dropped through the open aperture onto the plate ->
                           on the plate, yes — success False, score stays 0.50;
  13. inverted bowl      — the FRONT bowl upside-down in the open well -> upright gate
                           rejects (not on_plate), no success;
  14. plate extracted    — the literal seed goal geometry (plate on the open counter, front
                           bowl seated on it) -> front_on_plate True but plate_in_well
                           False, success False, score 0;
  15-16. velocity gate   — BLUE written INTO cell C moving fast latches nothing on the
                           fly-through frame; once it stops there it does latch (twin);
  17. finite at the end.

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
from isaaclab.utils.math import quat_apply_inverse

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Kit teardown hangs leave GPU zombies: global watchdog long before the forge timeout.
threading.Timer(1350.0, lambda: os._exit(4)).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.shutter_vault")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    origin = env.iscene.env_origins
    z0 = c.surface_z
    P = c.pitch

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = origin[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.35, -1.10, 0.95)) + o),
                                tuple(np.array((0.30, 0.05, 0.25)) + o),
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

    def tile_loc(t: int) -> torch.Tensor:
        return scene._vault_local(scene.tiles[t].data.root_pos_w)[0]

    def report(tag: str) -> None:
        l0, l1 = tile_loc(0), tile_loc(1)
        print(f"[smoke] {tag:18s} | front={int(scene.front_idx[0])} "
              f"red=({float(l0[0]):+.3f},{float(l0[1]):+.3f}) "
              f"blue=({float(l1[0]):+.3f},{float(l1[1]):+.3f}) "
              f"unlocked={bool(scene.ever_unlocked[0])} opened={bool(scene.ever_opened[0])} "
              f"served={bool(scene.ever_served[0])} "
              f"on_plate={bool(scene.front_on_plate()[0])} "
              f"in_well={bool(scene.plate_in_well()[0])} "
              f"settled={bool(scene.settled()[0])} score={float(scene.score()[0]):.3f} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    def flat_quat(yaw: float) -> tuple:
        return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

    def box_world(lx: float, ly: float) -> tuple:
        """vault-local xy -> env-local world xy (via the vault pose readback)."""
        bx = float(scene.box_pos_xy[0, 0])
        by = float(scene.box_pos_xy[0, 1])
        psi = float(scene.box_yaw[0])
        cy, sy = math.cos(psi), math.sin(psi)
        return bx + cy * lx - sy * ly, by + sy * lx + cy * ly

    def place_tile(t: int, lx: float, ly: float, vel_local: tuple = (0.0, 0.0),
                   settle_steps: int = 0) -> None:
        """Probe constructor: park (or launch) a tile at a vault-local cell position."""
        wx, wy = box_world(lx, ly)
        psi = float(scene.box_yaw[0])
        cy, sy = math.cos(psi), math.sin(psi)
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1] = wx, wy
        st[:, 2] = z0 + c.deck_top + c.tile_t / 2 + 0.0005
        st[:, 3:7] = torch.tensor(flat_quat(psi), device=device)
        st[:, 7] = cy * vel_local[0] - sy * vel_local[1]
        st[:, 8] = sy * vel_local[0] + cy * vel_local[1]
        st[:, 0:3] += origin
        scene.tiles[t].write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def place_bowl(b: int, x: float, y: float, z: float, quat: tuple | None = None,
                   settle_steps: int = 10) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3:7] = torch.tensor(quat or (1.0, 0.0, 0.0, 0.0), device=device)
        st[:, 0:3] += origin
        scene.bowls[b].write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def plate_top() -> float:
        return float(scene.plate.data.root_pos_w[0, 2]) + c.plate_h / 2

    def open_shutters(settle_steps: int = 40) -> None:
        """Probe constructor: BLUE parked in C, RED parked in B (the open state)."""
        place_tile(1, P, P)
        place_tile(0, P, 0.0)
        step(settle_steps)

    def close_shutters(settle_steps: int = 40) -> None:
        place_tile(1, P, 0.0)
        place_tile(0, 0.0, 0.0)
        step(settle_steps)

    # Frame quirk (pod-dependent): the default set_external_force_and_torque call may apply
    # the wrench in the BODY frame — a yaw-psi tile pushed with a raw world vector then
    # moves along 2*psi and wedge-jams (observed on the solve, seed 7). Default to the
    # body-frame pre-encode; a calibration probe before the interlock checks toggles it
    # if the pod turns out to apply world-frame directly.
    force_mode = ["body"]

    def push_tile(t: int, dir_local: tuple, force: float = 3.0, steps: int = 600,
                  v_max: float = 0.08) -> float:
        """Calibrated capped-speed push (the solve's drive recipe). Returns the tile's
        displacement ALONG the push direction (vault frame). Identical recipe for the
        blocked probe and its moving twin — anti-vacuity by construction."""
        psi = float(scene.box_yaw[0])
        cy, sy = math.cos(psi), math.sin(psi)
        dw = (cy * dir_local[0] - sy * dir_local[1],
              sy * dir_local[0] + cy * dir_local[1])
        start = tile_loc(t)[:2].clone()
        zero3 = torch.zeros(n, 1, 3, device=device)
        for _ in range(steps):
            v_w = scene.tiles[t].data.root_lin_vel_w[0]
            v_along = float(v_w[0] * dw[0] + v_w[1] * dw[1])
            f = torch.zeros(n, 1, 3, device=device)
            if v_along < v_max:
                f[:, 0, 0], f[:, 0, 1] = force * dw[0], force * dw[1]
                if force_mode[0] == "body":
                    q = scene.tiles[t].data.root_quat_w
                    f = quat_apply_inverse(q, f.squeeze(1)).unsqueeze(1)
            scene.tiles[t].set_external_force_and_torque(f, zero3)
            step(1)
        scene.tiles[t].set_external_force_and_torque(zero3, zero3)
        step(30)
        d = tile_loc(t)[:2] - start
        return float(d[0] * dir_local[0] + d[1] * dir_local[1])

    def layout_readback() -> tuple:
        b0 = scene.bowls[0].data.root_pos_w[0] - origin[0]
        b1 = scene.bowls[1].data.root_pos_w[0] - origin[0]
        return (int(scene.front_idx[0]), float(scene.box_pos_xy[0, 0]),
                float(scene.box_pos_xy[0, 1]), float(scene.box_yaw[0]),
                float(b0[0]), float(b0[1]), float(b1[0]), float(b1[1]))

    # =========================== 1-2. settle / no-NaN / clean rubric =========================
    env.reset(seed=11)
    settle()
    report("reset")
    finite = all(torch.isfinite(b.data.root_state_w).all() for b in scene.bowls) \
        and bool(torch.isfinite(scene.plate.data.root_state_w).all()) \
        and all(torch.isfinite(t.data.root_state_w).all() for t in scene.tiles)
    l0, l1 = tile_loc(0), tile_loc(1)
    check("settle: authored layout finite + settled; RED covers the well (cell A), BLUE in "
          "the middle cell (B), plate seated at the well bottom",
          finite and bool(scene.settled()[0])
          and float(l0[:2].norm()) < 0.02
          and float((l1[:2] - torch.tensor([P, 0.0], device=device)).norm()) < 0.02
          and bool(scene.plate_in_well()[0]))
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

    # =========================== 4-6. randomization is real ==================================
    reads = []
    for s in (21, 22, 23, 24, 25, 26):
        env.reset(seed=s)
        step(3)
        reads.append(layout_readback())
    arr = np.array(reads)
    print(f"[smoke] randomization readback (front, box_xy, box_yaw, bowl0_xy, bowl1_xy):\n"
          f"{arr}", flush=True)
    check("randomization: the front-slot bowl identity varies across seeds (readback)",
          len({int(v) for v in arr[:, 0]}) >= 2)
    check("randomization: the vault pose varies across seeds (xy and yaw readback)",
          (arr[:, 1].max() - arr[:, 1].min()) > 0.005
          and math.degrees(arr[:, 3].max() - arr[:, 3].min()) > 4.0)
    check("randomization: bowl positions vary across seeds (permutation + jitter readback)",
          (arr[:, 4].max() - arr[:, 4].min()) > 0.004
          or (arr[:, 5].max() - arr[:, 5].min()) > 0.004)

    # =========================== 7. null policy fails ========================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    check("null policy: score ~0 and no success after 240 idle steps",
          float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 8. seed strategy control =====================================
    # The seed's whole plan: carry the front bowl straight to where the plate is. The plate
    # is at the bottom of the covered well — the closest constructible state is the bowl on
    # the CLOSED shutter, beside its knob.
    env.reset(seed=41)
    settle()
    front = int(scene.front_idx[0])
    wx, wy = box_world(0.057, 0.0)  # beside the knob, on the RED flange over the well
    place_bowl(front, wx, wy, z0 + c.deck_top + c.tile_t + c.bowl_floor_h / 2 + 0.004,
               flat_quat(0.3))
    settle()
    report("seed-strategy")
    locf = scene._vault_local(scene.bowls[front].data.root_pos_w)[0]
    check("seed strategy (front bowl set down over the well on the CLOSED shutter): rests "
          "~82 mm up on the flange, SERVED never latches, score 0, no success",
          float(locf[2]) > 0.070 and not bool(scene.ever_served[0])
          and float(scene.score()[0]) <= 0.005 and not bool(scene.success()[0]))

    # =========================== 9-10. interlock (physics, calibrated pair) ==================
    # Force-encoding calibration: drive the FREE BLUE tile with the default (body-frame
    # pre-encode) recipe; if it fails to travel, the pod applies wrenches world-frame —
    # toggle and re-verify from a fresh reset. Keeps the actual checks non-vacuous
    # regardless of pod frame convention.
    env.reset(seed=45)
    settle()
    d_cal = push_tile(1, (0.0, 1.0), steps=400)
    if d_cal < 0.050:
        force_mode[0] = "world"
        print(f"[smoke] calibration: body-encoded push moved BLUE only "
              f"{d_cal * 1000:.1f} mm -> toggling force encoding to 'world'", flush=True)
    else:
        print(f"[smoke] calibration: body-encoded push moved BLUE {d_cal * 1000:.1f} mm "
              f"-> keeping 'body' encoding", flush=True)
    env.reset(seed=45)
    settle()
    d_red = push_tile(0, (1.0, 0.0))  # RED toward B — BLUE is in the way
    report("red-blocked")
    red_blocked = 0.001 < d_red < 0.030  # it presses the slack out, then BLUE blocks it
    check(f"interlock: RED pushed toward the middle cell while BLUE occupies it advances "
          f"only the slack (moved {d_red * 1000:.1f} mm < 30 mm)", red_blocked)
    d_blue = push_tile(1, (0.0, 1.0))  # BLUE toward C — free, same recipe
    report("blue-free-twin")
    check(f"interlock twin: the SAME push recipe moves the unobstructed BLUE tile its full "
          f"travel into C (moved {d_blue * 1000:.0f} mm > 100 mm) and fires UNLOCK",
          d_blue > 0.100 and bool(scene.ever_unlocked[0])
          and abs(float(scene.score()[0]) - 0.25) < 1e-3)

    # =========================== 11. latch persistence ========================================
    env.reset(seed=51)
    settle()
    front = int(scene.front_idx[0])
    open_shutters()
    settle()
    report("constructed-open")
    s_open = float(scene.score()[0])
    close_shutters()
    settle()
    report("closed-back")
    check("latched credit: a constructed OPEN state latches 0.50, and closing the tiles "
          "back over the empty well keeps the latched 0.50 (no success either way)",
          abs(s_open - 0.50) < 1e-3 and abs(float(scene.score()[0]) - 0.50) < 1e-3
          and not bool(scene.success()[0]))

    # =========================== 12. wrong bowl through the open aperture =====================
    open_shutters()
    settle()
    wrong = (front + 1) % c.n_bowls
    wx, wy = box_world(0.0, 0.0)
    place_bowl(wrong, wx, wy, plate_top() + c.bowl_floor_h / 2 + 0.006, flat_quat(0.2))
    settle()
    report("wrong-bowl")
    check("wrong bowl: a REAR bowl dropped through the open aperture onto the plate is on "
          "the plate — success False, score stays 0.50 (SERVED keys the FRONT bowl)",
          bool(scene.bowls_on_plate()[0, wrong]) and not bool(scene.front_on_plate()[0])
          and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.50) < 1e-3)

    # =========================== 13. inverted front bowl ======================================
    place_bowl(wrong, 0.62, -0.35, z0 + c.bowl_floor_h / 2 + 0.003, flat_quat(0.0),
               settle_steps=20)  # clear the well first (probe debris rule)
    place_bowl(front, wx, wy,
               plate_top() + c.bowl_wall_h + c.bowl_floor_h / 2 + 0.004,
               (0.0, 1.0, 0.0, 0.0), settle_steps=10)  # upside-down over the plate
    settle()
    report("inverted")
    check("inverted bowl: the FRONT bowl upside-down in the open well -> upright gate "
          "rejects (not on_plate), no success",
          not bool(scene.front_on_plate()[0]) and not bool(scene.success()[0]))

    # =========================== 14. plate extracted (literal seed goal) ======================
    env.reset(seed=61)
    settle()
    front = int(scene.front_idx[0])
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1] = 0.62, -0.35  # open counter, away from the vault and the row
    st[:, 2] = z0 + c.plate_h / 2 + 0.002
    st[:, 3] = 1.0
    st[:, 0:3] += origin
    scene.plate.write_root_state_to_sim(st, all_ids)
    step(20)
    place_bowl(front, 0.62, -0.35, plate_top() + c.bowl_floor_h / 2 + 0.006, flat_quat(0.1))
    settle()
    report("plate-extracted")
    check("plate extracted: the literal seed goal geometry (plate on the counter, front "
          "bowl on it) -> front_on_plate True but plate_in_well False, success False, "
          "score 0",
          bool(scene.front_on_plate()[0]) and not bool(scene.plate_in_well()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.005)

    # =========================== 15-16. velocity gate on the tile latches =====================
    env.reset(seed=71)
    settle()
    place_tile(1, P, P, vel_local=(0.0, 0.40))  # written INTO C, moving fast at the wall
    step(1)
    fast_blocked = not bool(scene.ever_unlocked[0])
    report("fast-flythrough")
    check("velocity gate: BLUE written into cell C moving fast (0.40 m/s > latch 0.05) "
          "latches NOTHING on the fly-through frame", fast_blocked)
    settle()
    report("slow-park")
    check("velocity gate twin: once the tile comes to rest in C it DOES latch (0.25)",
          bool(scene.ever_unlocked[0]) and abs(float(scene.score()[0]) - 0.25) < 1e-3)

    # =========================== 17. finite at the end ========================================
    finite = all(torch.isfinite(b.data.root_state_w).all() for b in scene.bowls) \
        and bool(torch.isfinite(scene.plate.data.root_state_w).all()) \
        and all(torch.isfinite(t.data.root_state_w).all() for t in scene.tiles)
    check("finite: all states finite at the end", finite)

    # =========================== save + verdict ==============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.shutter_vault")
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

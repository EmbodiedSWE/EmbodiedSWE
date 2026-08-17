"""Smoke / rubric-REJECTION battery for LeaningChainScene (sim_gen task
`screw_nail_i309`) — NullRobot, teleported probe states + force probes, RECORDED.

This is NOT a solution (solve.py — topple blue onto the anvil, green onto blue, red
onto green, all through contact dynamics — is the acceptance evidence that the rubric
ACCEPTS a correct outcome; it passes on seeds 0/1). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled state and
asserts the rubric REJECTS it. No probe in this battery ever reaches success(), and a
final audit asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: three tiles STANDING beside
                            the lane; score ~0 at rest, no success;
  3-4. randomization      — READBACK over 6 seeded resets: lane position + yaw, the
                            color->slot permutation and per-tile spawn poses vary;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed-strategy       — the seed family's move (press down and TWIST, the
                            screwdriver motion) applied to a tile achieves nothing:
                            no seat, score ~0;
  7.  collapsed cascade   — all three tiles lying flat/shingled in the lane in correct
                            color order (what a toppled-through run leaves) -> every
                            tile below the tilt band + head gate, no credit, no success;
  8.  standing in lane    — all three tiles standing VERTICAL in the lane in order ->
                            the tilt band rejects them, no success;
  9.  rail lean           — a tile dropped leaning across the low guide rail settles
                            shallow and cross-lane -> rejected (tilt / axis / head
                            gates), no success;
  10. wrong color order   — a REAL leaning chain built by hover-drop but with GREEN
                            on the anvil and BLUE mid-chain -> the identity + order
                            predicates reject every tile, no credit, no success;
  11. skip-chain          — blue correctly seated on the anvil (partial credit is
                            honest) but RED leaning directly on blue with green far
                            away -> red is rejected (its head must reach past GREEN),
                            no success;
  12. settle gate         — a tile in a perfect in-band pose but still MOVING is not
                            seated (velocity gates are real); removed before rest;
  13. latched credit      — knocking the seated blue tile away afterwards leaves the
                            latched stage credit unchanged while seated_now() drops;
  14. rejection audit     — success() was never True at ANY judged point;
  15. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.screw_nail_i309.smoke --headless
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

TILE_IDX = {"red": 0, "green": 1, "blue": 2}


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.leaning_chain")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(3, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.30, -1.00, 0.85)) + o),
                                tuple(np.array((0.45, 0.00, 0.05)) + o),
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
        s, ok = judge()
        g = scene.tile_geometry()
        seat = scene.seated_now(g)[0]
        parts = []
        for nm, k in TILE_IDX.items():
            th = math.degrees(math.asin(max(-1.0, min(1.0, float(g["axis"][0, k, 2])))))
            parts.append(f"{nm[0]}:x={float(g['foot'][0, k, 0]):+.3f} t={th:4.0f} "
                         f"hz={float(g['head'][0, k, 2]):.3f} s={int(bool(seat[k]))}")
        print(f"[smoke] {tag:16s} | " + " | ".join(parts)
              + f" | score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    # ---- lane frame helpers ---------------------------------------------------------------
    def lane_pose() -> tuple[torch.Tensor, float]:
        lp = (scene.lane.data.root_pos_w - scene.env_origins)[0]
        q = scene.lane.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return lp, yaw

    def lane_to_world(loc) -> torch.Tensor:
        lp, yaw = lane_pose()
        cy, sy = math.cos(yaw), math.sin(yaw)
        return torch.tensor([float(lp[0]) + cy * loc[0] - sy * loc[1],
                             float(lp[1]) + sy * loc[0] + cy * loc[1],
                             loc[2]], device=device)

    def place(body, xyz_world, quat=None, vel=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = xyz_world
        if quat is None:
            st[:, 3] = 1.0
        else:
            st[:, 3:7] = quat
        if vel is not None:
            st[:, 7:10] = torch.tensor(vel, device=device)
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def tile_quat(theta_deg: float, cross: bool = False) -> torch.Tensor:
        """World quat for a tile leaning `theta_deg` above horizontal toward lane +x
        (or toward +y when `cross`), wide face toward the support."""
        from isaaclab.utils.math import quat_from_angle_axis, quat_mul

        _lp, yaw = lane_pose()
        psi = yaw - math.pi / 2
        q0 = torch.tensor([math.cos(psi / 2), 0.0, 0.0, math.sin(psi / 2)],
                          device=device).view(1, 4)
        beta = math.radians(90.0 - theta_deg)
        if cross:
            axis = torch.tensor([[math.cos(yaw), math.sin(yaw), 0.0]], device=device)
            qr = quat_from_angle_axis(torch.tensor([-beta], device=device), axis)
        else:
            axis = torch.tensor([[-math.sin(yaw), math.cos(yaw), 0.0]], device=device)
            qr = quat_from_angle_axis(torch.tensor([beta], device=device), axis)
        return quat_mul(qr, q0)[0]

    def place_lean(name: str, foot_x: float, foot_y: float, theta_deg: float,
                   cross: bool = False, drop: float = 0.010,
                   settle_steps: int = 90) -> None:
        """Hover-drop a tile in a leaning pose slightly above contact; gravity +
        contact make (or refuse) the rest — probes never spawn load-bearing."""
        th = math.radians(theta_deg)
        dx, dy = (0.0, math.cos(th)) if cross else (math.cos(th), 0.0)
        loc = (foot_x + dx * c.tile_len / 2, foot_y + dy * c.tile_len / 2,
               c.tile_len / 2 * math.sin(th) + drop)
        q = tile_quat(theta_deg, cross=cross)
        place(scene.tiles[name], lane_to_world(loc), quat=q, settle_steps=settle_steps)

    def place_flat(name: str, foot_x: float, foot_y: float = 0.0,
                   settle_steps: int = 45) -> None:
        place_lean(name, foot_x, foot_y, 1.0, drop=0.004, settle_steps=settle_steps)

    def park(name: str, spot=(0.95, 0.6), settle_steps: int = 30) -> None:
        place(scene.tiles[name],
              torch.tensor([spot[0], spot[1], c.tile_len / 2 + 0.002], device=device),
              settle_steps=settle_steps)

    def wrench(body, f3: torch.Tensor, t3: torch.Tensor) -> None:
        """World wrench expressed in the body's current link frame (the house
        convention: is_global=True silently drops the torque on this stack)."""
        from isaaclab.utils.math import quat_apply_inverse

        q = body.data.root_link_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f3.view(1, 3).expand(n, 3)).unsqueeze(1),
            quat_apply_inverse(q, t3.view(1, 3).expand(n, 3)).unsqueeze(1),
            env_ids=all_ids)

    def layout_sane(tag: str) -> bool:
        """Reset honesty: every tile standing upright (axis_z ~ 1), outside the lane
        corridor, feet on the floor."""
        g = scene.tile_geometry()
        ok = True
        for k in range(3):
            standing = float(g["axis"][0, k, 2]) > 0.95
            off_lane = abs(float(g["foot"][0, k, 1])) > c.rail_y + 0.02
            ok = ok and standing and off_lane
        if not ok:
            print(f"[smoke] layout sanity VIOLATION at {tag}", flush=True)
        return ok

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    bodies = tuple(scene.tiles.values()) + (scene.lane,)
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("settle: all states finite; three tiles standing upright beside the lane",
          fin and layout_sane("reset"))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    sane = True
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(30)
        sane = sane and layout_sane(f"seed {sd}")
        lp, yaw = lane_pose()
        g = scene.tile_geometry()
        # nearest spawn slot per tile (lane frame) -> the permutation signature
        slots = np.array(c.spawn_slots)
        sig = []
        for k in range(3):
            f = np.array([float(g["foot"][0, k, 0]), float(g["foot"][0, k, 1])])
            sig.append(int(np.argmin(((slots - f) ** 2).sum(axis=1))))
        reads.append((float(lp[0]), float(lp[1]), yaw,
                      float(g["foot"][0, 0, 0]), float(g["foot"][0, 0, 1]),
                      float(g["foot"][0, 1, 0]), float(g["foot"][0, 1, 1]),
                      sig[0], sig[1], sig[2]))
    arr = np.array(reads)
    print("[smoke] randomization readback (lane_x, lane_y, lane_yaw, red_x, red_y, "
          f"green_x, green_y, slot_r, slot_g, slot_b):\n{np.round(arr, 4)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: lane position and yaw vary "
          f"(readback spreads pos=({spread[0]:.3f},{spread[1]:.3f}) "
          f"yaw={spread[2]:.2f} rad)",
          (spread[0] > 0.015 or spread[1] > 0.015) and spread[2] > 0.15)
    perms = {tuple(int(v) for v in row[7:10]) for row in arr}
    check("randomization: the color->slot permutation and tile spawn poses vary "
          f"(readback: {len(perms)} distinct permutations, tile spread "
          f"({spread[3]:.3f},{spread[4]:.3f})), and every reset spawns standing "
          "off-lane",
          len(perms) >= 2 and (spread[3] > 0.02 or spread[4] > 0.02) and sane)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. the seed family's move achieves nothing =================
    # rlbench/screw_nail's verb is press-down-and-TWIST (drive a fastener by rotation).
    # Apply exactly that to the blue tile: 3 N down + 0.05 N m about the vertical axis.
    env.reset(seed=41)
    step(30)
    f3 = torch.zeros(3, device=device)
    t3 = torch.zeros(3, device=device)
    f3[2], t3[2] = -3.0, 0.05
    blue = scene.tiles["blue"]
    for _ in range(240):
        wrench(blue, f3, t3)
        step(1)
    wrench(blue, zero3, zero3)
    step(60)
    report("press+twist")
    s, ok = judge()
    seat = scene.seated_now()[0]
    check("seed-strategy analog: pressing down and twisting a tile (the screwdriver "
          "motion) achieves nothing — no tile seated, score ~0",
          not bool(seat.any()) and s <= 0.02 and not ok)

    # =========================== 7. collapsed cascade (all flat, right order) ===============
    env.reset(seed=51)
    step(30)
    # The open lane (post at -rail_len/2-0.010 to anvil face) is 0.36 m — exactly three
    # tile lengths — so a real collapsed run leaves SHINGLED tiles: each head resting a
    # couple of centimetres onto the next tile's tail, propped only by one tile
    # thickness (~8 deg). Hover-drop each tile already in that pose (dropping one onto
    # a flat neighbour kicks the pile around), and assert the meaningful thing
    # directly: every tilt is BELOW the rubric band and every head below the head gate.
    place_flat("blue", c.anvil_face - 0.130)                       # foot +0.020
    place_lean("green", c.anvil_face - 0.235, 0.0, 8.0, drop=0.006)  # head on blue tail
    place_lean("red", c.anvil_face - 0.335, 0.0, 8.0, drop=0.006,
               settle_steps=120)                                   # head on green tail
    report("all-flat")
    g = scene.tile_geometry()
    sub_band = bool((g["axis"][0, :, 2] < math.sin(math.radians(c.tilt_min_deg))).all())
    heads_low = bool((g["head"][0, :, 2] < c.head_z_min).all())
    s, ok = judge()
    seat = scene.seated_now(g)[0]
    check("collapsed cascade: three tiles lying flat/shingled in the lane in correct "
          "color order, all below the tilt band and head gate — no seat, no credit, "
          "no success",
          sub_band and heads_low and not bool(seat.any()) and s <= 0.02 and not ok)

    # =========================== 8. standing in the lane ====================================
    env.reset(seed=61)
    step(30)
    for nm, fx in (("blue", 0.10), ("green", 0.00), ("red", -0.10)):
        _lp, yaw = lane_pose()
        psi = yaw - math.pi / 2
        q = torch.tensor([math.cos(psi / 2), 0.0, 0.0, math.sin(psi / 2)], device=device)
        place(scene.tiles[nm], lane_to_world((fx, 0.0, c.tile_len / 2 + 0.002)),
              quat=q, settle_steps=30)
    step(60)
    report("standing-in-lane")
    g = scene.tile_geometry()
    upright = bool((g["axis"][0, :, 2] > 0.95).all())
    s, ok = judge()
    seat = scene.seated_now(g)[0]
    check("standing in lane: three tiles standing VERTICAL in the lane in order are "
          "rejected by the tilt band — no seat, no credit, no success",
          upright and not bool(seat.any()) and s <= 0.02 and not ok)

    # =========================== 9. rail lean ===============================================
    env.reset(seed=71)
    step(30)
    place_lean("green", 0.00, c.rail_y - c.tile_len + 0.035, 12.0, cross=True,
               drop=0.006, settle_steps=120)
    report("rail-lean")
    g = scene.tile_geometry()
    k = TILE_IDX["green"]
    s, ok = judge()
    seat = scene.seated_now(g)[0]
    check("rail lean: a tile dropped leaning across the low guide rail settles "
          f"shallow/cross-lane (tilt={math.degrees(math.asin(max(-1, min(1, float(g['axis'][0, k, 2]))))):.0f} deg, "
          f"head_z={float(g['head'][0, k, 2]):.3f}) -> not seated, no success",
          not bool(seat[k]) and not ok)

    # =========================== 10. wrong color order ======================================
    env.reset(seed=81)
    step(30)
    # a REAL leaning chain, but GREEN seated on the anvil and BLUE mid-chain
    place_lean("green", c.anvil_face - 0.042, 0.0, 52.0, settle_steps=90)
    gg = scene.tile_geometry()
    gfx = float(gg["foot"][0, TILE_IDX["green"], 0])
    place_lean("blue", gfx - 0.052, 0.0, 31.0, settle_steps=90)
    gg = scene.tile_geometry()
    bfx = float(gg["foot"][0, TILE_IDX["blue"], 0])
    place_lean("red", bfx - 0.048, 0.0, 20.0, settle_steps=120)
    report("wrong-order")
    g = scene.tile_geometry()
    leaning = int(((g["axis"][0, :, 2] > math.sin(math.radians(10.0)))
                   & (g["axis"][0, :, 2] < math.sin(math.radians(70.0)))).sum())
    s, ok = judge()
    seat = scene.seated_now(g)[0]
    check("wrong color order: a real leaning chain with GREEN on the anvil and BLUE "
          f"mid-chain ({leaning}/3 tiles physically leaning) is rejected by the "
          "identity+order predicates — no seat, no credit, no success",
          leaning >= 2 and not bool(seat.any()) and s <= 0.02 and not ok)

    # =========================== 11. skip-chain =============================================
    env.reset(seed=91)
    step(30)
    park("green", (0.95, 0.6))
    place_lean("blue", c.anvil_face - 0.042, 0.0, 52.0, settle_steps=120)
    g = scene.tile_geometry()
    blue_ok = bool(scene.seated_now(g)[0, TILE_IDX["blue"]])
    bfx = float(g["foot"][0, TILE_IDX["blue"], 0])
    place_lean("red", bfx - 0.052, 0.0, 31.0, settle_steps=120)
    report("skip-chain")
    g = scene.tile_geometry()
    s, ok = judge()
    seat = scene.seated_now(g)[0]
    red_leans = float(g["axis"][0, TILE_IDX["red"], 2]) > math.sin(math.radians(12.0))
    check("skip-chain: blue seated on the anvil (honest partial credit) but RED "
          "leaning directly on blue with green parked away is rejected — red's head "
          f"must reach past GREEN (red leaning={red_leans}, seat_red="
          f"{bool(seat[TILE_IDX['red']])}), no success",
          blue_ok and red_leans and not bool(seat[TILE_IDX["red"]])
          and not bool(seat[TILE_IDX["green"]]) and not ok and s <= 0.27)

    # =========================== 12. settle gate ============================================
    env.reset(seed=101)
    step(30)
    place_lean("blue", c.anvil_face - 0.042, 0.0, 52.0, settle_steps=90)
    g = scene.tile_geometry()
    blue_ok = bool(scene.seated_now(g)[0, TILE_IDX["blue"]])
    gfoot = float(g["foot"][0, TILE_IDX["blue"], 0]) - 0.052
    # green in a perfect in-band pose but MOVING (lateral velocity), judged instantly
    th = math.radians(31.0)
    loc = (gfoot + math.cos(th) * c.tile_len / 2, 0.0,
           c.tile_len / 2 * math.sin(th) + 0.004)
    place(scene.tiles["green"], lane_to_world(loc), quat=tile_quat(31.0),
          vel=(0.0, 0.4, 0.0), settle_steps=2)
    v_now = float(scene.tiles["green"].data.root_lin_vel_w[0].norm())
    g = scene.tile_geometry()
    moving_not_seated = not bool(scene.seated_now(g)[0, TILE_IDX["green"]])
    _s, ok = judge()
    report("settle-gate")
    # remove green before it can come to rest leaning on blue (audit: never success)
    park("green", (0.95, 0.6))
    check("settle gate: a tile in an in-band pose but moving at "
          f"{v_now:.2f} m/s is NOT seated (stillness gates are real)",
          blue_ok and v_now > c.settle_lin and moving_not_seated and not ok)

    # =========================== 13. latched credit survives demolition =====================
    step(60)  # blue alone, still seated -> its streak latch is engaged
    s_before, ok = judge()
    place(scene.tiles["blue"],
          torch.tensor([0.95, -0.6, c.tile_len / 2 + 0.002], device=device),
          settle_steps=40)
    report("blue-demolished")
    s_after, ok = judge()
    seat = scene.seated_now()[0]
    check("latched credit: knocking the seated blue tile away leaves the latched "
          f"stage credit unchanged ({s_before:.2f} -> {s_after:.2f}) while "
          "seated_now() drops",
          s_before >= 0.24 and abs(s_after - s_before) < 1e-3
          and not bool(seat[TILE_IDX["blue"]]) and not ok)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.leaning_chain")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(okc for _nm, okc in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, okc in checks:
            if not okc:
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)

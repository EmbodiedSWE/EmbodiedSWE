"""Smoke / rubric-REJECTION battery for DominoRelayScene (sim_gen task
`native_libero_i157`) — NullRobot, teleported probe states + force probes, RECORDED.

This is NOT a solution (solve.py — stand six tiles at even pitch from the red disc to
the green plate, tip the first, let the cascade run — is the acceptance evidence that
the rubric ACCEPTS a correct outcome; it passes on seeds 0/1). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled state and
asserts the rubric REJECTS it — plus force probes that use solve's OWN 0.12 N trigger
to prove the cascade mechanism is physically real on both sides (correct pitch
propagates — demonstrated by solve; over-wide pitch demonstrably dies). No probe in
this battery ever reaches success(), and a final audit asserts exactly that.

  1-2. settle/no-NaN     — reset layout settles finite: tiles flat in the depot below
                           the lane, markers teleported to the stored A/B, score ~0;
  3-4. randomization     — READBACK over 6 seeded resets: A moves, the A->B heading
                           and distance vary (so every pitch plan changes), marker
                           poses match the judged A/B, depot slots are permuted,
                           tiles jitter and take free yaw;
  5.  null policy        — 240 idle steps -> score ~0, no success;
  6.  seed-strategy      — the seed family's move (relocate ONE object to the goal
      (gate probe)         region) is closed: a single tile laid with its head ON the
                           green plate earns only the latched pad/fall crumbs — the
                           chain audit needs >= 2 lane tiles and an anchor at A that
                           one tile can never also provide (cfg-asserted, verified by
                           readback here): no success;
  7.  flat carpet reject — the KEY discriminator: 5 tiles hand-laid FLAT end-to-end
                           from A to B pass every other gate (anchored at A, all
                           fallen toward B, in lane, laps within xy reach, last head
                           on the plate) yet are rejected by `lap_z_min` alone: their
                           heads rest on the GROUND (~6 mm), not propped ON the next
                           tile (~21 mm, solve's readback) — a settled outcome only a
                           real cascade can produce;
  8.  wide pitch dies    — mechanism (reject side): a row at 75 mm pitch (gap 63 mm,
                           beyond the fallen-head reach) + solve's OWN 0.12 N trigger:
                           tile 0 measurably topples (non-vacuous) but strikes air —
                           the rest stay standing, the chain dies, no delivery;
  9.  wrong-way cascade  — a correct-pitch row triggered BACKWARD from the last tile:
                           the tiles physically topple (axis horizontal — non-vacuous)
                           but away from the plate (dir_dot ~ -1): zero cascade
                           credit, no delivery, no success;
  10. stops short        — a 4-tile row at correct pitch: the cascade genuinely runs
                           (>= 3 tiles fall toward B) but the last head lands ~9 cm
                           short of the plate: no delivery, no success;
  11. extra tile spoils  — tiles 0-4 cascade correctly while tile 5 STANDS inside the
                           lane beside them: a lane tile that is not part of the
                           fallen chain fails the all-fallen audit — no success no
                           matter how complete the rest looks;
  12. latched credit     — teleporting every tile back to the depot afterwards leaves
                           the latched score unchanged while the lane empties
                           (credit never evaporates);
  13. settle gate        — a tile IN the lane but freshly teleported and moving is
                           not settled: the windowed pose-drift stillness gate holds
                           judging until the world is genuinely at rest;
  14. rejection audit    — success() was never True at ANY judged point;
  15. final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.native_libero_i157.smoke --headless
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


def main() -> None:
    from isaaclab.utils.math import quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.domino_relay")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    reach = math.sqrt(c.tile_h ** 2 - c.tile_t ** 2)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.05, -0.95, 0.80)) + o),
                                tuple(np.array((0.16, -0.10, 0.02)) + o),
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
        r = scene.tile_report()
        print(f"[smoke] {tag:16s} |"
              f" stand={int((r['standing'][0] & r['in_lane'][0]).sum())}"
              f" fallen={int((r['fallen'][0] & r['in_lane'][0]).sum())}"
              f" pad={bool(r['head_on_pad'][0].any())}"
              f" chain={bool(scene.chain_ok()[0])}"
              f" settled={bool(scene.settled()[0])}"
              f" score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def wrench(body, f3: torch.Tensor) -> None:
        """World force expressed in the body's current link frame (the house
        convention: is_global=True silently drops torques on this stack)."""
        q = body.data.root_link_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f3.view(1, 3).expand(n, 3)).unsqueeze(1),
            torch.zeros(n, 1, 3, device=device),
            env_ids=all_ids)

    def place(body, xyz, quat=None, vel=None, settle_steps: int = 90) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap; the pose-drift stillness gate needs
        >= 2 quiet 30-substep windows)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor(xyz, device=device)
        if quat is None:
            st[:, 3] = 1.0
        else:
            st[:, 3:7] = torch.tensor(quat, device=device)
        if vel is not None:
            st[:, 7:10] = torch.tensor(vel, device=device)
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def au(dist: float, perp: float = 0.0) -> tuple[float, float]:
        """Point A + dist * u + perp * n (n = u rotated +90 deg), env frame."""
        ux, uy = float(scene.U[0, 0]), float(scene.U[0, 1])
        ax, ay = float(scene.A[0, 0]), float(scene.A[0, 1])
        return ax + dist * ux - perp * uy, ay + dist * uy + perp * ux

    def q_upright() -> tuple[float, float, float, float]:
        yaw = math.atan2(float(scene.U[0, 1]), float(scene.U[0, 0]))
        return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

    def q_flat() -> tuple[float, float, float, float]:
        """Lying flat, long axis pointing along u: q = qz(yaw) * qy(90 deg)."""
        yaw = math.atan2(float(scene.U[0, 1]), float(scene.U[0, 0]))
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        c45 = math.cos(math.pi / 4)
        return (cy * c45, -sy * c45, cy * c45, sy * c45)

    def build_row(idxs, pitch: float, settle_steps: int = 75) -> None:
        """Stand tiles `idxs` upright at stations 0..len-1 * pitch along u from A
        (solve's own build move: 3 mm of free air, gravity seats them)."""
        qz = q_upright()
        for k, i in enumerate(idxs):
            x, y = au(k * pitch)
            place(scene.tiles[i], (x, y, c.tile_h / 2 + 0.003), quat=qz, settle_steps=8)
        step(settle_steps)
        r = scene.tile_report()
        for i in idxs:
            assert bool(r["standing"][0, i]), f"probe setup: tile {i} not standing"

    def lay_flat(i: int, dist: float) -> None:
        """Lay tile `i` FLAT on the ground, foot at A + dist*u, head toward B."""
        x, y = au(dist + c.tile_h / 2)
        place(scene.tiles[i], (x, y, c.tile_t / 2 + 0.002), quat=q_flat(), settle_steps=8)

    def trigger(i: int, sign: float = 1.0, max_steps: int = 600) -> bool:
        """Solve's OWN P2 trigger: 0.12 N horizontal CoM force along sign*u on tile
        `i`, re-set every step, cut past ~20 deg of tilt. Returns whether it tipped."""
        f_world = torch.zeros(3, device=device)
        f_world[0:2] = scene.U[0] * (0.12 * sign)
        tipped = False
        for _ in range(max_steps):
            if float(scene.tile_report()["axis"][0, i, 2]) < 0.94:
                tipped = True
                break
            wrench(scene.tiles[i], f_world)
            env.step(no_action)
        wrench(scene.tiles[i], torch.zeros(3, device=device))
        return tipped

    def wait_settled(max_windows: int = 28) -> None:
        for _ in range(max_windows):
            step(30)
            if bool(scene.settled()[0]):
                break

    def layout_sane(tag: str) -> bool:
        """Reset honesty: every tile flat in the depot strip below the lane; both
        markers physically AT the stored (judged) A/B."""
        r = scene.tile_report()
        tiles_ok = bool((r["axis"][0, :, 2].abs() < 0.30).all()
                        & (r["pos"][0, :, 2] < 0.02).all()
                        & (r["pos"][0, :, 1] < -0.22).all())
        sm = (scene.start_mark.data.root_pos_w - scene.env_origins)[0, :2]
        tm = (scene.target_mark.data.root_pos_w - scene.env_origins)[0, :2]
        marks_ok = float((sm - scene.A[0]).norm()) < 1e-3 \
            and float((tm - scene.B[0]).norm()) < 1e-3
        ok = tiles_ok and marks_ok
        if not ok:
            print(f"[smoke] layout sanity VIOLATION at {tag}: tiles_ok={tiles_ok} "
                  f"marks_ok={marks_ok}", flush=True)
        return ok

    def bodies():
        return (*scene.tiles, scene.start_mark, scene.target_mark)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies())
    check("settle: all states finite; tiles flat in the depot below the lane, markers "
          "physically at the stored A/B", fin and layout_sane("reset"))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads, perms, yaws = [], set(), []
    sane = True
    slots = np.array([list(sl) for sl in c.depot_slots])
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(30)
        sane = sane and layout_sane(f"seed {sd}")
        a = scene.A[0]
        u, d = scene.U[0], float(scene.D[0])
        reads.append([float(a[0]), float(a[1]),
                      math.atan2(float(u[1]), float(u[0])), d])
        r = scene.tile_report()
        assign = []
        for i in range(c.n_tiles):
            p = r["pos"][0, i]
            assign.append(int(np.argmin(((slots - np.array([float(p[0]), float(p[1])]))
                                         ** 2).sum(axis=1))))
            ax = r["axis"][0, i]  # flat tile: long axis horizontal — its heading is
            yaws.append(math.atan2(float(ax[1]), float(ax[0])))  # the free yaw
        perms.add(tuple(assign))
        reads[-1] += [float(r["pos"][0, 0, 0]), float(r["pos"][0, 0, 1])]
    arr = np.array(reads)
    print("[smoke] randomization readback (Ax, Ay, heading, D, tile0_xy):\n"
          f"{np.round(arr, 4)}\n[smoke] depot assignments: {sorted(perms)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    yaw_spread = max(yaws) - min(yaws)
    check("randomization: the START disc moves (spread "
          f"x={spread[0]:.3f}, y={spread[1]:.3f}), the A->B heading varies "
          f"({math.degrees(spread[2]):.0f} deg) and the distance varies "
          f"({spread[3] * 1000:.0f} mm) — every pitch plan is different — and every "
          "reset spawns sane with markers at the judged A/B",
          sane and spread[0] > 0.008 and spread[1] > 0.012
          and spread[2] > 0.15 and spread[3] > 0.015)
    check("randomization: depot slots are permuted across seeds "
          f"({len(perms)} distinct assignments), tiles jitter "
          f"(tile0 spread {spread[4]:.3f},{spread[5]:.3f}) and take free yaw "
          f"(readback spread {yaw_spread:.2f} rad)",
          len(perms) >= 2 and spread[4] > 0.010 and yaw_spread > 1.0)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0, no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed-strategy analog: one object to the goal ============
    # The seed family's move — relocate ONE object to the goal region — earns crumbs
    # only: a single tile laid with its head ON the green plate satisfies the delivery
    # geometry but the chain audit needs >= 2 lane tiles and an A-anchor this tile can
    # never also provide (one_tile_span < D, asserted in cfg — verified by readback).
    env.reset(seed=41)
    step(45)
    d = float(scene.D[0])
    lay_flat(0, d - c.tile_h)  # head lands exactly at B
    step(82)
    report("one-tile-on-pad")
    r = scene.tile_report()
    foot_a = float((r["foot"][0, 0, :2] - scene.A[0]).norm())
    s, ok = judge()
    check("seed-strategy analog: ONE tile with its head ON the green plate "
          f"(pad={bool(r['head_on_pad'][0, 0])}) is rejected — its foot sits "
          f"{foot_a * 100:.1f} cm from A (anchor_tol {c.anchor_tol * 100:.1f} cm; one "
          "tile can never bridge), chain needs >= 2 lane tiles: crumbs only, no "
          "success",
          bool(r["head_on_pad"][0, 0]) and foot_a > c.anchor_tol + 0.05
          and not bool(scene.chain_ok()[0]) and s <= 0.20 and not ok)

    # =========================== 7. flat hand-laid carpet is rejected =======================
    # THE discriminator: 5 tiles laid FLAT end-to-end from A to B pass anchor,
    # direction, lane, lap-xy and delivery — but their heads rest on the GROUND
    # (~6 mm < lap_z_min 9 mm), not propped ON the next tile (~21 mm in solve's
    # cascade readback). Only a real cascade leaves the lapped carpet.
    env.reset(seed=51)
    step(45)
    d = float(scene.D[0])
    k_flat = max(2, math.ceil(d / (c.tile_h + 0.002)))
    for i in range(k_flat):
        lay_flat(i, i * (c.tile_h + 0.002))
    step(120)
    report("flat-carpet")
    r = scene.tile_report()
    lane_idx = [i for i in range(c.n_tiles) if bool(r["in_lane"][0, i])]
    heads_low = all(float(r["head"][0, i, 2]) < c.lap_z_min for i in lane_idx)
    anchored = float((r["foot"][0, 0, :2] - scene.A[0]).norm()) < c.anchor_tol
    all_fallen = all(bool(r["fallen"][0, i]) for i in lane_idx)
    delivered = bool(r["head_on_pad"][0].any())
    s, ok = judge()
    check(f"flat carpet reject: {k_flat} tiles hand-laid flat end-to-end A->B pass "
          f"anchor({anchored}) all-fallen({all_fallen}) delivery({delivered}) with "
          f">= 2 lane tiles ({len(lane_idx)}) yet the chain audit rejects them — "
          "every head rests on the GROUND below lap_z_min (no cascade lap): no "
          "success",
          len(lane_idx) >= 2 and anchored and all_fallen and delivered and heads_low
          and bool(scene.settled()[0]) and not bool(scene.chain_ok()[0]) and not ok)

    # =========================== 8. over-wide pitch: the cascade dies =======================
    env.reset(seed=61)
    step(45)
    wide = reach + 0.016  # gap 63 mm: beyond the topple window AND the head reach
    build_row([0, 1, 2], wide)
    tipped = trigger(0)
    wait_settled()
    report("wide-pitch")
    r = scene.tile_report()
    s, ok = judge()
    check("mechanism (reject side): at 75 mm pitch (gap 63 mm > gap_max 42 mm) "
          f"solve's own 0.12 N trigger topples tile 0 (tipped={tipped}, "
          f"fallen={bool(r['fallen'][0, 0])} — non-vacuous) but the fall strikes "
          "air: tiles 1-2 stay standing, the chain dies, no delivery, no success",
          tipped and bool(r["fallen"][0, 0])
          and bool(r["standing"][0, 1]) and bool(r["standing"][0, 2])
          and not bool(r["head_on_pad"][0].any())
          and not bool(scene.chain_ok()[0]) and s <= 0.30 and not ok)

    # =========================== 9. wrong-way cascade =======================================
    env.reset(seed=71)
    step(45)
    d = float(scene.D[0])
    pitch = (d - reach) / (c.n_tiles - 1)
    build_row(list(range(c.n_tiles)), pitch)
    tipped = trigger(c.n_tiles - 1, sign=-1.0)  # push the LAST tile back toward A
    wait_settled()
    report("wrong-way")
    r = scene.tile_report()
    down = int((r["axis"][0, :, 2].abs() < c.fall_axis_z_max).sum())
    toward_b = int(r["fallen"][0].sum())
    s, ok = judge()
    check("direction: triggering the LAST tile toward A topples the row backward "
          f"(tipped={tipped}, {down} tiles horizontal — non-vacuous) but dir_dot ~ -1: "
          f"{toward_b} count as fallen-toward-B, no delivery, no success",
          tipped and down >= 3 and toward_b == 0
          and not bool(r["head_on_pad"][0].any())
          and not bool(scene.chain_ok()[0]) and not ok)

    # =========================== 10. chain stops short ======================================
    env.reset(seed=81)
    step(45)
    d = float(scene.D[0])
    pitch = (d - reach) / (c.n_tiles - 1)
    build_row([0, 1, 2, 3], pitch)  # stations 0..3 only: head lands ~9 cm short of B
    tipped = trigger(0)
    wait_settled()
    report("stops-short")
    r = scene.tile_report()
    fell = int((r["fallen"][0] & r["in_lane"][0]).sum())
    last_head = min(float((r["head"][0, i, :2] - scene.B[0]).norm()) for i in range(4))
    s, ok = judge()
    check("stops short: a 4-tile row at correct pitch cascades for real "
          f"(tipped={tipped}, {fell} fallen toward B in lane) but the nearest head "
          f"stops {last_head * 100:.1f} cm from the plate (pad_r "
          f"{c.pad_r * 100:.1f} cm): no delivery, no success",
          tipped and fell >= 3 and last_head > c.pad_r + 0.02
          and not bool(r["head_on_pad"][0].any())
          and not bool(scene.chain_ok()[0]) and s <= 0.45 and not ok)

    # =========================== 11. extra tile in the lane spoils ==========================
    env.reset(seed=91)
    step(45)
    d = float(scene.D[0])
    pitch = (d - reach) / (c.n_tiles - 1)
    build_row([0, 1, 2, 3, 4], pitch)  # stations 0..4 cascade toward the plate
    x, y = au(0.65 * d, perp=0.045)  # tile 5 STANDS inside the lane, clear of the row
    place(scene.tiles[5], (x, y, c.tile_h / 2 + 0.003), quat=q_upright(), settle_steps=60)
    tipped = trigger(0)
    wait_settled()
    report("extra-tile")
    r = scene.tile_report()
    fell = int((r["fallen"][0] & r["in_lane"][0]).sum())
    spoiler = bool(r["standing"][0, 5] & r["in_lane"][0, 5])
    s, ok = judge()
    check("extra tile spoils: tiles 0-4 cascade toward the plate "
          f"(tipped={tipped}, {fell} fallen in lane) while tile 5 still STANDS in "
          f"the lane ({spoiler}) — a lane tile outside the fallen chain fails the "
          "audit: no success",
          tipped and fell >= 4 and spoiler
          and not bool(scene.chain_ok()[0]) and not ok)

    # =========================== 12. latched credit survives removal ========================
    s_before, _ok = judge()
    for i in range(c.n_tiles):
        sx, sy = c.depot_slots[i]
        place(scene.tiles[i], (sx, sy, c.tile_t / 2 + 0.002), quat=q_flat(),
              settle_steps=6)
    step(60)
    report("tiles-removed")
    r = scene.tile_report()
    lane_now = int(r["in_lane"][0].sum())
    s_after, ok = judge()
    check("latched credit: teleporting every tile back to the depot leaves the "
          f"latched score unchanged ({s_before:.2f} -> {s_after:.2f}) while the lane "
          f"empties ({lane_now} tiles in lane)",
          abs(s_after - s_before) < 1e-3 and lane_now == 0 and not ok)

    # =========================== 13. settle gate ============================================
    env.reset(seed=101)
    step(45)
    x, y = au(0.4 * float(scene.D[0]))
    place(scene.tiles[0], (x, y, c.tile_h / 2 + 0.003), quat=q_upright(),
          vel=(0.4, 0.0, 0.0), settle_steps=2)
    moving = float(scene.tiles[0].data.root_lin_vel_w[0].norm())
    not_settled = not bool(scene.settled()[0])
    _s, ok = judge()
    check("settle gate: a freshly teleported tile moving at "
          f"{moving:.2f} m/s in the lane is not settled (windowed pose-drift "
          "stillness) and cannot be judged a success", moving > 0.05
          and not_settled and not ok)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies())
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.domino_relay")
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

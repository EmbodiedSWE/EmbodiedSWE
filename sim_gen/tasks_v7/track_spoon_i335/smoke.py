"""Smoke / rubric-REJECTION battery for SieveSorterScene (sim_gen task
`track_spoon_i335`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — feed every tray object onto the open upstream
stretch of the rails and let the sieve classify it by size — is the acceptance
evidence that the rubric ACCEPTS a correct outcome, verified on seeds 0 and 1).
Every teleport here is instrumentation that CONSTRUCTS a wrong (or partial) outcome
and asserts the rubric REJECTS it; no probe in this battery ever reaches success(),
and a final audit check asserts exactly that.

  1.  settle/no-NaN       — reset state settles: apparatus upright, batch resting in
                            the tray; score ~0, no success;
  2.  authored masses     — get_masses() readback: apparatus 25 kg and tray 1.2 kg
                            (custom compound spawners must author MassAPI themselves;
                            cfg mass_props are ignored there), bead 10 g, marble 60 g;
  3.  randomization       — READBACK over 8 seeded resets: apparatus xy + yaw, tray
                            xy + FREE yaw, bead/marble counts, and slot assignment
                            (tray-frame bead offset) all vary;
  4.  null policy         — 240 idle steps -> score ~0, no success;
  5.  SEED strategy       — the seed's whole plan ("carry the object along a path and
                            set it down") = every present object transported and SET
                            DOWN on the open floor around the machine: score ~0, NOT
                            success;
  6.  wrong-bin swap      — a marble placed at rest in the LOWER bin and a bead at
                            rest in the END bin (each in the OTHER type's bin): both
                            genuinely settled in-a-bin, but correct_bin is per-TYPE —
                            no latch, score ~0, NOT success;
  7.  roof denial         — a marble released vertically ABOVE the roofed end bin
                            lands ON the roof (readback: local z far above the bin
                            cap, not in-bin), rolls back down the roof onto the open
                            sieve, and only THEN rides the rails over the sill in —
                            trajectory readback proves the only entry is THROUGH the
                            machine (first sub-cap sample is already past the sill;
                            an earlier sample shows it riding below roof level while
                            still upstream of the sill);
  8.  settle gate         — a bead released mid-air INSIDE the lower-bin box:
                            geometric containment reads True mid-fall, but settled()
                            is False (velocity + pose-window readback) and success is
                            refused at that judged instant;
  9.  sieve passes beads  — one bead fed solve-style over a groove on the open
                            stretch falls THROUGH the 21 mm gap and latches its
                            per-object credit: score == 0.55/n_present exactly, and
                            with the rest of the batch still in the tray, NOT
                            success (partial credit is honest);
  10. sieve blocks marbles— the marble fed over the SAME groove line NEVER enters
                            the lower-bin box at any polled instant (34 mm > 21 mm),
                            rides the V-groove over the sill and latches in the END
                            bin — same feed zone, opposite verdict; still NOT
                            success (a bead remains in the tray);
  11. latched credit      — the delivered marble teleported OUT to the depot: the
                            latched credit survives unchanged, still NOT success;
  12. slab-rim near-miss  — a bead set down on the machine's own slab rim OUTSIDE
                            the side wall (|y| beyond the bin bound): adjacency to
                            the machine is worthless — no latch, score unchanged,
                            NOT success;
  13. rejection audit     — success() was never True at ANY judged point;
  14. final no-NaN        — all task-object states finite; frames.npz saved.

Run (forge): python -u -m simgen_tasks.track_spoon_i335.smoke --headless
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    from isaaclab.utils.math import quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.sieve_sorter")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.55, -1.15, 0.95)) + o),
                                tuple(np.array((0.42, -0.05, 0.12)) + o),
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
        d = scene.obj_local()[0]
        cb = scene.correct_bin()[0]
        pres = scene._present[0]
        locs = " ".join(
            f"{nm}=({float(d[j, 0]):+.3f},{float(d[j, 1]):+.3f},{float(d[j, 2]):+.3f})"
            f"{'C' if bool(cb[j]) else '.'}{'L' if bool(scene._binned[0, j]) else '.'}"
            for j, nm in enumerate(scene.OBJ_NAMES) if bool(pres[j]))
        print(f"[smoke] {tag:16s} | {locs}", flush=True)
        print(f"[smoke] {tag:16s} | upright={bool(scene.upright()[0])} "
              f"settled={bool(scene.settled()[0])} still={bool(scene._still_pos[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def write_state(body, pos_w: torch.Tensor) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3] = 1.0
        body.write_root_state_to_sim(st, all_ids)

    def app_world(x: float, y: float, z: float) -> torch.Tensor:
        """Apparatus-frame point -> world (per env)."""
        local = torch.tensor([x, y, z], device=device).expand(n, 3)
        return scene.apparatus.data.root_pos_w \
            + quat_apply(scene.apparatus.data.root_quat_w, local)

    def place_floor(body, wx: float, wy: float, z: float, settle_steps: int = 30) -> None:
        """Kinematic probe placement at WORLD floor coords (instrumentation, not a
        solution) + REAL physics steps before judging (the zero-step trap)."""
        pos = torch.zeros(n, 3, device=device)
        pos[:, 0], pos[:, 1], pos[:, 2] = wx, wy, z
        pos += scene.env_origins
        write_state(body, pos)
        step(settle_steps)

    def floorB_z(x: float) -> float:
        return c.slab_t + (x - c.up_wall_in_x) * math.tan(math.radians(c.floorB_tilt_deg))

    def floorA_z(x: float) -> float:
        return c.slab_t + (x - c.binA_x0) * math.tan(math.radians(c.floorA_tilt_deg))

    def in_boxB(j: int) -> bool:
        d = scene.obj_local()[0, j]
        return (c.binB_x0 <= float(d[0]) <= c.binB_x1 and abs(float(d[1])) <= c.y_half
                and 0.0 < float(d[2]) <= c.binB_zcap)

    def feed(j: int, groove_y: float, feed_x: float, drop: float,
             max_blocks: int = 160) -> tuple[bool, bool]:
        """Hover object j above the groove line on the open stretch (the solve's
        honest feed) and RELEASE; poll the per-object latch. Returns (delivered,
        ever_in_lower_bin_box during the run)."""
        r = c.marble_r if j >= 3 else c.bead_r
        z = c.waist_z(feed_x) + c.rail_half_w + r + drop
        write_state(scene.objs[j], app_world(feed_x, groove_y, z))
        ever_b = False
        for _ in range(max_blocks):  # blocks of 10 substeps
            step(10)
            ever_b = ever_b or in_boxB(j)
            if bool(scene._binned[0, j]):
                return True, ever_b
        return False, ever_b

    def fin_all() -> bool:
        ok = bool(torch.isfinite(scene.apparatus.data.root_state_w).all()) \
            and bool(torch.isfinite(scene.tray.data.root_state_w).all())
        for b in scene.objs:
            ok = ok and bool(torch.isfinite(b.data.root_state_w).all())
        return ok

    def yaw_of(q: torch.Tensor) -> float:
        return 2.0 * math.atan2(float(q[3]), float(q[0]))

    # =========================== 1. settle / no-NaN =========================================
    env.reset(seed=11)
    step(240)
    report("reset-settled")
    s, ok = judge()
    tray_xy = scene.tray.data.root_pos_w[0, :2] - scene.env_origins[0, :2]
    in_tray = True
    for j in range(5):
        if bool(scene._present[0, j]):
            oxy = scene.objs[j].data.root_pos_w[0, :2] - scene.env_origins[0, :2]
            in_tray = in_tray and float((oxy - tray_xy).norm()) < 0.16
    check("settle: states finite, apparatus upright and settled, every present object "
          "resting in the tray; score ~0, no success",
          fin_all() and bool(scene.upright()[0]) and bool(scene.settled()[0])
          and in_tray and s <= 0.03 and not ok)

    # =========================== 2. authored masses readback ================================
    am = float(scene.apparatus.root_physx_view.get_masses().flatten()[0])
    tm = float(scene.tray.root_physx_view.get_masses().flatten()[0])
    bm = float(scene.objs[0].root_physx_view.get_masses().flatten()[0])
    mm = float(scene.objs[3].root_physx_view.get_masses().flatten()[0])
    print(f"[smoke] mass readback: apparatus={am:.2f}kg (cfg {c.app_mass:.2f}kg) "
          f"tray={tm:.3f}kg (cfg {c.tray_mass:.3f}kg) bead={bm * 1000:.1f}g "
          f"(cfg {c.bead_m * 1000:.1f}g) marble={mm * 1000:.1f}g "
          f"(cfg {c.marble_m * 1000:.1f}g)", flush=True)
    check("authored masses: apparatus and tray read back their cfg masses (the custom "
          "compound spawners author MassAPI mass+CoM themselves), bead and marble "
          "read theirs",
          abs(am - c.app_mass) < 0.1 and abs(tm - c.tray_mass) < 0.02
          and abs(bm - c.bead_m) < 1e-3 and abs(mm - c.marble_m) < 1e-3)

    # =========================== 3. randomization is real ===================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(5)
        axy = scene._app_xy[0]
        txy = scene._tray_xy[0]
        tyaw = float(scene._tray_yaw[0])
        nb = int(scene._present[0, :3].sum())
        nm = int(scene._present[0, 3:].sum())
        b0 = scene.objs[0].data.root_pos_w[0, :2] - scene.env_origins[0, :2]
        rel = b0 - txy
        cy, sy = math.cos(tyaw), math.sin(tyaw)
        slot = (cy * float(rel[0]) + sy * float(rel[1]),
                -sy * float(rel[0]) + cy * float(rel[1]))  # bead0 slot, tray frame
        reads.append((float(axy[0]), float(axy[1]), float(scene._app_yaw[0]),
                      float(txy[0]), float(txy[1]), tyaw, float(nb), float(nm),
                      slot[0], slot[1]))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (app_x, app_y, app_yaw, tray_x, tray_y, "
          f"tray_yaw, n_beads, n_marbles, slot_x, slot_y):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: apparatus xy + yaw, tray xy + FREE yaw, bead and marble "
          "counts, and slot assignment all vary across seeded resets (readback)",
          spread[0] > 0.01 and spread[1] > 0.01 and spread[2] > 0.05
          and spread[3] > 0.01 and spread[4] > 0.01 and spread[5] > 0.5
          and spread[6] >= 1.0 and spread[7] >= 1.0
          and (spread[8] > 0.02 or spread[9] > 0.02))

    # =========================== 4. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.03 and not ok)

    # =========================== 5. SEED strategy ===========================================
    # The seed's whole plan is "carry the object along a path and set it down". Every
    # present object transported and SET DOWN on the open floor around the machine —
    # transport fidelity with no machine interaction — must be worthless.
    env.reset(seed=41)
    step(120)
    spots = ((0.00, 0.55), (-0.10, 0.30), (-0.20, 0.05), (0.05, -0.05), (-0.25, 0.45))
    for j in range(5):
        if not bool(scene._present[0, j]):
            continue
        r = c.marble_r if j >= 3 else c.bead_r
        wx, wy = spots[j]
        place_floor(scene.objs[j], wx, wy, r + 0.03, settle_steps=30)
    step(180)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy (every object carried and SET DOWN on the floor around the "
          "machine): transport without the machine — score <= 0.03, NOT success",
          s <= 0.03 and not ok)

    # =========================== 6. wrong-bin swap ==========================================
    # A marble at rest in the LOWER bin and a bead at rest in the END bin: both are
    # genuinely settled inside a bin, but correct_bin is per-TYPE.
    env.reset(seed=51)
    step(120)
    j_bead = 0  # bead0 is always present (n_beads_min = 2)
    write_state(scene.objs[3], app_world(-0.20, 0.0, floorB_z(-0.20) + c.marble_r + 0.005))
    write_state(scene.objs[j_bead], app_world(0.20, 0.03, floorA_z(0.20) + c.bead_r + 0.005))
    step(300)
    report("wrong-bin-swap")
    s, ok = judge()
    d = scene.obj_local()[0]
    check("wrong-bin swap: the marble rests in the LOWER bin and the bead in the END "
          "bin (readback: both inside a bin box, both quiet) — correct_bin is "
          "per-type, so no latch fires: score <= 0.03, NOT success",
          in_boxB(3) and (c.binA_x0 <= float(d[j_bead, 0]) <= c.binA_x1)
          and abs(float(d[j_bead, 1])) <= c.y_half
          and float(d[j_bead, 2]) <= c.binA_zcap
          and not bool(scene.correct_bin()[0, 3])
          and not bool(scene.correct_bin()[0, j_bead])
          and not bool(scene._binned[0].any()) and s <= 0.03 and not ok)

    # =========================== 7. roof denial =============================================
    # A marble released vertically ABOVE the end bin cannot enter it: it lands ON the
    # roof, rolls back down the tilted roof onto the open sieve, and only then rides
    # the rails over the sill in. The trajectory readback proves entry is only
    # THROUGH the machine.
    # clear the bead debris far off-stage first (probe debris blocks the next probe)
    place_floor(scene.objs[j_bead], c.depot[0], c.depot[1] - 0.4, c.bead_r + 0.002,
                settle_steps=30)
    write_state(scene.objs[3], app_world(0.20, 0.0, 0.31))  # directly above the end bin
    step(30)
    d7 = scene.obj_local()[0, 3]
    on_roof = float(d7[2]) > c.binA_zcap and not bool(scene.correct_bin()[0, 3])
    print(f"[smoke] roof-denial early readback: marble at local "
          f"({float(d7[0]):+.3f},{float(d7[1]):+.3f},{float(d7[2]):+.3f})", flush=True)
    recs: list[tuple[float, float]] = []
    delivered = False
    for _ in range(240):  # up to 10 s
        step(5)
        dd = scene.obj_local()[0, 3]
        recs.append((float(dd[0]), float(dd[2])))
        if bool(scene._binned[0, 3]):
            delivered = True
            break
    first_dip_x = next((x for x, z in recs if z < c.binA_zcap), None)
    rode_rails = any(z < c.roof_z0 - 0.01 and x < c.binB_x1 for x, z in recs)
    report("roof-denial")
    s, ok = judge()
    check("roof denial: the vertical drop over the end bin lands ON the roof (early "
          "readback z above the bin cap, not in-bin); it then rolls back onto the "
          "open sieve (a polled sample rides below roof level upstream of the sill) "
          "and its first sub-cap sample is already past the sill — the only entry "
          "into the end bin is THROUGH the machine; still NOT success",
          on_roof and delivered and rode_rails
          and first_dip_x is not None and first_dip_x >= c.binA_x0 - 0.005
          and not ok)

    # =========================== 8. settle gate =============================================
    # A bead released mid-air INSIDE the lower-bin box: geometric containment reads
    # True mid-fall, but the rest gate refuses the state at that judged instant.
    env.reset(seed=61)
    step(120)
    write_state(scene.objs[0], app_world(-0.20, -0.02, 0.088))  # inside the B box, airborne
    step(2)
    v_mid = float(scene.objs[0].data.root_lin_vel_w.norm(dim=-1)[0])
    geom_mid = bool(scene.correct_bin()[0, 0])
    settled_mid = bool(scene.settled()[0])
    s_mid, ok_mid = judge()
    check("settle gate: mid-fall inside the lower-bin box the bead reads "
          "geometrically contained but unsettled (|v| readback > settle_lin, pose "
          "window broken) and success is False at that judged instant",
          geom_mid and not settled_mid and v_mid > c.settle_lin and not ok_mid)
    step(120)
    report("settle-gate-after")

    # =========================== 9. the sieve passes beads ==================================
    env.reset(seed=71)
    step(240)
    npres = int(scene._present[0].sum())
    g = c.groove_ys
    ok_feed, _ = feed(0, g[1], -0.20, 0.030)
    step(60)
    report("bead-fed")
    s9, ok = judge()
    expect9 = 0.55 / npres
    check("sieve passes beads: one bead fed over a groove on the open stretch falls "
          "THROUGH the 21 mm gap and latches exactly its per-object share "
          f"(score {expect9:.3f} for {npres} present objects); the rest of the batch "
          "is still in the tray: NOT success",
          ok_feed and bool(scene.correct_bin()[0, 0]) and in_boxB(0)
          and abs(s9 - expect9) < 0.02 and not ok)

    # =========================== 10. the sieve blocks marbles ===============================
    ok_feed_m, ever_b = feed(3, g[2], -0.24, 0.020)
    step(60)
    report("marble-fed")
    s10, ok = judge()
    d10 = scene.obj_local()[0, 3]
    expect10 = 2 * 0.55 / npres
    check("sieve blocks marbles: the marble fed over the SAME groove line NEVER "
          "enters the lower-bin box at any polled instant (34 mm > 21 mm gap), rides "
          "the V-groove over the sill and latches in the END bin — same feed zone, "
          "opposite verdict; a bead remains in the tray: NOT success",
          ok_feed_m and not ever_b and bool(scene.correct_bin()[0, 3])
          and float(d10[0]) >= c.binA_x0 and abs(s10 - expect10) < 0.02 and not ok)

    # =========================== 11. latched credit survives removal ========================
    place_floor(scene.objs[3], 1.6, 1.4, c.marble_r + 0.002, settle_steps=60)
    report("marble-removed")
    s11, ok = judge()
    check("latched credit: the delivered marble teleported OUT to the depot leaves "
          "the latched score unchanged (and still no success)",
          not bool(scene.correct_bin()[0, 3]) and abs(s11 - s10) < 0.01
          and s11 >= expect10 - 0.02 and not ok)

    # =========================== 12. slab-rim near-miss =====================================
    # bead1 is always present (n_beads_min = 2): set it down on the machine's OWN
    # slab rim outside the side wall (|y| beyond the bin bound) — adjacency to the
    # machine is worthless.
    write_state(scene.objs[1], app_world(0.0, 0.137, c.slab_t + c.bead_r + 0.004))
    step(240)
    report("slab-rim")
    s12, ok = judge()
    check("slab-rim near-miss: a bead resting on the machine's slab rim OUTSIDE the "
          "side wall never reads in-bin and never latches — score unchanged, NOT "
          "success",
          not bool(scene.correct_bin()[0, 1]) and not bool(scene._binned[0, 1])
          and abs(s12 - s11) < 0.01 and not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", fin_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.sieve_sorter")
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
    main()

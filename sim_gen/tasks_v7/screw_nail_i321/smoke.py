"""Smoke / rubric-REJECTION battery for RamDispenserScene (sim_gen task
`screw_nail_i321`) — NullRobot, teleported probe states + force probes, RECORDED.

This is NOT a solution (solve.py — stage the bin, then pump the ram full-stroke once
per cube — is the acceptance evidence that the rubric ACCEPTS a correct outcome; it
passes on seeds 0/1). Every teleport here is instrumentation that CONSTRUCTS a wrong
(or partial) outcome as a settled state and asserts the rubric REJECTS it — plus
applied-force probes that prove the MAGAZINE is physically sealed: a regulated lift
that demonstrably raises a cube inside the tower can never take it out of the tower.
No probe in this battery ever reaches success(), and a final audit asserts exactly
that.

  1-2. settle/no-NaN      — reset layout settles finite: present cubes sealed in the
                            tower shaft, absent cubes in the far depot, ram tip
                            inside one of its two start bands, bin at a park slot;
                            score ~0 at rest, no success;
  3-4. randomization      — READBACK over 6 seeded resets: cube count AND which
                            cubes vary; ram tip spreads and BOTH start modes occur
                            (scanning further seeds); bin park position + yaw vary;
  5.  null policy         — 240 idle steps -> cubes stay sealed, score ~0, no success;
  6.  seed-strategy       — the seed family's move (press down and TWIST, the
                            screwdriver motion) applied to the ram achieves nothing:
                            the ram barely moves, nothing is dispensed, score ~0;
  7.  sealed tower        — a regulated lift (velocity servo, <= 2 N ~ 7x cube
                            weight) on the TOP cube demonstrably raises it (>= 15 mm
                            — the probe is not vacuous) yet the cube NEVER leaves
                            the capped shaft, and no credit appears;
  8.  mid-channel         — a cube parked mid-channel (fed but not ejected, a
                            half-done cycle) earns no ejection credit, no success;
  9.  missed catch        — a cube on the floor BESIDE the staged bin (dispensed
                            with the bin mis-placed): ejection credit only, NOT in
                            the bin, no success;
  10. wall-top perch      — a cube centered on the bin wall top reads outside the
                            containment window (xy beyond the inner faces, z above
                            the height window) -> rejected, removed before toppling;
  11. tipped bin          — the bin overturned at the catch point with a cube
                            resting on its upturned floor: upright gate rejects;
  12. settle gate         — a cube IN the bin window but still moving is NOT
                            success (k=1 seed: the settle gate is decisive);
                            removed before it can settle;
  13. latched credit      — staging the bin then parking it away again leaves the
                            latched 0.10 staging credit in place;
  14. rejection audit     — success() was never True at ANY judged point;
  15. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.screw_nail_i321.smoke --headless
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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ram_dispenser")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(3, device=device)
    rx, ry = c.rig_pos
    mid = (c.slot_x[0] + c.slot_x[1]) / 2
    catch = (rx + c.catch_x, ry)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.40, -1.05, 0.90)) + o),
                                tuple(np.array((0.45, 0.00, 0.10)) + o),
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

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def tip() -> float:
        return float(scene.ram_tip()[0])

    def report(tag: str) -> None:
        s, ok = judge()
        bp = scene.block_pos()[0]
        pres = scene.present[0]
        binp = rel(scene.bin)
        zs = " ".join(f"b{i}={'--' if not bool(pres[i]) else f'{float(bp[i, 2]):.3f}'}"
                      for i in range(3))
        print(f"[smoke] {tag:16s} | tip={tip():+.3f} {zs}"
              f" bin=({float(binp[0]):+.3f},{float(binp[1]):+.3f})"
              f" staged={bool(scene.bin_staged_now()[0])}"
              f" score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, xyz, quat=None, vel=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
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

    def wrench(body, f3: torch.Tensor, t3: torch.Tensor) -> None:
        """World wrench expressed in the body's current link frame (the house
        convention: is_global=True silently drops the torque on this stack)."""
        from isaaclab.utils.math import quat_apply_inverse

        q = body.data.root_link_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f3.view(1, 3).expand(n, 3)).unsqueeze(1),
            quat_apply_inverse(q, t3.view(1, 3).expand(n, 3)).unsqueeze(1),
            env_ids=all_ids)

    def present_idx() -> list[int]:
        return [i for i in range(3) if bool(scene.present[0, i])]

    def layout_sane(tag: str) -> bool:
        """Reset honesty: present cubes inside the shaft (sealed), absent cubes in
        the depot, ram tip inside a start band, bin upright at a park slot."""
        ok = True
        bp = scene.block_pos()[0]
        for i in range(3):
            x, y, z = (float(v) for v in bp[i])
            if bool(scene.present[0, i]):
                ok = ok and abs(x - rx - mid) < 0.020 and abs(y - ry) < 0.020 \
                    and 0.10 < z < 0.26
            else:
                ok = ok and math.hypot(x - c.depot[0], y - c.depot[1] - 0.10 * i) < 0.05 \
                    and z < 0.05
        ok = ok and 0.028 <= tip() <= 0.114
        binp = rel(scene.bin)
        near = min(math.hypot(float(binp[0]) - sx, float(binp[1]) - sy)
                   for sx, sy in c.bin_slots)
        ok = ok and near < c.bin_jitter * 1.5 + 0.02 and float(binp[2]) < 0.02 \
            and bool(scene.bin_upright()[0])
        if not ok:
            print(f"[smoke] layout sanity VIOLATION at {tag}", flush=True)
        return ok

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    bodies = (scene.ram, scene.bin, *scene.blocks)
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("settle: all states finite; cubes sealed in the tower, ram in-band, "
          "bin parked", fin and layout_sane("reset"))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    sane = True
    for sd in (31, 32, 33, 34, 35, 36):
        env.reset(seed=sd)
        step(30)
        sane = sane and layout_sane(f"seed {sd}")
        pres = tuple(int(scene.present[0, i]) for i in range(3))
        binp = rel(scene.bin)
        q = scene.bin.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        reads.append((pres, tip(), float(binp[0]), float(binp[1]), yaw))
        print(f"[smoke] seed {sd}: present={pres} tip={tip():+.3f} "
              f"bin=({float(binp[0]):+.3f},{float(binp[1]):+.3f}) yaw={yaw:+.2f}",
              flush=True)
    ks = {sum(r[0]) for r in reads}
    subsets = {r[0] for r in reads}
    check("randomization: cube count AND cube subset vary across seeds; every "
          "layout sane", sane and len(ks) >= 2 and len(subsets) >= 2)
    tips = [r[1] for r in reads]
    tip_spread = max(tips) - min(tips)
    modes = {t > 0.06 for t in tips}
    sd = 37
    while len(modes) < 2 and sd < 61:  # scan further seeds for the second start mode
        env.reset(seed=sd)
        step(10)
        modes.add(tip() > 0.06)
        sd += 1
    xy = [(r[2], r[3]) for r in reads]
    bin_spread = max(math.hypot(a[0] - b[0], a[1] - b[1]) for a in xy for b in xy)
    yaws = [r[4] for r in reads]
    yaw_spread = max(yaws) - min(yaws)
    check("randomization: ram tip spreads and BOTH start modes occur; bin park "
          f"position + yaw vary (tip {tip_spread:.3f}, modes {len(modes)}, "
          f"bin {bin_spread:.3f}, yaw {yaw_spread:.2f})",
          tip_spread >= 0.003 and len(modes) == 2
          and bin_spread >= 0.05 and yaw_spread >= 0.3)

    # =========================== 5. null policy =============================================
    env.reset(seed=41)
    step(240)
    report("null-240")
    s, ok = judge()
    check("null policy: 240 idle steps -> cubes stay sealed, score ~0, no success",
          layout_sane("null") and s <= 0.02 and not ok)

    # =========================== 6. seed-strategy analog ====================================
    # The seed task's move: press DOWN and TWIST (the screwdriver motion), applied to
    # the machine's only moving part. It dispenses nothing.
    env.reset(seed=61)
    step(30)
    tip0 = tip()
    f3 = torch.tensor([0.0, 0.0, -4.0], device=device)
    t3 = torch.tensor([0.0, 0.0, 0.3], device=device)
    for _ in range(240):
        wrench(scene.ram, f3, t3)
        step(1)
    wrench(scene.ram, zero3, zero3)
    step(45)
    report("press+twist")
    s, ok = judge()
    bp = scene.block_pos()[0]
    none_out = all(float(bp[i, 2]) > c.eject_z for i in present_idx())
    check("seed-strategy analog: press-down + twist on the ram moves it "
          f"{abs(tip() - tip0) * 1000:.1f} mm, dispenses nothing, score ~0",
          abs(tip() - tip0) < 0.015 and none_out and s <= 0.02 and not ok)

    # =========================== 7. the magazine is sealed (force probe) ====================
    # Regulated lift (velocity servo toward 0.12 m/s, clamped to [0, 2] N ~ 7x cube
    # weight) on the TOP cube. It must demonstrably RISE (the probe is real) yet stay
    # inside the capped shaft, and reseat with no credit.
    env.reset(seed=71)
    step(30)
    top = max(present_idx(), key=lambda i: float(scene.block_pos()[0, i, 2]))
    body = scene.blocks[top]
    z0 = float(rel(body)[2])
    zmax, xymax = z0, 0.0
    fl = torch.zeros(3, device=device)
    for _ in range(300):
        vz = float(body.data.root_lin_vel_w[0, 2])
        fl[2] = min(max(c.block_mass * 9.81 + 2.0 * (0.12 - vz), 0.0), 2.0)
        wrench(body, fl, zero3)
        step(1)
        p = rel(body)
        zmax = max(zmax, float(p[2]))
        xymax = max(xymax, abs(float(p[0]) - rx - mid), abs(float(p[1]) - ry))
    wrench(body, zero3, zero3)
    step(90)
    report("tower-lift")
    s, ok = judge()
    pend = rel(body)
    back_in = abs(float(pend[0]) - rx - mid) < 0.025 and abs(float(pend[1]) - ry) < 0.025 \
        and float(pend[2]) < 0.255
    check("sealed tower: the regulated lift raises the top cube "
          f"{(zmax - z0) * 1000:.0f} mm (probe is real) but it never leaves the capped "
          f"shaft (max z {zmax:.3f} < {c.tower_top - c.block_edge / 2 + 0.007:.3f}, "
          f"xy drift {xymax * 1000:.0f} mm) and reseats with no credit",
          zmax - z0 >= 0.015 and zmax <= c.tower_top - c.block_edge / 2 + 0.007
          and xymax < 0.030 and back_in and s <= 0.02 and not ok
          and float(scene.eject_latch[0, top]) < 0.5)

    # =========================== 8. mid-channel cube: a half-done cycle =====================
    sd, found = 81, False
    while sd < 101:  # need a retracted-ram layout (clear channel ahead of the nose)
        env.reset(seed=sd)
        step(30)
        if tip() < 0.05:
            found = True
            break
        sd += 1
    assert found, "no retracted-ram seed found in 81..100"
    bi = min(present_idx(), key=lambda i: float(scene.block_pos()[0, i, 2]))
    place(scene.blocks[bi], (rx + 0.115, ry, c.chan_floor_z + c.block_edge / 2 + 0.0015))
    report("mid-channel")
    s, ok = judge()
    check("mid-channel: a cube parked in the roofed channel (fed, not ejected) earns "
          "no ejection credit and no success",
          float(scene.eject_latch[0, bi]) < 0.5
          and not bool(scene.block_in_bin()[0, bi]) and s <= 0.02 and not ok)

    # =========================== 9. missed catch: cube beside the staged bin ================
    env.reset(seed=91)
    step(30)
    kpres = len(present_idx())
    place(scene.bin, (catch[0], catch[1], 0.001), settle_steps=45)
    bi = present_idx()[0]
    place(scene.blocks[bi],
          (catch[0] + c.bin_inner_half + c.bin_wall_t + 0.016, catch[1],
           c.block_edge / 2 + 0.0015))
    report("missed-catch")
    s, ok = judge()
    exp = 0.10 + 0.50 / kpres
    check("missed catch: a cube on the floor beside the staged bin gets ejection "
          f"credit only (score {s:.3f} ~ {exp:.3f}), is NOT in the bin, no success",
          abs(s - exp) < 0.02 and not bool(scene.block_in_bin()[0, bi])
          and float(scene.inbin_latch[0, bi]) < 0.5 and not ok)

    # =========================== 10. wall-top perch =========================================
    place(scene.blocks[bi],
          (catch[0] + c.bin_inner_half + c.bin_wall_t / 2, catch[1],
           c.bin_wall_top + c.block_edge / 2 + 0.0015), settle_steps=5)
    relb = scene._blocks_in_bin_frame()[0, bi]
    report("wall-perch")
    s, ok = judge()
    perch_ok = not bool(scene.block_in_bin()[0, bi]) and not ok \
        and float(relb[0].abs()) > c.bin_xy_tol
    # remove it BEFORE it can topple into the bin (this battery must never succeed)
    place(scene.blocks[bi], (1.50, -1.20, c.block_edge / 2 + 0.002), settle_steps=30)
    check("wall-top perch: a cube centered on the bin wall reads outside the window "
          f"(bin-frame x {float(relb[0]):+.3f} > {c.bin_xy_tol}) -> rejected",
          perch_ok)

    # =========================== 11. tipped bin =============================================
    env.reset(seed=111)
    step(30)
    bi = present_idx()[0]
    place(scene.bin, (catch[0], catch[1], c.bin_wall_top + 0.0015),
          quat=(0.0, 1.0, 0.0, 0.0), settle_steps=30)  # upside down at the catch point
    place(scene.blocks[bi],
          (catch[0], catch[1], c.bin_wall_top + 0.001 + c.block_edge / 2 + 0.002))
    report("tipped-bin")
    s, ok = judge()
    check("tipped bin: overturned bin with a cube resting on its upturned floor is "
          "rejected by the upright gate",
          not bool(scene.bin_upright()[0]) and not bool(scene.block_in_bin()[0, bi])
          and not ok)

    # =========================== 12. settle gate ============================================
    sd, found = 121, False
    while sd < 151:  # a k=1 seed: the settle gate is then the DECISIVE rejector
        env.reset(seed=sd)
        step(30)
        if len(present_idx()) == 1:
            found = True
            break
        sd += 1
    assert found, "no k=1 seed found in 121..150"
    bi = present_idx()[0]
    place(scene.bin, (catch[0], catch[1], 0.001), settle_steps=45)
    place(scene.blocks[bi], (catch[0], catch[1], 0.030), vel=(0.40, 0.0, 0.0),
          settle_steps=1)
    v_now = float(scene.blocks[bi].data.root_lin_vel_w[0].norm())
    in_win = bool(scene.block_in_bin()[0, bi])
    report("settle-gate")
    s, ok = judge()
    gate_ok = in_win and v_now > c.settle_lin and not ok
    # remove it BEFORE it can settle in the bin (this battery must never succeed)
    place(scene.blocks[bi], (1.50, -1.20, c.block_edge / 2 + 0.002), settle_steps=30)
    check("settle gate: the ONLY cube inside the bin window but moving at "
          f"{v_now:.2f} m/s is NOT success (velocity gates are real)", gate_ok)

    # =========================== 13. latched staging credit =================================
    env.reset(seed=131)
    step(30)
    place(scene.bin, (catch[0], catch[1], 0.001), settle_steps=30)
    s_staged, _ = judge()
    place(scene.bin, (c.bin_slots[0][0], c.bin_slots[0][1], 0.001), settle_steps=30)
    report("un-staged")
    s_after, ok = judge()
    check("latched credit: staging the bin then parking it away keeps the 0.10 "
          f"credit ({s_staged:.2f} -> {s_after:.2f}) while bin_staged_now() drops",
          s_staged >= 0.099 and s_after >= 0.099 and abs(s_after - s_staged) < 1e-3
          and not bool(scene.bin_staged_now()[0]) and not ok)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.ram_dispenser")
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

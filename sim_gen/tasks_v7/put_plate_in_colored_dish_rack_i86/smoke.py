"""Smoke / rubric-REJECTION battery for CarouselDishRackScene (sim_gen task
`put_plate_in_colored_dish_rack_i86`) — NullRobot, constructed probe states, RECORDED.

This is NOT a solution (solve.py — torque-servo align, shuffleboard push through the
gate, torque-servo stow, all applied forces, zero teleports — is the acceptance
evidence that the rubric ACCEPTS a correct outcome; it passes on forge seeds 0/1/2).
Every teleport here is instrumentation that CONSTRUCTS a wrong (or partial) outcome as
a settled state and asserts the rubric REJECTS it — plus mechanism-reality probes that
prove the carousel is a working instrument (a z-torque spins it; the loaded platform
CARRIES the plate; the roof genuinely blocks the seed's from-above strategy). No probe
in this battery ever reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset settles finite, plate flat on the shelf, blue bay
                            starts MISALIGNED (>= 40 deg by readback), score ~0;
  3-4. randomization      — READBACK over 8 seeded resets: the initial blue-bay offset
                            varies (both signs, wide spread, always inside the
                            configured 55..180 deg band, view yaw agrees with the
                            cached sample); the plate's shelf xy + yaw jitter is real;
  5.  null policy         — 240 idle steps -> score ~0, no success (NOTE: the blue bay
                            starts stowed=True — stow alone earns NOTHING);
  6.  wrong bay seated    — the seed-analog "just insert the plate" without aligning:
                            plate seated in the bay CURRENTLY facing the gate (not
                            blue) -> entered but NOT in_blue, NOT success, score<=0.11;
  7.  mechanism carries   — a constant z-torque spins the loaded carousel ~90 deg (yaw
                            readback) and the plate RIDES (carousel-frame pose nearly
                            unchanged, world pose moved): a working lazy susan;
  8.  wrong bay stowed    — after that spin the WRONG bay is stowed and settled ->
                            still NOT success, score <= 0.11 (color binding matters);
  9.  aligned, no insert  — blue bay parked AT the gate, plate left on the shelf ->
                            aligned credit only (score <= 0.16), NOT success;
  10. jammed in the gate  — plate half through the gate straddling the threshold
                            (r ~ 0.135, outside both entered_r and r_hi) -> NOT
                            entered, NOT in_blue, NOT success;
  11. roof drop           — the SEED's strategy (insert from above): plate dropped
                            over the blue bay lands on the covered top (roof /
                            compass rose, z readback) and the z window rejects it
                            -> NOT success;
  12. in blue, not stowed — plate seated in the blue bay with the bay still AT the
                            gate: in_blue True yet stowed False -> NOT success, score
                            capped at 0.55 (the partial-credit ceiling);
  13. latched credit      — removing that plate to the shelf keeps the latched 0.55
                            while live in_blue drops — score never decreases;
  14. settle gate         — plate in the blue bay, bay stowed, but the carousel still
                            SPINNING (w readback above the gate) -> NOT success until
                            ring-down; the probe is dismantled before it can settle;
  15. rejection audit     — success() was never True at ANY judged point;
  16. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.put_plate_in_colored_dish_rack_i86.smoke \
    --headless
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
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

BLUE_BAY = scene_mod.BLUE_BAY

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.carousel_dish_rack")().build(num_envs=args.num_envs,
                                                        device=device)
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
        env.sim.set_camera_view(tuple(np.array((0.80, -0.62, 0.55)) + o),
                                tuple(np.array((0.04, 0.00, 0.10)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video",
              flush=True)

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

    def off_deg() -> float:
        return math.degrees(float(scene.bay_offset()[0]))

    def car_w() -> float:
        return float(scene.carousel.data.root_ang_vel_w[0, 2])

    def report(tag: str) -> None:
        s, ok = judge()
        loc = scene.plate_local()[0]
        print(f"[smoke] {tag:16s} | offset={off_deg():+7.2f} w={car_w():+.3f} | "
              f"plate_loc=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
              f"{float(loc[2]):+.3f}) r={float(loc[:2].norm()):.3f} | "
              f"ent={bool(scene.entered()[0])} inb={bool(scene.in_blue_bay()[0])} "
              f"alg={bool(scene.aligned()[0])} stw={bool(scene.stowed()[0])} "
              f"stl={bool(scene.settled()[0])} | score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def write_plate(world_env, quat=None, lin_vel=None, settle_steps: int = 60) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.as_tensor(world_env, device=device, dtype=torch.float32) \
            + scene.env_origins[0]
        st[:, 3] = 1.0
        if quat is not None:
            st[:, 3:7] = torch.as_tensor(quat, device=device, dtype=torch.float32)
        if lin_vel is not None:
            st[:, 7:10] = torch.as_tensor(lin_vel, device=device, dtype=torch.float32)
        scene.plate.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def seat_in_bay(theta_car: float, r: float = 0.095, drop: float = 0.015,
                    settle_steps: int = 240) -> None:
        """Drop the plate `drop` above the floor of the bay whose wedge-center
        carousel-frame angle is `theta_car` (LIVE carousel pose) — gravity seats it."""
        cp = scene.carousel.data.root_pos_w[0]
        cq = scene.carousel.data.root_quat_w[0]
        local = torch.tensor([r * math.cos(theta_car), r * math.sin(theta_car),
                              c.plate_rest_z + drop], device=device)
        world = cp + quat_apply(cq.unsqueeze(0), local.unsqueeze(0))[0]
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = world
        st[:, 3:7] = cq
        scene.plate.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def write_carousel_yaw(yaw: float, w: float = 0.0, settle_steps: int = 0) -> None:
        """Write the hinge DOF (yaw about the spawn-authored axle — consistent with
        the joint; the kinematic housing is NEVER touched)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.env_origins[0]
        st[:, 2] += c.h0
        st[:, 3] = math.cos(yaw / 2)
        st[:, 6] = math.sin(yaw / 2)
        st[:, 12] = w
        scene.carousel.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def to_shelf(settle_steps: int = 60) -> None:
        write_plate((c.plate_spawn_x, 0.0, c.h0 + 0.001 + c.plate_h / 2 + 0.002),
                    settle_steps=settle_steps)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    bodies = (scene.housing, scene.carousel, scene.plate)
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    pz = float((scene.plate.data.root_pos_w - scene.env_origins)[0, 2])
    check("settle: all states finite, plate flat on the shelf "
          f"(z={pz:.3f} ~ {c.h0 + 0.001 + c.plate_h / 2:.3f}), blue bay starts "
          f"MISALIGNED (offset={off_deg():+.1f} deg), everything settled",
          fin and abs(pz - (c.h0 + 0.001 + c.plate_h / 2)) < 0.01
          and abs(off_deg()) >= 40.0 and bool(scene.settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(20)
        off_cache = math.degrees(float(scene.offset0[0]))
        off_live = off_deg()
        p = (scene.plate.data.root_pos_w - scene.env_origins)[0]
        q = scene.plate.data.root_quat_w[0]
        yawp = math.degrees(math.atan2(
            2 * (float(q[0]) * float(q[3]) + float(q[1]) * float(q[2])),
            1 - 2 * (float(q[2]) ** 2 + float(q[3]) ** 2)))
        reads.append((off_cache, off_live, float(p[0]), float(p[1]), yawp))
        print(f"[smoke] seed {sd}: offset cache={off_cache:+8.2f} live={off_live:+8.2f}"
              f" plate=({float(p[0]):+.3f},{float(p[1]):+.3f}) yaw={yawp:+7.1f}",
              flush=True)
    offs = [r[1] for r in reads]
    spread = max(offs) - min(offs)
    signs = {o > 0 for o in offs}
    band_ok = all(c.min_offset_deg - 0.5 <= abs(o) <= c.max_offset_deg + 0.5
                  for o in offs)
    agree = max(abs(r[0] - r[1]) for r in reads)
    check("randomization: initial blue-bay offset varies across 8 seeded resets "
          f"(spread {spread:.0f} deg, both signs seen: {len(signs) == 2}), always in "
          f"the [{c.min_offset_deg:.0f},{c.max_offset_deg:.0f}] deg band, view yaw "
          f"agrees with the cached sample (max delta {agree:.2f} deg)",
          spread > 60.0 and len(signs) == 2 and band_ok and agree < 1.0)
    xs = [r[2] for r in reads]
    ys = [r[3] for r in reads]
    yws = [r[4] for r in reads]
    check("randomization: plate shelf jitter is real (readback: x spread "
          f"{(max(xs) - min(xs)) * 1000:.0f} mm, y spread "
          f"{(max(ys) - min(ys)) * 1000:.0f} mm, yaw spread "
          f"{max(yws) - min(yws):.0f} deg)",
          (max(xs) - min(xs)) > 0.008 and (max(ys) - min(ys)) > 0.008
          and (max(yws) - min(yws)) > 20.0)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps (the blue bay "
          "starts stowed=True — stow ALONE earns nothing)",
          s <= 0.02 and not ok and bool(scene.stowed()[0]))

    # ============ 6-8. wrong bay seated -> mechanism carries -> wrong bay stowed ============
    env.reset(seed=41)
    step(30)
    # The seed-analog shortcut: insert into whatever bay faces the gate, skip aligning.
    # Gate direction in the carousel frame = -carousel_yaw; bay k center at k*90 deg.
    cy = float(scene._yaw(scene.carousel.data.root_quat_w)[0])
    th_gate = math.atan2(math.sin(-cy), math.cos(-cy))
    k_gate = round(th_gate / (math.pi / 2)) % 4
    assert k_gate != BLUE_BAY, "reset guarantees the gate-facing bay is not blue"
    seat_in_bay(k_gate * math.pi / 2)
    report("wrong-bay-seated")
    s, ok = judge()
    check(f"wrong bay seated (seed-analog, no aligning): plate rests in bay {k_gate} "
          "(the one at the gate) -> entered=True yet in_blue=False, NOT success, "
          f"score<=0.11 (got {s:.2f})",
          bool(scene.entered()[0]) and not bool(scene.in_blue_bay()[0]) and not ok
          and s <= 0.11)

    # Mechanism reality: a constant z-torque spins the LOADED carousel; the plate rides.
    loc_before = scene.plate_local()[0].clone()
    yaw_before = float(scene._yaw(scene.carousel.data.root_quat_w)[0])
    pw_before = scene.plate.data.root_pos_w[0].clone()
    sgn = 1.0 if off_deg() > 0 else -1.0  # spin AWAY from alignment (audit safety)
    t = torch.zeros(n, 1, 3, device=device)
    t[0, 0, 2] = sgn * 0.12
    spun, prev = 0.0, yaw_before
    for _ in range(240):  # fine-grained unwrapped yaw tracking (no 90-deg aliasing)
        # bang-bang speed cap: bounded coast-down (~1.0/damping rad) after the cut,
        # so the spin can NEVER sweep the blue bay through alignment (audit safety)
        drive = t if abs(car_w()) < 1.2 else zero_w
        scene.carousel.set_external_force_and_torque(zero_w, drive, env_ids=all_ids,
                                                     is_global=True)
        step(5)
        now = float(scene._yaw(scene.carousel.data.root_quat_w)[0])
        spun += abs(math.degrees(math.atan2(math.sin(now - prev),
                                            math.cos(now - prev))))
        prev = now
        if spun >= 90.0:
            break
    scene.carousel.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids,
                                                 is_global=True)
    for _ in range(20):  # ring-down (angular damping) until genuinely settled
        step(60)
        if bool(scene.settled()[0]):
            break
    loc_after = scene.plate_local()[0]
    carried = float((loc_after[:2] - loc_before[:2]).norm())
    moved_w = float((scene.plate.data.root_pos_w[0, :2] - pw_before[:2]).norm())
    report("mech-spin")
    check("mechanism reality: a 0.12 N*m z-torque spins the loaded carousel "
          f"{spun:.0f} deg (yaw readback) and the platform CARRIES the plate "
          f"(carousel-frame drift {carried * 1000:.0f} mm, world motion "
          f"{moved_w * 1000:.0f} mm) — a working lazy susan",
          spun >= 60.0 and carried < 0.02 and moved_w > 0.06)
    s, ok = judge()
    check("wrong bay stowed: after the spin the loaded WRONG bay is stowed and "
          f"settled -> still NOT success, score<=0.11 (got {s:.2f}) — the color "
          "binding matters",
          bool(scene.stowed()[0]) and bool(scene.settled()[0]) and not ok
          and s <= 0.11 and not bool(scene.in_blue_bay()[0]))

    # =========================== 9-10. aligned-no-insert, jammed ============================
    env.reset(seed=51)
    step(30)
    write_carousel_yaw(-math.pi, settle_steps=90)  # blue bay parked AT the gate
    report("aligned-shelf")
    s, ok = judge()
    check("stopped halfway: blue bay parked AT the gate, plate still on the shelf -> "
          f"aligned credit only (score {s:.2f} <= 0.16), NOT success",
          bool(scene.aligned()[0]) and not ok and 0.10 <= s <= 0.16
          and not bool(scene.entered()[0]))
    # Jam the plate half through the gate: straddling the platform-edge threshold.
    write_plate((0.135, 0.0, c.h0 + c.plate_h / 2 + 0.004), settle_steps=150)
    report("jammed-gate")
    s, ok = judge()
    loc = scene.plate_local()[0]
    check("jammed in the gate: plate straddles the threshold at r="
          f"{float(loc[:2].norm()):.3f} (> entered_r={c.entered_r:.3f}) -> NOT "
          "entered, NOT in_blue, NOT success",
          not bool(scene.entered()[0]) and not bool(scene.in_blue_bay()[0]) and not ok)

    # =========================== 11. the seed's strategy: from above ========================
    env.reset(seed=71)
    step(30)
    # Drop the plate from above the BLUE bay center — the seed's insert-from-above.
    cy = float(scene._yaw(scene.carousel.data.root_quat_w)[0])
    th_blue_w = cy + math.pi  # blue wedge center direction in world
    write_plate((0.095 * math.cos(th_blue_w), 0.095 * math.sin(th_blue_w),
                 c.h0 + 0.30), settle_steps=240)
    report("roof-drop")
    s, ok = judge()
    loc = scene.plate_local()[0]
    check("covered top blocks the seed's strategy: a plate dropped from above the "
          f"blue bay lands on the roof/compass rose (z_loc={float(loc[2]):+.3f} vs "
          f"window [{c.z_lo:.3f},{c.z_hi:.3f}]) -> NOT in_blue, NOT entered, NOT "
          "success",
          float(loc[2]) > c.z_hi and not bool(scene.in_blue_bay()[0])
          and not bool(scene.entered()[0]) and not ok)

    # ============ 12-14. in-blue-not-stowed, latched credit, the settle gate ================
    env.reset(seed=61)
    step(30)
    write_carousel_yaw(-math.pi, settle_steps=60)  # blue bay AT the gate
    seat_in_bay(math.pi)  # plate onto the BLUE bay floor (gravity seats it)
    report("blue-not-stowed")
    s_before, ok = judge()
    check("out of order: plate ON the blue bay floor but the bay still AT the gate "
          f"(stowed=False) -> NOT success, score {s_before:.2f} <= 0.56 "
          "(the partial-credit ceiling)",
          bool(scene.in_blue_bay()[0]) and not bool(scene.stowed()[0]) and not ok
          and s_before <= 0.56)
    to_shelf(settle_steps=60)
    report("latch-remove")
    s_after, ok = judge()
    check("latched credit: removing the plate back to the shelf keeps the latched "
          f"score ({s_before:.2f} -> {s_after:.2f}) while live in_blue drops",
          abs(s_after - s_before) < 1e-3 and not bool(scene.in_blue_bay()[0])
          and not ok)
    # Settle gate: blue loaded AND stowed but the carousel still SPINNING.
    w_spin = 0.9
    yaw_stow = -math.pi / 2  # offset +90 deg
    write_carousel_yaw(yaw_stow, w=w_spin)
    cq = scene.carousel.data.root_quat_w[0]
    cp = scene.carousel.data.root_pos_w[0]
    local = torch.tensor([0.095 * math.cos(math.pi), 0.095 * math.sin(math.pi),
                          c.plate_rest_z + 0.002], device=device)
    world = (cp + quat_apply(cq.unsqueeze(0), local.unsqueeze(0))[0]
             - scene.env_origins[0])
    # tangential velocity of that bay point (rides the spin, no impact transient)
    rvec = quat_apply(cq.unsqueeze(0), local.unsqueeze(0))[0]
    vel = (float(-w_spin * rvec[1]), float(w_spin * rvec[0]), 0.0)
    write_plate(tuple(float(v) for v in world), quat=tuple(float(v) for v in cq),
                lin_vel=vel, settle_steps=2)
    report("spinning-judge")
    s, ok = judge()
    w_live = abs(car_w())
    check("settle gate: plate in the BLUE bay, bay STOWED, but the carousel still "
          f"spinning (|w|={w_live:.2f} > gate {c.settle_car:.2f}) -> NOT success "
          "until the ring-down completes",
          not ok and w_live > c.settle_car and not bool(scene.settled()[0]))
    # Dismantle the probe BEFORE it can ring down into success (audit safety).
    to_shelf(settle_steps=10)
    write_carousel_yaw(yaw_stow, w=0.0, settle_steps=30)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.carousel_dish_rack")
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

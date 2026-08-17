"""Smoke / rubric-REJECTION battery for BottleSpringBayScene (sim_gen task
`stack_wine_i354`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — pitch the bottle into the bay, press the neck
into the sprung cup past the lip, lower, release so the spring seats the base under
the lip — is the acceptance evidence that the rubric ACCEPTS a correct outcome).
Every construct here builds a wrong (or partial) outcome and asserts the rubric
REJECTS it; no probe in this battery ever reaches success(), and a final audit
check asserts exactly that.

  1-2.  settle/no-NaN     — reset layout settles finite: bottle standing on the
                            floor at CoM height, both plungers at their extended
                            homes (compression ~0), settled; score ~0, no success;
  3.    randomization     — READBACK over 8 seeded resets: the TARGET BAY takes
                            both values and the beacon tile matches it; bottle
                            spawn xy and yaw vary;
  4.    spring principle  — PhysX MASS readback matches the authored bottle and
                            plunger masses, and a plunger teleported to deep
                            compression RETURNS to its home under the post_step
                            spring (the mechanism is live physics, not
                            bookkeeping);
  5.    null policy       — 240 idle steps -> score ~0, no success;
  6.    SEED strategy     — the seed's plan transplanted verbatim ("grasp the
                            bottle, LAY it on the rack"): bottle laid across the
                            TOPS of the bay walls, settled — support-from-below is
                            exactly what this task rejects: NOT success,
                            score <= 0.10;
  7.    wrong-bay clamp   — the full spring-retained clamp CONSTRUCTED in the
                            NON-target bay: it stays physically clamped (seat-band
                            compression readback holds through 2 s of contact —
                            retention is the spring, not bookkeeping) yet NOT
                            success (the beacon names the other bay);
  8.    base-first        — the bottle inserted REVERSED (base toward the cup):
                            the 60 mm body cannot enter the 34 mm cup opening, so
                            it rests pressed at ~c_rev = 30 mm compression, far
                            outside the seat band — NOT success;
  9.    held-press        — the mid-cycle state merely HELD at full press by a
                            constant external force (compression ~c_need, neck in
                            the cup): partial stage credit only (<= 0.60), NOT
                            success while held — release is load-bearing;
  10.   dropped-at-mouth  — the bottle dropped pitched at the cup mouth: gravity
                            can drive the neck in (compression readback may even
                            reach the seat band) but the base stays PERCHED on the
                            lip / wall top — the seat-plane and alignment clauses
                            reject the spoof, NOT success (the clamp needs the
                            full press-lower-release cycle, not just contact);
  11.   standing-in-bay   — the bottle STANDING upright on the deck inside the
                            target bay, xy aligned: NOT success (no clamp);
  12-13. partial + latch  — a lift above `lifted_z` earns exactly the lifted stage
                            credit (~0.10), NOT success; removing the bottle to
                            the far floor leaves the latched credit unchanged;
  14.   rejection audit   — success() was never True at ANY judged point;
  15.   final no-NaN      — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.stack_wine_i354.smoke --headless
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

# Level bottle, neck toward -x (the seating direction): rotate about +y by -90 deg.
Q_NECK_NEG_X = (math.cos(-math.pi / 4), 0.0, math.sin(-math.pi / 4), 0.0)
# Reversed: neck toward +x (base toward the cup).
Q_NECK_POS_X = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)
# Across the bays: local +z -> world -y (rotate about +x by +90 deg).
Q_ACROSS = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.bottle_spring_bay")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((0.72, -0.72, 0.52)) + o),
                                tuple(np.array((-0.05, 0.0, 0.06)) + o),
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
        return bool(torch.isfinite(scene.rack.data.root_state_w).all()
                    and torch.isfinite(scene.bottle.data.root_state_w).all()
                    and all(torch.isfinite(p.data.root_state_w).all()
                            for p in scene.plungers))

    def report(tag: str) -> None:
        s, ok = judge()
        cm = scene.compression()[0]
        print(f"[smoke] {tag:18s} | bay={int(scene.target_bay[0])} "
              f"comp=({float(cm[0]):+.4f},{float(cm[1]):+.4f}) "
              f"seated={bool(scene.seated_now()[0])} settled={bool(scene.settled()[0])} "
              f"latch(l/n/p/s)=({float(scene.lifted_latch[0]):.0f},"
              f"{float(scene.neckin_latch[0]):.0f},{float(scene.pressed_latch[0]):.0f},"
              f"{float(scene.seated_latch[0]):.0f}) max_seat_ctr="
              f"{int(scene.max_seat_ctr[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def bay() -> int:
        return int(scene.target_bay[0])

    def bay_y(k: int) -> float:
        return (-1.0, 1.0)[k] * c.bay_dy

    def comp(k: int) -> float:
        return float(scene.compression()[0, k])

    def bottle_env() -> torch.Tensor:
        return scene.bottle.data.root_pos_w[0] - scene.env_origins[0]

    def place_bottle(pos_env, quat=(1.0, 0.0, 0.0, 0.0), settle_steps: int = 30) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap). `pos_env` is the ROOT position."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = float(pos_env[0])
        st[:, 1] = float(pos_env[1])
        st[:, 2] = float(pos_env[2])
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += scene.env_origins[0]
        scene.bottle.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def place_plunger(k: int, compression: float) -> None:
        """Teleport plunger k along its own DOF (within the joint limits)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene._plunger_home[:, k]
        st[:, 0] -= float(compression)
        st[:, 3] = 1.0
        scene.plungers[k].write_root_state_to_sim(st, all_ids)

    # =========================== 1-2. settle / no-NaN ==========================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    z0 = float(bottle_env()[2])
    check("settle: states finite; bottle standing on the floor at CoM height "
          f"(readback z={z0:.3f}), both plungers at home (comp="
          f"{comp(0):+.4f},{comp(1):+.4f}), settled",
          finite() and abs(z0 - c.com_z) < 0.006
          and abs(comp(0)) < 0.003 and abs(comp(1)) < 0.003
          and bool(scene.settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3. randomization is real ======================================
    bays, beacon_ok, spawns, yaws = [], True, [], []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(5)
        b = bay()
        bays.append(b)
        by = float((scene.beacon.data.root_pos_w[0] - scene.env_origins[0])[1])
        beacon_ok = beacon_ok and abs(by - bay_y(b)) < 0.005
        p = bottle_env()
        spawns.append((float(p[0]), float(p[1])))
        yaws.append(float(scene.bottle.data.root_quat_w[0, 3]))
    sarr = np.array(spawns)
    print(f"[smoke] target bays across seeds: {bays} (beacon matched: {beacon_ok})",
          flush=True)
    print(f"[smoke] bottle spawn xy across seeds:\n{sarr}", flush=True)
    print(f"[smoke] bottle spawn qz across seeds: {[f'{v:+.2f}' for v in yaws]}", flush=True)
    check("randomization: the target bay takes BOTH values across seeded resets with "
          "the beacon tile matching it, and the bottle spawn xy + yaw vary (readback)",
          len(set(bays)) == 2 and beacon_ok
          and float((sarr.max(0) - sarr.min(0)).max()) > 0.02
          and (max(yaws) - min(yaws)) > 0.2)

    # =========================== 4. the spring principle is physical ===========================
    env.reset(seed=31)
    step(10)
    m_b = float(scene.bottle.root_physx_view.get_masses().cpu().view(-1)[0])
    m_p = [float(p.root_physx_view.get_masses().cpu().view(-1)[0]) for p in scene.plungers]
    print(f"[smoke] PhysX mass readback: bottle={m_b:.3f} plungers={m_p}", flush=True)
    place_plunger(0, 0.030)
    step(2)
    c_press = comp(0)
    step(120)  # 1 s: the post_step spring must drive it home
    report("spring-return")
    check("spring principle: authored masses read back "
          f"(bottle {m_b:.3f} kg, plunger {m_p[0]:.3f} kg), and a plunger teleported "
          f"to {c_press * 1000:.0f} mm compression RETURNS to its extended home under "
          f"the spring (residual comp={comp(0) * 1000:+.1f} mm) — live physics",
          abs(m_b - c.mass) < 1e-3 and all(abs(m - c.plunger_mass) < 1e-3 for m in m_p)
          and c_press > 0.024 and abs(comp(0)) < 0.004)

    # =========================== 5. null policy fails ==========================================
    env.reset(seed=41)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. SEED strategy (lay it ON the rack) =========================
    # The seed's whole plan is "grasp the bottle, LAY it on the rack" — object resting
    # ON TOP of the fixture. Transplant it verbatim: the bottle laid horizontally
    # ACROSS the tops of the bay side walls, settled.
    env.reset(seed=51)
    step(10)
    top_z = c.deck_t + c.wall_h + c.body_r
    place_bottle((0.02, 0.0, top_z + 0.002), quat=Q_ACROSS, settle_steps=300)
    report("seed-on-top")
    s, ok = judge()
    z_rb = float(bottle_env()[2])
    check("SEED strategy: the bottle laid ON TOP of the bay walls (readback "
          f"z={z_rb:.3f} vs seat height {c.axis_h:.3f}), settled — support-from-below "
          f"is NOT a spring clamp: NOT success, score <= 0.10 (got {s:.3f})",
          abs(z_rb - top_z) < 0.010 and bool(scene.settled()[0]) and not ok and s <= 0.10)

    # =========================== 7. wrong-bay clamp (physical, still rejected) =================
    env.reset(seed=61)
    step(10)
    wrong = 1 - bay()
    y_w = bay_y(wrong)
    place_plunger(wrong, c.seat_c + 0.001)
    tip_x = c.x_cb0 - c.seat_c - 0.0005  # base 0.5 mm short of the wall; the spring
    # closes the gap and presses the base onto the wall (never spawn interpenetrating)
    place_bottle((tip_x + c.tip_off, y_w, c.axis_h + 0.0005), quat=Q_NECK_NEG_X,
                 settle_steps=240)  # 2 s of real spring-loaded contact
    report("wrong-bay-clamp")
    s, ok = judge()
    base_x = float(bottle_env()[0]) + c.com_z  # base = root + com_z along +x (axis=-x)
    cw = comp(wrong)
    check("wrong-bay clamp: the full spring-retained clamp constructed in the "
          f"NON-target bay stays physically clamped through 2 s (comp readback "
          f"{cw * 1000:+.1f} mm in the seat band, base_x={base_x:+.4f} at the wall "
          f"{c.x_wall:+.4f}) — retention is the spring, yet NOT success (wrong bay)",
          c.c_lo <= cw <= c.c_hi and base_x >= c.x_wall - c.base_wall_tol - 0.002
          and not ok)

    # =========================== 8. base-first insertion =======================================
    env.reset(seed=71)
    step(10)
    y_t = bay_y(bay())
    place_plunger(bay(), c.c_rev + 0.001)
    # base face 0.5 mm in front of the pressed cup RIM (the 60 mm body cannot enter
    # the 34 mm opening); neck toward the end wall.
    base_x0 = (c.x_cb0 - c.c_rev - 0.001) + c.cup_d + 0.0005
    place_bottle((base_x0 + c.com_z, y_t, c.axis_h + 0.0005), quat=Q_NECK_POS_X,
                 settle_steps=240)
    report("base-first")
    s, ok = judge()
    cr = comp(bay())
    check("base-first insertion: the reversed bottle rests pressed at "
          f"comp={cr * 1000:+.1f} mm (~c_rev={c.c_rev * 1000:.0f} mm), far outside the "
          f"seat band [{c.c_lo * 1000:.0f}, {c.c_hi * 1000:.0f}] mm — NOT success",
          cr >= c.c_hi + 0.008 and cr <= c.travel - 0.002 and not ok)

    # =========================== 9. held-press near-miss =======================================
    env.reset(seed=81)
    step(10)
    y_t = bay_y(bay())
    hold_c = c.c_need + 0.002
    place_plunger(bay(), hold_c + 0.0005)
    tip_x = c.x_cb0 - hold_c
    place_bottle((tip_x + c.tip_off, y_t, c.axis_h + 0.0005), quat=Q_NECK_NEG_X,
                 settle_steps=0)
    f_hold = torch.tensor([-c.spring_k * hold_c, 0.0, 0.0], device=device)
    for _ in range(180):  # hold pressed by a constant axial force (body-frame encode)
        q = scene.bottle.data.root_quat_w[0]
        fb = quat_apply_inverse(q.unsqueeze(0), f_hold.unsqueeze(0))[0]
        scene.bottle.set_external_force_and_torque(
            fb.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            torch.zeros(n, 1, 3, device=device), env_ids=all_ids)
        step(1)
    report("held-press")
    s, ok = judge()
    ch = comp(bay())
    held_ok = ch >= c.press_c and not ok and s <= 0.60
    # Remove the bottle FIRST, then release the force (a bare release would seat it —
    # that is the SOLUTION, demonstrated by solve.py, not by this battery).
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1], st[:, 2] = -0.45, -0.22, c.com_z + 0.002
    st[:, 3] = 1.0
    st[:, 0:3] += scene.env_origins[0]
    scene.bottle.write_root_state_to_sim(st, all_ids)
    zw = torch.zeros(n, 1, 3, device=device)
    scene.bottle.set_external_force_and_torque(zw, zw, env_ids=all_ids)
    step(60)
    check("held-press: the mid-cycle state merely HELD at full press (comp readback "
          f"{ch * 1000:+.1f} mm >= press_c={c.press_c * 1000:.0f} mm, neck in the cup) "
          f"earns stage credit only (score {s:.3f} <= 0.60) and is NOT success — "
          "release is load-bearing", held_ok)

    # =========================== 10. no-compression perch ======================================
    env.reset(seed=91)
    step(10)
    y_t = bay_y(bay())
    th = math.radians(20.0)
    q_p = (math.cos(-(math.pi / 2 + th) / 2), 0.0, math.sin(-(math.pi / 2 + th) / 2), 0.0)
    # tip just in front of the cup face, base up over the end wall; dropped, not pressed
    tip = np.array([c.x_cb0 + c.cup_d + 0.002, y_t, c.axis_h + 0.025])
    a = np.array([-math.cos(th), 0.0, -math.sin(th)])
    root = tip - a * c.tip_off
    place_bottle(tuple(root), quat=q_p, settle_steps=300)
    report("no-press-perch")
    s, ok = judge()
    cp = comp(bay())
    pz = float(bottle_env()[2])
    check("dropped-at-the-mouth perch: a bottle dropped pitched at the cup mouth "
          f"settles PERCHED — gravity may even press the neck in (settled comp="
          f"{cp * 1000:+.1f} mm) but the base stays up on the lip/wall (CoM z "
          f"readback {pz:.3f} vs seat height {c.axis_h:.3f}) — the seat-plane and "
          "alignment clauses reject it: NOT success",
          not ok and (cp < c.c_lo or (pz - c.axis_h) > c.z_tol))

    # =========================== 11. standing in the bay =======================================
    env.reset(seed=101)
    step(10)
    y_t = bay_y(bay())
    place_bottle((0.02, y_t, c.deck_t + c.com_z + 0.002), settle_steps=120)
    report("standing-in-bay")
    s, ok = judge()
    p = bottle_env()
    check("standing-in-bay: the bottle standing upright on the deck inside the "
          f"target bay (readback z={float(p[2]):.3f}, y={float(p[1]):+.3f}) is no "
          "clamp — NOT success",
          abs(float(p[2]) - (c.deck_t + c.com_z)) < 0.008
          and abs(float(p[1]) - y_t) < 0.02 and not ok)

    # =========================== 12-13. partial credit + latched credit ========================
    env.reset(seed=111)
    step(10)
    place_bottle((-0.30, 0.0, c.lifted_z + 0.06), settle_steps=90)  # lift -> falls back
    report("lifted-only")
    s_in, ok = judge()
    check("partial credit: lifting the bottle above lifted_z earns exactly the "
          f"lifted stage credit (score {s_in:.3f} in [0.08, 0.12]), NOT success",
          0.08 <= s_in <= 0.12 and not ok)
    place_bottle((-0.45, 0.25, c.com_z + 0.002), settle_steps=60)
    report("lift-removed")
    s_out, ok = judge()
    check("latched credit: parking the bottle on the far floor leaves the latched "
          f"credit unchanged ({s_in:.3f} -> {s_out:.3f}), success stays gone",
          abs(s_out - s_in) < 0.02 and not ok)

    # =========================== 14-15. audit + no-NaN =========================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite())

    # =========================== save + verdict ================================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.bottle_spring_bay")
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

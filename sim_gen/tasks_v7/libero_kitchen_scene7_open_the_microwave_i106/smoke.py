"""Smoke / rubric-REJECTION battery for MicrowaveCarouselScene (sim_gen task
`libero_kitchen_scene7_open_the_microwave_i106`) — NullRobot, teleported probe states
+ a torque probe, RECORDED.

This is NOT a solution (solve.py — PD torque servo on the platter, cargo riding on
friction — is the acceptance evidence that the rubric ACCEPTS a correct outcome; it
passes on seeds 0/1). Every teleport here is instrumentation that CONSTRUCTS a wrong
(or partial) outcome as a settled state and asserts the rubric REJECTS it. Probes that
move the carousel move the WHOLE linkage (platter + both riding vessels) in one
batched write before stepping — teleporting one body of a linkage gets depenetrated
back. No probe in this battery ever reaches success() at a judged point, and a final
audit asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: both vessels upright ON
                            their platter slots, cup >= ~90 deg off the doorway;
                            score ~0 at rest, no success;
  3-4. randomization      — READBACK over 6 seeded resets: platter yaw, the cup's
                            signed start bearing, and the bottle's signed separation
                            all vary; every reset spawns riding-intact and far
                            off-door;
  5.  null policy         — 240 idle steps -> score ~0, no success. The SEED task's
                            own end state (microwave door wide open, contents
                            untouched) IS this start state, so the seed's strategy
                            earns ~0 by the same check;
  6.  mechanism sanity    — solve's capped rim torque (0.012 N m), pointed AWAY from
                            the doorway, really turns the loaded platter (>= 5 deg)
                            and the cargo RIDES it on friction (slot invariance held,
                            cup transported in the world) — the actuator is
                            non-vacuous and the transport mechanism is real; still
                            score ~0 (rotated the wrong way on purpose);
  7.  drag cheat          — the cup teleported straight to the doorway onto a
                            DIFFERENT spot of the (unrotated) platter: bearing
                            perfect, slot invariance broken -> rejected, score ~0
                            (milestones gated on riding never latched);
  8.  wrong object        — the whole carousel rotated so the BLUE BOTTLE faces the
                            doorway: identity, not geometry — score ~0, no success;
  9.  tipped cup          — carousel rotated so the cup's slot faces the doorway but
                            the cup lies ON ITS SIDE there -> uprightness rejects it,
                            score ~0;
  10. knocked bottle      — cup upright and riding at the doorway (otherwise a win)
                            but the bottle knocked over on its slot -> the care
                            constraint rejects it, score ~0;
  11. cup off platter     — cup standing on the cavity FLOOR at the doorway bearing
                            (radius outside the platter) -> slot invariance rejects
                            it, score ~0;
  12. near-miss           — carousel legitimately rotated to 28 deg off (outside the
                            20 deg window): NOT success, milestone credit only
                            (score latches ~0.45);
  13. settle gate         — cup aligned and riding but the whole carousel still
                            coherently spinning (2 rad/s) is NOT success (velocity
                            gates are real); rotated away before it can settle;
  14. latched credit      — rotating the carousel far off afterwards leaves the
                            latched score unchanged while the bearing is ruined;
  15. rejection audit     — success() was never True at ANY judged point;
  16. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene7_open_the_microwave_i106.smoke --headless
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

TAU_CAP = 0.012  # solve's torque cap (the fingertip-scale rim push)


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.microwave_carousel")().build(num_envs=args.num_envs,
                                                        device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    top = c.platter_z + c.platter_t / 2

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        # In front of the open doorway, offset to -y so the parked-open door
        # (which hangs off the +y hinge) never blocks the sightline into the cavity.
        env.sim.set_camera_view(tuple(np.array((-1.35, -0.35, 0.72)) + o),
                                tuple(np.array((0.55, 0.00, 0.38)) + o),
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

    def cup_err() -> float:
        return float(scene.cup_bearing_err()[0])

    def bottle_err() -> float:
        d = scene.bottle.data.root_pos_w[0, 0:2] - scene._center_xy()[0]
        return wrap(math.atan2(float(d[1]), float(d[0])) - math.pi)

    def slot_errs() -> tuple[float, float]:
        return (float(scene._slot_err(scene.cup, scene.cup_slot)[0]),
                float(scene._slot_err(scene.bottle, scene.bottle_slot)[0]))

    def report(tag: str) -> None:
        s, ok = judge()
        se_c, se_b = slot_errs()
        print(f"[smoke] {tag:16s} | err={math.degrees(cup_err()):+7.1f}deg "
              f"psi={math.degrees(float(scene.platter_yaw()[0])):+7.1f}deg "
              f"bot={math.degrees(bottle_err()):+7.1f}deg "
              f"slot=({se_c * 1000:.1f},{se_b * 1000:.1f})mm "
              f"riding={bool(scene.riding_ok()[0])} settled={bool(scene.settled()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def yaw_mul(dphi: float, q: torch.Tensor) -> tuple[float, float, float, float]:
        """q_z(dphi) * q for one (w,x,y,z) quaternion."""
        ch, sh = math.cos(dphi / 2), math.sin(dphi / 2)
        w, x, y, z = (float(v) for v in q)
        return (ch * w - sh * z, ch * x - sh * y, ch * y + sh * x, ch * z + sh * w)

    def place(body, xyz, quat=None, lin_vel=None, ang_vel=None) -> None:
        """One-body kinematic write (NO stepping here — batch linkage writes first)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor(xyz, device=device)
        st[:, 3:7] = torch.tensor(quat if quat is not None else (1.0, 0.0, 0.0, 0.0),
                                  device=device)
        if lin_vel is not None:
            st[:, 7:10] = torch.tensor(lin_vel, device=device)
        if ang_vel is not None:
            st[:, 10:13] = torch.tensor(ang_vel, device=device)
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def rotate_all(dphi: float, cup_tipped: bool = False, bottle_tipped: bool = False,
                   platter_w: float = 0.0, settle_steps: int = 45) -> None:
        """Probe-only: rotate the WHOLE carousel linkage (platter + both vessels) by
        `dphi` about the cavity axis — every body written BEFORE any stepping (the
        one-body-of-a-linkage teleport gets depenetrated back). Optional overrides
        construct a vessel knocked over ON its (rotated) slot, or a COHERENTLY
        spinning carousel (`platter_w`: the vessels get the matching tangential and
        angular velocity, otherwise cargo friction eats the spin within a step),
        without any intermediate judged state."""
        ctr = scene._center_xy()[0]
        cx0, cy0 = float(ctr[0]), float(ctr[1])
        co, si = math.cos(dphi), math.sin(dphi)

        def rot_xy(px: float, py: float) -> tuple[float, float]:
            dx, dy = px - cx0, py - cy0
            return (cx0 + co * dx - si * dy, cy0 + si * dx + co * dy)

        # platter (positions below are world-frame: subtract origins before place())
        org = scene.env_origins[0]
        ppos = scene.platter.data.root_pos_w[0]
        pq = yaw_mul(dphi, scene.platter.data.root_quat_w[0])
        place(scene.platter,
              (float(ppos[0] - org[0]), float(ppos[1] - org[1]), c.platter_z),
              quat=pq, ang_vel=(0.0, 0.0, platter_w))
        # vessels
        for body, tipped, half_h, rad in (
                (scene.cup, cup_tipped, c.cup_h / 2, c.cup_r),
                (scene.bottle, bottle_tipped, c.bottle_body_h / 2, c.bottle_r)):
            bp = body.data.root_pos_w[0]
            wx, wy = rot_xy(float(bp[0]), float(bp[1]))
            if tipped:
                q = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)  # on side
                z = top + rad + 0.003
            else:
                q = yaw_mul(dphi, body.data.root_quat_w[0])
                z = top + half_h + 0.002
            lv = (-platter_w * (wy - cy0), platter_w * (wx - cx0), 0.0) \
                if platter_w else None
            av = (0.0, 0.0, platter_w) if platter_w else None
            place(body, (wx - float(org[0]), wy - float(org[1]), z), quat=q,
                  lin_vel=lv, ang_vel=av)
        step(settle_steps)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    bodies = (scene.shell, scene.platter, scene.cup, scene.bottle)
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    se_c, se_b = slot_errs()
    check("settle: all states finite; both vessels upright ON their platter slots "
          f"(slot err {se_c * 1000:.1f}/{se_b * 1000:.1f} mm), cup "
          f"{abs(math.degrees(cup_err())):.0f} deg off the doorway",
          fin and bool(scene.riding_ok()[0]) and abs(math.degrees(cup_err())) >= 88.0)
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    sane = True
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(30)
        e = math.degrees(cup_err())
        sep = math.degrees(wrap(bottle_err() - cup_err()))
        psi = float(scene.platter_yaw()[0])
        sane = sane and bool(scene.riding_ok()[0]) and 88.0 <= abs(e) <= 180.0
        reads.append((psi, e, sep))
    arr = np.array(reads)
    print("[smoke] randomization readback (platter_psi_rad, cup_err_deg, "
          f"bottle_sep_deg):\n{np.round(arr, 3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: platter yaw and the cup's signed start bearing vary "
          f"(spreads psi={spread[0]:.2f} rad, err={spread[1]:.1f} deg)",
          spread[0] > 1.0 and spread[1] > 25.0)
    check("randomization: the bottle's signed separation varies "
          f"(spread {spread[2]:.1f} deg), and every reset spawns riding-intact "
          ">= 88 deg off-door", spread[2] > 20.0 and sane)

    # =========================== 5. null policy fails (= the seed's end state) ==============
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps — the SEED "
          "task's own end state (door wide open, contents untouched) is exactly this "
          "start state and earns nothing", s <= 0.02 and not ok)

    # =========================== 6. mechanism sanity (non-vacuous actuator) =================
    # Solve's capped rim torque, pointed AWAY from the doorway so no credit can latch:
    # the loaded platter must really turn and the cargo must RIDE it on friction.
    psi0 = float(scene.platter_yaw()[0])
    cup0 = scene.cup.data.root_pos_w[0, 0:2].clone()
    sgn = 1.0 if cup_err() > 0 else -1.0  # increases |err|
    t3 = torch.zeros(n, 1, 3, device=device)
    z3 = torch.zeros(n, 1, 3, device=device)
    for _ in range(120):
        t3[:, 0, 2] = sgn * TAU_CAP
        scene.platter.set_external_force_and_torque(z3, t3, env_ids=all_ids)
        step(1)
    t3[:, 0, 2] = 0.0
    scene.platter.set_external_force_and_torque(z3, z3, env_ids=all_ids)
    step(90)
    report("torque-probe")
    dpsi = abs(math.degrees(wrap(float(scene.platter_yaw()[0]) - psi0)))
    cup_moved = float((scene.cup.data.root_pos_w[0, 0:2] - cup0).norm())
    se_c, se_b = slot_errs()
    s, ok = judge()
    check("mechanism: the capped rim torque (0.012 N m) turned the loaded platter "
          f"{dpsi:.1f} deg (>= 5) and the cargo RODE it on friction (cup transported "
          f"{cup_moved * 1000:.0f} mm in the world, slot err {se_c * 1000:.1f} mm "
          "< slip tol) — non-vacuous, and rotating AWAY earns nothing",
          dpsi >= 5.0 and cup_moved > 0.005 and bool(scene.riding_ok()[0])
          and s <= 0.02 and not ok)

    # =========================== 7. drag cheat (slot invariance) ============================
    # Pick a seed whose BOTTLE is well away from the doorway (>= 55 deg; the cup at
    # the front needs ~43 deg of angular clearance to not touch it), so the
    # teleported cup lands on empty platter instead of bouncing off the bottle.
    found = False
    for sd in range(41, 61):
        env.reset(seed=sd)
        step(30)
        if abs(math.degrees(bottle_err())) > 55.0:
            found = True
            break
    print(f"[smoke] drag-cheat seed {sd}: bottle "
          f"{abs(math.degrees(bottle_err())):.0f} deg off-door (found={found})",
          flush=True)
    ctr = scene._center_xy()[0]
    org = scene.env_origins[0]
    front_xy = (float(ctr[0] - org[0]) - c.cargo_ring_r, float(ctr[1] - org[1]))
    place(scene.cup, (front_xy[0], front_xy[1], top + c.cup_h / 2 + 0.002))
    step(45)
    report("drag-cheat")
    se_c, _ = slot_errs()
    s, ok = judge()
    check("drag cheat: cup teleported straight to the doorway onto a DIFFERENT spot "
          f"of the unrotated platter — bearing {abs(math.degrees(cup_err())):.0f} deg "
          f"(perfect) but slot err {se_c * 1000:.0f} mm >> {c.slip_tol * 1000:.0f} mm "
          "-> rejected, score ~0",
          abs(math.degrees(cup_err())) < c.azimuth_tol_deg and se_c > c.slip_tol
          and s <= 0.02 and not ok)

    # =========================== 8. wrong object (identity) =================================
    env.reset(seed=51)
    step(30)
    rotate_all(-bottle_err())
    report("bottle-front")
    s, ok = judge()
    check("wrong object: carousel rotated so the BLUE BOTTLE faces the doorway "
          f"(bottle {abs(math.degrees(bottle_err())):.0f} deg, cup "
          f"{abs(math.degrees(cup_err())):.0f} deg off) — identity, not geometry: "
          "score ~0, no success",
          abs(math.degrees(bottle_err())) < 12.0
          and abs(math.degrees(cup_err())) > c.mile1_deg and s <= 0.02 and not ok)

    # =========================== 9. tipped cup at the doorway ===============================
    env.reset(seed=61)
    step(30)
    rotate_all(-cup_err(), cup_tipped=True)
    report("tipped-cup")
    s, ok = judge()
    check("tipped cup: the cup lies ON ITS SIDE at the doorway slot — uprightness "
          f"rejects it (bearing {abs(math.degrees(cup_err())):.0f} deg), score ~0",
          abs(math.degrees(cup_err())) < 35.0 and not bool(scene.riding_ok()[0])
          and s <= 0.02 and not ok)

    # =========================== 10. knocked bottle (the care constraint) ===================
    env.reset(seed=71)
    step(30)
    rotate_all(-cup_err(), bottle_tipped=True)
    report("knocked-bottle")
    se_c, _ = slot_errs()
    s, ok = judge()
    check("knocked bottle: cup upright and riding at the doorway (otherwise a win: "
          f"bearing {abs(math.degrees(cup_err())):.0f} deg, cup slot err "
          f"{se_c * 1000:.1f} mm) but the bottle knocked over -> the care constraint "
          "rejects it, score ~0",
          abs(math.degrees(cup_err())) < c.azimuth_tol_deg and se_c < c.slip_tol
          and not bool(scene.riding_ok()[0]) and s <= 0.02 and not ok)

    # =========================== 11. cup off the platter ====================================
    env.reset(seed=81)
    step(30)
    ctr = scene._center_xy()[0]
    place(scene.cup, (float(ctr[0] - org[0]) - 0.188, float(ctr[1] - org[1]),
                      c.floor_top_z + c.cup_h / 2 + 0.002))
    step(45)
    report("cup-on-floor")
    se_c, _ = slot_errs()
    s, ok = judge()
    check("cup off platter: cup standing on the cavity FLOOR at the doorway bearing "
          f"({abs(math.degrees(cup_err())):.0f} deg, slot err {se_c * 1000:.0f} mm) "
          "-> slot invariance rejects it, score ~0",
          abs(math.degrees(cup_err())) < c.azimuth_tol_deg and se_c > c.slip_tol
          and s <= 0.02 and not ok)

    # =========================== 12. near-miss (28 deg) =====================================
    env.reset(seed=91)
    step(30)
    e = cup_err()
    tgt = math.copysign(math.radians(c.azimuth_tol_deg + 8.0), e)
    rotate_all(tgt - e, settle_steps=60)
    report("near-miss")
    s_nm, ok = judge()
    check("near-miss: carousel legitimately rotated to "
          f"{abs(math.degrees(cup_err())):.0f} deg off (outside the "
          f"{c.azimuth_tol_deg:.0f} deg window) -> NOT success, milestone credit "
          f"only (score {s_nm:.2f} in [0.43, 0.46])",
          abs(math.degrees(cup_err())) > c.azimuth_tol_deg + 2.0 and not ok
          and 0.43 <= s_nm <= 0.46)

    # =========================== 13. settle gate ============================================
    rotate_all(-cup_err(), platter_w=2.0, settle_steps=2)
    w_now = abs(float(scene.platter.data.root_ang_vel_w[0, 2]))
    report("settle-gate")
    _s, ok = judge()
    gate_ok = (abs(math.degrees(cup_err())) < c.azimuth_tol_deg
               and w_now > c.platter_settle_ang and not ok)
    # rotate far away BEFORE the platter can spin down into a real success
    rotate_all(math.radians(90.0), settle_steps=45)
    check("settle gate: cup aligned and riding but the platter still spinning at "
          f"{w_now:.2f} rad/s is NOT success (velocity gates are real)", gate_ok)

    # =========================== 14. latched credit survives ================================
    report("rotated-away")
    s_after, ok = judge()
    check("latched credit: rotating the carousel far off afterwards leaves the "
          f"latched score unchanged ({s_nm:.2f} -> {s_after:.2f}) while the bearing "
          f"is ruined ({abs(math.degrees(cup_err())):.0f} deg)",
          abs(s_after - s_nm) < 1e-3 and abs(math.degrees(cup_err())) > 45.0 and not ok)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.microwave_carousel")
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

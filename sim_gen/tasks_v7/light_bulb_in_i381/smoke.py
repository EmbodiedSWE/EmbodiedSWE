"""Smoke / rubric-REJECTION battery for GuardCapScene (sim_gen task
`light_bulb_in_i381`) — NullRobot, teleported probe states + force probes, RECORDED.

This is NOT a solution (solve.py — stand the fresh bulb on the pad, key-align the
cage and servo-lower it to the flush seat — is the acceptance evidence that the
rubric ACCEPTS a correct outcome; it passes on seeds 0/1). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled state and
asserts the rubric REJECTS it — plus force probes that prove the yaw key and the
closed-lid order forcing are physically real geometry. No probe in this battery ever
reaches success(), and a final audit asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: both bulbs LYING on the
                            open floor in their bands, cage parked upright on the
                            ground, pad empty; score ~0 at rest, no success;
  3-4. randomization      — READBACK over 8 seeded resets: plinth xy + yaw vary;
                            bulb spawn varies, the dead bulb's side flips, the cage
                            park varies;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  lid denial (order)  — the seed's verb — deliver the bulb INTO the fixture from
                            above — is dead once the cage is seated: dropped over
                            the capped pad the bulb bounces off the closed lid and
                            never stands, score ~0 (cap-first can never be repaired
                            -> stand-then-cap is the only order);
  7.  yaw key             — the cage released concentric but rotated 45 deg rests ON
                            the red key bridges, ~8 mm proud — OUTSIDE the 3.5 mm
                            flush window: not capped, score ~0 (the key is real);
  8.  capped-empty        — the cage seated flush on an EMPTY pad reads capped but
                            earns nothing (score ~0), no success;
  9.  wrong object        — the DEAD bulb stood on the pad + the cage seated: capped
                            around the wrong bulb earns nothing, score ~0;
  10. off-pad stand       — the fresh bulb stood upright on the DECK beside the pad:
                            not standing (xy + z windows), score ~0;
  11. lying-on-pad        — the fresh bulb LYING across the pad: not standing (tilt
                            gate), score ~0;
  12. side-swipe          — a wrench-dragged cage carried LATERALLY at rim height
                            30 mm toward the standing bulb knocks it over before
                            ever covering it (the cage moved >= 100 mm, the bulb
                            toppled, cap latch never fired): the cage must descend
                            from above — stand credit only (0.25), no success;
  13. settle gate         — bulb standing + cage written in the flush window but
                            RISING at 0.25 m/s: capped + standing read True at that
                            instant yet success is blocked by the velocity gate;
  14. cap                 — stand + cap latches constructed, then the bulb yanked
                            away: score == 0.45 cap (float32 + eps), NOT success;
  15. latched credit      — 40 further steps: score unchanged, standing stays False;
  16. rejection audit     — success() was never True at ANY judged point;
  17. final no-NaN        — all task-object states finite at the end;
  18. camera              — >= 20 rgb frames captured -> frames.npz.

Run (forge): python -u -m simgen_tasks.light_bulb_in_i381.smoke --headless
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
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.bulb_guard_cap")().build(num_envs=args.num_envs,
                                                    device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    G = 9.81

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.55, -1.15, 0.95)) + o),
                                tuple(np.array((0.46, 0.00, 0.10)) + o),
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

    def report(tag: str) -> None:
        s, ok = judge()
        b = scene.bulb_local()[0]
        k = scene.cage_local()[0]
        print(f"[smoke] {tag:16s} |"
              f" bulb_loc=({float(b[0]):+.3f},{float(b[1]):+.3f},{float(b[2]):+.3f})"
              f" cage_loc=({float(k[0]):+.3f},{float(k[1]):+.3f},{float(k[2]):+.3f})"
              f" standing={bool(scene.bulb_standing()[0])}"
              f" capped={bool(scene.cage_capped()[0])}"
              f" score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def pl(off_xyz) -> tuple[float, float, float]:
        """Env-relative world position of a plinth-frame offset (deck top = z 0)."""
        off = torch.tensor(off_xyz, device=device).view(1, 3)
        p = scene.plinth.data.root_pos_w[:1] + quat_apply(
            scene.plinth.data.root_quat_w[:1], off)
        p = (p - scene.env_origins[:1])[0]
        return (float(p[0]), float(p[1]), float(p[2]))

    def yawq(extra_yaw: float) -> torch.Tensor:
        """Plinth yaw composed with an extra yaw about z (world quat rows)."""
        h = 0.5 * extra_yaw
        qe = torch.tensor([[math.cos(h), 0.0, 0.0, math.sin(h)]], device=device)
        return quat_mul(scene.plinth.data.root_quat_w[:1], qe)

    def lieq(extra_yaw: float) -> torch.Tensor:
        """Lying pose: plinth yaw (+extra) composed with qy(90 deg)."""
        qy = torch.tensor([[math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]],
                          device=device)
        return quat_mul(yawq(extra_yaw), qy)

    def place(body, xyz, quat=None, vel_w=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap). `xyz` is env-relative."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor(xyz, device=device)
        if quat is None:
            st[:, 3] = 1.0
        else:
            st[:, 3:7] = quat
        if vel_w is not None:
            st[:, 7:10] = torch.tensor(vel_w, device=device)
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def park(body) -> None:
        """Transport a body far off the plinth (env-relative floor park)."""
        place(body, (-0.30, -0.45, 0.05), settle_steps=45)

    def stand_bulb_probe() -> None:
        """Construct the standing bulb legitimately: carry to a 3 mm hover over the
        pad, release, gravity-seat (the same released drop the solve uses)."""
        place(scene.bulb, pl((0.0, 0.0, c.pad_h + 0.003)),
              quat=scene.plinth.data.root_quat_w[:1], settle_steps=90)
        assert bool(scene.bulb_standing()[0]), "probe setup: bulb failed to stand"

    def seat_cage_probe() -> None:
        """Construct the seated cage legitimately: key-aligned 2 mm hover, released."""
        place(scene.cage, pl((0.0, 0.0, 0.002)),
              quat=scene.plinth.data.root_quat_w[:1], settle_steps=90)
        assert bool(scene.cage_capped()[0]), "probe setup: cage failed to seat"

    def cage_swipe(x_from: float, z_ref: float, steps: int) -> tuple[float, float]:
        """Wrench-drag the cage laterally (+x, plinth frame) at rim height `z_ref`:
        gravity feed-forward + z/attitude hold + horizontal velocity servo (gains
        from the solve's auditable plant: kd*dt/m = 0.27 < 1). Returns (start x,
        max x reached)."""
        q_des = scene.plinth.data.root_quat_w
        x0, max_x = None, -1e9
        for _ in range(steps):
            kloc = scene.cage_local()[0]
            if x0 is None:
                x0 = float(kloc[0])
            max_x = max(max_x, float(kloc[0]))
            if float(kloc[0]) > -0.02:
                break
            v = scene.cage.data.root_lin_vel_w[0]
            w = scene.cage.data.root_ang_vel_w[0]
            v_des_w = quat_apply(scene.plinth.data.root_quat_w[:1],
                                 torch.tensor([[0.15, 0.0, 0.0]], device=device))[0]
            z_err = pl((0.0, 0.0, z_ref))[2] + float(scene.env_origins[0, 2]) \
                - float(scene.cage.data.root_pos_w[0, 2])
            f = torch.tensor([0.0, 0.0, c.cage_mass * G], device=device)
            f[0:2] += 8.0 * (v_des_w[0:2] - v[0:2])
            f[2] += 60.0 * max(min(z_err, 0.025), -0.025) - 8.0 * v[2]
            fn = f.norm()
            if fn > 8.0:
                f = f * (8.0 / fn)
            q_err = quat_mul(q_des, quat_conjugate(scene.cage.data.root_quat_w))[0]
            sgn = 1.0 if float(q_err[0]) >= 0.0 else -1.0
            t = 0.10 * (2.0 * sgn * q_err[1:4]) - 0.015 * w
            tn = t.norm()
            if tn > 0.05:
                t = t * (0.05 / tn)
            qcur = scene.cage.data.root_link_quat_w
            scene.cage.set_external_force_and_torque(
                quat_apply_inverse(qcur, f.unsqueeze(0).expand(n, 3)).unsqueeze(1),
                quat_apply_inverse(qcur, t.unsqueeze(0).expand(n, 3)).unsqueeze(1),
                env_ids=all_ids)
            step(1)
        scene.cage.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=device), torch.zeros(n, 1, 3, device=device),
            env_ids=all_ids)
        return float(x0), max_x

    def layout_sane(tag: str) -> bool:
        """Reset honesty: both bulbs LYING on the floor in their bands (off the
        deck), cage parked upright on the ground on the opposite side, pad empty."""
        b = scene.bulb_local()[0]
        d = scene.dead_local()[0]
        k = scene.cage_local()[0]
        up_b = float(scene._up_z(scene.bulb)[0])
        up_k = float(scene._up_z(scene.cage)[0])
        ok = (c.bulb_bx[0] - 0.04 <= float(b[0]) <= c.bulb_bx[1] + 0.04
              and abs(float(b[1])) <= c.bulb_by[1] + 0.04
              and float(b[2]) < -0.02  # on the ground, below deck-top level
              and abs(up_b) < 0.35  # lying, not standing
              and c.dead_dx[0] - 0.04 <= float(d[0]) <= c.dead_dx[1] + 0.04
              and c.dead_dy[0] - 0.04 <= abs(float(d[1])) <= c.dead_dy[1] + 0.04
              and float(d[2]) < -0.02
              and c.cage_cx[0] - 0.03 <= float(k[0]) <= c.cage_cx[1] + 0.03
              and c.cage_cy[0] - 0.03 <= abs(float(k[1])) <= c.cage_cy[1] + 0.03
              and abs(float(k[2]) + c.deck_h) < 0.010  # rim on the ground
              and up_k > 0.95
              and float(k[1]) * float(d[1]) < 0.0  # opposite sides
              and not bool(scene.bulb_standing()[0])
              and not bool(scene.cage_capped()[0]))
        if not ok:
            print(f"[smoke] layout sanity VIOLATION at {tag}", flush=True)
        return ok

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    bodies = (scene.plinth, scene.cage, scene.bulb, scene.dead)
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("settle: all states finite; both bulbs lying on the floor in their bands, "
          "cage parked upright on the opposite side, pad empty",
          fin and layout_sane("reset"))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    sane = True
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(30)
        sane = sane and layout_sane(f"seed {sd}")
        pp = (scene.plinth.data.root_pos_w - scene.env_origins)[0]
        pq = scene.plinth.data.root_quat_w[0]
        pyaw = 2.0 * math.atan2(float(pq[3]), float(pq[0]))
        b = scene.bulb_local()[0]
        d = scene.dead_local()[0]
        k = scene.cage_local()[0]
        reads.append((float(pp[0]), float(pp[1]), pyaw, float(b[0]), float(b[1]),
                      float(d[1]), float(k[0])))
    arr = np.array(reads)
    print("[smoke] randomization readback (plinth_x, plinth_y, yaw, bulb_x, bulb_y, "
          f"dead_y, cage_x):\n{np.round(arr, 4)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: plinth pose varies (readback spread "
          f"x={spread[0]:.3f}, y={spread[1]:.3f}, "
          f"yaw={math.degrees(spread[2]):.1f} deg)",
          sane and spread[0] > 0.02 and spread[1] > 0.02
          and spread[2] > math.radians(4.0))
    sides = {1 if v > 0 else -1 for v in arr[:, 5]}
    check("randomization: bulb spawn varies, the dead bulb's side flips, the cage "
          f"park varies (bulb spread ({spread[3]:.3f},{spread[4]:.3f}), "
          f"{len(sides)} dead sides, cage_x spread {spread[6]:.3f})",
          spread[3] > 0.02 and spread[4] > 0.03 and len(sides) == 2
          and spread[6] > 0.02)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 6. lid denial (seed-strategy family / order forcing) =======
    # rlbench/light_bulb_in's verb is deliver-the-bulb-into-the-fixture-from-above.
    # Once the cage is seated that whole family is dead: the lid is closed. Dropped
    # from directly above the capped pad, the bulb bounces off the knob/lid and never
    # stands — so cap-before-stand can never be repaired and stand-then-cap is the
    # only order.
    env.reset(seed=41)
    step(30)
    seat_cage_probe()
    place(scene.bulb, pl((0.0, 0.0, 0.30)),
          quat=scene.plinth.data.root_quat_w[:1], settle_steps=300)
    report("lid-drop")
    b = scene.bulb_local()[0]
    s, ok = judge()
    check("lid denial: the bulb dropped from directly above the CAPPED pad bounces "
          f"off the closed lid (rest local z={float(b[2]) * 1000:+.0f} mm), never "
          "stands on the pad, score ~0 — the capped state is unrepairable, so "
          "stand-then-cap is the only order",
          not bool(scene.bulb_standing()[0]) and s <= 0.02 and not ok)

    # =========================== 7. the yaw key is real =====================================
    env.reset(seed=51)
    step(30)
    place(scene.cage, pl((0.0, 0.0, 0.012)), quat=yawq(math.pi / 4.0),
          settle_steps=120)
    report("key-misalign")
    k = scene.cage_local()[0]
    up_k = float(scene._up_z(scene.cage)[0])
    s, ok = judge()
    check("yaw key: the cage released concentric but rotated 45 deg rests ON the "
          f"key bridges, {float(k[2]) * 1000:.1f} mm proud (window hi "
          f"{c.cap_z_hi * 1000:.1f} mm) — not capped, score ~0",
          float(k[2]) > c.cap_z_hi + 0.001 and float(k[2]) < 0.012 and up_k > 0.95
          and abs(float(k[0])) <= c.cap_xy_tol and abs(float(k[1])) <= c.cap_xy_tol
          and not bool(scene.cage_capped()[0]) and s <= 0.02 and not ok)

    # =========================== 8. capped-empty ============================================
    env.reset(seed=61)
    step(30)
    seat_cage_probe()
    report("capped-empty")
    s, ok = judge()
    check("capped-empty: the cage seated flush on an EMPTY pad reads capped but "
          "earns nothing (score ~0), no success",
          bool(scene.cage_capped()[0]) and not bool(scene.bulb_standing()[0])
          and s <= 0.02 and not ok)

    # =========================== 9. wrong object ============================================
    env.reset(seed=71)
    step(30)
    place(scene.dead, pl((0.0, 0.0, c.pad_h + 0.003)),
          quat=scene.plinth.data.root_quat_w[:1], settle_steps=90)
    seat_cage_probe()
    report("wrong-object")
    d = scene.dead_local()[0]
    s, ok = judge()
    check("wrong object: the DEAD bulb stood on the pad (local z="
          f"{float(d[2]) * 1000:.1f} mm) with the cage seated around it earns "
          "nothing — capping the wrong bulb is worthless, score ~0",
          bool(scene.cage_capped()[0]) and float(d[2]) > c.stand_z_lo
          and not bool(scene.bulb_standing()[0]) and s <= 0.02 and not ok)

    # =========================== 10. off-pad stand ==========================================
    env.reset(seed=81)
    step(30)
    place(scene.bulb, pl((0.035, 0.0, 0.002)),
          quat=scene.plinth.data.root_quat_w[:1], settle_steps=90)
    report("off-pad")
    b = scene.bulb_local()[0]
    up_b = float(scene._up_z(scene.bulb)[0])
    s, ok = judge()
    check("off-pad stand: the fresh bulb standing upright on the DECK beside the "
          f"pad (local x={float(b[0]) * 1000:.0f} mm, z={float(b[2]) * 1000:.1f} mm, "
          f"up_z={up_b:.2f}) is NOT standing-on-pad, score ~0",
          up_b > 0.95 and not bool(scene.bulb_standing()[0]) and s <= 0.02 and not ok)

    # =========================== 11. lying-on-pad ===========================================
    env.reset(seed=91)
    step(30)
    place(scene.bulb, pl((-0.03, 0.0, 0.026)), quat=lieq(0.0), settle_steps=90)
    report("lying-on-pad")
    b = scene.bulb_local()[0]
    up_b = float(scene._up_z(scene.bulb)[0])
    s, ok = judge()
    check("lying-on-pad: the fresh bulb LYING across the pad (up_z="
          f"{up_b:.2f}) is NOT standing (tilt gate), score ~0",
          abs(up_b) < 0.5 and not bool(scene.bulb_standing()[0])
          and s <= 0.02 and not ok)

    # =========================== 12. side-swipe knock =======================================
    # The cage carried LATERALLY at rim height 30 mm cannot deliver: its wall hits
    # the standing bulb's globe and knocks it over before the cage ever covers it.
    env.reset(seed=101)
    step(30)
    stand_bulb_probe()
    place(scene.cage, pl((-0.25, 0.0, 0.030)),
          quat=scene.plinth.data.root_quat_w[:1], settle_steps=2)
    x0, max_x = cage_swipe(-0.25, 0.030, 600)
    step(120)
    report("side-swipe")
    up_b = float(scene._up_z(scene.bulb)[0])
    s, ok = judge()
    check("side-swipe: the cage dragged laterally at rim height moved "
          f"{(max_x - x0) * 1000:.0f} mm (>= 100, non-vacuous) and knocked the "
          f"standing bulb over (up_z={up_b:.2f}) before ever covering it "
          f"(cap latch {float(scene.cap_latch[0]):.0f}) — lateral delivery is "
          "denied, stand credit only",
          max_x - x0 >= 0.100 and up_b < 0.7
          and not bool(scene.bulb_standing()[0])
          and float(scene.cap_latch[0]) == 0.0
          and 0.245 <= s <= 0.25 + 1e-5 and not ok)
    park(scene.cage)

    # =========================== 13. settle gate ============================================
    env.reset(seed=111)
    step(30)
    stand_bulb_probe()
    # cage written key-aligned INSIDE the flush window but RISING at 0.25 m/s: the
    # geometric predicates read True at that instant, the velocity gate blocks
    # success; the cage is then carried away before it can fall back and seat.
    place(scene.cage, pl((0.0, 0.0, 0.001)),
          quat=scene.plinth.data.root_quat_w[:1], vel_w=(0.0, 0.0, 0.25),
          settle_steps=0)
    step(1)
    gate_capped = bool(scene.cage_capped()[0])
    gate_standing = bool(scene.bulb_standing()[0])
    gate_settled = bool(scene.settled()[0])
    _s, gate_ok = judge()
    klin = float(scene.cage.data.root_lin_vel_w[0].norm())
    park(scene.cage)
    report("settle-gate")
    check("settle gate: bulb standing + cage in the flush window but rising at "
          f"{klin:.2f} m/s reads capped={gate_capped} standing={gate_standing} yet "
          "NOT settled and NOT success (velocity gates are real)",
          gate_capped and gate_standing and not gate_settled and not gate_ok
          and klin > c.settle_lin)

    # =========================== 14-15. cap + latched credit ================================
    # check 13 latched stand + cap (the rising cage passed through cage_over while
    # the bulb stood); now yank the bulb away: latches keep the 0.45 cap, live
    # success stays False.
    place(scene.bulb, (-0.35, -0.45, 0.05), quat=lieq(0.0), settle_steps=45)
    report("yanked")
    s_cap, ok = judge()
    check("cap: stand + cap latched, then the bulb yanked away -> score == 0.45 "
          f"cap ({s_cap:.4f}, float32 + eps), NOT success",
          not bool(scene.bulb_standing()[0]) and 0.445 <= s_cap <= 0.45 + 1e-5
          and not ok)
    step(40)
    s_after, ok = judge()
    check("latched credit: 40 further steps leave the latched score unchanged "
          f"({s_cap:.3f} -> {s_after:.3f}) and standing stays False",
          abs(s_after - s_cap) < 1e-3 and not bool(scene.bulb_standing()[0])
          and not ok)

    # =========================== 16-17. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== 18. camera + save + verdict ================================
    check(f"camera: >= 20 rgb frames captured ({len(frames)})", len(frames) >= 20)
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.bulb_guard_cap")
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

"""Smoke / rubric-REJECTION battery for SpindleRollScene (sim_gen task
`put_toilet_roll_on_stand_i188`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — load the roll into the cradle, push the spindle
through its core by regulated force, carry the assembly to the rack, and drop it into
both seats — is the acceptance evidence that the rubric ACCEPTS a correct outcome; it
passes on seeds 0/1/2). Every teleport here is instrumentation that CONSTRUCTS a wrong
(or partial) outcome and asserts the rubric REJECTS it. No probe in this battery ever
reaches success(), and a final audit asserts exactly that.

  1-2.  settle/no-NaN   — reset layout settles finite: roll and spindle lying on the
                          ground on the cradle's side, fixtures near their side-flipped
                          nominals; score ~0, no success;
  3-4.  randomization   — READBACK over 6 seeded resets: cradle and rack positions +
                          yaws, item ground poses, and the SIDE assignment all vary;
  5.    null policy     — 240 idle steps -> score ~0, no success;
  6.    seed-strategy   — the seed family's outcome (the bare ROLL put "on the stand")
                          reconstructed: roll resting on the rack slab between the
                          plates -> no spindle through it, below the hang band, not
                          clear of the base -> rejected, score ~0;
  7.    rod-only mount  — the spindle dropped into both slots WITHOUT the roll nests
                          at SEAT_Z (the rubric anchor for seating) yet scores ~0: a
                          seated empty spindle is not the goal;
  8.    out-of-order    — PHYSICAL probe: with the spindle mounted, the bare roll is
                          pushed axially (regulated force, velocity-capped) to thread
                          it in place — it slides over the exposed ball end and is
                          BLOCKED by the solid plate ~55 mm before the between-plates
                          zone (the probe asserts the roll really moved); threading
                          after mounting is geometrically impossible;
  9.    rod on top      — the spindle laid in the cradle ON TOP of the roll (parallel,
                          looks 'together' from above) reads a rod-line offset of
                          ~39 mm >> thread_radial_tol -> not threaded;
  10.   partial thread  — the spindle nosed only ~20 mm into the core (leading ball
                          inside, tail on the ground) leaves the far-face protrusion
                          NEGATIVE -> not threaded: THROUGH means out the other side;
  11.   assembly on slab— a genuinely threaded assembly RESTING on the rack base slab
                          (between the plates, not in the seats) is rejected by
                          seating (rod ~45 mm below SEAT_Z) and by the hang clauses;
                          only the thread stage latches (score 0.25);
  12.   settle gate +   — the TRUE goal state built with injected swing velocity is
        latched credit    NOT success while moving (sustained stillness required);
                          teleporting the assembly away keeps the latched 0.25 while
                          seated()/hanging() drop to False — credit never evaporates,
                          and an abandoned state never becomes success;
  13.   rejection audit — success() was never True at ANY judged point;
  14.   final no-NaN    — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.put_toilet_roll_on_stand_i188.smoke --headless
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
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod

CAP_Z = scene_mod.CAP_Z
CAP_R = scene_mod.CAP_R
ROD_R = scene_mod.ROD_R
ROD_TIP = scene_mod.ROD_TIP
OUT_FLAT = scene_mod.OUT_FLAT
ROLL_HALF_W = scene_mod.ROLL_HALF_W
CRADLE_REST_Z = scene_mod.CRADLE_REST_Z
SLAB_TOP = scene_mod.SLAB_TOP
SEAT_Z = scene_mod.SEAT_Z
PLATE_X_OUT = scene_mod.PLATE_X_OUT

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.spindle_roll")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.55, -1.25, 1.05)) + o),
                                tuple(np.array((0.38, 0.00, 0.10)) + o),
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

    def report(tag: str) -> None:
        s, ok = judge()
        r, sp = rel(scene.roll), rel(scene.spindle)
        radial, _al, plo, phi = scene.thread_metrics()
        print(f"[smoke] {tag:16s} | roll=({float(r[0]):+.3f},{float(r[1]):+.3f},"
              f"{float(r[2]):.3f}) rod=({float(sp[0]):+.3f},{float(sp[1]):+.3f},"
              f"{float(sp[2]):.3f}) rad={float(radial[0]):.4f}"
              f" prot=({float(plo[0]):+.3f},{float(phi[0]):+.3f})"
              f" thr={bool(scene.threaded()[0])} seat={bool(scene.seated()[0])}"
              f" hang={bool(scene.hanging()[0])} set={bool(scene.settled()[0])}"
              f" score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, xyz, quat=None, vel=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = (xyz if torch.is_tensor(xyz)
                      else torch.tensor([float(v) for v in xyz], device=device))
        if quat is None:
            st[:, 3] = 1.0
        elif torch.is_tensor(quat):
            st[:, 3:7] = quat
        else:
            st[:, 3:7] = torch.tensor([float(v) for v in quat], device=device)
        if vel is not None:
            st[:, 7:10] = (vel if torch.is_tensor(vel)
                           else torch.tensor([float(v) for v in vel], device=device))
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def wrench(body, f_w: torch.Tensor) -> None:
        """WORLD force at the CoM, expressed in the body's link frame (the house
        convention — is_global drops torques on this stack)."""
        q = body.data.root_link_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f_w).unsqueeze(1),
            torch.zeros(n, 1, 3, device=device), env_ids=all_ids)

    QY90 = torch.tensor([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0],
                        device=device)  # local +z -> local +x

    def frame(body) -> tuple[torch.Tensor, torch.Tensor, float]:
        """(env-rel pos (3,), quat (4,), yaw) of a fixture."""
        p = rel(body)
        q = body.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return p, q, yaw

    def local(body, loc) -> torch.Tensor:
        """Env-rel world position of a fixture-local point."""
        p, q, _ = frame(body)
        return p + quat_apply(q.view(1, 4), torch.tensor([loc], device=device))[0]

    def axis_quat(body) -> torch.Tensor:
        """Quat aligning a tube/rod local +z with the fixture's local +x."""
        _p, q, _ = frame(body)
        return quat_mul(q.view(1, 4), QY90.view(1, 4))[0]

    def layout_sane(tag: str) -> bool:
        """Reset honesty: items lying low on the cradle's side, fixtures on opposite
        sides near their side-flipped nominals."""
        r, sp = rel(scene.roll), rel(scene.spindle)
        cr, _qc, _cy = frame(scene.cradle)
        rk, _qr, _ry = frame(scene.rack)
        jf, ji = c.fixture_jitter + 0.006, c.item_jitter + 0.006
        ok = (float(r[2]) < 0.06 and float(sp[2]) < 0.06
              and abs(float(cr[0]) - c.cradle_pos[0]) < jf
              and abs(abs(float(cr[1])) - abs(c.cradle_pos[1])) < jf
              and abs(float(rk[0]) - c.rack_pos[0]) < jf
              and abs(abs(float(rk[1])) - abs(c.rack_pos[1])) < jf
              and float(cr[1]) * float(rk[1]) < 0
              and abs(float(r[0]) - c.roll_spawn[0]) < ji
              and abs(float(sp[0]) - c.rod_spawn[0]) < ji)
        if not ok:
            print(f"[smoke] layout sanity VIOLATION at {tag}", flush=True)
        return ok

    bodies = (scene.cradle, scene.rack, scene.roll, scene.spindle)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("settle: all states finite; roll and spindle lying on the ground, cradle and "
          "rack on opposite sides near nominal", fin and layout_sane("reset"))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    sane = True
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(30)
        sane = sane and layout_sane(f"seed {sd}")
        cr, _qc, cyaw = frame(scene.cradle)
        rk, _qr, ryaw = frame(scene.rack)
        r, sp = rel(scene.roll), rel(scene.spindle)
        reads.append((float(cr[0]), float(cr[1]), cyaw, float(rk[0]), float(rk[1]),
                      ryaw, float(r[0]), float(r[1]), float(sp[0]), float(sp[1]),
                      1.0 if float(cr[1]) < 0 else -1.0))
    arr = np.array(reads)
    print("[smoke] randomization readback (cradle x,y,yaw | rack x,y,yaw | roll x,y | "
          f"rod x,y | side):\n{np.round(arr, 4)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: cradle and rack poses vary (spreads "
          f"cradle=({spread[0]:.3f},{spread[1]:.3f},{spread[2]:.2f} rad) "
          f"rack=({spread[3]:.3f},{spread[4]:.3f},{spread[5]:.2f} rad))",
          spread[0] > 0.02 and spread[3] > 0.02 and spread[2] > 0.15
          and spread[5] > 0.08)
    sides = {float(v) for v in arr[:, 10]}
    check("randomization: item ground poses vary and BOTH side assignments occur "
          f"(roll spread=({spread[6]:.3f},{spread[7]:.3f}), {len(sides)} sides), "
          "every reset lies flat and sane",
          spread[6] > 0.02 and spread[8] > 0.02 and len(sides) == 2 and sane)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed-strategy: bare roll "on the stand" =================
    # rlbench/put_toilet_roll_on_stand's OUTCOME is the bare roll put on the stand.
    # Reconstruct its analog: the roll placed on the rack base slab between the
    # plates (as close to "on the stand" as the bare roll can physically get — the
    # slots are 22 mm, the roll 66 mm). No spindle through it, center 20 mm below
    # the hang band, resting on the slab -> rejected on every load-bearing clause.
    env.reset(seed=41)
    step(30)
    place(scene.roll, local(scene.rack, (0.0, 0.0, SLAB_TOP + OUT_FLAT + 0.0005)),
          quat=axis_quat(scene.rack), settle_steps=60)
    report("roll-on-slab")
    s, ok = judge()
    p_rr = quat_apply_inverse(scene.rack.data.root_quat_w,
                              scene.roll.data.root_pos_w
                              - scene.rack.data.root_pos_w)[0]
    check("seed-strategy analog: the bare roll resting on the bracket base between "
          f"the plates (z={float(p_rr[2]) * 1000:.0f} mm, hang band "
          f"{c.hang_z * 1000:.0f}±{c.hang_band * 1000:.0f} mm) is NOT hanging — "
          "not threaded, no success, score ~0",
          not bool(scene.threaded()[0]) and not bool(scene.hanging()[0])
          and s <= 0.02 and not ok)

    # =========================== 7. rod-only mount (anchors SEAT_Z) =========================
    env.reset(seed=51)
    step(30)
    place(scene.spindle, local(scene.rack, (0.0, 0.0, SEAT_Z + 0.013)),
          quat=axis_quat(scene.rack), settle_steps=60)
    report("rod-only-mount")
    p_sr = quat_apply_inverse(scene.rack.data.root_quat_w,
                              scene.spindle.data.root_pos_w
                              - scene.rack.data.root_pos_w)[0]
    s, ok = judge()
    check("rod-only mount: the empty spindle dropped into the slots nests at "
          f"z={float(p_sr[2]) * 1000:.1f} mm (SEAT_Z={SEAT_Z * 1000:.0f} mm — the "
          "seating anchor) and reads seated(), yet scores ~0: no roll, no goal",
          abs(float(p_sr[2]) - SEAT_Z) < 0.004 and bool(scene.seated()[0])
          and s <= 0.02 and not ok)

    # =========================== 8. out-of-order: thread AFTER mounting =====================
    # PHYSICAL probe (quasi-static, velocity-capped, support-regulated): the bare
    # roll approaches the mounted spindle coaxially from outside the +x plate and is
    # pushed inward. It can slide over the exposed ball end, but the SOLID PLATE
    # blocks its face ~55 mm before the between-plates zone: threading cannot happen
    # after mounting — the assembly order is geometrically forced.
    p_r0, q_r0, _ = frame(scene.rack)
    q_rack_t = scene.rack.data.root_quat_w
    start_loc = (PLATE_X_OUT + ROLL_HALF_W + 0.055, 0.0, SEAT_Z)
    place(scene.roll, local(scene.rack, start_loc), quat=axis_quat(scene.rack),
          settle_steps=2)
    x0 = float(quat_apply_inverse(q_rack_t, scene.roll.data.root_pos_w
                                  - scene.rack.data.root_pos_w)[0, 0])
    for _i in range(300):
        p_c = quat_apply_inverse(q_rack_t, scene.roll.data.root_pos_w
                                 - scene.rack.data.root_pos_w)
        v_c = quat_apply_inverse(q_rack_t, scene.roll.data.root_lin_vel_w)
        f_c = torch.zeros(n, 3, device=device)
        f_c[:, 0] = (4.0 * (-0.08 - v_c[:, 0])).clamp(-0.8, 0.8)
        f_c[:, 1] = (5.0 * (0.0 - p_c[:, 1]) - 1.2 * v_c[:, 1]).clamp(-0.8, 0.8)
        f_c[:, 2] = (c.roll_mass * 9.81 + 5.0 * (SEAT_Z - p_c[:, 2])
                     - 1.2 * v_c[:, 2]).clamp(0.0, 2 * c.roll_mass * 9.81)
        wrench(scene.roll, quat_apply(q_rack_t, f_c))
        env.step(no_action)
    wrench(scene.roll, torch.zeros(n, 3, device=device))
    report("thread-after")
    x1 = float(quat_apply_inverse(q_rack_t, scene.roll.data.root_pos_w
                                  - scene.rack.data.root_pos_w)[0, 0])
    s, ok = judge()
    check("out-of-order: pushing the bare roll onto the MOUNTED spindle moved it "
          f"{(x0 - x1) * 1000:.0f} mm (the probe is not vacuous) yet the plate stops "
          f"it at x={x1 * 1000:.0f} mm — it never reaches the between-plates zone "
          "and never reads threaded: threading after mounting is impossible",
          (x0 - x1) > 0.030 and x1 > 0.100 and not bool(scene.threaded()[0]) and not ok)
    place(scene.roll, (0.75, 0.55, OUT_FLAT + 0.002), quat=QY90, settle_steps=20)

    # =========================== 9. rod ON TOP of the cradled roll ==========================
    env.reset(seed=61)
    step(30)
    place(scene.roll, local(scene.cradle, (-0.002, 0.0, CRADLE_REST_Z + 0.020)),
          quat=axis_quat(scene.cradle), settle_steps=60)
    place(scene.spindle,
          local(scene.cradle, (-0.002, 0.0, CRADLE_REST_Z + OUT_FLAT + ROD_R + 0.001)),
          quat=axis_quat(scene.cradle), settle_steps=50)
    report("rod-on-top")
    radial = float(scene.thread_metrics()[0][0])
    s, ok = judge()
    check("rod on top: the spindle laid on/beside the cradled roll (parallel, "
          f"'together') reads rod-line offset {radial * 1000:.0f} mm >> "
          f"thread_radial_tol {c.thread_radial_tol * 1000:.0f} mm -> not threaded, "
          "only the cradle stage latches",
          radial > c.thread_radial_tol + 0.005 and not bool(scene.threaded()[0])
          and s <= 0.21 and not ok)

    # =========================== 10. partial threading ======================================
    env.reset(seed=71)
    step(30)
    place(scene.roll, local(scene.cradle, (-0.002, 0.0, CRADLE_REST_Z + 0.020)),
          quat=axis_quat(scene.cradle), settle_steps=60)
    p_roll_c = quat_apply_inverse(scene.cradle.data.root_quat_w,
                                  scene.roll.data.root_pos_w
                                  - scene.cradle.data.root_pos_w)[0]
    nose_x = float(p_roll_c[0]) - ROLL_HALF_W + 0.020 - CAP_Z
    place(scene.spindle,
          local(scene.cradle, (nose_x, float(p_roll_c[1]), float(p_roll_c[2]))),
          quat=axis_quat(scene.cradle), settle_steps=60)
    report("partial-thread")
    _rad2, _al2, _plo2, phi2 = scene.thread_metrics()
    s, ok = judge()
    check("partial threading: the spindle nosed ~20 mm into the core (tail dropped "
          f"to the ground) reads far-face protrusion {float(phi2[0]) * 1000:.0f} mm "
          f"< 0 < protrude_min — not THROUGH, not threaded",
          float(phi2[0]) < 0.0 and not bool(scene.threaded()[0])
          and s <= 0.21 and not ok)

    # =========================== 11. threaded assembly resting on the slab ==================
    env.reset(seed=81)
    step(30)
    q_ax = axis_quat(scene.rack)
    place(scene.roll, local(scene.rack, (0.0, 0.0, SLAB_TOP + OUT_FLAT + 0.0005)),
          quat=q_ax, settle_steps=0)
    p_roll_now = quat_apply_inverse(scene.rack.data.root_quat_w,
                                    scene.roll.data.root_pos_w
                                    - scene.rack.data.root_pos_w)[0]
    place(scene.spindle,
          local(scene.rack, (0.0, float(p_roll_now[1]),
                             float(p_roll_now[2]) - 0.0125)),
          quat=q_ax, settle_steps=60)
    report("assembly-on-slab")
    p_sr2 = quat_apply_inverse(scene.rack.data.root_quat_w,
                               scene.spindle.data.root_pos_w
                               - scene.rack.data.root_pos_w)[0]
    s, ok = judge()
    check("assembly on slab: a threaded assembly DUMPED on the bracket base reads "
          f"rod z={float(p_sr2[2]) * 1000:.0f} mm (seat needs {SEAT_Z * 1000:.0f}"
          f"±{c.seat_yz_tol * 1000:.0f} mm) -> not seated, roll not hanging: only "
          "the thread stage latches (score 0.25)",
          bool(scene.threaded()[0]) and not bool(scene.seated()[0])
          and not bool(scene.hanging()[0]) and abs(s - 0.25) < 0.002 and not ok)

    # =========================== 12. settle gate + latched credit ===========================
    # The TRUE goal state, built with an injected lateral swing: not success while
    # moving. Then the whole assembly is removed BEFORE it can settle (this battery
    # must never reach success) — the latched credit survives, the state checks drop.
    env.reset(seed=91)
    step(30)
    q_ax = axis_quat(scene.rack)
    _p, q_rk, _y = frame(scene.rack)
    place(scene.spindle, local(scene.rack, (0.0, 0.0, SEAT_Z + 0.003)), quat=q_ax,
          settle_steps=0)
    swing = quat_apply(q_rk.view(1, 4),
                       torch.tensor([[0.0, 0.28, 0.0]], device=device))[0]
    place(scene.roll, local(scene.rack, (0.0, 0.0, SEAT_Z + 0.003 - 0.0125)),
          quat=q_ax, vel=tuple(float(v) for v in swing), settle_steps=0)
    # The roll rams the rod through the 13 mm radial gap within a couple of
    # substeps, so a single delayed velocity sample under-reads the swing.
    # Sample the PEAK across the first substeps (measure at the moment of
    # motion, not after decay) and judge at every sample.
    v_peak, gate_ok = 0.0, False
    for _ in range(8):
        step(1)
        v_roll = float(scene.roll.data.root_lin_vel_w[0].norm())
        v_peak = max(v_peak, v_roll)
        s_before, ok = judge()
        if (bool(scene.threaded()[0]) and v_roll > c.settle_lin
                and not bool(scene.settled()[0]) and not ok):
            gate_ok = True
    report("goal-swinging")
    s_before, ok = judge()
    gate_ok = gate_ok and not ok and not bool(scene.settled()[0])
    check("settle gate: the exact goal state still SWINGING (roll peak "
          f"{v_peak:.2f} m/s > settle_lin) is NOT success — sustained stillness is "
          "load-bearing", gate_ok)
    place(scene.spindle, (0.75, -0.55, CAP_R + 0.002), quat=QY90, settle_steps=0)
    place(scene.roll, (0.75, 0.55, OUT_FLAT + 0.002), quat=QY90, settle_steps=40)
    report("moved-away")
    s_after, ok = judge()
    check("latched credit: teleporting the assembly away keeps the latched score "
          f"({s_before:.2f} -> {s_after:.2f}, thread = 0.25) while seated()/"
          "hanging() drop to False — and an abandoned goal state never succeeds",
          abs(s_after - 0.25) < 0.002 and s_after >= s_before - 1e-4
          and not bool(scene.seated()[0]) and not bool(scene.hanging()[0]) and not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.spindle_roll")
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

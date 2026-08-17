"""Smoke / rubric-REJECTION battery for RubbishChuteScene (sim_gen task
`put_rubbish_in_bin_i147`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — gravity load, force-held gate, gravity-return
close — is the acceptance evidence that the rubric ACCEPTS a correct outcome). Every
teleport here is instrumentation that CONSTRUCTS a wrong (or partial) outcome as a
settled state and asserts the rubric REJECTS it; no probe in this battery ever
reaches success(), and a final audit check asserts exactly that.

  1-2.  settle/no-NaN     — reset layout settles finite: gate seated shut in its
                            slots, both balls on the ground, score ~0 at rest;
  3-4.  randomization     — READBACK over 8 seeded resets: structure yaw + xy vary
                            and the gate follows it seated; both ball spawns vary,
                            the side bands really SWAP, balls never within 9 cm;
  5.   null policy        — 240 idle steps -> score ~0, no success;
  6.   seed strategy dead — the seed's whole plan (carry the rubbish over the bin,
                            release): dropped above the SEALED bin it settles on the
                            roof, never inside — the chute is the only way in;
  7.   wrong object load  — the tomato dropped into the trough earns NO load credit
                            (the load latch watches the paper ball only);
  8.   wrong object bin   — tomato constructed inside the bin, paper ball outside ->
                            no success, score ~0;
  9.   contamination      — BOTH balls constructed inside the bin -> NOT success
                            (the tomato poisons it), score <= 0.80;
  10.  shut gate blocks   — the paper ball dropped into the trough rolls down and
                            rests AGAINST the closed gate, never past it (the
                            load-then-lift order is forced by physics), score 0.25;
  11.  latched credit     — removing the loaded ball to the ground keeps the load
                            credit (latch regression probe), still no success;
  12.  gate captive       — a constant 1.3x-weight overpull presses the gate to its
                            retaining caps: it MOVED (non-vacuous probe) and never
                            escaped past the caps;
  13.  gravity return     — force cleared -> the gate falls back and reseats within
                            shut_tol ON ITS OWN; gate credit alone caps at 0.25;
  14.  incomplete         — paper ball constructed in the bin, tomato out, but the
                            gate HELD open by the probe force -> NOT success (the
                            reseated-gate condition is load-bearing), score <= 0.80;
  15.  monotonicity       — a partial lift latches strictly less gate credit than a
                            full lift;
  16.  rejection audit    — success() was never True at ANY judged point;
  17.  final no-NaN       — all task-object states finite at the end;
  18.  camera             — >= 20 rgb frames recorded -> frames.npz.

Run (forge): python -u -m simgen_tasks.put_rubbish_in_bin_i147.smoke --headless
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
# RTX recipe: kit mis-decodes some driver versions and silently rejects RTX -> the
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
    env = ENVS.get("simgen.rubbish_chute")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.70, -1.40, 1.00)) + o),
                                tuple(np.array((0.55, 0.00, 0.15)) + o),
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

    def struct_pose() -> tuple[torch.Tensor, float]:
        sp = (scene.structure.data.root_pos_w - scene.env_origins)[0]
        q = scene.structure.data.root_quat_w[0]
        return sp, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def report(tag: str) -> None:
        r = scene._local(scene.rubbish)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | rubbish_local=({float(r[0]):+.3f},{float(r[1]):+.3f},"
              f"{float(r[2]):.3f}) gate_lift={float(scene.gate_lift()[0]):+.4f} "
              f"loaded={bool(scene._loaded[0])} gate_max={float(scene._gate_max[0]):.3f} "
              f"passed={bool(scene._passed[0])} shut={bool(scene.gate_shut()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_obj(obj, x_l: float, y_l: float, z: float, settle_steps: int = 60) -> None:
        """Kinematic probe placement in STRUCTURE-LOCAL xy (the structure origin sits
        on the ground, so local z == world z) + REAL physics steps before judging
        (the zero-step trap). Instrumentation, not a solution."""
        sp, syaw = struct_pose()
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = float(sp[0]) + math.cos(syaw) * x_l - math.sin(syaw) * y_l
        st[:, 1] = float(sp[1]) + math.sin(syaw) * x_l + math.cos(syaw) * y_l
        st[:, 2] = z
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        obj.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def gate_force(f: float, steps: int) -> float:
        """Apply a constant vertical force at the gate CoM for `steps`; returns the
        max lift observed (the probe must PROVE the actuator moved — vacuity guard)."""
        wrench = torch.zeros(n, 1, 3, device=device)
        wrench[:, 0, 2] = f
        peak = -1.0
        for _ in range(steps):
            scene.gate.set_external_force_and_torque(wrench, zero_w, env_ids=all_ids,
                                                     is_global=True)
            step(1)
            peak = max(peak, float(scene.gate_lift()[0]))
        return peak

    def hold_gate(target: float, steps: int) -> float:
        """PD-hold the gate at `target` lift (gravity feedforward from the mass
        readback; kp*dt/m = 0.22 << 1). Returns the final lift."""
        m_gate = float(scene.gate.root_physx_view.get_masses().sum())
        ff = m_gate * 9.81
        wrench = torch.zeros(n, 1, 3, device=device)
        for _ in range(steps):
            lift = float(scene.gate_lift()[0])
            v = float(scene.gate.data.root_lin_vel_w[0, 2])
            wrench[:, 0, 2] = max(0.0, min(ff + 8.0 * (target - lift) - 2.5 * v, 10.0))
            scene.gate.set_external_force_and_torque(wrench, zero_w, env_ids=all_ids,
                                                     is_global=True)
            step(1)
        return float(scene.gate_lift()[0])

    def release_gate() -> None:
        scene.gate.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)

    def zf(x_l: float) -> float:
        return c.chute_h0 - x_l * math.tan(math.radians(c.theta_deg))

    max_lift = c.cap_h0 - c.plate_h  # 74.5 mm cap travel

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    fin0 = (torch.isfinite(scene.gate.data.root_state_w).all()
            and torch.isfinite(scene.rubbish.data.root_state_w).all()
            and torch.isfinite(scene.tomato.data.root_state_w).all())
    lift0 = float(scene.gate_lift()[0])
    rz = float((scene.rubbish.data.root_pos_w - scene.env_origins)[0, 2])
    tz = float((scene.tomato.data.root_pos_w - scene.env_origins)[0, 2])
    still = (float(scene.gate.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.rubbish.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    check("settle: states finite, gate seated shut in its slots, both balls resting on "
          "the ground, everything at rest",
          bool(fin0) and abs(lift0) < 0.006 and bool(scene.gate_shut()[0])
          and abs(rz - c.ball_r) < 0.01 and abs(tz - c.ball_r) < 0.01 and still)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        sp, syaw = struct_pose()
        rp = (scene.rubbish.data.root_pos_w - scene.env_origins)[0]
        tp = (scene.tomato.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(sp[0]), float(sp[1]), syaw, float(rp[0]), float(rp[1]),
                      float(tp[0]), float(tp[1]), float((rp[:2] - tp[:2]).norm()),
                      float(scene.gate_lift()[0])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (sx, sy, syaw, rub_x, rub_y, tom_x, tom_y, "
          f"sep, gate_lift):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: structure yaw + xy vary across seeded resets and the gate "
          "follows it seated shut (readback)",
          spread[2] > 0.05 and spread[0] > 0.008 and spread[1] > 0.008
          and float(np.abs(arr[:, 8]).max()) < 0.01)
    check("randomization: both ball spawns vary, the side bands really SWAP (paper ball "
          "seen on both sides), balls never within 9 cm of each other (readback)",
          spread[3] > 0.015 and (arr[:, 4] > 0).any() and (arr[:, 4] < 0).any()
          and float(arr[:, 7].min()) >= 0.09)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy is physically dead ========================
    # The seed's whole plan is "carry the rubbish above the bin and release". Executed
    # here, the drop lands ON the sealed bin's roof — the interior is unreachable from
    # above, so the chute (and its gate) is the only way in, by geometry.
    torch.manual_seed(41)
    env.reset()
    step(10)
    place_obj(scene.rubbish, c.bin_cx, 0.0, c.bin_wall_top + 0.012 + c.ball_r + 0.05,
              settle_steps=180)
    report("seed-strategy")
    s, ok = judge()
    r_loc = scene._local(scene.rubbish)[0]
    check("seed strategy: rubbish released above the SEALED bin settles on/off the roof, "
          "never inside — no success, score <= 0.02",
          not bool(scene._inside_bin(scene._local(scene.rubbish))[0])
          and float(r_loc[2]) > c.inside_z[1] and not ok and s <= 0.02)

    # =========================== 7. wrong object: tomato in the trough ======================
    torch.manual_seed(51)
    env.reset()
    step(10)
    place_obj(scene.tomato, 0.10, 0.0, zf(0.10) + 0.118, settle_steps=180)
    report("tomato-trough")
    s, ok = judge()
    t_loc = scene._local(scene.tomato)
    check("wrong object (load): the TOMATO dropped into the trough really rests there "
          "(readback) yet earns NO load credit — the latch watches the paper ball only",
          bool(scene._in_trough(t_loc)[0]) and not bool(scene._loaded[0])
          and s <= 0.02 and not ok)

    # =========================== 8. wrong object: tomato inside the bin =====================
    torch.manual_seed(61)
    env.reset()
    step(10)
    place_obj(scene.tomato, c.bin_cx + 0.02, 0.03, 0.055, settle_steps=90)
    report("tomato-bin")
    s, ok = judge()
    check("wrong object (bin): tomato constructed inside the bin, paper ball outside — "
          "no success, score <= 0.02",
          bool(scene._inside_bin(scene._local(scene.tomato))[0]) and not ok and s <= 0.02)

    # =========================== 9. contamination: BOTH balls inside ========================
    torch.manual_seed(71)
    env.reset()
    step(10)
    place_obj(scene.tomato, c.bin_cx - 0.04, -0.04, 0.055, settle_steps=30)
    place_obj(scene.rubbish, c.bin_cx + 0.04, 0.04, 0.055, settle_steps=90)
    report("both-in")
    s, ok = judge()
    check("contamination: BOTH balls settled inside the bin — the tomato poisons it: "
          "NOT success, score <= 0.80 (pass credit may latch, the goal does not)",
          bool(scene._inside_bin(scene._local(scene.rubbish))[0])
          and bool(scene._inside_bin(scene._local(scene.tomato))[0])
          and not ok and s <= 0.80 + 1e-6)

    # =========================== 10. the shut gate physically blocks the ball ===============
    torch.manual_seed(81)
    env.reset()
    step(10)
    place_obj(scene.rubbish, 0.10, 0.0, zf(0.10) + 0.118, settle_steps=240)
    report("blocked")
    s_blk, ok = judge()
    r_loc = scene._local(scene.rubbish)[0]
    check("order forced: the ball dropped into the trough rolls down and rests AGAINST "
          "the closed gate — loaded, never past it, score == load credit only (0.25)",
          bool(scene._loaded[0]) and not bool(scene._passed[0])
          and 0.15 < float(r_loc[0]) < c.gate_x
          and float(scene.rubbish.data.root_lin_vel_w[0].norm()) < 0.10
          and abs(s_blk - c.w_load) < 1e-3 and not ok)

    # =========================== 11. latched credit survives regression =====================
    place_obj(scene.rubbish, 0.05, 0.30, c.ball_r + 0.002, settle_steps=40)
    report("regressed")
    s_reg, ok = judge()
    check("latched credit: removing the loaded ball back to the ground keeps the load "
          "credit (running latch), still no success",
          not bool(scene._in_trough(scene._local(scene.rubbish))[0])
          and abs(s_reg - c.w_load) < 1e-3 and not ok)

    # =========================== 12-13. gate captive + gravity return =======================
    torch.manual_seed(91)
    env.reset()
    step(10)
    m_gate = float(scene.gate.root_physx_view.get_masses().sum())
    peak = gate_force(1.3 * m_gate * 9.81, 150)  # constant overpull presses it to the caps
    report("overpull")
    check("gate captive (non-vacuous probe): a constant 1.3x-weight overpull really "
          f"lifted the gate (peak {peak:.3f} m > 0.055) and the retaining caps held it "
          f"(never past {max_lift + 0.006:.3f} m)",
          0.055 < peak <= max_lift + 0.006)
    release_gate()
    step(120)
    report("released")
    s_ret, ok = judge()
    check("gravity return: with the force cleared the gate falls back and reseats "
          "within shut_tol ON ITS OWN; latched gate credit alone caps at 0.25, no "
          "success", bool(scene.gate_shut()[0]) and abs(s_ret - c.w_gate) < 1e-3 and not ok)

    # =========================== 14. incomplete: gate held open =============================
    torch.manual_seed(101)
    env.reset()
    step(10)
    place_obj(scene.rubbish, c.bin_cx, 0.02, 0.055, settle_steps=60)
    hold_gate(c.lift_ref + 0.006, 120)
    report("held-open")
    s_open, ok = judge()
    lift_now = float(scene.gate_lift()[0])
    check("incomplete: paper ball settled in the bin, tomato out, but the gate HELD "
          "open by the probe force — the reseated-gate condition is load-bearing: NOT "
          "success, score <= 0.80",
          bool(scene._inside_bin(scene._local(scene.rubbish))[0])
          and lift_now > c.shut_tol + 0.01 and not ok and s_open <= 0.80 + 1e-6)
    release_gate()  # NOTE: no judging between here and the next reset — releasing over
    env.reset()     # an already-delivered ball would complete the task for real.

    # =========================== 15. monotonicity of gate credit ============================
    torch.manual_seed(111)
    env.reset()
    step(10)
    hold_gate(0.024, 90)
    report("half-lift")
    s_half, _ok = judge()
    gm_half = float(scene._gate_max[0])
    hold_gate(c.lift_ref + 0.006, 90)
    report("full-lift")
    s_full, _ok = judge()
    gm_full = float(scene._gate_max[0])
    release_gate()
    step(60)
    check("monotonicity: a partial lift latches strictly less gate credit than a full "
          f"lift ({gm_half:.3f} < {gm_full:.3f}) and a lower score",
          0.15 < gm_half < 0.85 and gm_full > 0.95 and s_half < s_full)

    # =========================== 16-17. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.structure.data.root_state_w).all()
           and torch.isfinite(scene.gate.data.root_state_w).all()
           and torch.isfinite(scene.rubbish.data.root_state_w).all()
           and torch.isfinite(scene.tomato.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== 18. camera + save ==========================================
    check(f"camera: >= 20 rgb frames recorded ({len(frames)})", len(frames) >= 20)
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.rubbish_chute")
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
    except Exception as e:  # noqa: BLE001 — die fast and loudly, never hang in teardown
        print(f"[smoke] EXCEPTION: {e!r}", flush=True)
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)

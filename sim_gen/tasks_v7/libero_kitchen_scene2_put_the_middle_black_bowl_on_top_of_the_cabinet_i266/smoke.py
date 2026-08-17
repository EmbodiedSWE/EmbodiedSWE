"""Smoke / rubric-rejection battery for ScoopLiftCourtScene (sim_gen task
`libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet_i266`) —
NullRobot, teleported/forced probe states (instrumentation, NOT a solution), RECORDED.

Battery:
  1-2. settle/no-NaN    — reset settles finite: arm on the load stop, three bowls in a
                          row on the staging pad with the TARGET strictly in the middle,
                          court empty, no latches, score ~0;
  3. randomization      — READBACK across seeds: fixture root xy+yaw move (all
                          predicates fixture-frame), the TARGET BODY INDEX varies
                          (bowl->slot permutation is real), lift-base xy jitter, row
                          spacing and bowl yaw move;
  4. null-policy-fails  — 240 idle steps -> score ~0, no success;
  5-7. oracle x3 seeds  — the honest mechanism on torque drive: hold the arm level,
                          free-fall the middle bowl into the scoop (0.15), crank past
                          60 deg (0.30) and 95 deg (0.50), fast final sweep to the dump
                          stop -> pour through the window (0.95) -> release, settle:
                          success 1.0; persists 240 steps;
  8-9. monotonicity     — ladder from oracle 0: idle < loaded < lifted < delivered <
                          poured <= 1.0; all pre-pour partials < 1;
  10. occupied forfeit  — decoy bowl placed at rest in the court flips success OFF
                          (score falls back to the latched 0.95);
  11. latch permanence + plate forfeit — decoy removed: success RECOVERS (latches
                          persist); the white plate placed in the court kills it again;
  12. negative (seed strategy) — the seed's plan "release the bowl above the cabinet"
                          parks it ON THE ROOF (rest z readback above the roof top):
                          no latch, score 0, no success;
  13. negative (teleport bypass) — target placed at rest directly on the court seat:
                          live seated predicate TRUE (probe not vacuous) yet every
                          mechanism latch refuses -> score 0, no success;
  14. negative (window band gating) — target dropped onto the window SILL: the live
                          window-band predicate fires during the fall (not vacuous) but
                          the windowed latch refuses without the delivered history;
  15. negative (window walk-in) — the sill bowl force-PUSHED through the window into
                          the court: live courted + seated TRUE, all latches still
                          refuse -> score 0, no success;
  16. negative (out of order) — the EMPTY arm cranked to the dump stop earns nothing;
                          the bowl then fed into the pocket at the top slides out
                          through the window but loaded/lifted/delivered never fired,
                          so nothing latches -> score 0, no success.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet_i266.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--out", type=str, default="frames.npz")
parser.add_argument("--max_sec", type=float, default=1350.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit may mis-decode the driver version and silently reject RTX -> the
# annotator returns EMPTY frames. Disable the driver check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet_i266 import (  # noqa: F401,E501
        scene as scene_mod,
    )
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
_wd = threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                             os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.scoop_lift_court")().build(num_envs=args.num_envs,
                                                      device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(1, 1, 3, device=device)
    dt = env.dt

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.30, -1.85, 1.15)) + o),
                                tuple(np.array((0.05, 0.0, 0.32)) + o),
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

    def settle_until(pred, max_steps: int = 600, poll: int = 10,
                     min_steps: int = 30) -> bool:
        """Poll `pred` while stepping. ALWAYS steps at least `min_steps` first: right
        after a teleport all velocities are zero, so settled()-style predicates are
        vacuously true before physics has run (and post_step latches never fire)."""
        step(min_steps)
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def sc() -> float:
        return float(scene.score()[0])

    def ti() -> int:
        return int(scene.target_idx[0])

    def tgt():
        return scene.bowls[ti()]

    def tgt_fix() -> torch.Tensor:
        return scene.target_fix()[0]

    def th_now() -> float:
        return float(scene.arm_deg()[0])

    def latches() -> tuple:
        return (bool(scene._loaded_ever[0].any()), bool(scene._lifted_ever[0].any()),
                bool(scene._delivered_ever[0].any()),
                bool(scene._windowed_ever[0].any()), bool(scene._courted_ever[0].any()))

    def no_latch() -> bool:
        return not any(latches())

    def report(tag: str) -> None:
        tf = tgt_fix()
        print(f"[smoke] {tag:16s} th={th_now():+7.1f} tgt={ti()} "
              f"tgt_fix=({tf[0]:+.3f},{tf[1]:+.3f},{tf[2]:+.3f}) "
              f"L/U/D/W/C={latches()} settled={bool(scene.settled()[0])} "
              f"score={sc():.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402

    def fix_to_world(loc) -> torch.Tensor:
        p = quat_apply(scene.fixture.data.root_quat_w[0:1],
                       torch.tensor([loc], device=device))[0]
        return p + scene.fixture.data.root_pos_w[0]

    def fix_quat() -> torch.Tensor:
        return scene.fixture.data.root_quat_w[0]

    def arm_pt(loc) -> torch.Tensor:
        return scene.arm_to_world(torch.tensor([loc], device=device))[0]

    def tp(body, pos_w, quat_w) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat_w
        body.write_root_state_to_sim(st, all_ids)

    # ----- hinge torque servo (same plant recipe the solve demonstrated) -----------------------
    servo = {"kp": 1.8, "kd": 0.6, "cap": 4.0, "des": None}
    ARM_G = 0.941 * 9.81 * 0.074
    BOWL_G = c.bowl_mass * 9.81 * (c.cav_x0 + c.cav_w / 2)

    def servo_reset() -> None:
        servo.update(kp=1.8, kd=0.6, cap=4.0, des=None)

    def clear_torque() -> None:
        scene.arm.set_external_force_and_torque(zero3, zero3)

    def apply_tau(tau: float) -> None:
        t = torch.zeros(1, 1, 3, device=device)
        t[0, 0, 1] = -tau
        scene.arm.set_external_force_and_torque(zero3, t)

    def gravity_ff(th: float) -> float:
        aboard = bool(scene.bowls_in_scoop()[0].any())
        return (ARM_G + (BOWL_G if aboard else 0.0)) * math.cos(math.radians(th))

    def servo_tick(target: float, slew: float = 60.0) -> None:
        th = th_now()
        rate = float(scene.arm_rate[0])
        if servo["des"] is None:
            servo["des"] = th
        d = target - servo["des"]
        stp = slew * dt
        servo["des"] += max(-stp, min(stp, d))
        tau = (gravity_ff(th) + servo["kp"] * math.radians(servo["des"] - th)
               - servo["kd"] * math.radians(rate))
        apply_tau(max(-servo["cap"], min(servo["cap"], tau)))
        step(1)

    def swing_to(target: float, tol: float = 2.5, max_steps: int = 2400) -> bool:
        probe_th, probe_i = th_now(), 0
        for i in range(max_steps):
            if abs(th_now() - target) < tol:
                return True
            servo_tick(target)
            if i - probe_i >= 90:
                cur = th_now()
                if abs(cur - probe_th) < 0.5 and abs(cur - target) > tol:
                    servo["kp"] = min(servo["kp"] * 1.5, 4.0)
                    servo["cap"] = min(servo["cap"] * 1.4, 7.0)
                probe_th, probe_i = cur, i
        return abs(th_now() - target) < tol

    def hold(target: float, steps: int, done=None) -> bool:
        for _ in range(steps):
            if done is not None and done():
                return True
            servo_tick(target)
        return done() if done is not None else True

    def fast_sweep() -> None:
        """Rate-servo ~180 deg/s to just short of the dump stop (the slick pocket
        starts releasing near 104 deg; speed keeps the load pinned by centrifugal
        force until the mouth reaches the window)."""
        w_des = math.radians(180.0)
        for _ in range(240):
            if th_now() >= float(c.dump_stop_deg) - 2.0:
                return
            tau = gravity_ff(th_now()) + 2.0 * (w_des - math.radians(float(scene.arm_rate[0])))
            apply_tau(max(-6.0, min(6.0, tau)))
            step(1)

    def press_until(done, steps: int) -> bool:
        for _ in range(steps):
            if done():
                return True
            apply_tau(1.2)
            step(1)
        return done()

    # ----- force push with the pod-dependent frame probe ---------------------------------------
    def clear_forces() -> None:
        for b in scene.bowls:
            b.set_external_force_and_torque(zero3, zero3)

    def drive(body, axis: torch.Tensor, fmag: float, vmax: float, done,
              max_steps: int) -> bool:
        mode = 0
        last_probe = float((body.data.root_pos_w[0] * axis).sum())
        for i in range(max_steps):
            if done():
                clear_forces()
                return True
            v = float((body.data.root_lin_vel_w[0] * axis).sum())
            f = fmag if v < vmax else 0.0
            fw = f * axis
            if mode == 1:
                fw = quat_apply_inverse(body.data.root_quat_w[0:1], fw.view(1, 3))[0]
            clear_forces()
            body.set_external_force_and_torque(fw.view(1, 1, 3), zero3)
            step(1)
            if (i + 1) % 40 == 0:
                cur = float((body.data.root_pos_w[0] * axis).sum())
                if cur - last_probe < 0.001:
                    mode ^= 1
                last_probe = cur
        clear_forces()
        return done()

    # ----- the honest mechanism, on torque drive (mirrors the demonstrated solve) --------------
    def oracle(tag: str) -> tuple[bool, tuple]:
        step(30)
        s_idle = sc()
        servo_reset()
        ok_h = swing_to(0.0, tol=2.0)
        hold(0.0, 60)
        s_before = sc()
        tp(tgt(), arm_pt([float(c.cav_x0 + c.cav_w / 2), 0.0, 0.100]),
           scene.arm.data.root_quat_w[0])
        assert abs(sc() - s_before) < 1e-6, "transport teleport moved the score"
        ok_l = hold(0.0, 480,
                    done=lambda: bool(scene._loaded_ever[0, ti()])
                    and float(tgt().data.root_lin_vel_w[0].norm()) < 0.05)
        aboard_live = bool(scene.bowls_in_scoop()[0, ti()])  # probe not vacuous
        s_load = sc()
        swing_to(70.0, tol=8.0)
        hold(70.0, 120, done=lambda: bool(scene._lifted_ever[0, ti()]))
        s_lift = sc()
        swing_to(103.0, tol=8.0)
        hold(103.0, 120, done=lambda: bool(scene._delivered_ever[0, ti()]))
        s_del = sc()
        fast_sweep()
        ok_p = press_until(lambda: bool(scene._courted_ever[0, ti()]), 720)
        s_pour = sc()
        clear_torque()
        ok_fin = settle_until(lambda: bool(scene.success()[0]), max_steps=600)
        step(240)  # persistence: success must not flicker off
        report(f"{tag}-final")
        ok = (s_idle <= 0.02 and ok_h and ok_l and aboard_live and ok_p and ok_fin
              and bool(scene.success()[0]) and sc() == 1.0 and th_now() > 100.0)
        return ok, (s_idle, s_load, s_lift, s_del, s_pour)

    # =========================== 1-2. settle / no-NaN ============================================
    torch.manual_seed(11)
    env.reset()
    step(90)
    report("settled")
    st0 = torch.cat([scene.fixture.data.root_state_w, scene.base.data.root_state_w,
                     scene.arm.data.root_state_w, scene.plate.data.root_state_w]
                    + [b.data.root_state_w for b in scene.bowls], dim=-1)
    bp = scene.bowl_pos_fix()[0]  # (3, 3)
    ys = bp[:, 1]
    mid_ok = bool(ys[ti()] > ys.min() + 0.03) and bool(ys[ti()] < ys.max() - 0.03)
    check("settle: states finite, arm at rest on the load stop, three bowls in a row "
          "on the staging pad with the TARGET strictly in the middle, court empty",
          bool(torch.isfinite(st0).all())
          and abs(th_now() + c.load_stop_deg) < 2.0
          and bool((bp[:, 0] < -0.32).all()) and bool((bp[:, 2] > 0.11).all())
          and bool((bp[:, 2] < 0.15).all()) and mid_ok
          and not bool(scene.bowls_in_court()[0].any())
          and not bool(scene.plate_in_court()[0])
          and bool(scene.settled()[0]))
    check("settle: score ~0 at reset, no success, no latches",
          sc() <= 0.005 and not bool(scene.success()[0]) and no_latch())

    # =========================== 3. randomization is real ========================================
    reads = []
    for s in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(s)
        env.reset()
        step(4)
        fp = (scene.fixture.data.root_pos_w[0] - scene.env_origins[0])
        fq = scene.fixture.data.root_quat_w[0]
        fyaw = 2.0 * math.atan2(float(fq[3]), float(fq[0]))
        base_fix = scene._to_frame(scene.fixture, scene.base.data.root_pos_w)[0]
        bpx = scene.bowl_pos_fix()[0]
        spread_y = float(bpx[:, 1].max() - bpx[:, 1].min())  # row span = 2*spacing-ish
        ex = quat_apply(scene.bowls[0].data.root_quat_w[0:1],
                        torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
        byaw = math.atan2(float(ex[1]), float(ex[0]))
        reads.append((float(fp[0]), float(fp[1]), fyaw, float(scene.target_idx[0]),
                      float(base_fix[0]), float(base_fix[1]), spread_y, byaw))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (fix_x, fix_y, fix_yaw, target_idx, "
          f"base_x, base_y, row_span, bowl0_yaw):\n{np.round(arr, 3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    n_tgt = len(set(int(v) for v in arr[:, 3]))
    check("randomization: fixture root xy+yaw move, TARGET BODY INDEX varies (bowl->"
          "slot permutation), lift-base xy jitter, row spacing and bowl yaw move "
          "(READBACK from sim)",
          spread[0] > 0.02 and spread[1] > 0.02 and spread[2] > 0.05
          and n_tgt >= 2 and spread[4] > 0.004 and spread[5] > 0.008
          and spread[6] > 0.02 and spread[7] > 0.5)

    # =========================== 4. null policy fails ============================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    check("null policy: score ~0, no success after 240 idle steps",
          sc() <= 0.02 and not bool(scene.success()[0]) and no_latch())

    # =========================== 5-7. oracle on 3 seeds ==========================================
    ladder = None
    for s in (0, 1, 2):
        torch.manual_seed(s)
        env.reset()
        ok, lad = oracle(f"oracle{s}")
        if s == 0:
            ladder = lad
        check(f"oracle seed {s}: free-fall load (0.15), crank past 60 (0.30) and 95 "
              f"(0.50), fast sweep to the dump stop pours through the window into the "
              f"court (0.95) -> settled success 1.0, persists 240 steps", ok)

    # =========================== 8-9. rubric monotonicity ========================================
    s_idle, s_load, s_lift, s_del, s_pour = ladder
    print(f"[smoke] monotonicity ladder: idle={s_idle:.3f} loaded={s_load:.3f} "
          f"lifted={s_lift:.3f} delivered={s_del:.3f} poured={s_pour:.3f} final=1.000",
          flush=True)
    check("monotonicity: idle < loaded (0.15) < lifted (0.30) < delivered (0.50) < "
          "poured (>=0.95)",
          s_idle <= 0.02 and 0.14 <= s_load <= 0.16 and 0.29 <= s_lift <= 0.31
          and 0.49 <= s_del <= 0.51 and 0.93 <= s_pour <= 1.0
          and s_idle < s_load < s_lift < s_del < s_pour)
    check("monotonicity: pre-pour partial states score < 1.0",
          max(s_idle, s_load, s_lift, s_del) < 0.951)

    # =========================== 10. occupied-court forfeit ======================================
    # Continue from oracle 2's success state: a DECOY bowl at rest in the court.
    decoy = scene.bowls[(ti() + 1) % 3]
    tp(decoy, fix_to_world([-0.05, 0.04, 0.481]), fix_quat())
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    report("occupied")
    check("occupied forfeit: decoy bowl inside the court flips success OFF; score "
          "falls back to the latched 0.95",
          bool(scene.decoy_in_court()[0]) and not bool(scene.success()[0])
          and 0.94 <= sc() <= 0.951)

    # =========================== 11. latch permanence + plate forfeit ============================
    tp(decoy, fix_to_world([0.10, 0.45, 0.021]), fix_quat())  # back to the floor
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    recovered = bool(scene.success()[0])  # latches persist -> success recovers
    tp(scene.plate, fix_to_world([-0.03, -0.04, 0.470]), fix_quat())
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    report("plate-in-court")
    check("latch permanence + plate forfeit: decoy removed -> success RECOVERS "
          "(latched history persists); the plate at rest in the court kills it again",
          recovered and bool(scene.plate_in_court()[0])
          and not bool(scene.success()[0]) and 0.94 <= sc() <= 0.951)

    # =========================== 12. negative: the seed's strategy ===============================
    # The seed's plan — release the bowl above the cabinet — parks it ON THE ROOF.
    torch.manual_seed(51)
    env.reset()
    step(30)
    tp(tgt(), fix_to_world([0.0, 0.0, 0.72]), fix_quat())
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    step(60)
    zf = float(tgt_fix()[2])
    report("roof-park")
    check("negative (seed strategy): bowl released above the cabinet lands ON THE "
          "ROOF (rest z readback above the roof top) -> not seated, no latch, "
          "score 0, no success",
          zf > float(c.roof_top_z) + 0.005 and zf < 0.70
          and not bool(scene.target_seated()[0]) and no_latch()
          and sc() <= 1e-6 and not bool(scene.success()[0]))

    # =========================== 13. negative: teleport bypass ===================================
    torch.manual_seed(61)
    env.reset()
    step(30)
    tp(tgt(), fix_to_world([0.02, 0.0, 0.481]), fix_quat())
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    step(60)
    live_seat = bool(scene.target_seated()[0])
    report("bypass")
    check("negative (teleport bypass): target placed at rest ON the court seat — "
          "live seated predicate TRUE (probe not vacuous) yet every mechanism latch "
          "refuses -> score 0, no success",
          live_seat and no_latch() and sc() <= 1e-6
          and not bool(scene.success()[0]))

    # =========================== 14-15. negative: window gating + walk-in ========================
    # Drop the bowl onto the window SILL (the seat strip inside the window arc, outside
    # the court volume): the live window-band predicate fires during the fall.
    torch.manual_seed(71)
    env.reset()
    step(30)
    tp(tgt(), fix_to_world([0.147, 0.0, 0.55]), fix_quat())
    win_live = False
    for _ in range(60):
        step(1)
        win_live |= bool(scene.bowls_in_window()[0, ti()])
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    report("sill-drop")
    check("negative (window band gating): bowl dropped onto the window sill passes "
          "the live window-band predicate mid-fall (not vacuous) but the windowed "
          "latch refuses without the delivered history -> score 0",
          win_live and not bool(scene._windowed_ever[0].any()) and no_latch()
          and sc() <= 1e-6 and not bool(scene.success()[0]))

    # Now PUSH it from the sill through the window into the court (a hand-carried
    # entry): live courted+seated go TRUE, yet nothing latches.
    ax = quat_apply(scene.fixture.data.root_quat_w[0:1],
                    torch.tensor([[-1.0, 0.0, 0.0]], device=device))[0]
    x_start = float(tgt_fix()[0])
    ok_push = drive(tgt(), ax, 3.0, 0.08,
                    lambda: float(tgt_fix()[0]) < 0.09, 1500)
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    step(60)
    report("walk-in")
    check("negative (window walk-in): bowl force-pushed from the sill into the court "
          "(actuator verified: moved >= 4 cm) — live courted AND seated TRUE, all "
          "mechanism latches still refuse -> score 0, no success",
          ok_push and (x_start - float(tgt_fix()[0])) > 0.04
          and bool(scene.bowls_in_court()[0, ti()])
          and bool(scene.target_seated()[0]) and no_latch()
          and sc() <= 1e-6 and not bool(scene.success()[0]))

    # =========================== 16. negative: out of order ======================================
    # Crank the EMPTY arm to the dump stop (earns nothing), then feed the bowl into
    # the pocket at the top: it slides out through the window, but the loaded/lifted/
    # delivered history never existed, so nothing latches.
    torch.manual_seed(81)
    env.reset()
    step(30)
    servo_reset()
    swing_to(0.0, tol=2.0)
    swing_to(70.0, tol=8.0)
    fast_sweep()
    press_until(lambda: False, 120)  # seat firmly on the dump stop
    clear_torque()
    step(60)
    empty_clean = no_latch() and sc() <= 1e-6 and th_now() > 100.0
    tp(tgt(), arm_pt([float(c.cav_x0 + c.cav_w / 2), 0.0, 0.005]),
       scene.arm.data.root_quat_w[0])
    settle_until(lambda: bool(scene.settled()[0]), max_steps=600)
    step(60)
    left_scoop = not bool(scene.bowls_in_scoop()[0, ti()])  # it did slide out
    report("top-feed")
    check("negative (out of order): empty crank to the dump stop earns nothing; a "
          "bowl fed into the pocket at the top slides out through the window "
          "(readback: left the scoop) yet loaded/lifted/delivered never fired -> "
          "no latch, score 0, no success",
          empty_clean and left_scoop and no_latch() and sc() <= 1e-6
          and not bool(scene.success()[0]))

    # =========================== save + verdict ==================================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.scoop_lift_court")
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

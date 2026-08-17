"""Smoke / rubric-REJECTION battery for DumpHopperScene (sim_gen task
`living_room_scene2_pick_up_the_milk_and_put_it_in_the_basket_i177`) — NullRobot,
teleported probe states, RECORDED.

This is NOT a solution (solve.py — carry the basket to the mark, press the lever,
catch the discharge — is the acceptance evidence). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled state and
asserts the rubric REJECTS it, plus physics probes that prove the mechanism is real
(the cage genuinely holds the carton, the tray genuinely tips under the lever torque
and genuinely gravity-returns, the discharge is genuine sliding + ballistics). Note
on the seed's strategy: the seed's end state (milk inside the basket) IS this task's
goal, so "seed end state -> rejected" is not expressible; what IS rejected is the
seed's MEANS — grasping the milk — which the cage physically forbids (probe 7), and
the out-of-order shortcut of dumping without staging the basket (probe 8). Two probes
DO construct the genuine end state on purpose — the acceptance run (13) and the
decoy-removed flip (15) — every other judged point must stay success()=False and a
final audit asserts exactly that.

  1-2.  settle/no-NaN     — reset settles finite; tray LEVEL (readback |tilt| < 2
                            deg), milk caged; score ~0, no success;
  3-4.  randomization     — READBACK over 8 seeded resets: station xy + free yaw
                            vary; milk (tray frame), basket and juice all vary;
  5.   null policy        — 300 idle steps -> score ~0, no success (the basket
                            spawns well off the catch mark, the milk stays caged);
  6.   carried, not placed— the basket held (posed) ALOFT over the catch mark: no
                            zone credit (it must stand ON the floor), no success;
  7.   captivity (roof)   — the seed's MEANS, physically unavailable: 15 N of pull
                            (4x the carton's weight, a proxy for a grasp-and-lift)
                            hoists the caged carton — it RISES (non-vacuous) but is
                            capped by the roof, and settles back inside the cage; no
                            release credit, no success;
  8.   out-of-order dump  — the lever pressed with NO basket staged: the carton
                            slides out and lands on the FLOOR — partial credit only
                            (0.45), no success (dumping first strands the milk);
  9.   gravity-return     — after the torque clears, the tray returns level (< 3
                            deg) ON ITS OWN — the back-heavy return is real;
  10.  leaning outside    — the carton stood against the basket's OUTER wall:
                            released credit only, no success;
  11.  tipped basket      — the basket on its SIDE with the carton resting in its
                            mouth: the upright gate rejects (no success);
  12.  airborne basket    — the carton posed inside a basket held 0.4 m up, judged
                            immediately: the on-floor gate rejects (no success);
  13.  acceptance run     — basket placed on the mark, lever pressed and held: the
                            carton slides out, drops in, the tray returns, all
                            settle -> success TRUE (the rubric accepts exactly the
                            mechanism-produced end state);
  14.  decoy over basket  — the RED juice carton posed in the violation volume over
                            the basket: success flips FALSE (decoy clause);
  15.  decoy removed      — juice back on the floor: success returns TRUE (14's
                            rejection was the decoy clause and nothing else);
  16.  settle gate        — the delivered carton kicked and judged immediately: NOT
                            success (must be at rest);
  17.  wrong object       — the JUICE carton placed in the basket while the milk
                            stays caged: rejected twice over (milk not in the
                            basket AND the decoy clause);
  18.  rejection audit    — success() was never True at any judged point EXCEPT the
                            two constructed acceptance probes (13 and 15);
  19.  final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.<task>.smoke --headless
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
    from .scene import (  # noqa: F401
        BK_HALF, BK_PLATE_T, H_HINGE, MILK_H, MILK_W, TILT_MAX_DEG, ZONE_X,
        _qapply, _qmul, _qy, _qz,
    )
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import (  # noqa: F401
        BK_HALF, BK_PLATE_T, H_HINGE, MILK_H, MILK_W, TILT_MAX_DEG, ZONE_X,
        _qapply, _qmul, _qy, _qz,
    )
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

BASKET_X = 0.325  # solve's basket placement on the catch mark (station frame)
RATE_CAP = 0.33  # tilt-rate cap during a press, deg per step
TAU_LO = 0.25


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.dump_hopper")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.30, -1.10, 0.85)) + o),
                                tuple(np.array((0.15, 0.00, 0.25)) + o),
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

    ever_bad_success = [False]

    def judge() -> tuple[float, bool]:
        """Judge a REJECTION probe: success here is a rubric failure."""
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_bad_success[0] = ever_bad_success[0] or ok
        return s, ok

    def judge_accept() -> tuple[float, bool]:
        """Judge an ACCEPTANCE construct: success here is expected and allowed."""
        return float(scene.score()[0]), bool(scene.success()[0])

    def tilt() -> float:
        return float(scene.tilt_deg()[0])

    def report(tag: str, s: float, ok: bool) -> None:
        ms = scene.stn_local(scene.milk.data.root_pos_w)[0]
        bs = scene.stn_local(scene.basket.data.root_pos_w)[0]
        print(f"[smoke] {tag:18s} | tilt={tilt():+6.2f}deg "
              f"milk_stn=({float(ms[0]):+.3f},{float(ms[1]):+.3f},{float(ms[2]):+.3f}) "
              f"basket_stn=({float(bs[0]):+.3f},{float(bs[1]):+.3f}) "
              f"caged={bool(scene.milk_in_cage()[0])} "
              f"rel={bool(scene.milk_released()[0])} "
              f"inb={bool(scene.in_basket(scene.milk)[0])} "
              f"upright={bool(scene.basket_upright_on_floor()[0])} "
              f"decoy_out={bool(scene.decoy_out()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def stn_pose(local, extra_quat=None):
        q_stn = scene.station.data.root_quat_w
        pos = scene.station.data.root_pos_w + _qapply(
            q_stn, torch.tensor(local, device=device).expand(n, 3))
        q = q_stn if extra_quat is None else _qmul(q_stn, extra_quat)
        return pos, q

    def basket_pose(local, extra_quat=None):
        q_bk = scene.basket.data.root_quat_w
        pos = scene.basket.data.root_pos_w + _qapply(
            q_bk, torch.tensor(local, device=device).expand(n, 3))
        q = q_bk if extra_quat is None else _qmul(q_bk, extra_quat)
        return pos, q

    def teleport(body, pos, quat, vel=None, ang=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3:7] = quat
        if vel is not None:
            st[:, 7:10] = torch.tensor(vel, device=device)
        if ang is not None:
            st[:, 10:13] = torch.tensor(ang, device=device)
        body.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def settle_all(max_steps: int = 600) -> None:
        for _ in range(max_steps // 30):
            step(30)
            if bool(scene.settled(scene.milk)[0]) and bool(scene.settled(scene.basket)[0]) \
                    and bool(scene.settled(scene.tray)[0]) \
                    and bool(scene.settled(scene.juice)[0]):
                break

    def clear_tray_torque() -> None:
        scene.tray.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                 env_ids=all_ids)

    def _press_once(need_basket: bool, tau_max: float, frame: str, tag: str) -> bool:
        """One press attempt: bang-bang lever torque under a tilt-rate cap, held
        against the 35 deg stop until the carton is discharged (into the basket when
        `need_basket`, onto the floor otherwise). The torque is applied in the TRAY
        BODY frame (0, tau, 0) — the hinge axis is body-y at every tilt and station
        yaw, sidestepping the pod's global-wrench frame drag (its reference is
        captured at the FIRST application and goes stale across resets);
        frame="world" is the station-frame fallback. Clears the torque after."""
        th_prev = tilt()
        streak = 0
        for i in range(1800):
            th = tilt()
            w = th - th_prev
            th_prev = th
            tau_hi = min(0.4 + tau_max * (i / 300.0), tau_max)
            tau = TAU_LO if w > RATE_CAP else tau_hi
            t_vec = torch.tensor([0.0, tau, 0.0], device=device).expand(n, 3)
            if frame == "body":
                scene.tray.set_external_force_and_torque(
                    zero_wrench, t_vec.view(n, 1, 3), env_ids=all_ids)
            else:
                t_world = _qapply(scene.station.data.root_quat_w, t_vec)
                scene.tray.set_external_force_and_torque(
                    zero_wrench, t_world.view(n, 1, 3), env_ids=all_ids, is_global=True)
            step(1)
            if i == 479 and tilt() < 5.0:  # stall probe: wrong frame / torque too low
                clear_tray_torque()
                print(f"[smoke] {tag}: no tilt progress after 4 s "
                      f"(tilt {tilt():.2f} deg)", flush=True)
                return False
            if bool(scene.milk_released()[0]):
                if need_basket:
                    if bool(scene.in_basket(scene.milk)[0]):
                        streak += 1
                        if streak >= 20:
                            clear_tray_torque()
                            return True
                    else:
                        streak = 0
                else:
                    mz = float((scene.milk.data.root_pos_w - scene.env_origins)[0, 2])
                    mv = float(scene.milk.data.root_lin_vel_w[0].norm())
                    if mz < 0.10 and mv < 0.10:
                        clear_tray_torque()
                        return True
        clear_tray_torque()
        print(f"[smoke] {tag}: press budget exhausted (tilt {tilt():.2f} deg)", flush=True)
        return False

    def press_dump(need_basket: bool, tag: str) -> bool:
        """Operate the real mechanism, retrying across torque frames/magnitudes (a
        stalled attempt leaves the tray level and the carton caged — nothing moved)."""
        for tau_max, frame in ((2.2, "body"), (2.2, "world"), (3.5, "body")):
            if _press_once(need_basket, tau_max, frame, f"{tag}[{frame}/tau{tau_max:.1f}]"):
                return True
            step(60)  # let anything transient die out before the next attempt
        return False

    def all_finite() -> bool:
        bodies = [scene.station, scene.tray, scene.milk, scene.juice, scene.basket]
        return all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)

    up_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4)
    half_pi = torch.full((n,), math.pi / 2, device=device)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    settle_all(480)
    s, ok = judge()
    report("reset", s, ok)
    check("settle: all states finite, tray LEVEL (readback "
          f"tilt={tilt():+.2f}deg, |tilt| < 2), milk caged",
          all_finite() and abs(tilt()) < 2.0 and bool(scene.milk_in_cage()[0]))
    check(f"settle: score ~0 and no success on a fresh reset (score={s:.3f})",
          s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(20)
        sp = (scene.station.data.root_pos_w - scene.env_origins)[0]
        ex = _qapply(scene.station.data.root_quat_w,
                     torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))[0]
        yaw = math.atan2(float(ex[1]), float(ex[0]))
        ml = scene.tray_local(scene.milk.data.root_pos_w)[0]
        bk = (scene.basket.data.root_pos_w - scene.env_origins)[0]
        ju = (scene.juice.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(sp[0]), float(sp[1]), yaw, float(ml[0]), float(ml[1]),
                      float(bk[0]), float(bk[1]), float(ju[0]), float(ju[1])))
    arr = np.array(reads)
    print("[smoke] randomization readback (stn_x, stn_y, stn_yaw, milk_tx, milk_ty, "
          f"bk_x, bk_y, ju_x, ju_y):\n{arr.round(3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: station pose varies across seeded resets (readback: "
          f"dx={spread[0]:.3f} dy={spread[1]:.3f} dyaw={spread[2]:.2f} rad)",
          spread[0] > 0.02 and spread[1] > 0.02 and spread[2] > 0.8)
    check("randomization: milk (tray frame), basket and juice all vary (readback: "
          f"dmilk={max(spread[3], spread[4]):.3f} dbasket={max(spread[5], spread[6]):.3f} "
          f"djuice={max(spread[7], spread[8]):.3f})",
          max(spread[3], spread[4]) > 0.015 and max(spread[5], spread[6]) > 0.10
          and max(spread[7], spread[8]) > 0.10)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(300)
    s, ok = judge()
    report("null-policy", s, ok)
    check(f"null policy: score ~0 and no success after 300 idle steps (score={s:.3f})",
          s <= 0.02 and not ok)

    # =========================== 6. carried, not placed =====================================
    env.reset(seed=41)
    settle_all(300)
    pos, q = stn_pose((BASKET_X, 0.0, 0.20))
    teleport(scene.basket, pos, q, settle_steps=2)  # judge while held aloft
    s, ok = judge()
    report("carried", s, ok)
    check("carried, not placed: the basket POSED aloft over the catch mark earns no "
          f"zone credit (must stand ON the floor) — score={s:.3f} ~0, no success",
          s <= 0.02 and not ok and not bool(scene.basket_in_zone()[0]))
    # park the basket far away for the captivity + no-basket-dump probes
    pos, q = stn_pose((-0.60, 0.35, BK_PLATE_T + 0.003))
    teleport(scene.basket, pos, q, settle_steps=60)

    # =========================== 7. captivity: the roof bars lifting ========================
    # The seed's MEANS — grasp the milk and lift it out — attempted as physics: 15 N
    # of upward pull (4x the carton's 3.4 N weight). Non-vacuous: the carton must
    # actually RISE, then be capped by the roof, and it must settle back caged.
    z_rest = float(scene.tray_local(scene.milk.data.root_pos_w)[0, 2])
    peak_z = z_rest
    lift = torch.tensor([0.0, 0.0, 15.0], device=device).expand(n, 3)
    for _ in range(150):
        scene.milk.set_external_force_and_torque(
            lift.view(n, 1, 3), zero_wrench, env_ids=all_ids, is_global=True)
        step(1)
        peak_z = max(peak_z, float(scene.tray_local(scene.milk.data.root_pos_w)[0, 2]))
    scene.milk.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    settle_all(360)
    s, ok = judge()
    report("roof-captive", s, ok)
    check("captivity (roof): 15 N of lift hoists the caged carton "
          f"(rose {(peak_z - z_rest) * 1000:.0f}mm >= 25mm, non-vacuous) but the "
          f"roof caps it (peak z={peak_z * 1000:.0f}mm "
          f"< 140mm) and it settles back CAGED — no release credit, no success "
          f"(score={s:.3f})",
          peak_z - z_rest >= 0.025 and peak_z < 0.140
          and bool(scene.milk_in_cage()[0]) and not bool(scene.milk_released()[0])
          and s <= 0.02 and not ok)

    # =========================== 8-9. out-of-order dump (no basket) =========================
    dumped = press_dump(need_basket=False, tag="dump-no-basket")
    settle_all(480)
    s, ok = judge()
    report("dump-no-basket", s, ok)
    ms = scene.stn_local(scene.milk.data.root_pos_w)[0]
    check("out-of-order dump (no basket staged): the lever press discharges the "
          f"carton onto the FLOOR (landed at station ({float(ms[0]):+.3f},"
          f"{float(ms[1]):+.3f})) — released credit only (score={s:.3f} in "
          f"[0.40,0.50]), no success",
          dumped and bool(scene.milk_released()[0]) and 0.40 <= s <= 0.50 and not ok
          and not bool(scene.in_basket(scene.milk)[0]))
    check("gravity-return: with the torque cleared the tray returned LEVEL on its "
          f"own (readback tilt={tilt():+.2f}deg, was held at ~{TILT_MAX_DEG:.0f}deg)",
          abs(tilt()) < 3.0)

    # =========================== 10. leaning against the basket outside =====================
    pos, q = basket_pose((BK_HALF + 0.040, 0.0, MILK_H / 2 + 0.003), None)
    teleport(scene.milk, pos, up_quat, settle_steps=180)
    s, ok = judge()
    report("leaning-outside", s, ok)
    check("leaning outside: the carton stood against the basket's OUTER wall is "
          f"released credit only (score={s:.3f} <= 0.50), not in the basket, no "
          "success", s <= 0.50 and not ok and not bool(scene.in_basket(scene.milk)[0]))

    # =========================== 11. tipped basket ==========================================
    env.reset(seed=51)
    settle_all(300)
    q_roll = _qy(half_pi)  # basket rolled onto its side (mouth horizontal)
    pos, q = stn_pose((0.60, 0.10, BK_HALF + 0.003), q_roll)
    teleport(scene.basket, pos, q, settle_steps=30)
    pos, q = basket_pose((0.0, 0.0, 0.06), None)  # in the mouth of the tipped basket
    teleport(scene.milk, pos, scene.basket.data.root_quat_w, settle_steps=240)
    s, ok = judge()
    report("tipped-basket", s, ok)
    check("tipped basket: the basket on its SIDE with the carton resting in its "
          f"mouth is rejected by the upright gate (upright="
          f"{bool(scene.basket_upright_on_floor()[0])}) — no success",
          not ok and not bool(scene.basket_upright_on_floor()[0]))

    # =========================== 12. airborne basket ========================================
    pos, q = stn_pose((0.55, -0.30, 0.40))
    teleport(scene.basket, pos, q, settle_steps=0)
    pos, q = basket_pose((0.0, 0.0, 0.05), None)
    teleport(scene.milk, pos, scene.basket.data.root_quat_w, settle_steps=0)
    step(2)  # refresh buffers — judge while held aloft
    s, ok = judge()
    report("airborne-basket", s, ok)
    check("airborne basket: the carton posed inside a basket held 0.4 m up, judged "
          "immediately, is rejected by the on-floor gate — no success "
          f"(score={s:.3f})", not ok and not bool(scene.basket_upright_on_floor()[0]))

    # =========================== 13. acceptance run (the real mechanism) ====================
    env.reset(seed=61)
    settle_all(300)
    pos, q = stn_pose((BASKET_X, 0.0, BK_PLATE_T + 0.003))
    teleport(scene.basket, pos, q, settle_steps=90)
    caught = press_dump(need_basket=True, tag="acceptance")
    settle_all(600)
    s, ok = judge_accept()
    report("accept-run", s, ok)
    check("acceptance run: basket on the mark + lever pressed and held — the carton "
          "slid out, dropped in, the tray returned, everything settled -> success "
          f"TRUE (score={s:.3f}, tilt={tilt():+.2f}deg)",
          caught and ok and s >= 0.99 and abs(tilt()) < 3.0)

    # =========================== 14-15. decoy clause flips success ==========================
    pos, q = basket_pose((0.0, 0.09, 0.22), None)  # in the violation volume OVER the rim
    teleport(scene.juice, pos, up_quat, settle_steps=2)  # judge before it falls in
    s, ok = judge()
    report("decoy-over", s, ok)
    check("decoy over the basket: the RED juice carton posed in the violation volume "
          "over the rim flips success FALSE (decoy clause), milk untouched",
          not ok and not bool(scene.decoy_out()[0])
          and bool(scene.in_basket(scene.milk)[0]))
    pos, q = stn_pose((-0.55, -0.35, MILK_H / 2 + 0.003))
    teleport(scene.juice, pos, up_quat, settle_steps=120)
    s, ok = judge_accept()
    report("decoy-removed", s, ok)
    check("decoy removed: juice back on the floor -> success returns TRUE (14's "
          "rejection was the decoy clause and nothing else)", ok)

    # =========================== 16. settle gate ============================================
    pos = scene.milk.data.root_pos_w
    q = scene.milk.data.root_quat_w
    teleport(scene.milk, pos, q, vel=[0.0, 0.0, 0.35], ang=[0.0, 0.0, 6.0],
             settle_steps=0)
    step(1)  # refresh buffers only — judge while still moving
    lv = float(scene.milk.data.root_lin_vel_w[0].norm())
    av = float(scene.milk.data.root_ang_vel_w[0].norm())
    s, ok = judge()
    report("settle-gate", s, ok)
    check("settle gate: the delivered carton kicked (lin={:.2f} m/s, ang={:.1f} "
          "rad/s) and judged immediately is NOT success (must be at rest)"
          .format(lv, av), (lv > c.settle_lin or av > c.settle_ang) and not ok)

    # =========================== 17. wrong object ===========================================
    env.reset(seed=81)
    settle_all(300)
    pos, q = stn_pose((BASKET_X, 0.0, BK_PLATE_T + 0.003))
    teleport(scene.basket, pos, q, settle_steps=90)
    pos, q = basket_pose((0.0, 0.0, MILK_H / 2 + 0.005), None)
    teleport(scene.juice, pos, scene.basket.data.root_quat_w, settle_steps=240)
    s, ok = judge()
    report("wrong-object", s, ok)
    check("wrong object: the JUICE carton placed in the basket while the milk stays "
          f"caged is rejected twice over — milk not in the basket AND the decoy "
          f"clause (score={s:.3f} <= 0.16, zone credit at most)",
          not ok and s <= 0.16 and bool(scene.milk_in_cage()[0])
          and not bool(scene.decoy_out()[0]))

    # =========================== 18-19. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point except the "
          "two constructed acceptance probes", not ever_bad_success[0])
    check("final: all task-object states finite (no NaN)", all_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.dump_hopper")
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

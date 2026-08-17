"""Smoke / rubric-REJECTION battery for SauceCarouselScene (sim_gen task
`libero_pick_tomato_sauce_i235`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — spin the carousel by crank torque until the RED can
parks in the hatch, lift it over the sill, drop it into the basket — is the acceptance
evidence). Every teleport here is instrumentation that CONSTRUCTS a wrong (or partial)
outcome and asserts the rubric REJECTS it, plus physics probes that prove the mechanism
is real (the platter genuinely spins and carries the cans; the rotunda genuinely cages
a kicked can). One probe DOES construct the genuine end state on purpose — the
acceptance mini-solve — and the flip pairs re-enter it; every other judged point must
stay success()=False and a final audit asserts exactly that.

  1.    settle/no-NaN     — reset settles finite; both cans riding the platter deep
                            under the roof (>= 60 deg off the hatch), score ~0;
  2-4.  randomization     — READBACK over 10 seeded resets: rotunda xy + free yaw vary;
                            the sauce can's hatch bearing varies AND occurs on BOTH
                            sides (decoy always opposite); the basket ring pose varies;
  5.    null policy       — 300 idle steps -> score ~0, no success;
  6.    mechanism is real — a z-torque rate servo on the platter (the crank proxy)
                            spins it >= 25 deg and BOTH cans ride by friction alone
                            (sauce bearing tracks the spin, decoy stays aboard);
  7.    rotunda cages     — the riding sauce can kicked hard radially outward (under
                            the roofed sector) CANNOT leave the rotunda: fence + roof
                            + narrow moat cage it (readback: still inside the ring);
  8.    seed strategy /   — the can written INTO the basket by fiat from a fresh reset
        anti-teleport       (the seed's reach-in-and-carry, physically impossible
                            here): ordered latches never fired -> score ~0, no success;
  9.    wrong object      — the WHITE decoy written into the basket instead: rejected
                            (and the decoy clause is violated on top);
  10.   acceptance        — legal mini-solve: crank-torque spin parks the RED can in
                            the hatch (judged: 0.40, NOT success), then the
                            open-sky lift-out + basket drop -> success TRUE, 1.0;
  11-12. sauce removed    — from acceptance, the can stood on the open floor: success
                            flips FALSE and the score falls to the latched 0.60 cap;
                            set_state restore -> success returns TRUE;
  13-14. decoy dumped     — the decoy teleported off the platter to the floor: success
                            flips FALSE (decoy clause); restore -> TRUE;
  15.   near miss (wall)  — the can standing on the ground flush against the basket's
                            OUTER wall: containment gates reject it;
  16.   near miss (rim z) — the can hovering over the basket mouth ABOVE the height
                            band (xy inside): the z gate alone rejects it;
  17.   settle gate       — the placed can kicked inside the basket and judged
                            immediately: NOT success (must be at rest); resettled
                            against the walls -> success returns;
  18.   rejection audit   — success() was never True at any judged point EXCEPT the
                            acceptance construct and the flip-backs (10, 12, 14, 17b);
  19.   final no-NaN      — all task-object states finite at the end;
  20.   video             — > 10 frames captured and saved to frames.npz.

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
        BKT_FLOOR_T, BKT_IN_HALF, BKT_WALL_T, CAN_H, CAN_R, FENCE_IN, _qapply, _qz,
    )
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import (  # noqa: F401
        BKT_FLOOR_T, BKT_IN_HALF, BKT_WALL_T, CAN_H, CAN_R, FENCE_IN, _qapply, _qz,
    )
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.sauce_carousel")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
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
        env.sim.set_camera_view(tuple(np.array((1.30, -1.05, 0.95)) + o),
                                tuple(np.array((0.00, 0.00, 0.12)) + o),
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
            scene.score()  # keep the latches current (score is the running judge)
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

    def th_deg(body) -> float:
        return math.degrees(float(scene.hatch_angle(body)[0]))

    def plat_yaw() -> float:
        q = scene.platter.data.root_quat_w[0]
        return 2.0 * math.atan2(float(q[3]), float(q[0]))

    def report(tag: str, s: float, ok: bool) -> None:
        print(f"[smoke] {tag:18s} | th_s={th_deg(scene.sauce):+7.1f}deg "
              f"riding={bool(scene.on_platter(scene.sauce)[0])} "
              f"aligned={bool(scene.aligned_now()[0])} "
              f"out={bool(scene.out_now()[0])} "
              f"in_bkt={bool(scene.in_basket()[0])} "
              f"decoy={bool(scene.decoy_ok()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

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

    def clear_forces() -> None:
        scene.platter.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                    env_ids=all_ids)

    def apply_ztorque(tau: torch.Tensor) -> None:
        t = torch.zeros(n, 1, 3, device=device)
        t[:, 0, 2] = tau
        scene.platter.set_external_force_and_torque(zero_wrench, t, env_ids=all_ids,
                                                    is_global=True)

    ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4)

    def basket_pose(local_xyz) -> torch.Tensor:
        return scene.basket.data.root_pos_w + _qapply(
            scene.basket.data.root_quat_w,
            torch.tensor(local_xyz, device=device).expand(n, 3))

    def settle_all(max_steps: int = 600) -> None:
        for _ in range(max_steps // 30):
            step(30)
            if all(bool(scene.settled(b)[0]) for b in
                   (scene.platter, scene.sauce, scene.decoy, scene.basket)):
                break

    def all_finite() -> bool:
        bodies = [scene.rotunda, scene.platter, scene.sauce, scene.decoy, scene.basket]
        return all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)

    # =========================== 1. settle / no-NaN + layout ================================
    env.reset(seed=11)
    settle_all(480)
    s, ok = judge()
    report("reset", s, ok)
    check("settle: all states finite; both cans riding the platter >= 60 deg off the "
          f"hatch (sauce {th_deg(scene.sauce):+.1f} deg, decoy "
          f"{th_deg(scene.decoy):+.1f} deg), not aligned, score ~0, no success "
          f"(score={s:.3f})",
          all_finite() and bool(scene.on_platter(scene.sauce)[0])
          and bool(scene.decoy_ok()[0]) and abs(th_deg(scene.sauce)) >= 60.0
          and abs(th_deg(scene.decoy)) >= 60.0
          and not bool(scene.aligned_now()[0]) and s <= 0.02 and not ok)

    # =========================== 2-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28, 29, 30):
        env.reset(seed=sd)
        step(20)
        rp = (scene.rotunda.data.root_pos_w - scene.env_origins)[0]
        ex = _qapply(scene.rotunda.data.root_quat_w,
                     torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))[0]
        yaw = math.atan2(float(ex[1]), float(ex[0]))
        bp = (scene.basket.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(rp[0]), float(rp[1]), yaw, th_deg(scene.sauce),
                      th_deg(scene.decoy), float(bp[0]), float(bp[1])))
    arr = np.array(reads)
    print("[smoke] randomization readback (rot_x, rot_y, rot_yaw, th_sauce_deg, "
          f"th_decoy_deg, bkt_x, bkt_y):\n{arr.round(3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: rotunda xy + free yaw vary across seeded resets (readback: "
          f"dx={spread[0]:.3f} dy={spread[1]:.3f} dyaw={spread[2]:.2f} rad)",
          spread[0] > 0.02 and spread[1] > 0.02 and spread[2] > 0.5)
    sides = arr[:, 3] > 0
    check("randomization: the sauce can's hatch bearing varies "
          f"(dth={spread[3]:.1f} deg) and occurs on BOTH sides "
          f"({int(sides.sum())}/10 positive), decoy ALWAYS opposite",
          spread[3] > 20.0 and 0 < sides.sum() < len(sides)
          and bool((arr[:, 3] * arr[:, 4] < 0).all()))
    check("randomization: basket ring pose varies (readback: "
          f"dx={spread[5]:.3f} dy={spread[6]:.3f})",
          max(spread[5], spread[6]) > 0.15)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(300)
    s, ok = judge()
    report("null-policy", s, ok)
    check(f"null policy: score ~0 and no success after 300 idle steps (score={s:.3f}, "
          "cans still riding)",
          s <= 0.02 and not ok and bool(scene.on_platter(scene.sauce)[0]))

    # =========================== 6. the carousel mechanism is real ==========================
    # A z-torque rate servo (the crank-bar proxy) must SPIN the platter, and both cans
    # must RIDE it by friction alone. Drive toward the hatch but stop far outside the
    # alignment tolerance.
    yaw0, th0 = plat_yaw(), th_deg(scene.sauce)
    sgn = -math.copysign(1.0, th0)
    for _ in range(180):
        w = scene.platter.data.root_ang_vel_w[:, 2]
        apply_ztorque((1.5 * (sgn * 0.35 - w)).clamp(-0.40, 0.40))
        step(1)
    for _ in range(90):  # active brake: no coast into the alignment window
        w = scene.platter.data.root_ang_vel_w[:, 2]
        apply_ztorque((-3.0 * w).clamp(-0.40, 0.40))
        step(1)
        if abs(float(w[0])) < 0.03:
            break
    clear_forces()
    step(30)
    dyaw = math.degrees(abs(math.atan2(math.sin(plat_yaw() - yaw0),
                                       math.cos(plat_yaw() - yaw0))))
    dth = abs(th_deg(scene.sauce) - th0)
    s, ok = judge()
    report("mechanism-spin", s, ok)
    check("mechanism (the carousel is real): crank z-torque spins the platter "
          f"{dyaw:.1f} deg (>= 20) and the cans RIDE it by friction alone (sauce "
          f"bearing moved {dth:.1f} deg (>= 15), both cans still aboard), still "
          f"outside the alignment tolerance, no success",
          dyaw >= 20.0 and dth >= 15.0 and bool(scene.on_platter(scene.sauce)[0])
          and bool(scene.decoy_ok()[0]) and not bool(scene.aligned_now()[0]) and not ok)

    # =========================== 7. the rotunda cages the can ===============================
    # Kick the riding can hard, radially outward under the roofed sector: fence + roof
    # + the too-narrow moat must keep it inside the ring (this is why the spin-first
    # order is physics, not fiat).
    env.reset(seed=35)
    settle_all(300)
    rvec = (scene.sauce.data.root_pos_w[:, :2]
            - scene.rotunda.data.root_pos_w[:, :2])
    rdir = (rvec / rvec.norm(dim=-1, keepdim=True))[0]
    kick = [float(rdir[0]) * 1.2, float(rdir[1]) * 1.2, 0.4]
    teleport(scene.sauce, scene.sauce.data.root_pos_w,
             scene.sauce.data.root_quat_w, vel=kick, ang=[0.0, 0.0, 3.0],
             settle_steps=300)
    d = float((scene.sauce.data.root_pos_w[:, :2]
               - scene.rotunda.data.root_pos_w[:, :2]).norm(dim=-1)[0])
    s, ok = judge()
    report("rotunda-cage", s, ok)
    check("mechanism (the rotunda is real): the riding can kicked 1.3 m/s radially "
          "outward under the roofed sector cannot leave — fence, roof and the "
          f"too-narrow moat cage it (readback: r={d:.3f} < "
          f"{FENCE_IN + CAN_R:.3f}), not extracted, no success",
          d < FENCE_IN + CAN_R and not bool(scene.out_now()[0]) and not ok)

    # =========================== 8. seed strategy / anti-teleport ===========================
    # The seed's whole plan — reach in, pick the can where it stands, carry it to the
    # basket — is geometrically impossible here (probe 7). Constructed by fiat anyway:
    # the ordered latches (parked -> extracted) never fired, so the rubric gives ~0.
    env.reset(seed=41)
    settle_all(300)
    teleport(scene.sauce,
             basket_pose((0.0, 0.0, BKT_FLOOR_T + CAN_H / 2 + 0.004)), ident,
             settle_steps=150)
    s, ok = judge()
    report("seed-bypass", s, ok)
    check("seed strategy / anti-teleport: the RED can written INTO the basket by fiat "
          "(no spin, never parked in the hatch) is rejected by the ordered latches — "
          f"in_basket reads True yet score={s:.3f} <= 0.05 and no success",
          bool(scene.in_basket()[0]) and s <= 0.05 and not ok)

    # =========================== 9. wrong object ============================================
    env.reset(seed=45)
    settle_all(300)
    teleport(scene.decoy,
             basket_pose((0.0, 0.0, BKT_FLOOR_T + CAN_H / 2 + 0.004)), ident,
             settle_steps=150)
    s, ok = judge()
    report("wrong-object", s, ok)
    check("wrong object (the seed's move with the wrong item): the WHITE decoy in the "
          f"basket instead of the RED can is rejected — no success (score={s:.3f}), "
          "decoy clause violated on top",
          not ok and s <= 0.05 and not bool(scene.decoy_ok()[0]))

    # =========================== 10. acceptance (legal mini-solve) ==========================
    env.reset(seed=51)
    settle_all(300)

    def spin_to_park() -> bool:
        tau_cap, kw = 0.40, 1.5
        for _attempt in range(4):
            best, since, res = abs(th_deg(scene.sauce)), 0, ""
            for _i in range(4800):
                th = scene.hatch_angle(scene.sauce)
                speed = torch.where(th.abs() > math.radians(40.0),
                                    torch.full_like(th, 0.50),
                                    torch.full_like(th, 0.15))
                w = scene.platter.data.root_ang_vel_w[:, 2]
                apply_ztorque((kw * (-torch.sign(th) * speed - w))
                              .clamp(-tau_cap, tau_cap))
                step(1)
                a = abs(th_deg(scene.sauce))
                if a <= 10.0:
                    res = "cut"
                    break
                if a < best - 2.0:
                    best, since = a, 0
                else:
                    since += 1
                    if since > 240:
                        res = "stall"
                        break
            if res == "stall":
                tau_cap, kw = min(0.80, tau_cap * 1.5), min(8.0, kw * 2.0)
                continue
            for _b in range(240):
                w = scene.platter.data.root_ang_vel_w[:, 2]
                apply_ztorque((-3.0 * w).clamp(-0.40, 0.40))
                step(1)
                if abs(float(w[0])) < 0.03:
                    break
            clear_forces()
            step(90)
            if bool(scene.aligned_now()[0]):
                return True
        return False

    parked = spin_to_park()
    s_mid, ok_mid = judge()
    report("accept-parked", s_mid, ok_mid)
    teleport(scene.sauce,
             basket_pose((0.0, 0.0, BKT_FLOOR_T + CAN_H / 2 + 0.006)), ident,
             settle_steps=120)
    settle_all(300)
    s, ok = judge_accept()
    report("accept-construct", s, ok)
    check("acceptance: the legal path — crank-torque spin parks the RED can in the "
          f"hatch (judged mid-way: score={s_mid:.3f} ~ 0.40, NOT success), then the "
          "open-sky lift-out and basket drop -> success TRUE "
          f"(score={s:.3f})",
          parked and not ok_mid and 0.38 <= s_mid <= 0.45 and ok and s >= 0.99)
    st_accept = scene.get_state(all_ids)

    # =========================== 11-12. sauce removed flips success (+ cap) =================
    away = (scene.basket.data.root_pos_w[:, :2]
            - scene.rotunda.data.root_pos_w[:, :2])
    away = away / away.norm(dim=-1, keepdim=True)
    ground = scene.basket.data.root_pos_w.clone()
    ground[:, 0:2] += away * 0.22
    ground[:, 2] = scene.env_origins[:, 2] + CAN_H / 2 + 0.003
    teleport(scene.sauce, ground, ident, settle_steps=150)
    s, ok = judge()
    report("sauce-removed", s, ok)
    check("live + cap: the can stood on the open floor 22 cm past the basket flips "
          f"success FALSE and the score falls to the latched cap (score={s:.3f} in "
          "[0.55, 0.601])",
          not ok and 0.55 <= s <= 0.601 and not bool(scene.in_basket()[0]))
    scene.set_state(st_accept, all_ids)
    step(45)
    s, ok = judge_accept()
    report("sauce-restored", s, ok)
    check("set_state restore of the acceptance state -> success returns TRUE (11's "
          "rejection was the containment clause and nothing else)", ok and s >= 0.99)

    # =========================== 13-14. decoy clause flips success ==========================
    perp = torch.stack([-away[:, 1], away[:, 0]], dim=-1)
    dfloor = scene.rotunda.data.root_pos_w.clone()
    dfloor[:, 0:2] += perp * 0.55
    dfloor[:, 2] = scene.env_origins[:, 2] + CAN_H / 2 + 0.003
    teleport(scene.decoy, dfloor, ident, settle_steps=150)
    s, ok = judge()
    report("decoy-dumped", s, ok)
    check("decoy clause: the WHITE can pulled off the platter to the open floor flips "
          "success FALSE", not ok and not bool(scene.decoy_ok()[0]))
    scene.set_state(st_accept, all_ids)
    step(45)
    s, ok = judge_accept()
    report("decoy-restored", s, ok)
    check("set_state restore -> success returns TRUE (13's rejection was the decoy "
          "clause and nothing else)", ok and s >= 0.99)

    # =========================== 15. near miss: against the outer wall ======================
    wall = basket_pose((BKT_IN_HALF + BKT_WALL_T + CAN_R + 0.004, 0.0, 0.0))
    wall[:, 2] = scene.env_origins[:, 2] + CAN_H / 2 + 0.003
    teleport(scene.sauce, wall, ident, settle_steps=150)
    s, ok = judge()
    report("outer-wall", s, ok)
    check("near miss: the can standing on the ground flush against the basket's OUTER "
          f"wall is rejected by the containment gates (score={s:.3f}, capped), "
          "no success",
          not ok and not bool(scene.in_basket()[0]) and s <= 0.601)

    # =========================== 16. near miss: above the height band =======================
    hover = basket_pose((0.0, 0.0, 0.150))
    teleport(scene.sauce, hover, ident, settle_steps=1)  # judge in flight, xy inside
    loc = scene._local(scene.basket, scene.sauce.data.root_pos_w)[0]
    s, ok = judge()
    report("rim-hover", s, ok)
    check("near miss (z gate): the can over the basket mouth ABOVE the height band "
          f"(basket-frame z={float(loc[2]):.3f} > {scene.cfg.bkt_z_win[1]:.3f}, xy "
          "inside) is NOT contained and NOT success",
          float(loc[2]) > scene.cfg.bkt_z_win[1]
          and max(abs(float(loc[0])), abs(float(loc[1]))) <= scene.cfg.bkt_xy_max
          and not bool(scene.in_basket()[0]) and not ok)
    settle_all(300)  # it falls in — back to the genuine end state

    # =========================== 17. settle gate ============================================
    assert bool(scene.success()[0]), "acceptance state must reconstruct before the kick"
    teleport(scene.sauce, scene.sauce.data.root_pos_w,
             scene.sauce.data.root_quat_w, vel=[0.25, 0.0, 0.3], ang=[0.0, 0.0, 6.0],
             settle_steps=0)
    step(1)  # refresh buffers only — judge while still moving
    lv = float(scene.sauce.data.root_lin_vel_w[0].norm())
    av = float(scene.sauce.data.root_ang_vel_w[0].norm())
    s, ok = judge()
    report("settle-gate", s, ok)
    moving_rejected = (lv > scene.cfg.settle_lin or av > scene.cfg.settle_ang) and not ok
    settle_all(360)
    s2, ok2 = judge_accept()
    report("resettled", s2, ok2)
    check("settle gate: the placed can kicked (lin={:.2f} m/s, ang={:.1f} rad/s) and "
          "judged immediately is NOT success; resettled against the basket walls it "
          "IS again".format(lv, av), moving_rejected and ok2)

    # =========================== 18-19. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point except the "
          "acceptance construct and the flip-backs", not ever_bad_success[0])
    check("final: all task-object states finite (no NaN)", all_finite())

    # =========================== 20. video + verdict ========================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.sauce_carousel")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    check(f"video: > 10 frames captured ({len(frames)})", len(frames) > 10)

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

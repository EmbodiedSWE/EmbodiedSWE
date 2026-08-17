"""Teleport solution for ChairFoldawayScene (sim_gen task `stack_chairs_i326`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY. Both load-bearing interactions go through contact
dynamics, one chair at a time (red -> green -> blue; slots are identical, chair i takes
slot i):
  1. FOLD (dynamics, in place at the scatter pose): a hinge-axis torque servo on the
     SEAT body (gravity feed-forward + rate PD, clamped far under what could tip the
     0.5 kg frame) swings the seat up through its gravity balance (~95 deg) until it
     rests folded on the -100 deg joint stop. The torque frame/sign convention is
     POD-DEPENDENT, so it is PROBED at runtime on the first chair (4 combos; a wrong
     combo just presses the seat into a hard stop and moves nothing). The torque is
     dropped and the fold is verified by relative-pose readback — a gravity-stable
     mechanical state, not a held one.
  2. HOVER (transport): the folded chair — BOTH bodies, exact relative pose preserved,
     velocities zeroed — is teleported to a hover pose in FREE AIR above its slot
     (foot 5 mm above the fin tops, chair x-axis across the slot). Nothing is
     teleported into contact.
  3. INSERT (dynamics): a vertical velocity-servo force on the frame lowers the chair
     down through the slot's top opening (~0.2 m/s), with a gentle xy PD onto the slot
     center and a yaw PD keeping the 95 mm footprint across the 115 mm gap; the wrench
     is DROPPED 12 mm above the rack floor and the chair lands, settles, and rests
     where contact left it.
After all three slots read covered, hands off: `SIM_GEN_SCORE` is printed at every
phase boundary (non-decreasing — fold credit is latched, racked credit is physical and
stays), success() must hold through a >= 3.3 simulated-second persistence window with
no intervention, and only then `SIM_GEN_SOLVE: SUCCESS` is printed.

Run (forge): python -u -m simgen_tasks.stack_chairs_i326.solve --headless [--seed N]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import math
import os
import threading

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

CHAIRS = scene_mod.CHAIRS

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_inv, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.chair_foldaway")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    dt = 1.0 / 120.0

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    # Force-frame mode is POD-DEPENDENT: mode 0 = encode the desired WORLD wrench into
    # the body's current frame (quat_apply_inverse), mode 1 = pass the world vector
    # raw. Calibrated once, from the fold-torque probe on the first chair.
    frame_mode = [0]
    fold_sign = [-1.0]  # fold torque along the hinge axis (body +y); -1 folds by design

    zero3 = torch.zeros(n, 3, device=device)

    def wrench(body, f_w: torch.Tensor, t_w: torch.Tensor | None = None) -> None:
        """Apply a WORLD force/torque (n,3). Re-set every step (quat drifts)."""
        if t_w is None:
            t_w = zero3
        if frame_mode[0] == 0:
            q = body.data.root_link_quat_w
            f, t = quat_apply_inverse(q, f_w), quat_apply_inverse(q, t_w)
        else:
            f, t = f_w, t_w
        body.set_external_force_and_torque(f.unsqueeze(1), t.unsqueeze(1),
                                           env_ids=all_ids)

    def seat_torque(i: int, tau: float | torch.Tensor) -> None:
        """Torque `tau` (n,) or scalar about the hinge axis (seat body +y), applied to
        the seat body. Positive tau presses toward DEPLOYED; the fold direction is
        `fold_sign` (probed)."""
        if not torch.is_tensor(tau):
            tau = torch.full((n,), float(tau), device=device)
        t_b = torch.zeros(n, 3, device=device)
        t_b[:, 1] = tau
        t_w = quat_apply(scene.seats[i].data.root_link_quat_w, t_b)
        wrench(scene.seats[i], zero3, t_w)

    def clear(body) -> None:
        body.set_external_force_and_torque(zero3.unsqueeze(1), zero3.unsqueeze(1),
                                           env_ids=all_ids)

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        cov = scene.covered()[0]
        fd = scene.fold_deg()[0]
        up = scene.chair_up_z()[0]
        parts = " ".join(
            f"{nm}:fold={float(fd[i]):+.0f}deg up={float(up[i]):+.2f}"
            for i, (nm, _) in enumerate(CHAIRS))
        print(f"[solve] {tag:14s} | {parts} | covered={[bool(v) for v in cov]} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    ryaw = scene.rack_yaw  # (n,)
    cy, sy = torch.cos(ryaw), torch.sin(ryaw)
    ex_r = torch.stack([cy, sy], dim=1)  # rack +x (along the slots), world
    ey_r = torch.stack([-sy, cy], dim=1)  # rack +y (across the slots), world

    def slot_world_xy(s: int) -> torch.Tensor:
        """(n,2) world xy of slot s's center."""
        return (scene.rack_xy + ey_r * float(c.slot_y[s])
                + scene.env_origins[:, 0:2])

    W = (c.frame_mass + c.seat_mass) * 9.81  # full chair weight, N
    mgd = c.seat_mass * 9.81 * math.hypot(c.seat_com[0], c.seat_com[2])
    com_tilt = math.degrees(math.atan2(-c.seat_com[2], c.seat_com[0]))

    def grav_ff(theta_deg: float) -> float:
        """Gravity torque (N*m) pressing the seat toward DEPLOYED at fold angle theta."""
        e = math.radians(theta_deg - com_tilt)
        return mgd * max(math.cos(e), 0.0)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    print(f"[solve] layout (seed {args.seed}): rack_xy="
          f"({float(scene.rack_xy[0, 0]):+.3f},{float(scene.rack_xy[0, 1]):+.3f}) "
          f"yaw={math.degrees(float(ryaw[0])):+.1f}deg "
          f"slot_y={[f'{v * 1000:+.0f}' for v in c.slot_y]}mm", flush=True)
    for i, (nm, _) in enumerate(CHAIRS):
        p = rel(scene.frames[i])
        print(f"[solve]   chair {nm}: ({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):.3f}) fold={float(scene.fold_deg()[0, i]):+.1f}deg",
              flush=True)
    report("reset")
    last = print_score("P0 reset+settle")
    assert last <= 0.02, "null credit at reset — rubric leak"

    # ---------------- per chair: FOLD -> HOVER (transport) -> INSERT -----------------------
    probe_done = [False]

    def fold_chair(i: int, nm: str) -> None:
        """Torque-servo the seat from deployed (~0 deg) onto the folded stop, then
        drop the torque and verify the gravity-stable fold by readback."""
        seat = scene.seats[i]
        for attempt in range(3):
            # --- torque frame/sign probe (first chair, first attempt only) ---
            if not probe_done[0]:
                combos = [(0, -1.0), (1, -1.0), (0, 1.0), (1, 1.0)]
                found = False
                for fm, sg in combos:
                    frame_mode[0], fold_sign[0] = fm, sg
                    th0 = float(scene.fold_deg()[0, i])
                    for _ in range(25):
                        seat_torque(i, sg * 0.07)
                        env.step(no_action)
                    dth = float(scene.fold_deg()[0, i]) - th0
                    print(f"[solve] torque probe mode={fm} sign={sg:+.0f}: "
                          f"dtheta={dth:+.1f}deg", flush=True)
                    if dth > 3.0:
                        found = True
                        break
                    clear(seat)
                    step(30)  # let a wrong-combo press relax
                assert found, "no torque frame/sign combo moved the seat"
                probe_done[0] = True
            sg = fold_sign[0]
            # --- rate-servo swing to the stop ---
            th_prev = float(scene.fold_deg()[0, i])
            for k in range(600):
                th = float(scene.fold_deg()[0, i])
                w_meas = (th - th_prev) / dt * math.pi / 180.0  # FD hinge rate, rad/s
                th_prev = th
                if th >= 96.0:
                    break
                tau = grav_ff(th) + 0.015 * (2.0 - w_meas)
                tau = min(max(tau, -0.03), 0.09)
                seat_torque(i, sg * tau)
                env.step(no_action)
                if k % 60 == 0:
                    print(f"[solve] {nm} fold k={k:3d} th={th:+.1f}deg "
                          f"w={w_meas:+.2f}rad/s tau={tau:.3f}", flush=True)
            # --- press onto the stop briefly, then HANDS OFF ---
            for _ in range(30):
                seat_torque(i, sg * 0.02)
                env.step(no_action)
            clear(seat)
            step(50)
            th = float(scene.fold_deg()[0, i])
            print(f"[solve] {nm} fold attempt {attempt}: rest theta={th:+.1f}deg "
                  f"(hands off)", flush=True)
            if th >= c.fold_min_deg + 5.0:
                return
        raise AssertionError(f"{nm} seat did not stay folded after 3 attempts")

    def teleport_chair(i: int, xy_w: torch.Tensor, z_loc: float, yaw: torch.Tensor) -> None:
        """Transport ONLY: write BOTH bodies of chair i to a new frame pose (world xy,
        env-local z, world yaw), exact CURRENT relative fold pose preserved,
        velocities zeroed."""
        frame, seat = scene.frames[i], scene.seats[i]
        fq = frame.data.root_quat_w.clone()
        fp = frame.data.root_pos_w.clone()
        sq = seat.data.root_quat_w.clone()
        sp = seat.data.root_pos_w.clone()
        q_rel = quat_mul(quat_inv(fq), sq)
        p_rel = quat_apply_inverse(fq, sp - fp)
        half = yaw / 2
        tq = torch.stack([torch.cos(half), torch.zeros(n, device=device),
                          torch.zeros(n, device=device), torch.sin(half)], dim=1)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = xy_w
        st[:, 2] = scene.env_origins[:, 2] + z_loc
        st[:, 3:7] = tq
        frame.write_root_state_to_sim(st, all_ids)
        ss = torch.zeros(n, 13, device=device)
        ss[:, 0:3] = st[:, 0:3] + quat_apply(tq, p_rel)
        ss[:, 3:7] = quat_mul(tq, q_rel)
        seat.write_root_state_to_sim(ss, all_ids)

    ez_w = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
    ex_b = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)

    def insert_chair(i: int, s: int, nm: str) -> bool:
        """Dynamics: velocity-servo the hovering chair down through the slot opening;
        drop the wrench 12 mm above the rack floor; settle; report covered. The hover
        hold is an inverted pendulum (lift force at the frame CoM, combined CoM ~14 mm
        higher), so an UPRIGHTING attitude PD is essential; descent is gated on xy
        alignment while the foot is still above the fin tops."""
        frame = scene.frames[i]
        tgt_xy = slot_world_xy(s)
        yaw_tgt = ryaw + math.pi / 2
        release_z = c.rack_floor_z + 0.012
        entry_z = c.fin_top + c.origin_h + 0.002  # above this the foot is out of the slot
        z_best, k_best = 10.0, 0
        for k in range(700):
            p = frame.data.root_pos_w
            z_loc = float((p[:, 2] - scene.env_origins[:, 2])[0])
            if z_loc <= release_z:
                break
            if z_loc < z_best - 0.001:
                z_best, k_best = z_loc, k
            if k - k_best > 90 and z_loc > release_z + 0.005:
                print(f"[solve] {nm} insert: wedged at z={z_loc:.3f} — abort attempt",
                      flush=True)
                clear(frame)
                return False
            v = frame.data.root_lin_vel_w
            w = frame.data.root_ang_vel_w
            err_xy = tgt_xy - p[:, 0:2]
            aligned = float(err_xy[0].norm()) < 0.010
            # hold altitude until aligned (only matters above the slot mouth)
            v_des = -0.22 if (aligned or z_loc < entry_z) else 0.0
            fz = W * 1.0 + 6.0 * (v_des - v[:, 2])
            fxy = (12.0 * err_xy - 5.0 * v[:, 0:2]).clamp(-1.5, 1.5)
            f = torch.zeros(n, 3, device=device)
            f[:, 0:2] = fxy
            f[:, 2] = fz.clamp(0.0, 2.0 * W)
            # attitude: uprighting PD (z_b -> world up) + yaw PD keeping the 95 mm
            # footprint across the gap
            q = frame.data.root_quat_w
            z_b = quat_apply(q, ez_w)
            t_up = 0.40 * torch.cross(z_b, ez_w, dim=1) - 0.020 * w
            fx_w = quat_apply(q, ex_b)
            yaw_now = torch.atan2(fx_w[:, 1], fx_w[:, 0])
            yaw_err = torch.atan2(torch.sin(yaw_tgt - yaw_now),
                                  torch.cos(yaw_tgt - yaw_now))
            t = t_up.clamp(-0.15, 0.15)
            t[:, 2] = (0.060 * yaw_err - 0.012 * w[:, 2]).clamp(-0.06, 0.06)
            wrench(frame, f, t)
            env.step(no_action)
            if k % 80 == 0:
                print(f"[solve] {nm} insert k={k:3d} z={z_loc:.3f} "
                      f"xy_err={float(err_xy[0].norm()) * 1000:.0f}mm "
                      f"vz={float(v[0, 2]):+.2f} up={float(z_b[0, 2]):+.2f} "
                      f"fold={float(scene.fold_deg()[0, i]):+.0f}", flush=True)
        clear(frame)
        step(80)  # free fall the last 12 mm, land, settle
        report(f"inserted-{nm}")
        return bool(scene.covered()[0, s])

    hover_z = c.fin_top + 0.005 + c.origin_h  # foot bottom 5 mm above the fin tops
    for i, (nm, _) in enumerate(CHAIRS):
        s = i  # slots are identical — chair i takes slot i
        # --- FOLD (dynamics, in place) ---
        fold_chair(i, nm)
        report(f"folded-{nm}")
        sc = print_score(f"P{2 * i + 1} {nm} seat folded onto its stop (contact dynamics)")
        assert sc >= last - 1e-6, "score decreased at fold"
        last = sc
        # --- HOVER (transport only, free air asserted by construction: foot bottom
        #     5 mm above the fin tops, xy over an empty slot) -> INSERT (dynamics).
        #     On a failed insert the chair may have jolted open: restage it on the
        #     open floor BEHIND the rack, settle, RE-FOLD if needed, then re-hover.
        ok = False
        for attempt in range(3):
            assert float(scene.fold_deg()[0, i]) >= c.fold_min_deg, \
                "must be folded to hover"
            teleport_chair(i, slot_world_xy(s), hover_z, ryaw + math.pi / 2)
            ok = insert_chair(i, s, nm)
            if ok:
                break
            print(f"[solve] {nm}: slot {s} not covered (attempt {attempt}) — "
                  f"restage behind the rack and refold", flush=True)
            stage_xy = (scene.rack_xy + ex_r * 0.30
                        + scene.env_origins[:, 0:2])
            teleport_chair(i, stage_xy, c.origin_h + 0.003, ryaw + math.pi)
            step(60)
            if float(scene.fold_deg()[0, i]) < c.fold_min_deg + 5.0:
                fold_chair(i, nm)
        assert ok, f"{nm} chair's slot did not read covered"
        sc = print_score(f"P{2 * i + 2} {nm} chair lowered into slot {s} (contact dynamics)")
        assert sc >= last - 1e-6, "score decreased across insert"
        last = sc

    # ---------------- final: hands off, success + persistence ------------------------------
    step(60)
    report("all-racked")
    sc = print_score("P7 all three slots filled, hands off")
    assert sc >= last - 1e-6
    last = sc
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after racking all chairs)", flush=True)
        os._exit(1)

    hold, flickers = True, 0
    for k in range(400):  # 400 substeps = 3.33 s at 120 Hz, no intervention
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:
                print(f"[solve] persist flicker @step {k}: covered="
                      f"{[bool(v) for v in scene.covered()[0]]}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    sc = print_score("P8 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and sc >= last - 1e-6
    if ok:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)

    code = 0 if ok else 1
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
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)

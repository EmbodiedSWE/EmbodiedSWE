"""Teleport solution for ChairTuckScene (sim_gen task `stack_chairs_i110`) — the task's
legitimacy certificate.

Teleportation handles TRANSPORT ONLY. The load-bearing interaction — sliding each chair
horizontally through its bay's front opening, under the desk top, until the backrest
docks against the desk's front edge — goes through contact dynamics, one chair at a
time (wide -> mid -> narrow; the matching is forced by the widths, the order is free):
  1. STAGE (transport): the chair is teleported from its scatter pose to a staging pose
     on the floor CENTERED IN FRONT of its matched bay, aligned with the desk heading,
     entirely OUTSIDE the desk footprint (asserted: seat center still in front of the
     front edge; clear of the other chairs). Nothing is in contact with the desk.
  2. TUCK (dynamics): a velocity-regulated horizontal force (static-friction
     feed-forward + PD to +0.07 m/s along the desk's tuck direction, force at the CoM
     only — no orientation pinning) slides the chair across the floor in through the
     opening; a lateral PD keeps it on the bay's center line while the partition walls
     guide it. The seat passes under the slab; the push ends when the backrest DOCKS
     against the slab's front edge (depth stall) — the wrench is DROPPED and the chair
     rests where contact left it.
  3. Repeat for the other two chairs; nothing is ever teleported into the bay, past
     the opening, never welded, never held at the end.
After all three bays read covered, hands off: `SIM_GEN_SCORE` is printed at every phase
boundary (non-decreasing — tuck credit is latched, covered credit is physical and
stays), success() must hold through a >= 3.3 simulated-second persistence window with
no intervention, and only then `SIM_GEN_SOLVE: SUCCESS` is printed.

Run (forge): python -u -m simgen_tasks.stack_chairs_i110.solve --headless [--seed N]
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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.chair_tuck")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    # Force-frame mode is POD-DEPENDENT (see close_box_i26 / change_channel_i60):
    # mode 0 = encode the world force in the body's current frame (quat_apply_inverse),
    # mode 1 = pass the world vector raw. Calibrated once from measured displacement.
    frame_mode = [0]

    def wrench(body, f_w: torch.Tensor, t_w: torch.Tensor | None = None) -> None:
        """Apply a WORLD force (and optional WORLD z-torque) (n,3). Re-set every step."""
        from isaaclab.utils.math import quat_apply_inverse

        if t_w is None:
            t_w = torch.zeros(n, 3, device=device)
        if frame_mode[0] == 0:
            q = body.data.root_link_quat_w
            f, t = quat_apply_inverse(q, f_w), quat_apply_inverse(q, t_w)
        else:
            f, t = f_w, t_w
        body.set_external_force_and_torque(f.unsqueeze(1), t.unsqueeze(1),
                                           env_ids=all_ids)

    def front_xy(body) -> torch.Tensor:
        """(n,2) unit horizontal front axis of `body` (world)."""
        from isaaclab.utils.math import quat_apply

        ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
        f = quat_apply(body.data.root_quat_w, ex)[:, 0:2]
        return f / f.norm(dim=1, keepdim=True).clamp(min=1e-9)

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        cov = scene.covered()[0]
        dep = scene.chair_depth()[0]
        up = scene.chair_up_z()[0]
        parts = " ".join(
            f"{nm}:dep={float(dep[i]) * 1000:+.0f}mm up={float(up[i]):+.2f}"
            for i, (nm, _) in enumerate(CHAIRS))
        print(f"[solve] {tag:14s} | {parts} | covered={[bool(v) for v in cov]} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    dyaw = scene.desk_yaw  # (n,)
    cy, sy = torch.cos(dyaw), torch.sin(dyaw)
    fwd = torch.stack([cy, sy], dim=1)  # (n,2) tuck direction, world
    lat = torch.stack([-sy, cy], dim=1)  # (n,2) desk +y_t, world

    def desk_to_world(x_t: float, y_t: torch.Tensor) -> torch.Tensor:
        """(n,2) world xy of a desk-frame point (x_t scalar, y_t (n,))."""
        return (scene.desk_xy + fwd * x_t + lat * y_t.unsqueeze(1)
                + scene.env_origins[:, 0:2])

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    bw0 = [float(v) for v in scene.bay_w[0]]
    by0 = [float(v) for v in scene.bay_y[0]]
    print(f"[solve] layout (seed {args.seed}): desk_xy="
          f"({float(scene.desk_xy[0, 0]):+.3f},{float(scene.desk_xy[0, 1]):+.3f}) "
          f"yaw={math.degrees(float(dyaw[0])):+.1f}deg "
          f"bays L->R widths={[f'{v * 1000:.0f}' for v in bw0]}mm "
          f"centers_y={[f'{v * 1000:+.0f}' for v in by0]}mm", flush=True)
    for i, (nm, _) in enumerate(CHAIRS):
        p = rel(scene.chairs[i])
        print(f"[solve]   chair {nm} (w={c.chair_widths[i] * 1000:.0f}mm): "
              f"({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):.3f})", flush=True)
    report("reset")
    last = print_score("P0 reset+settle")
    assert last <= 0.02, "null credit at reset — rubric leak"

    # ---------------- per chair: STAGE (transport) then TUCK (dynamics) --------------------
    for i, (nm, _) in enumerate(CHAIRS):
        chair = scene.chairs[i]
        # matched bay: the bay whose width equals this chair's matched width
        b = int((scene.bay_w[0] - c.bay_widths[i]).abs().argmin())
        y_bay = scene.bay_y[:, b]  # (n,) desk frame

        # --- STAGE: on the floor, centered in front of bay b, aligned, clear of all ---
        others = [scene.chairs[j] for j in range(3) if j != i]
        stage_x = -c.desk_depth / 2 - 0.080 - c.seat_depth / 2  # seat front 80 mm out
        for cand in (stage_x, stage_x - 0.045, stage_x - 0.090):
            xy = desk_to_world(cand, y_bay)
            d_min = min(float((xy[0] - o.data.root_pos_w[0, 0:2]).norm()) for o in others)
            if d_min > 0.115:
                stage_x = cand
                break
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = desk_to_world(stage_x, y_bay)
        st[:, 2] = scene.env_origins[:, 2] + c.origin_h + 0.004
        st[:, 3] = torch.cos(dyaw / 2)
        st[:, 6] = torch.sin(dyaw / 2)
        chair.write_root_state_to_sim(st, all_ids)
        step(15)  # drop 4 mm onto the floor, free
        report(f"staged-{nm}")
        assert float(scene.chair_depth()[0, i]) < -0.02, \
            "staging must leave the seat center clearly IN FRONT of the desk edge"
        d_min = min(float((chair.data.root_pos_w[0, 0:2]
                           - o.data.root_pos_w[0, 0:2]).norm()) for o in others)
        assert d_min > 0.10, f"staged {nm} chair too close to another chair ({d_min:.3f})"
        s = print_score(f"P{2 * i + 1} {nm} chair staged before its bay (transport)")
        assert s >= last - 1e-6, "score decreased at staging"
        last = s

        # --- TUCK: velocity-regulated slide in through the opening until the dock.
        # A CoM push on a sliding body is YAW-UNSTABLE (the cocked green chair wedged
        # its 106 mm diagonal in the 90 mm mouth on the first attempt): steer by
        # emulating a PULL from a point ahead of the seat (tau_z = lever * f_lat) plus
        # an explicit yaw PD; on a stall, actively RETREAT and re-align, then re-enter.
        docked = False
        probe_p0 = chair.data.root_pos_w[0, 0:2].clone()  # frame probe anchor (chair 0)
        probe_done = i > 0
        ff = 0.65  # static-friction feed-forward (raised slightly after a backoff)
        backoffs = 0
        dep_best, k_best = -1.0, 0
        retreat_until = -1
        for k in range(1300):
            dep = float(scene.chair_depth()[0, i])
            v = chair.data.root_lin_vel_w[:, 0:2]
            v_along = (v * fwd).sum(dim=1)
            retreating = k < retreat_until
            if not retreating and (dep > 0.034 or (dep > 0.026
                                   and abs(float(v_along[0])) < 0.004 and k > 80)):
                docked = True
                break
            if dep > dep_best + 0.001:
                dep_best, k_best = dep, k
            # frame probe (first pushed chair only): after ~4 mm of measured travel,
            # compare displacement direction with the commanded tuck direction; if
            # badly off, the pod applies wrenches in the other frame convention — flip.
            if not probe_done:
                disp = chair.data.root_pos_w[0, 0:2] - probe_p0
                d = float(disp.norm())
                if d > 0.004:
                    cosang = float((disp / d * fwd[0]).sum())
                    if cosang < 0.5:  # > 60 deg off the commanded direction
                        frame_mode[0] = 1 - frame_mode[0]
                        print(f"[solve] frame probe: displacement {cosang:+.2f} off "
                              f"commanded -> flipping to mode {frame_mode[0]}", flush=True)
                    else:
                        print(f"[solve] frame probe: mode {frame_mode[0]} confirmed "
                              f"(cos={cosang:+.2f})", flush=True)
                    probe_done = True
            # stall backoff: no depth progress for 130 steps short of the mouth ->
            # RETREAT (reverse push, yaw PD re-aligns while the legs are unlocked).
            if not retreating and k - k_best > 130 and dep < 0.020 and backoffs < 5:
                backoffs += 1
                ff = min(ff + 0.10, 1.0)
                retreat_until = k + 90
                k_best = k + 90
                print(f"[solve] {nm}: stall at dep={dep * 1000:+.0f}mm "
                      f"face={float(scene.chair_facing()[0, i]):+.2f} -> retreat "
                      f"#{backoffs}, ff -> {ff:.2f} N", flush=True)
            if retreating and dep < -0.075:
                retreat_until = k  # far enough out — resume the approach
            if k % 80 == 0:
                _x_t0, y_t0 = scene.chair_desk_xy()
                print(f"[solve] {nm} tuck k={k:4d} dep={dep * 1000:+.0f}mm "
                      f"v={float(v_along[0]):+.3f} up={float(scene.chair_up_z()[0, i]):+.2f} "
                      f"face={float(scene.chair_facing()[0, i]):+.2f} "
                      f"y_err={float(y_bay[0] - y_t0[0, i]) * 1000:+.0f}mm", flush=True)
            _x_t, y_t = scene.chair_desk_xy()
            v_lat = (v * lat).sum(dim=1)
            v_des = -0.06 if retreating else 0.055
            f_along = (math.copysign(ff, v_des)
                       + 4.0 * (v_des - v_along)).clamp(-1.5, 1.5)
            f_lat = (3.0 * (y_bay - y_t[:, i]) - 1.5 * v_lat).clamp(-0.5, 0.5)
            # steering: signed yaw error chair-front -> tuck direction, world z
            fh = front_xy(chair)
            yaw_err = torch.atan2(fh[:, 0] * fwd[:, 1] - fh[:, 1] * fwd[:, 0],
                                  (fh * fwd).sum(dim=1))
            wz = chair.data.root_ang_vel_w[:, 2]
            lever = 0.06 if not retreating else -0.06  # pull-point ahead of the motion
            t_z = (lever * f_lat + 0.010 * yaw_err - 0.0025 * wz).clamp(-0.02, 0.02)
            f = torch.zeros(n, 3, device=device)
            f[:, 0:2] = fwd * f_along.unsqueeze(1) + lat * f_lat.unsqueeze(1)
            t = torch.zeros(n, 3, device=device)
            t[:, 2] = t_z
            wrench(chair, f, t)
            env.step(no_action)
        wrench(chair, torch.zeros(n, 3, device=device))
        assert docked, f"{nm} chair never docked (depth stalled short of the gate)"
        step(60)  # hands off: the chair rests where the dock left it
        report(f"tucked-{nm}")
        assert bool(scene.covered()[0, b]), f"{nm} chair's bay did not read covered"
        s = print_score(f"P{2 * i + 2} {nm} chair slid in and docked (contact dynamics)")
        assert s >= last - 1e-6, "score decreased across tucking"
        last = s

    # ---------------- final: hands off, success + persistence ------------------------------
    step(60)
    report("all-tucked")
    s = print_score("P7 all three bays filled, hands off")
    assert s >= last - 1e-6
    last = s
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after tucking all chairs)", flush=True)
        os._exit(1)

    hold, flickers = True, 0
    for k in range(400):  # 400 substeps = 3.33 s at 120 Hz, no intervention
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:
                cov = scene.covered()[0]
                print(f"[solve] persist flicker @step {k}: covered="
                      f"{[bool(v) for v in cov]}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s = print_score("P8 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s >= last - 1e-6
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

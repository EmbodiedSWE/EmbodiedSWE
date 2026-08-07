"""Teleport solution for ClipFenceScene (sim_gen task `native_liberoplus_i66`) — the
task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, one clip at a time): a single root-state write carries the
   spring-shut clip from the ground to a HOVER pose directly above its color's zone
   on the fence top — upright, hinge along the fence, jaw tips ~4 mm CLEAR of the
   blade top edge, zero velocity. The write never creates contact; a shut clip
   hovering there satisfies no astride clause (its spring pins psi at -10 deg and
   the 8 mm blade cannot be between jaws pinched shut).
2. SQUEEZE (applied torques — the actuation that IS the task): opposed torques
   about the hinge axis on the two lever bodies (exactly the couple a parallel-jaw
   gripper's closing stroke exerts on the flared tails) cam the jaws open against
   the torsion-spring preload, up to the +30 deg stop. A weak PD force servo at the
   (coincident) lever CoMs holds the hover against gravity meanwhile; being at the
   pivot, it exerts no torque about the hinge. Torque direction is verified by a
   runtime probe (does psi actually open?) and re-encoded with R_ref @ R_now^T if
   the backend's external-wrench frame drags with body rotation.
3. INSERTION (servo descent, jaws held open): the servo target descends ~55 mm so
   the open jaws pass astride the blade top; the outward-tilted inner faces form a
   self-centering funnel. Descent ends when the pivot-boss undersides SEAT on the
   blade top edge by contact.
4. CLAMP (spring + contact — the goal condition itself): the squeeze torque is
   ramped OFF; the preloaded spring closes the jaws flush onto the two blade faces
   (~2.6 N per face). All wrenches are cleared and the clip hangs by its own grip.
5. Repeat for the second clip. No order is required by the rubric.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still holds.

Run (forge): python -u -m simgen_tasks.native_liberoplus_i66.solve --headless [--seed N]
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
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod

Z_TOP = scene_mod.Z_TOP
PSI_CLOSED = scene_mod.PSI_CLOSED
PSI_OPEN_MAX = scene_mod.PSI_OPEN_MAX
LEVER_MASS = scene_mod.LEVER_MASS
clip_root_states = scene_mod.clip_root_states

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

HOVER_DZ = 0.064          # hover pivot height above the blade top (jaw tips ~4 mm clear)
SEAT_DZ = 0.001           # descent servo target: just below the boss-seat height
PSI_OPEN_TGT = 9.0        # jaws count as "held open" past this (deg): the backend
#                           enforces an effective +10 stop regardless of the authored
#                           +30 upper limit (measured: psi pins at +10.2 under 0.6 N*m)
TAU_HOLD = 0.30           # holding torque (N*m): pins the jaws against that stop
KP, KD = 400.0, 40.0      # hover/descent force servo gains (per-lever, at the CoM)
F_MAX = 1.2               # per-lever servo force clamp (N)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.clip_fence")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)
    g = 9.81

    from isaaclab.utils.math import matrix_from_quat, quat_apply

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def levers(k: int):
        return scene.levers[scene.CLIPS[k]]

    def psi_of(k: int) -> float:
        return float(scene.psi_deg()[0, k])

    def pivot_of(k: int) -> torch.Tensor:
        a, b = levers(k)
        return 0.5 * (a.data.root_pos_w + b.data.root_pos_w)

    def report(tag: str) -> None:
        psi = scene.psi_deg()[0]
        astr = scene.astride()[0]
        piv = [scene._fence_local(pivot_of(k))[0] for k in range(2)]
        locs = " ".join(
            f"{scene.CLIPS[k]}=({float(piv[k][0]):+.3f},{float(piv[k][1]):+.3f},"
            f"{float(piv[k][2]):+.3f})p{float(psi[k]):+.1f}" for k in range(2))
        _p, _q, vel, _w = scene._lever_tensors()
        print(f"[solve] {tag:14s} | {locs} astride={astr.tolist()} "
              f"near={scene._near[0].tolist()} act={scene._act[0].tolist()} "
              f"vmax={float(vel[0].max()):.3f} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_wrenches(k: int) -> None:
        """Clear the wrenches of clip k's levers ONLY. Never touch a body that has
        not been wrenched yet: the backend's wrench composer captures its frame
        reference at the FIRST set call per body, so a zero-write to a clip still
        lying on the ground would freeze its reference in the lying pose and every
        later world-frame wrench at the upright hover would arrive ~90 deg
        misrotated (measured: blue's mode-0 torque had dpsi = +0.0 exactly)."""
        for body in levers(k):
            body.set_external_force_and_torque(zero_wrench, zero_wrench,
                                               env_ids=all_ids)

    # --- external-wrench frame handling (pod-dependent drag with body rotation) --------------
    # mode 0: pass world-frame wrenches straight through. mode 1: pre-encode with
    # M = R_ref . R_now^T so the backend's rotation-since-reset drag cancels.
    R_ref: dict[str, torch.Tensor] = {}
    mode = {0: 0, 1: 0}

    def enc(body, name: str, k: int, vec_w: torch.Tensor) -> torch.Tensor:
        if mode[k] == 0:
            return vec_w
        m_now = matrix_from_quat(body.data.root_quat_w)
        m_enc = R_ref[name] @ m_now.transpose(-1, -2)
        return (m_enc @ vec_w.unsqueeze(-1)).squeeze(-1)

    def hinge_axis_w(k: int) -> torch.Tensor:
        """The clip's PHYSICAL hinge axis in world (mean of the two levers' local
        +y): opposed torques stay a pure hinge couple even when the clip tilts."""
        a, b = levers(k)
        e = torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3)
        y = quat_apply(a.data.root_quat_w, e) + quat_apply(b.data.root_quat_w, e)
        return y / y.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    def drive(k: int, tau: float, p_des: torch.Tensor, steps: int) -> None:
        """`steps` sim steps of: opposed hinge torques +-tau on the two levers plus a
        gravity-compensating PD force servo toward `p_des` at each lever CoM (= the
        pivot, so the servo adds no torque about the hinge)."""
        a, b = levers(k)
        names = (f"{scene.CLIPS[k]}_a", f"{scene.CLIPS[k]}_b")
        for _ in range(steps):
            y = hinge_axis_w(k)
            for body, name, sgn in ((a, names[0], -1.0), (b, names[1], 1.0)):
                acc = (KP * (p_des - body.data.root_pos_w)
                       - KD * body.data.root_lin_vel_w)
                f = LEVER_MASS * acc
                f[:, 2] += LEVER_MASS * g
                f = f.clamp(min=-F_MAX, max=F_MAX)
                t = sgn * tau * y
                body.set_external_force_and_torque(
                    enc(body, name, k, f).view(n, 1, 3),
                    enc(body, name, k, t).view(n, 1, 3),
                    env_ids=all_ids, is_global=True)
            env.step(no_action)

    def rehover(k: int, p_hover: torch.Tensor) -> None:
        """Recover the hover pose (transport only: shut clip, free space)."""
        a, b = levers(k)
        st_a, st_b = clip_root_states(p_hover, scene.fence.data.root_quat_w,
                                      PSI_CLOSED)
        a.write_root_state_to_sim(st_a, all_ids)
        b.write_root_state_to_sim(st_b, all_ids)
        drive(k, 0.0, p_hover, 12)

    def open_jaws(k: int, p_hover: torch.Tensor) -> None:
        """Squeeze the jaws open past PSI_OPEN_TGT with a MONOTONE torque ramp.
        Frame probe: a correctly-framed torque above the spring preload (0.075 N*m
        at the closed stop) must visibly open psi; if the first burst moves psi by
        < 2 deg, the wrench frame is wrong -> toggle the encoding mode and
        re-teleport the hover to undo whatever the misframed burst did. Once psi
        responds, only the magnitude ramps (an equilibrium stall against the
        spring means MORE torque, not a different frame)."""
        name = scene.CLIPS[k]
        for _probe in range(3):
            psi0 = psi_of(k)
            drive(k, 0.15, p_hover, 12)
            dpsi = psi_of(k) - psi0
            print(f"[solve] {name} frame probe: mode={mode[k]} tau=0.15 "
                  f"dpsi={dpsi:+.1f} deg", flush=True)
            if dpsi >= 2.0:
                break
            mode[k] ^= 1
            print(f"[solve] {name} psi did not respond — toggling wrench encoding "
                  f"to mode {mode[k]} and re-hovering", flush=True)
            rehover(k, p_hover)
        tau = 0.15
        for attempt in range(20):
            psi0 = psi_of(k)
            drive(k, tau, p_hover, 12)
            psi1 = psi_of(k)
            piv = pivot_of(k)[0]
            dev = float((piv - p_hover[0]).norm())
            print(f"[solve] {name} squeeze a{attempt}: tau={tau:.2f} "
                  f"psi {psi0:+.1f} -> {psi1:+.1f} (hover dev {dev * 1000:.1f} mm)",
                  flush=True)
            if psi1 >= PSI_OPEN_TGT:
                print(f"[solve] {name} jaws open: psi={psi1:+.1f} deg "
                      f"(tau={tau:.2f} N*m, mode={mode[k]})", flush=True)
                return
            if dev > 0.030:  # pose degraded — recover and continue the ramp
                print(f"[solve] {name} drifted {dev * 1000:.0f} mm off hover — "
                      f"re-hovering", flush=True)
                rehover(k, p_hover)
            if psi1 - psi0 < 0.8:  # equilibrium against the spring: more torque
                tau = min(tau * 1.4, 0.6)
        report(f"{scene.CLIPS[k]}-STUCK")
        print(f"SIM_GEN_SOLVE: FAIL ({scene.CLIPS[k]} jaws never opened)", flush=True)
        os._exit(1)

    def clamp_clip(k: int) -> None:
        """The full per-clip routine: teleport-hover (shut) -> squeeze open ->
        servo descent astride the blade -> seat -> ramp the squeeze off so the
        spring clamps -> hands-off settle."""
        name = scene.CLIPS[k]
        a, b = levers(k)
        top = scene.zone_top_w()[:, k]  # (n, 3) world zone-top point

        # TRANSPORT: hover above the zone top, upright, hinge along the fence, shut.
        p_hover = top.clone()
        p_hover[:, 2] += HOVER_DZ
        st_a, st_b = clip_root_states(p_hover, scene.fence.data.root_quat_w,
                                      PSI_CLOSED)
        a.write_root_state_to_sim(st_a, all_ids)
        b.write_root_state_to_sim(st_b, all_ids)
        drive(k, 0.0, p_hover, 24)  # servo-hold the hover, no squeeze yet
        report(f"{name}-hover")
        assert psi_of(k) < PSI_CLOSED + 3.0, \
            f"{name} must hover spring-shut, psi={psi_of(k):+.1f}"

        # SQUEEZE: opposed torques cam the jaws open against the stop.
        open_jaws(k, p_hover)
        drive(k, TAU_HOLD, p_hover, 24)  # stabilize pinned at the stop
        assert psi_of(k) >= PSI_OPEN_TGT - 1.5, \
            f"{name} jaws did not stay open, psi={psi_of(k):+.1f}"

        # INSERTION: descend the servo target so the open jaws pass astride the
        # blade; stop when the pivot bosses SEAT on the top edge (descent stalls).
        p_des = p_hover.clone()
        z_end = top[:, 2] + SEAT_DZ
        seated = False
        for burst in range(40):
            p_des[:, 2] = (p_des[:, 2] - 0.004).clamp(min=z_end)
            z0 = float(pivot_of(k)[0, 2])
            drive(k, TAU_HOLD, p_des, 8)
            z1 = float(pivot_of(k)[0, 2])
            at_bottom = bool((p_des[:, 2] <= z_end + 1e-6).all())
            if at_bottom and z0 - z1 < 0.0004:  # target reached and descent stalled
                seated = True
                print(f"[solve] {name} seated: pivot_z={z1:.4f} "
                      f"(blade top {float(top[0, 2]):.4f}, burst {burst}, "
                      f"psi={psi_of(k):+.1f})", flush=True)
                break
        if not seated:
            report(f"{name}-NOSEAT")
            print(f"SIM_GEN_SOLVE: FAIL ({name} never seated on the blade top)",
                  flush=True)
            os._exit(1)
        assert psi_of(k) >= 4.0, \
            f"{name} jaws closed during descent, psi={psi_of(k):+.1f}"

        # CLAMP: ramp the squeeze torque off; the spring closes the jaws onto the
        # blade faces. Servo stays on (position only) until the very end so the
        # closing jaws can't kick the clip off, then everything hands-off.
        tau = TAU_HOLD
        for _ in range(6):
            tau *= 0.5
            drive(k, tau, p_des, 16)
        drive(k, 0.0, p_des, 16)
        clear_wrenches(k)
        step(90)
        report(f"{name}-clamped")

    def summarize_layout() -> None:
        fpq = scene.fence.data.root_pos_w[0] - scene.env_origins[0]
        fq = scene.fence.data.root_quat_w[0]
        fyaw = math.degrees(2.0 * math.atan2(float(fq[3]), float(fq[0])))
        zy = scene._zone_y[0]
        piv = [scene._fence_local(pivot_of(k))[0] for k in range(2)]
        print(f"[solve] layout readback (seed {args.seed}): "
              f"fence=({float(fpq[0]):+.3f},{float(fpq[1]):+.3f}) "
              f"yaw={fyaw:+.1f}deg zones(red,blue)=({float(zy[0]):+.3f},"
              f"{float(zy[1]):+.3f}) "
              + " ".join(f"{scene.CLIPS[k]}_local=({float(piv[k][0]):+.3f},"
                         f"{float(piv[k][1]):+.3f},{float(piv[k][2]):+.3f})"
                         for k in range(2)), flush=True)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)  # clips settle lying on their sides, jaws spring-shut
    summarize_layout()
    report("reset")
    pos, _q, _v, _w = scene._lever_tensors()
    assert torch.isfinite(pos).all(), "NaN/inf in lever states after settle"
    for k in range(2):
        psi = psi_of(k)
        assert PSI_CLOSED - 4.0 <= psi <= PSI_CLOSED + 4.0, \
            f"free {scene.CLIPS[k]} clip must rest spring-shut near {PSI_CLOSED} " \
            f"deg, got {psi:+.1f} (joint sign/limit sanity)"
        assert float(pivot_of(k)[0, 2]) < 0.05, \
            f"{scene.CLIPS[k]} clip must start on the ground"
    # capture reset-frame rotations for the wrench-encoding probe
    for k in range(2):
        for body, tag in zip(levers(k), ("a", "b")):
            R_ref[f"{scene.CLIPS[k]}_{tag}"] = \
                matrix_from_quat(body.data.root_quat_w).clone()
    s0 = print_score("P0 reset+settle (both clips shut on the ground)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.01, f"baseline score should be ~0, got {s0}"
    try:  # diagnostic: what the USD stage actually holds for the hinge (on record)
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath("/World/envs/env_0/Clip_red_b/hinge")
        vals = {a: prim.GetAttribute(a).Get() for a in
                ("physics:lowerLimit", "physics:upperLimit",
                 "drive:angular:physics:stiffness",
                 "drive:angular:physics:targetPosition")}
        print(f"[solve] hinge USD readback: {vals}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[solve] hinge USD readback failed: {e}", flush=True)

    # ---------------- phase 1: red clip -----------------------------------------------------
    clamp_clip(0)
    assert bool(scene.astride()[0, 0]), "red clip must be astride its zone"
    s1 = print_score("P1 red clip clamped astride the red band's top edge")
    assert s1 >= max(s0, 0.39) - 1e-6, f"P1 score {s1} (expect >= 0.40)"

    # ---------------- phase 2: blue clip ----------------------------------------------------
    clamp_clip(1)
    assert bool(scene.astride()[0, 1]), "blue clip must be astride its zone"
    assert bool(scene.astride()[0, 0]), "red clip must remain astride"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success with both clips clamped)", flush=True)
        os._exit(1)
    s2 = print_score("P2 blue clip clamped astride the blue band's top edge")
    assert s2 >= s1 - 1e-6, "score decreased across the final clamp"

    # ---------------- phase 3: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s3 = print_score("P3 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s3 >= s2 - 1e-6
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
    main()

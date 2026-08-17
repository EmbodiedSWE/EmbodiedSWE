"""Teleport solution for RingBalanceScene (sim_gen task `insert_onto_square_peg_i268`)
— the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. COUNT: read the episode's ballast count k and side from the scene handles
   (the policy equivalent is counting the black stack visually or probing the
   beam's response ring by ring; the solve demonstrates the k-dependent plan).
2. FETCH each gold ring (applied force): a PD force + gravity feedforward — a
   firm force-limited pinch — lifts the ring flat off the floor. The `lifted`
   latch fires DURING this force lift.
3. TRANSPORT (teleport): one root-state write carries the held ring through
   free air to a hover 40+ mm ABOVE the free pan's funnel tip, bore aligned
   with the post axis, zero velocity. Open sky — the hover is OUTSIDE the
   capture window (asserted), so no rubric clause is satisfied by the write.
4. THREADING (guided force descent + funnel + gravity): the same force-limited
   carry lowers the ring along the BEAM-LOCAL post axis on a slow moving
   target; the stepped funnel tip (13 -> 20 mm over the 30 mm bore) and the
   26 mm shaft steer the bore; a yaw servo holds the square bore aligned with
   the square shaft (only +/-12 deg relative yaw admits). The ring is RELEASED
   and settles onto the stack under gravity — capture is produced by contact.
5. EQUILIBRIUM (pure mechanism physics, hands-off): while any imbalance
   remains the beam stays pressed on its ballast stop (asserted after every
   intermediate ring). After the k-th ring is released the beam ALONE swings
   up and settles level; the success streak (1 s settled-level) accumulates
   entirely hands-off, then must persist >= 3.3 more simulated seconds.

Order: fetch -> thread x k -> equilibrium is forced physically: only mass in
the free pan moves the settled angle (the stops are joint limits; nothing can
prop the airborne pans), and the pan post is the only place a ring counts.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's stage credit is latched) and `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds after the hands-off persistence window.

Run (forge): python -u -m simgen_tasks.insert_onto_square_peg_i268.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ring_balance")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so
    # the task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def ang() -> float:
        return float(scene.beam_angle_deg()[0])

    def report(tag: str) -> None:
        print(f"[solve] {tag:16s} | ang={ang():+7.2f}deg "
              f"n_free={int(scene.n_free()[0])} "
              f"ballast_ok={bool(scene.ballast_intact()[0])} "
              f"streak={int(scene._streak[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    zero = torch.zeros(n, 1, 3, device=device)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
    ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
    ring_m = float(scene.gold[0].root_physx_view.get_masses()[0].sum())

    def carry(body, tgt_fn, *, frame_body=None, kp: float = 15.0, kd: float = 6.0,
              clamp: float = 7.5, ku: float = 0.008, kw: float = 0.006,
              kyaw: float = 0.0, kspin: float = 0.004, steps: int = 600,
              done=None, log_every: int = 0, label: str = "") -> None:
        """Applied-force carry of one ring: PD toward tgt_fn(i) (frame_body-local
        if given, else world) + gravity feedforward — a firm force-limited grasp.
        Righting torque stands the ring bore-up along the frame's up axis; the
        yaw servo (mod 90 deg — the ring is square-symmetric) keeps the square
        bore aligned with the square shaft. Gains respect the 1-substep wrench
        delay for the 0.377 kg ring (kp*dt/m ~ 0.33; kw*dt/I_t ~ 0.2;
        kspin near-critically damps the kyaw yaw servo)."""
        for i in range(steps):
            q = body.data.root_quat_w
            p = body.data.root_pos_w
            v = body.data.root_lin_vel_w
            w = body.data.root_ang_vel_w
            tgt = torch.zeros(n, 3, device=device)
            tgt[:] = torch.tensor(tgt_fn(i), device=device)
            if frame_body is not None:
                fq = frame_body.data.root_quat_w
                tgt_w = frame_body.data.root_pos_w + quat_apply(fq, tgt)
                a_tgt = quat_apply(fq, ez)
                x_tgt = quat_apply(fq, ex)
            else:
                tgt_w = tgt
                a_tgt = ez
                x_tgt = ex
            f_w = ring_m * 9.81 * ez + kp * (tgt_w - p) - kd * v
            f_norm = f_w.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            f_w = f_w * (f_norm.clamp(max=clamp) / f_norm)
            axis = quat_apply(q, ez)
            w_ax = (w * axis).sum(dim=-1, keepdim=True)
            w_perp = w - w_ax * axis
            t_w = ku * torch.cross(axis, a_tgt, dim=-1) - kw * w_perp
            if kyaw > 0.0:
                # yaw error mod 90 deg between the ring's and the frame's x axes,
                # both projected onto the plane normal to the frame's up axis
                xr = quat_apply(q, ex)
                xr = xr - (xr * a_tgt).sum(dim=-1, keepdim=True) * a_tgt
                xf = x_tgt - (x_tgt * a_tgt).sum(dim=-1, keepdim=True) * a_tgt
                cross = (torch.cross(xf, xr, dim=-1) * a_tgt).sum(dim=-1)
                dot = (xf * xr).sum(dim=-1)
                err = torch.atan2(cross, dot)
                err = torch.remainder(err + math.pi / 4, math.pi / 2) - math.pi / 4
                t_w = t_w - (kyaw * err + kspin * w_ax.squeeze(-1)).unsqueeze(-1) * axis
            body.set_external_force_and_torque(
                quat_apply_inverse(q, f_w).reshape(n, 1, 3),
                quat_apply_inverse(q, t_w).reshape(n, 1, 3))
            env.step(no_action)
            if log_every and i % log_every == 0:
                loc = scene._beam_local(body.data.root_pos_w)[0]
                axis_now = quat_apply(body.data.root_quat_w, ez)
                tilt = math.degrees(math.acos(
                    min(1.0, max(-1.0, float((axis_now * a_tgt).sum(dim=-1)[0])))))
                xr = quat_apply(body.data.root_quat_w, ex)
                xr = xr - (xr * a_tgt).sum(dim=-1, keepdim=True) * a_tgt
                xf2 = x_tgt - (x_tgt * a_tgt).sum(dim=-1, keepdim=True) * a_tgt
                ye = math.degrees(float(torch.atan2(
                    (torch.cross(xf2, xr, dim=-1) * a_tgt).sum(dim=-1),
                    (xf2 * xr).sum(dim=-1))[0]))
                ye = (ye + 45.0) % 90.0 - 45.0
                print(f"[solve]   {label} i={i:4d} beam-local="
                      f"({float(loc[0]):+.4f},{float(loc[1]):+.4f},{float(loc[2]):+.4f}) "
                      f"tilt={tilt:5.2f}deg yaw_err={ye:+6.2f}deg", flush=True)
            if done is not None and done():
                break
        body.set_external_force_and_torque(zero, zero)
        pw = (body.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] carry {label}: at env "
              f"({float(pw[0]):+.3f},{float(pw[1]):+.3f},{float(pw[2]):+.3f}) "
              f"after <= {steps} steps", flush=True)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(240)  # beam settles pressed onto its ballast-side stop
    k = int(scene.k_count[0])
    sgn = float(scene.ballast_sign[0])
    sf = -sgn  # free-pan side
    sp = (scene.stand.data.root_pos_w - scene.env_origins)[0]
    sq = scene.stand.data.root_quat_w[0]
    syaw = math.degrees(2.0 * math.atan2(float(sq[3]), float(sq[0])))
    print(f"[solve] layout readback (seed {args.seed}): "
          f"stand=({float(sp[0]):+.3f},{float(sp[1]):+.3f}) yaw={syaw:+.1f}deg "
          f"ballast k={k} on {'+x' if sgn > 0 else '-x'} pan, ang={ang():+.2f}deg",
          flush=True)
    # mass readbacks: custom spawners apply no cfg mass schemas — guard the
    # authored per-child density / root MassAPI against regression
    beam_m = float(scene.beam.root_physx_view.get_masses()[0].sum())
    stand_m = float(scene.stand.root_physx_view.get_masses()[0].sum())
    print(f"[solve] mass readback: ring={ring_m:.4f} kg beam={beam_m:.3f} kg "
          f"stand={stand_m:.1f} kg (cfg ring_mass={c.ring_mass:.4f})", flush=True)
    assert abs(ring_m - c.ring_mass) < 0.05, f"ring mass {ring_m} (density not applied?)"
    assert 0.5 < beam_m < 6.0, f"beam mass {beam_m} (density not applied?)"
    assert stand_m > 25.0, f"stand mass {stand_m} (MassAPI not applied?)"
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert abs(ang()) >= 10.0 and ang() * sgn > 0, \
        f"beam must rest pressed on the BALLAST-side stop, ang={ang():+.2f} sgn={sgn:+.0f}"
    assert bool(scene.ballast_intact()[0]), "ballast stack must survive the settle"
    assert int(scene.n_free()[0]) == 0, "free post must start empty"
    s0 = print_score("P0 reset+settle (beam pressed on the ballast stop)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phases 1..k: fetch -> transport -> thread, one ring each -------------
    pitch = c.ring_t + c.stack_gap
    s_prev = s0
    for j in range(k):
        body = scene.gold[j]
        # --- fetch: force-lift the ring flat off the floor -----------------------------
        p0 = body.data.root_pos_w[0].clone()
        lift_tgt = (float(p0[0]), float(p0[1]), float(p0[2]) + 0.24)
        carry(body, lambda i: lift_tgt, kp=15.0, kd=6.0, clamp=7.5,
              ku=0.008, kw=0.006, steps=360,
              done=lambda: float(body.data.root_pos_w[0, 2]
                                 - scene.env_origins[0, 2]) > 0.20,
              label=f"gold_{j} lift")
        zlift = float(body.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
        assert zlift > c.lift_z, f"force lift failed, z={zlift:.3f}"
        assert bool(scene._lifted[0]), "lifted must latch during the force lift"

        # --- transport: teleport to a hover ABOVE the free post's funnel tip -----------
        hover = torch.zeros(n, 3, device=device)
        hover[:, 0] = sf * c.arm
        hover[:, 2] = 0.060  # beam-local: well above the funnel tip, OUTSIDE capture

        def to_hover(body=body) -> None:
            st = torch.zeros(n, 13, device=device)
            st[:, 0:3] = scene.beam.data.root_pos_w \
                + quat_apply(scene.beam.data.root_quat_w, hover)
            st[:, 3:7] = scene.beam.data.root_quat_w  # bore + yaw aligned with the post
            body.write_root_state_to_sim(st, all_ids)

        to_hover()
        assert not bool(scene._on_post(body, torch.full((n,), sf, device=device))[0]), \
            "hover must be outside the capture window"
        print(f"[solve] P{j + 1}: gold_{j} transported to free air above the "
              f"free pan's funnel tip (nothing judged is satisfied by the write)",
              flush=True)

        # --- thread: slow guided force descent along the post axis, then release -------
        z_seat = -c.pan_drop + c.ring_t / 2 + 0.001 + j * pitch
        z_hi = 0.060
        sf_t = torch.full((n,), sf, device=device)

        def seated(body=body, z_seat=z_seat, sf_t=sf_t) -> bool:
            loc = scene._beam_local(body.data.root_pos_w)[0]
            return bool(scene._on_post(body, sf_t)[0]) and float(loc[2]) < z_seat + 0.006

        # escalating retries: each attempt descends slower with a stiffer
        # orientation servo; a failed attempt force-lifts back above the tip and
        # re-zeroes at the hover (free air) before trying again
        for attempt in range(4):
            desc = int(300 * (1.5 ** attempt))
            kyaw = 0.010 * (2.0 ** attempt)
            ku_a = min(0.008 * (2.0 ** attempt), 0.03)
            kw_a = min(0.006 * (1.5 ** attempt), 0.015)
            ksp = min(0.004 * (2.0 ** attempt), 0.015)

            def tgt_fn(i: int, z_hi=z_hi, z_seat=z_seat, desc=desc):
                f = min(i / desc, 1.0)
                return (sf * c.arm, 0.0, z_hi + (z_seat - 0.002 - z_hi) * f)

            carry(body, tgt_fn, frame_body=scene.beam, kp=15.0, kd=6.0, clamp=7.5,
                  ku=ku_a, kw=kw_a, kyaw=kyaw, kspin=ksp, steps=desc + 300,
                  done=seated, log_every=120, label=f"gold_{j} thread a{attempt}")
            if seated():
                break
            loc = scene._beam_local(body.data.root_pos_w)[0]
            print(f"[solve] thread attempt {attempt} stuck at beam-local "
                  f"({float(loc[0]):+.4f},{float(loc[1]):+.4f},{float(loc[2]):+.4f})"
                  f" — lifting off and retrying slower", flush=True)
            carry(body, lambda i: (sf * c.arm, 0.0, 0.060), frame_body=scene.beam,
                  kp=15.0, kd=6.0, clamp=7.5, ku=0.008, kw=0.006, steps=300,
                  done=lambda: float(scene._beam_local(body.data.root_pos_w)[0, 2]) > 0.050,
                  label=f"gold_{j} recover a{attempt}")
            to_hover()  # re-zero in free air at the hover (transport only)
        assert seated(), f"gold_{j} failed to thread onto the free post"
        step(180)  # hands off: the ring settles onto the stack, the beam responds
        assert bool(scene._on_post(body, sf_t)[0]), \
            f"gold_{j} must rest captured on the free post after release"
        assert bool(scene.ballast_intact()[0]), "ballast stack must stay intact"
        report(f"ring {j + 1}/{k}")
        if j + 1 < k:
            assert abs(ang()) >= 8.0 and ang() * sgn > 0, \
                f"under-loaded beam must stay pressed on the ballast stop, ang={ang():+.2f}"
            assert not bool(scene.success()[0]), "under-load must not be success"
        s_j = print_score(f"P{j + 1} ring {j + 1}/{k} threaded on the free post")
        assert s_j >= s_prev - 1e-6, f"score decreased: {s_prev} -> {s_j}"
        want = min(c.w_lift + c.w_load * (j + 1) / k, 0.60)
        assert s_j >= want - 0.02 or s_j >= 0.999, f"P{j + 1} score {s_j} < {want}"
        s_prev = s_j

    # ---------------- phase k+1: the beam ALONE swings level (hands-off) -------------------
    for i in range(2400):  # up to 20 s; the streak gate needs 1 s settled-level
        env.step(no_action)
        if bool(scene.success()[0]):
            break
    report("equilibrium")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (beam did not settle level)", flush=True)
        os._exit(1)
    assert abs(ang()) <= c.succ_tol_deg, f"level? ang={ang():+.2f}"
    s_eq = print_score(f"P{k + 1} beam settled LEVEL hands-off ({k} vs {k} rings)")
    assert s_eq >= 0.999, f"success score must be 1.0, got {s_eq}"

    # ---------------- final: persistence (>= 3 simulated seconds, hands-off) ---------------
    hold = True
    n_bad = 0
    for tick in range(400):  # 400 substeps = 3.33 s at 120 Hz
        env.step(no_action)
        raw = bool(scene._raw_success()[0])
        if (not raw) and n_bad < 25:  # diagnose exactly which settle component broke
            n_bad += 1
            bav = float(scene.beam.data.root_ang_vel_w.norm(dim=-1)[0])
            sv = float(scene.stand.data.root_lin_vel_w.norm(dim=-1)[0])
            gv = [float(b.data.root_lin_vel_w.norm(dim=-1)[0]) for b in scene.gold]
            bv = [float(b.data.root_lin_vel_w.norm(dim=-1)[0]) for b in scene.black]
            gi = max(range(len(gv)), key=gv.__getitem__)
            bi = max(range(len(bv)), key=bv.__getitem__)
            print(f"[solve]   persist BREAK t={tick} ang={ang():+.2f} beam_avel={bav:.3f} "
                  f"stand_v={sv:.3f} gold_max=g{gi}:{gv[gi]:.3f} black_max=b{bi}:{bv[bi]:.3f} "
                  f"streak={int(scene._streak[0])}", flush=True)
        if tick % 40 == 39:
            hold = hold and bool(scene.success()[0])
    report("persist")
    s_f = print_score("Pfinal persistence 3.3 s hands-off")
    ok = hold and bool(scene.success()[0]) and s_f >= s_eq - 1e-6
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
    except Exception as exc:  # noqa: BLE001 - Kit teardown hangs; die loudly NOW
        print(f"SIM_GEN_SOLVE: FAIL (exception: {exc!r})", flush=True)
        import traceback

        traceback.print_exc()
        os._exit(1)

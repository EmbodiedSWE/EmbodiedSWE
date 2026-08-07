"""Teleport solution for DieTipPadScene (sim_gen task `approach_grasp_spoon_i12`) —
the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY, in ONE write that ends in FREE SPACE:
  P1 — the die is carried from its spawn ring to a staging point on the boarding
  line, `n_tips * die_a` short of the disk centre, 1 mm above the floor, with its
  ORIENTATION PRESERVED EXACTLY (the transport never turns the die) and velocities
  zeroed. Everything load-bearing then happens through CONTACT DYNAMICS:
  P2 — the die is tipped end-over-end by a pure torque about a horizontal world
  axis (the couple a fingertip pushing HIGH on a face produces), pivoting on its
  leading bottom edge under real floor contact and gravity; the torque is cut at
  55 degrees (past the 45-degree balance point) and gravity finishes each tip.
  Each 90-degree tip rolls the next face up AND advances the die one face length,
  so the staging point is chosen so the planned tip sequence ends blue-up ON the
  disk. If blue starts DOWN, two tips along the same bearing are needed; if blue
  starts HORIZONTAL, one tip against the blue normal. Between tips the solver
  re-reads the physical pose and re-plans; small placement errors are corrected by
  LOW push-slides (COM force plus the counter-torque of a push at 30 mm height —
  below the tip threshold a/(2*mu_eff) ~ 53 mm — so the die skates without
  tipping, exactly as a fingertip pushing low would move it).
  The die is never teleported onto the disk and never teleported into a new
  orientation; every orientation change and every metre of judged approach after
  staging goes through floor contact.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds
after success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds.

Run (forge): python -u -m simgen_tasks.approach_grasp_spoon_i12.solve --headless [--seed N]
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

BLUE_IDX = scene_mod.BLUE_IDX
FACE_NAMES = scene_mod.FACE_NAMES

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import matrix_from_quat

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.die_tip_pad")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)
    a = c.die_a

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def pad_xy() -> torch.Tensor:
        """(2,) disk centre, env-local."""
        return (scene.pad.data.root_pos_w - scene.env_origins)[0, :2]

    def die_xy() -> torch.Tensor:
        """(2,) die centre, env-local."""
        return (scene.die.data.root_pos_w - scene.env_origins)[0, :2]

    def blue_world() -> torch.Tensor:
        """(3,) world direction of the BLUE (+z body) face normal."""
        r = matrix_from_quat(scene.die.data.root_quat_w)  # body->world
        return r[0, :, 2]

    def face_nz() -> torch.Tensor:
        return scene._face_nz()[0]

    def clear_wrench() -> None:
        scene.die.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def report(tag: str) -> None:
        nz = face_nz()
        up = int(nz.argmax())
        d = float((die_xy() - pad_xy()).norm())
        print(f"[solve] {tag:14s} | d_pad={d:.3f} z={float(scene._die_z()[0]):.3f} "
              f"up={FACE_NAMES[up]}({float(nz[up]):.3f}) blue_nz={float(nz[BLUE_IDX]):+.3f} "
              f"flat={bool(scene.flat()[0])} on_pad={bool(scene.on_pad()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ----- contact-dynamics primitives ------------------------------------------------------
    def do_tip(d_hat: torch.Tensor) -> None:
        """One 90-degree tip toward world xy direction `d_hat` (unit 2-vector): pure
        torque about the horizontal axis z_hat x d_hat (the couple of a HIGH push),
        omega-regulated, cut at 55 degrees; gravity and floor contact finish the tip."""
        a_hat = torch.tensor([-float(d_hat[1]), float(d_hat[0]), 0.0], device=device)
        u = int(face_nz().argmax())  # face up at tip start
        cut = math.cos(math.radians(55.0))
        tau_cap = 0.55  # > m g a/2 = 0.245 N*m needed to start the pivot
        for i in range(900):
            if float(face_nz()[u]) < cut:
                break
            w_a = float((scene.die.data.root_ang_vel_w[0] * a_hat).sum())
            tau = max(0.0, min(tau_cap, 1.2 * (1.5 - w_a)))
            tq = (tau * a_hat).view(1, 1, 3).expand(n, 1, 3).contiguous()
            scene.die.set_external_force_and_torque(zero_wrench, tq, env_ids=all_ids,
                                                    is_global=True)
            env.step(no_action)
            if i % 300 == 299:  # stalled (stiction / rim lip): lean harder
                tau_cap = min(tau_cap + 0.15, 1.0)
                print(f"[solve] tip stalled, tau_cap={tau_cap:.2f} N*m", flush=True)
        clear_wrench()
        step(110)  # land on the far face + settle (~0.9 s)

    def do_slide(tgt_xy: torch.Tensor, tol: float = 0.020, max_steps: int = 900) -> None:
        """LOW push-slide to env-local `tgt_xy`: COM force plus the counter-torque of a
        push at 30 mm height (below the a/(2 mu_eff) ~ 53 mm tip threshold), velocity
        capped at 0.15 m/s so the die skates flat without tipping."""
        r_z = 0.030 - c.half  # push point 20 mm below the COM
        for _ in range(max_steps):
            e = tgt_xy - die_xy()
            dist = float(e.norm())
            v = scene.die.data.root_lin_vel_w[0, :2]
            if dist < tol and float(v.norm()) < 0.05:
                break
            e_hat = e / max(dist, 1e-6)
            v_des = e_hat * min(0.15, 2.0 * dist)
            f_xy = c.die_mass * 40.0 * (v_des - v) + e_hat * (0.8 * c.die_mass * 9.81)
            f_xy = f_xy.clamp(-8.0, 8.0)
            f = torch.tensor([float(f_xy[0]), float(f_xy[1]), 0.0], device=device)
            # torque of the same force applied at 30 mm height: r x F, r = (0,0,r_z)
            tq = torch.tensor([r_z * -float(f_xy[1]), r_z * float(f_xy[0]), 0.0],
                              device=device)
            scene.die.set_external_force_and_torque(
                f.view(1, 1, 3).expand(n, 1, 3).contiguous(),
                tq.view(1, 1, 3).expand(n, 1, 3).contiguous(),
                env_ids=all_ids, is_global=True)
            env.step(no_action)
        clear_wrench()
        step(60)  # settle

    # ---------------- phase 0: reset, settle, baseline --------------------------------------
    step(90)
    p_xy = pad_xy()
    d_xy0 = die_xy()
    dec0 = (scene.decoy.data.root_pos_w - scene.env_origins)[0, :2]
    nz0 = face_nz()
    print(f"[solve] layout readback (seed {args.seed}): pad=({float(p_xy[0]):+.3f},"
          f"{float(p_xy[1]):+.3f}) die=({float(d_xy0[0]):+.3f},{float(d_xy0[1]):+.3f}) "
          f"decoy=({float(dec0[0]):+.3f},{float(dec0[1]):+.3f}) "
          f"up_face={FACE_NAMES[int(nz0.argmax())]} blue_nz={float(nz0[BLUE_IDX]):+.3f} "
          f"d0={float(scene.d0[0]):.3f}", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: die TRANSPORT (teleport to staging, free space) --------------
    # Plan the tip sequence from the settled physical pose, then stage the die
    # n_tips * a short of the disk centre along the tip bearing. Orientation is
    # PRESERVED EXACTLY; velocities zeroed; the staging point is on the open floor.
    b = blue_world()
    if float(b[2]) < -0.7:
        # blue DOWN: two tips along one bearing. Pick the bearing whose staging point
        # is farthest from the decoy so the boarding corridor is clear.
        best_d, best_dist = None, -1.0
        for j in range(16):
            th = j * math.pi / 8
            cand = torch.tensor([math.cos(th), math.sin(th)], device=device)
            stg = p_xy - 2 * a * cand
            dd = float((stg - dec0).norm())
            if dd > best_dist:
                best_d, best_dist = cand, dd
        d_plan, n_tips = best_d, 2
    else:
        # blue HORIZONTAL: one tip against the blue normal maps blue to up.
        h = b[:2]
        d_plan = -h / max(float(h.norm()), 1e-6)
        n_tips = 1
    stage_xy = p_xy - n_tips * a * d_plan
    print(f"[solve] plan: n_tips={n_tips} d_hat=({float(d_plan[0]):+.2f},"
          f"{float(d_plan[1]):+.2f}) staging=({float(stage_xy[0]):+.3f},"
          f"{float(stage_xy[1]):+.3f})", flush=True)

    st = torch.zeros(n, 13, device=device)
    st[:, 0:2] = stage_xy
    st[:, 2] = c.half + 0.003
    st[:, 3:7] = scene.die.data.root_quat_w[0]  # orientation preserved: transport only
    st[:, 0:3] += scene.env_origins
    scene.die.write_root_state_to_sim(st, all_ids)
    step(60)  # settle on real floor contact
    report("staged")
    s1 = print_score("P1 die transport to staging")
    assert s1 >= s0 - 1e-6, "score decreased across the transport"

    # ---------------- phase 2: contact tips (+ corrective low slides) -----------------------
    for cycle in range(6):
        nz = face_nz()
        if float(nz[BLUE_IDX]) > math.cos(math.radians(6.0)):
            break  # blue is up (tighter than the rubric's 10 degrees)
        b = blue_world()
        if float(b[2]) < -0.7:
            d_hat, n_rem = d_plan, 2  # blue still down: keep the planned bearing
        else:
            h = b[:2]
            d_hat = -h / max(float(h.norm()), 1e-6)
            n_rem = 1
        pre_tgt = p_xy - n_rem * a * d_hat
        err = float((die_xy() - pre_tgt).norm())
        if err > 0.040:
            print(f"[solve] cycle {cycle}: pre-tip slide (err={err:.3f})", flush=True)
            do_slide(pre_tgt)
            continue  # re-read and re-plan after the slide
        print(f"[solve] cycle {cycle}: tip toward ({float(d_hat[0]):+.2f},"
              f"{float(d_hat[1]):+.2f}), {n_rem} remaining", flush=True)
        do_tip(d_hat)
        report(f"after-tip-{cycle}")

    # Centering: if the tips left the die off-centre, push it LOW onto the disk.
    dist = float((die_xy() - p_xy).norm())
    if dist > 0.055:
        print(f"[solve] centering slide (d_pad={dist:.3f})", flush=True)
        do_slide(p_xy, tol=0.025)
    step(120)  # 1 s hands-off settle before judging
    report("boarded")
    s2 = print_score("P2 contact tips + boarding + settle")
    assert s2 >= s1 - 1e-6, "score decreased across the tip sequence"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after tips+settle)", flush=True)
        os._exit(1)

    # ---------------- phase 3: persistence (>= 3.3 simulated seconds, no intervention) ------
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

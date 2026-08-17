"""Teleport solution for MugRackHangScene (sim_gen task `approach_grasp_ceramic_teapot_i109`)
— the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. READ THE TARGET: the solver reads the green marker's pose on the panel top edge
   (the same cue a camera would show) and identifies the target peg by its column;
   it cross-checks against the scene's sampled index by READBACK, never by trusting
   hidden state.
2. TRANSPORT (teleport, one write): a single root-state write carries the white mug
   from the floor to a FREE-SPACE staging pose 12 mm off the target peg's TIP —
   handle window centered on the peg's extended axis, loop plane square to it, zero
   velocity. Nothing touches anything: the write satisfies no rubric clause (the
   mug is held by the hand-wrench, not hanging, and the peg is not through the
   window).
3. INSERT (wrench hand, real forces): a bounded external wrench — the hand — holds
   the mug's weight (2.5 N) and pushes it along the peg axis (<= 2.5 N, fingertip
   scale) with a lateral PD that keeps the window centered and an uprighting/yaw
   torque (<= 0.15 N m) that keeps the loop square, sliding the open window over
   the peg into its span. Any contact en route (window edge on peg) is genuine
   collision response against the push.
4. RELEASE + HANG (gravity + contact, hands-off): the wrench is zeroed. The mug
   falls ~18 mm until the top handle bar CATCHES on the peg, the upward peg tilt
   slides the hang to the peg root (slick peg — asserted tan(tilt) > mu), the mug
   leans against the panel, swings, and settles. The suspended equilibrium that
   success() reads is produced entirely by gravity and contact, never written.
5. CLOSED LOOP: if the mug ends up on the floor (missed catch, knocked off), the
   solver re-stages through free space and repeats.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.approach_grasp_ceramic_teapot_i109.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401  (registers simgen.mug_rack_hang)
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.mug_rack_hang")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    white = scene.mugs[c.mug_names[0]]
    m_mug = c.mug_mass
    G = 9.81

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def peg_frame() -> tuple[torch.Tensor, torch.Tensor]:
        root_w, dir_w = scene._peg_world(scene.green_idx)
        return root_w[0], dir_w[0]

    def ap_w() -> torch.Tensor:
        return scene._aperture_w(white)[0]

    def zero_wrench() -> None:
        z = torch.zeros(n, 1, 3, device=device)
        white.set_external_force_and_torque(z, z.clone(), env_ids=all_ids, is_global=True)

    def report(tag: str) -> None:
        s = scene._status()
        p = (white.data.root_pos_w - scene.env_origins)[0]
        root, d = peg_frame()
        rel = ap_w() - root
        along = float(rel @ d)
        lat = float((rel - (rel @ d) * d).norm())
        print(f"[solve] {tag:14s} | thr={bool(s['threaded'][0])} air={bool(s['airborne'][0])} "
              f"still={bool(s['settled'][0])} streak={int(scene._streak[0])} "
              f"| mug=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f}) "
              f"ap: along={along * 1000:+.1f}mm lat={lat * 1000:.1f}mm "
              f"d_ap={float(s['d_aperture'][0]) * 1000:.0f}mm "
              f"| success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def stage() -> None:
        """TRANSPORT the white mug through free space to the staging pose: handle
        window centered on the target peg's extended axis, 12 mm off the TIP, loop
        plane square to the axis, mug upright, zero velocity. One write; the mug
        touches nothing there (window clearance is asserted in cfg)."""
        root, d = peg_frame()
        tip_pt = root + d * (c.peg_len + 0.012)
        y_m = -d / d.norm()                              # mug +y runs tip -> root
        z_w = torch.tensor([0.0, 0.0, 1.0], device=device)
        z_m = z_w - (z_w @ y_m) * y_m
        z_m = z_m / z_m.norm()
        x_m = torch.linalg.cross(y_m, z_m)
        # rotation matrix [x_m y_m z_m] -> quaternion (wxyz)
        R = torch.stack([x_m, y_m, z_m], dim=1)
        w = math.sqrt(max(1e-9, 1.0 + float(R[0, 0] + R[1, 1] + R[2, 2]))) / 2.0
        q = torch.tensor([w,
                          float(R[2, 1] - R[1, 2]) / (4 * w),
                          float(R[0, 2] - R[2, 0]) / (4 * w),
                          float(R[1, 0] - R[0, 1]) / (4 * w)], device=device)
        q = q / q.norm()
        ap_l = torch.tensor(c.ap_local, device=device)
        pos = tip_pt - quat_apply(q.unsqueeze(0), ap_l.unsqueeze(0))[0]
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3:7] = q
        white.write_root_state_to_sim(st, all_ids)
        zero_wrench()

    def insert() -> bool:
        """Wrench-hand insertion: hold the mug's weight, push along -d, keep the
        window centered on the peg axis (lateral PD) and the loop square
        (uprighting + yaw torque). Returns True when the window center has advanced
        inside the span (along < 32 mm from the root). All forces are bounded and
        fingertip-scale for a 0.25 kg mug."""
        root, d = peg_frame()
        y_tgt = -d
        push = 0.8
        last_s, last_k = 1e9, 0
        for k in range(900):
            rel = ap_w() - root
            s_along = float(rel @ d)
            if s_along < 0.032:
                zero_wrench()
                return True
            # stall escalation: no 3 mm of progress in 90 ticks -> push harder
            if s_along < last_s - 0.003:
                last_s, last_k = s_along, k
            elif k - last_k > 90:
                push = min(push + 0.4, 2.5)
                last_k = k
                print(f"[solve] insert stall -> push {push:.1f} N", flush=True)
            v = white.data.root_lin_vel_w[0]
            w_ang = white.data.root_ang_vel_w[0]
            e = (root + (rel @ d) * d) - ap_w()          # aperture -> axis, perp to d
            e = e - (e @ d) * d
            v_perp = v - (v @ d) * d
            v_along = float(v @ (-d))
            f = torch.zeros(3, device=device)
            f[2] += m_mug * G                            # weight hold
            if v_along < 0.10:
                f += push * (-d)                         # regulated advance
            f_lat = m_mug * (40.0 * e - 12.0 * v_perp)
            ln = f_lat.norm()
            if float(ln) > 2.0:
                f_lat = f_lat * (2.0 / ln)
            f += f_lat
            # torque: upright + loop square to the axis + damping
            qm = white.data.root_quat_w
            zb = quat_apply(qm, torch.tensor([[0.0, 0.0, 1.0]], device=device))[0]
            yb = quat_apply(qm, torch.tensor([[0.0, 1.0, 0.0]], device=device))[0]
            zw = torch.tensor([0.0, 0.0, 1.0], device=device)
            tq = 0.06 * torch.linalg.cross(zb, zw) \
                + 0.06 * torch.linalg.cross(yb, y_tgt) - 0.015 * w_ang
            tn = tq.norm()
            if float(tn) > 0.15:
                tq = tq * (0.15 / tn)
            white.set_external_force_and_torque(
                f.view(1, 1, 3).expand(n, 1, 3).contiguous(),
                tq.view(1, 1, 3).expand(n, 1, 3).contiguous(),
                env_ids=all_ids, is_global=True)
            env.step(no_action)
        zero_wrench()
        return False

    def wait_hang(max_steps: int) -> bool:
        """Hands-off until success() (live hang, streak included) or the mug falls."""
        for k in range(max_steps):
            env.step(no_action)
            if k % 30 == 0 or k > max_steps - 5:
                if bool(scene.success()[0]):
                    return True
                z = float((white.data.root_pos_w - scene.env_origins)[0, 2])
                if z < 0.12:
                    print(f"[solve] mug fell (z={z:.3f})", flush=True)
                    return False
        return bool(scene.success()[0])

    # ---------------- phase 0: reset, settle, baseline ------------------------------------------
    step(150)
    rp = (scene.rack.data.root_pos_w - scene.env_origins)[0]
    qr = scene.rack.data.root_quat_w
    ryaw = math.degrees(2.0 * math.atan2(float(qr[0, 3]), float(qr[0, 0])))
    gi = int(scene.green_idx[0])
    mk_loc = quat_apply_inverse(qr, scene.marker.data.root_pos_w - scene.rack.data.root_pos_w)[0]
    print(f"[solve] layout readback (seed {args.seed}): rack=({float(rp[0]):+.3f},"
          f"{float(rp[1]):+.3f}) yaw={ryaw:+.1f}deg green_idx={gi} "
          f"marker_local=({float(mk_loc[0]):+.3f},{float(mk_loc[1]):+.3f},"
          f"{float(mk_loc[2]):+.3f})", flush=True)
    # the marker (the visible cue) must sit over the target peg's column
    assert abs(float(mk_loc[1]) - c.peg_ys[gi]) < 0.005, "marker does not mark the target peg"
    for name in c.mug_names:
        p = (scene.mugs[name].data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] {name} spawn=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):+.3f})", flush=True)
        assert torch.isfinite(scene.mugs[name].data.root_pos_w).all(), "NaN/inf after settle"
        assert float(p[2]) < 0.08, f"{name} must start standing on the floor"
    report("reset")
    s0 = print_score("P0 reset+settle (both mugs on the floor)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phases 1-3: stage -> insert -> release+hang -------------------------------
    s_prev = s0
    hung = False
    for attempt in range(5):
        stage()
        step(2)  # let the write take; the hand-wrench holds from the first tick
        report(f"staged#{attempt}")
        s1 = print_score("P1 white mug staged at the target peg tip (held by the hand)")
        assert s1 >= s_prev - 1e-6, "score decreased at staging"
        s_prev = s1

        if not insert():
            print(f"[solve] insert retry {attempt + 1}", flush=True)
            step(120)
            continue
        report("inserted")
        assert bool(scene._status()["threaded"][0]), "window not threaded after insert"
        s2 = print_score("P2 handle window threaded over the green peg (wrench insertion)")
        assert s2 >= s_prev - 1e-6, "score decreased after threading"
        assert s2 >= c.w_near + c.w_lift + c.w_thread - 1e-6, f"thread latch missing: {s2}"
        s_prev = s2

        zero_wrench()  # RELEASE — everything from here is gravity + contact
        if wait_hang(1800):
            hung = True
            break
        print(f"[solve] hang retry {attempt + 1}", flush=True)
        step(120)
    if not hung:
        report("FAIL-hang")
        print("SIM_GEN_SOLVE: FAIL (mug never hung settled)", flush=True)
        os._exit(1)
    report("hung")
    s3 = print_score("P3 released — white mug hangs settled on the green peg")
    assert s3 >= s_prev - 1e-6, "score decreased at the hang"
    assert s3 >= 0.999, f"success should score 1.0, got {s3}"

    # ---------------- persistence (>= 3 simulated seconds, hands-off) ---------------------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P-persist 3.3 s hands-off")
    ok = hold and bool(scene.success()[0]) and s4 >= s3 - 1e-6
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
    except BaseException as e:  # noqa: BLE001 - die fast, never idle until the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL (exception: {e})", flush=True)
        os._exit(2)

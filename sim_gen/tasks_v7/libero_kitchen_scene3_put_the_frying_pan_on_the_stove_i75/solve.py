"""Teleport solution for PanHookScene (sim_gen task
`libero_kitchen_scene3_put_the_frying_pan_on_the_stove_i75`) — the task's
legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): single root-state writes carry the pan across FREE SPACE
   with zero velocity — exactly the carry a gripper performs. Two writes:
     (a) pan straight UP off the glowing burner to an open-air hover (the lift; the
         `clear` latch reads the physical fact that the pan bottom is far above the
         burner top);
     (b) pan reoriented VERTICAL (handle up, aperture axis aligned with the peg — the
         in-hand ~90-degree reorientation) to a hover with the loop aperture on the
         peg axis EXTENDED, 45 mm in FRONT of the red knob: the peg is NOT in the
         loop at the hover, and threaded() is asserted False there.
2. THREADING (applied forces, through contact dynamics): a PD wrench (world-frame,
   `is_global=True`) emulates the gripper's hold — gravity compensation plus a
   position servo on the LOOP CENTRE that tracks a straight-line target moving down
   the peg axis, plus an orientation servo holding the hanging attitude. The loop
   physically passes OVER the knob and onto the peg; any misalignment collides the
   loop bars with the knob/peg and the servo must work through it. Nothing is
   teleported once the loop is near the peg.
3. HANG (gravity, hands-off): the wrench is ZEROED; the pan drops ~9 mm until the
   aperture's upper bar lands on the peg, swings, and settles hanging — the final
   support is hook-through-loop contact under gravity, never written.

If a release leaves the pan un-hung (a bounce threw the loop off the knob), the pan
is picked up again (fresh transport to the same free-space hover) and re-threaded —
a retry, not a cheat: the final configuration is still 100 % contact-made.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene3_put_the_frying_pan_on_the_stove_i75.solve --headless [--seed N]
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

_qmul, _qz, _qy, _q_hang = (scene_mod._qmul, scene_mod._qz, scene_mod._qy,
                            scene_mod._q_hang)

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.pan_hook")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    from isaaclab.utils.math import quat_apply

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so
    # the task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    zero3 = torch.zeros(n, 1, 3, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def yaw_of(q) -> float:
        return math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))

    def report(tag: str) -> None:
        pp = scene.pan.data.root_pos_w[0]
        ax = scene._pan_axis_x()[0]
        v = float(scene.pan.data.root_lin_vel_w.norm(dim=-1)[0])
        w = float(scene.pan.data.root_ang_vel_w.norm(dim=-1)[0])
        print(f"[solve] {tag:12s} | pan_z={float(pp[2]):.3f} "
              f"handle_upz={float(ax[2]):+.2f} v={v:.3f} w={w:.2f} "
              f"on_burner={bool(scene.pan_on_burner()[0])} "
              f"clear={bool(scene.clear_of_burner()[0])} "
              f"threaded={bool(scene.threaded()[0])} "
              f"susp={bool(scene.suspended()[0])} "
              f"latches=(c={bool(scene._clear[0])},v={bool(scene._vert[0])},"
              f"t={bool(scene._thread[0])}) "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def write_pose(pos_w, quat_w) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat_w
        scene.pan.write_root_state_to_sim(st, all_ids)

    def hover_pose(s_along: float):
        """Pan pose (world) that puts the loop aperture centre at hook local
        (s_along, 0, 0), in the nominal hanging attitude."""
        qh = _q_hang(n, device)
        h0 = torch.tensor(c.hole_local, device=device).expand(n, 3)
        pos = scene.peg_point_w(s_along) - quat_apply(qh, h0)
        return pos, qh

    def rotvec_to(q_des: torch.Tensor, q_now: torch.Tensor) -> torch.Tensor:
        """(N,3) rotation vector taking q_now to q_des (world frame)."""
        q_conj = q_now.clone()
        q_conj[:, 1:] = -q_conj[:, 1:]
        qe = _qmul(q_des, q_conj)
        # hemisphere fix
        sgn = torch.where(qe[:, 0:1] < 0, -torch.ones_like(qe[:, 0:1]),
                          torch.ones_like(qe[:, 0:1]))
        qe = qe * sgn
        vec = qe[:, 1:4]
        sin_half = vec.norm(dim=-1, keepdim=True)
        angle = 2.0 * torch.atan2(sin_half, qe[:, 0:1].clamp(min=-1.0, max=1.0))
        axis = vec / sin_half.clamp(min=1e-9)
        return torch.where(sin_half > 1e-6, axis * angle, torch.zeros_like(vec))

    # threading servo constants
    s_hover = c.hook_plate_t + c.peg_len + c.knob_len + 0.045   # 45 mm past the knob
    s_seat = c.hook_plate_t + 0.037                             # near the wall plate
    kp, kd = 50.0, 10.0
    kq, kdw = 0.8, 0.08
    fg = c.pan_mass * 9.81

    def thread_servo(max_steps: int = 420, seat_off: float = 0.0) -> bool:
        """PD wrench (is_global) walks the loop centre down the peg axis. Returns
        True when threaded() has held for 30 consecutive substeps. `seat_off`
        shifts the release seat so retries explore, not replay (physics is
        deterministic: an identical retry ends in the identical state)."""
        s_stop = s_seat + seat_off
        qh = _q_hang(n, device)
        held = 0
        travel = s_hover - s_stop
        for i in range(max_steps):
            frac = min(1.0, i / 300.0)
            s_t = s_hover - travel * frac
            target = scene.peg_point_w(s_t)
            h_w = scene.hole_center_w()
            v = scene.pan.data.root_lin_vel_w
            f = kp * (target - h_w) - kd * v
            f[:, 2] += fg
            fn = f.norm(dim=-1, keepdim=True)
            f = torch.where(fn > 12.0, f * (12.0 / fn), f)
            tq = kq * rotvec_to(qh, scene.pan.data.root_quat_w) \
                - kdw * scene.pan.data.root_ang_vel_w
            tn = tq.norm(dim=-1, keepdim=True)
            tq = torch.where(tn > 1.0, tq * (1.0 / tn), tq)
            scene.pan.set_external_force_and_torque(
                f.view(n, 1, 3), tq.view(n, 1, 3), is_global=True)
            env.step(no_action)
            held = held + 1 if bool(scene.threaded()[0]) else 0
            if held >= 30 and frac >= 1.0:
                break
        # hold at the seat a moment so the thread latch (calm gate) fires
        for _ in range(30):
            target = scene.peg_point_w(s_stop)
            h_w = scene.hole_center_w()
            v = scene.pan.data.root_lin_vel_w
            f = kp * (target - h_w) - kd * v
            f[:, 2] += fg
            fn = f.norm(dim=-1, keepdim=True)
            f = torch.where(fn > 12.0, f * (12.0 / fn), f)
            tq = kq * rotvec_to(qh, scene.pan.data.root_quat_w) \
                - kdw * scene.pan.data.root_ang_vel_w
            scene.pan.set_external_force_and_torque(
                f.view(n, 1, 3), tq.view(n, 1, 3), is_global=True)
            env.step(no_action)
        ok = bool(scene.threaded()[0])
        # HANDS OFF: zero the persistent wrench
        scene.pan.set_external_force_and_torque(zero3, zero3, is_global=True)
        return ok

    def settle_pan(max_steps: int = 480) -> None:
        for i in range(max_steps):
            env.step(no_action)
            if i > 30 and bool(scene.settled()[0]):
                break
        step(60)  # extra hands-off settle
        # diagnostics: is the hang really moving, or a phantom velocity read?
        p_a = scene.pan.data.root_pos_w[0].clone()
        q_a = scene.pan.data.root_quat_w[0].clone()
        step(30)
        p_b = scene.pan.data.root_pos_w[0]
        q_b = scene.pan.data.root_quat_w[0]
        dp = float((p_b - p_a).norm())
        dq = float(1.0 - (q_a * q_b).sum().abs())
        print(f"[solve] settle diag: 30-substep drift dp={dp * 1000:.3f}mm "
              f"dq={dq:.2e} thr_streak={int(scene._thr_streak[0])} "
              f"v={float(scene.pan.data.root_lin_vel_w.norm(dim=-1)[0]):.3f} "
              f"w={float(scene.pan.data.root_ang_vel_w.norm(dim=-1)[0]):.3f}",
              flush=True)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # the pan seats on the glowing burner
    bp = (scene.burner.data.root_pos_w - scene.env_origins)[0]
    hp = (scene.hook.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"burner=({float(bp[0]):+.3f},{float(bp[1]):+.3f}) "
          f"hook=(y={float(hp[1]):+.3f},z={float(hp[2]):+.3f}) "
          f"pan_yaw={yaw_of(scene.pan.data.root_quat_w[0]):+.1f}deg", flush=True)
    report("reset")
    assert torch.isfinite(scene.pan.data.root_pos_w).all(), "NaN/inf after settle"
    assert bool(scene.pan_on_burner()[0]), "pan must start lying on the burner"
    s0 = print_score("P0 reset+settle (pan flat on the lit burner)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: lift the pan off the burner ---------------------------------
    kpos = scene.pan.data.root_pos_w.clone()
    kpos[:, 2] += 0.12
    write_pose(kpos, scene.pan.data.root_quat_w.clone())
    step(2)  # refresh + latch (falls < 1 mm in 2 substeps)
    report("lift")
    assert bool(scene._clear[0]), "clear latch must fire at the lift hover"
    assert not bool(scene.pan_on_burner()[0]), "pan still reads on the burner"
    s1 = print_score("P1 pan lifted clear of the glowing burner")
    assert s1 >= s0 - 1e-6 and s1 >= 0.14, f"P1 score {s1} (expect clear=0.15)"

    # ---------------- phases 2-4: carry+reorient, force-thread, release --------------------
    s2 = s1
    for attempt in range(4):
        # -- P2: transport to the aligned free-space hover (vertical, loop on the
        #        peg axis extended, 45 mm in front of the knob)
        pos, quat = hover_pose(s_hover)
        write_pose(pos, quat)
        step(2)
        if attempt == 0:
            report("hover")
            assert bool(scene._vert[0]), "vertical latch must fire at the hover"
            assert not bool(scene.threaded()[0]), \
                "the hover itself must NOT satisfy threaded() (peg not in the loop)"
            s2 = print_score("P2 pan vertical at the approach hover (not threaded)")
            assert s2 >= s1 - 1e-6 and s2 >= 0.29, f"P2 score {s2} (expect 0.30)"

        # -- P3: force-thread the loop over the knob and onto the peg (each retry
        #        releases at a slightly different seat depth)
        ok = thread_servo(seat_off=0.010 * attempt)
        if not ok:
            print(f"[solve] thread servo missed on attempt {attempt + 1}; retrying",
                  flush=True)
            report("thread-retry")
            continue
        if attempt == 0:
            report("threaded")
            assert bool(scene._thread[0]), "thread latch did not set"
            s3 = print_score("P3 peg through the loop (held by the servo wrench)")
            assert s3 >= s2 - 1e-6 and s3 >= 0.59, f"P3 score {s3} (expect 0.60)"

        # -- P4: hands-off release: the pan drops onto the peg and hangs
        settle_pan()
        if bool(scene.success()[0]):
            break
        print(f"[solve] release did not hang on attempt {attempt + 1}; retrying",
              flush=True)
        report("release-retry")
    else:
        report("hang-FAIL")
        print("SIM_GEN_SOLVE: FAIL (pan never hung on the hook)", flush=True)
        os._exit(1)

    report("hung")
    s4 = print_score("P4 pan hanging free on the hook (settled)")
    assert s4 >= 0.99, f"success must score 1.0, got {s4}"

    # ---------------- phase 5: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s5 = print_score("P5 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s5 >= s4 - 1e-6
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
    except BaseException:  # noqa: BLE001 - die fast; Kit teardown would hang until the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)

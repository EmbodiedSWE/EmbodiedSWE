"""Teleport solution for MugAlcoveScene (sim_gen task
`libero_kitchen_scene6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug_i291`)
— the task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, the only pose write on the yellow mug): one root-state write
   carries the yellow mug from its spawn across open floor to a STAGING pose in front
   of the doorway — upright on the floor, laterally aligned with the white mug, handle
   trailing, still fully OUTSIDE the alcove (local x = -0.15, free space). Nothing
   about this pose satisfies any placement condition; the write invalidates the
   stillness streak (pose jump), so it can never judge as success.
2. DOORWAY TRANSIT + PRECISION STOP (contact dynamics, no teleport): a horizontal
   external force at the mug's CoM (velocity-regulated bang-bang, small lateral
   P-correction toward the white mug's line, runtime frame-encode probe, stall
   escalation) pushes the mug ~24 cm ALONG THE FLOOR, through the doorway, under the
   awning, and stops it by readback in the goal window in front of the white mug. The
   mug slides on real floor friction the whole way; the white mug is never touched.
3. HANDS-OFF SETTLE: forces cleared; the mug brakes on friction and the scene's
   stillness streak (both mugs slow, no pose jump) earns success.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug_i291.solve --headless [--seed N]
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
import traceback

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)",
                                             flush=True), os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.mug_alcove")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    from isaaclab.utils.math import matrix_from_quat, quat_apply

    def y_loc() -> torch.Tensor:
        return scene._local(scene.yellow.data.root_pos_w)[0]

    def w_loc() -> torch.Tensor:
        return scene._local(scene.white.data.root_pos_w)[0]

    def report(tag: str) -> None:
        yl = y_loc()
        wl = w_loc()
        v = float(scene.yellow.data.root_lin_vel_w[0].norm())
        print(f"[solve] {tag:10s} | y_loc=({float(yl[0]):+.3f},{float(yl[1]):+.3f},"
              f"{float(yl[2]):.3f}) w_loc=({float(wl[0]):+.3f},{float(wl[1]):+.3f}) "
              f"|v|={v:.3f} placed={bool(scene._placed_now()[0])} "
              f"intact={bool(scene._white_intact()[0])} still={int(scene._still[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    last_score = [0.0]

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        assert s >= last_score[0] - 1e-6, f"score decreased at {tag}"
        last_score[0] = s
        return s

    def clear_force() -> None:
        scene.yellow.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                   env_ids=all_ids)

    # ---------------- phase 0: reset, settle, baseline --------------------------------------
    step(60)
    m_read = float(scene.yellow.root_physx_view.get_masses().reshape(-1)[0])
    assert abs(m_read - c.mug_mass) < 0.02, \
        f"authored mug mass not applied (read {m_read})"
    ap = (scene.alcove.data.root_pos_w - scene.env_origins)[0]
    aq = scene.alcove.data.root_quat_w[0]
    ayaw = 2.0 * math.atan2(float(aq[3]), float(aq[0]))
    yl0, wl0 = y_loc(), w_loc()
    print(f"[solve] layout readback (seed {args.seed}): alcove=({float(ap[0]):+.3f},"
          f"{float(ap[1]):+.3f}) yaw={math.degrees(ayaw):+.1f}deg "
          f"white_loc=({float(wl0[0]):+.3f},{float(wl0[1]):+.3f}) "
          f"yellow_loc=({float(yl0[0]):+.3f},{float(yl0[1]):+.3f}) "
          f"mug_mass={m_read:.3f}", flush=True)
    assert 0.17 < float(wl0[0]) < 0.22 and abs(float(wl0[1])) < 0.04, \
        "white mug not where reset should put it"
    assert float(yl0[0]) < -0.26, "yellow mug must spawn outside the approach latch"
    report("reset")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert bool(scene._white_intact()[0]), "white mug must be intact at reset"
    print_score("P0 reset+settle")
    R_ref = matrix_from_quat(scene.yellow.data.root_quat_w)[0].clone()

    # ---------------- phase 1: TRANSPORT (teleport across open floor only) ------------------
    # One pose write: yellow mug upright ON THE FLOOR at alcove-local (-0.15, y_white)
    # — in front of the doorway, aligned with the white mug's lane, handle trailing
    # (local -x) so nothing snags in the doorway. Entirely outside the alcove, in free
    # space; the goal window starts 21 cm deeper. The write leaves zero velocity and
    # the stillness streak invalidated (pose jump), so it can never judge as success.
    wl = w_loc()
    stage_l = torch.tensor([-0.15, float(wl[1]), c.body_h / 2 + 0.002], device=device)
    aq_n = scene.alcove.data.root_quat_w
    apos = scene.alcove.data.root_pos_w
    stage_w = apos[0] + quat_apply(aq_n, stage_l.unsqueeze(0))[0]
    yaw_stage = ayaw + math.pi  # handle (mug local +x) -> alcove local -x (trailing)
    q_stage = torch.tensor([math.cos(yaw_stage / 2), 0.0, 0.0,
                            math.sin(yaw_stage / 2)], device=device)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = stage_w.unsqueeze(0)
    st[:, 3:7] = q_stage.unsqueeze(0)
    scene.yellow.write_root_state_to_sim(st, all_ids)
    assert not bool(scene.success()[0]), \
        "freshly teleported staging must NOT judge as success"
    step(30)  # settle the 2 mm drop; approach latch fires on real frames
    assert bool(scene._approached[0]), "staging must fire the approach latch"
    assert not bool(scene._entered[0]), "staging is OUTSIDE the sill"
    report("staged")
    print_score("P1 transport to the staging pose in front of the doorway")

    # ---------------- phase 2: PUSH through the doorway (contact dynamics) ------------------
    # Velocity-regulated bang-bang horizontal force at the CoM along alcove local +x,
    # with a small lateral P-correction toward the white mug's lane; runtime
    # frame-encode probe (wrench frame-drag insurance) and stall escalation. The mug
    # slides on real floor friction through the doorway and under the awning; stop by
    # readback in the middle of the goal window, short of the white mug.
    x_hi = min(c.roof_len - c.body_r, float(wl[0]) - c.gap_min)
    x_target = 0.5 * (c.x_in_lo + x_hi)
    print(f"[solve] push: goal window x=[{c.x_in_lo:.3f},{x_hi:.3f}] "
          f"target={x_target:.3f}", flush=True)
    push_dir = quat_apply(aq_n, torch.tensor([1.0, 0.0, 0.0],
                                             device=device).unsqueeze(0))[0]
    lat_dir = quat_apply(aq_n, torch.tensor([0.0, 1.0, 0.0],
                                            device=device).unsqueeze(0))[0]
    R_now = lambda: matrix_from_quat(scene.yellow.data.root_quat_w)[0]  # noqa: E731
    mode = ["raw"]

    def encode(f_des: torch.Tensor) -> torch.Tensor:
        if mode[0] == "raw":
            return f_des
        return (R_ref @ R_now().T) @ f_des

    f_mag, v_des = 1.5, 0.07
    x0 = float(y_loc()[0])
    best, last_bump = x0, 0
    for i in range(2400):
        yl = y_loc()
        x_now = float(yl[0])
        if x_now >= x_target:
            break
        v_along = float((scene.yellow.data.root_lin_vel_w[0] * push_dir).sum())
        f_fwd = push_dir * (f_mag if v_along < v_des else 0.0)
        y_err = float(w_loc()[1]) - float(yl[1])
        f_lat = lat_dir * max(-0.4, min(0.4, 3.0 * y_err))
        f_w = encode(f_fwd + f_lat).view(1, 1, 3).expand(n, 1, 3)
        scene.yellow.set_external_force_and_torque(f_w.contiguous(), zero_wrench,
                                                   env_ids=all_ids, is_global=True)
        env.step(no_action)
        if x_now > best + 0.004:
            best, last_bump = x_now, i
        elif i - last_bump > 240:  # stalled: push harder
            f_mag = min(f_mag + 0.75, 4.0)
            last_bump = i
            print(f"[solve] push stalled at x={x_now:.3f}, raising force to "
                  f"{f_mag:.2f} N", flush=True)
        if i == 240 and best < x0 + 0.004 and mode[0] == "raw":
            mode[0] = "drag"  # frame-drag probe: no progress -> flip encoding
            print("[solve] push encode probe: no progress in raw mode, switching "
                  "to drag pre-encode", flush=True)
        if i % 240 == 0:
            assert bool(scene._upright(scene.yellow)[0]), "mug tipped during push"
    clear_force()
    yl = y_loc()
    assert float(yl[0]) >= c.x_in_lo, \
        f"push did not deliver the mug past x_in_lo (x={float(yl[0]):.3f})"
    assert bool(scene._upright(scene.yellow)[0]), "mug not upright after push"
    assert bool(scene._white_intact()[0]), "white mug disturbed by the push"
    report("pushed")
    print_score("P2 force-push through the doorway to the goal window")

    # ---------------- phase 3: hands-off settle to success ----------------------------------
    ok_settle = False
    for _ in range(40):  # up to 10 s
        step(30)
        if bool(scene.success()[0]):
            ok_settle = True
            break
    report("settled")
    if not ok_settle:
        print("SIM_GEN_SOLVE: FAIL (no success after settle)", flush=True)
        os._exit(1)
    print_score("P3 hands-off brake + settle to success")

    # ---------------- phase 4: persistence (>= 3 simulated seconds, no intervention) --------
    hold = True
    for _ in range(10):  # 10 x 42 steps = 420 substeps = 3.5 s at 120 Hz
        step(42)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.5 s")
    ok = hold and bool(scene.success()[0]) and s4 >= 1.0 - 1e-6
    if ok:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)

    code = 0 if ok else 1
    # Hard exit: Kit teardown hangs — watchdog then die.
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
    try:
        main()
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)

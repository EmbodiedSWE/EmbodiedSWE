"""Teleport solution for HookDenRetrievalScene (sim_gen task
`pick_and_lift_small_i55`) — the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TOOL STAGING (teleport = transport only): a single root-state write carries the
   ORANGE L-HOOK from wherever it spawned to the staging pose in the OPEN ground in
   front of the den mouth: corner at den-local (0.30, -s*lane_y), handle aligned
   with the den axis, rolled 180 deg about the handle when the cargo cube sits on
   the -y side (so the toe points AT the cube's side). s = sign of the cargo cube's
   den-local y. The pose is free space (1 mm hover), satisfies no rubric clause,
   and every interaction that matters happens after it.
2. INSERT (applied force + contact): a CoM force along den -x slides the hook flat
   along the ground through the mouth, hugging the side wall OPPOSITE the cube
   (lateral P-force + yaw servo torque about world z keep it in the lane), until
   the toe is DEEPER than the cube. The 16 mm bar clears the 60 mm roof; the tool
   never leaves the ground.
3. SHIFT (applied force + contact): a lateral CoM force walks the corner across the
   den floor until the toe midpoint sits behind the cube (pocket between handle
   edge and toe tip). This latches `engaged` (0.15) — a pure geometric consequence
   of the force-driven motion.
4. DRAG OUT (applied force + contact): a CoM force along den +x pulls the hook out;
   the toe's front face pushes the cube through the mouth. Inside the den the side
   wall backstops the cube against slipping past the toe tip. Force released once
   the cube is well clear; the cube coasts to rest on the ground in the open —
   `extracted` (0.35) is produced by sustained tool-mediated contact, never
   written. A retry loop (pull out, re-insert deeper/closer) covers slip-offs.
   FORCE-FRAME GUARD: some pods rotate an applied "global" wrench by the body's
   rotation since a reference. Every push PROBES progress along its intended
   direction each 45 steps and toggles the `encode_force` mode if the hook moved
   the wrong way (see scene.encode_force). The yaw-servo torque is about world z
   and both encodings agree on it for a body that only yaws after reference.
5. PLACE (teleport = transport only, gravity seats it): with the cube out in the
   open, a root-state write carries it through free air to 22 mm ABOVE its rest
   height on the pedestal top — on_ped_z_tol is 10 mm, so the write itself does
   NOT satisfy `on_pedestal`; the final 22 mm are free fall and the success
   clauses (on top, settled) are produced by contact. This is the pick-and-place
   any gripper would do once the cube is reachable.
6. IDENTITY: the RED decoy is never touched; success requires it NOT on the
   pedestal.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.pick_and_lift_small_i55.solve --headless [--seed N]
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
    from .scene import _qmul, _qx, encode_force
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scene import _qmul, _qx, encode_force

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.hook_den_retrieval")().build(num_envs=args.num_envs,
                                                        device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import quat_apply

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so
    # the task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)

    def den_q() -> torch.Tensor:
        return scene.den.data.root_quat_w

    def corner_den() -> torch.Tensor:
        return scene._den_local(scene.hook.data.root_pos_w)[0]

    def cargo_den() -> torch.Tensor:
        return scene._den_local(scene.cargo.data.root_pos_w)[0]

    def toe_den() -> torch.Tensor:
        return scene._den_local(scene.toe_mid_w())[0]

    def den_dir_w(v) -> torch.Tensor:
        t = torch.tensor(v, device=device, dtype=torch.float).expand(n, 3)
        return quat_apply(den_q(), t)

    def yaw_err() -> float:
        """Signed angle (rad) from den +x to the hook handle axis, xy-projected."""
        hx = quat_apply(scene.hook.data.root_quat_w, ex)[0]
        dx = quat_apply(den_q(), ex)[0]
        return float(torch.atan2(dx[0] * hx[1] - dx[1] * hx[0],
                                 dx[0] * hx[0] + dx[1] * hx[1]))

    def clear_wrench() -> None:
        scene.hook.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                 env_ids=all_ids)

    def report(tag: str) -> None:
        cd, gd, td = corner_den(), cargo_den(), toe_den()
        print(f"[solve] {tag:14s} | corner=({float(cd[0]):+.3f},{float(cd[1]):+.3f}) "
              f"toe=({float(td[0]):+.3f},{float(td[1]):+.3f}) "
              f"cargo=({float(gd[0]):+.3f},{float(gd[1]):+.3f},{float(gd[2]):+.3f}) "
              f"yaw_err={math.degrees(yaw_err()):+.1f}deg "
              f"eng={bool(scene._engaged[0])} ext={bool(scene._extracted[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    frame = {"mode": 0, "q_ref": None}

    def slide(dir_den, stop_fn, *, fmag0: float, vmax: float, max_steps: int,
              lat=None, tag: str = "") -> bool:
        """Force-slide the hook along den-local `dir_den` until `stop_fn`.

        Bang-bang CoM force with velocity regulation, world-z yaw-servo torque,
        optional lateral P-force `lat=(axis_den, err_fn)` (err = current - target,
        force pushes err to zero). Probes the force-frame convention every 45
        steps: if the corner moved AGAINST `dir_den`, toggles the encode mode; if
        it stalled, escalates the force. Always clears the wrench."""
        fmag = fmag0
        dvec = torch.tensor(dir_den, device=device, dtype=torch.float)
        frame["q_ref"] = scene.hook.data.root_quat_w.clone()

        def prog() -> float:
            cd = corner_den()
            return float(cd[0] * dvec[0] + cd[1] * dvec[1] + cd[2] * dvec[2])

        p0 = prog()
        done = False
        for i in range(max_steps):
            if stop_fn():
                done = True
                break
            dir_w = den_dir_w(dir_den)
            v_along = float((scene.hook.data.root_lin_vel_w[0] * dir_w[0]).sum())
            mag = fmag if v_along < 0.5 * vmax else (0.4 * fmag if v_along < vmax else 0.0)
            f_world = mag * dir_w
            if lat is not None:
                axis, err_fn = lat
                e = err_fn()
                if abs(e) > 0.002:
                    f_world = f_world + max(-0.35, min(0.35, -8.0 * e)) * den_dir_w(axis)
            tz = -(0.03 * yaw_err()
                   + 0.010 * float(scene.hook.data.root_ang_vel_w[0, 2]))
            tq = torch.zeros(n, 1, 3, device=device)
            tq[:, 0, 2] = max(-0.06, min(0.06, tz))
            f = encode_force(frame["mode"], frame["q_ref"],
                             scene.hook.data.root_quat_w, f_world).view(n, 1, 3)
            scene.hook.set_external_force_and_torque(f, tq, env_ids=all_ids,
                                                     is_global=True)
            env.step(no_action)
            if i % 45 == 44:
                p1 = prog()
                if p1 < p0 - 0.004:
                    frame["mode"] ^= 1
                    frame["q_ref"] = scene.hook.data.root_quat_w.clone()
                    print(f"[solve] {tag}: hook moved AGAINST the push "
                          f"({p0:+.3f} -> {p1:+.3f}); force-frame mode -> "
                          f"{frame['mode']}", flush=True)
                elif p1 < p0 + 0.002:
                    fmag = min(fmag + 0.4, 3.0)
                    print(f"[solve] {tag}: stalled at prog={p1:+.3f}; "
                          f"force -> {fmag:.1f} N", flush=True)
                p0 = p1
        clear_wrench()
        return done

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)
    dq = den_q()[0]
    dyaw = math.degrees(2.0 * math.atan2(float(dq[3]), float(dq[0])))
    gd0 = cargo_den()
    s = 1.0 if float(gd0[1]) > 0 else -1.0
    dp = (scene.den.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"den=({float(dp[0]):+.3f},{float(dp[1]):+.3f}) yaw={dyaw:+.1f}deg "
          f"cargo_den=({float(gd0[0]):+.3f},{float(gd0[1]):+.3f}) side={'+y' if s > 0 else '-y'}",
          flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert bool(scene.in_den(scene.cargo.data.root_pos_w)[0]), \
        "cargo cube must start inside the den"
    assert not bool(scene.hook_engaged()[0]), "hook must not start engaged"
    assert not bool(scene.cargo_extracted()[0]), "cargo must not start extracted"
    s0 = print_score("P0 reset+settle (cargo denned, hook in the open)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: stage the hook (transport, free space, in the open) ---------
    lane = -s * c.lane_y
    roll = math.pi if s < 0 else 0.0
    loc = torch.zeros(n, 3, device=device)
    loc[:, 0], loc[:, 1] = 0.30, lane
    loc[:, 2] = 0.018 if roll else 0.002  # 1 mm hover; rolled bars span z [-16, 0] mm
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.den.data.root_pos_w + quat_apply(den_q(), loc)
    st[:, 2] = loc[:, 2]
    roll_t = torch.full((n,), roll, device=device)
    st[:, 3:7] = _qmul(den_q(), _qx(roll_t))
    scene.hook.write_root_state_to_sim(st, all_ids)
    step(60)  # drop 1 mm, settle flat
    report("staged")
    cd = corner_den()
    assert abs(float(cd[0]) - 0.30) < 0.02 and abs(float(cd[1]) - lane) < 0.02, \
        f"staging drifted: corner=({float(cd[0]):+.3f},{float(cd[1]):+.3f})"
    td = toe_den()
    assert (float(td[1]) - float(cd[1])) * s > 0.015, "toe must point at the cube side"
    s1 = print_score("P1 hook staged at the mouth (no credit expected)")
    assert s1 <= 0.03, f"staging must not score, got {s1}"

    # ---------------- phase 2..4: insert / shift / drag, with retries ----------------------
    def stop_insert() -> bool:
        cd = corner_den()
        gx = float(cargo_den()[0])
        return float(cd[0]) <= max(gx - 0.040, -0.116)

    def do_attempt(k: int) -> bool:
        # -- insert along the lane opposite the cube --
        ok = slide((-1.0, 0.0, 0.0), stop_insert, fmag0=0.8, vmax=0.15,
                   max_steps=1400, lat=((0.0, 1.0, 0.0),
                                        lambda: float(corner_den()[1]) - lane),
                   tag=f"insert#{k}")
        report(f"insert#{k}")
        if not ok:
            return False

        # -- shift the toe behind the cube (pocket: cube nearer the handle) --
        x_hold = float(corner_den()[0])

        def shift_target() -> float:
            return float(cargo_den()[1]) - s * 0.029

        def stop_shift() -> bool:
            return s * (float(corner_den()[1]) - shift_target()) > -0.004

        ok = slide((0.0, s, 0.0), stop_shift, fmag0=0.6, vmax=0.08,
                   max_steps=900, lat=((1.0, 0.0, 0.0),
                                       lambda: float(corner_den()[0]) - x_hold),
                   tag=f"shift#{k}")
        report(f"shift#{k}")
        if not ok:
            return False
        y_hold = float(corner_den()[1])

        # -- drag out; the toe pushes the cube through the mouth --
        def stop_drag() -> bool:
            return float(cargo_den()[0]) > 0.20 or float(corner_den()[0]) > 0.26

        slide((1.0, 0.0, 0.0), stop_drag, fmag0=1.0, vmax=0.13,
              max_steps=1400, lat=((0.0, 1.0, 0.0),
                                   lambda: float(corner_den()[1]) - y_hold),
              tag=f"drag#{k}")
        step(120)  # hands-off: cube coasts to rest in the open
        report(f"drag#{k}")
        return bool(scene._extracted[0])

    extracted = False
    for k in range(4):
        if k > 0:
            # pull the hook back out to the staging depth with forces (no teleport
            # out from under the roof), then try again
            print(f"[solve] retry {k}: pulling the hook back out", flush=True)
            slide((1.0, 0.0, 0.0), lambda: float(corner_den()[0]) > 0.28,
                  fmag0=1.0, vmax=0.15, max_steps=1200,
                  lat=((0.0, 1.0, 0.0), lambda: float(corner_den()[1]) - lane),
                  tag=f"pullout#{k}")
            step(60)
        extracted = do_attempt(k)
        if extracted:
            break
    if not extracted:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (cargo never extracted)", flush=True)
        os._exit(1)
    assert bool(scene._engaged[0]), "engaged latch must have set during the rake"
    s2 = print_score("P2 hook engaged + cargo dragged out through the mouth")
    assert s2 >= c.w_engaged + c.w_extracted - 1e-6, \
        f"P2 score {s2} (expect {c.w_engaged + c.w_extracted})"

    # ---------------- phase 5: place the cargo on the pedestal (transport + gravity) -------
    ped_q = scene.pedestal.data.root_quat_w
    off = torch.zeros(n, 3, device=device)
    off[:, 2] = c.ped_size[2] / 2 + c.cube_size / 2 + 0.022  # 22 mm free fall
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.pedestal.data.root_pos_w + quat_apply(ped_q, off)
    st[:, 3:7] = ped_q  # square to the pedestal top
    scene.cargo.write_root_state_to_sim(st, all_ids)
    env.step(no_action)
    assert not bool(scene.on_pedestal(scene.cargo)[0]), \
        "the write itself must not satisfy on_pedestal (hover is outside z tol)"
    step(150)  # fall + settle, hands-off
    report("placed")
    assert bool(scene.on_pedestal(scene.cargo)[0]), "cargo must rest on the pedestal"
    assert not bool(scene.on_pedestal(scene.decoy)[0]), "decoy crept onto the pedestal?"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after placement)", flush=True)
        os._exit(1)
    s3 = print_score("P3 cargo placed on the pedestal")
    assert s3 >= 1.0 - 1e-6, f"success must score 1.0, got {s3}"

    # ---------------- phase 6: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
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
    except Exception:  # noqa: BLE001 - fail FAST; a hung Kit burns the forge slot
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)

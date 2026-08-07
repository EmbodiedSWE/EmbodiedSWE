"""Teleport solution for SlabEaselScene (sim_gen task
`libero_kitchen_scene9_put_the_frying_pan_on_top_of_the_cabinet_i53`) — the task's
legitimacy certificate.

Teleportation handles TRANSPORT ONLY, each write ending in FREE SPACE:
  - each colored slab is carried to a hover pose 4 mm above the work plane at its bay's
    STAGING point (lying flat, long edge parallel to the panel, foot edge ~8 mm in
    front of the panel foot) and released — the put-down a hand performs after sliding
    the slab across the deck;
  - if another loose slab happens to obstruct a staging spot, it is first carried
    (same flat hover-release) to a free parking spot on the deck's front strip —
    the shove-aside a hand performs. The blank slab is never stood up.
Everything load-bearing is DRIVEN, not bypassed:
  - ERECTION is pure contact dynamics: a torque governor applies a bounded torque about
    the bay's hinge axis (the rack's -y, read from the randomized studio yaw) —
    tau = clamp(12 * (0.9 - omega), +/-3.5 N*m), the wrench a fingertip lifting the
    slab's front edge through a rail gap applies. The slab pivots on its own grounded
    foot edge, sweeping the 30-70 degree rising-tilt band under the rubric's account.
    At ~81 degrees the torque is CUT; the slab's momentum carries it over its tip-over
    balance (~83 degrees) and gravity lays it onto the backrest panel, where it settles
    into a real ladder lean held by friction and the panel's normal force. Nothing is
    ever written upright: every degree of tilt is produced by the applied wrench and
    gravity through contact.
  - The external-wrench frame-drag quirk is pod-dependent, so the torque encoding is
    PROBED at runtime: if the slab does not start pivoting, the encoding (raw world
    torque vs pre-encoded with R_ref * R_now^T, and sign) is toggled and the slab is
    re-staged.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.4 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene9_put_the_frying_pan_on_top_of_the_cabinet_i53.solve --headless [--seed N]
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

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

NAMES = ("RED", "GREEN", "BLUE")
# torque-encoding candidates: (mode, sign); mode 0 = raw world torque, mode 1 =
# pre-encoded with R_ref * R_now^T (the documented frame-drag mitigation)
CANDS = ((0, 1.0), (1, 1.0), (0, -1.0), (1, -1.0))


def main() -> None:
    from isaaclab.utils.math import matrix_from_quat, quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.slab_easel")().build(num_envs=args.num_envs, device=device)
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

    def rack_to_world(lx: float, ly: float, lz: float) -> torch.Tensor:
        off = torch.tensor([[lx, ly, lz]], device=device)
        return (scene.studio.data.root_pos_w[0:1]
                + quat_apply(scene.studio.data.root_quat_w[0:1], off))[0]

    def hinge_axis_w() -> torch.Tensor:
        """World unit vector of the erection axis (-y of the rack): positive rotation
        about it hinges a staged slab's front edge up and back toward its panel."""
        off = torch.tensor([[0.0, -1.0, 0.0]], device=device)
        return quat_apply(scene.studio.data.root_quat_w[0:1], off)[0]

    def clear_wrench(k: int) -> None:
        scene.slabs[k].set_external_force_and_torque(
            zero_wrench, zero_wrench, env_ids=all_ids)

    def apply_torque(k: int, tau_w: torch.Tensor, mode: int, sign: float) -> None:
        if mode == 1:
            r_now = matrix_from_quat(scene.slabs[k].data.root_quat_w[0:1])[0]
            tau_w = ref_rot[k] @ r_now.T @ tau_w
        tw = (sign * tau_w).view(1, 1, 3).expand(n, 1, 3).contiguous()
        scene.slabs[k].set_external_force_and_torque(
            zero_wrench, tw, env_ids=all_ids, is_global=True)

    def tilt_deg(k: int) -> float:
        return math.degrees(float(scene.tilts()[0, k]))

    def rack_xy(k: int) -> tuple[float, float]:
        loc = scene.rack_local(scene.slabs[k].data.root_pos_w)[0]
        return float(loc[0]), float(loc[1])

    def teleport_flat(k: int, lx: float, ly: float, lz: float) -> None:
        """TRANSPORT-only write: slab k to a flat hover pose (rack frame), aligned
        with the rack, zero velocity — then it free-falls the last few mm."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = rack_to_world(lx, ly, lz)
        st[:, 3:7] = scene.studio.data.root_quat_w[0:1]
        scene.slabs[k].write_root_state_to_sim(st, all_ids)

    def stage(k: int) -> None:
        teleport_flat(k, c.x_bp + c.stage_dx, c.bay_ys[k],
                      c.z_work + c.slab_size[2] / 2 + 0.004)
        step(50)

    PARK = ((0.33, -0.45), (0.33, 0.0), (0.33, 0.45))  # deck front strip (rack frame)

    def clear_obstructors(k: int) -> None:
        """If any OTHER loose slab lies within 0.30 m of bay k's staging point, carry
        it (flat hover-release) to the parking spot farthest from everything."""
        sx, sy = c.x_bp + c.stage_dx, c.bay_ys[k]
        for j in range(4):
            if j == k or (j < 3 and bool(scene.seated_now(j)[0])):
                continue
            jx, jy = rack_xy(j)
            if math.hypot(jx - sx, jy - sy) >= 0.30:
                continue
            best, best_d = PARK[0], -1.0
            for px, py in PARK:
                dmin = min(math.hypot(px - qx, py - qy)
                           for i in range(4) if i != j
                           for qx, qy in [rack_xy(i)])
                if dmin > best_d:
                    best, best_d = (px, py), dmin
            print(f"[solve] parking slab {j} (obstructs bay {NAMES[k]}) at "
                  f"rack=({best[0]:+.2f},{best[1]:+.2f})", flush=True)
            teleport_flat(j, best[0], best[1], c.slab_size[2] / 2 + 0.004)
            step(40)

    def erect_attempt(k: int, cut_deg: float, mode: int, sign: float) -> str:
        """Drive slab k's hinge with the torque governor until the tilt crosses
        `cut_deg`, then cut. Returns 'cut', 'stall' (probe: wrong encoding), or
        'timeout'."""
        axis = hinge_axis_w()
        for i in range(700):
            t = tilt_deg(k)
            if t >= cut_deg:
                clear_wrench(k)
                return "cut"
            w = float(torch.dot(scene.slabs[k].data.root_ang_vel_w[0], axis))
            tau = max(-3.5, min(3.5, 12.0 * (0.9 - w)))
            apply_torque(k, tau * axis, mode, sign)
            env.step(no_action)
            if i == 40 and tilt_deg(k) < 6.0:
                clear_wrench(k)
                return "stall"
        clear_wrench(k)
        return "timeout"

    def report(tag: str) -> None:
        acc = scene.erect_acc[0]
        seat = [bool(scene.seated_now(k)[0]) for k in range(3)]
        tl = [f"{tilt_deg(k):.0f}" for k in range(4)]
        print(f"[solve] {tag:12s} | tilt(deg)={'/'.join(tl)} "
              f"seated={seat} staged={scene.staged_l[0].tolist()} "
              f"acc(deg)={[round(math.degrees(float(a)), 1) for a in acc]} "
              f"seated_l={scene.seated_l[0].tolist()} "
              f"blank_flat={bool(scene.distractor_flat()[0])} "
              f"still={int(scene.still_cnt[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline, layout readback --------------------
    step(60)
    ref_rot = [matrix_from_quat(scene.slabs[k].data.root_quat_w[0:1])[0].clone()
               for k in range(4)]  # reset-pose reference for mode-1 torque encoding
    sp = (scene.studio.data.root_pos_w - scene.env_origins)[0]
    syaw = math.degrees(float(scene._yaw_of(scene.studio)[0]))
    locs = " ".join(f"{NAMES[k] if k < 3 else 'BLANK'}=({rack_xy(k)[0]:+.3f},"
                    f"{rack_xy(k)[1]:+.3f})" for k in range(4))
    print(f"[solve] layout readback (seed {args.seed}): studio=({float(sp[0]):+.3f},"
          f"{float(sp[1]):+.3f}) yaw={syaw:+.1f}deg slabs[rack]: {locs}", flush=True)
    report("reset")
    prev_s = print_score("P0 reset+settle")

    # ---------------- phases 1-3: stage + ERECT each colored slab --------------------------
    cand = 0  # working torque encoding, shared across slabs once discovered
    for k in range(3):
        cut_deg = 81.0
        done = False
        for attempt in range(6):
            clear_obstructors(k)
            stage(k)
            if not bool(scene.staged_now(k)[0]):
                print(f"[solve] {NAMES[k]}: staging did not latch (attempt "
                      f"{attempt}) — re-staging", flush=True)
                continue
            mode, sign = CANDS[cand]
            res = erect_attempt(k, cut_deg, mode, sign)
            if res == "stall":
                cand = (cand + 1) % len(CANDS)
                print(f"[solve] {NAMES[k]}: no pivot under encoding (mode={mode},"
                      f"sign={sign:+.0f}) — switching to {CANDS[cand]}", flush=True)
                continue
            step(150)  # tip over the balance point + lay onto the panel + settle
            t_now = tilt_deg(k)
            seated = bool(scene.seated_now(k)[0])
            earned = bool(scene.erect_ok()[0, k])
            print(f"[solve] {NAMES[k]}: attempt {attempt} {res} cut={cut_deg:.1f} "
                  f"-> tilt={t_now:.1f}deg seated={seated} "
                  f"acc={math.degrees(float(scene.erect_acc[0, k])):.1f}deg",
                  flush=True)
            if seated and earned:
                done = True
                break
            if t_now < 30.0:  # fell back flat: cut was too early — push further next time
                cut_deg = min(cut_deg + 1.5, 86.0)
        if not done:
            report("FAIL-state")
            print(f"SIM_GEN_SOLVE: FAIL ({NAMES[k]} slab was not erected)", flush=True)
            os._exit(1)
        report(f"{NAMES[k]}-seated")
        s = print_score(f"P{k + 1} {NAMES[k]} slab staged + hinged up onto its panel")
        assert s >= prev_s - 1e-6, "score decreased across an erection phase"
        prev_s = s

    # ---------------- phase 4: persistence (>= 3.4 simulated seconds, hands-off) -----------
    step(60)  # let the stillness counter arm
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success before the persistence hold)", flush=True)
        os._exit(1)
    hold = True
    for _ in range(10):  # 10 x 41 = 410 substeps = 3.4 s at 120 Hz
        step(41)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.4 s")
    ok = hold and bool(scene.success()[0]) and s4 >= prev_s - 1e-6
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
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 — die loudly, don't wait for the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL (exception: {exc!r})", flush=True)
        os._exit(2)

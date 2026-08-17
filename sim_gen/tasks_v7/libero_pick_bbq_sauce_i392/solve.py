"""Teleport solution for TareLiftScene (sim_gen task `libero_pick_bbq_sauce_i392`) —
the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. UNLOAD the ballast (teleport = transport only, then gravity): each ballast can that
   rides the basket is carried through free air — up through the open lane between the
   car's roof strips — to a zero-velocity hover ABOVE the open-topped discard bin and
   released. It FALLS the last ~50 mm into the bin (restitution 0). The teleport ends
   in open space above the bin; the arrival is pure ballistics. A can that lands badly
   (perched on a neighbour, outside the bin band) is re-carried and re-dropped.
2. RISE (spring + gravity, hands-off): as cans leave, the preloaded prismatic drive
   raises the car. With only basket + bottle aboard the preload (14.1 N vs the 11.3 N
   pass load) pins the car at its TOP STOP, the car floor 3 mm proud of the deck. The
   solve applies NOTHING here — the rise is the scene's own force balance, and the
   `risen` latch is earned by physics.
3. EGRESS (applied force + contact): a velocity-regulated horizontal force at the
   basket CoM — the wrench of two closed fingers pressing the basket's back wall —
   slides the basket out the car's open front, over the sill, onto the deck, then
   steers it to the goal-pad centre and releases. The bottle rides seated in its
   pocket on real contact the whole way; it is never grasped or teleported. Pod
   force-frame quirk: the desired world force is pre-encoded per `encode_force` and
   the mode is PROBED from actual progress.
4. ORDER: cans-before-egress is forced by physics (any can aboard buries the basket's
   leading face >= 19 mm below the sill and the roof strips deny the lift bypass); the
   solve simply follows the only path that exists.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u solve.py --headless [--seed N]
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

import os
import threading

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from .scene import _qapply, encode_force
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scene import _qapply, encode_force

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_WD = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_WD.daemon = True
_WD.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # BUILD-time seed: the dock pose is drawn in assets() (RNGs are seeded before).
    env = ENVS.get("simgen.tare_lift")().build(
        num_envs=args.num_envs, device=device, seed=args.seed)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build too (the EnvCfg.build reseed trap) for the reset-level draws.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def clear_forces(body=None) -> None:
        for b in ((body,) if body is not None else
                  (scene.basket, scene.bottle, *scene.cans)):
            b.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def qnow() -> float:
        return float(scene.car_q()[0])

    def dock_pt(loc) -> torch.Tensor:
        loc_t = torch.tensor(loc, device=device, dtype=torch.float).expand(n, 3)
        return scene.dock.data.root_pos_w + _qapply(scene.dock.data.root_quat_w, loc_t)

    def aboard_cans() -> list[int]:
        out = []
        for i, b in enumerate(scene.cans):
            loc = scene._basket_local(b.data.root_pos_w)[0]
            if (abs(float(loc[0])) < 0.090 and abs(float(loc[1])) < 0.090
                    and -0.02 < float(loc[2]) < 0.12):
                out.append(i)
        return out

    def free_parks() -> list:
        occ = [tuple(float(v) for v in scene._dock_local(b.data.root_pos_w)[0])
               for b in scene.cans]
        free = []
        for px, py in c.bin_parks:
            taken = any(abs(ox - px) < 0.045 and abs(oy - py) < 0.045
                        and oz > c.deck_top for ox, oy, oz in occ)
            if not taken:
                free.append((px, py))
        return free

    def report(tag: str) -> None:
        bl = scene._dock_local(scene.basket.data.root_pos_w)[0]
        print(f"[solve] {tag:12s} | q={qnow():+.4f} "
              f"bask_loc=({float(bl[0]):+.3f},{float(bl[1]):+.3f},{float(bl[2]):+.3f}) "
              f"binned={int(scene.cans_binned_count()[0])} "
              f"seat={bool(scene.bottle_seated()[0])} "
              f"pad={bool(scene.basket_on_pad()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # Force-frame reference orientation for `encode_force` (readback at reset).
    q_ref = scene.basket.data.root_quat_w.clone()
    mode = 0  # probed from actual progress

    # ---------------- basket servo: press the basket toward a world xy target -------------
    def servo_basket(tgt_fn, tol: float, tag: str, *, speed: float = 0.08,
                     timeout: int = 2400) -> bool:
        """Velocity-regulated horizontal CoM force toward the (moving-frame) target.
        Gentle by design: the basket slides out under the car's roof strips, so speed
        stays low and the stiction floor escalates only when genuinely stalled."""
        nonlocal mode
        floor_f = 1.0
        win_i, hold = 0, 0
        err = tgt_fn() - scene.basket.data.root_pos_w
        err[:, 2] = 0.0
        win_d = float(err.norm(dim=-1)[0])
        for i in range(timeout):
            err = tgt_fn() - scene.basket.data.root_pos_w
            err[:, 2] = 0.0
            d = float(err.norm(dim=-1)[0])
            if d <= tol:
                hold += 1
                if hold >= 20:
                    clear_forces(scene.basket)
                    return True
            else:
                hold = 0
            dirn = err / max(d, 1e-6)
            v = scene.basket.data.root_lin_vel_w.clone()
            v[:, 2] = 0.0
            v_along = float((v * dirn).sum(dim=-1)[0])
            v_des = speed if d > 0.040 else 0.035
            f_mag = 6.0 * (v_des - v_along)
            if abs(v_along) < 0.02 and f_mag < floor_f:
                f_mag = floor_f
            f_mag = max(-8.0, min(8.0, f_mag))
            # damp cross-track drift so the basket tracks the lane, not an arc
            v_perp = v - dirn * (v * dirn).sum(dim=-1, keepdim=True)
            f_world = dirn * f_mag - 3.0 * v_perp
            f_arg = encode_force(mode, q_ref, scene.basket.data.root_quat_w, f_world)
            scene.basket.set_external_force_and_torque(
                f_arg.view(n, 1, 3), zero_wrench, env_ids=all_ids, is_global=True)
            env.step(no_action)
            if i - win_i >= 45:
                e2 = tgt_fn() - scene.basket.data.root_pos_w
                e2[:, 2] = 0.0
                e = float(e2.norm(dim=-1)[0])
                if e > win_d + 0.004:
                    mode = 1 - mode
                    print(f"[solve] {tag}: moving away (err {win_d:.3f} -> {e:.3f}); "
                          f"force-frame mode -> {mode}", flush=True)
                elif e > win_d - 0.002 and e > tol:
                    floor_f = min(floor_f + 0.5, 4.5)
                    print(f"[solve] {tag}: stalled at err {e:.3f}; "
                          f"stiction floor -> {floor_f:.1f} N", flush=True)
                win_i, win_d = i, e
        clear_forces(scene.basket)
        print(f"[solve] {tag}: timed out at err {win_d:.3f}", flush=True)
        return win_d <= tol

    # ---------------- phase 0: reset, settle, layout readback -----------------------------
    step(120)
    dx, dy, dyaw = scene._dock_build
    aboard0 = aboard_cans()
    pl = scene._dock_local(scene.pad.data.root_pos_w)[0]
    print(f"[solve] layout readback (seed {args.seed}): dock=({dx:+.3f},{dy:+.3f},"
          f"yaw={dyaw:+.3f}rad) q0={qnow():+.4f} n_aboard={len(aboard0)} "
          f"aboard={aboard0} pad_loc=({float(pl[0]):+.3f},{float(pl[1]):+.3f})",
          flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert len(aboard0) >= 2, f"expected >= 2 cans aboard, saw {len(aboard0)}"
    assert qnow() <= -(c.travel - 0.004), \
        f"loaded car must sit at the bottom stop, q={qnow():+.4f}"
    assert bool(scene.bottle_seated()[0]), "bottle must start seated in the pocket"
    s0 = print_score("P0 reset+settle (car buried, gate closed)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: carry each ballast can to the discard bin ------------------
    prev = s0
    while True:
        aboard = aboard_cans()
        if not aboard:
            break
        i = aboard[0]
        binned = False
        for attempt in range(4):
            parks = free_parks()
            assert parks, "no free bin park"
            px, py = parks[0]
            st = torch.zeros(n, 13, device=device)
            st[:, 0:3] = dock_pt((px, py, c.deck_top + c.can_h / 2 + 0.050))
            st[:, 3:7] = scene.dock.data.root_quat_w  # yaw-aligned with the bin
            scene.cans[i].write_root_state_to_sim(st, all_ids)
            step(120)  # free fall into the bin + spring response, hands-off
            binned = bool(scene.in_bin(scene.cans[i])[0])
            if binned:
                break
            print(f"[solve] can{i} drop {attempt}: landed outside the bin band; "
                  f"re-dropping", flush=True)
        report(f"can{i}->bin")
        if not binned:
            print("SIM_GEN_SOLVE: FAIL (a ballast can never landed in the bin)",
                  flush=True)
            os._exit(1)
        s = print_score(f"P1 ballast can {i} dropped into the discard bin")
        assert s >= prev - 1e-6, "score decreased across a can removal"
        prev = s
    assert prev >= (c.w_first + c.w_clear) - 1e-6, \
        f"all cans binned should score >= 0.35, got {prev}"

    # ---------------- phase 2: hands-off — the spring raises the car ----------------------
    risen = False
    for _ in range(600):
        env.step(no_action)
        if qnow() >= -c.risen_q_tol and bool(scene.settled()[0]):
            risen = True
            break
    step(120)  # full settle at the top stop (bottle re-seats after any hop)
    report("risen")
    if not (risen and qnow() >= -c.risen_q_tol):
        print("SIM_GEN_SOLVE: FAIL (car never rose to its top stop)", flush=True)
        os._exit(1)
    if not bool(scene.bottle_seated()[0]):
        print("SIM_GEN_SOLVE: FAIL (bottle unseated during the rise)", flush=True)
        os._exit(1)
    s2 = print_score("P2 spring raised the unloaded car to the top stop")
    assert s2 >= (c.w_first + c.w_clear + c.w_rise) - 1e-6, \
        f"risen should score >= 0.50, got {s2}"

    # ---------------- phase 3: push the basket out of the pit onto the pad ----------------
    # waypoint A: straight out the doorway, fully past the sill (dock x axis)
    ok = servo_basket(lambda: dock_pt((0.245, 0.0, 0.0)), 0.020, "egress")
    step(60)
    report("egress")
    bx = float(scene._dock_local(scene.basket.data.root_pos_w)[0][0])
    if not ok or bx <= c.egress_x_min:
        print("SIM_GEN_SOLVE: FAIL (basket never cleared the pit)", flush=True)
        os._exit(1)
    s3 = print_score("P3a basket slid over the sill, fully out of the pit")
    assert s3 >= (c.w_first + c.w_clear + c.w_rise + c.w_egress) - 1e-6, \
        f"egressed should score >= 0.70, got {s3}"

    # waypoint B: steer to the goal-pad centre, release, settle
    placed = False
    for push in range(3):
        ok = servo_basket(
            lambda: scene.pad.data.root_pos_w.clone(), 0.012, f"to-pad-{push}")
        clear_forces()
        step(150)  # hands-off settle on the pad
        placed = bool(scene.basket_on_pad()[0]) and bool(scene.bottle_seated()[0])
        if placed:
            break
        print(f"[solve] to-pad {push}: not centred yet; re-nudging", flush=True)
    report("on-pad")
    if not placed or not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (final state not reached)", flush=True)
        os._exit(1)
    s4 = print_score("P4 basket settled on the goal pad; goal state assembled")
    assert s4 >= s3 - 1e-6, "score decreased across the placement"
    assert s4 >= 1.0 - 1e-6, f"success must score 1.0, got {s4}"

    # ---------------- phase 5: persistence (>= 3 simulated seconds, hands-off) ------------
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
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 - die loudly, never idle until the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)

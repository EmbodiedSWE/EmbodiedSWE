"""Teleport solution for ShuttleVaultScene (sim_gen task
`libero_pick_chocolate_pudding_i201`) — the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. HOME the shuttle (applied force): the shuttle starts anywhere on its rail. A
   velocity-regulated horizontal force along the rail axis — the exact wrench of a hand
   pulling the external T-knob — drives it against the HOME stop (station A, under the
   loading window). The rail is frictionless (the joint pair is collision-filtered);
   body damping is the brake, and the authored joint limit is the hard stop, so a light
   press-and-hold parks it exactly. Pod force-frame quirk: some pods rotate an applied
   wrench by the body's rotation since a reference, so the desired world force is
   pre-encoded per `encode_force` and the mode is PROBED from actual rail progress.
2. LOAD (teleport = transport only, then gravity): the pudding is carried through free
   air to a zero-velocity hover ABOVE the roof window and released. It FALLS through
   the 72 mm window into the 78 mm pocket interior (restitution 0). The teleport ends
   in open space outside the vault; entry itself is pure ballistics through the
   aperture. If a drop perches on a pocket wall, a small horizontal nudge force (a
   fingertip through the window) topples it in; a drop that stays outside the vault is
   simply re-carried and re-dropped.
3. CONVEY + DEPOSIT (applied force + contact + gravity): the same knob-wrench drives
   the shuttle to station B. The pocket walls DRAG the pudding, sliding it across the
   gallery floor, until the floor disappears — at B the whole pocket interior is over
   the 80 mm drop hole — and the pudding falls into the sealed chamber by gravity.
   Nothing is ever teleported into (or out of, or inside) the vault.
4. HOME again (applied force): the shuttle is pulled back against the home stop —
   the declared end state (loader returned).
5. ORDER: the task declares no order beyond physical precedence (the chamber can only
   be fed through the pocket; the pocket can only be fed at home).

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
    # BUILD-time seed: the vault pose is drawn in assets() (RNGs are seeded before).
    env = ENVS.get("simgen.shuttle_vault")().build(
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
        for b in ((body,) if body is not None else (scene.shuttle, scene.pudding)):
            b.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def qnow() -> float:
        return float(scene.shuttle_q()[0])

    def housing_pt(loc) -> torch.Tensor:
        loc_t = torch.tensor(loc, device=device, dtype=torch.float).expand(n, 3)
        return scene.housing.data.root_pos_w + _qapply(scene.housing.data.root_quat_w,
                                                       loc_t)

    def rail_axis() -> torch.Tensor:
        ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
        return _qapply(scene.housing.data.root_quat_w, ex)

    def report(tag: str) -> None:
        pl = scene._housing_local(scene.pudding.data.root_pos_w)[0]
        print(f"[solve] {tag:12s} | q={qnow():+.4f} home={bool(scene.home()[0])} "
              f"pud_loc=({float(pl[0]):+.3f},{float(pl[1]):+.3f},{float(pl[2]):+.3f}) "
              f"gal={bool(scene.in_gallery(scene.pudding)[0])} "
              f"pock={bool(scene.in_pocket(scene.pudding)[0])} "
              f"cham={bool(scene.in_chamber(scene.pudding)[0])} "
              f"clear={bool(scene.distractors_clear()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # Force-frame reference orientation for `encode_force` (readback at reset).
    q_ref = scene.shuttle.data.root_quat_w.clone()
    mode = 0  # probed from actual rail progress

    # ---------------- knob servo: press the shuttle toward a rail stop --------------------
    def servo_to(q_tgt: float, tol: float, tag: str, *,
                 watch_deposit: bool = False, timeout: int = 2400) -> bool:
        """Velocity-regulated rail force toward `q_tgt` (the joint limits are the hard
        stops). Returns True at |q - q_tgt| <= tol (or on a live deposit when
        `watch_deposit`). Probes the force-frame mode from actual progress."""
        nonlocal mode
        floor_f = 0.4  # stiction floor (rail is frictionless; damping-only brake)
        win_i, win_err = 0, abs(q_tgt - qnow())
        hold = 0
        for i in range(timeout):
            if watch_deposit and bool(scene.in_chamber(scene.pudding)[0]):
                clear_forces(scene.shuttle)
                print(f"[solve] {tag}: pudding dropped into the chamber at q={qnow():+.4f}",
                      flush=True)
                return True
            err = q_tgt - qnow()
            if abs(err) <= tol:
                hold += 1
                if hold >= 30:  # pressed against the stop, parked
                    clear_forces(scene.shuttle)
                    return True
            else:
                hold = 0
            sgn = 1.0 if err > 0 else -1.0
            xhat = rail_axis()
            v_along = float((scene.shuttle.data.root_lin_vel_w * xhat).sum(dim=-1)[0])
            v_des = sgn * (0.10 if abs(err) > 0.030 else 0.04)
            f_along = 4.0 * (v_des - v_along)
            if abs(v_along) < 0.02 and f_along * sgn < floor_f:
                f_along = sgn * floor_f
            f_along = max(-6.0, min(6.0, f_along))
            f_world = xhat * f_along
            f_arg = encode_force(mode, q_ref, scene.shuttle.data.root_quat_w, f_world)
            scene.shuttle.set_external_force_and_torque(
                f_arg.view(n, 1, 3), zero_wrench, env_ids=all_ids, is_global=True)
            env.step(no_action)
            if i - win_i >= 45:
                e = abs(q_tgt - qnow())
                if e > win_err + 0.004:
                    mode = 1 - mode
                    print(f"[solve] {tag}: moving away (err {win_err:.3f} -> {e:.3f}); "
                          f"force-frame mode -> {mode}", flush=True)
                elif e > win_err - 0.002 and e > tol:
                    floor_f = min(floor_f + 0.3, 2.5)
                    print(f"[solve] {tag}: stalled at err {e:.3f}; "
                          f"stiction floor -> {floor_f:.1f} N", flush=True)
                win_i, win_err = i, e
        clear_forces(scene.shuttle)
        print(f"[solve] {tag}: timed out at q={qnow():+.4f}", flush=True)
        return abs(q_tgt - qnow()) <= tol

    # ---------------- phase 0: reset, settle, layout readback -----------------------------
    step(120)
    vx, vy, vyaw = scene._vault_build
    pl = (scene.pudding.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): vault=({vx:+.3f},{vy:+.3f},"
          f"yaw={vyaw:+.3f}rad) q0={qnow():+.4f} "
          f"pudding=({float(pl[0]):+.3f},{float(pl[1]):+.3f})", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert not bool(scene.in_vault(scene.pudding)[0]), "pudding must start outside"
    assert bool(scene.distractors_clear()[0]), "distractors must start outside"
    s0 = print_score("P0 reset+settle (vault sealed, everything outside)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: pull the shuttle HOME (knob wrench) ------------------------
    ok = servo_to(0.004, 0.008, "home-1")
    step(60)
    report("home-1")
    if not ok or not bool(scene.home()[0]):
        print("SIM_GEN_SOLVE: FAIL (could not home the shuttle)", flush=True)
        os._exit(1)
    s1 = print_score("P1 shuttle pulled home under the window")
    assert s1 >= s0 - 1e-6, "score decreased across homing"

    # ---------------- phase 2: drop the pudding through the window ------------------------
    drop = housing_pt((c.x_a, 0.0, c.roof_z1 + 0.065))
    loaded = False
    for attempt in range(4):
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = drop
        st[:, 3:7] = scene.housing.data.root_quat_w  # yaw-aligned with the window
        scene.pudding.write_root_state_to_sim(st, all_ids)
        step(150)  # free fall through the window + settle in the pocket, hands-off
        loaded = bool((scene.in_gallery(scene.pudding)
                       & scene.in_pocket(scene.pudding))[0])
        if loaded:
            break
        if bool(scene.in_gallery(scene.pudding)[0]):
            # perched inside near the window mouth: fingertip nudge toward the pocket
            print(f"[solve] drop {attempt}: perched in the gallery; nudging", flush=True)
            tgt = housing_pt((c.x_a, 0.0, c.rail_z))
            for _ in range(120):
                d = (tgt - scene.pudding.data.root_pos_w)
                d[:, 2] = 0.0
                nrm = float(d.norm(dim=-1)[0])
                if nrm < 1e-4:
                    break
                f = (d / max(nrm, 1e-6)) * 1.5
                scene.pudding.set_external_force_and_torque(
                    f.view(n, 1, 3), zero_wrench, env_ids=all_ids, is_global=True)
                env.step(no_action)
                if bool(scene.in_pocket(scene.pudding)[0]):
                    break
            clear_forces(scene.pudding)
            step(60)
            loaded = bool((scene.in_gallery(scene.pudding)
                           & scene.in_pocket(scene.pudding))[0])
            if loaded:
                break
        else:
            print(f"[solve] drop {attempt}: missed the window "
                  f"(outside the vault); re-dropping", flush=True)
    report("load")
    if not loaded:
        print("SIM_GEN_SOLVE: FAIL (pudding never loaded into the pocket)", flush=True)
        os._exit(1)
    assert not bool(scene.success()[0]), "cannot be success with the pudding loaded"
    s2 = print_score("P2 pudding fell through the window into the pocket")
    assert s2 >= s1 - 1e-6 and s2 >= c.w_load - 1e-6, f"P2 score {s2} (expect loaded=0.20)"

    # ---------------- phase 3: convey to B; gravity deposits through the hole -------------
    deposited = False
    for push in range(3):
        servo_to(c.travel - 0.004, 0.008, f"convey-{push}", watch_deposit=True)
        step(120)  # let the drop finish + settle, hands-off
        deposited = bool(scene.in_chamber(scene.pudding)[0])
        if deposited:
            break
        # straggler: rock the shuttle a little around B so the pudding finds the hole
        print(f"[solve] convey {push}: pudding not down yet "
              f"(q={qnow():+.4f}); rocking", flush=True)
        servo_to(c.travel - 0.030, 0.010, f"rock-{push}a")
        step(30)
    report("deposit")
    if not deposited:
        print("SIM_GEN_SOLVE: FAIL (pudding never reached the chamber)", flush=True)
        os._exit(1)
    s3 = print_score("P3 pocket conveyed the pudding over the hole; gravity deposited it")
    assert s3 >= s2 - 1e-6, "score decreased across the convey"
    assert s3 >= (c.w_load + c.w_conv + c.w_dep) - 1e-6, f"P3 score {s3} (expect 0.65)"

    # ---------------- phase 4: pull the shuttle home again --------------------------------
    ok = servo_to(0.004, 0.008, "home-2")
    step(120)  # full settle, hands-off
    report("home-2")
    if not ok or not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (final state not reached)", flush=True)
        os._exit(1)
    s4 = print_score("P4 shuttle returned home; goal state assembled")
    assert s4 >= s3 - 1e-6, "score decreased across the return"
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

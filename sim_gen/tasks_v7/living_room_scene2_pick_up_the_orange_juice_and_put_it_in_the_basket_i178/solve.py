"""Roll-roll-drop solution for CapsizeRecoveryScene (sim_gen task
`living_room_scene2_pick_up_the_orange_juice_and_put_it_in_the_basket_i178`) — the
task's legitimacy certificate.

The task's load-bearing interactions and how they are executed:

1. ROLL 1, capsized -> side (pure contact dynamics): a world-frame torque about the
   horizontal axis z_hat x d_hat — d = the crate-local face direction pointing AWAY
   from the milk carton — is applied to the crate EVERY step (rate-capped, escalated
   on stall), making it pivot on its ground rim edge. The torque is CUT just past the
   ~49 deg balance diagonal; gravity completes the quarter-roll onto the side wall.
   This roll is what frees (unveils) the caged juice carton. The crate's pose is never
   written: a wrong torque would just make it slide, spin, or fall back capsized.
2. ROLL 2, side -> upright (pure contact dynamics): the same torque axis, cut just
   past the ~31 deg side balance diagonal; the crate topples onto its base and rings
   down. (A pure roll torque is frame-drag-immune by construction: the rotation since
   reset is about the torque axis itself, which that rotation fixes.)
3. DEPOSIT (transport teleport + gravity): ONE pose write stages the freed orange
   carton LYING, 30 mm above the upright crate's mouth (asserted NOT contained at the
   staging pose), zero velocity. The fall over the rim, the impact on the crate floor
   and the ring-down of crate + carton to a settled contained state are contact
   physics.

Teleports move only the juice carton through free air; every change of the CRATE's
pose is torque-driven contact dynamics. success() demands the live settled state:
crate upright AND resting on the floor AND juice contained AND milk NOT contained AND
everything still.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: progress is
latched), then holds HANDS-OFF for >= 3.3 simulated seconds after success() first
turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.<task>.solve --headless [--seed N]
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.capsize_recovery")().build(num_envs=args.num_envs,
                                                      device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    from isaaclab.utils.math import quat_apply, quat_mul

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def upz() -> float:
        return float(scene.crate_up()[0, 2])

    def crate_still() -> bool:
        return (float(scene.crate.data.root_lin_vel_w[0].norm()) < c.settle_speed
                and float(scene.crate.data.root_ang_vel_w[0].norm()) < c.settle_omega)

    def clear_wrench() -> None:
        scene.crate.set_external_force_and_torque(zero3, zero3, env_ids=all_ids,
                                                  is_global=True)

    def report(tag: str) -> None:
        cz = float((scene.crate.data.root_pos_w - scene.env_origins)[0, 2])
        cv = float(scene.crate.data.root_lin_vel_w[0].norm())
        cw = float(scene.crate.data.root_ang_vel_w[0].norm())
        print(f"[solve] {tag:12s} | up_z={upz():+.3f} crate_z={cz:+.3f} "
              f"v={cv:.3f} w={cw:.3f} "
              f"covered={bool(scene.covered()[0])} "
              f"upright={bool(scene.upright()[0])} "
              f"grounded={bool(scene.grounded()[0])} "
              f"j_in={bool(scene.contained(scene.juice)[0])} "
              f"m_in={bool(scene.contained(scene.milk)[0])} "
              f"unv={bool(scene._unveiled_ever[0])} "
              f"rgt={bool(scene._righted_ever[0])} "
              f"ld={bool(scene._loaded_ever[0])} "
              f"still={bool(scene._still()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def roll_torque(phase: str, t_hat: torch.Tensor, cut_upz: float,
                    tau0: float, max_steps: int = 2400) -> float:
        """Apply a rate-capped world torque about t_hat every step until up_z crosses
        cut_upz; escalate on stall. Returns the final torque magnitude used."""
        tau = tau0
        best = -2.0
        stall = 0
        for i in range(max_steps):
            u = upz()
            if u >= cut_upz:
                print(f"[solve] {phase}: cut at up_z={u:+.3f} (step {i}, "
                      f"tau={tau:.2f})", flush=True)
                break
            w = float(scene.crate.data.root_ang_vel_w[0].norm())
            mag = tau if w < 2.2 else 0.0
            t_vec = (t_hat * mag).view(1, 1, 3).expand(n, 1, 3)
            scene.crate.set_external_force_and_torque(zero3, t_vec,
                                                      env_ids=all_ids,
                                                      is_global=True)
            env.step(no_action)
            if u > best + 0.005:
                best = u
                stall = 0
            else:
                stall += 1
                if stall >= 300:
                    tau = min(tau * 1.6, 3.0)
                    stall = 0
                    print(f"[solve] {phase}: stall at up_z={u:+.3f}, escalate "
                          f"tau -> {tau:.2f}", flush=True)
        clear_wrench()
        return tau

    def coast_settle(pred, max_steps: int = 1500, streak: int = 30) -> bool:
        """Hands-off until `pred()` and the crate is still for `streak` steps."""
        quiet = 0
        for _ in range(max_steps):
            env.step(no_action)
            quiet = quiet + 1 if (pred() and crate_still()) else 0
            if quiet >= streak:
                return True
        return False

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    cm = float(scene.crate.root_physx_view.get_masses().sum())
    jm = float(scene.juice.root_physx_view.get_masses().sum())
    mm = float(scene.milk.root_physx_view.get_masses().sum())
    cp = (scene.crate.data.root_pos_w - scene.env_origins)[0]
    jp = (scene.juice.data.root_pos_w - scene.env_origins)[0]
    mp = (scene.milk.data.root_pos_w - scene.env_origins)[0]
    cq = scene.crate.data.root_quat_w[0]
    # capsized quat (0, cos(psi/2), sin(psi/2), 0) -> psi = 2*atan2(qy, qx)
    psi = math.degrees(2.0 * math.atan2(float(cq[2]), float(cq[1])))
    print(f"[solve] layout readback (seed {args.seed}): "
          f"crate=({float(cp[0]):+.3f},{float(cp[1]):+.3f}) yaw={psi:+.1f}deg "
          f"juice=({float(jp[0]):+.3f},{float(jp[1]):+.3f}) "
          f"milk=({float(mp[0]):+.3f},{float(mp[1]):+.3f}) "
          f"masses crate={cm:.3f} juice={jm:.3f} milk={mm:.3f}", flush=True)
    assert abs(cm - c.crate_mass) < 0.02, f"crate mass wrong: {cm}"
    assert abs(jm - c.carton_mass) < 0.02, f"juice mass wrong: {jm}"
    report("reset")
    assert upz() < -0.95, "crate did not settle capsized"
    assert bool(scene.covered()[0]), "juice carton is not covered at reset?!"
    # The caged carton IS 'contained' in pure crate-frame geometry — and earns nothing,
    # because loaded/success gate on upright & grounded. Demonstrate that from stdout:
    assert bool(scene.contained(scene.juice)[0]), \
        "caged carton should satisfy body-frame containment geometry"
    assert not bool(scene.success()[0]), "success at reset?!"
    s0 = print_score("P0 reset+settle (caged containment earns nothing)")
    assert s0 <= 0.02, f"reset score should be ~0, got {s0}"

    # ---------------- roll direction: away from the milk carton -----------------------------
    my = float(mp[1])
    target = torch.tensor([0.0, -1.0 if my > 0 else 1.0, 0.0], device=device)
    best_dot, d_hat = -2.0, None
    q0 = scene.crate.data.root_quat_w
    for ax in ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)):
        e = torch.tensor(ax, device=device).expand(n, 3)
        w = quat_apply(q0, e)[0].clone()
        w[2] = 0.0
        w = w / w.norm()
        d = float(torch.dot(w, target))
        if d > best_dot:
            best_dot, d_hat = d, w.clone()
    t_hat = torch.tensor([-float(d_hat[1]), float(d_hat[0]), 0.0], device=device)
    print(f"[solve] roll dir d=({float(d_hat[0]):+.3f},{float(d_hat[1]):+.3f}) "
          f"(dot away-from-milk {best_dot:+.3f}), torque axis "
          f"t=({float(t_hat[0]):+.3f},{float(t_hat[1]):+.3f})", flush=True)

    # ---------------- phase 1: ROLL 1 — capsized -> side (contact) --------------------------
    tau = 0.6
    ok = False
    for attempt in range(3):
        tau = roll_torque(f"P1 roll1 a{attempt}", t_hat, cut_upz=-0.58, tau0=tau)
        ok = coast_settle(lambda: abs(upz()) <= 0.35)
        if ok or upz() > 0.9:
            break
        print(f"[solve] P1 retry: settled at up_z={upz():+.3f}", flush=True)
        tau = min(tau * 1.3, 3.0)
    report("side")
    assert ok or upz() > 0.9, f"roll 1 failed: up_z={upz():+.3f}"
    assert not bool(scene.covered()[0]), "carton still covered after roll 1"
    assert bool(scene._unveiled_ever[0]), "unveiled latch did not set"
    s1 = print_score("P1 crate rolled onto its side — juice carton unveiled")
    assert s1 >= s0 - 1e-6, "score decreased across roll 1"
    assert s1 >= 0.19, f"unveil credit missing, score {s1}"

    # ---------------- phase 2: ROLL 2 — side -> upright (contact) ---------------------------
    if upz() <= 0.9:
        cut = 0.62
        ok = False
        for attempt in range(3):
            roll_torque(f"P2 roll2 a{attempt}", t_hat, cut_upz=cut, tau0=tau)
            ok = coast_settle(lambda: bool(scene.upright()[0])
                              and bool(scene.grounded()[0]))
            if ok:
                break
            print(f"[solve] P2 retry: settled at up_z={upz():+.3f}", flush=True)
            cut = min(cut + 0.08, 0.80)
            tau = min(tau * 1.3, 3.0)
        assert ok, f"roll 2 failed: up_z={upz():+.3f}"
    else:
        # roll 1 carried straight through — just let the righted latch confirm below
        coast_settle(lambda: bool(scene.upright()[0]) and bool(scene.grounded()[0]))
    report("upright")
    assert bool(scene.upright()[0]) and bool(scene.grounded()[0]), \
        "crate is not upright+grounded"
    assert bool(scene._righted_ever[0]), "righted latch did not set"
    s2 = print_score("P2 crate righted onto its base (torque-driven edge pivots)")
    assert s2 >= s1 - 1e-6, "score decreased across roll 2"
    assert s2 >= 0.49, f"righted credit missing, score {s2}"

    # ---------------- phase 3: DEPOSIT (transport teleport + gravity drop) ------------------
    # One pose write in the upright crate's CURRENT frame: carton LYING (long axis
    # along crate-local x), bottom 30 mm above the rim plane, zero velocity.
    b_pos = scene.crate.data.root_pos_w
    b_quat = scene.crate.data.root_quat_w
    rim = c.floor_t + c.wall_h
    stage_loc = torch.tensor([0.0, 0.0, rim + 0.030 + c.carton_w / 2],
                             device=device).expand(n, 3)
    p_w = b_pos + quat_apply(b_quat, stage_loc)
    # carton local +z along crate-local +x: q = q_crate * qy(pi/2)
    qy90 = torch.tensor([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0],
                        device=device).expand(n, 4)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = p_w
    st[:, 3:7] = quat_mul(b_quat, qy90)
    scene.juice.write_root_state_to_sim(st, all_ids)
    assert not bool(scene.contained(scene.juice)[0]), \
        "staging pose is already contained (teleport must stay above the mouth)"

    done = False
    quiet = 0
    for i in range(1500):
        env.step(no_action)
        quiet = quiet + 1 if bool(scene.success()[0]) else 0
        if quiet >= 30:
            done = True
            break
        if (i + 1) % 240 == 0:
            report(f"P3 t+{(i + 1) / 120.0:.0f}s")
    report("loaded")
    assert bool(scene.contained(scene.juice)[0]), "juice did not land inside"
    assert not bool(scene.contained(scene.milk)[0]), "milk ended up contained?!"
    assert bool(scene.upright()[0]) and bool(scene.grounded()[0]), \
        "the drop knocked the crate over"
    assert done, "loaded crate did not settle to success"
    s3 = print_score("P3 juice dropped into the upright crate, settled")
    assert s3 >= s2 - 1e-6, "score decreased across the deposit"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after deposit)", flush=True)
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, hands-off) ------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 steps = 3.33 s at 120 Hz
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
    except Exception as e:  # noqa: BLE001 — fail fast, don't idle until the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)

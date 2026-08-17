"""Teleport solution for BalanceShelfScene (sim_gen task
`put_groceries_in_cupboard_i258`) — the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. EVICT (applied force): a PD force + gravity feedforward + righting torque
   (a firm force-limited grasp) lifts the heavy BLUE can out of its pocket,
   tracked in the BEAM frame so the grasp follows the shelf as the load comes
   off and the keel starts swinging it level. Once the can is clear of the
   beam's carry volume it is TRANSPORTED (one root-state write through free
   air) to the ground beside the cupboard — open air both ends, nothing judged
   is satisfied by the write.
2. LEVEL (free dynamics): all wrenches zeroed; the keel pendulum alone swings
   the shelf level against its rotary damping. `unloaded` latches here — it
   needs the blue can OFF the beam AND the beam level, so it certifies both
   the eviction and the wait.
3. LOAD FIRST (transport + gravity + contact): red can A is teleported to
   free air 35 mm above one pocket of the LEVEL beam (zero velocity, aligned,
   centred — the pocket mouth is open-topped free space) and DROPPED. Contact
   seats it; its weight out-torques the keel and slams the shelf to that
   stop. `first` latches from the seated contact state.
4. LOAD SECOND (transport + gravity + contact, into a TILTED pocket): the
   pinned beam's pose is read back and red can B is teleported aligned to the
   beam quaternion, bottom 8 mm above the RAISED pocket's floor, inside its
   fence rims, then dropped. Contact seats it; the load is now symmetric and
   the keel swings the shelf LEVEL — the success equilibrium is produced
   entirely by the pendulum dynamics.
5. HANDS-OFF: success must hold through >= 3.3 simulated seconds untouched —
   the keel alone keeps the beam level, both cans seated on friction.

Order is forced by physics, not rubric timestamps: with the blue can aboard
the beam can never level (it out-torques a red can even head-to-head), and
the beam levels only after the red load is symmetric.

Every teleport is TRANSPORT ONLY (held object through free air, released in
free air); every load-bearing interaction — the eviction lift, the pocket
seatings, the tip to the stop, the return to level — is contact/force
dynamics.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's stage credit is latched) and `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds after the hands-off persistence window.

Run (forge): python -u -m simgen_tasks.put_groceries_in_cupboard_i258.solve --headless [--seed N]
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
    env = ENVS.get("simgen.balance_shelf")().build(num_envs=args.num_envs, device=device)
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

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def tilt() -> float:
        return float(scene.tilt_deg()[0])

    def report(tag: str) -> None:
        la = scene._beam_local(scene.can_a.data.root_pos_w)[0]
        lb = scene._beam_local(scene.can_b.data.root_pos_w)[0]
        print(f"[solve] {tag:14s} | tilt={tilt():+7.2f}deg "
              f"beam_w={float(scene.beam.data.root_ang_vel_w[0].norm()):+.3f} "
              f"A_beam=({float(la[0]):+.3f},{float(la[1]):+.3f},{float(la[2]):+.3f}) "
              f"B_beam=({float(lb[0]):+.3f},{float(lb[1]):+.3f},{float(lb[2]):+.3f}) "
              f"matched={bool(scene.matched()[0])} "
              f"level={bool(scene.level()[0])} "
              f"decoy_aboard={bool(scene.aboard(scene.decoy)[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    zero = torch.zeros(n, 1, 3, device=device)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)

    def carry(body, mass: float, frame_body, tgt_local, *, kp: float, kd: float,
              clamp: float, ku: float, kw: float, steps: int, done=None,
              label: str = "") -> None:
        """Applied-force carry: PD toward a frame_body-local target + gravity
        feedforward (a firm force-limited grasp), plus a righting torque that
        stands in for the grasp's orientation constraint (transverse-only spin
        damping). Body-frame wrench recomputed from the CURRENT quat each step
        (the wrench frame-drag quirk)."""
        from isaaclab.utils.math import quat_apply_inverse

        tgt = torch.zeros(n, 3, device=device)
        tgt[:] = torch.tensor(tgt_local, device=device)
        for _ in range(steps):
            q = body.data.root_quat_w
            p = body.data.root_pos_w
            v = body.data.root_lin_vel_w
            w = body.data.root_ang_vel_w
            tgt_w = frame_body.data.root_pos_w \
                + quat_apply(frame_body.data.root_quat_w, tgt)
            f_w = mass * 9.81 * ez + kp * (tgt_w - p) - kd * v
            f_norm = f_w.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            f_w = f_w * (f_norm.clamp(max=clamp) / f_norm)
            f_b = quat_apply_inverse(q, f_w)
            axis = quat_apply(q, ez)
            w_perp = w - (w * axis).sum(dim=-1, keepdim=True) * axis
            t_w = ku * torch.cross(axis, ez, dim=-1) - kw * w_perp
            t_b = quat_apply_inverse(q, t_w)
            body.set_external_force_and_torque(f_b.reshape(n, 1, 3), t_b.reshape(n, 1, 3))
            env.step(no_action)
            if done is not None and done():
                break
        body.set_external_force_and_torque(zero, zero)
        loc = scene._beam_local(body.data.root_pos_w)[0]
        print(f"[solve] carry {label}: reached beam-local "
              f"({float(loc[0]):+.3f},{float(loc[1]):+.3f},{float(loc[2]):+.3f}) "
              f"after <= {steps} steps", flush=True)

    def drop_red(body, side: float, gap: float, tag: str) -> None:
        """TRANSPORT a red can to free air `gap` above the pocket floor at
        beam-local (0, side*well_y), aligned to the CURRENT beam quat, zero
        velocity, then let gravity and contact seat it."""
        loc = torch.zeros(n, 3, device=device)
        loc[:, 1] = side * c.well_y
        loc[:, 2] = c.plate_top + gap + c.can_h / 2
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.beam.data.root_pos_w \
            + quat_apply(scene.beam.data.root_quat_w, loc)
        st[:, 3:7] = scene.beam.data.root_quat_w
        body.write_root_state_to_sim(st, all_ids)
        print(f"[solve] {tag}: can transported to free air {gap * 1000:.0f} mm above "
              f"the pocket floor (open-topped mouth; gravity+contact do the seating)",
              flush=True)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(300)  # the decoy-pinned beam slams to its stop and settles there
    cp = (scene.cupboard.data.root_pos_w - scene.env_origins)[0]
    cq = scene.cupboard.data.root_quat_w[0]
    cyaw = math.degrees(2.0 * math.atan2(float(cq[3]), float(cq[0])))
    side = float(scene.decoy_side[0])
    print(f"[solve] layout readback (seed {args.seed}): "
          f"cup=({float(cp[0]):+.3f},{float(cp[1]):+.3f}) yaw={cyaw:+.1f}deg "
          f"decoy_side={side:+.0f} tilt={tilt():+.2f}deg", flush=True)
    # mass readbacks: authored MassAPI / per-child density must have produced
    # real masses (custom spawners apply no cfg mass schemas)
    cup_m = float(scene.cupboard.root_physx_view.get_masses()[0].sum())
    beam_m = float(scene.beam.root_physx_view.get_masses()[0].sum())
    red_m = float(scene.can_a.root_physx_view.get_masses()[0].sum())
    dec_m = float(scene.decoy.root_physx_view.get_masses()[0].sum())
    print(f"[solve] mass readback: cupboard={cup_m:.1f} kg beam={beam_m:.3f} kg "
          f"red={red_m:.3f} kg decoy={dec_m:.3f} kg", flush=True)
    assert cup_m > 30.0, f"cupboard mass {cup_m} (MassAPI not applied?)"
    assert 0.9 < beam_m < 1.6, f"beam mass {beam_m} (density not applied?)"
    assert 0.25 < red_m < 0.36, f"red can mass {red_m} (density not applied?)"
    assert 0.50 < dec_m < 0.75, f"decoy mass {dec_m} (density not applied?)"
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    # the decoy must have slammed the shelf to ITS stop (tilt sign = -side)
    assert -side * tilt() > c.stop_deg - 3.0, \
        f"beam must be pinned at the decoy's stop: tilt={tilt():.2f} side={side:+.0f}"
    assert abs(tilt()) < c.stop_deg + 1.5, f"tilt beyond the stop: {tilt():.2f}"
    assert bool(scene.aboard(scene.decoy)[0]), "decoy must start aboard"
    assert bool(scene.seated(scene.decoy, c.decoy_h, scene.decoy_side)[0]), \
        "decoy must start seated in its pocket"
    s0 = print_score("P0 reset+settle (decoy pins the shelf at its stop)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.01, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: evict the decoy, let the keel level the shelf ---------------
    # lift just clear of the fence rims (rim plane is beam-local z=0); the
    # roof underside is only ~0.16 above in beam coordinates, so stay low
    carry(scene.decoy, dec_m, scene.beam, (0.0, side * c.well_y, 0.12),
          kp=25.0, kd=8.0, clamp=12.0, ku=0.05, kw=0.02, steps=480,
          done=lambda: float(scene._beam_local(scene.decoy.data.root_pos_w)[0, 2]) > 0.10,
          label="decoy evict")
    assert float(scene._beam_local(scene.decoy.data.root_pos_w)[0, 2]) > 0.085, \
        "decoy must be lifted clear of the beam (bottom above the fence rims)"
    # TRANSPORT: held can through free air to the ground beside the cupboard
    park = torch.zeros(n, 3, device=device)
    park[:, 1] = side * 0.45
    st = torch.zeros(n, 13, device=device)
    st[:, 0:2] = (scene.cupboard.data.root_pos_w
                  + quat_apply(scene.cupboard.data.root_quat_w, park))[:, 0:2]
    st[:, 2] = scene.env_origins[:, 2] + c.decoy_h / 2 + 0.003
    st[:, 3:7] = scene.cupboard.data.root_quat_w
    scene.decoy.write_root_state_to_sim(st, all_ids)
    print("[solve] P1: decoy transported to the ground beside the cupboard "
          "(free air both ends; ~aboard was already false at release)", flush=True)
    # hands off: the keel pendulum swings the shelf level against its damping
    for _ in range(900):
        env.step(no_action)
        if bool(scene._unloaded[0]) and abs(tilt()) < 2.0 \
                and float(scene.beam.data.root_ang_vel_w[0].norm()) < 0.05:
            break
    report("unloaded")
    assert not bool(scene.aboard(scene.decoy)[0]), "decoy must be off the beam"
    assert bool(scene.level()[0]), f"keel must level the empty shelf, tilt={tilt():.2f}"
    assert bool(scene._unloaded[0]), "unloaded must latch after eviction + leveling"
    assert not bool(scene.success()[0]), "empty level shelf must not be success"
    s1 = print_score("P1 decoy evicted; keel swings the shelf level")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_unload - 1e-3, f"P1 score {s1}"

    # ---------------- phase 2: drop red A into a pocket — shelf tips to the stop -----------
    drop_red(scene.can_a, +1.0, 0.035, "P2 (level pocket, 35 mm drop)")
    for _ in range(600):
        env.step(no_action)
        if bool(scene._first[0]) and tilt() < -(c.stop_deg - 3.0) \
                and float(scene.beam.data.root_ang_vel_w[0].norm()) < 0.05:
            break
    report("first")
    assert bool(scene.seated(scene.can_a, c.can_h, +1.0)[0]), \
        "can A must be seated in the +y pocket"
    assert tilt() < -(c.stop_deg - 3.0), \
        f"one can must pin the shelf at the stop, tilt={tilt():.2f}"
    assert bool(scene._first[0]), "first must latch from the seated contact state"
    assert not bool(scene.success()[0]), "single-loaded shelf must not be success"
    s2 = print_score("P2 first red can seated; its weight pins the shelf")
    assert s2 >= s1 - 1e-6 and s2 >= c.w_unload + c.w_first - 1e-3, f"P2 score {s2}"

    # ---------------- phase 3: drop red B into the RAISED pocket — shelf levels ------------
    drop_red(scene.can_b, -1.0, 0.008, "P3 (raised tilted pocket, 8 mm drop)")
    for _ in range(900):
        env.step(no_action)
        if bool(scene._both[0]) and bool(scene.success()[0]):
            break
    report("both")
    assert bool(scene.matched()[0]), "both reds must be seated one per pocket"
    assert bool(scene._both[0]), "both must latch from the matched contact state"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (matched but success not reached)", flush=True)
        os._exit(1)
    s3 = print_score("P3 second red can seated; keel swings the shelf level")
    assert s3 >= s2 - 1e-6 and s3 >= 0.999, f"P3 score {s3}"

    # let the pendulum ring down before the strict persistence window: the P3
    # break fires at the EARLIEST success flicker, and a lightly damped beam
    # crossing level peaks above the settle gate. A single slow readback is a
    # turning point, not stillness — require a sustained slow STREAK after a
    # fixed decay window (angular damping kills the swing exponentially).
    step(240)
    streak = 0
    for _ in range(600):
        env.step(no_action)
        if abs(tilt()) < 2.0 \
                and float(scene.beam.data.root_ang_vel_w[0].norm()) < 0.03:
            streak += 1
            if streak >= 30:
                break
        else:
            streak = 0
    assert streak >= 30, f"beam must ring down, tilt={tilt():.2f}"
    assert bool(scene.success()[0]), \
        f"success must hold once the beam is quiet, tilt={tilt():.2f}"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for i in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        ok_i = bool(scene.success()[0])
        if not ok_i:
            print(f"[solve] P4 probe {i}: success=False | "
                  f"matched={bool(scene.matched()[0])} "
                  f"level={bool(scene.level()[0])} tilt={tilt():+.2f} "
                  f"decoy_aboard={bool(scene.aboard(scene.decoy)[0])} "
                  f"settled={bool(scene.settled()[0])} "
                  f"vA={float(scene.can_a.data.root_lin_vel_w[0].norm()):.4f} "
                  f"vB={float(scene.can_b.data.root_lin_vel_w[0].norm()):.4f} "
                  f"vCup={float(scene.cupboard.data.root_lin_vel_w[0].norm()):.4f} "
                  f"wBeam={float(scene.beam.data.root_ang_vel_w[0].norm()):.4f}",
                  flush=True)
        hold = hold and ok_i
    report("persist")
    s4 = print_score("P4 persistence 3.3 s hands-off")
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
    except Exception as exc:  # noqa: BLE001 - Kit teardown hangs; die loudly NOW
        print(f"SIM_GEN_SOLVE: FAIL (exception: {exc!r})", flush=True)
        import traceback

        traceback.print_exc()
        os._exit(1)

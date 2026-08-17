"""Teleport solution for PuddingDockDispenseScene (sim_gen task
`living_room_scene4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray_i254`) —
the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): a single root-state write carries the scattered tray from
   its random spawn pose to the open floor IN FRONT of the bay mouth (station-local
   x = -0.155, front face 23 mm clear of the mouth plane), aligned with the station,
   handle trailing, zero velocity. The write satisfies no rubric clause (the docked
   window is 0.37 m deeper, past the mouth, under the deck).
2. DOCKING (external force + contact): the tray is PUSHED through the mouth with a
   velocity-capped horizontal force on its root (the stand-in for a hand on the
   yellow handle), with small lateral/yaw PD corrections; the bay walls guide it and
   it seats against the BACK-WALL STOP. The `docked` credit (tray at rest inside the
   window) is produced entirely by sliding contact against the ground and the stop.
   The solve asserts success() is False here: a docked empty tray — and equally the
   SEED'S OWN PLAN of "pudding in a tray" without the dock — is not success.
3. LANE SELECTION (scene-state read = the oracle's stand-in for the color cue): the
   chute holding the brown pudding is read from `scene.pud_lane` (a visual solver
   reads the same fact off the box colors, per describe()). The solve asserts the
   readback against the actual station-local y of both boxes before pushing.
4. DISPENSING (external force + gravity + contact): the pudding box is pushed along
   its roofed chute with a small velocity-capped force (the stand-in for a fingertip
   through the chute mouth) until its CoM passes the drop-hole edge; the force is
   CUT the instant the box starts to fall, and it free-falls ~130 mm through the
   hole into the docked tray below. Landing, containment and settling are produced
   entirely by gravity and contact; nothing is ever teleported into a scoring state.
5. Force honesty: pushes act at the body CoM, horizontally, magnitudes ~6 N (tray,
   vs ~2.3 N ground friction) and ~0.6 N (box, vs ~0.33 N chute friction) — far
   below anything that could deform the 60 kg station — and every push is
   velocity-capped so credit is earned quasi-statically, not by slamming.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

Run (forge): python -u -m simgen_tasks.living_room_scene4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray_i254.solve --headless [--seed N]
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
    from . import scene as _scene  # noqa: F401 - registers simgen.pudding_dock_dispense
except ImportError:  # pragma: no cover - direct-script fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as _scene  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.pudding_dock_dispense")().build(num_envs=args.num_envs,
                                                           device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        dx, dy, dyaw = scene.tray_dock_err()
        pl = scene._station_local(scene.pudding.data.root_pos_w)[0]
        tv = float(scene.tray.data.root_lin_vel_w[0].norm())
        pv = float(scene.pudding.data.root_lin_vel_w[0].norm())
        print(f"[solve] {tag:16s} | dock_err=({float(dx[0]):+.3f},{float(dy[0]):+.3f},"
              f"{float(dyaw[0]):+.1f}deg) docked={bool(scene.tray_docked()[0])} "
              f"pud_loc=({float(pl[0]):+.3f},{float(pl[1]):+.3f},{float(pl[2]):+.3f}) "
              f"prog={float(scene.pudding_progress_now()[0]):.2f} "
              f"in_tray={bool(scene.pudding_in_tray()[0])} "
              f"decoy_in_tray={bool(scene.decoy_in_tray()[0])} "
              f"v_tray={tv:.3f} v_pud={pv:.3f} settled={bool(scene.settled()[0])} "
              f"L(dock={bool(scene._docked[0])},land={bool(scene._landed[0])},"
              f"prog={float(scene._progress[0]):.2f}) "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_wrench(body) -> None:
        body.set_external_force_and_torque(zero3, zero3, env_ids=all_ids, is_global=True)

    def station_axes() -> tuple[torch.Tensor, torch.Tensor]:
        q = scene.station.data.root_quat_w
        xh = _scene._qapply(q, torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(n, 3))
        yh = _scene._qapply(q, torch.tensor([[0.0, 1.0, 0.0]], device=device).expand(n, 3))
        return xh, yh

    def tele_tray_to_approach() -> None:
        """TRANSPORT ONLY: aligned pose on the open floor in front of the mouth
        (station-local x=-0.155 -> front face 23 mm clear of the mouth plane; far
        outside the docked window at x=0.218)."""
        q_st = scene.station.data.root_quat_w
        loc = torch.tensor([[-0.155, 0.0, 0.0]], device=device).expand(n, 3)
        pos = scene.station.data.root_pos_w + _scene._qapply(q_st, loc)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 2] = 0.001
        st[:, 3:7] = q_st
        scene.tray.write_root_state_to_sim(st, all_ids)

    # Wrench-frame strategy (some pods rotate an applied "global" wrench by the
    # body's rotation since its reference orientation). Probed once on the tray.
    strat = {"mode": 0, "qref": {}}

    def enc(body_key: str, body, f_world: torch.Tensor) -> torch.Tensor:
        if strat["mode"] == 0:
            return f_world
        return _scene.encode_force(1, strat["qref"][body_key], body.data.root_quat_w,
                                   f_world)

    def push_probe(qref: torch.Tensor | None) -> float:
        """Apply a 12-step raw/encoded +x_station push from the approach pose and
        return the horizontal alignment of the resulting displacement."""
        tele_tray_to_approach()
        step(15)
        xh, _ = station_axes()
        p0 = scene.tray.data.root_pos_w.clone()
        for _ in range(12):
            f = 6.0 * xh
            if qref is not None:
                f = _scene.encode_force(1, qref, scene.tray.data.root_quat_w, f)
            scene.tray.set_external_force_and_torque(f.view(n, 1, 3), zero3,
                                                     env_ids=all_ids, is_global=True)
            env.step(no_action)
        clear_wrench(scene.tray)
        d = (scene.tray.data.root_pos_w - p0)[0, :2]
        if float(d.norm()) < 0.003:
            return -1.0
        return float((d / d.norm() * xh[0, :2]).sum())

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # everything seats; boxes settle onto their chute floors
    q_tr_reset = scene.tray.data.root_quat_w.clone()
    q_pud_reset = scene.pudding.data.root_quat_w.clone()
    q_id = torch.zeros(n, 4, device=device)
    q_id[:, 0] = 1.0
    lane = int(scene.pud_lane[0])
    sq = scene.station.data.root_quat_w[0]
    syaw = math.degrees(2.0 * math.atan2(float(sq[3]), float(sq[0])))
    print(f"[solve] layout readback (seed {args.seed}): station_yaw={syaw:+.1f}deg "
          f"pud_lane={lane:+d} ({'left/+y' if lane > 0 else 'right/-y'} chute)",
          flush=True)
    report("reset")
    m_st = float(scene.station.root_physx_view.get_masses().sum())
    m_tr = float(scene.tray.root_physx_view.get_masses().sum())
    assert abs(m_st - c.station_mass) < 1.0, f"station mass readback {m_st} != 60"
    assert abs(m_tr - c.tray_mass) < 0.05, f"tray mass readback {m_tr} != 0.5"
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    pl = scene._station_local(scene.pudding.data.root_pos_w)[0]
    dl = scene._station_local(scene.decoy.data.root_pos_w)[0]
    assert abs(float(pl[2]) - (c.deck_top + c.box_size[2] / 2)) < 0.01, \
        f"pudding must sit on the deck, z_loc={float(pl[2]):.3f}"
    assert float(pl[1]) * lane > 0.03 and float(dl[1]) * lane < -0.03, \
        "boxes must sit in opposite chutes matching pud_lane"
    assert not bool(scene.tray_docked()[0]), "tray must start undocked"
    s0 = print_score("P0 reset+settle (tray scattered, boxes seated in the chutes)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: dock the tray (teleport transport, force push) --------------
    a0 = push_probe(None)
    if a0 > 0.7:
        strat["mode"] = 0
        print(f"[solve] wrench mode 0 (raw global) OK, align={a0:+.2f}", flush=True)
    else:
        a1 = push_probe(q_tr_reset)
        if a1 > 0.7:
            strat["mode"] = 1
            strat["qref"] = {"tray": q_tr_reset, "pudding": q_pud_reset}
            print(f"[solve] wrench mode 1 (reset-quat ref), align={a1:+.2f} "
                  f"(mode0 gave {a0:+.2f})", flush=True)
        else:
            a2 = push_probe(q_id)
            if a2 > 0.7:
                strat["mode"] = 1
                strat["qref"] = {"tray": q_id, "pudding": q_id}
                print(f"[solve] wrench mode 1 (identity ref), align={a2:+.2f} "
                      f"(mode0 {a0:+.2f}, mode1a {a1:+.2f})", flush=True)
            else:
                print(f"SIM_GEN_SOLVE: FAIL (no wrench mode moves the tray: "
                      f"{a0:+.2f}/{a1:+.2f}/{a2:+.2f})", flush=True)
                os._exit(1)

    tele_tray_to_approach()
    step(20)
    report("approach")

    docked = False
    for j in range(2400):
        xh, yh = station_axes()
        dx, dy, dyaw = scene.tray_dock_err()
        v = scene.tray.data.root_lin_vel_w
        v_along = (v * xh).sum(dim=-1)
        v_lat = (v * yh).sum(dim=-1)
        vcap = 0.20 if float(dx[0]) < -0.06 else 0.08
        fx = torch.where(v_along < vcap, torch.full_like(v_along, 6.0),
                         torch.zeros_like(v_along))
        fy = (-8.0 * dy - 2.0 * v_lat).clamp(-3.0, 3.0)
        f_world = fx.unsqueeze(-1) * xh + fy.unsqueeze(-1) * yh
        tz = (-0.15 * torch.deg2rad(dyaw)
              - 0.02 * scene.tray.data.root_ang_vel_w[:, 2]).clamp(-0.5, 0.5)
        tq = torch.zeros(n, 1, 3, device=device)
        tq[:, 0, 2] = tz
        scene.tray.set_external_force_and_torque(
            enc("tray", scene.tray, f_world).view(n, 1, 3), tq,
            env_ids=all_ids, is_global=True)
        env.step(no_action)
        if j % 240 == 239:
            report(f"dock-{j + 1}")
        if bool(scene.tray_docked()[0]) and float(v.norm(dim=-1)[0]) < 0.04:
            docked = True
            break
    clear_wrench(scene.tray)
    step(150)  # hands-off settle at the stop
    report("docked")
    if not (docked and bool(scene.tray_docked()[0])):
        print("SIM_GEN_SOLVE: FAIL (tray never docked)", flush=True)
        os._exit(1)
    assert bool(scene._docked[0]), "docked latch must be set"
    drift = float((scene.station.data.root_pos_w
                   - scene.env_origins)[0, :2].norm())
    assert abs(drift - float(torch.tensor(c.station_pos).norm())) < 0.09, \
        "station must not have been bulldozed"
    assert not bool(scene.success()[0]), \
        "a docked empty tray (and any 'pudding in tray' without the dock — the " \
        "seed's plan) must NOT be success"
    s1 = print_score("P1 tray docked against the back-wall stop")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_docked - 1e-6, \
        f"P1 score {s1} (expect docked={c.w_docked})"

    # ---------------- phase 2: push the pudding through its chute; gravity delivers --------
    pl = scene._station_local(scene.pudding.data.root_pos_w)[0]
    print(f"[solve] pushing the BROWN box in lane {lane:+d} "
          f"(y_loc={float(pl[1]):+.3f})", flush=True)
    fell = False
    for j in range(1500):
        loc_z = float(scene._station_local(scene.pudding.data.root_pos_w)[0, 2])
        if loc_z < 0.12:  # tipped through the hole -> cut the force, let it fall
            fell = True
            break
        xh, _ = station_axes()
        v_along = (scene.pudding.data.root_lin_vel_w * xh).sum(dim=-1)
        fx = torch.where(v_along < 0.08, torch.full_like(v_along, 0.6),
                         torch.zeros_like(v_along))
        f_world = fx.unsqueeze(-1) * xh
        scene.pudding.set_external_force_and_torque(
            enc("pudding", scene.pudding, f_world).view(n, 1, 3), zero3,
            env_ids=all_ids, is_global=True)
        env.step(no_action)
        if j % 120 == 119:
            report(f"push-{j + 1}")
    clear_wrench(scene.pudding)
    if not fell:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (pudding never tipped into the hole)", flush=True)
        os._exit(1)
    # hands-off: free fall through the hole, landing, containment, settling.
    ok_p2 = False
    for j in range(900):
        env.step(no_action)
        if bool(scene.success()[0]):
            ok_p2 = True
            break
        if j % 180 == 179:
            report(f"land-{j + 1}")
    report("landed")
    if not ok_p2:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (pudding never settled in the docked tray)",
              flush=True)
        os._exit(1)
    assert bool(scene._landed[0]), "landed latch must be set"
    assert not bool(scene.decoy_in_tray()[0]), "decoy must still be in its chute"
    s2 = print_score("P2 pudding dispensed into the docked tray; success")
    assert s2 >= s1 - 1e-6, "score decreased across the dispense"

    # ---------------- phase 3: persistence (>= 3.3 simulated seconds, hands-off) -----------
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
    try:
        main()
    except Exception:  # noqa: BLE001 - fail FAST; a hung Kit burns the forge slot
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)

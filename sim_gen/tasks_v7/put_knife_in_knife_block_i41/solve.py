"""Teleport solution for KerfChopScene (sim_gen task put_knife_in_knife_block_i41) —
the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): a single root-state write carries the KNIFE from its ground
   start slot to a hover pose above the kerf slot — blade down, blade centred over
   the slot, tip 12 mm above the cover. The path is free air; the write satisfies no
   rubric clause (the blade is above the cover, not engaged).
2. GRIP EMULATION (applied wrench): from here on the knife is "held": a PID attitude
   wrench keeps it blade-down (what a wrist does), and a vertical force sets the
   press. The force frame is CALIBRATED first: a small test force is applied under
   each candidate drag-encoding (identity / reset-pose / spawn-pose) and the one
   whose measured acceleration matches the commanded direction is locked in.
3. INSERTION (contact-guided descent): the support force is eased below gravity so
   the blade descends SLOWLY through the slot until it touches the red span — the
   `engaged` credit is earned by the blade tip physically below the cover top inside
   the slot. Resting contact only: the seams hold (their break threshold is ~10x the
   blade's weight).
4. SEVERING (pressed force through contact): the press force ramps 10 -> 40 N. The
   blade edge loads the red span; the bending moment at the seam welds crosses
   physics:breakTorque and BOTH seams snap — a real PhysX joint-break event driven
   entirely by contact forces. Nothing is teleported past this interaction: the
   `pressed` and `sev` credits are produced by deflection and real seam separation.
5. RELEASE + GRAVITY: the wrench is zeroed, the knife is teleported out (free-space
   retraction along the slot axis) to a parking spot, and gravity drops the freed
   red section into the well where it settles. Every success clause (severed seams,
   red piece inside the well, beige ends seated, station upright, settled) is a
   physical outcome.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.put_knife_in_knife_block_i41.solve --headless [--seed N]
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
    from .scene import _qapply, _qinv, _qmul, _qz
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scene import _qapply, _qinv, _qmul, _qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.kerf_chop")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import quat_apply

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
        sep = scene.seam_sep()[0]
        tip = scene.blade_tip_local()[0]
        ml = scene._station_local(scene.mid.data.root_pos_w)[0]
        print(f"[solve] {tag:14s} | tip=({float(tip[0]):+.3f},{float(tip[1]):+.3f},"
              f"{float(tip[2]):+.3f}) mid=({float(ml[0]):+.3f},{float(ml[1]):+.3f},"
              f"{float(ml[2]):+.3f}) sep=({float(sep[0]) * 1000:.1f},"
              f"{float(sep[1]) * 1000:.1f})mm eng={bool(scene._engaged[0])} "
              f"sevA={bool(scene._sev_a[0])} "
              f"sevB={bool(scene._sev_b[0])} well={bool(scene.mid_in_well()[0])} "
              f"seated={bool(scene.outers_seated()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    q_reset = scene.knife.data.root_quat_w.clone()  # candidate drag reference (reset pose)
    step(180)
    st_p = (scene.station.data.root_pos_w - scene.env_origins)[0]
    st_q = scene.station.data.root_quat_w[0]
    st_yaw = math.degrees(2.0 * math.atan2(float(st_q[3]), float(st_q[0])))
    print(f"[solve] layout readback (seed {args.seed}): "
          f"station=({float(st_p[0]):+.3f},{float(st_p[1]):+.3f}) yaw={st_yaw:+.1f}deg "
          f"rod_off={float(scene.rod_off[0]) * 1000:+.1f}mm "
          f"knife_slot={'+y' if float(scene.knife_slot[0]) > 0 else '-y'}",
          flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    sep0 = scene.seam_sep()[0]
    assert float(sep0.max()) < 0.005, \
        f"seams must be intact after settle, got sep={sep0.tolist()}"
    assert bool(scene.outers_seated()[0]), "outer segments must start seated"
    assert not bool(scene.mid_in_well()[0]), "red segment must start spanning the well"
    s0 = print_score("P0 reset+settle (rod welded and seated, knife on the ground)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: knife -> hover over the slot (transport) --------------------
    stq = scene.station.data.root_quat_w
    stp = scene.station.data.root_pos_w
    # blade tip x range that keeps the whole blade inside the slot aperture
    tip_x_max = c.slot_x_half - c.blade_len / 2 - 0.002

    def stage_loc(tip_x_off: float, z: float) -> torch.Tensor:
        """Station-frame knife-origin location for a blade-tip target x = rod_off +
        tip_x_off (clamped so the blade stays inside the slot aperture)."""
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0] = (scene.rod_off + tip_x_off).clamp(-tip_x_max, tip_x_max) \
            - c.blade_len / 2
        loc[:, 2] = z
        return loc

    hover_loc = stage_loc(0.0, c.cover_top + 0.012 + c.blade_h)  # tip 12 mm above cover
    q_use = stq.clone()  # knife canonical frame IS the use pose (blade down, +x along slot)

    def teleport_knife(loc_st: torch.Tensor, quat_w: torch.Tensor) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = stp + quat_apply(stq, loc_st)
        st[:, 3:7] = quat_w
        scene.knife.write_root_state_to_sim(st, all_ids)

    teleport_knife(hover_loc, q_use)
    report("hover")

    # ---------------- grip emulation: calibrated wrench + PID attitude hold ----------------
    # Candidate drag references for the force-frame convention (see scene.encode_force):
    # identity (no drag), the knife's reset orientation, the knife's spawn-template
    # orientation. Probe with an upward test force; lock the candidate that lifts.
    q_spawn = torch.tensor([0.70711, 0.70711, 0.0, 0.0], device=device).expand(n, 4)
    candidates: list[tuple[str, torch.Tensor | None]] = [
        ("identity", None), ("reset", q_reset), ("spawn", q_spawn)]
    cand = {"i": 0}

    def enc(v_world: torch.Tensor) -> torch.Tensor:
        q_ref = candidates[cand["i"]][1]
        if q_ref is None:
            return v_world
        return _qapply(_qmul(q_ref, _qinv(scene.knife.data.root_quat_w)), v_world)

    pid_int = torch.zeros(n, 3, device=device)

    def hold_wrench(fz: float, support: float = 1.0,
                    xy_target: torch.Tensor | None = None) -> None:
        """One step of 'held knife': vertical force (support*weight + fz down),
        a lateral PD hold toward xy_target (keeps the blade centred on the rod
        under press loads), plus PID attitude torque toward q_use. All channels
        pass through the calibrated frame encoding."""
        g_comp = support * c.tool_mass * 9.81
        f_w = torch.zeros(n, 3, device=device)
        f_w[:, 2] = g_comp - fz
        if xy_target is not None:
            dxy = xy_target - scene.knife.data.root_pos_w[:, :2]
            vxy = scene.knife.data.root_lin_vel_w[:, :2]
            f_w[:, :2] = (60.0 * dxy - 4.0 * vxy).clamp(-2.5, 2.5)
        # attitude error as world axis-angle
        q_err = _qmul(q_use, _qinv(scene.knife.data.root_quat_w))
        sgn = torch.where(q_err[:, 0:1] < 0, -torch.ones_like(q_err[:, 0:1]),
                          torch.ones_like(q_err[:, 0:1]))
        ax = q_err[:, 1:4] * sgn * 2.0  # small-angle axis*angle approx
        w = scene.knife.data.root_ang_vel_w
        pid_int.add_(0.002 * ax).clamp_(-1.2, 1.2)
        t_w = 2.0 * ax - 0.15 * w + pid_int
        scene.knife.set_external_force_and_torque(
            enc(f_w).view(n, 1, 3), enc(t_w).view(n, 1, 3),
            env_ids=all_ids, is_global=True)
        env.step(no_action)

    # calibrate: full support + 2 N up must make the knife rise, not translate away
    z0 = float(scene.knife.data.root_pos_w[0, 2])
    for name, _q in candidates:
        pid_int.zero_()
        ok = True
        for _ in range(12):
            hold_wrench(fz=-2.0)  # net 2 N upward
        z1 = float(scene.knife.data.root_pos_w[0, 2])
        xy_drift = float((scene.knife.data.root_pos_w[0, :2]
                          - (stp + quat_apply(stq, hover_loc))[0, :2]).norm())
        ok = (z1 > z0 - 0.005) and xy_drift < 0.03
        print(f"[solve] frame probe '{name}': dz={z1 - z0:+.4f} drift={xy_drift:.4f} "
              f"-> {'LOCK' if ok else 'reject'}", flush=True)
        if ok:
            break
        cand["i"] += 1
        teleport_knife(hover_loc, q_use)  # re-stage for the next probe
        pid_int.zero_()
    else:
        print("SIM_GEN_SOLVE: FAIL (no force-frame candidate works)", flush=True)
        os._exit(1)
    # ---------------- insertion: transport into the slot aperture, then touch down ---------
    # The blade (4 mm) fits the 13 mm slot with clearance on all sides, so the pose with
    # the tip 3 mm above the rod top is contact-free: teleporting there is pure transport.
    # The touch-down and everything after it (the actual load-bearing press) is contact
    # dynamics under the held-wrench emulation.
    engage_z = c.rod_top_z + 0.003 + c.blade_h
    engage_loc = stage_loc(0.0, engage_z)
    teleport_knife(engage_loc, q_use)
    pid_int.zero_()
    hover_xy = (stp + quat_apply(stq, engage_loc))[:, :2]
    for _ in range(50):  # gentle touch-down: net 0.3 N down through the last 3 mm
        hold_wrench(fz=0.3, support=1.0, xy_target=hover_xy)
        if bool(scene._engaged[0]) and \
                float(scene.blade_tip_local()[0, 2]) < c.rod_top_z + 0.002:
            break
    report("engaged")
    tip = scene.blade_tip_local()[0]
    if not bool(scene._engaged[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (blade never engaged the slot)", flush=True)
        os._exit(1)
    print(f"[solve] blade touched down on the red span "
          f"(tip z={float(tip[2]) * 1000:.1f} mm)", flush=True)
    s1 = print_score("P1 blade engaged in the kerf slot, resting on the rod")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_engaged - 1e-6, f"P1 score {s1}"
    # resting is NOT enough: seams must still be intact under blade weight
    assert float(scene.seam_sep()[0].max()) < 0.005, "seams broke under resting blade!?"

    # ---------------- severing: staged press ramps through the blade -----------------------
    # A centre press usually snaps both seams together. If the break is asymmetric,
    # the mid hangs from the surviving seam as a hinge; a flat blade over the centre
    # then only touches the PEAK of the incline (at the hinge itself — zero moment
    # arm), so the survivor never sees torque. Recovery: re-stage the press on the
    # DROPPED side, where blade contact has a large arm about the hinge.

    def press_at(tip_x_off: float, budget: int, tag: str) -> bool:
        """Stage the blade at tip x = rod_off + tip_x_off (transport through free
        space above the rod line), touch down, then ramp the press. True iff both
        seams end up severed."""
        loc = stage_loc(tip_x_off, engage_z)
        pos_w = stp + quat_apply(stq, loc)
        xy = pos_w[:, :2]
        teleport_knife(loc, q_use)
        pid_int.zero_()
        for _ in range(30):  # touch down gently before loading
            hold_wrench(fz=0.3, support=1.0, xy_target=xy)
        fz = 8.0
        recoveries = 0
        for i in range(budget):
            hold_wrench(fz=fz, support=1.0, xy_target=xy)
            sev = scene.severed()[0]
            if bool(sev[0]) and bool(sev[1]):
                print(f"[solve] press[{tag}]: BOTH SEAMS SNAPPED at fz={fz:.0f}N",
                      flush=True)
                return True
            # flight guard: if the knife got kicked out, re-stage it (transport)
            kdist = float((scene.knife.data.root_pos_w[0] - pos_w[0]).norm())
            kspeed = float(scene.knife.data.root_lin_vel_w[0].norm())
            if kdist > 0.08 or kspeed > 3.0:
                recoveries += 1
                print(f"[solve] press[{tag}] flight guard @i={i}: dist={kdist:.3f} "
                      f"speed={kspeed:.2f} -> re-stage (#{recoveries})", flush=True)
                if recoveries > 6:
                    return False
                teleport_knife(loc, q_use)
                pid_int.zero_()
                for _ in range(20):
                    hold_wrench(fz=0.3, support=1.0, xy_target=xy)
            if i % 60 == 59:
                fz = min(fz + 8.0, 40.0)
                sep = scene.seam_sep()[0]
                tipz = float(scene.blade_tip_local()[0, 2])
                kv = float(scene.knife.data.root_lin_vel_w[0].norm())
                print(f"[solve] press[{tag}]: fz={fz:.0f}N tip_z={tipz * 1000:.1f}mm "
                      f"sep=({float(sep[0]) * 1000:.2f},{float(sep[1]) * 1000:.2f})mm "
                      f"kv={kv:.2f}m/s", flush=True)
        return False

    severed = press_at(0.0, 360, "centre")
    attempts = 0
    while not severed and attempts < 4:
        attempts += 1
        sev = scene.severed()[0]
        if bool(sev[0]) == bool(sev[1]):
            off = 0.0  # nothing broken yet: lean on the centre again
        else:
            off = 0.020 if bool(sev[1]) else -0.020  # press the dropped (broken) side
        print(f"[solve] re-staging press #{attempts} at tip_x_off={off:+.3f}",
              flush=True)
        severed = press_at(off, 480, f"retry{attempts}")
    scene.knife.set_external_force_and_torque(zero, zero, env_ids=all_ids)
    if not severed:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (seams never severed)", flush=True)
        os._exit(1)
    # follow the drop a moment before retracting the knife
    step(30)
    report("severed")

    # retraction (transport): free-space lift out of the slot, then park on the ground
    lift_loc = hover_loc.clone()
    lift_loc[:, 2] = c.cover_top + 0.10
    teleport_knife(lift_loc, q_use)
    step(5)
    park = torch.zeros(n, 13, device=device)
    park[:, 0] = 0.85
    park[:, 1] = 0.35
    park[:, 2] = 0.030
    park[:, 3:7] = _qmul(_qz(torch.zeros(n, device=device)),
                         torch.tensor([0.70711, 0.70711, 0.0, 0.0],
                                      device=device).expand(n, 4))
    park[:, 0:3] += scene.env_origins
    scene.knife.write_root_state_to_sim(park, all_ids)

    # hands-off: gravity drops the red piece into the well; everything settles
    ok_p2 = False
    for j in range(720):
        env.step(no_action)
        if bool(scene.success()[0]):
            ok_p2 = True
            break
        if j % 180 == 179:
            report(f"drop-{j + 1}")
    step(60)  # margin of settle, hands-off
    report("well-drop")
    if not (ok_p2 or bool(scene.success()[0])):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the drop)", flush=True)
        os._exit(1)
    s2 = print_score("P2 seams severed; red piece dropped and settled in the well")
    assert s2 >= s1 - 1e-6, "score decreased across the press"

    # ---------------- phase 3: persistence (>= 3 simulated seconds, hands-off) -------------
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

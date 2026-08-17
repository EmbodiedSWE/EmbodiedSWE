"""Teleport solution for CardClockScene (sim_gen task `change_clock_i181`) — the
task's legitimacy certificate.

The task's load-bearing interactions and how they are executed:

1. EXTRACT THE OLD CARD (contact dynamics): a velocity-limited vertical force at
   the displayed card's CoM slides it UP and out of the snug display slot along
   the slot walls (the fingertip-pull emulation of the arm's pinch-and-lift).
   The card is never teleported out of the seated state — it exits through the
   slot's own guidance under the applied pull, and the force is cut only when
   readback shows it fully clear of the mouth.
2. DISCARD (teleport = transport, then contact): the airborne extracted card is
   moved across free space to a hover ABOVE the blue tray, laid flat, released —
   it falls in, hits the tray floor and settles. Containment is judged on the
   settled physical pose.
3. FETCH THE TARGET CARD (contact dynamics): the same velocity-limited pull
   slides the matching card up and out of its RACK slot.
4. INSERT (teleport = transport, then contact): the target card is moved to a
   hover above the display slot mouth with a DELIBERATE 3 mm lateral offset and
   3-degree tilt (the seating must tolerate a realistic place, not a magic one),
   released, and pressed DOWN with a gentle velocity-limited force: the funnel
   and slot walls align it under contact as it descends, and it seats on the
   slot floor. It is never spawned seated — the scene's own `_seated_display`
   readback (pose + pips-up + stillness) must confirm the seating.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds
after success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds.

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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)",
                                             flush=True), os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.card_clock")().build(num_envs=args.num_envs,
                                               device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    from isaaclab.utils.math import quat_apply, quat_mul

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        loc, up_z, _still = scene._card_tensors()
        d = int(scene.displayed_idx[0])
        t = int(scene.target_idx[0])
        dl, tl = loc[0, d], loc[0, t]
        print(f"[solve] {tag:14s} | old(card_{c.hours[d]})="
              f"({float(dl[0]):+.3f},{float(dl[1]):+.3f},{float(dl[2]):.3f}) "
              f"target(card_{c.hours[t]})=({float(tl[0]):+.3f},"
              f"{float(tl[1]):+.3f},{float(tl[2]):.3f}) up_z={float(up_z[0, t]):+.2f} "
              f"seated={bool(scene._gather(scene._seated_display(), scene.target_idx)[0])} "
              f"old_in_tray={bool(scene._gather(scene._in_tray(), scene.displayed_idx)[0])} "
              f"latches=({bool(scene._extracted_ever[0])},"
              f"{bool(scene._tray_ever[0])},{bool(scene._seated_ever[0])}) "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def loc_of(body) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        rel = body.data.root_pos_w - scene.console.data.root_pos_w
        return quat_apply_inverse(scene.console.data.root_quat_w, rel)

    def pull_up_until(body, clear_z: float, tag: str) -> bool:
        """Velocity-limited vertical pull: slide the card up out of its slot
        through the slot-wall contacts. Returns True when the card's centre
        (console frame) is above clear_z; force is then cut."""
        f_up, v_des = 0.6, 0.25
        best = float(loc_of(body)[0, 2])
        last_bump = 0
        ok = False
        for i in range(600):
            z = float(loc_of(body)[0, 2])
            if z > clear_z:
                ok = True
                break
            vz = float(body.data.root_lin_vel_w[0, 2])
            f = f_up if vz < v_des else 0.0
            fw = torch.zeros(n, 1, 3, device=device)
            fw[:, 0, 2] = f
            body.set_external_force_and_torque(fw, zero_wrench, env_ids=all_ids,
                                               is_global=True)
            env.step(no_action)
            if z > best + 0.002:
                best, last_bump = z, i
            elif i - last_bump > 120:  # stalled: escalate the pull
                f_up = min(f_up + 0.4, 3.0)
                last_bump = i
                print(f"[solve] {tag}: stall at z={z:.3f} -> f_up={f_up:.1f} N",
                      flush=True)
        body.set_external_force_and_torque(zero_wrench, zero_wrench,
                                           env_ids=all_ids)
        return ok

    def transport(body, loc_xyz, extra_quat=None, hold_vel_zero: bool = True) -> None:
        """Teleport = TRANSPORT ONLY: one pose write in free space (console-frame
        target), zero velocity. Never writes a body into a seated/contained goal
        state."""
        s_pos = scene.console.data.root_pos_w
        s_quat = scene.console.data.root_quat_w
        loc = torch.tensor(loc_xyz, device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = s_pos + quat_apply(s_quat, loc)
        if extra_quat is not None:
            eq = torch.tensor(extra_quat, device=device).expand(n, 4)
            st[:, 3:7] = quat_mul(s_quat, eq)
        else:
            st[:, 3:7] = s_quat
        body.write_root_state_to_sim(st, all_ids)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    d = int(scene.displayed_idx[0])
    t = int(scene.target_idx[0])
    old_card, tgt_card = scene.cards[d], scene.cards[t]
    cp = (scene.console.data.root_pos_w - scene.env_origins)[0]
    cq = scene.console.data.root_quat_w[0]
    c_yaw = 2.0 * math.atan2(float(cq[3]), float(cq[0]))
    loc, _up, _still = scene._card_tensors()
    print(f"[solve] layout readback (seed {args.seed}): "
          f"console=({float(cp[0]):+.3f},{float(cp[1]):+.3f}) "
          f"yaw={math.degrees(c_yaw):+.1f}deg "
          f"displayed=card_{c.hours[d]} target=card_{c.hours[t]} "
          f"rack_x=[{','.join(f'{float(loc[0, i, 0]):+.2f}' for i in range(4))}]",
          flush=True)
    # sanity: authored masses actually landed (custom spawners apply no cfg schemas)
    for i, card in enumerate(scene.cards):
        mass = float(card.data.default_mass.sum())
        assert abs(mass - c.card_mass) < 1e-4, \
            f"card_{c.hours[i]} mass {mass} != {c.card_mass}"
    report("reset")
    assert bool(scene._gather(scene._seated_display(),
                              scene.displayed_idx)[0]), "old card not seated at reset"
    assert not bool(scene.success()[0]), "success at reset?!"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"reset score should be ~0, got {s0}"

    # ---------------- phase 1: EXTRACT the old card (contact dynamics) ----------------------
    # clear height: above the slot mouth AND above the 0.18 display-zone ceiling
    # (the extraction latch fires when the card leaves that zone)
    assert pull_up_until(old_card, 0.195, "extract-old"), \
        "old card never cleared the display slot"
    report("extracted")
    assert bool(scene._extracted_ever[0]), "extraction did not latch"
    s1 = print_score("P1 old card pulled up out of the display slot")
    assert s1 >= max(s0, c.w_extract) - 1e-6, "extraction credit missing"

    # ---------------- phase 2: DISCARD into the tray (transport + drop) ---------------------
    # lay flat (rotate 90 deg about the card's width axis) above the tray, release
    q_flat = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)
    transport(old_card, (c.tray_x, c.tray_y, 0.13), extra_quat=q_flat)
    in_tray = False
    for _ in range(360):
        env.step(no_action)
        if bool(scene._gather(scene._in_tray(), scene.displayed_idx)[0]):
            in_tray = True
            break
    step(30)
    report("discarded")
    assert in_tray and bool(scene._tray_ever[0]), \
        "old card did not settle inside the tray"
    s2 = print_score("P2 old card dropped into the tray and settled")
    assert s2 >= s1 - 1e-6 and s2 >= c.w_extract + c.w_tray - 1e-6, \
        "tray credit missing"

    # ---------------- phase 3: FETCH the target card from the rack --------------------------
    assert pull_up_until(tgt_card, c.rack_seat_z + c.card_h / 2 + 0.015,
                         "fetch-target"), "target card never cleared its rack slot"
    report("fetched")
    s3 = print_score("P3 target card pulled up out of its rack slot")
    assert s3 >= s2 - 1e-6, "score decreased across the fetch"

    # ---------------- phase 4: INSERT into the display slot (contact dynamics) --------------
    # Hover above the mouth with a deliberate 3 mm lateral offset + 3 deg tilt:
    # the funnel and slot walls must do the final alignment under contact.
    tilt = math.radians(3.0)
    q_tilt = (math.cos(tilt / 2), math.sin(tilt / 2), 0.0, 0.0)
    hover_z = c.disp_mouth + c.card_h / 2 + 0.020
    transport(tgt_card, (c.stand_x, c.stand_y + 0.003, hover_z),
              extra_quat=q_tilt)
    step(2)
    assert not bool(scene._gather(scene._seated_display(),
                                  scene.target_idx)[0]), \
        "hover pose already reads as seated (teleport must not seat the card)"
    # gentle velocity-limited press until the scene reads it seated
    seated = False
    f_dn, v_des = 0.35, 0.40
    for i in range(600):
        if bool(scene._gather(scene._seated_display(), scene.target_idx)[0]):
            seated = True
            break
        vz = float(tgt_card.data.root_lin_vel_w[0, 2])
        f = f_dn if vz > -v_des else 0.0
        fw = torch.zeros(n, 1, 3, device=device)
        fw[:, 0, 2] = -f
        tgt_card.set_external_force_and_torque(fw, zero_wrench, env_ids=all_ids,
                                               is_global=True)
        env.step(no_action)
    tgt_card.set_external_force_and_torque(zero_wrench, zero_wrench,
                                           env_ids=all_ids)
    if not seated:  # wait out any residual rattle with no force applied
        for _ in range(240):
            env.step(no_action)
            if bool(scene._gather(scene._seated_display(), scene.target_idx)[0]):
                seated = True
                break
    report("inserted")
    assert seated and bool(scene._seated_ever[0]), \
        "target card did not seat in the display slot under the press"
    s4 = print_score("P4 target card pressed down and seated in the display slot")
    assert s4 >= s3 - 1e-6 and s4 >= 0.75, "seating credit missing"

    # ---------------- phase 5: settle + verdict ---------------------------------------------
    step(60)
    report("settled")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after settling)", flush=True)
        os._exit(1)
    s5 = print_score("P5 goal state settled (success)")
    assert s5 >= s4 - 1e-6

    # ---------------- phase 6: persistence (>= 3.3 simulated seconds, hands-off) ------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s6 = print_score("P6 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s6 >= s5 - 1e-6
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
    except Exception as exc:  # noqa: BLE001 — die fast, Kit teardown hangs
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({exc})", flush=True)
        os._exit(1)

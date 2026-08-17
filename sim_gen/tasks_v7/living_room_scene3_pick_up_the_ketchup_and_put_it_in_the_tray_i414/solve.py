"""Teleport solution for PryLidVaultScene (sim_gen task
`living_room_scene3_pick_up_the_ketchup_and_put_it_in_the_tray_i414`) — the task's
legitimacy certificate.

The load-bearing interactions and how each is executed:

1. INSERT THE PRY BAR (held carry + guided contact): a single root-state write
   carries the bar from its table slot to free air on the SLOT AXIS, tip 10 mm
   outside the chest's front face (transport of a plainly graspable free tool; the
   write satisfies no rubric clause — asserted). From there a gravity-compensating
   PD hold plus a velocity-capped forward drive slides the bar INTO the slot; the
   slot floor, cheeks and the seated lid guide it by contact. FORCE-FRAME GUARD:
   some pods rotate an applied "global" wrench by the body's rotation since a
   reference instant; the drive PROBES the convention at runtime and toggles
   `encode_force` mode if the bar moves the wrong way.
2. PRY THE LID (applied torque + lever contact): with the tip ~70 mm under the lid,
   a ramped pitch torque presses the bar's protruding tail DOWN; the bar levers
   over the slot's outer sill and its tip lifts the 1.2 kg lid's front edge proud
   of the rim. The `pried` credential (edge >= 6 mm proud while the bar tip is
   under the lid) is produced entirely by the lever contact — never written.
3. REMOVE THE LID (teleport = the grasp): only while the pry visibly HOLDS the
   edge proud (asserted live at the write) is the lid graspable at all; the
   teleport carries it — as a gripper would — to flat open table fully outside the
   chest footprint, zero velocity. The `lid_off` credit then latches from the
   settled resting pose, and only because `pried` was earned first.
4. WITHDRAW THE BAR (applied force + contact): the same bang-bang drive pulls the
   bar back out of the slot until it lies on the table clear of the cavity.
5. PLACE THE KETCHUP (teleport + gravity): a root-state write carries the KETCHUP
   bottle to free air ABOVE the open cavity (release point outside the in-cavity
   volume — asserted), and gravity drops it 70+ mm onto the cavity floor. The
   `in_cav` credit is produced by ballistics and contact.
6. IDENTITY: only the KETCHUP bottle is ever touched. Mustard and mayo stay put.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.living_room_scene3_pick_up_the_ketchup_and_put_it_in_the_tray_i414.solve --headless [--seed N]
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
    from .scene import encode_force
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scene import encode_force

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.pry_lid_vault")().build(num_envs=args.num_envs,
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

    def bar_loc() -> torch.Tensor:
        return scene._chest_local(scene.bar.data.root_pos_w)[0]

    def edge_z() -> float:
        return float(scene._chest_local(scene._lid_edge_w())[0, 2])

    def clear_bar() -> None:
        scene.bar.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                env_ids=all_ids)

    def report(tag: str) -> None:
        bl = bar_loc()
        ll = scene.lid_loc()[0]
        kl = scene._chest_local(scene.ketchup.data.root_pos_w)[0]
        print(f"[solve] {tag:14s} | bar=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):+.3f}) lid=({float(ll[0]):+.3f},{float(ll[1]):+.3f},"
              f"{float(ll[2]):+.3f}) edge_z={edge_z():+.3f} "
              f"ketchup=({float(kl[0]):+.3f},{float(kl[1]):+.3f},{float(kl[2]):+.3f}) "
              f"pried={bool(scene._pried[0])} lid_off={bool(scene._lid_off[0])} "
              f"in_cav={bool(scene.ketchup_in_cavity()[0])} "
              f"bar_clear={bool(scene.bar_clear()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)  # lid seats flush in the rabbet, bar + bottles settle on the ground
    # custom spawn funcs ignore cfg mass schemas — assert the authored masses took
    cm = float(scene.chest.root_physx_view.get_masses().sum())
    lm = float(scene.lid.root_physx_view.get_masses().sum())
    bm = float(scene.bar.root_physx_view.get_masses().sum())
    print(f"[solve] mass readback: chest={cm:.1f} kg lid={lm:.3f} kg bar={bm:.3f} kg",
          flush=True)
    assert 20.0 < cm < 30.0, f"chest mass wrong: {cm}"
    assert 1.0 < lm < 1.4, f"lid mass wrong: {lm}"
    assert 0.08 < bm < 0.16, f"bar mass wrong: {bm}"
    cp = (scene.chest.data.root_pos_w - scene.env_origins)[0]
    cq = scene.chest.data.root_quat_w[0]
    cyaw = math.degrees(2.0 * math.atan2(float(cq[3]), float(cq[0])))
    print(f"[solve] layout readback (seed {args.seed}): "
          f"chest=({float(cp[0]):+.3f},{float(cp[1]):+.3f}) yaw={cyaw:+.1f}deg "
          f"ketchup_slot={int(scene.ketchup_slot[0])}", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    ll = scene.lid_loc()[0]
    assert abs(float(ll[0])) < 0.01 and abs(float(ll[1])) < 0.01 \
        and 0.060 < float(ll[2]) < 0.072, f"lid must start seated, got {ll.tolist()}"
    assert edge_z() < c.rim_top + 0.002, "lid must start flush (edge not proud)"
    assert not bool(scene.ketchup_in_cavity()[0]), "ketchup must start outside"
    s0 = print_score("P0 reset+settle (lid seated flush, bar + bottles on the table)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    ch_q = scene.chest.data.root_quat_w
    x_hat = quat_apply(ch_q, torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))
    y_hat = quat_apply(ch_q, torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3))
    state = {"mode": 0, "q_ref": scene.bar.data.root_quat_w.clone()}

    def enc(f_world: torch.Tensor) -> torch.Tensor:
        return encode_force(state["mode"], state["q_ref"],
                            scene.bar.data.root_quat_w, f_world).view(n, 1, 3)

    # ---------------- phase 1: insert the pry bar through the slot --------------------------
    # TRANSPORT: write the bar to free air on the slot axis, tip 10 mm outside the
    # chest's front face, level, square to the slot, zero velocity.
    Z_CARRY = 0.0550  # slot spans z 0.050..0.060; carry the 5 mm bar dead centre
    z_unit = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)

    def teleport_to_mouth() -> None:
        ch_p = scene.chest.data.root_pos_w
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0] = c.outer_half + 0.010 + c.bar_size[0] / 2  # tip 10 mm outside
        loc[:, 2] = Z_CARRY
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = ch_p + quat_apply(ch_q, loc)
        st[:, 3:7] = ch_q  # bar +x along chest +x -> the -x end is the tip
        scene.bar.write_root_state_to_sim(st, all_ids)
        env.iscene.update(0.0)
        state["q_ref"] = scene.bar.data.root_quat_w.clone()
        assert bool(scene.bar_clear()[0]), "bar write must be outside the cavity"
        assert not bool(scene.pried_now()[0]), "bar write must not fire pried"

    def pd_hold(z_ref: float, f_fwd: float) -> None:
        """One step of the stiff held carry (gravity-compensating PD on chest-frame
        y/z, kp 600 / kd 50) plus a forward force along -x_chest."""
        bl = bar_loc()
        v_w = scene.bar.data.root_lin_vel_w
        vz = float(v_w[0, 2])
        vy = float((v_w[0] * y_hat[0]).sum())
        a_z = 9.81 + 600.0 * (z_ref - float(bl[2])) - 50.0 * vz
        a_y = 600.0 * (0.0 - float(bl[1])) - 50.0 * vy
        f = (bm * a_z) * z_unit + (bm * a_y) * y_hat - f_fwd * x_hat
        f = f.clamp(min=-6.0, max=6.0)
        scene.bar.set_external_force_and_torque(enc(f), zero_wrench,
                                                env_ids=all_ids, is_global=True)
        env.step(no_action)

    def insert_bar(max_steps: int) -> bool:
        """Held carry into the slot: hover-stabilize at the mouth, then a gentle
        velocity-capped forward drive. On a stall the bar PECKS — micro-retracts
        and re-approaches with the carry height nudged — a hard push would only
        friction-pin the tip on the wall face (mu*N beats any PD authority)."""
        for _ in range(40):  # converge the hover BEFORE touching anything
            pd_hold(Z_CARRY, 0.0)
        z_ref = Z_CARRY
        fwd = 0.6
        retract = 0
        retracted_in_window = False
        pecks = 0
        nudge = (0.0, +0.0015, -0.0015, +0.0025)
        probe_x = float(bar_loc()[0])
        done = False
        for i in range(max_steps):
            bl = bar_loc()
            if float(bl[0]) < 0.120:
                done = True
                break
            v_fwd = -float((scene.bar.data.root_lin_vel_w[0] * x_hat[0]).sum())
            if retract > 0:
                f_fwd = -0.8
                retract -= 1
                retracted_in_window = True
            else:
                f_fwd = fwd if v_fwd < 0.06 else (0.2 * fwd if v_fwd < 0.12 else -0.1)
            pd_hold(z_ref, f_fwd)
            if i % 45 == 44:
                x_new = float(bar_loc()[0])
                if x_new > probe_x + 0.006 and not retracted_in_window:
                    state["mode"] ^= 1
                    state["q_ref"] = scene.bar.data.root_quat_w.clone()
                    print(f"[solve] insert moved the bar the WRONG way "
                          f"(x {probe_x:+.3f} -> {x_new:+.3f}); "
                          f"force-frame mode -> {state['mode']}", flush=True)
                elif x_new > probe_x - 0.002 and not retracted_in_window:
                    pecks += 1
                    z_ref = Z_CARRY + nudge[pecks % 4]
                    fwd = min(fwd + 0.2, 1.4)
                    retract = 18
                    print(f"[solve] insert stalled at x={x_new:+.3f}; peck "
                          f"{pecks}: retract + carry z -> {z_ref:.4f}", flush=True)
                probe_x = x_new
                retracted_in_window = False
        clear_bar()
        return done

    inserted = False
    for attempt in range(3):
        teleport_to_mouth()
        if insert_bar(900):
            inserted = True
            break
        print(f"[solve] insert attempt {attempt + 1} failed; retrying", flush=True)
    if not inserted:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (bar never reached insertion depth)", flush=True)
        os._exit(1)
    step(30)  # hands-off: the bar rests in the slot (tail-heavy, tip wedges up)
    report("bar-inserted")
    bl = bar_loc()
    assert abs(float(bl[1])) < 0.012, f"bar off slot axis: y={float(bl[1]):+.3f}"
    assert not bool(scene.pried_now()[0]), "gravity wedge alone must not fire pried"
    s1 = print_score("P1 pry bar inserted through the slot")
    assert s1 <= 0.03, f"insertion alone must not score, got {s1}"

    # ---------------- phase 2: pry the lid proud of the rim ---------------------------------
    # Ramped pitch torque about chest +y presses the tail down; a small -x hold
    # force keeps the bar from squirting back out of the slot.
    pry_target = c.rim_top + c.pried_proud + 0.004  # 4 mm past the gate
    pry_state = {"tau": 0.15}

    def apply_pry() -> None:
        t_vec = pry_state["tau"] * y_hat
        f_vec = -1.5 * x_hat
        scene.bar.set_external_force_and_torque(enc(f_vec), enc(t_vec),
                                                env_ids=all_ids, is_global=True)
        env.step(no_action)

    def pry(max_steps: int) -> bool:
        pry_state["tau"] = 0.15
        tip_z0 = None
        for i in range(max_steps):
            if edge_z() > pry_target:
                return True
            apply_pry()
            e0, e1 = scene._bar_ends_w()
            tips = [scene._chest_local(e)[0] for e in (e0, e1)]
            tip = min(tips, key=lambda t: float(t[0]))  # inboard end
            if tip_z0 is None:
                tip_z0 = float(tip[2])
            if i % 30 == 29:
                if float(tip[2]) < tip_z0 - 0.004:
                    # torque pitched the tip DOWN -> wrong frame convention
                    state["mode"] ^= 1
                    state["q_ref"] = scene.bar.data.root_quat_w.clone()
                    tip_z0 = float(tip[2])
                    print(f"[solve] pry torque acted the WRONG way; "
                          f"force-frame mode -> {state['mode']}", flush=True)
                else:
                    pry_state["tau"] = min(pry_state["tau"] + 0.08, 3.0)
            if float(tip[0]) > 0.12:  # ejected from the slot
                return False
        return False

    def hold_pry(steps: int) -> None:
        """Keep the breakaway wrench ON at constant torque: the lever pitches
        until the slot's own angular stop (bar spanning the 20 mm wall exceeds the
        10 mm slot height at ~20 deg) arrests it — a stable mechanical hold with
        the lid edge ~30 mm proud and the tip still under the lid."""
        for _ in range(steps):
            apply_pry()

    pried = False
    for attempt in range(3):
        if pry(700):
            pried = True
            break
        clear_bar()
        print(f"[solve] pry attempt {attempt + 1} failed; re-inserting", flush=True)
        teleport_to_mouth()
        if not insert_bar(900):
            break
        step(30)
    if not pried:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (lid never pried proud)", flush=True)
        os._exit(1)
    # hold the pry at a servoed torque so the latch sees a sustained lever lift
    hold_pry(60)
    report("pried")
    assert bool(scene._pried[0]), "pried latch must be set under the held lever"
    assert bool(scene.pried_now()[0]), "pry must be holding live at the grasp"
    assert edge_z() > c.rim_top + c.pried_proud, "edge must be proud at the grasp"
    s2 = print_score("P2 lid front edge levered proud of the rim (held)")
    assert s2 >= s1 - 1e-6 and s2 >= c.w_pried - 1e-6, \
        f"P2 score {s2} (expect pried={c.w_pried})"

    # ---------------- phase 3: grasp the raised edge, set the lid on the table --------------
    # TRANSPORT: the pried edge stands proud (asserted just above) — that raised
    # 12 mm plate edge is the grasp a parallel jaw takes. One root-state write
    # carries the lid to flat open table fully outside the chest footprint.
    ch_p = scene.chest.data.root_pos_w
    loc = torch.zeros(n, 3, device=device)
    loc[:, 0], loc[:, 1] = -0.02, -0.30
    loc[:, 2] = c.lid_size[2] / 2 + 0.002
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = ch_p + quat_apply(ch_q, loc)
    st[:, 3:7] = ch_q
    scene.lid.write_root_state_to_sim(st, all_ids)
    env.iscene.update(0.0)
    clear_bar()  # release the lever; the bar drops back onto the slot floor
    step(120)  # hands-off settle: lid flat on the table, bar at rest in the slot
    report("lid-off")
    assert bool(scene.lid_off_now()[0]), "lid must rest flat on the table, clear"
    assert bool(scene._lid_off[0]), "lid_off latch must be set (pried was earned)"
    assert not bool(scene.success()[0]), "no success yet (cavity still empty)"
    s3 = print_score("P3 lid set flat on the table, clear of the chest")
    assert s3 >= s2 - 1e-6 and s3 >= c.w_pried + c.w_lid_off - 1e-6, \
        f"P3 score {s3} (expect {c.w_pried + c.w_lid_off})"

    # ---------------- phase 4: withdraw the bar from the slot -------------------------------
    def withdraw_bar(max_steps: int) -> bool:
        probe_x = float(bar_loc()[0])
        fwd = 0.9
        for i in range(max_steps):
            bl = bar_loc()
            if float(bl[0]) > 0.230:
                clear_bar()
                return True
            v_w = scene.bar.data.root_lin_vel_w
            v_out = float((v_w[0] * x_hat[0]).sum())
            f_fwd = fwd if v_out < 0.10 else (0.3 * fwd if v_out < 0.20 else 0.0)
            f = f_fwd * x_hat
            scene.bar.set_external_force_and_torque(enc(f), zero_wrench,
                                                    env_ids=all_ids, is_global=True)
            env.step(no_action)
            if i % 45 == 44:
                x_new = float(bar_loc()[0])
                if x_new < probe_x - 0.004:
                    state["mode"] ^= 1
                    state["q_ref"] = scene.bar.data.root_quat_w.clone()
                    print(f"[solve] withdraw moved the bar the WRONG way; "
                          f"force-frame mode -> {state['mode']}", flush=True)
                elif x_new < probe_x + 0.002:
                    fwd = min(fwd + 0.4, 3.0)
                    print(f"[solve] withdraw stalled at x={x_new:+.3f}; "
                          f"drive -> {fwd:.1f} N", flush=True)
                probe_x = x_new
        clear_bar()
        return False

    if not withdraw_bar(900):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (bar never withdrew from the slot)", flush=True)
        os._exit(1)
    step(90)  # hands-off: the bar falls flat onto the table in front of the chest
    report("bar-out")
    assert bool(scene.bar_clear()[0]), "bar must be clear of the cavity"
    s4 = print_score("P4 bar withdrawn, resting on the table")
    assert s4 >= s3 - 1e-6, "score decreased across the withdrawal"

    # ---------------- phase 5: place the ketchup bottle into the open cavity ----------------
    # TRANSPORT: write the bottle to free air above the open cavity, upright, zero
    # velocity. The release point is above the in-cavity volume (asserted) —
    # gravity does the placement.
    loc = torch.zeros(n, 3, device=device)
    loc[:, 2] = c.rim_top + c.bottle_size[2] / 2 + 0.012
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.chest.data.root_pos_w + quat_apply(ch_q, loc)
    st[:, 3:7] = ch_q
    scene.ketchup.write_root_state_to_sim(st, all_ids)
    env.iscene.update(0.0)
    assert not bool(scene.ketchup_in_cavity()[0]), \
        "release point must be OUTSIDE the in-cavity volume"
    assert not bool(scene._in_cav[0]), "in_cav latch must not fire at the write"
    got = False
    for j in range(600):
        env.step(no_action)
        if bool(scene.success()[0]):
            got = True
            break
        if j % 180 == 179:
            report(f"drop-{j + 1}")
    step(60)  # margin of settle, hands-off
    report("ketchup-in")
    if not (got or bool(scene.success()[0])):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the placement)", flush=True)
        os._exit(1)
    assert bool(scene._in_cav[0]), "in_cav latch must be set after the drop"
    s5 = print_score("P5 ketchup placed in the open cavity")
    assert s5 >= s4 - 1e-6, "score decreased across the placement"

    # ---------------- phase 6: persistence (>= 3 simulated seconds, hands-off) -------------
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
    except Exception:  # noqa: BLE001 - fail FAST; a hung Kit burns the forge slot
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)

"""Teleport solution for PackedToteScene (sim_gen task
`living_room_scene2_pick_up_the_alphabet_soup_and_put_it_in_the_basket_i327`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY: the blue can is teleported with zero velocity to
poses a pick-and-carry delivers (a hover over the tote, a set-aside spot on the floor,
a lowered release pose inside the tote's open top). Everything the rubric reads
happens through contact dynamics:

  drop      — the seed's own strategy, executed verbatim and shown to fail: the can is
              released over the packed tote's center and gravity lands it STANDING ON
              TOP OF THE RED BLOCKS, a stable perch ~6 cm above the floor. The rubric's
              floor z-window rejects it (partial credit only — the can entered the
              tote).
  set-aside — the can is carried back out to the floor (transport teleport): a free
              body cannot hover while the tote is re-packed.
  shove     — an applied force on the TRAILING red block (a proxy for a fingertip
              push into the >= 28 mm gap beside it) slides the pair along the tote
              floor against sliding friction — the trail block pushes the lead block
              through face contact — until the lead block reaches the end wall and a
              conservative span readout certifies a can-sized patch of open floor.
              The force is a feedforward + velocity servo (ff ~ the 3.5 N two-block
              friction budget, escalating on stall), world-encoded via `encode_force`
              with a runtime force-frame probe and snapshot rollback.
  seat      — the can is lowered upright into the cleared slot (release pose 6 mm
              above the floor, inside the open top — the interior leaves >= 25 mm per
              side for fingers gripping across the can's x diameter) and RELEASED;
              gravity seats its base on the tote floor. Both blocks remain stowed.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing:
0 -> 0 carry -> 0.15 perched -> 0.15 set-aside -> 0.40 room made -> 1.0 seated), then
holds HANDS-OFF for >= 3 simulated seconds after success() first turns True and prints
`SIM_GEN_SOLVE: SUCCESS` only if it still holds.

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
    from .scene import BLK_W, IN_Y, L_CAN, R_CAN, RIM_Z, Z_F, _qapply, _qz, encode_force  # noqa: F401
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import BLK_W, IN_Y, L_CAN, R_CAN, RIM_Z, Z_F, _qapply, _qz, encode_force  # noqa: F401
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

HOVER_Z = RIM_Z + L_CAN / 2 + 0.010  # can-center hover above the rim (drop release)
SEAT_DROP = 0.006  # release height of the can base above the floor at the seat
W_HALF = IN_Y / 2


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.packed_tote")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the statement on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def span() -> float:
        return float(scene.cleared_span()[0])

    def blue_loc() -> torch.Tensor:
        return scene.tote_local(scene.blue.data.root_pos_w)[0]

    def blk_y(body) -> float:
        return float(scene.tote_local(body.data.root_pos_w)[0, 1])

    def clear_forces(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def all_settled() -> bool:
        return bool((scene.settled(scene.blue) & scene.settled(scene.blk_a)
                     & scene.settled(scene.blk_b) & scene.settled(scene.tote))[0])

    def report(tag: str) -> None:
        loc = blue_loc()
        ax = scene.can_axis_local()[0]
        print(f"[solve] {tag:12s} | blue_tote=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
              f"z-Zf={float(loc[2]) - Z_F:+.3f}) ax_z={float(ax[2]):+.3f} "
              f"span={span() * 1000:5.1f}mm ya={blk_y(scene.blk_a):+.3f} "
              f"yb={blk_y(scene.blk_b):+.3f} in_tote={bool(scene.in_tote(scene.blue)[0])} "
              f"slot={bool(scene.can_in_slot()[0])} "
              f"stowed={bool(scene.blocks_stowed()[0])} "
              f"L1={bool(scene.l1_in_tote[0])} L2={bool(scene.l2_room[0])} "
              f"settled={all_settled()} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def wait_settled(max_steps: int = 600) -> None:
        for _ in range(max_steps // 30):
            step(30)
            if all_settled():
                break

    def teleport_can(pos_local, upright: bool = True) -> None:
        """Transport-only teleport of the blue can to a tote-frame pose, zero vel."""
        q_tote = scene.tote.data.root_quat_w
        local = torch.tensor(pos_local, device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.tote.data.root_pos_w + _qapply(q_tote, local)
        st[:, 3:7] = q_tote
        scene.blue.write_root_state_to_sim(st, all_ids)

    def teleport_can_world(pos_w) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = _qz(torch.zeros(n, device=device))
        scene.blue.write_root_state_to_sim(st, all_ids)

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    wait_settled(360)
    tote_p = (scene.tote.data.root_pos_w - scene.env_origins)[0]
    blue_p = (scene.blue.data.root_pos_w - scene.env_origins)[0]
    yaw = 2.0 * math.atan2(float(scene.tote.data.root_quat_w[0, 3]),
                           float(scene.tote.data.root_quat_w[0, 0]))
    print(f"[solve] layout readback (seed {args.seed}): "
          f"tote=({float(tote_p[0]):+.3f},{float(tote_p[1]):+.3f}) "
          f"yaw={math.degrees(yaw):+.0f}deg ya={blk_y(scene.blk_a):+.3f} "
          f"yb={blk_y(scene.blk_b):+.3f} blue=({float(blue_p[0]):+.3f},"
          f"{float(blue_p[1]):+.3f}) span={span() * 1000:.1f}mm", flush=True)
    report("reset")
    assert span() < c.room_span_min, \
        f"reset must be BLOCKED (span {span() * 1000:.1f}mm >= gate) — no free ride"
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    s0 = print_score("P0 reset+settle (pair parked mid-tote, can outside)")
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"
    park_w = scene.blue.data.root_pos_w[0].clone()  # the can's own floor spot

    # ---------------- phase 1: TRANSPORT ONLY — carry the can over the tote ----------------
    teleport_can((0.0, 0.0, HOVER_Z))
    report("carry")
    s1 = print_score("P1 carried over the tote (teleport transport, zero velocity)")
    assert s1 >= s0 - 1e-6, "score decreased across the carry"

    # ---------------- phase 2: hands-off drop -> PERCHED on the blocks ---------------------
    perched = False
    for attempt in range(4):
        wait_settled(600)
        loc = blue_loc()
        ax = scene.can_axis_local()[0]
        if bool(scene.in_tote(scene.blue)[0]) and float(loc[2]) - Z_F > 0.090 \
                and float(ax[2]) > 0.85:
            perched = True
            break
        print(f"[solve] drop attempt {attempt}: not a clean perch "
              f"(z-Zf={float(loc[2]) - Z_F:+.3f} ax_z={float(ax[2]):+.3f}) — re-dropping",
              flush=True)
        teleport_can((0.0, 0.0, HOVER_Z))
    report("perched")
    assert perched, "can failed to land perched on the block pair"
    assert not bool(scene.can_in_slot()[0]), "perched can must NOT read as floor-seated"
    assert not bool(scene.success()[0]), \
        "the casual drop must NOT be success (this is the seed's strategy, rejected)"
    s2 = print_score("P2 dropped -> standing ON the blocks (gravity + contact; the "
                     "naive end state, not success)")
    assert s2 >= s1 - 1e-6, "score decreased across the drop"

    # ---------------- phase 3: set the can aside (transport teleport) ----------------------
    teleport_can_world(park_w + torch.tensor([0.0, 0.0, 0.002], device=device))
    wait_settled(240)
    report("set-aside")
    s3 = print_score("P3 can set back down on the floor (transport; tote gets re-packed)")
    assert s3 >= s2 - 1e-6, "score decreased across the set-aside"

    # ---------------- phase 4: shove the block pair to one end -----------------------------
    # Push toward the end the pair is already offset toward (shorter travel).
    pair_mid = 0.5 * (blk_y(scene.blk_a) + blk_y(scene.blk_b))
    s_dir = 1.0 if pair_mid >= 0.0 else -1.0
    if blk_y(scene.blk_a) * s_dir < blk_y(scene.blk_b) * s_dir:
        trail, lead = scene.blk_a, scene.blk_b
        trail_nm, lead_nm = "blk_a", "blk_b"
    else:
        trail, lead = scene.blk_b, scene.blk_a
        trail_nm, lead_nm = "blk_b", "blk_a"
    print(f"[solve] shove: dir s={s_dir:+.0f} trail={trail_nm} lead={lead_nm} "
          f"(ff~{c.shove_force:.1f}N budget)", flush=True)
    snap = scene.get_state(all_ids)

    def blk_flat(body) -> bool:
        ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
        up = _qapply(body.data.root_quat_w, ez)[0]
        return float(up[2]) > math.cos(math.radians(20.0))

    def shove(mode: int, q_ref: torch.Tensor, tag: str) -> bool:
        """Feedforward + velocity-servo push on the trail block along tote +s*y.
        Escalates the feedforward on stall (squaring grinds); aborts on a tipped
        block or no progress (wrong force-frame mode). Exits on the span readback."""
        y0 = blk_y(trail)
        ff, v_des, kp, f_max = 3.0, 0.06, 30.0, 8.0
        last_y, last_i = y0, 0
        for i in range(2400):
            q_tote = scene.tote.data.root_quat_w
            y_dir_w = _qapply(q_tote, torch.tensor([0.0, s_dir, 0.0],
                                                   device=device).expand(n, 3))
            v_s = float((trail.data.root_lin_vel_w * y_dir_w).sum(dim=-1)[0])
            F = max(0.0, min(ff + kp * (v_des - v_s), f_max))
            f_world = y_dir_w * F
            q_now = trail.data.root_quat_w
            f_arg = encode_force(mode, q_ref, q_now, f_world)
            trail.set_external_force_and_torque(
                f_arg.view(n, 1, 3), zero_wrench, env_ids=all_ids, is_global=True)
            env.step(no_action)
            y_now = blk_y(trail)
            if i % 240 == 239:
                print(f"[solve] {tag} @{i + 1}: F={F:4.1f}N ff={ff:.1f} "
                      f"y_trail={y_now:+.4f} y_lead={blk_y(lead):+.4f} "
                      f"span={span() * 1000:5.1f}mm", flush=True)
            # wrong-mode probe: no progress after 1.5 s at base feedforward
            if i == 179 and (y_now - y0) * s_dir < 0.002:
                clear_forces(trail)
                print(f"[solve] {tag}: no progress after 1.5 s "
                      f"(dy={(y_now - y0) * s_dir * 1000:+.1f}mm) — wrong force-frame "
                      f"mode or jammed", flush=True)
                return False
            # stall escalation (memory: squaring grinds need a higher cap)
            if (y_now - last_y) * s_dir > 0.003:
                last_y, last_i = y_now, i
            elif i - last_i > 240 and ff < 6.0:
                ff += 1.0
                last_i = i
                print(f"[solve] {tag}: stall — escalating ff to {ff:.1f}N", flush=True)
            if not (blk_flat(trail) and blk_flat(lead)):
                clear_forces(trail)
                print(f"[solve] {tag}: a block tipped — aborting this attempt", flush=True)
                return False
            if span() >= c.room_span_min + 0.004:
                clear_forces(trail)
                return True
        clear_forces(trail)
        print(f"[solve] {tag}: shove budget exhausted (span {span() * 1000:.1f}mm)",
              flush=True)
        return False

    shoved = False
    for mode, tag in ((0, "shove[m0/world]"), (1, "shove[m1/start-ref]")):
        q_ref = trail.data.root_quat_w.clone()
        shoved = shove(mode, q_ref, tag)
        if shoved:
            print(f"[solve] {tag}: room made (span {span() * 1000:.1f}mm)", flush=True)
            break
        scene.set_state(snap, all_ids)
        step(60)  # re-settle the rolled-back parked state
    assert shoved, "shove failed in every force-frame configuration"

    # hands off; wait for the blocks to settle and the L2 latch to certify the room
    for _ in range(20):
        step(30)
        if bool(scene.l2_room[0]):
            break
    report("room-made")
    assert bool(scene.l2_room[0]), "L2 room latch did not fire after the shove settled"
    assert bool(scene.blocks_stowed()[0]), "blocks must still be stowed after the shove"
    s4 = print_score("P4 pair shoved to the end wall — can-sized floor span cleared "
                     "(contact dynamics, blocks stowed)")
    assert s4 >= s3 - 1e-6, "score decreased across the shove"

    # ---------------- phase 5: lower the can upright into the cleared slot -----------------
    def slot_center_y() -> tuple[float, float]:
        """Free-interval readback on the -s side: [wall, nearest block edge]."""
        edges = []
        for body in (scene.blk_a, scene.blk_b):
            y = blk_y(body)
            h = float(scene.blk_y_halfwidth(body)[0])
            edges.append(y - h if s_dir > 0 else y + h)
        if s_dir > 0:
            lo, hi = -W_HALF, min(edges)
        else:
            lo, hi = max(edges), W_HALF
        width = hi - lo
        center = 0.5 * (lo + hi)
        # keep the can >= 2.5 mm clear of both the wall and the block edge
        center = max(lo + R_CAN + 0.0025, min(hi - R_CAN - 0.0025, center))
        return center, width

    seated = False
    for attempt in range(5):
        y_c, width = slot_center_y()
        print(f"[solve] seat attempt {attempt}: slot y={y_c:+.4f} "
              f"width={width * 1000:.1f}mm", flush=True)
        if width < 2 * R_CAN + 0.005:
            print("[solve] slot too narrow — topping up the shove", flush=True)
            q_ref = trail.data.root_quat_w.clone()
            shove(0, q_ref, "shove[top-up]")
            step(120)
            continue
        teleport_can((0.0, y_c, Z_F + L_CAN / 2 + SEAT_DROP))
        wait_settled(600)
        if bool(scene.can_in_slot()[0]) and bool(scene.blocks_stowed()[0]):
            seated = True
            break
        report(f"seat-retry{attempt}")
        teleport_can_world(park_w + torch.tensor([0.0, 0.0, 0.002], device=device))
        wait_settled(240)
    assert seated, "can failed to seat upright on the tote floor"
    for _ in range(12):  # up to 3 s extra hands-off settling for success to fire
        if bool(scene.success()[0]):
            break
        step(30)
    report("seated")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after seating)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)
    s5 = print_score("P5 can released into the cleared slot — standing on the tote "
                     "floor (gravity seats it)")
    assert s5 >= s4 - 1e-6, "score decreased across the seat"

    # ---------------- persistence (>= 3 simulated seconds, no intervention) ----------------
    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                lv = float(scene.blue.data.root_lin_vel_w[0].norm())
                print(f"[solve] persist flicker @step {i}: "
                      f"slot={bool(scene.can_in_slot()[0])} "
                      f"stowed={bool(scene.blocks_stowed()[0])} blue_lin={lv:.4f} "
                      f"settled={all_settled()}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s6 = print_score("P-persist persistence 3.3 s (still seated, blocks stowed)")
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)

"""Teleport solution for CubbyRakeStoveScene (sim_gen task
`libero_kitchen_scene8_put_the_right_moka_pot_on_the_stove_i329`) — the task's
legitimacy certificate.

The load-bearing interactions and how each is executed:

1. RAKE WIELDING (external wrench = the hand): a 6-DOF gravity-feedforward +
   PD wrench holds the rake at a commanded pose and walks that pose along the
   insertion/extraction path — exactly the carry a gripper on the grip block
   performs. The reference is LAG-CLAMPED to the rake's actual pose so a jam
   cannot wind up spring force.
2. TRANSPORT (teleport): single root-state writes carry ONE object at a time
   across FREE SPACE with zero velocity. The rake is set at a hover in front
   of the mouth (never inside anything); the freed pot is set at a hover a few
   millimetres above the burner plate. Nothing is ever written into contact.
3. EXTRACTION (contact dynamics): the flange is walked in above the pot,
   lowered into the gap behind the knob, and dragged mouth-ward; the
   flange-on-knob contact and the sub-tipping friction slide on the slick
   floor do all the work. The pot's exit from the cubby is 100 % contact-made.
4. The stove placement is only possible NOW: the pot starts unreachable at the
   back of a lane no hand enters (that is the strategic gate).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's credit is latched), then holds HANDS-OFF for >= 3 simulated seconds
after success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene8_put_the_right_moka_pot_on_the_stove_i329.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401 - registers the scene
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

G = 9.81


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cubby_rake_stove")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    from isaaclab.utils.math import quat_apply_inverse

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so
    # the task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    zero3 = torch.zeros(n, 1, 3, device=device)
    q_hold = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).expand(n, 4)  # yaw 180

    # ----- geometry (world) -----------------------------------------------------------------
    cub = scene.cubby.data.root_pos_w
    floor_top = float(cub[0, 2]) + c.floor_t
    mouth_x = float(cub[0, 0]) + c.mouth_x
    z_ins = floor_top + c.knob_top + 0.008 + c.flange_drop + c.bar_t / 2
    z_eng = floor_top + c.head_bot - 0.004 + c.flange_drop + c.bar_t / 2
    x_start = mouth_x + c.bar_len / 2 + 0.010

    # ----- held-rake wrench servo ------------------------------------------------------------
    def rake_wrench(p_des: torch.Tensor, kp: float = 150.0, kd: float = 8.0,
                    kr: float = 2.0, kw: float = 0.12, scale: float = 1.0) -> None:
        """One step of the 6-DOF hold: world-frame gravity ff + PD to (p_des,
        q_hold), rotated into the BODY frame (isaaclab wrenches act there)."""
        p = scene.rake.data.root_pos_w
        v = scene.rake.data.root_lin_vel_w
        q = scene.rake.data.root_quat_w
        w = scene.rake.data.root_ang_vel_w
        f_w = kp * (p_des - p) - kd * v
        f_w = f_w.clamp(-5.0, 5.0)
        f_w[:, 2] = f_w[:, 2] + c.rake_mass * G
        # orientation error (q_hold * q^-1), small-angle vector form
        qw, qx, qy, qz = q.unbind(-1)
        hw, hx, hy, hz = q_hold.unbind(-1)
        ew = hw * qw + hx * qx + hy * qy + hz * qz
        ex = -hw * qx + hx * qw - hy * qz + hz * qy
        ey = -hw * qy + hx * qz + hy * qw - hz * qx
        ez = -hw * qz - hx * qy + hy * qx + hz * qw
        sign = torch.where(ew >= 0, torch.ones_like(ew), -torch.ones_like(ew))
        err = 2.0 * sign.unsqueeze(-1) * torch.stack((ex, ey, ez), dim=-1)
        t_w = (kr * err - kw * w).clamp(-0.8, 0.8)
        f_b = quat_apply_inverse(q, f_w * scale)
        t_b = quat_apply_inverse(q, t_w * scale)
        scene.rake.set_external_force_and_torque(f_b.unsqueeze(1), t_b.unsqueeze(1))
        env.step(no_action)

    def drop_wrench() -> None:
        scene.rake.set_external_force_and_torque(zero3, zero3)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        pt = scene.pot_t.data.root_pos_w[0]
        rk = scene.rake.data.root_pos_w[0]
        print(f"[solve] {tag:12s} | pot=({float(pt[0]):+.3f},{float(pt[1]):+.3f},"
              f"{float(pt[2]):+.3f}) rake=({float(rk[0]):+.3f},{float(rk[1]):+.3f},"
              f"{float(rk[2]):+.3f}) "
              f"in_cubby(t)={bool(scene.in_cubby(scene.pot_t)[0])} "
              f"out={bool(scene.pot_out()[0])} on_pad={bool(scene.pot_on_pad()[0])} "
              f"rake_clear={bool(scene.rake_clear()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"latches=(r={bool(scene._l_reach[0])},o={bool(scene._l_out[0])},"
              f"p={bool(scene._l_pad[0])}) "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    last_printed = [0.0]

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        assert s >= last_printed[0] - 1e-6, \
            f"score regressed: {last_printed[0]} -> {s}"
        last_printed[0] = s
        return s

    def write_pose(body, pos_w, quat_w) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat_w
        body.write_root_state_to_sim(st, all_ids)

    def move_to(p_des: torch.Tensor, tgt: torch.Tensor, v: float,
                settle: int = 30, lag: float = 0.020, max_steps: int = 2000,
                stop=None) -> torch.Tensor:
        """Walk the wrench reference from p_des toward tgt at v (m/s), reference
        lag-clamped to the rake's actual pose; optional readback stop."""
        dv = v / 120.0
        for _ in range(max_steps):
            d = tgt - p_des
            dist = d.norm(dim=-1, keepdim=True)
            if float(dist.max()) < 1e-4:
                break
            p_des = p_des + d / dist.clamp(min=1e-9) * min(dv, float(dist.max()))
            # lag clamp: never let the reference run away from the actual pose
            act = scene.rake.data.root_pos_w
            p_des = act + (p_des - act).clamp(-lag, lag)
            rake_wrench(p_des)
            if stop is not None and stop():
                break
        for _ in range(settle):
            rake_wrench(p_des)
        return p_des

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)  # everything seats (pots in their lanes, rake on the counter)
    report("reset")
    assert torch.isfinite(scene.rake.data.root_pos_w).all(), "NaN/inf after settle"
    assert bool(scene.in_cubby(scene.pot_t)[0]) and bool(scene.in_cubby(scene.pot_w)[0]), \
        "both pots must start inside the cubby"
    assert not bool(scene.pot_on_pad()[0]), "burner must start empty"
    s0 = print_score("P0 reset+settle (pots deep in their lanes, rake on the counter)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phases 1-3: rake the copper pot out ----------------------------------
    pot0 = scene.pot_t.data.root_pos_w.clone()
    lane_y = pot0[:, 1].clone()  # aim the bar down the copper pot's lane centre
    lane_y[:] = cub[:, 1] + torch.sign(lane_y - cub[:, 1]) * c.lane_y

    extracted = False
    for attempt in range(3):
        pot_x = scene.pot_t.data.root_pos_w[:, 0]
        # (1a) transport the rake to a free-space hover in front of the mouth,
        # nose (flange end) pointing at the lane, insertion height
        hover = torch.stack([torch.full_like(pot_x, x_start), lane_y,
                             torch.full_like(pot_x, z_ins)], dim=-1)
        write_pose(scene.rake, hover, q_hold)
        p_des = hover.clone()
        for _ in range(60):
            rake_wrench(p_des)
        # (1b) slide the bar nose-first down the lane, riding over the pot,
        # until the flange is past the knob
        x_desc = pot_x + 0.173  # flange centred over the gap behind the knob
        tgt = p_des.clone()
        tgt[:, 0] = x_desc
        p_des = move_to(p_des, tgt, v=0.06)
        err = float((scene.rake.data.root_pos_w - p_des).norm(dim=-1).max())
        report("inserted")
        if err > 0.015:
            print(f"[solve] insertion stalled (err {err:.3f}); retrying", flush=True)
            tgt[:, 2] = z_ins + 0.004
            p_des = move_to(p_des, tgt, v=0.04)
            continue
        # (1c) lower the flange into the gap behind the knob
        tgt = p_des.clone()
        tgt[:, 2] = z_eng
        p_des = move_to(p_des, tgt, v=0.03)
        report("hooked")
        print_score("P1 rake inserted and hooked behind the knob")
        # (2) drag the pot mouth-ward; break on the POT's readback, not the
        # reference (the pot trails the flange by the engagement slack)
        x_out = mouth_x + c.out_margin + 0.010
        tgt = p_des.clone()
        tgt[:, 0] = x_out + 0.190  # flange face at pot-out x when reference gets there
        p_des = move_to(
            p_des, tgt, v=0.05, settle=10,
            stop=lambda: float(scene.pot_t.data.root_pos_w[:, 0].min()) > x_out)
        report("dragged")
        moved = float((scene.pot_t.data.root_pos_w[:, 0] - pot_x).min())
        if float(scene.pot_t.data.root_pos_w[:, 0].min()) <= x_out - 0.002:
            print(f"[solve] drag came up short (moved {moved:.3f} m); retrying",
                  flush=True)
            # lift clear and go around again
            tgt = p_des.clone()
            tgt[:, 2] = z_ins + 0.030
            p_des = move_to(p_des, tgt, v=0.06)
            continue
        # hold still until the pot settles and the extraction latch fires
        for _ in range(120):
            rake_wrench(p_des)
        if bool(scene._l_out.all()):
            extracted = True
            break
        print("[solve] extraction latch did not fire; retrying", flush=True)
    assert extracted, "the rake never extracted the copper pot"
    assert bool(scene._l_reach[0]), "reach latch must have fired during the drag"
    report("extracted")
    s2 = print_score("P2+P3 copper pot dragged out of the cubby onto the apron")
    assert s2 >= 0.49, f"extraction score {s2} (expect 0.20 + 0.30)"

    # (3) lift the flange clear of the knob, then park the rake back on the
    # counter (transport through free space, released from a 3 mm hover)
    tgt = p_des.clone()
    tgt[:, 2] = z_ins + 0.060
    p_des = move_to(p_des, tgt, v=0.06)
    park = torch.tensor([c.rake_pos[0], c.rake_pos[1],
                         c.deck_top + c.bar_t / 2 + c.flange_drop + 0.003],
                        device=device).expand(n, 3).clone()
    write_pose(scene.rake, park, q_hold)
    drop_wrench()
    step(240)
    report("rake-parked")
    assert bool(scene.rake_clear()[0]), "parked rake must be clear of the stove"

    # ---------------- phase 4: copper pot onto the burner ----------------------------------
    sp = scene.stove.data.root_pos_w
    hover = sp.clone()
    hover[:, 2] += c.plate_t + c.pad_h + 0.008
    upq = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4)
    write_pose(scene.pot_t, hover, upq)
    step(360)  # falls 8 mm, seats upright on the grippy pad
    report("pot-set")
    assert bool(scene.pot_on_pad()[0]), "copper pot must rest upright on the burner"
    # wait for success to hold CONTINUOUSLY for 2 simulated seconds so the phase
    # boundary marks a genuinely settled state, not a transient
    streak = 0
    ok = False
    for _ in range(1200):
        env.step(no_action)
        streak = streak + 1 if bool(scene.success()[0]) else 0
        if streak >= 240:
            ok = True
            break
    report("stable")
    assert ok and bool(scene.success()[0]), "success did not stabilize"
    s3 = print_score("P4 copper pot on the burner, steel pot untouched — success")
    assert s3 >= 0.99, f"success must score 1.0, got {s3}"

    # ---------------- phase 5: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P5 persistence 3.3 s hands-off")
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
    except BaseException:  # noqa: BLE001 - die fast; Kit teardown would hang until the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)

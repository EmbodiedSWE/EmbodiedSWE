"""Teleport solution for GaugeAdapterScene (sim_gen task `plug_charger_i337`) —
the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY — two pose writes, each moving a FREE body
across free space to a staging pose (never into a mated or load-bearing state):
  T1 carries the green adapter from its floor spawn to the empty shelf, fingers
     facing the outlet, tips 10 mm SHORT of the panel face;
  T2 carries the charger from its floor spawn to a hover 8 mm ABOVE the seated
     adapter's deck, prongs down, tips above the funnel mouths (no contact).
Every load-bearing interaction goes through CONTACT DYNAMICS:
  P1 — SEAT THE ADAPTER: a floating-hand force controller (velocity-regulated push
  along the dock +y insertion axis, lateral PD centring the fingers on the slots,
  yaw/upright steadying torque, friction-stall escalation) slides the adapter along
  the real shelf until both fingers are deep in the outlet slots and the body face
  presses flush on the panel — the panel is what stops it. Release, settle.
  P2 — PLUG THE CHARGER: the same style of controller holds the hovering charger
  (gravity-feedforward vertical regulation, adapter-tracking lateral PD, full
  orientation PD keeping the prongs down the bores), then presses it down: the
  prongs thread the funnel mouths and the square bores under contact, until the
  charger body rests on the adapter deck — the deck is what stops it. Release,
  settle.
Both heterogeneous mates — the horizontal slide-to-flush and the vertical
drop-press, the two interactions the gauge chain exists to force — are pure
contact physics.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's credit is latched), then holds HANDS-OFF for >= 3.4 simulated seconds
after success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds.

Run (forge): python -u -m simgen_tasks.plug_charger_i337.solve --headless [--seed N]
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
    env = ENVS.get("simgen.gauge_adapter")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback
    # so distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def body_yaw(body) -> float:
        q = body.data.root_quat_w[0]
        return 2.0 * math.atan2(float(q[3]), float(q[0]))

    def dock_yaw() -> float:
        return body_yaw(scene.station)

    def dock_axes() -> tuple[torch.Tensor, torch.Tensor]:
        """(lat_w, in_w): dock-local +x (lateral) and +y (into the panel) in world."""
        gy = dock_yaw()
        lat_w = torch.tensor([math.cos(gy), math.sin(gy), 0.0], device=device)
        in_w = torch.tensor([-math.sin(gy), math.cos(gy), 0.0], device=device)
        return lat_w, in_w

    def report(tag: str) -> None:
        a = scene.adapter_dock()[0]
        t = scene.finger_tip_dock()[0]
        tp = scene.prong_tips_adapter()[0]
        print(f"[solve] {tag:12s} | tip_y={float(t[1]):+.4f} "
              f"a_dock=({float(a[0]):+.3f},{float(a[2]):.3f}) "
              f"tipz_max={float(tp[:, 2].max()):+.4f} "
              f"seated={bool(scene.adapter_seated()[0])} "
              f"mated={bool(scene.charger_mated()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_wrench(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def apply_wrench(body, f_world: torch.Tensor, tq_world: torch.Tensor) -> None:
        body.set_external_force_and_torque(
            f_world.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            tq_world.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            env_ids=all_ids, is_global=True)

    def yaw_x_quat(yaw: float, xrot: float) -> torch.Tensor:
        """quat = Rz(yaw) * Rx(xrot), wxyz."""
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        cx, sx = math.cos(xrot / 2), math.sin(xrot / 2)
        return torch.tensor([cy * cx, cy * sx, sy * sx, sy * cx], device=device)

    def orient_pd(body, q_tgt: torch.Tensor, k: float, d: float,
                  clamp: float) -> torch.Tensor:
        """Full orientation PD torque driving `body` to `q_tgt` (world, wxyz)."""
        q = body.data.root_quat_w[0]
        # q_err = q_tgt * conj(q)
        w1, x1, y1, z1 = (float(v) for v in q_tgt)
        w2, x2, y2, z2 = float(q[0]), -float(q[1]), -float(q[2]), -float(q[3])
        ew = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        ex = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        ey = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        ez = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        sgn = 1.0 if ew >= 0 else -1.0
        rv = torch.tensor([sgn * ex, sgn * ey, sgn * ez], device=device) * 2.0
        tq = k * rv - d * body.data.root_ang_vel_w[0]
        return tq.clamp(-clamp, clamp)

    def adapter_steady_torque() -> torch.Tensor:
        """Upright + dock-yaw-follow steadying torque for the adapter during the
        seat slide (what a hand on the block provides for free — a block that yaws
        in the slots wedges finger-to-cheek; the CoM push cannot prevent that)."""
        from isaaclab.utils.math import quat_apply

        ezv = torch.tensor([0.0, 0.0, 1.0], device=device)
        axis = quat_apply(scene.adapter.data.root_quat_w, ezv.expand(n, 3))[0]
        w_w = scene.adapter.data.root_ang_vel_w[0]
        dyaw = math.atan2(math.sin(dock_yaw() - body_yaw(scene.adapter)),
                          math.cos(dock_yaw() - body_yaw(scene.adapter)))
        tq = 0.08 * torch.linalg.cross(axis, ezv) - 0.010 * w_w
        tq[0] = tq[0].clamp(-0.08, 0.08)
        tq[1] = tq[1].clamp(-0.08, 0.08)
        tq[2] = max(-0.12, min(0.12, 0.30 * dyaw - 0.020 * float(w_w[2])))
        return tq

    g = 9.81

    # ---------------- phase 0: reset, settle, baseline -----------------------------------
    step(60)
    sp = scene.station.data.root_pos_w[0] - scene.env_origins[0]
    ad = scene.adapter_dock()[0]
    dd = scene.decoy_dock()[0]
    chd = scene.charger_dock()[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"dock=({float(sp[0]):+.3f},{float(sp[1]):+.3f}) "
          f"yaw={math.degrees(dock_yaw()):+.1f}deg "
          f"adapter_dock=({float(ad[0]):+.3f},{float(ad[1]):+.3f}) "
          f"yaw={math.degrees(body_yaw(scene.adapter)):+.1f}deg "
          f"decoy_dock=({float(dd[0]):+.3f},{float(dd[1]):+.3f}) "
          f"charger_dock=({float(chd[0]):+.3f},{float(chd[1]):+.3f}) "
          f"yaw={math.degrees(body_yaw(scene.charger)):+.1f}deg", flush=True)
    try:
        m_ad = float(scene.adapter.root_physx_view.get_masses()[0, 0])
        m_ch = float(scene.charger.root_physx_view.get_masses()[0, 0])
        print(f"[solve] masses readback: adapter={m_ad:.3f} kg charger={m_ch:.3f} kg",
              flush=True)
    except Exception:  # noqa: BLE001 - readback is informational
        pass
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- T1: transport the green adapter to the shelf -----------------------
    # The adapter is a free body loose on the floor. Carry it (pose write, free
    # space only — the shelf top is empty) to a staging pose ON the shelf: aligned
    # with the dock, centred, finger tips 10 mm SHORT of the panel face. It is NOT
    # teleported into the slots.
    gy = dock_yaw()
    lat_w, in_w = dock_axes()
    spos_w = scene.station.data.root_pos_w[0]
    stage_y = -0.010 - c.finger_tip_dy  # tips at dock y = -0.010 (outside the panel)
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = float(spos_w[0]) + stage_y * float(in_w[0])
    st[:, 1] = float(spos_w[1]) + stage_y * float(in_w[1])
    st[:, 2] = float(spos_w[2]) + c.shelf_top + c.ad_body_h / 2 + 0.0015
    st[:, 3] = math.cos(gy / 2)
    st[:, 6] = math.sin(gy / 2)
    scene.adapter.write_root_state_to_sim(st, all_ids)
    step(40)
    report("staged-ad")
    s1t = print_score("T1 transport adapter to the shelf")
    assert s1t >= s0 - 1e-6, "score decreased across the adapter transport"

    # ---------------- phase 1: SEAT THE ADAPTER (contact dynamics) -----------------------
    # Velocity-regulated push along dock +y: the adapter slides the real shelf,
    # its fingers thread the real slots, until the body face presses flush on the
    # panel. The panel is what stops it.
    m_a = c.adapter_mass
    push_ff, v_des_cap = 0.6, 0.06
    best_y, last_gain = -1.0, 0
    done_seat = False
    for i in range(1500):
        lat_w, in_w = dock_axes()
        t = scene.finger_tip_dock()[0]
        y = float(t[1])
        if y >= c.seat_tip_y - 0.0008:
            done_seat = True
            break
        v_w = scene.adapter.data.root_lin_vel_w[0]
        v_in = float((v_w * in_w).sum())
        v_des = min(v_des_cap, max(0.01, 3.0 * (c.seat_tip_y - y)))
        f_in = max(0.0, min(6.0, push_ff + m_a * 55.0 * (v_des - v_in)))
        ax = float(scene.adapter_dock()[0, 0])
        v_lat = float((v_w * lat_w).sum())
        f_lat = max(-2.0, min(2.0, m_a * (120.0 * (0.0 - ax) - 25.0 * v_lat)))
        apply_wrench(scene.adapter, in_w * f_in + lat_w * f_lat, adapter_steady_torque())
        env.step(no_action)
        if y > best_y + 0.0008:
            best_y, last_gain = y, i
        elif i - last_gain > 180:  # stalled
            if y >= c.seat_y_min + 0.0015:
                # stalled INSIDE the seat band while still pressing forward: the
                # panel face is what is stopping the body — flush by contact.
                print(f"[solve] panel stop reached (stall at tip_y={y:+.4f} "
                      f"under forward press)", flush=True)
                done_seat = True
                break
            push_ff = min(push_ff + 0.4, 4.0)
            last_gain = i
            print(f"[solve] seat push stalled at tip_y={y:+.4f}, ff={push_ff:.1f} N",
                  flush=True)
    clear_wrench(scene.adapter)
    step(90)  # release + settle: seating is judged on the SETTLED state
    report("seated")
    print(f"[solve] seat loop done (done_seat={done_seat}, "
          f"tip_y={float(scene.finger_tip_dock()[0, 1]):+.4f})", flush=True)
    if not bool(scene.adapter_seated()[0]):
        print("SIM_GEN_SOLVE: FAIL (adapter did not seat in the outlet)", flush=True)
        os._exit(1)
    s1 = print_score("P1 adapter seated (contact)")
    assert s1 >= s1t - 1e-6, "score decreased across the seat slide"

    # ---------------- T2: transport the charger to a hover over the deck -----------------
    # The charger is a free body loose on the floor. Carry it (pose write, free
    # space only) to a hover 8 mm above the seated adapter's deck, prongs DOWN,
    # tips above the funnel mouths — no contact with anything. It is NOT
    # teleported into the sockets.
    ap = scene.adapter.data.root_pos_w[0]
    ayaw = body_yaw(scene.adapter)
    q_tgt = yaw_x_quat(ayaw, -math.pi / 2)  # local +y (prongs) -> world -z
    hover_z = float(ap[2]) + c.deck_dz + c.prong_tip_dy + 0.008
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = float(ap[0])
    st[:, 1] = float(ap[1])
    st[:, 2] = hover_z
    st[:, 3:7] = q_tgt
    scene.charger.write_root_state_to_sim(st, all_ids)

    # Hold-align: the hand supports the charger against gravity while the lateral
    # PD and orientation PD null out any drift before the press begins.
    m_c = c.charger_mass
    for _ in range(40):
        v_w = scene.charger.data.root_lin_vel_w[0]
        p_w = scene.charger.data.root_pos_w[0]
        f = torch.zeros(3, device=device)
        f[0] = m_c * (60.0 * (float(ap[0]) - float(p_w[0])) - 15.0 * float(v_w[0]))
        f[1] = m_c * (60.0 * (float(ap[1]) - float(p_w[1])) - 15.0 * float(v_w[1]))
        f[2] = m_c * (g + 30.0 * (hover_z - float(p_w[2])) - 12.0 * float(v_w[2]))
        f[0:2] = f[0:2].clamp(-1.5, 1.5)
        f[2] = f[2].clamp(0.0, m_c * g + 3.0)
        apply_wrench(scene.charger, f, orient_pd(scene.charger, q_tgt, 0.08, 0.004, 0.06))
        env.step(no_action)
    report("staged-ch")
    s2t = print_score("T2 transport charger to hover over the deck")
    assert s2t >= s1 - 1e-6, "score decreased across the charger transport"

    # ---------------- phase 2: PLUG THE CHARGER (contact dynamics) -----------------------
    # Gravity-feedforward vertical regulation presses the charger down at a
    # controlled rate: the prongs thread the real funnel mouths and the real
    # square bores, until the body rests on the deck. The deck is what stops it.
    press_ff = 0.0  # extra downward bias, escalated on friction stalls
    tip_goal = c.deck_dz - c.prong_len + 0.0008  # body-on-deck stop, adapter frame
    best_z, last_gain = 1e9, 0
    done_mate = False
    for i in range(1500):
        ap = scene.adapter.data.root_pos_w[0]
        ayaw = body_yaw(scene.adapter)
        q_tgt = yaw_x_quat(ayaw, -math.pi / 2)
        tp = scene.prong_tips_adapter()[0]
        tipz = float(tp[:, 2].max())  # shallower tip governs
        if tipz <= tip_goal:
            done_mate = True
            break
        v_w = scene.charger.data.root_lin_vel_w[0]
        p_w = scene.charger.data.root_pos_w[0]
        f = torch.zeros(3, device=device)
        f[0] = m_c * (60.0 * (float(ap[0]) - float(p_w[0])) - 15.0 * float(v_w[0]))
        f[1] = m_c * (60.0 * (float(ap[1]) - float(p_w[1])) - 15.0 * float(v_w[1]))
        f[0:2] = f[0:2].clamp(-1.5, 1.5)
        vz_des = -min(0.05, max(0.01, 2.0 * (tipz - tip_goal)))
        f[2] = m_c * (g + 40.0 * (vz_des - float(v_w[2]))) - press_ff
        f[2] = f[2].clamp(-2.5, m_c * g + 3.0)
        apply_wrench(scene.charger, f, orient_pd(scene.charger, q_tgt, 0.08, 0.004, 0.06))
        env.step(no_action)
        if tipz < best_z - 0.0008:
            best_z, last_gain = tipz, i
        elif i - last_gain > 180:  # stalled
            depth = c.deck_dz - tipz
            if depth >= c.mate_depth_min + 0.002:
                # stalled DEEP in the bores while still pressing down: the deck is
                # what is stopping the body — mated by contact.
                print(f"[solve] deck stop reached (stall at tipz={tipz:+.4f} "
                      f"under downward press)", flush=True)
                done_mate = True
                break
            press_ff = min(press_ff + 0.5, 2.5)
            last_gain = i
            print(f"[solve] press stalled at tipz={tipz:+.4f}, press_ff={press_ff:.1f} N",
                  flush=True)
    clear_wrench(scene.charger)
    step(90)  # release + settle: mating is judged on the SETTLED state
    report("mated")
    print(f"[solve] press loop done (done_mate={done_mate}, "
          f"tipz={float(scene.prong_tips_adapter()[0, :, 2].max()):+.4f})", flush=True)
    if not bool(scene.charger_mated()[0]):
        print("SIM_GEN_SOLVE: FAIL (charger did not mate into the adapter)", flush=True)
        os._exit(1)
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after mating)", flush=True)
        os._exit(1)
    s2 = print_score("P2 charger plugged (contact) -> success")
    assert s2 >= s2t - 1e-6, "score decreased across the plug press"

    # ---------------- phase 3: persistence (>= 3.4 simulated seconds, hands-off) ---------
    hold = True
    for _ in range(10):  # 10 x 41 steps = 410 substeps = 3.4 s at 120 Hz
        step(41)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s3 = print_score("P3 persistence 3.4 s")
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
    except Exception:  # noqa: BLE001 - crash guard: Kit teardown hangs on exceptions
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)

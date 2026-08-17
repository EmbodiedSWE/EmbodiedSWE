"""Smoke / rubric-REJECTION battery for GaugeAdapterScene (sim_gen task
`plug_charger_i337`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — seat the green adapter in the outlet under a
floating-hand force controller, then press the charger down into its top sockets,
both by contact — is the acceptance evidence that the rubric ACCEPTS a correct
outcome). Every teleport here is instrumentation that CONSTRUCTS a wrong (or
partial) outcome and asserts the rubric REJECTS it; no probe in this battery ever
reaches success(), and a final audit check asserts exactly that.

  1.  settle/no-NaN     — reset layout settles finite: all three loose objects
                          verified OUTSIDE the station and the two adapters on
                          OPPOSITE sides by dock-frame READBACK;
  2.  SEED strategy     — the seed task's whole plan (align the charger's prongs
                          with the wall holes, push) tried literally as a CONTACT
                          probe: 2 simulated seconds of a 4 N aimed push at the
                          outlet. Non-vacuous (the charger really advances to the
                          panel) and REFUSED there: the prong tips never pass the
                          panel face, no credit latches, score ~0;
  3-4. randomization    — READBACK over 8 seeded resets: station xy + yaw vary;
                          the green adapter's SIDE flips, its spawn and the
                          charger's spawn xy + relative yaw vary; the two adapters
                          sit on opposite sides every time;
  5.  null policy       — 240 idle steps -> score ~0, no success;
  6.  decoy dead end    — decoy CONSTRUCTED seated in the outlet, charger pressed
                          down on ITS deck (2 s, orientation-held press):
                          non-vacuous (tips reach the deck) but the 8 mm sockets
                          refuse the 10 mm prongs — tips never sink in, no credit,
                          score ~0 (a seated decoy earns nothing);
  7.  seated-only       — green adapter CONSTRUCTED seated, charger untouched:
                          seat credit only (score ~0.35), NOT success;
  8.  removal latch     — pulling the seated adapter back OUT to the floor leaves
                          the latched credit unchanged, still no success;
  9.  floor mate        — the "mated pair left loose on the floor": charger
                          CONSTRUCTED prongs-down in the green adapter's sockets
                          with the adapter far from the outlet — mate credit only
                          (score ~0.30), mated() true but NOT success;
  10. near-miss seat    — adapter fingers partway into the slots (tip readback
                          short of `seat_y_min`): partial gated credit only,
                          seated() false, NOT success;
  11. deck percher      — charger dropped prongs-down but yawed 90 deg onto the
                          seated adapter's deck: tips land on SOLID deck (readback
                          near the deck plane), zero mate credit, NOT success;
  12. wrong station spot— green adapter pushed at the panel BESIDE the slots
                          (fingers on cheek + divider, 2 s of 4 N): non-vacuous,
                          tips never pass the face, zero seat credit;
  13. monotonicity      — a deeper finger insertion latches strictly more credit;
  14. rejection audit   — success() was never True at ANY judged point;
  15. final no-NaN      — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.plug_charger_i337.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version and silently rejects RTX -> the
# annotator returns EMPTY frames. Disable the driver check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math
import os
import threading

import numpy as np
import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
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

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.52, -0.55, 0.42)) + o),
                                tuple(np.array((0.0, -0.02, 0.06)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action)
            if (annot is not None and step_i % args.record_every == 0
                    and len(frames) < args.max_frames):
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    ever_success = [False]

    def judge() -> tuple[float, bool]:
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return s, ok

    def station_pose() -> tuple[torch.Tensor, float]:
        sp = (scene.station.data.root_pos_w - scene.env_origins)[0]
        q = scene.station.data.root_quat_w[0]
        return sp, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def body_yaw(body) -> float:
        q = body.data.root_quat_w[0]
        return 2.0 * math.atan2(float(q[3]), float(q[0]))

    def report(tag: str) -> None:
        a = scene.adapter_dock()[0]
        t = scene.finger_tip_dock()[0]
        tp = scene.prong_tips_adapter()[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | tip_y={float(t[1]):+.4f} "
              f"a_dock=({float(a[0]):+.3f},{float(a[2]):.3f}) "
              f"tipz_max={float(tp[:, 2].max()):+.4f} "
              f"seated={bool(scene.adapter_seated()[0])} "
              f"mated={bool(scene.charger_mated()[0])} "
              f"latches=({float(scene.seat_latch[0]):.2f},{float(scene.mate_latch[0]):.2f}) "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
        return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

    def yaw_x_quat(yaw: float, xrot: float) -> tuple[float, float, float, float]:
        """quat = Rz(yaw) * Rx(xrot), wxyz — prongs-down when xrot = -pi/2."""
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        cx, sx = math.cos(xrot / 2), math.sin(xrot / 2)
        return (cy * cx, cy * sx, sy * sx, sy * cx)

    def write_pose(body, x: float, y: float, z: float,
                   quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        """Probe write (instrumentation, not a solution); no stepping."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def dock_to_world(lx: float, ly: float) -> tuple[float, float, float]:
        sp, gy = station_pose()
        ca, sa = math.cos(gy), math.sin(gy)
        return (float(sp[0]) + ca * lx - sa * ly,
                float(sp[1]) + sa * lx + ca * ly, gy)

    def dock_axes() -> tuple[torch.Tensor, torch.Tensor]:
        _, gy = station_pose()
        lat_w = torch.tensor([math.cos(gy), math.sin(gy), 0.0], device=device)
        in_w = torch.tensor([-math.sin(gy), math.cos(gy), 0.0], device=device)
        return lat_w, in_w

    def orient_pd(body, q_tgt: tuple, k: float, d: float, clamp: float) -> torch.Tensor:
        """Full orientation PD torque driving `body` to `q_tgt` (world, wxyz)."""
        q = body.data.root_quat_w[0]
        w1, x1, y1, z1 = q_tgt
        w2, x2, y2, z2 = float(q[0]), -float(q[1]), -float(q[2]), -float(q[3])
        ew = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        ex = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        ey = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        ez = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        sgn = 1.0 if ew >= 0 else -1.0
        rv = torch.tensor([sgn * ex, sgn * ey, sgn * ez], device=device) * 2.0
        tq = k * rv - d * body.data.root_ang_vel_w[0]
        return tq.clamp(-clamp, clamp)

    def apply_wrench(body, f_world: torch.Tensor, tq_world: torch.Tensor) -> None:
        body.set_external_force_and_torque(
            f_world.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            tq_world.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            env_ids=all_ids, is_global=True)

    def clear_wrench(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def seat_construct(body, lx: float, tip_y: float, settle_steps: int = 40) -> None:
        """CONSTRUCT an adapter riding the shelf at dock-local x = lx with finger
        tips at dock y = tip_y (positive = in the slots), aligned to the dock."""
        wx, wy, gy = dock_to_world(lx, tip_y - c.finger_tip_dy)
        write_pose(body, wx, wy, c.ad_z_ride + 0.001, quat=yaw_quat(gy))
        step(settle_steps)

    def prong_tips_in(body) -> torch.Tensor:
        """(2,3) charger prong tips in `body`'s frame (works for the decoy too)."""
        from isaaclab.utils.math import quat_apply_inverse

        tips = scene.prong_tips_w()
        q = body.data.root_quat_w[:, None, :].expand(n, 2, 4).reshape(-1, 4)
        d = (tips - body.data.root_pos_w[:, None, :]).reshape(-1, 3)
        return quat_apply_inverse(q, d).reshape(n, 2, 3)[0]

    # =========================== 1. settle + reset readback ================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    ad = scene.adapter_dock()[0]
    dd = scene.decoy_dock()[0]
    chd = scene.charger_dock()[0]
    fin0 = bool(torch.isfinite(scene.station.data.root_state_w).all()
                and torch.isfinite(scene.adapter.data.root_state_w).all()
                and torch.isfinite(scene.decoy.data.root_state_w).all()
                and torch.isfinite(scene.charger.data.root_state_w).all())
    check("settle: states finite; adapter, decoy and charger verified OUTSIDE the "
          "station (dock-frame y readback) with the two adapters on OPPOSITE sides; "
          "everything settled",
          fin0 and bool(scene.settled()[0]) and float(ad[1]) < -0.10
          and float(dd[1]) < -0.10 and float(chd[1]) < -0.20
          and float(ad[0]) * float(dd[0]) < 0)

    # =========================== 2. SEED strategy (contact, the gauge gate) ================
    # The seed task's whole plan, tried literally: put the charger in front of the
    # outlet at slot height, prongs aimed at the holes, and PUSH. 2 simulated
    # seconds of a 4 N orientation-held push. The prongs overlap the divider at
    # every planar pose, so the tips must never pass the panel face.
    env.reset(seed=13)
    step(30)
    wx, wy, gy = dock_to_world(0.0, -0.015 - c.prong_tip_dy)  # tips 15 mm short
    write_pose(scene.charger, wx, wy, c.shelf_top + c.ch_body_h / 2 + 0.001,
               quat=yaw_quat(gy))
    step(10)
    tip_start = float(scene.prong_tips_dock()[0][:, 1].max())
    tip_max = -1.0
    for _ in range(240):
        lat_w, in_w = dock_axes()
        _, gy = station_pose()
        apply_wrench(scene.charger, in_w * 4.0,
                     orient_pd(scene.charger, yaw_quat(gy), 0.06, 0.004, 0.05))
        step(1)
        tip_max = max(tip_max, float(scene.prong_tips_dock()[0][:, 1].max()))
    clear_wrench(scene.charger)
    step(40)
    report("seed-strategy")
    s, ok = judge()
    check("SEED strategy: 2 s of a 4 N aimed push of the charger prongs at the wall "
          f"outlet — the charger really advanced (tip {tip_start:+.3f} -> max "
          f"{tip_max:+.3f}, non-vacuous) but the tips never passed the panel face "
          "(gauge mismatch), no credit latched, score <= 0.02, no success",
          tip_max > tip_start + 0.008 and tip_max < c.seat_gate_y0
          and float(scene.seat_latch[0]) == 0.0 and float(scene.mate_latch[0]) == 0.0
          and s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real ================================
    reads = []
    for sd_i in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd_i)
        step(5)
        sp, gy = station_pose()
        ad = scene.adapter_dock()[0]
        dd = scene.decoy_dock()[0]
        chd = scene.charger_dock()[0]
        rel = math.atan2(math.sin(body_yaw(scene.charger) - gy),
                         math.cos(body_yaw(scene.charger) - gy))
        reads.append((float(sp[0]), float(sp[1]), gy, float(ad[0]), float(ad[1]),
                      float(chd[0]), float(chd[1]), rel, float(ad[0]) * float(dd[0])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (station_x, station_y, station_yaw, "
          f"ad_lx, ad_ly, ch_lx, ch_ly, ch_rel_yaw, side_prod):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: station xy + yaw vary across 8 seeded resets (readback)",
          spread[0] > 0.01 and spread[1] > 0.01 and spread[2] > 0.2)
    check("randomization: the green adapter's SIDE flips across seeds, its spawn and "
          "the charger's spawn xy + relative yaw vary, and the two adapters sit on "
          "opposite sides every reset (readback)",
          arr[:, 3].min() < -0.05 and arr[:, 3].max() > 0.05
          and spread[5] > 0.02 and spread[6] > 0.01 and spread[7] > 0.5
          and bool((arr[:, 8] < 0).all()))

    # =========================== 5. null policy fails ======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. decoy dead end (contact) ===============================
    # Decoy CONSTRUCTED seated in the outlet, then the charger pressed prongs-down
    # onto ITS deck for 2 s with an orientation-held 2.5 N press. The decoy's 8 mm
    # sockets refuse the 10 mm prongs: the tips ride ON the deck, never sink in.
    env.reset(seed=41)
    step(30)
    seat_construct(scene.decoy, 0.0, c.seat_tip_y - 0.001, settle_steps=30)
    dt = scene.decoy_tip_dock()[0]
    dp = scene.decoy.data.root_pos_w[0]
    dyaw = body_yaw(scene.decoy)
    q_dn = yaw_x_quat(dyaw, -math.pi / 2)
    write_pose(scene.charger,
               float(dp[0] - scene.env_origins[0][0]),
               float(dp[1] - scene.env_origins[0][1]),
               float(dp[2] - scene.env_origins[0][2]) + c.deck_dz + c.prong_tip_dy + 0.006,
               quat=q_dn)
    pressed_at_mouth = False  # a tip really bore down AT a socket mouth (non-vacuity)
    deepest_in_socket = 1e9  # deepest tip excursion while INSIDE a socket footprint
    m_ch = c.charger_mass
    half_m = c.decoy_mouth_w / 2
    for _ in range(240):
        # decoy-tracking lateral PD (as in the solve's plug press): without it the
        # zero-clearance mouth ledges skid the charger clean off the deck, which
        # would make this probe vacuous
        dpw = scene.decoy.data.root_pos_w[0]
        cpw = scene.charger.data.root_pos_w[0]
        cv = scene.charger.data.root_lin_vel_w[0]
        fx = max(-1.5, min(1.5, m_ch * (80.0 * float(dpw[0] - cpw[0])
                                        - 18.0 * float(cv[0]))))
        fy = max(-1.5, min(1.5, m_ch * (80.0 * float(dpw[1] - cpw[1])
                                        - 18.0 * float(cv[1]))))
        f = torch.tensor([fx, fy, -2.5], device=device)
        apply_wrench(scene.charger, f, orient_pd(scene.charger, q_dn, 0.12, 0.008, 0.10))
        step(1)
        # judge tip depth ONLY where it means "entering a socket": tips inside a
        # mouth footprint (decoy frame). A tip that pitches off the deck edge dips
        # below the deck PLANE without ever being over a socket — that is not entry.
        tips_d = prong_tips_in(scene.decoy)
        for k in range(2):
            tx, ty, tz1 = float(tips_d[k, 0]), float(tips_d[k, 1]), float(tips_d[k, 2])
            in_fp = (abs(ty) < half_m
                     and (abs(tx - c.bore_cx) < half_m or abs(tx + c.bore_cx) < half_m))
            if in_fp:
                rel = tz1 - c.deck_dz
                deepest_in_socket = min(deepest_in_socket, rel)
                if rel < 0.004:
                    pressed_at_mouth = True
    clear_wrench(scene.charger)
    step(40)
    report("decoy-dead-end")
    s, ok = judge()
    check("decoy dead end: seated decoy (finger-tip readback "
          f"{float(dt[1]):+.4f} deep) + 2 s of a 2.5 N orientation-held press of the "
          f"charger on its deck — a prong really bore down AT a socket mouth "
          f"(non-vacuous) but the 8 mm sockets refused the 10 mm prongs (deepest "
          f"in-footprint tip excursion {deepest_in_socket * 1000:+.1f} mm — at worst "
          f"the funnel-mouth ledge, far short of the {c.mate_depth_min * 1000:.0f} mm "
          "mate depth), no credit, score <= 0.02, no success",
          float(dt[1]) > c.seat_y_min and pressed_at_mouth
          and deepest_in_socket > -0.008
          and float(scene.seat_latch[0]) == 0.0 and float(scene.mate_latch[0]) == 0.0
          and s <= 0.02 and not ok)

    # =========================== 7-8. seated-only + removal latch ==========================
    env.reset(seed=61)
    step(30)
    seat_construct(scene.adapter, 0.0, c.seat_tip_y - 0.001, settle_steps=50)
    report("seated-only")
    s_seat, ok = judge()
    check("seated-only: green adapter CONSTRUCTED seated in the outlet, charger "
          f"untouched on the floor — seated() true, seat credit only (0.33 <= "
          f"{s_seat:.3f} <= 0.37), NOT success",
          bool(scene.adapter_seated()[0]) and 0.33 <= s_seat <= 0.37 and not ok)
    # pull it back OUT to the floor (a spot clear of the decoy's spawn band):
    # latched credit survives, success stays gone
    wx, wy, gy = dock_to_world(0.30, -0.12)
    write_pose(scene.adapter, wx, wy, c.ad_z_floor + 0.0015, quat=yaw_quat(gy))
    step(40)
    report("seat-removed")
    s_rm, ok = judge()
    check("removal latch: pulling the seated adapter back OUT to the floor leaves "
          f"the latched credit unchanged ({s_seat:.3f} -> {s_rm:.3f}), seated() "
          "false again, still no success",
          abs(s_rm - s_seat) < 0.02 and not bool(scene.adapter_seated()[0]) and not ok)

    # =========================== 9. floor mate (the loose pair) ============================
    env.reset(seed=71)
    step(30)
    # park the green adapter upright on the open floor, away from the outlet and
    # clear of the decoy's / charger's spawn bands
    wx, wy, gy = dock_to_world(0.30, -0.20)
    write_pose(scene.adapter, wx, wy, c.ad_z_floor + 0.001, quat=yaw_quat(gy + 0.3))
    step(20)
    ap = scene.adapter.data.root_pos_w[0]
    ayaw = body_yaw(scene.adapter)
    # CONSTRUCT the charger prongs-down in its sockets, body just above the deck
    write_pose(scene.charger,
               float(ap[0] - scene.env_origins[0][0]),
               float(ap[1] - scene.env_origins[0][1]),
               float(ap[2] - scene.env_origins[0][2]) + c.deck_dz + c.ch_body_d / 2
               + 0.0008,
               quat=yaw_x_quat(ayaw, -math.pi / 2))
    step(60)
    report("floor-mate")
    s, ok = judge()
    check("floor mate: charger CONSTRUCTED prongs-down in the green adapter's "
          "sockets with the pair loose on the FLOOR far from the outlet — mated() "
          f"true, mate credit only (0.27 <= {s:.3f} <= 0.33), seated() false, NOT "
          "success (the chain must hang off the wall outlet)",
          bool(scene.charger_mated()[0]) and not bool(scene.adapter_seated()[0])
          and 0.27 <= s <= 0.33 and not ok)

    # =========================== 10. near-miss seat ========================================
    env.reset(seed=81)
    step(30)
    seat_construct(scene.adapter, 0.0, 0.010, settle_steps=40)  # 11 mm short of seated
    report("near-miss-seat")
    s, ok = judge()
    t = scene.finger_tip_dock()[0]
    check("near-miss seat: adapter fingers only partway into the slots (tip readback "
          f"{float(t[1]):+.4f} < seat_y_min {c.seat_y_min:.3f}) — partial gated "
          f"credit only (0.03 <= {s:.3f} <= 0.22), seated() false, NOT success",
          float(t[1]) < c.seat_y_min - 0.004 and not bool(scene.adapter_seated()[0])
          and 0.03 <= s <= 0.22 and not ok)

    # =========================== 11. deck percher (yaw-90 drop) ============================
    env.reset(seed=91)
    step(30)
    seat_construct(scene.adapter, 0.0, c.seat_tip_y - 0.001, settle_steps=30)
    ap = scene.adapter.data.root_pos_w[0]
    ayaw = body_yaw(scene.adapter)
    # prongs-down but yawed 90 deg: the prong line lands across SOLID deck
    write_pose(scene.charger,
               float(ap[0] - scene.env_origins[0][0]),
               float(ap[1] - scene.env_origins[0][1]),
               float(ap[2] - scene.env_origins[0][2]) + c.deck_dz + c.prong_tip_dy
               + 0.004,
               quat=yaw_x_quat(ayaw + math.pi / 2, -math.pi / 2))
    tip_min_rel = 1e9
    for _ in range(120):
        step(1)
        tz = prong_tips_in(scene.adapter)[:, 2]
        tip_min_rel = min(tip_min_rel, float(tz.max()) - c.deck_dz)
    step(60)
    report("deck-percher")
    s, ok = judge()
    check("deck percher: charger dropped prongs-down but yawed 90 deg onto the "
          f"seated adapter's deck — the tips really reached the deck (came within "
          f"{tip_min_rel * 1000:+.1f} mm of the deck plane, non-vacuous) but landed "
          "on SOLID deck between the sockets: zero mate credit, seat credit only "
          f"(score {s:.3f} <= 0.37), NOT success",
          tip_min_rel < 0.004 and float(scene.mate_latch[0]) == 0.0
          and s <= 0.37 and not ok)

    # =========================== 12. wrong station spot (contact) ==========================
    # The green adapter pushed at the panel BESIDE the slots: staged at dock x
    # = +22 mm, its fingers meet the right cheek and the divider — not the slots.
    env.reset(seed=101)
    step(30)
    seat_construct(scene.adapter, 0.022, -0.010, settle_steps=10)  # tips 10 mm short
    tip_start = float(scene.finger_tip_dock()[0, 1])
    tip_max = -1.0
    for _ in range(240):
        lat_w, in_w = dock_axes()
        _, gy = station_pose()
        apply_wrench(scene.adapter, in_w * 4.0,
                     orient_pd(scene.adapter, yaw_quat(gy), 0.08, 0.006, 0.08))
        step(1)
        tip_max = max(tip_max, float(scene.finger_tip_dock()[0, 1]))
    clear_wrench(scene.adapter)
    step(40)
    report("wrong-spot")
    s, ok = judge()
    check("wrong station spot: 2 s of a 4 N push of the adapter at the panel BESIDE "
          f"the slots (staged at dock x = +22 mm) — it really advanced (tip "
          f"{tip_start:+.3f} -> max {tip_max:+.3f}, non-vacuous) but the fingers met "
          "cheek + divider and never passed the face, zero seat credit, score <= "
          "0.02, no success",
          tip_max > tip_start + 0.005 and tip_max < c.seat_gate_y0
          and float(scene.seat_latch[0]) == 0.0 and s <= 0.02 and not ok)

    # =========================== 13. monotonicity ==========================================
    env.reset(seed=111)
    step(10)
    seat_construct(scene.adapter, 0.0, 0.008, settle_steps=8)
    a_part = float(scene.seat_latch[0])
    seat_construct(scene.adapter, 0.0, 0.016, settle_steps=8)
    a_deep = float(scene.seat_latch[0])
    judge()
    check("monotonicity: a deeper finger insertion latches strictly more seat credit "
          f"({a_part:.3f} < {a_deep:.3f})", a_part + 0.10 < a_deep)

    # =========================== 14-15. audit + no-NaN =====================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.station.data.root_state_w).all()
           and torch.isfinite(scene.adapter.data.root_state_w).all()
           and torch.isfinite(scene.decoy.data.root_state_w).all()
           and torch.isfinite(scene.charger.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict ============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.gauge_adapter")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, ok in checks:
            if not ok:
                print(f"[smoke]   FAILED: {nm}", flush=True)
    code = 0 if all_ok else 1
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
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)

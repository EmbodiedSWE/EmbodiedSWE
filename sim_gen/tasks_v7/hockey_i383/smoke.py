"""Smoke / rubric-REJECTION battery for PolarityDockScene (sim_gen task
`hockey_i383`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — both batteries dropped tip-to-terminal into
their bays, lid slid closed by force — is the acceptance evidence that the
rubric ACCEPTS a correct outcome). Every teleport here is instrumentation that
CONSTRUCTS a wrong (or partial) outcome as a settled state and asserts the
rubric REJECTS it; no probe in this battery ever reaches success(), and a final
audit check asserts exactly that.

  1.  settle/no-NaN      — reset layout settles finite: lid parked open,
                           batteries at their floor slots, all still; MASS
                           READBACK (root_physx_view.get_masses) proves the
                           authored masses took (custom spawners silently
                           ignore cfg mass_props);
  2.  score ~0 at reset  — no credit for existing;
  3-4. randomization     — READBACK over 8 seeded resets: per-bay terminal side
                           flips (Bernoulli yaw readback on BOTH bays), battery
                           xy jitter + heading spread, apparatus yaw + offset
                           spreads are real;
  5.  null policy        — 240 idle steps -> score ~0, no success, lid open;
  6.  seed strategy      — the end state the SEED's plan produces here (propel
                           the puck at the goal): a battery FIRED along the
                           floor at the dock bounces off the cassette outer
                           wall and ends outside -> score ~0 (the troughs are
                           only reachable from above);
  7.  empty-close        — with NO batteries seated the lid servo closes the
                           lid FULLY (free-travel proof: the jam probe below is
                           not vacuous) -> lid_closed() true but NO lid credit
                           and NOT success (lid credit requires a seated pair);
  8.  on-lid             — battery dropped onto the CLOSED lid rests on top ->
                           never seated, no credit;
  9.  reversed polarity  — battery dropped into a bay TIP-AWAY from the
                           terminal: 82 mm against a 76 mm trough — it rests
                           TILTED / PROUD of the deck (geometry readback) and
                           never seats;
  10. lid jam            — the same lid servo (stall-escalating force cap, same
                           as the empty-close that DID reach the stop) is
                           stopped >= 8 mm short of closed by the proud
                           battery after real travel (> 40 mm) -> no lid
                           credit, no success: wrong polarity is mechanically
                           un-closable;
  11. across-deck        — battery laid across the deck on top of the side
                           walls -> not seated (z band rejects);
  12. vertical           — battery standing upright in a trough -> not seated
                           (flatness + z band reject);
  13. beside             — battery settled on the floor against the cassette's
                           OUTSIDE wall -> bay-frame math rejects;
  14. one-seated partial — exactly one battery seated properly = exactly the
                           0.28 seat credit; the lid then closed fully over the
                           HALF-EMPTY dock earns NO lid credit and NOT success
                           (both bays must be filled);
  15. latched credit     — after that battery is teleported back out the
                           latched 0.28 is unchanged, still no success;
  16. rejection audit    — success() was never True at ANY judged point;
  17. final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.<task>.smoke --headless
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
# RTX recipe: kit mis-decodes the L20 driver version and silently rejects RTX -> the
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
    env = ENVS.get("simgen.polarity_dock")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.55, -0.62, 0.42)) + o),
                                tuple(np.array((0.00, -0.04, 0.03)) + o),
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

    def lidq() -> float:
        return float(scene.lid_q()[0])

    def bat_bay_local(i: int, j: int):
        bay, bat = scene.bays[j], scene.bats[i]
        return quat_apply_inverse(bay.data.root_quat_w,
                                  bat.data.root_pos_w - bay.data.root_pos_w)[0]

    def bat_flat(i: int) -> float:
        """|tip axis dot world up| for battery i (0 = lying flat)."""
        ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
        return float(quat_apply(scene.bats[i].data.root_quat_w, ez)[0, 2].abs())

    def report(tag: str) -> None:
        s, ok = judge()
        seated = scene.seated()[0]
        l0 = bat_bay_local(0, 0)
        l1 = bat_bay_local(1, 1)
        print(f"[smoke] {tag:16s} | lid_q={lidq() * 1000:+7.1f}mm "
              f"b0@bay0=({float(l0[0]) * 1000:+5.1f},{float(l0[1]) * 1000:+5.1f},"
              f"{float(l0[2]) * 1000:+5.1f})mm "
              f"b1@bay1=({float(l1[0]) * 1000:+5.1f},{float(l1[1]) * 1000:+5.1f},"
              f"{float(l1[2]) * 1000:+5.1f})mm "
              f"seat=({bool(seated[0])},{bool(seated[1])}) "
              f"ever=({bool(scene._seat_ever[0, 0])},{bool(scene._seat_ever[0, 1])},"
              f"{bool(scene._lid_ever[0])}) score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    qy90 = torch.tensor([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0],
                        device=device).expand(n, 4)
    qy90n = torch.tensor([math.cos(math.pi / 4), 0.0, -math.sin(math.pi / 4), 0.0],
                         device=device).expand(n, 4)
    qx90n = torch.tensor([math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0],
                         device=device).expand(n, 4)

    def place_bat_bay(i: int, j: int, x: float, y: float, z: float, q_loc) -> None:
        """Teleport battery i to a bay-j-frame pose (probe constructor)."""
        bay = scene.bays[j]
        loc = torch.tensor([x, y, z], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = bay.data.root_pos_w + quat_apply(bay.data.root_quat_w, loc)
        st[:, 3:7] = quat_mul(bay.data.root_quat_w, q_loc)
        scene.bats[i].write_root_state_to_sim(st, all_ids)

    def place_bat_frame(i: int, x: float, y: float, z: float, q_loc,
                        vel=None) -> None:
        """Teleport battery i to a FRAME-frame pose, optional frame-frame velocity."""
        fr = scene.frame
        loc = torch.tensor([x, y, z], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = fr.data.root_pos_w + quat_apply(fr.data.root_quat_w, loc)
        st[:, 3:7] = quat_mul(fr.data.root_quat_w, q_loc)
        if vel is not None:
            v = torch.tensor(vel, device=device).expand(n, 3)
            st[:, 7:10] = quat_apply(fr.data.root_quat_w, v)
        scene.bats[i].write_root_state_to_sim(st, all_ids)

    def seat_battery(i: int, j: int) -> None:
        """Drop battery i tip-to-terminal into bay j and square it home (the
        solve's placement, reused as a probe constructor for partial credit)."""
        place_bat_bay(i, j, c.seat_x, 0.0, 0.055, qy90)
        step(90)
        bay = scene.bays[j]
        push_dir = quat_apply(bay.data.root_quat_w,
                              torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))
        for _ in range(90):
            vel = scene.bats[i].data.root_lin_vel_w[0]
            f = 0.35 if float((vel * push_dir[0]).sum()) < 0.05 else 0.0
            scene.bats[i].set_external_force_and_torque(
                (push_dir * f).view(n, 1, 3).contiguous(), zero_wrench,
                env_ids=all_ids, is_global=True)
            env.step(no_action)
        scene.bats[i].set_external_force_and_torque(zero_wrench, zero_wrench,
                                                    env_ids=all_ids)
        step(60)

    def close_lid(max_steps: int, cap0: float = 2.0) -> tuple[float, float]:
        """The solve's lid servo (bang-bang velocity cap + stall-escalating force
        cap). Returns (q_start, q_end at force-off). Used BOTH to prove free
        travel (empty dock -> reaches the stop) and to prove the jam (proud
        battery -> stops short DESPITE the escalation)."""
        fq = scene.frame.data.root_quat_w
        close_dir = quat_apply(fq, torch.tensor([0.0, -1.0, 0.0],
                                                device=device).expand(n, 3))
        q_start = lidq()
        cap = cap0
        ref_q, ref_i = q_start, 0
        for i in range(max_steps):
            q = lidq()
            if q < 0.0015:
                break
            v = -(scene.lid.data.root_lin_vel_w[0] * close_dir[0]).sum()
            f = cap if float(v) < 0.10 else 0.0
            scene.lid.set_external_force_and_torque(
                (close_dir * f).view(n, 1, 3).contiguous(), zero_wrench,
                env_ids=all_ids, is_global=True)
            env.step(no_action)
            if i - ref_i >= 60:
                q_now = lidq()
                if ref_q - q_now < 0.002 and q_now > 0.004:
                    cap = min(cap * 1.7, 20.0)
                    print(f"[smoke] lid stalled at q={q_now * 1000:.1f}mm — "
                          f"force cap -> {cap:.2f} N", flush=True)
                ref_q, ref_i = q_now, i
        scene.lid.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                env_ids=all_ids)
        step(45)
        return q_start, lidq()

    def rel_yaw_deg(bay) -> float:
        fq = scene.frame.data.root_quat_w[0]
        conj = torch.stack([fq[0], -fq[1], -fq[2], -fq[3]]).unsqueeze(0)
        q_rel = quat_mul(conj, bay.data.root_quat_w[0:1])[0]
        return math.degrees(2.0 * math.atan2(float(q_rel[3]), float(q_rel[0])))

    def frame_yaw_deg() -> float:
        q = scene.frame.data.root_quat_w[0]
        return math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))

    def finite_all() -> bool:
        bodies = [scene.frame, scene.lid, *scene.bays, *scene.bats]
        return all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)

    # =========================== 1-2. settle / no-NaN / masses ==============================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(120)
    report("show")
    masses_ok = True
    try:
        mf = float(scene.frame.root_physx_view.get_masses()[0, 0])
        ml = float(scene.lid.root_physx_view.get_masses()[0, 0])
        m0 = float(scene.bats[0].root_physx_view.get_masses()[0, 0])
        m1 = float(scene.bats[1].root_physx_view.get_masses()[0, 0])
        print(f"[smoke] mass readback: frame={mf:.1f} lid={ml:.3f} "
              f"bat0={m0:.3f} bat1={m1:.3f}", flush=True)
        masses_ok = (mf > 30.0 and abs(ml - c.lid_mass) < 0.05 * c.lid_mass
                     and abs(m0 - c.bat_mass) < 0.05 * c.bat_mass
                     and abs(m1 - c.bat_mass) < 0.05 * c.bat_mass)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] mass readback FAILED ({exc!r})", flush=True)
        masses_ok = False
    still = (float(scene.bats[0].data.root_lin_vel_w[0].norm()) < c.settle_lin
             and float(scene.bats[1].data.root_lin_vel_w[0].norm()) < c.settle_lin
             and float(scene.lid.data.root_lin_vel_w[0].norm()) < c.lid_settle)
    check("settle: states finite, lid parked open (q > 90 mm), batteries at their "
          "floor slots unseated, all still; authored masses verified by readback",
          finite_all() and lidq() > 0.090 and still
          and not bool(scene.seated()[0].any()) and masses_ok)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        b0 = (scene.bats[0].data.root_pos_w - scene.env_origins)[0]
        fp = (scene.frame.data.root_pos_w - scene.env_origins)[0]
        ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
        head = quat_apply(scene.bats[0].data.root_quat_w, ez)[0]
        reads.append((
            1.0 if abs(rel_yaw_deg(scene.bays[0])) > 90.0 else 0.0,
            1.0 if abs(rel_yaw_deg(scene.bays[1])) > 90.0 else 0.0,
            float(b0[0]), float(b0[1]),
            math.degrees(math.atan2(float(head[1]), float(head[0]))),
            frame_yaw_deg(), float(fp[0]), float(fp[1]),
        ))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (bay0_flip, bay1_flip, b0_x, b0_y, "
          f"b0_heading_deg, frame_yaw_deg, frame_x, frame_y):\n{arr}", flush=True)
    check("randomization: per-bay terminal side flips across seeded resets "
          "(Bernoulli readback on BOTH bay cassettes)",
          0.0 < arr[:, 0].mean() < 1.0 and 0.0 < arr[:, 1].mean() < 1.0)
    jit = float((arr[:, 2:4].max(axis=0) - arr[:, 2:4].min(axis=0)).max())
    hd = np.sort(arr[:, 4])
    head_spread = float(hd.max() - hd.min())
    yaw_spread = float(arr[:, 5].max() - arr[:, 5].min())
    off_spread = float((arr[:, 6:8].max(axis=0) - arr[:, 6:8].min(axis=0)).max())
    check("randomization: battery xy jitter (> 8 mm), battery heading spread "
          "(> 60 deg), apparatus yaw (> 2 deg) + offset (> 8 mm) spreads are real",
          jit > 0.008 and head_spread > 60.0 and yaw_spread > 2.0 and off_spread > 0.008)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0, no success, lid still parked open after 240 idle steps",
          s <= 0.02 and not ok and lidq() > 0.080)

    # =========================== 6. seed strategy: fire a battery at the dock ===============
    # The seed's plan — propel the payload at the goal — executed here: a battery
    # FIRED along the floor at the dock hits the cassette outer wall and stays
    # OUTSIDE: the troughs are only reachable from above.
    torch.manual_seed(41)
    env.reset()
    step(30)
    place_bat_frame(0, 0.0, -0.24, c.body_r + 0.002, qy90, vel=(0.0, 1.5, 0.0))
    p0 = scene.bats[0].data.root_pos_w[0].clone()
    step(300)
    report("seed-strategy")
    p1 = scene.bats[0].data.root_pos_w[0]
    traveled = float((p1 - p0)[:2].norm())
    s, ok = judge()
    check("seed strategy: battery FIRED along the floor at the dock travels, bounces "
          "off the cassette outer wall and never seats — score ~0 (<= 0.02), no "
          "success",
          traveled > 0.10 and not bool(scene.seated()[0].any()) and not ok and s <= 0.02)

    # =========================== 7. empty-close: free travel, no credit =====================
    torch.manual_seed(51)
    env.reset()
    step(30)
    q_start, q_end = close_lid(720)
    report("empty-close")
    s, ok = judge()
    check("empty-close: with NO batteries seated the lid servo drives the lid to its "
          "closed stop (free-travel proof — the jam probe is not vacuous) but earns "
          "NO lid credit and NOT success (lid credit requires a seated pair): "
          f"q {q_start * 1000:.1f} -> {q_end * 1000:.1f} mm, score <= 0.02",
          q_end < c.closed_tol and bool(scene.lid_closed()[0])
          and not bool(scene._lid_ever[0]) and not ok and s <= 0.02)

    # =========================== 8. on-lid: battery on the closed lid =======================
    place_bat_bay(0, 0, c.seat_x, 0.0, 0.075, qy90)  # over bay0 but the lid is shut
    step(180)
    report("on-lid")
    l0 = bat_bay_local(0, 0)
    s, ok = judge()
    check("on-lid: battery dropped over the bay lands ON the closed lid (or rolls "
          "off) — never seated, no seat credit, no success",
          not bool(scene.seated()[0, 0]) and not bool(scene._seat_ever[0, 0])
          and not ok and s <= 0.02)

    # =========================== 9-10. reversed polarity + lid jam ==========================
    torch.manual_seed(61)
    env.reset()
    step(30)
    # tip AWAY from the terminal: 82 mm battery vs 76 mm trough — cannot lie flat
    place_bat_bay(0, 0, c.seat_x, 0.0, 0.055, qy90n)
    step(150)
    report("reversed")
    l0 = bat_bay_local(0, 0)
    flat0 = bat_flat(0)
    proud = float(l0[2]) > 0.028 or flat0 > c.seat_flat_max
    check("reversed polarity: battery dropped tip-AWAY from the terminal rests "
          f"TILTED/PROUD (bay-z {float(l0[2]) * 1000:.1f} mm, |axis.z| {flat0:.2f}) "
          "and never seats — no seat credit",
          proud and not bool(scene.seated()[0, 0]) and not bool(scene._seat_ever[0, 0]))
    q_start, q_end = close_lid(600)
    report("jam")
    s, ok = judge()
    # The proud battery in bay0 blocks the lid's leading edge almost immediately
    # (contact at q ~ 89 mm), so the free approach from the ~96-106 mm park is
    # only ~7-17 mm; the free-travel proof is the empty-close check above. What
    # must hold: the servo really approached (> 5 mm) and the jam held FAR short
    # of closed (>= 30 mm) despite the force cap escalating to 20 N.
    check("lid jam: the SAME escalating-force lid servo that closed the empty dock "
          f"is stopped by the proud battery ({q_start * 1000:.1f} -> "
          f"{q_end * 1000:.1f} mm: approached > 5 mm, jammed >= 30 mm short of "
          "closed at a 20 N cap) — no lid credit, no success: wrong polarity is "
          "mechanically un-closable",
          q_start - q_end > 0.005 and q_end > 0.030
          and not bool(scene._lid_ever[0]) and not ok and s <= 0.02)

    # =========================== 11. across-deck ============================================
    torch.manual_seed(71)
    env.reset()
    step(30)
    place_bat_frame(0, 0.0, 0.0, c.bay_floor_t + c.trough_d + c.body_r + 0.003, qx90n)
    step(180)
    report("across-deck")
    l0 = bat_bay_local(0, 0)
    s, ok = judge()
    check("across-deck: battery laid ACROSS the deck on top of the side walls rests "
          f"above the seat band (bay-z {float(l0[2]) * 1000:.1f} mm) — not seated, "
          "no credit",
          not bool(scene.seated()[0].any()) and not bool(scene._seat_ever[0].any())
          and not ok and s <= 0.02)

    # =========================== 12. vertical in the trough =================================
    place_bat_bay(0, 0, -0.015, 0.0, 0.046, torch.tensor(
        [1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4))
    step(180)
    report("vertical")
    s, ok = judge()
    check("vertical: battery standing upright in the trough (or toppled onto the "
          "apparatus) is rejected by the flatness/z gates — not seated, no credit",
          not bool(scene.seated()[0].any()) and not bool(scene._seat_ever[0].any())
          and not ok and s <= 0.02)

    # =========================== 13. beside the cassette ====================================
    torch.manual_seed(81)
    env.reset()
    step(30)
    place_bat_bay(0, 0, c.seat_x, c.bay_half_y + c.body_r + 0.004,
                  c.body_r + 0.002, qy90)
    step(150)
    report("beside")
    l0 = bat_bay_local(0, 0)
    s, ok = judge()
    check("wrong place: battery settled on the floor against the cassette's OUTSIDE "
          f"wall at seat height/heading (bay-y {float(l0[1]) * 1000:.1f} mm) — "
          "bay-frame math rejects: not seated, score ~0",
          abs(float(l0[1])) > c.seat_y_tol and not bool(scene.seated()[0, 0])
          and not ok and s <= 0.02)

    # =========================== 14-15. one-seated partial + latched credit =================
    torch.manual_seed(91)
    env.reset()
    step(30)
    seat_battery(0, 0)
    report("one-seated")
    s14a, ok14a = judge()
    one_ok = (bool(scene.seated()[0, 0]) and not bool(scene.seated()[0, 1])
              and 0.27 <= s14a <= 0.29 and not ok14a)
    q_start, q_end = close_lid(720)
    report("half-empty-close")
    s14b, ok14b = judge()
    check("one-seated partial: exactly one battery seated = exactly the 0.28 seat "
          "credit; the lid closed fully over the HALF-EMPTY dock earns NO lid "
          f"credit and NOT success (score stays {s14b:.3f}) — both bays must be "
          "filled",
          one_ok and q_end < c.closed_tol and not bool(scene._lid_ever[0])
          and not ok14b and 0.27 <= s14b <= 0.29)
    # remove the seated battery: the latched credit must survive, still no success
    place_bat_frame(0, 0.20, -0.30, c.body_r + 0.002, qy90)
    step(60)
    report("removed")
    s15, ok15 = judge()
    check("latched credit: after the seated battery is teleported back OUT the "
          f"latched seat credit is unchanged ({s14b:.3f} -> {s15:.3f}), still no "
          "success",
          abs(s15 - s14b) < 1e-3 and not bool(scene.seated()[0, 0]) and not ok15)

    # =========================== 16-17. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.polarity_dock")
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
    except Exception as e:  # noqa: BLE001 - fast fail beats a 20-min watchdog hang
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)

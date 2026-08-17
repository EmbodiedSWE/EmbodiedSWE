"""Smoke / rubric-REJECTION battery for JengaQuarryScene (sim_gen task
`block_pyramid_i366`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — two contact Jenga slides under the roof load,
then tray drops — is the acceptance evidence that the rubric ACCEPTS a correct
outcome). Every teleport here is instrumentation that CONSTRUCTS a wrong (or
partial) outcome and asserts the rubric REJECTS it; no probe in this battery ever
reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN     — reset layout settles finite: all 9 blocks in their
                           tower slots, still, score ~0, no success;
  3-4. randomization     — READBACK over 6 seeded resets: rig xy + yaw and tray
                           jitter really vary; WHICH slot is red varies per layer;
  5.  null policy        — 240 idle steps -> score ~0, no success;
  6.  seed strategy      — the nearest executable version of the SEED's plan
                           (STACK blocks up into a structure): both reds placed on
                           TOP of the tower as a fourth layer -> tower fine, but
                           NOT success and score ~0 (the goal is extraction to the
                           tray, not stacking);
  7.  topple-and-collect — the shortcut the task exists to reject: creams
                           scattered to the floor FIRST (the toppled-tower
                           rubble), THEN both reds dropped into the tray -> NOT
                           success and score ~0, because every latch is gated on
                           tower_standing() at the moment it is earned;
  8.  partial            — ONE red banked in the tray (tower intact) -> NOT
                           success, score in the partial band (<= 0.35);
  9.  near-miss          — red beside the tray on the table, and red held at the
                           tray's RIM height -> in_tray() False both times;
  10. wrong objects      — two CREAM blocks dropped into the tray -> preservation
                           broken, NOT success, no new credit;
  11. airborne           — red in free fall directly over the tray, judged
                           mid-air -> not in_tray (resting-height gate; relocated
                           before it can land);
  12. latched credit     — returning the banked red to its tower slot leaves the
                           latched score UNCHANGED, still no success;
  13. real Jenga slide   — red1 contact-slid out of the MIDDLE layer exactly like
                           solve.py (velocity-capped axial force, roof load
                           riding on it): slide+extract latches earn >= 0.15,
                           tower still standing, NOT success;
  14. roof-pin denial    — 2 N straight up on a BURIED red barely moves it (the
                           superstructure pins it), while the SAME pull on a free
                           block on the table lifts it away — the "just lift it
                           out" strategy is physically denied, non-vacuously;
  15. rejection audit    — success() was never True at ANY judged point;
  16. final no-NaN       — all task-object states finite at the end.

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
# RTX recipe: kit mis-decodes some driver versions and silently rejects RTX -> the
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
    from . import scene as scene_mod  # noqa: F401  (registers simgen.jenga_quarry)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.jenga_quarry")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)
    length, w, h = c.block_size

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.72, -0.85, 0.60)) + o),
                                tuple(np.array((0.15, 0.00, 0.08)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video",
              flush=True)

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
        sc, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return sc, ok

    def report(tag: str) -> None:
        p0, p1 = scene.block_local(0)[0], scene.block_local(1)[0]
        sc, ok = judge()
        print(f"[smoke] {tag:18s} | red0_l=({float(p0[0]):+.3f},{float(p0[1]):+.3f},"
              f"{float(p0[2]):.3f}) red1_l=({float(p1[0]):+.3f},{float(p1[1]):+.3f},"
              f"{float(p1[2]):.3f}) standing={bool(scene.tower_standing()[0])} "
              f"slid={[bool(v) for v in scene._slid[0]]} "
              f"ext={[bool(v) for v in scene._ext[0]]} "
              f"tray={[bool(v) for v in scene._tray_in[0]]} "
              f"score={sc:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, x: float, y: float, z: float, yaw_extra: float = 0.0) -> None:
        """Probe constructor: teleport a body to a RIG-LOCAL pose (rig yaw +
        yaw_extra orientation), zero velocity."""
        yaw = float(scene._rig_yaw[0]) + yaw_extra
        pos_l = torch.tensor([[float(x), float(y), float(z)]], device=device)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.to_world(pos_l, all_ids) + scene.env_origins
        st[:, 3] = math.cos(yaw / 2)
        st[:, 6] = math.sin(yaw / 2)
        body.write_root_state_to_sim(st, all_ids)

    def tray_l(dx: float = 0.0, dy: float = 0.0) -> tuple:
        return (float(scene._tray_xy[0, 0]) + dx, float(scene._tray_xy[0, 1]) + dy)

    def still() -> bool:
        return bool(scene.settled()[0])

    def finite_all() -> bool:
        ok = True
        for b in scene.blocks:
            ok = ok and bool(torch.isfinite(b.data.root_state_w).all())
        return ok

    def local_dir_to_world(dx: float, dy: float) -> torch.Tensor:
        yaw = float(scene._rig_yaw[0])
        return torch.tensor([dx * math.cos(yaw) - dy * math.sin(yaw),
                             dx * math.sin(yaw) + dy * math.cos(yaw), 0.0],
                            device=device)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    report("reset")
    step(90)
    report("show")
    slots_ok = True
    for i in range(9):
        d = scene.block_local(i)[0] - scene._rst_local[0, i]
        slots_ok = slots_ok and float(d[0:2].norm()) < 0.006 and abs(float(d[2])) < 0.006
    sc, ok = judge()
    check("settle: states finite, all 9 blocks resting in their tower slots, still",
          finite_all() and slots_ok and still())
    check("settle: score ~0 at reset (<= 0.02), no success", sc <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(5)
        reads.append((float(scene._rig_xy[0, 0]), float(scene._rig_xy[0, 1]),
                      math.degrees(float(scene._rig_yaw[0])),
                      float(scene._tray_xy[0, 0]), float(scene._tray_xy[0, 1]),
                      int(scene._slot[0, 0]), int(scene._slot[0, 1])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (rig_x, rig_y, rig_yaw_deg, tray_x, "
          f"tray_y, slot_L0, slot_L1):\n{arr}", flush=True)
    rig_spread = arr[:, 0:2].max(axis=0) - arr[:, 0:2].min(axis=0)
    yaw_spread = float(arr[:, 2].max() - arr[:, 2].min())
    tray_spread = arr[:, 3:5].max(axis=0) - arr[:, 3:5].min(axis=0)
    check("randomization: rig xy spread > 10 mm on both axes, rig yaw spread > "
          "10 deg, tray jitter spread > 8 mm (readback)",
          float(rig_spread.min()) > 0.010 and yaw_spread > 10.0
          and float(tray_spread.max()) > 0.008)
    s0_vals, s1_vals = set(int(r[5]) for r in reads), set(int(r[6]) for r in reads)
    pairs = set((int(r[5]), int(r[6])) for r in reads)
    check("randomization: WHICH slot is red varies — layer-0 and layer-1 red slots "
          "each take >= 2 values across 6 seeds, >= 3 distinct slot pairs (readback)",
          len(s0_vals) >= 2 and len(s1_vals) >= 2 and len(pairs) >= 3)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    sc, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          sc <= 0.02 and not ok)

    # =========================== 6. seed strategy: stack blocks up ==========================
    # The seed's plan is CONSTRUCTIVE: pick blocks and stack them into a structure.
    # Nearest executable end state here: both reds pulled out and STACKED ON TOP of
    # the tower as a fourth layer (perpendicular, like the pattern continues).
    env.reset(seed=41)
    step(30)
    # a 4th layer runs along local y (perpendicular to layer 2), so its blocks
    # occupy distinct X slots — two y-axis blocks at x=0 would interpenetrate
    z_top = c.layer_z(2) + h + 0.001
    place(scene.blocks[0], -c.pitch, 0.0, z_top, yaw_extra=math.pi / 2)
    place(scene.blocks[1], +c.pitch, 0.0, z_top, yaw_extra=math.pi / 2)
    step(150)
    report("seed-strategy")
    p0, p1 = scene.block_local(0)[0], scene.block_local(1)[0]
    sc, ok = judge()
    check("seed strategy: reds STACKED on top of the tower (the seed's constructive "
          "plan) — tower fine but NOT success and score <= 0.02 (the goal is "
          "extraction to the tray, not stacking)",
          still() and bool(scene.tower_standing()[0])
          and float(p0[2]) > c.layer_z(2) + h / 2 and float(p1[2]) > c.layer_z(2) + h / 2
          and not ok and sc <= 0.02)

    # =========================== 7. topple-and-collect ======================================
    # The shortcut this task exists to reject: wreck the tower, pick the reds from
    # the rubble. Constructed in the WORST order for the rubric: creams scattered
    # FIRST (so the reds are genuinely free), then both reds dropped into the tray.
    env.reset(seed=51)
    step(30)
    rubble = [(-0.22, -0.18), (-0.25, 0.00), (-0.22, 0.18), (-0.05, -0.25),
              (-0.05, 0.25), (0.12, -0.25), (0.12, 0.25)]
    for j, (rx, ry) in enumerate(rubble):
        place(scene.blocks[2 + j], rx, ry, h / 2 + 0.003, yaw_extra=0.35 * j)
    step(90)
    tx, ty = tray_l()
    place(scene.blocks[0], tx - 0.040, ty, 0.10, yaw_extra=math.pi / 2)
    step(90)
    place(scene.blocks[1], tx + 0.040, ty, 0.10, yaw_extra=math.pi / 2)
    step(150)
    report("topple-collect")
    sc, ok = judge()
    both_in = bool(scene.in_tray(0)[0]) and bool(scene.in_tray(1)[0])
    check("topple-and-collect: creams scattered to the floor FIRST, then BOTH reds "
          "dropped into the tray — reds really are in the tray, but NOT success "
          "and score <= 0.02 (every latch is gated on the tower standing when the "
          "credit is earned)",
          both_in and not bool(scene.tower_standing()[0]) and not ok and sc <= 0.02)

    # =========================== 8 + 12. partial, then latched credit =======================
    env.reset(seed=61)
    step(30)
    tx, ty = tray_l()
    place(scene.blocks[0], tx - 0.040, ty, 0.10, yaw_extra=math.pi / 2)
    step(150)
    report("one-red-banked")
    sc8, ok = judge()
    check("partial: ONE red banked in the tray, tower intact — NOT success, score "
          "in the partial band (0.15 <= score <= 0.35, float32 eps)",
          bool(scene.in_tray(0)[0]) and bool(scene.tower_standing()[0])
          and not ok and 0.145 <= sc8 <= 0.355)
    # ... and the latched credit survives regression (12): put it back in its slot
    rst = scene._rst_local[0, 0]
    place(scene.blocks[0], float(rst[0]), float(rst[1]), float(rst[2]) + 0.001)
    step(90)
    report("regressed")
    sc12, ok = judge()
    check("latched credit: returning the banked red to its tower slot leaves the "
          f"latched score unchanged ({sc8:.3f} -> {sc12:.3f}), still no success",
          abs(sc12 - sc8) < 1e-3 and not ok and not bool(scene.in_tray(0)[0]))

    # =========================== 9. near-miss beside / at rim height ========================
    env.reset(seed=71)
    step(30)
    tx, ty = tray_l()
    # long axis along local x (yaw_extra 0), PARALLEL to the +y wall: the
    # across-wall half-extent is then really w/2 (with pi/2 it is length/2 and
    # the block lands lying across the wall)
    beside_y = ty + c.tray_inner / 2 + c.tray_wall_t + w / 2 + 0.010
    place(scene.blocks[0], tx, beside_y, h / 2 + 0.002, yaw_extra=0.0)
    step(120)
    report("beside-tray")
    beside_ok = still() and not bool(scene.in_tray(0)[0])
    # rim-height denial: held at the wall-top height over the tray centre, judged
    # immediately (the z band is load-bearing; relocated before it can settle in)
    rim_z = c.tray_floor_top + c.tray_wall_h + h / 2 + 0.002
    place(scene.blocks[0], tx, ty, rim_z, yaw_extra=math.pi / 2)
    rim_ok = not bool(scene.in_tray(0)[0])
    judge()
    place(scene.blocks[0], tx, beside_y, h / 2 + 0.002, yaw_extra=0.0)
    step(60)
    check("near-miss: red settled beside the tray -> in_tray False; red at the "
          "tray's RIM height over the centre -> in_tray False (xy box and z band "
          "both load-bearing)", beside_ok and rim_ok)

    # =========================== 10. wrong objects in the tray ==============================
    env.reset(seed=81)
    step(30)
    tx, ty = tray_l()
    place(scene.blocks[8], tx - 0.040, ty, 0.10, yaw_extra=math.pi / 2)  # top-layer cream
    step(90)
    place(scene.blocks[7], tx + 0.040, ty, 0.10, yaw_extra=math.pi / 2)
    step(150)
    report("creams-in-tray")
    sc, ok = judge()
    check("wrong objects: two CREAM blocks dropped into the tray — preservation "
          "broken, NOT success, score <= 0.02 (only the reds are the cargo)",
          not bool(scene.tower_standing()[0]) and not ok and sc <= 0.02)

    # =========================== 11. airborne over the tray =================================
    env.reset(seed=91)
    step(30)
    tx, ty = tray_l()
    place(scene.blocks[0], tx, ty, 0.28, yaw_extra=math.pi / 2)
    step(2)
    p = scene.block_local(0)[0]
    air_ok = (float(p[2]) > c.tray_z[1] + 0.05 and not bool(scene.in_tray(0)[0]))
    judge()
    place(scene.blocks[0], tx, ty + c.tray_inner / 2 + c.tray_wall_t + w / 2 + 0.02,
          h / 2 + 0.002, yaw_extra=0.0)  # relocate pre-landing, parallel to the wall
    step(60)
    check("airborne: red in free fall directly over the tray, judged mid-air — not "
          "in_tray (the resting-height gate is load-bearing)", air_ok)

    # =========================== 13. the real Jenga slide (contact physics) =================
    # Exactly the solve's load-bearing interaction, in miniature: red1 slid out of
    # the MIDDLE layer by a velocity-capped axial force with the top layer riding
    # on it — earns the slide+extract latches, tower stays standing, NOT success.
    env.reset(seed=101)
    step(30)
    d_w = local_dir_to_world(0.0, 1.0)
    block = scene.blocks[1]
    yaw_target = float(scene._rig_yaw[0]) + math.pi / 2

    # mirror of solve.py's jolt-free extraction: slow feedforward ramp + viscous
    # brake with streak-based breakaway cut, yaw squaring, and (past the outer
    # support edge) a level-hold pitch torque + half-weight grip lift, so the
    # emerging block neither tips the pivot cream below nor presses the roof
    # block above into riding (center-column episodes)
    edge, mgl = c.span_half, c.block_mass * 9.81
    ux, uy = float(d_w[0]), float(d_w[1])
    n_axis = torch.tensor([-uy, ux, 0.0], device=device)
    z_slot = float(scene._rst_local[0, 1, 2])

    def lift_of(s_along: float, z_now: float) -> float:
        if s_along > 0.118 or z_now < z_slot - 0.003:
            return 0.0
        return 0.5 * mgl * min(max((s_along - 0.030) / 0.020, 0.0), 1.0)

    def control_torque(s_along: float, z_now: float, lift: float) -> torch.Tensor:
        q = block.data.root_quat_w[0]
        om = block.data.root_ang_vel_w[0]
        yaw_b = 2.0 * math.atan2(float(q[3]), float(q[0]))
        e = ((yaw_b - yaw_target + math.pi / 2) % math.pi) - math.pi / 2
        tz = max(-0.03, min(0.03, -0.02 * e - 0.002 * float(om[2])))
        qw, qx, qy, qz = (float(v) for v in q)
        ax = 1.0 - 2.0 * (qy * qy + qz * qz)
        ay = 2.0 * (qx * qy + qw * qz)
        az = 2.0 * (qx * qz - qw * qy)
        nose = 1.0 if (ax * ux + ay * uy) >= 0.0 else -1.0
        pitch = az * nose  # >0 = nose up
        wn = float(om @ n_axis)  # >0 = pitching nose-down
        riding = z_now > z_slot - 0.003
        ff = -(mgl - lift) * min(max(s_along - edge, 0.0), 0.065) if riding else 0.0
        tn = max(-0.12, min(0.12, ff + 0.4 * pitch - 0.02 * wn))
        t = torch.zeros(n, 1, 3, device=device)
        t[:, 0, 0] = tn * float(n_axis[0])
        t[:, 0, 1] = tn * float(n_axis[1])
        t[:, 0, 2] = tz
        return t

    F, F_cap, brake, v_cap = 0.4, 6.0, 25.0, 0.020
    mv = st = 0
    moving, armed = False, True
    mark_s, mark_i = -1.0, 0
    cleared = False
    for i in range(7000):
        p = scene.block_local(1)[0]
        hdist = float(p[0:2].norm())
        if hdist > c.ext_dist + 0.010 or float(p[2]) < z_slot - 0.020:
            cleared = True
            break
        v_along = float(torch.dot(block.data.root_lin_vel_w[0], d_w))
        if v_along > 0.008:
            mv, st = mv + 1, 0
        elif v_along < 0.004:
            mv, st = 0, st + 1
        if not moving and mv >= 8:
            moving = True
            if armed:
                F = max(0.3, F * 0.55)
                armed = False
        if moving and st >= 30:
            moving, armed = False, True
        if moving and v_along > v_cap:
            F = max(F - 0.02, 0.25)
        elif not (moving and v_along >= v_cap * 0.6):
            F = min(F + 0.004, F_cap)
        push = max(0.0, F - brake * max(v_along, 0.0))
        s_along = float((p[0:2] - scene._rst_local[0, 1, 0:2]) @
                        torch.tensor([0.0, 1.0], device=device))
        lift = lift_of(s_along, float(p[2]))
        cross = float((p[0:2] - scene._rst_local[0, 1, 0:2]) @ scene._axis[0])
        lat = max(-0.4, min(0.4, -20.0 * cross * 0.02))  # exactly solve's gain
        f_w = d_w * push + local_dir_to_world(lat, 0.0)
        f_w = f_w + torch.tensor([0.0, 0.0, lift], device=device)
        block.set_external_force_and_torque(
            f_w.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            control_torque(s_along, float(p[2]), lift),
            env_ids=all_ids, is_global=True)
        env.step(no_action)
        if s_along > mark_s + 0.004:
            mark_s, mark_i = s_along, i
            worst, wi = 0.0, -1
            for bi in range(2, 9):
                dd = scene.block_local(bi)[0] - scene._rst_local[0, bi]
                v = float(dd[0:2].norm())
                if v > worst:
                    worst, wi = v, bi
            print(f"[smoke] slide: s={s_along:+.3f} F={F:.2f} "
                  f"drift={worst * 1000:.1f}mm(b{wi}) "
                  f"standing={bool(scene.tower_standing()[0])}", flush=True)
        elif i - mark_i > 2000:
            break
    block.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
    for i in range(900):
        p = scene.block_local(1)[0]
        if float(p[0:2].norm()) > c.ext_dist + 0.010:
            break
        v_along = float(torch.dot(block.data.root_lin_vel_w[0], d_w))
        push = 1.2 if v_along < 0.05 else 0.0
        block.set_external_force_and_torque(
            (d_w * push).view(1, 1, 3).expand(n, 1, 3).contiguous(), zero_w,
            env_ids=all_ids, is_global=True)
        env.step(no_action)
    block.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
    step(150)
    report("real-slide")
    sc, ok = judge()
    check("real Jenga slide: red1 contact-slid out of the middle layer under the "
          "top layer's load (velocity-capped axial force, as in solve.py) — "
          "slide+extract latches earn >= 0.15, tower still standing, NOT success",
          cleared and bool(scene._slid[0, 1]) and bool(scene._ext[0, 1])
          and bool(scene.tower_standing()[0]) and not ok and 0.15 <= sc < 0.5)

    # =========================== 14. roof-pin denial (non-vacuous) ==========================
    # "Just lift it out": 2 N straight up on the buried red barely moves it — the
    # superstructure pins it (edge-lifting the layers above needs ~3 N). The SAME
    # 2 N on a free block on the table (weight ~1 N) takes off immediately.
    env.reset(seed=111)
    step(30)
    z0 = float(scene.block_local(0)[0][2])
    f_up = torch.tensor([0.0, 0.0, 2.0], device=device).view(1, 1, 3).expand(n, 1, 3)
    max_rise = 0.0
    for _ in range(120):
        scene.blocks[0].set_external_force_and_torque(
            f_up.contiguous(), zero_w, env_ids=all_ids, is_global=True)
        env.step(no_action)
        max_rise = max(max_rise, float(scene.block_local(0)[0][2]) - z0)
    scene.blocks[0].set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
    step(60)
    pinned_ok = max_rise < 0.008
    print(f"[smoke] roof-pin: buried red rose {max_rise * 1000:.1f} mm under 2 N up",
          flush=True)
    # control: the same pull on a FREE block, away from tower and tray
    place(scene.blocks[0], -0.10, 0.28, h / 2 + 0.002)
    step(60)
    z0 = float(scene.block_local(0)[0][2])
    max_rise_free = 0.0
    for _ in range(40):
        scene.blocks[0].set_external_force_and_torque(
            f_up.contiguous(), zero_w, env_ids=all_ids, is_global=True)
        env.step(no_action)
        max_rise_free = max(max_rise_free, float(scene.block_local(0)[0][2]) - z0)
    scene.blocks[0].set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
    print(f"[smoke] roof-pin control: free block rose {max_rise_free * 1000:.1f} mm "
          f"under the same 2 N", flush=True)
    step(120)
    report("roof-pin")
    check("roof-pin denial: 2 N straight up moves the BURIED red < 8 mm (the "
          "superstructure pins it) while the same pull lifts a FREE block > 40 mm "
          "— lifting out of the tower is physically denied, non-vacuously",
          pinned_ok and max_rise_free > 0.040)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.jenga_quarry")
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
    except Exception as exc:  # noqa: BLE001 — die fast, not at the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL (exception: {exc!r})", flush=True)
        os._exit(2)

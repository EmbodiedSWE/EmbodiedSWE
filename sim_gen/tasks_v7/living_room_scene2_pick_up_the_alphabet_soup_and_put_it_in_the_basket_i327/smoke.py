"""Smoke / rubric-REJECTION battery for PackedToteScene (sim_gen task
`living_room_scene2_pick_up_the_alphabet_soup_and_put_it_in_the_basket_i327`) —
NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — set the can aside, shove the block pair to one
end through contact, lower the can into the cleared slot — is the acceptance
evidence). Every teleport here is instrumentation that CONSTRUCTS a wrong (or
partial) outcome as a settled state and asserts the rubric REJECTS it, plus readback
probes that prove the blocking and the room-making are real. Two probes DO construct
the genuine end state on purpose — the acceptance construct and the block-returned
flip — every other judged point must stay success()=False and a final audit asserts
exactly that.

  1-2. settle/no-NaN      — reset settles finite; blocks parked flat mid-tote, can
                            upright on the floor outside; span readback BLOCKED
                            (every free gap narrower than the can); score ~0;
  3.  mass readback       — tote/blocks/can masses match the cfg (custom compound
                            spawners silently fall back to density mass otherwise);
  4-5. randomization      — READBACK over 8 seeded resets: tote xy + free yaw vary;
                            the pair's parking offset and the can position vary;
  6.  null policy         — 300 idle steps -> score ~0, no success;
  7.  carried, not placed — the can held (posed) at the hover pose above the tote:
                            no credit, no success;
  8.  seed strategy       — the seed's move (drop the object into the container):
                            released over the packed tote it lands STANDING ON TOP
                            OF THE RED BLOCKS ~6 cm above the floor — enter credit
                            only, floor z-window shut, no success;
  9.  side-gap drop       — the can dropped over the WIDEST free side gap (readback:
                            narrower than the can) cannot reach the floor upright —
                            rejected;
  10. room readout real   — blocks teleported consolidated to one end (with the can
                            far away): the conservative span readback crosses the
                            gate and the L2 latch fires — credit 0.25, NOT success;
  11. lying in the slot   — the can laid FLAT on the genuinely cleared floor:
                            upright/z gates reject it, no success;
  12. acceptance construct— the can released upright just above the cleared floor:
                            gravity alone seats it -> success TRUE;
  13. keep-in clause      — one red block teleported OUT of the tote while the can
                            stays perfectly seated: success flips FALSE;
  14. block returned      — the block STACKED back on its partner (stacking is a
                            legal stow): success returns TRUE (13's rejection was
                            the keep-in clause and nothing else);
  15. settle gate         — the seated can kicked and judged immediately: NOT
                            success (must be at rest); destroyed;
  16. wrong object        — a RED BLOCK stood on end in the slot windows (blue can
                            far outside): rejected — the rubric reads the blue can;
  17. rejection audit     — success() was never True at any judged point EXCEPT the
                            two constructed acceptance probes (12 and 14);
  18. final no-NaN        — all task-object states finite at the end.

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
    from .scene import (  # noqa: F401
        BLK_H, BLK_L, BLK_W, IN_Y, L_CAN, R_CAN, RIM_Z, Z_F,
        _qapply, _qmul, _qy, _qz,
    )
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import (  # noqa: F401
        BLK_H, BLK_L, BLK_W, IN_Y, L_CAN, R_CAN, RIM_Z, Z_F,
        _qapply, _qmul, _qy, _qz,
    )
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

W_HALF = IN_Y / 2
HOVER_Z = RIM_Z + L_CAN / 2 + 0.010  # solve's drop-release height (tote frame)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.packed_tote")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.00, -0.85, 0.75)) + o),
                                tuple(np.array((0.00, 0.00, 0.08)) + o),
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

    ever_bad_success = [False]

    def judge() -> tuple[float, bool]:
        """Judge a REJECTION probe: success here is a rubric failure."""
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_bad_success[0] = ever_bad_success[0] or ok
        return s, ok

    def judge_accept() -> tuple[float, bool]:
        """Judge an ACCEPTANCE construct: success here is expected and allowed."""
        return float(scene.score()[0]), bool(scene.success()[0])

    def span() -> float:
        return float(scene.cleared_span()[0])

    def report(tag: str, s: float, ok: bool) -> None:
        loc = scene.tote_local(scene.blue.data.root_pos_w)[0]
        ax = scene.can_axis_local()[0]
        print(f"[smoke] {tag:18s} | blue_tote=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
              f"z-Zf={float(loc[2]) - Z_F:+.3f}) ax_z={float(ax[2]):+.3f} "
              f"span={span() * 1000:5.1f}mm slot={bool(scene.can_in_slot()[0])} "
              f"stowed={bool(scene.blocks_stowed()[0])} "
              f"L1={bool(scene.l1_in_tote[0])} L2={bool(scene.l2_room[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def tote_pose(local, extra_quat=None):
        q_tote = scene.tote.data.root_quat_w
        pos = scene.tote.data.root_pos_w + _qapply(
            q_tote, torch.tensor(local, device=device).expand(n, 3))
        q = q_tote if extra_quat is None else _qmul(q_tote, extra_quat)
        return pos, q

    def teleport(body, pos, quat, vel=None, ang=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3:7] = quat
        if vel is not None:
            st[:, 7:10] = torch.tensor(vel, device=device)
        if ang is not None:
            st[:, 10:13] = torch.tensor(ang, device=device)
        body.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def teleport_floor(body, x: float, y: float, z: float, settle_steps: int = 45) -> None:
        pos = torch.tensor([x, y, z], device=device).expand(n, 3) + scene.env_origins
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4)
        teleport(body, pos, quat, settle_steps=settle_steps)

    half_pi = torch.full((n,), math.pi / 2, device=device)
    q_lie_x = _qy(half_pi)  # can local +z (its axis) -> tote +x (lying along x)
    q_on_end = _qy(half_pi)  # block local +x (its length) -> vertical

    def settle_all(max_steps: int = 600) -> None:
        for _ in range(max_steps // 30):
            step(30)
            if bool((scene.settled(scene.blue) & scene.settled(scene.blk_a)
                     & scene.settled(scene.blk_b) & scene.settled(scene.tote))[0]):
                break

    def consolidate_blocks(s: float, settle_steps: int = 240) -> None:
        """Instrumentation: teleport the pair flush-ish to the s-end of the tote
        (>= 4 mm wall clearance — flush spawns get depenetration-nudged)."""
        y_lead = s * (W_HALF - BLK_W / 2 - 0.004)
        y_trail = y_lead - s * (BLK_W + 0.001)
        for body, y in ((scene.blk_a, y_trail), (scene.blk_b, y_lead)):
            pos, q = tote_pose((0.0, y, Z_F + BLK_H / 2 + 0.001))
            teleport(body, pos, q, settle_steps=0)
        step(settle_steps)

    def slot_center(s: float) -> float:
        """Free-interval center on the -s side (same readback the solve uses)."""
        edges = []
        for body in (scene.blk_a, scene.blk_b):
            y = float(scene.tote_local(body.data.root_pos_w)[0, 1])
            h = float(scene.blk_y_halfwidth(body)[0])
            edges.append(y - h if s > 0 else y + h)
        if s > 0:
            lo, hi = -W_HALF, min(edges)
        else:
            lo, hi = max(edges), W_HALF
        center = 0.5 * (lo + hi)
        return max(lo + R_CAN + 0.0025, min(hi - R_CAN - 0.0025, center))

    def all_finite() -> bool:
        bodies = [scene.tote, scene.blk_a, scene.blk_b, scene.blue]
        return all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)

    def blk_flat(body) -> bool:
        ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
        return float(_qapply(body.data.root_quat_w, ez)[0, 2]) > math.cos(math.radians(15.0))

    # =========================== 1-2. settle / no-NaN / blocked =============================
    env.reset(seed=11)
    settle_all(480)
    s, ok = judge()
    report("reset", s, ok)
    can_up = abs(float(scene.blue.data.root_pos_w[0, 2]) - L_CAN / 2) < 0.012
    check("settle: all states finite, blocks parked FLAT mid-tote, can upright on the "
          f"floor outside, span readback BLOCKED (span={span() * 1000:.1f}mm < "
          f"gate {c.room_span_min * 1000:.0f}mm — every free gap is narrower than the can)",
          all_finite() and blk_flat(scene.blk_a) and blk_flat(scene.blk_b)
          and can_up and span() < c.room_span_min and bool(scene.blocks_stowed()[0]))
    check(f"settle: score ~0 and no success on a fresh reset (score={s:.3f})",
          s <= 0.02 and not ok)

    # =========================== 3. mass readback ===========================================
    m_tote = float(scene.tote.root_physx_view.get_masses().reshape(-1)[0])
    m_blk = float(scene.blk_a.root_physx_view.get_masses().reshape(-1)[0])
    m_can = float(scene.blue.root_physx_view.get_masses().reshape(-1)[0])
    check("mass readback: tote/block/can masses match the cfg "
          f"(tote={m_tote:.2f}/{c.tote_mass:.2f}, blk={m_blk:.2f}/{c.blk_mass:.2f}, "
          f"can={m_can:.2f}/{c.can_mass:.2f} kg)",
          abs(m_tote - c.tote_mass) < 0.15 and abs(m_blk - c.blk_mass) < 0.04
          and abs(m_can - c.can_mass) < 0.04)

    # =========================== 4-5. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(20)
        tp = (scene.tote.data.root_pos_w - scene.env_origins)[0]
        ex = _qapply(scene.tote.data.root_quat_w,
                     torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))[0]
        yaw = math.atan2(float(ex[1]), float(ex[0]))
        pair = 0.5 * (float(scene.tote_local(scene.blk_a.data.root_pos_w)[0, 1])
                      + float(scene.tote_local(scene.blk_b.data.root_pos_w)[0, 1]))
        bl = (scene.blue.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(tp[0]), float(tp[1]), yaw, pair, float(bl[0]), float(bl[1])))
    arr = np.array(reads)
    print("[smoke] randomization readback (tote_x, tote_y, tote_yaw, pair_off, "
          f"blue_x, blue_y):\n{arr.round(3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: tote pose varies across seeded resets (readback: "
          f"dx={spread[0]:.3f} dy={spread[1]:.3f} dyaw={spread[2]:.2f} rad)",
          spread[0] > 0.02 and spread[1] > 0.02 and spread[2] > 0.8)
    check("randomization: the pair parking offset and the can position vary "
          f"(readback: dpair={spread[3] * 1000:.1f}mm dblue={max(spread[4], spread[5]):.3f}m)",
          spread[3] > 0.006 and max(spread[4], spread[5]) > 0.05)

    # =========================== 6. null policy fails =======================================
    env.reset(seed=31)
    step(300)
    s, ok = judge()
    report("null-policy", s, ok)
    check(f"null policy: score ~0 and no success after 300 idle steps (score={s:.3f})",
          s <= 0.02 and not ok)

    # =========================== 7. carried, not placed =====================================
    env.reset(seed=41)
    settle_all(300)
    pos, q = tote_pose((0.0, 0.0, HOVER_Z))
    teleport(scene.blue, pos, q, settle_steps=2)  # judge while held aloft
    s, ok = judge()
    report("carried", s, ok)
    check("carried, not placed: the can POSED at the hover pose above the tote earns "
          f"no credit (score={s:.3f} <= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 8. seed strategy (drop it in) ==============================
    # The seed's move — carry the can over the container and RELEASE it. Here the
    # container is FULL: gravity lands the can standing ON TOP of the red blocks.
    settle_all(600)
    s, ok = judge()
    report("seed-strategy", s, ok)
    loc = scene.tote_local(scene.blue.data.root_pos_w)[0]
    check("seed strategy (drop into the container): the can settles PERCHED ON THE "
          f"BLOCKS (z-Zf={float(loc[2]) - Z_F:+.3f}m, floor window tops out at "
          f"{c.slot_z_win[1] * 1000:.0f}mm), span still blocked "
          f"(span={span() * 1000:.1f}mm) — enter credit only (score={s:.3f} <= 0.16), "
          "no success",
          bool(scene.in_tote(scene.blue)[0]) and float(loc[2]) - Z_F > 0.080
          and not bool(scene.can_in_slot()[0]) and span() < c.room_span_min
          and not bool(scene.l2_room[0]) and s <= 0.16 and not ok)

    # =========================== 9. side-gap drop ===========================================
    env.reset(seed=51)
    settle_all(300)
    pair = 0.5 * (float(scene.tote_local(scene.blk_a.data.root_pos_w)[0, 1])
                  + float(scene.tote_local(scene.blk_b.data.root_pos_w)[0, 1]))
    s_wide = -1.0 if pair >= 0.0 else 1.0  # the wider side gap is opposite the offset
    y_gap = slot_center(-s_wide)  # free-interval center on the s_wide side
    gap_w = 0.0
    edges = []
    for body in (scene.blk_a, scene.blk_b):
        yb = float(scene.tote_local(body.data.root_pos_w)[0, 1])
        h = float(scene.blk_y_halfwidth(body)[0])
        edges.append((yb - h, yb + h))
    if s_wide > 0:
        gap_w = W_HALF - max(e[1] for e in edges)
    else:
        gap_w = min(e[0] for e in edges) + W_HALF
    pos, q = tote_pose((0.0, y_gap, HOVER_Z))
    teleport(scene.blue, pos, q, settle_steps=0)
    settle_all(600)
    s, ok = judge()
    report("side-gap", s, ok)
    loc = scene.tote_local(scene.blue.data.root_pos_w)[0]
    ax_z = float(scene.can_axis_local()[0, 2])
    check("side-gap drop: the can released over the WIDEST free gap "
          f"(readback {gap_w * 1000:.1f}mm < can dia {2 * R_CAN * 1000:.0f}mm) cannot "
          f"stand on the floor (z-Zf={float(loc[2]) - Z_F:+.3f} ax_z={ax_z:+.2f}) — "
          f"no success (score={s:.3f})",
          gap_w < 2 * R_CAN and not bool(scene.can_in_slot()[0]) and not ok)

    # =========================== 10. the room readout is real ===============================
    env.reset(seed=61)
    settle_all(300)
    l2_before = bool(scene.l2_room[0])
    consolidate_blocks(+1.0, settle_steps=240)
    s, ok = judge()
    report("room-made", s, ok)
    check("room readout real: blocks teleported consolidated to one end (can far "
          f"away) -> span readback {span() * 1000:.1f}mm >= gate "
          f"{c.room_span_min * 1000:.0f}mm, L2 latch fires (was {l2_before}) — credit "
          f"0.25, NOT success (score={s:.3f})",
          not l2_before and span() >= c.room_span_min and bool(scene.l2_room[0])
          and bool(scene.blocks_stowed()[0]) and s <= 0.26 and not ok)

    # =========================== 11. lying in the slot ======================================
    y_slot = slot_center(+1.0)
    pos, q = tote_pose((0.0, y_slot, Z_F + R_CAN + 0.002), q_lie_x)
    teleport(scene.blue, pos, q, settle_steps=180)
    s, ok = judge()
    report("lying-in-slot", s, ok)
    loc = scene.tote_local(scene.blue.data.root_pos_w)[0]
    ax_z = float(scene.can_axis_local()[0, 2])
    check("lying in the slot: the can laid FLAT on the genuinely cleared floor "
          f"(z-Zf={float(loc[2]) - Z_F:+.3f} < window, ax_z={ax_z:+.2f}) is rejected "
          f"— no success (score={s:.3f})",
          not bool(scene.can_in_slot()[0]) and not ok)

    # =========================== 12. acceptance construct ===================================
    seated = False
    for _ in range(3):
        y_slot = slot_center(+1.0)
        pos, q = tote_pose((0.0, y_slot, Z_F + L_CAN / 2 + 0.006))
        teleport(scene.blue, pos, q, settle_steps=0)
        settle_all(480)
        if bool(scene.can_in_slot()[0]):
            seated = True
            break
    s, ok = judge_accept()
    report("accept-construct", s, ok)
    check("acceptance construct: the can RELEASED upright just above the cleared "
          "floor (inside the open top) — gravity alone seats its base on the tote "
          f"floor -> success TRUE (score={s:.3f})",
          seated and ok and s >= 0.99 and bool(scene.blocks_stowed()[0]))

    # =========================== 13-14. keep-in clause flips success ========================
    tote_p = scene.tote.data.root_pos_w[0]
    teleport_floor(scene.blk_a, float(tote_p[0]) + 0.35, float(tote_p[1]) + 0.20,
                   BLK_H / 2 + 0.002, settle_steps=120)
    s, ok = judge()
    report("block-ejected", s, ok)
    still_seated = bool(scene.can_in_slot()[0])
    check("keep-in clause: one red block teleported OUT of the tote while the can "
          f"stays seated (slot={still_seated}) flips success FALSE",
          still_seated and not ok and not bool(scene.blocks_stowed()[0]))
    yb = float(scene.tote_local(scene.blk_b.data.root_pos_w)[0, 1])
    pos, q = tote_pose((0.0, yb, Z_F + BLK_H + BLK_H / 2 + 0.002))
    teleport(scene.blk_a, pos, q, settle_steps=240)
    s, ok = judge_accept()
    report("block-returned", s, ok)
    check("block returned: the ejected block STACKED back on its partner (stacking "
          "is a legal stow) -> success returns TRUE (13's rejection was the keep-in "
          "clause and nothing else)", ok)

    # =========================== 15. settle gate ============================================
    # Kick the genuinely-seated can and judge IMMEDIATELY: statically correct
    # geometry, but not at rest -> not success. Destroyed after.
    pos = scene.blue.data.root_pos_w
    q = scene.blue.data.root_quat_w
    teleport(scene.blue, pos, q, vel=[0.0, 0.0, 0.35], ang=[0.0, 0.0, 6.0],
             settle_steps=0)
    step(1)  # refresh buffers only — judge while still moving
    lv = float(scene.blue.data.root_lin_vel_w[0].norm())
    av = float(scene.blue.data.root_ang_vel_w[0].norm())
    s, ok = judge()
    gate_ok = (lv > c.settle_lin or av > c.settle_ang) and not ok
    teleport_floor(scene.blue, float(tote_p[0]) - 0.45, float(tote_p[1]) + 0.30,
                   L_CAN / 2 + 0.003, settle_steps=60)  # destroy the construct
    report("settle-gate", s, ok)
    check("settle gate: the seated can kicked (lin={:.2f} m/s, ang={:.1f} rad/s) and "
          "judged immediately is NOT success (must be at rest); state destroyed"
          .format(lv, av), gate_ok)

    # =========================== 16. wrong object ===========================================
    env.reset(seed=81)
    settle_all(300)
    consolidate_blocks(-1.0, settle_steps=0)  # blk_b to the -y end...
    # ...then stand blk_a ON END inside the slot windows instead (same z as a
    # floor-standing can) while the blue can never enters the tote.
    pos, q = tote_pose((0.0, 0.5 * (W_HALF - BLK_W), Z_F + BLK_L / 2 + 0.002), q_on_end)
    teleport(scene.blk_a, pos, q, settle_steps=240)
    s, ok = judge()
    report("wrong-object", s, ok)
    la = scene.tote_local(scene.blk_a.data.root_pos_w)[0]
    check("wrong object: a RED BLOCK stood on end in the slot windows "
          f"(blk_a z-Zf={float(la[2]) - Z_F:+.3f} — same band as a standing can) with "
          f"the blue can far outside is rejected (score={s:.3f} <= 0.26)",
          not ok and s <= 0.26 and not bool(scene.can_in_slot()[0]))

    # =========================== 17-18. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point except the "
          "two constructed acceptance probes", not ever_bad_success[0])
    check("final: all task-object states finite (no NaN)", all_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.packed_tote")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(okc for _nm, okc in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, okc in checks:
            if not okc:
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)

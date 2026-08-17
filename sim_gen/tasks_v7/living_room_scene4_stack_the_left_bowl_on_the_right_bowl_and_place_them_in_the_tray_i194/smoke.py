"""Smoke / rubric-REJECTION battery for PagodaRelayScene (sim_gen task
`living_room_scene4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray_i194`)
— NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — the legal 7-move Hanoi relay — is the acceptance
evidence). Every teleport here is instrumentation that CONSTRUCTS a wrong (or partial)
outcome and asserts the rubric REJECTS it, plus two physics probes that prove the
captivity is real (the top tier lifts OFF the post under a modest force — the actuator
is proven live — while the same force machinery cannot extract the pinned base tier
sideways: it rattles inside the threading slack and stays threaded). Rejection probes
must never read success; two constructs DO build the genuine end state on purpose (the
acceptance drop and the latch-flip restore) and a final audit asserts exactly that.

  1-2.  settle/no-NaN     — reset settles finite; pagoda assembled on the START post at
                            levels 0/1/2; score ~0, no success;
  3-5.  randomization     — READBACK over 8 seeded resets: start-pedestal xy + yaw vary;
                            tray xy + yaw vary; tier resting heading varies;
  6.    null policy       — 300 idle steps -> score ~0, no success, no violation;
  7.    captivity is      — a ~4 N upward force lifts the TOP tier clear off the post
        top-only + live     (> 3 cm — past the post-proud height, so the hole really
                            cleared the tip); force cleared, the tier re-dropped and it
                            re-seats; no violation (single legal carry);
  8.    lateral captivity — a sustained 8 N sideways shove on the pinned BASE tier
                            rattles it across the threading slack (>= 3 mm peak-to-peak
                            — the shove verifiably acts) but CANNOT extract it: the
                            whole pagoda stays threaded on the start post, no violation;
  9.    seed strategy     — the seed's move: the WHOLE stack carried to the tray in one
                            go. It lands geometrically PERFECT (all three seated at
                            their exact levels on the tray post) yet the one-tier-in-
                            flight latch tripped mid-carry: violated, score 0, never
                            success;
  10.   parking is legal  — the small tier parked on the open table: ONE tier off-post
                            is allowed -> not violated (but it counts for nothing);
  11.   parked tier       — lifting the MID tier while the small one is parked = two
        freezes the relay   tiers off-post -> carry latch: violated, score 0;
  12.   wrong order       — mid tier seated in the tray, then the LARGE tier dropped on
                            top of it (bigger above smaller, threaded on the same post)
                            -> order latch: violated, score 0, no success;
  13.   near miss +       — 6 of the 7 legal moves replayed (small tier back on A, the
        partial credit      large+mid pagoda standing in the tray): no success, and the
                            latched score reads exactly 0.15+0.25+0.30 = 0.70;
  14.   beside-the-post   — the small tier rested on the BUFFER pad 6 cm off its post:
                            in the station but NOT threaded -> not seated, score stays
                            0.70, no success, no violation;
  15.   acceptance        — the small tier picked back up and dropped onto the tray
                            post: gravity threads it to level 2 -> success TRUE, 1.0;
  16-17. latch flip       — the identical accepted geometry with the violated latch
                            forced on -> success FALSE, score 0; the exact accepted
                            state restored -> success returns TRUE (16's rejection was
                            the rule latch and nothing else);
  18.   settle gate       — the seated small tier kicked and judged immediately: NOT
                            success (must be at rest);
  19.   rejection audit   — success() was never True at any judged point EXCEPT the
                            acceptance construct and the flip-back (15, 17);
  20.   final no-NaN      — all task-object states finite at the end.

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
    from .scene import PAD_H, POST_TOP, TIER_H, _qapply
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import PAD_H, POST_TOP, TIER_H, _qapply
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

HOVER = 0.012  # release height above the post tip (matches the solve)
DROP_OFF = 0.003  # lateral release offset (matches the solve)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.pagoda_relay")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)
    q_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.10, -1.10, 0.85)) + o),
                                tuple(np.array((0.00, 0.05, 0.08)) + o),
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

    def report(tag: str, s: float, ok: bool) -> None:
        seated, lz = scene.seat_state()
        stacks = []
        for si, nm in enumerate("ABT"):
            col = [("LMS"[t], float(lz[0, t, si]))
                   for t in range(3) if bool(seated[0, t, si])]
            col.sort(key=lambda p: p[1])
            stacks.append(nm + ":" + ("".join(p[0] for p in col) or "-"))
        print(f"[smoke] {tag:16s} | {' '.join(stacks)} "
              f"violated={bool(scene._violated[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

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

    def st_pose(s: int, local) -> torch.Tensor:
        stn = scene.stations[s]
        return stn.data.root_pos_w + _qapply(
            stn.data.root_quat_w, torch.tensor(local, device=device).expand(n, 3))

    def st_quat(s: int) -> torch.Tensor:
        return scene.stations[s].data.root_quat_w

    def drop(t: int, s: int, lvl: int | None, tries: int = 4) -> bool:
        """One hover-drop (the solve's move mechanics): release tier t with zero
        velocity just above station s's post tip; gravity threads it down. Verify
        seated (at the expected level, if given) and settled by READBACK."""
        loc = (DROP_OFF, -DROP_OFF, POST_TOP + HOVER)
        for _attempt in range(tries):
            teleport(scene.tiers[t], st_pose(s, loc), st_quat(s), settle_steps=0)
            for _ in range(80):
                step(3)
                seated, lz = scene.seat_state()
                if bool(seated[0, t, s]) and bool(scene.settled(scene.tiers[t])[0]) \
                        and (lvl is None or abs(float(lz[0, t, s]) - (PAD_H + lvl * TIER_H))
                             <= float(scene.cfg.lvl_tol)):
                    return True
        return False

    def settle_all(max_steps: int = 480) -> None:
        for _ in range(max_steps // 30):
            step(30)
            if all(bool(scene.settled(t)[0]) for t in scene.tiers):
                break

    def all_finite() -> bool:
        return all(bool(torch.isfinite(b.data.root_state_w).all())
                   for b in (*scene.stations, *scene.tiers))

    def yaw_of(q) -> float:
        y = 2.0 * math.atan2(float(q[0, 3]), float(q[0, 0]))
        return (y + math.pi) % (2.0 * math.pi) - math.pi

    def stack_at(s: int) -> tuple[bool, ...]:
        seated, _ = scene.seat_state()
        return tuple(bool(seated[0, t, s]) for t in range(3))

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    settle_all(480)
    s, ok = judge()
    report("reset", s, ok)
    seated, lz = scene.seat_state()
    lv_ok = all(bool(seated[0, t, 0])
                and abs(float(lz[0, t, 0]) - (PAD_H + t * TIER_H))
                <= float(scene.cfg.lvl_tol) + 0.004 for t in range(3))
    check("settle: all states finite, pagoda assembled on the START post at levels "
          "0/1/2, nothing violated", all_finite() and lv_ok and not bool(scene._violated[0]))
    check(f"settle: score ~0 and no success on a fresh reset (score={s:.3f})",
          s <= 0.02 and not ok)

    # =========================== 3-5. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(20)
        pa = (scene.stations[0].data.root_pos_w - scene.env_origins)[0]
        tr = (scene.stations[2].data.root_pos_w - scene.env_origins)[0]
        reads.append((float(pa[0]), float(pa[1]), yaw_of(scene.stations[0].data.root_quat_w),
                      float(tr[0]), float(tr[1]), yaw_of(scene.stations[2].data.root_quat_w),
                      yaw_of(scene.tiers[2].data.root_quat_w)))
    arr = np.array(reads)
    print("[smoke] randomization readback (pedA_x, pedA_y, pedA_yaw, tray_x, tray_y, "
          f"tray_yaw, tierS_yaw):\n{arr.round(3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: start-pedestal pose varies across seeded resets (readback: "
          f"dx={spread[0]:.3f} dy={spread[1]:.3f} dyaw={spread[2]:.2f} rad)",
          spread[0] > 0.020 and spread[1] > 0.020 and spread[2] > 0.15)
    check("randomization: tray pose varies across seeded resets (readback: "
          f"dx={spread[3]:.3f} dy={spread[4]:.3f} dyaw={spread[5]:.2f} rad)",
          spread[3] > 0.020 and spread[4] > 0.020 and spread[5] > 0.15)
    check("randomization: tier resting heading varies (readback: "
          f"dyaw={spread[6]:.2f} rad)", spread[6] > 1.0)

    # =========================== 6. null policy fails =======================================
    env.reset(seed=31)
    step(300)
    s, ok = judge()
    report("null-policy", s, ok)
    check(f"null policy: score ~0, no success, no violation after 300 idle steps "
          f"(score={s:.3f})", s <= 0.02 and not ok and not bool(scene._violated[0]))

    # =========================== 7. captivity is top-only (and the actuator is live) ========
    env.reset(seed=35)
    settle_all(360)
    tier_s = scene.tiers[2]
    z0 = float(tier_s.data.root_pos_w[0, 2])
    f_up = torch.zeros(n, 1, 3, device=device)
    f_up[:, 0, 2] = 4.0  # ~2.2x the small tier's weight
    rise = 0.0
    for _ in range(120):
        tier_s.set_external_force_and_torque(f_up, zero_w, env_ids=all_ids)
        step(1)
        rise = max(rise, float(tier_s.data.root_pos_w[0, 2]) - z0)
        if rise > 0.040:
            break
    tier_s.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
    re_ok = drop(2, 0, lvl=2)  # carry the in-flight tier back and re-seat it
    s, ok = judge()
    report("lift-probe", s, ok)
    check("captivity is top-only and the force machinery is LIVE: ~4 N lifts the top "
          f"tier clear off the post (rise={rise * 100:.1f} cm > post-proud height), and "
          "re-dropped it re-seats at level 2 with no violation (a single legal carry)",
          rise > 0.030 and re_ok and not bool(scene._violated[0]) and not ok)

    # =========================== 8. lateral captivity (the post pins the base) ==============
    tier_l = scene.tiers[0]

    def push(sign: float, steps: int = 150) -> torch.Tensor:
        f = torch.zeros(n, 1, 3, device=device)
        f[:, 0, 0] = sign * 8.0  # body-frame horizontal — direction is irrelevant,
        for _ in range(steps):  # ANY sideways shove meets the post inside the slack
            tier_l.set_external_force_and_torque(f, zero_w, env_ids=all_ids)
            step(1)
        tier_l.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
        return tier_l.data.root_pos_w[0, :2].clone()

    p1 = push(+1.0)
    p2 = push(-1.0)
    travel = float((p1 - p2).norm())
    step(120)
    s, ok = judge()
    report("shove-probe", s, ok)
    check("lateral captivity: a sustained 8 N shove on the loaded BASE tier verifiably "
          f"acts (peak-to-peak travel {travel * 1000:.1f} mm across the threading "
          "slack) but CANNOT extract it — the whole pagoda stays threaded on the start "
          "post, no violation, score ~0",
          travel >= 0.003 and all(stack_at(0)) and not bool(scene._violated[0])
          and s <= 0.02 and not ok)

    # =========================== 9. the seed's strategy =====================================
    # Stack-then-deliver in ONE carry: the whole assembled pagoda hover-dropped over the
    # tray post as a unit. Gravity threads all three PERFECTLY — geometry alone equals
    # the goal state — but three tiers were in flight at once, so the carry latch
    # tripped mid-air: the episode is void. Sequencing, not placement, is the task.
    env.reset(seed=41)
    settle_all(360)
    for k in range(3):
        loc = (DROP_OFF, -DROP_OFF, POST_TOP + HOVER + k * (TIER_H + 0.0015))
        teleport(scene.tiers[k], st_pose(2, loc), st_quat(2), settle_steps=0)
    step(300)
    seated, lz = scene.seat_state()
    geom = all(bool(seated[0, t, 2])
               and abs(float(lz[0, t, 2]) - (PAD_H + t * TIER_H))
               <= float(scene.cfg.lvl_tol) + 0.004 for t in range(3))
    s, ok = judge()
    report("seed-strategy", s, ok)
    check("seed strategy (carry the whole stack to the tray in one go): the pagoda "
          "lands geometrically PERFECT on the tray post, yet the one-tier-in-flight "
          f"latch tripped mid-carry — violated, score 0 ({s:.3f}), never success",
          geom and bool(scene._violated[0]) and s <= 1e-6 and not ok)

    # =========================== 10-11. parking freezes the relay ===========================
    env.reset(seed=51)
    settle_all(360)
    table_spot = scene.env_origins + torch.tensor([0.45, -0.45, 0.030], device=device)
    teleport(scene.tiers[2], table_spot, q_id, settle_steps=180)
    s, ok = judge()
    report("table-park", s, ok)
    check("parking is legal: the small tier set down on the open table is ONE tier "
          f"off-post — not violated (and worth nothing: score={s:.3f}, no success)",
          not bool(scene.seat_state()[0][0, 2, :].any())
          and not bool(scene._violated[0]) and s <= 0.02 and not ok)
    teleport(scene.tiers[1], st_pose(1, (DROP_OFF, -DROP_OFF, POST_TOP + HOVER)),
             st_quat(1), settle_steps=180)
    s, ok = judge()
    report("park-then-carry", s, ok)
    check("a parked tier freezes the relay: lifting the MID tier while the small one "
          f"sits on the table = two tiers off-post — violated, score 0 ({s:.3f})",
          bool(scene._violated[0]) and s <= 1e-6 and not ok)

    # =========================== 12. wrong order ============================================
    env.reset(seed=61)
    settle_all(360)
    ok_m = drop(1, 2, lvl=0)  # mid tier legally into the tray (small sinks, stays seated)
    ok_l = drop(0, 2, lvl=None)  # LARGE dropped on top of it — bigger above smaller
    seated, lz = scene.seat_state()
    inverted = ok_m and ok_l and bool(seated[0, 0, 2]) and bool(seated[0, 1, 2]) \
        and float(lz[0, 0, 2]) > float(lz[0, 1, 2]) + TIER_H / 2
    s, ok = judge()
    report("wrong-order", s, ok)
    check("wrong order: the LARGE tier threaded ABOVE the mid tier on the tray post "
          f"trips the order latch — violated, score 0 ({s:.3f}), no success",
          inverted and bool(scene._violated[0]) and s <= 1e-6 and not ok)

    # =========================== 13. near miss + partial credit =============================
    env.reset(seed=71)
    settle_all(360)
    six = ((2, 2, 0), (1, 1, 0), (2, 1, 1), (0, 2, 0), (2, 0, 0), (1, 2, 1))
    legal = True
    for t, s_, lvl in six:
        legal = legal and drop(t, s_, lvl)
    settle_all(240)
    s, ok = judge()
    report("near-miss", s, ok)
    check("near miss + partial credit: 6 of the 7 legal moves replayed (large+mid "
          "rebuilt in the tray, small back on the start post) — no success, no "
          f"violation, and the latched score reads exactly 0.15+0.25+0.30 = 0.70 "
          f"(score={s:.3f})",
          legal and not ok and not bool(scene._violated[0]) and abs(s - 0.70) < 0.005)

    # =========================== 14. beside the post is not placed ==========================
    # The small tier rested on the BUFFER pad 61 mm off its post: inside the station
    # but NOT threaded (the flange clears the post by 5 mm; the seated xy window —
    # asserted to separate threaded from beside — rejects it).
    teleport(scene.tiers[2], st_pose(1, (0.061, 0.0, PAD_H + 0.012)), st_quat(1),
             settle_steps=180)
    s, ok = judge()
    report("beside-post", s, ok)
    check("beside-the-post: the small tier resting on the buffer pad 6 cm off the post "
          f"is NOT seated anywhere — score stays 0.70 ({s:.3f}), no success, no "
          "violation (still just one tier off-post)",
          not bool(scene.seat_state()[0][0, 2, :].any()) and not ok
          and not bool(scene._violated[0]) and abs(s - 0.70) < 0.005)

    # =========================== 15. acceptance construct ===================================
    acc = drop(2, 2, lvl=2)  # pick the parked tier back up — the 7th legal move
    settle_all(240)
    s, ok = judge_accept()
    report("accept-construct", s, ok)
    check("acceptance construct: the small tier dropped onto the tray post threads to "
          f"level 2 — success TRUE, score 1.0 (score={s:.3f})", acc and ok and s >= 0.99)
    snap_acc = scene.get_state(all_ids)  # settled accepted state (near-zero velocities)

    # =========================== 16-17. the rule latch alone flips success ==================
    snap_bad = {k: v.clone() for k, v in snap_acc.items()}
    snap_bad["_latch"][:, 0] = 1.0  # force the violated latch on, geometry untouched
    scene.set_state(snap_bad, all_ids)
    step(3)
    s, ok = judge()
    report("latch-flip", s, ok)
    check("latch flip: the IDENTICAL accepted geometry with the violated latch forced "
          f"on is judged score 0 ({s:.3f}), success FALSE — the rules, not the pose, "
          "gate the credit", not ok and s <= 1e-6)
    scene.set_state(snap_acc, all_ids)
    step(5)
    s, ok = judge_accept()
    report("latch-restore", s, ok)
    check("exact accepted state restored -> success returns TRUE (16's rejection was "
          "the rule latch and nothing else)", ok and s >= 0.99)

    # =========================== 18. settle gate ============================================
    teleport(scene.tiers[2], scene.tiers[2].data.root_pos_w,
             scene.tiers[2].data.root_quat_w, vel=[0.0, 0.0, 0.5],
             ang=[0.0, 0.0, 6.0], settle_steps=0)
    step(1)  # refresh buffers only — judge while still moving
    lv_kick = float(scene.tiers[2].data.root_lin_vel_w[0].norm())
    av_kick = float(scene.tiers[2].data.root_ang_vel_w[0].norm())
    s, ok = judge()
    report("settle-gate", s, ok)
    check(f"settle gate: the seated small tier kicked (lin={lv_kick:.2f} m/s, "
          f"ang={av_kick:.1f} rad/s) and judged immediately is NOT success (must be "
          "at rest)",
          (lv_kick > float(scene.cfg.settle_lin) or av_kick > float(scene.cfg.settle_ang))
          and not ok)

    # =========================== 19-20. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point except the "
          "acceptance construct and the flip-back", not ever_bad_success[0])
    check("final: all task-object states finite (no NaN)", all_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.pagoda_relay")
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

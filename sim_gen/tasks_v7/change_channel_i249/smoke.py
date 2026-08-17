"""Smoke / rubric-REJECTION battery for ChannelConsoleScene (sim_gen task
`change_channel_i249`) — NullRobot, teleported probe states + force probes, RECORDED.

This is NOT a solution (solve.py — dynamic pin extraction, then a closed-loop slide
servo into the color-matched band — is the acceptance evidence that the rubric
ACCEPTS a correct outcome). Every teleport here is instrumentation that CONSTRUCTS a
wrong (or partial) outcome as a judged state and asserts the rubric REJECTS it — plus
applied-force probes that prove the interlock and the captivity are physically real.
No probe in this battery ever reaches success() at a judged point; a final audit
asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: console at the workspace,
                            pin seated and BLOCKING, slider in-track on its side,
                            matching indicator cube on the pedestal, spares in the
                            depot; score ~0 at rest, no success;
  3-4. randomization      — READBACK over 8 seeded resets: console yaw/xy jitter is
                            physically posed and varies; BOTH start sides occur,
                            >= 2 distinct target channels, the slider start varies
                            inside its window, and the pedestal cube always matches
                            the sampled target;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed strategy       — rlbench/change_channel's verb is grasp-the-remote +
                            press-a-button. No hand-held device or button exists
                            here by constructed premise (the console is one fixed
                            kinematic body); the closest transplant — parking the
                            slider at the target COLOR but on the open deck, outside
                            the track — is rejected by the track windows;
  7.  wrong channel       — slider settled in-track at a NON-target band: in_track
                            and settled both hold, in_band does not -> no success;
  8.  band near-miss      — slider settled in-track 8 mm outside band_tol -> no
                            success (the tolerance is real);
  9.  interlock probe     — with the pin SEATED, a regulated quasi-static push
                            (<= 3 N) drives the slider toward the target: it MOVES
                            >= 25 mm (non-vacuous) then STALLS against the pin,
                            never crossing the column; the pin stays seated; the
                            execution order is physically enforced;
  10. captivity probe     — a regulated upward pull (<= 4 N ~ 2.7x slider weight)
                            lifts the slider only a few mm before the lip strips
                            stop it: it cannot leave the track, and re-settles
                            in-track after release;
  11. on-lips imposter    — slider balanced ON TOP of the lip strips at the target
                            x reads ~38 mm high -> rejected by the z window;
  12. settle gate         — slider INSIDE the target band but still moving is NOT
                            success (velocity gates are real); removed before it
                            can settle;
  13. part-way + cap      — all three latches constructed (pin depoted, slider
                            carried across and held over the band on the lips):
                            score == 0.60 cap (float32 + eps), NOT success — full
                            credit short of success is impossible;
  14. latched credit      — teleporting the slider off to the floor leaves the
                            latched score unchanged, still no success;
  15. wrong object        — the extracted PIN laid across the track at the target
                            band does not satisfy the rubric: the SELECTOR must be
                            there;
  16. rejection audit     — success() was never True at ANY judged point;
  17. final no-NaN        — all task-object states finite at the end;
  18. camera              — >= 20 rgb frames captured -> frames.npz.

Run (forge): python -u -m simgen_tasks.change_channel_i249.smoke --headless
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)",
                                             flush=True), os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.channel_console")().build(num_envs=args.num_envs,
                                                     device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_rows = torch.zeros(n, 1, 3, device=device)
    G = scene_mod  # geometry constants module

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.10, -1.10, 0.95)) + o),
                                tuple(np.array((0.0, 0.0, 0.22)) + o),
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
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return s, ok

    def report(tag: str) -> None:
        s, ok = judge()
        b = scene.block_canon()[0]
        _pr, pb = scene.pin_canon()
        print(f"[smoke] {tag:16s} | side={float(scene.side[0]):+.0f}"
              f" tgt={c.chan_names[int(scene.target_idx[0])]}"
              f" block=({float(b[0]):+.3f},{float(b[1]):+.3f},{float(b[2]):.3f})"
              f" pin_bot_z={float(pb[0, 2]):.3f}"
              f" latches=({float(scene.pin_clear[0]):.0f},"
              f"{float(scene.crossed[0]):.0f},{float(scene.appr_max[0]):.2f})"
              f" in_band={bool(scene.in_band()[0])}"
              f" in_track={bool(scene.in_track()[0])}"
              f" score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, canon_xyz, quat_local=None, vel_canon=None,
              settle_steps: int = 45) -> None:
        """Kinematic probe placement at a CANONICAL-frame point (instrumentation,
        not a solution) + REAL physics steps before judging (the zero-step trap).
        The pose rides the live console frame, so probes track yaw/jitter."""
        canon = torch.tensor(canon_xyz, device=device).view(1, 3).expand(n, 3)
        cq = scene.console.data.root_quat_w
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.canon_to_world(canon.clone())
        if quat_local is None:
            st[:, 3:7] = cq
        else:
            ql = torch.tensor(quat_local, device=device).view(1, 4).expand(n, 4)
            st[:, 3:7] = quat_mul(cq, ql)
        if vel_canon is not None:
            st[:, 7:10] = quat_apply(
                cq, torch.tensor(vel_canon, device=device).view(1, 3).expand(n, 3))
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def place_rel(body, xyz, settle_steps: int = 45) -> None:
        """Env-relative world placement (parking on the open floor / depot)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor(xyz, device=device)
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def wrench(body, f_world: torch.Tensor) -> None:
        """World force encoded in the body's CURRENT link frame (house convention),
        re-computed by the caller every step."""
        body.set_external_force_and_torque(
            quat_apply_inverse(body.data.root_link_quat_w, f_world).unsqueeze(1),
            zero_rows, env_ids=all_ids)

    def clear(body) -> None:
        body.set_external_force_and_torque(zero_rows, zero_rows, env_ids=all_ids)

    def bodies() -> tuple:
        return (scene.console, scene.block, scene.pin, *scene.flags)

    def layout_sane(tag: str) -> bool:
        """Reset honesty: console at the workspace (within jitter), pin seated and
        blocking, slider in-track on its side, matching cube on the pedestal,
        spares in the depot, target on the opposite side, no success."""
        cp = (scene.console.data.root_pos_w - scene.env_origins)[0]
        pr, _pb = scene.pin_canon()
        b = scene.block_canon()[0]
        side = float(scene.side[0])
        tidx = int(scene.target_idx[0])
        ped = scene.canon_to_world(torch.tensor(
            [[G.PED_XY[0], G.PED_XY[1], G.FLAG_Z]], device=device).expand(n, 3))
        d_ped = float((scene.flags[tidx].data.root_pos_w - ped)[0].norm())
        spares_ok = True
        for i, f in enumerate(scene.flags):
            if i == tidx:
                continue
            fp = (f.data.root_pos_w - scene.env_origins)[0]
            spares_ok = spares_ok and \
                abs(float(fp[0]) - G.FLAG_DEPOT[0]) < 0.05 and \
                abs(float(fp[1]) - (G.FLAG_DEPOT[1] + 0.12 * i)) < 0.05
        ok = (abs(float(cp[0])) <= c.pos_jitter + 0.01
              and abs(float(cp[1])) <= c.pos_jitter + 0.01
              and abs(float(pr[0, 0])) < 0.006 and abs(float(pr[0, 1])) < 0.006
              and abs(float(pr[0, 2]) - G.PIN_SEAT_Z) < 0.006
              and bool(scene.pin_blocking()[0])
              and abs(float(b[1])) <= 0.012
              and abs(float(b[2]) - G.BLOCK_REST_Z) < 0.006
              and c.start_abs_lo - 0.02 <= side * float(b[0]) <= c.start_abs_hi + 0.02
              and float(scene.x_target[0]) * side < 0
              and d_ped < 0.02 and spares_ok
              and not bool(scene.success()[0]))
        if not ok:
            print(f"[smoke] layout sanity VIOLATION at {tag}: cp={cp} pin={pr[0]} "
                  f"block={b} d_ped={d_ped:.3f} spares_ok={spares_ok}", flush=True)
        return ok

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies())
    check("settle: all states finite; console at the workspace, pin seated and "
          "blocking, slider in-track on its side, matching cube on the pedestal, "
          "spares in the depot", fin and layout_sane("reset"))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    sane = True
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(30)
        sane = sane and layout_sane(f"seed {sd}")
        q = scene.console.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        cp = (scene.console.data.root_pos_w - scene.env_origins)[0]
        b = scene.block_canon()[0]
        reads.append((float(scene.side[0]), int(scene.target_idx[0]), yaw,
                      float(cp[0]), float(cp[1]), float(b[0])))
    arr = np.array(reads)
    print("[smoke] randomization readback (side, tgt_idx, yaw, cx, cy, block_x):\n"
          f"{np.round(arr, 4)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    sides = {float(x) for x in arr[:, 0]}
    tgts = {int(x) for x in arr[:, 1]}
    check("randomization: console yaw/xy jitter is physically posed and varies "
          f"(readback: yaw spread {math.degrees(spread[2]):.1f} deg, xy spread "
          f"({spread[3]:.3f},{spread[4]:.3f})); layout sane on all 8 seeds",
          spread[2] > 0.03 and (spread[3] > 0.008 or spread[4] > 0.008) and sane)
    check("randomization: BOTH start sides occur, >= 2 distinct target channels, "
          f"slider start varies (readback: sides={sorted(sides)}, "
          f"targets={sorted(tgts)}, block_x spread {spread[5]:.3f}); the pedestal "
          "cube matched the sampled target on every seed (layout sanity)",
          len(sides) == 2 and len(tgts) >= 2 and spread[5] > 0.01)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 6. seed strategy rejected ==================================
    # rlbench/change_channel's verb is grasp-the-remote + press-a-channel-button. No
    # hand-held device or pressable button exists here by constructed premise: the
    # console is ONE fixed kinematic body (nothing to grasp, nothing that depresses).
    # The closest transplant — putting the manipuland AT the target color but not in
    # the mechanism (parked on the open deck in front of the track) — is rejected.
    env.reset(seed=41)
    step(30)
    tx = float(scene.x_target[0])
    place(scene.block, (tx, -0.085, G.DECK_TOP + G.BLOCK_SZ[2] / 2 + 0.001),
          settle_steps=60)
    report("deck-transplant")
    b = scene.block_canon()[0]
    s, ok = judge()
    check("seed strategy: grasp+press is N/A by premise (single kinematic console, "
          "no hand-held device); the transplant — slider parked at the target "
          f"COLOR on the open deck (canon y={float(b[1]):+.3f}, "
          f"z={float(b[2]):.3f}) — matches in_band in x but is rejected by the "
          "track windows, no success",
          bool(scene.in_band()[0]) and not bool(scene.in_track()[0]) and not ok)

    # =========================== 7. wrong channel ===========================================
    env.reset(seed=51)
    step(30)
    side = float(scene.side[0])
    wrong_x = side * abs(G.CHAN_XS[2])  # inner band on the START side — never target
    place(scene.block, (wrong_x, 0.0, G.BLOCK_REST_Z + 0.001), settle_steps=60)
    report("wrong-channel")
    b = scene.block_canon()[0]
    s, ok = judge()
    check("wrong channel: slider settled IN-track at a non-target band "
          f"(canon x={float(b[0]):+.3f}, target {float(scene.x_target[0]):+.3f}) — "
          "in_track and settled hold, in_band does not -> no success",
          bool(scene.in_track()[0]) and bool(scene.settled()[0])
          and not bool(scene.in_band()[0]) and not ok)

    # =========================== 8. band near-miss ==========================================
    env.reset(seed=61)
    step(30)
    tx = float(scene.x_target[0])
    tsign = 1.0 if tx > 0 else -1.0
    off = c.band_tol + 0.008
    # stay clear of the end stop (outer band) and of the pin column (inner band)
    near_x = tx - tsign * off if abs(tx) > 0.10 else tx + tsign * off
    place(scene.block, (near_x, 0.0, G.BLOCK_REST_Z + 0.001), settle_steps=60)
    report("near-miss")
    b = scene.block_canon()[0]
    miss = abs(float(b[0]) - tx)
    s, ok = judge()
    check("band near-miss: slider settled in-track "
          f"{miss * 1000:.0f} mm from the target center (> band_tol "
          f"{c.band_tol * 1000:.0f} mm) -> no success (the tolerance is real)",
          bool(scene.in_track()[0]) and bool(scene.settled()[0])
          and miss > c.band_tol + 0.002 and not bool(scene.in_band()[0]) and not ok)

    # =========================== 9. interlock force probe ===================================
    # With the pin SEATED, a regulated quasi-static push (velocity servo, <= 3 N —
    # the static-vs-dynamic-probe trap) drives the slider toward the target side.
    env.reset(seed=71)
    step(30)
    tsign = 1.0 if float(scene.x_target[0]) > 0 else -1.0
    b0 = float(scene.block_canon()[0, 0])
    bx_best = b0 * tsign  # progress toward the target (canonical x * tsign)
    # stick-slip breaker (as in solve P2): feed-forward builds while stuck, decays
    # to a small bias while sliding; servo gain stays under the wrench-delay bound;
    # TOTAL force hard-capped at 3 N (quasi-static — the static-vs-dynamic trap)
    kv, v_des, cap = 12.0, 0.05, 3.0
    ff = 0.0
    for _i in range(700):
        v_canon = quat_apply_inverse(scene.console.data.root_quat_w,
                                     scene.block.data.root_lin_vel_w)
        moving = float(v_canon[0, 0]) * tsign > 0.02
        ff = max(ff - 0.10, 1.0) if moving else min(ff + 0.05, cap)
        f_canon = torch.zeros(n, 3, device=device)
        f_canon[:, 0] = ff * tsign + kv * (v_des * tsign - v_canon[:, 0])
        f_world = quat_apply(scene.console.data.root_quat_w, f_canon)
        f_norm = f_world.norm(dim=-1, keepdim=True)
        f_world = f_world * (f_norm.clamp(max=cap) / f_norm.clamp_min(1e-9))
        wrench(scene.block, f_world)
        step(1)
        bx_best = max(bx_best, float(scene.block_canon()[0, 0]) * tsign)
    clear(scene.block)
    step(45)
    report("interlock-stall")
    b = scene.block_canon()[0]
    pr, _pb = scene.pin_canon()
    moved = (float(b[0]) - b0) * tsign
    s, ok = judge()
    check("interlock: the regulated quasi-static push (<= 3 N) moves the seated-pin "
          f"slider {moved * 1000:.0f} mm (>= 25, non-vacuous) then STALLS it against "
          f"the pin — progress max {bx_best * 1000:.0f} mm stays on the start side "
          "(< -15 mm of the column), the crossed latch never fires, the pin stays "
          f"seated (root z {float(pr[0, 2]):.3f}) and blocking, no success",
          moved >= 0.025 and bx_best < -0.015
          and float(scene.crossed[0]) == 0.0
          and bool(scene.pin_blocking()[0])
          and abs(float(pr[0, 2]) - G.PIN_SEAT_Z) < 0.008 and not ok)

    # =========================== 10. captivity force probe ==================================
    env.reset(seed=81)
    step(30)
    z0 = float(scene.block_canon()[0, 2])
    zmax = z0
    m_blk = float(scene.block.root_physx_view.get_masses()[0])
    for _i in range(300):
        v = scene.block.data.root_lin_vel_w
        f = -2.0 * v
        f[:, 2] = m_blk * 9.81 + 3.0 * (0.06 - v[:, 2])
        f_norm = f.norm(dim=-1, keepdim=True)
        f = f * (f_norm.clamp(max=4.0) / f_norm.clamp_min(1e-9))
        wrench(scene.block, f)
        step(1)
        zmax = max(zmax, float(scene.block_canon()[0, 2]))
    clear(scene.block)
    step(60)
    report("captive-lift")
    s, ok = judge()
    rise = zmax - z0
    check("captivity: the regulated upward pull (<= 4 N ~ 2.7x slider weight) "
          f"lifts the slider only {rise * 1000:.1f} mm (in [2, 14] — it moved, "
          "non-vacuous, but the lip strips stop it far short of escaping) and it "
          "re-settles in-track after release, no success",
          0.002 <= rise <= 0.014 and bool(scene.in_track()[0]) and not ok)

    # =========================== 11. on-lips imposter =======================================
    env.reset(seed=91)
    step(30)
    tx = float(scene.x_target[0])
    place(scene.block, (tx, 0.0, G.LIP_TOP + G.BLOCK_SZ[2] / 2 + 0.001),
          settle_steps=45)
    report("on-lips")
    b = scene.block_canon()[0]
    s, ok = judge()
    lips_ok = (float(b[2]) > G.BLOCK_REST_Z + c.z_tol + 0.010
               and not bool(scene.in_track()[0]) and not ok)
    place_rel(scene.block, (-1.2, 1.2, G.BLOCK_SZ[2] / 2 + 0.001), settle_steps=20)
    check("on-lips imposter: a block balanced ON TOP of the lip strips at the "
          f"target x reads canon z={float(b[2]) * 1000:.0f} mm (rest "
          f"{G.BLOCK_REST_Z * 1000:.0f} mm) -> rejected by the z window", lips_ok)

    # =========================== 12. settle gate ============================================
    env.reset(seed=101)
    step(30)
    tx = float(scene.x_target[0])
    tsign = 1.0 if tx > 0 else -1.0
    place(scene.block, (tx, 0.0, G.BLOCK_REST_Z + 0.001),
          vel_canon=(0.30 * tsign, 0.0, 0.0), settle_steps=2)
    v_now = float(scene.block.data.root_lin_vel_w[0].norm())
    report("settle-gate")
    _s, ok = judge()
    gate_ok = (bool(scene.in_band()[0]) and bool(scene.in_track()[0])
               and v_now > c.settle_lin and not ok)
    # remove it BEFORE it can settle in the band (this battery must never succeed)
    place_rel(scene.block, (-1.2, -1.2, G.BLOCK_SZ[2] / 2 + 0.001), settle_steps=30)
    check("settle gate: slider INSIDE the target band but moving "
          f"(|v|={v_now:.2f} m/s > {c.settle_lin} m/s) is NOT success (velocity "
          "gates are real); removed before it can settle", gate_ok)

    # =========================== 13-14. part-way cap + latched credit =======================
    env.reset(seed=111)
    step(30)
    tx = float(scene.x_target[0])
    place_rel(scene.pin, (0.9, -0.9, 0.10), settle_steps=30)  # pin_clear latch
    place(scene.block, (tx, 0.0, G.LIP_TOP + G.BLOCK_SZ[2] / 2 + 0.001),
          settle_steps=30)  # crossed + full-approach latch, but ON the lips
    report("part-way")
    s_before, ok = judge()
    lat = (float(scene.pin_clear[0]), float(scene.crossed[0]),
           float(scene.appr_max[0]))
    check("part-way + cap: pin depoted and slider held over the target band ON the "
          f"lips constructs all three latches (latches={lat}) -> score == 0.60 cap "
          f"({s_before:.4f}, float32 + eps), NOT success — full credit short of "
          "success is impossible without an in-track settled slider",
          lat[0] == 1.0 and lat[1] == 1.0 and lat[2] >= 0.999
          and 0.595 <= s_before <= 0.60 + 1e-5 and not ok)
    place_rel(scene.block, (-1.2, -1.2, G.BLOCK_SZ[2] / 2 + 0.001), settle_steps=40)
    report("moved-away")
    s_after, ok = judge()
    check("latched credit: teleporting the slider off to the floor leaves the "
          f"latched score unchanged ({s_before:.2f} -> {s_after:.2f}), still no "
          "success", abs(s_after - s_before) < 1e-3 and not ok)

    # =========================== 15. wrong object ===========================================
    env.reset(seed=121)
    step(30)
    tx = float(scene.x_target[0])
    place_rel(scene.pin, (0.9, -0.9, 0.10), settle_steps=20)  # extract first
    # lay the pin ACROSS the track at the target band (shaft along canonical y,
    # bridging both lip strips)
    r2 = math.sqrt(0.5)
    place(scene.pin, (tx, 0.0, G.LIP_TOP + G.PIN_SHAFT[0] / 2 + 0.002),
          quat_local=(r2, r2, 0.0, 0.0), settle_steps=45)
    report("wrong-object")
    pr, _pb = scene.pin_canon()
    s, ok = judge()
    check("wrong object: the extracted PIN laid across the track at the target band "
          f"(pin canon x={float(pr[0, 0]):+.3f}, z={float(pr[0, 2]):.3f}) does not "
          "satisfy the rubric — the SELECTOR must be there (in_band False, no "
          "success)",
          abs(float(pr[0, 0]) - tx) < 0.03 and float(pr[0, 2]) > G.RAIL_TOP
          and not bool(scene.in_band()[0]) and not ok)

    # =========================== 16-17. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies())
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== 18. camera + save + verdict ================================
    check(f"camera: >= 20 rgb frames captured ({len(frames)})", len(frames) >= 20)
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.channel_console")
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
    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
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

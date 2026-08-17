"""smoke — REJECTION battery for the KeyturnSpreaderScene rubric (NullRobot, RECORDED).

This module is NOT a solution (solve.py — teleport-transport + wrench-servo key turn —
already proves the rubric ACCEPTS the correct outcome). Every check here CONSTRUCTS a
wrong strategy or a near-miss (teleports are instrumentation) and asserts the rubric
REJECTS it. success() is audited at every judged point and must NEVER be True anywhere:

  1./2.  settle/no-NaN   — reset settles finite; anchors physically at the sampled gap
                           (|gap - g0| readback); key lying far from the rig; run armed
                           (positive control of the arming gate); score 0, no latches;
  3.     randomization   — READBACK across 8 seeds: rig xy/yaw, g0, key spawn xy and
                           spawn yaw all vary, and the anchors sit at g0 every seed;
  4.     null policy     — 240 idle steps -> score 0, nothing latched;
  5.     seed strategy   — the seed's plan (grasp-lift-CARRY the knife-shaped object):
                           the key is pinned upright at carry poses, including the exact
                           hover directly above the slot, altitude verified -> zero
                           latches, score 0 (holding/carrying the tool is worthless);
  6.     turned key      — PHYSICAL: the key dropped onto the slot turned 90 deg cannot
                           pass (88 mm paddle vs 30 mm slot): while it stays upright its
                           root never crosses the roof plane; key_in never latches;
  7.     mouth ram       — PHYSICAL: the strongest inward push on an anchor (3 N, the
                           only push a letterbox mouth allows) slides the PAIR together
                           toward the far mouth: displacement verified, cover breaks,
                           the run dies, gap progress stays zero, score 0;
  8./9.  rearrange cheat — uncover an anchor (run breaks), then place both anchors
                           SPREAD, covered, settled: split must NOT latch and success
                           must stay False (armed-run semantics — re-attempts must
                           restart from a near-closed gap); re-closing re-arms the run
                           (readback) but still scores 0;
 10.     key shortcut    — the key teleported directly to its seated pose latches only
                           key_in + keyed (0.25): the key in place is not the goal;
 11.     near-miss       — anchors spread to 70 mm (< split_gap 80): partial armed-run
                           progress only (0.25 + 0.45*prog), no split, no success;
 12.     latched credit  — re-closing the anchors keeps the near-miss score;
 13.     settle gate     — the covered-spread configuration MOVING at 0.35 m/s is
                           refused (split latches — positive control — but success
                           stays False while the anchors move);
 14.     cover break     — after the split latch, an uncovered anchor or a re-closed
                           gap keeps success False forever (live conjunction); the
                           latched 0.60 never becomes 1.0;
 15./16. audit           — success() never True at any judged point, max score <= 0.61;
                           final no-NaN; frames.npz saved.

Run (forge): python -u -m simgen_tasks.approach_grasp_knife_i292.smoke --headless
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the L20/4090 driver version and silently rejects RTX -> the
# annotator returns EMPTY frames. Disable the driver check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.approach_grasp_knife_i292 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    from isaaclab.utils.math import quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.keyturn_spreader")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    ids = torch.zeros(1, dtype=torch.long, device=device)
    zero = torch.zeros(1, 1, 3, device=device)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device)

    # --- recording (viewport rgb annotator, the proven forge mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.42, -0.95, 0.70)) + o),
                                tuple(np.array((0.42, 0.0, 0.08)) + o),
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

    never_success = [True]
    max_score = [0.0]

    def judge() -> tuple[float, bool]:
        """Sample the rubric at a judged point and feed the global audit."""
        s = float(scene.score()[0])
        ok = bool(scene.success()[0])
        never_success[0] &= not ok
        max_score[0] = max(max_score[0], s)
        return s, ok

    def report(tag: str) -> None:
        pl, pr = scene.anchors_local()
        s, ok = judge()
        print(f"[smoke] {tag:18s} gap={float(scene.gap()[0]) * 1000:6.1f}mm "
              f"ax=({float(pl[0, 0]):+.3f},{float(pr[0, 0]):+.3f}) "
              f"in={bool(scene.key_in[0])} keyed={bool(scene.keyed[0])} "
              f"split={bool(scene.split_latch[0])} run={bool(scene.run_ok[0])} "
              f"cov={bool(scene.covered_both()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def all_finite() -> bool:
        return (bool(torch.isfinite(scene.key.data.root_state_w).all())
                and bool(torch.isfinite(scene.anchor_l.data.root_state_w).all())
                and bool(torch.isfinite(scene.anchor_r.data.root_state_w).all()))

    az = c.block_size[2] / 2 + 0.001  # anchor resting centre height (rig-local)

    def rig_quat(extra_yaw: float = 0.0) -> tuple[float, float]:
        half = (float(scene.r_yaw[0]) + extra_yaw) / 2
        return math.cos(half), math.sin(half)

    def body_state(x: float, y: float, z: float, extra_yaw: float = 0.0,
                   vx_local: float = 0.0) -> torch.Tensor:
        """13-state at a rig-local pose (upright, rig yaw + extra), rig-local x velocity."""
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = scene.local_to_world(torch.tensor([[x, y, z]], device=device), ids)
        qw, qz = rig_quat(extra_yaw)
        st[0, 3], st[0, 6] = qw, qz
        if vx_local:
            cy, sy = math.cos(float(scene.r_yaw[0])), math.sin(float(scene.r_yaw[0]))
            st[0, 7], st[0, 8] = cy * vx_local, sy * vx_local
        return st

    def put_anchors(xl: float, xr: float, settle: int = 30,
                    vl: float = 0.0, vr: float = 0.0) -> None:
        scene.anchor_l.write_root_state_to_sim(body_state(xl, 0.0, az, vx_local=vl), ids)
        scene.anchor_r.write_root_state_to_sim(body_state(xr, 0.0, az, vx_local=vr), ids)
        if settle:
            step(settle)

    def put_key(x: float, y: float, z: float, extra_yaw: float = 0.0,
                settle: int = 0) -> None:
        scene.key.write_root_state_to_sim(body_state(x, y, z, extra_yaw), ids)
        if settle:
            step(settle)

    def pin_key(x: float, y: float, z: float, n: int) -> None:
        """Hold the key at an upright rig-local pose for n substeps (carry emulation)."""
        for _ in range(n):
            scene.key.write_root_state_to_sim(body_state(x, y, z), ids)
            step(1)

    def key_root_local() -> torch.Tensor:
        return scene.world_to_local(scene.key.data.root_pos_w)[0]

    # =========================== 1./2. settle / no-NaN ======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    kp = key_root_local()
    g_read = float(scene.gap()[0])
    check("settle: finite state, anchors physically at the sampled gap g0, key lying "
          "far from the rig, everything still",
          all_finite() and abs(g_read - float(scene.g0[0])) < 0.004
          and abs(float(kp[1])) > 0.15 and float(kp[2]) < 0.03
          and bool(scene.settled()[0]))
    check("settle: score 0, no latches, run armed (positive control of the arming gate)",
          float(scene.score()[0]) <= 1e-6 and not bool(scene.key_in[0])
          and not bool(scene.keyed[0]) and not bool(scene.split_latch[0])
          and bool(scene.run_ok[0]) and not bool(scene.success()[0]))

    # =========================== 3. randomization (readback) ================================
    reads, gap_ok = [], []
    for s in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=s)
        step(5)
        kp = key_root_local()
        shank = quat_apply(scene.key.data.root_quat_w, ez.expand(1, 3))[0]  # lying: horizontal
        psi = math.atan2(float(shank[1]), float(shank[0]))
        reads.append((float(scene.r_pos[0, 0]), float(scene.r_pos[0, 1]),
                      float(scene.r_yaw[0]), float(scene.g0[0]),
                      float(kp[0]), float(kp[1]), psi))
        gap_ok.append(abs(float(scene.gap()[0]) - float(scene.g0[0])) < 0.004)
    arr = np.array(reads)
    spread = arr.max(axis=0) - arr.min(axis=0)
    print(f"[smoke] randomization readback (rx, ry, yaw, g0, kx, ky, psi):\n{arr}", flush=True)
    print(f"[smoke] spreads={spread} anchors_at_g0={gap_ok}", flush=True)
    check("randomization: rig pose, initial gap, key spawn pose all vary across seeds "
          "(readback), and the anchors physically sit at g0 every seed",
          spread[0] > 0.01 and spread[1] > 0.01 and spread[2] > 0.08
          and spread[3] > 0.003 and (spread[4] > 0.015 or spread[5] > 0.015)
          and spread[6] > 0.4 and all(gap_ok))

    # =========================== 4. null policy =============================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    check("null policy: 240 idle steps -> score 0, nothing latched, no success",
          float(scene.score()[0]) <= 1e-6 and not bool(scene.key_in[0])
          and not bool(scene.split_latch[0]) and not bool(scene.success()[0]))

    # =========================== 5. seed strategy: grasp-lift-carry =========================
    # The seed's whole plan: pick the knife-shaped object up and carry it. The key is held
    # (pinned) upright at carry poses — including the EXACT hover directly above the slot
    # that solve.py transports to — and the probe verifies its own altitude, so the
    # rejection cannot be vacuous. Holding the tool anywhere above the roof is worthless.
    env.reset(seed=41)
    step(60)
    alt_ok = True
    for (px, py, pz) in ((0.0, -0.20, 0.250), (0.15, 0.10, 0.200), (0.0, 0.0, 0.150)):
        pin_key(px, py, pz, 30)
        alt_ok &= float(scene.paddle_center_local()[0, 2]) > c.interior_h + 0.01
    report("carry-the-key")
    check("seed strategy: lifting/carrying the key (incl. hovering right above the slot) "
          "latches NOTHING (probe altitude verified), score 0",
          alt_ok and not bool(scene.key_in[0]) and not bool(scene.keyed[0])
          and float(scene.score()[0]) <= 1e-6 and not bool(scene.success()[0]))

    # =========================== 6. turned key cannot pass the slot =========================
    # PHYSICAL: the key is released upright over the slot but turned 90 deg (paddle along
    # the tunnel). The 88 mm paddle cannot pass the 30 mm slot: while the key remains
    # upright its root must never cross the roof plane. (If it later topples off the rig
    # it is no longer upright, and the latch gates on uprightness + the slot column.)
    env.reset(seed=51)
    step(30)
    put_key(0.0, 0.0, c.interior_h + c.roof_t + 0.020, extra_yaw=math.pi / 2)
    min_z_upright = float("inf")
    for _ in range(120):
        step(1)
        u = quat_apply(scene.key.data.root_quat_w, ez.expand(1, 3))[0]
        if float(u[2]) > math.cos(math.radians(c.upright_max_deg)):
            min_z_upright = min(min_z_upright, float(key_root_local()[2]))
    step(120)
    report("turned-key")
    check("turned key: the slot physically refuses the 90deg paddle (root never below "
          "the roof plane while upright), key_in never latches, score 0",
          min_z_upright > c.interior_h - 0.006 and min_z_upright < 0.30
          and not bool(scene.key_in[0]) and not bool(scene.keyed[0])
          and float(scene.score()[0]) <= 1e-6)

    # =========================== 7. mouth ram: pushes only close/evict ======================
    # PHYSICAL: the only interaction a letterbox mouth allows is an INWARD push on the
    # near anchor. The hardest such push (3 N, CoM height, sustained) shoves the PAIR
    # toward the far mouth: the far anchor leaves the covered span (run breaks) and the
    # gap never grows while armed — no progress credit, ever.
    env.reset(seed=61)
    step(30)
    cy, sy = math.cos(float(scene.r_yaw[0])), math.sin(float(scene.r_yaw[0]))
    f = torch.zeros(1, 1, 3, device=device)
    f[0, 0, 0], f[0, 0, 1] = 3.0 * cy, 3.0 * sy  # rig-local +x, world frame
    scene.anchor_l.set_external_force_and_torque(f, zero, env_ids=ids, is_global=True)
    step(400)
    scene.anchor_l.set_external_force_and_torque(zero, zero, env_ids=ids)
    step(60)
    report("mouth-ram")
    _pl, pr = scene.anchors_local()
    check("mouth ram: the pair slides to the far mouth (displacement verified), cover "
          "breaks, the run dies, zero progress credit",
          float(pr[0, 0]) > c.x_cov and not bool(scene.run_ok[0])
          and not bool(scene.split_latch[0]) and float(scene.score()[0]) <= 1e-6
          and not bool(scene.success()[0]))

    # =========================== 8./9. uncover -> rearrange-spread cheat ====================
    # Break cover first (as any drag-out route must), then CONSTRUCT the spread outcome:
    # both anchors covered, 90 mm apart, settled. The armed-run gate must refuse it —
    # the run died and can only re-arm from a near-closed gap.
    env.reset(seed=71)
    step(30)
    put_anchors(-0.045, c.x_cov + 0.010, settle=30)  # right anchor just outside cover
    run_after_uncover = bool(scene.run_ok[0])
    put_anchors(-0.075, 0.075, settle=60)  # the spread outcome, covered + settled
    report("rearranged-spread")
    check("rearrange cheat: a constructed covered spread AFTER a cover break latches "
          "NO split and never succeeds (armed-run semantics)",
          not run_after_uncover and not bool(scene.split_latch[0])
          and bool(scene.covered_both()[0]) and float(scene.gap()[0]) > c.split_gap
          and float(scene.score()[0]) <= 1e-6 and not bool(scene.success()[0]))
    put_anchors(-0.045, 0.045, settle=30)  # back together: gap 0.030 <= close_gap
    report("re-closed")
    check("re-arm only from near-closed: re-closing re-arms the run (readback) but "
          "score stays 0 and split stays off",
          bool(scene.run_ok[0]) and not bool(scene.split_latch[0])
          and float(scene.score()[0]) <= 1e-6 and not bool(scene.success()[0]))

    # =========================== 10.-12. key shortcut, near-miss, latched credit ============
    env.reset(seed=81)
    step(30)
    put_key(0.0, 0.0, 0.003, settle=60)  # seated in the gap, untwisted (instrumentation)
    report("key-seated")
    check("key shortcut: the key seated in the gap latches only key_in+keyed (0.25) — "
          "the key in place is not the goal",
          bool(scene.key_in[0]) and bool(scene.keyed[0])
          and abs(float(scene.score()[0]) - 0.25) < 1e-3
          and not bool(scene.success()[0]))
    put_anchors(-0.065, 0.065, settle=40)  # gap 70 mm < split_gap 80 (armed run, covered)
    prog = float(((scene.run_maxgap[0] - c.close_gap)
                  / (c.split_gap - c.close_gap)).clamp(0.0, 1.0))
    s_near = float(scene.score()[0])
    report("near-miss-70mm")
    check("near-miss: a 70 mm spread earns partial armed-run progress only "
          "(0.25 + 0.45*prog), no split, no success",
          0.30 < s_near < 0.70 and abs(s_near - (0.25 + 0.45 * prog)) < 1e-3
          and prog < 0.999 and not bool(scene.split_latch[0])
          and not bool(scene.success()[0]))
    put_anchors(-0.045, 0.045, settle=30)  # regress the gap back closed
    report("regressed")
    check("latched credit: re-closing the anchors keeps the near-miss score",
          abs(float(scene.score()[0]) - s_near) < 1e-3 and not bool(scene.success()[0]))

    # =========================== 13./14. settle gate + cover break ==========================
    # The covered-spread configuration MOVING: anchors written 90 mm apart with 0.35 m/s
    # outward velocity while the run is armed. split latches (positive control) but
    # success must refuse while they move; the cover is then broken before they can
    # settle, and afterwards no state without a live covered >=75 mm gap ever succeeds.
    env.reset(seed=91)
    step(30)
    put_anchors(-0.075, 0.075, settle=0, vl=-0.35, vr=0.35)
    step(1)
    v_now = max(float(scene.anchor_l.data.root_lin_vel_w[0].norm()),
                float(scene.anchor_r.data.root_lin_vel_w[0].norm()))
    s_move, ok_move = judge()
    split_now = bool(scene.split_latch[0])
    print(f"[smoke] settle-gate probe: |v|={v_now:.2f} split={split_now} "
          f"score={s_move:.3f} success={ok_move}", flush=True)
    put_anchors(-0.045, c.x_cov + 0.010, settle=30)  # break cover before it can settle
    s_unc, ok_unc = judge()
    check("settle gate: the covered spread MOVING at speed is refused (split latches "
          "— positive control — but success stays False while the anchors move)",
          v_now > c.settle_speed and split_now and not ok_move
          and abs(s_move - 0.60) < 2e-3)
    put_anchors(-0.045, 0.045, settle=40)  # re-closed: gap < live_gap
    report("post-split-closed")
    check("cover break: after the split latch, an uncovered anchor and a re-closed gap "
          "both keep success False; the latched 0.60 never becomes 1.0",
          not ok_unc and abs(s_unc - 0.60) < 2e-3
          and abs(float(scene.score()[0]) - 0.60) < 2e-3
          and not bool(scene.success()[0]))

    # =========================== 15./16. audit + final no-NaN ===============================
    check("audit: success() was never True at any judged point; max score <= 0.61",
          never_success[0] and max_score[0] <= 0.61)
    check("final: all body states finite", all_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.keyturn_spreader")
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
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()

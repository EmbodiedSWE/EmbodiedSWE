"""Smoke / rubric-REJECTION battery for TunnelShuttleScene (sim_gen task
`pick_single_egad_i3`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — force-extract the gate pin, force-slide the shuttle
out of the tunnel, gravity-drop into the bin — is the acceptance evidence that the
rubric ACCEPTS a correct outcome). Every teleport here is instrumentation that
CONSTRUCTS a wrong (or partial) outcome as a settled state and asserts the rubric
REJECTS it — plus two APPLIED-FORCE probes that prove the puzzle mechanism is
physically real (the cover really traps the shuttle; the seated gate really blocks the
slide). No probe in this battery ever reaches success(), and a final audit check
asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite and still, score ~0 at rest;
  3-4. randomization      — READBACK over 6 seeded resets: fixture xy+yaw and shuttle
                            start depth vary; bin bearing + decoy pose vary;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  SEED strategy       — the seed's plan (grasp the loose object, lift it, carry
                            it): the red decoy settled INSIDE the goal bin -> NOT
                            success, score ~0 (identity matters; the loose object is
                            worthless);
  7.  cover mechanism     — 2.5 N upward force on the mid-tunnel shuttle for 1.5 s:
                            the cover holds it (root rises < 2 cm, never clears the
                            walls) — the seed's lift is physically impossible here;
  8.  gate mechanism      — forward slide force with the gate still seated: the
                            shuttle stalls BEHIND the gate station, never exits;
  9.  wrong object        — the yellow gate pin settled in the bin -> score ~0, no
                            success;
  10. near-miss place     — shuttle settled on the ground just OUTSIDE the bin wall
                            -> NOT success, score < 0.9;
  11. wrong place         — shuttle resting ON TOP of the tunnel cover (a state no
                            legal path produces) -> NOT success;
  12. monotonicity        — a deeper in-tunnel probe latches strictly more slide
                            credit than a shallower one;
  13. latched credit      — returning the shuttle to its start leaves the latched
                            score unchanged (credit does not evaporate);
  14. rejection audit     — success() was never True at ANY judged point;
  15. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.pick_single_egad_i3.smoke --headless
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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tunnel_shuttle")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((0.85, -0.85, 0.65)) + o),
                                tuple(np.array((0.08, 0.00, 0.03)) + o),
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

    def fixture_pose() -> tuple[torch.Tensor, float]:
        fp = (scene.fixture.data.root_pos_w - scene.env_origins)[0]
        q = scene.fixture.data.root_quat_w[0]
        return fp, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def to_world(x_l: float, y_l: float) -> tuple[float, float]:
        fp, fyaw = fixture_pose()
        cy, sy = math.cos(fyaw), math.sin(fyaw)
        return float(fp[0]) + cy * x_l - sy * y_l, float(fp[1]) + sy * x_l + cy * y_l

    def report(tag: str) -> None:
        sl = scene._local(scene.shuttle)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | shuttle_local=({float(sl[0]):+.3f},{float(sl[1]):+.3f},"
              f"{float(sl[2]):.3f}) latches=(g {float(scene.gate_latch[0]):.2f}, "
              f"s {float(scene.slide_latch[0]):.2f}, a {float(scene.approach_latch[0]):.2f}) "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_body(body, x: float, y: float, z: float, yaw: float = 0.0,
                   settle_steps: int = 30) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = math.cos(yaw / 2)
        st[:, 6] = math.sin(yaw / 2)
        st[:, 0:3] += env.iscene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def place_local(body, x_l: float, y_l: float, z: float, settle_steps: int = 30) -> None:
        """Place a body at fixture-local xy, aligned with the channel."""
        _fp, fyaw = fixture_pose()
        wx, wy = to_world(x_l, y_l)
        place_body(body, wx, wy, z, yaw=fyaw, settle_steps=settle_steps)

    def apply_force_steps(body, fl_x: float, fl_y: float, fz: float, k: int) -> None:
        """Apply a constant fixture-frame force for k steps, then clear it."""
        _fp, fyaw = fixture_pose()
        cy, sy = math.cos(fyaw), math.sin(fyaw)
        f_w = torch.tensor([cy * fl_x - sy * fl_y, sy * fl_x + cy * fl_y, fz],
                           device=device).view(1, 1, 3).expand(n, 1, 3)
        for _ in range(k):
            body.set_external_force_and_torque(f_w.contiguous(), zero_wrench,
                                               env_ids=all_ids, is_global=True)
            step(1)
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def bin_center() -> torch.Tensor:
        return (scene.goal_bin.data.root_pos_w - scene.env_origins)[0]

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    st0 = scene.shuttle.data.root_state_w
    sl = scene._local(scene.shuttle)[0]
    check("settle: shuttle state finite, at rest on the channel floor under the cover",
          bool(torch.isfinite(st0).all()) and bool(scene.settled()[0])
          and abs(float(sl[2]) - c.shuttle_root_z) < 0.005
          and c.start_x_range[0] - 0.01 <= float(sl[0]) <= c.start_x_range[1] + 0.01)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        fp, fyaw = fixture_pose()
        sx = float(scene._local(scene.shuttle)[0, 0])
        bl = scene._local(scene.goal_bin)[0]
        dp = (scene.decoy.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(fp[0]), float(fp[1]), fyaw, sx,
                      math.atan2(float(bl[1]), float(bl[0]) - c.chan_half),
                      float(dp[0]), float(dp[1])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (fix_x, fix_y, fix_yaw, shuttle_start_x, "
          f"bin_bearing, decoy_x, decoy_y):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: fixture xy + yaw and shuttle start depth vary across seeded "
          "resets (readback)",
          spread[0] > 0.005 and spread[1] > 0.005 and spread[2] > 0.05 and spread[3] > 0.008)
    check("randomization: bin bearing + decoy pose vary across seeded resets (readback)",
          spread[4] > 0.1 and (spread[5] > 0.005 or spread[6] > 0.005))

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. SEED strategy: pick the loose object ====================
    # The seed's whole plan is "grasp the one loose object in the open and lift/carry
    # it". The only loose free-standing object here is the red decoy. Constructed end
    # state: decoy settled INSIDE the goal bin. Identity matters: no success, ~0 score.
    torch.manual_seed(41)
    env.reset()
    step(10)
    bp = bin_center()
    place_body(scene.decoy, float(bp[0]), float(bp[1]),
               c.bin_floor_t + c.decoy_size / 2 + 0.004, settle_steps=90)
    report("seed-strategy")
    s, ok = judge()
    dis = (scene.decoy.data.root_pos_w - scene.goal_bin.data.root_pos_w)[0, :2].norm()
    check("seed strategy (loose red decoy lifted and binned): decoy IS in the bin, yet "
          "score ~0 and NOT success",
          float(dis) < 0.03 and s <= 0.05 and not ok)

    # =========================== 7. cover mechanism: shuttle cannot be lifted ===============
    # The seed's core action — lift the object straight up — applied to the ACTUAL
    # target: 2.5 N up (25x the shuttle's weight-margin over gravity) for 1.5 s. The
    # cover must hold the shuttle in the tunnel.
    torch.manual_seed(51)
    env.reset()
    step(10)
    z_before = float(scene._local(scene.shuttle)[0, 2])
    apply_force_steps(scene.shuttle, 0.0, 0.0, 2.5, 180)
    z_peak = float(scene._local(scene.shuttle)[0, 2])
    report("lift-attempt")
    s, ok = judge()
    step(60)  # release, settle back
    check("cover mechanism: 2.5 N upward pull cannot extract the shuttle (rose "
          f"{(z_peak - z_before) * 1000:.0f} mm, stayed under the wall top), no success",
          z_peak < c.wall_top - 0.005 and not ok)

    # =========================== 8. gate mechanism: seated gate blocks the slide ============
    # Forward slide force with the gate still seated: the shuttle must stall BEHIND
    # the gate station and never reach the exit.
    torch.manual_seed(61)
    env.reset()
    step(10)
    apply_force_steps(scene.shuttle, 1.5, 0.0, 0.0, 300)
    x_stall = float(scene._local(scene.shuttle)[0, 0])
    gate_x_now = float(scene._local(scene.gate)[0, 0])
    report("blocked-slide")
    s, ok = judge()
    check("gate mechanism: with the pin seated, a 1.5 N slide force stalls the shuttle "
          f"behind the gate station (x={x_stall:+.3f} < {c.gate_x:+.3f}+eps, gate holds "
          f"at x={gate_x_now:+.3f}), no exit, no success",
          x_stall < c.gate_x + 0.01 and abs(gate_x_now - c.gate_x) < 0.03
          and not bool(scene.clear_of_tunnel()[0]) and not ok)

    # =========================== 9. wrong object: gate pin in the bin =======================
    torch.manual_seed(71)
    env.reset()
    step(10)
    bp = bin_center()
    place_body(scene.gate, float(bp[0]), float(bp[1]),
               c.bin_floor_t + c.blade_h / 2 + 0.004, settle_steps=90)
    report("wrong-object")
    s, ok = judge()
    check("wrong object: the yellow gate pin settled in the bin scores ~0, no success",
          s <= 0.05 and not ok)

    # =========================== 10. near-miss: beside the bin ==============================
    torch.manual_seed(81)
    env.reset()
    step(10)
    bp = bin_center()
    off = c.bin_inner_half + c.bin_wall_t + c.base_size / 2 + 0.006
    place_body(scene.shuttle, float(bp[0]) + off, float(bp[1]),
               c.base_size / 2 + 0.002, settle_steps=60)
    report("near-miss-out")
    s, ok = judge()
    check("near-miss: shuttle settled on the ground just OUTSIDE the bin wall — NOT "
          "success, score < 0.9",
          not bool(scene.in_bin()[0]) and not ok and s < 0.9)

    # =========================== 11. wrong place: on top of the cover =======================
    torch.manual_seed(91)
    env.reset()
    step(10)
    place_local(scene.shuttle, -0.06, 0.0, c.cover_top + c.base_size / 2 + 0.003,
                settle_steps=90)
    report("on-cover")
    s, ok = judge()
    check("wrong place: shuttle resting ON TOP of the tunnel cover — NOT success, "
          "no in-bin credit, score < 0.9",
          not bool(scene.in_bin()[0]) and not ok and s < 0.9)

    # =========================== 12. monotonicity of slide credit ===========================
    torch.manual_seed(101)
    env.reset()
    step(10)
    place_local(scene.shuttle, -0.045, 0.0, c.shuttle_root_z + 0.001, settle_steps=40)
    judge()
    lat_shallow = float(scene.slide_latch[0])
    place_local(scene.shuttle, 0.030, 0.0, c.shuttle_root_z + 0.001, settle_steps=40)
    judge()
    lat_deep = float(scene.slide_latch[0])
    report("slide-probes")
    check("monotonicity: a deeper in-tunnel probe latched strictly more slide credit "
          f"({lat_shallow:.3f} < {lat_deep:.3f})", lat_shallow + 0.05 < lat_deep)

    # =========================== 13. latched credit survives moving back ====================
    s_deep, _ = judge()
    place_local(scene.shuttle, float(scene.start_x[0]), 0.0, c.shuttle_root_z + 0.001,
                settle_steps=40)
    report("moved-back")
    s_back, ok = judge()
    check("latched credit: returning the shuttle to its start leaves the latched score "
          "unchanged",
          abs(s_back - s_deep) < 1e-3 and not ok)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.shuttle.data.root_state_w).all()
           and torch.isfinite(scene.gate.data.root_state_w).all()
           and torch.isfinite(scene.decoy.data.root_state_w).all()
           and torch.isfinite(scene.goal_bin.data.root_state_w).all()
           and torch.isfinite(scene.fixture.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.tunnel_shuttle")
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
    main()

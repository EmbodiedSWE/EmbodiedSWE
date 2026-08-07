"""Smoke / rubric-REJECTION battery for PanCubbyScene (sim_gen task
`libero_kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf_i9`) — NullRobot,
teleported probe states, RECORDED.

This is NOT a solution (solve.py — the contact-dynamics extraction + gravity set-down —
is the acceptance evidence that the rubric ACCEPTS a correct outcome). Every teleport
here is instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled state
and asserts the rubric REJECTS it; no probe in this battery ever reaches success(), and
a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite and still inside the cubby,
                            score ~0 at rest;
  3-4. randomization      — READBACK over 6 seeded resets: pan spawn depth + handle yaw,
                            burner y + cubby yaw all move;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed strategy       — the seed's literal goal pose, pan settled ON TOP of the
                            cabinet: horizontally inside the footprint -> counts as NOT
                            extracted, no success, score <= 0.3;
  7.  roof-block probe    — a 1 m/s upward velocity kick inside the cubby: the pan rises
                            < 4 cm before the roof stops it (the lift-first plan is
                            physically jammed, not just unscored);
  8.  near-miss           — pan settled flat ON the burner but 6.5 cm off its axis
                            (tol 5 cm): NOT success, score < 0.9;
  9.  wrong place         — pan settled upright on the GROUND beside the burner
                            (correct-looking pose, wrong surface): NOT success;
  10. flipped             — pan settled upside-down centred on the burner: the upright
                            gate rejects it, score capped at 0.85, NOT success;
  11. monotonicity        — a straddle probe (half out the mouth) latches strictly less
                            extraction credit than a fully-out probe;
  12. latched credit      — teleporting the pan back deep inside after the fully-out
                            probe leaves the latched score unchanged (credit does not
                            evaporate), still no success;
  13. rejection audit     — success() was never True at ANY judged point in this
                            battery (rejection-only contract);
  14. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf_i9.smoke --headless
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
    env = ENVS.get("simgen.pan_cubby_retrieval")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.15, -1.15, 0.85)) + o),
                                tuple(np.array((-0.02, 0.00, 0.05)) + o),
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

    def pan_local():
        return scene._pan_local()[0]

    def cubby_pose() -> tuple[torch.Tensor, float]:
        cp = (scene.cubby.data.root_pos_w - scene.env_origins)[0]
        q = scene.cubby.data.root_quat_w[0]
        return cp, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def report(tag: str) -> None:
        loc = pan_local()
        s, ok = judge()
        print(f"[smoke] {tag:16s} | pan_local=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
              f"{float(loc[2]):.3f}) pe_latch={float(scene._pe_max[0]):.3f} "
              f"ext={bool(scene._ext[0])} pp_latch={float(scene._pp_max[0]):.3f} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_pan_world(x: float, y: float, z: float, quat=(1.0, 0.0, 0.0, 0.0),
                        settle_steps: int = 60) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the i17/i53 zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += env.iscene.env_origins
        scene.pan.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def place_pan_cubby_local(x_l: float, y_l: float, z: float | None = None,
                              settle_steps: int = 60) -> None:
        """Place the pan flat at cubby-local (x_l, y_l), handle pointing out the mouth."""
        cp, cyaw = cubby_pose()
        cy, sy = math.cos(cyaw), math.sin(cyaw)
        wx = float(cp[0]) + cy * x_l - sy * y_l
        wy = float(cp[1]) + sy * x_l + cy * y_l
        wz = z if z is not None else c.floor_lift + c.base_t / 2 + 0.002
        place_pan_world(wx, wy, wz,
                        quat=(math.cos(cyaw / 2), 0.0, 0.0, math.sin(cyaw / 2)),
                        settle_steps=settle_steps)

    def burner_xy_top() -> tuple[float, float, float]:
        b = (scene.burner.data.root_pos_w - scene.env_origins)[0]
        return float(b[0]), float(b[1]), float(b[2]) + c.burner_h / 2

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    st0 = scene.pan.data.root_state_w
    loc0 = pan_local()
    inside = (abs(float(loc0[0])) < c.cubby_d / 2 and abs(float(loc0[1])) < c.cubby_w / 2
              and float(loc0[2]) < 0.03)
    still = float(scene.pan.data.root_lin_vel_w[0].norm()) < c.settle_speed
    check("settle: pan state finite, at rest on the cubby floor, inside the compartment",
          bool(torch.isfinite(st0).all()) and inside and still)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        loc = pan_local()
        pq = scene.pan.data.root_quat_w[0]
        pyaw = 2.0 * math.atan2(float(pq[3]), float(pq[0]))
        cp, cyaw = cubby_pose()
        bx, by, _bt = burner_xy_top()
        reads.append((float(loc[0]), float(loc[1]), pyaw - cyaw, cyaw, bx, by))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (pan_xloc, pan_yloc, handle_rel_yaw, cubby_yaw, "
          f"burner_x, burner_y):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: pan spawn depth + lateral + handle yaw vary across seeded resets "
          "(readback)", spread[0] > 0.02 and spread[1] > 0.01 and spread[2] > 0.1)
    check("randomization: cubby yaw + burner xy vary across seeded resets (readback)",
          spread[3] > 0.025 and spread[4] > 0.01 and spread[5] > 0.04)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy: pan on the cabinet TOP ===================
    # The seed's literal goal — pan placed on top of the cabinet — settles on the roof:
    # horizontally INSIDE the footprint, so it never counts as extracted and the burner
    # stages stay locked.
    torch.manual_seed(41)
    env.reset()
    step(10)
    roof_top = c.floor_lift + c.cubby_h + c.roof_t
    place_pan_cubby_local(0.0, 0.0, z=roof_top + c.base_t / 2 + 0.003, settle_steps=90)
    report("seed-strategy")
    s, ok = judge()
    loc = pan_local()
    check("seed strategy: pan settled ON TOP of the cabinet — not extracted, no success, "
          "score <= 0.3",
          float(loc[2]) > c.cubby_h and not bool(scene._ext[0]) and not ok and s <= 0.3)

    # =========================== 7. roof physically blocks lifting ==========================
    # Give the pan a 1 m/s upward kick inside the compartment: the roof stops it within
    # the 29 mm head room — the lift-first plan is jammed by geometry, not by the rubric.
    torch.manual_seed(51)
    env.reset()
    step(30)
    z_start = float(pan_local()[2])
    st = scene.pan.data.root_state_w.clone()
    st[:, 7:13] = 0.0
    st[:, 9] = 1.0  # +z world velocity kick
    scene.pan.write_root_state_to_sim(st, all_ids)
    max_rise = 0.0
    for _ in range(90):
        step(1)
        max_rise = max(max_rise, float(pan_local()[2]) - z_start)
    report("roof-kick")
    s, ok = judge()
    check("roof-block probe: 1 m/s upward kick rises < 4 cm (roof jams the lift), "
          "pan still in the compartment, no success",
          max_rise < 0.04 and not bool(scene._ext[0]) and not ok)
    print(f"[smoke] roof-kick max rise: {max_rise * 1000:.1f} mm (head room ~29 mm)",
          flush=True)

    # =========================== 8. near-miss: on the burner, off its axis ==================
    torch.manual_seed(61)
    env.reset()
    step(10)
    bx, by, btop = burner_xy_top()
    place_pan_world(bx + 0.065, by, btop + 0.015 + c.base_t / 2, settle_steps=90)
    report("near-miss")
    s_near, ok = judge()
    bottom_on = abs((float((scene.pan.data.root_pos_w - scene.env_origins)[0, 2])
                     - c.base_t / 2) - btop) < c.place_z_tol
    d_xy = float((scene.pan.data.root_pos_w[0, :2] - scene.burner.data.root_pos_w[0, :2]).norm())
    check("near-miss: pan settled flat ON the burner 6.5 cm off its axis (tol 5 cm) — "
          "NOT success, score < 0.9",
          bottom_on and d_xy > c.place_xy_tol and not ok and s_near < 0.9)

    # =========================== 9. wrong place: ground beside the burner ===================
    torch.manual_seed(71)
    env.reset()
    step(10)
    bx, by, btop = burner_xy_top()
    place_pan_world(bx, by + c.burner_r + c.pan_r + 0.01, c.base_t / 2 + 0.002,
                    settle_steps=60)
    report("wrong-place")
    s, ok = judge()
    check("wrong place: pan settled upright on the GROUND beside the burner — NOT success, "
          "score < 0.9", not ok and s < 0.9)

    # =========================== 10. flipped: upside-down on the burner =====================
    torch.manual_seed(81)
    env.reset()
    step(10)
    bx, by, btop = burner_xy_top()
    place_pan_world(bx, by, btop + c.pan_h + 0.005,
                    quat=(0.0, 1.0, 0.0, 0.0), settle_steps=90)  # 180 deg about x
    report("flipped")
    s, ok = judge()
    from isaaclab.utils.math import quat_apply
    up = quat_apply(scene.pan.data.root_quat_w,
                    torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(n, 3))
    check("flipped: pan upside-down centred on the burner — upright gate rejects, "
          "NOT success, score <= 0.86",
          float(up[0, 2]) < -0.5 and not ok and s <= 0.86)

    # =========================== 11. monotonicity of extraction credit ======================
    torch.manual_seed(91)
    env.reset()
    step(10)
    place_pan_cubby_local(c.cubby_d / 2, 0.0, settle_steps=40)  # straddling the mouth
    report("straddle")
    s_half, _ok = judge()
    pe_half = float(scene._pe_max[0])
    place_pan_cubby_local(c.x_exit + 0.06, 0.0, settle_steps=40)  # fully out, on the ground
    report("fully-out")
    s_out, _ok = judge()
    pe_out = float(scene._pe_max[0])
    check("monotonicity: fully-out probe latched strictly more extraction credit than the "
          f"straddle probe ({pe_half:.3f} < {pe_out:.3f}) and a higher score",
          pe_half + 0.05 < pe_out and s_half < s_out)

    # =========================== 12. latched credit survives going back in ==================
    place_pan_cubby_local(0.02, 0.0, settle_steps=40)  # back deep inside the compartment
    report("back-inside")
    s_back, ok = judge()
    check("latched credit: teleporting the pan back inside leaves the latched score "
          "unchanged, still no success", abs(s_back - s_out) < 1e-3 and not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.pan.data.root_state_w).all()
           and torch.isfinite(scene.cubby.data.root_state_w).all()
           and torch.isfinite(scene.burner.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.pan_cubby_retrieval")
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

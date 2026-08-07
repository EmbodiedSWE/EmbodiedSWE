"""Smoke / rubric-REJECTION battery for UmbrellaRailScene (sim_gen task
`put_umbrella_in_umbrella_stand_i72`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — carry the umbrella to a hover above the rail, lower
the open crook over the bar through contact, release, and let the pendulum settle — is
the acceptance evidence that the rubric ACCEPTS a correct outcome; it passes on seeds
0/1/2). Every teleport here is instrumentation that CONSTRUCTS a wrong (or partial)
outcome and asserts the rubric REJECTS it. No probe in this battery ever reaches
success(), and a final audit asserts exactly that.

  1-2. settle/no-NaN    — reset layout settles finite: umbrella and cane lying flat on
                          opposite sides, rack near nominal; score ~0, no success;
  3-4. randomization    — READBACK over 6 seeded resets: rack position + yaw, umbrella
                          and cane ground poses, and which SIDE each object spawns on
                          all vary; every reset lies flat and sane;
  5.  null policy       — 240 idle steps -> score ~0, no success;
  6.  seed-strategy     — the seed family's outcome (umbrella STOOD upright on the
                          floor at the stand, as if inserted into a floor receptacle)
                          reconstructed as a lean against the rack post: tip on the
                          ground -> NOT suspended, score ~0. The anti-seed clause:
                          a standing/leaning umbrella is not a hanging one;
  7.  drape             — umbrella laid HORIZONTALLY across the top of the bar
                          (supported by the shaft, not the hook): bar far outside the
                          crook arc and shaft not crook-up -> rejected;
  8.  post-top perch    — crook rested on the TOP OF A POST instead of the bar: it
                          hangs suspended and crook-up, but the bar reads ~70 mm from
                          the arc center (> engage_tol) -> rejected: engagement means
                          the BAR inside the hook, not "hanging somewhere on the rack";
  9.  cane on the rail  — the CANE draped across the bar counts for NOTHING (success
                          and score are judged on the umbrella by identity);
  10. settle gate       — a genuinely hooked umbrella still SWINGING (velocity
                          injected; the sustained-stillness counter reads zero) is NOT
                          success — then removed before it can settle. Also anchors
                          the rubric: a hooked pose reads engage_dist ~25 mm < tol;
  11. latched credit    — teleporting the umbrella away afterwards leaves the latched
                          score (0.60) unchanged while engaged() drops to False;
  12. rejection audit   — success() was never True at ANY judged point;
  13. final no-NaN      — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.put_umbrella_in_umbrella_stand_i72.smoke --headless
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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.umbrella_rail")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.55, -1.25, 1.10)) + o),
                                tuple(np.array((0.40, 0.00, 0.35)) + o),
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

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        s, ok = judge()
        u, ca, r = rel(scene.umbrella), rel(scene.cane), rel(scene.rack)
        print(f"[smoke] {tag:16s} | umb=({float(u[0]):+.3f},{float(u[1]):+.3f},"
              f"{float(u[2]):.3f}) cane=({float(ca[0]):+.3f},{float(ca[1]):+.3f})"
              f" rack=({float(r[0]):+.3f},{float(r[1]):+.3f})"
              f" eng_d={float(scene.engage_dist()[0]):.4f}"
              f" tip_h={float(scene.tip_height()[0]):.3f}"
              f" up={bool(scene.crook_up()[0])} eng={bool(scene.engaged()[0])}"
              f" set={bool(scene.settled()[0])}"
              f" score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, xyz, quat=None, vel=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor([float(v) for v in xyz], device=device)
        if quat is None:
            st[:, 3] = 1.0
        else:
            st[:, 3:7] = torch.tensor([float(v) for v in quat], device=device)
        if vel is not None:
            st[:, 7:10] = torch.tensor([float(v) for v in vel], device=device)
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def rack_frame() -> tuple[torch.Tensor, torch.Tensor, float]:
        """(rel rack pos (3,), rack quat (4,), yaw)."""
        r = rel(scene.rack)
        q = scene.rack.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return r, q, yaw

    def rack_local(loc) -> torch.Tensor:
        """World (env-rel) position of a rack-local point."""
        r, q, _ = rack_frame()
        return r + quat_apply(q.view(1, 4),
                              torch.tensor([loc], device=device))[0]

    def upright_pose(arc_target: torch.Tensor) -> tuple[tuple, tuple]:
        """Umbrella pos/quat that puts the crook ARC CENTER at `arc_target` with the
        shaft vertical, hook plane aligned to the rack yaw (solve's carry pose)."""
        _, _, yaw = rack_frame()
        qu = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        off = quat_apply(torch.tensor([qu], device=device),
                         torch.tensor([c.arc_center_local], device=device))[0]
        pos = arc_target - off
        return tuple(float(v) for v in pos), qu

    def layout_sane(tag: str) -> bool:
        """Reset honesty: both items lying flat on opposite sides, rack near nominal."""
        u, ca, r = rel(scene.umbrella), rel(scene.cane), rel(scene.rack)
        ok = (float(u[2]) < 0.08 and float(ca[2]) < 0.08
              and abs(float(r[0]) - c.rack_pos[0]) < c.rack_jitter + 0.005
              and abs(float(r[1]) - c.rack_pos[1]) < c.rack_jitter + 0.005
              and abs(float(u[1])) > 0.15 and abs(float(ca[1])) > 0.15
              and float(u[1]) * float(ca[1]) < 0)
        if not ok:
            print(f"[smoke] layout sanity VIOLATION at {tag}", flush=True)
        return ok

    bodies = (scene.rack, scene.umbrella, scene.cane)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("settle: all states finite; umbrella and cane lying flat on opposite sides, "
          "rack near nominal", fin and layout_sane("reset"))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    sane = True
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(30)
        sane = sane and layout_sane(f"seed {sd}")
        u, ca = rel(scene.umbrella), rel(scene.cane)
        r, _q, yaw = rack_frame()
        reads.append((float(r[0]), float(r[1]), yaw, float(u[0]), float(u[1]),
                      float(ca[0]), float(ca[1]), 1.0 if float(u[1]) > 0 else -1.0))
    arr = np.array(reads)
    print("[smoke] randomization readback (rack_x, rack_y, rack_yaw, umb_x, umb_y, "
          f"cane_x, cane_y, side):\n{np.round(arr, 4)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: rack pose varies (readback spreads "
          f"x={spread[0]:.3f} y={spread[1]:.3f} yaw={spread[2]:.2f} rad)",
          spread[0] > 0.02 and spread[1] > 0.02 and spread[2] > 0.15)
    sides = {float(v) for v in arr[:, 7]}
    check("randomization: umbrella and cane ground poses vary and BOTH side "
          f"assignments occur (umb spread=({spread[3]:.3f},{spread[4]:.3f}), "
          f"{len(sides)} sides), every reset lies flat and sane",
          spread[3] > 0.02 and spread[4] > 0.10 and len(sides) == 2 and sane)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed-strategy: standing at the stand ====================
    # rlbench/put_umbrella_in_umbrella_stand's OUTCOME is the umbrella standing
    # UPRIGHT on the floor, held by a floor receptacle. Reconstruct its analog here:
    # the umbrella stood tip-down at the rack, leaning against a post. The tip is on
    # the ground -> suspended() is False and no stage ever latched: score ~0.
    env.reset(seed=41)
    step(30)
    p_post = rack_local((0.0, -c.post_y, 0.0))
    th = math.radians(32.0)
    zdir = torch.tensor([math.sin(th), 0.0, math.cos(th)], device=device)
    tip = torch.tensor([float(p_post[0]) - 0.31, float(p_post[1]), 0.012],
                       device=device)
    orig = tip - c.tip_local_z * zdir  # tip_local_z < 0: origin is up the shaft
    place(scene.umbrella, tuple(float(v) for v in orig),
          quat=(math.cos(th / 2), 0.0, math.sin(th / 2), 0.0), settle_steps=90)
    report("lean-at-rack")
    s, ok = judge()
    check("seed-strategy analog: umbrella stood/leaning at the rack with its tip on "
          f"the ground (tip_h={float(scene.tip_height()[0]) * 1000:.0f} mm) is NOT "
          "suspended — no success, score ~0",
          float(scene.tip_height()[0]) < c.clear_min and s <= 0.02 and not ok)

    # =========================== 7. drape across the bar ====================================
    # Umbrella laid HORIZONTALLY on top of the bar, supported by the shaft: the bar
    # is ~150 mm from the crook arc center and the shaft is not crook-up.
    env.reset(seed=51)
    step(20)
    bar_c = rack_local((0.0, 0.0, c.bar_z))
    _r, qr, yaw = rack_frame()
    # body +z along the rack's +x (perpendicular to the bar): q_rack_yaw * rot_y(90)
    hy, hz = math.pi / 4, yaw / 2
    qu = (math.cos(hz) * math.cos(hy), -math.sin(hz) * math.sin(hy),
          math.cos(hz) * math.sin(hy), math.sin(hz) * math.cos(hy))
    place(scene.umbrella,
          (float(bar_c[0]), float(bar_c[1]), float(bar_c[2]) + c.bar_r + 0.012),
          quat=qu, settle_steps=4)
    report("drape")
    d_drape = float(scene.engage_dist()[0])
    ok_drape = (d_drape > c.engage_tol + 0.05 and not bool(scene.crook_up()[0])
                and not bool(scene.success()[0]))
    judge()
    # remove it before it can teeter off and swing anywhere near a hooked pose
    place(scene.umbrella, (0.90, 0.55, 0.04), quat=(0.5, 0.5, 0.5, 0.5),
          settle_steps=30)
    check("drape: umbrella laid horizontally ACROSS the bar (shaft support, no hook) "
          f"reads eng_d={d_drape * 1000:.0f} mm >> tol and not crook-up -> rejected",
          ok_drape)

    # =========================== 8. post-top perch ==========================================
    # The crook rested on the TOP OF A POST: genuinely suspended and crook-up, but
    # the BAR is ~70 mm from the arc center -> engaged() is False. Engagement means
    # the bar inside the hook, not "hanging anywhere on the rack".
    env.reset(seed=61)
    step(20)
    hang_d = c.crook_r - c.crook_tube_r - c.bar_r
    post_top = rack_local((0.0, c.post_y, c.post_top))
    arc_tgt = post_top + torch.tensor([0.0, 0.0, hang_d + 0.002], device=device)
    pos, qu = upright_pose(arc_tgt)
    place(scene.umbrella, pos, quat=qu, settle_steps=8)
    report("post-perch")
    d_post = float(scene.engage_dist()[0])
    perch_ok = (d_post > c.engage_tol + 0.01 and bool(scene.crook_up()[0])
                and bool(scene.suspended()[0]) and not bool(scene.engaged()[0])
                and not bool(scene.success()[0]))
    judge()
    place(scene.umbrella, (0.90, 0.55, 0.04), quat=(0.5, 0.5, 0.5, 0.5),
          settle_steps=30)
    check("post-top perch: crook hooked over a POST instead of the rail hangs "
          f"suspended + crook-up yet reads eng_d={d_post * 1000:.0f} mm > "
          f"{c.engage_tol * 1000:.0f} mm -> engagement rejects it", perch_ok)

    # =========================== 9. cane on the rail (identity) =============================
    env.reset(seed=71)
    step(20)
    bar_c = rack_local((0.0, 0.0, c.bar_z))
    _r, _qr, yaw = rack_frame()
    hy, hz = math.pi / 4, yaw / 2
    qc = (math.cos(hz) * math.cos(hy), -math.sin(hz) * math.sin(hy),
          math.cos(hz) * math.sin(hy), math.sin(hz) * math.cos(hy))
    place(scene.cane,
          (float(bar_c[0]), float(bar_c[1]), float(bar_c[2]) + c.bar_r + 0.012),
          quat=qc, settle_steps=40)
    report("cane-on-rail")
    s, ok = judge()
    check("cane on the rail: the CANE draped over the bar counts for NOTHING — "
          "score ~0, no success (identity: the umbrella is judged, not 'some object')",
          s <= 0.02 and not ok)

    # =========================== 10. settle gate (hooked but swinging) ======================
    # A genuinely hooked umbrella, still moving: inject lateral velocity so the
    # sustained-stillness counter reads zero (this also guards the teleport trap —
    # stillness accumulated lying on the ground must not carry into the new pose).
    env.reset(seed=81)
    step(20)
    bar_c = rack_local((0.0, 0.0, c.bar_z))
    arc_tgt = bar_c - torch.tensor([0.0, 0.0, hang_d], device=device)
    pos, qu = upright_pose(arc_tgt)
    place(scene.umbrella, pos, quat=qu, vel=(0.30, 0.0, 0.0), settle_steps=2)
    report("hooked-swinging")
    d_hook = float(scene.engage_dist()[0])
    v_now = float(scene.umbrella.data.root_lin_vel_w[0].norm())
    s_before, ok = judge()
    gate_ok = (d_hook < c.engage_tol and bool(scene.engaged()[0])
               and v_now > c.settle_lin and not bool(scene.settled()[0]) and not ok)
    check("settle gate: a hooked umbrella (eng_d="
          f"{d_hook * 1000:.0f} mm < tol — the rubric anchor) still swinging at "
          f"{v_now:.2f} m/s is NOT success (sustained stillness is required)", gate_ok)

    # =========================== 11. latched credit survives moving away ====================
    # remove it BEFORE it can settle into a real hang (this battery must never succeed)
    place(scene.umbrella, (0.90, 0.55, 0.04), quat=(0.5, 0.5, 0.5, 0.5),
          settle_steps=40)
    report("moved-away")
    s_after, ok = judge()
    check("latched credit: teleporting the hooked umbrella away leaves the latched "
          f"score unchanged ({s_before:.2f} -> {s_after:.2f}, lift+engage = 0.60) "
          "while engaged() drops to False",
          abs(s_after - s_before) < 1e-3 and abs(s_after - 0.60) < 1e-3
          and not bool(scene.engaged()[0]) and not ok)

    # =========================== 12-13. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.umbrella_rail")
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

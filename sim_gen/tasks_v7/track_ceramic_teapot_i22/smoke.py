"""Smoke / rubric-REJECTION battery for MugRackScene (sim_gen task
`track_ceramic_teapot_i22`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — thread each mug's handle over its color hook under
force control and hand the weight over through contact — is the acceptance evidence
that the rubric ACCEPTS a correct outcome). Every teleport here is instrumentation
that CONSTRUCTS a wrong (or partial) outcome as a settled state and asserts the rubric
REJECTS it; no probe in this battery ever reaches success(), and a final audit check
asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite; mugs upright on the floor at
                            rest height (readback); score ~0 at rest, no success;
  3-4. randomization      — READBACK over 6 seeded resets: mug positions (slot shuffle
                            + jitter), mug yaw, rack xy + yaw all move;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  SEED strategy       — the seed's plan end state: carry the objects to the goal
                            area and SET THEM DOWN (one at the foot of the rack, one on
                            the rack's base plate), settled -> NOT success, score ~0;
  7.  perch near-miss     — red mug set ON TOP of the crossbar, directly above its
                            hook: elevated, near, settled — but NOT threaded -> NOT
                            success, only the small lift share;
  8.  wrong hook          — red mug genuinely hung on the BLUE hook (threaded readback
                            true), settled -> identity matters: NOT success, no
                            threading credit;
  9.  wrong object        — GREEN decoy genuinely hung on the RED hook, settled ->
                            NOT success, score ~0;
  10. one-mug partial     — red mug hung on the RED hook only -> NOT success, exactly
                            the single-hang floor credit;
  11. latched credit      — teleporting the hung red mug back to the floor leaves the
                            latched score unchanged;
  12. rejection audit     — success() was never True at ANY judged point;
  13. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.track_ceramic_teapot_i22.smoke --headless
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


# ----- scalar quaternion helpers (w, x, y, z) --------------------------------------------------
def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def qz(a):
    return (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))


def qx(a):
    return (math.cos(a / 2), math.sin(a / 2), 0.0, 0.0)


def qy(a):
    return (math.cos(a / 2), 0.0, math.sin(a / 2), 0.0)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.mug_rack")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.60, -0.80, 0.70)) + o),
                                tuple(np.array((0.20, 0.00, 0.18)) + o),
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

    def rack_pose() -> tuple[torch.Tensor, float]:
        rp = scene.rack.data.root_pos_w[0]  # world incl. env origin
        q = scene.rack.data.root_quat_w[0]
        return rp, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def report(tag: str) -> None:
        bits = []
        for name, hook in (("mug_red", 0), ("mug_blue", 1)):
            p = (scene.mugs[name].data.root_pos_w - scene.env_origins)[0]
            b = scene.mugs[name]
            bits.append(f"{name[4:]}z={float(p[2]):.3f} "
                        f"thr={bool(scene.threaded(name, hook)[0])} "
                        f"hang={bool(scene.hanging(name, hook)[0])} "
                        f"v={float(b.data.root_lin_vel_w[0].norm()):.3f}/"
                        f"{float(b.data.root_ang_vel_w[0].norm()):.2f}")
        s, ok = judge()
        print(f"[smoke] {tag:16s} | " + " | ".join(bits)
              + f" | score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    ap_local = torch.tensor([c.ap_center_x, 0.0, 0.0], device=device)
    tilt = math.radians(c.hook_tilt_deg)

    def place_world(body, pos: torch.Tensor, quat, settle_steps: int) -> None:
        """Kinematic probe placement at a WORLD pose (instrumentation, not a solution)
        + REAL physics steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def place_rack_local(body, x_l: float, y_l: float, z: float,
                         settle_steps: int = 90) -> None:
        """Place a body upright at rack-local (x_l, y_l) and height z above the ground."""
        rp, ryaw = rack_pose()
        ca, sa = math.cos(ryaw), math.sin(ryaw)
        pos = torch.tensor([float(rp[0]) + ca * x_l - sa * y_l,
                            float(rp[1]) + sa * x_l + ca * y_l,
                            float(scene.env_origins[0, 2]) + z], device=device)
        place_world(body, pos, (1.0, 0.0, 0.0, 0.0), settle_steps)

    def hang_on_hook(body, hook: int, settle_steps: int = 240) -> None:
        """Construct a genuinely-threaded pose: handle up, hook axis through the
        aperture center at 30 mm from the root; release and let it settle hanging.
        The release swing is damped by zeroing velocities in place a few times
        (instrumentation: this battery constructs SETTLED states, it does not solve)."""
        _rp, ryaw = rack_pose()
        q_des = qmul(qz(ryaw - math.pi / 2), qmul(qx(tilt), qy(-math.pi / 2)))
        qd_t = torch.tensor(q_des, device=device)
        off = quat_apply(qd_t.unsqueeze(0), ap_local.unsqueeze(0))[0]
        a = scene.hook_root_w(hook)[0] + scene._hook_dir_w()[0] * 0.030
        place_world(body, a - off, q_des, 60)
        for _ in range(5):
            st = body.data.root_state_w.clone()
            st[:, 7:13] = 0.0
            body.write_root_state_to_sim(st, all_ids)
            step(60)
        step(settle_steps)
        # The pose is stationary (z and xy stable over seconds) but kinematic-rack
        # contact chatter keeps ~0.6 rad/s of jitter alive; freeze it just before
        # judging so the JUDGED probe state is the settled configuration.
        st = body.data.root_state_w.clone()
        st[:, 7:13] = 0.0
        body.write_root_state_to_sim(st, all_ids)
        step(5)

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    fin0 = bool(torch.isfinite(scene.rack.data.root_state_w).all()
                and all(torch.isfinite(b.data.root_state_w).all()
                        for b in scene.mugs.values()))
    zs = [float((b.data.root_pos_w - scene.env_origins)[0, 2]) for b in scene.mugs.values()]
    settled0 = all(bool(scene.mug_settled(nm)[0]) for nm in c.mug_names)
    check("settle: states finite; all three mugs upright on the floor at rest height "
          "(readback); settled",
          fin0 and all(abs(z - c.mug_rest_z) < 0.012 for z in zs) and settled0)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        pr = (scene.mugs["mug_red"].data.root_pos_w - scene.env_origins)[0]
        pb = (scene.mugs["mug_blue"].data.root_pos_w - scene.env_origins)[0]
        q = scene.mugs["mug_red"].data.root_quat_w[0]
        myaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        rp, ryaw = rack_pose()
        ro = (rp - scene.env_origins[0])
        reads.append((float(pr[0]), float(pr[1]), myaw,
                      float(pr[0] - pb[0]), float(pr[1] - pb[1]),
                      float(ro[0]), float(ro[1]), ryaw))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (red_x, red_y, red_yaw, red-blue_dx, "
          f"red-blue_dy, rack_x, rack_y, rack_yaw):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: red mug xy + yaw AND the red-vs-blue arrangement (slot "
          "shuffle) vary across seeded resets (readback)",
          spread[0] > 0.02 and spread[1] > 0.02 and spread[2] > 0.5
          and (spread[3] > 0.02 or spread[4] > 0.02))
    check("randomization: rack xy + yaw vary across seeded resets (readback)",
          spread[5] > 0.008 and spread[6] > 0.008 and spread[7] > 0.05)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. SEED strategy: carry to the goal and SET DOWN ===========
    # The seed's whole plan ends with the carried object RESTING near the goal. Built
    # here: red mug set down on the floor at the foot of the rack, blue mug set down ON
    # the rack's base plate. Settled, at the goal area — the rubric must refuse it.
    torch.manual_seed(41)
    env.reset()
    step(10)
    place_rack_local(scene.mugs["mug_red"], 0.16, c.hook_y, c.mug_rest_z + 0.002)
    place_rack_local(scene.mugs["mug_blue"], 0.05, -c.hook_y,
                     c.base_h + c.mug_rest_z + 0.002)
    report("seed-strategy")
    s, ok = judge()
    zr = float((scene.mugs["mug_red"].data.root_pos_w - scene.env_origins)[0, 2])
    zb = float((scene.mugs["mug_blue"].data.root_pos_w - scene.env_origins)[0, 2])
    check("seed strategy (mugs carried to the rack and SET DOWN, one on the floor at "
          "its foot, one on its base plate): NOT success, score <= 0.02",
          not ok and s <= 0.02 and zr < 0.08 and zb < 0.13
          and not bool(scene.threaded("mug_red", 0)[0])
          and not bool(scene.threaded("mug_blue", 1)[0]))

    # =========================== 7. perch near-miss (on top of the crossbar) ================
    torch.manual_seed(51)
    env.reset()
    step(10)
    place_rack_local(scene.mugs["mug_red"], 0.0, c.hook_y, c.bar_z1 + c.mug_rest_z + 0.003,
                     settle_steps=150)
    report("perch")
    s, ok = judge()
    zr = float((scene.mugs["mug_red"].data.root_pos_w - scene.env_origins)[0, 2])
    check("perch near-miss: red mug settled ON TOP of the crossbar directly above its "
          "hook — elevated + near but NOT threaded: NOT success, score <= 0.15",
          not ok and zr > c.hang_z_min and not bool(scene.threaded("mug_red", 0)[0])
          and bool(scene.mug_settled("mug_red")[0]) and s <= 0.15)

    # =========================== 8. wrong hook ==============================================
    torch.manual_seed(61)
    env.reset()
    step(10)
    hang_on_hook(scene.mugs["mug_red"], hook=1)
    report("wrong-hook")
    s, ok = judge()
    check("wrong hook: red mug genuinely hung on the BLUE hook (threaded readback) — "
          "identity matters: NOT success, no threading credit (score <= 0.20)",
          not ok and bool(scene.threaded("mug_red", 1)[0])
          and float((scene.mugs["mug_red"].data.root_pos_w - scene.env_origins)[0, 2])
          > c.hang_z_min and s <= 0.20)

    # =========================== 9. wrong object ============================================
    torch.manual_seed(71)
    env.reset()
    step(10)
    hang_on_hook(scene.mugs["mug_green"], hook=0)
    report("wrong-object")
    s, ok = judge()
    check("wrong object: the GREEN decoy genuinely hung on the RED hook (threaded "
          "readback) scores ~0, no success",
          not ok and bool(scene.threaded("mug_green", 0)[0]) and s <= 0.02)

    # =========================== 10. one-mug partial credit =================================
    torch.manual_seed(81)
    env.reset()
    step(10)
    hang_on_hook(scene.mugs["mug_red"], hook=0)
    report("one-mug")
    s_one, ok = judge()
    check("one-mug partial: red mug hung on the RED hook only (hanging readback true) — "
          "NOT success, exactly the single-hang floor credit (0.55 <= s <= 0.65)",
          not ok and bool(scene.hanging("mug_red", 0)[0]) and 0.55 <= s_one <= 0.65)

    # =========================== 11. latched credit survives regression =====================
    place_rack_local(scene.mugs["mug_red"], 0.20, 0.0, c.mug_rest_z + 0.002,
                     settle_steps=60)
    report("regressed")
    s_reg, ok = judge()
    check("latched credit: teleporting the hung red mug back to the floor leaves the "
          "latched score unchanged",
          not ok and abs(s_reg - s_one) < 1e-3)

    # =========================== 12-13. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.rack.data.root_state_w).all()
           and all(torch.isfinite(b.data.root_state_w).all() for b in scene.mugs.values()))
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.mug_rack")
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

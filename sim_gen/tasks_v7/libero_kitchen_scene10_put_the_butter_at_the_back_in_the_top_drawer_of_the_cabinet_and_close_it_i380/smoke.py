"""Smoke / rubric-REJECTION battery for ButterHatchScene (sim_gen task
`libero_kitchen_scene10_..._i380`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — the velocity-servoed drawer close followed by the
gravity deposit through the hatch — is the acceptance evidence that the rubric ACCEPTS
a correct outcome). Every teleport here is instrumentation that CONSTRUCTS a wrong (or
partial) outcome as a settled state and asserts the rubric REJECTS it; no probe in
this battery ever reaches success(), and a final audit check asserts exactly that.

  1-2.  settle/no-NaN     — reset layout settles finite: drawer open inside its q0
                            band, both blocks outside on the floor, everything still,
                            score ~0 at rest;
  3-4.  randomization     — READBACK over 8 seeded resets: drawer opening q0 varies;
                            the block->slot assignment flips; per-slot xy jitter and
                            spawn yaw vary;
  5.   null policy        — 240 idle steps -> score ~0, no success;
  6.   seed strategy      — the seed's plan (drop the butter into the OPEN drawer
                            cavity from above) physically denied: the butter released
                            over the open cavity lands ON the fixed canopy roof, never
                            enters the drawer -> no success;
  7.   out-of-order       — the butter posted through the hatch while the drawer is
                            still OPEN falls past the drawer into the reject cellar;
                            closing the drawer afterwards does NOT recover it -> no
                            success, score stays low;
  8.   near-miss ajar     — butter settled INSIDE the cavity but the drawer left ajar
                            at q = -0.030 (outside close_tol) -> rejected;
  9.   wrong object       — WHITE paraffin inside the closed drawer, butter outside ->
                            rejected (color identification is load-bearing);
  10.  paraffin exclusion — butter AND paraffin both inside the closed drawer -> the
                            paraffin-out clause rejects;
  11.  goal-adjacent park — butter parked on the canopy directly above the CLOSED
                            drawer (right side of the countertop, wrong side of it) ->
                            rejected;
  12.  latched credit     — drawer closed (latch earned), then re-opened: the latched
                            close credit holds (does not evaporate), still no success;
  13.  rejection audit    — success() was never True at ANY judged point;
  14.  final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it_i380.smoke --headless
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
    env = ENVS.get("simgen.butter_hatch")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.30, -1.05, 0.85)) + o),
                                tuple(np.array((0.56, 0.00, 0.22)) + o),
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

    def qd() -> float:
        return float(scene.q()[0])

    def report(tag: str) -> None:
        b = (scene.butter.data.root_pos_w - scene.env_origins)[0]
        p = (scene.paraffin.data.root_pos_w - scene.env_origins)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | q={qd():+.4f} butter=({float(b[0]):+.3f},"
              f"{float(b[1]):+.3f},{float(b[2]):.3f}) "
              f"paraffin=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):.3f}) "
              f"in_cav={bool(scene.in_cavity(scene.butter)[0])} "
              f"in_cellar={bool(scene.in_cellar(scene.butter)[0])} "
              f"L=({int(scene._closed[0])},{int(scene._chuted[0])},"
              f"{int(scene._inside[0])}) score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, x: float, y: float, z: float, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += env.iscene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    cx, cy = c.cab_pos
    bh = c.butter_size[2] / 2

    def write_drawer(qv: float) -> None:
        """Follower-only re-pose of the drawer along its unchanged slide (probe
        constructor; the cabinet is never moved)."""
        place(scene.drawer, cx + qv, cy, c.plane_z)

    def obj_xy(body) -> tuple[float, float]:
        p = (body.data.root_pos_w - scene.env_origins)[0]
        return float(p[0]), float(p[1])

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    fin0 = (torch.isfinite(scene.butter.data.root_state_w).all()
            and torch.isfinite(scene.paraffin.data.root_state_w).all()
            and torch.isfinite(scene.drawer.data.root_state_w).all()
            and torch.isfinite(scene.cabinet.data.root_state_w).all())
    still = (float(scene.butter.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.drawer.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    check("settle: states finite, drawer open inside its q0 band, both blocks outside "
          "the drawer on the floor, everything still",
          bool(fin0) and c.q0_min - 0.012 < qd() < c.q0_max + 0.012 and still
          and not bool(scene.in_cavity(scene.butter)[0])
          and not bool(scene.in_cavity(scene.paraffin)[0])
          and not bool(scene.in_cellar(scene.butter)[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    slots = np.array(c.spawn_slots)
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(2)
        row = [qd()]
        for body in (scene.butter, scene.paraffin):
            px, py = obj_xy(body)
            d = np.hypot(slots[:, 0] - px, slots[:, 1] - py)
            row += [px, py, int(d.argmin())]
        q = scene.butter.data.root_quat_w[0]
        row.append(math.degrees(2.0 * math.atan2(float(q[3]), float(q[0]))))
        reads.append(row)
    arr = np.array(reads)
    print(f"[smoke] randomization readback (q0, bx, by, b_slot, px, py, p_slot, "
          f"b_yaw_deg):\n{arr}", flush=True)
    q0_spread = float(arr[:, 0].max() - arr[:, 0].min())
    sigs = {int(r[3]) for r in arr}
    check("randomization: drawer opening q0 varies and the block->slot assignment "
          f"flips across seeded resets (readback: q0 spread {q0_spread:.3f} m, "
          f"{len(sigs)} distinct assignments)", q0_spread > 0.02 and len(sigs) >= 2)
    jit = 0.0
    for col, slot_col in ((1, 3), (4, 6)):
        for s_id in range(2):
            grp = arr[arr[:, slot_col] == s_id]
            if len(grp) >= 2:
                jit = max(jit, float((grp[:, col:col + 2].max(axis=0)
                                      - grp[:, col:col + 2].min(axis=0)).max()))
    yaw_spread = float(arr[:, 7].max() - arr[:, 7].min())
    check("randomization: per-slot xy jitter and spawn yaw vary (readback)",
          jit > 0.004 and yaw_spread > 20.0)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy: drop into the open drawer ================
    # The seed's plan — lower the butter into the OPEN drawer cavity from above — is
    # geometrically denied here: the canopy roofs the drawer over its whole travel
    # (6 mm gap << 26 mm butter). Released directly above the open cavity, the butter
    # lands ON the canopy and never enters the drawer.
    torch.manual_seed(41)
    env.reset()
    step(10)
    q_open = qd()
    place(scene.butter, cx + q_open, cy, 0.320)  # directly above the open cavity centre
    step(120)
    report("seed-strategy")
    s, ok = judge()
    bz = float((scene.butter.data.root_pos_w - scene.env_origins)[0, 2])
    check("seed strategy: butter released above the OPEN drawer cavity lands on the "
          f"canopy roof (z={bz:.3f} > countertop), never enters the drawer — no "
          "success, score <= 0.20",
          bz > c.top_under + c.top_t - 0.005 and not bool(scene.in_cavity(scene.butter)[0])
          and not ok and s <= 0.20)

    # =========================== 7. out-of-order: post first, close later ===================
    # Butter posted through the hatch while the drawer is OPEN: the aligned column
    # continues past the drawer plane and the butter is LOST in the reject cellar.
    # Closing the drawer afterwards does not recover it.
    torch.manual_seed(51)
    env.reset()
    step(10)
    place(scene.butter, cx + c.port_cx, cy, c.chute_z_hi + 0.008)
    step(200)
    report("posted-open")
    in_cellar = bool(scene.in_cellar(scene.butter)[0])
    write_drawer(0.0)  # now close the drawer (follower-only probe write)
    step(90)
    report("closed-late")
    s, ok = judge()
    check("out-of-order: butter posted while the drawer was open fell past the drawer "
          f"into the reject cellar (in_cellar={in_cellar}), closing afterwards does "
          "not recover it — no success, score <= 0.35",
          in_cellar and bool(scene.in_cellar(scene.butter)[0])
          and bool(scene.drawer_closed()[0])
          and not bool(scene.in_cavity(scene.butter)[0]) and not ok and s <= 0.35)

    # =========================== 8. near-miss: drawer ajar ==================================
    # Butter settled INSIDE the cavity but the drawer ajar at q=-0.030 — outside
    # close_tol=0.008. The inside/chute latches require the CLOSED conjunct too.
    torch.manual_seed(61)
    env.reset()
    step(10)
    write_drawer(-0.030)
    place(scene.butter, cx - 0.030, cy, c.plane_z + bh + 0.004)
    step(90)
    report("ajar")
    s, ok = judge()
    check("near-miss ajar: butter settled inside the cavity but the drawer left ajar "
          f"at q={qd():+.3f} (|q| > close_tol={c.close_tol}) — no success, score <= 0.35",
          bool(scene.in_cavity(scene.butter)[0]) and not bool(scene.drawer_closed()[0])
          and not ok and s <= 0.35)

    # =========================== 9. wrong object: the white paraffin ========================
    torch.manual_seed(71)
    env.reset()
    step(10)
    write_drawer(0.0)
    place(scene.paraffin, cx + 0.02, cy, c.plane_z + bh + 0.004)
    step(90)
    report("wrong-object")
    s, ok = judge()
    check("wrong object: WHITE paraffin inside the closed drawer, butter outside — no "
          "success, score <= 0.35 (color identification is load-bearing)",
          bool(scene.in_cavity(scene.paraffin)[0]) and bool(scene.drawer_closed()[0])
          and not bool(scene.in_cavity(scene.butter)[0]) and not ok and s <= 0.35)

    # =========================== 10. paraffin-exclusion clause ==============================
    torch.manual_seed(81)
    env.reset()
    step(10)
    write_drawer(0.0)
    place(scene.butter, cx + 0.02, cy + 0.035, c.plane_z + bh + 0.004)
    place(scene.paraffin, cx + 0.02, cy - 0.035, c.plane_z + bh + 0.004)
    step(90)
    report("both-in")
    s, ok = judge()
    check("paraffin exclusion: butter AND paraffin both inside the closed drawer — the "
          "paraffin-out clause rejects, no success, score <= 0.85",
          bool(scene.in_cavity(scene.butter)[0]) and bool(scene.in_cavity(scene.paraffin)[0])
          and bool(scene.drawer_closed()[0]) and not ok and s <= 0.85)

    # =========================== 11. goal-adjacent park on the canopy =======================
    torch.manual_seed(91)
    env.reset()
    step(10)
    write_drawer(0.0)
    place(scene.butter, cx - 0.15, cy, 0.250)  # onto the canopy over the closed drawer
    step(90)
    report("parked-on-top")
    s, ok = judge()
    bz = float((scene.butter.data.root_pos_w - scene.env_origins)[0, 2])
    check("goal-adjacent park: butter parked on the canopy directly above the CLOSED "
          f"drawer (z={bz:.3f}) — on top of the countertop is not inside the drawer, "
          "no success", bz > c.top_under + c.top_t - 0.005
          and not bool(scene.in_cavity(scene.butter)[0]) and not ok)

    # =========================== 12. latched credit survives regression =====================
    s_a, ok_a = judge()  # drawer closed from check 11 -> _closed latched
    write_drawer(-0.120)  # re-open the drawer
    step(60)
    report("reopened")
    s_b, ok_b = judge()
    check("latched credit: the close latch earned then the drawer re-opened — score "
          f"holds ({s_a:.3f} -> {s_b:.3f}), still no success",
          bool(scene._closed[0]) and s_a >= c.w_close - 0.01 and s_b >= s_a - 1e-3
          and not ok_a and not ok_b)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.butter.data.root_state_w).all()
           and torch.isfinite(scene.paraffin.data.root_state_w).all()
           and torch.isfinite(scene.drawer.data.root_state_w).all()
           and torch.isfinite(scene.cabinet.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.butter_hatch")
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
    except BaseException as exc:  # noqa: BLE001 — die loudly, never hang until the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({type(exc).__name__}: {exc})", flush=True)
        os._exit(1)

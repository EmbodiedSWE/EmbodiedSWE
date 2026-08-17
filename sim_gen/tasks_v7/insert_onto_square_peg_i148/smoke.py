"""Smoke / rubric-REJECTION battery for HookEscapeScene (sim_gen task
`insert_onto_square_peg_i148`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — wrench-servo the blue ring up the post, through the
elbow, off the open arm tip, then carry it to the dish — is the acceptance evidence
that the rubric ACCEPTS a correct outcome; it passes on forge covering both stack
orders). Every teleport here is instrumentation that CONSTRUCTS a wrong (or partial)
outcome as a settled state and asserts the rubric REJECTS it — plus force probes
proving the captivity mechanism is real (the rail actually retains the ring against
straight pulls; the probes assert the actuator MOVED the ring, so a jam cannot pass
vacuously). No probe in this battery ever reaches success(), and a final audit check
asserts exactly that.

  1.  settle/no-NaN     — reset settles finite; both rings threaded ON the rail at
                          rest (rail_dist readback ~0), score ~0, no success;
  2.  randomization     — READBACK over 12 seeded resets: rack xy + yaw and the dish
                          position vary;
  3.  randomization     — BOTH stacking orders (blue-on-top / orange-on-top) occur;
  4.  null policy       — 300 idle steps -> score ~0, no success;
  5.  seed strategy     — rlbench/insert_onto_square_peg's whole goal (the ring
                          dropped DOWN over a vertical post, encircling it, settled)
                          is this task's reset state: constructed fresh by dropping
                          the blue ring onto the post from just above the credit
                          deadband -> score ~0, no success;
  6.  captivity (up)    — QUASI-STATIC straight-up pull (velocity-servoed ~0.06 m/s,
                          force capped ~1.8x weight; a constant blast would be a
                          dynamic slam that can ratchet the ring through the elbow —
                          the legitimate exit; the force is pre-encoded with
                          R_ref * R_now^T against the pod's wrench frame-drag so it
                          STAYS world-up once the jammed ring pitches): the ring
                          CLIMBS (readback: z rises > 0.10 m — the probe moved it)
                          but JAMS under the arm and never leaves the rail (max
                          rail_dist < free_tol, ends on-rail, l_free latch 0): no
                          success, partial credit only (score <= 0.35);
  7.  captivity (side)  — 1.5 N horizontal yank for 3 s: the ring is displaced but
                          the post retains it (rail_dist < free_tol throughout):
                          score ~0, no success;
  8.  hang on the arm   — blue ring threaded onto the ARM (x ~ 0.08) and released:
                          it dangles settled on the rail -> partial credit only
                          (climb+arm latches, score <= 0.47), no success;
  9.  floor near-miss   — blue ring flat + settled on the FLOOR beside the dish:
                          free-of-rail credit only (score <= 0.17), no success;
  10. rim perch         — blue ring perched on the dish WALL, tipping outward:
                          outside the xy/z acceptance -> no success, score <= 0.17;
  11. wrong ring        — ORANGE ring laid flat in the dish (blue untouched on the
                          rack): the wrong ring earns NOTHING (score ~0), decoy_clear
                          is False, no success;
  12. z-band + decoy    — blue ring dropped flat ON TOP of the in-dish orange: xy
                          inside tol and plane flat but center ABOVE the flat-on-floor
                          z-band -> in_dish(blue) False AND decoy_clear False, no
                          success;
  13. rejection audit   — success() was never True at ANY judged point;
  14. final no-NaN      — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.insert_onto_square_peg_i148.smoke --headless
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
# RTX recipe: kit mis-decodes the pod driver version and silently rejects RTX -> the
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
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod

_qapply = scene_mod._qapply
_qmul = scene_mod._qmul
_qinv = scene_mod._qinv

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.hook_escape")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.25, -1.05, 0.90)) + o),
                                tuple(np.array((0.10, -0.05, 0.15)) + o),
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

    def report(tag: str) -> None:
        s, ok = judge()
        db = float(scene.rail_dist(scene.ring_blue)[0])
        do = float(scene.rail_dist(scene.ring_orange)[0])
        print(f"[smoke] {tag:18s} | rail_d(b)={db:.3f} rail_d(o)={do:.3f} "
              f"in_dish(b)={bool(scene.in_dish(scene.ring_blue)[0])} "
              f"decoy_clear={bool(scene.decoy_clear()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def teleport(body, pos, quat, settle_steps: int = 45) -> None:
        """Probe placement (instrumentation, not a solution) + REAL physics steps
        before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3:7] = quat
        body.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def rack_frame(local) -> torch.Tensor:
        return scene.rack.data.root_pos_w + _qapply(
            scene.rack.data.root_quat_w,
            torch.tensor(local, device=device, dtype=torch.float32).expand(n, 3))

    def dish_frame(local) -> torch.Tensor:
        return scene.dish.data.root_pos_w + _qapply(
            scene.dish.data.root_quat_w,
            torch.tensor(local, device=device, dtype=torch.float32).expand(n, 3))

    def q_flat() -> torch.Tensor:
        q = torch.zeros(n, 4, device=device)
        q[:, 0] = 1.0
        return q

    def q_arm_threaded() -> torch.Tensor:
        """Ring bore axis along the rack arm (+x): rack yaw x pitch-90-about-y."""
        h = math.pi / 4
        qp = torch.tensor([math.cos(h), 0.0, math.sin(h), 0.0],
                          device=device).expand(n, 4)
        return _qmul(scene.rack.data.root_quat_w, qp)

    def blue_z_rack() -> float:
        return float(scene._rack_local(scene.ring_blue.data.root_pos_w)[0, 2])

    def all_finite() -> bool:
        bodies = [scene.rack, scene.dish, scene.ring_blue, scene.ring_orange]
        return all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)

    def force_probe(body, f_world: torch.Tensor, steps: int) -> tuple[float, float]:
        """Apply a constant world force for `steps`; judge along the way. Returns
        (max rail_dist seen, max rack-local z seen)."""
        d_max, z_max = 0.0, -1.0
        for i in range(steps):
            body.set_external_force_and_torque(f_world.view(n, 1, 3), zero_wrench,
                                               env_ids=all_ids, is_global=True)
            step(1)
            d_max = max(d_max, float(scene.rail_dist(body)[0]))
            z_max = max(z_max, float(scene._rack_local(body.data.root_pos_w)[0, 2]))
            if i % 60 == 0:
                judge()
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
        step(60)
        return d_max, z_max

    # =========================== 1. settle / no-NaN =====================================
    env.reset(seed=3)
    step(240)
    report("reset-settle")
    s, ok = judge()
    db = float(scene.rail_dist(scene.ring_blue)[0])
    do = float(scene.rail_dist(scene.ring_orange)[0])
    check(f"settle (seed 3): all states finite, both rings threaded ON the rail at rest "
          f"(rail_dist b={db:.3f}, o={do:.3f} < 0.02), score ~0, no success",
          all_finite() and db < 0.02 and do < 0.02 and s <= 0.02 and not ok)

    # =========================== 2-3. randomization is real =============================
    reads, tops = [], []
    for sd in range(21, 33):
        env.reset(seed=sd)
        step(2)
        rp = (scene.rack.data.root_pos_w - scene.env_origins)[0]
        q = scene.rack.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        dp = (scene.dish.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(rp[0]), float(rp[1]), yaw, float(dp[0]), float(dp[1])))
        tops.append(bool(scene.blue_top[0]))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (rack_x, rack_y, yaw, dish_x, dish_y):\n"
          f"{arr.round(3)}\n[smoke] blue_top: {tops}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: rack pose (xy + yaw) and dish position vary across seeded "
          f"resets (readback: dx={spread[0]:.3f} dy={spread[1]:.3f} "
          f"dyaw={spread[2]:.2f} ddish=({spread[3]:.3f},{spread[4]:.3f}))",
          spread[0] > 0.03 and spread[1] > 0.03 and spread[2] > 1.0
          and max(spread[3], spread[4]) > 0.15)
    check("randomization: BOTH stacking orders occur (blue-on-top "
          f"{tops.count(True)}/12, orange-on-top {tops.count(False)}/12)",
          tops.count(True) >= 2 and tops.count(False) >= 2)

    # =========================== 4. null policy =========================================
    env.reset(seed=3)
    step(300)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 300 idle steps",
          s <= 0.02 and not ok)

    # =========================== 5. seed strategy =======================================
    # rlbench/insert_onto_square_peg's SUCCESS state: a ring dropped DOWN over a
    # vertical post, encircling it, settled. Constructed fresh: drop the blue ring
    # onto the post from just above the credit deadband. Here: worthless.
    pos = rack_frame([0.0, 0.0, 0.130])
    teleport(scene.ring_blue, pos, q_flat(), settle_steps=180)
    report("seed-strategy")
    s, ok = judge()
    db = float(scene.rail_dist(scene.ring_blue)[0])
    check("seed strategy (the seed task's goal: ring dropped down over the post, "
          f"encircling it, settled; rail_dist={db:.3f}): score ~0, no success",
          db < 0.02 and blue_z_rack() < c.climb_z0 and s <= 0.03 and not ok)

    def find_seed_blue_top(start: int) -> int:
        """Probe seeded resets until the BLUE ring is on top (readback), so a pull
        on it lifts one ring's weight, not the whole stack."""
        for sd in range(start, start + 16):
            env.reset(seed=sd)
            step(2)
            if bool(scene.blue_top[0]):
                return sd
        raise AssertionError("no blue-on-top seed in 16 probes")

    # =========================== 6. captivity: straight-up pull =========================
    # QUASI-STATIC on purpose: a constant 2x-weight blast is a dynamic slam that can
    # ratchet the ring through the elbow and off the tip (the legitimate exit!) — the
    # retention claim is about steady pulls, so approach at ~0.06 m/s with the force
    # capped at ~1.8x weight and then PRESS statically once jammed.
    sd_top = find_seed_blue_top(4)
    step(120)
    z0 = blue_z_rack()
    # The pod rotates applied wrenches by the body's rotation-since-reset; once the
    # jammed ring pitches, a raw "world-up" force gains an along-arm component that
    # walks it off the tip. Pre-encode with R_ref * R_now^T so the force STAYS
    # world-up (same encoding solve.py uses for its servo).
    q_ref6 = scene.ring_blue.data.root_quat_w.clone()
    mg = scene_mod._RING_M * 9.81
    f_cap = 1.4  # ~1.8x ring weight
    d_max, z_max = 0.0, -1.0
    for i in range(700):
        vz = float(scene.ring_blue.data.root_lin_vel_w[0, 2])
        fz = min(max(mg + 2.0 * (0.06 - vz), 0.0), f_cap)
        f = torch.zeros(n, 3, device=device)
        f[:, 2] = fz
        f_enc = _qapply(_qmul(q_ref6, _qinv(scene.ring_blue.data.root_quat_w)), f)
        scene.ring_blue.set_external_force_and_torque(f_enc.view(n, 1, 3), zero_wrench,
                                                      env_ids=all_ids, is_global=True)
        step(1)
        d_max = max(d_max, float(scene.rail_dist(scene.ring_blue)[0]))
        z_max = max(z_max, blue_z_rack())
        if i % 60 == 0:
            judge()
    scene.ring_blue.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                  env_ids=all_ids)
    step(60)
    report("pull-up-jam")
    s, ok = judge()
    d_end = float(scene.rail_dist(scene.ring_blue)[0])
    check(f"captivity (up, blue-on-top seed {sd_top}): a quasi-static up-pull (velocity-"
          f"servoed, force capped at {f_cap} N ~ 1.8x weight, drag-pre-encoded to stay "
          f"world-up) CLIMBS the ring (z {z0:.3f} -> max {z_max:.3f}, "
          f"moved {z_max - z0:.3f} > 0.10) but it JAMS under the arm and never leaves "
          f"the rail (max rail_dist {d_max:.3f} < free_tol {c.free_tol}, "
          f"final rail_dist {d_end:.3f} < on_rail_tol, l_free latch "
          f"{float(scene.l_free[0]):.2f} == 0): partial credit only "
          f"(score={s:.3f} <= 0.35), no success",
          z_max - z0 > 0.10 and z_max < 0.38 and d_max < c.free_tol
          and d_end < c.on_rail_tol and float(scene.l_free[0]) == 0.0
          and s <= 0.35 and not ok)

    # =========================== 7. captivity: sideways yank ============================
    env.reset(seed=5)
    step(120)
    p0 = scene.ring_blue.data.root_pos_w[0].clone()
    side = _qapply(scene.rack.data.root_quat_w,
                   torch.tensor([0.0, 1.5, 0.0], device=device).expand(n, 3))
    d_max, _z = force_probe(scene.ring_blue, side, 360)
    moved = float((scene.ring_blue.data.root_pos_w[0] - p0).norm())
    report("yank-side")
    s, ok = judge()
    check("captivity (side): a 1.5 N horizontal yank displaces the ring "
          f"(moved {moved:.3f} m) but the post retains it "
          f"(max rail_dist {d_max:.3f} < free_tol {c.free_tol}): score ~0, no success",
          moved > 0.005 and d_max < c.free_tol and s <= 0.05 and not ok)

    # =========================== 8. left hanging on the arm =============================
    env.reset(seed=6)
    step(120)
    # threaded onto the arm at x ~ 0.08, bore's top edge resting on the rail
    pos = rack_frame([0.080, 0.0, scene_mod._Z_ARM
                      - (scene_mod._BORE_IN - scene_mod._RAIL_W / 2)])
    teleport(scene.ring_blue, pos, q_arm_threaded(), settle_steps=300)
    report("hang-on-arm")
    s, ok = judge()
    db = float(scene.rail_dist(scene.ring_blue)[0])
    check("partial escape: blue ring left DANGLING on the arm (settled on the rail, "
          f"rail_dist={db:.3f} < on_rail_tol, z={blue_z_rack():.3f}): partial latched "
          f"credit only (score={s:.3f} <= 0.47), no success",
          db < c.on_rail_tol and blue_z_rack() > 0.25 and 0.25 <= s <= 0.47 and not ok)

    # =========================== 9. floor near-miss beside the dish =====================
    env.reset(seed=7)
    step(120)
    pos = dish_frame([0.19, 0.0, scene_mod._RING_T / 2 + 0.02])
    teleport(scene.ring_blue, pos, q_flat(), settle_steps=180)
    report("floor-near-miss")
    s, ok = judge()
    dxy = float(scene._dish_local(scene.ring_blue.data.root_pos_w)[0, :2].norm())
    check("near-miss: blue ring flat + settled on the FLOOR beside the dish "
          f"(|xy-dish|={dxy:.3f} > tol {c.dish_xy_tol}): free-of-rail credit only "
          f"(score={s:.3f} <= 0.17), no success",
          dxy > c.dish_xy_tol and bool(scene.settled(scene.ring_blue)[0])
          and s <= 0.17 and not ok)

    # =========================== 10. perched on the dish rim ============================
    pos = dish_frame([0.130, 0.0, scene_mod._DISH_FLOOR_TOP + scene_mod._DISH_WALL_H
                      + scene_mod._RING_T / 2 + 0.004])
    teleport(scene.ring_blue, pos, q_flat(), settle_steps=240)
    report("rim-perch")
    s, ok = judge()
    pl = scene._dish_local(scene.ring_blue.data.root_pos_w)[0]
    out_of_box = float(pl[:2].norm()) > c.dish_xy_tol or float(pl[2]) > c.dish_z_hi
    check("near-miss: blue ring perched on / tipped off the dish WALL "
          f"(dish-local xy={float(pl[:2].norm()):.3f}, z={float(pl[2]):.3f}): outside "
          f"the flat-on-floor acceptance, no success",
          out_of_box and s <= 0.17 and not ok)

    # =========================== 11. wrong ring in the dish =============================
    env.reset(seed=8)
    step(120)
    pos = dish_frame([0.0, 0.0, scene_mod._DISH_FLOOR_TOP + scene_mod._RING_T / 2 + 0.02])
    teleport(scene.ring_orange, pos, q_flat(), settle_steps=180)
    report("wrong-ring")
    s, ok = judge()
    check("wrong ring: ORANGE laid flat in the dish (blue untouched on the rack) — "
          f"in_dish(orange)={bool(scene.in_dish(scene.ring_orange)[0])}, "
          f"decoy_clear={bool(scene.decoy_clear()[0])}: the wrong ring earns NOTHING "
          f"(score={s:.3f} ~ 0), no success",
          bool(scene.in_dish(scene.ring_orange)[0]) and not bool(scene.decoy_clear()[0])
          and s <= 0.02 and not ok)

    # =========================== 12. z-band + decoy attribution =========================
    pos = dish_frame([0.0, 0.0, scene_mod._DISH_FLOOR_TOP + scene_mod._RING_T
                      + scene_mod._RING_T / 2 + 0.02])
    teleport(scene.ring_blue, pos, q_flat(), settle_steps=180)
    report("stacked-in-dish")
    s, ok = judge()
    pl = scene._dish_local(scene.ring_blue.data.root_pos_w)[0]
    flat = float(scene._ring_up(scene.ring_blue)[0, 2].abs()) \
        >= math.cos(math.radians(c.flat_max_deg))
    check("z-band: blue dropped flat ON TOP of the in-dish orange — xy inside tol "
          f"({float(pl[:2].norm()):.3f}) and flat={flat} but center z={float(pl[2]):.3f} "
          f"ABOVE the flat-on-floor band (hi={c.dish_z_hi}): in_dish(blue) False AND "
          "decoy_clear False, no success",
          float(pl[:2].norm()) < c.dish_xy_tol and flat
          and float(pl[2]) > c.dish_z_hi and not bool(scene.in_dish(scene.ring_blue)[0])
          and not bool(scene.decoy_clear()[0]) and not ok)

    # =========================== 13-14. audit + no-NaN ==================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", all_finite())

    # =========================== save + verdict =========================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.hook_escape")
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

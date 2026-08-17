"""Smoke / rubric-REJECTION battery for PinLatchDumbwaiterScene (sim_gen task
`pick_and_lift_small_i350`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — drop the blue cube into the pinned car, pull the
latch pin by force against the spring-preloaded friction, let the spring hoist the
load into the penthouse — is the acceptance evidence that the rubric ACCEPTS a
correct outcome). Every teleport here is instrumentation that CONSTRUCTS a wrong
(or partial, or physically unreachable) state and asserts the rubric REJECTS it.
No probe in this battery ever reaches success(), and a final audit asserts exactly
that.

  1-2.  settle/no-NaN   — reset layout settles finite: car pressed up under the pin
                          (q ~ q_pin), pin through BOTH slots (not clear), cube and
                          decoy in their ground lanes on OPPOSITE sides, tower near
                          nominal; score ~0, no success;
  3-4.  randomization   — READBACK over 10 seeded resets: tower x/y/yaw all vary;
                          the cube's lane station varies, the cube/decoy SIDES swap,
                          and the pin's handle side flips; every reset sane;
  5.    null policy     — 240 idle steps -> score ~0, no success;
  6.    seed strategy   — the seed's own strategy (hoist the cube to the goal
                          HEIGHT) is worthless here: the cube released at penthouse
                          height beside the tower earns ~0 while airborne at the
                          goal height and simply falls to the ground — score ~0,
                          no success, nothing durable;
  7.    wrong object    — the RED DECOY riding the car at the top stop (car held by
                          the spring, decoy settled in the cargo box) is rejected:
                          no success, no credit — cube identity is load-bearing;
  8.    doomed order    — the empty release: with the pin removed and NOTHING
                          aboard, the spring physically hoists the empty car to the
                          top stop (the trap is real, and this run proves the pin
                          probe of check 10 non-vacuous); the cube dropped onto the
                          tower afterwards can no longer reach the car (the roof
                          seals the penthouse) -> score ~0, no success, and the
                          empty ride earned NO pin/rise credit;
  9.    loaded partial  — the cube seated in the PINNED car earns exactly the
                          loaded latch (~0.20), far from the cap, no success;
  10.   pin gate        — PHYSICAL probe: with the cube aboard, an EXTRA upward
                          push on the car of 2x the pin preload (on top of the
                          scene spring, re-applied every substep for 2 s) does NOT
                          get the car past the pin: q stays in the pinned band, the
                          pin stays in both slots, no rise credit — paired with
                          check 8's free ascent, the pin is what holds the car;
  11.   latched credit  — teleporting the cube away afterwards keeps the latched
                          loaded credit (~0.20) while cube_in_car drops — credit
                          never evaporates, and an abandoned state never succeeds;
  12.   settle gate     — the EXACT goal geometry (car at the top stop, cube in the
                          cargo box) built with an injected sliding velocity is in
                          the goal bands but NOT success while moving (sustained
                          stillness is load-bearing; sampled per substep, the cube
                          removed before it can settle);
  13.   rejection audit — success() was never True at ANY judged point;
  14.   final no-NaN    — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.pick_and_lift_small_i350.smoke --headless
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
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod

_qapply = scene_mod._qapply
_qinv = scene_mod._qinv

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.pin_latch_dumbwaiter")().build(num_envs=args.num_envs,
                                                          device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.35, -1.15, 0.85)) + o),
                                tuple(np.array((0.32, 0.00, 0.20)) + o),
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

    def tower_frame() -> tuple[torch.Tensor, torch.Tensor, float]:
        p = rel(scene.tower)
        q = scene.tower.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return p, q, yaw

    def tower_pt(loc) -> torch.Tensor:
        """Env-rel world position of a tower-local point."""
        p, q, _ = tower_frame()
        return p + quat_apply(q.view(1, 4), torch.tensor([loc], device=device))[0]

    def pin_ends_y() -> tuple[float, float]:
        """Tower-frame y of the rod tail and head (env 0)."""
        q = scene.pin.data.root_quat_w
        p = scene.pin.data.root_pos_w
        tail = torch.tensor([0.0, -c.pin_tail, 0.0], device=device).expand(n, 3)
        head = torch.tensor([0.0, c.pin_head, 0.0], device=device).expand(n, 3)
        y1 = float(scene._tower_local(p + _qapply(q, tail))[0, 1])
        y2 = float(scene._tower_local(p + _qapply(q, head))[0, 1])
        return y1, y2

    def cargo_loc(body) -> torch.Tensor:
        return scene._tower_local(body.data.root_pos_w)[0]

    def report(tag: str) -> None:
        s, ok = judge()
        q = float(scene.q_car()[0])
        cl = scene._car_local(scene.cube.data.root_pos_w)[0]
        y1, y2 = pin_ends_y()
        print(f"[smoke] {tag:16s} | q={q:.4f}/{c.stroke:.3f} "
              f"cube_car=({float(cl[0]):+.3f},{float(cl[1]):+.3f},{float(cl[2]):+.3f}) "
              f"in={bool(scene.cube_in_car()[0])} dec={bool(scene.decoy_in_car()[0])} "
              f"pin_y=({y1:+.3f},{y2:+.3f}) clear={bool(scene.pin_clear()[0])} "
              f"L={int(scene._loaded[0])}P={int(scene._pin_out[0])}R={int(scene._risen[0])} "
              f"set={bool(scene.settled()[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, xyz, quat=None, vel=None, angvel=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = (xyz if torch.is_tensor(xyz)
                      else torch.tensor([float(v) for v in xyz], device=device))
        if quat is None:
            st[:, 3] = 1.0
        elif torch.is_tensor(quat):
            st[:, 3:7] = quat
        else:
            st[:, 3:7] = torch.tensor([float(v) for v in quat], device=device)
        if vel is not None:
            st[:, 7:10] = torch.tensor([float(v) for v in vel], device=device)
        if angvel is not None:
            st[:, 10:13] = torch.tensor([float(v) for v in angvel], device=device)
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def wrench(body, f_w: torch.Tensor) -> None:
        """WORLD force at the CoM, expressed in the body's link frame (the house
        convention — is_global drops torques on this stack)."""
        q = body.data.root_link_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f_w).unsqueeze(1),
            torch.zeros(n, 1, 3, device=device), env_ids=all_ids)

    def layout_sane(tag: str) -> bool:
        """Reset honesty: tower near nominal, car pinned, pin through both slots,
        cube and decoy in their ground lanes on opposite sides."""
        tp, _q, yaw = tower_frame()
        qv = float(scene.q_car()[0])
        y1, y2 = pin_ends_y()
        cl, dl = cargo_loc(scene.cube), cargo_loc(scene.decoy)
        jf = c.tower_jitter + 0.008
        lanes_ok = True
        for loc in (cl, dl):
            lanes_ok = lanes_ok and (c.obj_x_lo - 0.02 < float(loc[0]) < c.obj_x_hi + 0.02
                                     and c.obj_y_lo - 0.02 < abs(float(loc[1])) < c.obj_y_hi + 0.02
                                     and float(loc[2]) < 0.06)
        ok = (abs(float(tp[0]) - c.fix_pos[0]) < jf
              and abs(float(tp[1]) - c.fix_pos[1]) < jf
              and abs(yaw) < math.radians(c.tower_yaw_deg) + 0.03
              and abs(qv - c.q_pin) < 0.010
              and not bool(scene.pin_clear()[0])
              and min(y1, y2) < -c.wall_out + 0.004
              and max(y1, y2) > c.wall_out - 0.004
              and lanes_ok
              and float(cl[1]) * float(dl[1]) < 0.0)
        if not ok:
            print(f"[smoke] layout sanity VIOLATION at {tag}: "
                  f"tower=({float(tp[0]):+.3f},{float(tp[1]):+.3f}) yaw={yaw:+.2f} "
                  f"q={qv:.4f} pin=({y1:+.3f},{y2:+.3f}) "
                  f"cube=({float(cl[0]):+.3f},{float(cl[1]):+.3f},{float(cl[2]):+.3f}) "
                  f"decoy=({float(dl[0]):+.3f},{float(dl[1]):+.3f})", flush=True)
        return ok

    bodies = (scene.tower, scene.car, scene.pin, scene.cube, scene.decoy)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("settle: all states finite; car pressed up under the pin, pin through both "
          "slots, cube+decoy in opposite ground lanes, tower near nominal",
          fin and layout_sane("reset"))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    sane = True
    for sd in range(21, 31):
        env.reset(seed=sd)
        step(30)
        sane = sane and layout_sane(f"seed {sd}")
        tp, _q, yaw = tower_frame()
        cl = cargo_loc(scene.cube)
        _y1, y2 = pin_ends_y()
        reads.append((float(tp[0]), float(tp[1]), yaw, float(cl[0]),
                      float(cl[1]), 1.0 if y2 > 0 else -1.0))
    arr = np.array(reads)
    print("[smoke] randomization readback (tower x, y, yaw | cube x, y | knob side):\n"
          f"{np.round(arr, 4)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: the tower fixture pose varies (spreads "
          f"x={spread[0]:.3f} y={spread[1]:.3f} yaw={spread[2]:.2f} rad) and every "
          "reset is sane",
          spread[0] > 0.02 and spread[1] > 0.02 and spread[2] > 0.15 and sane)
    sides = set(np.sign(arr[:, 4]))
    knobs = set(arr[:, 5])
    check("randomization: the cube's lane station varies (spread "
          f"x={spread[3]:.3f} m), the cube/decoy sides swap ({sorted(sides)}) and "
          f"the pin handle side flips ({sorted(knobs)})",
          spread[3] > 0.015 and len(sides) == 2 and len(knobs) == 2)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy: hoist the cube high ======================
    # The seed task's own strategy — raise the cube to the goal height — is worthless
    # here: released at penthouse height beside the tower, the cube earns ~0 while
    # AIRBORNE AT THE GOAL HEIGHT and then simply falls to the ground.
    env.reset(seed=41)
    step(30)
    place(scene.cube, tower_pt([0.30, 0.0, 0.36]), quat=tower_frame()[1],
          settle_steps=2)
    z_air = float(rel(scene.cube)[2])
    s_air, ok_air = judge()
    report("hoisted-airborne")
    step(120)
    report("hoisted-fell")
    s, ok = judge()
    z_end = float(rel(scene.cube)[2])
    check("seed strategy: the cube AT penthouse height beside the tower "
          f"(z={z_air:.3f} m) earns ~0 (score={s_air:.3f}), then falls to the ground "
          f"(z={z_end:.3f} m) — score ~0, no success, nothing durable",
          z_air > 0.30 and s_air <= 0.02 and not ok_air
          and z_end < 0.10 and s <= 0.02 and not ok)

    # =========================== 7. wrong object: the decoy rides up ========================
    # Car teleported to the top stop (the spring holds it there), the RED decoy
    # settled in its cargo box: the full delivered GEOMETRY with the wrong cube.
    env.reset(seed=51)
    step(30)
    _tp, q_t, _y = tower_frame()
    place(scene.car, tower_pt([0.0, 0.0, c.car_z0 + c.stroke - 0.002]), quat=q_t,
          settle_steps=30)
    place(scene.decoy,
          tower_pt([0.0, 0.0, c.car_z0 + c.stroke + c.car_floor_t + c.cube_s / 2 + 0.001]),
          quat=q_t, settle_steps=90)
    report("decoy-delivered")
    s, ok = judge()
    qv = float(scene.q_car()[0])
    check("wrong object: the RED decoy riding the car at the top stop "
          f"(q={qv:.3f}, decoy_in_car={bool(scene.decoy_in_car()[0])}) is rejected — "
          f"no success, score={s:.3f} ~0",
          qv >= c.stroke - c.top_tol and bool(scene.decoy_in_car()[0])
          and s <= 0.02 and not ok)

    # =========================== 8. doomed order: the empty release =========================
    # Pin removed with NOTHING aboard (instrumented removal): the spring hoists the
    # EMPTY car to the top — the trap is real, and this free ascent proves check 10's
    # pin probe non-vacuous. The cube dropped onto the tower afterwards cannot reach
    # the car (the roof seals the penthouse): no way back.
    env.reset(seed=61)
    step(30)
    place(scene.pin, (0.90, -0.60, 0.010), settle_steps=0)
    step(360)
    report("empty-risen")
    q_empty = float(scene.q_car()[0])
    s_mid, ok_mid = judge()
    _tp, q_t, _y = tower_frame()
    place(scene.cube, tower_pt([0.0, 0.0, c.top_z + c.cube_s / 2 + 0.09]), quat=q_t,
          settle_steps=200)
    report("cube-on-roof")
    s, ok = judge()
    check("doomed order: the empty release sends the car to the top stop "
          f"(q={q_empty:.3f} >= {c.stroke - 0.03:.3f}) with NO credit "
          f"(score={s_mid:.3f}), and the cube dropped onto the sealed tower "
          f"afterwards never reaches the car (in_car={bool(scene.cube_in_car()[0])}) "
          f"— score={s:.3f} ~0, no success",
          q_empty >= c.stroke - 0.03 and s_mid <= 0.02 and not ok_mid
          and not bool(scene.cube_in_car()[0]) and s <= 0.02 and not ok)

    # =========================== 9. loaded partial credit ===================================
    env.reset(seed=71)
    step(30)
    _tp, q_t, _y = tower_frame()
    place(scene.cube,
          tower_pt([0.005, 0.0, c.car_z0 + c.q_pin + c.car_floor_t + c.cube_s / 2 + 0.003]),
          quat=q_t, settle_steps=60)
    report("loaded")
    s_loaded, ok = judge()
    check("loaded partial: the cube seated in the PINNED car earns exactly the "
          f"loaded latch (score={s_loaded:.3f} ~ {c.w_loaded:.2f}), far from the "
          "cap, no success",
          bool(scene.cube_in_car()[0]) and bool(scene._loaded[0])
          and c.w_loaded - 0.02 <= s_loaded <= c.w_loaded + 0.02 and not ok)

    # =========================== 10. pin gate: the physical latch holds =====================
    # EXTRA upward push on the car of 2x the pin preload, ON TOP of the scene spring
    # (the scene re-applies its spring each post_step; this probe re-applies
    # spring + extra each substep so the spring is never masked). The car must NOT
    # pass the pin. Check 8's free ascent is the non-vacuity pair.
    f_extra = 2.0 * c.pin_preload
    q_max = 0.0
    for _i in range(240):
        vz = scene.car.data.root_lin_vel_w[:, 2]
        f_w = torch.zeros(n, 3, device=device)
        f_w[:, 2] = c.spring_f0 - c.spring_damp * vz + f_extra
        wrench(scene.car, f_w)
        env.step(no_action)
        q_max = max(q_max, float(scene.q_car()[0]))
    step(30)
    report("pin-pressed")
    s, ok = judge()
    check("pin gate: an EXTRA upward push of 2x the pin preload "
          f"({f_extra:.1f} N, 2 s) does not get the loaded car past the pin "
          f"(q_max={q_max:.4f} < {c.low_q_max:.3f}; pin still in both slots) — no "
          "rise credit, no success (check 8's free ascent proves the probe "
          "non-vacuous)",
          q_max < c.low_q_max and not bool(scene.pin_clear()[0])
          and not bool(scene._risen[0]) and not bool(scene._pin_out[0])
          and s <= c.w_loaded + 0.02 and not ok)

    # =========================== 11. latched credit survives ================================
    place(scene.cube, (0.85, 0.55, c.cube_s / 2 + 0.002), settle_steps=40)
    report("cube-removed")
    s_after, ok = judge()
    check("latched credit: teleporting the cube away keeps the latched loaded "
          f"credit ({s_loaded:.2f} -> {s_after:.2f}) while cube_in_car drops — and "
          "an abandoned state never succeeds",
          s_after >= c.w_loaded - 0.02 and s_after <= c.w_loaded + 0.02
          and not bool(scene.cube_in_car()[0]) and not ok)

    # =========================== 12. settle gate ============================================
    # The EXACT goal geometry — car at the top stop, cube in the cargo box — built
    # with an injected sliding velocity on the cube: in the goal bands but NOT
    # success while moving. The cube is removed before it can settle (this battery
    # must never reach success).
    env.reset(seed=101)
    step(30)
    _tp, q_t, _y = tower_frame()
    place(scene.car, tower_pt([0.0, 0.0, c.car_z0 + c.stroke - 0.002]), quat=q_t,
          settle_steps=30)
    cube_w = (scene.car.data.root_pos_w[0] - scene.env_origins[0]
              + _qapply(scene.car.data.root_quat_w,
                        torch.tensor([[0.0, 0.0, c.car_floor_t + c.cube_s / 2 + 0.001]],
                                     device=device))[0])
    place(scene.cube, cube_w, quat=scene.car.data.root_quat_w[0],
          vel=(0.30, 0.0, 0.0), settle_steps=0)
    gate_ok, v_peak = False, 0.0
    for _i in range(5):
        step(1)
        v_now = float(scene.cube.data.root_lin_vel_w[0].norm())
        v_peak = max(v_peak, v_now)
        _s_g, ok_g = judge()
        if (float(scene.q_car()[0]) >= c.stroke - c.top_tol
                and bool(scene.cube_in_car()[0]) and v_now > c.settle_speed
                and not bool(scene.settled()[0]) and not ok_g):
            gate_ok = True
            break
    report("goal-sliding")
    place(scene.cube, (0.85, -0.55, c.cube_s / 2 + 0.002), settle_steps=30)
    report("gate-cleared")
    _s, ok = judge()
    check("settle gate: the exact goal geometry with the cube still SLIDING "
          f"(peak {v_peak:.2f} m/s > settle_speed) is in the bands but NOT success "
          "— sustained stillness is load-bearing", gate_ok and not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.pin_latch_dumbwaiter")
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

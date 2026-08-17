"""Smoke / rubric-rejection battery for SwitchbackRampScene (sim_gen task
`libero_kitchen_scene4_put_the_black_bowl_on_top_of_the_cabinet_i409`) — NullRobot,
teleported/forced probe states (instrumentation, NOT a solution), RECORDED.

Battery:
  1-2. settle/no-NaN    — reset settles finite: basin on the ground-level porch in its
                          spawn band, decoy standing on the floor; score ~0, no latches,
                          no success;
  3. randomization      — READBACK across seeds: fixture root xy+yaw, basin spawn x and
                          basin yaw all move (all predicates are fixture-frame);
  4. null-policy-fails  — 240 idle steps -> score ~0, no success, basin still at ground
                          level;
  5-7. oracle x3 seeds  — force-driven switchback climb (same mechanism as solve)
                          -> success() and score 1.0; persists 240 further steps;
  8. incline rest       — basin RELEASED mid-flight A holds its place on the slope
                          (mu 0.70 > tan 11 deg) with the flight-A credit latched;
  9-10. monotonicity    — 0 (idle) < 0.15 (flight A) < 0.35 (pad crossed) < 0.60
                          (flight B, NO success yet: basin on the top pad) < 1.0
                          (on the roof = success); partials < 1.0;
  11. negative A (seed strategy) — basin teleported straight onto the roof with no
                          climb: on_roof() geometrically true but every latch is gated
                          on the previous one -> score 0, no success (the seed's
                          "carry the bowl to the top" plan);
  12. negative A2 (order bypass) — basin teleported onto the turning pad's inner row
                          (skipping flight A), then LEGITIMATELY force-driven up
                          flight B and onto the roof: pad/flight-B latches never fire
                          (gate chain), so even standing on the roof it scores 0;
  13. negative B (near-miss y) — legit climb through flight B (0.60), then basin set
                          on the roof edge at |y| > band -> no success, score stays 0.60;
  14. negative C (inverted) — basin UPSIDE-DOWN on the roof after a legit climb -> not
                          on_roof (upright test), score stays 0.60, no success;
  15. negative D (wrong object) — the maroon DECOY teleported onto the roof center ->
                          score 0, no success (the goal object is the basin);
  16. calibration       — roof-band drop sweep (in/out table, incl. a rest on the top
                          pad and a rest astride the rail-gap seam).

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene4_put_the_black_bowl_on_top_of_the_cabinet_i409.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--max_sec", type=float, default=1350.0)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit may mis-decode the driver version and silently reject RTX -> the
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
    from simgen_tasks.libero_kitchen_scene4_put_the_black_bowl_on_top_of_the_cabinet_i409 import (  # noqa: F401,E501
        scene as scene_mod,
    )
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
_wd = threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                             os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.switchback_ramp")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(1, 1, 3, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-1.25, 1.40, 1.05)) + o),
                                tuple(np.array((-0.02, 0.18, 0.12)) + o),
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

    def settle_until(pred, max_steps: int = 400, poll: int = 10,
                     min_steps: int = 30) -> bool:
        """Poll `pred` while stepping. ALWAYS steps at least `min_steps` first: right
        after a teleport all velocities are zero, so settled()-style predicates are
        vacuously true before physics has run (and post_step latches never fire)."""
        step(min_steps)
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def sc() -> float:
        return float(scene.score()[0])

    def bowl_fix():
        return scene.bowl_fix()[0]

    def report(tag: str) -> None:
        bf = bowl_fix()
        print(f"[smoke] {tag:16s} bowl_fix=({float(bf[0]):+.3f},{float(bf[1]):+.3f},"
              f"{float(bf[2]):+.3f}) "
              f"A={bool(scene._flightA_ever[0])} P={bool(scene._pad_ever[0])} "
              f"B={bool(scene._flightB_ever[0])} roof={bool(scene.on_roof()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"score={sc():.2f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402

    def fix_to_world(loc) -> torch.Tensor:
        p = quat_apply(scene.fixture.data.root_quat_w[0:1],
                       torch.tensor([loc], device=device))[0]
        return p + scene.fixture.data.root_pos_w[0]

    def fixq() -> tuple:
        q = scene.fixture.data.root_quat_w[0]
        return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    def tp(body, pos_w, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor([float(v) for v in pos_w], device=device)
        st[:, 3:7] = torch.tensor([float(v) for v in quat], device=device)
        body.write_root_state_to_sim(st, all_ids)

    def v_along(body, axis: torch.Tensor) -> float:
        return float((body.data.root_lin_vel_w[0] * axis).sum())

    def clear_forces() -> None:
        scene.bowl.set_external_force_and_torque(zero3, zero3)
        scene.decoy.set_external_force_and_torque(zero3, zero3)

    def drive(body, axis: torch.Tensor, fmag: float, vmax: float, done,
              max_steps: int) -> bool:
        """Pulsed push along world `axis`. On these pods the default external-force
        call applies the wrench in the body's CURRENT frame, so mode 0 pre-encodes
        per step with quat_apply_inverse(q_now, f); mode 1 = raw world fallback,
        toggled by a measured-progress stall probe."""
        mode = 0
        last_probe = float((body.data.root_pos_w[0] * axis).sum())
        for i in range(max_steps):
            if done():
                clear_forces()
                return True
            f = fmag if v_along(body, axis) < vmax else 0.0
            fw = f * axis
            if mode == 0:
                fw = quat_apply_inverse(body.data.root_quat_w[0:1], fw.view(1, 3))[0]
            clear_forces()
            body.set_external_force_and_torque(fw.view(1, 1, 3), zero3)
            step(1)
            if (i + 1) % 40 == 0:
                cur = float((body.data.root_pos_w[0] * axis).sum())
                if cur - last_probe < 0.001:
                    mode ^= 1
                last_probe = cur
        clear_forces()
        return done()

    def fix_axes() -> tuple[torch.Tensor, torch.Tensor]:
        fq = scene.fixture.data.root_quat_w[0:1]
        ex = quat_apply(fq, torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
        ey = quat_apply(fq, torch.tensor([[0.0, 1.0, 0.0]], device=device))[0]
        return ex, ey

    def climb_flight_a() -> bool:
        ex, _ = fix_axes()
        ok = drive(scene.bowl, ex, 9.0, 0.10,
                   lambda: float(bowl_fix()[0]) > 0.38, 3600)
        return ok and settle_until(lambda: bool(scene.settled()[0]), max_steps=200)

    def cross_pad() -> bool:
        _, ey = fix_axes()
        ok = drive(scene.bowl, -ey, 7.0, 0.10,
                   lambda: float(bowl_fix()[1]) < 0.26, 2400)
        return ok and settle_until(lambda: bool(scene.settled()[0]), max_steps=200)

    def climb_flight_b() -> bool:
        ex, _ = fix_axes()
        ok = drive(scene.bowl, -ex, 13.0, 0.10,
                   lambda: float(bowl_fix()[0]) < -0.10, 3600)
        return ok and settle_until(lambda: bool(scene.settled()[0]), max_steps=200)

    def enter_roof() -> bool:
        _, ey = fix_axes()
        ok = drive(scene.bowl, -ey, 7.0, 0.08,
                   lambda: float(bowl_fix()[1]) < 0.06, 2400)
        return ok and settle_until(lambda: bool(scene.on_roof()[0])
                                   and bool(scene.settled()[0]), max_steps=300)

    def no_latches() -> bool:
        return not bool(scene._flightA_ever[0] | scene._pad_ever[0]
                        | scene._flightB_ever[0])

    # =========================== 1-2. settle / no-NaN ============================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("settled")
    st0 = torch.cat([scene.fixture.data.root_state_w, scene.bowl.data.root_state_w,
                     scene.decoy.data.root_state_w], dim=-1)
    bf = bowl_fix()
    df = scene._to_fix(scene.decoy.data.root_pos_w)[0]
    check("settle: states finite, basin standing on the ground-level porch in its "
          "spawn band, decoy upright on the floor, everything settled",
          bool(torch.isfinite(st0).all())
          and c.bowl_x0[0] - 0.02 < float(bf[0]) < c.bowl_x0[1] + 0.02
          and 0.41 < float(bf[1]) < 0.48 and 0.0 < float(bf[2]) < 0.02
          and bool(scene.bowl_upright()[0])
          and 0.05 < float(df[2]) < 0.10 and bool(scene.settled()[0]))
    check("settle: score ~0 at reset, no success, no latches",
          sc() <= 0.005 and not bool(scene.success()[0]) and no_latches())

    # =========================== 3. randomization is real ========================================
    reads = []
    for s in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(s)
        env.reset()
        step(4)
        fp = (scene.fixture.data.root_pos_w[0] - scene.env_origins[0])
        fq = scene.fixture.data.root_quat_w[0]
        fyaw = 2.0 * math.atan2(float(fq[3]), float(fq[0]))
        ex = quat_apply(scene.bowl.data.root_quat_w[0:1],
                        torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
        byaw = math.atan2(float(ex[1]), float(ex[0]))
        reads.append((float(fp[0]), float(fp[1]), fyaw,
                      float(bowl_fix()[0]), byaw))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (fix_x, fix_y, fix_yaw, bowl_x, bowl_yaw):\n"
          f"{np.round(arr, 3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: fixture root xy+yaw, basin spawn x and basin yaw all move "
          "across seeds (READBACK from sim)",
          spread[0] > 0.02 and spread[1] > 0.02 and spread[2] > 0.05
          and spread[3] > 0.008 and spread[4] > 0.5)

    # =========================== 4. null policy fails ============================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    check("null policy: score ~0, no success, basin still at ground level after 240 "
          "idle steps",
          sc() <= 0.02 and not bool(scene.success()[0])
          and float(bowl_fix()[2]) < 0.03 and no_latches())

    # =========================== 5-7. oracle on 3 seeds ==========================================
    for s in (0, 1, 2):
        torch.manual_seed(s)
        env.reset()
        step(30)
        ok = (climb_flight_a() and cross_pad() and climb_flight_b()
              and enter_roof())
        ok = ok and settle_until(lambda: bool(scene.success()[0]), max_steps=300)
        step(240)  # persistence: success must not flicker off
        report(f"oracle{s}-final")
        check(f"oracle seed {s}: switchback climb (A -> turn -> B -> roof) = success, "
              f"score 1.0, persists 240 steps",
              ok and bool(scene.success()[0]) and sc() == 1.0)

    # =========================== 8-10. rubric monotonicity =======================================
    torch.manual_seed(41)
    env.reset()
    step(30)
    s0 = sc()
    # release MID-FLIGHT on the outer incline: the basin must hold its place there
    # (grippy route: mu 0.70 > tan 11 deg) with the flight-A credit already latched
    ex, ey = fix_axes()
    drive(scene.bowl, ex, 9.0, 0.10, lambda: float(bowl_fix()[0]) > 0.0, 3600)
    settle_until(lambda: bool(scene.settled()[0]), max_steps=200)
    x_rest0 = float(bowl_fix()[0])
    step(120)
    x_rest1 = float(bowl_fix()[0])
    s_mid = sc()
    report("mono-midflight")
    check("incline rest: basin released mid-flight A holds its place on the slope "
          "(no slide-back) and the flight-A credit is latched (0.15)",
          bool(scene._flightA_ever[0]) and 0.13 <= s_mid <= 0.17
          and abs(x_rest1 - x_rest0) < 0.01 and 0.03 < float(bowl_fix()[2]) < 0.12)
    # continue the route stage by stage, releasing between stages
    ok_a = climb_flight_a()
    s_a = sc()
    ok_p = cross_pad()
    s_p = sc()
    ok_b = climb_flight_b()
    s_b = sc()
    b_no_success = not bool(scene.success()[0])
    on_top_pad = float(bowl_fix()[1]) > c.roof_y_abs and float(bowl_fix()[2]) > 0.22
    report("mono-toppad")
    ok_r = enter_roof()
    settle_until(lambda: bool(scene.success()[0]), max_steps=300)
    s_r = sc()
    report("mono-final")
    print(f"[smoke] monotonicity ladder: idle={s0:.3f} flightA={s_a:.3f} "
          f"pad={s_p:.3f} flightB={s_b:.3f} roof={s_r:.3f}", flush=True)
    check("monotonicity: idle < flight A (0.15) < pad crossed (0.35) < flight B "
          "(0.60, basin on the top pad, NO success yet) < on the roof (1.0)",
          s0 <= 0.02 and ok_a and 0.13 <= s_a <= 0.17
          and ok_p and 0.33 <= s_p <= 0.37
          and ok_b and 0.58 <= s_b <= 0.62 and b_no_success and on_top_pad
          and ok_r and s_r == 1.0 and s0 < s_a < s_p < s_b < s_r)
    check("monotonicity: partial states score < 1.0", max(s0, s_a, s_p, s_b) < 1.0)

    # =========================== 11. negative A: seed strategy ===================================
    # The seed's plan — pick the bowl up and set it on top of the cabinet. Teleport is
    # the only way to even construct it (no grasp exists); every latch is gated on the
    # previous one, so this scores ZERO.
    torch.manual_seed(51)
    env.reset()
    step(30)
    tp(scene.bowl, fix_to_world([-0.10, 0.0, 0.245]), fixq())
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    step(60)
    report("seed-strategy")
    roof_geom = bool(scene.on_roof()[0])
    check("negative A (seed strategy): basin placed straight onto the roof -> "
          "on_roof() geometrically true but the latch chain never fired -> score 0, "
          "no success",
          roof_geom and no_latches() and not bool(scene.success()[0]) and sc() <= 0.005)

    # =========================== 12. negative A2: order bypass ===================================
    # Skip flight A: basin teleported onto the turning pad's inner row, then driven
    # LEGITIMATELY up flight B and onto the roof. The pad latch is gated on flight A,
    # flight B on the pad — so the whole tail of the route earns nothing.
    torch.manual_seed(52)
    env.reset()
    step(30)
    tp(scene.bowl, fix_to_world([0.40, 0.25, 0.132]), fixq())
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    pad_geom = bool(scene.on_pad_turned()[0])
    ok_b2 = climb_flight_b()
    report("bypass-toppad")
    ok_r2 = enter_roof()
    step(60)
    report("bypass-roof")
    check("negative A2 (order bypass): basin dropped onto the pad (skipping flight A) "
          "then legitimately driven up flight B onto the roof -> latches stay gated, "
          "score 0, no success even though the basin stands on the roof",
          pad_geom and ok_b2 and ok_r2 and bool(scene.on_roof()[0])
          and no_latches() and not bool(scene.success()[0]) and sc() <= 0.005)

    # =========================== 13. negative B: roof-y near-miss ================================
    torch.manual_seed(61)
    env.reset()
    step(30)
    ok_legit = climb_flight_a() and cross_pad() and climb_flight_b()
    tp(scene.bowl, fix_to_world([-0.12, 0.135, 0.245]), fixq())
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    step(60)
    bf = bowl_fix()
    report("y-near-miss")
    check("negative B (near-miss y): legit climb through flight B, basin set astride "
          "the rail-gap seam at |y| > band -> not on_roof, score stays 0.60, no success",
          ok_legit and abs(float(bf[1])) > c.roof_y_abs
          and not bool(scene.on_roof()[0]) and not bool(scene.success()[0])
          and 0.58 <= sc() <= 0.62)

    # =========================== 14. negative C: inverted ========================================
    tp(scene.bowl, fix_to_world([-0.10, 0.0, 0.315]), _quat_mul_fix_x180(fixq()))
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    step(60)
    report("inverted")
    check("negative C (inverted): basin UPSIDE-DOWN on the roof after a legit climb "
          "-> not on_roof (upright test), score stays 0.60, no success",
          not bool(scene.bowl_upright()[0]) and not bool(scene.on_roof()[0])
          and not bool(scene.success()[0]) and 0.58 <= sc() <= 0.62)

    # =========================== 15. negative D: wrong object ====================================
    torch.manual_seed(71)
    env.reset()
    step(30)
    tp(scene.decoy, fix_to_world([0.0, 0.0, 0.32]), fixq())
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    step(60)
    df = scene._to_fix(scene.decoy.data.root_pos_w)[0]
    report("wrong-object")
    check("negative D (wrong object): the maroon decoy standing on the roof center "
          "-> score 0, no success (the goal object is the basin, still on the porch)",
          0.28 < float(df[2]) < 0.33 and float(bowl_fix()[2]) < 0.03
          and no_latches() and not bool(scene.success()[0]) and sc() <= 0.005)

    # =========================== 16. calibration =================================================
    print("[smoke] CALIBRATION: basin drop (fixture frame) -> on_roof() readback "
          "(roof band |x|<0.21 |y|<0.12 z[0.228,0.250])", flush=True)
    cal = []
    for (x_off, y_off, z_off) in ((-0.12, 0.0, 0.245), (0.12, 0.0, 0.245),
                                  (-0.12, 0.135, 0.245), (-0.14, 0.235, 0.245)):
        torch.manual_seed(101)
        env.reset()
        step(20)
        tp(scene.bowl, fix_to_world([x_off, y_off, z_off]), fixq())
        settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
        step(60)
        bf = bowl_fix()
        cal.append((x_off, y_off, float(bf[0]), float(bf[1]), bool(scene.on_roof()[0])))
        print(f"[smoke]   drop=({x_off:+.2f},{y_off:+.2f}) rest_fix="
              f"({float(bf[0]):+.3f},{float(bf[1]):+.3f},{float(bf[2]):+.3f}) "
              f"on_roof={cal[-1][4]}", flush=True)
    check("calibration: basin resting inside the roof band counts; astride the "
          "rail-gap seam and on the top pad do not",
          cal[0][4] and cal[1][4] and not cal[2][4] and not cal[3][4])

    # =========================== save + verdict ==================================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.switchback_ramp")
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
    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


def _quat_mul_fix_x180(fq: tuple) -> tuple:
    """fixture_yaw_quat * rot_x(180): an upside-down basin aligned with the fixture."""
    w, x, y, z = fq
    # (w,x,y,z) * (0,1,0,0)
    return (-x, w, z, -y)


if __name__ == "__main__":
    main()

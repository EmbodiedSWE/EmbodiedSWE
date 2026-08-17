"""Smoke / rubric-rejection battery for WedgeLiftScene (sim_gen task
`libero_kitchen_scene4_put_the_black_bowl_on_top_of_the_cabinet_i308`) — NullRobot,
teleported/forced probe states (instrumentation, NOT a solution), RECORDED.

Battery:
  1-2. settle/no-NaN    — reset settles finite: bowl aboard the lowered elevator in the
                          well, ram on the rail in its spawn band; score ~0, no latches,
                          no success;
  3. randomization      — READBACK across seeds: fixture root xy+yaw, ram spawn x and
                          bowl yaw all move (all predicates are fixture-frame);
  4. null-policy-fails  — 240 idle steps -> score ~0, no success, bowl still sunk;
  5-7. oracle x3 seeds  — force-driven ram-home + bowl-slide (same mechanism as solve)
                          -> success() and score 1.0; persists 240 further steps;
  8-9. monotonicity     — 0 (idle) < 0.15 (part-lift, ram released mid-travel: the
                          slope back-drives and the platform sinks back — latched
                          credit survives) < 0.40 (raised) < 1.0 (landed = success);
                          partials < 1.0;
  10. negative A (seed strategy) — bowl teleported straight onto the roof landing area
                          with the elevator still down: landed() is geometrically true
                          but the landed latch is GATED on the raised latch -> score 0,
                          no success (the seed's "carry the bowl to the top" plan);
  11. negative A2 (no retroactive credit) — afterwards the EMPTY elevator is rammed to
                          the top: raised requires the bowl aboard, so still score 0,
                          no success;
  12. negative B (near-miss landing y) — legit raise, then the bowl set on the roof at
                          |y| > band -> no landed latch, score stays 0.40, no success;
  13. negative C (inverted) — legit raise, bowl UPSIDE-DOWN on the landing area -> not
                          landed (upright test), score stays 0.40, no success;
  14. negative D (pull the ram OUT) — force-pull toward +x: the platform only drops a
                          few mm to the rail; nothing lifts, score ~0, no success;
  15. calibration       — landing-band drop sweep (in/out table for the roof band).

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene4_put_the_black_bowl_on_top_of_the_cabinet_i308.smoke --headless
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
    from simgen_tasks.libero_kitchen_scene4_put_the_black_bowl_on_top_of_the_cabinet_i308 import (  # noqa: F401,E501
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
    env = ENVS.get("simgen.wedge_lift")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-1.05, -1.05, 0.95)) + o),
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
        return scene._to_fix(scene.bowl.data.root_pos_w)[0]

    def report(tag: str) -> None:
        rx = float(scene.ram_fix_x()[0])
        ez = float(scene.elev_top_z()[0])
        bf = bowl_fix()
        print(f"[smoke] {tag:16s} ram_x={rx:+.3f} elev_top={ez:.3f} "
              f"bowl_fix=({bf[0]:+.3f},{bf[1]:+.3f},{bf[2]:+.3f}) "
              f"aboard={bool(scene.bowl_aboard()[0])} raised={bool(scene.raised()[0])} "
              f"landed={bool(scene.landed()[0])} settled={bool(scene.settled()[0])} "
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
        scene.ram.set_external_force_and_torque(zero3, zero3)
        scene.bowl.set_external_force_and_torque(zero3, zero3)

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
            scene.ram.set_external_force_and_torque(zero3, zero3)
            scene.bowl.set_external_force_and_torque(zero3, zero3)
            body.set_external_force_and_torque(fw.view(1, 1, 3), zero3)
            step(1)
            if (i + 1) % 40 == 0:
                cur = float((body.data.root_pos_w[0] * axis).sum())
                if cur - last_probe < 0.001:
                    mode ^= 1
                last_probe = cur
        clear_forces()
        return done()

    def lane_axis() -> torch.Tensor:
        return quat_apply(scene.fixture.data.root_quat_w[0:1],
                          torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]

    seat_x = c.ram_in_x + 0.012

    def ram_home() -> bool:
        ex = lane_axis()
        ok = drive(scene.ram, -ex, 25.0, 0.10,
                   lambda: bool(scene.raised()[0])
                   and float(scene.ram_fix_x()[0]) < seat_x
                   and abs(v_along(scene.ram, -ex)) < 0.02, 3600)
        return ok and settle_until(lambda: bool(scene.raised()[0])
                                   and bool(scene.settled()[0]))

    def slide_bowl(x_stop: float = -0.14) -> bool:
        ex = lane_axis()
        ok = drive(scene.bowl, -ex, 1.5, 0.08,
                   lambda: float(bowl_fix()[0]) < x_stop, 2400)
        return ok and settle_until(lambda: bool(scene.landed()[0])
                                   and bool(scene.settled()[0]))

    def no_latches() -> bool:
        return not bool(scene._partlift_ever[0] | scene._raised_ever[0]
                        | scene._landed_ever[0])

    # =========================== 1-2. settle / no-NaN ============================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("settled")
    st0 = torch.cat([scene.fixture.data.root_state_w, scene.ram.data.root_state_w,
                     scene.elevator.data.root_state_w, scene.bowl.data.root_state_w],
                    dim=-1)
    bf = bowl_fix()
    check("settle: states finite, bowl aboard the LOWERED elevator down the well, ram "
          "on the rail in its spawn band, everything settled",
          bool(torch.isfinite(st0).all()) and bool(scene.bowl_aboard()[0])
          and 0.16 < float(scene.elev_top_z()[0]) < 0.18
          and 0.19 < float(bf[2]) < 0.21
          and c.ram_x0[0] - 0.02 < float(scene.ram_fix_x()[0]) < c.ram_x0[1] + 0.02
          and bool(scene.settled()[0]))
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
                      float(scene.ram_fix_x()[0]), byaw))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (fix_x, fix_y, fix_yaw, ram_x, bowl_yaw):\n"
          f"{np.round(arr, 3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: fixture root xy+yaw, ram spawn x and bowl yaw all move "
          "across seeds (READBACK from sim)",
          spread[0] > 0.02 and spread[1] > 0.02 and spread[2] > 0.05
          and spread[3] > 0.008 and spread[4] > 0.5)

    # =========================== 4. null policy fails ============================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    check("null policy: score ~0, no success, bowl still sunk in the well after 240 "
          "idle steps",
          sc() <= 0.02 and not bool(scene.success()[0])
          and float(bowl_fix()[2]) < 0.22 and no_latches())

    # =========================== 5-7. oracle on 3 seeds ==========================================
    for s in (0, 1, 2):
        torch.manual_seed(s)
        env.reset()
        step(30)
        ok1 = ram_home()
        report(f"oracle{s}-raised")
        ok2 = slide_bowl()
        ok = ok1 and ok2 and settle_until(lambda: bool(scene.success()[0]),
                                          max_steps=300)
        step(240)  # persistence: success must not flicker off
        report(f"oracle{s}-final")
        check(f"oracle seed {s}: ram home -> lift -> slide onto the roof = success, "
              f"score 1.0, persists 240 steps",
              ok and bool(scene.success()[0]) and sc() == 1.0)

    # =========================== 8-9. rubric monotonicity ========================================
    torch.manual_seed(41)
    env.reset()
    step(30)
    s0 = sc()
    # part-lift only: drive the ram until the platform has risen ~55 mm, then RELEASE —
    # on the slope the elevator back-drives the ram and sinks back, but credit is latched.
    ex = lane_axis()
    drive(scene.ram, -ex, 25.0, 0.10,
          lambda: float(scene.elev_top_z()[0]) > 0.235, 2400)
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    s_part = sc()
    part_latched = bool(scene._partlift_ever[0])
    report("mono-partlift")
    # The chaotic ejection can yaw the ram while its rear end crosses the unguided
    # gap between the split tunnel walls; a yawed rear corner then catches the
    # guide_back end face on re-entry (diagonal jam). Re-square it by teleport
    # (instrumentation — the latched 0.15 must survive) before resuming.
    tp(scene.ram, fix_to_world([0.25, 0.0, c.ram_root_z + 0.002]), fixq())
    settle_until(lambda: bool(scene.settled()[0]), max_steps=200)
    ok_raise = ram_home()
    s_raised = sc()
    raised_no_success = not bool(scene.success()[0])
    report("mono-raised")
    slide_bowl()
    settle_until(lambda: bool(scene.success()[0]), max_steps=300)
    s_land = sc()
    report("mono-final")
    print(f"[smoke] monotonicity ladder: idle={s0:.3f} partlift={s_part:.3f} "
          f"raised={s_raised:.3f} landed={s_land:.3f}", flush=True)
    check("monotonicity: idle < part-lift (0.15, released mid-travel, latched credit "
          "survives the sink-back) < raised (0.40, NO success yet) < landed (1.0)",
          s0 <= 0.02 and 0.13 <= s_part <= 0.17 and part_latched and ok_raise
          and 0.38 <= s_raised <= 0.42 and raised_no_success and s_land == 1.0
          and s0 < s_part < s_raised < s_land)
    check("monotonicity: partial states score < 1.0", max(s0, s_part, s_raised) < 1.0)

    # =========================== 10. negative A: seed strategy ===================================
    # The seed's plan — pick the bowl up and set it on top of the cabinet. Teleport is
    # the only way to even construct it (no grasp exists); the landed latch is gated on
    # having ridden the lift, so this scores ZERO.
    torch.manual_seed(51)
    env.reset()
    step(30)
    tp(scene.bowl, fix_to_world([-0.15, 0.0, 0.345]), fixq())
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    step(60)
    report("seed-strategy")
    landed_geom = bool(scene.landed()[0])
    check("negative A (seed strategy): bowl placed straight onto the roof landing area "
          "-> landed() geometrically true but latch gated on the lift -> score 0, "
          "no success",
          landed_geom and not bool(scene._landed_ever[0])
          and not bool(scene.success()[0]) and sc() <= 0.005)

    # =========================== 11. negative A2: no retroactive credit ==========================
    # Same episode: now ram the EMPTY elevator to the top. raised credit requires the
    # bowl aboard, so nothing fires and the roof bowl still earns nothing.
    ram_home()
    step(60)
    report("empty-raise")
    check("negative A2 (no retroactive credit): EMPTY elevator rammed to the top with "
          "the bowl already on the roof -> still score 0, no success",
          bool(scene.raised()[0]) and not bool(scene._raised_ever[0])
          and not bool(scene._landed_ever[0]) and not bool(scene.success()[0])
          and sc() <= 0.005)

    # =========================== 12. negative B: landing y near-miss =============================
    torch.manual_seed(61)
    env.reset()
    step(30)
    ram_home()
    tp(scene.bowl, fix_to_world([-0.15, 0.155, 0.345]), fixq())
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    step(60)
    bf = bowl_fix()
    report("y-near-miss")
    check("negative B (near-miss landing y): legit raise, bowl set on the roof at "
          "|y| > band -> no landed latch, score stays 0.40, no success",
          abs(float(bf[1])) > c.land_y_abs and not bool(scene.landed()[0])
          and not bool(scene._landed_ever[0]) and not bool(scene.success()[0])
          and 0.38 <= sc() <= 0.42)

    # =========================== 13. negative C: inverted ========================================
    tp(scene.bowl, fix_to_world([-0.15, 0.0, 0.350]), _quat_mul_fix_x180(fixq()))
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    step(60)
    report("inverted")
    check("negative C (inverted): bowl UPSIDE-DOWN on the landing area -> not landed "
          "(upright test), score stays 0.40, no success",
          not bool(scene.bowl_upright()[0]) and not bool(scene.landed()[0])
          and not bool(scene._landed_ever[0]) and not bool(scene.success()[0])
          and 0.38 <= sc() <= 0.42)

    # =========================== 14. negative D: pull the ram OUT ================================
    torch.manual_seed(71)
    env.reset()
    step(30)
    ex = lane_axis()
    drive(scene.ram, ex, 15.0, 0.15,
          lambda: float(scene.ram_fix_x()[0]) > 0.34, 1200)
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    report("ram-out")
    check("negative D (pull the ram OUT): the platform only drops a few mm to the "
          "rail — nothing lifts, score ~0, no success",
          float(scene.ram_fix_x()[0]) > 0.32
          and float(scene.elev_top_z()[0]) < 0.175
          and sc() <= 0.005 and not bool(scene.success()[0]) and no_latches())

    # =========================== 15. calibration =================================================
    print("[smoke] CALIBRATION: bowl drop (fixture frame) -> landed() readback "
          "(roof landing band x[-0.30,-0.06] |y|<0.12)", flush=True)
    cal = []
    for (x_off, y_off) in ((-0.15, 0.0), (-0.27, 0.0), (-0.15, 0.155), (0.16, 0.0)):
        torch.manual_seed(101)
        env.reset()
        step(20)
        tp(scene.bowl, fix_to_world([x_off, y_off, 0.345]), fixq())
        settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
        step(60)
        bf = bowl_fix()
        cal.append((x_off, y_off, float(bf[0]), float(bf[1]), bool(scene.landed()[0])))
        print(f"[smoke]   drop=({x_off:+.2f},{y_off:+.2f}) rest_fix="
              f"({float(bf[0]):+.3f},{float(bf[1]):+.3f},{float(bf[2]):+.3f}) "
              f"landed={cal[-1][4]}", flush=True)
    check("calibration: bowl resting inside the roof landing band counts landed; "
          "y-outside and the front (paddle-side) roof do not",
          cal[0][4] and cal[1][4] and not cal[2][4] and not cal[3][4])

    # =========================== save + verdict ==================================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.wedge_lift")
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


def _quat_mul_fix_x180(fq: tuple) -> tuple:
    """fixture_yaw_quat * rot_x(180): an upside-down bowl aligned with the fixture."""
    w, x, y, z = fq
    # (w,x,y,z) * (0,1,0,0)
    return (-x, w, z, -y)


if __name__ == "__main__":
    main()

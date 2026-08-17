"""Smoke / rubric-rejection battery for BowlAirlockScene (sim_gen task
`libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet_i297`) — NullRobot,
teleported/forced probe states (instrumentation, NOT a solution), RECORDED.

Battery (15 checks):
  1-2. settle/no-NaN    — reset settles finite: bowl+plate on the pedestal, carriage at
                          one END of its travel; score ~0, no latches, no success;
  3. randomization      — READBACK across 8 seeds: fixture root xy+yaw move (all
                          predicates are fixture-frame), bowl xy+yaw move, and the
                          carriage start takes BOTH ends of the travel;
  4. null-policy-fails  — 240 idle steps -> score ~0, no success, bowl on the pedestal;
  5-7. oracle x3 seeds  — the full force pipeline of solve.py (drive the gate to LOAD,
                          stage the bowl on the sill, push it through the front window,
                          gravity feed, drive the gate to DISPENSE, gravity dispense)
                          -> success() and score 1.0; persists 240 further steps;
  8-9. monotonicity     — 0 (idle) < 0.15 (staged) < 0.40 (+chambered) < 0.70
                          (+delivered but TIPPED in the case: near-miss, NO success)
                          < 1.0 (upright on the deck = success); partials < 1.0;
  10. negative A (seed-analog) — the seed's strategy, "carry the bowl to the top and
                          set it down", is geometrically impossible: bowl released
                          from above the case lands on the ROOF -> no delivered latch,
                          no credit, no success;
  11. negative B (interlock, front) — carriage at DISPENSE seals the front window: the
                          same push that chambers the bowl in the oracle drives it
                          ~5 cm across the sill (actuator moved) and STALLS it against
                          the front panel -> chamber latch never fires;
  12. negative C (interlock, mid-travel) — carriage mid-travel: both windows are
                          part-open slits (4 cm each, on opposite sides) < bowl 9.8 cm;
                          a sustained push on a chambered bowl cannot dispense it ->
                          delivered latch never fires;
  13. negative D (wrong object) — the white PLATE settled on the case deck earns
                          nothing: no latch, no success, score ~0;
  14. negative E (wrong place / gate never cycled) — bowl at rest in the chamber with
                          the gate at LOAD scores only the chamber latch (0.25 here:
                          teleported straight in, so the sill latch never fired), and
                          no success;
  15. calibration       — upright/tipped/deck-position drop table for the placed band
                          (published in/out table).

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet_i297.smoke --headless
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
    from simgen_tasks.libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet_i297 import (  # noqa: F401,E501
        scene as scene_mod,
    )
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.bowl_airlock")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-1.30, -1.00, 0.95)) + o),
                                tuple(np.array((-0.15, 0.0, 0.30)) + o),
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
        s = float(scene.carriage_s()[0])
        print(f"[smoke] {tag:16s} bowl_fix=({bf[0]:+.3f},{bf[1]:+.3f},{bf[2]:+.3f}) "
              f"carriage_s={s:+.3f} load={bool(scene.carriage_at_load()[0])} "
              f"disp={bool(scene.carriage_at_dispense()[0])} "
              f"staged={bool(scene._staged_ever[0])} "
              f"chambered={bool(scene._chambered_ever[0])} "
              f"delivered={bool(scene._delivered_ever[0])} "
              f"placed={bool(scene.bowl_placed()[0])} settled={bool(scene.settled()[0])} "
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
        st[:, 0:3] = pos_w if torch.is_tensor(pos_w) else torch.tensor(
            [float(v) for v in pos_w], device=device)
        st[:, 3:7] = torch.tensor([float(v) for v in quat], device=device)
        body.write_root_state_to_sim(st, all_ids)

    def v_along(body, axis: torch.Tensor) -> float:
        return float((body.data.root_lin_vel_w[0] * axis).sum())

    def clear_forces() -> None:
        scene.carriage.set_external_force_and_torque(zero3, zero3)
        scene.bowl.set_external_force_and_torque(zero3, zero3)

    def drive(body, axis: torch.Tensor, fmag: float, vmax: float, done,
              max_steps: int) -> bool:
        """Pulsed push along world `axis` with the pod-dependent frame probe
        (toggle raw-world <-> body-frame encoding if progress stalls)."""
        mode = 0
        last_probe = float((body.data.root_pos_w[0] * axis).sum())
        for i in range(max_steps):
            if done():
                clear_forces()
                return True
            f = fmag if v_along(body, axis) < vmax else 0.0
            fw = f * axis
            if mode == 1:
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

    def fix_axis(v) -> torch.Tensor:
        return quat_apply(scene.fixture.data.root_quat_w[0:1],
                          torch.tensor([v], device=device))[0]

    def gate_to(side: float) -> None:
        """Teleport the carriage to an end of its travel (probe state)."""
        tp(scene.carriage, fix_to_world([-0.12, side, c.carriage_ride_z + 0.002]), fixq())
        settle_until(lambda: bool(scene.settled()[0]), max_steps=200)

    def stage_bowl(x_fix: float = -0.255) -> None:
        """Release the bowl just above the sill (transport, as in solve.py)."""
        tp(scene.bowl, fix_to_world([x_fix, 0.0, 0.3495]), fixq())
        settle_until(lambda: bool(scene.settled()[0]), max_steps=300)

    def push_bowl(done, fmag: float = 2.2, max_steps: int = 900) -> bool:
        return drive(scene.bowl, fix_axis([1.0, 0.0, 0.0]), fmag, 0.08, done, max_steps)

    # =========================== 1-2. settle / no-NaN ============================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("settled")
    st0 = torch.cat([scene.fixture.data.root_state_w, scene.carriage.data.root_state_w,
                     scene.bowl.data.root_state_w, scene.plate.data.root_state_w], dim=-1)
    bf = bowl_fix()
    s = float(scene.carriage_s()[0])
    check("settle: states finite, bowl on the pedestal, carriage parked at one END of "
          "its travel (fixture frame), everything settled",
          bool(torch.isfinite(st0).all()) and -0.62 < float(bf[0]) < -0.48
          and 0.13 < float(bf[2]) < 0.17 and 0.090 < abs(s) < 0.112
          and bool(scene.settled()[0]))
    check("settle: score ~0 at reset, no success, no latches",
          sc() <= 0.005 and not bool(scene.success()[0])
          and not bool(scene._staged_ever[0]) and not bool(scene._chambered_ever[0])
          and not bool(scene._delivered_ever[0]))

    # =========================== 3. randomization is real ========================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(4)
        fp = (scene.fixture.data.root_pos_w[0] - scene.env_origins[0])
        fq = scene.fixture.data.root_quat_w[0]
        fyaw = 2.0 * math.atan2(float(fq[3]), float(fq[0]))
        bl = bowl_fix()
        ex = quat_apply(scene.bowl.data.root_quat_w[0:1],
                        torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
        byaw = math.atan2(float(ex[1]), float(ex[0]))
        reads.append((float(fp[0]), float(fp[1]), fyaw, float(bl[0]), float(bl[1]),
                      byaw, float(scene.carriage_s()[0])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (fix_x, fix_y, fix_yaw, bowl_x, bowl_y, "
          f"bowl_yaw, carriage_s):\n{np.round(arr, 3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: fixture root xy+yaw, bowl xy+yaw all move across seeds, and "
          "the carriage start takes BOTH travel ends (READBACK from sim)",
          spread[0] > 0.02 and spread[1] > 0.02 and spread[2] > 0.05
          and spread[3] > 0.03 and spread[4] > 0.015 and spread[5] > 0.5
          and arr[:, 6].min() < -0.09 and arr[:, 6].max() > 0.09)

    # =========================== 4. null policy fails ============================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    check("null policy: score ~0, no success, bowl still on the pedestal after 240 "
          "idle steps",
          sc() <= 0.02 and not bool(scene.success()[0])
          and 0.13 < float(bowl_fix()[2]) < 0.17)

    # =========================== 5-7. oracle on 3 seeds ==========================================
    # The FULL force pipeline of solve.py — the single teleport is the sill staging
    # transport; the airlock itself is exercised by force + gravity only.
    ey = [0.0, 1.0, 0.0]
    for sd in (0, 1, 2):
        torch.manual_seed(sd)
        env.reset()
        step(30)
        drive(scene.carriage, -fix_axis(ey), 8.0, 0.10,
              lambda: bool(scene.carriage_at_load()[0])
              and abs(v_along(scene.carriage, fix_axis(ey))) < 0.02, 1800)
        settle_until(lambda: bool(scene.carriage_at_load()[0]) and bool(scene.settled()[0]))
        stage_bowl()
        push_bowl(lambda: float(bowl_fix()[0]) > -0.145, max_steps=2400)
        okc = settle_until(lambda: bool(scene.bowl_chambered()[0])
                           and bool(scene.settled()[0]), max_steps=960)
        report(f"oracle{sd}-chamber")
        drive(scene.carriage, fix_axis(ey), 8.0, 0.10,
              lambda: bool(scene.carriage_at_dispense()[0])
              and abs(v_along(scene.carriage, fix_axis(ey))) < 0.02, 1800)
        ok = okc and settle_until(lambda: bool(scene.success()[0]), max_steps=1200)
        step(240)  # persistence: success must not flicker off
        report(f"oracle{sd}-final")
        check(f"oracle seed {sd}: gate->LOAD, stage, push through front window, gravity "
              f"feed, gate->DISPENSE, gravity dispense = success, score 1.0, persists "
              f"240 steps", ok and bool(scene.success()[0]) and sc() == 1.0)

    # =========================== 8-9. rubric monotonicity ========================================
    torch.manual_seed(41)
    env.reset()
    step(30)
    s0 = sc()
    gate_to(-c.s_end)  # LOAD (probe)
    stage_bowl()
    s_stage = sc()
    report("mono-staged")
    tp(scene.bowl, fix_to_world([-0.10, 0.0, 0.305]), fixq())  # into the chamber
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    s_cham = sc()
    report("mono-chambered")
    # TIPPED into the case (lying on its side): delivered latch, near-miss, NO success
    tp(scene.bowl, fix_to_world([0.065, 0.0, 0.285]), (0.7071068, 0.0, 0.7071068, 0.0))
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    s_tip = sc()
    tipped_no_success = (not bool(scene.success()[0])
                         and not bool(scene.bowl_placed()[0])
                         and bool(scene._delivered_ever[0]))
    report("mono-tipped")
    tp(scene.bowl, fix_to_world([0.065, 0.0, 0.270]), fixq())  # upright on the deck
    settle_until(lambda: bool(scene.success()[0]), max_steps=400)
    s_up = sc()
    report("mono-final")
    print(f"[smoke] monotonicity ladder: idle={s0:.3f} staged={s_stage:.3f} "
          f"chambered={s_cham:.3f} tipped={s_tip:.3f} upright={s_up:.3f}", flush=True)
    check("monotonicity: idle < staged (0.15) < +chambered (0.40) < +delivered-tipped "
          "(0.70, in the case but NOT placed, NO success) < upright on deck (1.0)",
          s0 <= 0.02 and 0.14 <= s_stage <= 0.16 and 0.39 <= s_cham <= 0.41
          and 0.69 <= s_tip <= 0.71 and tipped_no_success and s_up == 1.0
          and s0 < s_stage < s_cham < s_tip < s_up)
    check("monotonicity: partial states score < 1.0", max(s0, s_stage, s_cham, s_tip) < 1.0)

    # =========================== 10. negative A: the seed's strategy is impossible ===============
    torch.manual_seed(51)
    env.reset()
    step(30)
    tp(scene.bowl, fix_to_world([0.065, 0.0, 0.55]), fixq())  # released above the case
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    step(60)
    bf = bowl_fix()
    report("case-drop")
    check("negative A (seed-analog): bowl released from above the case lands on the "
          "ROOF (z readback) — no delivered latch, no credit, no success",
          float(bf[2]) > 0.40 and not bool(scene._delivered_ever[0])
          and not bool(scene.success()[0]) and sc() <= 0.005)

    # =========================== 11. negative B: interlock (front) ===============================
    torch.manual_seed(61)
    env.reset()
    step(30)
    gate_to(c.s_end)  # DISPENSE: the front window is SEALED by the front panel
    stage_bowl(-0.30)  # stage further back so the stall leaves a real displacement
    x_before = float(bowl_fix()[0])
    push_bowl(lambda: float(bowl_fix()[0]) > -0.16, max_steps=600)  # must NOT happen
    settle_until(lambda: bool(scene.settled()[0]), max_steps=200)
    x_after = float(bowl_fix()[0])
    report("interlock-front")
    check("negative B (interlock front): with the carriage at DISPENSE the oracle's own "
          "push drives the bowl across the sill (>=3 cm, actuator moved) and STALLS it "
          "against the front panel — chamber latch never fires",
          x_after - x_before >= 0.03 and x_after < -0.20
          and not bool(scene._chambered_ever[0]) and not bool(scene.success()[0]))

    # =========================== 12. negative C: interlock (mid-travel) ==========================
    torch.manual_seed(71)
    env.reset()
    step(30)
    gate_to(0.0)  # mid-travel: both windows are 4 cm part-open slits (opposite sides)
    tp(scene.bowl, fix_to_world([-0.10, 0.0, 0.305]), fixq())  # chambered
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    push_bowl(lambda: bool(scene.bowl_delivered()[0]), max_steps=360)  # must NOT happen
    settle_until(lambda: bool(scene.settled()[0]), max_steps=200)
    bf = bowl_fix()
    report("interlock-mid")
    check("negative C (interlock mid-travel): a sustained push on the chambered bowl "
          "cannot dispense it through the part-open windows — bowl held in the chamber "
          "(pressed at the inner panel), delivered latch never fires",
          bool(scene._chambered_ever[0]) and float(bf[0]) < -0.04
          and not bool(scene._delivered_ever[0]) and not bool(scene.success()[0]))

    # =========================== 13. negative D: wrong object ====================================
    torch.manual_seed(81)
    env.reset()
    step(30)
    tp(scene.plate, fix_to_world([0.065, 0.0, 0.26]), fixq())  # plate onto the deck
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    step(60)
    pl = scene._to_fix(scene.plate.data.root_pos_w)[0]
    report("wrong-object")
    check("negative D (wrong object): the white PLATE settled on the case deck earns "
          "nothing — no latch, no success, score ~0 (bowl still on the pedestal)",
          0.0 < float(pl[0]) < 0.17 and float(pl[2]) < 0.30 and sc() <= 0.005
          and not bool(scene._delivered_ever[0]) and not bool(scene.success()[0]))

    # =========================== 14. negative E: gate never cycled ===============================
    torch.manual_seed(91)
    env.reset()
    step(30)
    gate_to(-c.s_end)  # LOAD (inner window sealed)
    tp(scene.bowl, fix_to_world([-0.10, 0.0, 0.305]), fixq())  # straight into the chamber
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    step(120)
    report("gate-never-cycled")
    check("negative E (wrong place): bowl at rest in the chamber with the gate never "
          "cycled scores only the chamber latch (0.25: teleported straight in, sill "
          "latch never fired) — no delivered latch, no success",
          bool(scene.bowl_chambered()[0]) and 0.24 <= sc() <= 0.26
          and not bool(scene._delivered_ever[0]) and not bool(scene.success()[0]))

    # =========================== 15. calibration =================================================
    print("[smoke] CALIBRATION: bowl drop (fixture frame) -> placed band readback",
          flush=True)
    cal = []
    for x_fix, y_fix, quat, tag in (
            (0.045, 0.0, fixq(), "deck-center-upright"),
            (0.10, 0.08, fixq(), "deck-offset-upright"),
            (0.065, 0.0, (0.7071068, 0.0, 0.7071068, 0.0), "deck-tipped"),
            (0.065, 0.0, "ROOF", "above-roof-upright")):
        torch.manual_seed(101)
        env.reset()
        step(20)
        if quat == "ROOF":
            tp(scene.bowl, fix_to_world([x_fix, y_fix, 0.55]), fixq())
        else:
            tp(scene.bowl, fix_to_world([x_fix, y_fix, 0.285]), quat)
        settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
        step(60)
        bf = bowl_fix()
        placed = bool(scene.bowl_placed()[0])
        cal.append(placed)
        print(f"[smoke]   {tag:22s} rest_fix=({float(bf[0]):+.3f},{float(bf[1]):+.3f},"
              f"{float(bf[2]):+.3f}) placed={placed}", flush=True)
    check("calibration: upright rests on the deck count placed; a tipped bowl and a "
          "bowl on the roof do not",
          cal[0] and cal[1] and not cal[2] and not cal[3])

    # =========================== save + verdict ==================================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.bowl_airlock")
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

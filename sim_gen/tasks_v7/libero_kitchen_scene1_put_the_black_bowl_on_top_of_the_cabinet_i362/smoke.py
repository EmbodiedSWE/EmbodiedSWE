"""Smoke / rubric-rejection battery for GondolaWheelScene (sim_gen task
`libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet_i362`) — NullRobot,
teleported/forced probe states (instrumentation, NOT a solution), RECORDED.

Battery:
  1-2. settle/no-NaN    — reset settles finite: wheel fallen onto its 30-deg loading
                          stop, gondola hanging level, bowl + plate on the staging
                          table; score ~0, no latches, no success;
  3. randomization      — READBACK across seeds: sampled wheel start angle, bowl/plate
                          staging-slot assignment + xy jitter + free yaw all move;
  4. null-policy-fails  — 240 idle steps -> score ~0, no success, bowl still on the table;
  5. negative SEED      — the seed task's strategy (lower the bowl onto the cabinet top
                          from above) lands on the ROOF slab -> not on the deck, no
                          success, score ~0;
  6. negative ORDER-A   — bowl hand-delivered through the window onto the deck with the
                          wheel still at the bottom stop: geometrically ON the deck and
                          settled, but no ride provenance -> success False, score ~0;
  7. negative ORDER-B (flagship) — EMPTY ride: wheel cranked to its over-center park
                          with no bowl aboard, then the bowl hand-delivered through the
                          window. The end state is IDENTICAL to the success end state
                          (wheel parked, bowl resting on the deck) but the bowl never
                          rode -> success False, score ~0 (order-chained latches);
  8. negative NEAR-MISS — bowl settled in the window mouth (y outside the deck band) ->
                          bowl_on_deck False, no success;
  9. negative INVERTED  — bowl upside-down on the deck -> upright check fails, no success;
  10. negative WRONG-OBJ — the white plate delivered into the gallery instead -> no
                          success, no credit, bowl untouched on the table;
  11-13. oracle x3 seeds — full ride (drop-load into the tray, torque-servo crank over
                          top-dead-center, gravity park, regulated push through the
                          window) -> success() and score 1.0; persists 240 steps;
  14. monotonicity      — one ladder run: idle (~0) < loaded (0.15) < mid-arc rode
                          (0.30, sampled during the crank) < parked+hoisted (0.55,
                          bowl still in the tray = near-miss with NO success) < served
                          (1.0); latched credit never drops;
  15. partials < 1.0.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet_i362.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=10)
parser.add_argument("--max_frames", type=int, default=450)
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
    from simgen_tasks.libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet_i362 import (  # noqa: F401,E501
        scene as scene_mod,
    )
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.gondola_wheel")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(1, 1, 3, device=device)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-1.30, -1.05, 1.00)) + o),
                                tuple(np.array((-0.10, 0.12, 0.35)) + o),
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

    def angd() -> float:
        return math.degrees(float(scene.wheel_angle()[0]))

    def bowl_env() -> torch.Tensor:
        return (scene.bowl.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        bl = scene._to_gondola_frame(scene.bowl.data.root_pos_w)[0]
        bw = bowl_env()
        print(f"[smoke] {tag:16s} ang={angd():+7.2f} "
              f"bowl_loc=({bl[0]:+.3f},{bl[1]:+.3f},{bl[2]:+.3f}) "
              f"bowl_w=({bw[0]:+.3f},{bw[1]:+.3f},{bw[2]:+.3f}) "
              f"aboard={bool(scene.bowl_in_tray()[0])} parked={bool(scene.wheel_parked()[0])} "
              f"deck={bool(scene.bowl_on_deck()[0])} settled={bool(scene.settled()[0])} "
              f"L={int(scene._loaded_ever[0])}{int(scene._rode_ever[0])}"
              f"{int(scene._hoisted_ever[0])}{int(scene._delivered_ever[0])} "
              f"score={sc():.2f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def tp(body, pos_env, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor([float(v) for v in pos_env], device=device)
        st[:, 3:7] = torch.tensor([float(v) for v in quat], device=device)
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def clear_wrenches() -> None:
        scene.disc.set_external_force_and_torque(zero3, zero3)
        scene.bowl.set_external_force_and_torque(zero3, zero3)

    def settle_to_stop() -> bool:
        return settle_until(lambda: bool(scene.wheel_still()[0])
                            and angd() < c.travel_lo_deg + 1.5,
                            max_steps=600, min_steps=60)

    def load_bowl() -> bool:
        """Drop the bowl into the tray from just above the aboard band (transport
        teleport releases OUTSIDE every credit region; gravity does the loading)."""
        drop_loc = torch.tensor([0.0, -0.01, 0.056], device=device)
        assert float(drop_loc[2]) > c.tray_z_hi
        pos = (scene.gondola.data.root_pos_w[0]
               + quat_apply(scene.gondola.data.root_quat_w[0:1], drop_loc.view(1, 3))[0]
               - scene.env_origins[0])
        tp(scene.bowl, [float(v) for v in pos])
        return settle_until(
            lambda: bool(scene._loaded_ever[0]) and bool(scene.bowl_in_tray()[0])
            and float(scene.bowl.data.root_lin_vel_w.norm(dim=-1)[0]) < c.settle_speed,
            max_steps=360, min_steps=20)

    def crank_to_park(m_load: float, sample_at_deg: float | None = None):
        """Torque-servo crank (gravity feedforward + clamped P velocity term), cut at
        182 deg; gravity parks the wheel on the 186-deg over-center stop. Optionally
        samples score() the first time the wheel passes `sample_at_deg`."""
        s_mid = None
        steps = 0
        while angd() < 182.0 and steps < 2000:
            th = float(scene.wheel_angle()[0])
            w = float(scene.disc.data.root_ang_vel_w[0, 1])
            w_des = 0.7 if th < math.radians(150.0) else 0.35
            tau_ff = m_load * 9.81 * c.arm_r * math.sin(th)
            tau = max(-0.8, min(2.6, tau_ff + 0.6 * (w_des - w)))
            t3 = torch.tensor([0.0, tau, 0.0], device=device).view(1, 1, 3)
            scene.disc.set_external_force_and_torque(zero3, t3)
            step(1)
            steps += 1
            if sample_at_deg is not None and s_mid is None and angd() >= sample_at_deg:
                s_mid = sc()
        clear_wrenches()
        ok = settle_until(lambda: bool(scene.wheel_parked()[0])
                          and bool(scene.wheel_still()[0]), max_steps=600, min_steps=30)
        return ok, s_mid

    def push_through() -> None:
        """Regulated horizontal push off the tray's open front edge through the
        window (velocity-regulated +y, weak x-centering; body-frame wrench)."""
        steps = 0
        while float(bowl_env()[1]) < 0.30 and steps < 900:
            bp = bowl_env()
            vy = float(scene.bowl.data.root_lin_vel_w[0, 1])
            fy = max(0.0, min(5.0, 1.9 + 6.0 * (0.12 - vy)))
            fx = max(-1.0, min(1.0, 4.0 * (0.02 - float(bp[0]))))
            fw = torch.tensor([fx, fy, 0.0], device=device).view(1, 3)
            fb = quat_apply_inverse(scene.bowl.data.root_quat_w[0:1], fw)
            scene.bowl.set_external_force_and_torque(fb.view(1, 1, 3), zero3)
            step(1)
            steps += 1
        clear_wrenches()

    def no_latches() -> bool:
        return (not bool(scene._loaded_ever[0]) and not bool(scene._rode_ever[0])
                and not bool(scene._hoisted_ever[0])
                and not bool(scene._delivered_ever[0]))

    # =========================== 1-2. settle / no-NaN ============================================
    torch.manual_seed(11)
    env.reset()
    ok_stop = settle_to_stop()
    report("reset")
    st0 = torch.cat([scene.disc.data.root_state_w, scene.gondola.data.root_state_w,
                     scene.bowl.data.root_state_w, scene.plate.data.root_state_w], dim=-1)
    bw = bowl_env()
    pw = (scene.plate.data.root_pos_w - scene.env_origins)[0]
    check("settle: states finite, wheel on the 30-deg loading stop, bowl and plate "
          "resting on the staging table, everything settled",
          bool(torch.isfinite(st0).all()) and ok_stop
          and float(bw[0]) < -0.30 and 0.16 < float(bw[2]) < 0.20
          and float(pw[0]) < -0.30 and 0.15 < float(pw[2]) < 0.18
          and bool(scene.settled()[0]))
    check("settle: score ~0 at reset, no success, no latches",
          sc() <= 0.005 and not bool(scene.success()[0]) and no_latches())

    # =========================== 3. randomization is real ========================================
    reads = []
    for s in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(s)
        env.reset()
        step(4)
        bwr = bowl_env()
        pwr = (scene.plate.data.root_pos_w - scene.env_origins)[0]
        ex = quat_apply(scene.bowl.data.root_quat_w[0:1],
                        torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
        byaw = math.atan2(float(ex[1]), float(ex[0]))
        reads.append((angd(), float(bwr[0]), float(bwr[1]), byaw,
                      float(pwr[0]), float(pwr[1])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (wheel_deg, bowl_x, bowl_y, bowl_yaw, "
          f"plate_x, plate_y):\n{np.round(arr, 3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: wheel start angle, bowl slot/jitter xy + yaw, plate slot/"
          "jitter xy all move across seeds (READBACK from sim)",
          spread[0] > 2.0 and spread[1] > 0.02 and spread[2] > 0.02
          and spread[3] > 0.5 and spread[4] > 0.02 and spread[5] > 0.02)

    # =========================== 4. null policy fails ============================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    check("null policy: score ~0, no success, bowl still on the staging table after "
          "240 idle steps",
          sc() <= 0.02 and not bool(scene.success()[0])
          and float(bowl_env()[0]) < -0.30 and 0.16 < float(bowl_env()[2]) < 0.20)

    # =========================== 5. negative SEED: drop from above ================================
    # The seed task's strategy — carry the bowl over the cabinet and lower it onto the
    # top — hits the ROOF slab (top z=0.67). It rests on the roof, never on the deck.
    torch.manual_seed(41)
    env.reset()
    settle_to_stop()
    tp(scene.bowl, (0.02, 0.294, 0.75))
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    report("roof-drop")
    check("negative SEED: bowl lowered from above lands on the ROOF (z>0.65), not the "
          "deck -> no success, score ~0",
          float(bowl_env()[2]) > 0.65 and not bool(scene.bowl_on_deck()[0])
          and not bool(scene.success()[0]) and sc() <= 0.005)

    # =========================== 6. negative ORDER-A: hand-delivery ==============================
    # Bowl carried straight through the window onto the deck, wheel untouched at the
    # bottom stop: geometrically a delivered bowl, but no ride provenance.
    torch.manual_seed(51)
    env.reset()
    settle_to_stop()
    tp(scene.bowl, (0.02, 0.30, 0.585))
    ok_rest = settle_until(lambda: bool(scene.bowl_on_deck()[0])
                           and bool(scene.settled()[0]), max_steps=300)
    report("hand-delivery")
    check("negative ORDER-A: bowl hand-delivered through the window rests ON the deck "
          "(geometric predicate true) yet success False, score ~0 (no ride latches)",
          ok_rest and not bool(scene.success()[0]) and sc() <= 0.005 and no_latches())

    # =========================== 7. negative ORDER-B: empty ride (flagship) ======================
    # Wheel genuinely cranked to its over-center park — but EMPTY — then the bowl
    # hand-delivered. End state is identical to success (wheel parked + bowl on deck);
    # the order-chained latches still reject it.
    torch.manual_seed(61)
    env.reset()
    settle_to_stop()
    ok_park, _ = crank_to_park(m_load=c.gondola_mass)
    report("empty-parked")
    empty_no_credit = ok_park and no_latches() and sc() <= 0.005
    tp(scene.bowl, (0.02, 0.30, 0.585))
    ok_rest = settle_until(lambda: bool(scene.bowl_on_deck()[0])
                           and bool(scene.settled()[0]), max_steps=300)
    step(120)
    report("empty+delivered")
    check("negative ORDER-B (flagship): empty ride to the park + hand-delivered bowl "
          "= end state IDENTICAL to success (wheel parked, bowl settled on deck) but "
          "success False, score ~0",
          empty_no_credit and ok_rest and bool(scene.wheel_parked()[0])
          and not bool(scene.success()[0]) and sc() <= 0.005)

    # =========================== 8. negative NEAR-MISS: window mouth =============================
    # Bowl settled in the window mouth: on the deck surface but y short of the deck
    # band (deck_y_lo) — a real settled near-miss outside the y tolerance.
    torch.manual_seed(71)
    env.reset()
    settle_to_stop()
    tp(scene.bowl, (0.02, 0.21, 0.585))
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    report("window-mouth")
    bwm = bowl_env()
    check("negative NEAR-MISS: bowl settled in the window mouth (y < deck_y_lo) -> "
          "bowl_on_deck False, no success",
          float(bwm[1]) < c.deck_y_lo and 0.53 < float(bwm[2]) < 0.57
          and not bool(scene.bowl_on_deck()[0]) and not bool(scene.success()[0]))

    # =========================== 9. negative INVERTED ============================================
    torch.manual_seed(81)
    env.reset()
    settle_to_stop()
    tp(scene.bowl, (0.02, 0.30, 0.585), quat=(0.0, 1.0, 0.0, 0.0))
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    report("inverted")
    check("negative INVERTED: bowl upside-down inside the gallery -> upright check "
          "fails, not on-deck, no success",
          not bool(scene.bowl_upright()[0]) and not bool(scene.bowl_on_deck()[0])
          and not bool(scene.success()[0]))

    # =========================== 10. negative WRONG-OBJECT =======================================
    torch.manual_seed(91)
    env.reset()
    settle_to_stop()
    tp(scene.plate, (0.02, 0.30, 0.56))
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    report("wrong-object")
    check("negative WRONG-OBJECT: white plate delivered into the gallery instead -> "
          "no success, no credit, bowl still on the table",
          not bool(scene.success()[0]) and sc() <= 0.005
          and float(bowl_env()[0]) < -0.30 and no_latches())

    # =========================== 11-13. oracle on 3 seeds ========================================
    for s in (0, 1, 2):
        torch.manual_seed(s)
        env.reset()
        ok0 = settle_to_stop()
        ok1 = load_bowl()
        ok2, _ = crank_to_park(m_load=c.gondola_mass + c.bowl_mass)
        hoisted = bool(scene._hoisted_ever[0])
        push_through()
        ok3 = settle_until(lambda: bool(scene.success()[0]), max_steps=600)
        step(240)  # persistence: success must not flicker off
        report(f"oracle{s}-final")
        check(f"oracle seed {s}: load -> crank over-center -> gravity park -> push "
              f"through window = success, score 1.0, persists 240 steps",
              ok0 and ok1 and ok2 and hoisted and ok3
              and bool(scene.success()[0]) and sc() == 1.0)

    # =========================== 14-15. rubric monotonicity ======================================
    torch.manual_seed(101)
    env.reset()
    settle_to_stop()
    s0 = sc()
    load_bowl()
    s_load = sc()
    report("mono-loaded")
    ok_park, s_mid = crank_to_park(m_load=c.gondola_mass + c.bowl_mass,
                                   sample_at_deg=130.0)
    s_park = sc()
    park_no_success = (ok_park and bool(scene.bowl_in_tray()[0])
                       and not bool(scene.success()[0]))
    report("mono-parked")
    push_through()
    settle_until(lambda: bool(scene.success()[0]), max_steps=600)
    s_final = sc()
    report("mono-final")
    s_mid = -1.0 if s_mid is None else s_mid
    print(f"[smoke] monotonicity ladder: idle={s0:.3f} loaded={s_load:.3f} "
          f"mid-arc={s_mid:.3f} parked={s_park:.3f} served={s_final:.3f}", flush=True)
    check("monotonicity: idle (~0) < loaded (0.15) < mid-arc rode (0.30) < parked+"
          "hoisted (0.55, bowl still aboard = near-miss, NO success) < served (1.0); "
          "latched credit never drops",
          s0 <= 0.005 and 0.14 <= s_load <= 0.16 and 0.29 <= s_mid <= 0.31
          and 0.54 <= s_park <= 0.56 and park_no_success and s_final == 1.0
          and s0 < s_load < s_mid < s_park < s_final)
    check("monotonicity: partial states score < 1.0", max(s0, s_load, s_mid, s_park) < 1.0)

    # =========================== save + verdict ==================================================
    if frames:
        farr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=farr, env="simgen.gondola_wheel")
        print(f"[smoke] saved {farr.shape} -> {args.out}", flush=True)
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


if __name__ == "__main__":
    main()

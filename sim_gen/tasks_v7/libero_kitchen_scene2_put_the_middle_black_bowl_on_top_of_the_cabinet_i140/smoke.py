"""Smoke / rubric-rejection battery for RampHutchScene (sim_gen task
`libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet_i140`) —
NullRobot, teleported/forced probe states (instrumentation, NOT a solution), RECORDED.

Battery:
  1-2. settle/no-NaN    — reset settles finite: three bowls in a floor row, tray empty,
                          score ~0, no latches, no success;
  3. randomization      — READBACK across seeds: fixture root xy+yaw move (all
                          predicates fixture-frame), the TARGET BODY INDEX varies
                          (bowl->slot permutation is real), row position and bowl
                          yaw move;
  4. null-policy-fails  — 240 idle steps -> score ~0, no success;
  5-7. oracle x3 seeds  — teleport the MIDDLE bowl to the LOWER ramp (zero credit),
                          force-push it up the ramp, through the doorway, into the
                          tray -> success() and score 1.0; persists 240 steps;
  8. occupied forfeit   — after oracle 2, a DECOY teleported into the tray flips
                          success OFF (score falls back to the 0.75 latch sum);
  9-10. monotonicity    — 0 (idle) < 0.15 (parked hi-ramp, NO success) < 0.40 (the
                          instant the doorway latch fires) < 1.0 (seated = success);
                          partials < 1.0;
  11. negative (seed strategy) — the seed's plan, "release the bowl above the
                          cabinet": it lands on the ROOF (readback z) -> score 0;
  12. negative (teleport bypass) — target placed directly INSIDE the tray, settled:
                          target_seated() is live-TRUE yet the seated latch stays
                          false (no doorway passage) -> score 0, no success;
  13. negative (inverted) — bowl parked UPSIDE-DOWN inside the hi-ramp spatial band
                          (position readback-verified in band) -> hi latch refuses,
                          score 0;
  14. credit permanence — parked hi (0.15), then bowl removed to the floor -> score
                          stays 0.15, no success;
  15. sill one-way      — from the seated state, a sustained speed-capped pull toward
                          the doorway: the bowl travels ~10 cm (probe moved — not
                          vacuous), presses the sill's inner step and CANNOT exit;
                          still in the tray, success recovers after release.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet_i140.smoke --headless
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
    from simgen_tasks.libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet_i140 import (  # noqa: F401,E501
        scene as scene_mod,
    )
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ramp_hutch")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(1, 1, 3, device=device)
    theta = math.atan(c.ramp_slope)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-1.55, -1.15, 0.85)) + o),
                                tuple(np.array((-0.30, 0.0, 0.20)) + o),
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

    def tgt():
        return scene.bowls[int(scene.target_idx[0])]

    def x_fix() -> float:
        return float(scene.target_fix()[0, 0])

    def report(tag: str) -> None:
        loc = scene.target_fix()[0]
        print(f"[smoke] {tag:16s} tgt={int(scene.target_idx[0])} "
              f"tgt_fix=({loc[0]:+.3f},{loc[1]:+.3f},{loc[2]:+.3f}) "
              f"hi={bool(scene._hi_ever[0])} doored={bool(scene._doored_ever[0])} "
              f"seated={bool(scene._seated_ever[0])} "
              f"decoy_in={bool(scene.decoy_in_tray()[0])} "
              f"settled={bool(scene.settled()[0])} score={sc():.3f} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul  # noqa: E402

    def fix_to_world(loc) -> torch.Tensor:
        p = quat_apply(scene.fixture.data.root_quat_w[0:1],
                       torch.tensor([loc], device=device))[0]
        return p + scene.fixture.data.root_pos_w[0]

    def tp(body, pos_w, quat_w) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat_w
        body.write_root_state_to_sim(st, all_ids)

    def ramp_quat(inverted: bool = False) -> torch.Tensor:
        """Tilt-matched bowl orientation on the ramp (fixture yaw ∘ pitch -theta),
        optionally rolled upside-down."""
        q_fix = scene.fixture.data.root_quat_w[0:1]
        q_pitch = torch.tensor([[math.cos(theta / 2), 0.0, -math.sin(theta / 2), 0.0]],
                               device=device)
        q = quat_mul(q_fix, q_pitch)
        if inverted:
            q_roll = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device)  # pi about x
            q = quat_mul(q, q_roll)
        return q[0]

    def park_on_ramp(body, x: float, inverted: bool = False) -> None:
        z_surf = (x - c.ramp_foot_x) * c.ramp_slope
        z = z_surf + (c.bowl_h / 2) / math.cos(theta) + 0.004
        tp(body, fix_to_world([x, 0.0, z]), ramp_quat(inverted))
        settle_until(lambda: bool(scene.settled()[0]), max_steps=300)

    def upq() -> torch.Tensor:
        return torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    def v_along(body, axis: torch.Tensor) -> float:
        return float((body.data.root_lin_vel_w[0] * axis).sum())

    def clear_forces() -> None:
        for b in scene.bowls:
            b.set_external_force_and_torque(zero3, zero3)

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

    def up_tan() -> torch.Tensor:
        return quat_apply(scene.fixture.data.root_quat_w[0:1],
                          torch.tensor([[math.cos(theta), 0.0, math.sin(theta)]],
                                       device=device))[0]

    def fix_x_axis() -> torch.Tensor:
        return quat_apply(scene.fixture.data.root_quat_w[0:1],
                          torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]

    def climb(done_x: float, max_steps: int = 3600) -> bool:
        return drive(tgt(), up_tan(), 2.5, 0.08, lambda: x_fix() > done_x, max_steps)

    # =========================== 1-2. settle / no-NaN ============================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("settled")
    st0 = torch.cat([scene.fixture.data.root_state_w, scene.plate.data.root_state_w]
                    + [b.data.root_state_w for b in scene.bowls], dim=-1)
    bp = scene._bowl_pos_fix()[0]  # (3, 3)
    check("settle: states finite, three bowls at rest in a floor row (fixture frame), "
          "tray empty, everything settled",
          bool(torch.isfinite(st0).all())
          and bool((bp[:, 2] > 0.015).all()) and bool((bp[:, 2] < 0.06).all())
          and bool((bp[:, 1] < -0.25).all())
          and not bool(scene.bowls_in_tray()[0].any())
          and bool(scene.settled()[0]))
    check("settle: score ~0 at reset, no success, no latches",
          sc() <= 0.005 and not bool(scene.success()[0])
          and not bool(scene._hi_ever[0]) and not bool(scene._doored_ever[0])
          and not bool(scene._seated_ever[0]))

    # =========================== 3. randomization is real ========================================
    reads = []
    for s in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(s)
        env.reset()
        step(4)
        fp = (scene.fixture.data.root_pos_w[0] - scene.env_origins[0])
        fq = scene.fixture.data.root_quat_w[0]
        fyaw = 2.0 * math.atan2(float(fq[3]), float(fq[0]))
        tl = scene.target_fix()[0]
        ex = quat_apply(scene.bowls[0].data.root_quat_w[0:1],
                        torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
        byaw = math.atan2(float(ex[1]), float(ex[0]))
        reads.append((float(fp[0]), float(fp[1]), fyaw,
                      float(scene.target_idx[0]), float(tl[0]), byaw))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (fix_x, fix_y, fix_yaw, target_idx, "
          f"tgt_row_x, bowl0_yaw):\n{np.round(arr, 3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    n_tgt = len(set(int(v) for v in arr[:, 3]))
    check("randomization: fixture root xy+yaw move, TARGET BODY INDEX varies (bowl->slot "
          "permutation), target row position and bowl yaw move (READBACK from sim)",
          spread[0] > 0.02 and spread[1] > 0.02 and spread[2] > 0.05
          and n_tgt >= 2 and spread[4] > 0.02 and spread[5] > 0.5)

    # =========================== 4. null policy fails ============================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    check("null policy: score ~0, no success after 240 idle steps",
          sc() <= 0.02 and not bool(scene.success()[0]))

    # =========================== 5-7. oracle on 3 seeds ==========================================
    for s in (0, 1, 2):
        torch.manual_seed(s)
        env.reset()
        step(30)
        park_on_ramp(tgt(), -0.55)
        zero_after_tp = sc() <= 1e-6  # transport teleport earns nothing
        ok = climb(-0.05) and settle_until(lambda: bool(scene.success()[0]),
                                           max_steps=400)
        step(240)  # persistence: success must not flicker off
        report(f"oracle{s}-final")
        check(f"oracle seed {s}: transport (zero credit) -> ramp climb -> doorway -> "
              f"tray = success, score 1.0, persists 240 steps",
              zero_after_tp and ok and bool(scene.success()[0]) and sc() == 1.0)

    # =========================== 8. occupied-tray forfeit ========================================
    # Continue from oracle 2's success state: teleport a DECOY into the tray.
    decoy = scene.bowls[(int(scene.target_idx[0]) + 1) % 3]
    tp(decoy, fix_to_world([0.0, 0.0, 0.26]), upq())
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    report("occupied")
    check("occupied forfeit: decoy bowl inside the tray flips success OFF; score falls "
          "back to the 0.75 latch sum",
          bool(scene.decoy_in_tray()[0]) and not bool(scene.success()[0])
          and 0.73 <= sc() <= 0.77)

    # =========================== 9-10. rubric monotonicity =======================================
    torch.manual_seed(41)
    env.reset()
    step(30)
    s0 = sc()
    park_on_ramp(tgt(), -0.25)  # inside the hi band, below the crest
    step(60)
    s_hi = sc()
    hi_no_success = (bool(scene._hi_ever[0]) and not bool(scene._doored_ever[0])
                     and not bool(scene.success()[0]))
    report("mono-hi")
    drive(tgt(), up_tan(), 2.5, 0.08, lambda: bool(scene._doored_ever[0]), 2400)
    s_door = sc()
    report("mono-door")
    ok_seat = climb(-0.05) and settle_until(lambda: bool(scene.success()[0]),
                                            max_steps=400)
    s_seat = sc()
    report("mono-final")
    print(f"[smoke] monotonicity ladder: idle={s0:.3f} hi={s_hi:.3f} "
          f"door={s_door:.3f} seated={s_seat:.3f}", flush=True)
    check("monotonicity: idle < parked-hi (0.15, NO success) < doorway latch (0.40) "
          "< seated (1.0)",
          s0 <= 0.02 and 0.13 <= s_hi <= 0.17 and hi_no_success
          and 0.38 <= s_door <= 0.42 and ok_seat and s_seat == 1.0
          and s0 < s_hi < s_door < s_seat)
    check("monotonicity: partial states score < 1.0", max(s0, s_hi, s_door) < 1.0)

    # =========================== 11. negative: the seed's strategy ===============================
    # The seed's plan — carry the bowl above the cabinet and release it — lands on the
    # ROOF, which is not the goal surface.
    torch.manual_seed(51)
    env.reset()
    step(30)
    tp(tgt(), fix_to_world([-0.04, 0.0, 0.42]), upq())
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    step(60)
    zf = float(scene.target_fix()[0, 2])
    report("roof-drop")
    check("negative (seed strategy): bowl released above the cabinet lands ON THE ROOF "
          "(rest z readback above the roof plane) -> no latches, score 0, no success",
          zf > 0.33 and not bool(scene._hi_ever[0]) and not bool(scene._doored_ever[0])
          and not bool(scene._seated_ever[0]) and sc() <= 1e-6
          and not bool(scene.success()[0]))

    # =========================== 12. negative: teleport bypass ===================================
    torch.manual_seed(61)
    env.reset()
    step(30)
    tp(tgt(), fix_to_world([-0.05, 0.0, 0.26]), upq())
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    step(60)
    live_seated = bool(scene.target_seated()[0])
    report("tray-bypass")
    check("negative (teleport bypass): target placed directly inside the tray settles "
          "into the seat band (live predicate TRUE — probe not vacuous) yet the seated "
          "latch stays false without the doorway passage -> score 0, no success",
          live_seated and not bool(scene._doored_ever[0])
          and not bool(scene._seated_ever[0]) and sc() <= 1e-6
          and not bool(scene.success()[0]))

    # =========================== 13. negative: inverted on the hi band ===========================
    torch.manual_seed(71)
    env.reset()
    step(30)
    park_on_ramp(tgt(), -0.25, inverted=True)
    step(60)
    loc = scene.target_fix()[0]
    in_band = (c.hi_x_lo < float(loc[0]) < c.hi_x_hi
               and abs(float(loc[1])) < c.hi_y_abs
               and c.hi_z_lo < float(loc[2]) < c.hi_z_hi)
    report("inverted-hi")
    check("negative (inverted): bowl UPSIDE-DOWN parked inside the hi-ramp spatial band "
          "(position readback IN band — probe not vacuous) -> hi latch refuses, score 0",
          in_band and not bool(scene._hi_ever[0]) and sc() <= 1e-6
          and not bool(scene.success()[0]))

    # =========================== 14. credit permanence ===========================================
    torch.manual_seed(81)
    env.reset()
    step(30)
    park_on_ramp(tgt(), -0.25)
    step(60)
    s_before = sc()
    tp(tgt(), fix_to_world([-0.55, -0.38, c.bowl_h / 2 + 0.004]), upq())  # back to floor
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    step(60)
    report("permanence")
    check("credit permanence: parked hi (0.15) then removed to the floor -> latched "
          "score stays 0.15, no success",
          0.13 <= s_before <= 0.17 and 0.13 <= sc() <= 0.17
          and not bool(scene.success()[0]))

    # =========================== 15. sill one-way ================================================
    torch.manual_seed(91)
    env.reset()
    step(30)
    park_on_ramp(tgt(), -0.55)
    climb(-0.02)
    settle_until(lambda: bool(scene.success()[0]), max_steps=400)
    x_start = x_fix()
    report("oneway-seated")
    # sustained quasi-static pull back toward the doorway: must NOT exit
    escaped = drive(tgt(), -fix_x_axis(), 2.5, 0.08, lambda: x_fix() < -0.16, 1500)
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    x_end = x_fix()
    still_in = bool(scene.bowls_in_tray()[0, int(scene.target_idx[0])])
    ok_recover = settle_until(lambda: bool(scene.success()[0]), max_steps=300)
    report("oneway-final")
    check("sill one-way: a sustained pull toward the doorway moves the bowl "
          f"({(x_start - x_end) * 1000:.0f} mm of travel — probe not vacuous) but the "
          "sill's inner step blocks the exit; bowl stays in the tray, success recovers",
          not escaped and (x_start - x_end) > 0.03 and x_end > -0.16
          and still_in and ok_recover)

    # =========================== save + verdict ==================================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.ramp_hutch")
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

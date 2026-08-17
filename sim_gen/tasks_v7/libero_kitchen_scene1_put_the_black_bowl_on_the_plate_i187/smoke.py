"""Smoke / rubric-rejection battery for BeamBalanceScene (sim_gen task
`libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i187`) — NullRobot, teleported
probe states (instrumentation, NOT a solution), RECORDED.

Battery:
  1. settle/no-NaN      — reset settles finite: beam resting on its PAN-UP stop, bowls on
                          the ground, score ~0, no latches, no success; bowl mass READBACK
                          matches the episode's sampled table (one heavy, two light);
  2. randomization      — READBACK across seeds: the heavy bowl's mass value, WHICH bowl
                          is heavy, and the heavy bowl's ground position all move; bowl xy
                          jitter and yaw are real; the beam always starts pan-up;
  3. null-policy-fails  — 240 idle steps -> score ~0, no success, beam still pan-up;
  4-6. oracle x3 seeds  — the heavy bowl dropped onto the pan drives the beam to its down
                          stop -> success() and score 1.0; PERSISTS over 240 further steps;
  7. monotonicity       — 0 (idle) < 0.25 (light bowl weighed) == 0.25 after removing it
                          (latched credit does not evaporate) < 1.0 (heavy panned);
  8. negative A (seed)  — the seed's blind plan: a LIGHT bowl placed on the plate ->
                          settled, beam stays pan-up, no success, score <= 0.25;
  9. negative B (mass cheat) — BOTH light bowls piled onto the pan: their combined weight
                          tips the beam, but success() rejects (exactly-one-bowl clause);
 10. negative C (prop on balance) — heavy bowl correctly panned but a light bowl left
                          lying on the balance (on the counterweight arm) -> no success;
 11. negative D (near-miss xy) — heavy bowl resting on the beam BAR just inboard of the
                          pan -> outside the pan region AND not enough lever arm to tip
                          the beam -> no success;
 12. negative E (inverted) — heavy bowl UPSIDE-DOWN on the pan: the beam tips (mass is
                          real) but the upright clause rejects;
 13. discrimination     — each bowl alone on the pan across one seed: EXACTLY the heavy
                          one drives the beam to its down stop (the physics is the mass
                          check the rubric relies on).

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i187.smoke --headless
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

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i187 import (  # noqa: F401,E501
        scene as scene_mod,
    )
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.beam_balance")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.55, -0.85, 0.62)) + o),
                                tuple(np.array((0.30, 0.0, 0.16)) + o),
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

    def settle_until(pred, max_steps: int = 400, poll: int = 10) -> bool:
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

    def ang() -> float:
        return math.degrees(float(scene.beam_angle()[0]))

    def report(tag: str) -> None:
        onp = scene.on_pan()[0]
        zs = (scene._bowl_pos_w() - scene.env_origins[:, None, :])[0, :, 2]
        print(f"[smoke] {tag:16s} ang={ang():+6.2f} "
              f"on_pan={[bool(v) for v in onp]} "
              f"bowl_z=({zs[0]:.3f},{zs[1]:.3f},{zs[2]:.3f}) "
              f"down={bool(scene.beam_down()[0])} up={bool(scene.beam_up()[0])} "
              f"settled={bool(scene.settled()[0])} score={sc():.2f} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def beam_local_to_world(loc):
        from isaaclab.utils.math import quat_apply

        p = quat_apply(scene.beam.data.root_quat_w[0:1],
                       torch.tensor([loc], device=device))[0]
        return p + scene.beam.data.root_pos_w[0] - scene.env_origins[0]

    def tp(body, pos_env, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.env_origins + torch.tensor(
            [float(v) for v in pos_env], device=device)
        st[:, 3:7] = torch.tensor([float(v) for v in quat], device=device)
        body.write_root_state_to_sim(st, all_ids)

    def drop_on_pan(bowl, dz: float = 0.045, inverted: bool = False) -> None:
        """Release a bowl just above the pan (beam-local, tracks the live tilt)."""
        q = (0.0, 1.0, 0.0, 0.0) if inverted else (1.0, 0.0, 0.0, 0.0)
        pos = beam_local_to_world([0.0, c.arm_len, c.plate_top_z + c.bowl_h / 2 + dz])
        tp(bowl, [float(v) for v in pos], q)
        # settled() is VACUOUSLY true at the teleport instant (zero velocities) — force
        # a minimum fall-and-impact window before consulting it.
        step(25)
        settle_until(lambda: bool(scene.settled()[0]), max_steps=400)

    def bowl_z(k: int) -> float:
        return float((scene.bowls[k].data.root_pos_w - scene.env_origins)[0, 2])

    def heavy_light() -> tuple[int, int, int]:
        order = scene._masses[0].argsort()
        return int(order[2]), int(order[0]), int(order[1])  # heavy, lightest, mid

    # =========================== 1. settle / no-NaN ==============================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("settled")
    st0 = torch.cat([scene.beam.data.root_state_w]
                    + [b.data.root_state_w for b in scene.bowls], dim=-1)
    check("settle: states finite, beam resting PAN-UP on its stop, all bowls on the "
          "ground, everything settled",
          bool(torch.isfinite(st0).all()) and bool(scene.beam_up()[0])
          and all(bowl_z(k) < 0.06 for k in range(3)) and bool(scene.settled()[0]))
    masses_rb = [float(b.root_physx_view.get_masses().reshape(-1)[0]) for b in scene.bowls]
    masses_auth = [float(v) for v in scene._masses[0]]
    srt = sorted(masses_rb)
    print(f"[smoke] mass readback={[round(v, 3) for v in masses_rb]} "
          f"authored={[round(v, 3) for v in masses_auth]}", flush=True)
    check("settle: physx mass readback matches the sampled table; exactly one heavy "
          "(>= 0.60), two light (<= 0.30)",
          all(abs(a - b) < 1e-4 for a, b in zip(masses_rb, masses_auth))
          and srt[2] >= c.heavy_mass_lo - 1e-6 and srt[1] <= c.light_mass_hi + 1e-6)
    check("settle: score ~0 at reset, no success, no latches",
          sc() <= 0.005 and not bool(scene.success()[0])
          and not bool(scene._lift_ever[0]) and not bool(scene._weigh_ever[0])
          and not bool(scene._tip_ever[0]))

    # =========================== 2. randomization is real ========================================
    reads = []
    for s in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(s)
        env.reset()
        step(4)
        hv = int(scene.heavy_index()[0])
        hp = (scene.bowls[hv].data.root_pos_w - scene.env_origins)[0]
        from isaaclab.utils.math import quat_apply

        ex = quat_apply(scene.bowls[0].data.root_quat_w[0:1],
                        torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
        yaw = math.atan2(float(ex[1]), float(ex[0]))
        b0 = (scene.bowls[0].data.root_pos_w - scene.env_origins)[0]
        reads.append((hv, float(scene._masses[0, hv]), float(hp[0]), float(hp[1]),
                      float(b0[0]), float(b0[1]), yaw, ang()))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (heavy_idx, heavy_mass, heavy_x, heavy_y, "
          f"b0_x, b0_y, b0_yaw, beam_ang):\n{np.round(arr, 3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: WHICH bowl is heavy varies; heavy mass value varies; the "
          "heavy bowl's ground slot moves; bowl yaw moves; beam always starts pan-up",
          len(set(arr[:, 0])) >= 2 and spread[1] > 0.02
          and (spread[2] > 0.02 or spread[3] > 0.05) and spread[6] > 0.5
          and all(v > c.up_ref_deg - 0.5 for v in arr[:, 7]))

    # =========================== 3. null policy fails ============================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    check("null policy: score ~0, no success, beam still pan-up after 240 idle steps",
          sc() <= 0.02 and not bool(scene.success()[0]) and bool(scene.beam_up()[0]))

    # =========================== 4-6. oracle on 3 seeds ==========================================
    for s in (0, 1, 2):
        torch.manual_seed(s)
        env.reset()
        step(30)
        hv, _lo, _mid = heavy_light()
        drop_on_pan(scene.bowls[hv])
        ok = settle_until(lambda: bool(scene.success()[0]), max_steps=400)
        step(240)  # persistence: success must not flicker off
        report(f"oracle{s}-final")
        check(f"oracle seed {s}: heavy bowl on the pan drives the beam to its down stop "
              f"-> success, score 1.0, persists 240 steps",
              ok and bool(scene.success()[0]) and sc() == 1.0)

    # =========================== 7. rubric monotonicity ==========================================
    torch.manual_seed(41)
    env.reset()
    step(30)
    hv, lo, _mid = heavy_light()
    s0 = sc()
    home = (scene.bowls[lo].data.root_pos_w[0] - scene.env_origins[0]).clone()
    drop_on_pan(scene.bowls[lo])
    s_weigh = sc()
    report("mono-weighed")
    up_after_light = bool(scene.beam_up()[0])
    tp(scene.bowls[lo], [float(home[0]), float(home[1]), c.bowl_h / 2 + 0.010])
    settle_until(lambda: bool(scene.settled()[0]) and bool(scene.beam_up()[0]),
                 max_steps=400)
    s_off = sc()
    report("mono-off")
    drop_on_pan(scene.bowls[hv])
    settle_until(lambda: bool(scene.success()[0]), max_steps=400)
    s_final = sc()
    report("mono-final")
    print(f"[smoke] monotonicity ladder: idle={s0:.3f} weighed={s_weigh:.3f} "
          f"off={s_off:.3f} success={s_final:.3f}", flush=True)
    check("monotonicity: idle ~0 < weighed (0.25) == after removal (latched credit "
          "does not evaporate) < success (1.0); beam stayed up for the light bowl",
          s0 <= 0.02 and 0.23 <= s_weigh <= 0.27 and abs(s_off - s_weigh) < 1e-6
          and s_final == 1.0 and up_after_light)
    check("monotonicity: partial states score < 1.0", max(s0, s_weigh, s_off) < 1.0)

    # =========================== 8. negative A: the seed's own strategy ==========================
    # "Put the black bowl on the plate" verbatim, picking a bowl blindly (here: a light
    # one, the 2-in-3 outcome). It rests on the plate — and the task rejects it.
    torch.manual_seed(51)
    env.reset()
    step(30)
    _hv, lo, _mid = heavy_light()
    drop_on_pan(scene.bowls[lo])
    step(120)
    report("seed-strategy")
    check("negative A (seed strategy): a LIGHT bowl placed on the plate settles there "
          "but the beam stays pan-up -> no success, score <= 0.25",
          bool(scene.on_pan()[0, lo]) and bool(scene.beam_up()[0])
          and not bool(scene.success()[0]) and sc() <= 0.255)

    # =========================== 9. negative B: combined-mass cheat ==============================
    # Two light bowls piled onto the pan outweigh the counterweight together — the beam
    # tips, but the exactly-one-bowl clause rejects the state.
    torch.manual_seed(61)
    env.reset()
    step(30)
    _hv, lo, mid = heavy_light()
    drop_on_pan(scene.bowls[lo])
    drop_on_pan(scene.bowls[mid], dz=0.075)  # second bowl nests into / stacks on the first
    settle_until(lambda: bool(scene.settled()[0]), max_steps=400)
    step(120)
    report("two-bowl-cheat")
    onp = scene.on_pan()[0]
    both_elevated = bowl_z(lo) > c.ground_z_max and bowl_z(mid) > c.ground_z_max
    check("negative B (mass cheat): both light bowls piled on the pan tip the beam, but "
          "success() rejects — never exactly one bowl on the pan with the rest grounded",
          both_elevated and not bool(scene.success()[0])
          and not (int(onp.sum()) == 1 and bool(scene.grounded()[0].sum() == 2)))
    print(f"[smoke]   two-bowl state: ang={ang():+.2f} on_pan={[bool(v) for v in onp]} "
          f"(beam_down={bool(scene.beam_down()[0])})", flush=True)

    # =========================== 10. negative C: bowl left on the balance ========================
    torch.manual_seed(71)
    env.reset()
    step(30)
    hv, lo, _mid = heavy_light()
    # light bowl parked on the counterweight arm of the beam (on the balance, not pan)
    pos = beam_local_to_world([0.0, -c.arm_len + 0.09,
                               c.bar_t / 2 + c.bowl_h / 2 + 0.006])
    tp(scene.bowls[lo], [float(v) for v in pos])
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    drop_on_pan(scene.bowls[hv])
    settle_until(lambda: bool(scene.settled()[0]), max_steps=400)
    step(120)
    report("prop-on-balance")
    check("negative C (bowl on the balance): heavy bowl panned but a light bowl lying "
          "on the balance arm -> no success (all other bowls must rest on the ground)",
          not bool(scene.success()[0]) and not bool(scene.grounded()[0, lo]))

    # =========================== 11. negative D: near-miss xy ====================================
    torch.manual_seed(81)
    env.reset()
    step(30)
    hv, _lo, _mid = heavy_light()
    # heavy bowl set on the bar just INBOARD of the pan: outside pan_xy_tol AND with too
    # little lever arm to overcome the bias
    pos = beam_local_to_world([0.0, c.arm_len - 0.105,
                               c.bar_t / 2 + c.bowl_h / 2 + 0.006])
    tp(scene.bowls[hv], [float(v) for v in pos])
    settle_until(lambda: bool(scene.settled()[0]), max_steps=400)
    step(120)
    report("near-miss-xy")
    check("negative D (near-miss xy): heavy bowl on the beam bar inboard of the pan -> "
          "outside the pan region, insufficient lever arm, no success",
          not bool(scene.on_pan()[0, hv]) and not bool(scene.success()[0]))

    # =========================== 12. negative E: inverted on the pan =============================
    torch.manual_seed(91)
    env.reset()
    step(30)
    hv, _lo, _mid = heavy_light()
    drop_on_pan(scene.bowls[hv], inverted=True)
    step(120)
    report("inverted")
    check("negative E (inverted): heavy bowl UPSIDE-DOWN on the pan tips the beam (its "
          "mass is real) but the upright clause rejects success",
          bool(scene.beam_down()[0]) and not bool(scene.bowls_upright()[0, hv])
          and not bool(scene.success()[0]) and sc() < 1.0)

    # =========================== 13. discrimination ==============================================
    print("[smoke] DISCRIMINATION: each bowl alone on the pan -> beam verdict", flush=True)
    torch.manual_seed(101)
    env.reset()
    step(30)
    masses = [float(v) for v in scene._masses[0]]
    hv = int(scene.heavy_index()[0])
    verdicts = []
    for k in range(3):
        torch.manual_seed(101)  # same episode layout each probe
        env.reset()
        step(30)
        drop_on_pan(scene.bowls[k])
        settle_until(lambda: bool(scene.beam_still()[0]), max_steps=400)
        step(90)
        verdicts.append(bool(scene.beam_down()[0]))
        print(f"[smoke]   bowl_{k} (mass {masses[k]:.3f} kg): ang={ang():+.2f} "
              f"beam_down={verdicts[-1]}", flush=True)
    check("discrimination: EXACTLY the heavy bowl drives the beam to its down stop; "
          "both light bowls leave it pan-up",
          verdicts[hv] and sum(verdicts) == 1)

    # =========================== save + verdict ==================================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.beam_balance")
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
    try:
        main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL exception", flush=True)
        os._exit(2)

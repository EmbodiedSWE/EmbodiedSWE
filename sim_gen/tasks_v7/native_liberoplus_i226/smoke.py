"""Smoke battery for BalanceVerdictScene — REJECTION tests for the rubric, NullRobot,
RECORDED.

This is NOT a solution (the solution is solve.py — weigh both canisters on the beam,
read the verdict from the tilt, seat the heavy one on the pedestal; the Franka strategy
is TASK.md's embodiment argument). Teleports here are TRANSPORT ONLY, exactly as in
solve.py: every canister is released ABOVE its target (outside every latch z-window)
and the landing / beam verdict / seating flow through contact dynamics. Probe forces go
through the scene-owned `drive_f` buffers (post_step owns the wrench slots).

One linear run, 11 named checks:
  1. settle    — clean reset: finite state, both canisters physically AT their sampled
                 pads (readback vs cfg.pads[heavy_pad]), beam level, the hidden masses
                 real (get_masses readback), score ~0, no success;
  2. random    — randomization is real (READBACK, 8 seeds): heavy_pad takes BOTH
                 values, canister xy and pedestal xy vary, can0's y-sign matches its
                 sampled pad;
  3. null      — 2 s of nothing: score < 0.02, no success;
  4. negative  — the SEED's plan / order violation: the heavy canister set straight on
                 the pedestal WITHOUT weighing seats fine (heavy_on_pedestal True) but
                 weighed stays False -> no success, score ~0.20 only;
  5. negative  — single load is not an experiment: only the light canister on one pan
                 slams the beam past tip_min (|tilt| > 6 deg) yet `weighed` never
                 latches (needs BOTH pans loaded), score < 0.12, no success;
  6. negative  — wrong object: full weigh, then the LIGHT canister seated on the
                 pedestal — weighed stays latched but no success, score ~0.40;
  7. negative  — near miss: full weigh, heavy released 5 cm off the pedestal axis —
                 lands off/topples, no success, score ~0.40;
  8. exactness — full correct strategy: weigh, read the beam's own verdict (down_side
                 must agree with the ground-truth heavy side), seat the heavy canister
                 -> success() and score == 1.0, still true 1 s hands-off later;
  9. latch     — shoving the heavy canister off the pedestal (drive_f probe) revokes
                 success; the latched weigh/pan credit remains (~0.40);
 10. negative  — clearance: light canister stacked on top of the seated heavy one sits
                 inside clear_r -> no success, score < 0.9;
 11. frames    — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.native_liberoplus_i226.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=10)
parser.add_argument("--max_frames", type=int, default=280)
parser.add_argument("--out", type=str, default="frames.npz")  # CWD — the pipeline fetches it
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe (proven on this render stack): kit mis-decodes the driver version and
# silently rejects RTX -> the annotator returns EMPTY frames.
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
    from simgen_tasks.native_liberoplus_i226 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie.
threading.Timer(1200.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                 os._exit(3))).start()

PROBE_SEED = 5
DROP_H_PAN = 0.012  # release height above a pan seat (m) — same as solve.py
DROP_H_PED = 0.015  # release height above the pedestal seat (m) — same as solve.py
DROP_H_STACK = 0.010  # release height when stacking the light on the heavy (m)
SHOVE_F = 6.0  # N lateral probe — >> the heavy canister's ~2 N friction budget


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.balance_verdict")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    from isaaclab.utils.math import quat_apply  # noqa: E402 (post-AppLauncher)

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.85, -0.75, 1.10)) + o),
                                tuple(np.array((0.0, 0.05, 0.48)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (800, 500))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames — check the RTX recipe",
                  flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action, render=annot is not None)
            if annot is not None and step_i % args.record_every == 0 \
                    and len(frames) < args.max_frames:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        tilt = math.degrees(float(scene.beam_tilt()[0]))
        on, side = scene._pan_metrics()
        print(f"[smoke] {tag:14s} | tilt={tilt:+6.2f}deg on_pan={on[0].tolist()} "
              f"side={side[0].tolist()} pan_latch={scene.pan_latch[0].tolist()} "
              f"weighed={bool(scene.weighed[0])} "
              f"on_ped={bool(scene.heavy_on_pedestal()[0])} "
              f"clear={bool(scene.light_clear()[0])} "
              f"score={sc():.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    # --- transport helpers (identical mechanics to solve.py) ---
    def teleport(j: int, xyz: torch.Tensor, quat: torch.Tensor | None = None) -> None:
        """TRANSPORT ONLY: pose canister j at world xyz, zero velocity."""
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = xyz
        if quat is None:
            st[0, 3] = 1.0
        else:
            st[0, 3:7] = quat
        scene.cans[j].write_root_state_to_sim(st, torch.tensor([0], device=device))

    def wait_until(fn, budget: int) -> bool:
        for _ in range(budget):
            if bool(fn()):
                return True
            step(1)
        return bool(fn())

    def drop_on_pan(j: int) -> bool:
        """Release canister j ~1.2 cm above its NEAR pan, beam-aligned (lands flat on a
        tilted pan); returns True once it is seated and slow. Landing is pure contact."""
        origin = scene.env_origins[0]
        side = 1.0 if float((scene.cans[j].data.root_pos_w[0] - origin)[1]) >= 0 else -1.0
        beam_q = scene.beam.data.root_quat_w[0].clone()
        up = quat_apply(beam_q.unsqueeze(0),
                        torch.tensor([[0.0, 0.0, 1.0]], device=device))[0]
        target = scene.pan_center_w(side)[0] + up * (c.can_h / 2 + DROP_H_PAN)
        teleport(j, target, quat=beam_q)
        ok = wait_until(
            lambda: scene._pan_metrics()[0][0, j]
            and scene.cans[j].data.root_lin_vel_w.norm() < c.can_slow, 600)
        step(40)
        return ok

    def weigh() -> bool:
        """Both canisters onto opposite pans, then wait for the `weighed` latch."""
        if not (drop_on_pan(0) and drop_on_pan(1)):
            return False
        return wait_until(lambda: scene.weighed[0], 720)

    def drop_on_pedestal(j: int, dx: float = 0.0, dy: float = 0.0) -> None:
        """Release canister j ~1.5 cm above the pedestal seat (+ optional xy offset)."""
        target = scene.pedestal.data.root_pos_w[0].clone()
        target[0] += dx
        target[1] += dy
        target[2] += c.ped_h / 2 + c.can_h / 2 + DROP_H_PED
        teleport(j, target)

    def finite_all() -> bool:
        return all(bool(torch.isfinite(b.data.root_state_w).all())
                   for b in scene._bodies().values()) \
            and bool(torch.isfinite(scene.score()).all())

    def reset(seed: int) -> None:
        torch.manual_seed(seed)
        env.reset()

    # ========================= 1. settle / clean-slate ========================================
    reset(PROBE_SEED)
    step(60)
    report("reset")
    origin = scene.env_origins[0]
    hp = int(scene.heavy_pad[0])
    pads = torch.tensor(c.pads, device=device)
    pad_err = 0.0
    for j, can in enumerate(scene.cans):
        pad = pads[hp if j == 0 else 1 - hp]
        got = (can.data.root_pos_w[0] - origin)[:2]
        pad_err = max(pad_err, float((got - pad).norm()))
    masses = [float(can.root_physx_view.get_masses().reshape(-1)[0]) for can in scene.cans]
    print(f"[smoke] heavy_pad={hp} pad readback err={pad_err * 1000:.1f}mm "
          f"masses={masses} (want [{c.heavy_mass}, {c.light_mass}])", flush=True)
    check("settle: clean reset (finite, canisters AT their pads, beam level, real masses, "
          "score ~0)",
          finite_all() and pad_err < c.pad_jitter * math.sqrt(2) + 0.01
          and abs(math.degrees(float(scene.beam_tilt()[0]))) < 2.0
          and abs(masses[0] - c.heavy_mass) < 0.02 and abs(masses[1] - c.light_mass) < 0.02
          and sc() < 0.02 and not bool(scene.success()[0]))

    # ========================= 2. randomization across seeds ==================================
    hps, can_xy, ped_xy, sign_ok = [], set(), set(), True
    for seed in (11, 12, 13, 14, 15, 16, 17, 18):
        reset(seed)
        step(2)
        hps.append(int(scene.heavy_pad[0]))
        p0 = scene.cans[0].data.root_pos_w[0] - scene.env_origins[0]
        pd = scene.pedestal.data.root_pos_w[0] - scene.env_origins[0]
        can_xy.add((round(float(p0[0]), 3), round(float(p0[1]), 3)))
        ped_xy.add((round(float(pd[0]), 3), round(float(pd[1]), 3)))
        # READBACK: can0 (heavy) physically stands on the SAMPLED pad's side
        sign_ok &= float(p0[1]) * c.pads[int(scene.heavy_pad[0])][1] > 0
    print(f"[smoke] heavy_pads={hps} unique can0 xy={len(can_xy)} "
          f"unique ped xy={len(ped_xy)}", flush=True)
    check("randomization is real (heavy pad takes both values; canister/pedestal xy vary; "
          "readback matches the draw)",
          len(set(hps)) == 2 and len(can_xy) >= 5 and len(ped_xy) >= 5 and sign_ok)

    # ========================= 3. null policy =================================================
    reset(PROBE_SEED)
    step(240)  # 2 s of nothing
    report("null")
    check("null policy: score < 0.02 and no success",
          sc() < 0.02 and not bool(scene.success()[0]))

    # ========================= 4. negative: seed strategy / order violation ===================
    # The LIBERO-plus move: just put the (guessed) target on the goal. Even guessing RIGHT
    # earns no success here, because the weighing experiment never happened.
    reset(PROBE_SEED)
    step(30)
    drop_on_pedestal(0)  # the heavy one — a LUCKY guess
    seated = wait_until(lambda: scene.heavy_on_pedestal()[0], 600)
    step(240)
    report("unweighed")
    check("negative (order violation): heavy seated WITHOUT weighing -> on_pedestal True "
          "but no success, score ~0.20",
          seated and bool(scene.heavy_on_pedestal()[0]) and not bool(scene.weighed[0])
          and not bool(scene.success()[0]) and 0.15 < sc() < 0.25)

    # ========================= 5. negative: single load is not an experiment ==================
    reset(PROBE_SEED)
    step(30)
    ok1 = drop_on_pan(1)  # ONLY the light canister
    step(240)  # beam parks on its stop
    report("single-load")
    tilt = abs(math.degrees(float(scene.beam_tilt()[0])))
    check("negative (single load): one canister slams the beam past tip_min yet `weighed` "
          "never latches, score < 0.12",
          ok1 and tilt > c.tip_min_deg and not bool(scene.weighed[0])
          and not bool(scene.success()[0]) and sc() < 0.12)

    # ========================= 6. negative: wrong object ======================================
    reset(PROBE_SEED)
    step(30)
    ok_w = weigh()
    report("weighed")
    drop_on_pedestal(1)  # the LIGHT canister onto the pedestal
    step(480)
    report("wrong-object")
    check("negative (wrong object): LIGHT canister on the pedestal after a real weigh -> "
          "no success, latched credit only (~0.40)",
          ok_w and bool(scene.weighed[0]) and not bool(scene.heavy_on_pedestal()[0])
          and not bool(scene.light_clear()[0])
          and not bool(scene.success()[0]) and sc() < 0.45)

    # ========================= 7. negative: near miss =========================================
    reset(PROBE_SEED)
    step(30)
    ok_w = weigh()
    drop_on_pedestal(0, dy=0.05)  # 5 cm off the axis: lands on the edge, topples off
    step(480)
    report("near-miss")
    check("negative (near miss): heavy released 5 cm off the pedestal axis -> not seated, "
          "no success, score < 0.45",
          ok_w and finite_all() and not bool(scene.heavy_on_pedestal()[0])
          and not bool(scene.success()[0]) and sc() < 0.45)

    # ========================= 8. exactness: full correct strategy ============================
    reset(PROBE_SEED)
    step(30)
    ok_w = weigh()
    report("weighed")
    # The beam's own verdict must agree with the ground truth (comparator sanity).
    down = float(scene.down_side()[0])
    _on, side = scene._pan_metrics()
    verdict_ok = float(side[0, 0]) == down  # can0 IS the heavy one by construction
    print(f"[smoke] verdict readback: down={down:+.0f} can0 side={float(side[0, 0]):+.0f} "
          f"agree={verdict_ok}", flush=True)
    drop_on_pedestal(0)
    ok_s = wait_until(lambda: scene.success()[0], 720)
    step(60)
    report("goal-state")
    good = ok_w and verdict_ok and ok_s and bool(scene.success()[0]) \
        and abs(sc() - 1.0) < 1e-3
    step(120)  # ...and it persists hands-off
    check("exactness: weigh -> beam verdict agrees with ground truth -> heavy seated -> "
          "success() and score == 1.0, stable",
          good and bool(scene.success()[0]) and abs(sc() - 1.0) < 1e-3)

    # ========================= 9. achievement latch (revocation) ==============================
    # Shove the heavy canister off the pedestal (through the scene-owned drive_f probe).
    # Break on a POSITION readback, not heavy_on_pedestal(): its stillness gate reads
    # False the instant the shove imparts speed, which would self-cancel the probe.
    ped_xy = scene.pedestal.data.root_pos_w[0, :2].clone()
    p_start = scene.cans[0].data.root_pos_w[0, :2].clone()
    for _ in range(240):
        scene.drive_f[0, 0, 0] = SHOVE_F * 0.5
        scene.drive_f[0, 0, 1] = SHOVE_F
        step(1)
        if float((scene.cans[0].data.root_pos_w[0, :2] - ped_xy).norm()) > 0.08:
            break
    scene.drive_f[0, 0] = 0.0
    step(360)
    moved = float((scene.cans[0].data.root_pos_w[0, :2] - p_start).norm())
    report("revoked")
    print(f"[smoke] shove displacement={moved * 100:.1f}cm", flush=True)
    check("achievement latch: shoving the heavy off the pedestal revokes success; latched "
          "weigh credit remains",
          moved > 0.05 and finite_all() and not bool(scene.heavy_on_pedestal()[0])
          and not bool(scene.success()[0]) and 0.35 < sc() < 0.45)

    # ========================= 10. negative: clearance violated ===============================
    reset(PROBE_SEED)
    step(30)
    ok_w = weigh()
    drop_on_pedestal(0)
    ok_s = wait_until(lambda: scene.success()[0], 720)
    # now stack the light canister ON TOP of the seated heavy one — inside clear_r
    top = scene.cans[0].data.root_pos_w[0].clone()
    top[2] += c.can_h + DROP_H_STACK
    teleport(1, top)
    step(360)
    report("stacked")
    check("negative (clearance): light canister stacked on the seated heavy one -> "
          "light not clear, success revoked, score < 0.9",
          ok_w and ok_s and not bool(scene.light_clear()[0])
          and not bool(scene.success()[0]) and sc() < 0.9)

    # ========================= 11. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.balance_verdict")
        print(f"[smoke] saved {arr.shape} -> {os.path.abspath(args.out)}", flush=True)
    check("video frames recorded", len(frames) >= 20)

    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        bad = [nm for nm, ok in checks if not ok]
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)} — failing: {bad}", flush=True)
    # Kit teardown hangs are routine: watchdog then hard exit.
    threading.Timer(10.0, lambda: os._exit(0 if all_ok else 1)).start()
    try:
        env.close()
        app.close()
    finally:
        os._exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

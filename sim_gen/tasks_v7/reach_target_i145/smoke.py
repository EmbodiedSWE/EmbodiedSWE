"""Smoke / rubric-REJECTION battery for ColorCarouselScene (sim_gen task
`reach_target_i145`) — NullRobot, teleported probe states + force probes, RECORDED.

This is NOT a solution (solve.py — torque-align the target cell under the chute, then
release the token above the chute and let gravity insert it — is the acceptance
evidence that the rubric ACCEPTS a correct outcome; it passes on seeds 0/1, resting at
carousel-frame uvz ~ (0.100, 0.000, 0.023) inside the containment window). Every
teleport here is instrumentation that CONSTRUCTS a wrong (or partial) outcome as a
settled state and asserts the rubric REJECTS it — plus physical probes that prove the
COVER is real geometry: a token dropped anywhere but the chute never enters a cell.
No probe in this battery ever reaches success(), and a final audit asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: token on the floor off the
                            machine, carousel >= 20 deg from alignment and still, the
                            matching cue card in the tray; score ~0, no success;
  3-4. randomization      — READBACK over 6 seeded resets: the target color, the
                            carousel angle (never aligned), and the token slot all
                            vary; the tray card always matches the target;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed-strategy       — the seed family's move (REACH out and touch the colored
                            thing) achieves nothing: a nudge that provably moves the
                            token leaves score ~0;
  7.  roof cover          — the token dropped from above the COVERED target cell
                            never enters it: it ends on the roof or off the machine,
                            never below the roof plane inside the rotor footprint;
  8.  wrong-cell chute    — carousel teleported along its own DOF to align the WRONG
                            cell, token dropped through the chute exactly as solve
                            does: it lands INSIDE that wrong cell (positive control —
                            the chute really feeds the aligned cell) yet in_target is
                            False, no drop credit, score ~0 (identity, not geometry);
  9.  alignment window    — 12 deg off: not aligned, no latch; 2 deg off, held still:
                            aligned, latch fires, score 0.25 exactly (the align stage);
  10. drive-by sweep      — spinning the rotor THROUGH the aligned window at ~0.8
                            rad/s (window verifiably crossed) latches NOTHING: the
                            align latch requires holding still inside the window;
  11. deck-rider          — token settled on the deck BETWEEN cells rides the rotor
                            but is in NO cell: containment windows reject it;
  12. settle gate         — token IN the target cell window but still moving is NOT
                            success (velocity gates are real); removed before it can
                            settle;
  13. latched credit      — teleporting the token far away afterwards leaves the
                            latched score (0.25 align + 0.45 drop) unchanged while
                            token_in_target_cell() drops;
  14. rejection audit     — success() was never True at ANY judged point;
  15. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.reach_target_i145.smoke --headless
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
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.color_carousel")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(3, device=device)
    hx, hy = c.hub_pos
    trx, try_ = hx + c.tray_pos[0], hy + c.tray_pos[1]

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.35, -1.25, 1.00)) + o),
                                tuple(np.array((0.42, 0.00, 0.08)) + o),
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

    def err_deg() -> float:
        return math.degrees(float(scene.align_err()[0]))

    def report(tag: str) -> None:
        s, ok = judge()
        tok = rel(scene.token)
        w = float(scene.carousel.data.root_ang_vel_w[0, 2])
        print(f"[smoke] {tag:16s} | target={int(scene.target[0])}"
              f" err={err_deg():+7.2f}deg w={w:+.3f}"
              f" tok=({float(tok[0]):+.3f},{float(tok[1]):+.3f},{float(tok[2]):.3f})"
              f" in_cell={bool(scene.token_in_target_cell()[0])}"
              f" score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, xyz, quat=None, vel=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor(xyz, device=device)
        if quat is None:
            st[:, 3] = 1.0
        else:
            st[:, 3:7] = torch.tensor(quat, device=device)
        if vel is not None:
            st[:, 7:10] = torch.tensor(vel, device=device)
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def set_yaw(psi: float, settle_steps: int = 30) -> None:
        """Probe-only: rotate the carousel ALONG ITS OWN revolute DOF (a pure-yaw
        root write about the joint axis is joint-consistent; the kinematic machine
        anchor is world-fixed)."""
        h = 0.5 * psi
        place(scene.carousel, (hx, hy, c.root_z),
              quat=(math.cos(h), 0.0, 0.0, math.sin(h)), settle_steps=settle_steps)

    def wrench_t(tau_z: float) -> None:
        """World z-torque on the carousel, expressed in its link frame (the house
        convention: is_global=True silently drops torques on this stack)."""
        from isaaclab.utils.math import quat_apply_inverse

        t3 = torch.tensor([0.0, 0.0, tau_z], device=device)
        q = scene.carousel.data.root_link_quat_w
        scene.carousel.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=device),
            quat_apply_inverse(q, t3.view(1, 3).expand(n, 3)).unsqueeze(1),
            env_ids=all_ids)

    def feed_token() -> None:
        """Solve's own transport: hover in free space above the chute collar, release
        with zero velocity, gravity + contact do the rest."""
        ap_x = hx + (c.ap_r_lo + c.ap_r_hi) / 2 * math.cos(c.ap_azimuth)
        ap_y = hy + (c.ap_r_lo + c.ap_r_hi) / 2 * math.sin(c.ap_azimuth)
        place(scene.token, (ap_x, ap_y, c.collar_top + 0.018), settle_steps=300)

    def cell_world_xy(i: int) -> tuple[float, float]:
        thw = float(scene.carousel_yaw()[0]) + i * math.pi / 2
        return (hx + c.cell_r * math.cos(thw), hy + c.cell_r * math.sin(thw))

    def in_any_cell() -> bool:
        return any(bool(scene.token_in_cell(i)[0]) for i in range(4))

    def layout_sane(tag: str) -> bool:
        """Reset honesty: token on the floor OFF the machine plate, carousel never
        aligned and still, the target's cue card in the tray, other cards parked."""
        t = int(scene.target[0])
        tok = rel(scene.token)
        card = rel(scene.cards[scene_mod.CELL_NAMES[t]])
        off_plate = (abs(float(tok[0]) - hx) > 0.315 or abs(float(tok[1]) - hy) > 0.315)
        others_parked = all(
            float(rel(scene.cards[nm])[0]) < -0.40
            for j, nm in enumerate(scene_mod.CELL_NAMES) if j != t)
        ok = (off_plate and float(tok[2]) < 0.05
              and abs(err_deg()) > c.align_tol_deg + 10.0
              and float(scene.carousel.data.root_ang_vel_w[0].norm()) < 0.10
              and abs(float(card[0]) - trx) < 0.03 and abs(float(card[1]) - try_) < 0.03
              and others_parked)
        if not ok:
            print(f"[smoke] layout sanity VIOLATION at {tag}: tok={tok.tolist()} "
                  f"err={err_deg():+.1f} card={card.tolist()}", flush=True)
        return ok

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    bodies = (scene.carousel, scene.token, *scene.cards.values())
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("settle: all states finite; token off-machine, carousel misaligned+still, "
          "matching cue card in the tray", fin and layout_sane("reset"))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    sane = True
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(30)
        sane = sane and layout_sane(f"seed {sd}")
        tok = rel(scene.token)
        reads.append((int(scene.target[0]), err_deg(), float(tok[0]), float(tok[1])))
    arr = np.array(reads)
    print("[smoke] randomization readback (target, err_deg, tok_x, tok_y):\n"
          f"{np.round(arr, 3)}", flush=True)
    targets = {int(x) for x in arr[:, 0]}
    err_spread = float(arr[:, 1].max() - arr[:, 1].min())
    min_abs_err = float(np.min(np.abs(arr[:, 1])))
    check("randomization: target color and carousel angle vary, never aligned at "
          f"spawn (readback: {len(targets)} colors, err spread {err_spread:.0f} deg, "
          f"min |err| {min_abs_err:.0f} deg)",
          len(targets) >= 2 and err_spread > 40.0
          and min_abs_err > c.align_tol_deg + 10.0)
    tok_spread = arr[:, 2:4].max(axis=0) - arr[:, 2:4].min(axis=0)
    check("randomization: token slot varies and every reset is sane (cue card "
          f"matches the target; readback token spread=({tok_spread[0]:.3f},"
          f"{tok_spread[1]:.3f}))",
          (tok_spread[0] > 0.03 or tok_spread[1] > 0.03) and sane)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. the seed family's move achieves nothing =================
    # rlbench/reach_target's verb is REACH: move to and touch the colored object.
    # Touch the token here — provably move it — and nothing scores.
    tok0 = rel(scene.token).clone()
    f3 = torch.tensor([1.0, 0.0, 0.0], device=device)
    from isaaclab.utils.math import quat_apply_inverse

    for _ in range(30):
        q = scene.token.data.root_link_quat_w
        scene.token.set_external_force_and_torque(
            quat_apply_inverse(q, f3.view(1, 3).expand(n, 3)).unsqueeze(1),
            torch.zeros(n, 1, 3, device=device), env_ids=all_ids)
        step(1)
    scene.token.set_external_force_and_torque(
        torch.zeros(n, 1, 3, device=device), torch.zeros(n, 1, 3, device=device),
        env_ids=all_ids)
    step(45)
    moved = float((rel(scene.token) - tok0)[0:2].norm())
    report("touch-token")
    s, ok = judge()
    check("seed-strategy analog: reaching out and touching/nudging the token (the "
          f"seed's whole goal) achieves nothing — token moved {moved * 1000:.0f} mm, "
          "score ~0", moved > 0.005 and s <= 0.02 and not ok)

    # =========================== 7. roof cover: no entry from above =========================
    for sd in (41, 42, 43, 44, 45):  # want the target cell well away from the chute
        env.reset(seed=sd)
        step(30)
        if abs(err_deg()) > 45.0:
            break
    assert abs(err_deg()) > 45.0, "probe setup: no misaligned-enough seed found"
    cxw, cyw = cell_world_xy(int(scene.target[0]))
    place(scene.token, (cxw, cyw, 0.24), settle_steps=150)
    report("roof-drop")
    tok = rel(scene.token)
    dist = math.hypot(float(tok[0]) - hx, float(tok[1]) - hy)
    s, ok = judge()
    check("roof cover: the token dropped from directly above the COVERED target cell "
          f"never enters it (ends z={float(tok[2]):.3f}, r={dist:.3f}: on the roof "
          "or off the rotor) — no credit",
          not in_any_cell() and (float(tok[2]) > 0.117 or dist > 0.17)
          and s <= 0.02 and not ok)

    # =========================== 8. wrong-cell chute drop (identity) ========================
    env.reset(seed=51)
    step(30)
    t = int(scene.target[0])
    wrong = (t + 1) % 4
    set_yaw((c.ap_azimuth - wrong * math.pi / 2) % (2 * math.pi))
    assert not bool(scene.aligned()[0]), "probe setup: target must NOT be aligned"
    feed_token()
    report("wrong-cell")
    s, ok = judge()
    check("wrong-cell chute drop: with the WRONG cell teleported under the chute, "
          "solve's own drop lands inside it (positive control: chute feeds the "
          f"aligned cell; in_cell[{wrong}]={bool(scene.token_in_cell(wrong)[0])}) — "
          "yet target credit is ZERO (identity, not geometry)",
          bool(scene.token_in_cell(wrong)[0])
          and not bool(scene.token_in_target_cell()[0])
          and float(scene.drop_latch[0]) == 0.0 and s <= 0.02 and not ok)

    # =========================== 9. alignment window ========================================
    env.reset(seed=61)
    step(30)
    t = int(scene.target[0])
    base_psi = c.ap_azimuth - t * math.pi / 2
    set_yaw(base_psi + math.radians(c.align_tol_deg + 5.0), settle_steps=20)
    near_ok = not bool(scene.aligned()[0]) and float(scene.align_latch[0]) == 0.0
    set_yaw(base_psi + math.radians(2.0), settle_steps=30)
    s, ok = judge()
    report("align-window")
    check("alignment window: 12 deg off is NOT aligned (no latch); 2 deg off held "
          f"still IS — align latch fires, score {s:.2f} = 0.25 exactly",
          near_ok and bool(scene.aligned()[0]) and float(scene.align_latch[0]) == 1.0
          and abs(s - 0.25) < 0.01 and not ok)

    # =========================== 12-13. settle gate + latched credit ========================
    # (same episode: alignment already latched at 0.25)
    cxw, cyw = cell_world_xy(t)
    place(scene.token, (cxw, cyw, c.deck_top + c.token_edge / 2 + 0.002),
          vel=(0.35, 0.0, 0.0), settle_steps=2)
    v_now = float(scene.token.data.root_lin_vel_w[0].norm())
    s_before, ok = judge()
    report("settle-gate")
    gate_ok = v_now > c.settle_lin and not ok and bool(scene.token_in_target_cell()[0])
    # remove it BEFORE it can settle in the cell (this battery must never succeed)
    place(scene.token, (0.85, 0.60, c.token_edge / 2 + 0.002), settle_steps=40)
    check("settle gate: the token IN the target cell window but moving at "
          f"{v_now:.2f} m/s is NOT success (velocity gates are real)", gate_ok)
    report("moved-away")
    s_after, ok = judge()
    check("latched credit: teleporting the token far away leaves the latched score "
          f"unchanged ({s_before:.2f} -> {s_after:.2f}, align 0.25 + drop 0.45) "
          "while token_in_target_cell() drops",
          abs(s_after - s_before) < 1e-3 and s_after >= 0.69
          and not bool(scene.token_in_target_cell()[0]) and not ok)

    # =========================== 10. drive-by sweep does not latch ==========================
    env.reset(seed=71)
    step(30)
    e0 = float(scene.align_err()[0])
    sgn = -1.0 if e0 > 0 else 1.0  # spin TOWARD (and through) alignment
    swept_window = False
    for _ in range(1500):
        e = float(scene.align_err()[0])
        w = float(scene.carousel.data.root_ang_vel_w[0, 2])
        swept_window = swept_window or bool(scene.aligned()[0])
        if swept_window and abs(e) > math.radians(30.0) and e * e0 < 0:
            break
        wrench_t(max(min(1.2 * (sgn * 0.8 - w), 0.25), -0.25))
        step(1)
    wrench_t(0.0)
    step(90)  # damping stops it wherever it coasts — well outside the window
    report("drive-by")
    s, ok = judge()
    check("drive-by sweep: spinning the rotor THROUGH the aligned window at ~0.8 "
          f"rad/s (window crossed: {swept_window}) latches NOTHING — final "
          f"err={err_deg():+.0f} deg, align latch 0, score ~0",
          swept_window and not bool(scene.aligned()[0])
          and float(scene.align_latch[0]) == 0.0 and s <= 0.02 and not ok)

    # =========================== 11. deck-rider between cells ===============================
    env.reset(seed=81)
    step(30)
    thw = float(scene.carousel_yaw()[0]) + math.pi / 4  # between cell 0 and cell 1
    place(scene.token, (hx + 0.11 * math.cos(thw), hy + 0.11 * math.sin(thw),
                        c.deck_top + c.token_edge / 2 + 0.002), settle_steps=60)
    report("deck-rider")
    tok = rel(scene.token)
    s, ok = judge()
    check("deck-rider: the token settled on the deck BETWEEN cells (z="
          f"{float(tok[2]):.3f}) rides the rotor but is inside NO cell — no credit",
          not in_any_cell() and float(tok[2]) < 0.09 and s <= 0.02 and not ok)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.color_carousel")
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

"""Smoke / rubric-REJECTION battery for DominoRelayScene (sim_gen task
`draw_svg_i271`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — build the chain, trigger it with one push,
hands-off cascade delivers the ball — is the acceptance evidence that the rubric
ACCEPTS a correct outcome). Every teleport here is instrumentation that
CONSTRUCTS a wrong (or partial) settled outcome and asserts the rubric REJECTS
it; no probe in this battery ever reaches success(), and a final audit asserts
exactly that.

Seed-strategy check: the seed (maniskill/draw_svg) is judged on the robot's own
CONTINUOUS TRACE — here nothing traces, so the analogous cheat surface is
"produce the final picture without the causal chain": every way of getting the
ball into the pocket, or dominoes onto the floor, WITHOUT a single pad-started
cascade must be rejected. The battery:

  1.  settle/premise    — states finite; dominoes verifiably FLAT in the depot,
                          ball verifiably SEATED on the pedestal; score ~0.
  2-3. randomization    — READBACK over 10 seeded resets from BODY poses: pad
                          centre moves and appears on both bench sides, run
                          bearing varies in magnitude AND sign, span sweeps its
                          band, the depot row swaps sides, staged domino yaws
                          vary; invariants hold (span in band, ball seated,
                          dominoes flat).
  4.  null policy       — 400 idle steps: dominoes stay flat (nothing ever
                          STOOD), the three-sided lip keeps the ball seated,
                          score ~0, no success.
  5.  dunk-only         — ball teleported straight into the pocket and settled
                          there: in-pocket and still, but NO cascade ever
                          happened -> ball_ok False, score ~0, NOT success.
  6.  held-in-air       — a domino held upright 6 cm ABOVE the bench for 25
                          steps (verifiably vertical the whole time): the
                          on-bench z-band keeps STOOD from latching.
  7.  fell-without-stood — a never-stood domino shoved flat near the pad: tilt
                          is far past fallen_deg yet NO fall event latches
                          (only stood-then-fell counts), score ~0.
  8.  piecemeal         — chain built (legitimate 0.24 stood credit), then the
                          dominoes knocked down ONE PER SECOND: all six fall
                          events latch, but the 5 s collapse window breaks
                          `window_s` -> relay_ok False, NOT success.
  9.  wrong start       — chain built, then triggered at the PEDESTAL END so
                          the real cascade runs backward: all six fall inside
                          the window, but the first faller is ~26 cm from the
                          pad (> start_r) -> relay_ok False, NOT success.
  10. relay without ball — chain built SHORT (last domino out of reach of the
                          pedestal), triggered at the pad: a fully valid relay
                          latches (relay_ok True) but the ball never moves ->
                          ball_ok False, NOT success, score capped.
  11. late hand delivery — continuing: >2.5 s after the last fall the ball is
                          teleported into the pocket and settles: ball entry
                          outside the delivery window -> ball_ok False, NOT
                          success (the cascade did not deliver it).
  12. ball before cascade — fresh episode: ball pre-dunked into the pocket,
                          THEN a full valid pad-started relay is run: every
                          success conjunct holds EXCEPT the delivery order
                          (ball_t < first fall) -> NOT success.
  13. latched credit    — after 12, all dominoes teleported back FLAT to the
                          depot: stood/fell/relay credit is latched, the score
                          does not move, still NOT success.
  14. rejection audit   — success() was never True at ANY judged point.
  15. final no-NaN      — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.draw_svg_i271.smoke --headless
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
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
# daemon=True: a crashed main thread must NOT idle until the timer fires.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

TILT_CUT = math.cos(math.radians(12.0))


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.domino_relay")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((0.95, -0.95, 0.85)) + o),
                                tuple(np.array((-0.05, 0.0, 0.12)) + o),
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
    ever_success = [False]

    def judge() -> tuple[float, bool]:
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return s, ok

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

    def report(tag: str) -> None:
        s, ok = judge()
        upz = scene._dom_up_z()[0]
        tilt = [f"{math.degrees(math.acos(max(-1.0, min(1.0, float(u))))):5.1f}" for u in upz]
        loc = scene.ball_local()[0]
        print(f"[smoke] {tag:18s} | tilt(deg)=[{' '.join(tilt)}] "
              f"was_up={scene.was_up[0].tolist()} "
              f"fell={[bool(v) for v in torch.isfinite(scene.fall_t[0])]} "
              f"ball=({float(loc[0]):+.3f},{float(loc[1]):+.3f},{float(loc[2]):+.3f}) "
              f"in_pocket={bool(scene.ball_in_pocket()[0])} relay={bool(scene.relay_ok()[0])} "
              f"ball_ok={bool(scene.ball_ok()[0])} success={ok} score={s:.3f} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    # ----- probe placement helpers (instrumentation, not a solution) ------------------------
    zero3 = torch.zeros(n, 1, 3, device=device)

    def push(j: int, newtons: float) -> None:
        f = torch.zeros(n, 1, 3, device=device)
        f[:, 0, 0] = newtons
        scene.doms[j].set_external_force_and_torque(f, zero3, env_ids=all_ids)

    def write_dom(j: int, xy_local: torch.Tensor, z: float, quat: torch.Tensor) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = scene.env_origins[:, :2] + xy_local
        st[:, 2] = scene.env_origins[:, 2] + z
        st[:, 3:7] = quat
        scene.doms[j].write_root_state_to_sim(st, all_ids)

    def dom_flat_depot(j: int) -> None:
        xy = torch.stack([
            torch.full((n,), c.depot_x0 + j * c.depot_dx, device=device),
            torch.sign(scene._dom_xy()[:, 0, 1].sign().sum() + 0.5)
            * torch.full((n,), c.depot_y, device=device)], dim=-1)
        q = scene_mod._qmul(scene_mod._qz(torch.zeros(n, device=device)),
                            scene_mod._qy(torch.full((n,), math.pi / 2, device=device)))
        write_dom(j, xy, c.bench_top + c.dom_t / 2 + 0.0005, q)

    def build_chain(standoff: float) -> bool:
        """Teleport the six dominoes upright along pad -> pedestal (the solve's
        transport step) and wait for the STOOD latches."""
        u = (scene.term_xy - scene.pad_xy) / scene.span.unsqueeze(-1)
        s = (scene.span - standoff) / (c.n_dominoes - 1)
        q_up = scene_mod._qz(scene.term_yaw)
        for i in range(c.n_dominoes):
            base = scene.pad_xy + (i * s).unsqueeze(-1) * u
            z = c.bench_top + c.dom_h / 2 + (0.0014 if i == 0 else 0.0005)
            write_dom(i, base, z, q_up)
            step(12)
        for _ in range(60):
            if bool(scene.was_up[0].all()):
                return True
            step(5)
        return bool(scene.was_up[0].all())

    def trigger(j: int, newtons: float) -> bool:
        """One push on domino j along its local +/-x, cut past the tipping angle."""
        push(j, newtons)
        tipped = False
        for _ in range(90):
            step(1)
            if float(scene._dom_up_z()[0, j]) < TILT_CUT:
                tipped = True
                break
        push(j, 0.0)
        return tipped

    def wait_all_fell(rounds: int = 30) -> bool:
        for _ in range(rounds):
            step(30)
            judge()
            if bool(torch.isfinite(scene.fall_t[0]).all()):
                return True
        return False

    def dunk_ball(lx: float = 0.11) -> None:
        """Teleport the ball into the pocket interior (terminal frame local +x)."""
        ca, sa = torch.cos(scene.term_yaw), torch.sin(scene.term_yaw)
        xy = scene.term_xy + torch.stack([ca * lx, sa * lx], dim=-1)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = scene.env_origins[:, :2] + xy
        st[:, 2] = scene.env_origins[:, 2] + c.bench_top + c.ball_r + 0.005
        st[:, 3] = 1.0
        scene.ball.write_root_state_to_sim(st, all_ids)

    def doms_flat_now() -> bool:
        return bool((scene._dom_up_z()[0].abs() < 0.25).all())

    def finite_all() -> bool:
        ok = True
        for b in [scene.pad, scene.terminal, scene.ball] + scene.doms:
            ok = ok and bool(torch.isfinite(b.data.root_state_w).all())
        return ok

    # =========================== 1. settle / premise ========================================
    env.reset(seed=41)
    step(60)
    report("reset")
    s, ok = judge()
    check("premise: states finite; all six dominoes verifiably FLAT in the depot "
          "and the ball verifiably SEATED on the pedestal after 0.5 s of settling; "
          "score ~0, no success",
          finite_all() and doms_flat_now() and bool(scene.ball_on_pedestal()[0])
          and s <= 0.02 and not ok)

    # =========================== 2-3. randomization is real =================================
    reads = []
    inv_ok = True
    for sd in (51, 52, 53, 54, 55, 56, 57, 58, 59, 60):
        env.reset(seed=sd)
        step(3)
        pad = (scene.pad.data.root_pos_w - scene.env_origins)[0]
        term = (scene.terminal.data.root_pos_w - scene.env_origins)[0]
        span_rb = float((term[:2] - pad[:2]).norm())
        bear_rb = math.atan2(float(term[1] - pad[1]), float(term[0] - pad[0]))
        dom0 = (scene.doms[0].data.root_pos_w - scene.env_origins)[0]
        # staged flat: the domino's local +z maps to a horizontal direction whose
        # azimuth IS the sampled yaw
        from isaaclab.utils.math import quat_apply
        ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
        zdir = quat_apply(scene.doms[0].data.root_quat_w, ez)[0]
        yaw_rb = math.atan2(float(zdir[1]), float(zdir[0]))
        reads.append((float(pad[0]), float(pad[1]), bear_rb, span_rb,
                      float(dom0[1]), yaw_rb))
        inv_ok = inv_ok and (c.span_min - 0.005 < span_rb < c.span_max + 0.005)
        inv_ok = inv_ok and bool(scene.ball_on_pedestal()[0]) and doms_flat_now()
        inv_ok = inv_ok and float(scene.score()[0]) <= 0.02
    arr = np.array(reads)
    print(f"[smoke] randomization readback (pad_x, pad_y, bearing, span, "
          f"depot0_y, dom0_yaw):\n{arr.round(3)}", flush=True)
    check("randomization: pad centre moves (x AND y jitter) and appears on both "
          "bench sides; run bearing varies in magnitude AND sign; span sweeps "
          "its band (all from BODY-pose readback)",
          arr[:, 0].ptp() > 0.01 and arr[:, 1].ptp() > 0.01
          and len(set(np.sign(arr[:, 1]))) == 2
          and arr[:, 2].ptp() > 0.10 and len(set(np.sign(arr[:, 2]))) == 2
          and arr[:, 3].ptp() > 0.02)
    check("randomization: the depot row swaps bench sides and staged domino yaws "
          "vary; invariants hold on every seed (span inside its band, ball "
          "seated, dominoes flat, score ~0)",
          len(set(np.sign(arr[:, 4]))) == 2 and arr[:, 5].ptp() > 1.0 and inv_ok)

    # =========================== 4. null policy fails =======================================
    env.reset(seed=61)
    step(400)
    report("null-policy")
    s, ok = judge()
    check("null policy: after 400 idle steps the dominoes still lie flat (no "
          "STOOD latch), the three-sided lip has kept the ball seated on the "
          "pedestal, score ~0, no success",
          doms_flat_now() and not bool(scene.was_up[0].any())
          and bool(scene.ball_on_pedestal()[0]) and s <= 0.02 and not ok)

    # =========================== 5. dunk-only ===============================================
    env.reset(seed=62)
    step(30)
    dunk_ball()
    for _ in range(10):
        step(30)
        judge()
        if int(scene.bin_streak[0]) >= c.still_steps:
            break
    report("dunk-only")
    s, ok = judge()
    check("dunk-only: ball teleported into the pocket rests there settled "
          f"(bin_streak {int(scene.bin_streak[0])} >= {c.still_steps}) — but no "
          "cascade ever happened: ball_ok False, score ~0, NOT success",
          bool(scene.ball_in_pocket()[0]) and int(scene.bin_streak[0]) >= c.still_steps
          and not bool(scene.ball_ok()[0]) and s <= 0.02 and not ok)

    # =========================== 6. held-in-air upright =====================================
    env.reset(seed=63)
    step(20)
    air_xy = torch.tensor([-0.45, 0.0], device=device).expand(n, 2)
    q_up = scene_mod._qz(torch.zeros(n, device=device))
    upz_min = 1.0
    for _ in range(2 * c.up_steps + 6):
        write_dom(2, air_xy, c.bench_top + c.dom_h / 2 + 0.06, q_up)
        step(1)
        upz_min = min(upz_min, float(scene._dom_up_z()[0, 2]))
    streak_air = int(scene.up_streak[0, 2])
    was_up_air = bool(scene.was_up[0, 2])
    # Park it flat again BEFORE releasing: a vertical prism dropped 6 cm lands on
    # its base and stands (which would latch a legitimate STOOD and pollute the
    # score) — the probe is about the hold, not the landing.
    dom_flat_depot(2)
    step(40)
    report("held-in-air")
    s, ok = judge()
    check("held-in-air: domino held VERIFIABLY vertical (min up_z "
          f"{upz_min:.3f} > {math.cos(math.radians(c.up_tol_deg)):.3f}) 6 cm above "
          f"the bench for {2 * c.up_steps + 6} steps — the on-bench z-band kept "
          f"STOOD from latching (streak {streak_air}, was_up {was_up_air}); "
          "score ~0",
          upz_min > math.cos(math.radians(c.up_tol_deg)) and streak_air == 0
          and not was_up_air and s <= 0.02 and not ok)

    # =========================== 7. fell-without-stood ======================================
    near_pad = scene.pad_xy + torch.tensor([0.06, 0.04], device=device).expand(n, 2)
    q_flat = scene_mod._qmul(scene_mod._qz(torch.full((n,), 0.4, device=device)),
                             scene_mod._qy(torch.full((n,), math.pi / 2, device=device)))
    write_dom(3, near_pad, c.bench_top + c.dom_t / 2 + 0.0005, q_flat)
    step(60)
    tilt3 = math.degrees(math.acos(max(-1.0, min(1.0, float(scene._dom_up_z()[0, 3])))))
    report("fell-no-stood")
    s, ok = judge()
    check("fell-without-stood: a never-stood domino laid flat near the pad sits "
          f"at {tilt3:.0f} deg (far past fallen_deg {c.fallen_deg:.0f}) yet NO "
          "fall event latches (only stood-then-fell counts); score ~0",
          tilt3 > c.fallen_deg + 20.0 and not bool(torch.isfinite(scene.fall_t[0, 3]))
          and s <= 0.02 and not ok)

    # =========================== 8. piecemeal knockdown =====================================
    env.reset(seed=64)
    step(30)
    built = build_chain(0.16)  # short chain: the ball must stay untouched here
    s_built, _ = judge()
    for j in range(c.n_dominoes):  # knock down one per second: window = 5 s > 3 s
        dom_flat_depot(j)
        step(120)
        judge()
    win = float(scene.fall_t[0].max() - scene.fall_t[0].min()) / 120.0
    report("piecemeal")
    s, ok = judge()
    check("piecemeal: chain built (legitimate stood credit "
          f"{s_built:.2f}); dominoes then knocked down one per second — all six "
          f"fall events latch but the {win:.1f} s collapse window breaks "
          f"window_s={c.window_s:.0f} s: relay_ok False, NOT success, score "
          "<= 0.48",
          built and abs(s_built - c.n_dominoes * c.w_stood) < 1e-3
          and bool(torch.isfinite(scene.fall_t[0]).all()) and win > c.window_s
          and not bool(scene.relay_ok()[0]) and not ok and s <= 0.485)

    # =========================== 9. wrong start (backward cascade) ==========================
    env.reset(seed=65)
    step(30)
    built = build_chain(0.065)
    tipped = trigger(c.n_dominoes - 1, -0.15)   # push the PEDESTAL-END domino backward
    fell_all = wait_all_fell()
    tmin = scene.fall_t[0].min()
    first = (scene.fall_t[0] == tmin) & torch.isfinite(scene.fall_t[0])
    d_first = float(scene.fall_dpad[0][first].min()) if bool(first.any()) else float("nan")
    win = float(scene.fall_t[0].max() - tmin) / 120.0
    report("wrong-start")
    s, ok = judge()
    check("wrong start: a REAL cascade triggered at the pedestal end runs "
          f"backward and fells all six inside {win:.1f} s, but the first faller "
          f"was {d_first * 1000:.0f} mm from the pad (> start_r "
          f"{c.start_r * 1000:.0f} mm): relay_ok False, NOT success",
          built and tipped and fell_all and win <= c.window_s
          and d_first > c.start_r and not bool(scene.relay_ok()[0]) and not ok)

    # =========================== 10. valid relay, no ball ===================================
    env.reset(seed=66)
    step(30)
    built = build_chain(0.16)   # short: last domino cannot reach the pedestal
    tipped = trigger(0, 0.12)
    fell_all = wait_all_fell()
    report("relay-no-ball")
    s, ok = judge()
    relay_now = bool(scene.relay_ok()[0])
    check("relay without ball: a SHORT chain (last domino out of pedestal reach) "
          "triggered at the pad latches a fully valid relay (relay_ok True) — "
          "but the ball never left its pedestal: ball_ok False, NOT success, "
          "score <= 0.60",
          built and tipped and fell_all and relay_now
          and bool(scene.ball_on_pedestal()[0]) and not bool(scene.ball_ok()[0])
          and not ok and s <= 0.60)

    # =========================== 11. late hand delivery =====================================
    step(int(c.ball_window_s * 120) + 90)   # let the delivery window expire
    dunk_ball()
    for _ in range(10):
        step(30)
        judge()
        if int(scene.bin_streak[0]) >= c.still_steps:
            break
    late = (float(scene.ball_t[0]) - float(scene.fall_t[0].max())) / 120.0
    report("late-delivery")
    s, ok = judge()
    check("late hand delivery: ball teleported into the pocket "
          f"{late:.1f} s after the last fall (> ball_window_s "
          f"{c.ball_window_s:.1f} s) rests settled in the pocket with a valid "
          "relay latched — ball_ok STILL False, NOT success",
          bool(scene.relay_ok()[0]) and bool(scene.ball_in_pocket()[0])
          and int(scene.bin_streak[0]) >= c.still_steps and late > c.ball_window_s
          and not bool(scene.ball_ok()[0]) and not ok)

    # =========================== 12. ball before cascade ====================================
    env.reset(seed=67)
    step(30)
    dunk_ball()
    step(90)                      # ball_t latches long before any fall
    built = build_chain(0.065)
    tipped = trigger(0, 0.12)
    fell_all = wait_all_fell()
    for _ in range(6):
        step(30)
        judge()
    order = float(scene.ball_t[0]) - float(scene.fall_t[0].min())
    report("ball-first")
    s, ok = judge()
    check("ball before cascade: ball pre-dunked, then a FULL valid pad-started "
          "relay runs — everything is down, the ball rests settled in the "
          f"pocket, relay_ok True, yet the delivery ORDER is wrong (ball entry "
          f"{-order / 120.0:.1f} s before the first fall): ball_ok False, NOT "
          "success",
          built and tipped and fell_all and bool(scene.relay_ok()[0])
          and bool(scene.ball_in_pocket()[0])
          and int(scene.bin_streak[0]) >= c.still_steps and order < 0
          and not bool(scene.ball_ok()[0]) and not ok)

    # =========================== 13. latched credit survives regression =====================
    s_before, _ = judge()
    for j in range(c.n_dominoes):
        dom_flat_depot(j)
    step(60)
    s_after, ok = judge()
    report("regressed")
    check("latched credit: teleporting all dominoes back flat to the depot after "
          f"the relay leaves the latched score unchanged ({s_before:.2f} -> "
          f"{s_after:.2f}), still NOT success",
          abs(s_after - s_before) < 1e-6 and s_before >= 0.55 and not ok)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.domino_relay")
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
    except Exception:  # noqa: BLE001 — die fast, don't let Kit teardown hang
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)

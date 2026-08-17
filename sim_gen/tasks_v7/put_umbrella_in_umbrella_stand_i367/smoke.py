"""Smoke / rubric-REJECTION battery for UmbrellaBayonetScene (sim_gen task
`put_umbrella_in_umbrella_stand_i367`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — pull the transit pin, align the lug to the keyed
gap, hoist through, twist 90 deg, seat the lug on the ring — is the acceptance
evidence; it passes on forge seeds). Every teleport here is instrumentation that
CONSTRUCTS a wrong (or partial) outcome and asserts the rubric REJECTS it — plus
physical probes that prove the pin-jam and the keyed shelf are working mechanisms,
not props. No probe in this battery ever reaches success(), and a final audit check
asserts exactly that.

   1. settle/no-NaN      — reset settles finite: runner on the bottom stop, pin
                           seated on the channel floor, spare on the ground, settled;
   2. fresh reset        — score ~0, no success;
  3-4. randomization     — READBACK over 6 seeded resets: runner yaw varies; pin
                           seat depth varies; spare pin position + yaw vary;
   5. null policy        — 240 idle steps -> score ~0, no success;
   6. blocked hoist      — the solve's own hoist servo with the pin still SEATED:
                           the track physically jams just above q_block (q readback;
                           the servo genuinely pushed — vacuous-probe guard), the
                           pass latch never fires, score ~0: pin-first order is
                           physics-forced;
   7. pin extraction     — the solve's own pull servo extracts the pin (mechanism
                           proof); partial credit 0.30, NOT success;
   8. untwisted release  — runner released over the gap at the GAP heading falls
                           all the way back down (q readback -> ~0): hoisting
                           without the twist parks nothing;
   9. yaw-margin miss    — runner dropped onto the ring at 45 deg heading GENUINELY
                           PARKS (q ~ q_park, settled) but 45 < align_min 55 -> the
                           alignment clause alone rejects it;
  10. below-shelf twist  — runner twisted to 90 deg BELOW the shelf falls to the
                           bottom: the park band is above the plate only;
  11. latched credit     — after all those falls the 0.30+0.30 credit persists
                           (latches never evaporate) yet success stays False;
  12. teleport-to-park   — FRESH episode (latches cleared), runner teleported
                           directly into the park pose (rests, aligned, settled —
                           looks perfect) is REJECTED: neither the pin latch nor the
                           pass latch ever fired, score ~0;
  13. wrong object       — the identical SPARE pin moved clear of the mast earns
                           nothing (the pin latch reads the transit pin only);
  14. settle gate        — a genuinely constructed park (pin honestly extracted,
                           runner dropped onto the ring at ~88 deg) judged WHILE
                           STILL RINGING: parked & aligned & latched all read True
                           yet success is False (stillness must persist); the runner
                           is removed before it settles (the battery never succeeds);
  15. rejection audit    — success() was never True at ANY judged point;
  16. final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.put_umbrella_in_umbrella_stand_i367.smoke --headless
"""

from __future__ import annotations

import argparse
import math

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

import os
import threading

import numpy as np
import torch

import robobench
from robobench.core import ENVS

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:  # older isaaclab name
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

G = 9.81


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.umbrella_bayonet")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.15, -0.90, 0.80)) + o),
                                tuple(np.array((0.00, 0.00, 0.35)) + o),
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

    def q0() -> float:
        return float(scene.runner_q()[0])

    def yaw0() -> float:
        qq = scene.runner.data.root_quat_w[0]
        dx = 1.0 - 2.0 * (float(qq[2]) ** 2 + float(qq[3]) ** 2)
        dy = 2.0 * (float(qq[1]) * float(qq[2]) + float(qq[0]) * float(qq[3]))
        return math.atan2(dy, dx)

    def pd0() -> float:
        return float(scene.pin_dist()[0])

    def loc(body) -> tuple[float, float, float]:
        p = (body.data.root_pos_w - scene.env_origins)[0]
        return float(p[0]), float(p[1]), float(p[2])

    def report(tag: str) -> None:
        s, ok = judge()
        print(f"[smoke] {tag:16s} | q={q0():+.4f} yaw={yaw0():+.3f} pin_d={pd0():+.4f} "
              f"parked={bool(scene.parked_now()[0])} aligned={bool(scene.aligned_now()[0])} "
              f"settled={bool(scene.settled()[0])} | p/p={float(scene.pin_latch[0]):.0f}/"
              f"{float(scene.pass_latch[0]):.0f} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def write_runner(q: float, yaw: float, settle_steps: int = 60) -> None:
        """Teleport the runner ALONG ITS TRACK (x=y=0, pure yaw — joint-consistent;
        instrumentation, not a solution) + REAL physics steps (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.env_origins
        st[:, 2] += c.runner_z0 + float(q)
        st[:, 3] = math.cos(yaw / 2)
        st[:, 6] = math.sin(yaw / 2)
        scene.runner.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def write_body(body, local_xyz, settle_steps: int = 60) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.env_origins
        st[:, 0] += float(local_xyz[0])
        st[:, 1] += float(local_xyz[1])
        st[:, 2] += float(local_xyz[2])
        st[:, 3] = 1.0
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def hoist_servo(q_tgt: float, max_steps: int) -> float:
        """The solve's own heave servo (gravity ff + velocity regulation; yaw held).
        Returns the max q reached; leaves the wrench OFF."""
        f = torch.zeros(n, 1, 3, device=device)
        tq = torch.zeros(n, 1, 3, device=device)
        yaw_hold = yaw0()
        q_max = q0()
        for _ in range(max_steps):
            q = scene.runner_q()
            vz = scene.runner.data.root_lin_vel_w[:, 2]
            wz = scene.runner.data.root_ang_vel_w[:, 2]
            v_des = (3.0 * (q_tgt - q)).clamp(-0.15, 0.15)
            f[:, 0, 2] = (c.runner_mass * G + 6.0 * (v_des - vz)).clamp(0.4, 6.0)
            qq = scene.runner.data.root_quat_w
            dx = 1.0 - 2.0 * (qq[:, 2] ** 2 + qq[:, 3] ** 2)
            dy = 2.0 * (qq[:, 1] * qq[:, 2] + qq[:, 0] * qq[:, 3])
            e = (yaw_hold - torch.atan2(dy, dx) + math.pi) % (2 * math.pi) - math.pi
            tq[:, 0, 2] = (0.008 * ((3.0 * e).clamp(-1.2, 1.2) - wz)).clamp(-0.02, 0.02)
            scene.runner.set_external_force_and_torque(f, tq, env_ids=all_ids)
            env.step(no_action)
            q_max = max(q_max, q0())
        scene.runner.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
        return q_max

    def pull_pin(max_steps: int = 1200) -> int:
        """The solve's own HELD pin-pull servo: mg feedforward + z/y PD at the CoM
        cancels the pitch moment (a bare horizontal pull drawer-jams once the CoM
        passes the channel-floor edge), x velocity servo slides the pin out.
        Returns servo steps used; leaves the wrench OFF."""
        zr = c.pin_rest_z + 0.0005
        done, i = 0, 0
        for i in range(max_steps):
            p = scene.pin.data.root_pos_w - scene.env_origins
            v = scene.pin.data.root_lin_vel_w
            x, vx = p[:, 0], v[:, 0]
            v_des = (3.0 * (0.20 - x)).clamp(-0.12, 0.12)
            fx = 2.0 * (v_des - vx)
            fx = fx + torch.where(vx.abs() < 0.02, 0.35 * v_des.sign(),
                                  torch.zeros_like(fx))
            f_world = torch.zeros(n, 3, device=device)
            f_world[:, 0] = fx.clamp(-1.5, 1.5)
            f_world[:, 1] = (4.0 * (0.0 - p[:, 1]) - 0.4 * v[:, 1]).clamp(-0.5, 0.5)
            f_world[:, 2] = (c.pin_mass * G + 4.0 * (zr - p[:, 2])
                             - 0.4 * v[:, 2]).clamp(0.0, 1.5)
            f_body = quat_apply_inverse(scene.pin.data.root_quat_w, f_world)
            scene.pin.set_external_force_and_torque(f_body.unsqueeze(1), zero_w,
                                                    env_ids=all_ids)
            env.step(no_action)
            done = done + 1 if pd0() >= c.pull_clear else 0
            if done >= 3:
                break
        scene.pin.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
        step(90)
        return i + 1

    bodies = (scene.base, scene.mast, scene.runner, scene.pin, scene.spare)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    px, py, pz = loc(scene.pin)
    sx, sy, sz = loc(scene.spare)
    check("settle: all states finite, runner on the bottom stop "
          f"(q={q0():+.4f}), pin seated on the channel floor (z={pz:+.4f} ~ "
          f"{c.pin_rest_z:+.4f}, d={pd0():+.4f}), spare on the ground (z={sz:+.3f}), "
          "everything settled",
          fin and abs(q0()) <= 0.008 and abs(pz - c.pin_rest_z) <= 0.004
          and pd0() <= 0.02 and sz <= 0.05 and bool(scene.settled()[0]))
    s, ok = judge()
    check("fresh reset: score ~0, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(20)
        px, _, _ = loc(scene.pin)
        sx, sy, _ = loc(scene.spare)
        reads.append((yaw0(), px, sx, sy))
        print(f"[smoke] seed {sd}: runner_yaw={yaw0():+.3f} pin_x={px:+.4f} "
              f"spare=({sx:+.3f},{sy:+.3f})", flush=True)
    yaw_spread = max(r[0] for r in reads) - min(r[0] for r in reads)
    pin_spread = max(r[1] for r in reads) - min(r[1] for r in reads)
    check("randomization: runner yaw varies (readback spread "
          f"{yaw_spread:.2f} rad) and pin seat depth varies (x spread "
          f"{pin_spread * 1000:.1f} mm)", yaw_spread > 0.8 and pin_spread > 0.002)
    sx_spread = max(r[2] for r in reads) - min(r[2] for r in reads)
    sy_spread = max(r[3] for r in reads) - min(r[3] for r in reads)
    check("randomization: spare pin position varies (readback spread "
          f"({sx_spread * 1000:.0f},{sy_spread * 1000:.0f}) mm)",
          sx_spread + sy_spread > 0.04)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 6. blocked hoist (pin seated) ==============================
    env.reset(seed=41)
    step(60)
    q_max = hoist_servo(c.hoist_q, 420)
    q_jam = q0()
    step(90)  # wrench off -> gravity returns the runner to the bottom stop
    report("blocked-hoist")
    s, ok = judge()
    check("blocked hoist: the solve's own hoist servo with the pin SEATED genuinely "
          f"pushed (q_max={q_max:+.4f} > 0.03) but the track jams just above "
          f"q_block={c.q_block:+.4f} (q_max <= {c.q_block + 0.014:+.4f}), the pass "
          "latch never fires, score ~0 — pin-first order is physics-forced",
          q_max > 0.03 and q_max <= c.q_block + 0.014
          and float(scene.pass_latch[0]) < 0.5 and s <= 0.02 and not ok)
    print(f"[smoke] blocked-hoist: wrench off -> runner fell back to q={q0():+.4f} "
          f"(jammed at {q_jam:+.4f})", flush=True)

    # =========================== 7. pin extraction (mechanism + partial credit) =============
    used = pull_pin()
    report("pin-out")
    s, ok = judge()
    check("pin extraction: the solve's own pull servo frees the pin "
          f"({used} servo steps, pin_d={pd0():+.4f} >= 0.12) — partial credit 0.30, "
          "NOT success", pd0() >= 0.12 and 0.28 <= s <= 0.32 and not ok)

    # =========================== 8. untwisted release over the gap ==========================
    write_runner(0.39, 0.0, settle_steps=240)
    report("untwisted")
    s, ok = judge()
    check("untwisted release: runner released above the shelf AT the gap heading "
          f"falls all the way back down (q readback {q0():+.4f} ~ 0) — hoisting "
          "without the twist parks nothing, NOT success",
          q0() <= 0.02 and not ok)

    # =========================== 9. yaw-margin near-miss ====================================
    write_runner(c.q_park + 0.020, math.radians(45.0), settle_steps=240)
    report("yaw-miss")
    s, ok = judge()
    check("yaw-margin near-miss: dropped onto the ring at 45 deg it GENUINELY parks "
          f"(q={q0():+.4f} ~ q_park {c.q_park:+.4f}, settled="
          f"{bool(scene.settled()[0])}) but 45 < align_min {c.align_min_deg:.0f} deg "
          "-> rejected by the alignment clause alone",
          abs(q0() - c.q_park) <= 0.010 and bool(scene.settled()[0])
          and not bool(scene.aligned_now()[0]) and not ok)

    # =========================== 10. below-shelf twist ======================================
    write_runner(c.q_park - 0.030, math.pi / 2, settle_steps=240)
    report("below-shelf")
    s, ok = judge()
    check("below-shelf twist: runner twisted to 90 deg with the lug UNDER the plate "
          f"falls to the bottom (q readback {q0():+.4f} ~ 0) — the park band exists "
          "only ON the ring, NOT success", q0() <= 0.02 and not ok)

    # =========================== 11. latched credit persists ================================
    s, ok = judge()
    check("latched credit: after every fall the 0.30+0.30 credit persists "
          f"(score={s:.2f}) yet success stays False", 0.55 <= s <= 0.65 and not ok)

    # =========================== 12. teleport-to-park cheat =================================
    env.reset(seed=51)  # FRESH episode: latches cleared, pin back in its seat
    step(60)
    write_runner(c.q_park + 0.001, math.pi / 2, settle_steps=240)
    report("cheat-park")
    s, ok = judge()
    check("teleport-to-park cheat: a FRESH episode's runner teleported straight into "
          f"the park pose rests there looking perfect (q={q0():+.4f}, aligned="
          f"{bool(scene.aligned_now()[0])}, settled={bool(scene.settled()[0])}) but "
          "neither the pin latch nor the pass latch ever fired -> score ~0, NOT success",
          bool(scene.parked_now()[0]) and bool(scene.aligned_now()[0])
          and bool(scene.settled()[0]) and float(scene.pin_latch[0]) < 0.5
          and float(scene.pass_latch[0]) < 0.5 and s <= 0.02 and not ok)

    # =========================== 13. wrong object ===========================================
    write_body(scene.spare, (0.15, 0.05, 0.03), settle_steps=90)
    report("wrong-object")
    s, ok = judge()
    check("wrong object: the identical SPARE pin moved clear of the mast earns "
          f"nothing (transit pin still seated, pin_d={pd0():+.4f}; score={s:.2f}) — "
          "the pin latch reads the transit pin only",
          pd0() <= 0.02 and float(scene.pin_latch[0]) < 0.5 and s <= 0.02 and not ok)

    # =========================== 14. settle gate ============================================
    env.reset(seed=61)
    step(60)
    pull_pin()  # honest extraction (latch 0.30)
    write_runner(c.q_park + 0.015, math.radians(88.0), settle_steps=0)
    step(12)  # it lands on the ring and is still RINGING; judge NOW
    parked_now = bool(scene.parked_now()[0])
    aligned_now = bool(scene.aligned_now()[0])
    latched = float(scene.pin_latch[0]) > 0.5 and float(scene.pass_latch[0]) > 0.5
    settled_now = bool(scene.settled()[0])
    s, ok = judge()
    report("settle-gate")
    # remove it BEFORE it settles (the battery must never reach success)
    write_runner(0.002, 0.0, settle_steps=30)
    check("settle gate: a genuinely constructed park judged while still ringing "
          f"(parked={parked_now}, aligned={aligned_now}, latched={latched}, "
          f"settled={settled_now}) is NOT success; runner removed before ring-down "
          "(battery never succeeds)",
          parked_now and aligned_now and latched and not settled_now and not ok
          and not bool(scene.success()[0]))

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.umbrella_bayonet")
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

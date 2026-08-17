"""Smoke / rubric-REJECTION battery for LanternShelterScene (sim_gen task
`light_bulb_in_i219`) — NullRobot, teleported probe states + force probes, RECORDED.

This is NOT a solution (solve.py — roll the globe to the doorway, shove it over the
four-post entry barrier, slide the shutter closed — is the acceptance evidence that
the rubric ACCEPTS a correct outcome; it passes on seeds 0/1). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled state and
asserts the rubric REJECTS it — plus applied-force probes that prove the entry
barrier and the roof denial are physically real geometry. No probe in this battery
ever reaches success(), and a final audit asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: globe + decoy on the open
                            floor outside, shutter parked fully open in its channel,
                            seat empty; score ~0 at rest, no success;
  3-4. randomization      — READBACK over 8 seeded resets: shelter xy + yaw vary and
                            the shutter TRACKS the shelter channel; globe/decoy
                            spawns vary and the decoy side flips;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  roof denial         — the seed's direct move — deliver the bulb from above
                            onto the socket — is physically unavailable: dropped
                            over the seat it lands on the ROOF (or rolls off
                            outside) and never reaches the posts, score ~0;
  7.  entry barrier       — sub-threshold retention: a quasi-static creep push
                            (<= 0.30 N < the ~0.52 N barrier force, 0.12 m/s <<
                            the ~0.38 m/s vault speed) reaches the posts (>= 20 mm
                            travel, non-vacuous) but STALLS before the `entered`
                            plane and rolls back on release — never seated, ~0;
  8.  doorway lodge       — globe constructed resting IN the doorway throat: not
                            entered, not seated, score ~0 (merely reaching the
                            doorway earns nothing);
  9.  seated-but-open     — globe constructed seated on the posts with the shutter
                            still parked open: enter+seat credit only (0.50 + eps),
                            NOT success — closure is required;
  10. closed-empty        — shutter slid closed on an EMPTY shelter: door_closed
                            reads True but earns nothing (score ~0), no success;
  11. wrong object        — the graspable 48 mm decoy dropped over the cradle falls
                            straight THROUGH the four posts to the floor (readback
                            z ~ 24 mm), never seats, score ~0;
  12. settle gate         — globe seated + shutter closed but still sliding fast is
                            NOT success (velocity gates are real) while the
                            shuttered-in latch legitimately fires;
  13. cap                 — all three latches constructed, then the globe yanked
                            away: score == 0.75 cap (float32 + eps), NOT success;
  14. latched credit      — 40 further steps: score unchanged, in_seat stays False;
  15. rejection audit     — success() was never True at ANY judged point;
  16. final no-NaN        — all task-object states finite at the end;
  17. camera              — >= 20 rgb frames captured -> frames.npz.

Run (forge): python -u -m simgen_tasks.light_bulb_in_i219.smoke --headless
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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.lantern_shelter")().build(num_envs=args.num_envs,
                                                     device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    R = c.globe_r
    reach = R + c.post_r
    diag = math.hypot(c.post_xy, c.post_xy)
    seat_z = c.post_z + math.sqrt(reach**2 - diag**2)  # ~0.0487

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.65, -1.25, 1.00)) + o),
                                tuple(np.array((0.50, 0.00, 0.10)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video",
              flush=True)

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

    def report(tag: str) -> None:
        s, ok = judge()
        g = scene.globe_local()[0]
        d = scene.shutter_local()[0]
        print(f"[smoke] {tag:16s} |"
              f" globe_loc=({float(g[0]):+.3f},{float(g[1]):+.3f},{float(g[2]):+.3f})"
              f" door_y={float(d[1]):+.3f}"
              f" in_seat={bool(scene.in_seat()[0])}"
              f" closed={bool(scene.door_closed()[0])}"
              f" score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def sh(off_xyz) -> tuple[float, float, float]:
        """Env-relative world position of a shelter-frame offset."""
        off = torch.tensor(off_xyz, device=device).view(1, 3)
        p = scene.shelter.data.root_pos_w[:1] + quat_apply(
            scene.shelter.data.root_quat_w[:1], off)
        p = (p - scene.env_origins[:1])[0]
        return (float(p[0]), float(p[1]), float(p[2]))

    def place(body, xyz, quat=None, vel_local=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap). `xyz` is env-relative;
        `vel_local` is a shelter-frame velocity."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor(xyz, device=device)
        if quat is None:
            st[:, 3] = 1.0
        else:
            st[:, 3:7] = quat
        if vel_local is not None:
            v = torch.tensor(vel_local, device=device).view(1, 3)
            st[:, 7:10] = quat_apply(scene.shelter.data.root_quat_w[:1], v)
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def force_world(body, f_w: torch.Tensor) -> None:
        """World force at the CoM, encoded into the CURRENT body frame (the house
        convention — critical for the rolling globe)."""
        body.set_external_force_and_torque(
            quat_apply_inverse(body.data.root_link_quat_w,
                               f_w.view(1, 3).expand(n, 3)).unsqueeze(1),
            torch.zeros(n, 1, 3, device=device), env_ids=all_ids)

    def globe_creep(v_des_local, cap: float, steps: int):
        """Velocity-servo push on the globe with a hard force cap; returns the
        maximum shelter-frame x and z reached DURING the push (sample the peak —
        gravity restores the shoved state before any post-settle readback)."""
        kv = 6.0
        max_x, max_z, seated_ever, moved0 = -1e9, -1e9, False, None
        for _ in range(steps):
            g = scene.globe_local()[0]
            if moved0 is None:
                moved0 = float(g[0])
            v_des_w = quat_apply(scene.shelter.data.root_quat_w[:1],
                                 torch.tensor(v_des_local, device=device).view(1, 3))[0]
            f = kv * (v_des_w - scene.globe.data.root_lin_vel_w[0])
            f[2] = 0.0
            fn = float(f.norm())
            if fn > cap:
                f = f * (cap / fn)
            force_world(scene.globe, f)
            step(1)
            g = scene.globe_local()[0]
            max_x = max(max_x, float(g[0]))
            max_z = max(max_z, float(g[2]))
            seated_ever = seated_ever or bool(scene.in_seat()[0])
        force_world(scene.globe, torch.zeros(3, device=device))
        return max_x, max_z, seated_ever, moved0

    def layout_sane(tag: str) -> bool:
        """Reset honesty: globe and decoy resting on the floor outside, shutter
        parked fully open in its channel, seat empty, doorway clear."""
        g = scene.globe_local()[0]
        dc = scene.decoy_local()[0]
        d = scene.shutter_local()[0]
        ok = (c.globe_gx[0] - 0.02 <= float(g[0]) <= c.globe_gx[1] + 0.02
              and abs(float(g[1])) <= c.globe_gy[1] + 0.02
              and abs(float(g[2]) - R) < 0.006
              and c.decoy_dx[0] - 0.02 <= float(dc[0]) <= c.decoy_dx[1] + 0.02
              and c.decoy_dy[0] - 0.02 <= abs(float(dc[1])) <= c.decoy_dy[1] + 0.02
              and abs(float(dc[2]) - c.decoy_r) < 0.006
              and abs(float(d[0]) - c.door_x) < 0.004
              and c.door_open[0] - 0.012 <= float(d[1]) <= c.door_open[1] + 0.012
              and abs(float(d[2]) - c.door_z) < 0.008
              and float(d[1]) - c.door_half_w >= c.doorway_half - 0.005
              and not bool(scene.in_seat()[0])
              and not bool(scene.door_closed()[0]))
        if not ok:
            print(f"[smoke] layout sanity VIOLATION at {tag}", flush=True)
        return ok

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    bodies = (scene.shelter, scene.shutter, scene.globe, scene.decoy)
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("settle: all states finite; globe + decoy on the open floor outside, "
          "shutter parked fully open, seat empty", fin and layout_sane("reset"))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    sane = True
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(30)
        sane = sane and layout_sane(f"seed {sd}")
        hp = (scene.shelter.data.root_pos_w - scene.env_origins)[0]
        hq = scene.shelter.data.root_quat_w[0]
        hyaw = 2.0 * math.atan2(float(hq[3]), float(hq[0]))
        g = scene.globe_local()[0]
        dc = scene.decoy_local()[0]
        d = scene.shutter_local()[0]
        reads.append((float(hp[0]), float(hp[1]), hyaw, float(d[1]),
                      float(g[0]), float(g[1]), float(dc[1])))
    arr = np.array(reads)
    print("[smoke] randomization readback (house_x, house_y, yaw, door_y, globe_x, "
          f"globe_y, decoy_y):\n{np.round(arr, 4)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: shelter pose varies and the shutter tracks its channel "
          f"(readback spread x={spread[0]:.3f}, y={spread[1]:.3f}, "
          f"yaw={math.degrees(spread[2]):.1f} deg, door_open={spread[3]:.3f})",
          sane and spread[0] > 0.02 and spread[1] > 0.02
          and spread[2] > math.radians(4.0) and spread[3] > 0.004)
    sides = {1 if v > 0 else -1 for v in arr[:, 6]}
    check("randomization: globe spawn varies and the decoy side flips (readback "
          f"globe spread ({spread[4]:.3f},{spread[5]:.3f}), {len(sides)} decoy sides)",
          spread[4] > 0.02 and spread[5] > 0.03 and len(sides) == 2)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 6. roof denial (seed-strategy family) ======================
    # rlbench/light_bulb_in's verb is carry-the-bulb-to-the-fixture-and-insert-from-
    # above. The carry itself is already denied statically (90 mm globe > 80 mm jaw
    # span, asserted in cfg); here the aerial DELIVERY is force-probed: released
    # directly above the socket, the globe lands on the roof and never reaches it.
    env.reset(seed=41)
    step(30)
    over = sh((0.0, 0.0, 0.0))
    place(scene.globe, (over[0], over[1], 0.45), settle_steps=300)
    report("roof-drop")
    gz = float((scene.globe.data.root_pos_w - scene.env_origins)[0][2])
    s, ok = judge()
    check("roof denial: the globe dropped from directly above the socket lands on "
          f"the roof / rolls off outside (z={gz:.3f}), never seats, score ~0",
          gz < 0.40 and not bool(scene.in_seat()[0]) and s <= 0.02 and not ok)

    # =========================== 7. entry barrier: sub-threshold creep ======================
    # Quasi-static approach (the static-threshold rule: fast slews would vault):
    # transport-roll to the staging point, then creep at 0.12 m/s with the force
    # capped at 0.30 N — below the ~0.52 N ~ m g tan(slope) barrier force and far
    # below the ~0.38 m/s vault speed. The globe must reach the posts and stall.
    env.reset(seed=51)
    step(30)
    place(scene.globe, sh((-0.20, 0.0, R + 0.002)), settle_steps=30)  # transport only
    max_x, max_z, seated_ever, x0 = globe_creep((0.12, 0.0, 0.0), cap=0.30, steps=480)
    step(180)  # release: the barrier slope rolls it back
    report("creep-stall")
    g = scene.globe_local()[0]
    s, ok = judge()
    check("entry barrier: the capped 0.30 N creep travels "
          f"{(max_x - x0) * 1000:.0f} mm (>= 20, non-vacuous) to the posts but "
          f"stalls at x={max_x * 1000:.0f} mm — short of the entered plane "
          f"({c.enter_x * 1000:.0f} mm) — never seats (max z={max_z * 1000:.1f} mm), "
          f"and rolls back on release (rest x={float(g[0]) * 1000:.0f} mm, "
          f"z={float(g[2]) * 1000:.1f} mm): score ~0",
          max_x - x0 >= 0.020 and max_x < c.enter_x and not seated_ever
          and not bool(scene.in_seat()[0]) and float(g[2]) < c.seat_z_lo
          and s <= 0.02 and not ok)

    # =========================== 8. doorway lodge ===========================================
    env.reset(seed=61)
    step(30)
    place(scene.globe, sh((-0.058, 0.0, R + 0.002)), settle_steps=60)
    report("doorway-lodge")
    g = scene.globe_local()[0]
    s, ok = judge()
    check("doorway lodge: globe resting in the doorway throat (local x="
          f"{float(g[0]) * 1000:.0f} mm) is neither entered nor seated — merely "
          "reaching the doorway earns nothing, score ~0",
          float(g[0]) < c.enter_x and not bool(scene.in_seat()[0])
          and s <= 0.02 and not ok)

    # =========================== 9. seated-but-open =========================================
    env.reset(seed=71)
    step(30)
    place(scene.globe, sh((0.0, 0.0, seat_z + 0.003)), settle_steps=90)
    report("seated-open")
    s, ok = judge()
    check("seated-but-open: globe constructed seated on the posts with the shutter "
          f"parked open earns enter+seat credit only (score {s:.3f} <= 0.50 + eps), "
          "NOT success — closure is required",
          bool(scene.in_seat()[0]) and not bool(scene.door_closed()[0])
          and 0.495 <= s <= 0.50 + 1e-5 and not ok)

    # =========================== 10. closed-empty ===========================================
    env.reset(seed=81)
    step(30)
    place(scene.shutter, sh((c.door_x, 0.0, c.door_z + 0.001)),
          quat=scene.shelter.data.root_quat_w[:1], settle_steps=60)
    report("closed-empty")
    s, ok = judge()
    check("closed-empty: the shutter slid closed on an EMPTY shelter reads "
          "door_closed but earns nothing (score ~0), no success",
          bool(scene.door_closed()[0]) and not bool(scene.in_seat()[0])
          and s <= 0.02 and not ok)

    # =========================== 11. wrong object: the decoy falls through ==================
    env.reset(seed=91)
    step(30)
    place(scene.decoy, sh((0.0, 0.0, 0.100)), settle_steps=120)
    report("decoy-drop")
    dc = scene.decoy_local()[0]
    s, ok = judge()
    check("wrong object: the graspable 48 mm decoy dropped over the cradle falls "
          f"straight through the four posts to the floor (readback z="
          f"{float(dc[2]) * 1000:.1f} mm ~ 24), never seats, score ~0",
          float(dc[2]) < 0.032 and not bool(scene.in_seat()[0])
          and s <= 0.02 and not ok)

    # =========================== 12-14. settle gate, cap, latched credit ====================
    env.reset(seed=101)
    step(30)
    place(scene.globe, sh((0.0, 0.0, seat_z + 0.003)), settle_steps=90)
    assert bool(scene.in_seat()[0]), "probe setup: globe not seated"
    # shutter written closed but SLIDING at 0.5 m/s along its channel (+y, away):
    # the shuttered-in latch fires while the velocity gate blocks success, and the
    # slide carries it out of the closed window before it can settle there.
    place(scene.shutter, sh((c.door_x, 0.0, c.door_z + 0.001)),
          quat=scene.shelter.data.root_quat_w[:1], vel_local=(0.0, 0.5, 0.0),
          settle_steps=0)
    step(1)
    gate_closed = bool(scene.door_closed()[0])
    gate_seated = bool(scene.in_seat()[0])
    gate_settled = bool(scene.settled()[0])
    _s, gate_ok = judge()
    dlin = float(scene.shutter.data.root_lin_vel_w[0].norm())
    check("settle gate: globe seated + shutter closed but sliding at "
          f"{dlin:.2f} m/s is NOT success (velocity gates are real) while the "
          "shuttered-in latch legitimately fires",
          gate_closed and gate_seated and not gate_settled and not gate_ok
          and dlin > c.door_settle_lin)
    step(60)  # the slide carries the shutter out of the closed window; friction parks it
    report("gate-drift")
    assert not bool(scene.door_closed()[0]), "probe setup: shutter failed to drift open"
    # yank the globe far away: every latch stays, the live predicate drops
    place(scene.globe, (-0.45 + sh((0.0, 0.0, 0.0))[0] * 0.0, 0.0, R + 0.002),
          settle_steps=40)
    report("yanked")
    s_cap, ok = judge()
    check("cap: enter + seat + shuttered-in all latched, then the globe yanked "
          f"away -> score == 0.75 cap ({s_cap:.4f}, float32 + eps), NOT success",
          not bool(scene.in_seat()[0]) and 0.745 <= s_cap <= 0.75 + 1e-5 and not ok)
    step(40)
    s_after, ok = judge()
    check("latched credit: 40 further steps leave the latched score unchanged "
          f"({s_cap:.3f} -> {s_after:.3f}) and in_seat stays False",
          abs(s_after - s_cap) < 1e-3 and not bool(scene.in_seat()[0]) and not ok)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== 17. camera + save + verdict ================================
    check(f"camera: >= 20 rgb frames captured ({len(frames)})", len(frames) >= 20)
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.lantern_shelter")
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

"""Smoke / rubric-REJECTION battery for UmbrellaUnrackCradleScene (sim_gen task
`put_umbrella_in_umbrella_stand_i220`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — force-extract the umbrella up out of its socket,
teleport-carry to a hover, force-lower into both V-notches, staged hands-off release —
is the acceptance evidence that the rubric ACCEPTS a correct outcome; it passes on
seeds 0/1/2). Every teleport here is instrumentation that CONSTRUCTS a wrong (or
partial) outcome and asserts the rubric REJECTS it. No probe in this battery ever
reaches success(), and a final audit asserts exactly that.

  1-2. settle/no-NaN    — reset layout settles finite: umbrella seated tip-down in one
                          socket, cane in the other, both fixtures near nominal;
                          score ~0, no success;
  3-4. randomization    — READBACK over 6 seeded resets: stand and cradle position +
                          yaw all vary, BOTH socket assignments occur, every reset sane;
  5.  null policy       — 240 idle steps -> score ~0, no success (the SEED task's goal
                          state — umbrella upright in a receptacle — persists and is
                          worth NOTHING here);
  6.  seed-strategy     — the seed's plan (vertical tip-down INSERTION into a socket)
                          re-executed as a single state write -> settles seated in the
                          stand -> score ~0. The anti-seed clause: this task judges the
                          cradle rest, and insertion earns nothing;
  7.  floor rest        — umbrella lying flat on the FLOOR beside the fixtures: it IS
                          horizontal, but both seat points read far from the axis ->
                          rejected by the seats, not the tilt;
  8.  crosswise balance — umbrella balanced PERPENDICULAR across a single notch: that
                          notch reads nested (~13 mm) but the OTHER seat reads ~0.2 m
                          -> rejected (BOTH notches must seat);
  9.  diagonal lean     — one end on the floor, shaft resting in a notch at ~16 deg:
                          seated-ish at one notch but outside the 10-deg horizontality
                          cone -> rejected;
  10. cane in cradle    — the CANE nested in BOTH V-notches counts for NOTHING (success
                          and score are judged on the umbrella by identity);
  11. settle gate       — the umbrella genuinely nested in both notches but still
                          MOVING (axial velocity injected; the sustained-stillness
                          counter reads zero) is NOT success — removed before it can
                          settle. Also anchors the rubric: a real nest reads ~13 mm;
  12. latched credit    — teleporting the nested umbrella away leaves the latched
                          score (0.30, cradle stage) unchanged while seated() drops;
                          the extraction latch never fired anywhere in this battery;
  13. rejection audit   — success() was never True at ANY judged point;
  14. final no-NaN      — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.put_umbrella_in_umbrella_stand_i220.smoke --headless
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
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

SOCK_Y = scene_mod.SOCK_Y
SEAT_Y = scene_mod.SEAT_Y
SEAT_Z = scene_mod.SEAT_Z
R_REST = scene_mod.R_REST
BASE_T = scene_mod.BASE_T
TIP_LOCAL_Z = scene_mod.TIP_LOCAL_Z


def qmul(a: tuple, b: tuple) -> tuple:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def qz(yaw: float) -> tuple:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def qx(th: float) -> tuple:
    return (math.cos(th / 2), math.sin(th / 2), 0.0, 0.0)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.umbrella_unrack_cradle")().build(num_envs=args.num_envs,
                                                            device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.80, 0.02, 1.05)) + o),
                                tuple(np.array((0.45, 0.00, 0.25)) + o),
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

    def report(tag: str) -> None:
        s, ok = judge()
        u, ca = rel(scene.umbrella), rel(scene.cane)
        d = scene.seat_dists()[0]
        print(f"[smoke] {tag:16s} | umb=({float(u[0]):+.3f},{float(u[1]):+.3f},"
              f"{float(u[2]):.3f}) cane_z={float(ca[2]):.3f}"
              f" low_end={float(scene.low_end_height()[0]):.3f}"
              f" seat_d=({float(d[0]):.4f},{float(d[1]):.4f})"
              f" seated={bool(scene.seated()[0])} horiz={bool(scene.horizontal()[0])}"
              f" set={bool(scene.settled()[0])}"
              f" score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, xyz, quat=None, vel=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor([float(v) for v in xyz], device=device)
        if quat is None:
            st[:, 3] = 1.0
        else:
            st[:, 3:7] = torch.tensor([float(v) for v in quat], device=device)
        if vel is not None:
            st[:, 7:10] = torch.tensor([float(v) for v in vel], device=device)
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def frame_of(body) -> tuple[torch.Tensor, torch.Tensor, float]:
        """(rel pos (3,), quat (4,), yaw) of a kinematic fixture."""
        r = rel(body)
        q = body.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return r, q, yaw

    def local_pt(body, loc) -> torch.Tensor:
        r, q, _ = frame_of(body)
        return r + quat_apply(q.view(1, 4), torch.tensor([loc], device=device))[0]

    def to_local(body, p: torch.Tensor) -> tuple[float, float]:
        """xy of an env-rel point in the fixture's local frame."""
        r, _q, yaw = frame_of(body)
        dx, dy = float(p[0] - r[0]), float(p[1] - r[1])
        cs, sn = math.cos(yaw), math.sin(yaw)
        return cs * dx + sn * dy, -sn * dx + cs * dy

    def cradled_pose(extra_yaw: float = 0.0, pitch: float = 0.0,
                     org_loc=None) -> tuple[tuple, tuple]:
        """Pos/quat that lays the umbrella axis along the cradle groove (+ optional
        extra yaw / pitch), origin at cradle-local `org_loc`."""
        _r, _q, yaw = frame_of(scene.cradle)
        qu = qmul(qz(yaw + extra_yaw), qx(-math.pi / 2 + pitch))
        loc = org_loc if org_loc is not None else (0.0, float(c.y_org),
                                                   SEAT_Z + R_REST)
        pos = local_pt(scene.cradle, tuple(float(v) for v in loc))
        return tuple(float(v) for v in pos), qu

    AWAY = ((0.05, -0.62, 0.030), qx(-math.pi / 2))  # flat on the floor, far from both

    def layout_sane(tag: str) -> bool:
        """Reset honesty: umbrella seated tip-down in one socket, cane in the other,
        both fixtures near nominal."""
        u, ca = rel(scene.umbrella), rel(scene.cane)
        sp, cp = rel(scene.stand), rel(scene.cradle)
        ux, uy = to_local(scene.stand, u)
        cx, cy = to_local(scene.stand, ca)
        ok = (abs(float(sp[0]) - c.stand_pos[0]) < c.stand_jitter + 0.005
              and abs(float(sp[1]) - c.stand_pos[1]) < c.stand_jitter + 0.005
              and abs(float(cp[0]) - c.cradle_pos[0]) < c.cradle_jitter + 0.005
              and abs(float(cp[1]) - c.cradle_pos[1]) < c.cradle_jitter + 0.005
              and abs(float(u[2]) - 0.377) < 0.02 and abs(float(ca[2]) - 0.368) < 0.02
              and abs(ux) < 0.03 and abs(cx) < 0.03
              and abs(abs(uy) - SOCK_Y) < 0.03 and abs(abs(cy) - SOCK_Y) < 0.03
              and uy * cy < 0
              and abs(uy - float(scene.umb_side[0]) * SOCK_Y) < 0.03)
        if not ok:
            print(f"[smoke] layout sanity VIOLATION at {tag}: "
                  f"umb_loc=({ux:.3f},{uy:.3f}) cane_loc=({cx:.3f},{cy:.3f})", flush=True)
        return ok

    bodies = (scene.stand, scene.cradle, scene.umbrella, scene.cane)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("settle: all states finite; umbrella seated tip-down in one socket, cane in "
          "the other, stand and cradle near nominal", fin and layout_sane("reset"))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    sane = True
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(30)
        sane = sane and layout_sane(f"seed {sd}")
        sp, _qs, syaw = frame_of(scene.stand)
        cp, _qc, cyaw = frame_of(scene.cradle)
        reads.append((float(sp[0]), float(sp[1]), syaw, float(cp[0]), float(cp[1]),
                      cyaw, float(scene.umb_side[0])))
    arr = np.array(reads)
    print("[smoke] randomization readback (stand_x, stand_y, stand_yaw, cradle_x, "
          f"cradle_y, cradle_yaw, side):\n{np.round(arr, 4)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: stand and cradle poses vary (spreads "
          f"stand=({spread[0]:.3f},{spread[1]:.3f},{spread[2]:.2f} rad) "
          f"cradle=({spread[3]:.3f},{spread[4]:.3f},{spread[5]:.2f} rad))",
          spread[0] > 0.02 and spread[1] > 0.02 and spread[2] > 0.15
          and spread[3] > 0.02 and spread[4] > 0.02 and spread[5] > 0.15)
    sides = {float(v) for v in arr[:, 6]}
    check(f"randomization: BOTH socket assignments occur ({len(sides)} sides seen), "
          "every reset seats correctly and sane", len(sides) == 2 and sane)

    # =========================== 5. null policy fails =======================================
    # The SEED task's goal state (umbrella upright in a receptacle) IS the reset state
    # here — it persists inertly and is worth nothing.
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    low = float(scene.low_end_height()[0])
    check("null policy: 240 idle steps in the seed's goal state (seated in the "
          f"socket, low_end={low:.3f} m far below the extraction latch) -> "
          "score ~0, no success", s <= 0.02 and not ok and low < 0.10)

    # =========================== 6. seed-strategy: insertion earns nothing ==================
    # Re-execute the seed's PLAN as a single state write: umbrella inserted vertically
    # tip-down into its socket (no stepped high poses -> no extraction latch).
    env.reset(seed=41)
    step(30)
    side = float(scene.umb_side[0])
    sock = local_pt(scene.stand, (0.0, side * SOCK_Y, 0.0))
    _r, _q, syaw = frame_of(scene.stand)
    place(scene.umbrella,
          (float(sock[0]), float(sock[1]), BASE_T + 0.004 - TIP_LOCAL_Z),
          quat=qz(syaw), settle_steps=90)
    report("seed-insertion")
    s, ok = judge()
    check("seed-strategy analog: vertical tip-down INSERTION into the stand socket "
          f"(the seed's whole plan) settles seated and earns score={s:.2f} ~0 — "
          "the reset state is the seed's success state and is worth nothing",
          s <= 0.02 and not ok)

    # =========================== 7. floor rest ==============================================
    # Horizontal, settled — but on the FLOOR: the seats reject it, not the tilt.
    env.reset(seed=51)
    step(20)
    place(scene.umbrella, (0.12, 0.05, 0.030), quat=qx(-math.pi / 2), settle_steps=50)
    report("floor-rest")
    d = scene.seat_dists()[0]
    s, ok = judge()
    check("floor rest: umbrella lying flat on the floor IS horizontal "
          f"(horiz={bool(scene.horizontal()[0])}) yet both seats read "
          f"({float(d[0]) * 1000:.0f},{float(d[1]) * 1000:.0f}) mm >> "
          f"{c.seat_tol * 1000:.0f} mm -> rejected by the seats",
          float(d.min()) > c.seat_tol + 0.05 and not ok and s <= 0.02)

    # =========================== 8. crosswise one-notch balance =============================
    env.reset(seed=61)
    step(20)
    pos, qu = cradled_pose(extra_yaw=math.pi / 2,
                           org_loc=(0.0, -SEAT_Y, SEAT_Z + R_REST))
    place(scene.umbrella, pos, quat=qu, settle_steps=12)
    report("crosswise")
    d = scene.seat_dists()[0]
    cross_ok = (float(d.max()) > c.seat_tol + 0.05 and not bool(scene.seated()[0])
                and not bool(scene.success()[0]))
    judge()
    place(scene.umbrella, AWAY[0], quat=AWAY[1], settle_steps=30)
    check("crosswise balance: umbrella laid PERPENDICULAR across one notch reads the "
          f"other seat at {float(d.max()) * 1000:.0f} mm >> tol -> rejected (BOTH "
          "notches must seat)", cross_ok)

    # =========================== 9. diagonal lean ===========================================
    # One end on the floor, shaft resting in notch A at ~16 deg: outside the 10-deg
    # horizontality cone by construction.
    env.reset(seed=71)
    step(20)
    al = math.radians(16.0)
    s_a = 0.171  # body-z where the axis crosses seat A (bare shaft)
    org_loc = (0.0, -SEAT_Y - s_a * math.cos(al), SEAT_Z + R_REST - s_a * math.sin(al))
    pos, qu = cradled_pose(pitch=al, org_loc=org_loc)
    place(scene.umbrella, pos, quat=qu, settle_steps=45)
    report("diagonal-lean")
    d = scene.seat_dists()[0]
    tilt = math.degrees(math.asin(abs(
        float((scene._axis_ends()[1] - scene._axis_ends()[0])[0, 2])
        / float((scene._axis_ends()[1] - scene._axis_ends()[0])[0].norm()))))
    lean_ok = (not bool(scene.horizontal()[0]) and not bool(scene.success()[0]))
    judge()
    place(scene.umbrella, AWAY[0], quat=AWAY[1], settle_steps=30)
    check(f"diagonal lean: one end grounded, shaft on a notch at ~{tilt:.0f} deg "
          f"(seats ({float(d[0]) * 1000:.0f},{float(d[1]) * 1000:.0f}) mm) violates "
          f"the {c.tilt_max_deg:.0f}-deg horizontality cone -> rejected", lean_ok)

    # =========================== 10. cane in the cradle (identity) ==========================
    env.reset(seed=81)
    step(20)
    _r, _q, cyaw = frame_of(scene.cradle)
    cane_pos = local_pt(scene.cradle, (0.0, 0.0, SEAT_Z + R_REST))
    place(scene.cane, tuple(float(v) for v in cane_pos),
          quat=qmul(qz(cyaw), qx(-math.pi / 2)), settle_steps=60)
    report("cane-cradled")
    s, ok = judge()
    check("cane in the cradle: the CANE nested in BOTH V-notches counts for NOTHING — "
          "score ~0, no success (identity: the umbrella is judged, not 'some stick')",
          s <= 0.02 and not ok)

    # =========================== 11. settle gate (nested but moving) ========================
    # Genuinely nested in both notches but sliding along the groove: the sustained-
    # stillness counter reads zero (also guards the teleport trap — stillness
    # accumulated seated in the socket must not carry into the new pose).
    env.reset(seed=91)
    step(20)
    _r, _q, cyaw = frame_of(scene.cradle)
    pos, qu = cradled_pose()
    v_ax = (-math.sin(cyaw) * 0.30, math.cos(cyaw) * 0.30, 0.0)
    place(scene.umbrella, pos, quat=qu, vel=v_ax, settle_steps=2)
    report("nested-moving")
    d = scene.seat_dists()[0]
    v_now = float(scene.umbrella.data.root_lin_vel_w[0].norm())
    s_before, ok = judge()
    gate_ok = (float(d.max()) < c.seat_tol and bool(scene.seated()[0])
               and bool(scene.horizontal()[0]) and v_now > c.settle_lin
               and not bool(scene.settled()[0]) and not ok)
    check("settle gate: a genuinely nested umbrella (seats "
          f"({float(d[0]) * 1000:.0f},{float(d[1]) * 1000:.0f}) mm < tol — the rubric "
          f"anchor) still sliding at {v_now:.2f} m/s is NOT success (sustained "
          "stillness required)", gate_ok)

    # =========================== 12. latched credit survives moving away ====================
    # remove it BEFORE it can settle into a real rest (this battery must never succeed)
    place(scene.umbrella, AWAY[0], quat=AWAY[1], settle_steps=40)
    report("moved-away")
    s_after, ok = judge()
    check("latched credit: teleporting the nested umbrella away leaves the latched "
          f"score unchanged ({s_before:.2f} -> {s_after:.2f}, cradle stage = 0.30; "
          "the extraction latch never fired in this battery) while seated() drops",
          abs(s_after - s_before) < 1e-3 and abs(s_after - 0.30) < 1e-3
          and not bool(scene.seated()[0]) and not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.umbrella_unrack_cradle")
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

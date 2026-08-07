"""Teleport solution for CanBalanceScene (sim_gen task `coke_task_i15`) — the task's
legitimacy certificate.

The load-bearing interactions and how each is executed:

1. READ THE PRELOAD: the solver reads which plates rest on the loaded pan (the scene
   exposes the same state a camera would show — plate thicknesses encode mass) and
   picks the unique can subset with the matching total. The judged quantity is never
   written: it is the beam's settled angle.
2. TRANSPORT (teleport, one can at a time): a single root-state write carries each
   REQUIRED can from its floor spawn to a FREE-SPACE hover 8 mm above the empty pan's
   floor (pan-frame pose, zero velocity) — exactly the pose a gripper would release
   it from. The write puts the can in open air over the pan; it satisfies no rubric
   clause by itself (a hovering can is not "resting", and the beam has not moved).
3. WEIGHING (gravity + contact, hands-off): the can FALLS onto the pan and the beam
   ANSWERS through its knife-edge bearing — it stays hard on its stop while the can
   total is short, and swings back to LEVEL only when the unique correct subset is
   aboard. The equilibrium that success() reads is produced entirely by gravity,
   contact and the pendulum restoring torque, never written.
4. NULL ADJUSTMENT (closed loop): if the settled beam misses the level gate (residual
   moment from landing scatter), the solver re-picks one can and re-drops it a
   computed few millimetres along the arm — a regrasp-and-replace, transport through
   free space again — and lets gravity re-answer.
5. Spare plates and unused cans are never touched; the preload plates ride their pan
   untouched throughout.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.coke_task_i15.solve --headless [--seed N]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import math
import os
import threading

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers simgen.can_balance)
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.can_balance")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        from isaaclab.utils.math import quat_apply

        s = scene._status()
        seat_loc = torch.tensor([0.0, 0.0, c.pivot_h], device=device).expand(n, 3)
        seat_w = scene.stand.data.root_pos_w \
            + quat_apply(scene.stand.data.root_quat_w, seat_loc)
        dseat = float((scene.beam.data.root_pos_w - seat_w).norm(dim=-1)[0])
        parts = []
        for i, nm in enumerate(c.can_names):
            loc = scene._beam_local(scene.cans[nm].data.root_pos_w)[0]
            parts.append(f"{nm}=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
                         f"{float(loc[2]):+.3f})on={bool(s['cans_on'][0, i])}")
        for i, nm in enumerate(c.plate_names):
            loc = scene._beam_local(scene.plates[nm].data.root_pos_w)[0]
            parts.append(f"{nm}=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
                         f"{float(loc[2]):+.3f})on={bool(s['plates_on'][0, i])}")
        pv = max(float(scene.plates[nm].data.root_lin_vel_w.norm(dim=-1)[0])
                 for nm in c.plate_names)
        pw = max(float(scene.plates[nm].data.root_ang_vel_w.norm(dim=-1)[0])
                 for nm in c.plate_names)
        print(f"[solve] {tag:14s} | tilt={float(s['tilt'][0]):+6.2f}deg "
              f"seated={bool(s['seated'][0])} dseat={dseat:.4f} "
              f"still={bool(s['beam_still'][0])} platevel={pv:.3f}/{pw:.3f} "
              f"preload_ok={bool(s['preload_ok'][0])} spares_ok={bool(s['spares_ok'][0])} "
              + " ".join(parts)
              + f" success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def pan_hover_pose(i: int, off_x: float, off_y: float) -> torch.Tensor:
        """(N,13) root state: can i hovering 8 mm above the EMPTY pan floor at
        pan-local (off_x, off_y), orientation matched to the (possibly tilted) pan
        plane so it lands flat; zero velocity. Free space: inside the rim ring,
        clear of the other cans by construction of the offsets."""
        from isaaclab.utils.math import quat_apply

        loc = torch.zeros(n, 3, device=device)
        loc[:, 0] = c.arm_len + off_x
        loc[:, 1] = off_y
        loc[:, 2] = c.pan_floor_z + c.can_h[i] / 2 + 0.008
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.beam.data.root_pos_w \
            + quat_apply(scene.beam.data.root_quat_w, loc)
        st[:, 3:7] = scene.beam.data.root_quat_w
        return st

    def denom_est() -> float:
        """Restoring-torque denominator (kg*m) for the CURRENT preload/can loading —
        the same statics as cfg.__post_init__, used only to size the closed-loop
        re-drop adjustment."""
        h_pan = -c.pan_floor_z
        dn = c.beam_mass * (-c.beam_com_z)
        for i in range(3):
            if bool(scene.preload[0, i]):
                dn += c.masses[i] * (h_pan - 0.016)          # plate stack on pan
                dn += c.masses[i] * (h_pan - c.can_h[i] / 2)  # matching can on pan
        return dn

    def wait_beam(max_steps: int, want_tilt_over: float | None = None) -> None:
        """Hands-off until the beam is still (optionally also past a tilt), or the
        step budget runs out."""
        for k in range(max_steps):
            env.step(no_action)
            if k < 60:
                continue
            s = scene._status()
            if bool(s["beam_still"][0]) and (
                    want_tilt_over is None
                    or abs(float(s["tilt"][0])) > want_tilt_over):
                return

    def settle_can(name: str, max_steps: int = 720) -> bool:
        """Hands-off: the dropped can falls onto the pan; the beam swings and damps.
        True once the can is resting on the pan (live predicate); False if it
        escaped the pan region (bounced out)."""
        body = scene.cans[name]
        i = list(c.can_names).index(name)
        for k in range(max_steps):
            env.step(no_action)
            loc = scene._beam_local(body.data.root_pos_w)[0]
            dx = float(loc[0]) - c.arm_len
            radial = math.hypot(dx, float(loc[1]))
            dz = float(loc[2]) - c.pan_floor_z
            if radial > c.rim_ring_r + 0.014 or dz < -0.03:
                print(f"[solve] {name} left the pan (radial={radial:.3f} "
                      f"dz={dz:+.3f})", flush=True)
                return False
            if k > 90 and bool(scene._status()["cans_on"][0, i]):
                return True
        return bool(scene._status()["cans_on"][0, i])

    def place_can(idx: int, off_x: float, off_y: float) -> None:
        """TRANSPORT can idx to the hover over the empty pan, then hands-off: it
        falls, the pan takes the weight, the beam answers. Retries the drop if the
        can bounces out."""
        name = c.can_names[idx]
        for attempt in range(4):
            scene.cans[name].write_root_state_to_sim(
                pan_hover_pose(idx, off_x, off_y), all_ids)
            if settle_can(name):
                return
            print(f"[solve] {name} retry {attempt + 1}", flush=True)
        print(f"SIM_GEN_SOLVE: FAIL ({name} would not stay on the pan)", flush=True)
        os._exit(1)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    # The beam was written LEVEL at reset; the preload torque must now tip it onto
    # its stop — the tilted balance is the initial condition.
    step(120)
    wait_beam(600, want_tilt_over=2.5 * c.level_tol_deg)
    sp = (scene.stand.data.root_pos_w - scene.env_origins)[0]
    syaw = math.degrees(float(scene._stand_yaw()[0]))
    pre = [bool(scene.preload[0, i]) for i in range(3)]
    req = [i for i in range(3) if pre[i]]
    target = sum(c.masses[i] for i in req)
    spawns = []
    for nm in (*c.can_names, *c.plate_names):
        body = scene.cans.get(nm) or scene.plates[nm]
        p = (body.data.root_pos_w - scene.env_origins)[0]
        spawns.append(f"{nm}=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f})")
    print(f"[solve] layout readback (seed {args.seed}): "
          f"stand=({float(sp[0]):+.3f},{float(sp[1]):+.3f}) yaw={syaw:+.1f}deg "
          f"preload={pre} -> target {target * 1000:.0f} g "
          f"(cans {[c.can_names[i] for i in req]})", flush=True)
    print("[solve] spawns: " + " ".join(spawns), flush=True)
    report("reset")
    st0 = scene._status()
    # A preload plate can take a moment longer than the beam to drop under the
    # settle-velocity gates (it is judged live) — give it hands-off time.
    for _ in range(20):
        if bool(st0["preload_ok"][0]):
            break
        step(30)
        st0 = scene._status()
    for body in (scene.beam, *scene.plates.values(), *scene.cans.values()):
        assert torch.isfinite(body.data.root_pos_w).all(), "NaN/inf after settle"
    assert bool(st0["seated"][0]), "beam must start seated on its knife edge"
    assert bool(st0["preload_ok"][0]), "preload plates must ride their pan"
    tilt0 = float(st0["tilt"][0])
    assert abs(tilt0) > 2.0 * c.level_tol_deg, \
        f"beam must start hard against its stop (tilt {tilt0:+.2f} deg)"
    s0 = print_score("P0 reset+settle (beam tipped onto its stop by the preload)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phases 1..k: dose the empty pan with the matching cans ---------------
    # Pan-local drop offsets chosen so (a) the NET moment about the pivot axis is
    # zero (only the local-x offset carries torque; the pivot axis is local y) and
    # (b) can hulls stay > 5 mm apart — beyond the 2 x 1.5 mm contact offsets, so
    # neighbours never sit in persistent light contact and skitter:
    #   k=1: the can on the pan axis;
    #   k=2: both at x=0 (zero moment trivially), y=+/-31 mm (62 mm separation);
    #   k=3: equilateral triangle, circumradius 35.5 mm (side 61.4 mm), rotated so
    #        the mass-weighted x sums to zero:
    #        0.1*0.0232 + 0.2*(-0.0349) + 0.4*0.0116 = -2e-5 kg*m (~0.02 deg).
    if len(req) == 1:
        offs = {req[0]: (0.0, 0.0)}
    elif len(req) == 2:
        offs = {req[0]: (0.0, -0.031), req[1]: (0.0, 0.031)}
    else:
        offs = {0: (0.0232, 0.0268), 1: (-0.0349, 0.0067), 2: (0.0116, -0.0335)}
    s_prev = s0
    placed = 0
    for idx in req:
        placed += 1
        place_can(idx, *offs[idx])
        wait_beam(480)
        report(f"can {c.can_names[idx]}")
        assert bool(scene._req_on_pan[0, idx]), \
            f"{c.can_names[idx]} on-pan latch did not set"
        sk = print_score(f"P{placed} {c.can_names[idx]} "
                         f"({c.masses[idx] * 1000:.0f} g) rests on the empty pan")
        floor = c.w_first + c.w_each * placed - 1e-6
        assert sk >= s_prev - 1e-6, "score decreased across a can placement"
        assert sk >= floor, f"P{placed} score {sk} (expected >= {floor:.2f})"
        s_prev = sk

    # ---------------- closed loop: let the beam answer; trim if it misses the gate ---------
    for trim in range(6):
        wait_beam(960)
        s = scene._status()
        tilt = float(s["tilt"][0])
        if bool(scene.success()[0]):
            break
        if abs(tilt) < 0.9 * c.level_tol_deg:
            # The beam already answers LEVEL — the blocker is a settle flicker
            # (a body's velocity briefly over a still gate), not a residual
            # moment. More hands-off time is the medicine, not a re-drop.
            print(f"[solve] wait {trim + 1}: tilt {tilt:+.2f} deg already inside "
                  f"the gate — settling on", flush=True)
            step(120)
            continue
        # Residual moment from landing scatter: re-pick one can and re-drop it a
        # computed few mm along the arm (tilt > 0 = empty end high = needs more
        # moment = move a can OUTWARD).
        adj = req[-1]  # heaviest required can: most authority per mm
        loc = scene._beam_local(scene.cans[c.can_names[adj]].data.root_pos_w)[0]
        x_now = float(loc[0]) - c.arm_len
        y_now = float(loc[1])
        dx = math.tan(math.radians(tilt)) * denom_est() / c.masses[adj]
        dx = max(-0.008, min(0.012, dx))
        x0 = offs[adj][0]
        x_new = max(x0 - 0.006, min(x0 + 0.014, x_now + dx))
        print(f"[solve] trim {trim + 1}: tilt {tilt:+.2f} deg -> re-drop "
              f"{c.can_names[adj]} at x {x_now:+.4f} -> {x_new:+.4f}", flush=True)
        place_can(adj, x_new, y_now if abs(y_now) > 0.01 else offs[adj][1])
    report("level")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (beam never settled level)", flush=True)
        os._exit(1)
    s_lvl = print_score("P-final beam settled LEVEL on its knife edge")
    assert s_lvl >= s_prev - 1e-6, "score decreased at the level phase"
    assert s_lvl >= 0.999, f"success should score 1.0, got {s_lvl}"

    # ---------------- persistence (>= 3 simulated seconds, hands-off) ----------------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P-persist 3.3 s hands-off")
    ok = hold and bool(scene.success()[0]) and s4 >= s_lvl - 1e-6
    if ok:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)

    code = 0 if ok else 1
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
    except BaseException as e:  # noqa: BLE001 - die fast, never idle until the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL (exception: {e})", flush=True)
        os._exit(2)

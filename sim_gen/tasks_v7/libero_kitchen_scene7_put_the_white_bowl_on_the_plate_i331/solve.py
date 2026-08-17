"""Teleport solution for BalanceServeScene (sim_gen task
libero_kitchen_scene7_put_the_white_bowl_on_the_plate_i331) — the task's
legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, small per-step pose increments): the steel weight and the
   loaded bowl are "held" by writing their root poses each physics step along
   smooth paths (<= 2.5 mm per step). The CUBES ARE NEVER WRITTEN: they ride
   inside the carried bowl on real contacts. The BOARD IS NEVER WRITTEN after
   reset: its every angle is produced by gravity, contact and the pivot spring.
2. COUNTERWEIGHT (the reasoning core): read back the present-cube count k, compute
   the balance radius r = plate_x * (bowl + k*cube) / weight  in {54, 90, 126,
   162} mm, carry the weight over the weighing half and RELEASE it 6 mm up —
   gravity loads the board, which tips onto its deck stop (plate end up).
3. SERVE (release + spring physics): the loaded bowl is released 6 mm above the
   LIVE plate pose (computed from the board's current tilt); the board swings
   back and — because the torques match — settles LEVEL inside the 3.5 deg band.
   The judged tilt is a pure physical outcome of the two releases.
4. TRIM (closed loop, like a cook nudging a scale weight): if the settled tilt is
   off, the residual is read and the weight re-carried by dr = k_spring * tilt /
   (m_w * g) — at most 3 trims (unneeded in practice; kept as safety).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds
after success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene7_put_the_white_bowl_on_the_plate_i331.solve --headless [--seed N]
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
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qconj(q: torch.Tensor) -> torch.Tensor:
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device)


def _yaw_quat(q: torch.Tensor) -> torch.Tensor:
    """(N,4) -> (N,4): the yaw-only part of a wxyz quaternion."""
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    out = torch.zeros_like(q)
    out[:, 0] = torch.cos(yaw / 2)
    out[:, 3] = torch.sin(yaw / 2)
    return out


def main() -> None:
    from isaaclab.utils.math import quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.balance_serve")().build(num_envs=args.num_envs, device=device)
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

    def tilt_deg() -> float:
        return math.degrees(float(scene.tilt()[0]))

    def report(tag: str) -> None:
        loc_w = scene._local(scene.beam, scene.weight.data.root_pos_w)[0]
        inb = scene.cubes_in_bowl()[0]
        pres = scene.present[0]
        marks = "".join(("P" if bool(pres[i]) else "-")
                        + ("B" if bool(inb[i]) else "") + " " for i in range(3))
        print(f"[solve] {tag:14s} | tilt={tilt_deg():+.2f}deg cubes {marks}"
              f"w_beam=({float(loc_w[0]):+.3f},{float(loc_w[1]):+.3f},{float(loc_w[2]):+.3f}) "
              f"seated={bool(scene.bowl_seated()[0])} cubes_ok={bool(scene.cubes_ok()[0])} "
              f"place={bool(scene._place[0])} seat={bool(scene._seat[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    G_DT = 9.81 / 120.0  # intra-step gravity velocity kick on a held body

    def hold(body, pos_w: torch.Tensor, quat: torch.Tensor,
             vel_w: torch.Tensor | None = None, bias: float = G_DT / 2) -> None:
        """One held-pose write: root at world `pos_w`, orientation `quat`. Writing
        the true carry velocity (not zero) lets the contact solver DRAG contents
        along by friction, like a real carry — zero-velocity teleports move riders
        only by depenetration impulses, which can pop them over the walls. The vz
        `bias` cancels the mean intra-step sag of a once-per-step pose-held body:
        gravity alone needs +g*dt/2; a bowl LOADED with riders is also pushed down
        by their reaction, so the bias scales by (m_body + m_load)/m_body —
        undercompensated, the floor sags away each substep and the riders
        cumulatively sink through it (observed: 8 mm embed over one lift)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat
        if vel_w is not None:
            st[:, 7:10] = vel_w
        st[:, 9] += bias
        body.write_root_state_to_sim(st, all_ids)

    def glide(body, p1_w: torch.Tensor, quat: torch.Tensor, v: float = 0.0025,
              bias: float = G_DT / 2) -> None:
        """Carry `body` from its current position to world `p1_w` at a CONSTANT
        `v` per step. Constant-and-fast beats a gentle ramp here: the carried
        bowl is dynamic and pose-held once per step, so its floor only presses
        its riders continuously while the written up-velocity exceeds the
        intra-step gravity kick (g*dt = 0.08 m/s) — a slow ramp start lets the
        floor sag away and sweep through resting cubes. Contents ride on real
        contacts; nothing but the carried body is written."""
        p0 = body.data.root_pos_w[0].clone()
        dist = float((p1_w - p0).norm())
        steps = max(int(dist / v) + 1, 8)
        delta = (p1_w - p0) / steps
        vel = (delta / (1.0 / 120.0)).unsqueeze(0).expand(n, 3)
        for s in range(1, steps + 1):
            f = s / steps
            hold(body, (p0 + (p1_w - p0) * f).unsqueeze(0).expand(n, 3), quat, vel,
                 bias)
            env.step(no_action)

    def plate_target_w(z_off: float) -> torch.Tensor:
        """LIVE world pose of a point `z_off` above the plate top (beam frame)."""
        loc = torch.tensor([c.plate_x, 0.0,
                            c.plank_t / 2 + c.plate_t + z_off], device=device)
        return (scene.beam.data.root_pos_w
                + quat_apply(scene.beam.data.root_quat_w, loc.expand(n, 3)))[0]

    def half_target_w(r: float, z_off: float) -> torch.Tensor:
        """LIVE world pose of the weight center placed at radius `r` on the
        weighing half, base `z_off` above the plank top (beam frame)."""
        loc = torch.tensor([-r, 0.0,
                            c.plank_t / 2 + z_off + c.weight_size[2] / 2],
                           device=device)
        return (scene.beam.data.root_pos_w
                + quat_apply(scene.beam.data.root_quat_w, loc.expand(n, 3)))[0]

    def carry_weight_to(r: float) -> None:
        """Lift the weight, carry it over the weighing half, release 6 mm up."""
        wq = scene.weight.data.root_quat_w[0:1].clone().expand(n, 4)
        p = scene.weight.data.root_pos_w[0].clone()
        up = p.clone()
        up[2] = 0.30
        glide(scene.weight, up, wq)                     # lift straight up
        over = half_target_w(r, 0.006)
        over_hi = over.clone()
        over_hi[2] = 0.30
        glide(scene.weight, over_hi, wq)                # traverse high
        # descend tracking the LIVE beam pose AND its tilt (the board may swing
        # under us; a tilt-matched base can't wedge an edge into the plank).
        # rel_q is the weight's YAW only: composed onto the live beam quat, the
        # base snaps parallel to the plank however the board is tipped.
        rel_q = _yaw_quat(wq)
        for s in range(90):
            tgt = half_target_w(r, 0.006 + 0.14 * (1 - (s + 1) / 90))
            cur = scene.weight.data.root_pos_w[0]
            stp = tgt - cur
            m = float(stp.norm())
            if m > 0.0025:
                stp = stp * (0.0025 / m)
            qt = _qmul(scene.beam.data.root_quat_w, rel_q)
            hold(scene.weight, (cur + stp).unsqueeze(0).expand(n, 3), qt,
                 (stp * 120.0).unsqueeze(0).expand(n, 3))
            env.step(no_action)
        # RELEASE: hands off — gravity loads the board

    # ---------------- phase 0: reset, settle, baseline, readback ----------------------------
    step(150)  # cubes settle into the bowl; board proves it holds level empty
    pres = scene.present[0]
    k_cubes = int(pres.sum())
    side = int(scene.weight_side[0])
    wpos = (scene.weight.data.root_pos_w - scene.env_origins)[0].clone()
    dpos = (scene.dummy.data.root_pos_w - scene.env_origins)[0].clone()
    bpos = (scene.bowl.data.root_pos_w - scene.env_origins)[0].clone()
    m_load = c.bowl_mass + c.cube_mass * k_cubes
    r_bal = c.plate_x * m_load / c.weight_mass
    print(f"[solve] layout readback (seed {args.seed}): weight_side={side:+d} "
          f"weight=({float(wpos[0]):+.3f},{float(wpos[1]):+.3f}) "
          f"dummy=({float(dpos[0]):+.3f},{float(dpos[1]):+.3f}) "
          f"bowl=({float(bpos[0]):+.3f},{float(bpos[1]):+.3f}) "
          f"cubes_present={k_cubes} mask={pres.tolist()} "
          f"-> m_load={m_load:.3f}kg r_bal={r_bal * 1000:.1f}mm", flush=True)
    report("reset")
    t0 = tilt_deg()
    assert abs(t0) < 1.5, \
        f"CALIBRATION: empty board should rest level, tilt={t0:.2f}deg (spring units?)"
    assert bool((scene.cubes_in_bowl()[0] | ~pres).all()), \
        "every present cube must start inside the bowl"
    assert bool(scene.cubes_ok()[0]), "cube accounting must hold at reset"
    s0 = print_score("P0 reset+settle (board empty and level)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.01, f"baseline score should be 0, got {s0}"

    # ---------------- phase 1: COUNTERWEIGHT — weight onto the weighing half ----------------
    carry_weight_to(r_bal)
    step(240)  # board tips onto the deck stop under the weight
    report("weight-placed")
    t1 = tilt_deg()
    print(f"[solve] board under counterweight alone: tilt={t1:+.2f}deg "
          f"(deck stop ~{c.deck_stop_deg:.1f})", flush=True)
    assert t1 > c.level_tol_deg, \
        "weight alone should tip the board out of the level band (plate end up)"
    assert bool(scene._place[0]), "place latch did not set"
    s1 = print_score("P1 weight released at the balance radius")
    assert s1 >= s0 - 1e-6 and s1 >= 0.24, f"P1 score {s1} (expect place=0.25)"

    # ---------------- phase 2: SERVE — loaded bowl onto the plate ---------------------------
    def cube_telemetry(tag: str) -> None:
        rel = scene._local(scene.bowl, scene._cube_tensors()[0])[0]  # (3,3) bowl-frame
        inb = scene.cubes_in_bowl()[0]
        rows = " ".join(
            f"c{i}[{'B' if bool(inb[i]) else '.'}]"
            f"({float(rel[i, 0]) * 1000:+.0f},{float(rel[i, 1]) * 1000:+.0f},"
            f"{float(rel[i, 2]) * 1000:+.0f})mm"
            for i in range(3) if bool(pres[i]))
        print(f"[solve] cubes @ {tag:12s} {rows}", flush=True)

    bq = scene.bowl.data.root_quat_w[0:1].clone().expand(n, 4)  # keep its yaw
    # the bowl carries k cubes: scale the anti-sag vz bias by (m_bowl+m_load)/m_bowl
    lbias = (c.bowl_mass + c.cube_mass * k_cubes) / c.bowl_mass * G_DT / 2
    p = scene.bowl.data.root_pos_w[0].clone()
    up = p.clone()
    up[2] = 0.30
    cube_telemetry("pre-lift")
    glide(scene.bowl, up, bq, v=0.002, bias=lbias)     # lift straight up (cubes ride)
    cube_telemetry("post-lift")
    over = plate_target_w(0.006)
    over_hi = over.clone()
    over_hi[2] = 0.30
    glide(scene.bowl, over_hi, bq, v=0.002, bias=lbias)  # traverse high over the board
    cube_telemetry("post-traverse")
    # Descend tracking the live plate, TILT-MATCHED: the board sits on its deck
    # stop (~5.7 deg) until the bowl lands, so a bowl held upright would wedge
    # its low edge into the tilted plate and hammer the cubes out. Composing the
    # bowl's yaw onto the live beam quat keeps its base parallel to the plate.
    rel_q = _yaw_quat(bq)
    for s in range(110):
        tgt = plate_target_w(0.006 + 0.15 * (1 - (s + 1) / 110))
        cur = scene.bowl.data.root_pos_w[0]
        stp = tgt - cur
        m = float(stp.norm())
        if m > 0.002:
            stp = stp * (0.002 / m)
        qt = _qmul(scene.beam.data.root_quat_w, rel_q)
        hold(scene.bowl, (cur + stp).unsqueeze(0).expand(n, 3), qt,
             (stp * 120.0).unsqueeze(0).expand(n, 3), lbias)
        env.step(no_action)
        if (s + 1) % 22 == 0:
            cube_telemetry(f"descend-{s + 1}")
    cube_telemetry("post-descend")
    assert bool((scene.cubes_in_bowl()[0] | ~pres).all()), \
        "a cube fell out of the bowl during transport"
    step(400)  # RELEASE: board swings toward level and settles (overdamped spring)
    report("bowl-served")
    s2 = print_score("P2 loaded bowl released onto the plate")
    assert s2 >= s1 - 1e-6 and s2 >= 0.54, f"P2 score {s2} (expect seat -> 0.55)"

    # ---------------- phase 3: TRIM — nudge the weight if the residual is off ---------------
    for trim in range(3):
        t = float(scene.tilt()[0])
        if abs(math.degrees(t)) <= 2.0:
            break
        loc_w = scene._local(scene.beam, scene.weight.data.root_pos_w)[0]
        r_cur = -float(loc_w[0])
        dr = c.spring_k * t / (c.weight_mass * 9.81)  # tilt>0: weight side heavy
        r_new = min(max(r_cur - dr, 0.035), 0.200)
        print(f"[solve] TRIM {trim + 1}: tilt={math.degrees(t):+.2f}deg "
              f"r {r_cur * 1000:.1f} -> {r_new * 1000:.1f}mm", flush=True)
        carry_weight_to(r_new)
        step(400)
        report(f"trim-{trim + 1}")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (board never settled level with the bowl seated)",
              flush=True)
        os._exit(1)
    tf = tilt_deg()
    print(f"[solve] BALANCED: tilt={tf:+.2f}deg (band +/-{c.level_tol_deg:.1f})",
          flush=True)
    s3 = print_score("P3 board level, bowl seated, all settled")
    assert s3 >= 0.99, f"P3 score {s3} (expect success -> 1.0)"

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, hands-off) ------------
    holds = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        holds = holds and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
    ok = holds and bool(scene.success()[0]) and s4 >= s3 - 1e-6
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
    except BaseException:  # noqa: BLE001 — Kit teardown hangs; die loudly instead
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)

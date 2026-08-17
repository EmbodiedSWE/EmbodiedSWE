"""Teleport solution for DieTumbleScene (sim_gen task `stack_pyramid_i255`) —
the task's legitimacy certificate.

Teleports are TRANSPORT ONLY: every teleport writes back the die's CURRENT
orientation readback unchanged (plus zero velocity) and only moves the die
across open floor — exactly what a solver pushing the die around would achieve,
compressed. Every REORIENTATION is a real contact-dynamics tumble:

1. PLAN (pure math): read back the die's settled orientation, snap the RED face
   direction to the nearest world axis, and BFS (<= 2 moves) over the four
   tumble generators for a sequence that parks red on world -z (DOWN) — the
   unique pre-image of "red up" under the TWO forced +x quarter-turns of the
   entry (rim vault, then in-tray seat tumble): -z -> -x -> up.
2. STAGING TUMBLES (contact dynamics): at a staging point on open floor, each
   planned tumble is executed by a torque servo about the tumble axis
   (tau = ff + kp*(w_des - w), capped at 0.30 N*m, well over the 0.132 N*m
   gravity moment about the pivot edge); ground friction holds the bottom edge
   as the pivot, the torque is cut at 55 deg (past the 45 deg balance point)
   and gravity finishes the quarter-turn. The resulting face-up change and any
   red-up rest latch the scene's `tumbled`/`oriented` credit — settled contact
   states, never authored.
3. ENTRY = RIM VAULT + IN-TRAY SEAT TUMBLE (the signature interaction): the
   die is teleported to the approach point (2 mm shy of the -x rim, red DOWN),
   then torqued about +y while a 0.7 N press (< the 1.6 N slide threshold,
   << the ~5 N quasi-static step-climb threshold) holds its face against the
   rim's top edge — the pivot. Torque is cut at 69 deg (past the 58.8 deg rim
   balance angle); the die vaults the rim (red -z -> -x) and settles in the
   shallow lean the 6 mm inner curb allows. A second, ordinary forward tumble
   — pivoting on the leading bottom edge at slab level — completes the entry
   (red -x -> UP) and lands the die flat near the tray center, far from every
   wall. Both quarter-turns are forced by the rolling-cube group; landing
   energy stays far below the re-tumble threshold, so red stays up.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's partial credit is latched), holds HANDS-OFF for >= 3 simulated seconds
after success() first holds, and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds at the end.

Run (forge): python -u -m simgen_tasks.stack_pyramid_i255.solve --headless [--seed N]
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
from collections import deque

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401 — registers the scene
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


# ---- the rolling-cube group: tumble toward d = 90 deg world rotation about z x d ----------------
# name: (matrix acting on world vectors, floor direction d, world torque axis a = z x d)
_MOVES: dict[str, tuple[list[list[int]], tuple[float, float], tuple[float, float, float]]] = {
    "px": ([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], (1.0, 0.0), (0.0, 1.0, 0.0)),
    "nx": ([[0, 0, -1], [0, 1, 0], [1, 0, 0]], (-1.0, 0.0), (0.0, -1.0, 0.0)),
    "py": ([[1, 0, 0], [0, 0, 1], [0, -1, 0]], (0.0, 1.0), (-1.0, 0.0, 0.0)),
    "ny": ([[1, 0, 0], [0, 0, -1], [0, 1, 0]], (0.0, -1.0), (1.0, 0.0, 0.0)),
}
_TARGET = (0, 0, -1)  # red DOWN + rim vault (-z -> -x) + in-tray seat tumble (-x -> up)


def _apply(mat: list[list[int]], v: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(sum(mat[i][j] * v[j] for j in range(3)) for i in range(3))  # type: ignore[return-value]


def plan_moves(red0: tuple[int, int, int]) -> list[str]:
    """Shortest tumble sequence taking the red world direction to -z (BFS, <= 2 moves).

    If red already sits on -z, execute [py, ny]: py sends -z -> -y, ny sends
    -y -> -z — the die still demonstrably tumbles (latching `tumbled`) and
    returns red to the bottom.
    """
    if red0 == _TARGET:
        return ["py", "ny"]
    seen: dict[tuple[int, int, int], list[str]] = {red0: []}
    q: deque[tuple[int, int, int]] = deque([red0])
    while q:
        r = q.popleft()
        seq = seen[r]
        if len(seq) >= 2:
            continue
        for name, (mat, _d, _a) in _MOVES.items():
            r2 = _apply(mat, r)
            if r2 not in seen:
                seen[r2] = seq + [name]
                if r2 == _TARGET:
                    return seen[r2]
                q.append(r2)
    raise RuntimeError(f"no tumble plan from red={red0}")  # unreachable: group is transitive


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.die_tumble")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so
    # the task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def clear_wrench() -> None:
        scene.die.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def settle(max_steps: int = 600, need: int = 12) -> bool:
        streak = 0
        for _ in range(max_steps):
            env.step(no_action)
            streak = streak + 1 if bool(scene.still()[0]) else 0
            if streak >= need:
                return True
        return False

    def red_snap() -> tuple[int, int, int]:
        r = scene.red_world()[0]
        ax = int(torch.argmax(r.abs()))
        v = [0, 0, 0]
        v[ax] = 1 if float(r[ax]) > 0 else -1
        return tuple(v)  # type: ignore[return-value]

    def report(tag: str) -> None:
        loc = scene.die_tray_local()[0]
        r = scene.red_world()[0]
        print(f"[solve] {tag:12s} | die tray-local=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
              f"{float(loc[2]):+.3f}) red_w=({float(r[0]):+.2f},{float(r[1]):+.2f},"
              f"{float(r[2]):+.2f}) up_idx={int(scene.body_up_idx()[0])} "
              f"tumbled={bool(scene._tumbled[0])} oriented={bool(scene._oriented[0])} "
              f"arrived={bool(scene._arrived[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def teleport_die(x_w: float, y_w: float) -> None:
        """TRANSPORT ONLY: write the die's CURRENT orientation back unchanged at a
        new floor position (zero velocity, rest height + 1.5 mm)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = x_w
        st[:, 1] = y_w
        st[:, 2] = scene.env_origins[:, 2] + c.die_s / 2 + 0.0015
        st[:, 3:7] = scene.die.data.root_quat_w.clone()
        scene.die.write_root_state_to_sim(st, all_ids)

    def do_tumble(move: str, *, over_curb: bool = False) -> bool:
        """One quarter-turn tumble via contact dynamics: world torque servo about
        the tumble axis (ground/curb edge is the pivot); for the curb entry a
        gentle press (0.7 N << the ~5 N push-climb threshold) keeps the die's
        face on the curb's top edge. Torque is CUT past the balance angle and
        gravity completes the turn."""
        _mat, d, a = _MOVES[move]
        axis_w = torch.tensor(a, device=device).expand(n, 3)
        d_w = torch.tensor([d[0], d[1], 0.0], device=device).expand(n, 3)
        # track rotation by how far the currently-up body axis tilts from vertical
        up_idx = int(scene.body_up_idx()[0])
        u_b = torch.zeros(n, 3, device=device)
        u_b[:, up_idx // 2] = 1.0 if up_idx % 2 == 0 else -1.0
        ff = 0.16 if over_curb else 0.15
        kp = 0.07
        w_des = 1.6 if over_curb else 1.8
        cut = math.radians(69.0 if over_curb else 55.0)
        press = 0.7 if over_curb else 0.0
        released = False
        for i in range(420):
            u_w = quat_apply(scene.die.data.root_quat_w, u_b)
            ang = float(torch.acos(u_w[0, 2].clamp(-1.0, 1.0)))
            if ang > cut:
                released = True
                print(f"[solve]   tumble {move}{' (curb)' if over_curb else ''}: "
                      f"released at {math.degrees(ang):.1f} deg (step {i})", flush=True)
                break
            w_a = (scene.die.data.root_ang_vel_w * axis_w).sum(dim=-1)
            tau = (ff + kp * (w_des - w_a)).clamp(-0.10, 0.30)
            # Frame pre-encode: this IsaacLab applies wrenches dragged by the body's
            # rotation since the FIRST application (stale R_ref, persists across
            # resets). Encode the desired WORLD wrench into the body frame per step
            # and use the default body-frame call — exact at every orientation.
            q = scene.die.data.root_quat_w
            tau_b = quat_apply_inverse(q, tau.unsqueeze(-1) * axis_w)
            f_b = quat_apply_inverse(q, press * d_w)
            scene.die.set_external_force_and_torque(
                f_b.unsqueeze(1), tau_b.unsqueeze(1), env_ids=all_ids)
            env.step(no_action)
            if i % 100 == 99:
                p = scene.die.data.root_pos_w[0]
                w = scene.die.data.root_ang_vel_w[0]
                print(f"[solve]   tumble {move} dbg step {i}: ang={math.degrees(ang):.1f} deg "
                      f"pos=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f}) "
                      f"w=({float(w[0]):+.2f},{float(w[1]):+.2f},{float(w[2]):+.2f}) "
                      f"tau={float(tau[0]):.3f}", flush=True)
        clear_wrench()
        if not released:
            p = scene.die.data.root_pos_w[0]
            q = scene.die.data.root_quat_w[0]
            print(f"[solve]   tumble {move} STUCK: ang={math.degrees(ang):.1f} deg "
                  f"pos=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f}) "
                  f"quat=({float(q[0]):+.4f},{float(q[1]):+.4f},{float(q[2]):+.4f},"
                  f"{float(q[3]):+.4f})", flush=True)
            return False
        return settle()

    # ---------------- phase 0: reset, settle, baseline ------------------------------------------
    step(180)
    loc0 = scene.die_tray_local()[0]
    red0 = red_snap()
    print(f"[solve] layout readback (seed {args.seed}): tray_xy="
          f"({float(scene.tray_xy[0, 0]):+.3f},{float(scene.tray_xy[0, 1]):+.3f}) "
          f"die tray-local=({float(loc0[0]):+.3f},{float(loc0[1]):+.3f}) "
          f"red_dir={red0} init_up={int(scene.init_up[0])}", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert not bool(scene.in_tray()[0]), "die must start outside the tray"
    assert red0 != (0, 0, 1), "red must not start up (start set excludes it)"
    assert not bool(scene.red_up()[0]), "red_up must be False at reset"
    s0 = print_score("P0 reset+settle (die outside, red not up)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: staging tumbles — park red on world -z (DOWN) --------------------
    plan = plan_moves(red0)
    print(f"[solve] tumble plan from red={red0}: {plan} "
          f"(then rim vault + seat tumble toward +x)", flush=True)
    # staging point: open floor, clear of the tray for any <= 2-move excursion
    sx = float(scene.tray_xy[0, 0]) - 0.36
    sy = float(scene.tray_xy[0, 1]) + 0.12
    teleport_die(sx, sy)
    assert settle(), "die must settle at the staging point"
    red_v = red_snap()
    for mv in plan:
        pos_before = scene.die.data.root_pos_w[0, 0:2].clone()
        assert do_tumble(mv), f"staging tumble {mv} did not complete"
        red_v = _apply(_MOVES[mv][0], red_v)
        got = red_snap()
        assert got == red_v, f"tumble {mv}: predicted red {red_v}, readback {got}"
        d = _MOVES[mv][1]
        disp = scene.die.data.root_pos_w[0, 0:2] - pos_before
        adv = float(disp[0]) * d[0] + float(disp[1]) * d[1]
        assert adv > 0.06, f"tumble {mv} advanced only {adv:.3f} m (expect ~{c.die_s})"
        print(f"[solve]   tumble {mv}: red now {got}, advanced {adv:.3f} m", flush=True)
    assert red_v == _TARGET, f"after plan, red must sit on -z (down), got {red_v}"
    report("staged")
    assert bool(scene._tumbled[0]), "tumbled latch must be set by the settled staging tumbles"
    s1 = print_score("P1 staging tumbles done (red parked on world -z, DOWN)")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_tumbled - 1e-6, f"P1 score {s1} (expect >= {c.w_tumbled})"

    # ---------------- phase 2: transport to the approach point ----------------------------------
    # die +x face 2 mm shy of the -x curb's outer face, centered on the tray
    x_app = float(scene.tray_xy[0, 0]) - (c.inner / 2 + c.wall_t + c.die_s / 2 + 0.002)
    y_app = float(scene.tray_xy[0, 1])
    teleport_die(x_app, y_app)
    assert settle(), "die must settle at the approach point"
    r = scene.red_world()[0]
    assert float(r[2]) < -0.95, f"red must face DOWN at the approach point, red_w z={float(r[2]):.3f}"
    assert not bool(scene.in_tray()[0]), "die must still be outside the tray pre-entry"
    report("approach")
    s2 = print_score("P2 at the approach point (red down, outside the rim)")
    assert s2 >= s1 - 1e-6, "score decreased across transport"

    # ---------------- phase 3: rim vault + in-tray seat tumble ----------------------------------
    # Vault 1: pivot on the rim's top edge; the forced +x quarter-turn sends red
    # -z -> -x and the die lands in a shallow lean against the inner curb.
    assert do_tumble("px", over_curb=True), "rim vault tumble did not complete"
    assert settle(900), "die must settle after the rim vault"
    r = scene.red_world()[0]
    assert float(r[0]) < -0.90, f"rim vault must land red on -x, red_w x={float(r[0]):.3f}"
    report("vaulted")
    # Vault 2: ordinary forward tumble inside the tray — pivot on the leading
    # bottom edge at slab level; red -x -> UP, die lands flat near tray center.
    assert do_tumble("px"), "in-tray seat tumble did not complete"
    assert settle(900), "die must settle inside the tray"
    step(120)  # hands-off margin before judging
    report("entered")
    assert bool(scene.in_tray()[0]), "die must rest fully inside the tray"
    assert bool(scene.red_up()[0]), "the forced entry quarter-turn must land red up"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after curb entry)", flush=True)
        os._exit(1)
    s3 = print_score("P3 die inside the tray, red up (settled)")
    assert s3 >= s2 - 1e-6, "score decreased across the entry tumble"
    assert s3 >= 0.999, f"success must score 1.0, got {s3}"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, hands-off) ------------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s4 >= s3 - 1e-6
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
    except Exception as exc:  # noqa: BLE001 — fail fast, don't hang until the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({exc})", flush=True)
        os._exit(1)

"""Teleport solution for DiceTumbleScene (sim_gen task `open_oven_i6`) — the task's
legitimacy certificate.

Teleportation handles TRANSPORT ONLY — and here there is nothing to transport: the
dice spawn on the floor and stay on the floor. EVERY goal-directed change goes
through contact dynamics, per die:
  1. PLAN (read-only): read both pad colors and the die's orientation; compute the
     tumble sequence on the cube's own face grid (one tip if the target face is on a
     side, two same-way tips if it is underneath).
  2. TUMBLE (contact): a slowly ramped horizontal push applied at edge height
     (force at the COM plus the matching couple — exactly the wrench a fingertip
     pressed high on the face exerts). Above the tip threshold the ground edge
     becomes a transient hinge: the die pivots, the push is CUT just past the 45 deg
     balance point, and gravity finishes the quarter-turn. Re-plan after every tip.
  3. SLIDE (contact): a velocity-regulated push at LOW effective height (couple
     shifts the application point ~20 mm above the ground — below the tip threshold)
     slides the die flat across the floor and across the pad marking (the pad has
     no collider — nothing to climb), and is cut near the pad center. A tilt guard
     cuts the push if the die ever starts to tip so it falls back onto the same face.
Nothing is ever teleported at all — no write_root_state after reset; the rubric state
(face-up, on-pad, settled) is reached exclusively by pushing.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.open_oven_i6.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.dice_tumble")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(3, device=device)
    face_names = [nm for nm, _col in c.palette]

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def wrench(nm: str, f3: torch.Tensor, t3: torch.Tensor) -> None:
        """Apply a WORLD wrench to die `nm`, expressed in its CURRENT link frame
        (`is_global=True` silently drops the torque on this stack — transform
        manually, the pc_ram_smoke convention). Re-set every step while pushing."""
        from isaaclab.utils.math import quat_apply_inverse

        body = scene.dice[nm]
        q = body.data.root_link_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f3.view(1, 3).expand(n, 3)).unsqueeze(1),
            quat_apply_inverse(q, t3.view(1, 3).expand(n, 3)).unsqueeze(1),
            env_ids=all_ids)

    def face_normals(nm: str) -> torch.Tensor:
        """(6, 3): world-frame outward normal of each face, palette order."""
        from isaaclab.utils.math import quat_apply

        q = scene.dice[nm].data.root_quat_w[0]
        r3 = quat_apply(q.unsqueeze(0).expand(3, 4),
                        torch.eye(3, device=device))  # rows: world of body x, y, z
        return torch.stack([r3[0], -r3[0], r3[1], -r3[1], r3[2], -r3[2]])

    def up_face(nm: str) -> str:
        return face_names[int(face_normals(nm)[:, 2].argmax())]

    def die_xy(nm: str) -> torch.Tensor:
        return (scene.dice[nm].data.root_pos_w - scene.env_origins)[0, :2]

    def pad_xy(p: int) -> torch.Tensor:
        pp, _pq = scene._pad_state(p)
        return (pp - scene.env_origins)[0, :2]

    def report(tag: str) -> None:
        bits = []
        for nm in scene.die_names:
            xy = die_xy(nm)
            bits.append(f"{nm}=({float(xy[0]):+.3f},{float(xy[1]):+.3f}) up={up_face(nm)}")
        for p in range(2):
            bits.append(f"pad{p}[{face_names[int(scene.pad_color[0, p])]}] "
                        f"served={bool(scene.served(p)[0])}")
        print(f"[solve] {tag:14s} | " + " | ".join(bits)
              + f" | success={bool(scene.success()[0])} "
                f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def settle_die(nm: str, max_steps: int = 480) -> None:
        wrench(nm, zero3, zero3)
        for _ in range(max_steps // 30):
            step(30)
            if bool(scene.settled(nm)[0]):
                break

    # ---------------- contact primitives ---------------------------------------------------
    def tip_once(nm: str, u: torch.Tensor, watch: int) -> None:
        """One quarter-turn: ramped horizontal push along `u` (unit, horizontal) at edge
        height — force at the COM plus the couple of an application point ~45 mm above
        it. Cut just past the balance point (watch-face dot > 0.72 ~ 46 deg); gravity
        finishes the turn."""
        die = scene.dice[nm]
        f3 = torch.zeros(3, device=device)
        t3 = torch.zeros(3, device=device)
        for i in range(360):
            f = 1.8 + 1.1 * min(i / 240.0, 1.0)  # ramp 1.8 -> 2.9 N (tip thr ~1.81,
            f3[0], f3[1] = f * u[0], f * u[1]     # static slide thr ~3.1)
            # couple of moving the application point 0.045 m above the COM:
            # tau = (0.045 z) x F
            t3[0], t3[1] = -0.045 * f3[1], 0.045 * f3[0]
            wrench(nm, f3, t3)
            env.step(no_action)
            m = float(face_normals(nm)[watch][2])
            if m > 0.72:  # PAST the 45 deg balance point (sin 46) — gravity finishes.
                break     # (Cutting any earlier makes the die fall back flat.)
        settle_die(nm)

    def tumble_to_color(nm: str, p: int) -> bool:
        """Tip die `nm` until arena-pad `p`'s color faces up. Re-plans from readback
        after every tip; returns True on success."""
        tgt = int(scene.pad_color[0, p])
        for attempt in range(6):
            nrm = face_normals(nm)
            d = nrm[tgt]
            if float(d[2]) > 0.97:
                return True
            delta = pad_xy(p) - die_xy(nm)
            to_pad = delta / delta.norm()
            if float(d[2]) < -0.5:
                # target underneath: two same-way tips; pick the side face normal
                # closest to the pad direction and push along it (walks toward the pad)
                best, best_dot = -1, -2.0
                for i in range(6):
                    if abs(float(nrm[i][2])) < 0.6:
                        h = nrm[i][:2] / nrm[i][:2].norm()
                        dd = float(h[0] * to_pad[0] + h[1] * to_pad[1])
                        if dd > best_dot:
                            best, best_dot = i, dd
                u = nrm[best][:2] / nrm[best][:2].norm()
                watch = best ^ 1  # opposite face (palette pairs +/-) comes up
            else:
                # target on a side: one tip pushing INTO it (u = -horizontal(d),
                # automatically a face normal) brings it up
                u = -d[:2] / d[:2].norm()
                watch = tgt
            print(f"[solve] {nm} tip {attempt}: target={face_names[tgt]} "
                  f"d_z={float(d[2]):+.2f} u=({float(u[0]):+.2f},{float(u[1]):+.2f}) "
                  f"watch={face_names[watch]}", flush=True)
            tip_once(nm, u, watch)
        return bool(float(face_normals(nm)[tgt][2]) > 0.97)

    def slide_to_pad(nm: str, p: int, tgt: int) -> bool:
        """Slide die `nm` flat to pad `p`'s center: velocity-regulated horizontal push
        with the application point 30 mm BELOW the COM (effective height 20 mm —
        under the tip threshold even at the force cap); the pad is a collision-free
        floor marking, so the run to its center is one flat slide. A tilt guard cuts
        the push before an incipient tip passes the balance point (the die falls
        back flat); if a tip nevertheless completed — die settled with the face
        lost — returns False so the caller re-tumbles. Stall recovery (stiction
        corner cases) backs off and re-charges FASTER, never higher."""
        die = scene.dice[nm]
        max_f, dz_app, v_hi = 4.5, -0.030, 0.22
        best, last_bump, guards = 99.0, 0, 0
        f3 = torch.zeros(3, device=device)
        t3 = torch.zeros(3, device=device)

        def push(vdes: torch.Tensor, cap: float) -> None:
            v = die.data.root_lin_vel_w[0, :2]
            f2 = 30.0 * (vdes - v)
            fn = float(f2.norm())
            if fn > cap:
                f2 = f2 * (cap / fn)
            elif fn < 3.4 and float(v.norm()) < 0.02:
                # stiction floor: the PD demand sags below the ~3.1 N static
                # threshold near the target — push through the deadband
                f2 = f2 * (3.4 / max(fn, 1e-6))
            f3[0], f3[1] = f2[0], f2[1]
            t3[0], t3[1] = -dz_app * f3[1], dz_app * f3[0]  # tau = (dz_app z) x F
            wrench(nm, f3, t3)
            env.step(no_action)

        for i in range(3000):
            delta = pad_xy(p) - die_xy(nm)
            dist = float(delta.norm())
            if dist < 0.014:
                break
            if float(face_normals(nm)[tgt][2]) < 0.866:  # tilting past ~30 deg
                wrench(nm, zero3, zero3)                 # cut -> falls back flat
                step(150)
                guards += 1
                if guards <= 4:
                    print(f"[solve] {nm} slide tilt-guard fired at d={dist:.3f}",
                          flush=True)
                if float(face_normals(nm)[tgt][2]) < 0.97:
                    # a tip completed: settled with the face LOST — re-tumble
                    print(f"[solve] {nm} lost the face during slide — re-tumbling",
                          flush=True)
                    settle_die(nm)
                    return False
                continue
            push((delta / dist) * min(v_hi, 3.0 * dist), max_f)
            if dist < best - 0.004:
                best, last_bump = dist, i
            elif i - last_bump > 200:  # stalled: back off, charge again faster
                max_f = min(max_f + 0.5, 6.0)
                v_hi = min(v_hi + 0.05, 0.32)
                nrm = face_normals(nm)
                horiz = nrm[:, :2] / nrm[:, :2].norm(dim=-1, keepdim=True).clamp_min(1e-6)
                align = float((horiz @ (delta / dist)).max())  # 1.0 = face-first at lip
                z = float(scene.dice[nm].data.root_pos_w[0, 2])
                print(f"[solve] {nm} slide stalled at d={dist:.3f} align={align:.2f} "
                      f"z={z:.4f} — backing off, recharge at {v_hi:.2f} m/s "
                      f"cap {max_f:.1f} N", flush=True)
                for _ in range(75):  # rebuild momentum for the lip
                    delta = pad_xy(p) - die_xy(nm)
                    push(-(delta / delta.norm()) * 0.15, 4.5)
                last_bump = i
        settle_die(nm)
        return True

    def serve(nm: str, p: int, tag: str, s_prev: float) -> float:
        """Full contact pipeline for one die: tumble to the pad color, then slide onto
        the pad. Retries the whole pipeline if the placement check fails."""
        tgt = int(scene.pad_color[0, p])
        for round_ in range(4):
            ok = tumble_to_color(nm, p)
            assert ok, f"{nm}: tumble to {face_names[tgt]} failed after retries"
            if round_ == 0:
                report(f"{tag}-tumbled")
                s_mid = print_score(f"{tag} tumble {nm} -> {face_names[tgt]} (contact)")
                assert s_mid >= s_prev - 1e-6, "score decreased across tumble"
                s_prev = s_mid
            slide_to_pad(nm, p, tgt)
            if bool((scene.face_up(nm, p) & scene.on_pad(nm, p))[0]):
                break
            print(f"[solve] {nm} placement check failed (round {round_}) — retrying",
                  flush=True)
        report(f"{tag}-placed")
        s_end = print_score(f"{tag} slide {nm} onto pad{p} (contact)")
        assert s_end >= s_prev - 1e-6, "score decreased across slide"
        return s_end

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(90)
    for p in range(2):
        idx = int(scene.pad_color[0, p])
        xy = pad_xy(p)
        print(f"[solve] layout readback (seed {args.seed}): pad{p} color={face_names[idx]} "
              f"at ({float(xy[0]):+.3f},{float(xy[1]):+.3f})", flush=True)
    for nm in scene.die_names:
        xy = die_xy(nm)
        print(f"[solve] layout readback (seed {args.seed}): {nm} at "
              f"({float(xy[0]):+.3f},{float(xy[1]):+.3f}) up={up_face(nm)}", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phases 1+2: serve both pads (nearest-assignment order) ----------------
    a, b = scene.die_names
    straight = float((die_xy(a) - pad_xy(0)).norm() + (die_xy(b) - pad_xy(1)).norm())
    swapped = float((die_xy(a) - pad_xy(1)).norm() + (die_xy(b) - pad_xy(0)).norm())
    order = [(a, 0), (b, 1)] if straight <= swapped else [(a, 1), (b, 0)]
    s1 = serve(order[0][0], order[0][1], "P1", s0)
    s2 = serve(order[1][0], order[1][1], "P2", s1)

    # ---------------- phase 3: settle to success -------------------------------------------
    for _ in range(16):  # up to 4 s of hands-off settling
        if bool(scene.success()[0]):
            break
        step(30)
    report("settled")
    s3 = print_score("P3 all settled")
    assert s3 >= s2 - 1e-6, "score decreased across settling"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after tumble+slide+settle)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3 simulated seconds, no intervention) -------
    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                for nm in scene.die_names:
                    die = scene.dice[nm]
                    lv = float(die.data.root_lin_vel_w[0].norm())
                    av = float(die.data.root_ang_vel_w[0].norm())
                    print(f"[solve] persist flicker @step {i}: {nm} up={up_face(nm)} "
                          f"lin={lv:.4f} ang={av:.4f} "
                          f"onpad=({bool(scene.on_pad(nm, 0)[0])},"
                          f"{bool(scene.on_pad(nm, 1)[0])})", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)

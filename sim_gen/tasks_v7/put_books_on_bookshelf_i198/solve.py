"""solve — demonstration solution for TiltShelfButtressScene (put_books_on_bookshelf_i198).

Scene-level env (robot="null"). Teleports are used for TRANSPORT ONLY: every body is
teleported to a pose slightly ABOVE its placement spot and RELEASED — the load-bearing
interaction of this task, a body coming to rest SUPPORTED by the structure on a slope
too steep to stand on, happens entirely through contact dynamics. Nothing is ever
spawned seated or wedged: the bookend drops onto the bare slope and friction seats it;
every book drops with a small downhill lean and is CAUGHT by the structure (bookend,
then book on book) — if the support were absent the book would topple and slide off
(exactly what the smoke negatives show). Mid-build, the row is compacted with a snug
PRESS: a fingertip stand-in CoM force (the scene's `drive_sel`/`drive_f` wrench slot)
pushes the row downhill against the bookend, then releases.

Build order (forced by the physics): bookend at the sampled downhill end, then books
0..3 growing uphill. Prints `SIM_GEN_SCORE <v>` at phase boundaries (asserted
non-decreasing: 0.000 -> 0.250 -> 0.400 -> 0.550 -> 0.550 (press) -> 0.700 -> 0.850 ->
1.000). After success() first holds, keeps simulating >= 3.5 more simulated seconds
hands-off (drives asserted zero); only if success() still holds prints exactly
`SIM_GEN_SOLVE: SUCCESS`. Hard exit (os._exit) after the verdict, watchdog Timer as
backstop — Kit teardown hangs otherwise.

Run (forge): python -u -m simgen_tasks.put_books_on_bookshelf_i198.solve --headless [--seed N]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_sec", type=float, default=900.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.put_books_on_bookshelf_i198 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()

U_BUT = -0.13  # uphill coordinate of the bookend center on the plank (m)
CLEAR = 0.003  # placement clearance between a new book's base face and the structure (m)
LEAN = 0.06  # downhill pre-lean at release (rad, ~3.4 deg) — falls INTO the support
DROP = 0.006  # release height above the geometric rest pose (m)
PRESS_N = 0.8  # snug-press CoM force (N, fingertip scale)
PRESS_STEPS = 120  # 1.0 s at 1/120


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tilt_shelf_buttress")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    ids0 = torch.tensor([0], device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        bo = scene.books_ok()[0].tolist()
        print(f"[solve] {tag:12s} but_ok={bool(scene.buttress_ok()[0])} "
              f"books_ok={[int(x) for x in bo]} chain={float(scene.chain_count()[0]):.0f} "
              f"but_latch={float(scene.but_latch[0]):.2f} k_latch={float(scene.k_latch[0]):.2f} "
              f"settled={bool(scene.settled()[0])} score={sc():.3f} "
              f"success={bool(scene.success()[0])}", flush=True)

    last_score = -1.0

    def phase_score(tag: str) -> None:
        nonlocal last_score
        s = sc()
        assert s >= last_score - 1e-6, f"score decreased at {tag}: {last_score} -> {s}"
        last_score = s
        print(f"SIM_GEN_SCORE {s:.3f}", flush=True)

    def verdict(ok: bool) -> None:
        print("SIM_GEN_SOLVE: SUCCESS" if ok else "SIM_GEN_SOLVE: FAIL", flush=True)
        code = 0 if ok else 1
        threading.Timer(10.0, lambda: os._exit(code)).start()
        try:
            env.close()
            app.close()
        except Exception:  # noqa: BLE001
            pass
        os._exit(code)

    def settle(max_steps: int = 480, thr: float = 0.03, need: int = 30) -> bool:
        streak = 0
        for _ in range(max_steps):
            step(1)
            _p, _q, lv, av = scene._body_tensors()
            if float(lv.max()) < thr and float(av.max()) < 0.4:
                streak += 1
                if streak >= need:
                    return True
            else:
                streak = 0
        return False

    # ================= reset + settle ==========================================================
    env.reset(seed=args.seed)
    step(60)
    s = float(scene.tilt_sign[0])
    theta = float(scene.tilt[0])
    phi = float(scene.phi[0])
    ppos = scene.plank_pos[0].tolist()
    origin = scene.env_origins[0]
    print(f"[solve] seed={args.seed} tilt={math.degrees(theta):.2f}deg sign={s:+.0f} "
          f"(downhill = {'-y' if s > 0 else '+y'}) plank_y={ppos[1]:+.3f}", flush=True)
    report("reset")
    assert not bool(scene.buttress_ok()[0]) and float(scene.chain_count()[0]) == 0.0
    assert float(scene.drive_f.abs().max()) == 0.0 and int(scene.drive_sel[0]) == -1
    phase_score("reset")  # ~0.000

    cphi, sphi = math.cos(phi), math.sin(phi)

    def world_from_local(lx: float, ly: float, lz: float) -> tuple[float, float, float]:
        """Plank-frame (depth, slope, normal) -> env-local world."""
        return (ppos[0] + lx,
                ppos[1] + ly * cphi - lz * sphi,
                ppos[2] + ly * sphi + lz * cphi)

    def tp(body, u: float, lz: float, lean: float) -> None:
        """TRANSPORT: teleport a body to plank-local uphill coord `u`, normal offset `lz`,
        tilted downhill by `lean` — always strictly ABOVE its rest pose (released, not
        seated: contact dynamics take it from here)."""
        x, y, z = world_from_local(0.0, s * u, lz)
        a = phi + s * lean
        st = torch.zeros(1, 13, device=device)
        st[0, 0] = x
        st[0, 1] = y
        st[0, 2] = z
        st[0, 3] = math.cos(a / 2)
        st[0, 4] = math.sin(a / 2)
        st[0, 0:3] += origin
        body.write_root_state_to_sim(st, ids0)

    def u_of(body) -> float:
        """Readback: uphill coordinate of a body center (plank frame)."""
        p = (body.data.root_pos_w[0] - origin - scene.plank_pos[0])
        ly = float(p[1]) * cphi + float(p[2]) * sphi  # R_x(-phi) applied to the y row
        return s * ly

    # ================= PHASE 1: install the bookend at the downhill end ========================
    tp(scene.buttress, U_BUT, c.plank_size[2] / 2 + c.but_size[2] / 2 + DROP, 0.0)
    ok = settle()
    report("buttress")
    assert ok and bool(scene.buttress_ok()[0]), "bookend must seat on the slope"
    phase_score("buttress")  # 0.250

    # ================= PHASE 2: books 0..3, each dropped INTO the structure ====================
    for k in range(c.n_books):
        t, h = c.book_t[k], c.book_h[k]
        face = (u_of(scene.buttress) + c.but_size[1] / 2 if k == 0
                else u_of(scene.books[k - 1]) + c.book_t[k - 1] / 2)
        u_c = face + CLEAR + t / 2 + (h / 2) * math.sin(LEAN)
        lz = c.plank_size[2] / 2 + (h / 2) * math.cos(LEAN) + (t / 2) * math.sin(LEAN) + DROP
        tp(scene.books[k], u_c, lz, LEAN)
        ok = settle()
        report(f"book{k}")
        assert ok, f"book{k} did not settle"
        assert bool(scene.books_ok()[0, k]), f"book{k} must stand on the structure"
        assert float(scene.chain_count()[0]) >= k + 1, f"chain must include book{k}"
        phase_score(f"book{k}")  # 0.25 + 0.15*(k+1)

        if k == 1:
            # ----- snug press (contact interaction): compact the 2-book row downhill ------
            scene.drive_sel[0] = 1
            scene.drive_f[0] = torch.tensor(
                [0.0, -s * math.cos(theta) * PRESS_N, -math.sin(theta) * PRESS_N],
                device=device)
            step(PRESS_STEPS)
            scene.drive_sel[0] = -1
            scene.drive_f[0] = 0.0
            ok = settle()
            report("press")
            assert ok and float(scene.chain_count()[0]) >= 2, "press must keep the row intact"
            phase_score("press")  # 0.550 (unchanged — press must not destroy credit)

    # ================= PHASE 3: success + persistence (>= 3.5 s hands off) =====================
    reached = False
    for _ in range(120):
        if bool(scene.success()[0]):
            reached = True
            break
        step(10)
    report("success")
    if not reached:
        print("[solve] PHASE 3 FAILED: success() not reached after the build", flush=True)
        verdict(False)
    phase_score("success")  # 1.000

    assert float(scene.drive_f.abs().max()) == 0.0 and int(scene.drive_sel[0]) == -1, \
        "drives must be zero for persistence"
    persist_steps = int(round(3.5 / env.dt))  # 420 physics steps at 1/120 s
    step(persist_steps)
    report("final")
    phase_score("final")
    still_ok = bool(scene.success()[0]) and abs(sc() - 1.0) < 1e-3
    print(f"[solve] persistence: {persist_steps} steps ({persist_steps * env.dt:.2f} s) "
          f"hands-off, success={still_ok}", flush=True)
    verdict(still_ok)


if __name__ == "__main__":
    main()

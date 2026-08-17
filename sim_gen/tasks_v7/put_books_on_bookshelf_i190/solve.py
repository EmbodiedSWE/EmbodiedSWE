"""solve — demonstration solution for LibrarianExtractScene (put_books_on_bookshelf_i190).

Scene-level env (robot="null"). Teleports are used for TRANSPORT ONLY (carrying the
already-tipped book from the case mouth to the tray — the trivial free-space move). The
load-bearing interaction — tipping the requested book out of its packed slot, the only
maneuver the geometry admits — happens entirely through contact dynamics: the solver
writes the scene's `drive_f`/`drive_t` buffers as a fingertip stand-in (a horizontal
pull at the top of the spine: force F at body point (d/2, 0, h/2-0.012), applied as the
equivalent CoM wrench F + r x F) and the book pivots on its bottom-front edge against
gravity, friction and the case floor.

  Tip servo: outer rate command w_des = clamp(K_ANG*(THETA_STOP - th), W_MIN, W_CAP)
  on the forward-pivot angle, inner force F = k_ff*F_ff(th) + KF*(w_des - w_fd),
  0 <= F <= 0.9*mu*m*g (never enough to slide the base: pivot breakaway needs only
  ~0.62*mu*m*g). F_ff is the quasi-static hold force m*g*L*sin(alpha-th)/(lever*cos th)
  (alpha = atan(d/h) is the topple angle — THETA_STOP = 31.5 deg stays >= 2.9 deg below
  every book's alpha, so the book is gravity-restoring the whole way and a stall means
  falling BACK, never toppling out). Pivot rate from finite differences (root_ang_vel_w
  is noisy under external wrenches); the gain product KF*lever^2*dt/I_pivot ~ 0.35
  respects the one-substep wrench delay. Stalls escalate the FEEDFORWARD multiplier,
  never past the slide budget.

At the tip peak (the grab moment — the book is self-restoring at 31.5 deg, so it is
held, not parked), the drives are zeroed and the book is teleported (transport) to
2 cm above the tray floor, covers down; it drops, settles flat, and success() judges
the live scene. Prints `SIM_GEN_SCORE <v>` at phase boundaries (asserted
non-decreasing). After success() first holds, keeps simulating >= 3.5 more simulated
seconds hands-off (drives asserted zero); only if success() still holds prints exactly
`SIM_GEN_SOLVE: SUCCESS`. Hard exit (os._exit) after the verdict, watchdog Timer as
backstop — Kit teardown hangs otherwise.

Run (forge): python -u -m simgen_tasks.put_books_on_bookshelf_i190.solve --headless [--seed N]
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
from isaaclab.utils.math import quat_apply  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.put_books_on_bookshelf_i190 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()

GRAV = 9.81
THETA_STOP = math.radians(31.5)  # grab angle: > tip_grab_deg=30, >= 2.9 deg under min alpha
K_ANG = 2.5  # 1/s outer angle->rate gain
W_MIN = 0.05  # rad/s floor so the approach never parks short of the latch angle
W_CAP = 0.50  # rad/s ceiling (tip takes ~1.5 s — quasi-static, fingertip-plausible)
KF_FRAC = 0.35  # inner rate gain as a fraction of the delay-stability bound I/(lever^2*dt)
K_FF_MAX = 1.45  # feedforward escalation ceiling (k_ff*F_ff(0) stays <= F_MAX)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.librarian_extract")().build(num_envs=1, device=device)
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
        print(f"[solve] {tag:12s} pitch={math.degrees(float(scene.pitch_fwd()[0])):+6.2f}deg "
              f"tip_latch={float(scene.tip_latch[0]):.3f} out_latch={float(scene.out_latch[0]):.0f} "
              f"others_ok={bool(scene.others_ok()[0])} tray_ok={bool(scene.tray_ok()[0])} "
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

    # ================= reset + settle ==========================================================
    env.reset(seed=args.seed)
    step(90)
    tgt = int(scene.target_idx[0])
    name = c.book_colors[tgt][0]
    t, h, m = c.book_t[tgt], c.book_h[tgt], c.book_mass[tgt]
    tray_y = float(scene.tray_y[0])
    print(f"[solve] seed={args.seed} target=book{tgt} ({name}, t={t * 1000:.0f}mm, "
          f"h={h * 1000:.0f}mm, m={m:.3f}kg) slot_y={float(scene.slot_y[0, tgt]):+.3f} "
          f"row_c={float(scene.row_c[0]):+.3f} tray_y={tray_y:+.3f}", flush=True)
    report("reset")
    assert bool(scene.others_ok()[0]), "row must start undisturbed"
    assert abs(math.degrees(float(scene.pitch_fwd()[0]))) < 3.0, "target must start upright"
    assert float(scene.drive_f.abs().max()) == 0.0 and float(scene.drive_t.abs().max()) == 0.0
    phase_score("reset")  # ~0.000

    # ================= PHASE 1: tip the requested book out of the row (contact dynamics) =======
    book = scene.books[tgt]
    lever = h - 0.012  # fingertip contact height above the bottom-front pivot edge
    r_b = torch.tensor([c.book_d / 2, 0.0, h / 2 - 0.012], device=device)  # contact, body frame
    L = math.hypot(c.book_d / 2, h / 2)  # pivot -> CoM distance
    alpha = math.atan2(c.book_d, h)  # topple angle atan(d/h)
    i_piv = m * (c.book_d ** 2 + h ** 2) / 3.0  # inertia about the pivot edge
    kf = KF_FRAC * i_piv / (lever * env.dt)  # N/(rad/s): torque gain kf*lever meets I*KF_FRAC/dt
    f_max = 0.9 * c.friction * m * GRAV  # never exceed the base friction budget (no slide)
    print(f"[solve] tip plant: alpha={math.degrees(alpha):.1f}deg L={L:.3f}m "
          f"lever={lever:.3f}m I_piv={i_piv:.4f} KF={kf:.2f}N/(rad/s) "
          f"F_ff(0)={m * GRAV * L * math.sin(alpha) / lever:.2f}N F_max={f_max:.2f}N", flush=True)

    k_ff = 1.0
    prev_th = float(scene.pitch_fwd()[0])
    stall_th, stall_i = prev_th, 0
    tipped = False
    for i in range(2400):
        th = float(scene.pitch_fwd()[0])
        w = (th - prev_th) / env.dt  # FD pivot rate (root_ang_vel_w noisy under wrenches)
        prev_th = th
        if th >= THETA_STOP:
            tipped = True
            break
        w_des = min(W_CAP, max(W_MIN, K_ANG * (THETA_STOP - th)))
        f_ff = m * GRAV * L * math.sin(alpha - th) / (lever * math.cos(th))
        f = min(f_max, max(0.0, k_ff * max(0.0, f_ff) + kf * (w_des - w)))
        f_w = torch.tensor([f, 0.0, 0.0], device=device)
        r_w = quat_apply(book.data.root_quat_w[0], r_b)
        scene.drive_f[0] = f_w
        scene.drive_t[0] = torch.linalg.cross(r_w, f_w)
        step(1)
        if i - stall_i >= 180:  # stall watch: escalate the FEEDFORWARD, never the slide budget
            if th - stall_th < math.radians(1.0) and k_ff < K_FF_MAX:
                k_ff = min(K_FF_MAX, k_ff * 1.25)
                print(f"[solve] tip stalled at {math.degrees(th):.1f}deg -> k_ff={k_ff:.2f}",
                      flush=True)
            stall_th, stall_i = th, i
        if i and i % 240 == 0:
            report(f"tipping@{i}")
    scene.drive_f[0] = 0.0
    scene.drive_t[0] = 0.0
    if not tipped:
        report("tip-fail")
        print("[solve] PHASE 1 FAILED: tip servo did not reach the grab angle", flush=True)
        verdict(False)
    report("tip-peak")
    assert float(scene.tip_latch[0]) >= 0.999, "tip stage must be fully latched at the peak"
    assert bool(scene.others_ok()[0]), "row must be undisturbed at the grab"
    phase_score("tipped")  # ~0.350

    # ================= PHASE 2: grab at the peak, transport to the tray (teleport), lay flat ===
    # The book is gravity-restoring at 31.5 deg — it is HELD at the peak (the grab moment),
    # then carried out over the tray: TRANSPORT ONLY. Covers-down: R_x(90 deg) maps the
    # thickness axis (body y) to world z; drop height 2 cm onto the restitution-0 floor.
    st = torch.zeros(1, 13, device=device)
    st[0, 0] = c.tray_x
    st[0, 1] = tray_y
    st[0, 2] = c.tray_floor_top + t / 2 + 0.02
    st[0, 3] = math.sqrt(0.5)  # w
    st[0, 4] = math.sqrt(0.5)  # x: R_x(90 deg)
    st[0, 0:3] += scene.env_origins[0]
    book.write_root_state_to_sim(st, ids0)
    step(30)
    report("placed")
    assert float(scene.out_latch[0]) >= 1.0, "carry-out must have latched"

    settled = False
    for _ in range(90):  # up to 7.5 s for the drop to ring down and success() to hold
        if bool(scene.success()[0]):
            settled = True
            break
        step(10)
    if not settled:
        report("settle-fail")
        print("[solve] PHASE 2 FAILED: success() not reached after placement", flush=True)
        verdict(False)
    report("success")
    phase_score("placed")  # 1.000

    # ================= PHASE 3: persistence (>= 3.5 simulated seconds, hands off) ==============
    assert float(scene.drive_f.abs().max()) == 0.0 and float(scene.drive_t.abs().max()) == 0.0, \
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

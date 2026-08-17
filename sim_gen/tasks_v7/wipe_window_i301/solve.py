"""solve — demonstration solution for GlazeWindowScene (wipe_window_i301).

Scene-level env (robot="null"). Teleports are used for TRANSPORT ONLY — carrying a
body that is already airborne across free space (cap tray -> parking hover, stand ->
above-frame hover, parking -> above-tray hover). Every load-bearing interaction runs
through contact dynamics driven by the scene's `drive_f` buffer (the stand-in for a
fingertip pinch, forces at the CoM, hard-capped at 4 N):

  1. UNCAP — vertical velocity-cascade lift (gravity feedforward + rate servo,
     K*dt/m ~ 0.11 << 1 against the one-substep wrench delay) raises the cap out of
     its nub tray; teleport-hover over an empty patch of table; free gravity drop;
     settle. The cap_off latch fires on readback.
  2. GLAZE — same lift servo raises the pane out of its stand slot; teleport-hover
     above the frame's now-open top; then a contact DESCENT: gravity-feedforward rate
     servo (slowing near the groove mouth) plus a small (0.3 N) frame-local lateral
     centering servo on the pane's BOTTOM point threads the 12 mm plate down between
     the posts, past the mullions, into the 22 mm sill groove; brief 0.6 N seat press;
     release; the seated latch matures through the scene's slow-gate streak. A stalled
     descent (pane resting on a rail top) is retried: lift back out, re-hover, descend
     again.
  3. RECAP — lift the parked cap off the table, teleport-hover centered above the
     tray, guided descent onto the post tops between the nubs, release, settle. The
     capped latch (order-aware: it requires the pane seated SIMULTANEOUSLY) matures.

Prints `SIM_GEN_SCORE <v>` at phase boundaries (asserted non-decreasing). After
success() first holds, keeps simulating >= 3.5 more simulated seconds with the drive
buffer zero (asserted); only if success() still holds prints exactly
`SIM_GEN_SOLVE: SUCCESS`. Hard exit (os._exit) after the verdict, watchdog Timer as
backstop — Kit teardown hangs otherwise.

Run (forge): python -u -m simgen_tasks.wipe_window_i301.solve --headless [--seed N]
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
    from simgen_tasks.wipe_window_i301 import scene as scene_mod
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod

CAP = scene_mod.CAP
PANE = scene_mod.PANE

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()

G = 9.81
F_MAX = 4.0  # N hard cap on any drive component — fingertip-pinch authority
KZ = 2.0  # N*s/m vertical rate gain (K*dt/m: 0.11 pane, 0.14 cap — << 1)
V_UP = 0.15  # m/s commanded lift rate
V_DOWN = 0.15  # m/s commanded descent rate (free corridor)
V_DOWN_SLOW = 0.06  # m/s near the groove mouth / tray
KP_XY = 20.0  # N/m lateral centering position gain
KD_XY = 1.0  # N*s/m lateral damping (K*dt/m ~ 0.06)
F_XY = 0.3  # N lateral force cap — nudge, not a grip


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.glaze_window")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    ids0 = torch.tensor([0], device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def fyaw() -> float:
        return float(scene.frame_yaw[0])

    def report(tag: str) -> None:
        b = scene.to_frame(scene.pane_bottom_env())[0]
        p = scene.to_frame(scene.cap_pos())[0]
        print(f"[solve] {tag:12s} pane_bot=({float(b[0]):+.3f},{float(b[1]):+.3f},"
              f"{float(b[2]):.3f}) cap=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):.3f}) latches=[off={bool(scene.cap_off_latch[0])} "
              f"seat={bool(scene.seated_latch[0])} cap={bool(scene.capped_latch[0])}] "
              f"score={sc():.3f} success={bool(scene.success()[0])}", flush=True)

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

    def teleport(body, row: int, pos_env, yaw: float) -> None:
        """TRANSPORT ONLY: place an already-airborne body at a free-space hover pose,
        velocities zeroed. Never used to create a load-bearing contact."""
        scene.drive_f[0, row, :] = 0.0
        st = torch.zeros(1, 13, device=device)
        st[0, 0] = scene.env_origins[0, 0] + pos_env[0]
        st[0, 1] = scene.env_origins[0, 1] + pos_env[1]
        st[0, 2] = pos_env[2]
        st[0, 3] = math.cos(yaw / 2)
        st[0, 6] = math.sin(yaw / 2)
        body.write_root_state_to_sim(st, ids0)
        step(1)

    def lift_to(body, row: int, mass: float, z_env: float, timeout: int = 600) -> bool:
        """Vertical velocity-cascade lift (contact break-away + carry) until the body
        centre reaches z_env. Gravity feedforward keeps the cascade linear."""
        for _ in range(timeout):
            z = float(body.data.root_pos_w[0, 2])
            if z >= z_env:
                scene.drive_f[0, row, :] = 0.0
                return True
            vz = float(body.data.root_lin_vel_w[0, 2])
            f = mass * G + KZ * (V_UP - vz)
            scene.drive_f[0, row, 2] = max(0.0, min(F_MAX, f))
            step(1)
        scene.drive_f[0, row, :] = 0.0
        return False

    def guided_descent(body, row: int, mass: float, bottom_fn, z_goal: float,
                       z_slow: float, timeout: int = 900) -> bool:
        """Contact descent: gravity-ff rate servo down + frame-local lateral centering
        (0.3 N cap) on the tracked point, until the point reaches z_goal (frame z above
        table) or the descent stalls on an obstruction. Returns True iff goal reached."""
        z_hist: list[float] = []
        for k in range(timeout):
            b = scene.to_frame(bottom_fn())[0]
            bz = float(b[2])
            if bz <= z_goal:
                scene.drive_f[0, row, :] = 0.0
                return True
            # stall watch: bottom height frozen well above the goal = parked on a rail
            z_hist.append(bz)
            if len(z_hist) > 150 and abs(z_hist[-1] - z_hist[-150]) < 0.002 \
                    and bz > z_goal + 0.015:
                scene.drive_f[0, row, :] = 0.0
                print(f"[solve] descent stalled at z={bz:.3f} (goal {z_goal:.3f})",
                      flush=True)
                return False
            v = body.data.root_lin_vel_w[0]
            v_des = -V_DOWN if bz > z_slow else -V_DOWN_SLOW
            fz = mass * G + KZ * (v_des - float(v[2]))
            # lateral: frame-local error of the tracked point -> world force
            cy, sy = math.cos(fyaw()), math.sin(fyaw())
            vx_f = cy * float(v[0]) + sy * float(v[1])
            vy_f = -sy * float(v[0]) + cy * float(v[1])
            fx_f = max(-F_XY, min(F_XY, -KP_XY * float(b[0]) - KD_XY * vx_f))
            fy_f = max(-F_XY, min(F_XY, -KP_XY * float(b[1]) - KD_XY * vy_f))
            scene.drive_f[0, row, 0] = cy * fx_f - sy * fy_f
            scene.drive_f[0, row, 1] = sy * fx_f + cy * fy_f
            scene.drive_f[0, row, 2] = max(0.0, min(F_MAX, fz))
            step(1)
        scene.drive_f[0, row, :] = 0.0
        return False

    # ================= reset + settle ==========================================================
    env.reset(seed=args.seed)
    step(60)
    print(f"[solve] seed={args.seed} "
          f"frame=({float(scene.frame_xy[0, 0]):.3f},{float(scene.frame_xy[0, 1]):.3f},"
          f"yaw={fyaw():+.3f}) stand=({float(scene.stand_xy[0, 0]):.3f},"
          f"{float(scene.stand_xy[0, 1]):.3f},yaw={float(scene.stand_yaw[0]):+.3f})",
          flush=True)
    report("reset")
    assert float(scene.drive_f.abs().max()) == 0.0
    assert bool(scene.cap_seated_now()[0]), "cap must start seated in its tray"
    assert not bool(scene.capped_latch[0]), "order-aware latch must NOT fire on an empty frame"
    assert not bool(scene.seated_latch[0]) and not bool(scene.cap_off_latch[0])
    assert sc() < 0.02, f"reset score must be ~0, got {sc()}"
    phase_score("reset")  # ~0.000

    # ================= phase 1: uncap ==========================================================
    # Lift the cap out of its nub tray (contact break-away via force), then transport
    # to a parking hover over an empty patch of table on the side AWAY from the stand.
    z_clear = c.table_top + c.post_top + 0.02 + 0.06  # centre well above the nub tops
    if not lift_to(scene.cap, CAP, c.cap_mass, z_clear):
        report("cap-lift-fail")
        print("[solve] PHASE 1 FAILED: cap never lifted clear of the tray", flush=True)
        verdict(False)
    park_y = -0.38 if float(scene.stand_xy[0, 1]) >= 0.0 else 0.38
    park = (-0.02, park_y)
    teleport(scene.cap, CAP, (park[0], park[1], c.table_top + 0.03), 0.0)
    step(90)  # free drop ~2 cm + settle
    if not bool(scene.cap_off_latch[0]):
        report("cap-off-fail")
        print("[solve] PHASE 1 FAILED: cap_off latch never fired", flush=True)
        verdict(False)
    report("cap-parked")
    phase_score("cap-off")  # 0.200

    # ================= phase 2: glaze — seat the pane in the sill groove =======================
    # Lift the pane out of its stand slot, transport to a hover centred above the
    # frame's open top, then thread it down into the groove under contact.
    if not lift_to(scene.pane, PANE, c.pane_mass, c.table_top + 0.24):
        report("pane-lift-fail")
        print("[solve] PHASE 2 FAILED: pane never lifted out of the stand", flush=True)
        verdict(False)
    seated = False
    for attempt in range(3):
        hover = scene.frame_point_env((0.0, 0.0, 0.41))[0]
        teleport(scene.pane, PANE, (float(hover[0]), float(hover[1]), float(hover[2])),
                 fyaw())
        if guided_descent(scene.pane, PANE, c.pane_mass, scene.pane_bottom_env,
                          z_goal=c.groove_floor + 0.005, z_slow=0.13):
            # brief seat press: settle the bottom edge onto the groove floor
            scene.drive_f[0, PANE, 2] = -0.6
            step(30)
            scene.drive_f[0, PANE, :] = 0.0
            step(60)  # hands off: slow-gate streak matures
            if bool(scene.pane_seated_now()[0]) and bool(scene.seated_latch[0]):
                seated = True
                break
            print(f"[solve] attempt {attempt}: contact reached but seat not latched",
                  flush=True)
        # stalled on a rail top (or bad seat) — lift back out through the open top, retry
        print(f"[solve] attempt {attempt}: retrying insertion", flush=True)
        if not lift_to(scene.pane, PANE, c.pane_mass, c.table_top + 0.41):
            break
    if not seated:
        report("seat-fail")
        print("[solve] PHASE 2 FAILED: pane never seated in the groove", flush=True)
        verdict(False)
    report("pane-seated")
    phase_score("seated")  # 0.600

    # ================= phase 3: recap ==========================================================
    # Lift the parked cap off the table, transport to a hover centred above the tray,
    # guided descent onto the post tops between the nubs.
    if not lift_to(scene.cap, CAP, c.cap_mass, c.table_top + 0.12):
        report("cap-relift-fail")
        print("[solve] PHASE 3 FAILED: cap never lifted off the table", flush=True)
        verdict(False)
    hover = scene.frame_point_env((0.0, 0.0, 0.36))[0]
    teleport(scene.cap, CAP, (float(hover[0]), float(hover[1]), float(hover[2])), fyaw())
    ok = guided_descent(scene.cap, CAP, c.cap_mass, scene.cap_pos,
                        z_goal=c.cap_rest_z + 0.004, z_slow=0.34)
    scene.drive_f[0, CAP, :] = 0.0
    step(90)  # hands off: cap settles into the nubs, order-aware streak matures
    if not (ok and bool(scene.cap_seated_now()[0]) and bool(scene.capped_latch[0])):
        report("recap-fail")
        print("[solve] PHASE 3 FAILED: cap never seated back in its tray", flush=True)
        verdict(False)
    report("recapped")
    phase_score("recap")  # 0.850 + success -> 1.000 once settled

    # ================= success must hold =======================================================
    okf = False
    for _ in range(48):  # up to 4 s for everything to settle
        if bool(scene.success()[0]):
            okf = True
            break
        step(10)
    if not okf:
        report("settle-fail")
        print("[solve] FINAL FAILED: success() not reached after recap", flush=True)
        verdict(False)
    report("assembled")
    phase_score("assembled")  # 1.000

    # ================= persistence (>= 3.5 simulated seconds, hands off) =======================
    assert float(scene.drive_f.abs().max()) == 0.0, "drive must be zero for persistence"
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

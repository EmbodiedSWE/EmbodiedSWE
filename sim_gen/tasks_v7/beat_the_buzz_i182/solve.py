"""solve — teleport solution for PressurePlateHeistScene (beat_the_buzz_i182).

Scene-level env (robot="null"). Teleports handle TRANSPORT ONLY (carrying a lifted
body across free space); every load-bearing interaction goes through contact
dynamics:

  PHASE 1 (counterweight): the granite block is LIFTED off the table with a
  vertical velocity-servo force through its CoM (the stand-in for the Franka's
  pinch-grasp lift, TASK.md), carried (teleport, zero velocity) to a hover point
  8 mm above the pressure plate on the far side from the idol, and RELEASED — it
  falls onto the plate and settles by real contact. The plate must stay down
  (idol + granite both on it): readback asserted.

  PHASE 2 (idol): the idol is lifted straight up with the same kind of servo
  force — the plate stays pinned by the granite (its extension is monitored the
  whole lift and must never approach the alarm trigger), carried to 8 mm above
  the sampled delivery pad, and RELEASED to drop and settle upright by contact.

  The spring/alarm plant runs in scene.post_step every step; nothing is ever
  pinned, and the alarm latch judges the WHOLE trajectory — a solution that
  unloaded the plate at any moment would zero its own score.

Servo sizing (post_step wrenches act one step late, so K*dt/m stays well under 1):
idol m=0.5 kg KV=10 -> 0.167; granite m=0.7 kg KV=12 -> 0.143. Gravity
feedforward + velocity servo, V_CAP=0.25 m/s; forces through the CoM so the
bodies do not rotate during the lift.

Prints `SIM_GEN_SCORE <v>` at phase boundaries (asserted non-decreasing). After
success() first holds, keeps simulating >= 3.5 more simulated seconds with the
drive buffers zero (asserted); only if success() still holds prints exactly
`SIM_GEN_SOLVE: SUCCESS`. Hard exit (os._exit) after the verdict, watchdog Timer
as backstop — Kit teardown hangs otherwise.

Run (forge): python -u -m simgen_tasks.beat_the_buzz_i182.solve --headless [--seed N]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_sec", type=float, default=600.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import os  # noqa: E402
import threading  # noqa: E402

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.beat_the_buzz_i182 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()

G = 9.81
V_CAP = 0.25  # m/s max lift rate
K_APP = 3.0  # v_des = clamp(K_APP * (z_des - z))
KV_IDOL = 10.0  # N*s/m (KV*dt/m = 0.167 — delay-stable)
KV_GRAN = 12.0  # N*s/m (0.143)
F_MIN, F_MAX = -3.0, 16.0  # N clamp on the vertical drive
LIFT_Z = 0.70  # env-local carry height (> clear_z, above everything)
HOVER = 0.008  # release height above the target surface
V_DONE = 0.05  # m/s: lift phase exits only when slow


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.pressure_plate_heist")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    origin = scene.env_origins[0]

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def epos(body) -> torch.Tensor:
        return body.data.root_pos_w[0] - origin

    def report(tag: str) -> None:
        print(f"[solve] {tag:12s} ext={float(scene.plate_ext()[0]) * 1000:6.2f}mm "
              f"alarm={int(scene.alarm[0])} "
              f"hold={float(scene.hold_latch[0]):.0f} clear={float(scene.clear_latch[0]):.0f} "
              f"pad={float(scene.pad_latch[0]):.0f} score={sc():.3f} "
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

    def lift(slot: int, body, mass: float, kv: float, z_des: float,
             budget: int = 900) -> tuple[bool, float]:
        """Vertical velocity-servo lift through the CoM to env-local height z_des.
        Returns (converged, max plate extension seen during the lift)."""
        max_ext = 0.0
        ok = False
        for _ in range(budget):
            z = float(epos(body)[2])
            v = float(body.data.root_lin_vel_w[0, 2])
            if abs(z - z_des) < 0.02 and abs(v) < V_DONE:
                ok = True
                break
            v_des = max(-V_CAP, min(V_CAP, K_APP * (z_des - z)))
            f = mass * G + kv * (v_des - v)
            scene.drive_f[0, slot, 2] = max(F_MIN, min(F_MAX, f))
            step(1)
            max_ext = max(max_ext, float(scene.plate_ext()[0]))
        return ok, max_ext

    def carry(slot: int, body, mass: float, pos_env: torch.Tensor,
              quat: torch.Tensor | None = None) -> None:
        """TRANSPORT: hold the body still, then set its pose across free space with
        zero velocity (the load-bearing set-down happens by contact after release)."""
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = pos_env + origin
        st[0, 3:7] = body.data.root_quat_w[0] if quat is None else quat
        scene.drive_f[0, slot] = 0.0
        body.write_root_state_to_sim(st, torch.tensor([0], device=device))
        # hold against gravity for the settle-in step so the write is not fought
        scene.drive_f[0, slot, 2] = mass * G
        step(1)

    def release_and_settle(slot: int, body, budget: int = 600, streak_need: int = 30) -> bool:
        """Zero the drive, let the body fall/settle by contact; settle is streak-gated
        (a velocity threshold alone fires vacuously at turning points)."""
        scene.drive_f[0, slot] = 0.0
        streak = 0
        for _ in range(budget):
            step(1)
            v = float(body.data.root_lin_vel_w[0].norm())
            w = float(body.data.root_ang_vel_w[0].norm())
            streak = streak + 1 if (v < 0.04 and w < 0.8) else 0
            if streak >= streak_need:
                return True
        return False

    # ================= reset + settle ==========================================================
    torch.manual_seed(args.seed)
    env.reset()
    step(60)
    ped = torch.tensor(c.pedestal_xy, device=device)
    g_xy = epos(scene.granite)[0:2].clone()
    i_xy = epos(scene.idol)[0:2].clone()
    pad_xy = epos(scene.pad)[0:2].clone()
    try:
        m_g = float(scene.granite.root_physx_view.get_masses().reshape(-1)[0])
        m_f = float(scene.foam.root_physx_view.get_masses().reshape(-1)[0])
        m_i = float(scene.idol.root_physx_view.get_masses().reshape(-1)[0])
        print(f"[solve] mass readback: granite={m_g:.3f}kg foam={m_f:.3f}kg "
              f"idol={m_i:.3f}kg (cfg {c.granite_mass}/{c.foam_mass}/{c.idol_mass})",
              flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[solve] mass readback unavailable ({exc!r})", flush=True)
    print(f"[solve] seed={args.seed} granite_side={int(scene.granite_side[0])} "
          f"granite=({float(g_xy[0]):+.3f},{float(g_xy[1]):+.3f}) "
          f"pad=({float(pad_xy[0]):+.3f},{float(pad_xy[1]):+.3f}) "
          f"idol_off=({float(i_xy[0] - ped[0]):+.3f},{float(i_xy[1] - ped[1]):+.3f})",
          flush=True)
    report("reset")
    assert float(scene.drive_f.abs().max()) == 0.0
    phase_score("reset")  # ~0.000

    # ================= PHASE 1: counterweight the plate ========================================
    ok, _ = lift(1, scene.granite, c.granite_mass, KV_GRAN, LIFT_Z)
    if not ok:
        report("lift-g-fail")
        print("[solve] PHASE 1 FAILED: granite lift servo did not converge", flush=True)
        verdict(False)
    report("granite-up")
    # hover spot: on the plate, opposite side from the idol (>= 62 mm from its centre,
    # inside the fully-on-plate band; separation to the idol >= 62 mm > the 59.3 mm
    # worst-case contact bound)
    off = i_xy - ped
    u = -off / max(float(off.norm()), 1e-6) if float(off.norm()) > 0.003 \
        else torch.tensor([1.0, 0.0], device=device)
    plate_top = float(epos(scene.plate)[2]) + c.plate_size[2] / 2
    hover = torch.tensor([0.0, 0.0, 0.0], device=device)
    hover[0:2] = ped + u * 0.062
    hover[2] = plate_top + c.block_s / 2 + HOVER
    carry(1, scene.granite, c.granite_mass, hover,
          quat=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))
    if not release_and_settle(1, scene.granite):
        report("g-settle-fail")
        print("[solve] PHASE 1 FAILED: granite did not settle on the plate", flush=True)
        verdict(False)
    report("granite-on")
    if not (bool(scene.granite_on_plate()[0]) and bool(scene.plate_down()[0])
            and float(scene.alarm[0]) == 0.0):
        print("[solve] PHASE 1 FAILED: counterweight not seated / plate not down", flush=True)
        verdict(False)
    phase_score("phase1")  # 0.350 (hold latch)

    # ================= PHASE 2: steal the idol =================================================
    ok, max_ext = lift(0, scene.idol, c.idol_mass, KV_IDOL, LIFT_Z)
    print(f"[solve] idol lift: max plate ext during lift = {max_ext * 1000:.2f}mm "
          f"(trigger {c.trigger_h * 1000:.0f}mm)", flush=True)
    if not ok or float(scene.alarm[0]) != 0.0:
        report("lift-i-fail")
        print("[solve] PHASE 2 FAILED: idol lift failed or alarm fired", flush=True)
        verdict(False)
    report("idol-up")
    hover = torch.tensor([0.0, 0.0, 0.0], device=device)
    hover[0:2] = pad_xy
    hover[2] = c.pad_top + c.idol_h / 2 + HOVER
    carry(0, scene.idol, c.idol_mass, hover)
    if not release_and_settle(0, scene.idol):
        report("i-settle-fail")
        print("[solve] PHASE 2 FAILED: idol did not settle on the pad", flush=True)
        verdict(False)
    ok = False
    for _ in range(48):  # up to 4 s for everything to co-settle
        if bool(scene.success()[0]):
            ok = True
            break
        step(10)
    report("delivered")
    if not ok:
        print("[solve] PHASE 2 FAILED: success() not reached after delivery", flush=True)
        verdict(False)
    phase_score("phase2")  # 1.000

    # ================= PHASE 3: persistence (>= 3.5 simulated seconds, hands off) ==============
    assert float(scene.drive_f.abs().max()) == 0.0, "drives must be zero for persistence"
    persist_steps = int(round(3.5 / env.dt))
    step(persist_steps)
    report("final")
    phase_score("final")
    still_ok = bool(scene.success()[0]) and abs(sc() - 1.0) < 1e-3
    print(f"[solve] persistence: {persist_steps} steps ({persist_steps * env.dt:.2f} s) "
          f"hands-off, success={still_ok}", flush=True)
    verdict(still_ok)


if __name__ == "__main__":
    main()

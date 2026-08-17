"""solve — TELEPORT-contract solution for BatchScaleScene (open_oven_i213).

Scene-level env (robot="null"). Teleport is used for TRANSPORT ONLY: each selected
counterweight's root pose is written ONCE, to a free-space release pose inside the
tray's open airspace (bottom face ~10 mm above the tray floor, clear of the rim and of
every other body), with zero velocity. Everything load-bearing happens through the live
dynamics after the release: the cube FALLS onto the tray floor under gravity, the
spring plant (scene.post_step, k = 78.48 N/m, c = 14 N*s/m) sinks under the new resting
load, rings once (zeta ~ 0.7..1.3 by design) and settles with the pointer one tick
lower per 120 g. Nothing else is ever written: the tray, marker, mast and unused
weights keep their reset states; the scene's `ext_force`/`ext_torque` probe buffers
stay zero for the whole run (asserted at every phase boundary — this solver never
pushes the tray); no velocity, no rubric state, no joint state is touched.

Plan (per episode):
  READ    the target from the SCENE, the way a vision policy would: read back the green
          tab's z, count ticks down from the white zero (u* = (z_zero - z_tab)/delta);
          asserted equal to the scene's sampled target.
  COMBO   greedy big-first exact decomposition of u* into the available denominations
          (2x iron = 2 units, 3x brass = 1 unit): 2 -> [big], 3 -> [big, small],
          4 -> [big, big], 5 -> [big, big, small], 6 -> [big, big, small, small].
          The resting load only ever GROWS toward u* -> the printed score sequence is
          monotone by construction.
  DROP    per weight: teleport to a free hover pose over an unoccupied tray quadrant
          (release z recomputed from the CURRENT tray readback — the tray is lower
          after every drop), release, hands off; wait for the scene's own accounted
          streak (deflection matches resting load, everything slow) before the next.
  VERIFY  settle_until success() — pointer in the band, load accounted, streak held.
  PERSIST >= 3 more simulated seconds hands-off (probe buffers asserted zero); only if
          success() still holds print exactly `SIM_GEN_SOLVE: SUCCESS`.

Prints `SIM_GEN_SCORE <score>` at every phase boundary (never decreasing — asserted).
Hard exit (os._exit) after the verdict, watchdog Timer as backstop (Kit teardown
hangs otherwise).

The single-Franka-arm strategy for the same plan (top-down pinch of 40/50 mm cubes —
both well inside the 80 mm jaw — carry at ~0.35 m height, open the gripper over the
tray quadrant; base at the origin faces the whole workspace at 0.30..0.68 m reach)
lives in TASK.md as the embodiment argument.

Run (forge): python -u -m simgen_tasks.open_oven_i213.solve --headless [--seed N]
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
    from simgen_tasks.open_oven_i213 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()

# Exact big-first decompositions of every sampled target into the manifest
# (2x big = 2 units, 3x small = 1 unit). Load only ever grows toward u*.
COMBOS = {
    2: ("big_0",),
    3: ("big_0", "small_0"),
    4: ("big_0", "big_1"),
    5: ("big_0", "big_1", "small_0"),
    6: ("big_0", "big_1", "small_0", "small_1"),
}
# Tray quadrants (xy offsets from the tray centre) — one per dropped weight, so no
# release pose ever overlaps an earlier cube (slot spacing 84 mm > cube 50 mm).
QUADRANTS = ((-0.042, -0.042), (0.042, -0.042), (-0.042, 0.042), (0.042, 0.042))


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.batch_scale")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    ids = torch.tensor([0], device=device)
    no_action = torch.empty(0, device=device)
    sizes = {name: size for name, size, _m, _u in c.manifest}

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def probes_zero() -> bool:
        return float(scene.ext_force.abs().max()) == 0.0 and \
            float(scene.ext_torque.abs().max()) == 0.0

    def report(tag: str) -> None:
        d = float(scene.deflection()[0]) * 1000
        t = float(scene.target_deflection()[0]) * 1000
        u = float(scene.units_on()[0])
        print(f"[solve] {tag:14s} defl={d:6.1f}mm tgt={t:5.1f}mm units_on={u:.2f} "
              f"acc={bool(scene.accounted()[0])} streak={int(scene.acc_streak[0])} "
              f"score={sc():.3f} success={bool(scene.success()[0])}", flush=True)

    last_score = -1.0

    def phase_score(tag: str) -> None:
        nonlocal last_score
        assert probes_zero(), f"probe buffers touched by {tag} — solver never pushes"
        s = sc()
        assert s >= last_score - 1e-6, f"score decreased at {tag}: {last_score} -> {s}"
        last_score = s
        print(f"SIM_GEN_SCORE {s:.3f}", flush=True)

    def verdict(ok: bool) -> None:
        if ok:
            print("SIM_GEN_SOLVE: SUCCESS", flush=True)
        else:
            print("SIM_GEN_SOLVE: FAIL", flush=True)
        code = 0 if ok else 1
        threading.Timer(10.0, lambda: os._exit(code)).start()
        try:
            env.close()
            app.close()
        except Exception:  # noqa: BLE001
            pass
        os._exit(code)

    def settle_until(pred, max_steps: int = 900, poll: int = 10) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def drop_weight(name: str, quadrant: tuple, expect_units: int) -> bool:
        """TRANSPORT-ONLY teleport of one counterweight to a free hover pose inside the
        tray airspace, then hands-off: gravity and the spring plant do all the work.
        Returns True iff the scene's own bookkeeping confirms the new resting load."""
        half = sizes[name] / 2
        plate_top = (float(scene.tray.data.root_pos_w[0, 2])
                     - float(scene.env_origins[0, 2]) + c.plate_size[2] / 2)
        st = torch.zeros(1, 13, device=device)
        st[0, 0] = c.scale_xy[0] + quadrant[0]
        st[0, 1] = c.scale_xy[1] + quadrant[1]
        st[0, 2] = plate_top + half + 0.010  # bottom face 10 mm above the CURRENT floor
        st[0, 3] = 1.0  # identity yaw — lands square in its quadrant
        st[0, 0:3] += scene.env_origins[0]
        scene.weights[name].write_root_state_to_sim(st, ids)
        print(f"[solve] drop {name}: released 10 mm over quadrant "
              f"({quadrant[0]:+.3f},{quadrant[1]:+.3f}), tray floor at "
              f"{plate_top:.3f} m -> expect {expect_units} units resting", flush=True)

        def resting() -> bool:
            return (abs(float(scene.units_on()[0]) - expect_units) < 0.25
                    and bool(scene.accounted()[0])
                    and int(scene.acc_streak[0]) >= c.latch_streak)

        ok = settle_until(resting, max_steps=900)
        report(f"after-{name}")
        return ok

    # ================= reset + settle ===========================================================
    env.reset(seed=args.seed)
    step(120)  # let the tray find its empty equilibrium and the weights seat on the floor
    report("reset")
    assert probes_zero()
    d0 = abs(float(scene.deflection()[0]))
    assert d0 < 0.004, f"empty tray should rest at the zero tick, defl={d0 * 1000:.1f}mm"
    phase_score("reset")  # ~0.000

    # ================= PHASE 1: read the target off the mast (readback, not privilege) ==========
    z_tab = float(scene.marker.data.root_pos_w[0, 2]) - float(scene.env_origins[0, 2])
    u_read = int(round((c.z_zero - z_tab) / c.delta_unit))
    u_true = int(scene.target_units[0])
    print(f"[solve] read: green tab at z={z_tab:.4f} -> {u_read} ticks below zero; "
          f"scene sampled u*={u_true}", flush=True)
    assert u_read == u_true, "tick count read off the mast must match the sampled target"
    combo = COMBOS[u_read]
    print(f"[solve] combo (big-first exact): {list(combo)} = {u_read} units", flush=True)
    phase_score("read")

    # ================= PHASE 2: load the tray, one hands-off drop at a time ====================
    expect = 0
    for k, name in enumerate(combo):
        expect += 2 if name.startswith("big") else 1
        if not drop_weight(name, QUADRANTS[k], expect):
            print(f"[solve] PHASE 2 FAILED: {name} not resting/accounted at "
                  f"{expect} units", flush=True)
            verdict(False)
        phase_score(f"drop-{name}")

    # ================= PHASE 3: pointer on the tab — the scene's own success ===================
    if not settle_until(lambda: bool(scene.success()[0]), max_steps=600):
        print("[solve] PHASE 3 FAILED: success() not reached with the exact load resting",
              flush=True)
        verdict(False)
    report("on-target")
    phase_score("on-target")  # ~1.000

    # ================= PHASE 4: persistence (>= 3 simulated seconds, hands off) ================
    assert probes_zero(), "probe buffers must be zero for persistence"
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

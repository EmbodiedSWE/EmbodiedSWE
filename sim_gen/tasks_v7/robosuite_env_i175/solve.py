"""solve — demonstration solution for CounterBalanceScene (robosuite_env_i175).

Scene-level env (robot="null"). Teleports are TRANSPORT ONLY: each chosen
counterweight is teleported to a hover point 10 mm above its seat in the counter
pan (computed from the pan's LIVE pose readback) and released with zero velocity
— the drop, the pan swing, the beam's rotation to equilibrium and every judged
degree of levelness flow through the passive D6 + gravity plant. Nothing ever
places the beam, and nothing touches the beam directly at all.

Plan (no execution order is required by the task; largest-first is a convention):
  1. reset + settle: the sampled cargo (k units) pins the beam at its stop.
  2. Read k off the scene, take its unique binary decomposition over the {1,2,4}
     weight set, and drop the chosen weights one at a time into the EMPTY pan,
     waiting out the quiet streak between drops (partial credit latches mature).
  3. When the exact subset is aboard, the beam — and only the beam — swings
     level and rests inside the +-3 deg band: success.
  4. >= 3.5 more simulated seconds hands off; success must still hold.

Prints `SIM_GEN_SCORE <v>` at phase boundaries (asserted non-decreasing). After
success() first holds, persistence, then exactly `SIM_GEN_SOLVE: SUCCESS`. Hard
exit (os._exit) after the verdict, watchdog Timer as backstop.

Run (forge): python -u -m simgen_tasks.robosuite_env_i175.solve --headless [--seed N]
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

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.robosuite_env_i175 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()

# The unique binary decomposition of every samplable cargo mass over {1, 2, 4}.
SUBSET = {2: [2], 3: [2, 1], 4: [4], 5: [4, 1], 6: [4, 2]}
# Pan-local y seats: one weight sits centred; two sit fore/aft of centre
# (inner half-width 0.079 m, max radius 0.024 m -> 0.040 m seats clear the rims
# and each other with >= 15 mm to spare).
Y_SEATS = {1: [0.0], 2: [-0.040, 0.040]}


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.counter_balance")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    ids = torch.tensor([0], device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        print(f"[solve] {tag:12s} tilt={math.degrees(float(scene.tilt()[0])):+6.2f}deg "
              f"counter={float(scene.counter_mass()[0]) / c.unit_mass:.1f}u "
              f"cargo={int(scene.k[0])}u quiet={bool(scene.quiet()[0])} "
              f"A={float(scene.a_latch[0]):.2f} B={float(scene.b_latch[0]):.2f} "
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

    def diag(tag: str) -> None:
        from isaaclab.utils.math import quat_apply

        parts = []
        for nm, b in scene._bodies().items():
            lv = float(b.data.root_lin_vel_w[0].norm())
            av = float(b.data.root_ang_vel_w[0].norm())
            if lv > 0.02 or av > 0.08:
                parts.append(f"{nm}(lv={lv:.3f},av={av:.3f})")
        cargo = scene.cargos[int(scene.k[0])]
        loc = scene._local(cargo.data.root_pos_w, scene.pans[float(scene.side[0])])[0]
        up = quat_apply(cargo.data.root_quat_w[0:1],
                        torch.tensor([[0.0, 0.0, 1.0]], device=device))[0, 2]
        print(f"[diag] {tag}: movers=[{', '.join(parts) or 'none'}] "
              f"cargo_local=({float(loc[0]):+.3f},{float(loc[1]):+.3f},{float(loc[2]):+.3f}) "
              f"cargo_up_z={float(up):+.3f} quiet_ctr={int(scene.quiet_ctr[0])}", flush=True)

    def wait_quiet(budget: int, tag: str = "") -> bool:
        for i in range(budget // 10):
            if bool(scene.quiet()[0]):
                return True
            step(10)
            if tag and i % 60 == 59:
                diag(f"{tag}@{(i + 1) * 10}")
        return bool(scene.quiet()[0])

    def drop_weight(units: int, y_local: float) -> None:
        """TRANSPORT teleport: hover the chosen weight 10 mm above its seat in the
        counter pan (live pan pose readback), zero velocity, release. Everything
        after the release is contact + joint physics."""
        from isaaclab.utils.math import quat_apply

        pan = scene.pans[float(-scene.side[0])]  # the EMPTY pan
        local = torch.tensor([[0.0, y_local, c.pan_floor[2] / 2 + c.weight_h / 2 + 0.010]],
                             device=device)
        pos = pan.data.root_pos_w[0:1] + quat_apply(pan.data.root_quat_w[0:1], local)
        st = torch.zeros(1, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3] = 1.0
        scene.weights[units].write_root_state_to_sim(st, ids)

    # ================= reset + settle ==============================================
    env.reset(seed=args.seed)
    if not wait_quiet(1440, tag="reset"):
        report("reset-unquiet")
        diag("reset-unquiet")
        print("[solve] FAILED: scale never settled after reset", flush=True)
        verdict(False)
    k = int(scene.k[0])
    side = float(scene.side[0])
    print(f"[solve] seed={args.seed} cargo={k} units in the "
          f"{'RIGHT (+x)' if side > 0 else 'LEFT (-x)'} pan; "
          f"subset={SUBSET[k]}", flush=True)
    report("reset")
    tilt0 = math.degrees(float(scene.tilt()[0]))
    assert abs(tilt0) > 2.0 * c.tol_deg and tilt0 * side > 0, \
        f"beam should rest pinned toward the cargo side, got {tilt0:+.2f} deg"
    phase_score("reset")  # ~0.000

    # ================= drops: the exact binary subset, largest first ===============
    chosen = SUBSET[k]
    seats = Y_SEATS[len(chosen)]
    for i, u in enumerate(chosen):
        drop_weight(u, seats[i])
        step(30)  # fall + first contact
        if not wait_quiet(1200):
            report(f"drop{u}-unquiet")
            print(f"[solve] FAILED: scale never re-settled after dropping w{u}", flush=True)
            verdict(False)
        report(f"dropped-w{u}")
        assert bool(scene.weight_in_pan(-scene.side)[0, list(c.weight_units).index(u)]), \
            f"w{u} did not come to rest inside the counter pan"
        phase_score(f"drop{i + 1}")

    # ================= success =====================================================
    ok = False
    for _ in range(120):  # up to 10 s for level + quiet to mature
        if bool(scene.success()[0]):
            ok = True
            break
        step(10)
    if not ok:
        report("no-success")
        print("[solve] FAILED: success() not reached after the full subset", flush=True)
        verdict(False)
    report("balanced")
    phase_score("balanced")  # 1.000

    # ================= persistence (>= 3.5 simulated seconds, hands off) ==========
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

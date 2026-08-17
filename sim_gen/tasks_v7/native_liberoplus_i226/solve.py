"""solve — demonstration solution for BalanceVerdictScene (native_liberoplus_i226).

Scene-level env (robot="null"). Teleports handle TRANSPORT ONLY: each canister is
teleported to ~2 cm ABOVE a pan and released — the landing, the beam's verdict tilt
(the load-bearing comparison) and the final seating on the pedestal all flow through
contact dynamics. Nothing is ever spawned in a goal or latched state:
  - PHASE 1 (weigh): drop canister 0 on its near pan (beam slams to that stop under
    the single load), then canister 1 on the OTHER pan (readback of the tilted pan's
    world pose); the beam re-settles with the HEAVY side down and the `weighed` latch
    matures through the scene's streak counter.
  - VERDICT: the solver reads the scene's own physical readout (`down_side()`) and
    picks the canister standing on the sunken pan — the oracle then cross-checks that
    choice against the ground-truth masses (they must agree, or the mechanism lies).
  - PHASE 2 (place): teleport the verdict canister to 1.5 cm above the red pedestal's
    top and release; it seats under gravity. The light canister stays on its pan
    (clear of the pedestal by construction).

Prints `SIM_GEN_SCORE <v>` at phase boundaries (asserted non-decreasing: latched
credit must not evaporate). After success() first holds, keeps simulating >= 3.5 more
simulated seconds hands-off (probe buffers asserted zero); only if success() still
holds prints exactly `SIM_GEN_SOLVE: SUCCESS`. Hard exit (os._exit) after the
verdict, watchdog Timer as backstop — Kit teardown hangs otherwise.

Run (forge): python -u -m simgen_tasks.native_liberoplus_i226.solve --headless [--seed N]
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
    from simgen_tasks.native_liberoplus_i226 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()

DROP_H_PAN = 0.012  # release height above the pan seat (m) — transport ends here
DROP_H_PED = 0.015  # release height above the pedestal seat (m)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.balance_verdict")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        tilt = math.degrees(float(scene.beam_tilt()[0]))
        on, side = scene._pan_metrics()
        print(f"[solve] {tag:12s} tilt={tilt:+6.2f}deg on_pan={on[0].tolist()} "
              f"side={side[0].tolist()} pan_latch={scene.pan_latch[0].tolist()} "
              f"weighed={bool(scene.weighed[0])} "
              f"on_ped={bool(scene.heavy_on_pedestal()[0])} "
              f"clear={bool(scene.light_clear()[0])} "
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

    def teleport(j: int, xyz: torch.Tensor, quat: torch.Tensor | None = None) -> None:
        """TRANSPORT ONLY: pose canister j at world xyz, zero velocity. `quat` (wxyz)
        defaults to upright — pass the beam's quat to land flat on a tilted pan."""
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = xyz
        if quat is None:
            st[0, 3] = 1.0
        else:
            st[0, 3:7] = quat
        scene.cans[j].write_root_state_to_sim(st, torch.tensor([0], device=device))

    def wait_until(fn, budget: int, tag: str) -> bool:
        for i in range(budget):
            if bool(fn()):
                return True
            step(1)
            if i and i % 240 == 0:
                report(f"{tag}-wait")
        return bool(fn())

    # ================= reset + settle ==========================================================
    env.reset(seed=args.seed)
    step(60)
    hp = int(scene.heavy_pad[0])
    print(f"[solve] seed={args.seed} heavy_pad={hp} "
          f"(ground truth: can0={c.heavy_mass}kg at pad{hp}, can1={c.light_mass}kg "
          f"at pad{1 - hp}) — the SOLVER decides from the beam readout only", flush=True)
    report("reset")
    assert float(scene.drive_f.abs().max()) == 0.0
    assert abs(math.degrees(float(scene.beam_tilt()[0]))) < 2.0, "beam must start level"
    phase_score("reset")  # ~0.000

    # ================= PHASE 1: weigh (both canisters onto opposite pans) ======================
    origin = scene.env_origins[0]
    from isaaclab.utils.math import quat_apply  # noqa: E402 (post-AppLauncher)

    for j in (0, 1):
        # this canister's near pan = the sign of its env-local y
        side = 1.0 if float((scene.cans[j].data.root_pos_w[0] - origin)[1]) >= 0 else -1.0
        # land FLAT on the (possibly tilted) pan: align with the beam, offset along
        # the beam's up axis — an edge-first landing on a tilted pan topples and rolls
        beam_q = scene.beam.data.root_quat_w[0].clone()
        up = quat_apply(beam_q.unsqueeze(0),
                        torch.tensor([[0.0, 0.0, 1.0]], device=device))[0]
        target = scene.pan_center_w(side)[0] + up * (c.can_h / 2 + DROP_H_PAN)
        teleport(j, target, quat=beam_q)
        ok = wait_until(
            lambda j=j: scene._pan_metrics()[0][0, j]
            and scene.cans[j].data.root_lin_vel_w.norm() < c.can_slow,
            600, f"drop{j}")
        if not ok:
            report(f"drop{j}-fail")
            print(f"[solve] PHASE 1 FAILED: canister {j} did not seat on its pan", flush=True)
            verdict(False)
        step(40)  # let the pan latch mature (24 calm substeps)
        report(f"pan{j}-seated")
        phase_score(f"pan{j}")  # ~0.075 then ~0.15 (+ weighed once calm)

    ok = wait_until(lambda: scene.weighed[0], 720, "weigh")
    if not ok:
        report("weigh-fail")
        print("[solve] PHASE 1 FAILED: weighed latch did not mature", flush=True)
        verdict(False)
    report("weighed")
    phase_score("weighed")  # ~0.400

    # ================= VERDICT: read the beam, pick the sunken pan's canister ==================
    down = float(scene.down_side()[0])
    _on, side = scene._pan_metrics()
    pick = 0 if float(side[0, 0]) == down else 1
    tilt = math.degrees(float(scene.beam_tilt()[0]))
    print(f"[solve] verdict: beam tilt={tilt:+.2f}deg, down side={down:+.0f} -> "
          f"canister {pick} is the heavy one", flush=True)
    # Oracle cross-check: the mechanism must agree with the ground-truth masses.
    assert pick == 0, "beam verdict disagrees with ground truth — the comparator lies!"

    # ================= PHASE 2: seat the verdict canister on the pedestal ======================
    ped = scene.pedestal.data.root_pos_w[0].clone()
    target = ped.clone()
    target[2] += c.ped_h / 2 + c.can_h / 2 + DROP_H_PED
    teleport(pick, target)
    ok = wait_until(lambda: scene.success()[0], 720, "seat")
    if not ok:
        report("seat-fail")
        print("[solve] PHASE 2 FAILED: success() not reached after seating", flush=True)
        verdict(False)
    report("seated")
    phase_score("seated")  # 1.000

    # ================= PHASE 3: persistence (>= 3.5 simulated seconds, hands off) ==============
    assert float(scene.drive_f.abs().max()) == 0.0, "probe buffers must be zero for persistence"
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

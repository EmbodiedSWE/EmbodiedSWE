"""Teleport solution for WeightCrankScene (sim_gen task
`libero_kitchen_scene5_close_the_top_drawer_of_the_cabinet_i163`) — the task's
legitimacy certificate.

Teleports are used for TRANSPORT ONLY: one teleport carries the brass weight
from the apron bench to a hover just above the crank's receiving pocket (what
a pick-and-carry does). Everything load-bearing after that is contact
dynamics under gravity:

1. PERCEPTION: the drawer's sampled opening q0 and both cubes' apron poses are
   read back from the episode state — never hard-coded. The hover point is
   computed from the MEASURED crank pose.
2. TRANSPORT (teleport): the brass weight is placed at rest just above the
   rubric's load box over the pocket mouth. Nothing else moves; the score is
   unchanged (asserted).
3. RELEASE (gravity does the work): the weight free-falls into the deep
   pocket; its torque overpowers the crank's up-bias; the crank swings down
   its arc while the red roller nose cam-presses the drawer's front plate;
   the drawer translates shut; the crank lands on its down-stop. The drawer
   is never wrenched, never teleported, never touched — its closing
   translation is delivered entirely by the machine's transmission. The foam
   decoy is never touched.
4. HANDS-OFF: the terminal state is a static force balance (weight-on-pan vs
   crank-on-stop + nose-on-plate); success() turns True on the live settled
   state and persists.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
rubric latches), then holds HANDS-OFF >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still holds.
The whole solve is then repeated on a SECOND seed (fresh episode, no score
prints — the rubric restarts at 0 on reset) to certify seed robustness.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene5_close_the_top_drawer_of_the_cabinet_i163.solve
             --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.weight_crank_cabinet")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    print("[solve] " + "=" * 70, flush=True)
    print(scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def deg(t: torch.Tensor) -> float:
        return math.degrees(float(t[0]))

    def report(tag: str) -> None:
        print(f"[solve] {tag:12s} | q={float(scene.drawer_open()[0]):+.4f} "
              f"theta={deg(scene.crank_theta()):+.2f}deg "
              f"loaded={bool(scene._floaded[0])} fc={float(scene._fcrank[0]):.2f} "
              f"fd={float(scene._fdrawer[0]):.2f} onpan={bool(scene.weight_on_pan()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def episode(seed: int, announce: bool) -> bool:
        """One full episode on `seed`. SIM_GEN_SCORE is printed only when
        `announce` (the rubric restarts at 0 on reset, and the score stream
        must be non-decreasing)."""
        env.reset(seed=seed)
        s_prev = 0.0

        def print_score(tag: str) -> float:
            nonlocal s_prev
            s = float(scene.score()[0])
            print(f"[solve] phase boundary: {tag}", flush=True)
            if announce:
                print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
            assert s >= s_prev - 1e-6, f"score decreased: {s_prev:.4f} -> {s:.4f}"
            s_prev = s
            return s

        # ---------------- phase 0: settle, layout readback, baseline --------------------
        step(120)
        q0 = float(scene.q0[0])
        w = (scene.weight.data.root_pos_w - scene.env_origins)[0]
        d = (scene.decoy.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] layout readback (seed {seed}): drawer opening q0={q0:+.4f} "
              f"weight=({float(w[0]):+.3f},{float(w[1]):+.3f},{float(w[2]):+.3f}) "
              f"decoy=({float(d[0]):+.3f},{float(d[1]):+.3f},{float(d[2]):+.3f}) "
              f"theta={deg(scene.crank_theta()):+.2f}deg", flush=True)
        report("reset")
        assert bool(scene._finite()[0]), "NaN/inf after settle"
        assert abs(float(scene.drawer_open()[0]) - q0) < 0.006, \
            "drawer must rest at its sampled opening"
        assert abs(deg(scene.crank_theta())) < 2.0, "crank must rest on its up-stop"
        assert not bool(scene.success()[0]), "fresh reset must not be success"
        s = print_score("P0 reset+settle (drawer out, crank up, cubes staged)")
        assert s <= 0.05, f"baseline score should be ~0, got {s}"

        # ---------------- phase 1: TRANSPORT — weight to a hover over the pocket --------
        # Hover point from the MEASURED crank pose (crank at rest: local == world
        # offsets). The hover sits just ABOVE the rubric's crank-local load box, so
        # the transport teleport itself earns nothing — the load credit only arrives
        # once gravity has carried the cube down into the pocket.
        pkt_cx = (c.pkt_in_x0 + c.pkt_in_x1) / 2
        hover = scene.crank.data.root_pos_w.clone()
        hover[:, 0] += pkt_cx
        hover[:, 2] += c.load_z[1] + c.cube_s / 2 + 0.004
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = hover
        st[:, 3] = 1.0
        scene.weight.write_root_state_to_sim(st, torch.arange(n, device=device))
        s = print_score("P1 weight transported to hover above the pocket (teleport)")
        assert s <= 0.05 + 1e-6, "transport alone must not earn credit"

        # ---------------- phase 2: RELEASE — gravity drives the transmission ------------
        landed_at = None
        for i in range(240):
            env.step(no_action)
            if bool(scene._floaded[0]):
                landed_at = i
                break
        report("landed")
        assert landed_at is not None, "weight must land in the pocket"
        s = print_score(f"P2 payload landed in the pocket ({landed_at} steps)")
        assert s >= c.w_load - 1e-6, f"P2 score {s:.3f} below load credit"

        won_at = None
        for i in range(720):
            env.step(no_action)
            if bool(scene.success()[0]):
                won_at = i
                break
        report("driven")
        assert won_at is not None and bool(scene.success()[0]), \
            (f"success must arrive hands-off: q={float(scene.drawer_open()[0]):+.4f} "
             f"theta={deg(scene.crank_theta()):+.2f}deg")
        assert float(scene.drawer_open()[0]) <= c.q_goal, "drawer must be seated"
        assert deg(scene.crank_theta()) >= c.crank_goal_deg, "crank must be at its down-stop"
        assert bool(scene.weight_on_pan()[0]), "weight must ride the pan"
        s = print_score(f"P3 crank down, drawer seated ({won_at} hands-off steps)")
        assert s >= 1.0 - 1e-6, "success must score 1.0"

        # ---------------- persistence (>= 3 simulated seconds, hands-off) ---------------
        hold = True
        for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
            step(40)
            hold = hold and bool(scene.success()[0])
        report("persist")
        s_final = print_score("P-final persistence 3.3 s")
        ok = hold and bool(scene.success()[0]) and s_final >= 1.0 - 1e-6
        print(f"[solve] episode seed={seed}: {'OK' if ok else 'FAILED'}", flush=True)
        return ok

    ok = episode(args.seed, announce=True)
    ok = episode(args.seed + 1, announce=False) and ok

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
    except Exception as e:  # noqa: BLE001 - fast fail beats a 20-min watchdog hang
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)

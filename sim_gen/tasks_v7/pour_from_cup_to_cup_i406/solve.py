"""solve — TELEPORT-contract solution for RationTicketScene (pour_from_cup_to_cup_i406).

Scene-level env (robot="null"). Teleports are TRANSPORT ONLY — every load-bearing
interaction ends in a gravity drop through live contact dynamics:

  - each cup is teleported from its mouth-down spawn to UPRIGHT, 20 mm ABOVE its pad,
    with zero velocity, and DROPPED: the landing, the upright settle and the on-pad
    stillness that arm the placed latch are physics, not a written pose;
  - each delivered ball is teleported from its bin rest to 25 mm ABOVE the standing
    cup's mouth and DROPPED: it falls past the rim, hits the cup floor (or the balls
    already inside) and settles by real contact — the in-cup containment the rubric
    counts is a physical outcome. Nothing is ever written INTO a seated/contained pose.
  - the surplus balls are NEVER TOUCHED: "leave the leftovers in the bin" is satisfied
    by doing nothing to them, which is the point of the stopping rule.

The plan the phases execute is the task's own: read the two ticket quotas, right and
place both cups first (nothing can enter a mouth-down cup), then deliver EXACTLY
quota_j counted balls to each cup, then stop while balls remain in the bin.

Phases (each ends with `SIM_GEN_SCORE`, printed sequence never decreases — asserted):
  P0 SETTLE   ~1.0 s hands-off; read back the randomized layout (quotas, surplus,
              present count, pad poses, cup poses); assert both cups mouth-down,
              score ~0 and no success.
  P1 PLACE    for each pad: teleport its cup upright 20 mm over the pad centre and
              drop; wait until the scene's own on_pad() holds (upright + centred +
              bottom on the floor + still). Placed latch arms physically.
  P2 FILL     for each pad: drop quota_j present bin balls one at a time 25 mm over
              the standing cup's mouth (small xy scatter so drops nest, not stack);
              wait for each to settle contained. Overfill never happens — the loop is
              counted; the leftovers stay untouched in the bin.
  P3 CONFIRM  hands-off settle until the scene's own success() holds live
              (exact counts on both pads, leftovers in the bin, everything still).
  P4 PERSIST  >= 3.5 more simulated seconds hands-off; only if success() still holds
              (live state — a ball hopping out or a cup drifting would revert it)
              print exactly `SIM_GEN_SOLVE: SUCCESS`.
Hard exit (os._exit) after the verdict, watchdog Timer as backstop.

The single-Franka-arm strategy for the same plan (pinch-grasp the 72 mm cup across
its flats, right it in the air, set it on the pad; pinch each 28 mm ball out of the
30 mm-walled open bin, release it over the cup mouth) lives in TASK.md.

Run (forge): python -u -m simgen_tasks.pour_from_cup_to_cup_i406.solve --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_sec", type=float, default=1350.0)
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
    from simgen_tasks.pour_from_cup_to_cup_i406 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()

DROP_H_CUP = 0.020    # cup released this far above its landed pose (m)
DROP_H_BALL = 0.025   # ball released this far above the cup rim (m)
# xy scatter for successive ball drops into one cup (m) — inside inner_r - ball_r
SCATTER = ((0.0, 0.0), (0.009, 0.0), (-0.005, 0.008))


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ration_ticket_station")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    origin = scene.env_origins[0]
    assert max(abs(x) for xy in SCATTER for x in xy) < c.cup_inner_r - c.ball_r - 0.003, \
        "ball drop scatter must stay inside the cup mouth"

    print("[solve] describe():", flush=True)
    print(scene.describe(), flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        counts = scene.pad_counts()[0].tolist()
        quota = scene.quota[0].tolist()
        onp = scene.on_pad()[0]
        n_bin = int((scene.in_bin()[0] & scene.present[0]).sum())
        print(f"[solve] {tag:12s} counts={counts} quota={quota} "
              f"on_pad={[int(onp[j].sum()) for j in range(2)]} in_bin={n_bin} "
              f"latch={scene._placed_latch[0].tolist()} score={sc():.3f} "
              f"success={bool(scene.success()[0])}", flush=True)

    last_score = -1.0

    def phase_score(tag: str) -> float:
        nonlocal last_score
        s = sc()
        assert s >= last_score - 1e-6, f"score decreased at {tag}: {last_score} -> {s}"
        last_score = s
        print(f"SIM_GEN_SCORE {s:.3f}", flush=True)
        return s

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

    def settle_until(pred, max_steps: int, poll: int = 10) -> bool:
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def drop_body(body, pos_local, quat) -> None:
        """TRANSPORT teleport: place the body ABOVE its target with zero velocity and
        let gravity + contact finish the job. Never writes a seated/contained pose."""
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = origin + torch.tensor(pos_local, device=device)
        st[0, 3:7] = torch.tensor(quat, device=device)
        body.write_root_state_to_sim(st, torch.tensor([0], device=device))

    # ================= P0: reset + settle + layout readback ====================================
    env.reset(seed=args.seed)
    step(120)  # 1.0 s hands-off
    quota = scene.quota[0].tolist()
    surplus = int(scene.surplus[0])
    n_pres = int(scene.present[0].sum())
    pads_xy = [(scene.pads[j].data.root_pos_w[0] - origin)[:2].tolist() for j in range(2)]
    _p, _q, upz, _v = scene._cup_tensors()
    print(f"[solve] layout: quota={quota} surplus={surplus} present={n_pres} "
          f"pads={[[round(v, 3) for v in xy] for xy in pads_xy]} "
          f"cup_upz={[round(float(upz[0, j]), 3) for j in range(2)]}", flush=True)
    report("settled")
    assert n_pres == quota[0] + quota[1] + surplus
    assert bool((upz[0] < -0.9).all()), "cups must start mouth-down"
    assert int((scene.in_bin()[0] & scene.present[0]).sum()) == n_pres, \
        "present balls must start inside the bin"
    assert sc() <= 0.02, f"null-state score {sc():.3f} not ~0"
    assert not bool(scene.success()[0])
    phase_score("P0-settle")  # ~0.000

    # ================= P1: right and place both cups (physically forced first step) ============
    for j, tag in enumerate(("a", "b")):
        px, py = pads_xy[j]
        drop_body(scene.cups[j], (px, py, c.cup_h / 2 + DROP_H_CUP), (1.0, 0.0, 0.0, 0.0))
        ok = settle_until(lambda j=j: bool(scene.on_pad()[0, j].any()),
                          max_steps=int(3.0 / env.dt))
        report(f"P1-cup-{tag}")
        if not ok:
            print(f"[solve] P1 FAILED: cup_{tag} never settled on its pad", flush=True)
            verdict(False)
        s = phase_score(f"P1-place-{tag}")
    assert s >= 2 * c.w_place - 1e-4, f"both placed latches missing at P1: {s:.3f}"

    # ================= P2: counted delivery — exactly quota_j balls per pad ====================
    used: set[int] = set()

    def next_bin_ball() -> int:
        inb = scene.in_bin()[0] & scene.present[0]
        for i in range(c.n_balls):
            if i not in used and bool(inb[i]):
                return i
        raise AssertionError("ran out of bin balls before the quotas were met")

    for j, tag in enumerate(("a", "b")):
        for k in range(quota[j]):
            i = next_bin_ball()
            used.add(i)
            # the cup's LIVE position (it may have micro-settled off the pad centre)
            cup_p = scene.cups[j].data.root_pos_w[0] - origin
            dx, dy = SCATTER[k % len(SCATTER)]
            drop_body(scene.balls[i],
                      (float(cup_p[0]) + dx, float(cup_p[1]) + dy,
                       float(cup_p[2]) + c.cup_h / 2 + c.ball_r + DROP_H_BALL),
                      (1.0, 0.0, 0.0, 0.0))
            ok = settle_until(
                lambda j=j, i=i: bool(scene.in_cup()[0, j, i])
                and float(scene.balls[i].data.root_lin_vel_w[0].norm()) < c.settle_speed,
                max_steps=int(3.0 / env.dt))
            if not ok:
                report(f"P2-ball-{i}-miss")
                print(f"[solve] P2 FAILED: ball_{i} not contained in cup_{tag}", flush=True)
                verdict(False)
        report(f"P2-cup-{tag}-full")
        phase_score(f"P2-fill-{tag}")
    s = sc()
    assert s >= 2 * c.w_place + c.w_fill - 1e-4 or bool(scene.success()[0]), \
        f"full ticket credit missing at P2: {s:.3f}"

    # ================= P3: hands-off confirm — the scene's own success() =======================
    ok = settle_until(lambda: bool(scene.success()[0]), max_steps=int(6.0 / env.dt))
    report("P3-confirm")
    if not ok:
        print("[solve] P3 FAILED: success() not reached hands-off", flush=True)
        verdict(False)
    s = phase_score("P3-confirm")  # 1.000
    assert s >= 1.0 - 1e-6

    # ================= P4: persistence (>= 3.5 simulated seconds, hands off) ===================
    persist_steps = int(round(3.5 / env.dt))
    step(persist_steps)
    report("P4-final")
    phase_score("P4-final")
    still_ok = bool(scene.success()[0]) and sc() == 1.0
    print(f"[solve] persistence: {persist_steps} steps ({persist_steps * env.dt:.2f} s) "
          f"hands-off, success={still_ok}", flush=True)
    verdict(still_ok)


if __name__ == "__main__":
    main()

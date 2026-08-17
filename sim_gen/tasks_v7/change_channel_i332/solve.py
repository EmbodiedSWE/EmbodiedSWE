"""solve — TELEPORT-contract solution for ChannelFlipboardScene (change_channel_i332).

Scene-level env (robot="null"). This task has NO transport component — nothing is carried
anywhere — so the teleport budget goes UNUSED: no root pose of any task object is ever
written by this solution. The load-bearing interaction (flipping stacked hinged channel
cards over the apex of the flip-board, in the physically forced outer-card-first order)
is executed entirely through the live dynamics:

  the solver writes the scene's `card_drive` buffer — a torque about the card's hinge,
  clamped to TAU_MAX = 0.12 N*m. At the card's free edge (r_gap + card_l = 0.18 m from
  the axle) that is a 0.67 N fingertip push — about 2x the 0.059 N*m gravity peak, so a
  card can be lifted over the top but an inner card that drags the cards stacked outside
  it stalls instead of bulldozing through (the smoke ordering probe uses this same cap).
  Every substep the scene's own post_step sums the drive with viscous hinge friction;
  the joint stops, the card-on-card interlock and gravity do the rest. Nothing is
  pinned, no velocity is written, no rubric state is touched.

Per flip (always the CURRENT OUTERMOST card of its slope):
  DRIVE    gravity feed-forward (-m*g*(L/2)*sin(phi), i.e. cancel the pull toward the
           nearer stop) + a rate servo KW*(dest*OMEGA - w), total clamped to TAU_MAX —
           a deliberate ~2.5 rad/s slew up the slope, not a flick;
  RELEASE  the moment the card is RELEASE_DEG = 10 deg past the apex on the GOAL side the
           drive is zeroed — with ~2.5 rad/s of goal-ward rate and gravity now aiding,
           the card cannot come back; gravity + hinge viscosity drop it onto the far
           slope (onto its joint stop, or propped on the corner of an outer card already
           resting there — both are legitimate rests, both far beyond side_min_deg);
  SETTLE   hands off until the card reads categorically fallen (dest side, > 60 deg) and
           its hinge rate has died down.

Flip schedule (start channel s, target t, read back from the scene per episode; the
scene samples only REACHABLE pairs — the board is plan-level one-way, see scene.py):
  t > s : cards s..t-1 front -> back, ascending index — each is the front slope's
          outermost card when its turn comes.
  t == 0: cards 0..s-1 back -> front, ascending index — each is the back slope's
          outermost card when its turn comes. (0 < t < s is never sampled: it would
          require pulling a helper card out from UNDER the mover propped on it, and
          the rider is carried geometrically all the way to the apex — locked.)
The plan asserts the sampler contract (t > s or t == 0) before flipping anything.

Prints the scene readouts and `SIM_GEN_SCORE <score>` at each phase boundary (the printed
sequence never decreases — asserted). After success() first holds, keeps simulating
>= 3 more simulated seconds with ALL drive buffers zero; only if success() still holds
(it is live state — a knocked-off board would revert it) prints exactly
`SIM_GEN_SOLVE: SUCCESS`. Hard exit (os._exit) after the verdict, watchdog Timer as
backstop — Kit teardown hangs otherwise.

The intended single-Franka-arm strategy for the same plan (fingertip under the outermost
card's free lower edge, lift it up-and-over the apex along its hinge arc, release past
the top) lives in TASK.md as the embodiment argument.

Run (forge): python -u -m simgen_tasks.change_channel_i332.solve --headless
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
    from simgen_tasks.change_channel_i332 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()

# Fingertip-scale drive (the honesty argument sits in these numbers): TAU_MAX 0.12 N*m
# = 0.67 N at the free edge, ~2x the 0.059 N*m gravity peak of ONE card — enough to lift
# one card over the top, NOT enough to bulldoze an inner card plus the cards it drags.
TAU_MAX = 0.12  # N*m about the hinge
OMEGA = 2.5  # rad/s target slew up the slope
KW = 0.02  # N*m*s/rad rate-servo gain (KW*dt/I ~ 0.21 < 1, wrench-delay safe)
RELEASE_DEG = 10.0  # zero the drive this far past the top on the goal side
FALLEN_DEG = 60.0  # categorically fallen readback (pile props read ~78-115 deg)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.channel_flipboard")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    m_g_l2 = c.card_mass * 9.81 * (c.r_gap + c.card_l / 2)  # 0.059 N*m gravity peak

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        a = [f"{v:+6.1f}" for v in scene.angles_deg()[0].tolist()]
        print(f"[solve] {tag:12s} ang=({','.join(a)})deg "
              f"ch={int(scene.visible_channel()[0])} tgt={int(scene.target_ch[0])} "
              f"latch={scene.flip_latch[0].int().tolist()} score={sc():.3f} "
              f"success={bool(scene.success()[0])}", flush=True)

    last_score = -1.0

    def phase_score(tag: str) -> None:
        nonlocal last_score
        s = sc()
        assert s >= last_score - 1e-6, f"score decreased at {tag}: {last_score} -> {s}"
        last_score = s
        print(f"SIM_GEN_SCORE {s:.3f}", flush=True)

    def verdict(ok: bool) -> None:
        print("SIM_GEN_SOLVE: " + ("SUCCESS" if ok else "FAIL"), flush=True)
        code = 0 if ok else 1
        threading.Timer(10.0, lambda: os._exit(code)).start()
        try:
            env.close()
            app.close()
        except Exception:  # noqa: BLE001
            pass
        os._exit(code)

    def settle_until(pred, max_steps: int = 480, poll: int = 10) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def flip_card(i: int, dest: float) -> bool:
        """Flip card i onto slope `dest` (+1 front / -1 back) through the live plant:
        gravity-ff + rate-servo torque up the slope, release past the apex, hands-off
        drop. Returns True iff the card reads categorically fallen on the dest side."""
        start = float(scene.angles_deg()[0, i])
        print(f"[solve] flip card {i}: {start:+.1f} deg -> {'front' if dest > 0 else 'back'}",
              flush=True)
        released = None
        for k in range(1800):  # 15 s budget; ~250 deg at ~2.5 rad/s needs ~2 s
            ang = float(scene.angles_deg()[0, i])
            if ang * dest > RELEASE_DEG:
                released = ang
                break
            w = float(scene.hinge_rate()[0, i])
            tau = -m_g_l2 * math.sin(math.radians(ang)) + KW * (dest * OMEGA - w)
            scene.card_drive[0, i] = max(-TAU_MAX, min(TAU_MAX, tau))
            step(1)
            if k and k % 600 == 0:
                report(f"card{i}-drive")
        scene.card_drive[0, i] = 0.0  # RELEASE: gravity finishes the flip
        if released is None:
            print(f"[solve] card {i}: DRIVE TIMED OUT", flush=True)
            return False
        print(f"[solve] card {i}: released at {released:+.1f} deg — gravity drops it",
              flush=True)
        ok = settle_until(lambda: (float(scene.angles_deg()[0, i]) * dest > FALLEN_DEG
                                   and abs(float(scene.hinge_rate()[0, i])) < 0.8))
        step(60)  # let neighbours it may have brushed re-seat too
        report(f"card{i}-down")
        return ok

    # ================= reset + settle ==========================================================
    env.reset(seed=args.seed)
    step(90)
    report("reset")
    assert float(scene.card_drive.abs().max()) == 0.0
    ang0 = scene.angles_deg()[0]
    assert bool((ang0.abs() > 100.0).all()), "cards must start resting on their slopes"
    s0, t0 = int(scene.start_ch[0]), int(scene.target_ch[0])
    print(f"[solve] channel {s0 + 1} -> {t0 + 1} "
          f"({'front->back' if t0 > s0 else 'back->front (clear the back slope)'})",
          flush=True)
    phase_score("reset")  # ~0.000

    # ================= PHASES: outer-first flip schedule =======================================
    assert t0 > s0 or t0 == 0, f"unreachable target sampled: {s0} -> {t0}"
    if t0 > s0:
        plan = [(i, -1.0) for i in range(s0, t0)]
    else:  # t0 == 0: clear the whole back slope to the front
        plan = [(i, +1.0) for i in range(0, s0)]
    for n_done, (i, dest) in enumerate(plan, start=1):
        if not flip_card(i, dest):
            print(f"[solve] FLIP FAILED: card {i} not settled on its dest slope", flush=True)
            verdict(False)
        phase_score(f"flip{n_done}")

    # ================= goal: partition + settled ===============================================
    if not settle_until(lambda: bool(scene.success()[0]), max_steps=600):
        print("[solve] FAILED: success() not reached after the flip schedule", flush=True)
        report("no-success")
        verdict(False)
    report("goal")
    phase_score("goal")  # 1.000

    # ================= persistence (>= 3 simulated seconds, hands off) =========================
    assert float(scene.card_drive.abs().max()) == 0.0, "drive must be zero for persistence"
    assert float(scene.card_force.abs().max()) == 0.0
    persist_steps = int(round(3.5 / env.dt))  # 420 physics steps at 1/120 s
    step(persist_steps)
    report("final")
    phase_score("final")
    still_ok = bool(scene.success()[0]) and sc() == 1.0
    print(f"[solve] persistence: {persist_steps} steps ({persist_steps * env.dt:.2f} s) "
          f"hands-off, success={still_ok}", flush=True)
    verdict(still_ok)


if __name__ == "__main__":
    main()

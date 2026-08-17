"""Teleport solution for RatchetPortcullisScene (sim_gen task
`libero_kitchen_scene9_put_the_frying_pan_on_top_of_the_cabinet_i429`) — the
task's legitimacy certificate.

This solve uses NO teleports at all: every interaction is applied force through
contact dynamics. (Teleports are permitted for transport only; this task needs
none — one lift and one floor-slide do everything.)

1. PERCEPTION: the pan's sampled start pose and the bowl's roof pose are read
   back from the episode state — never hard-coded.
2. LIFT (applied force): a clamped PD force servo (what a hand hooking the
   gate's red bar and lifting does, force-limited) raises the gate to its top
   stop — the teeth click past the gravity pawl on the way up, real ratchet
   contact — then EASES the gate back down until a tooth rests on the pawl (a
   hand letting the ratchet take the weight). FORCES OFF: the gate STAYS OPEN
   hands-off on that catch. That pawl-held opening is the mechanism's whole
   point: it frees the one arm for the pan.
3. SLIDE (applied force): a clamped PD force drag (fingers pressing the
   skillet's rim, sliding it on the floor) walks the pan toward the doorway,
   through it — the transit credential latches on the REAL crossing under the
   raised gate — and onto the stove pad inside. Forces off; everything settles;
   success() turns True on the live state.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
rubric latches), then holds HANDS-OFF >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still holds.
The whole solve is then repeated on a SECOND seed (fresh episode, no score
prints — the rubric restarts at 0 on reset) to certify seed robustness.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene9_put_the_frying_pan_on_top_of_the_cabinet_i429.solve
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ratchet_portcullis")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    from isaaclab.utils.math import quat_apply_inverse

    print("[solve] " + "=" * 70, flush=True)
    print(scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    zero = torch.zeros(n, 1, 3, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        p = scene.pan_pos()[0]
        print(f"[solve] {tag:12s} | lift={float(scene.gate_lift()[0]):+.4f} "
              f"pan=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f}) "
              f"up={float(scene.pan_up()[0]):+.3f} "
              f"open={bool(scene._opened[0])} prop={bool(scene._prop[0])} "
              f"transit={bool(scene._transit[0])} on_pad={bool(scene.on_pad()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f} "
              f"pawlw={float(scene.pawl.data.root_ang_vel_w.norm(dim=-1)[0]):.3f} "
              f"panv={float(scene.pan.data.root_lin_vel_w.norm(dim=-1)[0]):.3f} "
              f"gatev={float(scene.gate.data.root_lin_vel_w.norm(dim=-1)[0]):.3f} "
              f"bowlv={float(scene.bowl.data.root_lin_vel_w.norm(dim=-1)[0]):.3f}",
              flush=True)

    def servo(body, tgt_xyz, *, kp, kd, clamp, steps, done=None, label="") -> None:
        """Applied-force PD drag on `body`'s CoM toward world target (None
        entries are unservoed). Force-limited to a human hand's scale."""
        tgt = torch.zeros(n, 3, device=device)
        mask = torch.zeros(3, device=device)
        for a, v in enumerate(tgt_xyz):
            if v is not None:
                tgt[:, a] = v
                mask[a] = 1.0
        for _ in range(steps):
            pos = body.data.root_pos_w - scene.env_origins
            vel = body.data.root_lin_vel_w
            f_w = (kp * (tgt - pos) - kd * vel) * mask
            f_w = f_w.clamp(min=-clamp, max=clamp)
            f_b = quat_apply_inverse(body.data.root_quat_w, f_w)
            body.set_external_force_and_torque(f_b.reshape(n, 1, 3), zero)
            env.step(no_action)
            if done is not None and done():
                break
        body.set_external_force_and_torque(zero, zero)
        env.step(no_action)  # flush the cleared wrench
        print(f"[solve] servo {label}: lift={float(scene.gate_lift()[0]):+.4f} "
              f"pan_x={float(scene.pan_pos()[0, 0]):+.3f} after <= {steps} steps", flush=True)

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
        p0 = scene.p0[0]
        b0 = scene.b0[0]
        pan = scene.pan_pos()[0]
        print(f"[solve] layout readback (seed {seed}): pan sampled=({float(p0[0]):+.3f},"
              f"{float(p0[1]):+.3f},yaw {float(p0[2]):+.3f}) actual=({float(pan[0]):+.3f},"
              f"{float(pan[1]):+.3f}) bowl=({float(b0[0]):+.3f},{float(b0[1]):+.3f})",
              flush=True)
        report("reset")
        assert bool(scene._finite()[0]), "NaN/inf after settle"
        assert abs(float(pan[0]) - float(p0[0])) < 0.01, "pan must rest at its sampled x"
        assert float(scene.gate_lift()[0]) < 0.005, "gate must start closed"
        assert float(pan[0]) > 0.15, "pan must start outside"
        assert not bool(scene.success()[0]), "fresh reset must not be success"
        s = print_score("P0 reset+settle (gate shut, pan outside)")
        assert s <= 0.05, f"baseline score should be ~0, got {s}"

        # ---------------- phase 1: lift the gate; set it down on the ratchet ------------
        servo(scene.gate, (None, None, c.gate_hi + 0.01), kp=300.0, kd=30.0, clamp=12.0,
              steps=600, done=lambda: float(scene.gate_lift()[0]) >= c.gate_hi - 0.004,
              label="gate-to-stop")
        assert float(scene.gate_lift()[0]) >= c.gate_hi - 0.008, "gate must reach the stop"
        # HOLD at the stop half a second (the last tooth has just flicked the
        # pawl up — let it fall back onto its seat), then ease the gate DOWN
        # onto the pawl (what a hand does with a ratchet): dropping it from the
        # stop outruns the pawl's gravity re-seat and skips catches.
        servo(scene.gate, (None, None, c.gate_hi), kp=200.0, kd=25.0, clamp=8.0,
              steps=60, label="gate-hold")
        servo(scene.gate, (None, None, c.gate_hi - 0.015), kp=120.0, kd=25.0, clamp=8.0,
              steps=300, label="gate-set-down")
        step(150)  # hands off: the pawl carries the gate, prop-counter runs
        lift = float(scene.gate_lift()[0])
        report("pawl-hold")
        assert lift >= 0.105, f"a pawl catch must hold the gate open, lift={lift:.4f}"
        assert bool(scene._opened[0]) and bool(scene._prop[0]), \
            "opened+propped latches must fire from the hands-off hold"
        s = print_score("P1 gate ratcheted open, held hands-off by the pawl")
        assert s >= c.w_open + c.w_prop - 1e-6, f"P1 score {s:.3f} below open+prop credit"

        # ---------------- phase 2: slide the pan through the doorway --------------------
        servo(scene.pan, (0.15, 0.0, None), kp=60.0, kd=12.0, clamp=8.0, steps=480,
              done=lambda: abs(float(scene.pan_pos()[0, 0]) - 0.15) < 0.02
              and abs(float(scene.pan_pos()[0, 1])) < 0.015, label="pan-align")
        servo(scene.pan, (-0.05, 0.0, None), kp=50.0, kd=14.0, clamp=6.0, steps=600,
              done=lambda: float(scene.pan_pos()[0, 0]) < -0.045, label="pan-through")
        assert bool(scene._transit[0]), "transit credential must latch on the real crossing"
        s = print_score("P2 pan through the doorway under the raised gate")
        assert s >= 0.65 - 1e-6, f"P2 score {s:.3f} below full latch credit"

        # ---------------- phase 3: onto the pad; hands off; success ---------------------
        servo(scene.pan, (c.pad_x, c.pad_y, None), kp=40.0, kd=16.0, clamp=5.0, steps=720,
              done=lambda: abs(float(scene.pan_pos()[0, 0]) - c.pad_x) < 0.015
              and abs(float(scene.pan_pos()[0, 1]) - c.pad_y) < 0.015, label="pan-to-pad")
        won_at = None
        for i in range(360):
            env.step(no_action)
            if bool(scene.success()[0]):
                won_at = i
                break
        report("hands-off")
        assert won_at is not None and bool(scene.success()[0]), \
            f"success must hold hands-off: pan={scene.pan_pos()[0].tolist()}"
        assert bool(scene.on_pad()[0]), "pan must rest on the pad"
        s = print_score(f"P3 pan on the stove pad; settled ({won_at} hands-off steps)")
        assert s >= 1.0 - 1e-6, "success must score 1.0"

        # ---------------- persistence (>= 3 simulated seconds, hands-off) ---------------
        hold = True
        for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
            step(40)
            hold = hold and bool(scene.success()[0])
        report("persist")
        # the bystander bowl must still sit on the roof, the gate still open
        bw = (scene.bowl.data.root_pos_w - scene.env_origins)[0]
        assert float(bw[2]) > c.roof_z1, "the bowl must still stand on the roof"
        assert abs(float(bw[0]) - float(b0[0])) < 0.05 and \
            abs(float(bw[1]) - float(b0[1])) < 0.05, "the bowl must not have moved"
        assert float(scene.gate_lift()[0]) >= 0.105, "the pawl must still hold the gate"
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
    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
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

"""Teleport solution for BoltLatchStowScene (sim_gen task
`libero_kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_i420`)
— the task's legitimacy certificate.

Teleports are used for TRANSPORT ONLY: one write moves the bowl from the bench
to a hover pose above the open drawer, from which it free-falls in. Every
load-bearing interaction is an applied force through contact dynamics:

1. PERCEPTION: bowl/bottle sampled poses and the drawer's latched rest opening
   are read back from the episode state — never hard-coded.
2. RELEASE (applied force): a PD force servo lifts the bolt by its T-handle
   (what a Franka hook-grasp on the crossbar does, force-limited to a few N).
   The instant the bolt clears the panel, the cabinet's own SPRING throws the
   drawer to its out-stop — nobody ever pulls the drawer.
3. LOAD (transport teleport + gravity): the bowl is teleported to a hover pose
   over the exposed drawer box (clear of the dangling bolt) and DROPPED; the
   landing and settling are honest contact dynamics.
4. CLOSE (applied force): a velocity-limited force servo presses the blue
   panel inward against the live spring (a palm-push). Near the end of travel
   the panel cams the bolt up over its 45-degree tip wedge — a real collision
   chain — and the bolt gravity-drops back in front of the panel. Forces off:
   the spring presses the panel onto the bolt's flat rear face and the latch
   alone holds the drawer shut.
5. HANDS OFF: success() turns True on the live state and must keep holding for
   >= 3 simulated seconds — an unlatched drawer would spring back open here.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
rubric latches), then prints `SIM_GEN_SOLVE: SUCCESS` only if success held.
The whole solve is then repeated on a SECOND seed (fresh episode, no score
prints — the rubric restarts at 0 on reset) to certify seed robustness.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_i420.solve
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
    env = ENVS.get("simgen.boltlatch_stow")().build(num_envs=args.num_envs, device=device)
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
        print(f"[solve] {tag:12s} | open={float(scene.opening()[0]):+.4f} "
              f"lift={float(scene.bolt_lift()[0]):+.4f} "
              f"in_cav={bool(scene.bowl_in_cavity()[0])} shut={bool(scene.shut()[0])} "
              f"rel={bool(scene._released[0])} opn={bool(scene._opened[0])} "
              f"lod={bool(scene._loaded[0])} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def body_force(body, f_w: torch.Tensor) -> None:
        """Apply a world-frame force at the body's root (frame-safe encode)."""
        f_b = quat_apply_inverse(body.data.root_quat_w, f_w)
        body.set_external_force_and_torque(f_b.reshape(n, 1, 3), zero)

    def hold_bolt_step(tgt: float, *, kp: float = 30.0, kd: float = 4.0,
                       ff: float = 4.9, clamp: float = 9.0) -> None:
        """One step of the T-handle lift servo (hook-grasp, force-limited;
        ff ~= the 0.5 kg bolt's weight)."""
        z = scene.bolt_lift()
        vz = scene.bolt.data.root_lin_vel_w[:, 2]
        fz = (ff + kp * (tgt - z) - kd * vz).clamp(min=-3.0, max=clamp)
        f_w = torch.zeros(n, 3, device=device)
        f_w[:, 2] = fz
        body_force(scene.bolt, f_w)
        env.step(no_action)

    def close_drawer(*, v_des: float, kv: float, ff: float, clamp: float,
                     steps: int, done, label: str) -> None:
        """Velocity-limited palm-push on the drawer panel (world -x), against
        the live spring. Force-limited; the cam lift near the end is a real
        collision chain (panel edge -> 45-degree wedge -> bolt)."""
        for _ in range(steps):
            vx = scene.drawer.data.root_lin_vel_w[:, 0]
            fx = (-ff + kv * (-v_des - vx)).clamp(min=-clamp, max=2.0)
            f_w = torch.zeros(n, 3, device=device)
            f_w[:, 0] = fx
            body_force(scene.drawer, f_w)
            env.step(no_action)
            if done():
                break
        scene.drawer.set_external_force_and_torque(zero, zero)
        print(f"[solve] push {label}: open={float(scene.opening()[0]):+.4f} "
              f"lift={float(scene.bolt_lift()[0]):+.4f} after <= {steps} steps",
              flush=True)

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
        bw = (scene.bowl.data.root_pos_w - scene.env_origins)[0]
        bt = (scene.bottle.data.root_pos_w - scene.env_origins)[0]
        d_rest = float(scene.opening()[0])
        print(f"[solve] layout readback (seed {seed}): "
              f"bowl=({float(bw[0]):+.3f},{float(bw[1]):+.3f}) "
              f"bottle=({float(bt[0]):+.3f},{float(bt[1]):+.3f}) "
              f"drawer latched rest={d_rest:+.4f}", flush=True)
        report("reset")
        assert bool(scene._finite()[0]), "NaN/inf after settle"
        assert abs(float(bw[0]) - float(scene.bowl_xy0[0, 0])) < 0.01 and \
            abs(float(bw[1]) - float(scene.bowl_xy0[0, 1])) < 0.01, \
            "bowl must rest at its sampled bench pose"
        assert float(bw[1]) * float(bt[1]) < 0.0, "bowl and bottle on opposite sides"
        assert 0.003 <= d_rest <= c.shut_tol, \
            "spring must press the shut drawer onto the seated bolt"
        assert float(scene.bolt_lift()[0]) < 0.004, "bolt must start seated"
        assert not bool(scene.success()[0]), "fresh reset must not be success"
        s = print_score("P0 reset+settle (drawer latched shut, bowl on bench)")
        assert s <= 0.05, f"baseline score should be ~0, got {s}"

        # ---------------- phase 1: lift the bolt -> the spring throws the drawer --------
        lift_tgt = 0.035
        for _ in range(480):
            hold_bolt_step(lift_tgt)
            if float(scene.opening()[0]) >= c.opened_min + 0.01:
                break
        assert bool(scene._released[0]), "release latch must fire during the lift"
        assert bool(scene._opened[0]), "spring must throw the drawer open"
        # hands off the bolt: it gravity-drops and dangles above the open drawer
        scene.bolt.set_external_force_and_torque(zero, zero)
        for _ in range(240):
            env.step(no_action)
            if float(scene.opening()[0]) >= c.stroke - 0.01 and \
                    float(scene.drawer.data.root_lin_vel_w[0, 0].abs()) < 0.05:
                break
        report("released")
        assert float(scene.opening()[0]) >= c.stroke - 0.015, \
            "drawer must rest at its out-stop under the spring"
        assert float(scene.bolt_lift()[0]) < 0.006, "bolt must dangle back at its stop"
        s = print_score("P1 bolt lifted; spring threw the drawer to its out-stop")
        assert s >= c.w_release + c.w_open - 1e-6, f"P1 score {s:.3f} below release+open"

        # ---------------- phase 2: transport teleport + gravity drop --------------------
        hover = torch.zeros(n, 13, device=device)
        hover[:, 0] = scene.env_origins[:, 0] + c.drop_x()
        hover[:, 1] = scene.env_origins[:, 1]
        hover[:, 2] = scene.env_origins[:, 2] + 0.26
        hover[:, 3] = 1.0
        scene.bowl.write_root_state_to_sim(hover)
        step(150)
        report("loaded")
        assert bool(scene.bowl_in_cavity()[0]), \
            f"bowl must land inside the drawer box: z={float(scene.bowl.data.root_pos_w[0, 2]):.3f}"
        assert bool(scene._loaded[0]), "loaded latch must fire while the drawer is open"
        assert float(scene.opening()[0]) >= c.stroke - 0.02, \
            "spring must keep the drawer at its stop through the drop"
        s = print_score("P2 bowl dropped into the open drawer")
        assert s >= 0.60 - 1e-6, f"P2 score {s:.3f} below the full latch credit"

        # ---------------- phase 3: press shut -> cam over the bolt -> latched -----------
        close_drawer(v_des=0.18, kv=60.0, ff=7.0, clamp=14.0, steps=720,
                     done=lambda: float(scene.opening()[0]) <= 0.004, label="to-shut")
        assert float(scene.opening()[0]) <= 0.006, "drawer must reach the inner stop"
        # brief hold so the bolt finishes its gravity drop in front of the panel
        for _ in range(50):
            f_w = torch.zeros(n, 3, device=device)
            f_w[:, 0] = -8.0
            body_force(scene.drawer, f_w)
            env.step(no_action)
        scene.drawer.set_external_force_and_torque(zero, zero)
        step(90)  # spring presses the panel onto the bolt's flat rear face
        report("latched")
        assert float(scene.bolt_lift()[0]) < 0.005, "bolt must have re-seated"
        assert bool(scene.shut()[0]), \
            f"latch must hold the drawer shut: open={float(scene.opening()[0]):+.4f}"
        assert bool(scene.bowl_in_cavity()[0]), "bowl must still be inside the drawer"

        # ---------------- phase 4: hands off — success on the live state ----------------
        won_at = None
        for i in range(360):
            env.step(no_action)
            if bool(scene.success()[0]):
                won_at = i
                break
        report("hands-off")
        assert won_at is not None and bool(scene.success()[0]), \
            f"success must hold hands-off: open={float(scene.opening()[0]):+.4f}"
        s = print_score(f"P4 released; latched shut with the bowl sealed in "
                        f"({won_at} hands-off steps)")
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

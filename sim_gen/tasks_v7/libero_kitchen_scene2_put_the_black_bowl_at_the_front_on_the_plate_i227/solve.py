"""Teleport solution for ShutterVaultScene (sim_gen task
libero_kitchen_scene2_put_the_black_bowl_at_the_front_on_the_plate_i227) — the task's
legitimacy certificate.

Teleports handle TRANSPORT of the free bowl ONLY; every load-bearing interaction runs
through contact dynamics:

  * BOTH shutter slides are driven by a horizontal external force on the tile through
    `set_external_force_and_torque` — a capped-speed push standing in for a finger dragging
    the knob. The tiles ride deck friction under the hold-down rails; nothing about the
    tiles is ever written after reset. (Known forge quirk: external wrenches are rotated by
    the body's rotation-since-reset; the tiles keep their reset yaw — the channel walls
    admit only millimetres of wiggle — so the world-frame push is unaffected.)
  * the FRONT bowl is teleported through free air only: incremental raise off the counter,
    one hop above all geometry (carry z 0.42 > knob tops 0.327), incremental descent down
    the open well shaft, then RELEASED ~28 mm above the plate — it seats on the plate under
    gravity and contact, it is NEVER spawned seated;
  * the ORDER is the physical one: BLUE out of the middle cell first (RED is walled into
    the well cell until B is vacant), then RED off the well, then the drop.

Phases (score latches must be non-decreasing along the run; SIM_GEN_SCORE is printed at
every phase boundary):
  0. reset(seed), settle, readback (vault pose, front-bowl identity)        -> 0.000
  1. UNLOCK: force-servo the BLUE tile B -> C, brake, release              -> 0.250
  2. OPEN:   force-servo the RED tile A -> B, brake, release               -> 0.500
  3. SERVE:  raise/hop/lower the FRONT bowl down the shaft, release,
     gravity seating (retry with small offsets if it mis-seats)            -> 1.000
  4. persistence: >= 3 simulated seconds hands-off, success() must hold    -> SIM_GEN_SOLVE

Run (forge): python -u -m simgen_tasks.<task>.solve --headless [--seed N]
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
from isaaclab.utils.math import quat_apply_inverse

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Kit teardown hangs leave GPU zombies: global watchdog long before the forge timeout.
threading.Timer(1350.0, lambda: os._exit(4)).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.shutter_vault")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    origin = env.iscene.env_origins
    P = c.pitch

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    last_score = [0.0]

    def phase_score(tag: str) -> None:
        s = float(scene.score()[0])
        loc0 = scene._vault_local(scene.tiles[0].data.root_pos_w)[0]
        loc1 = scene._vault_local(scene.tiles[1].data.root_pos_w)[0]
        print(f"[solve] {tag}: unlocked={bool(scene.ever_unlocked[0])} "
              f"opened={bool(scene.ever_opened[0])} served={bool(scene.ever_served[0])} "
              f"red=({float(loc0[0]):+.3f},{float(loc0[1]):+.3f}) "
              f"blue=({float(loc1[0]):+.3f},{float(loc1[1]):+.3f}) "
              f"on_plate={bool(scene.front_on_plate()[0])} "
              f"plate_in_well={bool(scene.plate_in_well()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])}",
              flush=True)
        if s + 1e-6 < last_score[0]:
            print(f"[solve] FATAL: score decreased {last_score[0]:.3f} -> {s:.3f}", flush=True)
            print("SIM_GEN_SOLVE: FAIL", flush=True)
            os._exit(1)
        last_score[0] = s
        print(f"SIM_GEN_SCORE {s:.3f}", flush=True)

    # ---------------- tile drive: capped-speed horizontal push (finger-on-knob surrogate) -----
    # Frame quirk (pod-dependent): the default set_external_force_and_torque call may apply
    # the wrench in the BODY frame, so a yaw-psi tile pushed with a raw world vector moves
    # along direction 2*psi (seed 0's psi~0 masked this; seed 7's psi=-20deg wedge-jammed
    # the RED tile against the channel wall). We therefore pre-encode the intended WORLD
    # force with quat_apply_inverse per step ("body" mode), and keep a runtime progress
    # probe that toggles to raw-world mode if the tile moves AWAY from its target.
    zero3 = torch.zeros(n, 1, 3, device=device)
    tile_mode = ["body", "body"]  # per-tile encoding mode, probed at runtime

    def set_tile_force(t: int, fx: float, fy: float) -> None:
        f = torch.zeros(n, 1, 3, device=device)
        f[:, 0, 0] = fx
        f[:, 0, 1] = fy
        if tile_mode[t] == "body":
            q = scene.tiles[t].data.root_quat_w
            f = quat_apply_inverse(q, f.squeeze(1)).unsqueeze(1)
        scene.tiles[t].set_external_force_and_torque(f, zero3)

    def slide_tile(t: int, target_local: tuple, tag: str, v_max: float = 0.08,
                   f_push: float = 2.5, max_steps: int = 4000) -> None:
        """Push tile `t` to the vault-local cell centre `target_local` with a bang-bang
        capped-speed force (friction budget mu*m*g ~ 1.7 N; 2.5 N moves it gently), then
        brake and release. Escalates the push if friction pins the tile (escalate the
        DRIVE, never weaken a check); toggles the force-frame encoding if the probe shows
        no progress at full force (encoding suspect, not obstruction). Coast after cut is
        negligible: at 0.08 m/s deck friction stops the tile within ~1 mm."""
        tgt = torch.tensor(target_local, device=device)
        psi = float(scene.box_yaw[0])
        cy, sy = math.cos(psi), math.sin(psi)
        stall, best_err, force = 0, 1e9, f_push
        fruitless = 0  # escalations that bought no progress -> encoding suspect
        for _ in range(max_steps):
            loc = scene._vault_local(scene.tiles[t].data.root_pos_w)[0]
            err = tgt - loc[:2]
            e = float(err.norm())
            v_w = scene.tiles[t].data.root_lin_vel_w[0]
            if e < 0.012 and float(v_w[:2].norm()) < 0.03:
                break
            # stall escalation: friction can pin the tile against the wall micro-wedge
            if e < best_err - 0.002:
                best_err, stall, fruitless = e, 0, 0
            else:
                stall += 1
                if stall > 240:
                    stall = 0
                    if force >= 5.9:
                        fruitless += 1
                    force = min(force + 1.0, 6.0)
                    print(f"[solve] {tag}: escalating push to {force:.1f} N (err {e:.3f})",
                          flush=True)
                    if fruitless >= 2:
                        tile_mode[t] = "world" if tile_mode[t] == "body" else "body"
                        force, fruitless = f_push, 0
                        set_tile_force(t, 0.0, 0.0)
                        step(60)  # let the wedge relax before re-driving
                        print(f"[solve] {tag}: no progress at full force -> toggling "
                              f"force encoding to '{tile_mode[t]}'", flush=True)
            dl = err / max(e, 1e-9)
            dw = (cy * dl[0] - sy * dl[1], sy * dl[0] + cy * dl[1])  # local -> world
            v_along = float(v_w[0] * dw[0] + v_w[1] * dw[1])
            if v_along < v_max:
                set_tile_force(t, force * dw[0], force * dw[1])
            else:
                set_tile_force(t, 0.0, 0.0)
            step(1)
        # braking tail: oppose residual velocity, then cut (same encoding path)
        for _ in range(120):
            v_w = scene.tiles[t].data.root_lin_vel_w[0]
            if float(v_w[:2].norm()) < 0.01:
                break
            set_tile_force(t, -30.0 * float(v_w[0]), -30.0 * float(v_w[1]))
            step(1)
        set_tile_force(t, 0.0, 0.0)
        step(30)
        loc = scene._vault_local(scene.tiles[t].data.root_pos_w)[0]
        print(f"[solve] {tag}: tile_{t} local=({float(loc[0]):+.3f},{float(loc[1]):+.3f}) "
              f"target=({target_local[0]:.3f},{target_local[1]:.3f})", flush=True)

    # ---------------- bowl transport (free air only) ------------------------------------------
    def bowl_state(b: int) -> torch.Tensor:
        return scene.bowls[b].data.root_state_w[all_ids].clone()

    def write_bowl(b: int, st: torch.Tensor) -> None:
        st = st.clone()
        st[:, 7:13] = 0.0
        scene.bowls[b].write_root_state_to_sim(st, all_ids)

    def raise_bowl(b: int, z_to: float) -> None:
        """Incremental vertical lift: small pose writes with real physics steps between."""
        st = bowl_state(b)
        z = float(st[0, 2])
        while z < z_to:
            z = min(z + 0.006, z_to)
            st = bowl_state(b)
            st[:, 2] = z
            write_bowl(b, st)
            step(2)

    def hop_bowl(b: int, x: float, y: float, z: float, yaw: float) -> None:
        """One free-space transport write ABOVE every obstacle (env-local x, y)."""
        st = bowl_state(b)
        st[:, 0] = x
        st[:, 1] = y
        st[:, 2] = z
        st[:, 3:7] = torch.tensor(
            (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)), device=device)
        st[:, 0:2] += origin[:, 0:2]
        write_bowl(b, st)
        step(2)

    def lower_bowl(b: int, z_to: float) -> None:
        """Incremental descent down the open well shaft (aperture 120 mm vs bowl ~91 mm
        over the corners: 14+ mm radial clearance)."""
        st = bowl_state(b)
        z = float(st[0, 2])
        while z > z_to:
            z = max(z - 0.006, z_to)
            st = bowl_state(b)
            st[:, 2] = z
            write_bowl(b, st)
            step(2)

    def settle(max_steps: int = 600, poll: int = 10) -> None:
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if bool(scene.settled()[0]):
                return

    carry_z = 0.42  # above knob tops (0.327) and rail tops (0.292)

    def serve_bowl(b: int) -> None:
        """Raise the bowl, carry it over the open well, lower it down the shaft, release
        ~28 mm above the plate; gravity + contact seat it. Retry with small offsets."""
        offs = ((0.0, 0.0), (0.005, -0.004), (-0.005, 0.004), (0.004, 0.005))
        for attempt, (ox, oy) in enumerate(offs):
            raise_bowl(b, carry_z)
            pp = scene.plate.data.root_pos_w[0] - origin[0]
            plate_top = float(scene.plate.data.root_pos_w[0, 2]) + c.plate_h / 2
            release_z = plate_top + c.bowl_floor_h / 2 + 0.028
            hop_bowl(b, float(pp[0]) + ox, float(pp[1]) + oy, carry_z, 0.35 * attempt)
            lower_bowl(b, release_z)
            settle(max_steps=500)
            if bool(scene.front_on_plate()[0]):
                return
            print(f"[solve] serve attempt {attempt} missed; retrying", flush=True)
        print("[solve] FATAL: front bowl never seated on the plate", flush=True)
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)

    # ---------------- phase 0: reset + readback -----------------------------------------------
    env.reset(seed=args.seed)
    step(60)  # settle the authored layout
    front = int(scene.front_idx[0])
    slots = scene.slot_of[0].tolist()
    print(f"[solve] seed={args.seed} box=({float(scene.box_pos_xy[0, 0]):+.3f},"
          f"{float(scene.box_pos_xy[0, 1]):+.3f}) "
          f"yaw={math.degrees(float(scene.box_yaw[0])):+.1f}deg "
          f"front_bowl=bowl_{front} slot_of={slots}", flush=True)
    phase_score("phase0 reset")

    # ---------------- phase 1: unlock — BLUE tile B -> C --------------------------------------
    slide_tile(1, (P, P), "phase1 servo")
    settle(max_steps=300)
    if not bool(scene.ever_unlocked[0]):
        print("[solve] FATAL: BLUE tile never reached C (unlock latch cold)", flush=True)
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)
    phase_score("phase1 unlock")

    # ---------------- phase 2: open — RED tile A -> B -----------------------------------------
    slide_tile(0, (P, 0.0), "phase2 servo")
    settle(max_steps=300)
    if not bool(scene.ever_opened[0]):
        print("[solve] FATAL: RED tile never reached B (open latch cold)", flush=True)
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(1)
    phase_score("phase2 open")

    # ---------------- phase 3: serve — drop the FRONT bowl down the shaft onto the plate ------
    serve_bowl(front)
    settle(max_steps=400)
    phase_score("phase3 serve")

    if not bool(scene.success()[0]):
        settle(max_steps=600)
    ok = bool(scene.success()[0])
    print(f"[solve] goal state reached: success={ok} score={float(scene.score()[0]):.3f}",
          flush=True)
    phase_score("goal")

    # ---------------- phase 4: hands-off persistence (>= 3 simulated seconds) -----------------
    persist_ok = ok
    if ok:
        held = 0
        while held < 400:  # 400 steps at 120 Hz = 3.33 s, no further intervention
            step(40)
            held += 40
            if not bool(scene.success()[0]):
                persist_ok = False
                break
        phase_score("persistence")

    if persist_ok and bool(scene.success()[0]) and float(scene.score()[0]) == 1.0:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
        code = 0
    else:
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        code = 1

    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()

"""Teleport solution for DominoRelayScene (sim_gen task
`libero_kitchen_scene5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_i45`) —
the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY, in one write per tile, ending in FREE SPACE:
  P1 — the scattered relay tiles are carried from their lying scatter spawns to
  STANDING poses on the straight pad->window chain line, evenly spaced (~42-48 mm
  gaps, sized for cascade energy) from the crimson trigger tile to a spot ~50 mm
  outside the wall, thickness axis along the chain; tiles the chain does not need
  are parked lying flat far outside the corridor. Every tile ends standing (or
  lying) freely on open floor, in contact with nothing.
  Everything after that goes through CONTACT DYNAMICS:
  P2 — one finger-nudge on the TRIGGER tile only: a small horizontal world force
  (0.16 N, ~2.4x the tip threshold, well under the friction slide limit) applied at
  its COM along the chain direction while it is still near-upright (so the frozen
  body-frame wrench stays world-true), cut as soon as it passes ~15 deg of lean.
  Gravity and eight successive tile impacts do ALL the delivery: the cascade runs
  down the line, the last tile pitches through the letterbox window — self-centering,
  because the chain line points at the window centre — and bats the orange ball off
  its shelf seat into the sunken pit. The ball is never touched by any force or
  teleport; no tile after the trigger is ever touched again.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

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

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.domino_relay")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def gallery_frame() -> tuple[torch.Tensor, torch.Tensor]:
        """(env-origin-relative gallery pos (n,3), yaw (n,)) — pure-z-rotation fixture."""
        gp = scene.gallery.data.root_pos_w - scene.env_origins
        q = scene.gallery.data.root_quat_w
        return gp, 2.0 * torch.atan2(q[:, 3], q[:, 0])

    def to_world_xy(local_xy: torch.Tensor) -> torch.Tensor:
        """(n,2) gallery-local xy -> (n,2) env-origin-relative world xy."""
        gp, gyaw = gallery_frame()
        cy, sy = torch.cos(gyaw), torch.sin(gyaw)
        return torch.stack([gp[:, 0] + local_xy[:, 0] * cy - local_xy[:, 1] * sy,
                            gp[:, 1] + local_xy[:, 0] * sy + local_xy[:, 1] * cy], dim=-1)

    def report(tag: str) -> None:
        bl = scene._fix_local(scene.ball.data.root_pos_w)[0]
        both = bool(torch.isfinite(scene.t_ball[0])) and bool(torch.isfinite(scene.t_trig[0]))
        dt_tb = float(scene.t_ball[0] - scene.t_trig[0]) * env.dt if both else float("nan")
        print(f"[solve] {tag:12s} | ball_local=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) trig_upz={float(scene._up_z(scene.trigger)[0]):+.3f} "
              f"standing={float(scene.standing_in_corridor()[0]):.0f} "
              f"fallen={float(scene.fallen_in_corridor()[0]):.0f} "
              f"built={float(scene.built[0]):.0f} causal={float(scene.causal[0]):.0f} "
              f"dt_trig->ball={dt_tb:.2f}s pit={bool(scene.ball_in_pit()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_wrench() -> None:
        scene.trigger.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                    env_ids=all_ids)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    gp, gyaw = gallery_frame()
    pad = scene.pad_local[0]
    span0 = float(scene.span[0])
    bear = math.degrees(math.atan2(float(pad[0]), -float(pad[1])))
    print(f"[solve] layout readback (seed {args.seed}): gallery=({float(gp[0, 0]):+.3f},"
          f"{float(gp[0, 1]):+.3f}) yaw={math.degrees(float(gyaw[0])):+.1f}deg "
          f"pad_local=({float(pad[0]):+.3f},{float(pad[1]):+.3f}) span={span0:.3f} "
          f"bearing={bear:+.1f}deg gap={(span0 - 0.05) / c.n_tiles:.3f}", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: tile TRANSPORT (teleport to standing chain) -----------------
    # Straight chain: tile i at pad + u * (i * (span-50mm)/n_chain), standing, thickness
    # axis along the chain. Cascade energy grows with gap size, so use only as many
    # tiles as a ~48 mm target gap needs (each fall then gains real momentum) — the
    # leftovers are PARKED lying flat far outside the corridor. The chain line passes
    # through the window CENTRE, so the last tile — 50 mm from the wall — falls straight
    # at the letterbox and self-centres in it. Each tile ends in free space: nearest
    # face-to-face gap is >= 27 mm, nothing is in contact.
    chain_yaw_l = torch.atan2(scene.u_local[:, 1], scene.u_local[:, 0])
    span0 = float(scene.span.max())
    n_chain = min(c.n_tiles, max(c.k_relay + 1, math.ceil((span0 - 0.05) / 0.048)))
    print(f"[solve] chain plan: {n_chain} tiles, gap {(span0 - 0.05) / n_chain:.3f} m",
          flush=True)
    for i in range(n_chain):
        s_i = (float(i + 1)) * (scene.span - 0.05) / float(n_chain)
        local_xy = scene.pad_local + scene.u_local * s_i.unsqueeze(-1)
        wxy = to_world_xy(local_xy)
        wyaw = gyaw + chain_yaw_l
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = wxy
        st[:, 2] = c.tile_z0 + 0.002
        st[:, 3], st[:, 6] = torch.cos(wyaw / 2), torch.sin(wyaw / 2)
        st[:, 0:3] += scene.env_origins
        scene.tiles[i].write_root_state_to_sim(st, all_ids)
    c45 = math.cos(math.pi / 4)
    for j, i in enumerate(range(n_chain, c.n_tiles)):
        # park the unused tiles lying flat on open floor, far outside the corridor
        wxy = to_world_xy(torch.tensor([0.55, -0.55 + 0.14 * j], device=device)
                          .expand(n, 2))
        wyaw = gyaw  # lying flat: q = qz(yaw) * qy(90 deg)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = wxy
        st[:, 2] = c.tile_t / 2 + 0.002
        st[:, 3] = torch.cos(wyaw / 2) * c45
        st[:, 4] = -torch.sin(wyaw / 2) * c45
        st[:, 5] = torch.cos(wyaw / 2) * c45
        st[:, 6] = torch.sin(wyaw / 2) * c45
        st[:, 0:3] += scene.env_origins
        scene.tiles[i].write_root_state_to_sim(st, all_ids)
    step(60)  # free-stand settle; `built` latches the standing-in-corridor count
    report("chain built")
    s1 = print_score("P1 relay tiles transported to standing chain")
    assert s1 >= s0 - 1e-6, "score decreased across tile transport"
    assert float(scene.standing_in_corridor()[0]) >= float(c.k_relay), \
        "chain did not stand in the corridor"

    # ---------------- phase 2: trigger nudge (contact) + hands-off cascade ------------------
    # One horizontal push at the trigger's COM, along the chain, world-true because it is
    # applied only while the tile is near-upright (frozen body-frame wrench caveat).
    # 0.16 N: tip threshold is mg*t/H ~ 0.065 N, slide threshold is mu*m*g ~ 0.27 N.
    gp, gyaw = gallery_frame()
    chain_yaw_w = gyaw + chain_yaw_l
    fwd = torch.stack([torch.cos(chain_yaw_w), torch.sin(chain_yaw_w),
                       torch.zeros_like(chain_yaw_w)], dim=-1)
    f_push = (0.16 * fwd).view(n, 1, 3).contiguous()
    pushed = 0
    for _ in range(90):
        if float(scene._up_z(scene.trigger)[0]) < math.cos(math.radians(15.0)):
            break  # committed past the 7.6 deg balance point — gravity finishes it
        scene.trigger.set_external_force_and_torque(f_push, zero_wrench,
                                                    env_ids=all_ids, is_global=True)
        env.step(no_action)
        pushed += 1
    clear_wrench()
    print(f"[solve] trigger nudged for {pushed} substeps "
          f"({pushed * env.dt:.2f} s), hands off", flush=True)

    # Hands-off: let the cascade run; poll until causal delivery, then poll for rest —
    # the batted ball may carom around the sealed chamber for a few seconds first.
    for _ in range(24):  # up to 6 s of cascade time
        step(30)
        if bool((scene.causal[0] > 0.5)) and bool(scene.ball_in_pit()[0]):
            break
    for _ in range(20):  # up to 10 s: tiles flat, ball at rest on the pit floor
        step(60)
        if bool(scene.success()[0]):
            break
    report("post-cascade")
    s2 = print_score("P2 trigger nudge + cascade + settle")
    assert s2 >= s1 - 1e-6, "score decreased across the cascade"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the cascade settled)", flush=True)
        os._exit(1)

    # ---------------- phase 3: persistence (>= 3.3 simulated seconds, no intervention) -----
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s3 = print_score("P3 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s3 >= s2 - 1e-6
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
    except BaseException as exc:  # noqa: BLE001 - die loudly, never idle to the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL (exception: {exc!r})", flush=True)
        os._exit(2)

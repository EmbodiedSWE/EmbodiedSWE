"""Teleport solution for FlatPackCrateScene (sim_gen task
`libero_pick_salad_dressing_i151`) — the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. PANEL TRANSPORT (teleport) + INSERTION (gravity + contact): each blue panel is
   carried by a single root-state write from its flat ground pose to free air
   directly above its slot channel — upright, thickness axis square to the channel,
   zero velocity, bottom edge ~15 mm above the funnel mouth. That is exactly the
   carry+reorient a Franka performs with an edge pinch on the panel. The write
   satisfies no rubric clause (the panel is airborne, 80+ mm above its seat); the
   INSERTION is pure physics — the panel free-falls into the flared funnel, is
   guided by the rails, slides down the channel and seats on the floor pad by
   contact. If a drop wedges on the flare lip, the panel is lifted back to free
   air (transport) and dropped again with a tiny lateral offset — the seat credit
   is only ever produced by gravity and contact, never written.
2. BOTTLE TRANSPORT (teleport) + SEATING (gravity + contact): the amber bottle is
   carried over the walls into the open interior — upright, 10 mm above the pad,
   zero velocity, on the vertical free-air corridor through the open top — and
   FALLS the last 10 mm, seating upright on the pad by contact.
3. LID TRANSPORT (teleport) + SEATING (gravity + contact): the lid is carried to
   free air centred over the crate, level, lip down, its plate underside 10 mm
   above the wall tops (the lip hangs inside the wall opening but touches
   nothing). It FALLS onto the wall/panel tops; the lip registers inside the
   walls by contact. This is done LAST — with the lid seated, the plate covers
   the open top and both slot mouths, so panels/bottle physically cannot enter
   afterwards (the scene latches a breach fail if they do).
4. ORDER / IDENTITY: panels and bottle strictly before the lid; the RED decoy
   bottle is read back by station assignment and never touched.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_pick_salad_dressing_i151.solve --headless [--seed N]
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
import traceback

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as _task_scene  # noqa: F401 - importing registers the scene/env
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as _task_scene  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.flat_pack_crate")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so
    # the task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def base_q() -> torch.Tensor:
        return scene.base.data.root_quat_w

    def to_world(loc_xyz: tuple) -> torch.Tensor:
        loc = torch.tensor([list(loc_xyz)], device=device).expand(n, 3)
        return scene.base.data.root_pos_w + quat_apply(base_q(), loc)

    def qz_local(deg: float) -> torch.Tensor:
        h = math.radians(deg) / 2.0
        q = torch.zeros(n, 4, device=device)
        q[:, 0], q[:, 3] = math.cos(h), math.sin(h)
        return q

    def qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        aw, ax, ay, az = a.unbind(-1)
        bw, bx, by, bz = b.unbind(-1)
        return torch.stack([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ], dim=-1)

    def teleport(body, loc_xyz: tuple, q: torch.Tensor) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = to_world(loc_xyz)
        st[:, 3:7] = q
        body.write_root_state_to_sim(st, all_ids)

    def report(tag: str) -> None:
        dl = scene._base_local(scene.dressing.data.root_pos_w)[0]
        ll = scene._base_local(scene.lid.data.root_pos_w)[0]
        print(f"[solve] {tag:14s} | "
              f"fs={bool(scene.slot_filled('s')[0])} fw={bool(scene.slot_filled('w')[0])} "
              f"in={bool(scene.bottle_inside(scene.dressing)[0])} "
              f"lid={bool(scene.lid_seated()[0])} "
              f"dress=({float(dl[0]):+.3f},{float(dl[1]):+.3f},{float(dl[2]):+.3f}) "
              f"lid_z={float(ll[2]):+.3f} "
              f"latch=[{int(scene._filled_s[0])}{int(scene._filled_w[0])}"
              f"{int(scene._placed[0])}{int(scene._lidded[0])}] "
              f"breach={bool(scene._breach[0])} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # pieces settle at their scatter stations
    bp = (scene.base.data.root_pos_w - scene.env_origins)[0]
    bq = base_q()[0]
    byaw = math.degrees(2.0 * math.atan2(float(bq[3]), float(bq[0])))
    stn = int(scene.dressing_station[0])
    print(f"[solve] layout readback (seed {args.seed}): "
          f"base=({float(bp[0]):+.3f},{float(bp[1]):+.3f}) yaw={byaw:+.1f}deg "
          f"dressing at station {stn}", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert not bool(scene.slot_filled("s")[0]) and not bool(scene.slot_filled("w")[0]), \
        "slots must start empty"
    assert not bool(scene.bottle_inside(scene.dressing)[0]), \
        "dressing must start outside the crate"
    assert not bool(scene.lid_seated()[0]), "lid must start off the crate"
    assert bool(scene.decoy_excluded()[0]), "decoy must start outside"
    s0 = print_score("P0 reset+settle (flat-pack scattered)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phases 1+2: panels -> slot channels (gravity inserts them) -----------
    def insert_panel(panel, slot: str, tag: str) -> None:
        """Hover the panel in free air above its channel; gravity does the insert.
        Retries with a tiny lateral offset if a drop wedges on the flare lip."""
        if slot == "s":
            hover = lambda dx: (-c.slot_center + dx, 0.0, 0.185)  # noqa: E731
            q_t = base_q()  # thickness axis along base x
        else:
            hover = lambda dx: (0.0, -c.slot_center + dx, 0.185)  # noqa: E731
            q_t = qmul(base_q(), qz_local(90.0))  # thickness axis along base y
        for attempt, dx in enumerate((0.0, 0.003, -0.003)):
            teleport(panel, hover(dx), q_t)
            # descent: free fall into the funnel, rails guide, pad stops it
            for _ in range(300):
                env.step(no_action)
                if bool(scene._panel_seated(panel, slot)[0]):
                    break
            step(90)  # settle, hands-off
            if bool(scene._panel_seated(panel, slot)[0]):
                if attempt > 0:
                    print(f"[solve] {tag}: seated on retry {attempt} (dx={dx:+.3f})",
                          flush=True)
                return
            zloc = float(scene._base_local(panel.data.root_pos_w)[0][2])
            print(f"[solve] {tag}: drop attempt {attempt} did not seat "
                  f"(panel z={zloc:.3f}); lifting back to free air and retrying",
                  flush=True)
        report("FAIL-state")
        print(f"SIM_GEN_SOLVE: FAIL ({tag} never seated)", flush=True)
        os._exit(1)

    insert_panel(scene.panel_a, "s", "panel_a->S")
    report("panelA->S")
    assert bool(scene.slot_filled("s")[0]) and bool(scene._filled_s[0]), \
        "south slot must be filled and latched"
    assert not bool(scene.success()[0])
    s1 = print_score("P1 south panel seated in its channel")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_slot - 1e-6, f"P1 score {s1}"

    insert_panel(scene.panel_b, "w", "panel_b->W")
    report("panelB->W")
    assert bool(scene.slot_filled("w")[0]) and bool(scene._filled_w[0]), \
        "west slot must be filled and latched"
    assert not bool(scene.success()[0])
    s2 = print_score("P2 west panel seated; four walls up")
    assert s2 >= s1 - 1e-6 and s2 >= 2 * c.w_slot - 1e-6, f"P2 score {s2}"

    # ---------------- phase 3: bottle -> pad interior (gravity seats it) -------------------
    # Free-air pose on the vertical corridor through the still-open top: upright,
    # centred, bottom 10 mm above the pad. Falls and seats by contact.
    ident = torch.zeros(n, 4, device=device)
    ident[:, 0] = 1.0
    teleport(scene.dressing, (0.0, 0.0, c.pad_top + 0.010 + c.bottle_h / 2), ident)
    step(150)  # fall + settle, hands-off
    report("bottle->pad")
    assert bool(scene.bottle_inside(scene.dressing)[0]) and bool(scene._placed[0]), \
        "dressing must stand on the pad interior, latched"
    assert bool(scene.decoy_excluded()[0]), "decoy must remain outside"
    assert not bool(scene.success()[0]), "no success before the lid"
    s3 = print_score("P3 amber bottle standing inside the walls")
    assert s3 >= s2 - 1e-6 and s3 >= 2 * c.w_slot + c.w_bottle - 1e-6, f"P3 score {s3}"

    # ---------------- phase 4: lid -> wall tops (gravity seats it, lip registers) ----------
    # Free-air pose: level, lip down (spawn orientation), centred, plate underside
    # 10 mm above the wall tops; the lip hangs inside the wall opening touching
    # nothing (lip half 56 mm vs interior 65 mm). Falls onto the wall/panel tops.
    teleport(scene.lid, (0.0, 0.0, c.wall_top + 0.010 + c.lid_plate[2] / 2), base_q())
    step(240)  # fall + settle + still counter, hands-off
    report("lid->seated")
    if not bool(scene.lid_seated()[0]):
        step(240)
        report("lid-retry-wait")
    assert bool(scene.lid_seated()[0]) and bool(scene._lidded[0]), \
        "lid must seat level on the wall tops, latched"
    assert not bool(scene._breach[0]), "no breach may be latched in a clean solve"
    if not bool(scene.success()[0]):
        step(120)  # give the still counter margin
    assert bool(scene.success()[0]), "assembled + sealed + settled must be success"
    s4 = print_score("P4 lid seated; crate sealed around the bottle")
    assert s4 >= s3 - 1e-6 and s4 >= 1.0 - 1e-6, f"P4 score {s4} (expect 1.0)"

    # ---------------- phase 5: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s5 = print_score("P5 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s5 >= s4 - 1e-6
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
    except Exception:  # noqa: BLE001 - fail FAST; a hung Kit burns the forge slot
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)

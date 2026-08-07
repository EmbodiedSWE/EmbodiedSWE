"""Teleport solution for RidgePoiseScene (sim_gen task `approach_grasp_i29`) — the
task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. READ THE LOAD: the solver reads where each slug sits in its bar (the same state a
   camera would show — the pocket position is visible) and computes each bar's loaded
   balance point x_bal = m_slug * x_slug / (m_bar + m_slug) in the bar frame. The
   judged quantity is never written: it is the stack's settled equilibrium.
2. TRANSPORT RED (teleport, one co-move): a single root-state write per body carries
   the red bar — and its slug, preserving the slug's pose RELATIVE to the bar, as a
   level carry would — to a FREE-SPACE hover 8 mm above the ridge top, long axis
   across the fin, with the computed balance point over the fin's centerline; zero
   velocity. The write puts the bar in open air; it satisfies no rubric clause by
   itself (a hovering bar is not in the height band as "settled resting" — and it has
   not balanced anything).
3. POISE (gravity + contact, hands-off): the bar FALLS the 8 mm onto the 24 mm flat
   and either rests level — only if the balance point is inside the +/-12 mm window —
   or rotates off and falls to the floor. The equilibrium success() reads is produced
   entirely by gravity and contact, never written.
4. TRANSPORT BLUE + POISE: same co-move for the blue bar, crosswise, hovering over
   the red bar's rail deck with BLUE's balance point over the deck centerline and the
   bar centered over the ridge plane (so the combined mass stays over the fin). It
   falls onto the rails; the two-tier stack answers.
5. CLOSED LOOP: if a bar tips off (landing scatter), the solver re-reads the slug,
   recomputes the balance point and re-places — a regrasp-and-replace through free
   space — and lets gravity answer again. If the blue landing knocks the red bar off,
   the solver rebuilds from the red placement.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.approach_grasp_i29.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401  (registers simgen.ridge_poise)
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ridge_poise")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    RED, BLUE = 0, 1
    ratio = c.slug_mass / (c.slug_mass + c.bar_mass)
    hover = 0.008

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def bar(i):
        return scene.bars[c.bar_names[i]]

    def slug(i):
        return scene.slugs[c.slug_names[i]]

    def slug_loc(i) -> torch.Tensor:
        """Slug i's position in its bar's frame (the visible pocket readback)."""
        return scene._local(bar(i), slug(i).data.root_pos_w)

    def x_bal(i) -> float:
        """Bar i's loaded balance point along its long axis (bar frame, measured)."""
        return ratio * float(slug_loc(i)[0, 0])

    def ridge_loc(body) -> torch.Tensor:
        return scene._local(scene.ridge, body.data.root_pos_w)

    def report(tag: str) -> None:
        s = scene._status()
        parts = []
        for i, nm in enumerate(c.bar_names):
            rl = ridge_loc(bar(i))[0]
            sl = slug_loc(i)[0]
            parts.append(
                f"{nm}: ridge=({float(rl[0]):+.3f},{float(rl[1]):+.3f},{float(rl[2]):+.3f}) "
                f"lvl={bool(scene._level(bar(i))[0])} still={bool(scene._settled(bar(i))[0])} "
                f"slug=({float(sl[0]):+.3f},{float(sl[1]):+.3f},{float(sl[2]):+.3f})")
        print(f"[solve] {tag:14s} | red_poised={bool(s['red_poised'][0])} "
              f"blue_on_red={bool(s['blue_on_red'][0])} "
              f"slugs=({bool(s['slug_red'][0])},{bool(s['slug_blue'][0])} ) | "
              + " | ".join(parts)
              + f" | success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def co_place(i: int, loc_ridge: torch.Tensor, yaw_off: float) -> None:
        """TRANSPORT bar i (and its slug, pose preserved RELATIVE to the bar — a
        level carry) to a hover pose given in the RIDGE frame, long axis rotated
        `yaw_off` from the ridge x-axis; zero velocity. One write per body."""
        q_ridge = scene.ridge.data.root_quat_w
        p_ridge = scene.ridge.data.root_pos_w
        rel_p = scene._local(bar(i), slug(i).data.root_pos_w).clone()
        rel_q = quat_mul(quat_inv(bar(i).data.root_quat_w), slug(i).data.root_quat_w)
        yaw = torch.full((n,), yaw_off, device=device)
        qy = torch.zeros(n, 4, device=device)
        qy[:, 0], qy[:, 3] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        q_new = quat_mul(q_ridge, qy)
        p_new = p_ridge + quat_apply(q_ridge, loc_ridge)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = p_new
        st[:, 3:7] = q_new
        bar(i).write_root_state_to_sim(st, all_ids)
        ss = torch.zeros(n, 13, device=device)
        ss[:, 0:3] = p_new + quat_apply(q_new, rel_p)
        ss[:, 3:7] = quat_mul(q_new, rel_q)
        slug(i).write_root_state_to_sim(ss, all_ids)

    def place_red() -> None:
        """Hover the red bar across the fin with its measured balance point over the
        fin centerline; slug co-carried."""
        xb = x_bal(RED)
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0] = -xb  # bar center offset so the system CoM sits at ridge x = 0
        loc[:, 1] = 0.0
        loc[:, 2] = c.ridge_top_z + c.slab_t / 2 + hover
        print(f"[solve] red: slug_x={float(slug_loc(RED)[0, 0]):+.4f} "
              f"-> balance point {xb * 1000:+.1f} mm -> center at ridge x "
              f"{-xb * 1000:+.1f} mm", flush=True)
        co_place(RED, loc, 0.0)

    def place_blue() -> None:
        """Hover the blue bar crosswise over the red bar's rail deck: BLUE's balance
        point over the deck centerline (red's long axis), bar centered over the
        ridge plane so the combined mass stays over the fin."""
        xb = x_bal(BLUE)
        # red's actual center in the ridge frame (it settled where it balanced)
        red_y = float(ridge_loc(bar(RED))[0, 1])
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0] = 0.0            # blue mass over the fin: combined CoM stays put
        loc[:, 1] = red_y - xb     # blue balance point onto the deck centerline
        loc[:, 2] = c.ridge_top_z + c.slab_t + c.rail_h + c.slab_t / 2 + hover
        print(f"[solve] blue: slug_x={float(slug_loc(BLUE)[0, 0]):+.4f} "
              f"-> balance point {xb * 1000:+.1f} mm -> center at ridge y "
              f"{(red_y - xb) * 1000:+.1f} mm (red_y={red_y * 1000:+.1f} mm)", flush=True)
        co_place(BLUE, loc, math.pi / 2)

    def wait_pred(name: str, max_steps: int, lo_z: float) -> bool:
        """Hands-off until predicate `name` (live) holds, or the bar falls below
        `lo_z` in the ridge frame (tipped off — no point waiting), or budget out."""
        i = RED if name == "red_poised" else BLUE
        for k in range(max_steps):
            env.step(no_action)
            if k < 30:
                continue
            if bool(scene._status()[name][0]):
                return True
            if float(ridge_loc(bar(i))[0, 2]) < lo_z:
                print(f"[solve] {c.bar_names[i]} fell (ridge z="
                      f"{float(ridge_loc(bar(i))[0, 2]):+.3f})", flush=True)
                return False
        return bool(scene._status()[name][0])

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(150)
    rp = (scene.ridge.data.root_pos_w - scene.env_origins)[0]
    q = scene.ridge.data.root_quat_w
    ryaw = math.degrees(2.0 * math.atan2(float(q[0, 3]), float(q[0, 0])))
    print(f"[solve] layout readback (seed {args.seed}): "
          f"ridge=({float(rp[0]):+.3f},{float(rp[1]):+.3f}) yaw={ryaw:+.1f}deg "
          f"cell_x(sampled)={[round(float(v), 3) for v in scene.cell_x[0]]} "
          f"slug_x(measured)=({float(slug_loc(0)[0, 0]):+.4f},"
          f"{float(slug_loc(1)[0, 0]):+.4f})", flush=True)
    for i in range(2):
        p = (bar(i).data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] {c.bar_names[i]} spawn=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):+.3f})", flush=True)
    report("reset")
    for body in (*scene.bars.values(), *scene.slugs.values()):
        assert torch.isfinite(body.data.root_pos_w).all(), "NaN/inf after settle"
    st0 = scene._status()
    assert bool(st0["slug_red"][0]) and bool(st0["slug_blue"][0]), \
        "slugs must start seated in their pockets"
    assert not bool(st0["red_poised"][0]), "red bar must start on the floor"
    s0 = print_score("P0 reset+settle (both bars on the floor)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: poise the red bar on the ridge ------------------------------
    ok = False
    for attempt in range(5):
        place_red()
        if wait_pred("red_poised", 600, lo_z=c.ridge_top_z - 0.05):
            ok = True
            break
        print(f"[solve] red retry {attempt + 1}", flush=True)
        step(120)  # let the fallen bar come to rest before re-reading the slug
    if not ok:
        report("FAIL-red")
        print("SIM_GEN_SOLVE: FAIL (red bar never poised)", flush=True)
        os._exit(1)
    step(120)  # extra hands-off: the poise must be an equilibrium, not a pause
    report("red poised")
    assert bool(scene._status()["red_poised"][0]), "red poise did not persist"
    assert bool(scene._poised[0]), "poised latch did not set"
    s1 = print_score("P1 red bar poised level on the ridge, ends airborne")
    assert s1 >= s0 - 1e-6, "score decreased after the red poise"
    assert s1 >= c.w_poised - 1e-6, f"P1 score {s1} (expected >= {c.w_poised})"

    # ---------------- phase 2: poise the blue bar crosswise on the red bar -----------------
    s_prev = s1
    ok = False
    for attempt in range(6):
        if not bool(scene._status()["red_poised"][0]):
            # the blue landing knocked the red bar off — rebuild from the bottom
            print("[solve] rebuilding: red lost during blue placement", flush=True)
            place_red()
            if not wait_pred("red_poised", 600, c.ridge_top_z - 0.05):
                continue
            step(60)
        place_blue()
        if wait_pred("blue_on_red", 600, lo_z=c.ridge_top_z - 0.05):
            ok = True
            break
        print(f"[solve] blue retry {attempt + 1}", flush=True)
        step(120)
    if not ok:
        report("FAIL-blue")
        print("SIM_GEN_SOLVE: FAIL (blue bar never poised on red)", flush=True)
        os._exit(1)

    # closed loop: give settle flickers hands-off time; the stack must answer LIVE
    for k in range(12):
        if bool(scene.success()[0]):
            break
        step(60)
    report("stacked")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (cross-stack never settled as success)", flush=True)
        os._exit(1)
    s2 = print_score("P2 blue bar poised crosswise on red — full stack standing")
    assert s2 >= s_prev - 1e-6, "score decreased at the stack phase"
    assert s2 >= 0.999, f"success should score 1.0, got {s2}"

    # ---------------- persistence (>= 3 simulated seconds, hands-off) ----------------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s3 = print_score("P-persist 3.3 s hands-off")
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
    except BaseException as e:  # noqa: BLE001 - die fast, never idle until the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL (exception: {e})", flush=True)
        os._exit(2)

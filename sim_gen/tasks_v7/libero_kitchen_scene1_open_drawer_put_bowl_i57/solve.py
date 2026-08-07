"""Teleport solution for HanoiRingsScene (sim_gen task
`libero_kitchen_scene1_open_drawer_put_bowl_i57`) — the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, one ring at a time): a single root-state write carries a
   ring from wherever it rests to the free-space RELEASE POINT directly above the
   destination post's cone apex (post-local (0, 0, z_release)), lying flat with the
   post's yaw and zero velocity — exactly the pose a gripper pinching the ring's
   outer flats would carry it in after lifting it clear over its own post's tip.
   The write leaves the ring in open air; it satisfies no rubric clause by itself.
2. INSERTION (gravity + contact, hands-off): from the release point the ring FALLS —
   the cone tip funnels the hole onto the shaft, the shaft keeps it centred, and it
   lands on the base pad or on the ring below. Every fact the rubric checks
   (threaded, seated, order, contiguity, stillness) is produced by ballistics and
   contact, never written. If a ring hangs up on the cone or shaft, a small
   escalating downward force at its CoM (starting at ~0.5x its own weight) stands in
   for the fingertip push a robot would use — contact-consistent, cleared at once.
3. PLAN: the classic 7-move Tower of Hanoi recursion through the spare gray post —
   the ONLY move sequence of length 7 that never rests a larger ring on a smaller
   one. The scene's latched violation watchdog would cap the score at 0.15 forever
   if any move broke the rule; the printed score trace staying monotone to 1.0 IS
   the certificate that the rule was honoured.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene1_open_drawer_put_bowl_i57.solve --headless [--seed N]
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
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod

_qmul, _qz = scene_mod._qmul, scene_mod._qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

GRAV = 9.81


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.hanoi_rings")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)
    R, P = scene.RING_NAMES, scene.POST_NAMES
    ridx = {nm: i for i, nm in enumerate(R)}
    pidx = {nm: i for i, nm in enumerate(P)}
    letter = {"ring_large": "L", "ring_mid": "M", "ring_small": "S"}

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def posts_str() -> str:
        """Per-post bottom-up ring letters, e.g. G=[] g0=[LMS] g1=[]."""
        th = scene.threaded()[0]          # (R, P)
        z = scene.ring_post_z()[0]        # (R, P)
        out = []
        for p, tag in zip(range(3), ("G", "g0", "g1")):
            on = [(float(z[r, p]), letter[R[r]]) for r in range(3) if bool(th[r, p])]
            out.append(f"{tag}=[{''.join(s for _, s in sorted(on))}]")
        return " ".join(out)

    def report(tag: str) -> None:
        pos, _q, vel = scene._ring_tensors()
        print(f"[solve] {tag:12s} | {posts_str()} "
              f"vmax={float(vel[0].max()):.3f} "
              f"m1={bool(scene._m1[0])} lg={bool(scene._lg[0])} "
              f"md={bool(scene._md[0])} viol={bool(scene._viol[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_force(ring) -> None:
        ring.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def move(rname: str, pname: str, expect_below: int) -> None:
        """One Hanoi move. TRANSPORT the ring to the release point above `pname`'s
        cone apex (flat, post yaw, zero velocity), then let GRAVITY thread it down
        the shaft onto the pad / the `expect_below` rings already there. Escalating
        downward CoM nudge only if it hangs up on the cone or shaft."""
        ring = scene.rings[rname]
        post = scene.posts[pname]
        ri, pi = ridx[rname], pidx[pname]
        w_ring = float(c.ring_mass[ri]) * GRAV
        # seat gate: centre must come to rest at/below the top of the expected stack
        z_gate = c.pad_t + (expect_below + 0.5) * c.ring_h + 0.012
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.release_point_w(pname)
        st[:, 3:7] = post.data.root_quat_w
        ring.write_root_state_to_sim(st, all_ids)
        # hands-off fall: cone tip -> shaft -> pad/stack
        seated = False
        nudge = 0.0
        for i in range(720):
            if nudge > 0.0:
                f = torch.zeros(n, 1, 3, device=device)
                f[:, 0, 2] = -nudge
                ring.set_external_force_and_torque(f, zero_wrench, env_ids=all_ids,
                                                   is_global=True)
            env.step(no_action)
            zloc = float(scene.ring_post_z()[0, ri, pi])
            v = float(ring.data.root_lin_vel_w.norm(dim=-1)[0])
            th = bool(scene.threaded()[0, ri, pi])
            if i > 30 and th and zloc < z_gate and v < c.settle_speed:
                seated = True
                break
            if i >= 150 and i % 60 == 0 and v < 0.03 and zloc > z_gate:
                nudge = min(nudge + 0.5 * w_ring, 2.5 * w_ring)
                print(f"[solve] {rname} hung up over {pname} at z_loc={zloc:.3f}; "
                      f"downward nudge {nudge:.2f} N", flush=True)
        clear_force(ring)
        step(120)  # full settle, hands-off
        if not seated:
            report(f"{rname}-STUCK")
            print(f"SIM_GEN_SOLVE: FAIL ({rname} never threaded onto {pname})",
                  flush=True)
            os._exit(1)
        assert bool(scene.threaded()[0, ri, pi]), \
            f"{rname} left {pname} while settling"
        assert not bool(scene._viol[0]), \
            f"Hanoi violation latched after moving {rname} to {pname}"

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # tower seats onto the source post
    bp = (scene.board.data.root_pos_w - scene.env_origins)[0]
    bq = scene.board.data.root_quat_w[0]
    byaw = math.degrees(2.0 * math.atan2(float(bq[3]), float(bq[0])))
    slots = scene.slot_of_post[0].tolist()  # slot (0..2, -x..+x) of post i (G, g0, g1)
    print(f"[solve] layout readback (seed {args.seed}): "
          f"board=({float(bp[0]):+.3f},{float(bp[1]):+.3f}) yaw={byaw:+.1f}deg "
          f"slots(G,g0,g1)={slots} green_slot={slots[0]}", flush=True)
    report("reset")
    pos, _q, _v = scene._ring_tensors()
    assert torch.isfinite(pos).all(), "NaN/inf in ring states after settle"
    th0 = scene.threaded()[0]
    src = scene.SOURCE
    assert all(bool(th0[r, src]) for r in range(3)), \
        f"all rings must start threaded on {P[src]}, threaded={th0.tolist()}"
    assert int(th0[:, scene.TARGET].sum()) == 0, "no ring may start on the green post"
    z0 = scene.ring_post_z()[0, :, src]
    assert bool((z0[0] < z0[1]) & (z0[1] < z0[2])), \
        f"start tower must be L<M<S bottom-up, z={z0.tolist()}"
    assert not bool(scene._viol[0]), "fresh legal tower must not read as a violation"
    s0 = print_score("P0 reset+settle (tower of three rings on the source post)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phases 1-7: the Hanoi recursion --------------------------------------
    # S = post_gray0 (source), T = post_green (target), B = post_gray1 (buffer)
    S, T, B = P[1], P[0], P[2]

    move("ring_small", T, 0)          # 1. small S->T
    report("m1 S->T")
    s1 = print_score("P1 small ring threaded on the green post (mandatory first move)")
    assert bool(scene._m1[0]), "first-move latch did not set"
    assert s1 >= s0 - 1e-6 and s1 >= 0.14, f"P1 score {s1} (expect 0.15)"

    move("ring_mid", B, 0)            # 2. mid S->B
    report("m2 M->B")
    move("ring_small", B, 1)          # 3. small T->B (onto the mid ring)
    report("m3 S->B")
    s3 = print_score("P2 buffer stack built: mid+small parked on the spare gray post")
    assert s3 >= s1 - 1e-6, "score decreased while parking on the buffer"

    move("ring_large", T, 0)          # 4. large S->T  (the payoff move)
    report("m4 L->T")
    s4 = print_score("P3 large ring seated at the bottom of the green post")
    assert bool(scene._lg[0]), "large-home latch did not set"
    assert s4 >= 0.49, f"P3 score {s4} (expect m1+lg=0.50)"

    move("ring_small", S, 0)          # 5. small B->S
    report("m5 S->S")
    move("ring_mid", T, 1)            # 6. mid B->T (onto the large ring)
    report("m6 M->T")
    s6 = print_score("P4 mid ring stacked on the large: green tower two high")
    assert bool(scene._md[0]), "mid-home latch did not set"
    assert s6 >= 0.79, f"P4 score {s6} (expect m1+lg+md=0.80)"

    move("ring_small", T, 2)          # 7. small S->T — done
    report("m7 S->T")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the seventh move)", flush=True)
        os._exit(1)
    s7 = print_score("P5 tower rebuilt on the green post: large/mid/small")
    assert s7 >= s6 - 1e-6, "score decreased across the final move"

    # ---------------- phase 8: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s8 = print_score("P6 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s8 >= s7 - 1e-6
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
    main()

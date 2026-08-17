"""Teleport-transport solution for QCChuteScene (sim_gen task `robosuite_env_i408`) —
the task's legitimacy certificate.

Teleports are TRANSPORT ONLY (exactly what a gripper does: pick a disc off its pad,
set it down on the open test section of the ramp, release; or carry a rejected disc
back to its pad). Every load-bearing interaction is contact dynamics the solve merely
watches:

1. PROBE, pad by pad: set the disc from the next pad onto the OPEN upper test section
   of the ramp (hover 4 mm, released with zero velocity, aligned with the deck pitch)
   and do nothing for ~1.2 s. The physics answers: a ROUGH disc holds still (pair
   friction 0.70 vs tan 12 deg = 0.213, 3.3x margin — measured displacement < 2 mm);
   the POLISHED disc slides away on its own (pair friction 0.04, 4.2x margin).
2. A disc that held still is carried back to its pad (teleport-transport) and the next
   pad is probed.
3. The disc that slid needs nothing further: it accelerates down the ramp, shoots
   through the covered tunnel (where no gripper could follow), flies off the lip over
   the entry sill and is captured by the sealed cup — gravity and friction finish the
   task. The solve just waits, verifies success(), and holds hands-off.

The probe verdicts are cross-checked against the scene's hidden permutation readback
(`scene._slot`) and the authored materials (`get_material_properties`), so a wrong
verdict dies loudly instead of passing silently.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.robosuite_env_i408.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401 — import registers the scene + env
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.qc_chute")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        locs = scene.puck_locs()[0]
        s, y, h = scene.chute_coords(scene.puck_locs())
        print(f"[solve] {tag:12s} | "
              + " ".join(f"p{j}=({float(locs[j, 0]):+.3f},{float(locs[j, 1]):+.3f},"
                         f"{float(locs[j, 2]):+.3f})" for j in range(3))
              + f" s0={float(s[0, 0]):+.3f} track={bool(scene._track[0])}"
              f" tunnel={bool(scene._tunnel[0])}"
              f" in_cup={bool(scene.in_cup(scene.puck_locs()[:, 0])[0])}"
              f" success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    th = math.radians(c.pitch_deg)
    q_pitch = torch.tensor([math.cos(th / 2), 0.0, math.sin(th / 2), 0.0],
                           device=device).expand(n, 4)

    def put(body, loc_xyz, *, pitched: bool) -> None:
        """Teleport-transport: place the body at a rig-local position with zero
        velocity (deck-pitch aligned when released on the ramp)."""
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float32).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.rig.data.root_pos_w + quat_apply(scene.rig.data.root_quat_w, loc)
        q = scene.rig.data.root_quat_w
        st[:, 3:7] = quat_mul(q, q_pitch) if pitched else q
        body.write_root_state_to_sim(st, all_ids)

    def ramp_point(s: float, h: float) -> tuple:
        """Chute coords (s downslope, h above deck) -> rig-local xyz on the centreline."""
        return (c.mid_x + s * math.cos(th) + h * math.sin(th), 0.0,
                c.mid_z - s * math.sin(th) + h * math.cos(th))

    def s_of(body) -> float:
        s, _, _ = scene.chute_coords(scene.rig_local(body.data.root_pos_w))
        return float(s[0])

    # ---------------- phase 0: reset, settle, baseline ------------------------------------------
    step(180)
    rpos = (scene.rig.data.root_pos_w - scene.env_origins)[0]
    rq = scene.rig.data.root_quat_w[0]
    ryaw = math.degrees(2.0 * math.atan2(float(rq[3]), float(rq[0])))
    slot = scene._slot[0].tolist()          # slot[j] = pad of puck j; puck 0 = polished
    pad_of = {j: slot[j] for j in range(3)}
    puck_at = {slot[j]: j for j in range(3)}
    mats = [scene.pucks[nm].root_physx_view.get_material_properties().flatten().tolist()
            for nm in scene.PUCK_NAMES]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"rig=({float(rpos[0]):+.3f},{float(rpos[1]):+.3f}) yaw={ryaw:+.1f}deg "
          f"slot_of_puck={pad_of} (polished on pad {slot[0]}) "
          f"materials={[[round(v, 3) for v in m] for m in mats]}", flush=True)
    assert abs(mats[0][0] - c.mu_polished_s) < 0.01, f"puck_0 material did not take: {mats[0]}"
    for j in (1, 2):
        assert abs(mats[j][0] - c.mu_rough_s) < 0.01, f"puck_{j} material did not take"
    masses = scene.pucks["puck_0"].root_physx_view.get_masses()
    assert abs(float(masses.flatten()[0]) - c.puck_mass) < 0.02, \
        f"authored disc mass did not take: {masses}"
    report("reset")
    assert torch.isfinite(scene.puck_locs()).all(), "NaN after settle"
    s0 = print_score("P0 reset+settle (three identical discs on their pads)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1..k: probe pads in order until one disc slides ---------------------
    probe_loc = ramp_point(-0.10, c.puck_h / 2 + 0.004)
    scores = [s0]
    found = None
    for k, (px, py) in enumerate(c.pad_xy):
        j = puck_at[k]
        body = scene.pucks[scene.PUCK_NAMES[j]]
        put(body, probe_loc, pitched=True)
        step(6)                              # released; the latch (polished only) arms
        s_start = s_of(body)
        step(144)                            # ~1.2 s of pure physics — the experiment
        ds = s_of(body) - s_start
        moved = abs(ds) > 0.02
        print(f"[solve] probe pad {k} (puck_{j}): ds={ds * 1000:+.1f} mm -> "
              f"{'SLID (polished)' if moved else 'held (rough)'}", flush=True)
        assert moved == (j == 0), \
            f"probe verdict wrong: puck_{j} moved={moved} (polished is puck_0)"
        if not moved:
            assert abs(ds) < 0.02, "rough disc crept on the ramp"
            put(body, (px, py, c.pad_t + c.puck_h / 2 + 0.002), pitched=False)
            step(60)
            report(f"reject-pad{k}")
            s = print_score(f"P{k + 1} pad {k} probed: rough disc held, returned to its pad")
            assert s >= scores[-1] - 1e-6, "score decreased"
            assert not bool(scene.success()[0]), "no success from a rough probe"
            scores.append(s)
        else:
            found = k
            # it is already on its way: wait for tunnel entry, then capture
            for _ in range(300):
                env.step(no_action)
                if bool(scene._tunnel[0]):
                    break
            report("tunnel")
            assert bool(scene._track[0]), "track latch must be armed"
            assert bool(scene._tunnel[0]), "polished disc never entered the tunnel"
            s = print_score(f"P{k + 1} polished disc found on pad {k}: slid into the tunnel")
            assert s >= 0.60 - 1e-5, f"expected track+tunnel credit, got {s}"
            scores.append(s)
            break
    assert found is not None, "no disc slid — polished disc never found"
    assert found == slot[0], "probe order bookkeeping is broken"

    # ---------------- capture: flight + settle, all hands-off ------------------------------------
    for _ in range(600):
        env.step(no_action)
        if bool(scene.success()[0]):
            break
    step(90)
    report("captured")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the slide)", flush=True)
        os._exit(1)
    s_cap = print_score("P-final polished disc captured by the sealed cup")
    assert s_cap >= scores[-1] - 1e-6, "score decreased across the capture"
    assert s_cap >= 1.0 - 1e-5, f"success must score 1.0, got {s_cap}"

    # ---------------- persistence (>= 3.3 simulated seconds, hands-off) --------------------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P-persist persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s4 >= s_cap - 1e-6
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
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 — die loudly, never idle until the watchdog
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)

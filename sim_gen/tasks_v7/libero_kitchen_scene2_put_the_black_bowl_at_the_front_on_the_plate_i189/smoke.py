"""Smoke / rubric-REJECTION battery for WeighServeScene — NullRobot, teleported probes.

solve.py is the acceptance proof (weigh -> infer from the tilt readback -> serve, on all
three verdict branches). This battery proves the rubric REJECTS wrong outcomes — above
all that the plate is UNREACHABLE without a real, strictly pairwise comparison on the
balance, and that a comparison's verdict is BINDING. Every probe is CONSTRUCTED
(teleport, real physics steps, judge) — instrumentation, never a solution. Probes may
use the hidden mass assignment to BUILD states; judging reads only physics.

Checks:
  1. settle/no-NaN      — seeded reset settles finite: beam level on its pivot, bowls
                          at rest in the row slots, masses authored (one 0.40, two
                          0.06), score 0, no success, no spoil;
  2. randomization      — over 10 seeded resets the ballasted bowl's SLOT (position
                          readback) takes >= 2 values, the plate side takes both
                          values, and xy jitter readback differs between resets;
  3. null-policy        — 240 idle steps: nothing latches, score ~0, no success;
  4. SEED STRATEGY      — the seed's plan ("pick the front bowl, put it on the plate"):
                          the front-slot bowl carried straight over the plate with NO
                          weighing -> the spoil latch fires in the air; score 0,
                          success False, forever;
  5. history unforgiven — continuing after 4: an honest weighing is performed and the
                          BALLASTED bowl is served perfectly. End state = the goal
                          arrangement, score still 0, success still False;
  6. wrong-bowl (tilt)  — honest tilted weighing (verdict = lower pan's bowl), then
                          the LEFT-OUT light bowl is brought over the plate: it is not
                          the measurement-named bowl -> spoil, score 0;
  7. wrong-bowl (level) — the two LIGHT bowls weighed (beam level, verdict = left-out
                          ballasted bowl), then one of the PANNED bowls is served:
                          spoil, score 0 — serving what you weighed is not enough,
                          the verdict body is what counts;
  8. near-miss          — honest weighing, then the ballasted bowl set down 120 mm
                          from the plate axis (outside serve_xy_tol 55 mm, inside
                          near_r 150 mm): s_near latches, score capped at 0.60, no
                          success;
  9. not-a-comparison   — (a) a single bowl on one pan (beam slammed decisively) and
                          (b) two bowls stacked in one pan + one in the other: neither
                          is a strictly pairwise comparison -> the weighing never
                          latches either way;
 10. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run (forge): python -u -m simgen_tasks.<task>.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=12)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version on some GPUs and silently rejects RTX
# -> the annotator returns EMPTY frames. Disable the check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()

try:
    from . import scene as task_scene  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_WD = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_WD.daemon = True
_WD.start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        if rec and _REC["i"] % args.record_every == 0:
            for _f in range(3):  # flush accumulated history (ghosting fix)
                env.sim.render()
            arr = np.asarray(_REC["annot"].get_data())
            if arr.size:
                _REC["frames"].append(arr[..., :3].astype(np.uint8).copy())
        _REC["i"] += 1


def _refresh() -> None:
    _ENV.iscene.update(0.0)


def _all_ids():
    return torch.arange(_ENV.num_envs, device=_ENV.device)


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    in_p, in_n, _ob = scene.pan_occupancy()
    t = math.degrees(float(scene.tilt()[0]))
    print(f"[smoke] {tag:22s} | tilt={t:+6.2f}deg in_p={in_p[0].tolist()} "
          f"in_n={in_n[0].tolist()} weighed={bool(scene._s_weigh[0])} "
          f"verdict={int(scene._verdict[0])} spoiled={bool(scene._spoiled[0])} "
          f"served={bool(scene.served()[0])} success={bool(scene.success()[0])} "
          f"score={float(scene.score()[0]):.3f} frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.weigh_serve")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.60, -0.75, 1.05)) + o),
                                tuple(np.array((0.08, 0.02, 0.44)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        _REC["annot"] = annot if warm.size else None
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    ys = torch.tensor(c.slot_ys, device=device)

    def bowl_write(i: int, pos_w: torch.Tensor) -> None:
        """Teleport bowl i (upright, zero velocity) to a world point."""
        _refresh()
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3] = 1.0
        scene.bowls[i].write_root_state_to_sim(st, _all_ids())
        _refresh()

    def pan_target(sgn: float) -> torch.Tensor:
        """A point just above the (possibly tilted) pan's rim, from the LIVE beam pose."""
        _refresh()
        bp = scene.beam.data.root_pos_w
        bq = scene.beam.data.root_quat_w
        off = torch.tensor([[0.0, sgn * c.pan_dy, c.pan_z + c.rim_h + 0.006]],
                           device=device).expand(n, 3)
        return bp + quat_apply(bq, off)

    def weigh_pair(i_n: int, i_p: int) -> None:
        """Sequentially drop bowl i_n into the -y pan and i_p into the +y pan, then
        wait (hands-off) for the weighing latch — or 900 steps."""
        bowl_write(i_n, pan_target(-1.0))
        _step(300)
        bowl_write(i_p, pan_target(+1.0))
        for _ in range(900):
            _step(1)
            if bool(scene._s_weigh[0]):
                break
        _step(30)

    def plate_top(dx: float = 0.0, dy: float = 0.0, dz: float = 0.040) -> torch.Tensor:
        _refresh()
        tgt = scene.plate.data.root_pos_w.clone()
        tgt[:, 0] += dx
        tgt[:, 1] += dy
        tgt[:, 2] += c.plate_h / 2 + dz
        return tgt

    def serve(i: int, **kw) -> None:
        """Drop bowl i from 40 mm above the plate top (offsets via kw), settle."""
        bowl_write(i, plate_top(**kw))
        for _ in range(600):
            _step(1)
            if bool(scene.at_rest()[0, i]):
                break
        _step(30)

    def slot_of_body(i: int) -> int:
        _refresh()
        y = scene.bowls[i].data.root_pos_w[0, 1] - scene.env_origins[0, 1]
        return int((ys - y).abs().argmin())

    def body_in_slot(s: int) -> int:
        return [slot_of_body(i) for i in range(3)].index(s)

    # ================= 1. settle / no-NaN + mass authoring ========================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    masses = [float(scene.bowls[i].root_physx_view.get_masses().reshape(-1)[0])
              for i in range(3)]
    print(f"[smoke] authored masses readback: {masses}", flush=True)
    tilt0 = abs(math.degrees(float(scene.tilt()[0])))
    check("settle/no-NaN: seeded reset settles finite — beam level on its pivot, all "
          "bowls at rest in slots, masses authored (0.40/0.06/0.06), score 0, no "
          "success, no spoil",
          bool(scene._finite()[0]) and tilt0 < 1.0
          and bool(scene.at_rest()[0].all())
          and abs(masses[0] - c.mass_heavy) < 0.02
          and abs(masses[1] - c.mass_light) < 0.02
          and abs(masses[2] - c.mass_light) < 0.02
          and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]) and not bool(scene._spoiled[0]))

    # ================= 2. randomization is real (readback) ========================================
    heavy_slots, sides = set(), set()
    jit = []
    for s in range(10):
        torch.manual_seed(300 + 17 * s)
        env.reset()
        _step(10)
        _refresh()
        heavy_slots.add(slot_of_body(0))
        sides.add(int(scene.plate_side[0]))
        jit.append(torch.cat([scene.bowls[0].data.root_pos_w[0, :2],
                              scene.plate.data.root_pos_w[0, :2]]).clone())
    d_jit = max(float((a - b).abs().max()) for a in jit for b in jit)
    print(f"[smoke] randomization: heavy_slots={sorted(heavy_slots)} "
          f"plate_sides={sorted(sides)} max_jitter_delta={d_jit * 1000:.1f}mm", flush=True)
    check("randomization-is-real: over 10 seeded resets the ballasted bowl's slot "
          "takes >= 2 values, the plate side takes both values, xy jitter differs",
          len(heavy_slots) >= 2 and sides == {-1, 1} and d_jit > 0.003)

    # ================= 3. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps — nothing latches, score ~0, no success",
          not bool(scene._s_pan[0]) and not bool(scene._s_weigh[0])
          and not bool(scene._spoiled[0]) and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))

    # ================= 4. SEED STRATEGY: serve the front bowl, no weighing ========================
    # The seed's manipulation model is "pick the bowl at the front, put it on the
    # plate". Position readback names the front-slot bowl; it is carried straight
    # over the plate. The exclusion-zone latch fires while it is still in the AIR:
    # no weighing has happened, so the episode is spoiled before anything lands.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    front = body_in_slot(0)
    _REC["on"] = True
    bowl_write(front, plate_top())
    _step(5)
    spoiled_mid_air = bool(scene._spoiled[0])
    _step(240)
    _report("seed-strategy")
    check("SEED STRATEGY (no weighing): the front-slot bowl carried over the plate — "
          "spoil latch fires in the air, score 0, no success, forever",
          spoiled_mid_air and bool(scene._spoiled[0])
          and float(scene.score()[0]) <= 0.001 and not bool(scene.success()[0]))

    # ================= 5. history is not forgiven =================================================
    # Continue the spoiled episode: run a perfectly honest weighing, then serve the
    # BALLASTED bowl dead-centre. The final arrangement is exactly the goal state —
    # and it is still worth nothing.
    bowl_write(front, torch.tensor([[c.slot_x, c.slot_ys[0], c.surface_z + 0.003]],
                                   device=device).expand(n, 3)
               + scene.env_origins)
    _step(120)
    weigh_pair(1, 0)          # light vs BALLASTED -> tilt verdict = body 0
    serve(0)
    _report("history-unforgiven")
    check("history-unforgiven: honest weighing + perfect serve AFTER the spoil — end "
          "state equals the goal arrangement, success still False, score still 0",
          bool(scene._s_weigh[0]) and int(scene._verdict[0]) == 0
          and bool(scene.served()[0]) and bool(scene._spoiled[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.001)

    # ================= 6. wrong bowl after a TILTED weighing ======================================
    # Honest comparison of the ballasted bowl vs a light one: the beam slams, the
    # verdict names the lower pan's bowl (body 0). Bringing the LEFT-OUT light bowl
    # over the plate contradicts the measurement -> spoil.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    weigh_pair(1, 0)          # verdict: +y pan down -> body 0
    ok_weigh = bool(scene._s_weigh[0]) and int(scene._verdict[0]) == 0 \
        and not bool(scene._spoiled[0])
    s_mid = float(scene.score()[0])
    serve(2)                  # left-out LIGHT bowl over the plate
    _report("wrong-bowl-tilt")
    _REC["on"] = False
    check("wrong-bowl-after-tilt: honest tilted weighing (verdict = body in the lower "
          "pan), then the left-out light bowl brought over the plate — spoil, score "
          "wiped from 0.35 to 0, no success",
          ok_weigh and s_mid >= 0.34 and bool(scene._spoiled[0])
          and float(scene.score()[0]) <= 0.001 and not bool(scene.success()[0]))

    # ================= 7. wrong bowl after a LEVEL weighing =======================================
    # The two LIGHT bowls are compared: the beam settles level, so the verdict names
    # the LEFT-OUT (ballasted) bowl. Serving one of the bowls that WAS weighed is a
    # contradiction of the measurement -> spoil. (Serving the left-out bowl instead
    # is exactly solve.py's level branch, proven on the forge.)
    torch.manual_seed(100)
    env.reset()
    _step(60)
    weigh_pair(1, 2)          # light vs light -> LEVEL -> verdict = left-out body 0
    ok_level = bool(scene._s_weigh[0]) and int(scene._verdict[0]) == 0 \
        and abs(math.degrees(float(scene.tilt()[0]))) <= c.level_max_deg \
        and not bool(scene._spoiled[0])
    serve(1)                  # a PANNED (weighed, light) bowl over the plate
    _report("wrong-bowl-level")
    check("wrong-bowl-after-level: level weighing of the two light bowls (verdict = "
          "left-out ballasted bowl), then a panned bowl served — spoil, score 0",
          ok_level and bool(scene._spoiled[0])
          and float(scene.score()[0]) <= 0.001 and not bool(scene.success()[0]))

    # ================= 8. near-miss: right bowl, misses the plate =================================
    # Honest weighing, then the ballasted bowl is set down on the counter 120 mm from
    # the plate axis: inside near_r (150 mm, s_near latches) but far outside
    # serve_xy_tol (55 mm). Credit caps at 0.60; no success.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    weigh_pair(1, 0)
    _refresh()
    away = 1.0 if float(scene.env_origins[0, 1] - scene.plate.data.root_pos_w[0, 1]) > 0 \
        else -1.0  # offset toward the counter centre, away from the edge
    serve(0, dx=0.0, dy=away * 0.120, dz=0.030)
    _refresh()
    d_xy = float((scene.bowls[0].data.root_pos_w[0, :2]
                  - scene.plate.data.root_pos_w[0, :2]).norm())
    _report("near-miss")
    print(f"[smoke] near-miss: ballasted bowl {d_xy * 1000:.0f}mm from the plate axis "
          f"(serve gate {c.serve_xy_tol * 1000:.0f}mm, near latch "
          f"{c.near_r * 1000:.0f}mm)", flush=True)
    check("near-miss: after an honest weighing the ballasted bowl rests 120 mm off "
          "the plate axis — s_near latches, score caps at 0.60, no success, no spoil",
          bool(scene._s_weigh[0]) and not bool(scene._spoiled[0])
          and d_xy > c.serve_xy_tol + 0.02 and d_xy < c.near_r
          and bool(scene._s_near[0]) and not bool(scene.success()[0])
          and 0.59 <= float(scene.score()[0]) <= 0.6001)

    # ================= 9. not-a-comparison ========================================================
    # (a) ONE bowl on a pan: the beam slams decisively, everything settles — but a
    # one-sided load is not a comparison (no bowl in the other pan) -> no latch.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    bowl_write(0, pan_target(+1.0))
    _step(400)
    _report("single-bowl")
    only_one = not bool(scene._s_weigh[0]) \
        and abs(math.degrees(float(scene.tilt()[0]))) >= c.tilt_min_deg
    # (b) TWO bowls in one pan (stacked) + one in the other: pairwise means exactly
    # one per pan -> no latch either, no matter how settled and decisive it looks.
    # Construction order matters: the crowded pan is built FIRST so that no prefix of
    # the construction is itself a valid one-per-pan comparison.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    bowl_write(0, pan_target(+1.0))
    _step(240)
    _refresh()
    top = scene.bowls[0].data.root_pos_w.clone()
    top[:, 2] += c.bowl_h + 0.006
    bowl_write(2, top)        # stack the third bowl into/onto the +y pan's bowl
    _step(240)
    bowl_write(1, pan_target(-1.0))
    _step(400)
    in_p, in_n, _ob = scene.pan_occupancy()
    _report("two-in-one-pan")
    crowded = int(in_p[0].sum()) >= 2 and int(in_n[0].sum()) == 1 \
        and not bool(scene._s_weigh[0])
    check("not-a-comparison: a single-bowl load (decisive slam) and a 2-vs-1 load "
          "both settle without ever latching the weighing",
          only_one and crowded)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.weigh_serve")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    check("video: frames captured and saved to frames.npz", len(_REC["frames"]) > 10)

    n_pass = sum(ok for _nm, ok in checks)
    n_tot = len(checks)
    if n_pass == n_tot:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{n_tot}", flush=True)
        code = 0
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{n_tot}", flush=True)
        code = 1

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
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({exc!r})", flush=True)
        os._exit(1)

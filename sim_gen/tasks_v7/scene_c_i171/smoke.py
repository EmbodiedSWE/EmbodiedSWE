"""Smoke battery for BeamBalanceScene (sim_gen task `scene_c_i171`) —
REJECTION-ONLY: every check either verifies basic health/randomization or
CONSTRUCTS (or physically drives) a settled wrong outcome and asserts the
rubric refuses it. success() must never fire anywhere in the battery.

 1. settle/no-NaN     — beam resting ON its stop on the cargo side, cargo
                        seated, counter pocket empty, spares in the depot,
                        score ~0, no success.
 2. randomization A   — cargo band count k AND side vary across 10 seeded
                        resets, and the LIVE state tracks the sample every
                        time (the sampled slab is the seated one, the beam
                        pins on the sampled side's stop).
 3. randomization B   — the candidate cubes' sampled ground scatter varies
                        across resets and the live cubes track it (readback).
 4. null policy       — 300 idle steps -> beam still pinned at its stop,
                        score ~0, no success.
 5. under-by-one      — PHYSICAL: the subset for k-1 units really dropped
                        into the counter pocket (loaded fires — the probe is
                        live); the beam rises off its stop but settles far
                        outside theta_tol, still on the cargo side; the lifted
                        latch never fires; score capped at loaded credit, no
                        success.
 6. over-by-one       — PHYSICAL: the subset for k+1 units dropped in
                        (lightest first, so no intermediate stack ever totals
                        exactly k — a heaviest-first prefix would be a genuine
                        balanced success mid-probe); the
                        beam swings THROUGH level and settles far outside
                        theta_tol on the OTHER side (sign flip asserted — the
                        overweight really acted); no success.
 7. seed reflex       — the whole linkage (beam + seated cargo) written at
                        the OPPOSITE end-stop — the CALVIN "actuate the DOF
                        to its other end" move; hands off, gravity returns it
                        to the cargo-side stop; score ~0, no success.
 8. wrong pocket      — the correct-mass subset dropped into the CARGO
                        pocket (on top of the slab): the beam stays pinned,
                        the counter pocket stays empty, score ~0, no success.
 9. spare-slab cheat  — a spare depot slab dropped into the counter pocket
                        plus candidates topping up to exact balance: the beam
                        stands LEVEL (asserted — the cheat really balances),
                        candidates all legal, but the depot clause refuses
                        success; score capped, no success.
10. cargo removed     — the seated cargo teleported off to open ground: the
                        empty beam self-levels (asserted), but cargo_seated
                        is False -> score ~0, no success.
11. near-miss angle   — the whole linkage (beam + cargo + correct subset)
                        CONSTRUCTED consistently at theta_tol + 0.03: every
                        other clause holds (seated/loaded/legal/depot/
                        settled), only the LEVEL gate refuses (judged without
                        stepping; then reset).
12. level but moving  — the same construct AT level but with the beam swinging
                        (0.5 rad/s written): the settle gate refuses (judged
                        without stepping; then reset).
13. rejection audit   — success() observed False at every step of the battery.
14. final no-NaN.
15. video             — frames.npz (>10 frames) written to the CWD.

Run (forge): python -u -m simgen_tasks.scene_c_i171.smoke --headless
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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() ----------------------------------------------------------
_ENV = None
_REC = {"on": True, "annot": None, "frames": [], "i": 0}
_AUDIT = {"saw_success": False}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        _AUDIT["saw_success"] |= bool(env.scene.success()[0])
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
    print(f"[smoke] {tag:16s} | tilt={float(scene.beam_tilt()[0]):+.4f} "
          f"k={int(float(scene.layout[0, 0]))} side={float(scene.layout[0, 1]):+.0f} "
          f"seated={bool(scene.cargo_seated()[0])} loaded={bool(scene.counter_loaded()[0])} "
          f"legal={bool(scene.cands_legal()[0])} depot={bool(scene.spares_in_depot()[0])} "
          f"latch(ld/lf)=({int(scene._loaded_l[0])},{int(scene._lifted_l[0])}) "
          f"level={bool(scene.level()[0])} settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main --------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.beam_balance")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    stop = math.radians(c.stop_deg)

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.25, -1.05, 0.85)) + o),
                                tuple(np.array((0.0, -0.05, 0.22)) + o),
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
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video",
              flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def succ() -> bool:
        s = bool(scene.success()[0])
        _AUDIT["saw_success"] |= s
        return s

    def lay_k() -> int:
        return int(float(scene.layout[0, 0]))

    def lay_side() -> float:
        return float(scene.layout[0, 1])

    def reset_until(pred, base_seed: int, tag: str) -> None:
        """Seeded resets until the sampled layout satisfies `pred(k)` (then a
        long settle). The randomization is the honest source of variety."""
        for i in range(40):
            torch.manual_seed(base_seed + i)
            env.reset()
            _step(20)
            _refresh()
            if pred(lay_k()):
                _step(130)
                print(f"[smoke] {tag}: seed {base_seed + i} -> k={lay_k()} "
                      f"side={lay_side():+.0f}", flush=True)
                return
        raise AssertionError(f"{tag}: no seed produced a suitable k")

    def write_state(body, pos3, quat4, wy: float = 0.0) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = scene.env_origins[:, 0] + pos3[0]
        st[:, 1] = scene.env_origins[:, 1] + pos3[1]
        st[:, 2] = scene.env_origins[:, 2] + pos3[2]
        st[:, 3:7] = torch.tensor(quat4, device=device)
        st[:, 11] = wy
        body.write_root_state_to_sim(st, _all_ids())
        _refresh()

    def roty(local, a):
        """Rotate a local (x, y, z) by `a` about +y (matches beam_tilt = +a)."""
        ca, sa = math.cos(a), math.sin(a)
        return (local[0] * ca + local[2] * sa, local[1], -local[0] * sa + local[2] * ca)

    def place_in_pocket(body, pocket_side: float, seat_z: float, h_body: float) -> None:
        """Transport teleport: write `body` inside the pocket mouth on
        `pocket_side`, orientation-matched to the LIVE beam, its centre
        `seat_z` above the pocket-floor plane (beam-local), zero velocity."""
        bq = scene.beam.data.root_quat_w
        lp = torch.zeros(n, 3, device=device)
        lp[:, 0] = pocket_side * c.pock_x
        lp[:, 2] = c.floor_top + seat_z + h_body / 2
        world = scene.beam.data.root_pos_w + scene._quat_apply(bq, lp)
        st = torch.zeros(n, 13, device=device)
        st[:, :3] = world
        st[:, 3:7] = bq
        body.write_root_state_to_sim(st, _all_ids())
        _refresh()

    def settle(max_steps: int = 500, min_steps: int = 60) -> None:
        for i in range(max_steps):
            _step(1)
            if i >= min_steps and bool(scene.settled()[0]):
                break

    def drop_subset(units_total: int, pocket_side: float, base_z: float = 0.0,
                    ascending: bool = False) -> list[int]:
        """PHYSICAL probe: drop the candidate subset for `units_total` into the
        pocket on `pocket_side`, heaviest first by default, stacking; settle
        each. `ascending=True` drops lightest first — needed by the over-by-one
        probe so that no intermediate stack ever totals exactly k (the k+1
        subsets {1,2} and {1,4} contain the exact-k subsets {2} and {4}, and a
        heaviest-first prefix would be a GENUINE balanced success mid-probe)."""
        subset = [u for u in sorted(c.cand_units, reverse=not ascending)
                  if units_total & u]
        stack = base_z
        for u in subset:
            cand = scene.cands[c.cand_units.index(u)]
            place_in_pocket(cand, pocket_side, stack + 0.006, c.cand_s)
            stack += c.cand_s
            settle()
        return subset

    def write_assembly(tilt: float, counter_units: int, beam_w: float = 0.0) -> None:
        """Construct the WHOLE linkage consistently at `tilt` (teleporting one
        body of a jointed pair gets depenetrated back — write it all): beam at
        `tilt` about y, the sampled cargo seated in its side pocket, the
        candidate subset for `counter_units` stacked in the counter pocket."""
        k, side = lay_k(), lay_side()
        q = (math.cos(tilt / 2), 0.0, math.sin(tilt / 2), 0.0)
        write_state(scene.beam, (0.0, 0.0, c.pivot_z), q, wy=beam_w)
        h = k * c.band_h
        lp = (side * c.pock_x, 0.0, c.floor_top + h / 2 + 0.0005)
        px, py, pz = roty(lp, tilt)
        write_state(scene.cargos[k - 1], (px, py, pz + c.pivot_z), q, wy=beam_w)
        stack = 0.0
        for u in sorted(c.cand_units, reverse=True):
            if counter_units & u:
                lpc = (-side * c.pock_x, 0.0,
                       c.floor_top + stack + c.cand_s / 2 + 0.0005)
                cx, cy, cz = roty(lpc, tilt)
                write_state(scene.cands[c.cand_units.index(u)],
                            (cx, cy, cz + c.pivot_z), q, wy=beam_w)
                stack += c.cand_s

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(11)
    env.reset()
    _step(150)
    _report("reset")
    check("settle/no-NaN: beam ON its stop on the cargo side, cargo seated, "
          "counter empty, spares in depot, score ~0, no success",
          bool(scene._finite()[0])
          and abs(float(scene.beam_tilt()[0])) > stop - 0.05
          and float(scene.beam_tilt()[0]) * lay_side() > 0
          and bool(scene.cargo_seated()[0])
          and not bool(scene.counter_loaded()[0])
          and bool(scene.spares_in_depot()[0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 2+3. randomization readback ================================================
    ks, sides, cxy, tracks, cand_tracks = [], [], [], [], []
    for i in range(10):
        torch.manual_seed(20 + i)
        env.reset()
        _step(90)
        _refresh()
        ks.append(lay_k())
        sides.append(lay_side())
        tracks.append(bool(scene.cargo_seated()[0])
                      and float(scene.beam_tilt()[0]) * lay_side() > 0
                      and abs(float(scene.beam_tilt()[0])) > stop - 0.06)
        row, ok_c = [], True
        for j, cand in enumerate(scene.cands):
            live = (cand.data.root_pos_w - scene.env_origins)[0, :2]
            want = scene.layout[0, 2 + 2 * j:4 + 2 * j]
            ok_c &= float((live - want).norm()) < 0.05
            row += [float(want[0]), float(want[1])]
        cxy.append(row)
        cand_tracks.append(ok_c)
    c_std = float(np.std(np.asarray(cxy), axis=0).mean())
    print(f"[smoke] readback: ks={ks} sides={[int(s) for s in sides]} "
          f"cand_std={c_std:.3f} tracks={all(tracks)} cand_tracks={all(cand_tracks)}",
          flush=True)
    check("randomization A: cargo band count AND side vary across resets, and "
          "the live state tracks the sample every time (sampled slab seated, "
          "beam pinned on the sampled side)",
          len(set(ks)) >= 2 and len(set(sides)) == 2 and all(tracks))
    check("randomization B: the candidate ground scatter varies across resets "
          "and the live cubes track the sample",
          c_std > 0.02 and all(cand_tracks))

    # ================= 4. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(300)
    _report("null")
    check("null policy: 300 idle steps -> beam still pinned at its stop, "
          "score ~0, no success",
          abs(float(scene.beam_tilt()[0])) > stop - 0.05
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 5. under-by-one (physical) =================================================
    reset_until(lambda k: k >= 2, 40, "under-by-one")
    side5 = lay_side()
    drop_subset(lay_k() - 1, -side5)
    settle(800)
    _report("under-by-one")
    check("under-by-one: the k-1 subset really dropped into the counter pocket "
          "(loaded fires — the probe is live) but the beam settles far outside "
          "theta_tol, still on the cargo side; no lifted latch; score capped "
          "at loaded credit; no success",
          bool(scene.counter_loaded()[0])
          and abs(float(scene.beam_tilt()[0])) > c.theta_tol + 0.05
          and float(scene.beam_tilt()[0]) * side5 > 0
          and not bool(scene._lifted_l[0])
          and float(scene.score()[0]) <= c.w_loaded + 1e-4 and not succ())

    # ================= 6. over-by-one (physical) ==================================================
    reset_until(lambda k: k <= 4, 60, "over-by-one")
    side6 = lay_side()
    # lightest first: an ascending prefix is only ever {1u}, never exactly k
    # units, so the probe cannot pass through a genuine balanced-success state
    drop_subset(lay_k() + 1, -side6, ascending=True)
    settle(900)
    _report("over-by-one")
    check("over-by-one: the k+1 subset dropped in — the beam swings through "
          "level and settles far outside theta_tol on the OTHER side (sign "
          "flip: the overweight really acted); no success",
          bool(scene.counter_loaded()[0])
          and abs(float(scene.beam_tilt()[0])) > c.theta_tol + 0.05
          and float(scene.beam_tilt()[0]) * side6 < 0
          and not succ())

    # ================= 7. seed reflex (drive the DOF to its other end-stop) ======================
    torch.manual_seed(75)
    env.reset()
    _step(120)
    side7 = lay_side()
    write_assembly(-side7 * c.tilt0, counter_units=0)   # the CALVIN end-stop move
    _refresh()
    flipped = float(scene.beam_tilt()[0]) * side7 < 0
    _step(500)
    _report("seed-reflex")
    check("seed reflex: the beam+cargo linkage written at the OPPOSITE "
          "end-stop (the seed's 'actuate the DOF' move) — hands off, gravity "
          "returns it to the cargo-side stop; score ~0, no success",
          flipped
          and float(scene.beam_tilt()[0]) * side7 > 0
          and abs(float(scene.beam_tilt()[0])) > stop - 0.06
          and bool(scene.cargo_seated()[0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 8. wrong pocket (subset dropped onto the cargo) ============================
    reset_until(lambda k: k <= 2, 80, "wrong-pocket")
    side8 = lay_side()
    drop_subset(lay_k(), side8, base_z=lay_k() * c.band_h)  # onto the slab, cargo side
    settle(600)
    _report("wrong-pocket")
    check("wrong pocket: the correct-mass subset dropped into the CARGO pocket "
          "(stacked on the slab) — the beam stays pinned, the counter pocket "
          "stays empty, score ~0, no success",
          abs(float(scene.beam_tilt()[0])) > stop - 0.06
          and float(scene.beam_tilt()[0]) * side8 > 0
          and not bool(scene.counter_loaded()[0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 9. spare-slab cheat (depot clause isolated) ================================
    reset_until(lambda k: k >= 2, 100, "spare-slab")
    side9 = lay_side()
    # a spare 1-band slab from the depot + candidates for k-1 = exact balance
    place_in_pocket(scene.cargos[0], -side9, 0.006, c.band_h)
    settle()
    drop_subset(lay_k() - 1, -side9, base_z=c.band_h)
    settle(900)
    _report("spare-slab")
    check("spare-slab cheat: a depot slab + candidates really BALANCE the beam "
          "level (asserted) with every candidate legal — but the depot clause "
          "refuses success; score capped, no success",
          bool(scene.level()[0])
          and bool(scene.counter_loaded()[0])
          and bool(scene.cands_legal()[0])
          and not bool(scene.spares_in_depot()[0])
          and float(scene.score()[0]) <= 0.45 + 1e-4 and not succ())

    # ================= 10. cargo removed ==========================================================
    torch.manual_seed(115)
    env.reset()
    _step(120)
    k10 = lay_k()
    write_state(scene.cargos[k10 - 1], (0.60, 0.60, k10 * c.band_h / 2 + 0.002),
                (1.0, 0.0, 0.0, 0.0))
    settle(900)
    _report("cargo-removed")
    check("cargo removed: with the slab lifted off, the empty beam self-levels "
          "(asserted — the keel works) but cargo_seated is False: score ~0, "
          "no success",
          bool(scene.level()[0])
          and not bool(scene.cargo_seated()[0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 11. near-miss angle (level gate, judged pre-step) =========================
    torch.manual_seed(125)
    env.reset()
    _step(120)
    write_assembly(lay_side() * (c.theta_tol + 0.03), counter_units=lay_k())
    near_all_but_level = bool(scene.cargo_seated()[0]) \
        and bool(scene.counter_loaded()[0]) and bool(scene.cands_legal()[0]) \
        and bool(scene.spares_in_depot()[0]) and bool(scene.settled()[0]) \
        and not bool(scene.level()[0])
    near_rejected = not succ()  # judged WITHOUT stepping
    _report("near-miss")
    check("near-miss angle: the full correct-subset linkage constructed at "
          "theta_tol + 0.03 — every other clause holds and only the LEVEL "
          "gate refuses success (judged without stepping)",
          near_all_but_level and near_rejected)

    # ================= 12. level but moving (settle gate, judged pre-step) =======================
    torch.manual_seed(135)
    env.reset()
    _step(120)
    write_assembly(0.0, counter_units=lay_k(), beam_w=0.5)
    move_all_but_settled = bool(scene.cargo_seated()[0]) \
        and bool(scene.counter_loaded()[0]) and bool(scene.level()[0]) \
        and not bool(scene.settled()[0])
    move_rejected = not succ()  # judged WITHOUT stepping
    _report("level-moving")
    check("level but moving: the same construct AT level with the beam "
          "swinging at 0.5 rad/s — the settle gate refuses success (judged "
          "without stepping)",
          move_all_but_settled and move_rejected)

    # ================= 13-15. audit, no-NaN, video ================================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    torch.manual_seed(145)
    env.reset()
    _step(60)
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.beam_balance")
        print(f"[smoke] wrote {args.out}: {arr.shape}", flush=True)
    check("video: >10 frames recorded", len(_REC["frames"]) > 10)

    n_pass = sum(1 for _, ok in checks if ok)
    if n_pass == len(checks):
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
        code = 0
    else:
        for nm, ok in checks:
            if not ok:
                print(f"[smoke] FAILED: {nm}", flush=True)
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        code = 1
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
    except Exception as e:  # noqa: BLE001 - fast fail beats a 20-min watchdog hang
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)

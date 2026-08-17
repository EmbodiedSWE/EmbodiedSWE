"""Smoke battery for WeighStationScene (sim_gen task `handover_i124`) — REJECTION-ONLY:
every check either verifies basic health/randomization or CONSTRUCTS a settled wrong
outcome and asserts the rubric refuses it. success() must never fire anywhere in the
battery.

 1. settle/no-NaN     — empty beam rests LEVEL (pendulum bob works), cargo on the
                        apron, score ~0, no success.
 2. randomization A   — station yaw spans > 90 deg, xy jitters, k (which parcel
                        ships) varies across resets.
 3. randomization B   — READBACK: the shipment (parcel k-1) stands on the green pad,
                        the other two parcels park > 1 m away in the depot, all four
                        counterweights rest near their apron slots (station-local).
 4. null policy       — 240 idle steps -> score ~0, no success.
 5. SEED strategy     — the seed's whole skill (carry the object to the destination)
                        as a construct: parcel dropped into the cradle, NO ballast.
                        The scale keels far past the level tolerance (settled);
                        cradle latch pays 0.30, never more; no success.
 6. under-ballast     — k-1 weights seated (k >= 2 episode) + parcel delivered: the
                        one-unit shortfall reads ~14 deg cradle-down, outside
                        tolerance while everything is settled; no success.
 7. over-ballast      — k+1 weights seated (k = 1 episode) + parcel delivered: one
                        unit excess reads ~14 deg rack-down; exact-count refuses
                        too; latched credit capped at 0.60; no success.
 8. mirrored placement— (k = 1) the WEIGHT dropped in the cradle and the PARCEL in
                        a rack pocket: equal masses at equal radius — the beam is
                        LEVEL and settled, yet every rubric clause refuses (wrong
                        objects in wrong receptacles); score stays ~0.
 9. unclean rider     — exact k in rack AND parcel in cradle AND the beam LEVEL and
                        settled — but a spare weight rides the bar near the pivot
                        (~2.5 deg of bias, inside tolerance): the cleanliness
                        clause alone refuses. Placed FIRST so success is false
                        throughout the construction.
10. delivery miss     — ballast-only: k weights seated, parcel left on the pad. The
                        scale reads rack-down past tolerance; ballast latch pays
                        0.30, never more; no success.
11. latched credit    — parcel delivered (latch 0.30) then STOLEN off the scale,
                        then k weights seated (latch 0.30): score holds at the
                        0.60 cap, success never fires (nothing in the cradle).
12. rejection audit   — success() observed False at every step of the battery.
13. final no-NaN.
14. video             — frames.npz (>10 frames) written to the CWD.

Run (forge): python -u -m simgen_tasks.handover_i124.smoke --headless
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

# ----- module state wired up in main() ------------------------------------------------------------
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


def _write_body(body, pos_w: torch.Tensor, quat: torch.Tensor | None = None) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    in_rack, _ = scene._weight_flags()
    print(f"[smoke] {tag:18s} | tilt={math.degrees(float(scene.tilt()[0])):+6.1f}deg "
          f"rate={float(scene.rate_fd[0]):+.3f} k={int(scene.k_units[0])} "
          f"in_rack={int(in_rack[0].sum())} "
          f"cradled={bool(scene.in_cradle()[0])} level={bool(scene.level()[0])} "
          f"latch(b)={float(scene._ballast[0]):.2f} latch(p)={int(scene._parcel[0])} "
          f"settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.weigh_station")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.05, -0.95, 0.80)) + o),
                                tuple(np.array((0.00, 0.00, 0.12)) + o),
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

    def yaw_of(quat_row: torch.Tensor) -> float:
        return math.degrees(2.0 * math.atan2(float(quat_row[3]), float(quat_row[0])))

    def station_local(pos_w: torch.Tensor) -> torch.Tensor:
        return quat_apply_inverse(scene.station.data.root_quat_w,
                                  pos_w - scene.station.data.root_pos_w)

    def beam_release(local_xyz, up: float) -> torch.Tensor:
        """World release point: live beam-local point + `up` STRAIGHT UP in world
        (a world-vertical fall lands centred regardless of the beam's tilt)."""
        loc = torch.tensor(local_xyz, device=device, dtype=torch.float).expand(n, 3)
        p = scene.beam.data.root_pos_w + quat_apply(scene.beam.data.root_quat_w, loc)
        p = p.clone()
        p[:, 2] += up
        return p

    def kk() -> int:
        return int(scene.k_units[0])

    def drop_parcel_cradle(j: int, up: float = 0.030) -> None:
        """CONSTRUCT (gravity + contact): parcel j released above the cradle rest."""
        rest = c.floor_z1 + c.parcel_sizes[j] / 2
        _write_body(scene.parcels[j], beam_release([c.arm_r, 0.0, rest], up),
                    scene.beam.data.root_quat_w)
        _step(150)

    def drop_weight_pocket(u: int, pocket: int, up: float = 0.012) -> None:
        """CONSTRUCT (gravity + contact): weight u released just above pocket
        `pocket` (release inside the mouth: the beam-local entry offset
        up*sin(20 deg) ~ 4 mm stays inside the ±8 mm pocket slop at the stop)."""
        rest = c.floor_z1 + c.weight_size / 2
        _write_body(scene.weights[u],
                    beam_release([-c.arm_r, c.pocket_y[pocket], rest], up),
                    scene.beam.data.root_quat_w)
        _step(150)

    def reset_with_k(want: tuple, seed0: int) -> int:
        """Reset until the episode's k lands in `want` (k is a readback, so hunt)."""
        for s in range(seed0, seed0 + 60):
            torch.manual_seed(s)
            env.reset()
            _step(30)
            if kk() in want:
                return s
        raise AssertionError(f"no seed in [{seed0}, {seed0 + 60}) gives k in {want}")

    tol = math.radians(c.level_tol_deg)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(11)
    env.reset()
    _step(150)
    _report("reset")
    check("settle/no-NaN: empty beam rests LEVEL, cargo settled on the apron, "
          "score ~0, no success",
          bool(scene._finite()[0]) and bool(scene.settled()[0])
          and bool(scene.level()[0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 2+3. randomization readback ================================================
    yaws, xys, ks = [], [], []
    pad_ok, depot_ok, slots_ok = True, True, True
    for s in range(8):
        torch.manual_seed(20 + s)
        env.reset()
        _step(60)
        _refresh()
        yaws.append(yaw_of(scene.station.data.root_quat_w[0]))
        xys.append((scene.station.data.root_pos_w[0]
                    - scene.env_origins[0])[:2].tolist())
        ks.append(kk())
        px, py = c.pad_center
        for j in range(3):
            loc = station_local(scene.parcels[j].data.root_pos_w)[0]
            on_pad = (abs(float(loc[0]) - px) < 0.06 and abs(float(loc[1]) - py) < 0.06
                      and 0.0 < float(loc[2]) < 0.10)
            if j == ks[-1] - 1:
                pad_ok = pad_ok and on_pad
            else:
                d = float((scene.parcels[j].data.root_pos_w[0]
                           - scene.station.data.root_pos_w[0])[:2].norm())
                depot_ok = depot_ok and (not on_pad) and d > 1.0
        for j, (sx, sy) in enumerate(c.weight_slots):
            loc = station_local(scene.weights[j].data.root_pos_w)[0]
            slots_ok = slots_ok and abs(float(loc[0]) - sx) < 0.05 \
                and abs(float(loc[1]) - sy) < 0.05
    yspan = max(yaws) - min(yaws)
    xystd = float(np.std(np.asarray(xys), axis=0).mean())
    print(f"[smoke] readback: yaws={[f'{y:+.0f}' for y in yaws]} span={yspan:.0f} "
          f"xystd={xystd:.3f} ks={ks} pad_ok={pad_ok} depot_ok={depot_ok} "
          f"slots_ok={slots_ok}", flush=True)
    check("randomization A: station yaw spans > 90 deg, xy jitters, k varies",
          yspan > 90.0 and xystd > 0.008 and len(set(ks)) >= 2)
    check("randomization B: the shipment (parcel k-1) stands on the green pad, the "
          "other parcels park > 1 m away, weights rest near their apron slots",
          pad_ok and depot_ok and slots_ok)

    # ================= 4. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(240)
    _report("null")
    check("null policy: 240 idle steps -> score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 5. SEED strategy (carry the parcel to the destination, no weighing) ========
    reset_with_k((2, 3), 41)  # a heavy parcel keels the scale onto its stop
    drop_parcel_cradle(kk() - 1)
    _step(300)
    _report("seed-carry")
    t5 = float(scene.tilt()[0])
    s5 = float(scene.score()[0])
    check("SEED strategy: parcel delivered to the cradle with NO weighing — the "
          "scale keels far past tolerance (settled); latch pays 0.30, never more; "
          "no success",
          bool(scene.in_cradle()[0]) and t5 > 2.0 * tol
          and bool(scene.settled()[0]) and 0.29 <= s5 <= 0.31 and not succ())

    # ================= 6. under-ballast (one unit short) ==========================================
    reset_with_k((2, 3), 51)
    k6 = kk()
    for u in range(k6 - 1):
        drop_weight_pocket(u, u)
    drop_parcel_cradle(k6 - 1)
    _step(450)
    _report("under-ballast")
    in_rack6, _ = scene._weight_flags()
    t6 = float(scene.tilt()[0])
    s6 = float(scene.score()[0])
    cap6 = c.w_parcel + c.w_ballast * (k6 - 1) / k6
    check("under-ballast: k-1 weights + parcel — the one-unit shortfall reads far "
          "outside tolerance (settled), partial credit only, no success",
          int(in_rack6[0].sum()) == k6 - 1 and bool(scene.in_cradle()[0])
          and t6 > tol and bool(scene.settled()[0])
          and s6 <= cap6 + 0.01 and not succ())

    # ================= 7. over-ballast (one unit excess) ==========================================
    reset_with_k((1,), 61)
    drop_weight_pocket(0, 0)
    drop_weight_pocket(1, 1)
    drop_parcel_cradle(0)
    _step(450)
    _report("over-ballast")
    in_rack7, _ = scene._weight_flags()
    t7 = float(scene.tilt()[0])
    s7 = float(scene.score()[0])
    check("over-ballast: k+1 weights + parcel — one unit excess reads rack-down "
          "far outside tolerance; exact-count refuses; credit capped at 0.60; "
          "no success",
          int(in_rack7[0].sum()) == 2 and bool(scene.in_cradle()[0])
          and t7 < -tol and bool(scene.settled()[0])
          and s7 <= c.score_cap + 1e-3 and not succ())

    # ================= 8. mirrored placement (right masses, wrong receptacles) ====================
    reset_with_k((1,), 71)
    # the WEIGHT into the cradle ...
    rest_w = c.floor_z1 + c.weight_size / 2
    _write_body(scene.weights[0], beam_release([c.arm_r, 0.0, rest_w], 0.030),
                scene.beam.data.root_quat_w)
    _step(150)
    # ... and the PARCEL into the middle rack pocket (equal masses, equal radius)
    rest_p = c.floor_z1 + c.parcel_sizes[0] / 2
    _write_body(scene.parcels[0], beam_release([-c.arm_r, 0.0, rest_p], 0.012),
                scene.beam.data.root_quat_w)
    _step(450)
    _report("mirrored")
    in_rack8, _ = scene._weight_flags()
    s8 = float(scene.score()[0])
    check("mirrored placement: weight in the cradle, parcel in a rack pocket — the "
          "beam is LEVEL and settled, yet the rubric refuses (wrong objects in "
          "wrong receptacles); score stays ~0",
          bool(scene.level()[0]) and bool(scene.settled()[0])
          and int(in_rack8[0].sum()) == 0 and not bool(scene.in_cradle()[0])
          and s8 <= 0.05 and not succ())

    # ================= 9. unclean rider (everything else perfect) =================================
    reset_with_k((1, 2), 81)
    k9 = kk()
    # spare weight onto the BAR near the pivot FIRST (cleanliness is false from here
    # on, so success stays false through the whole construction); ~2.5 deg of bias
    rider_x = -0.045
    _write_body(scene.weights[3], beam_release([rider_x, 0.0,
                                                c.bar_hz + c.weight_size / 2], 0.010),
                scene.beam.data.root_quat_w)
    _step(150)
    for u in range(k9):
        drop_weight_pocket(u, u)
    drop_parcel_cradle(k9 - 1)
    _step(450)
    _report("unclean-rider")
    in_rack9, on_beam9 = scene._weight_flags()
    rider_on = bool(on_beam9[0, 3]) and not bool(in_rack9[0, 3])
    s9 = float(scene.score()[0])
    check("unclean rider: exact k in rack + parcel in cradle + beam LEVEL and "
          "settled — but a spare weight rides the bar: the cleanliness clause "
          "alone refuses",
          int(in_rack9[0].sum()) == k9 and bool(scene.in_cradle()[0])
          and bool(scene.level()[0]) and bool(scene.settled()[0]) and rider_on
          and s9 <= c.score_cap + 1e-3 and not succ())

    # ================= 10. delivery miss (ballast only) ===========================================
    reset_with_k((1, 2), 91)
    k10 = kk()
    for u in range(k10):
        drop_weight_pocket(u, u)
    _step(450)
    _report("delivery-miss")
    in_rack10, _ = scene._weight_flags()
    t10 = float(scene.tilt()[0])
    s10 = float(scene.score()[0])
    check("delivery miss: k weights seated, parcel left on the pad — the scale "
          "reads rack-down outside tolerance; ballast latch pays 0.30, never "
          "more; no success",
          int(in_rack10[0].sum()) == k10 and not bool(scene.in_cradle()[0])
          and t10 < -tol and bool(scene.settled()[0])
          and 0.29 <= s10 <= 0.31 and not succ())

    # ================= 11. latched credit =========================================================
    reset_with_k((1, 2), 101)
    k11 = kk()
    drop_parcel_cradle(k11 - 1)
    latched_p = bool(scene._parcel[0])
    # steal the shipment off the scale (construct), then ballast honestly
    _write_body(scene.parcels[k11 - 1],
                (scene.env_origins[0:1] + torch.tensor(
                    [0.9, 0.9, c.parcel_sizes[k11 - 1] / 2 + 0.002],
                    device=device)).expand(n, 3))
    _step(180)
    for u in range(k11):
        drop_weight_pocket(u, u)
    _step(300)
    _report("latch")
    s11 = float(scene.score()[0])
    check("latched credit: parcel delivered (latch pays) then STOLEN, then k "
          "weights seated — score holds at the 0.60 cap, success never fires",
          latched_p and not bool(scene.in_cradle()[0])
          and 0.59 <= s11 <= c.score_cap + 1e-3 and not succ())

    # ================= 12-14. audit, no-NaN, video ================================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.weigh_station")
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

"""Smoke / rubric-REJECTION battery for FireCribScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the latched
credit is monotone along a real trajectory). This battery proves the rubric REJECTS
wrong outcomes, and that the geometric claims the task rests on — the offcut cannot
span the air gap, the griddle's success height is reachable only through a two-layer
structure — are physically load-bearing. Every probe is CONSTRUCTED as a settled
state (teleport, real physics steps, judge) — instrumentation, never a solution: no
probe here reaches success().

Checks:
   1. settle/no-NaN      — seeded reset settles finite; five bars + griddle flat on
                           the ground, offcut off the hearth; score 0, no success;
   2. randomization      — two seeded resets: READBACK hearth yaw, hearth xy, and
                           split_0 xy all differ;
   3. slot permutation   — over 10 resets the offcut occupies >= 3 distinct scatter
                           slots and >= 5 distinct full permutations appear;
   4. null-policy        — 240 idle steps: score ~0, no success (everything stays put);
   5. OFFCUT CANNOT SPAN — a legal bottom pair is built at the NARROWEST legal gap and
                           the offcut is dropped across the middle of the gap: it falls
                           STRAIGHT THROUGH to the hearth surface ("too short to serve
                           as a crib member" is physics, not fiat);
   6. SEED-analog naive  — the griddle laid flat directly on the bare hearth plate
                           (the closest analog of the seed's "push the lid shut" end
                           state: a plate covering the appliance) -> score ~0, no
                           success;
   7. single-layer cheat — all FOUR splits laid flat side by side on the hearth and
                           the griddle rested on them: one layer, no crossings — the
                           griddle sits BELOW the success band -> no crib, no success;
   8. parallel-tower     — top pair stacked PARALLEL on the bottom pair (2x2 tower,
     cheat                 no crossings) with the griddle at the correct height: the
                           orthogonality clause refuses the crib -> no success;
   9. near-miss gap      — a full crossed crib + griddle built with the bottom pair
                           at 60 mm spacing (below the 78 mm window): everything else
                           perfect -> pair clause refuses, no crib credit, no success;
  10. near-miss spanner  — correct base + one correct spanner; the second spanner
                           shifted along its axis so it rests on only ONE bottom rail
                           (its end never reaches the other) -> no crib, no success;
  11. offcut restraint   — a FULL correct crib + griddle built with the offcut parked
                           ON the hearth plate: every structural clause holds, the
                           restraint clause alone refuses -> no success, score <= 0.75;
  12. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.close_grill_i8.smoke --headless
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
    from . import scene as task_scene
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene

_qmul, _qz = task_scene._qmul, task_scene._qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

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
    base, span1, crib = scene._structure_now()
    print(f"[smoke] {tag:18s} | base={bool(base[0])} span1={bool(span1[0])} "
          f"crib={bool(crib[0])} griddle={bool(scene.griddle_on_crib()[0])} "
          f"offcut_clear={bool(scene.offcut_clear()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.fire_crib")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.20, -0.80, 0.80)) + o),
                                tuple(np.array((0.30, -0.05, 0.08)) + o),
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

    def yaw_of(quat_row: torch.Tensor) -> float:
        return math.degrees(2.0 * math.atan2(float(quat_row[3]), float(quat_row[0])))

    def dyaw(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    def pad_world(local_xy, dz: float) -> torch.Tensor:
        """Hearth-frame xy + height above the hearth top -> world point (per-env)."""
        _refresh()
        pp = scene.pad.data.root_pos_w
        pq = scene.pad.data.root_quat_w
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0], loc[:, 1] = float(local_xy[0]), float(local_xy[1])
        pos = pp + quat_apply(pq, loc)
        pos[:, 2] = scene._pad_top_z() + dz
        return pos

    def pad_quat(yaw_local: float) -> torch.Tensor:
        _refresh()
        return _qmul(scene.pad.data.root_quat_w,
                     _qz(torch.full((n,), yaw_local, device=device)))

    def put(body, local_xy, dz_rest: float, yaw_local: float = 0.0,
            drop: float = 0.004) -> None:
        """CONSTRUCT: write the body just above its rest pose in the hearth frame
        (probe instrumentation; the caller settles when the arrangement is complete)."""
        _write_body(body, pad_world(local_xy, dz_rest + drop), pad_quat(yaw_local))

    z_bot = c.bar_w / 2
    z_top = 1.5 * c.bar_w
    z_gr = 2.0 * c.bar_w + c.plate_t / 2

    def build_pair(gap: float) -> None:
        """Bottom pair (splits 0, 1) along the hearth-local y at x = -/+ gap/2."""
        put(scene.bars["split_0"], (-gap / 2, 0.0), z_bot, math.pi / 2)
        put(scene.bars["split_1"], (+gap / 2, 0.0), z_bot, math.pi / 2)

    def build_crib(gap_bot: float, gap_top: float) -> None:
        """Full crossed crib: bottom pair + orthogonal top pair (splits 2, 3)."""
        build_pair(gap_bot)
        put(scene.bars["split_2"], (0.0, -gap_top / 2), z_top, 0.0)
        put(scene.bars["split_3"], (0.0, +gap_top / 2), z_top, 0.0)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    pos, _a, _v = scene._split_tensors()
    hz = pos[0, :, 2] - env.iscene.env_origins[0, 2]
    check("settle/no-NaN: layout settles finite; splits flat on the ground, offcut "
          "off the hearth; score 0, no success",
          bool(torch.isfinite(pos).all())
          and bool((hz > 0.005).all() and (hz < 0.03).all())
          and bool(scene.offcut_clear()[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (yaw_of(scene.pad.data.root_quat_w[0]),
                scene.pad.data.root_pos_w[0, :2].clone(),
                scene.bars["split_0"].data.root_pos_w[0, :2].clone())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_pp, a_s0 = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_pp, b_s0 = readback()
    d_yawv = dyaw(a_yaw, b_yaw)
    d_pp, d_s0 = float((a_pp - b_pp).norm()), float((a_s0 - b_s0).norm())
    print(f"[smoke] randomization deltas: hearth_yaw={d_yawv:.1f}deg "
          f"hearth_xy={d_pp * 1000:.1f}mm split0_xy={d_s0 * 1000:.1f}mm", flush=True)
    check("randomization-is-real: hearth yaw, hearth xy, split_0 xy readback differ",
          d_yawv > 2.0 and d_pp > 0.003 and d_s0 > 0.003)

    # ================= 3. slot permutation coverage ===============================================
    perms, offcut_slots = set(), set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        p = tuple(scene.slot_of[0].tolist())
        perms.add(p)
        offcut_slots.add(p[4])
    print(f"[smoke] over 10 resets: offcut slots {sorted(offcut_slots)}, "
          f"{len(perms)} distinct permutations", flush=True)
    check("slot permutation: offcut occupies >= 3 distinct scatter slots and >= 5 "
          "distinct permutations appear over 10 resets",
          len(offcut_slots) >= 3 and len(perms) >= 5)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. OFFCUT CANNOT SPAN (the identity claim is physics) ======================
    # Build a legal bottom pair at the NARROWEST legal gap, then drop the offcut from
    # a hover CENTRED across the air gap, axis perpendicular to the rails — the very
    # move that works for a long split. The 48 mm offcut vs the >= 56 mm daylight:
    # it must fall STRAIGHT THROUGH to the hearth surface.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    g5 = c.gap_min + 0.002
    build_pair(g5)
    _step(90)
    put(scene.bars["offcut"], (0.0, 0.0), z_top, 0.0, drop=0.012)
    _step(150)
    _report("offcut-span")
    off_dz = float(scene.bars["offcut"].data.root_pos_w[0, 2] - scene._pad_top_z()[0])
    print(f"[smoke] offcut dropped across the {g5 * 1000:.0f}mm pair: settled "
          f"dz={off_dz * 1000:.1f}mm (rail-top would be ~{2 * c.bar_w * 1000:.0f}mm)",
          flush=True)
    check("OFFCUT CANNOT SPAN: dropped across the narrowest legal gap it falls "
          "through to the hearth surface — never rests at rail height",
          off_dz < c.bar_w and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 6. negative: the SEED's naive analog =======================================
    # The seed's whole skill is "push the lid shut": end state = a plate covering the
    # appliance. CONSTRUCT the closest analog — the griddle laid flat directly on the
    # bare hearth plate. No structure, nothing carried: must earn ~0.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    put(scene.griddle, (0.0, 0.0), c.plate_t / 2, 0.0)
    _step(120)
    _report("seed-naive")
    check("negative (SEED analog): griddle laid flat directly on the bare hearth "
          "plate — below the success band, score ~0, no success",
          not bool(scene.griddle_on_crib()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.05)

    # ================= 7. negative: single-layer cheat ============================================
    # All FOUR splits laid flat side by side on the hearth (one layer, no crossings),
    # griddle rested on them. The griddle ends ~27 mm above the hearth — below the
    # 41 mm success band the two-layer structure exists to reach.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    for k_i, x in enumerate((-0.09, -0.03, 0.03, 0.09)):
        put(scene.bars[f"split_{k_i}"], (x, 0.0), z_bot, math.pi / 2)
    _step(60)
    put(scene.griddle, (0.0, 0.0), c.bar_w + c.plate_t / 2, 0.0)
    _step(120)
    _report("single-layer")
    g_dz = float(scene.griddle.data.root_pos_w[0, 2] - scene._pad_top_z()[0])
    print(f"[smoke] single-layer griddle dz={g_dz * 1000:.1f}mm "
          f"(success band {c.plate_z_lo * 1000:.0f}..{c.plate_z_hi * 1000:.0f}mm)",
          flush=True)
    check("negative (single layer): four splits flat side by side + griddle on them "
          "— griddle below the success band, no crib, no success, score <= 0.36",
          not bool(scene.griddle_on_crib()[0]) and not bool(scene._crib[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.36)
    _REC["on"] = False

    # ================= 8. negative: parallel-tower cheat ==========================================
    # Top pair stacked PARALLEL directly on the bottom pair (a 2x2 tower — right
    # heights, no crossings), griddle on top at the correct height. The orthogonal
    # crossing clauses must refuse the crib, so success must refuse too.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    build_pair(0.095)
    put(scene.bars["split_2"], (-0.095 / 2, 0.0), z_top, math.pi / 2)
    put(scene.bars["split_3"], (+0.095 / 2, 0.0), z_top, math.pi / 2)
    _step(60)
    put(scene.griddle, (0.0, 0.0), z_gr, 0.0)
    _step(150)
    _report("parallel-tower")
    check("negative (parallel tower): top pair stacked parallel on the bottom pair, "
          "griddle at the correct height — no crossings, no crib, no success",
          not bool(scene._crib[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.60)
    _REC["on"] = False

    # ================= 9. near-miss: full crib, gap below the window ==============================
    # A complete crossed crib + griddle, everything perfect EXCEPT the bottom pair
    # spacing: 60 mm, below the 78 mm window (no usable air gap). The pair clause
    # must refuse base/crib and success.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    build_crib(0.060, 0.095)
    _step(60)
    put(scene.griddle, (0.0, 0.0), z_gr, 0.0)
    _step(150)
    _report("narrow-gap")
    check("near-miss (gap): full crossed crib + griddle with the bottom pair at "
          "60 mm (< 78 mm window) — no base/crib credit, no success, score <= 0.15",
          not bool(scene._base[0]) and not bool(scene._crib[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.15)

    # ================= 10. near-miss: spanner reaches only one rail ===============================
    # Correct base + one correct spanner; the second spanner shifted 100 mm along its
    # own axis, so its end rests on ONE bottom rail and never reaches the other.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    build_pair(0.095)
    put(scene.bars["split_2"], (0.0, -0.095 / 2), z_top, 0.0)
    put(scene.bars["split_3"], (0.100, +0.095 / 2), z_top, 0.0)
    _step(150)
    _report("short-spanner")
    check("near-miss (spanner): second spanner shifted along its axis, resting on "
          "only ONE bottom rail — no crib credit, no success, score <= 0.56",
          bool(scene._span1[0]) and not bool(scene._crib[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.56)

    # ================= 11. negative: offcut restraint is load-bearing =============================
    # The offcut is parked ON the hearth plate FIRST, then a FULL correct crib +
    # griddle is built around it (so success() is never True at any judged instant).
    # Every structural clause holds; the restraint clause alone must refuse.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    put(scene.bars["offcut"], (0.095, 0.095), z_bot, 0.0)
    _step(30)
    build_crib(0.095, 0.095)
    _step(60)
    put(scene.griddle, (0.0, 0.0), z_gr, 0.0)
    _step(150)
    _report("offcut-on-pad")
    _b, _s, crib_now = scene._structure_now()
    check("negative (restraint): FULL correct crib + griddle with the offcut parked "
          "on the hearth plate — structure perfect, restraint refuses, no success, "
          "score <= 0.75",
          bool(crib_now[0]) and bool(scene.griddle_on_crib()[0])
          and not bool(scene.offcut_clear()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.75)
    _REC["on"] = False

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.fire_crib")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    check("video: frames captured and saved to frames.npz", len(_REC["frames"]) > 10)

    n_pass = sum(ok for _n, ok in checks)
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
    main()

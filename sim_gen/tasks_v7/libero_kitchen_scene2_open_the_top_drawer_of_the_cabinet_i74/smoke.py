"""Smoke / rubric-REJECTION battery for FoldCollarScene — NullRobot, teleported
probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the
latched credit is monotone along a real trajectory). This battery proves the rubric
REJECTS wrong outcomes — rigid transport without folding, folding without transport,
wrapping the decoy, under-folding, one-wing folds, a knocked-over screen — and that
the mechanism claims the task rests on (hinges hold their angle) are true. Every
probe is CONSTRUCTED (joint-consistent teleport, real physics steps, judge) —
instrumentation, never a solution. success() is monitored at EVERY step and must
never turn True anywhere in the rejection part of the battery (the audit is itself
a check); the one constructed-success sanity probe runs AFTER the audit closes.

Checks:
   1. settle/no-NaN     — seeded reset settles finite; collar upright in the zigzag
                          window, far from both columns; score 0, no success;
   2. randomization     — two seeded resets: READBACK collar xy+yaw, both wing fold
                          angles and the blue column xy all differ;
   3. slot swap         — over 10 resets the blue column occupies BOTH slots;
   4. null-policy       — 240 idle steps: score ~0, no success;
   5. RIGID TRANSPORT   — the seed-style outcome (one rigid displacement, no
                          articulation): the still-open collar parked flush against
                          the blue column, settled -> column NOT inside, no success,
                          score <= approach credit (transport alone earns 0.15);
   6. MECHANISM hold    — wings teleported to 60 deg and left alone for 2.5 s: the
                          damped hinges + ground friction hold the fold (the "stays
                          where you leave it" claim of describe());
   7. wrong object      — the collar folded shut around the RED decoy: folds hold,
                          the red column is inside the pocket, yet no success and
                          no fold credit (blue-only rubric);
   8. near-miss gap     — wrapped around the BLUE column but under-folded (95 deg):
                          column inside, both wings past fold_min, latches fire —
                          yet gap > gap_max -> no success, score <= 0.70;
   9. near-miss one-wing— one wing folded 112 deg, the other left open: no success,
                          no both-wings fold latch, score <= 0.40;
  10. fold-in-place     — the collar folded shut at its SPAWN, far from the columns:
                          closed gap, yet score 0 (articulation without transport
                          earns nothing), no success;
  11. tipped screen     — the folded collar knocked flat on the ground: upright
                          clause refuses, no success;
  12. rejection audit   — success() was never True at any step of checks 1-11;
  13. success sanity    — the exact goal state constructed around the BLUE column
                          settles into success() True and score 1.0 (the predicate
                          is satisfiable; audit closed before this probe);
  14. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_kitchen_scene2_open_the_top_drawer_of_the_cabinet_i74.smoke --headless
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

_qz, _qmul, _qapply = task_scene._qz, task_scene._qmul, task_scene._qapply

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}
_AUD = {"on": False, "hits": 0}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        if _AUD["on"] and bool(env.scene.success()[0]):
            _AUD["hits"] += 1
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
    fl, fr = scene._folds()
    print(f"[smoke] {tag:18s} | folds=({float(torch.rad2deg(fl[0])):+6.1f},"
          f"{float(torch.rad2deg(fr[0])):+6.1f})deg "
          f"gap={float(scene._gap()[0]) * 1000:6.1f}mm "
          f"inside={bool(scene._blue_inside()[0])} "
          f"upright={bool(scene._upright_standing()[0])} "
          f"settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.fold_collar")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.25, -0.60, 0.55)) + o),
                                tuple(np.array((0.35, 0.02, 0.08)) + o),
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

    def yaw_of(quat_row: torch.Tensor) -> float:
        return math.degrees(2.0 * math.atan2(float(quat_row[3]), float(quat_row[0])))

    def dyaw(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    def folds_deg() -> tuple[float, float]:
        _refresh()
        fl, fr = scene._folds()
        return float(torch.rad2deg(fl[0])), float(torch.rad2deg(fr[0]))

    def write_collar(base_xy_w: torch.Tensor, yaw_deg: float, fl_deg: float,
                     fr_deg: float) -> None:
        """Joint-consistent collar pose write: base at `base_xy_w` (world xy, (N,2)),
        heading `yaw_deg`, wings at the given fold angles, 2 mm hover."""
        qb = _qz(torch.full((n,), math.radians(yaw_deg), device=device))
        pp = torch.zeros(n, 3, device=device)
        pp[:, :2] = base_xy_w
        pp[:, 2] = scene.env_origins[:, 2] + 0.002
        _write_body(scene.base, pp, qb)
        for body, side, fdeg in ((scene.wing_l, -1.0, fl_deg),
                                 (scene.wing_r, 1.0, fr_deg)):
            hinge = torch.zeros(n, 3, device=device)
            hinge[:, 0] = side * c.panel_w / 2
            fold = torch.full((n,), math.radians(fdeg), device=device)
            _write_body(body, pp + _qapply(qb, hinge), _qmul(qb, _qz(side * fold)))

    def wrap_xy(column) -> torch.Tensor:
        """Base xy that parks the pocket at the given column (pocket faces +y)."""
        _refresh()
        d_park = c.col_r + c.panel_t / 2 + 0.008
        xy = column.data.root_pos_w[:, :2].clone()
        xy[:, 1] -= d_park
        return xy

    def red_inside() -> bool:
        _refresh()
        f_l, h_l, h_r, f_r = scene._collar_points()
        verts = torch.stack([f_l, h_l, h_r, f_r], dim=1)
        return bool(scene._quad_contains(verts, scene.red.data.root_pos_w[:, :2])[0])

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _AUD["on"] = True
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    fl0, fr0 = folds_deg()
    pos_fin = torch.isfinite(scene.base.data.root_pos_w).all() \
        and torch.isfinite(scene.wing_l.data.root_pos_w).all() \
        and torch.isfinite(scene.wing_r.data.root_pos_w).all()
    check("settle/no-NaN: layout settles finite; collar upright in the zigzag "
          "window, far from the columns; score 0, no success",
          bool(pos_fin) and 2.0 < fl0 < 34.0 and 2.0 < fr0 < 34.0
          and bool(scene._upright_standing()[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        fl, fr = scene._folds()
        return (scene.base.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.base.data.root_quat_w[0]),
                float(torch.rad2deg(fl[0])), float(torch.rad2deg(fr[0])),
                scene.blue.data.root_pos_w[0, :2].clone())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_bp, a_by, a_fl, a_fr, a_cp = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_bp, b_by, b_fl, b_fr, b_cp = readback()
    d_bp = float((a_bp - b_bp).norm())
    d_by = dyaw(a_by, b_by)
    d_f = max(abs(a_fl - b_fl), abs(a_fr - b_fr))
    d_cp = float((a_cp - b_cp).norm())
    print(f"[smoke] randomization deltas: collar_xy={d_bp * 1000:.1f}mm "
          f"collar_yaw={d_by:.1f}deg folds={d_f:.1f}deg "
          f"blue_xy={d_cp * 1000:.1f}mm", flush=True)
    check("randomization-is-real: collar xy+yaw, wing fold angles and blue column "
          "xy readback differ between seeds",
          d_bp > 0.003 and d_by > 2.0 and d_f > 1.0 and d_cp > 0.003)

    # ================= 3. slot swap ===============================================================
    slots = set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        slots.add("A" if float(scene.blue_slot[0]) > 0 else "B")
    print(f"[smoke] over 10 resets: blue column slots {sorted(slots)}", flush=True)
    check("slot swap: the blue column occupies BOTH slots over 10 resets",
          slots == {"A", "B"})

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. negative: RIGID TRANSPORT (the seed-style plan) =========================
    # The seed task's whole plan is one rigid displacement produced by a single
    # pulling contact. The analogous outcome here — the collar carried flush against
    # the blue column WITHOUT using its hinges — earns only the approach credit:
    # with the wings near-straight the free edges are ~3 leaf-widths apart, the
    # column is not inside the pocket, and success refuses.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    write_collar(wrap_xy(scene.blue), 0.0, 15.0, 15.0)
    _step(150)
    _report("rigid-press")
    _REC["on"] = False
    d_col = float((scene.base.data.root_pos_w[0, :2]
                   - scene.blue.data.root_pos_w[0, :2]).norm())
    check("negative (RIGID transport, the seed's plan): open collar parked flush "
          "against the blue column — not inside, no success, score <= approach",
          d_col < 0.08 and not bool(scene._blue_inside()[0])
          and not bool(scene.success()[0])
          and float(scene.score()[0]) <= c.w_appr + 1e-5)

    # ================= 6. MECHANISM: the hinges hold where they are left ==========================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _refresh()
    write_collar(scene.base.data.root_pos_w[:, :2].clone(), 30.0, 60.0, 60.0)
    _step(300)  # 2.5 s hands-off
    _report("hinge-hold")
    fl6, fr6 = folds_deg()
    check("MECHANISM (hinge hold): wings left at 60 deg keep their fold within "
          "12 deg over 2.5 s hands-off (damped hinges + ground friction)",
          abs(fl6 - 60.0) < 12.0 and abs(fr6 - 60.0) < 12.0
          and bool(scene._upright_standing()[0]))

    # ================= 7. negative: wrong object (the RED decoy wrapped) ==========================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    write_collar(wrap_xy(scene.red), 0.0, 114.0, 114.0)
    _step(150)
    _report("wrap-decoy")
    _REC["on"] = False
    fl7, fr7 = folds_deg()
    check("negative (wrong object): the collar folded shut around the RED decoy — "
          "folds hold and the red column sits in the pocket, yet no success and "
          "no fold credit (blue-only rubric)",
          fl7 > 95.0 and fr7 > 95.0 and red_inside()
          and float(scene._gap()[0]) < c.gap_max + 0.02
          and not bool(scene._blue_inside()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= c.w_appr + 1e-5)

    # ================= 8. near-miss: under-folded, gap above the capture bound ====================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    write_collar(wrap_xy(scene.blue), 0.0, 95.0, 95.0)
    _step(150)
    _report("under-fold")
    _REC["on"] = False
    fl8, fr8 = folds_deg()
    gap8 = float(scene._gap()[0])
    print(f"[smoke] under-fold: gap={gap8 * 1000:.1f}mm "
          f"(bound {c.gap_max * 1000:.1f}mm, column diameter "
          f"{c.col_r * 2000:.0f}mm)", flush=True)
    check("near-miss (gap): wrapped around the BLUE column at 95 deg — inside, "
          "both wings past fold_min, latches fire, yet the gap stays above the "
          "capture bound -> no success, score <= 0.70",
          fl8 > c.fold_min_deg and fr8 > c.fold_min_deg
          and bool(scene._blue_inside()[0]) and gap8 > c.gap_max
          and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.70 + 1e-5)

    # ================= 9. near-miss: one wing only ================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    write_collar(wrap_xy(scene.blue), 0.0, 112.0, 12.0)
    _step(150)
    _report("one-wing")
    fl9, fr9 = folds_deg()
    check("near-miss (one wing): left wing folded 112 deg, right left open — no "
          "both-wings latch, gap huge, no success, score <= 0.40",
          fl9 > c.fold_min_deg and fr9 < 40.0
          and float(scene._gap()[0]) > c.gap_max
          and not bool(scene.success()[0])
          and float(scene.score()[0]) <= c.w_appr + c.w_half + 1e-5)

    # ================= 10. negative: folding WITHOUT transport ====================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _refresh()
    write_collar(scene.base.data.root_pos_w[:, :2].clone(), 0.0, 114.0, 114.0)
    _step(150)
    _report("fold-in-place")
    fl10, fr10 = folds_deg()
    check("negative (fold in place): the collar folded shut at its SPAWN, far from "
          "the columns — closed gap yet score 0, no success (articulation without "
          "transport earns nothing)",
          fl10 > 95.0 and fr10 > 95.0
          and float(scene._gap()[0]) < c.gap_max + 0.02
          and not bool(scene._blue_inside()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.01)

    # ================= 11. negative: tipped screen ================================================
    # The folded collar knocked over flat (rigid rotation of the whole chain about
    # the base's bottom edge; joints stay consistent under a common rigid motion),
    # away from the columns. Whatever pose the collapse settles into, it is not
    # upright — a knocked-over screen never counts.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    q_tip = torch.zeros(n, 4, device=device)
    q_tip[:, 0] = math.cos(math.pi / 4)
    q_tip[:, 1] = math.sin(math.pi / 4)  # +90 deg about world x: +y -> +z
    pivot = torch.zeros(n, 3, device=device)
    pivot[:, 0], pivot[:, 1] = 0.15, -0.25
    pivot[:, 2] = scene.env_origins[:, 2] + c.panel_t / 2 + 0.002
    pivot[:, 0:2] += scene.env_origins[:, 0:2]
    qb = _qz(torch.zeros(n, device=device))
    _write_body(scene.base, pivot, _qmul(q_tip, qb))
    for body, side, fdeg in ((scene.wing_l, -1.0, 114.0), (scene.wing_r, 1.0, 114.0)):
        hinge = torch.zeros(n, 3, device=device)
        hinge[:, 0] = side * c.panel_w / 2
        fold = torch.full((n,), math.radians(fdeg), device=device)
        _write_body(body, pivot + _qapply(_qmul(q_tip, qb), hinge),
                    _qmul(_qmul(q_tip, qb), _qz(side * fold)))
    _step(300)  # let the knocked-over chain collapse and settle
    _report("tipped")
    tip_fin = torch.isfinite(scene.base.data.root_pos_w).all() \
        and torch.isfinite(scene.wing_l.data.root_pos_w).all() \
        and torch.isfinite(scene.wing_r.data.root_pos_w).all()
    check("negative (tipped): the folded screen knocked flat settles finite but "
          "not upright — no success",
          bool(tip_fin) and not bool(scene._upright_standing()[0])
          and not bool(scene.success()[0]))

    # ================= 12. rejection audit ========================================================
    check("rejection audit: success() was never True at any step of checks 1-11",
          _AUD["hits"] == 0)
    _AUD["on"] = False

    # ================= 13. constructed success sanity (audit closed) ==============================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    write_collar(wrap_xy(scene.blue), 0.0, 113.0, 113.0)
    _step(240)
    _report("success-sanity")
    _REC["on"] = False
    check("success sanity: the goal state constructed around the BLUE column "
          "settles into success() True and score 1.0 (predicate satisfiable)",
          bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 1.0) < 1e-5)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.fold_collar")
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

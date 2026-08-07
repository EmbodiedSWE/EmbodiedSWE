"""Smoke / rubric-REJECTION battery for TrayPackScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a real packed tray and the latched
credit climbs monotonically along a real push trajectory). This battery proves the
rubric REJECTS wrong outcomes, and that the physical claims the task rests on — the
walls confine a seated block, a mid-floor blocker makes the rest of the pack
IMPOSSIBLE, the floor plate separates in-tray from on-the-ground — are load-bearing.
Every probe is CONSTRUCTED as a settled state (teleport, real physics steps, judge) —
instrumentation, never a solution.

Checks:
   1. settle/no-NaN     — seeded reset settles finite: slab STANDING inside the tray
                          (not flat, scores nothing), bar+brick flat on the ground
                          outside; nothing seated; score 0, no success;
   2. randomization     — two seeded resets: READBACK tray yaw, tray xy, slab
                          tray-local spot, bar world xy all differ;
   3. slab coverage     — over 10 resets the slab's standing spot spans > 12 mm in
                          both tray axes and its heading takes >= 4 distinct 30-deg
                          bins (the blocker really moves);
   4. null-policy       — 240 idle steps: score ~0, no success (standing slab is
                          statically stable, nothing seats itself);
   5. WALL INTERLOCK    — a seated bar is shoved horizontally at 3x its weight for
                          1.5 s toward the near wall and then the far wall: it stays
                          seated inside (containment is physics, not fiat);
   6. SEED STRATEGY     — the seed suite's plan (each object to its own free-space
                          goal pose): all three blocks CONSTRUCTED flat in a neat
                          row on the ground beside the tray -> no success, score ~0;
   7. wrong place       — the CORRECT tight tiling, built on the GROUND beside the
                          tray: arrangement alone is not the task -> score ~0;
   8. stacking cheat    — StackCube's move inside the tray: slab seated mid-floor,
                          brick stacked ON TOP of it (flat, inside, 40 mm too high)
                          -> only the slab scores, no success, score <= 0.30;
   9. perched cheat     — brick laid flat on TOP of the STANDING slab (flat AND
                          inside the footprint, but at rim height) -> z gate
                          refuses, score ~0;
  10. near-miss         — bar and brick seated correctly, slab still STANDING on the
                          floor between them (inside, settled, wrong orientation)
                          -> no success, score capped at 0.50;
  11. JOINT CONSTRAINT  — slab seated DEAD CENTRE, then the bar dropped onto its
                          wall lane: rigid non-overlap leaves no 40 mm-wide flat
                          lane, the bar CANNOT seat -> the packing constraint is
                          physics ("a block placed thoughtlessly blocks the rest");
  12. mirror accepted   — the MIRRORED tiling (bar north, slab SE, brick SW),
                          CONSTRUCTED and settled -> success: the rubric judges the
                          arrangement class, not one hard-coded layout;
  13. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.native_tasks_i37.smoke --headless
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

_qmul, _qz, _qx = task_scene._qmul, task_scene._qz, task_scene._qx

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


def _push(body, force_w: torch.Tensor, steps: int) -> None:
    """Apply a world-frame force at the body's CoM for `steps` steps, then clear."""
    n = _ENV.num_envs
    zero = torch.zeros(n, 1, 3, device=_ENV.device)
    f = force_w.view(1, 1, 3).expand(n, 1, 3).contiguous()
    for _ in range(steps):
        body.set_external_force_and_torque(f, zero, env_ids=_all_ids(), is_global=True)
        _step(1)
    body.set_external_force_and_torque(zero, zero, env_ids=_all_ids())


def _qconj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[:, :1], -q[:, 1:]], dim=-1)


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    s = scene.seated()[0]
    lt = scene._seat[0]
    seat_s = "".join("Y" if bool(s[j]) else "." for j in range(3))
    lat_s = "".join("Y" if bool(lt[j]) else "." for j in range(3))
    print(f"[smoke] {tag:18s} | seated[bar,slab,brick]={seat_s} latch={lat_s} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tray_pack")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.20, -0.75, 0.80)) + o),
                                tuple(np.array((0.40, 0.02, 0.06)) + o),
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

    def tray_world(loc_xyz) -> torch.Tensor:
        """Tray-local point -> world (per-env)."""
        _refresh()
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.tray.data.root_pos_w + quat_apply(scene.tray.data.root_quat_w, loc)

    def tray_quat(yaw_deg: float = 0.0, standing: bool = False) -> torch.Tensor:
        """Tray-aligned block orientation, optional extra yaw, optional on-side."""
        _refresh()
        q = _qmul(scene.tray.data.root_quat_w,
                  _qz(torch.full((n,), math.radians(yaw_deg), device=device)))
        if standing:
            q = _qmul(q, _qx(torch.full((n,), math.pi / 2, device=device)))
        return q

    def seat_block(name: str, x: float, y: float, yaw_deg: float = 0.0) -> None:
        """CONSTRUCT `name` flat just above the tray floor at tray-local (x, y)."""
        _write_body(scene.blocks[name],
                    tray_world((x, y, c.z_floor + 0.020 + 0.003)), tray_quat(yaw_deg))

    def ground_world(x: float, y: float) -> torch.Tensor:
        p = env.iscene.env_origins.clone()
        p[:, 0] += x
        p[:, 1] += y
        p[:, 2] += 0.023
        return p

    def slab_local_xy() -> torch.Tensor:
        _refresh()
        return scene._tray_local(scene.blocks["slab"].data.root_pos_w)[0, :2].clone()

    def slab_psi() -> float:
        """Slab heading about the tray z axis (works flat or standing)."""
        _refresh()
        qr = _qmul(_qconj(scene.tray.data.root_quat_w),
                   scene.blocks["slab"].data.root_quat_w)
        return math.degrees(2.0 * math.atan2(float(qr[0, 3]), float(qr[0, 0]))) % 360.0

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    pos, _q, _v, _w = scene._block_tensors()
    slab_loc = scene._tray_local(scene.blocks["slab"].data.root_pos_w)[0]
    check("settle/no-NaN: layout settles finite; slab STANDING inside the tray "
          "(high, not flat), bar+brick flat on the ground outside; nothing seated; "
          "score 0, no success",
          bool(torch.isfinite(pos).all())
          and float(slab_loc[2]) > 0.050 and not bool(scene.flat()[0, 1])
          and float(slab_loc[:2].abs().max()) < c.inner_x / 2
          and int(scene.seated()[0].sum()) == 0
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (yaw_of(scene.tray.data.root_quat_w[0]),
                scene.tray.data.root_pos_w[0, :2].clone(),
                slab_local_xy(),
                scene.blocks["bar"].data.root_pos_w[0, :2].clone())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_tp, a_sl, a_bp = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_tp, b_sl, b_bp = readback()
    d_yawv = dyaw(a_yaw, b_yaw)
    d_tp = float((a_tp - b_tp).norm())
    d_sl = float((a_sl - b_sl).norm())
    d_bp = float((a_bp - b_bp).norm())
    print(f"[smoke] randomization deltas: tray_yaw={d_yawv:.1f}deg "
          f"tray_xy={d_tp * 1000:.1f}mm slab_spot={d_sl * 1000:.1f}mm "
          f"bar_xy={d_bp * 1000:.1f}mm", flush=True)
    check("randomization-is-real: tray yaw, tray xy, slab tray-local spot, bar xy "
          "readback all differ across seeds",
          d_yawv > 2.0 and d_tp > 0.003 and d_sl > 0.002 and d_bp > 0.003)

    # ================= 3. slab standing-spot / heading coverage ===================================
    xs, ys, bins = [], [], set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        xy = slab_local_xy()
        xs.append(float(xy[0]))
        ys.append(float(xy[1]))
        bins.add(int(slab_psi() // 30.0))
    rx, ry = max(xs) - min(xs), max(ys) - min(ys)
    print(f"[smoke] slab coverage over 10 resets: x-range={rx * 1000:.1f}mm "
          f"y-range={ry * 1000:.1f}mm heading-bins={sorted(bins)}", flush=True)
    check("slab coverage: standing spot spans > 12 mm in both tray axes and the "
          "heading takes >= 4 distinct 30-deg bins over 10 resets",
          rx > 0.012 and ry > 0.012 and len(bins) >= 4)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success (the standing "
          "slab is statically stable and scores nothing)",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. WALL INTERLOCK: a seated bar cannot be shoved out =======================
    # Containment is physics: seat the bar on its wall lane, then shove it at 3x its
    # own weight for 1.5 s toward the near wall and again toward the far wall. The
    # 50 mm walls must keep it seated inside both times.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    seat_block("bar", 0.0, -0.046)
    _step(90)
    _refresh()
    assert bool(scene.seated()[0, 0]), "probe setup: bar must seat on its wall lane"
    f_mag = 3.0 * c.bar_mass * 9.81
    for sgn in (-1.0, 1.0):  # toward the near (-y) wall, then across to the far (+y)
        d = quat_apply(scene.tray.data.root_quat_w,
                       torch.tensor([0.0, sgn, 0.0], device=device).expand(n, 3))[0]
        for _ in range(9):  # 9 x 20 = 180 steps = 1.5 s of shoving
            _push(scene.blocks["bar"], f_mag * d, 20)
        _step(60)
        _refresh()
    _step(60)
    _report("wall-interlock")
    bar_loc = scene._tray_local(scene.blocks["bar"].data.root_pos_w)[0]
    print(f"[smoke] after shoves: bar tray-local=({bar_loc[0] * 1000:.1f}, "
          f"{bar_loc[1] * 1000:.1f}, {bar_loc[2] * 1000:.1f})mm", flush=True)
    check("WALL INTERLOCK: 3x-weight shoves toward both walls for 1.5 s each leave "
          "the bar seated inside the tray",
          bool(scene.seated()[0, 0]) and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 6. negative: the SEED'S OWN STRATEGY =======================================
    # The seed suite moves each object to its own INDEPENDENT free-space goal pose.
    # CONSTRUCT that kind of end state: all three blocks flat, well separated, in a
    # neat row on the open ground beside the tray. Nothing is in the tray -> reject.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    for j, nm in enumerate(("bar", "slab", "brick")):
        _write_body(scene.blocks[nm], ground_world(0.75, -0.30 + 0.15 * j), None)
    _step(120)
    _report("seed-strategy")
    check("negative (SEED strategy): three blocks flat at tidy separate poses on "
          "the ground beside the tray — nothing seated, NO success, score ~0",
          int(scene.seated()[0].sum()) == 0 and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.05)
    _REC["on"] = False

    # ================= 7. negative: right arrangement, wrong place ================================
    # The CORRECT tight tiling (bar on the south lane, slab NW, brick NE rotated),
    # built flat on the GROUND beside the tray. Arrangement alone is not the task.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    ax, ay = 0.78, 0.10
    _write_body(scene.blocks["bar"], ground_world(ax + 0.0, ay - 0.046), None)
    _write_body(scene.blocks["slab"], ground_world(ax - 0.026, ay + 0.026), None)
    _write_body(scene.blocks["brick"], ground_world(ax + 0.046, ay + 0.026),
                _qz(torch.full((n,), math.pi / 2, device=device)))
    _step(120)
    _report("wrong-place")
    check("negative (wrong place): the exact tight tiling built on the ground "
          "beside the tray — no seated, no success, score ~0",
          int(scene.seated()[0].sum()) == 0 and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.05)

    # ================= 8. negative: stacking cheat (StackCube's move) =============================
    # Slab seated mid-floor, brick stacked ON TOP of it: the brick is flat, inside
    # the footprint, settled — and 40 mm too high. Only the slab may score.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    seat_block("slab", 0.0, 0.0)
    _step(60)
    _write_body(scene.blocks["brick"],
                tray_world((0.0, 0.0, c.z_floor + 0.040 + 0.020 + 0.003)), tray_quat())
    _step(90)
    _report("stacking-cheat")
    check("negative (stacking cheat): brick flat ON TOP of the seated slab inside "
          "the tray — z gate refuses the brick, no success, score <= 0.30",
          bool(scene.seated()[0, 1]) and not bool(scene.seated()[0, 2])
          and bool(scene.flat()[0, 2])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.30)
    _REC["on"] = False

    # ================= 9. negative: perched on the STANDING slab ==================================
    # Brick laid flat on the standing slab's narrow top face: flat AND inside the
    # interior footprint, but at rim height. Nothing may score.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    psi = slab_psi()
    p = scene.blocks["slab"].data.root_pos_w.clone()
    p[:, 2] += 0.040 + 0.020 + 0.002  # slab half-length + brick half-height
    _write_body(scene.blocks["brick"], p, tray_quat(psi))
    _step(90)
    _report("perched")
    check("negative (perched): brick flat on TOP of the standing slab — high above "
          "the floor, nothing seated, no success, score ~0",
          int(scene.seated()[0].sum()) == 0 and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.05)

    # ================= 10. near-miss: two packed, the blocker still standing ======================
    # Bar and brick seated correctly; the slab stands on the floor BETWEEN them —
    # inside, settled, centimetres from done, wrong orientation. Cap at 0.50.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _write_body(scene.blocks["slab"],
                tray_world((0.020, 0.030, c.z_floor + 0.040 + 0.003)),
                tray_quat(0.0, standing=True))
    _step(60)
    seat_block("bar", 0.0, -0.046)
    seat_block("brick", -0.046, 0.026, 90.0)
    _step(120)
    _report("near-miss")
    check("near-miss: bar+brick seated, slab still STANDING on the floor between "
          "them — no success, score capped at 0.50",
          bool(scene.seated()[0, 0]) and bool(scene.seated()[0, 2])
          and not bool(scene.seated()[0, 1])
          and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.50) < 0.01)
    _REC["on"] = False

    # ================= 11. JOINT CONSTRAINT: a mid-floor blocker blocks the pack ==================
    # Seat the slab DEAD CENTRE, then drop the bar onto its wall lane. The slab
    # leaves no 40 mm-wide flat lane anywhere (27.5 mm strips at best), so rigid
    # non-overlap must keep the bar from ever seating: the coupling between
    # placements is physics, not narrative.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    seat_block("slab", 0.0, 0.0)
    _step(60)
    assert bool(scene.seated()[0, 1]), "probe setup: slab must seat mid-floor"
    _write_body(scene.blocks["bar"],
                tray_world((0.0, -0.046, c.z_floor + 0.020 + 0.030)), tray_quat())
    _step(180)
    _report("joint-constraint")
    check("JOINT CONSTRAINT: with the slab seated mid-floor, the bar dropped onto "
          "its lane can never seat (no 40 mm lane exists) — no bar latch, no "
          "success, score <= 0.30",
          not bool(scene._seat[0, 0]) and not bool(scene.seated()[0, 0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.30)
    _REC["on"] = False

    # ================= 12. mirror arrangement is accepted =========================================
    # The rubric must judge the arrangement CLASS, not one hard-coded layout:
    # CONSTRUCT the mirrored tiling (bar on the NORTH lane, slab SE, brick SW
    # rotated) and settle. This is the battery's positive control.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    seat_block("bar", 0.0, 0.046)
    seat_block("slab", 0.026, -0.026)
    seat_block("brick", -0.046, -0.026, 90.0)
    _step(150)
    _report("mirror-pack")
    check("mirror arrangement accepted: bar north / slab SE / brick SW settles to "
          "success with score 1.0",
          bool(scene.success()[0]) and float(scene.score()[0]) >= 0.99)
    _REC["on"] = False

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.tray_pack")
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

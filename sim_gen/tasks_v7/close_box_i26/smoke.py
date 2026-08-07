"""Smoke / rubric-REJECTION battery for TrapCrateScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the latched
credit is monotone along a real trajectory). This battery proves the rubric REJECTS
wrong outcomes, and that the physical claims the task rests on — the walls trap the
block, sliding transports the trap, lifting frees it — are load-bearing. Every probe
is CONSTRUCTED (teleport, real physics steps, judge) — instrumentation, never a
solution: success() is monitored at EVERY step and must never turn True anywhere in
the battery (the audit is itself a check).

Checks:
   1. settle/no-NaN      — seeded reset settles finite; crate UPRIGHT, nothing
                           trapped; score 0, no success;
   2. randomization      — two seeded resets: READBACK crate xy+yaw, pen xy+yaw and
                           red-block xy all differ;
   3. slot swap          — over 10 resets the red block occupies BOTH scatter slots;
   4. null-policy        — 240 idle steps: score ~0, no success;
   5. SEED STRATEGY      — the seed family's naive plan (get the thing into the box /
                           box to the goal WITHOUT closing anything): red block placed
                           INSIDE the UPRIGHT crate, crate centred on the pen, settled
                           -> no success, score ~0 (an open box delivers nothing);
   6. wrong object       — the BLUE block trapped and delivered instead -> no success,
                           score ~0;
   7. near-miss capture  — crate inverted ON the pen, red block on the floor just
                           OUTSIDE a wall -> not trapped, no success;
   8. rim-perch          — crate dropped half-over the block so the RIM lands ON it:
                           crate perched/tilted, not floor-seated -> not capped;
   9. roof-park          — red block resting ON TOP of the properly inverted crate on
                           the pen -> the capture height window rejects it;
  10. restraint          — a CORRECT trap delivered to the pen, but the blue block
                           sits on the pen square (placed first) -> no success,
                           score <= 0.75: the distractor clause is load-bearing;
  11. near-miss delivery — correct trap seated 85 mm from the pen centre (window is
                           50 mm/axis) -> no success, score <= 0.75;
  12. mechanism: slide   — pushing the capped crate horizontally (~3.5 N at the CoM)
                           SLIDES it >= 60 mm and the trapped block is DRAGGED ALONG,
                           still inside the walls (containment transports);
  13. mechanism: lift    — teleporting the crate away (a carry) leaves the block
                           BEHIND (< 20 mm displacement): the trap only holds on the
                           floor, so a carried crate delivers nothing;
  14. latched credit     — after 13 the live trap is gone but the latched `trapped`
                           credit survives: trapped() False, score still >= 0.30;
  15. tipped crate       — crate lying on its SIDE next to the block -> not capped,
                           no success;
  16. settle gate        — the exact success pose but the crate still sliding at
                           ~0.45 m/s -> success refuses while anything moves (probe
                           dismantled before it can settle into a real success);
  17. rejection audit    — success() was never True at any step of this battery;
  18. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.close_box_i26.smoke --headless
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

_qx, _qz, _qapply = task_scene._qx, task_scene._qz, task_scene._qapply

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


def _write_body(body, pos_w: torch.Tensor, quat: torch.Tensor | None = None,
                lin_vel: torch.Tensor | None = None) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat
    if lin_vel is not None:
        st[:, 7:10] = lin_vel
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    cp = (scene.crate.data.root_pos_w - scene.env_origins)[0]
    print(f"[smoke] {tag:18s} | crate=({float(cp[0]):+.3f},{float(cp[1]):+.3f},"
          f"{float(cp[2]):+.3f}) up_z={float(scene.crate_up_z()[0]):+.3f} "
          f"capped={bool(scene.capped()[0])} trapped={bool(scene.trapped()[0])} "
          f"in_pen={bool(scene.in_pen()[0])} "
          f"blue_clear={bool(scene.distractor_clear()[0])} "
          f"settled={bool(scene.settled()[0])} score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.trap_crate")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    hover_z = c.height / 2 + c.block + 0.018

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.35, -0.60, 0.55)) + o),
                                tuple(np.array((0.38, 0.00, 0.05)) + o),
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

    def floor_pt(xy_w: torch.Tensor, z: float) -> torch.Tensor:
        p = torch.zeros(n, 3, device=device)
        p[:, :2] = xy_w
        p[:, 2] = env.iscene.env_origins[:, 2] + z
        return p

    def drop_crate_at(xy_w: torch.Tensor, drop_z: float = hover_z) -> None:
        """Gravity-drop the INVERTED crate from a free hover over `xy_w`."""
        _write_body(scene.crate, floor_pt(xy_w, drop_z),
                    _qx(torch.full((n,), math.pi, device=device)))
        _step(200)

    def trap_at(xy_w: torch.Tensor, body=None) -> None:
        """CONSTRUCT a trap: block placed at `xy_w` (transport), crate gravity-dropped
        over it — the capture itself is always made by contact."""
        _write_body(body if body is not None else scene.red,
                    floor_pt(xy_w, c.block / 2 + 0.003), None)
        _step(30)
        drop_crate_at((body if body is not None else scene.red).data.root_pos_w[:, :2])

    def pen_xy() -> torch.Tensor:
        _refresh()
        return scene.pen.data.root_pos_w[:, :2].clone()

    def pen_pt(loc_xy) -> torch.Tensor:
        """Pen-frame xy point -> world xy (per-env)."""
        _refresh()
        loc = torch.tensor([loc_xy[0], loc_xy[1], 0.0], device=device).expand(n, 3)
        return (scene.pen.data.root_pos_w + _qapply(scene.pen.data.root_quat_w, loc))[:, :2]

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _AUD["on"] = True
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    check("settle/no-NaN: layout settles finite; crate UPRIGHT (opening up), nothing "
          "trapped; score 0, no success",
          bool(scene._finite()[0]) and float(scene.crate_up_z()[0]) > 0.98
          and not bool(scene.trapped()[0]) and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (scene.crate.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.crate.data.root_quat_w[0]),
                scene.pen.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.pen.data.root_quat_w[0]),
                scene.red.data.root_pos_w[0, :2].clone())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_cp, a_cy, a_pp, a_py, a_rp = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_cp, b_cy, b_pp, b_py, b_rp = readback()
    d_cp, d_pp = float((a_cp - b_cp).norm()), float((a_pp - b_pp).norm())
    d_rp = float((a_rp - b_rp).norm())
    d_cy, d_py = dyaw(a_cy, b_cy), dyaw(a_py, b_py)
    print(f"[smoke] randomization deltas: crate_xy={d_cp * 1000:.1f}mm "
          f"crate_yaw={d_cy:.1f}deg pen_xy={d_pp * 1000:.1f}mm pen_yaw={d_py:.1f}deg "
          f"red_xy={d_rp * 1000:.1f}mm", flush=True)
    check("randomization-is-real: crate xy+yaw, pen xy+yaw and red xy readback differ",
          d_cp > 0.003 and d_cy > 2.0 and d_pp > 0.003 and d_py > 2.0 and d_rp > 0.003)

    # ================= 3. slot swap ===============================================================
    slots = set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        slots.add("0" if float(scene.red_slot[0]) > 0 else "1")
    print(f"[smoke] over 10 resets: red block slots {sorted(slots)}", flush=True)
    check("slot swap: the red block occupies BOTH scatter slots over 10 resets",
          slots == {"0", "1"})

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. negative: the SEED FAMILY'S OWN STRATEGY ================================
    # The seed closes a built lid over contents; its family's naive transfer here is
    # "get the block into the box / the box to the goal" with NO closure: red block
    # dropped INSIDE the UPRIGHT crate, crate centred on the pen. Settled and tidy —
    # and worth nothing: an open box does not trap.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _write_body(scene.crate, floor_pt(pen_xy(), c.height / 2 + 0.002),
                _qz(torch.zeros(n, device=device)))
    _step(30)
    _write_body(scene.red, floor_pt(pen_xy(), c.height + 0.03), None)  # falls inside
    _step(150)
    _report("seed-strategy")
    loc = scene._crate_local(scene.red.data.root_pos_w)[0]
    print(f"[smoke] red block crate-frame=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
          f"{float(loc[2]):+.3f}) (inside the upright crate)", flush=True)
    check("negative (SEED strategy): red block inside the UPRIGHT crate centred on "
          "the pen — an open box delivers nothing: no success, score ~0",
          abs(float(loc[0])) < c.half and abs(float(loc[1])) < c.half
          and bool(scene.in_pen()[0]) and float(scene.crate_up_z()[0]) > 0.98
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.05)
    _REC["on"] = False

    # ================= 6. negative: wrong object ==================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    trap_at(pen_xy(), body=scene.blue)
    _report("wrong-object")
    check("negative (wrong object): the BLUE block trapped and delivered instead — "
          "no success, score ~0",
          bool(scene.capped()[0]) and bool(scene.in_pen()[0])
          and bool(scene._under_crate(scene.blue, c.capture_xy_tol)[0])
          and not bool(scene.trapped()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.05)

    # ================= 7. near-miss: block just OUTSIDE a wall ====================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    drop_crate_at(pen_xy())
    beside = scene.crate.data.root_pos_w[:, :2].clone()
    beside[:, 0] += c.half + c.block / 2 + 0.012  # ~30 mm gap to the outer wall face
    _write_body(scene.red, floor_pt(beside, c.block / 2 + 0.003), None)
    _step(60)
    _report("near-miss-capture")
    loc = scene._crate_local(scene.red.data.root_pos_w)[0]
    print(f"[smoke] red block crate-frame |x|={abs(float(loc[0])) * 1000:.0f}mm "
          f"(capture tol {c.capture_xy_tol * 1000:.0f}mm)", flush=True)
    check("near-miss (capture): red block on the floor just outside a wall of the "
          "capped crate — not trapped, no success",
          bool(scene.capped()[0]) and abs(float(loc[0])) > c.capture_xy_tol
          and not bool(scene.trapped()[0]) and not bool(scene.success()[0]))

    # ================= 8. rim-perch: rim lands ON the block =======================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    spot = pen_xy()
    _write_body(scene.red, floor_pt(spot, c.block / 2 + 0.003), None)
    _step(30)
    off = scene.red.data.root_pos_w[:, :2].clone()
    off[:, 0] += c.half  # wall ring directly over the block -> rim lands on it
    _write_body(scene.crate, floor_pt(off, c.block + c.height / 2 + 0.010),
                _qx(torch.full((n,), math.pi, device=device)))
    _step(200)
    _report("rim-perch")
    z_rel = float(scene.crate.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    print(f"[smoke] perched crate z={z_rel * 1000:.1f}mm "
          f"(seat {c.height / 2 * 1000:.0f}±{c.seat_z_tol * 1000:.0f}mm) "
          f"up_z={float(scene.crate_up_z()[0]):+.3f}", flush=True)
    check("rim-perch: crate dropped half-over the block, rim resting ON it — "
          "perched/tilted, not floor-seated: not capped, no success",
          not bool(scene.capped()[0]) and not bool(scene.trapped()[0])
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 9. roof-park: block ON TOP of the inverted crate ===========================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    drop_crate_at(pen_xy())
    top = scene.crate.data.root_pos_w[:, :2].clone()
    _write_body(scene.red, floor_pt(top, c.height + c.block / 2 + 0.005), None)
    _step(90)
    _report("roof-park")
    z_rel = float(scene.red.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    print(f"[smoke] roof-parked red block z={z_rel * 1000:.1f}mm "
          f"(capture window {c.capture_z_lo * 1000:.0f}..{c.capture_z_hi * 1000:.0f}mm)",
          flush=True)
    check("roof-park: red block resting on TOP of the inverted crate on the pen — "
          "the height window rejects it: not trapped, no success",
          bool(scene.capped()[0]) and bool(scene.in_pen()[0]) and z_rel > c.capture_z_hi
          and not bool(scene.trapped()[0]) and not bool(scene.success()[0]))

    # ================= 10. restraint: blue block on the pen =======================================
    # Blue placed on the pen FIRST, then a CORRECT trap delivered — every other clause
    # holds, only the distractor clause refuses. Proves the restraint is load-bearing.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _write_body(scene.blue, floor_pt(pen_pt((0.095, 0.095)), c.block / 2 + 0.003), None)
    _step(30)
    trap_at(pen_xy())
    _report("restraint")
    check("restraint: correct trap delivered, but the BLUE block sits on the pen "
          "square — no success, score <= 0.75",
          bool(scene.trapped()[0]) and bool(scene.in_pen()[0])
          and not bool(scene.distractor_clear()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.75)
    _REC["on"] = False

    # ================= 11. near-miss delivery =====================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    trap_at(pen_pt((c.pen_xy_tol + 0.035, 0.0)))  # 85 mm out along the pen x-axis
    _report("near-miss-pen")
    d = scene._pen_local_xy(scene.crate.data.root_pos_w)[0].abs()
    print(f"[smoke] pen-frame offset=({float(d[0]) * 1000:.0f},{float(d[1]) * 1000:.0f})mm "
          f"(tol {c.pen_xy_tol * 1000:.0f}mm/axis)", flush=True)
    check("near-miss (delivery): correct trap 85 mm from the pen centre (window "
          "50 mm/axis) — not in the pen, no success, score <= 0.75",
          bool(scene.trapped()[0]) and float(d.max()) > c.pen_xy_tol
          and not bool(scene.in_pen()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.75)

    # ================= 12. mechanism: sliding transports the trap =================================
    # Push the capped crate horizontally AWAY from the pen: the crate must slide and
    # the trapped block must be dragged along INSIDE — the mechanism the task is
    # built on (and the reason delivery is possible at all).
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    trap_at(scene.red.data.root_pos_w[:, :2].clone())  # trap at the red block's slot
    assert bool(scene.trapped()[0]), "probe setup: trap must be made"
    c0 = scene.crate.data.root_pos_w[0, :2].clone()
    r0 = scene.red.data.root_pos_w[0, :2].clone()
    away = (c0 - scene.pen.data.root_pos_w[0, :2])
    away = away / float(away.norm())
    zero = torch.zeros(n, 1, 3, device=device)
    fmag = 3.5
    moved = 0.0
    for i in range(480):
        moved = float((scene.crate.data.root_pos_w[0, :2] - c0).norm())
        if moved > 0.10:
            break
        f = torch.zeros(n, 1, 3, device=device)
        f[0, 0, :2] = fmag * away
        scene.crate.set_external_force_and_torque(f, zero, env_ids=_all_ids(),
                                                  is_global=True)
        _step(1)
        if i % 90 == 89 and moved < 0.01:
            fmag = min(fmag + 0.8, 5.0)
            print(f"[smoke] slide probe: crate not moving; force -> {fmag:.1f} N",
                  flush=True)
    scene.crate.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
    _step(90)
    _report("mech-slide")
    dragged = float((scene.red.data.root_pos_w[0, :2] - r0).norm())
    print(f"[smoke] slide: crate moved {moved * 1000:.0f}mm, block dragged "
          f"{dragged * 1000:.0f}mm, still inside="
          f"{bool(scene._under_crate(scene.red, c.capture_xy_tol)[0])}", flush=True)
    check("MECHANISM (slide): a ~3.5 N CoM push slides the capped crate >= 60 mm and "
          "the walls drag the trapped block along, still inside",
          moved >= 0.06 and dragged >= 0.04
          and bool(scene._under_crate(scene.red, c.capture_xy_tol)[0])
          and bool(scene.trapped()[0]))
    _REC["on"] = False

    # ================= 13+14. mechanism: lifting frees / latched credit ===========================
    # Continue from the trap of check 12: CARRY the crate away (teleport = the most
    # perfect lift-and-carry possible). The block stays behind — the trap only holds
    # on the floor. The live trap is gone but the latched credit survives.
    r0 = scene.red.data.root_pos_w[0, :2].clone()
    far = scene.crate.data.root_pos_w[:, :2].clone()
    far[:, 0] += 0.25
    _write_body(scene.crate, floor_pt(far, c.height / 2 + 0.002),
                _qx(torch.full((n,), math.pi, device=device)))
    _step(90)
    _report("mech-lift")
    left = float((scene.red.data.root_pos_w[0, :2] - r0).norm())
    print(f"[smoke] carry: crate moved 250mm, block left behind (moved "
          f"{left * 1000:.0f}mm)", flush=True)
    check("MECHANISM (lift): carrying the crate away leaves the block behind "
          "(< 20 mm displacement) — a carried crate delivers nothing",
          left < 0.020 and not bool(scene.trapped()[0])
          and not bool(scene.success()[0]))
    check("latched credit: after the carry the live trap is gone but the latched "
          "`trapped` credit survives (score >= 0.30, still no success)",
          not bool(scene.trapped()[0]) and float(scene.score()[0]) >= c.w_trap - 1e-6
          and not bool(scene.success()[0]))

    # ================= 15. negative: crate on its side ============================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    side = scene.red.data.root_pos_w[:, :2].clone()
    side[:, 0] += 0.15
    _write_body(scene.crate, floor_pt(side, c.half + 0.005),
                _qx(torch.full((n,), math.pi / 2, device=device)))
    _step(120)
    _report("tipped-crate")
    check("negative (tipped): crate lying on its SIDE — neither upright nor "
          "inverted-seated: not capped, no success",
          not bool(scene.capped()[0]) and abs(float(scene.crate_up_z()[0])) < 0.5
          and not bool(scene.success()[0]))

    # ================= 16. settle gate ============================================================
    # The exact success pose — trap in the pen, blue clear — but the crate still
    # SLIDING at ~0.45 m/s. success() must refuse while anything moves. The probe is
    # dismantled (transport) well before friction could settle it into a real success.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _write_body(scene.red, floor_pt(pen_xy(), c.block / 2 + 0.003), None)
    _step(30)
    vel = torch.zeros(n, 3, device=device)
    vel[:, 0] = 0.45
    _write_body(scene.crate, floor_pt(scene.red.data.root_pos_w[:, :2].clone(),
                                      c.height / 2 + 0.002),
                _qx(torch.full((n,), math.pi, device=device)), lin_vel=vel)
    moving_ok = True
    for _ in range(5):
        _step(1)
        v = float(scene.crate.data.root_lin_vel_w[0].norm())
        moving_ok = moving_ok and v > 0.1 and bool(scene.trapped()[0]) \
            and not bool(scene.settled()[0]) and not bool(scene.success()[0])
    _report("settle-gate")
    # dismantle before it can settle into a real success
    parked = scene.crate.data.root_pos_w[:, :2].clone()
    parked[:, 0] += 0.30
    _write_body(scene.crate, floor_pt(parked, c.height / 2 + 0.003),
                _qz(torch.zeros(n, device=device)))
    _step(30)
    check("settle gate: the exact success pose still sliding at ~0.45 m/s is "
          "refused while anything moves",
          moving_ok)

    # ================= 17. rejection audit ========================================================
    check("rejection audit: success() was never True at any step of this battery",
          _AUD["hits"] == 0)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.trap_crate")
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

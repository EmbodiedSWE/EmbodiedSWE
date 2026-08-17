"""Smoke / rubric-REJECTION battery for LiddedShoeBoxScene — NullRobot, teleported probes.

solve.py is the acceptance proof (open -> antiparallel pack -> flush close, latched
credit monotone along a real trajectory). This battery proves the rubric REJECTS wrong
outcomes and that the physical claims are load-bearing: the closed lid SEALS the mouth
(the seed's pick-and-drop strategy delivers nothing, even pressed with 3x a shoe's
weight — the open-first order is enforced by geometry, not fiat), the interior is one
size too small for every packing except the flat antiparallel nesting (parallel and
stacked constructs leave the lid riding proud), near-misses outside the tolerances are
refused (shoe across the rim, lid ajar, lid upside-down), and the judged closure is
LIVE contact (remove the lid from a real success and success collapses while the
latched credit survives). Every probe is CONSTRUCTED as a state (teleport, real
physics steps, judge) — instrumentation, never a solution.

Checks:
   1. settle/no-NaN     — seeded reset settles finite; lid READBACK seated on the box,
                          both shoes on the floor; score 0, no success;
   2. randomization     — two seeded resets: READBACK box yaw, box xy and shoe_a
                          ground spawn all differ (the packing axis must be read);
   3. null-policy       — 240 idle steps: score ~0, no success;
   4. SEED STRATEGY     — the seed task's whole plan (pick each shoe, carry it over
                          the box, release) constructed against the CLOSED box: both
                          shoes dropped from above the lid land on it / slide off —
                          nothing enters, the lid stays seated, score 0, no success;
   5. CLOSED-LID PRESS  — a shoe pressed down onto the seated lid with 5 N (~3.4x its
                          weight): it descends 30 mm to CONTACT (the probe moved) and
                          stops ON the plate — never enters, the lid holds seat. The
                          open-first order is geometric, not rubric bookkeeping;
   6. PARALLEL jam      — both shoes SAME heading (the no-reasoning packing): heel
                          counters 2x72 mm in a 132 mm interior — the second shoe
                          perches tilted, not inside; the dropped lid cannot seat;
   7. STACKED jam       — one shoe on top of the other: top shoe above the plug line,
                          lid rides proud — not seated, no success;
   8. RIM near-miss     — a shoe lying flat ACROSS the rim (toe over the cavity, heel
                          outside): not inside (body points outside the interior);
                          the dropped lid must never be counted seated while the shoe
                          still crosses the mouth (perch or expel, both honest) — no
                          success either way;
   9. LID AJAR          — proper antiparallel pack, lid dropped 15 mm off-axis: plug
                          catches the rim, xy/tilt clauses refuse — no success,
                          score stays at the latched 0.65;
  10. UPSIDE-DOWN lid   — proper pack, lid dropped FLIPPED (handle down): the upright
                          clause refuses regardless of height — no success;
  11. lid-off only      — lid set aside, shoes untouched: score ~0.15, no success;
  12. packed, box open  — both shoes nested, lid on the table: score ~0.65 < 0.9,
                          no success (closure is required, not implied);
  13. REMOVE-LID        — full success CONSTRUCTED (pack + flush close, success live,
                          score 1.0), then the lid teleported to the floor: success
                          COLLAPSES while the latched score survives at ~0.65 — the
                          closure is judged live, not latched;
  14. rejection audit   — success() never fired at any step during checks 4-12;
  15. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.put_shoes_in_box_i143.smoke --headless
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

_qmul, _qx, _qz = task_scene._qmul, task_scene._qx, task_scene._qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}
_AUDIT = {"on": False, "fired": False}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        if _AUDIT["on"]:
            _AUDIT["fired"] |= bool(env.scene.success()[0])
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


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.shoe_box_lid")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.05, -0.80, 0.70)) + o),
                                tuple(np.array((0.30, 0.00, 0.05)) + o),
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

    def box_pose(x_loc: float, y_loc: float, z_loc: float, yaw_rel: float,
                 flip: bool = False) -> torch.Tensor:
        """(N,13) root state at box-local coords, box-relative yaw (roll pi if flip),
        zero velocity."""
        _refresh()
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0], loc[:, 1], loc[:, 2] = x_loc, y_loc, z_loc
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.box.data.root_pos_w + quat_apply(scene.box.data.root_quat_w, loc)
        q = _qmul(scene.box.data.root_quat_w, _qz(torch.full((n,), yaw_rel, device=device)))
        if flip:
            q = _qmul(q, _qx(torch.full((n,), math.pi, device=device)))
        st[:, 3:7] = q
        return st

    def world_pose(x: float, y: float, z: float) -> torch.Tensor:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = 1.0
        st[:, 0:3] += env.iscene.env_origins
        return st

    def loc_of(body) -> torch.Tensor:
        _refresh()
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(scene.box.data.root_quat_w,
                                  body.data.root_pos_w - scene.box.data.root_pos_w)[0]

    def report(tag: str) -> None:
        s = scene._status()
        ll = s["lid_loc"][0]
        parts = []
        for nm in scene.SHOE_NAMES:
            p = loc_of(scene.shoes[nm])
            parts.append(f"{nm}=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f})")
        print(f"[smoke] {tag:16s} | lid=({float(ll[0]):+.3f},{float(ll[1]):+.3f},"
              f"{float(ll[2]):+.3f}) up_z={float(s['lid_up_z'][0]):+.2f} "
              + " ".join(parts)
              + f" inside={s['inside'][0].tolist()} seated={bool(s['seated'][0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(_REC['frames'])}", flush=True)

    def lid_aside() -> None:
        """Set the lid on the floor far from the box and both shoes (opens the mouth)."""
        scene.lid.write_root_state_to_sim(world_pose(-0.60, 0.60, 0.05), _all_ids())
        _step(120)

    def place_shoe_flat(name: str, y_loc: float, yaw_rel: float, x_loc: float = 0.0) -> None:
        """Construct a shoe lying flat on the box floor (direct write, then settle)."""
        scene.shoes[name].write_root_state_to_sim(
            box_pose(x_loc, y_loc, c.sole_t / 2 + 0.001, yaw_rel), _all_ids())
        _step(90)

    def drop_lid(dx: float = 0.0, dy: float = 0.0, flip: bool = False,
                 extra_z: float = 0.0) -> None:
        """Release the lid from a free-space hover (plug underside ~2 mm above the rim;
        raise with extra_z when an obstacle pokes above the rim so the WRITE never
        interpenetrates it)."""
        hover = c.in_h + c.plug_t + c.plate_t / 2 + 0.002 + extra_z \
            + (0.010 if flip else 0.0)
        scene.lid.write_root_state_to_sim(box_pose(dx, dy, hover, 0.0, flip=flip),
                                          _all_ids())
        _step(240)

    def shoe_pts_loc(name: str) -> torch.Tensor:
        """The scene's three judged body points of a shoe, in the box frame (3,3)."""
        _refresh()
        from isaaclab.utils.math import quat_apply_inverse

        b = scene.shoes[name]
        q = b.data.root_quat_w[:, None, :].expand(n, 3, 4).reshape(n * 3, 4)
        pts_w = b.data.root_pos_w[:, None, :] + quat_apply(
            q, scene._pts_local.reshape(n * 3, 3)).reshape(n, 3, 3)
        return quat_apply_inverse(
            scene.box.data.root_quat_w[:, None, :].expand(n, 3, 4).reshape(n * 3, 4),
            (pts_w - scene.box.data.root_pos_w[:, None, :]).reshape(n * 3, 3)
        ).reshape(n, 3, 3)[0]

    def pack_proper() -> None:
        """Construct the legitimate antiparallel nesting (box open)."""
        lid_aside()
        place_shoe_flat("shoe_a", -0.028, 0.0)
        place_shoe_flat("shoe_b", +0.028, math.pi)
        _step(60)

    down5 = torch.zeros(n, 1, 3, device=device)
    down5[:, 0, 2] = -5.0
    zero_w = torch.zeros(n, 1, 3, device=device)

    # ================= 1. settle / no-NaN / lid starts seated =====================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    report("settle")
    _REC["on"] = False
    bodies = [scene.lid, *scene.shoes.values()]
    finite = all(bool(torch.isfinite(b.data.root_pos_w).all()) for b in bodies)
    s = scene._status()
    lid_seated_start = bool(s["seated"][0])
    shoes_out = not bool(s["inside"][0].any())
    check("settle/no-NaN: seeded reset settles finite, lid READBACK seated on the box, "
          "both shoes outside on the floor, score 0, no success",
          finite and lid_seated_start and shoes_out
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (yaw_of(scene.box.data.root_quat_w[0]),
                scene.box.data.root_pos_w[0, :2].clone(),
                scene.shoes["shoe_a"].data.root_pos_w[0, :2].clone())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_bp, a_sp = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_bp, b_sp = readback()
    d_yawv = dyaw(a_yaw, b_yaw)
    d_bp, d_sp = float((a_bp - b_bp).norm()), float((a_sp - b_sp).norm())
    print(f"[smoke] randomization deltas: box_yaw={d_yawv:.1f}deg "
          f"box_xy={d_bp * 1000:.1f}mm shoe_a_xy={d_sp * 1000:.1f}mm", flush=True)
    check("randomization-is-real: box yaw, box xy and shoe_a spawn readback differ",
          d_yawv > 5.0 and d_bp > 0.003 and d_sp > 0.03)

    # ================= 3. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    _AUDIT["on"] = True  # ---- negative probes below: success must never fire ----

    # ================= 4. SEED STRATEGY: pick-and-drop against the CLOSED box =====================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    scene.shoes["shoe_a"].write_root_state_to_sim(
        box_pose(-0.030, -0.030, c.seat_z + 0.050, 0.0), _all_ids())
    _step(150)
    scene.shoes["shoe_b"].write_root_state_to_sim(
        box_pose(+0.030, +0.030, c.seat_z + 0.050, math.pi / 2), _all_ids())
    _step(240)
    report("seed-strategy")
    _REC["on"] = False
    s = scene._status()
    check("negative (SEED strategy): both shoes carried over the box and released — "
          "they land on the CLOSED lid / slide off; nothing enters, lid stays seated, "
          "no packed credit, score 0, no success",
          not bool(s["inside"][0].any()) and not bool(scene._packed[0].any())
          and bool(s["seated"][0]) and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))

    # ================= 5. CLOSED-LID PRESS: the mouth is sealed ===================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    scene.shoes["shoe_a"].write_root_state_to_sim(
        box_pose(0.0, +0.045, c.seat_z + c.plate_t / 2 + c.sole_t / 2 + 0.030, 0.0),
        _all_ids())
    _step(150)  # falls ~30 mm and lands ON the plate (contact reached)
    z_rest = float(loc_of(scene.shoes["shoe_a"])[2])
    scene.shoes["shoe_a"].set_external_force_and_torque(
        down5, zero_w, env_ids=_all_ids(), is_global=True)
    _step(180)
    scene.shoes["shoe_a"].set_external_force_and_torque(
        zero_w, zero_w, env_ids=_all_ids())
    _step(60)
    report("closed-press")
    _REC["on"] = False
    z_press = float(loc_of(scene.shoes["shoe_a"])[2])
    s = scene._status()
    on_lid = 0.070 <= z_rest <= 0.100  # descended to the plate top (~0.079), not inside
    check("negative (CLOSED-LID PRESS): shoe released above the seated lid falls to "
          "CONTACT on the plate, then pressed with 5 N (~3.4x its weight) — it stays "
          "ON the lid, never enters, the lid holds its seat: open-first is enforced "
          "by geometry",
          on_lid and z_press > 0.055 and not bool(s["inside"][0].any())
          and not bool(scene._packed[0].any()) and bool(s["seated"][0])
          and not bool(scene.success()[0]))

    # ================= 6. PARALLEL packing jams ===================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    lid_aside()
    place_shoe_flat("shoe_a", -0.028, 0.0)
    scene.shoes["shoe_b"].write_root_state_to_sim(
        box_pose(0.0, +0.030, c.in_h + c.sole_t / 2 + 0.005, 0.0), _all_ids())
    _step(240)  # same heading: 2 x 72 mm heels in a 132 mm interior — perches tilted
    drop_lid()
    report("parallel-jam")
    _REC["on"] = False
    s = scene._status()
    check("negative (PARALLEL packing): both shoes same heading — the second shoe "
          "cannot lie flat (heels 144 mm > 132 mm interior) and the dropped lid "
          "cannot seat; no success, score < 0.9",
          not bool(s["inside"][0, 1]) and not bool(s["seated"][0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.9)

    # ================= 7. STACKED packing jams ====================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    lid_aside()
    place_shoe_flat("shoe_a", 0.0, 0.0)
    scene.shoes["shoe_b"].write_root_state_to_sim(
        box_pose(0.0, 0.0, c.sole_t + c.heel_h + c.sole_t / 2 + 0.004, 0.0), _all_ids())
    _step(240)
    drop_lid()
    report("stacked-jam")
    s = scene._status()
    check("negative (STACKED packing): one shoe on top of the other stands above the "
          "plug line — top shoe not inside, lid rides proud, not seated, no success",
          not bool(s["inside"][0, 1]) and not bool(s["seated"][0])
          and not bool(scene.success()[0]))

    # ================= 8. RIM near-miss: flat across the rim ======================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    lid_aside()
    # flat ON the rim, spanning the +x wall: CoM over the wall strip, toe over the cavity
    scene.shoes["shoe_a"].write_root_state_to_sim(
        box_pose(+0.0665, 0.0, c.in_h + c.sole_t / 2 + 0.002, math.pi), _all_ids())
    _step(180)
    s = scene._status()
    near_miss_ok = (not bool(s["inside"][0, 0])) and not bool(scene._packed[0, 0])
    report("rim-shoe")
    # drop the lid from a hover ABOVE the protruding shoe (the write must not
    # interpenetrate it). Honest outcomes: the lid perches on the shoe (not seated),
    # or it knocks the shoe clear of the mouth and seats on an empty box. The LEAK
    # would be: lid counted seated while the shoe still crosses the mouth.
    drop_lid(extra_z=0.030)
    report("rim-near-miss")
    s = scene._status()
    pts = shoe_pts_loc("shoe_a")
    over_mouth = bool((((pts[:, 0].abs() < c.in_l / 2 + c.wall_t)
                        & (pts[:, 1].abs() < c.in_w / 2 + c.wall_t)
                        & (pts[:, 2] > 0.020)).any()))
    leak = bool(s["seated"][0]) and over_mouth
    check("negative (RIM near-miss): a shoe lying across the rim (toe over the "
          "cavity, heel outside) earns no inside/packed credit, and the dropped lid "
          "is never counted seated while the shoe still crosses the mouth; no success",
          near_miss_ok and not leak and not bool(scene.success()[0]))

    # ================= 9. LID AJAR ================================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    pack_proper()
    drop_lid(dx=0.015)
    report("lid-ajar")
    _REC["on"] = False
    s = scene._status()
    lid_loc = s["lid_loc"][0]
    off_axis = float(lid_loc[0:2].norm()) > c.lid_xy_tol
    check("negative (LID AJAR): proper pack, lid dropped 15 mm off-axis — the plug "
          "catches the rim, off the xy tolerance, not seated, no success, score at "
          "the latched 0.65",
          off_axis and not bool(s["seated"][0]) and not bool(scene.success()[0])
          and 0.60 <= float(scene.score()[0]) <= 0.66)

    # ================= 10. UPSIDE-DOWN lid ========================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    pack_proper()
    drop_lid(flip=True)
    report("lid-flipped")
    _REC["on"] = False
    s = scene._status()
    check("negative (UPSIDE-DOWN lid): proper pack, lid dropped flipped (handle "
          "down) — the upright clause refuses whatever height it rests at; not "
          "seated, no success",
          float(s["lid_up_z"][0]) < 0.0 and not bool(s["seated"][0])
          and not bool(scene.success()[0]))

    # ================= 11. lid-off only ===========================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    lid_aside()
    report("lid-off-only")
    sc = float(scene.score()[0])
    check("partial (lid off only): score ~0.15, no success",
          0.14 <= sc <= 0.16 and not bool(scene.success()[0]))

    # ================= 12. packed but box open ====================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    pack_proper()
    report("packed-open")
    sc = float(scene.score()[0])
    s = scene._status()
    check("partial (packed, box open): both shoes nested antiparallel, lid on the "
          "table — score ~0.65 < 0.9, no success: closure is required",
          bool(s["inside"][0].all()) and 0.60 <= sc <= 0.66
          and not bool(scene.success()[0]))

    _AUDIT["on"] = False  # ---- check 13 constructs a REAL success on purpose ----

    # ================= 13. REMOVE-LID: closure is judged live =====================================
    _REC["on"] = True
    drop_lid()
    _step(120)
    report("constructed-win")
    won = bool(scene.success()[0])
    sc_win = float(scene.score()[0])
    scene.lid.write_root_state_to_sim(world_pose(-0.60, 0.60, 0.05), _all_ids())
    _step(90)
    report("lid-removed")
    _REC["on"] = False
    lost = not bool(scene.success()[0])
    sc_after = float(scene.score()[0])
    check("liveness (REMOVE-LID): constructed success (pack + flush close) is live "
          "(score 1.0); teleport the lid to the floor and success COLLAPSES while "
          "the latched score survives at ~0.65",
          won and sc_win >= 0.99 and lost and 0.60 <= sc_after <= 0.66)

    # ================= 14. rejection audit ========================================================
    check("rejection audit: success() never fired during the negative probes "
          "(checks 4-12)", not _AUDIT["fired"])

    # ================= 15. video ==================================================================
    frames = _REC["frames"]
    if frames:
        np.savez_compressed(args.out, frames=np.stack(frames, axis=0))
        print(f"[smoke] saved {len(frames)} frames to {args.out}", flush=True)
    check("frames.npz saved with video frames", len(frames) > 0 and os.path.exists(args.out))

    # ================= verdict ====================================================================
    n_pass = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        if not ok:
            print(f"[smoke] FAILED: {name}", flush=True)
    if n_pass == len(checks):
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
        code = 0
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
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

"""Smoke / rubric-REJECTION battery for PuddingDockDispenseScene — NullRobot, probes.

solve.py is the acceptance proof (dock -> dispense -> success on real contact
dynamics, two seeds). This battery proves the rubric REJECTS wrong outcomes and that
the mechanism claims the task rests on — covered-bay irreversibility, the docked
window, box identity — are physics, not fiat. Wrong outcomes are CONSTRUCTED as
settled states: teleports go to free space OUTSIDE every scoring volume (above the
drop holes, above the scattered tray, in front of the bay), then free fall / force
pushes / settling produce the judged state.

Checks:
   1. settle/no-NaN    — seeded reset settles finite; boxes seated in opposite
                         chutes matching pud_lane (readback); tray undocked; mass
                         readback (station 60 kg, tray 0.5 kg — custom spawners);
                         score ~0, no success;
   2. randomization    — two seeded resets: READBACK station yaw, station xy, tray
                         xy and tray yaw all differ;
   3. lane varies      — over 12 resets the pudding occupies BOTH chutes, and the
                         box y readback matches pud_lane every time;
   4. null-policy      — 240 idle steps: nothing moves, score ~0, no success;
   5. SEED STRATEGY    — the seed's whole plan ("put the pudding in the tray"):
                         pudding dropped into the scattered, UNDOCKED tray on the
                         open floor -> contained but score ~0, no success (the tray
                         being docked is load-bearing, not decorative);
   6. out-of-order (a) — pudding dispensed through its hole with NO tray below:
                         lands on the bay floor under the deck — not in any tray,
                         no success, score <= progress credit only;
   7. out-of-order (b) — the tray then lowered onto the docked pose OVER the lost
                         box: it perches on the box (z gate) / cannot contain it —
                         the dispense-first ordering is UNRECOVERABLE, no success;
   8. reversed dock    — tray pushed at the bay mouth HANDLE-FIRST: the tall handle
                         jams against the deck edge ~0.24 m short of the window —
                         not docked, no success (the declared handle-trailing
                         insertion is geometry, not convention);
   9. near-miss dock   — tray docked 60 mm SHORT (outside |dx|<=30 mm) and the
                         pudding dropped through its hole INTO that tray: contained
                         but undocked -> no landed credit, no success;
  10. deck perch       — (docked tray) pudding dropped onto the rear deck, directly
                         ABOVE the tray's interior footprint: xy inside, z above ->
                         not contained, no success;
  11. wrong object     — the RED decoy dispensed through its hole into the docked
                         tray: contained decoy is a violation -> no success;
  12. both boxes       — pudding dispensed too (docked + progress + landed all
                         latched): score capped at 0.80, decoy clause still refuses
                         success;
  13. frames.npz       — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.<task>.smoke --headless
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
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from . import scene as _scene  # noqa: F401 - registers simgen.pudding_dock_dispense
except ImportError:  # pragma: no cover - direct-script fallback
    import sys  # noqa: E402

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as _scene  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

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
    dx, dy, dyaw = scene.tray_dock_err()
    pl = scene._station_local(scene.pudding.data.root_pos_w)[0]
    print(f"[smoke] {tag:16s} | dock_err=({float(dx[0]):+.3f},{float(dy[0]):+.3f},"
          f"{float(dyaw[0]):+.1f}deg) docked={bool(scene.tray_docked()[0])} "
          f"pud_loc=({float(pl[0]):+.3f},{float(pl[1]):+.3f},{float(pl[2]):+.3f}) "
          f"in_tray={bool(scene.pudding_in_tray()[0])} "
          f"decoy_in_tray={bool(scene.decoy_in_tray()[0])} "
          f"L(dock={bool(scene._docked[0])},land={bool(scene._landed[0])},"
          f"prog={float(scene._progress[0]):.2f}) settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.pudding_dock_dispense")().build(num_envs=args.num_envs,
                                                           device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    zero3 = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.45, -0.85, 0.70)) + o),
                                tuple(np.array((0.35, 0.00, 0.12)) + o),
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

    def dyaw_deg(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    def station_point_w(loc_xyz) -> torch.Tensor:
        _refresh()
        loc = torch.tensor([list(loc_xyz)], device=device, dtype=torch.float).expand(n, 3)
        return scene.station.data.root_pos_w + _scene._qapply(
            scene.station.data.root_quat_w, loc)

    def station_axes() -> tuple[torch.Tensor, torch.Tensor]:
        q = scene.station.data.root_quat_w
        xh = _scene._qapply(q, torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(n, 3))
        yh = _scene._qapply(q, torch.tensor([[0.0, 1.0, 0.0]], device=device).expand(n, 3))
        return xh, yh

    def tele_tray(loc_x: float, *, flip: bool = False, z: float = 0.001) -> None:
        """Aligned tray teleport at station-local (loc_x, 0, z) — always used at
        poses OUTSIDE the docked window (|loc_x - dock_x| > tol or z above it)."""
        q_st = scene.station.data.root_quat_w
        q = q_st
        if flip:
            pi = torch.full((n,), math.pi, device=device)
            q = _scene._qmul(q_st, _scene._qz(pi))
        pos = station_point_w((loc_x, 0.0, 0.0))
        pos = pos.clone()
        pos[:, 2] = z
        _write_body(scene.tray, pos, q)

    def clear_wrench(body) -> None:
        body.set_external_force_and_torque(zero3, zero3, env_ids=_all_ids(),
                                           is_global=True)

    strat = {"mode": 0, "qref": None}

    def enc(body, f_world: torch.Tensor) -> torch.Tensor:
        if strat["mode"] == 0:
            return f_world
        return _scene.encode_force(1, strat["qref"], body.data.root_quat_w, f_world)

    def probe_wrench_mode() -> None:
        """Same probe discipline as solve.py: find the wrench encoding that moves
        the tray along station +x from the approach pose."""
        for mode, qref, tag in ((0, None, "raw global"),
                                (1, scene.tray.data.root_quat_w.clone(), "reset-quat"),
                                (1, None, "identity")):
            if mode == 1 and qref is None:
                qref = torch.zeros(n, 4, device=device)
                qref[:, 0] = 1.0
            tele_tray(-0.155)
            _step(15)
            xh, _ = station_axes()
            p0 = scene.tray.data.root_pos_w.clone()
            for _ in range(12):
                f = 6.0 * xh
                if mode == 1:
                    f = _scene.encode_force(1, qref, scene.tray.data.root_quat_w, f)
                scene.tray.set_external_force_and_torque(f.view(n, 1, 3), zero3,
                                                         env_ids=_all_ids(),
                                                         is_global=True)
                _step(1)
            clear_wrench(scene.tray)
            _refresh()
            d = (scene.tray.data.root_pos_w - p0)[0, :2]
            align = -1.0 if float(d.norm()) < 0.003 else float(
                (d / d.norm() * xh[0, :2]).sum())
            if align > 0.7:
                strat["mode"], strat["qref"] = mode, qref
                print(f"[smoke] wrench mode: {tag} (align={align:+.2f})", flush=True)
                return
        raise AssertionError("no wrench mode moves the tray")

    def push_tray(*, stop_dx: float | None = None, flip: bool = False,
                  max_steps: int = 2400) -> None:
        """Force-push the tray toward the back wall from the approach pose (the
        same velocity-capped contact push as solve.py). stop_dx: cut the force
        early (near-miss construction). flip: handle-first insertion attempt."""
        tele_tray(-0.155, flip=flip)
        _step(15)
        for j in range(max_steps):
            xh, yh = station_axes()
            dx, dy, _dyaw = scene.tray_dock_err()
            if stop_dx is not None and float(dx[0]) >= stop_dx:
                break
            v = scene.tray.data.root_lin_vel_w
            v_along = (v * xh).sum(dim=-1)
            vcap = 0.20 if float(dx[0]) < -0.06 else 0.08
            if stop_dx is not None:
                vcap = 0.10
            fx = torch.where(v_along < vcap, torch.full_like(v_along, 6.0),
                             torch.zeros_like(v_along))
            fy = torch.zeros_like(fx) if flip \
                else (-8.0 * dy - 2.0 * (v * yh).sum(dim=-1)).clamp(-3.0, 3.0)
            f_world = fx.unsqueeze(-1) * xh + fy.unsqueeze(-1) * yh
            scene.tray.set_external_force_and_torque(
                enc(scene.tray, f_world).view(n, 1, 3), zero3,
                env_ids=_all_ids(), is_global=True)
            _step(1)
            if stop_dx is None and not flip and bool(scene.tray_docked()[0]) \
                    and float(v.norm(dim=-1)[0]) < 0.04:
                break
            if flip and j > 500 and float(v.norm(dim=-1)[0]) < 0.005:
                break  # jammed against the deck edge — that IS the result
        clear_wrench(scene.tray)
        _step(150)

    def drop_through_hole(body, lane_sign: float, settle: int = 300) -> None:
        """CONSTRUCT 'box dispensed': teleport ABOVE the open hole slot (z_loc
        0.25 — above the lane-wall tops at 0.21, outside every scoring volume),
        free-fall through the deck hole, settle."""
        pos = station_point_w((0.235, lane_sign * c.lane_y, 0.25))
        _write_body(body, pos, scene.station.data.root_quat_w)
        _step(settle)

    # ================= 1. settle / no-NaN / seating / masses ======================================
    env.reset(seed=100)
    _REC["on"] = True
    _step(120)
    _report("settle")
    _REC["on"] = False
    lane = int(scene.pud_lane[0])
    pl = scene._station_local(scene.pudding.data.root_pos_w)[0]
    dl = scene._station_local(scene.decoy.data.root_pos_w)[0]
    m_st = float(scene.station.root_physx_view.get_masses().sum())
    m_tr = float(scene.tray.root_physx_view.get_masses().sum())
    print(f"[smoke] masses: station={m_st:.1f}kg tray={m_tr:.2f}kg | lane={lane:+d} "
          f"pud_y={float(pl[1]):+.3f} decoy_y={float(dl[1]):+.3f}", flush=True)
    check("settle/no-NaN: layout settles finite, boxes seated on the deck in "
          "opposite chutes matching pud_lane, tray undocked, authored masses read "
          "back, score ~0, no success",
          bool(scene._finite()[0])
          and abs(float(pl[2]) - (c.deck_top + c.box_size[2] / 2)) < 0.01
          and float(pl[1]) * lane > 0.03 and float(dl[1]) * lane < -0.03
          and not bool(scene.tray_docked()[0])
          and abs(m_st - c.station_mass) < 1.0 and abs(m_tr - c.tray_mass) < 0.05
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (yaw_of(scene.station.data.root_quat_w[0]),
                scene.station.data.root_pos_w[0, :2].clone(),
                scene.tray.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.tray.data.root_quat_w[0]))

    env.reset(seed=101)
    _step(10)
    a_yaw, a_sp, a_tp, a_tyaw = readback()
    env.reset(seed=202)
    _step(10)
    b_yaw, b_sp, b_tp, b_tyaw = readback()
    d_yaw = dyaw_deg(a_yaw, b_yaw)
    d_sp = float((a_sp - b_sp).norm())
    d_tp = float((a_tp - b_tp).norm())
    d_ty = dyaw_deg(a_tyaw, b_tyaw)
    print(f"[smoke] randomization deltas: station_yaw={d_yaw:.1f}deg "
          f"station_xy={d_sp * 1000:.1f}mm tray_xy={d_tp * 1000:.1f}mm "
          f"tray_yaw={d_ty:.1f}deg", flush=True)
    check("randomization-is-real: station yaw, station xy, tray xy and tray yaw "
          "readback all differ across seeds",
          d_yaw > 1.0 and d_sp > 0.003 and d_tp > 0.010 and d_ty > 3.0)

    # ================= 3. lane assignment varies ==================================================
    lanes = set()
    consistent = True
    for s in range(12):
        env.reset(seed=300 + s)
        _refresh()
        lv = int(scene.pud_lane[0])
        lanes.add(lv)
        py = float(scene._station_local(scene.pudding.data.root_pos_w)[0, 1])
        dy_ = float(scene._station_local(scene.decoy.data.root_pos_w)[0, 1])
        consistent = consistent and (py * lv > 0.03) and (dy_ * lv < -0.03)
    print(f"[smoke] over 12 resets: lanes seen {sorted(lanes)}, "
          f"readback consistent={consistent}", flush=True)
    check("lane-varies: the pudding occupies BOTH chutes over 12 resets and the "
          "box y readback matches pud_lane every time",
          lanes == {-1, 1} and consistent)

    # ================= 4. null policy fails =======================================================
    env.reset(seed=100)
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps — score ~0, no success",
          float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 5. negative: the SEED'S OWN STRATEGY =======================================
    # The seed's complete plan is "pick up the pudding and put it in the tray".
    # Executed here — pudding dropped into the scattered tray on the open floor —
    # it must be worth nothing: the DOCK is load-bearing.
    env.reset(seed=100)
    _step(60)
    _REC["on"] = True
    pos = scene.tray.data.root_pos_w.clone()
    pos[:, 2] = 0.15  # free space above the open tray, outside every scoring volume
    _write_body(scene.pudding, pos, scene.tray.data.root_quat_w)
    _step(300)
    _report("seed-strategy")
    _REC["on"] = False
    check("negative (SEED strategy): pudding dropped into the scattered UNDOCKED "
          "tray — contained, but score ~0 and no success (docking is load-bearing)",
          bool(scene.pudding_in_tray()[0]) and not bool(scene.tray_docked()[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 6+7. out-of-order: dispense first, then try to dock ========================
    env.reset(seed=100)
    _step(60)
    lane = int(scene.pud_lane[0])
    _REC["on"] = True
    drop_through_hole(scene.pudding, float(lane), settle=300)
    _report("dispense-first")
    pl = scene._station_local(scene.pudding.data.root_pos_w)[0]
    s6 = float(scene.score()[0])
    check("out-of-order (a): pudding dispensed with NO tray below — lands on the "
          "bay floor under the deck, not contained, no success, score <= progress "
          "credit",
          float(pl[2]) < 0.05 and not bool(scene.pudding_in_tray()[0])
          and not bool(scene.success()[0]) and s6 <= c.w_progress + 0.01)
    # (b) now bring the tray: lower it onto the docked pose OVER the lost box (the
    # write is at z=0.05, above the docked window's z gate) — it cannot seat.
    tele_tray(c.dock_x, z=0.05)
    _step(300)
    _report("dock-over-box")
    check("out-of-order (b): the tray lowered onto the dock OVER the lost box "
          "perches on it — the dispense-first ordering is unrecoverable: pudding "
          "not in the tray, no success",
          not bool(scene.pudding_in_tray()[0]) and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 8. reversed (handle-first) insertion is blocked ============================
    env.reset(seed=100)
    _step(60)
    probe_wrench_mode()
    _REC["on"] = True
    push_tray(flip=True, max_steps=900)
    _report("reversed-dock")
    dx, _dy, dyw = scene.tray_dock_err()
    _REC["on"] = False
    check("reversed-dock-blocked: pushed handle-FIRST, the tall handle jams "
          "against the deck edge far short of the window — not docked, no success",
          float(dx[0]) < -0.15 and not bool(scene.tray_docked()[0])
          and not bool(scene._docked[0]) and not bool(scene.success()[0]))

    # ================= 9. near-miss: docked 60 mm short + pudding delivered =======================
    env.reset(seed=100)
    _step(60)
    lane = int(scene.pud_lane[0])
    tele_tray(c.dock_x - 0.060)  # outside |dx| <= 30 mm: NOT docked
    _step(60)
    _REC["on"] = True
    drop_through_hole(scene.pudding, float(lane), settle=360)
    _report("near-miss-dock")
    _REC["on"] = False
    check("near-miss (short dock): tray 60 mm short of the window, pudding dropped "
          "through its hole INTO that tray — contained but undocked: no landed "
          "credit, no success",
          bool(scene.pudding_in_tray()[0]) and not bool(scene.tray_docked()[0])
          and not bool(scene._landed[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= c.w_progress + 0.01)

    # ================= 10-12. docked episode: perch, wrong object, both boxes =====================
    env.reset(seed=100)
    _step(60)
    lane = int(scene.pud_lane[0])
    push_tray()  # genuine contact dock, as in solve.py
    _report("docked")
    assert bool(scene.tray_docked()[0]), "smoke dock construction failed"
    # 10: pudding onto the rear deck — directly ABOVE the docked tray's interior
    # footprint in xy, but on the station roof: containment must refuse on z.
    pos = station_point_w((0.3275, 0.0, 0.20))
    _write_body(scene.pudding, pos, scene.station.data.root_quat_w)
    _step(240)
    _report("deck-perch")
    tl = scene._tray_local(scene.pudding.data.root_pos_w)[0]
    check("near-miss (deck perch): pudding at rest on the deck directly above the "
          "docked tray's interior footprint (tray-local xy inside, z above) — not "
          "contained, no success",
          abs(float(tl[0])) < c.tray_int_xhalf and abs(float(tl[1])) < c.tray_int_yhalf
          and float(tl[2]) > c.tray_wall_top
          and not bool(scene.pudding_in_tray()[0]) and not bool(scene.success()[0]))
    # 11: the RED decoy dispensed into the docked tray — violation.
    _REC["on"] = True
    drop_through_hole(scene.decoy, float(-lane), settle=300)
    _report("wrong-object")
    check("negative (wrong object): RED decoy dispensed through its hole into the "
          "docked tray — contained decoy is a violation: no success",
          bool(scene.decoy_in_tray()[0]) and bool(scene.tray_docked()[0])
          and not bool(scene.success()[0]))
    # 12: pudding too (from the deck, through its own hole) — every partial-credit
    # latch set, score at the 0.80 cap, still refused by the decoy clause.
    drop_through_hole(scene.pudding, float(lane), settle=360)
    _report("both-boxes")
    _REC["on"] = False
    s12 = float(scene.score()[0])
    check("negative (both boxes): pudding AND decoy in the docked tray — docked/"
          "progress/landed all latched, score capped at 0.80, success still refused",
          bool(scene.pudding_in_tray()[0]) and bool(scene.decoy_in_tray()[0])
          and bool(scene._landed[0]) and s12 <= 0.80 + 1e-5
          and not bool(scene.success()[0]))

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.pudding_dock_dispense")
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
    try:
        main()
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(2)

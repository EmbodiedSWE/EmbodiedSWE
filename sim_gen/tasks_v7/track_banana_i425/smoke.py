"""Smoke / rubric-REJECTION battery for GlazingBenchScene (sim_gen task
`track_banana_i425`) — NullRobot, teleported/pushed probe states, RECORDED.

This is NOT a solution (solve.py — clear the offcuts, gravity-drop the pane flush,
force-slide the keeper home — is the acceptance evidence that the rubric ACCEPTS a
correct outcome). Every teleport/velocity-write here is instrumentation that
CONSTRUCTS a wrong (or partial) outcome and asserts the rubric REJECTS it — plus
physics probes that prove the interference geometry is real (a pane dropped on the
fouled seat genuinely rests proud; a keeper locked first genuinely blocks the pane
drop; the housing genuinely stops the head). Force probes assert the actuator MOVED,
so no rejection is vacuous. No probe in this battery ever reaches success(), and a
final audit check asserts exactly that.

  1-2.  settle/no-NaN      — reset settles finite: pebbles foul the seat (out_frac
                             0), pane/decoy/keeper flat on the ground; score ~0;
  3-4.  randomization      — READBACK over 12 seeded resets: pebble count 2-4 and
                             pane/decoy slot shuffle; bench anchor + yaw vary;
  5.   null policy         — 300 idle steps -> score ~0, no success;
  6.   seed strategy       — pick_place/track_banana's move (carry a held object
                             along a 5-waypoint free-space arc): the pane carried
                             along an arc and set down on the apron -> score ~0;
  7.   drop w/o clearing   — the pane dropped centered on the FOULED seat rests
                             PROUD on the offcuts (z above the flush band): no seat
                             credit;
  8.   straddle near miss  — seat cleared, pane released half-over the recess edge:
                             settles tilted/off-center -> rejected by xy/upz/z;
  9.   upside-down pane    — knob-down in the cleared recess props the pane high:
                             rejected;
  10.  wrong object        — the GRAY decoy dropped on the cleared recess spans the
                             opening (readback: rests at bench-top height, cannot
                             enter): no seat credit, no success;
  11.  out-of-order        — seat cleared, keeper GENUINELY force-slid home first
                             (locked() true), then the pane flat-dropped: the tongue
                             overhang shrinks the opening below the pane — it cannot
                             seat; locked + cleared yet NO success;
  12.  partial slide       — pane seated honestly, keeper genuinely pushed but
                             stopped ~2 mm of overlap short of the threshold:
                             locked() false, score capped at 0.50;
  13.  backwards keeper    — keeper laid at the lock station rotated 180 deg
                             (head over the seat): signed-alignment gate rejects;
  14.  fly-through         — keeper written AT the lock pose sliding fast: the
                             settle gate rejects while it transits the window;
  15.  geometry+mass audit — bench-frame readback matches the interference margins
                             (slot admits tongue not head, propped rest above the
                             flush band, decoy wider than the recess) and the
                             custom-spawned bodies carry their authored masses;
  16.  rejection audit     — success() never True at any judged point;
  17.  final no-NaN        — all task-object states finite.

Run (forge): python -u -m simgen_tasks.track_banana_i425.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version and silently rejects RTX -> the
# annotator returns EMPTY frames. Disable the driver check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math
import os
import threading

import numpy as np
import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod

_TOP = scene_mod._TOP
_BAY_HW = scene_mod._BAY_HW
_BAY_FLOOR = scene_mod._BAY_FLOOR
_PANE_S = scene_mod._PANE_S
_PANE_T = scene_mod._PANE_T
_KNOB = scene_mod._KNOB
_DECOY_S = scene_mod._DECOY_S
_PEB = scene_mod._PEB
_TON_T = scene_mod._TON_T
_HEAD_H = scene_mod._HEAD_H
_GAP = scene_mod._GAP
_TIP_OFF = scene_mod._TIP_OFF
_TIP_LOCK = scene_mod._TIP_LOCK
_LOCK_X = scene_mod._LOCK_X
_DROP_X = scene_mod._DROP_X

_DEPOT = ((0.42, 0.30), (0.42, -0.30), (0.50, 0.30), (0.50, -0.30))

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.glazing_bench")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.95, -0.95, 0.70)) + o),
                                tuple(np.array((-0.02, 0.00, 0.05)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action)
            if (annot is not None and step_i % args.record_every == 0
                    and len(frames) < args.max_frames):
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    ever_success = [False]

    def judge() -> tuple[float, bool]:
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return s, ok

    def bench_p(body):
        return scene.to_bench(body.data.root_pos_w)[0]

    def tip_x() -> float:
        loc = scene.to_bench(scene.keeper.data.root_pos_w)
        proj = (scene.body_axis(scene.keeper, (1.0, 0.0, 0.0))
                * scene.bench_dir([1.0, 0.0, 0.0])).sum(dim=-1)
        return float((loc[:, 0] - _TIP_OFF * proj)[0])

    def report(tag: str) -> None:
        s, ok = judge()
        pl = bench_p(scene.pane)
        print(f"[smoke] {tag:18s} | out={float(scene.pebs_out_frac()[0]):.2f} "
              f"seated={bool(scene.pane_seated()[0])} "
              f"pane_z={float(pl[2]):.4f} upz={float(scene.up_z(scene.pane)[0]):+.3f} "
              f"locked={bool(scene.keeper_locked()[0])} tip={tip_x():+.4f} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def bench_pose(local, yaw_extra: float = 0.0, roll_pi: bool = False):
        """(pos, quat) world pose for a bench-frame `local` position; quat = bench
        yaw (+optional extra yaw, +optional 180 deg roll = upside-down)."""
        pos = scene.env_origins.clone()
        pos[:, 0:2] += scene.anchor
        pos += scene.bench_dir([local[0], local[1], 0.0])
        pos[:, 2] = local[2]
        yaw = scene.yaw + yaw_extra
        ch, sh = torch.cos(yaw / 2), torch.sin(yaw / 2)
        z = torch.zeros_like(ch)
        if roll_pi:  # qz(yaw) * qx(pi)
            quat = torch.stack([z, ch, sh, z], dim=-1)
        else:
            quat = torch.stack([ch, z, z, sh], dim=-1)
        return pos, quat

    def teleport(body, pos, quat, vel=None, settle_steps: int = 45) -> None:
        """Probe placement (instrumentation, not a solution) + REAL physics steps
        before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3:7] = quat
        if vel is not None:
            st[:, 7:10] = vel
        body.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def clear_pebbles() -> None:
        """Instrumentation shortcut for the honest P1 (carry each offcut out)."""
        for i in range(len(scene.pebs)):
            if bool(scene.present[0, i]):
                pos, quat = bench_pose([_DEPOT[i][0], _DEPOT[i][1], _PEB / 2 + 0.003])
                teleport(scene.pebs[i], pos, quat, settle_steps=15)
        step(60)

    def clear_wrench() -> None:
        scene.keeper.set_external_force_and_torque(zero3, zero3)

    def encode(f_w, mode: int, q_ref):
        q_now = scene.keeper.data.root_quat_w
        if mode == 0:
            return quat_apply_inverse(q_now, f_w)
        if mode == 1:
            return f_w
        if mode == 2:
            return quat_apply(quat_mul(q_ref, quat_conjugate(q_now)), f_w)
        return quat_apply_inverse(q_ref, f_w)

    def push(mag: float, mode: int, q_ref, substeps: int, stop_at: float | None = None,
             v_cap: float = 0.12) -> float:
        f_w = scene.bench_dir([-1.0, 0.0, 0.0]) * mag
        for _ in range(substeps):
            v = float(scene.keeper.data.root_lin_vel_w[0].norm())
            f = encode(f_w, mode, q_ref).view(n, 1, 3) if v < v_cap else zero3
            scene.keeper.set_external_force_and_torque(f, zero3)
            env.step(no_action)
            if stop_at is not None and tip_x() <= stop_at:
                break
        clear_wrench()
        return tip_x()

    def slide_keeper(stop_at: float, v_cap: float = 0.12) -> float:
        """GENUINE force slide: stage the keeper on the apron, calibrate the wrench
        frame from a saved state, then push until `stop_at` tip-x. Returns tip-x."""
        pos, quat = bench_pose([_DROP_X, 0.0, _TOP + _TON_T / 2 + 0.003])
        teleport(scene.keeper, pos, quat, settle_steps=60)
        t0 = tip_x()
        q_ref = scene.keeper.data.root_quat_w.clone()
        saved = scene.get_state(all_ids)
        best_mode, best_d = 0, -1.0
        for mode in range(4):
            r = push(0.55, mode, q_ref, 25)
            d = t0 - r
            scene.set_state(saved, all_ids)
            step(3)
            if d > best_d:
                best_mode, best_d = mode, d
        print(f"[smoke] slide calib: mode {best_mode} ({best_d * 1000:+.1f} mm)", flush=True)
        r = push(0.7, best_mode, q_ref, 600, stop_at=stop_at, v_cap=v_cap)
        if r > stop_at + 0.004:
            r = push(1.4, best_mode, q_ref, 400, stop_at=stop_at, v_cap=v_cap)
        clear_wrench()
        step(90)
        return tip_x()

    def settle_all(max_steps: int = 480) -> None:
        for _ in range(max_steps // 30):
            step(30)
            vmax = max(float(b.data.root_lin_vel_w[0].norm())
                       for b in [scene.pane, scene.decoy, scene.keeper, *scene.pebs])
            if vmax < 0.03:
                break

    def all_finite() -> bool:
        bodies = [scene.bench, scene.pane, scene.decoy, scene.keeper, *scene.pebs]
        return all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    settle_all(360)
    report("reset")
    s, ok = judge()
    pane_z = float(bench_p(scene.pane)[2])
    check("settle: all states finite; every present offcut fouls the seat (out_frac="
          f"{float(scene.pebs_out_frac()[0]):.2f}), pane/keeper flat on the ground",
          all_finite() and float(scene.pebs_out_frac()[0]) < 1e-6
          and pane_z < 0.02 and not bool(scene.pane_seated()[0])
          and not bool(scene.keeper_locked()[0]))
    check(f"settle: score ~0 (={s:.3f}), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in range(21, 33):
        env.reset(seed=sd)
        step(10)
        reads.append((int(scene.present[0].sum()), int(scene.pane_slot[0]),
                      float(scene.anchor[0, 0]), float(scene.anchor[0, 1]),
                      float(scene.yaw[0])))
    arr = np.array(reads)
    print("[smoke] randomization readback (n_peb, pane_slot, ax, ay, yaw):\n"
          f"{arr.round(3)}", flush=True)
    counts = {int(r[0]) for r in reads}
    slots = {int(r[1]) for r in reads}
    spread = arr.max(axis=0) - arr.min(axis=0)
    check(f"randomization: pebble count varies in [2,4] (saw {sorted(counts)}) and the "
          f"pane/decoy ground slots shuffle (saw {len(slots)} slots)",
          len(counts) >= 2 and min(counts) >= c.peb_min and max(counts) <= c.peb_max
          and len(slots) == 2)
    check("randomization: bench anchor + yaw vary (readback: "
          f"dax={spread[2]:.3f} day={spread[3]:.3f} dyaw={spread[4]:.2f})",
          spread[2] > 0.01 and spread[3] > 0.01 and spread[4] > 0.15)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(300)
    report("null-policy")
    s, ok = judge()
    check("null policy: 300 idle steps -> seat still fouled, nothing seated/locked, "
          f"score ~0 (={s:.3f}), no success", s <= 0.02 and not ok)

    # =========================== 6. seed strategy (waypoint carry) ==========================
    # pick_place/track_banana's move: carry a held object along a free-space waypoint
    # arc. Carrying the pane along an arc and setting it down changes no judged state.
    env.reset(seed=41)
    step(30)
    for wp in ([-0.20, -0.20, 0.25], [-0.05, -0.05, 0.32], [0.10, 0.05, 0.28],
               [0.18, 0.10, 0.20], [0.20, 0.10, 0.12]):
        pos, quat = bench_pose(wp)
        teleport(scene.pane, pos, quat, settle_steps=0)
        step(6)
    pos, quat = bench_pose([0.20, 0.10, _TOP + _PANE_T / 2 + 0.003])
    teleport(scene.pane, pos, quat, settle_steps=120)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy (5-waypoint free-space carry, set down on the apron): "
          f"no judged state changes — score ~0 (={s:.3f}), no success",
          s <= 0.02 and not bool(scene.pane_seated()[0]) and not ok)

    # =========================== 7. drop WITHOUT clearing rests proud =======================
    env.reset(seed=51)
    settle_all(240)
    pos, quat = bench_pose([0.0, 0.0, 0.080])
    teleport(scene.pane, pos, quat, settle_steps=240)
    report("drop-on-foul")
    s, ok = judge()
    pz = float(bench_p(scene.pane)[2])
    check("drop without clearing: the pane released over the FOULED seat genuinely "
          f"rests PROUD on the offcuts (z={pz:.4f} > flush band top {c.seat_z[1]}), "
          "no seat credit, no success",
          pz > c.seat_z[1] + 0.002 and not bool(scene.pane_seated()[0])
          and s <= 0.02 and not ok)

    # =========================== 8. straddle near miss ======================================
    env.reset(seed=61)
    settle_all(240)
    clear_pebbles()
    pos, quat = bench_pose([0.048, 0.0, 0.080])
    teleport(scene.pane, pos, quat, settle_steps=240)
    report("straddle")
    s, ok = judge()
    check("straddle near miss (seat cleared, pane released half-over the recess "
          f"edge): settles off-center/tilted (x={float(bench_p(scene.pane)[0]):+.3f}, "
          f"upz={float(scene.up_z(scene.pane)[0]):+.3f}) -> no seat credit; "
          f"score = cleared credit only (={s:.3f})",
          not bool(scene.pane_seated()[0]) and 0.13 <= s <= 0.17 and not ok)

    # =========================== 9. upside-down pane ========================================
    env.reset(seed=71)
    settle_all(240)
    clear_pebbles()
    pos, quat = bench_pose([0.0, 0.0, 0.080], roll_pi=True)
    teleport(scene.pane, pos, quat, settle_steps=240)
    report("upside-down")
    s, ok = judge()
    pz = float(bench_p(scene.pane)[2])
    check("upside-down pane (knob-down in the cleared recess): the knob props it "
          f"(z={pz:.4f}) / it cannot lie flat in the band -> no seat credit",
          not bool(scene.pane_seated()[0]) and 0.13 <= s <= 0.17 and not ok)

    # =========================== 10. wrong object (decoy refusal) ===========================
    env.reset(seed=81)
    settle_all(240)
    clear_pebbles()
    pos, quat = bench_pose([0.0, 0.0, 0.080])
    teleport(scene.decoy, pos, quat, settle_steps=240)
    report("decoy")
    s, ok = judge()
    dz = float(bench_p(scene.decoy)[2])
    check("wrong object: the GRAY decoy dropped on the cleared recess spans the "
          f"opening and rests at bench-top height (z={dz:.4f} >= {_TOP + _PANE_T / 2 - 0.004:.3f}"
          "), it can never enter; pane unseated -> no success",
          dz >= _TOP + _PANE_T / 2 - 0.004 and not bool(scene.pane_seated()[0])
          and 0.13 <= s <= 0.17 and not ok)

    # =========================== 11. out-of-order (lock first) ==============================
    env.reset(seed=91)
    settle_all(240)
    clear_pebbles()
    t_end = slide_keeper(_TIP_LOCK + 0.0015)
    locked_first = bool(scene.keeper_locked()[0])
    pos, quat = bench_pose([0.0, 0.0, 0.080])
    teleport(scene.pane, pos, quat, settle_steps=300)
    report("out-of-order")
    s, ok = judge()
    check("out-of-order FLAGSHIP: keeper genuinely force-slid home FIRST "
          f"(tip={t_end:+.4f}, locked={locked_first}), then the pane flat-dropped: "
          "the tongue overhang blocks it "
          f"(seated={bool(scene.pane_seated()[0])}, z={float(bench_p(scene.pane)[2]):.4f}, "
          f"upz={float(scene.up_z(scene.pane)[0]):+.3f}) — cleared + locked yet NO success",
          locked_first and not bool(scene.pane_seated()[0]) and not ok)

    # =========================== 12. partial slide near miss ================================
    env.reset(seed=101)
    settle_all(240)
    clear_pebbles()
    pos, quat = bench_pose([0.0, 0.0, 0.080])
    teleport(scene.pane, pos, quat, settle_steps=240)
    seated_ok = bool(scene.pane_seated()[0])
    # Stop ~10 mm of tip-travel short of the lock threshold, approaching slowly
    # (v_cap 0.05 m/s) so post-force coast (~1 mm, vs ~4.5 mm at the default cap)
    # cannot carry the tip into the guard band around the threshold.
    t_end = slide_keeper(c.lock_tip + 0.010, v_cap=0.05)
    report("partial-slide")
    s, ok = judge()
    check("partial slide near miss: pane seated honestly "
          f"(seated={seated_ok}), keeper genuinely pushed (tip {t_end:+.4f}, moved "
          f"{(_DROP_X - _TIP_OFF - t_end) * 1000:.0f} mm) but short of the overlap "
          f"threshold {c.lock_tip} -> locked() false, score capped (={s:.3f} <= 0.50)",
          seated_ok and t_end > c.lock_tip + 0.002 and t_end < _DROP_X - _TIP_OFF - 0.05
          and not bool(scene.keeper_locked()[0]) and s <= 0.501 and not ok)

    # =========================== 13. backwards keeper =======================================
    env.reset(seed=111)
    settle_all(240)
    clear_pebbles()
    pos, quat = bench_pose([_LOCK_X, 0.0, _TOP + _TON_T / 2 + 0.002], yaw_extra=math.pi)
    teleport(scene.keeper, pos, quat, settle_steps=120)
    report("backwards")
    s, ok = judge()
    align = float((scene.body_axis(scene.keeper, (1.0, 0.0, 0.0))
                   * scene.bench_dir([1.0, 0.0, 0.0])).sum(dim=-1)[0])
    check("backwards keeper: laid at the lock station rotated 180 deg (head over the "
          f"seat, align={align:+.2f}): the signed-alignment gate rejects -> not locked",
          align < 0.0 and not bool(scene.keeper_locked()[0]) and not ok)

    # =========================== 14. fly-through settle gate ================================
    env.reset(seed=121)
    settle_all(240)
    clear_pebbles()
    pos, quat = bench_pose([_LOCK_X, 0.0, _TOP + _TON_T / 2 + 0.002])
    vel = scene.bench_dir([1.0, 0.0, 0.0]) * 0.5
    teleport(scene.keeper, pos, quat, vel=vel, settle_steps=0)
    in_win, locked_mid = False, False
    for _ in range(6):
        env.step(no_action)
        in_win = in_win or tip_x() <= c.lock_tip
        locked_mid = locked_mid or bool(scene.keeper_locked()[0])
    step(90)
    report("fly-through")
    s, ok = judge()
    check("fly-through: the keeper written AT the lock pose sliding fast transits "
          f"the window (in_win={in_win}) but the settle gate never grants locked",
          in_win and not locked_mid and not ok)

    # =========================== 15. geometry + mass audit ==================================
    env.reset(seed=131)
    step(20)
    propped_lb = _BAY_FLOOR + _PEB / 2 + (_PANE_T / 2) * 0.9
    masses_ok = True
    try:
        mk = float(scene.keeper.root_physx_view.get_masses().reshape(-1)[0])
        mp = float(scene.pane.root_physx_view.get_masses().reshape(-1)[0])
        md = float(scene.decoy.root_physx_view.get_masses().reshape(-1)[0])
        print(f"[smoke] masses readback: keeper={mk:.3f} pane={mp:.3f} decoy={md:.3f}",
              flush=True)
        masses_ok = (abs(mk - c.keeper_mass) < 0.02 and abs(mp - c.pane_mass) < 0.02
                     and abs(md - c.decoy_mass) < 0.03)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] mass readback unavailable ({exc!r})", flush=True)
    check("geometry+mass audit: slot admits the tongue not the head "
          f"({_TON_T * 1000:.0f} < {_GAP * 1000:.0f} < {_HEAD_H * 1000:.0f} mm); locked "
          f"overhang {(_BAY_HW - _TIP_LOCK) * 1000:.0f} mm shrinks the opening below the "
          f"pane ({(2 * _BAY_HW - (_BAY_HW - _TIP_LOCK)) * 1000:.0f} < {_PANE_S * 1000:.0f} mm); "
          f"a propped pane rests above the flush band ({propped_lb:.4f} > {c.seat_z[1]}); "
          f"decoy {_DECOY_S * 1000:.0f} mm > recess {2 * _BAY_HW * 1000:.0f} mm; authored "
          "masses live on the compound bodies",
          _TON_T < _GAP < _HEAD_H and 2 * _BAY_HW - (_BAY_HW - _TIP_LOCK) < _PANE_S
          and propped_lb > c.seat_z[1] and _DECOY_S > 2 * _BAY_HW and masses_ok)

    # =========================== 16-17. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", all_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.glazing_bench")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(okc for _nm, okc in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, okc in checks:
            if not okc:
                print(f"[smoke]   FAILED: {nm}", flush=True)
    code = 0 if all_ok else 1
    # Hard exit: Kit teardown hangs — watchdog then die.
    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)

"""Smoke / rubric-REJECTION battery for GaugeSortScene (sim_gen task
`pick_single_egad_i216`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — gauge each bar against the slot, drop the
fitting ones through, lay the thick ones in the tray — is the acceptance evidence
that the rubric ACCEPTS a correct outcome). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) settled outcome and asserts
the rubric REJECTS it — plus physics probes that prove the gauge is real (an
oversize bar genuinely dropped onto the slot is mechanically refused; a fitting
bar dropped identically genuinely threads through). Force-style probes assert
the actuator MOVED (the refused bar demonstrably descended onto the gauge), so
no rejection is vacuous. No probe in this battery ever reaches success(), and a
final audit check asserts exactly that.

  1-2.  settle/no-NaN     — reset settles finite, all four bars resting low on
                            the open floor, in neither destination; score ~0;
  3-4.  randomization     — READBACK over 12 seeded resets: the slot gap
                            recomputed from the two plates' world poses matches
                            the sampled W, >= 2 distinct fit-count classes
                            appear (the routing decision really flips), and the
                            station anchor + yaw, scatter permutation and bar
                            poses all vary;
  5.   null policy        — 300 idle steps -> score ~0, no success;
  6.   seed strategy      — maniskill/pick_single_egad's judged end state is the
                            object held aloft (z-shift 7.5 cm) — no settled
                            analog exists; the nearest expressible state, the
                            bar at rest ELEVATED on the bin roof, is rejected by
                            the below-sill gate: score ~0, no success;
  7.   gauge refuses      — the smallest OVERSIZE bar dropped nose-down onto the
                            slot with downward velocity, three spots, genuinely
                            descends onto the plates (readback) yet is NEVER
                            in_bin at any sampled step — the aperture is real;
  8.   gauge admits       — the always-fitting bar dropped identically DOES
                            thread the slot and rests in_bin (mechanism proven
                            both ways; one bar correct, still no success);
  9.   roof rest          — a fitting bar lying ACROSS the slot on both plates
                            sits directly over the bin (xy gates alone would
                            pass) but is rejected by the z gate;
  10.  wall-hug outside   — a bar settled on the floor hugging the bin's outer
                            wall, and another beside the tray lip, are rejected
                            by the xy gates: score ~0;
  11.  dump-all-in-tray   — all four bars laid in the tray: NO gauging happened;
                            only the genuinely thick bars are credited, every
                            fitting bar in the tray is wrong, no success;
  12.  near-complete      — all bars routed correctly EXCEPT one fitting bar
                            left on the floor: score ~0.675, no success;
  13.  fly-through        — a bar written INSIDE the bin volume moving fast is
                            not credited while moving (the settled gate holds);
  14.  rejection audit    — success() never True at any judged point;
  15.  final no-NaN       — all task-object states finite.

Run (forge): python -u -m simgen_tasks.pick_single_egad_i216.smoke --headless
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
# RTX recipe: kit mis-decodes the L20 driver version and silently rejects RTX ->
# the annotator returns EMPTY frames. Disable the driver check.
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

_PL = scene_mod._PL
_BIN_C = scene_mod._BIN_C
_TRAY_C = scene_mod._TRAY_C
_TRAY_IH = scene_mod._TRAY_IH
_TRAY_FT = scene_mod._TRAY_FT
_BIN_IX = scene_mod._BIN_IX
_BIN_IY = scene_mod._BIN_IY
_WALL_T = scene_mod._WALL_T
_SILL = scene_mod._SILL
_PLATE_Z = scene_mod._PLATE_Z
_PLATE_T = scene_mod._PLATE_T
_PLATE_W = scene_mod._PLATE_W

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.gauge_sort")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.35, -0.95, 0.85)) + o),
                                tuple(np.array((0.45, 0.00, 0.10)) + o),
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

    def report(tag: str) -> None:
        s, ok = judge()
        loc = scene.bar_local()[0]
        f = scene.fits()[0]
        print(f"[smoke] {tag:16s} | W={float(scene.W[0]) * 1000:.1f}mm "
              f"fits={f.tolist()} correct={scene.correct()[0].tolist()} "
              f"z={[round(float(v), 3) for v in loc[:, 2]]} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def station_pose(local, world_yaw_off: float = 0.0, flat: bool = False):
        """(pos, quat) world pose for a station-frame `local` position; quat =
        station yaw + offset, upright (long axis vertical) or lying flat."""
        pos = scene.env_origins.clone()
        pos[:, 0:2] += scene.anchor
        pos += scene.station_dir([local[0], local[1], 0.0])
        pos[:, 2] = local[2]
        u = scene.yaw + world_yaw_off
        ch, sh = torch.cos(u / 2), torch.sin(u / 2)
        z = torch.zeros_like(ch)
        if flat:
            c45 = math.sqrt(0.5)
            quat = torch.stack([ch * c45, ch * c45, sh * c45, sh * c45], dim=-1)
        else:
            quat = torch.stack([ch, z, z, sh], dim=-1)
        return pos, quat

    def teleport(body, pos, quat, vel=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL
        physics steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3:7] = quat
        if vel is not None:
            st[:, 7:10] = vel
        body.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def settle_all(max_steps: int = 480) -> None:
        for _ in range(max_steps // 30):
            step(30)
            vmax = max(float(b.data.root_lin_vel_w[0].norm()) for b in scene.bars)
            if vmax < 0.05:
                break

    def all_finite() -> bool:
        bodies = [scene.station, *scene.plates, *scene.bars]
        return all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)

    def tray_slot(i: int):
        """A per-bar spot in the 2x2 tray layout (length along station x)."""
        xs = (-0.0485, 0.0485)
        ys = (-0.045, 0.045)  # wide pair (c, d) in one row, narrow (a, b) in the other
        row = 0 if i >= 2 else 1
        col = i % 2
        return _TRAY_C[0] + xs[col], _TRAY_C[1] + ys[row]

    parts = c.parts

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    settle_all(360)
    report("reset")
    loc = scene.bar_local()[0]
    on_floor = bool((loc[:, 2] < 0.05).all())
    in_dest = bool((scene.in_bin()[0] | scene.in_tray()[0]).any())
    check("settle: all states finite, all four bars resting LOW on the open floor "
          f"(z={[round(float(v), 3) for v in loc[:, 2]]}), in neither destination",
          all_finite() and on_floor and not in_dest)
    s, ok = judge()
    check(f"settle: score ~0 (score={s:.3f}), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in range(21, 33):
        env.reset(seed=sd)
        step(10)
        w_samp = float(scene.W[0])
        # recompute the slot gap from the two plates' WORLD poses (readback)
        ps = scene.to_station(scene.plates[0].data.root_pos_w)[0]
        pn = scene.to_station(scene.plates[1].data.root_pos_w)[0]
        w_read = float(pn[1] - ps[1]) - _PLATE_W
        nfit = int(scene.fits()[0].sum())
        ba = scene.bar_local()[0, 0]
        reads.append((w_samp, w_read, nfit, float(scene.anchor[0, 0]),
                      float(scene.anchor[0, 1]), float(scene.yaw[0]),
                      tuple(int(v) for v in scene.slot_perm[0]),
                      float(ba[0]), float(ba[1]), float(pn[2])))
    arr = np.array([[r[0], r[1], r[2], r[3], r[4], r[5], r[7], r[8], r[9]]
                    for r in reads])
    print("[smoke] randomization readback (W, W_readback, n_fit, ax, ay, yaw, "
          f"bar_a_x, bar_a_y, plate_z):\n{arr.round(4)}", flush=True)
    gap_err = float(np.abs(arr[:, 0] - arr[:, 1]).max())
    plate_z_err = float(np.abs(arr[:, 8] - _PLATE_Z).max())
    fit_classes = {r[2] for r in reads}
    perms = {r[6] for r in reads}
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: the slot gap recomputed from the plates' world poses "
          f"matches the sampled W (max err {gap_err * 1000:.2f} mm), plates at the "
          f"sill (z err {plate_z_err * 1000:.2f} mm), and >= 2 fit-count classes "
          f"appear over 12 seeds (classes={sorted(fit_classes)})",
          gap_err < 0.003 and plate_z_err < 0.003 and len(fit_classes) >= 2)
    check("randomization: station anchor + yaw, scatter permutation and bar poses "
          f"vary (dax={spread[3]:.3f} dyaw={spread[5]:.2f} dbar={spread[6]:.3f}, "
          f"{len(perms)} distinct perms)",
          spread[3] > 0.01 and spread[5] > 0.15 and spread[6] > 0.03
          and len(perms) >= 3)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(300)
    report("null-policy")
    s, ok = judge()
    check(f"null policy: 300 idle steps -> score ~0 (score={s:.3f}), no success",
          s <= 0.02 and not ok)

    # =========================== 6. seed strategy (lift) ====================================
    # maniskill/pick_single_egad's judged end state is the object HELD ALOFT
    # (z-shift 7.5 cm) — a held state has no settled analog. The nearest
    # expressible state is the bar at rest ELEVATED on the bin roof: settled,
    # high, and directly over the bin — the below-sill gate must reject it.
    env.reset(seed=41)
    step(30)
    t_a = parts[0][1]
    wgap = float(scene.W[0])
    pos, quat = station_pose(
        [_BIN_C[0], _BIN_C[1] + wgap / 2 + 0.030, _SILL + _PLATE_T + t_a / 2 + 0.003],
        world_yaw_off=math.pi / 2, flat=True)
    teleport(scene.bars[0], pos, quat, settle_steps=90)
    report("seed-lift")
    loc = scene.bar_local()[0, 0]
    s, ok = judge()
    check("seed strategy (pick_single_egad lifts the object 7.5 cm): the bar at "
          f"rest ELEVATED on the bin roof (z={float(loc[2]):.3f} > 0.14, settled="
          f"{bool(scene.settled()[0, 0])}) earns nothing — the below-sill gate "
          f"rejects it: not in_bin, score ~0 (score={s:.3f}), no success",
          float(loc[2]) > 0.14 and bool(scene.settled()[0, 0])
          and not bool(scene.in_bin()[0, 0]) and s <= 0.02 and not ok)

    # =========================== 7. the gauge refuses an oversize bar =======================
    env.reset(seed=51)
    settle_all(240)
    fits = scene.fits()[0].tolist()
    idx_thick = fits.index(False)  # smallest oversize bar: hardest refusal case
    name_thick = parts[idx_thick][0]
    hover_z = _PLATE_Z + _PLATE_T / 2 + _PL / 2 + 0.005
    moved_down = False
    ever_in_bin = False
    min_z = float("inf")
    for xoff in (0.05, 0.0, -0.05):
        pos, quat = station_pose([_BIN_C[0] + xoff, _BIN_C[1], hover_z])
        vel = torch.zeros(n, 3, device=device)
        vel[:, 2] = -0.5
        teleport(scene.bars[idx_thick], pos, quat, vel=vel, settle_steps=0)
        for _ in range(150):
            step(1)
            z_now = float(scene.bar_local()[0, idx_thick, 2])
            min_z = min(min_z, z_now)
            moved_down = moved_down or z_now < hover_z - 0.003
            ever_in_bin = ever_in_bin or bool(scene.in_bin()[0, idx_thick])
    step(90)
    report("gauge-refuse")
    s, ok = judge()
    check(f"gauge refuses oversize: {name_thick} (t={parts[idx_thick][1] * 1000:.0f}mm "
          f"> gap {float(scene.W[0]) * 1000:.1f}mm) dropped nose-down onto the slot "
          f"with downward velocity at 3 spots genuinely descended onto the gauge "
          f"(moved_down={moved_down}, min_z={min_z:.3f}) yet was NEVER in_bin at "
          "any sampled step — the aperture is real, no success",
          moved_down and not ever_in_bin and not bool(scene.in_bin()[0, idx_thick])
          and not ok)

    # =========================== 8. the gauge admits a fitting bar ==========================
    # Same drop with the always-fitting bar: it must genuinely thread the slot.
    # First clear the refused bar off the roof (it rests wedged over the slot),
    # or the admit drop would land on IT instead of the gauge.
    pos, quat = station_pose([-0.10, 0.0, parts[idx_thick][1] / 2 + 0.005],
                             world_yaw_off=math.pi / 2, flat=True)
    teleport(scene.bars[idx_thick], pos, quat, settle_steps=60)
    seated = False
    for attempt in range(4):
        pos, quat = station_pose([_BIN_C[0] - 0.05, _BIN_C[1], hover_z])
        vel = torch.zeros(n, 3, device=device)
        vel[:, 2] = -0.4 if attempt else 0.0
        teleport(scene.bars[0], pos, quat, vel=vel, settle_steps=0)
        for _ in range(8):
            step(30)
            if bool(scene.in_bin()[0, 0]) and bool(scene.settled()[0, 0]):
                break
        if bool(scene.in_bin()[0, 0]):
            seated = True
            break
    report("gauge-admit")
    s, ok = judge()
    check("gauge admits fitting: bar_a dropped identically DOES thread the slot "
          f"and rests in_bin (seated={seated}) — the mechanism is proven both "
          f"ways; one bar correct (score={s:.3f} <= 0.24), still no success",
          seated and s <= 0.24 and not ok)

    # =========================== 9. roof rest across the slot ===============================
    env.reset(seed=61)
    step(30)
    fits = scene.fits()[0].tolist()
    idx_fit = max(i for i, f in enumerate(fits) if f)  # widest fitting bar
    t_f = parts[idx_fit][1]
    # lying flat ACROSS the slot (length along station y): rests on both plates
    pos, quat = station_pose([_BIN_C[0], _BIN_C[1], _SILL + _PLATE_T + t_f / 2 + 0.004],
                             flat=True)
    teleport(scene.bars[idx_fit], pos, quat, settle_steps=90)
    report("roof-rest")
    loc = scene.bar_local()[0, idx_fit]
    xy_pass = (abs(float(loc[0]) - _BIN_C[0]) < c.bin_xy_x
               and abs(float(loc[1]) - _BIN_C[1]) < c.bin_xy_y)
    s, ok = judge()
    check(f"roof rest: fitting bar {parts[idx_fit][0]} lying ACROSS the slot on "
          f"both plates sits directly over the bin (xy gates alone pass={xy_pass}) "
          f"but z={float(loc[2]):.3f} > gate {c.bin_z} -> not in_bin, no credit "
          f"(score={s:.3f}), no success",
          xy_pass and float(loc[2]) > c.bin_z + 0.02
          and not bool(scene.in_bin()[0, idx_fit]) and s <= 0.02 and not ok)

    # =========================== 10. wall-hug + beside-tray near misses =====================
    env.reset(seed=71)
    step(30)
    t_a, w_a = parts[0][1], parts[0][2]
    w_d = parts[3][2]
    # bar_a on the floor hugging the bin's outer side wall (length along x)
    hug_y = _BIN_C[1] + _BIN_IY + _WALL_T + t_a / 2 + 0.002
    pos, quat = station_pose([_BIN_C[0], hug_y, w_a / 2 + 0.003],
                             world_yaw_off=math.pi / 2, flat=True)
    teleport(scene.bars[0], pos, quat, settle_steps=0)
    # bar_d on the floor just beyond the tray lip
    out_y = _TRAY_C[1] + _TRAY_IH + _WALL_T + w_d / 2 + 0.004
    pos, quat = station_pose([_TRAY_C[0], out_y, parts[3][1] / 2 + 0.003],
                             world_yaw_off=math.pi / 2, flat=True)
    teleport(scene.bars[3], pos, quat, settle_steps=90)
    report("near-miss-xy")
    la = scene.bar_local()[0, 0]
    ld = scene.bar_local()[0, 3]
    s, ok = judge()
    check("wall-hug near misses: bar_a settled LOW against the bin's outer wall "
          f"(y offset {abs(float(la[1]) - _BIN_C[1]) * 1000:.0f}mm) is not in_bin, "
          f"and bar_d settled beside the tray lip (y offset "
          f"{abs(float(ld[1]) - _TRAY_C[1]) * 1000:.0f}mm) is not in_tray — the xy "
          f"gates hold; score ~0 (score={s:.3f}), no success",
          float(la[2]) < 0.05 and not bool(scene.in_bin()[0, 0])
          and float(ld[2]) < 0.06 and not bool(scene.in_tray()[0, 3])
          and s <= 0.02 and not ok)

    # =========================== 11. dump everything in the tray ============================
    env.reset(seed=81)
    step(30)
    fits = scene.fits()[0].tolist()
    n_thick = sum(1 for f in fits if not f)
    for i, (name, t, w) in enumerate(parts):
        sx, sy = tray_slot(i)
        pos, quat = station_pose([sx, sy, _TRAY_FT + t / 2 + 0.010],
                                 world_yaw_off=math.pi / 2, flat=True)
        teleport(scene.bars[i], pos, quat, settle_steps=0)
    settle_all(300)
    report("dump-in-tray")
    all_in_tray = bool(scene.in_tray()[0].all())
    fit_credited = bool((scene.correct()[0] & scene.fits()[0]).any())
    s, ok = judge()
    s_want = 0.9 * n_thick / 4
    check("dump-all-in-tray: all four bars laid in the tray (all in_tray="
          f"{all_in_tray}) — NO gauging happened; only the {n_thick} genuinely "
          f"thick bar(s) are credited, every fitting bar in the tray is wrong "
          f"(score={s:.3f} ~ {s_want:.3f}), no success",
          all_in_tray and not fit_credited and abs(s - s_want) < 0.03 and not ok)

    # =========================== 12. near-complete misses success ===========================
    env.reset(seed=91)
    step(30)
    fits = scene.fits()[0].tolist()
    fit_idx = [i for i, f in enumerate(fits) if f]
    extra_fit = [i for i in fit_idx if i != 0]  # route these into the bin
    bin_ys = (-0.03, 0.03)
    for j, i in enumerate(extra_fit):
        pos, quat = station_pose([_BIN_C[0], _BIN_C[1] + bin_ys[j % 2],
                                  parts[i][1] / 2 + 0.010],
                                 world_yaw_off=math.pi / 2, flat=True)
        teleport(scene.bars[i], pos, quat, settle_steps=0)
    for i, f in enumerate(fits):
        if not f:
            sx, sy = tray_slot(i)
            pos, quat = station_pose([sx, sy, _TRAY_FT + parts[i][1] / 2 + 0.010],
                                     world_yaw_off=math.pi / 2, flat=True)
            teleport(scene.bars[i], pos, quat, settle_steps=0)
    settle_all(300)  # bar_a (fitting) stays on the open floor
    report("near-complete")
    s, ok = judge()
    n_corr = int(scene.correct()[0].sum())
    check("near-complete: every bar routed correctly EXCEPT bar_a (fitting) left "
          f"on the floor -> {n_corr}/4 correct, score={s:.3f} ~ 0.675, NO success "
          "— the last routing is load-bearing",
          n_corr == 3 and 0.655 <= s <= 0.695 and not ok)

    # =========================== 13. fly-through settle gate ================================
    env.reset(seed=101)
    step(30)
    pos, quat = station_pose([_BIN_C[0] - 0.04, _BIN_C[1], 0.055],
                             world_yaw_off=math.pi / 2, flat=True)
    vel = scene.station_dir([1.2, 0.0, 0.0])
    teleport(scene.bars[0], pos, quat, vel=vel, settle_steps=0)
    step(2)
    in_bin_now = bool(scene.in_bin()[0, 0])
    settled_now = bool(scene.settled()[0, 0])
    corr_now = bool(scene.correct()[0, 0])
    step(150)
    report("fly-through")
    check("fly-through: a fitting bar written INSIDE the bin volume moving at "
          f"1.2 m/s is inside (in_bin={in_bin_now}) but NOT credited while moving "
          f"(settled={settled_now}, correct={corr_now}) — the settle gate holds",
          in_bin_now and not settled_now and not corr_now)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    check("final: all task-object states finite (no NaN)", all_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.gauge_sort")
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

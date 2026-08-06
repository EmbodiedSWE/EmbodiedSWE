"""Smoke / rubric-REJECTION battery for MilkCarouselScene — NullRobot, teleported probes.

solve.py is the acceptance proof (it demonstrates the rubric accepts a correct outcome
and that latched credit is monotone along a real trajectory). This battery proves the
rubric REJECTS wrong outcomes, and that the roof interlock the task's ordering claim
rests on is physically load-bearing. Every probe is CONSTRUCTED as a settled state
(teleport, real physics steps, judge) — instrumentation, never a solution: no probe
here reaches success().

Checks:
   1. settle/no-NaN      — reset layout settles finite; milk starts in a COVERED bay
                           (in the dispenser, outside the aligned gate); score 0;
   2. randomization      — two seeded resets: READBACK rotor yaw, milk sector error,
                           sector azimuth, pad position all differ;
   3. bay permutation    — milk's bay index varies across 10 resets, and the start
                           offset occurs on both sides of the sector;
   4. null-policy        — 240 idle steps: score ~0, no success (the milk is not even
                           graspable at reset — nothing happens by itself);
   5. SEED STRATEGY      — the seed's plan ("put the milk in the container"): milk
                           settled INSIDE the gray bin -> rejected, score <= 0.50;
   6. wrong object       — the ORANGE JUICE carton stood perfectly on the pad, milk
                           untouched -> score ~0, no success;
   7. near-miss (pad xy) — milk upright, settled, 60 mm off the pad center (gate
                           45 mm) -> not delivered, no success;
   8. near-miss (upright)— milk LYING on its side centered on the pad -> rejected;
   9. INTERLOCK covered  — 3x-gravity lift force on the milk in a COVERED bay for
                           1.5 s: the roof pins it below the freed plane, it never
                           leaves the dispenser (the "rotate first" ordering is
                           physics, not rubric fiat);
  10. INTERLOCK aligned  — the same force with the bay CONSTRUCTED at the open
                           sector: the carton rises past the roof plane (the opening
                           is real); the carton is then parked on the floor off-pad;
  11. latch persistence  — after probe 10 the aligned+freed latches hold with the
                           carton parked on the floor: earned credit (0.55) does not
                           evaporate, and it is still not success;
  12. aligned-gate honesty — constructed 40 deg offset -> aligned_now False;
                           constructed 10 deg offset -> aligned_now True;
  13. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_pick_milk_i3.smoke --headless
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


def _settle_until(pred, max_steps: int = 300, poll: int = 15) -> bool:
    _step(poll)  # ALWAYS force real steps first (zero-step teleport trap)
    if pred():
        return True
    waited = poll
    while waited < max_steps:
        _step(poll)
        waited += poll
        if pred():
            return True
    return False


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    print(f"[smoke] {tag:18s} | err={math.degrees(float(scene.milk_sector_err()[0])):6.1f}deg "
          f"in_disp={bool(scene.in_dispenser()[0])} aligned={bool(scene.aligned_now()[0])} "
          f"freed_l={bool(scene._freed[0])} milk_z={float(scene.milk.data.root_pos_w[0, 2]):.3f} "
          f"delivered={bool(scene.delivered()[0])} score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}", flush=True)


def _pad_top_pose() -> torch.Tensor:
    scene = _ENV.scene
    _refresh()
    pp = scene.pad.data.root_pos_w.clone()
    pp[:, 2] += scene.cfg.pad_t / 2
    return pp


def _construct_offset(offset_deg: float) -> None:
    """CONSTRUCT a carousel state with the milk bay at `offset_deg` from the sector
    centre: follower-only rotor yaw write about the unchanged spindle, then all three
    items re-dealt into their bays consistently (mirrors reset), then settled."""
    scene = _ENV.scene
    c = scene.cfg
    n = _ENV.num_envs
    dev = _ENV.device
    _refresh()
    sec = scene.sector_az.clone()
    bay = scene.milk_bay
    theta = sec + math.radians(offset_deg) - bay.float() * math.radians(120.0)
    hub = scene._hub_xy()

    def yaw_quat(a: torch.Tensor) -> torch.Tensor:
        q = torch.zeros(n, 4, device=dev)
        q[:, 0], q[:, 3] = torch.cos(a / 2), torch.sin(a / 2)
        return q

    p = torch.zeros(n, 3, device=dev)
    p[:, 0:2] = hub
    p[:, 2] = c.disc_z
    _write_body(scene.rotor, p, yaw_quat(theta))
    for body, b_idx, half_h in ((scene.milk, bay, c.milk_h / 2),
                                (scene.oj, (bay + 1) % 3, c.oj_h / 2),
                                (scene.can, (bay + 2) % 3, c.can_h / 2)):
        az = theta + b_idx.float() * math.radians(120.0)
        ip = torch.zeros(n, 3, device=dev)
        ip[:, 0] = hub[:, 0] + c.bay_r * torch.cos(az)
        ip[:, 1] = hub[:, 1] + c.bay_r * torch.sin(az)
        ip[:, 2] = c.disc_top + half_h + 0.003
        _write_body(body, ip, yaw_quat(az))
    _step(30)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.milk_carousel")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.95, -0.85, 0.80)) + o),
                                tuple(np.array((0.05, 0.00, 0.12)) + o),
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

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(60)
    _report("show")
    _REC["on"] = False
    states = torch.cat([b.data.root_state_w for b in
                        (scene.rotor, scene.roof, scene.milk, scene.oj, scene.can,
                         scene.pad, scene.bin)], dim=-1)
    check("settle/no-NaN: layout settles finite; milk starts in a COVERED bay "
          "(in dispenser, not aligned); score 0",
          bool(torch.isfinite(states).all()) and bool(scene.in_dispenser()[0])
          and not bool(scene.aligned_now()[0]) and float(scene.score()[0]) == 0.0
          and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (yaw_of(scene.rotor.data.root_quat_w[0]),
                math.degrees(float(scene.milk_sector_err()[0])),
                math.degrees(float(scene.sector_az[0])),
                scene.pad.data.root_pos_w[0, :2].clone())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_ry, a_err, a_sec, a_pp = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_ry, b_err, b_sec, b_pp = readback()
    d_ry, d_err = dyaw(a_ry, b_ry), abs(a_err - b_err)
    d_sec, d_pp = abs(a_sec - b_sec), float((a_pp - b_pp).norm())
    print(f"[smoke] randomization deltas: rotor_yaw={d_ry:.1f}deg milk_err={d_err:.1f}deg "
          f"sector={d_sec:.1f}deg pad={d_pp * 1000:.1f}mm", flush=True)
    check("randomization-is-real: rotor yaw, milk sector error, sector azimuth, pad "
          "readback differ",
          d_ry > 3.0 and d_err > 3.0 and d_sec > 1.0 and d_pp > 0.005)

    # ================= 3. bay permutation + offset sign vary ======================================
    bays, signs = set(), set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        bays.add(int(scene.milk_bay[0]))
        rel = scene.milk.data.root_pos_w[0, :2] - scene._hub_xy()[0]
        phi = math.atan2(float(rel[1]), float(rel[0]))
        d = (phi - float(scene.sector_az[0]) + math.pi) % (2 * math.pi) - math.pi
        signs.add(d > 0)
    print(f"[smoke] milk bays over 10 resets: {sorted(bays)}, offset signs: {signs}",
          flush=True)
    check("bay permutation: milk bay index varies and start offsets occur on both sides",
          len(bays) >= 2 and signs == {True, False})

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. negative: the SEED'S OWN STRATEGY =======================================
    # "Pick the milk and place it in the container" — the gray bin IS the container on
    # offer. A carton settled inside the bin must be rejected.
    torch.manual_seed(100)
    env.reset()
    _step(40)
    _REC["on"] = True
    _refresh()
    bp = scene.bin.data.root_pos_w.clone()
    bp[:, 2] = c.bin_bot_t + c.milk_h / 2 + 0.006
    _write_body(scene.milk, bp)
    _step(120)
    _report("seed-strategy")
    check("negative (SEED strategy): milk settled INSIDE the gray bin — no success, "
          "score <= 0.50",
          not bool(scene.delivered()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.50)
    _REC["on"] = False

    # ================= 6. wrong object on the pad =================================================
    torch.manual_seed(100)
    env.reset()
    _step(40)
    pt = _pad_top_pose()
    p = pt.clone()
    p[:, 2] += c.oj_h / 2 + 0.004
    _write_body(scene.oj, p)
    _step(90)
    _report("wrong-object")
    check("negative (wrong object): juice carton stood perfectly on the pad, milk "
          "untouched — score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 7. near-miss: milk upright but 60 mm off the pad center ====================
    torch.manual_seed(100)
    env.reset()
    _step(40)
    p = _pad_top_pose()
    p[:, 1] += c.pad_tol + 0.015  # 60 mm: just past the 45 mm gate
    p[:, 2] += c.milk_h / 2 + 0.004
    _write_body(scene.milk, p)
    _step(90)
    _report("milk-off-pad")
    check("negative (near-miss pad xy): milk upright, settled, 60 mm off the pad "
          "center — not delivered, no success",
          not bool(scene.delivered()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.50)

    # ================= 8. near-miss: milk lying flat ON the pad ===================================
    torch.manual_seed(100)
    env.reset()
    _step(40)
    p = _pad_top_pose()
    p[:, 2] += c.milk_w / 2 + 0.004
    q = torch.zeros(env.num_envs, 4, device=env.device)
    c45 = math.cos(math.pi / 4)
    q[:, 0], q[:, 2] = c45, c45  # pitched 90 deg: lying on a side face
    _write_body(scene.milk, p, q)
    _step(90)
    _report("milk-lying")
    check("negative (near-miss upright): milk lying on its side ON the pad — rejected "
          "by the upright gate, no success",
          not bool(scene.delivered()[0]) and not bool(scene.success()[0]))

    # ================= 9. INTERLOCK: covered bay defeats a 3x-gravity lift ========================
    torch.manual_seed(100)
    env.reset()
    _step(40)
    _REC["on"] = True
    _refresh()
    print(f"[smoke] interlock-covered: start err="
          f"{math.degrees(float(scene.milk_sector_err()[0])):.1f}deg", flush=True)
    scene.milk_force[:, 2] = 3.0 * c.milk_mass * 9.81
    z_max, r_at_zmax = 0.0, 0.0
    for _ in range(90):
        _step(2)
        _refresh()
        z = float(scene.milk.data.root_pos_w[0, 2])
        if z > z_max:
            z_max = z
            r_at_zmax = float((scene.milk.data.root_pos_w[0, :2]
                               - scene._hub_xy()[0]).norm())
    scene.milk_force[:] = 0.0
    _step(90)
    _report("interlock-covered")
    print(f"[smoke] covered-bay lift: z_max={z_max:.3f} (roof underside "
          f"{c.roof_under:.3f}, freed plane {c.freed_z:.3f}), r={r_at_zmax:.3f}", flush=True)
    check("INTERLOCK (covered): 3x-gravity lift for 1.5 s never clears the roof — "
          "milk stays below the freed plane, inside the dispenser, freed latch False",
          z_max < c.freed_z - 0.01 and bool(scene.in_dispenser()[0])
          and not bool(scene._freed[0]) and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 10. INTERLOCK: the aligned bay lets the same force out =====================
    torch.manual_seed(100)
    env.reset()
    _step(40)
    _REC["on"] = True
    _construct_offset(0.0)  # milk bay parked at the sector centre (constructed)
    _report("aligned-built")
    scene.milk_force[:, 2] = 3.0 * c.milk_mass * 9.81
    rose = False
    for _ in range(120):
        _step(2)
        _refresh()
        if float(scene.milk.data.root_pos_w[0, 2]) > 0.32:
            rose = True
            break
    scene.milk_force[:] = 0.0
    _refresh()
    check("INTERLOCK (aligned): the same 3x-gravity lift rises past the roof plane "
          "through the open sector (the opening is real)",
          rose and bool(scene._freed[0]))
    # park the carton on the floor, far from pad AND bin — a probe never reaches success
    park = torch.zeros(env.num_envs, 3, device=env.device)
    park[:, 0] = env.iscene.env_origins[:, 0] + 0.45
    park[:, 1] = env.iscene.env_origins[:, 1] + 0.35
    park[:, 2] = c.milk_h / 2 + 0.004
    _write_body(scene.milk, park)
    _step(90)
    _report("parked")
    _REC["on"] = False

    # ================= 11. latch persistence ======================================================
    check("latch persistence: aligned+freed credit (0.55) holds with the carton parked "
          "on the floor; still no success",
          abs(float(scene.score()[0]) - 0.55) < 1e-4 and not bool(scene.success()[0]))

    # ================= 12. aligned-gate honesty ===================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _construct_offset(40.0)  # outside the 25 deg gate
    out_ok = not bool(scene.aligned_now()[0]) and bool(scene.in_dispenser()[0])
    err_out = math.degrees(float(scene.milk_sector_err()[0]))
    _construct_offset(10.0)  # inside the gate
    in_ok = bool(scene.aligned_now()[0])
    err_in = math.degrees(float(scene.milk_sector_err()[0]))
    print(f"[smoke] aligned gate: 40deg-> err={err_out:.1f} rejected={out_ok} | "
          f"10deg-> err={err_in:.1f} accepted={in_ok}", flush=True)
    check("aligned-gate honesty: constructed 40 deg offset rejected, 10 deg accepted",
          out_ok and in_ok)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.milk_carousel")
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

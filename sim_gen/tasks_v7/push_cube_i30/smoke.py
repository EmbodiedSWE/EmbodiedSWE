"""Smoke / rubric-REJECTION battery for SiloScoopScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the latched
credit is monotone along a real trajectory). This battery proves the rubric REJECTS
wrong outcomes and that the claims the task rests on are load-bearing. Every probe is
CONSTRUCTED (teleport, real physics steps, judge) — instrumentation, never a solution:
success() is monitored at EVERY step and must never turn True anywhere in the battery
(the audit is itself a check).

Checks:
   1. settle/no-NaN     — seeded reset settles finite; present balls IN the silo,
                          scoop lying low on open floor; score ~0, no success;
   2. randomization     — two seeded resets: READBACK silo xy+yaw, tray xy+yaw,
                          scoop xy and ball positions all differ;
   3. present coverage  — over seeded resets the present-ball count takes BOTH
                          values (1 and 2): the judged subset really varies;
   4. null-policy       — 300 idle steps: score ~0, no success;
   5. SEED STRATEGY     — the seed's plan is a planar PUSH of the payload along the
                          support toward the goal. The balls are unreachable, so the
                          only pushable thing is the scoop: shove it across the floor
                          to the tray (and shove the silo with 30 N — bolted, it must
                          not move) -> balls still in the silo, score ~0, no success;
   6. scoop-in-tray     — both present balls dropped INTO the tray but the scoop
                          dropped in there too: delivered, NOT stowed -> no success,
                          score capped at 0.70;
   7. stow near-miss    — scoop relaid right against the silo (origin inside the
                          stow_clear ring): still not stowed -> no success;
   8. latched credit    — the delivered balls teleported BACK into the silo: live
                          in_basin gone, but the latched 0.70 survives; no success;
   9. floor-drop        — present balls dropped on OPEN FLOOR (out of the silo, not
                          in the tray): escape credit only, score <= 0.30, no success;
  10. subset judging    — a k=1 seed: delivering ONLY the present ball satisfies the
                          delivered clause (absent ball parked in the depot neither
                          blocks nor inflates: score exactly 0.70; scoop parked
                          non-stowed so success stays False);
  11. ball-in-pan       — a ball resting IN THE PAN of the scoop on open floor is not
                          delivery: not in_basin, score <= 0.30, no success;
  12. settle gate       — delivered balls + a scoop in a perfect stow pose but still
                          SLIDING at ~0.45 m/s: success refuses while anything moves
                          (probe dismantled before it can settle into a real success);
  13. rejection audit   — success() was never True at any step of this battery;
  14. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.push_cube_i30.smoke --headless
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
    from . import scene as task_scene  # noqa: F401  (registers)
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() ----------------------------------------------------------
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


def _place(body, xy, z: float, quat: torch.Tensor | None = None,
           lin_vel=None) -> None:
    """Teleport `body` to env-local (xy, z) with zero (or given) velocity."""
    n, dev = _ENV.num_envs, _ENV.device
    st = torch.zeros(n, 13, device=dev)
    st[:, 0] = float(xy[0])
    st[:, 1] = float(xy[1])
    st[:, 2] = float(z)
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat.view(1, 4)
    if lin_vel is not None:
        st[:, 7] = float(lin_vel[0])
        st[:, 8] = float(lin_vel[1])
    st[:, 0:3] += _ENV.scene.env_origins
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def main() -> None:  # noqa: PLR0915
    global _ENV  # noqa: PLW0603
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.silo_scoop")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    origin0 = scene.env_origins[0]

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = scene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.55, -0.35, 0.85)) + o),
                                tuple(np.array((0.38, 0.00, 0.10)) + o),
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

    def yaw_of(qrow: torch.Tensor) -> float:
        return math.degrees(2.0 * math.atan2(float(qrow[3]), float(qrow[0])))

    def dyaw(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    def report(tag: str) -> None:
        _refresh()
        sp = (scene.scoop.data.root_pos_w - scene.env_origins)[0]
        print(f"[smoke] {tag:16s} | present={scene.present[0].tolist()} "
              f"in_silo={scene.in_silo()[0].tolist()} "
              f"in_basin={scene.in_basin()[0].tolist()} "
              f"scoop=({float(sp[0]):+.3f},{float(sp[1]):+.3f},{float(sp[2]):+.3f}) "
              f"stowed={bool(scene.scoop_stowed()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"score={float(scene.score()[0]):.3f} "
              f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
              flush=True)

    def finite() -> bool:
        ok = bool(torch.isfinite(scene.scoop.data.root_state_w).all())
        for b in scene.balls:
            ok = ok and bool(torch.isfinite(b.data.root_state_w).all())
        return ok

    def silo_xy() -> torch.Tensor:
        _refresh()
        return (scene.silo.data.root_pos_w[0, :2] - origin0[:2]).clone()

    def basin_xy() -> torch.Tensor:
        _refresh()
        return (scene.basin.data.root_pos_w[0, :2] - origin0[:2]).clone()

    def lying_q(yaw: float) -> torch.Tensor:
        """qz(yaw) x qy(-90 deg): the scoop lying on its back-wall face (as reset)."""
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        c45 = math.cos(math.pi / 4)
        return torch.tensor([cy * c45, sy * c45, -cy * c45, sy * c45], device=device)

    def lay_scoop(xy, away_from) -> None:
        """Lay the scoop flat at env-local `xy`, handle pointing away from
        `away_from` (so only the pan end is near the fixture)."""
        d = torch.tensor([float(xy[0]) - float(away_from[0]),
                          float(xy[1]) - float(away_from[1])])
        yaw = math.atan2(-float(d[1]), -float(d[0]))  # handle = -x(yaw) direction
        _place(scene.scoop, xy, -c.back_x + 0.004, lying_q(yaw))
        _step(60)

    def drop_balls_into_tray(idx: list[int]) -> None:
        """Contact delivery: teleport each ball to a free hover over the tray and
        let it FALL in (the landing itself is real physics)."""
        bxy = basin_xy()
        for k_, i in enumerate(idx):
            off = (-1) ** k_ * 0.040
            _place(scene.balls[i], (float(bxy[0]) + off, float(bxy[1])), 0.15)
            _step(90)
        _step(60)

    def delivered() -> bool:
        return bool((scene.in_basin() | ~scene.present).all(dim=1)[0])

    def pres_idx() -> list[int]:
        return [i for i in range(c.n_balls) if bool(scene.present[0, i])]

    # ================= 1. settle / no-NaN =======================================================
    env.reset(seed=11)
    _AUD["on"] = True
    _REC["on"] = True
    _step(90)
    report("settle")
    _REC["on"] = False
    balls_in = all(bool(scene.in_silo()[0, i]) for i in pres_idx())
    sp_z = float(scene.scoop.data.root_pos_w[0, 2] - origin0[2])
    check("settle/no-NaN: seeded reset settles finite; present balls IN the silo, "
          "scoop lying low; score ~0, no success",
          finite() and balls_in and sp_z < 0.10
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ======================================
    def readback():
        _refresh()
        return (silo_xy(), yaw_of(scene.silo.data.root_quat_w[0]),
                basin_xy(), yaw_of(scene.basin.data.root_quat_w[0]),
                (scene.scoop.data.root_pos_w[0, :2] - origin0[:2]).clone(),
                torch.cat([b.data.root_pos_w[0, :2] - origin0[:2]
                           for b in scene.balls]).clone())

    env.reset(seed=21)
    _step(10)
    a_sp, a_sy, a_bp, a_by, a_cp, a_bl = readback()
    env.reset(seed=22)
    _step(10)
    b_sp, b_sy, b_bp, b_by, b_cp, b_bl = readback()
    d_sp, d_bp = float((a_sp - b_sp).norm()), float((a_bp - b_bp).norm())
    d_cp, d_bl = float((a_cp - b_cp).norm()), float((a_bl - b_bl).norm())
    d_sy, d_by = dyaw(a_sy, b_sy), dyaw(a_by, b_by)
    print(f"[smoke] randomization deltas: silo_xy={d_sp * 1000:.1f}mm "
          f"silo_yaw={d_sy:.1f}deg tray_xy={d_bp * 1000:.1f}mm "
          f"tray_yaw={d_by:.1f}deg scoop_xy={d_cp * 1000:.1f}mm "
          f"balls={d_bl * 1000:.1f}mm", flush=True)
    check("randomization-is-real: silo xy+yaw, tray xy+yaw, scoop xy and ball "
          "positions all read back different across two seeds",
          d_sp > 0.003 and d_sy > 1.0 and d_bp > 0.003 and d_by > 1.0
          and d_cp > 0.003 and d_bl > 0.003)

    # ================= 3. present-count coverage ================================================
    seen: dict[int, int] = {}
    for s in range(300, 340):
        env.reset(seed=s)
        _refresh()
        k_ = int(scene.present[0].sum())
        seen.setdefault(k_, s)
        if len(seen) == 2:
            break
    print(f"[smoke] present-count coverage: {{k: first seed}} = {seen}", flush=True)
    check("present coverage: the sampled present-ball count takes both values 1 "
          "and 2 across seeded resets", set(seen) == {1, 2})
    k1_seed, k2_seed = seen.get(1, 300), seen.get(2, 300)

    # ================= 4. null policy fails =====================================================
    env.reset(seed=11)
    _step(300)
    report("null-policy")
    check("null-policy-fails: 300 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. negative: the SEED FAMILY'S OWN STRATEGY ==============================
    # maniskill/push_cube pushes the payload along the support toward the goal. The
    # balls are at the bottom of the silo — unreachable by any planar push — so the
    # seed's move can only shove what is reachable: the scoop, across the floor to
    # the tray. Also shove the silo itself with 30 N (it is bolted: it must not move).
    env.reset(seed=11)
    _step(30)
    _REC["on"] = True
    s0 = silo_xy().clone()
    zero = torch.zeros(n, 1, 3, device=device)
    tgt = basin_xy()
    for i_ in range(420):
        _refresh()
        d = tgt - (scene.scoop.data.root_pos_w[0, :2] - origin0[:2])
        dist = float(d.norm())
        if dist < 0.06:
            break
        f = torch.zeros(n, 1, 3, device=device)
        f[0, 0, :2] = 2.0 * d / max(dist, 1e-6)
        scene.scoop.set_external_force_and_torque(f, zero, env_ids=_all_ids(),
                                                  is_global=True)
        _step(1)
    scene.scoop.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
    fs = torch.zeros(n, 1, 3, device=device)
    fs[0, 0, 0] = 30.0
    for _ in range(60):
        scene.silo.set_external_force_and_torque(fs, zero, env_ids=_all_ids(),
                                                 is_global=True)
        _step(1)
    scene.silo.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
    _step(120)
    report("seed-strategy")
    _REC["on"] = False
    silo_moved = float((silo_xy() - s0).norm())
    sd = float((scene.scoop.data.root_pos_w[0, :2] - origin0[:2] - tgt).norm())
    print(f"[smoke] seed-strategy: scoop pushed to {sd * 1000:.0f}mm from the tray "
          f"centre; silo moved {silo_moved * 1000:.2f}mm under 30 N", flush=True)
    check("negative (SEED strategy): planar-pushing the reachable object to the "
          "goal moves nothing that matters — balls still in the silo, silo bolted "
          "(< 1 mm under 30 N), score ~0, no success",
          all(bool(scene.in_silo()[0, i]) for i in pres_idx())
          and silo_moved < 0.001 and float(scene.score()[0]) <= 0.05
          and not bool(scene.success()[0]))

    # ================= 6. scoop-in-tray: delivered but NOT stowed ===============================
    # The scoop must be moved OFF its (already stow-legal) spawn BEFORE the balls
    # land, or the constructed delivery would be a real success (audited).
    env.reset(seed=k2_seed)
    _step(30)
    _REC["on"] = True
    bxy = basin_xy()
    _place(scene.scoop, (float(bxy[0]) - 0.030, float(bxy[1]) + 0.055), 0.12,
           lying_q(math.radians(90.0)))
    _step(120)
    drop_balls_into_tray(pres_idx())
    report("scoop-in-tray")
    _REC["on"] = False
    d_basin = float((scene.scoop.data.root_pos_w[0, :2] - origin0[:2] - bxy).norm())
    check("scoop-in-tray: all present balls delivered but the scoop dropped into "
          "the tray too — delivered yet NOT stowed: no success, score capped 0.70",
          delivered() and d_basin < c.stow_clear and not bool(scene.scoop_stowed()[0])
          and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.70) < 1e-3)

    # ================= 7. stow near-miss: scoop against the silo ================================
    sxy = silo_xy()
    ring = float(c.stow_clear) - 0.050
    lay_scoop((float(sxy[0]), float(sxy[1]) - ring), sxy)
    _step(60)
    report("stow-near-miss")
    d_silo = float((scene.scoop.data.root_pos_w[0, :2] - origin0[:2] - sxy).norm())
    check("stow near-miss: scoop laid right against the silo (origin inside the "
          "stow_clear ring) — not stowed, no success",
          d_silo < c.stow_clear and not bool(scene.scoop_stowed()[0])
          and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.70) < 1e-3)

    # ================= 8. latched credit ========================================================
    # Same episode: teleport the delivered balls BACK into the silo. The live
    # in_basin evaporates; the latched credit must not.
    for k_, i in enumerate(pres_idx()):
        cyw = math.cos(math.radians(yaw_of(scene.silo.data.root_quat_w[0])))
        syw = math.sin(math.radians(yaw_of(scene.silo.data.root_quat_w[0])))
        lx, ly = 0.050, (-1) ** k_ * 0.025
        _place(scene.balls[i], (float(sxy[0]) + lx * cyw - ly * syw,
                                float(sxy[1]) + lx * syw + ly * cyw),
               c.floor_t + c.ball_r + 0.005)
        _step(30)
    _step(60)
    report("latched-credit")
    check("latched credit: delivered balls returned to the silo — live in_basin "
          "gone but the latched 0.70 survives; no success",
          all(bool(scene.in_silo()[0, i]) for i in pres_idx())
          and not delivered()
          and abs(float(scene.score()[0]) - 0.70) < 1e-3
          and not bool(scene.success()[0]))

    # ================= 9. floor-drop: escape credit only ========================================
    env.reset(seed=k2_seed)
    _step(30)
    for k_, i in enumerate(pres_idx()):
        _place(scene.balls[i], (-0.25, 0.55 + 0.08 * k_), 0.05)
    _step(120)
    report("floor-drop")
    out_all = all(not bool(scene.in_silo()[0, i]) for i in pres_idx())
    check("floor-drop: present balls on OPEN FLOOR — out of the silo but not in "
          "the tray: escape credit only (score <= 0.30), no success",
          out_all and not delivered() and float(scene.score()[0]) <= 0.30
          and float(scene.score()[0]) >= 0.20 and not bool(scene.success()[0]))

    # ================= 10. subset judging (k = 1) ===============================================
    env.reset(seed=k1_seed)
    _step(30)
    assert int(scene.present[0].sum()) == 1, "probe setup: k=1 seed"
    sxy = silo_xy()
    lay_scoop((float(sxy[0]), float(sxy[1]) - (float(c.stow_clear) - 0.050)), sxy)
    drop_balls_into_tray(pres_idx())
    report("subset-k1")
    absent = [i for i in range(c.n_balls) if not bool(scene.present[0, i])]
    check("subset judging: with k=1, delivering ONLY the present ball satisfies "
          "the delivered clause; the absent (depot) ball neither blocks nor "
          "inflates (score exactly 0.70; success held off only by the stow)",
          delivered() and len(absent) == 1
          and abs(float(scene.score()[0]) - 0.70) < 1e-3
          and not bool(scene.scoop_stowed()[0]) and not bool(scene.success()[0]))

    # ================= 11. ball-in-pan is not delivery ==========================================
    env.reset(seed=k2_seed)
    _step(30)
    _REC["on"] = True
    # stand the scoop upright on open floor (pan flat, handle up), drop a ball in
    up_xy = (-0.30, 0.55)
    _place(scene.scoop, up_xy, c.pan_t + 0.002,
           torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))
    _step(60)
    i0 = pres_idx()[0]
    sp = scene.scoop.data.root_pos_w[0] - origin0
    _place(scene.balls[i0], (float(sp[0]), float(sp[1])), float(sp[2]) + 0.030)
    _step(120)
    report("ball-in-pan")
    _REC["on"] = False
    from isaaclab.utils.math import quat_apply_inverse

    rel = quat_apply_inverse(scene.scoop.data.root_quat_w[0].view(1, 4),
                             (scene.balls[i0].data.root_pos_w[0]
                              - scene.scoop.data.root_pos_w[0]).view(1, 3)).view(3)
    in_pan = (abs(float(rel[0])) < c.pan_len / 2 + c.ramp_len
              and abs(float(rel[1])) < c.pan_w / 2
              and -0.005 < float(rel[2]) < c.side_h + 0.02)
    check("ball-in-pan: a ball resting IN the scoop's pan on open floor is not "
          "delivery — not in_basin, score <= 0.30, no success",
          in_pan and not bool(scene.in_basin()[0, i0])
          and float(scene.score()[0]) <= 0.30 and not bool(scene.success()[0]))

    # ================= 12. settle gate ==========================================================
    # Delivered balls + a scoop in a PERFECT stow pose — but sliding at 0.45 m/s.
    # success() must refuse while anything moves. Dismantled (transport) before
    # friction could settle it into a real success.
    env.reset(seed=k2_seed)
    _step(30)
    sxy = silo_xy()
    lay_scoop((float(sxy[0]), float(sxy[1]) - (float(c.stow_clear) - 0.050)), sxy)
    drop_balls_into_tray(pres_idx())
    assert delivered(), "probe setup: delivery must be in place"
    _place(scene.scoop, (-0.35, 0.10), -c.back_x + 0.004, lying_q(0.0),
           lin_vel=(0.45, 0.0))
    moving_ok = True
    for _ in range(5):
        _step(1)
        v = float(scene.scoop.data.root_lin_vel_w[0].norm())
        moving_ok = moving_ok and v > 0.10 and bool(scene.scoop_stowed()[0]) \
            and not bool(scene.settled()[0]) and not bool(scene.success()[0])
    report("settle-gate")
    # dismantle before it can settle into a real success
    bxy = basin_xy()
    _place(scene.scoop, (float(bxy[0]) - 0.030, float(bxy[1]) + 0.055), 0.12,
           lying_q(math.radians(90.0)))
    _step(90)
    check("settle gate: the exact success pose with the scoop still sliding at "
          "~0.45 m/s is refused while anything moves", moving_ok)

    # ================= 13. rejection audit ======================================================
    check("rejection audit: success() was never True at any step of this battery",
          _AUD["hits"] == 0)

    # ================= save + verdict ===========================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.silo_scoop")
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

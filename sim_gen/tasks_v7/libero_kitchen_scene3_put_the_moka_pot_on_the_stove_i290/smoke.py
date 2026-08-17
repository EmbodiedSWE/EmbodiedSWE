"""Smoke / rubric-REJECTION battery for MokaCarouselScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a real load-ride-stop delivery
and the latched credit is monotone along it). This battery proves the rubric REJECTS
wrong outcomes and that the claims the task rests on — the canopy as a physical
barrier, the bay as the only ride, the azimuth window, the at-rest judgement — are
load-bearing. Every probe is CONSTRUCTED as a settled state (teleport transport,
real physics steps, judge) — instrumentation, never a solution: no probe here
reaches success().

Checks:
  1.  settle/no-NaN    — seeded reset settles finite; pot on the counter, frypan
                         riding its bay, score ~0, no success;
  2.  randomization    — two seeded resets: READBACK turntable yaw, pot xy and pot
                         yaw all differ;
  3.  frypan-bay spread— over 10 resets the frypan's bay (verified by position
                         readback against the turntable yaw) takes all 3 values;
  4.  null-policy      — 240 idle steps: score ~0, no success;
  5.  seed-strategy    — the SEED task's move (lower the pot from above onto the
                         warming target) physically INTERCEPTED: dropped over the
                         station it lands ON the canopy roof — no bay, no latch,
                         score ~0;
  6.  open-disc rest   — pot settled upright ON the bare turntable between bays:
                         moved onto the carousel, but not IN a bay — no latch,
                         score ~0;
  7.  seated only      — pot dropped into an open empty bay far from the station:
                         seat latch fires (0.25 + a little progress), no success;
  8.  azimuth near-miss— the loaded bay parked ~28 deg from the station (outside
                         the 15 deg window): still seated, still at rest, but no
                         success, score <= 0.605;
  9.  wrong object     — the FRYPAN's bay parked dead on the station (pot still on
                         the counter): no success, score ~0;
  10. lying pot        — pot lying on its SIDE inside the station bay: the upright
                         clause refuses — no latch, no success, score ~0;
  11. fly-through      — pot seated, then the carousel set spinning so the bay
                         SWEEPS through the azimuth window: the window is crossed
                         (err dips below tol) yet success NEVER fires while the
                         carousel turns — the at-rest clauses are load-bearing;
  12. latched credit   — after the fly-through the pot is removed to the counter:
                         the latched partial credit PERSISTS (>= 0.25) but success
                         stays False — latches never grant the win;
  13. frames.npz       — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_kitchen_scene3_put_the_moka_pot_on_the_stove_i290.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=24)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version on some GPUs and silently rejects
# RTX -> the annotator returns EMPTY frames. Disable the check.
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

_qz = task_scene._qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

# ----- module state wired up in main() ----------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}


def _wrapf(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


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


def _write_body(body, pos_w: torch.Tensor, quat: torch.Tensor | None = None,
                ang_vel_z: float = 0.0, lin_vel: torch.Tensor | None = None) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat
    if lin_vel is not None:
        st[:, 7:10] = lin_vel
    st[:, 12] = ang_vel_z
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _center_pos(z: float) -> torch.Tensor:
    c = _ENV.scene.cfg
    n = _ENV.num_envs
    pos = torch.zeros(n, 3, device=_ENV.device)
    pos[:, 0], pos[:, 1], pos[:, 2] = c.center[0], c.center[1], z
    pos += _ENV.iscene.env_origins
    return pos


def _bay_spot(k: int, dz: float) -> torch.Tensor:
    """World point `dz` above bay k's floor centre, from the LIVE disc pose."""
    from isaaclab.utils.math import quat_apply

    scene = _ENV.scene
    c = scene.cfg
    n = _ENV.num_envs
    phi = float(scene._phi[k])
    lp = torch.tensor([c.r_bay * math.cos(phi), c.r_bay * math.sin(phi),
                       c.slab_t / 2 + dz], device=_ENV.device).expand(n, 3)
    return scene.disc.data.root_pos_w + quat_apply(scene.disc.data.root_quat_w, lp)


def _rotate_carousel_to(yaw_new: float, riders: list) -> None:
    """ONE consistent linkage write: disc at `yaw_new` (zero vel) and every
    rider re-seated at its bay centre (teleport-vs-linkage rule: never move one
    body of a linkage alone)."""
    scene = _ENV.scene
    c = scene.cfg
    n = _ENV.num_envs
    q = _qz(torch.full((n,), yaw_new, device=_ENV.device))
    _write_body(scene.disc, _center_pos(c.disc_z), q)
    for body, k in riders:
        _write_body(body, _bay_spot(k, 0.002))


def _drop_pot(k: int, settle: int = 360) -> bool:
    """Transport the pot to the in-bay hover, release, settle; retry a bounce."""
    scene = _ENV.scene
    for _ in range(3):
        _write_body(scene.pot, _bay_spot(k, 0.015))
        _step(settle)
        if int(scene.pot_bay()[0]) == k:
            return True
    return False


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    azs = scene.pocket_azimuths()[0]
    print(f"[smoke] {tag:16s} | yaw={math.degrees(float(scene.disc_yaw()[0])):+7.1f}deg "
          f"bays=({math.degrees(float(azs[0])):+6.1f},{math.degrees(float(azs[1])):+6.1f},"
          f"{math.degrees(float(azs[2])):+6.1f}) pot_bay={int(scene.pot_bay()[0])} "
          f"err={math.degrees(float(scene.bay_err()[0])):6.1f}deg "
          f"still={bool(scene.still()[0])} seat={int(scene._l_seat[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main -------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.moka_carousel")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.45, -1.00, 0.90)) + o),
                                tuple(np.array((0.42, 0.00, 0.12)) + o),
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

    def yaw_deg() -> float:
        return math.degrees(float(scene.disc_yaw()[0]))

    def dyaw(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    oxy = env.iscene.env_origins[:, :2]
    oz = float(env.iscene.env_origins[0, 2])

    def deck_spot(x: float, y: float) -> torch.Tensor:
        pos = torch.zeros(n, 3, device=device)
        pos[:, 0], pos[:, 1], pos[:, 2] = x, y, oz + c.deck_top + 0.002
        pos[:, :2] += oxy
        return pos

    def pot_xyz():
        _refresh()
        p = scene.pot.data.root_pos_w[0].clone()
        p[:2] -= oxy[0]
        p[2] -= oz
        return p

    def empty_bays() -> list:
        pb = int(scene.pan_bay[0])
        return [k for k in range(3) if k != pb]

    az_tol = math.radians(c.az_tol_deg)

    # ================= 1. settle / no-NaN =======================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(120)
    _report("settle")
    _REC["on"] = False
    fin = bool(torch.isfinite(scene.pot.data.root_pos_w).all()
               and torch.isfinite(scene.disc.data.root_pos_w).all()
               and torch.isfinite(scene.pot.data.root_lin_vel_w).all())
    pl = scene._local_of(scene.frypan)[0]
    pp = pot_xyz()
    check("settle/no-NaN: pot on the counter, frypan riding its bay, score 0, "
          "no success",
          fin and abs(float(pp[2]) - c.deck_top) < 0.01
          and int(scene.pot_bay()[0]) == -1
          and abs(float(pl[:2].norm()) - c.r_bay) < 0.02
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ======================================
    def readback():
        _refresh()
        return (yaw_deg(),
                (scene.pot.data.root_pos_w[0, :2] - oxy[0]).clone(),
                math.degrees(2.0 * math.atan2(float(scene.pot.data.root_quat_w[0, 3]),
                                              float(scene.pot.data.root_quat_w[0, 0]))))

    rbs = []
    for s in (101, 202, 303):
        torch.manual_seed(s)
        env.reset()
        _step(10)
        rbs.append(readback())
    pairs = [(0, 1), (0, 2), (1, 2)]
    d_disc = max(dyaw(rbs[i][0], rbs[j][0]) for i, j in pairs)
    d_xy = max(float((rbs[i][1] - rbs[j][1]).norm()) for i, j in pairs)
    d_py = max(dyaw(rbs[i][2], rbs[j][2]) for i, j in pairs)
    print(f"[smoke] randomization max pairwise deltas over 3 seeds: "
          f"disc_yaw={d_disc:.1f}deg pot_xy={d_xy * 1000:.1f}mm "
          f"pot_yaw={d_py:.1f}deg", flush=True)
    check("randomization-is-real: turntable yaw, pot xy and pot yaw readback differ "
          "across seeds",
          d_disc > 10.0 and d_xy > 0.02 and d_py > 5.0)

    # ================= 3. frypan-bay spread (position readback) =================================
    bays_seen = set()
    consistent = True
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _step(5)
        _refresh()
        k = int(scene.pan_bay[0])
        bays_seen.add(k)
        # verify the record against the frypan's actual azimuth vs the disc yaw
        fp = scene.frypan.data.root_pos_w[0, :2] - oxy[0]
        az = math.atan2(float(fp[1]) - c.center[1], float(fp[0]) - c.center[0])
        want = _wrapf(float(scene.disc_yaw()[0]) + float(scene._phi[k]))
        consistent &= abs(_wrapf(az - want)) < math.radians(6.0)
    print(f"[smoke] frypan bays over 10 resets: {sorted(bays_seen)} "
          f"(readback consistent={consistent})", flush=True)
    check("frypan-bay spread: all 3 bays drawn over 10 resets, record matches "
          "position readback", bays_seen == {0, 1, 2} and consistent)

    # ================= 4. null policy fails =====================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. seed strategy intercepted by the canopy ===============================
    # The seed task's move class: carry the pot ABOVE the warming target and
    # lower it. Here the canopy roof is in the way — the pot lands ON the roof.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    roof_top = c.deck_top + c.roof_under_local + c.roof_t
    pos = torch.zeros(n, 3, device=device)
    pos[:, 0] = c.center[0] + c.r_bay
    pos[:, 1] = c.center[1]
    pos[:, 2] = oz + roof_top + 0.03
    pos[:, :2] += oxy
    _REC["on"] = True
    _write_body(scene.pot, pos)
    _step(300)
    _report("seed-drop")
    _REC["on"] = False
    pp = pot_xyz()
    check("seed-strategy rejected: pot lowered from above the station lands ON "
          "the canopy roof — no bay, no latch, score ~0",
          float(pp[2]) > roof_top - 0.02 and int(scene.pot_bay()[0]) == -1
          and not bool(scene._l_seat[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.005)

    # ================= 6. pot on the open turntable, not in a bay ===============================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    # spot on the slab between two bays (r_bay, phi0 + 60 deg), clear of posts
    from isaaclab.utils.math import quat_apply

    lp = torch.tensor([c.r_bay * math.cos(math.pi / 3), c.r_bay * math.sin(math.pi / 3),
                       c.slab_t / 2 + 0.003], device=device).expand(n, 3)
    spot = scene.disc.data.root_pos_w + quat_apply(scene.disc.data.root_quat_w, lp)
    _write_body(scene.pot, spot)
    _step(240)
    _report("open-disc")
    pp = pot_xyz()
    check("negative (open disc): pot settled upright ON the turntable between "
          "bays — not in a bay, no latch, no success, score ~0",
          abs(float(pp[2]) - (c.disc_top + 0.0)) < 0.02 and bool(scene.pot_upright()[0])
          and int(scene.pot_bay()[0]) == -1 and not bool(scene._l_seat[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.005)

    # ================= 7. seated in an open bay far from the station ============================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    k7 = max(empty_bays(), key=lambda j: abs(float(scene.pocket_azimuths()[0, j])))
    _REC["on"] = True
    ok_drop = _drop_pot(k7)
    _report("seated-only")
    _REC["on"] = False
    err7 = float(scene.bay_err()[0])
    s7 = float(scene.score()[0])
    check("partial (seated only): pot seated in an open bay far from the station "
          "— seat latch fires, 0.24 <= score <= 0.55, no success",
          ok_drop and bool(scene._l_seat[0]) and err7 > az_tol + math.radians(10.0)
          and 0.24 <= s7 <= 0.55 and not bool(scene.success()[0]))

    # ================= 8. azimuth near-miss (outside the window) ================================
    # Park the LOADED bay 28 deg short of the station: one consistent linkage
    # write (disc + pot + frypan), then settle at rest.
    target8 = math.radians(28.0)
    yaw8 = _wrapf(target8 - float(scene._phi[k7]))
    _rotate_carousel_to(yaw8, [(scene.frypan, int(scene.pan_bay[0]))])
    _write_body(scene.pot, _bay_spot(k7, 0.004))
    _step(300)
    _report("near-miss-28deg")
    err8 = float(scene.bay_err()[0])
    check("near-miss (28 deg): loaded bay at rest just OUTSIDE the 15 deg window "
          "— still seated, no success, score <= 0.605",
          int(scene.pot_bay()[0]) == k7 and bool(scene.still()[0])
          and az_tol + math.radians(5.0) < err8 < math.radians(45.0)
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.605)

    # ================= 9. wrong object at the station ===========================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    pb = int(scene.pan_bay[0])
    _rotate_carousel_to(_wrapf(-float(scene._phi[pb])), [(scene.frypan, pb)])
    _step(240)
    _report("wrong-object")
    fp = scene.frypan.data.root_pos_w[0, :2] - oxy[0]
    az_pan = abs(_wrapf(math.atan2(float(fp[1]) - c.center[1],
                                   float(fp[0]) - c.center[0])))
    check("negative (wrong object): the FRYPAN's bay parked dead on the station, "
          "pot still on the counter — no success, score ~0",
          az_pan < math.radians(8.0) and int(scene.pot_bay()[0]) == -1
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.005)

    # ================= 10. lying pot in the station bay =========================================
    # Even IN the right bay AT the station, a toppled pot is refused (upright
    # clause). Constructed state — a policy could not reach it (the canopy),
    # but the rubric must reject it regardless.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    k10 = empty_bays()[0]
    _rotate_carousel_to(_wrapf(-float(scene._phi[k10])), [(scene.frypan,
                                                           int(scene.pan_bay[0]))])
    q_side = torch.tensor([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0],
                          device=device).expand(n, 4)  # 90 deg about x: lying
    _write_body(scene.pot, _bay_spot(k10, 0.032), q_side)
    _step(300)
    _report("lying-pot")
    check("negative (lying pot): pot on its SIDE inside the station bay — the "
          "upright clause refuses: no latch, no success, score ~0",
          not bool(scene.pot_upright()[0]) and int(scene.pot_bay()[0]) == -1
          and not bool(scene._l_seat[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.005)

    # ================= 11. spinning fly-through of the window ===================================
    # Pot seated, then the carousel is SET SPINNING so the loaded bay sweeps
    # through the station window while turning. The window is genuinely crossed
    # (err dips below tol) yet success must never fire (at-rest clauses).
    torch.manual_seed(100)
    env.reset()
    _step(30)
    k11 = max(empty_bays(), key=lambda j: abs(float(scene.pocket_azimuths()[0, j])))
    yaw11 = _wrapf(math.radians(70.0) - float(scene._phi[k11]))
    _rotate_carousel_to(yaw11, [(scene.frypan, int(scene.pan_bay[0]))])
    ok_drop = _drop_pot(k11)
    seat11 = bool(scene._l_seat[0])
    # one consistent linkage write: disc spinning at -1.3 rad/s, riders given the
    # matching tangential velocity (v = w x r about the axis)
    w_spin = -1.3
    _write_body(scene.disc, _center_pos(c.disc_z),
                _qz(torch.full((n,), _wrapf(math.radians(70.0) - float(scene._phi[k11])),
                               device=device)), ang_vel_z=w_spin)
    for body in (scene.pot, scene.frypan):
        _refresh()
        r = body.data.root_pos_w - scene.disc.data.root_pos_w
        v = torch.zeros(n, 3, device=device)
        v[:, 0] = -w_spin * r[:, 1]
        v[:, 1] = w_spin * r[:, 0]
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = body.data.root_pos_w
        st[:, 3:7] = body.data.root_quat_w
        st[:, 7:10] = v
        st[:, 12] = w_spin
        body.write_root_state_to_sim(st, _all_ids())
    _refresh()
    min_err = math.pi
    fired = False
    _REC["on"] = True
    for _ in range(900):
        _step(1)
        min_err = min(min_err, float(scene.bay_err()[0]))
        fired |= bool(scene.success()[0])
    _REC["on"] = False
    _report("fly-through")
    check("negative (fly-through): the spinning bay crosses the window (min err "
          f"{math.degrees(min_err):.1f} deg < tol) yet success never fires while "
          "turning — score <= 0.605",
          ok_drop and seat11 and min_err < az_tol and not fired
          and float(scene.score()[0]) <= 0.605)

    # ================= 12. latched credit persists, success does not ============================
    _write_body(scene.pot, deck_spot(0.02, -0.28))
    _step(150)
    _report("latch-persist")
    check("latched credit: pot removed to the counter after the fly-through — "
          "partial credit persists (>= 0.25) but success stays False",
          int(scene.pot_bay()[0]) == -1 and float(scene.score()[0]) >= 0.25
          and float(scene.score()[0]) <= 0.605 and not bool(scene.success()[0]))

    # ================= save + verdict ===========================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.moka_carousel")
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
    except BaseException:  # noqa: BLE001 - die fast; Kit teardown would hang until the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)

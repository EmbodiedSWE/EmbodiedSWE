"""Smoke / rubric-REJECTION battery for ChockRampScene — NullRobot, teleported probes.

solve.py is the acceptance proof (chock keyed into the starred pocket, then gravity
delivers the cube into the band, 1.0). This battery proves the rubric REJECTS wrong
outcomes and that the physical claims the task rests on are load-bearing: the face
really is too slick for anything to rest unsupported (the seed's place-at-the-target
strategy, executed literally AT the goal pose, ends on the floor past the foot — and
the yellow marker arrests nothing on the way), an unkeyed chock really slides away,
a WRONG-STATION park that is physically stable and visually identical pays ZERO,
credit never leaks (a cube released downhill of the seated chock leaves the score at
exactly 0.375), and the non-success score caps at exactly 0.75 while success()
refuses anything not still for a full streak. Every probe is CONSTRUCTED (teleport,
real physics steps, judge) — instrumentation, never a solution: success() is
monitored at EVERY step and must never turn True anywhere in this battery (the
audit is itself a check).

Checks:
   1. settle/no-NaN     — seeded reset settles finite; chock + cube on the floor
                          clear of the ramp (readback); authored masses; score ~0;
   2. randomization     — ramp xy + yaw and both spawns differ pairwise across seeds
                          0/1/2 (READBACK); the starred station k* takes >= 2 values
                          and the marker tracks the starred band across seeds 0..7;
                          the spawn-slot swap takes both values;
   3. null policy       — 240 idle steps: nothing latches, score ~0, no success;
   4. SLICK FACE        — the seed's own strategy: the cube is SET DOWN exactly on
                          the goal band (on_band geometry true at the hold, readback)
                          and released with no chock anywhere. It slides straight
                          off the foot onto the floor — across the marker, which
                          arrests nothing (visual-only). Nothing latches, score ~0;
   5. unkeyed chock     — the chock laid flat ON the face beside the starred pocket
                          (tab on the surface, ~11 mm proud, not keyed): the seated
                          z/angle windows reject it and it slides away (moved,
                          readback). No seat credit;
   6. WRONG STATION     — the full end state built at a NON-starred pocket: chock
                          really keyed there (manual readback), cube really parked
                          braced against it (still, on the face, face-to-face gap —
                          readback). Physically stable and visually identical to
                          success — yet NOTHING latches and score stays ~0: the
                          station choice is load-bearing;
   7. downhill release  — chock properly seated at the STARRED pocket (real drop —
                          `seated` latches, exactly 0.375), then the cube released
                          DOWNHILL of it: the plate cannot catch from below, the
                          cube ends on the floor, and the score stays EXACTLY 0.375
                          (float32) — park credit requires the braced rest;
   8. settle gate + cap — cube teleported against the seated chock inside the band
                          moving 0.2 m/s: `parked` latches, the non-success score
                          caps at EXACTLY 0.75 (float32), success() refuses for all
                          monitored steps (stillness streak < required), and the
                          probe is dismantled before the streak completes; the
                          latched 0.75 survives;
   9. rejection audit   — success() was never True at any step of this battery;
  10. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.pick_i330.smoke --headless
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

_qapply, _qinv, _qmul = task_scene._qapply, task_scene._qinv, task_scene._qmul

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() -----------------------------------------------------------
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


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    kl = scene._local(scene.chock.data.root_pos_w)[0]
    cl = scene._local(scene.cube.data.root_pos_w)[0]
    print(f"[smoke] {tag:16s} | chock_l=({float(kl[0]):+.4f},{float(kl[1]):+.4f},"
          f"{float(kl[2]):+.4f}) cube_l=({float(cl[0]):+.4f},{float(cl[1]):+.4f},"
          f"{float(cl[2]):+.4f}) seated={bool(scene.chock_seated()[0])} "
          f"band={bool(scene.on_band()[0])} braced={bool(scene.braced()[0])} "
          f"l=({int(scene._l_seated[0])},{int(scene._l_parked[0])}) "
          f"streak={int(scene._streak[0])} score={float(scene.score()[0]):.4f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}", flush=True)


# ----- main --------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.chock_ramp")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.25, -0.85, 0.70)) + o),
                                tuple(np.array((0.46, 0.0, 0.10)) + o),
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

    def rq() -> torch.Tensor:
        _refresh()
        return scene.ramp.data.root_quat_w

    def r_yaw() -> float:
        xw = _qapply(rq()[:1], torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
        return math.degrees(math.atan2(float(xw[1]), float(xw[0])))

    def chock_local() -> torch.Tensor:
        _refresh()
        return scene._local(scene.chock.data.root_pos_w)[0]

    def cube_local() -> torch.Tensor:
        _refresh()
        return scene._local(scene.cube.data.root_pos_w)[0]

    def cube_world() -> torch.Tensor:
        _refresh()
        return (scene.cube.data.root_pos_w - scene.env_origins)[0]

    def chock_world() -> torch.Tensor:
        _refresh()
        return (scene.chock.data.root_pos_w - scene.env_origins)[0]

    def write_local(body, local_xyz, *, lin_vel_l=None) -> None:
        """Teleport a body to a SLOPE-frame pose, aligned to the slope frame (optional
        slope-frame velocity). Pure state construction — physics and the judge do the
        rest."""
        _refresh()
        st = torch.zeros(n, 13, device=device)
        loc = torch.tensor([list(local_xyz)], device=device, dtype=torch.float)
        st[:, 0:3] = scene.ramp.data.root_pos_w + _qapply(rq(), loc.expand(n, 3))
        st[:, 3:7] = rq()
        if lin_vel_l is not None:
            v = torch.tensor([list(lin_vel_l)], device=device, dtype=torch.float)
            st[:, 7:10] = _qapply(rq(), v.expand(n, 3))
        body.write_root_state_to_sim(st, _all_ids())
        _refresh()

    def seat_chock_at(xk: float, settle: int = 180) -> None:
        """Real seating: hover the chock tab-in-socket-mouth over pocket `xk`, release,
        let gravity drop it flush and key it downhill (same physics as the solve)."""
        write_local(scene.chock, (xk + 0.002, 0.0, 0.004))
        _step(settle)

    def manual_keyed(xk: float) -> bool:
        """Station-agnostic seat readback (chock_seated() only checks the STARRED one)."""
        kl = chock_local()
        dq = _qmul(_qinv(rq()), scene.chock.data.root_quat_w)
        ang = math.degrees(2.0 * math.acos(min(1.0, abs(float(dq[0, 0])))))
        return (abs(float(kl[0]) - xk) < c.seat_tol_x and abs(float(kl[1])) < c.seat_tol_y
                and c.seat_z_lo < float(kl[2]) < c.seat_z_hi
                and ang < c.seat_ang_deg)

    # ================= 1. settle / no-NaN / baseline zero =======================================
    env.reset(seed=0)
    _AUD["on"] = True
    _step(240)
    _report("reset")
    cw, kw = cube_world(), chock_world()
    m_chock = float(scene.chock.root_physx_view.get_masses().sum())
    m_cube = float(scene.cube.root_physx_view.get_masses().sum())
    print(f"[smoke] mass readback: chock={m_chock:.3f} cube={m_cube:.3f}", flush=True)
    d_cube = math.hypot(float(cw[0]) - c.ramp_pos[0], float(cw[1]) - c.ramp_pos[1])
    d_chock = math.hypot(float(kw[0]) - c.ramp_pos[0], float(kw[1]) - c.ramp_pos[1])
    check("settle/no-NaN: seeded reset settles finite; chock and cube rest on the "
          "floor clear of the ramp (readback); masses authored; score ~0, no success",
          bool(scene._finite()[0])
          and d_cube > 0.25 and d_chock > 0.25
          and float(cw[2]) < 0.05 and float(kw[2]) < 0.05
          and abs(m_chock - c.chock_mass) < 0.02 and abs(m_cube - c.cube_mass) < 0.02
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization readback ================================================
    ramps, yaws, chocks, cubes = [], [], [], []
    for s in (0, 1, 2):
        env.reset(seed=s)
        _step(30)
        _refresh()
        rp = (scene.ramp.data.root_pos_w - scene.env_origins)[0]
        ramps.append((float(rp[0]), float(rp[1])))
        yaws.append(r_yaw())
        kw, cw = chock_world(), cube_world()
        chocks.append((float(kw[0]), float(kw[1])))
        cubes.append((float(cw[0]), float(cw[1])))
    stars, swaps, marker_ok = [], [], True
    for s in range(8):
        env.reset(seed=s)
        _refresh()
        k = int(scene.k_star[0])
        stars.append(k)
        swaps.append(float(scene.swap[0]))
        ml = scene._local(scene.marker.data.root_pos_w)[0]
        marker_ok = marker_ok and abs(float(ml[0]) - float(scene.bands_t[k])) < 0.005 \
            and abs(float(ml[1])) < 0.005
    print(f"[smoke] rand readback: ramps={ramps} yaws={[f'{y:+.1f}' for y in yaws]} "
          f"chocks={chocks} cubes={cubes} stars={stars} swaps={swaps} "
          f"marker_ok={marker_ok}", flush=True)

    def pairwise(vals, tol) -> bool:
        return all(abs(a - b) > tol for i, a in enumerate(vals) for b in vals[i + 1:])

    check("randomization: ramp xy+yaw and chock/cube spawn xy differ pairwise across "
          "seeds 0/1/2; k* takes >= 2 values and the marker tracks the starred band "
          "across seeds 0..7; the spawn-slot swap takes both values",
          pairwise([g[0] for g in ramps], 3e-4) and pairwise([g[1] for g in ramps], 3e-4)
          and pairwise(yaws, 0.1)
          and pairwise([r[0] for r in chocks], 3e-4) and pairwise([r[1] for r in chocks], 3e-4)
          and pairwise([b[0] for b in cubes], 3e-4) and pairwise([b[1] for b in cubes], 3e-4)
          and len(set(stars)) >= 2 and marker_ok
          and (1.0 in swaps) and (-1.0 in swaps))

    # ================= 3. null policy ===========================================================
    env.reset(seed=0)
    _step(240)
    _report("null-policy")
    check("null policy: 240 idle steps latch nothing — score ~0, no success",
          float(scene.score()[0]) <= 0.01 and not bool(scene._l_seated[0])
          and not bool(scene._l_parked[0]) and not bool(scene.success()[0]))

    # ================= 4. SLICK FACE: the seed strategy at the goal pose ========================
    # Place-at-the-target, literally: the cube is set down exactly ON the yellow band
    # (the goal x, resting z) with no chock anywhere, and released.
    _REC["on"] = True
    b_star = float(scene.bands_t[scene.k_star[0]])
    write_local(scene.cube, (b_star, 0.0, c.cube_s / 2 + 0.003))
    held_on_band = bool(scene.on_band()[0])  # the goal geometry IS momentarily occupied
    x0 = float(cube_local()[0])
    _step(300)
    cw = cube_world()
    _report("slick-face")
    check("slick face: the cube set down exactly on the band (on_band readback True "
          "at the hold) with no chock slides straight off the foot to the floor — "
          "across the visual-only marker, which arrests nothing; score ~0",
          held_on_band and float(cw[2]) < 0.05
          and not bool(scene.on_band()[0]) and not bool(scene._l_parked[0])
          and not bool(scene._l_seated[0]) and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))
    print(f"[smoke] slick-face: start x_l={x0:+.3f} -> floor z={float(cw[2]):+.3f}",
          flush=True)

    # ================= 5. unkeyed chock slides away =============================================
    x_star = float(scene.pockets_t[scene.k_star[0]])
    # Tab NOT in any pocket: offset across-slope so the tab lands on the face beside
    # the starred socket (tab bottom on the surface -> origin ~11 mm proud, tilted).
    write_local(scene.chock, (x_star, 0.055, c.tab_h + 0.003))
    _step(10)
    kl_held = chock_local()
    proud = float(kl_held[2]) > c.seat_z_hi + 0.003
    _step(240)
    kw = chock_world()
    moved = (abs(float(chock_local()[0]) - x_star) > 0.05) or (float(kw[2]) < 0.06)
    _report("unkeyed-chock")
    check("unkeyed chock: laid flat on the face beside the socket it sits ~11 mm "
          "PROUD (readback, outside the seated z-window) and slides away (moved "
          "readback) — no seat credit, score ~0",
          proud and moved and not bool(scene._l_seated[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 6. WRONG-STATION full park pays zero =====================================
    env.reset(seed=3)
    _step(60)
    k_star_i = int(scene.k_star[0])
    k_wrong = (k_star_i + 1) % 3
    x_wrong = float(scene.pockets_t[k_wrong])
    seat_chock_at(x_wrong)
    keyed_wrong = manual_keyed(x_wrong)
    # deliver the cube to the WRONG band exactly as the solve delivers to the right one
    write_local(scene.cube, (x_wrong + c.band_off + c.band_half + 0.006, 0.0,
                             c.cube_s / 2 + 0.004))
    _step(300)
    kl, cl = chock_local(), cube_local()
    gap = float(cl[0]) - (float(kl[0]) + c.plate_t / 2 + c.cube_s / 2)
    parked_wrong = (abs(float(cl[2]) - c.cube_s / 2) < 0.008
                    and c.gap_lo < gap < c.gap_hi
                    and float(scene.cube.data.root_lin_vel_w[0].norm()) < 0.10)
    _report("wrong-station")
    check("wrong station: the SAME physically stable park built at a non-starred "
          "pocket (chock keyed there + cube braced against it, still — readbacks) "
          "latches NOTHING and scores ~0: the station choice is load-bearing",
          keyed_wrong and parked_wrong
          and not bool(scene.chock_seated()[0]) and not bool(scene.on_band()[0])
          and not bool(scene._l_seated[0]) and not bool(scene._l_parked[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 7. downhill release: credit stays at exactly 0.375 =======================
    env.reset(seed=5)
    _step(60)
    x_star = float(scene.pockets_t[scene.k_star[0]])
    seat_chock_at(x_star)
    _report("seat-star")
    seated_ok = bool(scene.chock_seated()[0]) and bool(scene._l_seated[0])
    s_seat = float(scene.score()[0])
    # release the cube DOWNHILL of the seated chock: the plate cannot catch from below
    write_local(scene.cube, (x_star - 0.060, 0.0, c.cube_s / 2 + 0.004))
    _step(300)
    cw = cube_world()
    _report("downhill-rel")
    check("downhill release: with the chock properly seated at the starred pocket "
          "(seated latch = exactly 0.375), a cube released DOWNHILL of it slides to "
          "the floor uncaught and the score stays EXACTLY 0.375 (float32) — park "
          "credit requires the braced rest, and never leaks",
          seated_ok and abs(s_seat - 0.375) < 1e-6
          and float(cw[2]) < 0.05 and not bool(scene._l_parked[0])
          and abs(float(scene.score()[0]) - 0.375) < 1e-6
          and not bool(scene.success()[0]))

    # ================= 8. settle gate + exact 0.75 cap ==========================================
    # Cube constructed against the seated chock inside the band, MOVING 0.2 m/s
    # downhill: `parked` latches (gate 0.25), success() must refuse (stillness
    # streak) for every monitored step; dismantle before the streak completes.
    kl = chock_local()
    write_local(scene.cube,
                (float(kl[0]) + c.plate_t / 2 + c.cube_s / 2 + 0.002, float(kl[1]),
                 c.cube_s / 2 + 0.001),
                lin_vel_l=(-0.20, 0.0, 0.0))
    refuse = True
    for _ in range(6):
        _step(1)
        refuse = refuse and not bool(scene.success()[0])
    s_cap = float(scene.score()[0])
    latched = bool(scene._l_parked[0])
    _report("settle-gate")
    # dismantle before the stillness streak can complete; the latched 0.75 survives
    write_local(scene.cube, (x_star - 0.060, 0.0, c.cube_s / 2 + 0.004))
    _step(120)
    s_after = float(scene.score()[0])
    _report("cap-latched")
    _REC["on"] = False
    check("settle gate + cap: a moving cube braced in the band latches `parked`, the "
          "non-success score is capped at EXACTLY 0.75 (float32), success() refuses "
          "every monitored step (streak short), and the latched 0.75 survives the "
          "dismantle",
          latched and refuse and abs(s_cap - 0.75) < 1e-6
          and abs(s_after - 0.75) < 1e-6 and not bool(scene.success()[0]))

    # ================= 9. rejection audit =======================================================
    check("rejection audit: success() was never True at any step of this battery",
          _AUD["hits"] == 0)

    # ================= save + verdict ===========================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.chock_ramp")
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

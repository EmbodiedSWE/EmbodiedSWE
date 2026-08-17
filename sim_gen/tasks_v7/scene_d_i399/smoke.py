"""Smoke battery for MastLoweringScene (sim_gen task `scene_d_i399`) —
REJECTION-ONLY: every check either verifies basic health/randomization or
CONSTRUCTS (or physically drives) a settled wrong outcome and asserts the
rubric refuses it. success() must never fire anywhere in the battery.

 1. settle/no-NaN     — mast leaning on its back stop, bottle upright at its
                        sampled marker, cradle at its sampled rack, score ~0.
 2. randomization A   — the bottle marker (db, py) varies across resets and
                        BOTH the live pad and the standing bottle track the
                        sample every time (readback).
 3. randomization B   — the cradle rack spot (x, signed y, yaw) varies across
                        resets and the live cradle tracks the sample.
 4. null policy       — 240 idle steps -> mast still on its stop (phi < 0),
                        bottle untouched, score ~0, no success.
 5. fell-first        — the REAL solve-grade hinge torque with the cradle
                        still RACKED: the mast sweeps down THROUGH the catch
                        band (the moving pass must not latch arrest) and past
                        it, ending propped on the bottle's flat cap (~107+ deg)
                        or on the floor (~116 deg) — both OUTSIDE the band —
                        the physics-forced order: score exactly the commit
                        credit (0.20), no success.
 6. bottle lying      — caught-mast state CONSTRUCTED (cradle deployed, mast
                        settling 2 deg onto the crest) but the bottle LYING on
                        the court off to the side: band+settled hold, the
                        bottle gate refuses — no success.
 7. displaced bottle  — same caught construct, bottle STANDING but 7 cm off
                        its marker: the xy gate refuses — no success.
 8. perched bottle    — same caught construct, bottle stood ON TOP of the
                        caught beam over the marker xy: the rest-height gate
                        refuses (base_z far above the pad) — no success.
 9. deploy-only       — the solve's real hover-drop + press with NOTHING else:
                        cradle stands deployed, mast never moves — score
                        exactly the deploy credit (0.20), no success.
10. caught but moving — the full success pose WRITTEN with mast angular
                        velocity 0.4 rad/s: the settle gate refuses success on
                        the moving state (judged without stepping; then the
                        state is broken up before stepping).
11. rejection audit   — success() observed False at every step of the battery.
12. final no-NaN.
13. video             — frames.npz (>10 frames) written to the CWD.

Run (forge): python -u -m simgen_tasks.scene_d_i399.smoke --headless
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

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- actuation magnitudes (the same limited hand as solve.py) ---------------------------------
F_PRESS = 8.0                     # cradle seating press (N, downward)
KP_M, KD_M, T_M = 2.0, 0.3, 1.6   # mast hinge push
CUT_DEG = 18.0

# ----- module state wired up in main() ----------------------------------------------------------
_ENV = None
_REC = {"on": True, "annot": None, "frames": [], "i": 0}
_AUDIT = {"saw_success": False}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        _AUDIT["saw_success"] |= bool(env.scene.success()[0])
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
    b = (scene.bottle.data.root_pos_w - scene.env_origins)[0]
    cr = (scene.cradle.data.root_pos_w - scene.env_origins)[0]
    print(f"[smoke] {tag:16s} | phi={math.degrees(float(scene.phi()[0])):+7.2f}deg "
          f"w={float(scene.mast_w()[0]):+.3f} "
          f"bottle=({float(b[0]):+.3f},{float(b[1]):+.3f},{float(b[2]):+.3f}) "
          f"cradle=({float(cr[0]):+.3f},{float(cr[1]):+.3f},{float(cr[2]):+.3f}) "
          f"latch(d/c/a)=({int(scene._deploy_l[0])},{int(scene._commit_l[0])},"
          f"{int(scene._arrest_l[0])}) bok={bool(scene.bottle_ok()[0])} "
          f"settled={bool(scene.settled()[0])} score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main --------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.mast_lowering")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply_inverse

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.25, -1.05, 0.85)) + o),
                                tuple(np.array((0.30, 0.0, 0.25)) + o),
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

    def succ() -> bool:
        s = bool(scene.success()[0])
        _AUDIT["saw_success"] |= s
        return s

    zero = torch.zeros(n, 1, 3, device=device)

    def write_body(body, dx: float, dy: float, dz: float, yaw: float = 0.0,
                   pitch: float = 0.0, wy: float = 0.0) -> None:
        """Write a full root state: position, yaw OR pitch, optional hinge-axis
        angular velocity (velocity written WITH the pose — settle-gate probes)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = scene.env_origins[:, 0] + dx
        st[:, 1] = scene.env_origins[:, 1] + dy
        st[:, 2] = scene.env_origins[:, 2] + dz
        if pitch != 0.0:
            st[:, 3] = math.cos(pitch / 2.0)
            st[:, 5] = math.sin(pitch / 2.0)
        else:
            st[:, 3] = math.cos(yaw / 2.0)
            st[:, 6] = math.sin(yaw / 2.0)
        st[:, 11] = wy
        body.write_root_state_to_sim(st, _all_ids())
        _refresh()

    def write_mast(phi_deg: float, wy: float = 0.0) -> None:
        write_body(scene.mast, 0.0, 0.0, c.hinge_z, pitch=math.radians(phi_deg), wy=wy)

    def press_cradle(steps: int) -> None:
        """REAL actuation: the same downward seating press as solve.py."""
        f_w = torch.zeros(n, 3, device=device)
        f_w[:, 2] = -F_PRESS
        for _ in range(steps):
            f_b = quat_apply_inverse(scene.cradle.data.root_quat_w, f_w)
            scene.cradle.set_external_force_and_torque(f_b.reshape(n, 1, 3), zero)
            _step(1)
        scene.cradle.set_external_force_and_torque(zero, zero)
        _step(60)

    def deploy_real(db: float) -> None:
        """The solve's deployment: hover-teleport to the corridor target, drop,
        REAL press to seat."""
        write_body(scene.cradle, db + c.deploy_off, 0.0, c.court_z1 + 0.010)
        _step(30)
        press_cradle(60)

    def push_mast(steps: int = 600) -> None:
        """REAL actuation: the same ramped, torque-limited hinge push as
        solve.py, cut at CUT_DEG."""
        phi0 = float(scene.phi()[0])
        tgt = math.radians(25.0)
        for i in range(steps):
            a = min(1.0, i / 120.0)
            phi_t = phi0 + (tgt - phi0) * a
            p = scene.phi()
            w = scene.mast_w()
            tq = (KP_M * (phi_t - p) - KD_M * w).clamp(min=-T_M, max=T_M)
            t_b = torch.zeros(n, 3, device=device)
            t_b[:, 1] = tq
            scene.mast.set_external_force_and_torque(zero, t_b.reshape(n, 1, 3))
            _step(1)
            if float(scene.phi()[0]) >= math.radians(CUT_DEG):
                break
        scene.mast.set_external_force_and_torque(zero, zero)

    def construct_caught(db: float) -> float:
        """Construct the caught-mast state: cradle written deployed at the
        solve's target, mast written 2 deg above its rest-on-crest angle (it
        settles gently onto the saddle). Returns the expected rest angle."""
        d_c = db + c.deploy_off + c.plate_hx
        pr = c.phi_rest_deg(d_c)
        write_body(scene.cradle, db + c.deploy_off, 0.0, c.court_z1 + 0.002)
        write_mast(pr - 2.0)
        _step(180)
        return pr

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(11)
    env.reset()
    _step(150)
    _report("reset")
    db1, py1 = float(scene.layout[0, 0]), float(scene.layout[0, 1])
    b1 = (scene.bottle.data.root_pos_w - scene.env_origins)[0]
    cr1 = (scene.cradle.data.root_pos_w - scene.env_origins)[0]
    phi1 = math.degrees(float(scene.phi()[0]))
    check("settle/no-NaN: mast on its back stop, bottle upright at its sampled "
          "marker, cradle at its rack, score ~0, no success",
          bool(scene._finite()[0])
          and c.limit_lo_deg - 0.8 <= phi1 <= -4.0
          and bool(scene.bottle_ok()[0])
          and abs(float(b1[0]) - db1) < 0.01 and abs(float(b1[1]) - py1) < 0.01
          and abs(float(cr1[0]) - float(scene.layout[0, 2])) < 0.02
          and abs(float(cr1[1]) - float(scene.layout[0, 3])) < 0.02
          and not bool(scene.deployed()[0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 2+3. randomization readback ================================================
    dbs, pys, rxs, ryaws, marker_tracks, rack_tracks = [], [], [], [], [], []
    for k in range(8):
        torch.manual_seed(20 + k)
        env.reset()
        _step(60)
        _refresh()
        lay = scene.layout[0]
        db, py, rx, ry, ryaw = (float(lay[i]) for i in range(5))
        dbs.append(db)
        pys.append(py)
        rxs.append(rx)
        ryaws.append(ryaw)
        b = (scene.bottle.data.root_pos_w - scene.env_origins)[0]
        pd = (scene.pad.data.root_pos_w - scene.env_origins)[0]
        cr = (scene.cradle.data.root_pos_w - scene.env_origins)[0]
        marker_tracks.append(abs(float(pd[0]) - db) < 0.005 and abs(float(pd[1]) - py) < 0.005
                             and abs(float(b[0]) - db) < 0.01 and abs(float(b[1]) - py) < 0.01
                             and bool(scene.bottle_ok()[0]))
        rack_tracks.append(abs(float(cr[0]) - rx) < 0.02 and abs(float(cr[1]) - ry) < 0.02)
    db_span = max(dbs) - min(dbs)
    py_span = max(pys) - min(pys)
    rx_span = max(rxs) - min(rxs)
    yaw_span = max(ryaws) - min(ryaws)
    print(f"[smoke] readback: db span={db_span:.3f} py span={py_span:.3f} "
          f"rx span={rx_span:.3f} yaw span={yaw_span:.2f} "
          f"marker_tracks={all(marker_tracks)} rack_tracks={all(rack_tracks)}", flush=True)
    check("randomization A: the bottle marker (db, py) varies across resets "
          "and the live pad + standing bottle track the sample every time",
          db_span > 0.03 and py_span > 0.008 and all(marker_tracks))
    check("randomization B: the cradle rack spot varies across resets "
          "(x span + free yaw) and the live cradle tracks the sample",
          rx_span > 0.03 and yaw_span > 0.8 and all(rack_tracks))

    # ================= 4. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(240)
    _report("null")
    check("null policy: 240 idle steps -> mast still leaning on its back stop, "
          "bottle untouched, score ~0, no success",
          math.degrees(float(scene.phi()[0])) < -4.0
          and bool(scene.bottle_ok()[0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 5. fell-first (the physics-forced order, real torque) =====================
    torch.manual_seed(41)
    env.reset()
    _step(120)
    push_mast()
    _step(600)  # hands off: sweep through the band, past it, onto bottle cap or floor
    _report("fell-first")
    phi5 = math.degrees(float(scene.phi()[0]))
    check("fell-first: the real solve-grade push with the cradle still racked "
          "sends the mast THROUGH the catch band and past it — it ends either "
          "propped on the bottle's cap or on the floor, both OUT of band — "
          "arrest never latches, score exactly the commit credit, no success",
          phi5 >= c.phi_hi_deg + 2.0
          and not bool(scene._arrest_l[0])
          and not bool(scene._deploy_l[0])
          and abs(float(scene.score()[0]) - c.w_commit) <= 1e-4
          and not succ())

    # ================= 6. bottle lying (caught mast, bottle gate) ================================
    torch.manual_seed(51)
    env.reset()
    _step(120)
    db6 = float(scene.layout[0, 0])
    # the bottle lies on the court off to the side, clear of the construct
    write_body(scene.bottle, db6, 0.18, c.court_z1 + c.bottle_r + 0.002,
               pitch=math.pi / 2)
    construct_caught(db6)
    _report("bottle-lying")
    check("bottle lying: the caught-mast state constructed but the bottle "
          "LYING on the court — band+settled hold, the bottle gate refuses: "
          "no success",
          bool(scene.in_band()[0])
          and abs(float(scene.mast_w()[0])) < c.settle_ang
          and not bool(scene.bottle_ok()[0])
          and float(scene.score()[0]) <= 0.60 + 1e-4
          and not succ())

    # ================= 7. displaced bottle (xy gate) =============================================
    torch.manual_seed(61)
    env.reset()
    _step(120)
    db7, py7 = float(scene.layout[0, 0]), float(scene.layout[0, 1])
    write_body(scene.bottle, db7, py7 + 0.07, c.court_z1 + c.bottle_h / 2 + 0.002)
    construct_caught(db7)
    _report("displaced")
    check("displaced bottle: caught mast, bottle STANDING upright but 7 cm off "
          "its marker — the xy gate refuses: no success",
          bool(scene.in_band()[0])
          and not bool(scene.bottle_ok()[0])
          and not succ())

    # ================= 8. perched bottle (rest-height gate) ======================================
    torch.manual_seed(71)
    env.reset()
    _step(120)
    db8, py8 = float(scene.layout[0, 0]), float(scene.layout[0, 1])
    # park the bottle OFF its marker first (standing, clear of the construct) so the
    # construct's settle window never passes through a genuine success state
    write_body(scene.bottle, 0.66, 0.30, c.court_z1 + c.bottle_h / 2 + 0.002)
    pr8 = construct_caught(db8)
    th8 = math.radians(pr8 - 90.0)
    beam_top = c.hinge_z - db8 * math.tan(th8) + c.shaft_hx / math.cos(th8)
    write_body(scene.bottle, db8, py8, beam_top + c.bottle_h / 2 + 0.003)
    _step(150)
    _report("perched")
    b8 = (scene.bottle.data.root_pos_w - scene.env_origins)[0]
    check("perched bottle: caught mast with the bottle stood ON the beam over "
          "the marker xy — the rest-height gate refuses: no success",
          bool(scene.in_band()[0])
          and float(b8[2]) > c.bottle_rest_z() + 0.10
          and not bool(scene.bottle_ok()[0])
          and not succ())

    # ================= 9. deploy-only (real press, nothing else) =================================
    torch.manual_seed(81)
    env.reset()
    _step(120)
    deploy_real(float(scene.layout[0, 0]))
    _step(120)
    _report("deploy-only")
    check("deploy-only: the solve's real hover-drop + press stands the cradle "
          "deployed and it stays put hands-off — but the mast never moved: "
          "score exactly the deploy credit, no success",
          bool(scene.deployed()[0])
          and math.degrees(float(scene.phi()[0])) < -4.0
          and abs(float(scene.score()[0]) - c.w_deploy) <= 1e-4
          and not succ())

    # ================= 10. caught but moving (settle gate, judged pre-step) ======================
    torch.manual_seed(91)
    env.reset()
    _step(120)
    db10 = float(scene.layout[0, 0])
    d_c10 = db10 + c.deploy_off + c.plate_hx
    write_body(scene.cradle, db10 + c.deploy_off, 0.0, c.court_z1 + 0.002)
    write_mast(c.phi_rest_deg(d_c10), wy=0.40)  # the success pose, but SWINGING
    in_band_now = bool(scene.in_band()[0])
    bok_now = bool(scene.bottle_ok()[0])
    moving_rejected = not succ()  # judged WITHOUT stepping: right pose, still moving
    # break the state up BEFORE stepping (a settled copy would be a real success)
    write_body(scene.bottle, 0.62, 0.30, c.court_z1 + c.bottle_h / 2 + 0.002)
    write_mast(c.phi_rest_deg(d_c10) - 2.0)
    _step(120)
    _report("moving")
    check("caught but moving: the success pose written WITH 0.4 rad/s on the "
          "hinge — the settle gate refuses success on the moving state",
          in_band_now and bok_now and moving_rejected and not succ())

    # ================= 11-13. audit, no-NaN, video ================================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.mast_lowering")
        print(f"[smoke] wrote {args.out}: {arr.shape}", flush=True)
    check("video: >10 frames recorded", len(_REC["frames"]) > 10)

    n_pass = sum(1 for _, ok in checks if ok)
    if n_pass == len(checks):
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
        code = 0
    else:
        for nm, ok in checks:
            if not ok:
                print(f"[smoke] FAILED: {nm}", flush=True)
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        code = 1
    # Hard exit: Kit teardown hangs — watchdog then die.
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
    except Exception as e:  # noqa: BLE001 - fast fail beats a 20-min watchdog hang
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)

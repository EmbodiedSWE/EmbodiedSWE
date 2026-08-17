"""Smoke battery for WeightCrankScene (sim_gen task
`libero_kitchen_scene5_close_the_top_drawer_of_the_cabinet_i163`) — REJECTION-ONLY:
every check either verifies basic health/randomization or CONSTRUCTS/EXECUTES a
wrong outcome and asserts the rubric refuses it. success() must never fire
anywhere in the battery. (The drawer and the crank are JOINTED to the kinematic
station — prismatic/revolute limits are the hard stops — so "stolen drawer" /
"off-pivot crank" fakes are structurally impossible and are guarded, not
constructed.)

 1. settle/no-NaN     — crank resting on its up-stop (~0 deg), drawer at its
                        sampled opening q0, cubes staged on the apron, score ~0,
                        no success.
 2. randomization A   — the drawer opening q0 varies across resets and the
                        SETTLED drawer tracks the sample every time (readback).
 3. randomization B   — the cubes' apron poses vary (weight x span) and the
                        side split (weight left vs right) is seen BOTH ways,
                        decoy always opposite; cubes settle ON the apron.
 4. null policy       — 240 idle steps -> score ~0, no success.
 5. SEED strategy     — the seed's whole skill (push the drawer shut) executed
                        for REAL: an 8 N PD push on the drawer front. The drawer
                        seats — but the pan is empty and the crank stays up: no
                        success, drawer credit only (~0.35).
 6. latch regression  — (same episode) the drawer is pulled back OPEN with a
                        real 8 N PD force: the drawer-progress latch survives,
                        the live clause reads open again, no success.
 7. DECOY drop        — mass discrimination: the foam decoy dropped into the
                        receiving pocket for real. It lands and stays — but its
                        torque never reaches half the crank's bias: the crank
                        does not move, the drawer does not move, score ~0.
 8. hand-crank drive  — a real (force-limited ~1.5 N.m) torque servo swings the
                        crank to its down-stop: the roller nose cam-presses the
                        drawer fully shut (the transmission is REAL). Torque
                        released -> the unloaded bias swings the crank back UP;
                        the drawer stays seated. Latched credit ~0.60 (crank +
                        drawer fractions), never success: the pan stays empty.
 9. stolen-weight fake— crank CONSTRUCTED at its down-stop with the drawer
                        seated but the weight far away on the ground: the load
                        clause refuses instantly, and physics agrees — the
                        unloaded bias returns the crank to its up-stop.
10. rejection audit   — success() observed False at every step of the battery.
11. final no-NaN.
12. video             — frames.npz (>10 frames) written to the CWD.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene5_close_the_top_drawer_of_the_cabinet_i163.smoke --headless
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

# ----- module state wired up in main() ------------------------------------------------------------
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
    print(f"[smoke] {tag:16s} | q={float(scene.drawer_open()[0]):+.4f} "
          f"theta={math.degrees(float(scene.crank_theta()[0])):+.2f}deg "
          f"onpan={bool(scene.weight_on_pan()[0])} "
          f"latch(l/c/d)=({int(scene._floaded[0])},{float(scene._fcrank[0]):.2f},"
          f"{float(scene._fdrawer[0]):.2f}) "
          f"settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.weight_crank_cabinet")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.75, -1.45, 1.30)) + o),
                                tuple(np.array((0.05, 0.00, 0.55)) + o),
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

    def origin(dx: float, dy: float, dz: float) -> torch.Tensor:
        return (scene.env_origins
                + torch.tensor([dx, dy, dz], device=device)).expand(n, 3)

    def theta_deg() -> float:
        return math.degrees(float(scene.crank_theta()[0]))

    def q_now() -> float:
        return float(scene.drawer_open()[0])

    def qy(theta_rad: float) -> torch.Tensor:
        """wxyz quat for a rotation about +Y (the crank's pivot axis)."""
        q = torch.zeros(n, 4, device=device)
        q[:, 0] = math.cos(theta_rad / 2)
        q[:, 2] = math.sin(theta_rad / 2)
        return q

    def drop_into_pocket(body) -> None:
        """REAL physics: the body released just above the rubric's load box over
        the pocket mouth (same transport point the solve uses) — gravity carries
        it down into the receiving pocket."""
        pkt_cx = (c.pkt_in_x0 + c.pkt_in_x1) / 2
        pos = scene.crank.data.root_pos_w.clone()
        pos[:, 0] += pkt_cx
        pos[:, 2] += c.load_z[1] + c.cube_s / 2 + 0.004
        _write_body(body, pos)
        _step(240)

    def pd_drawer(q_target: float, steps: int, clamp: float = 8.0) -> None:
        """REAL actuation: a force-limited PD push/pull on the drawer along its
        slide axis (the seed's whole skill), what a hand would apply."""
        zero = torch.zeros(n, 1, 3, device=device)
        for _ in range(steps):
            v = scene.drawer.data.root_lin_vel_w[:, 0]
            f = (200.0 * (q_target - scene.drawer_open()) - 30.0 * v).clamp(-clamp, clamp)
            fw = torch.zeros(n, 3, device=device)
            fw[:, 0] = f
            scene.drawer.set_external_force_and_torque(fw.reshape(n, 1, 3), zero)
            _step(1)
        scene.drawer.set_external_force_and_torque(zero, zero)
        _step(90)

    def crank_servo(theta_target_deg: float, steps: int, clamp: float = 1.5) -> None:
        """REAL actuation: a force-limited torque servo about the pivot axis (a
        hand pressing the pan arm down), ramped to the target angle. A pure
        Y-torque on a body that only rotates about Y is frame-encoding
        invariant."""
        zero = torch.zeros(n, 1, 3, device=device)
        th0 = float(scene.crank_theta()[0])
        tgt_final = math.radians(theta_target_deg)
        for i in range(steps):
            ramp = min(1.0, (i + 1) / (0.6 * steps))
            tgt = th0 + (tgt_final - th0) * ramp
            w = scene.crank.data.root_ang_vel_w[:, 1]
            tau = (3.0 * (tgt - scene.crank_theta()) - 1.0 * w).clamp(-clamp, clamp)
            tw = torch.zeros(n, 3, device=device)
            tw[:, 1] = tau
            scene.crank.set_external_force_and_torque(zero, tw.reshape(n, 1, 3))
            _step(1)
        scene.crank.set_external_force_and_torque(zero, zero)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(11)
    env.reset()
    _step(150)
    _report("reset")
    w_loc = (scene.weight.data.root_pos_w - scene.env_origins)[0]
    d_loc = (scene.decoy.data.root_pos_w - scene.env_origins)[0]
    apron_top = c.apron_z1 + c.cube_s / 2
    check("settle/no-NaN: crank resting on its up-stop, drawer at its sampled q0, "
          "cubes staged on the apron, score ~0, no success",
          bool(scene._finite()[0]) and abs(theta_deg()) < 2.0
          and bool(scene.crank_on_pivot()[0]) and bool(scene.drawer_in_channel()[0])
          and abs(q_now() - float(scene.q0[0])) < 0.006
          and abs(float(w_loc[2]) - apron_top) < 0.012
          and abs(float(d_loc[2]) - apron_top) < 0.012
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 2+3. randomization readback ================================================
    q0s, wxs, wsides, dsides, tracks, staged = [], [], [], [], True, True
    for k in range(10):
        torch.manual_seed(20 + k)
        env.reset()
        _step(90)
        _refresh()
        q0s.append(float(scene.q0[0]))
        wl = (scene.weight.data.root_pos_w - scene.env_origins)[0]
        dl = (scene.decoy.data.root_pos_w - scene.env_origins)[0]
        wxs.append(float(wl[0]))
        wsides.append(1.0 if float(wl[1]) > 0 else -1.0)
        dsides.append(1.0 if float(dl[1]) > 0 else -1.0)
        tracks = tracks and abs(q_now() - float(scene.q0[0])) < 0.006
        staged = staged and abs(float(wl[2]) - apron_top) < 0.012 \
            and abs(float(dl[2]) - apron_top) < 0.012
    qspan = max(q0s) - min(q0s)
    wxspan = max(wxs) - min(wxs)
    opposite = all(w * d < 0 for w, d in zip(wsides, dsides))
    print(f"[smoke] readback: q0={[f'{q * 1000:.0f}' for q in q0s]}mm span={qspan * 1000:.0f}mm "
          f"wx_span={wxspan * 1000:.0f}mm wsides={wsides} opposite={opposite} "
          f"tracks={tracks} staged={staged}", flush=True)
    check("randomization A: the drawer opening q0 varies across resets and the "
          "settled drawer tracks the sample every time",
          qspan > 0.012 and tracks)
    check("randomization B: cube apron poses vary, the side split is seen both "
          "ways with the decoy always opposite, and the cubes settle on the apron",
          wxspan > 0.040 and (1.0 in wsides) and (-1.0 in wsides)
          and opposite and staged)

    # ================= 4. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(240)
    _report("null")
    check("null policy: 240 idle steps -> score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 5. SEED strategy (real push on the drawer front) ==========================
    torch.manual_seed(41)
    env.reset()
    _step(150)
    pd_drawer(0.0, 420)          # the seed's whole skill: hand-push the drawer shut
    _report("seed-skill")
    s5 = float(scene.score()[0])
    check("SEED strategy: the seed's whole skill (push the drawer shut) executed "
          "for real — the drawer seats but the pan is empty and the crank stays "
          "up: no success, drawer credit only (~0.35)",
          q_now() <= c.q_goal and abs(theta_deg()) < 3.0
          and not bool(scene.weight_on_pan()[0])
          and 0.30 <= s5 <= 0.45 and not succ())

    # ================= 6. latch regression (pull the drawer back open) ===========================
    fd_latched = float(scene._fdrawer[0])
    q0_6 = float(scene.q0[0])
    pd_drawer(q0_6, 420)         # real 8 N pull back toward the sampled opening
    _report("latch-regress")
    check("latch regression: the drawer pulled back open with a real force — the "
          "drawer-progress latch survives, the live clause reads open again, no "
          "success",
          fd_latched >= 0.90
          and float(scene._fdrawer[0]) >= fd_latched - 1e-4
          and q_now() >= 0.5 * q0_6
          and float(scene.score()[0]) >= c.w_drawer * fd_latched - 1e-4
          and not succ())

    # ================= 7. DECOY drop (mass discrimination) ========================================
    torch.manual_seed(51)
    env.reset()
    _step(150)
    q_before = q_now()
    drop_into_pocket(scene.decoy)
    _report("decoy-drop")
    dl = scene._crank_local(scene.decoy)[0]
    check("DECOY drop: the foam decoy dropped into the receiving pocket for real "
          "— it lands and stays, but the crank does not move and the drawer does "
          "not move: score ~0, no success",
          float(dl[0]) > c.pkt_in_x0 - 0.02 and float(dl[0]) < c.pkt_in_x1 + 0.02
          and float(dl[2]) < c.pkt_top
          and abs(theta_deg()) < 3.0
          and abs(q_now() - q_before) < 0.010
          and float(scene.score()[0]) <= 0.08 and not succ())

    # ================= 8. hand-crank drive (the i98-style strategy, refused) =====================
    torch.manual_seed(61)
    env.reset()
    _step(150)
    # Target well PAST the stop: a pure P-servo stalls where K*(tgt-theta) equals
    # the bias load, so the error term must keep >=1.7x torque margin at the stop.
    crank_servo(c.theta_max_deg + 14.0, 540)  # hand presses the pan arm to the stop
    _report("hand-crank-held")
    th_held = theta_deg()
    q_held = q_now()
    _step(360)                                 # hands off: the bias takes over
    _report("hand-crank-free")
    s8 = float(scene.score()[0])
    check("hand-crank drive: a real force-limited torque servo swings the crank "
          "to its down-stop and the nose cam-presses the drawer fully shut (the "
          "transmission is real); released, the unloaded bias returns the crank "
          "to its up-stop — latched credit ~0.60, never success (pan empty)",
          th_held >= c.crank_goal_deg and q_held <= c.q_goal + 0.004
          and theta_deg() < 6.0
          and q_now() <= c.q_goal + 0.006
          and float(scene._fcrank[0]) >= 0.90
          and 0.50 <= s8 <= 0.65 and not succ())

    # ================= 9. stolen-weight fake ======================================================
    torch.manual_seed(71)
    env.reset()
    _step(150)
    _write_body(scene.weight, origin(1.10, -0.90, c.cube_s / 2 + 0.002))
    _write_body(scene.drawer, origin(0.004, 0.0, 0.0))
    _write_body(scene.crank, origin(c.pivot_x, 0.0, c.pivot_z),
                qy(math.radians(c.theta_max_deg)))
    _refresh()
    instant_refused = not succ() and not bool(scene.weight_on_pan()[0])
    _step(360)
    _report("stolen-weight")
    check("stolen-weight fake: crank constructed at its down-stop with the drawer "
          "seated but the weight far away — the load clause refuses instantly, "
          "and the unloaded bias swings the crank back to its up-stop",
          instant_refused and theta_deg() < 6.0
          and not bool(scene.weight_on_pan()[0])
          and float(scene.score()[0]) <= 0.65 and not succ())

    # ================= 10-12. audit, no-NaN, video ================================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.weight_crank_cabinet")
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

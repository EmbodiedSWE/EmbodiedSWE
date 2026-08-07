"""Smoke / rubric-REJECTION battery for RampChockScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts the constructed equilibrium and
the latched credit is monotone along a real trajectory). This battery proves the rubric
REJECTS wrong outcomes, and that the physical claims the task rests on are load-bearing:
a free sphere cannot rest anywhere on the bare 12-degree deck (so the chock is
NECESSARY, and release-order chock-first is enforced by gravity), the round decoy
cannot chock, and the judged equilibrium is maintained by live contact (remove the
chock and success collapses while latched credit survives). Every probe is CONSTRUCTED
as a state (teleport, real physics steps, judge) — instrumentation, never a solution.

Checks:
   1. settle/no-NaN     — seeded reset settles finite; all four loose bodies on the
                          floor (none on the ramp); score 0, no success;
   2. randomization     — two seeded resets: READBACK ramp yaw, ramp xy and ball_a
                          ground spawn all differ (the fall line must be read);
   3. null-policy       — 240 idle steps: score ~0, no success;
   4. NO-CHOCK          — ball_a released 8 mm above the band with no chock: it ROLLS
                          OFF the ramp (spheres are analytic — no rest pose on the
                          bare deck). The physics that makes the chock necessary and
                          the order gravity-enforced. No park, no success;
   5. SEED STRATEGY     — the seed's plan is transport-and-deliver-over-the-target
                          (grasp, carry along waypoints, end above the vase). Its end
                          state CONSTRUCTED here: both balls delivered onto the band,
                          nothing else -> both roll off; no success, score <= 0.11;
   6. DECOY chock       — the red ball wedged at the chock's spot, ball_a released
                          uphill of it: the round "chock" rolls off itself and the
                          chain collapses -> no chock credit, no park, no success;
   7. angled chock      — chock on the band yawed ~45 deg off cross-slope: the axis
                          clause refuses ch_zone; a ball released against it -> no
                          park, no success;
   8. below-zone        — chock wedged well BELOW the band (u ~ 0.08): ball_a rolls
                          back and rests against it — a REAL settled equilibrium,
                          centimetres downhill of the band -> rejected (no park, no
                          success): where matters, not just how;
   9. chock-only        — chock correctly wedged in the zone, balls untouched: score
                          ~0.15, no success;
  10. one-ball          — chock + ball_a parked, ball_b on the floor: score ~0.40
                          < 0.9, no success;
  11. REMOVE-CHOCK      — full success CONSTRUCTED (chock + both balls, success live),
                          then the chock teleported to the floor: both balls roll off,
                          success COLLAPSES while the latched score survives >= 0.65 —
                          the equilibrium is contact-maintained, not bookkeeping;
  12. rejection audit   — success() never fired at any judged step during the
                          negative probes (checks 4-10);
  13. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.pour_water_i7.smoke --headless
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

_qmul, _qy, _qz = task_scene._qmul, task_scene._qy, task_scene._qz

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
    env = ENVS.get("simgen.ramp_chock")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    th = math.radians(c.slope_deg)
    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.15, -0.85, 0.80)) + o),
                                tuple(np.array((0.30, 0.00, 0.08)) + o),
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

    def uvh(body) -> tuple[float, float, float]:
        _refresh()
        u, v, h = scene._slope_coords(body.data.root_pos_w)
        return float(u[0]), float(v[0]), float(h[0])

    def deck_pose(u: float, v: float, h: float, deck_quat: bool,
                  extra_yaw: float = 0.0) -> torch.Tensor:
        """(N,13) root state at ramp-frame slope coords (u, v, h), zero velocity."""
        _refresh()
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0] = u * math.cos(th) - h * math.sin(th)
        loc[:, 1] = v
        loc[:, 2] = u * math.sin(th) + h * math.cos(th)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.ramp.data.root_pos_w + quat_apply(scene.ramp.data.root_quat_w, loc)
        if deck_quat:
            q = _qmul(scene.ramp.data.root_quat_w,
                      _qz(torch.full((n,), extra_yaw, device=device)))
            st[:, 3:7] = _qmul(q, _qy(torch.full((n,), -th, device=device)))
        else:
            st[:, 3] = 1.0
        return st

    def off_ramp(body) -> bool:
        u, v, h = uvh(body)
        return h < -0.005 or u < 0.02 or abs(v) > c.deck_w / 2 + 0.03

    def report(tag: str) -> None:
        s = scene._status()
        cu, cv, chh = uvh(scene.chock)
        parts = []
        for nm in scene.BALL_NAMES:
            u, v, h = uvh(scene.balls[nm])
            parts.append(f"{nm}=(u{u:+.3f},v{v:+.3f},h{h:+.3f})")
        print(f"[smoke] {tag:16s} | chock=(u{cu:+.3f},v{cv:+.3f},h{chh:+.3f}) "
              + " ".join(parts)
              + f" zone={bool(s['ch_zone'][0])} park={s['park'][0].tolist()} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(_REC['frames'])}", flush=True)

    def wedge_chock(u: float, extra_yaw: float = 0.0) -> None:
        scene.chock.write_root_state_to_sim(
            deck_pose(u, 0.0, c.chock_h / 2 + 0.008, deck_quat=True,
                      extra_yaw=extra_yaw), _all_ids())
        _step(150)

    def drop_ball(name: str, du: float = None, v: float = 0.0) -> None:
        """Release the ball 8 mm above the band, `du` uphill of the chock's face
        (or at the band center if du is None), then hands-off settle."""
        if du is None:
            u = (c.band_lo + c.band_hi) / 2
        else:
            cu, cv, _ = uvh(scene.chock)
            u = cu + c.chock_w / 2 + c.ball_r + du
            v = cv + v
        scene.balls[name].write_root_state_to_sim(
            deck_pose(u, v, c.ball_r + 0.008, deck_quat=False), _all_ids())
        _step(300)

    u_zone = c.band_lo + 0.005  # the correct chock station (as in solve.py)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    report("settle")
    _REC["on"] = False
    bodies = [scene.chock, scene.decoy, *scene.balls.values()]
    finite = all(bool(torch.isfinite(b.data.root_pos_w).all()) for b in bodies)
    on_floor = all(off_ramp(b) for b in bodies)
    check("settle/no-NaN: seeded reset settles finite, all four loose bodies on the "
          "floor (none on the ramp), score 0, no success",
          finite and on_floor and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (yaw_of(scene.ramp.data.root_quat_w[0]),
                scene.ramp.data.root_pos_w[0, :2].clone(),
                scene.balls["ball_a"].data.root_pos_w[0, :2].clone())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_rp, a_bp = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_rp, b_bp = readback()
    d_yawv = dyaw(a_yaw, b_yaw)
    d_rp, d_bp = float((a_rp - b_rp).norm()), float((a_bp - b_bp).norm())
    print(f"[smoke] randomization deltas: ramp_yaw={d_yawv:.1f}deg "
          f"ramp_xy={d_rp * 1000:.1f}mm ball_a_xy={d_bp * 1000:.1f}mm", flush=True)
    check("randomization-is-real: ramp yaw, ramp xy and ball_a spawn readback differ",
          d_yawv > 5.0 and d_rp > 0.003 and d_bp > 0.03)

    # ================= 3. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    _AUDIT["on"] = True  # ---- negative probes below: success must never fire ----

    # ================= 4. NO-CHOCK: a sphere cannot rest on the bare deck =========================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    drop_ball("ball_a")  # band center, no chock anywhere on the ramp
    report("no-chock")
    _REC["on"] = False
    check("NO-CHOCK counterfactual: ball released on the band with no chock ROLLS "
          "OFF the ramp (no rest pose on the bare deck) — no park, no success, "
          "score <= 0.06",
          off_ramp(scene.balls["ball_a"]) and not bool(scene._parked[0].any())
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.06)

    # ================= 5. SEED STRATEGY: transport-and-deliver both balls =========================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    drop_ball("ball_a", du=None, v=-0.05)
    drop_ball("ball_b", du=None, v=+0.05)
    _step(120)
    report("seed-strategy")
    _REC["on"] = False
    check("negative (SEED strategy): both balls delivered onto the band and released "
          "— transport alone leaves nothing standing; both roll off, no success, "
          "score <= 0.11",
          off_ramp(scene.balls["ball_a"]) and off_ramp(scene.balls["ball_b"])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.11)

    # ================= 6. DECOY as chock ==========================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    scene.decoy.write_root_state_to_sim(
        deck_pose(u_zone, 0.0, c.decoy_r + 0.008, deck_quat=False), _all_ids())
    _step(60)  # the round "chock" is already rolling away
    drop_ball("ball_a", du=None)
    report("decoy-chock")
    _REC["on"] = False
    check("negative (DECOY): the red ball wedged at the chock station rolls off "
          "itself; the green ball released uphill follows — no chock credit, no "
          "park, no success",
          off_ramp(scene.decoy) and off_ramp(scene.balls["ball_a"])
          and not bool(scene._chock_zone[0]) and not bool(scene._parked[0].any())
          and not bool(scene.success()[0]))

    # ================= 7. angled chock refused ====================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    wedge_chock(u_zone, extra_yaw=math.radians(45.0))
    s = scene._status()
    zone_45 = bool(s["ch_zone"][0])
    drop_ball("ball_a", du=0.012)
    report("angled-chock")
    check("negative (angled chock): chock yawed 45 deg off cross-slope on the band — "
          "axis clause refuses ch_zone; ball released against it earns no park, no "
          "success",
          not zone_45 and not bool(scene._parked[0].any())
          and not bool(scene.success()[0]))

    # ================= 8. real equilibrium BELOW the band =========================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    wedge_chock(0.08)  # well below band_lo - zone_lo_slack = 0.16
    drop_ball("ball_a", du=0.012)
    _step(120)
    ua, va, ha = uvh(scene.balls["ball_a"])
    lv = float(scene.balls["ball_a"].data.root_lin_vel_w.norm(dim=-1)[0])
    report("below-zone")
    _REC["on"] = False
    check("near-miss (below zone): chock wedged below the band, ball rests against "
          "it — a REAL settled equilibrium (on deck, still) centimetres downhill of "
          "the band — rejected: no park, no success",
          abs(ha - c.ball_r) < 0.012 and lv < 0.05 and ua < c.band_lo
          and not bool(scene._parked[0].any()) and not bool(scene.success()[0]))

    # ================= 9. chock only ==============================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    wedge_chock(u_zone)
    report("chock-only")
    check("chock-only: correctly wedged chock without balls scores ~0.15, no success",
          bool(scene._chock_zone[0]) and not bool(scene.success()[0])
          and 0.14 <= float(scene.score()[0]) <= 0.20)

    # ================= 10. one ball only ==========================================================
    # (continues from check 9's wedged chock)
    drop_ball("ball_a", du=0.012, v=-0.045)
    report("one-ball")
    check("one-ball: chock + one parked ball scores ~0.40 (< 0.9), no success",
          bool(scene._parked[0, 0]) and not bool(scene.success()[0])
          and 0.35 <= float(scene.score()[0]) <= 0.55)

    _AUDIT["on"] = False  # ---- end of negative probes ----

    # ================= 11. REMOVE-CHOCK: success collapses, latches survive =======================
    # (continues from check 10: park ball_b too -> full success, then yank the chock)
    _REC["on"] = True
    drop_ball("ball_b", du=0.012, v=+0.045)
    report("full-chain")
    built = bool(scene.success()[0])
    score_before = float(scene.score()[0])
    floor = torch.zeros(n, 13, device=device)
    floor[:, 0:3] = env.iscene.env_origins
    floor[:, 0] += -0.45
    floor[:, 1] += -0.45
    floor[:, 2] += c.chock_h / 2
    floor[:, 3] = 1.0
    scene.chock.write_root_state_to_sim(floor, _all_ids())
    _step(300)
    report("chock-removed")
    _REC["on"] = False
    collapsed = not bool(scene.success()[0])
    balls_gone = off_ramp(scene.balls["ball_a"]) and off_ramp(scene.balls["ball_b"])
    score_after = float(scene.score()[0])
    check("REMOVE-CHOCK: constructed success is live (chock + both balls), then the "
          "chock teleported away -> both balls roll off, success COLLAPSES while the "
          "latched score survives >= 0.65",
          built and score_before == 1.0 and collapsed and balls_gone
          and 0.64 <= score_after <= 0.66)

    # ================= 12. rejection audit ========================================================
    check("rejection audit: success() never fired at any step of the negative "
          "probes (checks 4-10)", not _AUDIT["fired"])

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.ramp_chock")
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

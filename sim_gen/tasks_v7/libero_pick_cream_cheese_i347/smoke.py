"""Smoke / rubric-REJECTION battery for BalanceVerdictScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a real weigh-then-deliver
trajectory with monotone latched credit, two+ seeds). This battery proves the rubric
REJECTS wrong outcomes, and that the claims the task rests on — the balance is a real
instrument whose display cannot be forged, the empty beam self-levels, the weighing is
MANDATORY — are physics, not fiat. Every probe is CONSTRUCTED (teleport, real physics
steps, judge) — instrumentation, never a solution; force probes assert the actuator
actually moved (no vacuous rejections).

Checks:
   1. settle/no-NaN      — seeded reset settles finite; empty beam LEVEL, cartons on
                           their ground spots, basket upright on the ground; score ~0;
   2. null-policy        — 240 more idle steps: beam stays level, score ~0, no success;
   3. randomization      — three seeded resets: max pairwise readback deltas of stand
                           yaw/xy, basket xy/yaw and carton xy all real;
   4. side swap          — over 10 resets the HEAVY carton starts on BOTH spots, and
                           its readback side always matches the hidden draw;
   5. single-carton      — ONE carton alone tips the beam to its stop (the balance is
                           sensitive — assert it moved) yet loaded/weighed stay False:
                           a one-sided load is not a weighing;
   6. forged verdict     — with both cartons loaded, an external press HOLDS the
                           EMPTY side down (display false, assert >= 8 deg the wrong
                           way): `weighed` never latches — the truth clause is load-
                           bearing, the instrument cannot be forged;
   7. self-truth         — the press released, gravity alone swings the beam to the
                           HEAVY side and the truthful verdict latches (0.40);
   8. keel self-levels   — cartons lifted off: the unloaded beam returns level by its
                           own keel;
   9. SEED STRATEGY foul — the seed's plan (pick a carton, place it in the basket):
                           the CORRECT carton teleported straight into the basket
                           unweighed -> permanent FOUL, score ~0; a full weighing
                           performed afterwards cannot atone (credit stays frozen,
                           re-delivery never succeeds);
  10. wrong carton       — after a LEGAL weighing the EMPTY carton is delivered
                           instead: no success, delivery credit never latches;
  11. contamination live — the full carton added too (both inside): still no success;
                           the empty one removed -> success RETURNS (live clause);
  12. containment miss   — the full carton set on the ground against the basket's
                           OUTER wall: not "inside"; re-dropped in -> success again
                           (post-weighing moves are free);
  13. legal run          — fresh seed: load both pans, read the verdict, deliver the
                           full carton -> success; success never fired while the
                           carton was still moving fast;
  14. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_pick_cream_cheese_i347.smoke --headless
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
    from . import scene as task_scene  # noqa: F401 - importing registers the scene/env
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

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
    from isaaclab.utils.math import quat_apply_inverse

    hl = quat_apply_inverse(scene.stand.data.root_quat_w,
                            scene.heavy.data.root_pos_w - scene.stand.data.root_pos_w)[0]
    ll = quat_apply_inverse(scene.stand.data.root_quat_w,
                            scene.light.data.root_pos_w - scene.stand.data.root_pos_w)[0]
    print(f"[smoke] {tag:16s} | angle={float(scene.beam_angle_deg()[0]):+.1f}deg "
          f"down={float(scene.down_side()[0]):+.0f} "
          f"heavy_std=({float(hl[0]):+.3f},{float(hl[1]):+.3f},{float(hl[2]):+.3f}) "
          f"light_std=({float(ll[0]):+.3f},{float(ll[1]):+.3f},{float(ll[2]):+.3f}) "
          f"loaded={bool(scene._loaded[0])} weighed={bool(scene._weighed[0])} "
          f"delivered={bool(scene._delivered[0])} foul={bool(scene._foul[0])} "
          f"in_h={bool(scene.in_basket(scene.heavy.data.root_pos_w)[0])} "
          f"in_l={bool(scene.in_basket(scene.light.data.root_pos_w)[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.balance_verdict")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.55, -0.80, 0.62)) + o),
                                tuple(np.array((0.28, -0.05, 0.14)) + o),
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

    def angle() -> float:
        _refresh()
        return float(scene.beam_angle_deg()[0])

    def stand_world(loc_xyz) -> torch.Tensor:
        """Stand-local point -> world (per-env)."""
        _refresh()
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.stand.data.root_pos_w + quat_apply(scene.stand.data.root_quat_w, loc)

    def heavy_side() -> float:
        """Stand-local y sign of the heavy carton (readback, not the hidden draw)."""
        from isaaclab.utils.math import quat_apply_inverse

        _refresh()
        loc = quat_apply_inverse(scene.stand.data.root_quat_w,
                                 scene.heavy.data.root_pos_w - scene.stand.data.root_pos_w)
        return 1.0 if float(loc[0, 1]) > 0 else -1.0

    def drop_on_pan(body, side_sign: float, drop: float = 0.015) -> None:
        """Transport-only: body to free space above the (live) pan floor centre,
        oriented with the beam; it falls in by gravity."""
        _refresh()
        side = torch.full((n,), float(side_sign), device=device)
        pan = scene.pan_center_w(side)
        up = quat_apply(scene.beam.data.root_quat_w,
                        torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(n, 3))
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pan + up * (c.box_size[2] / 2 + drop)
        st[:, 3:7] = scene.beam.data.root_quat_w
        body.write_root_state_to_sim(st, _all_ids())
        _refresh()

    def basket_drop_point(dz: float = 0.145) -> torch.Tensor:
        _refresh()
        p = scene.basket.data.root_pos_w.clone()
        p[:, 2] += dz
        return p

    _zero_w = torch.zeros(1, 1, 3, device=device).expand(n, 1, 3).contiguous()

    def press_beam(tau: float) -> None:
        """Pure torque about the hinge axis (stand-local X in world — the only axis
        the beam can turn about, so the wrench is frame-drag invariant)."""
        if tau == 0.0:
            scene.beam.set_external_force_and_torque(_zero_w, _zero_w, env_ids=_all_ids())
        else:
            ax = quat_apply(scene.stand.data.root_quat_w,
                            torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))
            scene.beam.set_external_force_and_torque(
                _zero_w, (tau * ax).view(n, 1, 3), env_ids=_all_ids(), is_global=True)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(120)
    _report("settle")
    _REC["on"] = False
    check("settle/no-NaN: layout settles finite; the empty beam is LEVEL (keel), "
          "cartons on their ground spots, basket upright on the ground; score ~0",
          bool(scene._finite()[0]) and abs(angle()) < 3.0
          and not bool(scene.on_pan(scene.heavy, 1.0)[0])
          and not bool(scene.on_pan(scene.heavy, -1.0)[0])
          and bool(scene.basket_upright()[0]) and bool(scene.basket_on_ground()[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. null policy fails =======================================================
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 more idle steps — beam stays level, cartons stay "
          "on the ground, score ~0, no success",
          abs(angle()) < 3.0 and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))

    # ================= 3. randomization is real (readback, 3-seed max-pairwise) ===================
    def readback():
        _refresh()
        return (yaw_of(scene.stand.data.root_quat_w[0]),
                scene.stand.data.root_pos_w[0, :2].clone(),
                scene.basket.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.basket.data.root_quat_w[0]),
                scene.heavy.data.root_pos_w[0, :2].clone())

    obs = []
    for sd in (101, 202, 303):
        torch.manual_seed(sd)
        env.reset()
        _step(10)
        obs.append(readback())
    pairs = [(0, 1), (0, 2), (1, 2)]
    d_yaw = max(dyaw(obs[i][0], obs[j][0]) for i, j in pairs)
    d_sp = max(float((obs[i][1] - obs[j][1]).norm()) for i, j in pairs)
    d_bp = max(float((obs[i][2] - obs[j][2]).norm()) for i, j in pairs)
    d_by = max(dyaw(obs[i][3], obs[j][3]) for i, j in pairs)
    d_hp = max(float((obs[i][4] - obs[j][4]).norm()) for i, j in pairs)
    print(f"[smoke] randomization max-pairwise deltas: stand_yaw={d_yaw:.1f}deg "
          f"stand_xy={d_sp * 1000:.1f}mm basket_xy={d_bp * 1000:.1f}mm "
          f"basket_yaw={d_by:.1f}deg heavy_xy={d_hp * 1000:.1f}mm", flush=True)
    check("randomization-is-real: stand yaw/xy, basket pose and carton position "
          "readbacks differ across seeds",
          d_yaw > 1.0 and d_sp > 0.002 and d_bp > 0.005 and d_by > 3.0 and d_hp > 0.005)

    # ================= 4. heavy-side swap =========================================================
    sides, consistent = set(), True
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _step(2)
        hs = heavy_side()
        sides.add("+" if hs > 0 else "-")
        consistent = consistent and (hs * float(scene.heavy_side_start[0]) > 0)
    print(f"[smoke] over 10 resets: heavy carton start sides {sorted(sides)}", flush=True)
    check("side swap: the heavy carton starts on BOTH spots over 10 resets, and its "
          "physical position always matches the hidden draw",
          sides == {"+", "-"} and consistent)

    # ================= 5-8: the instrument (one episode) ==========================================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    s_h = heavy_side()
    s_l = -s_h

    # --- 5. a single carton tips the beam (sensitive) but is NOT a weighing ---
    _REC["on"] = True
    drop_on_pan(scene.heavy, s_h)
    _step(240)
    _report("single-carton")
    a1 = angle()
    check("single-carton near-miss: ONE carton tips the beam to its stop (it moved: "
          f"|angle| {abs(a1):.1f} >= {c.tilt_min_deg} deg) yet loaded/weighed stay "
          "False and score stays 0 — a one-sided load is not a weighing",
          abs(a1) >= c.tilt_min_deg and not bool(scene._loaded[0])
          and not bool(scene._weighed[0]) and float(scene.score()[0]) <= 0.01)

    # --- 6. forged verdict: hold the EMPTY side down; weighed must never latch ---
    # Press the beam toward the light side (theta -> -s_l * stop): torque sign -s_l.
    forged = False
    for tau_sign in (-s_l, s_l):  # convention + safety flip; assert the press MOVED it
        for _ in range(180):
            press_beam(tau_sign * 0.60)
            _step(1)
        if float(scene.down_side()[0]) == s_l and abs(angle()) >= c.tilt_min_deg:
            forged = True
            break
    assert forged, f"probe setup: press failed to hold the empty side down (angle {angle():+.1f})"
    drop_on_pan(scene.light, s_l)  # onto the (now lower) empty-side pan
    weighed_during = False
    for _ in range(240):
        press_beam(tau_sign * 0.60)
        _step(1)
        weighed_during = weighed_during or bool(scene._weighed[0])
    _report("forged-press")
    display_false = (float(scene.down_side()[0]) == s_l) and abs(angle()) >= c.tilt_min_deg
    loaded_now = bool(scene._loaded[0])
    check("forged verdict rejected: both cartons loaded while a press holds the "
          "EMPTY side down (display false the whole time) — `weighed` never "
          "latches; the balance's verdict cannot be forged",
          display_false and loaded_now and not weighed_during
          and not bool(scene._weighed[0]) and float(scene.score()[0]) <= c.w_loaded + 0.01)

    # --- 7. released, gravity renders the TRUTHFUL verdict on its own ---
    press_beam(0.0)
    _step(300)
    _report("self-truth")
    check("instrument self-truth: the press released, gravity alone swings the beam "
          "to the HEAVY side and the truthful verdict latches (0.40)",
          float(scene.down_side()[0]) == s_h and abs(angle()) >= c.tilt_min_deg
          and bool(scene._weighed[0])
          and abs(float(scene.score()[0]) - (c.w_loaded + c.w_weighed)) < 0.01)

    # --- 8. unloaded, the keel re-levels the beam ---
    _write_body(scene.heavy, stand_world((0.26, s_h * 0.13, 0.050)))
    _write_body(scene.light, stand_world((0.26, s_l * 0.13, 0.050)))
    _step(360)
    _report("keel-relevel")
    _REC["on"] = False
    check("keel self-levels: cartons lifted off, the unloaded beam returns level "
          "by its own keel alone",
          abs(angle()) < 3.0)

    # ================= 9. negative: the SEED'S OWN STRATEGY = permanent foul ======================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    _write_body(scene.heavy, basket_drop_point(), scene.basket.data.root_quat_w)
    _step(150)
    _report("seed-strategy")
    fouled = bool(scene._foul[0]) and bool(scene.in_basket(scene.heavy.data.root_pos_w)[0]) \
        and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.01
    # atonement attempt: a full, tidy weighing AFTER the foul (pan side arbitrary)...
    s_h2 = 1.0
    drop_on_pan(scene.heavy, s_h2)
    _step(120)
    drop_on_pan(scene.light, -s_h2)
    _step(300)
    _report("foul-weigh")
    atone_blocked = bool(scene.loaded_live()[0]) and not bool(scene._weighed[0]) \
        and not bool(scene._loaded[0]) and float(scene.score()[0]) <= 0.01
    # ...and a re-delivery of the correct carton: still dead
    _write_body(scene.heavy, basket_drop_point(), scene.basket.data.root_quat_w)
    _step(200)
    _report("foul-redeliver")
    _REC["on"] = False
    check("negative (SEED strategy): the CORRECT carton teleported straight into the "
          "basket unweighed -> permanent FOUL, score ~0; a full weighing performed "
          "afterwards earns nothing (credit frozen) and re-delivery never succeeds",
          fouled and atone_blocked and bool(scene.in_basket(scene.heavy.data.root_pos_w)[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.01)

    # ================= 10-12: wrong carton / contamination / containment (one episode) ============
    torch.manual_seed(100)
    env.reset()
    _step(60)
    s_h = heavy_side()
    drop_on_pan(scene.heavy, s_h)
    _step(120)
    drop_on_pan(scene.light, -s_h)
    _step(300)
    assert bool(scene._weighed[0]), "probe setup: legal weighing must latch first"
    _REC["on"] = True

    # --- 10. deliver the WRONG (empty) carton ---
    _write_body(scene.light, basket_drop_point(), scene.basket.data.root_quat_w)
    _step(200)
    _report("wrong-carton")
    wrong = bool(scene.in_basket(scene.light.data.root_pos_w)[0]) \
        and not bool(scene.success()[0]) and not bool(scene._delivered[0]) \
        and abs(float(scene.score()[0]) - (c.w_loaded + c.w_weighed)) < 0.01
    check("wrong carton: after a LEGAL weighing the EMPTY carton is delivered "
          "instead — no success, the delivery credit never latches (score stays 0.40)",
          wrong)

    # --- 11. contamination is judged live ---
    # Construct BOTH cartons side-by-side on the basket floor (a centre drop lands
    # the heavy one perched ON the light one — z 0.091, correctly rejected by the
    # anti-perch window — so build the genuinely-shared state deterministically).
    bq = scene.basket.data.root_quat_w
    for body, bx, bz in ((scene.light, -0.042, 0.040), (scene.heavy, 0.042, 0.060)):
        p = scene.basket.data.root_pos_w \
            + quat_apply(bq, torch.tensor([[bx, 0.0, 0.0]], device=device).expand(n, 3))
        p = p.clone()
        p[:, 2] = scene.basket.data.root_pos_w[:, 2] + bz
        _write_body(body, p, bq)
    _step(200)
    _report("both-in")
    both_no = bool(scene.in_basket(scene.heavy.data.root_pos_w)[0]) \
        and bool(scene.in_basket(scene.light.data.root_pos_w)[0]) \
        and not bool(scene.success()[0])
    _write_body(scene.light, stand_world((0.60, -0.30, 0.040)))
    _step(240)
    _report("light-out")
    check("contamination live: with BOTH cartons inside there is no success; the "
          "empty one removed, success RETURNS (the exclusion clause is live)",
          both_no and bool(scene.success()[0]))

    # --- 12. containment near-miss: against the OUTER wall is not inside ---
    bq = scene.basket.data.root_quat_w
    beside = scene.basket.data.root_pos_w \
        + quat_apply(bq, torch.tensor([[0.125, 0.0, 0.0]], device=device).expand(n, 3))
    beside = beside.clone()
    beside[:, 2] = scene.env_origins[:, 2] + 0.033
    _write_body(scene.heavy, beside, bq)
    _step(180)
    _report("beside-wall")
    outside = not bool(scene.in_basket(scene.heavy.data.root_pos_w)[0]) \
        and not bool(scene.success()[0]) \
        and abs(float(scene.score()[0]) - 0.60) < 2e-3  # latched cap, no success
    _write_body(scene.heavy, basket_drop_point(), scene.basket.data.root_quat_w)
    _step(240)
    _report("back-in")
    _REC["on"] = False
    check("containment near-miss: the full carton resting on the ground against the "
          "basket's OUTER wall is not 'inside' (score = 0.60 latched cap); "
          "re-dropped in, success returns — post-weighing moves are free",
          outside and bool(scene.success()[0]))

    # ================= 13. legal run on a fresh seed (never mid-air) ==============================
    torch.manual_seed(400)
    env.reset()
    _step(90)
    _REC["on"] = True
    s_h = heavy_side()
    drop_on_pan(scene.heavy, s_h)
    _step(120)
    drop_on_pan(scene.light, -s_h)
    _step(300)
    verdict_ok = (float(scene.down_side()[0]) == s_h) and bool(scene._weighed[0])
    _write_body(scene.heavy, basket_drop_point(), scene.basket.data.root_quat_w)
    got, air_succ = False, False
    for _j in range(700):
        _step(1)
        _refresh()
        fast = float(scene.heavy.data.root_lin_vel_w[0].norm()) > 0.15
        if bool(scene.success()[0]) and fast:
            air_succ = True
        if bool(scene.success()[0]):
            got = True
            break
    _step(60)
    _report("legal-run")
    _REC["on"] = False
    check("legal run (fresh seed): load both pans, verdict points at the heavy "
          "carton, deliver it -> success; success never fired while the carton "
          "was still moving fast",
          verdict_ok and (got or bool(scene.success()[0])) and not air_succ)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.balance_verdict")
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

"""Smoke / rubric-REJECTION battery for CoinOpWasherScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the latched
credit is monotone along a real trajectory). This battery proves the rubric REJECTS
wrong outcomes and that the claims the task rests on are physically load-bearing: the
lock really blocks the door, the slug really cannot pay, the token really can, and a
hand-press on the empty lever earns no payment credit. Probes are CONSTRUCTED states
(teleport, real physics steps, judge) or wrench-slot pushes — instrumentation, never
a hidden solution.

Checks:
   1. settle/no-NaN   — seeded reset settles finite; door at its sampled crack, rocker
                        at the locked stop, token/slug/ball/basket at their sampled
                        poses — all by READBACK; score ~0, no success;
   2. randomization   — 3 seeded resets: max-pairwise deltas of token/slug/ball/basket
                        positions and the door crack are all nonzero;
   3. null-policy     — 240 idle steps: score < 0.05, no success;
   4. LOCK holds      — 8 N shove on the door for 1.2 s: the door moves (> 1 mm — the
                        tab is engaged, not fused) but stays under 15 mm; the rocker
                        stays locked (the shove torques it INTO its stop);
   5. slug can't pay  — the slug dropped into the coin tray: the rocker STAYS locked
                        (0.4x the holding torque), the same 8 N shove is still blocked,
                        no pay credit;
   6. token pays      — slug removed, the TOKEN dropped in: the rocker tips to the paid
                        stop (readback angle), and the SAME 8 N shove now slides the
                        door past open_thresh — the mechanism differential;
   7. no-pay delivery — fresh episode, ball teleport-constructed resting in the basket,
                        nothing paid (seed-analog "just get the thing out" breach):
                        no success, score < 0.5;
   8. slug + delivery — the slug parked in the tray on top of check 7's end state:
                        still locked, still no success;
   9. prefix stop     — paid + door opened but the ball left inside the chamber:
                        no success, score < 0.55;
  10. press cheat     — a probe torque presses the EMPTY lever to the paid stop: the
                        door physically opens (the gate is the angle), but paid_latch
                        stays 0 (payment means the TOKEN aboard), the lever falls back
                        locked once the door is shut and the press released, and a
                        delivered ball still isn't success;
  11. exactness       — full correct end state (token paid in tray, door opened by
                        shove, ball dropped into the basket): success() and
                        score == 1.0, stable 1 s hands-off; then REVOCATION: the token
                        teleported off the tray revokes success while latched credit
                        remains (0.5 <= score < 1);
  12. frames.npz      — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.open_washing_machine_i393.smoke --headless
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

# ----- module state wired up in main() -------------------------------------------------------------
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
    print(f"[smoke] {tag:18s} | door={float(scene.door_open()[0]) * 1000:6.1f}mm "
          f"rocker={math.degrees(float(scene.rocker_angle()[0])):+6.1f}deg "
          f"tok_tray={bool(scene.token_in_tray()[0])} paid={bool(scene.paid_now()[0])} "
          f"out={bool(scene.ball_out()[0])} in_basket={bool(scene.ball_in_basket()[0])} "
          f"latches=({float(scene.token_latch[0]):.0f},{float(scene.paid_latch[0]):.0f},"
          f"{float(scene.open_latch[0]):.0f},{float(scene.out_latch[0]):.0f}) "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.coinop_washer")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((0.85, -0.65, 0.85)) + o),
                                tuple(np.array((0.02, 0.02, 0.47)) + o),
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

    def epos(body) -> torch.Tensor:
        _refresh()
        return body.data.root_pos_w - scene.env_origins

    def rocker_deg() -> float:
        _refresh()
        return math.degrees(float(scene.rocker_angle()[0]))

    def drop_coin(coin) -> None:
        """TRANSPORT a coin to free air ~30 mm above the live tray pocket; gravity
        and contact do everything else."""
        _refresh()
        tray_local = torch.tensor([[0.0, c.tray_by, c.tray_floor_top]],
                                  device=device).repeat(n, 1)
        pos = scene.rocker.data.root_pos_w + quat_apply(scene.rocker.data.root_quat_w,
                                                        tray_local)
        pos = pos.clone()
        pos[:, 2] += 0.030
        _write_body(coin, pos)
        _step(30)
        for _ in range(16):
            _step(24)
            if float(scene.rocker.data.root_ang_vel_w.norm(dim=-1)[0]) < 0.3 \
                    and float(coin.data.root_lin_vel_w.norm(dim=-1)[0]) < c.slow_gate:
                break

    def shove(force: float = 8.0, steps: int = 144) -> float:
        """Constant +y push on the door through the wrench slot; returns the PEAK
        opening (sampled during the push — a parked door can't hide the peak, but
        symmetric with the memory rule anyway). Leaves the slot zeroed."""
        peak = 0.0
        scene.door_f = torch.full((n,), float(force), device=device)
        for _ in range(steps):
            _step(1)
            peak = max(peak, float(scene.door_open()[0]))
        scene.door_f = torch.zeros(n, device=device)
        _step(30)
        return peak

    def ball_to_basket() -> None:
        """CONSTRUCT: ball dropped from free air above the basket (readback)."""
        _refresh()
        pos = scene.basket.data.root_pos_w.clone()
        pos[:, 2] += c.basket_wall_t + c.basket_wall_h + c.ball_r + 0.020
        _write_body(scene.ball, pos)
        _step(90)

    # ================= 1. settle / no-NaN =========================================================
    env.reset(seed=100)
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    finite = all(bool(torch.isfinite(b.data.root_pos_w).all())
                 for b in scene._bodies().values())
    d_tok = float((epos(scene.token) - scene.token_start).norm(dim=-1)[0])
    d_slug = float((epos(scene.slug) - scene.slug_start).norm(dim=-1)[0])
    d_ball = float((epos(scene.ball) - scene.ball_start).norm(dim=-1)[0])
    d_bask = float((epos(scene.basket) - scene.basket_start).norm(dim=-1)[0])
    d_crack = abs(float(scene.door_open()[0]) - float(scene.crack0[0]))
    d_ang = abs(rocker_deg() - c.locked_deg)
    check("settle/no-NaN: reset settles finite; door AT its crack, rocker AT the locked "
          "stop, coins/ball/basket AT their sampled poses by readback; score ~0",
          finite and d_tok < 0.008 and d_slug < 0.008 and d_ball < 0.010
          and d_bask < 0.008 and d_crack < 0.003 and d_ang < 2.5
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ================= 2. randomization (3-seed max-pairwise) =====================================
    samples = []
    for sd in (3, 4, 5):
        env.reset(seed=sd)
        _step(4)
        samples.append({
            "tok": scene.token_start[0, :2].clone(), "slug": scene.slug_start[0, :2].clone(),
            "ball": scene.ball_start[0, :2].clone(), "bask": scene.basket_start[0, :2].clone(),
            "crack": float(scene.crack0[0]),
        })

    def maxpair(key: str) -> float:
        vals = [s[key] for s in samples]
        out = 0.0
        for i in range(3):
            for j in range(i + 1, 3):
                if key == "crack":
                    out = max(out, abs(vals[i] - vals[j]) * 1000)  # mm
                else:
                    out = max(out, float((vals[i] - vals[j]).norm()))
        return out

    print(f"[smoke] randomization max-pairwise: tok={maxpair('tok'):.3f} "
          f"slug={maxpair('slug'):.3f} ball={maxpair('ball'):.3f} "
          f"bask={maxpair('bask'):.3f} crack={maxpair('crack'):.2f}mm", flush=True)
    check("randomization: token/slug/ball/basket poses and the door crack all vary "
          "across 3 seeds (max-pairwise)",
          maxpair("tok") > 0.01 and maxpair("slug") > 0.01 and maxpair("ball") > 0.01
          and maxpair("bask") > 0.01 and maxpair("crack") > 0.1)

    # ================= 3. null policy =============================================================
    env.reset(seed=100)
    _step(240)
    _report("null")
    check("null policy: 2 s of nothing -> score < 0.05, no success",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ================= 4. the lock holds ==========================================================
    ang_before = rocker_deg()
    q0 = float(scene.door_open()[0])
    peak = shove(8.0, 144)
    _report("locked shove")
    check("lock holds: 8 N door shove moves the door (>1 mm, tab engaged) but is "
          "blocked (<15 mm); the rocker stays at its locked stop",
          peak - q0 > 0.001 and peak < 0.015 and rocker_deg() > c.locked_thresh_deg
          and abs(rocker_deg() - ang_before) < 3.0
          and float(scene.open_latch[0]) == 0.0 and not bool(scene.success()[0]))

    # ================= 5. the slug cannot pay =====================================================
    env.reset(seed=7)
    _step(60)
    _REC["on"] = True
    drop_coin(scene.slug)
    _report("slug in tray")
    slug_locked = rocker_deg() > c.locked_thresh_deg
    peak = shove(8.0, 144)
    _REC["on"] = False
    _report("slug shove")
    check("slug can't pay: slug dropped in the tray leaves the rocker locked, the "
          "8 N shove still blocked (<15 mm), no pay credit",
          slug_locked and peak < 0.015 and float(scene.paid_latch[0]) == 0.0
          and float(scene.token_latch[0]) == 0.0 and float(scene.score()[0]) < 0.05)

    # ================= 6. the token pays (mechanism differential) =================================
    # park the slug back on the table, then pay with the token
    slug_home = scene.slug_start + scene.env_origins
    _write_body(scene.slug, slug_home)
    _step(30)
    _REC["on"] = True
    drop_coin(scene.token)
    _report("token in tray")
    paid_ang = rocker_deg()
    peak = shove(8.0, 144)
    _REC["on"] = False
    _report("token shove")
    check("token pays: the rocker tips to the paid stop (readback angle) and the SAME "
          "8 N shove now opens the door past open_thresh",
          paid_ang < c.paid_thresh_deg + 1.0 and abs(paid_ang - c.paid_deg) < 3.0
          and peak >= c.open_thresh and float(scene.paid_latch[0]) == 1.0
          and float(scene.open_latch[0]) == 1.0)

    # ================= 7. no-pay delivery is not success ==========================================
    env.reset(seed=11)
    _step(60)
    ball_to_basket()
    _report("no-pay delivery")
    check("no-pay delivery: ball resting in the basket with nothing paid -> "
          "no success, score < 0.5",
          bool(scene.ball_in_basket()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) < 0.5)

    # ================= 8. slug + delivery is not success ==========================================
    drop_coin(scene.slug)
    _report("slug + delivery")
    check("slug + delivery: slug in the tray on top of a delivered ball -> still "
          "locked, still no success",
          rocker_deg() > c.locked_thresh_deg and not bool(scene.success()[0])
          and float(scene.score()[0]) < 0.5)

    # ================= 9. prefix stop: paid + open, ball left inside ==============================
    env.reset(seed=13)
    _step(60)
    drop_coin(scene.token)
    peak = shove(8.0, 144)
    _report("prefix stop")
    check("prefix stop: paid and door opened but the ball left inside the chamber -> "
          "no success, score < 0.55",
          peak >= c.open_thresh and not bool(scene.ball_out()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.55)

    # ================= 10. press cheat: hand-pressing the empty lever pays nothing ================
    env.reset(seed=17)
    _step(60)
    scene.rocker_tau = torch.full((n,), -0.08, device=device)  # ~5x the holding torque
    _step(120)
    pressed_ang = rocker_deg()
    pressed_latch = float(scene.paid_latch[0])
    peak = shove(8.0, 144)  # the door DOES open — the gate is the angle, not the token
    # return the door shut while the press still holds the tab clear (an open door
    # plate sits under the tab's return path), then release and watch the relock
    for _ in range(400):
        q = scene.door_open()
        v = scene.door.data.root_lin_vel_w[:, 1]
        v_des = torch.clamp(6.0 * (0.0 - q), -0.10, 0.10)
        scene.door_f = torch.clamp(12.0 * (v_des - v), -8.0, 8.0)
        _step(1)
        if float(q[0]) < 0.004 and abs(float(v[0])) < 0.03:
            break
    scene.door_f = torch.zeros(n, device=device)
    _step(30)
    scene.rocker_tau = torch.zeros(n, device=device)
    _step(120)
    released_ang = rocker_deg()
    ball_to_basket()
    _report("press cheat")
    check("press cheat: pressing the EMPTY lever reaches the paid stop and opens the "
          "door, but paid_latch stays 0, the lever falls back locked once the door "
          "is shut and the press released, and the delivered ball is not success",
          pressed_ang < c.paid_thresh_deg + 1.0 and pressed_latch == 0.0
          and peak >= c.open_thresh and released_ang > c.locked_thresh_deg
          and float(scene.paid_latch[0]) == 0.0 and not bool(scene.success()[0])
          and float(scene.score()[0]) < 0.7)

    # ================= 11. exactness + revocation =================================================
    env.reset(seed=21)
    _step(60)
    _REC["on"] = True
    drop_coin(scene.token)
    shove(8.0, 144)
    ball_to_basket()
    _step(60)
    _report("exact end state")
    ok_succ = bool(scene.success()[0]) and float(scene.score()[0]) > 0.999
    _step(120)  # 1 s hands-off
    ok_hold = bool(scene.success()[0]) and float(scene.score()[0]) > 0.999
    # revocation: token teleported off the tray
    tok_out = scene.env_origins + torch.tensor([[0.30, 0.20, c.table_top_z + 0.02]],
                                               device=device)
    _write_body(scene.token, tok_out)
    _step(90)
    _REC["on"] = False
    _report("revoked")
    rev_score = float(scene.score()[0])
    check("exactness: full correct end state -> success and score == 1.0, stable 1 s; "
          "removing the token revokes success while latched credit remains",
          ok_succ and ok_hold and not bool(scene.success()[0])
          and 0.5 <= rev_score < 0.999)

    # ================= 12. frames =================================================================
    frames = _REC["frames"]
    if frames:
        np.savez_compressed(args.out, frames=np.stack(frames))
        print(f"[smoke] saved {len(frames)} frames -> {args.out}", flush=True)
    check("frames: >= 20 rgb frames recorded and saved", len(frames) >= 20)

    # ================= verdict ====================================================================
    n_pass = sum(1 for _nm, ok in checks if ok)
    n_all = len(checks)
    for nm, ok in checks:
        if not ok:
            print(f"[smoke] FAILED CHECK: {nm}", flush=True)
    if n_pass == n_all:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{n_all}", flush=True)
        code = 0
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{n_all}", flush=True)
        code = 1

    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()

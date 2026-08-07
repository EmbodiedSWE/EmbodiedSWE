"""Smoke / rubric-REJECTION battery for BowlDecantScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct pour-then-park and the
latched credit is monotone along a real trajectory). This battery proves the rubric
REJECTS wrong outcomes — above all the SEED task's own end state (bowl set on the
dish) — and that the geometric claims the task rests on (the dish rim retains a
settled ball; containment-exclusion; the order encoding) are physically load-bearing.
Every probe is CONSTRUCTED as a settled state (teleport, real physics steps, judge) —
instrumentation, never a solution: no probe here reaches success().

Checks:
   1. settle/no-NaN      — seeded reset settles finite; every present ball inside the
                           bowl, none in the dish; score 0, no success;
   2. randomization      — two seeded resets: READBACK dish xy, tray xy/yaw, bowl xy
                           all differ, and the dish/tray SIDE swaps across seeds;
   3. ball-count subset  — over 10 resets the present-ball count takes >= 2 distinct
                           values in {2, 3, 4} (count what you see);
   4. null-policy        — 240 idle steps: score ~0, no success (nothing moves);
   5. RIM INTERLOCK      — a ball settled in the dish is shoved outward at 3x its own
                           weight for 1.5 s: the 30 mm rim holds it in ("balls cannot
                           be pushed back out" is physics, not fiat);
   6. SEED STRATEGY      — the seed's plan ("put the white bowl on the plate"): the
                           LOADED bowl set upright, centered ON the dish, contents
                           still inside -> NO success, score ~0 (containment-exclusion
                           rejects every "in the dish" claim);
   7. half-done order    — balls correctly poured into the dish but the bowl left
                           standing UPRIGHT in the tray (not inverted) -> no success;
   8. near-miss pour     — all but one ball in the dish, the last settled on the
                           ground just OUTSIDE the rim, bowl parked -> no success;
   9. near-miss park     — balls in the dish, bowl inverted on the ground BESIDE the
                           tray (centimetres from its goal) -> no success;
  10. park-before-pour   — the bowl inverted in the tray FIRST, balls dumped on the
                           open ground: `park` latch must NOT set (order encoding),
                           no success, score ~0;
  11. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_kitchen_scene7_put_the_white_bowl_on_the_plate_i2.smoke --headless
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

_qy = task_scene._qy

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
    pres = scene.present[0]
    inb, ind = scene.in_bowl()[0], scene.in_dish()[0]
    marks = "".join(
        ("P" if bool(pres[i]) else "-") + ("B" if bool(inb[i]) else "")
        + ("D" if bool(ind[i]) else "") + " " for i in range(4))
    print(f"[smoke] {tag:18s} | balls {marks}| lift={bool(scene._lift[0])} "
          f"m1={bool(scene._m1[0])} mall={bool(scene._mall[0])} "
          f"park={bool(scene._park[0])} parked_now={bool(scene.bowl_parked()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


def _push(body, force_w: torch.Tensor, steps: int) -> None:
    """Apply a world-frame force at the body's CoM for `steps` steps, then clear."""
    n = _ENV.num_envs
    zero = torch.zeros(n, 1, 3, device=_ENV.device)
    f = force_w.view(1, 1, 3).expand(n, 1, 3).contiguous()
    for _ in range(steps):
        body.set_external_force_and_torque(f, zero, env_ids=_all_ids(), is_global=True)
        _step(1)
    body.set_external_force_and_torque(zero, zero, env_ids=_all_ids())


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.bowl_decant")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.05, -0.75, 0.70)) + o),
                                tuple(np.array((0.30, 0.00, 0.05)) + o),
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

    def dish_c() -> torch.Tensor:
        _refresh()
        return scene.dish.data.root_pos_w.clone()

    def tray_c() -> torch.Tensor:
        _refresh()
        return scene.tray.data.root_pos_w.clone()

    def seat_balls_in_dish(idx: list[int], center_w: torch.Tensor) -> None:
        """CONSTRUCT balls `idx` resting on the dish basin floor around `center_w`."""
        for j, i in enumerate(idx):
            a = j * 2 * math.pi / max(len(idx), 1)
            p = center_w.clone()
            p[:, 0] += 0.030 * math.cos(a)
            p[:, 1] += 0.030 * math.sin(a)
            p[:, 2] += c.dish_base_t + c.ball_r + 0.002
            _write_body(scene.balls[scene.BALL_NAMES[i]], p, None)

    def park_bowl_inverted(center_w: torch.Tensor, dz: float) -> None:
        """CONSTRUCT the bowl upside-down: origin at center + dz (rim faces down)."""
        p = center_w.clone()
        p[:, 2] += dz
        q = _qy(torch.full((n,), math.pi, device=device))
        _write_body(scene.bowl, p, q)

    def present_idx() -> list[int]:
        return [i for i in range(4) if bool(scene.present[0, i])]

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(150)
    _report("settle")
    _REC["on"] = False
    pos, _v = scene._ball_tensors()
    pres = scene.present[0]
    check("settle/no-NaN: layout settles finite; every present ball inside the bowl, "
          "none in the dish; score 0, no success",
          bool(torch.isfinite(pos).all())
          and bool((scene.in_bowl()[0] | ~pres).all())
          and int((scene.in_dish()[0] & pres).sum()) == 0
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (scene.dish.data.root_pos_w[0, :2].clone(),
                scene.tray.data.root_pos_w[0, :2].clone(),
                scene.bowl.data.root_pos_w[0, :2].clone(),
                int(scene.dish_side[0]))

    sides = set()
    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_d, a_t, a_b, a_s = readback()
    sides.add(a_s)
    b_d = b_t = b_b = None
    for s in range(8):  # find a seed whose side differs, proving side_swap is real
        torch.manual_seed(202 + s)
        env.reset()
        _step(10)
        b_d, b_t, b_b, b_s = readback()
        sides.add(b_s)
        if b_s != a_s:
            break
    d_d, d_t, d_b = (float((a_d - b_d).norm()), float((a_t - b_t).norm()),
                     float((a_b - b_b).norm()))
    print(f"[smoke] randomization deltas: dish_xy={d_d * 1000:.1f}mm "
          f"tray_xy={d_t * 1000:.1f}mm bowl_xy={d_b * 1000:.1f}mm "
          f"sides_seen={sorted(sides)}", flush=True)
    check("randomization-is-real: dish xy, tray xy, bowl xy readback differ and the "
          "dish/tray SIDE swaps across seeds",
          d_d > 0.003 and d_t > 0.003 and d_b > 0.003 and sides == {-1, 1})

    # ================= 3. ball-count subset =======================================================
    counts = set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        counts.add(int(scene.present[0].sum()))
    print(f"[smoke] over 10 resets: present-ball counts {sorted(counts)}", flush=True)
    check("ball-count subset: present count takes >= 2 distinct values in {2,3,4} "
          "over 10 resets",
          len(counts) >= 2 and counts <= {2, 3, 4})

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. RIM INTERLOCK: a settled ball cannot be pushed out ======================
    # "the dish's raised rim keeps balls from rolling out": seat one ball in the
    # basin, then shove it radially outward at 3x its weight for 1.5 s — the 30 mm
    # rim vs the 24 mm ball must hold it. The physics behind "a ball in the dish
    # stays in the dish".
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    dc = dish_c()
    seat_balls_in_dish([0], dc)
    _step(60)
    _refresh()
    assert bool(scene.in_dish()[0, 0]), "probe setup: ball_0 must seat in the dish"
    f = torch.tensor([3.0 * c.ball_mass * 9.81, 0.0, 0.0], device=device)
    r_max = -1.0
    for _ in range(9):  # 9 x 20 = 180 steps = 1.5 s of shoving (+x, radially out)
        _push(scene.balls["ball_0"], f, 20)
        _refresh()
        r_max = max(r_max, float(
            scene._local(scene.dish, scene.balls["ball_0"].data.root_pos_w)[0, :2].norm()))
    _step(60)
    _report("rim-interlock")
    print(f"[smoke] rim shove: max dish-frame r={r_max * 1000:.1f}mm "
          f"(rim inner face {c.dish_inner_r * 1000:.0f}mm, rim {c.dish_wall_h * 1000:.0f}mm "
          f"tall vs ball {2 * c.ball_r * 1000:.0f}mm)", flush=True)
    check("RIM INTERLOCK: 3x-weight outward shove for 1.5 s never gets the settled "
          "ball out of the dish",
          bool(scene.in_dish()[0, 0]) and r_max < c.dish_inner_r
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 6. negative: the SEED'S OWN STRATEGY =======================================
    # "Put the white bowl on the plate" — CONSTRUCT the seed's end state: the LOADED
    # bowl set upright, centered on the dish, contents still inside. In the seed's
    # rubric this is a full success; here the containment-exclusion clause voids
    # every "in the dish" claim and the bowl is neither inverted nor in the tray.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    dc = dish_c()
    pidx = present_idx()
    p = dc.clone()
    p[:, 2] += c.dish_base_t + 0.002  # bowl floor resting on the basin floor
    _write_body(scene.bowl, p, None)  # upright, contents to follow
    for j, i in enumerate(pidx):      # balls INSIDE the bowl (ride along)
        a = j * 2 * math.pi / max(len(pidx), 1)
        bp = p.clone()
        bp[:, 0] += c.ball_ring_r * math.cos(a)
        bp[:, 1] += c.ball_ring_r * math.sin(a)
        bp[:, 2] += c.bowl_floor_t + c.ball_r + 0.002
        _write_body(scene.balls[scene.BALL_NAMES[i]], bp, None)
    _step(150)
    _report("seed-strategy")
    ind = scene.in_dish()[0]
    check("negative (SEED strategy): loaded bowl set upright ON the dish, balls "
          "inside — containment-exclusion voids 'in the dish', NO success, score ~0",
          int((ind & scene.present[0]).sum()) == 0
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.05)
    _REC["on"] = False

    # ================= 7. negative: poured, but the bowl parked UPRIGHT ===========================
    # Balls correctly in the dish; the bowl standing right-side-up in the tray.
    # Inversion is part of the goal — must NOT be success (and _park does latch
    # only for inverted parks, so score caps at lift+first+all).
    torch.manual_seed(100)
    env.reset()
    _step(30)
    seat_balls_in_dish(present_idx(), dish_c())
    p = tray_c()
    p[:, 2] += c.tray_base_t + 0.002  # upright: floor on the pad
    _write_body(scene.bowl, p, None)
    _step(150)
    _report("upright-park")
    check("negative (wrong orientation): balls in the dish but the bowl parked "
          "UPRIGHT in the tray — not inverted, no success, score <= 0.75",
          bool((scene.in_dish()[0] | ~scene.present[0]).all())
          and not bool(scene.bowl_parked()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.75)

    # ================= 8. near-miss: one ball just OUTSIDE the dish rim ===========================
    # Bowl perfectly parked, all balls in the dish EXCEPT one settled on the ground
    # against the OUTSIDE of the rim (centimetres from its goal). "All present
    # balls" must refuse.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    pidx = present_idx()
    dc = dish_c()
    seat_balls_in_dish(pidx[:-1], dc)
    stray = pidx[-1]
    p = dc.clone()
    p[:, 0] += c.dish_base_r + c.ball_r + 0.004  # on the ground, hugging the base
    p[:, 2] = _ENV.iscene.env_origins[:, 2] + c.ball_r + 0.002
    _write_body(scene.balls[scene.BALL_NAMES[stray]], p, None)
    park_bowl_inverted(tray_c(), c.tray_base_t + c.bowl_h + 0.004)
    _step(150)
    _report("near-miss-pour")
    check("near-miss (pour): bowl parked, all balls in the dish except one settled "
          "on the ground just OUTSIDE the rim — no success",
          not bool(scene.in_dish()[0, stray]) and bool(scene.bowl_parked()[0])
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 9. near-miss: bowl inverted BESIDE the tray ================================
    # All balls in the dish; the bowl inverted and at rest on the open ground just
    # outside the tray. The tray gate must refuse.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    seat_balls_in_dish(present_idx(), dish_c())
    p = tray_c()
    p[:, 0] += c.tray_side / 2 + c.bowl_outer_r + 0.020  # ground beside the tray
    p[:, 2] = _ENV.iscene.env_origins[:, 2] + c.bowl_h + 0.004
    q = _qy(torch.full((n,), math.pi, device=device))
    _write_body(scene.bowl, p, q)
    _step(150)
    _report("near-miss-park")
    check("near-miss (park): balls in the dish, bowl inverted on the ground BESIDE "
          "the tray — outside the tray gate, no success",
          bool((scene.in_dish()[0] | ~scene.present[0]).all())
          and not bool(scene.bowl_parked()[0]) and not bool(scene.success()[0]))

    # ================= 10. negative: park BEFORE pour =============================================
    # The order encoding: the bowl inverted in the tray FIRST, the balls dumped on
    # the open ground (where an early inversion would leave them). The park latch
    # requires `_mall` already set -> must stay False; score ~0, no success.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    park_bowl_inverted(tray_c(), c.tray_base_t + c.bowl_h + 0.004)
    for j, i in enumerate(present_idx()):  # balls scattered on the open ground
        p = tray_c()
        p[:, 0] -= 0.16
        p[:, 1] += 0.05 * j - 0.08
        p[:, 2] = _ENV.iscene.env_origins[:, 2] + c.ball_r + 0.002
        _write_body(scene.balls[scene.BALL_NAMES[i]], p, None)
    _step(150)
    _report("park-first")
    check("negative (park-before-pour): bowl inverted in the tray, balls dumped on "
          "the ground — park latch stays unset (order encoding), no success, "
          "score ~0",
          bool(scene.bowl_parked()[0]) and not bool(scene._park[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.05)
    _REC["on"] = False

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.bowl_decant")
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

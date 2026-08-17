"""Smoke / rubric-REJECTION battery for MokaBalanceScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a real weighing and the latched
credit is monotone along it). This battery proves the rubric REJECTS wrong outcomes
and that the claims the task rests on — the balance mechanism as the measuring
instrument, the only-weights-in-blue / only-pot-in-red placement clauses, and the
cached-mass anti-pinning backstop — are load-bearing. Every probe is CONSTRUCTED as
a settled state (teleport transport, real physics steps, judge) — instrumentation,
never a solution: no probe here reaches success().

Checks:
  1.  settle/no-NaN    — seeded reset settles finite; pot stands on the stove (the
                         SEED task's goal state scores ZERO here), beam level, no
                         success;
  2.  randomization    — two seeded resets: READBACK stove xy, pot yaw and the
                         weight-grid permutation all differ;
  3.  pot-mass spread  — over 10 resets the pot's ENGINE-READBACK mass takes >= 3
                         distinct values, every one a legal choice;
  4.  null-policy      — 240 idle steps: score ~0, no success;
  5.  deck set-down    — pot moved off the stove and settled on the bare deck (the
                         seed's own move class): no latch, score ~0;
  6.  pot-in-red only  — pot alone in the red pan: the pot latch fires (0.2) and
                         the unmatched beam SLAMS to its stop — no success;
  7.  mirrored pans    — pot in the BLUE pan + the exact counterweights in the RED
                         pan: the beam genuinely balances LEVEL, yet every latch
                         stays cold and success is refused — score ~0 (the colored
                         placement clauses are load-bearing even when the physics
                         balances);
  8.  near-miss 50 g   — pot in red + a combo 50 g SHORT in blue: pot+weight
                         latches fire (0.4) but the beam holds at its stop, no
                         balanced latch, no success;
  9.  frypan cheat     — INSTRUMENTATION: the frypan's engine mass is set equal to
                         the pot's, then used as the counterweight. The beam
                         genuinely balances level — but the frypan is not a labeled
                         weight: no weight/balanced latch, no success, score <= 0.205;
  10. shared red pan   — pot + a 50 g weight in RED vs an exact-sum combo in BLUE:
                         genuinely level, all three latches fire (cap 0.60) but
                         misplacement refuses success;
  11. pinned beam      — a wrong (50 g short) load, then an external PD torque
                         wrench pins the free beam LEVEL and CALM (hold verified —
                         non-vacuous): every geometric clause passes but the
                         cached-mass clause refuses — no balanced latch, no
                         success, score <= 0.405;
  12. frames.npz       — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_kitchen_scene3_put_the_moka_pot_on_the_stove_i141.smoke --headless
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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() ----------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}

# oracle sum -> weights-list indices (masses: 50,50,100,100,200,200 g)
_COMBOS = {100: [2], 150: [2, 0], 200: [4], 250: [4, 0], 300: [4, 2],
           350: [4, 2, 0], 400: [4, 5]}
# same, but avoiding index 0 (w50a — used inside the red pan in check 10)
_COMBOS_NO_W50A = {200: [4], 250: [4, 1], 300: [4, 2], 350: [4, 2, 1], 400: [4, 5]}
_SPOTS = {1: [(0.0, 0.0)],
          2: [(-0.032, 0.0), (0.032, 0.0)],
          3: [(-0.032, 0.016), (0.032, 0.016), (0.0, -0.032)]}


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


def _hover(basket, xy, dz: float = 0.008) -> torch.Tensor:
    """World hover point `dz` above the pan platform at pan-local xy (LIVE pose)."""
    from isaaclab.utils.math import quat_apply

    c = _ENV.scene.cfg
    n = _ENV.num_envs
    lp = torch.tensor([xy[0], xy[1], -c.platform_drop + dz],
                      device=_ENV.device).expand(n, 3)
    return basket.data.root_pos_w + quat_apply(basket.data.root_quat_w, lp)


def _drop(body, basket, xy, payload_idx: int, settle: int = 420) -> bool:
    """Transport `body` to the in-pan hover, release, settle; retry a bounce."""
    scene = _ENV.scene
    for _ in range(3):
        _write_body(body, _hover(basket, xy))
        _step(settle)
        if bool(scene.in_basket(basket)[0, payload_idx]):
            return True
    return False


def _wait_level(max_steps: int = 3600, need: int = 240) -> bool:
    """Free physics until level+calm+settled holds for `need` consecutive steps."""
    scene = _ENV.scene
    streak = 0
    for _ in range(max_steps):
        _step(1)
        ok = bool(scene.level()[0]) and bool(scene.beam_calm()[0]) \
            and bool(scene.settled()[0])
        streak = streak + 1 if ok else 0
        if streak >= need:
            return True
    return False


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    in_r = scene.in_basket(scene.basket_r)[0]
    in_b = scene.in_basket(scene.basket_b)[0]
    print(f"[smoke] {tag:16s} | tilt={math.degrees(float(scene.beam_tilt()[0])):+6.2f}deg "
          f"pot_red={bool(in_r[0])} wts_blue={int(in_b[1:7].sum())} "
          f"imb={float(scene.imbalance()[0]) * 1000:+.1f}g "
          f"level={bool(scene.level()[0])} settled={bool(scene.settled()[0])} "
          f"latches=({int(scene._l_pot[0])},{int(scene._l_wt[0])},{int(scene._l_lvl[0])}) "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main -------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.moka_balance")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.25, -0.95, 0.85)) + o),
                                tuple(np.array((0.32, 0.00, 0.22)) + o),
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

    def pot_grams() -> int:
        return round(float(scene.masses_cache[0, 0]) * 1000)

    oxy = env.iscene.env_origins[:, :2]
    oz = float(env.iscene.env_origins[0, 2])

    def deck_spot(x: float, y: float) -> torch.Tensor:
        pos = torch.zeros(n, 3, device=device)
        pos[:, 0], pos[:, 1], pos[:, 2] = x, y, oz + c.deck_top + 0.002
        pos[:, :2] += oxy
        return pos

    # ================= 1. settle / no-NaN (start = the SEED task's goal) ========================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(120)
    _report("settle")
    _REC["on"] = False
    fin = bool(torch.isfinite(scene.pot.data.root_pos_w).all()
               and torch.isfinite(scene.beam.data.root_pos_w).all()
               and torch.isfinite(scene.pot.data.root_lin_vel_w).all())
    check("settle/no-NaN: pot standing on the stove (the seed's GOAL state), beam "
          "level and empty, score 0, no success",
          fin and bool(scene.pot_on_stove()[0]) and bool(scene.level()[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ======================================
    def readback():
        _refresh()
        wxy = torch.stack([w.data.root_pos_w[0, :2] - oxy[0] for w in scene.weights])
        return (scene.stove.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.pot.data.root_quat_w[0]), wxy)

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_s, a_y, a_w = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_s, b_y, b_w = readback()
    d_s = float((a_s - b_s).norm())
    d_y = dyaw(a_y, b_y)
    d_w = float((a_w - b_w).norm(dim=-1).sum())
    print(f"[smoke] randomization deltas: stove_xy={d_s * 1000:.1f}mm "
          f"pot_yaw={d_y:.1f}deg weight_layout_sum={d_w * 1000:.1f}mm", flush=True)
    check("randomization-is-real: stove xy, pot yaw and the weight-grid "
          "permutation readback differ",
          d_s > 0.003 and d_y > 5.0 and d_w > 0.05)

    # ================= 3. pot mass spread + in-choices (ENGINE readback) ========================
    grams = []
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        grams.append(pot_grams())
    legal = {round(m * 1000) for m in c.pot_mass_choices}
    print(f"[smoke] pot masses over 10 resets: {grams}", flush=True)
    check("pot-mass spread: >= 3 distinct engine-readback masses over 10 resets, "
          "all legal choices",
          len(set(grams)) >= 3 and all(g in legal for g in grams))

    # ================= 4. null policy fails =====================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. deck set-down (moved, but not onto the balance) =======================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _write_body(scene.pot, deck_spot(0.65, 0.02))
    _step(240)
    _report("deck-set-down")
    check("negative (set-down): pot settled on the bare deck, off the stove — "
          "no latch, no success, score <= 0.005",
          not bool(scene.pot_on_stove()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.005)

    # ================= 6. pot in the red pan only ===============================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    ok_drop = _drop(scene.pot, scene.basket_r, (0.0, 0.0), 0, settle=600)
    _report("pot-in-red")
    _REC["on"] = False
    tilt = math.degrees(float(scene.beam_tilt()[0]))
    check("partial (pot in red): pot latch fires, the unmatched beam slams to its "
          "stop — no success, score <= 0.205",
          ok_drop and bool(scene._l_pot[0]) and tilt < -(c.beam_limit_deg - 3.0)
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.205)

    # ================= 7. mirrored pans: level, but zero =========================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    combo = _COMBOS[pot_grams()]
    ok_drop = _drop(scene.pot, scene.basket_b, (0.0, 0.0), 0, settle=600)
    for widx, xy in zip(combo, _SPOTS[len(combo)]):
        ok_drop &= _drop(scene.weights[widx], scene.basket_r, xy, 1 + widx, settle=360)
    _REC["on"] = False  # don't record the (long) settle wait
    lvl = _wait_level()
    _report("mirrored-pans")
    check("negative (mirrored pans): pot in BLUE + exact weights in RED — the "
          "beam genuinely balances level, yet no latch fires and success is "
          "refused: score <= 0.005",
          ok_drop and lvl and bool(scene.level()[0])
          and not bool(scene._l_pot[0]) and not bool(scene._l_wt[0])
          and not bool(scene._l_lvl[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.005)

    # ================= 8. near-miss: combo 50 g short ===========================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    pg = pot_grams()
    wrong = _COMBOS[pg - 50 if pg >= 200 else pg + 50]
    ok_drop = _drop(scene.pot, scene.basket_r, (0.0, 0.0), 0, settle=600)
    for widx, xy in zip(wrong, _SPOTS[len(wrong)]):
        ok_drop &= _drop(scene.weights[widx], scene.basket_b, xy, 1 + widx, settle=360)
    _step(600)
    _report("near-miss-50g")
    tilt = math.degrees(float(scene.beam_tilt()[0]))
    check("near-miss (50 g off): pot+weight latches fire but the beam holds at "
          "its stop — no balanced latch, no success, score <= 0.405",
          ok_drop and bool(scene._l_pot[0]) and bool(scene._l_wt[0])
          and abs(tilt) > c.beam_limit_deg - 3.0 and not bool(scene._l_lvl[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.405)

    # ================= 9. frypan-as-counterweight cheat =========================================
    # INSTRUMENTATION: make the cheat PHYSICALLY perfect — set the frypan's engine
    # mass equal to the pot's and re-sync the cache the mass clause reads. The
    # beam then balances level for real; only the placement clause refuses.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    pot_mass = scene.masses_cache[:, 0].clone()
    scene._set_body_mass(scene.frypan, _all_ids(), pot_mass)
    scene.refresh_cached_masses()
    _REC["on"] = True
    ok_drop = _drop(scene.pot, scene.basket_r, (0.0, 0.0), 0, settle=600)
    ok_drop &= _drop(scene.frypan, scene.basket_b, (0.0, 0.0), 7, settle=600)
    _REC["on"] = False  # don't record the (long) settle wait
    lvl = _wait_level()
    _report("frypan-cheat")
    check("negative (frypan cheat): frypan mass set EQUAL to the pot's and used "
          "as counterweight — beam genuinely level, but it is not a labeled "
          "weight: no weight/balanced latch, no success, score <= 0.205",
          ok_drop and lvl and bool(scene.level()[0]) and bool(scene._l_pot[0])
          and not bool(scene._l_wt[0]) and not bool(scene._l_lvl[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.205)
    scene._set_body_mass(scene.frypan, _all_ids(),
                         torch.full((n,), c.frypan_mass, device=device))
    scene.refresh_cached_masses()

    # ================= 10. weights sharing the red pan ==========================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    pg = pot_grams()
    blue = _COMBOS_NO_W50A[pg + 50]
    ok_drop = _drop(scene.pot, scene.basket_r, (-0.018, 0.0), 0, settle=600)
    ok_drop &= _drop(scene.weights[0], scene.basket_r, (0.047, 0.0), 1, settle=360)
    for widx, xy in zip(blue, _SPOTS[len(blue)]):
        ok_drop &= _drop(scene.weights[widx], scene.basket_b, xy, 1 + widx, settle=360)
    lvl = _wait_level()
    _report("shared-red-pan")
    check("negative (shared red pan): pot + 50 g in RED vs exact sum in BLUE — "
          "genuinely level, latches cap at 0.60, but misplacement refuses "
          "success",
          ok_drop and lvl and bool(scene.level()[0]) and bool(scene._l_lvl[0])
          and bool(scene.misplaced()[0]) and not bool(scene.success()[0])
          # 0.605: the 0.60 cap is stored in float32 (0.60000002...), so an
          # exact <= 0.60 comparison in float64 would fail on the cap itself.
          and float(scene.score()[0]) <= 0.605)

    # ================= 11. pinned beam: the mass clause is load-bearing =========================
    # Wrong (50 g short) load, then an external torque wrench holds the beam
    # level and calm. Every geometric clause then passes; only the cached-mass
    # clause stands between a pinned beam and success.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    pg = pot_grams()
    wrong = _COMBOS[pg - 50 if pg >= 200 else pg + 50]
    _REC["on"] = True
    ok_drop = _drop(scene.pot, scene.basket_r, (0.0, 0.0), 0, settle=600)
    for widx, xy in zip(wrong, _SPOTS[len(wrong)]):
        ok_drop &= _drop(scene.weights[widx], scene.basket_b, xy, 1 + widx, settle=360)
    _step(300)
    _REC["on"] = False  # the pin loop steps the env directly (no recording)
    tilt0 = math.degrees(float(scene.beam_tilt()[0]))
    g_acc, kp, kd = 9.81, 2.0, 0.35
    zero3 = torch.zeros(n, 1, 3, device=device)
    no_action = torch.empty(0, device=device)
    for _ in range(720):
        th = scene.beam_tilt()
        om = scene.beam.data.root_ang_vel_w[:, 0]
        # feedforward: cancel the payload imbalance torque; PD: regulate level.
        # Torque is about the beam's own hinge axis (x), invariant under the
        # beam's rotation — no wrench frame drag to compensate.
        tau = (scene.imbalance() * g_acc * c.hang_y * torch.cos(th)
               - kp * th - kd * om).clamp(-0.6, 0.6)
        t3 = zero3.clone()
        t3[:, 0, 0] = tau
        scene.beam.set_external_force_and_torque(zero3, t3)
        env.step(no_action)
    _refresh()
    tilt1 = math.degrees(float(scene.beam_tilt()[0]))
    held = abs(tilt1) <= c.level_max_deg and bool(scene.beam_calm()[0]) \
        and bool(scene.settled()[0])
    print(f"[smoke] pin: tilt {tilt0:+.2f} -> {tilt1:+.2f}deg "
          f"(hold achieved={held})", flush=True)
    _report("pinned-beam")
    _REC["on"] = False
    check("anti-pinning: wrong weights + external torque holds the beam LEVEL "
          "and CALM (verified) — cached-mass clause refuses: no balanced latch, "
          "no success, score <= 0.405",
          ok_drop and abs(tilt0) > c.beam_limit_deg - 3.0 and held
          and not bool(scene._l_lvl[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.405)
    scene.beam.set_external_force_and_torque(zero3, zero3)  # drop the wrench

    # ================= save + verdict ===========================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.moka_balance")
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

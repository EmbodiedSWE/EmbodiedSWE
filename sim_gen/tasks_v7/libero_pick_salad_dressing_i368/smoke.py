"""Smoke / rubric-REJECTION battery for DressingFerryDepotScene — NullRobot, probes.

solve.py is the acceptance proof (extract -> load -> ferry earns monotone latched
credit and success). This battery proves the rubric REJECTS wrong outcomes and that
the physical claim the task rests on — the ROOF-BLOCK interlock that forbids loading
while the cart is docked — is load-bearing. Probes are constructed settled states /
instrumented force pushes, never a solution.

Checks:
   1. settle/no-NaN     — seeded reset settles finite; cart docked in the garage,
                          both bottles upright at their floor stations (swap
                          respected), score 0, no success;
   2. randomization     — 3 seeded resets: max-pairwise depot xy / depot yaw /
                          target-bottle xy readback deltas all real (GPU 2-seed
                          collision guard);
   3. swap coverage     — across 10 resets both station assignments are drawn and
                          the depot yaw readback shows a real spread;
   4. null-policy       — 240 idle steps: score ~0, no success;
   5. SEED STRATEGY     — the seed's move ("carry the bottle to the receptacle and
                          put it down") aimed at the docked cart: the bottle stands
                          settled ON the cart slab, under the roof, next to the well
                          -> not seated, no success, score 0;
   6. roof-block probe  — PHYSICAL: cart docked, the bottle dropped onto the garage
                          roof over the well and pressed DOWN with a real force: it
                          rests ON the roof (anti-vacuity readback: it fell and the
                          press held it there), never reaches the well, no success;
   7. undelivered load  — bottle genuinely seated in the well but the cart still
                          parked OUT at the extraction line: honest partial credit
                          (0.50), no success;
   8. dock near-miss    — seated bottle, cart 30 mm short of the dock line: docked
                          False, score strictly below the pre-success cap, no
                          success;
   9. wrong object      — the RED decoy seated in the well with the cart docked:
                          success False, score ~0 (every credit stage is
                          target-specific);
  10. latch persistence — PHYSICAL: the cart is pulled out with the real drive
                          channel (extraction latch fires, 0.15), then pushed back
                          empty to the dock: credit persists at 0.15, no success;
  11. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_pick_salad_dressing_i368.smoke --headless
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


def _settle_until(pred, max_steps: int = 300, poll: int = 15) -> bool:
    _step(poll)  # ALWAYS force real steps first (zero-step teleport trap)
    if pred():
        return True
    waited = poll
    while waited < max_steps:
        _step(poll)
        waited += poll
        if pred():
            return True
    return False


def _depot_yaw() -> float:
    q = _ENV.scene.depot.data.root_quat_w[0]
    return 2.0 * math.atan2(float(q[3]), float(q[0]))


def _write_local(body, lx: float, ly: float, lz: float, lyaw: float = 0.0) -> None:
    """Teleport `body` (env 0 broadcast) to a depot-local pose, zero velocity."""
    env = _ENV
    scene = env.scene
    dp = scene.depot.data.root_pos_w[0]
    yaw = _depot_yaw()
    cy, sy = math.cos(yaw), math.sin(yaw)
    st = torch.zeros(env.num_envs, 13, device=env.device)
    st[:, 0] = float(dp[0]) + cy * lx - sy * ly
    st[:, 1] = float(dp[1]) + sy * lx + cy * ly
    st[:, 2] = float(scene.env_origins[0, 2]) + lz
    st[:, 3] = math.cos((yaw + lyaw) / 2)
    st[:, 6] = math.sin((yaw + lyaw) / 2)
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _place_cart_loaded(cart_x: float, seat_target: bool) -> None:
    """Teleport the cart to depot-local x AND the chosen bottle seated in its well
    in ONE write pass (whole-linkage teleport: writing only one body would get
    depenetrated by the other)."""
    scene = _ENV.scene
    c = scene.cfg
    z_cart = c.deck_top + c.slab_t / 2 + 0.002
    _write_local(scene.cart, cart_x, 0.0, z_cart, 0.0)
    body = scene.target if seat_target else scene.decoy
    _write_local(body, cart_x + c.well_x, 0.0, z_cart + c.slab_t / 2 + c.bot_off + 0.002, 0.0)


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    p = scene._cart_local()[0]
    tb = scene._to_depot(scene.target.data.root_pos_w)[0]
    print(f"[smoke] {tag:16s} | cart=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):.3f}) "
          f"bottle=({float(tb[0]):+.3f},{float(tb[1]):+.3f},{float(tb[2]):.3f}) "
          f"extr={bool(scene._extracted[0])} load={bool(scene._loaded[0])} "
          f"carry={float(scene._carry[0]):.2f} "
          f"seat={bool(scene.seated_now(scene.target)[0])} "
          f"dock={bool(scene.docked_now()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


def _push_cart(sign: float, pred, *, force: float, vmax: float, iters: int) -> bool:
    """Velocity-regulated CoM push on the cart along the depot x-axis (the solve's
    drive channel), depot-local y P-steered to 0. Returns pred() at exit."""
    scene = _ENV.scene
    buf = scene.cart_force
    ok = False
    for _ in range(iters):
        _refresh()
        if pred():
            ok = True
            break
        yaw = _depot_yaw()
        cy, sy = math.cos(yaw), math.sin(yaw)
        p = scene._cart_local()[0]
        d = [float(sign), max(-0.5, min(0.5, 25.0 * (0.0 - float(p[1]))))]
        nrm = math.hypot(d[0], d[1])
        wx = (cy * d[0] - sy * d[1]) / nrm
        wy = (sy * d[0] + cy * d[1]) / nrm
        v = scene.cart.data.root_lin_vel_w[0]
        along = float(v[0]) * wx + float(v[1]) * wy
        f = 0.0 if along > vmax else force
        buf[0, 0] = f * wx
        buf[0, 1] = f * wy
        _step(2)
    buf[:] = 0.0
    _refresh()
    return ok or pred()


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.dressing_ferry_depot")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.25, -0.95, 0.80)) + o),
                                tuple(np.array((0.30, 0.00, 0.10)) + o),
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

    def cart_local():
        _refresh()
        return scene._cart_local()[0]

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(60)
    _report("settle")
    _REC["on"] = False
    states = torch.cat([b.data.root_state_w for b in
                        (scene.depot, scene.cart, scene.target, scene.decoy)], dim=-1)
    p = cart_local()
    tw = scene.target.data.root_pos_w[0]
    dw = scene.decoy.data.root_pos_w[0]
    sw = bool(scene.swap[0])
    t_side = -1.0 if sw else 1.0
    check("settle/no-NaN: layout settles finite; cart docked in the garage, both "
          "bottles upright at their floor stations (swap respected), score 0, no "
          "success",
          bool(torch.isfinite(states).all())
          and bool(scene.docked_now()[0]) and abs(float(p[0]) - (c.x_dock - 0.004)) < 0.01
          and abs(float(tw[1]) - t_side * c.station_y) < 0.04
          and abs(float(dw[1]) + t_side * c.station_y) < 0.04
          and float(tw[2]) > c.bot_off - 0.01 and float(dw[2]) > c.bot_off - 0.01
          and float(scene.score()[0]) == 0.0 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (3-seed max-pairwise readback) ====================
    obs = []
    for s in (101, 202, 303):
        torch.manual_seed(s)
        env.reset()
        _step(10)
        _refresh()
        obs.append((scene.depot.data.root_pos_w[0, :2].clone(),
                    math.degrees(_depot_yaw()),
                    scene.target.data.root_pos_w[0, :2].clone()))
    d_xy = max(float((a[0] - bb[0]).norm()) for i, a in enumerate(obs) for bb in obs[i + 1:])
    d_yaw = max(abs(a[1] - bb[1]) for i, a in enumerate(obs) for bb in obs[i + 1:])
    d_t = max(float((a[2] - bb[2]).norm()) for i, a in enumerate(obs) for bb in obs[i + 1:])
    print(f"[smoke] randomization max-pairwise deltas: depot_xy={d_xy * 1000:.1f}mm "
          f"depot_yaw={d_yaw:.2f}deg target_xy={d_t * 1000:.1f}mm", flush=True)
    check("randomization-is-real: depot xy / depot yaw / target-bottle xy readbacks "
          "differ across 3 seeds",
          d_xy > 0.005 and d_yaw > 1.0 and d_t > 0.005)

    # ================= 3. station-swap coverage ===================================================
    swaps, yaws = set(), []
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        swaps.add(bool(scene.swap[0]))
        yaws.append(math.degrees(_depot_yaw()))
    yaw_spread = max(yaws) - min(yaws)
    print(f"[smoke] over 10 resets: swaps={swaps} depot_yaw_spread={yaw_spread:.2f}deg",
          flush=True)
    check("swap coverage: both station assignments drawn over 10 resets and a real "
          "depot-yaw spread",
          swaps == {True, False} and yaw_spread > 3.0)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. SEED STRATEGY: put the bottle down at the docked receptacle =============
    # The seed's whole move — carry the object to the receptacle and set it down —
    # aimed at the docked cart. The closest reachable spot is the free slab area in
    # front of the well (the well itself is roof-blocked). Settled ON the cart,
    # under the roof, touching the receptacle: still not seated -> nothing.
    torch.manual_seed(100)
    env.reset()
    _step(40)
    _REC["on"] = True
    z_slab_top = c.deck_top + c.slab_t + 0.004
    _write_local(scene.target, c.x_dock - 0.004 - 0.050, 0.0, z_slab_top + c.bot_off + 0.002)
    ok_settle = _settle_until(lambda: bool(scene.settled(scene.target)[0]), max_steps=300)
    _report("seed-strategy")
    tb = scene._to_depot(scene.target.data.root_pos_w)[0]
    check("negative (SEED strategy): bottle set down settled ON the docked cart's "
          "slab under the roof (next to the roof-blocked well) -> not seated, no "
          "success, score 0",
          ok_settle and float(tb[0]) > 0.0 and float(tb[2]) > c.deck_top + c.slab_t
          and not bool(scene.seated_now(scene.target)[0])
          and float(scene.score()[0]) <= 1e-4 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 6. roof-block interlock (physical probe) ===================================
    # Cart docked. The bottle is dropped over the well's xy from above the garage and
    # then PRESSED down with a real 4 N force: the roof carries it. Anti-vacuity: the
    # bottle demonstrably fell onto the roof and the press held it there; its bottom
    # never got below the roof plane, so the well is unreachable while docked.
    torch.manual_seed(100)
    env.reset()
    _step(40)
    _REC["on"] = True
    roof_top = c.deck_t + c.roof_int + c.roof_t
    hover_z = roof_top + c.bot_off + 0.012
    _write_local(scene.target, c.x_dock - 0.004 + c.well_x, 0.0, hover_z)
    _step(30)
    scene.bottle_force[0, 2] = -4.0
    min_bot = 1e9
    for _ in range(120):
        _step(1)
        _refresh()
        tb = scene._to_depot(scene.target.data.root_pos_w)[0]
        min_bot = min(min_bot, float(tb[2]) - c.bot_off)
    scene.bottle_force[:] = 0.0
    _step(30)
    _report("roof-block")
    tb = scene._to_depot(scene.target.data.root_pos_w)[0]
    dropped = hover_z - float(tb[2])
    print(f"[smoke] roof-block: dropped={dropped * 1000:.1f}mm min_bottom={min_bot:.3f} "
          f"roof_top={roof_top:.3f}", flush=True)
    check("roof-block interlock (physical): bottle dropped over the well and pressed "
          "down 4 N with the cart docked FELL onto the roof (readback) and stayed ON "
          "it — bottom never below the roof plane, never seated, no success, score 0",
          dropped > 0.005 and min_bot > roof_top - 0.010
          and not bool(scene.seated_now(scene.target)[0]) and bool(scene.docked_now()[0])
          and float(scene.score()[0]) <= 1e-4 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 7. undelivered load: seated but still parked out ===========================
    torch.manual_seed(100)
    env.reset()
    _step(40)
    _place_cart_loaded(c.x_pull, seat_target=True)
    ok_settle = _settle_until(
        lambda: bool(scene.seated_now(scene.target)[0]) and bool(scene.settled(scene.target)[0])
        and bool(scene.settled(scene.cart)[0]), max_steps=300)
    _report("undelivered")
    check("undelivered load: bottle genuinely seated in the well but the cart parked "
          "OUT at the extraction line -> honest partial credit 0.50, no success",
          ok_settle and abs(float(scene.score()[0]) - 0.50) < 0.02
          and not bool(scene.docked_now()[0]) and not bool(scene.success()[0]))

    # ================= 8. dock near-miss ==========================================================
    torch.manual_seed(100)
    env.reset()
    _step(40)
    _place_cart_loaded(c.x_dock_min - 0.030, seat_target=True)
    ok_settle = _settle_until(
        lambda: bool(scene.seated_now(scene.target)[0]) and bool(scene.settled(scene.target)[0])
        and bool(scene.settled(scene.cart)[0]), max_steps=300)
    _report("dock-nearmiss")
    check("dock near-miss: seated bottle, cart 30 mm short of the dock line -> docked "
          "False, score strictly below the pre-success cap, no success",
          ok_settle and not bool(scene.docked_now()[0])
          and float(scene.score()[0]) <= 0.85 - 0.01 + 1e-6
          and not bool(scene.success()[0]))

    # ================= 9. wrong object: decoy in the well =========================================
    torch.manual_seed(100)
    env.reset()
    _step(40)
    _place_cart_loaded(c.x_dock - 0.004, seat_target=False)
    ok_settle = _settle_until(
        lambda: bool(scene.seated_now(scene.decoy)[0]) and bool(scene.settled(scene.decoy)[0]),
        max_steps=300)
    _report("wrong-object")
    check("wrong object: the RED decoy seated in the well with the cart docked -> "
          "success False, score ~0 (all credit stages are target-specific)",
          ok_settle and bool(scene.seated_now(scene.decoy)[0]) and bool(scene.docked_now()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.05)

    # ================= 10. latch persistence (physical pull + empty return) =======================
    torch.manual_seed(100)
    env.reset()
    _step(40)
    _REC["on"] = True
    ok_pull = _push_cart(-1.0, lambda: float(cart_local()[0]) <= c.x_pull,
                         force=4.0, vmax=0.10, iters=900)
    _step(30)
    _refresh()
    extracted = bool(scene._extracted[0])
    s_out = float(scene.score()[0])
    ok_back = _push_cart(+1.0, lambda: bool(scene.docked_now()[0]),
                         force=5.0, vmax=0.08, iters=1200)
    _step(30)
    _report("empty-return")
    check("latch persistence: real pull fires the extraction latch (0.15); pushing "
          "the EMPTY cart back to the dock keeps the credit at 0.15 (no decay, no "
          "growth), still no success",
          ok_pull and extracted and abs(s_out - 0.15) < 1e-4 and ok_back
          and not bool(scene.extracted_now()[0])
          and abs(float(scene.score()[0]) - 0.15) < 1e-4
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.dressing_ferry_depot")
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

    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()

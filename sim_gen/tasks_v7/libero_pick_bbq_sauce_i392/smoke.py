"""Smoke / rubric-REJECTION battery for TareLiftScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the latched
credit is monotone along a real trajectory). This battery proves the rubric REJECTS
wrong outcomes, and that the physical claims the task rests on — the weight gate
buries the loaded basket, the strips cap any free rise, unloading opens the gate — are
load-bearing. Every probe is CONSTRUCTED (teleport as transport/instrumentation, real
physics steps, judge); success() is monitored at EVERY step and must never turn True
anywhere in the battery (the audit is itself a check).

Checks:
   1. settle/no-NaN      — seeded reset settles finite; car at its BOTTOM stop (>= 2
                           cans aboard), bottle seated, basket in the car, the other
                           cans pre-binned; score 0, no success;
   2. randomization      — three seeded resets, max-pairwise READBACK: goal-pad dock
                           xy + yaw, bottle yaw, and the can arrangement all differ;
   3. ballast coverage   — over 8 resets the aboard-can count spans >= 2 distinct
                           values within [2, 4];
   4. null-policy        — 300 idle steps: score ~0, no success;
   5. SEED STRATEGY      — the seed family's naive plan ("pick up the bbq sauce and
                           put it at the goal"): the bottle alone transported out of
                           the pit and dropped at the goal pad — no success, score ~0
                           (nothing the rubric credits ever happened);
   6. loaded egress      — FLAGSHIP: worst-case load (exactly 1 can left aboard; the
                           readback depression matches the spring arithmetic), then
                           an escalated 10 N push (the jam is geometric, so the probe
                           wrench may exceed the solve's) drives the basket +x: it
                           presses the sill (moved, non-vacuous) and jams — never
                           out, egress latch never fires;
   7. loaded free-rise   — a full-weight regulated LIFT on the loaded basket: it only
                           presses into the roof strips and drags the CAR up through
                           them (readback) — strip-capped, never out of the pit;
                           released, falls back to the loaded depth, bottle seated;
   8. gate opens         — the same episode: the last can is dropped into the bin, the
                           spring alone raises the car to the top stop, and the SAME
                           push servo now slides the basket straight out past the
                           egress line (score hits the 0.70 latch cap, no success);
   9. near-miss off-pad  — basket + bottle settled on the deck 95 mm off the pad
                           centre (cans all binned): outside the pad tolerance — no
                           success, score <= 0.70;
  10. can still aboard   — basket + bottle centred ON the pad but one ballast can
                           still riding the basket: cans-binned clause refuses;
  11. missing bottle     — basket centred on the pad, cans binned, but the bottle
                           standing on the deck outside: seated clause refuses;
  12. tipped bottle      — the bottle lying on its side INSIDE the basket on the pad:
                           uprightness refuses;
  13. can on the deck    — one can parked on the open deck instead of in the bin
                           (basket + bottle on the pad): cans-binned clause refuses;
  14. settle gate        — the exact success state but the basket (bottle riding)
                           still sliding at ~0.45 m/s: success refuses while anything
                           moves (probe dismantled before it can settle into a real
                           success);
  15. rejection audit    — success() was never True at any step of this battery;
  16. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -u smoke.py --headless
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

_qapply, _qz, _qmul = task_scene._qapply, task_scene._qz, task_scene._qmul
encode_force = task_scene.encode_force
G = task_scene.G

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_WD = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_WD.daemon = True
_WD.start()

# ----- module state wired up in main() ------------------------------------------------------------
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


def _write_body(body, pos_w: torch.Tensor, quat: torch.Tensor | None = None,
                lin_vel: torch.Tensor | None = None) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat
    if lin_vel is not None:
        st[:, 7:10] = lin_vel
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    bl = scene._dock_local(scene.basket.data.root_pos_w)[0]
    print(f"[smoke] {tag:18s} | q={float(scene.car_q()[0]):+.4f} "
          f"bask_loc=({float(bl[0]):+.3f},{float(bl[1]):+.3f},{float(bl[2]):+.3f}) "
          f"binned={int(scene.cans_binned_count()[0])} "
          f"seat={bool(scene.bottle_seated()[0])} pad={bool(scene.basket_on_pad()[0])} "
          f"incar={bool(scene.basket_in_car()[0])} "
          f"settled={bool(scene.settled()[0])} score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tare_lift")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    zero = torch.zeros(n, 1, 3, device=device)

    def dock_pt(loc) -> torch.Tensor:
        _refresh()
        loc_t = torch.tensor(loc, device=device, dtype=torch.float).expand(n, 3)
        return scene.dock.data.root_pos_w + _qapply(scene.dock.data.root_quat_w, loc_t)

    def dquat() -> torch.Tensor:
        _refresh()
        return scene.dock.data.root_quat_w.clone()

    def pad_pt(loc) -> torch.Tensor:
        _refresh()
        loc_t = torch.tensor(loc, device=device, dtype=torch.float).expand(n, 3)
        return scene.pad.data.root_pos_w + _qapply(scene.pad.data.root_quat_w, loc_t)

    def pquat() -> torch.Tensor:
        _refresh()
        return scene.pad.data.root_quat_w.clone()

    def bask_x() -> float:
        _refresh()
        return float(scene._dock_local(scene.basket.data.root_pos_w)[0, 0])

    def bask_zloc() -> float:
        _refresh()
        return float(scene._dock_local(scene.basket.data.root_pos_w)[0, 2])

    def aboard_cans() -> list:
        _refresh()
        out = []
        for i, b in enumerate(scene.cans):
            loc = scene._basket_local(b.data.root_pos_w)[0]
            if (abs(float(loc[0])) < 0.090 and abs(float(loc[1])) < 0.090
                    and -0.02 < float(loc[2]) < 0.12):
                out.append(i)
        return out

    def free_parks() -> list:
        _refresh()
        occ = [tuple(float(v) for v in scene._dock_local(b.data.root_pos_w)[0])
               for b in scene.cans]
        free = []
        for px, py in c.bin_parks:
            taken = any(abs(ox - px) < 0.045 and abs(oy - py) < 0.045
                        and oz > c.deck_top for ox, oy, oz in occ)
            if not taken:
                free.append((px, py))
        return free

    def drop_can_to_bin(i: int) -> bool:
        """Transport can i to a hover over a free bin park; gravity does the rest."""
        for _attempt in range(4):
            parks = free_parks()
            assert parks, "no free bin park"
            px, py = parks[0]
            _write_body(scene.cans[i],
                        dock_pt((px, py, c.deck_top + c.can_h / 2 + 0.050)), dquat())
            _step(120)
            if bool(scene.in_bin(scene.cans[i])[0]):
                return True
        return False

    def unload_to(k: int) -> None:
        """Remove aboard cans (transport + gravity drop) until only k remain."""
        while len(aboard_cans()) > k:
            assert drop_can_to_bin(aboard_cans()[0]), "probe setup: can must bin"

    def push_basket(tgt_fn, tag: str, timeout: int = 900, stop_x: float | None = None,
                    tol: float = 0.015, floor_start: float = 1.0,
                    floor_cap: float = 4.5, f_clamp: float = 8.0):
        """The solve's own escalating velocity-regulated push (never a teleport).
        Loaded probes pass a raised floor/clamp: the LOADED basket weighs 2.2x the
        solve's, so the solve-tuned stiction floor cannot even reach the sill — the
        jam claim is geometric, so the probe wrench may exceed the solve's.
        Returns (max dock-x reached, final dock-x)."""
        mode = 0
        q_ref = scene.basket.data.root_quat_w.clone()
        floor_f = floor_start
        err0 = tgt_fn() - scene.basket.data.root_pos_w
        err0[:, 2] = 0.0
        win_i, win_d = 0, float(err0.norm(dim=-1)[0])
        max_x = bask_x()
        for i in range(timeout):
            if stop_x is not None and bask_x() >= stop_x:
                break
            err = tgt_fn() - scene.basket.data.root_pos_w
            err[:, 2] = 0.0
            d = float(err.norm(dim=-1)[0])
            if d <= tol:
                break
            dirn = err / max(d, 1e-6)
            v = scene.basket.data.root_lin_vel_w.clone()
            v[:, 2] = 0.0
            v_along = float((v * dirn).sum(dim=-1)[0])
            v_des = 0.08 if d > 0.040 else 0.035
            f_mag = 6.0 * (v_des - v_along)
            if abs(v_along) < 0.02 and f_mag < floor_f:
                f_mag = floor_f
            f_mag = max(-f_clamp, min(f_clamp, f_mag))
            v_perp = v - dirn * (v * dirn).sum(dim=-1, keepdim=True)
            f_world = dirn * f_mag - 3.0 * v_perp
            f_arg = encode_force(mode, q_ref, scene.basket.data.root_quat_w, f_world)
            scene.basket.set_external_force_and_torque(
                f_arg.view(n, 1, 3), zero, env_ids=_all_ids(), is_global=True)
            _step(1)
            max_x = max(max_x, bask_x())
            if i - win_i >= 45:
                e2 = tgt_fn() - scene.basket.data.root_pos_w
                e2[:, 2] = 0.0
                e = float(e2.norm(dim=-1)[0])
                if e > win_d + 0.004:
                    mode = 1 - mode
                    print(f"[smoke] {tag}: moving away; force-frame mode -> {mode}",
                          flush=True)
                elif e > win_d - 0.002:
                    floor_f = min(floor_f + 0.5, floor_cap)
                win_i, win_d = i, e
        scene.basket.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
        _step(45)
        return max_x, bask_x()

    def place_basket(pos_w: torch.Tensor, quat: torch.Tensor, *,
                     with_bottle: bool = True, vel=None) -> None:
        """Write the basket (and its seated bottle) as one construct; settle later."""
        _write_body(scene.basket, pos_w, quat, lin_vel=vel)
        if with_bottle:
            bot = pos_w + _qapply(quat, torch.tensor(
                [0.0, 0.0, 0.006], device=device).expand(n, 3))
            _write_body(scene.bottle, bot, quat, lin_vel=vel)

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.32, -0.72, 0.62)) + o),
                                tuple(np.array((0.36, 0.00, 0.10)) + o),
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

    checks: list = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def yaw_of(quat_row: torch.Tensor) -> float:
        return math.degrees(2.0 * math.atan2(float(quat_row[3]), float(quat_row[0])))

    def dyaw(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    db = scene._dock_build
    print(f"[smoke] dock build pose: ({db[0]:+.3f},{db[1]:+.3f},yaw={db[2]:+.3f}rad)",
          flush=True)
    d1 = ((c.car_mass + c.bask_mass + c.bot_mass + c.can_mass) * G
          - c.spring_k * c.spring_target) / c.spring_k  # 1-can depression

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _AUD["on"] = True
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    ab0 = aboard_cans()
    check("settle/no-NaN: layout settles finite; car at its BOTTOM stop under >= 2 "
          "cans, bottle seated, basket in the car, other cans pre-binned; score 0, "
          "no success",
          bool(scene._finite()[0]) and 2 <= len(ab0) <= 4
          and float(scene.car_q()[0]) <= -(c.travel - 0.004)
          and bool(scene.bottle_seated()[0]) and bool(scene.basket_in_car()[0])
          and int(scene.cans_binned_count()[0]) == 4 - len(ab0)
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (3-seed max-pairwise readback) ===================
    def readback():
        _refresh()
        pl = scene._dock_local(scene.pad.data.root_pos_w)[0, :2].clone()
        pyaw = dyaw(yaw_of(scene.pad.data.root_quat_w[0]),
                    yaw_of(scene.dock.data.root_quat_w[0]))
        byaw = yaw_of(scene.bottle.data.root_quat_w[0])
        cans = torch.cat([scene._dock_local(b.data.root_pos_w)[0, :2]
                          for b in scene.cans]).clone()
        return pl, pyaw, byaw, cans, len(aboard_cans())

    rb = []
    for s in (101, 202, 303):
        torch.manual_seed(s)
        env.reset()
        _step(10)
        rb.append(readback())
    d_pad = max(float((a[0] - b[0]).norm()) for a in rb for b in rb)
    d_pyaw = max(dyaw(a[1], b[1]) for a in rb for b in rb)
    d_byaw = max(dyaw(a[2], b[2]) for a in rb for b in rb)
    d_cans = max(float((a[3] - b[3]).abs().max()) for a in rb for b in rb)
    print(f"[smoke] randomization deltas (max-pairwise over 3 seeds): "
          f"pad_xy={d_pad * 1000:.1f}mm pad_yaw={d_pyaw:.1f}deg "
          f"bottle_yaw={d_byaw:.1f}deg cans={d_cans * 1000:.1f}mm "
          f"n_aboard={[r[4] for r in rb]}", flush=True)
    check("randomization-is-real: goal-pad xy + yaw, bottle yaw and the can "
          "arrangement readback all differ across seeds",
          d_pad > 0.005 and d_pyaw > 1.0 and d_byaw > 5.0 and d_cans > 0.005)

    # ================= 3. ballast coverage ========================================================
    ns = []
    for s in range(8):
        torch.manual_seed(500 + s)
        env.reset()
        _step(5)
        ns.append(len(aboard_cans()))
    print(f"[smoke] aboard-can count over 8 resets: {ns}", flush=True)
    check("ballast coverage: over 8 resets the aboard count spans >= 2 distinct "
          "values within [2, 4]",
          all(2 <= v <= 4 for v in ns) and len(set(ns)) >= 2)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(300)
    _report("null-policy")
    check("null-policy-fails: 300 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.03 and not bool(scene.success()[0]))

    # ================= 5. negative: the SEED FAMILY'S OWN STRATEGY ================================
    # The seed picks up the bbq sauce and places it at the goal. Transport the BOTTLE
    # alone out of the pit and drop it at the goal pad — tidy, settled, and worthless:
    # the rubric pays for the basket delivery and the ballast, not the bottle's spot.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _write_body(scene.bottle, pad_pt((0.0, 0.0, 0.040)), pquat())
    _step(150)
    _report("seed-strategy")
    check("negative (SEED strategy): the bottle alone transported onto the goal pad "
          "— no success, score ~0 (no clause and no latch credits it)",
          not bool(scene.bottle_seated()[0])
          and float(scene.bottle.data.root_pos_w[0, 2]
                    - env.iscene.env_origins[0, 2]) > c.deck_top - 0.010
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 6 + 7 + 8: the WEIGHT GATE (flagship) ======================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    unload_to(1)  # worst case: exactly ONE can left aboard
    _step(90)
    q_loaded = float(scene.car_q()[0])
    print(f"[smoke] 1-can depression readback: q={q_loaded * 1000:+.1f}mm "
          f"(predicted {-d1 * 1000:+.1f}mm)", flush=True)
    x0 = bask_x()
    _REC["on"] = True
    max_x, end_x = push_basket(lambda: dock_pt((0.245, 0.0, 0.0)), "loaded-push",
                               timeout=600, floor_start=6.0, floor_cap=10.0,
                               f_clamp=12.0)
    _REC["on"] = False
    _step(30)
    _report("loaded-push")
    adv = max_x - x0
    print(f"[smoke] loaded push: x {x0 * 1000:+.1f} -> max {max_x * 1000:+.1f}mm "
          f"(advance {adv * 1000:.1f}mm, egress line {c.egress_x_min * 1000:.0f}mm)",
          flush=True)
    check("FLAGSHIP (loaded egress denied): with one can aboard the car sits at the "
          "predicted depth and an escalated 10 N push presses the sill "
          "(moved >= 2 mm, non-vacuous) yet jams far short of the egress line",
          abs(q_loaded + d1) <= 0.006 and 0.002 <= adv <= 0.035
          and max_x < 0.06 and not bool(scene._egressed[0])
          and bool(scene.basket_in_car()[0]) and bool(scene.bottle_seated()[0])
          and not bool(scene.success()[0]))

    # re-square the basket in the car by teleport (accumulated yaw would corner-catch
    # later probes; teleport = instrumentation, the physics claims are already judged).
    # The can keeps its world pose (a few mm of basket shift under it is fine).
    _refresh()
    tgt = dock_pt((0.002, 0.0, 0.0))
    tgt[:, 2] = scene.basket.data.root_pos_w[:, 2]
    place_basket(tgt, dquat())
    _step(90)

    # ----- 7. loaded free-rise denied (strip cap + car-follow readback) ---------------------------
    z0, q0 = bask_zloc(), float(scene.car_q()[0])
    mode_l = 0
    q_ref_l = scene.basket.data.root_quat_w.clone()
    max_z, max_q, max_xl = z0, q0, bask_x()
    # mg feedforward (basket + bottle + one can) + velocity regulation, 25 N cap:
    # a genuine LIFT, not a token tug — the strip cap must carry the whole load.
    mg_lift = (c.bask_mass + c.bot_mass + c.can_mass) * G
    _REC["on"] = True
    for i in range(300):
        vz = float(scene.basket.data.root_lin_vel_w[0, 2])
        f_up = mg_lift + 40.0 * (0.04 - vz)
        f_up = max(0.0, min(25.0, f_up))
        f_world = torch.zeros(n, 3, device=device)
        f_world[:, 2] = f_up
        f_arg = encode_force(mode_l, q_ref_l, scene.basket.data.root_quat_w, f_world)
        scene.basket.set_external_force_and_torque(
            f_arg.view(n, 1, 3), zero, env_ids=_all_ids(), is_global=True)
        _step(1)
        max_z = max(max_z, bask_zloc())
        max_q = max(max_q, float(scene.car_q()[0]))
        max_xl = max(max_xl, bask_x())
        if i == 45 and max_z < z0 + 0.001:
            mode_l = 1 - mode_l
            print("[smoke] loaded-lift: no rise; force-frame mode flipped", flush=True)
    scene.basket.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
    _step(240)
    _report("loaded-lift")
    print(f"[smoke] loaded lift: bask z {z0 * 1000:+.1f} -> max {max_z * 1000:+.1f}mm "
          f"(deck {c.deck_top * 1000:.0f}mm), car q {q0 * 1000:+.1f} -> max "
          f"{max_q * 1000:+.1f}mm, settled back q={float(scene.car_q()[0]) * 1000:+.1f}mm",
          flush=True)
    check("loaded free-rise denied: a full-weight 25 N-capped lift only presses the "
          "basket into the roof strips and drags the CAR up through them (readback) "
          "— strip-capped, never above the car's top stop, never out of the pit; "
          "released, everything falls back loaded and the bottle stays seated",
          0.004 <= max_z - z0 <= 0.045 and max_z <= c.q_top_z + 0.020
          and max_q >= q0 + 0.004 and max_xl < c.egress_x_min - 0.05
          and float(scene.car_q()[0]) <= -(d1 - 0.008)
          and bool(scene.basket_in_car()[0]) and bool(scene.bottle_seated()[0])
          and not bool(scene._egressed[0]) and not bool(scene.success()[0]))

    # ----- 8. the same probe with the gate OPEN ---------------------------------------------------
    assert drop_can_to_bin(aboard_cans()[0]), "probe setup: last can must bin"
    risen = False
    for _ in range(400):
        _step(1)
        if float(scene.car_q()[0]) >= -c.risen_q_tol and bool(scene.settled()[0]):
            risen = True
            break
    _step(60)
    # re-square by teleport before the push: yaw accumulated across the loaded probes
    # corner-catches the doorway (the solve pushes from a pristine spawn; the CLAIM
    # here is only that the latch fires on a real slide, so squaring is fair setup).
    _refresh()
    tgt = dock_pt((0.002, 0.0, 0.0))
    tgt[:, 2] = scene.basket.data.root_pos_w[:, 2]
    place_basket(tgt, dquat())
    _step(60)
    x1 = bask_x()
    # Same 10 N grind budget the LOADED probe was allowed — the honest comparison is
    # identical wrench, loaded jams at the sill vs open slides clean out. The sill
    # step-down can corner-catch a slow slide, so rock back and re-approach on a stall.
    max_x2, end_x2 = x1, x1
    for _try in range(3):
        mx, ex = push_basket(lambda: dock_pt((0.245, 0.0, 0.0)), f"open-push-{_try}",
                             timeout=1200, stop_x=0.212, floor_start=6.0,
                             floor_cap=10.0, f_clamp=12.0)
        max_x2, end_x2 = max(max_x2, mx), ex
        if ex > c.egress_x_min:
            break
        back_x = max(ex - 0.030, 0.002)
        push_basket(lambda bx=back_x: dock_pt((bx, 0.0, 0.0)), "rock-back",
                    timeout=240, tol=0.010, floor_start=6.0, floor_cap=10.0,
                    f_clamp=12.0)
        _step(30)
    _REC["on"] = False
    _step(60)
    _report("open-push")
    s8 = float(scene.score()[0])
    print(f"[smoke] open push: x {x1 * 1000:+.1f} -> {end_x2 * 1000:+.1f}mm, "
          f"score={s8:.4f}", flush=True)
    check("gate opens: last can binned -> the spring ALONE raises the car to its top "
          "stop, and the SAME push servo with the SAME 10 N grind budget now slides "
          "the basket past the egress line (score at the 0.70 cap, no success)",
          risen and end_x2 > c.egress_x_min and bool(scene._egressed[0])
          and bool(scene.bottle_seated()[0])
          and abs(s8 - 0.70) <= 0.005 and not bool(scene.success()[0]))

    # ================= 9. near-miss: off the pad ==================================================
    # place at pad-frame x = -95 mm, deck rest height
    tgt = pad_pt((-0.095, 0.0, 0.0))
    tgt[:, 2] = env.iscene.env_origins[:, 2] + c.deck_top + c.bask_floor_t / 2 + 0.003
    place_basket(tgt, pquat())
    _step(90)
    _report("off-pad")
    check("near-miss (off-pad): basket + bottle settled on the deck 95 mm off the "
          "pad centre, cans binned — outside tolerance: no success, score <= 0.70",
          bool(scene.cans_binned()[0]) and bool(scene.bottle_seated()[0])
          and not bool(scene.basket_on_pad()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.70 + 1e-6)

    # ================= 10. can still aboard on the pad ============================================
    # FIRST re-board a can (off-pad), THEN move to the pad centre: no success window.
    _refresh()
    can_i = 0
    slot = torch.tensor([0.055, 0.055, c.bask_floor_t / 2 + c.can_h / 2 + 0.003],
                        device=device).expand(n, 3)
    _write_body(scene.cans[can_i],
                scene.basket.data.root_pos_w + _qapply(scene.basket.data.root_quat_w,
                                                       slot),
                scene.basket.data.root_quat_w.clone())
    _step(60)
    tgt = pad_pt((0.0, 0.0, 0.0))
    tgt[:, 2] = env.iscene.env_origins[:, 2] + c.deck_top + c.bask_floor_t / 2 + 0.003
    place_basket(tgt, pquat())
    _write_body(scene.cans[can_i],
                scene.basket.data.root_pos_w + _qapply(scene.basket.data.root_quat_w,
                                                       slot),
                scene.basket.data.root_quat_w.clone())
    _REC["on"] = True
    _step(90)
    _REC["on"] = False
    _report("can-aboard")
    check("wrong content (can still aboard): basket + bottle centred ON the pad but "
          "one ballast can riding the basket — cans-binned clause refuses",
          bool(scene.basket_on_pad()[0]) and bool(scene.bottle_seated()[0])
          and not bool(scene.cans_binned()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.70 + 1e-6)

    # ================= 11 + 13. missing bottle / can on the deck ==================================
    # Bottle out FIRST (kills the seated clause), then the can placements.
    _write_body(scene.bottle, dock_pt((0.50, 0.20, c.deck_top + 0.003)), dquat())
    _step(60)
    # can on the open deck (outside the bin)
    _write_body(scene.cans[can_i], dock_pt((0.52, -0.24, c.deck_top + c.can_h / 2
                                            + 0.003)), dquat())
    _step(90)
    _report("can-on-deck")
    check("wrong place (can on the deck): one can parked on the open deck instead of "
          "the bin (basket on the pad) — no success",
          bool(scene.basket_on_pad()[0]) and not bool(scene.cans_binned()[0])
          and not bool(scene.in_bin(scene.cans[can_i])[0])
          and not bool(scene.success()[0]))
    assert drop_can_to_bin(can_i), "probe setup: can must return to the bin"
    _step(60)
    _report("missing-bottle")
    check("missing bottle: basket centred on the pad, cans all binned, bottle "
          "standing on the deck outside the basket — seated clause refuses",
          bool(scene.basket_on_pad()[0]) and bool(scene.cans_binned()[0])
          and not bool(scene.bottle_seated()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.70 + 1e-6)

    # ================= 12. tipped bottle ==========================================================
    _refresh()
    qx90 = torch.zeros(n, 4, device=device)
    qx90[:, 0] = math.cos(math.pi / 4)
    qx90[:, 1] = math.sin(math.pi / 4)
    lie = _qmul(scene.basket.data.root_quat_w, qx90)
    _write_body(scene.bottle,
                scene.basket.data.root_pos_w + _qapply(
                    scene.basket.data.root_quat_w,
                    torch.tensor([0.0, 0.0, 0.050], device=device).expand(n, 3)),
                lie)
    _REC["on"] = True
    _step(120)
    _REC["on"] = False
    _report("tipped-bottle")
    check("tipped bottle: the bottle lying on its side INSIDE the basket on the pad "
          "— uprightness refuses",
          bool(scene.basket_on_pad()[0]) and bool(scene.cans_binned()[0])
          and not bool(scene.bottle_seated()[0]) and not bool(scene.success()[0]))

    # ================= 14. settle gate ============================================================
    # The exact success construct — but SLIDING at ~0.45 m/s. success() must refuse
    # while anything moves; the probe is dismantled before it could settle for real.
    _refresh()
    vel = _qapply(dquat(), torch.tensor([0.45, 0.0, 0.0], device=device).expand(n, 3))
    tgt = pad_pt((-0.030, 0.0, 0.0))
    tgt[:, 2] = env.iscene.env_origins[:, 2] + c.deck_top + c.bask_floor_t / 2 + 0.003
    place_basket(tgt, pquat(), vel=vel)
    moving_ok = True
    for _ in range(5):
        _step(1)
        v = float(scene.basket.data.root_lin_vel_w[0].norm())
        moving_ok = moving_ok and v > 0.05 and not bool(scene.settled()[0]) \
            and not bool(scene.success()[0])
    _report("settle-gate")
    # dismantle before it can settle into a real success: bottle out of the basket
    _write_body(scene.bottle, dock_pt((0.50, 0.20, c.deck_top + 0.003)), dquat())
    _step(60)
    check("settle gate: the exact success state with the basket (bottle riding) "
          "still sliding at ~0.45 m/s is refused while anything moves",
          moving_ok)

    # ================= 15. rejection audit ========================================================
    check("rejection audit: success() was never True at any step of this battery",
          _AUD["hits"] == 0)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.tare_lift")
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
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 - die loudly, never idle until the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)

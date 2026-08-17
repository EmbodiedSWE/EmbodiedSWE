"""Smoke / rubric-REJECTION battery for CrateFlipPackScene — NullRobot, teleported probes.

solve.py is the acceptance proof (probe -> two contact-dynamics quarter-tips -> drop-in
pack, latched credit monotone along a real trajectory). This battery proves the rubric
REJECTS wrong outcomes and that the physical claims are load-bearing: the trapped shoes
really are CAPTIVE (a 4 N push — ~3x a shoe's weight — slides one into the interior
wall and it never leaves the crate's footprint), the seed task's pick-and-drop strategy
delivers NOTHING against the inverted crate (a shoe released above it lands on the
upturned base and earns zero), GEOMETRIC CONTAINMENT WITHOUT ERECTION is worthless
(at spawn the shoes are fully "inside" the crate body frame and the score is 0),
near-misses are refused (crate perched on a shoe, shoe across the rim, crate on its
side), and the judged erection is LIVE (tip a real success back over and success
collapses while the latched credit survives). Every probe is CONSTRUCTED as a state
(teleport, real physics steps, judge) — instrumentation, never a solution.

Checks:
   1. settle/no-NaN     — seeded reset settles finite; crate READBACK mouth-down at
                          rim-rest height, both shoes COVERED beneath it; score 0;
   2. randomization     — three seeded resets: max-pairwise READBACK deltas of crate
                          yaw, crate xy and shoe_a spawn all differ;
   3. null-policy       — 240 idle steps: score ~0, no success;
   4. SEED STRATEGY     — the seed's whole plan (carry a shoe over the container and
                          release) against the INVERTED crate: shoe_a dropped twice
                          from above lands on the upturned base / slides off — never
                          inside, no credit, score 0, no success;
   5. CAPTIVITY PUSH    — a trapped shoe pushed with 4 N (~3x its weight): it MOVES
                          (>= 5 mm — the probe is not vacuous), jams against the
                          interior wall and never leaves the footprint; still
                          covered, score stays 0: the shoes are unreachable until
                          the crate itself is moved;
   6. GEOMETRIC-CONTAINMENT flagship — at spawn the rubric's own `inside` predicate
                          is TRUE for both shoes (mouth-down crate standing over
                          them) yet packed credit stays unset, score 0, no success:
                          containment without erection is worthless;
   7. uncover only      — crate set on its SIDE away from the shoes: uncover latch
                          0.15, no erect credit, no success;
   8. PERCHED crate     — upright crate dropped with one wall over a shoe: it rests
                          tilted / raised on the shoe — receptacle refused (z or
                          up-axis clause), no erect credit, no success;
   9. RIM near-miss     — upright crate, a shoe balanced flat on the rim wall with
                          its toe over the cavity: above the interior (z clause),
                          not inside, no packed credit, no success;
  10. crate-on-side pack — both shoes tucked into the cavity of a crate lying on its
                          SIDE (geometrically inside): no success, no packed credit
                          (receptacle false);
  11. uncover+erect only — crate upright at rest, shoes on the floor: score ~0.30,
                          no success;
  12. one shoe in      — crate upright, shoe A inside settled, B on the floor:
                          score ~0.50, no success;
  13. TIP-BACK liveness — full success CONSTRUCTED (crate upright, both shoes
                          dropped in; success live, score 1.0), then the crate
                          tipped back onto its side: success COLLAPSES while the
                          latched score survives at ~0.70 — erection is judged
                          live, not latched;
  14. rejection audit   — success() never fired at any step during checks 4-12;
  15. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.put_shoes_in_box_i379.smoke --headless
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

_qmul, _qx, _qz = task_scene._qmul, task_scene._qx, task_scene._qz

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
    env = ENVS.get("simgen.crate_flip_pack")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.20, -0.85, 0.75)) + o),
                                tuple(np.array((0.42, 0.05, 0.06)) + o),
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

    ez1 = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
    ex1 = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)

    def crate_up_z() -> float:
        _refresh()
        return float(quat_apply(scene.crate.data.root_quat_w, ez1)[0, 2])

    def crate_axis_yaw() -> float:
        """Heading of the crate's long axis in the ground plane (deg) — well-defined
        at any roll, unlike a naive quat yaw."""
        _refresh()
        u = quat_apply(scene.crate.data.root_quat_w, ex1)[0]
        return math.degrees(math.atan2(float(u[1]), float(u[0])))

    def dyaw(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    def crate_pose(x_loc: float, y_loc: float, z_loc: float, yaw_rel: float) -> torch.Tensor:
        """(N,13) root state at crate-local coords, crate-relative yaw, zero velocity."""
        _refresh()
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0], loc[:, 1], loc[:, 2] = x_loc, y_loc, z_loc
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.crate.data.root_pos_w + quat_apply(
            scene.crate.data.root_quat_w, loc)
        st[:, 3:7] = _qmul(scene.crate.data.root_quat_w,
                           _qz(torch.full((n,), yaw_rel, device=device)))
        return st

    def world_pose(x: float, y: float, z: float, yaw: float = 0.0,
                   roll_x: float | None = None) -> torch.Tensor:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        q = _qz(torch.full((n,), yaw, device=device))
        if roll_x is not None:
            q = _qmul(q, _qx(torch.full((n,), roll_x, device=device)))
        st[:, 3:7] = q
        st[:, 0:3] += env.iscene.env_origins
        return st

    def loc_of(body) -> torch.Tensor:
        _refresh()
        return quat_apply_inverse(scene.crate.data.root_quat_w,
                                  body.data.root_pos_w - scene.crate.data.root_pos_w)[0]

    def report(tag: str) -> None:
        s = scene._status()
        parts = []
        for nm in scene.SHOE_NAMES:
            p = loc_of(scene.shoes[nm])
            parts.append(f"{nm}=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f})")
        print(f"[smoke] {tag:16s} | up_z={float(s['up_z'][0]):+.2f} "
              f"crate_z={float(s['crate_z'][0]):+.4f} " + " ".join(parts)
              + f" inside={s['inside'][0].tolist()} covered={s['covered'][0].tolist()} "
              f"receptacle={bool(s['receptacle'][0])} score={float(scene.score()[0]):.3f} "
              f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
              flush=True)

    def crate_upright_at(x: float, y: float, yaw: float = 0.0) -> None:
        """Construct: crate standing on its base at a world spot (drop 2 mm, settle)."""
        scene.crate.write_root_state_to_sim(
            world_pose(x, y, c.rest_z + 0.002, yaw=yaw), _all_ids())
        _step(120)

    def crate_on_side_at(x: float, y: float) -> None:
        """Construct: crate resting on its -y wall outer face (up axis horizontal)."""
        scene.crate.write_root_state_to_sim(
            world_pose(x, y, c.in_w / 2 + c.wall_t + 0.002, roll_x=math.pi / 2),
            _all_ids())
        _step(150)

    def drop_shoe_in(name: str, x_loc: float) -> None:
        """Release a shoe from a hover above the (upright) crate mouth; settles inside."""
        scene.shoes[name].write_root_state_to_sim(
            crate_pose(x_loc, 0.0, c.in_h + c.sole_t / 2 + 0.008, 0.0), _all_ids())
        _step(150)

    def shoe_floor(name: str, x: float, y: float, yaw: float = 0.0) -> None:
        scene.shoes[name].write_root_state_to_sim(
            world_pose(x, y, c.sole_t / 2 + 0.002, yaw=yaw), _all_ids())
        _step(60)

    zero_w = torch.zeros(n, 1, 3, device=device)

    # ================= 1. settle / no-NaN / crate starts mouth-down over the shoes ===============
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    report("settle")
    _REC["on"] = False
    bodies = [scene.crate, *scene.shoes.values()]
    finite = all(bool(torch.isfinite(b.data.root_pos_w).all()) for b in bodies)
    s = scene._status()
    mouth_down = float(s["up_z"][0]) < -0.95
    at_inv_height = abs(float(s["crate_z"][0]) - c.inv_z) < 0.008
    covered0 = bool(s["covered"][0].all())
    check("settle/no-NaN: seeded reset settles finite, crate READBACK mouth-down at "
          "rim-rest height, both shoes covered beneath it, score 0, no success",
          finite and mouth_down and at_inv_height and covered0
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (3-seed max-pairwise readback) ===================
    def readback():
        _refresh()
        return (crate_axis_yaw(),
                scene.crate.data.root_pos_w[0, :2].clone(),
                scene.shoes["shoe_a"].data.root_pos_w[0, :2].clone())

    obs = []
    for sd in (101, 202, 303):
        torch.manual_seed(sd)
        env.reset()
        _step(10)
        obs.append(readback())
    d_yawv = max(dyaw(a[0], b[0]) for a, b in ((obs[0], obs[1]), (obs[0], obs[2]),
                                               (obs[1], obs[2])))
    d_bp = max(float((a[1] - b[1]).norm()) for a, b in ((obs[0], obs[1]),
                                                        (obs[0], obs[2]), (obs[1], obs[2])))
    d_sp = max(float((a[2] - b[2]).norm()) for a, b in ((obs[0], obs[1]),
                                                        (obs[0], obs[2]), (obs[1], obs[2])))
    print(f"[smoke] randomization max-pairwise deltas: crate_yaw={d_yawv:.1f}deg "
          f"crate_xy={d_bp * 1000:.1f}mm shoe_a_xy={d_sp * 1000:.1f}mm", flush=True)
    check("randomization-is-real: crate long-axis yaw, crate xy and shoe_a spawn "
          "readback differ across seeds (max-pairwise)",
          d_yawv > 5.0 and d_bp > 0.003 and d_sp > 0.01)

    # ================= 3. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    _AUDIT["on"] = True  # ---- negative probes below: success must never fire ----

    # ================= 4. SEED STRATEGY: carry-and-release onto the INVERTED crate ================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    # shoe_b stays trapped (so no probe-granted uncover credit); shoe_a gets the
    # seed's transport for free: released twice from above the crate.
    top_z = c.inv_z + c.floor_t + 0.040
    for (dx, dy) in ((0.03, 0.03), (-0.05, -0.06)):
        _refresh()
        cp = scene.crate.data.root_pos_w[0] - env.iscene.env_origins[0]
        scene.shoes["shoe_a"].write_root_state_to_sim(
            world_pose(float(cp[0]) + dx, float(cp[1]) + dy, top_z, yaw=0.7), _all_ids())
        _step(200)
    report("seed-strategy")
    _REC["on"] = False
    s = scene._status()
    check("negative (SEED strategy): a shoe carried over the inverted crate and "
          "released lands on the upturned base / slides off — never inside, no "
          "packed/erect credit, score 0, no success",
          not bool(s["inside"][0, 0]) and not bool(scene._packed[0].any())
          and not bool(scene._erected[0]) and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))

    # ================= 5. CAPTIVITY PUSH: trapped shoes cannot leave the footprint ================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    p0 = scene.shoes["shoe_a"].data.root_pos_w[0].clone()
    # push horizontally along the crate's long axis with 4 N (~3x shoe weight)
    _refresh()
    u = quat_apply(scene.crate.data.root_quat_w, ex1)
    u[:, 2] = 0.0
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    f = (4.0 * u).reshape(n, 1, 3)
    scene.shoes["shoe_a"].set_external_force_and_torque(f, zero_w, env_ids=_all_ids(),
                                                        is_global=True)
    _step(180)
    scene.shoes["shoe_a"].set_external_force_and_torque(zero_w, zero_w, env_ids=_all_ids())
    _step(60)
    report("captivity-push")
    _REC["on"] = False
    p1 = scene.shoes["shoe_a"].data.root_pos_w[0]
    moved = float((p1[:2] - p0[:2]).norm())
    sl = loc_of(scene.shoes["shoe_a"])
    in_footprint = (abs(float(sl[0])) < c.in_l / 2 + c.wall_t
                    and abs(float(sl[1])) < c.in_w / 2 + c.wall_t)
    s = scene._status()
    print(f"[smoke] captivity push: moved {moved * 1000:.1f} mm, "
          f"loc=({float(sl[0]):+.3f},{float(sl[1]):+.3f},{float(sl[2]):+.3f})", flush=True)
    check("negative (CAPTIVITY): a trapped shoe pushed with 4 N (~3x its weight) "
          "MOVES (probe not vacuous) but jams against the interior wall and never "
          "leaves the crate footprint; still covered, score stays 0",
          moved >= 0.005 and in_footprint and bool(s["covered"][0].all())
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 6. GEOMETRIC CONTAINMENT without erection is worthless =====================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    s = scene._status()
    report("geom-containment")
    check("flagship (GEOMETRIC CONTAINMENT): at spawn the rubric's own `inside` "
          "predicate is TRUE for both shoes under the mouth-down crate, yet packed "
          "credit stays unset and score is 0 — containment without erection is "
          "worthless",
          bool(s["inside"][0].all()) and not bool(scene._packed[0].any())
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 7. uncover only ============================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    crate_on_side_at(0.85, -0.35)
    report("uncover-only")
    sc = float(scene.score()[0])
    check("partial (uncover only): crate set on its side away from the shoes — "
          "score ~0.15, no erect credit, no success",
          0.14 <= sc <= 0.16 and not bool(scene._erected[0])
          and not bool(scene.success()[0]))

    # ================= 8. PERCHED crate ===========================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    # clear the crate away first, park the shoes, then drop the crate onto shoe_a
    crate_on_side_at(0.85, -0.35)
    shoe_floor("shoe_a", 0.40, 0.30, yaw=0.0)      # length along x
    shoe_floor("shoe_b", 0.10, -0.30, yaw=0.5)
    # crate upright, its -y wall midline directly over the shoe: rests on the heel
    scene.crate.write_root_state_to_sim(
        world_pose(0.40, 0.30 + c.in_w / 2 + c.wall_t / 2, c.shoe_h + c.floor_t + 0.006,
                   yaw=0.0), _all_ids())
    _step(240)
    report("perched")
    _REC["on"] = False
    s = scene._status()
    perched = (float(s["crate_z"][0]) > c.rest_z + c.rest_z_tol
               or float(s["up_z"][0]) < math.cos(math.radians(c.up_tol_deg)))
    on_shoe = float(s["crate_z"][0]) > c.rest_z + 0.010
    check("negative (PERCHED): upright crate dropped with one wall over a shoe rests "
          "raised/tilted on it — receptacle refused (z or up-axis clause), no erect "
          "credit, no success",
          perched and on_shoe and not bool(s["receptacle"][0])
          and not bool(scene._erected[0]) and not bool(scene.success()[0]))

    # ================= 9. RIM near-miss ===========================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    crate_upright_at(0.42, 0.0)
    # shoe balanced flat on the +x rim wall, toe pointing over the cavity; its CoM
    # (15 mm behind the sole center) sits over the wall strip so it balances there
    scene.shoes["shoe_a"].write_root_state_to_sim(
        crate_pose(c.in_l / 2 + c.wall_t / 2 - 0.015, 0.0,
                   c.in_h + c.sole_t / 2 + 0.002, math.pi), _all_ids())
    _step(240)
    report("rim-near-miss")
    s = scene._status()
    sl = loc_of(scene.shoes["shoe_a"])
    still_up_there = float(sl[2]) > c.in_h - 0.03  # did not fall into the cavity
    check("negative (RIM near-miss): a shoe balanced on the rim wall with its toe "
          "over the cavity is above the interior — not inside, no packed credit, "
          "no success",
          still_up_there and not bool(s["inside"][0, 0])
          and not bool(scene._packed[0, 0]) and not bool(scene.success()[0]))

    # ================= 10. crate-on-side pack =====================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    crate_on_side_at(0.75, 0.30)
    # tuck both shoes into the sideways cavity (resting on the -y wall, which is now
    # the floor of the sideways pocket): geometrically inside the interior box
    for i, nm in enumerate(scene.SHOE_NAMES):
        scene.shoes[nm].write_root_state_to_sim(
            crate_pose(-0.08 + 0.16 * i, -c.in_w / 2 + c.sole_w / 2 + 0.004, 0.05,
                       0.0), _all_ids())
        # note: crate-local pose; the shoe then falls under gravity inside the pocket
        _step(150)
    report("side-pack")
    s = scene._status()
    check("negative (CRATE ON SIDE): shoes tucked into the cavity of a crate lying "
          "on its side earn nothing — receptacle false, no packed credit, no success",
          not bool(s["receptacle"][0]) and not bool(scene._packed[0].any())
          and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.5)

    # ================= 11. uncover + erect only ===================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    crate_upright_at(0.70, -0.30)
    report("erect-only")
    sc = float(scene.score()[0])
    check("partial (uncover+erect only): crate upright at rest, shoes still on the "
          "floor — score ~0.30, no success",
          0.29 <= sc <= 0.31 and not bool(scene.success()[0]))

    # ================= 12. one shoe in ============================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    crate_upright_at(0.70, -0.30)
    drop_shoe_in("shoe_a", -0.08)
    report("one-shoe")
    sc = float(scene.score()[0])
    s = scene._status()
    check("partial (one shoe in): crate upright, shoe A inside and settled, B on "
          "the floor — score ~0.50, no success",
          bool(s["inside"][0, 0]) and 0.49 <= sc <= 0.51
          and not bool(scene.success()[0]))

    _AUDIT["on"] = False  # ---- check 13 constructs a REAL success on purpose ----

    # ================= 13. TIP-BACK: erection is judged live ======================================
    _REC["on"] = True
    drop_shoe_in("shoe_b", +0.08)
    _step(120)
    report("constructed-win")
    won = bool(scene.success()[0])
    sc_win = float(scene.score()[0])
    # tip the crate back onto its side (transport away; the shoes tumble out or stay
    # behind): the erection clause must collapse success while latches survive
    crate_on_side_at(0.75, 0.30)
    _step(60)
    report("tipped-back")
    _REC["on"] = False
    lost = not bool(scene.success()[0])
    sc_after = float(scene.score()[0])
    check("liveness (TIP-BACK): constructed success (upright crate, both shoes in) "
          "is live (score 1.0); put the crate back on its side and success "
          "COLLAPSES while the latched score survives at ~0.70",
          won and sc_win >= 0.99 and lost and 0.69 <= sc_after <= 0.71)

    # ================= 14. rejection audit ========================================================
    check("rejection audit: success() never fired during the negative probes "
          "(checks 4-12)", not _AUDIT["fired"])

    # ================= 15. video ==================================================================
    frames = _REC["frames"]
    if frames:
        np.savez_compressed(args.out, frames=np.stack(frames, axis=0))
        print(f"[smoke] saved {len(frames)} frames to {args.out}", flush=True)
    check("frames.npz saved with video frames", len(frames) > 0 and os.path.exists(args.out))

    # ================= verdict ====================================================================
    n_pass = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        if not ok:
            print(f"[smoke] FAILED: {name}", flush=True)
    if n_pass == len(checks):
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
        code = 0
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
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
    except BaseException:  # noqa: BLE001 — Kit keeps the process alive; die loudly
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(2)

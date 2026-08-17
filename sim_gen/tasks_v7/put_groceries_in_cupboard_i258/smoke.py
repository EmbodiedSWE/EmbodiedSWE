"""Smoke / rubric-REJECTION battery for BalanceShelfScene — NullRobot, teleported probes.

solve.py is the acceptance proof (force-evict the blue can, let the keel level
the shelf, drop the reds one per pocket, keel restores equilibrium). This
battery proves the rubric REJECTS wrong outcomes and that the BALANCE — the
task's strategic differentiator from the seed — is physically load-bearing:
there is no passive set-down that scores, a single can really pins the shelf,
and the blue can really vetoes level. Every probe is CONSTRUCTED as a settled
state (teleport, real physics steps, judge); constructed partial states may
earn latched partial credit but none may reach success() unless both reds
genuinely end seated one per pocket on a LEVEL beam with the blue can off it.

Checks:
   1. settle/no-NaN    — seeded reset settles finite; the decoy pins the shelf
                         at ITS stop (tilt sign = -side, magnitude ~stop_deg),
                         decoy seated, reds upright on the ground in front;
                         score ~0, no success;
   2. randomization    — two seeded resets: READBACK cupboard yaw, cupboard xy
                         and both red-can ground spots all differ;
   3. side shuffle     — across seeds the decoy pocket side takes BOTH values,
                         the stored side matches the beam-local decoy readback,
                         and the settled tilt sign follows the side;
   4. null-policy      — 240 idle steps: shelf stays pinned at the decoy's
                         stop, score ~0, no success;
   5. SEED STRATEGY    — the seed's whole plan ("set the groceries down in the
                         cupboard"): both reds set down passively (plinth floor
                         inside + roof) with the decoy untouched: rests there,
                         nothing latches, no success;
   6. wrong object     — a red seated in the free pocket with the DECOY still
                         in the other: the decoy out-torques the red, shelf
                         stays pinned, level False, no success, score <= first;
   7. single can       — decoy removed, ONE red seated: shelf pinned at that
                         stop — the near-miss of a policy that stops after one
                         grocery; no success, score <= 0.40;
   8. same-side stack  — both reds STACKED in one pocket: matched False (the
                         upper can is out of the seat band), still pinned, no
                         success;
   9. toppled bridger  — second red lying HORIZONTALLY across the raised
                         pocket's rims: its weight balances the beam LEVEL,
                         but seated rejects the pose (axis test) — the rubric
                         demands seated cans, not just torque; no success;
  10. latch regression — decoy stacked back aboard BEFORE the second red is
                         seated (so no prefix of the construction is a true
                         success): `both` latches (score 0.70) but the aboard
                         decoy vetoes success; removing the second red keeps
                         the latched 0.70 while success stays False;
  11. frames.npz       — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.put_groceries_in_cupboard_i258.smoke --headless
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

_qmul, _qz, _wrap_deg = task_scene._qmul, task_scene._qz, task_scene._wrap_deg

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

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
    t = float(scene.tilt_deg()[0])
    la = scene._beam_local(scene.can_a.data.root_pos_w)[0]
    lb = scene._beam_local(scene.can_b.data.root_pos_w)[0]
    print(f"[smoke] {tag:16s} | tilt={t:+7.2f}deg "
          f"A_beam=({float(la[0]):+.3f},{float(la[1]):+.3f},{float(la[2]):+.3f}) "
          f"B_beam=({float(lb[0]):+.3f},{float(lb[1]):+.3f},{float(lb[2]):+.3f}) "
          f"matched={bool(scene.matched()[0])} level={bool(scene.level()[0])} "
          f"decoy_aboard={bool(scene.aboard(scene.decoy)[0])} "
          f"settled={bool(scene.settled()[0])} score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.balance_shelf")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.45, -0.85, 0.90)) + o),
                                tuple(np.array((0.50, 0.00, 0.22)) + o),
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

    def tilt() -> float:
        _refresh()
        return float(scene.tilt_deg()[0])

    def put_beam_local(body, loc_xyz, extra_quat=None) -> None:
        """Teleport a body to a beam-local point, aligned to the CURRENT beam
        quat (optionally composed with extra_quat on the right)."""
        _refresh()
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        pos = scene.beam.data.root_pos_w \
            + quat_apply(scene.beam.data.root_quat_w, loc)
        q = scene.beam.data.root_quat_w
        if extra_quat is not None:
            q = _qmul(q, extra_quat.expand(n, 4))
        _write_body(body, pos, q)

    def put_cup_local(body, loc_xyz) -> None:
        _refresh()
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        pos = scene.cupboard.data.root_pos_w \
            + quat_apply(scene.cupboard.data.root_quat_w, loc)
        # cupboard quat is a pure yaw -> the body stays upright
        _write_body(body, pos, scene.cupboard.data.root_quat_w)

    def park_decoy() -> None:
        """Construction move: decoy to the open ground beside the cupboard."""
        _refresh()
        loc = torch.tensor([0.0, 0.55, 0.0], device=device).expand(n, 3)
        pos = scene.cupboard.data.root_pos_w \
            + quat_apply(scene.cupboard.data.root_quat_w, loc)
        pos = pos.clone()
        pos[:, 2] = env.iscene.env_origins[:, 2] + c.decoy_h / 2 + 0.003
        _write_body(body=scene.decoy, pos_w=pos, quat=scene.cupboard.data.root_quat_w)

    def seat_red(body, side: float, gap: float) -> None:
        """Construction move: drop a red can from `gap` above the pocket floor
        (beam-aligned, inside the fence rims for small gaps)."""
        put_beam_local(body, (0.0, side * c.well_y,
                              c.plate_top + gap + c.can_h / 2))

    def wait_level(max_steps: int = 900) -> None:
        for _ in range(max_steps):
            _step(1)
            _refresh()
            if abs(float(scene.tilt_deg()[0])) < 2.0 \
                    and float(scene.beam.data.root_ang_vel_w[0].norm()) < 0.05:
                break

    def wait_pinned(sign: float, max_steps: int = 600) -> None:
        for _ in range(max_steps):
            _step(1)
            _refresh()
            if sign * float(scene.tilt_deg()[0]) > c.stop_deg - 3.0 \
                    and float(scene.beam.data.root_ang_vel_w[0].norm()) < 0.05:
                break

    ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    side = float(scene.decoy_side[0])
    _REC["on"] = True
    _step(300)
    _report("settle")
    _REC["on"] = False
    up_a = float(quat_apply(scene.can_a.data.root_quat_w, ez)[0, 2])
    up_b = float(quat_apply(scene.can_b.data.root_quat_w, ez)[0, 2])
    za = float(scene.can_a.data.root_pos_w[0, 2] - env.iscene.env_origins[0, 2])
    zb = float(scene.can_b.data.root_pos_w[0, 2] - env.iscene.env_origins[0, 2])
    check("settle/no-NaN: layout settles finite; the decoy pins the shelf at "
          "ITS stop (tilt sign = -side, magnitude ~stop), decoy seated, reds "
          "upright on the ground; score ~0, no success",
          bool(scene._finite()[0])
          and -side * tilt() > c.stop_deg - 3.0
          and abs(tilt()) < c.stop_deg + 1.5
          and bool(scene.seated(scene.decoy, c.decoy_h, scene.decoy_side)[0])
          and up_a > 0.95 and up_b > 0.95
          and abs(za - c.can_h / 2) < 0.01 and abs(zb - c.can_h / 2) < 0.01
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        cyaw = yaw_of(scene.cupboard.data.root_quat_w[0])
        cp = scene.cupboard.data.root_pos_w[0, :2].clone()
        pa = scene.can_a.data.root_pos_w[0, :2].clone()
        pb = scene.can_b.data.root_pos_w[0, :2].clone()
        return cyaw, cp, pa, pb

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_cp, a_pa, a_pb = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_cp, b_pa, b_pb = readback()
    d_yawv = dyaw(a_yaw, b_yaw)
    d_cp = float((a_cp - b_cp).norm())
    d_pa = float((a_pa - b_pa).norm())
    d_pb = float((a_pb - b_pb).norm())
    print(f"[smoke] randomization deltas: cup_yaw={d_yawv:.1f}deg "
          f"cup_xy={d_cp * 1000:.1f}mm canA_xy={d_pa * 1000:.1f}mm "
          f"canB_xy={d_pb * 1000:.1f}mm", flush=True)
    check("randomization-is-real: cupboard yaw, cupboard xy and both red-can "
          "ground spots readback all differ across seeds",
          d_yawv > 1.5 and d_cp > 0.003 and d_pa > 0.03 and d_pb > 0.03)

    # ================= 3. randomization: decoy side shuffle =======================================
    sides = set()
    consistent = True
    for sd in range(300, 310):
        torch.manual_seed(sd)
        env.reset()
        _refresh()
        s = float(scene.decoy_side[0])
        sides.add(s)
        # stored side matches the beam-local decoy readback at spawn
        ly = float(scene._beam_local(scene.decoy.data.root_pos_w)[0, 1])
        consistent &= abs(ly - s * c.well_y) < 0.01
        # ...and the settled tilt sign follows the side
        _step(150)
        consistent &= -s * tilt() > c.stop_deg - 4.0
        consistent &= bool(scene.seated(scene.decoy, c.decoy_h, scene.decoy_side)[0])
    print(f"[smoke] side shuffle over seeds 300-309: sides={sorted(sides)} "
          f"consistent={consistent}", flush=True)
    check("randomization (side shuffle): the decoy pocket side takes both "
          "values across seeds, matches the beam-local readback, and the "
          "settled tilt sign follows it",
          sides == {1.0, -1.0} and consistent)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    side = float(scene.decoy_side[0])
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, the shelf stays pinned at the "
          "decoy's stop, reds stay on the ground, score ~0, no success",
          -side * tilt() > c.stop_deg - 3.0
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 5. negative: the SEED'S OWN STRATEGY =======================================
    # "Put the groceries in the cupboard" seed-style: set them down on whatever
    # passive surface the cupboard offers, decoy untouched. The only passive
    # surfaces are the plinth floor inside and the roof — construct both rest
    # states; nothing latches, no success.
    torch.manual_seed(100)
    env.reset()
    _step(120)
    _REC["on"] = True
    put_cup_local(scene.can_a, (0.05, 0.10, c.plinth_h + c.can_h / 2 + 0.003))
    put_cup_local(scene.can_b, (0.00, -0.10, c.roof_z1 + c.can_h / 2 + 0.003))
    _step(300)
    _report("seed-setdown")
    _REC["on"] = False
    la = scene._beam_local(scene.can_a.data.root_pos_w)[0]
    lb = scene._beam_local(scene.can_b.data.root_pos_w)[0]
    check("negative (SEED strategy): both reds set down passively — one on the "
          "plinth floor inside, one on the roof — rest there; neither is "
          "seated or aboard, the pinned shelf never levels, score ~0, no "
          "success",
          bool(scene._finite()[0])
          and float(la[2]) < -0.05 and float(lb[2]) > 0.20
          and not bool(scene.any_seated()[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 6. negative: WRONG OBJECT in the pocket ====================================
    # A red seated in the free pocket while the DECOY sits in the other: the
    # decoy out-torques the red head-to-head, so the shelf stays pinned at the
    # decoy's stop. level must stay False and only `first` may latch.
    torch.manual_seed(100)
    env.reset()
    side = float(scene.decoy_side[0])
    _step(300)  # shelf pinned at the decoy's stop
    _REC["on"] = True
    seat_red(scene.can_a, -side, 0.008)  # into the RAISED pocket, 8 mm drop
    _step(420)
    _report("wrong-object")
    _REC["on"] = False
    check("negative (wrong object): red seated in the free pocket, decoy still "
          "in the other — the heavier decoy keeps the shelf pinned, level "
          "False, no success, score <= first-credit",
          bool(scene.seated(scene.can_a, c.can_h,
                            torch.full((n,), -side, device=device))[0])
          and -side * tilt() > c.stop_deg - 3.0
          and not bool(scene.level()[0])
          and not bool(scene.success()[0])
          and float(scene.score()[0]) <= c.w_first + 1e-3)

    # ================= 7. near-miss: SINGLE red can ===============================================
    # Decoy parked on the ground (construction), keel levels the shelf, then
    # ONE red seated: its weight pins the shelf at the stop — the natural
    # stopping point of a policy that stows only one grocery.
    torch.manual_seed(100)
    env.reset()
    _step(300)
    park_decoy()
    wait_level()
    _report("evicted-level")
    assert bool(scene._unloaded[0]), "construction: unloaded should latch"
    _REC["on"] = True
    seat_red(scene.can_a, +1.0, 0.035)
    wait_pinned(-1.0)
    _step(60)
    _report("single-can")
    _REC["on"] = False
    s7 = float(scene.score()[0])
    check("near-miss (single can): one red seated on the empty shelf pins it "
          "at the stop — seated holds, level fails, no success, score <= 0.40",
          bool(scene.seated(scene.can_a, c.can_h, +1.0)[0])
          and tilt() < -(c.stop_deg - 3.0)
          and not bool(scene.level()[0])
          and not bool(scene.success()[0]) and s7 <= 0.4005)

    # ================= 8. negative: both reds STACKED in one pocket ===============================
    # Continue: drop red B centred ON TOP of red A (same pocket). matched needs
    # one can per pocket; the upper can is far out of the seat z band.
    _REC["on"] = True
    put_beam_local(scene.can_b, (0.0, c.well_y,
                                 c.plate_top + c.can_h + 0.008 + c.can_h / 2))
    _step(420)
    _report("stacked")
    _REC["on"] = False
    check("negative (same-side stack): both reds stacked in ONE pocket — "
          "matched False (upper can out of the seat band), shelf still "
          "pinned, no success, score <= 0.40",
          not bool(scene.matched()[0])
          and not bool(scene.seated(scene.can_b, c.can_h, +1.0)[0])
          and not bool(scene.level()[0])
          and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.4005)

    # ================= 9. negative: toppled can bridging the raised pocket ========================
    # Fresh construction to the single-can state, then red B lying HORIZONTALLY
    # across the raised pocket's fence rims: its weight BALANCES the beam level
    # — pure torque parity — but `seated` rejects the pose (axis test). The
    # rubric demands seated groceries, not just equilibrium.
    torch.manual_seed(100)
    env.reset()
    _step(300)
    park_decoy()
    wait_level()
    seat_red(scene.can_a, +1.0, 0.035)
    wait_pinned(-1.0)
    _step(60)
    _REC["on"] = True
    qx90 = torch.tensor([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0],
                        device=device)  # axis -> beam-local +/-y: lies across x-fences
    put_beam_local(scene.can_b,
                   (0.0, -c.well_y, c.fence_h + c.can_r + 0.004), qx90)
    _step(600)
    _report("toppled-bridge")
    _REC["on"] = False
    axis_b = scene._beam_local(scene.can_b.data.root_pos_w
                               + quat_apply(scene.can_b.data.root_quat_w,
                                            ez) * 0.01)[0] \
        - scene._beam_local(scene.can_b.data.root_pos_w)[0]
    check("negative (toppled bridger): red B lying across the raised pocket "
          "rims levels the beam by pure torque parity, but seated rejects the "
          "lying pose — matched False, no success",
          abs(float(axis_b[2])) < 0.006
          and not bool(scene.seated(scene.can_b, c.can_h,
                                    torch.full((n,), -1.0, device=device))[0])
          and not bool(scene.matched()[0])
          and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.4005)

    # ================= 10. latch regression + aboard-decoy veto ===================================
    # Construction ordered so NO prefix is a true success: decoy comes back
    # ABOARD (stacked on red A) BEFORE red B is seated. matched then latches
    # `both` (score 0.70) but the aboard decoy vetoes success; removing red B
    # afterwards keeps the latched 0.70 while success stays False.
    torch.manual_seed(100)
    env.reset()
    _step(300)
    park_decoy()
    wait_level()
    seat_red(scene.can_a, +1.0, 0.035)
    wait_pinned(-1.0)
    _step(60)
    put_beam_local(scene.decoy, (0.0, c.well_y,
                                 c.plate_top + c.can_h + 0.008 + c.decoy_h / 2))
    _step(240)
    _refresh()
    assert bool(scene.aboard(scene.decoy)[0]), "construction: decoy must be aboard"
    _REC["on"] = True
    seat_red(scene.can_b, -1.0, 0.008)
    _step(420)
    _report("veto-aboard")
    s10a = float(scene.score()[0])
    both_latched = bool(scene._both[0])
    veto_ok = bool(scene.matched()[0]) and not bool(scene.success()[0]) \
        and abs(s10a - 0.70) < 1e-3
    # remove red B to the open ground: latched credit is memory, success is live
    put_cup_local(scene.can_b, (0.0, -0.55, c.can_h / 2 + 0.003))
    _step(120)
    _report("latch-regress")
    _REC["on"] = False
    s10b = float(scene.score()[0])
    check("latch-regression + veto: with the decoy stacked back aboard, "
          "seating the second red latches `both` (0.70) but success is vetoed "
          "by the aboard decoy; removing the red keeps the latched 0.70, "
          "success stays False",
          both_latched and veto_ok and abs(s10b - 0.70) < 1e-3
          and not bool(scene.matched()[0]) and not bool(scene.success()[0]))

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.balance_shelf")
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
    except Exception as exc:  # noqa: BLE001 - Kit teardown hangs; die loudly NOW
        print(f"SIM_GEN_SMOKE: FAIL (exception: {exc!r})", flush=True)
        import traceback

        traceback.print_exc()
        os._exit(1)

"""Smoke / rubric-REJECTION battery for FlapChutePantryScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the latched
credit is monotone along a real push-through trajectory, three seeds). This battery
proves the rubric REJECTS wrong outcomes, and that the mechanism claims the task rests
on — the flap yields inward and gravity-recloses, the flap cannot open outward, an
interned carton is sealed in — are physics, not fiat. Every probe is CONSTRUCTED as a
settled state (teleport, real physics steps, judge) — instrumentation, never a
solution.

Checks:
   1. settle/no-NaN      — seeded reset settles finite; flap hanging closed, both
                           cartons on the ground outside; score ~0, no success;
   2. randomization      — two seeded resets: READBACK pantry yaw, pantry xy and the
                           cream carton's pose all differ;
   3. slot swap          — over 10 resets the cream carton occupies BOTH start slots;
   4. null-policy        — 240 idle steps: flap stays closed, score ~0, no success;
   5. flap yields+returns— 0.3 N inward push swings the flap open past 30 deg;
                           released, it falls SHUT again by gravity alone;
   6. flap one-way       — 0.6 N OUTWARD push for 1 s: the flap's margins press
                           against the wall from behind and it never opens outward;
   7. SEAL INTERLOCK     — cream carton CONSTRUCTED inside on the chute, shoved
                           toward the doorway at 3x its own weight for 1.5 s: it
                           never leaves the chamber ("once inside, sealed in" is
                           load-bearing physics);
   8. SEED STRATEGY      — the seed's plan ("lower it into the open-top container
                           from above"): cream carton settled on the pantry ROOF ->
                           no success, score ~0 (the roof is closed; there is no
                           open top);
   9. apron-only         — cream carton staged on the apron and abandoned -> the
                           staged latch alone (0.15), no success;
  10. doorway wedge      — cream carton wedged in the doorway holding the flap ~55
                           deg open (the failure mode observed during development)
                           -> flap not closed, no success, score < 0.70;
  11. wrong object       — BROWN carton inside, cream left outside -> no success;
  12. both inside        — cream deep inside AND brown inside, flap closed: only the
                           brown clause fails -> no success;
  13. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.<task>.smoke --headless
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

_qmul, _qy = task_scene._qmul, task_scene._qy

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
    cl = scene._pantry_local(scene.cream.data.root_pos_w)[0]
    print(f"[smoke] {tag:16s} | cream_loc=({float(cl[0]):+.3f},{float(cl[1]):+.3f},"
          f"{float(cl[2]):+.3f}) flap={float(scene.flap_deg()[0]):+.1f}deg "
          f"inside={bool(scene.cream_inside()[0])} deep={bool(scene.cream_deep()[0])} "
          f"brown_in={bool(scene.brown_inside()[0])} "
          f"closed={bool(scene.flap_closed()[0])} "
          f"staged={bool(scene._staged[0])} breach={bool(scene._breach[0])} "
          f"in_latch={bool(scene._inside[0])} score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


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
    env = ENVS.get("simgen.flap_chute_pantry")().build(num_envs=args.num_envs,
                                                       device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.55, -0.70, 0.55)) + o),
                                tuple(np.array((0.30, -0.04, 0.10)) + o),
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

    def pantry_world(loc_xyz) -> torch.Tensor:
        """Pantry-local point -> world (per-env)."""
        _refresh()
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.pantry.data.root_pos_w + quat_apply(scene.pantry.data.root_quat_w,
                                                         loc)

    def pantry_quat(q_extra: torch.Tensor | None = None) -> torch.Tensor:
        _refresh()
        q = scene.pantry.data.root_quat_w
        return q if q_extra is None else _qmul(q, q_extra)

    def door_out_w() -> torch.Tensor:
        """World unit vector pointing OUT of the doorway (pantry-local +x)."""
        _refresh()
        return quat_apply(scene.pantry.data.root_quat_w,
                          torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))[0]

    def cream_deep_inside() -> None:
        """CONSTRUCT the cream carton settled deep on the chute (the goal region)."""
        _write_body(scene.cream, pantry_world((-0.090, 0.0, 0.050)), pantry_quat())
        _step(120)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    check("settle/no-NaN: layout settles finite; flap hanging closed, both cartons "
          "on the ground outside; score ~0, no success",
          bool(scene._finite()[0]) and bool(scene.flap_closed()[0])
          and not bool(scene.cream_inside()[0]) and not bool(scene.brown_inside()[0])
          and not bool(scene.on_apron(scene.cream.data.root_pos_w)[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (yaw_of(scene.pantry.data.root_quat_w[0]),
                scene.pantry.data.root_pos_w[0, :2].clone(),
                scene.cream.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.cream.data.root_quat_w[0]))

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_pp, a_cp, a_cy = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_pp, b_cp, b_cy = readback()
    d_yawv, d_pp = dyaw(a_yaw, b_yaw), float((a_pp - b_pp).norm())
    d_cp, d_cy = float((a_cp - b_cp).norm()), dyaw(a_cy, b_cy)
    print(f"[smoke] randomization deltas: pantry_yaw={d_yawv:.1f}deg "
          f"pantry_xy={d_pp * 1000:.1f}mm cream_xy={d_cp * 1000:.1f}mm "
          f"cream_yaw={d_cy:.1f}deg", flush=True)
    check("randomization-is-real: pantry yaw, pantry xy and the cream carton's pose "
          "readback differ across seeds",
          d_yawv > 2.0 and d_pp > 0.003 and d_cp > 0.005 and d_cy > 2.0)

    # ================= 3. carton slot swap ========================================================
    sides = set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        sides.add("+" if float(scene.cream_slot[0]) > 0 else "-")
    print(f"[smoke] over 10 resets: cream carton slots {sorted(sides)}", flush=True)
    check("slot swap: the cream carton occupies BOTH start slots over 10 resets",
          sides == {"+", "-"})

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, flap stays closed, score ~0, no success",
          bool(scene.flap_closed()[0]) and float(scene.score()[0]) <= 0.05
          and not bool(scene.success()[0]))

    # ================= 5. the flap yields inward and gravity-recloses =============================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    _push(scene.flap, -0.3 * door_out_w(), 60)
    _refresh()
    opened = float(scene.flap_deg()[0])
    _step(240)
    _report("flap-return")
    reclosed = float(scene.flap_deg()[0])
    print(f"[smoke] flap swing: opened to {opened:+.1f}deg under 0.3 N, "
          f"reclosed to {reclosed:+.1f}deg after release", flush=True)
    check("flap mechanism: a 0.3 N inward push swings the flap open (>30deg) and it "
          "falls SHUT again by gravity alone",
          opened > 30.0 and bool(scene.flap_closed()[0]))
    _REC["on"] = False

    # ================= 6. the flap is ONE-WAY: it cannot open outward =============================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    min_deg = 0.0
    zero = torch.zeros(n, 1, 3, device=device)
    f_out = (0.6 * door_out_w()).view(1, 1, 3).expand(n, 1, 3).contiguous()
    for _ in range(120):
        scene.flap.set_external_force_and_torque(f_out, zero, env_ids=_all_ids(),
                                                 is_global=True)
        _step(1)
        min_deg = min(min_deg, float(scene.flap_deg()[0]))
    scene.flap.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
    _step(120)
    _report("flap-one-way")
    print(f"[smoke] outward push: most-negative flap angle {min_deg:+.1f}deg "
          f"(margins press against the wall)", flush=True)
    check("flap one-way: a 0.6 N OUTWARD push for 1 s never swings the flap outward "
          "(angle stays above -10deg) and it hangs closed after",
          min_deg > -10.0 and bool(scene.flap_closed()[0]))

    # ================= 7. SEAL INTERLOCK: an interned carton cannot be raided out =================
    # "once inside, sealed in for good": construct the goal state, then shove the
    # cream carton toward the doorway at 3x its own weight for 1.5 s — it must never
    # leave the chamber (the flap's margins are the physical stop). This is the
    # physics behind the strategic claim; note the constructed state itself is the
    # GOAL region, so success beforehand is expected — the check is that the shove
    # cannot undo it.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    cream_deep_inside()
    _refresh()
    assert bool(scene.cream_inside()[0]), "probe setup: carton must rest inside"
    f = 3.0 * c.carton_mass * 9.81 * door_out_w()
    escaped = False
    for _ in range(9):  # 9 x 20 = 180 steps = 1.5 s of shoving
        _push(scene.cream, f, 20)
        _refresh()
        escaped |= not bool(scene.cream_inside()[0])
    _step(120)
    _report("seal-interlock")
    xl = float(scene._pantry_local(scene.cream.data.root_pos_w)[0, 0])
    print(f"[smoke] interlock shove: cream pantry-frame x={xl * 1000:.0f}mm "
          f"(front wall inner face {c.int_half * 1000:.0f}mm)", flush=True)
    check("SEAL INTERLOCK: 3x-weight shove toward the doorway for 1.5 s never gets "
          "the interned carton out of the chamber",
          not escaped and bool(scene.cream_inside()[0]))
    _REC["on"] = False

    # ================= 8. negative: the SEED'S OWN STRATEGY =======================================
    # The seed's plan is "carry it over and lower it into the open-top container".
    # This pantry has NO open top — the plan's end state is the carton resting on the
    # closed ROOF. Must not be success and must score ~0.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _write_body(scene.cream,
                pantry_world((0.0, 0.0, c.int_h + 0.012 + c.carton_size[2] / 2 + 0.005)),
                pantry_quat())
    _step(120)
    _report("seed-strategy")
    check("negative (SEED strategy): cream carton lowered-from-above ends ON THE "
          "CLOSED ROOF — no success, score ~0",
          not bool(scene.cream_inside()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.05)
    _REC["on"] = False

    # ================= 9. near-miss: staged on the apron and abandoned ============================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _write_body(scene.cream,
                pantry_world((0.242, 0.0, c.sill_z + c.carton_size[2] / 2 + 0.010)),
                pantry_quat())
    _step(120)
    _report("apron-only")
    check("near-miss (apron-only): cream carton staged on the apron and abandoned — "
          "staged latch only (score 0.15), no success",
          bool(scene.on_apron(scene.cream.data.root_pos_w)[0])
          and abs(float(scene.score()[0]) - c.w_staged) < 0.01
          and not bool(scene.success()[0]))

    # ================= 10. near-miss: wedged in the doorway, flap held open =======================
    # The failure mode observed during development: the carton straddling the sill,
    # the flap resting ~55deg open on top of it. Construct it directly (flap opened,
    # carton in the throat, settle) — stable, and must NOT be success.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _write_body(scene.flap, pantry_world((c.hinge_x, 0.0, c.hinge_z)),
                pantry_quat(_qy(torch.full((n,), math.radians(58.0), device=device))))
    _write_body(scene.cream, pantry_world((0.118, 0.0, 0.076)), pantry_quat())
    _step(240)
    _report("doorway-wedge")
    wedge_deg = float(scene.flap_deg()[0])
    print(f"[smoke] wedge state: flap held at {wedge_deg:+.1f}deg by the carton",
          flush=True)
    check("near-miss (doorway wedge): carton stuck in the doorway holding the flap "
          "open — flap not closed, no success, score < 0.70",
          not bool(scene.flap_closed()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) < 0.70)
    _REC["on"] = False

    # ================= 11. negative: wrong object =================================================
    # The BROWN decoy interned instead, the cream carton left on the ground: a
    # complete, tidy push-through episode — of the wrong carton. Must not be success.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _write_body(scene.brown, pantry_world((-0.090, 0.0, 0.050)), pantry_quat())
    _step(120)
    _report("wrong-object")
    check("negative (wrong object): BROWN carton inside, cream left outside — "
          "no success",
          bool(scene.brown_inside()[0]) and not bool(scene.cream_inside()[0])
          and not bool(scene.success()[0]))

    # ================= 12. negative: both cartons inside ==========================================
    # Cream deep in the goal region AND the brown decoy pushed in too, flap closed:
    # every clause but ~brown_inside holds — the decoy clause alone must refuse.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    cream_deep_inside()
    _write_body(scene.brown, pantry_world((-0.040, 0.060, 0.060)), pantry_quat())
    _step(180)
    _report("both-inside")
    check("negative (both inside): cream deep inside AND brown inside, flap closed — "
          "the brown clause refuses, no success",
          bool(scene.cream_deep()[0]) and bool(scene.brown_inside()[0])
          and bool(scene.flap_closed()[0]) and not bool(scene.success()[0]))

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.flap_chute_pantry")
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

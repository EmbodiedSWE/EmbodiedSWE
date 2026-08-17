"""Smoke / rubric-REJECTION battery for TotemTunnelScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the latched
credit is monotone along a real trajectory). This battery proves the rubric REJECTS
wrong outcomes, and that the geometric claim the task rests on — the tunnel admits the
totem ONLY lying lengthwise — is physically load-bearing. Every probe is CONSTRUCTED
(teleport, real physics steps, judge) — instrumentation, never a solution: no probe
here reaches success().

On granted latches (checks 6-8): the transit latch chain is a trajectory property
(did the body thread the bore?), which solve.py proves fires honestly along the real
slide. To test the OTHER success clauses in isolation, those probes GRANT the transit
latches directly and then construct a wrong end state — showing that even a solver
credited with a perfect transit still fails on a bad erection.

Checks:
   1. settle/no-NaN      — seeded reset settles finite; totem standing inside the pen;
                           score ~0, no success;
   2. randomization      — THREE seeded resets: max-pairwise readback deltas of pen
                           yaw, pen xy, totem pen-local xy, disc pen-local xy all real;
   3. null-policy        — 240 idle steps: score ~0, no success (totem just stands);
   4. CROSSWISE CANNOT   — totem laid CROSSWISE at the tunnel mouth and pushed at the
      PASS                 aperture with the standard force for 500 steps: it MOVES
                           (the probe is non-vacuous) but is walled at the front face
                           — never mid-bore, never through ("only lengthwise" is
                           physics, not fiat);
   5. SEED-analog        — the totem CARRIED OVER THE WALLS (high waypoints, the
      over-the-wall        seed's grasp-lift-place plan) and stood perfectly on the
      carry                goal disc: end state IDENTICAL to success except history —
                           on_pad_upright holds, yet no transit chain -> no success,
                           score ~0;
   6. lying at the disc  — transit GRANTED, totem lying across the disc: the upright
                           clause alone refuses -> no success;
   7. sloppy erection    — transit GRANTED, totem released at the disc tilted 30 deg
                           (past the ~20 deg critical angle): it falls over -> no
                           success;
   8. upright off-disc   — transit GRANTED, totem standing on the ground 15 cm from
                           the disc: the placement clauses refuse -> no success;
   9. mid-bore teleport  — fresh reset, totem teleported directly INTO the bore
                           (fits: 40 mm body in the 66 x 56 mm aperture): without the
                           entered link the chain stays cold — no mid/through credit;
  10. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.registry_i373.smoke --headless
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

_qmul, _qz = task_scene._qmul, task_scene._qz

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
    loc = scene._pen_local(scene.totem.data.root_pos_w)
    print(f"[smoke] {tag:18s} | loc=({float(loc[0, 0]):+.3f},{float(loc[0, 1]):+.3f},"
          f"{float(loc[0, 2]):+.3f}) lying={bool(scene.lying()[0])} "
          f"upright={bool(scene.upright()[0])} ent={bool(scene._entered[0])} "
          f"mid={bool(scene._mid[0])} thr={bool(scene._through[0])} "
          f"prog={float(scene._prog[0]):.2f} onpad={bool(scene.on_pad_upright()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.totem_tunnel")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    from isaaclab.utils.math import quat_apply_inverse

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.30, -0.95, 0.85)) + o),
                                tuple(np.array((0.32, 0.00, 0.06)) + o),
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

    def pen_world(lx: float, ly: float, z: float) -> torch.Tensor:
        _refresh()
        cy, sy = torch.cos(scene.pen_yaw), torch.sin(scene.pen_yaw)
        pos = torch.zeros(n, 3, device=device)
        pos[:, 0] = scene.pen_pos_w[:, 0] + cy * lx - sy * ly
        pos[:, 1] = scene.pen_pos_w[:, 1] + sy * lx + cy * ly
        pos[:, 2] = scene.pen_pos_w[:, 2] + z
        return pos

    def qpen() -> torch.Tensor:
        _refresh()
        return _qz(scene.pen_yaw)

    def qy(deg: float) -> torch.Tensor:
        half = math.radians(deg) / 2
        q = torch.zeros(n, 4, device=device)
        q[:, 0], q[:, 2] = math.cos(half), math.sin(half)
        return q

    q_lie_x = lambda: _qmul(qpen(), qy(90.0))                        # noqa: E731  axis -> pen +x
    q_lie_y = lambda: _qmul(_qmul(qpen(), _qz(                        # noqa: E731  axis -> pen +y
        torch.full((n,), math.pi / 2, device=device))), qy(90.0))

    def grant_transit() -> None:
        """Instrumentation: credit a perfect transit (see module docstring)."""
        scene._entered[:] = True
        scene._mid[:] = True
        scene._through[:] = True

    def pad_local():
        _refresh()
        return scene._pen_local(scene.pad_pos_w)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    loc = scene._pen_local(scene.totem.data.root_pos_w)
    check("settle/no-NaN: layout settles finite; totem standing inside the pen; "
          "score 0, no success",
          bool(torch.isfinite(scene.totem.data.root_pos_w).all())
          and bool(scene.upright()[0])
          and bool((loc[0, :2].abs() < c.interior_half).all())
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (3-seed max-pairwise readback) ===================
    def readback():
        _refresh()
        t_loc = scene._pen_local(scene.totem.data.root_pos_w)[0, :2].clone()
        p_loc = scene._pen_local(scene.pad_pos_w)[0, :2].clone()
        return (float(scene.pen_yaw[0]),
                (scene.pen_pos_w - env.iscene.env_origins)[0, :2].clone(), t_loc, p_loc)

    obs = []
    for s in (101, 202, 303):
        torch.manual_seed(s)
        env.reset()
        _step(10)
        obs.append(readback())

    def maxpair(fn) -> float:
        best = 0.0
        for i in range(3):
            for j in range(i + 1, 3):
                best = max(best, fn(obs[i], obs[j]))
        return best

    d_yaw = maxpair(lambda a, b: abs(math.degrees(
        (a[0] - b[0] + math.pi) % (2 * math.pi) - math.pi)))
    d_pen = maxpair(lambda a, b: float((a[1] - b[1]).norm()))
    d_tot = maxpair(lambda a, b: float((a[2] - b[2]).norm()))
    d_pad = maxpair(lambda a, b: float((a[3] - b[3]).norm()))
    print(f"[smoke] randomization max-pairwise deltas: pen_yaw={d_yaw:.1f}deg "
          f"pen_xy={d_pen * 1000:.1f}mm totem_local={d_tot * 1000:.1f}mm "
          f"pad_local={d_pad * 1000:.1f}mm", flush=True)
    check("randomization-is-real: pen yaw, pen xy, totem local xy, disc local xy "
          "readback deltas over 3 seeds",
          d_yaw > 5.0 and d_pen > 0.003 and d_tot > 0.003 and d_pad > 0.003)

    # ================= 3. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # ================= 4. CROSSWISE CANNOT PASS (the aperture claim is physics) ===================
    # Totem laid CROSSWISE (long axis along pen y) at the tunnel mouth, then pushed at
    # the aperture with MORE force than the solve's slide servo ever applies, for 500
    # steps. The 110 mm body vs the 66 mm aperture: it must be walled at the front
    # face. The probe asserts the totem MOVED (non-vacuous) yet never entered the bore.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _write_body(scene.totem, pen_world(0.055, 0.0, c.totem_a / 2 + 0.003), q_lie_y())
    _step(30)
    _refresh()
    x0 = float(scene._pen_local(scene.totem.data.root_pos_w)[0, 0])
    yaw0 = scene.pen_yaw
    ex_w = torch.stack([torch.cos(yaw0), torch.sin(yaw0), torch.zeros_like(yaw0)], dim=-1)
    all_ids = _all_ids()
    for _i in range(500):
        q = scene.totem.data.root_quat_w
        fb = quat_apply_inverse(q, ex_w * 1.6)
        scene.totem.set_external_force_and_torque(
            fb.unsqueeze(1), torch.zeros(n, 1, 3, device=device), env_ids=all_ids)
        _step(1)
    z3 = torch.zeros(n, 1, 3, device=device)
    scene.totem.set_external_force_and_torque(z3, z3, env_ids=all_ids)
    _step(90)
    _report("crosswise-push")
    _REC["on"] = False
    x1 = float(scene._pen_local(scene.totem.data.root_pos_w)[0, 0])
    print(f"[smoke] crosswise push: x_l {x0:+.3f} -> {x1:+.3f} "
          f"(front wall inner face at {c.interior_half:.3f})", flush=True)
    check("CROSSWISE CANNOT PASS: pushed hard at the aperture the crosswise totem "
          "moves (probe non-vacuous) but is walled at the front face — never "
          "mid-bore, never through",
          (x1 - x0) > 0.015 and x1 < c.interior_half - 0.005
          and not bool(scene._mid[0]) and not bool(scene._through[0])
          and not bool(scene.success()[0]))

    # ================= 5. SEED-analog: carry over the walls =======================================
    # The seed registry's whole skill is grasp-lift-place through free space. CONSTRUCT
    # exactly that: lift the standing totem high over the walls (waypoints well above
    # wall height), set it down PERFECTLY upright on the goal disc, settle. The end
    # state is identical to success in every live clause — yet the tunnel was never
    # threaded, so the rubric must refuse and pay ~0.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    pl = pad_local()
    px, py = float(pl[0, 0]), float(pl[0, 1])
    for lx, ly, z in ((0.0, 0.0, 0.30), (0.15, py / 2, 0.30), (px, py, 0.30)):
        _write_body(scene.totem, pen_world(lx, ly, z), qpen())
        _step(2)
    _write_body(scene.totem,
                pen_world(px, py, c.pad_t + c.totem_h / 2 + 0.008), qpen())
    _step(240)
    _report("over-the-wall")
    _REC["on"] = False
    check("SEED analog (over-the-wall carry): totem stood perfectly upright on the "
          "goal disc without threading the tunnel — on_pad holds, no transit chain, "
          "no success, score ~0",
          bool(scene.on_pad_upright()[0]) and not bool(scene._through[0])
          and not bool(scene._entered[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.02)

    # ================= 6. transit granted, totem LYING at the disc ================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    grant_transit()
    pl = pad_local()
    _write_body(scene.totem,
                pen_world(float(pl[0, 0]), float(pl[0, 1]),
                          c.pad_t + c.totem_a / 2 + 0.004), q_lie_x())
    _step(150)
    _report("lying-at-disc")
    check("negative (upright clause): transit granted, totem lying across the disc "
          "— no success, score <= cap",
          bool(scene._through[0]) and not bool(scene.on_pad_upright()[0])
          and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.70 + 1e-6)

    # ================= 7. transit granted, sloppy 30-deg erection =================================
    # Released tilted 30 deg — past the atan(a/h) ~= 20 deg critical angle — the totem
    # MUST fall over: "stand it upright" tolerates only a real, seated erection.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    grant_transit()
    pl = pad_local()
    _write_body(scene.totem,
                pen_world(float(pl[0, 0]), float(pl[0, 1]),
                          c.pad_t + c.totem_h / 2 + 0.006),
                _qmul(qpen(), qy(30.0)))
    _step(240)
    _report("sloppy-erect")
    _REC["on"] = False
    check("negative (sloppy erection): transit granted, totem released 30 deg "
          "tilted at the disc — it topples, no success",
          bool(scene._through[0]) and not bool(scene.upright()[0])
          and not bool(scene.success()[0]))

    # ================= 8. transit granted, upright but OFF the disc ===============================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    grant_transit()
    pl = pad_local()
    _write_body(scene.totem,
                pen_world(float(pl[0, 0]), float(pl[0, 1]) + 0.15,
                          c.totem_h / 2 + 0.004), qpen())
    _step(150)
    _report("off-disc")
    check("negative (placement): transit granted, totem standing on the ground "
          "15 cm from the disc — placement clauses refuse, no success",
          bool(scene._through[0]) and bool(scene.upright()[0])
          and not bool(scene.on_pad_upright()[0]) and not bool(scene.success()[0]))

    # ================= 9. mid-bore teleport: the chain needs its first link =======================
    # Fresh reset (cold latches). The lying-aligned totem FITS mid-bore (40 mm body in
    # the 66 x 56 mm aperture — a feasible construct, no depenetration), but without
    # ever visiting the entered band the chain must stay cold: no mid/through credit.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _write_body(scene.totem,
                pen_world((c.mid_lo + c.mid_hi) / 2, 0.0, c.totem_a / 2 + 0.003),
                q_lie_x())
    _step(120)
    _report("mid-bore-cold")
    loc = scene._pen_local(scene.totem.data.root_pos_w)
    check("order chain: totem teleported directly into the bore (fits, settles in "
          "place) — entered never fired, so mid/through stay cold, score ~0",
          bool(torch.isfinite(scene.totem.data.root_pos_w).all())
          and abs(float(loc[0, 0]) - (c.mid_lo + c.mid_hi) / 2) < 0.03
          and not bool(scene._entered[0]) and not bool(scene._mid[0])
          and not bool(scene._through[0]) and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.totem_tunnel")
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
        os._exit(1)

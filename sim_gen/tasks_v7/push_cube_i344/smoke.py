"""Smoke / rubric-REJECTION battery for BalanceTriageScene — NullRobot, teleported
probes.

solve.py is the acceptance proof (weigh on the balance, then deliver by verdict).
This battery proves the rubric REJECTS every wrong outcome — above all the seed's
own strategy transplanted here: just MOVE the cubes into goal regions without ever
measuring. Every probe is CONSTRUCTED (teleport, real physics steps, judge) —
instrumentation, never a solution: success() is monitored at EVERY step and must
never turn True anywhere in the battery (the audit is itself a check).

Checks:
   1. settle/no-NaN      — seeded reset settles finite; beam level, cubes at their
                           floor spots (not on pans), score 0, no success;
   2. randomization      — three seeded resets: READBACK hidden masses, the
                           heavy-cube spawn SIDE (the coin flip), bin xy+yaw and
                           pedestal yaw; max-pairwise deltas real;
   3. null-policy        — 240 idle steps: score ~0, no success;
   4. BLIND SORT (seed)  — the seed's own plan (move objects into goal regions):
                           both cubes hover-dropped into the CORRECT bins, settled —
                           the live end state matches success's placement clause
                           exactly, but no weighing ever happened -> no success,
                           score ~0 (bin credit is gated on the weighing);
   5. single-cube tilt   — heavy cube alone on pan A tips the beam to the stop and
                           holds it there calmly, far longer than the streak — a
                           one-pan tilt is NOT a weighing: `weighed` stays False;
   6. one-pan pile       — both cubes piled on the SAME pan (light dropped onto the
                           heavy's back): beam hard at the stop, but never opposite
                           pans -> `weighed` stays False, no success;
   7. wrong bins         — a REAL weighing (latch fires), then the cubes delivered
                           SWAPPED (heavy->blue, light->red), settled: no bin latch,
                           no success, score stays at the weigh credit 0.45;
   8. near-miss: outside — after a real weighing, light correctly in BLUE (latches)
                           but the heavy cube parked on the floor just OUTSIDE the
                           red bin's wall -> not in_bin, no success, score 0.60;
   9. settle gate        — after a real weighing, the exact success placement but
                           the heavy cube still sliding at ~0.35 m/s inside the red
                           bin: success refuses while anything moves (probe
                           dismantled before it can settle into a real success);
  10. rejection audit    — success() was never True at any step of this battery;
  11. score-cap audit    — score never exceeded 0.70 anywhere in this battery;
  12. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.push_cube_i344.smoke --headless
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

_qapply = task_scene._qapply

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}
_AUD = {"on": False, "hits": 0, "smax": 0.0}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        if _AUD["on"]:
            if bool(env.scene.success()[0]):
                _AUD["hits"] += 1
            _AUD["smax"] = max(_AUD["smax"], float(env.scene.score()[0]))
        if rec and _REC["i"] % args.record_every == 0:
            for _f in range(3):  # flush accumulated history (ghosting fix)
                env.sim.render()
            arr = np.asarray(_REC["annot"].get_data())
            if arr.size:
                _REC["frames"].append(arr[..., :3].astype(np.uint8).copy())
        _REC["i"] += 1


def _refresh() -> None:
    _ENV.iscene.update(0.0)


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    ang = math.degrees(float(scene.beam_angle()[0]))
    oph = scene.on_pan(scene.cube_h)[0]
    opl = scene.on_pan(scene.cube_l)[0]
    print(f"[smoke] {tag:18s} | beam={ang:+.1f}deg "
          f"h_pan=({bool(oph[0])},{bool(oph[1])}) "
          f"l_pan=({bool(opl[0])},{bool(opl[1])}) "
          f"weighed={bool(scene._weighed[0])} streak={int(scene._streak[0])} "
          f"h_red={bool(scene.in_bin(scene.cube_h, scene.bin_red)[0])} "
          f"l_blue={bool(scene.in_bin(scene.cube_l, scene.bin_blue)[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.balance_triage")().build(num_envs=args.num_envs,
                                                    device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    ids = torch.arange(n, device=device)

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.42, -0.85, 0.72)) + o),
                                tuple(np.array((0.35, 0.00, 0.10)) + o),
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

    def teleport(body, pos_w: torch.Tensor, quat_w: torch.Tensor,
                 lin_vel: torch.Tensor | None = None) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat_w
        if lin_vel is not None:
            st[:, 7:10] = lin_vel
        body.write_root_state_to_sim(st, ids)
        _refresh()

    def pan_hover(side: float, clearance: float) -> tuple[torch.Tensor, torch.Tensor]:
        _refresh()
        q = scene.beam.data.root_quat_w.clone()
        local = torch.tensor([side * c.arm_l, 0.0, c.cube / 2 + clearance],
                             device=device).expand(n, 3)
        return scene.beam.data.root_pos_w + _qapply(q, local), q

    def bin_pt(bin_body, loc_xyz) -> tuple[torch.Tensor, torch.Tensor]:
        _refresh()
        q = bin_body.data.root_quat_w.clone()
        local = torch.tensor(list(loc_xyz), device=device).expand(n, 3)
        return bin_body.data.root_pos_w + _qapply(q, local), q

    def drop_on_pan(body, side: float, name: str) -> bool:
        pan_idx = 0 if side > 0 else 1
        for attempt in range(4):
            pos, q = pan_hover(side, 0.012 + 0.004 * attempt)
            teleport(body, pos, q)
            _step(110)
            if bool(scene.on_pan(body)[0, pan_idx]):
                return True
            print(f"[smoke] {name} missed pan (attempt {attempt}); retrying", flush=True)
        return False

    def drop_in_bin(body, bin_body, name: str) -> bool:
        for attempt in range(4):
            pos, q = bin_pt(bin_body,
                            (0.0, 0.0, c.bin_floor_t + c.cube / 2 + 0.012 + 0.004 * attempt))
            teleport(body, pos, q)
            _step(130)
            if bool(scene.in_bin(body, bin_body)[0]):
                return True
            print(f"[smoke] {name} missed bin (attempt {attempt}); retrying", flush=True)
        return False

    def do_weigh(tag: str) -> bool:
        """A REAL weighing: heavy dropped on pan A, light on pan B, wait for the
        streak latch. Used by the checks that need a legitimately-armed state."""
        ok = drop_on_pan(scene.cube_h, +1.0, f"{tag}: heavy")
        ok = ok and drop_on_pan(scene.cube_l, -1.0, f"{tag}: light")
        if not ok:
            return False
        for _ in range(900):
            _step(1)
            if bool(scene._weighed[0]):
                return True
        return False

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _AUD["on"] = True
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    ang0 = abs(math.degrees(float(scene.beam_angle()[0])))
    check("settle/no-NaN: layout settles finite; beam level, cubes at their floor "
          "spots (not on pans, not in bins), score 0, no success",
          bool(scene._finite()[0]) and ang0 < 3.0
          and not bool(scene.on_pan(scene.cube_h).any())
          and not bool(scene.on_pan(scene.cube_l).any())
          and not bool(scene.in_bin(scene.cube_h, scene.bin_red)[0])
          and not bool(scene.in_bin(scene.cube_l, scene.bin_blue)[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (3-seed max-pairwise readback) ===================
    def readback():
        _refresh()
        mh, ml = scene.masses()
        side_h = 1.0 if float(scene.cube_h.data.root_pos_w[0, 1]
                              - env.iscene.env_origins[0, 1]) > 0 else -1.0
        return (mh, ml, side_h,
                scene.bin_red.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.bin_red.data.root_quat_w[0]),
                yaw_of(scene.pedestal.data.root_quat_w[0]),
                scene.cube_h.data.root_pos_w[0, :2].clone())

    obs = []
    for s in (101, 202, 303):
        torch.manual_seed(s)
        env.reset()
        _step(10)
        obs.append(readback())
    d_mh = max(abs(a[0] - b[0]) for a in obs for b in obs)
    d_ml = max(abs(a[1] - b[1]) for a in obs for b in obs)
    flip_diff = len({a[2] for a in obs}) > 1
    d_bin = max(float((a[3] - b[3]).norm()) for a in obs for b in obs)
    d_by = max(dyaw(a[4], b[4]) for a in obs for b in obs)
    d_py = max(dyaw(a[5], b[5]) for a in obs for b in obs)
    d_sp = max(float((a[6] - b[6]).norm()) for a in obs for b in obs)
    print(f"[smoke] randomization max-pairwise deltas: mh={d_mh * 1000:.0f}g "
          f"ml={d_ml * 1000:.0f}g flip_differs={flip_diff} bin_xy={d_bin * 1000:.1f}mm "
          f"bin_yaw={d_by:.1f}deg ped_yaw={d_py:.1f}deg spot_xy={d_sp * 1000:.1f}mm",
          flush=True)
    check("randomization-is-real: hidden masses, heavy-spawn coin flip, bin xy+yaw, "
          "pedestal yaw and spawn spots differ across seeds (physx readback)",
          d_mh > 0.005 and d_ml > 0.005 and flip_diff and d_bin > 0.004
          and d_by > 2.0 and d_py > 1.0 and d_sp > 0.003)

    # ================= 3. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 4. flagship: BLIND SORT (the seed's own strategy) ==========================
    # The seed solves by MOVING the object into a goal region. Transplanted here:
    # both cubes delivered to the CORRECT bins, settled — the live end state
    # satisfies success's entire placement clause, but no weighing ever happened.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    ok4 = drop_in_bin(scene.cube_h, scene.bin_red, "blind-sort heavy")
    ok4 = ok4 and drop_in_bin(scene.cube_l, scene.bin_blue, "blind-sort light")
    _step(120)
    _report("blind-sort")
    _REC["on"] = False
    check("BLIND SORT (seed strategy): both cubes genuinely settled in the CORRECT "
          "bins but never weighed — placement clause fully satisfied live, yet no "
          "success and score ~0 (bin credit is gated on the weighing)",
          ok4 and bool(scene.in_bin(scene.cube_h, scene.bin_red)[0])
          and bool(scene.in_bin(scene.cube_l, scene.bin_blue)[0])
          and bool(scene.cubes_settled()[0]) and not bool(scene._weighed[0])
          and not bool(scene._hbin[0]) and not bool(scene._lbin[0])
          and float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. single-cube tilt is not a weighing ======================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    ok5 = drop_on_pan(scene.cube_h, +1.0, "single-tilt heavy")
    _step(240)  # ~10x the streak length, calm at the stop
    _report("single-tilt")
    _REC["on"] = False
    ang5 = math.degrees(float(scene.beam_angle()[0]))
    check("single-cube tilt: the heavy cube alone tips the beam to the stop and "
          "holds it calmly far longer than the streak — but a one-pan tilt is not "
          "a weighing: `weighed` stays False, score is the pan visit only",
          ok5 and ang5 >= c.theta_min_deg and not bool(scene._weighed[0])
          and int(scene._streak[0]) == 0
          and float(scene.score()[0]) <= c.w_pan + 0.01
          and not bool(scene.success()[0]))

    # ================= 6. both cubes piled on ONE pan =============================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    ok6 = drop_on_pan(scene.cube_h, +1.0, "pile heavy")
    # drop the light cube right onto the heavy cube's back (same pan; the pan
    # cannot seat two side by side — asserted in cfg)
    _refresh()
    top = scene.cube_h.data.root_pos_w.clone()
    top[:, 2] += c.cube + 0.012
    teleport(scene.cube_l, top, scene.cube_h.data.root_quat_w.clone())
    _step(240)
    _report("one-pan-pile")
    ang6 = math.degrees(float(scene.beam_angle()[0]))
    check("one-pan pile: both cubes piled on the SAME pan slam the beam to the stop "
          "— never opposite pans, so `weighed` stays False, no success",
          ok6 and ang6 >= c.theta_min_deg and not bool(scene._weighed[0])
          and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 2 * c.w_pan + 0.01)

    # ================= 7. wrong bins after a REAL weighing ========================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    ok7 = do_weigh("wrong-bins")
    s7_weigh = float(scene.score()[0])
    ok7 = ok7 and drop_in_bin(scene.cube_h, scene.bin_blue, "wrong-bins heavy->blue")
    ok7 = ok7 and drop_in_bin(scene.cube_l, scene.bin_red, "wrong-bins light->red")
    _step(120)
    _report("wrong-bins")
    _REC["on"] = False
    check("wrong bins: after a genuine weighing (latch fired, 0.45) the cubes are "
          "delivered SWAPPED and settle there — no bin latch, no success, score "
          "stays at the weigh credit",
          ok7 and abs(s7_weigh - 0.45) <= 1e-3 and bool(scene._weighed[0])
          and bool(scene.in_bin(scene.cube_h, scene.bin_blue)[0])
          and bool(scene.in_bin(scene.cube_l, scene.bin_red)[0])
          and not bool(scene._hbin[0]) and not bool(scene._lbin[0])
          and abs(float(scene.score()[0]) - 0.45) <= 1e-3
          and not bool(scene.success()[0]))

    # ================= 8. near-miss: heavy parked just OUTSIDE the red bin ========================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    ok8 = do_weigh("near-miss")
    ok8 = ok8 and drop_in_bin(scene.cube_l, scene.bin_blue, "near-miss light")
    # park the heavy cube on the floor just outside the red bin's +x wall
    bin_half = c.bin_inner / 2 + c.bin_wall_t
    pos8, q8 = bin_pt(scene.bin_red,
                      (bin_half + c.cube / 2 + 0.012, 0.0, c.cube / 2 + 0.010))
    teleport(scene.cube_h, pos8, q8)
    _step(150)
    _report("near-miss")
    check("near-miss (outside the wall): weighed and light correctly in BLUE "
          "(latches, 0.60) but the heavy cube rests on the floor just OUTSIDE the "
          "red bin — not in_bin, no success",
          ok8 and bool(scene._weighed[0]) and bool(scene._lbin[0])
          and not bool(scene.in_bin(scene.cube_h, scene.bin_red)[0])
          and not bool(scene._hbin[0])
          and abs(float(scene.score()[0]) - 0.60) <= 1e-3
          and not bool(scene.success()[0]))

    # ================= 9. settle gate =============================================================
    # The exact success placement — weighed, light in blue, heavy inside the red
    # bin — but the heavy cube still sliding at ~0.35 m/s. success() must refuse
    # while anything moves. Dismantled (transport) before friction could stop it
    # into a real success.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    ok9 = do_weigh("settle-gate")
    ok9 = ok9 and drop_in_bin(scene.cube_l, scene.bin_blue, "settle-gate light")
    pos9, q9 = bin_pt(scene.bin_red,
                      (-0.015, 0.0, c.bin_floor_t + c.cube / 2 + 0.002))
    u9 = _qapply(q9, torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(n, 3))
    vel9 = 0.35 * u9
    teleport(scene.cube_h, pos9, q9, lin_vel=vel9)
    moving_ok = bool(scene.in_bin(scene.cube_h, scene.bin_red)[0])
    for _ in range(3):
        _step(1)
        v = float(scene.cube_h.data.root_lin_vel_w[0].norm())
        moving_ok = moving_ok and v > 0.1 and not bool(scene.cubes_settled()[0]) \
            and not bool(scene.success()[0])
    _report("settle-gate")
    # dismantle before it can settle into a real success
    far9 = scene.bin_red.data.root_pos_w.clone()
    far9[:, 0] -= 0.30
    far9[:, 1] += 0.15
    far9[:, 2] = c.cube / 2 + 0.01
    teleport(scene.cube_h, far9, q9)
    _step(60)
    check("settle gate: the exact success placement with the heavy cube still "
          "sliding at ~0.35 m/s in the red bin is refused while anything moves",
          ok9 and moving_ok and not bool(scene.success()[0]))

    # ================= 10+11. audits ==============================================================
    check("rejection audit: success() was never True at any step of this battery",
          _AUD["hits"] == 0)
    print(f"[smoke] max score observed anywhere in the battery: {_AUD['smax']:.4f}",
          flush=True)
    check("score-cap audit: score never exceeded 0.70 anywhere in this battery",
          _AUD["smax"] <= 0.70 + 1e-4)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.balance_triage")
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

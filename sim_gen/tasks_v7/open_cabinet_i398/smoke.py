"""Smoke / rubric-REJECTION battery for NookCabinetScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the latched
credit is monotone along a real trajectory). This battery proves the rubric REJECTS
wrong outcomes, and that the physical claims the task rests on — the nook blocks the
door, the shut door seals the cube in, only repositioning the cabinet frees both —
are load-bearing physics, not fiat. Probes are CONSTRUCTED settled states (teleport,
real physics steps, judge) or bounded principled wrenches through the scene's
velocity-gated slots — instrumentation, never a solution.

Checks:
   1. settle/no-NaN   — seeded reset settles finite; cabinet NOOKED by readback
                        (clearance ~gap-level), door shut, cube inside, upright;
                        score 0, no success;
   2. randomization   — 3 seeded resets, max-pairwise READBACK deltas: cabinet slot
                        (y), nose gap (x), cube chamber pose, mat xy all vary; both
                        mat SIDES appear over 8 resets;
   3. null-policy     — 240 idle steps: score ~0, no success;
   4. NOOK BLOCKS DOOR— the solve's own door torque (0.08 N m), then a 3x escalation
                        (0.24 N m), pressed for seconds while nooked: the door MOVES
                        (>0.5 deg, hinge not fused) but the wall stops it far below
                        open_min (<15 deg peak); the cabinet is not shoved out
                        (pry force << friction breakaway); no open credit;
   5. SHUT DOOR SEALS — the solve's own cube push force (0.6 N, mouth-ward) pressed
                        for 2 s while nooked: the cube presses the door, the door
                        presses the wall, the chain holds — the cube stays inside,
                        zero out credit;
   6. FREED DOOR OPENS— the SAME 0.08 N m torque on the cabinet teleported out of
                        the nook swings the door past open_min — the mechanism
                        differential: the blocker is the spatial relation, not the
                        hinge. Cube still inside -> no success (seed-analog "operate
                        the articulation and stop" is worth <= 0.55);
   7. delivery bypass — cube dropped straight onto the mat, cabinet untouched in the
                        nook, door shut: cube_on_mat holds but success refuses
                        (door + clearance clauses), score <= 0.30;
   8. near-miss door  — full otherwise-perfect end state with the door parked at
                        35 deg (< open_min 45): success refuses, no open credit,
                        score <= 0.56;
   9. near-miss mat   — full otherwise-perfect end state with the cube settled on
                        the floor ~12 cm from the mat centre: success refuses,
                        score <= 0.76;
  10. toppled cabinet — cabinet lying on its back (mouth up, door hanging open past
                        open_min), cube ON the mat: the upright clause alone
                        refuses success, score <= 0.76;
  11. exactness +     — the exact goal state constructed: success True, score 1.0,
      revocation        stable 1 s hands-off; removing the cube from the mat revokes
                        success while latched credit remains (score back to <= 0.76);
  12. frames.npz      — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.open_cabinet_i398.smoke --headless
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


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    print(f"[smoke] {tag:18s} | clear={float(scene.front_clearance()[0]):+.3f} "
          f"door={math.degrees(float(scene.door_angle()[0])):+.1f}deg "
          f"inside={bool(scene.cube_inside()[0])} on_mat={bool(scene.cube_on_mat()[0])} "
          f"upright={bool(scene.cab_upright()[0])} "
          f"latch(m/c/o/x)={int(scene._moved[0])}{int(scene._clear[0])}"
          f"{int(scene._open[0])}{int(scene._out[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.nook_cabinet")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.55, -1.05, 0.85)) + o),
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

    def score0() -> float:
        _refresh()
        return float(scene.score()[0])

    def place_linkage(cab_x: float, cab_y: float, door_deg: float,
                      quat_c: torch.Tensor | None = None, z: float | None = None) -> None:
        """CONSTRUCT: write the cabinet+door LINKAGE consistently (env-local pose).
        Default: upright at rest height via the scene helper."""
        if quat_c is None:
            xy = torch.tensor([[cab_x, cab_y]], device=device).expand(n, 2)
            st_c, st_d = scene.cab_door_states(
                xy, torch.zeros(n, device=device),
                torch.full((n,), door_deg, device=device))
        else:
            st_c = torch.zeros(n, 13, device=device)
            st_c[:, 0], st_c[:, 1], st_c[:, 2] = cab_x, cab_y, z
            st_c[:, 3:7] = quat_c
            st_d = torch.zeros(n, 13, device=device)
            hinge = torch.tensor([c.hinge_x, c.hinge_y, 0.0], device=device).expand(n, 3)
            st_d[:, 0:3] = st_c[:, 0:3] + quat_apply(st_c[:, 3:7], hinge)
            st_d[:, 3:7] = _qmul(st_c[:, 3:7],
                                 _qz(torch.full((n,), math.radians(door_deg), device=device)))
        st_c[:, 0:3] += scene.env_origins
        st_d[:, 0:3] += scene.env_origins
        scene.cabinet.write_root_state_to_sim(st_c, _all_ids())
        scene.door.write_root_state_to_sim(st_d, _all_ids())
        _refresh()

    def put_cube(x: float, y: float, z: float) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        scene.cube.write_root_state_to_sim(st, _all_ids())
        _refresh()

    def drop_cube_on_mat() -> None:
        _refresh()
        mc = scene.mat_center[0]
        put_cube(float(mc[0]), float(mc[1]), c.mat_t + c.cube_s / 2 + 0.030)
        _step(150)

    def press_door(tau: float, steps: int) -> float:
        """Apply a door torque through the scene's omega-gated slot; return the PEAK
        angle reached during the press (measure at peak, not after back-drive)."""
        peak = -1e9
        scene.door_tau[:] = tau
        for _ in range(steps):
            _step(1)
            peak = max(peak, float(scene.door_angle()[0]))
        scene.door_tau[:] = 0.0
        _step(60)
        return math.degrees(peak)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    fin = all(bool(torch.isfinite(b.data.root_state_w).all())
              for b in (scene.cabinet, scene.door, scene.cube, scene.mat))
    d0 = abs(math.degrees(float(scene.door_angle()[0])))
    check("settle/no-NaN: reset settles finite; cabinet NOOKED (clearance < 0.09), "
          "door shut, cube inside, upright; score 0, no success",
          fin and float(scene.front_clearance()[0]) < 0.09 and d0 < 5.0
          and bool(scene.cube_inside()[0]) and bool(scene.cab_upright()[0])
          and score0() <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (3-seed max-pairwise readback) ===================
    def readback():
        _refresh()
        cab = (scene.cabinet.data.root_pos_w - scene.env_origins)[0, :2].clone()
        cube_l = scene._cab_local(scene.cube.data.root_pos_w)[0, :2].clone()
        mat = scene.mat_center[0].clone()
        return cab, cube_l, mat

    obs = []
    for s in (101, 202, 303):
        torch.manual_seed(s)
        env.reset()
        _step(10)
        obs.append(readback())

    def maxpair(idx: int, dim: int) -> float:
        vals = [float(ob[idx][dim]) for ob in obs]
        return max(abs(a - b) for a in vals for b in vals)

    d_gap, d_slot = maxpair(0, 0), maxpair(0, 1)
    d_cx, d_cy = maxpair(1, 0), maxpair(1, 1)
    d_mx, d_my = maxpair(2, 0), maxpair(2, 1)
    sides = set()
    for s in range(8):
        torch.manual_seed(400 + s)
        env.reset()
        _refresh()
        sides.add(1 if float(scene.mat_center[0, 1]) > 0 else -1)
    print(f"[smoke] randomization max-pairwise: gap(x)={d_gap * 1000:.1f}mm "
          f"slot(y)={d_slot * 1000:.1f}mm cube=({d_cx * 1000:.1f},{d_cy * 1000:.1f})mm "
          f"mat=({d_mx * 1000:.1f},{d_my * 1000:.1f})mm sides={sorted(sides)}", flush=True)
    check("randomization-is-real: nose gap, wall slot, cube chamber pose, mat xy all "
          "vary by readback; both mat sides appear over 8 resets",
          d_gap > 0.0015 and d_slot > 0.01 and (d_cx > 0.005 or d_cy > 0.005)
          and d_mx > 0.01 and len(sides) == 2)

    # ================= 3. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          score0() <= 0.05 and not bool(scene.success()[0]))

    # ================= 4. THE NOOK BLOCKS THE DOOR (identity claim is physics) ====================
    # The solve's own opening torque, then a 3x escalation, pressed while nooked:
    # the knob strikes the wall. Door must MOVE (hinge not fused) yet stay far below
    # open_min; the cabinet must not be shoved out (pry force ~tau/r << mu*m*g).
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _refresh()
    cab0 = (scene.cabinet.data.root_pos_w - scene.env_origins)[0, :2].clone()
    _REC["on"] = True
    peak1 = press_door(0.08, 240)
    peak2 = press_door(0.24, 120)
    _REC["on"] = False
    _refresh()
    cab_disp = float(((scene.cabinet.data.root_pos_w - scene.env_origins)[0, :2]
                      - cab0).norm())
    _report("nook-blocks")
    print(f"[smoke] nooked door press: peak@0.08Nm={peak1:.1f}deg "
          f"peak@0.24Nm={peak2:.1f}deg cab_disp={cab_disp * 1000:.1f}mm", flush=True)
    check("NOOK BLOCKS DOOR: pressed at the solve's torque and 3x that, the door "
          "moves (>0.5deg, not fused) but the wall stops it <15deg; cabinet not "
          "shoved out; no open credit",
          0.5 < peak1 and max(peak1, peak2) < 15.0 and cab_disp < 0.03
          and float(scene.front_clearance()[0]) < c.clear_min
          and not bool(scene._open[0]) and score0() <= 0.05
          and not bool(scene.success()[0]))

    # ================= 5. THE SHUT DOOR SEALS THE CUBE IN =========================================
    # The solve's own extraction push (0.6 N mouth-ward) pressed for 2 s while
    # nooked: cube -> door -> wall chain must hold; the cube stays inside.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    fwd = quat_apply(scene.cabinet.data.root_quat_w,
                     torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))
    scene.cube_f[:] = 0.6 * fwd
    scene.cube_f[:, 2] = 0.0
    _step(240)
    scene.cube_f[:] = 0.0
    _step(60)
    _REC["on"] = False
    _report("sealed")
    check("SHUT DOOR SEALS: the solve's own push force cannot get the cube out "
          "while the cabinet is nooked — cube still inside, zero out credit",
          bool(scene.cube_inside()[0]) and not bool(scene._out[0])
          and score0() <= 0.05 and not bool(scene.success()[0]))

    # ================= 6. FREED, THE SAME TORQUE OPENS THE DOOR ===================================
    # Teleport the linkage out of the nook (construction), press the SAME 0.08 N m:
    # the door passes open_min — the blocker was the spatial relation, not the hinge.
    # Cube still inside -> the seed-analog "operate the articulation and stop" end
    # state is refused (score <= 0.55, no success).
    torch.manual_seed(100)
    env.reset()
    _step(60)
    place_linkage(0.26, float(scene.spawn_xy[0, 1]), 0.0)
    # The linkage teleport moves cabinet+door only — re-seat the cube INSIDE the
    # relocated interior (otherwise it is left on the floor at the old nook spot
    # and the out latch fires, poisoning the score budget).
    put_cube(0.26 - 0.02, float(scene.spawn_xy[0, 1]),
             c.panel_t + c.cube_s / 2 + 0.005)
    _step(60)
    _REC["on"] = True
    peak3 = press_door(0.08, 360)
    _REC["on"] = False
    _report("freed-door")
    print(f"[smoke] freed door press: peak@0.08Nm={peak3:.1f}deg "
          f"(same torque, nook removed)", flush=True)
    check("FREED DOOR OPENS: the same 0.08 N m passes open_min once the cabinet is "
          "out — and door-open-with-cube-inside is NOT success (score <= 0.55)",
          peak3 > c.open_min_deg + 5.0 and bool(scene.cube_inside()[0])
          and not bool(scene.success()[0]) and score0() <= 0.55)

    # ================= 7. delivery bypass: cube on mat, task never done ===========================
    # Drop the cube straight onto the mat with the cabinet untouched in the nook and
    # the door shut. cube_on_mat holds — success must refuse on the door clause.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    drop_cube_on_mat()
    _step(60)
    _REC["on"] = False
    _report("bypass")
    check("delivery bypass: cube ON the mat but cabinet nooked and door shut — "
          "cube_on_mat holds yet success refuses; score <= 0.30",
          bool(scene.cube_on_mat()[0]) and not bool(scene.success()[0])
          and math.degrees(float(scene.door_angle()[0])) < c.open_min_deg
          and score0() <= 0.30)

    # ================= 8. near-miss: door parked below open_min ===================================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    place_linkage(0.26, float(scene.spawn_xy[0, 1]), 35.0)
    _step(60)
    drop_cube_on_mat()
    _step(60)
    _report("door-35deg")
    check("near-miss (door): full end state with the door at 35deg (<45) — success "
          "refuses, no open credit, score <= 0.56",
          bool(scene.cube_on_mat()[0])
          and not bool(scene._open[0]) and not bool(scene.success()[0])
          and score0() <= 0.56)

    # ================= 9. near-miss: cube on the floor beside the mat =============================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    place_linkage(0.26, float(scene.spawn_xy[0, 1]), 110.0)
    _step(60)
    _refresh()
    mc = scene.mat_center[0]
    off = 0.12 if float(mc[1]) < 0 else -0.12
    put_cube(float(mc[0]) + 0.06, float(mc[1]) + off, c.cube_s / 2 + 0.01)
    _step(150)
    _report("off-mat")
    check("near-miss (mat): cube settled on the floor ~12 cm from the mat centre, "
          "everything else perfect — success refuses, score <= 0.76",
          not bool(scene.cube_on_mat()[0]) and not bool(scene.success()[0])
          and score0() <= 0.76)

    # ================= 10. toppled cabinet ========================================================
    # Cabinet on its BACK (mouth up; pitch -90 about y), door hanging open past
    # open_min under gravity, cube ON the mat: the upright clause alone refuses.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    qy = torch.zeros(n, 4, device=device)
    half = math.radians(-90.0) / 2
    qy[:, 0], qy[:, 2] = math.cos(half), math.sin(half)
    place_linkage(0.30, float(scene.spawn_xy[0, 1]), 110.0, quat_c=qy,
                  z=(c.inner_d / 2 + c.panel_t) + 0.015)
    _step(180)
    drop_cube_on_mat()
    _step(60)
    _report("toppled")
    check("toppled cabinet: mouth-up on its back, door hanging open, cube ON the "
          "mat — the upright clause alone refuses success; score <= 0.76",
          not bool(scene.cab_upright()[0]) and bool(scene.cube_on_mat()[0])
          and math.degrees(float(scene.door_angle()[0])) > c.open_min_deg
          and not bool(scene.success()[0]) and score0() <= 0.76)

    # ================= 11. exactness + revocation =================================================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    place_linkage(0.26, float(scene.spawn_xy[0, 1]), 110.0)
    _step(60)
    _REC["on"] = True
    drop_cube_on_mat()
    _step(60)
    _report("exact-goal")
    ok_exact = bool(scene.success()[0]) and score0() >= 0.999
    _step(120)  # stable 1 s hands-off
    ok_hold = bool(scene.success()[0]) and score0() >= 0.999
    # revoke: take the cube off the mat (park it on the open floor, settled)
    _refresh()
    mc = scene.mat_center[0]
    put_cube(float(mc[0]) - 0.05, -float(mc[1] / mc[1].abs()) * 0.15, c.cube_s / 2 + 0.01)
    _step(120)
    _REC["on"] = False
    _report("revoked")
    s_rev = score0()
    check("exactness+revocation: the exact goal state scores success and 1.0, holds "
          "1 s; removing the cube revokes success while latched credit remains",
          ok_exact and ok_hold and not bool(scene.success()[0])
          and 0.70 <= s_rev <= 0.76)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.nook_cabinet")
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
    except BaseException as e:  # noqa: BLE001 — die loudly, never idle to the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL (exception: {e})", flush=True)
        os._exit(1)
